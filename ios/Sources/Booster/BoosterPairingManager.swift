//
//  BoosterPairingManager.swift
//  Booster
//
//  Pairing flow: the phone asks a Booster to start a pairing request, the PC
//  Booster GUI shows a short confirmation code, and the user types that code
//  into this phone. This is deliberately a two-device, human-witnessed
//  handshake (no cloud account, no QR needed, works even the very first time
//  two devices meet on a LAN) rather than silent auto-trust of anything that
//  answers on port 8760.
//
//  On success the returned token is stored in the Keychain (BoosterKeychain)
//  and the boosterID <-> Bonjour-name mapping is recorded in BoosterJobStore
//  so future discovery immediately recognises this device as paired.
//

import Foundation
import UIKit

@MainActor
public final class BoosterPairingManager: ObservableObject {

    public enum State: Equatable {
        case idle
        case requesting
        case awaitingCode(requestID: String)
        case confirming
        case paired(boosterID: String, boosterName: String)
        case failed(String)
    }

    @Published public private(set) var state: State = .idle

    private var address: BoosterEndpointAddress?
    private var bonjourName: String = ""

    public init() {}

    /// Step 1: reach the Booster and ask it to begin pairing. The Booster's
    /// own GUI is expected to show a code at this point.
    public func beginPairing(with device: BoosterDevice, host: String, port: UInt16) async {
        state = .requesting
        bonjourName = device.name
        let address = BoosterEndpointAddress(host: host, port: port)
        self.address = address

        do {
            let body = BoosterPairRequestBody(
                deviceID: BoosterDeviceIdentity.current,
                deviceName: UIDevice.current.name,
                appVersion: BrandConfig.versionString
            )
            let response: BoosterPairRequestResponse = try await BoosterHTTP.post(
                address,
                path: BoosterAPI.pairRequest(),
                body: body
            )
            state = .awaitingCode(requestID: response.requestID)
        } catch let error as BoosterError {
            state = .failed(error.errorDescription ?? "Pairing failed.")
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    /// Step 2: the user has read a code off the Booster's window and typed
    /// it into this app.
    public func confirm(code: String) async {
        guard case .awaitingCode(let requestID) = state, let address else { return }
        state = .confirming

        do {
            let body = BoosterPairConfirmBody(code: code)
            let response: BoosterPairStatusResponse = try await BoosterHTTP.post(
                address,
                path: BoosterAPI.pairConfirm(requestID: requestID),
                body: body
            )
            try await handle(response, requestID: requestID, address: address)
        } catch let error as BoosterError {
            state = .failed(error.errorDescription ?? "Pairing failed.")
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    /// Polls status, used only if the confirm call itself returns "pending"
    /// (e.g. the server models confirmation as async on its side).
    public func pollStatus() async {
        guard case .awaitingCode(let requestID) = state, let address else { return }
        do {
            let response: BoosterPairStatusResponse = try await BoosterHTTP.get(
                address,
                path: BoosterAPI.pairStatus(requestID: requestID)
            )
            try await handle(response, requestID: requestID, address: address)
        } catch {
            // Silent: this is a background poll, the user is looking at the
            // code-entry screen, not this call.
        }
    }

    public func reset() {
        state = .idle
        address = nil
    }

    // MARK: - Private

    private func handle(
        _ response: BoosterPairStatusResponse,
        requestID: String,
        address: BoosterEndpointAddress
    ) async throws {
        switch response.status {
        case .approved:
            guard let token = response.token, let boosterID = response.boosterID else {
                state = .failed("The Booster approved pairing but did not send a token.")
                return
            }
            try BoosterKeychain.saveToken(token, boosterID: boosterID)
            BoosterJobStore.shared.recordPairing(
                boosterID: boosterID,
                boosterName: response.boosterName ?? bonjourName,
                bonjourName: bonjourName
            )
            state = .paired(boosterID: boosterID, boosterName: response.boosterName ?? bonjourName)
        case .denied:
            state = .failed(BoosterError.pairingDenied.errorDescription ?? "")
        case .expired:
            state = .failed(BoosterError.pairingExpired.errorDescription ?? "")
        case .pending:
            state = .awaitingCode(requestID: requestID)
        }
    }
}

/// A stable per-install identifier, independent of device name (which the
/// user can change in Settings at any time). Cached in UserDefaults under the
/// module's own prefix so it survives app relaunches without touching Core.
enum BoosterDeviceIdentity {
    static var current: String {
        let key = BrandConfig.defaultsPrefix + "booster.deviceID"
        if let existing = UserDefaults.standard.string(forKey: key) {
            return existing
        }
        let fresh = UUID().uuidString
        UserDefaults.standard.set(fresh, forKey: key)
        return fresh
    }
}
