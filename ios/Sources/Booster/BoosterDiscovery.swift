//
//  BoosterDiscovery.swift
//  Booster
//
//  Bonjour discovery of PC Boosters on the local network, via NWBrowser.
//  Publishes a merged list: devices currently visible on the LAN, plus any
//  previously-paired devices that are not visible right now (so the UI can
//  show "not reachable" instead of the row disappearing).
//
//  Service type comes from BrandConfig.boosterServiceType
//  (e.g. "_nimbusboost._tcp"), which is generated into Info.plist's
//  NSBonjourServices from the same project.yml brand block - so a product
//  rename never desyncs discovery from what the Booster actually advertises.
//

import Combine
import Foundation
import Network
import os

@MainActor
public final class BoosterDiscovery: ObservableObject {

    @Published public private(set) var devices: [BoosterDevice] = []
    @Published public private(set) var isSearching = false
    @Published public private(set) var lastError: BoosterError?

    private var browser: NWBrowser?
    /// Bonjour instance name -> raw NWBrowser result, so we can re-resolve on
    /// demand (host:port is not needed until the user actually taps a
    /// device).
    private var liveResults: [String: NWBrowser.Result] = [:]

    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "BoosterDiscovery"
    )

    public init() {}

    public func start() {
        guard browser == nil else { return }
        isSearching = true
        lastError = nil

        let descriptor = NWBrowser.Descriptor.bonjour(
            type: BrandConfig.boosterServiceType,
            domain: BrandConfig.boosterServiceDomain
        )
        let parameters = NWParameters()
        parameters.includePeerToPeer = false

        let browser = NWBrowser(for: descriptor, using: parameters)
        self.browser = browser

        browser.stateUpdateHandler = { [weak self] state in
            Task { @MainActor [weak self] in
                self?.handleState(state)
            }
        }
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            Task { @MainActor [weak self] in
                self?.handleResults(results)
            }
        }
        browser.start(queue: .main)
    }

    public func clearError() {
        lastError = nil
    }

    public func stop() {
        browser?.cancel()
        browser = nil
        liveResults.removeAll()
        isSearching = false
        markAllUnreachable()
    }

    /// Resolves a discovered (or re-discovered) device to a live host/port so
    /// the rest of the module can build HTTP/WebSocket URLs. Throws if the
    /// device is no longer visible on the LAN at all.
    public func resolveHostPort(for device: BoosterDevice) async throws -> (
        host: String, port: UInt16
    ) {
        guard let endpoint = device.endpoint ?? liveResults[device.id]?.endpoint
        else {
            throw BoosterError.connectionFailed(
                "That computer is not on your Wi-Fi network right now."
            )
        }
        return try await Self.resolve(endpoint: endpoint)
    }

    // MARK: - Private

    private func handleState(_ state: NWBrowser.State) {
        switch state {
        case .ready:
            isSearching = true
        case .failed(let error):
            isSearching = false
            lastError = .discoveryUnavailable(error.localizedDescription)
            log.error("NWBrowser failed: \(error.localizedDescription, privacy: .public)")
        case .cancelled:
            isSearching = false
        case .waiting(let error):
            log.notice("NWBrowser waiting: \(error.localizedDescription, privacy: .public)")
        default:
            break
        }
    }

    private func handleResults(_ results: Set<NWBrowser.Result>) {
        var seenNames = Set<String>()

        for result in results {
            guard case .service(let name, _, _, _) = result.endpoint else { continue }
            seenNames.insert(name)
            liveResults[name] = result
        }

        var merged: [BoosterDevice] = []

        // Devices visible right now.
        for name in seenNames {
            let alreadyPairedID = pairedRecord(forBonjourName: name)
            merged.append(
                BoosterDevice(
                    id: alreadyPairedID ?? name,
                    name: name,
                    isPaired: alreadyPairedID != nil,
                    isReachable: true,
                    endpoint: liveResults[name]?.endpoint
                )
            )
        }

        // Paired devices not currently on the LAN.
        let reachableIDs = Set(merged.map(\.id))
        for boosterID in BoosterKeychain.allPairedBoosterIDs()
        where !reachableIDs.contains(boosterID) {
            merged.append(
                BoosterDevice(
                    id: boosterID,
                    name: BoosterJobStore.shared.displayName(forBoosterID: boosterID)
                        ?? "A previously paired Booster",
                    isPaired: true,
                    isReachable: false,
                    endpoint: nil
                )
            )
        }

        devices = merged.sorted { lhs, rhs in
            if lhs.isReachable != rhs.isReachable { return lhs.isReachable }
            return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }
    }

    private func markAllUnreachable() {
        devices = devices.map {
            var device = $0
            device.isReachable = false
            return device
        }
    }

    /// Looks up whether a Bonjour instance name corresponds to an already
    /// paired boosterID, by matching against saved pairing records. Falls
    /// back to nil (unpaired) when there is no record for this name yet -
    /// the very first pairing has no boosterID to key on until the handshake
    /// completes.
    private func pairedRecord(forBonjourName name: String) -> String? {
        BoosterJobStore.shared.boosterID(forBonjourName: name)
    }

    // MARK: - Endpoint resolution (Bonjour service -> literal host:port)

    private static func resolve(endpoint: NWEndpoint) async throws -> (
        host: String, port: UInt16
    ) {
        try await withCheckedThrowingContinuation { continuation in
            let parameters = NWParameters.tcp
            parameters.prohibitExpensivePaths = false
            let connection = NWConnection(to: endpoint, using: parameters)
            var didResume = false

            let finish: (Result<(String, UInt16), Error>) -> Void = { result in
                guard !didResume else { return }
                didResume = true
                connection.cancel()
                switch result {
                case .success(let value):
                    continuation.resume(returning: value)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }

            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if case .hostPort(let host, let port) =
                        connection.currentPath?.remoteEndpoint ?? endpoint
                    {
                        finish(.success((hostString(host), port.rawValue)))
                    } else {
                        finish(
                            .failure(
                                BoosterError.connectionFailed(
                                    "Could not work out that computer's address."
                                )
                            )
                        )
                    }
                case .failed(let error):
                    finish(
                        .failure(BoosterError.connectionFailed(error.localizedDescription))
                    )
                default:
                    break
                }
            }
            connection.start(queue: .global(qos: .utility))
        }
    }

    /// `nonisolated` because it is called from `stateUpdateHandler`, which
    /// Network delivers on its own queue rather than the main actor. It reads
    /// nothing but its argument, so there is no state to protect.
    nonisolated private static func hostString(_ host: NWEndpoint.Host) -> String {
        switch host {
        case .ipv4(let address):
            return "\(address)"
        case .ipv6(let address):
            return "\(address)"
        case .name(let name, _):
            return name
        @unknown default:
            return "\(host)"
        }
    }
}
