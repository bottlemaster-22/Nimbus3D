//
//  BoosterDevice.swift
//  Booster
//
//  A Booster as it shows up in the "Nearby Boosters" list: something found on
//  the LAN via Bonjour, which may or may not already be paired with this
//  phone (paired-but-not-currently-discoverable devices, e.g. the PC is
//  asleep, also appear here so "Send scan" can show a clear "not reachable"
//  state instead of the device just vanishing).
//

import Foundation
import Network

public struct BoosterDevice: Identifiable, Hashable, Sendable {
    /// Stable identity: the Bonjour instance name until the Booster answers
    /// /v1/info with its real boosterID, then the boosterID takes over so a
    /// paired device keeps the same identity across app launches even if its
    /// Bonjour name or IP changes.
    public var id: String
    public var name: String
    public var isPaired: Bool
    public var isReachable: Bool

    /// nil once the device has gone away (was paired, not currently seen on
    /// the LAN).
    public var endpoint: NWEndpoint?

    public init(
        id: String,
        name: String,
        isPaired: Bool,
        isReachable: Bool,
        endpoint: NWEndpoint?
    ) {
        self.id = id
        self.name = name
        self.isPaired = isPaired
        self.isReachable = isReachable
        self.endpoint = endpoint
    }

    public static func == (lhs: BoosterDevice, rhs: BoosterDevice) -> Bool {
        lhs.id == rhs.id
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
