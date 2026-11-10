//
//  BoosterKeychain.swift
//  Booster
//
//  Minimal Keychain wrapper for the one secret this module owns: the opaque
//  pairing token for each paired Booster, keyed by boosterID. There is no
//  shared Keychain helper in Core (Contracts.swift is geometry/pipeline
//  vocabulary, not a security utility), so this stays self-contained.
//
//  Service string comes from BrandConfig.keychainService so a rename never
//  orphans a stored token, and so this module never hardcodes the app name.
//

import Foundation
import Security

enum BoosterKeychain {

    /// Saves (or overwrites) the pairing token for one Booster.
    static func saveToken(_ token: String, boosterID: String) throws {
        let data = Data(token.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: BrandConfig.keychainService,
            kSecAttrAccount as String: boosterID,
        ]

        let attributesToUpdate: [String: Any] = [
            kSecValueData as String: data
        ]
        let updateStatus = SecItemUpdate(
            query as CFDictionary,
            attributesToUpdate as CFDictionary
        )

        if updateStatus == errSecItemNotFound {
            var addQuery = query
            addQuery[kSecValueData as String] = data
            addQuery[kSecAttrAccessible as String] =
                kSecAttrAccessibleAfterFirstUnlock
            let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw BoosterError.connectionFailed(
                    "Could not save this Booster's pairing on your phone "
                        + "(Keychain error \(addStatus))."
                )
            }
        } else if updateStatus != errSecSuccess {
            throw BoosterError.connectionFailed(
                "Could not update this Booster's pairing on your phone "
                    + "(Keychain error \(updateStatus))."
            )
        }
    }

    /// Returns nil if no token is stored for this Booster (never paired, or
    /// the pairing was forgotten).
    static func loadToken(boosterID: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: BrandConfig.keychainService,
            kSecAttrAccount as String: boosterID,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    static func deleteToken(boosterID: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: BrandConfig.keychainService,
            kSecAttrAccount as String: boosterID,
        ]
        SecItemDelete(query as CFDictionary)
    }

    /// Every boosterID this phone currently holds a saved token for. Used to
    /// populate "Paired Boosters" even before one is discovered on the LAN
    /// again.
    static func allPairedBoosterIDs() -> [String] {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: BrandConfig.keychainService,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String: kSecMatchLimitAll,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess,
            let items = result as? [[String: Any]]
        else { return [] }
        return items.compactMap { $0[kSecAttrAccount as String] as? String }
    }
}
