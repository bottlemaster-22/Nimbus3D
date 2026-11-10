//
//  BoosterError.swift
//  Booster
//
//  Every error this module can raise, in plain language. The user is
//  non-technical: every message here is meant to be read out loud on the
//  Booster tab as-is, with no translation layer. Specific > generic; say what
//  happened and, where there is one, what to do about it.
//

import Foundation

public enum BoosterError: LocalizedError, Sendable {
    case discoveryUnavailable(String)
    case noDeviceSelected
    case connectionFailed(String)
    case notPaired
    case pairingDenied
    case pairingExpired
    case pairingCodeIncorrect
    case serverRejected(String)
    case checksumMismatch(fileName: String)
    case uploadIncomplete(fileName: String)
    case jobFailed(String)
    case jobCancelled
    case resultDownloadFailed(String)
    case decodingFailed(String)
    case scanFolderUnreadable(String)
    case apiVersionMismatch(boosterVersion: String)
    case timedOut

    public var errorDescription: String? {
        switch self {
        case .discoveryUnavailable(let reason):
            return "Could not search your Wi-Fi network for a Booster. \(reason)"
        case .noDeviceSelected:
            return "Choose a Booster from the list first."
        case .connectionFailed(let reason):
            return "Could not reach that computer. \(reason) Make sure your "
                + "phone and computer are on the same Wi-Fi network and the "
                + "Booster app is open."
        case .notPaired:
            return "This phone is not paired with that Booster yet. Pair "
                + "with it first."
        case .pairingDenied:
            return "Pairing was declined on the computer."
        case .pairingExpired:
            return "That pairing code expired. Try again."
        case .pairingCodeIncorrect:
            return "That code does not match what is showing on the "
                + "computer. Check the number and try again."
        case .serverRejected(let reason):
            return reason
        case .checksumMismatch(let fileName):
            return "\"\(fileName)\" did not arrive correctly and is being "
                + "sent again."
        case .uploadIncomplete(let fileName):
            return "Sending \"\(fileName)\" stopped partway through. It will "
                + "pick back up from where it left off."
        case .jobFailed(let reason):
            return "The Booster could not finish this scan. \(reason)"
        case .jobCancelled:
            return "Sending to the Booster was cancelled."
        case .resultDownloadFailed(let reason):
            return "The finished scan could not be brought back to your "
                + "phone. \(reason)"
        case .decodingFailed:
            return "The computer sent back something this app did not "
                + "understand. It may be running a different version."
        case .scanFolderUnreadable(let reason):
            return "This scan could not be read from your phone's storage. "
                + "\(reason)"
        case .apiVersionMismatch(let boosterVersion):
            return "This phone and that Booster are running different "
                + "versions (Booster is on \(boosterVersion)). Update both to "
                + "the same version and try again."
        case .timedOut:
            return "That took too long and was stopped. Check the Wi-Fi "
                + "connection and try again."
        }
    }
}
