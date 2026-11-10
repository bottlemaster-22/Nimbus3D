//
//  OnboardingFormat.swift
//  Onboarding
//
//  Turning measurements into words a person reads without flinching.
//
//  Every number the onboarding screens show passes through here, for one
//  reason: a raw measurement is honest but unreadable ("5996904448 bytes"),
//  and a rounded one is readable but easy to round dishonestly. These
//  functions round in a fixed, documented direction and never invent
//  precision the measurement did not have.
//
//  Nothing here decides anything. It only formats.
//

import Foundation

/// Human-readable forms of the raw device measurements.
public enum OnboardingFormat {

    // MARK: - Bytes

    /// A file-size-style byte count, e.g. `4.29 GB`, `812 MB`.
    ///
    /// `ByteCountFormatter` with `.file` style, which is what iOS itself uses
    /// in Settings for storage, so the number here matches the number the user
    /// can go and check. Powers of 1000, deliberately: that is what Settings
    /// shows, and disagreeing with Settings makes a person doubt the app
    /// rather than doubt Apple.
    ///
    /// A negative count is a bug somewhere upstream, so it is reported as
    /// `unknown` rather than printed as a negative size.
    public static func bytes(_ count: Int64) -> String {
        guard count >= 0 else { return "an unknown amount" }
        guard count > 0 else { return "0 bytes" }
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB, .useTB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter.string(fromByteCount: count)
    }

    /// The unsigned form, for `ProcessInfo.physicalMemory` and
    /// `os_proc_available_memory()`, which are both `UInt64`.
    ///
    /// `Int64(clamping:)` rather than `Int64(_:)` so an absurd value from a
    /// future API cannot trap in a formatting function.
    public static func bytes(_ count: UInt64) -> String {
        bytes(Int64(clamping: count))
    }

    /// RAM, the way a person says it: `6 GB`, `8 GB`.
    ///
    /// `physicalMemory` reports slightly under the marketing figure (an
    /// "8 GB" iPhone reports something like 7.79 GB once the display and the
    /// baseband have taken their reservations), so printing it exactly makes
    /// the user think we misread their phone. This rounds to the nearest whole
    /// gibibyte, which lands on the number written on the box.
    ///
    /// Used only for total installed memory. **Never** use it for
    /// `availableBytes`: that number moves, and rounding a moving number to
    /// whole gigabytes throws away the entire signal. Available memory goes
    /// through `bytes(_:)`.
    public static func memory(_ count: UInt64) -> String {
        let gibibyte = 1_073_741_824.0
        let gigabytes = (Double(count) / gibibyte).rounded()
        guard gigabytes >= 1 else { return bytes(count) }
        return "\(Int(gigabytes)) GB"
    }

    // MARK: - Classes and states

    /// The contract's 0...3 `sustainedPerformanceClass` as a phrase.
    ///
    /// This is a heuristic, not a benchmark (see `SustainedPerformanceEstimate`),
    /// so the wording is deliberately soft. It never appears alone: the screen
    /// that shows it also shows the chip and the thermal state it was derived
    /// from.
    public static func performanceClass(_ value: Int) -> String {
        switch value {
        case 3: return "Excellent"
        case 2: return "Good"
        case 1: return "Modest"
        default: return "Not determined"
        }
    }

    /// How warm the phone is right now, in words.
    public static func thermal(_ level: ThermalLevel) -> String {
        switch level {
        case .nominal: return "Cool"
        case .fair: return "Slightly warm"
        case .serious: return "Warm"
        case .critical: return "Very hot"
        }
    }

    /// `true` / `false` as something readable in a details list.
    public static func yesNo(_ value: Bool) -> String {
        value ? "Yes" : "No"
    }

    // MARK: - Dates

    /// When the check ran, e.g. `4 September 2026 at 14:32`.
    public static func timestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .long
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    // MARK: - Lists

    /// Joins clauses the way a person writes them: `a`, `a and b`,
    /// `a, b and c`. Used by the copy to build one sentence out of however
    /// many reasons a device actually has, instead of a bulleted list of
    /// fragments.
    public static func sentenceList(_ parts: [String]) -> String {
        switch parts.count {
        case 0: return ""
        case 1: return parts[0]
        case 2: return "\(parts[0]) and \(parts[1])"
        default:
            let head = parts.dropLast().joined(separator: ", ")
            return "\(head) and \(parts[parts.count - 1])"
        }
    }
}
