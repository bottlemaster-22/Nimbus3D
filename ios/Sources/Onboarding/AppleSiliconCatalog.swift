//
//  AppleSiliconCatalog.swift
//  Onboarding
//
//  Hardware model identifier -> chip, marketing name, and an ADVISORY note on
//  whether Apple's own spec sheet says that model has a LiDAR scanner.
//
//  ---------------------------------------------------------------------------
//  READ THIS BEFORE TRUSTING ANYTHING IN THIS FILE
//  ---------------------------------------------------------------------------
//
//  A lookup table of device identifiers is, by construction, always one Apple
//  keynote out of date. So this file is deliberately built so that being out of
//  date can only ever cost us a NAME, never a VERDICT:
//
//    * The compatibility VERDICT (`DeviceTier`) is decided in
//      `DeviceCompatibilityProbe` from things that are actually MEASURED on the
//      running device: what ARKit says about scene depth, what Metal says about
//      the GPU family, what ProcessInfo says about memory and the OS version.
//      None of it is read out of this table.
//
//    * This table supplies the human-readable extras: "A19 Pro",
//      "iPhone 17 Pro Max". If an identifier is unknown, every one of those is
//      allowed to be nil and every screen in this module already handles nil.
//
//    * `catalogHasLiDAR` is ADVISORY ONLY and exists for exactly one purpose:
//      to let the incompatible screen say something specific and true, like
//      "on iPhone, only the Pro models have the scanner", instead of something
//      vague. Where the table and ARKit disagree, ARKit wins and the probe
//      switches to a different, honest explanation of the disagreement.
//
//  There is also a real regularity worth exploiting rather than re-typing every
//  year: for every iPhone from iPhone8,x (iPhone 6s, A9) through iPhone18,x
//  (iPhone 17, A19), the chip's A-number is the identifier's major number plus
//  one. See `inferredChip(forPhoneMajor:)`. That is an observed pattern, not a
//  documented API, so it is used only as a fallback and is labelled
//  `isInferred` wherever it is shown.
//

import Foundation
#if canImport(Darwin)
import Darwin
#endif

// MARK: - Chip

/// Which line of Apple silicon a chip belongs to.
public enum AppleChipFamily: String, Codable, Sendable {
    case aSeries
    case mSeries
    case unknown
}

/// One Apple chip, as far as this app needs to care about it.
public struct AppleChip: Codable, Hashable, Sendable {

    /// What a person would call it, e.g. `A19 Pro`, `M4`.
    public var name: String

    public var family: AppleChipFamily

    /// A single monotonic number used to compare chips.
    ///
    /// For the A series it is just the A number: A12 is 12, A19 is 19.
    /// The M series is mapped onto the same scale by the A-series generation it
    /// shares its core design with (M1 -> 14, M2 -> 15, M3 -> 17, M4 -> 18,
    /// M5 -> 19), so that one comparison works for both lines.
    ///
    /// 0 means "not determined". Callers must treat 0 as unknown and fall back
    /// to a measured signal, never as "very old".
    public var generation: Int

    /// The "Pro" suffix, e.g. A17 Pro. Display only.
    public var isPro: Bool

    /// True when `name` and `generation` came from the major-number pattern
    /// rule rather than from a hand-checked table entry. Shown to the user as
    /// a softer phrasing so we never state a guess as a fact.
    public var isInferred: Bool

    public init(
        name: String,
        family: AppleChipFamily,
        generation: Int,
        isPro: Bool,
        isInferred: Bool = false
    ) {
        self.name = name
        self.family = family
        self.generation = generation
        self.isPro = isPro
        self.isInferred = isInferred
    }
}

// MARK: - Device model

/// One row of the catalogue: what we know about a hardware model identifier.
public struct AppleDeviceModel: Codable, Hashable, Sendable {

    /// The raw `hw.machine` value, e.g. `iPhone18,2`.
    public var identifier: String

    /// e.g. "iPhone 17 Pro Max". nil when the identifier is not in the table.
    public var marketingName: String?

    public var chip: AppleChip?

    /// What Apple's spec sheet says about a LiDAR scanner on this model.
    ///
    /// ADVISORY ONLY. ARKit is the authority. nil means "not in the table, so
    /// no opinion", which is different from `false` ("in the table, and Apple
    /// does not fit a scanner to this one").
    public var catalogHasLiDAR: Bool?

    public var isPhone: Bool
    public var isPad: Bool

    /// True when this device is the iPhone Simulator running on a Mac, which
    /// deserves its own explanation rather than being called an incompatible
    /// phone.
    public var isSimulator: Bool

    public init(
        identifier: String,
        marketingName: String?,
        chip: AppleChip?,
        catalogHasLiDAR: Bool?,
        isPhone: Bool,
        isPad: Bool,
        isSimulator: Bool
    ) {
        self.identifier = identifier
        self.marketingName = marketingName
        self.chip = chip
        self.catalogHasLiDAR = catalogHasLiDAR
        self.isPhone = isPhone
        self.isPad = isPad
        self.isSimulator = isSimulator
    }

    /// The best short thing to call this device on screen.
    public var displayName: String {
        marketingName ?? identifier
    }
}

// MARK: - The catalogue

public enum AppleSiliconCatalog {

    // MARK: Reading the current device

    /// The hardware model identifier of the device this code is running on.
    ///
    /// On a real device this is `hw.machine`, e.g. `iPhone18,2`. In the
    /// Simulator `hw.machine` reports the Mac's own architecture ("arm64" or
    /// "x86_64"), and the phone being simulated is in an environment variable
    /// instead, so both are handled.
    public static func currentIdentifier() -> String {
        #if targetEnvironment(simulator)
        if let simulated = ProcessInfo.processInfo
            .environment["SIMULATOR_MODEL_IDENTIFIER"], !simulated.isEmpty
        {
            return simulated
        }
        return "Simulator"
        #else
        return sysctlString("hw.machine") ?? "unknown"
        #endif
    }

    /// Everything the catalogue knows about the running device.
    public static func currentModel() -> AppleDeviceModel {
        var model = lookup(identifier: currentIdentifier())
        #if targetEnvironment(simulator)
        model.isSimulator = true
        #endif
        return model
    }

    /// Everything the catalogue knows about one identifier.
    ///
    /// Never fails. An unknown identifier comes back with `marketingName` nil,
    /// `catalogHasLiDAR` nil, and a `chip` that is either inferred from the
    /// major-number pattern or nil.
    public static func lookup(identifier: String) -> AppleDeviceModel {
        let isPhone = identifier.hasPrefix("iPhone")
        let isPad = identifier.hasPrefix("iPad")

        if let row = table[identifier] {
            return AppleDeviceModel(
                identifier: identifier,
                marketingName: row.name,
                chip: row.chip,
                catalogHasLiDAR: row.hasLiDAR,
                isPhone: isPhone,
                isPad: isPad,
                isSimulator: false
            )
        }

        // Not in the table. For an iPhone we can still infer the chip from the
        // major number; for anything else we admit we do not know.
        var inferred: AppleChip?
        if isPhone, let major = majorNumber(of: identifier) {
            inferred = inferredChip(forPhoneMajor: major)
        }

        return AppleDeviceModel(
            identifier: identifier,
            marketingName: nil,
            chip: inferred,
            catalogHasLiDAR: nil,
            isPhone: isPhone,
            isPad: isPad,
            isSimulator: false
        )
    }

    // MARK: Inference

    /// The observed pattern: an iPhone's chip A-number is its identifier major
    /// number plus one, and has been for every model from iPhone8,x (iPhone 6s,
    /// A9) to iPhone18,x (iPhone 17, A19).
    ///
    /// Used only when the exact identifier is not in the table, which in
    /// practice means a model released after this file was last edited. It
    /// gives a newer phone a sensible chip generation instead of "unknown", and
    /// the result is marked `isInferred` so the UI can hedge the wording.
    ///
    /// It cannot tell Pro from non-Pro, because that is not encoded in the
    /// major number, so it never claims Pro.
    public static func inferredChip(forPhoneMajor major: Int) -> AppleChip? {
        guard major >= 8, major <= 40 else { return nil }
        let aNumber = major + 1
        return AppleChip(
            name: "A\(aNumber)",
            family: .aSeries,
            generation: aNumber,
            isPro: false,
            isInferred: true
        )
    }

    /// A last-resort chip generation derived from the Metal GPU family, used
    /// when the model identifier could not be read or parsed at all.
    ///
    /// The mapping is the historical one: A11 shipped Apple4, A12 Apple5,
    /// A13 Apple6, A14 Apple7, A15 and A16 Apple8, A17 Pro and M3 Apple9.
    /// Because Apple8 covers two chip generations this returns the LOWER of
    /// them, which is the safe direction: it can only ever understate what the
    /// device is, never overstate it.
    public static func generation(forMetalAppleFamily family: Int) -> Int? {
        switch family {
        case 4: return 11
        case 5: return 12
        case 6: return 13
        case 7: return 14
        case 8: return 15
        case 9: return 17
        case let higher where higher > 9:
            // A family newer than anything known here. Apple has never gone
            // backwards, so "at least as new as the newest we know" is true.
            return 17 + (higher - 9)
        default: return nil
        }
    }

    // MARK: Helpers

    /// "iPhone18,2" -> 18.
    public static func majorNumber(of identifier: String) -> Int? {
        let fromFirstDigit = identifier.drop { !$0.isNumber }
        let major = fromFirstDigit.prefix { $0.isNumber }
        return Int(major)
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
            return nil
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
            return nil
        }
        let value = String(cString: buffer)
        return value.isEmpty ? nil : value
    }

    // MARK: The table

    private struct Row {
        let name: String
        let chip: AppleChip
        /// Apple's spec sheet. Advisory only, see the file header.
        let hasLiDAR: Bool
    }

    private static func a(_ number: Int, pro: Bool = false) -> AppleChip {
        AppleChip(
            name: pro ? "A\(number) Pro" : "A\(number)",
            family: .aSeries,
            generation: number,
            isPro: pro
        )
    }

    private static func m(_ number: Int, generation: Int) -> AppleChip {
        AppleChip(
            name: "M\(number)",
            family: .mSeries,
            generation: generation,
            isPro: false
        )
    }

    private static let a12z = AppleChip(
        name: "A12Z Bionic",
        family: .aSeries,
        generation: 12,
        isPro: false
    )

    // iPhone rows are hand-checked against Apple's own model-identifier and
    // tech-spec pages. iPad rows are a courtesy: this app is iPhone-only
    // (TARGETED_DEVICE_FAMILY is 1 in project.yml) but an iPhone-only build
    // still runs on iPad in compatibility mode, and naming the device
    // correctly there costs a few lines.
    //
    // TODO(nimbus): the iPhone18,x rows (iPhone 17 family, 2025) are the
    // newest and the least independently verified here. If one is wrong the
    // only visible effect is a wrong marketing NAME on the device-details
    // line; the tier, the feature list and the incompatible reason are all
    // decided from measured ARKit / Metal / ProcessInfo values, and the probe
    // explicitly detects and reports a table-versus-ARKit disagreement rather
    // than believing the table. Correct a row here and nothing else changes.
    private static let table: [String: Row] = [

        // --- iPhone, A9 through A11: no ARKit scene depth, no LiDAR.
        "iPhone8,1": Row(name: "iPhone 6s", chip: a(9), hasLiDAR: false),
        "iPhone8,2": Row(name: "iPhone 6s Plus", chip: a(9), hasLiDAR: false),
        "iPhone8,4": Row(name: "iPhone SE (1st generation)", chip: a(9), hasLiDAR: false),
        "iPhone9,1": Row(name: "iPhone 7", chip: a(10), hasLiDAR: false),
        "iPhone9,3": Row(name: "iPhone 7", chip: a(10), hasLiDAR: false),
        "iPhone9,2": Row(name: "iPhone 7 Plus", chip: a(10), hasLiDAR: false),
        "iPhone9,4": Row(name: "iPhone 7 Plus", chip: a(10), hasLiDAR: false),
        "iPhone10,1": Row(name: "iPhone 8", chip: a(11), hasLiDAR: false),
        "iPhone10,4": Row(name: "iPhone 8", chip: a(11), hasLiDAR: false),
        "iPhone10,2": Row(name: "iPhone 8 Plus", chip: a(11), hasLiDAR: false),
        "iPhone10,5": Row(name: "iPhone 8 Plus", chip: a(11), hasLiDAR: false),
        "iPhone10,3": Row(name: "iPhone X", chip: a(11), hasLiDAR: false),
        "iPhone10,6": Row(name: "iPhone X", chip: a(11), hasLiDAR: false),

        // --- A12 / A13: ARKit world tracking, still no LiDAR.
        "iPhone11,2": Row(name: "iPhone XS", chip: a(12), hasLiDAR: false),
        "iPhone11,4": Row(name: "iPhone XS Max", chip: a(12), hasLiDAR: false),
        "iPhone11,6": Row(name: "iPhone XS Max", chip: a(12), hasLiDAR: false),
        "iPhone11,8": Row(name: "iPhone XR", chip: a(12), hasLiDAR: false),
        "iPhone12,1": Row(name: "iPhone 11", chip: a(13), hasLiDAR: false),
        "iPhone12,3": Row(name: "iPhone 11 Pro", chip: a(13), hasLiDAR: false),
        "iPhone12,5": Row(name: "iPhone 11 Pro Max", chip: a(13), hasLiDAR: false),
        "iPhone12,8": Row(name: "iPhone SE (2nd generation)", chip: a(13), hasLiDAR: false),

        // --- A14: the first iPhones with a LiDAR scanner (Pro only).
        "iPhone13,1": Row(name: "iPhone 12 mini", chip: a(14), hasLiDAR: false),
        "iPhone13,2": Row(name: "iPhone 12", chip: a(14), hasLiDAR: false),
        "iPhone13,3": Row(name: "iPhone 12 Pro", chip: a(14), hasLiDAR: true),
        "iPhone13,4": Row(name: "iPhone 12 Pro Max", chip: a(14), hasLiDAR: true),

        // --- A15.
        "iPhone14,4": Row(name: "iPhone 13 mini", chip: a(15), hasLiDAR: false),
        "iPhone14,5": Row(name: "iPhone 13", chip: a(15), hasLiDAR: false),
        "iPhone14,2": Row(name: "iPhone 13 Pro", chip: a(15), hasLiDAR: true),
        "iPhone14,3": Row(name: "iPhone 13 Pro Max", chip: a(15), hasLiDAR: true),
        "iPhone14,6": Row(name: "iPhone SE (3rd generation)", chip: a(15), hasLiDAR: false),
        "iPhone14,7": Row(name: "iPhone 14", chip: a(15), hasLiDAR: false),
        "iPhone14,8": Row(name: "iPhone 14 Plus", chip: a(15), hasLiDAR: false),

        // --- A16.
        "iPhone15,2": Row(name: "iPhone 14 Pro", chip: a(16), hasLiDAR: true),
        "iPhone15,3": Row(name: "iPhone 14 Pro Max", chip: a(16), hasLiDAR: true),
        "iPhone15,4": Row(name: "iPhone 15", chip: a(16), hasLiDAR: false),
        "iPhone15,5": Row(name: "iPhone 15 Plus", chip: a(16), hasLiDAR: false),

        // --- A17 Pro.
        "iPhone16,1": Row(name: "iPhone 15 Pro", chip: a(17, pro: true), hasLiDAR: true),
        "iPhone16,2": Row(name: "iPhone 15 Pro Max", chip: a(17, pro: true), hasLiDAR: true),

        // --- A18 / A18 Pro.
        "iPhone17,1": Row(name: "iPhone 16 Pro", chip: a(18, pro: true), hasLiDAR: true),
        "iPhone17,2": Row(name: "iPhone 16 Pro Max", chip: a(18, pro: true), hasLiDAR: true),
        "iPhone17,3": Row(name: "iPhone 16", chip: a(18), hasLiDAR: false),
        "iPhone17,4": Row(name: "iPhone 16 Plus", chip: a(18), hasLiDAR: false),
        "iPhone17,5": Row(name: "iPhone 16e", chip: a(18), hasLiDAR: false),

        // --- A19 / A19 Pro. See the TODO above: newest rows, least verified.
        "iPhone18,1": Row(name: "iPhone 17 Pro", chip: a(19, pro: true), hasLiDAR: true),
        "iPhone18,2": Row(name: "iPhone 17 Pro Max", chip: a(19, pro: true), hasLiDAR: true),
        "iPhone18,3": Row(name: "iPhone 17", chip: a(19), hasLiDAR: false),
        "iPhone18,4": Row(name: "iPhone Air", chip: a(19, pro: true), hasLiDAR: false),

        // --- iPad Pro, the other LiDAR family. Courtesy rows only.
        "iPad8,9": Row(name: "iPad Pro 11-inch (2nd generation)", chip: a12z, hasLiDAR: true),
        "iPad8,10": Row(name: "iPad Pro 11-inch (2nd generation)", chip: a12z, hasLiDAR: true),
        "iPad8,11": Row(name: "iPad Pro 12.9-inch (4th generation)", chip: a12z, hasLiDAR: true),
        "iPad8,12": Row(name: "iPad Pro 12.9-inch (4th generation)", chip: a12z, hasLiDAR: true),
        "iPad13,4": Row(name: "iPad Pro 11-inch (3rd generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad13,5": Row(name: "iPad Pro 11-inch (3rd generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad13,6": Row(name: "iPad Pro 11-inch (3rd generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad13,7": Row(name: "iPad Pro 11-inch (3rd generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad13,8": Row(name: "iPad Pro 12.9-inch (5th generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad13,9": Row(name: "iPad Pro 12.9-inch (5th generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad13,10": Row(name: "iPad Pro 12.9-inch (5th generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad13,11": Row(name: "iPad Pro 12.9-inch (5th generation)", chip: m(1, generation: 14), hasLiDAR: true),
        "iPad14,3": Row(name: "iPad Pro 11-inch (4th generation)", chip: m(2, generation: 15), hasLiDAR: true),
        "iPad14,4": Row(name: "iPad Pro 11-inch (4th generation)", chip: m(2, generation: 15), hasLiDAR: true),
        "iPad14,5": Row(name: "iPad Pro 12.9-inch (6th generation)", chip: m(2, generation: 15), hasLiDAR: true),
        "iPad14,6": Row(name: "iPad Pro 12.9-inch (6th generation)", chip: m(2, generation: 15), hasLiDAR: true),
        "iPad16,3": Row(name: "iPad Pro 11-inch (M4)", chip: m(4, generation: 18), hasLiDAR: true),
        "iPad16,4": Row(name: "iPad Pro 11-inch (M4)", chip: m(4, generation: 18), hasLiDAR: true),
        "iPad16,5": Row(name: "iPad Pro 13-inch (M4)", chip: m(4, generation: 18), hasLiDAR: true),
        "iPad16,6": Row(name: "iPad Pro 13-inch (M4)", chip: m(4, generation: 18), hasLiDAR: true),
    ]
}
