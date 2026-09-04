//
//  DeviceCompatibilityProbe.swift
//  Onboarding
//
//  The one place the compatibility VERDICT is decided.
//
//  `DeviceHardwareProbes.swift` measures. This file judges, and it judges only
//  from measurements: what ARKit reports about scene depth, whether Metal
//  produced a device and which Apple GPU family it supports, what ProcessInfo
//  says about memory, thermal state and Low Power Mode, and the free space on
//  the volume scans are written to. The model-identifier table in
//  `AppleSiliconCatalog.swift` contributes a chip generation and a name; it is
//  deliberately never allowed to be the thing that decides a tier, so a stale
//  table cannot produce a wrong verdict.
//
//  Every threshold is a named constant in `Thresholds` below, with the reason
//  it has the value it has written next to it. If a verdict ever looks wrong,
//  that enum is the whole argument.
//

import Foundation

#if canImport(os)
import os
#endif

// MARK: - Findings

/// Everything one compatibility check produced: the contract report the rest of
/// the app consumes, plus the raw measurements behind it.
///
/// `DeviceCapabilityReport` is the contract type and is deliberately narrow.
/// The onboarding screens want to say things like "your phone is in Low Power
/// Mode" and "you have 1.8 GB of space left", so the raw probes travel
/// alongside rather than being flattened away.
public struct DeviceCompatibilityFindings: Codable, Sendable {

    /// The contract type, exactly as `DeviceCompatibilityService` returns it.
    public var report: DeviceCapabilityReport

    public var model: AppleDeviceModel
    public var lidar: LiDARCapability
    public var metal: MetalCapability
    public var memory: DeviceMemoryFacts
    public var performance: SustainedPerformanceEstimate

    /// The chip generation actually used for the decision, and where it came
    /// from, so a support conversation can start from a fact.
    public var effectiveChipGeneration: Int?
    public var chipGenerationSource: ChipGenerationSource

    /// True when the model table claims this device has a scanner but ARKit
    /// says it does not. That is not a "no scanner" situation and it does not
    /// get the "no scanner" explanation; see `Reason.scannerNotRespondin`.
    public var catalogDisagreesWithARKit: Bool

    public var checkedAt: Date

    public enum ChipGenerationSource: String, Codable, Sendable {
        /// A hand-checked row in the model table.
        case modelTable
        /// The iPhone major-number pattern, for a model newer than the table.
        case identifierPattern
        /// The Metal GPU family, when the identifier told us nothing.
        case metalFamily
        /// Nothing worked. The tier falls back to memory and Metal alone.
        case undetermined
    }

    /// A ready-made error for any module that needs to refuse work on this
    /// device and wants the same sentence the onboarding screen shows.
    public var incompatibilityError: NimbusError? {
        guard report.tier == .incompatible else { return nil }
        return .deviceIncompatible(
            report.incompatibleReason ?? OnboardingCopy.lastResortReason
        )
    }
}

// MARK: - The service

/// `DeviceCompatibilityService` for `Sources/Onboarding`.
///
/// Cheap: one sysctl, a handful of ARKit and Metal static queries, and a
/// ProcessInfo read. Comfortably under a frame on a real device, which is why
/// the contract says it is fine to run on every launch. The result is cached
/// here as well as persisted by `DeviceReportStore`, so the onboarding screens
/// can show the raw findings without probing a second time.
///
/// `@unchecked Sendable` is deliberate and narrow: the only mutable state is
/// `cachedFindings`, and every read and write of it goes through `lock`.
public final class DeviceCompatibilityProbe: DeviceCompatibilityService, @unchecked Sendable {

    // MARK: Thresholds
    //
    // The entire argument for every verdict this file reaches.

    public enum Thresholds {

        /// A15 and later for the full tier.
        ///
        /// A14 (iPhone 12 Pro, the first iPhone with a scanner) can capture and
        /// pre-pass perfectly well, and it can train, but it runs hot and its
        /// GPU is roughly half the width of an A17 Pro's. Putting it in
        /// `.limited` is not a judgement about the phone, it is the difference
        /// between a smaller on-device budget plus the option of the PC helper,
        /// and the unrestricted one. A15 is where a room-sized scan trains
        /// on-device without the thermal policy fighting it the whole way.
        public static let fullTierMinimumChipGeneration = 15

        /// Every iPhone fitted with a scanner has at least 6 GB of RAM, so this
        /// bar is not there to exclude a phone. It is there so that an unknown
        /// or misread device with genuinely little memory lands in `.limited`
        /// rather than being promoted on a chip guess. 5.5 GB rather than 6 GB
        /// because `physicalMemory` reports slightly under the marketing
        /// figure on some models.
        public static let fullTierMinimumTotalMemoryBytes: UInt64 = 5_500_000_000

        /// What `os_proc_available_memory()` must report for the full tier.
        ///
        /// Deliberately low. This number moves around during a session, and the
        /// tier must not flicker between launches, so the tier bar is a floor
        /// that only a genuinely memory-starved device falls through. The
        /// tighter bar that decides whether full-detail training is offered is
        /// `highDetailMinimumAvailableMemoryBytes`, and that one is allowed to
        /// change from one check to the next because it is a feature, not a
        /// verdict.
        public static let fullTierMinimumAvailableMemoryBytes: UInt64 = 1_000_000_000

        /// Enough headroom to hold a few hundred thousand splats plus their
        /// Adam moments, the tile lists and a frame cache at once.
        public static let highDetailMinimumAvailableMemoryBytes: UInt64 = 1_500_000_000

        /// Free disk below which a whole-house scan is likely to run out of
        /// room part-way. A house scan is thousands of frames plus their native
        /// depth and confidence sidecars.
        public static let comfortableScanDiskBytes: Int64 = 4_000_000_000

        /// Below this, even a single-room scan is at risk.
        public static let minimumUsefulScanDiskBytes: Int64 = 1_000_000_000

        /// Apple7 (A14) is the oldest GPU family any scanner-equipped iPhone
        /// has. A device reporting older than this alongside a working scanner
        /// would be a contradiction, and is treated as "do not promote".
        public static let fullTierMinimumMetalAppleFamily = 7
    }

    // MARK: State

    private let lock = NSLock()
    private var cachedFindings: DeviceCompatibilityFindings?

    #if canImport(os)
    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "Onboarding"
    )
    #endif

    public init() {}

    // MARK: DeviceCompatibilityService

    public func evaluate() async -> DeviceCapabilityReport {
        evaluateDetailed().report
    }

    // MARK: The real entry point

    /// The full check, synchronously. `evaluate()` is the contract's async
    /// front door onto this; the work genuinely is synchronous and fast, and
    /// wrapping it in a fake `await` would be dishonest about what it does.
    @discardableResult
    public func evaluateDetailed() -> DeviceCompatibilityFindings {
        let model = AppleSiliconCatalog.currentModel()
        let lidar = LiDARCapability.probe()
        let metal = MetalCapability.probe()
        let memory = DeviceMemoryFacts.probe()

        // Chip generation, best source first.
        var generation: Int?
        var source: DeviceCompatibilityFindings.ChipGenerationSource = .undetermined
        if let chip = model.chip {
            generation = chip.generation
            source = chip.isInferred ? .identifierPattern : .modelTable
        }
        if generation == nil, let family = metal.appleFamilyOrdinal {
            generation = AppleSiliconCatalog.generation(forMetalAppleFamily: family)
            if generation != nil { source = .metalFamily }
        }

        let performance = SustainedPerformanceEstimate.probe(chipGeneration: generation)

        // The model table thinks there is a scanner, ARKit says otherwise.
        let disagreement =
            (model.catalogHasLiDAR == true)
            && !lidar.hasLiDAR
            && !lidar.isSimulator

        let tier = Self.tier(
            lidar: lidar,
            metal: metal,
            memory: memory,
            chipGeneration: generation,
            performance: performance
        )

        let context = ReasonContext(
            model: model,
            lidar: lidar,
            metal: metal,
            memory: memory,
            performance: performance,
            chipGeneration: generation,
            catalogDisagreesWithARKit: disagreement
        )

        let features = Self.features(tier: tier, context: context)

        let report = DeviceCapabilityReport(
            tier: tier,
            hasLiDAR: lidar.hasLiDAR,
            deviceModel: model.identifier,
            chipName: Self.chipDisplayName(for: model, metal: metal),
            totalMemoryBytes: memory.totalBytes,
            availableMemoryBytes: memory.availableBytes,
            systemVersion: ProcessInfo.processInfo
                .operatingSystemVersionString(short: true),
            metalGPUFamily: metal.familyName,
            lowPowerModeEnabled: performance.lowPowerModeEnabled,
            sustainedPerformanceClass: performance.performanceClass,
            features: features,
            incompatibleReason: tier == .incompatible
                ? OnboardingCopy.incompatibleReason(context)
                : nil
        )

        let findings = DeviceCompatibilityFindings(
            report: report,
            model: model,
            lidar: lidar,
            metal: metal,
            memory: memory,
            performance: performance,
            effectiveChipGeneration: generation,
            chipGenerationSource: source,
            catalogDisagreesWithARKit: disagreement,
            checkedAt: Date()
        )

        lock.lock()
        cachedFindings = findings
        lock.unlock()

        #if canImport(os)
        log.info(
            """
            Device check: model=\(model.identifier, privacy: .public) \
            tier=\(tier.rawValue, privacy: .public) \
            lidar=\(lidar.hasLiDAR, privacy: .public) \
            metalFamily=\(metal.familyName ?? "none", privacy: .public) \
            perfClass=\(performance.performanceClass, privacy: .public)
            """
        )
        #endif

        DeviceReportStore.shared.save(findings)
        return findings
    }

    /// The most recent result this probe produced, if it has run. Lets a screen
    /// show the raw measurements without paying for a second check.
    public var lastFindings: DeviceCompatibilityFindings? {
        lock.lock()
        defer { lock.unlock() }
        return cachedFindings
    }

    /// The findings to draw a screen from: the cached ones when this probe has
    /// already run, the persisted ones when a previous launch ran it, and a
    /// fresh check otherwise. Never returns nil, so no screen has to have an
    /// "unknown" state that only exists because of plumbing.
    public func findingsForDisplay() -> DeviceCompatibilityFindings {
        if let cached = lastFindings { return cached }
        if let stored = DeviceReportStore.shared.load()?.findings { return stored }
        return evaluateDetailed()
    }

    // MARK: - Tier

    /// The verdict.
    ///
    /// Note what is NOT consulted here: the live thermal state and Low Power
    /// Mode. Both are temporary, and a tier that changes because the phone is
    /// warm would tell the user their iPhone had somehow become a worse iPhone.
    /// They are shown on screen, they lower the sustained-performance class,
    /// and the trainer reacts to them live. They do not move the tier.
    static func tier(
        lidar: LiDARCapability,
        metal: MetalCapability,
        memory: DeviceMemoryFacts,
        chipGeneration: Int?,
        performance: SustainedPerformanceEstimate
    ) -> DeviceTier {

        // No GPU, no product: the trainer and the renderer are both pure Metal.
        guard metal.hasDevice else { return .incompatible }

        // No scanner, no product. This is the whole idea of the app.
        guard lidar.hasLiDAR else { return .incompatible }

        // Everything from here is a scanner-equipped device, so the only
        // question left is full or reduced.
        let chipIsRecentEnough =
            (chipGeneration ?? 0) >= Thresholds.fullTierMinimumChipGeneration

        let gpuIsRecentEnough =
            (metal.appleFamilyOrdinal ?? 0) >= Thresholds.fullTierMinimumMetalAppleFamily

        let hasEnoughTotalMemory =
            memory.totalBytes >= Thresholds.fullTierMinimumTotalMemoryBytes

        // An estimated available figure is not evidence, so it is not allowed
        // to fail this test on its own; only a real measurement can.
        let hasEnoughAvailableMemory =
            memory.availableIsEstimated
            || memory.availableBytes >= Thresholds.fullTierMinimumAvailableMemoryBytes

        if chipIsRecentEnough
            && gpuIsRecentEnough
            && hasEnoughTotalMemory
            && hasEnoughAvailableMemory
        {
            return .full
        }
        return .limited
    }

    // MARK: - Chip name for display

    static func chipDisplayName(
        for model: AppleDeviceModel,
        metal: MetalCapability
    ) -> String? {
        if let chip = model.chip {
            // A name from the pattern rule is a guess, and is labelled as one
            // rather than printed as though it were read off the die.
            return chip.isInferred ? "\(chip.name) (best guess)" : chip.name
        }
        // No idea what the chip is called. The GPU family is a real measurement
        // and is better than nothing, but it is not a chip name, so it is not
        // presented as one.
        if let family = metal.familyName {
            return "\(family) graphics"
        }
        return nil
    }

    // MARK: - Feature list

    /// Everything the report carries into the copy that builds the feature and
    /// reason text, in one bag, so those functions cannot quietly start
    /// depending on something that was not measured.
    struct ReasonContext {
        var model: AppleDeviceModel
        var lidar: LiDARCapability
        var metal: MetalCapability
        var memory: DeviceMemoryFacts
        var performance: SustainedPerformanceEstimate
        var chipGeneration: Int?
        var catalogDisagreesWithARKit: Bool
    }

    /// The per-feature availability list the onboarding screens render verbatim.
    ///
    /// Rules this list follows, because the whole point of it is to be trusted:
    ///   * every `false` entry carries a `detail` that names the actual reason
    ///     on THIS device, with the real number in it where there is one;
    ///   * no entry is marked available unless the thing that makes it work was
    ///     actually measured;
    ///   * features that depend on something outside the phone say so.
    static func features(
        tier: DeviceTier,
        context ctx: ReasonContext
    ) -> [FeatureAvailability] {

        var list: [FeatureAvailability] = []

        // 1. The scanner itself.
        list.append(
            FeatureAvailability(
                key: "lidar_scan",
                title: "Measure the room with the laser scanner",
                isAvailable: ctx.lidar.hasLiDAR,
                detail: ctx.lidar.hasLiDAR
                    ? nil
                    : OnboardingCopy.noScannerFeatureDetail(ctx)
            )
        )

        // 2. Surface labels. Every scanner-equipped device has had this, but it
        //    is a separate ARKit capability so it is asked about separately.
        list.append(
            FeatureAvailability(
                key: "surface_labels",
                title: "Tell walls, floors, ceilings and windows apart",
                isAvailable: ctx.lidar.supportsMeshClassification,
                detail: ctx.lidar.supportsMeshClassification
                    ? nil
                    : "This needs the scanner, which this device does not have. "
                        + "Knowing which surface is a window is how the app "
                        + "handles glass, which is the hardest thing in a room "
                        + "to scan."
            )
        )

        // 3. Live guidance during capture.
        list.append(
            FeatureAvailability(
                key: "guided_capture",
                title: "Live guidance while you scan",
                isAvailable: ctx.lidar.hasLiDAR,
                detail: ctx.lidar.hasLiDAR
                    ? "The app paints the room as you cover it and tells you, "
                        + "out loud and by vibration, when to walk around "
                        + "something, get closer, or slow down."
                    : "The guidance is drawn onto the surfaces the scanner "
                        + "finds, so without a scanner there is nothing to "
                        + "draw on."
            )
        )

        // 4. Pre-pass.
        let prePassWorks = ctx.lidar.hasLiDAR && ctx.metal.hasDevice
        list.append(
            FeatureAvailability(
                key: "on_device_prepass",
                title: "Check and tidy up the scan on this phone",
                isAvailable: prePassWorks,
                detail: prePassWorks
                    ? nil
                    : "This step lines up the scan and reports what is missing. "
                        + "It reads the scanner's measurements, so it needs the "
                        + "scanner."
            )
        )

        // 5. On-device training.
        let canTrain = tier != .incompatible
        list.append(
            FeatureAvailability(
                key: "on_device_training",
                title: "Build the 3D model on this phone",
                isAvailable: canTrain,
                detail: OnboardingCopy.trainingFeatureDetail(tier: tier, context: ctx)
            )
        )

        // 6. Full detail on-device.
        let highDetail =
            tier == .full
            && ctx.performance.baseClass >= 2
            && (ctx.memory.availableIsEstimated
                || ctx.memory.availableBytes
                    >= Thresholds.highDetailMinimumAvailableMemoryBytes)
        list.append(
            FeatureAvailability(
                key: "high_detail_model",
                title: "Build it at full detail, without cutting corners",
                isAvailable: highDetail,
                detail: highDetail
                    ? nil
                    : OnboardingCopy.highDetailFeatureDetail(tier: tier, context: ctx)
            )
        )

        // 7. The PC helper. The phone half of this works on every device that
        //    can capture at all; the other half is a free program on a PC, and
        //    saying so is the honest version of "available".
        list.append(
            FeatureAvailability(
                key: "booster_offload",
                title: "Hand a big scan to a PC on your Wi-Fi",
                isAvailable: ctx.lidar.hasLiDAR,
                detail: ctx.lidar.hasLiDAR
                    ? "Optional. You need the free helper program running on a "
                        + "computer on the same Wi-Fi. Nothing is ever uploaded "
                        + "to the internet, and there is nothing to pay for."
                    : "There would be nothing to send: the scan itself needs the "
                        + "scanner."
            )
        )

        // 8. Preview.
        list.append(
            FeatureAvailability(
                key: "preview_walkthrough",
                title: "Walk around the finished scan on the phone",
                isAvailable: ctx.metal.hasDevice,
                detail: ctx.metal.hasDevice
                    ? nil
                    : "The preview is drawn on the graphics chip, and the app "
                        + "could not start it on this device."
            )
        )

        // 9. Export. Pure file writing, works anywhere.
        list.append(
            FeatureAvailability(
                key: "export_share",
                title: "Save and share the model, or open it in Blender",
                isAvailable: true,
                detail: nil
            )
        )

        // 10. Disk space. Real, current, and the number people most often get
        //     caught out by half way through a big scan.
        if let free = ctx.memory.freeDiskBytes {
            let comfortable = free >= Thresholds.comfortableScanDiskBytes
            list.append(
                FeatureAvailability(
                    key: "storage_headroom",
                    title: "Room to store a large scan",
                    isAvailable: comfortable,
                    detail: comfortable
                        ? "You have \(OnboardingFormat.bytes(free)) free."
                        : OnboardingCopy.storageDetail(freeBytes: free)
                )
            )
        }

        return list
    }
}

// MARK: - OS version helper

extension ProcessInfo {
    /// "17.4.1", built from `operatingSystemVersion` rather than parsed out of
    /// `operatingSystemVersionString`, which is a sentence and not a version.
    ///
    /// The patch component is dropped when it is zero, because "18.2" is what
    /// the Settings app shows and "18.2.0" would make a person doubt they were
    /// reading about their own phone.
    func operatingSystemVersionString(short: Bool) -> String {
        let version = operatingSystemVersion
        guard short else { return operatingSystemVersionString }
        if version.patchVersion == 0 {
            return "\(version.majorVersion).\(version.minorVersion)"
        }
        return "\(version.majorVersion).\(version.minorVersion)."
            + "\(version.patchVersion)"
    }
}
