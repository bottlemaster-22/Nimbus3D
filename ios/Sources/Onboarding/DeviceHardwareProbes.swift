//
//  DeviceHardwareProbes.swift
//  Onboarding
//
//  The four real measurements the compatibility verdict is built from. Each one
//  is a small value type with a single `probe()` that asks the system and
//  records exactly what it was told, including "I could not find out".
//
//    LiDARCapability            ARKit: scene depth, scene reconstruction,
//                               mesh classification
//    MetalCapability            Metal: is there a GPU, which Apple family
//    DeviceMemoryFacts          ProcessInfo + os_proc_available_memory + free
//                               disk space
//    SustainedPerformanceEstimate  chip generation and thermal headroom,
//                               folded into the 0...3 class the contract asks
//                               for
//
//  Nothing here decides a tier. `DeviceCompatibilityProbe` does that, from
//  these values. Keeping measurement and judgement apart is what makes the
//  judgement auditable.
//

import Foundation

#if canImport(ARKit)
import ARKit
#endif

#if canImport(Metal)
import Metal
#endif

#if canImport(os)
// `os_proc_available_memory()` is declared in <os/proc.h>, which the `os`
// clang module covers. If a future SDK moves it, this one import and the one
// call site in `DeviceMemoryFacts.probe()` are the only things to change; the
// call is already wrapped in a fallback.
import os
#endif

// MARK: - LiDAR

/// What ARKit says about the laser scanner on this device.
///
/// There is no direct "does this iPhone have LiDAR" API and no
/// `UIRequiredDeviceCapabilities` key for it. The supported way to ask is to
/// query the ARKit configuration for the frame semantics and the scene
/// reconstruction modes that only the LiDAR hardware can provide, which is what
/// this does. Both queries are static, cheap, and independent of camera
/// permission, so this runs safely at first launch before any prompt.
public struct LiDARCapability: Codable, Hashable, Sendable {

    /// `ARWorldTrackingConfiguration.isSupported`. False on a device with no
    /// A12 or later chip, and false in the Simulator.
    public var supportsWorldTracking: Bool

    /// `supportsFrameSemantics(.sceneDepth)`. This is the important one: it is
    /// the per-frame 256x192 depth map the whole pipeline is built on.
    public var supportsSceneDepth: Bool

    /// `supportsFrameSemantics(.smoothedSceneDepth)`. We deliberately do NOT
    /// use the smoothed map for training (the data format stores the raw one),
    /// but its presence is a second confirmation of the same hardware.
    public var supportsSmoothedSceneDepth: Bool

    /// `supportsSceneReconstruction(.mesh)`. The ARKit world mesh.
    public var supportsSceneReconstruction: Bool

    /// `supportsSceneReconstruction(.meshWithClassification)`. The mesh with
    /// per-face wall / floor / ceiling / window / door labels, which the
    /// capture HUD paints coverage onto and the window handling depends on.
    public var supportsMeshClassification: Bool

    /// True when this code is running in the Simulator, where every answer
    /// above is a stub and none of it says anything about a real phone.
    public var isSimulator: Bool

    /// The verdict: does this device have a usable laser scanner.
    ///
    /// Scene depth is the requirement. Everything downstream, the native depth
    /// sidecars, free-space carving, the trust field, reads from it.
    public var hasLiDAR: Bool { supportsSceneDepth }

    public static func probe() -> LiDARCapability {
        #if targetEnvironment(simulator)
        // The Simulator has no cameras and no scanner. ARKit's static queries
        // there report nothing useful, so rather than pretend, say plainly
        // that this is the Simulator and let the caller explain that.
        return LiDARCapability(
            supportsWorldTracking: false,
            supportsSceneDepth: false,
            supportsSmoothedSceneDepth: false,
            supportsSceneReconstruction: false,
            supportsMeshClassification: false,
            isSimulator: true
        )
        #elseif canImport(ARKit)
        let worldTracking = ARWorldTrackingConfiguration.isSupported
        guard worldTracking else {
            // No ARKit world tracking at all, so none of the finer queries
            // mean anything. Report the zeroes honestly rather than asking.
            return LiDARCapability(
                supportsWorldTracking: false,
                supportsSceneDepth: false,
                supportsSmoothedSceneDepth: false,
                supportsSceneReconstruction: false,
                supportsMeshClassification: false,
                isSimulator: false
            )
        }
        return LiDARCapability(
            supportsWorldTracking: true,
            supportsSceneDepth:
                ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth),
            supportsSmoothedSceneDepth:
                ARWorldTrackingConfiguration
                    .supportsFrameSemantics(.smoothedSceneDepth),
            supportsSceneReconstruction:
                ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh),
            supportsMeshClassification:
                ARWorldTrackingConfiguration
                    .supportsSceneReconstruction(.meshWithClassification),
            isSimulator: false
        )
        #else
        // ARKit could not even be imported. Not a situation an iOS build should
        // ever be in, but returning false is the honest answer if it happens.
        return LiDARCapability(
            supportsWorldTracking: false,
            supportsSceneDepth: false,
            supportsSmoothedSceneDepth: false,
            supportsSceneReconstruction: false,
            supportsMeshClassification: false,
            isSimulator: false
        )
        #endif
    }
}

// MARK: - Metal

/// What Metal says about the GPU. The trainer and the renderer are both pure
/// Metal, so no GPU means no product, and the GPU family is the most reliable
/// measured proxy for how new the chip is.
public struct MetalCapability: Codable, Hashable, Sendable {

    /// False when `MTLCreateSystemDefaultDevice()` returned nil, which on a
    /// real iPhone means something is badly wrong.
    public var hasDevice: Bool

    /// The GPU's own name, e.g. "Apple A19 GPU". Display only.
    public var deviceName: String?

    /// Highest supported `MTLGPUFamily.apple*` as a plain number, e.g. 9 for
    /// Apple9. nil when no family in the probed range was supported.
    public var appleFamilyOrdinal: Int?

    /// The readable form the contract asks for, e.g. "Apple9".
    public var familyName: String? {
        appleFamilyOrdinal.map { "Apple\($0)" }
    }

    /// Whether the GPU supports the Metal 3 feature set.
    public var supportsMetal3: Bool

    /// `MTLDevice.hasUnifiedMemory`. Always true on iPhone; recorded because
    /// the memory budget maths assumes it.
    public var hasUnifiedMemory: Bool

    /// The Apple GPU family raw values run 1001 (Apple1) upwards, one per
    /// family. Probing by raw value rather than by named case means a GPU
    /// newer than this SDK still reports a number instead of falling off the
    /// end, and an SDK that does not know a case simply skips it: the failable
    /// `MTLGPUFamily(rawValue:)` returns nil for a value the enum has no case
    /// for, so nothing here can crash or fail to compile on an older SDK.
    private static let appleFamilyRawBase = 1001
    private static let appleFamilyMaxOrdinal = 16

    /// `MTLGPUFamily.metal3` is raw value 5001. Probed by raw value for the
    /// same forward and backward compatibility reason.
    private static let metal3Raw = 5001

    public static func probe() -> MetalCapability {
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            return MetalCapability(
                hasDevice: false,
                deviceName: nil,
                appleFamilyOrdinal: nil,
                supportsMetal3: false,
                hasUnifiedMemory: false
            )
        }

        var highest: Int?
        for ordinal in 1...appleFamilyMaxOrdinal {
            guard
                let family = MTLGPUFamily(
                    rawValue: appleFamilyRawBase + ordinal - 1
                )
            else { continue }
            if device.supportsFamily(family) {
                highest = ordinal
            }
        }

        let metal3: Bool
        if let family = MTLGPUFamily(rawValue: metal3Raw) {
            metal3 = device.supportsFamily(family)
        } else {
            metal3 = false
        }

        return MetalCapability(
            hasDevice: true,
            deviceName: device.name,
            appleFamilyOrdinal: highest,
            supportsMetal3: metal3,
            hasUnifiedMemory: device.hasUnifiedMemory
        )
        #else
        return MetalCapability(
            hasDevice: false,
            deviceName: nil,
            appleFamilyOrdinal: nil,
            supportsMetal3: false,
            hasUnifiedMemory: false
        )
        #endif
    }
}

// MARK: - Memory and storage

/// How much room this app actually has, in RAM and on disk.
public struct DeviceMemoryFacts: Codable, Hashable, Sendable {

    /// `ProcessInfo.physicalMemory`: the RAM soldered to the phone. This is the
    /// number people quote, and it is NOT the number that matters.
    public var totalBytes: UInt64

    /// `os_proc_available_memory()`: how many more bytes THIS process may
    /// allocate before iOS kills it. On a 6 GB iPhone this is typically well
    /// under 2 GB unless the increased-memory-limit entitlement is both
    /// requested (it is, in project.yml) and actually granted by the signing
    /// team. It is the number the trainer's budget is built from.
    public var availableBytes: UInt64

    /// True when `availableBytes` is a fallback estimate because
    /// `os_proc_available_memory()` returned 0, which it does outside a normal
    /// app process. Never present a guessed number as a measurement.
    public var availableIsEstimated: Bool

    /// Free space on the volume the app writes scans to, in bytes, as iOS
    /// reports it for "important" usage. nil when it could not be read.
    public var freeDiskBytes: Int64?

    public static func probe() -> DeviceMemoryFacts {
        let info = ProcessInfo.processInfo
        let total = info.physicalMemory

        var available: UInt64 = 0
        var estimated = false

        #if canImport(os) && os(iOS)
        let reported = os_proc_available_memory()
        if reported > 0 {
            available = UInt64(reported)
        }
        #endif

        if available == 0 {
            // No measurement. A quarter of physical RAM is a deliberately
            // conservative stand-in, and the flag says it is a stand-in so
            // nothing downstream reports it as measured.
            available = total / 4
            estimated = true
        }

        return DeviceMemoryFacts(
            totalBytes: total,
            availableBytes: available,
            availableIsEstimated: estimated,
            freeDiskBytes: freeDiskBytes()
        )
    }

    private static func freeDiskBytes() -> Int64? {
        let fileManager = FileManager.default
        guard
            let documents = try? fileManager.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: false
            )
        else { return nil }
        let values = try? documents.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        )
        return values?.volumeAvailableCapacityForImportantUsage
    }
}

// MARK: - Sustained performance

/// The contract's `sustainedPerformanceClass`, 0 (unknown) to 3 (best),
/// "derived from chip generation and thermal headroom", plus the working it
/// showed so the UI can explain a low score instead of just printing a number.
///
/// This is a HEURISTIC, not a benchmark. It is not measured by running work; it
/// is inferred from the chip generation, the current thermal state and whether
/// Low Power Mode is on. That is enough for its only job, which is choosing a
/// starting training budget that the trainer then lowers as it measures real
/// heat and memory. It is never used to block anything.
public struct SustainedPerformanceEstimate: Codable, Hashable, Sendable {

    /// 0...3, as the contract defines it.
    public var performanceClass: Int

    /// The class before the live penalties below were applied. Useful because
    /// it is the number that will come back once the phone cools down or Low
    /// Power Mode is switched off.
    public var baseClass: Int

    /// True when Low Power Mode cost this device a point.
    public var reducedByLowPowerMode: Bool

    /// True when the phone being warm right now cost this device a point.
    public var reducedByHeat: Bool

    public var thermalLevel: ThermalLevel
    public var lowPowerModeEnabled: Bool

    /// The number of cores iOS is currently willing to schedule on.
    public var activeProcessorCount: Int

    /// - Parameter chipGeneration: the A-series-equivalent generation number
    ///   from `AppleChip.generation`, or nil when it could not be determined.
    ///   nil produces class 0, "unknown", which is exactly what the contract
    ///   reserves 0 for.
    public static func probe(chipGeneration: Int?) -> SustainedPerformanceEstimate {
        let info = ProcessInfo.processInfo
        let thermal = ThermalLevel(info.thermalState)
        let lowPower = info.isLowPowerModeEnabled

        // Base class from the chip generation alone.
        //   3  A17 Pro and later: hardware ray tracing, a much wider GPU, and
        //      the thermal design to hold a clock for minutes rather than
        //      seconds. This is the tier that trains comfortably.
        //   2  A15 and A16: trains, and holds up well enough for a room.
        //   1  A12 to A14: works, gets hot, wants a smaller budget.
        //   0  older, or not determined.
        let base: Int
        switch chipGeneration {
        case .some(let generation) where generation >= 17: base = 3
        case .some(let generation) where generation >= 15: base = 2
        case .some(let generation) where generation >= 12: base = 1
        case .some: base = 0
        case nil: base = 0
        }

        var value = base
        var lostToLowPower = false
        var lostToHeat = false

        if lowPower, value > 0 {
            // Low Power Mode caps the CPU and GPU clocks. This is temporary and
            // the user can undo it, which is why the screen offers a re-check.
            value -= 1
            lostToLowPower = true
        }
        if thermal >= .serious, value > 0 {
            value -= 1
            lostToHeat = true
        }

        return SustainedPerformanceEstimate(
            performanceClass: value,
            baseClass: base,
            reducedByLowPowerMode: lostToLowPower,
            reducedByHeat: lostToHeat,
            thermalLevel: thermal,
            lowPowerModeEnabled: lowPower,
            activeProcessorCount: info.activeProcessorCount
        )
    }
}
