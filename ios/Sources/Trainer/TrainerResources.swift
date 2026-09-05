//
//  TrainerResources.swift
//  Trainer
//
//  EVERY MTLBuffer THE TRAINER OWNS, SIZED FROM THE BUDGET AND MEASURED.
//
//  One class, one allocation site, one byte counter. That last part is not
//  bookkeeping for its own sake: `TrainerProgress.residentBytes` has to be a
//  measured number, and the F10 rule that the trainer may LOWER its budget as
//  it measures real memory only means anything if there is a real measurement
//  to compare against `os_proc_available_memory()`.
//
//  Every buffer is `.storageModeShared`. On Apple silicon that is one physical
//  allocation the CPU and the GPU both address, so a readback is a pointer
//  dereference rather than a blit, and the densifier can rewrite the splat
//  array in place between command buffers.
//
//  TWO capacities matter and they are not the same number:
//
//    * `splatCapacity` - how many Gaussians the parameter, gradient, moment
//      and statistics buffers can hold. This is the budget's `splatCap`.
//    * `instanceCapacity` - how many (Gaussian, tile) pairs the sort can hold
//      in one frame. A Gaussian covering a 4x3 block of tiles is twelve
//      instances. This is sized from a measured average and GROWS when a frame
//      needs more, because `trainer_duplicate_keys` stops writing at the cap
//      rather than overrunning, which turns an under-estimate into missing
//      splats instead of a crash. Missing splats are still wrong, so the
//      overflow is detected and the buffer is grown before the next frame.
//

import Foundation
import Metal
import simd

/// Pixel-grid dimensions one training render runs at.
struct TrainerRenderSize: Equatable, Sendable {
    var width: Int
    var height: Int

    var pixelCount: Int { Swift.max(width, 0) * Swift.max(height, 0) }
    var tileCountX: Int { (Swift.max(width, 1) + TrainerGPUConstants.tileWidth - 1) / TrainerGPUConstants.tileWidth }
    var tileCountY: Int { (Swift.max(height, 1) + TrainerGPUConstants.tileHeight - 1) / TrainerGPUConstants.tileHeight }
    var tileCount: Int { tileCountX * tileCountY }
}

final class TrainerResources {

    let device: MTLDevice

    // MARK: Shape

    private(set) var splatCapacity: Int
    private(set) var instanceCapacity: Int
    private(set) var renderSize: TrainerRenderSize
    private(set) var shCoefficientCount: Int
    /// Native depth samples per frame, which sizes the depth-supervision buffer.
    private(set) var depthSampleCapacity: Int

    /// Floats in the SH buffer per Gaussian: three per coefficient.
    var shFloatsPerSplat: Int { shCoefficientCount * 3 }

    // MARK: Per-Gaussian buffers

    private(set) var splats: MTLBuffer
    private(set) var sh: MTLBuffer
    private(set) var stats: MTLBuffer
    private(set) var draws: MTLBuffer
    private(set) var samplingTopK: MTLBuffer
    private(set) var tilesTouched: MTLBuffer
    private(set) var offsets: MTLBuffer

    private(set) var splatGrad: MTLBuffer
    private(set) var shGrad: MTLBuffer
    private(set) var adamM: MTLBuffer
    private(set) var adamV: MTLBuffer
    private(set) var shAdamM: MTLBuffer
    private(set) var shAdamV: MTLBuffer

    private(set) var gradMean2D: MTLBuffer
    private(set) var gradConic: MTLBuffer
    private(set) var gradColor: MTLBuffer
    private(set) var gradOpacity: MTLBuffer

    private(set) var centers: MTLBuffer

    // MARK: Binning and sorting

    private(set) var keysA: MTLBuffer
    private(set) var keysB: MTLBuffer
    private(set) var valuesA: MTLBuffer
    private(set) var valuesB: MTLBuffer
    private(set) var tileRanges: MTLBuffer
    private(set) var radixHistogram: MTLBuffer
    private(set) var radixHistogramScan: MTLBuffer
    /// One scratch pair per level of the recursive block-sum scan. Three
    /// levels cover 1024^3 elements, which is a billion; the trainer never
    /// gets within three orders of magnitude of that, and the array is built
    /// from the real capacity rather than assumed.
    private(set) var scanBlockSums: [MTLBuffer]
    private(set) var scanBlockSumsScanned: [MTLBuffer]

    // MARK: Per-pixel buffers

    private(set) var renderColor: MTLBuffer
    private(set) var renderAlpha: MTLBuffer
    private(set) var renderDepth: MTLBuffer
    private(set) var renderTFinal: MTLBuffer
    private(set) var renderNContrib: MTLBuffer

    private(set) var gtColor: MTLBuffer
    private(set) var bgColor: MTLBuffer
    private(set) var composited: MTLBuffer
    private(set) var gradFinal: MTLBuffer
    private(set) var gradSplatColor: MTLBuffer
    private(set) var gradDepthRend: MTLBuffer
    private(set) var gradTFinal: MTLBuffer
    private(set) var unknownMask: MTLBuffer

    private(set) var ssimSrc: MTLBuffer
    private(set) var ssimMid: MTLBuffer
    private(set) var ssimTmp: MTLBuffer

    // MARK: Scalars

    private(set) var lossAccum: MTLBuffer
    private(set) var exposureGrad: MTLBuffer
    private(set) var cameraGrad: MTLBuffer
    private(set) var depthSamples: MTLBuffer

    // MARK: Accounting

    /// Every allocation this object holds, in bytes, as Metal reports it.
    /// Recomputed on every reallocation, never estimated.
    private(set) var residentBytes: UInt64 = 0

    // MARK: - Init

    /// Allocates everything. Throws a named error the moment one allocation
    /// fails, rather than continuing with a nil buffer that would read as
    /// zeros at the far end of a shader.
    init(
        device: MTLDevice,
        splatCapacity: Int,
        renderSize: TrainerRenderSize,
        shCoefficientCount: Int,
        depthSampleCapacity: Int,
        instanceCapacity: Int
    ) throws {
        self.device = device
        self.splatCapacity = Swift.max(splatCapacity, 1)
        self.renderSize = renderSize
        self.shCoefficientCount = Swift.max(shCoefficientCount, 1)
        self.depthSampleCapacity = Swift.max(depthSampleCapacity, 1)
        self.instanceCapacity = Swift.max(instanceCapacity, TrainerGPUConstants.scanBlockElements)

        let n = self.splatCapacity
        let shFloats = n * self.shCoefficientCount * 3
        let px = Swift.max(renderSize.pixelCount, 1)
        let tiles = Swift.max(renderSize.tileCount, 1)
        let inst = self.instanceCapacity

        func make(_ name: String, _ bytes: Int) throws -> MTLBuffer {
            let want = Swift.max(bytes, 16)
            guard let buffer = device.makeBuffer(length: want, options: .storageModeShared) else {
                throw TrainerError.allocationFailed(name: name, bytes: want)
            }
            buffer.label = "trainer.\(name)"
            memset(buffer.contents(), 0, want)
            return buffer
        }

        splats = try make("splats", n * MemoryLayout<TrainerSplat>.stride)
        sh = try make("sh", shFloats * MemoryLayout<Float>.stride)
        stats = try make("stats", n * MemoryLayout<TrainerSplatStats>.stride)
        draws = try make("draws", n * MemoryLayout<TrainerSplatDraw>.stride)
        samplingTopK = try make("samplingTopK", n * MemoryLayout<TrainerSamplingTopK>.stride)

        // The scan kernels read a whole 1024-element block whether or not the
        // tail exists, guarded by `idx < count`; the guard is on the READ, so
        // the allocation is padded to a block anyway to keep the write side
        // (`output[idx]`) inside the buffer for every in-range index.
        let paddedSplats = ((n + TrainerGPUConstants.scanBlockElements - 1)
            / TrainerGPUConstants.scanBlockElements) * TrainerGPUConstants.scanBlockElements
        tilesTouched = try make("tilesTouched", paddedSplats * 4)
        offsets = try make("offsets", paddedSplats * 4)

        splatGrad = try make("splatGrad", n * MemoryLayout<TrainerSplatGrad>.stride)
        shGrad = try make("shGrad", shFloats * MemoryLayout<Float>.stride)
        adamM = try make("adamM", n * MemoryLayout<TrainerSplatGrad>.stride)
        adamV = try make("adamV", n * MemoryLayout<TrainerSplatGrad>.stride)
        shAdamM = try make("shAdamM", shFloats * MemoryLayout<Float>.stride)
        shAdamV = try make("shAdamV", shFloats * MemoryLayout<Float>.stride)

        gradMean2D = try make("gradMean2D", n * 2 * 4)
        gradConic = try make("gradConic", n * 3 * 4)
        gradColor = try make("gradColor", n * 3 * 4)
        gradOpacity = try make("gradOpacity", n * 4)
        centers = try make("centers", n * 3 * 4)

        keysA = try make("keysA", inst * 4)
        keysB = try make("keysB", inst * 4)
        valuesA = try make("valuesA", inst * 4)
        valuesB = try make("valuesB", inst * 4)
        tileRanges = try make("tileRanges", tiles * 2 * 4)

        let maxSortBlocks = (inst + TrainerGPUConstants.scanBlockElements - 1)
            / TrainerGPUConstants.scanBlockElements
        let histogramEntries = TrainerGPUConstants.radixBins * Swift.max(maxSortBlocks, 1)
        let histogramPadded = ((histogramEntries + TrainerGPUConstants.scanBlockElements - 1)
            / TrainerGPUConstants.scanBlockElements) * TrainerGPUConstants.scanBlockElements
        radixHistogram = try make("radixHistogram", histogramPadded * 4)
        radixHistogramScan = try make("radixHistogramScan", histogramPadded * 4)

        // Levels for the recursive scan. The longest thing ever scanned is
        // the padded histogram or the padded splat array, whichever is bigger.
        var sums: [MTLBuffer] = []
        var scanned: [MTLBuffer] = []
        var levelCount = Swift.max(histogramPadded, paddedSplats)
        var level = 0
        while levelCount > 1 {
            let blocks = (levelCount + TrainerGPUConstants.scanBlockElements - 1)
                / TrainerGPUConstants.scanBlockElements
            let padded = ((blocks + TrainerGPUConstants.scanBlockElements - 1)
                / TrainerGPUConstants.scanBlockElements) * TrainerGPUConstants.scanBlockElements
            sums.append(try make("scanSums\(level)", padded * 4))
            scanned.append(try make("scanSumsScanned\(level)", padded * 4))
            levelCount = blocks
            level += 1
            if level > 6 { break }   // 1024^6 is unreachable; the guard is a stop, not a limit
        }
        if sums.isEmpty {
            sums.append(try make("scanSums0", TrainerGPUConstants.scanBlockElements * 4))
            scanned.append(try make("scanSumsScanned0", TrainerGPUConstants.scanBlockElements * 4))
        }
        scanBlockSums = sums
        scanBlockSumsScanned = scanned

        renderColor = try make("renderColor", px * 3 * 4)
        renderAlpha = try make("renderAlpha", px * 4)
        renderDepth = try make("renderDepth", px * 4)
        renderTFinal = try make("renderTFinal", px * 4)
        renderNContrib = try make("renderNContrib", px * 4)

        gtColor = try make("gtColor", px * 3 * 4)
        bgColor = try make("bgColor", px * 3 * 4)
        composited = try make("composited", px * 3 * 4)
        gradFinal = try make("gradFinal", px * 3 * 4)
        gradSplatColor = try make("gradSplatColor", px * 3 * 4)
        gradDepthRend = try make("gradDepthRend", px * 4)
        gradTFinal = try make("gradTFinal", px * 4)
        unknownMask = try make("unknownMask", px * 4)

        let planeBytes = TrainerGPUConstants.ssimPlaneCount * px * 4
        ssimSrc = try make("ssimSrc", planeBytes)
        ssimMid = try make("ssimMid", planeBytes)
        ssimTmp = try make("ssimTmp", planeBytes)

        lossAccum = try make("lossAccum", 16)
        exposureGrad = try make("exposureGrad", 16)
        cameraGrad = try make("cameraGrad", 32)
        depthSamples = try make(
            "depthSamples",
            self.depthSampleCapacity * MemoryLayout<TrainerDepthSample>.stride
        )

        recomputeResidentBytes()
    }

    // MARK: - Accounting

    private func recomputeResidentBytes() {
        var total = 0
        for buffer in allBuffers { total += buffer.allocatedSize }
        residentBytes = UInt64(total)
    }

    private var allBuffers: [MTLBuffer] {
        var list: [MTLBuffer] = [
            splats, sh, stats, draws, samplingTopK, tilesTouched, offsets,
            splatGrad, shGrad, adamM, adamV, shAdamM, shAdamV,
            gradMean2D, gradConic, gradColor, gradOpacity, centers,
            keysA, keysB, valuesA, valuesB, tileRanges,
            radixHistogram, radixHistogramScan,
            renderColor, renderAlpha, renderDepth, renderTFinal, renderNContrib,
            gtColor, bgColor, composited, gradFinal, gradSplatColor,
            gradDepthRend, gradTFinal, unknownMask,
            ssimSrc, ssimMid, ssimTmp,
            lossAccum, exposureGrad, cameraGrad, depthSamples
        ]
        list.append(contentsOf: scanBlockSums)
        list.append(contentsOf: scanBlockSumsScanned)
        return list
    }

    /// What ONE Gaussian costs across every per-Gaussian buffer, measured from
    /// the layouts rather than guessed. This is the single source of truth for
    /// the figure: `TrainingBudget.recommended` (Core) and
    /// `ProcessingBudgetPlanner` (Pipeline) both call it rather than keeping
    /// their own copy, and it is what the budget governor steps down against.
    /// They used to hardcode 200, which was low by roughly a factor of three
    /// and quietly stopped both the memory-derived cap and the "this will be
    /// tight" warning from ever doing anything.
    static func bytesPerSplat(shCoefficientCount: Int) -> Int {
        let shFloats = Swift.max(shCoefficientCount, 1) * 3
        return MemoryLayout<TrainerSplat>.stride           // parameters
            + MemoryLayout<TrainerSplatStats>.stride       // statistics
            + MemoryLayout<TrainerSplatDraw>.stride        // per-frame projection
            + MemoryLayout<TrainerSamplingTopK>.stride     // Mip-Splatting rates
            + MemoryLayout<TrainerSplatGrad>.stride * 3    // gradient + Adam m + v
            + shFloats * 4 * 4                             // sh + grad + m + v
            + 4 * 2                                        // tilesTouched, offsets
            + 4 * (2 + 3 + 3 + 1)                          // mean2D, conic, colour, opacity
            + 4 * 3                                        // extracted centres
    }

    /// What ONE render pixel costs across every per-pixel buffer.
    static func bytesPerPixel() -> Int {
        let perPixelFloats =
            3   // renderColor
            + 1 // renderAlpha
            + 1 // renderDepth
            + 1 // renderTFinal
            + 1 // renderNContrib (uint, same width)
            + 3 // gtColor
            + 3 // bgColor
            + 3 // composited
            + 3 // gradFinal
            + 3 // gradSplatColor
            + 1 // gradDepthRend
            + 1 // gradTFinal
            + 1 // unknownMask
        let ssimFloats = TrainerGPUConstants.ssimPlaneCount * 3   // src, mid, tmp
        return (perPixelFloats + ssimFloats) * 4
    }

    // MARK: - Reshaping
    //
    // Both of these throw away and re-make the affected allocations. That is
    // deliberate: `MTLBuffer` cannot be resized, and a pool that keeps the old
    // one alive "in case" is exactly how a phone runs out of memory while a
    // log line claims the budget was lowered.

    /// Rebuilds every per-Gaussian buffer at a new capacity, preserving the
    /// first `keepCount` Gaussians' parameters, SH, statistics and Adam
    /// moments. Everything else is transient and is simply re-zeroed.
    func resizeSplatCapacity(to newCapacity: Int, keeping keepCount: Int) throws {
        let target = Swift.max(newCapacity, 1)
        guard target != splatCapacity else { return }

        let previous = splatCapacity
        let keep = Swift.max(0, Swift.min(keepCount, Swift.min(splatCapacity, target)))
        let shPerSplat = shFloatsPerSplat

        let oldSplats = splats.readArray(TrainerSplat.self, count: keep)
        let oldStats = stats.readArray(TrainerSplatStats.self, count: keep)
        let oldM = adamM.readArray(TrainerSplatGrad.self, count: keep)
        let oldV = adamV.readArray(TrainerSplatGrad.self, count: keep)
        let oldSH = sh.readArray(Float.self, count: keep * shPerSplat)
        let oldSHM = shAdamM.readArray(Float.self, count: keep * shPerSplat)
        let oldSHV = shAdamV.readArray(Float.self, count: keep * shPerSplat)
        let oldTopK = samplingTopK.readArray(TrainerSamplingTopK.self, count: keep)

        let rebuilt = try TrainerResources(
            device: device,
            splatCapacity: target,
            renderSize: renderSize,
            shCoefficientCount: shCoefficientCount,
            depthSampleCapacity: depthSampleCapacity,
            instanceCapacity: instanceCapacity
        )

        rebuilt.splats.writeArray(oldSplats)
        rebuilt.stats.writeArray(oldStats)
        rebuilt.adamM.writeArray(oldM)
        rebuilt.adamV.writeArray(oldV)
        rebuilt.sh.writeArray(oldSH)
        rebuilt.shAdamM.writeArray(oldSHM)
        rebuilt.shAdamV.writeArray(oldSHV)
        rebuilt.samplingTopK.writeArray(oldTopK)

        adopt(rebuilt)
        TrainerLog.gpu.info(
            "Splat capacity is now \(target) (was \(previous)), keeping \(keep)"
        )
    }

    /// Rebuilds the per-pixel buffers at a new render size. Nothing per-pixel
    /// survives an iteration boundary, so nothing is copied.
    func resizeRenderSize(to newSize: TrainerRenderSize) throws {
        guard newSize != renderSize, newSize.pixelCount > 0 else { return }
        let rebuilt = try TrainerResources(
            device: device,
            splatCapacity: splatCapacity,
            renderSize: newSize,
            shCoefficientCount: shCoefficientCount,
            depthSampleCapacity: depthSampleCapacity,
            instanceCapacity: instanceCapacity
        )
        let shPerSplat = shFloatsPerSplat
        rebuilt.splats.writeArray(splats.readArray(TrainerSplat.self, count: splatCapacity))
        rebuilt.stats.writeArray(stats.readArray(TrainerSplatStats.self, count: splatCapacity))
        rebuilt.adamM.writeArray(adamM.readArray(TrainerSplatGrad.self, count: splatCapacity))
        rebuilt.adamV.writeArray(adamV.readArray(TrainerSplatGrad.self, count: splatCapacity))
        rebuilt.sh.writeArray(sh.readArray(Float.self, count: splatCapacity * shPerSplat))
        rebuilt.shAdamM.writeArray(shAdamM.readArray(Float.self, count: splatCapacity * shPerSplat))
        rebuilt.shAdamV.writeArray(shAdamV.readArray(Float.self, count: splatCapacity * shPerSplat))
        rebuilt.samplingTopK.writeArray(
            samplingTopK.readArray(TrainerSamplingTopK.self, count: splatCapacity)
        )
        adopt(rebuilt)
        TrainerLog.gpu.info("Render size is now \(newSize.width)x\(newSize.height)")
    }

    /// Grows the sort buffers after a frame needed more (Gaussian, tile) pairs
    /// than fitted. Nothing in them survives a frame, so nothing is copied.
    func growInstanceCapacity(to newCapacity: Int) throws {
        let target = Swift.max(newCapacity, instanceCapacity)
        guard target > instanceCapacity else { return }
        let rebuilt = try TrainerResources(
            device: device,
            splatCapacity: splatCapacity,
            renderSize: renderSize,
            shCoefficientCount: shCoefficientCount,
            depthSampleCapacity: depthSampleCapacity,
            instanceCapacity: target
        )
        let shPerSplat = shFloatsPerSplat
        rebuilt.splats.writeArray(splats.readArray(TrainerSplat.self, count: splatCapacity))
        rebuilt.stats.writeArray(stats.readArray(TrainerSplatStats.self, count: splatCapacity))
        rebuilt.adamM.writeArray(adamM.readArray(TrainerSplatGrad.self, count: splatCapacity))
        rebuilt.adamV.writeArray(adamV.readArray(TrainerSplatGrad.self, count: splatCapacity))
        rebuilt.sh.writeArray(sh.readArray(Float.self, count: splatCapacity * shPerSplat))
        rebuilt.shAdamM.writeArray(shAdamM.readArray(Float.self, count: splatCapacity * shPerSplat))
        rebuilt.shAdamV.writeArray(shAdamV.readArray(Float.self, count: splatCapacity * shPerSplat))
        rebuilt.samplingTopK.writeArray(
            samplingTopK.readArray(TrainerSamplingTopK.self, count: splatCapacity)
        )
        adopt(rebuilt)
        TrainerLog.gpu.info("Tile instance capacity grown to \(target)")
    }

    private func adopt(_ other: TrainerResources) {
        splatCapacity = other.splatCapacity
        instanceCapacity = other.instanceCapacity
        renderSize = other.renderSize
        shCoefficientCount = other.shCoefficientCount
        depthSampleCapacity = other.depthSampleCapacity

        splats = other.splats
        sh = other.sh
        stats = other.stats
        draws = other.draws
        samplingTopK = other.samplingTopK
        tilesTouched = other.tilesTouched
        offsets = other.offsets
        splatGrad = other.splatGrad
        shGrad = other.shGrad
        adamM = other.adamM
        adamV = other.adamV
        shAdamM = other.shAdamM
        shAdamV = other.shAdamV
        gradMean2D = other.gradMean2D
        gradConic = other.gradConic
        gradColor = other.gradColor
        gradOpacity = other.gradOpacity
        centers = other.centers
        keysA = other.keysA
        keysB = other.keysB
        valuesA = other.valuesA
        valuesB = other.valuesB
        tileRanges = other.tileRanges
        radixHistogram = other.radixHistogram
        radixHistogramScan = other.radixHistogramScan
        scanBlockSums = other.scanBlockSums
        scanBlockSumsScanned = other.scanBlockSumsScanned
        renderColor = other.renderColor
        renderAlpha = other.renderAlpha
        renderDepth = other.renderDepth
        renderTFinal = other.renderTFinal
        renderNContrib = other.renderNContrib
        gtColor = other.gtColor
        bgColor = other.bgColor
        composited = other.composited
        gradFinal = other.gradFinal
        gradSplatColor = other.gradSplatColor
        gradDepthRend = other.gradDepthRend
        gradTFinal = other.gradTFinal
        unknownMask = other.unknownMask
        ssimSrc = other.ssimSrc
        ssimMid = other.ssimMid
        ssimTmp = other.ssimTmp
        lossAccum = other.lossAccum
        exposureGrad = other.exposureGrad
        cameraGrad = other.cameraGrad
        depthSamples = other.depthSamples

        recomputeResidentBytes()
    }
}
