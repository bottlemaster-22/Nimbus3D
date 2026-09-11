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

    /// The four screen-space gradients of every splat, interleaved into one
    /// 64-byte record each. Was four separate buffers, which meant one pixel's
    /// contribution to one Gaussian touched four cache lines in four
    /// allocations. See TrainerSplatGrad2DAtomic in TrainerShaders.metal.
    private(set) var splatGrad2D: MTLBuffer
    /// The background cubemap, 6 faces of faceSize^2 linear-RGB texels, for
    /// `trainer_background`. 72 KB at the default face size of 32.
    private(set) var bgCubemap: MTLBuffer
    /// The 32-byte subset of `draws` that the sort and both rasterisers read.
    /// Halves the largest per-iteration DRAM item; see TrainerSplatRaster.
    private(set) var raster: MTLBuffer

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
    /// Build 314: 256 floats, `Float(i) / 255` computed in Swift, so the
    /// loss kernel turns a ground-truth byte into the decoder's exact float.
    let gtLevels: MTLBuffer
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

    // OVERLAPPED ITERATIONS (build 292). The three buffers the CPU writes a
    // frame's inputs into have a twin, so the next iteration's inputs can be
    // uploaded while the previous iteration's command buffer B is still
    // reading its own. `inputSlot` picks which pair every write and every
    // binding uses; the trainer flips it only between steps.
    private(set) var gtColorAlt: MTLBuffer
    private(set) var bgCubemapAlt: MTLBuffer
    private(set) var depthSamplesAlt: MTLBuffer
    var inputSlot: Int = 0
    var gtColorIn: MTLBuffer { inputSlot == 0 ? gtColor : gtColorAlt }
    var bgCubemapIn: MTLBuffer { inputSlot == 0 ? bgCubemap : bgCubemapAlt }
    var depthSamplesIn: MTLBuffer { inputSlot == 0 ? depthSamples : depthSamplesAlt }
    /// 2 x 64 bytes: per input slot, the loss (float 0), the exposure gradient
    /// (floats 4..5) and the camera gradient (floats 8..13), copied at the end
    /// of command buffer B so the CPU can read them after the NEXT iteration's
    /// buffer A has already cleared the originals.
    private(set) var readbackStaging: MTLBuffer

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

        // IF YOU ADD A BUFFER BELOW, ADD IT TO THE MATCHING RESHAPE METHOD.
        //
        // `resizeSplatCapacity`, `resizeRenderSize` and `growInstanceCapacity`
        // each rebuild only the allocations sized from the ONE dimension they
        // change, so each of them carries its own transcribed copy of the sizes
        // below rather than re-entering this initialiser. A new allocation that
        // depends on `n` or `shFloats`, on `px` or `tiles`, or on `inst`, and
        // that is added only here, will silently keep its old capacity across
        // the corresponding resize. That reads as out-of-range Gaussians or a
        // short buffer at the far end of a shader rather than as a crash, which
        // is the expensive kind of wrong.
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
        raster = try make("raster", n * MemoryLayout<TrainerSplatRaster>.stride)
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

        // 64 bytes each: nine gradient floats and seven of padding, so one
        // splat's record is exactly one cache line and neighbours never share.
        // Costs 28 bytes per splat over the four buffers it replaces, about
        // 8 MB at the 300,000 cap.
        splatGrad2D = try make("splatGrad2D", n * 64)
        // 6 faces, 32x32, three floats each. Fixed size: it does not scale
        // with splats or pixels, so it is allocated once and never resized.
        bgCubemap = try make("bgCubemap", 6 * 32 * 32 * 3 * 4)
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

        gtColor = try make("gtColor", px * 3)
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
        gtColorAlt = try make("gtColorAlt", px * 3)
        bgCubemapAlt = try make("bgCubemapAlt", 6 * 32 * 32 * 3 * 4)
        depthSamplesAlt = try make(
            "depthSamplesAlt",
            self.depthSampleCapacity * MemoryLayout<TrainerDepthSample>.stride
        )
        readbackStaging = try make("readbackStaging", 128)
        gtLevels = try make("gtLevels", 256 * 4)

        recomputeResidentBytes()
        _ = gtLevels.writeArray((0..<256).map { Float($0) / 255 })
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
            splatGrad2D, bgCubemap, raster, centers,
            keysA, keysB, valuesA, valuesB, tileRanges,
            radixHistogram, radixHistogramScan,
            renderColor, renderAlpha, renderDepth, renderTFinal, renderNContrib,
            gtColor, bgColor, composited, gradFinal, gradSplatColor,
            gradDepthRend, gradTFinal, unknownMask,
            ssimSrc, ssimMid, ssimTmp,
            lossAccum, exposureGrad, cameraGrad, depthSamples,
            gtColorAlt, bgCubemapAlt, depthSamplesAlt, readbackStaging, gtLevels
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
            + 3 // bgColor
            + 3 // composited
            + 3 // gradFinal
            + 3 // gradSplatColor
            + 1 // gradDepthRend
            + 1 // gradTFinal
            + 1 // unknownMask
        let ssimFloats = TrainerGPUConstants.ssimPlaneCount * 3   // src, mid, tmp
        // gtColor and gtColorAlt: three BYTES per pixel each (build 314).
        return (perPixelFloats + ssimFloats) * 4 + 3 * 2
    }

    // MARK: - Allocation
    //
    // `init` builds every buffer through a local `make` closure, which is the
    // reason every reshape below used to route through a whole second
    // `TrainerResources`: making ONE buffer was not something this class could
    // do outside of `init`. These helpers are that missing ability, and they
    // are what lets a reshape replace buffers one at a time instead of building
    // a duplicate of everything the trainer owns.

    /// One shared buffer, labelled and zeroed. Deliberately identical to the
    /// local `make` in `init`, and the two have to stay identical: a buffer
    /// that is zeroed on one path and not the other shows up as noise in a
    /// trained scene rather than as a crash, which is the hardest kind of bug
    /// to trace back to where it came from.
    private func makeBuffer(_ name: String, _ bytes: Int) throws -> MTLBuffer {
        let want = Swift.max(bytes, 16)
        guard let buffer = device.makeBuffer(length: want, options: .storageModeShared) else {
            throw TrainerError.allocationFailed(name: name, bytes: want)
        }
        buffer.label = "trainer.\(name)"
        memset(buffer.contents(), 0, want)
        return buffer
    }

    /// A sixteen-byte stand-in, assigned to a property purely so that the
    /// assignment RELEASES whatever that property was holding. The buffer
    /// properties are not optional, so there is no `nil` to assign, and holding
    /// a large old allocation alive across the `makeBuffer` call for its
    /// replacement is exactly the transient double this whole section exists to
    /// remove.
    ///
    /// A property holding the placeholder is not safe to encode against.
    /// Nothing can: every caller of the reshaping methods below runs on the
    /// trainer's own thread between command buffers, after a
    /// `waitUntilCompleted` has already returned, and `TrainerGPU` reads
    /// `resources.<buffer>` at encode time rather than holding references of
    /// its own, so there is no observer of the intermediate state.
    private func makePlaceholder() throws -> MTLBuffer {
        return try makeBuffer("reshapePlaceholder", 16)
    }

    /// Copies the first `bytes` bytes of one buffer into another.
    ///
    /// Every allocation here is `.storageModeShared`, so `contents()` on both
    /// sides is a pointer into the same physical memory the GPU reads, and a
    /// `memcpy` between them IS the copy. The reshaping code used to go out
    /// through `readArray` into a Swift array and back in through `writeArray`,
    /// which allocates a complete host-side duplicate of the data before a
    /// single byte of it reaches the destination. Across the eight preserved
    /// per-Gaussian buffers at a real capacity that is on the order of a
    /// hundred megabytes of transient heap, taken at the exact moment the
    /// budget governor has decided memory is short.
    ///
    /// A blit encoder would copy no faster on unified memory and would need a
    /// command buffer and a wait on a path that deliberately runs with no GPU
    /// work in flight, so it would be more code and a new synchronisation point
    /// for the same bytes.
    ///
    /// The length is clamped to both allocations, so a shrink copies what fits
    /// and a grow leaves the tail exactly as `makeBuffer` zeroed it.
    private static func copyFront(from source: MTLBuffer, to destination: MTLBuffer, bytes: Int) {
        let n = Swift.min(bytes, Swift.min(source.length, destination.length))
        guard n > 0 else { return }
        memcpy(destination.contents(), source.contents(), n)
    }

    /// Allocates the replacement for one buffer and fills it from the old one.
    /// The caller assigns the result straight back onto the property it came
    /// from, and that assignment releases the old allocation, so exactly one
    /// old/new pair is resident at a time rather than two complete sets.
    private func replacing(
        _ old: MTLBuffer,
        _ name: String,
        bytes: Int,
        copying copyBytes: Int
    ) throws -> MTLBuffer {
        let fresh = try makeBuffer(name, bytes)
        Self.copyFront(from: old, to: fresh, bytes: copyBytes)
        return fresh
    }

    /// The scan kernels write `output[idx]` across a whole block whether or not
    /// the tail exists, so anything they scan is allocated up to a block
    /// boundary. Same arithmetic `init` uses for `paddedSplats`.
    private static func paddedToScanBlock(_ count: Int) -> Int {
        let block = TrainerGPUConstants.scanBlockElements
        return ((count + block - 1) / block) * block
    }

    /// Padded radix-histogram length for a sort capacity. Same arithmetic
    /// `init` uses for `histogramPadded`.
    private static func paddedHistogramLength(instanceCapacity inst: Int) -> Int {
        let maxSortBlocks = (inst + TrainerGPUConstants.scanBlockElements - 1)
            / TrainerGPUConstants.scanBlockElements
        let histogramEntries = TrainerGPUConstants.radixBins * Swift.max(maxSortBlocks, 1)
        return paddedToScanBlock(histogramEntries)
    }

    /// Rebuilds the recursive scan scratch. Level zero is sized from the LONGER
    /// of the padded histogram and the padded splat array, so it has to be
    /// rebuilt whenever either of those two moves. That is why
    /// `resizeSplatCapacity` and `growInstanceCapacity` both call this and
    /// `resizeRenderSize` does not: the pixel grid feeds neither length.
    ///
    /// The loop is the one in `init`, kept in step with it on purpose.
    ///
    /// Unlike everything else in this section, the old buffers are held until
    /// the new ones are all built, and that is deliberate rather than an
    /// oversight. Two levels of a few kilobytes each is the entire allocation,
    /// so releasing first would free nothing worth measuring, and it would be
    /// the one place a failed allocation could leave this object holding LESS
    /// than it started with instead of a placeholder it can be seen to hold.
    private func rebuildScanLevels(paddedHistogram: Int, paddedSplats: Int) throws {
        var sums: [MTLBuffer] = []
        var scanned: [MTLBuffer] = []
        var levelCount = Swift.max(paddedHistogram, paddedSplats)
        var level = 0
        while levelCount > 1 {
            let blocks = (levelCount + TrainerGPUConstants.scanBlockElements - 1)
                / TrainerGPUConstants.scanBlockElements
            let padded = Self.paddedToScanBlock(blocks)
            sums.append(try makeBuffer("scanSums\(level)", padded * 4))
            scanned.append(try makeBuffer("scanSumsScanned\(level)", padded * 4))
            levelCount = blocks
            level += 1
            if level > 6 { break }   // 1024^6 is unreachable; the guard is a stop, not a limit
        }
        if sums.isEmpty {
            sums.append(try makeBuffer("scanSums0", TrainerGPUConstants.scanBlockElements * 4))
            scanned.append(
                try makeBuffer("scanSumsScanned0", TrainerGPUConstants.scanBlockElements * 4)
            )
        }
        scanBlockSums = sums
        scanBlockSumsScanned = scanned
    }

    // MARK: - Reshaping
    //
    // All three of these throw away and re-make the affected allocations. That
    // is deliberate: `MTLBuffer` cannot be resized, and a pool that keeps the
    // old one alive "in case" is exactly how a phone runs out of memory while a
    // log line claims the budget was lowered.
    //
    // They used to break that rule in the worst possible place. Each one built a
    // second complete `TrainerResources` while the first was still alive, so the
    // moment the budget governor decided memory was short was the moment
    // resident memory doubled, with about a hundred megabytes of Swift arrays
    // holding a host-side copy of eight per-Gaussian buffers on the way past.
    // Two of the three did not need to copy anything at all: the pixel grid and
    // the sort capacity do not size a single per-Gaussian buffer, so those
    // copies were writing the same numbers into a different allocation at the
    // highest available price.
    //
    // Each method now touches ONLY the buffers whose size actually depends on
    // the dimension that changed, releases each old allocation before or as its
    // replacement is filled, and copies between two shared buffers directly.

    /// Rebuilds every per-Gaussian buffer at a new capacity, preserving the
    /// first `keepCount` Gaussians' parameters, SH, statistics and Adam
    /// moments. Everything else is transient and is simply re-zeroed.
    ///
    /// One buffer at a time, because of WHEN this runs. The budget governor
    /// calls it after `degradeForMemory` has decided the process is close to
    /// its limit. The previous version answered that by constructing a second
    /// complete `TrainerResources` while the first was still alive, which
    /// duplicated the per-pixel and sort buffers this call does not even
    /// change, and pushed eight per-Gaussian buffers out to Swift arrays and
    /// straight back on top of that: roughly two full resource sets plus a
    /// hundred megabytes of host heap, at the exact moment memory was scarce.
    ///
    /// The order below is chosen for the peak rather than for readability. The
    /// ten transient buffers are released FIRST and are not reallocated until
    /// the very end, so the eight that carry state forward are allocated into
    /// the headroom that release just freed and never have to fit alongside a
    /// second copy of anything else. Peak over the steady state is one
    /// replacement buffer, and while the preserved eight are being rebuilt it
    /// is BELOW the steady state.
    func resizeSplatCapacity(to newCapacity: Int, keeping keepCount: Int) throws {
        let target = Swift.max(newCapacity, 1)
        guard target != splatCapacity else { return }

        let previous = splatCapacity
        let keep = Swift.max(0, Swift.min(keepCount, Swift.min(splatCapacity, target)))
        let shPerSplat = shFloatsPerSplat

        let n = target
        let shFloats = n * shCoefficientCount * 3
        let paddedSplats = Self.paddedToScanBlock(n)
        let floatStride = MemoryLayout<Float>.stride
        let splatStride = MemoryLayout<TrainerSplat>.stride
        let statsStride = MemoryLayout<TrainerSplatStats>.stride
        let gradStride = MemoryLayout<TrainerSplatGrad>.stride
        let topKStride = MemoryLayout<TrainerSamplingTopK>.stride
        let keptSHBytes = keep * shPerSplat * floatStride

        // Release the ten buffers whose contents cannot outlive an iteration,
        // before a single byte is allocated.
        //
        // Three of them, `splatGrad`, `shGrad` and `splatGrad2D`, start every
        // step at zero, cleared by trainer_preprocess_backward for exactly the
        // rows anything later reads (a fresh buffer from makeBuffer is zero). The other four are NOT cleared there and do
        // not need to be, which is worth writing down because the obvious
        // reading of that function is that it covers everything transient:
        // `draws` and `tilesTouched` are written for every Gaussian by
        // `trainer_preprocess`, `offsets` by the exclusive scan over them, and
        // `centers` by `trainer_extract_centers`, each of which runs before
        // anything reads the buffer it fills. Either way nothing downstream can
        // observe what was in these ten when this method was called.
        let placeholder = try makePlaceholder()
        draws = placeholder
        raster = placeholder
        tilesTouched = placeholder
        offsets = placeholder
        splatGrad = placeholder
        shGrad = placeholder
        splatGrad2D = placeholder
        centers = placeholder

        // The eight that carry state across the resize, allocated into the
        // headroom the release above just freed. Each replacement is filled
        // from its predecessor by a direct `memcpy` between two shared
        // allocations, and the assignment releases the predecessor, so only one
        // old/new pair is ever live.
        splats = try replacing(
            splats, "splats", bytes: n * splatStride, copying: keep * splatStride
        )
        stats = try replacing(
            stats, "stats", bytes: n * statsStride, copying: keep * statsStride
        )
        samplingTopK = try replacing(
            samplingTopK, "samplingTopK", bytes: n * topKStride, copying: keep * topKStride
        )
        adamM = try replacing(
            adamM, "adamM", bytes: n * gradStride, copying: keep * gradStride
        )
        adamV = try replacing(
            adamV, "adamV", bytes: n * gradStride, copying: keep * gradStride
        )
        sh = try replacing(
            sh, "sh", bytes: shFloats * floatStride, copying: keptSHBytes
        )
        shAdamM = try replacing(
            shAdamM, "shAdamM", bytes: shFloats * floatStride, copying: keptSHBytes
        )
        shAdamV = try replacing(
            shAdamV, "shAdamV", bytes: shFloats * floatStride, copying: keptSHBytes
        )

        // Only now the transient ten, at the new capacity. Last, so that none of
        // the eight above ever had to be allocated around them.
        draws = try makeBuffer("draws", n * MemoryLayout<TrainerSplatDraw>.stride)
        raster = try makeBuffer("raster", n * MemoryLayout<TrainerSplatRaster>.stride)
        tilesTouched = try makeBuffer("tilesTouched", paddedSplats * 4)
        offsets = try makeBuffer("offsets", paddedSplats * 4)
        splatGrad = try makeBuffer("splatGrad", n * gradStride)
        shGrad = try makeBuffer("shGrad", shFloats * floatStride)
        splatGrad2D = try makeBuffer("splatGrad2D", n * 64)
        centers = try makeBuffer("centers", n * 3 * 4)

        // Level zero of the scan scratch is sized from the padded splat array,
        // so it moves with the capacity even though the sort capacity has not.
        try rebuildScanLevels(
            paddedHistogram: Self.paddedHistogramLength(instanceCapacity: instanceCapacity),
            paddedSplats: paddedSplats
        )

        splatCapacity = target
        recomputeResidentBytes()
        TrainerLog.gpu.info(
            "Splat capacity is now \(target) (was \(previous)), keeping \(keep)"
        )
    }

    /// Rebuilds the per-pixel buffers at a new render size. Nothing per-pixel
    /// survives an iteration boundary, so nothing is copied.
    ///
    /// `tileRanges` and the sixteen per-pixel buffers are the only allocations
    /// in this class sized from the render grid. The per-Gaussian buffers, the
    /// sort buffers and the scan scratch are not, so they are left exactly
    /// where they are. The version that built a second `TrainerResources`
    /// rebuilt all of them at their existing sizes and copied eight out to
    /// Swift arrays and back, which is the most expensive available way of
    /// putting the same numbers into a different allocation, and it doubled
    /// resident memory to do it. This is called by the budget governor when it
    /// lowers the resolution to save memory, so that doubling landed on a
    /// process that had just been told it was short.
    ///
    /// Leaving the sort buffers alone is safe for the same reason they are
    /// never cleared between the eight passes of a single sort:
    /// `trainer_radix_histogram` STORES each bin rather than accumulating into
    /// it, and the keys and values are written and read over the same
    /// [0, instanceCount) range every frame.
    ///
    /// Each old allocation is released before its replacement is requested,
    /// which is safe because nothing here carries data forward and because this
    /// only ever runs between command buffers.
    func resizeRenderSize(to newSize: TrainerRenderSize) throws {
        guard newSize != renderSize, newSize.pixelCount > 0 else { return }

        let px = Swift.max(newSize.pixelCount, 1)
        let tiles = Swift.max(newSize.tileCount, 1)
        let planeBytes = TrainerGPUConstants.ssimPlaneCount * px * 4

        let placeholder = try makePlaceholder()
        tileRanges = placeholder
        renderColor = placeholder
        renderAlpha = placeholder
        renderDepth = placeholder
        renderTFinal = placeholder
        renderNContrib = placeholder
        gtColor = placeholder
        gtColorAlt = placeholder
        bgColor = placeholder
        composited = placeholder
        gradFinal = placeholder
        gradSplatColor = placeholder
        gradDepthRend = placeholder
        gradTFinal = placeholder
        unknownMask = placeholder
        ssimSrc = placeholder
        ssimMid = placeholder
        ssimTmp = placeholder

        tileRanges = try makeBuffer("tileRanges", tiles * 2 * 4)

        renderColor = try makeBuffer("renderColor", px * 3 * 4)
        renderAlpha = try makeBuffer("renderAlpha", px * 4)
        renderDepth = try makeBuffer("renderDepth", px * 4)
        renderTFinal = try makeBuffer("renderTFinal", px * 4)
        renderNContrib = try makeBuffer("renderNContrib", px * 4)

        gtColor = try makeBuffer("gtColor", px * 3)
        gtColorAlt = try makeBuffer("gtColorAlt", px * 3)
        bgColor = try makeBuffer("bgColor", px * 3 * 4)
        composited = try makeBuffer("composited", px * 3 * 4)
        gradFinal = try makeBuffer("gradFinal", px * 3 * 4)
        gradSplatColor = try makeBuffer("gradSplatColor", px * 3 * 4)
        gradDepthRend = try makeBuffer("gradDepthRend", px * 4)
        gradTFinal = try makeBuffer("gradTFinal", px * 4)
        unknownMask = try makeBuffer("unknownMask", px * 4)

        ssimSrc = try makeBuffer("ssimSrc", planeBytes)
        ssimMid = try makeBuffer("ssimMid", planeBytes)
        ssimTmp = try makeBuffer("ssimTmp", planeBytes)

        renderSize = newSize
        recomputeResidentBytes()
        TrainerLog.gpu.info("Render size is now \(newSize.width)x\(newSize.height)")
    }

    /// Grows the sort buffers after a frame needed more (Gaussian, tile) pairs
    /// than fitted. Nothing in them survives a frame, so nothing is copied.
    ///
    /// The four sort arrays, the two histograms and the scan scratch are the
    /// only allocations sized from `instanceCapacity`. The per-Gaussian and
    /// per-pixel buffers are not, and this leaves them alone. The version that
    /// rebuilt the whole object turned a routine mid-frame growth of a few
    /// megabytes of sort keys into a brief doubling of every buffer the trainer
    /// owns, with eight of them pushed through host memory on the way, and this
    /// runs from inside `runIteration` rather than at a quiet moment.
    ///
    /// The frame that triggered this is abandoned and retried
    /// (`.grewTileBufferAndRetried`), so nothing that was mid-computation when
    /// it was called needs to survive.
    func growInstanceCapacity(to newCapacity: Int) throws {
        let target = Swift.max(newCapacity, instanceCapacity)
        guard target > instanceCapacity else { return }

        let inst = target
        let paddedHistogram = Self.paddedHistogramLength(instanceCapacity: inst)

        let placeholder = try makePlaceholder()
        keysA = placeholder
        keysB = placeholder
        valuesA = placeholder
        valuesB = placeholder
        radixHistogram = placeholder
        radixHistogramScan = placeholder

        keysA = try makeBuffer("keysA", inst * 4)
        keysB = try makeBuffer("keysB", inst * 4)
        valuesA = try makeBuffer("valuesA", inst * 4)
        valuesB = try makeBuffer("valuesB", inst * 4)
        radixHistogram = try makeBuffer("radixHistogram", paddedHistogram * 4)
        radixHistogramScan = try makeBuffer("radixHistogramScan", paddedHistogram * 4)

        // Level zero of the scan scratch is sized from the longer of the padded
        // histogram and the padded splat array, and the histogram just grew.
        try rebuildScanLevels(
            paddedHistogram: paddedHistogram,
            paddedSplats: Self.paddedToScanBlock(splatCapacity)
        )

        instanceCapacity = target
        recomputeResidentBytes()
        TrainerLog.gpu.info("Tile instance capacity grown to \(target)")
    }
}
