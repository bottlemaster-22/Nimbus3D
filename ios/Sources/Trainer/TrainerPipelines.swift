//
//  TrainerPipelines.swift
//  Trainer
//
//  COMPILES ALL 28 KERNELS AND ENCODES THEM.
//
//  Two halves:
//
//   * `TrainerPipelines` builds one `MTLComputePipelineState` per entry point
//     in TrainerShaders.metal, by name, at start-up. A name that is not in the
//     metallib is a named error before a single buffer is allocated, not a
//     mystery at iteration 400.
//
//   * `TrainerGPU` is the encode layer: one method per GPU stage, each binding
//     its buffers at the indices in `TrainerBind` (transcribed from the
//     shader's own `[[buffer(n)]]` attributes) and dispatching at the grid
//     shape that kernel actually requires.
//
//  THREE DISPATCH SHAPES, and they are not interchangeable:
//
//   1. Element-parallel kernels guard on `gid >= count` and are dispatched
//      with `dispatchThreads`, which lets the grid be any size. Apple GPUs
//      from A11 onwards support non-uniform threadgroups, and the LiDAR
//      devices this app runs on are all A12 or newer.
//   2. `trainer_scan_block`, `trainer_radix_histogram` and
//      `trainer_radix_scatter` index threadgroup memory by
//      `thread_position_in_threadgroup` and compute their element base from
//      `threadgroup_position_in_grid * 1024`. They MUST get exactly 256
//      threads per threadgroup and exactly `blockCount` threadgroups, so they
//      use `dispatchThreadgroups`.
//   3. The two rasteriser kernels are one threadgroup per 16x16 tile with one
//      thread per pixel, and their threadgroup arrays are 256 wide.
//

import Foundation
import Metal
import simd

// MARK: - Pipelines

final class TrainerPipelines {

    let fillUInt: MTLComputePipelineState
    let fillFloat: MTLComputePipelineState
    let resetDensifyStats: MTLComputePipelineState
    let scanBlock: MTLComputePipelineState
    let scanAdd: MTLComputePipelineState
    let radixHistogram: MTLComputePipelineState
    let radixScatter: MTLComputePipelineState
    /// The SIMD-prefix scatter (build 290), nil where it cannot be built (pre-
    /// Apple7, a SIMD group other than 32 wide) or failed to. Used only once the
    /// trainer's on-device calibration has shown it sorts identically and faster.
    let radixScatterSimdScan: MTLComputePipelineState?
    let preprocess: MTLComputePipelineState
    let duplicateKeys: MTLComputePipelineState
    let tileRanges: MTLComputePipelineState
    let rasterizeForward: MTLComputePipelineState
    let background: MTLComputePipelineState
    let lossPhotometric: MTLComputePipelineState
    let blurH: MTLComputePipelineState
    let blurV: MTLComputePipelineState
    let ssimStats: MTLComputePipelineState
    let lossDepth: MTLComputePipelineState
    let lossFinalize: MTLComputePipelineState
    let rasterizeBackward: MTLComputePipelineState
    /// The SIMD-summed backward (build 288), nil where it cannot be built
    /// (pre-Apple7) or failed to. Used only once the trainer's on-device
    /// calibration has shown it agrees with `rasterizeBackward` and is faster.
    let rasterizeBackwardSimdSum: MTLComputePipelineState?
    /// The two-pixel forward rasteriser (build 302), nil if it cannot run a
    /// 128-thread threadgroup or failed to build. Used only after the
    /// trainer calibration has shown it renders the same image, faster.
    let rasterizeForward2: MTLComputePipelineState?
    let preprocessBackward: MTLComputePipelineState
    let samplingRateUpdate: MTLComputePipelineState
    let filter3DFinalize: MTLComputePipelineState
    let regularizer: MTLComputePipelineState
    let adamSplat: MTLComputePipelineState
    let adamSH: MTLComputePipelineState
    let extractCenters: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {

        /// Builds a kernel that reads `kTrainerSimdReduce`. Specialisation
        /// happens here, so the branch the constant rules out is gone before
        /// the function is validated.
        func build(
            _ name: String, simdReduce: Bool
        ) throws -> MTLComputePipelineState {
            let values = MTLFunctionConstantValues()
            var flag = simdReduce
            values.setConstantValue(&flag, type: .bool, index: 0)
            // Set EXPLICITLY off (build 290). The kernel reads constant 1
            // through the is_function_constant_defined default pattern, so
            // leaving it unset is legal, but this pipeline must not depend on
            // that on a device no one has tested it on.
            var noSimdSum = false
            values.setConstantValue(&noSimdSum, type: .bool, index: 1)
            let function: MTLFunction
            do {
                function = try library.makeFunction(name: name, constantValues: values)
            } catch {
                throw TrainerError.missingKernel(name)
            }
            function.label = name
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                throw TrainerError.pipelineFailed(
                    kernel: name, reason: error.localizedDescription
                )
            }
        }

        /// The backward rasteriser with BOTH specialisations on: the simd_max
        /// batch bound and the SIMD-summed accumulation.
        func buildBackwardSimdSum(_ name: String) throws -> MTLComputePipelineState {
            let values = MTLFunctionConstantValues()
            var reduce = true
            values.setConstantValue(&reduce, type: .bool, index: 0)
            var sum = true
            values.setConstantValue(&sum, type: .bool, index: 1)
            let function = try library.makeFunction(name: name, constantValues: values)
            function.label = name + ".simdSum"
            return try device.makeComputePipelineState(function: function)
        }

        /// The radix scatter with constant 2 set explicitly either way.
        func buildRadixScatter(_ name: String, simdScan: Bool) throws -> MTLComputePipelineState {
            let values = MTLFunctionConstantValues()
            var scan = simdScan
            values.setConstantValue(&scan, type: .bool, index: 2)
            let function = try library.makeFunction(name: name, constantValues: values)
            function.label = simdScan ? name + ".simdScan" : name
            return try device.makeComputePipelineState(function: function)
        }

        func build(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw TrainerError.missingKernel(name)
            }
            function.label = name
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                throw TrainerError.pipelineFailed(
                    kernel: name,
                    reason: error.localizedDescription
                )
            }
        }

        fillUInt = try build(TrainerKernel.fillUInt)
        fillFloat = try build(TrainerKernel.fillFloat)
        resetDensifyStats = try build(TrainerKernel.resetDensifyStats)
        scanBlock = try build(TrainerKernel.scanBlock)
        scanAdd = try build(TrainerKernel.scanAdd)
        radixHistogram = try build(TrainerKernel.radixHistogram)
        do {
            radixScatter = try buildRadixScatter(TrainerKernel.radixScatter, simdScan: false)
        } catch {
            throw TrainerError.pipelineFailed(
                kernel: TrainerKernel.radixScatter, reason: error.localizedDescription
            )
        }
        let scatterSimd = device.supportsFamily(.apple7)
            ? (try? buildRadixScatter(TrainerKernel.radixScatter, simdScan: true)) : nil
        radixScatterSimdScan = (scatterSimd?.threadExecutionWidth == 32) ? scatterSimd : nil
        preprocess = try build(TrainerKernel.preprocess)
        duplicateKeys = try build(TrainerKernel.duplicateKeys)
        tileRanges = try build(TrainerKernel.tileRanges)
        rasterizeForward = try build(TrainerKernel.rasterizeForward)
        let forward2 = try? build(TrainerKernel.rasterizeForward2)
        let forward2Threads = TrainerGPUConstants.tileWidth * TrainerGPUConstants.tileHeight / 2
        rasterizeForward2 = (forward2?.maxTotalThreadsPerThreadgroup ?? 0) >= forward2Threads
            ? forward2 : nil
        background = try build(TrainerKernel.background)
        lossPhotometric = try build(TrainerKernel.lossPhotometric)
        blurH = try build(TrainerKernel.blurH)
        blurV = try build(TrainerKernel.blurV)
        ssimStats = try build(TrainerKernel.ssimStats)
        lossDepth = try build(TrainerKernel.lossDepth)
        lossFinalize = try build(TrainerKernel.lossFinalize)
        // Specialised: its batch bound uses simd_max, which is Apple7 (A14)
        // and later. Asking the device means an A12 or A13 builds the
        // unbounded variant instead of failing to build a pipeline.
        rasterizeBackward = try build(
            TrainerKernel.rasterizeBackward,
            simdReduce: device.supportsFamily(.apple7)
        )
        // Optional by design: a device or compiler that cannot build it
        // trains exactly as before rather than failing to start.
        rasterizeBackwardSimdSum = device.supportsFamily(.apple7)
            ? (try? buildBackwardSimdSum(TrainerKernel.rasterizeBackward)) : nil
        preprocessBackward = try build(TrainerKernel.preprocessBackward)
        samplingRateUpdate = try build(TrainerKernel.samplingRateUpdate)
        filter3DFinalize = try build(TrainerKernel.filter3DFinalize)
        regularizer = try build(TrainerKernel.regularizer)
        adamSplat = try build(TrainerKernel.adamSplat)
        adamSH = try build(TrainerKernel.adamSH)
        extractCenters = try build(TrainerKernel.extractCenters)

        // The block kernels are not merely "faster with 256 threads": their
        // threadgroup arrays are declared 256 wide and their element base is
        // `blockIndex * 1024`. A device that cannot run 256 threads in a
        // threadgroup for one of them would silently sort the wrong elements,
        // so it is refused with a sentence naming the kernel.
        let blockThreads = TrainerGPUConstants.scanThreads
        for (state, name) in [
            (scanBlock, TrainerKernel.scanBlock),
            (radixHistogram, TrainerKernel.radixHistogram),
            (radixScatter, TrainerKernel.radixScatter)
        ] where state.maxTotalThreadsPerThreadgroup < blockThreads {
            throw TrainerError.threadgroupTooSmall(
                kernel: name,
                available: state.maxTotalThreadsPerThreadgroup,
                needed: blockThreads
            )
        }

        let tileThreads = TrainerGPUConstants.tileArea
        for (state, name) in [
            (rasterizeForward, TrainerKernel.rasterizeForward),
            (rasterizeBackward, TrainerKernel.rasterizeBackward)
        ] where state.maxTotalThreadsPerThreadgroup < tileThreads {
            throw TrainerError.threadgroupTooSmall(
                kernel: name,
                available: state.maxTotalThreadsPerThreadgroup,
                needed: tileThreads
            )
        }
    }
}

// MARK: - Encoding

/// Everything that touches an `MTLComputeCommandEncoder`. Stateless apart from
/// the pipelines and the resources it encodes against, so the training loop
/// reads as a list of stages rather than a wall of `setBuffer` calls.
struct TrainerGPU {

    let pipelines: TrainerPipelines
    let resources: TrainerResources

    // MARK: Dispatch helpers

    private func dispatch1D(
        _ encoder: MTLComputeCommandEncoder,
        _ state: MTLComputePipelineState,
        count: Int
    ) {
        guard count > 0 else { return }
        let width = Swift.min(state.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Swift.max(width, 1), height: 1, depth: 1)
        )
    }

    private func dispatch2D(
        _ encoder: MTLComputeCommandEncoder,
        _ state: MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        guard width > 0, height > 0 else { return }
        let side = state.maxTotalThreadsPerThreadgroup >= 256 ? 16 : 8
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: side, height: side, depth: 1)
        )
    }

    private func dispatchBlocks(
        _ encoder: MTLComputeCommandEncoder,
        blockCount: Int
    ) {
        guard blockCount > 0 else { return }
        encoder.dispatchThreadgroups(
            MTLSize(width: blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: TrainerGPUConstants.scanThreads, height: 1, depth: 1
            )
        )
    }

    static func blockCount(for elements: Int) -> Int {
        guard elements > 0 else { return 0 }
        return (elements + TrainerGPUConstants.scanBlockElements - 1)
            / TrainerGPUConstants.scanBlockElements
    }

    // MARK: Fills

    func fillUInt(_ encoder: MTLComputeCommandEncoder, buffer: MTLBuffer, count: Int, value: UInt32) {
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.fillUInt)
        encoder.setBuffer(buffer, offset: 0, index: TrainerBind.FillUInt.target)
        var args = SIMD2<UInt32>(UInt32(count), value)
        encoder.setBytes(&args, length: MemoryLayout<SIMD2<UInt32>>.size, index: TrainerBind.FillUInt.args)
        dispatch1D(encoder, pipelines.fillUInt, count: count)
    }

    func fillFloat(_ encoder: MTLComputeCommandEncoder, buffer: MTLBuffer, count: Int, value: Float) {
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.fillFloat)
        encoder.setBuffer(buffer, offset: 0, index: TrainerBind.FillFloat.target)
        var n = UInt32(count)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: TrainerBind.FillFloat.count)
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<Float>.size, index: TrainerBind.FillFloat.value)
        dispatch1D(encoder, pipelines.fillFloat, count: count)
    }

    func resetDensifyStats(_ encoder: MTLComputeCommandEncoder, count: Int) {
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.resetDensifyStats)
        encoder.setBuffer(resources.stats, offset: 0, index: TrainerBind.ResetDensifyStats.stats)
        var n = UInt32(count)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: TrainerBind.ResetDensifyStats.count)
        dispatch1D(encoder, pipelines.resetDensifyStats, count: count)
    }

    // MARK: Exclusive prefix scan
    //
    // `trainer_scan_block` writes a per-block exclusive scan into `output` and
    // that block's TOTAL into `blockSums[blockIndex]`. Adding the exclusive
    // scan of the block sums back on turns the per-block scans into a global
    // one, and the block sums are themselves scanned by re-entering this
    // routine one level down. Two levels cover a million elements, three cover
    // a billion; `TrainerResources` allocated exactly as many levels as the
    // real capacity needs.

    func exclusiveScan(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        count: Int,
        level: Int = 0
    ) {
        guard count > 0 else { return }
        let blocks = TrainerGPU.blockCount(for: count)
        let haveLevel = level < resources.scanBlockSums.count
        let sums = haveLevel ? resources.scanBlockSums[level] : nil
        let sumsScanned = haveLevel ? resources.scanBlockSumsScanned[level] : nil

        encoder.setComputePipelineState(pipelines.scanBlock)
        encoder.setBuffer(input, offset: 0, index: TrainerBind.ScanBlock.input)
        encoder.setBuffer(output, offset: 0, index: TrainerBind.ScanBlock.output)
        encoder.setBuffer(sums, offset: 0, index: TrainerBind.ScanBlock.blockSums)
        var u = TrainerScanUniforms(count: UInt32(count), blockCount: UInt32(blocks))
        encoder.setBytes(
            &u, length: MemoryLayout<TrainerScanUniforms>.stride, index: TrainerBind.ScanBlock.uniforms
        )
        dispatchBlocks(encoder, blockCount: blocks)

        // One block means the per-block scan IS the global scan.
        guard blocks > 1 else { return }
        guard let sums, let sumsScanned else {
            // Unreachable: `TrainerResources` allocates one scratch level per
            // level this recursion can reach, computed from the same
            // capacities. Said out loud anyway, because the failure mode
            // without it is a scan that is correct within each 1024-element
            // block and wrong across them, which looks like a rasteriser bug
            // and is not one.
            TrainerLog.gpu.error(
                "Prefix scan ran out of scratch levels at level \(level) for \(count) elements; the tile ordering for this frame is not trustworthy"
            )
            return
        }

        exclusiveScan(encoder, input: sums, output: sumsScanned, count: blocks, level: level + 1)

        encoder.setComputePipelineState(pipelines.scanAdd)
        encoder.setBuffer(output, offset: 0, index: TrainerBind.ScanAdd.output)
        encoder.setBuffer(sumsScanned, offset: 0, index: TrainerBind.ScanAdd.blockOffsets)
        var addUniforms = TrainerScanUniforms(count: UInt32(count), blockCount: UInt32(blocks))
        encoder.setBytes(
            &addUniforms,
            length: MemoryLayout<TrainerScanUniforms>.stride,
            index: TrainerBind.ScanAdd.uniforms
        )
        dispatch1D(encoder, pipelines.scanAdd, count: count)
    }

    // MARK: Radix sort
    //
    // Six LSD passes of four bits over the 24-bit key
    // `(tileID << 12) | logDepth12`. Four bits and not eight because the
    // scatter's per-thread histogram is `bins * threads * 4` bytes of
    // threadgroup memory: 16 KB at 16 bins, 256 KB at 256 bins, and only the
    // first fits. Six passes is even, so the sorted result lands back in the
    // buffer it started in and the caller does not have to track parity.

    func radixSort(_ encoder: MTLComputeCommandEncoder, count: Int, simdScan: Bool = false) {
        guard count > 1 else { return }
        let scatter = (simdScan ? pipelines.radixScatterSimdScan : nil) ?? pipelines.radixScatter
        let blocks = TrainerGPU.blockCount(for: count)
        let histogramEntries = TrainerGPUConstants.radixBins * blocks

        var keysIn = resources.keysA
        var keysOut = resources.keysB
        var valuesIn = resources.valuesA
        var valuesOut = resources.valuesB

        for pass in 0..<TrainerGPUConstants.radixPasses {
            let shift = UInt32(pass * TrainerGPUConstants.radixBits)
            var u = TrainerRadixUniforms(
                count: UInt32(count), blockCount: UInt32(blocks), bitShift: shift
            )

            encoder.setComputePipelineState(pipelines.radixHistogram)
            encoder.setBuffer(keysIn, offset: 0, index: TrainerBind.RadixHistogram.keys)
            encoder.setBuffer(
                resources.radixHistogram, offset: 0, index: TrainerBind.RadixHistogram.histogram
            )
            encoder.setBytes(
                &u,
                length: MemoryLayout<TrainerRadixUniforms>.stride,
                index: TrainerBind.RadixHistogram.uniforms
            )
            dispatchBlocks(encoder, blockCount: blocks)

            exclusiveScan(
                encoder,
                input: resources.radixHistogram,
                output: resources.radixHistogramScan,
                count: histogramEntries
            )

            encoder.setComputePipelineState(scatter)
            encoder.setBuffer(keysIn, offset: 0, index: TrainerBind.RadixScatter.keysIn)
            encoder.setBuffer(valuesIn, offset: 0, index: TrainerBind.RadixScatter.valuesIn)
            encoder.setBuffer(keysOut, offset: 0, index: TrainerBind.RadixScatter.keysOut)
            encoder.setBuffer(valuesOut, offset: 0, index: TrainerBind.RadixScatter.valuesOut)
            encoder.setBuffer(
                resources.radixHistogramScan,
                offset: 0,
                index: TrainerBind.RadixScatter.histogramScan
            )
            encoder.setBytes(
                &u,
                length: MemoryLayout<TrainerRadixUniforms>.stride,
                index: TrainerBind.RadixScatter.uniforms
            )
            dispatchBlocks(encoder, blockCount: blocks)

            swap(&keysIn, &keysOut)
            swap(&valuesIn, &valuesOut)
        }
    }

    // MARK: Forward

    func preprocess(
        _ encoder: MTLComputeCommandEncoder,
        camera: inout TrainerCameraUniforms,
        splatCount: Int
    ) {
        guard splatCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.preprocess)
        encoder.setBuffer(resources.splats, offset: 0, index: TrainerBind.Preprocess.splats)
        encoder.setBuffer(resources.sh, offset: 0, index: TrainerBind.Preprocess.sh)
        encoder.setBuffer(resources.stats, offset: 0, index: TrainerBind.Preprocess.stats)
        encoder.setBuffer(resources.draws, offset: 0, index: TrainerBind.Preprocess.draws)
        encoder.setBuffer(resources.raster, offset: 0, index: TrainerBind.Preprocess.raster)
        encoder.setBuffer(
            resources.tilesTouched, offset: 0, index: TrainerBind.Preprocess.tilesTouched
        )
        encoder.setBytes(
            &camera,
            length: MemoryLayout<TrainerCameraUniforms>.stride,
            index: TrainerBind.Preprocess.camera
        )
        dispatch1D(encoder, pipelines.preprocess, count: splatCount)
    }

    func duplicateKeys(
        _ encoder: MTLComputeCommandEncoder,
        camera: inout TrainerCameraUniforms,
        splatCount: Int
    ) {
        guard splatCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.duplicateKeys)
        encoder.setBuffer(resources.raster, offset: 0, index: TrainerBind.DuplicateKeys.raster)
        encoder.setBuffer(
            resources.tilesTouched, offset: 0, index: TrainerBind.DuplicateKeys.tilesTouched
        )
        encoder.setBuffer(resources.offsets, offset: 0, index: TrainerBind.DuplicateKeys.offsets)
        encoder.setBuffer(resources.keysA, offset: 0, index: TrainerBind.DuplicateKeys.keys)
        encoder.setBuffer(resources.valuesA, offset: 0, index: TrainerBind.DuplicateKeys.values)
        encoder.setBytes(
            &camera,
            length: MemoryLayout<TrainerCameraUniforms>.stride,
            index: TrainerBind.DuplicateKeys.camera
        )
        var cap = UInt32(resources.instanceCapacity)
        encoder.setBytes(
            &cap, length: MemoryLayout<UInt32>.size, index: TrainerBind.DuplicateKeys.instanceCap
        )
        dispatch1D(encoder, pipelines.duplicateKeys, count: splatCount)
    }

    func tileRanges(_ encoder: MTLComputeCommandEncoder, instanceCount: Int) {
        // Every tile's range has to start at (0, 0) or a tile that received no
        // instances this frame would rasterise last frame's list.
        fillUInt(
            encoder,
            buffer: resources.tileRanges,
            count: resources.renderSize.tileCount * 2,
            value: 0
        )
        guard instanceCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.tileRanges)
        encoder.setBuffer(resources.keysA, offset: 0, index: TrainerBind.TileRanges.keys)
        encoder.setBuffer(resources.tileRanges, offset: 0, index: TrainerBind.TileRanges.tileRanges)
        var n = UInt32(instanceCount)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: TrainerBind.TileRanges.count)
        dispatch1D(encoder, pipelines.tileRanges, count: instanceCount)
    }

    func rasterizeForward(
        _ encoder: MTLComputeCommandEncoder,
        camera: inout TrainerCameraUniforms,
        twoPixels: Bool = false
    ) {
        let size = resources.renderSize
        guard size.tileCount > 0 else { return }
        // Same bindings for both; the two-pixel kernel runs half the threads.
        let pipeline = twoPixels ? pipelines.rasterizeForward2 : nil
        encoder.setComputePipelineState(pipeline ?? pipelines.rasterizeForward)
        let rows = pipeline == nil
            ? TrainerGPUConstants.tileHeight : TrainerGPUConstants.tileHeight / 2
        encoder.setBuffer(resources.valuesA, offset: 0, index: TrainerBind.RasterizeForward.values)
        encoder.setBuffer(
            resources.tileRanges, offset: 0, index: TrainerBind.RasterizeForward.tileRanges
        )
        encoder.setBuffer(resources.raster, offset: 0, index: TrainerBind.RasterizeForward.draws)
        encoder.setBuffer(
            resources.renderColor, offset: 0, index: TrainerBind.RasterizeForward.outColor
        )
        encoder.setBuffer(
            resources.renderAlpha, offset: 0, index: TrainerBind.RasterizeForward.outAlpha
        )
        encoder.setBuffer(
            resources.renderDepth, offset: 0, index: TrainerBind.RasterizeForward.outDepth
        )
        encoder.setBuffer(
            resources.renderTFinal, offset: 0, index: TrainerBind.RasterizeForward.outTFinal
        )
        encoder.setBuffer(
            resources.renderNContrib, offset: 0, index: TrainerBind.RasterizeForward.outNContrib
        )
        encoder.setBytes(
            &camera,
            length: MemoryLayout<TrainerCameraUniforms>.stride,
            index: TrainerBind.RasterizeForward.camera
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: size.tileCountX, height: size.tileCountY, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: TrainerGPUConstants.tileWidth,
                height: rows,
                depth: 1
            )
        )
    }

    // MARK: Losses

    func lossPhotometric(
        _ encoder: MTLComputeCommandEncoder,
        loss: inout TrainerLossUniforms
    ) {
        let px = resources.renderSize.pixelCount
        guard px > 0 else { return }
        encoder.setComputePipelineState(pipelines.lossPhotometric)
        encoder.setBuffer(
            resources.renderColor, offset: 0, index: TrainerBind.LossPhotometric.renderColor
        )
        encoder.setBuffer(
            resources.renderTFinal, offset: 0, index: TrainerBind.LossPhotometric.renderTFinal
        )
        encoder.setBuffer(resources.gtColorIn, offset: 0, index: TrainerBind.LossPhotometric.gtColor)
        encoder.setBuffer(resources.bgColor, offset: 0, index: TrainerBind.LossPhotometric.bgColor)
        encoder.setBuffer(
            resources.composited, offset: 0, index: TrainerBind.LossPhotometric.composited
        )
        encoder.setBuffer(
            resources.gradFinal, offset: 0, index: TrainerBind.LossPhotometric.gradFinal
        )
        encoder.setBuffer(
            resources.ssimSrc, offset: 0, index: TrainerBind.LossPhotometric.ssimPlanes
        )
        encoder.setBuffer(
            resources.lossAccum, offset: 0, index: TrainerBind.LossPhotometric.lossAccum
        )
        encoder.setBytes(
            &loss,
            length: MemoryLayout<TrainerLossUniforms>.stride,
            index: TrainerBind.LossPhotometric.uniforms
        )
        dispatch1D(encoder, pipelines.lossPhotometric, count: px)
    }

    /// The whole SSIM forward and backward, in the plane order documented at
    /// the top of the SSIM section of TrainerShaders.metal.
    ///
    /// Three plane buffers, and which one holds what matters:
    ///   ssimSrc  X and Y luma, written by the photometric kernel. It must
    ///            survive to the very last step, which reads it as `lumaPlanes`.
    ///   ssimMid  blurred moments, then the h-blurred partials.
    ///   ssimTmp  h-blurred moments, then the raw partials, then the fully
    ///            blurred partials.
    func ssim(_ encoder: MTLComputeCommandEncoder, loss: inout TrainerLossUniforms) {
        let size = resources.renderSize
        let px = size.pixelCount
        guard px > 0 else { return }

        var blur = TrainerBlurUniforms(
            width: UInt32(size.width),
            height: UInt32(size.height),
            planeCount: UInt32(TrainerGPUConstants.ssimPlaneCount)
        )

        // X*X, Y*Y and X*Y are formed inside trainer_blur_h's moments pass;
        // there is no separate prepare kernel. pad0 MUST go back to 0 before
        // the partials blur below, or that pass squares the partials.
        // setBytes copies at encode time, so resetting after the call is right.
        blur.pad0 = 1
        blurBoth(encoder, from: resources.ssimSrc, through: resources.ssimTmp, into: resources.ssimMid, blur: &blur)
        blur.pad0 = 0

        // Moments -> SSIM value and the five partial-derivative planes.
        encoder.setComputePipelineState(pipelines.ssimStats)
        encoder.setBuffer(resources.ssimMid, offset: 0, index: TrainerBind.SSIMStats.blurred)
        encoder.setBuffer(resources.ssimTmp, offset: 0, index: TrainerBind.SSIMStats.partials)
        encoder.setBuffer(resources.lossAccum, offset: 0, index: TrainerBind.SSIMStats.lossAccum)
        encoder.setBytes(
            &loss, length: MemoryLayout<TrainerLossUniforms>.stride, index: TrainerBind.SSIMStats.uniforms
        )
        dispatch1D(encoder, pipelines.ssimStats, count: px)

        // THREE planes now, not five: trainer_ssim_stats folds the three
        // terms that carry no per-pixel factor into plane 0 before the blur,
        // which is legal because the blur is linear. Set AFTER ssimStats
        // above, which does not read planeCount, and after the moments blur at
        // the top, which genuinely needs all five. `blur` is one var passed
        // inout to both calls, so the order of this line is load-bearing.
        blur.planeCount = 3
        // The partials get the same separable blur, and land back in ssimTmp.
        blurBoth(encoder, from: resources.ssimTmp, through: resources.ssimMid, into: resources.ssimTmp, blur: &blur)
        // The SSIM backward is folded into trainer_loss_finalize, the next
        // dispatch, which reads ssimTmp (blurred partials) and ssimSrc (luma).
    }

    /// Horizontal then vertical. `scratch` must be a third buffer, never
    /// `source` or `destination`, because the horizontal pass reads a whole
    /// row while the vertical pass writes one pixel.
    private func blurBoth(
        _ encoder: MTLComputeCommandEncoder,
        from source: MTLBuffer,
        through scratch: MTLBuffer,
        into destination: MTLBuffer,
        blur: inout TrainerBlurUniforms
    ) {
        let size = resources.renderSize
        encoder.setComputePipelineState(pipelines.blurH)
        encoder.setBuffer(source, offset: 0, index: TrainerBind.Blur.src)
        encoder.setBuffer(scratch, offset: 0, index: TrainerBind.Blur.dst)
        encoder.setBytes(
            &blur, length: MemoryLayout<TrainerBlurUniforms>.stride, index: TrainerBind.Blur.uniforms
        )
        dispatch2D(encoder, pipelines.blurH, width: size.width, height: size.height)

        encoder.setComputePipelineState(pipelines.blurV)
        encoder.setBuffer(scratch, offset: 0, index: TrainerBind.Blur.src)
        encoder.setBuffer(destination, offset: 0, index: TrainerBind.Blur.dst)
        encoder.setBytes(
            &blur, length: MemoryLayout<TrainerBlurUniforms>.stride, index: TrainerBind.Blur.uniforms
        )
        dispatch2D(encoder, pipelines.blurV, width: size.width, height: size.height)
    }

    func lossFinalize(_ encoder: MTLComputeCommandEncoder, loss: inout TrainerLossUniforms) {
        let px = resources.renderSize.pixelCount
        guard px > 0 else { return }
        encoder.setComputePipelineState(pipelines.lossFinalize)
        encoder.setBuffer(resources.gradFinal, offset: 0, index: TrainerBind.LossFinalize.gradFinal)
        encoder.setBuffer(
            resources.ssimTmp, offset: 0, index: TrainerBind.LossFinalize.blurredPartials
        )
        encoder.setBuffer(
            resources.ssimSrc, offset: 0, index: TrainerBind.LossFinalize.lumaPlanes
        )
        encoder.setBuffer(
            resources.renderColor, offset: 0, index: TrainerBind.LossFinalize.renderColor
        )
        encoder.setBuffer(
            resources.renderTFinal, offset: 0, index: TrainerBind.LossFinalize.renderTFinal
        )
        encoder.setBuffer(resources.bgColor, offset: 0, index: TrainerBind.LossFinalize.bgColor)
        encoder.setBuffer(
            resources.gradSplatColor, offset: 0, index: TrainerBind.LossFinalize.gradSplat
        )
        encoder.setBuffer(
            resources.gradTFinal, offset: 0, index: TrainerBind.LossFinalize.gradTFinal
        )
        encoder.setBuffer(
            resources.exposureGrad, offset: 0, index: TrainerBind.LossFinalize.exposureGrad
        )
        encoder.setBytes(
            &loss,
            length: MemoryLayout<TrainerLossUniforms>.stride,
            index: TrainerBind.LossFinalize.uniforms
        )
        dispatch1D(encoder, pipelines.lossFinalize, count: px)
    }

    func lossDepth(
        _ encoder: MTLComputeCommandEncoder,
        loss: inout TrainerLossUniforms,
        sampleCount: Int
    ) {
        guard sampleCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.lossDepth)
        encoder.setBuffer(resources.depthSamplesIn, offset: 0, index: TrainerBind.LossDepth.samples)
        encoder.setBuffer(
            resources.renderDepth, offset: 0, index: TrainerBind.LossDepth.renderDepth
        )
        encoder.setBuffer(
            resources.renderAlpha, offset: 0, index: TrainerBind.LossDepth.renderAlpha
        )
        encoder.setBuffer(
            resources.gradDepthRend, offset: 0, index: TrainerBind.LossDepth.gradDepth
        )
        encoder.setBuffer(resources.gradTFinal, offset: 0, index: TrainerBind.LossDepth.gradTFinal)
        encoder.setBuffer(
            resources.unknownMask, offset: 0, index: TrainerBind.LossDepth.unknownMask
        )
        encoder.setBuffer(resources.lossAccum, offset: 0, index: TrainerBind.LossDepth.lossAccum)
        encoder.setBytes(
            &loss,
            length: MemoryLayout<TrainerLossUniforms>.stride,
            index: TrainerBind.LossDepth.uniforms
        )
        dispatch1D(encoder, pipelines.lossDepth, count: sampleCount)
    }

    // MARK: Backward

    func rasterizeBackward(
        _ encoder: MTLComputeCommandEncoder,
        camera: inout TrainerCameraUniforms,
        loss: inout TrainerLossUniforms,
        simdSum: Bool = false
    ) {
        let size = resources.renderSize
        guard size.tileCount > 0 else { return }
        // Same bindings and dispatch for both: only the specialisation differs.
        let pipeline = (simdSum ? pipelines.rasterizeBackwardSimdSum : nil)
            ?? pipelines.rasterizeBackward
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(resources.valuesA, offset: 0, index: TrainerBind.RasterizeBackward.values)
        encoder.setBuffer(
            resources.tileRanges, offset: 0, index: TrainerBind.RasterizeBackward.tileRanges
        )
        encoder.setBuffer(resources.raster, offset: 0, index: TrainerBind.RasterizeBackward.draws)
        encoder.setBuffer(
            resources.renderTFinal, offset: 0, index: TrainerBind.RasterizeBackward.renderTFinal
        )
        encoder.setBuffer(
            resources.renderNContrib,
            offset: 0,
            index: TrainerBind.RasterizeBackward.renderNContrib
        )
        encoder.setBuffer(
            resources.gradSplatColor,
            offset: 0,
            index: TrainerBind.RasterizeBackward.gradSplatColor
        )
        encoder.setBuffer(
            resources.gradDepthRend, offset: 0, index: TrainerBind.RasterizeBackward.gradDepth
        )
        encoder.setBuffer(
            resources.gradTFinal, offset: 0, index: TrainerBind.RasterizeBackward.gradTFinal
        )
        encoder.setBuffer(resources.bgColor, offset: 0, index: TrainerBind.RasterizeBackward.bgColor)
        encoder.setBuffer(
            resources.unknownMask, offset: 0, index: TrainerBind.RasterizeBackward.unknownMask
        )
        encoder.setBuffer(
            resources.splatGrad2D, offset: 0, index: TrainerBind.RasterizeBackward.splatGrad2D
        )
        encoder.setBytes(
            &camera,
            length: MemoryLayout<TrainerCameraUniforms>.stride,
            index: TrainerBind.RasterizeBackward.camera
        )
        encoder.setBytes(
            &loss,
            length: MemoryLayout<TrainerLossUniforms>.stride,
            index: TrainerBind.RasterizeBackward.lossUniforms
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: size.tileCountX, height: size.tileCountY, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: TrainerGPUConstants.tileWidth,
                height: TrainerGPUConstants.tileHeight,
                depth: 1
            )
        )
    }

    func preprocessBackward(
        _ encoder: MTLComputeCommandEncoder,
        camera: inout TrainerCameraUniforms,
        splatCount: Int
    ) {
        guard splatCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.preprocessBackward)
        encoder.setBuffer(resources.splats, offset: 0, index: TrainerBind.PreprocessBackward.splats)
        encoder.setBuffer(resources.sh, offset: 0, index: TrainerBind.PreprocessBackward.sh)
        encoder.setBuffer(resources.draws, offset: 0, index: TrainerBind.PreprocessBackward.draws)
        encoder.setBuffer(
            resources.splatGrad2D, offset: 0, index: TrainerBind.PreprocessBackward.splatGrad2D
        )
        encoder.setBuffer(
            resources.splatGrad, offset: 0, index: TrainerBind.PreprocessBackward.splatGrad
        )
        encoder.setBuffer(resources.shGrad, offset: 0, index: TrainerBind.PreprocessBackward.shGrad)
        encoder.setBuffer(
            resources.cameraGrad, offset: 0, index: TrainerBind.PreprocessBackward.cameraGrad
        )
        encoder.setBuffer(resources.stats, offset: 0, index: TrainerBind.PreprocessBackward.stats)
        // The "was this splat drawn" predicate. Replaces two scattered stores
        // per splat in trainer_preprocess; see the note at that kernel's
        // prologue.
        encoder.setBuffer(
            resources.tilesTouched, offset: 0,
            index: TrainerBind.PreprocessBackward.tilesTouched
        )
        encoder.setBytes(
            &camera,
            length: MemoryLayout<TrainerCameraUniforms>.stride,
            index: TrainerBind.PreprocessBackward.camera
        )
        dispatch1D(encoder, pipelines.preprocessBackward, count: splatCount)
    }

    // MARK: Mip-Splatting 3D filter

    func samplingRateUpdate(
        _ encoder: MTLComputeCommandEncoder,
        camera: inout TrainerCameraUniforms,
        splatCount: Int
    ) {
        guard splatCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.samplingRateUpdate)
        encoder.setBuffer(resources.splats, offset: 0, index: TrainerBind.SamplingRateUpdate.splats)
        encoder.setBuffer(
            resources.samplingTopK, offset: 0, index: TrainerBind.SamplingRateUpdate.topK
        )
        encoder.setBytes(
            &camera,
            length: MemoryLayout<TrainerCameraUniforms>.stride,
            index: TrainerBind.SamplingRateUpdate.camera
        )
        dispatch1D(encoder, pipelines.samplingRateUpdate, count: splatCount)
    }

    func filter3DFinalize(
        _ encoder: MTLComputeCommandEncoder,
        splatCount: Int,
        filterScale: Float,
        fallback: Float
    ) {
        guard splatCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.filter3DFinalize)
        encoder.setBuffer(
            resources.samplingTopK, offset: 0, index: TrainerBind.Filter3DFinalize.topK
        )
        encoder.setBuffer(resources.stats, offset: 0, index: TrainerBind.Filter3DFinalize.stats)
        var n = UInt32(splatCount)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: TrainerBind.Filter3DFinalize.count)
        var scale = filterScale
        encoder.setBytes(
            &scale, length: MemoryLayout<Float>.size, index: TrainerBind.Filter3DFinalize.filterScale
        )
        var back = fallback
        encoder.setBytes(
            &back, length: MemoryLayout<Float>.size, index: TrainerBind.Filter3DFinalize.fallback
        )
        dispatch1D(encoder, pipelines.filter3DFinalize, count: splatCount)
    }

    // MARK: Regulariser and optimiser

    func regularizer(_ encoder: MTLComputeCommandEncoder, reg: inout TrainerRegUniforms) {
        let count = Int(reg.count)
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.regularizer)
        encoder.setBuffer(resources.splats, offset: 0, index: TrainerBind.Regularizer.splats)
        encoder.setBuffer(resources.stats, offset: 0, index: TrainerBind.Regularizer.stats)
        encoder.setBuffer(resources.splatGrad, offset: 0, index: TrainerBind.Regularizer.grad)
        encoder.setBuffer(resources.lossAccum, offset: 0, index: TrainerBind.Regularizer.lossAccum)
        encoder.setBuffer(
            resources.tilesTouched, offset: 0, index: TrainerBind.Regularizer.tilesTouched
        )
        encoder.setBytes(
            &reg,
            length: MemoryLayout<TrainerRegUniforms>.stride,
            index: TrainerBind.Regularizer.uniforms
        )
        dispatch1D(encoder, pipelines.regularizer, count: count)
    }

    func adamSplat(_ encoder: MTLComputeCommandEncoder, adam: inout TrainerAdamUniforms) {
        let count = Int(adam.count)
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.adamSplat)
        encoder.setBuffer(resources.splats, offset: 0, index: TrainerBind.AdamSplat.splats)
        encoder.setBuffer(resources.splatGrad, offset: 0, index: TrainerBind.AdamSplat.grad)
        encoder.setBuffer(resources.adamM, offset: 0, index: TrainerBind.AdamSplat.m)
        encoder.setBuffer(resources.adamV, offset: 0, index: TrainerBind.AdamSplat.v)
        encoder.setBuffer(resources.stats, offset: 0, index: TrainerBind.AdamSplat.stats)
        encoder.setBuffer(
            resources.tilesTouched, offset: 0, index: TrainerBind.AdamSplat.tilesTouched
        )
        encoder.setBytes(
            &adam,
            length: MemoryLayout<TrainerAdamUniforms>.stride,
            index: TrainerBind.AdamSplat.uniforms
        )
        dispatch1D(encoder, pipelines.adamSplat, count: count)
    }

    /// One thread per Gaussian, each walking its own coefficient run. The SH
    /// Adam MUST be encoded after `adamSplat`, because `adamSplat` is what
    /// advances `stats.stepCount` and the SH bias correction reads it.
    func adamSH(_ encoder: MTLComputeCommandEncoder, adam: inout TrainerAdamUniforms) {
        let count = Int(adam.count)
        guard count > 0 else { return }
        encoder.setComputePipelineState(pipelines.adamSH)
        encoder.setBuffer(resources.sh, offset: 0, index: TrainerBind.AdamSH.sh)
        encoder.setBuffer(resources.shGrad, offset: 0, index: TrainerBind.AdamSH.grad)
        encoder.setBuffer(resources.shAdamM, offset: 0, index: TrainerBind.AdamSH.m)
        encoder.setBuffer(resources.shAdamV, offset: 0, index: TrainerBind.AdamSH.v)
        encoder.setBuffer(resources.stats, offset: 0, index: TrainerBind.AdamSH.stats)
        encoder.setBuffer(
            resources.tilesTouched, offset: 0, index: TrainerBind.AdamSH.tilesTouched
        )
        encoder.setBytes(
            &adam,
            length: MemoryLayout<TrainerAdamUniforms>.stride,
            index: TrainerBind.AdamSH.uniforms
        )
        dispatch1D(encoder, pipelines.adamSH, count: count)
    }

    // MARK: Snapshot support

    func extractCenters(_ encoder: MTLComputeCommandEncoder, splatCount: Int) {
        guard splatCount > 0 else { return }
        encoder.setComputePipelineState(pipelines.extractCenters)
        encoder.setBuffer(resources.splats, offset: 0, index: TrainerBind.ExtractCenters.splats)
        encoder.setBuffer(resources.centers, offset: 0, index: TrainerBind.ExtractCenters.centers)
        var n = UInt32(splatCount)
        encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: TrainerBind.ExtractCenters.count)
        dispatch1D(encoder, pipelines.extractCenters, count: splatCount)
    }

    // MARK: Per-iteration clears
    //
    // Every buffer a kernel ACCUMULATES into with `+=` or an atomic add has to
    // start the iteration at zero. Every buffer a kernel assigns with `=` does
    // not. Getting that list wrong is a slow drift rather than a crash, so it
    // is written out explicitly rather than "zero everything", which would
    // also wipe the densification statistics the interval is meant to gather.

    /// Zeroes everything the step accumulates into, with the BLIT engine.
    ///
    /// Was nine compute dispatches running `trainer_fill_float`, one per
    /// buffer, each with its own threadgroup setup and its own barrier against
    /// the next. A blit `fill` is a DMA: no threads, no dispatch, and it runs
    /// at copy bandwidth rather than at whatever a one-float-per-thread kernel
    /// manages. Nine dispatches become nine fills inside one encoder.
    ///
    /// Only legal because the value is ZERO. `MTLBlitCommandEncoder.fill`
    /// writes a repeated BYTE, and the float whose four bytes are all zero is
    /// 0.0. Any other value would still need the compute kernel, which is why
    /// `fillFloat` stays.
    ///
    /// The visibility reset stays on the compute side: it writes a flag, not a
    /// float, and it is one dispatch.
    func clearPerIteration(_ blit: MTLBlitCommandEncoder, splatCount: Int) {
        let px = resources.renderSize.pixelCount

        func zero(_ buffer: MTLBuffer, floats: Int) {
            let bytes = Swift.min(floats * 4, buffer.length)
            guard bytes > 0 else { return }
            blit.fill(buffer: buffer, range: 0..<bytes, value: 0)
        }

        // splatGrad AND shGrad ARE NOT CLEARED HERE ANY MORE either.
        // trainer_preprocess_backward zeroes a drawn splat's two rows before
        // it accumulates into them, and an undrawn splat's rows are never
        // read: both Adam kernels and trainer_regularizer skip
        // tilesTouched == 0, with sparse = 1. 28.8 MB of fill an iteration at
        // 300k splats.
        // splatGrad2D IS NOT CLEARED HERE ANY MORE. trainer_preprocess_backward
        // zeroes each row as it consumes it, which it can do exactly because it
        // early-returns on the same `tilesTouched == 0` predicate that decides
        // whether the backward rasteriser wrote that row at all. This was
        // 19.13 MB of the ~52.5 MB this function fills every iteration.

        // Accumulated per pixel with `+=`.
        zero(resources.gradDepthRend, floats: px)
        zero(resources.gradTFinal, floats: px)
        zero(resources.unknownMask, floats: px)

        // Scalars.
        zero(resources.lossAccum, floats: 1)
        zero(resources.exposureGrad, floats: 2)
        zero(resources.cameraGrad, floats: 6)

    }

    /// Rasterises the far field straight into `bgColor` on the GPU.
    ///
    /// Replaces a Swift loop over every pixel on the prefetch worker plus a
    /// 4.67 MB memcpy on the main thread. The cubemap is 72 KB and is uploaded
    /// whole each time rather than versioned: at that size the copy is noise
    /// and a version counter is one more thing to get wrong.
    func background(
        _ encoder: MTLComputeCommandEncoder,
        uniforms: inout TrainerBackgroundUniforms
    ) {
        let px = resources.renderSize.pixelCount
        guard px > 0 else { return }
        encoder.setComputePipelineState(pipelines.background)
        encoder.setBuffer(
            resources.bgCubemapIn, offset: 0, index: TrainerBind.Background.texels
        )
        encoder.setBuffer(
            resources.bgColor, offset: 0, index: TrainerBind.Background.bgColor
        )
        encoder.setBytes(
            &uniforms,
            length: MemoryLayout<TrainerBackgroundUniforms>.stride,
            index: TrainerBind.Background.uniforms
        )
        dispatch1D(encoder, pipelines.background, count: px)
    }
}
