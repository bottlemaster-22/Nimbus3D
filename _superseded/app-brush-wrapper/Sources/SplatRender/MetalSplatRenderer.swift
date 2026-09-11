//
//  MetalSplatRenderer.swift — SplatRender module.
//
//  REAL Metal implementation of the `SplatRenderer` contract (Sources/Core/
//  Contracts.swift). Pipeline per frame, all in one command buffer:
//    1. computeDepthKeys  (compute) — view-space z per splat
//    2. bitonicSortStep   (compute) x O(log^2 N) — far-to-near order
//    3. splatVertex/splatFragment (render) — instanced EWA rasterization, blended
//
//  Not Sendable by contract: create and drive it from the render-loop owner
//  (the SwiftUI MTKView coordinator on the main actor).
//

import Foundation
import Metal
import QuartzCore
import simd

public final class MetalSplatRenderer: SplatRenderer {

    // MARK: Metal objects
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let library: MTLLibrary

    private let keyPipeline: MTLComputePipelineState
    private let sortPipeline: MTLComputePipelineState
    private var renderPipelines: [MTLPixelFormat: MTLRenderPipelineState] = [:]

    // MARK: Model state
    private var splatBuffer: MTLBuffer?
    private var sortBuffer: MTLBuffer?
    private var splatCount = 0
    private var paddedCount = 0

    /// Bounding box of the currently loaded model (useful for the preview camera).
    public private(set) var boundingBox: AxisAlignedBoundingBox?

    // Sorting is order-invariant to translation-only? No — it depends on the full
    // view. We re-sort whenever the view matrix changes; identical views reuse the
    // existing order (sorting is a pure function of view + splats).
    private var lastSortView: simd_float4x4?
    private var needsSort = false

    public var isModelLoaded: Bool { splatBuffer != nil && splatCount > 0 }

    // MARK: Init

    /// - Throws: `NimbusError.renderingFailed` if Metal or the shader library is unavailable.
    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw NimbusError.renderingFailed("No Metal device available on this hardware.")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw NimbusError.renderingFailed("Could not create a Metal command queue.")
        }
        self.queue = queue

        guard let library = device.makeDefaultLibrary() else {
            throw NimbusError.renderingFailed("default.metallib not found (SplatRender shaders did not compile into the app).")
        }
        self.library = library

        func computePipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw NimbusError.renderingFailed("Missing Metal function \(name).")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        self.keyPipeline = try computePipeline("computeDepthKeys")
        self.sortPipeline = try computePipeline("bitonicSortStep")
    }

    // MARK: - SplatRenderer

    public func load(_ model: SplatModel) async throws {
        guard model.format == .ply else {
            // Honest stub: the compressed Niantic .spz path is not decoded yet.
            // TODO(nimbus): implement an on-device SPZ loader: gunzip the container,
            // then decode fixed-point positions (24-bit), scales, packed rotations,
            // and quantized SH into the same GPUSplat layout used here. Until then,
            // the trainer's native .ply is the supported preview input.
            throw NimbusError.notImplemented("SplatRenderer currently loads .ply (Brush/INRIA) only; .spz decoding is not implemented.")
        }

        let url = model.splatFileURL
        let parsed: ParsedSplats
        do {
            parsed = try await Task.detached(priority: .userInitiated) {
                try SplatPLYParser.parse(contentsOf: url)
            }.value
        } catch {
            throw NimbusError.renderingFailed("Failed to parse splat PLY: \(error)")
        }

        guard !parsed.splats.isEmpty else {
            throw NimbusError.renderingFailed("Splat file \(url.lastPathComponent) contained no Gaussians.")
        }

        let count = parsed.splats.count
        let padded = Self.nextPowerOfTwo(count)

        guard let splatBuf = parsed.splats.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!,
                              length: count * MemoryLayout<GPUSplat>.stride,
                              options: .storageModeShared)
        }) else {
            throw NimbusError.renderingFailed("Could not allocate GPU buffer for \(count) splats.")
        }
        guard let sortBuf = device.makeBuffer(length: padded * MemoryLayout<SortEntry>.stride,
                                              options: .storageModeShared) else {
            throw NimbusError.renderingFailed("Could not allocate sort buffer for \(padded) entries.")
        }
        splatBuf.label = "SplatBuffer(\(count))"
        sortBuf.label = "SortBuffer(\(padded))"

        self.splatBuffer = splatBuf
        self.sortBuffer = sortBuf
        self.splatCount = count
        self.paddedCount = padded
        self.boundingBox = AxisAlignedBoundingBox(minCorner: parsed.boundingMin,
                                                  maxCorner: parsed.boundingMax)
        self.lastSortView = nil
        self.needsSort = true
    }

    public func unload() {
        splatBuffer = nil
        sortBuffer = nil
        splatCount = 0
        paddedCount = 0
        boundingBox = nil
        lastSortView = nil
        needsSort = false
    }

    public func render(to drawable: CAMetalDrawable,
                       viewMatrix: simd_float4x4,
                       projectionMatrix: simd_float4x4,
                       viewportSize: SIMD2<UInt32>) throws {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw NimbusError.renderingFailed("Could not create a command buffer.")
        }

        // Nothing loaded: just clear the drawable so the preview shows a background.
        guard isModelLoaded, let splatBuffer, let sortBuffer, splatCount > 0 else {
            encodeClear(commandBuffer, drawable: drawable)
            commandBuffer.present(drawable)
            commandBuffer.commit()
            return
        }

        // (Re)sort only when the view changed. Order depends purely on view+splats.
        if needsSort || lastSortView.map({ !Self.matricesEqual($0, viewMatrix) }) ?? true {
            encodeSort(commandBuffer, splatBuffer: splatBuffer, sortBuffer: sortBuffer, view: viewMatrix)
            lastSortView = viewMatrix
            needsSort = false
        }

        // --- Rasterize ---
        let pipeline = try renderPipeline(for: drawable.texture.pixelFormat)

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = drawable.texture
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        rpd.colorAttachments[0].storeAction = .store

        guard let enc = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else {
            throw NimbusError.renderingFailed("Could not create a render command encoder.")
        }
        enc.label = "SplatRaster"
        enc.setRenderPipelineState(pipeline)

        let vw = Float(viewportSize.x)
        let vh = Float(viewportSize.y)
        let fx = projectionMatrix.columns.0.x * vw * 0.5
        let fy = projectionMatrix.columns.1.y * vh * 0.5
        var uniforms = SplatUniforms(view: viewMatrix,
                                     projection: projectionMatrix,
                                     viewport: SIMD2<Float>(vw, vh),
                                     focal: SIMD2<Float>(fx, fy))

        enc.setVertexBuffer(splatBuffer, offset: 0, index: 0)
        enc.setVertexBuffer(sortBuffer, offset: 0, index: 1)
        enc.setVertexBytes(&uniforms, length: MemoryLayout<SplatUniforms>.stride, index: 2)
        enc.drawPrimitives(type: .triangleStrip,
                           vertexStart: 0,
                           vertexCount: 4,
                           instanceCount: splatCount)
        enc.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Encoding helpers

    private func encodeClear(_ commandBuffer: MTLCommandBuffer, drawable: CAMetalDrawable) {
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = drawable.texture
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        rpd.colorAttachments[0].storeAction = .store
        commandBuffer.makeRenderCommandEncoder(descriptor: rpd)?.endEncoding()
    }

    private func encodeSort(_ commandBuffer: MTLCommandBuffer,
                            splatBuffer: MTLBuffer,
                            sortBuffer: MTLBuffer,
                            view: simd_float4x4) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "SplatSort"

        // 1. Depth keys. (Default/serial compute encoder inserts hazard barriers
        // between successive dispatches on the same buffer automatically.)
        enc.setComputePipelineState(keyPipeline)
        enc.setBuffer(splatBuffer, offset: 0, index: 0)
        enc.setBuffer(sortBuffer, offset: 0, index: 1)
        var kp = KeyParams(view: view,
                           splatCount: UInt32(splatCount),
                           paddedCount: UInt32(paddedCount))
        enc.setBytes(&kp, length: MemoryLayout<KeyParams>.stride, index: 2)
        dispatch(enc, pipeline: keyPipeline, count: paddedCount)

        // 2. Bitonic network over the padded array.
        enc.setComputePipelineState(sortPipeline)
        enc.setBuffer(sortBuffer, offset: 0, index: 0)
        var k = 2
        while k <= paddedCount {
            var j = k >> 1
            while j > 0 {
                var sp = SortParams(k: UInt32(k), j: UInt32(j), paddedCount: UInt32(paddedCount))
                enc.setBytes(&sp, length: MemoryLayout<SortParams>.stride, index: 1)
                dispatch(enc, pipeline: sortPipeline, count: paddedCount)
                j >>= 1
            }
            k <<= 1
        }
        enc.endEncoding()
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          count: Int) {
        let w = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    private func renderPipeline(for pixelFormat: MTLPixelFormat) throws -> MTLRenderPipelineState {
        if let cached = renderPipelines[pixelFormat] { return cached }

        guard let vfn = library.makeFunction(name: "splatVertex"),
              let ffn = library.makeFunction(name: "splatFragment") else {
            throw NimbusError.renderingFailed("Missing splat vertex/fragment functions.")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "SplatRasterPipeline"
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn

        let ca = desc.colorAttachments[0]!
        ca.pixelFormat = pixelFormat
        ca.isBlendingEnabled = true
        ca.rgbBlendOperation = .add
        ca.alphaBlendOperation = .add
        ca.sourceRGBBlendFactor = .one                     // fragment is premultiplied
        ca.sourceAlphaBlendFactor = .one
        ca.destinationRGBBlendFactor = .oneMinusSourceAlpha
        ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha

        let pipeline = try device.makeRenderPipelineState(descriptor: desc)
        renderPipelines[pixelFormat] = pipeline
        return pipeline
    }

    // MARK: - Utilities

    private static func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 1 else { return max(1, n) }
        return 1 << (Int.bitWidth - (n - 1).leadingZeroBitCount)
    }

    private static func matricesEqual(_ a: simd_float4x4, _ b: simd_float4x4) -> Bool {
        a.columns.0 == b.columns.0 && a.columns.1 == b.columns.1 &&
        a.columns.2 == b.columns.2 && a.columns.3 == b.columns.3
    }
}
