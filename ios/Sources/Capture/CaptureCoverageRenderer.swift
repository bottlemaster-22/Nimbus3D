//
//  CaptureCoverageRenderer.swift
//  Capture
//
//  The CPU half of the capture HUD: the camera feed, and ARKit's classified
//  mesh painted with the three coverage channels (F9).
//
//  IT DOES NOT HOLD ON TO `ARFrame`s. ARKit's frame pool is small, and a
//  retained frame is a frame the tracker cannot reuse - holding one across a
//  render is a documented way to make the session stutter. So `ingest` pulls
//  four things out of the frame while it is alive (two Metal textures backed
//  by the image's IOSurface, the view-projection matrix, the camera position,
//  and the image-to-view transform) and lets the frame go. Nothing in this
//  class outlives that call except those values.
//
//  IT DOES NOT TOUCH `ARMeshAnchor` EITHER. Geometry comes from
//  `CaptureMeshStore`'s snapshots, which are plain Swift arrays. ARKit's mesh
//  buffers are only safe to read on the delegate thread and only while the
//  callback lasts; the snapshot is what makes it legal to build GPU buffers on
//  the render thread at all.
//
//  GPU buffers are cached per mesh chunk and rebuilt only when that chunk's
//  snapshot version changes. The coverage attribute is refreshed on its own,
//  slower schedule, because coverage moves at walking pace and re-sampling a
//  house's worth of vertices sixty times a second would be most of the
//  thermal budget for a HUD.
//

import ARKit
import Metal
import MetalKit
import UIKit
import simd

// MARK: - Uniform layouts
//
// Byte-matched to CaptureCoverageShaders.metal. Built only from `simd_float4x4`,
// `SIMD4<Float>` and `simd_float3x3`, all of which have identical size and
// alignment on both sides. A `SIMD3<Float>` next to a `Float` would not, and
// that mismatch is how a uniform buffer silently shifts by four bytes.

struct CaptureCoverageUniforms {
    var viewProjection: simd_float4x4
    /// xyz world position of the camera; w unused.
    var cameraPosition: SIMD4<Float>
    /// x overlay opacity, y done threshold, z isolated channel (-1 for all),
    /// w unused.
    var params: SIMD4<Float>
}

struct CaptureBackgroundUniforms {
    /// View UV (0...1, origin top-left) -> camera image UV.
    var textureTransform: simd_float3x3
}

/// Draws the capture HUD.
///
/// `@MainActor` because `MTKView` calls its delegate on the main thread and
/// because `ingest` is called straight from the ARKit delegate.
@MainActor
final class CaptureCoverageRenderer: NSObject, MTKViewDelegate {

    // MARK: - User-facing switches

    /// 0 hides the overlay entirely, 1 is full strength. The capture screen
    /// exposes this because someone framing a shot sometimes wants to see the
    /// room and not the paint.
    var overlayOpacity: Float = 0.75

    /// When set, only that channel is drawn, as a ramp from its own colour to
    /// green. nil draws all three, each patch showing its worst.
    var isolatedChannel: CaptureCoverageChannel?

    // MARK: - Metal

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var backgroundPipeline: MTLRenderPipelineState?
    private var coveragePipeline: MTLRenderPipelineState?
    private var depthState: MTLDepthStencilState?
    private var textureCache: CVMetalTextureCache?

    // MARK: - Data sources

    private let coverageField: CaptureCoverageField
    private let meshStore: CaptureMeshStore

    // MARK: - Per-frame state, extracted from the ARFrame and nothing more

    private var lumaTexture: MTLTexture?
    private var chromaTexture: MTLTexture?
    private var backgroundUniforms = CaptureBackgroundUniforms(
        textureTransform: matrix_identity_float3x3
    )
    private var viewProjection = matrix_identity_float4x4
    private var cameraPosition = SIMD4<Float>(0, 0, 0, 1)
    private var hasFrame = false

    // Keeping the CVMetalTexture wrappers alive for the life of the draw is
    // required: releasing them invalidates the MTLTextures they vend.
    private var lumaWrapper: CVMetalTexture?
    private var chromaWrapper: CVMetalTexture?

    // MARK: - Cached GPU geometry

    private struct ChunkBuffers {
        var version: Int
        var vertexCount: Int
        var indexCount: Int
        var positions: MTLBuffer
        var normals: MTLBuffer
        var coverage: MTLBuffer
        var indices: MTLBuffer
        /// Kept CPU-side so the coverage attribute can be re-sampled without
        /// reading back from the GPU.
        var worldPositions: [SIMD3<Float>]
    }

    private var chunks: [UUID: ChunkBuffers] = [:]
    private var lastCoverageRefresh: CFTimeInterval = 0

    // MARK: - Init

    /// - Returns: nil when the device has no Metal, or when the shader
    ///   functions are missing from the default library. Both are reported to
    ///   the caller rather than papered over: a HUD that silently draws nothing
    ///   is worse than a screen that says the overlay is unavailable.
    init?(
        device: MTLDevice,
        coverageField: CaptureCoverageField,
        meshStore: CaptureMeshStore
    ) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = queue
        self.coverageField = coverageField
        self.meshStore = meshStore
        super.init()

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        self.textureCache = cache
    }

    /// Builds the pipelines against the view's pixel formats.
    ///
    /// Separate from `init` because a pipeline is tied to the colour and depth
    /// formats of the view it draws into, and those are not known until the
    /// view exists.
    func configure(for view: MTKView) {
        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.framebufferOnly = true
        view.preferredFramesPerSecond = 30
        view.isOpaque = true

        guard let library = device.makeDefaultLibrary() else {
            CaptureLog.renderer.error(
                "No default Metal library; the coverage overlay cannot draw."
            )
            return
        }

        backgroundPipeline = makePipeline(
            library: library,
            vertex: "capture_background_vertex",
            fragment: "capture_background_fragment",
            view: view,
            blending: false
        )
        coveragePipeline = makePipeline(
            library: library,
            vertex: "capture_coverage_vertex",
            fragment: "capture_coverage_fragment",
            view: view,
            blending: true
        )

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: depthDescriptor)
    }

    private func makePipeline(
        library: MTLLibrary,
        vertex: String,
        fragment: String,
        view: MTKView,
        blending: Bool
    ) -> MTLRenderPipelineState? {
        guard
            let vertexFunction = library.makeFunction(name: vertex),
            let fragmentFunction = library.makeFunction(name: fragment)
        else {
            let message = "Missing Metal function \(vertex) or "
                    + "\(fragment)."
            CaptureLog.renderer.error("\(message, privacy: .public)")
            return nil
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        descriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat
        if blending {
            let attachment = descriptor.colorAttachments[0]
            attachment?.isBlendingEnabled = true
            attachment?.rgbBlendOperation = .add
            attachment?.alphaBlendOperation = .add
            attachment?.sourceRGBBlendFactor = .sourceAlpha
            attachment?.sourceAlphaBlendFactor = .sourceAlpha
            attachment?.destinationRGBBlendFactor = .oneMinusSourceAlpha
            attachment?.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        }
        do {
            return try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            let message = "Pipeline \(vertex) failed: "
                    + "\(error.localizedDescription)"
            CaptureLog.renderer.error("\(message, privacy: .public)")
            return nil
        }
    }

    // MARK: - Frame intake

    /// Pulls what the renderer needs out of one `ARFrame` and lets it go.
    ///
    /// Uses ARKit's OWN view and projection matrices, unmodified. They are in
    /// ARKit's camera convention (+Y up, -Z forward) and are applied to
    /// world-space vertices, so nothing here has to know about - or convert
    /// to - this app's COLMAP camera convention. That conversion belongs to
    /// `Pose.fromARKitCameraTransform` and to the data on disk, not to a
    /// preview.
    func ingest(
        frame: ARFrame,
        viewportSize: CGSize,
        orientation: UIInterfaceOrientation
    ) {
        guard viewportSize.width > 0, viewportSize.height > 0 else { return }

        makeCameraTextures(from: frame.capturedImage)

        let display = frame.displayTransform(
            for: orientation,
            viewportSize: viewportSize
        ).inverted()
        backgroundUniforms = CaptureBackgroundUniforms(
            textureTransform: Self.float3x3(from: display)
        )

        let view = frame.camera.viewMatrix(for: orientation)
        let projection = frame.camera.projectionMatrix(
            for: orientation,
            viewportSize: viewportSize,
            zNear: 0.02,
            zFar: 60
        )
        viewProjection = projection * view

        let transform = frame.camera.transform
        cameraPosition = SIMD4<Float>(
            transform.columns.3.x,
            transform.columns.3.y,
            transform.columns.3.z,
            1
        )
        hasFrame = true
    }

    private func makeCameraTextures(from pixelBuffer: CVPixelBuffer) {
        guard
            let textureCache,
            CVPixelBufferGetPlaneCount(pixelBuffer) >= 2
        else { return }

        func makeTexture(
            plane: Int,
            format: MTLPixelFormat
        ) -> (CVMetalTexture, MTLTexture)? {
            let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, plane)
            let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane)
            var wrapper: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                textureCache,
                pixelBuffer,
                nil,
                format,
                width,
                height,
                plane,
                &wrapper
            )
            guard
                status == kCVReturnSuccess,
                let wrapper,
                let texture = CVMetalTextureGetTexture(wrapper)
            else { return nil }
            return (wrapper, texture)
        }

        if let (wrapper, texture) = makeTexture(plane: 0, format: .r8Unorm) {
            lumaWrapper = wrapper
            lumaTexture = texture
        }
        if let (wrapper, texture) = makeTexture(plane: 1, format: .rg8Unorm) {
            chromaWrapper = wrapper
            chromaTexture = texture
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Nothing cached is size-dependent: the background is a full-screen
        // triangle and the projection comes fresh from ARKit every frame.
    }

    func draw(in view: MTKView) {
        guard
            hasFrame,
            let descriptor = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable,
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeRenderCommandEncoder(
                descriptor: descriptor
            )
        else { return }

        drawBackground(with: encoder)
        refreshChunksIfNeeded()
        drawCoverage(with: encoder)

        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func drawBackground(with encoder: MTLRenderCommandEncoder) {
        guard
            let backgroundPipeline,
            let lumaTexture,
            let chromaTexture
        else { return }
        encoder.setRenderPipelineState(backgroundPipeline)
        // The background writes no depth, so the mesh overlay behind it still
        // occludes itself correctly.
        encoder.setDepthStencilState(nil)
        var uniforms = backgroundUniforms
        encoder.setVertexBytes(
            &uniforms,
            length: MemoryLayout<CaptureBackgroundUniforms>.stride,
            index: 0
        )
        encoder.setFragmentTexture(lumaTexture, index: 0)
        encoder.setFragmentTexture(chromaTexture, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }

    private func drawCoverage(with encoder: MTLRenderCommandEncoder) {
        guard overlayOpacity > 0.001, let coveragePipeline else { return }
        encoder.setRenderPipelineState(coveragePipeline)
        if let depthState { encoder.setDepthStencilState(depthState) }

        var uniforms = CaptureCoverageUniforms(
            viewProjection: viewProjection,
            cameraPosition: cameraPosition,
            params: SIMD4<Float>(
                overlayOpacity,
                CaptureTuning.coverageChannelDoneThreshold,
                Float(isolatedChannel?.rawValue ?? -1),
                0
            )
        )

        for chunk in chunks.values where chunk.indexCount > 0 {
            encoder.setVertexBuffer(chunk.positions, offset: 0, index: 0)
            encoder.setVertexBuffer(chunk.normals, offset: 0, index: 1)
            encoder.setVertexBuffer(chunk.coverage, offset: 0, index: 2)
            encoder.setVertexBytes(
                &uniforms,
                length: MemoryLayout<CaptureCoverageUniforms>.stride,
                index: 3
            )
            encoder.setFragmentBytes(
                &uniforms,
                length: MemoryLayout<CaptureCoverageUniforms>.stride,
                index: 0
            )
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: chunk.indexCount,
                indexType: .uint32,
                indexBuffer: chunk.indices,
                indexBufferOffset: 0
            )
        }
    }

    // MARK: - Buffer maintenance

    /// Rebuilds geometry for chunks whose snapshot changed, and refreshes the
    /// coverage attribute on the coverage field's own slower schedule.
    private func refreshChunksIfNeeded() {
        let snapshots = meshStore.allSnapshots
        var live = Set<UUID>()

        let now = CACurrentMediaTime()
        let refreshCoverage =
            now - lastCoverageRefresh >= 1.0 / CaptureTuning.coverageUpdateHz
        if refreshCoverage { lastCoverageRefresh = now }

        for snapshot in snapshots {
            live.insert(snapshot.identifier)
            let existing = chunks[snapshot.identifier]
            if existing?.version != snapshot.version {
                if let rebuilt = makeBuffers(for: snapshot) {
                    chunks[snapshot.identifier] = rebuilt
                }
            } else if refreshCoverage, let existing {
                updateCoverage(of: existing)
            }
        }

        // Collected first, then removed: mutating a dictionary while iterating
        // its keys is undefined.
        let stale = chunks.keys.filter { !live.contains($0) }
        for identifier in stale {
            chunks.removeValue(forKey: identifier)
        }
    }

    private func makeBuffers(for snapshot: CaptureMeshSnapshot) -> ChunkBuffers? {
        let vertexCount = snapshot.positions.count
        let indexCount = snapshot.indices.count
        guard vertexCount > 0, indexCount > 0 else { return nil }

        guard
            let positions = device.makeBuffer(
                bytes: snapshot.positions,
                length: MemoryLayout<SIMD3<Float>>.stride * vertexCount,
                options: .storageModeShared
            ),
            let normals = device.makeBuffer(
                bytes: snapshot.normals,
                length: MemoryLayout<SIMD3<Float>>.stride * vertexCount,
                options: .storageModeShared
            ),
            let indices = device.makeBuffer(
                bytes: snapshot.indices,
                length: MemoryLayout<UInt32>.stride * indexCount,
                options: .storageModeShared
            ),
            let coverage = device.makeBuffer(
                length: MemoryLayout<SIMD4<Float>>.stride * vertexCount,
                options: .storageModeShared
            )
        else { return nil }

        let buffers = ChunkBuffers(
            version: snapshot.version,
            vertexCount: vertexCount,
            indexCount: indexCount,
            positions: positions,
            normals: normals,
            coverage: coverage,
            indices: indices,
            worldPositions: snapshot.positions
        )
        updateCoverage(of: buffers)
        return buffers
    }

    /// Re-samples the coverage field (and the class map) into the per-vertex
    /// attribute buffer.
    ///
    /// `sampleBatch` takes the field's lock exactly once for the whole chunk,
    /// which is the reason it exists: a per-vertex lock acquisition on a
    /// 40,000-vertex chunk would cost more than the sampling.
    private func updateCoverage(of chunk: ChunkBuffers) {
        let count = chunk.vertexCount
        guard count > 0 else { return }

        let destination = chunk.coverage.contents()
            .bindMemory(to: SIMD4<Float>.self, capacity: count)
        let output = UnsafeMutableBufferPointer(start: destination, count: count)

        chunk.worldPositions.withUnsafeBufferPointer { positions in
            coverageField.sampleBatch(positions: positions, into: output)

            // Fold the surface class into the flag channel. `sampleBatch`
            // writes w = 1 for "seen"; +2 marks the patch optically
            // unreliable, which the shader draws calm rather than angry.
            var classes = [UInt8](repeating: 0, count: count)
            classes.withUnsafeMutableBufferPointer { classBuffer in
                meshStore.classBatch(positions: positions, into: classBuffer)
                let all = SurfaceClass.allCases
                for index in 0..<count {
                    let byte = Int(classBuffer[index])
                    guard byte > 0, byte < all.count else { continue }
                    if all[byte].isOpticallyUnreliable {
                        output[index].w += 2
                    }
                }
            }
        }
    }

    func reset() {
        chunks.removeAll(keepingCapacity: false)
        lumaTexture = nil
        chromaTexture = nil
        lumaWrapper = nil
        chromaWrapper = nil
        hasFrame = false
    }

    // MARK: - Small conversions

    /// `CGAffineTransform` maps `(x, y)` to `(a*x + c*y + tx, b*x + d*y + ty)`.
    /// As a column-major 3x3 that is columns `(a, b, 0)`, `(c, d, 0)`,
    /// `(tx, ty, 1)`.
    static func float3x3(from transform: CGAffineTransform) -> simd_float3x3 {
        simd_float3x3(
            SIMD3<Float>(Float(transform.a), Float(transform.b), 0),
            SIMD3<Float>(Float(transform.c), Float(transform.d), 0),
            SIMD3<Float>(Float(transform.tx), Float(transform.ty), 1)
        )
    }
}
