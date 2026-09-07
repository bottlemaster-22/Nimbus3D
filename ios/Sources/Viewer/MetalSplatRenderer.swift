//
//  MetalSplatRenderer.swift
//  Viewer
//
//  THE PREVIEW RENDERER. Core's `SplatRenderer`, implemented for real.
//
//  Pipeline per frame, all in one command buffer, matching the five stages
//  documented at the top of SplatRenderShaders.metal:
//
//    blit    zero the visible counter
//    compute viewer_splat_preprocess    project, cull, shade, write depth keys
//    compute viewer_sort_local/global   bitonic sort, front to back
//    compute viewer_prepare_indirect    counter -> indirect draw arguments
//    render  viewer_splat_vertex/fragment  one quad per visible splat into an
//                                          offscreen colour + aux pair
//    render  viewer_composite_*         honesty mask + artefact heatmap into
//                                       the drawable
//
//  Three things about this file are worth knowing before changing it.
//
//  1. It is `@MainActor`, because Core's protocol is. Every expensive thing it
//     does - parsing a .ply, building the honesty field - happens in a
//     detached task and comes back as a value. Nothing blocks the main thread
//     except the GPU encode itself, which is microseconds.
//
//  2. Loading is progressive in the way the user experiences it. The file
//     parse is the slow part (seconds for a 500k-splat .ply) and it is fully
//     off the main actor; `load` returns as soon as the parsed cloud has been
//     uploaded and the first slice is drawable, and `residentSplatCount` then
//     climbs a slice per frame until the whole cloud is being drawn. The
//     splats are reordered by bit-reversed index first, so any prefix is a
//     spatially uniform sample of the whole scene rather than whichever corner
//     happened to be written first - the preview thickens rather than growing
//     from one edge.
//
//  3. There is no depth buffer and no fixed-function blending. Ordering comes
//     from the GPU sort; compositing is done by the fragment shader reading
//     its own colour attachment. That is what lets the aux attachment
//     accumulate the two per-pixel quantities the review UX needs (an
//     alpha-weighted depth, and a raw splat overlap count), neither of which
//     survives fixed-function blending.
//

import Foundation
import Metal
import MetalKit
import QuartzCore
import simd

/// How the render intrinsics are derived from the source camera.
enum ViewerFramingMode: Sendable {
    /// Keep the source camera's horizontal field of view and centre the
    /// principal point. This is what the preview fly-through and the A/B
    /// slider want: the render frames the scene the way the photo did.
    case matchHorizontalFOV
    /// Widen (or narrow) to an explicit horizontal field of view in degrees.
    /// The preview path uses this to open up to 100-110 degrees (F9).
    case horizontalFOV(Float)
}

/// Real-time preview rendering. Core's `SplatRenderer`, and the concrete type
/// name CONTRACTS.md reserves for `Sources/Viewer`.
@MainActor
public final class MetalSplatRenderer: SplatRenderer {

    // MARK: - Public state

    /// Non-nil when the renderer could not start. The review screen shows this
    /// verbatim instead of a black rectangle: a preview that silently draws
    /// nothing is indistinguishable from a scan that came out empty, and the
    /// user has no way to tell those apart.
    private(set) var startupProblem: String?

    /// True once a cloud is uploaded and at least one slice is drawable.
    private(set) var hasContent = false

    /// Splats currently being drawn. Climbs to `totalSplatCount` over the
    /// first few frames after a load.
    private(set) var residentSplatCount = 0
    private(set) var totalSplatCount = 0

    /// Whether the honesty mask has anything to say. False means no
    /// observation record was available, and the review screen says so rather
    /// than showing an un-hatched image that looks like a clean bill of health.
    private(set) var hasObservationField = false

    /// Why the honesty mask is unavailable, when it is. Plain sentence.
    private(set) var observationFieldProblem: String?

    /// Bounds of the loaded cloud, for framing the camera.
    private(set) var contentBounds: BoundingBox?

    /// What was actually in the last splat file this renderer opened: how many
    /// of its points can draw at all, and what shape they came out.
    ///
    /// Measured here because this is the only place in the app that already has
    /// the whole cloud in memory - counting it anywhere else would mean parsing
    /// a half-gigabyte .ply a second time. It is measured even for a cloud that
    /// then fails to load, because "the file has 400,000 points and none of
    /// them can draw" is the single most useful sentence this app could say
    /// about a scan that looks like nothing.
    ///
    /// Nil until a file has been opened. The review screen prints "not counted
    /// yet" for nil rather than a zero.
    private(set) var loadedCloudMeasurement: ScanCensus.Drawable?

    /// Set by the view; the renderer never guesses a size.
    private(set) var drawableSize: CGSize = .zero

    /// How the render camera is derived from the intrinsics handed to
    /// `setCamera`.
    var framingMode: ViewerFramingMode = .matchHorizontalFOV

    /// Overall strength of the heatmap tint, 0...1.
    var heatmapGain: Float = 0.75

    /// Rendering statistics for the debug overlay. Honest: `visibleSplats` is
    /// read back from the GPU counter of the PREVIOUS frame, because reading
    /// the current one would mean a stall.
    private(set) var lastVisibleSplatCount: Int = 0

    // MARK: - Metal

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let library: MTLLibrary?

    private var preprocessPipeline: MTLComputePipelineState?
    private var sortLocalPipeline: MTLComputePipelineState?
    private var sortGlobalPipeline: MTLComputePipelineState?
    private var indirectPipeline: MTLComputePipelineState?
    private var rasterPipeline: MTLRenderPipelineState?
    /// Keyed by the drawable's pixel format, because a Metal render pipeline
    /// is bound to its attachment formats and the view can hand us either
    /// bgra8Unorm or its sRGB sibling.
    private var compositePipelines: [MTLPixelFormat: MTLRenderPipelineState] = [:]

    // MARK: - Buffers

    private var baseBuffer: MTLBuffer?
    private var shRestBuffer: MTLBuffer?
    private var drawBuffer: MTLBuffer?
    private var sortBuffer: MTLBuffer?
    private var counterBuffer: MTLBuffer?
    private var indirectBuffer: MTLBuffer?
    private var directionCellBuffer: MTLBuffer?
    /// A 16-byte zero buffer bound wherever a real one is absent. Metal
    /// requires a bound buffer for every argument the shader declares, even
    /// one it never indexes.
    private var emptyBuffer: MTLBuffer?

    private var colorTexture: MTLTexture?
    private var auxTexture: MTLTexture?
    private var targetSize: CGSize = .zero

    /// Padded (power of two) sort length for the CURRENT resident count.
    private var paddedSortCount = 1

    // MARK: - Content

    private var shDegree: SHDegree = .zero
    private var shRestCount: Int = 0
    private var observationField: ObservedDirectionField?

    /// Splats revealed per frame while a load settles. 60k a frame reaches a
    /// 500k cloud in nine frames, which is under a fifth of a second and still
    /// leaves the first frame cheap.
    private var residencyStep = 60_000

    /// Whether a freshly loaded cloud is revealed over several frames.
    ///
    /// TRUE for the review screen, where the surface is running anyway and
    /// spreading a large upload over a few frames keeps the first one
    /// cheap. FALSE for the training preview, where it is actively
    /// harmful: that surface is parked between snapshots and has to be
    /// woken and run at 60 Hz purely to let the reveal finish, which cost
    /// about 42 rendered frames per snapshot where one would do, on the
    /// same GPU the trainer is waiting on.
    var revealsProgressively = true

    // MARK: - Camera

    private var cameraPose: Pose = .identity
    private var sourceIntrinsics: CameraIntrinsics?

    // MARK: - Overlays

    private var honestyMaskEnabled = true
    private var artifactHeatmapEnabled = false

    // MARK: - Attached view

    private weak var attachedView: MTKView?

    // MARK: - Init

    public init() {
        let device = MTLCreateSystemDefaultDevice()
        self.device = device
        self.commandQueue = device?.makeCommandQueue()
        self.library = device?.makeDefaultLibrary()

        if let problem = ViewerGPULayouts.verify() {
            startupProblem = problem
            ViewerLog.renderer.fault("\(problem, privacy: .public)")
            return
        }
        #if DEBUG
        if let problem = ObservedDirectionField.selfCheck() {
            assertionFailure("ObservedDirectionField.selfCheck: \(problem)")
        }
        #endif

        guard device != nil else {
            startupProblem = ViewerError.metalUnavailable(
                "No Metal device is available on this system."
            ).localizedDescription
            return
        }
        guard commandQueue != nil else {
            startupProblem = ViewerError.metalUnavailable(
                "A Metal command queue could not be created."
            ).localizedDescription
            return
        }
        guard library != nil else {
            startupProblem = ViewerError.shaderLibraryMissing(
                "SplatRenderShaders.metal did not make it into this build."
            ).localizedDescription
            return
        }

        do {
            try buildPipelines()
        } catch {
            startupProblem = error.localizedDescription
            ViewerLog.renderer.error(
                "pipeline build failed: \(error.localizedDescription, privacy: .public)"
            )
        }
    }

    private func buildPipelines() throws {
        guard let device, let library else { return }

        func compute(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw ViewerError.shaderLibraryMissing("The kernel \(name) is missing.")
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                throw ViewerError.pipelineFailed("\(name): \(error.localizedDescription)")
            }
        }

        preprocessPipeline = try compute("viewer_splat_preprocess")
        sortLocalPipeline = try compute("viewer_sort_local")
        sortGlobalPipeline = try compute("viewer_sort_global")
        indirectPipeline = try compute("viewer_prepare_indirect")

        guard let vertexFunction = library.makeFunction(name: "viewer_splat_vertex"),
              let fragmentFunction = library.makeFunction(name: "viewer_splat_fragment")
        else {
            throw ViewerError.shaderLibraryMissing("The splat raster shaders are missing.")
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "viewer.splat.raster"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.rasterSampleCount = 1
        // Two float16 attachments: 16 bytes per pixel of tile memory, which
        // leaves headroom under the per-pixel limit on every Apple GPU this
        // app runs on. Blending is OFF on purpose: the fragment shader does
        // the compositing itself so it can also accumulate the aux channels.
        descriptor.colorAttachments[0].pixelFormat = .rgba16Float
        descriptor.colorAttachments[0].isBlendingEnabled = false
        descriptor.colorAttachments[1].pixelFormat = .rgba16Float
        descriptor.colorAttachments[1].isBlendingEnabled = false

        do {
            rasterPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw ViewerError.pipelineFailed("splat raster: \(error.localizedDescription)")
        }

        emptyBuffer = device.makeBuffer(length: 16, options: [.storageModeShared])
        emptyBuffer?.label = "viewer.empty"
        counterBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        )
        counterBuffer?.label = "viewer.visibleCounter"
        indirectBuffer = device.makeBuffer(
            length: MemoryLayout<ViewerIndirectDrawArgs>.stride,
            options: [.storageModeShared]
        )
        indirectBuffer?.label = "viewer.indirectArgs"
    }

    private func compositePipeline(for pixelFormat: MTLPixelFormat) -> MTLRenderPipelineState? {
        if let existing = compositePipelines[pixelFormat] { return existing }
        guard let device, let library else { return nil }
        guard let vertexFunction = library.makeFunction(name: "viewer_composite_vertex"),
              let fragmentFunction = library.makeFunction(name: "viewer_composite_fragment")
        else {
            startupProblem = ViewerError.shaderLibraryMissing(
                "The composite shaders are missing."
            ).localizedDescription
            return nil
        }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "viewer.composite"
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.rasterSampleCount = 1
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        do {
            let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
            compositePipelines[pixelFormat] = pipeline
            return pipeline
        } catch {
            startupProblem = ViewerError.pipelineFailed(
                "composite: \(error.localizedDescription)"
            ).localizedDescription
            return nil
        }
    }

    // MARK: - View attachment

    /// Binds the renderer to the `MTKView` whose drawable `renderFrame()`
    /// draws into. Core's protocol says "renders one frame into the currently
    /// bound drawable"; this is what binds it.
    func attach(to view: MTKView) {
        attachedView = view
        drawableSize = view.drawableSize
        view.device = device
    }

    /// Unbinds the renderer from a view that is going away.
    ///
    /// `from:` matters when two preview surfaces exist for a moment (a screen
    /// pushing another one): the outgoing view must not clear an attachment
    /// the incoming view has already made, or the surface that is still on
    /// screen goes black.
    func detach(from view: MTKView? = nil) {
        if let view, attachedView !== view { return }
        attachedView = nil
    }

    func setDrawableSize(_ size: CGSize) {
        drawableSize = size
    }

    // MARK: - SplatRenderer: loading

    public func load(_ model: SplatModel, at ref: CaptureBundleRef) async throws {
        let paths = ViewerScanPaths(ref: ref)

        // Prefer .ply: it is the format `Sources/Export` writes losslessly.
        // .spz is quantised, which is right for sending over a LAN and wrong
        // for the reference the A/B slider is compared against.
        let candidate: String?
        if let ply = model.plyPath, FileManager.default.fileExists(atPath: paths.url(ply).path) {
            candidate = ply
        } else if let spz = model.spzPath,
                  FileManager.default.fileExists(atPath: paths.url(spz).path) {
            candidate = spz
        } else {
            candidate = model.plyPath ?? model.spzPath
        }

        guard let relativePath = candidate else {
            throw ViewerError.modelHasNoSplatFile(model.scanID)
        }
        let url = paths.url(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ViewerError.fileMissing(relativePath)
        }

        guard let exporter = Self.exporter() else {
            throw ViewerError.exporterUnavailable
        }

        let (cloud, warning) = try await exporter.importSplatCloud(from: url)
        if let warning {
            ViewerLog.renderer.notice("import warning: \(warning, privacy: .public)")
        }

        try await load(cloud)

        // `load(_:)` counts the cloud but is handed no path, so the file it was
        // counted in is stamped on here. The census prints this name next to
        // every number it took from the file, and this is the one place that
        // knows whether the .ply or the .spz fallback was the one opened.
        loadedCloudMeasurement?.sourceFile = relativePath

        // The honesty mask is a separate, optional file. Its absence is a
        // labelled fact on screen, never a silent pass.
        await loadObservationField(model: model, paths: paths)
    }

    public func load(_ cloud: SplatCloud) async throws {
        guard startupProblem == nil, let device else {
            throw ViewerError.metalUnavailable(startupProblem ?? "The GPU is not available.")
        }

        // Flattening for the GPU and counting for the census happen in the same
        // detached task, off the main actor. The census is a separate walk of
        // the same arrays (two of them: one for the model's extent, one for the
        // per-splat tests) rather than a second parse of the file, which is the
        // part that costs seconds.
        //
        // Deliberately BEFORE the empty check below: a cloud with no splats in
        // it is exactly the case the census exists for, and throwing first
        // would leave the review screen with nothing to say about it.
        let outcome = await Task.detached(priority: .userInitiated) {
            () -> (upload: ViewerCloudUpload, measurement: ScanCensus.Drawable) in
            (ViewerCloudUpload.prepare(cloud), ScanCensus.Drawable.measure(cloud))
        }.value
        loadedCloudMeasurement = outcome.measurement
        // Stamped, not measured: `Drawable.measure` is handed a cloud and has
        // no way to know. Same reason `sourceFile` is stamped here.
        loadedCloudMeasurement?.filter3DFused = cloud.filter3DFused

        // The trainer fits every Gaussian through a Mip-Splatting 3D low-pass
        // filter and through the opacity compensation that goes with it. That
        // per-Gaussian filter width exists nowhere but the trainer's own GPU
        // stats buffer, so it has to be folded into the stored scale and
        // opacity before the cloud leaves (`SplatCloud.fuse3DFilter`). A cloud
        // that says outright it was NOT fused is a different model from the one
        // that was trained, and the difference goes the way that makes a good
        // run look like noise. Say so. `nil` is a cloud read back from a file,
        // which carries no marker either way and gets no claim made about it.
        if cloud.filter3DFused == false {
            let complaint: String = "This model arrived without the 3D low-pass filter fused into "
                + "its sizes and opacities. Every splat will draw sharper and more solid "
                + "than the trainer fitted it, and the held-out PSNR was not measured on "
                + "what is about to be drawn."
            ViewerLog.renderer.warning("\(complaint, privacy: .public)")
        }

        guard cloud.count > 0 else {
            throw ViewerError.emptyModel
        }

        try uploadPrepared(outcome.upload, device: device)
    }

    private func uploadPrepared(_ prepared: ViewerCloudUpload, device: MTLDevice) throws {
        let count = prepared.bases.count
        guard count > 0 else { throw ViewerError.emptyModel }

        let baseLength = count * MemoryLayout<ViewerSplatBase>.stride
        guard let bases = device.makeBuffer(length: baseLength, options: [.storageModeShared])
        else {
            throw ViewerError.pipelineFailed(
                "There is not enough graphics memory for \(count) splats."
            )
        }
        bases.label = "viewer.splatBases"
        prepared.bases.withUnsafeBytes { raw in
            guard let source = raw.baseAddress else { return }
            bases.contents().copyMemory(from: source, byteCount: baseLength)
        }

        let shBuffer: MTLBuffer?
        if prepared.shRest.isEmpty {
            shBuffer = nil
        } else {
            let length = prepared.shRest.count * MemoryLayout<Float>.stride
            guard let buffer = device.makeBuffer(length: length, options: [.storageModeShared])
            else {
                throw ViewerError.pipelineFailed(
                    "There is not enough graphics memory for this model's colour detail."
                )
            }
            buffer.label = "viewer.shRest"
            prepared.shRest.withUnsafeBytes { raw in
                guard let source = raw.baseAddress else { return }
                buffer.contents().copyMemory(from: source, byteCount: length)
            }
            shBuffer = buffer
        }

        let draws = device.makeBuffer(
            length: count * MemoryLayout<ViewerSplatDraw>.stride,
            options: [.storageModePrivate]
        )
        draws?.label = "viewer.splatDraws"

        let padded = ViewerMath.nextPowerOfTwo(count)
        let sort = device.makeBuffer(
            length: padded * MemoryLayout<ViewerSortEntry>.stride,
            options: [.storageModePrivate]
        )
        sort?.label = "viewer.sortEntries"

        guard draws != nil, sort != nil else {
            throw ViewerError.pipelineFailed(
                "There is not enough graphics memory to draw \(count) splats."
            )
        }

        baseBuffer = bases
        shRestBuffer = shBuffer
        drawBuffer = draws
        sortBuffer = sort
        totalSplatCount = count
        shDegree = prepared.shDegree
        shRestCount = prepared.restCoefficientsPerSplat
        contentBounds = prepared.bounds
        hasContent = true

        // First slice is drawable immediately and the rest arrives over the
        // next few frames, unless the caller needs one frame to be the
        // whole picture.
        residentSplatCount = revealsProgressively
            ? Swift.min(count, Swift.max(residencyStep, count / 8))
            : count
        paddedSortCount = ViewerMath.nextPowerOfTwo(residentSplatCount)

        ViewerLog.renderer.notice(
            "loaded \(count) splats, SH degree \(prepared.shDegree.rawValue, privacy: .public)"
        )
        attachedView?.setNeedsDisplay()
    }

    /// Drops the current cloud. Called when a review screen goes away, so a
    /// house-sized model is not still resident behind the library list.
    func unload() {
        baseBuffer = nil
        shRestBuffer = nil
        drawBuffer = nil
        sortBuffer = nil
        directionCellBuffer = nil
        observationField = nil
        hasObservationField = false
        observationFieldProblem = nil
        hasContent = false
        residentSplatCount = 0
        totalSplatCount = 0
        contentBounds = nil
        loadedCloudMeasurement = nil
    }

    // MARK: - SplatRenderer: camera and overlays

    public func setCamera(pose: Pose, intrinsics: CameraIntrinsics) {
        cameraPose = pose
        sourceIntrinsics = intrinsics
    }

    public func setHonestyMaskEnabled(_ enabled: Bool) {
        honestyMaskEnabled = enabled
    }

    public func setArtifactHeatmapEnabled(_ enabled: Bool) {
        artifactHeatmapEnabled = enabled
    }

    // MARK: - Honesty field

    /// Loads `model/observed_directions.bin` if it exists.
    func loadObservationField(model: SplatModel, paths: ViewerScanPaths) async {
        let url: URL
        if let relative = model.observedDirectionsPath {
            url = paths.url(relative)
        } else {
            url = paths.observedDirectionsBin
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            hasObservationField = false
            observationField = nil
            directionCellBuffer = nil
            observationFieldProblem =
                "This model has no record of which directions the scene was looked at from, "
                + "so nothing can be marked as invented yet."
            return
        }
        await loadObservationField(from: url)
    }

    func loadObservationField(from url: URL) async {
        // Deliberately returns a Sendable pair rather than a `Result` with an
        // `any Error` payload: `any Error` is not Sendable, so a Result would
        // be a concurrency warning today and an error under a future language
        // mode, for no benefit - the message is all the caller shows anyway.
        let outcome = await Task.detached(priority: .userInitiated) {
            () -> (field: ObservedDirectionField?, problem: String?) in
            do {
                return (try ObservedDirectionField.read(from: url), nil)
            } catch {
                return (nil, error.localizedDescription)
            }
        }.value

        if let field = outcome.field {
            adopt(field)
        } else {
            hasObservationField = false
            observationField = nil
            directionCellBuffer = nil
            observationFieldProblem = outcome.problem
            ViewerLog.renderer.error(
                "observation field: \(outcome.problem ?? "unknown", privacy: .public)"
            )
        }
    }

    /// Adopts a field built in memory (by `ObservedDirectionBuilder`) without
    /// a round trip through disk.
    func adopt(_ field: ObservedDirectionField) {
        observationField = field
        observationFieldProblem = nil

        guard let device, field.cellCount > 0 else {
            hasObservationField = false
            directionCellBuffer = nil
            if field.cellCount == 0 {
                observationFieldProblem =
                    "The observation record for this scan is empty, so nothing can be "
                    + "marked as invented."
            }
            return
        }

        // Interleaved (key, mask) pairs: exactly the `ulong2` array the
        // fragment shader binary-searches.
        var packed = [UInt64]()
        packed.reserveCapacity(field.cellCount * 2)
        for i in 0..<field.cellCount {
            packed.append(field.keys[i])
            packed.append(field.masks[i])
        }
        let length = packed.count * MemoryLayout<UInt64>.stride
        guard let buffer = device.makeBuffer(length: length, options: [.storageModeShared]) else {
            hasObservationField = false
            directionCellBuffer = nil
            observationFieldProblem =
                "There was not enough memory to load this scan's observation record."
            return
        }
        buffer.label = "viewer.observedDirections"
        packed.withUnsafeBytes { raw in
            guard let source = raw.baseAddress else { return }
            buffer.contents().copyMemory(from: source, byteCount: length)
        }
        directionCellBuffer = buffer
        hasObservationField = true
    }

    // MARK: - SplatRenderer: draw

    public func renderFrame() {
        guard startupProblem == nil,
              let commandQueue,
              let view = attachedView,
              let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor
        else { return }

        drawableSize = view.drawableSize
        let width = Int(view.drawableSize.width)
        let height = Int(view.drawableSize.height)
        guard width > 0, height > 0 else { return }

        advanceResidency()
        ensureTargets(width: width, height: height)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "viewer.frame"

        if hasContent, residentSplatCount > 0 {
            encodeSplatPasses(into: commandBuffer, width: width, height: height)
        } else {
            clearTargets(in: commandBuffer)
        }

        encodeComposite(
            into: commandBuffer,
            descriptor: descriptor,
            pixelFormat: view.colorPixelFormat,
            width: width,
            height: height
        )

        // The visible count of THIS frame is only readable after the GPU is
        // done, so the overlay shows the previous frame's number and says so.
        if let counterBuffer {
            commandBuffer.addCompletedHandler { [weak self] _ in
                let value = counterBuffer.contents()
                    .assumingMemoryBound(to: UInt32.self).pointee
                Task { @MainActor [weak self] in
                    self?.lastVisibleSplatCount = Int(value)
                }
            }
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func advanceResidency() {
        guard hasContent, residentSplatCount < totalSplatCount else { return }
        residentSplatCount = Swift.min(totalSplatCount, residentSplatCount + residencyStep)
        paddedSortCount = ViewerMath.nextPowerOfTwo(residentSplatCount)
    }

    private func ensureTargets(width: Int, height: Int) {
        let size = CGSize(width: width, height: height)
        if targetSize == size, colorTexture != nil, auxTexture != nil { return }
        guard let device else { return }

        func makeTarget(_ label: String) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float,
                width: width,
                height: height,
                mipmapped: false
            )
            descriptor.usage = [.renderTarget, .shaderRead]
            descriptor.storageMode = .private
            let texture = device.makeTexture(descriptor: descriptor)
            texture?.label = label
            return texture
        }

        colorTexture = makeTarget("viewer.colorTarget")
        auxTexture = makeTarget("viewer.auxTarget")
        targetSize = size
    }

    private func uniforms(width: Int, height: Int) -> ViewerUniforms {
        var u = ViewerUniforms()
        u.viewMatrix = cameraPose.matrix
        let centre = cameraPose.center.simd
        u.camPos = SIMD4<Float>(centre, 1)

        let render = renderIntrinsics(width: width, height: height)
        u.focal = SIMD2<Float>(render.fx, render.fy)
        u.principal = SIMD2<Float>(render.cx, render.cy)
        u.viewportPx = SIMD2<Float>(Float(width), Float(height))
        u.nearZ = 0.05
        u.farZ = 200
        u.splatCount = UInt32(residentSplatCount)
        u.paddedCount = UInt32(paddedSortCount)
        u.shDegree = UInt32(shDegree.rawValue)
        u.shRestCount = UInt32(shRestCount)
        u.alphaCutoff = 1.0 / 255.0
        u.scaleBoost = 1
        // Matches the trainer's `camera.filter2DVariance` exactly. See
        // `ViewerUniforms.filterVariancePx` for why the two must not drift.
        u.filterVariancePx = 0.25
        return u
    }

    /// The intrinsics the render actually uses, derived from the ones handed
    /// to `setCamera` and the drawable's real size.
    ///
    /// The principal point is centred and only the focal length is derived,
    /// because a preview whose principal point sits where the phone's sensor
    /// put it looks subtly off-axis on a differently shaped drawable. The
    /// pixel aspect from the source camera IS preserved (fy is scaled by the
    /// source's fy/fx), so a non-square-pixel capture does not get stretched.
    func renderIntrinsics(width: Int, height: Int) -> CameraIntrinsics {
        let source = sourceIntrinsics
            ?? CameraIntrinsics(
                width: width,
                height: height,
                fx: Float(width) * 0.8,
                fy: Float(width) * 0.8,
                cx: Float(width) / 2,
                cy: Float(height) / 2
            )

        let targetFOV: Float
        switch framingMode {
        case .matchHorizontalFOV:
            targetFOV = source.horizontalFOVDegrees
        case .horizontalFOV(let degrees):
            targetFOV = ViewerMath.clamp(degrees, 20, 170)
        }

        let halfRadians = targetFOV * .pi / 360
        let fx = Float(width) * 0.5 / Swift.max(tan(halfRadians), 1e-4)
        let aspectRatioOfPixels = source.fx > 0 ? source.fy / source.fx : 1
        return CameraIntrinsics(
            width: width,
            height: height,
            fx: fx,
            fy: fx * aspectRatioOfPixels,
            cx: Float(width) / 2,
            cy: Float(height) / 2
        )
    }

    private func encodeSplatPasses(
        into commandBuffer: MTLCommandBuffer,
        width: Int,
        height: Int
    ) {
        guard let baseBuffer,
              let drawBuffer,
              let sortBuffer,
              let counterBuffer,
              let indirectBuffer,
              let emptyBuffer,
              let preprocessPipeline,
              let indirectPipeline,
              let rasterPipeline,
              let colorTexture,
              let auxTexture
        else { return }

        var u = uniforms(width: width, height: height)

        // 0. Zero the visible counter.
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "viewer.clearCounter"
            blit.fill(
                buffer: counterBuffer,
                range: 0..<MemoryLayout<UInt32>.stride,
                value: 0
            )
            blit.endEncoding()
        }

        // 1. Preprocess.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "viewer.preprocess"
            encoder.setComputePipelineState(preprocessPipeline)
            encoder.setBuffer(baseBuffer, offset: 0, index: ViewerBufferIndex.baseSplats)
            encoder.setBuffer(
                shRestBuffer ?? emptyBuffer,
                offset: 0,
                index: ViewerBufferIndex.shRest
            )
            encoder.setBuffer(drawBuffer, offset: 0, index: ViewerBufferIndex.draws)
            encoder.setBuffer(sortBuffer, offset: 0, index: ViewerBufferIndex.sortEntries)
            encoder.setBuffer(counterBuffer, offset: 0, index: ViewerBufferIndex.visibleCounter)
            encoder.setBytes(
                &u,
                length: MemoryLayout<ViewerUniforms>.stride,
                index: ViewerBufferIndex.uniforms
            )
            dispatch(encoder, pipeline: preprocessPipeline, threads: paddedSortCount)
            encoder.endEncoding()
        }

        // 2. Sort.
        encodeSort(into: commandBuffer, count: paddedSortCount)

        // 3. Indirect arguments.
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "viewer.indirectArgs"
            encoder.setComputePipelineState(indirectPipeline)
            encoder.setBuffer(counterBuffer, offset: 0, index: ViewerBufferIndex.indirectCounter)
            encoder.setBuffer(indirectBuffer, offset: 0, index: ViewerBufferIndex.indirectArgs)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        // 4. Raster.
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = colorTexture
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        descriptor.colorAttachments[1].texture = auxTexture
        descriptor.colorAttachments[1].loadAction = .clear
        descriptor.colorAttachments[1].storeAction = .store
        descriptor.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
        else { return }
        encoder.label = "viewer.raster"
        encoder.setRenderPipelineState(rasterPipeline)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(sortBuffer, offset: 0, index: ViewerBufferIndex.rasterSorted)
        encoder.setVertexBuffer(drawBuffer, offset: 0, index: ViewerBufferIndex.rasterDraws)
        encoder.setVertexBytes(
            &u,
            length: MemoryLayout<ViewerUniforms>.stride,
            index: ViewerBufferIndex.rasterUniforms
        )
        encoder.setFragmentBytes(&u, length: MemoryLayout<ViewerUniforms>.stride, index: 0)
        encoder.drawPrimitives(
            type: .triangleStrip,
            indirectBuffer: indirectBuffer,
            indirectBufferOffset: 0
        )
        encoder.endEncoding()
    }

    /// One bitonic sort, ascending, over `count` entries (a power of two).
    ///
    /// Every step whose partner stride fits inside a threadgroup is folded
    /// into ONE local dispatch out of threadgroup memory, which is why a 2^19
    /// sort costs about 60 dispatches instead of 190.
    private func encodeSort(into commandBuffer: MTLCommandBuffer, count: Int) {
        guard count > 1,
              let sortBuffer,
              let sortLocalPipeline,
              let sortGlobalPipeline
        else { return }

        // Threadgroup size: a power of two, no larger than the kernel allows,
        // no larger than the array, and small enough that `count * 8` bytes of
        // threadgroup memory fits.
        var tile = Swift.min(
            ViewerMath.nextPowerOfTwo(sortLocalPipeline.maxTotalThreadsPerThreadgroup + 1) / 2,
            1024
        )
        tile = Swift.min(tile, count)
        tile = Swift.max(tile, 1)
        let tileMemory = tile * MemoryLayout<ViewerSortEntry>.stride
        let canUseLocal = tile >= 2 && tileMemory <= 32 * 1024

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "viewer.sort"

        var k = 2
        while k <= count {
            var j = k / 2
            while j > 0 {
                if canUseLocal && j < tile {
                    // This dispatch consumes every remaining j for this k.
                    encoder.setComputePipelineState(sortLocalPipeline)
                    encoder.setBuffer(sortBuffer, offset: 0, index: ViewerBufferIndex.sortEntriesIn)
                    var kj = SIMD2<UInt32>(UInt32(k), UInt32(j))
                    encoder.setBytes(
                        &kj,
                        length: MemoryLayout<SIMD2<UInt32>>.stride,
                        index: ViewerBufferIndex.sortParams
                    )
                    encoder.setThreadgroupMemoryLength(tileMemory, index: 0)
                    encoder.dispatchThreads(
                        MTLSize(width: count, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tile, height: 1, depth: 1)
                    )
                    j = 0
                } else {
                    encoder.setComputePipelineState(sortGlobalPipeline)
                    encoder.setBuffer(sortBuffer, offset: 0, index: ViewerBufferIndex.sortEntriesIn)
                    var kj = SIMD2<UInt32>(UInt32(k), UInt32(j))
                    encoder.setBytes(
                        &kj,
                        length: MemoryLayout<SIMD2<UInt32>>.stride,
                        index: ViewerBufferIndex.sortParams
                    )
                    dispatch(encoder, pipeline: sortGlobalPipeline, threads: count)
                    j >>= 1
                }
            }
            k <<= 1
        }

        encoder.endEncoding()
    }

    private func clearTargets(in commandBuffer: MTLCommandBuffer) {
        guard let colorTexture, let auxTexture else { return }
        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = colorTexture
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        descriptor.colorAttachments[1].texture = auxTexture
        descriptor.colorAttachments[1].loadAction = .clear
        descriptor.colorAttachments[1].storeAction = .store
        descriptor.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)?.endEncoding()
    }

    private func encodeComposite(
        into commandBuffer: MTLCommandBuffer,
        descriptor: MTLRenderPassDescriptor,
        pixelFormat: MTLPixelFormat,
        width: Int,
        height: Int
    ) {
        guard let pipeline = compositePipeline(for: pixelFormat),
              let colorTexture,
              let auxTexture,
              let emptyBuffer
        else { return }

        var u = ViewerCompositeUniforms()
        u.invView = cameraPose.matrix.inverse
        u.camPos = SIMD4<Float>(cameraPose.center.simd, 1)
        let render = renderIntrinsics(width: width, height: height)
        u.focal = SIMD2<Float>(render.fx, render.fy)
        u.principal = SIMD2<Float>(render.cx, render.cy)
        u.viewportPx = SIMD2<Float>(Float(width), Float(height))
        u.heatmapGain = heatmapGain
        // About five points between stripes, whatever the screen's density.
        let screenScale = Float(attachedView?.contentScaleFactor ?? 3)
        u.hatchPitchPx = Swift.max(screenScale, 1) * 5

        var flags: UInt32 = 0
        if honestyMaskEnabled { flags |= ViewerCompositeUniforms.flagHonestyMask }
        if artifactHeatmapEnabled { flags |= ViewerCompositeUniforms.flagArtifactHeatmap }
        if let field = observationField, hasObservationField, directionCellBuffer != nil {
            flags |= ViewerCompositeUniforms.flagFieldLoaded
            u.gridOrigin = SIMD4<Float>(field.origin, 0)
            u.voxelSizeMeters = field.voxelSizeMeters
            u.cellCount = UInt32(field.cellCount)
        }
        u.flags = flags

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
        else { return }
        encoder.label = "viewer.composite"
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(colorTexture, index: ViewerTextureIndex.compositeColor)
        encoder.setFragmentTexture(auxTexture, index: ViewerTextureIndex.compositeAux)
        encoder.setFragmentBytes(
            &u,
            length: MemoryLayout<ViewerCompositeUniforms>.stride,
            index: ViewerBufferIndex.compositeUniforms
        )
        encoder.setFragmentBuffer(
            directionCellBuffer ?? emptyBuffer,
            offset: 0,
            index: ViewerBufferIndex.compositeDirectionCells
        )
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    private func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        threads: Int
    ) {
        guard threads > 0 else { return }
        let width = Swift.min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Swift.max(width, 1), height: 1, depth: 1)
        )
    }

    // MARK: - Export lookup

    /// The splat-file reader. `Sources/Export` is a required module, so this
    /// is expected to succeed; going through the registry first means an
    /// integrator can swap the implementation without touching the Viewer.
    /// Internal rather than private because the export screen needs exactly
    /// the same lookup, and two copies of "which exporter is this build using"
    /// is one copy too many.
    static func exporter() -> (any SplatExporting)? {
        if let registered = NimbusServices.shared.exporter { return registered }
        return ExportService()
    }
}

// MARK: - CPU-side upload preparation

/// A `SplatCloud` flattened into exactly the bytes the GPU wants.
///
/// Built off the main actor. `Sendable` because it is nothing but arrays of
/// trivial values.
struct ViewerCloudUpload: Sendable {
    var bases: [ViewerSplatBase]
    /// Flat SH rest coefficients, three floats per coefficient. Deliberately
    /// NOT `[SIMD3<Float>]`: MSL's `packed_float3` is 12 bytes and Swift's
    /// `SIMD3<Float>` is 16, so an array of the latter would be silently
    /// mis-strided on the GPU.
    var shRest: [Float]
    var shDegree: SHDegree
    var restCoefficientsPerSplat: Int
    var bounds: BoundingBox

    /// Flattens a cloud, reordering it so that any PREFIX of `bases` is a
    /// spatially uniform sample of the whole scene.
    ///
    /// The order is by bit-reversed index (a van der Corput sequence over the
    /// index space). Splat files are written in whatever order the trainer
    /// held them, which for a densified 3DGS run is strongly spatially
    /// clustered; drawing the first 10% of that order shows one corner of the
    /// room and nothing else. Drawing the first 10% of this order shows the
    /// whole room, thinly, which is what "enough is resident to draw
    /// something" should look like.
    static func prepare(_ cloud: SplatCloud) -> ViewerCloudUpload {
        let count = cloud.count
        let restPerSplat = cloud.shDegree.restCoefficientCount
        let hasRest = restPerSplat > 0 && cloud.shRest.count == count

        var bases = [ViewerSplatBase](repeating: ViewerSplatBase(), count: count)
        var shRest = [Float]()
        if hasRest {
            shRest.reserveCapacity(count * restPerSplat * 3)
        }

        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var sawFinite = false

        let order = bitReversedOrder(count: count)

        for (destination, source) in order.enumerated() {
            let p = cloud.positions[source]
            var base = ViewerSplatBase()
            base.positionX = p.x
            base.positionY = p.y
            base.positionZ = p.z
            base.opacityLogit = cloud.opacityLogits[source]
            let r = cloud.rotations[source]
            base.rotationX = r.x
            base.rotationY = r.y
            base.rotationZ = r.z
            base.rotationW = r.w
            let s = cloud.logScales[source]
            base.logScaleX = s.x
            base.logScaleY = s.y
            base.logScaleZ = s.z
            let c = cloud.colorDC[source]
            base.colorDCR = c.x
            base.colorDCG = c.y
            base.colorDCB = c.z

            if hasRest {
                base.shRestOffset = UInt32(destination * restPerSplat)
                for coefficient in cloud.shRest[source] {
                    shRest.append(coefficient.x)
                    shRest.append(coefficient.y)
                    shRest.append(coefficient.z)
                }
            } else {
                base.shRestOffset = .max
            }

            bases[destination] = base

            if p.x.isFinite && p.y.isFinite && p.z.isFinite {
                lo = simd_min(lo, p)
                hi = simd_max(hi, p)
                sawFinite = true
            }
        }

        let bounds: BoundingBox
        if sawFinite {
            bounds = BoundingBox(min: Vector3(lo), max: Vector3(hi))
        } else {
            bounds = BoundingBox(min: Vector3(-1, -1, -1), max: Vector3(1, 1, 1))
        }

        return ViewerCloudUpload(
            bases: bases,
            shRest: shRest,
            shDegree: hasRest ? cloud.shDegree : .zero,
            restCoefficientsPerSplat: hasRest ? restPerSplat : 0,
            bounds: bounds
        )
    }

    /// Indices 0..<count, ordered by the bit reversal of the index within the
    /// next power of two. Deterministic, allocation-light, and O(n).
    static func bitReversedOrder(count: Int) -> [Int] {
        guard count > 1 else { return count == 1 ? [0] : [] }
        let padded = ViewerMath.nextPowerOfTwo(count)
        let bits = padded.trailingZeroBitCount
        var order = [Int]()
        order.reserveCapacity(count)
        for i in 0..<padded {
            var reversed = 0
            var value = i
            for _ in 0..<bits {
                reversed = (reversed << 1) | (value & 1)
                value >>= 1
            }
            if reversed < count { order.append(reversed) }
        }
        return order
    }
}
