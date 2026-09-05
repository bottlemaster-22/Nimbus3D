//
//  SplatPreviewView.swift
//  Viewer
//
//  THE SWIFTUI SURFACE THE RENDERER DRAWS INTO, AND THE HANDS THAT MOVE IT.
//
//  Two camera modes, because the review UX genuinely needs both:
//
//    .freeOrbit    orbit / pan / pinch around the scene, for poking at it
//    .path         a fly-through pinned to where the user actually walked
//                  (F9), which is the mode a review starts in
//
//  The path mode is the one that matters. A preview camera that wanders off
//  the walked path shows the user artefacts they will never see again and
//  cannot fix, so the camera is smoothed along the path with a spline and then
//  pulled back to within `PreviewCameraPath.maxDeviationMeters` (~20 cm) of
//  the real track. Free orbit exists for inspection and is clearly a different
//  mode, not a slippery slope out of the first one.
//

import Foundation
import MetalKit
import SwiftUI
import UIKit
import simd

// MARK: - Camera controller

/// The preview camera. Owns both modes and hands the renderer one `Pose` per
/// frame.
@MainActor
final class ViewerCameraController: ObservableObject {

    enum Mode: Equatable {
        case freeOrbit
        case path
    }

    @Published private(set) var mode: Mode = .freeOrbit

    // MARK: Orbit state

    /// Point the orbit revolves around, world space.
    private(set) var target = SIMD3<Float>(0, 0, 0)
    /// Distance from `target`, metres.
    private(set) var distance: Float = 3
    /// Rotation about world Y, radians.
    private(set) var yaw: Float = 0
    /// Elevation above the horizontal, radians, clamped away from the poles so
    /// the up vector never degenerates.
    private(set) var pitch: Float = 0.2

    private var minDistance: Float = 0.15
    private var maxDistance: Float = 400

    // MARK: Path state

    private(set) var path: PreviewCameraPath?
    /// Position along the fly-through, seconds.
    private(set) var pathTime: Double = 0
    private(set) var isPlaying = false
    /// Metres/second of path time per second of wall clock. 1 replays the walk
    /// at the speed it was walked.
    var playbackRate: Double = 1

    /// The pose the renderer should use next frame.
    private(set) var pose: Pose = .identity

    /// Horizontal field of view the renderer should frame with.
    private(set) var horizontalFOVDegrees: Float = 60

    /// The captured frame nearest the current path position, so the A/B
    /// slider knows which photo to put beside the render.
    private(set) var currentSourceFrame: FrameID?

    /// The capture camera, when the scan is known. Only its PIXEL ASPECT is
    /// used by the renderer (the field of view comes from
    /// `horizontalFOVDegrees`), so a non-square-pixel capture is not stretched
    /// in the preview. Nil is fine: the renderer falls back to square pixels.
    var sourceIntrinsics: CameraIntrinsics?

    // MARK: - Framing

    /// Frames a bounding box in free-orbit mode.
    func frame(bounds: BoundingBox, fovDegrees: Float = 60) {
        let lo = bounds.min.simd
        let hi = bounds.max.simd
        let centre = (lo + hi) * 0.5
        let radius = Swift.max(simd_length(hi - lo) * 0.5, 0.25)
        target = centre
        horizontalFOVDegrees = fovDegrees
        let halfFOV = fovDegrees * .pi / 360
        distance = ViewerMath.clamp(radius / Swift.max(tan(halfFOV), 0.05) * 1.15, 0.3, maxDistance)
        maxDistance = Swift.max(distance * 12, 40)
        minDistance = Swift.max(radius * 0.02, 0.1)
        yaw = 0.6
        pitch = 0.25
        mode = .freeOrbit
        recompute()
    }

    /// Switches to the walked-path fly-through.
    ///
    /// The keyframe poses must ALREADY be rolled the right way up, with
    /// `Pose.rolledForDisplay(quarterTurnsClockwise:)`. A capture pose is in
    /// the sensor's landscape frame however the phone was held, so a portrait
    /// scan replayed raw draws the room on its side. It is done once where the
    /// path is made (`PreviewCameraPathBuilder.build`, and `ScanReviewModel`
    /// for the single-frame compare path) rather than here, so the sampler,
    /// the deviation measurement and the switch back to free look all see one
    /// consistent set of poses. Do not roll it a second time in `recompute()`:
    /// that would turn the picture through half a turn instead of a quarter.
    func adopt(path: PreviewCameraPath) {
        self.path = path
        pathTime = path.keyframes.first?.timeSeconds ?? 0
        horizontalFOVDegrees = path.horizontalFOVDegrees
        mode = .path
        isPlaying = true
        recompute()
    }

    func setMode(_ newMode: Mode) {
        guard newMode != mode else { return }
        if newMode == .path && path == nil { return }
        if newMode == .freeOrbit {
            // Start the orbit from wherever the fly-through had got to, so the
            // switch does not teleport the user.
            let eye = pose.center.simd
            let forward = pose.forward.simd
            target = eye + forward * Swift.max(distance, 2)
            distance = Swift.max(distance, 2)
            let flat = SIMD3<Float>(forward.x, 0, forward.z)
            if simd_length(flat) > 1e-4 {
                yaw = atan2(-flat.x, -flat.z)
            }
            pitch = ViewerMath.clamp(asin(ViewerMath.clamp(-forward.y, -1, 1)), -1.4, 1.4)
        }
        mode = newMode
        recompute()
    }

    var pathDuration: Double {
        guard let path, let last = path.keyframes.last, let first = path.keyframes.first
        else { return 0 }
        return Swift.max(0, last.timeSeconds - first.timeSeconds)
    }

    var pathProgress: Double {
        guard let first = path?.keyframes.first, pathDuration > 0 else { return 0 }
        return ViewerMath.clamp((pathTime - first.timeSeconds) / pathDuration, 0, 1)
    }

    func setPlaying(_ playing: Bool) {
        isPlaying = playing
    }

    func seek(toProgress progress: Double) {
        guard let first = path?.keyframes.first else { return }
        pathTime = first.timeSeconds + ViewerMath.clamp(progress, 0, 1) * pathDuration
        recompute()
    }

    /// Advances the fly-through. Called once per displayed frame with the real
    /// elapsed time, never with a fixed 1/60 - a dropped frame should not slow
    /// the walk down.
    func advance(by seconds: Double) {
        guard mode == .path, isPlaying, let path, let first = path.keyframes.first
        else { return }
        pathTime += seconds * playbackRate
        let end = first.timeSeconds + pathDuration
        if pathTime > end { pathTime = first.timeSeconds }
        recompute()
    }

    // MARK: - Gestures

    func orbit(deltaX: Float, deltaY: Float) {
        guard mode == .freeOrbit else { return }
        yaw -= deltaX * 0.006
        pitch = ViewerMath.clamp(pitch + deltaY * 0.006, -1.45, 1.45)
        recompute()
    }

    func pan(deltaX: Float, deltaY: Float) {
        guard mode == .freeOrbit else { return }
        // Pan in the camera's own plane, scaled by distance so the scene
        // tracks the finger at any zoom.
        let basis = ViewerPoseMath.basis(eye: eye(), target: target)
        let scale = distance * 0.0018
        target += basis.right * (-deltaX * scale) + basis.down * (-deltaY * scale)
        recompute()
    }

    func dolly(scale: Float) {
        guard mode == .freeOrbit, scale > 0 else { return }
        distance = ViewerMath.clamp(distance / scale, minDistance, maxDistance)
        recompute()
    }

    // MARK: - Pose

    private func eye() -> SIMD3<Float> {
        let cosPitch = cos(pitch)
        let offset = SIMD3<Float>(
            cosPitch * sin(yaw),
            sin(pitch),
            cosPitch * cos(yaw)
        )
        return target + offset * distance
    }

    private func recompute() {
        switch mode {
        case .freeOrbit:
            pose = ViewerPoseMath.lookAt(eye: eye(), target: target)
            currentSourceFrame = nil
        case .path:
            guard let path, !path.keyframes.isEmpty else { return }
            let sample = PreviewPathSampler.sample(path, atTime: pathTime)
            pose = sample.pose
            currentSourceFrame = sample.sourceFrame
            horizontalFOVDegrees = path.horizontalFOVDegrees
        }
    }
}

// MARK: - The Metal surface

/// A SwiftUI view that hosts an `MTKView` driven by `MetalSplatRenderer`.
///
/// The renderer is owned by the caller (usually the review screen), not by
/// this view, because a review screen swaps between a fly-through, an A/B
/// comparison and a training preview and all three want the same loaded cloud.
struct SplatPreviewView: UIViewRepresentable {

    let renderer: MetalSplatRenderer
    @ObservedObject var camera: ViewerCameraController
    /// Continuous redraw. On for the fly-through and the live training
    /// preview; off for a still comparison, where redrawing a static image 60
    /// times a second is just a battery bill.
    var isAnimating: Bool = true
    /// Whether touch gestures move the camera. Off in path mode by default.
    var gesturesEnabled: Bool = true

    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer, camera: camera)
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: nil)
        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .invalid
        view.framebufferOnly = true
        view.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.06, alpha: 1)
        view.isOpaque = true
        view.preferredFramesPerSecond = 60
        view.delegate = context.coordinator
        renderer.attach(to: view)

        context.coordinator.install(on: view)
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {
        context.coordinator.camera = camera
        context.coordinator.gesturesEnabled = gesturesEnabled
        view.isPaused = !isAnimating
        view.enableSetNeedsDisplay = !isAnimating
        if !isAnimating {
            view.setNeedsDisplay()
        }
    }

    static func dismantleUIView(_ view: MTKView, coordinator: Coordinator) {
        view.delegate = nil
        coordinator.renderer.detach(from: view)
    }

    // MARK: Coordinator

    /// Deliberately NOT `@MainActor`.
    ///
    /// `MTKViewDelegate` and `UIGestureRecognizer` actions are not declared
    /// with an actor, so satisfying them with main-actor-isolated methods is a
    /// diagnostic that varies by compiler version. Every one of these
    /// callbacks genuinely does arrive on the main thread, so the honest and
    /// portable spelling is to say so with `MainActor.assumeIsolated` rather
    /// than to annotate the class and hope.
    final class Coordinator: NSObject, MTKViewDelegate, UIGestureRecognizerDelegate {
        let renderer: MetalSplatRenderer
        var camera: ViewerCameraController
        var gesturesEnabled = true

        private var lastFrameTime: CFTimeInterval?

        init(renderer: MetalSplatRenderer, camera: ViewerCameraController) {
            self.renderer = renderer
            self.camera = camera
        }

        func install(on view: MTKView) {
            let orbit = UIPanGestureRecognizer(target: self, action: #selector(handleOrbit(_:)))
            orbit.maximumNumberOfTouches = 1
            orbit.delegate = self
            view.addGestureRecognizer(orbit)

            let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            pan.minimumNumberOfTouches = 2
            pan.maximumNumberOfTouches = 2
            pan.delegate = self
            view.addGestureRecognizer(pan)

            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            pinch.delegate = self
            view.addGestureRecognizer(pinch)
        }

        // MARK: MTKViewDelegate

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            MainActor.assumeIsolated {
                renderer.setDrawableSize(size)
            }
        }

        func draw(in view: MTKView) {
            let now = CACurrentMediaTime()
            let elapsed = lastFrameTime.map { now - $0 } ?? 0
            lastFrameTime = now

            let width = Swift.max(Int(view.drawableSize.width), 1)
            let height = Swift.max(Int(view.drawableSize.height), 1)

            MainActor.assumeIsolated {
                // A hitch (a load, a backgrounded app) must not fast-forward
                // the fly-through by ten seconds.
                camera.advance(by: Swift.min(elapsed, 0.1))

                renderer.framingMode = .horizontalFOV(camera.horizontalFOVDegrees)
                let intrinsics = camera.sourceIntrinsics
                    ?? CameraIntrinsics(
                        width: width,
                        height: height,
                        fx: Float(width) * 0.8,
                        fy: Float(width) * 0.8,
                        cx: Float(width) / 2,
                        cy: Float(height) / 2
                    )
                renderer.setCamera(pose: camera.pose, intrinsics: intrinsics)
                renderer.renderFrame()
            }
        }

        // MARK: Gestures

        @objc private func handleOrbit(_ recognizer: UIPanGestureRecognizer) {
            MainActor.assumeIsolated {
                guard gesturesEnabled else { return }
                let translation = recognizer.translation(in: recognizer.view)
                camera.orbit(deltaX: Float(translation.x), deltaY: Float(translation.y))
                recognizer.setTranslation(.zero, in: recognizer.view)
                recognizer.view?.setNeedsDisplay()
            }
        }

        @objc private func handlePan(_ recognizer: UIPanGestureRecognizer) {
            MainActor.assumeIsolated {
                guard gesturesEnabled else { return }
                let translation = recognizer.translation(in: recognizer.view)
                camera.pan(deltaX: Float(translation.x), deltaY: Float(translation.y))
                recognizer.setTranslation(.zero, in: recognizer.view)
                recognizer.view?.setNeedsDisplay()
            }
        }

        @objc private func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            MainActor.assumeIsolated {
                guard gesturesEnabled else { return }
                camera.dolly(scale: Float(recognizer.scale))
                recognizer.scale = 1
                recognizer.view?.setNeedsDisplay()
            }
        }

        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
        ) -> Bool {
            true
        }
    }
}

// MARK: - Startup problem

/// What the preview area shows when the renderer could not start. Naming the
/// actual reason beats a black rectangle, which is indistinguishable from an
/// empty scan.
struct SplatPreviewUnavailableView: View {
    let reason: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)
            Text("The preview could not start")
                .font(.headline)
            Text(reason)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(white: 0.08))
    }
}
