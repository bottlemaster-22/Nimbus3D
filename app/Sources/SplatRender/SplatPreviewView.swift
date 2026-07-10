//
//  SplatPreviewView.swift — SplatRender module.
//
//  SwiftUI live orbit/preview for a trained SplatModel, backed by an MTKView that
//  drives MetalSplatRenderer each frame. Drag to orbit, pinch to zoom. The camera
//  frames the model's bounding box automatically once it loads.
//
//  Usage:
//     SplatPreviewView(model: someSplatModel)
//

import SwiftUI
import MetalKit
import UIKit
import simd

public struct SplatPreviewView: View {
    private let model: SplatModel?

    public init(model: SplatModel?) {
        self.model = model
    }

    public var body: some View {
        if MTLCreateSystemDefaultDevice() != nil {
            SplatMetalView(model: model)
                .ignoresSafeArea()
        } else {
            ContentUnavailableView(
                "Metal Unavailable",
                systemImage: "cube.transparent",
                description: Text("This device does not provide a Metal renderer for the splat preview.")
            )
        }
    }
}

// MARK: - UIViewRepresentable bridge

private struct SplatMetalView: UIViewRepresentable {
    let model: SplatModel?

    func makeCoordinator() -> Coordinator {
        Coordinator(model: model)
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true
        view.isOpaque = true
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.preferredFramesPerSecond = 60
        view.delegate = context.coordinator

        let pan = UIPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handlePan(_:)))
        let pinch = UIPinchGestureRecognizer(target: context.coordinator,
                                             action: #selector(Coordinator.handlePinch(_:)))
        view.addGestureRecognizer(pan)
        view.addGestureRecognizer(pinch)

        context.coordinator.attach(view: view)
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.update(model: model)
    }

    static func dismantleUIView(_ uiView: MTKView, coordinator: Coordinator) {
        coordinator.tearDown()
    }

    // MARK: Coordinator (owns the renderer + camera state; MainActor by MTKView contract)

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        private var renderer: MetalSplatRenderer?
        private var loadedModelID: UUID?
        private var pendingModel: SplatModel?
        private var loadTask: Task<Void, Never>?

        // Orbit camera state.
        private var yaw: Float = 0.3
        private var pitch: Float = 0.2
        private var distance: Float = 3.0
        private var target = SIMD3<Float>(0, 0, 0)
        private var defaultDistance: Float = 3.0

        private var drawableSize = CGSize(width: 1, height: 1)
        private var lastPan: CGPoint = .zero

        init(model: SplatModel?) {
            self.pendingModel = model
            super.init()
            self.renderer = try? MetalSplatRenderer()
        }

        func attach(view: MTKView) {
            drawableSize = view.drawableSize
            if let model = pendingModel {
                pendingModel = nil
                loadModel(model)
            }
        }

        func update(model: SplatModel?) {
            guard let model else { return }
            if model.id != loadedModelID { loadModel(model) }
        }

        func tearDown() {
            loadTask?.cancel()
            renderer?.unload()
            renderer = nil
        }

        private func loadModel(_ model: SplatModel) {
            guard let renderer else { return }
            loadedModelID = model.id
            loadTask?.cancel()
            loadTask = Task { [weak self] in
                do {
                    try await renderer.load(model)
                } catch {
                    // Preview stays on the clear background if load fails.
                    return
                }
                await MainActor.run { self?.frameCameraToModel() }
            }
        }

        private func frameCameraToModel() {
            guard let box = renderer?.boundingBox else { return }
            target = box.center
            let radius = max(0.001, simd_length(box.extents) * 0.5)
            defaultDistance = radius * 2.5
            distance = defaultDistance
        }

        // MARK: Gestures

        @objc func handlePan(_ gr: UIPanGestureRecognizer) {
            let p = gr.translation(in: gr.view)
            if gr.state == .began { lastPan = .zero }
            let dx = Float(p.x - lastPan.x)
            let dy = Float(p.y - lastPan.y)
            lastPan = p
            yaw -= dx * 0.01
            pitch -= dy * 0.01
            let limit = Float.pi / 2 - 0.01
            pitch = min(max(pitch, -limit), limit)
        }

        @objc func handlePinch(_ gr: UIPinchGestureRecognizer) {
            if gr.state == .changed {
                distance /= Float(gr.scale)
                gr.scale = 1
                distance = min(max(distance, defaultDistance * 0.1), defaultDistance * 10)
            }
        }

        // MARK: MTKViewDelegate

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            drawableSize = size
        }

        func draw(in view: MTKView) {
            guard let renderer,
                  let drawable = view.currentDrawable else { return }
            let width = max(1, Int(view.drawableSize.width))
            let height = max(1, Int(view.drawableSize.height))

            let eye = orbitEye()
            let viewMatrix = SplatMath.lookAt(eye: eye, center: target, up: SIMD3<Float>(0, 1, 0))
            let aspect = Float(width) / Float(height)
            let proj = SplatMath.perspective(fovyRadians: 60 * .pi / 180,
                                             aspect: aspect,
                                             near: 0.01,
                                             far: 1000)

            try? renderer.render(to: drawable,
                                 viewMatrix: viewMatrix,
                                 projectionMatrix: proj,
                                 viewportSize: SIMD2<UInt32>(UInt32(width), UInt32(height)))
        }

        private func orbitEye() -> SIMD3<Float> {
            let cp = cos(pitch)
            let dir = SIMD3<Float>(cp * sin(yaw), sin(pitch), cp * cos(yaw))
            return target + dir * distance
        }
    }
}
