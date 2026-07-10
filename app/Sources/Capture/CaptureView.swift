//
//  CaptureView.swift — guided object-capture screen.
//  Owned by the Capture agent.
//
//  `CaptureRootView` is the entry point the App shell can drop into the Capture tab
//  (replacing CapturePlaceholderView). It shows the live ARKit camera, a coverage ring,
//  tracking status, guidance copy, and start / phase / finish / cancel controls.
//

import SwiftUI
import ARKit
import SceneKit

// MARK: - AR camera preview

/// Wraps an ARSCNView bound to the service's shared ARSession so the preview shows
/// exactly the frames the service is tracking. The service owns run/pause lifecycle.
struct ARViewportView: UIViewRepresentable {
    let service: ARCaptureService

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = service.session
        view.automaticallyUpdatesLighting = true
        view.rendersContinuously = true
        view.scene = SCNScene()
        // Binding the session can reset delegates; make sure we still receive frames.
        service.reassertDelegate()
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - Root screen

public struct CaptureRootView: View {
    @StateObject private var model = CaptureViewModel()

    public init() {}

    public var body: some View {
        ZStack {
            if model.worldTrackingSupported {
                ARViewportView(service: model.service)
                    .ignoresSafeArea()
            } else {
                unsupportedView
            }

            overlay
        }
        .animation(.easeInOut(duration: 0.2), value: model.isCapturing)
        .animation(.easeInOut(duration: 0.2), value: model.bundle?.id)
    }

    // MARK: Overlay

    private var overlay: some View {
        VStack {
            topBar
            Spacer()
            if let error = model.errorMessage {
                banner(error)
            }
            if let bundle = model.bundle {
                CaptureSummaryCard(bundle: bundle) { model.startNewCapture() }
            } else {
                guidanceAndControls
            }
        }
        .padding()
    }

    private var topBar: some View {
        HStack(alignment: .top) {
            TrackingPill(quality: model.trackingQuality)
            Spacer()
            VStack(alignment: .trailing, spacing: 6) {
                if model.lidarSupported {
                    CaptureBadge(text: "LiDAR", systemImage: "dot.radiowaves.left.and.right")
                }
                if model.isCapturing {
                    CaptureBadge(text: "\(model.frameCount)/\(model.targetCount) frames",
                                 systemImage: "camera.aperture")
                }
            }
        }
    }

    private var guidanceAndControls: some View {
        VStack(spacing: 16) {
            if model.isCapturing {
                CoverageRing(coverage: model.coverage)
                    .frame(width: 96, height: 96)
            }

            Text(model.guidance)
                .font(.callout.weight(.medium))
                .multilineTextAlignment(.center)
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14))

            if model.isCapturing {
                activeControls
            } else {
                setupControls
            }
        }
    }

    // MARK: Controls

    private var setupControls: some View {
        VStack(spacing: 14) {
            VStack(spacing: 10) {
                Stepper("Target frames: \(model.targetFrameSetting)",
                        value: $model.targetFrameSetting, in: 40...200, step: 10)
                if model.lidarSupported {
                    Toggle("Use LiDAR depth prior", isOn: $model.captureLiDAR)
                }
                Toggle("Capture lighting for HDRI", isOn: $model.captureHDRI)
            }
            .font(.subheadline)
            .foregroundStyle(.white)
            .padding(14)
            .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 16))

            Button(action: model.start) {
                Label("Start Capture", systemImage: "record.circle")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!model.worldTrackingSupported)
        }
    }

    private var activeControls: some View {
        VStack(spacing: 14) {
            Picker("Mode", selection: phaseBinding) {
                Text("Object").tag(ARCaptureService.Phase.object)
                if model.captureHDRI {
                    Text("Environment").tag(ARCaptureService.Phase.environment)
                }
            }
            .pickerStyle(.segmented)

            HStack(spacing: 12) {
                Button(role: .cancel, action: model.cancel) {
                    Label("Cancel", systemImage: "xmark")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)

                Button(action: model.finish) {
                    Label("Finish", systemImage: "checkmark")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
        }
        .tint(.white)
    }

    private var phaseBinding: Binding<ARCaptureService.Phase> {
        Binding(get: { model.phase }, set: { model.setPhase($0) })
    }

    // MARK: Fallbacks

    private var unsupportedView: some View {
        ContentUnavailableView(
            "AR Capture Unavailable",
            systemImage: "arkit",
            description: Text("This device does not support ARKit world tracking. Capture needs an ARKit-capable iPhone.")
        )
    }

    private func banner(_ message: String) -> some View {
        Text(message)
            .font(.footnote.weight(.medium))
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding(12)
            .frame(maxWidth: .infinity)
            .background(.red.opacity(0.85), in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Subviews

/// Circular progress ring for angular coverage around the subject.
struct CoverageRing: View {
    let coverage: Float

    var body: some View {
        ZStack {
            Circle()
                .stroke(.white.opacity(0.25), lineWidth: 8)
            Circle()
                .trim(from: 0, to: CGFloat(max(0, min(1, coverage))))
                .stroke(ringColor, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.easeInOut(duration: 0.25), value: coverage)
            Text("\(Int((coverage * 100).rounded()))%")
                .font(.headline)
                .foregroundStyle(.white)
        }
    }

    private var ringColor: Color {
        switch coverage {
        case ..<0.35: return .orange
        case ..<0.85: return .yellow
        default: return .green
        }
    }
}

struct TrackingPill: View {
    let quality: TrackingQuality

    var body: some View {
        Label(text, systemImage: "circle.fill")
            .font(.caption.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(color.opacity(0.85), in: Capsule())
    }

    private var text: String {
        switch quality {
        case .notAvailable: return "No tracking"
        case .limited: return "Tracking limited"
        case .normal: return "Tracking good"
        }
    }

    private var color: Color {
        switch quality {
        case .notAvailable: return .gray
        case .limited: return .orange
        case .normal: return .green
        }
    }
}

struct CaptureBadge: View {
    let text: String
    let systemImage: String

    var body: some View {
        Label(text, systemImage: systemImage)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.black.opacity(0.55), in: Capsule())
    }
}

/// Shown after a successful capture: a short summary plus a "new capture" action.
struct CaptureSummaryCard: View {
    let bundle: CaptureBundle
    let onNewCapture: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Label("Capture Complete", systemImage: "checkmark.seal.fill")
                .font(.headline)
                .foregroundStyle(.white)

            VStack(alignment: .leading, spacing: 6) {
                row("Frames", "\(bundle.frames.count)")
                row("HDRI frames", "\(bundle.hdriBrackets.count)")
                row("LiDAR depth", bundle.hasLiDARDepth ? "Yes" : "No")
                if let radius = bundle.sceneBoundingRadius {
                    row("Object radius", String(format: "%.2f m", radius))
                }
            }
            .font(.subheadline)
            .foregroundStyle(.white)

            Text("Saved to \(bundle.rootDirectory.lastPathComponent)")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(1)
                .truncationMode(.middle)

            Button(action: onNewCapture) {
                Label("New Capture", systemImage: "plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(18)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 18))
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).fontWeight(.semibold)
        }
    }
}

#Preview {
    CaptureRootView()
}
