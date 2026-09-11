//
//  ScanReviewScreen.swift
//  Viewer
//
//  THE REVIEW EXPERIENCE (F9).
//
//  What this screen is FOR: telling the user the truth about a scan they just
//  spent ten minutes walking around a room to make. Everything on it points at
//  that one job.
//
//   * The walk-through replays where they actually stood, widened to about a
//     hundred degrees so it feels like a room rather than a keyhole. It never
//     invents a viewpoint, and the measured worst deviation from the real
//     track is printed on screen rather than promised in a comment.
//
//   * The honesty mask hatches, in yellow diagonal stripes, every pixel whose
//     viewing direction was never actually observed. It is on by default and
//     it is loud on purpose. When there is no observation record, the screen
//     says the mask has nothing to go on rather than showing a clean image
//     that reads as a clean bill of health.
//
//   * The artefact heatmap tints the pixels most likely to be wrong.
//
//   * Photo versus scan puts a real photo, held out of training, next to the
//     render from that photo's exact camera, and says plainly how the photos
//     were chosen.
//
//  Nothing here flatters the scan. A viewer that flatters a bad scan wastes
//  the user's afternoon and then blames their walking.
//

import Combine
import SwiftUI
import UIKit

struct ScanReviewScreen: View {

    let summary: ScanSummary

    @StateObject private var model: ScanReviewModel
    /// Where the walk-through scrubber's knob sits. Sampled from the camera on
    /// a slow timer rather than bound to it; see the timer below.
    @State private var scrubber: Double = 0

    init(summary: ScanSummary) {
        self.summary = summary
        _model = StateObject(wrappedValue: ScanReviewModel(summary: summary))
    }

    var body: some View {
        VStack(spacing: 0) {
            previewArea
                .frame(maxWidth: .infinity)
                .background(Color.black)

            controls
        }
        .navigationTitle(summary.displayName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                NavigationLink("Share") {
                    ScanExportScreen(summary: summary, detail: model.detail)
                }
            }
        }
        .task { await model.open() }
        .onDisappear { model.close() }
    }

    // MARK: - Preview

    @ViewBuilder
    private var previewArea: some View {
        if let problem = model.renderer.startupProblem {
            SplatPreviewUnavailableView(reason: problem)
                .frame(height: 320)
        } else if model.detail?.model == nil {
            NoModelYetView(summary: summary, loadingMessage: model.loadingMessage)
                .frame(height: 320)
        } else if let problem = model.problem {
            SplatPreviewUnavailableView(reason: problem)
                .frame(height: 320)
        } else {
            surface
        }
    }

    @ViewBuilder
    private var surface: some View {
        let content = ZStack {
            SplatPreviewView(
                renderer: model.renderer,
                camera: model.camera,
                isAnimating: model.mode != .compare,
                gesturesEnabled: model.mode == .freeLook
            )

            if model.mode == .compare {
                PhotoVersusScanOverlay(model: model)
            }

            if let message = model.loadingMessage {
                LoadingBadge(message: message)
            }
        }

        if model.mode == .compare {
            content.aspectRatio(model.compareAspectRatio, contentMode: .fit)
        } else {
            content.frame(height: 360)
        }
    }

    // MARK: - Controls

    private var controls: some View {
        List {
            // FIRST, above everything, including the mode picker. When a scan
            // comes out looking like nothing, the preview above is a black
            // rectangle and this sentence is the next thing the eye lands on.
            // Putting it below the controls would mean the one screen that
            // explains the black rectangle is the one thing the user has to go
            // looking for.
            ScanCensusSummarySection(census: model.census)

            Section {
                Picker("What to show", selection: $model.mode) {
                    ForEach(ScanReviewModel.Mode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
            }

            switch model.mode {
            case .walkThrough: walkThroughSection
            case .freeLook: freeLookSection
            case .compare: compareSection
            }

            nextStepSection
            honestySection
            heatmapSection
            findingsSection
        }
        .listStyle(.insetGrouped)
    }

    // MARK: Next step

    /// The way back to the screen that runs the check-over and the build.
    ///
    /// A scan with no model needs it to get one; a scan with a model can still
    /// want a better one after a re-walk, and this is the only route to that,
    /// so the link is always here and only its wording changes.
    private var nextStepSection: some View {
        Section {
            NavigationLink {
                ScanProcessingScreen(summary: summary)
            } label: {
                Label(
                    model.detail?.model == nil
                        ? "Finish this scan"
                        : "Build this one again",
                    systemImage: "wand.and.stars"
                )
            }
        } header: {
            Text("Next step")
        } footer: {
            Text(
                model.detail?.model == nil
                    ? "Checking the scan over and building the 3D model both happen on "
                        + "this phone, and both can be started from there."
                    : "Building it again replaces the model you have now. Your photos and "
                        + "measurements are never touched."
            )
        }
    }

    // MARK: Walk-through

    @ViewBuilder
    private var walkThroughSection: some View {
        Section {
            if model.flyThroughPath != nil {
                HStack(spacing: 14) {
                    Button {
                        model.setPlaying(!model.isPlaying)
                    } label: {
                        Image(systemName: model.isPlaying ? "pause.fill" : "play.fill")
                            .font(.title3)
                            .frame(width: 34, height: 34)
                    }
                    .buttonStyle(.bordered)

                    Slider(
                        value: Binding(
                            get: { scrubber },
                            set: {
                                scrubber = $0
                                model.seek(toProgress: $0)
                            }
                        ),
                        in: 0...1
                    )
                }
                // The camera advances sixty times a second. Republishing that
                // would redraw this whole list sixty times a second for a
                // scrubber the eye reads four times a second, so the scrubber
                // samples the camera on a slow timer instead.
                .onReceive(
                    Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()
                ) { _ in
                    if model.isPlaying { scrubber = model.camera.pathProgress }
                }

                if let worst = model.worstDeviationMeters {
                    Text(deviationLine(worst))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            } else if let problem = model.flyThroughProblem {
                Text(problem)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                Text("Working out the path you walked...")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("Walk-through")
        } footer: {
            Text(
                "This replays the route you actually walked, opened up to about a "
                + "hundred degrees so you can see the room. It never moves the camera "
                + "somewhere you did not stand, because a view from a spot you never "
                + "occupied would show you problems you can neither judge nor fix."
            )
        }
    }

    private func deviationLine(_ worst: Float) -> String {
        let centimeters = worst * 100
        return "The smoothed camera never gets more than "
            + "\(ViewerFormat.centimeters(centimeters)) away from where you really stood. "
            + "That is measured along this path, not an estimate."
    }

    // MARK: Free look

    private var freeLookSection: some View {
        Section {
            Text(
                "Drag to turn, use two fingers to slide sideways, pinch to move closer "
                + "or further away."
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
        } header: {
            Text("Look around")
        } footer: {
            Text(
                "This lets you go anywhere, including places you never stood. Anything "
                + "that was never really looked at is still hatched, so you can tell "
                + "the difference."
            )
        }
    }

    // MARK: Photo versus scan

    @ViewBuilder
    private var compareSection: some View {
        Section {
            if let heldOut = model.heldOut, !heldOut.isEmpty {
                HStack {
                    Button {
                        model.stepCompare(by: -1)
                    } label: {
                        Image(systemName: "chevron.left")
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.compareIndex == 0)

                    Spacer()

                    Text("Photo \(model.compareIndex + 1) of \(heldOut.frames.count)")
                        .font(.subheadline)
                        .monospacedDigit()

                    Spacer()

                    Button {
                        model.stepCompare(by: 1)
                    } label: {
                        Image(systemName: "chevron.right")
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.compareIndex >= heldOut.frames.count - 1)
                }

                Slider(value: $model.compareWipe, in: 0...1)

                HStack {
                    Text("Photo")
                    Spacer()
                    Text("Scan")
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if let problem = model.comparePhotoProblem {
                    Text(problem)
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }

                Label {
                    Text(heldOut.explanation)
                } icon: {
                    Image(
                        systemName: heldOut.isTrustworthy
                            ? "checkmark.seal" : "questionmark.circle"
                    )
                }
                .font(.footnote)
                .foregroundStyle(.secondary)
            } else {
                Text(
                    "This scan does not have enough photos to keep any of them back, "
                    + "so there is nothing to compare against."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            }
        } header: {
            Text("Photo vs scan")
        } footer: {
            Text(
                "Drag the slider to wipe between the real photo and the 3D scan from "
                + "that same spot. They are framed the same way on purpose, so what "
                + "does not line up is a real difference and not a trick of the lens."
            )
        }
    }

    // MARK: Honesty mask

    private var honestySection: some View {
        Section {
            Toggle("Show what was never really seen", isOn: $model.honestyMaskEnabled)

            HStack(spacing: 12) {
                HonestyHatchSwatch()
                    .frame(width: 54, height: 30)
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                Text(
                    "Yellow diagonal stripes mean this app is guessing: nothing in your "
                    + "walk ever looked at that surface from anything like this angle."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            }

            if model.isBuildingHonestyRecord {
                HStack(spacing: 10) {
                    ProgressView()
                    Text(
                        model.honestyRecordNote
                            ?? "Working out which parts you really looked at..."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                }
            } else if !model.honestyMaskHasEvidence {
                Label {
                    Text(
                        model.honestyRecordNote
                            ?? "There is no record of which directions this scene was "
                                + "looked at from, so nothing can be marked as guesswork "
                                + "yet. An un-striped picture here does not mean the scan "
                                + "is clean."
                    )
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .font(.footnote)
                .foregroundStyle(.orange)
            } else if let note = model.honestyRecordNote {
                Text(note)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("Honesty")
        } footer: {
            Text(
                "Every 3D scan fills in surfaces it never saw. Most apps hide that. "
                + "This one paints over it, so you know which parts of the picture you "
                + "can trust and which parts you would need to go back and walk again."
            )
        }
    }

    // MARK: Heatmap

    private var heatmapSection: some View {
        Section {
            Toggle("Show where the scan is likely wrong", isOn: $model.artifactHeatmapEnabled)

            if model.artifactHeatmapEnabled {
                HeatScaleLegend()
            }
        } header: {
            Text("Trouble spots")
        } footer: {
            Text(
                "Warmer colours mean more of the things that go wrong in a scan piled "
                + "up on that pixel: gaps the model never filled, layers of half-"
                + "transparent haze, and surfaces nothing ever looked at."
            )
        }
    }

    // MARK: Findings

    @ViewBuilder
    private var findingsSection: some View {
        if let card = model.detail?.prePass?.qcCard, !card.findings.isEmpty {
            Section {
                ForEach(card.findings) { finding in
                    VStack(alignment: .leading, spacing: 4) {
                        Label {
                            Text(finding.message)
                                .font(.subheadline)
                        } icon: {
                            Image(systemName: icon(for: finding.severity))
                                .foregroundStyle(color(for: finding.severity))
                        }
                        if let hint = finding.fixHint {
                            Text(hint)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 2)
                }
            } header: {
                Text("How this one came out")
            }
        }
    }

    private func icon(for severity: QCFinding.Severity) -> String {
        switch severity {
        case .good: return "checkmark.circle"
        case .warning: return "info.circle"
        case .problem: return "exclamationmark.triangle"
        }
    }

    private func color(for severity: QCFinding.Severity) -> Color {
        switch severity {
        case .good: return .green
        case .warning: return .secondary
        case .problem: return .orange
        }
    }
}

// MARK: - The wipe

/// The photo half of the A/B comparison, wiped across the render underneath.
///
/// The photo is on the LEFT of the wipe and the scan on the right, which is
/// the order the labels under the slider promise. The handle can be dragged
/// directly, because reaching for a slider at the bottom of the screen while
/// looking at the top of it is exactly the kind of small friction that stops
/// people from checking their work.
struct PhotoVersusScanOverlay: View {
    @ObservedObject var model: ScanReviewModel

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let split = width * CGFloat(model.compareWipe)

            ZStack(alignment: .leading) {
                if let photo = model.comparePhoto {
                    Image(uiImage: photo)
                        .resizable()
                        .scaledToFill()
                        .frame(width: width, height: geometry.size.height)
                        .clipped()
                        .mask(alignment: .leading) {
                            Rectangle().frame(width: split)
                        }
                }

                Rectangle()
                    .fill(Color.white.opacity(0.9))
                    .frame(width: 2)
                    .offset(x: split - 1)

                Circle()
                    .fill(Color.white.opacity(0.9))
                    .frame(width: 26, height: 26)
                    .overlay {
                        Image(systemName: "arrow.left.and.right")
                            .font(.caption.weight(.bold))
                            .foregroundStyle(.black)
                    }
                    .position(x: split, y: geometry.size.height / 2)
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard width > 0 else { return }
                        model.compareWipe = ViewerMath.clamp(
                            Double(value.location.x / width),
                            0,
                            1
                        )
                    }
            )
        }
    }
}

// MARK: - Small pieces

/// A swatch of the exact hatching the renderer draws, so the legend and the
/// picture cannot drift apart in the user's head.
struct HonestyHatchSwatch: View {
    var body: some View {
        Canvas { context, size in
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.gray))
            var stripes = Path()
            let pitch: CGFloat = 8
            var x = -size.height
            while x < size.width + size.height {
                stripes.move(to: CGPoint(x: x, y: size.height))
                stripes.addLine(to: CGPoint(x: x + size.height, y: 0))
                x += pitch
            }
            context.stroke(
                stripes,
                with: .color(Color(red: 0.95, green: 0.86, blue: 0.35)),
                lineWidth: 3
            )
        }
        .accessibilityLabel("Yellow diagonal stripes over grey")
    }
}

/// The heatmap's colour ramp, matching `viewer_heat_colour` in
/// SplatRenderShaders.metal.
struct HeatScaleLegend: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            LinearGradient(
                colors: [
                    Color(red: 0.16, green: 0.42, blue: 0.85),
                    Color(red: 0.98, green: 0.72, blue: 0.18),
                    Color(red: 0.90, green: 0.16, blue: 0.16)
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
            .frame(height: 12)
            .clipShape(Capsule())

            HStack {
                Text("Looks solid")
                Spacer()
                Text("Least trustworthy")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}

/// What the preview area shows for a scan that has no 3D model yet. It names
/// the real next step instead of showing an empty black box.
struct NoModelYetView: View {
    let summary: ScanSummary
    let loadingMessage: String?

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(.secondary)
            Text("No 3D model for this scan yet")
                .font(.headline)
                .foregroundStyle(.white)
            Text(loadingMessage ?? summary.nextStep)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(white: 0.08))
    }
}

/// A small "still working" badge over the preview. Deliberately not a full
/// screen cover: the first slice of the model is already drawable underneath
/// it, and hiding that would make loading feel slower than it is.
struct LoadingBadge: View {
    let message: String

    var body: some View {
        VStack {
            Spacer()
            HStack(spacing: 8) {
                ProgressView()
                    .tint(.white)
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.white)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(.black.opacity(0.55), in: Capsule())
            .padding(.bottom, 14)
        }
    }
}
