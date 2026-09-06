//
//  CaptureHUDComponents.swift
//  Capture
//
//  The pieces the capture HUD is assembled from.
//
//  TWO RULES HOLD THROUGHOUT THIS FILE.
//
//  1. Every colour here is the same colour the Metal shader paints the room
//     with. Violet on the wall and violet in the legend have to mean the same
//     thing or the legend is worse than no legend at all, so the values below
//     are copied from `CaptureCoverageShaders.metal` and must be changed
//     together with it.
//
//  2. Every string is something a person would say out loud. No "angular
//     spread", no "variance of Laplacian", no percentages of things nobody
//     asked about. Where a real term is unavoidable (pixels, for the blur
//     meter) it is glossed on the spot, once, in the same breath.
//

import SwiftUI

// =============================================================================
//  MARK: - Palette
// =============================================================================

/// The HUD's colours. Kept byte-identical to the shader's constants.
enum CaptureHUDPalette {
    static let satisfied = Color(red: 0.16, green: 0.84, blue: 0.40)
    static let angles = Color(red: 0.68, green: 0.36, blue: 1.00)
    static let distance = Color(red: 0.24, green: 0.60, blue: 1.00)
    static let sharpness = Color(red: 1.00, green: 0.70, blue: 0.16)
    static let unseen = Color(red: 0.62, green: 0.62, blue: 0.66)
    static let unreliable = Color(red: 0.35, green: 0.78, blue: 0.85)
    static let problem = Color(red: 1.00, green: 0.35, blue: 0.30)
}

extension CaptureCoverageChannel {

    /// Two or three words, for a segmented control.
    var shortTitle: String {
        switch self {
        case .angles: return "Sides seen"
        case .distance: return "Distance"
        case .sharpness: return "Sharpness"
        }
    }

    /// One sentence saying what the colour on the wall actually means.
    var explanation: String {
        switch self {
        case .angles:
            return "How many different directions each patch has been looked "
                + "at from. Four well spread out is plenty."
        case .distance:
            return "Whether you stood at a helpful distance: about half a step "
                + "to three steps away is where the laser and the camera agree."
        case .sharpness:
            return "How sharp the best picture of each patch was."
        }
    }

    var color: Color {
        switch self {
        case .angles: return CaptureHUDPalette.angles
        case .distance: return CaptureHUDPalette.distance
        case .sharpness: return CaptureHUDPalette.sharpness
        }
    }
}

// =============================================================================
//  MARK: - Before the scan
// =============================================================================

/// What is on screen before the first frame: what is about to happen, in
/// plain words, and one button.
@MainActor
struct CaptureStartPanel: View {

    @ObservedObject var model: CaptureScreenModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Spacer(minLength: 0)

            Text("Scan a room")
                .font(.largeTitle.weight(.bold))
                .foregroundStyle(.white)

            Text(
                "Hold the phone up and walk slowly around, keeping whatever "
                + "you care about in view for a few seconds each, from two or "
                + "three different sides. The room fills in with colour as you "
                + "go, and the app talks to you and buzzes when something "
                + "needs fixing, so you can watch the room instead of the "
                + "screen."
            )
            .font(.body)
            .foregroundStyle(.white.opacity(0.85))

            TextField("Name this scan (optional)", text: $model.scanName)
                .textFieldStyle(.plain)
                .padding(12)
                .background(Color.white.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(.white)

            VStack(alignment: .leading, spacing: 8) {
                Toggle("Talk to me", isOn: $model.soundOn)
                Toggle("Buzz when the picture smears", isOn: $model.hapticsOn)
            }
            .tint(CaptureHUDPalette.satisfied)
            .foregroundStyle(.white)

            Text(
                "These are the main way the app guides you, because you will "
                + "be looking at the room and not at the screen."
            )
            .font(.footnote)
            .foregroundStyle(.white.opacity(0.65))

            Button {
                Task { await model.startScanning() }
            } label: {
                HStack {
                    if model.stage == .starting {
                        ProgressView().tint(Color.black)
                    }
                    Text(model.stage == .starting ? "Starting" : "Start scanning")
                        .font(.title3.weight(.semibold))
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
            }
            .background(Color.white, in: RoundedRectangle(cornerRadius: 14))
            .foregroundStyle(.black)
            .disabled(model.stage == .starting)

            Spacer(minLength: 0)
        }
        .padding(24)
        .background(Color.black.opacity(0.75))
    }
}

// =============================================================================
//  MARK: - Top bar
// =============================================================================

/// Tracking, heat and elapsed time. Small, quiet, and only loud when something
/// is genuinely wrong.
@MainActor
struct CaptureTopBar: View {

    @ObservedObject var model: CaptureScreenModel

    var body: some View {
        HStack(spacing: 12) {
            Label(
                CaptureFormat.duration(model.elapsedSeconds),
                systemImage: "record.circle"
            )
            .foregroundStyle(CaptureHUDPalette.problem)

            Text("\(model.frameCount) shots")
                .foregroundStyle(.white.opacity(0.8))

            Text(CaptureFormat.bytes(model.bytesWritten))
                .foregroundStyle(.white.opacity(0.6))

            Spacer()

            if model.thermalLevel >= .serious {
                Label("Warm", systemImage: "thermometer.high")
                    .foregroundStyle(CaptureHUDPalette.sharpness)
            }

            if model.trackingQuality != .normal {
                Label("Finding its place", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(CaptureHUDPalette.sharpness)
            }
        }
        .font(.footnote.weight(.medium))
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.35))
    }
}

/// A one-sentence strip. Used for the stop reason and nothing else, so it
/// never becomes wallpaper.
struct CaptureBanner: View {

    enum Tone {
        case problem
        case good
    }

    let text: String
    let tone: Tone

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(
                systemName: tone == .problem
                    ? "exclamationmark.circle.fill"
                    : "checkmark.circle.fill"
            )
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .font(.callout.weight(.medium))
        .foregroundStyle(.white)
        .padding(14)
        .background(
            (tone == .problem ? CaptureHUDPalette.problem : CaptureHUDPalette.satisfied)
                .opacity(0.85),
            in: RoundedRectangle(cornerRadius: 12)
        )
    }
}

// =============================================================================
//  MARK: - Window mode
// =============================================================================

/// The card that appears when ARKit says the middle of the screen is a window.
///
/// A window fails three ways at once: the laser goes through it and measures
/// nothing, the glass is many times brighter than the room so the picture
/// clips to white, and there is nothing to walk around. The card says what the
/// app is doing about it and gives the user the two things only they can do.
@MainActor
struct CaptureWindowCard: View {

    @ObservedObject var model: CaptureScreenModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("There is a window here", systemImage: "window.casement")
                    .font(.headline)
                Spacer()
                Button {
                    model.dismissWindowCard()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.6))
                }
                .accessibilityLabel("Hide this")
            }

            Text(
                model.shouldStandBackFromWindow
                    ? "Step back a little so some of the wall around the "
                        + "window is in shot. The frame is what the shape of "
                        + "the window actually hangs off."
                    : "Windows are the hardest thing in a room: the laser goes "
                        + "straight through and the glass is far brighter than "
                        + "everything else."
            )
            .font(.subheadline)
            .fixedSize(horizontal: false, vertical: true)

            if model.isBracketingAvailable {
                Text(
                    "While you are here the app is also slipping in a darker "
                    + "shot now and then, which is the only way to keep any "
                    + "detail of what is outside."
                )
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.75))
                .fixedSize(horizontal: false, vertical: true)
            }

            if let note = model.windowActionNote {
                Text(note)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(CaptureHUDPalette.satisfied)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 10) {
                // A hold has to be releasable from the same button that made
                // it. Locking the exposure at a window and then walking into a
                // dark hallway with no way to let it go is a worse scan than
                // never having held it at all.
                Button {
                    if model.exposureIsLocked {
                        model.releaseExposureLock()
                    } else {
                        model.lockExposureForWindow()
                    }
                } label: {
                    Text(model.exposureIsLocked ? "Let the brightness go" : "Hold the brightness")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .background(
                    Color.white.opacity(model.exposureIsLocked ? 0.15 : 0.9),
                    in: RoundedRectangle(cornerRadius: 10)
                )
                .foregroundStyle(model.exposureIsLocked ? Color.white : Color.black)

                Button {
                    model.markWindow()
                } label: {
                    Text("Mark this window")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .background(Color.white.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(.white)
            }
            .font(.subheadline.weight(.semibold))
        }
        .foregroundStyle(.white)
        .padding(16)
        .background(CaptureHUDPalette.unreliable.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(CaptureHUDPalette.unreliable.opacity(0.8), lineWidth: 1)
        )
        .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
    }
}

// =============================================================================
//  MARK: - Bottom panel
// =============================================================================

/// The hint, the numbers, the blur meter, the finish button, and the settings
/// that are worth having but not worth reading while walking.
@MainActor
struct CaptureBottomPanel: View {

    @ObservedObject var model: CaptureScreenModel
    @State private var showsDetails = false

    var body: some View {
        VStack(spacing: 14) {
            hintLine

            CaptureBlurMeter(pixels: model.blurPixels)

            coverageLine

            if showsDetails {
                CaptureChannelPicker(model: model)
                settings
            }

            HStack(spacing: 12) {
                Button {
                    showsDetails.toggle()
                } label: {
                    Image(systemName: showsDetails ? "chevron.down" : "slider.horizontal.3")
                        .font(.title3)
                        .frame(width: 52, height: 52)
                }
                .background(Color.white.opacity(0.15), in: Circle())
                .foregroundStyle(.white)
                .accessibilityLabel(showsDetails ? "Hide settings" : "Show settings")

                finishButton

                Button {
                    Task { await model.throwAwayScan() }
                } label: {
                    Image(systemName: "trash")
                        .font(.title3)
                        .frame(width: 52, height: 52)
                }
                .background(Color.white.opacity(0.15), in: Circle())
                .foregroundStyle(.white)
                .accessibilityLabel("Throw this scan away")
                .disabled(model.stage == .finishing)
            }
        }
        .padding(16)
        .background(
            LinearGradient(
                colors: [Color.black.opacity(0), Color.black.opacity(0.75)],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    /// The single most useful sentence right now. It is also being spoken, so
    /// this is the copy you read when you stop and look down.
    private var hintLine: some View {
        Text(model.hint ?? (model.isCoverageDone ? "That is enough. Stop whenever you like." : " "))
            .font(.title3.weight(.semibold))
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, minHeight: 56)
            .shadow(radius: 6)
    }

    private var coverageLine: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text("\(Int((model.coverageFraction * 100).rounded()))%")
                .font(.system(size: 34, weight: .bold, design: .rounded))
                .foregroundStyle(
                    model.isCoverageDone ? CaptureHUDPalette.satisfied : Color.white
                )
            VStack(alignment: .leading, spacing: 2) {
                Text("covered")
                    .font(.subheadline.weight(.medium))
                Text(
                    "Enough at \(Int(CaptureTuning.coverageDoneFraction * 100))%"
                )
                .font(.caption)
                .foregroundStyle(.white.opacity(0.65))
            }
            .foregroundStyle(.white)
            Spacer()
            ProgressView(value: Double(model.coverageFraction))
                .tint(model.isCoverageDone ? CaptureHUDPalette.satisfied : Color.white)
                .frame(width: 120)
        }
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 10) {
            Toggle("Talk to me", isOn: $model.soundOn)
            Toggle("Buzz when the picture smears", isOn: $model.hapticsOn)

            // F5's switch. It lives here rather than only on the window card
            // because the card is only on screen while a window is in front of
            // you, and a user who wants the darker shots off wants them off
            // for the whole walk.
            Toggle("Slip in a darker shot for windows", isOn: $model.darkShotsOn)

            // The release for a brightness hold, in a place that stays on
            // screen after the window card has gone. The hold is made on the
            // window card, but by the time it is hurting you, you are in the
            // next room and the card is long gone.
            if model.exposureIsLocked {
                Button {
                    model.releaseExposureLock()
                } label: {
                    Text("Let the brightness adjust itself again")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .background(
                    Color.white.opacity(0.18),
                    in: RoundedRectangle(cornerRadius: 10)
                )
                .foregroundStyle(.white)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("How strong the colour is")
                    .font(.footnote)
                Slider(value: $model.overlayStrength, in: 0...1)
            }

            // Answers the one question the percentage cannot: why it is not
            // moving. It is not stuck, the room it is a percentage OF is still
            // growing as you walk into it.
            Text(
                "\(model.countedCoverageVoxels) patches counted so far. The "
                + "percentage is out of these, and walking into a new room "
                + "adds more."
            )
            .font(.caption)
            .foregroundStyle(.white.opacity(0.6))
            .fixedSize(horizontal: false, vertical: true)
        }
        .tint(CaptureHUDPalette.satisfied)
        .foregroundStyle(.white)
        .font(.subheadline)
    }

    @ViewBuilder
    private var finishButton: some View {
        Button {
            Task { await model.finishScanning() }
        } label: {
            HStack {
                if model.stage == .finishing {
                    ProgressView().tint(Color.black)
                    Text("Saving your scan")
                } else {
                    Text("I am done")
                }
            }
            .font(.title3.weight(.bold))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
        }
        .background(
            model.isCoverageDone ? CaptureHUDPalette.satisfied : Color.white,
            in: RoundedRectangle(cornerRadius: 14)
        )
        .foregroundStyle(.black)
        .disabled(model.stage == .finishing)
    }
}

// =============================================================================
//  MARK: - The blur meter
// =============================================================================

/// Pixels of smear, live.
///
/// The number is the turn rate the gyro measured multiplied by how long the
/// shutter was open, divided by how much of the picture one pixel covers
/// (0.0426 of a degree on this camera). The amber and red marks come from
/// `CaptureTuning`, which explains where they are and why: the trainer only
/// ever looks at these photographs shrunk to 720 px on the long edge, so a few
/// pixels of smear at capture size is a fraction of a pixel where it counts.
/// It is the only thing on the HUD that answers to what your hands are doing
/// this second, which is why it gets its own bar.
struct CaptureBlurMeter: View {

    let pixels: Float

    private var fraction: Double {
        Double(min(pixels, CaptureTuning.blurRedPixels * 1.5))
            / Double(CaptureTuning.blurRedPixels * 1.5)
    }

    private var color: Color {
        if pixels >= CaptureTuning.blurRedPixels { return CaptureHUDPalette.problem }
        if pixels >= CaptureTuning.blurAmberPixels { return CaptureHUDPalette.sharpness }
        return CaptureHUDPalette.satisfied
    }

    /// Describes the picture, never the person holding the phone.
    private var verdict: String {
        if pixels >= CaptureTuning.blurRedPixels { return "smearing" }
        if pixels >= CaptureTuning.blurAmberPixels { return "softening" }
        return "sharp"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Smear: \(String(format: "%.1f", pixels)) px")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(color)
                Text("(\(verdict))")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                // A gauge, not a test. The old wording stated a pass mark
                // ("under 2 is fine"), which someone with shaky hands cannot
                // clear indoors and which then reads as failing.
                Text("px means pixels of smear; lower is sharper")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.white.opacity(0.18))
                    Capsule()
                        .fill(color)
                        .frame(width: max(4, geometry.size.width * CGFloat(fraction)))
                    // The two thresholds, drawn where they actually are so the
                    // bar can be read without a legend.
                    marker(at: Double(CaptureTuning.blurAmberPixels), in: geometry.size)
                    marker(at: Double(CaptureTuning.blurRedPixels), in: geometry.size)
                }
            }
            .frame(height: 10)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Smear \(String(format: "%.1f", pixels)) pixels, \(verdict)")
    }

    private func marker(at value: Double, in size: CGSize) -> some View {
        let limit = Double(CaptureTuning.blurRedPixels * 1.5)
        let x = size.width * CGFloat(value / limit)
        return Rectangle()
            .fill(Color.black.opacity(0.55))
            .frame(width: 2, height: size.height)
            .offset(x: x - 1)
    }
}

// =============================================================================
//  MARK: - Channel picker
// =============================================================================

/// Lets someone ask the room a single question instead of three at once.
@MainActor
struct CaptureChannelPicker: View {

    @ObservedObject var model: CaptureScreenModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                chip(title: "All three", channel: nil, color: Color.white)
                ForEach(CaptureCoverageChannel.allCases, id: \.rawValue) { channel in
                    chip(
                        title: channel.shortTitle,
                        channel: channel,
                        color: channel.color
                    )
                }
            }

            if let channel = model.channel {
                VStack(alignment: .leading, spacing: 3) {
                    Text(channel.explanation)
                    Text("Wherever it is not green yet: \(channel.fixHint.lowercased()).")
                        .foregroundStyle(channel.color)
                }
                .font(.footnote)
                .fixedSize(horizontal: false, vertical: true)
            } else {
                CaptureChannelLegend()
            }
        }
        .foregroundStyle(.white.opacity(0.9))
    }

    private func chip(
        title: String,
        channel: CaptureCoverageChannel?,
        color: Color
    ) -> some View {
        let selected = model.channel == channel
        return Button {
            model.applyChannel(channel)
        } label: {
            Text(title)
                .font(.caption.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
        }
        .background(
            selected ? color.opacity(0.9) : Color.white.opacity(0.14),
            in: Capsule()
        )
        .foregroundStyle(selected ? Color.black : Color.white)
    }
}

/// What each colour on the wall means, and the one thing to do about it.
struct CaptureChannelLegend: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(CaptureCoverageChannel.allCases, id: \.rawValue) { channel in
                HStack(spacing: 8) {
                    Circle().fill(channel.color).frame(width: 9, height: 9)
                    Text(channel.fixHint)
                }
            }
            HStack(spacing: 8) {
                Circle().fill(CaptureHUDPalette.satisfied).frame(width: 9, height: 9)
                Text("Done, nothing to do here")
            }
            HStack(spacing: 8) {
                Circle().fill(CaptureHUDPalette.unreliable).frame(width: 9, height: 9)
                Text("Glass, which cannot be fixed and is not counted against you")
            }
        }
        .font(.caption)
    }
}

// =============================================================================
//  MARK: - Problems
// =============================================================================

/// A full-screen explanation when a scan could not start or could not be
/// saved. One sentence about what happened, one button.
struct CaptureProblemPanel: View {

    let message: String
    let onDismiss: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 44))
                .foregroundStyle(CaptureHUDPalette.sharpness)
            Text(message)
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button("Back", action: onDismiss)
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(Color.white, in: RoundedRectangle(cornerRadius: 14))
                .foregroundStyle(.black)
        }
        .padding(28)
        .background(Color.black.opacity(0.85))
    }
}

// =============================================================================
//  MARK: - Formatting
// =============================================================================

/// Numbers as a person would write them.
enum CaptureFormat {

    /// The upper bound is not decoration. One of the two callers is
    /// `CaptureReportView`, whose `durationSeconds` is
    /// `max(0, last.timestampSeconds - first.timestampSeconds)` over
    /// `CaptureFrame.timestampSeconds`, a plain `Double` on a `Codable` struct
    /// decoded straight out of `capture_bundle.json` with nothing validating
    /// it. `max(0, seconds)` alone stops a NaN, because the literal is first
    /// and Swift's `max` absorbs its second argument, but it stops nothing at
    /// the top: a bundle carrying 1e308, or two timestamps whose subtraction
    /// overflows to infinity, walks through it and `Int(_:)` traps above
    /// 9.2e18. That is the same trapping conversion the owner reported as
    /// "crashing quite a bit, even with RAM free", reached from a formatter.
    /// `PrePassPoseRefiner` already caps the same subtraction at 86_400; this
    /// is that fix, in the shape `tools/trapconv.py` recognises as safe, with
    /// both literals first so a NaN absorbs to 0 and an infinity to 86_400.
    /// A capture claiming longer than a day now reads "1440:00" instead of
    /// killing the app, and no caller reads the string back.
    static func duration(_ seconds: Double) -> String {
        let total = Int(Swift.min(86_400, Swift.max(0, seconds)))
        return String(format: "%d:%02d", total / 60, total % 60)
    }

    static func bytes(_ count: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useMB, .useGB]
        return formatter.string(fromByteCount: max(0, count))
    }

    static func percent(_ fraction: Float) -> String {
        "\(Int((max(0, min(1, fraction)) * 100).rounded()))%"
    }

    static func meters(_ value: Float) -> String {
        String(format: "%.2f m", value)
    }

    static func centimeters(_ value: Float) -> String {
        String(format: "%.1f cm", value)
    }
}
