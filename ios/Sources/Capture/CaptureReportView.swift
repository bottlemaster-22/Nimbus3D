//
//  CaptureReportView.swift
//  Capture
//
//  THE CARD THAT COMES UP THE MOMENT THE USER STOPS (F9).
//
//  The point of showing anything at all here is that the user is still
//  STANDING IN THE ROOM. Everything on this card is either something they can
//  act on in the next thirty seconds, or something they should know before
//  they walk away and it is too late.
//
//  WHERE THE NUMBERS COME FROM, AND WHO OWNS WHICH.
//
//  The real quality card is `QCCard`, and it is the pre-pass's to produce:
//  `PrePassService.quickQCCard` is contracted to answer from poses and LiDAR
//  alone in under three seconds. This file calls it and renders it. It does
//  not reimplement any of it.
//
//  What this file does compute is the smaller, different thing: what the
//  RECORDING knows about itself. How many shots, how many were smeared, how
//  far the anchors moved between being placed and being re-read at the end,
//  how many times the walk crossed its own path. Those are facts capture
//  measured while it was running, they are already in the bundle, and they are
//  useful on their own when the pre-pass module is not in the build. They are
//  shown under their own heading and never dressed up as the full check-over.
//

import SwiftUI

// =============================================================================
//  MARK: - What the report is
// =============================================================================

/// One finished scan, plus whatever has been said about it so far.
struct CaptureReport {
    let bundle: CaptureBundle
    let ref: CaptureBundleRef
    let summary: CaptureSessionSummary
    /// The pre-pass's quality card, once it arrives. nil while it is still
    /// being computed, or when that module is not in this build.
    var card: QCCard?
    /// Why there is no card, in plain words. Never left nil at the same time
    /// as `card`: one of the two is always filled in.
    var cardNote: String?
    /// Set when the session stopped itself rather than the user stopping it.
    var stopReason: String?
}

/// What the recording measured about itself, straight out of the bundle.
struct CaptureSessionSummary {

    var frameCount: Int
    var framesWithDepth: Int
    var durationSeconds: Double
    var coverageFraction: Float
    var bytesWritten: Int64

    var medianBlurPixels: Float
    var smearyFrameCount: Int
    var lowWeightFrameCount: Int
    var bracketedFrameCount: Int

    var meshChunkCount: Int
    var revisitCount: Int
    var sceneExtentMeters: Float?

    /// Median distance an anchor moved between being logged live and being
    /// re-read once the session ended. This is the map sliding under the walk,
    /// measured directly rather than estimated (F8). nil when no anchor
    /// appeared in both lists.
    var anchorDriftCentimeters: Float?

    /// The camera-to-motion clock offset the session measured, milliseconds,
    /// or nil when the sweep had no clear answer.
    var timeOffsetMilliseconds: Double?

    static func make(
        bundle: CaptureBundle,
        coverageFraction: Float,
        bytesWritten: Int64
    ) -> CaptureSessionSummary {
        let frames = bundle.frames
        let blurValues = frames.map { $0.qc.motionBlurPixels }.sorted()
        let median: Float = blurValues.isEmpty
            ? 0
            : blurValues[blurValues.count / 2]

        let duration: Double
        if let first = frames.first, let last = frames.last {
            duration = max(0, last.timestampSeconds - first.timestampSeconds)
        } else {
            duration = 0
        }

        return CaptureSessionSummary(
            frameCount: frames.count,
            framesWithDepth: frames.filter { $0.depthPath != nil }.count,
            durationSeconds: duration,
            coverageFraction: coverageFraction,
            bytesWritten: bytesWritten,
            medianBlurPixels: median,
            smearyFrameCount: frames.filter {
                $0.qc.motionBlurPixels >= CaptureTuning.blurRedPixels
            }.count,
            lowWeightFrameCount: frames.filter {
                $0.qc.weight < CaptureTuning.keyframeMinQCWeight
            }.count,
            bracketedFrameCount: frames.filter { $0.bracket == .darker }.count,
            meshChunkCount: bundle.meshChunks.count,
            revisitCount: bundle.revisitPairs.count,
            sceneExtentMeters: bundle.sceneBounds?.longestEdgeMeters,
            anchorDriftCentimeters: anchorDrift(bundle: bundle),
            timeOffsetMilliseconds: bundle.cameraToIMUTimeOffsetSeconds.map { $0 * 1000 }
        )
    }

    /// The median of how far each anchor moved between the two lists.
    ///
    /// Median rather than mean: one anchor that ARKit threw across the room
    /// during a relocalisation should not become the headline number.
    private static func anchorDrift(bundle: CaptureBundle) -> Float? {
        var finalByID: [UUID: AnchorRecord] = [:]
        for record in bundle.anchorsAtEndOfSession {
            finalByID[record.identifier] = record
        }
        var distances: [Float] = []
        for live in bundle.anchorsDuringSession {
            guard let end = finalByID[live.identifier] else { continue }
            guard live.transform.count == 16, end.transform.count == 16 else { continue }
            let dx = end.transform[12] - live.transform[12]
            let dy = end.transform[13] - live.transform[13]
            let dz = end.transform[14] - live.transform[14]
            distances.append((dx * dx + dy * dy + dz * dz).squareRoot())
        }
        guard !distances.isEmpty else { return nil }
        distances.sort()
        return distances[distances.count / 2] * 100
    }
}

// =============================================================================
//  MARK: - The card
// =============================================================================

/// What the user sees the moment they stop.
@MainActor
struct CaptureReportPanel: View {

    let report: CaptureReport
    let onDone: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    header

                    if let reason = report.stopReason {
                        CaptureBanner(text: reason, tone: .problem)
                    }

                    if let card = report.card {
                        checkOver(card)
                    } else if let note = report.cardNote {
                        pendingCheckOver(note)
                    } else {
                        HStack(spacing: 10) {
                            ProgressView()
                            Text("Checking the scan over...")
                                .foregroundStyle(.white.opacity(0.8))
                        }
                    }

                    recordingSection

                    detailsSection
                }
                .padding(24)
            }

            Button("Done", action: onDone)
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(Color.white, in: RoundedRectangle(cornerRadius: 14))
                .foregroundStyle(.black)
                .padding(.horizontal, 24)
                .padding(.bottom, 20)
        }
        .background(Color.black.opacity(0.92))
        .foregroundStyle(.white)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Scan saved")
                .font(.largeTitle.weight(.bold))
            Text(report.bundle.displayName)
                .font(.title3)
                .foregroundStyle(.white.opacity(0.8))
            Text(
                "\(report.summary.frameCount) shots, "
                    + CaptureFormat.duration(report.summary.durationSeconds)
                    + ", " + CaptureFormat.bytes(report.summary.bytesWritten)
            )
            .font(.footnote)
            .foregroundStyle(.white.opacity(0.6))
        }
    }

    // MARK: - The pre-pass's card

    @ViewBuilder
    private func checkOver(_ card: QCCard) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("The check-over")
                .font(.headline)

            ForEach(card.findings) { finding in
                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Circle()
                            .fill(Self.color(for: finding.severity))
                            .frame(width: 9, height: 9)
                        Text(finding.message)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if let fix = finding.fixHint {
                        Text(fix)
                            .font(.footnote)
                            .foregroundStyle(.white.opacity(0.7))
                            .padding(.leading, 17)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            CaptureFactGrid(
                facts: [
                    CaptureFact("Covered", CaptureFormat.percent(card.coverageFraction)),
                    CaptureFact(
                        "Ceiling covered",
                        CaptureFormat.percent(card.ceilingCoverageFraction)
                    ),
                    CaptureFact(
                        "Map slid",
                        CaptureFormat.centimeters(card.driftCentimeters)
                    ),
                    CaptureFact(
                        "Crossed your own path",
                        "\(card.loopClosureCount) times"
                    ),
                    CaptureFact(
                        "Angles a surface was seen from",
                        String(format: "%.0f degrees apart", card.medianAngularSpreadDegrees)
                    ),
                    CaptureFact(
                        "How far you stood back",
                        CaptureFormat.meters(card.medianCameraToSurfaceMeters)
                    ),
                    CaptureFact(
                        "How much you varied your height",
                        CaptureFormat.meters(card.cameraHeightSpreadMeters)
                    ),
                    CaptureFact(
                        "Glass and windows",
                        CaptureFormat.percent(card.glassAreaFraction)
                    ),
                ]
            )
        }
    }

    private func pendingCheckOver(_ note: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("The check-over")
                .font(.headline)
            Text(note)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.75))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - What capture measured itself

    private var recordingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What the recording itself says")
                .font(.headline)

            ForEach(notes) { note in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Circle()
                        .fill(note.color)
                        .frame(width: 9, height: 9)
                    Text(note.text)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// The honest, small set of things the recording can say about itself.
    private var notes: [CaptureReportNote] {
        let summary = report.summary
        var result: [CaptureReportNote] = []

        if summary.coverageFraction >= CaptureTuning.coverageDoneFraction {
            result.append(
                CaptureReportNote(
                    text: "You covered \(CaptureFormat.percent(summary.coverageFraction)) "
                        + "of what you looked at, which is plenty.",
                    color: CaptureHUDPalette.satisfied
                )
            )
        } else {
            result.append(
                CaptureReportNote(
                    text: "You covered \(CaptureFormat.percent(summary.coverageFraction)). "
                        + "Around \(CaptureFormat.percent(CaptureTuning.coverageDoneFraction)) "
                        + "is where a scan usually has enough to work with, so "
                        + "expect some thin patches.",
                    color: CaptureHUDPalette.sharpness
                )
            )
        }

        if summary.frameCount > 0 {
            let smearyShare = Double(summary.smearyFrameCount) / Double(summary.frameCount)
            if smearyShare >= 0.15 {
                result.append(
                    CaptureReportNote(
                        text: "About \(Int((smearyShare * 10).rounded())) shots in 10 were "
                            + "smeared from moving too fast. Walking more slowly is the "
                            + "whole fix.",
                        color: CaptureHUDPalette.problem
                    )
                )
            } else {
                result.append(
                    CaptureReportNote(
                        text: "Your shots were steady: a typical one smeared by "
                            + String(format: "%.1f", summary.medianBlurPixels)
                            + " pixels, and 2 is the point where detail starts to go.",
                        color: CaptureHUDPalette.satisfied
                    )
                )
            }
        }

        if let drift = summary.anchorDriftCentimeters {
            if drift > 3 {
                result.append(
                    CaptureReportNote(
                        text: "While you walked, the phone's idea of where things are "
                            + "slid by about \(CaptureFormat.centimeters(drift)). Walking "
                            + "back through a doorway you have already been through "
                            + "helps it settle.",
                        color: CaptureHUDPalette.sharpness
                    )
                )
            } else {
                result.append(
                    CaptureReportNote(
                        text: "The phone's idea of where things are barely moved "
                            + "(\(CaptureFormat.centimeters(drift))), which is a good sign.",
                        color: CaptureHUDPalette.satisfied
                    )
                )
            }
        }

        if summary.revisitCount > 0 {
            result.append(
                CaptureReportNote(
                    text: "You walked back over the same spot \(summary.revisitCount) "
                        + "times. That is what lets the app straighten the scan out later.",
                    color: CaptureHUDPalette.satisfied
                )
            )
        } else {
            result.append(
                CaptureReportNote(
                    text: "You never walked back over a spot you had already been. "
                        + "Finishing where you started makes a noticeable difference.",
                    color: CaptureHUDPalette.sharpness
                )
            )
        }

        if summary.framesWithDepth < summary.frameCount {
            let missing = summary.frameCount - summary.framesWithDepth
            result.append(
                CaptureReportNote(
                    text: "\(missing) shots came without any laser measurements. They "
                        + "still count as pictures, they just carry no distances.",
                    color: CaptureHUDPalette.unseen
                )
            )
        }

        if summary.bracketedFrameCount > 0 {
            result.append(
                CaptureReportNote(
                    text: "\(summary.bracketedFrameCount) darker shots were slipped in "
                        + "for the bright spots, so windows and lamps still have some "
                        + "detail in them.",
                    color: CaptureHUDPalette.satisfied
                )
            )
        }

        return result
    }

    // MARK: - The small print

    private var detailsSection: some View {
        DisclosureGroup("Technical details") {
            VStack(alignment: .leading, spacing: 6) {
                detail("Scan name on disk", report.bundle.scanID)
                detail("Shots kept", "\(report.summary.frameCount)")
                detail("With laser depth", "\(report.summary.framesWithDepth)")
                detail(
                    "Shots the trainer will lean on less",
                    "\(report.summary.lowWeightFrameCount)"
                )
                detail("Room mesh pieces", "\(report.summary.meshChunkCount)")
                if let extent = report.summary.sceneExtentMeters {
                    detail("Longest side of what you scanned", CaptureFormat.meters(extent))
                }
                detail(
                    "Depth map size",
                    "\(report.bundle.settings.depthWidth) by "
                        + "\(report.bundle.settings.depthHeight)"
                )
                if let offset = report.summary.timeOffsetMilliseconds {
                    detail(
                        "Camera and motion clocks lined up to",
                        String(format: "%.1f ms", offset)
                    )
                } else {
                    detail(
                        "Camera and motion clocks",
                        "not lined up on this scan"
                    )
                }
            }
            .padding(.top, 8)
        }
        .font(.footnote)
        .tint(Color.white)
        .foregroundStyle(.white.opacity(0.8))
    }

    private func detail(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
            Spacer(minLength: 12)
            Text(value)
                .foregroundStyle(.white)
                .multilineTextAlignment(.trailing)
        }
    }

    private static func color(for severity: QCFinding.Severity) -> Color {
        switch severity {
        case .good: return CaptureHUDPalette.satisfied
        case .warning: return CaptureHUDPalette.sharpness
        case .problem: return CaptureHUDPalette.problem
        }
    }
}

/// One sentence of the recording's own account of itself, with the colour that
/// says whether it is good news. Identified by its own text, which is unique
/// within a report and stable across a redraw.
struct CaptureReportNote: Identifiable {
    var text: String
    var color: Color
    var id: String { text }
}

/// One "label, value" row of the fact grid.
struct CaptureFact: Identifiable {
    let label: String
    let value: String
    var id: String { label }

    init(_ label: String, _ value: String) {
        self.label = label
        self.value = value
    }
}

/// A two-column run of "label, value" rows.
struct CaptureFactGrid: View {

    let facts: [CaptureFact]

    var body: some View {
        VStack(spacing: 6) {
            ForEach(facts) { fact in
                HStack(alignment: .firstTextBaseline) {
                    Text(fact.label)
                        .foregroundStyle(.white.opacity(0.7))
                    Spacer(minLength: 12)
                    Text(fact.value)
                        .fontWeight(.medium)
                }
            }
        }
        .font(.subheadline)
    }
}
