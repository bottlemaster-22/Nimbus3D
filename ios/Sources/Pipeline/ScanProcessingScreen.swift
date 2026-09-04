//
//  ScanProcessingScreen.swift
//  Pipeline
//
//  "FINISH THIS SCAN": the screen with the buttons that were missing.
//
//  The scan library already said the right sentences - "Recorded. Not checked
//  over yet.", "Checked over. Ready to build the 3D model.", "Ready to look
//  at." - and there was nothing to tap. This screen is what those sentences
//  point at: one button per state, plus one button that does the whole thing,
//  because nobody wants to press two buttons and wait twice.
//
//  WHAT THIS SCREEN WILL NOT DO
//  * It will not draw a percentage the services did not give it. The pre-pass
//    reports a real fraction; the trainer's is optional and is nil in the
//    stages that genuinely cannot say, and those get a spinner rather than a
//    bar stuck at zero.
//  * It will not hide a smaller model. If the phone lowered the budget, the
//    trainer's own sentence about it stays on screen, and the finished model is
//    described by what it actually is.
//  * It will not build a second preview. A finished model opens in the review
//    screen the Viewer already owns.
//

import Foundation
import SwiftUI

/// `@MainActor` for the same reason `BoosterTabView` is: it holds a reference
/// to main-actor isolated shared objects in its own stored properties, and
/// saying so is cheaper than relying on inference.
@MainActor
struct ScanProcessingScreen: View {

    let summary: ScanSummary

    @ObservedObject private var coordinator = ScanProcessingCoordinator.shared
    @ObservedObject private var discovery = BoosterClient.shared.discovery

    /// What is on disk right now. Re-read when a run ends, so the buttons and
    /// the "ready to look at" link tell the truth without a manual refresh.
    @State private var detail: ScanDetail?
    /// Whether the model's splat file is really there. Worked out once, when the
    /// scan is read, rather than hitting the file system on every redraw.
    @State private var modelReady = false
    @State private var reuseExistingPrePass = true
    @State private var boosterMessage: String?

    var body: some View {
        List {
            stateSection

            if isLiveHere {
                progressSection
            } else {
                actionsSection
            }

            if let problem = coordinator.problem, coordinator.isShowing(summary.scanID) {
                problemSection(problem)
            }

            if coordinator.isShowing(summary.scanID), !coordinator.notices.isEmpty {
                noticesSection
            }

            if let plan = coordinator.plan, coordinator.isShowing(summary.scanID) {
                planSection(plan)
            }

            resultSection
            boosterSection
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Finish this scan")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await reload()
            // Started, never stopped: this browser object is shared with the
            // Booster tab, and stopping it from here could take the tab's own
            // list away underneath it.
            discovery.start()
        }
        .onChange(of: coordinator.phase) { _, _ in
            Task { await reload() }
        }
    }

    // MARK: - Where this scan has got to

    private var stateSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                Text(summary.displayName)
                    .font(.headline)
                Text(recordedLine)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Text(stateSentence)
                    .font(.subheadline)
            }
            .padding(.vertical, 2)

            if let card = detail?.prePass?.qcCard {
                Text(qualityLine(card))
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("This scan")
        }
    }

    private var recordedLine: String {
        var parts: [String] = [ViewerFormat.date(summary.createdAt)]
        if summary.frameCount > 0 { parts.append("\(summary.frameCount) photos") }
        if let duration = summary.durationSeconds, duration > 0 {
            parts.append(ViewerFormat.duration(duration))
        }
        parts.append(ViewerFormat.bytes(summary.byteCount))
        return parts.joined(separator: "  -  ")
    }

    private var stateSentence: String {
        // Until the folder has been read, the library's own sentence is the
        // honest one: this screen does not yet know any better.
        guard detail != nil else { return summary.nextStep }
        if hasModel { return "Ready to look at." }
        if hasPrePass { return "Checked over. Ready to build the 3D model." }
        if summary.frameCount > 0 { return "Recorded. Not checked over yet." }
        return "This folder has no photos in it."
    }

    private func qualityLine(_ card: QCCard) -> String {
        let coverage = ViewerFormat.percent(card.coverageFraction)
        let notes = card.findings.count
        let noun = notes == 1 ? "note" : "notes"
        return "The check-over reached \(coverage) of the surfaces it could see, with "
            + "\(notes) \(noun) about how it came out."
    }

    // MARK: - Buttons

    @ViewBuilder
    private var actionsSection: some View {
        Section {
            if prePassAvailable && trainerAvailable {
                Button {
                    coordinator.start(
                        .everything, for: summary, reuseExistingPrePass: reuseExistingPrePass
                    )
                } label: {
                    Label("Do all of it", systemImage: "wand.and.stars")
                        .font(.body.weight(.semibold))
                }
                .disabled(!canStart)
            }

            if prePassAvailable {
                Button {
                    coordinator.start(
                        .checkOver, for: summary, reuseExistingPrePass: false
                    )
                } label: {
                    Label(
                        hasPrePass ? "Check it over again" : "Check it over",
                        systemImage: "checkmark.magnifyingglass"
                    )
                }
                .disabled(!canStart)
            }

            if trainerAvailable {
                Button {
                    coordinator.start(
                        .buildModel, for: summary, reuseExistingPrePass: true
                    )
                } label: {
                    Label(
                        hasModel ? "Build the 3D model again" : "Build the 3D model",
                        systemImage: "cube"
                    )
                }
                .disabled(!canStart || !hasPrePass)
            }

            if hasPrePass, let completed = detail?.prePass?.completedAt {
                Toggle(isOn: $reuseExistingPrePass) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Use the check-over already done")
                        Text(
                            "Checked over on \(ProcessingFormat.date(completed)). Reusing it "
                            + "skips about twenty minutes of work."
                        )
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    }
                }
            }
        } header: {
            Text("What happens next")
        } footer: {
            Text(footerText)
        }
    }

    private var footerText: String {
        var lines: [String] = []
        if coordinator.isRunning && !coordinator.isShowing(summary.scanID) {
            lines.append(
                "Your phone is busy with another scan right now. This one can start as "
                + "soon as that one is done."
            )
        }
        if !prePassAvailable {
            lines.append("Checking a scan over is not part of this build yet.")
        }
        if !trainerAvailable {
            lines.append("Building the 3D model is not part of this build yet.")
        }
        if !hasPrePass && trainerAvailable {
            lines.append(
                "The 3D model can only be built after the check-over, because that is "
                + "where the camera positions and the first points come from."
            )
        }
        if hasModel {
            lines.append(
                "Building it again replaces the model you already have. The photos and "
                + "measurements are never touched."
            )
        }
        if lines.isEmpty {
            lines.append(
                "This all happens on your phone. Nothing is uploaded anywhere. Keep the "
                + "app open while it works, and plug the phone in if you can."
            )
        }
        return lines.joined(separator: " ")
    }

    // MARK: - Live progress

    @ViewBuilder
    private var progressSection: some View {
        Section {
            switch coordinator.phase {
            case .reading:
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Opening this scan...")
                }
            case .checkingOver:
                prePassProgress
            case .buildingModel:
                trainerProgress
            default:
                EmptyView()
            }

            Button(role: .destructive) {
                coordinator.cancel()
            } label: {
                Label("Stop", systemImage: "stop.circle")
            }
        } header: {
            Text(progressHeader)
        } footer: {
            Text(
                "This keeps going while you look at other screens in the app, but it "
                + "stops if you close the app."
            )
        }
    }

    /// Says which of the two jobs is running, and whether there is another one
    /// after it, because "step 1 of 2" is the difference between waiting
    /// patiently and thinking it has stalled.
    private var progressHeader: String {
        let bothJobs = coordinator.intent == .everything
        switch coordinator.phase {
        case .buildingModel:
            return bothJobs ? "Building the 3D model (step 2 of 2)" : "Building the 3D model"
        case .checkingOver:
            return bothJobs ? "Checking it over (step 1 of 2)" : "Checking it over"
        default:
            return "Working on it"
        }
    }

    @ViewBuilder
    private var prePassProgress: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let tick = coordinator.prePassTick {
                Text(tick.message)
                    .font(.subheadline)
                ProgressView(value: min(max(tick.fractionComplete, 0), 1))
            } else {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Getting your scan ready to check over.")
                        .font(.subheadline)
                }
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var trainerProgress: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let tick = coordinator.trainerTick {
                Text(Self.stageTitle(tick.stage))
                    .font(.subheadline.weight(.semibold))
                Text(tick.message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                // The trainer's fraction is optional by contract, and a nil
                // means "this stage genuinely cannot say". A spinner is the
                // honest drawing of that; a bar at zero is not.
                if let fraction = tick.fractionComplete {
                    ProgressView(value: min(max(fraction, 0), 1))
                } else {
                    ProgressView()
                }

                Text(trainerDetailLine(tick))
                    .font(.caption)
                    .foregroundStyle(.tertiary)

                if tick.thermalLevel >= .serious {
                    Label(
                        "Your phone is hot. It will slow itself down or pause rather than "
                        + "lose what it has built.",
                        systemImage: "thermometer.high"
                    )
                    .font(.footnote)
                    .foregroundStyle(.orange)
                }
            } else {
                HStack(spacing: 10) {
                    ProgressView()
                    Text("Getting the graphics ready.")
                        .font(.subheadline)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func trainerDetailLine(_ tick: TrainerProgress) -> String {
        var parts: [String] = []
        if tick.splatCount > 0 {
            parts.append("\(ProcessingFormat.count(tick.splatCount)) detail points so far")
        }
        if tick.totalIterations > 0 {
            parts.append("round \(tick.iteration) of \(tick.totalIterations)")
        }
        if tick.residentBytes > 0 {
            parts.append("\(ProcessingFormat.bytes(tick.residentBytes)) of memory in use")
        }
        return parts.joined(separator: "  -  ")
    }

    static func stageTitle(_ stage: TrainerStage) -> String {
        switch stage {
        case .preparing: return "Getting ready"
        case .initializing: return "Placing the first points"
        case .warmup: return "Settling everything into place"
        case .densifying: return "Adding detail where it is needed"
        case .refining: return "Sharpening it up"
        case .binarizing: return "Making the surfaces solid"
        case .finalizing: return "Saving your model"
        case .done: return "Finished"
        case .failed: return "Stopped with a problem"
        case .cancelled: return "Stopped"
        case .pausedThermal: return "Paused while your phone cools down"
        case .pausedMemory: return "Making the model smaller so it fits"
        }
    }

    // MARK: - Notices, plan, problem

    private var noticesSection: some View {
        Section {
            ForEach(coordinator.notices) { notice in
                Label {
                    Text(notice.text)
                        .font(.subheadline)
                } icon: {
                    Image(systemName: notice.iconName)
                        .foregroundStyle(notice.kind == .warning ? Color.orange : Color.secondary)
                }
                .padding(.vertical, 2)
            }
        } header: {
            Text("What happened along the way")
        } footer: {
            Text(
                "Anything your phone decided to change while it worked is written here "
                + "rather than kept quiet."
            )
        }
    }

    private func planSection(_ plan: ProcessingBudgetPlan) -> some View {
        Section {
            ForEach(plan.lines.indices, id: \.self) { index in
                Text(plan.lines[index])
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text("The size it aimed for, and why")
        }
    }

    private func problemSection(_ problem: String) -> some View {
        Section {
            Label {
                Text(problem)
                    .font(.subheadline)
            } icon: {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
            }
            if let hint = coordinator.problemHint {
                Text(hint)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Button("Put this away") { coordinator.dismissOutcome() }
        } header: {
            Text("That did not finish")
        }
    }

    // MARK: - The finished model

    @ViewBuilder
    private var resultSection: some View {
        if hasModel {
            Section {
                NavigationLink {
                    ScanReviewScreen(summary: summary)
                } label: {
                    Label("Look at it", systemImage: "eye")
                        .font(.body.weight(.semibold))
                }

                if let model = detail?.model {
                    Text(modelLine(model))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Your 3D model")
            }
        }
    }

    private func modelLine(_ model: SplatModel) -> String {
        var parts: [String] = [
            "\(ProcessingFormat.count(model.splatCount)) detail points",
            "\(ProcessingFormat.count(model.iterationsCompleted)) rounds",
            ViewerFormat.date(model.createdAt)
        ]
        switch model.source {
        case .onDevice: parts.append("built on this phone")
        case .booster: parts.append("built on your computer")
        case .imported: parts.append("brought in from elsewhere")
        }
        return parts.joined(separator: "  -  ")
    }

    // MARK: - The computer on your Wi-Fi

    @ViewBuilder
    private var boosterSection: some View {
        if NimbusServices.shared.booster != nil {
            Section {
                if let message = boosterMessage {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if readyBoosters.isEmpty {
                    Text(
                        "No computer on your Wi-Fi is set up yet. The Booster tab is where "
                        + "you connect one."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                } else {
                    ForEach(readyBoosters) { device in
                        Button {
                            send(to: device)
                        } label: {
                            Label("Send it to \(device.name)", systemImage: "desktopcomputer")
                        }
                        .disabled(coordinator.isRunning)
                    }
                }
            } header: {
                Text("Or let a computer do it")
            } footer: {
                Text(
                    "A computer on your own Wi-Fi can build a scan that is too big for a "
                    + "phone. Nothing leaves your home network. Watch it on the Booster tab."
                )
            }
        }
    }

    private var readyBoosters: [BoosterDevice] {
        discovery.devices.filter { $0.isPaired && $0.isReachable }
    }

    private func send(to device: BoosterDevice) {
        guard let booster = NimbusServices.shared.booster else { return }
        booster.sendScan(
            scanID: summary.scanID,
            scanDirectory: summary.rootURL,
            to: device
        )
        boosterMessage = "Sent to \(device.name). The Booster tab shows how it is getting on."
    }

    // MARK: - State

    private var isLiveHere: Bool {
        coordinator.isRunning && coordinator.isShowing(summary.scanID)
    }

    private var canStart: Bool {
        !coordinator.isRunning && summary.frameCount > 0 && detail?.bundle != nil
    }

    private var hasPrePass: Bool { detail?.prePass != nil }

    /// A model counts as there only when its splat file is there too. A
    /// `model.json` with no `.ply` beside it would offer a "Look at it" button
    /// that opens onto an error.
    private var hasModel: Bool { modelReady }

    private var prePassAvailable: Bool { NimbusServices.shared.prePass != nil }
    private var trainerAvailable: Bool { NimbusServices.shared.trainer != nil }

    private func reload() async {
        let target = summary
        let outcome = await Task.detached(priority: .userInitiated) {
            () -> (detail: ScanDetail, modelReady: Bool) in
            let loaded = ScanLibraryReader.readDetail(target)
            let ready = loaded.model.map {
                ProcessingArtifacts.splatFileExists(for: $0, at: target.paths)
            } ?? false
            return (loaded, ready)
        }.value

        detail = outcome.detail
        modelReady = outcome.modelReady
        reuseExistingPrePass = outcome.detail.prePass != nil
    }
}
