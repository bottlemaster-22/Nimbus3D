//
//  ProcessView.swift
//  Nimbus3D - Pipeline module
//
//  The "Process" tab. `ProcessRootView` is the entry point the App shell drops
//  into the Process tab (replacing ProcessPlaceholderView in NimbusApp.swift).
//  It lets the user pick a captured bundle, kick off a full pipeline run, and
//  watch each stage report PipelineProgress live.
//
//  This screen is purely a view over PipelineRunViewModel. It never invents
//  stage results: stages backed by unwired/stubbed dependencies show as
//  "Skipped" with the honest reason the orchestrator provided.
//

import SwiftUI

// MARK: - Display helpers

extension PipelineStage {
    /// Human-readable stage name for the Process board.
    var displayName: String {
        switch self {
        case .capture: return "Capture"
        case .splatTraining: return "Splat Training"
        case .meshExtraction: return "Mesh Extraction"
        case .delighting: return "Delighting"
        case .materialClassification: return "Material Classification"
        case .textureBuild: return "Texture Build"
        case .hdriAssembly: return "HDRI Assembly"
        case .export: return "Export"
        }
    }

    var systemImage: String {
        switch self {
        case .capture: return "camera.viewfinder"
        case .splatTraining: return "sparkles"
        case .meshExtraction: return "grid"
        case .delighting: return "sun.max"
        case .materialClassification: return "square.stack.3d.up"
        case .textureBuild: return "paintpalette"
        case .hdriAssembly: return "globe.americas"
        case .export: return "square.and.arrow.up"
        }
    }
}

extension PipelineStageStatus {
    var isRunning: Bool { if case .running = self { return true }; return false }

    /// Deterministic fraction for a linear progress bar, or nil when the row
    /// should show an indeterminate/complete state instead.
    var determinateFraction: Double? {
        switch self {
        case let .running(fraction, _, isIndeterminate):
            return isIndeterminate ? nil : fraction
        case .completed, .skipped:
            return 1
        case .pending, .failed:
            return nil
        }
    }

    var statusLabel: String {
        switch self {
        case .pending: return "Waiting"
        case let .running(_, message, _): return message
        case .completed: return "Done"
        case let .skipped(reason): return "Skipped: \(reason)"
        case let .failed(message): return "Failed: \(message)"
        }
    }

    var iconName: String {
        switch self {
        case .pending: return "circle"
        case .running: return "circle.dotted"
        case .completed: return "checkmark.circle.fill"
        case .skipped: return "minus.circle"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }

    var tint: Color {
        switch self {
        case .pending: return .secondary
        case .running: return .accentColor
        case .completed: return .green
        case .skipped: return .orange
        case .failed: return .red
        }
    }
}

// MARK: - Root screen

public struct ProcessRootView: View {
    @State private var model = PipelineRunViewModel()

    public init() {}

    public var body: some View {
        Group {
            if model.bundles.isEmpty {
                ContentUnavailableView(
                    "No Captures Yet",
                    systemImage: "camera.viewfinder",
                    description: Text("Scan an object in the Capture tab. Finished captures appear here, ready to turn into a game-ready asset.")
                )
            } else {
                List {
                    bundlePickerSection
                    stageBoardSection
                    resultSection
                }
            }
        }
        .navigationTitle("Process")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    model.refreshBundles()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(model.isRunning)
                .accessibilityLabel("Reload captures")
            }
        }
        .safeAreaInset(edge: .bottom) {
            if !model.bundles.isEmpty {
                runControlBar
            }
        }
        .onAppear { model.refreshBundles() }
    }

    // MARK: Sections

    private var bundlePickerSection: some View {
        Section("Capture") {
            ForEach(model.bundles) { bundle in
                Button {
                    model.select(bundle)
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(bundle.createdAt.formatted(date: .abbreviated, time: .shortened))
                                .foregroundStyle(.primary)
                            Text(bundleSubtitle(bundle))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if bundle.id == model.selectedBundleID {
                            Image(systemName: "checkmark")
                                .foregroundStyle(Color.accentColor)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(model.isRunning)
            }
        }
    }

    private var stageBoardSection: some View {
        Section {
            ForEach(PipelineRunViewModel.orderedStages, id: \.self) { stage in
                StageRow(stage: stage, status: model.status(for: stage))
            }
        } header: {
            HStack {
                Text("Pipeline")
                Spacer()
                Text("\(Int((model.overallFraction * 100).rounded()))%")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        } footer: {
            ProgressView(value: model.overallFraction)
                .padding(.top, 4)
        }
    }

    @ViewBuilder
    private var resultSection: some View {
        switch model.phase {
        case let .finished(asset):
            Section("Result") {
                Label("Export complete", systemImage: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                Text(asset.rootDirectory.lastPathComponent)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Text("Find it in the Library tab.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case let .failed(message):
            Section("Result") {
                Label("Pipeline failed", systemImage: "xmark.octagon.fill")
                    .foregroundStyle(.red)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .cancelled:
            Section("Result") {
                Label("Cancelled", systemImage: "stop.circle.fill")
                    .foregroundStyle(.orange)
            }
        case .idle, .running:
            EmptyView()
        }
    }

    private var runControlBar: some View {
        HStack {
            if model.isRunning {
                Button(role: .destructive) {
                    model.cancel()
                } label: {
                    Label("Cancel", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            } else {
                Button {
                    model.run()
                } label: {
                    Label("Process Capture", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.selectedBundle == nil)
            }
        }
        .padding()
        .background(.bar)
    }

    private func bundleSubtitle(_ bundle: CaptureBundle) -> String {
        var parts = ["\(bundle.frames.count) frames"]
        if bundle.hasLiDARDepth { parts.append("LiDAR depth") }
        if !bundle.hdriBrackets.isEmpty { parts.append("\(bundle.hdriBrackets.count) HDRI brackets") }
        return parts.joined(separator: " · ")
    }
}

// MARK: - Stage row

private struct StageRow: View {
    let stage: PipelineStage
    let status: PipelineStageStatus

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: status.iconName)
                .foregroundStyle(status.tint)
                .symbolEffect(.pulse, isActive: status.isRunning)
                .frame(width: 22)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Label(stage.displayName, systemImage: stage.systemImage)
                        .labelStyle(.titleOnly)
                        .font(.body.weight(.medium))
                    Spacer()
                }
                Text(status.statusLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)

                if status.isRunning {
                    if let fraction = status.determinateFraction {
                        ProgressView(value: fraction)
                    } else {
                        ProgressView()
                            .progressViewStyle(.linear)
                    }
                }
            }
        }
        .padding(.vertical, 2)
    }
}
