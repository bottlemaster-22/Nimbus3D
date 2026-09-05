//
//  ScanLibraryScreen.swift
//  Viewer
//
//  THE "YOUR SCANS" TAB.
//
//  Every scan folder on the phone, newest first, each row saying what that
//  scan actually is right now rather than what it will eventually be. A
//  capture that was interrupted, a scan waiting on a computer, a model that
//  came back from the Booster and a scan that is quietly broken all look
//  different from each other here, because they ARE different and the user has
//  to be able to tell.
//
//  This is the view `NimbusUI.libraryScreen` hands the app shell.
//

import SwiftUI
import UIKit

// MARK: - The screen

/// `@MainActor` on the whole view, matching `CaptureScreen` and `MainTabView`:
/// it holds two main-actor-isolated observed objects (`ScanProcessingCoordinator`
/// and `AppNavigation`) and mutates one of them from a `.task`, so isolating the
/// view rather than relying on `body`'s implicit isolation keeps every helper
/// property and method on the same actor.
@MainActor
struct ScanLibraryScreen: View {

    @StateObject private var store = ScanLibraryStore()
    /// Watched, not driven: when a check-over or a build starts or ends, the
    /// rows have to change what they say about themselves.
    @ObservedObject private var processing = ScanProcessingCoordinator.shared
    /// Set by another screen (the capture report) when it wants this list to
    /// put one scan forward. Consumed and cleared below: nothing else clears
    /// it, because only this screen knows when it has acted on it.
    @ObservedObject private var navigation = AppNavigation.shared
    /// The scan we were asked to lead with, kept after the request is cleared
    /// so the row can stay highlighted while the user looks at it.
    @State private var leadScanID: ScanID?
    @State private var renaming: ScanSummary?
    @State private var newName = ""
    @State private var deleting: ScanSummary?
    @State private var actionProblem: String?

    var body: some View {
        NavigationStack {
            Group {
                if store.problem != nil {
                    libraryUnreadable
                } else if store.scans.isEmpty && !store.isLoading {
                    emptyLibrary
                } else {
                    list
                }
            }
            .navigationTitle("Your scans")
            .task { await store.refresh() }
            .refreshable { await store.refresh() }
            .onChange(of: processing.phase) { _, _ in
                Task { await store.refresh() }
            }
            .alert("Rename this scan", isPresented: renamingBinding) {
                TextField("Name", text: $newName)
                Button("Cancel", role: .cancel) { renaming = nil }
                Button("Save") { commitRename() }
            } message: {
                Text("Give it a name you will recognise later, like \"Front room\".")
            }
            .alert(
                "Move this scan to the trash?",
                isPresented: deletingBinding
            ) {
                Button("Cancel", role: .cancel) { deleting = nil }
                Button("Move to trash", role: .destructive) { commitDelete() }
            } message: {
                Text(
                    "The photos and measurements go to the Files trash, where you can "
                    + "still get them back. You cannot take this walk again."
                )
            }
            .alert(
                "That did not work",
                isPresented: Binding(
                    get: { actionProblem != nil },
                    set: { if !$0 { actionProblem = nil } }
                )
            ) {
                Button("OK", role: .cancel) { actionProblem = nil }
            } message: {
                Text(actionProblem ?? "")
            }
        }
    }

    // MARK: List

    private var list: some View {
        ScrollViewReader { proxy in
            listBody
                // A scan handed over from the capture report will not be in
                // `store.scans` yet, so refresh first, then scroll to it. The
                // request is cleared either way: leaving it set would drag the
                // user back to the same row every time they opened this tab.
                .task(id: navigation.scanToLeadWith) {
                    guard let wanted = navigation.scanToLeadWith else { return }
                    navigation.scanToLeadWith = nil
                    leadScanID = wanted
                    if !store.scans.contains(where: { $0.scanID == wanted }) {
                        await store.refresh()
                    }
                    guard store.scans.contains(where: { $0.scanID == wanted }) else {
                        // The scan is genuinely not on disk. Say nothing and
                        // drop the highlight rather than scrolling to a row
                        // that is not there.
                        leadScanID = nil
                        return
                    }
                    withAnimation { proxy.scrollTo(wanted, anchor: .center) }
                }
        }
    }

    private var listBody: some View {
        List {
            Section {
                ForEach(store.scans) { scan in
                    // Tapping a scan goes to whatever it needs next. A scan
                    // with a model to look at opens the review screen; one
                    // that still has work to do opens the screen with the
                    // buttons that do it, which is the step the row's own
                    // sentence is promising.
                    NavigationLink {
                        if scan.hasModel {
                            ScanReviewScreen(summary: scan)
                        } else {
                            ScanProcessingScreen(summary: scan)
                        }
                    } label: {
                        ScanLibraryRow(scan: scan)
                    }
                    // The scan the user has just recorded, marked so it is
                    // obvious which row the app has brought them to.
                    .listRowBackground(
                        scan.scanID == leadScanID
                            ? Color.accentColor.opacity(0.14)
                            : Color.clear
                    )
                    .swipeActions(edge: .trailing) {
                        Button("Delete", role: .destructive) { deleting = scan }
                        Button("Rename") {
                            newName = scan.displayName
                            renaming = scan
                        }
                        .tint(.blue)
                    }
                }
            } footer: {
                Text(storageLine)
            }
        }
        .listStyle(.insetGrouped)
        .overlay {
            if store.isLoading && store.scans.isEmpty {
                ProgressView("Looking for your scans...")
            }
        }
    }

    private var storageLine: String {
        let count = store.scans.count
        let noun = count == 1 ? "scan" : "scans"
        return "\(count) \(noun), using \(ViewerFormat.bytes(store.totalBytes)) on this iPhone."
    }

    // MARK: Empty and broken states

    private var emptyLibrary: some View {
        VStack(spacing: 14) {
            Image(systemName: "square.stack.3d.up.slash")
                .font(.system(size: 42, weight: .light))
                .foregroundStyle(.secondary)
            Text("No scans yet")
                .font(.headline)
            Text(
                "Scans you record show up here. Each one keeps its photos and its "
                + "laser measurements, so you can come back to it later."
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var libraryUnreadable: some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 38, weight: .light))
                .foregroundStyle(.secondary)
            Text("The scans folder could not be read")
                .font(.headline)
            Text(store.problem ?? "")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Actions

    private var renamingBinding: Binding<Bool> {
        Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })
    }

    private var deletingBinding: Binding<Bool> {
        Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })
    }

    private func commitRename() {
        guard let scan = renaming else { return }
        let name = newName
        renaming = nil
        Task {
            if let problem = await store.rename(scan, to: name) {
                actionProblem = "This scan could not be renamed: \(problem)"
            }
        }
    }

    private func commitDelete() {
        guard let scan = deleting else { return }
        deleting = nil
        Task {
            if let problem = await store.delete(scan) {
                actionProblem = "This scan could not be moved to the trash: \(problem)"
            }
        }
    }
}

// MARK: - One row

struct ScanLibraryRow: View {
    let scan: ScanSummary

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            ScanThumbnailView(scan: scan)
                .frame(width: 64, height: 64)
                .clipShape(RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                Text(scan.displayName)
                    .font(.body.weight(.medium))
                    .lineLimit(1)

                Text(scan.nextStep)
                    .font(.footnote)
                    .foregroundStyle(scan.problem == nil ? Color.secondary : Color.orange)
                    .lineLimit(2)

                Text(detailLine)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)

                if let severity = scan.worstFindingSeverity, severity != .good {
                    Label(findingLine(severity), systemImage: findingIcon(severity))
                        .font(.caption)
                        .foregroundStyle(severity == .problem ? Color.orange : .secondary)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var detailLine: String {
        var parts: [String] = [ViewerFormat.date(scan.createdAt)]
        if let duration = scan.durationSeconds, duration > 0 {
            parts.append(ViewerFormat.duration(duration))
        }
        if scan.frameCount > 0 {
            parts.append("\(scan.frameCount) photos")
        }
        if let splats = scan.splatCount {
            parts.append(ViewerFormat.splatCount(splats))
        }
        parts.append(ViewerFormat.bytes(scan.byteCount))
        return parts.joined(separator: "  -  ")
    }

    private func findingLine(_ severity: QCFinding.Severity) -> String {
        let count = scan.qcFindingCount
        let noun = count == 1 ? "note" : "notes"
        return severity == .problem
            ? "\(count) \(noun) about how this one came out"
            : "\(count) \(noun) worth a look"
    }

    private func findingIcon(_ severity: QCFinding.Severity) -> String {
        severity == .problem ? "exclamationmark.circle" : "info.circle"
    }
}

// MARK: - Thumbnail

/// The row's picture: a real frame from the middle of the walk, loaded off the
/// main thread and downscaled before it is held.
///
/// A grey rectangle when the scan has no photo yet, which is a true statement
/// about a capture that was interrupted, rather than a spinner that never
/// stops.
struct ScanThumbnailView: View {
    let scan: ScanSummary

    @State private var image: UIImage?
    @State private var didTry = false

    var body: some View {
        ZStack {
            Rectangle()
                .fill(Color.secondary.opacity(0.15))
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else if didTry {
                Image(systemName: "photo")
                    .foregroundStyle(.tertiary)
            }
        }
        .clipped()
        .task(id: scan.scanID) { await load() }
    }

    private func load() async {
        guard image == nil, !didTry else { return }
        guard let relative = scan.thumbnailRelativePath else {
            didTry = true
            return
        }
        let url = scan.paths.url(relative)
        let data = await Task.detached(priority: .utility) { () -> Data? in
            guard let raw = try? Data(contentsOf: url, options: [.mappedIfSafe]) else {
                return nil
            }
            // Downscale on the way in: a library of forty rows must not hold
            // forty full-resolution photos in memory.
            guard let full = UIImage(data: raw) else { return nil }
            let side: CGFloat = 192
            let scale = min(side / max(full.size.width, 1), side / max(full.size.height, 1))
            guard scale < 1 else { return raw }
            let size = CGSize(
                width: full.size.width * scale,
                height: full.size.height * scale
            )
            let renderer = UIGraphicsImageRenderer(size: size)
            let small = renderer.image { _ in
                full.draw(in: CGRect(origin: .zero, size: size))
            }
            return small.jpegData(compressionQuality: 0.8) ?? raw
        }.value

        didTry = true
        // Turned upright only as it is shown. The file itself is the raw
        // sensor frame and stays that way, because the intrinsics and poses
        // beside it describe that frame.
        if let data, let decoded = UIImage(data: data) {
            image = ViewerPhoto.upright(
                decoded,
                quarterTurnsClockwise: scan.photoQuarterTurns
            )
        }
    }
}
