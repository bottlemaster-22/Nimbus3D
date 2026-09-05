//
//  BoosterClient.swift
//  Booster
//
//  The one public entry point the rest of the app needs. Everything else in
//  this module (discovery, pairing, upload, live progress, download) is an
//  implementation detail behind this class and BoosterTabView.
//
//  PUBLIC SURFACE (App's NimbusApp.swift registers `BoosterClient.shared` as
//  NimbusServices.shared.booster via a `BoosterService` conformance, and
//  `BoosterTabView()` as the Booster tab - see CONTRACTS.md section 5):
//
//      BoosterTabView()                 - drop-in SwiftUI screen
//      BoosterClient.shared             - the singleton this view drives
//      BoosterClient.sendScan(...)      - programmatic "send this scan" call,
//                                          for a future Library screen's
//                                          "Send to Booster" button
//
//  Never required: every call here is opt-in and additive. Nothing in
//  Capture/PrePass/Trainer/Viewer/Export needs to know this type exists.
//

import Combine
import Foundation
import os

@MainActor
public final class BoosterClient: ObservableObject {

    public static let shared = BoosterClient()

    public let discovery = BoosterDiscovery()

    @Published public private(set) var activeUpload: BoosterUploadProgress?
    @Published public private(set) var activeJobProgress: BoosterProgressEvent?
    @Published public private(set) var activeDownload: BoosterDownloadProgress?
    @Published public private(set) var lastError: BoosterError?
    @Published public private(set) var isBusy = false

    private var currentTask: Task<Void, Never>?
    private var currentAddress: BoosterEndpointAddress?
    private var currentJobID: String?

    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem,
        category: "BoosterClient"
    )

    private init() {}

    /// Sends the scan folder at `scanDirectory` (a Documents/<brand>/Scans/
    /// <scanID> folder, per BrandConfig.Folder) to a paired, reachable
    /// Booster, watches its progress live, and downloads the finished result
    /// into that same scan's model/ folder when done.
    public func sendScan(
        scanID: String,
        scanDirectory: URL,
        to device: BoosterDevice
    ) {
        guard currentTask == nil else {
            lastError = .serverRejected("A scan is already being sent. Wait for it to finish.")
            return
        }
        guard device.isPaired else {
            lastError = .notPaired
            return
        }
        guard let boosterToken = BoosterKeychain.loadToken(boosterID: device.id) else {
            lastError = .notPaired
            return
        }

        isBusy = true
        lastError = nil
        activeUpload = nil
        activeJobProgress = nil
        activeDownload = nil

        currentTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.runSendScan(
                    scanID: scanID,
                    scanDirectory: scanDirectory,
                    device: device,
                    token: boosterToken
                )
            } catch let error as BoosterError {
                self.lastError = error
                self.log.error("sendScan failed: \(error.errorDescription ?? "", privacy: .public)")
            } catch is CancellationError {
                self.lastError = .jobCancelled
            } catch {
                self.lastError = .connectionFailed(error.localizedDescription)
            }
            self.isBusy = false
            self.currentTask = nil
        }
    }

    public func cancelActiveSend() {
        currentTask?.cancel()
        currentTask = nil
        guard let address = currentAddress, let jobID = currentJobID else {
            isBusy = false
            return
        }
        Task {
            try? await BoosterUploadManager(address: address).cancel(jobID: jobID)
        }
        isBusy = false
        lastError = .jobCancelled
    }

    /// Dismisses the currently shown error, e.g. after the user taps OK on
    /// the Booster tab's error alert.
    public func clearError() {
        lastError = nil
    }

    public func forgetPairing(boosterID: String) {
        BoosterJobStore.shared.forgetPairing(boosterID: boosterID)
    }

    public func resultsHistory() -> [BoosterJobRecord] {
        BoosterJobStore.shared.allJobs()
    }

    // MARK: - Private orchestration

    private func runSendScan(
        scanID: String,
        scanDirectory: URL,
        device: BoosterDevice,
        token: String
    ) async throws {
        // Captured once and reused for every BoosterJobRecord below. Each
        // progress update replaces the stored record wholesale (JobStore
        // matches by jobID), so re-stamping this with Date() on every event
        // would make "createdAt" silently drift to "last updated" and churn
        // the Results list's sort order mid-transfer.
        let jobCreatedAt = Date()

        let (host, port) = try await discovery.resolveHostPort(for: device)
        let address = BoosterEndpointAddress(host: host, port: port, token: token)
        currentAddress = address

        let uploader = BoosterUploadManager(address: address)
        let jobID = try await uploader.upload(
            scanID: scanID,
            scanDirectory: scanDirectory
        ) { [weak self] progress in
            Task { @MainActor in self?.activeUpload = progress }
        }
        currentJobID = jobID
        try Task.checkCancellation()

        BoosterJobStore.shared.upsertJob(
            BoosterJobRecord(
                jobID: jobID,
                scanID: scanID,
                boosterID: device.id,
                boosterName: device.name,
                createdAt: jobCreatedAt,
                stage: .receiving,
                message: "Sending to \(device.name)...",
                resultDirectory: nil
            )
        )

        try await uploader.finalize(jobID: jobID)
        activeUpload = nil

        let monitor = BoosterJobMonitor(address: address, jobID: jobID)
        var finalStage: BoosterJobStage = .queued
        for await event in await monitor.events() {
            try Task.checkCancellation()
            activeJobProgress = event
            finalStage = event.stage
            BoosterJobStore.shared.upsertJob(
                BoosterJobRecord(
                    jobID: jobID,
                    scanID: scanID,
                    boosterID: device.id,
                    boosterName: device.name,
                    createdAt: jobCreatedAt,
                    stage: event.stage,
                    message: event.message,
                    resultDirectory: nil
                )
            )
        }

        guard finalStage == .ready else {
            if finalStage == .cancelled {
                throw BoosterError.jobCancelled
            }
            throw BoosterError.jobFailed(activeJobProgress?.message ?? "Unknown error.")
        }

        let destination = try Self.modelDirectory(scanID: scanID)
        let downloader = BoosterDownloadManager(address: address)
        _ = try await downloader.downloadResult(jobID: jobID, into: destination) {
            [weak self] progress in
            Task { @MainActor in self?.activeDownload = progress }
        }
        activeDownload = nil

        BoosterJobStore.shared.upsertJob(
            BoosterJobRecord(
                jobID: jobID,
                scanID: scanID,
                boosterID: device.id,
                boosterName: device.name,
                createdAt: jobCreatedAt,
                stage: .ready,
                message: "Finished. Saved to your library.",
                resultDirectory: destination.path
            )
        )

        currentAddress = nil
        currentJobID = nil
    }

    private static func modelDirectory(scanID: String) throws -> URL {
        let scans = try BrandConfig.scansDirectory()
        return scans
            .appendingPathComponent(scanID, isDirectory: true)
            .appendingPathComponent(BrandConfig.Folder.model, isDirectory: true)
    }
}
