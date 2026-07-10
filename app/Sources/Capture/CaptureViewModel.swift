//
//  CaptureViewModel.swift — MainActor bridge between ARCaptureService and SwiftUI.
//  Owned by the Capture agent.
//
//  Subscribes to the service's CaptureEvent stream and republishes it as observable
//  state (coverage, frame count, tracking quality, guidance copy), and exposes the
//  start / finish / cancel / phase actions the capture screen drives.
//

import Foundation
import SwiftUI
import AVFoundation

@MainActor
final class CaptureViewModel: ObservableObject {

    let service = ARCaptureService()

    // MARK: - Published UI state

    @Published var isCapturing = false
    @Published var coverage: Float = 0
    @Published var frameCount = 0
    @Published var targetCount = 100
    @Published var trackingQuality: TrackingQuality = .notAvailable
    @Published var phase: ARCaptureService.Phase = .object
    @Published var guidance = "Aim at your object, then tap Start."
    @Published var bundle: CaptureBundle?
    @Published var errorMessage: String?

    // MARK: - Session options (bound to the setup controls)

    @Published var targetFrameSetting = 100
    @Published var captureLiDAR = true
    @Published var captureHDRI = true

    // MARK: - Static capabilities (for the UI to show badges / hide toggles)

    let worldTrackingSupported = CaptureDeviceCapabilities.isWorldTrackingSupported
    let lidarSupported = CaptureDeviceCapabilities.supportsLiDARDepth

    private var eventTask: Task<Void, Never>?

    init() {
        listenForEvents()
    }

    deinit {
        eventTask?.cancel()
    }

    // MARK: - Actions

    func start() {
        errorMessage = nil
        bundle = nil
        Task { @MainActor in
            guard await Self.ensureCameraPermission() else {
                errorMessage = "Camera access is required. Enable it for Nimbus3D in Settings."
                return
            }
            do {
                try await service.startCapture(options: makeOptions())
                phase = .object
                frameCount = 0
                coverage = 0
                targetCount = targetFrameSetting
                isCapturing = true
                guidance = "Move slowly around the object, keeping it centered."
            } catch {
                errorMessage = describe(error)
            }
        }
    }

    func finish() {
        Task { @MainActor in
            do {
                let result = try await service.finishCapture()
                bundle = result
                isCapturing = false
                guidance = "Capture complete. \(result.frames.count) frames saved."
            } catch {
                isCapturing = false
                errorMessage = describe(error)
            }
        }
    }

    func cancel() {
        Task { @MainActor in
            await service.cancelCapture()
            isCapturing = false
            coverage = 0
            frameCount = 0
            phase = .object
            guidance = "Capture cancelled. Aim at your object, then tap Start."
        }
    }

    func setPhase(_ newPhase: ARCaptureService.Phase) {
        phase = newPhase
        service.setPhase(newPhase)
        guidance = Self.guidanceMessage(phase: newPhase,
                                        tracking: trackingQuality,
                                        coverage: coverage,
                                        frames: frameCount,
                                        target: targetCount,
                                        capturing: isCapturing)
    }

    func startNewCapture() {
        bundle = nil
        errorMessage = nil
        guidance = "Aim at your object, then tap Start."
    }

    // MARK: - Event loop

    private func listenForEvents() {
        // Capture the stream (not self) so the Task does not retain the view model.
        // Task inherits MainActor isolation, so apply(_:) runs on the main actor.
        let events = service.events
        eventTask = Task { [weak self] in
            for await event in events {
                guard let self else { break }
                self.apply(event)
            }
        }
    }

    private func apply(_ event: CaptureEvent) {
        switch event {
        case .trackingStateChanged(let quality):
            trackingQuality = quality
        case .frameCaptured(let index, let target):
            frameCount = index + 1
            targetCount = target
        case .coverageUpdated(let value):
            coverage = value
        case .warning(let message):
            errorMessage = message
        }
        guidance = Self.guidanceMessage(phase: phase,
                                        tracking: trackingQuality,
                                        coverage: coverage,
                                        frames: frameCount,
                                        target: targetCount,
                                        capturing: isCapturing)
    }

    // MARK: - Helpers

    private func makeOptions() -> CaptureOptions {
        CaptureOptions(
            targetFrameCount: targetFrameSetting,
            captureLiDARDepth: captureLiDAR && lidarSupported,
            captureHDRIBrackets: captureHDRI,
            outputDirectory: nil
        )
    }

    private func describe(_ error: Error) -> String {
        (error as? NimbusError)?.errorDescription ?? error.localizedDescription
    }

    static func ensureCameraPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        default:
            return false
        }
    }

    static func guidanceMessage(phase: ARCaptureService.Phase,
                                tracking: TrackingQuality,
                                coverage: Float,
                                frames: Int,
                                target: Int,
                                capturing: Bool) -> String {
        guard capturing else {
            return "Aim at your object, then tap Start."
        }
        switch tracking {
        case .notAvailable:
            return "Starting tracking. Hold the phone steady."
        case .limited:
            return "Tracking limited. Move slower and add more light."
        case .normal:
            break
        }

        switch phase {
        case .environment:
            return "Slowly pan around to capture the lighting for HDRI."
        case .object:
            let percent = Int((coverage * 100).rounded())
            if frames >= target {
                return "Target reached (\(frames) frames). Finish, or keep filling gaps."
            }
            if coverage < 0.35 {
                return "Keep orbiting the object. Coverage \(percent)%."
            }
            if coverage < 0.85 {
                return "Good. Fill the remaining angles. Coverage \(percent)%."
            }
            return "Almost fully covered (\(percent)%). Capture top and bottom, then Finish."
        }
    }
}
