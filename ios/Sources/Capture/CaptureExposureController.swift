//
//  CaptureExposureController.swift
//  Capture
//
//  Exposure lock, white-balance lock, and the ~3-stop dark bracket (F5, F8).
//
//  HOW THIS IS POSSIBLE AT ALL. ARKit owns the capture device, so the usual
//  `AVCaptureSession` route is closed. Since iOS 16, ARKit hands the device
//  back through `ARConfiguration.configurableCaptureDeviceForPrimaryCamera`,
//  and that is the only supported way to set manual exposure inside an ARKit
//  session. On a device or OS where that property is nil, bracketing is
//  reported as unavailable and `CaptureSettings.bracketEveryNFrames` is
//  written as 0 - the format's own way of saying "bracketing was off" - rather
//  than pretending frames were bracketed when they were not.
//
//  WHY THE GUARD IS ONE-WAY. Taking manual control of exposure is the one
//  capture-side feature in this app that can actively damage the result: a
//  three-stop-dark frame has fewer trackable features, and if it lands during
//  a turn, ARKit's VIO has both a feature drought and real motion to explain
//  at the same instant. So bracketing is attempted only while tracking is
//  normal and the phone is turning slowly, and if tracking degrades within a
//  few frames of a bracket the feature switches itself off for the rest of the
//  session and says so. A scan that tracks is worth more than a scan with good
//  window pixels.
//

import ARKit
import AVFoundation
import CoreMedia
import Foundation
import QuartzCore

/// Drives manual exposure on the ARKit capture device.
///
/// Thread-safe: the pipeline queue drives it and AVFoundation completion
/// handlers land on an arbitrary queue, so all mutable state is behind a lock.
final class CaptureExposureController: @unchecked Sendable {

    /// What the controller is doing right now.
    private enum State {
        /// Normal auto (or user-locked) exposure.
        case idle
        /// A dark bracket has been requested and not yet taken effect.
        case applying
        /// The dark exposure is live. Frames at or after `sinceTimestamp` are
        /// the dark ones.
        case active(sinceTimestamp: TimeInterval)
        /// Restoring; frames may still be dark until the restore lands.
        case restoring
    }

    /// Why bracketing is not running, when it is not.
    enum Unavailability: String {
        case deviceNotConfigurable
        case configurationFailed
        case disabledByTrackingGuard
        case turnedOffByUser
    }

    private let lock = NSLock()
    private var state: State = .idle
    private var lastBracketEndedAt: TimeInterval = 0
    private var framesSinceBracket: Int = .max
    private var keyframesSinceBracket: Int = 0

    private var bracketingEnabled = true
    private(set) var unavailability: Unavailability?

    /// Exposure the user locked the session at, when they did. Restoring after
    /// a bracket returns here rather than to auto.
    private var lockedExposure: (duration: CMTime, iso: Float)?

    private(set) var isExposureLocked = false
    private(set) var isWhiteBalanceLocked = false

    /// The capture device ARKit is using, when the OS will hand it over.
    private var device: AVCaptureDevice? {
        ARConfiguration.configurableCaptureDeviceForPrimaryCamera
    }

    // MARK: - Availability

    /// Whether a dark bracket can actually be taken on this device, right now.
    var isBracketingAvailable: Bool {
        lock.lock()
        defer { lock.unlock() }
        guard bracketingEnabled else { return false }
        guard device != nil else { return false }
        return true
    }

    /// Called once the session is running. ARKit only publishes the capture
    /// device after it owns it, so this cannot be answered at configuration
    /// time.
    func sessionDidStart() {
        capExposureDuration()
        lock.lock()
        let hasDevice = device != nil
        if !hasDevice {
            bracketingEnabled = false
            unavailability = .deviceNotConfigurable
        }
        lock.unlock()
        if !hasDevice {
            let message = "No configurable capture device: exposure bracketing is off "
                    + "for this session and will be recorded as off."
            CaptureLog.exposure.notice("\(message, privacy: .public)")
        }
    }

    /// Stops the camera choosing a shutter long enough to make the blur
    /// meter unsatisfiable.
    ///
    /// Nothing capped this before. In a dim room auto-exposure would go to
    /// 1/15 s or slower, at which point the blur meter goes red at about
    /// 5 degrees per second of turn, which is slower than an ordinary hand
    /// shakes. The app was telling the owner to slow down while he was
    /// already holding still, and he said it felt like it needed a gimbal.
    ///
    /// `activeMaxExposureDuration` only bounds the AUTO exposure algorithm.
    /// It makes the camera reach for ISO instead of shutter, which is the
    /// right trade here: noise is independent between frames and the
    /// trainer averages it out, while blur is not and it does not.
    ///
    /// Clamped into the range the active format actually supports, because
    /// setting a value outside it is a hard exception rather than an error
    /// return. A device that will not hand over its capture device gets
    /// nothing done here and carries on, same as bracketing.
    private func capExposureDuration() {
        guard let device else { return }
        let format = device.activeFormat
        let requested = CMTime(
            seconds: CaptureTuning.maxExposureDurationSeconds,
            preferredTimescale: 1_000_000
        )
        var capped = requested
        if CMTimeCompare(capped, format.minExposureDuration) < 0 {
            capped = format.minExposureDuration
        }
        if CMTimeCompare(capped, format.maxExposureDuration) > 0 {
            capped = format.maxExposureDuration
        }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.activeMaxExposureDuration = capped
            CaptureLog.exposure.notice(
                """
                Shutter capped at \(CMTimeGetSeconds(capped), privacy: .public) s so the blur meter is reachable by hand.
                """
            )
        } catch {
            CaptureLog.exposure.error(
                "Could not cap the shutter: \(error.localizedDescription, privacy: .public)"
            )
        }
    }

    // MARK: - Current sensor readings

    /// The device's live ISO, which is where `CaptureFrame.iso` comes from.
    ///
    /// `ARCamera` does not publish ISO, and `ARFrame` has no EXIF dictionary,
    /// so this is the honest source. It is optional in the format precisely
    /// because on a device that will not hand the capture device over, there
    /// is no way to know it, and inventing a number would be worse than a null.
    var currentISO: Float? {
        guard let device else { return nil }
        let iso = device.iso
        return iso.isFinite && iso > 0 ? iso : nil
    }

    // MARK: - Locks (F8)

    /// Locks or unlocks exposure for the whole session.
    ///
    /// - Returns: whether the request took effect. A false return is reported
    ///   to the user as "this phone will not let the app hold the exposure",
    ///   not swallowed.
    @discardableResult
    func setExposureLocked(_ locked: Bool) -> Bool {
        guard let device else { return false }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            if locked {
                let duration = device.exposureDuration
                let iso = device.iso
                if device.isExposureModeSupported(.locked) {
                    device.exposureMode = .locked
                } else if device.isExposureModeSupported(.custom) {
                    device.setExposureModeCustom(
                        duration: duration,
                        iso: iso,
                        completionHandler: nil
                    )
                } else {
                    return false
                }
                lock.lock()
                lockedExposure = (duration, iso)
                isExposureLocked = true
                lock.unlock()
            } else {
                guard device.isExposureModeSupported(.continuousAutoExposure) else {
                    return false
                }
                device.exposureMode = .continuousAutoExposure
                lock.lock()
                lockedExposure = nil
                isExposureLocked = false
                lock.unlock()
            }
            return true
        } catch {
            CaptureLog.exposure.error(
                "Exposure lock failed: \(error.localizedDescription, privacy: .public)"
            )
            return false
        }
    }

    @discardableResult
    func setWhiteBalanceLocked(_ locked: Bool) -> Bool {
        guard let device else { return false }
        let mode: AVCaptureDevice.WhiteBalanceMode =
            locked ? .locked : .continuousAutoWhiteBalance
        guard device.isWhiteBalanceModeSupported(mode) else { return false }
        do {
            try device.lockForConfiguration()
            device.whiteBalanceMode = mode
            device.unlockForConfiguration()
            lock.lock()
            isWhiteBalanceLocked = locked
            lock.unlock()
            return true
        } catch {
            let message = "White balance lock failed: "
                    + "\(error.localizedDescription)"
            CaptureLog.exposure.error("\(message, privacy: .public)")
            return false
        }
    }

    // MARK: - Bracketing

    /// Records that a keyframe was written, so the "every Nth" cadence counts
    /// keyframes rather than delivered frames. A user standing still delivers
    /// hundreds of frames and writes none; bracketing off that count would fire
    /// constantly while nothing was being recorded.
    func keyframeWasWritten() {
        lock.lock()
        keyframesSinceBracket += 1
        lock.unlock()
    }

    /// Whether the next keyframe should be a dark bracket.
    ///
    /// - Parameters:
    ///   - everyN: cadence, from `CaptureTuning`; higher in window mode.
    ///   - angularSpeed: current turn rate, rad/s.
    ///   - trackingQuality: ARKit's verdict this instant.
    ///   - now: `ARFrame.timestamp`.
    func shouldBracketNextKeyframe(
        everyN: Int,
        angularSpeed: Float,
        trackingQuality: TrackingQuality,
        now: TimeInterval
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard bracketingEnabled, everyN > 0, device != nil else { return false }
        guard case .idle = state else { return false }
        guard trackingQuality == .normal else { return false }
        guard angularSpeed <= CaptureTuning.bracketMaxAngularSpeedRadPerSec else {
            return false
        }
        guard now - lastBracketEndedAt >= CaptureTuning.bracketCooldownSeconds else {
            return false
        }
        return keyframesSinceBracket >= everyN
    }

    /// Asks the device for a ~3-stop-darker exposure.
    ///
    /// ISO is reduced first and the shutter only afterwards, and the order
    /// matters: shortening the shutter would change the frame's motion blur,
    /// which would make the bracketed frame differ from its neighbours in two
    /// ways instead of one and corrupt the very comparison it exists for.
    func beginBracket() {
        guard let device else { return }
        lock.lock()
        guard bracketingEnabled, case .idle = state else {
            lock.unlock()
            return
        }
        state = .applying
        lock.unlock()

        let desiredFactor = powf(2, -CaptureTuning.bracketStops)  // 3 stops -> 1/8
        let format = device.activeFormat
        let currentISO = device.iso
        let currentDuration = device.exposureDuration

        let minISO = format.minISO
        let targetISO = max(minISO, currentISO * desiredFactor)
        let isoFactor = currentISO > 0 ? targetISO / currentISO : 1
        let remainingFactor = isoFactor > 0 ? desiredFactor / isoFactor : 1

        var targetDuration = CMTimeMultiplyByFloat64(
            currentDuration,
            multiplier: Float64(remainingFactor)
        )
        let minDuration = format.minExposureDuration
        if CMTimeCompare(targetDuration, minDuration) < 0 {
            targetDuration = minDuration
        }

        guard device.isExposureModeSupported(.custom) else {
            lock.lock()
            bracketingEnabled = false
            unavailability = .configurationFailed
            state = .idle
            lock.unlock()
            return
        }

        do {
            try device.lockForConfiguration()
            device.setExposureModeCustom(
                duration: targetDuration,
                iso: targetISO
            ) { [weak self] syncTime in
                guard let self else { return }
                let effectiveAt = CMTimeGetSeconds(syncTime)
                self.lock.lock()
                // A non-finite sync time means AVFoundation could not tell us
                // when it landed; the current media time is the same clock and
                // is close enough that the next frame is still tagged right.
                let stamp = effectiveAt.isFinite && effectiveAt > 0
                    ? effectiveAt
                    : CACurrentMediaTime()
                self.state = .active(sinceTimestamp: stamp)
                self.framesSinceBracket = 0
                self.lock.unlock()
            }
            device.unlockForConfiguration()
            let message = "Bracket: ISO \(String(format: "%.0f", currentISO)) -> "
                    + "\(String(format: "%.0f", targetISO))"
            CaptureLog.exposure.debug("\(message, privacy: .public)")
        } catch {
            lock.lock()
            state = .idle
            bracketingEnabled = false
            unavailability = .configurationFailed
            lock.unlock()
            let message = "Could not set a dark bracket: "
                    + "\(error.localizedDescription). "
                    + "Bracketing is off for the rest of this session."
            CaptureLog.exposure.error("\(message, privacy: .public)")
        }
    }

    /// Whether the frame at `timestamp` is one of the dark ones.
    func bracketClass(forFrameAt timestamp: TimeInterval) -> ExposureBracket {
        lock.lock()
        defer { lock.unlock() }
        switch state {
        case .active(let since) where timestamp >= since:
            return .darker
        case .restoring:
            // Still possibly dark; a frame in flight when the restore was
            // requested has not seen the new settings yet. Tagging it darker
            // is the conservative error: an ordinary frame mislabelled dark
            // gets a per-frame exposure the trainer will solve for anyway,
            // while a dark frame mislabelled normal poisons a photometric
            // comparison.
            return .darker
        default:
            return .normal
        }
    }

    /// Returns exposure to where it was before the bracket.
    func endBracket(at now: TimeInterval) {
        guard let device else { return }
        lock.lock()
        guard case .active = state else {
            lock.unlock()
            return
        }
        state = .restoring
        let restoreTo = lockedExposure
        lock.unlock()

        do {
            try device.lockForConfiguration()
            if let restoreTo {
                device.setExposureModeCustom(
                    duration: restoreTo.duration,
                    iso: restoreTo.iso
                ) { [weak self] _ in
                    self?.finishRestore(at: now)
                }
            } else if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
                device.unlockForConfiguration()
                finishRestore(at: now)
                return
            } else {
                device.unlockForConfiguration()
                finishRestore(at: now)
                return
            }
            device.unlockForConfiguration()
        } catch {
            let message = "Could not restore exposure after a bracket: "
                    + "\(error.localizedDescription)"
            CaptureLog.exposure.error("\(message, privacy: .public)")
            finishRestore(at: now)
        }
    }

    /// Gives up on a dark exposure that was asked for and never arrived.
    ///
    /// `beginBracket` moves the controller to `.applying` and only
    /// `setExposureModeCustom`'s completion handler moves it on. If that
    /// handler never fires - the session was interrupted, the capture device
    /// was taken away mid-request - the controller would sit in `.applying`
    /// for the rest of the session and `shouldBracketNextKeyframe` would never
    /// return true again. This is the way out: it clears the request, restarts
    /// the cadence, and does nothing at all if a bracket is genuinely live.
    func abandonPendingBracket(at now: TimeInterval) {
        lock.lock()
        guard case .applying = state else {
            lock.unlock()
            return
        }
        state = .idle
        lastBracketEndedAt = now
        keyframesSinceBracket = 0
        lock.unlock()
        let message = "A dark bracket was asked for and never arrived. The request has "
                + "been dropped and the cadence starts again."
        CaptureLog.exposure.notice("\(message, privacy: .public)")
    }

    private func finishRestore(at now: TimeInterval) {
        lock.lock()
        state = .idle
        lastBracketEndedAt = now
        keyframesSinceBracket = 0
        lock.unlock()
    }

    /// Feeds the one-way tracking guard. Called once per delivered frame.
    ///
    /// If tracking stops being normal within a few frames of a bracket, the
    /// bracket is the prime suspect and the feature turns itself off.
    func observeTracking(_ quality: TrackingQuality) {
        lock.lock()
        if framesSinceBracket < CaptureTuning.bracketTrackingGuardFrames {
            framesSinceBracket += 1
            if quality != .normal, bracketingEnabled {
                bracketingEnabled = false
                unavailability = .disabledByTrackingGuard
                lock.unlock()
                let message = "Tracking degraded right after a dark bracket. Bracketing "
                        + "is off for the rest of this session."
                CaptureLog.exposure.notice("\(message, privacy: .public)")
                return
            }
        }
        lock.unlock()
    }

    /// Whether the user is the reason bracketing is off.
    ///
    /// Deliberately narrower than `!isBracketingAvailable`: the tracking guard
    /// and a device that will not be configured also turn bracketing off, and
    /// a switch in the HUD must not claim the user did that.
    var isTurnedOffByUser: Bool {
        lock.lock()
        defer { lock.unlock() }
        return unavailability == .turnedOffByUser
    }

    /// The user's own switch, from the capture screen.
    func setBracketingEnabledByUser(_ enabled: Bool) {
        lock.lock()
        if enabled {
            if unavailability == .turnedOffByUser {
                unavailability = nil
                bracketingEnabled = true
            }
        } else {
            bracketingEnabled = false
            unavailability = .turnedOffByUser
        }
        lock.unlock()
    }

    /// What to write into `CaptureSettings.bracketEveryNFrames`: the real
    /// cadence when brackets actually happened, 0 when they did not.
    func settingsCadence(requested: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return bracketingEnabled ? requested : 0
    }
}
