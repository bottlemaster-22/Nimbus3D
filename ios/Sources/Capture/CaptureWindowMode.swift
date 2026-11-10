//
//  CaptureWindowMode.swift
//  Capture
//
//  WINDOW MODE (F5, F9): what changes when the phone is pointed at the one
//  thing in the room that the sensor cannot measure and the camera cannot
//  expose for.
//
//  A window is the hardest surface in a scan and it fails in three ways at
//  once: the laser goes straight through and returns nothing, so there is no
//  geometry; the glass is many stops brighter than the room, so every pixel is
//  clipped white; and there is nothing to walk around, so extra viewpoints do
//  not help. Generic scanners produce either a white hole or a solid wall of
//  fog where the window was.
//
//  So when ARKit's classifier says the middle of the screen is a window, four
//  things change and the user is told about all of them in one sentence:
//
//    1. EXPOSURE LOCKS, so auto-exposure stops pumping between the bright
//       glass and the dark room and destroying photometric consistency.
//    2. BRACKETING GETS MORE FREQUENT (every 5th keyframe instead of every
//       12th), because a dark frame is the only correctly exposed evidence
//       this scan will ever have for what is outside.
//    3. THE USER IS ASKED TO STAND BACK, because at 1.5 m or more the window
//       frame and the wall around it are both in view, and the frame is what
//       the geometry actually hangs off.
//    4. THE REGION IS MARKED. Not as a new file: the window faces are already
//       carried in `mesh/chunk_*.cls` and the coverage field already stores a
//       per-voxel class, so the pre-pass's glass detector (F5) has what it
//       needs to confirm or reject the label. This module deliberately does
//       NOT write `SurfaceClass.glass` anywhere - that class means "we
//       detected it, LiDAR-silent and image-bright and planar", and detecting
//       it is the pre-pass's job. Writing it here would put an unearned label
//       in the file.
//
//  HYSTERESIS IS NOT OPTIONAL. A mode that engages at 18% and disengages at
//  17% will flicker every time the user's hand moves, and a flickering
//  exposure lock is worse than no exposure lock at all. Enter at 18%, leave at
//  8%, and never in under 1.5 seconds.
//

import Foundation
import simd

/// Detects "the user is looking at a window" and holds the mode steady.
///
/// Owned by, and only touched from, the capture pipeline's serial queue.
/// `@unchecked Sendable` for the same reason `CaptureFrameWriter` is: the
/// serial queue is the synchronisation.
final class CaptureWindowMode: @unchecked Sendable {

    /// Fraction of the sampled centre region currently classified window or
    /// glass, 0...1. Exposed for the HUD's detail view.
    private(set) var centreWindowFraction: Float = 0

    private(set) var isActive = false
    private var enteredAt: TimeInterval = 0

    /// Distance to the nearest thing in the centre of frame, metres, or nil
    /// when the centre is entirely no-return - which is itself the classic
    /// glass signature and is why "no depth at all" does not mean "no
    /// distance to report".
    private(set) var centreDistanceMeters: Float?

    func reset() {
        centreWindowFraction = 0
        isActive = false
        enteredAt = 0
        centreDistanceMeters = nil
    }

    /// Re-evaluates the mode for one frame.
    ///
    /// Works off the NATIVE depth map rather than raycasting the mesh: the
    /// samples are already in hand, unprojecting a strided centre region is a
    /// few hundred multiplies, and the class voxel map turns each one into a
    /// class with a dictionary lookup. Raycasting per pixel would cost
    /// hundreds of times more and answer the same question.
    ///
    /// - Returns: true when the mode changed state on this call, so the caller
    ///   knows when to announce it rather than announcing it every frame.
    @discardableResult
    func update(
        depth: CaptureDepthFrame,
        pose: Pose,
        intrinsics: CameraIntrinsics,
        meshStore: CaptureMeshStore,
        now: TimeInterval
    ) -> Bool {
        let cameraToWorld = pose.matrix.inverse

        // The middle third of the frame in both axes: what the user is
        // pointing at, not what is at the edge of their peripheral vision.
        let x0 = depth.width / 3
        let x1 = depth.width - x0
        let y0 = depth.height / 3
        let y1 = depth.height - y0
        let step = 2

        var classified = 0
        var windowish = 0
        var nearestDistance = Float.greatestFiniteMagnitude

        var y = y0
        while y < y1 {
            var x = x0
            while x < x1 {
                guard
                    let cameraPoint = depth.unprojectToCameraSpace(
                        x: x,
                        y: y,
                        intrinsics: intrinsics
                    )
                else {
                    x += step
                    continue
                }
                nearestDistance = Swift.min(
                    nearestDistance,
                    simd_length(cameraPoint)
                )
                let world = cameraToWorld * SIMD4<Float>(cameraPoint, 1)
                let surface = meshStore.surfaceClass(
                    at: SIMD3<Float>(world.x, world.y, world.z)
                )
                if surface != .none {
                    classified += 1
                    if surface == .window || surface == .glass {
                        windowish += 1
                    }
                }
                x += step
            }
            y += step
        }

        centreDistanceMeters = nearestDistance.isFinite
            && nearestDistance < .greatestFiniteMagnitude
            ? nearestDistance
            : nil

        // Denominator is CLASSIFIED samples, not all samples. A window's own
        // pixels are no-returns and contribute nothing either way; measuring
        // against every sample would make the fraction collapse exactly when
        // the user is closest to the glass, which is backwards.
        centreWindowFraction = classified > 0
            ? Float(windowish) / Float(classified)
            : 0

        return applyHysteresis(now: now)
    }

    /// True when the user is close enough to the glass that the window frame
    /// is out of shot and they should back off.
    var shouldStandBack: Bool {
        guard isActive, let distance = centreDistanceMeters else { return false }
        return distance < CaptureTuning.windowModeStandBackMeters
    }

    /// The bracket cadence to use right now.
    var bracketEveryNKeyframes: Int {
        isActive
            ? CaptureTuning.bracketEveryNKeyframesInWindowMode
            : CaptureTuning.bracketEveryNKeyframes
    }

    /// One plain sentence for the HUD and the speech channel.
    var guidanceHint: String? {
        guard isActive else { return nil }
        if shouldStandBack {
            return "Window: step back a little so the frame is in shot"
        }
        return "Window: holding the exposure steady"
    }

    // MARK: - Private

    private func applyHysteresis(now: TimeInterval) -> Bool {
        if isActive {
            guard centreWindowFraction < CaptureTuning.windowModeExitFraction else {
                return false
            }
            guard now - enteredAt >= CaptureTuning.windowModeMinDwellSeconds else {
                return false
            }
            isActive = false
            CaptureLog.exposure.debug("Window mode off.")
            return true
        }

        guard centreWindowFraction >= CaptureTuning.windowModeEnterFraction else {
            return false
        }
        isActive = true
        enteredAt = now
        let message = "Window mode on at "
                + "\(String(format: "%.2f", self.centreWindowFraction)) "
                + "of the classified centre."
        CaptureLog.exposure.debug("\(message, privacy: .public)")
        return true
    }
}
