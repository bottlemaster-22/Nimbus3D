//
//  CaptureTuning.swift
//  Capture
//
//  Every tunable number the capture module uses, in one place, each with the
//  reason it has the value it has. Nothing else in this directory may declare
//  a magic constant: if a number needs justifying, it belongs here next to its
//  justification.
//
//  Values that the whole product agrees on (native depth size, LiDAR range,
//  the blur-meter divisor, the bracket stops) are quoted from CONTRACTS.md
//  section 7 and docs/DATA_FORMAT.md. Where this file repeats one of those,
//  the comment says so, so a future change lands in both places knowingly.
//

import CoreGraphics
import Foundation

/// Capture-time constants. Namespaced as an enum so nothing can instantiate it.
public enum CaptureTuning {

    // MARK: - Frame rate and keyframe selection

    /// Upper bound on how often a frame is even *considered* for keyframing.
    /// ARKit delivers 60 fps; evaluating QC and coverage on all of them is
    /// wasted heat, and two frames 16 ms apart are the same photograph.
    public static let frameEvaluationHz: Double = 15

    /// A new keyframe needs at least this much baseline from the last one.
    /// Below ~5 cm two views are nearly the same ray bundle and add nothing a
    /// splat optimiser can triangulate against.
    public static let keyframeMinBaselineMeters: Float = 0.05

    /// ...or at least this much rotation, for the pan-in-place case where the
    /// camera barely translates but sees a completely different wall.
    public static let keyframeMinRotationDegrees: Float = 4.0

    /// Never two keyframes closer together than this in time, whatever the
    /// motion says. Caps the write rate and therefore the storage burn.
    public static let keyframeMinIntervalSeconds: Double = 1.0 / 6.0

    /// If nothing has been written for this long, take one anyway. A user who
    /// stands perfectly still while thinking should not leave a hole in the
    /// temporal track that the pre-pass's submap builder then has to bridge.
    public static let keyframeMaxIntervalSeconds: Double = 2.0

    /// A frame whose QC weight is below this is not written unless
    /// `keyframeMaxIntervalSeconds` has elapsed. Not zero: a slightly blurry
    /// frame in an otherwise unseen direction is worth more than nothing.
    public static let keyframeMinQCWeight: Float = 0.25

    // MARK: - The blur meter (F9)

    /// Angular pixel pitch of the wide camera at 1920 px, degrees per pixel.
    ///
    /// CONTRACTS.md section 7 fixes the blur meter as
    /// `degreesPerSecond * exposureDuration / 0.0426`, so this divisor is a
    /// contract value, not a free parameter. It is the pitch of the capture
    /// stream the app requests; it is deliberately NOT recomputed from
    /// per-device intrinsics, because the amber/red thresholds below and the
    /// QC card downstream are calibrated against this exact number.
    public static let angularPixelPitchDegrees: Float = 0.0426

    /// Smear in pixels at which the HUD goes amber ("slow down").
    public static let blurAmberPixels: Float = 2.0
    /// Smear in pixels at which the HUD goes red and guidance interrupts.
    public static let blurRedPixels: Float = 4.0

    // MARK: - Exposure bracketing (F5)

    /// Every Nth *keyframe* is captured darker so a bright window has
    /// unsaturated pixels somewhere in the dataset.
    public static let bracketEveryNKeyframes: Int = 12

    /// In window mode the phone is pointed at the one thing in the room that
    /// blows out, so brackets get more frequent.
    public static let bracketEveryNKeyframesInWindowMode: Int = 5

    /// How much darker, in stops. Three stops is 1/8 the light.
    public static let bracketStops: Float = 3.0

    /// A bracket is only attempted while tracking is `.normal` and the phone
    /// is turning slower than this. Changing exposure mid-turn is the one way
    /// this feature can actually hurt: VIO loses features on the dark frame
    /// *and* has motion to explain at the same time.
    public static let bracketMaxAngularSpeedRadPerSec: Float = 0.35

    /// After a bracket, wait this long before allowing another, so the
    /// auto-exposure ramp has settled and the tracker has a clean run of
    /// normally-exposed frames.
    public static let bracketCooldownSeconds: Double = 0.75

    /// If tracking degrades within this many frames of a bracket, bracketing
    /// is disabled for the rest of the session and the reason is logged. The
    /// guard is one-way on purpose: a scan that tracks is worth more than a
    /// scan with good window pixels.
    public static let bracketTrackingGuardFrames: Int = 6

    /// How long to wait for a requested dark exposure to actually appear in a
    /// delivered frame before giving up on that request.
    ///
    /// `AVCaptureDevice.setExposureModeCustom` normally lands within two or
    /// three frames. If its completion handler never fires (the device was
    /// taken away, the session was interrupted mid-request) the request would
    /// otherwise sit half-applied forever and no further bracket could be
    /// asked for, so the request is abandoned and the cadence starts again.
    public static let bracketApplyTimeoutSeconds: Double = 1.5

    // MARK: - Coverage (F9)

    /// Edge length of a coverage voxel. Coarser than the occupancy grid (5 cm,
    /// CONTRACTS.md section 7) because coverage is a UI signal about a patch of
    /// wall, not a geometric decision about a Gaussian.
    public static let coverageVoxelSizeMeters: Float = 0.10

    /// Directions on the sphere tracked per voxel, as a bit in a `UInt32`.
    /// 32 buckets over the hemisphere a surface can be seen from is about
    /// 20 degrees of angular resolution, which is the granularity at which
    /// "walk around it a bit more" is useful advice.
    public static let coverageDirectionBuckets: Int = 32

    /// Distinct directions at which the "angles seen" channel reads 1.0.
    /// Four well-separated viewpoints is the practical floor for a surface to
    /// be reconstructable rather than guessed.
    public static let coverageDirectionsForFull: Int = 4

    /// The distance band the LiDAR and the camera agree best at. Closer than
    /// `min` and the depth map's 256x192 footprint is bigger than the detail;
    /// further than `max` and the laser's return is thin.
    public static let coverageIdealDistanceMinMeters: Float = 0.6
    public static let coverageIdealDistanceMaxMeters: Float = 2.5

    /// Beyond this the surface is out of the near regime entirely and the
    /// distance channel reads 0 (F5's 4.5 m near/mid boundary).
    public static let coverageMaxUsefulDistanceMeters: Float = 4.5

    /// Sharpness (0...1) at which the sharpness channel is satisfied.
    public static let coverageSharpnessTarget: Float = 0.55

    /// A voxel counts as done when all three channels are at or above this.
    public static let coverageChannelDoneThreshold: Float = 0.7

    /// Coverage fraction at which the app tells the user they are finished.
    /// Not 1.0: a real room has surfaces behind the sofa that nobody is going
    /// to reach, and demanding perfection is how a capture UI becomes a chore.
    public static let coverageDoneFraction: Float = 0.85

    /// Voxels seen fewer than this many times are ignored by the coverage
    /// percentage. A single grazing hit on a distant surface should not create
    /// a permanent red patch the user cannot clear.
    public static let coverageMinObservationsToCount: Int = 2

    /// Stride through the native depth map when feeding the coverage field.
    /// 2 gives ~12k samples per frame at 256x192, which is plenty for a 10 cm
    /// voxel grid and a quarter of the work.
    public static let coverageDepthStride: Int = 2

    /// How often the coverage field is updated, Hz. Coverage changes at
    /// walking speed, not at frame rate.
    public static let coverageUpdateHz: Double = 8

    /// How often the coverage PERCENTAGE is recomputed, Hz.
    ///
    /// Slower than the field update on purpose: recomputing the fraction walks
    /// every voxel in the map, and a whole floor is hundreds of thousands of
    /// them. Twice a second is faster than a person can walk into new geometry
    /// and cheap enough that the number never competes with the writer.
    public static let coverageFractionRecomputeHz: Double = 2

    // MARK: - Point cloud (`sparse/0/points3D.txt`)

    /// Voxel size the LiDAR cloud is downsampled at before writing.
    /// docs/DATA_FORMAT.md section 4 fixes this at 1 cm.
    public static let pointCloudVoxelSizeMeters: Float = 0.01

    /// Hard cap on accumulated points. Past this the accumulator doubles its
    /// voxel size and re-bins, which loses detail but never runs the phone out
    /// of memory mid-walk. The degradation is recorded and reported.
    public static let pointCloudMaxPoints: Int = 2_000_000

    /// Stride through the native depth map when feeding the point cloud.
    public static let pointCloudDepthStride: Int = 2

    // MARK: - Depth sidecars

    /// docs/DATA_FORMAT.md section 5: `0` in a `.depth16` means no return.
    public static let depthNoReturn: UInt16 = 0

    /// Depth values above this are past the sensor's honest range and are
    /// written as a no-return rather than as a confident long measurement.
    /// CONTRACTS.md section 7: LiDAR useful range ~5 m.
    public static let lidarMaxRangeMeters: Float = 5.0

    /// JPEG quality for `images/frame_*.jpg`. docs/DATA_FORMAT.md section 5.
    public static let jpegQuality: CGFloat = 0.92

    // MARK: - Window mode (F9)

    /// Fraction of the centre of the screen that must be classified `window`
    /// (or `glass`) before window mode engages.
    public static let windowModeEnterFraction: Float = 0.18
    /// ...and the lower fraction it must fall below to disengage. The gap is
    /// hysteresis: a mode that flickers is worse than no mode.
    public static let windowModeExitFraction: Float = 0.08
    /// Minimum time in window mode, so a glance does not trigger a whole
    /// exposure-lock announcement.
    public static let windowModeMinDwellSeconds: Double = 1.5
    /// How far back from the glass the user is asked to stand.
    public static let windowModeStandBackMeters: Float = 1.5

    // MARK: - Guidance (F9)

    /// Minimum gap between two spoken hints. Faster than this and the app is
    /// nagging rather than guiding.
    public static let guidanceSpeechMinIntervalSeconds: Double = 4.0
    /// A hint must stay true for this long before it is worth saying out loud.
    public static let guidanceHintDebounceSeconds: Double = 1.2
    /// Gap between the repeating "too fast" ticks while blur is red.
    public static let guidanceBlurTickIntervalSeconds: Double = 0.45

    // MARK: - Storage and heat

    /// Capture stops cleanly when free disk falls below this. Stopping with a
    /// complete bundle beats being killed with a half-written one.
    public static let minFreeDiskBytes: Int64 = 500 * 1024 * 1024

    /// At or above this thermal level the keyframe interval is stretched.
    public static let thermalDegradeAt: ThermalLevel = .serious

    /// Multiplier applied to the minimum keyframe interval when hot.
    public static let thermalKeyframeIntervalMultiplier: Double = 2.0

    /// At or above this thermal level the capture stops itself and writes the
    /// bundle. `critical` is the level at which iOS starts shutting things
    /// down on its own; being killed there would leave a scan with no index,
    /// so the app stops first, finishes the file, and says why.
    public static let thermalStopAt: ThermalLevel = .critical

    /// How often free disk space is checked while recording, seconds. The
    /// check is a filesystem call, so it happens on its own slow schedule
    /// rather than on every HUD tick.
    public static let diskCheckIntervalSeconds: Double = 2.0

    // MARK: - Camera-to-IMU time offset (F1)
    //
    // CONTRACTS.md section 7 fixes the sweep at -50...+50 ms in 5 ms steps.
    // Capture measures it live by correlating how fast the camera pose says
    // the phone turned against how fast the gyro says it turned; the pre-pass
    // re-derives it from reprojection error and may overwrite the answer.

    /// Widest offset considered, seconds, in each direction.
    public static let timeOffsetSweepMaxSeconds: Double = 0.050
    /// Step between candidate offsets, seconds.
    public static let timeOffsetSweepStepSeconds: Double = 0.005

    /// How long a camera sample waits before it is correlated, seconds.
    ///
    /// The sweep looks the gyro up at `frameTime + offset`, and the largest
    /// offset is in the future when the frame arrives. Holding each sample for
    /// twice the sweep width means every lookup lands on real samples rather
    /// than on the ring buffer's clamped end.
    public static let timeOffsetGyroSettleSeconds: Double = 0.100

    /// Fewest correlated samples before an answer is offered at all. At the
    /// evaluation rate this is roughly eight seconds of walking.
    public static let timeOffsetMinSamples: Int = 120

    /// Correlation the winning offset has to reach. Below this the two signals
    /// do not describe the same motion and the honest answer is "not measured".
    public static let timeOffsetMinCorrelation: Float = 0.5

    /// The camera's turn rate has to vary at least this much, rad/s, for the
    /// correlation to mean anything. A phone carried perfectly level down a
    /// corridor gives a flat signal that correlates with everything.
    public static let timeOffsetMinCameraSpeedStdRadPerSec: Float = 0.10

    // MARK: - Revisit detection at capture time (F1)
    //
    // Capture finds revisit CANDIDATES only: two keyframes that sit close
    // together, look the same way, and are far apart in time. It runs no depth
    // alignment, so every pair it emits is `RevisitMethod.poseProximity` and
    // its residuals are unmeasured. The pre-pass's ICP is what turns a
    // candidate into a measurement.

    /// How close two camera positions have to be to be candidates, metres.
    public static let revisitMaxCameraDistanceMeters: Float = 0.6
    /// ...and how closely their view directions have to agree, degrees.
    public static let revisitMaxViewAngleDegrees: Float = 30
    /// Minimum time between the two frames of a pair, seconds. Anything
    /// shorter is the same continuous look at the same wall, not a return to
    /// it, and it constrains nothing the raw track does not already say.
    public static let revisitMinTimeGapSeconds: Double = 20
    /// Minimum spacing between two emitted pairs, seconds, so a slow walk past
    /// a doorway produces a handful of constraints and not four hundred.
    public static let revisitMinPairSpacingSeconds: Double = 1.0
    /// Hard cap on emitted pairs, so a long walk around a loop cannot make the
    /// bundle enormous.
    public static let revisitMaxPairs: Int = 2_000

    // MARK: - Live HUD

    /// How often `CaptureLiveState` is published. 10 Hz is smooth to a human
    /// eye and cheap enough that the HUD never competes with the writer.
    public static let liveStateHz: Double = 10

    // MARK: - Derived helpers

    /// The blur meter (F9): pixels of smear for a given turn rate and shutter.
    ///
    ///     pixels = (radiansPerSecond in degrees) * exposureSeconds / 0.0426
    public static func motionBlurPixels(
        angularSpeedRadPerSec: Float,
        exposureDurationSeconds: Double
    ) -> Float {
        let degreesPerSecond = angularSpeedRadPerSec * 180 / .pi
        return degreesPerSecond * Float(exposureDurationSeconds)
            / angularPixelPitchDegrees
    }

    /// Expected metric error of one LiDAR return, metres. Written into the
    /// `ERROR` column of `points3D.txt`, where docs/DATA_FORMAT.md section 4
    /// documents the deliberate reinterpretation (metres, not reprojection
    /// pixels).
    ///
    /// The physics: error grows with the square of range (the return weakens
    /// and the beam footprint widens) and with grazing incidence (the footprint
    /// smears along the surface). The constants are a conservative fit to
    /// Apple's published dToF behaviour rather than a measurement of this
    /// specific unit, which is why the pre-pass's trust field (F6) exists to
    /// replace this estimate with an observed one.
    public static func expectedDepthErrorMeters(
        rangeMeters: Float,
        incidenceCosine: Float,
        confidence: UInt8
    ) -> Float {
        let base: Float = 0.006
        let quadratic: Float = 0.004 * rangeMeters * rangeMeters
        // cos 0 is head-on. Clamped at ~84 degrees so the term stays finite.
        let cosine = Swift.max(0.10, Swift.min(1.0, incidenceCosine))
        let grazing: Float = 0.010 * (1.0 / cosine - 1.0)
        let confidencePenalty: Float
        switch confidence {
        case 2: confidencePenalty = 1.0
        case 1: confidencePenalty = 1.5
        default: confidencePenalty = 2.5
        }
        return (base + quadratic + grazing) * confidencePenalty
    }
}
