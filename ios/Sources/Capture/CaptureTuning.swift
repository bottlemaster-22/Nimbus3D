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
    ///
    /// WHY 0.30 AND NOT 1/6, WHICH IS WHAT THIS WAS. Start with what this
    /// number is, because it is easy to read it as a rate and it is not one:
    /// it is a CEILING. A frame is only considered on the 15 Hz evaluation
    /// grid above, so the effective floor is the first grid step at or past
    /// this value. 1/6 landed on the third step, 0.200 s, so the old ceiling
    /// was 5.0 keyframes a second. 0.30 lands on the fifth, 0.333 s, so the
    /// new ceiling is 3.0 a second.
    ///
    /// WHAT THE PHONE ACTUALLY DID. Measured on the owner's own device and
    /// not derivable from anything in this repository, so it is recorded here
    /// as the outside measurement it is: one scan of 4 minutes 36 seconds
    /// wrote 868 keyframes and about 651 MB. That is 3.1 keyframes a second
    /// against a ceiling of 5.0, which says plainly that this constant was
    /// not what set the rate. Motion, the QC floor and the
    /// `isKeyframeInFlight` gate set it; the ceiling only clipped the bursts.
    /// The per-keyframe split IS checkable from source:
    /// docs/DATA_FORMAT.md section 5 fixes 98,304 bytes of `.depth16` and
    /// 49,152 of `.conf8`, so of the 750 KB each keyframe cost, about 603 KB
    /// was the JPEG. Roughly 80% of a scan is the photograph, and the same
    /// section fixes that JPEG at quality 0.92, which makes it a contract
    /// value. The count is the only lever this file has.
    ///
    /// WHAT 0.30 BUYS, WHICH IS LESS THAN IT LOOKS. Mean spacing on that scan
    /// was 276 / 868 = 0.318 s, already just under the 0.333 the new ceiling
    /// imposes. The same walk is therefore capped at 276 / 0.333 = 828
    /// keyframes, a guaranteed cut of only about 5%. The rest of the cut is
    /// whatever share of the walk was packed against the old 0.200 floor, and
    /// a 3.1/s average says that was a minority of it, so expect something
    /// near a tenth: 651 MB down to roughly 580 MB, with the pre-pass's
    /// frame-major outputs falling in step (`trust_noise.bin` and
    /// `confidence_recal.bin` at 4 bytes per native sample each, plus one
    /// `.edge8` at 1 byte, is 442,368 bytes a frame, so about 384 MB becomes
    /// about 366 MB).
    ///
    /// THIS DOES NOT FIX THE WRITE BUDGET and must not be read as having done
    /// so. `Smart/TwoScaleTrustField` records what that budget looks like from
    /// the phone's side: 1 GB of file writes in a 32-minute session and 4 GB
    /// across one day, both tiers of the daily allowance breached. 651 MB in
    /// 276 s is 2.4 MB/s sustained, which is orders of magnitude above what
    /// iOS treats as sustainable, and a tenth off it is still orders of
    /// magnitude above it. The only two levers that would move that are the
    /// JPEG, which is a contract value, and an interval long enough to starve
    /// the carver below. This change is worth making because it is nearly
    /// free, not because it is the answer.
    ///
    /// THE COUNT NOBODY DOWNSTREAM ASKED FOR is the better argument for it.
    /// Every consumer of these frames states its own appetite and 868 is far
    /// above all of them. `TrainingBudget.recommended` wants 120 supervision
    /// views for a room and 240 for a house.
    /// `PrePassInitialSplatBuilder.Settings.maxKeyframes` is 160,
    /// `PrePassSurveyor` 48 and `PrePassGlassDetector` 40, and all three
    /// stride over the frames rather than reading more of them, so extra
    /// frames only widen the stride. `PrePassBundleAdjuster` selects on 0.30 m
    /// OR 0.75 s and then thins to 80, and the time term on its own selects
    /// more than 80 on any scan past about a minute, so it thins either way.
    /// `HeldOutFrameSelector.stride` is 20 whatever the total.
    ///
    /// THE ONE STAGE THAT WALKS EVERY FRAME AND WANTS THEM DENSE is the
    /// carver, and it declares TWO appetites, the second of which argues
    /// against this change and so has to be written down rather than left out.
    /// `VoxelFreeSpaceCarver.Tuning` has `keyframeSpacingMeters` at 0.10 AND
    /// `keyframeSpacingSeconds` at 0.20, and `keyframes(from:)` keeps a frame
    /// when EITHER is satisfied. A stated appetite of five frames a second is
    /// exactly the rate being capped. Two things answer it. First, that
    /// thinner already never fires: at the measured 0.318 s mean spacing every
    /// captured frame is already more than 0.20 s from the one before, so the
    /// carver already carves from every frame the capture wrote and will still
    /// do so at 0.333. Second, the time term exists to thin a camera that is
    /// NOT moving, so a phone held still does not carve the same air thirty
    /// times over; the frames it would have added are frames from a stationary
    /// camera, carving air that is already carved. The SPATIAL term is the one
    /// that binds, and against it three a second means anyone scanning at
    /// 0.33 m/s or slower leaves consecutive keyframes 10 cm apart or closer.
    /// Faster than that the spacing opens up, but it did at five a second too:
    /// the crossover moves from 0.5 m/s to 0.33 m/s, it does not appear.
    ///
    /// WHAT IT DOES NOT TOUCH, which matters because earlier scans came out
    /// too sparse and that must not be repeated. All four causes of that were
    /// found and fixed and not one of them was the frame count: densification
    /// never fired on a units mismatch, the thermal cap deleted live geometry,
    /// the prune schedule was dead code, and the trust gate rejected
    /// essentially every sample. `frameEvaluationHz` is untouched, so the
    /// coverage field still updates on its own 8 Hz branch, the camera-to-IMU
    /// calibrator still collects its `timeOffsetMinSamples` of 120 in about
    /// eight seconds because `observe` is called ABOVE the in-flight gate, and
    /// the blur meter, the HUD and the spoken guidance all run exactly as
    /// before, because every one of them sits on the evaluation grid and not
    /// on this one. `keyframeMaxIntervalSeconds` is untouched too, so someone
    /// standing still still leaves no hole in the temporal track. Coverage in
    /// fact gets slightly MORE reliable, because its branch sits below the
    /// `isKeyframeInFlight` gate and a writer asked for less work blocks it
    /// less often.
    ///
    /// WHAT IT DOES TOUCH. Three things move, and the measurement shrinks all
    /// three to almost nothing, which is worth recording because sizing them
    /// off the 5/s ceiling instead of the 3.1/s the phone really produced
    /// makes every one of them sound serious. FIRST, short scans lose
    /// supervision: `ProcessingBudgetPlanner.plan` clamps `keyframeCount` to
    /// `min(budget, frameCount)` and every adjustment in that function is a
    /// `min`, so reaching the full 120 views for a room takes about 40 seconds
    /// of scanning rather than the 38 it took at 3.1/s. SECOND, the LiDAR
    /// cloud loses that same tenth of its input, because `pointCloud.add` runs
    /// only for accepted keyframes; its EXTENT does not shrink, since voxel
    /// occupancy follows coverage rather than frame count, but each voxel
    /// averages slightly fewer measurements. THIRD, brackets are counted in
    /// KEYFRAMES, so `bracketEveryNKeyframes` of 12 arrives about every 4.0
    /// seconds instead of about every 3.8, and window mode's 5 about every 1.7
    /// instead of about every 1.6. One thing moves the other way for free:
    /// `SmartLossSettings.houseSizedFrameCount` is 1200, above which the trust
    /// build downshifts to the economy preset and verifies depth with two
    /// partner frames instead of four and no plane sweep at all, and that
    /// cliff moves from about 6.4 minutes of scanning to about 6.7.
    ///
    /// WHY 0.30 RATHER THAN 1.0 / 3.0, and how firm any of this is. The
    /// evaluation grid is not a clean lattice: `shouldEvaluate` compares
    /// against the last ACCEPTED evaluation timestamp, and at 60 fps four
    /// inter-frame gaps sum to exactly 1/15, so ARKit's timestamp jitter
    /// decides each time whether the step is 0.067 or 0.083. Landing points at
    /// or past 0.30 are therefore around 0.333 to 0.35, about 2.9 to 3.0 a
    /// second, and by the same argument the old ceiling was somewhere in four
    /// to five a second rather than a firm five. 1.0 / 3.0 would sit exactly
    /// on the 0.333 boundary and inherit that coin flip a second time, in this
    /// comparison as well as in the grid, for nothing. 0.30 sits clear of it.
    /// Hot, `thermalKeyframeIntervalMultiplier` doubles this to 0.60, which
    /// lands on the tenth step at 0.667 for about 1.5 a second; the same
    /// looseness applies there and neither is worth a second constant.
    public static let keyframeMinIntervalSeconds: Double = 0.30

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

    /// Smear in pixels at which the HUD goes amber.
    ///
    /// WHY 4 AND NOT 2, which is what the first draft used. These numbers are
    /// CALIBRATION, not contract: the formula and the 0.0426 divisor above are
    /// what CONTRACTS.md fixes, and `FrameQC.motionBlurPixels` still records
    /// the raw figure, so nothing on disk changes when these move.
    ///
    /// They are quoted in CAPTURE pixels, against a stream this app asks to be
    /// about 1920 px wide (`ARCaptureService.preferredVideoFormat`). The
    /// trainer never sees that. `TrainerBudget.resolutionLadder` is
    /// [720, 600, 480, 384] and `TrainerBudget.lower` is a one-way valve, so
    /// 720 px on the long edge is the most supervision ever gets, and every
    /// number here is divided by 1920/720 = 2.67 before it can touch a
    /// Gaussian. The old amber of 2 px was three quarters of one pixel in the
    /// image being fitted, which is below what that grid can even represent,
    /// and the old red of 4 px interrupted the user at 1.5 trainer pixels.
    ///
    /// So: amber at 1.5 trainer px (4.0 here) is where smear starts costing
    /// real detail, and red at 3 trainer px (8.0 here) is where a trainer with
    /// no blur model starts explaining the smear with geometry instead. The
    /// 3 px ceiling is judgement, not a measurement, and it is written down as
    /// judgement so it can be argued with.
    public static let blurAmberPixels: Float = 4.0

    /// Smear in pixels at which the HUD goes red and guidance interrupts.
    /// See `blurAmberPixels` for where this number comes from.
    public static let blurRedPixels: Float = 8.0

    /// Smear at which the blur term of the QC weight reaches its floor.
    ///
    /// Named separately, and deliberately NOT derived from `blurRedPixels`, so
    /// that moving a HUD threshold can never silently drag the weight curve
    /// with it. 12 capture px is 4.5 px in the trainer's 720 px supervision
    /// image: past that there is genuinely nothing left to supervise against.
    public static let blurWeightFloorPixels: Float = 12.0

    /// Longest shutter the capture camera may choose, seconds.
    ///
    /// Nothing capped this before, and in a dim room the auto-exposure
    /// algorithm happily went to 1/15 s or longer. Run that through the
    /// blur meter: red is 8 px, so the turn rate that trips it is
    /// 8 * 0.0426 / shutter degrees per second.
    ///
    ///     1/15 s  ->  red at  5.1 deg/s
    ///     1/30 s  ->  red at 10.2 deg/s
    ///     1/60 s  ->  red at 20.4 deg/s
    ///
    /// Ordinary hand tremor alone reaches something like 10 deg/s at its
    /// peaks. At 1/15 s the app was therefore telling a person holding the
    /// phone perfectly still that they were moving too fast, which is what
    /// the owner ran into: "it feels like I have to use a gimbal".
    ///
    /// Capping the shutter makes the camera raise ISO instead of dragging
    /// the shutter, and that is the right trade for THIS app. Sensor noise
    /// is close to independent between frames, so the trainer averages it
    /// away across the many photographs that see the same surface. Motion
    /// blur is not: it destroys the high-frequency detail in every frame
    /// that has it, and no amount of averaging brings it back.
    public static let maxExposureDurationSeconds: Double = 1.0 / 60.0

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
    /// Gap between the repeating smear ticks while blur is red.
    public static let guidanceBlurTickIntervalSeconds: Double = 0.9

    /// The same sentence is not said again inside this window, even while the
    /// thing that caused it is still true.
    ///
    /// Without this, `setHint` re-speaks an unchanged sentence every
    /// `guidanceSpeechMinIntervalSeconds`, so a scan that has not finished
    /// covering the room (which is most of a scan) hears the same coverage
    /// sentence fifteen times a minute from the first second to the last.
    public static let guidanceHintRepeatSeconds: Double = 25.0

    /// Smear at which an active smear warning CLEARS.
    ///
    /// The gap below `blurRedPixels` is a dead band. Blur is sampled once per
    /// frame with no smoothing, so a reading sitting on the line would
    /// otherwise flip the warning on and off at the HUD's 10 Hz, which for
    /// someone whose hands shake is a strobe rather than a warning.
    public static let blurRedClearPixels: Float = 6.0

    /// How long smear has to stay red before guidance says anything at all.
    /// A single spike is not worth interrupting for.
    public static let guidanceBlurEnterSeconds: Double = 0.8

    /// How long the smear ticks keep going before they give up. If someone has
    /// not slowed down in three seconds, a fourth second of ticking is not
    /// information, it is nagging.
    public static let guidanceBlurMaxTickSeconds: Double = 3.0

    /// Quiet after the smear ticks give up, before they may start again. The
    /// meter stays on screen throughout, so nothing is hidden by the silence.
    public static let guidanceBlurCooldownSeconds: Double = 20.0

    /// Minimum quiet between two falling chimes, so a tracking state that
    /// flickers around the edge of `.limited` cannot ring repeatedly.
    public static let guidanceChimeMinIntervalSeconds: Double = 6.0

    /// Shutter at or above which the ROOM, not the hand, is the larger half of
    /// the smear product, and the copy should say so.
    ///
    /// Smear is turn rate multiplied by how long the shutter was open. Telling
    /// someone to move more slowly when the real problem is that the light is
    /// low is both useless and untrue, and to a person with a tremor it reads
    /// as being told they are doing it wrong.
    public static let dimShutterSeconds: Double = 1.0 / 40.0

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

    /// The same meter, fed the rotation the camera ACTUALLY turned through
    /// while the shutter was open rather than a rate sampled at one instant
    /// and assumed to hold for the whole window.
    ///
    /// `degreesPerSecond * exposureDuration` in the function above is just
    /// an estimate of this angle that is exact only when the rate is
    /// constant. See `CaptureGyroSampler.netRotationRadians` for why that
    /// assumption fails badly for a shaking hand. The divisor, and so the
    /// meaning of the number and every threshold quoted against it, is
    /// unchanged.
    public static func motionBlurPixels(netRotationRadians: Float) -> Float {
        let degrees = netRotationRadians * 180 / .pi
        return degrees / angularPixelPitchDegrees
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
