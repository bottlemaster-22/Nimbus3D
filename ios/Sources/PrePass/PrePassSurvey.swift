//
//  PrePassSurvey.swift
//  PrePass
//
//  THE CHEAP PASS OVER POSES AND LiDAR THAT THE QC CARD IS BUILT FROM (F9).
//
//  Everything here reads two things and nothing else: where the phone was, and
//  what the laser came back with. No images are decoded, no carving is run, no
//  trust field is built. That is what lets `quickQCCard` put a real, honest
//  answer on screen in under three seconds instead of asking the user to wait
//  out a full pre-pass to find out their scan was taken too fast.
//
//  The numbers are MEASURED, not scored. "Coverage 64%" here means "64% of the
//  10 cm surface patches this scan actually touched met the coverage test
//  below", and the test is written out in full so nobody has to guess what the
//  number promises.
//
//  ---------------------------------------------------------------------------
//  WHERE THE THRESHOLDS COME FROM
//  ---------------------------------------------------------------------------
//  Every tolerance below is derived from the sensor, not picked because it is
//  a round number.
//
//  ANGULAR SPREAD, 15 degrees. A surface only pins down in depth if it was
//  looked at from meaningfully different directions. The wide camera's angular
//  pixel pitch is 0.0426 deg/px at 1920 px (the same constant the live blur
//  meter uses, see `FrameQC.motionBlurPixels`). Localising a point to about
//  2 px therefore costs ~0.085 degrees of angular error. The LiDAR's own
//  1-sigma at 2 m is ~1 cm (`SmartDepthNoiseModel`: 0.004 + 0.0015 * z^2),
//  which is 0.5% of the range. For a triangulated depth to be as good as the
//  laser's, the baseline angle has to satisfy
//
//      spread >= angular localisation error / relative depth target
//             =  0.085 deg / 0.005  ~= 17 degrees
//
//  15 degrees is that figure rounded to where the useful signal starts, and it
//  is the point below which a second view adds almost nothing the first view
//  did not already have.
//
//  USABLE RANGE BAND, 0.25 m to 90% of the sensor's stated reach. The lower
//  end is where the depth camera stops resolving at all; the upper end is not
//  a cliff, it is where `SmartDepthNoiseModel`'s quadratic range term has
//  grown the 1-sigma error to ~4 cm, which is already comparable to the 5 cm
//  occupancy voxel. Beyond that a "covered" claim would be about the sensor's
//  noise rather than about the surface.
//
//  THREE HITS AND TWO SEPARATE VISITS. One grazing sample lands on a lot of
//  surfaces it did not really resolve. Two visits rather than two frames
//  because 400 samples from one walk past a wall is one observation repeated,
//  which is the same reasoning the trust field's distinct-times count exists
//  for (F6).
//

import Foundation
import simd

// MARK: - The measured result

/// What one survey pass measured. Plain numbers only: the sentences the user
/// reads are built from these by `PrePassQCBuilder`, so the measurement and
/// the wording stay separable and each can be checked on its own.
struct PrePassSurvey {
    /// Distinct occupied surface patches the scan touched, at
    /// `voxelSizeMeters`.
    var surfaceCellCount: Int = 0
    /// Of those, how many met every part of the coverage test.
    var coveredCellCount: Int = 0
    var coverageFraction: Float = 0

    var ceilingCellCount: Int = 0
    var ceilingCoveredCellCount: Int = 0
    var ceilingCoverageFraction: Float = 0

    /// Median over surface patches of the angle between the first direction a
    /// patch was seen from and the most different direction after it. A lower
    /// bound on the true spread, never an overstatement.
    var medianAngularSpreadDegrees: Float = 0
    /// Median over surface patches of the mean distance the phone stood at.
    var medianCameraToSurfaceMeters: Float = 0
    /// 95th minus 5th percentile of camera height, metres. Percentiles rather
    /// than min and max so one moment of crouching does not read as a varied
    /// scan.
    var cameraHeightSpreadMeters: Float = 0

    /// Fraction of sampled depth pixels that came back with nothing while
    /// their neighbourhood came back fine: the depth-only signature of an
    /// aperture, which is glass, a doorway, or a mirror. The full pre-pass
    /// replaces this with the fitted `[GlassRegion]` measurement; on the fast
    /// path it is the only glass evidence available without decoding images.
    var apertureSampleFraction: Float = 0

    /// Frames whose step distance or time gap was an outlier. A frame flagged
    /// by both tests is counted once.
    var outlierFrameCount: Int = 0

    /// World extent of the LiDAR returns. `nil` when nothing came back at all.
    var surfaceBounds: BoundingBox?

    var keyframesUsed: Int = 0
    /// Frames whose depth sidecar was present but could not be read. Reported,
    /// never swallowed.
    var unreadableDepthFrames: Int = 0
    var voxelSizeMeters: Float = 0.10
    /// True when the survey found no LiDAR returns at all, so every coverage
    /// number above is zero because there was nothing to measure rather than
    /// because the scan was bad.
    var hadNoDepth: Bool = true
}

// MARK: - The survey pass

enum PrePassSurveyor {

    struct Settings {
        /// Coverage granularity. 10 cm is about the smallest patch it is
        /// meaningful to call "covered" for a splat field: it is twice the
        /// occupancy voxel and roughly the native sample spacing at 4 m.
        var voxelSizeMeters: Float = 0.10
        /// Frames actually opened. 48 depth sidecars is ~4.7 MB of IO, which
        /// is what keeps the fast path inside its three-second promise.
        var maxKeyframes = 48
        /// Use every Nth native sample in each direction: 4 gives ~3000 rays
        /// per frame out of 49152.
        var raySubsampleStride = 4
        var minObservations: UInt16 = 3
        var minDistinctKeyframes: UInt16 = 2
        var minAngularSpreadDegrees: Float = 15
        /// See the header: below this the depth camera is not resolving.
        var minUsableRangeMeters: Float = 0.25
        /// Fraction of the stated LiDAR reach still treated as usable.
        var usableRangeFraction: Float = 0.90
        /// Half-width of the neighbourhood an aperture test looks at.
        var apertureNeighbourhoodRadius = 3
        /// A silent sample counts as an aperture only when at least this much
        /// of its neighbourhood did come back. Half, because a pane inside a
        /// wall has returning frame on every side of it, while a ray into open
        /// sky has silence all around.
        var apertureNeighbourReturnFraction: Float = 0.5

        init() {}
    }

    /// Runs the pass.
    ///
    /// - Parameter poseFor: where the phone was for a given frame. The caller
    ///   passes raw VIO poses on the fast path and refined poses afterwards,
    ///   which is the whole reason this is a closure: the same measurement is
    ///   worth much more once the poses are fixed.
    static func survey(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        poseFor: (CaptureFrame) -> Pose,
        settings: Settings = Settings()
    ) throws -> PrePassSurvey {
        var result = PrePassSurvey()
        result.voxelSizeMeters = settings.voxelSizeMeters

        let frames = bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard !frames.isEmpty else { return result }

        // --- Camera height spread and motion outliers: poses only, every frame.
        var heights: [Float] = []
        heights.reserveCapacity(frames.count)
        for frame in frames { heights.append(poseFor(frame).center.y) }
        result.cameraHeightSpreadMeters = Swift.max(
            PrePassStats.percentile(heights, 0.95) - PrePassStats.percentile(heights, 0.05),
            0
        )
        result.outlierFrameCount = outlierFrames(frames: frames, poseFor: poseFor).count

        // --- Keyframes: evenly spread over the walk, not the first N.
        let usable = frames.filter {
            $0.depthPath != nil && $0.qc.trackingQuality != .notAvailable
        }
        guard !usable.isEmpty else { return result }
        let step = Swift.max(1, usable.count / Swift.max(settings.maxKeyframes, 1))
        var keyframes: [CaptureFrame] = []
        var i = 0
        while i < usable.count {
            keyframes.append(usable[i])
            i += step
        }
        result.keyframesUsed = keyframes.count

        let geometry = PrePassDepthGeometry(
            rgbIntrinsics: bundle.intrinsics, settings: bundle.settings
        )
        let maxRange = Swift.max(bundle.settings.lidarMaxRangeMeters, 0.5)
        let usableRange = maxRange * settings.usableRangeFraction

        // The grid needs an origin before the first point exists, so it is the
        // camera path grown by the sensor's own reach: the measured envelope of
        // what could possibly have been seen.
        var pathMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        for frame in frames {
            pathMin = simd_min(pathMin, poseFor(frame).center.simd)
        }
        let reach = SIMD3<Float>(repeating: maxRange)
        let voxelFrame = PrePassVoxelFrame(
            origin: pathMin - reach - SIMD3<Float>(repeating: 1),
            voxelSize: settings.voxelSizeMeters
        )

        var hash = PrePassVoxelHash(expectedCount: 1 << 14)
        var firstDirection: [SIMD3<Float>] = []
        var spreadDegrees: [Float] = []
        var hitCount: [UInt16] = []
        var distinctKeyframes: [UInt16] = []
        var lastKeyframe: [Int32] = []
        var rangeSum: [Float] = []
        var upwardSum: [Float] = []

        var surfaceMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var surfaceMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var sampledReturns = 0
        var apertureSamples = 0

        let stride = Swift.max(settings.raySubsampleStride, 1)
        let width = geometry.width
        let height = geometry.height

        for (keyframeIndex, frame) in keyframes.enumerated() {
            try Task.checkCancellation()

            let depthFrame: PrePassDepthFrame?
            do {
                depthFrame = try PrePassDepthFrame.load(
                    frame: frame, settings: bundle.settings, at: ref
                )
            } catch {
                // Corrupt sidecar. Counted and reported, never guessed at.
                result.unreadableDepthFrames += 1
                continue
            }
            guard let depthFrame else { continue }

            let pose = poseFor(frame)
            let origin = pose.center.simd
            let rotationInverse = pose.rotation.simd.inverse

            var y = 0
            while y < height {
                var x = 0
                while x < width {
                    let index = y * width + x
                    if depthFrame.hasReturn(at: index) {
                        let z = depthFrame.depthMeters(at: index)
                        let range = geometry.range(index: index, depthMeters: z)
                        if range >= settings.minUsableRangeMeters, range <= maxRange {
                            sampledReturns += 1
                            let direction = rotationInverse.act(geometry.rayDirections[index])
                            let world = origin + direction * range
                            surfaceMin = simd_min(surfaceMin, world)
                            surfaceMax = simd_max(surfaceMax, world)

                            if let key = voxelFrame.key(world) {
                                let slot = hash.indexOrInsert(key)
                                if slot.inserted {
                                    firstDirection.append(direction)
                                    spreadDegrees.append(0)
                                    hitCount.append(1)
                                    distinctKeyframes.append(1)
                                    lastKeyframe.append(Int32(keyframeIndex))
                                    rangeSum.append(range)
                                    upwardSum.append(direction.y)
                                } else {
                                    let s = slot.index
                                    if hitCount[s] < UInt16.max { hitCount[s] += 1 }
                                    rangeSum[s] += range
                                    upwardSum[s] += direction.y
                                    if lastKeyframe[s] != Int32(keyframeIndex) {
                                        lastKeyframe[s] = Int32(keyframeIndex)
                                        if distinctKeyframes[s] < UInt16.max {
                                            distinctKeyframes[s] += 1
                                        }
                                    }
                                    let angle = PrePassAngle.degreesBetween(
                                        firstDirection[s], direction
                                    )
                                    if angle > spreadDegrees[s] { spreadDegrees[s] = angle }
                                }
                            }
                        }
                    } else {
                        if isAperture(
                            depthFrame: depthFrame, geometry: geometry,
                            x: x, y: y, maxRange: maxRange, settings: settings
                        ) {
                            apertureSamples += 1
                        }
                    }
                    x += stride
                }
                y += stride
            }
        }

        guard hash.count > 0 else { return result }
        result.hadNoDepth = false
        result.surfaceBounds = BoundingBox(min: Vector3(surfaceMin), max: Vector3(surfaceMax))

        let denominator = sampledReturns + apertureSamples
        result.apertureSampleFraction = denominator > 0
            ? Float(apertureSamples) / Float(denominator)
            : 0

        // Where the ceiling starts. ARKit's own ceiling anchors when it found
        // any, because a measured plane beats a rule of thumb; otherwise a
        // little above where the phone was actually held.
        let ceilingHeight = ceilingThreshold(bundle: bundle, cameraHeights: heights)

        var spreads: [Float] = []
        var ranges: [Float] = []
        spreads.reserveCapacity(hash.count)
        ranges.reserveCapacity(hash.count)

        for entry in hash.entries() {
            let s = entry.slot
            let count = Float(hitCount[s])
            let meanRange = count > 0 ? rangeSum[s] / count : 0
            let meanUpward = count > 0 ? upwardSum[s] / count : 0
            spreads.append(spreadDegrees[s])
            ranges.append(meanRange)

            let covered = hitCount[s] >= settings.minObservations
                && distinctKeyframes[s] >= settings.minDistinctKeyframes
                && spreadDegrees[s] >= settings.minAngularSpreadDegrees
                && meanRange >= settings.minUsableRangeMeters
                && meanRange <= usableRange
            if covered { result.coveredCellCount += 1 }

            let centre = voxelFrame.centre(ofKey: entry.key)
            // A ceiling patch is one that is above head height AND was looked
            // at from below. Height alone would count the top of a bookcase.
            if centre.y >= ceilingHeight && meanUpward > 0.5 {
                result.ceilingCellCount += 1
                if covered { result.ceilingCoveredCellCount += 1 }
            }
        }

        result.surfaceCellCount = hash.count
        result.coverageFraction = Float(result.coveredCellCount) / Float(hash.count)
        result.ceilingCoverageFraction = result.ceilingCellCount > 0
            ? Float(result.ceilingCoveredCellCount) / Float(result.ceilingCellCount)
            : 0
        result.medianAngularSpreadDegrees = PrePassStats.median(spreads)
        result.medianCameraToSurfaceMeters = PrePassStats.median(ranges)

        return result
    }

    // MARK: Internals

    /// A silent sample surrounded by returning ones. That is an aperture in a
    /// surface, which is what a window, a doorway or a mirror looks like to a
    /// LiDAR; silence with silence all around it is just open space.
    private static func isAperture(
        depthFrame: PrePassDepthFrame,
        geometry: PrePassDepthGeometry,
        x: Int,
        y: Int,
        maxRange: Float,
        settings: Settings
    ) -> Bool {
        let radius = settings.apertureNeighbourhoodRadius
        guard radius > 0 else { return false }
        var neighbours = 0
        var returning = 0
        var dy = -radius
        while dy <= radius {
            let ny = y + dy
            if ny >= 0 && ny < geometry.height {
                var dx = -radius
                while dx <= radius {
                    let nx = x + dx
                    if nx >= 0 && nx < geometry.width, !(dx == 0 && dy == 0) {
                        neighbours += 1
                        let index = ny * geometry.width + nx
                        if depthFrame.hasReturn(at: index) {
                            let z = depthFrame.depthMeters(at: index)
                            if z > 0.05, geometry.range(index: index, depthMeters: z) <= maxRange {
                                returning += 1
                            }
                        }
                    }
                    dx += 1
                }
            }
            dy += 1
        }
        guard neighbours > 0 else { return false }
        return Float(returning) / Float(neighbours) >= settings.apertureNeighbourReturnFraction
    }

    /// Frames whose step distance or time gap is a robust outlier.
    ///
    /// A `Set` rather than two counters because a stumble shows up in both
    /// tests at once, and reporting "12 frames had problems" when six frames
    /// had two problems each is the kind of small dishonesty that makes a
    /// whole quality report untrustworthy.
    static func outlierFrames(
        frames: [CaptureFrame],
        poseFor: (CaptureFrame) -> Pose
    ) -> Set<FrameID> {
        var flagged = Set<FrameID>()
        guard frames.count >= 8 else { return flagged }

        var steps: [Float] = []
        var gaps: [Float] = []
        steps.reserveCapacity(frames.count - 1)
        gaps.reserveCapacity(frames.count - 1)
        for i in 1..<frames.count {
            let a = poseFor(frames[i - 1]).center.simd
            let b = poseFor(frames[i]).center.simd
            steps.append(simd_distance(a, b))
            gaps.append(Float(frames[i].timestampSeconds - frames[i - 1].timestampSeconds))
        }

        let stepMedian = PrePassStats.median(steps)
        let stepMAD = PrePassStats.medianAbsoluteDeviation(steps, median: stepMedian)
        let gapMedian = PrePassStats.median(gaps)
        let gapMAD = PrePassStats.medianAbsoluteDeviation(gaps, median: gapMedian)

        // 4 MAD-sigmas. MAD is scaled by 1.4826 in `PrePassStats`, so this is
        // a genuine 4-sigma test on data that has real outliers in it, which
        // is exactly the case a plain standard deviation gets wrong (the
        // outliers inflate the threshold that is meant to catch them).
        let stepLimit = stepMedian + 4 * Swift.max(stepMAD, 1e-4)
        let gapLimit = gapMedian + 4 * Swift.max(gapMAD, 1e-4)

        for i in 1..<frames.count {
            if steps[i - 1] > stepLimit || gaps[i - 1] > gapLimit {
                flagged.insert(frames[i].index)
            }
        }
        return flagged
    }

    /// Height above which a surface patch is treated as ceiling.
    private static func ceilingThreshold(
        bundle: CaptureBundle,
        cameraHeights: [Float]
    ) -> Float {
        var anchorCeiling = -Float.greatestFiniteMagnitude
        let anchors = bundle.anchorsAtEndOfSession.isEmpty
            ? bundle.anchorsDuringSession
            : bundle.anchorsAtEndOfSession
        for anchor in anchors where anchor.classification == .ceiling {
            anchorCeiling = Swift.max(anchorCeiling, simd_make_float3(anchor.matrix.columns.3).y)
        }
        if anchorCeiling > -Float.greatestFiniteMagnitude {
            // Just below the measured plane, so the patches ON it still count.
            return anchorCeiling - 0.15
        }
        // No ceiling anchor. Half a metre above the highest the phone was held
        // is the lowest a ceiling can plausibly be while the user still walked
        // under it.
        let highCamera = PrePassStats.percentile(cameraHeights, 0.90)
        return highCamera + 0.5
    }
}

// MARK: - The card

/// Turns measurements into the sentences the user actually reads.
///
/// Every string here is shown verbatim on the post-capture screen. They are
/// written for somebody who has never heard of a pose graph: they say what
/// happened, in what units a person can picture, and what to do about it.
enum PrePassQCBuilder {

    /// Where each finding changes severity. Gathered so the wording and the
    /// thresholds sit next to each other and cannot drift apart.
    struct Thresholds {
        /// Drift a splat field genuinely does not notice: under 2 cm over a
        /// room is inside the LiDAR's own 1-sigma at typical stand-off.
        var driftGoodCentimeters: Float = 2
        /// Above this the same wall seen twice lands more than a splat's width
        /// apart and the field starts hedging into a soft cloud.
        var driftProblemCentimeters: Float = 6
        var lowCoverageFraction: Float = 0.60
        var lowCeilingCoverageFraction: Float = 0.40
        var lowAngularSpreadDegrees: Float = 20
        var lowCameraHeightSpreadMeters: Float = 0.35
        var farMedianDistanceMeters: Float = 3.5
        var closeMedianDistanceMeters: Float = 0.40
        var notableGlassFraction: Float = 0.05
        var outlierFrameFraction: Float = 0.05

        init() {}
    }

    /// - Parameters:
    ///   - driftCentimeters: `nil` when nothing measured it, which produces an
    ///     explicit "we could not tell" finding instead of a fake zero.
    ///   - loopClosureCount: confirmed depth alignments, not candidates.
    ///   - isFastPath: the fast card says so, because its glass number comes
    ///     from depth alone and its drift comes from ARKit's anchors rather
    ///     than from a measured revisit.
    static func card(
        survey: PrePassSurvey,
        driftCentimeters: Float?,
        loopClosureCount: Int,
        glassAreaFraction: Float,
        totalFrameCount: Int,
        rejectedRevisitCandidates: Int,
        timeOffsetSeconds: Double?,
        isFastPath: Bool,
        thresholds: Thresholds = Thresholds()
    ) -> QCCard {
        var findings: [QCFinding] = []

        // --- Nothing to measure at all.
        if survey.hadNoDepth {
            findings.append(
                QCFinding(
                    code: "no_depth",
                    severity: .problem,
                    message: "No laser measurements came back from this scan, so there is "
                        + "nothing to check yet.",
                    fixHint: "Scan again in a room with normal lighting, holding the phone "
                        + "about a metre from what you are scanning."
                )
            )
        }

        // --- Drift.
        if let drift = driftCentimeters {
            let rounded = (drift * 10).rounded() / 10
            if drift <= thresholds.driftGoodCentimeters {
                findings.append(
                    QCFinding(
                        code: "drift",
                        severity: .good,
                        message: "The phone kept track of where it was to within about "
                            + "\(format(rounded)) cm across the whole walk.",
                        fixHint: nil
                    )
                )
            } else if drift <= thresholds.driftProblemCentimeters {
                findings.append(
                    QCFinding(
                        code: "drift",
                        severity: .warning,
                        message: "The phone's idea of where it was slid by about "
                            + "\(format(rounded)) cm over the walk. That is enough to soften "
                            + "fine detail.",
                        fixHint: "Walking a loop back to where you started, and pausing for a "
                            + "second on something you already scanned, lets the app pull it "
                            + "back into line."
                    )
                )
            } else {
                findings.append(
                    QCFinding(
                        code: "drift",
                        severity: .problem,
                        message: "The phone's idea of where it was slid by about "
                            + "\(format(rounded)) cm. Surfaces seen twice will not line up.",
                        fixHint: "Scan in smaller sections, and walk back over ground you have "
                            + "already covered before moving on to the next part."
                    )
                )
            }
        } else {
            findings.append(
                QCFinding(
                    code: "drift",
                    severity: .warning,
                    message: "There was no moment where you looked at the same spot twice, so "
                        + "there is no way to tell how much the phone's tracking slid.",
                    fixHint: "Next time, finish by walking back to where you started."
                )
            )
        }

        // --- Loop closures.
        if loopClosureCount == 0 && !survey.hadNoDepth {
            findings.append(
                QCFinding(
                    code: "loop_closures",
                    severity: .warning,
                    message: "You did not pass back over anywhere you had already scanned, so "
                        + "the app had nothing to check its own tracking against.",
                    fixHint: "Walk a loop rather than a line, and end where you began."
                )
            )
        } else if loopClosureCount > 0 {
            findings.append(
                QCFinding(
                    code: "loop_closures",
                    severity: .good,
                    message: loopClosureCount == 1
                        ? "You passed back over one spot you had already scanned, which let the "
                            + "app straighten out its tracking."
                        : "You passed back over \(loopClosureCount) spots you had already "
                            + "scanned, which let the app straighten out its tracking.",
                    fixHint: nil
                )
            )
        }

        if rejectedRevisitCandidates > 0 && loopClosureCount == 0 {
            findings.append(
                QCFinding(
                    code: "revisits_unconfirmed",
                    severity: .warning,
                    message: "You did walk back past \(rejectedRevisitCandidates) places you "
                        + "had scanned before, but the laser could not line the two visits up.",
                    fixHint: "That usually means moving too fast, or standing too far back. "
                        + "Slow down and get within about two metres of the surface."
                )
            )
        }

        // --- Coverage.
        if !survey.hadNoDepth {
            let percent = Int((survey.coverageFraction * 100).rounded())
            if survey.coverageFraction < thresholds.lowCoverageFraction {
                findings.append(
                    QCFinding(
                        code: "coverage",
                        severity: .warning,
                        message: "About \(percent) out of every 100 surfaces you touched were "
                            + "seen properly, meaning from more than one direction and from "
                            + "close enough to measure.",
                        fixHint: "Move around what you are scanning rather than turning on the "
                            + "spot, and keep within a couple of metres of it."
                    )
                )
            } else {
                findings.append(
                    QCFinding(
                        code: "coverage",
                        severity: .good,
                        message: "About \(percent) out of every 100 surfaces you touched were "
                            + "seen from enough directions to build them properly.",
                        fixHint: nil
                    )
                )
            }

            if survey.ceilingCellCount > 0,
               survey.ceilingCoverageFraction < thresholds.lowCeilingCoverageFraction {
                findings.append(
                    QCFinding(
                        code: "ceiling_coverage",
                        severity: .warning,
                        message: "The ceiling was barely looked at. It will come out soft or "
                            + "patchy.",
                        fixHint: "Tilt the phone up and walk a slow lap with it pointed at the "
                            + "ceiling."
                    )
                )
            }
        }

        // --- Viewing geometry.
        if !survey.hadNoDepth {
            if survey.medianAngularSpreadDegrees < thresholds.lowAngularSpreadDegrees {
                let degrees = Int(survey.medianAngularSpreadDegrees.rounded())
                findings.append(
                    QCFinding(
                        code: "angular_spread",
                        severity: .warning,
                        message: "Most surfaces were only seen from about \(degrees) degrees "
                            + "of different angles, which is close to seeing them from one "
                            + "side only.",
                        fixHint: "Walk around things instead of standing still and panning."
                    )
                )
            }

            if survey.cameraHeightSpreadMeters < thresholds.lowCameraHeightSpreadMeters {
                let centimetres = Int((survey.cameraHeightSpreadMeters * 100).rounded())
                findings.append(
                    QCFinding(
                        code: "camera_height",
                        severity: .warning,
                        message: "The phone stayed within about \(centimetres) cm of the same "
                            + "height the whole time, so the tops and undersides of things "
                            + "were never seen.",
                        fixHint: "Take a second pass holding the phone low, and another "
                            + "holding it high."
                    )
                )
            }

            if survey.medianCameraToSurfaceMeters > thresholds.farMedianDistanceMeters {
                let metres = (survey.medianCameraToSurfaceMeters * 10).rounded() / 10
                findings.append(
                    QCFinding(
                        code: "standoff_far",
                        severity: .warning,
                        message: "You were about \(format(metres)) m away from things on "
                            + "average. The laser gets noticeably less accurate past a few "
                            + "metres.",
                        fixHint: "Get closer, around one to two metres, especially for "
                            + "anything you want detail on."
                    )
                )
            } else if survey.medianCameraToSurfaceMeters > 0,
                      survey.medianCameraToSurfaceMeters < thresholds.closeMedianDistanceMeters {
                findings.append(
                    QCFinding(
                        code: "standoff_close",
                        severity: .warning,
                        message: "You were very close to everything, closer than the depth "
                            + "sensor reads reliably.",
                        fixHint: "Back off to about an arm and a half from the surface."
                    )
                )
            }
        }

        // --- Glass.
        if glassAreaFraction >= thresholds.notableGlassFraction {
            let percent = Int((glassAreaFraction * 100).rounded())
            findings.append(
                QCFinding(
                    code: "glass_area",
                    severity: .warning,
                    message: isFastPath
                        ? "Roughly \(percent) out of every 100 depth readings came back empty "
                            + "with solid surface all around them, which is what glass, a "
                            + "mirror or an open doorway looks like."
                        : "Roughly \(percent) out of every 100 surfaces in this scan are "
                            + "glass or an opening. The app will leave what is beyond them "
                            + "alone rather than inventing it.",
                    fixHint: "Nothing to fix. Scans with a lot of glass simply have less the "
                        + "app can be certain about."
                )
            )
        }

        // --- Shaky frames.
        if totalFrameCount > 0 {
            let fraction = Float(survey.outlierFrameCount) / Float(totalFrameCount)
            if fraction >= thresholds.outlierFrameFraction {
                findings.append(
                    QCFinding(
                        code: "motion_outliers",
                        severity: .warning,
                        message: "\(survey.outlierFrameCount) frames were taken during a sudden "
                            + "jump or a pause, out of \(totalFrameCount).",
                        fixHint: "Move at a slow, even walking pace and try not to swing the "
                            + "phone around between shots."
                    )
                )
            }
        }

        // --- Camera and motion sensor timing.
        if let offset = timeOffsetSeconds {
            // Absolute value: which of the two was ahead is a detail nobody
            // outside the optimiser needs, and a minus sign in a sentence like
            // this reads as an error rather than as a direction.
            let milliseconds = (abs(offset) * 1000 * 10).rounded() / 10
            findings.append(
                QCFinding(
                    code: "time_offset",
                    severity: .good,
                    message: "The camera and the motion sensors were "
                        + "\(format(Float(milliseconds))) thousandths of a second out of "
                        + "step, and that has been corrected.",
                    fixHint: nil
                )
            )
        } else if !isFastPath {
            findings.append(
                QCFinding(
                    code: "time_offset",
                    severity: .warning,
                    message: "The walk was too smooth to work out the exact timing between the "
                        + "camera and the motion sensors, so it was left as it was.",
                    fixHint: nil
                )
            )
        }

        // --- Unreadable files.
        if survey.unreadableDepthFrames > 0 {
            findings.append(
                QCFinding(
                    code: "unreadable_depth",
                    severity: .warning,
                    message: "\(survey.unreadableDepthFrames) frames had depth data that could "
                        + "not be read back, so they were left out of the checks.",
                    fixHint: "If this keeps happening, the phone may have been low on storage "
                        + "while recording."
                )
            )
        }

        return QCCard(
            driftCentimeters: driftCentimeters ?? 0,
            loopClosureCount: loopClosureCount,
            medianAngularSpreadDegrees: survey.medianAngularSpreadDegrees,
            medianCameraToSurfaceMeters: survey.medianCameraToSurfaceMeters,
            cameraHeightSpreadMeters: survey.cameraHeightSpreadMeters,
            outlierFrameCount: survey.outlierFrameCount,
            coverageFraction: survey.coverageFraction,
            ceilingCoverageFraction: survey.ceilingCoverageFraction,
            glassAreaFraction: glassAreaFraction,
            findings: findings
        )
    }

    /// One decimal place, and no trailing ".0" on a whole number, because
    /// "3 cm" reads better than "3.0 cm" to somebody who is not a programmer.
    private static func format(_ value: Float) -> String {
        guard value.isFinite else { return "0" }
        if value == value.rounded() { return String(Int(value.rounded())) }
        return String(format: "%.1f", Double(value))
    }
}

// MARK: - Drift from ARKit's own anchors

/// The free drift measurement (F8): ARKit silently moves its anchors when it
/// relocalises, so differencing the anchors logged during the session against
/// the same anchors re-read at the end says exactly how far the map slid, with
/// no computation and no assumptions.
///
/// This is what the fast card uses, because measuring drift properly needs ICP
/// on the depth maps and ICP does not fit in three seconds.
enum PrePassAnchorDrift {

    /// - Returns: median anchor movement in centimetres, or nil when the two
    ///   anchor sets do not overlap (so there is nothing to difference).
    static func medianCentimeters(bundle: CaptureBundle) -> Float? {
        guard !bundle.anchorsDuringSession.isEmpty,
              !bundle.anchorsAtEndOfSession.isEmpty
        else { return nil }

        var startByID: [UUID: SIMD3<Float>] = [:]
        startByID.reserveCapacity(bundle.anchorsDuringSession.count)
        for anchor in bundle.anchorsDuringSession {
            startByID[anchor.identifier] = simd_make_float3(anchor.matrix.columns.3)
        }

        var movements: [Float] = []
        for anchor in bundle.anchorsAtEndOfSession {
            guard let start = startByID[anchor.identifier] else { continue }
            let end = simd_make_float3(anchor.matrix.columns.3)
            let distance = simd_distance(start, end)
            if distance.isFinite { movements.append(distance * 100) }
        }
        guard movements.count >= 2 else { return nil }
        return PrePassStats.median(movements)
    }
}
