//
//  PrePassGlassDetector.swift
//  PrePass
//
//  F5: FINDING THE GLASS, THE WINDOWS AND THE OPEN DOORWAYS.
//
//  A pane of glass returns nothing to a LiDAR. Every generic pipeline reads
//  "nothing came back" as "nothing is there", carves the free space straight
//  through the window, and deletes the garden. The fix is not to be timid
//  everywhere: it is to find the panes specifically and mark them, so the
//  carver can leave what is beyond them UNKNOWN while still carving normally
//  everywhere else.
//
//  ---------------------------------------------------------------------------
//  THE SIGNATURE, AND WHY ALL FOUR PARTS ARE NEEDED
//  ---------------------------------------------------------------------------
//  A pane is a patch that is, at the same time:
//
//   1. LiDAR-SILENT. No returns at all across a connected patch.
//   2. IMAGE-BRIGHT. Something is visibly there. A silent patch that is also
//      DARK is a black sofa or a matte black speaker swallowing the beam, and
//      treating that as glass would be wrong in the opposite direction.
//   3. PLANAR, and
//   4. inside an OTHERWISE-PLANAR surround. The ring of returning samples
//      around the patch has to fit one plane tightly. That is what separates a
//      window in a wall from an open doorway into a cluttered hall, and from
//      the ordinary silence you get looking down a corridor.
//
//  Then, independently: ARKit's own mesh classification labels some faces
//  `window`. Where its window anchors agree with a patch we found, confidence
//  goes up. Where they do not, we do not throw our own measurement away -
//  ARKit's classifier misses plenty of glass - the patch is simply held at the
//  lower "suspected" confidence, which `SmartGlassMask` already treats
//  differently (it tapers authority rather than zeroing it).
//
//  DIVISION OF LABOUR: detection is here. Applying a detected region per
//  pixel, per frame, is `Sources/Smart`'s `SmartGlassMask`, which reads the
//  `[GlassRegion]` this file produces. Neither re-does the other's work.
//
//  ---------------------------------------------------------------------------
//  TOLERANCES
//  ---------------------------------------------------------------------------
//  The planarity test is not a fixed number of millimetres. It is derived per
//  patch from `SmartDepthNoiseModel` at the range the surround was actually
//  measured at: a wall at 1 m has a ~6 mm 1-sigma and a wall at 4 m has a
//  ~2.8 cm one, so a single threshold would either reject every far wall or
//  accept every near pile of cushions. The test is "the ring fits a plane to
//  within twice its own measurement noise", which is the same statement at
//  every range.
//

import Foundation
import simd

// MARK: - Result

/// What one glass pass found.
struct PrePassGlassResult {
    var regions: [GlassRegion] = []
    /// Fraction of the surveyed depth samples that fell inside a detected
    /// pane. This is the QC card's `glassAreaFraction`: it is a fraction of
    /// SAMPLED SURFACE, not of the room's floor plan, and the card's wording
    /// says so.
    var areaFraction: Float = 0
    var keyframesUsed: Int = 0
}

// MARK: - Detector

enum PrePassGlassDetector {

    struct Settings {
        /// Frames opened. Each one costs a depth read plus a small JPEG
        /// decode, so this is the main cost knob.
        var maxKeyframes = 40
        /// Normalised luma at or above which a silent pixel counts as bright.
        /// Matches `SmartGlassMask`'s own default so the detector and the
        /// per-pixel mask that consumes its output agree about what "bright"
        /// means.
        var brightLuma: Float = 0.55
        /// Smallest silent patch worth calling a pane, in native samples. 40
        /// samples of a 256x192 map is about 0.08% of the frame, which at 2 m
        /// is roughly a 20 cm square: smaller than that and a plane fit
        /// through the surround is not measuring a window, it is measuring
        /// noise around a dark object.
        var minComponentSamples = 40
        /// Largest patch worth calling a pane. Past this the "silence" is the
        /// whole view, which is a corridor or the sky, not a window.
        var maxComponentSampleFraction: Float = 0.35
        /// How far outside the patch the surround ring is gathered, in native
        /// pixels.
        var ringRadius = 3
        /// Ring samples needed before a plane fit means anything.
        var minRingSamples = 60
        /// Fraction of the ring that has to sit on the fitted plane.
        var minRingInlierFraction: Float = 0.75
        /// Multiple of the ring's own measurement noise the plane fit has to
        /// come in under. See the file header.
        var planarityNoiseMultiple: Float = 2.0
        /// How close an ARKit window or door anchor has to be to count as
        /// independent agreement.
        var arkitAgreementMeters: Float = 1.0
        /// Two detections merge into one region when their planes agree this
        /// closely and their extents nearly touch.
        var mergeNormalDegrees: Float = 15
        var mergeOffsetMeters: Float = 0.12
        var mergeGapMeters: Float = 0.50
        /// Below this a detection is not reported at all.
        var minConfidence: Float = 0.30
        /// Frames listed per region. A pane seen in four hundred frames does
        /// not need four hundred entries in a JSON sidecar to be believed.
        var maxObservedFrames = 64

        init() {}
    }

    /// Finds panes across a spread of keyframes and merges what is really one
    /// window seen many times into one region.
    static func detect(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        poseFor: (CaptureFrame) -> Pose,
        noiseModel: SmartDepthNoiseModel = .default,
        settings: Settings = Settings()
    ) throws -> PrePassGlassResult {
        var result = PrePassGlassResult()

        let frames = bundle.frames
            .filter { $0.depthPath != nil && $0.qc.trackingQuality != .notAvailable }
            .sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard !frames.isEmpty else { return result }

        let step = Swift.max(1, frames.count / Swift.max(settings.maxKeyframes, 1))
        var keyframes: [CaptureFrame] = []
        var i = 0
        while i < frames.count {
            keyframes.append(frames[i])
            i += step
        }
        result.keyframesUsed = keyframes.count

        let geometry = PrePassDepthGeometry(
            rgbIntrinsics: bundle.intrinsics, settings: bundle.settings
        )
        let width = geometry.width
        let height = geometry.height
        let sampleCount = width * height
        guard sampleCount > 0 else { return result }
        let maxRange = Swift.max(bundle.settings.lidarMaxRangeMeters, 0.5)
        let maxComponent = Int(Float(sampleCount) * settings.maxComponentSampleFraction)

        let anchorCentres = windowAnchorCentres(bundle: bundle)

        var merged: [MutableRegion] = []
        var returningSamples = 0
        var paneSamples = 0

        // Scratch reused across frames: allocating three 49k arrays per frame
        // over forty frames is pure churn.
        var label = [Int32](repeating: -1, count: sampleCount)
        var stack: [Int] = []
        stack.reserveCapacity(1024)

        for frame in keyframes {
            try Task.checkCancellation()

            let depthFrame: PrePassDepthFrame?
            do {
                depthFrame = try PrePassDepthFrame.load(
                    frame: frame, settings: bundle.settings, at: ref
                )
            } catch {
                continue
            }
            guard let depthFrame else { continue }

            // The image at NATIVE depth resolution. ARKit's depth map is
            // spatially aligned to the colour image, so one native depth
            // sample and one pixel of this decode are the same ray. Decoding
            // at 256x192 rather than 1920x1440 is what keeps this affordable.
            guard let gray = PrePassImageLoader.loadGray(
                url: ref.url(forRelativePath: frame.imagePath),
                width: width, height: height
            ) else { continue }

            let pose = poseFor(frame)
            let cameraCentre = pose.center.simd
            let rotationInverse = pose.rotation.simd.inverse

            for index in 0..<sampleCount { label[index] = -1 }

            var silentBright = [Bool](repeating: false, count: sampleCount)
            for index in 0..<sampleCount {
                if depthFrame.hasReturn(at: index) {
                    let z = depthFrame.depthMeters(at: index)
                    if z > 0.05, geometry.range(index: index, depthMeters: z) <= maxRange {
                        returningSamples += 1
                    }
                } else {
                    silentBright[index] = Float(gray.pixels[index]) / 255 >= settings.brightLuma
                }
            }

            var componentID: Int32 = 0
            for seed in 0..<sampleCount {
                guard silentBright[seed], label[seed] < 0 else { continue }

                // --- Flood fill the silent-and-bright patch, 4-connected.
                let currentLabel = componentID
                componentID += 1
                var members: [Int] = []
                stack.removeAll(keepingCapacity: true)
                stack.append(seed)
                label[seed] = currentLabel
                while let current = stack.popLast() {
                    members.append(current)
                    if members.count > maxComponent { break }
                    let cx = current % width
                    let cy = current / width
                    if cx > 0, silentBright[current - 1], label[current - 1] < 0 {
                        label[current - 1] = currentLabel
                        stack.append(current - 1)
                    }
                    if cx < width - 1, silentBright[current + 1], label[current + 1] < 0 {
                        label[current + 1] = currentLabel
                        stack.append(current + 1)
                    }
                    if cy > 0, silentBright[current - width], label[current - width] < 0 {
                        label[current - width] = currentLabel
                        stack.append(current - width)
                    }
                    if cy < height - 1, silentBright[current + width],
                       label[current + width] < 0 {
                        label[current + width] = currentLabel
                        stack.append(current + width)
                    }
                }

                guard members.count >= settings.minComponentSamples,
                      members.count <= maxComponent
                else { continue }

                // --- The returning ring around it.
                var ringIndices = Set<Int>()
                let radius = settings.ringRadius
                for member in members {
                    let mx = member % width
                    let my = member / width
                    var dy = -radius
                    while dy <= radius {
                        let ny = my + dy
                        if ny >= 0 && ny < height {
                            var dx = -radius
                            while dx <= radius {
                                let nx = mx + dx
                                if nx >= 0 && nx < width {
                                    let neighbour = ny * width + nx
                                    if label[neighbour] != currentLabel,
                                       depthFrame.hasReturn(at: neighbour) {
                                        let z = depthFrame.depthMeters(at: neighbour)
                                        let r = geometry.range(
                                            index: neighbour, depthMeters: z
                                        )
                                        if r > 0.1, r <= maxRange {
                                            ringIndices.insert(neighbour)
                                        }
                                    }
                                }
                                dx += 1
                            }
                        }
                        dy += 1
                    }
                }
                guard ringIndices.count >= settings.minRingSamples else { continue }

                var ringPoints: [SIMD3<Float>] = []
                ringPoints.reserveCapacity(ringIndices.count)
                var ringRangeSum: Float = 0
                for ringIndex in ringIndices {
                    let z = depthFrame.depthMeters(at: ringIndex)
                    let camera = geometry.cameraPoint(index: ringIndex, depthMeters: z)
                    ringPoints.append(cameraCentre + rotationInverse.act(camera))
                    ringRangeSum += geometry.range(index: ringIndex, depthMeters: z)
                }
                let meanRingRange = ringRangeSum / Float(ringPoints.count)

                guard var plane = PrePassPlane.fit(ringPoints) else { continue }

                // Planarity, judged against the sensor's own noise at the
                // range this ring was actually measured at.
                let ringSigma = noiseModel.sigma(
                    rangeMeters: meanRingRange, incidenceCosine: 1
                )
                let planeLimit = Swift.max(
                    ringSigma * settings.planarityNoiseMultiple, 0.005
                )
                guard plane.rmsMeters <= planeLimit else { continue }

                // "Otherwise planar": most of the ring, not merely its average,
                // has to be on that plane. An average can be dragged flat by a
                // symmetric pair of outliers.
                var inliers = 0
                for point in ringPoints
                where abs(plane.signedDistance(to: point)) <= planeLimit * 1.5 {
                    inliers += 1
                }
                let inlierFraction = Float(inliers) / Float(ringPoints.count)
                guard inlierFraction >= settings.minRingInlierFraction else { continue }

                // Orient the plane towards the camera, so two detections of
                // the same window from the same side are directly comparable.
                // A total-least-squares normal has an arbitrary sign.
                if plane.signedDistance(to: cameraCentre) < 0 {
                    plane.normal = -plane.normal
                    plane.offset = -plane.offset
                }

                // --- Where the pane sits in the world: the patch's own rays,
                //     intersected with the surround's plane. No geometry is
                //     invented beyond the plane the wall itself measured.
                var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
                var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
                var hits = 0
                var lumaSum: Float = 0
                for member in members {
                    let direction = rotationInverse.act(geometry.rayDirections[member])
                    let denominator = simd_dot(plane.normal, direction)
                    guard abs(denominator) > 1e-4 else { continue }
                    let t = -(simd_dot(plane.normal, cameraCentre) + plane.offset) / denominator
                    guard t > 0.1, t <= maxRange * 1.5 else { continue }
                    let world = cameraCentre + direction * t
                    minimum = simd_min(minimum, world)
                    maximum = simd_max(maximum, world)
                    lumaSum += Float(gray.pixels[member]) / 255
                    hits += 1
                }
                guard hits >= settings.minComponentSamples else { continue }
                paneSamples += hits

                let bounds = BoundingBox(min: Vector3(minimum), max: Vector3(maximum))
                let centre = 0.5 * (minimum + maximum)
                let confirmed = anchorCentres.contains {
                    simd_distance($0, centre) <= settings.arkitAgreementMeters
                }

                let brightness = lumaSum / Float(hits)
                let confidence = confidenceScore(
                    brightness: brightness,
                    brightLuma: settings.brightLuma,
                    planeRMS: plane.rmsMeters,
                    planeLimit: planeLimit,
                    sampleCount: hits,
                    ringInlierFraction: inlierFraction,
                    confirmedByARKit: confirmed
                )

                mergeIn(
                    &merged,
                    plane: plane,
                    bounds: bounds,
                    frame: frame.index,
                    confidence: confidence,
                    confirmedByARKit: confirmed,
                    settings: settings
                )
            }
        }

        result.regions = merged
            .filter { $0.confidence >= settings.minConfidence }
            .map { region in
                GlassRegion(
                    planeNormal: Vector3(region.normal),
                    planeOffset: region.offset,
                    bounds: BoundingBox(min: Vector3(region.minimum), max: Vector3(region.maximum)),
                    observedInFrames: Array(region.frames.prefix(settings.maxObservedFrames)),
                    confidence: region.confidence,
                    confirmedByARKit: region.confirmedByARKit
                )
            }
            .sorted { $0.confidence > $1.confidence }

        let denominator = returningSamples + paneSamples
        result.areaFraction = denominator > 0
            ? Float(paneSamples) / Float(denominator)
            : 0
        return result
    }

    // MARK: Internals

    /// Confidence from the four independent signals, weighted by how much each
    /// one actually discriminates.
    ///
    /// Planarity carries the most weight because it is the part that is hard
    /// to fake: a bright silent patch is common (a lamp, a white wall in
    /// sunlight, a black television), a bright silent patch cut out of a
    /// tightly-fitting plane is not. ARKit agreement is a bonus rather than a
    /// requirement, because its window classifier misses a great deal of
    /// glass and treating its silence as a veto would throw away real panes.
    static func confidenceScore(
        brightness: Float,
        brightLuma: Float,
        planeRMS: Float,
        planeLimit: Float,
        sampleCount: Int,
        ringInlierFraction: Float,
        confirmedByARKit: Bool
    ) -> Float {
        let brightnessTerm = clamp01((brightness - brightLuma) / Swift.max(1 - brightLuma, 1e-3))
        let planarityTerm = clamp01(1 - planeRMS / Swift.max(planeLimit, 1e-4))
        // 200 samples is roughly a 45 cm square at 2 m: a real window rather
        // than a gap between two cushions.
        let sizeTerm = clamp01(Float(sampleCount) / 200)
        let ringTerm = clamp01((ringInlierFraction - 0.5) / 0.5)

        let base = 0.25 * brightnessTerm
            + 0.35 * planarityTerm
            + 0.20 * sizeTerm
            + 0.20 * ringTerm
        return clamp01(confirmedByARKit ? base + 0.25 : base)
    }

    @inline(__always)
    private static func clamp01(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return Swift.min(Swift.max(value, 0), 1)
    }

    /// World positions of ARKit anchors it classified as a window or a door.
    private static func windowAnchorCentres(bundle: CaptureBundle) -> [SIMD3<Float>] {
        let anchors = bundle.anchorsAtEndOfSession.isEmpty
            ? bundle.anchorsDuringSession
            : bundle.anchorsAtEndOfSession
        var centres: [SIMD3<Float>] = []
        for anchor in anchors
        where anchor.classification == .window || anchor.classification == .door {
            centres.append(simd_make_float3(anchor.matrix.columns.3))
        }
        return centres
    }

    /// One window under construction, before it becomes a `GlassRegion`.
    struct MutableRegion {
        var normal: SIMD3<Float>
        var offset: Float
        var minimum: SIMD3<Float>
        var maximum: SIMD3<Float>
        var frames: [FrameID]
        var confidence: Float
        var confirmedByARKit: Bool
        /// Running count, so the merged plane is the average of its
        /// detections rather than whichever one happened to be first.
        var detectionCount: Int
    }

    private static func mergeIn(
        _ regions: inout [MutableRegion],
        plane: PrePassPlane,
        bounds: BoundingBox,
        frame: FrameID,
        confidence: Float,
        confirmedByARKit: Bool,
        settings: Settings
    ) {
        let minimum = bounds.min.simd
        let maximum = bounds.max.simd

        for index in regions.indices {
            let angle = PrePassAngle.degreesBetween(regions[index].normal, plane.normal)
            guard angle <= settings.mergeNormalDegrees else { continue }
            guard abs(regions[index].offset - plane.offset) <= settings.mergeOffsetMeters
            else { continue }
            // The two extents have to nearly touch: two windows in the same
            // wall are coplanar and must not merge into one enormous pane.
            let gapLow = regions[index].minimum - maximum
            let gapHigh = minimum - regions[index].maximum
            let gap = simd_max(gapLow, gapHigh)
            guard Swift.max(gap.x, Swift.max(gap.y, gap.z)) <= settings.mergeGapMeters
            else { continue }

            let n = Float(regions[index].detectionCount)
            regions[index].normal = simd_normalize(
                (regions[index].normal * n + plane.normal) / (n + 1)
            )
            regions[index].offset = (regions[index].offset * n + plane.offset) / (n + 1)
            regions[index].minimum = simd_min(regions[index].minimum, minimum)
            regions[index].maximum = simd_max(regions[index].maximum, maximum)
            if !regions[index].frames.contains(frame) { regions[index].frames.append(frame) }
            // Seeing the same pane again is corroboration, so confidence
            // climbs towards 1 rather than being replaced. Each further
            // detection closes a fifth of the remaining gap: real windows get
            // seen many times, and a one-off never reaches the same score.
            regions[index].confidence += (1 - regions[index].confidence)
                * Swift.min(confidence, 1) * 0.2
            regions[index].confidence = clamp01(regions[index].confidence)
            regions[index].confirmedByARKit = regions[index].confirmedByARKit || confirmedByARKit
            regions[index].detectionCount += 1
            return
        }

        regions.append(
            MutableRegion(
                normal: plane.normal,
                offset: plane.offset,
                minimum: minimum,
                maximum: maximum,
                frames: [frame],
                confidence: confidence,
                confirmedByARKit: confirmedByARKit,
                detectionCount: 1
            )
        )
    }
}
