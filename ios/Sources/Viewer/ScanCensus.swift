//
//  ScanCensus.swift
//  Viewer
//
//  WHERE THIS SCAN'S GEOMETRY WENT, COUNTED RATHER THAN GUESSED.
//
//  ---------------------------------------------------------------------------
//  WHY THIS FILE EXISTS
//  ---------------------------------------------------------------------------
//  The first real scan this app produced "looked like nothing", and finding out
//  why took a day of reading code, because the app recorded nothing about its
//  own behaviour. Four separate faults had each destroyed most of the model and
//  not one of them logged, warned or failed:
//
//    * a densification threshold in the wrong units, so the step that adds
//      detail created ZERO splats on every run that has ever happened;
//    * a thermal cut that landed below the live splat population, so real
//      geometry was deleted and densification was switched off for the rest of
//      the run;
//    * two pruning-window settings that were declared, defaulted, assigned and
//      read by nothing, so pruning ran 29 times instead of about 10;
//    * a trust gate set at a sigma no handheld scan reaches, so essentially
//      every seed was rejected.
//
//  Every one of those is a NUMBER GOING DOWN AT A NAMED STEP. None of them is
//  subtle once the numbers are on a screen next to each other. This file is
//  that screen's model: one ladder of counts from "photos you recorded" to
//  "points that can actually draw", each one carrying where it was read from.
//
//  ---------------------------------------------------------------------------
//  THE THREE RULES THIS FILE ENFORCES STRUCTURALLY
//  ---------------------------------------------------------------------------
//  1. A NUMBER IS NEVER SHOWN AS MEASURED WHEN IT WAS DEFAULTED. `Figure` has
//     no public initialiser that takes a count without also naming the file the
//     count came from, and no way to construct an absent figure that carries a
//     number. A missing count is `nil` and prints as "not recorded", never 0.
//
//  2. A MISSING RECORD SAYS SO. A scan built before the app kept a build record
//     does not render as a row of zeros that reads like catastrophe; it renders
//     as "this was built before the app kept a record of its own steps".
//
//  3. NOTHING HERE BLAMES THE USER FOR SOMETHING THE APP DID. Sentences about
//     the build name the build. Sentences about the recording state a fact
//     about the photos without a "you".
//
//  ---------------------------------------------------------------------------
//  WHAT IS MEASURED HERE AND WHAT HAS TO BE ASKED FOR
//  ---------------------------------------------------------------------------
//  Everything the Viewer can read for itself is measured for itself, TODAY,
//  with no help from any other module:
//
//    capture_bundle.json     photos recorded; photos whose quality weight is
//                            above zero (`FrameQC.weight` is the multiplier the
//                            trainer applies to a frame's loss, so a zero
//                            weight photo contributes nothing photometric -
//                            that is Core's own wording, not an inference)
//    prepass/prepass_result.json
//                            starting points the depth data produced
//                            (`InitialSplatSetRef.splatCount`), and the budget
//                            the pre-pass suggested
//    model/model.json        points in the finished model, iterations
//                            completed, and the budget the run ACTUALLY ended
//                            on (`SplatModel.budgetUsed`)
//    model/model.ply         measured by `Drawable.measure`: how many of those
//                            points can draw at all, how many are needles
//
//  The four counts inside the build that only the build can see - seeds
//  rejected by the trust gate, points created by densification, points deleted
//  by pruning, points deleted by a thermal cut - are read from two OPTIONAL
//  sidecars, `prepass/census.json` and `model/census.json`, whose schema is
//  `ScanCensus.Record` below and which are requested from the PrePass and
//  Trainer modules in INTEGRATION_REQUESTS.md. When they are absent the ladder
//  says "not recorded" for those rungs and the headline says the breakdown is
//  incomplete. It never guesses.
//

import Foundation
import simd

// MARK: - The census

/// One scan's count of where its geometry went, plus the one sentence that
/// says it out loud.
///
/// A pure value built from readings that are already in memory: making one
/// touches no files, so a library row can hold one per scan.
struct ScanCensus: Sendable, Equatable {

    // MARK: Where a number came from

    /// The provenance of a single number. Carried beside every count so a
    /// screen can never present a setting as a measurement or a default as a
    /// fact.
    enum Source: Sendable, Equatable {
        /// Counted from real data in this file, relative to the scan folder.
        case measured(file: String)
        /// Recorded in this file, but it is a SETTING the run was given, not a
        /// count of anything that happened.
        case declared(file: String)
        /// Nobody wrote this down. The string is why, in plain words.
        case notRecorded(reason: String)
        /// This step did not happen at all. The string is why.
        case notApplicable(reason: String)

        var isMeasured: Bool {
            if case .measured = self { return true }
            return false
        }

        /// The small grey line under a figure, naming the evidence.
        var evidence: String {
            switch self {
            case .measured(let file): return "counted in \(file)"
            case .declared(let file): return "a setting recorded in \(file)"
            case .notRecorded(let reason): return reason
            case .notApplicable(let reason): return reason
            }
        }

        /// True only for "nobody wrote it down". A step that did not run is not
        /// a hole in the record; it is a fact about the scan.
        var isMissingRecord: Bool {
            if case .notRecorded = self { return true }
            return false
        }

        /// What to print INSTEAD of a number when there is no number.
        var absenceText: String {
            switch self {
            case .measured, .declared: return ""
            case .notRecorded: return "not recorded"
            case .notApplicable: return "did not run"
            }
        }
    }

    /// One count and its provenance. There is deliberately no way to build one
    /// of these with a number and no source, and no way to build an absent one
    /// that still carries a number: rule 1 at the top of this file is enforced
    /// by the type, not by everyone remembering it.
    struct Figure: Sendable, Equatable {
        private(set) var count: Int?
        private(set) var source: Source

        static func measured(_ count: Int, in file: String) -> Figure {
            Figure(count: count, source: .measured(file: file))
        }

        static func declared(_ count: Int, in file: String) -> Figure {
            Figure(count: count, source: .declared(file: file))
        }

        static func notRecorded(_ reason: String) -> Figure {
            Figure(count: nil, source: .notRecorded(reason: reason))
        }

        static func notApplicable(_ reason: String) -> Figure {
            Figure(count: nil, source: .notApplicable(reason: reason))
        }

        /// True only when this is a real count of a real thing. A declared
        /// setting is not a measurement and answers false.
        var isMeasured: Bool { count != nil && source.isMeasured }

        /// What to draw where the number goes.
        var text: String {
            guard let count else { return source.absenceText }
            return ScanCensus.grouped(count)
        }
    }

    // MARK: One rung of the ladder

    /// What a rung is counting. Printed after the number, so "43,112" is never
    /// left to be guessed at.
    enum Unit: Sendable, Equatable {
        case photos
        case points
        case rounds

        func noun(_ count: Int) -> String {
            switch self {
            case .photos: return count == 1 ? "photo" : "photos"
            case .points: return count == 1 ? "point" : "points"
            case .rounds: return count == 1 ? "round" : "rounds"
            }
        }
    }

    /// Where a stage sits in the story: a level the ladder passes through, or a
    /// change that happened between two levels.
    enum Kind: Sendable, Equatable {
        /// A population at a moment in time.
        case level
        /// Points this step ADDED.
        case added
        /// Points this step REMOVED.
        case removed
    }

    struct Stage: Identifiable, Sendable, Equatable {
        var id: String
        /// The heading, in the user's words.
        var title: String
        var figure: Figure
        var unit: Unit
        var kind: Kind
        /// One plain sentence saying what this step is for. No jargon.
        var plain: String
        /// Set when this rung is worth pointing at. Plain sentence.
        var alert: String?
    }

    /// A group of rungs, so the breakdown reads as three short lists rather
    /// than one wall of eleven numbers.
    struct Section: Identifiable, Sendable, Equatable {
        var id: String
        var title: String
        var stages: [Stage]
    }

    // MARK: The headline

    enum Tone: Sendable, Equatable {
        /// Nothing was lost anywhere the app can see.
        case good
        /// A fact, neither good nor bad.
        case neutral
        /// A named step lost most of the model.
        case lost
        /// The app cannot say, and says so.
        case unknown
    }

    struct Headline: Sendable, Equatable {
        var tone: Tone
        /// The one sentence at the top of the review screen.
        var sentence: String
        /// The same thing in a library row's width. Nil when a row should say
        /// nothing at all.
        var shortLine: String?
    }

    // MARK: Stored

    var headline: Headline
    var sections: [Section]
    /// Which files were read, and which were looked for and not found. This is
    /// the anti-archaeology part: it says where each number came from, so the
    /// next investigation starts from evidence instead of a code read.
    var evidence: [String]
    /// True when at least one rung of the ladder is missing a number.
    var isIncomplete: Bool

    /// A census for a scan nothing has been read for yet. Never shown as real:
    /// its headline says exactly what it is.
    static let unread = ScanCensus(
        headline: Headline(
            tone: .unknown,
            sentence: "This scan has not been looked at yet.",
            shortLine: nil
        ),
        sections: [],
        evidence: [],
        isIncomplete: true
    )
}

// MARK: - The optional on-disk build record

extension ScanCensus {

    /// `prepass/census.json` and `model/census.json`: what a pipeline stage
    /// counted about ITSELF while it ran.
    ///
    /// EVERY FIELD IS OPTIONAL AND THAT IS THE WHOLE POINT. A writer fills in
    /// what it genuinely counted and leaves out what it did not; a missing key
    /// reads as "not recorded" on screen, which is true, rather than as zero,
    /// which would look like a catastrophic failure of that step. Unknown
    /// extra keys are ignored, so a writer may add fields ahead of this reader.
    ///
    /// Two files rather than one shared file, each in the folder its writer
    /// already owns, so two modules never write the same path.
    ///
    /// The schema is requested from PrePass and Trainer in
    /// INTEGRATION_REQUESTS.md. Neither module is required to write it for the
    /// app to work: without it the ladder is shorter and says so.
    struct Record: Codable, Sendable, Equatable {

        /// Bumped only if the meaning of an existing key changes. A reader
        /// seeing a version it does not know refuses the file and says so,
        /// per docs/DATA_FORMAT.md section 9.
        static let currentFormatVersion = 1

        var formatVersion: Int?
        /// Free text naming the writer, e.g. "prepass" or "trainer". Shown to
        /// nobody; useful in a bug report.
        var writtenBy: String?

        // -- Pre-pass side ----------------------------------------------------

        /// Depth samples the seeder looked at before any gate.
        var depthSamplesOffered: Int?
        /// Depth samples that passed the trust gate.
        var depthSamplesAccepted: Int?
        /// Seeds actually written to `prepass/init_splats.ply`.
        var seedsWritten: Int?
        /// Seeds that came out shaped as confident discs rather than
        /// ray-elongated blobs. The fingerprint of a misplaced trust gate is
        /// this being near zero while `seedsWritten` is large.
        var seedsShapedAsConfidentDiscs: Int?
        /// The sigma, in metres, the trust gate demanded.
        var trustGateSigmaMeters: Float?
        /// The sigma the scan actually measured, middle value, metres. A gate
        /// tighter than this rejects most of the scan.
        var measuredMedianSigmaMeters: Float?

        // -- Trainer side -----------------------------------------------------

        /// Points the run began with, after seeding.
        var splatsAtStart: Int?
        /// Points densification created over the whole run.
        var splatsCreatedByDensification: Int?
        /// How many times the densification step ran.
        var densificationPassCount: Int?
        /// How many points densification CONSIDERED. Zero candidates with many
        /// passes is a threshold problem; many candidates and zero created is a
        /// different one, and the two need telling apart.
        var densificationCandidateCount: Int?
        /// Points pruning removed over the whole run.
        var splatsDeletedByPruning: Int?
        var pruningPassCount: Int?
        /// Points deleted because the budget ceiling was cut under heat.
        var splatsDeletedByHeatCut: Int?
        var heatCutCount: Int?
        /// Points alive when the run finished.
        var splatsAtEnd: Int?

        /// The cap the run was given, and the cap it ended on. Kept even
        /// though `SplatModel.budgetUsed` carries the final one, because
        /// `finalSplatCap` below `splatsAtEnd` is the signature of a ceiling
        /// that cut into the population it was supposed to be a ceiling for,
        /// and that comparison needs both numbers from the same moment.
        ///
        /// Iterations are deliberately NOT here: `model.json` already records
        /// `iterationsCompleted` and `budgetUsed.iterations`, and asking a
        /// second writer for the same number only creates a disagreement
        /// nobody would know how to resolve.
        var plannedSplatCap: Int?
        var finalSplatCap: Int?

        /// Whether this record's version is one this build understands.
        var isReadable: Bool {
            guard let formatVersion else { return true }
            return formatVersion <= Record.currentFormatVersion
        }
    }

    /// Reads one census sidecar. Returns the record, or a plain sentence saying
    /// why there is not one. Both may be nil: nil/nil means the file simply is
    /// not there, which is the ordinary case for every scan built so far.
    static func readRecord(at url: URL) -> (record: Record?, problem: String?) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return (nil, nil)
        }
        guard let data = try? Data(contentsOf: url) else {
            return (nil, "The build record \(url.lastPathComponent) could not be opened.")
        }
        guard let decoded = try? ContractsJSON.decoder().decode(Record.self, from: data) else {
            return (nil, "The build record \(url.lastPathComponent) could not be read.")
        }
        guard decoded.isReadable else {
            return (
                nil,
                "The build record \(url.lastPathComponent) was written by a newer "
                    + "version of the app, and this one will not guess at it."
            )
        }
        return (decoded, nil)
    }
}

// MARK: - What the Viewer measures out of the finished file

extension ScanCensus {

    /// Measured by walking the finished splat cloud: how much of it can put
    /// anything on screen at all, and what shape it is.
    ///
    /// EVERY TEST HERE IS VIEW INDEPENDENT. A splat rejected by one of these is
    /// rejected from every camera angle, so "cannot draw" means cannot draw,
    /// not "did not happen to be on screen". The renderer additionally rejects
    /// splats that land off screen or come out under a quarter of a pixel, and
    /// applies a screen-space filter that can only LOWER alpha further, so the
    /// counts below are a floor on what is discarded and never an
    /// exaggeration.
    struct Drawable: Sendable, Equatable {

        /// Alpha below which `viewer_splat_preprocess` discards a splat. This
        /// is `ViewerUniforms.alphaCutoff`, the renderer's own constant, not a
        /// number invented here.
        static let alphaCutoff: Float = 1.0 / 255.0

        /// Largest-to-smallest axis ratio at which a Gaussian has stopped being
        /// a surface patch and become a streak. Chosen for reporting only:
        /// nothing is discarded for being a needle, it is counted so that a
        /// field made entirely of streaks is visible on a screen instead of
        /// being inferred from a photograph of a smear.
        static let needleAxisRatio: Float = 50

        var total: Int
        /// Position, rotation, scale or opacity that is NaN or infinite. These
        /// can never draw.
        var nonFinite: Int
        /// Finite, but fainter than the renderer's alpha cutoff at every angle.
        var belowAlphaCutoff: Int
        /// Finite, and opaque enough to survive the cutoff.
        var drawable: Int
        /// Of the drawable ones, how many are streaks rather than patches.
        var needles: Int
        /// Of the drawable ones, how many span more than a quarter of the whole
        /// model. One of these can grey out a whole frame on its own.
        var giants: Int
        /// Middle axis ratio across the drawable splats, to within one
        /// histogram bin (see `measure`). 1 is a sphere, 50 is a streak.
        var medianAxisRatio: Float

        /// Which file these numbers were counted in, relative to the scan
        /// folder. Filled in by whoever opened the file, because `measure`
        /// itself is handed a cloud and has no idea where it came from.
        ///
        /// It matters: the renderer prefers `model/model.ply` but falls back to
        /// `model/model.spz`, and a census that names the wrong file is exactly
        /// the sort of small, confident inaccuracy that sends the next
        /// investigation to the wrong place.
        var sourceFile: String?

        /// Whether the model being drawn had the trainer's Mip-Splatting 3D
        /// low-pass filter folded into its scales and opacities before it left
        /// the trainer.
        ///
        ///  * `true`  - it did, so this is the model that was actually fitted.
        ///  * `false` - it did not, and the producer KNEW. Every splat draws
        ///              sharper and more solid than it was trained, and the
        ///              held-out score was not measured on what is on screen.
        ///  * `nil`   - the file could not say. `.ply`, `.spz` and `.glb` carry
        ///              no marker, so an imported model gets no claim made
        ///              about it either way.
        ///
        /// Stamped by `MetalSplatRenderer.load(_:)` next to `sourceFile`, for
        /// the same reason: `measure` is handed a cloud and cannot know.
        var filter3DFused: Bool?

        /// Walks a cloud once and counts. O(n) in time and O(1) in extra
        /// memory: the middle axis ratio comes from a fixed histogram rather
        /// than from sorting half a million floats.
        static func measure(_ cloud: SplatCloud) -> Drawable {
            let count = cloud.count
            // sigmoid(x) < c  <=>  x < ln(c / (1 - c)). Done once, in logit
            // space, so the per-splat test is one comparison.
            let logitCutoff = log(alphaCutoff / (1 - alphaCutoff))

            // log10 of the axis ratio, 0 (a sphere) to 6 (a million to one),
            // in 240 bins: each bin is 0.025 of a decade, about 6% in ratio,
            // which is far finer than any decision anyone makes from it.
            let binCount = 240
            let binsPerDecade = Float(binCount) / 6
            var histogram = [Int](repeating: 0, count: binCount)

            var nonFinite = 0
            var faint = 0
            var drawable = 0
            var needles = 0
            var giants = 0

            // "Giant" is relative to the model, so the model's own extent has
            // to be known first. One extra pass over positions only.
            var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            var sawFinite = false
            for p in cloud.positions where p.x.isFinite && p.y.isFinite && p.z.isFinite {
                lo = simd_min(lo, p)
                hi = simd_max(hi, p)
                sawFinite = true
            }
            let longestEdge = sawFinite ? (hi - lo).max() : 0
            let giantSize = longestEdge > 0 ? longestEdge * 0.25 : Float.greatestFiniteMagnitude

            for index in 0..<count {
                let p = cloud.positions[index]
                let s = cloud.logScales[index]
                let r = cloud.rotations[index]
                let o = cloud.opacityLogits[index]

                let finite = p.x.isFinite && p.y.isFinite && p.z.isFinite
                    && s.x.isFinite && s.y.isFinite && s.z.isFinite
                    && r.x.isFinite && r.y.isFinite && r.z.isFinite && r.w.isFinite
                    && o.isFinite
                if !finite {
                    nonFinite += 1
                    continue
                }

                if o < logitCutoff {
                    faint += 1
                    continue
                }

                drawable += 1

                // Axis lengths in metres. `logScales` is per-axis log sigma, so
                // exp() is the sigma itself.
                let axes = SIMD3<Float>(exp(s.x), exp(s.y), exp(s.z))
                let largest = axes.max()
                let smallest = Swift.max(axes.min(), 1e-9)
                let ratio = largest / smallest
                if ratio >= needleAxisRatio { needles += 1 }
                if largest >= giantSize { giants += 1 }

                let decade = ViewerMath.clamp(log10(Swift.max(ratio, 1)), 0, 5.999)
                let bin = Swift.min(binCount - 1, Int(decade * binsPerDecade))
                histogram[bin] += 1
            }

            // Middle of the histogram, reported at the bin's centre.
            var medianRatio: Float = 1
            if drawable > 0 {
                let target = drawable / 2
                var running = 0
                for bin in 0..<binCount {
                    running += histogram[bin]
                    if running > target {
                        let decade = (Double(bin) + 0.5) / Double(binsPerDecade)
                        medianRatio = Float(Foundation.pow(10.0, decade))
                        break
                    }
                }
            }

            return Drawable(
                total: count,
                nonFinite: nonFinite,
                belowAlphaCutoff: faint,
                drawable: drawable,
                needles: needles,
                giants: giants,
                medianAxisRatio: medianRatio
            )
        }
    }
}

// MARK: - What the census is built from

extension ScanCensus {

    /// Everything the census needs, already read off disk by
    /// `ScanLibraryReader`. Kept as a plain value on `ScanSummary` so that
    /// building a census costs no file access and a library of forty rows does
    /// not re-open forty scans.
    struct Inputs: Sendable, Equatable {

        // Capture
        var hasCaptureIndex = false
        /// Set when `capture_bundle.json` is there but this build will not read
        /// it (a format version it does not know). Plain sentence. Kept apart
        /// from `hasCaptureIndex` so the census never says "there is no index"
        /// about a scan that has one.
        var captureIndexProblem: String?
        var frameCount: Int?
        /// Photos whose `FrameQC.weight` is above zero. Core's own comment on
        /// that field: "0 does not mean discard; it means contributes nothing
        /// photometric", so this is a count of the photos that can move the
        /// model at all.
        var contributingPhotoCount: Int?
        /// Photos taken while ARKit was not tracking normally.
        var lostTrackingPhotoCount: Int?

        // Pre-pass
        var hasPrePass = false
        /// Set when `prepass/prepass_result.json` is there but this build will
        /// not read it. Same reasoning as `captureIndexProblem`.
        var prePassProblem: String?
        var seedSplatCount: Int?
        var suggestedBudget: TrainingBudget?
        var prePassRecord: Record?
        var prePassRecordProblem: String?

        // Model
        var hasModel = false
        var modelFileMissing = false
        var modelSplatCount: Int?
        var iterationsCompleted: Int?
        var usedBudget: TrainingBudget?
        var modelSource: ModelSource?
        var trainRecord: Record?
        var trainRecordProblem: String?
    }
}

// MARK: - Building the census

extension ScanCensus {

    /// A stage lost this much of what reached it before anything is called a
    /// loss. Below it, a step that removes some points is doing its job:
    /// pruning is SUPPOSED to remove bad Gaussians, and a headline that shouts
    /// about a healthy 12% prune would teach the owner to ignore this screen.
    static let lossFraction: Double = 0.35

    /// Builds the census. Pure: no file access, no main actor, no surprises.
    ///
    /// - Parameter drawable: the measurement of the finished splat file, when
    ///   the review screen has loaded it. Nil everywhere else, and the last
    ///   rung then honestly says the file has not been opened yet rather than
    ///   showing a zero.
    static func make(_ inputs: Inputs, drawable: Drawable? = nil) -> ScanCensus {
        let photos = photoSection(inputs)
        let build = buildSection(inputs)
        let finished = finishedSection(inputs, drawable: drawable)
        let sections = [photos, build, finished]

        let incomplete = sections
            .flatMap(\.stages)
            .contains { $0.figure.source.isMissingRecord }

        return ScanCensus(
            headline: headline(inputs, drawable: drawable, isIncomplete: incomplete),
            sections: sections,
            evidence: evidence(inputs, drawable: drawable),
            isIncomplete: incomplete
        )
    }

    // MARK: Section 1: the photos

    private static func photoSection(_ inputs: Inputs) -> Section {
        let indexFile = "capture_bundle.json"
        var stages: [Stage] = []

        stages.append(
            Stage(
                id: "photos.recorded",
                title: "Photos recorded",
                figure: inputs.frameCount.map { Figure.measured($0, in: indexFile) }
                    ?? .notRecorded("this scan has no index file, so its photos cannot be counted"),
                unit: .photos,
                kind: .level,
                plain: "Every picture the walk saved, with the laser depth taken at the same moment.",
                alert: nil
            )
        )

        var contributingAlert: String?
        if let total = inputs.frameCount, let usable = inputs.contributingPhotoCount,
           total > 0, Double(total - usable) >= Double(total) * lossFraction {
            contributingAlert =
                "\(grouped(total - usable)) of the \(grouped(total)) photos carry no weight, "
                + "so the build had far less to work from than the photo count suggests."
        }

        stages.append(
            Stage(
                id: "photos.contributing",
                title: "Photos the build can use",
                figure: inputs.contributingPhotoCount.map { Figure.measured($0, in: indexFile) }
                    ?? .notRecorded("no photo quality was recorded for this scan"),
                unit: .photos,
                kind: .level,
                plain: "A photo that came out too smeared, or was taken while the phone had lost "
                    + "track of where it was, is kept but counts for nothing when the model is "
                    + "built.",
                alert: contributingAlert
            )
        )

        if let lost = inputs.lostTrackingPhotoCount, lost > 0 {
            stages.append(
                Stage(
                    id: "photos.lostTracking",
                    title: "Photos taken while the phone was lost",
                    figure: .measured(lost, in: indexFile),
                    unit: .photos,
                    kind: .removed,
                    plain: "The phone did not know where it was standing for these, so where they "
                        + "belong in the room is a guess.",
                    alert: nil
                )
            )
        }

        if let budget = inputs.usedBudget {
            stages.append(
                Stage(
                    id: "photos.keyframes",
                    title: "Photos the build was set to train on",
                    figure: .declared(budget.keyframeCount, in: "model/model.json"),
                    unit: .photos,
                    kind: .level,
                    plain: "A number the build was given before it started, not a count of what "
                        + "it did. A long walk does not need every photo as a teaching example.",
                    alert: nil
                )
            )
        }

        return Section(id: "photos", title: "Your photos", stages: stages)
    }

    // MARK: Section 2: the build

    private static func buildSection(_ inputs: Inputs) -> Section {
        let prePassFile = "prepass/prepass_result.json"
        // Same distinction on the trainer side: no record at all, a record this
        // build refuses to read, and a record that simply left this key out are
        // three different facts and get three different sentences.
        let noTrainRecord: String
        if let problem = inputs.trainRecordProblem {
            noTrainRecord = problem
        } else if inputs.trainRecord != nil {
            noTrainRecord = "the build wrote a record but did not count this"
        } else {
            noTrainRecord = missingTrainRecordReason(inputs)
        }
        let record = inputs.trainRecord
        var stages: [Stage] = []

        // Depth readings either side of a gate that REJECTS readings.
        //
        // These two rungs appear only when a writer actually supplied them, and
        // that is a deliberate exception to this file's "a missing key reads as
        // not recorded" rule. The pre-pass does not have a gate of this shape:
        // it uses the trust weight to decide whether a seed is laid as a disc
        // or as a ray-stretched blob, and never to discard a reading. Rendering
        // two permanent "not recorded" rows for a step that does not exist
        // would teach the eye to skip a screen whose whole job is to be read.
        // What the pre-pass DOES measure about that decision is the next rung
        // down, "Starting points pinned to a surface", and the sigma comparison
        // attached to it.
        let notCounted = "the check-over wrote a build record but did not count this"
        if let pre = inputs.prePassRecord,
           pre.depthSamplesOffered != nil || pre.depthSamplesAccepted != nil {
            stages.append(
                Stage(
                    id: "build.samplesOffered",
                    title: "Depth readings looked at",
                    figure: pre.depthSamplesOffered.map {
                        Figure.measured($0, in: "prepass/census.json")
                    } ?? .notRecorded(notCounted),
                    unit: .points,
                    kind: .level,
                    plain: "Every laser measurement the check-over considered turning into a "
                        + "starting point.",
                    alert: nil
                )
            )
            stages.append(
                Stage(
                    id: "build.samplesAccepted",
                    title: "Depth readings accepted",
                    figure: pre.depthSamplesAccepted.map {
                        Figure.measured($0, in: "prepass/census.json")
                    } ?? .notRecorded(notCounted),
                    unit: .points,
                    kind: .level,
                    plain: "Readings the check-over judged precise enough to trust. A test set "
                        + "tighter than a handheld phone can reach would reject nearly all of "
                        + "them, which is a fault in the test and not in the walk.",
                    alert: trustGateAlert(pre)
                )
            )
        }

        stages.append(
            Stage(
                id: "build.seeds",
                title: "Starting points from the depth data",
                figure: inputs.seedSplatCount.map { Figure.measured($0, in: prePassFile) }
                    ?? seedAbsence(inputs),
                unit: .points,
                kind: .level,
                plain: "The rough cloud the build starts from, placed straight from the laser "
                    + "depth before any learning happens.",
                alert: seedMismatchAlert(inputs)
            )
        )

        // How many seeds came out as pinned discs rather than blobs free to
        // slide along the viewing ray. Only the check-over can see this, and it
        // is the earliest possible warning of the fault that turns a finished
        // model into a field of smears: by the time the Viewer can measure the
        // streaks itself, the whole build has already happened.
        if let discs = inputs.prePassRecord?.seedsShapedAsConfidentDiscs {
            stages.append(
                Stage(
                    id: "build.discs",
                    title: "Starting points pinned to a surface",
                    figure: .measured(discs, in: "prepass/census.json"),
                    unit: .points,
                    kind: .level,
                    plain: "A starting point the check-over trusts is pinned flat against the "
                        + "surface it measured. One it does not trust is left as a blob, free "
                        + "to slide towards or away from the camera while the model trains.",
                    // The count and the reason, on one rung. `discAlert` says
                    // how few were pinned; `trustGateSigmaAlert` says why, in
                    // the two numbers that make it a fault in the test rather
                    // than an opinion about the walk. Joined rather than
                    // preferred, because either on its own is half the story.
                    alert: joined(
                        discAlert(inputs.prePassRecord),
                        trustGateSigmaAlert(inputs.prePassRecord)
                    )
                )
            )
        }

        stages.append(
            Stage(
                id: "build.start",
                title: "Points the build started with",
                figure: record?.splatsAtStart.map { Figure.measured($0, in: "model/census.json") }
                    ?? .notRecorded(noTrainRecord),
                unit: .points,
                kind: .level,
                plain: "What was alive the moment training began.",
                alert: nil
            )
        )

        stages.append(
            Stage(
                id: "build.densified",
                title: "Points the build added itself",
                figure: record?.splatsCreatedByDensification.map {
                    Figure.measured($0, in: "model/census.json")
                } ?? .notRecorded(noTrainRecord),
                unit: .points,
                kind: .added,
                plain: "The step that finds a blurry patch and splits it into finer pieces. This "
                    + "is where nearly all of a good model's detail comes from.",
                alert: densificationAlert(record)
            )
        )

        stages.append(
            Stage(
                id: "build.pruned",
                title: "Points the build removed while tidying",
                figure: record?.splatsDeletedByPruning.map {
                    Figure.measured($0, in: "model/census.json")
                } ?? .notRecorded(noTrainRecord),
                unit: .points,
                kind: .removed,
                plain: "Removing faint or useless pieces is normal and makes the model smaller "
                    + "and sharper. Removing most of it is not.",
                alert: pruningAlert(record)
            )
        )

        stages.append(
            Stage(
                id: "build.heat",
                title: "Points deleted because the phone got warm",
                figure: record?.splatsDeletedByHeatCut.map {
                    Figure.measured($0, in: "model/census.json")
                } ?? .notRecorded(noTrainRecord),
                unit: .points,
                kind: .removed,
                plain: "When the phone heats up the build lowers its own size limit. That is "
                    + "meant to give back memory it has not used, not to throw away work "
                    + "already done.",
                alert: nil
            )
        )

        if let used = inputs.usedBudget {
            stages.append(
                Stage(
                    id: "build.cap",
                    title: "Size limit the build finished on",
                    figure: .declared(used.splatCap, in: "model/model.json"),
                    unit: .points,
                    kind: .level,
                    plain: "The most points this run was allowed to hold. The model can never be "
                        + "bigger than this, so a limit that dropped is a ceiling the model "
                        + "spent the rest of the run under.",
                    alert: capAlert(inputs)
                )
            )
        }

        if let used = inputs.usedBudget, let completed = inputs.iterationsCompleted {
            stages.append(
                Stage(
                    id: "build.iterations",
                    title: "Rounds of learning finished",
                    figure: .measured(completed, in: "model/model.json"),
                    unit: .rounds,
                    kind: .level,
                    plain: "Out of the \(grouped(used.iterations)) this run was set to do. "
                        + "Stopping early leaves the model soft rather than empty.",
                    alert: completed < used.iterations
                        ? "This run stopped \(grouped(used.iterations - completed)) rounds short "
                            + "of what it planned."
                        : nil
                )
            )
        }

        return Section(id: "build", title: "Building the model", stages: stages)
    }

    /// Why there is no build record, said accurately rather than generically.
    /// A model that was built on a computer or imported from somewhere else was
    /// never going to have one, and calling that a missing record would be
    /// blaming the phone for a file it never wrote.
    private static func missingTrainRecordReason(_ inputs: Inputs) -> String {
        guard let source = inputs.modelSource else {
            return "this build did not write model/census.json"
        }
        switch source {
        case .booster:
            return "this model was built on a computer, which does not write a build record"
        case .imported:
            return "this model was brought in from somewhere else, so this phone has no record "
                + "of how it was built"
        case .onDevice:
            return "this build did not write model/census.json"
        }
    }

    /// What to say where the seed count would be, when there is not one.
    private static func seedAbsence(_ inputs: Inputs) -> Figure {
        if let problem = inputs.prePassProblem { return .notRecorded(problem) }
        if inputs.hasPrePass {
            return .notRecorded("the check-over did not record how many starting points it made")
        }
        return .notApplicable("this scan has not been checked over yet")
    }

    // MARK: Section 3: the finished file

    private static func finishedSection(_ inputs: Inputs, drawable: Drawable?) -> Section {
        var stages: [Stage] = []

        stages.append(
            Stage(
                id: "final.count",
                title: "Detail points in the finished model",
                figure: finalFigure(inputs),
                unit: .points,
                kind: .level,
                plain: "What is actually in the file on this phone.",
                alert: nil
            )
        )

        let notOpened =
            "the model file has not been opened on this screen, so nothing has been counted in it"

        stages.append(
            Stage(
                id: "final.drawable",
                title: "Points that can actually draw",
                figure: drawable.map { Figure.measured($0.drawable, in: modelFileName($0)) }
                    ?? .notRecorded(notOpened),
                unit: .points,
                kind: .level,
                plain: "A point too faint to reach one step of transparency is thrown away by "
                    + "the preview from every angle. It is in the file and it is not in the "
                    + "picture.",
                alert: drawableAlert(drawable)
            )
        )

        if let measured = drawable {
            // The faint count is shown even when it is zero, because a zero
            // here is a real and reassuring measurement: it says nothing was
            // trained into invisibility. Every other rung in this file hides
            // itself when it has nothing to say; this one earns its place by
            // being the single most common way a model that looks fine on
            // paper puts nothing on screen.
            stages.append(
                Stage(
                    id: "final.faint",
                    title: "Points too faint to draw",
                    figure: .measured(measured.belowAlphaCutoff, in: modelFileName(measured)),
                    unit: .points,
                    kind: .removed,
                    plain: "Training can push a point's opacity so low that it falls under the "
                        + "preview's cutoff of one step in 255. Those points still take up room "
                        + "in the file, so the size and the count both look healthy while the "
                        + "screen stays empty.",
                    alert: nil
                )
            )

            if measured.nonFinite > 0 {
                stages.append(
                    Stage(
                        id: "final.broken",
                        title: "Points with broken numbers in them",
                        figure: .measured(measured.nonFinite, in: modelFileName(measured)),
                        unit: .points,
                        kind: .removed,
                        plain: "A position or size that came out as nonsense during training. "
                            + "These can never be drawn and should never have been written.",
                        alert: "The build wrote \(grouped(measured.nonFinite)) points that are "
                            + "not real numbers. That is a fault in the build, not in the scan."
                    )
                )
            }

            let shape = "A good point is a small flat patch lying on a surface. A streak is one "
                + "that was never pinned down, so it stretched out along the direction the "
                + "camera was looking. The middle point in this model is "
            stages.append(
                Stage(
                    id: "final.needles",
                    title: "Points that came out as streaks",
                    figure: .measured(measured.needles, in: modelFileName(measured)),
                    unit: .points,
                    kind: .level,
                    plain: shape + shapeWords(measured.medianAxisRatio) + ".",
                    alert: needleAlert(measured)
                )
            )

            if measured.giants > 0 {
                stages.append(
                    Stage(
                        id: "final.giants",
                        title: "Points bigger than a quarter of the model",
                        figure: .measured(measured.giants, in: modelFileName(measured)),
                        unit: .points,
                        kind: .level,
                        plain: "One of these can wash a whole view out with haze on its own.",
                        alert: nil
                    )
                )
            }
        }

        return Section(id: "final", title: "The finished file", stages: stages)
    }

    /// The name of the file the splat counts were taken from, or an honest
    /// stand-in when whoever opened it did not say.
    private static func modelFileName(_ drawable: Drawable) -> String {
        drawable.sourceFile ?? "this scan's model file"
    }

    private static func finalFigure(_ inputs: Inputs) -> Figure {
        if inputs.modelFileMissing {
            return .notRecorded("the model file this scan points at is not on the phone")
        }
        if let count = inputs.modelSplatCount {
            return .measured(count, in: "model/model.json")
        }
        return .notApplicable("this scan has not been built yet")
    }

    private static func shapeWords(_ ratio: Float) -> String {
        if ratio < 3 { return "a round patch, which is what a surface should look like" }
        if ratio < 10 { return "about \(Int(ratio.rounded())) times longer than it is thick" }
        if ratio < ScanCensus.Drawable.needleAxisRatio {
            return "about \(Int(ratio.rounded())) times longer than it is thick, which is "
                + "getting stretched"
        }
        return "about \(grouped(Int(ratio.rounded()))) times longer than it is thick, which is a "
            + "streak and not a surface"
    }

    // MARK: Per-stage alerts

    /// Two optional sentences as one alert, or nil when there is nothing to
    /// say. Written out rather than done inline because `a.map { ... } ?? b`
    /// over two optionals is the sort of expression that is easy to get subtly
    /// backwards.
    private static func joined(_ first: String?, _ second: String?) -> String? {
        switch (first, second) {
        case (nil, nil): return nil
        case (let one?, nil): return one
        case (nil, let two?): return two
        case (let one?, let two?): return one + " " + two
        }
    }

    /// The gate, held up against the data it was applied to.
    ///
    /// THIS IS THE SENTENCE THAT WOULD HAVE CAUGHT THE FIRST SCAN'S FOURTH
    /// FAULT, and it deliberately stands on its own two numbers. It used to be
    /// an optional postscript inside `trustGateAlert` below, which fires only
    /// when a writer supplies a rejected-sample count either side of a trust
    /// gate. This pipeline has no such gate: `PrePassInitialSplatBuilder` uses
    /// the trust weight to choose a seed's SHAPE, never to throw a reading
    /// away, so those two counts are correctly absent from
    /// `prepass/census.json` and always will be. The sigma pair, which the
    /// pre-pass does write, would have gone with them and never once been
    /// printed.
    ///
    /// Fires only when the gate is genuinely outside the data: a gate at 2 cm
    /// against a measured 3 cm is a threshold in the wrong place, and a gate at
    /// 4 cm against a measured 3 cm is a threshold doing its job.
    private static func trustGateSigmaAlert(_ record: Record?) -> String? {
        guard let record,
              let gate = record.trustGateSigmaMeters,
              let measured = record.measuredMedianSigmaMeters,
              gate > 0, measured > 0, gate < measured
        else { return nil }
        return "The test that decides whether a starting point can be trusted asked for depth "
            + "readings accurate to " + ViewerFormat.centimeters(gate * 100)
            + ", and this scan's readings are typically "
            + ViewerFormat.centimeters(measured * 100)
            + " out. A phone held in the hand does not reach that, so the test is in the wrong "
            + "place rather than the walk being bad."
    }

    /// The rejection-rate version of the same story, for a writer that does
    /// count samples either side of a gate. Nothing writes those two keys
    /// today; this stays because the reader is allowed to be ahead of its
    /// writers and because a future gate would want exactly this sentence.
    private static func trustGateAlert(_ record: Record) -> String? {
        guard let offered = record.depthSamplesOffered, offered > 0,
              let accepted = record.depthSamplesAccepted
        else { return nil }
        let rejected = offered - accepted
        guard Double(rejected) >= Double(offered) * lossFraction else { return nil }
        var sentence = "\(percentText(rejected, of: offered)) of the depth readings were rejected "
            + "before the build even began."
        if let sigma = trustGateSigmaAlert(record) {
            sentence += " " + sigma
        }
        return sentence
    }

    private static func densificationAlert(_ record: Record?) -> String? {
        guard let record else { return nil }
        guard let created = record.splatsCreatedByDensification else { return nil }
        guard let passes = record.densificationPassCount, passes > 0 else { return nil }
        guard created == 0 else { return nil }
        var sentence = "The step that adds detail ran \(grouped(passes)) times and created "
            + "nothing at all."
        if let candidates = record.densificationCandidateCount {
            sentence += candidates == 0
                ? " It never found a single point worth splitting, which points at the test it "
                    + "uses to choose them rather than at this scan."
                : " It found \(grouped(candidates)) points worth splitting and then split none "
                    + "of them."
        }
        return sentence
    }

    /// Pruning measured against every point the build ever had, not just the
    /// ones it started with.
    ///
    /// The denominator matters. Against `splatsAtStart` alone, a healthy run
    /// that seeds 43,000 points, creates 260,000 more and prunes 180,000 of
    /// them reads as having deleted "417% of what it started with", which is
    /// arithmetic nonsense dressed up as a warning. Against start plus created
    /// it reads as 59%, which is the truth and is below the threshold.
    private static func pruningAlert(_ record: Record?) -> String? {
        guard let record,
              let deleted = record.splatsDeletedByPruning,
              let start = record.splatsAtStart
        else { return nil }
        let everExisted = start + (record.splatsCreatedByDensification ?? 0)
        guard everExisted > 0, Double(deleted) >= Double(everExisted) * 0.6 else { return nil }
        var sentence = "Tidying removed \(percentText(deleted, of: everExisted)) of every point "
            + "this build ever had: \(grouped(deleted)) of \(grouped(everExisted))."
        if let passes = record.pruningPassCount {
            sentence += " It ran \(grouped(passes)) times."
        }
        return sentence
    }

    /// The three numbers the cap story needs, taken from the best source that
    /// recorded each one.
    ///
    /// The build's own record is preferred where it exists, because its planned
    /// and final caps come from the same moment as its splat counts. Where it
    /// does not exist the pre-pass's suggestion and the model's `budgetUsed`
    /// stand in, which is weaker (a build may legitimately start below the
    /// suggestion) and is why the sentences below never assert a cause the
    /// evidence does not carry.
    private static func capFacts(
        _ inputs: Inputs
    ) -> (planned: Int?, ceiling: Int?, population: Int?) {
        let record = inputs.trainRecord
        return (
            planned: record?.plannedSplatCap ?? inputs.suggestedBudget?.splatCap,
            ceiling: record?.finalSplatCap ?? inputs.usedBudget?.splatCap,
            population: record?.splatsAtEnd ?? inputs.modelSplatCount
        )
    }

    /// The cap comparison: a ceiling that came down mid-run, and the one state
    /// that should be impossible.
    private static func capAlert(_ inputs: Inputs) -> String? {
        let facts = capFacts(inputs)
        guard let ceiling = facts.ceiling else { return nil }

        // The impossible state. A ceiling below the population it is supposed
        // to be a ceiling FOR means points were deleted to fit under it. This
        // is the exact shape of the thermal ratchet that destroyed the first
        // scan, and it is here so that if anything ever reintroduces it, it is
        // one line on one screen instead of a day of reading code.
        if let population = facts.population, ceiling < population {
            return "The size limit ended up BELOW the number of points in the model "
                + "(\(grouped(ceiling)) against \(grouped(population))). A limit is supposed to "
                + "sit above what exists, so something cut it after the points were already "
                + "there. This is worth reporting."
        }

        guard let planned = facts.planned, ceiling < planned else { return nil }

        // Why the limit came down is only asserted when the build actually
        // recorded a heat cut. A build is allowed to start below the
        // check-over's suggestion for reasons that have nothing to do with
        // heat (an older phone, less free memory), and naming the wrong cause
        // is how a diagnosis screen loses its credibility.
        let cause: String
        if let cuts = inputs.trainRecord?.heatCutCount, cuts > 0 {
            cause = cuts == 1
                ? " It cut its limit once because the phone was warm."
                : " It cut its limit \(grouped(cuts)) times because the phone was warm."
        } else {
            cause = " A build lowers its own limit when the phone gets warm or when memory runs "
                + "short; this one did not record which, so both are possible."
        }

        let headroom = "This run set out with a limit of \(grouped(planned)) points and "
            + "finished on \(grouped(ceiling))."
        return headroom + cause + " The model could not grow past the lower number."
    }

    /// The headline version of the cap story, which is deliberately harder to
    /// trigger than the row version above.
    ///
    /// A ceiling that came down but was never reached cost this scan nothing,
    /// and leading with "most of this scan was lost" about it would be a lie
    /// that teaches the owner to ignore this screen. So the headline fires only
    /// when the lower ceiling actually bit: either the impossible case (a
    /// ceiling below the population it is supposed to cap, which means points
    /// were deleted to fit under it), or a model sitting hard against the
    /// reduced number with nowhere left to grow.
    private static func capHeadline(_ inputs: Inputs) -> Headline? {
        guard let sentence = capAlert(inputs) else { return nil }
        let facts = capFacts(inputs)
        guard let ceiling = facts.ceiling, let population = facts.population else { return nil }

        if ceiling < population {
            return Headline(
                tone: .lost,
                sentence: "Something cut this build's size limit after the model was already "
                    + "bigger than it. " + sentence,
                shortLine: "The size limit ended up below the model itself"
            )
        }

        guard Double(population) >= Double(ceiling) * 0.9 else { return nil }

        let warmth = (inputs.trainRecord?.heatCutCount ?? 0) > 0
        let opening = warmth
            ? "Most of this scan was lost when the phone got warm and the build shrank itself. "
            : "This scan stopped growing because the build shrank its own size limit. "
        return Headline(
            tone: .lost,
            sentence: opening + sentence + " It finished pressed right against that lower limit, "
                + "so it had more to give and was not allowed to.",
            shortLine: warmth
                ? "The build shrank itself when the phone got warm"
                : "The build shrank its own size limit part way through"
        )
    }

    /// Two records of the same number disagreeing. Worth a line on its own:
    /// when two files that should agree do not, one of them is wrong, and
    /// finding out which is far cheaper with the disagreement written down than
    /// with it discovered later.
    private static func seedMismatchAlert(_ inputs: Inputs) -> String? {
        guard let claimed = inputs.prePassRecord?.seedsWritten,
              let counted = inputs.seedSplatCount,
              claimed != counted
        else { return nil }
        return "The check-over's own record says it wrote \(grouped(claimed)) starting points, "
            + "and the file it points at says \(grouped(counted)). One of those two is wrong."
    }

    /// Seeds that came out as blobs rather than pinned discs. The fingerprint
    /// of a trust test set tighter than a handheld phone can reach.
    private static func discAlert(_ record: Record?) -> String? {
        guard let record,
              let discs = record.seedsShapedAsConfidentDiscs,
              let total = record.seedsWritten, total > 0,
              Double(discs) <= Double(total) * 0.25
        else { return nil }
        return "Only \(grouped(discs)) of \(grouped(total)) starting points were pinned to a "
            + "surface. The rest were left free to slide, which is what turns a model into "
            + "smears rather than walls. When nearly none are pinned, the test that decides "
            + "is asking for more precision than a phone in the hand can give."
    }

    private static func drawableAlert(_ drawable: Drawable?) -> String? {
        guard let drawable else { return nil }
        // Two independent things can be wrong with the same rung, so they are
        // joined rather than one hiding the other.
        var faint: String?
        if drawable.total > 0 {
            let lost = drawable.total - drawable.drawable
            if Double(lost) >= Double(drawable.total) * lossFraction {
                faint = "\(percentText(lost, of: drawable.total)) of the points in this file "
                    + "cannot put anything on screen from any angle."
            }
        }
        return joined(faint, unfusedFilterAlert(drawable))
    }

    /// Only `false` says anything. `nil` is a model read back from a file,
    /// which genuinely cannot know, and an alert that fired on every import
    /// would be noise rather than information.
    private static func unfusedFilterAlert(_ drawable: Drawable) -> String? {
        guard drawable.filter3DFused == false else { return nil }
        return "This model is being drawn sharper and more solid than it was actually built. "
            + "The softening the build fitted every point against was not saved into the file, "
            + "so what is on screen is not quite the model that was scored."
    }

    private static func needleAlert(_ drawable: Drawable) -> String? {
        guard drawable.drawable > 0 else { return nil }
        guard Double(drawable.needles) >= Double(drawable.drawable) * 0.5 else { return nil }
        return "More than half of this model is streaks rather than surfaces. That is what a "
            + "model looks like when the build never trusted the laser depth enough to pin its "
            + "points down."
    }

    // MARK: The headline

    // swiftlint:disable:next cyclomatic_complexity
    private static func headline(
        _ inputs: Inputs,
        drawable: Drawable?,
        isIncomplete: Bool
    ) -> Headline {

        // Nothing to count yet. The unreadable case is kept apart from the
        // missing case: telling someone their scan has no index when it has one
        // this build will not open is a different and untrue statement.
        if let problem = inputs.captureIndexProblem {
            return Headline(
                tone: .unknown,
                sentence: problem + " Until it can be read, nothing about this scan can be "
                    + "counted.",
                shortLine: "This build cannot read this scan's index"
            )
        }

        if !inputs.hasCaptureIndex {
            return Headline(
                tone: .unknown,
                sentence: "This folder has no scan index in it, so there is nothing to count.",
                shortLine: nil
            )
        }

        if inputs.modelFileMissing {
            return Headline(
                tone: .unknown,
                sentence: "The 3D file for this scan is missing from the phone, so there is "
                    + "nothing left to count. Building it again replaces it.",
                shortLine: "The 3D file is missing"
            )
        }

        if !inputs.hasModel {
            let photos = inputs.frameCount.map { grouped($0) + " " + Unit.photos.noun($0) }
                ?? "photos"
            return Headline(
                tone: .neutral,
                sentence: "Nothing has been built from this scan yet, so there is nothing to "
                    + "account for. It has \(photos) waiting.",
                shortLine: nil
            )
        }

        // From here on there IS a model. Named faults first, because a named
        // fault is more useful than the largest arithmetic drop.
        if let capped = capHeadline(inputs) {
            return capped
        }

        if let sentence = densificationAlert(inputs.trainRecord) {
            return Headline(
                tone: .lost,
                sentence: "This scan never got its detail. " + sentence
                    + " Whatever the walk was like, this model was never going to be more than "
                    + "the rough cloud it started from.",
                shortLine: "The step that adds detail added nothing"
            )
        }

        if let pre = inputs.prePassRecord, let sentence = trustGateAlert(pre) {
            return Headline(
                tone: .lost,
                sentence: "Most of this scan was thrown away before the build started. "
                    + sentence,
                shortLine: "Most depth readings were rejected before building"
            )
        }

        // The same fault, seen through the numbers this pipeline actually
        // records. It leads only when almost nothing was pinned: a gate a
        // little tighter than the data costs some sharpness and is not worth
        // the top line, while a gate that pinned nearly nothing is why the
        // model came out as smears.
        if let disc = discAlert(inputs.prePassRecord),
           let sigma = trustGateSigmaAlert(inputs.prePassRecord) {
            return Headline(
                tone: .lost,
                sentence: "This scan's starting points were never pinned to anything. "
                    + disc + " " + sigma,
                shortLine: "Almost no starting point was pinned to a surface"
            )
        }

        if let sentence = pruningAlert(inputs.trainRecord) {
            return Headline(
                tone: .lost,
                sentence: "Most of this scan was removed by the build's own tidying step. "
                    + sentence,
                shortLine: "Tidying removed most of the model"
            )
        }

        if let measured = drawable, let sentence = drawableAlert(measured) {
            return Headline(
                tone: .lost,
                sentence: "This model is mostly invisible. " + sentence
                    + " They are in the file, so the size looks right, and none of them reach "
                    + "the screen.",
                shortLine: "Most of the model is too faint to draw"
            )
        }

        if let measured = drawable, let sentence = needleAlert(measured) {
            return Headline(
                tone: .lost,
                sentence: "This model came out as smears rather than surfaces. " + sentence,
                shortLine: "The model came out as streaks, not surfaces"
            )
        }

        // Seeds in, points out. The plainest arithmetic there is, and it caught
        // three of the four original faults on its own.
        if let seeds = inputs.seedSplatCount, seeds > 0, let final = inputs.modelSplatCount {
            if Double(final) <= Double(seeds) * (1 - lossFraction) {
                return Headline(
                    tone: .lost,
                    sentence: "This scan finished smaller than it started: "
                        + "\(grouped(seeds)) starting points went in and \(grouped(final)) came "
                        + "out. The building step took away more than it added.",
                    shortLine: "Finished smaller than it started"
                )
            }
            if final <= seeds {
                return Headline(
                    tone: .lost,
                    sentence: "This build added no detail of its own: it started with "
                        + "\(grouped(seeds)) rough points and finished with \(grouped(final)). "
                        + "A finished model should be several times larger than the cloud it "
                        + "began with.",
                    shortLine: "The build added no detail of its own"
                )
            }
        }

        // Photos. Stated as a fact about the recording, never as a fault of the
        // person holding the phone.
        if let total = inputs.frameCount, let usable = inputs.contributingPhotoCount,
           total > 0, Double(total - usable) >= Double(total) * lossFraction {
            return Headline(
                tone: .neutral,
                sentence: "This scan had less to work from than it looks: of \(grouped(total)) "
                    + "photos, \(grouped(usable)) carry any weight. The rest came out smeared or "
                    + "were taken while the phone had lost track of the room, and the build "
                    + "counts those as nothing.",
                shortLine: "Only \(grouped(usable)) of \(grouped(total)) photos carried weight"
            )
        }

        // Nothing went wrong that the app can see. Say how much there is, and
        // say plainly if part of the story is missing.
        let final = inputs.modelSplatCount ?? 0
        let photos = inputs.frameCount ?? 0
        var built = "\(grouped(final)) detail \(Unit.points.noun(final)) from \(grouped(photos)) "
            + Unit.photos.noun(photos)
        // The ceiling is stated alongside, with no verdict attached to it. A
        // cap is a wall and not a target, so a model well under it is not
        // automatically a bad model and this screen will not say it is. But
        // 12,000 points against a limit of 300,000 is a fact worth seeing at a
        // glance, and it is the fact that makes the next question obvious.
        if let ceiling = capFacts(inputs).ceiling, ceiling > 0 {
            built += ", against a size limit of \(grouped(ceiling))"
        }

        if isIncomplete {
            return Headline(
                tone: .neutral,
                sentence: "This built \(built), and nothing went wrong at any step this app can "
                    + "see. Part of how it got there was never written down, though, so the "
                    + "breakdown has gaps: those are marked as not recorded rather than filled "
                    + "in with zeros.",
                shortLine: nil
            )
        }

        return Headline(
            tone: .good,
            sentence: "This built well: \(built), and nothing was lost at any step this app can "
                + "see.",
            shortLine: nil
        )
    }

    // MARK: Evidence

    private static func evidence(_ inputs: Inputs, drawable: Drawable?) -> [String] {
        var lines: [String] = []

        if let problem = inputs.captureIndexProblem {
            lines.append(problem)
        } else {
            lines.append(
                inputs.hasCaptureIndex
                    ? "Read capture_bundle.json."
                    : "capture_bundle.json is not there."
            )
        }
        if let problem = inputs.prePassProblem {
            lines.append(problem)
        } else {
            lines.append(
                inputs.hasPrePass
                    ? "Read prepass/prepass_result.json."
                    : "prepass/prepass_result.json is not there: this scan has not been "
                        + "checked over."
            )
        }
        if let problem = inputs.prePassRecordProblem {
            lines.append(problem)
        } else if let record = inputs.prePassRecord {
            lines.append("Read prepass/census.json" + writerSuffix(record))
        } else {
            lines.append(
                "prepass/census.json is not there, so what the check-over threw away is "
                    + "unknown rather than zero."
            )
        }
        lines.append(
            inputs.hasModel || inputs.modelFileMissing
                ? "Read model/model.json."
                : "model/model.json is not there: nothing has been built."
        )
        if let problem = inputs.trainRecordProblem {
            lines.append(problem)
        } else if let record = inputs.trainRecord {
            lines.append("Read model/census.json" + writerSuffix(record))
        } else {
            let reason = missingTrainRecordReason(inputs)
            lines.append(reason.prefix(1).uppercased() + String(reason.dropFirst()) + ".")
        }
        if let drawable {
            lines.append("Counted every point in \(modelFileName(drawable)).")
        } else {
            lines.append(
                "This scan's model file has not been opened here, so nothing in it has been "
                    + "counted."
            )
        }
        return lines
    }

    /// ", written by trainer" - free text the writer put in the file, shown
    /// only in the evidence list, where knowing which module produced a number
    /// is exactly what a bug report needs.
    private static func writerSuffix(_ record: Record) -> String {
        guard let writer = record.writtenBy, !writer.isEmpty else { return "." }
        return ", written by \(writer)."
    }

    // MARK: Formatting

    /// Digit grouping without a `NumberFormatter`, which is not a value type
    /// and would need a shared instance this file has no safe place for.
    static func grouped(_ value: Int) -> String {
        let digits = String(value.magnitude)
        var out = ""
        for (offset, character) in digits.enumerated() {
            if offset > 0 && (digits.count - offset) % 3 == 0 { out.append(",") }
            out.append(character)
        }
        return value < 0 ? "-" + out : out
    }

    /// "94 out of every 100" style text, which reads better to a non-technical
    /// eye than "94.2%", and rounds honestly.
    static func percentText(_ part: Int, of whole: Int) -> String {
        guard whole > 0 else { return "an unknown share" }
        let fraction = Double(part) / Double(whole)
        let percent = Int((fraction * 100).rounded())
        if percent >= 99 && part < whole { return "almost all" }
        if percent >= 100 { return "all" }
        return "\(percent)%"
    }
}
