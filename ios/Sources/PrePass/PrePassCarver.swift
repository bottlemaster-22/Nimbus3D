//
//  PrePassCarver.swift
//  PrePass
//
//  F2: "EMPTY AIR IS EVIDENCE."
//
//  A LiDAR beam that comes back from 3.2 m has told you two things, and every
//  generic 3DGS pipeline throws one of them away. It says there is a surface
//  at 3.2 m - everyone uses that. It ALSO says there is nothing at 1 m, or
//  2 m, or 3.1 m, along that exact line, because the beam got through. That
//  second fact is what kills floaters: a Gaussian sitting in mid-air is not
//  merely unsupported by the photometry, it is CONTRADICTED by tens of
//  thousands of beams that flew straight through where it claims to be.
//
//  The one rule that makes this safe rather than destructive:
//
//      UNKNOWN IS NOT EMPTY.
//
//  Space past the sensor's range, and space behind a no-return, is UNKNOWN
//  forever. Glass returns nothing, and "nothing came back" is not evidence of
//  absence - carving on no-returns without this rule erases everything visible
//  through every window in the scan. Deletion is licensed by `.empty` alone.
//
//  The useful exception, and it is worth having: a no-return ray still proves
//  the space between the sensor and the wall its NEIGHBOURS hit is empty. A
//  window pane sits in a wall, the wall around it returns, so the free-space
//  bound at the window is the neighbourhood's own depth. That recovers the
//  free space at an aperture without inventing one millimetre of geometry
//  beyond it.
//
//  ADAPTIVE, NEVER ASSUME: a whole-house scan at 5 cm is tens of millions of
//  cells and would not fit. The voxel size is therefore RAISED, and the frames
//  and rays SUBSAMPLED, until the estimate fits the cell budget - and the size
//  actually used is written into `OccupancyGridRef`, so nothing downstream has
//  to assume it got the 5 cm it asked for.
//

import Foundation
import os
import simd

// MARK: - Ray traversal

/// Amanatides-Woo 3D digital differential analyser: walks the exact sequence
/// of voxels a ray passes through, in order, with no gaps and no duplicates.
///
/// A naive "step along the ray by half a voxel and round" loop is the usual
/// shortcut and it is wrong in a way that matters here: it skips cells at
/// grazing angles and visits some twice, so the free space it carves has
/// pinholes in it exactly where the geometry is most oblique - which is where
/// floaters live.
enum PrePassRayWalk {

    /// Visits every cell the segment from `origin` to `origin + direction *
    /// distance` passes through, starting at `origin`'s own cell.
    ///
    /// - Parameter visit: return `false` to stop the walk early.
    @inline(__always)
    static func traverse(
        origin: SIMD3<Float>,
        direction: SIMD3<Float>,
        distance: Float,
        frame: PrePassVoxelFrame,
        maxSteps: Int,
        visit: (SIMD3<Int32>) -> Bool
    ) {
        guard distance > 0, distance.isFinite else { return }
        let length = simd_length(direction)
        guard length > 1e-9 else { return }
        let unit = direction / length

        var cell = frame.cell(origin)
        let endCell = frame.cell(origin + unit * distance)

        let step = SIMD3<Int32>(
            unit.x > 0 ? 1 : (unit.x < 0 ? -1 : 0),
            unit.y > 0 ? 1 : (unit.y < 0 ? -1 : 0),
            unit.z > 0 ? 1 : (unit.z < 0 ? -1 : 0)
        )

        // Distance along the ray to the next voxel boundary on each axis, and
        // the distance between successive boundaries on each axis.
        var tMax = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var tDelta = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        let voxel = frame.voxelSize
        let cellMin = frame.origin + SIMD3<Float>(Float(cell.x), Float(cell.y), Float(cell.z)) * voxel

        for axis in 0..<3 {
            let u = unit[axis]
            guard abs(u) > 1e-12 else { continue }
            let boundary = u > 0 ? (cellMin[axis] + voxel) : cellMin[axis]
            tMax[axis] = (boundary - origin[axis]) / u
            tDelta[axis] = voxel / abs(u)
        }

        var steps = 0
        while steps < maxSteps {
            if !visit(cell) { return }
            if cell == endCell { return }

            // Advance on whichever axis reaches its next boundary first.
            if tMax.x < tMax.y {
                if tMax.x < tMax.z {
                    if tMax.x > distance { return }
                    cell.x &+= step.x
                    tMax.x += tDelta.x
                } else {
                    if tMax.z > distance { return }
                    cell.z &+= step.z
                    tMax.z += tDelta.z
                }
            } else {
                if tMax.y < tMax.z {
                    if tMax.y > distance { return }
                    cell.y &+= step.y
                    tMax.y += tDelta.y
                } else {
                    if tMax.z > distance { return }
                    cell.z &+= step.z
                    tMax.z += tDelta.z
                }
            }
            steps += 1
        }
    }
}

// MARK: - The carver

/// `FreeSpaceCarver`, owned by `Sources/PrePass` (CONTRACTS.md section 5).
public final class VoxelFreeSpaceCarver: FreeSpaceCarver, @unchecked Sendable {

    public struct Tuning: Sendable {
        /// Hard ceiling on distinct cells. The voxel size is raised until the
        /// scene's volume is estimated to fit; at roughly 29 bytes per cell
        /// (hash slot plus payload) four million cells is about 120 MB, which
        /// is a sane share of a phone's budget for a stage that runs before
        /// training has allocated anything.
        public var maxCells = 4_000_000
        /// Never go finer than this even if the scene is tiny; below it the
        /// grid is measuring the sensor's own noise.
        public var minVoxelSizeMeters: Float = 0.02
        /// Or coarser than this even if the scene is enormous; past it a
        /// "cell" is bigger than the furniture in it and carving is meaningless.
        public var maxVoxelSizeMeters: Float = 0.20
        /// Carve from a keyframe subset spaced by movement or time. Every
        /// frame of a 30 fps walk carves the same air thirty times over.
        public var keyframeSpacingMeters: Float = 0.10
        public var keyframeSpacingSeconds: Double = 0.20
        /// Use every Nth native sample in each direction. 2 gives ~12k rays
        /// per frame out of 49k, which at 10 cm keyframe spacing still hits
        /// every 5 cm cell many times over.
        public var raySubsampleStride = 2
        /// Stop carving a beam this far short of its endpoint, so the surface
        /// cell itself is never marked empty by its own ray.
        public var surfaceMarginMeters: Float = 0.03
        /// Half-width, in native pixels, of the neighbourhood a no-return ray
        /// borrows its free-space bound from.
        public var noReturnNeighbourhoodRadius = 3
        /// And the safety margin subtracted from that borrowed bound.
        public var noReturnMarginMeters: Float = 0.15

        public init() {}
    }

    public var tuning: Tuning

    /// Set after `carve` so the pipeline can report honestly when the grid was
    /// coarsened to fit rather than silently claiming 5 cm.
    public private(set) var requestedVoxelSizeMeters: Float = 0
    public private(set) var actualVoxelSizeMeters: Float = 0
    /// True when the cell cap was reached and further EMPTY cells were
    /// dropped. Surfaces are never dropped.
    public private(set) var hitCellCap = false

    /// What the last carve counted about itself: rays followed, rays that
    /// proved nothing, and the empty / surface / unknown split of the cells.
    ///
    /// Assigned ONCE, at the end of `carve`, from plain local counters that
    /// the ray loop increments. The loop does integer adds and nothing else,
    /// so the measurement costs nothing and cannot itself be the reason a
    /// carve is slow. See `PrePassCensus`.
    public private(set) var lastCensus = PrePassCensus.Carving()

    /// The carve says out loud, here, what it could not say through a return
    /// value: a stage that produced nothing has to reach the log even when the
    /// census file itself fails to write.
    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem, category: "PrePassCarver"
    )

    // Loaded state, for `state(atWorldPoint:)` during training.
    private var loadedKeys: [UInt64] = []
    private var loadedStates: [UInt8] = []
    private var loadedFrame: PrePassVoxelFrame?

    public init(tuning: Tuning = Tuning()) {
        self.tuning = tuning
    }

    // MARK: Carving

    public func carve(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        voxelSizeMeters: Float
    ) async throws -> OccupancyGridRef {
        requestedVoxelSizeMeters = voxelSizeMeters
        // Reset to a census that says "this ran and found nothing" rather than
        // leaving the previous run's numbers in place. A stale number is worse
        // than a zero: a zero is a fact about this scan.
        var census = PrePassCensus.Carving()
        census.attempted = true
        census.requestedVoxelSizeMeters = voxelSizeMeters
        lastCensus = census
        // Drop any grid a previous carve left loaded. Without this, a carve
        // that is refused below would leave `state(atWorldPoint:)` answering
        // from the LAST scan's geometry, which is the worst kind of stale: a
        // confident answer about the wrong room.
        loadedKeys = []
        loadedStates = []
        loadedFrame = nil

        let frames = bundle.frames.sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard !frames.isEmpty else {
            throw NimbusError.prePassFailed("there are no frames to carve from")
        }

        // Bounds: the LiDAR extent if the capture measured it, otherwise the
        // camera path plus the sensor's own range in every direction. Never a
        // fixed guess.
        let bounds = sceneBounds(bundle: bundle, frames: frames)
        let size = bounds.sizeMeters.simd
        let volume = Double(Swift.max(size.x, 0.1))
            * Double(Swift.max(size.y, 0.1))
            * Double(Swift.max(size.z, 0.1))

        var voxel = Swift.max(voxelSizeMeters, tuning.minVoxelSizeMeters)
        let neededForBudget = Float((volume / Double(tuning.maxCells)).cubeRoot())
        if neededForBudget > voxel {
            voxel = Swift.min(neededForBudget, tuning.maxVoxelSizeMeters)
        }
        voxel = Swift.min(Swift.max(voxel, tuning.minVoxelSizeMeters), tuning.maxVoxelSizeMeters)
        actualVoxelSizeMeters = voxel
        hitCellCap = false

        let voxelFrame = PrePassVoxelFrame(bounds: bounds, voxelSize: voxel, marginMeters: 0.5)
        let geometry = PrePassDepthGeometry(rgbIntrinsics: bundle.intrinsics, settings: bundle.settings)
        let maxRange = bundle.settings.lidarMaxRangeMeters

        var hash = PrePassVoxelHash(expectedCount: 1 << 16)
        var states: [UInt8] = []
        var hits: [UInt16] = []
        states.reserveCapacity(1 << 16)
        hits.reserveCapacity(1 << 16)

        let keyframes = self.keyframes(from: frames)
        census.keyframesSelected = keyframes.count
        census.actualVoxelSizeMeters = voxel
        census.boundsSizeMeters = size

        // Local counters, incremented by the ray loop. Plain Ints on the
        // stack: no allocation, no retain, nothing to synchronise.
        var raysCast = 0
        var raysWithReturnInRange = 0
        var raysBeyondMaxRange = 0
        var raysTooClose = 0
        var raysNoReturn = 0
        var raysNoReturnBounded = 0
        var keyframesLoaded = 0
        var keyframesMissing = 0
        // The two halves of `keyframesMissing`, kept apart. Before this split
        // a depth sidecar that would not open and a frame that never recorded
        // depth were the same nil, so a scan whose laser files had all been
        // lost was indistinguishable from a scan with no laser.
        var keyframesDepthUnreadable = 0
        var keyframesNoDepthRecorded = 0
        var firstUnreadableDepthPath: String?

        // The cell counts, declared out here rather than at the bottom of the
        // function so the `defer` below can copy them out on EVERY exit. When
        // they lived only on the success path, a carve that filled millions of
        // cells and then failed to WRITE them reported "0 cells" - a number
        // the code never measured, and exactly the sort of false zero this
        // census exists to prevent.
        var cellsRecorded = 0
        var emptyCells = 0
        var surfaceCells = 0
        var unknownCells = 0

        // Copied out on EVERY exit, including a corrupt sidecar throwing
        // halfway through. A carve that died at keyframe 40 of 300 should say
        // so; reporting zeros for it would look exactly like a carve that
        // found nothing, which is a different problem with a different fix.
        // On the normal path this runs after the block at the end of the
        // function and simply rewrites the same values.
        defer {
            census.keyframesWithDepthLoaded = keyframesLoaded
            census.keyframesDepthMissing = keyframesMissing
            // The data-loss half of that total, in its own slot. Written here
            // in the `defer` with everything else, so a carve that threw part
            // way still reports the split it had reached rather than nothing.
            census.keyframesDepthUnreadable = keyframesDepthUnreadable
            census.raysCast = raysCast
            census.raysWithReturnInRange = raysWithReturnInRange
            census.raysBeyondMaxRange = raysBeyondMaxRange
            census.raysTooClose = raysTooClose
            census.raysNoReturn = raysNoReturn
            census.raysNoReturnBounded = raysNoReturnBounded
            census.raysNoReturnUnbounded = raysNoReturn - raysNoReturnBounded
            census.cellsRecorded = cellsRecorded
            census.emptyCells = emptyCells
            census.surfaceCells = surfaceCells
            census.unknownCellsInBounds = unknownCells
            census.hitCellCap = hitCellCap
            lastCensus = census
        }

        let stride = Swift.max(tuning.raySubsampleStride, 1)
        // A ray can only cross so many cells before it has left the scene.
        //
        // `maxRange` is `bundle.settings.lidarMaxRangeMeters`, read straight
        // off disk with no validation, and this was the ONE reader in the
        // module that did not clamp it (the bundle adjuster, the glass
        // detector, the initial splats and the survey all write
        // `Swift.max(..., 0.5)`). `Int(someFloat)` is a trapping conversion:
        // a NaN or a finite-but-absurd range from a half-written bundle killed
        // the app here rather than being carved badly. The upper clamp also
        // stops a corrupt range asking for a ray march of a billion steps.
        // Types written out rather than inferred: a ternary whose branches are
        // a clamped Float and a bare literal is the shape the type checker has
        // given up on in this project before, and CI is the only compiler.
        let clampedRange: Float = maxRange.isFinite
            ? Swift.min(Swift.max(maxRange, 0.5), 50)
            : 5
        let stepsNeeded: Float = (clampedRange / Swift.max(voxel, 0.001)).rounded(.up)
        let boundedSteps: Float = stepsNeeded.isFinite ? Swift.min(stepsNeeded, 100_000) : 100
        let maxSteps = Int(boundedSteps) + 4

        for frame in keyframes {
            try Task.checkCancellation()
            // `loadOutcome` rather than `load`: `load` answers nil for both
            // "this frame never recorded depth" and "this frame's depth file
            // will not open", and only the second is data loss. Skipped in
            // silence before the census existed, and indistinguishable from
            // each other before this split.
            let depthFrame: PrePassDepthFrame
            switch try PrePassDepthFrame.loadOutcome(
                frame: frame, settings: bundle.settings, at: ref
            ) {
            case .loaded(let opened):
                depthFrame = opened
            case .unreadable(let path):
                keyframesDepthUnreadable += 1
                keyframesMissing += 1
                if firstUnreadableDepthPath == nil { firstUnreadableDepthPath = path }
                continue
            case .noDepthRecorded:
                // `keyframes(from:)` only ever selects frames whose
                // `depthPath` is non-nil, so this should be unreachable.
                // Counted anyway, and reported below, so that if that filter
                // ever changes the fact shows up as a number instead of
                // quietly inflating the unreadable count.
                keyframesNoDepthRecorded += 1
                keyframesMissing += 1
                continue
            }
            keyframesLoaded += 1

            let pose = frame.refinedPose ?? frame.rawPose
            let sensorOrigin = pose.center.simd
            let width = geometry.width
            let height = geometry.height

            var y = 0
            while y < height {
                var x = 0
                while x < width {
                    let index = y * width + x
                    let directionWorld = PrePassRigid.worldDirection(
                        cameraDirection: geometry.rayDirections[index], pose: pose
                    )
                    raysCast += 1

                    if depthFrame.hasReturn(at: index) {
                        let z = depthFrame.depthMeters(at: index)
                        let range = geometry.range(index: index, depthMeters: z)
                        if range > maxRange {
                            raysBeyondMaxRange += 1
                        } else if range <= 0.05 {
                            raysTooClose += 1
                        }
                        if range > 0.05, range <= maxRange {
                            raysWithReturnInRange += 1
                            // Free space up to just short of the surface, then
                            // the surface cell itself.
                            let free = Swift.max(range - tuning.surfaceMarginMeters, 0)
                            carveFree(
                                origin: sensorOrigin, direction: directionWorld, distance: free,
                                voxelFrame: voxelFrame, maxSteps: maxSteps,
                                hash: &hash, states: &states, hits: &hits
                            )
                            let endpoint = sensorOrigin + directionWorld * range
                            markSurface(
                                at: endpoint, voxelFrame: voxelFrame,
                                hash: &hash, states: &states, hits: &hits
                            )
                        }
                        // range > maxRange: the beam came back from beyond the
                        // sensor's specified reach. Neither the surface nor the
                        // space in front of it is trustworthy, so nothing is
                        // written. UNKNOWN.
                    } else {
                        // No return. NOT empty. The only thing this ray proves
                        // is that it reached at least as far as its neighbours
                        // did before something stopped them - which for a
                        // window pane in a wall is the wall's own distance.
                        raysNoReturn += 1
                        if let bound = noReturnFreeBound(
                            depthFrame: depthFrame, geometry: geometry,
                            x: x, y: y, maxRange: maxRange
                        ) {
                            raysNoReturnBounded += 1
                            carveFree(
                                origin: sensorOrigin, direction: directionWorld, distance: bound,
                                voxelFrame: voxelFrame, maxSteps: maxSteps,
                                hash: &hash, states: &states, hits: &hits
                            )
                        }
                    }
                    x += stride
                }
                y += stride
            }
        }

        // Sort by key: the file is a sorted run so a reader can binary-search
        // it and two grids can be merged with a linear scan.
        let entries = hash.entries().sorted { $0.key < $1.key }
        var keys = [UInt64](repeating: 0, count: entries.count)
        var sortedStates = [UInt8](repeating: 0, count: entries.count)
        var sortedHits = [UInt16](repeating: 0, count: entries.count)
        var emptyCount = 0
        var surfaceCount = 0
        for (i, entry) in entries.enumerated() {
            keys[i] = entry.key
            sortedStates[i] = states[entry.slot]
            sortedHits[i] = hits[entry.slot]
            if sortedStates[i] == OccupancyState.empty.rawValue { emptyCount += 1 }
            if sortedStates[i] == OccupancyState.surface.rawValue { surfaceCount += 1 }
        }

        // --- The census numbers, written into the locals the `defer` copies
        //     out. Set BEFORE anything below can throw, so a failed write or a
        //     refused carve still reports what this run actually measured.
        cellsRecorded = entries.count
        emptyCells = emptyCount
        surfaceCells = surfaceCount
        // Everything inside the grid's own box that no ray ever reached. Not
        // stored anywhere (the file is sparse and absence means unknown), so
        // it has to be counted here or it cannot be known at all. Computed in
        // Double because a large room at a small voxel overflows Int32 easily.
        let cellsAcross = Double(Swift.max(size.x, 0)) / Double(voxel)
        let cellsUp = Double(Swift.max(size.y, 0)) / Double(voxel)
        let cellsDeep = Double(Swift.max(size.z, 0)) / Double(voxel)
        let cellsInBounds = cellsAcross * cellsUp * cellsDeep
        if cellsInBounds.isFinite, cellsInBounds > Double(entries.count) {
            unknownCells = Int(
                Swift.min(cellsInBounds - Double(entries.count), Double(Int.max / 2))
            )
        }

        // The depth-read split, said out loud whether or not anything else
        // went wrong. `keyframesDepthMissing` in the census is the sum of
        // these two, and the two mean completely different things.
        if keyframesDepthUnreadable > 0 || keyframesNoDepthRecorded > 0 {
            let lost = "\(keyframesDepthUnreadable) depth files would not open"
            let never = "\(keyframesNoDepthRecorded) frames recorded no depth"
            let read = "\(keyframesLoaded) of \(keyframes.count) keyframes read"
            let example = firstUnreadableDepthPath ?? "none"
            // Assembled by interpolation rather than a chain of `+`: a long
            // concatenation is the classic way to make the Swift type checker
            // give up on a file.
            let line = "Carving depth reads: \(read), \(lost), \(never), first unreadable: \(example)"
            log.error("\(line, privacy: .public)")
        }

        let data = PrePassOccupancyFile.encode(keys: keys, states: sortedStates, hits: sortedHits)
        // Written before the refusal below on purpose: whatever this run
        // carved is what should be on disk, so a stale grid from an earlier
        // run can never be mistaken for this one's work.
        try PrePassBinary.write(data, to: ref.url(forRelativePath: PrePassPaths.occupancy))

        // --- A CARVE THAT PROVED NO AIR EMPTY IS NOT A RESULT.
        //
        // `emptyCellCount` is the entire product of this stage. The only thing
        // in the app that consumes an occupancy grid is
        // `certifiedEmptyIndices`, and with no empty cells that returns an
        // empty list to every question for the rest of the run. Returning a
        // structurally valid `OccupancyGridRef` here would have the pipeline
        // store it as a successful occupancy result and the trainer install a
        // carver that deletes nothing, for ever, in silence - which is what
        // used to happen, right down to a zero-byte file when nothing at all
        // was recorded.
        //
        // Throwing instead means the pipeline's own error boundary catches it,
        // adds the `carving_failed` finding to the QC card, and leaves
        // `result.occupancy` nil. Nothing crashes, nothing is stored, and the
        // numbers are already in the census by way of the `defer` above, where
        // `census_no_empty_cells` and `census_no_surface_cells` turn them into
        // alarms on the screen the owner already looks at.
        if emptyCount == 0 {
            let cells = "\(entries.count) cells recorded"
            let split = "\(emptyCount) empty, \(surfaceCount) surface"
            let rays = "\(raysCast) rays cast, \(raysWithReturnInRange) hit something in range"
            // Named `framesText` because `frames` is the whole sorted frame
            // list in this scope and shadowing it here would read as a bug.
            let framesText = "\(keyframesLoaded) of \(keyframes.count) keyframes read"
            let reason = "the free-space carve proved no air empty: "
                + "\(cells) (\(split)), \(rays), \(framesText)"
            log.error("\(reason, privacy: .public)")
            throw NimbusError.prePassFailed(reason)
        }

        let grid = OccupancyGridRef(
            path: PrePassPaths.occupancy,
            voxelSizeMeters: voxel,
            origin: Vector3(voxelFrame.origin),
            cellCount: entries.count,
            emptyCellCount: emptyCount,
            surfaceCellCount: surfaceCount
        )

        // Keep it loaded: the caller almost always wants to query it straight
        // away, and re-reading a file we just wrote would be silly.
        loadedKeys = keys
        loadedStates = sortedStates
        loadedFrame = voxelFrame

        // Every census number is now set on the locals the `defer` copies out,
        // so there is nothing left to fill in here. `lastCensus` is written
        // once, by that `defer`, on this exit and on every other one.
        return grid
    }

    // MARK: Loading and querying

    public func load(_ grid: OccupancyGridRef, at ref: CaptureBundleRef) async throws {
        let url = ref.url(forRelativePath: grid.path)
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            throw NimbusError.malformedData("the occupancy grid \(grid.path) could not be read")
        }
        guard let decoded = PrePassOccupancyFile.decode(data) else {
            throw NimbusError.malformedData(
                "the occupancy grid \(grid.path) is \(data.count) bytes, which is not a whole "
                    + "number of \(PrePassOccupancyFile.recordSize)-byte records"
            )
        }
        // --- THE REFERENCE IS CHECKED AGAINST THE FILE.
        //
        // `cellCount`, `emptyCellCount` and `surfaceCellCount` were written by
        // `carve` and read by nothing in the entire app, which is why a grid
        // truncated to nothing used to load "successfully" as a grid that
        // answers UNKNOWN to every question - and `certifiedEmptyIndices` then
        // returned an empty list for the whole training run without one line
        // of evidence that anything was wrong. Both halves are checked here,
        // which is also what finally makes those three fields load-bearing.
        guard decoded.keys.count == grid.cellCount else {
            throw NimbusError.malformedData(
                "the occupancy grid \(grid.path) holds \(decoded.keys.count) cells "
                    + "but the scan recorded \(grid.cellCount)"
            )
        }
        guard grid.emptyCellCount > 0, !decoded.keys.isEmpty else {
            throw NimbusError.malformedData(
                "the occupancy grid \(grid.path) proves no air empty "
                    + "(\(decoded.keys.count) cells, \(grid.emptyCellCount) of them empty), "
                    + "so nothing could be cleaned up from it"
            )
        }
        loadedKeys = decoded.keys
        loadedStates = decoded.states
        loadedFrame = PrePassVoxelFrame(
            origin: grid.origin.simd, voxelSize: grid.voxelSizeMeters
        )
        actualVoxelSizeMeters = grid.voxelSizeMeters
    }

    /// `.unknown` when the grid is not loaded. Never a guess: a caller that
    /// deletes Gaussians on this answer must get "I do not know" rather than a
    /// plausible default, or a missing file becomes silent data loss.
    public func state(atWorldPoint point: Vector3) -> OccupancyState {
        guard let frame = loadedFrame, !loadedKeys.isEmpty else { return .unknown }
        guard let key = frame.key(point.simd) else { return .unknown }
        guard let index = binarySearch(key) else { return .unknown }
        return OccupancyState(rawValue: loadedStates[index]) ?? .unknown
    }

    public func certifiedEmptyIndices(centers: [SIMD3<Float>]) -> [Int] {
        guard let frame = loadedFrame, !loadedKeys.isEmpty else { return [] }
        var result: [Int] = []
        for (i, centre) in centers.enumerated() {
            guard let key = frame.key(centre), let index = binarySearch(key) else { continue }
            if loadedStates[index] == OccupancyState.empty.rawValue { result.append(i) }
        }
        return result
    }

    private func binarySearch(_ key: UInt64) -> Int? {
        var low = 0
        var high = loadedKeys.count - 1
        while low <= high {
            let mid = (low + high) / 2
            let value = loadedKeys[mid]
            if value == key { return mid }
            if value < key { low = mid + 1 } else { high = mid - 1 }
        }
        return nil
    }

    // MARK: Internals

    private func carveFree(
        origin: SIMD3<Float>,
        direction: SIMD3<Float>,
        distance: Float,
        voxelFrame: PrePassVoxelFrame,
        maxSteps: Int,
        hash: inout PrePassVoxelHash,
        states: inout [UInt8],
        hits: inout [UInt16]
    ) {
        guard distance > voxelFrame.voxelSize else { return }
        var capped = false
        PrePassRayWalk.traverse(
            origin: origin, direction: direction, distance: distance,
            frame: voxelFrame, maxSteps: maxSteps
        ) { cell in
            guard let key = PrePassMorton.key(cell: cell) else { return false }
            if let existing = hash.index(of: key) {
                // Already known. A SURFACE cell is NEVER downgraded to empty -
                // one stray beam clipping a corner must not erase a wall that
                // thousands of beams landed on - and an EMPTY cell is already
                // what this beam would make it. Either way, nothing to write.
                _ = existing
                return true
            }
            guard hash.count < self.tuning.maxCells else { capped = true; return false }
            let inserted = hash.indexOrInsert(key)
            if inserted.inserted {
                states.append(OccupancyState.empty.rawValue)
                hits.append(0)
            }
            return true
        }
        if capped { hitCellCap = true }
    }

    private func markSurface(
        at point: SIMD3<Float>,
        voxelFrame: PrePassVoxelFrame,
        hash: inout PrePassVoxelHash,
        states: inout [UInt8],
        hits: inout [UInt16]
    ) {
        guard let key = voxelFrame.key(point) else { return }
        let slot = hash.indexOrInsert(key)
        if slot.inserted {
            states.append(OccupancyState.surface.rawValue)
            hits.append(1)
        } else {
            states[slot.index] = OccupancyState.surface.rawValue
            // Saturating: a wall seen 70000 times is not usefully different
            // from one seen 65535 times, and wrapping to zero would say the
            // opposite of the truth.
            if hits[slot.index] < UInt16.max { hits[slot.index] += 1 }
        }
    }

    /// How far a no-return ray is certified free, from the depths its
    /// neighbours came back with. `nil` when the whole neighbourhood is silent
    /// too, which is the honest answer for a ray pointing at the sky.
    private func noReturnFreeBound(
        depthFrame: PrePassDepthFrame,
        geometry: PrePassDepthGeometry,
        x: Int,
        y: Int,
        maxRange: Float
    ) -> Float? {
        let radius = tuning.noReturnNeighbourhoodRadius
        var nearest = Float.greatestFiniteMagnitude
        var found = false
        let width = geometry.width
        let height = geometry.height

        var dy = -radius
        while dy <= radius {
            let ny = y + dy
            if ny >= 0 && ny < height {
                var dx = -radius
                while dx <= radius {
                    let nx = x + dx
                    if nx >= 0 && nx < width {
                        let index = ny * width + nx
                        if depthFrame.hasReturn(at: index) {
                            let z = depthFrame.depthMeters(at: index)
                            let range = geometry.range(index: index, depthMeters: z)
                            if range > 0.05, range <= maxRange, range < nearest {
                                nearest = range
                                found = true
                            }
                        }
                    }
                    dx += 1
                }
            }
            dy += 1
        }

        guard found, nearest.isFinite else { return nil }
        // The NEAREST neighbour's range, minus a margin. Nearest rather than
        // median because this is a lower bound and it has to hold for the
        // worst case in the neighbourhood: a window frame in the corner of the
        // patch must not license carving through the pane's own plane.
        let bound = nearest - tuning.noReturnMarginMeters
        // Under a voxel of certified free space is not worth a traversal, and
        // a negative bound would mean the neighbours are closer than the
        // margin - in which case this ray proves nothing at all.
        return bound > 0.10 ? bound : nil
    }

    private func keyframes(from frames: [CaptureFrame]) -> [CaptureFrame] {
        var chosen: [CaptureFrame] = []
        var lastCentre = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var lastTime = -Double.greatestFiniteMagnitude
        for frame in frames where frame.depthPath != nil {
            // A frame captured while tracking was lost has a pose that would
            // carve free space through a wall. Down-weighting is the right
            // answer for photometry; for carving, which is a hard geometric
            // claim, the right answer is to skip it.
            guard frame.qc.trackingQuality != .notAvailable else { continue }
            let centre = frame.rawPose.center.simd
            let movedFar = simd_distance(centre, lastCentre) >= tuning.keyframeSpacingMeters
            let waitedLong = frame.timestampSeconds - lastTime >= tuning.keyframeSpacingSeconds
            guard chosen.isEmpty || movedFar || waitedLong else { continue }
            chosen.append(frame)
            lastCentre = centre
            lastTime = frame.timestampSeconds
        }
        return chosen
    }

    private func sceneBounds(bundle: CaptureBundle, frames: [CaptureFrame]) -> BoundingBox {
        if let bounds = bundle.sceneBounds {
            let size = bounds.sizeMeters
            if size.x > 0.2 && size.y > 0.2 && size.z > 0.2 { return bounds }
        }
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for frame in frames {
            let centre = frame.rawPose.center.simd
            minimum = simd_min(minimum, centre)
            maximum = simd_max(maximum, centre)
        }
        guard minimum.x <= maximum.x else {
            return BoundingBox(min: Vector3(-1, -1, -1), max: Vector3(1, 1, 1))
        }
        // The camera path plus the sensor's reach in every direction: the
        // measured envelope of what could possibly have been seen, rather than
        // a number somebody picked.
        let reach = SIMD3<Float>(repeating: bundle.settings.lidarMaxRangeMeters)
        return BoundingBox(min: Vector3(minimum - reach), max: Vector3(maximum + reach))
    }
}

private extension Double {
    /// Named `cubeRoot` rather than `cbrt` so it cannot shadow, or be confused
    /// with, the free function it calls.
    func cubeRoot() -> Double { Foundation.cbrt(self) }
}
