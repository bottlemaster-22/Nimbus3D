//
//  Contracts.swift
//  Core
//
//  THE SHARED VOCABULARY OF THE WHOLE APP.
//
//  Every module talks to every other module through the types and protocols
//  in this file. Nothing here imports ARKit or Metal: Core is the bottom of
//  the stack. The one exception is the `NimbusUI` screen registry at the very
//  end, which needs SwiftUI's `AnyView` so a module can hand the app shell a
//  screen without the shell having to know that module's type names.
//
//  ---------------------------------------------------------------------------
//  READ THIS BEFORE ADDING A TYPE
//  ---------------------------------------------------------------------------
//  The app is ONE Xcode target. "Module" here means "a directory under
//  ios/Sources with one owning agent", not a Swift module. There is therefore
//  no `import Core`, no access-control wall, and - the part that bites - no
//  namespacing: two files anywhere in the project may not declare the same
//  top-level type name, or the build fails with a duplicate-symbol error.
//
//  Consequences, all of them deliberate:
//
//   1. A type declared here must not already exist elsewhere in ios/Sources.
//   2. Two modules (Export and Booster) were built BEFORE this file existed
//      and had to invent their own contracts. Those types are not re-declared
//      here. They are PROMOTED: the file that already declares them stays
//      where it is and is treated as part of the contract surface. Each one is
//      marked "PROMOTED" below with its owning file. CONTRACTS.md lists the
//      (small) reconciliation the integrator may optionally do later; none of
//      it is required for the build to work.
//   3. Where a name this file wanted was already taken by a concrete class,
//      the protocol here got a different name rather than forcing a rename in
//      shipped, working code. Those two cases are `SplatExporting` (because
//      Export already ships a concrete `ExportService`) and `BoosterService`
//      (because Booster already ships a concrete `BoosterClient`).
//
//  ---------------------------------------------------------------------------
//  GEOMETRY CONVENTIONS - the single most expensive thing to get wrong
//  ---------------------------------------------------------------------------
//  World frame:   right-handed, Y vertical (up), metres. This is ARKit's
//                 gravity-aligned world frame, used unchanged.
//
//  Camera frame:  +X right, +Y DOWN, +Z FORWARD (into the scene). This is the
//                 COLMAP / OpenCV convention, NOT ARKit's (+Y up, -Z forward).
//                 `Pose.fromARKitCameraTransform(_:)` does the conversion and
//                 is the only place it may be done.
//
//  A `Pose` is world -> camera:      X_cam = R * X_world + t
//  so the camera centre in world space is  C = -R^T * t  (`Pose.center`).
//
//  Projection:    u = fx * (Xc / Zc) + cx
//                 v = fy * (Yc / Zc) + cy      with Zc > 0 in front.
//
//  Splat storage frame: RUB (X right, Y up, Z back) - see the PROMOTED
//  `SplatCloud` in Sources/Export/SplatCloud.swift. That is the world frame
//  with no rotation applied, so trainer output needs no conversion.
//

import Foundation
import simd

// MARK: - Identifiers

/// A scan's folder name under `Documents/<brand>/Scans`, and its identity
/// everywhere else. Format: `scan_YYYYMMDD_HHMMSS` (see docs/DATA_FORMAT.md).
///
/// Deliberately a `String` typealias rather than a wrapper struct: the Export
/// and Booster modules already pass `scanID: String` through every API, and a
/// wrapper would be a rename with no safety payoff at this size.
public typealias ScanID = String

/// Zero-based index of a frame within a capture, in capture order. Stable for
/// the life of the scan; it is the join key between `images.txt`, the frame
/// sidecar log, and every per-frame array in this file.
public typealias FrameID = UInt32

/// Zero-based index of a time-sliced submap within a capture (F1).
public typealias SubmapID = UInt32

// MARK: - Small geometry value types
//
// Codable stand-ins for the simd types, because `SIMD3<Float>` and friends do
// not encode to anything a Python trainer would enjoy reading. Everything
// bridges to simd in both directions with no allocation.

/// A 3-vector that survives a round trip through JSON as `[x, y, z]`.
public struct Vector3: Codable, Hashable, Sendable {
    public var x: Float
    public var y: Float
    public var z: Float

    public init(_ x: Float, _ y: Float, _ z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }

    public init(_ v: SIMD3<Float>) { self.init(v.x, v.y, v.z) }

    public var simd: SIMD3<Float> { SIMD3(x, y, z) }

    public static let zero = Vector3(0, 0, 0)

    // Encoded as a bare array so a 200 MB pose log is not 60% key names.
    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        x = try c.decode(Float.self)
        y = try c.decode(Float.self)
        z = try c.decode(Float.self)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(x)
        try c.encode(y)
        try c.encode(z)
    }
}

/// A unit quaternion in `(x, y, z, w)` component order, encoded as
/// `[x, y, z, w]`.
///
/// Note the ordering trap: this is simd's and glTF's order. COLMAP's
/// `images.txt` writes `QW QX QY QZ`, and the PLY splat convention writes
/// `rot_0..rot_3` as `(w, x, y, z)`. Both of those re-orderings happen inside
/// the writer that needs them, never in memory.
public struct Quaternion: Codable, Hashable, Sendable {
    public var x: Float
    public var y: Float
    public var z: Float
    public var w: Float

    public init(x: Float, y: Float, z: Float, w: Float) {
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    }

    public init(_ q: simd_quatf) {
        self.init(x: q.vector.x, y: q.vector.y, z: q.vector.z, w: q.vector.w)
    }

    public var simd: simd_quatf { simd_quatf(ix: x, iy: y, iz: z, r: w) }

    public static let identity = Quaternion(x: 0, y: 0, z: 0, w: 1)

    public var normalized: Quaternion { Quaternion(simd.normalized) }

    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        x = try c.decode(Float.self)
        y = try c.decode(Float.self)
        z = try c.decode(Float.self)
        w = try c.decode(Float.self)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(x)
        try c.encode(y)
        try c.encode(z)
        try c.encode(w)
    }
}

/// A rigid world -> camera transform: `X_cam = R * X_world + t`.
public struct Pose: Codable, Hashable, Sendable {
    /// Rotation R, world -> camera.
    public var rotation: Quaternion
    /// Translation t, in metres, world -> camera.
    public var translation: Vector3

    public init(rotation: Quaternion, translation: Vector3) {
        self.rotation = rotation
        self.translation = translation
    }

    public static let identity = Pose(rotation: .identity, translation: .zero)

    /// The camera's optical centre in world space, `C = -R^T * t`.
    public var center: Vector3 {
        Vector3(-(rotation.simd.inverse.act(translation.simd)))
    }

    /// The camera's viewing direction in world space (camera +Z, forward).
    public var forward: Vector3 {
        Vector3(rotation.simd.inverse.act(SIMD3<Float>(0, 0, 1)))
    }

    /// Full 4x4 world -> camera matrix.
    public var matrix: simd_float4x4 {
        var m = simd_float4x4(rotation.simd)
        m.columns.3 = SIMD4<Float>(translation.simd, 1)
        return m
    }

    /// Converts an `ARCamera.transform` (camera -> world, ARKit convention:
    /// +X right, +Y up, -Z forward) into this app's world -> camera pose in
    /// the COLMAP/OpenCV convention (+Y down, +Z forward).
    ///
    /// The flip is a 180 degree rotation about the camera's X axis, applied on
    /// the camera side, then inverted:
    ///
    ///     T_world_cvcam = T_world_glcam * diag(1, -1, -1, 1)
    ///     pose          = inverse(T_world_cvcam)
    ///
    /// This is THE conversion. Nowhere else in the app may negate an axis.
    public static func fromARKitCameraTransform(_ transform: simd_float4x4) -> Pose {
        let flip = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))
        let worldToCamera = (transform * flip).inverse
        let rotation = simd_quatf(
            simd_float3x3(
                simd_make_float3(worldToCamera.columns.0),
                simd_make_float3(worldToCamera.columns.1),
                simd_make_float3(worldToCamera.columns.2)
            )
        )
        return Pose(
            rotation: Quaternion(rotation.normalized),
            translation: Vector3(simd_make_float3(worldToCamera.columns.3))
        )
    }

    /// Shortest-arc interpolation between two poses. Used when re-sampling
    /// poses at a shifted timestamp during camera-to-IMU time-offset
    /// calibration (F1) - rotation slerps, translation lerps.
    public static func interpolate(_ a: Pose, _ b: Pose, t: Float) -> Pose {
        Pose(
            rotation: Quaternion(simd_slerp(a.rotation.simd, b.rotation.simd, t)),
            translation: Vector3(
                a.translation.simd + (b.translation.simd - a.translation.simd) * t
            )
        )
    }
}

/// An axis-aligned box in world space, metres.
public struct BoundingBox: Codable, Hashable, Sendable {
    public var min: Vector3
    public var max: Vector3

    public init(min: Vector3, max: Vector3) {
        self.min = min
        self.max = max
    }

    public var sizeMeters: Vector3 {
        Vector3(max.x - min.x, max.y - min.y, max.z - min.z)
    }

    /// Longest edge, the cheap "how big is this scene" number that
    /// `TrainingBudget.recommended(for:sceneExtentMeters:)` scales on.
    public var longestEdgeMeters: Float {
        let s = sizeMeters
        return Swift.max(s.x, Swift.max(s.y, s.z))
    }
}

/// A single shared PINHOLE camera model, as written to `sparse/0/cameras.txt`.
/// One per capture: the RGB stream is a fixed format for the whole session, so
/// there is exactly one camera and every image references it.
public struct CameraIntrinsics: Codable, Hashable, Sendable {
    /// Pixel width of the RGB frames these intrinsics describe.
    public var width: Int
    /// Pixel height of the RGB frames these intrinsics describe.
    public var height: Int
    public var fx: Float
    public var fy: Float
    public var cx: Float
    public var cy: Float

    public init(width: Int, height: Int, fx: Float, fy: Float, cx: Float, cy: Float) {
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    }

    /// Rescales for a different render resolution (the trainer runs at
    /// 480-720 px, the frames are captured at 1920).
    public func scaled(toWidth newWidth: Int, height newHeight: Int) -> CameraIntrinsics {
        let sx = Float(newWidth) / Float(width)
        let sy = Float(newHeight) / Float(height)
        return CameraIntrinsics(
            width: newWidth,
            height: newHeight,
            fx: fx * sx,
            fy: fy * sy,
            cx: cx * sx,
            cy: cy * sy
        )
    }

    /// Horizontal field of view in degrees. Used by the preview camera path,
    /// which deliberately widens it (F9).
    public var horizontalFOVDegrees: Float {
        2 * atan(Float(width) / (2 * fx)) * 180 / .pi
    }
}

// MARK: - JSON coding

/// The one JSON configuration every sidecar in `docs/DATA_FORMAT.md` uses.
///
/// ISO-8601 dates, matching what `Sources/Booster/BoosterHTTP.swift` already
/// puts on the wire, so a `Date` means the same thing on disk and over the
/// LAN. Sorted keys so two runs over identical data produce identical bytes,
/// which makes a diff meaningful and a checksum stable.
public enum ContractsJSON {
    public static func encoder(prettyPrinted: Bool = true) -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting =
            prettyPrinted ? [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
                          : [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

// MARK: - Capture: per-frame records

/// ARKit's own tracking verdict for a frame, flattened to something Codable.
/// A frame captured while tracking was limited is still written out - it is
/// down-weighted, never silently dropped, because throwing data away is how
/// you get a hole in the scan you cannot explain to the user.
public enum TrackingQuality: String, Codable, Sendable {
    case notAvailable
    case limitedInitializing
    case limitedExcessiveMotion
    case limitedInsufficientFeatures
    case limitedRelocalizing
    case normal

    /// Whether poses from this frame can be trusted as a pose-graph
    /// constraint (as opposed to merely being usable pixels).
    public var isPoseTrustworthy: Bool { self == .normal }
}

/// Why a frame's exposure differs from its neighbours (F5 bracketing).
public enum ExposureBracket: String, Codable, Sendable {
    /// Auto-exposed frame, the overwhelming majority.
    case normal
    /// Deliberately under-exposed (~3 stops) so a bright window or sky has
    /// unsaturated pixels somewhere in the dataset.
    case darker
}

/// Per-frame quality inputs and the single weight derived from them (F8).
/// Every field is kept, not just the weight, so the QC card can tell the user
/// *which* problem cost them coverage rather than just "low quality".
public struct FrameQC: Codable, Hashable, Sendable {
    /// Gyro magnitude at exposure midpoint, radians/second.
    public var angularSpeedRadPerSec: Float
    /// The live blur meter, in pixels of smear (F9):
    /// `degreesPerSecond * exposureDuration / 0.0426`, where 0.0426 deg/px is
    /// the angular pixel pitch of the wide camera at 1920 px. Amber at 2 px,
    /// red at 4 px.
    public var motionBlurPixels: Float
    /// Variance-of-Laplacian sharpness, normalised 0...1 against the running
    /// maximum for this session. Higher is sharper.
    public var sharpness: Float
    /// Fraction of the 256x192 native depth map with usable returns, 0...1.
    public var depthValidFraction: Float
    /// Absolute exposure change from the previous frame in EV. A jump means
    /// auto-exposure moved mid-walk and photometric losses should not treat
    /// the two frames as directly comparable.
    public var exposureJumpEV: Float
    public var trackingQuality: TrackingQuality
    /// The product of the above, clamped to 0...1. This is the number the
    /// trainer multiplies its per-frame loss by. 0 does not mean "discard";
    /// it means "contributes nothing photometric".
    public var weight: Float

    public init(
        angularSpeedRadPerSec: Float,
        motionBlurPixels: Float,
        sharpness: Float,
        depthValidFraction: Float,
        exposureJumpEV: Float,
        trackingQuality: TrackingQuality,
        weight: Float
    ) {
        self.angularSpeedRadPerSec = angularSpeedRadPerSec
        self.motionBlurPixels = motionBlurPixels
        self.sharpness = sharpness
        self.depthValidFraction = depthValidFraction
        self.exposureJumpEV = exposureJumpEV
        self.trackingQuality = trackingQuality
        self.weight = weight
    }
}

/// Everything recorded for one captured frame.
///
/// Paths are POSIX, forward-slash, relative to the scan folder root, so a
/// bundle can be zipped, sent to the Booster, and unzipped anywhere without a
/// single absolute path to rewrite. `docs/DATA_FORMAT.md` fixes the filenames.
public struct CaptureFrame: Codable, Hashable, Sendable, Identifiable {
    public var id: FrameID { index }

    public var index: FrameID
    /// ARKit frame timestamp, seconds on the device's mach-continuous clock.
    /// The same clock the gyro samples are stamped with, which is what makes
    /// the camera-to-IMU offset in `CaptureBundle` meaningful.
    public var timestampSeconds: Double

    /// e.g. `images/frame_20260903_141205_512.jpg`
    public var imagePath: String
    /// Native 256x192 LiDAR depth, UInt16 millimetres, row major.
    /// e.g. `sensor_data/depth/frame_20260903_141205_512.depth16`
    /// This is `ARFrame.sceneDepth.depthMap` at its true resolution - never
    /// the smoothed or upsampled variant.
    public var depthPath: String?
    /// Native 256x192 ARKit confidence, UInt8, 0 low / 1 medium / 2 high.
    /// e.g. `sensor_data/confidence/frame_20260903_141205_512.conf8`
    public var confidencePath: String?

    /// Raw ARKit VIO pose, already converted to this app's camera convention.
    /// Written to `sparse/0/images.txt`. Never overwritten - the raw track is
    /// evidence, and the refinement has to be auditable against it.
    public var rawPose: Pose
    /// Pose after submap pose-graph optimisation and time-offset calibration
    /// (F1). `nil` until the pre-pass has run. Written to
    /// `prepass/sparse_refined/images.txt`.
    public var refinedPose: Pose?

    /// `ARCamera.exposureDuration`, seconds.
    public var exposureDurationSeconds: Double
    /// `ARCamera.exposureOffset`, EV.
    public var exposureOffsetEV: Float
    /// ISO, when obtainable. ARKit does not expose it on `ARCamera`; it is
    /// readable from the frame's `exifData` on some devices, so this is
    /// optional by design rather than by omission.
    public var iso: Float?

    /// Gyro angular velocity at this frame's exposure midpoint, radians/second
    /// in the device body frame.
    public var angularVelocity: Vector3
    public var qc: FrameQC
    public var bracket: ExposureBracket
    /// Which time-sliced submap owns this frame (F1). `nil` before the
    /// pre-pass assigns submaps.
    public var submap: SubmapID?

    public init(
        index: FrameID,
        timestampSeconds: Double,
        imagePath: String,
        depthPath: String?,
        confidencePath: String?,
        rawPose: Pose,
        refinedPose: Pose? = nil,
        exposureDurationSeconds: Double,
        exposureOffsetEV: Float,
        iso: Float? = nil,
        angularVelocity: Vector3,
        qc: FrameQC,
        bracket: ExposureBracket = .normal,
        submap: SubmapID? = nil
    ) {
        self.index = index
        self.timestampSeconds = timestampSeconds
        self.imagePath = imagePath
        self.depthPath = depthPath
        self.confidencePath = confidencePath
        self.rawPose = rawPose
        self.refinedPose = refinedPose
        self.exposureDurationSeconds = exposureDurationSeconds
        self.exposureOffsetEV = exposureOffsetEV
        self.iso = iso
        self.angularVelocity = angularVelocity
        self.qc = qc
        self.bracket = bracket
        self.submap = submap
    }
}

/// ARKit's mesh face classification, plus the two extras the glass detector
/// (F5) contributes. Values are stable on the wire; do not renumber.
public enum SurfaceClass: String, Codable, Sendable, CaseIterable {
    case none
    case wall
    case floor
    case ceiling
    case table
    case seat
    case window
    case door
    /// Detected by us, not by ARKit: LiDAR-silent, image-bright, planar, and
    /// embedded in an otherwise planar wall.
    case glass
    /// Beyond LiDAR range with no parallax - routed to the background model.
    case sky

    /// Surfaces where LiDAR returns are not evidence of geometry.
    public var isOpticallyUnreliable: Bool {
        self == .window || self == .glass || self == .sky
    }
}

/// An ARKit anchor as logged during the session and again at the end.
///
/// The end-of-session re-read matters because ARKit silently moves anchors
/// when it relocalises: comparing `transform` to the value logged live is a
/// free, direct measurement of how much the map drifted (F8).
public struct AnchorRecord: Codable, Hashable, Sendable {
    public var identifier: UUID
    /// Anchor -> world transform, ARKit convention, unmodified.
    public var transform: [Float]  // 16 values, column-major
    /// Frame index this anchor was first seen at, for the live log.
    public var firstSeenFrame: FrameID?
    public var classification: SurfaceClass

    public init(
        identifier: UUID,
        transform: [Float],
        firstSeenFrame: FrameID?,
        classification: SurfaceClass
    ) {
        self.identifier = identifier
        self.transform = transform
        self.firstSeenFrame = firstSeenFrame
        self.classification = classification
    }

    public var matrix: simd_float4x4 {
        guard transform.count == 16 else { return matrix_identity_float4x4 }
        return simd_float4x4(
            SIMD4(transform[0], transform[1], transform[2], transform[3]),
            SIMD4(transform[4], transform[5], transform[6], transform[7]),
            SIMD4(transform[8], transform[9], transform[10], transform[11]),
            SIMD4(transform[12], transform[13], transform[14], transform[15])
        )
    }
}

/// One chunk of the ARKit scene mesh on disk, with its per-face classification.
public struct MeshChunkRef: Codable, Hashable, Sendable {
    /// e.g. `mesh/chunk_0007.ply`
    public var geometryPath: String
    /// e.g. `mesh/chunk_0007.cls` - one `SurfaceClass` byte per face, in the
    /// same order as the PLY's face list. Byte values are the index into
    /// `SurfaceClass.allCases`.
    public var classificationPath: String
    public var faceCount: Int
    public var bounds: BoundingBox

    public init(
        geometryPath: String,
        classificationPath: String,
        faceCount: Int,
        bounds: BoundingBox
    ) {
        self.geometryPath = geometryPath
        self.classificationPath = classificationPath
        self.faceCount = faceCount
        self.bounds = bounds
    }
}

/// How a revisit between two frames was established (F1).
public enum RevisitMethod: String, Codable, Sendable {
    /// Poses are close and view directions agree; no depth alignment run yet.
    case poseProximity
    /// Point-to-plane ICP on the two frames' native depth maps converged.
    case depthICP
}

/// A detected loop closure: two frames that look at the same surface from
/// nearly the same place at different times. These are the constraints that
/// let the submap pose graph fix drift, and their residuals are the honest
/// measurement of how much drift there was.
public struct RevisitPair: Codable, Hashable, Sendable {
    public var frameA: FrameID
    public var frameB: FrameID
    public var method: RevisitMethod
    /// Relative pose B-in-A measured by the alignment, i.e. the constraint.
    public var measuredRelativePose: Pose
    /// Residual translation error against the raw VIO poses, metres. This is
    /// the number the QC card turns into "your scan drifted N cm".
    public var translationResidualMeters: Float
    public var rotationResidualDegrees: Float
    /// ICP inlier count, or matched-keyframe count for `poseProximity`.
    public var inlierCount: Int
    /// 0...1 confidence used to weight this edge in the pose graph.
    public var confidence: Float

    public init(
        frameA: FrameID,
        frameB: FrameID,
        method: RevisitMethod,
        measuredRelativePose: Pose,
        translationResidualMeters: Float,
        rotationResidualDegrees: Float,
        inlierCount: Int,
        confidence: Float
    ) {
        self.frameA = frameA
        self.frameB = frameB
        self.method = method
        self.measuredRelativePose = measuredRelativePose
        self.translationResidualMeters = translationResidualMeters
        self.rotationResidualDegrees = rotationResidualDegrees
        self.inlierCount = inlierCount
        self.confidence = confidence
    }
}

/// Capture-time settings that a downstream stage must know about to interpret
/// the frames correctly.
public struct CaptureSettings: Codable, Hashable, Sendable {
    /// Every Nth frame is captured darker (F5). 0 = bracketing was off.
    public var bracketEveryNFrames: Int
    /// How much darker, in stops.
    public var bracketStops: Float
    /// The user locked exposure and white balance for the session (F8).
    public var exposureLocked: Bool
    public var whiteBalanceLocked: Bool
    /// Native depth map dimensions actually delivered by this device.
    /// 256x192 on every LiDAR iPhone to date, but measured, never assumed.
    public var depthWidth: Int
    public var depthHeight: Int
    /// Manufacturer's usable LiDAR range, metres. Beyond this, "no return"
    /// means "unknown", never "empty" (F2).
    public var lidarMaxRangeMeters: Float

    public init(
        bracketEveryNFrames: Int,
        bracketStops: Float,
        exposureLocked: Bool,
        whiteBalanceLocked: Bool,
        depthWidth: Int,
        depthHeight: Int,
        lidarMaxRangeMeters: Float
    ) {
        self.bracketEveryNFrames = bracketEveryNFrames
        self.bracketStops = bracketStops
        self.exposureLocked = exposureLocked
        self.whiteBalanceLocked = whiteBalanceLocked
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.lidarMaxRangeMeters = lidarMaxRangeMeters
    }
}

/// Everything one capture session produced, as an index over the files on
/// disk. Serialised to `capture_bundle.json` at the scan folder root.
///
/// This struct is the index, not the data: pixels, depth and mesh stay in
/// their own files and are referenced by relative path, so a whole-house scan
/// does not have to be resident in memory to be described.
public struct CaptureBundle: Codable, Sendable {
    /// Bumped when a field changes meaning. A reader that sees a version it
    /// does not know must refuse, not guess.
    public static let currentFormatVersion = 1

    public var formatVersion: Int
    public var scanID: ScanID
    public var createdAt: Date
    /// User-visible name; defaults to a date, editable in the library.
    public var displayName: String
    /// e.g. `iPhone17,2`. Raw identifier, not marketing name.
    public var deviceModel: String
    public var appVersion: String

    public var intrinsics: CameraIntrinsics
    public var settings: CaptureSettings
    public var frames: [CaptureFrame]

    /// Camera-to-IMU time offset in seconds, from the F1 sweep
    /// (-50...+50 ms in 5 ms steps). Add this to a camera timestamp to get the
    /// IMU timestamp that actually corresponds to it. `nil` if calibration did
    /// not converge - which is reported, not hidden.
    public var cameraToIMUTimeOffsetSeconds: Double?

    /// Anchors as logged live during the session.
    public var anchorsDuringSession: [AnchorRecord]
    /// The SAME anchors re-read once the session ended. Diffing the two is a
    /// free drift measurement (F8).
    public var anchorsAtEndOfSession: [AnchorRecord]

    public var meshChunks: [MeshChunkRef]
    public var revisitPairs: [RevisitPair]

    /// World-space extent of the LiDAR points, used for budgeting.
    public var sceneBounds: BoundingBox?
    /// Path to the baked LiDAR cloud, `sparse/0/points3D.txt`.
    public var pointCloudPath: String

    public init(
        formatVersion: Int = CaptureBundle.currentFormatVersion,
        scanID: ScanID,
        createdAt: Date,
        displayName: String,
        deviceModel: String,
        appVersion: String,
        intrinsics: CameraIntrinsics,
        settings: CaptureSettings,
        frames: [CaptureFrame],
        cameraToIMUTimeOffsetSeconds: Double? = nil,
        anchorsDuringSession: [AnchorRecord] = [],
        anchorsAtEndOfSession: [AnchorRecord] = [],
        meshChunks: [MeshChunkRef] = [],
        revisitPairs: [RevisitPair] = [],
        sceneBounds: BoundingBox? = nil,
        pointCloudPath: String
    ) {
        self.formatVersion = formatVersion
        self.scanID = scanID
        self.createdAt = createdAt
        self.displayName = displayName
        self.deviceModel = deviceModel
        self.appVersion = appVersion
        self.intrinsics = intrinsics
        self.settings = settings
        self.frames = frames
        self.cameraToIMUTimeOffsetSeconds = cameraToIMUTimeOffsetSeconds
        self.anchorsDuringSession = anchorsDuringSession
        self.anchorsAtEndOfSession = anchorsAtEndOfSession
        self.meshChunks = meshChunks
        self.revisitPairs = revisitPairs
        self.sceneBounds = sceneBounds
        self.pointCloudPath = pointCloudPath
    }
}

/// A capture bundle plus where it lives. `CaptureBundle` is deliberately
/// path-relative and portable; this pairs it with the one absolute URL needed
/// to actually open a file.
public struct CaptureBundleRef: Sendable {
    public let scanID: ScanID
    /// `Documents/<brand>/Scans/<scanID>`
    public let rootURL: URL

    public init(scanID: ScanID, rootURL: URL) {
        self.scanID = scanID
        self.rootURL = rootURL
    }

    public func url(forRelativePath path: String) -> URL {
        path.split(separator: "/").reduce(rootURL) { $0.appendingPathComponent(String($1)) }
    }
}

// MARK: - Pre-pass results

/// One time-sliced submap: a 15-30 s window of frames with 20-30% overlap into
/// its neighbours. VIO is near-perfect inside a window, so each submap is
/// treated as internally rigid and only its single SE(3) placement is
/// optimised by the pose graph (F1).
public struct Submap: Codable, Hashable, Sendable, Identifiable {
    public var id: SubmapID { index }

    public var index: SubmapID
    /// First and last frame owned by this submap, inclusive. Stored as two
    /// fields rather than a `ClosedRange` so the JSON is `"firstFrame": 0,
    /// "lastFrame": 419` instead of a two-element array nobody can read.
    public var firstFrame: FrameID
    public var lastFrame: FrameID
    public var startTimeSeconds: Double
    public var endTimeSeconds: Double
    /// The rigid correction the pose graph applied to this whole submap.
    /// Identity means the submap was left where VIO put it.
    public var correction: Pose
    /// Fraction of this submap's frames also covered by a neighbour, 0...1.
    public var overlapFraction: Float
    public var bounds: BoundingBox

    /// Inclusive frame range, for the many call sites that want to iterate.
    public var frameRange: ClosedRange<FrameID> { firstFrame...lastFrame }

    public init(
        index: SubmapID,
        firstFrame: FrameID,
        lastFrame: FrameID,
        startTimeSeconds: Double,
        endTimeSeconds: Double,
        correction: Pose,
        overlapFraction: Float,
        bounds: BoundingBox
    ) {
        self.index = index
        self.firstFrame = firstFrame
        self.lastFrame = lastFrame
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.correction = correction
        self.overlapFraction = overlapFraction
        self.bounds = bounds
    }
}

/// What a voxel of the carved occupancy grid says about the world (F2).
///
/// The whole point of this type is that `unknown` is not `empty`. Space beyond
/// LiDAR range, or behind a no-return (glass), is UNKNOWN and Gaussians there
/// are left alone. Only `empty` - a cell some beam demonstrably passed through
/// - licenses deletion.
public enum OccupancyState: UInt8, Codable, Sendable {
    case unknown = 0
    case empty = 1
    case surface = 2
}

/// On-disk sparse occupancy grid produced by free-space carving.
///
/// Layout: a flat run of records, each `(UInt64 mortonKey, UInt8 state,
/// UInt8 reserved, UInt16 hitCount)`, little-endian, sorted by key. Sparse
/// hash semantics: any cell not present is `.unknown`.
public struct OccupancyGridRef: Codable, Hashable, Sendable {
    /// e.g. `prepass/occupancy.bin`
    public var path: String
    /// Edge length of one voxel, metres. ~0.05 by default.
    public var voxelSizeMeters: Float
    /// World position of the (0,0,0) cell's minimum corner.
    public var origin: Vector3
    /// Number of records in the file.
    public var cellCount: Int
    /// Cells whose state is `.empty`, for the QC card's "how much of this
    /// scene did we actually see through" number.
    public var emptyCellCount: Int
    public var surfaceCellCount: Int

    public init(
        path: String,
        voxelSizeMeters: Float,
        origin: Vector3,
        cellCount: Int,
        emptyCellCount: Int,
        surfaceCellCount: Int
    ) {
        self.path = path
        self.voxelSizeMeters = voxelSizeMeters
        self.origin = origin
        self.cellCount = cellCount
        self.emptyCellCount = emptyCellCount
        self.surfaceCellCount = surfaceCellCount
    }
}

/// The two trust scales of F6, as files on disk.
///
/// They are deliberately separate because they must never be mixed: the bias
/// field is spatially averaged (that is what makes it a bias estimate), and
/// the noise field must never be (averaging outliers away is how you get a
/// confidently wrong surface).
public struct TrustFieldRefs: Codable, Hashable, Sendable {
    /// Coarse voxel grid of running mean signed depth residual, variance,
    /// sample count and distinct capture-time count.
    /// Records: `(UInt64 mortonKey, Float mean, Float variance,
    /// UInt32 sampleCount, UInt16 distinctTimeCount, UInt16 reserved)`.
    /// e.g. `prepass/trust_bias.bin`
    public var biasFieldPath: String
    /// 0.25-0.50 m. Coarse on purpose: a bias estimate needs samples.
    public var biasVoxelSizeMeters: Float

    /// Per native-depth-sample noise / outlier estimate, never spatially
    /// averaged. One Float32 per sample, frame-major, same order as the
    /// depth sidecars. e.g. `prepass/trust_noise.bin`
    public var noiseFieldPath: String

    /// Per-frame learned scale and shift on the SENSOR depth, tightly
    /// constrained. Records: `(UInt32 frameIndex, Float scale, Float shift)`.
    /// e.g. `prepass/depth_affine.bin`
    public var depthAffinePath: String?

    /// The remapped ARKit confidence. ARKit flags only ~1.6% of samples low
    /// and is badly calibrated, so its three levels are treated as a RANKING
    /// and remapped through observed revisit residuals into an actual
    /// probability. Records: one Float32 per native sample, frame-major.
    /// e.g. `prepass/confidence_recal.bin`
    public var recalibratedConfidencePath: String?

    public init(
        biasFieldPath: String,
        biasVoxelSizeMeters: Float,
        noiseFieldPath: String,
        depthAffinePath: String? = nil,
        recalibratedConfidencePath: String? = nil
    ) {
        self.biasFieldPath = biasFieldPath
        self.biasVoxelSizeMeters = biasVoxelSizeMeters
        self.noiseFieldPath = noiseFieldPath
        self.depthAffinePath = depthAffinePath
        self.recalibratedConfidencePath = recalibratedConfidencePath
    }
}

/// Per-pixel edge class (F3). Written as one byte per pixel of the NATIVE
/// depth map, not the RGB frame - the whole idea is that depth edges are
/// computed where depth actually exists.
public enum EdgeClass: UInt8, Codable, Sendable {
    /// Not an edge.
    case none = 0
    /// A real depth step: sharpen here.
    case geometric = 1
    /// Strong image gradient over flat depth (a poster, a rug pattern):
    /// flatten here, do not invent geometry.
    case texture = 2
    /// Glass, out of range, or low confidence: ignore here, do not guess.
    case unknown = 3
    /// Inside the dilation band around a geometric edge, where the upsampled
    /// depth map is untrustworthy. Depth loss is zeroed here.
    case band = 4
}

/// Where the per-frame edge classification maps live.
public struct EdgeClassificationRefs: Codable, Hashable, Sendable {
    /// Directory of per-frame maps, e.g. `prepass/edges`. Each file is
    /// `frame_<stamp>.edge8`, `depthWidth * depthHeight` bytes of `EdgeClass`.
    public var directory: String
    /// Dilation applied around geometric edges, in NATIVE depth pixels. One
    /// native pixel is ~7.5 RGB pixels at 1920 wide, hence the ~8 px band in
    /// the spec expressed at RGB resolution.
    public var bandRadiusNativePixels: Int

    public init(directory: String, bandRadiusNativePixels: Int) {
        self.directory = directory
        self.bandRadiusNativePixels = bandRadiusNativePixels
    }
}

/// A detected pane of glass or an open aperture (F5).
public struct GlassRegion: Codable, Hashable, Sendable {
    /// Plane in world space: `dot(normal, X) + offset = 0`.
    public var planeNormal: Vector3
    public var planeOffset: Float
    /// World-space extent of the detected region.
    public var bounds: BoundingBox
    /// Frames in which it was observed.
    public var observedInFrames: [FrameID]
    /// 0...1. Combines LiDAR silence, image brightness, planarity of the
    /// surrounding wall, and whether ARKit independently classified it as a
    /// window.
    public var confidence: Float
    /// True when ARKit's own mesh classification agreed.
    public var confirmedByARKit: Bool

    public init(
        planeNormal: Vector3,
        planeOffset: Float,
        bounds: BoundingBox,
        observedInFrames: [FrameID],
        confidence: Float,
        confirmedByARKit: Bool
    ) {
        self.planeNormal = planeNormal
        self.planeOffset = planeOffset
        self.bounds = bounds
        self.observedInFrames = observedInFrames
        self.confidence = confidence
        self.confirmedByARKit = confirmedByARKit
    }
}

/// The initial Gaussian set the trainer starts from, already shaped by trust
/// (F6): confident samples become thin, opaque, position-pinned discs;
/// doubtful ones become translucent ellipsoids elongated along the viewing
/// ray, free to slide to where the photometry actually wants them.
public struct InitialSplatSetRef: Codable, Hashable, Sendable {
    /// e.g. `prepass/init_splats.ply` - the PLY layout of
    /// `Sources/Export/PLYCodec.swift`, so it round-trips through the same
    /// reader everything else uses.
    public var path: String
    public var splatCount: Int
    /// Per-splat flags, one byte each, same order as the PLY vertices:
    /// bit 0 = position pinned, bit 1 = elongated along ray, bit 2 = on a
    /// detected 3D edge curve (exempt from the disc prior, F4).
    /// e.g. `prepass/init_splats.flags`
    public var flagsPath: String?
    /// Whether per-splat normals were estimated and stored in the PLY's
    /// `nx, ny, nz` properties (they are otherwise written as zeros).
    public var hasNormals: Bool

    public init(path: String, splatCount: Int, flagsPath: String?, hasNormals: Bool) {
        self.path = path
        self.splatCount = splatCount
        self.flagsPath = flagsPath
        self.hasNormals = hasNormals
    }
}

/// One thing worth telling the user about their capture, in their language.
public struct QCFinding: Codable, Hashable, Sendable, Identifiable {
    public enum Severity: String, Codable, Sendable {
        case good
        case warning
        case problem
    }

    /// Stable machine key, e.g. `drift`, `ceiling_coverage`, `glass_area`.
    /// Never shown to the user.
    public var code: String
    public var severity: Severity
    /// One plain sentence, no jargon. Shown as-is.
    public var message: String
    /// What to actually do about it, or nil when there is nothing to do.
    public var fixHint: String?

    public var id: String { code }

    public init(code: String, severity: Severity, message: String, fixHint: String?) {
        self.code = code
        self.severity = severity
        self.message = message
        self.fixHint = fixHint
    }
}

/// The post-capture quality card (F9). Computed from poses and LiDAR only, so
/// it can be on screen in under three seconds - long before any training.
public struct QCCard: Codable, Sendable {
    /// Drift measured from revisit depth residuals, centimetres.
    public var driftCentimeters: Float
    public var loopClosureCount: Int
    /// Median, over surfaces, of the angular spread of viewing directions.
    /// Small means everything was shot from one side.
    public var medianAngularSpreadDegrees: Float
    public var medianCameraToSurfaceMeters: Float
    /// Vertical spread of camera positions. A scan shot entirely at eye level
    /// has a characteristic failure and this is how it is detected.
    public var cameraHeightSpreadMeters: Float
    /// Frames whose step distance or time gap was an outlier, cross-referenced
    /// so a single stumble is not counted twice.
    public var outlierFrameCount: Int
    /// Fraction of the ARKit mesh that met the coverage criteria, 0...1. This
    /// is the "done" number the capture HUD shows.
    public var coverageFraction: Float
    public var ceilingCoverageFraction: Float
    /// Fraction of observed surface area classified as glass or window.
    public var glassAreaFraction: Float
    public var findings: [QCFinding]

    public init(
        driftCentimeters: Float,
        loopClosureCount: Int,
        medianAngularSpreadDegrees: Float,
        medianCameraToSurfaceMeters: Float,
        cameraHeightSpreadMeters: Float,
        outlierFrameCount: Int,
        coverageFraction: Float,
        ceilingCoverageFraction: Float,
        glassAreaFraction: Float,
        findings: [QCFinding]
    ) {
        self.driftCentimeters = driftCentimeters
        self.loopClosureCount = loopClosureCount
        self.medianAngularSpreadDegrees = medianAngularSpreadDegrees
        self.medianCameraToSurfaceMeters = medianCameraToSurfaceMeters
        self.cameraHeightSpreadMeters = cameraHeightSpreadMeters
        self.outlierFrameCount = outlierFrameCount
        self.coverageFraction = coverageFraction
        self.ceilingCoverageFraction = ceilingCoverageFraction
        self.glassAreaFraction = glassAreaFraction
        self.findings = findings
    }
}

/// Everything the no-training pre-pass produced. Serialised to
/// `prepass/prepass_result.json`.
public struct PrePassResult: Codable, Sendable {
    public static let currentFormatVersion = 1

    public var formatVersion: Int
    public var scanID: ScanID
    public var completedAt: Date

    public var submaps: [Submap]
    /// Refined world -> camera pose per frame, keyed by frame index rendered
    /// as a string (JSON object keys are strings; `FrameID` keys would encode
    /// as an array of alternating key/value, which is unreadable by hand).
    public var refinedPoses: [String: Pose]
    /// The calibrated camera-to-IMU offset actually used, seconds.
    public var cameraToIMUTimeOffsetSeconds: Double?

    public var occupancy: OccupancyGridRef?
    public var trust: TrustFieldRefs?
    public var edges: EdgeClassificationRefs?
    public var glassRegions: [GlassRegion]
    public var initialSplats: InitialSplatSetRef?

    public var qcCard: QCCard
    /// Suggested budget for this scene, given the device that ran the
    /// pre-pass. The trainer may lower it further; it must never raise it.
    public var suggestedBudget: TrainingBudget?

    public init(
        formatVersion: Int = PrePassResult.currentFormatVersion,
        scanID: ScanID,
        completedAt: Date,
        submaps: [Submap],
        refinedPoses: [String: Pose],
        cameraToIMUTimeOffsetSeconds: Double?,
        occupancy: OccupancyGridRef?,
        trust: TrustFieldRefs?,
        edges: EdgeClassificationRefs?,
        glassRegions: [GlassRegion],
        initialSplats: InitialSplatSetRef?,
        qcCard: QCCard,
        suggestedBudget: TrainingBudget?
    ) {
        self.formatVersion = formatVersion
        self.scanID = scanID
        self.completedAt = completedAt
        self.submaps = submaps
        self.refinedPoses = refinedPoses
        self.cameraToIMUTimeOffsetSeconds = cameraToIMUTimeOffsetSeconds
        self.occupancy = occupancy
        self.trust = trust
        self.edges = edges
        self.glassRegions = glassRegions
        self.initialSplats = initialSplats
        self.qcCard = qcCard
        self.suggestedBudget = suggestedBudget
    }

    public func refinedPose(for frame: FrameID) -> Pose? {
        refinedPoses[String(frame)]
    }
}

// MARK: - Device capability

/// What this specific iPhone can do with this app.
public enum DeviceTier: String, Codable, Sendable, CaseIterable {
    /// LiDAR plus a recent chip and enough RAM: capture, pre-pass, on-device
    /// training, preview, export - the whole product.
    case full
    /// LiDAR, but an older chip or less RAM: capture, pre-pass and preview
    /// work; training is either reduced or handed to the PC Booster.
    case limited
    /// No LiDAR. The core idea of the app does not work here, and saying so
    /// plainly is kinder than shipping a bad scan.
    case incompatible
}

/// One capability, and whether this device has it. The onboarding screen shows
/// this list verbatim, so `title` and `detail` are user-facing sentences.
public struct FeatureAvailability: Codable, Hashable, Sendable, Identifiable {
    public var id: String { key }
    /// Stable machine key, e.g. `on_device_training`.
    public var key: String
    /// Short label, e.g. "Train scans on this phone".
    public var title: String
    public var isAvailable: Bool
    /// Why not, in plain language, when `isAvailable` is false. Specific to
    /// THIS device: "this iPhone has 4 GB of memory and training needs 6 GB",
    /// never "insufficient resources".
    public var detail: String?

    public init(key: String, title: String, isAvailable: Bool, detail: String?) {
        self.key = key
        self.title = title
        self.isAvailable = isAvailable
        self.detail = detail
    }
}

/// The result of the first-launch device check.
public struct DeviceCapabilityReport: Codable, Sendable {
    public var tier: DeviceTier

    /// Whether ARKit reports scene reconstruction / scene depth support, which
    /// is the only reliable proxy for "has a LiDAR scanner" - there is no
    /// direct API and no `UIRequiredDeviceCapabilities` key for it.
    public var hasLiDAR: Bool
    /// Raw model identifier, e.g. `iPhone17,2`.
    public var deviceModel: String
    /// Marketing-ish chip name when it can be determined, e.g. "A19 Pro".
    public var chipName: String?
    public var totalMemoryBytes: UInt64
    /// `os_proc_available_memory()` at check time: what this process may
    /// actually allocate right now, which is the number that matters and is
    /// usually far below `totalMemoryBytes`.
    public var availableMemoryBytes: UInt64
    public var systemVersion: String
    /// Highest supported Metal GPU family, as a readable string, e.g.
    /// "Apple9".
    public var metalGPUFamily: String?
    /// `ProcessInfo.isLowPowerModeEnabled` at check time.
    public var lowPowerModeEnabled: Bool
    /// Rough sustained-performance class, 0 (unknown) to 3 (best), derived
    /// from chip generation and thermal headroom.
    public var sustainedPerformanceClass: Int

    public var features: [FeatureAvailability]

    /// For `.incompatible` only: the honest, specific, plain-language answer
    /// to the big bold "Why is my device incompatible?" heading. One or two
    /// sentences, naming the actual missing thing.
    public var incompatibleReason: String?

    public init(
        tier: DeviceTier,
        hasLiDAR: Bool,
        deviceModel: String,
        chipName: String?,
        totalMemoryBytes: UInt64,
        availableMemoryBytes: UInt64,
        systemVersion: String,
        metalGPUFamily: String?,
        lowPowerModeEnabled: Bool,
        sustainedPerformanceClass: Int,
        features: [FeatureAvailability],
        incompatibleReason: String?
    ) {
        self.tier = tier
        self.hasLiDAR = hasLiDAR
        self.deviceModel = deviceModel
        self.chipName = chipName
        self.totalMemoryBytes = totalMemoryBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.systemVersion = systemVersion
        self.metalGPUFamily = metalGPUFamily
        self.lowPowerModeEnabled = lowPowerModeEnabled
        self.sustainedPerformanceClass = sustainedPerformanceClass
        self.features = features
        self.incompatibleReason = incompatibleReason
    }
}

// MARK: - Conformances added to PROMOTED types
//
// `SHDegree` is declared in Sources/Export/SplatCloud.swift as
// `enum SHDegree: Int, Sendable, CaseIterable` - no Codable. `TrainingBudget`
// and `SplatModel` both carry one and both have to serialise, so the
// conformance is added here rather than by editing another module's file.
//
// The stdlib supplies `encode(to:)` / `init(from:)` for any `RawRepresentable`
// whose `RawValue` is `Int`, so this empty extension is a complete, correct
// conformance: an `SHDegree` encodes as the bare integer 0...3.
//
// OWNERSHIP: this conformance belongs to Core. Sources/Export must NOT also
// declare `SHDegree: Codable` - two conformances of the same type to the same
// protocol is a hard compile error. See CONTRACTS.md.
extension SHDegree: Codable {}

// MARK: - Training budget

/// `ProcessInfo.ThermalState`, made Codable and storable.
public enum ThermalLevel: Int, Codable, Comparable, Sendable {
    case nominal = 0
    case fair = 1
    case serious = 2
    case critical = 3

    public init(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal: self = .nominal
        case .fair: self = .fair
        case .serious: self = .serious
        case .critical: self = .critical
        @unknown default: self = .serious
        }
    }

    public static func < (lhs: ThermalLevel, rhs: ThermalLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// What to do as the phone heats up. Training a splat field is the hottest
/// thing this app does, and a phone that shuts down mid-train has produced
/// nothing.
public struct ThermalPolicy: Codable, Hashable, Sendable {
    /// At or above this level, start shedding work (drop render resolution a
    /// step, then halve the densification rate).
    public var degradeAt: ThermalLevel
    /// At or above this level, pause and checkpoint. Training resumes when it
    /// cools back below `degradeAt`.
    public var pauseAt: ThermalLevel
    /// Give up and keep the last checkpoint at this level.
    public var abortAt: ThermalLevel
    /// How long to wait between thermal polls, seconds.
    public var sampleIntervalSeconds: Double

    public init(
        degradeAt: ThermalLevel = .fair,
        pauseAt: ThermalLevel = .serious,
        abortAt: ThermalLevel = .critical,
        sampleIntervalSeconds: Double = 5
    ) {
        self.degradeAt = degradeAt
        self.pauseAt = pauseAt
        self.abortAt = abortAt
        self.sampleIntervalSeconds = sampleIntervalSeconds
    }

    public static let `default` = ThermalPolicy()
}

/// Where a training run happens.
public enum TrainingTarget: String, Codable, Sendable {
    case onDevice
    case booster
}

/// The hard ceiling a training run must stay inside. Budget first,
/// densification second: the cap is not a target to grow into, it is a wall.
///
/// Every number here is measured or derived, never assumed - a whole-house
/// scan and a single chair get very different budgets from the same device.
public struct TrainingBudget: Codable, Hashable, Sendable {
    /// Hard maximum number of Gaussians. Densification uses MCMC-style
    /// relocation once this is reached: split something, delete something,
    /// never exceed.
    public var splatCap: Int
    public var iterations: Int
    /// Longest edge of the training render, pixels. Frames are downsampled to
    /// this before any loss is computed.
    public var renderLongEdgePixels: Int
    /// Spherical-harmonics degree. Degree 1 is the default (F10); the type is
    /// `SHDegree` from Sources/Export/SplatCloud.swift (PROMOTED).
    public var shDegree: SHDegree
    /// How many frames are actually trained on. A 4000-frame house walk does
    /// not need 4000 supervision views.
    public var keyframeCount: Int
    /// Ceiling on resident bytes for splat parameters and gradients. Compared
    /// against `os_proc_available_memory()` at runtime, not against
    /// `physicalMemory` - the entitlement may not have been granted.
    public var memoryCeilingBytes: UInt64
    public var thermalPolicy: ThermalPolicy
    /// fp16 for gradients and SH where the dynamic range allows it.
    public var useHalfPrecision: Bool
    /// Visibility-masked sparse Adam: only touch Gaussians that were actually
    /// rasterised this step.
    public var useSparseAdam: Bool
    public var target: TrainingTarget

    public init(
        splatCap: Int,
        iterations: Int,
        renderLongEdgePixels: Int,
        shDegree: SHDegree,
        keyframeCount: Int,
        memoryCeilingBytes: UInt64,
        thermalPolicy: ThermalPolicy = .default,
        useHalfPrecision: Bool = true,
        useSparseAdam: Bool = true,
        target: TrainingTarget = .onDevice
    ) {
        self.splatCap = splatCap
        self.iterations = iterations
        self.renderLongEdgePixels = renderLongEdgePixels
        self.shDegree = shDegree
        self.keyframeCount = keyframeCount
        self.memoryCeilingBytes = memoryCeilingBytes
        self.thermalPolicy = thermalPolicy
        self.useHalfPrecision = useHalfPrecision
        self.useSparseAdam = useSparseAdam
        self.target = target
    }

    /// A starting budget for a device tier and a scene size, in the ranges the
    /// product spec fixes (F10: ~500-3000 iterations, ~150-500k splats,
    /// 480-720 px). The trainer is expected to lower these live as it measures
    /// real memory and heat; it must never raise them.
    ///
    /// `sceneExtentMeters` is the longest edge of the LiDAR bounds: roughly 2
    /// for one object, 5 for a room, 20+ for a house.
    /// `availableMemoryBytes` should come from `os_proc_available_memory()`.
    public static func recommended(
        for tier: DeviceTier,
        sceneExtentMeters: Float,
        availableMemoryBytes: UInt64
    ) -> TrainingBudget {
        // Roughly 200 bytes per splat resident (position, rotation, scale,
        // opacity, degree-1 SH, plus Adam moments), then halved again for
        // safety because the rasteriser's tile lists and the frame cache also
        // have to fit.
        let bytesPerSplat: UInt64 = 200
        let memoryForSplats = availableMemoryBytes / 2
        let capFromMemory = Int(memoryForSplats / bytesPerSplat)

        let scaleCap: Int
        switch sceneExtentMeters {
        case ..<3: scaleCap = 150_000    // one object
        case ..<8: scaleCap = 300_000    // a room
        default: scaleCap = 500_000      // a floor or a house
        }

        switch tier {
        case .full:
            return TrainingBudget(
                splatCap: Swift.max(50_000, Swift.min(scaleCap, capFromMemory)),
                iterations: sceneExtentMeters < 8 ? 3_000 : 2_000,
                renderLongEdgePixels: 720,
                shDegree: .one,
                keyframeCount: sceneExtentMeters < 8 ? 120 : 240,
                memoryCeilingBytes: memoryForSplats,
                target: .onDevice
            )
        case .limited:
            return TrainingBudget(
                splatCap: Swift.max(40_000, Swift.min(150_000, capFromMemory)),
                iterations: 800,
                renderLongEdgePixels: 480,
                shDegree: .zero,
                keyframeCount: 80,
                memoryCeilingBytes: memoryForSplats,
                target: .onDevice
            )
        case .incompatible:
            // Not reachable in practice - the compatibility gate stops before
            // a capture exists to train. Returned rather than crashed so a
            // caller that ignores the gate degrades instead of trapping.
            return TrainingBudget(
                splatCap: 0,
                iterations: 0,
                renderLongEdgePixels: 480,
                shDegree: .zero,
                keyframeCount: 0,
                memoryCeilingBytes: 0,
                target: .booster
            )
        }
    }
}

// MARK: - Trained model

/// Where a finished model came from.
public enum ModelSource: String, Codable, Sendable {
    case onDevice
    case booster
    case imported
}

/// A trained splat field on disk, with the metadata a viewer or exporter needs
/// before it opens a single byte of it. Serialised to `model/model.json`.
///
/// The pixels live in the referenced `.ply` / `.spz`, which are read and
/// written by `Sources/Export` - this is the index, not the data.
public struct SplatModel: Codable, Sendable, Identifiable {
    public var id: String { modelID }

    public var modelID: String
    public var scanID: ScanID
    public var createdAt: Date
    public var source: ModelSource

    /// e.g. `model/model.ply`. At least one of `plyPath` / `spzPath` is
    /// non-nil.
    public var plyPath: String?
    /// e.g. `model/model.spz`.
    public var spzPath: String?

    public var splatCount: Int
    public var shDegree: SHDegree
    public var bounds: BoundingBox
    public var iterationsCompleted: Int
    /// The budget this run actually ran under, after any live degradation.
    public var budgetUsed: TrainingBudget?

    /// Frozen direction-only background model (F5), if one was trained.
    /// e.g. `model/background.bin` plus its shape in `model/background.json`.
    public var backgroundModelPath: String?
    /// Per-frame learned exposure/gain, `(UInt32 frameIndex, Float gain,
    /// Float bias)` records. e.g. `model/exposure.bin`
    public var exposurePath: String?
    /// Coarse per-voxel direction bitmask backing the honesty mask (F9): which
    /// directions each bit of the scene was actually observed from.
    /// e.g. `model/observed_directions.bin`
    public var observedDirectionsPath: String?

    /// PSNR on the ~5% held-out frames, when held-out evaluation ran. This is
    /// the honest quality number; it is reported even when it is bad.
    public var heldOutPSNR: Float?

    public init(
        modelID: String,
        scanID: ScanID,
        createdAt: Date,
        source: ModelSource,
        plyPath: String?,
        spzPath: String?,
        splatCount: Int,
        shDegree: SHDegree,
        bounds: BoundingBox,
        iterationsCompleted: Int,
        budgetUsed: TrainingBudget?,
        backgroundModelPath: String? = nil,
        exposurePath: String? = nil,
        observedDirectionsPath: String? = nil,
        heldOutPSNR: Float? = nil
    ) {
        self.modelID = modelID
        self.scanID = scanID
        self.createdAt = createdAt
        self.source = source
        self.plyPath = plyPath
        self.spzPath = spzPath
        self.splatCount = splatCount
        self.shDegree = shDegree
        self.bounds = bounds
        self.iterationsCompleted = iterationsCompleted
        self.budgetUsed = budgetUsed
        self.backgroundModelPath = backgroundModelPath
        self.exposurePath = exposurePath
        self.observedDirectionsPath = observedDirectionsPath
        self.heldOutPSNR = heldOutPSNR
    }
}

// MARK: - Training progress

/// Coarse training stage. The UI shows a fixed review camera and a stage name,
/// not an iteration counter, because "1847 / 3000" means nothing to anyone.
public enum TrainerStage: String, Codable, Sendable {
    case preparing
    case initializing
    /// Everything free except per-camera pose deltas, which stay frozen for
    /// roughly the first 5k iterations (F1).
    case warmup
    case densifying
    case refining
    /// Late opacity binarization, last ~20% of iterations, skipped on
    /// UNKNOWN regions (F4).
    case binarizing
    case finalizing
    case done
    case failed
    case cancelled
    /// Stopped for heat; will resume on its own.
    case pausedThermal
    /// Stopped because the memory ceiling was hit; the budget is being cut and
    /// training will resume from the last checkpoint.
    case pausedMemory

    public var isTerminal: Bool {
        self == .done || self == .failed || self == .cancelled
    }
}

/// One progress tick from the trainer.
public struct TrainerProgress: Codable, Sendable {
    public var stage: TrainerStage
    public var iteration: Int
    public var totalIterations: Int
    public var splatCount: Int
    /// Exponential moving average of the total loss. Diagnostic only.
    public var lossEMA: Float?
    /// 0...1 over the whole run, or nil when a stage genuinely cannot say.
    /// Never fake a number here: the UI draws an indeterminate spinner for
    /// nil, which is honest, and a stuck bar at 0%, which is not.
    public var fractionComplete: Double?
    public var thermalLevel: ThermalLevel
    /// Bytes currently resident for splat parameters and optimiser state.
    public var residentBytes: UInt64
    /// One plain sentence for the UI, e.g. "Sharpening edges".
    public var message: String
    /// True once a preview render of the current state is worth showing.
    public var previewAvailable: Bool

    public init(
        stage: TrainerStage,
        iteration: Int,
        totalIterations: Int,
        splatCount: Int,
        lossEMA: Float?,
        fractionComplete: Double?,
        thermalLevel: ThermalLevel,
        residentBytes: UInt64,
        message: String,
        previewAvailable: Bool
    ) {
        self.stage = stage
        self.iteration = iteration
        self.totalIterations = totalIterations
        self.splatCount = splatCount
        self.lossEMA = lossEMA
        self.fractionComplete = fractionComplete
        self.thermalLevel = thermalLevel
        self.residentBytes = residentBytes
        self.message = message
        self.previewAvailable = previewAvailable
    }
}

// MARK: - Preview

/// A virtual camera fly-through constrained to stay near where the user
/// actually walked (F9), because a camera that leaves the observed set shows
/// the user artefacts they will never see again and cannot fix.
public struct PreviewCameraPath: Codable, Sendable {
    public struct Keyframe: Codable, Hashable, Sendable {
        public var pose: Pose
        /// Seconds from the start of the fly-through.
        public var timeSeconds: Double
        /// Index of the captured frame this keyframe was derived from, so the
        /// A/B slider can put the real photo next to the render.
        public var sourceFrame: FrameID?

        public init(pose: Pose, timeSeconds: Double, sourceFrame: FrameID?) {
            self.pose = pose
            self.timeSeconds = timeSeconds
            self.sourceFrame = sourceFrame
        }
    }

    public var keyframes: [Keyframe]
    /// Widened relative to the capture camera (100-110 degrees) so the preview
    /// feels like a room rather than a keyhole.
    public var horizontalFOVDegrees: Float
    /// Hard limit on how far a preview camera may stray from the walked path.
    /// ~0.20 m.
    public var maxDeviationMeters: Float
    /// Draw diagonal hatching over pixels whose viewing ray was never observed
    /// (from `SplatModel.observedDirectionsPath`). On by default: showing the
    /// user what is invented is the whole point.
    public var honestyMaskEnabled: Bool

    public init(
        keyframes: [Keyframe],
        horizontalFOVDegrees: Float = 105,
        maxDeviationMeters: Float = 0.20,
        honestyMaskEnabled: Bool = true
    ) {
        self.keyframes = keyframes
        self.horizontalFOVDegrees = horizontalFOVDegrees
        self.maxDeviationMeters = maxDeviationMeters
        self.honestyMaskEnabled = honestyMaskEnabled
    }
}

// MARK: - Export

/// A finished file the user can share, open elsewhere, or send to a PC.
public struct ExportedAsset: Codable, Hashable, Sendable, Identifiable {
    public var id: String { url.path }

    public var url: URL
    /// `ExportFormat` from Sources/Export/ExportService.swift is the splat
    /// formats only (ply/spz/glb); a packaged capture bundle is a zip and has
    /// no `ExportFormat` case, so this is a plain lowercase extension string.
    public var fileExtension: String
    public var byteCount: Int64
    public var createdAt: Date
    public var scanID: ScanID
    /// Present for splat exports, nil for a packaged capture bundle.
    public var splatCount: Int?

    public init(
        url: URL,
        fileExtension: String,
        byteCount: Int64,
        createdAt: Date,
        scanID: ScanID,
        splatCount: Int?
    ) {
        self.url = url
        self.fileExtension = fileExtension
        self.byteCount = byteCount
        self.createdAt = createdAt
        self.scanID = scanID
        self.splatCount = splatCount
    }
}

// MARK: - Booster
//
// PROMOTED, not re-declared: the wire protocol already exists in
// Sources/Booster/BoosterProtocol.swift and is specified byte-for-byte in
// docs/BOOSTER_PROTOCOL.md. `BoosterJobStage`, `BoosterProgressEvent`,
// `BoosterManifest`, `BoosterManifestFile`, `BoosterInfo`, the pairing bodies
// and the job bodies all live there and are part of the contract surface.
// Only the app-level view of a job is added here.

/// One Booster job as the rest of the app cares about it: which scan, which
/// PC, how far along, and where the result landed.
///
/// `Sources/Booster/BoosterJobStore.swift` persists the equivalent
/// `BoosterJobRecord`; the two have the same fields and map 1:1. Folding one
/// into the other is optional cleanup, not a prerequisite (see CONTRACTS.md).
public struct BoosterJob: Codable, Hashable, Sendable, Identifiable {
    public var id: String { jobID }

    public var jobID: String
    public var scanID: ScanID
    public var boosterID: String
    public var boosterName: String
    public var createdAt: Date
    public var stage: BoosterJobStage
    /// Plain sentence, shown verbatim.
    public var message: String
    public var fractionComplete: Double?
    /// Absolute path of the downloaded result, once there is one.
    public var resultDirectory: String?

    public init(
        jobID: String,
        scanID: ScanID,
        boosterID: String,
        boosterName: String,
        createdAt: Date,
        stage: BoosterJobStage,
        message: String,
        fractionComplete: Double? = nil,
        resultDirectory: String? = nil
    ) {
        self.jobID = jobID
        self.scanID = scanID
        self.boosterID = boosterID
        self.boosterName = boosterName
        self.createdAt = createdAt
        self.stage = stage
        self.message = message
        self.fractionComplete = fractionComplete
        self.resultDirectory = resultDirectory
    }
}

// MARK: - Errors

/// Errors that cross a module boundary. Each module keeps its own richer error
/// type (`ExportError`, `BoosterError`); this is what a caller sees when it
/// does not care which module failed, only what to tell the user.
public enum NimbusError: LocalizedError, Sendable {
    case deviceIncompatible(String)
    case moduleNotInstalled(module: String)
    case captureFailed(String)
    case prePassFailed(String)
    case trainingFailed(String)
    case outOfMemory(needBytes: UInt64, haveBytes: UInt64)
    case thermalAbort
    case cancelled
    case scanNotFound(ScanID)
    case malformedData(String)

    public var errorDescription: String? {
        switch self {
        case .deviceIncompatible(let reason):
            return reason
        case .moduleNotInstalled(let module):
            return "The \(module) part of the app is not installed in this build yet."
        case .captureFailed(let reason):
            return "The scan could not be recorded. \(reason)"
        case .prePassFailed(let reason):
            return "The scan could not be checked over. \(reason)"
        case .trainingFailed(let reason):
            return "The 3D model could not be built. \(reason)"
        case .outOfMemory(let need, let have):
            let needMB = need / 1_048_576
            let haveMB = have / 1_048_576
            return "This scan needs about \(needMB) MB of memory and only "
                + "\(haveMB) MB is free. Try a smaller area, or send it to a "
                + "computer on your Wi-Fi to finish."
        case .thermalAbort:
            return "Your phone got too warm to keep going. What was finished "
                + "has been saved. Let it cool down and pick this back up."
        case .cancelled:
            return "Stopped."
        case .scanNotFound(let scanID):
            return "That scan (\(scanID)) is not on this phone any more."
        case .malformedData(let what):
            return "Some of this scan's data could not be read: \(what)"
        }
    }
}

// MARK: - Services
//
// Each protocol below is implemented by exactly one module. The concrete type
// name is fixed in CONTRACTS.md so two modules can never collide on it.

/// First-launch device check. Implemented by `Sources/Onboarding`.
public protocol DeviceCompatibilityService: AnyObject {
    /// Probes ARKit, Metal, `ProcessInfo` and the device model, and returns
    /// the tier plus the per-feature availability list the onboarding screen
    /// renders. Cheap enough to run on every launch; the result is cached by
    /// the caller, not here.
    func evaluate() async -> DeviceCapabilityReport
}

/// What the capture HUD needs, live, at frame rate. Implemented by
/// `Sources/Capture`.
public struct CaptureLiveState: Sendable {
    public var frameCount: Int
    public var elapsedSeconds: Double
    /// 0...1, the "done" number.
    public var coverageFraction: Float
    /// The live blur meter in pixels of smear. Amber at 2, red at 4.
    public var motionBlurPixels: Float
    public var trackingQuality: TrackingQuality
    public var thermalLevel: ThermalLevel
    /// Bytes written so far, so the UI can say "1.2 GB" before storage runs
    /// out rather than after.
    public var bytesWritten: Int64
    /// The single most useful thing to say right now: "walk around the left
    /// side", "move closer", "slow down". nil when nothing needs saying.
    public var guidanceHint: String?

    public init(
        frameCount: Int,
        elapsedSeconds: Double,
        coverageFraction: Float,
        motionBlurPixels: Float,
        trackingQuality: TrackingQuality,
        thermalLevel: ThermalLevel,
        bytesWritten: Int64,
        guidanceHint: String?
    ) {
        self.frameCount = frameCount
        self.elapsedSeconds = elapsedSeconds
        self.coverageFraction = coverageFraction
        self.motionBlurPixels = motionBlurPixels
        self.trackingQuality = trackingQuality
        self.thermalLevel = thermalLevel
        self.bytesWritten = bytesWritten
        self.guidanceHint = guidanceHint
    }
}

/// Runs the ARKit session and writes the capture bundle. Implemented by
/// `Sources/Capture`.
///
/// `@MainActor` because it owns an `ARSession` and drives the HUD; the heavy
/// per-frame writing happens on its own queue behind this facade.
@MainActor
public protocol CaptureService: AnyObject {
    /// Live HUD state. Finishes when the session stops.
    var liveState: AsyncStream<CaptureLiveState> { get }

    /// Starts a new capture and returns its scan id and folder immediately, so
    /// the UI has somewhere to point before a single frame is written.
    func start(displayName: String) async throws -> CaptureBundleRef

    /// Stops the session, re-reads the ARKit anchors, writes
    /// `capture_bundle.json`, and returns the finished index.
    func finish() async throws -> CaptureBundle

    /// Stops and deletes everything written so far.
    func cancel() async
}

/// The no-training pre-pass (F1, F2, F3, F6, F8). Implemented by
/// `Sources/PrePass`.
public protocol PrePassService: AnyObject {
    /// Runs submap construction, revisit detection, pose-graph optimisation,
    /// time-offset calibration, free-space carving, edge classification, the
    /// trust fields and the QC card, writing everything under `prepass/` and
    /// returning the index.
    ///
    /// The QC card alone must be available in under three seconds; the stream
    /// therefore yields a partial `PrePassResult` as soon as the card exists
    /// and continues yielding as the heavier stages land.
    func run(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) -> AsyncThrowingStream<PrePassResult, Error>

    /// The fast path on its own: poses and LiDAR only, no carving, no trust.
    /// This is what the post-capture screen calls.
    func quickQCCard(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async throws -> QCCard
}

/// On-device 3DGS training. Implemented by `Sources/Trainer`.
public protocol SplatTrainer: AnyObject {
    /// Trains and streams progress. The stream finishes after a terminal
    /// `TrainerStage`; the finished model is then available from
    /// `finishedModel()`.
    ///
    /// The trainer owns the right to LOWER `budget` as it measures real memory
    /// and heat, and reports having done so in `TrainerProgress`. It may never
    /// raise it.
    func train(
        bundle: CaptureBundle,
        prePass: PrePassResult,
        at ref: CaptureBundleRef,
        budget: TrainingBudget
    ) -> AsyncThrowingStream<TrainerProgress, Error>

    /// The model produced by the last completed `train`, or nil if none has
    /// completed.
    func finishedModel() async -> SplatModel?

    /// A snapshot of the current splat field, for a live preview or an early
    /// export. Cheap enough to call between iterations.
    func snapshot() async throws -> SplatCloud

    func cancel() async
}

/// Real-time preview rendering. Implemented by `Sources/Viewer`.
@MainActor
public protocol SplatRenderer: AnyObject {
    /// Loads a model for display. Streaming: returns once enough is resident
    /// to draw something, and keeps loading behind that.
    func load(_ model: SplatModel, at ref: CaptureBundleRef) async throws

    /// Draws directly from an in-memory cloud, for the live training preview.
    func load(_ cloud: SplatCloud) async throws

    /// Sets the camera for the next frame.
    func setCamera(pose: Pose, intrinsics: CameraIntrinsics)

    /// Diagonal hatching over never-observed directions (F9).
    func setHonestyMaskEnabled(_ enabled: Bool)

    /// Per-pixel artefact heatmap overlay.
    func setArtifactHeatmapEnabled(_ enabled: Bool)

    /// Renders one frame into the currently bound drawable.
    func renderFrame()
}

/// Writing splat files and packaging capture bundles. Implemented by
/// `Sources/Export`'s concrete `ExportService`.
///
/// NAME NOTE: the obvious name `ExportService` is already the concrete class,
/// and `ExportServicing` is already Export's own local protocol. This protocol
/// is therefore `SplatExporting`, and Export conforms to it with a one-line
/// extension - nothing has to be deleted or renamed for the build to work.
public protocol SplatExporting: AnyObject {
    /// Writes `cloud` to `export/<scanID>.<format>`.
    func exportAsset(
        _ cloud: SplatCloud,
        scanID: ScanID,
        format: ExportFormat
    ) async throws -> ExportedAsset

    /// Zips the capture bundle for the PC Booster or for the user to keep.
    func packageCaptureBundle(scanID: ScanID) async throws -> ExportedAsset

    /// Reads a `.ply` or `.spz` back in. The warning is non-fatal and worth
    /// showing: "this file had data this app skipped".
    func importSplatCloud(from url: URL) async throws -> (cloud: SplatCloud, warning: String?)
}

/// The optional LAN Booster. Implemented by `Sources/Booster`'s concrete
/// `BoosterClient`.
///
/// NAME NOTE: `BoosterClient` is already the concrete class, hence
/// `BoosterService` for the protocol.
///
/// Nothing in the app may require this to exist. Every entry point is opt-in.
@MainActor
public protocol BoosterService: AnyObject {
    /// Sends a scan to a paired Booster and brings the result back into
    /// `Scans/<scanID>/model`. Progress is observable on the concrete type.
    func sendScan(scanID: ScanID, scanDirectory: URL, to device: BoosterDevice)

    func cancelActiveSend()

    /// Everything this phone has ever sent, newest first.
    func jobs() -> [BoosterJob]
}

// MARK: - Smart-component protocols
//
// The five pieces that make this app smart rather than merely a 3DGS trainer.
// Implemented by `Sources/Smart`, except `FreeSpaceCarver` and `PoseRefiner`
// which are owned by `Sources/PrePass` (they run before training, on poses and
// LiDAR only). The protocol lives here either way so `Sources/Trainer` can
// depend on the behaviour without depending on the module.

/// F2. "Empty air is evidence."
public protocol FreeSpaceCarver: AnyObject {
    /// Carves the occupancy grid from every LiDAR ray of every frame. Cells a
    /// beam passed through become `.empty`; endpoint cells become `.surface`;
    /// space past `lidarMaxRangeMeters` or behind a no-return stays
    /// `.unknown` and must never be marked empty.
    func carve(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        voxelSizeMeters: Float
    ) async throws -> OccupancyGridRef

    /// Loads the grid for querying during training.
    func load(_ grid: OccupancyGridRef, at ref: CaptureBundleRef) async throws

    /// State of the cell containing a world point. `.unknown` when the grid
    /// has not been loaded, never a guess.
    func state(atWorldPoint point: Vector3) -> OccupancyState

    /// The pruning step: indices, into the caller's splat array, of Gaussians
    /// whose centres sit in certified-`.empty` cells and can be hard-deleted.
    func certifiedEmptyIndices(centers: [SIMD3<Float>]) -> [Int]
}

/// F6. Two scales, never mixed.
public protocol TrustField: AnyObject {
    func build(
        bundle: CaptureBundle,
        prePassPoses: [String: Pose],
        at ref: CaptureBundleRef
    ) async throws -> TrustFieldRefs

    func load(_ refs: TrustFieldRefs, at ref: CaptureBundleRef) async throws

    /// Inverse-variance weight for one native depth sample, 0...1. Soft by
    /// construction: hard per-region switches produce visible seams, so this
    /// never returns a step function.
    func weight(frame: FrameID, sampleIndex: Int) -> Float

    /// Coarse spatial bias estimate at a world point: the running mean signed
    /// residual and its variance, or nil where too few samples landed.
    func bias(atWorldPoint point: Vector3) -> (mean: Float, variance: Float)?

    /// Multiplier on the depth loss at iteration `iteration` of `total`.
    /// Strong for the first ~3-7k iterations, then tapers - depth is a prior,
    /// not the answer.
    func depthLossScale(iteration: Int, of total: Int) -> Float
}

/// F3. Geometric / texture / unknown.
public protocol EdgeClassifier: AnyObject {
    /// Classifies every native depth pixel of every frame and writes the maps.
    func classify(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async throws -> EdgeClassificationRefs

    /// The classification map for one frame, `depthWidth * depthHeight` bytes.
    func map(for frame: FrameID) -> [EdgeClass]
}

/// F5. The direction-only far-field model, global and frozen after warm-up.
public protocol BackgroundModel: AnyObject {
    /// Warm-up: trained jointly with the Gaussians for the first stretch, then
    /// frozen so it cannot absorb foreground error.
    func warmUp(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef,
        iterations: Int
    ) async throws

    func freeze()

    /// Radiance for a world-space viewing direction, composited behind the
    /// Gaussians using accumulated alpha.
    func radiance(forDirection direction: Vector3) -> SIMD3<Float>

    /// Per-pixel authority: how much this pixel's depth may be believed,
    /// 0 (not at all - route to infinity) to 1 (LiDAR-authoritative). Built
    /// from LiDAR validity, recalibrated confidence, a soft 4.5-5.5 m range
    /// ramp, the glass/sky mask, saturation, and the parallax gate.
    func authority(frame: FrameID, sampleIndex: Int) -> Float

    func write(to ref: CaptureBundleRef) async throws -> String
}

/// F1. The single biggest quality lever.
public protocol PoseRefiner: AnyObject {
    /// Splits the capture into 15-30 s submaps with 20-30% overlap.
    func buildSubmaps(bundle: CaptureBundle) -> [Submap]

    /// Finds revisits by pose proximity, view-direction similarity, and
    /// point-to-plane ICP on the native depth maps.
    func detectRevisits(
        bundle: CaptureBundle,
        submaps: [Submap],
        at ref: CaptureBundleRef
    ) async throws -> [RevisitPair]

    /// Sweeps the camera-to-IMU offset over -50...+50 ms in 5 ms steps,
    /// re-interpolating poses at each step and minimising tracked-feature
    /// reprojection error. Returns the winning offset in seconds, or nil when
    /// the sweep has no clear minimum - which is reported, not smoothed over.
    func calibrateTimeOffset(
        bundle: CaptureBundle,
        at ref: CaptureBundleRef
    ) async throws -> Double?

    /// Optimises one rigid SE(3) per submap against the revisit constraints
    /// with a robust (Huber / annealed) loss, then returns the per-frame
    /// refined poses keyed by `String(frameIndex)`.
    ///
    /// This never runs COLMAP from scratch and never triangulates-then-bundle-
    /// adjusts from the ARKit prior: measured on 15 of 15 rooms, that made the
    /// poses worse (arXiv 2608.21008).
    func optimize(
        bundle: CaptureBundle,
        submaps: [Submap],
        revisits: [RevisitPair],
        timeOffsetSeconds: Double?
    ) async throws -> [String: Pose]
}

// MARK: - Service registry

/// Where the app finds the modules that are actually present in this build.
///
/// Every property is optional and starts nil. A nil service is not an error:
/// it means that module has not landed yet, and the UI shows an honest
/// placeholder naming it rather than pretending. The one place these get
/// filled in is the integration block at the top of
/// `Sources/App/NimbusApp.swift` - deliberately one visible list, not a
/// scattering of `+load`-style side effects nobody can find.
@MainActor
public final class NimbusServices {
    public static let shared = NimbusServices()

    public var deviceCompatibility: (any DeviceCompatibilityService)?
    public var capture: (any CaptureService)?
    public var prePass: (any PrePassService)?
    public var trainer: (any SplatTrainer)?
    public var renderer: (any SplatRenderer)?
    public var exporter: (any SplatExporting)?
    public var booster: (any BoosterService)?

    private init() {}

    /// Modules present in this build, by name, for the About screen and for
    /// the honest "not installed yet" placeholders.
    public var installedModules: [String] {
        var names: [String] = ["Core", "App"]
        if deviceCompatibility != nil { names.append("Onboarding") }
        if capture != nil { names.append("Capture") }
        if prePass != nil { names.append("PrePass") }
        if trainer != nil { names.append("Trainer") }
        if renderer != nil { names.append("Viewer") }
        if exporter != nil { names.append("Export") }
        if booster != nil { names.append("Booster") }
        return names
    }
}

// MARK: - Screen registry

#if canImport(SwiftUI)
import SwiftUI

/// Where a module hands the app shell a screen to show.
///
/// `Sources/App` owns the tab bar but must not have to know that
/// `Sources/Capture` calls its screen `CaptureView` - otherwise App cannot
/// compile until every module has landed, which is exactly the deadlock this
/// registry exists to avoid. A module registers its screen in the integration
/// block in `Sources/App/NimbusApp.swift`; anything still nil renders as an
/// honest placeholder naming the module that will fill it.
@MainActor
public final class NimbusUI {
    public static let shared = NimbusUI()

    /// The capture tab. Owned by `Sources/Capture`.
    public var captureScreen: (() -> AnyView)?

    /// The scan library / review / preview / export tab. Owned by
    /// `Sources/Viewer`.
    public var libraryScreen: (() -> AnyView)?

    /// The first-run flow. Owned by `Sources/Onboarding`. Given the report
    /// and a completion callback to invoke when the user is through it.
    ///
    /// When this is nil the app shell renders its own minimal gate from the
    /// report's data, which is enough to be correct but is not the finished
    /// first-run experience.
    public var onboardingFlow: ((DeviceCapabilityReport, @escaping () -> Void) -> AnyView)?

    /// The Booster tab. Owned by `Sources/Booster`, which ships
    /// `BoosterTabView` - registered in the integration block.
    public var boosterScreen: (() -> AnyView)?

    private init() {}
}
#endif
