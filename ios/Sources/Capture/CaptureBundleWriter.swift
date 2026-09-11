//
//  CaptureBundleWriter.swift
//  Capture
//
//  Everything a finished session leaves behind, written in the order that
//  survives being interrupted.
//
//  ORDER MATTERS AND IT IS THIS:
//
//    1. the COLMAP model      (sparse/0/cameras.txt, images.txt, points3D.txt)
//    2. the anchors           (anchors/anchors_session.json, anchors_final.json)
//    3. the mesh              (mesh/chunk_*.ply, .cls, mesh_index.json)
//    4. capture_bundle.json   LAST
//
//  `capture_bundle.json` is the index, and docs/DATA_FORMAT.md section 3 says
//  that when it and `sensor_data/frames.jsonl` disagree, the bundle wins. So
//  the bundle must not exist until everything it claims exists really does. If
//  the app is killed at step 3, there is no bundle, the append-only frame log
//  is still complete and correct, and the scan is recoverable. If the bundle
//  were written first, a kill would leave an index pointing at files that were
//  never created, and every reader downstream would trust it.
//
//  The mesh is written before the bundle for the same reason: `meshChunks` in
//  the bundle is a list of paths, and a path in an index that is not on disk is
//  a lie the pre-pass would then act on.
//

import Foundation
import simd

/// Assembles and writes the end-of-session outputs.
enum CaptureBundleWriter {

    /// Writes every file a finished capture owns and returns the index.
    ///
    /// - Parameter sceneBoundsFallback: used when the LiDAR cloud is empty,
    ///   which happens on a very short capture. The mesh's own bounds are the
    ///   fallback; nil bounds are written as nil rather than as a zero box,
    ///   because a zero box would make the trainer's budget think the scene is
    ///   a point.
    /// - Parameter revisitPairs: the candidates capture found by pose
    ///   proximity. Empty means "none found", which on a walk that never
    ///   returns anywhere is the true answer. The pre-pass runs depth ICP over
    ///   these and replaces them with measured constraints.
    /// - Parameter cameraToIMUTimeOffsetSeconds: the offset capture measured
    ///   live by correlating pose turn rate against gyro turn rate, or nil
    ///   when the sweep had no clear peak. The pre-pass re-derives it from
    ///   reprojection error and may overwrite this.
    static func write(
        folder: CaptureScanFolder,
        displayName: String,
        intrinsics: CameraIntrinsics,
        settings: CaptureSettings,
        frames: [CaptureFrame],
        anchorsDuringSession: [AnchorRecord],
        anchorsAtEndOfSession: [AnchorRecord],
        meshSnapshots: [CaptureMeshSnapshot],
        pointCloud: CapturePointCloudAccumulator,
        sceneBoundsFallback: BoundingBox?,
        revisitPairs: [RevisitPair] = [],
        cameraToIMUTimeOffsetSeconds: Double? = nil
    ) throws -> CaptureBundle {

        // 1. COLMAP model.
        try CaptureCOLMAPWriter.writeCameras(
            intrinsics,
            to: folder.sparseModelDirectory.appendingPathComponent("cameras.txt")
        )
        try CaptureCOLMAPWriter.writeImages(
            frames: frames,
            to: folder.sparseModelDirectory.appendingPathComponent("images.txt")
        )
        let points = pointCloud.makePoints()
        try CaptureCOLMAPWriter.writePoints(
            points,
            to: folder.sparseModelDirectory.appendingPathComponent("points3D.txt")
        )

        // 2. Anchors, both lists (F8).
        try CaptureAnchorRecorder.write(
            session: anchorsDuringSession,
            final: anchorsAtEndOfSession,
            to: folder
        )

        // 3. Mesh chunks and their per-face classification.
        let meshChunks = try CaptureMeshWriter.write(
            snapshots: meshSnapshots,
            to: folder
        )

        // 4. The index, last.
        let bundle = CaptureBundle(
            scanID: folder.scanID,
            createdAt: folder.startedAt,
            displayName: displayName,
            deviceModel: deviceModelIdentifier(),
            appVersion: BrandConfig.versionString,
            intrinsics: intrinsics,
            settings: settings,
            frames: frames,
            // Measured live during the session by correlating the pose's turn
            // rate against the gyro's over the -50...+50 ms sweep (F1), or nil
            // when that sweep had no clear peak. Nil is reported, never
            // replaced with a plausible-looking zero.
            cameraToIMUTimeOffsetSeconds: cameraToIMUTimeOffsetSeconds,
            anchorsDuringSession: anchorsDuringSession,
            anchorsAtEndOfSession: anchorsAtEndOfSession,
            meshChunks: meshChunks,
            // Pose-proximity CANDIDATES only: two keyframes that sit in the
            // same place, look the same way, and are far apart in time. No
            // depth alignment has run, so their residuals are unmeasured and
            // their method is `poseProximity`. The pre-pass's ICP turns these
            // into measurements (F1).
            revisitPairs: revisitPairs,
            sceneBounds: pointCloud.bounds() ?? sceneBoundsFallback,
            pointCloudPath: CaptureScanFolder.pointCloudRelativePath
        )

        let data = try ContractsJSON.encoder().encode(bundle)
        try data.write(to: folder.bundleURL, options: .atomic)

        let message = "Wrote \(frames.count) frames, "
                + "\(points.count) points and "
                + "\(meshChunks.count) mesh chunks for "
                + "\(folder.scanID)."
        CaptureLog.writer.notice("\(message, privacy: .public)")

        return bundle
    }

    /// `iPhone17,2` and friends: the raw hardware identifier, not the
    /// marketing name.
    ///
    /// `CaptureBundle.deviceModel` is documented as the raw identifier because
    /// a marketing name is localised, changes between OS releases, and cannot
    /// be parsed back into a chip generation - which is exactly what a
    /// downstream reader wants it for.
    static func deviceModelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let identifier = withUnsafeBytes(of: &systemInfo.machine) { raw -> String in
            let bytes = raw.bindMemory(to: CChar.self)
            guard let base = bytes.baseAddress else { return "" }
            return String(cString: base)
        }
        return identifier.isEmpty ? "unknown" : identifier
    }
}
