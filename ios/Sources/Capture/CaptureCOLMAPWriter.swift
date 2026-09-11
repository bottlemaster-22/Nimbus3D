//
//  CaptureCOLMAPWriter.swift
//  Capture
//
//  `sparse/0/cameras.txt`, `images.txt` and `points3D.txt`, byte-shaped
//  exactly as docs/DATA_FORMAT.md section 4 specifies.
//
//  THREE THINGS THAT ARE EASY TO GET WRONG AND EXPENSIVE TO FIND LATER:
//
//  1. Quaternion order. In memory and in JSON this app stores `(x, y, z, w)`.
//     COLMAP's `images.txt` writes `QW QX QY QZ`. The re-ordering happens
//     HERE, in the writer, and nowhere else.
//
//  2. The blank second line of every image record. This app does not
//     triangulate, so there are no `POINTS2D`, but COLMAP's own reader
//     requires the line to exist. Omitting it makes the file unreadable by
//     roughly half the tools that claim to read COLMAP text.
//
//  3. The `ERROR` column of `points3D.txt` is expected metric error in METRES,
//     not reprojection pixels. That is a deliberate reinterpretation, it is
//     the only per-point uncertainty slot the format has, and leaving it at
//     zero would be a lie. DATA_FORMAT section 4 documents it so nothing
//     downstream mistakes it for pixels.
//
//  Written with a buffered append rather than one giant string: a house scan
//  is a few million points, and building that as a single Swift `String`
//  before writing peaks at several hundred megabytes for no reason.
//

import Foundation

/// Writes the COLMAP text model for a capture.
enum CaptureCOLMAPWriter {

    /// How many bytes to buffer before flushing to the file descriptor.
    private static let flushThreshold = 1 << 20  // 1 MiB

    // MARK: - cameras.txt

    /// One shared PINHOLE camera, referenced by every image.
    ///
    /// `PINHOLE` and not `SIMPLE_PINHOLE` (fx and fy differ in general) and not
    /// `OPENCV` (ARKit hands back rectified frames and publishes no usable
    /// distortion coefficients, so a distortion model would be inventing
    /// numbers).
    static func writeCameras(
        _ intrinsics: CameraIntrinsics,
        to url: URL
    ) throws {
        var text = """
            # Camera list with one line of data per camera:
            #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            # Number of cameras: 1

            """
        text += String(
            format: "1 PINHOLE %d %d %.6f %.6f %.6f %.6f\n",
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy
        )
        try Data(text.utf8).write(to: url, options: .atomic)
    }

    // MARK: - images.txt

    /// Raw ARKit poses, one two-line record per frame.
    ///
    /// - Parameter usingRefinedPoses: false everywhere in this module. The
    ///   parameter exists because `prepass/sparse_refined/images.txt` is the
    ///   same writer with the same conventions, and duplicating it there would
    ///   be how the two files quietly drift apart.
    static func writeImages(
        frames: [CaptureFrame],
        usingRefinedPoses: Bool = false,
        to url: URL
    ) throws {
        let handle = try makeHandle(at: url)
        defer {
            try? handle.synchronize()
            try? handle.close()
        }

        var buffer = Data()
        buffer.reserveCapacity(flushThreshold + 4096)

        append(
            """
            # Image list with two lines of data per image:
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            #   POINTS2D[] as (X, Y, POINT3D_ID)
            # Number of images: \(frames.count)

            """,
            to: &buffer
        )

        for frame in frames {
            let pose = usingRefinedPoses ? (frame.refinedPose ?? frame.rawPose)
                                         : frame.rawPose
            let q = pose.rotation.normalized
            let t = pose.translation
            let name = (frame.imagePath as NSString).lastPathComponent

            // COLMAP ids are 1-based; CaptureFrame.index is 0-based.
            append(
                String(
                    format: "%u %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 %@\n\n",
                    UInt32(frame.index) &+ 1,
                    q.w, q.x, q.y, q.z,
                    t.x, t.y, t.z,
                    name
                ),
                to: &buffer
            )

            if buffer.count >= flushThreshold {
                try handle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
            }
        }

        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
        }
    }

    // MARK: - points3D.txt

    /// The baked LiDAR cloud.
    static func writePoints(
        _ points: [CapturePointCloudAccumulator.Point],
        to url: URL
    ) throws {
        let handle = try makeHandle(at: url)
        defer {
            try? handle.synchronize()
            try? handle.close()
        }

        var buffer = Data()
        buffer.reserveCapacity(flushThreshold + 4096)

        append(
            """
            # 3D point list with one line of data per point:
            #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            # Number of points: \(points.count)
            # ERROR is the expected metric error in METRES (see docs/DATA_FORMAT.md
            # section 4), not a reprojection residual in pixels.

            """,
            to: &buffer
        )

        for (offset, point) in points.enumerated() {
            append(
                String(
                    format: "%d %.6f %.6f %.6f %d %d %d %.6f\n",
                    offset + 1,
                    point.position.x,
                    point.position.y,
                    point.position.z,
                    Int(point.color.x),
                    Int(point.color.y),
                    Int(point.color.z),
                    point.errorMeters
                ),
                to: &buffer
            )

            if buffer.count >= flushThreshold {
                try handle.write(contentsOf: buffer)
                buffer.removeAll(keepingCapacity: true)
            }
        }

        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
        }
    }

    // MARK: - Private

    private static func makeHandle(at url: URL) throws -> FileHandle {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: url.path) {
            try fileManager.removeItem(at: url)
        }
        guard fileManager.createFile(atPath: url.path, contents: nil) else {
            throw NimbusError.captureFailed(
                "Could not create \(url.lastPathComponent)."
            )
        }
        return try FileHandle(forWritingTo: url)
    }

    @inline(__always)
    private static func append(_ string: String, to data: inout Data) {
        data.append(contentsOf: Array(string.utf8))
    }
}
