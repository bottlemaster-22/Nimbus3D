//
//  NerfstudioDatasetExporter.swift
//  Nimbus3D - SplatEngine module
//
//  Converts a CaptureBundle (posed RGB frames + pinhole intrinsics from ARKit)
//  into a Nerfstudio-style dataset on disk that the Brush engine loads directly:
//
//      <datasetDir>/
//        transforms.json      (camera model + per-frame pose & intrinsics)
//        images/frame_00000.jpg ...
//
//  Brush accepts Nerfstudio (transforms.json) or COLMAP inputs. Because ARKit
//  already gives us world-space camera poses, we do NOT need COLMAP structure-
//  from-motion: the poses go straight into transforms.json and Brush random-
//  initialises the point cloud.
//

import Foundation
import simd

enum NerfstudioDatasetExporter {

    /// Writes the dataset and returns its root directory.
    static func export(bundle: CaptureBundle) throws -> URL {
        let fileManager = FileManager.default
        let root = try datasetsDirectory().appendingPathComponent(bundle.id.uuidString, isDirectory: true)

        // Start clean so a re-run never mixes stale frames in.
        if fileManager.fileExists(atPath: root.path) {
            try fileManager.removeItem(at: root)
        }
        let imagesDir = root.appendingPathComponent("images", isDirectory: true)
        try fileManager.createDirectory(at: imagesDir, withIntermediateDirectories: true)

        var frameEntries: [[String: Any]] = []
        frameEntries.reserveCapacity(bundle.frames.count)

        for frame in bundle.frames.sorted(by: { $0.index < $1.index }) {
            // Copy the image into the dataset so file_path is relative + portable.
            let ext = frame.imageURL.pathExtension.isEmpty ? "jpg" : frame.imageURL.pathExtension
            let imageName = String(format: "frame_%05d.%@", frame.index, ext)
            let destination = imagesDir.appendingPathComponent(imageName)
            // TODO(nimbus): SEAM(image-format) if `ext` is HEIC/HEIF, transcode to
            // JPEG/PNG here. Brush's image loader may not decode HEIC; ARKit can
            // emit either depending on capture settings. Left as a straight copy
            // for now so the wiring is real and the seam is explicit.
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.copyItem(at: frame.imageURL, to: destination)

            let intrinsics = frame.intrinsics
            frameEntries.append([
                "file_path": "images/\(imageName)",
                "w": intrinsics.imageWidth,
                "h": intrinsics.imageHeight,
                "fl_x": Double(intrinsics.focalLength.x),
                "fl_y": Double(intrinsics.focalLength.y),
                "cx": Double(intrinsics.principalPoint.x),
                "cy": Double(intrinsics.principalPoint.y),
                "transform_matrix": transformMatrixRows(frame.pose.matrix)
            ])
        }

        // PINHOLE: fx/fy/cx/cy per frame, no distortion (ARKit intrinsics are
        // already rectified). Intrinsics are also repeated per-frame above so
        // datasets with mixed image sizes still load.
        let root_json: [String: Any] = [
            "camera_model": "PINHOLE",
            "frames": frameEntries
        ]

        let data = try JSONSerialization.data(withJSONObject: root_json, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: root.appendingPathComponent("transforms.json"), options: .atomic)

        return root
    }

    /// Nerfstudio expects `transform_matrix` as a ROW-major 4x4 camera-to-world
    /// matrix in the OpenGL/Blender convention (+x right, +y up, camera looks
    /// down -z). ARKit's `ARCamera.transform` uses the same convention, so we
    /// emit it unchanged.
    ///
    /// SEAM(coord-convention): verify on-device against a known scene. If the
    /// reconstruction comes out mirrored or upside-down, the fix is to negate
    /// columns 1 and 2 of the rotation (flip Y and Z) here, NOT to guess blindly.
    private static func transformMatrixRows(_ m: simd_float4x4) -> [[Double]] {
        // simd_float4x4 is column-major: m[col][row]. Emit rows.
        (0..<4).map { row in
            (0..<4).map { col in Double(m[col][row]) }
        }
    }

    private static func datasetsDirectory() throws -> URL {
        // Caches: this is regenerable intermediate data, not a user document.
        let base = try FileManager.default.url(for: .cachesDirectory,
                                               in: .userDomainMask,
                                               appropriateFor: nil,
                                               create: true)
        let dir = base.appendingPathComponent("SplatDatasets", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
