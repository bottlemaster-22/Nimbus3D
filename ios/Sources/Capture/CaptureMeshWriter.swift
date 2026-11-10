//
//  CaptureMeshWriter.swift
//  Capture
//
//  `mesh/chunk_NNNN.ply`, `mesh/chunk_NNNN.cls` and `mesh/mesh_index.json`,
//  exactly as docs/DATA_FORMAT.md section 6 defines them.
//
//  The PLY is binary little-endian with `vertex` (x, y, z, nx, ny, nz) and
//  `face`, in WORLD space. The `.cls` file is one byte per face in the same
//  order as the PLY's face list, holding the index into `SurfaceClass.allCases`
//  (`0 none  1 wall  2 floor  3 ceiling  4 table  5 seat  6 window  7 door
//  8 glass  9 sky`). Classes 8 and 9 are ours, not ARKit's; nothing in this
//  module writes them, because glass detection is the pre-pass's job (F5) and
//  guessing here would put an unearned label in the file.
//
//  This is NOT `Sources/Export`'s `PLYCodec`. That codec reads and writes
//  GAUSSIAN SPLAT plys (`f_dc_*`, `opacity`, `scale_*`, `rot_*`) in the INRIA
//  convention. This writes a triangle mesh. Sharing one type between the two
//  would mean a format with two disjoint halves and a mode flag, which is how
//  a reader ends up silently parsing the wrong one.
//

import Foundation
import simd

/// Writes the scene mesh chunks for a capture.
enum CaptureMeshWriter {

    /// Writes every snapshot as a chunk and returns the index the bundle
    /// carries.
    ///
    /// Chunks are ordered by identifier so two runs over the same session
    /// produce the same file names for the same geometry.
    static func write(
        snapshots: [CaptureMeshSnapshot],
        to folder: CaptureScanFolder
    ) throws -> [MeshChunkRef] {
        let ordered = snapshots.sorted {
            $0.identifier.uuidString < $1.identifier.uuidString
        }

        var refs: [MeshChunkRef] = []
        refs.reserveCapacity(ordered.count)

        for (index, snapshot) in ordered.enumerated() {
            let name = String(format: "chunk_%04d", index)
            let geometryURL = folder.meshDirectory
                .appendingPathComponent("\(name).ply")
            let classificationURL = folder.meshDirectory
                .appendingPathComponent("\(name).cls")

            try plyData(for: snapshot).write(to: geometryURL, options: .atomic)
            try snapshot.faceClasses
                .withUnsafeBufferPointer { Data(buffer: $0) }
                .write(to: classificationURL, options: .atomic)

            refs.append(
                MeshChunkRef(
                    geometryPath: "\(BrandConfig.Folder.mesh)/\(name).ply",
                    classificationPath: "\(BrandConfig.Folder.mesh)/\(name).cls",
                    faceCount: snapshot.faceCount,
                    bounds: snapshot.bounds
                )
            )
        }

        let indexURL = folder.meshDirectory
            .appendingPathComponent("mesh_index.json")
        try ContractsJSON.encoder().encode(refs).write(
            to: indexURL,
            options: .atomic
        )

        return refs
    }

    // MARK: - PLY

    /// Binary little-endian PLY for one chunk.
    ///
    /// Face indices are written as `int` (32-bit signed) after a `uchar` count
    /// of 3, which is the layout every PLY reader in existence expects from
    /// `property list uchar int vertex_indices`. Writing `uint` there is legal
    /// PLY and is read wrongly by enough tools to be a bad idea.
    static func plyData(for snapshot: CaptureMeshSnapshot) -> Data {
        let vertexCount = snapshot.positions.count
        let faceCount = snapshot.faceCount

        let header = """
            ply
            format binary_little_endian 1.0
            comment world space, right-handed, Y up, metres
            comment per-face SurfaceClass bytes are in the sibling .cls file
            element vertex \(vertexCount)
            property float x
            property float y
            property float z
            property float nx
            property float ny
            property float nz
            element face \(faceCount)
            property list uchar int vertex_indices
            end_header

            """

        var data = Data(header.utf8)
        data.reserveCapacity(
            data.count + vertexCount * 24 + faceCount * 13
        )

        for index in 0..<vertexCount {
            let position = snapshot.positions[index]
            let normal = index < snapshot.normals.count
                ? snapshot.normals[index]
                : SIMD3<Float>(0, 1, 0)
            appendFloat(position.x, to: &data)
            appendFloat(position.y, to: &data)
            appendFloat(position.z, to: &data)
            appendFloat(normal.x, to: &data)
            appendFloat(normal.y, to: &data)
            appendFloat(normal.z, to: &data)
        }

        for face in 0..<faceCount {
            data.append(3)
            appendInt32(Int32(snapshot.indices[face * 3]), to: &data)
            appendInt32(Int32(snapshot.indices[face * 3 + 1]), to: &data)
            appendInt32(Int32(snapshot.indices[face * 3 + 2]), to: &data)
        }

        return data
    }

    // MARK: - Little-endian primitives

    @inline(__always)
    private static func appendFloat(_ value: Float, to data: inout Data) {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }

    @inline(__always)
    private static func appendInt32(_ value: Int32, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
}
