//
//  CaptureMeshStore.swift
//  Capture
//
//  ARKit's classified scene mesh, copied out of ARKit and held in plain
//  arrays.
//
//  WHY A SNAPSHOT RATHER THAN THE LIVE ANCHORS. `ARMeshAnchor.geometry` is
//  backed by `MTLBuffer`s that ARKit owns and rewrites; the vertex count of a
//  chunk changes between frames, vertices are renumbered, and the whole anchor
//  can be replaced. Reading those buffers is only safe on the thread ARKit
//  hands them to you on, and only for as long as that callback lasts. The
//  coverage field, the window-mode detector and the HUD renderer all want the
//  mesh, all on different queues, at different rates. So: one copy, made on the
//  delegate thread, versioned, and shared read-only.
//
//  It also carries the CLASS VOXEL MAP - "what kind of surface is at this
//  point in the world" at the coverage grid's resolution. That is what lets a
//  depth sample be attributed to a window without raycasting the mesh per
//  sample, and it is what drives window mode.
//

import ARKit
import Foundation
import simd

/// One chunk of scene mesh, in world space, owned by this module.
struct CaptureMeshSnapshot: Sendable {
    let identifier: UUID
    /// Bumped every time the chunk is re-copied, so the renderer knows whether
    /// its GPU buffers are stale without comparing megabytes.
    let version: Int
    /// World-space positions, one per vertex.
    let positions: [SIMD3<Float>]
    /// World-space normals, one per vertex.
    let normals: [SIMD3<Float>]
    /// Triangle indices, three per face.
    let indices: [UInt32]
    /// One `SurfaceClass.sidecarByte` per FACE, in the same order as `indices`
    /// triples. docs/DATA_FORMAT.md section 6.
    let faceClasses: [UInt8]
    let bounds: BoundingBox

    var faceCount: Int { faceClasses.count }
}

/// Holds every chunk's latest snapshot plus the derived class voxel map.
///
/// Thread-safe. Written from the ARKit delegate thread, read by the pipeline
/// queue (coverage, window mode) and the render thread (HUD).
final class CaptureMeshStore: @unchecked Sendable {

    private let lock = NSLock()
    private var snapshots: [UUID: CaptureMeshSnapshot] = [:]
    private var versionCounter = 0

    /// World voxel -> surface class, at the coverage grid's resolution. Coarse
    /// on purpose: this answers "is the thing in front of me glass", not "which
    /// triangle is this".
    private var classVoxels: [Int64: SurfaceClass] = [:]

    // MARK: - Update

    /// Copies one `ARMeshAnchor` into a snapshot.
    ///
    /// MUST be called on the thread ARKit delivered the anchor on, while the
    /// anchor is still alive. Everything it touches is copied before it
    /// returns.
    func update(from anchor: ARMeshAnchor) {
        let geometry = anchor.geometry
        let vertexCount = geometry.vertices.count
        let faceCount = geometry.faces.count
        guard vertexCount > 0, faceCount > 0 else { return }

        let transform = anchor.transform
        let rotation = simd_float3x3(
            simd_make_float3(transform.columns.0),
            simd_make_float3(transform.columns.1),
            simd_make_float3(transform.columns.2)
        )

        var positions = [SIMD3<Float>](repeating: .zero, count: vertexCount)
        var normals = [SIMD3<Float>](repeating: SIMD3<Float>(0, 1, 0), count: vertexCount)
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        let vertexSource = geometry.vertices
        let vertexBase = vertexSource.buffer.contents()
            .advanced(by: vertexSource.offset)
        for index in 0..<vertexCount {
            let raw = vertexBase.advanced(by: index * vertexSource.stride)
                .assumingMemoryBound(to: Float.self)
            let local = SIMD3<Float>(raw[0], raw[1], raw[2])
            let world = transform * SIMD4<Float>(local, 1)
            let point = SIMD3<Float>(world.x, world.y, world.z)
            positions[index] = point
            minimum = simd_min(minimum, point)
            maximum = simd_max(maximum, point)
        }

        let normalSource = geometry.normals
        if normalSource.count == vertexCount {
            let normalBase = normalSource.buffer.contents()
                .advanced(by: normalSource.offset)
            for index in 0..<vertexCount {
                let raw = normalBase.advanced(by: index * normalSource.stride)
                    .assumingMemoryBound(to: Float.self)
                let local = SIMD3<Float>(raw[0], raw[1], raw[2])
                let world = rotation * local
                let length = simd_length(world)
                normals[index] = length > 1e-6 ? world / length : SIMD3<Float>(0, 1, 0)
            }
        }

        // Face indices. ARKit uses 4-byte indices in practice, but the element
        // publishes its own width and this reads whatever it says rather than
        // assuming.
        let faces = geometry.faces
        guard faces.indexCountPerPrimitive == 3 else {
            let message = "Scene mesh element is not triangles "
                    + "(\(faces.indexCountPerPrimitive) indices "
                    + "per primitive); this chunk is skipped rather than "
                    + "reinterpreted."
            CaptureLog.session.error("\(message, privacy: .public)")
            return
        }
        var indices = [UInt32](repeating: 0, count: faceCount * 3)
        let faceBase = faces.buffer.contents()
        switch faces.bytesPerIndex {
        case 4:
            let typed = faceBase.assumingMemoryBound(to: UInt32.self)
            for slot in 0..<(faceCount * 3) { indices[slot] = typed[slot] }
        case 2:
            let typed = faceBase.assumingMemoryBound(to: UInt16.self)
            for slot in 0..<(faceCount * 3) { indices[slot] = UInt32(typed[slot]) }
        default:
            let message = "Unexpected mesh index width "
                    + "\(faces.bytesPerIndex); chunk skipped."
            CaptureLog.session.error("\(message, privacy: .public)")
            return
        }

        var faceClasses = [UInt8](repeating: SurfaceClass.none.sidecarByte, count: faceCount)
        if let classification = geometry.classification,
            classification.count == faceCount
        {
            let base = classification.buffer.contents()
                .advanced(by: classification.offset)
            for face in 0..<faceCount {
                let raw = base.advanced(by: face * classification.stride)
                    .assumingMemoryBound(to: UInt8.self).pointee
                faceClasses[face] = SurfaceClass(
                    arkitMeshClassificationRawValue: raw
                ).sidecarByte
            }
        }

        let snapshot = CaptureMeshSnapshot(
            identifier: anchor.identifier,
            version: nextVersion(),
            positions: positions,
            normals: normals,
            indices: indices,
            faceClasses: faceClasses,
            bounds: BoundingBox(min: Vector3(minimum), max: Vector3(maximum))
        )

        lock.lock()
        snapshots[anchor.identifier] = snapshot
        lock.unlock()

        indexClasses(of: snapshot)
    }

    func remove(identifier: UUID) {
        lock.lock()
        snapshots.removeValue(forKey: identifier)
        lock.unlock()
        // The class voxels this chunk contributed are deliberately left in
        // place. ARKit removes and re-adds mesh anchors as it re-chunks the
        // room; clearing the classification every time would make window mode
        // flicker for reasons the user cannot see or act on.
    }

    func reset() {
        lock.lock()
        snapshots.removeAll(keepingCapacity: false)
        classVoxels.removeAll(keepingCapacity: false)
        versionCounter = 0
        lock.unlock()
    }

    // MARK: - Read

    var allSnapshots: [CaptureMeshSnapshot] {
        lock.lock()
        defer { lock.unlock() }
        return Array(snapshots.values)
    }

    var chunkCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return snapshots.count
    }

    /// The surface class at a world position, or `.none` where nothing has
    /// been classified. Cheap: one dictionary lookup at the coverage grid's
    /// resolution.
    func surfaceClass(at worldPosition: SIMD3<Float>) -> SurfaceClass {
        lock.lock()
        defer { lock.unlock() }
        return classVoxels[CaptureCoverageField.key(for: worldPosition)] ?? .none
    }

    /// Bulk class lookup, one lock acquisition for a whole vertex batch.
    func classBatch(
        positions: UnsafeBufferPointer<SIMD3<Float>>,
        into output: UnsafeMutableBufferPointer<UInt8>
    ) {
        lock.lock()
        defer { lock.unlock() }
        let count = Swift.min(positions.count, output.count)
        for index in 0..<count {
            let key = CaptureCoverageField.key(for: positions[index])
            output[index] = (classVoxels[key] ?? .none).sidecarByte
        }
    }

    /// World extent of the whole mesh, or nil when nothing has been meshed.
    func bounds() -> BoundingBox? {
        lock.lock()
        defer { lock.unlock() }
        guard !snapshots.isEmpty else { return nil }
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for snapshot in snapshots.values {
            minimum = simd_min(minimum, snapshot.bounds.min.simd)
            maximum = simd_max(maximum, snapshot.bounds.max.simd)
        }
        return BoundingBox(min: Vector3(minimum), max: Vector3(maximum))
    }

    // MARK: - Private

    private func nextVersion() -> Int {
        lock.lock()
        versionCounter += 1
        let version = versionCounter
        lock.unlock()
        return version
    }

    /// Stamps every classified face's centroid into the class voxel map.
    ///
    /// Later writes win, and that is the right default: ARKit's classifier
    /// improves as it sees more of a surface, so the most recent opinion about
    /// a patch is its best one. `.none` never overwrites a real class, because
    /// "I no longer have an opinion" is not evidence that the window stopped
    /// being a window.
    private func indexClasses(of snapshot: CaptureMeshSnapshot) {
        var updates: [Int64: SurfaceClass] = [:]
        updates.reserveCapacity(snapshot.faceCount)
        let all = SurfaceClass.allCases
        for face in 0..<snapshot.faceCount {
            let byte = Int(snapshot.faceClasses[face])
            guard byte > 0, byte < all.count else { continue }
            let surface = all[byte]
            let a = Int(snapshot.indices[face * 3])
            let b = Int(snapshot.indices[face * 3 + 1])
            let c = Int(snapshot.indices[face * 3 + 2])
            guard
                a < snapshot.positions.count,
                b < snapshot.positions.count,
                c < snapshot.positions.count
            else { continue }
            let centroid =
                (snapshot.positions[a] + snapshot.positions[b]
                    + snapshot.positions[c]) / 3
            updates[CaptureCoverageField.key(for: centroid)] = surface
        }
        guard !updates.isEmpty else { return }
        lock.lock()
        for (key, value) in updates { classVoxels[key] = value }
        lock.unlock()
    }
}
