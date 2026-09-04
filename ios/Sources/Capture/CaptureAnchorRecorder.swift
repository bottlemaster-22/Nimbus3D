//
//  CaptureAnchorRecorder.swift
//  Capture
//
//  ARKit anchors, logged as they appear AND re-read once the session has
//  ended (F8).
//
//  WHY THE RE-READ IS FREE DRIFT MEASUREMENT. ARKit silently moves anchors
//  when it relocalises or closes a loop internally: an anchor stuck to a
//  doorframe at second 12 may sit 4 cm from where it was once the walk comes
//  back round at second 200. Nothing announces that. But the anchor's
//  transform at the end of the session is ARKit's final, best-informed opinion
//  of where that doorframe is, and the transform logged when it first appeared
//  is the opinion it held at the time. The difference between the two is drift,
//  measured directly, with no extra computation and no assumptions - which is
//  why `CaptureBundle` carries both lists and why the QC card can put a number
//  in centimetres in front of the user.
//
//  Both files are `[AnchorRecord]` JSON, per docs/DATA_FORMAT.md section 6.
//

import ARKit
import Foundation
import simd

/// Keeps the live anchor log and produces the end-of-session re-read.
///
/// `@MainActor` because it is only ever fed from `ARSessionDelegate` callbacks,
/// which ARKit delivers on the main queue unless a different queue is set, and
/// this module does not set one.
@MainActor
final class CaptureAnchorRecorder {

    /// Anchors as first seen, keyed so a re-appearance does not overwrite the
    /// original observation - the original is precisely the thing being
    /// compared against.
    private var live: [UUID: AnchorRecord] = [:]

    /// Frame index at the moment of the callback, so `firstSeenFrame` is real.
    private var currentFrameIndex: FrameID = 0

    func reset() {
        live.removeAll()
        currentFrameIndex = 0
    }

    func noteFrameIndex(_ index: FrameID) {
        currentFrameIndex = index
    }

    // MARK: - Live log

    /// Records anchors ARKit has just added. Called from
    /// `session(_:didAdd:)`.
    func anchorsWereAdded(_ anchors: [ARAnchor]) {
        for anchor in anchors where live[anchor.identifier] == nil {
            live[anchor.identifier] = Self.record(
                for: anchor,
                firstSeenFrame: currentFrameIndex
            )
        }
    }

    /// Records anchors ARKit has removed. They stay in the log: an anchor that
    /// existed and was withdrawn is evidence about the session, and deleting
    /// the row would quietly shrink the drift measurement's sample.
    func anchorsWereRemoved(_ anchors: [ARAnchor]) {
        // Nothing to do beyond keeping the record. The method exists so the
        // delegate has somewhere honest to call and so this comment has
        // somewhere to live.
        _ = anchors
    }

    /// The live log, ordered so two runs produce byte-identical JSON.
    var sessionAnchors: [AnchorRecord] {
        live.values.sorted { $0.identifier.uuidString < $1.identifier.uuidString }
    }

    // MARK: - End-of-session re-read

    /// Re-reads every anchor still in the session, matching the live log by
    /// identifier.
    ///
    /// Call this while the session is still RUNNING, immediately before
    /// pausing it. After `ARSession.pause()` the current frame is gone and
    /// there is nothing left to read.
    func rereadAnchors(from session: ARSession) -> [AnchorRecord] {
        guard let frame = session.currentFrame else {
            let message = "No current frame at the end of the session, so anchors could "
                    + "not be re-read. The bundle records an empty final list "
                    + "rather than a copy of the live one."
            CaptureLog.session.notice("\(message, privacy: .public)")
            return []
        }
        return frame.anchors
            .map { anchor in
                Self.record(
                    for: anchor,
                    firstSeenFrame: live[anchor.identifier]?.firstSeenFrame
                )
            }
            .sorted { $0.identifier.uuidString < $1.identifier.uuidString }
    }

    // MARK: - Writing

    /// `anchors/anchors_session.json` and `anchors/anchors_final.json`.
    static func write(
        session sessionAnchors: [AnchorRecord],
        final finalAnchors: [AnchorRecord],
        to folder: CaptureScanFolder
    ) throws {
        let encoder = ContractsJSON.encoder()
        try encoder.encode(sessionAnchors).write(
            to: folder.anchorsDirectory
                .appendingPathComponent("anchors_session.json"),
            options: .atomic
        )
        try encoder.encode(finalAnchors).write(
            to: folder.anchorsDirectory
                .appendingPathComponent("anchors_final.json"),
            options: .atomic
        )
    }

    // MARK: - Bridging

    private static func record(
        for anchor: ARAnchor,
        firstSeenFrame: FrameID?
    ) -> AnchorRecord {
        AnchorRecord(
            identifier: anchor.identifier,
            transform: flatten(anchor.transform),
            firstSeenFrame: firstSeenFrame,
            classification: classification(of: anchor),
            isUserMarked: anchor.name == userMarkedWindowAnchorName
        )
    }

    /// 16 floats, COLUMN-MAJOR, ARKit's own convention, unmodified.
    /// docs/DATA_FORMAT.md section 6.
    static func flatten(_ matrix: simd_float4x4) -> [Float] {
        [
            matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z, matrix.columns.0.w,
            matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z, matrix.columns.1.w,
            matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z, matrix.columns.2.w,
            matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z, matrix.columns.3.w,
        ]
    }

    /// The name given to an anchor the USER placed by tapping "that is a
    /// window" in window mode. A plain `ARAnchor` carries no classification of
    /// its own, so the name is the only place the label can live.
    static let userMarkedWindowAnchorName = "user_marked_window"

    /// What kind of surface this anchor sits on, when ARKit will say.
    ///
    /// A plane anchor publishes a classification directly. A mesh anchor does
    /// not - its classification is per FACE - so the anchor-level answer is the
    /// most common class among its faces, which is the honest summary of a
    /// chunk that is mostly one thing.
    ///
    /// The one non-ARKit source is a window the user marked by hand. Such an
    /// anchor is classified `window` exactly like an ARKit-classified one (the
    /// person holding the phone is at least as reliable a witness as the
    /// classifier), and `AnchorRecord.isUserMarked` records which of the two
    /// said so.
    private static func classification(of anchor: ARAnchor) -> SurfaceClass {
        if anchor.name == userMarkedWindowAnchorName {
            return .window
        }
        if let plane = anchor as? ARPlaneAnchor {
            return SurfaceClass(plane.classification)
        }
        if let mesh = anchor as? ARMeshAnchor {
            return dominantClass(of: mesh.geometry)
        }
        return .none
    }

    private static func dominantClass(of geometry: ARMeshGeometry) -> SurfaceClass {
        guard let classification = geometry.classification else { return .none }
        let count = classification.count
        guard count > 0 else { return .none }
        var histogram = [Int](repeating: 0, count: SurfaceClass.allCases.count)
        let base = classification.buffer.contents()
            .advanced(by: classification.offset)
            .assumingMemoryBound(to: UInt8.self)
        let stride = classification.stride
        for face in 0..<count {
            let raw = base.advanced(by: face * stride).pointee
            let surface = SurfaceClass(arkitMeshClassificationRawValue: raw)
            if let index = SurfaceClass.allCases.firstIndex(of: surface) {
                histogram[index] += 1
            }
        }
        // `.none` is excluded from the vote: a chunk that is 60% unclassified
        // and 40% wall is a wall as far as any consumer is concerned.
        var best: (SurfaceClass, Int) = (.none, 0)
        for (index, value) in histogram.enumerated()
        where SurfaceClass.allCases[index] != .none {
            if value > best.1 { best = (SurfaceClass.allCases[index], value) }
        }
        return best.0
    }
}

// MARK: - ARKit classification bridging

extension SurfaceClass {

    /// Maps `ARPlaneAnchor.Classification` onto the format's enum.
    init(_ classification: ARPlaneAnchor.Classification) {
        switch classification {
        case .wall: self = .wall
        case .floor: self = .floor
        case .ceiling: self = .ceiling
        case .table: self = .table
        case .seat: self = .seat
        case .window: self = .window
        case .door: self = .door
        // `.none` carries a `Status` (not available / undetermined / unknown).
        // All three mean the same thing to this format: ARKit would not say.
        case .none: self = .none
        @unknown default: self = .none
        }
    }

    /// Maps an `ARMeshClassification` raw value onto the format's enum.
    ///
    /// Deliberately compares raw values rather than switching over
    /// `ARMeshClassification` cases: this is also the decoder for the bytes
    /// read back out of a `.cls` file, where all that survives is a number, and
    /// `ARMeshClassification.none` colliding with `Optional.none` in a switch
    /// is a well-known way to write something that compiles and means the
    /// wrong thing. The raw values come from ARKit itself, not from a table
    /// copied into this file.
    init(arkitMeshClassificationRawValue raw: UInt8) {
        let value = Int(raw)
        switch value {
        case ARMeshClassification.wall.rawValue: self = .wall
        case ARMeshClassification.floor.rawValue: self = .floor
        case ARMeshClassification.ceiling.rawValue: self = .ceiling
        case ARMeshClassification.table.rawValue: self = .table
        case ARMeshClassification.seat.rawValue: self = .seat
        case ARMeshClassification.window.rawValue: self = .window
        case ARMeshClassification.door.rawValue: self = .door
        default: self = .none
        }
    }

    /// The byte written into `mesh/chunk_NNNN.cls`: the index into
    /// `SurfaceClass.allCases`, exactly as docs/DATA_FORMAT.md section 6
    /// tabulates it (`0 none  1 wall  2 floor  3 ceiling  4 table  5 seat
    /// 6 window  7 door  8 glass  9 sky`).
    var sidecarByte: UInt8 {
        UInt8(SurfaceClass.allCases.firstIndex(of: self) ?? 0)
    }
}
