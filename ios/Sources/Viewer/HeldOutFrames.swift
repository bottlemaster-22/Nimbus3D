//
//  HeldOutFrames.swift
//  Viewer
//
//  WHICH PHOTOS THE PHOTO-VERSUS-SCAN SLIDER IS ALLOWED TO USE.
//
//  A comparison against a photo the model was trained on proves nothing: the
//  model was optimised until it matched that photo, so of course it matches
//  it. The only comparison worth showing a user is against frames that were
//  HELD OUT of training, which for this project is roughly 5% of them.
//
//  Two sources, in order of how much they can be trusted, and the UI says
//  which one it used:
//
//   1. `model/held_out_frames.json`, written by whoever trained the model.
//      This is the real answer. When it is there, the slider can honestly say
//      the model never saw these photos.
//
//   2. Failing that, every 20th frame, chosen the same way every time. That
//      is 5% of the capture and it is deterministic, so the same photos come
//      up each time the scan is opened. It is NOT proof of anything, because
//      nothing recorded what the training run actually set aside, and the
//      copy on screen says exactly that instead of implying more.
//
//  There is deliberately no third option that quietly falls back to training
//  frames. If a scan has too few frames to hold any back, the slider says so
//  and shows nothing.
//

import Foundation

// MARK: - The set

/// The frames the A/B slider may compare against, plus where the list came
/// from, so the screen can be honest about how much it proves.
struct HeldOutFrameSet: Sendable {

    enum Source: Sendable, Equatable {
        /// A list the trainer wrote down. Relative path, for the caption.
        case recordedByTrainer(String)
        /// Every Nth frame, picked by this app.
        case everyNthFrame(Int)
    }

    var frames: [CaptureFrame]
    var source: Source
    var totalFrameCount: Int

    var isEmpty: Bool { frames.isEmpty }

    /// Share of the capture that is in this set, 0...1.
    var fraction: Float {
        totalFrameCount > 0 ? Float(frames.count) / Float(totalFrameCount) : 0
    }

    /// One or two plain sentences under the slider. Never overclaims.
    var explanation: String {
        switch source {
        case .recordedByTrainer(let path):
            return "These \(frames.count) photos were set aside before training, "
                + "so the 3D model never got to copy them. "
                + "The list came from the model itself (\(path))."
        case .everyNthFrame(let n):
            return "Every \(HeldOutFrameSelector.ordinal(n)) photo from the walk, "
                + "\(frames.count) of them (about \(ViewerFormat.percent(fraction)) "
                + "of the scan), picked the same way every time you open this scan. "
                + "The training run did not write down which photos it kept back, "
                + "so treat this as a spot check rather than proof."
        }
    }

    /// How much weight the caption should carry. Used to pick the icon.
    var isTrustworthy: Bool {
        if case .recordedByTrainer = source { return true }
        return false
    }
}

// MARK: - Choosing

enum HeldOutFrameSelector {

    /// Every 20th frame is 5% of the capture, which is the share this project
    /// holds out. Changing this number changes which photos the slider shows,
    /// so it lives here, once, with a name.
    static let stride = 20

    /// Offset into each block of `stride`. Deliberately not 0: frame 0 of a
    /// handheld capture is almost always the floor or the ceiling while the
    /// user was still getting into position.
    static let offset = 10

    /// Name of the optional sidecar, relative to the scan root.
    static let sidecarRelativePath = "\(BrandConfig.Folder.model)/held_out_frames.json"

    /// Resolves the held-out set for a scan. Pure file reads; safe to call off
    /// the main actor, and cheap enough to call on it for a loaded bundle.
    static func resolve(bundle: CaptureBundle, paths: ViewerScanPaths) -> HeldOutFrameSet {
        let total = bundle.frames.count

        if let recorded = readSidecar(paths: paths) {
            let wanted = Set(recorded)
            let frames = bundle.frames.filter { wanted.contains($0.index) }
            if !frames.isEmpty {
                return HeldOutFrameSet(
                    frames: frames,
                    source: .recordedByTrainer(sidecarRelativePath),
                    totalFrameCount: total
                )
            }
        }

        return HeldOutFrameSet(
            frames: everyNth(bundle.frames),
            source: .everyNthFrame(stride),
            totalFrameCount: total
        )
    }

    /// Every `stride`-th frame, counted by position in the capture rather than
    /// by frame index, so a capture that dropped frames still yields an evenly
    /// spread 5%.
    static func everyNth(_ frames: [CaptureFrame]) -> [CaptureFrame] {
        guard frames.count > stride else { return [] }
        var chosen: [CaptureFrame] = []
        chosen.reserveCapacity(frames.count / stride + 1)
        var index = Swift.min(offset, frames.count - 1)
        while index < frames.count {
            chosen.append(frames[index])
            index += stride
        }
        return chosen
    }

    /// Reads `model/held_out_frames.json`. Accepts either a bare array of
    /// frame indices or an object with a `frames` array, because both spellings
    /// are plausible and refusing one on a technicality would lose real
    /// information the trainer went to the trouble of writing down.
    static func readSidecar(paths: ViewerScanPaths) -> [FrameID]? {
        let url = paths.url(sidecarRelativePath)
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = ContractsJSON.decoder()
        if let wrapped = try? decoder.decode(Sidecar.self, from: data) {
            return wrapped.frames
        }
        if let bare = try? decoder.decode([FrameID].self, from: data) {
            return bare
        }
        ViewerLog.review.notice(
            "held_out_frames.json is present but not in a shape this app understands"
        )
        return nil
    }

    private struct Sidecar: Codable {
        var frames: [FrameID]
    }

    /// "20" -> "20th". Only ever used on small positive numbers.
    static func ordinal(_ n: Int) -> String {
        let tens = n % 100
        if tens >= 11 && tens <= 13 { return "\(n)th" }
        switch n % 10 {
        case 1: return "\(n)st"
        case 2: return "\(n)nd"
        case 3: return "\(n)rd"
        default: return "\(n)th"
        }
    }
}
