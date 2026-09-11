//
//  DiagnosticsSender.swift
//  Booster
//
//  PUSHING A SCAN'S DIAGNOSTIC FILES TO THE PC BOOSTER, FOR A HUMAN TO READ.
//
//  DEVELOPMENT ONLY, AND DELIBERATELY EASY TO REMOVE. This file plus one
//  Section in BoosterTabView is the whole feature, and nothing else refers to
//  either. On the PC side it is one route and one handler.
//
//  WHY IT EXISTS. Every measurement that has actually moved this project
//  forward came from a file on the phone: train_census.json said densification
//  was fine and binarisation was eating the model, an exported PLY said half
//  the population was invisible, and a screen recording said the crash was in
//  the trainer rather than the pre-pass. Getting each one required exporting by
//  hand, finding it in the Files app, and moving it to a computer. The owner,
//  reasonably: "My downloads folder is getting VERY bloated from all of these
//  file transfers."
//
//  So the phone sends them itself, to the machine that is already paired with
//  it, over the LAN it is already on.
//
//  WHAT IT IS NOT. It is not part of the boost protocol. It creates no job,
//  starts no training, and touches none of the manifest or chunking machinery.
//  Processing stays on the phone; this moves READ-ONLY copies of what the
//  processing already wrote. The Bottle relay is not involved either: that is
//  for installing the app, and a Booster on your own network should never need
//  a machine on the internet to accept a file from your own phone.
//
//  WHAT IT SENDS. A fixed list, plus anything else asked for later, because the
//  useful file is rarely the one you predicted. Files that do not exist are
//  skipped silently rather than reported as failures: a scan that was never
//  trained has no model census, and that is not an error.
//

import Foundation
import os

/// Sends a scan's diagnostic files to the paired PC Booster.
@MainActor
public final class DiagnosticsSender: ObservableObject {

    /// One file to try to send, and the name it lands under.
    ///
    /// The relative path is used verbatim on the PC, so a sent tree mirrors the
    /// scan folder and two scans never collide.
    private static let wanted: [String] = [
        "capture_bundle.json",
        "model/train_census.json",
        "model/model.json",
        "model/model.ply",
        "model/held_out_frames.json",
        "prepass/prepass_result.json",
        "prepass/census.json",
        "prepass/revisits.json",
        "model/exposure.bin"
    ]

    @Published public private(set) var isSending = false
    @Published public private(set) var message: String?

    private let log = Logger(
        subsystem: BrandConfig.loggingSubsystem, category: "Booster.Diagnostics"
    )

    public init() {}

    /// Sends what exists of `wanted` for one scan.
    ///
    /// Reports what was sent and what was skipped, in one sentence, because the
    /// point of this feature is to stop a person having to go and look.
    public func send(scanID: String, scanDirectory: URL, to device: BoosterDevice) async {
        guard !isSending else { return }
        guard device.isPaired,
              let token = BoosterKeychain.loadToken(boosterID: device.id)
        else {
            message = "Pair with this Booster first."
            return
        }

        isSending = true
        message = nil
        defer { isSending = false }

        do {
            let (host, port) = try await BoosterClient.shared.discovery
                .resolveHostPort(for: device)
            let address = BoosterEndpointAddress(host: host, port: port, token: token)

            var sent = 0
            var bytes = 0
            var missing = 0

            for relative in Self.wanted {
                let source = scanDirectory.appendingPathComponent(relative)
                guard FileManager.default.fileExists(atPath: source.path) else {
                    missing += 1
                    continue
                }
                // Read whole. The largest of these is a model.ply at tens of
                // megabytes, which this phone has just finished holding in GPU
                // buffers several times over, so the copy is not the expensive
                // part of anything.
                let data = try Data(contentsOf: source, options: .mappedIfSafe)
                try await BoosterHTTP.putFile(
                    address,
                    path: BoosterAPI.diagnosticsFile(
                        scanID: scanID, relativePath: relative
                    ),
                    data: data
                )
                sent += 1
                bytes += data.count
                log.notice(
                    "Sent \(relative, privacy: .public) (\(data.count, privacy: .public) bytes)"
                )
            }

            let megabytes = Double(bytes) / 1_000_000
            message = String(
                format: "Sent %d file%@ (%.1f MB) to %@.%@",
                sent, sent == 1 ? "" : "s", megabytes, device.name,
                missing > 0 ? " \(missing) not made yet." : ""
            )
        } catch {
            message = "Could not send: \(error.localizedDescription)"
            log.error(
                "Diagnostics send failed: \(error.localizedDescription, privacy: .public)"
            )
        }
    }
}
