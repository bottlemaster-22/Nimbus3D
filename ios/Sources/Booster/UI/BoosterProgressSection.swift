//
//  BoosterProgressSection.swift
//  Booster
//
//  Shows exactly one of: uploading, the Booster's own live progress (queued
//  / receiving / verifying / training / exporting), or downloading the
//  result back - whichever is currently happening - plus a Cancel button.
//

import SwiftUI

struct BoosterProgressSection: View {
    @ObservedObject var client: BoosterClient

    var body: some View {
        if client.isBusy || client.activeUpload != nil || client.activeJobProgress != nil
            || client.activeDownload != nil
        {
            Section("Sending") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(stageLabel)
                        .font(.subheadline.weight(.medium))
                    if let fraction {
                        ProgressView(value: fraction)
                    } else {
                        // No meaningful percentage yet (e.g. the Booster is
                        // still queued or verifying). A determinate bar stuck
                        // at 0% reads as frozen; a spinner reads as "working".
                        ProgressView()
                    }
                    if let detail {
                        Text(detail)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Button("Cancel", role: .destructive) {
                        client.cancelActiveSend()
                    }
                    .padding(.top, 4)
                }
                .padding(.vertical, 4)
            }
        }
    }

    private var stageLabel: String {
        if let upload = client.activeUpload {
            return "Sending \"\(upload.currentFileName)\" to your Booster"
        }
        if let progress = client.activeJobProgress {
            return progress.message
        }
        if let download = client.activeDownload {
            return "Bringing back \"\(download.currentFileName)\""
        }
        return "Getting ready..."
    }

    /// nil when the current stage has no meaningful percentage yet (e.g. the
    /// Booster reports "queued" or "verifying" with no fractionComplete) -
    /// distinct from 0%, which means "just started, but we know the size".
    private var fraction: Double? {
        if let upload = client.activeUpload { return upload.fraction }
        if let progress = client.activeJobProgress { return progress.fractionComplete }
        if let download = client.activeDownload { return download.fraction }
        return nil
    }

    private var detail: String? {
        if let upload = client.activeUpload {
            return Self.percentString(upload.fraction) + " sent"
        }
        if let download = client.activeDownload {
            return Self.percentString(download.fraction) + " received"
        }
        return nil
    }

    private static func percentString(_ fraction: Double) -> String {
        "\(Int((fraction * 100).rounded()))%"
    }
}
