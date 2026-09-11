//
//  BoosterResultsSection.swift
//  Booster
//
//  History of jobs sent to Boosters: what stage each one reached, and, for
//  finished ones, where the result landed on this phone.
//

import SwiftUI

struct BoosterResultsSection: View {
    let jobs: [BoosterJobRecord]

    var body: some View {
        if !jobs.isEmpty {
            Section("Results") {
                ForEach(jobs) { job in
                    HStack(spacing: 12) {
                        Image(systemName: icon(for: job.stage))
                            .foregroundStyle(color(for: job.stage))
                            .frame(width: 20)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(job.scanID)
                                .font(.body)
                            Text(job.message)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                        Spacer()
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    private func icon(for stage: BoosterJobStage) -> String {
        switch stage {
        case .ready: return "checkmark.circle.fill"
        case .failed: return "xmark.circle.fill"
        case .cancelled: return "minus.circle.fill"
        case .queued, .receiving, .verifying, .training, .exporting:
            return "clock.fill"
        }
    }

    private func color(for stage: BoosterJobStage) -> Color {
        switch stage {
        case .ready: return .green
        case .failed: return .red
        case .cancelled: return .gray
        case .queued, .receiving, .verifying, .training, .exporting:
            return .orange
        }
    }
}
