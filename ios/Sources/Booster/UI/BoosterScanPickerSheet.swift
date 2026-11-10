//
//  BoosterScanPickerSheet.swift
//  Booster
//
//  "Which scan do you want to send?" - lists scans found on this phone (via
//  BoosterScanLister) and hands the chosen one back to the caller.
//

import SwiftUI

struct BoosterScanPickerSheet: View {
    let device: BoosterDevice
    let onPick: (BoosterScanSummary) -> Void

    @State private var scans: [BoosterScanSummary] = []
    @Environment(\.dismiss) private var dismiss

    private static let byteFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter
    }()

    var body: some View {
        NavigationStack {
            Group {
                if scans.isEmpty {
                    ContentUnavailableView(
                        "No scans yet",
                        systemImage: "cube.transparent",
                        description: Text("Finish a scan first, then send it to \(device.name).")
                    )
                } else {
                    List(scans) { scan in
                        Button {
                            onPick(scan)
                            dismiss()
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(scan.scanID)
                                        .font(.body)
                                        .foregroundStyle(.primary)
                                    Text(
                                        "\(scan.fileCount) files, "
                                            + Self.byteFormatter.string(
                                                fromByteCount: scan.totalByteCount
                                            )
                                    )
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .font(.footnote)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Send to \(device.name)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onAppear {
                scans = BoosterScanLister.listAvailableScans()
            }
        }
    }
}
