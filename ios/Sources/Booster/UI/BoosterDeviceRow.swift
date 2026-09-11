//
//  BoosterDeviceRow.swift
//  Booster
//
//  One row in the "Nearby Boosters" list: name, reachability, and the one
//  action that makes sense for its current state (Pair / Send a scan).
//

import SwiftUI

struct BoosterDeviceRow: View {
    let device: BoosterDevice
    let isBusy: Bool
    let onPair: () -> Void
    let onSendScan: () -> Void
    /// Development only. Nil on any build without the diagnostics sender,
    /// and the button then does not exist rather than being greyed out.
    let onSendDiagnostics: (() -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)

            VStack(alignment: .leading, spacing: 2) {
                Text(device.name)
                    .font(.body.weight(.medium))
                Text(statusText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                // A BUTTON, NOT A SWIPE.
                //
                // This used to be a trailing swipe action, which meant the
                // only way to know it existed was to be told. The owner had
                // to be talked through finding it twice, which is the
                // definition of the wrong control. It is a development
                // feature, so it can afford to look plain, but it cannot
                // afford to be invisible.
                //
                // `.bordered` rather than the automatic style on purpose:
                // a List row containing more than one button needs an
                // explicit style, or a tap anywhere on the row fires all of
                // them.
                if let sendDiagnostics = onSendDiagnostics, device.isPaired {
                    Button("Send diagnostics", action: sendDiagnostics)
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(!device.isReachable || isBusy)
                        .padding(.top, 4)
                }
            }

            Spacer()

            actionButton
        }
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        if !device.isReachable { return .gray }
        return device.isPaired ? .green : .orange
    }

    private var statusText: String {
        switch (device.isPaired, device.isReachable) {
        case (true, true): return "Paired - ready to send"
        case (true, false): return "Paired, but not on your Wi-Fi right now"
        case (false, true): return "Found on your Wi-Fi - not paired yet"
        case (false, false): return "Not available"
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        if device.isPaired {
            Button("Send a scan", action: onSendScan)
                .buttonStyle(.borderedProminent)
                .disabled(!device.isReachable || isBusy)
        } else {
            Button("Pair", action: onPair)
                .buttonStyle(.bordered)
                .disabled(!device.isReachable || isBusy)
        }
    }
}
