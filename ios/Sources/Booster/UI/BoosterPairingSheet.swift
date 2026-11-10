//
//  BoosterPairingSheet.swift
//  Booster
//
//  Walks the user through pairing with one Booster: "Look at your computer
//  screen" -> type the code shown there -> confirmed.
//

import SwiftUI

struct BoosterPairingSheet: View {
    let device: BoosterDevice
    @ObservedObject var pairing: BoosterPairingManager
    let onFinished: () -> Void

    @State private var code = ""
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                switch pairing.state {
                case .idle, .requesting:
                    ProgressView("Reaching \(device.name)...")
                        .padding(.top, 40)

                case .awaitingCode:
                    VStack(spacing: 16) {
                        Image(systemName: "desktopcomputer")
                            .font(.system(size: 44))
                            .foregroundStyle(.secondary)
                        Text("Look at \(device.name)'s screen")
                            .font(.title3.weight(.semibold))
                        Text("A short code should be showing there. Type it below to finish pairing.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)

                        TextField("Code", text: $code)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                            .multilineTextAlignment(.center)
                            .font(.title2.monospacedDigit())
                            .frame(width: 160)
                            .padding(.top, 8)

                        Button("Confirm") {
                            Task { await pairing.confirm(code: code) }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(code.isEmpty)
                    }
                    .padding(.top, 24)

                case .confirming:
                    ProgressView("Confirming...")
                        .padding(.top, 40)

                case .paired(_, let boosterName):
                    VStack(spacing: 16) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(.green)
                        Text("Paired with \(boosterName)")
                            .font(.title3.weight(.semibold))
                        Button("Done") {
                            onFinished()
                            dismiss()
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding(.top, 40)

                case .failed(let message):
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(.orange)
                        Text(message)
                            .font(.subheadline)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        Button("Try again") {
                            pairing.reset()
                        }
                        .buttonStyle(.bordered)
                    }
                    .padding(.top, 40)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Pair with \(device.name)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
