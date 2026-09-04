//
//  BoosterTabView.swift
//  Booster
//
//  The Booster tab: find a PC on the same Wi-Fi network, pair with it, send
//  a scan for it to finish training, watch it work, and see what came back.
//  Entirely optional - nothing else in the app depends on this screen ever
//  being opened.
//
//  Drop-in usage from whichever module owns the app's tab bar:
//
//      TabView {
//          ...
//          BoosterTabView()
//              .tabItem { Label("Booster", systemImage: "bolt.horizontal.circle") }
//      }
//

import SwiftUI

@MainActor
public struct BoosterTabView: View {

    @StateObject private var client = BoosterClient.shared
    @ObservedObject private var discovery = BoosterClient.shared.discovery
    @StateObject private var pairing = BoosterPairingManager()

    @State private var pairingDevice: BoosterDevice?
    @State private var scanPickerDevice: BoosterDevice?
    @State private var jobs: [BoosterJobRecord] = []
    @State private var localError: BoosterError?

    public init() {}

    public var body: some View {
        NavigationStack {
            List {
                Section {
                    Text(
                        "Send a large scan to a computer on your Wi-Fi to have it "
                            + "finish training. Everything stays on your own network - "
                            + "nothing goes to the internet."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                }

                Section {
                    if discovery.devices.isEmpty {
                        HStack {
                            if discovery.isSearching {
                                ProgressView()
                                Text("Looking for a Booster on your Wi-Fi...")
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("No Boosters found yet.")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    } else {
                        ForEach(discovery.devices) { device in
                            BoosterDeviceRow(
                                device: device,
                                isBusy: client.isBusy,
                                onPair: { startPairing(with: device) },
                                onSendScan: { scanPickerDevice = device }
                            )
                            .swipeActions(edge: .trailing) {
                                if device.isPaired {
                                    Button("Forget", role: .destructive) {
                                        client.forgetPairing(boosterID: device.id)
                                    }
                                }
                            }
                        }
                    }
                } header: {
                    Text("Nearby Boosters")
                }

                BoosterProgressSection(client: client)
                BoosterResultsSection(jobs: jobs)
            }
            .navigationTitle("Booster")
            .onAppear {
                discovery.start()
                jobs = client.resultsHistory()
            }
            .onDisappear {
                discovery.stop()
            }
            .onChange(of: client.activeJobProgress?.stage) { _, _ in
                jobs = client.resultsHistory()
            }
            .sheet(item: $pairingDevice) { device in
                BoosterPairingSheet(device: device, pairing: pairing) {
                    jobs = client.resultsHistory()
                }
            }
            .sheet(item: $scanPickerDevice) { device in
                BoosterScanPickerSheet(device: device) { scan in
                    client.sendScan(
                        scanID: scan.scanID,
                        scanDirectory: scan.directory,
                        to: device
                    )
                }
            }
            .alert(
                "Something went wrong",
                isPresented: Binding(
                    get: {
                        client.lastError != nil || discovery.lastError != nil
                            || localError != nil
                    },
                    set: { isPresented in
                        if !isPresented {
                            client.clearError()
                            discovery.clearError()
                            localError = nil
                        }
                    }
                ),
                presenting: client.lastError ?? discovery.lastError ?? localError
            ) { _ in
                Button("OK") {}
            } message: { error in
                Text(error.errorDescription ?? "Please try again.")
            }
        }
    }

    private func startPairing(with device: BoosterDevice) {
        Task {
            do {
                let (host, port) = try await discovery.resolveHostPort(for: device)
                pairing.reset()
                pairingDevice = device
                await pairing.beginPairing(with: device, host: host, port: port)
            } catch let error as BoosterError {
                localError = error
            } catch {
                localError = .connectionFailed(error.localizedDescription)
            }
        }
    }
}
