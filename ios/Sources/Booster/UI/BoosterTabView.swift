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
    // DEVELOPMENT ONLY. Removing the self-update feature is this line, the
    // `selfUpdateSection` below, its one use in the List, and
    // SelfUpdateService.swift. Nothing else refers to it.
    @StateObject private var selfUpdate = SelfUpdateService()

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
                selfUpdateSection
            }
            .navigationTitle("Booster")
            .onAppear {
                discovery.start()
                jobs = client.resultsHistory()
                // Says whether an update this app quit to allow actually
                // landed. Cheap and local: it compares the build number
                // written down before quitting against this one.
                selfUpdate.resumeAfterRelaunch()
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

    // MARK: - Development self-update

    /// Asks the Bottle broker to reinstall this app over the air.
    ///
    /// Renders NOTHING on a build CI did not configure, which is every
    /// local build, because a button that cannot work is worse than no
    /// button. See SelfUpdateService for why the whole feature is designed
    /// to be deleted rather than switched off.
    @ViewBuilder
    private var selfUpdateSection: some View {
        if SelfUpdateService.isConfigured {
            Section {
                // Which build this is, so a report about it can name it.
                // Selectable for the same reason it is on the Scans tab.
                Text(BrandConfig.buildIdentity)
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)

                if selfUpdate.isWorking {
                    HStack(spacing: 10) {
                        ProgressView()
                        Text(selfUpdate.status == .updating
                             ? "Installing the new build..."
                             : "Asking for the newest build...")
                    }
                } else {
                    Button("Check for a new build") {
                        Task { await selfUpdate.update() }
                    }
                }

                // The broker writes these sentences for a screen, so they
                // are shown as sent rather than reworded here. It cannot
                // know this app is a scanner, and this app cannot know why
                // a relay is offline.
                if let message = selfUpdate.message {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(
                            selfUpdate.status == .failed ? Color.red : Color.secondary
                        )
                }

                if selfUpdate.status == .updated {
                    Text("Installed. Reopen the app to run the new build.")
                        .font(.footnote)
                }

                // Only offered after a failure, and worded for the one case
                // it is for. The broker skips work when the installed build
                // is already newest, which is what makes an ordinary check
                // cheap, but that same skip would refuse the seven-day
                // certificate re-sign a lapsed app needs. Force asks anyway.
                if selfUpdate.status == .failed, !selfUpdate.isWorking {
                    Button("Reinstall anyway") {
                        Task { await selfUpdate.update(force: true) }
                    }
                    .foregroundStyle(.secondary)
                }
            } header: {
                Text("This build")
            } footer: {
                Text(
                    "Development only. The update is signed and installed by "
                        + "your own Bottle relay at home, so it needs that "
                        + "machine powered on and on the same network."
                )
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
