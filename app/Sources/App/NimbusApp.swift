//
//  NimbusApp.swift
//  Nimbus3D
//
//  App entry point and root navigation shell. The three tabs mount the real
//  module screens:
//    - Capture  -> CaptureRootView     (Capture module: ARKit guided scan)
//    - Process  -> ProcessRootView     (Pipeline module: stage-by-stage run)
//    - Library  -> LibraryRootView     (Pipeline module: exported assets + share)
//
//  Concrete stage services are injected into PipelineServices at launch by
//  NimbusServices.registerAll() (see NimbusServices.swift). The Process screen
//  reads whatever is registered when a run starts; anything unwired surfaces as
//  an honest "not wired" skip rather than a faked result.
//

import SwiftUI

@main
struct NimbusApp: App {

    init() {
        // Inject the real per-module implementations behind the Core protocols.
        // Runs on the main thread (App.init is main-actor context) before any UI.
        NimbusServices.registerAll()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

// MARK: - Root shell

struct RootView: View {
    var body: some View {
        TabView {
            // Capture is a full-screen immersive AR view; it owns its own chrome,
            // so it is not wrapped in a NavigationStack.
            CaptureRootView()
                .ignoresSafeArea()
                .tabItem {
                    Label("Capture", systemImage: "camera.viewfinder")
                }

            NavigationStack {
                ProcessRootView()
            }
            .tabItem {
                Label("Process", systemImage: "gearshape.2")
            }

            NavigationStack {
                LibraryRootView()
            }
            .tabItem {
                Label("Library", systemImage: "square.grid.2x2")
            }
        }
    }
}

#Preview {
    RootView()
}
