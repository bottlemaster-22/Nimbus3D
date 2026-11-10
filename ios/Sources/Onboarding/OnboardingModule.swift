//
//  OnboardingModule.swift
//  Onboarding
//
//  How this module gets wired into the app.
//
//  ---------------------------------------------------------------------------
//  THE INTEGRATION BLOCK STILL OWNS THE DECISION
//  ---------------------------------------------------------------------------
//
//  CONTRACTS.md section 5 is explicit: `NimbusServices` and `NimbusUI` are
//  filled in from exactly one place, the integration block at the top of
//  `Sources/App/NimbusApp.swift`, so that "what is actually built?" has one
//  visible answer rather than a scattering of registration side effects nobody
//  can find.
//
//  This file does NOT break that rule. There is no `+load`, no static
//  initialiser, no side effect. `register()` does nothing until somebody calls
//  it, and the only place that should call it is the integration block. It
//  exists so that block can say
//
//      OnboardingModule.register()
//
//  instead of repeating this module's concrete type names, which means a
//  rename inside `Sources/Onboarding` stops being an edit to another module's
//  file.
//
//  Either form is correct. The equivalent explicit form, which is what the
//  block currently has commented out, is:
//
//      services.deviceCompatibility = DeviceCompatibilityProbe()
//      ui.onboardingFlow = { report, done in
//          AnyView(OnboardingFlowView(report: report, onFinish: done))
//      }
//
//  ---------------------------------------------------------------------------
//  ONE THING THE APP SHELL DOES NOT ROUTE HERE YET
//  ---------------------------------------------------------------------------
//
//  `RootView.runCompatibilityCheck()` sends an `.incompatible` verdict to its
//  own `IncompatibleDeviceView` rather than through `ui.onboardingFlow`, so
//  `OnboardingIncompatibleView` (the specified screen, with the honest
//  device-specific reason and the conditional re-check) is only reached when
//  the flow itself re-checks and the verdict changes. Both screens follow the
//  same required four-part structure, so nothing is wrong today; the shell's
//  version simply has less to say, because it only receives the report and not
//  the measurements behind it.
//
//  Routing `.blocked` through the flow as well is a one-line change in
//  `RootView`, and it belongs to the App agent, not to this module. It is
//  written down here so it is not lost.
//

#if canImport(SwiftUI)
import SwiftUI

/// The module's front door.
public enum OnboardingModule {

    /// Registers the compatibility service and the first-run flow.
    ///
    /// Idempotent in the sense that calling it twice simply replaces the
    /// registrations with equivalent ones. It does allocate a second probe,
    /// which costs an object with a lock in it, so there is no reason to.
    @MainActor
    public static func register() {
        NimbusServices.shared.deviceCompatibility = DeviceCompatibilityProbe()
        NimbusUI.shared.onboardingFlow = { report, done in
            AnyView(OnboardingFlowView(report: report, onFinish: done))
        }
    }
}
#endif
