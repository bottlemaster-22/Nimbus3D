//
//  OnboardingIncompatibleView.swift
//  Onboarding
//
//  The screen somebody gets when their iPhone cannot run this app.
//
//  ---------------------------------------------------------------------------
//  THIS SCREEN IS SPECIFIED EXACTLY. DO NOT REARRANGE IT.
//  ---------------------------------------------------------------------------
//
//  The owner's requirement, in order, is:
//
//    1. an apology;
//    2. the line "we are trying our best to support as many devices as
//       possible";
//    3. BOLD, LARGE text reading "Why is my device incompatible?";
//    4. an honest, specific, plain-language reason for THAT device.
//
//  Those four are the screen. Everything below them (the feature list, the
//  technical details, the conditional re-check button) is supporting material
//  and may be scrolled past without missing anything.
//
//  The reason text comes from `OnboardingCopy.incompatibleReason(_:)`, which
//  builds it from what the probe actually measured on this phone. This view
//  never composes a reason itself, and it never falls back to a generic one
//  while a specific one is available.
//
//  There is deliberately no way past this screen. The core of the app is a
//  laser that this phone either has or does not have; letting somebody scan
//  anyway would waste their time and then look like their fault.
//

#if canImport(SwiftUI)
import SwiftUI

/// The incompatible-device screen, in the exact order specified above.
///
/// Named `OnboardingIncompatibleView` rather than `IncompatibleDeviceView`
/// because `Sources/App` already owns that name for its own minimal stand-in,
/// and this is a single-target project where two types may not share a name.
@MainActor
struct OnboardingIncompatibleView: View {

    /// The verdict to render. Always the freshest one available.
    let report: DeviceCapabilityReport

    /// The raw measurements behind it. Optional because the app shell may hand
    /// this view a report on its own; without findings the screen still shows
    /// the four required elements, and only the technical details and the
    /// conditional re-check button are omitted.
    var findings: DeviceCompatibilityFindings?

    var isRechecking: Bool = false

    /// nil means "no re-check is possible here", which is different from a
    /// re-check that is possible but pointless. See `showsRecheck`.
    var onRecheck: (() -> Void)?

    @State private var showsDetails = false

    init(
        report: DeviceCapabilityReport,
        findings: DeviceCompatibilityFindings? = nil,
        isRechecking: Bool = false,
        onRecheck: (() -> Void)? = nil
    ) {
        self.report = report
        self.findings = findings
        self.isRechecking = isRechecking
        self.onRecheck = onRecheck
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {

                // 1. The apology.
                Text(OnboardingCopy.incompatibleApology)
                    .font(.title2.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)

                // 2. The owner's line, verbatim.
                Text(OnboardingCopy.tryingOurBest)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                // 3 and 4. The bold, large heading and the honest reason.
                VStack(alignment: .leading, spacing: 14) {
                    Text(OnboardingCopy.whyIncompatibleHeading)
                        .font(.largeTitle.weight(.bold))
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityAddTraits(.isHeader)

                    Text(reasonText)
                        .font(.body)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if showsRecheck, let onRecheck = onRecheck {
                    OnboardingRecheckButton(
                        isRunning: isRechecking,
                        note: OnboardingCopy.recheckNoteRecoverable,
                        action: { onRecheck() }
                    )
                }

                Divider()

                // Supporting material. The screen is already complete above
                // this line.
                if !report.features.isEmpty {
                    OnboardingFeatureListView(features: report.features)
                }

                if let findings = findings {
                    DisclosureGroup(
                        OnboardingCopy.technicalDetailsHeading,
                        isExpanded: $showsDetails
                    ) {
                        OnboardingDeviceFactsView(findings: findings)
                            .padding(.top, 12)
                    }
                    .font(.callout.weight(.medium))
                } else {
                    // No raw findings, so show the two facts the report itself
                    // carries. A support conversation can start from these.
                    Text("Device: \(report.deviceModel), iOS \(report.systemVersion)")
                        .font(.footnote)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Content

    /// The device-specific reason.
    ///
    /// The probe fills `incompatibleReason` in on every `.incompatible`
    /// verdict, so the first branch is what actually runs. The second is for
    /// the case where some other code path built a report without one: the
    /// user still gets a true sentence rather than an empty space, and the
    /// sentence is the same one the copy file owns rather than a second copy
    /// of it drifting inside a view.
    private var reasonText: String {
        if let reason = report.incompatibleReason, !reason.isEmpty {
            return reason
        }
        if let findings = findings {
            return OnboardingCopy.incompatibleReason(findings.reasonContext)
        }
        return OnboardingCopy.lastResortReason
    }

    /// Whether to offer "Check again".
    ///
    /// Only where a re-check could honestly change the answer. A phone with no
    /// scanner will never grow one, and putting a hopeful button under that
    /// sentence would be a small cruelty. Without findings we cannot tell, so
    /// we do not offer it.
    private var showsRecheck: Bool {
        guard onRecheck != nil, let findings = findings else { return false }
        return OnboardingCopy.recheckCouldHelp(findings.reasonContext)
    }
}
#endif
