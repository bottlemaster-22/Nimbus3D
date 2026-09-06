//
//  OnboardingFlowView.swift
//  Onboarding
//
//  The first-run flow, and the entry point the app shell asks for.
//
//  `NimbusUI.shared.onboardingFlow` is a
//  `(DeviceCapabilityReport, @escaping () -> Void) -> AnyView`, and the
//  integration block in `Sources/App/NimbusApp.swift` fills it with
//  `AnyView(OnboardingFlowView(report: report, onFinish: done))`. That call
//  shape is fixed by the contract, so the initialiser below matches it exactly
//  and is written out by hand rather than left to the memberwise synthesiser,
//  which would be private the moment this struct gained a private stored
//  property.
//
//  Three pages, in this order:
//
//    0  Welcome        what the app does, and what it will not do (no cloud,
//                      no account, no subscription)
//    1  This iPhone    the tier, the reason for it, the per-feature list, the
//                      re-check button and the technical details
//    2  Good scans     four plain tips, so the live guidance makes sense when
//                      it starts talking
//
//  An incompatible verdict short-circuits all three and shows
//  `OnboardingIncompatibleView`, which has no way past it. That is deliberate:
//  see the header of that file.
//
//  The report handed in by the shell is the starting point, not the last word.
//  This view loads the full findings behind it (measurements, not just the
//  verdict) so the copy can be specific, and a re-check replaces both.
//

#if canImport(SwiftUI)
import SwiftUI

/// The first-run flow.
@MainActor
struct OnboardingFlowView: View {

    /// The verdict the app shell already has. Used immediately so the screen
    /// draws without a spinner, then superseded by `findings` a moment later.
    let report: DeviceCapabilityReport

    /// Called when the user is through the flow. The shell records that and
    /// moves to the tabs.
    let onFinish: () -> Void

    @State private var findings: DeviceCompatibilityFindings?

    /// True when `findings` came off disk and is old enough that the screen
    /// must say when it was measured rather than implying "now".
    @State private var findingsAreStale = false

    @State private var page = 0
    @State private var isRechecking = false
    @State private var showsDetails = false

    private static let pageCount = 3

    init(report: DeviceCapabilityReport, onFinish: @escaping () -> Void) {
        self.report = report
        self.onFinish = onFinish
    }

    // MARK: - Body

    var body: some View {
        Group {
            if currentReport.tier == .incompatible {
                OnboardingIncompatibleView(
                    report: currentReport,
                    findings: findings,
                    findingsAreStale: findingsAreStale,
                    isRechecking: isRechecking,
                    onRecheck: { recheck() }
                )
            } else {
                flow
            }
        }
        .task { await loadFindings() }
    }

    /// The freshest verdict available: a re-check if one has run, the loaded
    /// findings if they have arrived, otherwise the one the shell handed in.
    private var currentReport: DeviceCapabilityReport {
        findings?.report ?? report
    }

    // MARK: - The paged flow

    private var flow: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    switch page {
                    case 0: welcomePage
                    case 1: devicePage
                    default: tipsPage
                    }
                }
                .padding(24)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            footer
        }
    }

    private var footer: some View {
        VStack(spacing: 14) {
            OnboardingPageDots(count: Self.pageCount, current: page)

            HStack(spacing: 12) {
                if page > 0 {
                    Button(OnboardingCopy.backButton) {
                        withAnimation { page -= 1 }
                    }
                    .buttonStyle(.bordered)
                }

                Button(isLastPage
                    ? OnboardingCopy.startButton
                    : OnboardingCopy.continueButton
                ) {
                    advance()
                }
                .buttonStyle(.borderedProminent)
                .frame(maxWidth: .infinity)
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 12)
        .padding(.bottom, 10)
        .background(.bar)
    }

    private var isLastPage: Bool { page >= Self.pageCount - 1 }

    private func advance() {
        if isLastPage {
            onFinish()
        } else {
            withAnimation { page += 1 }
        }
    }

    // MARK: - Page 0: welcome

    private var welcomePage: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text(OnboardingCopy.welcomeTitle)
                .font(.largeTitle.weight(.bold))
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)

            Text(OnboardingCopy.welcomeBody)
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 22) {
                ForEach(OnboardingCopy.welcomePoints.indices, id: \.self) { index in
                    let point = OnboardingCopy.welcomePoints[index]
                    OnboardingPointRow(
                        symbol: point.symbol,
                        title: point.title,
                        text: point.body
                    )
                }
            }
        }
    }

    // MARK: - Page 1: what this iPhone can do

    private var devicePage: some View {
        // The freshest verdict, which is not necessarily the one the shell
        // handed in: a re-check on this very page can have replaced it.
        let shown = currentReport

        return VStack(alignment: .leading, spacing: 24) {
            VStack(alignment: .leading, spacing: 12) {
                Text(OnboardingCopy.tierHeadline(shown.tier))
                    .font(.title.weight(.bold))
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)

                OnboardingTierBadge(tier: shown.tier)

                Text(OnboardingCopy.tierSummary(
                    shown.tier,
                    context: findings?.reasonContext
                ))
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            OnboardingFeatureListView(features: shown.features)

            OnboardingRecheckButton(
                isRunning: isRechecking,
                note: OnboardingCopy.recheckNoteGeneral,
                action: { recheck() }
            )

            if let findings = findings {
                DisclosureGroup(
                    OnboardingCopy.technicalDetailsHeading,
                    isExpanded: $showsDetails
                ) {
                    OnboardingDeviceFactsView(
                        findings: findings,
                        findingsAreStale: findingsAreStale
                    )
                    .padding(.top, 12)
                }
                .font(.callout.weight(.medium))
            }
        }
    }

    // MARK: - Page 2: getting a good scan

    private var tipsPage: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text(OnboardingCopy.tipsTitle)
                .font(.largeTitle.weight(.bold))
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)

            Text(OnboardingCopy.tipsBody)
                .font(.body)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 22) {
                ForEach(OnboardingCopy.tips.indices, id: \.self) { index in
                    let tip = OnboardingCopy.tips[index]
                    OnboardingPointRow(
                        symbol: tip.symbol,
                        title: tip.title,
                        text: tip.body
                    )
                }
            }
        }
    }

    // MARK: - Probing

    /// The probe to use.
    ///
    /// Prefers the one the app shell registered, so a re-check updates the
    /// same cache the rest of the app reads. Falls back to a single shared
    /// instance of our own, which matters when this view is shown from a
    /// preview or a test harness with nothing registered. Both write through
    /// `DeviceReportStore.shared`, so they cannot disagree for long.
    private static func resolvedProbe() -> DeviceCompatibilityProbe {
        if let registered = NimbusServices.shared.deviceCompatibility
            as? DeviceCompatibilityProbe
        {
            return registered
        }
        return fallbackProbe
    }

    private static let fallbackProbe = DeviceCompatibilityProbe()

    /// Fills in `findings` without re-probing if a previous launch saved one.
    ///
    /// The probe work runs off the main thread. It is only a few milliseconds
    /// (one sysctl, some static ARKit queries, a Metal device creation and a
    /// volume-capacity read), but creating the Metal device is the slowest of
    /// those and there is no reason for it to be on the thread that is drawing.
    private func loadFindings() async {
        guard findings == nil else { return }
        let probe = Self.resolvedProbe()
        let value = await Task.detached(priority: .userInitiated) {
            probe.findingsForDisplay()
        }.value
        findings = value.findings
        findingsAreStale = value.isStale
    }

    /// Runs the whole check again and redraws from the new answer.
    ///
    /// This is what makes the "Check again" button honest: it is a real
    /// re-probe, not a re-read of a cached verdict, so turning off Low Power
    /// Mode, closing other apps, or letting the phone cool genuinely changes
    /// what the screen says.
    private func recheck() {
        guard !isRechecking else { return }
        isRechecking = true
        let probe = Self.resolvedProbe()
        Task {
            let value = await Task.detached(priority: .userInitiated) {
                probe.evaluateDetailed()
            }.value
            findings = value
            // A re-check just measured this phone, so whatever was on disk is
            // no longer what is on screen.
            findingsAreStale = false
            isRechecking = false
        }
    }
}

// MARK: - Page dots

/// Which page of the flow this is. File-private: it is a detail of this
/// screen, and a single-target project has no namespace to hide it in.
@MainActor
private struct OnboardingPageDots: View {

    let count: Int
    let current: Int

    var body: some View {
        HStack(spacing: 8) {
            ForEach(0..<count, id: \.self) { index in
                Circle()
                    .fill(index == current ? Color.accentColor : Color.secondary.opacity(0.3))
                    .frame(width: 7, height: 7)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Step \(current + 1) of \(count)")
    }
}
#endif
