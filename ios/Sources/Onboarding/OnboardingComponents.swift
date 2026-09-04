//
//  OnboardingComponents.swift
//  Onboarding
//
//  The small views the first-run screens are built from.
//
//  Every type here is prefixed `Onboarding` on purpose. This project is a
//  SINGLE Xcode target: there are no Swift modules and no namespacing, so two
//  files anywhere in the app may not declare the same top-level type name.
//  `Sources/App` already owns `FeatureListView`, `IncompatibleDeviceView` and
//  `MinimalCompatibilitySummaryView` (its own honest stand-ins for what this
//  module provides), so nothing here may take those names. See CONTRACTS.md
//  section 2.
//
//  Accessibility is not decoration here. The owner's requirement is that the
//  app be genuinely usable, and a compatibility verdict delivered only as a
//  green tick and a red cross is unusable to anyone who cannot tell those two
//  colours apart. So every row carries a shape as well as a colour, and an
//  accessibility label that says the verdict in words.
//

#if canImport(SwiftUI)
import SwiftUI

// MARK: - Feature list

/// The per-feature availability list, split into what works and what does not.
///
/// Splitting rather than interleaving is deliberate. A mixed list of ticks and
/// crosses makes a person scan every row to work out where they stand; two
/// headed groups answer "can I use this?" in one glance, which is the entire
/// job of this screen.
@MainActor
struct OnboardingFeatureListView: View {

    let features: [FeatureAvailability]

    private var available: [FeatureAvailability] {
        features.filter(\.isAvailable)
    }

    private var unavailable: [FeatureAvailability] {
        features.filter { !$0.isAvailable }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 28) {
            if !available.isEmpty {
                group(
                    heading: OnboardingCopy.worksHeading,
                    features: available
                )
            }
            if showsUnavailable, !unavailable.isEmpty {
                group(
                    heading: OnboardingCopy.doesNotWorkHeading,
                    features: unavailable
                )
            }
            Text(OnboardingCopy.featureListNote)
                .font(.footnote)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func group(
        heading: String,
        features: [FeatureAvailability]
    ) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(heading)
                .font(.headline)
                .accessibilityAddTraits(.isHeader)

            ForEach(features) { feature in
                OnboardingFeatureRow(feature: feature)
            }
        }
    }
}

/// One capability, its verdict, and the plain-language detail underneath.
@MainActor
struct OnboardingFeatureRow: View {

    let feature: FeatureAvailability

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Image(systemName: feature.isAvailable
                ? "checkmark.circle.fill"
                : "xmark.circle.fill")
                .foregroundStyle(feature.isAvailable ? Color.green : Color.secondary)
                .imageScale(.large)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 4) {
                Text(feature.title)
                    .font(.body.weight(.medium))
                    .fixedSize(horizontal: false, vertical: true)

                if let detail = feature.detail, !detail.isEmpty {
                    Text(detail)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        // Read as one sentence rather than as three unlabelled fragments.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        let verdict = feature.isAvailable ? "Available" : "Not available"
        guard let detail = feature.detail, !detail.isEmpty else {
            return "\(feature.title). \(verdict)."
        }
        return "\(feature.title). \(verdict). \(detail)"
    }
}

// MARK: - Illustrated points

/// A symbol, a short title and a sentence. Used for the welcome page and the
/// scanning tips, which are the two places the app is explaining itself rather
/// than reporting a measurement.
@MainActor
struct OnboardingPointRow: View {

    let symbol: String
    let title: String

    /// The sentence under the title. Named `text` rather than `body` because
    /// `body` is `View`'s own requirement and a stored property would collide
    /// with it.
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: symbol)
                .font(.title2)
                .frame(width: 32, alignment: .center)
                .foregroundStyle(.tint)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.body.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                Text(text)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Tier badge

/// The one-word verdict, as a pill.
///
/// It never appears without the sentence that explains it: a badge on its own
/// is a grade, and grading somebody's phone is not the job.
@MainActor
struct OnboardingTierBadge: View {

    let tier: DeviceTier

    var body: some View {
        Text(label)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(badgeBackground, in: Capsule())
            .foregroundStyle(badgeForeground)
            .accessibilityLabel("Device status: \(label)")
    }

    private var label: String {
        switch tier {
        case .full: return "Everything works"
        case .limited: return "Most things work"
        case .incompatible: return "Cannot run here"
        }
    }

    private var badgeBackground: Color {
        switch tier {
        case .full: return Color.green.opacity(0.15)
        case .limited: return Color.orange.opacity(0.18)
        case .incompatible: return Color.secondary.opacity(0.15)
        }
    }

    private var badgeForeground: Color {
        switch tier {
        case .full: return .green
        case .limited: return .orange
        case .incompatible: return .secondary
        }
    }
}

// MARK: - Technical details

/// Everything the probe measured, as a plain table.
///
/// Collapsed by default, because none of it is needed to decide whether to
/// scan a room. It exists so that when something looks wrong, the answer is
/// one tap away and can be read out over a phone call, rather than requiring a
/// debug build.
@MainActor
struct OnboardingDeviceFactsView: View {

    let findings: DeviceCompatibilityFindings

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            let report = findings.report

            OnboardingFactRow(
                symbol: "iphone",
                label: "Device",
                value: findings.model.displayName
            )

            if findings.model.displayName != findings.model.identifier {
                OnboardingFactRow(
                    symbol: "number",
                    label: "Model identifier",
                    value: findings.model.identifier
                )
            }

            OnboardingFactRow(
                symbol: "cpu",
                label: "Chip",
                value: report.chipName ?? "Not determined",
                note: chipSourceNote
            )

            OnboardingFactRow(
                symbol: "memorychip",
                label: "Memory",
                value: OnboardingFormat.memory(report.totalMemoryBytes)
            )

            OnboardingFactRow(
                symbol: "memorychip",
                label: "Memory this app may use now",
                value: OnboardingFormat.bytes(report.availableMemoryBytes),
                note: findings.memory.availableIsEstimated
                    ? "Estimated. iOS did not report a figure, so this is a "
                        + "cautious stand-in, not a measurement."
                    : nil
            )

            OnboardingFactRow(
                symbol: "square.stack.3d.up",
                label: "Graphics",
                value: report.metalGPUFamily ?? "Not reported",
                note: findings.metal.deviceName
            )

            OnboardingFactRow(
                symbol: "gear",
                label: "iOS version",
                value: report.systemVersion
            )

            OnboardingFactRow(
                symbol: "speedometer",
                label: "Sustained speed",
                value: OnboardingFormat.performanceClass(report.sustainedPerformanceClass),
                note: performanceNote
            )

            OnboardingFactRow(
                symbol: "thermometer",
                label: "Temperature now",
                value: OnboardingFormat.thermal(findings.performance.thermalLevel)
            )

            if report.lowPowerModeEnabled {
                OnboardingFactRow(
                    symbol: "bolt.slash",
                    label: "Low Power Mode",
                    value: "On",
                    note: "Turning it off makes everything here faster."
                )
            }

            if let free = findings.memory.freeDiskBytes {
                OnboardingFactRow(
                    symbol: "internaldrive",
                    label: "Free space",
                    value: OnboardingFormat.bytes(free)
                )
            }

            OnboardingFactRow(
                symbol: "clock",
                label: "Checked",
                value: OnboardingFormat.timestamp(findings.checkedAt)
            )
        }
    }

    /// Where the chip generation came from. Shown because a guessed chip and a
    /// looked-up chip are different kinds of fact and the screen should not
    /// present them as the same one.
    private var chipSourceNote: String? {
        switch findings.chipGenerationSource {
        case .modelTable:
            return nil
        case .identifierPattern:
            return "Worked out from the model number. This phone is newer than "
                + "the list built into the app, so the name is a best guess. "
                + "It does not affect what the app will let you do."
        case .metalFamily:
            return "Worked out from the graphics chip, because the model "
                + "number could not be read."
        case .undetermined:
            return "Could not be determined. The app judged this device on its "
                + "measured memory and graphics instead, which is the safe "
                + "direction."
        }
    }

    private var performanceNote: String? {
        var parts: [String] = []
        if findings.performance.reducedByLowPowerMode {
            parts.append("lowered because Low Power Mode is on")
        }
        if findings.performance.reducedByHeat {
            parts.append("lowered because the phone is warm right now")
        }
        guard !parts.isEmpty else { return nil }
        return "Currently "
            + OnboardingFormat.sentenceList(parts)
            + ". It returns to "
            + OnboardingFormat.performanceClass(findings.performance.baseClass).lowercased()
            + " on its own."
    }
}

/// One labelled measurement.
@MainActor
struct OnboardingFactRow: View {

    let symbol: String
    let label: String
    let value: String
    var note: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Image(systemName: symbol)
                    .font(.caption)
                    .frame(width: 18, alignment: .center)
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)

                Text(label)
                    .font(.callout)
                    .foregroundStyle(.secondary)

                Spacer(minLength: 12)

                Text(value)
                    .font(.callout.weight(.medium))
                    .multilineTextAlignment(.trailing)
            }

            if let note = note, !note.isEmpty {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 28)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Re-check button

/// "Check again", with the honest note about when it is worth pressing.
///
/// The note is not the same sentence in both cases. On a device where a
/// re-check genuinely could change the verdict, it says so. On one where it
/// cannot, this control is simply not shown.
@MainActor
struct OnboardingRecheckButton: View {

    let isRunning: Bool
    var note: String?
    let action: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: action) {
                HStack(spacing: 8) {
                    if isRunning {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                    Text(isRunning
                        ? OnboardingCopy.recheckingLabel
                        : OnboardingCopy.recheckButton)
                }
            }
            .buttonStyle(.bordered)
            .disabled(isRunning)

            if let note = note, !note.isEmpty {
                Text(note)
                    .font(.footnote)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
#endif
