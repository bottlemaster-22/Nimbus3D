//
//  ScanCensusViews.swift
//  Viewer
//
//  SHOWING THE CENSUS TO SOMEONE WHO IS NOT AN ENGINEER.
//
//  The whole design is one rule: ONE SENTENCE FIRST, EVERYTHING ELSE BEHIND A
//  TAP.
//
//  The owner of this app opens a scan that looks like nothing and wants one
//  thing: to be told, in his own language, which step lost it. He does not want
//  eleven numbers. So the review screen shows a single line - "Most of this
//  scan was lost when the phone got warm and the build shrank itself" - and a
//  button. The eleven numbers live behind the button, for the moment when
//  somebody wants to know exactly which file said what.
//
//  Three things this file will not do:
//
//    * It will not print a zero where a number is missing. A missing number is
//      drawn in grey italics as "not recorded", with the reason underneath, so
//      an older scan reads as unrecorded rather than as a catastrophe.
//
//    * It will not print a setting as though it were a measurement. A declared
//      number is drawn differently and its evidence line says "a setting
//      recorded in model/model.json".
//
//    * It will not blame the person holding the phone for something the app
//      did. Every sentence about a build step names the build step.
//

import SwiftUI

// MARK: - Tone

/// The look of a census verdict. Kept here rather than on `ScanCensus` so the
/// data model stays free of SwiftUI.
private enum ScanCensusStyle {

    static func icon(_ tone: ScanCensus.Tone) -> String {
        switch tone {
        case .good: return "checkmark.circle"
        case .neutral: return "info.circle"
        case .lost: return "exclamationmark.triangle"
        case .unknown: return "questionmark.circle"
        }
    }

    static func color(_ tone: ScanCensus.Tone) -> Color {
        switch tone {
        case .good: return .green
        case .neutral: return .secondary
        case .lost: return .orange
        case .unknown: return .secondary
        }
    }
}

// MARK: - The review screen's section

/// The census as one section of the review screen's list: the headline
/// sentence, and the way into the full breakdown.
///
/// A `View` that returns a `Section`, so the review screen can drop it into its
/// `List` and keep its own body readable.
struct ScanCensusSummarySection: View {
    let census: ScanCensus

    var body: some View {
        Section {
            ScanCensusCard(census: census)
        } header: {
            Text("Where this scan's detail went")
        } footer: {
            Text(footerText)
        }
    }

    /// Split out of `body` because a long conditional string built inline is
    /// both hard to read and slow for the type checker.
    private var footerText: String {
        if census.isIncomplete {
            return "Some steps of this scan did not write down what they did. Those are marked "
                + "as not recorded in the breakdown rather than shown as zero, because nobody "
                + "counted them: it does not mean they did nothing."
        }
        return "Every number here was counted in a file on this phone. The breakdown names "
            + "which file each one came from."
    }
}

// MARK: - The card

/// One sentence, one button. This is the five second read.
struct ScanCensusCard: View {
    let census: ScanCensus

    @State private var showingBreakdown = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label {
                Text(census.headline.sentence)
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: ScanCensusStyle.icon(census.headline.tone))
                    .foregroundStyle(ScanCensusStyle.color(census.headline.tone))
            }

            if !census.sections.isEmpty {
                Button {
                    showingBreakdown = true
                } label: {
                    Label("Show the full breakdown", systemImage: "list.number")
                        .font(.footnote)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(.vertical, 4)
        .sheet(isPresented: $showingBreakdown) {
            ScanCensusDetailSheet(census: census)
        }
    }
}

// MARK: - The breakdown

/// Every rung of the ladder, in order, each one saying which file it came from.
///
/// This is the screen that turns a day of code reading into a scroll. It is
/// deliberately behind a tap: it is a reference, not a headline.
struct ScanCensusDetailSheet: View {
    let census: ScanCensus

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Label {
                        Text(census.headline.sentence)
                            .font(.subheadline)
                            .fixedSize(horizontal: false, vertical: true)
                    } icon: {
                        Image(systemName: ScanCensusStyle.icon(census.headline.tone))
                            .foregroundStyle(ScanCensusStyle.color(census.headline.tone))
                    }
                    .padding(.vertical, 2)
                }

                ForEach(census.sections) { section in
                    if !section.stages.isEmpty {
                        Section(section.title) {
                            ForEach(section.stages) { stage in
                                ScanCensusStageRow(stage: stage)
                            }
                        }
                    }
                }

                Section {
                    ForEach(Array(census.evidence.enumerated()), id: \.offset) { entry in
                        Text(entry.element)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Where these numbers came from")
                } footer: {
                    Text(
                        "This list is here so that the next time a scan comes out wrong, the "
                        + "answer is on this screen instead of somewhere in the code."
                    )
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Where the detail went")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - One rung

/// One count, what it means, where it was read, and a plain warning when the
/// number is worth pointing at.
struct ScanCensusStageRow: View {
    let stage: ScanCensus.Stage

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(stage.title)
                    .font(.subheadline.weight(.medium))
                Spacer(minLength: 8)
                figure
            }

            Text(stage.plain)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Text(stage.figure.source.evidence)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)

            if let alert = stage.alert {
                Label {
                    Text(alert)
                        .font(.footnote)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .foregroundStyle(.orange)
                .padding(.top, 2)
            }
        }
        .padding(.vertical, 3)
    }

    /// The number itself. A measured count is solid; a setting the run was
    /// given is drawn in italics so it can never be mistaken for a count; an
    /// absent number is words, never a zero.
    @ViewBuilder
    private var figure: some View {
        if let count = stage.figure.count {
            if stage.figure.isMeasured {
                Text(figureText(count))
                    .font(.subheadline.weight(.semibold))
                    .monospacedDigit()
            } else {
                Text(figureText(count))
                    .font(.subheadline.italic())
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        } else {
            Text(stage.figure.text)
                .font(.subheadline.italic())
                .foregroundStyle(.secondary)
        }
    }

    /// "+268,430 points". The unit is always printed, because a bare number in
    /// a list where some rows count photos and some count points is a number
    /// somebody has to guess at.
    private func figureText(_ count: Int) -> String {
        prefix + stage.figure.text + " " + stage.unit.noun(count)
    }

    /// "+" for a step that added, "-" for one that took away, nothing for a
    /// population. A count with no sign is a level, not a change.
    private var prefix: String {
        switch stage.kind {
        case .level: return ""
        case .added: return "+"
        case .removed: return "-"
        }
    }
}

// MARK: - The library row's one line

/// The census squeezed into one line of a library row, and only when it has
/// something to say. A scan that built cleanly says nothing here: the row
/// already shows its splat count, and a green tick on every row teaches the eye
/// to skip the line that matters.
struct ScanCensusLibraryLine: View {
    let census: ScanCensus

    var body: some View {
        if let line = census.headline.shortLine {
            Label {
                Text(line)
                    .lineLimit(2)
            } icon: {
                Image(systemName: ScanCensusStyle.icon(census.headline.tone))
            }
            .font(.caption)
            .foregroundStyle(ScanCensusStyle.color(census.headline.tone))
        }
    }
}
