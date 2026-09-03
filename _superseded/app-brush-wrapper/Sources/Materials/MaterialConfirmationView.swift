//
//  MaterialConfirmationView.swift
//  Nimbus3D (Materials module)
//
//  ONE-TAP material confirmation. Classification is NEVER applied silently: the
//  Core ML classifier only ever proposes, and the chosen class is whatever the
//  user taps here. A tap on any material tile confirms that class and calls
//  `onConfirm`; the classifier's best guess is shown as "Suggested" but nothing
//  proceeds until the user taps.
//
//  The six tiles are the coarse classes the library and classifier share
//  (masonry / fabric / granular / wood / metal / tile). The confirmed
//  CoarseMaterial is reported as its contract-level MaterialClass so downstream
//  stages (DynamicTextureBuilder) receive a plain MaterialClass.
//

import SwiftUI
import UIKit

@MainActor
public struct MaterialConfirmationView: View {

    private let classification: MaterialClassification
    private let previewAlbedoURL: URL?
    private let onConfirm: (MaterialClass) -> Void

    /// The classifier's best coarse guess, if it maps to a library class.
    private let suggested: CoarseMaterial?

    @State private var lastTapped: CoarseMaterial?

    private let columns = [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())]

    public init(classification: MaterialClassification,
                previewAlbedoURL: URL? = nil,
                onConfirm: @escaping (MaterialClass) -> Void) {
        self.classification = classification
        self.previewAlbedoURL = previewAlbedoURL
        self.onConfirm = onConfirm
        self.suggested = CoarseMaterial(contractClass: classification.materialClass)
    }

    public var body: some View {
        VStack(spacing: 20) {
            header

            if let previewAlbedoURL, let uiImage = Self.loadImage(previewAlbedoURL) {
                Image(uiImage: uiImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(height: 140)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }

            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(CoarseMaterial.allCases) { material in
                    tile(for: material)
                }
            }

            Text("Tap the material that matches the surface. Your choice is used; the app never picks silently.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    // MARK: - Header

    @ViewBuilder
    private var header: some View {
        VStack(spacing: 4) {
            Text("Confirm material")
                .font(.title2.weight(.semibold))
            if let suggested, classification.confidence > 0 {
                Text("Suggested: \(suggested.displayName) (\(Int((classification.confidence * 100).rounded()))%)")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                Text("No confident guess yet, please choose")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Tile

    private func tile(for material: CoarseMaterial) -> some View {
        let isSuggested = material == suggested
        return Button {
            lastTapped = material
            onConfirm(material.contractClass)
        } label: {
            VStack(spacing: 8) {
                Image(systemName: material.symbolName)
                    .font(.system(size: 28))
                    .frame(height: 32)
                Text(material.displayName)
                    .font(.callout.weight(.medium))
                if isSuggested {
                    Text("Suggested")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.accentColor.opacity(0.18)))
                        .foregroundStyle(Color.accentColor)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.secondary.opacity(0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(isSuggested ? Color.accentColor : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text(material.displayName + (isSuggested ? ", suggested" : "")))
    }

    // MARK: - Image loading

    private static func loadImage(_ url: URL) -> UIImage? {
        UIImage(contentsOfFile: url.path)
    }
}

#if DEBUG
#Preview {
    MaterialConfirmationView(
        classification: MaterialClassification(
            materialClass: .brick,
            confidence: 0.72,
            alternatives: [MaterialScore(materialClass: .stone, confidence: 0.18)])
    ) { chosen in
        print("Confirmed: \(chosen.rawValue)")
    }
}
#endif
