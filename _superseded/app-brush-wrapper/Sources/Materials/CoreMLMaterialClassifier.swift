//
//  CoreMLMaterialClassifier.swift
//  Nimbus3D (Materials module)
//
//  Coarse material classifier. Real Core ML + Vision inference path with a
//  model SLOT: it loads NimbusMaterialClassifier.mlmodelc from the app bundle
//  when present. No trained model ships yet, so until one is bundled the
//  classifier honestly returns .unknown with confidence 0 and lists all six
//  coarse classes as equal alternatives. That result is deliberately unusable
//  without the one-tap MaterialConfirmationView: classification is NEVER
//  applied silently, the user always confirms or corrects the class.
//
//  TODO(nimbus): train and bundle the real model. What a real impl needs:
//  a small image classifier (MobileNetV3 or FastViT backbone) fine-tuned on
//  surface crops labelled with the six coarse classes (masonry / fabric /
//  granular / wood / metal / tile), e.g. from the Materials in Context (MINC)
//  dataset remapped to these labels, exported with coremltools as an
//  image-input classifier whose class labels are exactly the CoarseMaterial
//  raw values, compiled to NimbusMaterialClassifier.mlmodelc and added to the
//  app bundle. This file's inference path then works unchanged.
//

import Foundation
import CoreML
import Vision

public final class CoreMLMaterialClassifier: MaterialClassifier, @unchecked Sendable {

    /// Compiled Core ML model name expected in the app bundle.
    public static let bundledModelName = "NimbusMaterialClassifier"

    /// Nil when no model is bundled. Immutable after init; Vision requests on a
    /// VNCoreMLModel are thread-safe, hence the @unchecked Sendable above.
    private let visionModel: VNCoreMLModel?

    /// True when a real Core ML model was found and loaded.
    public var hasModel: Bool { visionModel != nil }

    public init(bundle: Bundle = .main) {
        if let modelURL = bundle.url(forResource: Self.bundledModelName, withExtension: "mlmodelc") {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            if let mlModel = try? MLModel(contentsOf: modelURL, configuration: configuration),
               let vnModel = try? VNCoreMLModel(for: mlModel) {
                self.visionModel = vnModel
            } else {
                self.visionModel = nil
            }
        } else {
            self.visionModel = nil
        }
    }

    public func classify(albedoURL: URL) async throws -> MaterialClassification {
        guard let visionModel else {
            // Honest no-model result: unknown at zero confidence, all coarse
            // classes offered as equal alternatives. The confirm UI turns this
            // into a required user choice; nothing downstream may treat it as
            // a real prediction.
            let uniform = 1.0 / Float(CoarseMaterial.allCases.count)
            let alternatives = CoarseMaterial.allCases.map {
                MaterialScore(materialClass: $0.contractClass, confidence: uniform)
            }
            return MaterialClassification(materialClass: .unknown,
                                          confidence: 0,
                                          alternatives: alternatives)
        }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .centerCrop
        let handler = VNImageRequestHandler(url: albedoURL, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw NimbusError.classificationFailed("Vision inference failed: \(error.localizedDescription)")
        }

        guard let observations = request.results as? [VNClassificationObservation],
              !observations.isEmpty else {
            throw NimbusError.classificationFailed("Model returned no classification observations")
        }

        // Model class labels are CoarseMaterial raw values (see TODO header).
        let scored: [MaterialScore] = observations.compactMap { obs in
            guard let coarse = CoarseMaterial(rawValue: obs.identifier) else { return nil }
            return MaterialScore(materialClass: coarse.contractClass, confidence: obs.confidence)
        }
        guard let top = scored.first else {
            throw NimbusError.classificationFailed(
                "Model labels do not match the expected coarse classes (\(CoarseMaterial.allCases.map(\.rawValue).joined(separator: ", ")))")
        }
        return MaterialClassification(materialClass: top.materialClass,
                                      confidence: top.confidence,
                                      alternatives: Array(scored.dropFirst()))
    }
}
