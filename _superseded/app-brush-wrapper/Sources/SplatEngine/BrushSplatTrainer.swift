//
//  BrushSplatTrainer.swift
//  Nimbus3D - SplatEngine module
//
//  `SplatTrainer` implementation: drives the Rust/Brush core
//  (NimbusSplatCore.xcframework) from a CaptureBundle to an on-disk .ply
//  SplatModel, reporting progress and supporting cancellation.
//
//  Register at app launch:
//      PipelineServices.shared.splatTrainer = BrushSplatTrainer()
//
//  Real vs stubbed (see MODULE_STATUS.md): training itself is REAL (Brush runs
//  3D Gaussian Splat optimization on-device via wgpu/Metal). Two seams remain
//  and are documented at their source: (1) a few Brush config field names for
//  sh-degree / splat-cap / image-downscale are applied Rust-side behind
//  SEAM markers; (2) the ARKit -> Nerfstudio coordinate convention in
//  NerfstudioDatasetExporter needs on-device validation.
//

import Foundation
import simd
import NimbusSplatCore

public final class BrushSplatTrainer: SplatTrainer, @unchecked Sendable {

    /// Guards `handle` so cancelTraining() (any thread/task) races safely with
    /// init/deinit. The Rust side has its own atomic cancel flag; this lock only
    /// protects the pointer itself.
    private let lock = NSLock()
    private var handle: OpaquePointer?

    public init() {
        self.handle = nimbus_trainer_create()
    }

    deinit {
        if let handle {
            nimbus_trainer_destroy(handle)
        }
    }

    public func cancelTraining() async {
        let current = withLock { handle }
        if let current {
            nimbus_trainer_cancel(current)
        }
    }

    public func train(_ bundle: CaptureBundle,
                      config: SplatTrainingConfig,
                      progress: @escaping ProgressHandler) async throws -> SplatModel {

        guard let handle = withLock({ handle }) else {
            throw NimbusError.trainingFailed("splat core failed to initialize")
        }
        guard !bundle.frames.isEmpty else {
            throw NimbusError.trainingFailed("capture bundle has no frames to train from")
        }

        // 1. Convert the capture bundle into a Brush-readable dataset on disk.
        let datasetDir = try NerfstudioDatasetExporter.export(bundle: bundle)

        // 2. Decide the output location for the trained model.
        let modelID = UUID()
        let modelDir = try Self.modelsDirectory()
            .appendingPathComponent(modelID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let plyURL = modelDir.appendingPathComponent("model.ply")

        // 3. Run the blocking Rust trainer off the main thread, bridging the C
        //    progress callback to `progress` and Task cancellation to the core.
        let result = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<NimbusTrainResult, Error>) in
                let worker = Thread {
                    let outcome = Self.runBlocking(handle: handle,
                                                   datasetDir: datasetDir,
                                                   plyURL: plyURL,
                                                   config: config,
                                                   progress: progress)
                    continuation.resume(with: outcome)
                }
                worker.name = "com.nimbus3d.splat-training"
                worker.stackSize = 8 << 20 // 8 MiB: Brush/wgpu use deep call stacks.
                worker.start()
            }
        } onCancel: {
            if let current = self.withLock({ self.handle }) {
                nimbus_trainer_cancel(current)
            }
        }

        // 4. Derive the bounding box (and cross-check the splat count) from the
        //    produced PLY. This is authoritative geometry read back from disk.
        let (boundingBox, plySplatCount) = try PLYBoundingBox.compute(url: plyURL)
        let splatCount = plySplatCount > 0 ? plySplatCount : Int(result.splat_count)

        return SplatModel(
            id: modelID,
            splatFileURL: plyURL,
            format: .ply,
            splatCount: splatCount,
            trainingIterations: Int(result.iterations_completed),
            sourceCaptureID: bundle.id,
            boundingBox: boundingBox
        )
    }

    // MARK: - Blocking FFI call (runs on the worker thread)

    private static func runBlocking(handle: OpaquePointer,
                                    datasetDir: URL,
                                    plyURL: URL,
                                    config: SplatTrainingConfig,
                                    progress: @escaping ProgressHandler) -> Result<NimbusTrainResult, Error> {

        let context = TrainingContext(progress: progress)
        let contextPtr = Unmanaged.passRetained(context).toOpaque()
        defer { Unmanaged<TrainingContext>.fromOpaque(contextPtr).release() }

        var trainConfig = NimbusTrainConfig(
            iterations: UInt32(max(config.iterations, 1)),
            max_splat_count: UInt32(max(config.maxSplatCount, 0)),
            max_image_dimension: UInt32(max(config.maxImageDimension, 0)),
            sh_degree: UInt32(max(config.shDegree, 0)),
            seed: 42
        )
        var result = NimbusTrainResult(splat_count: 0, sh_degree: 0, iterations_completed: 0)
        var errorBuffer = [CChar](repeating: 0, count: 1024)

        let status: Int32 = datasetDir.path.withCString { datasetC in
            plyURL.path.withCString { plyC in
                nimbus_trainer_run(handle,
                                   datasetC,
                                   plyC,
                                   &trainConfig,
                                   nimbusProgressTrampoline,
                                   contextPtr,
                                   &result,
                                   &errorBuffer,
                                   errorBuffer.count)
            }
        }

        // 0 == NimbusStatusOk (see nimbus_splat_core.h).
        if status == 0 {
            return .success(result)
        }
        let message = String(cString: errorBuffer)
        return .failure(Self.mapStatus(status, message: message))
    }

    /// Maps a NimbusStatus code to a typed NimbusError. Values mirror the C enum.
    private static func mapStatus(_ status: Int32, message: String) -> NimbusError {
        let detail = message.isEmpty ? "splat core returned status \(status)" : message
        switch status {
        case 2: return .trainingFailed("dataset load failed: \(detail)")   // DatasetLoadFailed
        case 4: return .trainingFailed("splat export failed: \(detail)")   // ExportFailed
        case 5: return .cancelled                                          // Cancelled
        default: return .trainingFailed(detail)                            // Training/Invalid/Panic
        }
    }

    // MARK: - Helpers

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }

    private static func modelsDirectory() throws -> URL {
        let base = try FileManager.default.url(for: .documentDirectory,
                                               in: .userDomainMask,
                                               appropriateFor: nil,
                                               create: true)
        let dir = base.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

// MARK: - Progress bridging

/// Retained box passed to Rust as `user_data`; forwards C progress snapshots to
/// the Swift `ProgressHandler`. Reference type so the raw pointer stays stable.
final class TrainingContext {
    let progress: ProgressHandler
    init(progress: @escaping ProgressHandler) {
        self.progress = progress
    }

    /// Returns 0 to keep training (Task/flag cancellation is handled Rust-side
    /// via nimbus_trainer_cancel, so this callback never itself requests cancel).
    func report(_ snapshot: NimbusTrainProgress) -> Int32 {
        let message: String
        switch snapshot.phase {
        case 0: // NimbusPhaseLoading
            message = "Loading capture data"
        case 2: // NimbusPhaseExporting
            message = "Exporting splat model"
        default: // NimbusPhaseTraining
            if snapshot.splat_count > 0 {
                message = "Training splats: iteration \(snapshot.current_iter)/\(snapshot.total_iters) (\(snapshot.splat_count) splats)"
            } else {
                message = "Training splats: iteration \(snapshot.current_iter)/\(snapshot.total_iters)"
            }
        }
        progress(PipelineProgress(
            stage: .splatTraining,
            fractionCompleted: Double(snapshot.fraction),
            message: message,
            isIndeterminate: snapshot.total_iters == 0
        ))
        return 0
    }
}

/// C-callable trampoline (`@convention(c)`, no captures): recovers the
/// TrainingContext from `user_data` and forwards the snapshot.
func nimbusProgressTrampoline(_ progress: UnsafePointer<NimbusTrainProgress>?,
                              _ userData: UnsafeMutableRawPointer?) -> Int32 {
    guard let userData, let progress else { return 0 }
    let context = Unmanaged<TrainingContext>.fromOpaque(userData).takeUnretainedValue()
    return context.report(progress.pointee)
}
