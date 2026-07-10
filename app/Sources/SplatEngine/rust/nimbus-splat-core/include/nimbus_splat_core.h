/*
 * nimbus_splat_core.h
 *
 * C ABI for the Nimbus3D on-device 3D Gaussian Splat trainer. The Swift wrapper
 * (Sources/SplatEngine/BrushSplatTrainer.swift) imports this as the C module
 * `NimbusSplatCore` and drives it from a CaptureBundle to a .ply SplatModel.
 *
 * Threading: nimbus_trainer_run() BLOCKS the calling thread for the whole
 * training run (minutes). Call it off the main thread. The progress callback is
 * invoked on that same thread. nimbus_trainer_cancel() is the ONLY function safe
 * to call from another thread while a run is in flight.
 */
#ifndef NIMBUS_SPLAT_CORE_H
#define NIMBUS_SPLAT_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque training session. Owns the cancel flag; reusable across runs. */
typedef struct NimbusTrainer NimbusTrainer;

/* Return code from nimbus_trainer_run(). Mirrors NimbusStatus in Rust. */
typedef enum {
    NimbusStatusOk = 0,
    NimbusStatusInvalidArgument = 1,
    NimbusStatusDatasetLoadFailed = 2,
    NimbusStatusTrainingFailed = 3,
    NimbusStatusExportFailed = 4,
    NimbusStatusCancelled = 5,
    NimbusStatusPanic = 6
} NimbusStatus;

/* progress->phase values. */
typedef enum {
    NimbusPhaseLoading = 0,
    NimbusPhaseTraining = 1,
    NimbusPhaseExporting = 2
} NimbusPhase;

/* Training hyper-parameters. Mirrors SplatTrainingConfig on the Swift side. */
typedef struct {
    uint32_t iterations;            /* total optimization steps */
    uint32_t max_splat_count;       /* hard cap on splats (0 = engine default)  [SEAM: not yet applied] */
    uint32_t max_image_dimension;   /* long-edge px cap for training images (0 = default) [SEAM: not yet applied] */
    uint32_t sh_degree;             /* spherical-harmonics degree 0..3           [SEAM: not yet applied] */
    uint64_t seed;                  /* RNG seed for reproducibility */
} NimbusTrainConfig;

/* Snapshot handed to the progress callback. */
typedef struct {
    uint32_t phase;                 /* NimbusPhase */
    uint32_t current_iter;
    uint32_t total_iters;
    uint32_t splat_count;           /* current live splat count (0 until known) */
    uint32_t sh_degree;
    float    fraction;              /* 0..1 within the run */
} NimbusTrainProgress;

/* Populated on NimbusStatusOk. */
typedef struct {
    uint32_t splat_count;
    uint32_t sh_degree;
    uint32_t iterations_completed;
} NimbusTrainResult;

/*
 * Progress callback. Return 0 to keep training, non-zero to request cancellation
 * (training then unwinds and nimbus_trainer_run() returns NimbusStatusCancelled).
 */
typedef int32_t (*NimbusProgressCallback)(const NimbusTrainProgress *progress,
                                          void *user_data);

/* Create / destroy a session. destroy() must not be called during a run. */
NimbusTrainer *nimbus_trainer_create(void);
void nimbus_trainer_destroy(NimbusTrainer *trainer);

/* Thread-safe: request cancellation of an in-flight nimbus_trainer_run(). */
void nimbus_trainer_cancel(NimbusTrainer *trainer);

/*
 * Train a splat model. BLOCKS until done / cancelled / failed.
 *
 *   dataset_path   directory Brush can load (Nerfstudio transforms.json + images,
 *                  or COLMAP). See BrushSplatTrainer.NerfstudioDatasetExporter.
 *   out_ply_path   absolute path the final .ply is written to.
 *   config         hyper-parameters (must be non-NULL).
 *   progress       may be NULL.
 *   user_data      passed back to `progress` verbatim.
 *   out_result     filled on success (may be NULL to ignore).
 *   error_buffer   on failure, a NUL-terminated message is copied here (may be
 *                  NULL); truncated to error_buffer_len.
 *
 * Returns a NimbusStatus code.
 */
int32_t nimbus_trainer_run(NimbusTrainer *trainer,
                           const char *dataset_path,
                           const char *out_ply_path,
                           const NimbusTrainConfig *config,
                           NimbusProgressCallback progress,
                           void *user_data,
                           NimbusTrainResult *out_result,
                           char *error_buffer,
                           size_t error_buffer_len);

/* Static version string ("nimbus-splat-core x.y.z"); never NULL, do not free. */
const char *nimbus_splat_core_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIMBUS_SPLAT_CORE_H */
