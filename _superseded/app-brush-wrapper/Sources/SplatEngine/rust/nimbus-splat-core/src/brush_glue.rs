//! Brush integration: drive `brush_process::create_process` from a dataset on
//! disk to a trained .ply, forwarding progress to the C callback and honouring
//! cooperative cancellation.
//!
//! Grounded against Brush's public API (crates/brush-process, `main` branch,
//! researched 2026-07):
//!   * `create_process(source: DataSource, config_fn) -> RunningProcess`
//!         where `config_fn: FnOnce(TrainStreamConfig) -> Future<Option<TrainStreamConfig>>`
//!   * `RunningProcess { stream: Pin<Box<dyn ProcessStream>>, splat_view: Slot<Splats> }`
//!         `ProcessStream: Stream<Item = Result<ProcessMessage, anyhow::Error>>`
//!   * `ProcessMessage::{SplatsUpdated { num_splats, sh_degree, .. },
//!                       TrainMessage(TrainMessage), Warning { error }, DoneLoading, ..}`
//!   * `TrainMessage::{TrainStep { iter, .. }, RefineStep { cur_splat_count, iter },
//!                     DoneTraining, ..}`
//!   * `config::TrainStreamConfig { train_config, model_config, load_config,
//!                                  process_config, .. }`
//!   * `config::ProcessConfig { seed, eval_every, export_every, export_path,
//!                              export_name, .. }`  (export writes a .ply to disk)
//!   * `brush_train::config::TrainConfig { total_train_iters: u32, .. }`
//!   * `burn_init_setup() -> WgpuDevice`, `DataSource::Path(String)`
//!
//! SEAMs are called out inline where a field/behaviour could not be verified
//! from source and needs confirming on the first real macOS/device build.

use std::os::raw::{c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use crate::{
    NimbusPhase, NimbusProgressCallback, NimbusStatus, NimbusTrainConfig, NimbusTrainProgress,
    NimbusTrainResult, NimbusTrainer, TrainError,
};

/// Blocking training run. Builds a single-threaded tokio runtime (Brush's
/// pipeline is async) and pumps the process message stream to completion.
pub fn run_training(
    trainer: &NimbusTrainer,
    dataset_path: &str,
    out_ply_path: &str,
    config: &NimbusTrainConfig,
    progress: NimbusProgressCallback,
    user_data: *mut c_void,
) -> Result<NimbusTrainResult, TrainError> {
    // SEAM(runtime): see Cargo.toml. Multi-thread + timers so Brush's internal
    // spawns / dataloader / any tokio::time usage all work while we block_on.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_time()
        .build()
        .map_err(|e| TrainError::new(NimbusStatus::TrainingFailed, format!("tokio runtime: {e}")))?;

    runtime.block_on(run_async(
        trainer,
        dataset_path,
        out_ply_path,
        config,
        progress,
        user_data,
    ))
}

async fn run_async(
    trainer: &NimbusTrainer,
    dataset_path: &str,
    out_ply_path: &str,
    config: &NimbusTrainConfig,
    progress: NimbusProgressCallback,
    user_data: *mut c_void,
) -> Result<NimbusTrainResult, TrainError> {
    use brush_process::message::{ProcessMessage, TrainMessage};
    use futures::StreamExt;

    let total_iters = config.iterations.max(1);

    // Emit a progress snapshot; returns false when cancellation is requested
    // (either via the flag or a non-zero callback return).
    let emit = |phase: NimbusPhase, current: u32, splats: u32, sh: u32| -> bool {
        if trainer.cancel.load(Ordering::SeqCst) {
            return false;
        }
        if let Some(cb) = progress {
            let fraction = if total_iters > 0 {
                (current as f32 / total_iters as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let snapshot = NimbusTrainProgress {
                phase: phase as u32,
                current_iter: current,
                total_iters,
                splat_count: splats,
                sh_degree: sh,
                fraction,
            };
            let rc: c_int = cb(&snapshot as *const _, user_data);
            if rc != 0 {
                trainer.cancel.store(true, Ordering::SeqCst);
                return false;
            }
        }
        true
    };

    emit(NimbusPhase::Loading, 0, 0, 0);

    // SEAM(brush-api): device bring-up. Brush lazily initialises a global wgpu
    // device; `burn_init_setup()` performs that setup and hands back the device.
    // If a future Brush revision makes `create_process` self-initialise, this
    // call becomes a harmless no-op. TODO(nimbus): confirm this is the intended
    // one-time init entry point on device.
    let _device = brush_process::burn_init_setup().await;

    // Where Brush's built-in exporter should drop the final .ply.
    let export_dir = export_directory(out_ply_path)?;
    let export_dir_for_cfg = export_dir.to_string_lossy().into_owned();

    let seed = config.seed;
    let source = brush_process::DataSource::Path(dataset_path.to_owned());

    // Apply our hyper-parameters onto the config Brush derived from the dataset.
    let mut process = brush_process::create_process(
        source,
        move |mut cfg: brush_process::config::TrainStreamConfig| async move {
            // Confirmed field: total optimization steps.
            cfg.train_config.total_train_iters = total_iters;
            // Confirmed ProcessConfig fields: reproducibility + on-disk export.
            cfg.process_config.seed = seed;
            cfg.process_config.export_every = total_iters; // export exactly once, at the end
            cfg.process_config.eval_every = total_iters; // skip mid-run eval passes
            cfg.process_config.export_path = ensure_trailing_slash(&export_dir_for_cfg);
            cfg.process_config.export_name = "nimbus_export_{iter}.ply".to_string();

            // SEAM(brush-api): sh_degree lives on `model_config`; the field name
            // ("sh_degree" vs "max_sh_degree") is unverified from source.
            // TODO(nimbus): once confirmed, apply it:
            //     cfg.model_config.sh_degree = sh_degree;
            //
            // SEAM(brush-api): a hard splat-count cap and training-image
            // downscale are exposed on model_config / load_config under
            // unverified names. TODO(nimbus): wire NimbusTrainConfig
            // .max_splat_count and .max_image_dimension here. Until then the app
            // relies on Brush's own densification/limits (honest: these two
            // knobs are accepted by the API but NOT yet applied to training).
            Some(cfg)
        },
    );

    let mut last_splats: u32 = 0;
    let mut last_sh: u32 = config.sh_degree; // best-effort until Brush reports it
    let mut last_iter: u32 = 0;
    let mut done_training = false;

    while let Some(message) = process.stream.next().await {
        let message = message
            .map_err(|e| TrainError::new(NimbusStatus::TrainingFailed, format!("brush stream: {e}")))?;

        match message {
            ProcessMessage::SplatsUpdated { num_splats, sh_degree, .. } => {
                last_splats = num_splats;
                last_sh = sh_degree;
                if !emit(NimbusPhase::Training, last_iter, last_splats, last_sh) {
                    return Err(cancelled());
                }
            }
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainStep { iter, .. } => {
                    last_iter = iter;
                    if !emit(NimbusPhase::Training, iter, last_splats, last_sh) {
                        return Err(cancelled());
                    }
                }
                TrainMessage::RefineStep { cur_splat_count, iter } => {
                    last_splats = cur_splat_count;
                    last_iter = iter;
                    if !emit(NimbusPhase::Training, iter, last_splats, last_sh) {
                        return Err(cancelled());
                    }
                }
                TrainMessage::DoneTraining => {
                    done_training = true;
                    break;
                }
                _ => {}
            },
            ProcessMessage::Warning { error } => {
                // Non-fatal: Brush surfaces recoverable issues (e.g. a frame it
                // could not decode) as warnings. Keep training.
                eprintln!("[nimbus-splat-core] brush warning: {error}");
            }
            _ => {}
        }
    }

    if trainer.cancel.load(Ordering::SeqCst) {
        return Err(cancelled());
    }
    if !done_training {
        // Stream ended without a DoneTraining message: treat as failure rather
        // than silently exporting a half-trained model.
        return Err(TrainError::new(
            NimbusStatus::TrainingFailed,
            "brush training stream ended before completion",
        ));
    }

    emit(NimbusPhase::Exporting, total_iters, last_splats, last_sh);

    // Brush's exporter wrote a .ply into export_dir at the final step. Pick the
    // newest one and move it to the caller's requested path.
    let produced = newest_ply(&export_dir).ok_or_else(|| {
        TrainError::new(
            NimbusStatus::ExportFailed,
            format!(
                "brush produced no .ply in {} (expected export at iter {total_iters})",
                export_dir.display()
            ),
        )
    })?;

    if produced != Path::new(out_ply_path) {
        std::fs::copy(&produced, out_ply_path).map_err(|e| {
            TrainError::new(
                NimbusStatus::ExportFailed,
                format!("copying {} -> {out_ply_path}: {e}", produced.display()),
            )
        })?;
    }

    emit(NimbusPhase::Exporting, total_iters, last_splats, last_sh);

    Ok(NimbusTrainResult {
        splat_count: last_splats,
        sh_degree: last_sh,
        iterations_completed: last_iter.max(total_iters),
    })
}

fn cancelled() -> TrainError {
    TrainError::new(NimbusStatus::Cancelled, "training cancelled")
}

/// Parent directory of `out_ply_path`, created if needed. Brush's exporter
/// writes into a directory, so we hand it the model's containing folder.
fn export_directory(out_ply_path: &str) -> Result<PathBuf, TrainError> {
    let dir = Path::new(out_ply_path)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&dir).map_err(|e| {
        TrainError::new(
            NimbusStatus::ExportFailed,
            format!("creating export dir {}: {e}", dir.display()),
        )
    })?;
    Ok(dir)
}

fn ensure_trailing_slash(dir: &str) -> String {
    if dir.ends_with('/') {
        dir.to_string()
    } else {
        format!("{dir}/")
    }
}

/// Newest `*.ply` under `dir` (Brush's export template may embed the iteration
/// number, so we resolve by modification time rather than a fixed name).
fn newest_ply(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ply") {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        match &best {
            Some((t, _)) if *t >= modified => {}
            _ => best = Some((modified, path)),
        }
    }
    best.map(|(_, p)| p)
}
