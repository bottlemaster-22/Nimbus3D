//! nimbus-splat-core: C-FFI wrapper around the Brush engine
//! (github.com/ArthurBrussee/brush) for on-device 3D Gaussian Splat training.
//!
//! This file is the ABI boundary only: raw-pointer marshalling, panic
//! containment, and the opaque `NimbusTrainer` handle. All real work lives in
//! `brush_glue.rs`, which talks to Brush's `create_process` streaming API.
//!
//! Every `extern "C"` entry point wraps its body in `catch_unwind` so a panic
//! from deep inside brush / wgpu / burn becomes a `NimbusStatusPanic` return
//! instead of unwinding across the FFI boundary (which is UB).

mod brush_glue;

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// C ABI types (must byte-match include/nimbus_splat_core.h)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NimbusStatus {
    Ok = 0,
    InvalidArgument = 1,
    DatasetLoadFailed = 2,
    TrainingFailed = 3,
    ExportFailed = 4,
    Cancelled = 5,
    Panic = 6,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum NimbusPhase {
    Loading = 0,
    Training = 1,
    Exporting = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NimbusTrainConfig {
    pub iterations: u32,
    pub max_splat_count: u32,
    pub max_image_dimension: u32,
    pub sh_degree: u32,
    pub seed: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NimbusTrainProgress {
    pub phase: u32,
    pub current_iter: u32,
    pub total_iters: u32,
    pub splat_count: u32,
    pub sh_degree: u32,
    pub fraction: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NimbusTrainResult {
    pub splat_count: u32,
    pub sh_degree: u32,
    pub iterations_completed: u32,
}

/// `int32_t (*)(const NimbusTrainProgress*, void*)`; non-zero return => cancel.
pub type NimbusProgressCallback =
    Option<extern "C" fn(*const NimbusTrainProgress, *mut c_void) -> c_int>;

/// Structured failure carried out of `brush_glue` back to the C caller.
pub struct TrainError {
    pub status: NimbusStatus,
    pub message: String,
}

impl TrainError {
    pub fn new(status: NimbusStatus, message: impl Into<String>) -> Self {
        Self { status, message: message.into() }
    }
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Reusable training session. The only mutable shared state is the cancel flag,
/// which `nimbus_trainer_cancel` may set from any thread while a run is active.
pub struct NimbusTrainer {
    pub(crate) cancel: AtomicBool,
}

#[no_mangle]
pub extern "C" fn nimbus_trainer_create() -> *mut NimbusTrainer {
    let boxed = Box::new(NimbusTrainer { cancel: AtomicBool::new(false) });
    Box::into_raw(boxed)
}

/// # Safety
/// `trainer` must be a pointer returned by `nimbus_trainer_create` and not used
/// again after this call. Must not be called while a run is in flight.
#[no_mangle]
pub unsafe extern "C" fn nimbus_trainer_destroy(trainer: *mut NimbusTrainer) {
    if !trainer.is_null() {
        drop(Box::from_raw(trainer));
    }
}

/// # Safety
/// `trainer` must be a live handle. Safe to call from a thread other than the
/// one running `nimbus_trainer_run`.
#[no_mangle]
pub unsafe extern "C" fn nimbus_trainer_cancel(trainer: *mut NimbusTrainer) {
    if let Some(t) = trainer.as_ref() {
        t.cancel.store(true, Ordering::SeqCst);
    }
}

#[no_mangle]
pub extern "C" fn nimbus_splat_core_version() -> *const c_char {
    // Static, NUL-terminated, never freed.
    concat!("nimbus-splat-core ", env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

// ---------------------------------------------------------------------------
// Training entry point
// ---------------------------------------------------------------------------

/// # Safety
/// All pointers must satisfy the contract in the header. `dataset_path` and
/// `out_ply_path` must be valid NUL-terminated UTF-8 C strings; `config` must be
/// non-NULL. `progress`, when set, is invoked on the calling thread.
#[no_mangle]
pub unsafe extern "C" fn nimbus_trainer_run(
    trainer: *mut NimbusTrainer,
    dataset_path: *const c_char,
    out_ply_path: *const c_char,
    config: *const NimbusTrainConfig,
    progress: NimbusProgressCallback,
    user_data: *mut c_void,
    out_result: *mut NimbusTrainResult,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> c_int {
    let trainer = match trainer.as_ref() {
        Some(t) => t,
        None => return NimbusStatus::InvalidArgument as c_int,
    };
    let config = match config.as_ref() {
        Some(c) => *c,
        None => {
            write_error(error_buffer, error_buffer_len, "config pointer is null");
            return NimbusStatus::InvalidArgument as c_int;
        }
    };
    let dataset = match cstr_to_str(dataset_path) {
        Some(s) => s.to_owned(),
        None => {
            write_error(error_buffer, error_buffer_len, "dataset_path is null or not UTF-8");
            return NimbusStatus::InvalidArgument as c_int;
        }
    };
    let out_ply = match cstr_to_str(out_ply_path) {
        Some(s) => s.to_owned(),
        None => {
            write_error(error_buffer, error_buffer_len, "out_ply_path is null or not UTF-8");
            return NimbusStatus::InvalidArgument as c_int;
        }
    };

    // Fresh run: clear any leftover cancel request.
    trainer.cancel.store(false, Ordering::SeqCst);

    // Raw pointers (`user_data`) are not UnwindSafe; we assert it because a panic
    // only propagates OUT of the callback and we never observe torn state after.
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        brush_glue::run_training(trainer, &dataset, &out_ply, &config, progress, user_data)
    }));

    match outcome {
        Ok(Ok(result)) => {
            if let Some(slot) = out_result.as_mut() {
                *slot = result;
            }
            NimbusStatus::Ok as c_int
        }
        Ok(Err(err)) => {
            write_error(error_buffer, error_buffer_len, &err.message);
            err.status as c_int
        }
        Err(panic) => {
            let msg = panic_message(panic);
            write_error(error_buffer, error_buffer_len, &format!("panic in splat core: {msg}"));
            NimbusStatus::Panic as c_int
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Copy `msg` (truncated) as a NUL-terminated string into `buf`.
unsafe fn write_error(buf: *mut c_char, len: usize, msg: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let copy = bytes.len().min(len - 1); // reserve 1 for the terminator
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy);
    *buf.add(copy) = 0;
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}
