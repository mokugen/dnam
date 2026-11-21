//! Error types returned by planner, wide-Gram, shader manifest, CPU, and `wgpu` runners.

use core::fmt;

use dnam_vm::VmError;

/// Unified error type for `dnam-train` solver kernels, planners, and execution runners.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrainError {
    /// Dimension, alignment, or buffer-length mismatch in a solver kernel call.
    InvalidDimensions {
        /// Human-readable operation or buffer name.
        context: &'static str,
        /// Expected frame, lane, or element count.
        expected: usize,
        /// Observed frame, lane, or element count.
        actual: usize,
    },
    /// Input sample, parameter, or loss term contained `NaN` or `+/-inf`.
    NonFiniteInput {
        /// Human-readable operation or parameter slice name.
        context: &'static str,
    },
    /// L-BFGS inner step sizes failed strict monotonic decrease (`alpha[i] > alpha[i + 1]`).
    NonMonotonicStepSize {
        /// Previous candidate step size index.
        prev: usize,
        /// Violating subsequent candidate step size index.
        next: usize,
    },
    /// Planned buffer binding exceeded the target execution profile's binding cap.
    BindingExceedsCap {
        /// Binding buffer label.
        buffer: &'static str,
        /// Computed binding size in bytes.
        bytes: u64,
        /// Maximum allowed binding size in bytes.
        cap_bytes: u64,
    },
    /// Kernel configuration or dispatch precondition violated (`ref[train.pipeline.webgpu-scans-and-ozaki]`).
    KernelMismatch {
        /// Failing kernel or runner entry point.
        context: &'static str,
        /// Specific contract violation detail.
        detail: &'static str,
    },
    /// Embedded `.wgsl` SHA-256 digest did not match the checked-in `wgsl/manifest.json`.
    ShaderManifestMismatch {
        /// Kernel or variant identifier.
        kernel: &'static str,
        /// Digest recorded in `wgsl/manifest.json`.
        expected: [u8; 32],
        /// Computed SHA-256 digest over the embedded WGSL source.
        actual: [u8; 32],
    },
    /// No suitable `wgpu` adapter/device could be acquired on this host.
    GpuUnavailable(&'static str),
    /// Reading back a mapped GPU staging buffer failed.
    GpuBufferMapFailed(&'static str),
    /// Forwarded `dnam-vm` error.
    Vm(VmError),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions {
                context,
                expected,
                actual,
            } => write!(
                f,
                "{context}: dimension mismatch (expected {expected}, got {actual})"
            ),
            Self::NonFiniteInput { context } => {
                write!(f, "{context}: non-finite input encountered")
            }
            Self::NonMonotonicStepSize { prev, next } => write!(
                f,
                "inner_opt_step: non-monotonic step sizes at indices {prev} -> {next}"
            ),
            Self::BindingExceedsCap {
                buffer,
                bytes,
                cap_bytes,
            } => write!(
                f,
                "{buffer}: binding size {bytes} B exceeds cap {cap_bytes} B"
            ),
            Self::KernelMismatch { context, detail } => write!(f, "{context}: {detail}"),
            Self::ShaderManifestMismatch { kernel, .. } => {
                write!(
                    f,
                    "{kernel}: embedded WGSL SHA-256 mismatch against manifest"
                )
            }
            Self::GpuUnavailable(reason) => write!(f, "wgpu adapter unavailable: {reason}"),
            Self::GpuBufferMapFailed(reason) => write!(f, "wgpu staging map failed: {reason}"),
            Self::Vm(err) => write!(f, "vm error: {err}"),
        }
    }
}

impl std::error::Error for TrainError {}

impl From<VmError> for TrainError {
    fn from(value: VmError) -> Self {
        Self::Vm(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_error_display_and_vm_conversion() {
        let cases = [
            (
                TrainError::InvalidDimensions {
                    context: "fused_core_scan",
                    expected: 256,
                    actual: 128,
                },
                "expected 256, got 128",
            ),
            (
                TrainError::NonFiniteInput {
                    context: "prescreen",
                },
                "non-finite input",
            ),
            (
                TrainError::NonMonotonicStepSize { prev: 0, next: 1 },
                "indices 0 -> 1",
            ),
            (
                TrainError::BindingExceedsCap {
                    buffer: "tile_ckpts",
                    bytes: 200,
                    cap_bytes: 100,
                },
                "exceeds cap 100 B",
            ),
            (
                TrainError::KernelMismatch {
                    context: "gabor",
                    detail: "hop_size < fft_size",
                },
                "hop_size < fft_size",
            ),
            (
                TrainError::ShaderManifestMismatch {
                    kernel: "ozaki",
                    expected: [0; 32],
                    actual: [1; 32],
                },
                "SHA-256 mismatch",
            ),
            (TrainError::GpuUnavailable("no adapter"), "no adapter"),
            (TrainError::GpuBufferMapFailed("cancelled"), "cancelled"),
        ];
        for (err, substr) in cases {
            let msg = err.to_string();
            assert!(msg.contains(substr), "unexpected message: {msg}");
        }
        let vm_err = VmError::InvalidTier;
        let converted: TrainError = vm_err.into();
        assert!(matches!(converted, TrainError::Vm(_)));
        assert!(converted.to_string().contains("vm error:"));
    }
}
