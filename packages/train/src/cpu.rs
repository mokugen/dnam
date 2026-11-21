//! Deterministic CPU fallback compute runner (`CpuRunner`) executing all 6 32-bit compute
//! kernels (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[train.pipeline.isomorphism-parity]`).

pub mod core_and_adjoint;
pub mod gabor_and_prescreen;
pub mod ozaki_and_opt;

use crate::error::TrainError;
use crate::math::{BackendKind, detect_best_backend};
use crate::planner::CoreScanPlan;
use crate::runner::types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, GaborBezoutInput,
    GaborBezoutOutput, InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput,
    PrescreenInput,
};

/// Deterministic host CPU implementation of all 6 GPU/CPU training kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuRunner {
    backend_kind: BackendKind,
}

impl CpuRunner {
    /// Constructs a [`CpuRunner`] pinned to the fastest host CPU SIMD [`BackendKind`].
    #[inline]
    #[must_use]
    pub fn detect() -> Self {
        Self {
            backend_kind: detect_best_backend(),
        }
    }

    /// Constructs a [`CpuRunner`] targeting the portable scalar backend.
    #[inline]
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            backend_kind: BackendKind::Scalar,
        }
    }

    /// Constructs a [`CpuRunner`] targeting an explicit [`BackendKind`].
    #[inline]
    #[must_use]
    pub const fn from_kind(backend_kind: BackendKind) -> Self {
        Self { backend_kind }
    }

    /// Returns the underlying host [`BackendKind`].
    #[inline]
    #[must_use]
    pub const fn backend_kind(self) -> BackendKind {
        self.backend_kind
    }

    /// Evaluates `prescreen_population` deterministically on CPU.
    pub fn prescreen_population(&self, input: &PrescreenInput<'_>) -> Result<Vec<f32>, TrainError> {
        gabor_and_prescreen::run_prescreen_population_cpu(input)
    }

    /// Evaluates `fused_core_scan_256` deterministically on CPU.
    pub fn fused_core_scan_256(
        &self,
        input: &FusedCoreScanInput<'_>,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        core_and_adjoint::run_fused_core_scan_cpu(input)
    }

    /// Evaluates `fused_core_scan_256` according to a 2-D [`CoreScanPlan`].
    pub fn run_fused_core_scan_planned(
        &self,
        input: &FusedCoreScanInput<'_>,
        plan: &CoreScanPlan,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        core_and_adjoint::run_fused_core_scan_planned_cpu(input, plan)
    }

    /// Evaluates exact reverse BPTT `adjoint_scan_256` on CPU.
    pub fn adjoint_scan_256(
        &self,
        input: &AdjointScanInput<'_>,
    ) -> Result<AdjointScanOutput, TrainError> {
        core_and_adjoint::run_adjoint_scan_cpu(input)
    }

    /// Evaluates `gabor_bezout_grad` (`Mode A` or `Mode B`) deterministically on CPU.
    pub fn gabor_bezout_grad(
        &self,
        input: &GaborBezoutInput<'_>,
    ) -> Result<GaborBezoutOutput, TrainError> {
        gabor_and_prescreen::run_gabor_bezout_grad_cpu(input)
    }

    /// Evaluates integer-exact `ozaki_gram_rhs_i32` deterministically on CPU.
    pub fn ozaki_gram_rhs_i32(
        &self,
        input: &OzakiGramInput<'_>,
    ) -> Result<OzakiGramOutput, TrainError> {
        ozaki_and_opt::run_ozaki_gram_rhs_i32_cpu(input)
    }

    /// Reconstructs `uplo='U'` `f64` Gram + RHS totals from `ozaki_buckets`.
    pub fn reconstruct_ozaki_gram_rhs_f64(
        &self,
        output: &OzakiGramOutput,
        feature_dim: usize,
        target_channels: usize,
        col_scales: &[[f32; 2]],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
        ozaki_and_opt::reconstruct_ozaki_gram_rhs_f64(
            output,
            feature_dim,
            target_channels,
            col_scales,
        )
    }

    /// Evaluates bounded `m <= 8` L-BFGS `inner_opt_step` deterministically on CPU.
    pub fn inner_opt_step(&self, input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
        ozaki_and_opt::run_inner_opt_step_cpu(input)
    }
}
