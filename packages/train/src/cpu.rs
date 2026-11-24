//! Deterministic CPU fallback compute runner (`CpuRunner`) executing all 7 32-bit compute
//! kernels plus wave-batched core cascade/adjoint (`ref[train.pipeline.webgpu-scans-and-ozaki]`,
//! `ref[train.pipeline.isomorphism-parity]`).

pub mod core_and_adjoint;
pub mod gabor_and_prescreen;
pub mod ozaki_and_opt;
pub mod tile_mix;

use crate::error::TrainError;
use crate::math::{BackendKind, detect_best_backend};
use crate::planner::CoreScanPlan;
use crate::runner::types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, GaborBezoutInput,
    GaborBezoutOutput, InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput,
    OzakiSplit64Output, PrescreenInput, TileMixInput, TileMixOutput, WaveCoreAdjointInput,
    WaveCoreAdjointOutput, WaveCoreCascadeInput, WaveCoreCascadeOutput,
};

/// Deterministic host CPU implementation of all 7 GPU/CPU training kernels.
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

    /// Evaluates `tile_mix_fwd_adj` deterministically on CPU.
    pub fn tile_mix_fwd_adj(&self, input: &TileMixInput<'_>) -> Result<TileMixOutput, TrainError> {
        tile_mix::run_tile_mix_fwd_adj_cpu(input)
    }

    /// Evaluates wave-batched forward cascade (`Waves 0..=3` + `tile_mix(0)`) on CPU.
    pub fn run_wave_core_cascade(
        &self,
        input: &WaveCoreCascadeInput<'_>,
    ) -> Result<WaveCoreCascadeOutput, TrainError> {
        tile_mix::run_wave_core_cascade_cpu(input)
    }

    /// Evaluates wave-batched reverse BPTT (`Waves 3..=0` + `tile_mix(1,2,3)`) on CPU.
    pub fn run_wave_core_adjoint(
        &self,
        input: &WaveCoreAdjointInput<'_>,
    ) -> Result<WaveCoreAdjointOutput, TrainError> {
        tile_mix::run_wave_core_adjoint_cpu(input)
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

    /// Evaluates full 2-pass split-64 `ozaki_gram_rhs_i32` deterministically on CPU.
    pub fn ozaki_gram_rhs_split64(
        &self,
        features_col_major: &[f32],
        targets_col_major: &[f32],
        num_frames: usize,
        feature_dim: usize,
        target_channels: usize,
        col_scales: &[[f32; 2]],
    ) -> Result<OzakiSplit64Output, TrainError> {
        ozaki_and_opt::run_ozaki_gram_rhs_split64_cpu(
            features_col_major,
            targets_col_major,
            num_frames,
            feature_dim,
            target_channels,
            col_scales,
        )
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

    /// Reconstructs `0-ULP` `f64` Gram + RHS totals directly from split-64 limb output.
    pub fn reconstruct_ozaki_split64_f64(
        &self,
        output: &OzakiSplit64Output,
        feature_dim: usize,
        target_channels: usize,
        col_scales: &[[f32; 2]],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
        ozaki_and_opt::reconstruct_ozaki_split64_f64(
            output,
            feature_dim,
            target_channels,
            col_scales,
        )
    }

    /// Evaluates multi-candidate `(1, num_candidates, 1)` `inner_opt_step` on CPU.
    pub fn inner_opt_step(&self, input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
        ozaki_and_opt::run_inner_opt_step_cpu(input)
    }
}
