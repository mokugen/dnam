//! Unified runtime-dispatched [`TrainRunner`] (`Wgpu` with automatic `< 1 ms` startup probe +
//! fallback to deterministic SIMD [`CpuRunner`]) and kernel I/O bundle types (`types`).

pub mod types;

use core::fmt;

pub use types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, GaborBezoutInput,
    GaborBezoutOutput, InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput,
    OzakiSplit64Output, PrescreenInput, TileMixInput, TileMixOutput, WaveCoreAdjointInput,
    WaveCoreAdjointOutput, WaveCoreCascadeInput, WaveCoreCascadeOutput, pack_core_quad_params,
    pack_tile_descriptor, unpack_core_quad_params,
};

use crate::cpu::CpuRunner;
use crate::error::TrainError;
#[cfg(feature = "gpu-wgpu")]
use crate::gpu::{ShaderVariant, WgpuRunner};
use crate::math::BackendKind;
use crate::planner::CoreScanPlan;

/// Active compute backend descriptor returned by [`TrainRunner::active_backend`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveComputeBackend {
    /// Deterministic host CPU SIMD compute fallback.
    CpuSimd {
        /// Selected host CPU SIMD instruction level.
        backend_kind: BackendKind,
    },
    /// Native `wgpu 30.0.2` GPU compute backend (`ShaderVariant::Subgroups` or `Portable`).
    #[cfg(feature = "gpu-wgpu")]
    Wgpu {
        /// Active WGSL variant after startup probes.
        variant: ShaderVariant,
        /// Effective per-binding storage cap reported by the adapter (`<= 128 MiB`).
        max_storage_binding_bytes: u64,
        /// Adapter-reported `max_compute_workgroup_storage_size` (`>= 16,384 B`).
        max_workgroup_storage_bytes: u32,
    },
}

/// Unified compute runner dispatching all 7 32-bit SONA kernels + wave-batched cascades transparently
/// across `wgpu 30.0.2` hardware adapters and deterministic CPU SIMD (`ref[train.pipeline.webgpu-scans-and-ozaki]`).
pub enum TrainRunner {
    /// Deterministic host CPU runner.
    Cpu(CpuRunner),
    /// Native `wgpu 30.0.2` compute runner.
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(WgpuRunner),
}

impl fmt::Debug for TrainRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu(cpu) => f.debug_tuple("TrainRunner::Cpu").field(cpu).finish(),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => f.debug_tuple("TrainRunner::Wgpu").field(gpu).finish(),
        }
    }
}

impl TrainRunner {
    /// Probes native `wgpu` (`Subgroups` -> `Portable`, `< 1 ms` startup probes) using the
    /// process-wide cached [`WgpuRunner`] / `GpuTrainArena`, and falls back transparently to
    /// [`CpuRunner::detect`] when no qualifying GPU adapter is available.
    #[must_use]
    pub fn auto_detect() -> Self {
        #[cfg(feature = "gpu-wgpu")]
        if let Ok(gpu) = WgpuRunner::try_shared() {
            return Self::Wgpu(gpu);
        }
        Self::Cpu(CpuRunner::detect())
    }

    /// Forces deterministic host CPU execution using runtime SIMD auto-detection.
    #[inline]
    #[must_use]
    pub fn cpu() -> Self {
        Self::Cpu(CpuRunner::detect())
    }

    /// Returns metadata describing the currently selected compute backend.
    #[must_use]
    pub fn active_backend(&self) -> ActiveComputeBackend {
        match self {
            Self::Cpu(cpu) => ActiveComputeBackend::CpuSimd {
                backend_kind: cpu.backend_kind(),
            },
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => ActiveComputeBackend::Wgpu {
                variant: gpu.context().variant(),
                max_storage_binding_bytes: gpu.context().max_storage_binding_bytes(),
                max_workgroup_storage_bytes: gpu.context().max_workgroup_storage_bytes(),
            },
        }
    }

    /// Evaluates `prescreen_population` on the active compute backend.
    pub fn prescreen_population(&self, input: &PrescreenInput<'_>) -> Result<Vec<f32>, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.prescreen_population(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.prescreen_population(input),
        }
    }

    /// Evaluates `fused_core_scan_256` on the active compute backend.
    pub fn fused_core_scan_256(
        &self,
        input: &FusedCoreScanInput<'_>,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.fused_core_scan_256(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.fused_core_scan_256(input),
        }
    }

    /// Executes `fused_core_scan_256` according to a 2-D [`CoreScanPlan`].
    pub fn run_fused_core_scan_planned(
        &self,
        input: &FusedCoreScanInput<'_>,
        plan: &CoreScanPlan,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.run_fused_core_scan_planned(input, plan),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.run_fused_core_scan_planned(input, plan),
        }
    }

    /// Evaluates reverse BPTT `adjoint_scan_256` on the active compute backend.
    pub fn adjoint_scan_256(
        &self,
        input: &AdjointScanInput<'_>,
    ) -> Result<AdjointScanOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.adjoint_scan_256(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.adjoint_scan_256(input),
        }
    }

    /// Evaluates `tile_mix_fwd_adj` on the active compute backend.
    pub fn tile_mix_fwd_adj(&self, input: &TileMixInput<'_>) -> Result<TileMixOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.tile_mix_fwd_adj(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.tile_mix_fwd_adj(input),
        }
    }

    /// Executes wave-batched forward cascade (`Waves 0..=3` + `tile_mix(0)`) on the active backend.
    pub fn run_wave_core_cascade(
        &self,
        input: &WaveCoreCascadeInput<'_>,
    ) -> Result<WaveCoreCascadeOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.run_wave_core_cascade(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.run_wave_core_cascade(input),
        }
    }

    /// Executes wave-batched reverse BPTT (`Waves 3..=0` + `tile_mix(1,2,3)`) on the active backend.
    pub fn run_wave_core_adjoint(
        &self,
        input: &WaveCoreAdjointInput<'_>,
    ) -> Result<WaveCoreAdjointOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.run_wave_core_adjoint(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.run_wave_core_adjoint(input),
        }
    }

    /// Evaluates `gabor_bezout_grad` (`Mode A` or `Mode B`) on the active compute backend.
    pub fn gabor_bezout_grad(
        &self,
        input: &GaborBezoutInput<'_>,
    ) -> Result<GaborBezoutOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.gabor_bezout_grad(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.gabor_bezout_grad(input),
        }
    }

    /// Evaluates integer-exact `ozaki_gram_rhs_i32` on the active compute backend.
    pub fn ozaki_gram_rhs_i32(
        &self,
        input: &OzakiGramInput<'_>,
    ) -> Result<OzakiGramOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.ozaki_gram_rhs_i32(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.ozaki_gram_rhs_i32(input),
        }
    }

    /// Executes 2-pass batched split-64 `ozaki_gram_rhs_i32` (`<= 1.2 MiB` Ultra readback).
    pub fn ozaki_gram_rhs_split64(
        &self,
        features_col_major: &[f32],
        targets_col_major: &[f32],
        num_frames: usize,
        feature_dim: usize,
        target_channels: usize,
        col_scales: &[[f32; 2]],
    ) -> Result<OzakiSplit64Output, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.ozaki_gram_rhs_split64(
                features_col_major,
                targets_col_major,
                num_frames,
                feature_dim,
                target_channels,
                col_scales,
            ),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.ozaki_gram_rhs_split64(
                features_col_major,
                targets_col_major,
                num_frames,
                feature_dim,
                target_channels,
                col_scales,
            ),
        }
    }

    /// Reconstructs column-major packed `uplo='U'` `f64` Gram matrix and RHS vectors on host.
    pub fn reconstruct_ozaki_gram_rhs_f64(
        &self,
        output: &OzakiGramOutput,
        feature_dim: usize,
        target_channels: usize,
        col_scales: &[[f32; 2]],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
        CpuRunner::detect().reconstruct_ozaki_gram_rhs_f64(
            output,
            feature_dim,
            target_channels,
            col_scales,
        )
    }

    /// Reconstructs `0-ULP` `f64` Gram matrix (`uplo='U'`) and RHS vectors from split-64 limbs.
    pub fn reconstruct_ozaki_split64_f64(
        &self,
        output: &OzakiSplit64Output,
        feature_dim: usize,
        target_channels: usize,
        col_scales: &[[f32; 2]],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrainError> {
        CpuRunner::detect().reconstruct_ozaki_split64_f64(
            output,
            feature_dim,
            target_channels,
            col_scales,
        )
    }

    /// Evaluates multi-candidate `(1, num_candidates, 1)` `inner_opt_step` on the active backend.
    pub fn inner_opt_step(&self, input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
        match self {
            Self::Cpu(cpu) => cpu.inner_opt_step(input),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(gpu) => gpu.inner_opt_step(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::abi::{
        COORD_TRANSFORM_FREE, INNER_OPT_MODE_LSSBROYDEN, OZAKI_PAIR_TILE_ROWS_DEFAULT,
        OzakiGramUniforms,
    };

    #[test]
    fn test_train_runner_cpu_and_auto_detect_dispatch() {
        let cpu = TrainRunner::cpu();
        assert!(matches!(
            cpu.active_backend(),
            ActiveComputeBackend::CpuSimd { .. }
        ));
        let auto = TrainRunner::auto_detect();
        assert!(!format!("{auto:?}").is_empty());
        let feats = [1.0_f32; 8];
        let scales = [[1.0_f32, 1.0_f32]; 2];
        let out = auto
            .ozaki_gram_rhs_i32(&OzakiGramInput {
                uniforms: OzakiGramUniforms {
                    num_frames: 4,
                    feature_dim: 2,
                    target_channels: 0,
                    chunk_len: 1024,
                    num_chunks: 1,
                    pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
                    reduce_pass: 0,
                    _pad: 0,
                },
                features: &feats,
                targets: &[],
                col_scales: &scales,
                seed_buckets: None,
            })
            .unwrap();
        let (g, _) = auto
            .reconstruct_ozaki_gram_rhs_f64(&out, 2, 0, &scales)
            .unwrap();
        assert!((g[0] - 4.0).abs() <= 1.0e-5);
        let s64 = auto
            .ozaki_gram_rhs_split64(&feats, &[], 4, 2, 0, &scales)
            .unwrap();
        let (g_s64, _) = auto
            .reconstruct_ozaki_split64_f64(&s64, 2, 0, &scales)
            .unwrap();
        assert_eq!(g_s64, g);

        // Exercise all TrainRunner dispatch arms cleanly on CPU backend.
        let mut p_in = [0.0_f32; 32];
        p_in[0] = 0.05;
        p_in[1] = 0.8;
        p_in[2] = 0.5;
        p_in[6] = 0.8;
        p_in[12] = 1.0;
        let audio = vec![[0.2_f32, -0.1_f32]; 16];
        let pre = cpu
            .prescreen_population(&PrescreenInput {
                uniforms: crate::abi::PrescreenUniforms {
                    num_seeds: 1,
                    num_frames: 16,
                    num_quads: 1,
                    num_stems: 1,
                    sample_rate_hz: 48_000.0,
                    eps_floor: 1.0e-5,
                    _pad0: 0,
                    _pad1: 0,
                },
                input_audio: &audio,
                target_audio: &audio,
                seed_params: &p_in[..16],
            })
            .unwrap();
        assert_eq!(pre.len(), 1);

        let quad: [[f32; 4]; crate::gpu::abi::QUAD_PARAMS_VEC4_STRIDE] = [
            [0.8, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.2, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [0.2; 4],
            [1.0; 4],
            [0.2; 4],
            [1.2; 4],
            [0.1; 4],
            [0.2; 4],
            [0.3; 4],
            [0.1; 4],
            [0.0; 4],
        ];
        let core_in = vec![[0.25_f32; 4]; 32];
        let fwd_u = crate::abi::FusedScanUniforms {
            num_frames: 16,
            num_tiles: 1,
            num_quads: 1,
            num_stems: 1,
            tile_start: 0,
            soft_rect_eps: dnam_kernels::SOFT_RECT_EPS,
            tiny_crossing_eps: dnam_kernels::TINY_CROSSING_EPS,
            block_idx: 0,
        };
        let fwd = cpu
            .fused_core_scan_256(&FusedCoreScanInput {
                uniforms: fwd_u,
                core_inputs: &core_in,
                quad_params: &quad,
                initial_checkpoints: None,
            })
            .unwrap();
        let plan = crate::plan_core_scan_execution(
            crate::CoreScanWorkload {
                num_frames: 16,
                num_quads: 1,
                num_stems: 1,
                include_adjoint: false,
            },
            crate::MemoryBudgetProfile::DesktopCapsule,
            None,
        )
        .unwrap();
        let planned = cpu
            .run_fused_core_scan_planned(
                &FusedCoreScanInput {
                    uniforms: fwd_u,
                    core_inputs: &core_in,
                    quad_params: &quad,
                    initial_checkpoints: None,
                },
                &plan,
            )
            .unwrap();
        assert_eq!(fwd, planned);
        let d_loss = vec![[0.1_f32; 4]; 16 * 6];
        let adj = cpu
            .adjoint_scan_256(&AdjointScanInput {
                uniforms: crate::abi::AdjointScanUniforms {
                    num_frames: 16,
                    num_tiles: 1,
                    num_quads: 1,
                    num_stems: 1,
                    pass_mode: 2,
                    soft_rect_eps: dnam_kernels::SOFT_RECT_EPS,
                    tiny_crossing_eps: dnam_kernels::TINY_CROSSING_EPS,
                    block_idx: 0,
                },
                core_inputs: &core_in,
                quad_params: &quad,
                tile_checkpoints: &fwd.tile_checkpoints,
                d_loss_d_features: &d_loss,
                feature_outputs: &fwd.feature_outputs,
                exit_adj_boundary: None,
            })
            .unwrap();
        assert_eq!(adj.d_loss_d_inputs.len(), 32);
        let gb = cpu
            .gabor_bezout_grad(&GaborBezoutInput {
                uniforms: crate::abi::GaborBezoutUniforms {
                    num_frames: 16,
                    fft_size: 16,
                    hop_size: 16,
                    num_stems: 1,
                    w_lf: 1.0,
                    w_mf: 0.7,
                    w_hf: 0.4,
                    eps_noise: 1.0e-5,
                },
                pred_audio: &audio,
                target_audio: &audio,
                bezout_taps: &[[1.0; 4]; 16],
            })
            .unwrap();
        assert_eq!(gb.band_losses.len(), 1);
        let opt = cpu
            .inner_opt_step(&InnerOptInput {
                uniforms: crate::abi::InnerOptUniforms {
                    num_params: 4,
                    history_m: 2,
                    step_idx: 0,
                    mode_flags: INNER_OPT_MODE_LSSBROYDEN,
                    step_size: 0.2,
                    damping_lambda: 1.0e-4,
                    min_bound_eps: 1.0e-4,
                    grad_clip_norm: 0.0,
                },
                num_candidates: 1,
                theta: &[0.5; 4],
                grad: &[0.1; 4],
                param_bounds: &[[-1.0, 1.0, 1.0, f32::from_bits(COORD_TRANSFORM_FREE)]; 4],
                lbfgs_s_y: &[[0.0; 4]; 12],
                search_dir: &[0.0; 4],
            })
            .unwrap();
        assert_eq!(opt.theta.len(), 4);
        assert!(
            cpu.ozaki_gram_rhs_i32(&OzakiGramInput {
                uniforms: OzakiGramUniforms {
                    num_frames: 4,
                    feature_dim: 2,
                    target_channels: 0,
                    chunk_len: 1024,
                    num_chunks: 1,
                    pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
                    reduce_pass: 0,
                    _pad: 0,
                },
                features: &feats,
                targets: &[],
                col_scales: &scales,
                seed_buckets: None,
            })
            .is_ok()
        );
        assert!(
            !format!(
                "{:?}",
                TrainRunner::Cpu(CpuRunner::from_kind(BackendKind::Scalar))
            )
            .is_empty()
        );
    }
}
