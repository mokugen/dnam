//! Unified runtime-dispatched [`TrainRunner`] (`Wgpu` with automatic `< 1 ms` startup probe +
//! fallback to deterministic SIMD [`CpuRunner`]) and kernel I/O bundle types (`types`).

pub mod types;

use core::fmt;

pub use types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, GaborBezoutInput,
    GaborBezoutOutput, InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput,
    PrescreenInput, pack_core_quad_params, unpack_core_quad_params,
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

/// Unified compute runner dispatching all 6 32-bit SONA kernels transparently across
/// `wgpu 30.0.2` hardware adapters and deterministic CPU SIMD (`ref[train.pipeline.webgpu-scans-and-ozaki]`).
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
    /// Probes native `wgpu` (`Subgroups` -> `Portable`, `< 1 ms` startup probes) and falls back
    /// transparently to [`CpuRunner::detect`] when no qualifying GPU adapter is available.
    #[must_use]
    pub fn auto_detect() -> Self {
        #[cfg(feature = "gpu-wgpu")]
        if let Ok(gpu) = WgpuRunner::try_new() {
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

    /// Evaluates bounded `m <= 8` L-BFGS `inner_opt_step` on the active compute backend.
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
    use crate::gpu::abi::OzakiGramUniforms;

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
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
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

        let quad = [
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
            [0.2, 0.0, 0.0, 0.0],
            [1.0; 4],
            [0.2; 4],
            [1.2; 4],
            [0.1; 4],
            [0.2; 4],
            [0.3; 4],
            [0.1; 4],
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
                    line_search_iter: 0,
                    step_size: 0.2,
                    damping_lambda: 1.0e-4,
                    min_bound_eps: 1.0e-4,
                    _pad: 0,
                },
                theta: &[0.5; 4],
                grad: &[0.1; 4],
                param_bounds: &[[-1.0, 1.0]; 4],
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
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
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
