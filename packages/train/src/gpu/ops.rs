//! Native `wgpu 30.0.2` implementations of all 7 SONA compute kernels, persistent
//! [`GpuTrainArena`] buffer pooling, wave-batched core cascade/adjoint, and 2-pass split-64 Ozaki
//! reductions (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

#[cfg(target_arch = "wasm32")]
use std::cell::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

use super::SharedGpuRef;

use super::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms, QUAD_PARAMS_VEC4_STRIDE,
    TILE_CKPT_VEC4_STRIDE, TILE_FRAMES,
};
use super::arena::GpuTrainArena;
use super::context::WgpuContext;
use super::shaders::ShaderVariant;
use crate::cpu::ozaki_and_opt;
use crate::error::TrainError;
use crate::planner::{CoreScanPlan, ExecutionMode, MemoryBudgetProfile};
use crate::runner::types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, GaborBezoutInput,
    GaborBezoutOutput, InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput,
    OzakiSplit64Output, PrescreenInput, TileMixInput, TileMixOutput, WaveCoreAdjointInput,
    WaveCoreAdjointOutput, WaveCoreCascadeInput, WaveCoreCascadeOutput,
};

#[cfg(not(target_arch = "wasm32"))]
static SHARED_WGPU_ARENA: OnceLock<Result<SharedGpuRef<GpuTrainArena>, &'static str>> =
    OnceLock::new();

#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED_WGPU_ARENA: OnceCell<Result<SharedGpuRef<GpuTrainArena>, &'static str>> =
        const { OnceCell::new() };
}

/// Native `wgpu` compute kernel runner delegating to a persistent [`GpuTrainArena`].
#[derive(Clone, Debug)]
pub struct WgpuRunner {
    arena: SharedGpuRef<GpuTrainArena>,
}

impl WgpuRunner {
    /// Returns a [`WgpuRunner`] backed by the process-wide cached [`WgpuContext`] and [`GpuTrainArena`].
    pub fn try_shared() -> Result<Self, TrainError> {
        let init = || match WgpuContext::shared_or_init() {
            Ok(ctx) => Ok(SharedGpuRef::new(GpuTrainArena::new(ctx))),
            Err(TrainError::GpuUnavailable(msg)) => Err(msg),
            Err(_) => Err("wgpu shared arena initialization failed"),
        };
        let to_result = |cached: &Result<SharedGpuRef<GpuTrainArena>, &'static str>| match cached {
            Ok(arena) => Ok(Self {
                arena: SharedGpuRef::clone(arena),
            }),
            Err(msg) => Err(TrainError::GpuUnavailable(msg)),
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            to_result(SHARED_WGPU_ARENA.get_or_init(init))
        }
        #[cfg(target_arch = "wasm32")]
        {
            SHARED_WGPU_ARENA.with(|cell| to_result(cell.get_or_init(init)))
        }
    }

    /// Initializes a dedicated [`WgpuRunner`] with automatic `Subgroups -> Portable` probe selection.
    pub fn try_new() -> Result<Self, TrainError> {
        Ok(Self {
            arena: SharedGpuRef::new(GpuTrainArena::new(SharedGpuRef::new(
                WgpuContext::try_new()?
            ))),
        })
    }

    /// Initializes a dedicated [`WgpuRunner`] pinned to an explicit [`ShaderVariant`].
    pub fn try_new_with_variant(variant: ShaderVariant) -> Result<Self, TrainError> {
        Ok(Self {
            arena: SharedGpuRef::new(GpuTrainArena::new(SharedGpuRef::new(
                WgpuContext::try_new_with_variant(variant)?,
            ))),
        })
    }

    /// Initializes a dedicated [`WgpuRunner`] bounded by a specific [`MemoryBudgetProfile`].
    pub fn with_budget_profile(profile: MemoryBudgetProfile) -> Result<Self, TrainError> {
        Ok(Self {
            arena: SharedGpuRef::new(GpuTrainArena::with_budget_profile(
                SharedGpuRef::new(WgpuContext::try_new()?),
                profile,
            )),
        })
    }

    /// Wraps an existing shared [`GpuTrainArena`].
    #[inline]
    #[must_use]
    pub const fn with_arena(arena: SharedGpuRef<GpuTrainArena>) -> Self {
        Self { arena }
    }

    /// Borrows the underlying [`WgpuContext`].
    #[inline]
    #[must_use]
    pub fn context(&self) -> &WgpuContext {
        self.arena.context()
    }

    /// Borrows the underlying persistent [`GpuTrainArena`].
    #[inline]
    #[must_use]
    pub const fn arena(&self) -> &SharedGpuRef<GpuTrainArena> {
        &self.arena
    }

    /// Returns total GPU bytes currently allocated inside the persistent arena.
    #[inline]
    #[must_use]
    pub fn arena_allocated_bytes(&self) -> u64 {
        self.arena.allocated_bytes()
    }

    /// Dispatches `prescreen_population` on GPU via the persistent arena.
    pub fn prescreen_population(&self, input: &PrescreenInput<'_>) -> Result<Vec<f32>, TrainError> {
        self.arena.prescreen_population(input)
    }

    /// Dispatches `fused_core_scan_256` on GPU via the persistent arena.
    pub fn fused_core_scan_256(
        &self,
        input: &FusedCoreScanInput<'_>,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        self.arena.fused_core_scan_256(input)
    }

    /// Dispatches `fused_core_scan_256` under a 2-D `ResidentBatched` / `Streamed` [`CoreScanPlan`].
    pub fn run_fused_core_scan_planned(
        &self,
        input: &FusedCoreScanInput<'_>,
        plan: &CoreScanPlan,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        if matches!(plan.mode, ExecutionMode::ResidentBatched) {
            return self.fused_core_scan_256(input);
        }
        let u = &input.uniforms;
        let num_frames = u.num_frames as usize;
        let num_tiles = num_frames.div_ceil(TILE_FRAMES).max(1);
        let num_quads = u.num_quads as usize;
        let num_stems = u.num_stems as usize;
        let total_q = num_quads * num_stems;
        if input.core_inputs.len() < total_q * num_frames * CORE_INPUT_VEC4_STRIDE
            || input.quad_params.len() < total_q * QUAD_PARAMS_VEC4_STRIDE
        {
            return Err(TrainError::InvalidDimensions {
                context: "run_fused_core_scan_planned",
                expected: total_q * num_frames * CORE_INPUT_VEC4_STRIDE,
                actual: input.core_inputs.len(),
            });
        }

        let mut tile_checkpoints = vec![[0.0_f32; 4]; total_q * num_tiles * TILE_CKPT_VEC4_STRIDE];
        let mut feature_outputs = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];
        let stem_group = plan.stem_group.max(1);
        let block_frames = plan.time_block_frames.max(TILE_FRAMES);

        for stem_start in (0..num_stems).step_by(stem_group) {
            let cur_stems = (num_stems - stem_start).min(stem_group);
            let cur_q = cur_stems * num_quads;
            let q_offset = stem_start * num_quads;
            let sub_params = &input.quad_params
                [q_offset * QUAD_PARAMS_VEC4_STRIDE..(q_offset + cur_q) * QUAD_PARAMS_VEC4_STRIDE];
            let mut carry_ckpt = vec![[0.0_f32; 4]; cur_q * TILE_CKPT_VEC4_STRIDE];
            let mut frame_start = 0_usize;

            while frame_start < num_frames {
                let chunk_frames = (num_frames - frame_start).min(block_frames);
                let chunk_tiles = chunk_frames.div_ceil(TILE_FRAMES);
                let has_prev = frame_start > 0;
                let has_next = frame_start + chunk_frames < num_frames;
                let lead_pad_frames = if has_prev { TILE_FRAMES } else { 0 };
                let tail_pad_frames = usize::from(has_next);
                let local_frames = lead_pad_frames + chunk_frames + tail_pad_frames;
                let local_tiles = usize::from(has_prev) + chunk_tiles + usize::from(has_next);

                let mut local_in =
                    vec![[0.0_f32; 4]; cur_q * local_frames * CORE_INPUT_VEC4_STRIDE];
                for ql in 0..cur_q {
                    let qg = q_offset + ql;
                    let dst_base = (ql * local_frames + lead_pad_frames) * CORE_INPUT_VEC4_STRIDE;
                    let src_base = (qg * num_frames + frame_start) * CORE_INPUT_VEC4_STRIDE;
                    let count = chunk_frames * CORE_INPUT_VEC4_STRIDE;
                    local_in[dst_base..dst_base + count]
                        .copy_from_slice(&input.core_inputs[src_base..src_base + count]);
                }

                let mut seed_storage =
                    vec![[0.0_f32; 4]; cur_q * local_tiles * TILE_CKPT_VEC4_STRIDE];
                let init_ckpts = if has_prev {
                    for ql in 0..cur_q {
                        let dst = (ql * local_tiles + 1) * TILE_CKPT_VEC4_STRIDE;
                        let src = ql * TILE_CKPT_VEC4_STRIDE;
                        seed_storage[dst..dst + TILE_CKPT_VEC4_STRIDE]
                            .copy_from_slice(&carry_ckpt[src..src + TILE_CKPT_VEC4_STRIDE]);
                    }
                    Some(seed_storage.as_slice())
                } else {
                    None
                };

                let local_u = FusedScanUniforms {
                    num_frames: local_frames as u32,
                    num_tiles: local_tiles as u32,
                    num_quads: num_quads as u32,
                    num_stems: cur_stems as u32,
                    tile_start: u32::from(has_prev),
                    soft_rect_eps: u.soft_rect_eps,
                    tiny_crossing_eps: u.tiny_crossing_eps,
                    block_idx: u.block_idx,
                };
                let local_out = self.fused_core_scan_256(&FusedCoreScanInput {
                    uniforms: local_u,
                    core_inputs: &local_in,
                    quad_params: sub_params,
                    initial_checkpoints: init_ckpts,
                })?;

                let tile_base = frame_start / TILE_FRAMES;
                let local_tile_off = usize::from(has_prev);
                for ql in 0..cur_q {
                    let qg = q_offset + ql;
                    let f_dst = (qg * num_frames + frame_start) * FEATURE_VEC4_STRIDE;
                    let f_src = (ql * local_frames + lead_pad_frames) * FEATURE_VEC4_STRIDE;
                    let f_cnt = chunk_frames * FEATURE_VEC4_STRIDE;
                    feature_outputs[f_dst..f_dst + f_cnt]
                        .copy_from_slice(&local_out.feature_outputs[f_src..f_src + f_cnt]);

                    let c_dst = (qg * num_tiles + tile_base) * TILE_CKPT_VEC4_STRIDE;
                    let c_src = (ql * local_tiles + local_tile_off) * TILE_CKPT_VEC4_STRIDE;
                    let c_cnt = chunk_tiles * TILE_CKPT_VEC4_STRIDE;
                    tile_checkpoints[c_dst..c_dst + c_cnt]
                        .copy_from_slice(&local_out.tile_checkpoints[c_src..c_src + c_cnt]);

                    if has_next {
                        let exit_src = (ql * local_tiles + local_tile_off + chunk_tiles)
                            * TILE_CKPT_VEC4_STRIDE;
                        let carry_dst = ql * TILE_CKPT_VEC4_STRIDE;
                        carry_ckpt[carry_dst..carry_dst + TILE_CKPT_VEC4_STRIDE].copy_from_slice(
                            &local_out.tile_checkpoints[exit_src..exit_src + TILE_CKPT_VEC4_STRIDE],
                        );
                    }
                }
                frame_start += chunk_frames;
            }
        }
        Ok(FusedCoreScanOutput {
            tile_checkpoints,
            feature_outputs,
        })
    }

    /// Dispatches 3-pass `adjoint_scan_256` on GPU via the persistent arena.
    pub fn adjoint_scan_256(
        &self,
        input: &AdjointScanInput<'_>,
    ) -> Result<AdjointScanOutput, TrainError> {
        self.arena.adjoint_scan_256(input)
    }

    /// Dispatches `tile_mix_fwd_adj` on GPU via the persistent arena.
    pub fn tile_mix_fwd_adj(&self, input: &TileMixInput<'_>) -> Result<TileMixOutput, TrainError> {
        self.arena.tile_mix_fwd_adj(input)
    }

    /// Dispatches `gabor_bezout_grad` on GPU via the persistent arena.
    pub fn gabor_bezout_grad(
        &self,
        input: &GaborBezoutInput<'_>,
    ) -> Result<GaborBezoutOutput, TrainError> {
        self.arena.gabor_bezout_grad(input)
    }

    /// Dispatches raw pass-0 / pass-1 `ozaki_gram_rhs_i32` on GPU via the persistent arena.
    pub fn ozaki_gram_rhs_i32(
        &self,
        input: &OzakiGramInput<'_>,
    ) -> Result<OzakiGramOutput, TrainError> {
        self.arena.ozaki_gram_rhs_i32(input)
    }

    /// Reconstructs column-major packed `uplo='U'` `f64` Gram matrix and RHS vectors on host.
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

    /// Reconstructs `0-ULP` `f64` Gram matrix and RHS vectors from split-64 limbs on host.
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

    /// Executes full 2-pass batched split-64 `ozaki_gram_rhs_i32` reading back `<= 1.2 MiB`.
    pub fn ozaki_gram_rhs_split64(
        &self,
        features_col_major: &[f32],
        targets_col_major: &[f32],
        num_frames: usize,
        feature_dim: usize,
        target_channels: usize,
        col_scales: &[[f32; 2]],
    ) -> Result<OzakiSplit64Output, TrainError> {
        self.arena.ozaki_gram_rhs_split64(
            features_col_major,
            targets_col_major,
            num_frames,
            feature_dim,
            target_channels,
            col_scales,
        )
    }

    /// Dispatches multi-candidate `(1, num_candidates, 1)` `inner_opt_step` on GPU via the persistent arena.
    pub fn inner_opt_step(&self, input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
        self.arena.inner_opt_step(input)
    }

    /// Executes wave-batched forward cascade (`Waves 0..=3` + `tile_mix_fwd_adj(0)`) on GPU.
    pub fn run_wave_core_cascade(
        &self,
        input: &WaveCoreCascadeInput<'_>,
    ) -> Result<WaveCoreCascadeOutput, TrainError> {
        self.arena.run_wave_core_cascade(input)
    }

    /// Executes wave-batched reverse BPTT (`Waves 3..=0` + `tile_mix_fwd_adj(1,2,3)`) on GPU.
    pub fn run_wave_core_adjoint(
        &self,
        input: &WaveCoreAdjointInput<'_>,
    ) -> Result<WaveCoreAdjointOutput, TrainError> {
        self.arena.run_wave_core_adjoint(input)
    }
}
