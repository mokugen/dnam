//! Grow-only persistent `wgpu::Buffer` arena (`GpuTrainArena`) enforcing `<= 128 MiB` storage
//! binding caps and profile total-memory ceilings across wave-batched core cascade/adjoint,
//! 2-pass split-64 Ozaki normal equations, and all 7 SONA kernels
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod kernels;
mod wave;

use core::fmt;
use std::sync::Mutex;

use zerocopy::{Immutable, IntoBytes};

use super::SharedGpuRef;

use super::abi::{
    EXIT_ADJ_BOUNDARY_VEC4_STRIDE, OZAKI_SPLIT64_WORDS_PER_PAIR, OZAKI_WORDS_PER_PAIR,
};
use super::context::WgpuContext;
use super::dispatch::{create_uniform_buffer, submit_and_readback};
use crate::error::TrainError;
use crate::planner::MemoryBudgetProfile;
use crate::runner::types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, GaborBezoutInput,
    GaborBezoutOutput, InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput,
    OzakiSplit64Output, PrescreenInput, TileMixInput, TileMixOutput, WaveCoreAdjointInput,
    WaveCoreAdjointOutput, WaveCoreCascadeInput, WaveCoreCascadeOutput,
};

const fn align_slot_bytes(need_bytes: u64) -> u64 {
    let clamped = if need_bytes < 16 { 16 } else { need_bytes };
    (clamped + 15) & !15
}

#[derive(Debug)]
pub(super) struct ArenaSlot {
    pub label: &'static str,
    buf: Option<wgpu::Buffer>,
}

impl ArenaSlot {
    const fn new(label: &'static str) -> Self {
        Self { label, buf: None }
    }

    fn size(&self) -> u64 {
        self.buf.as_ref().map_or(0, wgpu::Buffer::size)
    }

    fn ensure_storage(
        &mut self,
        device: &wgpu::Device,
        need_bytes: u64,
        storage_cap_bytes: u64,
        total_budget_bytes: u64,
        allocated_bytes: &mut u64,
    ) -> Result<wgpu::Buffer, TrainError> {
        self.ensure_buffer_with_usage(
            device,
            need_bytes,
            storage_cap_bytes,
            total_budget_bytes,
            allocated_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    }

    fn ensure_readback(
        &mut self,
        device: &wgpu::Device,
        need_bytes: u64,
        storage_cap_bytes: u64,
        total_budget_bytes: u64,
        allocated_bytes: &mut u64,
    ) -> Result<wgpu::Buffer, TrainError> {
        self.ensure_buffer_with_usage(
            device,
            need_bytes,
            storage_cap_bytes,
            total_budget_bytes,
            allocated_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }

    fn ensure_buffer_with_usage(
        &mut self,
        device: &wgpu::Device,
        need_bytes: u64,
        binding_cap_bytes: u64,
        total_budget_bytes: u64,
        allocated_bytes: &mut u64,
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, TrainError> {
        let aligned = align_slot_bytes(need_bytes);
        if aligned > binding_cap_bytes {
            return Err(TrainError::BindingExceedsCap {
                buffer: self.label,
                bytes: aligned,
                cap_bytes: binding_cap_bytes,
            });
        }
        if let Some(ref existing) = self.buf
            && existing.size() >= aligned
        {
            return Ok(existing.clone());
        }
        let old_size = self.size();
        let next_total = allocated_bytes
            .saturating_sub(old_size)
            .saturating_add(aligned);
        if next_total > total_budget_bytes {
            return Err(TrainError::BindingExceedsCap {
                buffer: "gpu_train_arena_total",
                bytes: next_total,
                cap_bytes: total_budget_bytes,
            });
        }
        // Drop old buffer before allocating larger capacity to avoid transient double-counting in VRAM.
        self.buf = None;
        let created = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: aligned,
            usage,
            mapped_at_creation: false,
        });
        *allocated_bytes = next_total;
        self.buf = Some(created.clone());
        Ok(created)
    }
}

#[derive(Debug)]
pub(super) struct ArenaState {
    pub allocated_bytes: u64,
    pub total_budget_bytes: u64,
    pub core_inputs: ArenaSlot,
    pub quad_params: ArenaSlot,
    pub tile_checkpoints: ArenaSlot,
    pub feature_outputs: ArenaSlot,
    pub d_loss_d_features: ArenaSlot,
    pub d_loss_d_inputs: ArenaSlot,
    pub param_grads: ArenaSlot,
    pub tile_adj_boundaries: ArenaSlot,
    pub exit_adj_wave: [ArenaSlot; 4],
    pub tile_descriptors: ArenaSlot,
    pub tile_weight_grads: ArenaSlot,
    pub ozaki_features: ArenaSlot,
    pub ozaki_targets: ArenaSlot,
    pub ozaki_scales: ArenaSlot,
    pub ozaki_buckets: ArenaSlot,
    pub opt_theta: ArenaSlot,
    pub opt_grad: ArenaSlot,
    pub opt_bounds: ArenaSlot,
    pub opt_state: ArenaSlot,
    pub opt_dir: ArenaSlot,
    pub opt_diag_precond: ArenaSlot,
    pub gabor_pred: ArenaSlot,
    pub gabor_target: ArenaSlot,
    pub gabor_taps: ArenaSlot,
    pub gabor_d_pred: ArenaSlot,
    pub gabor_losses: ArenaSlot,
    pub prescreen_in: ArenaSlot,
    pub prescreen_tgt: ArenaSlot,
    pub prescreen_seeds: ArenaSlot,
    pub prescreen_losses: ArenaSlot,
    pub dummy_tiny: ArenaSlot,
    pub readbacks: [ArenaSlot; 4],
}

impl ArenaState {
    fn new(total_budget_bytes: u64) -> Self {
        Self {
            allocated_bytes: 0,
            total_budget_bytes,
            core_inputs: ArenaSlot::new("arena_core_inputs"),
            quad_params: ArenaSlot::new("arena_quad_params"),
            tile_checkpoints: ArenaSlot::new("arena_tile_checkpoints"),
            feature_outputs: ArenaSlot::new("arena_feature_outputs"),
            d_loss_d_features: ArenaSlot::new("arena_d_loss_d_features"),
            d_loss_d_inputs: ArenaSlot::new("arena_d_loss_d_inputs"),
            param_grads: ArenaSlot::new("arena_param_grads"),
            tile_adj_boundaries: ArenaSlot::new("arena_tile_adj_boundaries"),
            exit_adj_wave: [
                ArenaSlot::new("arena_exit_adj_w0"),
                ArenaSlot::new("arena_exit_adj_w1"),
                ArenaSlot::new("arena_exit_adj_w2"),
                ArenaSlot::new("arena_exit_adj_w3"),
            ],
            tile_descriptors: ArenaSlot::new("arena_tile_descriptors"),
            tile_weight_grads: ArenaSlot::new("arena_tile_weight_grads"),
            ozaki_features: ArenaSlot::new("arena_ozaki_features"),
            ozaki_targets: ArenaSlot::new("arena_ozaki_targets"),
            ozaki_scales: ArenaSlot::new("arena_ozaki_scales"),
            ozaki_buckets: ArenaSlot::new("arena_ozaki_buckets"),
            opt_theta: ArenaSlot::new("arena_opt_theta"),
            opt_grad: ArenaSlot::new("arena_opt_grad"),
            opt_bounds: ArenaSlot::new("arena_opt_bounds"),
            opt_state: ArenaSlot::new("arena_opt_state"),
            opt_dir: ArenaSlot::new("arena_opt_dir"),
            opt_diag_precond: ArenaSlot::new("arena_opt_diag_precond"),
            gabor_pred: ArenaSlot::new("arena_gabor_pred"),
            gabor_target: ArenaSlot::new("arena_gabor_target"),
            gabor_taps: ArenaSlot::new("arena_gabor_taps"),
            gabor_d_pred: ArenaSlot::new("arena_gabor_d_pred"),
            gabor_losses: ArenaSlot::new("arena_gabor_losses"),
            prescreen_in: ArenaSlot::new("arena_prescreen_in"),
            prescreen_tgt: ArenaSlot::new("arena_prescreen_tgt"),
            prescreen_seeds: ArenaSlot::new("arena_prescreen_seeds"),
            prescreen_losses: ArenaSlot::new("arena_prescreen_losses"),
            dummy_tiny: ArenaSlot::new("arena_dummy_tiny"),
            readbacks: [
                ArenaSlot::new("arena_rb0"),
                ArenaSlot::new("arena_rb1"),
                ArenaSlot::new("arena_rb2"),
                ArenaSlot::new("arena_rb3"),
            ],
        }
    }

    pub(super) fn project_allocations(&self, updates: &[(&ArenaSlot, u64)]) -> (u64, u64) {
        let mut max_single = 0_u64;
        let mut total = self.allocated_bytes;
        for (slot, need_bytes) in updates {
            let aligned = align_slot_bytes(*need_bytes);
            max_single = max_single.max(aligned);
            let old = slot.size();
            if aligned > old {
                total = total.saturating_sub(old).saturating_add(aligned);
            }
        }
        (max_single, total)
    }
}

pub(super) fn create_and_record_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    label: &str,
    workgroups: (u32, u32, u32),
    uniform_bytes: &[u8],
    bindings: &[&wgpu::Buffer],
) {
    let u_buf = create_uniform_buffer(device, label, uniform_bytes);
    let mut entries = Vec::with_capacity(1 + bindings.len());
    entries.push(wgpu::BindGroupEntry {
        binding: 0,
        resource: u_buf.as_entire_binding(),
    });
    for (idx, buf) in bindings.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: (idx + 1) as u32,
            resource: buf.as_entire_binding(),
        });
    }
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &bgl,
        entries: &entries,
    });
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
}

pub(super) fn aligned_upload<T: IntoBytes + Immutable>(
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    slice: &[T],
) {
    let bytes = slice.as_bytes();
    if !bytes.is_empty() {
        queue.write_buffer(buf, 0, bytes);
    }
}

/// Shared `(cap_bytes, budget_bytes)` bounds passed to arena allocation helpers.
#[derive(Clone, Copy)]
pub(super) struct ArenaLimits {
    pub cap_bytes: u64,
    pub budget_bytes: u64,
}

pub(super) fn encode_readback_copy(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    rb_slot: &mut ArenaSlot,
    keep_bytes: usize,
    limits: ArenaLimits,
    allocated_bytes: &mut u64,
) -> Result<(wgpu::Buffer, usize), TrainError> {
    let copy_len = ((keep_bytes.max(4) + 3) & !3) as u64;
    let rb = rb_slot.ensure_readback(
        device,
        copy_len,
        limits.cap_bytes,
        limits.budget_bytes,
        allocated_bytes,
    )?;
    encoder.copy_buffer_to_buffer(src, 0, &rb, 0, copy_len);
    Ok((rb, keep_bytes))
}

pub(super) fn finish_readbacks(
    ctx: &WgpuContext,
    encoder: wgpu::CommandEncoder,
    staging: Vec<(wgpu::Buffer, usize)>,
    _guard: &std::sync::MutexGuard<'_, ArenaState>,
) -> Result<Vec<Vec<u8>>, TrainError> {
    submit_and_readback(ctx.device(), ctx.queue(), encoder, staging)
}

/// Persistent GPU compute buffer arena bound to one shared or dedicated [`WgpuContext`].
pub struct GpuTrainArena {
    ctx: SharedGpuRef<WgpuContext>,
    state: Mutex<ArenaState>,
}

impl fmt::Debug for GpuTrainArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuTrainArena")
            .field("allocated_bytes", &self.allocated_bytes())
            .field("total_budget_bytes", &self.total_budget_bytes())
            .finish_non_exhaustive()
    }
}

impl GpuTrainArena {
    /// Creates a [`GpuTrainArena`] with [`MemoryBudgetProfile::DesktopCapsule`] total budget (`512 MiB`).
    #[must_use]
    pub fn new(ctx: SharedGpuRef<WgpuContext>) -> Self {
        Self::with_budget_profile(ctx, MemoryBudgetProfile::DesktopCapsule)
    }

    /// Creates a [`GpuTrainArena`] enforcing the given [`MemoryBudgetProfile`] ceiling.
    #[must_use]
    pub fn with_budget_profile(
        ctx: SharedGpuRef<WgpuContext>,
        profile: MemoryBudgetProfile,
    ) -> Self {
        let (total_budget, _) = profile.limits();
        Self {
            ctx,
            state: Mutex::new(ArenaState::new(total_budget)),
        }
    }

    /// Borrows the underlying [`WgpuContext`].
    #[inline]
    #[must_use]
    pub const fn context(&self) -> &SharedGpuRef<WgpuContext> {
        &self.ctx
    }

    pub(super) fn lock_state(&self) -> std::sync::MutexGuard<'_, ArenaState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Returns total bytes currently retained in GPU storage + readback staging buffers.
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.lock_state().allocated_bytes
    }

    /// Returns the configured total arena budget ceiling in bytes.
    #[must_use]
    pub fn total_budget_bytes(&self) -> u64 {
        self.lock_state().total_budget_bytes
    }

    /// Executes `prescreen_population` on GPU using persistent arena buffers.
    pub fn prescreen_population(&self, input: &PrescreenInput<'_>) -> Result<Vec<f32>, TrainError> {
        kernels::prescreen_population(self, input)
    }

    /// Executes `fused_core_scan_256` on GPU using persistent arena buffers.
    pub fn fused_core_scan_256(
        &self,
        input: &FusedCoreScanInput<'_>,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        kernels::fused_core_scan_256(self, input)
    }

    /// Executes 3-pass `adjoint_scan_256` (with `Slot 7` wave boundary header) in 1 submit.
    pub fn adjoint_scan_256(
        &self,
        input: &AdjointScanInput<'_>,
    ) -> Result<AdjointScanOutput, TrainError> {
        kernels::adjoint_scan_256(self, input)
    }

    /// Executes direct `tile_mix_fwd_adj` (`dir_mode in 0..=3`) using persistent arena buffers.
    pub fn tile_mix_fwd_adj(&self, input: &TileMixInput<'_>) -> Result<TileMixOutput, TrainError> {
        kernels::tile_mix_fwd_adj(self, input)
    }

    /// Executes `gabor_bezout_grad` (`Mode A` or `Mode B`) using persistent arena buffers.
    pub fn gabor_bezout_grad(
        &self,
        input: &GaborBezoutInput<'_>,
    ) -> Result<GaborBezoutOutput, TrainError> {
        kernels::gabor_bezout_grad(self, input)
    }

    /// Executes raw pass-0 / pass-1 `ozaki_gram_rhs_i32` using persistent arena buffers.
    pub fn ozaki_gram_rhs_i32(
        &self,
        input: &OzakiGramInput<'_>,
    ) -> Result<OzakiGramOutput, TrainError> {
        kernels::ozaki_gram_rhs_i32(self, input)
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
        kernels::ozaki_gram_rhs_split64(
            self,
            features_col_major,
            targets_col_major,
            num_frames,
            feature_dim,
            target_channels,
            col_scales,
        )
    }

    /// Executes multi-candidate `(1, num_candidates, 1)` `inner_opt_step` using persistent arena buffers.
    pub fn inner_opt_step(&self, input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
        kernels::inner_opt_step(self, input)
    }

    /// Executes all 4 waves (`Wave 0..=3` + `tile_mix_fwd_adj(0)`) in 1 submit per `<= 16,384`-frame chunk.
    pub fn run_wave_core_cascade(
        &self,
        input: &WaveCoreCascadeInput<'_>,
    ) -> Result<WaveCoreCascadeOutput, TrainError> {
        wave::run_wave_core_cascade(self, input)
    }

    /// Executes reverse waves (`Wave 3..=0` + `tile_mix_fwd_adj(1,2,3)`) in 1 submit per chunk.
    pub fn run_wave_core_adjoint(
        &self,
        input: &WaveCoreAdjointInput<'_>,
    ) -> Result<WaveCoreAdjointOutput, TrainError> {
        wave::run_wave_core_adjoint(self, input)
    }
}

const _: () = assert!(EXIT_ADJ_BOUNDARY_VEC4_STRIDE == 24);
const _: () = assert!(OZAKI_WORDS_PER_PAIR == 6 && OZAKI_SPLIT64_WORDS_PER_PAIR == 4);

#[cfg(test)]
pub(super) mod tests {
    use dnam_kernels::{
        CoreQuadV4ParamsStereo4, GpuVec4, KernelVec4, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4,
        SonaToneFoldedBlock0, SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4,
        prewarp_tan_pi_f_over_fs_f32,
    };

    use super::super::abi::{CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE};
    #[cfg(not(target_arch = "wasm32"))]
    use super::super::ops::WgpuRunner;
    use super::*;
    use crate::cpu::CpuRunner;
    use crate::runner::types::pack_core_quad_params;

    pub(super) struct ConcurrentWorkerRef {
        pub(super) quad_params: Vec<[f32; 4]>,
        pub(super) base_core_inputs: Vec<[f32; 4]>,
        pub(super) d_loss_d_features: Vec<[f32; 4]>,
        pub(super) cpu_fwd: WaveCoreCascadeOutput,
        pub(super) cpu_adj: WaveCoreAdjointOutput,
    }

    pub(super) fn max_abs_diff_vec4(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .flat_map(|(x, y)| x.iter().zip(y.iter()).map(|(u, v)| (u - v).abs()))
            .fold(0.0_f32, f32::max)
    }

    pub(super) fn build_worker_ref(
        cpu: &CpuRunner,
        worker_idx: usize,
        num_frames: usize,
    ) -> ConcurrentWorkerRef {
        let w = worker_idx as f32;
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(0.03 + 0.01 * w),
            prewarp_tan_pi_f_over_fs_f32(0.08 + 0.015 * w),
            prewarp_tan_pi_f_over_fs_f32(0.15 + 0.01 * w),
            prewarp_tan_pi_f_over_fs_f32(0.24),
        ]);
        let k = GpuVec4::from_array([0.65 + 0.08 * w, 0.85, 1.15, 1.35]);
        let svf = TptSvfCoeffs4::from_gk(g, k);
        let tone_weights = SonaToneWeights4 {
            lp: GpuVec4::from_array([0.40 + 0.04 * w, 0.20, -0.15, 0.10]),
            bp: GpuVec4::from_array([0.25, 0.35 + 0.03 * w, 0.20, -0.15]),
            hp: GpuVec4::from_array([-0.10, 0.15, 0.30, 0.40]),
            ap: GpuVec4::from_array([0.08, -0.06, 0.12, 0.09]),
            dir: GpuVec4::from_array([0.20, 0.15, -0.08, 0.22]),
        };
        let params = CoreQuadV4ParamsStereo4 {
            svf,
            tone: SonaToneFoldedBlock0::from_weights(&tone_weights),
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(0.99),
                GpuVec4::from_array([0.30, 0.42, 0.55, 0.68]),
                GpuVec4::from_array([0.15, 0.28, 0.45, 0.62]),
                GpuVec4::from_array([0.10, 0.22, 0.38, 0.52]),
                GpuVec4::from_array([0.20, 0.35, 0.50, 0.65]),
                GpuVec4::from_array([0.05, 0.18, 0.32, 0.48]),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::from_array([0.04 * (w + 1.0), 0.02, 0.03, 0.01]),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            g0: GpuVec4::from_array([0.90, 1.05, 0.85, 1.15]),
            g_inf: GpuVec4::from_array([1.20, 0.85, 1.10, 0.95]),
            d_pre: GpuVec4::from_array([0.16, 0.28, 0.36, 0.20]),
            se_w_sq: GpuVec4::from_array([0.45, 0.25, 0.60, 0.15]),
            se_d_bus: GpuVec4::from_array([0.35, 0.20, 0.40, 0.15]),
            adaa_drive: GpuVec4::from_array([1.10 + 0.12 * w, 1.45, 1.85, 1.25]),
            lru_bias_drive: GpuVec4::from_array([0.12, -0.18, 0.24, -0.14]),
            lin_mix: GpuVec4::from_array([0.05, 0.12, 0.08, 0.15]),
            wavefolder_mix: GpuVec4::from_array([0.10, 0.30, 0.55, 0.80]),
            glu_mix: GpuVec4::from_array([0.55, -0.40, 0.70, -0.50]),
            out_gain: GpuVec4::from_array([1.0, 0.95, 1.05, 0.90]),
        };
        let quad_params = pack_core_quad_params(&params)[..QUAD_PARAMS_VEC4_STRIDE].to_vec();

        let mut base_core_inputs = vec![[0.0_f32; 4]; num_frames * CORE_INPUT_VEC4_STRIDE];
        let mut d_loss_d_features = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
        let freq_scale = 0.04 + 0.017 * (w + 1.0);
        for t in 0..num_frames {
            let tf = t as f32;
            let amp = 0.45 + 0.07 * w;
            let s0 = (tf * freq_scale).sin() * amp;
            let s1 = (tf * freq_scale * 1.37 + 0.4).cos() * amp;
            let base = t * CORE_INPUT_VEC4_STRIDE;
            base_core_inputs[base] = [s0, 0.8 * s1, -0.6 * s0, 0.5 * s1];
            base_core_inputs[base + 1] = [0.7 * s1, -0.9 * s0, 0.4 * s1, -0.3 * s0];
            for s in 0..FEATURE_VEC4_STRIDE {
                let g_val = ((tf * 0.031 + (s as f32) * 0.7 + w * 0.5).sin()) * (0.2 + 0.05 * w);
                d_loss_d_features[t * FEATURE_VEC4_STRIDE + s] =
                    [g_val, -0.5 * g_val, 0.75 * g_val, -0.25 * g_val];
            }
        }

        let fwd_in = WaveCoreCascadeInput {
            num_frames,
            num_quads: 1,
            num_stems: 1,
            active_tiles: 0,
            write_checkpoints: true,
            fixed_point_scan: false,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            base_core_inputs: &base_core_inputs,
            quad_params: &quad_params,
            tile_descriptors: &[],
        };
        let cpu_fwd = cpu.run_wave_core_cascade(&fwd_in).unwrap();
        let adj_in = WaveCoreAdjointInput {
            num_frames,
            num_quads: 1,
            num_stems: 1,
            active_tiles: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            core_inputs: &cpu_fwd.core_inputs,
            quad_params: &quad_params,
            tile_checkpoints: &cpu_fwd.tile_checkpoints,
            d_loss_d_features: &d_loss_d_features,
            feature_outputs: &cpu_fwd.feature_outputs,
            tile_descriptors: &[],
        };
        let cpu_adj = cpu.run_wave_core_adjoint(&adj_in).unwrap();
        ConcurrentWorkerRef {
            quad_params,
            base_core_inputs,
            d_loss_d_features,
            cpu_fwd,
            cpu_adj,
        }
    }

    #[test]
    fn test_arena_slot_cap_and_reuse_invariants() {
        let Ok(ctx) = WgpuContext::try_new() else {
            return;
        };
        let arena = GpuTrainArena::with_budget_profile(
            SharedGpuRef::new(ctx),
            MemoryBudgetProfile::StreamingHost {
                total_budget_bytes: 256,
                max_tensor_bytes: 128,
            },
        );
        assert_eq!(arena.total_budget_bytes(), 256);
        assert_eq!(arena.allocated_bytes(), 0);
        let mut st_guard = arena.state.lock().unwrap();
        let st = &mut *st_guard;
        let dev = arena.ctx.device();
        let err = st
            .core_inputs
            .ensure_storage(dev, 512, 128, 256, &mut st.allocated_bytes)
            .unwrap_err();
        assert!(matches!(err, TrainError::BindingExceedsCap { .. }));

        st.core_inputs
            .ensure_storage(dev, 64, 128, 256, &mut st.allocated_bytes)
            .unwrap();
        assert_eq!(st.allocated_bytes, 64);
        st.core_inputs
            .ensure_storage(dev, 48, 128, 256, &mut st.allocated_bytes)
            .unwrap();
        assert_eq!(st.allocated_bytes, 64);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_shared_arena_concurrent_dispatches_and_chunked_adjoint_match_cpu() {
        let Ok(shared_gpu) = WgpuRunner::try_shared() else {
            return;
        };
        let Ok(ctx) = WgpuContext::try_new() else {
            return;
        };
        const NUM_WORKERS: usize = 4;
        const NUM_ITERS: usize = 4;
        const NUM_FRAMES: usize = 512;

        let cpu = CpuRunner::detect();
        let refs: Vec<ConcurrentWorkerRef> = (0..NUM_WORKERS)
            .map(|w| build_worker_ref(&cpu, w, NUM_FRAMES))
            .collect();

        // Subtest A: Concurrent resident cascade + adjoint sharing global `WgpuRunner::try_shared()`.
        std::thread::scope(|scope| {
            for (worker_idx, wref) in refs.iter().enumerate() {
                let gpu = shared_gpu.clone();
                scope.spawn(move || {
                    for iter in 0..NUM_ITERS {
                        let fwd = gpu
                            .run_wave_core_cascade(&WaveCoreCascadeInput {
                                num_frames: NUM_FRAMES,
                                num_quads: 1,
                                num_stems: 1,
                                active_tiles: 0,
                                write_checkpoints: true,
                                fixed_point_scan: false,
                                soft_rect_eps: SOFT_RECT_EPS,
                                tiny_crossing_eps: TINY_CROSSING_EPS,
                                base_core_inputs: &wref.base_core_inputs,
                                quad_params: &wref.quad_params,
                                tile_descriptors: &[],
                            })
                            .unwrap();
                        let feat_diff =
                            max_abs_diff_vec4(&fwd.feature_outputs, &wref.cpu_fwd.feature_outputs);
                        assert!(
                            feat_diff < 1.0e-4,
                            "worker {worker_idx} iter {iter} cascade mismatch: {feat_diff}"
                        );

                        let adj = gpu
                            .run_wave_core_adjoint(&WaveCoreAdjointInput {
                                num_frames: NUM_FRAMES,
                                num_quads: 1,
                                num_stems: 1,
                                active_tiles: 0,
                                soft_rect_eps: SOFT_RECT_EPS,
                                tiny_crossing_eps: TINY_CROSSING_EPS,
                                core_inputs: &fwd.core_inputs,
                                quad_params: &wref.quad_params,
                                tile_checkpoints: &fwd.tile_checkpoints,
                                d_loss_d_features: &wref.d_loss_d_features,
                                feature_outputs: &fwd.feature_outputs,
                                tile_descriptors: &[],
                            })
                            .unwrap();
                        let du_diff =
                            max_abs_diff_vec4(&adj.d_loss_d_inputs, &wref.cpu_adj.d_loss_d_inputs);
                        let pg_diff =
                            max_abs_diff_vec4(&adj.param_grads, &wref.cpu_adj.param_grads);
                        assert!(
                            du_diff < 1.0e-4 && pg_diff < 1.0e-4,
                            "worker {worker_idx} iter {iter} shared adj mismatch: du={du_diff}, pg={pg_diff}"
                        );
                    }
                });
            }
        });

        // Subtest B: Tight 120 KiB budget forces 2x 256-frame reverse chunks so `st.exit_adj_wave[0]`
        // persists across reverse chunks under shared multi-threaded contention.
        let chunked_arena = SharedGpuRef::new(GpuTrainArena::with_budget_profile(
            SharedGpuRef::new(ctx),
            MemoryBudgetProfile::StreamingHost {
                total_budget_bytes: 120_000,
                max_tensor_bytes: 64_000,
            },
        ));
        std::thread::scope(|scope| {
            for (worker_idx, wref) in refs.iter().enumerate() {
                let arena = SharedGpuRef::clone(&chunked_arena);
                scope.spawn(move || {
                    for iter in 0..NUM_ITERS {
                        let adj = arena
                            .run_wave_core_adjoint(&WaveCoreAdjointInput {
                                num_frames: NUM_FRAMES,
                                num_quads: 1,
                                num_stems: 1,
                                active_tiles: 0,
                                soft_rect_eps: SOFT_RECT_EPS,
                                tiny_crossing_eps: TINY_CROSSING_EPS,
                                core_inputs: &wref.cpu_fwd.core_inputs,
                                quad_params: &wref.quad_params,
                                tile_checkpoints: &wref.cpu_fwd.tile_checkpoints,
                                d_loss_d_features: &wref.d_loss_d_features,
                                feature_outputs: &wref.cpu_fwd.feature_outputs,
                                tile_descriptors: &[],
                            })
                            .unwrap();
                        let du_diff =
                            max_abs_diff_vec4(&adj.d_loss_d_inputs, &wref.cpu_adj.d_loss_d_inputs);
                        let pg_diff =
                            max_abs_diff_vec4(&adj.param_grads, &wref.cpu_adj.param_grads);
                        assert!(
                            du_diff < 1.0e-4 && pg_diff < 1.0e-4,
                            "worker {worker_idx} iter {iter} chunked adj mismatch: du={du_diff}, pg={pg_diff}"
                        );
                    }
                });
            }
        });
        assert!(chunked_arena.allocated_bytes() > 0);
        assert!(chunked_arena.allocated_bytes() <= 120_000);
    }

    #[test]
    fn test_arena_project_allocations_tracks_alignment_and_growth() {
        let st = ArenaState::new(1024);
        assert_eq!(st.project_allocations(&[(&st.core_inputs, 1)]), (16, 16));
        assert_eq!(
            st.project_allocations(&[(&st.core_inputs, 17), (&st.quad_params, 20)]),
            (32, 64)
        );
    }
}
