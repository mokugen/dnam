//! Native `wgpu 30.0.2` implementations of all 6 SONA compute kernels plus 2-D planned
//! `ResidentBatched` / `Streamed` execution (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use core::mem::size_of;

use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS};
use zerocopy::IntoBytes;

use super::abi::{
    AdjointScanUniforms, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms,
    GPU_OZAKI_MAX_CHUNK_LEN, LBFGS_MAX_HISTORY_M, OZAKI_WORDS_PER_PAIR, QUAD_PARAMS_VEC4_STRIDE,
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES,
};
use super::context::WgpuContext;
use super::dispatch::{
    SinglePassDispatch, StorageBinding, as_bytes, create_storage_buffer, create_uniform_buffer,
    dispatch_single_pass, read_f32_vec, read_i32_vec, read_vec2_vec, read_vec4_vec,
    submit_and_readback,
};
use super::shaders::{KernelId, ShaderVariant};
use crate::cpu::ozaki_and_opt;
use crate::error::TrainError;
use crate::planner::{CoreScanPlan, ExecutionMode};
use crate::runner::types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, GaborBezoutInput,
    GaborBezoutOutput, InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput,
    PrescreenInput,
};
use crate::wide_gram::uplo_u_packed_len;

/// Native `wgpu` compute kernel runner wrapping a verified [`WgpuContext`].
#[derive(Debug)]
pub struct WgpuRunner {
    ctx: WgpuContext,
}

impl WgpuRunner {
    /// Initializes a [`WgpuRunner`] using automatic `Subgroups -> Portable` probe selection.
    pub fn try_new() -> Result<Self, TrainError> {
        Ok(Self {
            ctx: WgpuContext::try_new()?,
        })
    }

    /// Initializes a [`WgpuRunner`] pinned to an explicit [`ShaderVariant`].
    pub fn try_new_with_variant(variant: ShaderVariant) -> Result<Self, TrainError> {
        Ok(Self {
            ctx: WgpuContext::try_new_with_variant(variant)?,
        })
    }

    /// Borrows the underlying [`WgpuContext`].
    #[inline]
    #[must_use]
    pub const fn context(&self) -> &WgpuContext {
        &self.ctx
    }

    /// Dispatches `prescreen_population` on the GPU.
    pub fn prescreen_population(&self, input: &PrescreenInput<'_>) -> Result<Vec<f32>, TrainError> {
        let (num_seeds, num_frames, _num_quads, num_stems) = input.validate()?;
        let num_tiles = num_frames.div_ceil(TILE_FRAMES);
        let init_losses = vec![0.0_f32; num_stems * num_seeds * num_tiles];
        let pipeline = self.ctx.pipeline(KernelId::PrescreenPopulation);
        let out = dispatch_single_pass(
            self.ctx.device(),
            self.ctx.queue(),
            &pipeline,
            &SinglePassDispatch {
                label: "prescreen_population",
                workgroups: (num_seeds as u32, num_tiles as u32, num_stems as u32),
                uniform_bytes: input.uniforms.as_bytes(),
                bindings: &[
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.input_audio),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.target_audio),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.seed_params),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(&init_losses),
                    },
                ],
                max_binding_bytes: self.ctx.max_storage_binding_bytes(),
            },
        )?;
        read_f32_vec(&out[0])
    }

    /// Dispatches `fused_core_scan_256` on the GPU.
    pub fn fused_core_scan_256(
        &self,
        input: &FusedCoreScanInput<'_>,
    ) -> Result<FusedCoreScanOutput, TrainError> {
        let u = &input.uniforms;
        let num_frames = u.num_frames as usize;
        let num_tiles = (u.num_tiles as usize).max(1);
        let num_quads = u.num_quads as usize;
        let num_stems = u.num_stems as usize;
        let total_q = num_quads * num_stems;
        if num_frames == 0 || total_q == 0 {
            return Err(TrainError::InvalidDimensions {
                context: "fused_core_scan_256",
                expected: 1,
                actual: 0,
            });
        }
        let need_in = total_q * num_frames * CORE_INPUT_VEC4_STRIDE;
        if input.core_inputs.len() < need_in
            || input.quad_params.len() < total_q * QUAD_PARAMS_VEC4_STRIDE
        {
            return Err(TrainError::InvalidDimensions {
                context: "fused_core_scan_256(buffers)",
                expected: need_in,
                actual: input.core_inputs.len(),
            });
        }

        let mut ckpt_init = vec![[0.0_f32; 4]; total_q * num_tiles * TILE_CKPT_VEC4_STRIDE];
        if let Some(seed) = input.initial_checkpoints {
            let n = seed.len().min(ckpt_init.len());
            ckpt_init[..n].copy_from_slice(&seed[..n]);
        }
        let feat_init = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];
        let pipeline = self.ctx.pipeline(KernelId::FusedCoreScan256);
        let out = dispatch_single_pass(
            self.ctx.device(),
            self.ctx.queue(),
            &pipeline,
            &SinglePassDispatch {
                label: "fused_core_scan_256",
                workgroups: (num_quads as u32, 1, num_stems as u32),
                uniform_bytes: u.as_bytes(),
                bindings: &[
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(&input.core_inputs[..need_in]),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(&input.quad_params[..total_q * QUAD_PARAMS_VEC4_STRIDE]),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(&ckpt_init),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(&feat_init),
                    },
                ],
                max_binding_bytes: self.ctx.max_storage_binding_bytes(),
            },
        )?;
        Ok(FusedCoreScanOutput {
            tile_checkpoints: read_vec4_vec(&out[0])?,
            feature_outputs: read_vec4_vec(&out[1])?,
        })
    }

    /// Executes `fused_core_scan_256` under a 2-D [`CoreScanPlan`], chaining exit checkpoints via
    /// `+1` lead dummy tiles and `+1` trailing zero frame so [`ExecutionMode::Streamed`] matches
    /// [`ExecutionMode::ResidentBatched`] at `0-ULP` bit identity.
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
                let tail_pad_frames = if has_next { 1 } else { 0 };
                let local_frames = lead_pad_frames + chunk_frames + tail_pad_frames;
                let local_tiles =
                    (if has_prev { 1 } else { 0 }) + chunk_tiles + (if has_next { 1 } else { 0 });

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

    /// Dispatches the full 3-pass reverse BPTT pipeline (`pass_mode` in `{0, 1, 2}`) for
    /// `adjoint_scan_256` on the GPU without intermediate host roundtrips.
    pub fn adjoint_scan_256(
        &self,
        input: &AdjointScanInput<'_>,
    ) -> Result<AdjointScanOutput, TrainError> {
        let u = &input.uniforms;
        let num_frames = u.num_frames as usize;
        let num_tiles = (u.num_tiles as usize).max(1);
        let num_quads = u.num_quads as usize;
        let num_stems = u.num_stems as usize;
        let total_q = num_quads * num_stems;
        if num_frames == 0 || total_q == 0 {
            return Err(TrainError::InvalidDimensions {
                context: "adjoint_scan_256",
                expected: 1,
                actual: 0,
            });
        }

        let need_in = total_q * num_frames * CORE_INPUT_VEC4_STRIDE;
        let need_feat = total_q * num_frames * FEATURE_VEC4_STRIDE;
        let need_ckpt = total_q * num_tiles * TILE_CKPT_VEC4_STRIDE;
        if input.core_inputs.len() < need_in
            || input.d_loss_d_features.len() < need_feat
            || input.feature_outputs.len() < need_feat
            || input.tile_checkpoints.len() < need_ckpt
            || input.quad_params.len() < total_q * QUAD_PARAMS_VEC4_STRIDE
        {
            return Err(TrainError::InvalidDimensions {
                context: "adjoint_scan_256(buffers)",
                expected: need_feat,
                actual: input.feature_outputs.len(),
            });
        }

        let max_cap = self.ctx.max_storage_binding_bytes();
        let dev = self.ctx.device();
        let queue = self.ctx.queue();
        let pipe = self.ctx.pipeline(KernelId::AdjointScan256);

        let b_in = create_storage_buffer(
            dev,
            queue,
            "adj_in",
            as_bytes(&input.core_inputs[..need_in]),
            false,
            max_cap,
        )?;
        let b_params = create_storage_buffer(
            dev,
            queue,
            "adj_params",
            as_bytes(&input.quad_params[..total_q * QUAD_PARAMS_VEC4_STRIDE]),
            false,
            max_cap,
        )?;
        let b_ckpt = create_storage_buffer(
            dev,
            queue,
            "adj_ckpt",
            as_bytes(&input.tile_checkpoints[..need_ckpt]),
            false,
            max_cap,
        )?;
        let b_dl_f = create_storage_buffer(
            dev,
            queue,
            "adj_dl_f",
            as_bytes(&input.d_loss_d_features[..need_feat]),
            false,
            max_cap,
        )?;
        let du_init = vec![[0.0_f32; 4]; need_in];
        let pg_init = vec![[0.0_f32; 4]; total_q * num_tiles * QUAD_PARAMS_VEC4_STRIDE];
        let bnd_init = vec![[0.0_f32; 4]; total_q * num_tiles * TILE_ADJ_BOUNDARY_VEC4_STRIDE];
        let b_du = create_storage_buffer(dev, queue, "adj_du", as_bytes(&du_init), true, max_cap)?;
        let b_pg = create_storage_buffer(dev, queue, "adj_pg", as_bytes(&pg_init), true, max_cap)?;
        let b_bnd =
            create_storage_buffer(dev, queue, "adj_bnd", as_bytes(&bnd_init), true, max_cap)?;
        let b_feats = create_storage_buffer(
            dev,
            queue,
            "adj_feats",
            as_bytes(&input.feature_outputs[..need_feat]),
            false,
            max_cap,
        )?;

        let bgl = pipe.get_bind_group_layout(0);
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adjoint_3pass"),
        });
        let soft_eps = if u.soft_rect_eps > 0.0 {
            u.soft_rect_eps
        } else {
            SOFT_RECT_EPS
        };
        let tiny_eps = if u.tiny_crossing_eps > 0.0 {
            u.tiny_crossing_eps
        } else {
            TINY_CROSSING_EPS
        };

        for (pass_mode, wg_y) in [(0_u32, num_tiles as u32), (1, 1), (2, num_tiles as u32)] {
            let pass_u = AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: num_quads as u32,
                num_stems: num_stems as u32,
                pass_mode,
                soft_rect_eps: soft_eps,
                tiny_crossing_eps: tiny_eps,
                block_idx: u.block_idx,
            };
            let u_buf = create_uniform_buffer(dev, "adj_u", pass_u.as_bytes());
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("adj_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_ckpt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: b_dl_f.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: b_du.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: b_pg.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: b_bnd.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: b_feats.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adjoint_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipe);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(num_quads as u32, wg_y, num_stems as u32);
        }

        let du_bytes = need_in * size_of::<[f32; 4]>();
        let pg_bytes = pg_init.len() * size_of::<[f32; 4]>();
        let rb_du = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb_du"),
            size: b_du.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rb_pg = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb_pg"),
            size: b_pg.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&b_du, 0, &rb_du, 0, b_du.size());
        encoder.copy_buffer_to_buffer(&b_pg, 0, &rb_pg, 0, b_pg.size());

        let mapped = submit_and_readback(
            dev,
            queue,
            encoder,
            vec![(rb_du, du_bytes), (rb_pg, pg_bytes)],
        )?;
        Ok(AdjointScanOutput {
            d_loss_d_inputs: read_vec4_vec(&mapped[0])?,
            param_grads: read_vec4_vec(&mapped[1])?,
        })
    }

    /// Dispatches `gabor_bezout_grad` (`Mode A` causal FIR or `Mode B` disjoint Gabor FFT) on GPU.
    pub fn gabor_bezout_grad(
        &self,
        input: &GaborBezoutInput<'_>,
    ) -> Result<GaborBezoutOutput, TrainError> {
        let u = &input.uniforms;
        let num_frames = u.num_frames as usize;
        let num_stems = u.num_stems as usize;
        if num_frames == 0 || num_stems == 0 {
            return Err(TrainError::InvalidDimensions {
                context: "gabor_bezout_grad",
                expected: 1,
                actual: 0,
            });
        }
        let num_windows = if u.fft_size == 0 {
            num_frames.div_ceil(TILE_FRAMES)
        } else {
            let n = u.fft_size as usize;
            let hop = u.hop_size as usize;
            if hop < n {
                return Err(TrainError::KernelMismatch {
                    context: "gabor_bezout_grad",
                    detail: "Mode B requires hop_size >= fft_size for disjoint writebacks",
                });
            }
            num_frames.div_ceil(hop)
        };
        let d_init = vec![[0.0_f32; 2]; num_stems * num_frames];
        let bl_init = vec![[0.0_f32; 4]; num_stems * num_windows];
        let pipe = self.ctx.pipeline(KernelId::GaborBezoutGrad);
        let out = dispatch_single_pass(
            self.ctx.device(),
            self.ctx.queue(),
            &pipe,
            &SinglePassDispatch {
                label: "gabor_bezout_grad",
                workgroups: (num_windows as u32, num_stems as u32, 1),
                uniform_bytes: u.as_bytes(),
                bindings: &[
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.pred_audio),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.target_audio),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.bezout_taps),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(&d_init),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(&bl_init),
                    },
                ],
                max_binding_bytes: self.ctx.max_storage_binding_bytes(),
            },
        )?;
        Ok(GaborBezoutOutput {
            d_loss_d_pred: read_vec2_vec(&out[0])?,
            band_losses: read_vec4_vec(&out[1])?,
        })
    }

    /// Dispatches exact `0-ULP` integer-sliced `ozaki_gram_rhs_i32` on the GPU.
    pub fn ozaki_gram_rhs_i32(
        &self,
        input: &OzakiGramInput<'_>,
    ) -> Result<OzakiGramOutput, TrainError> {
        let u = &input.uniforms;
        let num_frames = u.num_frames as usize;
        let dim = u.feature_dim as usize;
        let tc = u.target_channels as usize;
        let chunk_len = (u.chunk_len as usize).clamp(1, GPU_OZAKI_MAX_CHUNK_LEN);
        let num_chunks = (u.num_chunks as usize).max(num_frames.div_ceil(chunk_len));
        if num_frames == 0 || dim == 0 {
            return Err(TrainError::InvalidDimensions {
                context: "ozaki_gram_rhs_i32",
                expected: 1,
                actual: 0,
            });
        }
        let gram_pairs = uplo_u_packed_len(dim);
        let min_words = (gram_pairs + dim * tc) * num_chunks * OZAKI_WORDS_PER_PAIR;
        let buckets_init = match input.seed_buckets {
            Some(seed) => {
                if seed.len() < min_words {
                    return Err(TrainError::InvalidDimensions {
                        context: "ozaki_gram_rhs_i32(seed_buckets)",
                        expected: min_words,
                        actual: seed.len(),
                    });
                }
                seed.to_vec()
            }
            None => vec![0_i32; min_words],
        };
        let num_pairs = buckets_init.len() / (num_chunks * OZAKI_WORDS_PER_PAIR);
        let pipe = self.ctx.pipeline(KernelId::OzakiGramRhsI32);
        let out = dispatch_single_pass(
            self.ctx.device(),
            self.ctx.queue(),
            &pipe,
            &SinglePassDispatch {
                label: "ozaki_gram_rhs_i32",
                workgroups: (num_pairs as u32, num_chunks as u32, 1),
                uniform_bytes: u.as_bytes(),
                bindings: &[
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.features),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.targets),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.col_scales),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(&buckets_init),
                    },
                ],
                max_binding_bytes: self.ctx.max_storage_binding_bytes(),
            },
        )?;
        Ok(OzakiGramOutput {
            ozaki_buckets: read_i32_vec(&out[0])?,
            num_chunks,
            num_pairs,
        })
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

    /// Dispatches single-workgroup `(1, 1, 1)` bounded `m <= 8` L-BFGS `inner_opt_step` on GPU.
    pub fn inner_opt_step(&self, input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
        let m = input.uniforms.history_m as usize;
        if input.uniforms.num_params == 0 || m > LBFGS_MAX_HISTORY_M {
            return Err(TrainError::InvalidDimensions {
                context: "inner_opt_step",
                expected: LBFGS_MAX_HISTORY_M,
                actual: m,
            });
        }
        let pipe = self.ctx.pipeline(KernelId::InnerOptStep);
        let out = dispatch_single_pass(
            self.ctx.device(),
            self.ctx.queue(),
            &pipe,
            &SinglePassDispatch {
                label: "inner_opt_step",
                workgroups: (1, 1, 1),
                uniform_bytes: input.uniforms.as_bytes(),
                bindings: &[
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(input.theta),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.grad),
                    },
                    StorageBinding {
                        writable: false,
                        bytes: as_bytes(input.param_bounds),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(input.lbfgs_s_y),
                    },
                    StorageBinding {
                        writable: true,
                        bytes: as_bytes(input.search_dir),
                    },
                ],
                max_binding_bytes: self.ctx.max_storage_binding_bytes(),
            },
        )?;
        Ok(InnerOptOutput {
            theta: read_f32_vec(&out[0])?,
            lbfgs_s_y: read_vec4_vec(&out[1])?,
            search_dir: read_f32_vec(&out[2])?,
        })
    }
}
