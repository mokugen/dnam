//! Wave-batched forward cascade (`Waves 0..=3` + `tile_mix_fwd_adj(0)`) and reverse BPTT adjoint
//! (`Waves 3..=0` + `tile_mix_fwd_adj(1,2,3)`) executing inside persistent [`GpuTrainArena`] with
//! automatic `16,384`-frame `2×` time-block streaming (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

mod adjoint;

pub(super) use adjoint::run_wave_core_adjoint;

use core::mem::size_of_val;

use zerocopy::IntoBytes;

use super::super::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FUSED_FLAG_FIXED_POINT_SCAN, FusedScanUniforms,
    QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_DESC_VEC4_STRIDE, TILE_FRAMES,
    TILE_MIX_DIR_FORWARD, TileMixUniforms, block_quad_count,
};
use super::super::dispatch::read_vec4_vec;
use super::super::shaders::KernelId;
use super::{
    ArenaLimits, GpuTrainArena, aligned_upload, create_and_record_pass, encode_readback_copy,
    finish_readbacks,
};
use crate::error::TrainError;
use crate::planner::WAVE_STREAMING_MAX_FRAMES;
use crate::runner::types::{WaveCoreCascadeInput, WaveCoreCascadeOutput};

#[derive(Clone, Copy)]
struct StemFrameSpan {
    stem_start: usize,
    cur_stems: usize,
    num_quads: usize,
    total_frames: usize,
    frame_start: usize,
    chunk_frames: usize,
    stride: usize,
    lead_pad: usize,
}

fn slice_stem_frame_chunk<T: Copy>(
    src: &[T],
    span: StemFrameSpan,
    tail_pad: usize,
    zero: T,
) -> Vec<T> {
    let cur_q = span.cur_stems * span.num_quads;
    let local_frames = span.lead_pad + span.chunk_frames + tail_pad;
    let mut dst = vec![zero; cur_q * local_frames * span.stride];
    let q_offset = span.stem_start * span.num_quads;
    for ql in 0..cur_q {
        let qg = q_offset + ql;
        let src_off = (qg * span.total_frames + span.frame_start) * span.stride;
        let dst_off = (ql * local_frames + span.lead_pad) * span.stride;
        let count = span.chunk_frames * span.stride;
        dst[dst_off..dst_off + count].copy_from_slice(&src[src_off..src_off + count]);
    }
    dst
}

fn store_stem_frame_chunk<T: Copy>(
    dst: &mut [T],
    src: &[T],
    span: StemFrameSpan,
    local_frames: usize,
) {
    let cur_q = span.cur_stems * span.num_quads;
    let q_offset = span.stem_start * span.num_quads;
    for ql in 0..cur_q {
        let qg = q_offset + ql;
        let src_off = (ql * local_frames + span.lead_pad) * span.stride;
        let dst_off = (qg * span.total_frames + span.frame_start) * span.stride;
        let count = span.chunk_frames * span.stride;
        dst[dst_off..dst_off + count].copy_from_slice(&src[src_off..src_off + count]);
    }
}

#[derive(Clone, Copy)]
pub(super) struct CascadeChunkSpec {
    pub num_stems: usize,
    pub num_quads: usize,
    pub chunk_frames: usize,
    pub active_tiles: usize,
    pub write_checkpoints: bool,
    pub has_prev: bool,
    pub has_next: bool,
}

pub(super) fn cascade_chunk_fits(
    st: &super::ArenaState,
    spec: CascadeChunkSpec,
    limits: ArenaLimits,
) -> bool {
    let cur_q = (spec.num_stems * spec.num_quads) as u64;
    let lead_pad = usize::from(spec.has_prev) * TILE_FRAMES;
    let tail_pad = usize::from(spec.has_next);
    let local_frames = (lead_pad + spec.chunk_frames + tail_pad) as u64;
    let local_tiles =
        (usize::from(spec.has_prev) + spec.chunk_frames.div_ceil(TILE_FRAMES) + tail_pad) as u64;
    let in_bytes = cur_q * local_frames * (CORE_INPUT_VEC4_STRIDE as u64) * 16;
    let params_bytes = cur_q * (QUAD_PARAMS_VEC4_STRIDE as u64) * 16;
    let ckpt_bytes = cur_q * local_tiles * (TILE_CKPT_VEC4_STRIDE as u64) * 16;
    let feat_bytes = cur_q * local_frames * (FEATURE_VEC4_STRIDE as u64) * 16;
    let need_ckpt_rb = spec.write_checkpoints || spec.has_next;

    let mut updates = [
        (&st.core_inputs, in_bytes),
        (&st.quad_params, params_bytes),
        (&st.tile_checkpoints, ckpt_bytes),
        (&st.feature_outputs, feat_bytes),
        (&st.dummy_tiny, 16),
        (&st.readbacks[0], in_bytes),
        (&st.readbacks[1], feat_bytes),
        (&st.tile_descriptors, 0),
        (&st.readbacks[2], 0),
    ];
    let mut len = 7_usize;
    if spec.active_tiles > 0 {
        updates[len] = (
            &st.tile_descriptors,
            (spec.active_tiles * TILE_DESC_VEC4_STRIDE * 16) as u64,
        );
        len += 1;
    }
    if need_ckpt_rb {
        updates[len] = (&st.readbacks[2], ckpt_bytes);
        len += 1;
    }
    let (max_single, total) = st.project_allocations(&updates[..len]);
    max_single <= limits.cap_bytes && total <= limits.budget_bytes
}

fn choose_stream_frames<F: Fn(usize) -> bool>(max_req: usize, fits: F) -> usize {
    let mut frames = WAVE_STREAMING_MAX_FRAMES.min(max_req);
    frames = frames.div_ceil(TILE_FRAMES) * TILE_FRAMES;
    while frames > TILE_FRAMES && !fits(frames) {
        frames = (frames / 2 / TILE_FRAMES).max(1) * TILE_FRAMES;
    }
    frames.max(TILE_FRAMES)
}

fn validate_descriptors(
    descriptors: &[[f32; 4]],
    active_tiles: usize,
    context: &'static str,
) -> Result<(), TrainError> {
    let need = active_tiles * TILE_DESC_VEC4_STRIDE;
    if descriptors.len() < need {
        return Err(TrainError::InvalidDimensions {
            context,
            expected: need,
            actual: descriptors.len(),
        });
    }
    Ok(())
}

/// Executes all 4 waves (`Waves 0..=3` + `tile_mix_fwd_adj(0)`) inside [`GpuTrainArena`].
pub(super) fn run_wave_core_cascade(
    arena: &GpuTrainArena,
    input: &WaveCoreCascadeInput<'_>,
) -> Result<WaveCoreCascadeOutput, TrainError> {
    let num_frames = input.num_frames;
    let num_quads = input.num_quads;
    let num_stems = input.num_stems;
    let active_tiles = input.active_tiles;
    if num_frames == 0 || num_stems == 0 || !(1..=10).contains(&num_quads) {
        return Err(TrainError::InvalidDimensions {
            context: "run_wave_core_cascade",
            expected: 1,
            actual: 0,
        });
    }
    let total_q = num_stems * num_quads;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    if input.base_core_inputs.len() < total_q * num_frames * CORE_INPUT_VEC4_STRIDE
        || input.quad_params.len() < total_q * QUAD_PARAMS_VEC4_STRIDE
    {
        return Err(TrainError::InvalidDimensions {
            context: "run_wave_core_cascade(buffers)",
            expected: total_q * num_frames * CORE_INPUT_VEC4_STRIDE,
            actual: input.base_core_inputs.len(),
        });
    }
    validate_descriptors(
        input.tile_descriptors,
        active_tiles,
        "run_wave_core_cascade(tile_descriptors)",
    )?;

    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let budget = arena.total_budget_bytes();
    let limits = ArenaLimits {
        cap_bytes: cap,
        budget_bytes: budget,
    };
    let pipe_scan = ctx.pipeline(KernelId::FusedCoreScan256);
    let pipe_mix = ctx.pipeline(KernelId::TileMixFwdAdj);
    let dev = ctx.device();
    let queue = ctx.queue();
    let mut st_guard = arena.lock_state();
    let resident = num_frames <= WAVE_STREAMING_MAX_FRAMES
        && cascade_chunk_fits(
            &st_guard,
            CascadeChunkSpec {
                num_stems,
                num_quads,
                chunk_frames: num_frames,
                active_tiles,
                write_checkpoints: input.write_checkpoints,
                has_prev: false,
                has_next: false,
            },
            limits,
        );
    let (stem_group, block_frames) = if resident {
        (num_stems, num_frames)
    } else {
        let bf = choose_stream_frames(num_frames, |f| {
            let cf = f.min(num_frames);
            let multi = num_frames > cf;
            cascade_chunk_fits(
                &st_guard,
                CascadeChunkSpec {
                    num_stems: 1,
                    num_quads,
                    chunk_frames: cf,
                    active_tiles,
                    write_checkpoints: input.write_checkpoints,
                    has_prev: multi,
                    has_next: multi,
                },
                limits,
            )
        });
        (1, bf)
    };

    let mut out_in = vec![[0.0_f32; 4]; total_q * num_frames * CORE_INPUT_VEC4_STRIDE];
    let mut out_feat = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];
    let mut out_ckpt = if input.write_checkpoints {
        vec![[0.0_f32; 4]; total_q * num_tiles * TILE_CKPT_VEC4_STRIDE]
    } else {
        Vec::new()
    };

    let fp_flag = if input.fixed_point_scan {
        FUSED_FLAG_FIXED_POINT_SCAN
    } else {
        0
    };

    for stem_start in (0..num_stems).step_by(stem_group) {
        let cur_stems = (num_stems - stem_start).min(stem_group);
        let cur_q = cur_stems * num_quads;
        let p_off = stem_start * num_quads * QUAD_PARAMS_VEC4_STRIDE;
        let sub_params = &input.quad_params[p_off..p_off + cur_q * QUAD_PARAMS_VEC4_STRIDE];
        let mut carry_ckpt = vec![[0.0_f32; 4]; cur_q * TILE_CKPT_VEC4_STRIDE];
        let mut frame_start = 0_usize;

        while frame_start < num_frames {
            let chunk_frames = (num_frames - frame_start).min(block_frames);
            let chunk_tiles = chunk_frames.div_ceil(TILE_FRAMES);
            let has_prev = frame_start > 0;
            let has_next = frame_start + chunk_frames < num_frames;
            let lead_pad = if has_prev { TILE_FRAMES } else { 0 };
            let tail_pad = usize::from(has_next);
            let local_frames = lead_pad + chunk_frames + tail_pad;
            let local_tiles = usize::from(has_prev) + chunk_tiles + tail_pad;

            let span_in = StemFrameSpan {
                stem_start,
                cur_stems,
                num_quads,
                total_frames: num_frames,
                frame_start,
                chunk_frames,
                stride: CORE_INPUT_VEC4_STRIDE,
                lead_pad,
            };
            let local_in =
                slice_stem_frame_chunk(input.base_core_inputs, span_in, tail_pad, [0.0_f32; 4]);
            let seed_ckpts = if has_prev {
                let mut seed = vec![[0.0_f32; 4]; cur_q * local_tiles * TILE_CKPT_VEC4_STRIDE];
                for ql in 0..cur_q {
                    let dst = (ql * local_tiles + 1) * TILE_CKPT_VEC4_STRIDE;
                    let src = ql * TILE_CKPT_VEC4_STRIDE;
                    seed[dst..dst + TILE_CKPT_VEC4_STRIDE]
                        .copy_from_slice(&carry_ckpt[src..src + TILE_CKPT_VEC4_STRIDE]);
                }
                Some(seed)
            } else {
                None
            };

            let in_bytes = size_of_val(local_in.as_slice());
            let feat_bytes = cur_q * local_frames * FEATURE_VEC4_STRIDE * 16;
            let ckpt_bytes = cur_q * local_tiles * TILE_CKPT_VEC4_STRIDE * 16;
            let need_ckpt_rb = input.write_checkpoints || has_next;

            let (encoder, staging) = {
                let st = &mut *st_guard;
                let alloc = &mut st.allocated_bytes;
                let b_in =
                    st.core_inputs
                        .ensure_storage(dev, in_bytes as u64, cap, budget, alloc)?;
                let b_params = st.quad_params.ensure_storage(
                    dev,
                    size_of_val(sub_params) as u64,
                    cap,
                    budget,
                    alloc,
                )?;
                let b_ckpt = st.tile_checkpoints.ensure_storage(
                    dev,
                    ckpt_bytes as u64,
                    cap,
                    budget,
                    alloc,
                )?;
                let b_feat = st.feature_outputs.ensure_storage(
                    dev,
                    feat_bytes as u64,
                    cap,
                    budget,
                    alloc,
                )?;
                let b_dummy = st.dummy_tiny.ensure_storage(dev, 16, cap, budget, alloc)?;
                let b_desc = if active_tiles == 0 {
                    b_dummy.clone()
                } else {
                    let s = &input.tile_descriptors[..active_tiles * TILE_DESC_VEC4_STRIDE];
                    let b = st.tile_descriptors.ensure_storage(
                        dev,
                        size_of_val(s) as u64,
                        cap,
                        budget,
                        alloc,
                    )?;
                    aligned_upload(queue, &b, s);
                    b
                };

                aligned_upload(queue, &b_in, &local_in);
                aligned_upload(queue, &b_params, sub_params);
                if let Some(ref seed) = seed_ckpts {
                    aligned_upload(queue, &b_ckpt, seed);
                }

                let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("arena_wave_cascade"),
                });
                if seed_ckpts.is_none() {
                    encoder.clear_buffer(&b_ckpt, 0, Some(ckpt_bytes as u64));
                }
                encoder.clear_buffer(&b_feat, 0, Some(feat_bytes as u64));

                for block_idx in 0..4_u32 {
                    let wg_x = block_quad_count(block_idx as usize, num_quads);
                    if wg_x == 0 {
                        continue;
                    }
                    if block_idx > 0 && active_tiles > 0 {
                        let mix_u = TileMixUniforms {
                            num_frames: local_frames as u32,
                            num_tiles: local_tiles as u32,
                            num_active_tiles: active_tiles as u32,
                            dir_mode: TILE_MIX_DIR_FORWARD,
                            num_stems: cur_stems as u32,
                            block_wave: block_idx,
                            num_quads: num_quads as u32,
                            _pad1: 0,
                        };
                        create_and_record_pass(
                            dev,
                            &mut encoder,
                            &pipe_mix,
                            "wave_mix_fwd",
                            (wg_x as u32, local_tiles as u32, cur_stems as u32),
                            mix_u.as_bytes(),
                            &[&b_in, &b_desc, &b_feat, &b_dummy, &b_dummy],
                        );
                    }
                    let scan_u = FusedScanUniforms {
                        num_frames: local_frames as u32,
                        num_tiles: local_tiles as u32,
                        num_quads: num_quads as u32,
                        num_stems: cur_stems as u32,
                        tile_start: u32::from(has_prev),
                        soft_rect_eps: input.soft_rect_eps,
                        tiny_crossing_eps: input.tiny_crossing_eps,
                        block_idx: block_idx | fp_flag,
                    };
                    create_and_record_pass(
                        dev,
                        &mut encoder,
                        &pipe_scan,
                        "wave_fused_scan",
                        (wg_x as u32, 1, cur_stems as u32),
                        scan_u.as_bytes(),
                        &[&b_in, &b_params, &b_ckpt, &b_feat],
                    );
                }

                let limits = ArenaLimits {
                    cap_bytes: cap,
                    budget_bytes: budget,
                };
                let rb_in = encode_readback_copy(
                    dev,
                    &mut encoder,
                    &b_in,
                    &mut st.readbacks[0],
                    in_bytes,
                    limits,
                    &mut st.allocated_bytes,
                )?;
                let rb_feat = encode_readback_copy(
                    dev,
                    &mut encoder,
                    &b_feat,
                    &mut st.readbacks[1],
                    feat_bytes,
                    limits,
                    &mut st.allocated_bytes,
                )?;
                let mut rbs = vec![rb_in, rb_feat];
                if need_ckpt_rb {
                    let rb_ckpt = encode_readback_copy(
                        dev,
                        &mut encoder,
                        &b_ckpt,
                        &mut st.readbacks[2],
                        ckpt_bytes,
                        limits,
                        &mut st.allocated_bytes,
                    )?;
                    rbs.push(rb_ckpt);
                }
                (encoder, rbs)
            };

            let read = finish_readbacks(ctx, encoder, staging, &st_guard)?;
            let chunk_in = read_vec4_vec(&read[0])?;
            let chunk_feat = read_vec4_vec(&read[1])?;
            store_stem_frame_chunk(&mut out_in, &chunk_in, span_in, local_frames);
            store_stem_frame_chunk(
                &mut out_feat,
                &chunk_feat,
                StemFrameSpan {
                    stride: FEATURE_VEC4_STRIDE,
                    ..span_in
                },
                local_frames,
            );

            if need_ckpt_rb {
                let chunk_ckpts = read_vec4_vec(&read[2])?;
                let local_tile_off = usize::from(has_prev);
                let tile_base = frame_start / TILE_FRAMES;
                let q_off = stem_start * num_quads;
                for ql in 0..cur_q {
                    if input.write_checkpoints {
                        let qg = q_off + ql;
                        let dst = (qg * num_tiles + tile_base) * TILE_CKPT_VEC4_STRIDE;
                        let src = (ql * local_tiles + local_tile_off) * TILE_CKPT_VEC4_STRIDE;
                        let cnt = chunk_tiles * TILE_CKPT_VEC4_STRIDE;
                        out_ckpt[dst..dst + cnt].copy_from_slice(&chunk_ckpts[src..src + cnt]);
                    }
                    if has_next {
                        let exit_src = (ql * local_tiles + local_tile_off + chunk_tiles)
                            * TILE_CKPT_VEC4_STRIDE;
                        let carry_dst = ql * TILE_CKPT_VEC4_STRIDE;
                        carry_ckpt[carry_dst..carry_dst + TILE_CKPT_VEC4_STRIDE].copy_from_slice(
                            &chunk_ckpts[exit_src..exit_src + TILE_CKPT_VEC4_STRIDE],
                        );
                    }
                }
            }
            frame_start += chunk_frames;
        }
    }
    drop(st_guard);

    Ok(WaveCoreCascadeOutput {
        core_inputs: out_in,
        tile_checkpoints: out_ckpt,
        feature_outputs: out_feat,
    })
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{SOFT_RECT_EPS, TINY_CROSSING_EPS};

    use super::super::super::abi::{EXIT_ADJ_BOUNDARY_VEC4_STRIDE, TILE_ADJ_BOUNDARY_VEC4_STRIDE};
    use super::super::super::context::WgpuContext;
    use super::super::tests::{build_worker_ref, max_abs_diff_vec4};
    use super::super::{ArenaState, SharedGpuRef};
    use super::adjoint::adjoint_chunk_fits;
    use super::*;
    use crate::cpu::CpuRunner;
    use crate::planner::MemoryBudgetProfile;
    use crate::runner::types::WaveCoreAdjointInput;

    #[test]
    fn test_wave_budget_projection_accounts_for_all_slots_and_non_wave_occupancy() {
        let (browser_total, browser_cap) = MemoryBudgetProfile::BrowserWebGpu.limits();
        let st_browser = ArenaState::new(browser_total);
        let browser_limits = ArenaLimits {
            cap_bytes: browser_cap,
            budget_bytes: browser_total,
        };
        assert!(!cascade_chunk_fits(
            &st_browser,
            CascadeChunkSpec {
                num_stems: 16,
                num_quads: 10,
                chunk_frames: 7_000,
                active_tiles: 35,
                write_checkpoints: true,
                has_prev: false,
                has_next: false,
            },
            browser_limits,
        ));
        assert!(cascade_chunk_fits(
            &st_browser,
            CascadeChunkSpec {
                num_stems: 1,
                num_quads: 10,
                chunk_frames: 7_000,
                active_tiles: 35,
                write_checkpoints: true,
                has_prev: false,
                has_next: false,
            },
            browser_limits,
        ));
        assert!(!adjoint_chunk_fits(
            &st_browser,
            16,
            10,
            7_000,
            35,
            browser_cap,
            browser_total,
        ));
        assert!(adjoint_chunk_fits(
            &st_browser,
            1,
            10,
            7_000,
            35,
            browser_cap,
            browser_total,
        ));

        let Ok(ctx) = WgpuContext::try_new() else {
            return;
        };
        let ctx_ref = SharedGpuRef::new(ctx);
        let tight_arena = GpuTrainArena::with_budget_profile(
            SharedGpuRef::clone(&ctx_ref),
            MemoryBudgetProfile::StreamingHost {
                total_budget_bytes: 230_000,
                max_tensor_bytes: 256_000,
            },
        );
        let tight_limits = ArenaLimits {
            cap_bytes: 256_000,
            budget_bytes: 230_000,
        };
        let full_spec = CascadeChunkSpec {
            num_stems: 1,
            num_quads: 1,
            chunk_frames: 768,
            active_tiles: 0,
            write_checkpoints: true,
            has_prev: false,
            has_next: false,
        };
        {
            let mut st = tight_arena.lock_state();
            assert!(cascade_chunk_fits(&st, full_spec, tight_limits));
            assert!(adjoint_chunk_fits(&st, 1, 1, 768, 0, 256_000, 230_000));
            let dev = tight_arena.context().device();
            let budget = st.total_budget_bytes;
            let mut alloc = st.allocated_bytes;
            st.ozaki_buckets
                .ensure_storage(dev, 55_000, 256_000, budget, &mut alloc)
                .unwrap();
            st.allocated_bytes = alloc;
            assert!(!cascade_chunk_fits(&st, full_spec, tight_limits));
            assert!(!adjoint_chunk_fits(&st, 1, 1, 768, 0, 256_000, 230_000));
        }

        let cpu = CpuRunner::detect();
        let wref = build_worker_ref(&cpu, 0, 768);
        let fwd = tight_arena
            .run_wave_core_cascade(&WaveCoreCascadeInput {
                num_frames: 768,
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
        assert!(max_abs_diff_vec4(&fwd.feature_outputs, &wref.cpu_fwd.feature_outputs) < 1.0e-4);
        let adj = tight_arena
            .run_wave_core_adjoint(&WaveCoreAdjointInput {
                num_frames: 768,
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
        assert!(max_abs_diff_vec4(&adj.d_loss_d_inputs, &wref.cpu_adj.d_loss_d_inputs) < 1.0e-4);
        assert!(max_abs_diff_vec4(&adj.param_grads, &wref.cpu_adj.param_grads) < 1.0e-4);
        assert!(tight_arena.allocated_bytes() <= 230_000);

        let q10_arena = GpuTrainArena::with_budget_profile(
            ctx_ref,
            MemoryBudgetProfile::StreamingHost {
                total_budget_bytes: 2_048_000,
                max_tensor_bytes: 256_000,
            },
        );
        let q10_params: Vec<[f32; 4]> = (0..10)
            .flat_map(|_| wref.quad_params.iter().copied())
            .collect();
        let q10_in: Vec<[f32; 4]> = (0..10)
            .flat_map(|_| {
                wref.base_core_inputs[..256 * CORE_INPUT_VEC4_STRIDE]
                    .iter()
                    .copied()
            })
            .collect();
        let q10_fwd = q10_arena
            .run_wave_core_cascade(&WaveCoreCascadeInput {
                num_frames: 256,
                num_quads: 10,
                num_stems: 1,
                active_tiles: 0,
                write_checkpoints: true,
                fixed_point_scan: false,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                base_core_inputs: &q10_in,
                quad_params: &q10_params,
                tile_descriptors: &[],
            })
            .unwrap();
        let q10_dlf: Vec<[f32; 4]> = (0..10)
            .flat_map(|_| {
                wref.d_loss_d_features[..256 * FEATURE_VEC4_STRIDE]
                    .iter()
                    .copied()
            })
            .collect();
        let _ = q10_arena
            .run_wave_core_adjoint(&WaveCoreAdjointInput {
                num_frames: 256,
                num_quads: 10,
                num_stems: 1,
                active_tiles: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                core_inputs: &q10_fwd.core_inputs,
                quad_params: &q10_params,
                tile_checkpoints: &q10_fwd.tile_checkpoints,
                d_loss_d_features: &q10_dlf,
                feature_outputs: &q10_fwd.feature_outputs,
                tile_descriptors: &[],
            })
            .unwrap();
        let expected_slot7 =
            (4 * (EXIT_ADJ_BOUNDARY_VEC4_STRIDE + TILE_ADJ_BOUNDARY_VEC4_STRIDE) * 16) as u64;
        assert_eq!(
            q10_arena.lock_state().tile_adj_boundaries.size(),
            expected_slot7
        );
    }
}
