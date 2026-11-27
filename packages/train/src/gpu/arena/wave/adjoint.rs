//! Wave-batched reverse BPTT adjoint (`Waves 3..=0` + `tile_mix_fwd_adj(1,2,3)`) inside
//! [`GpuTrainArena`] with automatic `16,384`-frame `2×` chunked boundary streaming
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use core::mem::{size_of, size_of_val};

use zerocopy::IntoBytes;

use super::super::super::abi::{
    AdjointScanUniforms, BLOCK_WAVE_ALL, CORE_INPUT_VEC4_STRIDE, EXIT_ADJ_BOUNDARY_VEC4_STRIDE,
    FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE, TILE_ADJ_BOUNDARY_VEC4_STRIDE,
    TILE_CKPT_VEC4_STRIDE, TILE_DESC_VEC4_STRIDE, TILE_FRAMES, TILE_MIX_DIR_PARTIAL_WEIGHTS,
    TILE_MIX_DIR_PULLBACK, TILE_MIX_DIR_REDUCE_WEIGHTS, TILE_WEIGHT_GRAD_FLOATS, TileMixUniforms,
    block_quad_count,
};
use super::super::super::dispatch::{read_f32_vec, read_vec4_vec};
use super::super::super::shaders::KernelId;
use super::super::{
    ArenaLimits, GpuTrainArena, aligned_upload, create_and_record_pass, encode_readback_copy,
    finish_readbacks,
};
use super::{
    StemFrameSpan, choose_stream_frames, slice_stem_frame_chunk, store_stem_frame_chunk,
    validate_descriptors,
};
use crate::error::TrainError;
use crate::planner::WAVE_STREAMING_MAX_FRAMES;
use crate::runner::types::{WaveCoreAdjointInput, WaveCoreAdjointOutput};

pub(in super::super) fn adjoint_chunk_fits(
    st: &super::super::ArenaState,
    num_stems: usize,
    num_quads: usize,
    chunk_frames: usize,
    active_tiles: usize,
    cap: u64,
    budget: u64,
) -> bool {
    let cur_q = (num_stems * num_quads) as u64;
    let f = chunk_frames as u64;
    let tiles = chunk_frames.div_ceil(TILE_FRAMES) as u64;
    let du_bytes = cur_q * f * (CORE_INPUT_VEC4_STRIDE as u64) * 16;
    let params_bytes = cur_q * (QUAD_PARAMS_VEC4_STRIDE as u64) * 16;
    let ckpt_bytes = cur_q * tiles * (TILE_CKPT_VEC4_STRIDE as u64) * 16;
    let feat_bytes = cur_q * f * (FEATURE_VEC4_STRIDE as u64) * 16;
    let pgrad_bytes = cur_q * tiles * (QUAD_PARAMS_VEC4_STRIDE as u64) * 16;
    let max_wg_x = (0..4_usize)
        .map(|b| block_quad_count(b, num_quads))
        .max()
        .unwrap_or(0) as u64;
    let max_slot7_bytes = (num_stems as u64)
        * max_wg_x
        * ((EXIT_ADJ_BOUNDARY_VEC4_STRIDE as u64) + tiles * (TILE_ADJ_BOUNDARY_VEC4_STRIDE as u64))
        * 16;

    let mut updates = [
        (&st.core_inputs, du_bytes),
        (&st.quad_params, params_bytes),
        (&st.tile_checkpoints, ckpt_bytes),
        (&st.d_loss_d_features, feat_bytes),
        (&st.feature_outputs, feat_bytes),
        (&st.d_loss_d_inputs, du_bytes),
        (&st.param_grads, pgrad_bytes),
        (&st.dummy_tiny, 16),
        (&st.tile_adj_boundaries, max_slot7_bytes),
        (&st.readbacks[0], du_bytes),
        (&st.readbacks[1], pgrad_bytes),
        (&st.tile_descriptors, 0),
        (&st.tile_weight_grads, 0),
        (&st.readbacks[2], 0),
        (&st.exit_adj_wave[0], 0),
        (&st.exit_adj_wave[1], 0),
        (&st.exit_adj_wave[2], 0),
        (&st.exit_adj_wave[3], 0),
    ];
    let mut len = 11_usize;
    if active_tiles > 0 {
        let desc_bytes = (active_tiles * TILE_DESC_VEC4_STRIDE * 16) as u64;
        let tw_bytes = (active_tiles
            * (1 + num_stems * (tiles as usize))
            * TILE_WEIGHT_GRAD_FLOATS
            * size_of::<f32>()) as u64;
        let tw_rb_bytes = (active_tiles * TILE_WEIGHT_GRAD_FLOATS * size_of::<f32>()) as u64;
        updates[len] = (&st.tile_descriptors, desc_bytes);
        updates[len + 1] = (&st.tile_weight_grads, tw_bytes);
        updates[len + 2] = (&st.readbacks[2], tw_rb_bytes);
        len += 3;
    }
    for b in 0..4_usize {
        let wg_x = block_quad_count(b, num_quads);
        if wg_x > 0 {
            let exit_bytes = (num_stems * wg_x * EXIT_ADJ_BOUNDARY_VEC4_STRIDE * 16) as u64;
            updates[len] = (&st.exit_adj_wave[b], exit_bytes);
            len += 1;
        }
    }
    let (max_single, total) = st.project_allocations(&updates[..len]);
    max_single <= cap && total <= budget
}

/// Executes reverse waves (`Waves 3..=0` + `tile_mix_fwd_adj(1,2,3)`) inside [`GpuTrainArena`].
pub(in super::super) fn run_wave_core_adjoint(
    arena: &GpuTrainArena,
    input: &WaveCoreAdjointInput<'_>,
) -> Result<WaveCoreAdjointOutput, TrainError> {
    let num_frames = input.num_frames;
    let num_quads = input.num_quads;
    let num_stems = input.num_stems;
    let active_tiles = input.active_tiles;
    if num_frames == 0 || num_stems == 0 || !(1..=10).contains(&num_quads) {
        return Err(TrainError::InvalidDimensions {
            context: "run_wave_core_adjoint",
            expected: 1,
            actual: 0,
        });
    }
    let total_q = num_stems * num_quads;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    if input.core_inputs.len() < total_q * num_frames * CORE_INPUT_VEC4_STRIDE
        || input.quad_params.len() < total_q * QUAD_PARAMS_VEC4_STRIDE
        || input.tile_checkpoints.len() < total_q * num_tiles * TILE_CKPT_VEC4_STRIDE
        || input.d_loss_d_features.len() < total_q * num_frames * FEATURE_VEC4_STRIDE
        || input.feature_outputs.len() < total_q * num_frames * FEATURE_VEC4_STRIDE
    {
        return Err(TrainError::InvalidDimensions {
            context: "run_wave_core_adjoint(buffers)",
            expected: total_q * num_frames * FEATURE_VEC4_STRIDE,
            actual: input.feature_outputs.len(),
        });
    }
    validate_descriptors(
        input.tile_descriptors,
        active_tiles,
        "run_wave_core_adjoint(tile_descriptors)",
    )?;

    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let budget = arena.total_budget_bytes();
    let pipe_adj = ctx.pipeline(KernelId::AdjointScan256);
    let pipe_mix = ctx.pipeline(KernelId::TileMixFwdAdj);
    let dev = ctx.device();
    let queue = ctx.queue();
    // Hold arena lock from budget projection through all stem groups and reverse chunks so
    // non-wave allocations cannot invalidate chunk sizing and `st.exit_adj_wave[0..4]` stays intact.
    let mut st_guard = arena.lock_state();
    let resident = num_frames <= WAVE_STREAMING_MAX_FRAMES
        && adjoint_chunk_fits(
            &st_guard,
            num_stems,
            num_quads,
            num_frames,
            active_tiles,
            cap,
            budget,
        );
    let (stem_group, block_frames) = if resident {
        (num_stems, num_frames)
    } else {
        let bf = choose_stream_frames(num_frames, |f| {
            adjoint_chunk_fits(
                &st_guard,
                1,
                num_quads,
                f.min(num_frames),
                active_tiles,
                cap,
                budget,
            )
        });
        (1, bf)
    };
    let num_chunks = num_frames.div_ceil(block_frames);

    let mut out_du = vec![[0.0_f32; 4]; total_q * num_frames * CORE_INPUT_VEC4_STRIDE];
    let mut out_pgrad = vec![[0.0_f32; 4]; total_q * num_tiles * QUAD_PARAMS_VEC4_STRIDE];
    let mut tw_acc = vec![0.0_f64; active_tiles * TILE_WEIGHT_GRAD_FLOATS];

    for stem_start in (0..num_stems).step_by(stem_group) {
        let cur_stems = (num_stems - stem_start).min(stem_group);
        let cur_q = cur_stems * num_quads;
        let p_off = stem_start * num_quads * QUAD_PARAMS_VEC4_STRIDE;
        let sub_params = &input.quad_params[p_off..p_off + cur_q * QUAD_PARAMS_VEC4_STRIDE];

        for chunk_idx in (0..num_chunks).rev() {
            let frame_start = chunk_idx * block_frames;
            let chunk_frames = (num_frames - frame_start).min(block_frames);
            let chunk_tiles = chunk_frames.div_ceil(TILE_FRAMES);
            let tile_base = frame_start / TILE_FRAMES;
            let is_last_chunk = chunk_idx + 1 == num_chunks;
            let has_prev_chunk = chunk_idx > 0;

            let span_du = StemFrameSpan {
                stem_start,
                cur_stems,
                num_quads,
                total_frames: num_frames,
                frame_start,
                chunk_frames,
                stride: CORE_INPUT_VEC4_STRIDE,
                lead_pad: 0,
            };
            let span_feat = StemFrameSpan {
                stride: FEATURE_VEC4_STRIDE,
                ..span_du
            };
            let span_ckpt = StemFrameSpan {
                total_frames: num_tiles,
                frame_start: tile_base,
                chunk_frames: chunk_tiles,
                stride: TILE_CKPT_VEC4_STRIDE,
                ..span_du
            };
            let sub_in = slice_stem_frame_chunk(input.core_inputs, span_du, 0, [0.0_f32; 4]);
            let sub_dlf =
                slice_stem_frame_chunk(input.d_loss_d_features, span_feat, 0, [0.0_f32; 4]);
            let sub_feat =
                slice_stem_frame_chunk(input.feature_outputs, span_feat, 0, [0.0_f32; 4]);
            let sub_ckpt =
                slice_stem_frame_chunk(input.tile_checkpoints, span_ckpt, 0, [0.0_f32; 4]);

            let du_bytes = size_of_val(sub_in.as_slice());
            let pgrad_bytes = cur_q * chunk_tiles * QUAD_PARAMS_VEC4_STRIDE * 16;
            let tw_floats = active_tiles * (1 + cur_stems * chunk_tiles) * TILE_WEIGHT_GRAD_FLOATS;
            let tw_bytes = tw_floats * size_of::<f32>();

            let (encoder, staging) = {
                let st = &mut *st_guard;
                let alloc = &mut st.allocated_bytes;
                let b_in =
                    st.core_inputs
                        .ensure_storage(dev, du_bytes as u64, cap, budget, alloc)?;
                let b_params = st.quad_params.ensure_storage(
                    dev,
                    size_of_val(sub_params) as u64,
                    cap,
                    budget,
                    alloc,
                )?;
                let b_ckpt = st.tile_checkpoints.ensure_storage(
                    dev,
                    size_of_val(sub_ckpt.as_slice()) as u64,
                    cap,
                    budget,
                    alloc,
                )?;
                let b_dlf = st.d_loss_d_features.ensure_storage(
                    dev,
                    size_of_val(sub_dlf.as_slice()) as u64,
                    cap,
                    budget,
                    alloc,
                )?;
                let b_feat = st.feature_outputs.ensure_storage(
                    dev,
                    size_of_val(sub_feat.as_slice()) as u64,
                    cap,
                    budget,
                    alloc,
                )?;
                let b_du =
                    st.d_loss_d_inputs
                        .ensure_storage(dev, du_bytes as u64, cap, budget, alloc)?;
                let b_pgrad =
                    st.param_grads
                        .ensure_storage(dev, pgrad_bytes as u64, cap, budget, alloc)?;
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
                let b_tw = if active_tiles == 0 {
                    b_dummy.clone()
                } else {
                    st.tile_weight_grads
                        .ensure_storage(dev, tw_bytes as u64, cap, budget, alloc)?
                };

                let max_wg_x = (0..4_usize)
                    .map(|b| block_quad_count(b, num_quads))
                    .max()
                    .unwrap_or(0);
                let max_slot7_bytes = (cur_stems
                    * max_wg_x
                    * (EXIT_ADJ_BOUNDARY_VEC4_STRIDE + chunk_tiles * TILE_ADJ_BOUNDARY_VEC4_STRIDE)
                    * 16) as u64;
                // Pre-size Slot 7 once for the widest active wave block before recording passes
                // so earlier reverse passes never reference a reallocated/dropped buffer.
                let b_adj_b = st.tile_adj_boundaries.ensure_storage(
                    dev,
                    max_slot7_bytes,
                    cap,
                    budget,
                    &mut st.allocated_bytes,
                )?;

                aligned_upload(queue, &b_in, &sub_in);
                aligned_upload(queue, &b_params, sub_params);
                aligned_upload(queue, &b_ckpt, &sub_ckpt);
                aligned_upload(queue, &b_dlf, &sub_dlf);
                aligned_upload(queue, &b_feat, &sub_feat);

                let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("arena_wave_adjoint"),
                });
                encoder.clear_buffer(&b_du, 0, Some(du_bytes as u64));
                encoder.clear_buffer(&b_pgrad, 0, Some(pgrad_bytes as u64));

                for b in (0..4_usize).rev() {
                    let wg_x = block_quad_count(b, num_quads);
                    if wg_x == 0 {
                        continue;
                    }
                    if b < 3 && active_tiles > 0 {
                        let mix_u = TileMixUniforms {
                            num_frames: chunk_frames as u32,
                            num_tiles: chunk_tiles as u32,
                            num_active_tiles: active_tiles as u32,
                            dir_mode: TILE_MIX_DIR_PULLBACK,
                            num_stems: cur_stems as u32,
                            block_wave: b as u32,
                            num_quads: num_quads as u32,
                            _pad1: 0,
                        };
                        create_and_record_pass(
                            dev,
                            &mut encoder,
                            &pipe_mix,
                            "wave_mix_pullback",
                            (wg_x as u32, chunk_tiles as u32, cur_stems as u32),
                            mix_u.as_bytes(),
                            &[&b_du, &b_desc, &b_feat, &b_tw, &b_dlf],
                        );
                    }

                    let wave_count = cur_stems * wg_x;
                    let exit_bytes = (wave_count * EXIT_ADJ_BOUNDARY_VEC4_STRIDE * 16) as u64;
                    let tile_bound_bytes =
                        (wave_count * chunk_tiles * TILE_ADJ_BOUNDARY_VEC4_STRIDE * 16) as u64;
                    let b_exit_b = st.exit_adj_wave[b].ensure_storage(
                        dev,
                        exit_bytes,
                        cap,
                        budget,
                        &mut st.allocated_bytes,
                    )?;
                    if is_last_chunk {
                        encoder.clear_buffer(&b_exit_b, 0, Some(exit_bytes));
                        encoder.clear_buffer(&b_adj_b, 0, Some(exit_bytes + tile_bound_bytes));
                    } else {
                        encoder.copy_buffer_to_buffer(&b_exit_b, 0, &b_adj_b, 0, exit_bytes);
                        if tile_bound_bytes > 0 {
                            encoder.clear_buffer(&b_adj_b, exit_bytes, Some(tile_bound_bytes));
                        }
                    }

                    let adj_bindings = [
                        &b_in, &b_params, &b_ckpt, &b_dlf, &b_du, &b_pgrad, &b_adj_b, &b_feat,
                    ];
                    for (pass_mode, wg_y) in
                        [(0_u32, chunk_tiles as u32), (1, 1), (2, chunk_tiles as u32)]
                    {
                        let adj_u = AdjointScanUniforms {
                            num_frames: chunk_frames as u32,
                            num_tiles: chunk_tiles as u32,
                            num_quads: num_quads as u32,
                            num_stems: cur_stems as u32,
                            pass_mode,
                            soft_rect_eps: input.soft_rect_eps,
                            tiny_crossing_eps: input.tiny_crossing_eps,
                            block_idx: b as u32,
                        };
                        create_and_record_pass(
                            dev,
                            &mut encoder,
                            &pipe_adj,
                            "wave_adjoint_scan",
                            (wg_x as u32, wg_y, cur_stems as u32),
                            adj_u.as_bytes(),
                            &adj_bindings,
                        );
                    }
                    if has_prev_chunk {
                        const HALF_EXIT_ADJ_VEC4: usize = EXIT_ADJ_BOUNDARY_VEC4_STRIDE / 2;
                        for qw in 0..wave_count {
                            let src_off = ((qw * EXIT_ADJ_BOUNDARY_VEC4_STRIDE
                                + HALF_EXIT_ADJ_VEC4)
                                * 16) as u64;
                            let dst_off = (qw * EXIT_ADJ_BOUNDARY_VEC4_STRIDE * 16) as u64;
                            encoder.copy_buffer_to_buffer(
                                &b_adj_b,
                                src_off,
                                &b_exit_b,
                                dst_off,
                                (HALF_EXIT_ADJ_VEC4 * 16) as u64,
                            );
                        }
                    }
                }

                if active_tiles > 0 {
                    encoder.clear_buffer(&b_tw, 0, Some(tw_bytes as u64));
                    let u_part = TileMixUniforms {
                        num_frames: chunk_frames as u32,
                        num_tiles: chunk_tiles as u32,
                        num_active_tiles: active_tiles as u32,
                        dir_mode: TILE_MIX_DIR_PARTIAL_WEIGHTS,
                        num_stems: cur_stems as u32,
                        block_wave: BLOCK_WAVE_ALL,
                        num_quads: num_quads as u32,
                        _pad1: 0,
                    };
                    create_and_record_pass(
                        dev,
                        &mut encoder,
                        &pipe_mix,
                        "wave_mix_partials",
                        (active_tiles as u32, chunk_tiles as u32, cur_stems as u32),
                        u_part.as_bytes(),
                        &[&b_du, &b_desc, &b_feat, &b_tw, &b_dlf],
                    );
                    let u_red = TileMixUniforms {
                        dir_mode: TILE_MIX_DIR_REDUCE_WEIGHTS,
                        ..u_part
                    };
                    create_and_record_pass(
                        dev,
                        &mut encoder,
                        &pipe_mix,
                        "wave_mix_reduce",
                        (active_tiles as u32, 1, 1),
                        u_red.as_bytes(),
                        &[&b_du, &b_desc, &b_feat, &b_tw, &b_dlf],
                    );
                }

                let limits = ArenaLimits {
                    cap_bytes: cap,
                    budget_bytes: budget,
                };
                let rb_du = encode_readback_copy(
                    dev,
                    &mut encoder,
                    &b_du,
                    &mut st.readbacks[0],
                    du_bytes,
                    limits,
                    &mut st.allocated_bytes,
                )?;
                let rb_pg = encode_readback_copy(
                    dev,
                    &mut encoder,
                    &b_pgrad,
                    &mut st.readbacks[1],
                    pgrad_bytes,
                    limits,
                    &mut st.allocated_bytes,
                )?;
                let mut rbs = vec![rb_du, rb_pg];
                if active_tiles > 0 {
                    let rb_tw = encode_readback_copy(
                        dev,
                        &mut encoder,
                        &b_tw,
                        &mut st.readbacks[2],
                        active_tiles * TILE_WEIGHT_GRAD_FLOATS * size_of::<f32>(),
                        limits,
                        &mut st.allocated_bytes,
                    )?;
                    rbs.push(rb_tw);
                }
                (encoder, rbs)
            };

            let read = finish_readbacks(ctx, encoder, staging, &st_guard)?;
            let chunk_du = read_vec4_vec(&read[0])?;
            let chunk_pg = read_vec4_vec(&read[1])?;
            store_stem_frame_chunk(&mut out_du, &chunk_du, span_du, chunk_frames);
            store_stem_frame_chunk(
                &mut out_pgrad,
                &chunk_pg,
                StemFrameSpan {
                    stride: QUAD_PARAMS_VEC4_STRIDE,
                    ..span_ckpt
                },
                chunk_tiles,
            );
            if active_tiles > 0 {
                let tw_chunk = read_f32_vec(&read[2])?;
                for (acc, &v) in tw_acc.iter_mut().zip(tw_chunk.iter()) {
                    *acc += f64::from(v);
                }
            }
        }
    }
    drop(st_guard);

    Ok(WaveCoreAdjointOutput {
        d_loss_d_inputs: out_du,
        param_grads: out_pgrad,
        tile_weight_grads: tw_acc.into_iter().map(|v| v as f32).collect(),
    })
}
