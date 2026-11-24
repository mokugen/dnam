//! Persistent [`GpuTrainArena`] dispatches for 2D/split-64 `ozaki_gram_rhs_i32`
//! (`ozaki_gram_rhs_split64`) and multi-candidate `(1, num_candidates, 1)` `inner_opt_step`.

use core::mem::{size_of, size_of_val};

use zerocopy::IntoBytes;

use super::super::super::abi::{
    GPU_OZAKI_MAX_CHUNK_LEN, INNER_OPT_MODE_LSSBROYDEN, LSSBROYDEN_MAX_HISTORY_M,
    OZAKI_FLAG_ACCUMULATE_SPLIT64, OZAKI_PAIR_TILE_ROWS_DEFAULT, OZAKI_SPLIT64_WORDS_PER_PAIR,
    OZAKI_WORDS_PER_PAIR, OzakiGramUniforms,
};
use super::super::super::dispatch::{read_f32_vec, read_i32_vec, read_u32_vec, read_vec4_vec};
use super::super::super::shaders::KernelId;
use super::super::{
    ArenaLimits, GpuTrainArena, aligned_upload, create_and_record_pass, encode_readback_copy,
    finish_readbacks,
};
use super::upload_or_dummy;
use crate::error::TrainError;
use crate::runner::types::{
    InnerOptInput, InnerOptOutput, OzakiGramInput, OzakiGramOutput, OzakiSplit64Output,
};
use crate::wide_gram::uplo_u_packed_len;

const OZAKI_MAX_BATCH_CHUNKS: usize = 16;

fn ozaki_2d_grid(active_pairs: usize, tile_rows: u32) -> (u32, u32) {
    let rows = (tile_rows as usize).max(1);
    let wg_x = active_pairs.min(rows).max(1) as u32;
    let wg_y = active_pairs.div_ceil(rows).max(1) as u32;
    (wg_x, wg_y)
}

fn pack_col_major_window(
    src: &[f32],
    cols: usize,
    total_frames: usize,
    start: usize,
    len: usize,
) -> Vec<f32> {
    let mut dst = Vec::with_capacity(cols * len);
    for c in 0..cols {
        let base = c * total_frames + start;
        dst.extend_from_slice(&src[base..base + len]);
    }
    dst
}

/// Dispatches single pass (`reduce_pass = 0` or `1`) of 2D `ozaki_gram_rhs_i32` inside [`GpuTrainArena`].
pub(in super::super) fn ozaki_gram_rhs_i32(
    arena: &GpuTrainArena,
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
    if input.features.len() < dim * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32(features)",
            expected: dim * num_frames,
            actual: input.features.len(),
        });
    }
    if tc > 0 && input.targets.len() < tc * num_frames {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32(targets)",
            expected: tc * num_frames,
            actual: input.targets.len(),
        });
    }
    if input.col_scales.len() < dim + tc {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32(col_scales)",
            expected: dim + tc,
            actual: input.col_scales.len(),
        });
    }
    let active_pairs = uplo_u_packed_len(dim) + dim * tc;
    let tail_base = active_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let min_words = tail_base + active_pairs * num_chunks * OZAKI_WORDS_PER_PAIR;
    if let Some(seed) = input.seed_buckets
        && seed.len() < min_words
    {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_i32(seed_buckets)",
            expected: min_words,
            actual: seed.len(),
        });
    }
    let bucket_words = input.seed_buckets.map_or(min_words, <[i32]>::len);
    let dispatch_pairs = (bucket_words - tail_base) / (num_chunks * OZAKI_WORDS_PER_PAIR);
    let tile_rows = u.pair_tile_rows.max(OZAKI_PAIR_TILE_ROWS_DEFAULT);
    let (wg_x, wg_y) = ozaki_2d_grid(dispatch_pairs, tile_rows);
    let wg_z = if u.reduce_pass == 0 {
        num_chunks as u32
    } else {
        1
    };

    let bucket_bytes = bucket_words * size_of::<i32>();
    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let pipe = ctx.pipeline(KernelId::OzakiGramRhsI32);
    let dev = ctx.device();
    let queue = ctx.queue();

    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let budget = st.total_budget_bytes;
        let limits = ArenaLimits {
            cap_bytes: cap,
            budget_bytes: budget,
        };
        let alloc = &mut st.allocated_bytes;
        let b_feat = st.ozaki_features.ensure_storage(
            dev,
            size_of_val(input.features) as u64,
            cap,
            budget,
            alloc,
        )?;
        let b_tgt = upload_or_dummy(
            dev,
            queue,
            &mut st.ozaki_targets,
            &mut st.dummy_tiny,
            input.targets,
            limits,
            alloc,
        )?;
        let b_scales = st.ozaki_scales.ensure_storage(
            dev,
            size_of_val(input.col_scales) as u64,
            cap,
            budget,
            alloc,
        )?;
        let b_buckets =
            st.ozaki_buckets
                .ensure_storage(dev, bucket_bytes as u64, cap, budget, alloc)?;

        aligned_upload(queue, &b_feat, input.features);
        aligned_upload(queue, &b_scales, input.col_scales);
        if let Some(seed) = input.seed_buckets {
            aligned_upload(queue, &b_buckets, seed);
        }

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena_ozaki_raw"),
        });
        if input.seed_buckets.is_none() {
            encoder.clear_buffer(&b_buckets, 0, Some(bucket_bytes as u64));
        }
        create_and_record_pass(
            dev,
            &mut encoder,
            &pipe,
            "ozaki_gram_rhs_i32",
            (wg_x, wg_y, wg_z),
            u.as_bytes(),
            &[&b_feat, &b_tgt, &b_scales, &b_buckets],
        );

        let rb = encode_readback_copy(
            dev,
            &mut encoder,
            &b_buckets,
            &mut st.readbacks[0],
            bucket_bytes,
            limits,
            &mut st.allocated_bytes,
        )?;
        (encoder, vec![rb])
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    Ok(OzakiGramOutput {
        ozaki_buckets: read_i32_vec(&out[0])?,
        num_chunks,
        num_pairs: dispatch_pairs,
    })
}

/// Runs batched 2-pass exact split-64 `ozaki_gram_rhs_i32` inside [`GpuTrainArena`], returning
/// only `4 × u32` split-64 limbs per pair (`<= 1.2 MiB` readback at Ultra).
pub(in super::super) fn ozaki_gram_rhs_split64(
    arena: &GpuTrainArena,
    features_col_major: &[f32],
    targets_col_major: &[f32],
    num_frames: usize,
    feature_dim: usize,
    target_channels: usize,
    col_scales: &[[f32; 2]],
) -> Result<OzakiSplit64Output, TrainError> {
    if num_frames == 0 || feature_dim == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_split64",
            expected: 1,
            actual: 0,
        });
    }
    if features_col_major.len() < feature_dim * num_frames
        || (target_channels > 0 && targets_col_major.len() < target_channels * num_frames)
        || col_scales.len() < feature_dim + target_channels
    {
        return Err(TrainError::InvalidDimensions {
            context: "ozaki_gram_rhs_split64(buffers)",
            expected: feature_dim * num_frames,
            actual: features_col_major.len(),
        });
    }

    let active_pairs = uplo_u_packed_len(feature_dim) + feature_dim * target_channels;
    let split64_words = active_pairs * OZAKI_SPLIT64_WORDS_PER_PAIR;
    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let budget = arena.total_budget_bytes();
    let bytes_per_chunk_feat = (feature_dim * GPU_OZAKI_MAX_CHUNK_LEN * size_of::<f32>()) as u64;
    let bytes_per_chunk_bkt = (active_pairs * OZAKI_WORDS_PER_PAIR * size_of::<i32>()) as u64;
    let tail_bytes = (split64_words * size_of::<u32>()) as u64;
    let max_chunks_feat = (cap / bytes_per_chunk_feat.max(1)) as usize;
    let max_chunks_bkt = ((cap.saturating_sub(tail_bytes)) / bytes_per_chunk_bkt.max(1)) as usize;
    let half_budget = (budget / 2).max(tail_bytes);
    let max_chunks_bgt = ((half_budget.saturating_sub(tail_bytes))
        / (bytes_per_chunk_feat + bytes_per_chunk_bkt).max(1)) as usize;
    let batch_chunks = max_chunks_feat
        .min(max_chunks_bkt)
        .min(max_chunks_bgt)
        .clamp(1, OZAKI_MAX_BATCH_CHUNKS);
    let max_batch_frames = batch_chunks * GPU_OZAKI_MAX_CHUNK_LEN;

    let pipe = ctx.pipeline(KernelId::OzakiGramRhsI32);
    let dev = ctx.device();
    let queue = ctx.queue();
    let (wg_x, wg_y) = ozaki_2d_grid(active_pairs, OZAKI_PAIR_TILE_ROWS_DEFAULT);
    let scales_slice = &col_scales[..feature_dim + target_channels];

    let total_batches = num_frames.div_ceil(max_batch_frames);
    let mut final_readback = None;
    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let arena_budget = st.total_budget_bytes;
        let b_scales = st.ozaki_scales.ensure_storage(
            dev,
            size_of_val(scales_slice) as u64,
            cap,
            arena_budget,
            &mut st.allocated_bytes,
        )?;
        aligned_upload(queue, &b_scales, scales_slice);

        let mut frame_start = 0_usize;
        for batch_idx in 0..total_batches {
            let batch_frames = (num_frames - frame_start).min(max_batch_frames);
            let num_chunks = batch_frames.div_ceil(GPU_OZAKI_MAX_CHUNK_LEN);
            let bucket_bytes =
                tail_bytes + (active_pairs * num_chunks * OZAKI_WORDS_PER_PAIR * 4) as u64;
            let alloc = &mut st.allocated_bytes;
            let b_feat = st.ozaki_features.ensure_storage(
                dev,
                (feature_dim * batch_frames * 4) as u64,
                cap,
                arena_budget,
                alloc,
            )?;
            let b_tgt = if target_channels == 0 {
                st.dummy_tiny
                    .ensure_storage(dev, 16, cap, arena_budget, alloc)?
            } else {
                st.ozaki_targets.ensure_storage(
                    dev,
                    (target_channels * batch_frames * 4) as u64,
                    cap,
                    arena_budget,
                    alloc,
                )?
            };
            let b_buckets =
                st.ozaki_buckets
                    .ensure_storage(dev, bucket_bytes, cap, arena_budget, alloc)?;

            if batch_frames == num_frames {
                aligned_upload(
                    queue,
                    &b_feat,
                    &features_col_major[..feature_dim * num_frames],
                );
                if target_channels > 0 {
                    aligned_upload(
                        queue,
                        &b_tgt,
                        &targets_col_major[..target_channels * num_frames],
                    );
                }
            } else {
                let win_f = pack_col_major_window(
                    features_col_major,
                    feature_dim,
                    num_frames,
                    frame_start,
                    batch_frames,
                );
                aligned_upload(queue, &b_feat, &win_f);
                if target_channels > 0 {
                    let win_t = pack_col_major_window(
                        targets_col_major,
                        target_channels,
                        num_frames,
                        frame_start,
                        batch_frames,
                    );
                    aligned_upload(queue, &b_tgt, &win_t);
                }
            }

            let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("arena_ozaki_split64_batch"),
            });
            if batch_idx == 0 {
                encoder.clear_buffer(&b_buckets, 0, Some(bucket_bytes));
            } else {
                encoder.clear_buffer(&b_buckets, tail_bytes, Some(bucket_bytes - tail_bytes));
            }
            let u_p0 = OzakiGramUniforms {
                num_frames: batch_frames as u32,
                feature_dim: feature_dim as u32,
                target_channels: target_channels as u32,
                chunk_len: GPU_OZAKI_MAX_CHUNK_LEN as u32,
                num_chunks: num_chunks as u32,
                pair_tile_rows: OZAKI_PAIR_TILE_ROWS_DEFAULT,
                reduce_pass: 0,
                _pad: 0,
            };
            create_and_record_pass(
                dev,
                &mut encoder,
                &pipe,
                "ozaki_pass0",
                (wg_x, wg_y, num_chunks as u32),
                u_p0.as_bytes(),
                &[&b_feat, &b_tgt, &b_scales, &b_buckets],
            );
            let u_p1 = OzakiGramUniforms {
                reduce_pass: 1,
                _pad: if batch_idx > 0 {
                    OZAKI_FLAG_ACCUMULATE_SPLIT64
                } else {
                    0
                },
                ..u_p0
            };
            create_and_record_pass(
                dev,
                &mut encoder,
                &pipe,
                "ozaki_pass1",
                (wg_x, wg_y, 1),
                u_p1.as_bytes(),
                &[&b_feat, &b_tgt, &b_scales, &b_buckets],
            );

            if batch_idx + 1 == total_batches {
                let rb = encode_readback_copy(
                    dev,
                    &mut encoder,
                    &b_buckets,
                    &mut st.readbacks[0],
                    tail_bytes as usize,
                    ArenaLimits {
                        cap_bytes: cap,
                        budget_bytes: arena_budget,
                    },
                    &mut st.allocated_bytes,
                )?;
                final_readback = Some((encoder, vec![rb]));
            } else {
                queue.submit([encoder.finish()]);
            }
            frame_start += batch_frames;
        }
        final_readback.expect("non-empty batch schedule")
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    Ok(OzakiSplit64Output {
        split64_words: read_u32_vec(&out[0])?,
        num_pairs: active_pairs,
    })
}

/// Dispatches multi-candidate `(1, num_candidates, 1)` `inner_opt_step` inside [`GpuTrainArena`].
pub(in super::super) fn inner_opt_step(
    arena: &GpuTrainArena,
    input: &InnerOptInput<'_>,
) -> Result<InnerOptOutput, TrainError> {
    let u = &input.uniforms;
    let m = u.history_m as usize;
    let n_p = u.num_params as usize;
    let num_cands = input.num_candidates;
    if n_p == 0 || num_cands == 0 || m > LSSBROYDEN_MAX_HISTORY_M {
        return Err(TrainError::InvalidDimensions {
            context: "inner_opt_step",
            expected: LSSBROYDEN_MAX_HISTORY_M,
            actual: m,
        });
    }
    let exp_vec = num_cands * n_p;
    let is_lssb = (u.mode_flags & INNER_OPT_MODE_LSSBROYDEN) != 0;
    let min_state = if is_lssb {
        num_cands * (m + 1) * n_p
    } else {
        exp_vec
    };
    if input.theta.len() < exp_vec
        || input.grad.len() < exp_vec
        || input.param_bounds.len() < n_p
        || input.lbfgs_s_y.len() < min_state
        || input.search_dir.len() < exp_vec
    {
        return Err(TrainError::InvalidDimensions {
            context: "inner_opt_step(buffers)",
            expected: min_state,
            actual: input.lbfgs_s_y.len(),
        });
    }

    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let pipe = ctx.pipeline(KernelId::InnerOptStep);
    let dev = ctx.device();
    let queue = ctx.queue();
    let theta_bytes = exp_vec * size_of::<f32>();
    let state_bytes = size_of_val(input.lbfgs_s_y);

    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let budget = st.total_budget_bytes;
        let alloc = &mut st.allocated_bytes;
        let b_theta = st
            .opt_theta
            .ensure_storage(dev, theta_bytes as u64, cap, budget, alloc)?;
        let b_grad = st
            .opt_grad
            .ensure_storage(dev, theta_bytes as u64, cap, budget, alloc)?;
        let b_bounds = st.opt_bounds.ensure_storage(
            dev,
            (n_p * size_of::<[f32; 4]>()) as u64,
            cap,
            budget,
            alloc,
        )?;
        let b_state = st
            .opt_state
            .ensure_storage(dev, state_bytes as u64, cap, budget, alloc)?;
        let b_dir = st
            .opt_dir
            .ensure_storage(dev, theta_bytes as u64, cap, budget, alloc)?;

        aligned_upload(queue, &b_theta, &input.theta[..exp_vec]);
        aligned_upload(queue, &b_grad, &input.grad[..exp_vec]);
        aligned_upload(queue, &b_bounds, &input.param_bounds[..n_p]);
        aligned_upload(queue, &b_state, input.lbfgs_s_y);
        aligned_upload(queue, &b_dir, &input.search_dir[..exp_vec]);

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena_inner_opt"),
        });
        create_and_record_pass(
            dev,
            &mut encoder,
            &pipe,
            "inner_opt_step",
            (1, num_cands as u32, 1),
            u.as_bytes(),
            &[&b_theta, &b_grad, &b_bounds, &b_state, &b_dir],
        );

        let limits = ArenaLimits {
            cap_bytes: cap,
            budget_bytes: budget,
        };
        let rb_th = encode_readback_copy(
            dev,
            &mut encoder,
            &b_theta,
            &mut st.readbacks[0],
            theta_bytes,
            limits,
            &mut st.allocated_bytes,
        )?;
        let rb_st = encode_readback_copy(
            dev,
            &mut encoder,
            &b_state,
            &mut st.readbacks[1],
            state_bytes,
            limits,
            &mut st.allocated_bytes,
        )?;
        let rb_dr = encode_readback_copy(
            dev,
            &mut encoder,
            &b_dir,
            &mut st.readbacks[2],
            theta_bytes,
            limits,
            &mut st.allocated_bytes,
        )?;
        (encoder, vec![rb_th, rb_st, rb_dr])
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    Ok(InnerOptOutput {
        theta: read_f32_vec(&out[0])?,
        lbfgs_s_y: read_vec4_vec(&out[1])?,
        search_dir: read_f32_vec(&out[2])?,
    })
}
