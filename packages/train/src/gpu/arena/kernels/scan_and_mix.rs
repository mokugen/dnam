//! Persistent [`GpuTrainArena`] dispatches for `fused_core_scan_256`, `adjoint_scan_256`,
//! and `tile_mix_fwd_adj`.

use core::mem::size_of_val;

use zerocopy::IntoBytes;

use super::super::super::abi::{
    BLOCK_IDX_DIRECT_LINEAR, BLOCK_WAVE_ALL, CORE_INPUT_VEC4_STRIDE, EXIT_ADJ_BOUNDARY_VEC4_STRIDE,
    FEATURE_VEC4_STRIDE, FUSED_FLAG_FIXED_POINT_SCAN, QUAD_PARAMS_VEC4_STRIDE,
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES, TILE_MIX_DIR_FORWARD,
    TILE_MIX_DIR_PARTIAL_WEIGHTS, TILE_MIX_DIR_PULLBACK, TILE_MIX_DIR_REDUCE_WEIGHTS,
    TILE_WEIGHT_GRAD_FLOATS, block_quad_count,
};
use super::super::super::dispatch::{read_f32_vec, read_vec4_vec};
use super::super::super::shaders::KernelId;
use super::super::{
    ArenaLimits, GpuTrainArena, aligned_upload, create_and_record_pass, encode_readback_copy,
    finish_readbacks,
};
use super::upload_or_dummy;
use crate::error::TrainError;
use crate::runner::types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput, TileMixInput,
    TileMixOutput,
};

/// Dispatches `fused_core_scan_256` inside [`GpuTrainArena`].
pub(in super::super) fn fused_core_scan_256(
    arena: &GpuTrainArena,
    input: &FusedCoreScanInput<'_>,
) -> Result<FusedCoreScanOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_quads = u.num_quads as usize;
    let num_stems = u.num_stems as usize;
    let num_tiles = (u.num_tiles as usize).max(num_frames.div_ceil(TILE_FRAMES));
    if num_frames == 0 || num_quads == 0 || num_stems == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256",
            expected: 1,
            actual: 0,
        });
    }
    let total_q = num_stems * num_quads;
    let exp_in = total_q * num_frames * CORE_INPUT_VEC4_STRIDE;
    let exp_params = total_q * QUAD_PARAMS_VEC4_STRIDE;
    let exp_ckpt = total_q * num_tiles * TILE_CKPT_VEC4_STRIDE;
    let exp_feat = total_q * num_frames * FEATURE_VEC4_STRIDE;
    if input.core_inputs.len() < exp_in {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256(core_inputs)",
            expected: exp_in,
            actual: input.core_inputs.len(),
        });
    }
    if input.quad_params.len() < exp_params {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256(quad_params)",
            expected: exp_params,
            actual: input.quad_params.len(),
        });
    }
    if let Some(seed) = input.initial_checkpoints
        && seed.len() < exp_ckpt
    {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256(initial_checkpoints)",
            expected: exp_ckpt,
            actual: seed.len(),
        });
    }

    let is_direct = u.block_idx == BLOCK_IDX_DIRECT_LINEAR;
    let raw_block = u.block_idx & !FUSED_FLAG_FIXED_POINT_SCAN;
    let wg_x = if is_direct {
        num_quads
    } else {
        block_quad_count((raw_block & 3) as usize, num_quads)
    };
    if wg_x == 0 {
        return Ok(FusedCoreScanOutput {
            tile_checkpoints: input
                .initial_checkpoints
                .map_or_else(|| vec![[0.0; 4]; exp_ckpt], |s| s[..exp_ckpt].to_vec()),
            feature_outputs: vec![[0.0; 4]; exp_feat],
        });
    }

    let in_bytes = exp_in * 16;
    let params_bytes = exp_params * 16;
    let ckpt_bytes = exp_ckpt * 16;
    let feat_bytes = exp_feat * 16;
    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let pipe = ctx.pipeline(KernelId::FusedCoreScan256);
    let dev = ctx.device();
    let queue = ctx.queue();

    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let budget = st.total_budget_bytes;
        let b_in = st.core_inputs.ensure_storage(
            dev,
            in_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_params = st.quad_params.ensure_storage(
            dev,
            params_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_ckpt = st.tile_checkpoints.ensure_storage(
            dev,
            ckpt_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_feat = st.feature_outputs.ensure_storage(
            dev,
            feat_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;

        aligned_upload(queue, &b_in, &input.core_inputs[..exp_in]);
        aligned_upload(queue, &b_params, &input.quad_params[..exp_params]);
        if let Some(seed) = input.initial_checkpoints {
            aligned_upload(queue, &b_ckpt, &seed[..exp_ckpt]);
        }

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena_fused_core_scan"),
        });
        if input.initial_checkpoints.is_none() {
            encoder.clear_buffer(&b_ckpt, 0, Some(ckpt_bytes as u64));
        }
        encoder.clear_buffer(&b_feat, 0, Some(feat_bytes as u64));
        create_and_record_pass(
            dev,
            &mut encoder,
            &pipe,
            "fused_core_scan_256",
            (wg_x as u32, 1, num_stems as u32),
            u.as_bytes(),
            &[&b_in, &b_params, &b_ckpt, &b_feat],
        );

        let limits = ArenaLimits {
            cap_bytes: cap,
            budget_bytes: budget,
        };
        let rb_ckpt = encode_readback_copy(
            dev,
            &mut encoder,
            &b_ckpt,
            &mut st.readbacks[0],
            ckpt_bytes,
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
        (encoder, vec![rb_ckpt, rb_feat])
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    Ok(FusedCoreScanOutput {
        tile_checkpoints: read_vec4_vec(&out[0])?,
        feature_outputs: read_vec4_vec(&out[1])?,
    })
}

/// Dispatches 3-pass `adjoint_scan_256` (with `Slot 7` wave header) in 1 submit.
pub(in super::super) fn adjoint_scan_256(
    arena: &GpuTrainArena,
    input: &AdjointScanInput<'_>,
) -> Result<AdjointScanOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_quads = u.num_quads as usize;
    let num_stems = u.num_stems as usize;
    let num_tiles = (u.num_tiles as usize).max(num_frames.div_ceil(TILE_FRAMES));
    if num_frames == 0 || num_quads == 0 || num_stems == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "adjoint_scan_256",
            expected: 1,
            actual: 0,
        });
    }
    let total_q = num_stems * num_quads;
    let exp_in = total_q * num_frames * CORE_INPUT_VEC4_STRIDE;
    let exp_params = total_q * QUAD_PARAMS_VEC4_STRIDE;
    let exp_ckpt = total_q * num_tiles * TILE_CKPT_VEC4_STRIDE;
    let exp_feat = total_q * num_frames * FEATURE_VEC4_STRIDE;
    let exp_pgrad = total_q * num_tiles * QUAD_PARAMS_VEC4_STRIDE;
    if input.core_inputs.len() < exp_in
        || input.quad_params.len() < exp_params
        || input.tile_checkpoints.len() < exp_ckpt
        || input.d_loss_d_features.len() < exp_feat
        || input.feature_outputs.len() < exp_feat
    {
        return Err(TrainError::InvalidDimensions {
            context: "adjoint_scan_256(inputs)",
            expected: exp_feat,
            actual: input.feature_outputs.len(),
        });
    }

    let wg_x = if u.block_idx == BLOCK_IDX_DIRECT_LINEAR {
        num_quads
    } else {
        block_quad_count((u.block_idx & 3) as usize, num_quads)
    };
    let exit_vecs = num_stems * wg_x * EXIT_ADJ_BOUNDARY_VEC4_STRIDE;
    if wg_x == 0 {
        return Ok(AdjointScanOutput {
            d_loss_d_inputs: vec![[0.0; 4]; exp_in],
            param_grads: vec![[0.0; 4]; exp_pgrad],
            entry_adj_boundary: Vec::new(),
        });
    }
    if let Some(seed) = input.exit_adj_boundary
        && seed.len() < exit_vecs
    {
        return Err(TrainError::InvalidDimensions {
            context: "adjoint_scan_256(exit_adj_boundary)",
            expected: exit_vecs,
            actual: seed.len(),
        });
    }

    let exit_bytes = exit_vecs * 16;
    let tile_bound_bytes = num_stems * wg_x * num_tiles * TILE_ADJ_BOUNDARY_VEC4_STRIDE * 16;
    let slot7_bytes = exit_bytes + tile_bound_bytes;
    let du_bytes = exp_in * 16;
    let pgrad_bytes = exp_pgrad * 16;
    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let pipe = ctx.pipeline(KernelId::AdjointScan256);
    let dev = ctx.device();
    let queue = ctx.queue();

    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let budget = st.total_budget_bytes;
        let b_in = st.core_inputs.ensure_storage(
            dev,
            du_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_params = st.quad_params.ensure_storage(
            dev,
            (exp_params * 16) as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_ckpt = st.tile_checkpoints.ensure_storage(
            dev,
            (exp_ckpt * 16) as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_dlf = st.d_loss_d_features.ensure_storage(
            dev,
            (exp_feat * 16) as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_du = st.d_loss_d_inputs.ensure_storage(
            dev,
            du_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_pgrad = st.param_grads.ensure_storage(
            dev,
            pgrad_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_adj_b = st.tile_adj_boundaries.ensure_storage(
            dev,
            slot7_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_feat = st.feature_outputs.ensure_storage(
            dev,
            (exp_feat * 16) as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;

        aligned_upload(queue, &b_in, &input.core_inputs[..exp_in]);
        aligned_upload(queue, &b_params, &input.quad_params[..exp_params]);
        aligned_upload(queue, &b_ckpt, &input.tile_checkpoints[..exp_ckpt]);
        aligned_upload(queue, &b_dlf, &input.d_loss_d_features[..exp_feat]);
        aligned_upload(queue, &b_feat, &input.feature_outputs[..exp_feat]);
        if let Some(seed) = input.exit_adj_boundary {
            aligned_upload(queue, &b_adj_b, &seed[..exit_vecs]);
        }

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena_adjoint_3pass"),
        });
        encoder.clear_buffer(&b_du, 0, Some(du_bytes as u64));
        encoder.clear_buffer(&b_pgrad, 0, Some(pgrad_bytes as u64));
        if input.exit_adj_boundary.is_some() {
            if tile_bound_bytes > 0 {
                encoder.clear_buffer(&b_adj_b, exit_bytes as u64, Some(tile_bound_bytes as u64));
            }
        } else {
            encoder.clear_buffer(&b_adj_b, 0, Some(slot7_bytes as u64));
        }

        let bindings = [
            &b_in, &b_params, &b_ckpt, &b_dlf, &b_du, &b_pgrad, &b_adj_b, &b_feat,
        ];
        for (pass_mode, wg_y) in [(0_u32, num_tiles as u32), (1, 1), (2, num_tiles as u32)] {
            let mut pass_u = *u;
            pass_u.num_tiles = num_tiles as u32;
            pass_u.pass_mode = pass_mode;
            create_and_record_pass(
                dev,
                &mut encoder,
                &pipe,
                "adjoint_scan_256",
                (wg_x as u32, wg_y, num_stems as u32),
                pass_u.as_bytes(),
                &bindings,
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
        let rb_exit = encode_readback_copy(
            dev,
            &mut encoder,
            &b_adj_b,
            &mut st.readbacks[2],
            exit_bytes,
            limits,
            &mut st.allocated_bytes,
        )?;
        (encoder, vec![rb_du, rb_pg, rb_exit])
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    Ok(AdjointScanOutput {
        d_loss_d_inputs: read_vec4_vec(&out[0])?,
        param_grads: read_vec4_vec(&out[1])?,
        entry_adj_boundary: read_vec4_vec(&out[2])?,
    })
}

/// Dispatches `tile_mix_fwd_adj` (`dir_mode in 0..=3`) inside [`GpuTrainArena`].
pub(in super::super) fn tile_mix_fwd_adj(
    arena: &GpuTrainArena,
    input: &TileMixInput<'_>,
) -> Result<TileMixOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_tiles = (u.num_tiles as usize).max(num_frames.div_ceil(TILE_FRAMES));
    let num_active_tiles = u.num_active_tiles as usize;
    let num_stems = u.num_stems as usize;
    let num_quads = u.num_quads as usize;
    if num_frames == 0 || num_stems == 0 || num_quads == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "tile_mix_fwd_adj",
            expected: 1,
            actual: 0,
        });
    }
    let wg = match u.dir_mode {
        TILE_MIX_DIR_FORWARD | TILE_MIX_DIR_PULLBACK => {
            let wg_x = if u.block_wave == BLOCK_WAVE_ALL {
                num_quads
            } else {
                block_quad_count((u.block_wave & 3) as usize, num_quads)
            };
            (wg_x as u32, num_tiles as u32, num_stems as u32)
        }
        TILE_MIX_DIR_PARTIAL_WEIGHTS => {
            (num_active_tiles as u32, num_tiles as u32, num_stems as u32)
        }
        TILE_MIX_DIR_REDUCE_WEIGHTS => (num_active_tiles as u32, 1, 1),
        _ => {
            return Err(TrainError::KernelMismatch {
                context: "tile_mix_fwd_adj",
                detail: "dir_mode must be in 0..=3",
            });
        }
    };
    if wg.0 == 0 {
        return Ok(TileMixOutput {
            block_inputs_or_d_loss_d_inputs: input.block_inputs_or_d_loss_d_inputs.to_vec(),
            tile_weight_grads: input.tile_weight_grads.to_vec(),
            d_loss_d_features: input.d_loss_d_features.to_vec(),
        });
    }

    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let pipe = ctx.pipeline(KernelId::TileMixFwdAdj);
    let dev = ctx.device();
    let queue = ctx.queue();
    let tw_need = num_active_tiles * (1 + num_stems * num_tiles) * TILE_WEIGHT_GRAD_FLOATS;
    let tw_alloc_len = tw_need.max(input.tile_weight_grads.len());

    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let budget = st.total_budget_bytes;
        let limits = ArenaLimits {
            cap_bytes: cap,
            budget_bytes: budget,
        };
        let alloc = &mut st.allocated_bytes;
        let b_in = upload_or_dummy(
            dev,
            queue,
            &mut st.core_inputs,
            &mut st.dummy_tiny,
            input.block_inputs_or_d_loss_d_inputs,
            limits,
            alloc,
        )?;
        let b_desc = upload_or_dummy(
            dev,
            queue,
            &mut st.tile_descriptors,
            &mut st.dummy_tiny,
            input.tile_descriptors,
            limits,
            alloc,
        )?;
        let b_feat = upload_or_dummy(
            dev,
            queue,
            &mut st.feature_outputs,
            &mut st.dummy_tiny,
            input.feature_outputs,
            limits,
            alloc,
        )?;
        let b_tw = if tw_alloc_len == 0 {
            st.dummy_tiny.ensure_storage(dev, 16, cap, budget, alloc)?
        } else {
            let b = st.tile_weight_grads.ensure_storage(
                dev,
                (tw_alloc_len * 4) as u64,
                cap,
                budget,
                alloc,
            )?;
            if !input.tile_weight_grads.is_empty() {
                aligned_upload(queue, &b, input.tile_weight_grads);
            }
            b
        };
        let b_dlf = upload_or_dummy(
            dev,
            queue,
            &mut st.d_loss_d_features,
            &mut st.dummy_tiny,
            input.d_loss_d_features,
            limits,
            alloc,
        )?;

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena_tile_mix"),
        });
        if input.tile_weight_grads.is_empty() && tw_alloc_len > 0 {
            encoder.clear_buffer(&b_tw, 0, Some((tw_alloc_len * 4) as u64));
        }
        let mut local_u = *u;
        local_u.num_tiles = num_tiles as u32;
        create_and_record_pass(
            dev,
            &mut encoder,
            &pipe,
            "tile_mix_fwd_adj",
            wg,
            local_u.as_bytes(),
            &[&b_in, &b_desc, &b_feat, &b_tw, &b_dlf],
        );

        let rb = match u.dir_mode {
            TILE_MIX_DIR_FORWARD => encode_readback_copy(
                dev,
                &mut encoder,
                &b_in,
                &mut st.readbacks[0],
                size_of_val(input.block_inputs_or_d_loss_d_inputs).max(16),
                limits,
                &mut st.allocated_bytes,
            )?,
            TILE_MIX_DIR_PULLBACK => encode_readback_copy(
                dev,
                &mut encoder,
                &b_dlf,
                &mut st.readbacks[0],
                size_of_val(input.d_loss_d_features).max(16),
                limits,
                &mut st.allocated_bytes,
            )?,
            _ => encode_readback_copy(
                dev,
                &mut encoder,
                &b_tw,
                &mut st.readbacks[0],
                (tw_alloc_len * 4).max(16),
                limits,
                &mut st.allocated_bytes,
            )?,
        };
        (encoder, vec![rb])
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    Ok(match u.dir_mode {
        TILE_MIX_DIR_FORWARD => TileMixOutput {
            block_inputs_or_d_loss_d_inputs: if input.block_inputs_or_d_loss_d_inputs.is_empty() {
                Vec::new()
            } else {
                read_vec4_vec(&out[0])?
            },
            tile_weight_grads: input.tile_weight_grads.to_vec(),
            d_loss_d_features: input.d_loss_d_features.to_vec(),
        },
        TILE_MIX_DIR_PULLBACK => TileMixOutput {
            block_inputs_or_d_loss_d_inputs: input.block_inputs_or_d_loss_d_inputs.to_vec(),
            tile_weight_grads: input.tile_weight_grads.to_vec(),
            d_loss_d_features: if input.d_loss_d_features.is_empty() {
                Vec::new()
            } else {
                read_vec4_vec(&out[0])?
            },
        },
        _ => TileMixOutput {
            block_inputs_or_d_loss_d_inputs: input.block_inputs_or_d_loss_d_inputs.to_vec(),
            tile_weight_grads: if tw_alloc_len == 0 {
                Vec::new()
            } else {
                read_f32_vec(&out[0])?
            },
            d_loss_d_features: input.d_loss_d_features.to_vec(),
        },
    })
}
