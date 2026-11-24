//! Persistent [`GpuTrainArena`] dispatches for `prescreen_population` and `gabor_bezout_grad`.

use core::mem::{size_of, size_of_val};

use zerocopy::IntoBytes;

use super::super::super::abi::TILE_FRAMES;
use super::super::super::dispatch::{read_f32_vec, read_vec2_vec, read_vec4_vec};
use super::super::super::shaders::KernelId;
use super::super::{
    ArenaLimits, GpuTrainArena, aligned_upload, create_and_record_pass, encode_readback_copy,
    finish_readbacks,
};
use crate::error::TrainError;
use crate::runner::types::{GaborBezoutInput, GaborBezoutOutput, PrescreenInput};

/// Dispatches `prescreen_population` inside [`GpuTrainArena`].
pub(in super::super) fn prescreen_population(
    arena: &GpuTrainArena,
    input: &PrescreenInput<'_>,
) -> Result<Vec<f32>, TrainError> {
    let (num_seeds, num_frames, _num_quads, num_stems) = input.validate()?;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES).max(1);
    let loss_bytes = (num_stems * num_seeds * num_tiles * size_of::<f32>()) as u64;
    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let pipe = ctx.pipeline(KernelId::PrescreenPopulation);
    let dev = ctx.device();
    let queue = ctx.queue();

    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let budget = st.total_budget_bytes;
        let alloc = &mut st.allocated_bytes;
        let b_in = st.prescreen_in.ensure_storage(
            dev,
            size_of_val(input.input_audio) as u64,
            cap,
            budget,
            alloc,
        )?;
        let b_tgt = st.prescreen_tgt.ensure_storage(
            dev,
            size_of_val(input.target_audio) as u64,
            cap,
            budget,
            alloc,
        )?;
        let b_seeds = st.prescreen_seeds.ensure_storage(
            dev,
            size_of_val(input.seed_params) as u64,
            cap,
            budget,
            alloc,
        )?;
        let b_loss = st
            .prescreen_losses
            .ensure_storage(dev, loss_bytes, cap, budget, alloc)?;

        aligned_upload(queue, &b_in, input.input_audio);
        aligned_upload(queue, &b_tgt, input.target_audio);
        aligned_upload(queue, &b_seeds, input.seed_params);

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena_prescreen"),
        });
        encoder.clear_buffer(&b_loss, 0, Some(loss_bytes));
        create_and_record_pass(
            dev,
            &mut encoder,
            &pipe,
            "prescreen_population",
            (num_seeds as u32, num_tiles as u32, num_stems as u32),
            input.uniforms.as_bytes(),
            &[&b_in, &b_tgt, &b_seeds, &b_loss],
        );
        let rb = encode_readback_copy(
            dev,
            &mut encoder,
            &b_loss,
            &mut st.readbacks[0],
            loss_bytes as usize,
            ArenaLimits {
                cap_bytes: cap,
                budget_bytes: budget,
            },
            &mut st.allocated_bytes,
        )?;
        (encoder, vec![rb])
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    read_f32_vec(&out[0])
}

/// Dispatches `gabor_bezout_grad` (`Mode A` or `Mode B`) inside [`GpuTrainArena`].
pub(in super::super) fn gabor_bezout_grad(
    arena: &GpuTrainArena,
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
    let d_bytes = num_stems * num_frames * size_of::<[f32; 2]>();
    let bl_bytes = num_stems * num_windows * size_of::<[f32; 4]>();
    let ctx = arena.context();
    let cap = ctx.max_storage_binding_bytes();
    let pipe = ctx.pipeline(KernelId::GaborBezoutGrad);
    let dev = ctx.device();
    let queue = ctx.queue();

    let mut st_guard = arena.lock_state();
    let (encoder, staging) = {
        let st = &mut *st_guard;
        let budget = st.total_budget_bytes;
        let b_pred = st.gabor_pred.ensure_storage(
            dev,
            size_of_val(input.pred_audio) as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_tgt = st.gabor_target.ensure_storage(
            dev,
            size_of_val(input.target_audio) as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_taps = st.gabor_taps.ensure_storage(
            dev,
            size_of_val(input.bezout_taps) as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_dpred = st.gabor_d_pred.ensure_storage(
            dev,
            d_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;
        let b_loss = st.gabor_losses.ensure_storage(
            dev,
            bl_bytes as u64,
            cap,
            budget,
            &mut st.allocated_bytes,
        )?;

        aligned_upload(queue, &b_pred, input.pred_audio);
        aligned_upload(queue, &b_tgt, input.target_audio);
        aligned_upload(queue, &b_taps, input.bezout_taps);

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arena_gabor_bezout"),
        });
        encoder.clear_buffer(&b_dpred, 0, Some(d_bytes as u64));
        encoder.clear_buffer(&b_loss, 0, Some(bl_bytes as u64));
        create_and_record_pass(
            dev,
            &mut encoder,
            &pipe,
            "gabor_bezout_grad",
            (num_windows as u32, num_stems as u32, 1),
            u.as_bytes(),
            &[&b_pred, &b_tgt, &b_taps, &b_dpred, &b_loss],
        );

        let limits = ArenaLimits {
            cap_bytes: cap,
            budget_bytes: budget,
        };
        let rb_d = encode_readback_copy(
            dev,
            &mut encoder,
            &b_dpred,
            &mut st.readbacks[0],
            d_bytes,
            limits,
            &mut st.allocated_bytes,
        )?;
        let rb_l = encode_readback_copy(
            dev,
            &mut encoder,
            &b_loss,
            &mut st.readbacks[1],
            bl_bytes,
            limits,
            &mut st.allocated_bytes,
        )?;
        (encoder, vec![rb_d, rb_l])
    };

    let out = finish_readbacks(ctx, encoder, staging, &st_guard)?;
    drop(st_guard);
    Ok(GaborBezoutOutput {
        d_loss_d_pred: read_vec2_vec(&out[0])?,
        band_losses: read_vec4_vec(&out[1])?,
    })
}
