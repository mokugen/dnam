//! Planned (`ResidentBatched` / `Streamed`) CPU reference execution for `fused_core_scan_256`
//! with `0-ULP` checkpoint chaining across streamed time blocks
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[train.pipeline.isomorphism-parity]`).

use crate::error::TrainError;
use crate::gpu::abi::{
    CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms, QUAD_PARAMS_VEC4_STRIDE,
    TILE_CKPT_VEC4_STRIDE, TILE_FRAMES,
};
use crate::planner::{CoreScanPlan, ExecutionMode};
use crate::runner::types::{FusedCoreScanInput, FusedCoreScanOutput};

use super::run_fused_core_scan_cpu;

/// Runs `fused_core_scan_256` under [`CoreScanPlan`], yielding `0-ULP` identical results between
/// [`ExecutionMode::ResidentBatched`] and [`ExecutionMode::Streamed`].
pub fn run_fused_core_scan_planned_cpu(
    input: &FusedCoreScanInput<'_>,
    plan: &CoreScanPlan,
) -> Result<FusedCoreScanOutput, TrainError> {
    if matches!(plan.mode, ExecutionMode::ResidentBatched) {
        return run_fused_core_scan_cpu(input);
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
            let local_tiles = usize::from(has_prev) + chunk_tiles + usize::from(has_next);

            let mut local_in = vec![[0.0_f32; 4]; cur_q * local_frames * CORE_INPUT_VEC4_STRIDE];
            for ql in 0..cur_q {
                let qg = q_offset + ql;
                let dst_base = (ql * local_frames + lead_pad_frames) * CORE_INPUT_VEC4_STRIDE;
                let src_base = (qg * num_frames + frame_start) * CORE_INPUT_VEC4_STRIDE;
                let count = chunk_frames * CORE_INPUT_VEC4_STRIDE;
                local_in[dst_base..dst_base + count]
                    .copy_from_slice(&input.core_inputs[src_base..src_base + count]);
            }

            let mut init_ckpts = None;
            let mut seed_storage = vec![[0.0_f32; 4]; cur_q * local_tiles * TILE_CKPT_VEC4_STRIDE];
            if has_prev {
                for ql in 0..cur_q {
                    let dst = (ql * local_tiles + 1) * TILE_CKPT_VEC4_STRIDE;
                    let src = ql * TILE_CKPT_VEC4_STRIDE;
                    seed_storage[dst..dst + TILE_CKPT_VEC4_STRIDE]
                        .copy_from_slice(&carry_ckpt[src..src + TILE_CKPT_VEC4_STRIDE]);
                }
                init_ckpts = Some(seed_storage.as_slice());
            }

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
            let local_out = run_fused_core_scan_cpu(&FusedCoreScanInput {
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
                    let exit_src =
                        (ql * local_tiles + local_tile_off + chunk_tiles) * TILE_CKPT_VEC4_STRIDE;
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
