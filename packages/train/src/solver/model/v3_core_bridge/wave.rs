//! Wave-batched forward cascade & reverse BPTT runner for the `SONA v3.0` CPU bridge
//! (`[4,3,2,1]` blocks, `35` tiles, `25`-slot `CoreQuadParamsStereo4`, `8`-Vec4 ckpts).

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4, SonaBlockStateStereo8,
    StereoVec8, TINY_CROSSING_EPS, canonicalize_zero_f32_bits, step_core_quad_phs_stereo8,
    step_core_quad_stereo8,
};

use super::math::{
    FwdFrameState, LaneParams, ReverseCarry, ReverseStepContext, TileGradAccum, flush_zero,
    step_lane_forward, step_lane_reverse,
};
use super::{
    V3_QUAD_PARAMS_VEC4_STRIDE, V3_TILE_CKPT_VEC4_STRIDE, canon_vec4, read_v3_checkpoint,
    unpack_v3_core_quad_params, v3_block_quad_count, v3_quad_param_slots, v3_resolve_quad_idx,
    write_v3_checkpoint, write_v3_lane_param_grads,
};
use crate::cpu::tile_mix::run_tile_mix_fwd_adj_cpu;
use crate::error::TrainError;
use crate::gpu::abi::{
    BLOCK_WAVE_ALL, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, TILE_DESC_VEC4_STRIDE,
    TILE_FRAMES, TILE_MIX_DIR_PARTIAL_WEIGHTS, TILE_MIX_DIR_REDUCE_WEIGHTS, TileMixUniforms,
};
use crate::runner::types::{
    TileMixInput, WaveCoreAdjointInput, WaveCoreAdjointOutput, WaveCoreCascadeInput,
    WaveCoreCascadeOutput,
};

fn flush_zero_vec4(v: [f32; 4]) -> [f32; 4] {
    v.map(canonicalize_zero_f32_bits)
}

fn load_mat4(descriptors: &[[f32; 4]], base: usize) -> [[f32; 4]; 4] {
    [
        descriptors[base],
        descriptors[base + 1],
        descriptors[base + 2],
        descriptors[base + 3],
    ]
}

fn accum_col_major(acc: &mut [f32; 4], mat: &[[f32; 4]; 4], src: [f32; 4]) {
    for d in 0..4 {
        acc[d] += mat[0][d] * src[0] + mat[1][d] * src[1] + mat[2][d] * src[2] + mat[3][d] * src[3];
    }
}

fn accum_transpose_dot(acc: &mut [f32; 4], mat: &[[f32; 4]; 4], downstream: [f32; 4]) {
    for s in 0..4 {
        acc[s] += mat[s][0] * downstream[0]
            + mat[s][1] * downstream[1]
            + mat[s][2] * downstream[2]
            + mat[s][3] * downstream[3];
    }
}

fn v3_wave_quads(block_wave: u32, num_quads: usize) -> Vec<usize> {
    let count = v3_block_quad_count((block_wave & 3) as usize, num_quads);
    (0..count)
        .filter_map(|ql| {
            let q = v3_resolve_quad_idx(block_wave, ql, num_quads);
            (q < num_quads).then_some(q)
        })
        .collect()
}

fn unpack_all_v3_quads(
    quad_params: &[[f32; 4]],
    total_q: usize,
) -> Result<Vec<CoreQuadParamsStereo4<GpuVec4>>, TrainError> {
    let needed = total_q * V3_QUAD_PARAMS_VEC4_STRIDE;
    if quad_params.len() < needed {
        return Err(TrainError::InvalidDimensions {
            context: "v3_quad_params",
            expected: needed,
            actual: quad_params.len(),
        });
    }
    let mut out = Vec::with_capacity(total_q);
    for q in 0..total_q {
        let base = q * V3_QUAD_PARAMS_VEC4_STRIDE;
        out.push(unpack_v3_core_quad_params(
            &quad_params[base..base + V3_QUAD_PARAMS_VEC4_STRIDE],
        )?);
    }
    Ok(out)
}

fn apply_v3_tile_mix_forward(
    block_wave: u32,
    num_frames: usize,
    num_quads: usize,
    active_tiles: usize,
    tile_descriptors: &[[f32; 4]],
    feature_outputs: &[[f32; 4]],
    core_inputs: &mut [[f32; 4]],
) {
    let dst_quads = v3_wave_quads(block_wave, num_quads);
    for &dst_q in &dst_quads {
        for tile in 0..active_tiles {
            let d_base = tile * TILE_DESC_VEC4_STRIDE;
            let hdr = tile_descriptors[d_base];
            let src_q = hdr[0].to_bits() as usize;
            let desc_dst = hdr[1].to_bits() as usize;
            if desc_dst != dst_q || src_q >= num_quads {
                continue;
            }
            let w_odd = load_mat4(tile_descriptors, d_base + 1);
            let w_even = load_mat4(tile_descriptors, d_base + 5);
            let w_lru = load_mat4(tile_descriptors, d_base + 9);
            for t in 0..num_frames {
                let f_base = (src_q * num_frames + t) * FEATURE_VEC4_STRIDE;
                let i_base = (dst_q * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                let mut acc_l = core_inputs[i_base];
                let mut acc_r = core_inputs[i_base + 1];
                accum_col_major(&mut acc_l, &w_odd, feature_outputs[f_base + 2]);
                accum_col_major(&mut acc_l, &w_even, feature_outputs[f_base + 4]);
                accum_col_major(&mut acc_l, &w_lru, feature_outputs[f_base]);
                accum_col_major(&mut acc_r, &w_odd, feature_outputs[f_base + 3]);
                accum_col_major(&mut acc_r, &w_even, feature_outputs[f_base + 5]);
                accum_col_major(&mut acc_r, &w_lru, feature_outputs[f_base + 1]);
                core_inputs[i_base] = flush_zero_vec4(acc_l);
                core_inputs[i_base + 1] = flush_zero_vec4(acc_r);
            }
        }
    }
}

fn apply_v3_tile_mix_pullback(
    block_wave: u32,
    num_frames: usize,
    num_quads: usize,
    active_tiles: usize,
    tile_descriptors: &[[f32; 4]],
    d_loss_d_inputs: &[[f32; 4]],
    d_loss_d_features: &mut [[f32; 4]],
) {
    let src_quads = v3_wave_quads(block_wave, num_quads);
    for &src_q in &src_quads {
        for tile in 0..active_tiles {
            let d_base = tile * TILE_DESC_VEC4_STRIDE;
            let hdr = tile_descriptors[d_base];
            let desc_src = hdr[0].to_bits() as usize;
            let dst_q = hdr[1].to_bits() as usize;
            if desc_src != src_q || dst_q >= num_quads {
                continue;
            }
            let w_odd = load_mat4(tile_descriptors, d_base + 1);
            let w_even = load_mat4(tile_descriptors, d_base + 5);
            let w_lru = load_mat4(tile_descriptors, d_base + 9);
            for t in 0..num_frames {
                let f_base = (src_q * num_frames + t) * FEATURE_VEC4_STRIDE;
                let dl_base = (dst_q * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                let dl_l = d_loss_d_inputs[dl_base];
                let dl_r = d_loss_d_inputs[dl_base + 1];
                accum_transpose_dot(&mut d_loss_d_features[f_base + 2], &w_odd, dl_l);
                accum_transpose_dot(&mut d_loss_d_features[f_base + 3], &w_odd, dl_r);
                accum_transpose_dot(&mut d_loss_d_features[f_base + 4], &w_even, dl_l);
                accum_transpose_dot(&mut d_loss_d_features[f_base + 5], &w_even, dl_r);
                accum_transpose_dot(&mut d_loss_d_features[f_base], &w_lru, dl_l);
                accum_transpose_dot(&mut d_loss_d_features[f_base + 1], &w_lru, dl_r);
                for k in 0..FEATURE_VEC4_STRIDE {
                    d_loss_d_features[f_base + k] = flush_zero_vec4(d_loss_d_features[f_base + k]);
                }
            }
        }
    }
}

pub(crate) fn run_v3_wave_core_cascade(
    input: &WaveCoreCascadeInput<'_>,
) -> Result<WaveCoreCascadeOutput, TrainError> {
    let num_frames = input.num_frames;
    let num_quads = input.num_quads;
    let num_stems = input.num_stems;
    let active_tiles = input.active_tiles;
    if num_frames == 0 || num_stems != 1 || !(1..=10).contains(&num_quads) {
        return Err(TrainError::InvalidDimensions {
            context: "run_v3_wave_core_cascade",
            expected: 1,
            actual: 0,
        });
    }
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let params = unpack_all_v3_quads(input.quad_params, num_quads)?;
    let mut core_inputs =
        input.base_core_inputs[..num_quads * num_frames * CORE_INPUT_VEC4_STRIDE].to_vec();
    let mut feature_outputs = vec![[0.0_f32; 4]; num_quads * num_frames * FEATURE_VEC4_STRIDE];
    let mut tile_checkpoints = if input.write_checkpoints {
        vec![[0.0_f32; 4]; num_quads * num_tiles * V3_TILE_CKPT_VEC4_STRIDE]
    } else {
        Vec::new()
    };

    for block_idx in 0..4_u32 {
        let wg_x = v3_block_quad_count(block_idx as usize, num_quads);
        if wg_x == 0 {
            continue;
        }
        if block_idx > 0 && active_tiles > 0 {
            apply_v3_tile_mix_forward(
                block_idx,
                num_frames,
                num_quads,
                active_tiles,
                input.tile_descriptors,
                &feature_outputs,
                &mut core_inputs,
            );
        }
        let is_phs = block_idx == 0;
        for &quad_idx in &v3_wave_quads(block_idx, num_quads) {
            let mut state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
            for tile in 0..num_tiles {
                if input.write_checkpoints {
                    write_v3_checkpoint(&mut tile_checkpoints, quad_idx, num_tiles, tile, &state);
                }
                let base_frame = tile * TILE_FRAMES;
                let end_frame = (base_frame + TILE_FRAMES).min(num_frames);
                for t in base_frame..end_frame {
                    let in_idx = (quad_idx * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                    let u_vec = InterleavedStereo8::from_lr(
                        GpuVec4::from_array(core_inputs[in_idx].map(flush_zero)),
                        GpuVec4::from_array(core_inputs[in_idx + 1].map(flush_zero)),
                    );
                    let out = if is_phs {
                        step_core_quad_phs_stereo8(u_vec, &mut state, &params[quad_idx])
                    } else {
                        step_core_quad_stereo8(u_vec, &mut state, &params[quad_idx])
                    };
                    let (vl, vr) = out.lru_state.into_lr();
                    let (ol, or) = out.adaa_odd.into_lr();
                    let (el, er) = out.adaa_even.into_lr();
                    let f_idx = (quad_idx * num_frames + t) * FEATURE_VEC4_STRIDE;
                    feature_outputs[f_idx] = canon_vec4(vl);
                    feature_outputs[f_idx + 1] = canon_vec4(vr);
                    feature_outputs[f_idx + 2] = canon_vec4(ol);
                    feature_outputs[f_idx + 3] = canon_vec4(or);
                    feature_outputs[f_idx + 4] = canon_vec4(el);
                    feature_outputs[f_idx + 5] = canon_vec4(er);
                }
            }
        }
    }

    Ok(WaveCoreCascadeOutput {
        core_inputs,
        tile_checkpoints,
        feature_outputs,
    })
}

pub(crate) fn run_v3_wave_core_adjoint(
    input: &WaveCoreAdjointInput<'_>,
) -> Result<WaveCoreAdjointOutput, TrainError> {
    let num_frames = input.num_frames;
    let num_quads = input.num_quads;
    let num_stems = input.num_stems;
    let active_tiles = input.active_tiles;
    if num_frames == 0 || num_stems != 1 || !(1..=10).contains(&num_quads) {
        return Err(TrainError::InvalidDimensions {
            context: "run_v3_wave_core_adjoint",
            expected: 1,
            actual: 0,
        });
    }
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let quads = unpack_all_v3_quads(input.quad_params, num_quads)?;
    let mut cur_dlf =
        input.d_loss_d_features[..num_quads * num_frames * FEATURE_VEC4_STRIDE].to_vec();
    let mut out_du = vec![[0.0_f32; 4]; num_quads * num_frames * CORE_INPUT_VEC4_STRIDE];
    let mut out_pgrad = vec![[0.0_f32; 4]; num_quads * num_tiles * V3_QUAD_PARAMS_VEC4_STRIDE];
    let tiny_eps = if input.tiny_crossing_eps > 0.0 {
        input.tiny_crossing_eps
    } else {
        TINY_CROSSING_EPS
    };

    for block_idx in (0..4_u32).rev() {
        let wg_x = v3_block_quad_count(block_idx as usize, num_quads);
        if wg_x == 0 {
            continue;
        }
        if block_idx < 3 && active_tiles > 0 {
            apply_v3_tile_mix_pullback(
                block_idx,
                num_frames,
                num_quads,
                active_tiles,
                input.tile_descriptors,
                &out_du,
                &mut cur_dlf,
            );
        }
        let is_phs = block_idx == 0;
        let mut tile_fwd = vec![FwdFrameState::default(); TILE_FRAMES];
        for &quad_idx in &v3_wave_quads(block_idx, num_quads) {
            let svf_g_arr = input.quad_params
                [quad_idx * V3_QUAD_PARAMS_VEC4_STRIDE + v3_quad_param_slots::SVF_G];
            for c in 0..4 {
                let lp = LaneParams::extract(&quads[quad_idx], svf_g_arr, c);
                let svf_c = lp.svf_coeffs();
                let lru_c = lp.lru_coeffs();
                let mut carry = ReverseCarry::default();
                for tile in (0..num_tiles).rev() {
                    let tile_base = tile * TILE_FRAMES;
                    let valid_len = num_frames.saturating_sub(tile_base).min(TILE_FRAMES);
                    if valid_len == 0 {
                        continue;
                    }
                    let ckpt =
                        read_v3_checkpoint(input.tile_checkpoints, quad_idx, num_tiles, tile);
                    let (s1_l, s1_r) = ckpt.svf_s1.into_lr();
                    let (s2_l, s2_r) = ckpt.svf_s2.into_lr();
                    let (v_l, v_r) = ckpt.lru_v.into_lr();
                    let (ap_l, ap_r) = ckpt.adaa_prev.into_lr();
                    let mut lane_state = [
                        s1_l.to_array()[c],
                        s2_l.to_array()[c],
                        s1_r.to_array()[c],
                        s2_r.to_array()[c],
                        v_l.to_array()[c],
                        v_r.to_array()[c],
                        ap_l.to_array()[c],
                        ap_r.to_array()[c],
                    ];
                    for local_t in 0..valid_len {
                        let frame = tile_base + local_t;
                        let in_base = (quad_idx * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
                        let u_l = flush_zero(input.core_inputs[in_base][c]);
                        let u_r = flush_zero(input.core_inputs[in_base + 1][c]);
                        tile_fwd[local_t] = step_lane_forward(
                            &lp,
                            &svf_c,
                            &lru_c,
                            u_l,
                            u_r,
                            &mut lane_state,
                            is_phs,
                        );
                    }
                    let ctx = ReverseStepContext {
                        d_loss_d_features: &cur_dlf,
                        feature_outputs: input.feature_outputs,
                        q_global: quad_idx,
                        num_frames,
                        c,
                        tiny_eps,
                    };
                    let mut acc = TileGradAccum::default();
                    for local_t in (0..valid_len).rev() {
                        let frame = tile_base + local_t;
                        let (du_l, du_r) = step_lane_reverse(
                            &ctx,
                            &tile_fwd[local_t],
                            &lp,
                            frame,
                            &mut carry,
                            &mut acc,
                            is_phs,
                        );
                        let out_base = (quad_idx * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
                        out_du[out_base][c] = du_l;
                        out_du[out_base + 1][c] = du_r;
                    }
                    let pg_base = (quad_idx * num_tiles + tile) * V3_QUAD_PARAMS_VEC4_STRIDE;
                    write_v3_lane_param_grads(&mut out_pgrad, pg_base, c, &lp, &acc, is_phs);
                }
            }
        }
    }

    let tile_weight_grads = if active_tiles > 0 {
        let u_part = TileMixUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_active_tiles: active_tiles as u32,
            dir_mode: TILE_MIX_DIR_PARTIAL_WEIGHTS,
            num_stems: num_stems as u32,
            block_wave: BLOCK_WAVE_ALL,
            num_quads: num_quads as u32,
            _pad1: 0,
        };
        let part = run_tile_mix_fwd_adj_cpu(&TileMixInput {
            uniforms: u_part,
            block_inputs_or_d_loss_d_inputs: &out_du,
            tile_descriptors: input.tile_descriptors,
            feature_outputs: input.feature_outputs,
            tile_weight_grads: &[],
            d_loss_d_features: &[],
        })?;
        let red = run_tile_mix_fwd_adj_cpu(&TileMixInput {
            uniforms: TileMixUniforms {
                dir_mode: TILE_MIX_DIR_REDUCE_WEIGHTS,
                ..u_part
            },
            block_inputs_or_d_loss_d_inputs: &out_du,
            tile_descriptors: input.tile_descriptors,
            feature_outputs: input.feature_outputs,
            tile_weight_grads: &part.tile_weight_grads,
            d_loss_d_features: &[],
        })?;
        let need_tw = active_tiles * crate::gpu::abi::TILE_WEIGHT_GRAD_FLOATS;
        red.tile_weight_grads[..need_tw].to_vec()
    } else {
        Vec::new()
    };

    Ok(WaveCoreAdjointOutput {
        d_loss_d_inputs: out_du,
        param_grads: out_pgrad,
        tile_weight_grads,
    })
}
