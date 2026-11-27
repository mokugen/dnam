//! CPU deterministic reference for `tile_mix_fwd_adj` (`dir_mode in 0..=3`) and wave-batched
//! forward cascade / reverse BPTT (`Waves 0..=3` / `Waves 3..=0`) matching `GpuTrainArena`
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::error::TrainError;
use crate::gpu::abi::{
    AdjointScanUniforms, BLOCK_WAVE_ALL, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE,
    FUSED_FLAG_FIXED_POINT_SCAN, FusedScanUniforms, QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE,
    TILE_DESC_VEC4_STRIDE, TILE_FRAMES, TILE_MIX_DIR_FORWARD, TILE_MIX_DIR_PARTIAL_WEIGHTS,
    TILE_MIX_DIR_PULLBACK, TILE_MIX_DIR_REDUCE_WEIGHTS, TILE_WEIGHT_GRAD_FLOATS, TileMixUniforms,
    block_quad_count, resolve_quad_idx,
};
use crate::runner::types::{
    AdjointScanInput, FusedCoreScanInput, TileMixInput, TileMixOutput, WaveCoreAdjointInput,
    WaveCoreAdjointOutput, WaveCoreCascadeInput, WaveCoreCascadeOutput,
};

use super::core_and_adjoint::{run_adjoint_scan_cpu, run_fused_core_scan_cpu};

#[inline]
fn flush_zero_vec4(v: [f32; 4]) -> [f32; 4] {
    v.map(|x| if x == 0.0 || !x.is_finite() { 0.0 } else { x })
}

#[inline]
fn flush_zero_f32(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() { 0.0 } else { x }
}

fn wave_quads(block_wave: u32, num_quads: usize) -> Vec<usize> {
    if block_wave == BLOCK_WAVE_ALL {
        (0..num_quads).collect()
    } else {
        let count = block_quad_count((block_wave & 3) as usize, num_quads);
        (0..count)
            .filter_map(|ql| {
                let q = resolve_quad_idx(block_wave, ql, num_quads);
                (q < num_quads).then_some(q)
            })
            .collect()
    }
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

fn tree_sum_f32(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut cur = vals.to_vec();
    while cur.len() > 1 {
        let mut next = Vec::with_capacity(cur.len().div_ceil(2));
        for pair in cur.chunks(2) {
            next.push(pair.iter().copied().sum());
        }
        cur = next;
    }
    cur[0]
}

fn run_forward_cpu(input: &TileMixInput<'_>) -> TileMixOutput {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_active = u.num_active_tiles as usize;
    let num_stems = u.num_stems as usize;
    let num_quads = u.num_quads as usize;
    let mut out_in = input.block_inputs_or_d_loss_d_inputs.to_vec();
    let dst_quads = wave_quads(u.block_wave, num_quads);

    for stem in 0..num_stems {
        for &dst_q in &dst_quads {
            for tile in 0..num_active {
                let d_base = tile * TILE_DESC_VEC4_STRIDE;
                let hdr = input.tile_descriptors[d_base];
                let src_q = hdr[0].to_bits() as usize;
                let desc_dst = hdr[1].to_bits() as usize;
                if desc_dst != dst_q || src_q >= num_quads {
                    continue;
                }
                let w_odd = load_mat4(input.tile_descriptors, d_base + 1);
                let w_even = load_mat4(input.tile_descriptors, d_base + 5);
                let w_lru = load_mat4(input.tile_descriptors, d_base + 9);
                for t in 0..num_frames {
                    let f_base =
                        ((stem * num_quads + src_q) * num_frames + t) * FEATURE_VEC4_STRIDE;
                    let i_base =
                        ((stem * num_quads + dst_q) * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                    let mut acc_l = out_in[i_base];
                    let mut acc_r = out_in[i_base + 1];
                    accum_col_major(&mut acc_l, &w_odd, input.feature_outputs[f_base + 2]);
                    accum_col_major(&mut acc_l, &w_even, input.feature_outputs[f_base + 4]);
                    accum_col_major(&mut acc_l, &w_lru, input.feature_outputs[f_base]);
                    accum_col_major(&mut acc_r, &w_odd, input.feature_outputs[f_base + 3]);
                    accum_col_major(&mut acc_r, &w_even, input.feature_outputs[f_base + 5]);
                    accum_col_major(&mut acc_r, &w_lru, input.feature_outputs[f_base + 1]);
                    out_in[i_base] = flush_zero_vec4(acc_l);
                    out_in[i_base + 1] = flush_zero_vec4(acc_r);
                }
            }
        }
    }
    TileMixOutput {
        block_inputs_or_d_loss_d_inputs: out_in,
        tile_weight_grads: input.tile_weight_grads.to_vec(),
        d_loss_d_features: input.d_loss_d_features.to_vec(),
    }
}

fn run_pullback_cpu(input: &TileMixInput<'_>) -> TileMixOutput {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_active = u.num_active_tiles as usize;
    let num_stems = u.num_stems as usize;
    let num_quads = u.num_quads as usize;
    let mut out_dlf = input.d_loss_d_features.to_vec();
    let src_quads = wave_quads(u.block_wave, num_quads);

    for stem in 0..num_stems {
        for &src_q in &src_quads {
            for tile in 0..num_active {
                let d_base = tile * TILE_DESC_VEC4_STRIDE;
                let hdr = input.tile_descriptors[d_base];
                let desc_src = hdr[0].to_bits() as usize;
                let dst_q = hdr[1].to_bits() as usize;
                if desc_src != src_q || dst_q >= num_quads {
                    continue;
                }
                let w_odd = load_mat4(input.tile_descriptors, d_base + 1);
                let w_even = load_mat4(input.tile_descriptors, d_base + 5);
                let w_lru = load_mat4(input.tile_descriptors, d_base + 9);
                for t in 0..num_frames {
                    let f_base =
                        ((stem * num_quads + src_q) * num_frames + t) * FEATURE_VEC4_STRIDE;
                    let dl_base =
                        ((stem * num_quads + dst_q) * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                    let dl_l = input.block_inputs_or_d_loss_d_inputs[dl_base];
                    let dl_r = input.block_inputs_or_d_loss_d_inputs[dl_base + 1];
                    accum_transpose_dot(&mut out_dlf[f_base + 2], &w_odd, dl_l);
                    accum_transpose_dot(&mut out_dlf[f_base + 3], &w_odd, dl_r);
                    accum_transpose_dot(&mut out_dlf[f_base + 4], &w_even, dl_l);
                    accum_transpose_dot(&mut out_dlf[f_base + 5], &w_even, dl_r);
                    accum_transpose_dot(&mut out_dlf[f_base], &w_lru, dl_l);
                    accum_transpose_dot(&mut out_dlf[f_base + 1], &w_lru, dl_r);
                    for k in 0..FEATURE_VEC4_STRIDE {
                        out_dlf[f_base + k] = flush_zero_vec4(out_dlf[f_base + k]);
                    }
                }
            }
        }
    }
    TileMixOutput {
        block_inputs_or_d_loss_d_inputs: input.block_inputs_or_d_loss_d_inputs.to_vec(),
        tile_weight_grads: input.tile_weight_grads.to_vec(),
        d_loss_d_features: out_dlf,
    }
}

fn run_partial_weights_cpu(input: &TileMixInput<'_>) -> TileMixOutput {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_tiles = (u.num_tiles as usize).max(num_frames.div_ceil(TILE_FRAMES));
    let num_active = u.num_active_tiles as usize;
    let num_stems = u.num_stems as usize;
    let num_quads = u.num_quads as usize;
    let need_len = num_active * (1 + num_stems * num_tiles) * TILE_WEIGHT_GRAD_FLOATS;
    let mut out_tw = input.tile_weight_grads.to_vec();
    if out_tw.len() < need_len {
        out_tw.resize(need_len, 0.0);
    }
    let mut vals = vec![0.0_f32; TILE_FRAMES];

    for stem in 0..num_stems {
        for tile in 0..num_active {
            let hdr = input.tile_descriptors[tile * TILE_DESC_VEC4_STRIDE];
            let src_q = hdr[0].to_bits() as usize;
            let dst_q = hdr[1].to_bits() as usize;
            for ft in 0..num_tiles {
                let out_base = (num_active + ((stem * num_active + tile) * num_tiles + ft))
                    * TILE_WEIGHT_GRAD_FLOATS;
                if src_q >= num_quads || dst_q >= num_quads {
                    out_tw[out_base..out_base + TILE_WEIGHT_GRAD_FLOATS].fill(0.0);
                    continue;
                }
                let start = ft * TILE_FRAMES;
                let end = (start + TILE_FRAMES).min(num_frames);
                let len = end - start;
                for (mat_idx, (feat_off_l, feat_off_r, tw_off)) in
                    [(2_usize, 3_usize, 0_usize), (4, 5, 16), (0, 1, 32)]
                        .into_iter()
                        .enumerate()
                {
                    let _ = mat_idx;
                    for s in 0..4 {
                        for d in 0..4 {
                            for (i, t) in (start..end).enumerate() {
                                let fb = ((stem * num_quads + src_q) * num_frames + t)
                                    * FEATURE_VEC4_STRIDE;
                                let dlb = ((stem * num_quads + dst_q) * num_frames + t)
                                    * CORE_INPUT_VEC4_STRIDE;
                                let fl = input.feature_outputs[fb + feat_off_l][s];
                                let fr = input.feature_outputs[fb + feat_off_r][s];
                                let dl = input.block_inputs_or_d_loss_d_inputs[dlb][d];
                                let dr = input.block_inputs_or_d_loss_d_inputs[dlb + 1][d];
                                vals[i] = fl * dl + fr * dr;
                            }
                            out_tw[out_base + tw_off + s * 4 + d] = tree_sum_f32(&vals[..len]);
                        }
                    }
                }
            }
        }
    }

    TileMixOutput {
        block_inputs_or_d_loss_d_inputs: input.block_inputs_or_d_loss_d_inputs.to_vec(),
        tile_weight_grads: out_tw,
        d_loss_d_features: input.d_loss_d_features.to_vec(),
    }
}

fn run_reduce_weights_cpu(input: &TileMixInput<'_>) -> TileMixOutput {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_tiles = (u.num_tiles as usize).max(num_frames.div_ceil(TILE_FRAMES));
    let num_active = u.num_active_tiles as usize;
    let num_stems = u.num_stems as usize;
    let mut out_tw = input.tile_weight_grads.to_vec();
    let total_chunks = num_stems * num_tiles;
    let mut vals = Vec::with_capacity(total_chunks);
    for tile in 0..num_active {
        let out_base = tile * TILE_WEIGHT_GRAD_FLOATS;
        for w in 0..TILE_WEIGHT_GRAD_FLOATS {
            vals.clear();
            for stem in 0..num_stems {
                for ft in 0..num_tiles {
                    let src_base = (num_active + ((stem * num_active + tile) * num_tiles + ft))
                        * TILE_WEIGHT_GRAD_FLOATS;
                    vals.push(out_tw[src_base + w]);
                }
            }
            out_tw[out_base + w] = flush_zero_f32(tree_sum_f32(&vals));
        }
    }
    TileMixOutput {
        block_inputs_or_d_loss_d_inputs: input.block_inputs_or_d_loss_d_inputs.to_vec(),
        tile_weight_grads: out_tw,
        d_loss_d_features: input.d_loss_d_features.to_vec(),
    }
}

/// Deterministic CPU reference matching WGSL `tile_mix_fwd_adj` (`dir_mode in 0..=3`).
pub fn run_tile_mix_fwd_adj_cpu(input: &TileMixInput<'_>) -> Result<TileMixOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_active = u.num_active_tiles as usize;
    let num_stems = u.num_stems as usize;
    let num_quads = u.num_quads as usize;
    if num_frames == 0 || num_stems == 0 || num_quads == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "run_tile_mix_fwd_adj_cpu",
            expected: 1,
            actual: 0,
        });
    }
    let need_desc = num_active * TILE_DESC_VEC4_STRIDE;
    if input.tile_descriptors.len() < need_desc {
        return Err(TrainError::InvalidDimensions {
            context: "run_tile_mix_fwd_adj_cpu(tile_descriptors)",
            expected: need_desc,
            actual: input.tile_descriptors.len(),
        });
    }
    match u.dir_mode {
        TILE_MIX_DIR_FORWARD => Ok(run_forward_cpu(input)),
        TILE_MIX_DIR_PULLBACK => Ok(run_pullback_cpu(input)),
        TILE_MIX_DIR_PARTIAL_WEIGHTS => Ok(run_partial_weights_cpu(input)),
        TILE_MIX_DIR_REDUCE_WEIGHTS => Ok(run_reduce_weights_cpu(input)),
        _ => Err(TrainError::KernelMismatch {
            context: "run_tile_mix_fwd_adj_cpu",
            detail: "dir_mode must be in 0..=3",
        }),
    }
}

/// Deterministic CPU reference matching [`WaveCoreCascadeInput`] (`Waves 0..=3` + `tile_mix(0)`).
pub fn run_wave_core_cascade_cpu(
    input: &WaveCoreCascadeInput<'_>,
) -> Result<WaveCoreCascadeOutput, TrainError> {
    let num_frames = input.num_frames;
    let num_quads = input.num_quads;
    let num_stems = input.num_stems;
    let active_tiles = input.active_tiles;
    if num_frames == 0 || num_stems == 0 || !(1..=10).contains(&num_quads) {
        return Err(TrainError::InvalidDimensions {
            context: "run_wave_core_cascade_cpu",
            expected: 1,
            actual: 0,
        });
    }
    let total_q = num_stems * num_quads;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let mut core_inputs =
        input.base_core_inputs[..total_q * num_frames * CORE_INPUT_VEC4_STRIDE].to_vec();
    let mut feature_outputs = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];
    let mut tile_checkpoints = if input.write_checkpoints {
        vec![[0.0_f32; 4]; total_q * num_tiles * TILE_CKPT_VEC4_STRIDE]
    } else {
        Vec::new()
    };
    let fp_flag = if input.fixed_point_scan {
        FUSED_FLAG_FIXED_POINT_SCAN
    } else {
        0
    };

    for block_idx in 0..4_u32 {
        let wg_x = block_quad_count(block_idx as usize, num_quads);
        if wg_x == 0 {
            continue;
        }
        if block_idx > 0 && active_tiles > 0 {
            let mixed = run_tile_mix_fwd_adj_cpu(&TileMixInput {
                uniforms: TileMixUniforms {
                    num_frames: num_frames as u32,
                    num_tiles: num_tiles as u32,
                    num_active_tiles: active_tiles as u32,
                    dir_mode: TILE_MIX_DIR_FORWARD,
                    num_stems: num_stems as u32,
                    block_wave: block_idx,
                    num_quads: num_quads as u32,
                    _pad1: 0,
                },
                block_inputs_or_d_loss_d_inputs: &core_inputs,
                tile_descriptors: input.tile_descriptors,
                feature_outputs: &feature_outputs,
                tile_weight_grads: &[],
                d_loss_d_features: &[],
            })?;
            core_inputs = mixed.block_inputs_or_d_loss_d_inputs;
        }
        let scan_out = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: num_quads as u32,
                num_stems: num_stems as u32,
                tile_start: 0,
                soft_rect_eps: input.soft_rect_eps,
                tiny_crossing_eps: input.tiny_crossing_eps,
                block_idx: block_idx | fp_flag,
            },
            core_inputs: &core_inputs,
            quad_params: input.quad_params,
            initial_checkpoints: None,
        })?;
        for stem in 0..num_stems {
            for &q in &wave_quads(block_idx, num_quads) {
                let qt = stem * num_quads + q;
                let fs = qt * num_frames * FEATURE_VEC4_STRIDE;
                let fc = num_frames * FEATURE_VEC4_STRIDE;
                feature_outputs[fs..fs + fc]
                    .copy_from_slice(&scan_out.feature_outputs[fs..fs + fc]);
                if input.write_checkpoints {
                    let cs = qt * num_tiles * TILE_CKPT_VEC4_STRIDE;
                    let cc = num_tiles * TILE_CKPT_VEC4_STRIDE;
                    tile_checkpoints[cs..cs + cc]
                        .copy_from_slice(&scan_out.tile_checkpoints[cs..cs + cc]);
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

/// Deterministic CPU reference matching [`WaveCoreAdjointInput`] (`Waves 3..=0` + `tile_mix(1,2,3)`).
pub fn run_wave_core_adjoint_cpu(
    input: &WaveCoreAdjointInput<'_>,
) -> Result<WaveCoreAdjointOutput, TrainError> {
    let num_frames = input.num_frames;
    let num_quads = input.num_quads;
    let num_stems = input.num_stems;
    let active_tiles = input.active_tiles;
    if num_frames == 0 || num_stems == 0 || !(1..=10).contains(&num_quads) {
        return Err(TrainError::InvalidDimensions {
            context: "run_wave_core_adjoint_cpu",
            expected: 1,
            actual: 0,
        });
    }
    let total_q = num_stems * num_quads;
    let num_tiles = num_frames.div_ceil(TILE_FRAMES);
    let mut cur_dlf =
        input.d_loss_d_features[..total_q * num_frames * FEATURE_VEC4_STRIDE].to_vec();
    let mut out_du = vec![[0.0_f32; 4]; total_q * num_frames * CORE_INPUT_VEC4_STRIDE];
    let mut out_pgrad = vec![[0.0_f32; 4]; total_q * num_tiles * QUAD_PARAMS_VEC4_STRIDE];

    for block_idx in (0..4_u32).rev() {
        let wg_x = block_quad_count(block_idx as usize, num_quads);
        if wg_x == 0 {
            continue;
        }
        if block_idx < 3 && active_tiles > 0 {
            let pb = run_tile_mix_fwd_adj_cpu(&TileMixInput {
                uniforms: TileMixUniforms {
                    num_frames: num_frames as u32,
                    num_tiles: num_tiles as u32,
                    num_active_tiles: active_tiles as u32,
                    dir_mode: TILE_MIX_DIR_PULLBACK,
                    num_stems: num_stems as u32,
                    block_wave: block_idx,
                    num_quads: num_quads as u32,
                    _pad1: 0,
                },
                block_inputs_or_d_loss_d_inputs: &out_du,
                tile_descriptors: input.tile_descriptors,
                feature_outputs: input.feature_outputs,
                tile_weight_grads: &[],
                d_loss_d_features: &cur_dlf,
            })?;
            cur_dlf = pb.d_loss_d_features;
        }
        let adj_out = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: num_quads as u32,
                num_stems: num_stems as u32,
                pass_mode: 2,
                soft_rect_eps: input.soft_rect_eps,
                tiny_crossing_eps: input.tiny_crossing_eps,
                block_idx,
            },
            core_inputs: input.core_inputs,
            quad_params: input.quad_params,
            tile_checkpoints: input.tile_checkpoints,
            d_loss_d_features: &cur_dlf,
            feature_outputs: input.feature_outputs,
            exit_adj_boundary: None,
        })?;
        for stem in 0..num_stems {
            for &q in &wave_quads(block_idx, num_quads) {
                let qt = stem * num_quads + q;
                let ds = qt * num_frames * CORE_INPUT_VEC4_STRIDE;
                let dc = num_frames * CORE_INPUT_VEC4_STRIDE;
                out_du[ds..ds + dc].copy_from_slice(&adj_out.d_loss_d_inputs[ds..ds + dc]);
                let ps = qt * num_tiles * QUAD_PARAMS_VEC4_STRIDE;
                let pc = num_tiles * QUAD_PARAMS_VEC4_STRIDE;
                out_pgrad[ps..ps + pc].copy_from_slice(&adj_out.param_grads[ps..ps + pc]);
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
            d_loss_d_features: &cur_dlf,
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
            d_loss_d_features: &cur_dlf,
        })?;
        red.tile_weight_grads[..active_tiles * TILE_WEIGHT_GRAD_FLOATS].to_vec()
    } else {
        Vec::new()
    };

    Ok(WaveCoreAdjointOutput {
        d_loss_d_inputs: out_du,
        param_grads: out_pgrad,
        tile_weight_grads,
    })
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        CoreQuadV4ParamsStereo4, GpuVec4, KernelVec4, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4,
        SonaToneFolded3, SonaToneFoldedBlock0, SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4,
        prewarp_tan_pi_f_over_fs_f32,
    };

    use super::*;
    use crate::runner::types::{pack_core_quad_params, pack_tile_descriptor};

    fn test_quad_params(idx: usize) -> [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE] {
        let g = GpuVec4::splat(prewarp_tan_pi_f_over_fs_f32(0.05 + 0.015 * (idx as f32)));
        let k = GpuVec4::splat(0.75);
        let weights = SonaToneWeights4 {
            lp: GpuVec4::splat(0.45),
            bp: GpuVec4::splat(0.20),
            hp: GpuVec4::splat(-0.08),
            ap: GpuVec4::splat(0.04),
            dir: GpuVec4::splat(0.25),
        };
        let is_block0 = matches!(idx, 0 | 4 | 8);
        let tone = if is_block0 {
            SonaToneFoldedBlock0::from_weights(&weights)
        } else {
            SonaToneFoldedBlock0::from_folded3(&SonaToneFolded3::from_weights(&weights, k))
        };
        pack_core_quad_params(&CoreQuadV4ParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone,
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(0.82),
                GpuVec4::splat(0.32),
                GpuVec4::splat(0.24),
                GpuVec4::splat(0.18),
                GpuVec4::splat(0.42),
                GpuVec4::splat(0.15),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::splat(if is_block0 { 0.2 } else { 0.0 }),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            g0: GpuVec4::splat(0.95),
            g_inf: GpuVec4::splat(1.15),
            d_pre: GpuVec4::splat(0.18),
            se_w_sq: GpuVec4::splat(0.35),
            se_d_bus: GpuVec4::splat(0.22),
            adaa_drive: GpuVec4::splat(1.35),
            lru_bias_drive: GpuVec4::splat(0.08),
            lin_mix: GpuVec4::splat(0.06),
            wavefolder_mix: GpuVec4::splat(0.22),
            glu_mix: GpuVec4::splat(0.30),
            out_gain: GpuVec4::splat(1.05),
        })
    }

    #[test]
    fn test_cpu_wave_cascade_and_adjoint_fd_parity() {
        let num_frames = 300_usize;
        let num_quads = 10_usize;
        let num_stems = 1_usize;
        let mut base_in = vec![[0.0_f32; 4]; num_quads * num_frames * CORE_INPUT_VEC4_STRIDE];
        for q in 0..num_quads {
            for t in 0..num_frames {
                let p = (t as f32) * 0.055 + (q as f32) * 0.27;
                let b = (q * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                base_in[b] = [0.25 * p.sin(), 0.2 * p.cos(), 0.15 * (0.7 * p).sin(), 0.1];
                base_in[b + 1] = [0.2 * p.cos(), -0.18 * p.sin(), 0.12 * p.sin(), -0.09];
            }
        }
        let params: Vec<[f32; 4]> = (0..num_quads).flat_map(test_quad_params).collect();
        let w = [[0.08_f32, -0.02, 0.01, 0.03]; 4];
        let desc = [
            pack_tile_descriptor(0, 1, 0, 1, &w, &w, &w),
            pack_tile_descriptor(4, 5, 0, 1, &w, &w, &w),
            pack_tile_descriptor(1, 2, 1, 2, &w, &w, &w),
            pack_tile_descriptor(2, 3, 2, 3, &w, &w, &w),
        ]
        .concat();

        let fwd = run_wave_core_cascade_cpu(&WaveCoreCascadeInput {
            num_frames,
            num_quads,
            num_stems,
            active_tiles: 4,
            write_checkpoints: true,
            fixed_point_scan: false,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            base_core_inputs: &base_in,
            quad_params: &params,
            tile_descriptors: &desc,
        })
        .unwrap();

        let dl_feat = vec![[0.15_f32, -0.1, 0.08, -0.05]; fwd.feature_outputs.len()];
        let adj = run_wave_core_adjoint_cpu(&WaveCoreAdjointInput {
            num_frames,
            num_quads,
            num_stems,
            active_tiles: 4,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            core_inputs: &fwd.core_inputs,
            quad_params: &params,
            tile_checkpoints: &fwd.tile_checkpoints,
            d_loss_d_features: &dl_feat,
            feature_outputs: &fwd.feature_outputs,
            tile_descriptors: &desc,
        })
        .unwrap();
        assert_eq!(adj.tile_weight_grads.len(), 4 * TILE_WEIGHT_GRAD_FLOATS);

        // Central FD check on tile 0 weight w_odd[0][0] through full 4-wave causal cascade.
        let eps = 2.0e-3_f32;
        let eval_w = |delta: f32| -> f64 {
            let mut wm = w;
            wm[0][0] += delta;
            let d = [
                pack_tile_descriptor(0, 1, 0, 1, &wm, &w, &w),
                pack_tile_descriptor(4, 5, 0, 1, &w, &w, &w),
                pack_tile_descriptor(1, 2, 1, 2, &w, &w, &w),
                pack_tile_descriptor(2, 3, 2, 3, &w, &w, &w),
            ]
            .concat();
            let o = run_wave_core_cascade_cpu(&WaveCoreCascadeInput {
                num_frames,
                num_quads,
                num_stems,
                active_tiles: 4,
                write_checkpoints: false,
                fixed_point_scan: false,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                base_core_inputs: &base_in,
                quad_params: &params,
                tile_descriptors: &d,
            })
            .unwrap();
            o.feature_outputs
                .iter()
                .zip(dl_feat.iter())
                .map(|(a, b)| {
                    (0..4)
                        .map(|k| f64::from(a[k]) * f64::from(b[k]))
                        .sum::<f64>()
                })
                .sum()
        };
        let fd = (eval_w(eps) - eval_w(-eps)) / f64::from(2.0 * eps);
        let got = f64::from(adj.tile_weight_grads[0]);
        assert!(
            (got - fd).abs() <= 5.0e-3 * fd.abs().max(1.0),
            "wave cascade/adjoint tile weight FD mismatch: got={got} fd={fd}"
        );
    }
}
