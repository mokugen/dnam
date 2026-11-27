//! Deterministic CPU reverse BPTT runner for `SONA v4.0` `adjoint_scan_256` across
//! `12 × Vec4` tile checkpoints, `24 × Vec4` cross-chunk `EXIT_ADJ` boundaries, `28 × Vec4`
//! `param_grads`, Block `0` Pred-Corr PHS, Blocks `1..3` 3-tap folded SVF, and `SE-Bus`.

use dnam_kernels::{KernelVec4, SOFT_RECT_EPS, StereoVec8, TINY_CROSSING_EPS};

use super::adjoint_math::{
    LaneForwardFrame, QuadLaneParams, ReverseCarry, ReverseStepContext, TileGradAccum, flush_zero,
    needs_b_staging, stage_b_all_for_tile, step_lane_forward_v4, step_lane_reverse_v4,
};
use super::{read_checkpoint, unpack_all_quads, write_lane_param_grads};
use crate::error::TrainError;
use crate::gpu::abi::{
    BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE, EXIT_ADJ_BOUNDARY_VEC4_STRIDE,
    FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES,
    block_quad_count, resolve_quad_idx,
};
use crate::runner::types::{AdjointScanInput, AdjointScanOutput};

/// Deterministic CPU reference implementation of `SONA v4.0` `adjoint_scan_256`.
pub fn run_adjoint_scan_cpu(input: &AdjointScanInput<'_>) -> Result<AdjointScanOutput, TrainError> {
    let u = &input.uniforms;
    let is_phs = u.block_idx != BLOCK_IDX_DIRECT_LINEAR && (u.block_idx & 3) == 0;
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
    let wg_x = if u.block_idx == BLOCK_IDX_DIRECT_LINEAR {
        num_quads
    } else {
        block_quad_count((u.block_idx & 3) as usize, num_quads)
    };
    let exit_vecs = num_stems * wg_x * EXIT_ADJ_BOUNDARY_VEC4_STRIDE;
    let need_in = total_q * num_frames * CORE_INPUT_VEC4_STRIDE;
    let need_feat = total_q * num_frames * FEATURE_VEC4_STRIDE;
    let need_ckpt = total_q * num_tiles * TILE_CKPT_VEC4_STRIDE;
    if input.core_inputs.len() < need_in
        || input.d_loss_d_features.len() < need_feat
        || input.feature_outputs.len() < need_feat
        || input.tile_checkpoints.len() < need_ckpt
    {
        return Err(TrainError::InvalidDimensions {
            context: "adjoint_scan_256(buffers)",
            expected: need_feat,
            actual: input
                .feature_outputs
                .len()
                .min(input.d_loss_d_features.len()),
        });
    }
    let mut entry_adj_boundary = vec![[0.0_f32; 4]; exit_vecs];
    if let Some(seed) = input.exit_adj_boundary {
        if seed.len() < exit_vecs {
            return Err(TrainError::InvalidDimensions {
                context: "adjoint_scan_256(exit_adj_boundary)",
                expected: exit_vecs,
                actual: seed.len(),
            });
        }
        entry_adj_boundary.copy_from_slice(&seed[..exit_vecs]);
    }
    let quads = unpack_all_quads(input.quad_params, total_q)?;
    let mut d_loss_d_inputs = vec![[0.0_f32; 4]; need_in];
    let mut param_grads = vec![[0.0_f32; 4]; total_q * QUAD_PARAMS_VEC4_STRIDE * num_tiles];
    let tiny_eps = if u.tiny_crossing_eps > 0.0 {
        u.tiny_crossing_eps
    } else {
        TINY_CROSSING_EPS
    };

    for stem in 0..num_stems {
        for q_local in 0..wg_x {
            let quad_idx = resolve_quad_idx(u.block_idx, q_local, num_quads);
            if quad_idx >= num_quads {
                continue;
            }
            let q_global = stem * num_quads + quad_idx;
            let exit_base = (stem * wg_x + q_local) * EXIT_ADJ_BOUNDARY_VEC4_STRIDE;
            let q_params = &quads[q_global];
            let stage_b = needs_b_staging(q_params);
            let lps: [QuadLaneParams; 4] =
                core::array::from_fn(|c| QuadLaneParams::extract(q_params, c));
            let mut carries: [ReverseCarry; 4] = core::array::from_fn(|c| {
                let s_exit = entry_adj_boundary[exit_base + c];
                let s_lin_exit = if is_phs {
                    entry_adj_boundary[exit_base + 8 + c]
                } else {
                    [0.0; 4]
                };
                ReverseCarry {
                    adj_r_next_l: entry_adj_boundary[exit_base + 6][c],
                    adj_r_next_r: entry_adj_boundary[exit_base + 7][c],
                    lam_after_l: entry_adj_boundary[exit_base + 4][c],
                    lam_after_r: entry_adj_boundary[exit_base + 5][c],
                    adj_s_out_l: [s_exit[0], s_exit[1]],
                    adj_s_out_r: [s_exit[2], s_exit[3]],
                    adj_s_lin_out_l: [s_lin_exit[0], s_lin_exit[1]],
                    adj_s_lin_out_r: [s_lin_exit[2], s_lin_exit[3]],
                }
            });

            for tile in (0..num_tiles).rev() {
                let tile_base = tile * TILE_FRAMES;
                let valid_len = num_frames.saturating_sub(tile_base).min(TILE_FRAMES);
                if valid_len == 0 {
                    continue;
                }
                let ckpt = read_checkpoint(input.tile_checkpoints, q_global, num_tiles, tile);
                let (s1l, s1r) = ckpt.block.svf_s1.into_lr();
                let (s2l, s2r) = ckpt.block.svf_s2.into_lr();
                let (vl, vr) = ckpt.block.lru_v.into_lr();
                let (apl, apr) = ckpt.block.adaa_prev.into_lr();
                let (s1ll, s1lr) = ckpt.svf_lin_s1.into_lr();
                let (s2ll, s2lr) = ckpt.svf_lin_s2.into_lr();

                let lane_frames: [Vec<LaneForwardFrame>; 4] = core::array::from_fn(|c| {
                    let lp = &lps[c];
                    let svf_c = lp.svf_coeffs();
                    let lru_c = lp.lru_coeffs();
                    let mut lane_state = [
                        s1l.to_array()[c],
                        s2l.to_array()[c],
                        s1r.to_array()[c],
                        s2r.to_array()[c],
                        vl.to_array()[c],
                        vr.to_array()[c],
                        apl.to_array()[c],
                        apr.to_array()[c],
                        s1ll.to_array()[c],
                        s2ll.to_array()[c],
                        s1lr.to_array()[c],
                        s2lr.to_array()[c],
                    ];
                    (0..valid_len)
                        .map(|local_t| {
                            let frame = tile_base + local_t;
                            let in_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
                            let u_l = flush_zero(input.core_inputs[in_base][c]);
                            let u_r = flush_zero(input.core_inputs[in_base + 1][c]);
                            step_lane_forward_v4(
                                lp,
                                &svf_c,
                                &lru_c,
                                u_l,
                                u_r,
                                &mut lane_state,
                                is_phs,
                            )
                        })
                        .collect()
                });

                let staged_tile: Option<Vec<([f32; 4], [f32; 4])>> = if stage_b {
                    let tile_feats: Vec<Vec<[f32; 4]>> = (0..valid_len)
                        .map(|local_t| {
                            let fb =
                                (q_global * num_frames + tile_base + local_t) * FEATURE_VEC4_STRIDE;
                            vec![input.feature_outputs[fb], input.feature_outputs[fb + 1]]
                        })
                        .collect();
                    Some(stage_b_all_for_tile(
                        q_params,
                        &lane_frames,
                        &tile_feats,
                        tiny_eps,
                    ))
                } else {
                    None
                };

                for c in 0..4 {
                    let ctx = ReverseStepContext {
                        d_loss_d_features: input.d_loss_d_features,
                        feature_outputs: input.feature_outputs,
                        params: q_params,
                        lp: &lps[c],
                        q_global,
                        num_frames,
                        c,
                        tiny_eps,
                        is_phs,
                    };
                    let mut acc = TileGradAccum::default();
                    for local_t in (0..valid_len).rev() {
                        let frame = tile_base + local_t;
                        let b_staged = staged_tile.as_ref().map(|s| s[local_t]);
                        let (du_l, du_r) = step_lane_reverse_v4(
                            &ctx,
                            &lane_frames[c][local_t],
                            frame,
                            b_staged,
                            &mut carries[c],
                            &mut acc,
                        );
                        let out_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
                        d_loss_d_inputs[out_base][c] = du_l;
                        d_loss_d_inputs[out_base + 1][c] = du_r;
                    }
                    let pg_base = (q_global * num_tiles + tile) * QUAD_PARAMS_VEC4_STRIDE;
                    write_lane_param_grads(&mut param_grads, pg_base, c, &lps[c], &acc, is_phs);
                }
            }

            for c in 0..4 {
                entry_adj_boundary[exit_base + 12 + c] = [
                    carries[c].adj_s_out_l[0],
                    carries[c].adj_s_out_l[1],
                    carries[c].adj_s_out_r[0],
                    carries[c].adj_s_out_r[1],
                ];
                entry_adj_boundary[exit_base + 16][c] = carries[c].lam_after_l;
                entry_adj_boundary[exit_base + 17][c] = carries[c].lam_after_r;
                entry_adj_boundary[exit_base + 18][c] = carries[c].adj_r_next_l;
                entry_adj_boundary[exit_base + 19][c] = carries[c].adj_r_next_r;
                entry_adj_boundary[exit_base + 20 + c] = if is_phs {
                    [
                        carries[c].adj_s_lin_out_l[0],
                        carries[c].adj_s_lin_out_l[1],
                        carries[c].adj_s_lin_out_r[0],
                        carries[c].adj_s_lin_out_r[1],
                    ]
                } else {
                    [0.0; 4]
                };
            }
        }
    }

    let _ = SOFT_RECT_EPS;
    Ok(AdjointScanOutput {
        d_loss_d_inputs,
        param_grads,
        entry_adj_boundary,
    })
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        CoreQuadV4ParamsStereo4, GpuVec4, KernelVec4, SonaParaLruDuhemCoeffs4,
        SonaToneFoldedBlock0, SonaToneWeights4, TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
    };

    use super::*;
    use crate::cpu::core_and_adjoint::run_fused_core_scan_cpu;
    use crate::gpu::abi::{AdjointScanUniforms, FusedScanUniforms};
    use crate::runner::types::{FusedCoreScanInput, pack_core_quad_params};

    #[test]
    fn test_v4_multi_chunk_24_vec4_exit_boundary_matches_resident_bptt() {
        let g = GpuVec4::splat(prewarp_tan_pi_f_over_fs_f32(0.09));
        let k = GpuVec4::splat(0.65);
        let params = CoreQuadV4ParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneFoldedBlock0::from_weights(&SonaToneWeights4 {
                lp: GpuVec4::splat(0.42),
                bp: GpuVec4::splat(0.28),
                hp: GpuVec4::splat(-0.12),
                ap: GpuVec4::splat(0.08),
                dir: GpuVec4::splat(0.22),
            }),
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(0.95),
                GpuVec4::splat(0.35),
                GpuVec4::splat(0.30),
                GpuVec4::splat(0.22),
                GpuVec4::splat(0.45),
                GpuVec4::splat(0.20),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::splat(1.4),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            g0: GpuVec4::splat(0.92),
            g_inf: GpuVec4::splat(1.22),
            d_pre: GpuVec4::splat(0.35),
            se_w_sq: GpuVec4::splat(0.45),
            se_d_bus: GpuVec4::splat(0.30),
            adaa_drive: GpuVec4::splat(1.8),
            lru_bias_drive: GpuVec4::splat(0.18),
            lin_mix: GpuVec4::splat(0.08),
            wavefolder_mix: GpuVec4::splat(0.30),
            glu_mix: GpuVec4::splat(0.40),
            out_gain: GpuVec4::splat(1.0),
        };
        let packed = pack_core_quad_params(&params);
        let n = 2 * TILE_FRAMES;
        let mut core_in = vec![[0.0_f32; 4]; n * 2];
        for t in 0..n {
            let tf = t as f32;
            core_in[t * 2] = [
                0.45 * (0.09 * tf).sin(),
                0.35 * (0.13 * tf).cos(),
                0.25,
                -0.3,
            ];
            core_in[t * 2 + 1] = [
                -0.35 * (0.07 * tf).cos(),
                0.4 * (0.11 * tf).sin(),
                -0.2,
                0.3,
            ];
        }
        let fwd = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: n as u32,
                num_tiles: 2,
                num_quads: 1,
                num_stems: 1,
                tile_start: 0,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_in,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        let dl_feat = vec![[0.12_f32, -0.09, 0.14, -0.07]; n * FEATURE_VEC4_STRIDE];
        let adj_full = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: n as u32,
                num_tiles: 2,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_in,
            quad_params: &packed,
            tile_checkpoints: &fwd.tile_checkpoints,
            d_loss_d_features: &dl_feat,
            feature_outputs: &fwd.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();

        // Chunk 1 -> Chunk 0 using 24-Vec4 exit boundary (including adj_s_lin_out at 20..24 -> 8..12)
        let adj_c1 = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: TILE_FRAMES as u32,
                num_tiles: 1,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_in[TILE_FRAMES * 2..],
            quad_params: &packed,
            tile_checkpoints: &fwd.tile_checkpoints[TILE_CKPT_VEC4_STRIDE..],
            d_loss_d_features: &dl_feat[TILE_FRAMES * FEATURE_VEC4_STRIDE..],
            feature_outputs: &fwd.feature_outputs[TILE_FRAMES * FEATURE_VEC4_STRIDE..],
            exit_adj_boundary: None,
        })
        .unwrap();
        let mut seed_c0 = vec![[0.0_f32; 4]; EXIT_ADJ_BOUNDARY_VEC4_STRIDE];
        seed_c0[..12].copy_from_slice(&adj_c1.entry_adj_boundary[12..24]);
        let adj_c0 = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: TILE_FRAMES as u32,
                num_tiles: 1,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_in[..TILE_FRAMES * 2],
            quad_params: &packed,
            tile_checkpoints: &fwd.tile_checkpoints[..TILE_CKPT_VEC4_STRIDE],
            d_loss_d_features: &dl_feat[..TILE_FRAMES * FEATURE_VEC4_STRIDE],
            feature_outputs: &fwd.feature_outputs[..TILE_FRAMES * FEATURE_VEC4_STRIDE],
            exit_adj_boundary: Some(&seed_c0),
        })
        .unwrap();

        let max_du_diff = adj_full.d_loss_d_inputs[..TILE_FRAMES * 2]
            .iter()
            .zip(adj_c0.d_loss_d_inputs.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0_f32, f32::max);
        assert!(max_du_diff <= 1.0e-6, "24-Vec4 exit diff: {max_du_diff}");
    }
}
