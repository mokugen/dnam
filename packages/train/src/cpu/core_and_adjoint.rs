//! Deterministic CPU SIMD reference runners for `fused_core_scan_256` (including `Streamed`
//! multi-block checkpoint chaining) and 3-pass-equivalent reverse BPTT `adjoint_scan_256`
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`, `ref[train.pipeline.isomorphism-parity]`).

mod adjoint_math;
mod streamed;

pub use streamed::run_fused_core_scan_planned_cpu;

pub(crate) use adjoint_math::{
    FwdFrameState, LaneParams, ReverseCarry, ReverseStepContext, TileGradAccum, flush_zero,
    step_lane_forward, step_lane_reverse,
};
use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4, MIN_TPT_G, SOFT_RECT_EPS,
    SonaBlockStateStereo8, StereoVec8, TINY_CROSSING_EPS, canonicalize_zero_f32_bits,
    step_core_quad_phs_stereo8, step_core_quad_stereo8,
};

use crate::error::TrainError;
use crate::gpu::abi::{
    BLOCK_IDX_DIRECT_LINEAR, CORE_INPUT_VEC4_STRIDE, EXIT_ADJ_BOUNDARY_VEC4_STRIDE,
    FEATURE_VEC4_STRIDE, FUSED_FLAG_FIXED_POINT_SCAN, QUAD_PARAMS_VEC4_STRIDE,
    TILE_CKPT_VEC4_STRIDE, TILE_FRAMES, block_quad_count, quad_param_slots, resolve_quad_idx,
};
use crate::runner::types::{
    AdjointScanInput, AdjointScanOutput, FusedCoreScanInput, FusedCoreScanOutput,
    unpack_core_quad_params,
};

#[inline(always)]
pub(crate) fn canon_vec4(v: GpuVec4) -> [f32; 4] {
    v.to_array().map(canonicalize_zero_f32_bits)
}

pub(crate) fn read_checkpoint(
    ckpts: &[[f32; 4]],
    q_global: usize,
    num_tiles: usize,
    tile: usize,
) -> SonaBlockStateStereo8<InterleavedStereo8<GpuVec4>> {
    let base = (q_global * num_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    SonaBlockStateStereo8 {
        svf_s1: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base]),
            GpuVec4::from_array(ckpts[base + 1]),
        ),
        svf_s2: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 2]),
            GpuVec4::from_array(ckpts[base + 3]),
        ),
        lru_v: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 4]),
            GpuVec4::from_array(ckpts[base + 5]),
        ),
        adaa_prev: InterleavedStereo8::from_lr(
            GpuVec4::from_array(ckpts[base + 6]),
            GpuVec4::from_array(ckpts[base + 7]),
        ),
    }
}

pub(crate) fn write_checkpoint(
    ckpts: &mut [[f32; 4]],
    q_global: usize,
    num_tiles: usize,
    tile: usize,
    state: &SonaBlockStateStereo8<InterleavedStereo8<GpuVec4>>,
) {
    let base = (q_global * num_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
    let (s1_l, s1_r) = state.svf_s1.into_lr();
    let (s2_l, s2_r) = state.svf_s2.into_lr();
    let (v_l, v_r) = state.lru_v.into_lr();
    let (ap_l, ap_r) = state.adaa_prev.into_lr();
    ckpts[base] = canon_vec4(s1_l);
    ckpts[base + 1] = canon_vec4(s1_r);
    ckpts[base + 2] = canon_vec4(s2_l);
    ckpts[base + 3] = canon_vec4(s2_r);
    ckpts[base + 4] = canon_vec4(v_l);
    ckpts[base + 5] = canon_vec4(v_r);
    ckpts[base + 6] = canon_vec4(ap_l);
    ckpts[base + 7] = canon_vec4(ap_r);
}

fn unpack_all_quads(
    quad_params: &[[f32; 4]],
    total_q: usize,
) -> Result<Vec<CoreQuadParamsStereo4<GpuVec4>>, TrainError> {
    let needed = total_q * QUAD_PARAMS_VEC4_STRIDE;
    if quad_params.len() < needed {
        return Err(TrainError::InvalidDimensions {
            context: "quad_params",
            expected: needed,
            actual: quad_params.len(),
        });
    }
    let mut out = Vec::with_capacity(total_q);
    for q in 0..total_q {
        out.push(unpack_core_quad_params(
            &quad_params[q * QUAD_PARAMS_VEC4_STRIDE..(q + 1) * QUAD_PARAMS_VEC4_STRIDE],
        )?);
    }
    Ok(out)
}

/// Deterministic CPU reference implementation of `fused_core_scan_256` with `tile_start > 0`
/// resume support and exact `+0.0` Zero-DC canonicalization.
pub fn run_fused_core_scan_cpu(
    input: &FusedCoreScanInput<'_>,
) -> Result<FusedCoreScanOutput, TrainError> {
    let u = &input.uniforms;
    let num_frames = u.num_frames as usize;
    let num_tiles = (u.num_tiles as usize).max(1);
    let num_quads = u.num_quads as usize;
    let num_stems = u.num_stems as usize;
    let total_q = num_quads * num_stems;
    if num_frames == 0 || num_quads == 0 || num_stems == 0 {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256",
            expected: 1,
            actual: 0,
        });
    }
    let needed_in = total_q * num_frames * CORE_INPUT_VEC4_STRIDE;
    if input.core_inputs.len() < needed_in {
        return Err(TrainError::InvalidDimensions {
            context: "fused_core_scan_256(core_inputs)",
            expected: needed_in,
            actual: input.core_inputs.len(),
        });
    }
    let params = unpack_all_quads(input.quad_params, total_q)?;
    let mut tile_checkpoints = vec![[0.0_f32; 4]; total_q * num_tiles * TILE_CKPT_VEC4_STRIDE];
    if let Some(seed) = input.initial_checkpoints {
        let n = seed.len().min(tile_checkpoints.len());
        tile_checkpoints[..n].copy_from_slice(&seed[..n]);
    }
    let mut feature_outputs = vec![[0.0_f32; 4]; total_q * num_frames * FEATURE_VEC4_STRIDE];

    let max_tiles = num_frames.div_ceil(TILE_FRAMES).min(num_tiles);
    let tile_start = (u.tile_start as usize).min(max_tiles);
    let resuming = tile_start > 0 && tile_start < num_tiles;
    let is_direct = u.block_idx == BLOCK_IDX_DIRECT_LINEAR;
    let raw_block = if is_direct {
        BLOCK_IDX_DIRECT_LINEAR
    } else {
        u.block_idx & !FUSED_FLAG_FIXED_POINT_SCAN
    };
    let is_phs = !is_direct && (raw_block & 3) == 0;
    let wg_x = if is_direct {
        num_quads
    } else {
        block_quad_count((raw_block & 3) as usize, num_quads)
    };

    for stem in 0..num_stems {
        for q_local in 0..wg_x {
            let quad_idx = resolve_quad_idx(raw_block, q_local, num_quads);
            if quad_idx >= num_quads {
                continue;
            }
            let q_global = stem * num_quads + quad_idx;
            let mut state = if resuming {
                read_checkpoint(&tile_checkpoints, q_global, num_tiles, tile_start)
            } else {
                SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero()
            };
            for tile in tile_start..max_tiles {
                if !(resuming && tile == tile_start) {
                    write_checkpoint(&mut tile_checkpoints, q_global, num_tiles, tile, &state);
                }
                let base_frame = tile * TILE_FRAMES;
                let end_frame = (base_frame + TILE_FRAMES).min(num_frames);
                for t in base_frame..end_frame {
                    let in_idx = (q_global * num_frames + t) * CORE_INPUT_VEC4_STRIDE;
                    let u_vec = InterleavedStereo8::from_lr(
                        GpuVec4::from_array(input.core_inputs[in_idx].map(flush_zero)),
                        GpuVec4::from_array(input.core_inputs[in_idx + 1].map(flush_zero)),
                    );
                    let out = if is_phs {
                        step_core_quad_phs_stereo8(u_vec, &mut state, &params[q_global])
                    } else {
                        step_core_quad_stereo8(u_vec, &mut state, &params[q_global])
                    };
                    let (vl, vr) = out.lru_state.into_lr();
                    let (ol, or) = out.adaa_odd.into_lr();
                    let (el, er) = out.adaa_even.into_lr();
                    let f_idx = (q_global * num_frames + t) * FEATURE_VEC4_STRIDE;
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
    Ok(FusedCoreScanOutput {
        tile_checkpoints,
        feature_outputs,
    })
}

/// Runs `fused_core_scan_256` under [`CoreScanPlan`], yielding `0-ULP` identical results between
pub(crate) fn write_lane_param_grads(
    out_grads: &mut [[f32; 4]],
    pg_base: usize,
    c: usize,
    lp: &LaneParams,
    acc: &TileGradAccum,
    is_phs: bool,
) {
    for slot in 0..QUAD_PARAMS_VEC4_STRIDE {
        out_grads[pg_base + slot][c] = acc.slots[slot];
    }
    let (g_svf_k, g_svf_k_nl, g_svf_g) = if is_phs {
        (
            if lp.k >= 0.0 {
                acc.slots[quad_param_slots::SVF_K]
            } else {
                0.0
            },
            if lp.svf_k_nl >= 0.0 {
                acc.slots[quad_param_slots::SVF_K_NL]
            } else {
                0.0
            },
            if lp.svf_g >= MIN_TPT_G {
                acc.slots[quad_param_slots::SVF_G] + 2.0 * acc.slots[quad_param_slots::TWO_SVF_G]
            } else {
                0.0
            },
        )
    } else {
        let r0 = [acc.slots[0], acc.slots[1], acc.slots[2]];
        let g_k = acc.slots[3] + 2.0 * acc.slots[4]
            - lp.a2 * (r0[0] * lp.a1 + r0[1] * lp.a2 + r0[2] * lp.a3);
        let g_g = if lp.svf_g >= MIN_TPT_G {
            let denom_g_term = 2.0 * lp.a2 + lp.k * lp.a1;
            let da1_dg = -lp.a1 * denom_g_term;
            let da2_dg = lp.a1 - lp.a2 * denom_g_term;
            let da3_dg = (lp.a2 + lp.a2) - lp.a3 * denom_g_term;
            r0[0] * da1_dg
                + r0[1] * da2_dg
                + r0[2] * da3_dg
                + 2.0 * acc.slots[quad_param_slots::TWO_SVF_G]
        } else {
            0.0
        };
        (g_k, 0.0, g_g)
    };
    out_grads[pg_base + quad_param_slots::SVF_K][c] = g_svf_k;
    out_grads[pg_base + quad_param_slots::SVF_G][c] = g_svf_g;
    out_grads[pg_base + quad_param_slots::SVF_K_NL][c] = g_svf_k_nl;
}

/// Deterministic CPU reference implementation of `adjoint_scan_256` across multi-tile
/// trajectories (`Block 0` PHS + `Blocks 1..3` linear `TPT-SVF`, `SVF_K_NL`, `SVF_K`,
/// `SVF_G >= MIN_TPT_G`, cross-lane GLU, and `PolyADAA1` tile seams).
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
            let q_packed_base = q_global * QUAD_PARAMS_VEC4_STRIDE;
            let svf_g_arr = input.quad_params[q_packed_base + quad_param_slots::SVF_G];
            let mut tile_fwd = vec![FwdFrameState::default(); TILE_FRAMES];

            for c in 0..4 {
                let lp = LaneParams::extract(q_params, svf_g_arr, c);
                let svf_c = lp.svf_coeffs();
                let lru_c = lp.lru_coeffs();
                let s_exit = entry_adj_boundary[exit_base + c];
                let mut carry = ReverseCarry {
                    adj_r_next_l: entry_adj_boundary[exit_base + 6][c],
                    adj_r_next_r: entry_adj_boundary[exit_base + 7][c],
                    lam_after_l: entry_adj_boundary[exit_base + 4][c],
                    lam_after_r: entry_adj_boundary[exit_base + 5][c],
                    adj_s_out_l: [s_exit[0], s_exit[1]],
                    adj_s_out_r: [s_exit[2], s_exit[3]],
                };

                for tile in (0..num_tiles).rev() {
                    let tile_base = tile * TILE_FRAMES;
                    let valid_len = num_frames.saturating_sub(tile_base).min(TILE_FRAMES);
                    if valid_len == 0 {
                        continue;
                    }
                    let ckpt_base = (q_global * num_tiles + tile) * TILE_CKPT_VEC4_STRIDE;
                    let mut lane_state = [
                        input.tile_checkpoints[ckpt_base][c],
                        input.tile_checkpoints[ckpt_base + 2][c],
                        input.tile_checkpoints[ckpt_base + 1][c],
                        input.tile_checkpoints[ckpt_base + 3][c],
                        input.tile_checkpoints[ckpt_base + 4][c],
                        input.tile_checkpoints[ckpt_base + 5][c],
                        input.tile_checkpoints[ckpt_base + 6][c],
                        input.tile_checkpoints[ckpt_base + 7][c],
                    ];
                    for local_t in 0..valid_len {
                        let frame = tile_base + local_t;
                        let in_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
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
                        d_loss_d_features: input.d_loss_d_features,
                        feature_outputs: input.feature_outputs,
                        q_global,
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
                        let out_base = (q_global * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
                        d_loss_d_inputs[out_base][c] = du_l;
                        d_loss_d_inputs[out_base + 1][c] = du_r;
                    }

                    let pg_base = (q_global * num_tiles + tile) * QUAD_PARAMS_VEC4_STRIDE;
                    write_lane_param_grads(&mut param_grads, pg_base, c, &lp, &acc, is_phs);
                }

                entry_adj_boundary[exit_base + 8 + c] = [
                    carry.adj_s_out_l[0],
                    carry.adj_s_out_l[1],
                    carry.adj_s_out_r[0],
                    carry.adj_s_out_r[1],
                ];
                entry_adj_boundary[exit_base + 12][c] = carry.lam_after_l;
                entry_adj_boundary[exit_base + 13][c] = carry.lam_after_r;
                entry_adj_boundary[exit_base + 14][c] = carry.adj_r_next_l;
                entry_adj_boundary[exit_base + 15][c] = carry.adj_r_next_r;
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
    use super::*;
    use crate::gpu::abi::FusedScanUniforms;
    use crate::planner::{
        CoreScanWorkload, ExecutionMode, MemoryBudgetProfile, plan_core_scan_execution,
    };
    use crate::runner::types::pack_core_quad_params;
    use dnam_kernels::{
        SonaParaLruDuhemCoeffs4, SonaToneWeights4, TptSvfCoeffs4, prewarp_tan_pi_f_over_fs_f32,
    };

    fn make_test_quad(
        svf_g_zero: bool,
    ) -> (
        CoreQuadParamsStereo4<GpuVec4>,
        [[f32; 4]; QUAD_PARAMS_VEC4_STRIDE],
    ) {
        let g = GpuVec4::from_array([0.04, 0.10, 0.20, 0.34].map(prewarp_tan_pi_f_over_fs_f32));
        let k = GpuVec4::from_array([0.65, 0.85, 1.10, 1.40]);
        let params = CoreQuadParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneWeights4 {
                lp: GpuVec4::from_array([0.45, 0.20, -0.15, 0.10]),
                bp: GpuVec4::from_array([0.25, 0.40, 0.30, -0.20]),
                hp: GpuVec4::from_array([-0.15, 0.15, 0.35, 0.45]),
                ap: GpuVec4::from_array([0.10, -0.10, 0.15, 0.10]),
                dir: GpuVec4::from_array([0.20, 0.15, -0.10, 0.25]),
            },
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::from_array([0.90, 0.82, 0.74, 0.66]),
                GpuVec4::from_array([0.30, 0.45, 0.55, 0.70]),
                GpuVec4::from_array([0.15, 0.30, 0.50, 0.75]),
                GpuVec4::from_array([0.12, 0.25, 0.42, 0.60]),
                GpuVec4::from_array([0.22, 0.38, 0.52, 0.68]),
                GpuVec4::from_array([0.10, 0.25, 0.40, 0.55]),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::zero(),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            sag_depth: GpuVec4::from_array([0.18, 0.28, 0.36, 0.22]),
            adaa_drive: GpuVec4::from_array([1.2, 1.5, 1.9, 1.3]),
            lru_bias_drive: GpuVec4::from_array([0.15, -0.20, 0.25, -0.15]),
            wavefolder_mix: GpuVec4::from_array([0.2, 0.4, 0.6, 0.8]),
            glu_mix: GpuVec4::from_array([0.55, -0.40, 0.70, -0.45]),
        };
        let mut packed = pack_core_quad_params(&params);
        if svf_g_zero {
            packed[quad_param_slots::SVF_G] = [0.0; 4];
        }
        (params, packed)
    }

    #[test]
    fn test_cpu_core_scan_streamed_parity_zero_dc_and_adjoint_fd() {
        // verify[train.pipeline.webgpu-scans-and-ozaki]
        let num_frames = 540_usize;
        let num_tiles = num_frames.div_ceil(TILE_FRAMES);
        let (_, packed) = make_test_quad(false);
        let mut core_inputs = vec![[0.0_f32; 4]; num_frames * CORE_INPUT_VEC4_STRIDE];
        for t in 0..num_frames {
            let ph = (t as f32) * 0.083;
            core_inputs[t * 2] = [
                0.45 * ph.sin(),
                0.35 * (1.3 * ph).cos(),
                0.40 * ph.cos(),
                0.25,
            ];
            core_inputs[t * 2 + 1] = [
                0.35 * (0.9 * ph).cos(),
                0.42 * ph.sin(),
                0.30,
                -0.25 * ph.sin(),
            ];
        }
        let u = FusedScanUniforms {
            num_frames: num_frames as u32,
            num_tiles: num_tiles as u32,
            num_quads: 1,
            num_stems: 1,
            tile_start: 0,
            soft_rect_eps: SOFT_RECT_EPS,
            tiny_crossing_eps: TINY_CROSSING_EPS,
            block_idx: 0,
        };
        let res = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u,
            core_inputs: &core_inputs,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();

        let mut streamed_plan = plan_core_scan_execution(
            CoreScanWorkload {
                num_frames,
                num_quads: 1,
                num_stems: 1,
                include_adjoint: false,
            },
            MemoryBudgetProfile::BrowserWebGpu,
            None,
        )
        .unwrap();
        streamed_plan.mode = ExecutionMode::Streamed;
        streamed_plan.time_block_frames = TILE_FRAMES;
        streamed_plan.time_block_tiles = 1;
        let streamed = run_fused_core_scan_planned_cpu(
            &FusedCoreScanInput {
                uniforms: u,
                core_inputs: &core_inputs,
                quad_params: &packed,
                initial_checkpoints: None,
            },
            &streamed_plan,
        )
        .unwrap();
        for (a, b) in res.feature_outputs.iter().zip(&streamed.feature_outputs) {
            assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
        }

        // Zero-DC canonical +0.0 verification on pure -0.0 inputs with negative weights.
        let neg_z = vec![[-0.0_f32; 4]; core_inputs.len()];
        let mut neg_p = packed;
        neg_p[quad_param_slots::TONE_LP] = [-0.6; 4];
        neg_p[quad_param_slots::GLU_MIX] = [-0.5; 4];
        let z_out = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u,
            core_inputs: &neg_z,
            quad_params: &neg_p,
            initial_checkpoints: None,
        })
        .unwrap();
        assert!(
            z_out
                .feature_outputs
                .iter()
                .all(|v| v.iter().all(|x| x.to_bits() == 0))
        );

        // Multi-tile adjoint finite-difference check across tile seam (t = 255).
        let mut d_loss = vec![[0.0_f32; 4]; num_frames * FEATURE_VEC4_STRIDE];
        for (idx, item) in d_loss.iter_mut().enumerate() {
            let ph = (idx as f32) * 0.017;
            *item = [
                0.3 * ph.sin(),
                0.25 * ph.cos(),
                -0.2 * ph.sin(),
                0.35 * ph.cos(),
            ];
        }
        let adj_out = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: crate::gpu::abi::AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_inputs,
            quad_params: &packed,
            tile_checkpoints: &res.tile_checkpoints,
            d_loss_d_features: &d_loss,
            feature_outputs: &res.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();

        let eval_loss = |inputs: &[[f32; 4]], params_block: &[[f32; 4]]| -> f64 {
            let r = run_fused_core_scan_cpu(&FusedCoreScanInput {
                uniforms: u,
                core_inputs: inputs,
                quad_params: params_block,
                initial_checkpoints: None,
            })
            .unwrap();
            r.feature_outputs
                .iter()
                .zip(&d_loss)
                .map(|(f, w)| {
                    (0..4)
                        .map(|l| f64::from(f[l]) * f64::from(w[l]))
                        .sum::<f64>()
                })
                .sum()
        };
        let eps = 2.0e-3_f32;
        for t in [15_usize, 255, 256, 512] {
            let mut p_p = core_inputs.clone();
            let mut p_m = core_inputs.clone();
            p_p[t * 2][0] += eps;
            p_m[t * 2][0] -= eps;
            let fd = (eval_loss(&p_p, &packed) - eval_loss(&p_m, &packed))
                / f64::from(p_p[t * 2][0] - p_m[t * 2][0]);
            let got = f64::from(adj_out.d_loss_d_inputs[t * 2][0]);
            assert!((got - fd).abs() <= 4.0e-3 * fd.abs().max(1.0));
        }

        // Verify SVF_G == 0 produces exact zero gradient when below MIN_TPT_G.
        let (_, packed_zero_g) = make_test_quad(true);
        let adj_zero = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: crate::gpu::abi::AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_inputs,
            quad_params: &packed_zero_g,
            tile_checkpoints: &res.tile_checkpoints,
            d_loss_d_features: &d_loss,
            feature_outputs: &res.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();
        assert_eq!(adj_zero.param_grads[quad_param_slots::SVF_G], [0.0; 4]);

        // Verify non-zero block_idx masks SVF_K_NL in forward (0-ULP vs k_nl=0) and backward (exact 0).
        let mut u_b1 = u;
        u_b1.block_idx = 1;
        let mut packed_b1 = packed;
        packed_b1[quad_param_slots::SVF_K_NL] = [0.5; 4];
        let fwd_b1_zero = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u_b1,
            core_inputs: &core_inputs,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        let fwd_b1_nz = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: u_b1,
            core_inputs: &core_inputs,
            quad_params: &packed_b1,
            initial_checkpoints: None,
        })
        .unwrap();
        assert_eq!(
            fwd_b1_nz
                .tile_checkpoints
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>(),
            fwd_b1_zero
                .tile_checkpoints
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            fwd_b1_nz
                .feature_outputs
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>(),
            fwd_b1_zero
                .feature_outputs
                .iter()
                .map(|v| v.map(f32::to_bits))
                .collect::<Vec<_>>()
        );
        let adj_b1 = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: crate::gpu::abi::AdjointScanUniforms {
                num_frames: num_frames as u32,
                num_tiles: num_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 1,
            },
            core_inputs: &core_inputs,
            quad_params: &packed_b1,
            tile_checkpoints: &fwd_b1_nz.tile_checkpoints,
            d_loss_d_features: &d_loss,
            feature_outputs: &fwd_b1_nz.feature_outputs,
            exit_adj_boundary: None,
        })
        .unwrap();
        for tile in 0..num_tiles {
            assert_eq!(
                adj_b1.param_grads[tile * QUAD_PARAMS_VEC4_STRIDE + quad_param_slots::SVF_K_NL]
                    .map(f32::to_bits),
                [0; 4]
            );
        }

        // Cover ResidentBatched planned delegation and dimension error branches.
        let res_plan = plan_core_scan_execution(
            CoreScanWorkload {
                num_frames,
                num_quads: 1,
                num_stems: 1,
                include_adjoint: false,
            },
            MemoryBudgetProfile::DesktopCapsule,
            None,
        )
        .unwrap();
        let res_planned = run_fused_core_scan_planned_cpu(
            &FusedCoreScanInput {
                uniforms: u,
                core_inputs: &core_inputs,
                quad_params: &packed,
                initial_checkpoints: None,
            },
            &res_plan,
        )
        .unwrap();
        assert_eq!(res_planned.feature_outputs, res.feature_outputs);

        for (bad_u, bad_in, bad_p) in [
            (
                FusedScanUniforms { num_frames: 0, ..u },
                &core_inputs[..],
                &packed[..],
            ),
            (u, &[][..], &packed[..]),
            (u, &core_inputs[..], &[][..]),
        ] {
            let err = run_fused_core_scan_cpu(&FusedCoreScanInput {
                uniforms: bad_u,
                core_inputs: bad_in,
                quad_params: bad_p,
                initial_checkpoints: None,
            });
            assert!(matches!(err, Err(TrainError::InvalidDimensions { .. })));
        }
        assert!(matches!(
            run_adjoint_scan_cpu(&AdjointScanInput {
                uniforms: crate::gpu::abi::AdjointScanUniforms {
                    num_frames: 0,
                    num_tiles: 1,
                    num_quads: 1,
                    num_stems: 1,
                    pass_mode: 2,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: 0,
                },
                core_inputs: &core_inputs,
                quad_params: &packed,
                tile_checkpoints: &res.tile_checkpoints,
                d_loss_d_features: &d_loss,
                feature_outputs: &res.feature_outputs,
                exit_adj_boundary: None,
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            run_adjoint_scan_cpu(&AdjointScanInput {
                uniforms: crate::gpu::abi::AdjointScanUniforms {
                    num_frames: num_frames as u32,
                    num_tiles: num_tiles as u32,
                    num_quads: 1,
                    num_stems: 1,
                    pass_mode: 2,
                    soft_rect_eps: SOFT_RECT_EPS,
                    tiny_crossing_eps: TINY_CROSSING_EPS,
                    block_idx: 0,
                },
                core_inputs: &[],
                quad_params: &packed,
                tile_checkpoints: &res.tile_checkpoints,
                d_loss_d_features: &d_loss,
                feature_outputs: &res.feature_outputs,
                exit_adj_boundary: None,
            }),
            Err(TrainError::InvalidDimensions { .. })
        ));

        // Verify 2-chunk exit_adj_boundary -> entry_adj_boundary chaining matches resident execution.
        let rem_frames = num_frames - TILE_FRAMES;
        let rem_tiles = rem_frames.div_ceil(TILE_FRAMES);
        let mut seed_chunk0 = vec![[0.0_f32; 4]; EXIT_ADJ_BOUNDARY_VEC4_STRIDE];
        let c1 = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: crate::gpu::abi::AdjointScanUniforms {
                num_frames: rem_frames as u32,
                num_tiles: rem_tiles as u32,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_inputs[TILE_FRAMES * CORE_INPUT_VEC4_STRIDE..],
            quad_params: &packed,
            tile_checkpoints: &res.tile_checkpoints[TILE_CKPT_VEC4_STRIDE..],
            d_loss_d_features: &d_loss[TILE_FRAMES * FEATURE_VEC4_STRIDE..],
            feature_outputs: &res.feature_outputs[TILE_FRAMES * FEATURE_VEC4_STRIDE..],
            exit_adj_boundary: None,
        })
        .unwrap();
        seed_chunk0[0..8].copy_from_slice(&c1.entry_adj_boundary[8..16]);
        let c0 = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: crate::gpu::abi::AdjointScanUniforms {
                num_frames: TILE_FRAMES as u32,
                num_tiles: 1,
                num_quads: 1,
                num_stems: 1,
                pass_mode: 2,
                soft_rect_eps: SOFT_RECT_EPS,
                tiny_crossing_eps: TINY_CROSSING_EPS,
                block_idx: 0,
            },
            core_inputs: &core_inputs[..TILE_FRAMES * CORE_INPUT_VEC4_STRIDE],
            quad_params: &packed,
            tile_checkpoints: &res.tile_checkpoints[..TILE_CKPT_VEC4_STRIDE],
            d_loss_d_features: &d_loss[..TILE_FRAMES * FEATURE_VEC4_STRIDE],
            feature_outputs: &res.feature_outputs[..TILE_FRAMES * FEATURE_VEC4_STRIDE],
            exit_adj_boundary: Some(&seed_chunk0),
        })
        .unwrap();
        for (a, b) in adj_out
            .d_loss_d_inputs
            .iter()
            .zip(c0.d_loss_d_inputs.iter().chain(c1.d_loss_d_inputs.iter()))
        {
            for k in 0..4 {
                assert!((a[k] - b[k]).abs() <= 1.0e-6);
            }
        }

        // Regression: BLOCK_IDX_DIRECT_LINEAR must not be masked by FUSED_FLAG_FIXED_POINT_SCAN.
        let direct = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                block_idx: BLOCK_IDX_DIRECT_LINEAR,
                ..u
            },
            core_inputs: &core_inputs,
            quad_params: &packed,
            initial_checkpoints: None,
        })
        .unwrap();
        assert_eq!(direct.feature_outputs, res.feature_outputs);
        assert!(
            direct.tile_checkpoints[TILE_CKPT_VEC4_STRIDE..]
                .iter()
                .flat_map(|v| v.iter())
                .any(|&x| x.abs() > 1.0e-4)
        );
    }
}
