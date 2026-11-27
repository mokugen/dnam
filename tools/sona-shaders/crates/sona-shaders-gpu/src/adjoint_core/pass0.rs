#![forbid(unsafe_code)]

//! `adjoint_scan_256` `pass_mode = 0`: optional race-free `Slot 5` `b_all` staging, parallel
//! per-tile forward trajectory recomputation, exact inter-tile seam `r_{256}` scalar evaluation,
//! and level-1 reverse `ParaLRU` / `TPT-SVF` / 2-pass `Pred-Corr PHS` boundary summary emission
//! into `11`-Vec4-per-lane `Slot 7` (`tile_adj_boundaries`)
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::common::{phs_pass_a_lin_coeffs, recompute_tile_forward_lane};
use super::pullback::{
    StereoPullback, build_tpt_adjoint_elem, eval_duhem_and_tone_pullback,
    eval_frame_duhem_terms_lr, eval_phs_adj_v1_lin, eval_stereo_link_pullback,
};
use super::se_and_adaa::{
    compute_next_tile_first_seam, eval_se_and_adaa_local_and_seam_pullback, needs_b_staging,
    stage_b_all_into_slot5,
};
use crate::abi::{
    ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S, ADJOINT_V_TRAJ2_VEC2S, AdjointScanUniforms,
    EXIT_ADJ_BOUNDARY_VEC4_STRIDE, TILE_FRAMES,
};
use crate::core_math::{
    QUAD_PARAMS_VEC4_STRIDE, QuadParamsUnpacked, TILE_ADJ_BOUNDARY_VEC4_STRIDE,
    TILE_ADJ_LANE_VEC4_STRIDE, eval_tpt_svf_lp_bp_scalar, get_lane,
    linear_adjoint_reverse_suffix_256_lr, load_vec4, store_vec4, sub_sat_usize,
    tpt_adjoint_const_m_reduce_256_lr, tpt_adjoint_reverse_suffix_with_after_256,
};
use dnam_kernels::{
    LinearScanElem, PhsPredCorrPassBCoeffs, TptAdjointElem, eval_phs_pred_corr_pass_b_coeffs,
    phs_pred_corr_pass_b_adjoint_elem,
};
use spirv_std::glam::{Vec2, Vec4};

/// Executes `adjoint_scan_256` `pass_mode = 0` for one `(q_tier, q_wave, tile)` workgroup,
/// emitting level-1 reverse summaries, frame-0 `adj_r_prev`, and cross-operator coupling into `Slot 7`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn run_pass0(
    q_tier: usize,
    q_wave: usize,
    tile_base: usize,
    is_phs: bool,
    tile: usize,
    lid: usize,
    uniforms: &AdjointScanUniforms,
    core_inputs: &[Vec4],
    quad_params: &[Vec4],
    tile_checkpoints: &[Vec4],
    d_loss_d_features: &[Vec4],
    d_loss_d_inputs: &mut [Vec4],
    tile_adj_boundaries: &mut [Vec4],
    feature_outputs: &[Vec4],
    s_prev2: &mut [Vec4; TILE_FRAMES],
    u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
) {
    let i = lid & 255;
    let num_frames = uniforms.num_frames as usize;
    let num_tiles = (uniforms.num_tiles as usize).max(1);
    let valid_len = if tile < num_tiles {
        sub_sat_usize(num_frames, tile * TILE_FRAMES).min(TILE_FRAMES)
    } else {
        0
    };
    let is_active = i < valid_len;
    let tiny_eps = uniforms.tiny_crossing_eps;
    let exit_base = q_wave * EXIT_ADJ_BOUNDARY_VEC4_STRIDE;
    let params = QuadParamsUnpacked::load(quad_params, q_tier * QUAD_PARAMS_VEC4_STRIDE);
    let stage_b = needs_b_staging(&params);

    if stage_b {
        stage_b_all_into_slot5(
            core_inputs,
            tile_checkpoints,
            feature_outputs,
            d_loss_d_inputs,
            s_prev2,
            u_tone2,
            v_traj2,
            scan_scratch,
            &params,
            q_tier,
            num_frames,
            num_tiles,
            tile,
            valid_len,
            i,
            is_phs,
        );
    }

    for c in 0..4 {
        let lp = params.lane(c);
        let fwd = recompute_tile_forward_lane(
            core_inputs,
            tile_checkpoints,
            s_prev2,
            u_tone2,
            v_traj2,
            scan_scratch,
            q_tier,
            num_frames,
            num_tiles,
            tile,
            valid_len,
            i,
            c,
            &lp,
            is_phs,
        );
        let r_first_l = fwd.r_curr_l;
        let r_first_r = fwd.r_curr_r;

        let (r_next_seam, b_next_seam) = if i == TILE_FRAMES - 1 && tile + 1 < num_tiles {
            let (rs, bl, br) = compute_next_tile_first_seam(
                core_inputs,
                tile_checkpoints,
                feature_outputs,
                &params,
                q_tier,
                num_frames,
                num_tiles,
                tile + 1,
                c,
                is_phs,
                stage_b,
            );
            (rs, (bl, br))
        } else {
            ((0.0, 0.0), (Vec4::ZERO, Vec4::ZERO))
        };
        let exit_adj_r_prev_seam = if tile + 1 == num_tiles && valid_len > 0 && i + 1 == valid_len {
            (
                get_lane(load_vec4(tile_adj_boundaries, exit_base + 6), c),
                get_lane(load_vec4(tile_adj_boundaries, exit_base + 7), c),
            )
        } else {
            (0.0, 0.0)
        };

        let nonlin = eval_se_and_adaa_local_and_seam_pullback(
            d_loss_d_features,
            feature_outputs,
            d_loss_d_inputs,
            scan_scratch,
            &fwd,
            &params,
            &lp,
            q_tier,
            num_frames,
            num_tiles,
            tile,
            valid_len,
            i,
            c,
            tiny_eps,
            r_next_seam,
            b_next_seam,
            exit_adj_r_prev_seam,
            stage_b,
        );
        let st = eval_stereo_link_pullback(&fwd, &nonlin, &lp, is_active);

        let sp = s_prev2[i];
        let (pb_l, pb_r, lp_l, bp_l, lp_r, bp_r, k_eff_l, k_eff_r) = if is_phs {
            let pbl = eval_phs_pred_corr_pass_b_coeffs(fwd.v1_lin_l, lp.svf_g, lp.k, lp.svf_k_nl);
            let pbr = eval_phs_pred_corr_pass_b_coeffs(fwd.v1_lin_r, lp.svf_g, lp.k, lp.svf_k_nl);
            let bpl = pbl.a1 * sp.x + pbl.a2 * (fwd.u_l - sp.y);
            let lpl = sp.y + pbl.a2 * sp.x + pbl.a3 * (fwd.u_l - sp.y);
            let bpr = pbr.a1 * sp.z + pbr.a2 * (fwd.u_r - sp.w);
            let lpr = sp.w + pbr.a2 * sp.z + pbr.a3 * (fwd.u_r - sp.w);
            (pbl, pbr, lpl, bpl, lpr, bpr, pbl.k_eff, pbr.k_eff)
        } else {
            let svf_c = lp.to_svf_coeffs();
            let (lpl, bpl) = eval_tpt_svf_lp_bp_scalar(fwd.u_l, sp.x, sp.y, &svf_c);
            let (lpr, bpr) = eval_tpt_svf_lp_bp_scalar(fwd.u_r, sp.z, sp.w, &svf_c);
            let z_pb = PhsPredCorrPassBCoeffs {
                v1_lin: 0.0,
                e: 0.0,
                k_eff: lp.k,
                a1: lp.a1,
                a2: lp.a2,
                a3: lp.a3,
            };
            (z_pb, z_pb, lpl, bpl, lpr, bpr, lp.k, lp.k)
        };
        let (tl, tr) = eval_frame_duhem_terms_lr(v_traj2, &fwd, bp_l, bp_r, i, &lp);

        let lru_elem_l = if is_active {
            LinearScanElem {
                a: tl.jac,
                b: tl.jac * st.adj_v_raw_l,
            }
        } else {
            LinearScanElem::identity()
        };
        let lru_elem_r = if is_active {
            LinearScanElem {
                a: tr.jac,
                b: tr.jac * st.adj_v_raw_r,
            }
        } else {
            LinearScanElem::identity()
        };
        let (sum_lru_l, sum_lru_r, lam_after_l, lam_after_r, p_after_l, p_after_r) =
            linear_adjoint_reverse_suffix_256_lr(scan_scratch, i, lru_elem_l, lru_elem_r, 0.0, 0.0);

        let lam_l = if is_active {
            st.adj_v_raw_l + lam_after_l
        } else {
            0.0
        };
        let lam_r = if is_active {
            st.adj_v_raw_r + lam_after_r
        } else {
            0.0
        };
        let dt = eval_duhem_and_tone_pullback(
            &fwd, &st, &tl, &tr, lam_l, lam_r, lp_l, bp_l, lp_r, bp_r, k_eff_l, k_eff_r, &lp,
            is_active,
        );

        let unit_lam_l = if is_active { p_after_l } else { 0.0 };
        let unit_lam_r = if is_active { p_after_r } else { 0.0 };
        let zero_st = StereoPullback::zero();
        let dt_k = eval_duhem_and_tone_pullback(
            &fwd, &zero_st, &tl, &tr, unit_lam_l, unit_lam_r, lp_l, bp_l, lp_r, bp_r, k_eff_l,
            k_eff_r, &lp, is_active,
        );

        if is_phs {
            let elem_b_l = if is_active {
                phs_pred_corr_pass_b_adjoint_elem(&pb_l, dt.adj_v1_l, dt.adj_v2_l)
            } else {
                TptAdjointElem::identity()
            };
            let elem_b_r = if is_active {
                phs_pred_corr_pass_b_adjoint_elem(&pb_r, dt.adj_v1_r, dt.adj_v2_r)
            } else {
                TptAdjointElem::identity()
            };
            let elem_kb_l = if is_active {
                phs_pred_corr_pass_b_adjoint_elem(&pb_l, dt_k.adj_v1_l, dt_k.adj_v2_l)
            } else {
                TptAdjointElem::identity()
            };
            let elem_kb_r = if is_active {
                phs_pred_corr_pass_b_adjoint_elem(&pb_r, dt_k.adj_v1_r, dt_k.adj_v2_r)
            } else {
                TptAdjointElem::identity()
            };

            let (sum_b_l, after_b_l) =
                tpt_adjoint_reverse_suffix_with_after_256(scan_scratch, i, elem_b_l);
            let (sum_b_r, after_b_r) =
                tpt_adjoint_reverse_suffix_with_after_256(scan_scratch, i, elem_b_r);
            let (sum_kb_l, after_kb_l) =
                tpt_adjoint_reverse_suffix_with_after_256(scan_scratch, i, elem_kb_l);
            let (sum_kb_r, after_kb_r) =
                tpt_adjoint_reverse_suffix_with_after_256(scan_scratch, i, elem_kb_r);

            let (lin_a1, lin_a2, lin_a3) = phs_pass_a_lin_coeffs(&lp);
            let v1_0_l = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_l,
                    [sp.x, sp.y],
                    fwd.u_l,
                    lp.svf_g,
                    lp.svf_k_nl,
                    dt.adj_v1_l,
                    dt.adj_v2_l,
                    dt.adj_k_eff_tone_l,
                    [after_b_l.g0, after_b_l.g1],
                )
            } else {
                0.0
            };
            let v1_0_r = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_r,
                    [sp.z, sp.w],
                    fwd.u_r,
                    lp.svf_g,
                    lp.svf_k_nl,
                    dt.adj_v1_r,
                    dt.adj_v2_r,
                    dt.adj_k_eff_tone_r,
                    [after_b_r.g0, after_b_r.g1],
                )
            } else {
                0.0
            };
            let v1_k_l = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_l,
                    [sp.x, sp.y],
                    fwd.u_l,
                    lp.svf_g,
                    lp.svf_k_nl,
                    dt_k.adj_v1_l,
                    dt_k.adj_v2_l,
                    dt_k.adj_k_eff_tone_l,
                    [after_kb_l.g0, after_kb_l.g1],
                )
            } else {
                0.0
            };
            let v1_k_r = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_r,
                    [sp.z, sp.w],
                    fwd.u_r,
                    lp.svf_g,
                    lp.svf_k_nl,
                    dt_k.adj_v1_r,
                    dt_k.adj_v2_r,
                    dt_k.adj_k_eff_tone_r,
                    [after_kb_r.g0, after_kb_r.g1],
                )
            } else {
                0.0
            };
            let v1_b0_l = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_l,
                    [sp.x, sp.y],
                    fwd.u_l,
                    lp.svf_g,
                    lp.svf_k_nl,
                    0.0,
                    0.0,
                    0.0,
                    [after_b_l.t00, after_b_l.t10],
                )
            } else {
                0.0
            };
            let v1_b0_r = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_r,
                    [sp.z, sp.w],
                    fwd.u_r,
                    lp.svf_g,
                    lp.svf_k_nl,
                    0.0,
                    0.0,
                    0.0,
                    [after_b_r.t00, after_b_r.t10],
                )
            } else {
                0.0
            };
            let v1_b1_l = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_l,
                    [sp.x, sp.y],
                    fwd.u_l,
                    lp.svf_g,
                    lp.svf_k_nl,
                    0.0,
                    0.0,
                    0.0,
                    [after_b_l.t01, after_b_l.t11],
                )
            } else {
                0.0
            };
            let v1_b1_r = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_r,
                    [sp.z, sp.w],
                    fwd.u_r,
                    lp.svf_g,
                    lp.svf_k_nl,
                    0.0,
                    0.0,
                    0.0,
                    [after_b_r.t01, after_b_r.t11],
                )
            } else {
                0.0
            };

            let (sum_lin_0_l, sum_lin_0_r, sum_lin_k_l_g, sum_lin_k_r_g) =
                tpt_adjoint_const_m_reduce_256_lr(
                    scan_scratch,
                    i,
                    valid_len,
                    lin_a1,
                    lin_a2,
                    lin_a3,
                    [lin_a1 * v1_0_l, -lin_a2 * v1_0_l],
                    [lin_a1 * v1_0_r, -lin_a2 * v1_0_r],
                    [lin_a1 * v1_k_l, -lin_a2 * v1_k_l],
                    [lin_a1 * v1_k_r, -lin_a2 * v1_k_r],
                );
            let (sum_lin_b0_l, sum_lin_b0_r, sum_lin_b1_l_g, sum_lin_b1_r_g) =
                tpt_adjoint_const_m_reduce_256_lr(
                    scan_scratch,
                    i,
                    valid_len,
                    lin_a1,
                    lin_a2,
                    lin_a3,
                    [lin_a1 * v1_b0_l, -lin_a2 * v1_b0_l],
                    [lin_a1 * v1_b0_r, -lin_a2 * v1_b0_r],
                    [lin_a1 * v1_b1_l, -lin_a2 * v1_b1_l],
                    [lin_a1 * v1_b1_r, -lin_a2 * v1_b1_r],
                );

            if i == 0 && tile < num_tiles {
                let slot = tile_base
                    + (q_wave * num_tiles + tile) * TILE_ADJ_BOUNDARY_VEC4_STRIDE
                    + c * TILE_ADJ_LANE_VEC4_STRIDE;
                store_vec4(
                    tile_adj_boundaries,
                    slot,
                    Vec4::new(sum_b_l.t00, sum_b_l.t01, sum_b_l.t10, sum_b_l.t11),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 1,
                    Vec4::new(sum_b_l.g0, sum_b_l.g1, sum_b_r.g0, sum_b_r.g1),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 2,
                    Vec4::new(sum_lru_l.a, sum_lru_l.b, sum_lru_r.a, sum_lru_r.b),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 3,
                    Vec4::new(
                        r_first_l,
                        r_first_r,
                        nonlin.adj_r_prev_l,
                        nonlin.adj_r_prev_r,
                    ),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 4,
                    Vec4::new(sum_b_r.t00, sum_b_r.t01, sum_b_r.t10, sum_b_r.t11),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 5,
                    Vec4::new(sum_kb_l.g0, sum_kb_l.g1, sum_kb_r.g0, sum_kb_r.g1),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 6,
                    Vec4::new(
                        sum_lin_0_l.t00,
                        sum_lin_0_l.t01,
                        sum_lin_0_l.t10,
                        sum_lin_0_l.t11,
                    ),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 7,
                    Vec4::new(
                        sum_lin_0_l.g0,
                        sum_lin_0_l.g1,
                        sum_lin_0_r.g0,
                        sum_lin_0_r.g1,
                    ),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 8,
                    Vec4::new(
                        sum_lin_k_l_g[0],
                        sum_lin_k_l_g[1],
                        sum_lin_k_r_g[0],
                        sum_lin_k_r_g[1],
                    ),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 9,
                    Vec4::new(
                        sum_lin_b0_l.g0,
                        sum_lin_b0_l.g1,
                        sum_lin_b0_r.g0,
                        sum_lin_b0_r.g1,
                    ),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 10,
                    Vec4::new(
                        sum_lin_b1_l_g[0],
                        sum_lin_b1_l_g[1],
                        sum_lin_b1_r_g[0],
                        sum_lin_b1_r_g[1],
                    ),
                );
            }
        } else {
            let tpt_elem_l =
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_l, dt.adj_v2_l, is_active);
            let tpt_elem_r =
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_r, dt.adj_v2_r, is_active);
            let tpt_k_l = build_tpt_adjoint_elem(
                lp.a1,
                lp.a2,
                lp.a3,
                dt_k.adj_v1_l,
                dt_k.adj_v2_l,
                is_active,
            );
            let tpt_k_r = build_tpt_adjoint_elem(
                lp.a1,
                lp.a2,
                lp.a3,
                dt_k.adj_v1_r,
                dt_k.adj_v2_r,
                is_active,
            );
            let (sum_tpt_l, sum_tpt_r, sum_k_l_g, sum_k_r_g) = tpt_adjoint_const_m_reduce_256_lr(
                scan_scratch,
                i,
                valid_len,
                lp.a1,
                lp.a2,
                lp.a3,
                [tpt_elem_l.g0, tpt_elem_l.g1],
                [tpt_elem_r.g0, tpt_elem_r.g1],
                [tpt_k_l.g0, tpt_k_l.g1],
                [tpt_k_r.g0, tpt_k_r.g1],
            );

            if i == 0 && tile < num_tiles {
                let slot = tile_base
                    + (q_wave * num_tiles + tile) * TILE_ADJ_BOUNDARY_VEC4_STRIDE
                    + c * TILE_ADJ_LANE_VEC4_STRIDE;
                store_vec4(
                    tile_adj_boundaries,
                    slot,
                    Vec4::new(sum_tpt_l.t00, sum_tpt_l.t01, sum_tpt_l.t10, sum_tpt_l.t11),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 1,
                    Vec4::new(sum_tpt_l.g0, sum_tpt_l.g1, sum_tpt_r.g0, sum_tpt_r.g1),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 2,
                    Vec4::new(sum_lru_l.a, sum_lru_l.b, sum_lru_r.a, sum_lru_r.b),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 3,
                    Vec4::new(
                        r_first_l,
                        r_first_r,
                        nonlin.adj_r_prev_l,
                        nonlin.adj_r_prev_r,
                    ),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 4,
                    Vec4::new(sum_tpt_r.t00, sum_tpt_r.t01, sum_tpt_r.t10, sum_tpt_r.t11),
                );
                store_vec4(
                    tile_adj_boundaries,
                    slot + 5,
                    Vec4::new(sum_k_l_g[0], sum_k_l_g[1], sum_k_r_g[0], sum_k_r_g[1]),
                );
                store_vec4(tile_adj_boundaries, slot + 6, Vec4::ZERO);
                store_vec4(tile_adj_boundaries, slot + 7, Vec4::ZERO);
                store_vec4(tile_adj_boundaries, slot + 8, Vec4::ZERO);
                store_vec4(tile_adj_boundaries, slot + 9, Vec4::ZERO);
                store_vec4(tile_adj_boundaries, slot + 10, Vec4::ZERO);
            }
        }
    }
}
