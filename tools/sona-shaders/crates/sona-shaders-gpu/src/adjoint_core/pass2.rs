#![forbid(unsafe_code)]

//! `adjoint_scan_256` `pass_mode = 2`: full tile BPTT using resolved cross-tile outgoing
//! boundaries (`Slot 7` Pass B / Pass A / `ParaLRU`), inter-tile seam `r_{256}` (`Slot 7`), and
//! `Slot 8` (`feature_outputs`) for `SE-Bus` + bounded `GLU` pullback, emitting `d_loss_d_inputs`
//! (`Slot 5`) and deterministic `28`-slot `param_grads` (`Slot 6`, zero `f32` atomics)
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use super::common::{phs_pass_a_lin_coeffs, recompute_tile_forward_lane};
use super::pullback::{
    build_tpt_adjoint_elem, eval_duhem_and_tone_pullback, eval_frame_duhem_terms_lr,
    eval_phs_adj_v1_lin, eval_phs_k_eff_tone_extras, eval_stereo_link_pullback,
};
use super::se_and_adaa::{
    compute_next_tile_first_seam, eval_se_and_adaa_local_and_seam_pullback, needs_b_staging,
};
use crate::abi::{
    ADJOINT_AUX_FLOATS, ADJOINT_SCAN_SCRATCH_FLOATS, ADJOINT_U_TONE2_VEC2S, ADJOINT_V_TRAJ2_VEC2S,
    AdjointScanUniforms, EXIT_ADJ_BOUNDARY_VEC4_STRIDE, TILE_FRAMES,
};
use crate::core_math::{
    CORE_INPUT_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE, QuadParamsUnpacked,
    TILE_ADJ_BOUNDARY_VEC4_STRIDE, TILE_ADJ_LANE_VEC4_STRIDE, eval_tpt_svf_lp_bp_scalar, get_lane,
    linear_adjoint_reverse_suffix_256_lr, load_vec4, param_slots, reduce_batch6_in_scratch,
    reduce_sum_256_in_aux, set_lane, store_vec4, sub_sat_usize, tpt_adjoint_reverse_suffix_256,
};
use dnam_kernels::{
    LinearScanElem, MIN_TPT_G, PhsPredCorrPassBCoeffs, PhsPredCorrStepContext, TptAdjointElem,
    eval_phs_pred_corr_pass_b_coeffs, eval_phs_pred_corr_step_adjoint,
    phs_pred_corr_pass_a_adjoint_elem, phs_pred_corr_pass_b_adjoint_elem,
};
use spirv_std::glam::{Vec2, Vec4};

#[inline(always)]
fn write_param_grad_lane(param_grads: &mut [Vec4], base: usize, slot: usize, c: usize, val: f32) {
    let idx = base + slot;
    let mut vec = if c == 0 {
        Vec4::ZERO
    } else {
        load_vec4(param_grads, idx)
    };
    set_lane(&mut vec, c, val);
    store_vec4(param_grads, idx, vec);
}

/// Executes `adjoint_scan_256` `pass_mode = 2` for one `(q_tier, q_wave, tile)` workgroup.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn run_pass2(
    q_tier: usize,
    q_wave: usize,
    tile_base: usize,
    is_phs: bool,
    tile: usize,
    lid: usize,
    sg_lid: u32,
    uniforms: &AdjointScanUniforms,
    core_inputs: &[Vec4],
    quad_params: &[Vec4],
    tile_checkpoints: &[Vec4],
    d_loss_d_features: &[Vec4],
    d_loss_d_inputs: &mut [Vec4],
    param_grads: &mut [Vec4],
    tile_adj_boundaries: &[Vec4],
    feature_outputs: &[Vec4],
    s_prev2: &mut [Vec4; TILE_FRAMES],
    u_tone2: &mut [Vec2; ADJOINT_U_TONE2_VEC2S],
    v_traj2: &mut [Vec2; ADJOINT_V_TRAJ2_VEC2S],
    scan_scratch: &mut [f32; ADJOINT_SCAN_SCRATCH_FLOATS],
    aux: &mut [f32; ADJOINT_AUX_FLOATS],
) {
    let i = lid & 255;
    let num_frames = uniforms.num_frames as usize;
    let num_tiles = (uniforms.num_tiles as usize).max(1);
    let tile_active = tile < num_tiles;
    let valid_len = if tile_active {
        sub_sat_usize(num_frames, tile * TILE_FRAMES).min(TILE_FRAMES)
    } else {
        0
    };
    let frame = tile * TILE_FRAMES + i;
    let is_active = i < valid_len;
    let tiny_eps = uniforms.tiny_crossing_eps;
    let exit_base = q_wave * EXIT_ADJ_BOUNDARY_VEC4_STRIDE;

    let params = QuadParamsUnpacked::load(quad_params, q_tier * QUAD_PARAMS_VEC4_STRIDE);
    let stage_b = needs_b_staging(&params);
    let mut out_du_l = Vec4::ZERO;
    let mut out_du_r = Vec4::ZERO;

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

        let slot = tile_base
            + (q_wave * num_tiles + tile) * TILE_ADJ_BOUNDARY_VEC4_STRIDE
            + c * TILE_ADJ_LANE_VEC4_STRIDE;
        let (adj_s_out_l, adj_s_out_r, adj_v_out_l, adj_v_out_r, adj_s_lin_out_l, adj_s_lin_out_r) =
            if tile_active {
                let s0 = load_vec4(tile_adj_boundaries, slot);
                let s2 = load_vec4(tile_adj_boundaries, slot + 2);
                let s6 = if is_phs {
                    load_vec4(tile_adj_boundaries, slot + 6)
                } else {
                    Vec4::ZERO
                };
                (
                    [s0.x, s0.y],
                    [s0.z, s0.w],
                    s2.x,
                    s2.y,
                    [s6.x, s6.y],
                    [s6.z, s6.w],
                )
            } else {
                ([0.0, 0.0], [0.0, 0.0], 0.0, 0.0, [0.0, 0.0], [0.0, 0.0])
            };

        let (r_next_seam, b_next_seam) = if i == TILE_FRAMES - 1 && tile + 1 < num_tiles {
            let seam = load_vec4(
                tile_adj_boundaries,
                slot + TILE_ADJ_BOUNDARY_VEC4_STRIDE + 3,
            );
            if stage_b {
                let (_, bl, br) = compute_next_tile_first_seam(
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
                    true,
                );
                ((seam.x, seam.y), (bl, br))
            } else {
                ((seam.x, seam.y), (Vec4::ZERO, Vec4::ZERO))
            }
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
        let (_sum_lru_l, _sum_lru_r, lam_after_l, lam_after_r, _, _) =
            linear_adjoint_reverse_suffix_256_lr(
                scan_scratch,
                i,
                lru_elem_l,
                lru_elem_r,
                adj_v_out_l,
                adj_v_out_r,
            );

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

        let (g_m0, g_m1, g_m2, g_m3, du_l, du_r) = if is_phs {
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
            let mut lam_b_l0 = 0.0_f32;
            let mut lam_b_l1 = 0.0_f32;
            let _ = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                elem_b_l,
                adj_s_out_l,
                &mut lam_b_l0,
                &mut lam_b_l1,
            );
            let mut lam_b_r0 = 0.0_f32;
            let mut lam_b_r1 = 0.0_f32;
            let _ = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                elem_b_r,
                adj_s_out_r,
                &mut lam_b_r0,
                &mut lam_b_r1,
            );

            let (lin_a1, lin_a2, lin_a3) = phs_pass_a_lin_coeffs(&lp);
            let v1_lin_l = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_l,
                    [sp.x, sp.y],
                    fwd.u_l,
                    lp.svf_g,
                    lp.svf_k_nl,
                    dt.adj_v1_l,
                    dt.adj_v2_l,
                    dt.adj_k_eff_tone_l,
                    [lam_b_l0, lam_b_l1],
                )
            } else {
                0.0
            };
            let v1_lin_r = if is_active {
                eval_phs_adj_v1_lin(
                    &pb_r,
                    [sp.z, sp.w],
                    fwd.u_r,
                    lp.svf_g,
                    lp.svf_k_nl,
                    dt.adj_v1_r,
                    dt.adj_v2_r,
                    dt.adj_k_eff_tone_r,
                    [lam_b_r0, lam_b_r1],
                )
            } else {
                0.0
            };
            let elem_a_l = if is_active {
                phs_pred_corr_pass_a_adjoint_elem(lin_a1, lin_a2, lin_a3, v1_lin_l)
            } else {
                TptAdjointElem::identity()
            };
            let elem_a_r = if is_active {
                phs_pred_corr_pass_a_adjoint_elem(lin_a1, lin_a2, lin_a3, v1_lin_r)
            } else {
                TptAdjointElem::identity()
            };
            let mut lam_lin_l0 = 0.0_f32;
            let mut lam_lin_l1 = 0.0_f32;
            let _ = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                elem_a_l,
                adj_s_lin_out_l,
                &mut lam_lin_l0,
                &mut lam_lin_l1,
            );
            let mut lam_lin_r0 = 0.0_f32;
            let mut lam_lin_r1 = 0.0_f32;
            let _ = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                elem_a_r,
                adj_s_lin_out_r,
                &mut lam_lin_r0,
                &mut lam_lin_r1,
            );

            if is_active {
                let ctx_l = PhsPredCorrStepContext {
                    u: fwd.u_l,
                    s_lin_prev: [fwd.s_lin_prev_l.x, fwd.s_lin_prev_l.y],
                    s_b_prev: [sp.x, sp.y],
                    g: lp.svf_g,
                    k: lp.k,
                    k_nl: lp.svf_k_nl,
                    pass_a_a1: lin_a1,
                    pass_a_a2: lin_a2,
                    pass_a_a3: lin_a3,
                };
                let ctx_r = PhsPredCorrStepContext {
                    u: fwd.u_r,
                    s_lin_prev: [fwd.s_lin_prev_r.x, fwd.s_lin_prev_r.y],
                    s_b_prev: [sp.z, sp.w],
                    g: lp.svf_g,
                    k: lp.k,
                    k_nl: lp.svf_k_nl,
                    pass_a_a1: lin_a1,
                    pass_a_a2: lin_a2,
                    pass_a_a3: lin_a3,
                };
                let terms_l = eval_phs_pred_corr_step_adjoint(
                    &ctx_l,
                    &pb_l,
                    [lam_b_l0, lam_b_l1],
                    [lam_lin_l0, lam_lin_l1],
                    dt.adj_v1_l,
                    dt.adj_v2_l,
                );
                let terms_r = eval_phs_pred_corr_step_adjoint(
                    &ctx_r,
                    &pb_r,
                    [lam_b_r0, lam_b_r1],
                    [lam_lin_r0, lam_lin_r1],
                    dt.adj_v1_r,
                    dt.adj_v2_r,
                );
                let (ex_u_l, ex_g_l, ex_k_l, ex_knl_l) = eval_phs_k_eff_tone_extras(
                    fwd.u_l,
                    [fwd.s_lin_prev_l.x, fwd.s_lin_prev_l.y],
                    lp.svf_g,
                    lp.k,
                    lp.svf_k_nl,
                    lin_a1,
                    lin_a2,
                    &pb_l,
                    dt.adj_k_eff_tone_l,
                );
                let (ex_u_r, ex_g_r, ex_k_r, ex_knl_r) = eval_phs_k_eff_tone_extras(
                    fwd.u_r,
                    [fwd.s_lin_prev_r.x, fwd.s_lin_prev_r.y],
                    lp.svf_g,
                    lp.k,
                    lp.svf_k_nl,
                    lin_a1,
                    lin_a2,
                    &pb_r,
                    dt.adj_k_eff_tone_r,
                );
                (
                    terms_l.adj_k + ex_k_l + terms_r.adj_k + ex_k_r,
                    terms_l.adj_k_nl + ex_knl_l + terms_r.adj_k_nl + ex_knl_r,
                    terms_l.adj_g + ex_g_l + terms_r.adj_g + ex_g_r,
                    0.0,
                    dt.adj_u_nonrec_l + terms_l.adj_u + ex_u_l,
                    dt.adj_u_nonrec_r + terms_r.adj_u + ex_u_r,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        } else {
            let tpt_elem_l =
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_l, dt.adj_v2_l, is_active);
            let tpt_elem_r =
                build_tpt_adjoint_elem(lp.a1, lp.a2, lp.a3, dt.adj_v1_r, dt.adj_v2_r, is_active);
            let mut psi0_l = 0.0_f32;
            let mut psi1_l = 0.0_f32;
            let _ = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                tpt_elem_l,
                adj_s_out_l,
                &mut psi0_l,
                &mut psi1_l,
            );
            let mut psi0_r = 0.0_f32;
            let mut psi1_r = 0.0_f32;
            let _ = tpt_adjoint_reverse_suffix_256(
                scan_scratch,
                i,
                tpt_elem_r,
                adj_s_out_r,
                &mut psi0_r,
                &mut psi1_r,
            );

            if is_active {
                let tot_v1_l = dt.adj_v1_l + (psi0_l + psi0_l);
                let tot_v2_l = dt.adj_v2_l + (psi1_l + psi1_l);
                let tot_v1_r = dt.adj_v1_r + (psi0_r + psi0_r);
                let tot_v2_r = dt.adj_v2_r + (psi1_r + psi1_r);
                let v3_l = fwd.u_l - sp.y;
                let v3_r = fwd.u_r - sp.w;
                let a1_g = tot_v1_l * sp.x + tot_v1_r * sp.z;
                let a2_g =
                    (tot_v1_l * v3_l + tot_v2_l * sp.x) + (tot_v1_r * v3_r + tot_v2_r * sp.z);
                let a3_g = tot_v2_l * v3_l + tot_v2_r * v3_r;
                let d_u_l = dt.adj_u_nonrec_l + lp.a2 * tot_v1_l + lp.a3 * tot_v2_l;
                let d_u_r = dt.adj_u_nonrec_r + lp.a2 * tot_v1_r + lp.a3 * tot_v2_r;
                (a1_g, a2_g, a3_g, dt.g_svf_k_tone, d_u_l, d_u_r)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            }
        };

        set_lane(&mut out_du_l, c, du_l);
        set_lane(&mut out_du_r, c, du_r);

        let r0 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [g_m0, g_m1, g_m2, g_m3, dt.g_tone_c_lp, dt.g_tone_c_bp],
        );
        let r1 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [
                dt.g_tone_c_kbp,
                dt.g_tone_c_u,
                dt.g_alpha_rel,
                dt.g_alpha_atk,
                dt.g_kappa,
                dt.g_gamma,
            ],
        );
        let r2 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [
                dt.g_signed_mix,
                st.g_stereo_link,
                dt.g_two_svf_g,
                dt.g_rect_sign,
                nonlin.grad_g0,
                nonlin.grad_g_inf,
            ],
        );
        let r3 = reduce_batch6_in_scratch(
            scan_scratch,
            i,
            sg_lid,
            [
                nonlin.grad_d_pre,
                nonlin.grad_se_w_sq,
                nonlin.grad_se_d_bus,
                nonlin.grad_adaa_drive,
                nonlin.grad_lru_bias_drive,
                nonlin.grad_lin_mix,
            ],
        );
        let g_wf_mix = reduce_sum_256_in_aux(aux, i, sg_lid, nonlin.grad_wf_mix);
        let g_glu_mix = reduce_sum_256_in_aux(aux, i, sg_lid, nonlin.grad_glu_mix);
        let g_out_gain = reduce_sum_256_in_aux(aux, i, sg_lid, nonlin.grad_out_gain);

        if i == 0 && tile_active {
            let pg_base = (q_tier * num_tiles + tile) * QUAD_PARAMS_VEC4_STRIDE;
            if is_phs {
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A1, c, 0.0);
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A2, c, 0.0);
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A3, c, 0.0);
                write_param_grad_lane(
                    param_grads,
                    pg_base,
                    param_slots::SVF_K,
                    c,
                    if lp.k >= 0.0 { r0[0] } else { 0.0 },
                );
                write_param_grad_lane(
                    param_grads,
                    pg_base,
                    param_slots::SVF_K_NL,
                    c,
                    if lp.svf_k_nl >= 0.0 { r0[1] } else { 0.0 },
                );
                let g_svf_g = (if lp.svf_g >= MIN_TPT_G { r0[2] } else { 0.0 }) + 2.0 * r2[2];
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_G, c, g_svf_g);
            } else {
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A1, c, r0[0]);
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A2, c, r0[1]);
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_A3, c, r0[2]);
                let g_k_lin = r0[3] - lp.a2 * (r0[0] * lp.a1 + r0[1] * lp.a2 + r0[2] * lp.a3);
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_K, c, g_k_lin);
                write_param_grad_lane(param_grads, pg_base, param_slots::SVF_K_NL, c, 0.0);
                let g_g_tpt = if lp.svf_g >= MIN_TPT_G {
                    let denom_g_term = 2.0 * lp.a2 + lp.k * lp.a1;
                    let da1_dg = -lp.a1 * denom_g_term;
                    let da2_dg = lp.a1 - lp.a2 * denom_g_term;
                    let da3_dg = (lp.a2 + lp.a2) - lp.a3 * denom_g_term;
                    r0[0] * da1_dg + r0[1] * da2_dg + r0[2] * da3_dg
                } else {
                    0.0
                };
                write_param_grad_lane(
                    param_grads,
                    pg_base,
                    param_slots::SVF_G,
                    c,
                    g_g_tpt + 2.0 * r2[2],
                );
            }

            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_C_LP, c, r0[4]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_C_BP, c, r0[5]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_C_KBP, c, r1[0]);
            write_param_grad_lane(param_grads, pg_base, param_slots::TONE_C_U, c, r1[1]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_ALPHA_REL, c, r1[2]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_ALPHA_ATK, c, r1[3]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_KAPPA, c, r1[4]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_GAMMA_DUHEM, c, r1[5]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_SIGNED_MIX, c, r2[0]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_STEREO_LINK, c, r2[1]);
            write_param_grad_lane(param_grads, pg_base, param_slots::RECT_SIGN, c, r2[3]);
            write_param_grad_lane(param_grads, pg_base, param_slots::G0, c, r2[4]);
            write_param_grad_lane(param_grads, pg_base, param_slots::G_INF, c, r2[5]);
            write_param_grad_lane(param_grads, pg_base, param_slots::D_PRE, c, r3[0]);
            write_param_grad_lane(param_grads, pg_base, param_slots::SE_W_SQ, c, r3[1]);
            write_param_grad_lane(param_grads, pg_base, param_slots::SE_D_BUS, c, r3[2]);
            write_param_grad_lane(param_grads, pg_base, param_slots::ADAA_DRIVE, c, r3[3]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LRU_BIAS_DRIVE, c, r3[4]);
            write_param_grad_lane(param_grads, pg_base, param_slots::LIN_MIX, c, r3[5]);
            write_param_grad_lane(
                param_grads,
                pg_base,
                param_slots::WAVEFOLDER_MIX,
                c,
                g_wf_mix,
            );
            write_param_grad_lane(param_grads, pg_base, param_slots::GLU_MIX, c, g_glu_mix);
            write_param_grad_lane(param_grads, pg_base, param_slots::OUT_GAIN, c, g_out_gain);
        }
    }

    let in_base = (q_tier * num_frames + frame) * CORE_INPUT_VEC4_STRIDE;
    if is_active {
        store_vec4(d_loss_d_inputs, in_base, out_du_l);
        store_vec4(d_loss_d_inputs, in_base + 1, out_du_r);
    }
}
