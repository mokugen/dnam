//! Block `0` 1-step Predictor-Corrector `PHS` (`SVF_G`, `SVF_K`, `SVF_K_NL`, `TONE_C_KBP`) and
//! Blocks `1..3` constant-`M` `TPT-SVF` forward/reverse pullback matching `tools/sona-shaders`'s
//! `adjoint_core/{common,pullback,pass2}.rs` (`ref[dsp.op-n.core.tpt-tone-basis]`).

use dnam_kernels::{
    GpuVec4, KernelVec4, MIN_TPT_G, PhsPredCorrPassBCoeffs, PhsPredCorrStepContext, SVF_PHS_E_MAX,
    TptAdjointElem, TptSvfCoeffs4, apply_tpt_adjoint, eval_phs_pred_corr_pass_b_coeffs,
    eval_phs_pred_corr_step_adjoint, phs_pred_corr_pass_a_adjoint_elem,
    phs_pred_corr_pass_b_adjoint_elem, step_tpt_svf_vec4,
};

use super::duhem::DuhemTonePullback;
use super::{LaneForwardFrame, QuadLaneParams, ReverseCarry, TileGradAccum};
use crate::gpu::abi::quad_param_slots;

/// Computes `(a1_lin, a2_lin, a3_lin)` for Block `0` Predictor-Corrector Pass A strictly from
/// `(SVF_G, SVF_K)` via `TptSvfCoeffs4::from_gk`, matching `dnam_kernels` (`0-ULP`).
#[inline(always)]
pub(super) fn phs_pass_a_lin_coeffs(lp: &QuadLaneParams) -> (f32, f32, f32) {
    let g_safe = lp.svf_g.max(MIN_TPT_G);
    let k_safe = lp.k.max(0.0);
    let c = TptSvfCoeffs4::from_gk(GpuVec4::splat(g_safe), GpuVec4::splat(k_safe));
    (c.a1.to_array()[0], c.a2.to_array()[0], c.a3.to_array()[0])
}

/// Evaluates scalar linear `TPT-SVF` step via `dnam_kernels::step_tpt_svf_vec4`, returning `(lp, bp)`.
#[inline(always)]
pub(super) fn eval_tpt_svf_lp_bp_scalar(
    u: f32,
    s1_prev: f32,
    s2_prev: f32,
    svf: &TptSvfCoeffs4<GpuVec4>,
) -> (f32, f32) {
    let mut s1 = GpuVec4::splat(s1_prev);
    let mut s2 = GpuVec4::splat(s2_prev);
    let out = step_tpt_svf_vec4(GpuVec4::splat(u), &mut s1, &mut s2, svf);
    (out.lp.to_array()[0], out.bp.to_array()[0])
}

#[inline(always)]
pub(super) fn build_tpt_adjoint_elem(
    a1: f32,
    a2: f32,
    a3: f32,
    adj_v1: f32,
    adj_v2: f32,
) -> TptAdjointElem {
    let two_a2 = a2 + a2;
    let two_a3 = a3 + a3;
    TptAdjointElem {
        t00: (a1 + a1) - 1.0,
        t01: two_a2,
        t10: -two_a2,
        t11: 1.0 - two_a3,
        g0: a1 * adj_v1 + a2 * adj_v2,
        g1: -a2 * adj_v1 + (1.0 - a3) * adj_v2,
    }
}

/// Evaluates Pass B -> Pass A coupling scalar `adj_v1_lin` for Block `0` 1-step Predictor-Corrector
/// `PHS`, including both Pass B `k_eff` sensitivity and Block `0` `c_kbp * k_eff * BP` tone coupling.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn eval_phs_adj_v1_lin(
    pb: &PhsPredCorrPassBCoeffs,
    s_b_prev: [f32; 2],
    u: f32,
    g: f32,
    k_nl: f32,
    adj_bp_out: f32,
    adj_lp_out: f32,
    adj_k_eff_tone: f32,
    lam_b_post: [f32; 2],
) -> f32 {
    let g_safe = g.max(MIN_TPT_G);
    let k_nl_safe = k_nl.max(0.0);
    let adj_v1_b = adj_bp_out + 2.0 * lam_b_post[0];
    let adj_v2_b = adj_lp_out + 2.0 * lam_b_post[1];
    let v3_b = u - s_b_prev[1];
    let adj_a1_b = adj_v1_b * s_b_prev[0];
    let adj_a2_b = adj_v1_b * v3_b + adj_v2_b * s_b_prev[0];
    let adj_a3_b = adj_v2_b * v3_b;
    let dl_da1_b = adj_a1_b + g_safe * adj_a2_b + (g_safe * g_safe) * adj_a3_b;
    let a1_b_sq = pb.a1 * pb.a1;
    let adj_k_eff = -(g_safe * a1_b_sq) * dl_da1_b + adj_k_eff_tone;
    if pb.v1_lin * pb.v1_lin < SVF_PHS_E_MAX {
        2.0 * k_nl_safe * pb.v1_lin * adj_k_eff
    } else {
        0.0
    }
}

/// Evaluates the extra `(adj_u, adj_g, adj_k, adj_k_nl)` contributions from Block `0` folded tone
/// `c_kbp * k_eff * BP` (`adj_k_eff_tone`), complementing `eval_phs_pred_corr_step_adjoint`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn eval_phs_k_eff_tone_extras(
    u: f32,
    s_lin_prev: [f32; 2],
    g: f32,
    k: f32,
    k_nl: f32,
    pass_a_a1: f32,
    pass_a_a2: f32,
    pb: &PhsPredCorrPassBCoeffs,
    adj_k_eff_tone: f32,
) -> (f32, f32, f32, f32) {
    let g_safe = g.max(MIN_TPT_G);
    let k_safe = k.max(0.0);
    let k_nl_safe = k_nl.max(0.0);
    let delta_v1_lin = if pb.v1_lin * pb.v1_lin < SVF_PHS_E_MAX {
        2.0 * k_nl_safe * pb.v1_lin * adj_k_eff_tone
    } else {
        0.0
    };
    let v3_a = u - s_lin_prev[1];
    let adj_a1_a = delta_v1_lin * s_lin_prev[0];
    let adj_a2_a = delta_v1_lin * v3_a;
    let extra_u = delta_v1_lin * pass_a_a2;
    let dl_da1_a = adj_a1_a + g_safe * adj_a2_a;
    let a1_a_sq = pass_a_a1 * pass_a_a1;
    let extra_g = pass_a_a1 * adj_a2_a - (2.0 * g_safe + k_safe) * a1_a_sq * dl_da1_a;
    let extra_k = adj_k_eff_tone - (g_safe * a1_a_sq) * dl_da1_a;
    let extra_k_nl = pb.e * adj_k_eff_tone;
    (extra_u, extra_g, extra_k, extra_k_nl)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn step_phs_pred_corr_reverse(
    fwd: &LaneForwardFrame,
    lp: &QuadLaneParams,
    pb_l: &PhsPredCorrPassBCoeffs,
    pb_r: &PhsPredCorrPassBCoeffs,
    dt: &DuhemTonePullback,
    carry: &mut ReverseCarry,
    acc: &mut TileGradAccum,
) -> (f32, f32) {
    let (lin_a1, lin_a2, lin_a3) = phs_pass_a_lin_coeffs(lp);
    let lam_b_l = carry.adj_s_out_l;
    let lam_b_r = carry.adj_s_out_r;
    let lam_lin_l = carry.adj_s_lin_out_l;
    let lam_lin_r = carry.adj_s_lin_out_r;

    let v1_lin_l = eval_phs_adj_v1_lin(
        pb_l,
        [fwd.s1_prev_l, fwd.s2_prev_l],
        fwd.u_l,
        lp.svf_g,
        lp.svf_k_nl,
        dt.adj_v1_l,
        dt.adj_v2_l,
        dt.adj_k_eff_tone_l,
        lam_b_l,
    );
    let v1_lin_r = eval_phs_adj_v1_lin(
        pb_r,
        [fwd.s1_prev_r, fwd.s2_prev_r],
        fwd.u_r,
        lp.svf_g,
        lp.svf_k_nl,
        dt.adj_v1_r,
        dt.adj_v2_r,
        dt.adj_k_eff_tone_r,
        lam_b_r,
    );

    let ctx_l = PhsPredCorrStepContext {
        u: fwd.u_l,
        s_lin_prev: fwd.s_lin_prev_l,
        s_b_prev: [fwd.s1_prev_l, fwd.s2_prev_l],
        g: lp.svf_g,
        k: lp.k,
        k_nl: lp.svf_k_nl,
        pass_a_a1: lin_a1,
        pass_a_a2: lin_a2,
        pass_a_a3: lin_a3,
    };
    let ctx_r = PhsPredCorrStepContext {
        u: fwd.u_r,
        s_lin_prev: fwd.s_lin_prev_r,
        s_b_prev: [fwd.s1_prev_r, fwd.s2_prev_r],
        g: lp.svf_g,
        k: lp.k,
        k_nl: lp.svf_k_nl,
        pass_a_a1: lin_a1,
        pass_a_a2: lin_a2,
        pass_a_a3: lin_a3,
    };
    let terms_l =
        eval_phs_pred_corr_step_adjoint(&ctx_l, pb_l, lam_b_l, lam_lin_l, dt.adj_v1_l, dt.adj_v2_l);
    let terms_r =
        eval_phs_pred_corr_step_adjoint(&ctx_r, pb_r, lam_b_r, lam_lin_r, dt.adj_v1_r, dt.adj_v2_r);
    let (ex_u_l, ex_g_l, ex_k_l, ex_knl_l) = eval_phs_k_eff_tone_extras(
        fwd.u_l,
        fwd.s_lin_prev_l,
        lp.svf_g,
        lp.k,
        lp.svf_k_nl,
        lin_a1,
        lin_a2,
        pb_l,
        dt.adj_k_eff_tone_l,
    );
    let (ex_u_r, ex_g_r, ex_k_r, ex_knl_r) = eval_phs_k_eff_tone_extras(
        fwd.u_r,
        fwd.s_lin_prev_r,
        lp.svf_g,
        lp.k,
        lp.svf_k_nl,
        lin_a1,
        lin_a2,
        pb_r,
        dt.adj_k_eff_tone_r,
    );

    acc.slots[quad_param_slots::SVF_K] += terms_l.adj_k + ex_k_l + terms_r.adj_k + ex_k_r;
    acc.slots[quad_param_slots::SVF_K_NL] +=
        terms_l.adj_k_nl + ex_knl_l + terms_r.adj_k_nl + ex_knl_r;
    acc.slots[quad_param_slots::SVF_G] += terms_l.adj_g + ex_g_l + terms_r.adj_g + ex_g_r;

    let elem_b_l = phs_pred_corr_pass_b_adjoint_elem(pb_l, dt.adj_v1_l, dt.adj_v2_l);
    let elem_b_r = phs_pred_corr_pass_b_adjoint_elem(pb_r, dt.adj_v1_r, dt.adj_v2_r);
    carry.adj_s_out_l = apply_tpt_adjoint(elem_b_l, lam_b_l);
    carry.adj_s_out_r = apply_tpt_adjoint(elem_b_r, lam_b_r);

    let elem_a_l = phs_pred_corr_pass_a_adjoint_elem(lin_a1, lin_a2, lin_a3, v1_lin_l);
    let elem_a_r = phs_pred_corr_pass_a_adjoint_elem(lin_a1, lin_a2, lin_a3, v1_lin_r);
    carry.adj_s_lin_out_l = apply_tpt_adjoint(elem_a_l, lam_lin_l);
    carry.adj_s_lin_out_r = apply_tpt_adjoint(elem_a_r, lam_lin_r);

    let _ = eval_phs_pred_corr_pass_b_coeffs;
    (
        dt.adj_u_nonrec_l + terms_l.adj_u + ex_u_l,
        dt.adj_u_nonrec_r + terms_r.adj_u + ex_u_r,
    )
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{
        CoreQuadV4ParamsStereo4, GpuVec4, KernelVec4, SOFT_RECT_EPS, SonaParaLruDuhemCoeffs4,
        SonaToneFoldedBlock0, SonaToneWeights4, TINY_CROSSING_EPS, TptSvfCoeffs4,
        prewarp_tan_pi_f_over_fs_f32,
    };

    use crate::cpu::core_and_adjoint::{run_adjoint_scan_cpu, run_fused_core_scan_cpu};
    use crate::gpu::abi::{
        AdjointScanUniforms, CORE_INPUT_VEC4_STRIDE, FEATURE_VEC4_STRIDE, FusedScanUniforms,
        QUAD_PARAMS_VEC4_STRIDE, TILE_FRAMES, quad_param_slots,
    };
    use crate::runner::types::{AdjointScanInput, FusedCoreScanInput, pack_core_quad_params};

    #[test]
    fn test_v4_phs_pred_corr_and_folded_tone_central_fd() {
        let g = GpuVec4::from_array([
            prewarp_tan_pi_f_over_fs_f32(0.06),
            prewarp_tan_pi_f_over_fs_f32(0.11),
            prewarp_tan_pi_f_over_fs_f32(0.17),
            prewarp_tan_pi_f_over_fs_f32(0.24),
        ]);
        let k = GpuVec4::from_array([0.45, 0.85, 1.15, 1.45]);
        let params = CoreQuadV4ParamsStereo4 {
            svf: TptSvfCoeffs4::from_gk(g, k),
            tone: SonaToneFoldedBlock0::from_weights(&SonaToneWeights4 {
                lp: GpuVec4::from_array([0.5, 0.3, -0.2, 0.1]),
                bp: GpuVec4::from_array([0.2, 0.4, 0.15, -0.1]),
                hp: GpuVec4::from_array([-0.15, 0.25, 0.35, 0.2]),
                ap: GpuVec4::from_array([0.1, -0.05, 0.12, -0.08]),
                dir: GpuVec4::from_array([0.2, 0.1, 0.05, 0.15]),
            }),
            lru: SonaParaLruDuhemCoeffs4::from_clamped(
                GpuVec4::splat(0.92),
                GpuVec4::splat(0.35),
                GpuVec4::splat(0.25),
                GpuVec4::splat(0.2),
                GpuVec4::splat(0.4),
                GpuVec4::splat(0.18),
            ),
            two_svf_g: g.add(g),
            svf_k_nl: GpuVec4::from_array([1.2, 0.6, 0.25, 2.5]),
            rect_sign: GpuVec4::from_array([1.0, -1.0, 1.0, -1.0]),
            g0: GpuVec4::splat(0.95),
            g_inf: GpuVec4::splat(1.2),
            d_pre: GpuVec4::splat(0.35),
            se_w_sq: GpuVec4::splat(0.4),
            se_d_bus: GpuVec4::splat(0.25),
            adaa_drive: GpuVec4::splat(2.1),
            lru_bias_drive: GpuVec4::splat(0.15),
            lin_mix: GpuVec4::splat(0.12),
            wavefolder_mix: GpuVec4::splat(0.3),
            glu_mix: GpuVec4::splat(0.35),
            out_gain: GpuVec4::splat(1.05),
        };
        let packed = pack_core_quad_params(&params);
        let n = TILE_FRAMES + 40;
        let num_tiles = n.div_ceil(TILE_FRAMES);
        let mut core_in = vec![[0.0_f32; 4]; n * CORE_INPUT_VEC4_STRIDE];
        for t in 0..n {
            let tf = t as f32;
            core_in[t * 2] = [0.6 * (0.07 * tf).sin(), 0.5 * (0.11 * tf).cos(), 0.4, -0.35];
            core_in[t * 2 + 1] = [
                -0.45 * (0.09 * tf).cos(),
                0.55 * (0.05 * tf).sin(),
                -0.3,
                0.4,
            ];
        }
        let fwd = run_fused_core_scan_cpu(&FusedCoreScanInput {
            uniforms: FusedScanUniforms {
                num_frames: n as u32,
                num_tiles: num_tiles as u32,
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
        let dl_feat = vec![[0.2_f32, -0.15, 0.12, -0.08]; n * FEATURE_VEC4_STRIDE];
        let adj = run_adjoint_scan_cpu(&AdjointScanInput {
            uniforms: AdjointScanUniforms {
                num_frames: n as u32,
                num_tiles: num_tiles as u32,
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

        let eps = 2.0e-3_f32;
        for &slot in &[
            quad_param_slots::SVF_K,
            quad_param_slots::SVF_K_NL,
            quad_param_slots::SVF_G,
            quad_param_slots::TONE_C_KBP,
        ] {
            let eval_loss = |delta: f32| -> f64 {
                let mut p_pert = packed;
                p_pert[slot][0] += delta;
                if matches!(slot, quad_param_slots::SVF_G | quad_param_slots::SVF_K) {
                    let g0 = p_pert[quad_param_slots::SVF_G][0];
                    let k0 = p_pert[quad_param_slots::SVF_K][0];
                    let c = TptSvfCoeffs4::from_gk(GpuVec4::splat(g0), GpuVec4::splat(k0));
                    p_pert[quad_param_slots::SVF_A1][0] = c.a1.to_array()[0];
                    p_pert[quad_param_slots::SVF_A2][0] = c.a2.to_array()[0];
                    p_pert[quad_param_slots::SVF_A3][0] = c.a3.to_array()[0];
                }
                let f = run_fused_core_scan_cpu(&FusedCoreScanInput {
                    uniforms: FusedScanUniforms {
                        num_frames: n as u32,
                        num_tiles: num_tiles as u32,
                        num_quads: 1,
                        num_stems: 1,
                        tile_start: 0,
                        soft_rect_eps: SOFT_RECT_EPS,
                        tiny_crossing_eps: TINY_CROSSING_EPS,
                        block_idx: 0,
                    },
                    core_inputs: &core_in,
                    quad_params: &p_pert,
                    initial_checkpoints: None,
                })
                .unwrap();
                f.feature_outputs
                    .iter()
                    .zip(dl_feat.iter())
                    .map(|(a, b)| {
                        (0..4)
                            .map(|j| f64::from(a[j]) * f64::from(b[j]))
                            .sum::<f64>()
                    })
                    .sum()
            };
            let fd = (eval_loss(eps) - eval_loss(-eps)) / f64::from(2.0 * eps);
            let got = f64::from(
                adj.param_grads[slot][0] + adj.param_grads[QUAD_PARAMS_VEC4_STRIDE + slot][0],
            );
            assert!(
                (got - fd).abs() <= 5.0e-3 * fd.abs().max(1.0),
                "Block 0 PHS slot {slot}: got={got} fd={fd}"
            );
        }
    }
}
