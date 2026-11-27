//! Block 0 linearly-implicit Port-Hamiltonian (`PHS`) `TPT-SVF` forward & adjoint pullback
//! for the `SONA v3.0` CPU bridge (`solver/` compatibility until `C3.1`).

use dnam_kernels::{
    GpuVec4, KernelVec4, MIN_TPT_G, SVF_PHS_E_MAX, TptAdjointElem, step_tpt_svf_phs_vec4,
};

use super::super::v3_quad_param_slots;
use super::{FwdFrameState, LaneParams, ReverseCarry, TileGradAccum};

#[derive(Copy, Clone, Default)]
pub(super) struct PhsFrameTerms {
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
    pub k_eff: f32,
    pub e: f32,
    pub gamma: f32,
    pub taps: [f32; 4],
}

#[inline(always)]
pub(super) fn step_tpt_svf_phs_scalar(
    u: f32,
    s1: &mut f32,
    s2: &mut f32,
    lp: &LaneParams,
) -> [f32; 4] {
    let mut s1_v = GpuVec4::splat(*s1);
    let mut s2_v = GpuVec4::splat(*s2);
    let out = step_tpt_svf_phs_vec4(
        GpuVec4::splat(u),
        &mut s1_v,
        &mut s2_v,
        GpuVec4::splat(lp.two_svf_g * 0.5),
        GpuVec4::splat(lp.k),
        GpuVec4::splat(lp.svf_k_nl),
    );
    *s1 = s1_v.to_array()[0];
    *s2 = s2_v.to_array()[0];
    [
        out.lp.to_array()[0],
        out.bp.to_array()[0],
        out.hp.to_array()[0],
        out.ap.to_array()[0],
    ]
}

#[inline(always)]
pub(super) fn eval_phs_frame_terms(
    u: f32,
    s1_prev: f32,
    s2_prev: f32,
    lp: &LaneParams,
) -> PhsFrameTerms {
    let g_safe = (lp.two_svf_g * 0.5).max(MIN_TPT_G);
    let k_safe = lp.k.max(0.0);
    let k_nl_safe = lp.svf_k_nl.max(0.0);
    let s1_sq = s1_prev * s1_prev;
    let e = s1_sq.min(SVF_PHS_E_MAX);
    let gamma = if s1_sq < SVF_PHS_E_MAX {
        (k_nl_safe + k_nl_safe) * s1_prev
    } else {
        0.0
    };
    let k_eff = k_safe + k_nl_safe * e;
    let denom = 1.0 + g_safe * (g_safe + k_eff);
    let a1 = GpuVec4::splat(denom).rcp_ge1_nr2().to_array()[0];
    let a2 = g_safe * a1;
    let a3 = g_safe * a2;
    let mut s1_tmp = s1_prev;
    let mut s2_tmp = s2_prev;
    let taps = step_tpt_svf_phs_scalar(u, &mut s1_tmp, &mut s2_tmp, lp);
    PhsFrameTerms {
        a1,
        a2,
        a3,
        k_eff,
        e,
        gamma,
        taps,
    }
}

#[inline(always)]
fn build_tpt_phs_elem(
    p: &PhsFrameTerms,
    adj_v1: f32,
    adj_v2: f32,
    k_eff_loc_bar: f32,
) -> TptAdjointElem {
    let two_a2 = p.a2 + p.a2;
    let two_a3 = p.a3 + p.a3;
    let v1_gamma = p.taps[1] * p.gamma;
    TptAdjointElem {
        t00: (p.a1 + p.a1 - 1.0) - two_a2 * v1_gamma,
        t01: two_a2 - two_a3 * v1_gamma,
        t10: -two_a2,
        t11: 1.0 - two_a3,
        g0: p.a1 * adj_v1 + p.a2 * adj_v2 + k_eff_loc_bar * p.gamma,
        g1: -p.a2 * adj_v1 + (1.0 - p.a3) * adj_v2,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn step_phs_tpt_reverse(
    fwd: &FwdFrameState,
    lp: &LaneParams,
    phs_l: &PhsFrameTerms,
    phs_r: &PhsFrameTerms,
    adj_v1: [f32; 2],
    adj_v2: [f32; 2],
    adj_hp: [f32; 2],
    adj_ap: [f32; 2],
    adj_u_nonrec: [f32; 2],
    carry: &mut ReverseCarry,
    acc: &mut TileGradAccum,
) -> (f32, f32) {
    let psi_l = carry.adj_s_out_l;
    let psi_r = carry.adj_s_out_r;
    let tot_v1_l = adj_v1[0] + 2.0 * psi_l[0];
    let tot_v2_l = adj_v2[0] + 2.0 * psi_l[1];
    let tot_v1_r = adj_v1[1] + 2.0 * psi_r[0];
    let tot_v2_r = adj_v2[1] + 2.0 * psi_r[1];

    let v1_l = phs_l.taps[1];
    let v1_r = phs_r.taps[1];
    let k_loc_l = (-adj_hp[0] - 2.0 * adj_ap[0]) * v1_l
        - v1_l * (phs_l.a2 * adj_v1[0] + phs_l.a3 * adj_v2[0]);
    let k_loc_r = (-adj_hp[1] - 2.0 * adj_ap[1]) * v1_r
        - v1_r * (phs_r.a2 * adj_v1[1] + phs_r.a3 * adj_v2[1]);

    let k_eff_bar_l = k_loc_l - 2.0 * v1_l * (phs_l.a2 * psi_l[0] + phs_l.a3 * psi_l[1]);
    let k_eff_bar_r = k_loc_r - 2.0 * v1_r * (phs_r.a2 * psi_r[0] + phs_r.a3 * psi_r[1]);
    acc.slots[v3_quad_param_slots::SVF_K] += k_eff_bar_l + k_eff_bar_r;
    acc.slots[v3_quad_param_slots::SVF_K_NL] += k_eff_bar_l * phs_l.e + k_eff_bar_r * phs_r.e;

    let two_g_safe = 2.0 * (lp.two_svf_g * 0.5).max(MIN_TPT_G);
    let v3_l = fwd.u_l - fwd.s2_prev_l;
    let v3_r = fwd.u_r - fwd.s2_prev_r;
    let g_tpt_l = tot_v2_l * v1_l
        + (phs_l.a1 * tot_v1_l + phs_l.a2 * tot_v2_l) * (v3_l - (two_g_safe + phs_l.k_eff) * v1_l);
    let g_tpt_r = tot_v2_r * v1_r
        + (phs_r.a1 * tot_v1_r + phs_r.a2 * tot_v2_r) * (v3_r - (two_g_safe + phs_r.k_eff) * v1_r);
    acc.slots[v3_quad_param_slots::SVF_G] += g_tpt_l + g_tpt_r;

    let du_l = adj_u_nonrec[0] + phs_l.a2 * tot_v1_l + phs_l.a3 * tot_v2_l;
    let du_r = adj_u_nonrec[1] + phs_r.a2 * tot_v1_r + phs_r.a3 * tot_v2_r;
    carry.adj_s_out_l = dnam_kernels::apply_tpt_adjoint(
        build_tpt_phs_elem(phs_l, adj_v1[0], adj_v2[0], k_loc_l),
        psi_l,
    );
    carry.adj_s_out_r = dnam_kernels::apply_tpt_adjoint(
        build_tpt_phs_elem(phs_r, adj_v1[1], adj_v2[1], k_loc_r),
        psi_r,
    );
    (du_l, du_r)
}
