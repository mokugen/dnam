//! `O(log2 N)` 2-pass associative scan and reverse-BPTT adjoint helpers for Block `0`
//! 1-step Predictor-Corrector Passive Port-Hamiltonian (`PHS`) `TPT-SVF`
//! (`ref[dsp.op-n.core.tpt-tone-basis]`, `ref[dsp.op-m.recurrence.symplectic]`,
//! `ref[arch.operators.differentiability]`).
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`,
//! and `spirv-unknown-naga-wgsl`-clean.

use crate::simd::{GpuVec4, KernelVec4};

use super::super::{MIN_TPT_G, TptAdjointElem, TptScanElem, TptSvfCoeffs4};
use super::SVF_PHS_E_MAX;

/// Pass B time-varying coefficients derived from Pass A linear predictor `v1_lin[n]`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PhsPredCorrPassBCoeffs {
    pub v1_lin: f32,
    pub e: f32,
    pub k_eff: f32,
    pub a1: f32,
    pub a2: f32,
    pub a3: f32,
}

/// Evaluates Pass B passive `TPT-SVF` coefficients `(a1, a2, a3)` from linear predictor output
/// `v1_lin` using `rcp_ge1_nr2` (zero `FDIV`, `ref[dsp.op-n.core.tpt-tone-basis]`).
#[inline(always)]
#[must_use]
pub fn eval_phs_pred_corr_pass_b_coeffs(
    v1_lin: f32,
    g: f32,
    k: f32,
    k_nl: f32,
) -> PhsPredCorrPassBCoeffs {
    let g_safe = if g > MIN_TPT_G { g } else { MIN_TPT_G };
    let k_safe = if k > 0.0 { k } else { 0.0 };
    let k_nl_safe = if k_nl > 0.0 { k_nl } else { 0.0 };
    let e_raw = v1_lin * v1_lin;
    let e = if e_raw < SVF_PHS_E_MAX {
        e_raw
    } else {
        SVF_PHS_E_MAX
    };
    let k_eff = k_safe + k_nl_safe * e;
    let denom = 1.0 + g_safe * (g_safe + k_eff);
    let a1 = GpuVec4::splat(denom).rcp_ge1_nr2().to_array()[0];
    let a2 = g_safe * a1;
    let a3 = g_safe * a2;
    PhsPredCorrPassBCoeffs {
        v1_lin,
        e,
        k_eff,
        a1,
        a2,
        a3,
    }
}

/// Builds a Pass B forward [`TptScanElem`] from Pass A predictor output `v1_lin` and input `u`.
#[inline(always)]
#[must_use]
pub fn phs_pred_corr_pass_b_scan_elem(pass_b: &PhsPredCorrPassBCoeffs, u: f32) -> TptScanElem {
    TptScanElem::from_step(pass_b.a1, pass_b.a2, pass_b.a3, u)
}

/// Builds a Pass B reverse-BPTT [`TptAdjointElem`] pulling output sensitivities `(adj_bp, adj_lp)`
/// back onto `s_b[n-1]` (`ref[arch.operators.differentiability]`).
#[inline(always)]
#[must_use]
pub fn phs_pred_corr_pass_b_adjoint_elem(
    pass_b: &PhsPredCorrPassBCoeffs,
    adj_bp: f32,
    adj_lp: f32,
) -> TptAdjointElem {
    let two_a2 = pass_b.a2 + pass_b.a2;
    let two_a3 = pass_b.a3 + pass_b.a3;
    let t00 = (pass_b.a1 + pass_b.a1) - 1.0;
    let t01 = two_a2;
    let t10 = -two_a2;
    let t11 = 1.0 - two_a3;
    TptAdjointElem {
        t00,
        t01,
        t10,
        t11,
        g0: pass_b.a1 * adj_bp + pass_b.a2 * adj_lp,
        g1: -pass_b.a2 * adj_bp + (1.0 - pass_b.a3) * adj_lp,
    }
}

/// Builds a Pass A linear reverse-BPTT [`TptAdjointElem`] pulling `adj_v1_lin` back onto `s_lin[n-1]`.
#[inline(always)]
#[must_use]
pub fn phs_pred_corr_pass_a_adjoint_elem(
    a1_lin: f32,
    a2_lin: f32,
    a3_lin: f32,
    adj_v1_lin: f32,
) -> TptAdjointElem {
    let two_a2 = a2_lin + a2_lin;
    let two_a3 = a3_lin + a3_lin;
    TptAdjointElem {
        t00: (a1_lin + a1_lin) - 1.0,
        t01: two_a2,
        t10: -two_a2,
        t11: 1.0 - two_a3,
        g0: a1_lin * adj_v1_lin,
        g1: -a2_lin * adj_v1_lin,
    }
}

/// Single-step backward parameter and input adjoints for Predictor-Corrector Passive `PHS`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PhsPredCorrAdjointTerms {
    pub adj_u: f32,
    pub adj_g: f32,
    pub adj_k: f32,
    pub adj_k_nl: f32,
    pub adj_v1_lin: f32,
}

/// Context bundle for [`eval_phs_pred_corr_step_adjoint`], keeping signatures concise and typed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PhsPredCorrStepContext {
    pub u: f32,
    pub s_lin_prev: [f32; 2],
    pub s_b_prev: [f32; 2],
    pub g: f32,
    pub k: f32,
    pub k_nl: f32,
    pub pass_a_a1: f32,
    pub pass_a_a2: f32,
    pub pass_a_a3: f32,
}

/// Computes Pass B -> Pass A step gradients `(adj_u, adj_g, adj_k, adj_k_nl, adj_v1_lin)` given
/// exclusive future suffix adjoints `lam_b_post` (`dL_{>n}/ds_b[n]`) and `lam_lin_post`
/// (`dL_{>n}/ds_lin[n]`).
#[inline(always)]
#[must_use]
pub fn eval_phs_pred_corr_step_adjoint(
    ctx: &PhsPredCorrStepContext,
    pass_b: &PhsPredCorrPassBCoeffs,
    lam_b_post: [f32; 2],
    lam_lin_post: [f32; 2],
    adj_bp_out: f32,
    adj_lp_out: f32,
) -> PhsPredCorrAdjointTerms {
    let g_safe = if ctx.g > MIN_TPT_G { ctx.g } else { MIN_TPT_G };
    let k_safe = if ctx.k > 0.0 { ctx.k } else { 0.0 };
    let k_nl_safe = if ctx.k_nl > 0.0 { ctx.k_nl } else { 0.0 };

    // Pass B pullback
    let adj_v1_b = adj_bp_out + 2.0 * lam_b_post[0];
    let adj_v2_b = adj_lp_out + 2.0 * lam_b_post[1];
    let v3_b = ctx.u - ctx.s_b_prev[1];
    let adj_a1_b = adj_v1_b * ctx.s_b_prev[0];
    let adj_a2_b = adj_v1_b * v3_b + adj_v2_b * ctx.s_b_prev[0];
    let adj_a3_b = adj_v2_b * v3_b;
    let adj_u_b = adj_v1_b * pass_b.a2 + adj_v2_b * pass_b.a3;

    let dl_da1_b = adj_a1_b + g_safe * adj_a2_b + (g_safe * g_safe) * adj_a3_b;
    let a1_b_sq = pass_b.a1 * pass_b.a1;
    let adj_g_b = pass_b.a1 * adj_a2_b + 2.0 * pass_b.a2 * adj_a3_b
        - (2.0 * g_safe + pass_b.k_eff) * a1_b_sq * dl_da1_b;
    let adj_k_eff = -(g_safe * a1_b_sq) * dl_da1_b;
    let adj_k_nl = pass_b.e * adj_k_eff;
    let adj_k_b = adj_k_eff;

    // Coupling k_eff = k + k_nl * min(v1_lin^2, 64.0)
    let adj_v1_lin = if pass_b.v1_lin * pass_b.v1_lin < SVF_PHS_E_MAX {
        2.0 * k_nl_safe * pass_b.v1_lin * adj_k_eff
    } else {
        0.0
    };

    // Pass A pullback
    let adj_v1_a = adj_v1_lin + 2.0 * lam_lin_post[0];
    let adj_v2_a = 2.0 * lam_lin_post[1];
    let v3_a = ctx.u - ctx.s_lin_prev[1];
    let adj_a1_a = adj_v1_a * ctx.s_lin_prev[0];
    let adj_a2_a = adj_v1_a * v3_a + adj_v2_a * ctx.s_lin_prev[0];
    let adj_a3_a = adj_v2_a * v3_a;
    let adj_u_a = adj_v1_a * ctx.pass_a_a2 + adj_v2_a * ctx.pass_a_a3;

    let dl_da1_a = adj_a1_a + g_safe * adj_a2_a + (g_safe * g_safe) * adj_a3_a;
    let a1_a_sq = ctx.pass_a_a1 * ctx.pass_a_a1;
    let adj_g_a = ctx.pass_a_a1 * adj_a2_a + 2.0 * ctx.pass_a_a2 * adj_a3_a
        - (2.0 * g_safe + k_safe) * a1_a_sq * dl_da1_a;
    let adj_k_a = -(g_safe * a1_a_sq) * dl_da1_a;

    PhsPredCorrAdjointTerms {
        adj_u: adj_u_a + adj_u_b,
        adj_g: adj_g_a + adj_g_b,
        adj_k: adj_k_a + adj_k_b,
        adj_k_nl,
        adj_v1_lin,
    }
}

/// Evaluates Pass A linear `TPT-SVF` scan element from precomputed scalar `(a1, a2, a3)`.
#[inline(always)]
#[must_use]
pub fn phs_pred_corr_pass_a_scan_elem(coeffs: &TptSvfCoeffs4<GpuVec4>, u: f32) -> TptScanElem {
    TptScanElem::from_step(
        coeffs.a1.to_array()[0],
        coeffs.a2.to_array()[0],
        coeffs.a3.to_array()[0],
        u,
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::{apply_tpt_init, compose_tpt_adjoint_scan, compose_tpt_scan};
    use super::super::{TptSvfPhsPredCorrState, step_tpt_svf_pred_corr_phs_vec4};
    use super::*;
    use crate::simd::KernelVec4;
    use proptest::prelude::*;

    fn pred_corr_forward_loss_f64(
        u: &[f64],
        s0_lin: [f64; 2],
        s0_b: [f64; 2],
        gk_knl: [f64; 3],
        w_bp: &[f64],
        w_lp: &[f64],
    ) -> f64 {
        let [g, k, k_nl] = gk_knl;
        let mut s1_a = s0_lin[0];
        let mut s2_a = s0_lin[1];
        let mut s1_b = s0_b[0];
        let mut s2_b = s0_b[1];
        let a1_0 = 1.0 / (1.0 + g * (g + k));
        let a2_0 = g * a1_0;
        let a3_0 = g * a2_0;
        let mut loss = 0.0;
        for t in 0..u.len() {
            let v3_a = u[t] - s2_a;
            let v1_lin = a1_0 * s1_a + a2_0 * v3_a;
            let v2_a = s2_a + a2_0 * s1_a + a3_0 * v3_a;
            s1_a = 2.0 * v1_lin - s1_a;
            s2_a = 2.0 * v2_a - s2_a;

            let e = (v1_lin * v1_lin).min(64.0);
            let k_eff = k + k_nl * e;
            let a1_b = 1.0 / (1.0 + g * (g + k_eff));
            let a2_b = g * a1_b;
            let a3_b = g * a2_b;
            let v3_b = u[t] - s2_b;
            let bp = a1_b * s1_b + a2_b * v3_b;
            let lp = s2_b + a2_b * s1_b + a3_b * v3_b;
            s1_b = 2.0 * bp - s1_b;
            s2_b = 2.0 * lp - s2_b;
            loss += w_bp[t] * bp + w_lp[t] * lp;
        }
        loss
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// verify[dsp.op-n.core.tpt-tone-basis]
        /// verify[arch.operators.differentiability]
        #[test]
        fn prop_phs_pred_corr_two_pass_scan_and_adjoint_fd(
            g in 0.08_f32..2.5_f32,
            k in 0.1_f32..2.0_f32,
            k_nl in 0.2_f32..6.0_f32,
            u_seq in proptest::array::uniform8(-2.0_f32..2.0_f32),
            w_bp in proptest::array::uniform8(-1.0_f32..1.0_f32),
            w_lp in proptest::array::uniform8(-1.0_f32..1.0_f32),
            s0_lin in proptest::array::uniform2(-0.5_f32..0.5_f32),
            s0_b in proptest::array::uniform2(-0.5_f32..0.5_f32),
        ) {
            let coeffs_a = TptSvfCoeffs4::from_gk(GpuVec4::splat(g), GpuVec4::splat(k));
            let (a1_0, a2_0, a3_0) = (
                coeffs_a.a1.to_array()[0],
                coeffs_a.a2.to_array()[0],
                coeffs_a.a3.to_array()[0],
            );

            // Sequential forward via step_tpt_svf_pred_corr_phs_vec4
            let mut st_seq = TptSvfPhsPredCorrState {
                s1_lin: GpuVec4::splat(s0_lin[0]),
                s2_lin: GpuVec4::splat(s0_lin[1]),
                s1: GpuVec4::splat(s0_b[0]),
                s2: GpuVec4::splat(s0_b[1]),
            };
            let mut bp_seq = [0.0_f32; 8];
            let mut lp_seq = [0.0_f32; 8];
            for t in 0..8 {
                let o = step_tpt_svf_pred_corr_phs_vec4(
                    GpuVec4::splat(u_seq[t]),
                    &mut st_seq,
                    GpuVec4::splat(g),
                    GpuVec4::splat(k),
                    GpuVec4::splat(k_nl),
                );
                bp_seq[t] = o.bp.to_array()[0];
                lp_seq[t] = o.lp.to_array()[0];
            }

            // O(log2 N) 2-pass binary-tree scan: Pass A -> v1_lin -> Pass B
            let elems_a: [TptScanElem; 8] =
                core::array::from_fn(|t| phs_pred_corr_pass_a_scan_elem(&coeffs_a, u_seq[t]));
            let mut s_lin_hist = [[0.0_f32; 2]; 8];
            let mut acc_a = TptScanElem::identity();
            let mut v1_lin = [0.0_f32; 8];
            let mut pass_b_list = [eval_phs_pred_corr_pass_b_coeffs(0.0, g, k, k_nl); 8];
            for t in 0..8 {
                let s_prev = apply_tpt_init(acc_a, s0_lin);
                s_lin_hist[t] = s_prev;
                v1_lin[t] = a1_0 * s_prev[0] + a2_0 * (u_seq[t] - s_prev[1]);
                pass_b_list[t] = eval_phs_pred_corr_pass_b_coeffs(v1_lin[t], g, k, k_nl);
                acc_a = compose_tpt_scan(acc_a, elems_a[t]);
            }

            // Binary-tree reduction on Pass B elems
            let elems_b: [TptScanElem; 8] =
                core::array::from_fn(|t| phs_pred_corr_pass_b_scan_elem(&pass_b_list[t], u_seq[t]));
            let p1 = core::array::from_fn::<_, 4, _>(|i| compose_tpt_scan(elems_b[2 * i], elems_b[2 * i + 1]));
            let p2 = core::array::from_fn::<_, 2, _>(|i| compose_tpt_scan(p1[2 * i], p1[2 * i + 1]));
            let tree_b = compose_tpt_scan(p2[0], p2[1]);
            let final_b = apply_tpt_init(tree_b, s0_b);
            prop_assert!((final_b[0] - st_seq.s1.to_array()[0]).abs() < 1.0e-4);
            prop_assert!((final_b[1] - st_seq.s2.to_array()[0]).abs() < 1.0e-4);

            let mut s_b_hist = [[0.0_f32; 2]; 8];
            let mut acc_b = TptScanElem::identity();
            for t in 0..8 {
                let sb_prev = apply_tpt_init(acc_b, s0_b);
                s_b_hist[t] = sb_prev;
                let pb = &pass_b_list[t];
                let bp = pb.a1 * sb_prev[0] + pb.a2 * (u_seq[t] - sb_prev[1]);
                let lp = sb_prev[1] + pb.a2 * sb_prev[0] + pb.a3 * (u_seq[t] - sb_prev[1]);
                prop_assert!((bp - bp_seq[t]).abs() < 1.0e-4);
                prop_assert!((lp - lp_seq[t]).abs() < 1.0e-4);
                acc_b = compose_tpt_scan(acc_b, elems_b[t]);
            }

            // Reverse Pass B scan -> coupling -> Reverse Pass A scan
            let mut adj_u_total = [0.0_f32; 8];
            let mut adj_g_total = 0.0_f32;
            let mut adj_k_total = 0.0_f32;
            let mut adj_knl_total = 0.0_f32;
            let mut lam_b_post = [[0.0_f32; 2]; 8];
            let mut rev_b = TptAdjointElem::identity();
            for rev in 0..8 {
                let t = 7 - rev;
                lam_b_post[t] = [rev_b.g0, rev_b.g1];
                let eb = phs_pred_corr_pass_b_adjoint_elem(&pass_b_list[t], w_bp[t], w_lp[t]);
                rev_b = compose_tpt_adjoint_scan(eb, rev_b);
            }

            let mut rev_a = TptAdjointElem::identity();
            for rev in 0..8 {
                let t = 7 - rev;
                let terms = eval_phs_pred_corr_step_adjoint(
                    &PhsPredCorrStepContext {
                        u: u_seq[t],
                        s_lin_prev: s_lin_hist[t],
                        s_b_prev: s_b_hist[t],
                        g,
                        k,
                        k_nl,
                        pass_a_a1: a1_0,
                        pass_a_a2: a2_0,
                        pass_a_a3: a3_0,
                    },
                    &pass_b_list[t],
                    lam_b_post[t],
                    [rev_a.g0, rev_a.g1],
                    w_bp[t],
                    w_lp[t],
                );
                adj_u_total[t] = terms.adj_u;
                adj_g_total += terms.adj_g;
                adj_k_total += terms.adj_k;
                adj_knl_total += terms.adj_k_nl;
                let ea = phs_pred_corr_pass_a_adjoint_elem(a1_0, a2_0, a3_0, terms.adj_v1_lin);
                rev_a = compose_tpt_adjoint_scan(ea, rev_a);
            }

            // Verify against f64 central finite differences (< 1e-4 / < 1e-6 when avoiding kink)
            let u64 = u_seq.map(f64::from);
            let wbp64 = w_bp.map(f64::from);
            let wlp64 = w_lp.map(f64::from);
            let s0_a64 = s0_lin.map(f64::from);
            let s0_b64 = s0_b.map(f64::from);
            let (g64, k64, knl64) = (f64::from(g), f64::from(k), f64::from(k_nl));
            let eps = 1.0e-6_f64;

            let fd_g = (pred_corr_forward_loss_f64(
                &u64,
                s0_a64,
                s0_b64,
                [g64 + eps, k64, knl64],
                &wbp64,
                &wlp64,
            ) - pred_corr_forward_loss_f64(
                &u64,
                s0_a64,
                s0_b64,
                [g64 - eps, k64, knl64],
                &wbp64,
                &wlp64,
            )) / (2.0 * eps);
            let fd_k = (pred_corr_forward_loss_f64(
                &u64,
                s0_a64,
                s0_b64,
                [g64, k64 + eps, knl64],
                &wbp64,
                &wlp64,
            ) - pred_corr_forward_loss_f64(
                &u64,
                s0_a64,
                s0_b64,
                [g64, k64 - eps, knl64],
                &wbp64,
                &wlp64,
            )) / (2.0 * eps);
            let fd_knl = (pred_corr_forward_loss_f64(
                &u64,
                s0_a64,
                s0_b64,
                [g64, k64, knl64 + eps],
                &wbp64,
                &wlp64,
            ) - pred_corr_forward_loss_f64(
                &u64,
                s0_a64,
                s0_b64,
                [g64, k64, knl64 - eps],
                &wbp64,
                &wlp64,
            )) / (2.0 * eps);
            prop_assert!(
                (f64::from(adj_g_total) - fd_g).abs() <= 1.5e-3 * fd_g.abs().max(1.0),
                "g: {} vs {}",
                adj_g_total,
                fd_g
            );
            prop_assert!(
                (f64::from(adj_k_total) - fd_k).abs() <= 1.5e-3 * fd_k.abs().max(1.0),
                "k: {} vs {}",
                adj_k_total,
                fd_k
            );
            prop_assert!(
                (f64::from(adj_knl_total) - fd_knl).abs() <= 1.5e-3 * fd_knl.abs().max(1.0),
                "knl: {} vs {}",
                adj_knl_total,
                fd_knl
            );

            for t in 0..8 {
                let mut up = u64;
                let mut um = u64;
                up[t] += eps;
                um[t] -= eps;
                let fd_ut = (pred_corr_forward_loss_f64(
                    &up,
                    s0_a64,
                    s0_b64,
                    [g64, k64, knl64],
                    &wbp64,
                    &wlp64,
                ) - pred_corr_forward_loss_f64(
                    &um,
                    s0_a64,
                    s0_b64,
                    [g64, k64, knl64],
                    &wbp64,
                    &wlp64,
                )) / (2.0 * eps);
                prop_assert!(
                    (f64::from(adj_u_total[t]) - fd_ut).abs() <= 1.5e-3 * fd_ut.abs().max(1.0)
                );
            }
        }
    }
}
