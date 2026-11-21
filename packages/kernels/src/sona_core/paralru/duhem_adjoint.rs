//! Reverse-BPTT `O(log2 N)` joint `(L, R)` parallel suffix adjoint scan for Selective-Duhem
//! stereo-linked `ParaLRU` (`ref[dsp.op-n.core.matryoshka-blocks]`,
//! `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::simd::scrub_and_flush_f32_bits;

use super::adjoint::{compose_paralru_adjoint_scan, parallel_suffix_scan};
use super::duhem_scans::{ParaLruDuhemForwardOut, ParaLruDuhemParams};
use super::scans::{LinearScanElem, apply_linear_scan};
use super::{ONE_THIRD_F32, soft_rect_zero_dc_f32};

/// Borrowed trajectory inputs for a joint `(L, R)` Selective-Duhem `ParaLRU` adjoint tile scan.
#[derive(Copy, Clone, Debug)]
pub struct ParaLruDuhemAdjointTrace<'a, const N: usize> {
    pub u: [&'a [f32; N]; 2],
    pub bp: [&'a [f32; N]; 2],
    pub v_traj: [&'a [f32; N]; 2],
    pub v_init: [f32; 2],
}

impl<'a, const N: usize> ParaLruDuhemAdjointTrace<'a, N> {
    /// Builds an adjoint trace directly from inputs and a [`ParaLruDuhemForwardOut`] record.
    #[inline]
    #[must_use]
    pub fn from_forward(
        u: [&'a [f32; N]; 2],
        bp: [&'a [f32; N]; 2],
        v_init: [f32; 2],
        fwd: &'a ParaLruDuhemForwardOut<N>,
    ) -> Self {
        Self {
            u,
            bp,
            v_traj: [&fwd.v_traj_l, &fwd.v_traj_r],
            v_init,
        }
    }
}

/// Reverse-BPTT gradients and cross-tile summaries computed over an `N`-sample joint `(L, R)`
/// Selective-Duhem `ParaLRU` tile (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParaLruDuhemTileGrads<const N: usize> {
    pub adj_u_l: [f32; N],
    pub adj_u_r: [f32; N],
    pub adj_bp_l: [f32; N],
    pub adj_bp_r: [f32; N],
    pub adj_v_init_l: f32,
    pub adj_v_init_r: f32,
    pub adj_alpha_rel: f32,
    pub adj_alpha_atk: f32,
    pub adj_kappa: f32,
    pub adj_gamma_duhem: f32,
    pub adj_signed_mix: f32,
    pub adj_stereo_link: f32,
    pub lambda_l: [f32; N],
    pub lambda_r: [f32; N],
    pub tile_summary_l: LinearScanElem,
    pub tile_summary_r: LinearScanElem,
}

#[derive(Copy, Clone)]
struct DuhemStepTerms {
    flux_abs: f32,
    rho: f32,
    rho_open: bool,
    d: f32,
    one_minus_eta: f32,
    phi_rect: f32,
    psi_1: f32,
    phi_eta: f32,
    phi_eta_prime: f32,
    jac_div_rel: f32,
    jac: f32,
    sign_bp: f32,
}

#[inline(always)]
fn eval_duhem_step_terms(
    u: f32,
    bp: f32,
    v_prev: f32,
    params: ParaLruDuhemParams,
) -> DuhemStepTerms {
    let u_clean = scrub_and_flush_f32_bits(u) + 0.0;
    let bp_clean = scrub_and_flush_f32_bits(bp) + 0.0;
    let v_clean = scrub_and_flush_f32_bits(v_prev) + 0.0;
    let flux_abs = scrub_and_flush_f32_bits(params.two_svf_g * bp_clean).abs();
    let duhem_mod = if params.gamma_duhem <= 0.0 {
        0.0
    } else {
        params.gamma_duhem * flux_abs
    };
    let rho = ((1.0 - params.alpha_atk) + duhem_mod).clamp(0.0, 1.0);
    let rho_open = rho > 0.0 && rho < 1.0;

    let d = params.alpha_rel * v_clean;
    let one_minus_eta = 1.0 - params.signed_mix;
    let z = u_clean - params.kappa * one_minus_eta * d;
    let (phi_rect, phi_rect_prime) = soft_rect_zero_dc_f32(z);
    let z_bar = z.clamp(-1.0, 1.0);
    let psi_1 = z_bar * (1.0 - ONE_THIRD_F32 * (z_bar * z_bar));
    let psi_1_prime = if z.abs() < 1.0 { 1.0 - z * z } else { 0.0 };
    let (phi_eta, phi_eta_prime) = if params.signed_mix <= 0.0 {
        (phi_rect, phi_rect_prime)
    } else {
        (
            one_minus_eta * phi_rect + params.signed_mix * psi_1,
            one_minus_eta * phi_rect_prime + params.signed_mix * psi_1_prime,
        )
    };
    let jac_div_rel =
        (1.0 - params.signed_mix * rho) - rho * params.kappa * one_minus_eta * phi_eta_prime;
    let jac = params.alpha_rel * jac_div_rel;
    let sign_bp = if bp_clean > 0.0 {
        1.0
    } else if bp_clean < 0.0 {
        -1.0
    } else {
        0.0
    };
    DuhemStepTerms {
        flux_abs,
        rho,
        rho_open,
        d,
        one_minus_eta,
        phi_rect,
        psi_1,
        phi_eta,
        phi_eta_prime,
        jac_div_rel,
        jac,
        sign_bp,
    }
}

/// Forms the reverse-BPTT linear adjoint scan element `(J_n, J_n * adj_v_n)` at step `n`
/// for Selective-Duhem `ParaLRU` (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline(always)]
#[must_use]
pub fn paralru_duhem_adjoint_step_elem(
    u: f32,
    bp: f32,
    v_prev: f32,
    params: ParaLruDuhemParams,
    adj_v: f32,
) -> LinearScanElem {
    let jac = eval_duhem_step_terms(u, bp, v_prev, params).jac;
    LinearScanElem {
        a: jac,
        b: jac * adj_v,
    }
}

struct ChannelDuhemGrads<const N: usize> {
    adj_u: [f32; N],
    adj_bp: [f32; N],
    lambda: [f32; N],
    adj_v_init: f32,
    adj_alpha_rel: f32,
    adj_alpha_atk: f32,
    adj_kappa: f32,
    adj_gamma_duhem: f32,
    adj_signed_mix: f32,
    tile_summary: LinearScanElem,
}

#[inline]
fn run_channel_duhem_adjoint_scan<const N: usize>(
    trace: &ParaLruDuhemAdjointTrace<'_, N>,
    ch: usize,
    adj_v: &[f32; N],
    adj_v_out: f32,
    active: usize,
    params: ParaLruDuhemParams,
) -> ChannelDuhemGrads<N> {
    let u = trace.u[ch];
    let bp = trace.bp[ch];
    let v_traj = trace.v_traj[ch];
    let v_init = trace.v_init[ch];
    let mut adj_u = [0.0_f32; N];
    let mut adj_bp = [0.0_f32; N];
    let mut lambda = [0.0_f32; N];
    let mut adj_elems = [LinearScanElem::identity(); N];
    for i in 0..active {
        let v_prev = if i == 0 { v_init } else { v_traj[i - 1] };
        adj_elems[i] = paralru_duhem_adjoint_step_elem(u[i], bp[i], v_prev, params, adj_v[i]);
    }
    parallel_suffix_scan(&mut adj_elems, compose_paralru_adjoint_scan);

    let mut adj_alpha_rel = 0.0_f32;
    let mut adj_alpha_atk = 0.0_f32;
    let mut adj_kappa = 0.0_f32;
    let mut adj_gamma_duhem = 0.0_f32;
    let mut adj_signed_mix = 0.0_f32;

    for i in 0..active {
        let v_prev = if i == 0 { v_init } else { v_traj[i - 1] };
        let v_clean = scrub_and_flush_f32_bits(v_prev) + 0.0;
        let suffix_after = if i + 1 < active {
            adj_elems[i + 1]
        } else {
            LinearScanElem::identity()
        };
        let lam_i = adj_v[i] + apply_linear_scan(suffix_after, adj_v_out);
        lambda[i] = lam_i;

        let t = eval_duhem_step_terms(u[i], bp[i], v_clean, params);
        adj_u[i] = lam_i * t.rho * t.phi_eta_prime;
        let adj_rho = lam_i * (t.phi_eta - params.signed_mix * t.d);
        if t.rho_open {
            adj_alpha_atk -= adj_rho;
            adj_gamma_duhem += adj_rho * t.flux_abs;
            if params.gamma_duhem > 0.0 {
                adj_bp[i] = adj_rho * params.two_svf_g * params.gamma_duhem * t.sign_bp;
            }
        }
        adj_alpha_rel += lam_i * v_clean * t.jac_div_rel;
        adj_kappa -= lam_i * t.rho * t.one_minus_eta * t.d * t.phi_eta_prime;
        adj_signed_mix +=
            lam_i * t.rho * ((t.psi_1 - t.phi_rect) + params.kappa * t.d * t.phi_eta_prime - t.d);
    }

    let tile_summary = adj_elems[0];
    ChannelDuhemGrads {
        adj_u,
        adj_bp,
        lambda,
        adj_v_init: apply_linear_scan(tile_summary, adj_v_out),
        adj_alpha_rel,
        adj_alpha_atk,
        adj_kappa,
        adj_gamma_duhem,
        adj_signed_mix,
        tile_summary,
    }
}

/// Executes the `O(log2 N)` joint `(L, R)` reverse-time inclusive suffix adjoint scan over
/// `valid_len.min(N)` active frames with stereo-link transpose pullback and Selective-Duhem
/// `adj_bp` accumulation (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline]
#[must_use]
pub fn run_paralru_duhem_adjoint_scan_partial_tile<const N: usize>(
    trace: &ParaLruDuhemAdjointTrace<'_, N>,
    valid_len: usize,
    params: ParaLruDuhemParams,
    adj_veff: [&[f32; N]; 2],
    adj_v_out: [f32; 2],
) -> ParaLruDuhemTileGrads<N> {
    let active = valid_len.min(N);
    if active == 0 {
        return ParaLruDuhemTileGrads {
            adj_u_l: [0.0; N],
            adj_u_r: [0.0; N],
            adj_bp_l: [0.0; N],
            adj_bp_r: [0.0; N],
            adj_v_init_l: adj_v_out[0],
            adj_v_init_r: adj_v_out[1],
            adj_alpha_rel: 0.0,
            adj_alpha_atk: 0.0,
            adj_kappa: 0.0,
            adj_gamma_duhem: 0.0,
            adj_signed_mix: 0.0,
            adj_stereo_link: 0.0,
            lambda_l: [0.0; N],
            lambda_r: [0.0; N],
            tile_summary_l: LinearScanElem::identity(),
            tile_summary_r: LinearScanElem::identity(),
        };
    }

    let mut adj_v_l = [0.0_f32; N];
    let mut adj_v_r = [0.0_f32; N];
    let mut adj_stereo_link = 0.0_f32;
    let half_lam = 0.5 * params.stereo_link;
    let one_minus_half_lam = 1.0 - half_lam;

    for i in 0..active {
        let vl = trace.v_traj[0][i];
        let vr = trace.v_traj[1][i];
        let v_mid = 0.5 * (vl + vr);
        let ae_l = adj_veff[0][i];
        let ae_r = adj_veff[1][i];
        adj_v_l[i] = one_minus_half_lam * ae_l + half_lam * ae_r;
        adj_v_r[i] = one_minus_half_lam * ae_r + half_lam * ae_l;
        adj_stereo_link += ae_l * (v_mid - vl) + ae_r * (v_mid - vr);
    }

    let gl = run_channel_duhem_adjoint_scan(trace, 0, &adj_v_l, adj_v_out[0], active, params);
    let gr = run_channel_duhem_adjoint_scan(trace, 1, &adj_v_r, adj_v_out[1], active, params);

    ParaLruDuhemTileGrads {
        adj_u_l: gl.adj_u,
        adj_u_r: gr.adj_u,
        adj_bp_l: gl.adj_bp,
        adj_bp_r: gr.adj_bp,
        adj_v_init_l: gl.adj_v_init,
        adj_v_init_r: gr.adj_v_init,
        adj_alpha_rel: gl.adj_alpha_rel + gr.adj_alpha_rel,
        adj_alpha_atk: gl.adj_alpha_atk + gr.adj_alpha_atk,
        adj_kappa: gl.adj_kappa + gr.adj_kappa,
        adj_gamma_duhem: gl.adj_gamma_duhem + gr.adj_gamma_duhem,
        adj_signed_mix: gl.adj_signed_mix + gr.adj_signed_mix,
        adj_stereo_link,
        lambda_l: gl.lambda,
        lambda_r: gr.lambda,
        tile_summary_l: gl.tile_summary,
        tile_summary_r: gr.tile_summary,
    }
}

/// Executes the `O(log2 N)` joint `(L, R)` reverse-time inclusive suffix adjoint scan over all `N`
/// frames (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline]
#[must_use]
pub fn run_paralru_duhem_adjoint_scan_tile<const N: usize>(
    trace: &ParaLruDuhemAdjointTrace<'_, N>,
    params: ParaLruDuhemParams,
    adj_veff: [&[f32; N]; 2],
    adj_v_out: [f32; 2],
) -> ParaLruDuhemTileGrads<N> {
    run_paralru_duhem_adjoint_scan_partial_tile(trace, N, params, adj_veff, adj_v_out)
}

#[cfg(test)]
mod tests {
    use super::super::adjoint::run_paralru_adjoint_scan_tile;
    use super::super::duhem_scans::run_paralru_duhem_forward_tile;
    use super::super::{ParaLruParams, SOFT_RECT_EPS};
    use super::*;
    use crate::simd::MAX_PARALRU_R_F32;
    use proptest::prelude::*;

    #[derive(Copy, Clone)]
    struct DuhemOracleParamsF64 {
        alpha_rel: f64,
        alpha_atk: f64,
        kappa: f64,
        gamma_duhem: f64,
        signed_mix: f64,
        stereo_link: f64,
        two_svf_g: f64,
    }

    fn duhem_loss_f64(
        u: [&[f64]; 2],
        bp: [&[f64]; 2],
        v_init: [f64; 2],
        p: DuhemOracleParamsF64,
        adj_veff: [&[f64]; 2],
        adj_v_out: [f64; 2],
    ) -> f64 {
        let eps = SOFT_RECT_EPS as f64;
        let step_ch = |ui: f64, bpi: f64, v_prev: f64| -> f64 {
            let flux_abs = (p.two_svf_g * bpi).abs();
            let rho = ((1.0 - p.alpha_atk) + p.gamma_duhem * flux_abs).clamp(0.0, 1.0);
            let d = p.alpha_rel * v_prev;
            let z = ui - p.kappa * (1.0 - p.signed_mix) * d;
            let phi_rect = 0.5 * (z + (z * z + eps * eps).sqrt() - eps);
            let z_bar = z.clamp(-1.0, 1.0);
            let psi_1 = z_bar * (1.0 - (1.0 / 3.0) * z_bar * z_bar);
            let phi_eta = (1.0 - p.signed_mix) * phi_rect + p.signed_mix * psi_1;
            (1.0 - p.signed_mix * rho) * d + rho * phi_eta
        };
        let mut vl = v_init[0];
        let mut vr = v_init[1];
        let mut loss = 0.0;
        for i in 0..u[0].len() {
            vl = step_ch(u[0][i], bp[0][i], vl);
            vr = step_ch(u[1][i], bp[1][i], vr);
            let v_mid = 0.5 * (vl + vr);
            let veff_l = vl + p.stereo_link * (v_mid - vl);
            let veff_r = vr + p.stereo_link * (v_mid - vr);
            loss += adj_veff[0][i] * veff_l + adj_veff[1][i] * veff_r;
        }
        loss + adj_v_out[0] * vl + adj_v_out[1] * vr
    }

    #[test]
    fn test_paralru_duhem_adjoint_edge_cases_and_neutral_parity() {
        let p = ParaLruDuhemParams::clamped(0.9, 0.3, 0.5, 1.5, 0.4, 0.25, 0.6);
        let empty_trace = ParaLruDuhemAdjointTrace::<0> {
            u: [&[], &[]],
            bp: [&[], &[]],
            v_traj: [&[], &[]],
            v_init: [0.0, 0.0],
        };
        let empty = run_paralru_duhem_adjoint_scan_tile(&empty_trace, p, [&[], &[]], [0.4, -0.7]);
        assert_eq!(empty.adj_v_init_l, 0.4);
        assert_eq!(empty.adj_v_init_r, -0.7);
        assert_eq!(empty.tile_summary_l, LinearScanElem::identity());
        assert_eq!(empty.tile_summary_r, LinearScanElem::identity());

        let u_l = [1.2_f32, -0.8, 0.5, 2.0, -1.4, 0.3, 1.7, -0.4];
        let u_r = [-0.9_f32, 1.5, -0.3, 0.7, 1.1, -1.8, 0.2, 1.3];
        let bp_l = [0.2_f32, -0.4, 0.3, -0.1, 0.5, -0.2, 0.15, -0.35];
        let bp_r = [-0.3_f32, 0.15, -0.25, 0.4, -0.1, 0.3, -0.45, 0.2];
        let ae_l = [0.5_f32, -0.2, 0.8, 0.1, -0.6, 0.4, 0.7, -0.3];
        let ae_r = [-0.4_f32, 0.6, -0.3, 0.9, 0.2, -0.5, 0.1, 0.8];

        let p_neutral = ParaLruDuhemParams::clamped(0.88, 0.35, 0.6, 0.0, 0.0, 0.0, 0.5);
        let fwd_neutral =
            run_paralru_duhem_forward_tile([&u_l, &u_r], [&bp_l, &bp_r], [0.2, -0.4], p_neutral);
        let tr_neutral = ParaLruDuhemAdjointTrace::from_forward(
            [&u_l, &u_r],
            [&bp_l, &bp_r],
            [0.2, -0.4],
            &fwd_neutral,
        );
        let g_duhem = run_paralru_duhem_adjoint_scan_tile(
            &tr_neutral,
            p_neutral,
            [&ae_l, &ae_r],
            [0.3, -0.5],
        );
        let lru_p = ParaLruParams::clamped(0.88, 0.35, 0.6);
        let gl = run_paralru_adjoint_scan_tile(&u_l, &fwd_neutral.v_traj_l, 0.2, lru_p, &ae_l, 0.3);
        let gr =
            run_paralru_adjoint_scan_tile(&u_r, &fwd_neutral.v_traj_r, -0.4, lru_p, &ae_r, -0.5);
        assert!((g_duhem.adj_v_init_l - gl.adj_v_init).abs() < 1.0e-6);
        assert!((g_duhem.adj_v_init_r - gr.adj_v_init).abs() < 1.0e-6);
        assert!((g_duhem.adj_alpha_rel - (gl.adj_alpha_rel + gr.adj_alpha_rel)).abs() < 1.0e-6);
        assert!((g_duhem.adj_alpha_atk - (gl.adj_alpha_atk + gr.adj_alpha_atk)).abs() < 1.0e-6);
        assert!((g_duhem.adj_kappa - (gl.adj_kappa + gr.adj_kappa)).abs() < 1.0e-6);
        assert_eq!(g_duhem.adj_bp_l, [0.0; 8]);
        assert_eq!(g_duhem.adj_bp_r, [0.0; 8]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_paralru_duhem_adjoint_matches_fd_and_split_tile_invariance(
            u_l_v in proptest::collection::vec(-2.5_f32..2.5_f32, 16),
            u_r_v in proptest::collection::vec(-2.5_f32..2.5_f32, 16),
            bp_l_v in proptest::collection::vec(prop_oneof![-0.4_f32..-0.05_f32, 0.05_f32..0.4_f32], 16),
            bp_r_v in proptest::collection::vec(prop_oneof![-0.4_f32..-0.05_f32, 0.05_f32..0.4_f32], 16),
            ae_l_v in proptest::collection::vec(-1.0_f32..1.0_f32, 16),
            ae_r_v in proptest::collection::vec(-1.0_f32..1.0_f32, 16),
            v_init_l in prop_oneof![-2.0_f32..-0.1_f32, 0.1_f32..2.0_f32],
            v_init_r in prop_oneof![-2.0_f32..-0.1_f32, 0.1_f32..2.0_f32],
            alpha_rel in 0.1_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.25_f32..0.85_f32,
            kappa in 0.05_f32..0.95_f32,
            gamma_duhem in 0.05_f32..0.6_f32,
            signed_mix in 0.05_f32..0.95_f32,
            stereo_link in 0.05_f32..0.95_f32,
            two_svf_g in 0.1_f32..0.6_f32,
            adj_out_l in -1.0_f32..1.0_f32,
            adj_out_r in -1.0_f32..1.0_f32,
        ) {
            let mut u_l = [0.0_f32; 16];
            let mut u_r = [0.0_f32; 16];
            let mut bp_l = [0.0_f32; 16];
            let mut bp_r = [0.0_f32; 16];
            let mut ae_l = [0.0_f32; 16];
            let mut ae_r = [0.0_f32; 16];
            u_l.copy_from_slice(&u_l_v);
            u_r.copy_from_slice(&u_r_v);
            bp_l.copy_from_slice(&bp_l_v);
            bp_r.copy_from_slice(&bp_r_v);
            ae_l.copy_from_slice(&ae_l_v);
            ae_r.copy_from_slice(&ae_r_v);

            let p = ParaLruDuhemParams::clamped(
                alpha_rel,
                alpha_atk,
                kappa,
                gamma_duhem,
                signed_mix,
                stereo_link,
                two_svf_g,
            );
            let fwd = run_paralru_duhem_forward_tile(
                [&u_l, &u_r],
                [&bp_l, &bp_r],
                [v_init_l, v_init_r],
                p,
            );
            let tr = ParaLruDuhemAdjointTrace::from_forward(
                [&u_l, &u_r],
                [&bp_l, &bp_r],
                [v_init_l, v_init_r],
                &fwd,
            );
            let grads = run_paralru_duhem_adjoint_scan_tile(
                &tr,
                p,
                [&ae_l, &ae_r],
                [adj_out_l, adj_out_r],
            );

            // Split-tile invariance: N = 16 == 8 + 8 chain.
            let mut u0_l = [0.0_f32; 8];
            let mut u1_l = [0.0_f32; 8];
            let mut u0_r = [0.0_f32; 8];
            let mut u1_r = [0.0_f32; 8];
            let mut bp0_l = [0.0_f32; 8];
            let mut bp1_l = [0.0_f32; 8];
            let mut bp0_r = [0.0_f32; 8];
            let mut bp1_r = [0.0_f32; 8];
            let mut v0_l = [0.0_f32; 8];
            let mut v1_l = [0.0_f32; 8];
            let mut v0_r = [0.0_f32; 8];
            let mut v1_r = [0.0_f32; 8];
            let mut a0_l = [0.0_f32; 8];
            let mut a1_l = [0.0_f32; 8];
            let mut a0_r = [0.0_f32; 8];
            let mut a1_r = [0.0_f32; 8];
            u0_l.copy_from_slice(&u_l[..8]);
            u1_l.copy_from_slice(&u_l[8..]);
            u0_r.copy_from_slice(&u_r[..8]);
            u1_r.copy_from_slice(&u_r[8..]);
            bp0_l.copy_from_slice(&bp_l[..8]);
            bp1_l.copy_from_slice(&bp_l[8..]);
            bp0_r.copy_from_slice(&bp_r[..8]);
            bp1_r.copy_from_slice(&bp_r[8..]);
            v0_l.copy_from_slice(&fwd.v_traj_l[..8]);
            v1_l.copy_from_slice(&fwd.v_traj_l[8..]);
            v0_r.copy_from_slice(&fwd.v_traj_r[..8]);
            v1_r.copy_from_slice(&fwd.v_traj_r[8..]);
            a0_l.copy_from_slice(&ae_l[..8]);
            a1_l.copy_from_slice(&ae_l[8..]);
            a0_r.copy_from_slice(&ae_r[..8]);
            a1_r.copy_from_slice(&ae_r[8..]);

            let tr1 = ParaLruDuhemAdjointTrace {
                u: [&u1_l, &u1_r],
                bp: [&bp1_l, &bp1_r],
                v_traj: [&v1_l, &v1_r],
                v_init: [v0_l[7], v0_r[7]],
            };
            let g1 = run_paralru_duhem_adjoint_scan_tile(&tr1, p, [&a1_l, &a1_r], [adj_out_l, adj_out_r]);
            let tr0 = ParaLruDuhemAdjointTrace {
                u: [&u0_l, &u0_r],
                bp: [&bp0_l, &bp0_r],
                v_traj: [&v0_l, &v0_r],
                v_init: [v_init_l, v_init_r],
            };
            let g0 = run_paralru_duhem_adjoint_scan_tile(
                &tr0,
                p,
                [&a0_l, &a0_r],
                [g1.adj_v_init_l, g1.adj_v_init_r],
            );
            prop_assert!((g0.adj_v_init_l - grads.adj_v_init_l).abs() < 1.0e-5);
            prop_assert!((g0.adj_v_init_r - grads.adj_v_init_r).abs() < 1.0e-5);
            prop_assert!(((g0.adj_alpha_rel + g1.adj_alpha_rel) - grads.adj_alpha_rel).abs() < 1.0e-4);
            prop_assert!(((g0.adj_alpha_atk + g1.adj_alpha_atk) - grads.adj_alpha_atk).abs() < 1.0e-4);
            prop_assert!(((g0.adj_kappa + g1.adj_kappa) - grads.adj_kappa).abs() < 1.0e-4);
            prop_assert!(((g0.adj_gamma_duhem + g1.adj_gamma_duhem) - grads.adj_gamma_duhem).abs() < 1.0e-4);
            prop_assert!(((g0.adj_signed_mix + g1.adj_signed_mix) - grads.adj_signed_mix).abs() < 1.0e-4);
            prop_assert!(((g0.adj_stereo_link + g1.adj_stereo_link) - grads.adj_stereo_link).abs() < 1.0e-4);

            // Track proximity to C^0 kinks (z = ±1, rho = 0 or 1) so central FD stays smooth.
            let mut min_kink_dist = f64::MAX;
            for ch in 0..2 {
                for i in 0..16 {
                    let vp = if i == 0 {
                        f64::from(tr.v_init[ch])
                    } else {
                        f64::from(tr.v_traj[ch][i - 1])
                    };
                    let d = f64::from(p.alpha_rel) * vp;
                    let z = f64::from(tr.u[ch][i]) - f64::from(p.kappa) * (1.0 - f64::from(p.signed_mix)) * d;
                    let rho_raw = (1.0 - f64::from(p.alpha_atk))
                        + f64::from(p.gamma_duhem) * (f64::from(p.two_svf_g) * f64::from(tr.bp[ch][i])).abs();
                    min_kink_dist = min_kink_dist
                        .min((z.abs() - 1.0).abs())
                        .min(rho_raw.abs())
                        .min((1.0 - rho_raw).abs());
                }
            }
            if min_kink_dist < 1.0e-3 {
                return Ok(());
            }

            let ul64: Vec<f64> = u_l.iter().map(|&x| f64::from(x)).collect();
            let ur64: Vec<f64> = u_r.iter().map(|&x| f64::from(x)).collect();
            let bpl64: Vec<f64> = bp_l.iter().map(|&x| f64::from(x)).collect();
            let bpr64: Vec<f64> = bp_r.iter().map(|&x| f64::from(x)).collect();
            let ael64: Vec<f64> = ae_l.iter().map(|&x| f64::from(x)).collect();
            let aer64: Vec<f64> = ae_r.iter().map(|&x| f64::from(x)).collect();
            let v0_64 = [f64::from(v_init_l), f64::from(v_init_r)];
            let aout_64 = [f64::from(adj_out_l), f64::from(adj_out_r)];
            let p64 = DuhemOracleParamsF64 {
                alpha_rel: f64::from(p.alpha_rel),
                alpha_atk: f64::from(p.alpha_atk),
                kappa: f64::from(p.kappa),
                gamma_duhem: f64::from(p.gamma_duhem),
                signed_mix: f64::from(p.signed_mix),
                stereo_link: f64::from(p.stereo_link),
                two_svf_g: f64::from(p.two_svf_g),
            };
            let eps = 1.0e-6_f64;
            let assert_fd = |got: f32, plus: f64, minus: f64| -> Result<(), TestCaseError> {
                let fd = (plus - minus) / (2.0 * eps);
                let err = (f64::from(got) - fd).abs();
                let tol = 5.0e-4 * fd.abs() + 2.0e-4;
                prop_assert!(err <= tol, "got={got}, fd={fd}, err={err}, tol={tol}");
                Ok(())
            };

            assert_fd(
                grads.adj_v_init_l,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], [v0_64[0] + eps, v0_64[1]], p64, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], [v0_64[0] - eps, v0_64[1]], p64, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_v_init_r,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], [v0_64[0], v0_64[1] + eps], p64, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], [v0_64[0], v0_64[1] - eps], p64, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_alpha_rel,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { alpha_rel: p64.alpha_rel + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { alpha_rel: p64.alpha_rel - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_alpha_atk,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { alpha_atk: p64.alpha_atk + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { alpha_atk: p64.alpha_atk - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_kappa,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { kappa: p64.kappa + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { kappa: p64.kappa - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_gamma_duhem,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { gamma_duhem: p64.gamma_duhem + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { gamma_duhem: p64.gamma_duhem - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_signed_mix,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { signed_mix: p64.signed_mix + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { signed_mix: p64.signed_mix - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_stereo_link,
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { stereo_link: p64.stereo_link + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemOracleParamsF64 { stereo_link: p64.stereo_link - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;

            for t in 0..16 {
                let (mut ulp, mut ulm) = (ul64.clone(), ul64.clone());
                ulp[t] += eps;
                ulm[t] -= eps;
                assert_fd(
                    grads.adj_u_l[t],
                    duhem_loss_f64([&ulp, &ur64], [&bpl64, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                    duhem_loss_f64([&ulm, &ur64], [&bpl64, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                )?;

                let (mut urp, mut urm) = (ur64.clone(), ur64.clone());
                urp[t] += eps;
                urm[t] -= eps;
                assert_fd(
                    grads.adj_u_r[t],
                    duhem_loss_f64([&ul64, &urp], [&bpl64, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                    duhem_loss_f64([&ul64, &urm], [&bpl64, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                )?;

                let (mut bplp, mut bplm) = (bpl64.clone(), bpl64.clone());
                bplp[t] += eps;
                bplm[t] -= eps;
                assert_fd(
                    grads.adj_bp_l[t],
                    duhem_loss_f64([&ul64, &ur64], [&bplp, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                    duhem_loss_f64([&ul64, &ur64], [&bplm, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                )?;

                let (mut bprp, mut bprm) = (bpr64.clone(), bpr64.clone());
                bprp[t] += eps;
                bprm[t] -= eps;
                assert_fd(
                    grads.adj_bp_r[t],
                    duhem_loss_f64([&ul64, &ur64], [&bpl64, &bprp], v0_64, p64, [&ael64, &aer64], aout_64),
                    duhem_loss_f64([&ul64, &ur64], [&bpl64, &bprm], v0_64, p64, [&ael64, &aer64], aout_64),
                )?;
            }
        }
    }
}
