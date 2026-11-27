//! `O(log2 N)` reverse-time inclusive suffix adjoint scan for smooth non-saturating SONA v4.0
//! Selective-Duhem `ParaLRU` (`ref[dsp.op-n.core.matryoshka-blocks]`).

use crate::simd::{GpuVec4, KernelVec4, MAX_DUHEM_FLUX_ABS, scrub_and_flush_f32_bits};

use super::super::ONE_THIRD_F32;
use super::super::adjoint::{compose_paralru_adjoint_scan, parallel_suffix_scan};
use super::super::duhem_adjoint::{ParaLruDuhemAdjointTrace, ParaLruDuhemTileGrads};
use super::super::duhem_scans::ParaLruDuhemParams;
use super::super::scans::{LinearScanElem, apply_linear_scan};
use super::super::soft_rect::soft_rect_quartic_f32;

#[derive(Copy, Clone)]
struct DuhemV4StepTerms {
    flux_abs: f32,
    inv_denom: f32,
    rho: f32,
    d_rho_d_mod: f32,
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
fn eval_duhem_v4_step_terms(
    u: f32,
    bp: f32,
    v_prev: f32,
    params: ParaLruDuhemParams,
) -> DuhemV4StepTerms {
    let u_clean = scrub_and_flush_f32_bits(u) + 0.0;
    let bp_clean = scrub_and_flush_f32_bits(bp) + 0.0;
    let v_clean = scrub_and_flush_f32_bits(v_prev) + 0.0;
    let flux_abs = scrub_and_flush_f32_bits(params.two_svf_g * bp_clean)
        .abs()
        .min(MAX_DUHEM_FLUX_ABS);
    let duhem_mod = if params.gamma_duhem <= 0.0 {
        0.0
    } else {
        params.gamma_duhem * flux_abs
    };
    let inv_denom = if duhem_mod <= 0.0 {
        1.0
    } else {
        GpuVec4::splat(1.0 + duhem_mod).rcp_ge1_nr2().to_array()[0]
    };
    let rho = (1.0 - params.alpha_atk * inv_denom).clamp(0.0, 1.0);
    let d_rho_d_mod = params.alpha_atk * inv_denom * inv_denom;

    let d = params.alpha_rel * v_clean;
    let one_minus_eta = 1.0 - params.signed_mix;
    let z = u_clean - params.kappa * one_minus_eta * d;
    let (phi_rect, phi_rect_prime) = soft_rect_quartic_f32(z);
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
    DuhemV4StepTerms {
        flux_abs,
        inv_denom,
        rho,
        d_rho_d_mod,
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

/// Forms the reverse-BPTT linear adjoint scan element `(J_n, J_n * adj_v_n)` for SONA v4.0
/// smooth Selective-Duhem `ParaLRU` (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline(always)]
#[must_use]
pub fn paralru_duhem_v4_adjoint_step_elem(
    u: f32,
    bp: f32,
    v_prev: f32,
    params: ParaLruDuhemParams,
    adj_v: f32,
) -> LinearScanElem {
    let jac = eval_duhem_v4_step_terms(u, bp, v_prev, params).jac;
    LinearScanElem {
        a: jac,
        b: jac * adj_v,
    }
}

struct ChannelDuhemV4Grads<const N: usize> {
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
fn run_channel_duhem_v4_adjoint_scan<const N: usize>(
    trace: &ParaLruDuhemAdjointTrace<'_, N>,
    ch: usize,
    adj_v: &[f32; N],
    adj_v_out: f32,
    active: usize,
    params: ParaLruDuhemParams,
) -> ChannelDuhemV4Grads<N> {
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
        adj_elems[i] = paralru_duhem_v4_adjoint_step_elem(u[i], bp[i], v_prev, params, adj_v[i]);
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

        let t = eval_duhem_v4_step_terms(u[i], bp[i], v_clean, params);
        adj_u[i] = lam_i * t.rho * t.phi_eta_prime;
        let adj_rho = lam_i * (t.phi_eta - params.signed_mix * t.d);
        adj_alpha_atk -= adj_rho * t.inv_denom;
        adj_gamma_duhem += adj_rho * t.d_rho_d_mod * t.flux_abs;
        if params.gamma_duhem > 0.0 && t.flux_abs < MAX_DUHEM_FLUX_ABS {
            adj_bp[i] = adj_rho * t.d_rho_d_mod * params.two_svf_g * params.gamma_duhem * t.sign_bp;
        }
        adj_alpha_rel += lam_i * v_clean * t.jac_div_rel;
        adj_kappa -= lam_i * t.rho * t.one_minus_eta * t.d * t.phi_eta_prime;
        adj_signed_mix +=
            lam_i * t.rho * ((t.psi_1 - t.phi_rect) + params.kappa * t.d * t.phi_eta_prime - t.d);
    }

    let tile_summary = adj_elems[0];
    ChannelDuhemV4Grads {
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

/// Executes the `O(log2 N)` smooth non-saturating joint `(L, R)` reverse-time inclusive suffix
/// adjoint scan over `valid_len.min(N)` active frames with zero `1_{rho in (0,1)}` gradient
/// collapse (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline]
#[must_use]
pub fn run_paralru_duhem_v4_adjoint_scan_partial_tile<const N: usize>(
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

    let gl = run_channel_duhem_v4_adjoint_scan(trace, 0, &adj_v_l, adj_v_out[0], active, params);
    let gr = run_channel_duhem_v4_adjoint_scan(trace, 1, &adj_v_r, adj_v_out[1], active, params);

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

/// Executes the `O(log2 N)` smooth non-saturating joint `(L, R)` reverse-time inclusive suffix
/// adjoint scan over all `N` frames (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline]
#[must_use]
pub fn run_paralru_duhem_v4_adjoint_scan_tile<const N: usize>(
    trace: &ParaLruDuhemAdjointTrace<'_, N>,
    params: ParaLruDuhemParams,
    adj_veff: [&[f32; N]; 2],
    adj_v_out: [f32; 2],
) -> ParaLruDuhemTileGrads<N> {
    run_paralru_duhem_v4_adjoint_scan_partial_tile(trace, N, params, adj_veff, adj_v_out)
}

#[cfg(test)]
mod tests {
    use super::super::super::SOFT_RECT_EPS;
    use super::super::{
        run_paralru_duhem_v4_forward_partial_tile, run_paralru_duhem_v4_forward_tile,
    };
    use super::*;
    use crate::simd::MAX_PARALRU_R_F32;
    use proptest::prelude::*;

    #[derive(Copy, Clone)]
    struct DuhemV4OracleParamsF64 {
        alpha_rel: f64,
        alpha_atk: f64,
        kappa: f64,
        gamma_duhem: f64,
        signed_mix: f64,
        stereo_link: f64,
        two_svf_g: f64,
    }

    fn duhem_v4_loss_f64(
        u: [&[f64]; 2],
        bp: [&[f64]; 2],
        v_init: [f64; 2],
        p: DuhemV4OracleParamsF64,
        adj_veff: [&[f64]; 2],
        adj_v_out: [f64; 2],
    ) -> f64 {
        let eps = f64::from(SOFT_RECT_EPS);
        let two_eps = 2.0 * eps;
        let step_ch = |ui: f64, bpi: f64, v_prev: f64| -> f64 {
            let flux_abs = (p.two_svf_g * bpi).abs().min(f64::from(MAX_DUHEM_FLUX_ABS));
            let rho = (1.0 - p.alpha_atk / (1.0 + p.gamma_duhem * flux_abs)).clamp(0.0, 1.0);
            let d = p.alpha_rel * v_prev;
            let z = ui - p.kappa * (1.0 - p.signed_mix) * d;
            let t = (z / two_eps).clamp(0.0, 1.0);
            let phi_rect = eps * t * t * t * (2.0 - t) + (z - two_eps).max(0.0);
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

    /// verify[dsp.op-n.core.matryoshka-blocks]
    #[test]
    fn test_duhem_v4_adjoint_nonvanishing_grad_and_partial_tile_parity() {
        // Non-vanishing gradients when (1 - alpha_atk) + gamma_duhem * |flux| >= 1.0!
        let p_sat = ParaLruDuhemParams::clamped(0.9, 0.5, 0.4, 4.0, 0.3, 0.25, 2.0);
        let u_l = [0.8_f32; 4];
        let u_r = [-0.6_f32; 4];
        let bp_l = [0.5_f32; 4];
        let bp_r = [-0.5_f32; 4];
        let fwd =
            run_paralru_duhem_v4_forward_tile([&u_l, &u_r], [&bp_l, &bp_r], [0.2, -0.1], p_sat);
        let tr =
            ParaLruDuhemAdjointTrace::from_forward([&u_l, &u_r], [&bp_l, &bp_r], [0.2, -0.1], &fwd);
        let g =
            run_paralru_duhem_v4_adjoint_scan_tile(&tr, p_sat, [&[1.0; 4], &[0.5; 4]], [0.1, -0.2]);
        assert!(g.adj_alpha_atk.abs() > 1.0e-3);
        assert!(g.adj_gamma_duhem.abs() > 1.0e-3);
        assert!(g.adj_bp_l[0].abs() > 1.0e-3);

        let empty_adj = run_paralru_duhem_v4_adjoint_scan_partial_tile(
            &tr,
            0,
            p_sat,
            [&u_l, &u_r],
            [0.25, -0.75],
        );
        assert_eq!(empty_adj.adj_v_init_l, 0.25);
        assert_eq!(empty_adj.adj_v_init_r, -0.75);

        let p = ParaLruDuhemParams::clamped(0.91, 0.45, 0.55, 3.5, 0.35, 0.4, 1.8);
        let u8_l = [0.6_f32, -0.4, 0.9, 0.2, -0.7, 99.0, 99.0, 99.0];
        let u8_r = [-0.3_f32, 0.8, -0.5, 0.4, 0.1, -99.0, -99.0, -99.0];
        let bp8_l = [0.4_f32, -0.6, 0.2, -0.8, 0.5, 50.0, 50.0, 50.0];
        let bp8_r = [-0.5_f32, 0.3, -0.7, 0.1, -0.2, -50.0, -50.0, -50.0];
        let ae8_l = [1.0_f32, -0.5, 0.25, 0.75, -0.4, 10.0, 10.0, 10.0];
        let ae8_r = [-0.8_f32, 0.6, -0.2, 0.3, 0.9, -10.0, -10.0, -10.0];
        let (u5_l, u5_r): ([f32; 5], [f32; 5]) =
            (u8_l[..5].try_into().unwrap(), u8_r[..5].try_into().unwrap());
        let (bp5_l, bp5_r): ([f32; 5], [f32; 5]) = (
            bp8_l[..5].try_into().unwrap(),
            bp8_r[..5].try_into().unwrap(),
        );
        let (ae5_l, ae5_r): ([f32; 5], [f32; 5]) = (
            ae8_l[..5].try_into().unwrap(),
            ae8_r[..5].try_into().unwrap(),
        );

        let f5 =
            run_paralru_duhem_v4_forward_tile([&u5_l, &u5_r], [&bp5_l, &bp5_r], [0.25, -0.35], p);
        let f8 = run_paralru_duhem_v4_forward_partial_tile(
            [&u8_l, &u8_r],
            [&bp8_l, &bp8_r],
            5,
            [0.25, -0.35],
            p,
        );
        assert_eq!(
            (f8.v_end_l.to_bits(), f8.v_end_r.to_bits()),
            (f5.v_end_l.to_bits(), f5.v_end_r.to_bits())
        );
        assert_eq!(
            (&f8.v_traj_l[..5], &f8.v_eff_traj_r[..5]),
            (&f5.v_traj_l[..], &f5.v_eff_traj_r[..])
        );
        assert!(f8.v_traj_l[5..].iter().all(|v| v.to_bits() == 0));

        let t5 = ParaLruDuhemAdjointTrace::from_forward(
            [&u5_l, &u5_r],
            [&bp5_l, &bp5_r],
            [0.25, -0.35],
            &f5,
        );
        let t8 = ParaLruDuhemAdjointTrace::from_forward(
            [&u8_l, &u8_r],
            [&bp8_l, &bp8_r],
            [0.25, -0.35],
            &f8,
        );
        let g5 = run_paralru_duhem_v4_adjoint_scan_tile(&t5, p, [&ae5_l, &ae5_r], [0.15, -0.45]);
        let g8 = run_paralru_duhem_v4_adjoint_scan_partial_tile(
            &t8,
            5,
            p,
            [&ae8_l, &ae8_r],
            [0.15, -0.45],
        );
        assert_eq!(
            (
                g8.adj_v_init_l.to_bits(),
                g8.adj_v_init_r.to_bits(),
                g8.adj_alpha_rel.to_bits(),
                g8.adj_alpha_atk.to_bits(),
                g8.adj_kappa.to_bits(),
                g8.adj_gamma_duhem.to_bits(),
                g8.adj_signed_mix.to_bits(),
                g8.adj_stereo_link.to_bits(),
            ),
            (
                g5.adj_v_init_l.to_bits(),
                g5.adj_v_init_r.to_bits(),
                g5.adj_alpha_rel.to_bits(),
                g5.adj_alpha_atk.to_bits(),
                g5.adj_kappa.to_bits(),
                g5.adj_gamma_duhem.to_bits(),
                g5.adj_signed_mix.to_bits(),
                g5.adj_stereo_link.to_bits(),
            ),
        );
        assert_eq!(
            (&g8.adj_u_l[..5], &g8.adj_bp_r[..5]),
            (&g5.adj_u_l[..], &g5.adj_bp_r[..])
        );
        assert!(g8.adj_u_l[5..].iter().all(|v| v.to_bits() == 0));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// verify[dsp.op-n.core.matryoshka-blocks]
        /// verify[dsp.op-n.core.stereo-simd]
        #[test]
        fn prop_duhem_v4_forward_and_adjoint_match_f64_oracle_and_fd(
            u_l_v in proptest::collection::vec(-2.5_f32..2.5_f32, 12),
            u_r_v in proptest::collection::vec(-2.5_f32..2.5_f32, 12),
            bp_l_v in proptest::collection::vec(prop_oneof![-2.0_f32..-0.05_f32, 0.05_f32..2.0_f32], 12),
            bp_r_v in proptest::collection::vec(prop_oneof![-2.0_f32..-0.05_f32, 0.05_f32..2.0_f32], 12),
            ae_l_v in proptest::collection::vec(-1.0_f32..1.0_f32, 12),
            ae_r_v in proptest::collection::vec(-1.0_f32..1.0_f32, 12),
            v_init_l in prop_oneof![-2.0_f32..-0.1_f32, 0.1_f32..2.0_f32],
            v_init_r in prop_oneof![-2.0_f32..-0.1_f32, 0.1_f32..2.0_f32],
            alpha_rel in 0.1_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.1_f32..0.9_f32,
            kappa in 0.05_f32..0.95_f32,
            gamma_duhem in 0.1_f32..8.0_f32,
            signed_mix in 0.05_f32..0.95_f32,
            stereo_link in 0.05_f32..0.95_f32,
            two_svf_g in 0.1_f32..4.0_f32,
            adj_out_l in -1.0_f32..1.0_f32,
            adj_out_r in -1.0_f32..1.0_f32,
        ) {
            let mut u_l = [0.0_f32; 12];
            let mut u_r = [0.0_f32; 12];
            let mut bp_l = [0.0_f32; 12];
            let mut bp_r = [0.0_f32; 12];
            let mut ae_l = [0.0_f32; 12];
            let mut ae_r = [0.0_f32; 12];
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
            let fwd = run_paralru_duhem_v4_forward_tile([&u_l, &u_r], [&bp_l, &bp_r], [v_init_l, v_init_r], p);
            let tr = ParaLruDuhemAdjointTrace::from_forward([&u_l, &u_r], [&bp_l, &bp_r], [v_init_l, v_init_r], &fwd);
            let grads = run_paralru_duhem_v4_adjoint_scan_tile(&tr, p, [&ae_l, &ae_r], [adj_out_l, adj_out_r]);

            // Avoid C^0 kink at z = ±1 on psi_1 only so central FD is well-defined.
            for ch in 0..2 {
                for i in 0..12 {
                    let vp = if i == 0 {
                        f64::from(tr.v_init[ch])
                    } else {
                        f64::from(tr.v_traj[ch][i - 1])
                    };
                    let d = f64::from(p.alpha_rel) * vp;
                    let z = f64::from(tr.u[ch][i]) - f64::from(p.kappa) * (1.0 - f64::from(p.signed_mix)) * d;
                    if (z.abs() - 1.0).abs() < 1.0e-3 {
                        return Ok(());
                    }
                }
            }

            let ul64: Vec<f64> = u_l.iter().map(|&x| f64::from(x)).collect();
            let ur64: Vec<f64> = u_r.iter().map(|&x| f64::from(x)).collect();
            let bpl64: Vec<f64> = bp_l.iter().map(|&x| f64::from(x)).collect();
            let bpr64: Vec<f64> = bp_r.iter().map(|&x| f64::from(x)).collect();
            let ael64: Vec<f64> = ae_l.iter().map(|&x| f64::from(x)).collect();
            let aer64: Vec<f64> = ae_r.iter().map(|&x| f64::from(x)).collect();
            let v0_64 = [f64::from(v_init_l), f64::from(v_init_r)];
            let aout_64 = [f64::from(adj_out_l), f64::from(adj_out_r)];
            let p64 = DuhemV4OracleParamsF64 {
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
                grads.adj_alpha_rel,
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { alpha_rel: p64.alpha_rel + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { alpha_rel: p64.alpha_rel - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_alpha_atk,
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { alpha_atk: p64.alpha_atk + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { alpha_atk: p64.alpha_atk - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_kappa,
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { kappa: p64.kappa + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { kappa: p64.kappa - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_gamma_duhem,
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { gamma_duhem: p64.gamma_duhem + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { gamma_duhem: p64.gamma_duhem - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_signed_mix,
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { signed_mix: p64.signed_mix + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { signed_mix: p64.signed_mix - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;
            assert_fd(
                grads.adj_stereo_link,
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { stereo_link: p64.stereo_link + eps, ..p64 }, [&ael64, &aer64], aout_64),
                duhem_v4_loss_f64([&ul64, &ur64], [&bpl64, &bpr64], v0_64, DuhemV4OracleParamsF64 { stereo_link: p64.stereo_link - eps, ..p64 }, [&ael64, &aer64], aout_64),
            )?;

            for t in 0..12 {
                let (mut bplp, mut bplm) = (bpl64.clone(), bpl64.clone());
                bplp[t] += eps;
                bplm[t] -= eps;
                assert_fd(
                    grads.adj_bp_l[t],
                    duhem_v4_loss_f64([&ul64, &ur64], [&bplp, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                    duhem_v4_loss_f64([&ul64, &ur64], [&bplm, &bpr64], v0_64, p64, [&ael64, &aer64], aout_64),
                )?;
            }
        }
    }
}
