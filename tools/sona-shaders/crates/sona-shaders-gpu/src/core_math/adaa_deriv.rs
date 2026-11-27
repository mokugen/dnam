#![forbid(unsafe_code)]

//! Exact analytical derivatives for `PolyADAA1`, the $C^2$ cubed-envelope wavefolder, and
//! Selective-Duhem `ParaLRU` steps (`ref[dsp.op-n.polyadaa.rail-crossing]`,
//! `ref[dsp.op-n.wavefolder.cubic-c2]`, `ref[dsp.op-n.core.matryoshka-blocks]`).

use super::{QuadLaneParams, is_finite_bits, scrub_endpoint};
use dnam_kernels::{
    GpuVec4, KernelVec4, LinearScanElem, MAX_ADAA_ENDPOINT, ONE_THIRD_F32, TINY_CROSSING_EPS,
    canonicalize_zero_f32_bits, const_f32_pow2, soft_rect_zero_dc_f32,
};

const TWO_THIRDS: f32 = 2.0 / 3.0;
const ONE_FIFTH: f32 = 1.0 / 5.0;
const FOUR_FIFTHS: f32 = 4.0 / 5.0;
const TWO_POW_64: f32 = const_f32_pow2(64);

const _: () = assert!(TWO_THIRDS > 0.666 && TWO_THIRDS < 0.667);
const _: () = assert!(FOUR_FIFTHS == 0.8);
const _: () = assert!(TINY_CROSSING_EPS > 0.0);
const _: () = assert!(MAX_ADAA_ENDPOINT > 1.0);

/// Forward values and exact endpoint partial derivatives for `(PolyADAA1, C^2 wavefolder)`
/// evaluated on `(u_prev_raw, u_curr_raw)`.
#[derive(Copy, Clone)]
pub struct AdaaWfDerivResult {
    pub odd_poly: f32,
    #[allow(dead_code)]
    pub even_poly: f32,
    pub wf_odd: f32,
    pub d_odd_da: f32,
    pub d_odd_db: f32,
    pub d_even_da: f32,
    pub d_even_db: f32,
    pub d_wf_da: f32,
    pub d_wf_db: f32,
}

#[derive(Copy, Clone)]
struct InteriorEval {
    odd: f32,
    even: f32,
    wf_odd: f32,
    d_odd_dlo: f32,
    d_odd_dhi: f32,
    d_even_dlo: f32,
    d_even_dhi: f32,
    d_wf_dlo: f32,
    d_wf_dhi: f32,
}

#[inline(always)]
fn eval_interior_and_derivs(lo: f32, hi: f32, max_abs: f32) -> InteriorEval {
    let m = 0.5 * (hi + lo);
    let d = 0.5 * (hi - lo);
    let m2 = m * m;
    let d2 = d * d;
    let m4 = m2 * m2;
    let d4 = d2 * d2;
    let s = m2 + d2;
    let om = 1.0 - s;

    let (odd, even, wf_odd) = if max_abs <= 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        let o = canonicalize_zero_f32_bits(m * (1.5 - 0.5 * s));
        let lin = m2 + ONE_THIRD_F32 * d2;
        let quar = (m4 + 2.0 * m2 * d2) + ONE_FIFTH * d4;
        let e = canonicalize_zero_f32_bits(lin - quar);
        let w = canonicalize_zero_f32_bits(m * om * (om * om + 4.0 * m2 * d2));
        (o, e, w)
    };

    let d_odd_dm = 1.5 - 1.5 * m2 - 0.5 * d2;
    let d_odd_dd = -m * d;
    let d_even_dm = 2.0 * m * (1.0 - 2.0 * s);
    let d_even_dd = d * (TWO_THIRDS - 4.0 * m2 - FOUR_FIFTHS * d2);

    let p = m * om;
    let q = om * om + 4.0 * m2 * d2;
    let dp_dm = om - 2.0 * m2;
    let dq_dm = 4.0 * m * (2.0 * d2 - om);
    let dp_dd = -2.0 * m * d;
    let dq_dd = 4.0 * d * (2.0 * m2 - om);
    let d_wf_dm = dp_dm * q + p * dq_dm;
    let d_wf_dd = dp_dd * q + p * dq_dd;

    InteriorEval {
        odd,
        even,
        wf_odd,
        d_odd_dlo: 0.5 * (d_odd_dm - d_odd_dd),
        d_odd_dhi: 0.5 * (d_odd_dm + d_odd_dd),
        d_even_dlo: 0.5 * (d_even_dm - d_even_dd),
        d_even_dhi: 0.5 * (d_even_dm + d_even_dd),
        d_wf_dlo: 0.5 * (d_wf_dm - d_wf_dd),
        d_wf_dhi: 0.5 * (d_wf_dm + d_wf_dd),
    }
}

#[inline(always)]
fn sat_point_values(x: f32) -> (f32, f32, f32) {
    if x <= -1.0 {
        (-1.0, 0.0, 0.0)
    } else if x >= 1.0 {
        (1.0, 0.0, 0.0)
    } else {
        let x2 = x * x;
        let om = 1.0 - x2;
        (x * (1.5 - 0.5 * x2), x2 - x2 * x2, x * om * om * om)
    }
}

#[inline(always)]
fn rcp_crossing_dist(dist: f32) -> f32 {
    if dist < TINY_CROSSING_EPS {
        let d_safe = if dist > 1.0e-24 { dist } else { 1.0e-24 };
        let inv_scaled = GpuVec4::splat(d_safe * TWO_POW_64).rcp_pos_nr2().to_array()[0];
        inv_scaled * TWO_POW_64
    } else {
        GpuVec4::splat(dist).rcp_pos_nr2().to_array()[0]
    }
}

/// Evaluates `PolyADAA1` `(odd, even)`, `C^2` wavefolder `wf_odd`, and exact analytical
/// endpoint derivatives `(∂/∂u_prev_raw, ∂/∂u_curr_raw)` with zero `FDIV` or cancellation.
#[inline]
#[must_use]
pub fn eval_adaa_and_wf_with_derivs(
    u_prev_raw: f32,
    u_curr_raw: f32,
    tiny_eps: f32,
) -> AdaaWfDerivResult {
    let a = scrub_endpoint(u_prev_raw);
    let b = scrub_endpoint(u_curr_raw);
    let a_abs = if is_finite_bits(u_prev_raw) {
        u_prev_raw.abs()
    } else {
        MAX_ADAA_ENDPOINT
    };
    let b_abs = if is_finite_bits(u_curr_raw) {
        u_curr_raw.abs()
    } else {
        MAX_ADAA_ENDPOINT
    };
    let a_grad_ok = is_finite_bits(u_prev_raw) && a_abs < MAX_ADAA_ENDPOINT;
    let b_grad_ok = is_finite_bits(u_curr_raw) && b_abs < MAX_ADAA_ENDPOINT;

    let lo_is_a = a <= b;
    let lo = if lo_is_a { a } else { b };
    let hi = if lo_is_a { b } else { a };
    let dist = hi - lo;
    let max_abs = a.abs().max(b.abs());
    let tiny = if is_finite_bits(tiny_eps) && tiny_eps > 0.0 {
        tiny_eps
    } else {
        TINY_CROSSING_EPS
    };

    let (
        odd_poly,
        even_poly,
        wf_odd,
        d_odd_dlo,
        d_odd_dhi,
        d_even_dlo,
        d_even_dhi,
        d_wf_dlo,
        d_wf_dhi,
    ) = if lo >= 1.0 {
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else if hi <= -1.0 {
        (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else if lo >= -1.0 && hi <= 1.0 {
        let ev = eval_interior_and_derivs(lo, hi, max_abs);
        (
            ev.odd,
            ev.even,
            ev.wf_odd,
            ev.d_odd_dlo,
            ev.d_odd_dhi,
            ev.d_even_dlo,
            ev.d_even_dhi,
            ev.d_wf_dlo,
            ev.d_wf_dhi,
        )
    } else {
        let cl_lo = if lo < -1.0 { -1.0 } else { lo };
        let cl_hi = if hi > 1.0 { 1.0 } else { hi };
        let ev = eval_interior_and_derivs(cl_lo, cl_hi, max_abs);
        if dist <= tiny {
            let lo_mask = if lo >= -1.0 { 1.0 } else { 0.0 };
            let hi_mask = if hi <= 1.0 { 1.0 } else { 0.0 };
            (
                ev.odd,
                ev.even,
                ev.wf_odd,
                ev.d_odd_dlo * lo_mask,
                ev.d_odd_dhi * hi_mask,
                ev.d_even_dlo * lo_mask,
                ev.d_even_dhi * hi_mask,
                ev.d_wf_dlo * lo_mask,
                ev.d_wf_dhi * hi_mask,
            )
        } else {
            let inv = rcp_crossing_dist(dist);
            let inside = (cl_hi - cl_lo).max(0.0);
            let above = (hi - lo.max(1.0)).max(0.0);
            let below = (hi.min(-1.0) - lo).max(0.0);
            let odd_val = canonicalize_zero_f32_bits(((above - below) + inside * ev.odd) * inv);
            let even_val = canonicalize_zero_f32_bits((inside * ev.even) * inv);
            let wf_val = canonicalize_zero_f32_bits((inside * ev.wf_odd) * inv);

            let (sat_odd_lo, sat_even_lo, sat_wf_lo) = sat_point_values(lo);
            let (sat_odd_hi, sat_even_hi, sat_wf_hi) = sat_point_values(hi);
            (
                odd_val,
                even_val,
                wf_val,
                (odd_val - sat_odd_lo) * inv,
                (sat_odd_hi - odd_val) * inv,
                (even_val - sat_even_lo) * inv,
                (sat_even_hi - even_val) * inv,
                (wf_val - sat_wf_lo) * inv,
                (sat_wf_hi - wf_val) * inv,
            )
        }
    };

    let (mut d_odd_da, mut d_odd_db) = if lo_is_a {
        (d_odd_dlo, d_odd_dhi)
    } else {
        (d_odd_dhi, d_odd_dlo)
    };
    let (mut d_even_da, mut d_even_db) = if lo_is_a {
        (d_even_dlo, d_even_dhi)
    } else {
        (d_even_dhi, d_even_dlo)
    };
    let (mut d_wf_da, mut d_wf_db) = if lo_is_a {
        (d_wf_dlo, d_wf_dhi)
    } else {
        (d_wf_dhi, d_wf_dlo)
    };

    if !a_grad_ok {
        d_odd_da = 0.0;
        d_even_da = 0.0;
        d_wf_da = 0.0;
    }
    if !b_grad_ok {
        d_odd_db = 0.0;
        d_even_db = 0.0;
        d_wf_db = 0.0;
    }

    AdaaWfDerivResult {
        odd_poly,
        even_poly,
        wf_odd,
        d_odd_da,
        d_odd_db,
        d_even_da,
        d_even_db,
        d_wf_da,
        d_wf_db,
    }
}

/// Scalar Selective-Duhem `ParaLRU` forward terms & Jacobian needed for reverse BPTT.
#[derive(Copy, Clone)]
pub struct DuhemStepTerms {
    pub flux_abs: f32,
    pub rho: f32,
    pub rho_open: bool,
    pub d: f32,
    pub one_minus_eta: f32,
    pub phi_rect: f32,
    pub psi_1: f32,
    pub phi_eta_prime: f32,
    pub jac_div_rel: f32,
    pub jac: f32,
    pub sign_bp: f32,
    pub v_clean: f32,
    pub bp_clean: f32,
}

/// Evaluates Selective-Duhem `ParaLRU` Jacobian and gate sensitivities matching
/// `dnam_kernels::sona_core::paralru::duhem_adjoint` (`ref[dsp.op-n.core.matryoshka-blocks]`).
#[inline]
#[must_use]
pub fn eval_duhem_step_terms(
    u_rect: f32,
    bp: f32,
    v_prev: f32,
    alpha_rel: f32,
    alpha_atk: f32,
    kappa: f32,
    gamma_duhem: f32,
    signed_mix: f32,
    two_svf_g: f32,
) -> DuhemStepTerms {
    let u_clean = canonicalize_zero_f32_bits(u_rect);
    let bp_clean = canonicalize_zero_f32_bits(bp);
    let v_clean = canonicalize_zero_f32_bits(v_prev);
    let flux_abs = canonicalize_zero_f32_bits(two_svf_g * bp_clean).abs();
    let duhem_mod = if gamma_duhem <= 0.0 {
        0.0
    } else {
        gamma_duhem * flux_abs
    };
    let rho_raw = (1.0 - alpha_atk) + duhem_mod;
    let rho = if rho_raw <= 0.0 {
        0.0
    } else if rho_raw >= 1.0 {
        1.0
    } else {
        rho_raw
    };
    let rho_open = rho > 0.0 && rho < 1.0;

    let d = alpha_rel * v_clean;
    let one_minus_eta = 1.0 - signed_mix;
    let z = u_clean - kappa * one_minus_eta * d;
    let (phi_rect, phi_rect_prime) = soft_rect_zero_dc_f32(z);
    let z_bar = if z <= -1.0 {
        -1.0
    } else if z >= 1.0 {
        1.0
    } else {
        z
    };
    let psi_1 = z_bar * (1.0 - ONE_THIRD_F32 * (z_bar * z_bar));
    let psi_1_prime = if z.abs() < 1.0 { 1.0 - z * z } else { 0.0 };
    let phi_eta_prime = if signed_mix <= 0.0 {
        phi_rect_prime
    } else {
        one_minus_eta * phi_rect_prime + signed_mix * psi_1_prime
    };
    let jac_div_rel = (1.0 - signed_mix * rho) - rho * kappa * one_minus_eta * phi_eta_prime;
    let jac = alpha_rel * jac_div_rel;
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
        phi_eta_prime,
        jac_div_rel,
        jac,
        sign_bp,
        v_clean,
        bp_clean,
    }
}

/// Returns `true` when `kappa * (1.0 - signed_mix) == 0.0` (after `[0, 1]` clamping), reducing
/// Selective-Duhem `ParaLRU` to an exact `O(log2 N)` linear affine recurrence.
#[inline(always)]
#[must_use]
pub fn is_duhem_decoupled(lp: &QuadLaneParams) -> bool {
    lp.kappa == 0.0 || lp.signed_mix >= 1.0
}

/// Evaluates the exact `O(log2 N)` `LinearScanElem` `{ a, b }` for a decoupled
/// (`kappa * (1 - signed_mix) == 0`) Selective-Duhem `ParaLRU` frame.
#[inline]
#[must_use]
pub fn eval_paralru_duhem_affine_step(u_rect: f32, bp: f32, lp: &QuadLaneParams) -> LinearScanElem {
    let flux_abs = canonicalize_zero_f32_bits(lp.two_svf_g * canonicalize_zero_f32_bits(bp)).abs();
    let duhem_mod = if lp.gamma_duhem <= 0.0 {
        0.0
    } else {
        lp.gamma_duhem * flux_abs
    };
    let rho_raw = (1.0 - lp.alpha_atk) + duhem_mod;
    let rho = if rho_raw <= 0.0 {
        0.0
    } else if rho_raw >= 1.0 {
        1.0
    } else {
        rho_raw
    };
    let z = canonicalize_zero_f32_bits(u_rect);
    let (phi_rect, _) = soft_rect_zero_dc_f32(z);
    let z_bar = if z <= -1.0 {
        -1.0
    } else if z >= 1.0 {
        1.0
    } else {
        z
    };
    let psi_1 = z_bar * (1.0 - ONE_THIRD_F32 * (z_bar * z_bar));
    let one_minus_eta = 1.0 - lp.signed_mix;
    let phi_eta = if lp.signed_mix <= 0.0 {
        phi_rect
    } else {
        one_minus_eta * phi_rect + lp.signed_mix * psi_1
    };
    LinearScanElem {
        a: canonicalize_zero_f32_bits((1.0 - lp.signed_mix * rho) * lp.alpha_rel),
        b: canonicalize_zero_f32_bits(rho * phi_eta),
    }
}
