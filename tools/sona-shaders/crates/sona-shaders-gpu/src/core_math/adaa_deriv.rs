#![forbid(unsafe_code)]

//! Exact analytical derivatives for the $C^2$ cubed-envelope wavefolder and SONA v4.0 smooth
//! non-saturating Selective-Duhem `ParaLRU` steps (`ref[dsp.op-n.polyadaa.rail-crossing]`,
//! `ref[dsp.op-n.wavefolder.cubic-c2]`, `ref[dsp.op-n.core.matryoshka-blocks]`).

use super::{QuadLaneParams, is_finite_bits};
use crate::reduce::clamp_f32_safe;
use dnam_kernels::{
    GpuVec4, KernelVec4, LinearScanElem, MAX_DUHEM_FLUX_ABS, ONE_THIRD_F32, TINY_CROSSING_EPS,
    canonicalize_zero_f32_bits, const_f32_pow2, soft_rect_quartic_f32,
};

const TWO_POW_64: f32 = const_f32_pow2(64);

const _: () = assert!(TINY_CROSSING_EPS > 0.0);

/// Forward value and exact endpoint partial derivatives for the odd-only $C^2$ wavefolder
/// evaluated on clamped `[-4096.0, 4096.0]` endpoints `(r_prev_safe, r_curr_safe)`.
#[derive(Copy, Clone)]
pub struct WfOddDerivResult {
    pub wf_odd: f32,
    pub d_wf_da: f32,
    pub d_wf_db: f32,
}

#[inline(always)]
fn eval_wf_odd_interior_and_derivs(lo: f32, hi: f32, max_abs: f32) -> (f32, f32, f32) {
    let m = 0.5 * (hi + lo);
    let d = 0.5 * (hi - lo);
    let m2 = m * m;
    let d2 = d * d;
    let s = m2 + d2;
    let om = 1.0 - s;

    let wf_odd = if max_abs <= 0.0 {
        0.0
    } else {
        canonicalize_zero_f32_bits(m * om * (om * om + 4.0 * m2 * d2))
    };

    let p = m * om;
    let q = om * om + 4.0 * m2 * d2;
    let dp_dm = om - 2.0 * m2;
    let dq_dm = 4.0 * m * (2.0 * d2 - om);
    let dp_dd = -2.0 * m * d;
    let dq_dd = 4.0 * d * (2.0 * m2 - om);
    let d_wf_dm = dp_dm * q + p * dq_dm;
    let d_wf_dd = dp_dd * q + p * dq_dd;

    (wf_odd, 0.5 * (d_wf_dm - d_wf_dd), 0.5 * (d_wf_dm + d_wf_dd))
}

#[inline(always)]
fn sat_wf_odd_point_value(x: f32) -> f32 {
    if x <= -1.0 || x >= 1.0 {
        0.0
    } else {
        let x2 = x * x;
        let om = 1.0 - x2;
        x * om * om * om
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

/// Evaluates odd-only $C^2$ wavefolder `wf_odd` and exact analytical endpoint derivatives
/// `(∂wf_odd/∂r_prev_safe, ∂wf_odd/∂r_curr_safe)` on clamped `[-4096.0, 4096.0]` endpoints
/// with zero `FDIV`.
#[inline]
#[must_use]
pub fn eval_c2_wf_odd_with_derivs(
    r_prev_safe: f32,
    r_curr_safe: f32,
    tiny_eps: f32,
) -> WfOddDerivResult {
    let a = r_prev_safe;
    let b = r_curr_safe;
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

    let (wf_odd, d_wf_dlo, d_wf_dhi) = if lo >= 1.0 || hi <= -1.0 {
        (0.0, 0.0, 0.0)
    } else if lo >= -1.0 && hi <= 1.0 {
        eval_wf_odd_interior_and_derivs(lo, hi, max_abs)
    } else {
        let cl_lo = if lo < -1.0 { -1.0 } else { lo };
        let cl_hi = if hi > 1.0 { 1.0 } else { hi };
        let (wf_int, d_lo_int, d_hi_int) = eval_wf_odd_interior_and_derivs(cl_lo, cl_hi, max_abs);
        if dist <= tiny {
            let lo_mask = if lo >= -1.0 { 1.0 } else { 0.0 };
            let hi_mask = if hi <= 1.0 { 1.0 } else { 0.0 };
            (wf_int, d_lo_int * lo_mask, d_hi_int * hi_mask)
        } else {
            let inv = rcp_crossing_dist(dist);
            let inside = (cl_hi - cl_lo).max(0.0);
            let wf_val = canonicalize_zero_f32_bits((inside * wf_int) * inv);
            let sat_wf_lo = sat_wf_odd_point_value(lo);
            let sat_wf_hi = sat_wf_odd_point_value(hi);
            (
                wf_val,
                (wf_val - sat_wf_lo) * inv,
                (sat_wf_hi - wf_val) * inv,
            )
        }
    };

    let (d_wf_da, d_wf_db) = if lo_is_a {
        (d_wf_dlo, d_wf_dhi)
    } else {
        (d_wf_dhi, d_wf_dlo)
    };

    WfOddDerivResult {
        wf_odd,
        d_wf_da,
        d_wf_db,
    }
}

/// Scalar smooth SONA v4.0 Selective-Duhem `ParaLRU` forward terms & Jacobian needed for reverse BPTT.
#[derive(Copy, Clone)]
pub struct DuhemStepTerms {
    pub flux_abs: f32,
    pub inv_denom: f32,
    pub rho: f32,
    pub d_rho_d_mod: f32,
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

/// Evaluates SONA v4.0 smooth Selective-Duhem `ParaLRU` Jacobian and gate sensitivities matching
/// `dnam_kernels::sona_core::paralru::duhem_v4::adjoint` (`ref[dsp.op-n.core.matryoshka-blocks]`).
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
    let flux_abs = canonicalize_zero_f32_bits(two_svf_g * bp_clean)
        .abs()
        .min(MAX_DUHEM_FLUX_ABS);
    let duhem_mod = if gamma_duhem <= 0.0 {
        0.0
    } else {
        gamma_duhem * flux_abs
    };
    let inv_denom = if duhem_mod <= 0.0 {
        1.0
    } else {
        GpuVec4::splat(1.0 + duhem_mod).rcp_ge1_nr2().to_array()[0]
    };
    let rho = clamp_f32_safe(1.0 - alpha_atk * inv_denom, 0.0, 1.0);
    let d_rho_d_mod = alpha_atk * inv_denom * inv_denom;

    let d = alpha_rel * v_clean;
    let one_minus_eta = 1.0 - signed_mix;
    let z = u_clean - kappa * one_minus_eta * d;
    let (phi_rect, phi_rect_prime) = soft_rect_quartic_f32(z);
    let z_bar = clamp_f32_safe(z, -1.0, 1.0);
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
        inv_denom,
        rho,
        d_rho_d_mod,
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
/// (`kappa * (1 - signed_mix) == 0`) smooth SONA v4.0 Selective-Duhem `ParaLRU` frame.
#[inline]
#[must_use]
pub fn eval_paralru_duhem_affine_step(u_rect: f32, bp: f32, lp: &QuadLaneParams) -> LinearScanElem {
    let flux_abs = canonicalize_zero_f32_bits(lp.two_svf_g * canonicalize_zero_f32_bits(bp))
        .abs()
        .min(MAX_DUHEM_FLUX_ABS);
    let duhem_mod = if lp.gamma_duhem <= 0.0 {
        0.0
    } else {
        lp.gamma_duhem * flux_abs
    };
    let inv_denom = if duhem_mod <= 0.0 {
        1.0
    } else {
        GpuVec4::splat(1.0 + duhem_mod).rcp_ge1_nr2().to_array()[0]
    };
    let rho = clamp_f32_safe(1.0 - lp.alpha_atk * inv_denom, 0.0, 1.0);
    let z = canonicalize_zero_f32_bits(u_rect);
    let (phi_rect, _) = soft_rect_quartic_f32(z);
    let z_bar = clamp_f32_safe(z, -1.0, 1.0);
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
