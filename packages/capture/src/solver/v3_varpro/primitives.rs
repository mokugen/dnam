#![forbid(unsafe_code)]
//! Differentiable Double-Precision Forward/Adjoint DSP Primitives (`TPT-SVF`, `tan` Prewarp,
//! Padé `[5/4]` `tanh`/`sigmoid`, Zero-Lag `2x` Oversampling, & Multi-Timescale GRU Tapes).

use std::f64::consts::LN_2;

use dnam_engine::stages::stage1_svf::SVF_W_MAX;
use dnam_engine::stages::stage2_gru::math_pade::{
    TANH_PADE_D1_F64, TANH_PADE_D2_F64, TANH_PADE_N1_F64, TANH_PADE_N2_F64,
};
use dnam_engine::stages::stage2_gru::{HB_A0, HB_A1};

mod parallel;
mod tapes;
pub(crate) use parallel::*;
pub use tapes::*;

/// Evaluates `2^(oct.clamp(-5, 5))` and its smooth derivative w.r.t. `oct`.
#[inline(always)]
pub fn exp2_oct_with_deriv(oct: f64) -> (f64, f64) {
    let clamped = oct.clamp(-5.0, 5.0);
    let y = clamped.exp2();
    let d = if (-5.0..=5.0).contains(&oct) {
        y * LN_2
    } else {
        0.0
    };
    (y, d)
}

/// Evaluates division-free 7th-order bilinear `tan(w_half)` prewarp matching [`dnam_engine::stages::stage1_svf::tan_prewarp_f32`] and its derivative.
#[inline(always)]
pub fn tan_prewarp_with_deriv(w_half: f64) -> (f64, f64) {
    let w_max = f64::from(SVF_W_MAX);
    let x = w_half.clamp(1.0e-4, w_max);
    let x2 = x * x;
    let y = x * (1.0 + x2 * ((1.0 / 3.0) + x2 * ((2.0 / 15.0) + (17.0 / 315.0) * x2)));
    let dy = if w_half > 1.0e-4 && w_half < w_max {
        1.0 + x2 * (1.0 + x2 * ((2.0 / 3.0) + (17.0 / 45.0) * x2))
    } else {
        0.0
    };
    (y, dy)
}

/// Evaluates Montgomery simultaneous reciprocal `(1/d0, 1/d1)` with a single division.
#[inline(always)]
pub fn inv2_pos_f64(d0: f64, d1: f64) -> (f64, f64) {
    let inv = 1.0 / (d0 * d1);
    (inv * d1, inv * d0)
}

#[inline(always)]
fn prod4_parts(d: [f64; 4]) -> (f64, f64, f64) {
    let p01 = d[0] * d[1];
    let p23 = d[2] * d[3];
    (p01, p23, p01 * p23)
}

#[inline(always)]
fn unpack4_recip(inv_all: f64, d: [f64; 4], p01: f64, p23: f64) -> [f64; 4] {
    let inv01 = inv_all * p23;
    let inv23 = inv_all * p01;
    [inv01 * d[1], inv01 * d[0], inv23 * d[3], inv23 * d[2]]
}

/// Evaluates Montgomery simultaneous reciprocal `[1/d0, 1/d1, 1/d2, 1/d3]` with a single division.
#[inline(always)]
pub fn inv4_pos_f64(d: [f64; 4]) -> [f64; 4] {
    let (p01, p23, tot) = prod4_parts(d);
    unpack4_recip(1.0 / tot, d, p01, p23)
}

/// Evaluates Montgomery simultaneous reciprocal for four 4-lane vectors with one division total.
#[inline(always)]
pub fn inv16_pos_f64(
    d0: [f64; 4],
    d1: [f64; 4],
    d2: [f64; 4],
    d3: [f64; 4],
) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    let (p001, p023, t0) = prod4_parts(d0);
    let (p101, p123, t1) = prod4_parts(d1);
    let (p201, p223, t2) = prod4_parts(d2);
    let (p301, p323, t3) = prod4_parts(d3);
    let inv_t = inv4_pos_f64([t0, t1, t2, t3]);
    (
        unpack4_recip(inv_t[0], d0, p001, p023),
        unpack4_recip(inv_t[1], d1, p101, p123),
        unpack4_recip(inv_t[2], d2, p201, p223),
        unpack4_recip(inv_t[3], d3, p301, p323),
    )
}

/// Evaluates Padé `[5/4]` `tanh(x)` (`C^1` clamped at `±4`) matching engine math and its derivative.
#[inline(always)]
pub fn tanh_pade5_with_deriv(x: f64) -> (f64, f64) {
    let xc = if x <= -4.0 || x >= 4.0 { 0.0 } else { x };
    let s = xc * xc;
    let p = xc * (1.0 + s * (TANH_PADE_N1_F64 + s * TANH_PADE_N2_F64));
    let q = 1.0 + s * (TANH_PADE_D1_F64 + s * TANH_PADE_D2_F64);
    let inv_q = 1.0 / q;
    let raw = p * inv_q;
    let raw_cl = raw.clamp(-1.0, 1.0);
    let val = if x <= -4.0 {
        -1.0
    } else if x >= 4.0 {
        1.0
    } else {
        raw_cl
    };
    let dp = 1.0 + s * (3.0 * TANH_PADE_N1_F64 + 5.0 * TANH_PADE_N2_F64 * s);
    let dq = xc * (2.0 * TANH_PADE_D1_F64 + 4.0 * TANH_PADE_D2_F64 * s);
    let d_raw = (dp * q - p * dq) * inv_q * inv_q;
    let in_range = x > -4.0 && x < 4.0 && raw > -1.0 && raw < 1.0;
    let deriv = if in_range || x.is_nan() { d_raw } else { 0.0 };
    (val, deriv)
}

#[inline(always)]
fn prepare_tanh_pade5_quad(x: [f64; 4]) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    let mut xc = [0.0_f64; 4];
    let mut s = [0.0_f64; 4];
    let mut p = [0.0_f64; 4];
    let mut q = [0.0_f64; 4];
    for i in 0..4 {
        let xi = x[i];
        let c = if xi <= -4.0 || xi >= 4.0 { 0.0 } else { xi };
        let si = c * c;
        xc[i] = c;
        s[i] = si;
        p[i] = c * (1.0 + si * (TANH_PADE_N1_F64 + si * TANH_PADE_N2_F64));
        q[i] = 1.0 + si * (TANH_PADE_D1_F64 + si * TANH_PADE_D2_F64);
    }
    (xc, s, p, q)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn finish_tanh_pade5_quad(
    x: [f64; 4],
    xc: [f64; 4],
    s: [f64; 4],
    p: [f64; 4],
    q: [f64; 4],
    inv_q: [f64; 4],
    need_deriv: bool,
) -> ([f64; 4], [f64; 4]) {
    let mut val = [0.0_f64; 4];
    let mut deriv = [0.0_f64; 4];
    for i in 0..4 {
        let xi = x[i];
        let raw = p[i] * inv_q[i];
        let raw_cl = raw.clamp(-1.0, 1.0);
        val[i] = if xi <= -4.0 {
            -1.0
        } else if xi >= 4.0 {
            1.0
        } else {
            raw_cl
        };
        if need_deriv {
            let dp = 1.0 + s[i] * (3.0 * TANH_PADE_N1_F64 + 5.0 * TANH_PADE_N2_F64 * s[i]);
            let dq = xc[i] * (2.0 * TANH_PADE_D1_F64 + 4.0 * TANH_PADE_D2_F64 * s[i]);
            let d_raw = (dp * q[i] - p[i] * dq) * inv_q[i] * inv_q[i];
            let in_range = xi > -4.0 && xi < 4.0 && raw > -1.0 && raw < 1.0;
            deriv[i] = if in_range || xi.is_nan() { d_raw } else { 0.0 };
        }
    }
    (val, deriv)
}

/// Vectorized 4-lane Padé `[5/4]` `tanh(x)` (`C^1` clamped at `±4`) sharing one division.
#[cfg(test)]
#[inline(always)]
pub fn tanh_pade5_quad_with_deriv(x: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    let (xc, s, p, q) = prepare_tanh_pade5_quad(x);
    let inv_q = inv4_pos_f64(q);
    finish_tanh_pade5_quad(x, xc, s, p, q, inv_q, true)
}

/// Evaluates `sigmoid(x) = 0.5 + 0.5 * tanh_pade5(0.5 * x)` matching engine math and its derivative.
#[inline(always)]
pub fn sigmoid_pade5_with_deriv(x: f64) -> (f64, f64) {
    let (t, dt) = tanh_pade5_with_deriv(0.5 * x);
    let sig = (0.5 + 0.5 * t).clamp(0.0, 1.0);
    (sig, 0.25 * dt)
}

/// Vectorized 4-lane `sigmoid(x) = 0.5 + 0.5 * tanh_pade5(0.5 * x)` sharing one division.
#[cfg(test)]
#[inline(always)]
pub fn sigmoid_pade5_quad_with_deriv(x: [f64; 4]) -> ([f64; 4], [f64; 4]) {
    let (t, dt) = tanh_pade5_quad_with_deriv([0.5 * x[0], 0.5 * x[1], 0.5 * x[2], 0.5 * x[3]]);
    let sig = [
        (0.5 + 0.5 * t[0]).clamp(0.0, 1.0),
        (0.5 + 0.5 * t[1]).clamp(0.0, 1.0),
        (0.5 + 0.5 * t[2]).clamp(0.0, 1.0),
        (0.5 + 0.5 * t[3]).clamp(0.0, 1.0),
    ];
    (
        sig,
        [0.25 * dt[0], 0.25 * dt[1], 0.25 * dt[2], 0.25 * dt[3]],
    )
}

/// 4-lane GRU group nonlinearities `(sigmoid(az), sigmoid(ar), 1/g_sag, tanh(an))` and optional derivatives.
pub struct GruQuadNonlinearOut {
    pub z: [f64; 4],
    pub dz: [f64; 4],
    pub r: [f64; 4],
    pub dr: [f64; 4],
    pub h_sag: [f64; 4],
    pub tan_n: [f64; 4],
    pub dtan_n: [f64; 4],
}

/// Evaluates an entire 4-lane GRU group's `sigmoid(az)`, `sigmoid(ar)`, `1/g_sag`, and `tanh(an)`
/// using one 16-reciprocal Montgomery division total.
#[inline(always)]
pub fn eval_gru_group_nonlinear_fwd(
    az: [f64; 4],
    ar: [f64; 4],
    g_sag: [f64; 4],
    an: [f64; 4],
    need_deriv: bool,
) -> GruQuadNonlinearOut {
    let az_h = [0.5 * az[0], 0.5 * az[1], 0.5 * az[2], 0.5 * az[3]];
    let ar_h = [0.5 * ar[0], 0.5 * ar[1], 0.5 * ar[2], 0.5 * ar[3]];
    let (xcz, sz, pz, qz) = prepare_tanh_pade5_quad(az_h);
    let (xcr, sr, pr, qr) = prepare_tanh_pade5_quad(ar_h);
    let (xcn, sn, pn, qn) = prepare_tanh_pade5_quad(an);
    let (inv_qz, inv_qr, h_sag, inv_qn) = inv16_pos_f64(qz, qr, g_sag, qn);
    let (tz, dtz) = finish_tanh_pade5_quad(az_h, xcz, sz, pz, qz, inv_qz, need_deriv);
    let (tr, dtr) = finish_tanh_pade5_quad(ar_h, xcr, sr, pr, qr, inv_qr, need_deriv);
    let (tan_n, dtan_n) = finish_tanh_pade5_quad(an, xcn, sn, pn, qn, inv_qn, need_deriv);
    let z = [
        (0.5 + 0.5 * tz[0]).clamp(0.0, 1.0),
        (0.5 + 0.5 * tz[1]).clamp(0.0, 1.0),
        (0.5 + 0.5 * tz[2]).clamp(0.0, 1.0),
        (0.5 + 0.5 * tz[3]).clamp(0.0, 1.0),
    ];
    let r = [
        (0.5 + 0.5 * tr[0]).clamp(0.0, 1.0),
        (0.5 + 0.5 * tr[1]).clamp(0.0, 1.0),
        (0.5 + 0.5 * tr[2]).clamp(0.0, 1.0),
        (0.5 + 0.5 * tr[3]).clamp(0.0, 1.0),
    ];
    let dz = if need_deriv {
        [0.25 * dtz[0], 0.25 * dtz[1], 0.25 * dtz[2], 0.25 * dtz[3]]
    } else {
        [0.0; 4]
    };
    let dr = if need_deriv {
        [0.25 * dtr[0], 0.25 * dtr[1], 0.25 * dtr[2], 0.25 * dtr[3]]
    } else {
        [0.0; 4]
    };
    GruQuadNonlinearOut {
        z,
        dz,
        r,
        dr,
        h_sag,
        tan_n,
        dtan_n,
    }
}

const HB_A0_F64: [f64; 2] = [HB_A0[0] as f64, HB_A0[1] as f64];
const HB_A1_F64: [f64; 2] = [HB_A1[0] as f64, HB_A1[1] as f64];

#[inline(always)]
fn step_allpass_1_f64(x: f64, a: f64, state: &mut f64) -> f64 {
    let s_old = *state;
    let s_new = x - a * s_old;
    *state = s_new;
    a * s_new + s_old
}

#[inline(always)]
fn rev_allpass_1_f64(dy: f64, a: f64, r_state: &mut f64) -> f64 {
    let r_old = *r_state;
    let d_s_new = a * dy + r_old;
    *r_state = dy - a * d_s_new;
    d_s_new
}

/// Forward 2nd-order allpass section cascade (`f64`).
#[inline(always)]
pub fn step_allpass_2_f64(x: f64, a: [f64; 2], s0: &mut f64, s1: &mut f64) -> f64 {
    let m = step_allpass_1_f64(x, a[0], s0);
    step_allpass_1_f64(m, a[1], s1)
}

/// Reverse-mode transpose of [`step_allpass_2_f64`] (LTI: requires no forward tape).
#[inline(always)]
pub fn rev_allpass_2_f64(dy: f64, a: [f64; 2], r0: &mut f64, r1: &mut f64) -> f64 {
    let dm = rev_allpass_1_f64(dy, a[1], r1);
    rev_allpass_1_f64(dm, a[0], r0)
}

/// Causal `2x` hybrid quadratic/ZOH/half-band upsampler matching `ZeroLagOversampler2x::upsample`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn upsample_2x_fwd(
    x: f64,
    up_quad: f64,
    up_zoh_mix: f64,
    up_pre_emph: f64,
    up_aa_mix: f64,
    raw_prev: &mut f64,
    prev1: &mut f64,
    prev2: &mut f64,
    hb_state: &mut [f64; 4],
    record_diff: bool,
) -> ([f64; 2], f64, f64, f64, [f64; 2]) {
    let dxp_diff = *raw_prev - x;
    let xp = (1.0 - up_pre_emph) * x + up_pre_emph * (*raw_prev);
    *raw_prev = x;
    let d2 = xp - 2.0 * (*prev1) + (*prev2);
    let zoh_diff = 0.5 * (xp - *prev1);
    let x0_poly = 0.5 * (*prev1 + xp) + up_zoh_mix * zoh_diff + up_quad * d2;
    let x1_poly = xp;
    *prev2 = *prev1;
    *prev1 = xp;
    if up_aa_mix > 0.0 || record_diff {
        let [s0, s1, s2, s3] = hb_state;
        let x0_hb = step_allpass_2_f64(xp, HB_A0_F64, s0, s1);
        let x1_hb = step_allpass_2_f64(xp, HB_A1_F64, s2, s3);
        let hb_diff = [x0_hb - x0_poly, x1_hb - x1_poly];
        (
            [
                x0_poly + up_aa_mix * hb_diff[0],
                x1_poly + up_aa_mix * hb_diff[1],
            ],
            d2,
            zoh_diff,
            dxp_diff,
            hb_diff,
        )
    } else {
        ([x0_poly, x1_poly], d2, zoh_diff, dxp_diff, [0.0, 0.0])
    }
}

/// Reverse-mode adjoint of [`upsample_2x_fwd`] returning `(dx, d_up_quad, d_up_zoh_mix, d_up_pre_emph, d_up_aa_mix)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn upsample_2x_rev(
    dx0: f64,
    dx1: f64,
    d2: f64,
    zoh_diff: f64,
    dxp_diff: f64,
    hb_diff: [f64; 2],
    up_quad: f64,
    up_zoh_mix: f64,
    up_pre_emph: f64,
    up_aa_mix: f64,
    r_raw_prev: &mut f64,
    r_prev1: &mut f64,
    r_prev2: &mut f64,
    r_hb: &mut [f64; 4],
) -> (f64, f64, f64, f64, f64) {
    let d_up_aa_mix = hb_diff[0] * dx0 + hb_diff[1] * dx1;
    let one_m = 1.0 - up_aa_mix;
    let dx0_p = one_m * dx0;
    let dx1_p = one_m * dx1;
    let d_up_quad = d2 * dx0_p;
    let d_up_zoh_mix = zoh_diff * dx0_p;
    let c_xp = 0.5 * (1.0 + up_zoh_mix);
    let c_p1 = 0.5 * (1.0 - up_zoh_mix);
    let d_xp_poly = (c_xp + up_quad) * dx0_p + dx1_p + *r_prev1;
    let next_r1 = (c_p1 - 2.0 * up_quad) * dx0_p + *r_prev2;
    *r_prev2 = up_quad * dx0_p;
    *r_prev1 = next_r1;
    let d_xp_hb = if up_aa_mix > 0.0 {
        let [r0, r1, r2, r3] = r_hb;
        rev_allpass_2_f64(up_aa_mix * dx0, HB_A0_F64, r0, r1)
            + rev_allpass_2_f64(up_aa_mix * dx1, HB_A1_F64, r2, r3)
    } else {
        0.0
    };
    let d_xp = d_xp_poly + d_xp_hb;
    let d_up_pre_emph = dxp_diff * d_xp;
    let dx = (1.0 - up_pre_emph) * d_xp + *r_raw_prev;
    *r_raw_prev = up_pre_emph * d_xp;
    (dx, d_up_quad, d_up_zoh_mix, d_up_pre_emph, d_up_aa_mix)
}

/// Forward step of hybrid 3-tap FIR + 9th-order half-band decimator returning `(out, old_y1, aa_diff)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn downsample_2x_fwd(
    y0: f64,
    y1: f64,
    prev_y1: &mut f64,
    aa_state: &mut [f64; 4],
    c_prev1: f64,
    c0: f64,
    aa_mix: f64,
    record_diff: bool,
) -> (f64, f64, f64) {
    let old_y1 = *prev_y1;
    let c1 = 1.0 - c_prev1 - c0;
    let y_poly = c_prev1 * old_y1 + c0 * y0 + c1 * y1;
    *prev_y1 = y1;
    if aa_mix > 0.0 || record_diff {
        let [s0, s1, s2, s3] = aa_state;
        let b0 = step_allpass_2_f64(y1, HB_A0_F64, s0, s1);
        let b1 = step_allpass_2_f64(y0, HB_A1_F64, s2, s3);
        let y_aa = 0.5 * (b0 + b1);
        let aa_diff = y_aa - y_poly;
        (y_poly + aa_mix * aa_diff, old_y1, aa_diff)
    } else {
        (y_poly, old_y1, 0.0)
    }
}

/// Reverse-mode adjoint of [`downsample_2x_fwd`] returning `(dy0, dy1, dc_prev1, dc0, d_aa_mix)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn downsample_2x_rev(
    d_out: f64,
    old_y1: f64,
    y0: f64,
    y1: f64,
    aa_diff: f64,
    c_prev1: f64,
    c0: f64,
    aa_mix: f64,
    r_prev_y1: &mut f64,
    r_aa: &mut [f64; 4],
) -> (f64, f64, f64, f64, f64) {
    let d_aa_mix = aa_diff * d_out;
    let d_poly = (1.0 - aa_mix) * d_out;
    let c1 = 1.0 - c_prev1 - c0;
    let dc_prev1 = (old_y1 - y1) * d_poly;
    let dc0 = (y0 - y1) * d_poly;
    let dy0_poly = c0 * d_poly;
    let dy1_poly = c1 * d_poly + *r_prev_y1;
    *r_prev_y1 = c_prev1 * d_poly;
    let (dy0_aa, dy1_aa) = if aa_mix > 0.0 {
        let [r0, r1, r2, r3] = r_aa;
        let half_d = 0.5 * aa_mix * d_out;
        let dy1 = rev_allpass_2_f64(half_d, HB_A0_F64, r0, r1);
        let dy0 = rev_allpass_2_f64(half_d, HB_A1_F64, r2, r3);
        (dy0, dy1)
    } else {
        (0.0, 0.0)
    };
    (
        dy0_poly + dy0_aa,
        dy1_poly + dy1_aa,
        dc_prev1,
        dc0,
        d_aa_mix,
    )
}

/// Outputs and saved sub-section intermediates from [`step_tpt_pair_fwd`].
#[derive(Clone, Copy, Debug)]
pub struct SvfPairFwdOut {
    pub taps: [f64; 7],
    pub s1_c1: f64,
    pub s1_a1: f64,
    pub s1_a2: f64,
    pub s1_a3: f64,
    pub s1_v1: f64,
    pub s1_v3: f64,
    pub s2_c1: f64,
    pub s2_a1: f64,
    pub s2_a2: f64,
    pub s2_a3: f64,
    pub s2_v1: f64,
    pub s2_v3: f64,
}

#[inline(always)]
fn eval_tpt_section(u: f64, g: f64, k: f64, a1: f64, ic1: &mut f64, ic2: &mut f64) -> [f64; 7] {
    let c1 = *ic1;
    let c2 = *ic2;
    let a2 = g * a1;
    let a3 = g * a2;
    let v3 = u - c2;
    let v1 = a1 * c1 + a2 * v3;
    let v2 = c2 + a2 * c1 + a3 * v3;
    *ic1 = 2.0 * v1 - c1;
    *ic2 = 2.0 * v2 - c2;
    [v2, v1, u - k * v1 - v2, c1, a2, a3, v3]
}

/// Forward step of one zero-delay Cytomic TPT 2nd-order SVF section.
#[cfg(test)]
#[inline(always)]
pub fn step_tpt_fwd(
    u: f64,
    g: f64,
    k: f64,
    ic1: &mut f64,
    ic2: &mut f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let a1 = 1.0 / (1.0 + g * (g + k));
    let [v2, v1, hp, c1, a2, a3, v3] = eval_tpt_section(u, g, k, a1, ic1, ic2);
    (v2, v1, hp, c1, a1, a2, a3, v1, v3)
}

/// Forward step of two series Cytomic TPT SVF sections sharing one denominator division.
#[inline(always)]
pub fn step_tpt_pair_fwd(
    x: f64,
    g: [f64; 2],
    k: [f64; 2],
    s2_feed: [f64; 2],
    ic1: &mut [f64; 2],
    ic2: &mut [f64; 2],
) -> SvfPairFwdOut {
    let (s1_a1, s2_a1) = inv2_pos_f64(1.0 + g[0] * (g[0] + k[0]), 1.0 + g[1] * (g[1] + k[1]));
    let [lp1, bp1, hp1, s1_c1, s1_a2, s1_a3, s1_v3] =
        eval_tpt_section(x, g[0], k[0], s1_a1, &mut ic1[0], &mut ic2[0]);
    let u2 = lp1 + s2_feed[0] * bp1 + s2_feed[1] * hp1;
    let [lp2, bp2, hp2, s2_c1, s2_a2, s2_a3, s2_v3] =
        eval_tpt_section(u2, g[1], k[1], s2_a1, &mut ic1[1], &mut ic2[1]);
    SvfPairFwdOut {
        taps: [x, lp1, bp1, hp1, lp2, bp2, hp2],
        s1_c1,
        s1_a1,
        s1_a2,
        s1_a3,
        s1_v1: bp1,
        s1_v3,
        s2_c1,
        s2_a1,
        s2_a2,
        s2_a3,
        s2_v1: bp2,
        s2_v3,
    }
}

/// Reverse-mode adjoint step of [`step_tpt_pair_fwd`] returning `(du, dg, dk)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn step_tpt_rev(
    d_lp: f64,
    d_bp: f64,
    d_hp: f64,
    g: f64,
    k: f64,
    c1: f64,
    v1: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    v3: f64,
    r_ic1: &mut f64,
    r_ic2: &mut f64,
) -> (f64, f64, f64) {
    let dv1_tot = d_bp - k * d_hp + 2.0 * *r_ic1;
    let dv2_tot = d_lp - d_hp + 2.0 * *r_ic2;
    let dc1_out = -(*r_ic1) + a1 * dv1_tot + a2 * dv2_tot;
    let dv3 = a2 * dv1_tot + a3 * dv2_tot;
    let du = d_hp + dv3;
    let dc2_out = dv2_tot - (*r_ic2) - dv3;
    let da1 = c1 * dv1_tot;
    let da2 = v3 * dv1_tot + c1 * dv2_tot;
    let da3 = v3 * dv2_tot;
    let da2_tot = da2 + g * da3;
    let da1_tot = da1 + g * da2_tot;
    let d_den = -a1 * a1 * da1_tot;
    let dg = a1 * da2_tot + a2 * da3 + (2.0 * g + k) * d_den;
    let dk = g * d_den - v1 * d_hp;
    *r_ic1 = dc1_out;
    *r_ic2 = dc2_out;
    (du, dg, dk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quad_pade5_and_inv4_match_scalar_reference() {
        let d = [1.0, 1.75, 3.25, 9.5];
        let inv = inv4_pos_f64(d);
        for (i, &di) in d.iter().enumerate() {
            assert!((inv[i] - 1.0 / di).abs() <= 1.0e-15);
        }
        for xs in [
            [-6.0, -4.0, -2.5, -0.75],
            [-0.01, 0.0, 1.35, 3.999],
            [4.0, 6.0, -3.999, 2.25],
        ] {
            let (tq, dtq) = tanh_pade5_quad_with_deriv(xs);
            let (sq, dsq) = sigmoid_pade5_quad_with_deriv(xs);
            for i in 0..4 {
                let (ts, dts) = tanh_pade5_with_deriv(xs[i]);
                let (ss, dss) = sigmoid_pade5_with_deriv(xs[i]);
                assert!(
                    (tq[i] - ts).abs() <= 1.0e-15,
                    "tanh val mismatch at {}",
                    xs[i]
                );
                assert!(
                    (dtq[i] - dts).abs() <= 1.0e-15,
                    "tanh deriv mismatch at {}",
                    xs[i]
                );
                assert!(
                    (sq[i] - ss).abs() <= 1.0e-15,
                    "sig val mismatch at {}",
                    xs[i]
                );
                assert!(
                    (dsq[i] - dss).abs() <= 1.0e-15,
                    "sig deriv mismatch at {}",
                    xs[i]
                );
            }
        }
    }

    #[test]
    fn test_inv2_inv16_step_tpt_pair_and_gru_nonlinear_match_scalar() {
        let (r0, r1) = inv2_pos_f64(2.25, 5.5);
        assert!((r0 - 1.0 / 2.25).abs() <= 1.0e-15);
        assert!((r1 - 1.0 / 5.5).abs() <= 1.0e-15);

        let d0 = [1.1, 2.3, 3.7, 4.9];
        let d1 = [1.4, 2.6, 5.1, 7.2];
        let d2 = [1.0, 1.8, 4.4, 12.5];
        let d3 = [1.2, 2.9, 6.3, 9.8];
        let (i0, i1, i2, i3) = inv16_pos_f64(d0, d1, d2, d3);
        for k in 0..4 {
            assert!((i0[k] - 1.0 / d0[k]).abs() <= 1.0e-14);
            assert!((i1[k] - 1.0 / d1[k]).abs() <= 1.0e-14);
            assert!((i2[k] - 1.0 / d2[k]).abs() <= 1.0e-14);
            assert!((i3[k] - 1.0 / d3[k]).abs() <= 1.0e-14);
        }

        let (mut p_ic1, mut p_ic2) = ([0.15, -0.08], [-0.11, 0.21]);
        let (mut s_ic1_1, mut s_ic2_1, mut s_ic1_2, mut s_ic2_2) = (0.15, -0.11, -0.08, 0.21);
        let pair = step_tpt_pair_fwd(
            0.65,
            [0.32, 0.61],
            [0.45, 0.90],
            [0.25, -0.35],
            &mut p_ic1,
            &mut p_ic2,
        );
        let (lp1, bp1, hp1, c1_1, a1_1, a2_1, a3_1, v1_1, v3_1) =
            step_tpt_fwd(0.65, 0.32, 0.45, &mut s_ic1_1, &mut s_ic2_1);
        let u2 = lp1 + 0.25 * bp1 - 0.35 * hp1;
        let (lp2, bp2, hp2, c1_2, a1_2, a2_2, a3_2, v1_2, v3_2) =
            step_tpt_fwd(u2, 0.61, 0.90, &mut s_ic1_2, &mut s_ic2_2);
        let ref_taps = [0.65, lp1, bp1, hp1, lp2, bp2, hp2];
        for (a, b) in pair.taps.iter().zip(&ref_taps) {
            assert!((a - b).abs() <= 1.0e-14);
        }
        assert!((pair.s1_c1 - c1_1).abs() <= 1.0e-15);
        assert!((pair.s1_a1 - a1_1).abs() <= 1.0e-14);
        assert!((pair.s1_a2 - a2_1).abs() <= 1.0e-14);
        assert!((pair.s1_a3 - a3_1).abs() <= 1.0e-14);
        assert!((pair.s1_v1 - v1_1).abs() <= 1.0e-14);
        assert!((pair.s1_v3 - v3_1).abs() <= 1.0e-14);
        assert!((pair.s2_c1 - c1_2).abs() <= 1.0e-15);
        assert!((pair.s2_a1 - a1_2).abs() <= 1.0e-14);
        assert!((pair.s2_a2 - a2_2).abs() <= 1.0e-14);
        assert!((pair.s2_a3 - a3_2).abs() <= 1.0e-14);
        assert!((pair.s2_v1 - v1_2).abs() <= 1.0e-14);
        assert!((pair.s2_v3 - v3_2).abs() <= 1.0e-14);
        assert!((p_ic1[0] - s_ic1_1).abs() <= 1.0e-14);
        assert!((p_ic2[0] - s_ic2_1).abs() <= 1.0e-14);
        assert!((p_ic1[1] - s_ic1_2).abs() <= 1.0e-14);
        assert!((p_ic2[1] - s_ic2_2).abs() <= 1.0e-14);

        let az = [-5.0, -0.4, 1.8, 9.0];
        let ar = [-1.2, 0.0, 2.5, -9.5];
        let g_sag = [1.0, 1.4, 2.8, 6.2];
        let an = [-4.5, -0.8, 1.35, 4.1];
        let fwd_deriv = eval_gru_group_nonlinear_fwd(az, ar, g_sag, an, true);
        let fwd_noderiv = eval_gru_group_nonlinear_fwd(az, ar, g_sag, an, false);
        assert_eq!(fwd_deriv.z, fwd_noderiv.z);
        assert_eq!(fwd_deriv.r, fwd_noderiv.r);
        assert_eq!(fwd_deriv.h_sag, fwd_noderiv.h_sag);
        assert_eq!(fwd_deriv.tan_n, fwd_noderiv.tan_n);
        assert_eq!(fwd_noderiv.dz, [0.0; 4]);
        assert_eq!(fwd_noderiv.dr, [0.0; 4]);
        assert_eq!(fwd_noderiv.dtan_n, [0.0; 4]);
        for i in 0..4 {
            let (sz, dsz) = sigmoid_pade5_with_deriv(az[i]);
            let (sr, dsr) = sigmoid_pade5_with_deriv(ar[i]);
            let (tn, dtn) = tanh_pade5_with_deriv(an[i]);
            assert!((fwd_deriv.z[i] - sz).abs() <= 1.0e-14);
            assert!((fwd_deriv.dz[i] - dsz).abs() <= 1.0e-14);
            assert!((fwd_deriv.r[i] - sr).abs() <= 1.0e-14);
            assert!((fwd_deriv.dr[i] - dsr).abs() <= 1.0e-14);
            assert!((fwd_deriv.h_sag[i] - 1.0 / g_sag[i]).abs() <= 1.0e-14);
            assert!((fwd_deriv.tan_n[i] - tn).abs() <= 1.0e-14);
            assert!((fwd_deriv.dtan_n[i] - dtn).abs() <= 1.0e-14);
        }
    }
}
