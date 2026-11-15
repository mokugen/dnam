#![forbid(unsafe_code)]
//! Differentiable Double-Precision Forward/Adjoint DSP Primitives (`TPT-SVF`, `tan` Prewarp,
//! Padé `[5/4]` `tanh`/`sigmoid`, Zero-Lag `2x` Oversampling, & Multi-Timescale GRU Tapes).

use std::f64::consts::{LN_2, PI};

use dnam_engine::stages::v3_core::cascade::{
    HB_A0, HB_A1, MAX_BLOCK_LANES, MAX_KAUTZ_POLES, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0,
};
use dnam_engine::stages::v3_core::math_pade::{
    TANH_PADE_D1_F64, TANH_PADE_D2_F64, TANH_PADE_N1_F64, TANH_PADE_N2_F64,
};

use super::model_f64::V3MatryoshkaOptF64;

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

/// Evaluates division-free 5th-order bilinear `tan(w_half)` prewarp matching [`dnam_engine::stages::v3_core::svf::tan_prewarp_f32`] and its derivative.
#[inline(always)]
pub fn tan_prewarp_with_deriv(w_half: f64) -> (f64, f64) {
    let x = w_half.clamp(1.0e-4, 0.95);
    let x2 = x * x;
    let y = x * (1.0 + x2 * ((1.0 / 3.0) + (2.0 / 15.0) * x2));
    let dy = if w_half > 1.0e-4 && w_half < 0.95 {
        1.0 + x2 + (2.0 / 3.0) * x2 * x2
    } else {
        0.0
    };
    (y, dy)
}

/// Evaluates Padé `[5/4]` `tanh(x)` (`C^1` clamped at `±4`) matching engine math and its derivative.
#[inline(always)]
pub fn tanh_pade5_with_deriv(x: f64) -> (f64, f64) {
    if x <= -4.0 {
        return (-1.0, 0.0);
    }
    if x >= 4.0 {
        return (1.0, 0.0);
    }
    let s = x * x;
    let p = x * (1.0 + s * (TANH_PADE_N1_F64 + s * TANH_PADE_N2_F64));
    let q = 1.0 + s * (TANH_PADE_D1_F64 + s * TANH_PADE_D2_F64);
    let inv_q = 1.0 / q;
    let raw = p * inv_q;
    if raw <= -1.0 {
        return (-1.0, 0.0);
    }
    if raw >= 1.0 {
        return (1.0, 0.0);
    }
    let dp = 1.0 + s * (3.0 * TANH_PADE_N1_F64 + 5.0 * TANH_PADE_N2_F64 * s);
    let dq = x * (2.0 * TANH_PADE_D1_F64 + 4.0 * TANH_PADE_D2_F64 * s);
    let deriv = (dp * q - p * dq) * inv_q * inv_q;
    (raw, deriv)
}

/// Evaluates `sigmoid(x) = 0.5 + 0.5 * tanh_pade5(0.5 * x)` matching engine math and its derivative.
#[inline(always)]
pub fn sigmoid_pade5_with_deriv(x: f64) -> (f64, f64) {
    let (t, dt) = tanh_pade5_with_deriv(0.5 * x);
    let sig = (0.5 + 0.5 * t).clamp(0.0, 1.0);
    (sig, 0.25 * dt)
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

/// Causal `2x` hybrid quadratic/half-band upsampler matching `ZeroLagOversampler2x::upsample`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn upsample_2x_fwd(
    x: f64,
    up_quad: f64,
    up_pre_emph: f64,
    up_aa_mix: f64,
    raw_prev: &mut f64,
    prev1: &mut f64,
    prev2: &mut f64,
    hb_state: &mut [f64; 4],
    record_diff: bool,
) -> ([f64; 2], f64, f64, [f64; 2]) {
    let dxp_diff = *raw_prev - x;
    let xp = (1.0 - up_pre_emph) * x + up_pre_emph * (*raw_prev);
    *raw_prev = x;
    let d2 = xp - 2.0 * (*prev1) + (*prev2);
    let x0_poly = 0.5 * (*prev1 + xp) + up_quad * d2;
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
            dxp_diff,
            hb_diff,
        )
    } else {
        ([x0_poly, x1_poly], d2, dxp_diff, [0.0, 0.0])
    }
}

/// Reverse-mode adjoint of [`upsample_2x_fwd`] returning `(dx, d_up_quad, d_up_pre_emph, d_up_aa_mix)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn upsample_2x_rev(
    dx0: f64,
    dx1: f64,
    d2: f64,
    dxp_diff: f64,
    hb_diff: [f64; 2],
    up_quad: f64,
    up_pre_emph: f64,
    up_aa_mix: f64,
    r_raw_prev: &mut f64,
    r_prev1: &mut f64,
    r_prev2: &mut f64,
    r_hb: &mut [f64; 4],
) -> (f64, f64, f64, f64) {
    let d_up_aa_mix = hb_diff[0] * dx0 + hb_diff[1] * dx1;
    let one_m = 1.0 - up_aa_mix;
    let dx0_p = one_m * dx0;
    let dx1_p = one_m * dx1;
    let d_up_quad = d2 * dx0_p;
    let d_xp_poly = (0.5 + up_quad) * dx0_p + dx1_p + *r_prev1;
    let next_r1 = (0.5 - 2.0 * up_quad) * dx0_p + *r_prev2;
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
    (dx, d_up_quad, d_up_pre_emph, d_up_aa_mix)
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

/// Forward/reverse per-substep tape for Stage I 2-section Cytomic TPT SVF.
#[derive(Clone, Copy, Debug, Default)]
pub struct SvfChanTape {
    pub taps: [f64; 7],
    pub det_lp_prev: f64,
    pub det_lp: f64,
    pub v_det_hp: f64,
    pub det_raw_sign: f64,
    pub env_prev: f64,
    pub atk_taken: bool,
    pub rsq3: f64,
    pub e: f64,
    pub w1_unclamped: f64,
    pub w1_clamped: f64,
    pub g1: f64,
    pub g2: f64,
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

/// Forward step of one zero-delay Cytomic TPT 2nd-order SVF section.
#[inline(always)]
pub fn step_tpt_fwd(
    u: f64,
    g: f64,
    k: f64,
    ic1: &mut f64,
    ic2: &mut f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let c1 = *ic1;
    let c2 = *ic2;
    let a1 = 1.0 / (1.0 + g * (g + k));
    let a2 = g * a1;
    let a3 = g * a2;
    let v3 = u - c2;
    let v1 = a1 * c1 + a2 * v3;
    let v2 = c2 + a2 * c1 + a3 * v3;
    *ic1 = 2.0 * v1 - c1;
    *ic2 = 2.0 * v2 - c2;
    (v2, v1, u - k * v1 - v2, c1, a1, a2, a3, v1, v3)
}

/// Reverse-mode adjoint step of [`step_tpt_fwd`] returning `(du, dg, dk)`.
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

/// Forward/reverse substep tape for multi-timescale Stage II input features.
#[derive(Clone, Copy, Debug, Default)]
pub struct InputFeatSubTape {
    pub feats: [f64; NUM_IN_FEATS_B0],
    pub u_lp_prev: f64,
    pub env_fast_prev: f64,
    pub env_slow_prev: f64,
    pub sign_u: f64,
    pub fast_atk_taken: bool,
    pub slow_atk_taken: bool,
    pub pinch_inv: f64,
    pub rect_slow: f64,
    pub slow_rsq3: f64,
    pub up_d2: f64,
    pub up_dxp_diff: f64,
    pub up_hb_diff: [f64; 2],
}

/// Forward/reverse substep tape for one Stage II Matryoshka GRU block (`16` lanes max).
#[derive(Clone, Copy, Debug)]
pub struct GruBlockSubTape {
    pub z: [f64; MAX_BLOCK_LANES],
    pub dz_daz: [f64; MAX_BLOCK_LANES],
    pub r: [f64; MAX_BLOCK_LANES],
    pub dr_dar: [f64; MAX_BLOCK_LANES],
    pub an_in: [f64; MAX_BLOCK_LANES],
    pub rec_n: [f64; MAX_BLOCK_LANES],
    pub a_sig: [f64; MAX_BLOCK_LANES],
    pub a_sig_mask: [f64; MAX_BLOCK_LANES],
    pub a_eff: [f64; MAX_BLOCK_LANES],
    pub h_sag: [f64; MAX_BLOCK_LANES],
    pub delta_tanh: [f64; MAX_BLOCK_LANES],
    pub dtanh_an: [f64; MAX_BLOCK_LANES],
    pub dtanh_bn: [f64; MAX_BLOCK_LANES],
    pub n: [f64; MAX_BLOCK_LANES],
    pub h_old: [f64; MAX_BLOCK_LANES],
    pub h_new: [f64; MAX_BLOCK_LANES],
    pub clamp_mask: [f64; MAX_BLOCK_LANES],
}

impl Default for GruBlockSubTape {
    fn default() -> Self {
        Self {
            z: [0.0; MAX_BLOCK_LANES],
            dz_daz: [0.0; MAX_BLOCK_LANES],
            r: [0.0; MAX_BLOCK_LANES],
            dr_dar: [0.0; MAX_BLOCK_LANES],
            an_in: [0.0; MAX_BLOCK_LANES],
            rec_n: [0.0; MAX_BLOCK_LANES],
            a_sig: [0.0; MAX_BLOCK_LANES],
            a_sig_mask: [1.0; MAX_BLOCK_LANES],
            a_eff: [0.0; MAX_BLOCK_LANES],
            h_sag: [1.0; MAX_BLOCK_LANES],
            delta_tanh: [0.0; MAX_BLOCK_LANES],
            dtanh_an: [0.0; MAX_BLOCK_LANES],
            dtanh_bn: [0.0; MAX_BLOCK_LANES],
            n: [0.0; MAX_BLOCK_LANES],
            h_old: [0.0; MAX_BLOCK_LANES],
            h_new: [0.0; MAX_BLOCK_LANES],
            clamp_mask: [1.0; MAX_BLOCK_LANES],
        }
    }
}

#[inline(always)]
fn cl_mask(v: f64, lo: f64, hi: f64) -> (f64, f64) {
    if v <= lo {
        (lo, 0.0)
    } else if v >= hi {
        (hi, 0.0)
    } else {
        (v, 1.0)
    }
}

/// Clamped parameter values paired with straight-through boundary masks `(val, mask)`.
pub(crate) struct ClampedCoreParams {
    pub up_quad: (f64, f64),
    pub dec_c_prev1: (f64, f64),
    pub dec_c0: (f64, f64),
    pub up_pre_emph: (f64, f64),
    pub dec_aa_mix: (f64, f64),
    pub up_aa_mix: (f64, f64),
    pub base_g: (f64, f64),
    pub sec2_r: (f64, f64),
    pub env_oct: (f64, f64),
    pub env_comp: (f64, f64),
    pub atk: (f64, f64),
    pub rel: (f64, f64),
    pub det_lp: (f64, f64),
    pub det_hp_mix: (f64, f64),
    pub dk1: (f64, f64),
    pub dk2: (f64, f64),
    pub s2_bp: (f64, f64),
    pub s2_hp: (f64, f64),
    pub link: (f64, f64),
    pub g_min: (f64, f64),
    pub g_max: (f64, f64),
    pub hp_beta: (f64, f64),
    pub fast_atk: (f64, f64),
    pub fast_rel: (f64, f64),
    pub pinch_k: (f64, f64),
    pub slow_atk: (f64, f64),
    pub slow_rel: (f64, f64),
    pub slow_det_comp: (f64, f64),
    pub b_z: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub b_r: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub b_n: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub k_sag: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub k_quad: [[(f64, f64); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub tanh_bn: [[f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub dtanh_bn: [[f64; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS],
    pub kautz_pre: [(f64, f64, f64, f64, f64, f64); MAX_KAUTZ_POLES],
}

impl ClampedCoreParams {
    pub(crate) fn from_model(m: &V3MatryoshkaOptF64) -> Self {
        let g_min = cl_mask(m.svf_g_min, 1.0e-4, 0.95);
        let g_max = if m.svf_g_max < g_min.0 {
            (g_min.0, 0.0)
        } else {
            cl_mask(m.svf_g_max, 1.0e-4, 0.95)
        };
        let mut b_z = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut b_r = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut b_n = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut k_sag = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut k_quad = [[(0.0, 0.0); MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut tanh_bn = [[0.0; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        let mut dtanh_bn = [[0.0; MAX_BLOCK_LANES]; NUM_GRU_BLOCKS];
        for b in 0..NUM_GRU_BLOCKS {
            for l in 0..MAX_BLOCK_LANES {
                b_z[b][l] = cl_mask(m.blocks[b].b_z[l], -8.0, 8.2);
                b_r[b][l] = cl_mask(m.blocks[b].b_r[l], -8.0, 8.2);
                b_n[b][l] = cl_mask(m.blocks[b].b_n[l], -3.8, 3.8);
                k_sag[b][l] = cl_mask(m.blocks[b].k_sag[l], 0.0, 8.0);
                k_quad[b][l] = cl_mask(m.blocks[b].k_quad[l], -0.45, 0.45);
                (tanh_bn[b][l], dtanh_bn[b][l]) = tanh_pade5_with_deriv(b_n[b][l].0);
            }
        }
        let kautz_pre = m.kautz_poles.map(|(rr, tt)| {
            let (r, r_m) = cl_mask(rr, 0.25, 0.9985);
            let (th, t_m) = cl_mask(tt, 0.0, PI - 1.0e-3);
            let bg = (1.0 - r * r).max(1.0e-7).sqrt();
            let (st, ct) = th.sin_cos();
            (r, bg, ct, st, r_m, t_m)
        });
        Self {
            up_quad: cl_mask(m.os_up_quad, -0.20, 0.05),
            dec_c_prev1: cl_mask(m.os_dec_c_prev1, 0.0, 0.28),
            dec_c0: cl_mask(m.os_dec_c0, 0.0, 0.55),
            up_pre_emph: cl_mask(m.os_up_pre_emph, -0.28, 1.0),
            dec_aa_mix: cl_mask(m.os_dec_aa_mix, 0.0, 1.0),
            up_aa_mix: cl_mask(m.os_up_aa_mix, 0.0, 1.0),
            base_g: cl_mask(m.svf_base_g, 1.0e-4, 0.95),
            sec2_r: cl_mask(m.svf_sec2_g_ratio, 0.25, 4.0),
            env_oct: cl_mask(m.svf_env_oct_sens, -24.0, 24.0),
            env_comp: cl_mask(m.svf_env_comp, 0.0, 16.0),
            atk: cl_mask(m.svf_atk_alpha, 0.001, 0.9999),
            rel: cl_mask(m.svf_rel_alpha, 0.001, 0.99995),
            det_lp: cl_mask(m.svf_det_lp_pole, 0.05, 0.995),
            det_hp_mix: cl_mask(m.svf_det_hp_mix, -1.0, 2.0),
            dk1: cl_mask(m.svf_damp_k1, 0.05, 4.0),
            dk2: cl_mask(m.svf_damp_k2, 0.05, 4.0),
            s2_bp: cl_mask(m.svf_sec2_bp_feed, -2.0, 2.0),
            s2_hp: cl_mask(m.svf_sec2_hp_feed, -2.0, 2.0),
            link: cl_mask(m.svf_stereo_link, 0.0, 1.0),
            g_min,
            g_max,
            hp_beta: cl_mask(m.feat_in_hp_beta, 1.0e-4, 0.50),
            fast_atk: cl_mask(m.feat_fast_atk_beta, 1.0e-4, 0.999),
            fast_rel: cl_mask(m.feat_fast_rel_beta, 1.0e-5, 0.999),
            pinch_k: cl_mask(m.feat_pinch_k, 0.0, 25_000.0),
            slow_atk: cl_mask(m.feat_slow_atk_beta, 1.0e-5, 0.50),
            slow_rel: cl_mask(m.feat_slow_rel_beta, 1.0e-6, 0.10),
            slow_det_comp: cl_mask(m.feat_slow_det_comp, 0.0, 64.0),
            b_z,
            b_r,
            b_n,
            k_sag,
            k_quad,
            tanh_bn,
            dtanh_bn,
            kautz_pre,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
static ACTIVE_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static IN_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(not(target_arch = "wasm32"))]
struct WorkerReservation(usize);

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WorkerReservation {
    fn drop(&mut self) {
        if self.0 > 0 {
            ACTIVE_THREADS.fetch_sub(self.0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct InWorkerGuard(bool);

#[cfg(not(target_arch = "wasm32"))]
impl Drop for InWorkerGuard {
    fn drop(&mut self) {
        IN_WORKER.with(|c| c.set(self.0));
    }
}

/// Evaluates `f(idx)` for `idx in 0..len` in parallel using scoped threads, preserving index order.
pub(crate) fn par_map_indexed<R: Send>(len: usize, f: impl Fn(usize) -> R + Send + Sync) -> Vec<R> {
    if len <= 1 {
        return (0..len).map(&f).collect();
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..len).map(f).collect()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let total_cpus = std::thread::available_parallelism()
            .map_or(4, usize::from)
            .max(1);
        let nested = IN_WORKER.with(std::cell::Cell::get);
        let max_budget = if nested {
            total_cpus.saturating_sub(ACTIVE_THREADS.load(Ordering::Relaxed))
        } else {
            total_cpus
        };
        let n_workers = max_budget.min(len);
        if n_workers <= 1 {
            return (0..len).map(f).collect();
        }
        let spawned_extra = n_workers.saturating_sub(usize::from(nested));
        ACTIVE_THREADS.fetch_add(spawned_extra, Ordering::Relaxed);
        let _reservation = WorkerReservation(spawned_extra);

        let cursor = AtomicUsize::new(0);
        let outputs: Vec<Mutex<Option<R>>> = (0..len).map(|_| Mutex::new(None)).collect();
        let f_ref = &f;
        std::thread::scope(|scope| {
            for _ in 0..n_workers {
                scope.spawn(|| {
                    let prev_in = IN_WORKER.with(|c| {
                        let p = c.get();
                        c.set(true);
                        p
                    });
                    let _in_guard = InWorkerGuard(prev_in);
                    while let Some(idx) = {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        (i < len).then_some(i)
                    } {
                        let res = f_ref(idx);
                        if let Ok(mut slot) = outputs[idx].lock() {
                            *slot = Some(res);
                        }
                    }
                });
            }
        });
        outputs
            .into_iter()
            .filter_map(|m| m.into_inner().ok().flatten())
            .collect()
    }
}

/// Transforms `items` in parallel using scoped threads while preserving input order.
pub(crate) fn par_map_vec<T: Send, R: Send>(
    items: Vec<T>,
    f: impl Fn(usize, T) -> R + Send + Sync,
) -> Vec<R> {
    let len = items.len();
    if len <= 1 {
        return items
            .into_iter()
            .enumerate()
            .map(|(i, x)| f(i, x))
            .collect();
    }
    #[cfg(target_arch = "wasm32")]
    {
        items
            .into_iter()
            .enumerate()
            .map(|(i, x)| f(i, x))
            .collect()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::Mutex;

        let slots: Vec<Mutex<Option<T>>> = items.into_iter().map(|x| Mutex::new(Some(x))).collect();
        par_map_indexed(len, |idx| {
            let item = slots[idx].lock().ok().and_then(|mut g| g.take())?;
            Some(f(idx, item))
        })
        .into_iter()
        .flatten()
        .collect()
    }
}
