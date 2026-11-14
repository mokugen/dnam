#![forbid(unsafe_code)]
//! Exact `f64` forward/reverse primitives and forward tape structs for `v3_varpro` adjoints.

use dnam_engine::stages::v3_core::cascade::MAX_CASCADE_LANES;

#[allow(clippy::approx_constant)]
#[inline(always)]
pub(crate) fn exp2_oct_with_deriv(oct: f64) -> (f64, f64) {
    const C1: f64 = 0.693_147_2;
    const C2: f64 = 0.240_226_51;
    const C3: f64 = 0.055_712_965;
    const C4: f64 = 0.009_011_465;
    const C5: f64 = 0.001_901_882;
    let clamped = oct.clamp(-5.0, 5.0);
    let fl = clamped.floor();
    let f = (clamped - fl).clamp(0.0, 1.0);
    let pow2_n = (fl as i32)
        .checked_add(1023)
        .map_or(1.0, |e| f64::from_bits((e as u64) << 52));
    let poly = 1.0 + f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * C5))));
    let dpoly = C1 + f * (2.0 * C2 + f * (3.0 * C3 + f * (4.0 * C4 + f * 5.0 * C5)));
    let mask = if oct > -5.0 && oct < 5.0 { 1.0 } else { 0.0 };
    (pow2_n * poly, mask * pow2_n * dpoly)
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SvfChanTape {
    pub(crate) det_lp_prev: f64,
    pub(crate) det_lp: f64,
    pub(crate) det_raw_sign: f64,
    pub(crate) v_det_hp: f64,
    pub(crate) env_prev: f64,
    pub(crate) atk_taken: bool,
    pub(crate) e: f64,
    pub(crate) rsq3: f64,
    pub(crate) g1_unclamped: f64,
    pub(crate) g1: f64,
    pub(crate) g2: f64,
    pub(crate) s1_c1: f64,
    pub(crate) s1_v1: f64,
    pub(crate) s1_a1: f64,
    pub(crate) s1_a2: f64,
    pub(crate) s1_a3: f64,
    pub(crate) s1_v3: f64,
    pub(crate) s2_c1: f64,
    pub(crate) s2_v1: f64,
    pub(crate) s2_a1: f64,
    pub(crate) s2_a2: f64,
    pub(crate) s2_a3: f64,
    pub(crate) s2_v3: f64,
    pub(crate) taps: [f64; 7],
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BlockSubTape {
    pub(crate) u_s2: f64,
    pub(crate) lp_prev: [f64; MAX_CASCADE_LANES],
    pub(crate) lp_next: [f64; MAX_CASCADE_LANES],
    pub(crate) v: [f64; MAX_CASCADE_LANES],
    pub(crate) up: [f64; MAX_CASCADE_LANES],
    pub(crate) up_sign: [f64; MAX_CASCADE_LANES],
    pub(crate) env_q_prev: f64,
    pub(crate) atk_taken: bool,
    pub(crate) q: f64,
    pub(crate) g_gate: f64,
    pub(crate) gate_inv: f64,
    pub(crate) dphi_dzc: [f64; MAX_CASCADE_LANES],
    pub(crate) dphi0_db0: [f64; MAX_CASCADE_LANES],
    pub(crate) h_next: [f64; MAX_CASCADE_LANES],
}

#[inline(always)]
pub(crate) fn ap2_fwd(x: f64, a: [f64; 2], s: &mut [f64; 2]) -> f64 {
    let s_new0 = x - a[0] * s[0];
    let y0 = a[0] * s_new0 + s[0];
    s[0] = s_new0;
    let s_new1 = y0 - a[1] * s[1];
    let y1 = a[1] * s_new1 + s[1];
    s[1] = s_new1;
    y1
}

#[inline(always)]
pub(crate) fn ap2_rev(dy: f64, a: [f64; 2], s_rev: &mut [f64; 2]) -> f64 {
    let ds1 = s_rev[1] + a[1] * dy;
    s_rev[1] = dy - a[1] * ds1;
    let ds0 = s_rev[0] + a[0] * ds1;
    s_rev[0] = ds1 - a[0] * ds0;
    ds0
}

#[inline(always)]
pub(crate) fn step_tpt_fwd(
    u: f64,
    g: f64,
    k: f64,
    ic1: &mut f64,
    ic2: &mut f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let a1 = 1.0 / (1.0 + g * (g + k)).max(1.0e-6);
    let a2 = g * a1;
    let a3 = g * a2;
    let (c1, c2) = (*ic1, *ic2);
    let v3 = u - c2;
    let v1 = a1 * c1 + a2 * v3;
    let v2 = c2 + a2 * c1 + a3 * v3;
    *ic1 = 2.0 * v1 - c1;
    *ic2 = 2.0 * v2 - c2;
    (v2, v1, u - k * v1 - v2, c1, a1, a2, a3, v3)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn step_tpt_rev(
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
    d_ic1: &mut f64,
    d_ic2: &mut f64,
) -> (f64, f64, f64) {
    let d_v1 = d_bp - k * d_hp + 2.0 * *d_ic1;
    let d_v2 = d_lp - d_hp + 2.0 * *d_ic2;
    let d_c1 = -*d_ic1 + a1 * d_v1 + a2 * d_v2;
    let d_v3 = a2 * d_v1 + a3 * d_v2;
    let d_c2 = -*d_ic2 + d_v2 - d_v3;
    let d_u = d_hp + d_v3;
    let d_a2_tot = v3 * d_v1 + c1 * d_v2 + g * (v3 * d_v2);
    let d_a1_tot = c1 * d_v1 + g * d_a2_tot;
    let d_denom = -a1 * a1 * d_a1_tot;
    let d_g = a2 * (v3 * d_v2) + a1 * d_a2_tot + (2.0 * g + k) * d_denom;
    let d_k = -v1 * d_hp + g * d_denom;
    *d_ic1 = d_c1;
    *d_ic2 = d_c2;
    (d_u, d_g, d_k)
}
