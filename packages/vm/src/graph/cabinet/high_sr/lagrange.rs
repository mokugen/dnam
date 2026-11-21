//! Off-`48 kHz` Cabinet Lagrange difference kernels, `44.1 kHz` first-tap prototype,
//! `u64` shift-division resampler coefficients, and Catmull-Rom FIR resampling
//! (`ref[dsp.zoh.cabinet-fir-onset]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use crate::math::{
    IEEE754_F64_EXP_ALL_ONES, IEEE754_F64_EXP_BIAS, IEEE754_F64_IMPLICIT_BIT,
    IEEE754_F64_MANTISSA_BITS, IEEE754_F64_MANTISSA_MASK, const_abs_f64, const_f64_pow2,
    const_inv_factorial_f64, floor_f64, rcp_nr6_f64,
};

use super::{MAX_CABINET_BASE_TAPS, STAGE3_EXT_FIR_CAP};

const INV_120_F64: f64 = const_inv_factorial_f64(5);
const TWO_POW_NEG56_F64: f64 = const_f64_pow2(-56);
const DC_NORM_MIN_SUM: f64 = 1.0e-9;

/// Canonical `64`-tap gain-bounded fractional first-tap prototype (`44.1 kHz` reference) for
/// `Case (b)` live `1×` difference synthesis (`ref[dsp.zoh.cabinet-fir-onset]`).
pub const GAIN_BOUNDED_FIRST_TAP_64_BITS: [u32; 64] = [
    0x3d6c_9268,
    0x3f83_725f,
    0xbdf0_a4f1,
    0x3dad_b502,
    0xbd9d_a17d,
    0x3d95_380c,
    0xbd94_9c5f,
    0x3d91_792e,
    0xbd91_ce40,
    0x3d8d_d068,
    0xbd8b_eae0,
    0x3d85_3dac,
    0xbd80_189f,
    0x3d6c_7fa0,
    0xbd5b_a083,
    0x3d42_31cb,
    0xbd2b_f99b,
    0x3d0e_88f9,
    0xbcea_4a93,
    0x3cac_58c1,
    0xbc71_4876,
    0x3bf4_c5f0,
    0xbae2_2743,
    0xbb96_2d07,
    0x3c16_34d0,
    0xbc65_add4,
    0x3c8b_7e54,
    0xbca4_d3dc,
    0x3caf_0e76,
    0xbcb9_82b1,
    0x3cb5_d3c2,
    0xbcb2_e27a,
    0x3ca3_932b,
    0xbc96_3f77,
    0x3c7d_c51e,
    0xbc56_5369,
    0x3c1f_d21c,
    0xbbe7_f768,
    0x3b77_5394,
    0xbaa9_8a21,
    0xbacf_4513,
    0x3b66_e3c0,
    0xbbba_af04,
    0x3be2_53a0,
    0xbc07_70ed,
    0x3c0e_a839,
    0xbc18_1cd3,
    0x3c14_420d,
    0xbc13_3bf5,
    0x3c07_4843,
    0xbbfe_526f,
    0x3bdd_6c26,
    0xbbc6_2d29,
    0x3ba3_1dc4,
    0xbb8b_1039,
    0x3b56_8017,
    0xbb2d_823b,
    0x3af6_3e3f,
    0xbaba_d495,
    0x3a6a_e4c8,
    0xba23_1f37,
    0x39a6_0fed,
    0xb943_301a,
    0x386d_c990,
];

const _: () = assert!(GAIN_BOUNDED_FIRST_TAP_64_BITS[0] == 0x3d6c_9268);
const _: () = assert!(GAIN_BOUNDED_FIRST_TAP_64_BITS[63] == 0x386d_c990);

/// Exact `u64` bit-shift Euclidean division (`0 UDIV`, `0 u128 /`).
#[inline]
pub(super) fn div_rem_u64_shift(num: u64, den: u64) -> (u64, u64) {
    if den == 0 || num < den {
        return (0, num);
    }
    let shift = den.leading_zeros().saturating_sub(num.leading_zeros());
    let mut q = 0_u64;
    let mut rem = num;
    let mut i = shift + 1;
    while i > 0 {
        i -= 1;
        let d_shifted = den << i;
        if rem >= d_shifted {
            rem -= d_shifted;
            q |= 1_u64 << i;
        }
    }
    (q, rem)
}

/// Evaluates `rem / den` in `[0.0, 1.0)` to `56`-bit precision via pure `u64` bit-shift
/// division with ties-to-even rounding (`0 UDIV`, `0 FDIV`).
#[inline]
pub(super) fn frac_f64_u64_shift(rem: u64, den: u64) -> f64 {
    if rem == 0 || den == 0 {
        return 0.0;
    }
    let mut r = rem;
    let mut d = den;
    while d >= (1_u64 << 62) {
        r >>= 1;
        d >>= 1;
    }
    let mut q_bits = 0_u64;
    for _ in 0..56 {
        r <<= 1;
        q_bits <<= 1;
        if r >= d {
            r -= d;
            q_bits |= 1;
        }
    }
    let half_d = d >> 1;
    let round_up = r > half_d || (r == half_d && (q_bits & 1) != 0);
    let q_rounded = q_bits + u64::from(round_up);
    (q_rounded as f64) * TWO_POW_NEG56_F64
}

/// Precomputed `4×` offline-bounce `6`-point Lagrange difference filter for base FIR tap `k >= 1`
/// (`ref[dsp.zoh.cabinet-fir-onset]` `Case (c)`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BounceLagrangeTap {
    /// Integer sample delay base `m = floor(k * fs_vm / 48_000)` (`>= 3`).
    pub m: u16,
    /// Anchor offset `a in {2, 3}` (`2` when `mu <= 0.5`, `3` when `mu > 0.5`).
    pub anchor: u8,
    /// Explicit struct padding byte.
    pub _pad: u8,
    /// Five difference coefficients `p[0..5]` rounded to `f32`.
    pub p: [f32; 5],
}

/// Prepares the exact `6`-point Lagrange difference kernel for tap `tap_k in 1..128` at effective
/// bounce rate `fs_vm` using purely `u64` bit-shift long division (`0 UDIV`, `0 u128 /`, `0 FDIV`,
/// `ref[dsp.zoh.cabinet-fir-onset]`).
pub fn prepare_bounce_6point_u64(tap_k: usize, fs_vm: f64) -> BounceLagrangeTap {
    if tap_k == 0 || tap_k >= MAX_CABINET_BASE_TAPS || !fs_vm.is_finite() || fs_vm <= 0.0 {
        return BounceLagrangeTap::default();
    }
    const F64_MANT_BITS_I32: i32 = IEEE754_F64_MANTISSA_BITS as i32;
    let bits = fs_vm.to_bits();
    let exp = (((bits >> IEEE754_F64_MANTISSA_BITS) & (IEEE754_F64_EXP_ALL_ONES as u64)) as i32)
        - IEEE754_F64_EXP_BIAS;
    if !(5..=F64_MANT_BITS_I32).contains(&exp) {
        return BounceLagrangeTap::default();
    }
    let significand = IEEE754_F64_IMPLICIT_BIT | (bits & IEEE754_F64_MANTISSA_MASK);
    let shift_d = (F64_MANT_BITS_I32 - exp) as u32;
    let den = 48_000_u64 << shift_d;
    let num = (tap_k as u64) * significand;

    let (m_u64, rem) = div_rem_u64_shift(num, den);
    let anchor = if rem > (den >> 1) { 3_u8 } else { 2_u8 };
    let mu = frac_f64_u64_shift(rem, den);

    const C_SCALES: [f64; 5] = [-1.0, 5.0, -10.0, 10.0, -5.0];
    let diffs = [mu + 2.0, mu + 1.0, mu, mu - 1.0, mu - 2.0, mu - 3.0];
    let mut q_acc = 0.0_f64;
    let mut p = [0.0_f32; 5];
    for j in 0..5 {
        let mut prod = 1.0_f64;
        for (l, &d) in diffs.iter().enumerate() {
            if l != j {
                prod *= d;
            }
        }
        q_acc += C_SCALES[j] * prod;
        let q_j = if j >= usize::from(anchor) {
            q_acc - 120.0
        } else {
            q_acc
        };
        p[j] = (q_j * INV_120_F64) as f32;
    }

    BounceLagrangeTap {
        m: m_u64.min(STAGE3_EXT_FIR_CAP as u64 - 4) as u16,
        anchor,
        _pad: 0,
        p,
    }
}

#[inline]
pub(super) fn resample_catmull_rom_dc_norm(
    b_in: &[f32],
    rho_vm: f64,
    out_len: usize,
    dense_out: &mut [f32],
) {
    let in_len = b_in.len();
    if in_len == 0 || out_len == 0 {
        return;
    }
    let mut s_in = 0.0_f64;
    for &t in b_in {
        s_in += f64::from(t);
    }
    let max_idx = (in_len - 1) as isize;
    let mut s_out = 0.0_f64;
    for (n, dst) in dense_out.iter_mut().take(out_len).enumerate() {
        let u = (n as f64) * rho_vm;
        let u_floor = floor_f64(u);
        let i_floor = u_floor as isize;
        let a = (u - u_floor) as f32;
        let a2 = a * a;
        let a3 = a2 * a;
        let mut acc = 0.0_f64;
        for (k, wk) in [
            (-1_isize, -0.5 * a3 + a2 - 0.5 * a),
            (0, 1.5 * a3 - 2.5 * a2 + 1.0),
            (1, -1.5 * a3 + 2.0 * a2 + 0.5 * a),
            (2, 0.5 * a3 - 0.5 * a2),
        ] {
            acc += f64::from(wk) * f64::from(b_in[(i_floor + k).clamp(0, max_idx) as usize]);
        }
        let val = rho_vm * acc;
        s_out += val;
        *dst = val as f32;
    }

    if const_abs_f64(s_out) > DC_NORM_MIN_SUM && const_abs_f64(s_in) > DC_NORM_MIN_SUM {
        let sign_s_out = if s_out >= 0.0 { 1.0 } else { -1.0 };
        let m_dc = s_in * sign_s_out * rcp_nr6_f64(const_abs_f64(s_out));
        for dst in dense_out.iter_mut().take(out_len) {
            *dst = (m_dc * f64::from(*dst)) as f32;
        }
    }
}

#[inline(always)]
pub(super) fn eval_tap1_diff(hist: &[f32], base: usize, tap1_diff: &[f32; 63]) -> f32 {
    let mut y = hist[base - 1];
    for j in 0..63 {
        y += tap1_diff[j] * (hist[base - j] - hist[base - j - 1]);
    }
    y
}

#[inline(always)]
pub(super) fn eval_bounce_diff5(hist: &[f32], base: usize, tap: &BounceLagrangeTap) -> f32 {
    let m = usize::from(tap.m).clamp(2, STAGE3_EXT_FIR_CAP - 4);
    let anchor = usize::from(tap.anchor);
    let mut y = hist[base - m - anchor + 2];
    for j in 0..5 {
        y += tap.p[j] * (hist[base - m + 2 - j] - hist[base - m + 1 - j]);
    }
    y
}
