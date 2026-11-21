//! Canonical 15-rate `HILBERT-H1-ANCHORS` IEEE-754 bit tables (`44.1 ..= 384` kHz) and
//! exact `128-bit` rational ties-to-even coefficient interpolation (`ref[dsp.op-m.hilbert.lattice]`,
//! `ref[dsp.op-m.hilbert.ties-to-even]`).
//!
//! Runtime lookup constants are pure `u32` / `f32` (`#![no_std]`); cold/host 128-bit rational
//! interpolation is guarded behind `#[cfg(not(target_arch = "spirv"))]`.

use crate::simd::{IEEE754_F32_EXP_ALL_ONES, IEEE754_F32_EXP_BIAS, IEEE754_F32_MANTISSA_BITS};
#[cfg(not(target_arch = "spirv"))]
use crate::simd::{
    IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_MANTISSA_MASK, IEEE754_F32_MIN_NORMAL_EXP,
    IEEE754_F32_ONE_BITS, IEEE754_F64_EXP_ALL_ONES, IEEE754_F64_EXP_BIAS,
    IEEE754_F64_MANTISSA_BITS,
};

const HILBERT_ANCHOR_MIN_NEG_EXP: u32 = 1;
const HILBERT_ANCHOR_MAX_NEG_EXP: u32 = 4;
/// Biased `f32` exponent lower bound across canonical `HILBERT-H1-ANCHORS` (`2^-4`, `127 - 4 == 123`).
const HILBERT_ANCHOR_MIN_BIASED_EXP: u32 = IEEE754_F32_EXP_BIAS - HILBERT_ANCHOR_MAX_NEG_EXP;
/// Biased `f32` exponent upper bound across canonical `HILBERT-H1-ANCHORS` (`< 1.0`, `127 - 1 == 126`).
const HILBERT_ANCHOR_MAX_BIASED_EXP: u32 = IEEE754_F32_EXP_BIAS - HILBERT_ANCHOR_MIN_NEG_EXP;
/// Fixed-point fractional precision (`23 + 4 == 27` bits) for exact 128-bit anchor arithmetic.
#[cfg(not(target_arch = "spirv"))]
const HILBERT_ANCHOR_Q_BITS: u32 = IEEE754_F32_MANTISSA_BITS + HILBERT_ANCHOR_MAX_NEG_EXP;

/// Number of second-order all-pass stages per quadrature branch in `H1` lattices.
pub const HILBERT_H1_STAGES: usize = 6;

/// Canonical 15-rate anchor sample-rate ladder `[44.1 ..= 384]` kHz for `HILBERT-H1-ANCHORS`
/// (`ref[dsp.op-m.hilbert.ties-to-even]`).
pub const HILBERT_H1_ANCHOR_RATES_HZ: [u32; 15] = [
    44_100, 48_000, 64_000, 80_000, 96_000, 176_400, 192_000, 208_000, 224_000, 240_000, 256_000,
    288_000, 320_000, 352_000, 384_000,
];

/// Canonical 15-rate `HILBERT-H1-ANCHORS` IEEE-754 `binary32` `(alpha_i, alpha_q)` bit tables
/// (`ref[dsp.op-m.hilbert.ties-to-even]`). Row index `1` (`48_000 Hz`) is shared with
/// [`HILBERT_ALPHA_I_BITS`] and [`HILBERT_ALPHA_Q_BITS`].
pub const HILBERT_H1_ANCHOR_BITS: [([u32; HILBERT_H1_STAGES], [u32; HILBERT_H1_STAGES]); 15] = [
    (
        [
            0x3D8E_C459,
            0x3EE9_35E1,
            0x3F47_659D,
            0x3F6C_CEAA,
            0x3F7A_0188,
            0x3F7E_7297,
        ],
        [
            0x3E7B_1AD6,
            0x3F24_3369,
            0x3F5E_97F2,
            0x3F75_2D7D,
            0x3F7C_CA31,
            0x3F7F_8A32,
        ],
    ),
    (
        [
            0x3D91_BEE6,
            0x3EEC_B8E1,
            0x3F48_DE0D,
            0x3F6D_91C9,
            0x3F7A_5939,
            0x3F7E_8E9D,
        ],
        [
            0x3E80_025F,
            0x3F25_F6B0,
            0x3F5F_ADA1,
            0x3F75_B332,
            0x3F7C_FFA1,
            0x3F7F_92D2,
        ],
    ),
    (
        [
            0x3DA2_AF27,
            0x3EFE_4625,
            0x3F4F_BDA5,
            0x3F70_B815,
            0x3F7B_8289,
            0x3F7E_E2C7,
        ],
        [
            0x3E8C_7CF3,
            0x3F2E_9DD3,
            0x3F64_7F75,
            0x3F77_A752,
            0x3F7D_A785,
            0x3F7F_AC3E,
        ],
    ),
    (
        [
            0x3DA8_8500,
            0x3F02_12AF,
            0x3F52_3110,
            0x3F72_069B,
            0x3F7C_1032,
            0x3F7F_0F58,
        ],
        [
            0x3E90_CC38,
            0x3F31_8514,
            0x3F66_5AC6,
            0x3F78_8657,
            0x3F7D_FC4C,
            0x3F7F_BA14,
        ],
    ),
    (
        [
            0x3DB0_73C5,
            0x3F05_A9C4,
            0x3F54_E508,
            0x3F73_43B2,
            0x3F7C_852B,
            0x3F7F_30C7,
        ],
        [
            0x3E96_3AF7,
            0x3F34_F3ED,
            0x3F68_3FD9,
            0x3F79_4AFF,
            0x3F7E_3E99,
            0x3F7F_C438,
        ],
    ),
    (
        [
            0x3E0D_917D,
            0x3F21_6E28,
            0x3F62_4034,
            0x3F77_D413,
            0x3F7D_E528,
            0x3F7F_8887,
        ],
        [
            0x3ECD_F9DC,
            0x3F49_A494,
            0x3F70_3BD8,
            0x3F7B_D39E,
            0x3F7E_F730,
            0x3F7F_DDFB,
        ],
    ),
    (
        [
            0x3E1A_79AD,
            0x3F26_91E2,
            0x3F64_674D,
            0x3F78_7EBC,
            0x3F7E_12E1,
            0x3F7F_9298,
        ],
        [
            0x3ED9_9328,
            0x3F4D_2EB7,
            0x3F71_7613,
            0x3F7C_2D21,
            0x3F7F_0DA9,
            0x3F7F_E0D4,
        ],
    ),
    (
        [
            0x3E26_46E1,
            0x3F2A_6E87,
            0x3F65_C7A3,
            0x3F78_E266,
            0x3F7E_2EFD,
            0x3F7F_99AA,
        ],
        [
            0x3EE3_3E4A,
            0x3F4F_9D5A,
            0x3F72_3264,
            0x3F7C_621C,
            0x3F7F_1C5F,
            0x3F7F_E2ED,
        ],
    ),
    (
        [
            0x3E34_BA6C,
            0x3F2F_8A17,
            0x3F67_DE51,
            0x3F79_8480,
            0x3F7E_584A,
            0x3F7F_A231,
        ],
        [
            0x3EEF_3412,
            0x3F53_121B,
            0x3F73_6170,
            0x3F7C_B521,
            0x3F7F_3010,
            0x3F7F_E547,
        ],
    ),
    (
        [
            0x3E41_51D9,
            0x3F33_437E,
            0x3F69_3973,
            0x3F79_E8DF,
            0x3F7E_72F8,
            0x3F7F_A845,
        ],
        [
            0x3EF8_BC96,
            0x3F55_6DD1,
            0x3F74_1F05,
            0x3F7C_E93F,
            0x3F7F_3D5F,
            0x3F7F_E706,
        ],
    ),
    (
        [
            0x3E4C_BD88,
            0x3F36_3B81,
            0x3F6A_3C44,
            0x3F7A_3322,
            0x3F7E_877C,
            0x3F7F_AD32,
        ],
        [
            0x3F00_6617,
            0x3F57_3C54,
            0x3F74_AA8C,
            0x3F7D_1079,
            0x3F7F_47DE,
            0x3F7F_E876,
        ],
    ),
    (
        [
            0x3E64_086E,
            0x3F3B_DB10,
            0x3F6C_20CB,
            0x3F7A_BA23,
            0x3F7E_ABAF,
            0x3F7F_B5AD,
        ],
        [
            0x3F08_3609,
            0x3F5A_A15E,
            0x3F75_ACD3,
            0x3F7D_5697,
            0x3F7F_5A2A,
            0x3F7F_EAEB,
        ],
    ),
    (
        [
            0x3E7D_6662,
            0x3F41_DCD7,
            0x3F6E_3EDD,
            0x3F7B_5343,
            0x3F7E_D216,
            0x3F7F_BDBA,
        ],
        [
            0x3F10_68CD,
            0x3F5E_589B,
            0x3F76_D2C7,
            0x3F7D_A41A,
            0x3F7F_6C8B,
            0x3F7F_ED2A,
        ],
    ),
    (
        [
            0x3E88_E622,
            0x3F45_A7EB,
            0x3F6F_6903,
            0x3F7B_A4C3,
            0x3F7E_E847,
            0x3F7F_C310,
        ],
        [
            0x3F16_3553,
            0x3F60_82FF,
            0x3F77_6EE6,
            0x3F7D_CEB5,
            0x3F7F_77E5,
            0x3F7F_EEB9,
        ],
    ),
    (
        [
            0x3E94_2913,
            0x3F49_F3A3,
            0x3F70_D45F,
            0x3F7C_094D,
            0x3F7F_0173,
            0x3F7F_C869,
        ],
        [
            0x3F1C_84D4,
            0x3F63_1031,
            0x3F78_3122,
            0x3F7E_017B,
            0x3F7F_8401,
            0x3F7F_F039,
        ],
    ),
];

/// `48 kHz` anchor all-pass pole bit patterns for the `I` branch (`ref[dsp.op-m.hilbert.lattice]`).
pub const HILBERT_ALPHA_I_BITS: [u32; HILBERT_H1_STAGES] = HILBERT_H1_ANCHOR_BITS[1].0;

/// `48 kHz` anchor all-pass pole bit patterns for the `Q` branch (`ref[dsp.op-m.hilbert.lattice]`).
pub const HILBERT_ALPHA_Q_BITS: [u32; HILBERT_H1_STAGES] = HILBERT_H1_ANCHOR_BITS[1].1;

/// Compile-time `48 kHz` `I`-branch Hilbert all-pass coefficients derived from
/// [`HILBERT_ALPHA_I_BITS`].
pub const HILBERT_ALPHA_I: [f32; HILBERT_H1_STAGES] = bits_to_f32_array6(HILBERT_ALPHA_I_BITS);

/// Compile-time `48 kHz` `Q`-branch Hilbert all-pass coefficients derived from
/// [`HILBERT_ALPHA_Q_BITS`].
pub const HILBERT_ALPHA_Q: [f32; HILBERT_H1_STAGES] = bits_to_f32_array6(HILBERT_ALPHA_Q_BITS);

#[inline(always)]
const fn bits_to_f32_array6(bits: [u32; HILBERT_H1_STAGES]) -> [f32; HILBERT_H1_STAGES] {
    [
        f32::from_bits(bits[0]),
        f32::from_bits(bits[1]),
        f32::from_bits(bits[2]),
        f32::from_bits(bits[3]),
        f32::from_bits(bits[4]),
        f32::from_bits(bits[5]),
    ]
}

const _: () = {
    let mut r = 1;
    while r < HILBERT_H1_ANCHOR_RATES_HZ.len() {
        assert!(HILBERT_H1_ANCHOR_RATES_HZ[r - 1] < HILBERT_H1_ANCHOR_RATES_HZ[r]);
        r += 1;
    }
    let mut row = 0;
    while row < HILBERT_H1_ANCHOR_BITS.len() {
        let mut stage = 0;
        while stage < HILBERT_H1_STAGES {
            let exp_i = (HILBERT_H1_ANCHOR_BITS[row].0[stage] >> IEEE754_F32_MANTISSA_BITS)
                & IEEE754_F32_EXP_ALL_ONES;
            let exp_q = (HILBERT_H1_ANCHOR_BITS[row].1[stage] >> IEEE754_F32_MANTISSA_BITS)
                & IEEE754_F32_EXP_ALL_ONES;
            assert!(
                exp_i >= HILBERT_ANCHOR_MIN_BIASED_EXP && exp_i <= HILBERT_ANCHOR_MAX_BIASED_EXP
            );
            assert!(
                exp_q >= HILBERT_ANCHOR_MIN_BIASED_EXP && exp_q <= HILBERT_ANCHOR_MAX_BIASED_EXP
            );
            stage += 1;
        }
        row += 1;
    }
    assert!(HILBERT_ALPHA_I[0] > 0.0 && HILBERT_ALPHA_I[5] < 1.0);
    assert!(HILBERT_ALPHA_Q[0] > 0.0 && HILBERT_ALPHA_Q[5] < 1.0);
};

/// Exact `u128` bit-shift Euclidean division (`0 UDIV`, `0 __udivti3` / `__umodti3`).
#[cfg(not(target_arch = "spirv"))]
#[inline]
fn div_rem_u128_shift(num: u128, den: u128) -> (u128, u128) {
    if den == 0 || num < den {
        return (0, num);
    }
    let shift = den.leading_zeros().saturating_sub(num.leading_zeros());
    let mut q = 0_u128;
    let mut rem = num;
    let mut i = shift + 1;
    while i > 0 {
        i -= 1;
        let d_shifted = den << i;
        if rem >= d_shifted {
            rem -= d_shifted;
            q |= 1_u128 << i;
        }
    }
    (q, rem)
}

/// Host/cold-setup 128-bit exact rational `binary32` ties-to-even rounding of `n / d` (`0 < n < d`),
/// returning the IEEE-754 `u32` bit pattern (`ref[dsp.op-m.hilbert.ties-to-even]`).
#[cfg(not(target_arch = "spirv"))]
fn round_ratio_u128_to_f32_bits(n: u128, d: u128) -> u32 {
    const MAX_NORMAL_NEG_EXP: u32 = (-IEEE754_F32_MIN_NORMAL_EXP) as u32;
    if n == 0 {
        return 0;
    }
    if n >= d {
        return IEEE754_F32_ONE_BITS;
    }
    let n_bits = u128::BITS - n.leading_zeros();
    let max_e = (u128::BITS - IEEE754_F32_MANTISSA_BITS)
        .saturating_sub(n_bits)
        .min(MAX_NORMAL_NEG_EXP);
    let mut e = 1_u32.min(max_e);
    while (n << e) < d && e < max_e {
        e += 1;
    }
    let exp_f32 = IEEE754_F32_EXP_BIAS - e;
    let num = n << (e + IEEE754_F32_MANTISSA_BITS);
    let (q, rem) = div_rem_u128_shift(num, d);
    let two_rem = rem << 1;
    let round_up = two_rem > d || (two_rem == d && (q & 1) == 1);
    let m = (q as u32) + u32::from(round_up);
    ((exp_f32 - 1) << IEEE754_F32_MANTISSA_BITS) + m
}

#[cfg(not(target_arch = "spirv"))]
#[inline(always)]
fn anchor_bits_to_num27(bits: u32) -> u128 {
    let exp = (bits >> IEEE754_F32_MANTISSA_BITS) & IEEE754_F32_EXP_ALL_ONES;
    let sig = IEEE754_F32_IMPLICIT_BIT | (bits & IEEE754_F32_MANTISSA_MASK);
    // `sig * 2^(exp - 127 - 23) == (sig << (exp - (IEEE754_F32_EXP_BIAS - 4))) * 2^-27`.
    u128::from(sig) << (exp - HILBERT_ANCHOR_MIN_BIASED_EXP)
}

/// Computes exact `128-bit` rational ties-to-even interpolated `(alpha_i, alpha_q)` vectors over
/// the 15-rate `HILBERT-H1-ANCHORS` table for any finite positive `binary64` effective rate `rate_hz`
/// clamped into `[44_100 ..= 384_000]` Hz (`ref[dsp.op-m.hilbert.ties-to-even]`).
///
/// Non-finite or `<= 0.0` rates fall back cleanly to the `48_000 Hz` anchor row.
#[cfg(not(target_arch = "spirv"))]
#[must_use]
pub fn interpolate_hilbert_h1_alphas_f64(
    rate_hz: f64,
) -> ([f32; HILBERT_H1_STAGES], [f32; HILBERT_H1_STAGES]) {
    if !rate_hz.is_finite() || rate_hz <= 0.0 {
        return (HILBERT_ALPHA_I, HILBERT_ALPHA_Q);
    }
    if rate_hz <= f64::from(HILBERT_H1_ANCHOR_RATES_HZ[0]) {
        return (
            bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[0].0),
            bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[0].1),
        );
    }
    let last_idx = HILBERT_H1_ANCHOR_RATES_HZ.len() - 1;
    if rate_hz >= f64::from(HILBERT_H1_ANCHOR_RATES_HZ[last_idx]) {
        return (
            bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[last_idx].0),
            bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[last_idx].1),
        );
    }

    let bits = rate_hz.to_bits();
    let e_bin = (((bits >> IEEE754_F64_MANTISSA_BITS) & u64::from(IEEE754_F64_EXP_ALL_ONES))
        as i32)
        - IEEE754_F64_EXP_BIAS;
    let mant53 =
        (1_u64 << IEEE754_F64_MANTISSA_BITS) | (bits & ((1_u64 << IEEE754_F64_MANTISSA_BITS) - 1));
    let exp_neg = ((IEEE754_F64_MANTISSA_BITS as i32) - e_bin) as u32;
    let tz = mant53.trailing_zeros().min(exp_neg);
    let r = u128::from(mant53 >> tz);
    let k = exp_neg - tz;

    let mut row = 0;
    while row + 1 < HILBERT_H1_ANCHOR_RATES_HZ.len() {
        let l = u128::from(HILBERT_H1_ANCHOR_RATES_HZ[row]);
        let h = u128::from(HILBERT_H1_ANCHOR_RATES_HZ[row + 1]);
        let l2k = l << k;
        let h2k = h << k;
        if r <= h2k {
            if r == l2k {
                return (
                    bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[row].0),
                    bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[row].1),
                );
            }
            if r == h2k {
                return (
                    bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[row + 1].0),
                    bits_to_f32_array6(HILBERT_H1_ANCHOR_BITS[row + 1].1),
                );
            }
            let d = (h - l) << (k + HILBERT_ANCHOR_Q_BITS);
            let w_lo = h2k - r;
            let w_hi = r - l2k;
            let mut out_i = [0.0_f32; HILBERT_H1_STAGES];
            let mut out_q = [0.0_f32; HILBERT_H1_STAGES];
            for stage in 0..HILBERT_H1_STAGES {
                let ai = anchor_bits_to_num27(HILBERT_H1_ANCHOR_BITS[row].0[stage]);
                let bi = anchor_bits_to_num27(HILBERT_H1_ANCHOR_BITS[row + 1].0[stage]);
                out_i[stage] =
                    f32::from_bits(round_ratio_u128_to_f32_bits(w_lo * ai + w_hi * bi, d));

                let aq = anchor_bits_to_num27(HILBERT_H1_ANCHOR_BITS[row].1[stage]);
                let bq = anchor_bits_to_num27(HILBERT_H1_ANCHOR_BITS[row + 1].1[stage]);
                out_q[stage] =
                    f32::from_bits(round_ratio_u128_to_f32_bits(w_lo * aq + w_hi * bq, d));
            }
            return (out_i, out_q);
        }
        row += 1;
    }
    (HILBERT_ALPHA_I, HILBERT_ALPHA_Q)
}

/// Convenience `f32` wrapper over [`interpolate_hilbert_h1_alphas_f64`].
#[cfg(not(target_arch = "spirv"))]
#[inline]
#[must_use]
pub fn interpolate_hilbert_h1_alphas(
    rate_hz: f32,
) -> ([f32; HILBERT_H1_STAGES], [f32; HILBERT_H1_STAGES]) {
    interpolate_hilbert_h1_alphas_f64(f64::from(rate_hz))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// verify[dsp.op-m.hilbert.h1-anchors]
    /// verify[dsp.op-m.hilbert.ties-to-even]
    #[test]
    fn test_hilbert_h1_anchors_ties_to_even_and_bridged_128k() {
        for (idx, &rate) in HILBERT_H1_ANCHOR_RATES_HZ.iter().enumerate() {
            let (ai, aq) = interpolate_hilbert_h1_alphas_f64(f64::from(rate));
            assert_eq!(ai.map(f32::to_bits), HILBERT_H1_ANCHOR_BITS[idx].0);
            assert_eq!(aq.map(f32::to_bits), HILBERT_H1_ANCHOR_BITS[idx].1);
        }
        assert_eq!(
            interpolate_hilbert_h1_alphas_f64(22_050.0)
                .0
                .map(f32::to_bits),
            HILBERT_H1_ANCHOR_BITS[0].0
        );
        assert_eq!(
            interpolate_hilbert_h1_alphas_f64(768_000.0)
                .1
                .map(f32::to_bits),
            HILBERT_H1_ANCHOR_BITS[14].1
        );
        assert_eq!(
            interpolate_hilbert_h1_alphas(f32::NAN).0.map(f32::to_bits),
            HILBERT_ALPHA_I_BITS
        );

        let (ai_128, aq_128) = interpolate_hilbert_h1_alphas_f64(128_000.0);
        for s in 0..HILBERT_H1_STAGES {
            let lo_i = f32::from_bits(HILBERT_H1_ANCHOR_BITS[4].0[s]);
            let hi_i = f32::from_bits(HILBERT_H1_ANCHOR_BITS[5].0[s]);
            let lo_q = f32::from_bits(HILBERT_H1_ANCHOR_BITS[4].1[s]);
            let hi_q = f32::from_bits(HILBERT_H1_ANCHOR_BITS[5].1[s]);
            assert!(ai_128[s] > lo_i && ai_128[s] < hi_i);
            assert!(aq_128[s] > lo_q && aq_128[s] < hi_q);
        }

        // Direct exact-ties-to-even edge cases (n == 0, n >= d, half-ULP tie).
        assert_eq!(round_ratio_u128_to_f32_bits(0, 10), 0);
        assert_eq!(round_ratio_u128_to_f32_bits(10, 10), 0x3F80_0000);
        // Halfway between 0.5 and 0.5 + 2^-24 (even lsb 0 stays even; odd lsb 1 rounds up).
        let base_even = 1_u128 << 24;
        assert_eq!(
            round_ratio_u128_to_f32_bits(base_even + 1, 1_u128 << 25),
            0x3F00_0000
        );
        assert_eq!(
            round_ratio_u128_to_f32_bits(base_even + 3, 1_u128 << 25),
            0x3F00_0002
        );
        assert_eq!(div_rem_u128_shift(7, 0), (0, 7));
        assert_eq!(div_rem_u128_shift(7, 9), (0, 7));
        let large_n = (u128::MAX >> 5) - 17;
        let large_d = 97_654_321_012_345_u128;
        assert_eq!(
            div_rem_u128_shift(large_n, large_d),
            (large_n / large_d, large_n % large_d)
        );
    }

    proptest! {
        #[test]
        fn prop_hilbert_interpolation_is_monotone_across_rates(
            rate_a in 44_100.0_f64..200_000.0_f64,
            delta in 1.0_f64..180_000.0_f64,
        ) {
            let rate_b = (rate_a + delta).min(384_000.0);
            let (ia, qa) = interpolate_hilbert_h1_alphas_f64(rate_a);
            let (ib, qb) = interpolate_hilbert_h1_alphas_f64(rate_b);
            for s in 0..HILBERT_H1_STAGES {
                prop_assert!(ia[s] <= ib[s] && qa[s] <= qb[s]);
                prop_assert!(ia[s] > 0.0 && ib[s] < 1.0 && qa[s] > 0.0 && qb[s] < 1.0);
            }
        }
    }
}
