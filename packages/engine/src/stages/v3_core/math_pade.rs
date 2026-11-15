//! Division-free Padé `[5/4]` `tanh` & `sigmoid` SIMD/scalar primitives (`0 FDIV`, `0 FSQRT`, `0 libm`).
//!
//! Matches `x - x^3 / 3 + 2 x^5 / 15` to 5th order around `0` with smooth $C^1$ clamping
//! (`f(±4) = ±1`, `f'(±4) = 0`), piecewise-linear Chebyshev initial seed on `den in [1, 11.2]`,
//! and 3 FMA Newton–Raphson reciprocal refinements (`< 1 ULP` in `f32`, exact `+0.0` at rest).

use crate::math::{MathBackend, SimdVec4};
use crate::stages::v3_core::svf::recip_nr3_pos_f32;

/// Numerator $x^2$ coefficient (`61 / 600`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_N1_F64: f64 = 61.0 / 600.0;
/// Numerator $x^4$ coefficient (`11 / 19200`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_N2_F64: f64 = 11.0 / 19_200.0;
/// Denominator $x^2$ coefficient (`87 / 200`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_D1_F64: f64 = 87.0 / 200.0;
/// Denominator $x^4$ coefficient (`47 / 3840`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_D2_F64: f64 = 47.0 / 3_840.0;

const N1: f32 = 61.0 / 600.0;
const N2: f32 = 11.0 / 19_200.0;
const D1: f32 = 87.0 / 200.0;
const D2: f32 = 47.0 / 3_840.0;

// 3-piece convex-envelope Chebyshev reciprocal seed coefficients for den in [1.0, 11.20]:
// [1.0, 2.2], [2.2, 5.0], [5.0, 11.20] -> max relative error < 8.2% -> < 2e-9 after 3 NR steps.
const CHEB_A1: f32 = 1.344_537_8;
const CHEB_B1: f32 = 0.420_168_07;
const CHEB_A2: f32 = 0.601_001_7;
const CHEB_B2: f32 = 0.083_472_45;
const CHEB_A3: f32 = 0.266_425_46;
const CHEB_B3: f32 = 0.016_446_016;

/// Division-free 4-lane SIMD Padé `[5/4]` `tanh(x)` (`C^1` clamped at `±4`, exact `+0.0` at `0`).
#[inline(always)]
#[must_use]
pub fn tanh_pade5_vec4<B: MathBackend>(x: B::Vec4) -> B::Vec4 {
    let four = B::Vec4::splat(4.0);
    let xc = x.clamp(B::Vec4::splat(-4.0), four);
    let s = xc.mul(xc);
    let one = B::Vec4::splat(1.0);
    let num_poly = one.fma(s, B::Vec4::splat(N1).fma(s, B::Vec4::splat(N2)));
    let num = xc.mul(num_poly);
    let den = one.fma(s, B::Vec4::splat(D1).fma(s, B::Vec4::splat(D2)));

    let p1 = B::Vec4::splat(CHEB_A1).fms(B::Vec4::splat(CHEB_B1), den);
    let p2 = B::Vec4::splat(CHEB_A2).fms(B::Vec4::splat(CHEB_B2), den);
    let p3 = B::Vec4::splat(CHEB_A3).fms(B::Vec4::splat(CHEB_B3), den);
    let mut y = p1.max(p2).max(p3);

    let two = B::Vec4::splat(2.0);
    y = y.mul(two.fms(den, y));
    y = y.mul(two.fms(den, y));
    y = y.mul(two.fms(den, y));

    num.mul(y).clamp(B::Vec4::splat(-1.0), one)
}

/// Scalar division-free Padé `[5/4]` `tanh(x)` (`0 FDIV`, `0 FSQRT`, `0 libm`, exact `+0.0` at `0`).
#[inline(always)]
#[must_use]
pub fn tanh_pade5_f32(x: f32) -> f32 {
    let xc = x.clamp(-4.0, 4.0);
    let s = xc * xc;
    let num = xc * (1.0 + s * (N1 + s * N2));
    let den = 1.0 + s * (D1 + s * D2);
    let p1 = CHEB_A1 - CHEB_B1 * den;
    let p2 = CHEB_A2 - CHEB_B2 * den;
    let p3 = CHEB_A3 - CHEB_B3 * den;
    let mut y = p1.max(p2).max(p3);
    y *= 2.0 - den * y;
    y *= 2.0 - den * y;
    y *= 2.0 - den * y;
    (num * y).clamp(-1.0, 1.0)
}

/// Division-free 4-lane SIMD `sigmoid(x) = 0.5 + 0.5 * tanh(0.5 * x)` using [`tanh_pade5_vec4`].
#[inline(always)]
#[must_use]
pub fn sigmoid_pade5_vec4<B: MathBackend>(x: B::Vec4) -> B::Vec4 {
    let half = B::Vec4::splat(0.5);
    let s = half.fma(tanh_pade5_vec4::<B>(x.mul(half)), half);
    s.clamp(B::Vec4::zero(), B::Vec4::splat(1.0))
}

/// Scalar division-free `sigmoid(x) = 0.5 + 0.5 * tanh(0.5 * x)` using [`tanh_pade5_f32`].
#[inline(always)]
#[must_use]
pub fn sigmoid_pade5_f32(x: f32) -> f32 {
    (0.5 + 0.5 * tanh_pade5_f32(0.5 * x)).clamp(0.0, 1.0)
}

/// Positive reciprocal `1.0 / d` for `d >= 1.0` (`0 FDIV`).
#[inline(always)]
#[must_use]
pub fn recip_pos_ge1_f32(d: f32) -> f32 {
    recip_nr3_pos_f32(d.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};
    use proptest::prelude::*;

    fn tanh_pade5_exact_f64(x: f64) -> f64 {
        let xc = x.clamp(-4.0, 4.0);
        let s = xc * xc;
        let num = xc * (1.0 + s * (TANH_PADE_N1_F64 + s * TANH_PADE_N2_F64));
        let den = 1.0 + s * (TANH_PADE_D1_F64 + s * TANH_PADE_D2_F64);
        (num / den).clamp(-1.0, 1.0)
    }

    #[test]
    fn test_tanh_and_sigmoid_pade5_zero_and_parity() {
        const {
            assert!(TANH_PADE_N1_F64 > 0.0 && TANH_PADE_N2_F64 > 0.0);
            assert!(TANH_PADE_D1_F64 > 0.0 && TANH_PADE_D2_F64 > 0.0);
        }
        assert_eq!(tanh_pade5_f32(0.0).to_bits(), 0);
        let v_zero = tanh_pade5_vec4::<DefaultMathBackend>(
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
        );
        for lane in v_zero.to_array() {
            assert_eq!(lane.to_bits(), 0);
        }
        assert!((sigmoid_pade5_f32(0.0) - 0.5).abs() < 1.0e-7);
        assert_eq!(tanh_pade5_f32(4.0), 1.0);
        assert_eq!(tanh_pade5_f32(-4.0), -1.0);
        assert_eq!(sigmoid_pade5_f32(-8.0), 0.0);
        assert_eq!(sigmoid_pade5_f32(8.0), 1.0);
        assert!((recip_pos_ge1_f32(4.0) - 0.25).abs() < 1.0e-6);
        assert!((recip_pos_ge1_f32(0.25) - 1.0).abs() < 1.0e-6);

        for &x in &[-10.0_f32, -1.5, 0.0, 1.5, 10.0] {
            assert!(tanh_pade5_f32(x).abs() <= 1.0);
            let sig = sigmoid_pade5_f32(x);
            assert!((0.0..=1.0).contains(&sig));
        }

        let vx_s = <ScalarBackend as MathBackend>::Vec4::load(&[-2.0, -0.5, 0.5, 2.0]);
        let vx_d = <DefaultMathBackend as MathBackend>::Vec4::load(&[-2.0, -0.5, 0.5, 2.0]);
        let sig_s = sigmoid_pade5_vec4::<ScalarBackend>(vx_s).to_array();
        let sig_d = sigmoid_pade5_vec4::<DefaultMathBackend>(vx_d).to_array();
        for i in 0..4 {
            assert!((0.0..=1.0).contains(&sig_s[i]));
            assert!((sig_s[i] - sig_d[i]).abs() < 1.0e-6);
        }
    }

    proptest! {
        #[test]
        fn prop_tanh_and_sigmoid_pade5_accurate_and_monotone(
            a in -10.0_f32..10.0_f32,
            b in -10.0_f32..10.0_f32,
        ) {
            let v_in = [a, b, -a, 0.0];
            let simd_t = tanh_pade5_vec4::<DefaultMathBackend>(
                <DefaultMathBackend as MathBackend>::Vec4::from_array(v_in),
            ).to_array();
            let scal_t = tanh_pade5_vec4::<ScalarBackend>(
                <ScalarBackend as MathBackend>::Vec4::from_array(v_in),
            ).to_array();
            for i in 0..4 {
                let exact = tanh_pade5_exact_f64(f64::from(v_in[i]));
                let sf = tanh_pade5_f32(v_in[i]);
                prop_assert!((f64::from(simd_t[i]) - exact).abs() < 1.5e-6);
                prop_assert!((f64::from(scal_t[i]) - exact).abs() < 1.5e-6);
                prop_assert!((f64::from(sf) - exact).abs() < 1.5e-6);
                prop_assert!(simd_t[i].abs() <= 1.0);
                // Global proximity to true tanh(x)
                prop_assert!((exact - f64::from(v_in[i]).tanh()).abs() < 1.5e-3);
            }
            // Anti-symmetry & monotonicity
            prop_assert!((simd_t[0] + simd_t[2]).abs() < 1.0e-6);
            if a <= b {
                prop_assert!(simd_t[0] <= simd_t[1] + 1.0e-6);
            } else {
                prop_assert!(simd_t[1] <= simd_t[0] + 1.0e-6);
            }
            let sig_a = sigmoid_pade5_f32(a);
            let sig_b = sigmoid_pade5_f32(b);
            prop_assert!((0.0..=1.0).contains(&sig_a));
            if a <= b {
                prop_assert!(sig_a <= sig_b + 1.0e-6);
            }
        }
    }
}
