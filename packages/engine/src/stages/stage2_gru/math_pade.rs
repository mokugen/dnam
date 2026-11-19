//! Division-free Padé `[5/4]` `tanh` & `sigmoid` SIMD/scalar primitives (`0 FDIV`, `0 FSQRT`, `0 libm`).
//!
//! Matches `x - x^3 / 3 + 2 x^5 / 15` to 5th order around `0` with smooth $C^1$ clamping
//! (`f(±4) ≈ ±1`, `f'(±4) = 0`), in-register IEEE-754 integer reciprocal seed (`0x7EF3_11C3 - bits`)
//! + 2 FMA Newton–Raphson refinements (`NR2`: `< 6.6e-6` error = `-103 dB`, `0-ULP` odd symmetry, exact `+0.0`).
//!
//! Implements `ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`, and
//! `ref[dsp.op-n.polyadaa.rail-crossing]`.

use crate::math::{MathBackend, SimdVec4};
use crate::stages::stage1_svf::recip_nr2_pos_f32;

const TANH_PADE_R4_F64: [f64; 4] = crate::math::const_c1_clamped_tanh_coeffs(4.0);
/// Numerator $x^2$ coefficient (`61 / 600`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_N1_F64: f64 = TANH_R4_CONST[0];
/// Numerator $x^4$ coefficient (`11 / 19200`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_N2_F64: f64 = TANH_R4_CONST[1];
/// Denominator $x^2$ coefficient (`87 / 200`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_D1_F64: f64 = TANH_R4_CONST[2];
/// Denominator $x^4$ coefficient (`47 / 3840`) for Padé `[5/4]` $\tanh(x)$.
pub const TANH_PADE_D2_F64: f64 = TANH_R4_CONST[3];

const TANH_R4_CONST: [f64; 4] = TANH_PADE_R4_F64;
const N1: f32 = TANH_PADE_N1_F64 as f32;
const N2: f32 = TANH_PADE_N2_F64 as f32;
const D1: f32 = TANH_PADE_D1_F64 as f32;
const D2: f32 = TANH_PADE_D2_F64 as f32;

#[inline(always)]
fn eval_num_den_pade5<B: MathBackend>(
    x: B::Vec4,
    c: B::Vec4,
    one: B::Vec4,
    n_four: B::Vec4,
    four: B::Vec4,
) -> (B::Vec4, B::Vec4) {
    let xc = x.clamp(n_four, four);
    let s = xc.mul(xc);
    let s2 = s.mul(s);
    let num_poly = one.fma_lane::<0>(s, c).fma_lane::<1>(s2, c);
    let den = one.fma_lane::<2>(s, c).fma_lane::<3>(s2, c);
    (xc.mul(num_poly), den)
}

/// Evaluates Padé `[5/4]` `tanh(x)` simultaneously across `N` independent 4-lane vectors to hide NR2 FMA latency.
#[inline(always)]
pub fn tanh_pade5_batch_vec4<B: MathBackend, const N: usize>(xs: [B::Vec4; N]) -> [B::Vec4; N] {
    let four = B::Vec4::splat(4.0);
    let n_four = B::Vec4::splat(-4.0);
    let one = B::Vec4::splat(1.0);
    let n_one = B::Vec4::splat(-1.0);
    let c = B::Vec4::from_array([N1, N2, D1, D2]);

    let mut num = [B::Vec4::zero(); N];
    let mut den = [B::Vec4::zero(); N];
    for i in 0..N {
        let (n_v, d_v) = eval_num_den_pade5::<B>(xs[i], c, one, n_four, four);
        num[i] = n_v;
        den[i] = d_v;
    }
    let mut out = [B::Vec4::zero(); N];
    let mut i = 0;
    while i + 3 < N {
        let [y0, y1, y2, y3] =
            B::Vec4::rcp_ge1_quad_nr2([den[i], den[i + 1], den[i + 2], den[i + 3]]);
        out[i] = num[i].mul(y0).clamp(n_one, one);
        out[i + 1] = num[i + 1].mul(y1).clamp(n_one, one);
        out[i + 2] = num[i + 2].mul(y2).clamp(n_one, one);
        out[i + 3] = num[i + 3].mul(y3).clamp(n_one, one);
        i += 4;
    }
    while i + 1 < N {
        let (y0, y1) = den[i].rcp_ge1_pair_nr2(den[i + 1]);
        out[i] = num[i].mul(y0).clamp(n_one, one);
        out[i + 1] = num[i + 1].mul(y1).clamp(n_one, one);
        i += 2;
    }
    if i < N {
        out[i] = num[i].mul(den[i].rcp_ge1_nr2()).clamp(n_one, one);
    }
    out
}

/// Evaluates `6` `tanh` vectors alongside `2` `d >= 1.0` reciprocals in two `4`-vector `NR2` waves.
#[inline(always)]
#[must_use]
pub fn tanh_pade5_6_with_rcp2_vec4<B: MathBackend>(
    xs: [B::Vec4; 6],
    d0: B::Vec4,
    d1: B::Vec4,
) -> ([B::Vec4; 6], B::Vec4, B::Vec4) {
    let four = B::Vec4::splat(4.0);
    let n_four = B::Vec4::splat(-4.0);
    let one = B::Vec4::splat(1.0);
    let n_one = B::Vec4::splat(-1.0);
    let c = B::Vec4::from_array([N1, N2, D1, D2]);

    let (num0, den0) = eval_num_den_pade5::<B>(xs[0], c, one, n_four, four);
    let (num1, den1) = eval_num_den_pade5::<B>(xs[1], c, one, n_four, four);
    let (num2, den2) = eval_num_den_pade5::<B>(xs[2], c, one, n_four, four);
    let (num3, den3) = eval_num_den_pade5::<B>(xs[3], c, one, n_four, four);
    let [y0, y1, y2, y3] = B::Vec4::rcp_ge1_quad_nr2([den0, den1, den2, den3]);
    let out0 = num0.mul(y0).clamp(n_one, one);
    let out1 = num1.mul(y1).clamp(n_one, one);
    let out2 = num2.mul(y2).clamp(n_one, one);
    let out3 = num3.mul(y3).clamp(n_one, one);

    let (num4, den4) = eval_num_den_pade5::<B>(xs[4], c, one, n_four, four);
    let (num5, den5) = eval_num_den_pade5::<B>(xs[5], c, one, n_four, four);
    let [y4, y5, r0, r1] = B::Vec4::rcp_ge1_quad_nr2([den4, den5, d0.max(one), d1.max(one)]);
    let out4 = num4.mul(y4).clamp(n_one, one);
    let out5 = num5.mul(y5).clamp(n_one, one);

    ([out0, out1, out2, out3, out4, out5], r0, r1)
}

/// Division-free 4-lane SIMD Padé `[5/4]` `tanh(x)` (`C^1` clamped at `±4`, `NR2` in-register rcp, exact `+0.0`).
#[inline(always)]
#[must_use]
pub fn tanh_pade5_vec4<B: MathBackend>(x: B::Vec4) -> B::Vec4 {
    let [out] = tanh_pade5_batch_vec4::<B, 1>([x]);
    out
}

/// Evaluates two independent 4-lane Padé `[5/4]` `tanh` vectors in lockstep (`0-ULP` identical to [`tanh_pade5_vec4`]).
#[inline(always)]
#[must_use]
pub fn tanh_pade5_pair_vec4<B: MathBackend>(x0: B::Vec4, x1: B::Vec4) -> (B::Vec4, B::Vec4) {
    let [out0, out1] = tanh_pade5_batch_vec4::<B, 2>([x0, x1]);
    (out0, out1)
}

/// Scalar division-free Padé `[5/4]` `tanh(x)` (`0 FDIV`, `0 FSQRT`, `0 libm`, `NR2`, exact `+0.0` at `0`).
#[inline(always)]
#[must_use]
pub fn tanh_pade5_f32(x: f32) -> f32 {
    let xc = x.clamp(-4.0, 4.0);
    let s = xc * xc;
    let num = xc * (1.0 + s * (N1 + s * N2));
    let den = 1.0 + s * (D1 + s * D2);
    let y = recip_nr2_pos_f32(den);
    (num * y).clamp(-1.0, 1.0)
}

/// Division-free 4-lane SIMD `sigmoid(x) = 0.5 + 0.5 * tanh(0.5 * x)` using [`tanh_pade5_vec4`].
#[inline(always)]
#[must_use]
pub fn sigmoid_pade5_vec4<B: MathBackend>(x: B::Vec4) -> B::Vec4 {
    let half = B::Vec4::splat(0.5);
    half.fma(tanh_pade5_vec4::<B>(x.mul(half)), half)
}

/// Evaluates two independent 4-lane `sigmoid` vectors in lockstep (`0-ULP` identical to [`sigmoid_pade5_vec4`]).
#[inline(always)]
#[must_use]
pub fn sigmoid_pade5_pair_vec4<B: MathBackend>(x0: B::Vec4, x1: B::Vec4) -> (B::Vec4, B::Vec4) {
    let half = B::Vec4::splat(0.5);
    let (t0, t1) = tanh_pade5_pair_vec4::<B>(x0.mul(half), x1.mul(half));
    (half.fma(t0, half), half.fma(t1, half))
}

/// Scalar division-free `sigmoid(x) = 0.5 + 0.5 * tanh(0.5 * x)` using [`tanh_pade5_f32`].
#[inline(always)]
#[must_use]
pub fn sigmoid_pade5_f32(x: f32) -> f32 {
    (0.5 + 0.5 * tanh_pade5_f32(0.5 * x)).clamp(0.0, 1.0)
}

/// Positive reciprocal `1.0 / d` for `d >= 1.0` (`0 FDIV`, `NR2`).
#[inline(always)]
#[must_use]
pub fn recip_pos_ge1_f32(d: f32) -> f32 {
    recip_nr2_pos_f32(d.max(1.0))
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

    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.polyadaa.midpoint]
    /// verify[dsp.op-n.polyadaa.rail-crossing]
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
        assert!((tanh_pade5_f32(4.0) - 1.0).abs() < 1.0e-5);
        assert!((tanh_pade5_f32(-4.0) + 1.0).abs() < 1.0e-5);
        assert!(sigmoid_pade5_f32(-8.0).abs() < 1.0e-5);
        assert!((sigmoid_pade5_f32(8.0) - 1.0).abs() < 1.0e-5);
        assert!((recip_pos_ge1_f32(4.0) - 0.25).abs() < 1.0e-5);
        assert!((recip_pos_ge1_f32(0.25) - 1.0).abs() < 1.0e-5);

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
            assert!((sig_s[i] - sig_d[i]).abs() < 5.0e-6);
        }
        let batch_s = tanh_pade5_batch_vec4::<ScalarBackend, 5>([vx_s; 5]);
        let batch_d = tanh_pade5_batch_vec4::<DefaultMathBackend, 5>([vx_d; 5]);
        for i in 0..5 {
            for l in 0..4 {
                assert!((batch_s[i].to_array()[l] - batch_d[i].to_array()[l]).abs() < 1.0e-5);
            }
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
                prop_assert!((f64::from(simd_t[i]) - exact).abs() < 1.0e-5);
                prop_assert!((f64::from(scal_t[i]) - exact).abs() < 1.0e-5);
                prop_assert!((f64::from(sf) - exact).abs() < 1.0e-5);
                prop_assert!(simd_t[i].abs() <= 1.0);
                prop_assert!((exact - f64::from(v_in[i]).tanh()).abs() < 1.5e-3);
            }
            prop_assert!((simd_t[0] + simd_t[2]).abs() < 1.0e-6);
            if a <= b {
                prop_assert!(simd_t[0] <= simd_t[1] + 1.5e-5);
            } else {
                prop_assert!(simd_t[1] <= simd_t[0] + 1.5e-5);
            }
            let sig_a = sigmoid_pade5_f32(a);
            let sig_b = sigmoid_pade5_f32(b);
            prop_assert!((0.0..=1.0).contains(&sig_a));
            if a <= b {
                prop_assert!(sig_a <= sig_b + 1.5e-5);
            }
            let va = <DefaultMathBackend as MathBackend>::Vec4::from_array([a, b, -a, 0.5 * b]);
            let vb = <DefaultMathBackend as MathBackend>::Vec4::from_array([b, -b, a, -0.5 * a]);
            let vc = <DefaultMathBackend as MathBackend>::Vec4::from_array([0.1 * a, a + 1.0, b - 1.0, 0.0]);
            let vd = <DefaultMathBackend as MathBackend>::Vec4::from_array([-2.0, 2.0, a * 0.25, b * 0.25]);
            let (pt0, pt1) = tanh_pade5_pair_vec4::<DefaultMathBackend>(va, vb);
            prop_assert_eq!(
                pt0.to_array().map(f32::to_bits),
                tanh_pade5_vec4::<DefaultMathBackend>(va).to_array().map(f32::to_bits)
            );
            prop_assert_eq!(
                pt1.to_array().map(f32::to_bits),
                tanh_pade5_vec4::<DefaultMathBackend>(vb).to_array().map(f32::to_bits)
            );
            let (qs0, qs1) = sigmoid_pade5_pair_vec4::<DefaultMathBackend>(va, vb);
            for (q, v) in [(qs0, va), (qs1, vb)] {
                prop_assert_eq!(
                    q.to_array().map(f32::to_bits),
                    sigmoid_pade5_vec4::<DefaultMathBackend>(v).to_array().map(f32::to_bits)
                );
            }
            let ve = va.mul(<DefaultMathBackend as MathBackend>::Vec4::splat(0.5));
            let vf = vb.mul(<DefaultMathBackend as MathBackend>::Vec4::splat(0.5));
            let d0 = <DefaultMathBackend as MathBackend>::Vec4::from_array([
                1.0 + a.abs(),
                1.0 + b.abs(),
                0.5,
                2.5,
            ]);
            let d1 = <DefaultMathBackend as MathBackend>::Vec4::from_array([
                1.0,
                3.0 + a.abs(),
                1.2 + b.abs(),
                0.1,
            ]);
            let (t_six, r0, r1) = tanh_pade5_6_with_rcp2_vec4::<DefaultMathBackend>(
                [va, vb, vc, vd, ve, vf],
                d0,
                d1,
            );
            let (p2, p3) = tanh_pade5_pair_vec4::<DefaultMathBackend>(vc, vd);
            let (p4, p5) = tanh_pade5_pair_vec4::<DefaultMathBackend>(ve, vf);
            for (lhs, rhs) in [
                (t_six[0], pt0),
                (t_six[1], pt1),
                (t_six[2], p2),
                (t_six[3], p3),
                (t_six[4], p4),
                (t_six[5], p5),
            ] {
                prop_assert_eq!(
                    lhs.to_array().map(f32::to_bits),
                    rhs.to_array().map(f32::to_bits)
                );
            }
            let one = <DefaultMathBackend as MathBackend>::Vec4::splat(1.0);
            let (exp_r0, exp_r1) = d0.max(one).rcp_ge1_pair_nr2(d1.max(one));
            prop_assert_eq!(r0.to_array().map(f32::to_bits), exp_r0.to_array().map(f32::to_bits));
            prop_assert_eq!(r1.to_array().map(f32::to_bits), exp_r1.to_array().map(f32::to_bits));
        }
    }
}
