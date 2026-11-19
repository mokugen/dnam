//! Standalone scalar (`f32`) and high-precision oracle (`f64`) DSP kernels for `PolyADAA` (`ref[dsp.op-n.polyadaa.midpoint]`),
//! Algebraic Newton Unit Rotor (`ref[dsp.op-m.rotor.contractive]`), and 4-Tap Cubic Lagrange interpolation (`ref[dsp.op-t.reader.catmull-rom]`).

/// Fixed-bank `PolyADAA` pre-activation scale factor `2 / 3` (`ref[dsp.op-n.polyadaa.clamp]`).
const PRE_ACTIVATION_SCALE_F32: f32 = 2.0 / 3.0;

/// Unit-slope `PolyADAA` odd cubic linear coefficient `3 / 2` (`ref[dsp.op-n.polyadaa.midpoint]`).
const ADAA_ODD_LINEAR_COEFF_F64: f64 = 1.5;
const ADAA_ODD_LINEAR_COEFF_F32: f32 = ADAA_ODD_LINEAR_COEFF_F64 as f32;

/// Unit-slope `PolyADAA` odd cubic third-order coefficient `1 / 2` (`ref[dsp.op-n.polyadaa.midpoint]`).
const ADAA_ODD_CUBIC_COEFF_F64: f64 = 0.5;
const ADAA_ODD_CUBIC_COEFF_F32: f32 = ADAA_ODD_CUBIC_COEFF_F64 as f32;

/// `PolyADAA` odd antiderivative quadratic coefficient `3 / 4` (`integral of 1.5 u`).
const ADAA_ANTIDERIV_U2_COEFF_F64: f64 = ADAA_ODD_LINEAR_COEFF_F64 / 2.0;

/// `PolyADAA` odd antiderivative quartic coefficient `1 / 8` (`integral of 0.5 u^3`).
const ADAA_ANTIDERIV_U4_COEFF_F64: f64 = ADAA_ODD_CUBIC_COEFF_F64 / 4.0;

/// `PolyADAA` even antiderivative cubic coefficient `1 / 3` (`integral of u^2`).
const ADAA_ANTIDERIV_U3_COEFF_F64: f64 = 1.0 / 3.0;

/// `PolyADAA` even antiderivative quintic coefficient `1 / 5` (`integral of u^4`).
const ADAA_ANTIDERIV_U5_COEFF_F64: f64 = 1.0 / 5.0;

/// Midpoint/half-difference scale `1 / 2` (`ref[dsp.op-n.polyadaa.midpoint]`).
const ADAA_MIDPOINT_HALF_F64: f64 = 0.5;
const ADAA_MIDPOINT_HALF_F32: f32 = ADAA_MIDPOINT_HALF_F64 as f32;

/// Even `PolyADAA` half-difference quadratic coefficient `1 / 3` (`ref[dsp.op-n.polyadaa.midpoint]`).
const ADAA_EVEN_D2_COEFF_F64: f64 = 1.0 / 3.0;
const ADAA_EVEN_D2_COEFF_F32: f32 = ADAA_EVEN_D2_COEFF_F64 as f32;

/// Even `PolyADAA` quartic cross-term coefficient `2` (`ref[dsp.op-n.polyadaa.midpoint]`).
const ADAA_EVEN_CROSS_COEFF_F64: f64 = 2.0;
const ADAA_EVEN_CROSS_COEFF_F32: f32 = ADAA_EVEN_CROSS_COEFF_F64 as f32;

/// Even `PolyADAA` quartic half-difference coefficient `1 / 5` (`ref[dsp.op-n.polyadaa.midpoint]`).
const ADAA_EVEN_D4_COEFF_F64: f64 = 1.0 / 5.0;
const ADAA_EVEN_D4_COEFF_F32: f32 = ADAA_EVEN_D4_COEFF_F64 as f32;

/// Largest admissible `f32` angle magnitude strictly below `pi` (`0x4049_0FDA`,
/// `ref[dsp.op-m.rotor.contractive]`).
const MAX_ROTOR_ANGLE_F32: f32 = f32::from_bits(crate::math::MAX_ROTOR_ANGLE_F32_BITS);

/// First-order Newton inverse-sqrt stabilization constants `(3/2, 1/2)` (`ref[dsp.op-m.rotor.contractive]`).
const ROTOR_NR_THREE_HALVES_F32: f32 = 1.5;
const ROTOR_NR_HALF_F32: f32 = 0.5;

/// Compile-time 4-tap Cubic Lagrange basis weight factors `1/6` and `1/2` (`0 FDIV`,
/// `ref[dsp.op-t.reader.catmull-rom]`).
const LAGRANGE_INV_6_F64: f64 = 1.0 / 6.0;
const LAGRANGE_HALF_F64: f64 = 0.5;

const _: () = assert!(MAX_ROTOR_ANGLE_F32 < core::f32::consts::PI);
const _: () =
    assert!((2.0 * ADAA_ANTIDERIV_U2_COEFF_F64 - ADAA_ODD_LINEAR_COEFF_F64).to_bits() == 0);
const _: () =
    assert!((4.0 * ADAA_ANTIDERIV_U4_COEFF_F64 - ADAA_ODD_CUBIC_COEFF_F64).to_bits() == 0);
const _: () = assert!(LAGRANGE_INV_6_F64 > 0.16 && LAGRANGE_INV_6_F64 < 0.17);

/// Clamps pre-activation input $u_n = \operatorname{clamp}(\frac{2}{3}x, -1.0, +1.0)$ (`ref[dsp.op-n.polyadaa.clamp]`).
#[inline(always)]
pub fn clamp_pre_activation(x: f32) -> f32 {
    (PRE_ACTIVATION_SCALE_F32 * x).clamp(-1.0, 1.0)
}

/// Evaluates the instantaneous `PolyADAA` waveshaper $g_{\text{nl}}(u) = \beta(1.5u - 0.5u^3) + \delta(u^2 - u^4)$
/// used to initialize the exact 3-state DC resting equilibrium.
#[inline(always)]
pub fn eval_poly_adaa_waveshaper(u: f32, beta: f32, delta: f32) -> f32 {
    let u2 = u * u;
    beta * u * (ADAA_ODD_LINEAR_COEFF_F32 - ADAA_ODD_CUBIC_COEFF_F32 * u2) + delta * (u2 - u2 * u2)
}

/// High-precision `f64` oracle for $g_{\text{nl}}(u) = \beta(1.5u - 0.5u^3) + \delta(u^2 - u^4)$.
#[inline(always)]
pub fn eval_poly_adaa_waveshaper_f64(u: f64, beta: f64, delta: f64) -> f64 {
    let u2 = u * u;
    beta * u * (ADAA_ODD_LINEAR_COEFF_F64 - ADAA_ODD_CUBIC_COEFF_F64 * u2) + delta * (u2 - u2 * u2)
}

/// Evaluates the analytical antiderivative $F(u) = \beta(\frac{3}{4}u^2 - \frac{1}{8}u^4) + \delta(\frac{1}{3}u^3 - \frac{1}{5}u^5)$ in `f64`.
#[inline(always)]
pub fn eval_poly_adaa_antiderivative_f64(u: f64, beta: f64, delta: f64) -> f64 {
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u2 * u2;
    let u5 = u4 * u;
    beta * (ADAA_ANTIDERIV_U2_COEFF_F64 * u2 - ADAA_ANTIDERIV_U4_COEFF_F64 * u4)
        + delta * (ADAA_ANTIDERIV_U3_COEFF_F64 * u3 - ADAA_ANTIDERIV_U5_COEFF_F64 * u5)
}

/// Exact division-free algebraic `PolyADAA` kernel in `f32` (`ref[dsp.op-n.polyadaa.midpoint]`, `0 FDIV`, `0 FSQRT`).
#[inline(always)]
pub fn eval_poly_adaa_algebraic(u_prev: f32, u_curr: f32, beta: f32, delta: f32) -> f32 {
    let m = ADAA_MIDPOINT_HALF_F32 * (u_curr + u_prev);
    let d = ADAA_MIDPOINT_HALF_F32 * (u_curr - u_prev);
    let m2 = m * m;
    let d2 = d * d;
    let odd_term = beta * m * (ADAA_ODD_LINEAR_COEFF_F32 - ADAA_ODD_CUBIC_COEFF_F32 * (m2 + d2));
    let even_term = delta
        * ((m2 + ADAA_EVEN_D2_COEFF_F32 * d2)
            - (m2 * m2 + ADAA_EVEN_CROSS_COEFF_F32 * m2 * d2 + ADAA_EVEN_D4_COEFF_F32 * d2 * d2));
    odd_term + even_term
}

/// High-precision `f64` oracle for the exact division-free algebraic `PolyADAA` kernel (`Test 1B`).
#[inline(always)]
pub fn eval_poly_adaa_algebraic_f64(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    let m = ADAA_MIDPOINT_HALF_F64 * (u_curr + u_prev);
    let d = ADAA_MIDPOINT_HALF_F64 * (u_curr - u_prev);
    let m2 = m * m;
    let d2 = d * d;
    let odd_term = beta * m * (ADAA_ODD_LINEAR_COEFF_F64 - ADAA_ODD_CUBIC_COEFF_F64 * (m2 + d2));
    let even_term = delta
        * ((m2 + ADAA_EVEN_D2_COEFF_F64 * d2)
            - (m2 * m2 + ADAA_EVEN_CROSS_COEFF_F64 * m2 * d2 + ADAA_EVEN_D4_COEFF_F64 * d2 * d2));
    odd_term + even_term
}

/// Quadrant-reduced degree-13/14 Horner contractive rotor in `f32` (`ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.ulp-guard]`).
/// Returns `(cos_theta, sin_theta)` with $c_{32}^2 + s_{32}^2 \le 1.0$.
#[inline(always)]
pub fn algebraic_unit_rotor(theta_rad: f32) -> (f32, f32) {
    if !theta_rad.is_finite() {
        return (1.0, 0.0);
    }
    const PI: f32 = core::f32::consts::PI;
    const TAU: f32 = core::f32::consts::TAU;
    const INV_TAU: f32 = 1.0 / core::f32::consts::TAU;
    let wrapped = if (-MAX_ROTOR_ANGLE_F32..=MAX_ROTOR_ANGLE_F32).contains(&theta_rad) {
        theta_rad
    } else {
        let q = crate::math::floor_f32((theta_rad + PI) * INV_TAU);
        (theta_rad - q * TAU).clamp(-MAX_ROTOR_ANGLE_F32, MAX_ROTOR_ANGLE_F32)
    };
    crate::math::contractive_rotor_f32(wrapped).unwrap_or((1.0, 0.0))
}

/// High-precision `f64` oracle for the quadrant-reduced contractive unit-circle rotor (`Test 2`).
#[inline(always)]
pub fn algebraic_unit_rotor_f64(theta_rad: f64) -> (f64, f64) {
    if !theta_rad.is_finite() || theta_rad > 1.0e15 || theta_rad < -1.0e15 {
        return (1.0, 0.0);
    }
    let (sin, cos) = crate::math::const_eval::const_sin_cos_f64(theta_rad);
    let norm2_err = cos * cos + sin * sin - 1.0;
    if cos.is_finite() && sin.is_finite() && norm2_err < 1.0e-6 && norm2_err > -1.0e-6 {
        (cos, sin)
    } else {
        (1.0, 0.0)
    }
}

/// Stabilizes an autonomous LFO complex phasor on the unit circle via one Algebraic Newton step (`ref[dsp.op-m.rotor.contractive]`).
#[inline(always)]
pub fn stabilize_complex_rotor(re: f32, im: f32) -> (f32, f32) {
    let y1 = ROTOR_NR_THREE_HALVES_F32 - ROTOR_NR_HALF_F32 * (re * re + im * im);
    (y1 * re, y1 * im)
}

/// Evaluates the 4-tap Cubic Lagrange basis weights $[c_{-1}(\mu), c_0(\mu), c_1(\mu), c_2(\mu)]$
/// in `f32` (`ref[dsp.op-t.reader.catmull-rom]`, `0 FDIV`), delegated to
/// [`dnam_kernels::cubic_lagrange_weights_f32`].
#[inline(always)]
pub fn cubic_lagrange_weights(mu: f32) -> [f32; 4] {
    dnam_kernels::cubic_lagrange_weights_f32(mu)
}

/// High-precision `f64` oracle for 4-tap Cubic Lagrange basis weights (`Test 3`, `0 FDIV`).
#[inline(always)]
pub fn cubic_lagrange_weights_f64(mu: f64) -> [f64; 4] {
    let c_m1 = -mu * (mu - 1.0) * (mu - 2.0) * LAGRANGE_INV_6_F64;
    let c_0 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) * LAGRANGE_HALF_F64;
    let c_1 = -(mu + 1.0) * mu * (mu - 2.0) * LAGRANGE_HALF_F64;
    let c_2 = (mu + 1.0) * mu * (mu - 1.0) * LAGRANGE_INV_6_F64;
    [c_m1, c_0, c_1, c_2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_reference_dsp_kernels_f32_vs_f64() {
        assert_eq!(clamp_pre_activation(3.0), 1.0);
        assert_eq!(clamp_pre_activation(-3.0), -1.0);
        let ws32 = eval_poly_adaa_waveshaper(0.5, 0.4, 0.1);
        let ws64 = eval_poly_adaa_waveshaper_f64(0.5, 0.4, 0.1);
        assert!((ws32 as f64 - ws64).abs() < 1e-6);

        let anti = eval_poly_adaa_antiderivative_f64(0.5, 0.4, 0.1);
        assert!(anti.is_finite());

        let ad32 = eval_poly_adaa_algebraic(0.2, 0.4, 0.5, 0.1);
        let ad64 = eval_poly_adaa_algebraic_f64(0.2, 0.4, 0.5, 0.1);
        assert!((ad32 as f64 - ad64).abs() < 1e-6);

        let (r32_c, r32_s) = algebraic_unit_rotor(1.0);
        let (r64_c, r64_s) = algebraic_unit_rotor_f64(1.0);
        assert!((r32_c as f64 - r64_c).abs() < 1e-6);
        assert!((r32_s as f64 - r64_s).abs() < 1e-6);
        assert!((r64_c - 1.0_f64.cos()).abs() < 1e-14);
        assert!((r64_s - 1.0_f64.sin()).abs() < 1e-14);
        for bad32 in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(algebraic_unit_rotor(bad32), (1.0, 0.0));
        }
        for bad64 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(algebraic_unit_rotor_f64(bad64), (1.0, 0.0));
        }

        let (sc, ss) = stabilize_complex_rotor(0.99, 0.0);
        assert!((sc - 1.0).abs() < 1e-3 && ss.abs() < 1e-6);

        let cw32 = cubic_lagrange_weights(0.3);
        let cw64 = cubic_lagrange_weights_f64(0.3);
        assert!((cw32[0] as f64 - cw64[0]).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_poly_adaa_algebraic_matches_antiderivative_difference_quotient(
            u_prev in -1.0_f64..1.0_f64,
            u_curr in -1.0_f64..1.0_f64,
            beta in 0.0_f64..2.0_f64,
            delta in -0.25_f64..0.25_f64,
        ) {
            let alg = eval_poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
            let du = u_curr - u_prev;
            if du.abs() > 1e-5 {
                let f_curr = eval_poly_adaa_antiderivative_f64(u_curr, beta, delta);
                let f_prev = eval_poly_adaa_antiderivative_f64(u_prev, beta, delta);
                let quotient = (f_curr - f_prev) / du;
                prop_assert!((alg - quotient).abs() < 1e-9);
            } else {
                let u_mid = 0.5 * (u_curr + u_prev);
                let g_mid = eval_poly_adaa_waveshaper_f64(u_mid, beta, delta);
                prop_assert!((alg - g_mid).abs() < 1e-8);
            }
        }

        #[test]
        fn prop_algebraic_unit_rotor_stays_on_unit_circle(
            theta_rad in 0.0001_f64..(std::f64::consts::PI - 0.0001),
        ) {
            let (c64, s64) = algebraic_unit_rotor_f64(theta_rad);
            let norm64 = (c64 * c64 + s64 * s64).sqrt();
            prop_assert!(norm64 <= 1.000_000_000_1);
            prop_assert!((norm64 - 1.0).abs() < 1e-7);

            let (c32, s32) = algebraic_unit_rotor(theta_rad as f32);
            let norm32 = ((c32 as f64) * (c32 as f64) + (s32 as f64) * (s32 as f64)).sqrt();
            prop_assert!(norm32 <= 1.000_001);
            prop_assert!((norm32 - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_algebraic_unit_rotor_f64_handles_nonfinite_and_extreme_angles() {
        for &theta in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0e16, -1.0e16] {
            assert_eq!(algebraic_unit_rotor_f64(theta), (1.0, 0.0));
        }
        let (c, s) = algebraic_unit_rotor_f64(1.0e6);
        assert!(c.is_finite() && s.is_finite());
        assert!((c * c + s * s - 1.0).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_cubic_lagrange_partition_of_unity_and_cubic_exactness(
            mu in 0.0_f64..1.0_f64,
            a0 in -5.0_f64..5.0_f64,
            a1 in -5.0_f64..5.0_f64,
            a2 in -5.0_f64..5.0_f64,
            a3 in -5.0_f64..5.0_f64,
        ) {
            let w = cubic_lagrange_weights_f64(mu);
            let sum = w[0] + w[1] + w[2] + w[3];
            prop_assert!((sum - 1.0).abs() < 1e-14);

            let p = |x: f64| a0 + a1 * x + a2 * x * x + a3 * x * x * x;
            let interp = w[0] * p(-1.0) + w[1] * p(0.0) + w[2] * p(1.0) + w[3] * p(2.0);
            let exact = p(mu);
            prop_assert!((interp - exact).abs() < 1e-11);
        }
    }
}
