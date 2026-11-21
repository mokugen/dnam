//! Standalone scalar (`f32`) and high-precision (`f64`) math helpers for `PolyADAA`,
//! algebraic contractive rotor, and cubic Lagrange interpolation.

const PRE_ACTIVATION_SCALE_F32: f32 = 2.0 / 3.0;
const ADAA_ODD_LINEAR_COEFF_F64: f64 = 1.5;
const ADAA_ODD_LINEAR_COEFF_F32: f32 = ADAA_ODD_LINEAR_COEFF_F64 as f32;
const ADAA_ODD_CUBIC_COEFF_F64: f64 = 0.5;
const ADAA_ODD_CUBIC_COEFF_F32: f32 = ADAA_ODD_CUBIC_COEFF_F64 as f32;
const ADAA_ANTIDERIV_U2_COEFF_F64: f64 = ADAA_ODD_LINEAR_COEFF_F64 / 2.0;
const ADAA_ANTIDERIV_U4_COEFF_F64: f64 = ADAA_ODD_CUBIC_COEFF_F64 / 4.0;
const ADAA_ANTIDERIV_U3_COEFF_F64: f64 = 1.0 / 3.0;
const ADAA_ANTIDERIV_U5_COEFF_F64: f64 = 1.0 / 5.0;
const ADAA_MIDPOINT_HALF_F64: f64 = 0.5;
const ADAA_MIDPOINT_HALF_F32: f32 = ADAA_MIDPOINT_HALF_F64 as f32;
const ADAA_EVEN_D2_COEFF_F64: f64 = 1.0 / 3.0;
const ADAA_EVEN_D2_COEFF_F32: f32 = ADAA_EVEN_D2_COEFF_F64 as f32;
const ADAA_EVEN_CROSS_COEFF_F64: f64 = 2.0;
const ADAA_EVEN_CROSS_COEFF_F32: f32 = ADAA_EVEN_CROSS_COEFF_F64 as f32;
const ADAA_EVEN_D4_COEFF_F64: f64 = 1.0 / 5.0;
const ADAA_EVEN_D4_COEFF_F32: f32 = ADAA_EVEN_D4_COEFF_F64 as f32;
const MAX_ROTOR_ANGLE_F32: f32 = f32::from_bits(crate::math::MAX_ROTOR_ANGLE_F32_BITS);
const ROTOR_NR_THREE_HALVES_F32: f32 = 1.5;
const ROTOR_NR_HALF_F32: f32 = 0.5;
const LAGRANGE_INV_6_F64: f64 = crate::math::const_inv_factorial_f64(3);
const LAGRANGE_HALF_F64: f64 = 0.5;

const _: () = assert!(MAX_ROTOR_ANGLE_F32 < core::f32::consts::PI);

/// Clamps pre-activation input $u_n = \mathrm{clamp}(2/3 \cdot x, -1.0, +1.0)$.
#[inline(always)]
pub fn clamp_pre_activation(x: f32) -> f32 {
    (PRE_ACTIVATION_SCALE_F32 * x).clamp(-1.0, 1.0)
}

/// Evaluates the instantaneous `PolyADAA` waveshaper in `f32`.
#[inline(always)]
pub fn eval_poly_adaa_waveshaper(u: f32, beta: f32, delta: f32) -> f32 {
    let u2 = u * u;
    beta * u * (ADAA_ODD_LINEAR_COEFF_F32 - ADAA_ODD_CUBIC_COEFF_F32 * u2) + delta * (u2 - u2 * u2)
}

/// High-precision `f64` oracle for `eval_poly_adaa_waveshaper`.
#[inline(always)]
pub fn eval_poly_adaa_waveshaper_f64(u: f64, beta: f64, delta: f64) -> f64 {
    let u2 = u * u;
    beta * u * (ADAA_ODD_LINEAR_COEFF_F64 - ADAA_ODD_CUBIC_COEFF_F64 * u2) + delta * (u2 - u2 * u2)
}

/// Evaluates the analytical antiderivative in `f64`.
#[inline(always)]
pub fn eval_poly_adaa_antiderivative_f64(u: f64, beta: f64, delta: f64) -> f64 {
    let u2 = u * u;
    let u3 = u2 * u;
    let u4 = u2 * u2;
    let u5 = u4 * u;
    beta * (ADAA_ANTIDERIV_U2_COEFF_F64 * u2 - ADAA_ANTIDERIV_U4_COEFF_F64 * u4)
        + delta * (ADAA_ANTIDERIV_U3_COEFF_F64 * u3 - ADAA_ANTIDERIV_U5_COEFF_F64 * u5)
}

/// Exact division-free algebraic `PolyADAA` kernel in `f32` (`0 FDIV`, `0 FSQRT`).
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

/// High-precision `f64` oracle for the algebraic `PolyADAA` kernel.
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

/// Pure-32-bit quadrant-reduced degree-13/14 Horner contractive rotor returning `(cos, sin)`
/// (`ref[dsp.op-m.rotor.quadrant-horner]`, zero `f64` math in runtime/VM paths).
#[inline(always)]
pub fn algebraic_unit_rotor(theta_rad: f32) -> (f32, f32) {
    if !theta_rad.is_finite() {
        return (1.0, 0.0);
    }
    if (-65_536.0..=65_536.0).contains(&theta_rad) {
        return dnam_kernels::algebraic_unit_rotor_f32(theta_rad);
    }
    const PI: f32 = core::f32::consts::PI;
    const TAU: f32 = core::f32::consts::TAU;
    const INV_TAU: f32 = 1.0 / core::f32::consts::TAU;
    let q = crate::math::floor_f32((theta_rad + PI) * INV_TAU);
    let wrapped = (theta_rad - q * TAU).clamp(-MAX_ROTOR_ANGLE_F32, MAX_ROTOR_ANGLE_F32);
    dnam_kernels::algebraic_unit_rotor_f32(wrapped)
}

/// High-precision `f64` oracle for the contractive unit-circle rotor.
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

/// Stabilizes an autonomous LFO complex phasor on the unit circle via one Newton step.
#[inline(always)]
pub fn stabilize_complex_rotor(re: f32, im: f32) -> (f32, f32) {
    let y1 = ROTOR_NR_THREE_HALVES_F32 - ROTOR_NR_HALF_F32 * (re * re + im * im);
    (y1 * re, y1 * im)
}

/// Evaluates 4-tap Cubic Lagrange basis weights in `f32`, delegated to `dnam-kernels`.
#[inline(always)]
pub fn cubic_lagrange_weights(mu: f32) -> [f32; 4] {
    dnam_kernels::cubic_lagrange_weights_f32(mu)
}

/// High-precision `f64` oracle for 4-tap Cubic Lagrange basis weights.
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
        assert!(
            (eval_poly_adaa_waveshaper(0.5, 0.4, 0.1) as f64
                - eval_poly_adaa_waveshaper_f64(0.5, 0.4, 0.1))
            .abs()
                < 1e-6
        );
        assert!(eval_poly_adaa_antiderivative_f64(0.5, 0.4, 0.1).is_finite());
        assert!(
            (eval_poly_adaa_algebraic(0.2, 0.4, 0.5, 0.1) as f64
                - eval_poly_adaa_algebraic_f64(0.2, 0.4, 0.5, 0.1))
            .abs()
                < 1e-6
        );

        let (r32_c, r32_s) = algebraic_unit_rotor(7.2);
        let (r64_c, r64_s) = algebraic_unit_rotor_f64(7.2);
        assert!((r32_c as f64 - r64_c).abs() < 1e-5);
        assert!((r32_s as f64 - r64_s).abs() < 1e-5);
        for bad32 in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(algebraic_unit_rotor(bad32), (1.0, 0.0));
        }
        for bad64 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(algebraic_unit_rotor_f64(bad64), (1.0, 0.0));
        }
        let (sc, ss) = stabilize_complex_rotor(0.99, 0.0);
        assert!((sc - 1.0).abs() < 1e-3 && ss.abs() < 1e-6);
        assert!(
            (cubic_lagrange_weights(0.3)[0] as f64 - cubic_lagrange_weights_f64(0.3)[0]).abs()
                < 1e-6
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_poly_adaa_and_unit_rotor_identities(
            u_prev in -1.0_f64..1.0_f64,
            u_curr in -1.0_f64..1.0_f64,
            theta_rad in 0.0001_f64..(core::f64::consts::PI - 0.0001),
        ) {
            let alg = eval_poly_adaa_algebraic_f64(u_prev, u_curr, 0.8, -0.1);
            let du = u_curr - u_prev;
            if du.abs() > 1e-5 {
                let q = (eval_poly_adaa_antiderivative_f64(u_curr, 0.8, -0.1)
                    - eval_poly_adaa_antiderivative_f64(u_prev, 0.8, -0.1))
                    / du;
                prop_assert!((alg - q).abs() < 1e-9);
            }
            let (c, s) = algebraic_unit_rotor(theta_rad as f32);
            prop_assert!((f64::from(c) * f64::from(c) + f64::from(s) * f64::from(s)) <= 1.0 + 1e-12);
        }
    }
}
