//! Compile-time derived operator constants, wide-range base-2 exponential (`exp2_clamped16`),
//! and `OS2X` `1x -> 2x` linear midpoint interpolation helpers (`ref[dsp.op-t.window.partition-of-unity]`,
//! `ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-m.rotor.q0-192-lfo]`,
//! `ref[repo.code.constant-eval-and-derivations]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`, zero 64-bit types, and zero `libm` calls.

use super::const_eval::const_max_f32_bits_le_ratio;
use super::{
    BIASED_INT_MAGIC_F32, EXP2_C2, EXP2_C3, EXP2_C4, EXP2_C5, IEEE754_F32_EXP_BIAS, KernelVec4,
};

/// Derives an IEEE-754 `binary32` exact power-of-two scalar `2^exp` (`exp in -126..=127`)
/// purely via exponent bit shifting (`ref[repo.code.constant-eval-and-derivations]`).
#[inline(always)]
#[must_use]
pub const fn const_f32_pow2(exp: i32) -> f32 {
    let biased = (IEEE754_F32_EXP_BIAS as i32) + exp;
    f32::from_bits((biased as u32) << 23)
}

/// Bit pattern of `TANK_MAX_G_LIN_F32` (`<= 0.9990`, `0x3F7F_BE76`).
pub const TANK_MAX_G_LIN_BITS: u32 = const_max_f32_bits_le_ratio(999, 1_000);

/// Largest `binary32` spatial-tank loop gain `<= 0.9990` (`0x3F7F_BE76`).
pub const TANK_MAX_G_LIN_F32: f32 = f32::from_bits(TANK_MAX_G_LIN_BITS);

/// Bit pattern of `TANK_MAX_V_CEIL_F32` (`<= 0.985`, `0x3F7C_28F5`).
pub const TANK_MAX_V_CEIL_BITS: u32 = const_max_f32_bits_le_ratio(985, 1_000);

/// Largest `binary32` cubic feedback saturator ceiling `<= 0.985` (`0x3F7C_28F5`).
pub const TANK_MAX_V_CEIL_F32: f32 = f32::from_bits(TANK_MAX_V_CEIL_BITS);

/// Exact IEEE-754 `binary32` `2^-31` (`0x3000_0000`) for signed `Q0.32` triangle folding.
pub const Q32_INV_2POW31_F32: f32 = const_f32_pow2(-31);

/// Exact IEEE-754 `binary32` `2^-32` (`0x2F80_0000`) for `Q0.32` increment scaling.
pub const Q32_INV_2POW32_F32: f32 = const_f32_pow2(-32);

/// Degree-5 Hermite quarter-sine coefficient `c1 = pi / 2` (`f(0)=0, f'(0)=pi/2`).
pub const HERMITE5_SIN_C1_F32: f32 = core::f32::consts::FRAC_PI_2;

/// Degree-5 Hermite quarter-sine coefficient `c3 = pi - 2.5`.
pub const HERMITE5_SIN_C3_F32: f32 = core::f32::consts::PI - 2.5;

/// Degree-5 Hermite quarter-sine coefficient `c5 = pi / 2 - 1.5`.
pub const HERMITE5_SIN_C5_F32: f32 = core::f32::consts::FRAC_PI_2 - 1.5;

#[inline(always)]
const fn const_abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

const _: () = assert!(TANK_MAX_G_LIN_BITS == 0x3F7F_BE76);
const _: () = assert!(TANK_MAX_G_LIN_F32 <= 0.9990);
const _: () = assert!(TANK_MAX_V_CEIL_BITS == 0x3F7C_28F5);
const _: () = assert!(TANK_MAX_V_CEIL_F32 <= 0.985);
const _: () = assert!(Q32_INV_2POW31_F32.to_bits() == 0x3000_0000);
const _: () = assert!(Q32_INV_2POW32_F32.to_bits() == 0x2F80_0000);
const _: () = assert!(
    const_abs_f32((HERMITE5_SIN_C1_F32 - HERMITE5_SIN_C3_F32 + HERMITE5_SIN_C5_F32) - 1.0) < 1.0e-6
);

/// Evaluates `2^clamp(x, lo, hi)` via nearest-integer split (`BIASED_INT_MAGIC_F32`),
/// IEEE-754 exponent injection, and the canonical degree-5 minimax Horner polynomial.
#[inline(always)]
pub(super) fn eval_exp2_clamped_scalar_f32(x: f32, lo: f32, hi: f32) -> f32 {
    let x_ge_lo = if x >= lo { x } else { lo };
    let xc = if x_ge_lo <= hi { x_ge_lo } else { hi };
    let k_biased = xc + BIASED_INT_MAGIC_F32;
    let k = k_biased - BIASED_INT_MAGIC_F32;
    let f = xc - k;
    let pow2_k = f32::from_bits(k_biased.to_bits().wrapping_add(IEEE754_F32_EXP_BIAS) << 23);
    let poly = EXP2_C4 + EXP2_C5 * f;
    let poly = EXP2_C3 + poly * f;
    let poly = EXP2_C2 + poly * f;
    let poly = core::f32::consts::LN_2 + poly * f;
    let poly = 1.0 + poly * f;
    pow2_k * poly
}

/// Evaluates `2^clamp(x, -16.0, 16.0)` in pure IEEE-754 `f32` (`Op-N[Pre]` uniform quantizer
/// scale `D_q` and step `Delta_q`, `ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline(always)]
#[must_use]
pub fn exp2_clamped16_f32(x: f32) -> f32 {
    eval_exp2_clamped_scalar_f32(x, -16.0, 16.0)
}

/// Computes the `OS2X` `1x -> 2x` linear midpoint `0.5 * (prev + curr)` across 4 `KernelVec4`
/// lanes (`ref[dsp.op-m.hilbert.lattice]`).
#[inline(always)]
#[must_use]
pub fn os2x_midpoint_vec4<V: KernelVec4>(prev: V, curr: V) -> V {
    V::splat(0.5).mul(prev.add(curr))
}

/// Computes the `OS2X` `1x -> 2x` linear midpoint `0.5 * (prev + curr)` across `[f32; 4]`.
#[inline(always)]
#[must_use]
pub fn os2x_midpoint_f32x4(prev: [f32; 4], curr: [f32; 4]) -> [f32; 4] {
    [
        0.5 * (prev[0] + curr[0]),
        0.5 * (prev[1] + curr[1]),
        0.5 * (prev[2] + curr[2]),
        0.5 * (prev[3] + curr[3]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::GpuVec4;
    use proptest::prelude::*;

    #[test]
    fn test_op_constants_exp2_clamped16_and_os2x_midpoints() {
        assert_eq!(TANK_MAX_G_LIN_BITS, 0x3F7F_BE76);
        assert_eq!(TANK_MAX_V_CEIL_BITS, 0x3F7C_28F5);
        assert_eq!(Q32_INV_2POW31_F32.to_bits(), 0x3000_0000);
        assert_eq!(Q32_INV_2POW32_F32.to_bits(), 0x2F80_0000);
        assert_eq!(const_f32_pow2(-31).to_bits(), Q32_INV_2POW31_F32.to_bits());
        assert_eq!(const_abs_f32(-1.5), 1.5);

        for n in -16..=16 {
            let expected = 2.0_f64.powi(n) as f32;
            assert_eq!(exp2_clamped16_f32(n as f32).to_bits(), expected.to_bits());
        }
        assert_eq!(
            exp2_clamped16_f32(f32::NAN).to_bits(),
            (2.0_f32.powi(-16)).to_bits()
        );
        assert_eq!(exp2_clamped16_f32(-99.0), exp2_clamped16_f32(-16.0));
        assert_eq!(exp2_clamped16_f32(99.0), exp2_clamped16_f32(16.0));

        let mid_arr = os2x_midpoint_f32x4([1.0, -2.0, 0.25, 4.0], [3.0, 2.0, 0.75, -8.0]);
        let mid_vec = os2x_midpoint_vec4(
            GpuVec4::from_array([1.0, -2.0, 0.25, 4.0]),
            GpuVec4::from_array([3.0, 2.0, 0.75, -8.0]),
        )
        .to_array();
        assert_eq!(mid_arr, [2.0, 0.0, 0.5, -2.0]);
        assert_eq!(mid_arr.map(f32::to_bits), mid_vec.map(f32::to_bits));
    }

    proptest! {
        #[test]
        fn prop_exp2_clamped16_matches_f64_pow2_and_vec4(
            x in prop::array::uniform4(-16.0_f32..16.0_f32),
        ) {
            let v = GpuVec4::from_array(x).exp2_clamped16().to_array();
            for lane in 0..4 {
                let s = exp2_clamped16_f32(x[lane]);
                prop_assert_eq!(v[lane].to_bits(), s.to_bits());
                let oracle = 2.0_f64.powf(f64::from(x[lane]));
                let rel_err = (f64::from(s) - oracle).abs() / oracle;
                prop_assert!(rel_err < 3.0e-7, "rel_err = {rel_err}");
            }
        }
    }
}
