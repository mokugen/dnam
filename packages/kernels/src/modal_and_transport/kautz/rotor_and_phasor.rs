//! Contractive pure-`f32` `Op-M` unit rotor and 4-lane `Q0.32` quadrature phasor bank
//! (`ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.zero-hold]`,
//! `ref[dsp.op-m.rotor.quadrant-horner]`, `ref[dsp.op-m.rotor.ulp-guard]`,
//! `ref[dsp.op-m.rotor.q0-192-lfo]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`/`UDIV`, zero 64-bit scalars, and zero `libm`
//! calls in runtime code (`ref[sdk.rust.shared-nostd-kernels]`).

use crate::simd::{
    BIASED_INT_MAGIC_F32, IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_MASK, KernelVec4, const_abs_f32,
    const_f32_pow2, const_taylor_inv_fact_f32, flush_subnormal_f32_bits,
};

/// Bit-level finiteness check avoiding `f32::is_finite()` (`OpConstant Inf` fails Naga validation).
#[inline(always)]
const fn is_finite_f32_bits(x: f32) -> bool {
    (x.to_bits() & IEEE754_F32_EXP_MASK) != IEEE754_F32_EXP_MASK
}

/// Exact `2^32` scale factor in `binary32` (`4_294_967_296.0`) for converting normalized
/// frequency `cycles/sample` to `Q0.32` phase increments (`32-bit` integer wrap only).
const TWO_POW_32_F32: f32 = const_f32_pow2(32);

/// Exact `2^-30` quarter-cycle scaling factor for `Q0.32` quadrant fractional extraction.
const INV_TWO_POW_30_F32: f32 = const_f32_pow2(-30);

/// Upper Nyquist-safe normalized frequency bound (`< 0.5 cycles/sample`) for `Q0.32` phasors
/// (`0x3EFF_FFFE` = `0.4999999403953552`).
const MAX_PHASOR_CYCLES_PER_SAMPLE: f32 = f32::from_bits(0x3EFF_FFFE);

/// Exact `2^31` upper bound for `freq_hz_to_q32_inc` conversion (`0x4F00_0000`).
const MAX_Q32_INC_F32: f32 = const_f32_pow2(31);

/// Cody-Waite `pi/2` high chunk (`0x3FC9_0000`: 16 trailing mantissa zero bits so `q * PIO2_1` is
/// exact for `|q| <= 65_536`, `ref[dsp.op-m.rotor.quadrant-horner]`).
const PIO2_1: f32 = f32::from_bits(core::f32::consts::FRAC_PI_2.to_bits() & 0xFFFF_0000);

/// Cody-Waite `pi/2` middle residual chunk (`0x39FD_A000`, 13 trailing mantissa zero bits).
const PIO2_2: f32 = f32::from_bits(0x39FD_A000);

/// Cody-Waite `pi/2` low residual chunk (`0x33A2_2169`, remaining `binary32` precision tail).
const PIO2_3: f32 = f32::from_bits(0x33A2_2169);

/// Reciprocal `2 / pi` for quadrant index estimation (`1 / FRAC_PI_2`).
const INV_PIO2: f32 = 1.0 / core::f32::consts::FRAC_PI_2;

/// Angle magnitude threshold above which `Q * PIO2_2` loses phase precision (`65_536.0 rad`).
const MAX_CODY_WAITE_RAD: f32 = 65_536.0;

/// Reduced quadrant half-width ceiling `pi/4 + margin` rejecting overflowed reduction arguments.
const REDUCED_QUADRANT_GUARD_RAD: f32 = 0.85;

/// Degree-5 Hermite quarter-sine coefficient `c1 = pi / 2` (`ref[dsp.op-m.rotor.q0-192-lfo]`).
const HERMITE5_C1: f32 = crate::simd::HERMITE5_SIN_C1_F32;

/// Degree-5 Hermite quarter-sine coefficient `c3 = pi - 2.5`.
const HERMITE5_C3: f32 = crate::simd::HERMITE5_SIN_C3_F32;

/// Degree-5 Hermite quarter-sine coefficient `c5 = pi / 2 - 1.5`.
const HERMITE5_C5: f32 = crate::simd::HERMITE5_SIN_C5_F32;

const SIN_C3: f32 = const_taylor_inv_fact_f32(3, true);
const SIN_C5: f32 = const_taylor_inv_fact_f32(5, false);
const SIN_C7: f32 = const_taylor_inv_fact_f32(7, true);
const SIN_C9: f32 = const_taylor_inv_fact_f32(9, false);
const SIN_C11: f32 = const_taylor_inv_fact_f32(11, true);
const SIN_C13: f32 = const_taylor_inv_fact_f32(13, false);

const COS_C2: f32 = const_taylor_inv_fact_f32(2, true);
const COS_C4: f32 = const_taylor_inv_fact_f32(4, false);
const COS_C6: f32 = const_taylor_inv_fact_f32(6, true);
const COS_C8: f32 = const_taylor_inv_fact_f32(8, false);
const COS_C10: f32 = const_taylor_inv_fact_f32(10, true);
const COS_C12: f32 = const_taylor_inv_fact_f32(12, false);
const COS_C14: f32 = const_taylor_inv_fact_f32(14, true);

/// Integer bit pattern of `BIASED_INT_MAGIC_F32` (`0x4B40_0000`) for fast-math-immune RNE
/// extraction (`ref[dsp.op-m.rotor.quadrant-horner]`).
const RNE_MAGIC_BITS: i32 = BIASED_INT_MAGIC_F32.to_bits() as i32;

const _: () = assert!(BIASED_INT_MAGIC_F32.to_bits() == 0x4B40_0000);
const _: () = assert!(RNE_MAGIC_BITS == 0x4B40_0000);
const _: () = assert!(MAX_CODY_WAITE_RAD * INV_PIO2 < const_f32_pow2(22));
const _: () = assert!(PIO2_1.to_bits() == 0x3FC9_0000);
const _: () =
    assert!(const_abs_f32((PIO2_1 + PIO2_2 + PIO2_3) - core::f32::consts::FRAC_PI_2) < 1.0e-7);
const _: () = assert!(const_abs_f32((HERMITE5_C1 - HERMITE5_C3 + HERMITE5_C5) - 1.0) < 1.0e-6);
const _: () = assert!(COS_C2 == -0.5);

/// Rounds `x` (`|x| <= 2^22 - 1`) to nearest-even `i32` via an integer bitcast barrier
/// preventing GPU fast-math compilers from folding `(x + C) - C -> x`.
#[inline(always)]
fn round_ties_even_small_i32(x: f32) -> i32 {
    ((x + BIASED_INT_MAGIC_F32).to_bits() as i32).wrapping_sub(RNE_MAGIC_BITS)
}

#[inline(always)]
fn hermite5_quarter_sine(u: f32) -> f32 {
    let u2 = u * u;
    u * (HERMITE5_C1 + u2 * (-HERMITE5_C3 + u2 * HERMITE5_C5))
}

/// Evaluates quadrature `(cos, sin)` on a single `Q0.32` phase word via quadrant-folded
/// degree-5 Hermite sine (`ref[dsp.op-m.rotor.q0-192-lfo]`).
#[inline(always)]
#[must_use]
pub fn eval_q32_quadrature_f32(phase: u32) -> (f32, f32) {
    eval_q32_lane(phase)
}

#[inline(always)]
fn eval_q32_lane(phase: u32) -> (f32, f32) {
    let quadrant = phase >> 30;
    let u = ((phase & 0x3FFF_FFFF) as f32) * INV_TWO_POW_30_F32;
    let sin_q = hermite5_quarter_sine(u);
    let cos_q = hermite5_quarter_sine(1.0 - u);
    match quadrant {
        0 => (cos_q, sin_q),
        1 => (-sin_q, cos_q),
        2 => (-cos_q, -sin_q),
        _ => (sin_q, -cos_q),
    }
}

#[inline(always)]
fn freq_hz_to_q32_inc(freq_hz: f32, inv_sample_rate_hz: f32) -> u32 {
    if !freq_hz.is_finite() || !inv_sample_rate_hz.is_finite() || inv_sample_rate_hz <= 0.0 {
        return 0;
    }
    let abs_freq = const_abs_f32(freq_hz);
    let cycles = (abs_freq * inv_sample_rate_hz).clamp(0.0, MAX_PHASOR_CYCLES_PER_SAMPLE);
    let inc_f = cycles * TWO_POW_32_F32 + 0.5;
    (inc_f.clamp(0.0, MAX_Q32_INC_F32)) as u32
}

/// Pure-`f32` contractive unit rotor evaluating `(cos(theta_rad), sin(theta_rad))` with zero
/// runtime `FDIV`, zero `libm`, strictly 32-bit scalars, and guaranteed `cos^2 + sin^2 <= 1.0`
/// (`0-ULP` overshoot guard, `ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.zero-hold]`,
/// `ref[dsp.op-m.rotor.quadrant-horner]`, `ref[dsp.op-m.rotor.ulp-guard]`).
#[inline(always)]
#[must_use]
pub fn algebraic_unit_rotor_f32(theta_rad: f32) -> (f32, f32) {
    if !is_finite_f32_bits(theta_rad)
        || theta_rad == 0.0
        || const_abs_f32(theta_rad) > MAX_CODY_WAITE_RAD
    {
        return (1.0, 0.0);
    }

    let q_i = round_ties_even_small_i32(theta_rad * INV_PIO2);
    let q_f = q_i as f32;
    let r = ((theta_rad - q_f * PIO2_1) - q_f * PIO2_2) - q_f * PIO2_3;
    if !is_finite_f32_bits(r) || const_abs_f32(r) > REDUCED_QUADRANT_GUARD_RAD {
        return (1.0, 0.0);
    }

    let r2 = r * r;
    let s_poly =
        SIN_C3 + r2 * (SIN_C5 + r2 * (SIN_C7 + r2 * (SIN_C9 + r2 * (SIN_C11 + r2 * SIN_C13))));
    let sin_r = r + (r * r2) * s_poly;
    let c_poly = COS_C2
        + r2 * (COS_C4
            + r2 * (COS_C6 + r2 * (COS_C8 + r2 * (COS_C10 + r2 * (COS_C12 + r2 * COS_C14)))));
    let cos_r = 1.0 + r2 * c_poly;

    let quadrant = q_i & 3;
    let (mut cos_out, mut sin_out) = match quadrant {
        0 => (cos_r, sin_r),
        1 => (-sin_r, cos_r),
        2 => (-cos_r, -sin_r),
        _ => (sin_r, -cos_r),
    };

    let mut c_bits = cos_out.to_bits();
    let mut s_bits = sin_out.to_bits();
    let on_axis = (c_bits & IEEE754_F32_ABS_MASK) == 0 || (s_bits & IEEE754_F32_ABS_MASK) == 0;
    for _ in 0..2 {
        let norm2 = cos_out * cos_out + sin_out * sin_out;
        if norm2 > 1.0 || (!on_axis && norm2 == 1.0) {
            if (c_bits & IEEE754_F32_ABS_MASK) >= (s_bits & IEEE754_F32_ABS_MASK) {
                c_bits -= 1;
                cos_out = f32::from_bits(c_bits);
            } else {
                s_bits -= 1;
                sin_out = f32::from_bits(s_bits);
            }
        } else {
            break;
        }
    }

    (
        flush_subnormal_f32_bits(cos_out),
        flush_subnormal_f32_bits(sin_out),
    )
}

/// 4-lane `Q0.32` wrapped audio-rate quadrature phasor bank (`0.05 Hz..20 kHz`) using exact
/// wrapping `u32` accumulators and degree-5 Hermite quarter-sine quadrature
/// (strictly 32-bit, `0 libm`, `0 FDIV`, `ref[dsp.op-m.rotor.q0-192-lfo]`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PhasorQ32Bank4 {
    /// Unsigned `Q0.32` phase accumulators (`2^32 == 2 * pi rad`).
    pub phase: [u32; 4],
    /// Unsigned `Q0.32` per-sample phase increments.
    pub inc: [u32; 4],
}

impl PhasorQ32Bank4 {
    /// Constructs a [`PhasorQ32Bank4`] from raw `Q0.32` phases and increments.
    #[inline(always)]
    #[must_use]
    pub const fn new(phase: [u32; 4], inc: [u32; 4]) -> Self {
        Self { phase, inc }
    }

    /// Constructs a zero-phase [`PhasorQ32Bank4`] from 4 frequencies in Hz and precomputed
    /// reciprocal sample rate `1.0 / fs_hz`.
    #[inline(always)]
    #[must_use]
    pub fn from_freqs_hz(freqs_hz: [f32; 4], inv_sample_rate_hz: f32) -> Self {
        Self {
            phase: [0; 4],
            inc: [
                freq_hz_to_q32_inc(freqs_hz[0], inv_sample_rate_hz),
                freq_hz_to_q32_inc(freqs_hz[1], inv_sample_rate_hz),
                freq_hz_to_q32_inc(freqs_hz[2], inv_sample_rate_hz),
                freq_hz_to_q32_inc(freqs_hz[3], inv_sample_rate_hz),
            ],
        }
    }

    /// Updates lane `lane` (`< 4`) increment from `freq_hz` and `inv_sample_rate_hz = 1.0 / fs_hz`
    /// without division (`0 FDIV`). Ignores out-of-range lanes cleanly.
    #[inline(always)]
    pub fn set_freq_hz(&mut self, lane: usize, freq_hz: f32, inv_sample_rate_hz: f32) {
        if lane < 4 {
            self.inc[lane] = freq_hz_to_q32_inc(freq_hz, inv_sample_rate_hz);
        }
    }

    /// Evaluates current `(cos, sin)` arrays across all 4 lanes without advancing phase.
    #[inline(always)]
    #[must_use]
    pub fn eval_quadrature(&self) -> ([f32; 4], [f32; 4]) {
        let (c0, s0) = eval_q32_lane(self.phase[0]);
        let (c1, s1) = eval_q32_lane(self.phase[1]);
        let (c2, s2) = eval_q32_lane(self.phase[2]);
        let (c3, s3) = eval_q32_lane(self.phase[3]);
        ([c0, c1, c2, c3], [s0, s1, s2, s3])
    }

    /// Advances all 4 `Q0.32` phase accumulators by `self.inc` with wrapping `u32` addition.
    #[inline(always)]
    pub fn advance(&mut self) {
        for lane in 0..4 {
            self.phase[lane] = self.phase[lane].wrapping_add(self.inc[lane]);
        }
    }

    /// Evaluates current `(cos, sin)` across all 4 lanes and advances phase accumulators cleanly.
    #[inline(always)]
    pub fn step_quadrature(&mut self) -> ([f32; 4], [f32; 4]) {
        let out = self.eval_quadrature();
        self.advance();
        out
    }

    /// Evaluates and advances all 4 lanes returning `(cos, sin)` as a pair of [`KernelVec4`].
    #[inline(always)]
    pub fn step_quadrature_vec4<V: KernelVec4>(&mut self) -> (V, V) {
        let (cos_arr, sin_arr) = self.step_quadrature();
        (V::from_array(cos_arr), V::from_array(sin_arr))
    }

    /// Returns the `OS2X`-halved unsigned `Q0.32` phase increments (`inc >> 1`, `ref[dsp.op-m.rotor.q0-192-lfo]`).
    #[inline(always)]
    #[must_use]
    pub const fn halved_increments(&self) -> [u32; 4] {
        [
            self.inc[0] >> 1,
            self.inc[1] >> 1,
            self.inc[2] >> 1,
            self.inc[3] >> 1,
        ]
    }

    /// Evaluates all 4 quadrature lanes and advances phases by caller-supplied `Q0.32` increments
    /// (`inc >> 1` at `OS2X`, `ref[dsp.op-m.rotor.q0-192-lfo]`).
    #[inline(always)]
    pub fn step_with_inc_vec4<V: KernelVec4>(&mut self, inc_q32: [u32; 4]) -> (V, V) {
        let (cos_arr, sin_arr) = self.eval_quadrature();
        self.phase[0] = self.phase[0].wrapping_add(inc_q32[0]);
        self.phase[1] = self.phase[1].wrapping_add(inc_q32[1]);
        self.phase[2] = self.phase[2].wrapping_add(inc_q32[2]);
        self.phase[3] = self.phase[3].wrapping_add(inc_q32[3]);
        (V::from_array(cos_arr), V::from_array(sin_arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::GpuVec4;
    use proptest::prelude::*;

    /// verify[dsp.op-m.rotor.contractive]
    /// verify[dsp.op-m.rotor.zero-hold]
    /// verify[dsp.op-m.rotor.quadrant-horner]
    /// verify[dsp.op-m.rotor.ulp-guard]
    #[test]
    fn test_algebraic_unit_rotor_extremes_zero_hold_and_accuracy() {
        assert_eq!(algebraic_unit_rotor_f32(0.0), (1.0, 0.0));
        assert_eq!(algebraic_unit_rotor_f32(-0.0), (1.0, 0.0));
        assert_eq!(algebraic_unit_rotor_f32(f32::NAN), (1.0, 0.0));
        assert_eq!(algebraic_unit_rotor_f32(f32::INFINITY), (1.0, 0.0));
        assert_eq!(algebraic_unit_rotor_f32(1.0e6), (1.0, 0.0));

        for step in -400..=400 {
            let theta = (step as f32) * 0.015;
            let (c, s) = algebraic_unit_rotor_f32(theta);
            let norm2_f64 = f64::from(c) * f64::from(c) + f64::from(s) * f64::from(s);
            assert!(c * c + s * s <= 1.0);
            assert!(norm2_f64 <= 1.0000002);
            let ref_c = f64::from(theta).cos();
            let ref_s = f64::from(theta).sin();
            assert!((f64::from(c) - ref_c).abs() < 3.0e-7);
            assert!((f64::from(s) - ref_s).abs() < 3.0e-7);
        }

        // Verify half-pi axes, ULP-guard trigger angles, and large Cody-Waite angles.
        for th in [
            0.001_f32,
            0.786_f32,
            core::f32::consts::FRAC_PI_2,
            core::f32::consts::PI,
            -core::f32::consts::FRAC_PI_2,
            5_000.0_f32,
            -5_000.0_f32,
        ] {
            let (c, s) = algebraic_unit_rotor_f32(th);
            assert!(c * c + s * s <= 1.0);
            assert!((f64::from(c) - f64::from(th).cos()).abs() < 5.0e-7);
            assert!((f64::from(s) - f64::from(th).sin()).abs() < 5.0e-7);
        }
    }

    /// verify[dsp.op-m.rotor.q0-192-lfo]
    #[test]
    fn test_phasor_q32_bank4_quadrants_and_freq_updates() {
        let mut bank =
            PhasorQ32Bank4::from_freqs_hz([0.05, 440.0, 6_000.0, 20_000.0], 1.0 / 48_000.0);
        assert!(bank.inc[0] > 0 && bank.inc[3] > bank.inc[2]);
        bank.set_freq_hz(9, 100.0, 1.0 / 48_000.0);
        bank.set_freq_hz(0, f32::NAN, 1.0 / 48_000.0);
        assert_eq!(bank.inc[0], 0);
        bank.set_freq_hz(1, 440.0, 0.0);
        assert_eq!(bank.inc[1], 0);
        bank.set_freq_hz(0, 0.05, 1.0 / 48_000.0);
        assert!(bank.inc[0] > 0);
        assert_eq!(PhasorQ32Bank4::default().phase, [0; 4]);

        for _ in 0..64 {
            let (c_vec, s_vec) = bank.step_quadrature_vec4::<GpuVec4>();
            let c = c_vec.to_array();
            let s = s_vec.to_array();
            for lane in 0..4 {
                assert!(((c[lane] * c[lane] + s[lane] * s[lane]) - 1.0).abs() < 1.0e-3);
            }
        }

        let mut corners = PhasorQ32Bank4::new([0, 0x4000_0000, 0x8000_0000, 0xC000_0000], [0; 4]);
        let (cc, cs) = corners.step_quadrature();
        assert_eq!(cc, [1.0, 0.0, -1.0, 0.0]);
        assert_eq!(cs, [0.0, 1.0, 0.0, -1.0]);

        let mut os2x_bank = PhasorQ32Bank4::new([0; 4], [4, 8, 16, 32]);
        let half = os2x_bank.halved_increments();
        assert_eq!(half, [2, 4, 8, 16]);
        let (c0, s0) = os2x_bank.step_with_inc_vec4::<GpuVec4>(half);
        assert_eq!(c0.to_array(), [1.0; 4]);
        assert_eq!(s0.to_array(), [0.0; 4]);
        assert_eq!(os2x_bank.phase, [2, 4, 8, 16]);
    }

    proptest! {
        #[test]
        fn prop_algebraic_unit_rotor_is_contractive_and_matches_f64(
            theta in -2_000.0_f32..2_000.0_f32,
        ) {
            let (c, s) = algebraic_unit_rotor_f32(theta);
            prop_assert!(c.is_finite() && s.is_finite());
            prop_assert!(c * c + s * s <= 1.0);
            prop_assert!((f64::from(c) - f64::from(theta).cos()).abs() < 5.0e-7);
            prop_assert!((f64::from(s) - f64::from(theta).sin()).abs() < 5.0e-7);
        }

        #[test]
        fn prop_phasor_q32_matches_hermite5_accuracy(
            phase in any::<u32>(),
            freq_hz in 0.05_f32..20_000.0_f32,
        ) {
            let mut bank = PhasorQ32Bank4::new([phase, 0, 0, 0], [0; 4]);
            bank.set_freq_hz(1, freq_hz, 1.0 / 48_000.0);
            let (c, s) = bank.step_quadrature();
            let rad = (f64::from(phase) / 4_294_967_296.0) * core::f64::consts::TAU;
            prop_assert!((f64::from(c[0]) - rad.cos()).abs() < 5.0e-4);
            prop_assert!((f64::from(s[0]) - rad.sin()).abs() < 5.0e-4);
            prop_assert!(bank.inc[1] > 0);
        }
    }
}
