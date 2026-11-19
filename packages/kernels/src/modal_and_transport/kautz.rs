//! Contractive pure-`f32` `Op-M` rotors, 4-lane `Q0.32` quadrature phasors, 8-lane stereo Stage III
//! Kautz modal state-space steps, and associative complex modal scan operators
//! (`ref[dsp.op-m.rotor.contractive]`, `ref[dsp.op-m.rotor.q0-192-lfo]`,
//! `ref[dsp.op-m.stage3.coupled-form]`, `ref[dsp.op-m.recurrence.stereo-simd]`).
//!
//! Strictly 32-bit, `#![no_std]`, zero `FDIV`/`FSQRT`/`UDIV`, zero 64-bit scalars, and zero `libm`
//! calls in runtime code (`ref[repo.code.shared-nostd-kernels]`).

use crate::simd::{KernelVec4, MAX_PARALRU_R_F32, StereoVec8, flush_subnormal_f32_bits};

/// Compile-time biased integer rounding magic `1.5 * 2^23` (`0x4B40_0000`).
const BIASED_INT_MAGIC_F32: f32 = 1.5 * ((1_u32 << 23) as f32);

/// Exact `2^32` scale factor in `binary32` (`4_294_967_296.0`) for converting normalized
/// frequency `cycles/sample` to `Q0.32` phase increments (`32-bit` integer wrap only).
const TWO_POW_32_F32: f32 = f32::from_bits((127 + 32) << 23);

/// Exact `2^-30` quarter-cycle scaling factor for `Q0.32` quadrant fractional extraction.
const INV_TWO_POW_30_F32: f32 = f32::from_bits((127 - 30) << 23);

/// Upper Nyquist-safe normalized frequency bound (`< 0.5 cycles/sample`) for `Q0.32` phasors
/// (`0x3EFF_FFFE` = `0.4999999403953552`).
const MAX_PHASOR_CYCLES_PER_SAMPLE: f32 = f32::from_bits(0x3EFF_FFFE);

/// Exact `2^31` upper bound for `freq_hz_to_q32_inc` conversion (`0x4F00_0000`).
const MAX_Q32_INC_F32: f32 = f32::from_bits((127 + 31) << 23);

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
const HERMITE5_C1: f32 = core::f32::consts::FRAC_PI_2;

/// Degree-5 Hermite quarter-sine coefficient `c3 = pi - 2.5`.
const HERMITE5_C3: f32 = core::f32::consts::PI - 2.5;

/// Degree-5 Hermite quarter-sine coefficient `c5 = pi / 2 - 1.5`.
const HERMITE5_C5: f32 = core::f32::consts::FRAC_PI_2 - 1.5;

/// Minimum contracted pole radius allowed during rapid Stage III formant sweeps
/// (`ref[dsp.op-m.stage3.sweep-contraction]`).
const STAGE3_SWEEP_MIN_R: f32 = 0.25;

/// Zero-sweep baseline radius ceiling for Stage III dynamic sweep contraction.
const STAGE3_SWEEP_BASE_CEIL: f32 = 0.9950;

/// Sweep-rate slope coefficient (`0.15 * |delta_theta|`) for Stage III contraction.
const STAGE3_SWEEP_SLOPE: f32 = 0.15;

/// Compile-time Taylor inverse factorial `(-1)^neg / n!` computed purely in `f32`
/// (`ref[repo.code.constant-eval-and-derivations]`).
const fn taylor_inv_fact_f32(n: u32, negative: bool) -> f32 {
    let mut fact = 1.0_f32;
    let mut k = 2_u32;
    while k <= n {
        fact *= k as f32;
        k += 1;
    }
    let mag = 1.0_f32 / fact;
    if negative { -mag } else { mag }
}

const SIN_C3: f32 = taylor_inv_fact_f32(3, true);
const SIN_C5: f32 = taylor_inv_fact_f32(5, false);
const SIN_C7: f32 = taylor_inv_fact_f32(7, true);
const SIN_C9: f32 = taylor_inv_fact_f32(9, false);
const SIN_C11: f32 = taylor_inv_fact_f32(11, true);
const SIN_C13: f32 = taylor_inv_fact_f32(13, false);

const COS_C2: f32 = taylor_inv_fact_f32(2, true);
const COS_C4: f32 = taylor_inv_fact_f32(4, false);
const COS_C6: f32 = taylor_inv_fact_f32(6, true);
const COS_C8: f32 = taylor_inv_fact_f32(8, false);
const COS_C10: f32 = taylor_inv_fact_f32(10, true);
const COS_C12: f32 = taylor_inv_fact_f32(12, false);
const COS_C14: f32 = taylor_inv_fact_f32(14, true);

const _: () = assert!(BIASED_INT_MAGIC_F32.to_bits() == 0x4B40_0000);
const _: () = assert!(PIO2_1.to_bits() == 0x3FC9_0000);
const _: () = assert!(abs_f32((PIO2_1 + PIO2_2 + PIO2_3) - core::f32::consts::FRAC_PI_2) < 1.0e-7);
const _: () = assert!(abs_f32((HERMITE5_C1 - HERMITE5_C3 + HERMITE5_C5) - 1.0) < 1.0e-6);
const _: () = assert!(COS_C2 == -0.5);

#[inline(always)]
const fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

#[inline(always)]
fn round_ties_even_small_f32(x: f32) -> f32 {
    let signed_magic = if x >= 0.0 {
        BIASED_INT_MAGIC_F32
    } else {
        -BIASED_INT_MAGIC_F32
    };
    (x + signed_magic) - signed_magic
}

#[inline(always)]
fn hermite5_quarter_sine(u: f32) -> f32 {
    let u2 = u * u;
    u * (HERMITE5_C1 + u2 * (-HERMITE5_C3 + u2 * HERMITE5_C5))
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
    let abs_freq = abs_f32(freq_hz);
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
    if !theta_rad.is_finite() || theta_rad == 0.0 || abs_f32(theta_rad) > MAX_CODY_WAITE_RAD {
        return (1.0, 0.0);
    }

    let q_f = round_ties_even_small_f32(theta_rad * INV_PIO2);
    let r = ((theta_rad - q_f * PIO2_1) - q_f * PIO2_2) - q_f * PIO2_3;
    if !r.is_finite() || abs_f32(r) > REDUCED_QUADRANT_GUARD_RAD {
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

    let quadrant = (q_f as i32) & 3;
    let (mut cos_out, mut sin_out) = match quadrant {
        0 => (cos_r, sin_r),
        1 => (-sin_r, cos_r),
        2 => (-cos_r, -sin_r),
        _ => (sin_r, -cos_r),
    };

    let mut c_bits = cos_out.to_bits();
    let mut s_bits = sin_out.to_bits();
    let on_axis = (c_bits & 0x7FFF_FFFF) == 0 || (s_bits & 0x7FFF_FFFF) == 0;
    for _ in 0..2 {
        let norm2 = cos_out * cos_out + sin_out * sin_out;
        if norm2 > 1.0 || (!on_axis && norm2 == 1.0) {
            if (c_bits & 0x7FFF_FFFF) >= (s_bits & 0x7FFF_FFFF) {
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

/// Evaluates Stage III rapid-sweep radial contraction (`ref[dsp.op-m.stage3.sweep-contraction]`)
/// purely in 32-bit `f32` (`min(radius, max(0.25, 0.9950 - 0.15 * |delta_theta|))`).
#[inline(always)]
#[must_use]
pub fn eval_stage3_sweep_contraction_f32(radius: f32, delta_theta_rad: f32) -> f32 {
    let r_clamped = if radius.is_finite() {
        radius.clamp(0.0, MAX_PARALRU_R_F32)
    } else {
        STAGE3_SWEEP_MIN_R
    };
    let abs_dtheta = if delta_theta_rad.is_finite() {
        abs_f32(delta_theta_rad)
    } else {
        8.0
    };
    let ceiling = (STAGE3_SWEEP_BASE_CEIL - STAGE3_SWEEP_SLOPE * abs_dtheta)
        .clamp(STAGE3_SWEEP_MIN_R, MAX_PARALRU_R_F32);
    r_clamped.min(ceiling)
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
}

/// 4-pole stereo Stage III Kautz modal state block (`4` shared complex conjugate pole pairs
/// across simultaneous `L` and `R` states, `ref[dsp.op-m.stage3.dual-state]`,
/// `ref[dsp.op-m.stage3.coupled-form]`, `ref[dsp.op-m.recurrence.stereo-simd]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KautzQuadState<V: KernelVec4> {
    /// Left-channel modal real state across 4 conjugate pole pairs.
    pub z_re_l: V,
    /// Left-channel modal imaginary state across 4 conjugate pole pairs.
    pub z_im_l: V,
    /// Right-channel modal real state across 4 conjugate pole pairs.
    pub z_re_r: V,
    /// Right-channel modal imaginary state across 4 conjugate pole pairs.
    pub z_im_r: V,
    /// Shared complex pole real part `r * cos(theta)`.
    pub alpha_re: V,
    /// Shared complex pole imaginary part `r * sin(theta)`.
    pub alpha_im: V,
    /// Shared normalized input drive weight `b_re = kappa * sqrt(1 - r_high^2)`.
    pub b_re: V,
    /// Left-channel modal readout real component.
    pub c_re_l: V,
    /// Left-channel modal readout imaginary component.
    pub c_im_l: V,
    /// Right-channel modal readout real component.
    pub c_re_r: V,
    /// Right-channel modal readout imaginary component.
    pub c_im_r: V,
}

impl<V: KernelVec4> KautzQuadState<V> {
    /// Constructs a zeroed stereo [`KautzQuadState`] from shared pole/drive parameters and
    /// independent `L`/`R` complex modal readouts.
    #[inline(always)]
    #[must_use]
    pub fn new(
        alpha_re: V,
        alpha_im: V,
        b_re: V,
        c_re_l: V,
        c_im_l: V,
        c_re_r: V,
        c_im_r: V,
    ) -> Self {
        Self {
            z_re_l: V::zero(),
            z_im_l: V::zero(),
            z_re_r: V::zero(),
            z_im_r: V::zero(),
            alpha_re,
            alpha_im,
            b_re,
            c_re_l,
            c_im_l,
            c_re_r,
            c_im_r,
        }
    }

    /// Zeros both Left and Right modal complex state vectors.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.z_re_l = V::zero();
        self.z_im_l = V::zero();
        self.z_re_r = V::zero();
        self.z_im_r = V::zero();
    }
}

/// Steps 4 conjugate Kautz modal pole pairs in both Left and Right channels simultaneously using
/// 4-lane [`KernelVec4`] vectors (`ref[dsp.op-m.stage3.coupled-form]`,
/// `ref[dsp.op-m.recurrence.symplectic]`).
///
/// Uses identical operation order and pairwise reductions as [`step_kautz_quad_stereo8`], yielding
/// exact `0-ULP` stereo parity against `InterleavedStereo8<V>`.
#[inline(always)]
pub fn step_kautz_quad_vec4<V: KernelVec4>(
    state: &mut KautzQuadState<V>,
    x_l: f32,
    x_r: f32,
) -> (f32, f32) {
    let xl = V::splat(x_l);
    let xr = V::splat(x_r);

    let next_re_l = state
        .b_re
        .mul(xl)
        .fma(state.alpha_re, state.z_re_l)
        .fms(state.alpha_im, state.z_im_l)
        .flush_subnormal();
    let next_im_l = state
        .alpha_re
        .mul(state.z_im_l)
        .fma(state.alpha_im, state.z_re_l)
        .flush_subnormal();

    let next_re_r = state
        .b_re
        .mul(xr)
        .fma(state.alpha_re, state.z_re_r)
        .fms(state.alpha_im, state.z_im_r)
        .flush_subnormal();
    let next_im_r = state
        .alpha_re
        .mul(state.z_im_r)
        .fma(state.alpha_im, state.z_re_r)
        .flush_subnormal();

    state.z_re_l = next_re_l;
    state.z_im_l = next_im_l;
    state.z_re_r = next_re_r;
    state.z_im_r = next_im_r;

    let modal_l = state.c_re_l.mul(next_re_l).fms(state.c_im_l, next_im_l);
    let modal_r = state.c_re_r.mul(next_re_r).fms(state.c_im_r, next_im_r);
    let (sum_l, sum_r) = modal_l.horizontal_sum_pair(modal_r);
    (
        flush_subnormal_f32_bits(2.0 * sum_l),
        flush_subnormal_f32_bits(2.0 * sum_r),
    )
}

/// Steps 4 shared conjugate Kautz modal pole pairs in both Left and Right channels simultaneously
/// in 8-lane [`StereoVec8`] with shared `broadcast_half` pole loads and exact `0-ULP` stereo
/// parity (`ref[dsp.op-m.recurrence.stereo-simd]`, `ref[arch.simd.two-level-traits]`).
#[inline(always)]
pub fn step_kautz_quad_stereo8<S: StereoVec8>(
    state: &mut KautzQuadState<S::Half>,
    x_l: f32,
    x_r: f32,
) -> (f32, f32) {
    let z_re = S::from_lr(state.z_re_l, state.z_re_r);
    let z_im = S::from_lr(state.z_im_l, state.z_im_r);
    let alpha_re = S::broadcast_half(state.alpha_re);
    let alpha_im = S::broadcast_half(state.alpha_im);
    let b_re = S::broadcast_half(state.b_re);
    let c_re = S::from_lr(state.c_re_l, state.c_re_r);
    let c_im = S::from_lr(state.c_im_l, state.c_im_r);
    let x = S::splat_lr(x_l, x_r);

    let next_re = b_re
        .mul(x)
        .fma(alpha_re, z_re)
        .fms(alpha_im, z_im)
        .flush_subnormal();
    let next_im = alpha_re.mul(z_im).fma(alpha_im, z_re).flush_subnormal();
    (state.z_re_l, state.z_re_r) = next_re.into_lr();
    (state.z_im_l, state.z_im_r) = next_im.into_lr();

    let modal = c_re.mul(next_re).fms(c_im, next_im);
    let (sum_l, sum_r) = modal.horizontal_sum_lr();
    (
        flush_subnormal_f32_bits(2.0 * sum_l),
        flush_subnormal_f32_bits(2.0 * sum_r),
    )
}

/// Associative affine complex recurrence element representing `z -> a * z + b` for Blelloch /
/// Kogge-Stone parallel modal block scans (`ref[repo.code.shared-nostd-kernels]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComplexModalScanElem {
    /// State transition real coefficient `a_re`.
    pub a_re: f32,
    /// State transition imaginary coefficient `a_im`.
    pub a_im: f32,
    /// Forced input injection real term `b_re`.
    pub b_re: f32,
    /// Forced input injection imaginary term `b_im`.
    pub b_im: f32,
}

impl Default for ComplexModalScanElem {
    fn default() -> Self {
        Self::identity()
    }
}

impl ComplexModalScanElem {
    /// Returns the scan monoid identity element `(a = 1 + 0i, b = 0 + 0i)`.
    #[inline(always)]
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            a_re: 1.0,
            a_im: 0.0,
            b_re: 0.0,
            b_im: 0.0,
        }
    }

    /// Constructs the per-sample complex modal transition `z_next = alpha * z + b_re * x`.
    #[inline(always)]
    #[must_use]
    pub const fn from_step(alpha_re: f32, alpha_im: f32, b_re: f32, x: f32) -> Self {
        Self {
            a_re: flush_subnormal_f32_bits(alpha_re),
            a_im: flush_subnormal_f32_bits(alpha_im),
            b_re: flush_subnormal_f32_bits(b_re * x),
            b_im: 0.0,
        }
    }
}

/// Associative composition `(first ; second)` applying `first` followed by `second`:
/// `z -> second.a * (first.a * z + first.b) + second.b`.
#[inline(always)]
#[must_use]
pub const fn compose_complex_modal_scan(
    first: ComplexModalScanElem,
    second: ComplexModalScanElem,
) -> ComplexModalScanElem {
    let a_re = second.a_re * first.a_re - second.a_im * first.a_im;
    let a_im = second.a_re * first.a_im + second.a_im * first.a_re;
    let b_re = second.a_re * first.b_re - second.a_im * first.b_im + second.b_re;
    let b_im = second.a_re * first.b_im + second.a_im * first.b_re + second.b_im;
    ComplexModalScanElem {
        a_re: flush_subnormal_f32_bits(a_re),
        a_im: flush_subnormal_f32_bits(a_im),
        b_re: flush_subnormal_f32_bits(b_re),
        b_im: flush_subnormal_f32_bits(b_im),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{GpuVec4, InterleavedStereo8};
    use proptest::prelude::*;

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

        assert_eq!(eval_stage3_sweep_contraction_f32(0.999, 0.0), 0.9950);
        assert_eq!(eval_stage3_sweep_contraction_f32(0.999, 10.0), 0.25);
        assert_eq!(eval_stage3_sweep_contraction_f32(f32::NAN, f32::NAN), 0.25);
    }

    #[test]
    fn test_phasor_q32_bank4_quadrants_and_kautz_zero_ulp_parity() {
        let mut bank =
            PhasorQ32Bank4::from_freqs_hz([0.05, 440.0, 6_000.0, 20_000.0], 1.0 / 48_000.0);
        assert!(bank.inc[0] > 0 && bank.inc[3] > bank.inc[2]);
        bank.set_freq_hz(9, 100.0, 1.0 / 48_000.0);
        bank.set_freq_hz(0, f32::NAN, 1.0 / 48_000.0);
        assert_eq!(bank.inc[0], 0);
        bank.set_freq_hz(0, 0.05, 1.0 / 48_000.0);
        assert!(bank.inc[0] > 0);

        for _ in 0..64 {
            let (c_vec, s_vec) = bank.step_quadrature_vec4::<GpuVec4>();
            let c = c_vec.to_array();
            let s = s_vec.to_array();
            for lane in 0..4 {
                assert!(((c[lane] * c[lane] + s[lane] * s[lane]) - 1.0).abs() < 1.0e-3);
            }
        }

        // Exact quadrant corners check.
        let mut corners = PhasorQ32Bank4::new([0, 0x4000_0000, 0x8000_0000, 0xC000_0000], [0; 4]);
        let (cc, cs) = corners.step_quadrature();
        assert_eq!(cc, [1.0, 0.0, -1.0, 0.0]);
        assert_eq!(cs, [0.0, 1.0, 0.0, -1.0]);

        let alpha_re = GpuVec4::from_array([0.92, 0.85, 0.70, 0.40]);
        let alpha_im = GpuVec4::from_array([0.12, 0.25, 0.45, 0.65]);
        let b_re = GpuVec4::from_array([0.1, 0.15, 0.2, 0.25]);
        let c_re_l = GpuVec4::from_array([0.5, -0.3, 0.2, 0.1]);
        let c_im_l = GpuVec4::from_array([0.1, 0.2, -0.15, 0.05]);
        let c_re_r = GpuVec4::from_array([0.4, 0.25, -0.1, 0.3]);
        let c_im_r = GpuVec4::from_array([-0.2, 0.1, 0.05, -0.15]);

        let mut kv = KautzQuadState::new(alpha_re, alpha_im, b_re, c_re_l, c_im_l, c_re_r, c_im_r);
        let mut ks = kv;
        for n in 0..64 {
            let xl = if n == 0 {
                1.0
            } else {
                ((n as f32) * 0.21).clamp(-1.0, 1.0) * 0.1
            };
            let xr = if n == 1 {
                -0.8
            } else {
                ((n as f32) * 0.13).clamp(-1.0, 1.0) * 0.1
            };
            let out_v = step_kautz_quad_vec4(&mut kv, xl, xr);
            let out_s = step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut ks, xl, xr);
            assert_eq!(out_v.0.to_bits(), out_s.0.to_bits());
            assert_eq!(out_v.1.to_bits(), out_s.1.to_bits());
            assert_eq!(kv, ks);
        }
        kv.reset();
        assert_eq!(kv.z_re_l, GpuVec4::zero());
    }

    #[test]
    fn test_complex_modal_scan_identity_associativity_and_sequential_equivalence() {
        let id = ComplexModalScanElem::default();
        let e = ComplexModalScanElem::from_step(0.9, 0.2, 0.15, 0.5);
        assert_eq!(compose_complex_modal_scan(id, e), e);
        assert_eq!(compose_complex_modal_scan(e, id), e);

        let a_re = 0.91_f32;
        let a_im = 0.23_f32;
        let b_re = 0.25_f32;
        let xs = [1.0_f32, -0.5, 0.25, 0.75, -0.125, 0.0, 0.4, -0.6];

        let mut state = KautzQuadState::new(
            GpuVec4::from_array([a_re, 0.0, 0.0, 0.0]),
            GpuVec4::from_array([a_im, 0.0, 0.0, 0.0]),
            GpuVec4::from_array([b_re, 0.0, 0.0, 0.0]),
            GpuVec4::from_array([0.5, 0.0, 0.0, 0.0]),
            GpuVec4::zero(),
            GpuVec4::from_array([0.5, 0.0, 0.0, 0.0]),
            GpuVec4::zero(),
        );
        let mut scan = ComplexModalScanElem::identity();
        for &x in &xs {
            let _ = step_kautz_quad_vec4(&mut state, x, x);
            scan = compose_complex_modal_scan(
                scan,
                ComplexModalScanElem::from_step(a_re, a_im, b_re, x),
            );
        }
        assert!((state.z_re_l.to_array()[0] - scan.b_re).abs() < 1.0e-5);
        assert!((state.z_im_l.to_array()[0] - scan.b_im).abs() < 1.0e-5);

        // Tree vs linear associativity check on 8 steps.
        let elems: [ComplexModalScanElem; 8] =
            core::array::from_fn(|i| ComplexModalScanElem::from_step(a_re, a_im, b_re, xs[i]));
        let p01 = compose_complex_modal_scan(elems[0], elems[1]);
        let p23 = compose_complex_modal_scan(elems[2], elems[3]);
        let p45 = compose_complex_modal_scan(elems[4], elems[5]);
        let p67 = compose_complex_modal_scan(elems[6], elems[7]);
        let tree = compose_complex_modal_scan(
            compose_complex_modal_scan(p01, p23),
            compose_complex_modal_scan(p45, p67),
        );
        assert!((tree.b_re - scan.b_re).abs() < 1.0e-5);
        assert!((tree.b_im - scan.b_im).abs() < 1.0e-5);
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

        #[test]
        fn prop_complex_modal_scan_associativity(
            xs in prop::array::uniform4(-1.0_f32..1.0_f32),
            r in 0.5_f32..0.99_f32,
            theta in -3.0_f32..3.0_f32,
        ) {
            let (ct, st) = algebraic_unit_rotor_f32(theta);
            let ar = r * ct;
            let ai = r * st;
            let b = 0.25_f32;
            let e0 = ComplexModalScanElem::from_step(ar, ai, b, xs[0]);
            let e1 = ComplexModalScanElem::from_step(ar, ai, b, xs[1]);
            let e2 = ComplexModalScanElem::from_step(ar, ai, b, xs[2]);
            let e3 = ComplexModalScanElem::from_step(ar, ai, b, xs[3]);
            let left = compose_complex_modal_scan(
                compose_complex_modal_scan(compose_complex_modal_scan(e0, e1), e2),
                e3,
            );
            let right = compose_complex_modal_scan(
                compose_complex_modal_scan(e0, e1),
                compose_complex_modal_scan(e2, e3),
            );
            prop_assert!((left.a_re - right.a_re).abs() < 1.0e-5);
            prop_assert!((left.a_im - right.a_im).abs() < 1.0e-5);
            prop_assert!((left.b_re - right.b_re).abs() < 1.0e-5);
            prop_assert!((left.b_im - right.b_im).abs() < 1.0e-5);
        }
    }
}
