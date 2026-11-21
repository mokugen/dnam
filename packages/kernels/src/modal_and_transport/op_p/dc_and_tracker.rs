//! `Op-P` `T1` upstream `5 Hz` passband-normalized DC blocker (mono `T1DcBlockerKernel`, `24 B`
//! stereo `T1DcBlockerStereo`, and unified `[44.1 ..= 384]` kHz 160-bit midpoint coefficient
//! synthesis) plus the 4-feature Riemann sub-harmonic tracker (`T1SubHarmonicKernel`)
//! (`ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.envelope-norm]`,
//! `ref[dsp.op-m.tracker.toggle-sheet]`).
//!
//! Runtime execution paths are strictly `#![no_std]` 32-bit `f32`; the host/cold 160-bit search
//! (`compute_t1_dc_blocker_coeffs_bits`) is guarded under `#[cfg(not(target_arch = "spirv"))]`.

use crate::simd::{
    DC_BLOCKER_5HZ_48K_F32, GpuVec4, KernelVec4, MAX_PARALRU_R_F32, const_abs_f32, const_f32_pow2,
    flush_subnormal_f32_bits,
};
#[cfg(not(target_arch = "spirv"))]
use crate::simd::{
    IEEE754_F32_EXP_BIAS, IEEE754_F32_IMPLICIT_BIT, IEEE754_F32_MANTISSA_BITS,
    IEEE754_F32_MANTISSA_MASK, IEEE754_F32_SIGNIFICAND_BITS, IEEE754_F64_EXP_ALL_ONES,
    IEEE754_F64_EXP_BIAS, IEEE754_F64_MANTISSA_BITS,
};

/// Compile-time `48 kHz` `5 Hz` passband-normalized DC blocker gain `0.5 * (1 + R)`
/// (`0x3F7F_EA90`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
const DC_BLOCKER_5HZ_48K_GAIN_F32: f32 = 0.5 * (1.0 + DC_BLOCKER_5HZ_48K_F32);

/// Silence squelch squared-magnitude threshold (`r^2 < 1e-12`,
/// `ref[dsp.op-m.tracker.envelope-norm]`).
const T1_SILENCE_SQUELCH_R2: f32 = 1.0e-12;

/// Half-angle `(1 + |u_re|) * 0.5` singularity floor (`ref[dsp.op-m.tracker.toggle-sheet]`).
const T1_HALF_ANGLE_EPS: f32 = 1.0e-12;

/// Exact `binary32` `2^-18` tolerance for antipodal near-origin chord gating in
/// [`tracker_crosses_branch_cut_f32`] (`ref[dsp.op-m.tracker.toggle-sheet]`).
const BRANCH_CUT_CHORD_TOL: f32 = const_f32_pow2(-18);

const _: () = assert!(DC_BLOCKER_5HZ_48K_F32.to_bits() == 0x3F7F_D51F);
const _: () = assert!(DC_BLOCKER_5HZ_48K_GAIN_F32.to_bits() == 0x3F7F_EA90);
const _: () = assert!(BRANCH_CUT_CHORD_TOL > 0.0 && BRANCH_CUT_CHORD_TOL < 1.0e-5);

#[inline(always)]
fn clamp_t1_pole(dc_pole: f32) -> (f32, f32) {
    let pole = if dc_pole.is_finite() {
        dc_pole.clamp(0.0, MAX_PARALRU_R_F32)
    } else {
        DC_BLOCKER_5HZ_48K_F32
    };
    (pole, 0.5 * (1.0 + pole))
}

#[inline(always)]
fn step_dc_blocker_channel(
    pole: f32,
    gain: f32,
    prev_x: &mut f32,
    prev_y: &mut f32,
    x: f32,
) -> f32 {
    let x_clean = flush_subnormal_f32_bits(if x.is_finite() { x } else { 0.0 });
    let y = flush_subnormal_f32_bits(gain * (x_clean - *prev_x) + pole * *prev_y);
    if !y.is_finite() {
        *prev_x = 0.0;
        *prev_y = 0.0;
        return 0.0;
    }
    *prev_x = x_clean;
    *prev_y = y;
    y
}

/// Big-endian 256-bit unsigned integer used strictly in cold/host `160-bit` `T1` midpoint search.
#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct U256Be {
    limbs: [u64; 4],
}

#[cfg(not(target_arch = "spirv"))]
impl U256Be {
    const ZERO: Self = Self { limbs: [0; 4] };

    fn add(self, rhs: Self) -> Self {
        let mut out = [0_u64; 4];
        let mut carry = 0_u128;
        let mut idx = 4;
        while idx > 0 {
            idx -= 1;
            let s = u128::from(self.limbs[idx]) + u128::from(rhs.limbs[idx]) + carry;
            out[idx] = s as u64;
            carry = s >> 64;
        }
        Self { limbs: out }
    }

    fn mul_u64(self, rhs: u64) -> Self {
        let mut out = [0_u64; 4];
        let mut carry = 0_u128;
        let r = u128::from(rhs);
        let mut idx = 4;
        while idx > 0 {
            idx -= 1;
            let p = u128::from(self.limbs[idx]) * r + carry;
            out[idx] = p as u64;
            carry = p >> 64;
        }
        Self { limbs: out }
    }

    fn div_u64(self, divisor: u64) -> Self {
        let mut out = [0_u64; 4];
        let mut rem = 0_u128;
        for idx in 0..4 {
            let cur = (rem << 64) | u128::from(self.limbs[idx]);
            let (q, r) = div_rem_u128_by_u64_shift(cur, divisor);
            out[idx] = q;
            rem = r;
        }
        Self { limbs: out }
    }
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
fn div_rem_u128_by_u64_shift(num: u128, den: u64) -> (u64, u128) {
    let d = u128::from(den);
    if d == 0 || num < d {
        return (0, num);
    }
    let shift = d.leading_zeros().saturating_sub(num.leading_zeros());
    let mut q = 0_u128;
    let mut rem = num;
    let mut i = shift + 1;
    while i > 0 {
        i -= 1;
        let d_shifted = d << i;
        if rem >= d_shifted {
            rem -= d_shifted;
            q |= 1_u128 << i;
        }
    }
    (q as u64, rem)
}

#[cfg(not(target_arch = "spirv"))]
fn atanh_bounds_160bit(numerator: u64, denominator: u64) -> (U256Be, U256Be) {
    let z_num = denominator - numerator;
    let z_den = denominator + numerator;
    let one_shl_160 = U256Be {
        limbs: [0, 1_u64 << 32, 0, 0],
    };
    let mut cur = one_shl_160.mul_u64(z_num).div_u64(z_den);
    let z2_num = z_num * z_num;
    let z2_den = z_den * z_den;
    let mut lower = U256Be::ZERO;
    for idx in 0..8_u64 {
        lower = lower.add(cur.div_u64(2 * idx + 1));
        cur = cur.mul_u64(z2_num).div_u64(z2_den);
    }
    let upper = lower.add(U256Be {
        limbs: [0, 0, 0, 16],
    });
    (lower, upper)
}

/// Computes the unified `T1` tracker `5 Hz` DC blocker `(pole_bits, gain_bits)` over continuous
/// `44,100.0 ..= 384,000.0` Hz (`T1-MIDPOINT-SEARCH`, `op_m_modal_kautz.md`) without `libm` calls
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
///
/// Uses the unified `(first, last) = (0x00FF_D000, 0x00FF_FBFF)` bracket (`<= 14` comparisons) so
/// bridged rates in `(96,000, 176,400)` Hz such as `128,000` Hz resolve cleanly alongside exact
/// `T1-COEFFICIENT-VECTORS` anchors. Returns `None` outside `[44,100.0 ..= 384,000.0]` or non-finite.
#[cfg(not(target_arch = "spirv"))]
#[must_use]
pub fn compute_t1_dc_blocker_coeffs_bits(rate_hz: f64) -> Option<(u32, u32)> {
    if !rate_hz.is_finite() || !(44_100.0..=384_000.0).contains(&rate_hz) {
        return None;
    }
    const F64_MANT_BITS_I32: i32 = IEEE754_F64_MANTISSA_BITS as i32;
    let bits = rate_hz.to_bits();
    let exp = (((bits >> IEEE754_F64_MANTISSA_BITS) & u64::from(IEEE754_F64_EXP_ALL_ONES)) as i32)
        - IEEE754_F64_EXP_BIAS;
    let mant =
        (1_u64 << IEEE754_F64_MANTISSA_BITS) | (bits & ((1_u64 << IEEE754_F64_MANTISSA_BITS) - 1));
    let (rate_num, rate_den): (u64, u64) = if exp >= F64_MANT_BITS_I32 {
        (mant << ((exp - F64_MANT_BITS_I32) as u32), 1)
    } else {
        let raw_k = (F64_MANT_BITS_I32 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, 1_u64 << (raw_k - tz))
    };
    let exponent_num = 48_000_u64 * rate_den;
    let ref_mant24 = u64::from(
        IEEE754_F32_IMPLICIT_BIT | (DC_BLOCKER_5HZ_48K_F32.to_bits() & IEEE754_F32_MANTISSA_MASK),
    );
    let ref_bounds = atanh_bounds_160bit(ref_mant24, 1_u64 << IEEE754_F32_SIGNIFICAND_BITS);

    let mut lower: u32 = 0x00FF_D000 + 1;
    let mut upper: u32 = 0x00FF_FBFF;
    while lower < upper {
        let candidate = (lower + upper) >> 1;
        let mid_bounds = atanh_bounds_160bit(
            u64::from(2 * candidate + 1),
            1_u64 << (IEEE754_F32_SIGNIFICAND_BITS + 1),
        );
        let lhs = mid_bounds.0.mul_u64(rate_num);
        let rhs = ref_bounds.1.mul_u64(exponent_num);
        if lhs > rhs {
            lower = candidate + 1;
        } else {
            upper = candidate;
        }
    }

    const F32_HALF_EXP_BITS: u32 = (IEEE754_F32_EXP_BIAS - 1) << IEEE754_F32_MANTISSA_BITS;
    let pole_bits = F32_HALF_EXP_BITS + lower - IEEE754_F32_IMPLICIT_BIT;
    let num_gain = (1_u32 << IEEE754_F32_SIGNIFICAND_BITS) + lower;
    let q = num_gain >> 1;
    let rem = num_gain & 1;
    let rounded_gain = q + u32::from(rem == 1 && (q & 1) != 0);
    let gain_bits = F32_HALF_EXP_BITS + rounded_gain - IEEE754_F32_IMPLICIT_BIT;
    Some((pole_bits, gain_bits))
}

/// Computes `(pole, gain)` as `f32`, clamping finite positive rates to `[44,100.0, 384,000.0]` Hz
/// and falling back to `48,000 Hz` on non-finite / non-positive inputs.
#[cfg(not(target_arch = "spirv"))]
#[must_use]
pub fn compute_t1_dc_blocker_coeffs_clamped(rate_hz: f64) -> (f32, f32) {
    if !rate_hz.is_finite() || rate_hz <= 0.0 {
        return (DC_BLOCKER_5HZ_48K_F32, DC_BLOCKER_5HZ_48K_GAIN_F32);
    }
    let clamped = rate_hz.clamp(44_100.0, 384_000.0);
    let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(clamped).unwrap_or((
        DC_BLOCKER_5HZ_48K_F32.to_bits(),
        DC_BLOCKER_5HZ_48K_GAIN_F32.to_bits(),
    ));
    (f32::from_bits(p_bits), f32::from_bits(g_bits))
}

/// Single upstream `5 Hz` passband-normalized 1-pole DC blocker placed ahead of mono
/// `HilbertH1Kernel` (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1DcBlockerKernel {
    pole: f32,
    gain: f32,
    prev_x: f32,
    prev_y: f32,
}

impl Default for T1DcBlockerKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl T1DcBlockerKernel {
    /// Creates a new [`T1DcBlockerKernel`] configured for `48 kHz` (`DC_BLOCKER_5HZ_48K_F32`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pole: DC_BLOCKER_5HZ_48K_F32,
            gain: DC_BLOCKER_5HZ_48K_GAIN_F32,
            prev_x: 0.0,
            prev_y: 0.0,
        }
    }

    /// Creates a [`T1DcBlockerKernel`] with a rate-adapted ZOH pole clamped to
    /// `[0.0, MAX_PARALRU_R_F32]` and passband-normalized gain `0.5 * (1.0 + pole)`.
    #[inline(always)]
    #[must_use]
    pub fn with_pole(dc_pole: f32) -> Self {
        let (pole, gain) = clamp_t1_pole(dc_pole);
        Self {
            pole,
            gain,
            prev_x: 0.0,
            prev_y: 0.0,
        }
    }

    /// Clears internal delay state `(prev_x, prev_y)` to `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_y = 0.0;
    }

    /// Returns the configured `(pole, alpha_gain)` pair.
    #[inline(always)]
    #[must_use]
    pub const fn params(&self) -> (f32, f32) {
        (self.pole, self.gain)
    }

    /// Advances the `5 Hz` upstream DC blocker by one sample with subnormal flushing and
    /// non-finite scrubbing.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        step_dc_blocker_channel(self.pole, self.gain, &mut self.prev_x, &mut self.prev_y, x)
    }
}

/// Exact `24 B` (`align(4)`) true-stereo upstream `5 Hz` passband-normalized DC blocker conditioning
/// Left and Right inputs ahead of [`super::HilbertH1StereoKernel`]
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1DcBlockerStereo {
    /// Clamped DF-I feedback pole (`0x3F7F_D51F` at `48 kHz`).
    pub pole: f32,
    /// Passband-normalized DF-I input difference gain `0.5 * (1.0 + pole)` (`0x3F7F_EA90` @ 48 k).
    pub gain: f32,
    /// Previous scrubbed stereo input `[x_L[n-1], x_R[n-1]]`.
    pub prev_x: [f32; 2],
    /// Previous flushed stereo output `[y_L[n-1], y_R[n-1]]`.
    pub prev_y: [f32; 2],
}

const _: () = assert!(core::mem::size_of::<T1DcBlockerStereo>() == 24);
const _: () = assert!(core::mem::align_of::<T1DcBlockerStereo>() == 4);

impl Default for T1DcBlockerStereo {
    fn default() -> Self {
        Self::new()
    }
}

impl T1DcBlockerStereo {
    /// Creates a `48 kHz` stereo `5 Hz` `T1` DC blocker with zeroed state.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pole: DC_BLOCKER_5HZ_48K_F32,
            gain: DC_BLOCKER_5HZ_48K_GAIN_F32,
            prev_x: [0.0; 2],
            prev_y: [0.0; 2],
        }
    }

    /// Creates a stereo DC blocker with rate-adapted pole clamped to `[0.0, MAX_PARALRU_R_F32]`.
    #[inline(always)]
    #[must_use]
    pub fn with_pole(dc_pole: f32) -> Self {
        let (pole, gain) = clamp_t1_pole(dc_pole);
        Self {
            pole,
            gain,
            prev_x: [0.0; 2],
            prev_y: [0.0; 2],
        }
    }

    /// Updates `(pole, gain)` on sample-rate changes while preserving delay history.
    #[inline(always)]
    pub fn set_pole(&mut self, dc_pole: f32) {
        let (pole, gain) = clamp_t1_pole(dc_pole);
        self.pole = pole;
        self.gain = gain;
    }

    /// Zeroes stereo `(prev_x, prev_y)` delay lines.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.prev_x = [0.0; 2];
        self.prev_y = [0.0; 2];
    }

    /// Returns `(pole, gain)`.
    #[inline(always)]
    #[must_use]
    pub const fn params(&self) -> (f32, f32) {
        (self.pole, self.gain)
    }

    /// Steps Left and Right `5 Hz` DF-I DC blockers in lockstep.
    #[inline(always)]
    pub fn step_stereo(&mut self, x_l: f32, x_r: f32) -> (f32, f32) {
        let y_l = step_dc_blocker_channel(
            self.pole,
            self.gain,
            &mut self.prev_x[0],
            &mut self.prev_y[0],
            x_l,
        );
        let y_r = step_dc_blocker_channel(
            self.pole,
            self.gain,
            &mut self.prev_x[1],
            &mut self.prev_y[1],
            x_r,
        );
        (y_l, y_r)
    }
}

/// Pure-`f32` negative-real-axis branch-cut detector with near-origin `2^-18` antipodal chord
/// gating (`ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline(always)]
#[must_use]
pub fn tracker_crosses_branch_cut_f32(
    prev_re: f32,
    prev_im: f32,
    cur_re: f32,
    cur_im: f32,
) -> bool {
    let dot = prev_re * cur_re + prev_im * cur_im;
    let det = prev_re * cur_im - prev_im * cur_re;
    if dot < 0.0 && const_abs_f32(det) <= BRANCH_CUT_CHORD_TOL {
        return false;
    }
    (prev_im >= 0.0 && cur_im < 0.0 && det > 0.0) || (prev_im < 0.0 && cur_im >= 0.0 && det < 0.0)
}

/// Evaluates the cancellation-free principal half-angle `(cos(theta / 2), sin(theta / 2))` of a
/// unit phasor `(re, im)` via mandatory 3-step `rsqrt_pos_nr3`
/// (`ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[inline(always)]
fn cancellation_free_half_angle_nr3(re: f32, im: f32) -> (f32, f32) {
    let h = (0.5 * (1.0 + const_abs_f32(re))).max(T1_HALF_ANGLE_EPS);
    let rho_half = GpuVec4::splat(h).rsqrt_pos_nr3().to_array()[0];
    let v_maj = (h * rho_half).clamp(0.0, 1.0);
    let v_min = (const_abs_f32(im) * (0.5 * rho_half)).clamp(0.0, 1.0);
    let (c, abs_s) = if re >= 0.0 {
        (v_maj, v_min)
    } else {
        (v_min, v_maj)
    };
    let sigma_pos = if im >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    (c, sigma_pos * abs_s)
}

/// Exact `28 B` (`align(4)`) pure 32-bit 2-stage Riemann `T1` sub-harmonic rotor operating on
/// `(i_in, q_in)` conditioned by upstream `5 Hz` DC blocker -> `H1`
/// (`ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1SubHarmonicKernel {
    prev_u_re: f32,
    prev_u_im: f32,
    sheet: f32,
    prev_w1_re: f32,
    prev_w1_im: f32,
    sheet2: f32,
    prev_valid: bool,
}

const _: () = assert!(core::mem::size_of::<T1SubHarmonicKernel>() == 28);
const _: () = assert!(core::mem::align_of::<T1SubHarmonicKernel>() == 4);

impl Default for T1SubHarmonicKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl T1SubHarmonicKernel {
    /// Creates a new [`T1SubHarmonicKernel`] in clean startup state (`prev_u = (1.0, 0.0)`,
    /// `prev_w1 = (1.0, 0.0)`, `sheet = +1.0`, `sheet2 = +1.0`, `prev_valid = false`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            sheet: 1.0,
            prev_w1_re: 1.0,
            prev_w1_im: 0.0,
            sheet2: 1.0,
            prev_valid: false,
        }
    }

    /// Resets both Riemann sheets and phase continuity to startup defaults.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the current `(sheet, sheet2, prev_w1_re, prev_valid)` state.
    #[inline(always)]
    #[must_use]
    pub const fn state(&self) -> (f32, f32, f32, bool) {
        (self.sheet, self.sheet2, self.prev_w1_re, self.prev_valid)
    }

    /// Steps the 2-stage `T1` sub-harmonic rotor on upstream `5 Hz` DC-blocked analytic
    /// `(i_in, q_in)`, returning
    /// `[sub1_cos * env_gate, sub1_sin * env_gate, sub2_cos * env_gate, sub2_sin * env_gate]`.
    #[inline(always)]
    pub fn step(&mut self, i_in: f32, q_in: f32) -> [f32; 4] {
        if !i_in.is_finite() || !q_in.is_finite() {
            self.prev_valid = false;
            return [0.0; 4];
        }
        let i_clean = flush_subnormal_f32_bits(i_in);
        let q_clean = flush_subnormal_f32_bits(q_in);
        let r2 = i_clean * i_clean + q_clean * q_clean;
        if !r2.is_finite() || r2 < T1_SILENCE_SQUELCH_R2 {
            self.prev_valid = false;
            return [0.0; 4];
        }

        let inv_r = GpuVec4::splat(r2).rsqrt_pos_nr3().to_array()[0];
        let u_re = (i_clean * inv_r).clamp(-1.0, 1.0);
        let u_im = (q_clean * inv_r).clamp(-1.0, 1.0);
        let env_gate = flush_subnormal_f32_bits(r2 * inv_r);

        // Stage 1: z -> w1 = z^(1/2)
        if self.prev_valid
            && tracker_crosses_branch_cut_f32(self.prev_u_re, self.prev_u_im, u_re, u_im)
        {
            self.sheet = -self.sheet;
        }
        let (c1, s1) = cancellation_free_half_angle_nr3(u_re, u_im);
        let sub1_cos = self.sheet * c1;
        let sub1_sin = self.sheet * s1;

        // Stage 2: w1 -> w2 = w1^(1/2) = z^(1/4)
        if self.prev_valid
            && tracker_crosses_branch_cut_f32(self.prev_w1_re, self.prev_w1_im, sub1_cos, sub1_sin)
        {
            self.sheet2 = -self.sheet2;
        }
        let (c2, s2) = cancellation_free_half_angle_nr3(sub1_cos, sub1_sin);
        let sub2_cos = self.sheet2 * c2;
        let sub2_sin = self.sheet2 * s2;

        self.prev_u_re = u_re;
        self.prev_u_im = u_im;
        self.prev_w1_re = sub1_cos;
        self.prev_w1_im = sub1_sin;
        self.prev_valid = true;

        [
            flush_subnormal_f32_bits(sub1_cos * env_gate) + 0.0,
            flush_subnormal_f32_bits(sub1_sin * env_gate) + 0.0,
            flush_subnormal_f32_bits(sub2_cos * env_gate) + 0.0,
            flush_subnormal_f32_bits(sub2_sin * env_gate) + 0.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modal_and_transport::op_p::hilbert_stereo::HilbertH1Kernel;
    use proptest::prelude::*;

    /// verify[dsp.op-m.tracker.t1-dc-blocker]
    #[test]
    fn test_t1_dc_blocker_5hz_rejects_dc_and_preserves_audio() {
        let mut dcb = T1DcBlockerKernel::default();
        assert_eq!(dcb.params().0.to_bits(), 0x3F7F_D51F);
        assert_eq!(dcb.params().1.to_bits(), 0x3F7F_EA90);

        let mut dc_out = 0.0_f32;
        for _ in 0..24_000 {
            dc_out = dcb.step(1.0);
        }
        assert!(dc_out.abs() < 1.0e-5, "residual DC = {dc_out}");

        dcb.reset();
        let mut peak_1k = 0.0_f32;
        let omega = core::f64::consts::TAU * 1_000.0 / 48_000.0;
        for n in 0..25_000 {
            let x = (0.75 + 0.5 * (omega * (n as f64)).sin()) as f32;
            let y = dcb.step(x);
            if n >= 24_000 {
                peak_1k = peak_1k.max(y.abs());
            }
        }
        assert!((peak_1k - 0.5).abs() < 1.0e-3, "peak_1k = {peak_1k}");

        assert_eq!(
            T1DcBlockerKernel::with_pole(f32::NAN).params().0.to_bits(),
            0x3F7F_D51F
        );
        assert_eq!(
            T1DcBlockerKernel::with_pole(2.0).params().0,
            MAX_PARALRU_R_F32
        );
        assert_eq!(T1DcBlockerKernel::with_pole(-0.5).params(), (0.0, 0.5));
        assert!(dcb.step(f32::NAN).is_finite());
        dcb.reset();
        assert_eq!(dcb.step(f32::NAN), 0.0);
        let _ = dcb.step(-3.0e38);
        assert_eq!(dcb.step(3.0e38), 0.0);
    }

    /// verify[dsp.op-m.tracker.t1-dc-blocker]
    #[test]
    fn test_t1_160bit_vectors_bridged_rate_and_stereo_blocker() {
        let anchors: [(f64, u32, u32); 8] = [
            (44_100.0, 0x3F7F_D155, 0x3F7F_E8AA),
            (48_000.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (48_001.0, 0x3F7F_D51F, 0x3F7F_EA90),
            (88_200.0, 0x3F7F_E8A9, 0x3F7F_F454),
            (96_000.0, 0x3F7F_EA8F, 0x3F7F_F548),
            (176_400.0, 0x3F7F_F454, 0x3F7F_FA2A),
            (192_000.0, 0x3F7F_F547, 0x3F7F_FAA4),
            (384_000.0, 0x3F7F_FAA3, 0x3F7F_FD52),
        ];
        for (rate, exp_pole, exp_gain) in anchors {
            assert_eq!(
                compute_t1_dc_blocker_coeffs_bits(rate),
                Some((exp_pole, exp_gain)),
                "rate {rate}"
            );
        }
        assert_eq!(compute_t1_dc_blocker_coeffs_bits(f64::NAN), None);
        assert_eq!(compute_t1_dc_blocker_coeffs_bits(44_099.9), None);

        // Bridged 128 kHz rate resolves cleanly inside unified bracket without None.
        let bridged = compute_t1_dc_blocker_coeffs_bits(128_000.0).expect("128 kHz bridged");
        assert!(bridged.0 > 0x3F7F_EA8F && bridged.0 < 0x3F7F_F454);

        // Adjacent f64 inputs from op_m_modal_kautz.md round cleanly without naive double-rounding.
        let adj_lo = compute_t1_dc_blocker_coeffs_bits(f64::from_bits(0x40E7_7045_FD95_5545));
        let adj_hi = compute_t1_dc_blocker_coeffs_bits(f64::from_bits(0x40E7_7045_FD95_5546));
        assert!(adj_lo.is_some() && adj_hi.is_some());
        let (p_def, g_def) = compute_t1_dc_blocker_coeffs_clamped(-1.0);
        assert_eq!(
            (p_def.to_bits(), g_def.to_bits()),
            (0x3F7F_D51F, 0x3F7F_EA90)
        );

        // Stereo blocker matches independent mono blockers at 0 ULP.
        let mut st = T1DcBlockerStereo::default();
        let (p96, _) = compute_t1_dc_blocker_coeffs_clamped(96_000.0);
        st.set_pole(p96);
        assert_eq!(st.params().0, p96);
        let mut st2 = T1DcBlockerStereo::with_pole(p96);
        let mut mono_l = T1DcBlockerKernel::with_pole(p96);
        let mut mono_r = T1DcBlockerKernel::with_pole(p96);
        for n in 0..32 {
            let xl = 0.25 + 0.5 * ((0.1 * n as f32).sin());
            let xr = -0.5 + 0.3 * ((0.2 * n as f32).cos());
            let (yl, yr) = st.step_stereo(xl, xr);
            let (yl2, yr2) = st2.step_stereo(xl, xr);
            assert_eq!(yl.to_bits(), mono_l.step(xl).to_bits());
            assert_eq!(yr.to_bits(), mono_r.step(xr).to_bits());
            assert_eq!((yl.to_bits(), yr.to_bits()), (yl2.to_bits(), yr2.to_bits()));
        }
        st.reset();
        assert_eq!(st.prev_x, [0.0; 2]);
        assert_eq!(st.prev_y, [0.0; 2]);
    }

    fn assert_sweep_range(
        t1: &mut T1SubHarmonicKernel,
        amp: f64,
        steps: usize,
        indices: impl Iterator<Item = usize>,
    ) -> u8 {
        let mut prev_feat = [0.0_f32; 4];
        let mut mask_sheets = 0_u8;
        for (idx, n) in indices.enumerate() {
            let theta = core::f64::consts::TAU * (n as f64) / (steps as f64);
            let f = t1.step((amp * theta.cos()) as f32, (amp * theta.sin()) as f32);
            let exp = [
                (amp * (0.5 * theta).cos()) as f32,
                (amp * (0.5 * theta).sin()) as f32,
                (amp * (0.25 * theta).cos()) as f32,
                (amp * (0.25 * theta).sin()) as f32,
            ];
            for k in 0..4 {
                assert!(
                    (f[k] - exp[k]).abs() < 1.0e-5,
                    "n={n} k={k}: {} vs {}",
                    f[k],
                    exp[k]
                );
                if idx > 0 {
                    assert!((f[k] - prev_feat[k]).abs() < 0.04, "jump n={n} k={k}");
                }
            }
            prev_feat = f;
            let (sheet, sheet2, _, valid) = t1.state();
            assert!(valid);
            mask_sheets |= match (sheet > 0.0, sheet2 > 0.0) {
                (true, true) => 1,
                (false, true) => 2,
                (false, false) => 4,
                (true, false) => 8,
            };
        }
        mask_sheets
    }

    /// verify[dsp.op-m.tracker.envelope-norm]
    /// verify[dsp.op-m.tracker.toggle-sheet]
    #[test]
    fn test_t1_subharmonic_kernel_4_features_gates_and_continuity() {
        assert_eq!(core::mem::size_of::<T1SubHarmonicKernel>(), 28);
        assert_eq!(core::mem::align_of::<T1SubHarmonicKernel>(), 4);

        let mut t1 = T1SubHarmonicKernel::default();
        assert_eq!(t1.state(), (1.0, 1.0, 1.0, false));
        for &(si, sq) in &[
            (0.0_f32, 0.0_f32),
            (-0.0, -0.0),
            (1.0e-8, -1.0e-8),
            (f32::NAN, f32::INFINITY),
            (-1.0e20, 0.0),
        ] {
            assert_eq!(t1.step(si, sq).map(f32::to_bits), [0_u32; 4]);
            assert!(!t1.state().3);
        }

        let resume = t1.step(0.5, 0.0);
        let st_resume = t1.state();
        assert_eq!((st_resume.0, st_resume.1, st_resume.3), (1.0, 1.0, true));
        assert!((st_resume.2 - 1.0).abs() < 1.0e-5);
        assert!((resume[0] - 0.5).abs() < 1.0e-5 && resume[1].to_bits() == 0);
        assert!((resume[2] - 0.5).abs() < 1.0e-5 && resume[3].to_bits() == 0);

        t1.reset();
        let at_pi = t1.step(-1.0, 0.0);
        assert_eq!(at_pi[0].to_bits(), 0);
        assert!((at_pi[1] - 1.0).abs() < 1.0e-5);
        assert!((at_pi[2] - core::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-5);
        assert!((at_pi[3] - core::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-5);

        assert!(tracker_crosses_branch_cut_f32(-0.98, 0.2, -0.98, -0.2));
        assert!(tracker_crosses_branch_cut_f32(-0.98, -0.2, -0.98, 0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.98, 0.2, 0.98, -0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.0, 1.0, 0.0, -1.0));
        assert!(!tracker_crosses_branch_cut_f32(-1.0, 1.0e-7, 1.0, -1.0e-7));
        assert!(!tracker_crosses_branch_cut_f32(-1.0, 2.0e-6, 1.0, -1.0e-6));

        // 4 full revolutions (8*pi rad), forward & reverse, aligned (128) and straddling (127) grids.
        let amp = 0.75_f64;
        for &steps in &[128_usize, 127_usize] {
            let total = steps * 4;
            t1.reset();
            assert_eq!(assert_sweep_range(&mut t1, amp, steps, 0..=total), 0b1111);
            t1.reset();
            assert_eq!(
                assert_sweep_range(&mut t1, amp, steps, (0..=total).rev()),
                0b1111
            );
        }

        // Antipodal HYSTERESIS chord gating (dot < 0, |det| <= 2^-18) holds (+1, +1) and (-1, -1).
        t1.reset();
        let _ = t1.step(-1.0, 1.0e-7);
        let _ = t1.step(1.0, -2.0e-7);
        assert_eq!((t1.state().0, t1.state().1), (1.0, 1.0));

        t1.reset();
        let _ = assert_sweep_range(&mut t1, amp, 128, 0..=160);
        let saved = t1.state();
        assert_eq!((saved.0, saved.1, saved.3), (-1.0, -1.0, true));
        assert!(saved.2 < 0.0);

        // SILENCE holds (sheet, sheet2, prev_w1_re), clears prev_valid, emits +0.0, resumes smoothly.
        assert_eq!(t1.step(0.0, 0.0).map(f32::to_bits), [0_u32; 4]);
        assert_eq!(t1.step(1.0e-8, -1.0e-8).map(f32::to_bits), [0_u32; 4]);
        assert_eq!(t1.step(f32::NAN, 0.0).map(f32::to_bits), [0_u32; 4]);
        assert_eq!(t1.state(), (saved.0, saved.1, saved.2, false));
        let _ = assert_sweep_range(&mut t1, amp, 128, 161..=512);

        // Canonical +0.0 bits when sheet == -1.0 multiplies zero cosine, plus (+1, -1) hysteresis.
        t1.reset();
        let _ = assert_sweep_range(&mut t1, amp, 128, 0..=160);
        assert_eq!(t1.step(0.0, 0.0).map(f32::to_bits), [0_u32; 4]);
        let neg_zero_guard = t1.step(-1.0, 0.0);
        assert_eq!(neg_zero_guard[0].to_bits(), 0_u32);
        assert!((neg_zero_guard[1] + 1.0).abs() < 1.0e-5);

        t1.reset();
        let _ = assert_sweep_range(&mut t1, amp, 128, 0..=288);
        assert_eq!((t1.state().0, t1.state().1), (1.0, -1.0));
        let _ = t1.step(-1.0, 1.0e-7);
        let _ = t1.step(1.0, -2.0e-7);
        assert_eq!((t1.state().0, t1.state().1), (1.0, -1.0));

        // Extreme amplitudes [1e-5, 1e15] with rsqrt_pos_nr3 pure 32-bit f32 precision.
        use core::f64::consts::{FRAC_PI_3, FRAC_PI_4};
        for &extreme_amp in &[1.0e-5_f64, 1.0e-3, 1.0, 1.0e6, 1.0e12, 1.0e15] {
            for &theta in &[0.0_f64, FRAC_PI_3, 3.0 * FRAC_PI_4, -2.0 * FRAC_PI_3] {
                t1.reset();
                let f = t1.step(
                    (extreme_amp * theta.cos()) as f32,
                    (extreme_amp * theta.sin()) as f32,
                );
                let exp = [
                    extreme_amp * (0.5 * theta).cos(),
                    extreme_amp * (0.5 * theta).sin(),
                    extreme_amp * (0.25 * theta).cos(),
                    extreme_amp * (0.25 * theta).sin(),
                ];
                for k in 0..4 {
                    assert!((f64::from(f[k]) - exp[k]).abs() <= extreme_amp * 1.0e-5);
                }
            }
        }

        // Verify shift-subtract U256Be::div_u64 across edge cases (zero, shift=64, primes).
        let (q0, r0) = div_rem_u128_by_u64_shift(5, 0);
        assert_eq!((q0, r0), (0, 5));
        let num_edge = (3_u128 << 64) - 1;
        let (q_edge, r_edge) = div_rem_u128_by_u64_shift(num_edge, 3);
        assert_eq!((q_edge, r_edge), ((num_edge / 3) as u64, num_edge % 3));
        let u = U256Be {
            limbs: [2, u64::MAX, 0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210],
        };
        let divided = u.div_u64(3);
        assert_eq!(divided.limbs[0], 0);
        assert!(divided.limbs[1] > (u64::MAX / 3) * 2);
    }

    proptest! {
        #[test]
        fn prop_op_p_three_stage_pipeline_always_bounded_finite(
            samples in prop::collection::vec(-2.0_f32..2.0_f32, 1..64),
        ) {
            let mut dc = T1DcBlockerKernel::new();
            let mut h = HilbertH1Kernel::new();
            let mut t1 = T1SubHarmonicKernel::new();
            for x in samples {
                let x_dc = dc.step(x + 0.25);
                let (yi, yq) = h.step(x_dc);
                prop_assert!(x_dc.is_finite() && yi.is_finite() && yq.is_finite());
                let feat = t1.step(yi, yq);
                let env_ref = f64::from(yi).hypot(f64::from(yq)) as f32;
                for lane in feat {
                    prop_assert!(lane.is_finite());
                    prop_assert!(lane.abs() <= env_ref * 1.01 + 1.0e-5);
                }
                let half_norm = f64::from(feat[0]).hypot(f64::from(feat[1])) as f32;
                let quarter_norm = f64::from(feat[2]).hypot(f64::from(feat[3])) as f32;
                if t1.state().3 {
                    prop_assert!((half_norm - env_ref).abs() <= env_ref * 1.0e-5 + 1.0e-6);
                    prop_assert!((quarter_norm - env_ref).abs() <= env_ref * 1.0e-5 + 1.0e-6);
                } else {
                    prop_assert_eq!(feat.map(f32::to_bits), [0_u32; 4]);
                }
            }
        }

        #[test]
        fn prop_t1_subharmonic_nr3_extreme_amplitude_and_continuity(
            log10_amp in -5.0_f64..=15.0_f64,
            theta0 in -12.0_f64..=12.0_f64,
            dtheta in -0.2_f64..=0.2_f64,
        ) {
            let amp = 10.0_f64.powf(log10_amp);
            let mut t1 = T1SubHarmonicKernel::new();
            let mut prev = [0.0_f32; 4];
            for k in 0..8 {
                let theta = theta0 + (k as f64) * dtheta;
                let yi = (amp * theta.cos()) as f32;
                let yq = (amp * theta.sin()) as f32;
                let feat = t1.step(yi, yq);
                let env_ref = f64::from(yi).hypot(f64::from(yq));
                let half_norm = f64::from(feat[0]).hypot(f64::from(feat[1]));
                let quarter_norm = f64::from(feat[2]).hypot(f64::from(feat[3]));
                prop_assert!(t1.state().3);
                prop_assert!((half_norm - env_ref).abs() <= env_ref * 1.0e-5);
                prop_assert!((quarter_norm - env_ref).abs() <= env_ref * 1.0e-5);

                // Sheet-invariant algebraic square-root reconstruction: w2^2 = env * w1, w1^2 = env * z.
                let w1_sq_re = f64::from(feat[0]) * f64::from(feat[0])
                    - f64::from(feat[1]) * f64::from(feat[1]);
                let w1_sq_im = 2.0 * f64::from(feat[0]) * f64::from(feat[1]);
                let w2_sq_re = f64::from(feat[2]) * f64::from(feat[2])
                    - f64::from(feat[3]) * f64::from(feat[3]);
                let w2_sq_im = 2.0 * f64::from(feat[2]) * f64::from(feat[3]);
                prop_assert!((w1_sq_re - env_ref * f64::from(yi)).abs() <= env_ref * env_ref * 2.0e-5);
                prop_assert!((w1_sq_im - env_ref * f64::from(yq)).abs() <= env_ref * env_ref * 2.0e-5);
                prop_assert!((w2_sq_re - env_ref * f64::from(feat[0])).abs() <= env_ref * env_ref * 2.0e-5);
                prop_assert!((w2_sq_im - env_ref * f64::from(feat[1])).abs() <= env_ref * env_ref * 2.0e-5);

                if k > 0 {
                    let step_bound = amp * (dtheta.abs() * 0.55) + amp * 1.0e-5;
                    for lane in 0..4 {
                        prop_assert!((f64::from(feat[lane]) - f64::from(prev[lane])).abs() <= step_bound);
                    }
                }
                prev = feat;
            }
        }
    }
}
