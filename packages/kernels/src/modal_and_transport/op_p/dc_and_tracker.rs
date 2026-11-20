//! `Op-P` `T1` upstream `5 Hz` passband-normalized DC blocker (mono `T1DcBlockerKernel`, `24 B`
//! stereo `T1DcBlockerStereo`, and unified `[44.1 ..= 384]` kHz 160-bit midpoint coefficient
//! synthesis) plus the 4-feature Riemann sub-harmonic tracker (`T1SubHarmonicKernel`)
//! (`ref[dsp.op-m.tracker.t1-dc-blocker]`, `ref[dsp.op-m.tracker.envelope-norm]`,
//! `ref[dsp.op-m.tracker.toggle-sheet]`).
//!
//! Runtime execution paths are strictly `#![no_std]` 32-bit `f32`; the host/cold 160-bit search
//! (`compute_t1_dc_blocker_coeffs_bits`) is guarded under `#[cfg(not(target_arch = "spirv"))]`.

#[cfg(not(target_arch = "spirv"))]
use crate::simd::IEEE754_F32_EXP_BIAS;
use crate::simd::{
    DC_BLOCKER_5HZ_48K_F32, GpuVec4, KernelVec4, MAX_PARALRU_R_F32, flush_subnormal_f32_bits,
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
const BRANCH_CUT_CHORD_TOL: f32 = f32::from_bits((127 - 18) << 23);

const _: () = assert!(DC_BLOCKER_5HZ_48K_F32.to_bits() == 0x3F7F_D51F);
const _: () = assert!(DC_BLOCKER_5HZ_48K_GAIN_F32.to_bits() == 0x3F7F_EA90);
const _: () = assert!(BRANCH_CUT_CHORD_TOL > 0.0 && BRANCH_CUT_CHORD_TOL < 1.0e-5);

#[inline(always)]
const fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

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
        let d = u128::from(divisor);
        for idx in 0..4 {
            let cur = (rem << 64) | u128::from(self.limbs[idx]);
            out[idx] = (cur / d) as u64;
            rem = cur % d;
        }
        Self { limbs: out }
    }
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
    let bits = rate_hz.to_bits();
    let exp = (((bits >> 52) & 0x7FF) as i32) - 1023;
    let mant = (1_u64 << 52) | (bits & ((1_u64 << 52) - 1));
    let (rate_num, rate_den): (u64, u64) = if exp >= 52 {
        (mant << ((exp - 52) as u32), 1)
    } else {
        let raw_k = (52 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, 1_u64 << (raw_k - tz))
    };
    let exponent_num = 48_000_u64 * rate_den;
    let ref_mant24 = u64::from((1_u32 << 23) | (DC_BLOCKER_5HZ_48K_F32.to_bits() & 0x007F_FFFF));
    let ref_bounds = atanh_bounds_160bit(ref_mant24, 1_u64 << 24);

    let mut lower: u32 = 0x00FF_D000 + 1;
    let mut upper: u32 = 0x00FF_FBFF;
    while lower < upper {
        let candidate = (lower + upper) >> 1;
        let mid_bounds = atanh_bounds_160bit(u64::from(2 * candidate + 1), 1_u64 << 25);
        let lhs = mid_bounds.0.mul_u64(rate_num);
        let rhs = ref_bounds.1.mul_u64(exponent_num);
        if lhs > rhs {
            lower = candidate + 1;
        } else {
            upper = candidate;
        }
    }

    const F32_HALF_EXP_BITS: u32 = (IEEE754_F32_EXP_BIAS - 1) << 23;
    const F32_IMPLICIT_MANT_BIT: u32 = 1_u32 << 23;
    let pole_bits = F32_HALF_EXP_BITS + lower - F32_IMPLICIT_MANT_BIT;
    let num_gain = (1_u32 << 24) + lower;
    let q = num_gain >> 1;
    let rem = num_gain & 1;
    let rounded_gain = q + u32::from(rem == 1 && (q & 1) != 0);
    let gain_bits = F32_HALF_EXP_BITS + rounded_gain - F32_IMPLICIT_MANT_BIT;
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
    if dot < 0.0 && abs_f32(det) <= BRANCH_CUT_CHORD_TOL {
        return false;
    }
    (prev_im >= 0.0 && cur_im < 0.0 && det > 0.0) || (prev_im < 0.0 && cur_im >= 0.0 && det < 0.0)
}

/// Pure 32-bit `T1` sub-harmonic tracker operating on `(i_clean, q_clean)` conditioned by upstream
/// DC blocker -> `H1` (`ref[dsp.op-m.tracker.envelope-norm]`, `ref[dsp.op-m.tracker.toggle-sheet]`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct T1SubHarmonicKernel {
    prev_x: f32,
    prev_u_re: f32,
    prev_u_im: f32,
    prev_q: f32,
    prev_q2: f32,
    sheet: f32,
    prev_valid: bool,
}

impl Default for T1SubHarmonicKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl T1SubHarmonicKernel {
    /// Creates a new [`T1SubHarmonicKernel`] in clean startup state (`prev_q = +1`, `prev_q2 = +1`,
    /// `sheet = +1`, `prev_valid = false`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            prev_x: 0.0,
            prev_u_re: 1.0,
            prev_u_im: 0.0,
            prev_q: 1.0,
            prev_q2: 1.0,
            sheet: 1.0,
            prev_valid: false,
        }
    }

    /// Resets Riemann sheet, `-1/-2` octave divider latches, and phase continuity to startup
    /// defaults.
    #[inline(always)]
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the current `(q1_toggle, q2_toggle, sheet_sign, prev_valid)` state.
    #[inline(always)]
    #[must_use]
    pub const fn state(&self) -> (f32, f32, f32, bool) {
        (self.prev_q, self.prev_q2, self.sheet, self.prev_valid)
    }

    /// Steps the `T1` sub-harmonic tracker on upstream `5 Hz` DC-blocked analytic `(i_in, q_in)`,
    /// returning
    /// `[sub1_cos * env_gate, sub1_sin * env_gate, sub1_sq * env_gate, sub2_sq * env_gate]`.
    #[inline(always)]
    pub fn step(&mut self, i_in: f32, q_in: f32) -> [f32; 4] {
        if !i_in.is_finite() || !q_in.is_finite() {
            self.prev_valid = false;
            self.prev_x = 0.0;
            return [0.0; 4];
        }
        let i_clean = flush_subnormal_f32_bits(i_in);
        let q_clean = flush_subnormal_f32_bits(q_in);
        let r2 = i_clean * i_clean + q_clean * q_clean;
        if !r2.is_finite() || r2 < T1_SILENCE_SQUELCH_R2 {
            self.prev_valid = false;
            self.prev_x = i_clean;
            return [0.0; 4];
        }

        let inv_r = GpuVec4::splat(r2).rsqrt_pos_nr2().to_array()[0];
        let u_re = (i_clean * inv_r).clamp(-1.0, 1.0);
        let u_im = (q_clean * inv_r).clamp(-1.0, 1.0);
        let env_gate = flush_subnormal_f32_bits(r2 * inv_r);

        if self.prev_valid && self.prev_x <= 0.0 && i_clean > 0.0 {
            self.prev_q = -self.prev_q;
            if self.prev_q > 0.0 {
                self.prev_q2 = -self.prev_q2;
            }
        }
        if self.prev_valid
            && tracker_crosses_branch_cut_f32(self.prev_u_re, self.prev_u_im, u_re, u_im)
        {
            self.sheet = -self.sheet;
        }

        let h = (0.5 * (1.0 + abs_f32(u_re))).max(T1_HALF_ANGLE_EPS);
        let rho_half = GpuVec4::splat(h).rsqrt_pos_nr2().to_array()[0];
        let v_maj = (h * rho_half).clamp(0.0, 1.0);
        let v_min = (abs_f32(u_im) * (0.5 * rho_half)).clamp(0.0, 1.0);
        let (c_1, abs_s_1) = if u_re >= 0.0 {
            (v_maj, v_min)
        } else {
            (v_min, v_maj)
        };
        let sigma_pos = if u_im >= 0.0 { 1.0_f32 } else { -1.0_f32 };
        let sub1_cos = self.sheet * c_1;
        let sub1_sin = self.sheet * sigma_pos * abs_s_1;

        self.prev_x = i_clean;
        self.prev_u_re = u_re;
        self.prev_u_im = u_im;
        self.prev_valid = true;

        [
            flush_subnormal_f32_bits(sub1_cos * env_gate),
            flush_subnormal_f32_bits(sub1_sin * env_gate),
            flush_subnormal_f32_bits(self.prev_q * env_gate),
            flush_subnormal_f32_bits(self.prev_q2 * env_gate),
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

    /// verify[dsp.op-m.tracker.envelope-norm]
    /// verify[dsp.op-m.tracker.toggle-sheet]
    #[test]
    fn test_t1_subharmonic_kernel_4_features_gates_and_continuity() {
        let mut t1 = T1SubHarmonicKernel::default();
        assert_eq!(t1.state(), (1.0, 1.0, 1.0, false));
        assert_eq!(t1.step(0.0, 0.0), [0.0; 4]);
        assert_eq!(t1.step(1.0e-8, -1.0e-8), [0.0; 4]);
        assert_eq!(t1.step(f32::NAN, f32::INFINITY), [0.0; 4]);
        assert_eq!(t1.step(-1.0e20, 0.0), [0.0; 4]);
        assert!(!t1.state().3);

        let resume = t1.step(0.5, 0.0);
        assert_eq!(t1.state(), (1.0, 1.0, 1.0, true));
        assert!((resume[0] - 0.5).abs() < 1.0e-3 && resume[1].abs() < 1.0e-6);
        assert!((resume[2] - 0.5).abs() < 1.0e-3 && (resume[3] - 0.5).abs() < 1.0e-3);

        t1.reset();
        let at_pi = t1.step(-1.0, 0.0);
        assert!(at_pi[0].abs() < 1.0e-6 && (at_pi[1] - 1.0).abs() < 2.0e-3);

        assert!(tracker_crosses_branch_cut_f32(-0.98, 0.2, -0.98, -0.2));
        assert!(tracker_crosses_branch_cut_f32(-0.98, -0.2, -0.98, 0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.98, 0.2, 0.98, -0.2));
        assert!(!tracker_crosses_branch_cut_f32(0.0, 1.0, 0.0, -1.0));

        t1.reset();
        let steps = 128;
        let amp = 0.75_f64;
        let (mut saw_neg_sheet, mut saw_neg_q1, mut saw_neg_q2) = (false, false, false);
        for n in 0..(steps * 2) {
            let theta = core::f64::consts::TAU * (n as f64) / (steps as f64);
            let f = t1.step((amp * theta.cos()) as f32, (amp * theta.sin()) as f32);
            let exp_cos = (amp * (0.5 * theta).cos()) as f32;
            let exp_sin = (amp * (0.5 * theta).sin()) as f32;
            assert!((f[0] - exp_cos).abs() < 2.0e-3);
            assert!((f[1] - exp_sin).abs() < 2.0e-3);
            assert!((f[2].abs() - (amp as f32)).abs() < 2.0e-3);
            assert!((f[3].abs() - (amp as f32)).abs() < 2.0e-3);
            let (q1, q2, sheet, valid) = t1.state();
            assert!(valid);
            saw_neg_sheet |= sheet < 0.0;
            saw_neg_q1 |= q1 < 0.0;
            saw_neg_q2 |= q2 < 0.0;
        }
        assert!(saw_neg_sheet && saw_neg_q1 && saw_neg_q2);
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
                if feat[2] != 0.0 {
                    prop_assert!((half_norm - env_ref).abs() <= env_ref * 1.0e-2 + 1.0e-5);
                }
            }
        }
    }
}
