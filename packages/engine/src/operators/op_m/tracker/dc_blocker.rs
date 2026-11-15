//! `T1` 160-bit midpoint DC blocker coefficient synthesis, passband filter, and SIMD exponent-split scaling.

use crate::math::flush_recursive_state_f32;

/// Outward 160-bit interval bounds on $\mathrm{atanh}((D - N) / (D + N)) \cdot 2^{160}$ using 8 terms
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct U256Be {
    limbs: [u64; 4], // big-endian limbs for natural Ord comparison: limbs[0] = MSW, limbs[3] = LSW
}

impl U256Be {
    const ZERO: Self = Self { limbs: [0; 4] };

    fn add(self, rhs: Self) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in (0..4).rev() {
            let s = (self.limbs[i] as u128) + (rhs.limbs[i] as u128) + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        Self { limbs: out }
    }

    fn add_u64(self, rhs: u64) -> Self {
        self.add(Self {
            limbs: [0, 0, 0, rhs],
        })
    }

    fn mul_u64(self, rhs: u64) -> Self {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for i in (0..4).rev() {
            let p = (self.limbs[i] as u128) * (rhs as u128) + carry;
            out[i] = p as u64;
            carry = p >> 64;
        }
        Self { limbs: out }
    }

    fn div_u64(self, divisor: u64) -> Self {
        let mut out = [0u64; 4];
        let mut rem = 0u128;
        let d = divisor as u128;
        for i in 0..4 {
            let cur = (rem << 64) | (self.limbs[i] as u128);
            out[i] = (cur / d) as u64;
            rem = cur % d;
        }
        Self { limbs: out }
    }
}

fn atanh_bounds_160bit(numerator: u64, denominator: u64) -> (U256Be, U256Be) {
    let z_num = denominator - numerator;
    let z_den = denominator + numerator;
    // Start with (1 << 160) * z_num / z_den: (1 << 160) has bit 32 set in limbs[1]
    let one_shl_160 = U256Be {
        limbs: [0, 1u64 << 32, 0, 0],
    };
    let mut cur = one_shl_160.mul_u64(z_num).div_u64(z_den);
    let z2_num = z_num * z_num;
    let z2_den = z_den * z_den;
    let mut lower = U256Be::ZERO;
    for idx in 0..8u64 {
        lower = lower.add(cur.div_u64(2 * idx + 1));
        cur = cur.mul_u64(z2_num).div_u64(z2_den);
    }
    (lower, lower.add_u64(16))
}

/// Computes the `T1` tracker `5 Hz` DC blocker `(pole_bits, gain_bits)` at `rate_hz` (`44.1..96 kHz`
/// or `176.4..384 kHz`) using the normative 160-bit midpoint binary search without `libm` calls
/// (`ref[dsp.op-m.tracker.t1-dc-blocker]`).
pub fn compute_t1_dc_blocker_coeffs_bits(rate_hz: f64) -> Option<(u32, u32)> {
    if !rate_hz.is_finite() {
        return None;
    }
    let (first, last) = if (44_100.0..=96_000.0).contains(&rate_hz) {
        (0x00ff_d000u32, 0x00ff_ebffu32)
    } else if (176_400.0..=384_000.0).contains(&rate_hz) {
        (0x00ff_f400u32, 0x00ff_fbffu32)
    } else {
        return None;
    };

    // Express rate_hz = rate_num / rate_den in reduced integer form (rate_num < 2^53, rate_den <= 2^37)
    let bits = rate_hz.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    let mant = (1u64 << 52) | (bits & ((1u64 << 52) - 1));
    let (rate_num, rate_den): (u64, u64) = if exp >= 52 {
        (mant << ((exp - 52) as u32), 1)
    } else {
        let raw_k = (52 - exp) as u32;
        let tz = mant.trailing_zeros().min(raw_k);
        (mant >> tz, 1u64 << (raw_k - tz))
    };
    let exponent_num = 48_000u64 * rate_den;
    let ref_bounds = atanh_bounds_160bit(0x00ff_d51f, 1u64 << 24);

    let mut lower = first + 1;
    let mut upper = last;
    while lower < upper {
        let candidate = (lower + upper) / 2;
        let mid_bounds = atanh_bounds_160bit((2 * candidate + 1) as u64, 1u64 << 25);
        let lhs = mid_bounds.0.mul_u64(rate_num);
        let rhs = ref_bounds.1.mul_u64(exponent_num);
        if lhs > rhs {
            lower = candidate + 1;
        } else {
            upper = candidate;
        }
    }
    let pole_bits = 0x3f00_0000 + lower - (1 << 23);
    let num_gain = (1u32 << 24) + lower;
    let q = num_gain / 2;
    let r = num_gain % 2;
    let rounded_gain = q + u32::from(r == 1 && (q & 1) != 0);
    let gain_bits = 0x3f00_0000 + rounded_gain - (1 << 23);
    Some((pole_bits, gain_bits))
}

/// Tracker-only passband-normalized `5 Hz` one-pole DC blocker (`T1`, `ref[dsp.op-m.tracker.t1-dc-blocker]`).
#[derive(Clone, Copy, Debug)]
pub struct T1DcBlocker {
    pub pole: f32,
    pub gain: f32,
    pub prev_x: f32,
    pub prev_d: f32,
}

impl Default for T1DcBlocker {
    fn default() -> Self {
        Self::new_at_rate(48_000.0).unwrap()
    }
}

impl T1DcBlocker {
    pub const fn new_48khz() -> Self {
        Self {
            pole: f32::from_bits(0x3f7f_d51f),
            gain: f32::from_bits(0x3f7f_ea90),
            prev_x: 0.0,
            prev_d: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.prev_x = 0.0;
        self.prev_d = 0.0;
    }

    pub fn new_at_rate(rate_hz: f64) -> Option<Self> {
        let (p_bits, g_bits) = compute_t1_dc_blocker_coeffs_bits(rate_hz)?;
        Some(Self {
            pole: f32::from_bits(p_bits),
            gain: f32::from_bits(g_bits),
            prev_x: 0.0,
            prev_d: 0.0,
        })
    }

    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        let d = self.gain * (x - self.prev_x) + self.pole * self.prev_d;
        self.prev_x = x;
        self.prev_d = flush_recursive_state_f32(d);
        d
    }
}

/// Domain-specific SIMD backend trait for `Op-M` (Symplectic Modal, Hilbert & Sub-Harmonic Tracker).
pub trait OpMBackend {
    /// Evaluates `T1` tracker exponent-split scales `(sd_l, su_l, sd_r, su_r)` for normal positive
    /// maxima `(max_l, max_r)` inside SIMD registers without scalar `fmov` / `movd` transfers.
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32);
}

/// Portable scalar reference implementation of [`OpMBackend::tracker_exp_scales_f32`].
#[inline(always)]
pub fn tracker_exp_scales_scalar_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
    let cl_l = ((max_l.to_bits() >> 23) & 0xff).min(253);
    let cl_r = ((max_r.to_bits() >> 23) & 0xff).min(253);
    (
        f32::from_bits((254 - cl_l) << 23),
        f32::from_bits(cl_l << 23),
        f32::from_bits((254 - cl_r) << 23),
        f32::from_bits(cl_r << 23),
    )
}

impl OpMBackend for crate::math::ScalarBackend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        tracker_exp_scales_scalar_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpMBackend for crate::math::NeonBackend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::aarch64::neon::neon_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpMBackend for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::aarch64::sve2::sve2_tracker_exp_scales_f32(max_l, max_r)
            .unwrap_or_else(|| tracker_exp_scales_scalar_f32(max_l, max_r))
    }
}

#[cfg(target_arch = "aarch64")]
impl OpMBackend for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::aarch64::sve2::sve2_tracker_exp_scales_f32(max_l, max_r).unwrap_or_else(|| {
            crate::math::aarch64::neon::neon_tracker_exp_scales_f32(max_l, max_r)
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl OpMBackend for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::x86::sse2::sse2_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpMBackend for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::x86::sse42::sse42_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpMBackend for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        crate::math::x86::sse42::sse42_tracker_exp_scales_f32(max_l, max_r)
    }
}

#[cfg(target_arch = "wasm32")]
impl OpMBackend for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn tracker_exp_scales_f32(max_l: f32, max_r: f32) -> (f32, f32, f32, f32) {
        #[cfg(target_feature = "simd128")]
        {
            crate::math::wasm::wasm_tracker_exp_scales_f32(max_l, max_r)
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            tracker_exp_scales_scalar_f32(max_l, max_r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;

    #[test]
    fn test_t1_dc_blocker_edges_and_scalar_exp_scales() {
        assert!(compute_t1_dc_blocker_coeffs_bits(f64::NAN).is_none());
        let def = T1DcBlocker::default();
        assert_eq!(def.pole.to_bits(), 0x3f7f_d51f);
        assert_eq!(def.gain.to_bits(), 0x3f7f_ea90);

        let (sd_l, su_l, sd_r, su_r) = tracker_exp_scales_scalar_f32(0.5, 16.0);
        assert_eq!(
            (sd_l, su_l, sd_r, su_r),
            ScalarBackend::tracker_exp_scales_f32(0.5, 16.0)
        );
        assert!((sd_l * su_l - 1.0).abs() < 1e-6);
        assert!((sd_r * su_r - 1.0).abs() < 1e-6);
    }
}
