//! Op-N Fixed-Hertz `Q64.32` ZOH Bitcrusher & Division-Free Quantizer (`Family 5`,
//! `ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`,
//! `ref[dsp.op-n.bitcrusher.rate-transfer]`).

/// Fixed-Hertz `Q64.32` Integer ZOH Sample-Rate Reducer & Bitcrusher (`Family 5`,
/// `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`,
/// `ref[dsp.op-n.bitcrusher.quantizer]`).
#[derive(Clone, Copy, Debug)]
pub struct ZohBitcrusher {
    pub step_size: f32,
    pub step_inv: f32,
    pub phase: u64,
    pub increment: u64,
    pub denominator: u64,
    pub reciprocal_r: u64,
    pub bit_depth: u8,
    pub target_fs_hz: f32,
    pub held_sample: f32,
}

impl Default for ZohBitcrusher {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn bitcrusher_scale_and_step(bits: u32) -> (f32, f32) {
    let b = bits.clamp(2, 16);
    // Exact IEEE-754 powers of two: scale = 2^(b-1), step_size = 2^-(b-1) (0 FDIV).
    let scale = f32::from_bits((126 + b) << 23);
    let step_size = f32::from_bits((128 - b) << 23);
    (scale, step_size)
}

#[inline(always)]
fn step_q64_32_bitcrusher(
    phase: &mut u64,
    increment: u64,
    denominator: u64,
    held_sample: &mut f32,
    bit_depth: u32,
    x: f32,
) -> f32 {
    if increment == 0 {
        return *held_sample;
    }
    let latch = if increment >= denominator {
        true
    } else {
        let (next_phase, l) = hertz_phase_step(*phase, increment, denominator);
        *phase = next_phase;
        l
    };
    if latch {
        *held_sample = quantize_bitcrusher_nearest_ties_pos_inf(x, bit_depth);
    }
    *held_sample
}

fn rescale_q64_32_sample_rate(
    phase: &mut u64,
    increment: &mut u64,
    denominator: &mut u64,
    reciprocal_r: &mut u64,
    target_hz: f64,
    sample_rate_hz: f64,
) -> Option<()> {
    let eff_target = target_hz.min(sample_rate_hz);
    let (new_inc, new_den, new_rec) = hertz_phase_parameters(eff_target, sample_rate_hz)?;
    *phase = rescale_bitcrusher_phase_u128(*phase, *denominator, new_den, *reciprocal_r);
    *increment = new_inc;
    *denominator = new_den;
    *reciprocal_r = new_rec;
    Some(())
}

impl ZohBitcrusher {
    pub const fn new() -> Self {
        let den = 48_000u64 << 32;
        Self {
            step_size: f32::from_bits((128 - 16) << 23),
            step_inv: f32::from_bits((126 + 16) << 23),
            phase: 0,
            increment: den,
            denominator: den,
            reciprocal_r: ((1u128 << 104) / (den as u128)) as u64,
            bit_depth: 16,
            target_fs_hz: 48_000.0,
            held_sample: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.phase = 0;
        self.held_sample = 0.0;
    }

    /// Sets bit depth `bits in [2, 16]` and downsampled sample rate `target_fs_hz` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
    pub fn set_parameters(&mut self, bits: u32, target_fs_hz: f32, sample_rate_hz: f32) {
        let b = bits.clamp(2, 16);
        let (scale, step_size) = bitcrusher_scale_and_step(b);
        self.bit_depth = b as u8;
        self.step_inv = scale;
        self.step_size = step_size;
        self.target_fs_hz = target_fs_hz.max(0.0);

        let sr = sample_rate_hz.max(1000.0) as f64;
        if let Some((inc, den, rec)) =
            hertz_phase_parameters(self.target_fs_hz.min(sample_rate_hz) as f64, sr)
        {
            self.increment = inc;
            self.denominator = den;
            self.reciprocal_r = rec;
            if self.phase >= self.denominator {
                self.phase %= self.denominator;
            }
        }
    }

    /// Transfers the `Q64.32` phase accumulator across a host sample-rate change without `u128` division (`ref[dsp.op-n.bitcrusher.rate-transfer]`).
    pub fn reconfigure_sample_rate(&mut self, new_sample_rate_hz: f32) {
        let sr = new_sample_rate_hz.max(1000.0) as f64;
        let _ = rescale_q64_32_sample_rate(
            &mut self.phase,
            &mut self.increment,
            &mut self.denominator,
            &mut self.reciprocal_r,
            f64::from(self.target_fs_hz),
            sr,
        );
    }

    /// Steps the ZOH bitcrusher with zero audio-rate divisions (`ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.quantizer]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        step_q64_32_bitcrusher(
            &mut self.phase,
            self.increment,
            self.denominator,
            &mut self.held_sample,
            u32::from(self.bit_depth),
            x,
        )
    }
}

/// Evaluates the uniform nearest-step quantizer $Q_B(x) = \lfloor x 2^{B-1} + 0.5 \rfloor 2^{-(B-1)}$
/// (halfway ties toward $+\infty$) for $B \in [2, 16]$ using floor-residual `i32` truncation
/// (`0 FDIV`, `ref[dsp.op-n.bitcrusher.quantizer]`).
#[inline]
pub fn quantize_bitcrusher_nearest_ties_pos_inf(x: f32, bits: u32) -> f32 {
    let (scale, step_size) = bitcrusher_scale_and_step(bits);
    if !x.is_finite() || x.abs() >= 8_388_608.0 * step_size {
        return x;
    }
    let v = x * scale;
    let t = v as i32;
    let k = t - i32::from(v < (t as f32));
    let frac = v - (k as f32);
    let q = k + i32::from(frac >= 0.5);
    (q as f32) * step_size
}

/// Pure `#![no_std]` non-negative round-ties-to-even for `f64`.
#[inline(always)]
fn round_ties_even_nonneg_f64(x: f64) -> f64 {
    if !x.is_finite() || x >= 4_503_599_627_370_496.0 {
        return x;
    }
    let floored = crate::math::floor_f64(x);
    let frac = x - floored;
    if frac > 0.5 || (frac == 0.5 && ((floored as u64) & 1) != 0) {
        floored + 1.0
    } else {
        floored
    }
}

/// Computes the fixed-hertz integer ZOH phase parameters `(increment_i, denominator_d, reciprocal_r)`
/// at $F = 2^{32}\text{ units/Hz}$ with $R = \lfloor 2^{104} / D \rfloor$ (`u64`,
/// `ref[dsp.op-n.bitcrusher.zoh-phase]`, `ref[dsp.op-n.bitcrusher.rate-transfer]`).
pub fn hertz_phase_parameters(target_hz: f64, sample_rate_hz: f64) -> Option<(u64, u64, u64)> {
    if !target_hz.is_finite()
        || !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !(0.0..=sample_rate_hz).contains(&target_hz)
    {
        return None;
    }
    let scale = (1u64 << 32) as f64;
    let d_f64 = round_ties_even_nonneg_f64(sample_rate_hz * scale);
    let i_f64 = round_ties_even_nonneg_f64(target_hz * scale);
    if d_f64 <= 0.0 || d_f64 >= ((1u64 << 63) as f64) {
        return None;
    }
    let d = d_f64 as u64;
    let i = i_f64 as u64;
    if (target_hz > 0.0 && i == 0) || i > d {
        return None;
    }
    let r_u128 = (1u128 << 104) / (d as u128);
    let r = u64::try_from(r_u128).unwrap_or(u64::MAX);
    Some((i, d, r))
}

/// Advances the fixed-hertz `u64` phase accumulator by 1 sample: `A = P + I`, `b = A >= D`,
/// `P <- A - b * D` (`ref[dsp.op-n.bitcrusher.zoh-phase]`).
#[inline]
pub const fn hertz_phase_step(phase: u64, increment: u64, denominator: u64) -> (u64, bool) {
    let advanced = phase.wrapping_add(increment);
    let latch = advanced >= denominator;
    let next_phase = if latch {
        advanced - denominator
    } else {
        advanced
    };
    (next_phase, latch)
}

/// Transfers a fixed-hertz `u64` phase `phase_old` from `old_denominator` to `new_denominator`
/// without hardware division using precomputed $R = \lfloor 2^{104} / D_{\text{old}} \rfloor$
/// (`ref[dsp.op-n.bitcrusher.rate-transfer]`).
#[inline]
pub const fn rescale_bitcrusher_phase_u128(
    phase_old: u64,
    old_denominator: u64,
    new_denominator: u64,
    reciprocal_r: u64,
) -> u64 {
    let numerator = (phase_old as u128) * (new_denominator as u128);
    let n_lo = (numerator & (u64::MAX as u128)) as u64;
    let n_hi = (numerator >> 64) as u64;
    let a = (n_lo as u128) * (reciprocal_r as u128);
    let c = (n_hi as u128) * (reciprocal_r as u128);
    let m = (a >> 64) + (c & (u64::MAX as u128));
    let h = (c >> 64) + (m >> 64);
    let mut q0 = ((h as u64) << 24) | (((m & (u64::MAX as u128)) as u64) >> 40);
    let mut r0 = numerator - (q0 as u128) * (old_denominator as u128);
    let d_old_128 = old_denominator as u128;
    if r0 >= d_old_128 {
        q0 += 1;
        r0 -= d_old_128;
    }
    if 2 * r0 > d_old_128 || (2 * r0 == d_old_128 && (q0 & 1) != 0) {
        q0 += 1;
    }
    let max_phase = new_denominator.saturating_sub(1);
    if q0 > max_phase { max_phase } else { q0 }
}

/// Fixed-hertz `u64`-phase ZOH Sample-Rate Reducer & Bitcrusher (`Family 5`,
/// `ref[dsp.op-n.bitcrusher.quantizer]`, `ref[dsp.op-n.bitcrusher.zoh-phase]`,
/// `ref[dsp.op-n.bitcrusher.rate-transfer]`).
#[derive(Clone, Copy, Debug)]
pub struct FixedHzBitcrusher {
    pub bit_depth: u32,
    pub target_hz: f64,
    pub phase: u64,
    pub increment: u64,
    pub denominator: u64,
    pub reciprocal_r: u64,
    pub held_sample: f32,
}

impl Default for FixedHzBitcrusher {
    fn default() -> Self {
        Self::new(16, 48_000.0, 48_000.0).unwrap()
    }
}

impl FixedHzBitcrusher {
    pub fn new(bit_depth: u32, target_hz: f64, sample_rate_hz: f64) -> Option<Self> {
        let (inc, den, rec) = hertz_phase_parameters(target_hz, sample_rate_hz)?;
        Some(Self {
            bit_depth: bit_depth.clamp(2, 16),
            target_hz,
            phase: 0,
            increment: inc,
            denominator: den,
            reciprocal_r: rec,
            held_sample: 0.0,
        })
    }

    pub fn set_sample_rate(&mut self, new_sample_rate_hz: f64) -> Option<()> {
        rescale_q64_32_sample_rate(
            &mut self.phase,
            &mut self.increment,
            &mut self.denominator,
            &mut self.reciprocal_r,
            self.target_hz,
            new_sample_rate_hz,
        )
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        step_q64_32_bitcrusher(
            &mut self.phase,
            self.increment,
            self.denominator,
            &mut self.held_sample,
            self.bit_depth,
            x,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use proptest::prelude::*;

    /// verify[dsp.op-n.bitcrusher.zoh-phase]
    #[test]
    fn test_zoh_bitcrusher() {
        let _audio = FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        cr.reset();
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert!((cr.step_size - 0.125).abs() < 1e-6);
        assert_eq!(cr.increment * 4, cr.denominator);

        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.step(0.5), 0.0);
        let latched = cr.step(0.5);
        let expected_q = (0.5 * 8.0_f32 + 0.5).floor() * 0.125;
        assert!((latched - expected_q).abs() < 1e-6);
        assert_eq!(cr.step(0.9), latched);

        cr.reset();
        assert_eq!(cr.phase, 0);
        assert_eq!(cr.held_sample, 0.0);
    }

    #[test]
    fn test_zoh_bitcrusher_full_and_zero_rate() {
        let _audio = FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        cr.set_parameters(4, 48_000.0, 48_000.0);
        let den = cr.denominator;
        for phase in [0_u64, den / 10, den / 4, den - 1] {
            cr.phase = phase;
            for sample in [0.125, -0.25, 0.5, -0.75] {
                assert_eq!(cr.step(sample), sample);
                assert_eq!(cr.phase, phase);
            }
        }

        cr.phase = den / 4;
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.phase, den / 4);
        cr.set_parameters(4, 12_000.0, 48_000.0);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), -0.75);
        assert_eq!(cr.step(0.5), 0.5);

        cr.reset();
        cr.set_parameters(4, 0.0, 48_000.0);
        assert_eq!(cr.step(0.5), 0.0);
        assert_eq!(cr.phase, 0);
    }

    /// verify[dsp.op-n.bitcrusher.quantizer]
    #[test]
    fn test_zoh_bitcrusher_quantization_boundaries() {
        let _audio = FtzDazGuard::new();
        let mut cr = ZohBitcrusher::default();
        for bits in 2..=16 {
            cr.set_parameters(bits, 48_000.0, 48_000.0);
            let scale = f64::from(cr.step_inv);
            for midpoint in [-1.5_f32, -0.5, 0.5, 1.5] {
                let encoded = midpoint.to_bits();
                for scaled in [
                    f32::from_bits(encoded - 1),
                    midpoint,
                    f32::from_bits(encoded + 1),
                ] {
                    let input = scaled * cr.step_size;
                    let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
                    assert_eq!(cr.step(input), expected, "bits={bits}, input={input}");
                }
            }

            for input in [-100.0, 100.0, -f32::MAX, f32::MAX] {
                assert_eq!(cr.step(input), input, "bits={bits}, input={input}");
            }

            let threshold = 8_388_608.0 * cr.step_size;
            for value in [
                f32::from_bits(threshold.to_bits() - 1),
                threshold,
                f32::from_bits(threshold.to_bits() + 1),
            ] {
                for input in [value, -value] {
                    let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
                    assert_eq!(cr.step(input), expected, "bits={bits}, input={input}");
                }
            }
        }
    }

    /// verify[dsp.op-n.bitcrusher.rate-transfer]
    #[test]
    fn test_fixed_hz_bitcrusher_rate_transfer() {
        let _guard = FtzDazGuard::new();

        for bits in 2..=16u32 {
            let scale = (1u32 << (bits - 1)) as f64;
            for mid in [-1.5_f32, -0.5, 0.5, 1.5] {
                let x = mid / (scale as f32);
                let expected = ((f64::from(x) * scale + 0.5).floor() / scale) as f32;
                assert_eq!(quantize_bitcrusher_nearest_ties_pos_inf(x, bits), expected);
            }
        }

        for &(target_hz, period) in &[(16_000.0_f64, 3usize), (12_000.0, 4), (48_000.0, 1)] {
            let (inc, den, _) = hertz_phase_parameters(target_hz, 48_000.0).unwrap();
            let mut p = 0u64;
            for sample in 1..=(period * 8) {
                let (next_p, latch) = hertz_phase_step(p, inc, den);
                p = next_p;
                assert_eq!(latch, sample % period == 0);
            }
            assert_eq!(p, 0);
        }

        let mut cr = FixedHzBitcrusher::new(4, 12_000.0, 48_000.0).unwrap();
        for _ in 0..3 {
            assert_eq!(cr.step(0.5), 0.0);
        }
        assert_eq!(cr.step(0.5), 0.5);
        let saved_phase = cr.phase;
        cr.increment = 0;
        assert_eq!(cr.step(0.875), 0.5);
        assert_eq!(cr.phase, saved_phase);

        let mut fhz = FixedHzBitcrusher::default();
        assert!(fhz.set_sample_rate(96_000.0).is_some());
        let _ = fhz.step(0.25);
        assert_eq!(fhz.step(0.25), 0.25);
        assert!(FixedHzBitcrusher::new(8, -1.0, 48_000.0).is_none());
        assert!(hertz_phase_parameters(1.0, 1e12).is_none());
        assert!(hertz_phase_parameters(1e-30, 48_000.0).is_none());

        let rates = [44_100.0_f64, 48_000.0, 96_000.0, 192_000.0, 384_000.0];
        for &old_r in &rates {
            let (_, d_old, r_recip) = hertz_phase_parameters(12_000.0, old_r).unwrap();
            for &new_r in &rates {
                let (_, d_new, _) = hertz_phase_parameters(12_000.0, new_r).unwrap();
                for p_old in [0u64, 1, d_old / 3, d_old / 2, d_old - 1] {
                    let actual = rescale_bitcrusher_phase_u128(p_old, d_old, d_new, r_recip);
                    let num = (p_old as u128) * (d_new as u128);
                    let den = d_old as u128;
                    let mut q = (num / den) as u64;
                    let rem = num % den;
                    if 2 * rem > den || (2 * rem == den && (q & 1) != 0) {
                        q += 1;
                    }
                    let expected = q.min(d_new - 1);
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_zoh_bitcrusher_quantization_against_f64(
            bits in 2u32..=16,
            input in -100.0_f32..100.0,
        ) {
            let _audio = FtzDazGuard::new();
            let mut cr = ZohBitcrusher::default();
            cr.set_parameters(bits, 48_000.0, 48_000.0);
            let scale = f64::from(cr.step_inv);
            let expected = ((f64::from(input) * scale + 0.5).floor() / scale) as f32;
            prop_assert_eq!(cr.step(input), expected);
        }

        #[test]
        fn prop_bitcrusher_quantizer_and_rate_transfer(
            bits in 2u32..=16u32,
            sample_x in -100.0_f32..100.0_f32,
            old_sr in 44_100.0_f64..=96_000.0_f64,
            new_sr in 44_100.0_f64..=96_000.0_f64,
            frac in 0.0_f64..0.999_f64,
        ) {
            let q = quantize_bitcrusher_nearest_ties_pos_inf(sample_x, bits);
            let step = f32::from_bits((128 - bits) << 23);
            prop_assert!((q - sample_x).abs() <= 0.5 * step + 1e-5);

            let (_, d_old, r_recip) = hertz_phase_parameters(12_000.0, old_sr).unwrap();
            let (_, d_new, _) = hertz_phase_parameters(12_000.0, new_sr).unwrap();
            let p_old = ((d_old as f64) * frac) as u64;
            let p_new = rescale_bitcrusher_phase_u128(p_old, d_old, d_new, r_recip);
            prop_assert!(p_new < d_new);
        }
    }
}
