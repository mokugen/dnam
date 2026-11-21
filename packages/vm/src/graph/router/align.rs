//! Fixed-ring `!alloc` parallel-path sub-sample alignment line (`ParallelAlignLine`) with cold
//! normalized 28-term `Kaiser I0` windowed `Sinc` (`FullBand64Tap`, `LowLatency16Tap`) and
//! centered `4`-tap Catmull-Rom cubic Hermite (`MinLatency4Tap`) synthesis
//! (`0 FDIV`, `0 libm`, `ref[dsp.router.parallel-alignment]`).

#![forbid(unsafe_code)]

use crate::math::{
    const_abs_f64, const_inv_factorial_f64, floor_f64, flush_recursive_state_f32, rcp_nr6_f64,
    rcp_signed_f64, round_half_away_f64, sqrt_f64,
};

/// Fixed power-of-two ring capacity for [`ParallelAlignLine`] (`256` frames).
pub const ALIGN_RING_CAPACITY: usize = 256;
/// Bitmask for branchless `&` ring wrap (`255`).
pub const ALIGN_RING_MASK: usize = ALIGN_RING_CAPACITY - 1;

const KAISER_BETA_64: f64 = 8.6;
const KAISER_BETA_16: f64 = 6.5;

const fn build_inv_m_sq_table() -> [f64; 29] {
    let mut out = [0.0_f64; 29];
    let mut m = 1_usize;
    while m <= 28 {
        out[m] = 1.0 / ((m * m) as f64);
        m += 1;
    }
    out
}

const INV_M_SQ: [f64; 29] = build_inv_m_sq_table();
const INV_FACT_3: f64 = const_inv_factorial_f64(3);
const INV_FACT_5: f64 = const_inv_factorial_f64(5);
const INV_FACT_7: f64 = const_inv_factorial_f64(7);
const INV_FACT_9: f64 = const_inv_factorial_f64(9);
const INV_FACT_11: f64 = const_inv_factorial_f64(11);

const _: () = assert!(ALIGN_RING_CAPACITY.is_power_of_two());
const _: () = assert!(ALIGN_RING_MASK == 255);
const _: () = assert!(INV_M_SQ[1] == 1.0 && INV_M_SQ[2] == 0.25);

/// Parallel-path sub-sample alignment interpolation mode (`ref[dsp.router.parallel-alignment]`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoAlignmentMode {
    /// 64-tap zero-ended Kaiser windowed sinc (`beta = 8.6`, center offset `G = 31`).
    FullBand64Tap = 0,
    /// 16-tap zero-ended Kaiser windowed sinc (`beta = 6.5`, center offset `G = 7`).
    LowLatency16Tap = 1,
    /// Centered 4-tap Catmull-Rom cubic Hermite (`G = 1`).
    MinLatency4Tap = 2,
}

impl IoAlignmentMode {
    /// Returns the host-rate center offset `G` in samples (`31`, `7`, or `1`).
    #[inline]
    pub const fn center_offset(self) -> usize {
        match self {
            Self::FullBand64Tap => 31,
            Self::LowLatency16Tap => 7,
            Self::MinLatency4Tap => 1,
        }
    }

    /// Returns the FIR tap count `K` (`64`, `16`, or `4`).
    #[inline]
    pub const fn tap_count(self) -> usize {
        match self {
            Self::FullBand64Tap => 64,
            Self::LowLatency16Tap => 16,
            Self::MinLatency4Tap => 4,
        }
    }

    /// Returns the Kaiser window parameter `beta` (`8.6`, `6.5`, or `0.0`).
    #[inline]
    pub const fn kaiser_beta(self) -> f64 {
        match self {
            Self::FullBand64Tap => KAISER_BETA_64,
            Self::LowLatency16Tap => KAISER_BETA_16,
            Self::MinLatency4Tap => 0.0,
        }
    }
}

#[inline]
fn kaiser_i0_f64(z: f64) -> f64 {
    let h2 = 0.25 * z * z;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for &inv_m2 in &INV_M_SQ[1..=28] {
        term *= h2 * inv_m2;
        sum += term;
    }
    sum
}

#[inline]
fn normalized_sinc_f64(x: f64) -> f64 {
    if const_abs_f64(x) < 1.0e-12 {
        return 1.0;
    }
    let k = round_half_away_f64(x) as i64;
    let u = core::f64::consts::PI * (x - (k as f64));
    let u2 = u * u;
    let sin_u = u
        * (1.0
            - u2 * (INV_FACT_3
                - u2 * (INV_FACT_5 - u2 * (INV_FACT_7 - u2 * (INV_FACT_9 - u2 * INV_FACT_11)))));
    let signed_sin = if (k & 1) == 0 { sin_u } else { -sin_u };
    signed_sin * rcp_signed_f64(core::f64::consts::PI * x)
}

fn synthesize_kaiser_sinc_taps(tap_count: usize, beta: f64, mu: f64, out: &mut [f32; 64]) {
    let g = ((tap_count - 2) / 2) as f64;
    let inv_r = 2.0 * rcp_nr6_f64(tap_count as f64);
    let inv_denom = rcp_nr6_f64((kaiser_i0_f64(beta) - 1.0).max(1.0e-15));
    let mut tmp = [0.0_f64; 64];
    let mut sum = 0.0_f64;
    for (k, slot) in tmp[..tap_count].iter_mut().enumerate() {
        let s = (k as f64) - g - mu;
        let ratio = s * inv_r;
        let w = if const_abs_f64(ratio) >= 1.0 {
            0.0
        } else {
            let arg = beta * sqrt_f64((1.0 - ratio * ratio).max(0.0));
            ((kaiser_i0_f64(arg) - 1.0) * inv_denom).max(0.0)
        };
        let h = normalized_sinc_f64(s) * w;
        *slot = h;
        sum += h;
    }
    out.fill(0.0);
    if const_abs_f64(sum) <= 1.0e-15 {
        out[(tap_count - 2) / 2] = 1.0;
        return;
    }
    let inv_sum = rcp_signed_f64(sum);
    for (dst, &src) in out[..tap_count].iter_mut().zip(tmp[..tap_count].iter()) {
        *dst = (src * inv_sum) as f32;
    }
}

fn synthesize_catmull_rom_taps(mu: f64, out: &mut [f32; 64]) {
    let one_minus_mu = 1.0 - mu;
    let mu2 = mu * mu;
    let c_m1 = -0.5 * mu * one_minus_mu * one_minus_mu;
    let c_0 = 1.0 - 0.5 * mu2 * (5.0 - 3.0 * mu);
    let c_1 = 0.5 * mu * (1.0 + 4.0 * mu - 3.0 * mu2);
    let c_2 = 0.5 * mu2 * (mu - 1.0);
    out.fill(0.0);
    out[0] = c_m1 as f32;
    out[1] = c_0 as f32;
    out[2] = c_1 as f32;
    out[3] = c_2 as f32;
}

/// Fixed-storage `!alloc` sub-sample alignment FIR line (`256`-sample power-of-two ring).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParallelAlignLine {
    ring: [f32; ALIGN_RING_CAPACITY],
    taps: [f32; 64],
    write_idx: usize,
    base_lag: usize,
    tap_count: usize,
    mode: IoAlignmentMode,
    effective_delay_samples: f64,
}

impl Default for ParallelAlignLine {
    fn default() -> Self {
        Self::new(IoAlignmentMode::FullBand64Tap, 0.0, 1)
    }
}

impl ParallelAlignLine {
    /// Creates a zeroed [`ParallelAlignLine`] for `mode`, fractional offset `delay_samples >= 0.0`,
    /// and oversampling factor `m_os in {1, 4}`.
    pub fn new(mode: IoAlignmentMode, delay_samples: f64, m_os: usize) -> Self {
        let mut line = Self {
            ring: [0.0; ALIGN_RING_CAPACITY],
            taps: [0.0; 64],
            write_idx: 0,
            base_lag: 0,
            tap_count: mode.tap_count(),
            mode,
            effective_delay_samples: 0.0,
        };
        line.reconfigure(mode, delay_samples, m_os);
        line
    }

    /// Recomputes cold Kaiser-sinc or Catmull-Rom tap weights while preserving ring history.
    pub fn reconfigure(&mut self, mode: IoAlignmentMode, delay_samples: f64, m_os: usize) {
        let m = if m_os >= 4 { 4 } else { 1 };
        let g = mode.center_offset();
        let tap_count = mode.tap_count();
        let b_base = (m - 1) * g;
        let max_j = ALIGN_RING_MASK.saturating_sub(b_base + tap_count - 1);
        let delta = if delay_samples.is_finite() {
            delay_samples.clamp(0.0, max_j as f64)
        } else {
            0.0
        };
        let mut j_int = floor_f64(delta) as usize;
        let mut mu = delta - (j_int as f64);
        if mu >= 1.0 && j_int < max_j {
            j_int += 1;
            mu = 0.0;
        } else {
            mu = mu.clamp(0.0, 1.0 - 1.0e-12);
            if j_int >= max_j && delta >= (max_j as f64) {
                j_int = max_j;
                mu = 0.0;
            }
        }

        self.mode = mode;
        self.tap_count = tap_count;
        self.base_lag = b_base + j_int;
        self.effective_delay_samples = (m * g) as f64 + delta;

        match mode {
            IoAlignmentMode::FullBand64Tap | IoAlignmentMode::LowLatency16Tap => {
                synthesize_kaiser_sinc_taps(tap_count, mode.kaiser_beta(), mu, &mut self.taps);
            }
            IoAlignmentMode::MinLatency4Tap => {
                synthesize_catmull_rom_taps(mu, &mut self.taps);
            }
        }
    }

    /// Clears the `256`-sample ring buffer and resets the write cursor to `0`.
    #[inline]
    pub fn reset(&mut self) {
        self.ring = [0.0; ALIGN_RING_CAPACITY];
        self.write_idx = 0;
    }

    /// Pushes one sample `x` and evaluates the aligned output (`0 FDIV`, `0 alloc`, `& 255` wrap).
    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        self.ring[self.write_idx] = flush_recursive_state_f32(x);
        let base = self.write_idx.wrapping_sub(self.base_lag);
        let mut acc = 0.0_f32;
        for k in 0..self.tap_count {
            let idx = base.wrapping_sub(k) & ALIGN_RING_MASK;
            acc += self.taps[k] * self.ring[idx];
        }
        self.write_idx = (self.write_idx + 1) & ALIGN_RING_MASK;
        acc
    }

    /// Returns the configured [`IoAlignmentMode`].
    #[inline]
    pub const fn mode(&self) -> IoAlignmentMode {
        self.mode
    }

    /// Returns the effective delay `M_os * G + Delta` in effective-rate samples.
    #[inline]
    pub const fn effective_delay_samples(&self) -> f64 {
        self.effective_delay_samples
    }

    /// Returns the active tap count (`64`, `16`, or `4`).
    #[inline]
    pub const fn tap_count(&self) -> usize {
        self.tap_count
    }

    /// Returns a slice view over the active FIR taps.
    #[inline]
    pub fn taps(&self) -> &[f32] {
        &self.taps[..self.tap_count]
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_kaiser_sinc_and_catmull_rom_exact_properties() {
        let mut line64 = ParallelAlignLine::new(IoAlignmentMode::FullBand64Tap, 0.0, 4);
        assert_eq!(line64.tap_count(), 64);
        assert_eq!(
            line64.effective_delay_samples().to_bits(),
            124.0_f64.to_bits()
        );
        assert_eq!(line64.step(0.0).to_bits(), 0);
        let tap_sum64: f32 = line64.taps().iter().sum();
        assert!((tap_sum64 - 1.0).abs() < 1.0e-6);

        line64.reset();
        let mut impulse_peak = 0usize;
        for n in 0..160 {
            let y = line64.step(if n == 0 { 1.0 } else { 0.0 });
            if (y - 1.0).abs() < 1.0e-6 {
                impulse_peak = n;
            }
        }
        assert_eq!(impulse_peak, 124);

        let mut line16 = ParallelAlignLine::new(IoAlignmentMode::LowLatency16Tap, 2.35, 1);
        assert_eq!(line16.tap_count(), 16);
        assert!((line16.effective_delay_samples() - 9.35).abs() < 1.0e-12);
        let tap_sum16: f32 = line16.taps().iter().sum();
        assert!((tap_sum16 - 1.0).abs() < 1.0e-6);
        for n in 0..48 {
            let y = line16.step(n as f32);
            if n >= 24 {
                assert!((y - ((n as f32) - 9.35)).abs() < 2.0e-3);
            }
        }

        let mut line4 = ParallelAlignLine::new(IoAlignmentMode::MinLatency4Tap, 0.75, 1);
        assert_eq!(line4.tap_count(), 4);
        assert!((line4.effective_delay_samples() - 1.75).abs() < 1.0e-12);
        let tap_sum4: f32 = line4.taps().iter().sum();
        assert!((tap_sum4 - 1.0).abs() < 1.0e-6);
        for n in 0..16 {
            let y = line4.step(n as f32);
            if n >= 6 {
                assert!((y - ((n as f32) - 1.75)).abs() < 1.0e-5);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_parallel_align_unity_dc_and_ramp_delay(
            delay in 0.0_f64..=24.0_f64,
            mode_idx in 0_usize..3,
            bounce in any::<bool>(),
        ) {
            let mode = match mode_idx {
                0 => IoAlignmentMode::FullBand64Tap,
                1 => IoAlignmentMode::LowLatency16Tap,
                _ => IoAlignmentMode::MinLatency4Tap,
            };
            let m_os = if bounce { 4 } else { 1 };
            let mut line = ParallelAlignLine::new(mode, delay, m_os);
            let sum: f64 = line.taps().iter().map(|&v| f64::from(v)).sum();
            prop_assert!((sum - 1.0).abs() < 1.0e-5);

            for n in 0..200 {
                let y = line.step(n as f32);
                prop_assert!(y.is_finite());
                if n >= 180 {
                    let expected = (n as f64) - line.effective_delay_samples();
                    prop_assert!((f64::from(y) - expected).abs() < 5.0e-2);
                }
            }
        }
    }
}
