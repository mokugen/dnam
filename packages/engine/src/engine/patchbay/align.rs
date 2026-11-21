//! Host-shell sub-sample latency aligners (`ReturnAligner`, `ParallelAlignMode`,
//! and `ParallelAlignLine`) for Layer 1 multi-port routing.
//!
//! Canonical Reference: `docs/specs/dsp/router_and_crossbar_topology.md` §3.2
//! (`ref[dsp.router.parallel-alignment]`).

#![forbid(unsafe_code)]

mod kaiser;

pub use kaiser::{
    IoAlignmentMode, KAISER_16_HALF_SAMPLE_TAPS, KAISER_16_ZERO_OFFSET_TAPS,
    KAISER_64_HALF_SAMPLE_TAPS, KAISER_64_ZERO_OFFSET_TAPS, KAISER_FULL_BAND_BETA,
    KAISER_FULL_BAND_CENTER_G, KAISER_FULL_BAND_TAPS, KAISER_LOW_LATENCY_BETA,
    KAISER_LOW_LATENCY_CENTER_G, KAISER_LOW_LATENCY_TAPS, MIN_LATENCY_CENTER_G,
    ParallelAlignmentMode, eval_min_latency_4tap_alignment,
    prepare_kaiser_sinc_alignment_weights_into,
};

use dnam_kernels::{catmull_rom_hermite_f32, scrub_and_flush_f32_bits};
use dnam_vm::math::{floor_f32, floor_f64, rcp_nr6_f64};

/// Power-of-two ring capacity per return channel (`256 >= 157` effective-rate samples @ `4x`).
pub const RETURN_ALIGN_CAPACITY: usize = 256;
const RETURN_ALIGN_MASK: usize = RETURN_ALIGN_CAPACITY - 1;

/// Maximum fractional delay supported by [`ReturnAligner`] in VM samples.
pub const RETURN_ALIGN_MAX_DELAY_SAMPLES: usize = RETURN_ALIGN_CAPACITY - 4;

/// Power-of-two ring capacity per parallel alignment channel (`256 >= 157` @ `4x` `FullBand64Tap`).
pub const PARALLEL_ALIGN_CAPACITY: usize = 256;
const PARALLEL_ALIGN_MASK: usize = PARALLEL_ALIGN_CAPACITY - 1;

const INV_48000_F64: f64 = 1.0 / 48_000.0;
const FRAC_SNAP_EPS: f64 = 1.0e-6;

const _: () = assert!(RETURN_ALIGN_CAPACITY.is_power_of_two() && RETURN_ALIGN_CAPACITY >= 157);
const _: () = assert!(PARALLEL_ALIGN_CAPACITY.is_power_of_two() && PARALLEL_ALIGN_CAPACITY >= 157);
const _: () = assert!(
    PARALLEL_ALIGN_CAPACITY >= KAISER_FULL_BAND_TAPS + crate::engine::OFFLINE_BOUNCE_PDC_SAMPLES
);

#[inline(always)]
fn clamp_finite_f32_nonneg(val: f32, hi: f32) -> f32 {
    if val.is_finite() {
        val.clamp(0.0, hi)
    } else {
        0.0
    }
}

/// Catmull-Rom cubic fractional latency aligner (`0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`).
#[derive(Clone, Debug)]
pub struct ReturnAligner {
    pub ring: [f32; RETURN_ALIGN_CAPACITY],
    pub write_ptr: usize,
    pub delay_samples: f32,
    delay_48k_samples: f32,
    fs_vm_hz: f64,
}

impl Default for ReturnAligner {
    fn default() -> Self {
        Self::new()
    }
}

impl ReturnAligner {
    /// Creates a zeroed [`ReturnAligner`] with `0.0`-sample pass-through latency at `48 kHz`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ring: [0.0; RETURN_ALIGN_CAPACITY],
            write_ptr: 0,
            delay_samples: 0.0,
            delay_48k_samples: 0.0,
            fs_vm_hz: 48_000.0,
        }
    }

    /// Clears the circular delay ring while preserving the configured alignment delay.
    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.write_ptr = 0;
    }

    /// Sets fractional return delay directly in active VM-rate samples (`[0.0, 252.0]`).
    pub fn set_delay(&mut self, delay: f32) {
        let clamped = clamp_finite_f32_nonneg(delay, RETURN_ALIGN_MAX_DELAY_SAMPLES as f32);
        self.delay_samples = clamped;
        let inv_fs = rcp_nr6_f64(self.fs_vm_hz.max(1.0));
        let scaled_48k = f64::from(clamped) * 48_000.0 * inv_fs;
        self.delay_48k_samples =
            clamp_finite_f32_nonneg(scaled_48k as f32, RETURN_ALIGN_MAX_DELAY_SAMPLES as f32);
    }

    /// Sets `48 kHz`-anchored fractional return delay and rescales it to the active VM rate.
    pub fn set_delay_48k(&mut self, delay_48k: f32) {
        self.delay_48k_samples =
            clamp_finite_f32_nonneg(delay_48k, RETURN_ALIGN_MAX_DELAY_SAMPLES as f32);
        let scaled_vm = f64::from(self.delay_48k_samples) * self.fs_vm_hz * INV_48000_F64;
        self.delay_samples =
            clamp_finite_f32_nonneg(scaled_vm as f32, RETURN_ALIGN_MAX_DELAY_SAMPLES as f32);
    }

    /// Reconfigures return alignment delay for `fs_vm_hz` without resetting history (`0 FDIV`).
    pub fn reconfigure_sample_rate(&mut self, fs_vm_hz: f64) {
        if !fs_vm_hz.is_finite() || fs_vm_hz <= 0.0 {
            return;
        }
        self.fs_vm_hz = fs_vm_hz;
        let scaled_vm = f64::from(self.delay_48k_samples) * fs_vm_hz * INV_48000_F64;
        self.delay_samples =
            clamp_finite_f32_nonneg(scaled_vm as f32, RETURN_ALIGN_MAX_DELAY_SAMPLES as f32);
    }

    /// Returns the active VM-rate alignment delay in samples.
    #[inline]
    #[must_use]
    pub const fn delay_samples(&self) -> f32 {
        self.delay_samples
    }

    /// Advances the aligner by one sample via causal 4-tap Catmull-Rom cubic interpolation (`0 FDIV`).
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        self.ring[self.write_ptr] = scrub_and_flush_f32_bits(x);

        let d_clamped =
            clamp_finite_f32_nonneg(self.delay_samples, RETURN_ALIGN_MAX_DELAY_SAMPLES as f32);
        let d_int = floor_f32(d_clamped) as usize;
        let mu = (d_clamped - (d_int as f32)).clamp(0.0, 1.0);

        let p0 = (self.write_ptr + RETURN_ALIGN_CAPACITY - d_int) & RETURN_ALIGN_MASK;
        let pm1 = if d_int == 0 {
            p0
        } else {
            (p0 + 1) & RETURN_ALIGN_MASK
        };
        let p1 = (p0 + RETURN_ALIGN_CAPACITY - 1) & RETURN_ALIGN_MASK;
        let p2 = (p0 + RETURN_ALIGN_CAPACITY - 2) & RETURN_ALIGN_MASK;

        let s_m1 = self.ring[pm1];
        let s_0 = self.ring[p0];
        let s_1 = self.ring[p1];
        let s_2 = self.ring[p2];

        self.write_ptr = (self.write_ptr + 1) & RETURN_ALIGN_MASK;
        catmull_rom_hermite_f32(s_m1, s_0, s_1, s_2, mu)
    }

    /// Advances the aligner using precomputed cold Kaiser-sinc taps (`FullBand64Tap` /
    /// `LowLatency16Tap`) when `kaiser_taps` is active and delay covers centering offset `G`, or
    /// falls back cleanly to centered 4-tap Catmull-Rom (`MinLatency4Tap`).
    #[inline(always)]
    pub fn step_aligned(
        &mut self,
        x: f32,
        mode: ParallelAlignmentMode,
        kaiser_taps: Option<&[f32; 64]>,
    ) -> f32 {
        let g = mode.centering_offset_host_samples();
        let tap_count = match mode {
            ParallelAlignmentMode::FullBand64Tap => KAISER_FULL_BAND_TAPS,
            ParallelAlignmentMode::LowLatency16Tap => KAISER_LOW_LATENCY_TAPS,
            ParallelAlignmentMode::MinLatency4Tap => 0,
        };
        let d_clamped =
            clamp_finite_f32_nonneg(self.delay_samples, RETURN_ALIGN_MAX_DELAY_SAMPLES as f32);
        let d_int = floor_f32(d_clamped) as usize;
        if tap_count > 0
            && d_int >= g
            && let Some(taps) = kaiser_taps
        {
            let base_offset = d_int - g;
            if base_offset + tap_count <= RETURN_ALIGN_CAPACITY {
                self.ring[self.write_ptr] = scrub_and_flush_f32_bits(x);
                let mut acc0 = 0.0_f32;
                let mut acc1 = 0.0_f32;
                for k in (0..tap_count).step_by(2) {
                    let i0 = (self.write_ptr + RETURN_ALIGN_CAPACITY - (base_offset + k))
                        & RETURN_ALIGN_MASK;
                    let i1 = (self.write_ptr + RETURN_ALIGN_CAPACITY - (base_offset + k + 1))
                        & RETURN_ALIGN_MASK;
                    acc0 += taps[k] * self.ring[i0];
                    acc1 += taps[k + 1] * self.ring[i1];
                }
                self.write_ptr = (self.write_ptr + 1) & RETURN_ALIGN_MASK;
                return acc0 + acc1;
            }
        }
        self.step(x)
    }
}

/// Per-channel parallel-path sub-sample alignment line supporting `1x` live and `4x` bounce
/// (`B = (M - 1) * G`, `d_eff = M * G + Delta_i`, `ref[dsp.router.parallel-alignment]`).
#[derive(Clone, Debug)]
pub struct ParallelAlignLine {
    ring: [f32; PARALLEL_ALIGN_CAPACITY],
    write_ptr: usize,
    mode: ParallelAlignmentMode,
    oversampling: usize,
    base_offset_b: usize,
    delay_int_j: usize,
    delay_frac_mu: f32,
    delta_48k_samples: f32,
    delta_samples_eff: f32,
    kaiser_taps: [f32; KAISER_FULL_BAND_TAPS],
    kaiser_active: bool,
}

impl Default for ParallelAlignLine {
    fn default() -> Self {
        Self::new(ParallelAlignmentMode::FullBand64Tap)
    }
}

impl ParallelAlignLine {
    /// Creates a zeroed [`ParallelAlignLine`] for `mode` at `1x` with `Delta_i = 0.0`.
    #[inline]
    #[must_use]
    pub const fn new(mode: ParallelAlignmentMode) -> Self {
        Self {
            ring: [0.0; PARALLEL_ALIGN_CAPACITY],
            write_ptr: 0,
            mode,
            oversampling: 1,
            base_offset_b: 0,
            delay_int_j: 0,
            delay_frac_mu: 0.0,
            delta_48k_samples: 0.0,
            delta_samples_eff: 0.0,
            kaiser_taps: [0.0; KAISER_FULL_BAND_TAPS],
            kaiser_active: false,
        }
    }

    /// Clears the history ring while retaining active alignment mode and filter taps.
    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.write_ptr = 0;
    }

    #[inline]
    const fn max_delta_int_for(mode: ParallelAlignmentMode, oversampling: usize) -> usize {
        let m = if oversampling == 4 { 4 } else { 1 };
        let g = mode.centering_offset_host_samples();
        let b = (m - 1) * g;
        match mode {
            ParallelAlignmentMode::FullBand64Tap => {
                PARALLEL_ALIGN_CAPACITY.saturating_sub(b + KAISER_FULL_BAND_TAPS)
            }
            ParallelAlignmentMode::LowLatency16Tap => {
                PARALLEL_ALIGN_CAPACITY.saturating_sub(b + KAISER_LOW_LATENCY_TAPS)
            }
            ParallelAlignmentMode::MinLatency4Tap => PARALLEL_ALIGN_CAPACITY.saturating_sub(b + 4),
        }
    }

    /// Updates [`ParallelAlignmentMode`] and recomputes `B` and filter weights (`0 FDIV`).
    #[inline]
    pub fn set_mode(&mut self, mode: ParallelAlignmentMode) {
        self.mode = mode;
        let os = self.oversampling;
        let delta = self.delta_samples_eff;
        self.configure(os, delta);
    }

    /// Configures oversampling factor `M in {1, 4}` and effective-rate arrival skew `Delta_i >= 0`
    /// (`0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`, `0` heap allocs).
    pub fn configure(&mut self, oversampling: usize, delta_samples_eff: f32) {
        let m = if oversampling >= 4 { 4 } else { 1 };
        self.oversampling = m;
        let g = self.mode.centering_offset_host_samples();
        self.base_offset_b = (m - 1) * g;

        let max_j = Self::max_delta_int_for(self.mode, m);
        let max_delta = (max_j as f64) + 0.999_999;
        let raw = if delta_samples_eff.is_finite() {
            f64::from(delta_samples_eff).clamp(0.0, max_delta)
        } else {
            0.0
        };
        let mut d_int = (floor_f64(raw) as usize).min(max_j);
        let mut mu = (raw - (d_int as f64)).clamp(0.0, 1.0);
        if mu >= 1.0 - FRAC_SNAP_EPS {
            if d_int < max_j {
                d_int += 1;
                mu = 0.0;
            } else {
                mu = 0.999_999;
            }
        } else if mu <= FRAC_SNAP_EPS {
            mu = 0.0;
        }

        self.delay_int_j = d_int;
        self.delay_frac_mu = mu as f32;
        self.delta_samples_eff = ((d_int as f64) + mu) as f32;

        self.kaiser_active = match self.mode {
            ParallelAlignmentMode::FullBand64Tap if mu > FRAC_SNAP_EPS => {
                let mut w = [0.0_f64; KAISER_FULL_BAND_TAPS];
                if prepare_kaiser_sinc_alignment_weights_into(
                    KAISER_FULL_BAND_TAPS,
                    KAISER_FULL_BAND_BETA,
                    mu,
                    &mut w,
                ) {
                    for (dst, &src) in self.kaiser_taps.iter_mut().zip(w.iter()) {
                        *dst = src as f32;
                    }
                    true
                } else {
                    false
                }
            }
            ParallelAlignmentMode::LowLatency16Tap if mu > FRAC_SNAP_EPS => {
                let mut w = [0.0_f64; KAISER_FULL_BAND_TAPS];
                if prepare_kaiser_sinc_alignment_weights_into(
                    KAISER_LOW_LATENCY_TAPS,
                    KAISER_LOW_LATENCY_BETA,
                    mu,
                    &mut w,
                ) {
                    self.kaiser_taps.fill(0.0);
                    for (dst, &src) in self.kaiser_taps[..KAISER_LOW_LATENCY_TAPS]
                        .iter_mut()
                        .zip(w[..KAISER_LOW_LATENCY_TAPS].iter())
                    {
                        *dst = src as f32;
                    }
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
    }

    /// Sets `48 kHz`-anchored arrival delta and updates alignment for `(fs_vm_hz, offline_bounce)`.
    #[inline]
    pub fn set_delta_48k(&mut self, delta_48k_samples: f32, fs_vm_hz: f64, offline_bounce: bool) {
        self.delta_48k_samples = if delta_48k_samples.is_finite() {
            delta_48k_samples.max(0.0)
        } else {
            0.0
        };
        self.reconfigure_sample_rate(fs_vm_hz, offline_bounce);
    }

    /// Recomputes `M`, `B = (M - 1)G`, and Kaiser weights for `(fs_vm_hz, offline_bounce)` (`0 FDIV`).
    pub fn reconfigure_sample_rate(&mut self, fs_vm_hz: f64, offline_bounce: bool) {
        if !fs_vm_hz.is_finite() || fs_vm_hz <= 0.0 {
            return;
        }
        let m = if offline_bounce { 4 } else { 1 };
        let scaled = f64::from(self.delta_48k_samples) * fs_vm_hz * INV_48000_F64;
        self.configure(m, scaled as f32);
    }

    /// Sets `48 kHz`-anchored arrival delta and updates `1x` host-rate alignment for
    /// `(fs_host_hz, offline_bounce)`.
    #[inline]
    pub fn set_delta_48k_host(
        &mut self,
        delta_48k_samples: f32,
        fs_host_hz: f64,
        offline_bounce: bool,
    ) {
        self.delta_48k_samples = if delta_48k_samples.is_finite() {
            delta_48k_samples.max(0.0)
        } else {
            0.0
        };
        self.reconfigure_host_rate(fs_host_hz, offline_bounce);
    }

    /// Recomputes `1x` host-rate alignment (`M = 1`, `B = 0`) including the `12`-host-sample
    /// offline `4x` bounce VM PDC offset (`0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`, `0` allocs).
    pub fn reconfigure_host_rate(&mut self, fs_host_hz: f64, offline_bounce: bool) {
        if !fs_host_hz.is_finite() || fs_host_hz <= 0.0 {
            return;
        }
        let pdc_offset = if offline_bounce {
            crate::engine::OFFLINE_BOUNCE_PDC_SAMPLES as f64
        } else {
            0.0
        };
        let delta_host =
            f64::from(self.delta_48k_samples) * fs_host_hz * INV_48000_F64 + pdc_offset;
        self.configure(1, delta_host as f32);
    }

    /// Returns the active alignment mode.
    #[inline]
    #[must_use]
    pub const fn mode(&self) -> ParallelAlignmentMode {
        self.mode
    }

    /// Returns the active oversampling factor (`1` or `4`).
    #[inline]
    #[must_use]
    pub const fn oversampling(&self) -> usize {
        self.oversampling
    }

    /// Returns `B = (M - 1)G` in effective-rate samples.
    #[inline]
    #[must_use]
    pub const fn base_offset_b(&self) -> usize {
        self.base_offset_b
    }

    /// Returns whether cold Kaiser-sinc FIR taps are currently active.
    #[inline]
    #[must_use]
    pub const fn kaiser_active(&self) -> bool {
        self.kaiser_active
    }

    /// Returns effective alignment delay `d_eff = M * G + Delta_i` in effective-rate samples.
    #[inline]
    #[must_use]
    pub fn effective_delay_samples(&self) -> f64 {
        (self.oversampling * self.mode.centering_offset_host_samples()) as f64
            + f64::from(self.delta_samples_eff)
    }

    /// Returns combined host-sample PDC (`G` live, `G + 12` under `4x` bounce).
    #[inline]
    #[must_use]
    pub const fn pdc_host_samples(&self, offline_bounce: bool) -> usize {
        self.mode.pdc_host_samples(offline_bounce)
    }

    /// Advances the alignment line by one effective-rate sample (`0 FDIV`, `0 FSQRT`, `0 IDIV`, `0 libm`).
    #[inline(always)]
    pub fn step(&mut self, x: f32) -> f32 {
        self.ring[self.write_ptr] = scrub_and_flush_f32_bits(x);

        let y = match self.mode {
            ParallelAlignmentMode::MinLatency4Tap => {
                let d_center = self.base_offset_b + MIN_LATENCY_CENTER_G + self.delay_int_j;
                let pm1 = (self.write_ptr + PARALLEL_ALIGN_CAPACITY - (d_center - 1))
                    & PARALLEL_ALIGN_MASK;
                let p0 =
                    (self.write_ptr + PARALLEL_ALIGN_CAPACITY - d_center) & PARALLEL_ALIGN_MASK;
                let p1 = (self.write_ptr + PARALLEL_ALIGN_CAPACITY - (d_center + 1))
                    & PARALLEL_ALIGN_MASK;
                let p2 = (self.write_ptr + PARALLEL_ALIGN_CAPACITY - (d_center + 2))
                    & PARALLEL_ALIGN_MASK;
                catmull_rom_hermite_f32(
                    self.ring[pm1],
                    self.ring[p0],
                    self.ring[p1],
                    self.ring[p2],
                    self.delay_frac_mu,
                )
            }
            ParallelAlignmentMode::FullBand64Tap | ParallelAlignmentMode::LowLatency16Tap => {
                if self.kaiser_active {
                    let tap_count = self.mode.tap_count();
                    let base = self.base_offset_b + self.delay_int_j;
                    let mut acc0 = 0.0_f32;
                    let mut acc1 = 0.0_f32;
                    for k in (0..tap_count).step_by(2) {
                        let i0 = (self.write_ptr + PARALLEL_ALIGN_CAPACITY - (base + k))
                            & PARALLEL_ALIGN_MASK;
                        let i1 = (self.write_ptr + PARALLEL_ALIGN_CAPACITY - (base + k + 1))
                            & PARALLEL_ALIGN_MASK;
                        acc0 += self.kaiser_taps[k] * self.ring[i0];
                        acc1 += self.kaiser_taps[k + 1] * self.ring[i1];
                    }
                    acc0 + acc1
                } else {
                    let d_exact = self.base_offset_b
                        + self.mode.centering_offset_host_samples()
                        + self.delay_int_j;
                    let idx =
                        (self.write_ptr + PARALLEL_ALIGN_CAPACITY - d_exact) & PARALLEL_ALIGN_MASK;
                    self.ring[idx]
                }
            }
        };

        self.write_ptr = (self.write_ptr + 1) & PARALLEL_ALIGN_MASK;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_return_aligner_catmull_rom_zoh_and_step_aligned() {
        let mut aligner = ReturnAligner::default();
        assert_eq!(aligner.delay_samples(), 0.0);
        // Zero-delay exact pass-through.
        for n in 0..8 {
            let x = (n as f32) * 0.25 - 0.5;
            assert_eq!(aligner.step(x).to_bits(), x.to_bits());
        }

        aligner.set_delay(8.5);
        aligner.reset();
        assert_eq!(aligner.delay_samples(), 8.5);
        for n in 0..24 {
            let out = aligner.step(n as f32);
            if n >= 12 {
                assert!((out - ((n as f32) - 8.5)).abs() < 1.0e-4);
            }
        }

        // Sub-sample delay (< 1.0) remains strictly causal and never reads future write_ptr + 1.
        aligner.set_delay(0.5);
        aligner.reset();
        aligner.ring[1] = 100.0;
        let out0 = aligner.step(1.0);
        assert!((out0 - 0.5).abs() < 1.0e-5);

        // 48 kHz-anchored ZOH scaling.
        aligner.set_delay_48k(10.0);
        aligner.reconfigure_sample_rate(96_000.0);
        assert!((aligner.delay_samples() - 20.0).abs() < 1.0e-5);
        aligner.reconfigure_sample_rate(f64::NAN);
        assert!((aligner.delay_samples() - 20.0).abs() < 1.0e-5);

        // Aligned step with Kaiser taps.
        aligner.set_delay(40.0);
        aligner.reset();
        let mut dc_out = 0.0_f32;
        for _ in 0..128 {
            dc_out = aligner.step_aligned(
                1.0,
                ParallelAlignmentMode::FullBand64Tap,
                Some(&KAISER_64_HALF_SAMPLE_TAPS),
            );
        }
        assert!((dc_out - 1.0).abs() < 1.0e-5);

        // Non-finite delay and sample guard.
        aligner.set_delay(f32::NAN);
        assert_eq!(aligner.delay_samples(), 0.0);
        assert_eq!(aligner.step(f32::NAN).to_bits(), 0);
    }

    /// verify[dsp.router.parallel-alignment]
    #[test]
    fn test_parallel_align_line_4x_bounce_and_all_modes() {
        // 1. FullBand64Tap at 4x bounce: B = 93, M * G = 124, combined PDC = 43 host samples.
        let mut line = ParallelAlignLine::default();
        assert_eq!(line.mode(), ParallelAlignmentMode::FullBand64Tap);
        line.configure(4, 0.0);
        assert_eq!(line.oversampling(), 4);
        assert_eq!(line.base_offset_b(), 93);
        assert!(!line.kaiser_active());
        assert!((line.effective_delay_samples() - 124.0).abs() < 1.0e-12);
        assert_eq!(line.pdc_host_samples(true), 43);

        for n in 0..160 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let y = line.step(x);
            if n == 124 {
                assert_eq!(y, 1.0);
            } else {
                assert_eq!(y, 0.0);
            }
        }

        // Fractional Delta_i = 5.5 -> d_eff = 129.5 (symmetric split across samples 129 and 130).
        line.reset();
        line.configure(4, 5.5);
        assert!(line.kaiser_active());
        assert!((line.effective_delay_samples() - 129.5).abs() < 1.0e-6);
        let mut imp_out = [0.0_f32; 170];
        for (n, slot) in imp_out.iter_mut().enumerate() {
            *slot = line.step(if n == 0 { 1.0 } else { 0.0 });
        }
        assert!(imp_out[129] > 0.6 && imp_out[130] > 0.6);
        assert!((imp_out[129] - imp_out[130]).abs() < 1.0e-6);

        // LowLatency16Tap with fractional ZOH scaling.
        line.set_mode(ParallelAlignmentMode::LowLatency16Tap);
        line.set_delta_48k(2.0, 88_200.0, false);
        assert!(line.kaiser_active());
        line.reset();
        let mut dc_16 = 0.0_f32;
        for _ in 0..64 {
            dc_16 = line.step(1.0);
        }
        assert!((dc_16 - 1.0).abs() < 1.0e-5);

        // MinLatency4Tap at 4x bounce: linear ramp reproduces n - (M * G + Delta_i).
        line.set_mode(ParallelAlignmentMode::MinLatency4Tap);
        line.configure(4, 2.37);
        assert!(!line.kaiser_active());
        assert!((line.effective_delay_samples() - 6.37).abs() < 1.0e-5);
        line.reset();
        for n in 0..32 {
            let y = line.step(n as f32);
            if n >= 16 {
                assert!((y - ((n as f32) - 6.37)).abs() < 1.0e-4);
            }
        }
    }

    /// verify[dsp.router.parallel-alignment]
    #[test]
    fn test_parallel_align_line_host_rate_4x_bounce_pdc() {
        let mut line = ParallelAlignLine::new(ParallelAlignmentMode::FullBand64Tap);
        line.set_delta_48k_host(0.0, 48_000.0, true);
        assert_eq!(line.oversampling(), 1);
        assert_eq!(line.base_offset_b(), 0);
        assert_eq!(line.pdc_host_samples(true), 43);
        assert!((line.effective_delay_samples() - 43.0).abs() < 1.0e-12);
        for n in 0..64 {
            let y = line.step(if n == 0 { 1.0 } else { 0.0 });
            if n == 43 {
                assert_eq!(y, 1.0);
            } else {
                assert_eq!(y, 0.0);
            }
        }

        line.reset();
        line.set_delta_48k_host(1.5, 96_000.0, true);
        assert_eq!(line.oversampling(), 1);
        assert!(!line.kaiser_active());
        assert!((line.effective_delay_samples() - 46.0).abs() < 1.0e-5);
        line.reconfigure_host_rate(f64::NAN, false);
        assert!((line.effective_delay_samples() - 46.0).abs() < 1.0e-5);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// verify[dsp.router.parallel-alignment]
        #[test]
        fn prop_kaiser_weights_and_align_line_preserve_unity_dc(
            mode_idx in 0_usize..3_usize,
            bounce in prop::bool::ANY,
            mu in 0.0_f64..1.0_f64,
            delta_eff in 0.0_f32..32.0_f32,
        ) {
            let mut w = [0.0_f64; 64];
            for &(taps, beta) in &[(64_usize, 8.6_f64), (16, 6.5)] {
                prop_assert!(prepare_kaiser_sinc_alignment_weights_into(taps, beta, mu, &mut w));
                let sum: f64 = w[..taps].iter().sum();
                prop_assert!((sum - 1.0).abs() < 1.0e-12);
                prop_assert!(w[taps..].iter().all(|&v| v == 0.0));
            }

            let mode = match mode_idx {
                0 => ParallelAlignmentMode::FullBand64Tap,
                1 => ParallelAlignmentMode::LowLatency16Tap,
                _ => ParallelAlignmentMode::MinLatency4Tap,
            };
            let m = if bounce { 4 } else { 1 };
            let mut line = ParallelAlignLine::new(mode);
            line.configure(m, delta_eff);
            let mut y = 0.0_f32;
            for _ in 0..260 {
                y = line.step(1.0);
            }
            prop_assert!(y.is_finite());
            prop_assert!((y - 1.0).abs() < 1.0e-4);
            prop_assert!((eval_min_latency_4tap_alignment(1.0, 1.0, 1.0, 1.0, mu as f32) - 1.0).abs() < 1.0e-6);
        }
    }
}
