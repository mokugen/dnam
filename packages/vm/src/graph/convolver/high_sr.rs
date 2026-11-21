//! Off-`48 kHz` `ModalConvolver` High-SR Early FIR and Modal Onset Extension
//! (`ref[dsp.zoh.cabinet-fir-onset]`, `ref[dsp.cabinet.mirrored-fir-simd]`).
//!
//! Covers all four mutually exclusive rate domains across `44.1–384 kHz` (`Case (a)` exact
//! `48 kHz * q` sparse stride, `Case (b)` live `1×` `44.1..<48 kHz` `64`-tap first-tap difference
//! with `0-ULP` unit DC, `Case (c)` `4×` bounce exact `u64` 6-point Lagrange decomposition, and
//! `Case (d)` DC-normalized Catmull-Rom FIR resampler) with `!alloc` [`ColdBuf`] backing.

#![forbid(unsafe_code)]

mod lagrange;

pub use lagrange::{BounceLagrangeTap, GAIN_BOUNDED_FIRST_TAP_64_BITS, prepare_bounce_6point_u64};

#[cfg(feature = "alloc")]
use alloc::vec;

use crate::math::{
    ceil_f64, clamp_finite_f32, const_abs_f64, flush_recursive_state_f32, rcp_nr6_f64,
    round_half_away_f64,
};
use crate::types::{ColdBuf, VmError};

use lagrange::{eval_bounce_diff5, eval_tap1_diff, resample_catmull_rom_dc_norm};

/// Double-mirrored off-`48 kHz` extension ring frame capacity (`1,024` samples/channel).
pub const STAGE3_EXT_FIR_CAP: usize = 1_024;

/// Power-of-two index bitmask for [`STAGE3_EXT_FIR_CAP`] (`1,023`).
pub const STAGE3_EXT_FIR_MASK: usize = STAGE3_EXT_FIR_CAP - 1;

/// Maximum cumulative `48 kHz` early FIR taps per channel (`Ultra`: `128`).
pub const MAX_CONVOLVER_BASE_TAPS: usize = 128;

const EXT_HIST_STRIDE: usize = STAGE3_EXT_FIR_CAP * 2;
const EXT_HIST_R_OFF: usize = EXT_HIST_STRIDE;
const EXT_DENSE_L_OFF: usize = EXT_HIST_STRIDE * 2;
const EXT_DENSE_R_OFF: usize = EXT_DENSE_L_OFF + STAGE3_EXT_FIR_CAP;

/// Required [`ColdBuf`] `f32` length (`6,144`) when executing off-`48 kHz`:
/// `[hist_l(2048), hist_r(2048), dense_l(1024), dense_r(1024)]`.
pub const EXT_FIR_STORAGE_F32_LEN: usize = EXT_DENSE_R_OFF + STAGE3_EXT_FIR_CAP;

const RATE_REF_48K_F64: f64 = 48_000.0;
const RATE_44K1_F64: f64 = 44_100.0;
const RATE_176K4_F64: f64 = 176_400.0;
const RATE_384K_F64: f64 = 384_000.0;
const INV_48000_F64: f64 = 1.0 / RATE_REF_48K_F64;
const INV_3900_F64: f64 = 1.0 / 3_900.0;
const MAX_FIR_TAP_ABS: f32 = 16.0;
const ONSET_48K_SAMPLES: f64 = 16.0;
const ONSET_MIN_SAMPLES: usize = 14;
const ONSET_MAX_SAMPLES: usize = STAGE3_EXT_FIR_CAP - 1;

const _: () = assert!(STAGE3_EXT_FIR_CAP.is_power_of_two());
const _: () = assert!(EXT_FIR_STORAGE_F32_LEN == 6_144);

/// Active sample-rate adaptation regime selected by [`ConvolverHighSrFirExt`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvolverHighSrMode {
    /// Native `48,000 Hz` (`q = 1`), zero cold extension storage (`0` floats).
    Native48k,
    /// `Case (a)`: exact integer multiples `q * 48,000 Hz` (`q in {2, 4, 8}`).
    CaseA {
        /// Integer delay stride `q in {2, 4, 8}`.
        stride_q: usize,
    },
    /// `Case (b)`: live `1×` `44,100 <= f_{s,vm} < 48,000 Hz` (`64`-tap first-tap diff + tail).
    CaseB,
    /// `Case (c)`: `4×` offline bounce `176,400..=384,000 Hz` (`6`-point Lagrange taps).
    CaseC,
    /// `Case (d)`: remaining off-`48 kHz` rates via DC-normalized Catmull-Rom FIR resampling.
    CaseD,
}

#[inline]
fn classify_rate_mode(fs_vm: f64, is_bounce_4x: bool) -> Result<ConvolverHighSrMode, VmError> {
    if !fs_vm.is_finite() || fs_vm <= 0.0 {
        return Err(VmError::InvalidSampleRate);
    }
    for q in [1_usize, 2, 4, 8] {
        if const_abs_f64(fs_vm - (q as f64) * RATE_REF_48K_F64) <= 1.0e-6 {
            return Ok(if q == 1 && !is_bounce_4x {
                ConvolverHighSrMode::Native48k
            } else {
                ConvolverHighSrMode::CaseA { stride_q: q }
            });
        }
    }
    if !is_bounce_4x && (RATE_44K1_F64..RATE_REF_48K_F64).contains(&fs_vm) {
        return Ok(ConvolverHighSrMode::CaseB);
    }
    if is_bounce_4x && (RATE_176K4_F64..=RATE_384K_F64).contains(&fs_vm) {
        return Ok(ConvolverHighSrMode::CaseC);
    }
    Ok(ConvolverHighSrMode::CaseD)
}

/// Cold off-`48 kHz` `ModalConvolver` FIR & modal onset extension backed by [`ColdBuf<'a, f32>`]
/// (`0` floats at `48,000 Hz`, `6,144` floats off-`48 kHz`, `ref[dsp.zoh.cabinet-fir-onset]`).
#[derive(Debug)]
pub struct ConvolverHighSrFirExt<'a> {
    storage: Option<ColdBuf<'a, f32>>,
    mode: ConvolverHighSrMode,
    active_base: usize,
    out_len: usize,
    onset_delay: usize,
    write_pos: usize,
    case_b_b0: [f32; 2],
    case_b_b1: [f32; 2],
    tap1_diff: [f32; 63],
    bounce_taps: [BounceLagrangeTap; MAX_CONVOLVER_BASE_TAPS],
}

impl<'a> ConvolverHighSrFirExt<'a> {
    /// Returns an empty `Native48k` stub with zero backing storage for `!alloc` initialization.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            storage: None,
            mode: ConvolverHighSrMode::Native48k,
            active_base: 0,
            out_len: 0,
            onset_delay: ONSET_MIN_SAMPLES,
            write_pos: 0,
            case_b_b0: [0.0; 2],
            case_b_b1: [0.0; 2],
            tap1_diff: [0.0; 63],
            bounce_taps: [BounceLagrangeTap::default(); MAX_CONVOLVER_BASE_TAPS],
        }
    }

    #[inline(always)]
    fn storage_len(&self) -> usize {
        self.storage.as_ref().map_or(0, ColdBuf::len)
    }

    /// Returns allocated backing capacity in `f32` elements (`0` or `>= 6,144`).
    #[inline]
    pub fn backing_f32_capacity(&self) -> usize {
        self.storage_len()
    }

    #[inline(always)]
    fn storage_slice(&self) -> &[f32] {
        self.storage.as_ref().map_or(&[], ColdBuf::as_slice)
    }

    #[inline(always)]
    fn storage_mut_slice(&mut self) -> &mut [f32] {
        self.storage.as_mut().map_or(&mut [], ColdBuf::as_mut_slice)
    }

    /// Returns the required [`ColdBuf`] `f32` element count (`0` at `48 kHz` live, `6,144` otherwise).
    #[inline]
    pub fn required_f32_len(fs_vm: f64, is_bounce_4x: bool) -> usize {
        if !is_bounce_4x && fs_vm.is_finite() && const_abs_f64(fs_vm - RATE_REF_48K_F64) <= 1.0e-6 {
            0
        } else {
            EXT_FIR_STORAGE_F32_LEN
        }
    }

    /// Wraps [`ColdBuf<'a, f32>`], failing closed with [`VmError::InsufficientBackingCapacity`] if undersized.
    pub fn from_cold_buf(
        mut storage: ColdBuf<'a, f32>,
        fs_vm: f64,
        is_bounce_4x: bool,
    ) -> Result<Self, VmError> {
        let mode = classify_rate_mode(fs_vm, is_bounce_4x)?;
        let required = Self::required_f32_len(fs_vm, is_bounce_4x);
        if storage.len() < required {
            return Err(VmError::InsufficientBackingCapacity);
        }
        storage.as_mut_slice()[..required].fill(0.0);
        Ok(Self {
            storage: Some(storage),
            mode,
            active_base: 0,
            out_len: 0,
            onset_delay: ONSET_MIN_SAMPLES,
            write_pos: 0,
            case_b_b0: [0.0; 2],
            case_b_b1: [0.0; 2],
            tap1_diff: [0.0; 63],
            bounce_taps: [BounceLagrangeTap::default(); MAX_CONVOLVER_BASE_TAPS],
        })
    }

    /// Convenience constructor wrapping `&'a mut [f32]` without `alloc`.
    #[inline]
    pub fn from_slice(
        slice: &'a mut [f32],
        fs_vm: f64,
        is_bounce_4x: bool,
    ) -> Result<Self, VmError> {
        Self::from_cold_buf(ColdBuf::borrowed(slice), fs_vm, is_bounce_4x)
    }

    /// Allocates owned cold storage sized for `(fs_vm, is_bounce_4x)` (`feature = "alloc"`).
    #[cfg(feature = "alloc")]
    pub fn new_owned_for_rate(fs_vm: f64, is_bounce_4x: bool) -> Result<Self, VmError> {
        classify_rate_mode(fs_vm, is_bounce_4x)?;
        let len = Self::required_f32_len(fs_vm, is_bounce_4x);
        Self::from_cold_buf(ColdBuf::Owned(vec![0.0_f32; len]), fs_vm, is_bounce_4x)
    }

    /// Returns the active [`ConvolverHighSrMode`].
    #[inline]
    pub const fn mode(&self) -> ConvolverHighSrMode {
        self.mode
    }

    /// Returns the effective-rate modal onset delay in VM samples (`[14, 1_023]`).
    #[inline]
    pub const fn onset_delay(&self) -> usize {
        self.onset_delay
    }

    /// Returns the prepared dense resampled tap count (`out_len`).
    #[inline]
    pub const fn out_len(&self) -> usize {
        self.out_len
    }

    /// Zeroes the double-mirrored stereo input history ring while preserving prepared FIR taps.
    #[inline]
    pub fn reset(&mut self) {
        let slice = self.storage_mut_slice();
        let hist_len = slice.len().min(EXT_DENSE_L_OFF);
        slice[..hist_len].fill(0.0);
        self.write_pos = 0;
    }

    /// Prepares off-`48 kHz` FIR taps and modal onset delay for `fir_l`/`fir_r` (`<= 128` taps).
    pub fn prepare(
        &mut self,
        fir_l: &[f32],
        fir_r: &[f32],
        fs_vm: f64,
        is_bounce_4x: bool,
    ) -> Result<(), VmError> {
        let mode = classify_rate_mode(fs_vm, is_bounce_4x)?;
        let required = Self::required_f32_len(fs_vm, is_bounce_4x);
        if self.storage_len() < required {
            return Err(VmError::InsufficientBackingCapacity);
        }
        let active = fir_l.len().min(fir_r.len()).min(MAX_CONVOLVER_BASE_TAPS);
        self.mode = mode;
        self.active_base = active;
        self.case_b_b0 = [0.0; 2];
        self.case_b_b1 = [0.0; 2];

        if matches!(mode, ConvolverHighSrMode::Native48k) {
            self.out_len = 0;
            self.onset_delay = 16;
            return Ok(());
        }

        let rho_vm = RATE_REF_48K_F64 * rcp_nr6_f64(fs_vm);
        let default_onset = (round_half_away_f64(ONSET_48K_SAMPLES * fs_vm * INV_48000_F64)
            as usize)
            .clamp(ONSET_MIN_SAMPLES, ONSET_MAX_SAMPLES);
        self.storage_mut_slice()[EXT_DENSE_L_OFF..EXT_DENSE_R_OFF + STAGE3_EXT_FIR_CAP].fill(0.0);

        match mode {
            ConvolverHighSrMode::CaseA { stride_q } => {
                self.onset_delay = (16 * stride_q).clamp(ONSET_MIN_SAMPLES, ONSET_MAX_SAMPLES);
                self.prepare_case_a_or_c(fir_l, fir_r, active);
            }
            ConvolverHighSrMode::CaseB => {
                self.onset_delay = default_onset;
                self.prepare_case_b(fir_l, fir_r, active, fs_vm, rho_vm);
            }
            ConvolverHighSrMode::CaseC => {
                self.onset_delay = default_onset;
                self.prepare_case_a_or_c(fir_l, fir_r, active);
                for k in 1..active {
                    self.bounce_taps[k] = prepare_bounce_6point_u64(k, fs_vm);
                }
            }
            ConvolverHighSrMode::CaseD => {
                self.onset_delay = default_onset;
                let out_len = (ceil_f64((active as f64) * fs_vm * INV_48000_F64) as usize)
                    .clamp(active, STAGE3_EXT_FIR_CAP);
                self.out_len = out_len;
                let buf = self.storage_mut_slice();
                resample_stereo_pair_catmull_rom(
                    buf,
                    [fir_l, fir_r],
                    0..active,
                    active,
                    out_len,
                    rho_vm,
                );
            }
            ConvolverHighSrMode::Native48k => {}
        }
        Ok(())
    }

    fn prepare_case_a_or_c(&mut self, fir_l: &[f32], fir_r: &[f32], active: usize) {
        self.out_len = active;
        let buf = self.storage_mut_slice();
        for k in 0..active {
            buf[EXT_DENSE_L_OFF + k] =
                clamp_finite_f32(fir_l[k], -MAX_FIR_TAP_ABS, MAX_FIR_TAP_ABS, 0.0);
            buf[EXT_DENSE_R_OFF + k] =
                clamp_finite_f32(fir_r[k], -MAX_FIR_TAP_ABS, MAX_FIR_TAP_ABS, 0.0);
        }
    }

    fn prepare_case_b(
        &mut self,
        fir_l: &[f32],
        fir_r: &[f32],
        active: usize,
        fs_vm: f64,
        rho_vm: f64,
    ) {
        let lambda = (RATE_REF_48K_F64 - fs_vm) * INV_3900_F64;
        let mut acc = f64::from(f32::from_bits(GAIN_BOUNDED_FIRST_TAP_64_BITS[0]));
        self.tap1_diff[0] = (lambda * acc) as f32;
        for j in 1..63 {
            acc += f64::from(f32::from_bits(GAIN_BOUNDED_FIRST_TAP_64_BITS[j]));
            self.tap1_diff[j] = (lambda * (acc - 1.0)) as f32;
        }
        let clamp_tap = |x: f32| clamp_finite_f32(x, -MAX_FIR_TAP_ABS, MAX_FIR_TAP_ABS, 0.0);
        self.case_b_b0 = [
            fir_l.first().copied().map(clamp_tap).unwrap_or(0.0),
            fir_r.first().copied().map(clamp_tap).unwrap_or(0.0),
        ];
        self.case_b_b1 = [
            fir_l.get(1).copied().map(clamp_tap).unwrap_or(0.0),
            fir_r.get(1).copied().map(clamp_tap).unwrap_or(0.0),
        ];
        self.out_len = active;
        if active > 2 {
            let buf = self.storage_mut_slice();
            resample_stereo_pair_catmull_rom(
                buf,
                [fir_l, fir_r],
                2..active,
                active,
                active,
                rho_vm,
            );
        }
    }

    /// Steps one stereo frame `(x_l, x_r)` returning `((early_l, early_r), (onset_l, onset_r))`.
    #[inline]
    pub fn step_frame(&mut self, x_l: f32, x_r: f32) -> ((f32, f32), (f32, f32)) {
        if matches!(self.mode, ConvolverHighSrMode::Native48k)
            || self.storage_len() < EXT_FIR_STORAGE_F32_LEN
        {
            return ((0.0, 0.0), (0.0, 0.0));
        }
        let (xl, xr) = (
            flush_recursive_state_f32(x_l),
            flush_recursive_state_f32(x_r),
        );
        let w = self.write_pos & STAGE3_EXT_FIR_MASK;
        let base = w + STAGE3_EXT_FIR_CAP;
        let onset_idx = base - self.onset_delay;

        let onset = {
            let buf = self.storage_mut_slice();
            buf[w] = xl;
            buf[base] = xl;
            buf[EXT_HIST_R_OFF + w] = xr;
            buf[EXT_HIST_R_OFF + base] = xr;
            (buf[onset_idx], buf[EXT_HIST_R_OFF + onset_idx])
        };

        let early = match self.mode {
            ConvolverHighSrMode::Native48k => (0.0, 0.0),
            ConvolverHighSrMode::CaseA { stride_q } => {
                step_dense_frame(self.storage_slice(), base, self.active_base, stride_q)
            }
            ConvolverHighSrMode::CaseB => self.step_case_b_frame(base),
            ConvolverHighSrMode::CaseC => self.step_case_c_frame(base),
            ConvolverHighSrMode::CaseD => {
                step_dense_frame(self.storage_slice(), base, self.out_len, 1)
            }
        };
        self.write_pos = (w + 1) & STAGE3_EXT_FIR_MASK;
        (early, onset)
    }

    #[inline]
    fn step_case_b_frame(&self, base: usize) -> (f32, f32) {
        let buf = self.storage_slice();
        let hist_l = &buf[..EXT_HIST_R_OFF];
        let hist_r = &buf[EXT_HIST_R_OFF..EXT_DENSE_L_OFF];
        let y_tap1_l = eval_tap1_diff(hist_l, base, &self.tap1_diff);
        let y_tap1_r = eval_tap1_diff(hist_r, base, &self.tap1_diff);
        let (dense_l, dense_r) = step_dense_frame(buf, base, self.out_len, 1);
        (
            self.case_b_b0[0] * hist_l[base] + self.case_b_b1[0] * y_tap1_l + dense_l,
            self.case_b_b0[1] * hist_r[base] + self.case_b_b1[1] * y_tap1_r + dense_r,
        )
    }

    #[inline]
    fn step_case_c_frame(&self, base: usize) -> (f32, f32) {
        if self.active_base == 0 {
            return (0.0, 0.0);
        }
        let buf = self.storage_slice();
        let hist_l = &buf[..EXT_HIST_R_OFF];
        let hist_r = &buf[EXT_HIST_R_OFF..EXT_DENSE_L_OFF];
        let mut early_l = buf[EXT_DENSE_L_OFF] * hist_l[base];
        let mut early_r = buf[EXT_DENSE_R_OFF] * hist_r[base];
        for k in 1..self.active_base {
            let tap = &self.bounce_taps[k];
            early_l += buf[EXT_DENSE_L_OFF + k] * eval_bounce_diff5(hist_l, base, tap);
            early_r += buf[EXT_DENSE_R_OFF + k] * eval_bounce_diff5(hist_r, base, tap);
        }
        (early_l, early_r)
    }

    /// Steps a `C`-frame stereo chunk into caller stack arrays with `0-ULP` parity against [`Self::step_frame`].
    #[inline]
    pub fn step_chunk<const C: usize>(
        &mut self,
        inputs_lr: &[[f32; 2]; C],
        early_out: &mut [[f32; 2]; C],
        onset_out: &mut [[f32; 2]; C],
    ) {
        for n in 0..C {
            let ((el, er), (ol, or)) = self.step_frame(inputs_lr[n][0], inputs_lr[n][1]);
            (early_out[n], onset_out[n]) = ([el, er], [ol, or]);
        }
    }
}

fn resample_stereo_pair_catmull_rom(
    buf: &mut [f32],
    fir_lr: [&[f32]; 2],
    tap_range: core::ops::Range<usize>,
    active: usize,
    out_len: usize,
    rho_vm: f64,
) {
    if active == 0 {
        return;
    }
    let in_len = active + 1;
    for (fir, dense_off) in [(fir_lr[0], EXT_DENSE_L_OFF), (fir_lr[1], EXT_DENSE_R_OFF)] {
        let mut padded = [0.0_f32; MAX_CONVOLVER_BASE_TAPS + 1];
        for k in tap_range.clone() {
            padded[k] = clamp_finite_f32(fir[k], -MAX_FIR_TAP_ABS, MAX_FIR_TAP_ABS, 0.0);
        }
        resample_catmull_rom_dc_norm(
            &padded[..in_len],
            rho_vm,
            out_len,
            &mut buf[dense_off..dense_off + out_len],
        );
    }
}

#[inline]
fn step_dense_frame(buf: &[f32], base: usize, len: usize, stride: usize) -> (f32, f32) {
    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for k in 0..len {
        let idx = base - k * stride;
        sum_l += buf[EXT_DENSE_L_OFF + k] * buf[idx];
        sum_r += buf[EXT_DENSE_R_OFF + k] * buf[EXT_HIST_R_OFF + idx];
    }
    (sum_l, sum_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::const_abs_f32;

    fn step_dc(ext: &mut ConvolverHighSrFirExt<'_>, frames: usize) -> ((f32, f32), (f32, f32)) {
        for _ in 0..frames {
            let _ = ext.step_frame(1.0, 1.0);
        }
        ext.step_frame(1.0, 1.0)
    }

    #[test]
    fn test_first_tap_bits_required_len_and_fail_closed() {
        let sum_f32: f32 = GAIN_BOUNDED_FIRST_TAP_64_BITS
            .iter()
            .fold(0.0_f32, |acc, &w| acc + f32::from_bits(w));
        assert_eq!(sum_f32.to_bits(), 0x3f80_0002);
        assert_eq!(ConvolverHighSrFirExt::required_f32_len(48_000.0, false), 0);
        assert_eq!(
            ConvolverHighSrFirExt::required_f32_len(44_100.0, false),
            6_144
        );
        assert_eq!(
            ConvolverHighSrFirExt::required_f32_len(192_000.0, true),
            6_144
        );

        let mut empty = [];
        assert!(ConvolverHighSrFirExt::from_slice(&mut empty, 48_000.0, false).is_ok());
        assert_eq!(
            ConvolverHighSrFirExt::from_slice(&mut empty, 96_000.0, false).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
        let mut short = [0.0_f32; 6_143];
        assert_eq!(
            ConvolverHighSrFirExt::from_slice(&mut short, 44_100.0, false).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
        assert_eq!(
            ConvolverHighSrFirExt::from_slice(&mut short, -1.0, false).unwrap_err(),
            VmError::InvalidSampleRate
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_owned_alloc_helper_matches_required_len() {
        assert_eq!(
            ConvolverHighSrFirExt::new_owned_for_rate(48_000.0, false)
                .unwrap()
                .mode(),
            ConvolverHighSrMode::Native48k
        );
        assert_eq!(
            ConvolverHighSrFirExt::new_owned_for_rate(96_000.0, false)
                .unwrap()
                .mode(),
            ConvolverHighSrMode::CaseA { stride_q: 2 }
        );
    }

    #[test]
    fn test_all_four_rate_cases_unit_dc_onset_and_reset() {
        let mut buf = [0.0_f32; EXT_FIR_STORAGE_F32_LEN];
        let (mut fir_l, mut fir_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        (fir_l[0], fir_l[1], fir_l[5], fir_r[1]) = (0.25, 0.50, 0.25, 1.0);

        // Case (a): 96 kHz (q = 2) -> sparse stride 2, onset = 32.
        let mut ext_a = ConvolverHighSrFirExt::from_slice(&mut buf, 96_000.0, false).unwrap();
        ext_a.prepare(&fir_l, &fir_r, 96_000.0, false).unwrap();
        assert_eq!(
            (ext_a.mode(), ext_a.onset_delay()),
            (ConvolverHighSrMode::CaseA { stride_q: 2 }, 32)
        );
        let mut onset_hit = false;
        for n in 0..40 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let ((el, er), (ol, _)) = ext_a.step_frame(x, x);
            if n == 2 {
                assert_eq!((el, er), (0.50, 1.0));
            }
            if n == 32 {
                assert_eq!(ol, 1.0);
                onset_hit = true;
            }
        }
        assert!(onset_hit);

        // Case (b): 44.1 kHz live 1x -> 0-ULP unit DC for tap-1 impulse and normalized sum.
        let mut ext_b = ConvolverHighSrFirExt::from_slice(&mut buf, 44_100.0, false).unwrap();
        let mut unit_tap1 = [0.0_f32; 32];
        unit_tap1[1] = 1.0;
        ext_b.prepare(&unit_tap1, &fir_l, 44_100.0, false).unwrap();
        assert_eq!(
            (ext_b.mode(), ext_b.onset_delay()),
            (ConvolverHighSrMode::CaseB, 15)
        );
        let ((dc_l, dc_r), (ol, _)) = step_dc(&mut ext_b, 96);
        assert_eq!((dc_l, ol), (1.0, 1.0));
        assert!(const_abs_f32(dc_r - 1.0) < 1.0e-5, "dc_r={dc_r}");
        ext_b.reset();
        assert_eq!(ext_b.step_frame(0.0, 0.0), ((0.0, 0.0), (0.0, 0.0)));

        // Case (c): 176.4 kHz 4x bounce -> exact u64 Lagrange decomposition, unit DC, onset = 59.
        let t1 = prepare_bounce_6point_u64(1, 176_400.0);
        let t127 = prepare_bounce_6point_u64(127, 352_800.0);
        assert_eq!((t1.m, t1.anchor, t127.m, t127.anchor), (3, 3, 933, 2));
        let mut ext_c = ConvolverHighSrFirExt::from_slice(&mut buf, 176_400.0, true).unwrap();
        ext_c.prepare(&fir_l, &fir_r, 176_400.0, true).unwrap();
        assert_eq!(
            (ext_c.mode(), ext_c.onset_delay()),
            (ConvolverHighSrMode::CaseC, 59)
        );
        let ((dc_cl, dc_cr), _) = step_dc(&mut ext_c, 80);
        assert!(const_abs_f32(dc_cl - 1.0) < 1.0e-6 && const_abs_f32(dc_cr - 1.0) < 1.0e-6);

        // Case (d): 88.2 kHz -> out_len = 59, onset = 29, unit DC, chunk parity.
        let mut ext_d = ConvolverHighSrFirExt::from_slice(&mut buf, 88_200.0, false).unwrap();
        ext_d.prepare(&fir_l, &fir_r, 88_200.0, false).unwrap();
        assert_eq!(
            (ext_d.mode(), ext_d.out_len(), ext_d.onset_delay()),
            (ConvolverHighSrMode::CaseD, 59, 29)
        );
        let (mut early_chk, mut onset_chk) = ([[0.0_f32; 2]; 4], [[0.0_f32; 2]; 4]);
        for _ in 0..20 {
            ext_d.step_chunk(&[[1.0, 1.0]; 4], &mut early_chk, &mut onset_chk);
        }
        assert!(const_abs_f32(early_chk[3][0] - 1.0) < 1.0e-5 && onset_chk[3][0] == 1.0);
    }
}
