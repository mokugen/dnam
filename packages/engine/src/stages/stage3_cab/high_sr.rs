//! Cold Stage III Early FIR & Kautz Onset Extension (`stage3_cab/high_sr.rs`).
//!
//! Handles all non-`48 kHz` rates via the four mutually exclusive cases of
//! `docs/specs/dsp/router_and_crossbar_topology.md` §9 (`ref[dsp.zoh.cabinet-fir-onset]`):
//! - **Case (a):** Exact integer multiples `q * 48,000 Hz` (`q in {2, 4, 8}` = `96 / 192 / 384 kHz`),
//!   sparse integer-stride taps `h[k] * x[n - q*k]` across all `32/48/96/128` base taps + `16*q` onset.
//! - **Case (b):** Live `1x` (`os_factor == 1`, `44,100 <= fs_eff < 48,000`), `64`-tap first-tap
//!   difference filter + ZOH-rescaled remaining taps `1..num_taps`.
//! - **Case (c):** `4x` offline bounce (`is_bounce_4x`, `176,400 <= fs_eff <= 384,000`), 6-point
//!   Lagrange difference mapping via exact `u128` IEEE-754 integer decomposition across all `1..128`
//!   base taps (never truncating `Full/Pro/Ultra` taps > `63`).
//! - **Case (d):** All remaining `(f_host, M_os)` pairs (`88.2 kHz 2x`, arbitrary host rates),
//!   DC-normalized Catmull-Rom FIR resampling up to `<= 1,024` taps.

use alloc::boxed::Box;

use super::kautz_zoh::compute_onset_delay_samples;
use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, SimdVec4, newton6_reciprocal_f64};
use crate::stages::sona_core::model::SONA_FIR_MAX_TAPS;
use crate::stages::topology::{
    eval_bounce_6point_lagrange_tap, eval_live_first_tap_difference,
    prepare_live_first_tap_difference_coeffs,
};

/// Cold high-SR / ZOH FIR delay ring capacity (`1,024` frames, double-mirrored to `2,048`).
pub const STAGE3_EXT_FIR_CAP: usize = 1_024;
/// Power-of-two mask (`1,023`) for [`STAGE3_EXT_FIR_CAP`].
pub const STAGE3_EXT_FIR_MASK: usize = STAGE3_EXT_FIR_CAP - 1;

const _: () = assert!(STAGE3_EXT_FIR_CAP.is_power_of_two());
const _: () = assert!(8 * (SONA_FIR_MAX_TAPS - 1) + 4 <= STAGE3_EXT_FIR_CAP);

#[derive(Clone, Copy, Debug, Default)]
struct BounceLagrangeTap {
    tap_l: f32,
    tap_r: f32,
    m: u16,
    anchor_a: u8,
    p_coeffs: [f32; 5],
}

#[derive(Clone, Debug)]
enum ColdFirMode {
    SparseStride {
        stride: usize,
        num_taps: usize,
    },
    Live1xDiff {
        tap0_l: f32,
        tap0_r: f32,
        tap1_l: f32,
        tap1_r: f32,
        rest_segs: usize,
    },
    Bounce4xLagrange {
        tap0_l: f32,
        tap0_r: f32,
        num_taps: usize,
    },
    DenseResampled {
        num_segs: usize,
    },
}

/// Cold-allocated Stage III extended FIR + onset evaluator for `effective_sr != 48_000 Hz`.
#[derive(Clone, Debug)]
pub struct Stage3HighSrFirExt {
    hist_l: Box<[f32; STAGE3_EXT_FIR_CAP * 2]>,
    hist_r: Box<[f32; STAGE3_EXT_FIR_CAP * 2]>,
    bounce_taps: Box<[BounceLagrangeTap; SONA_FIR_MAX_TAPS]>,
    diff_coeffs: Box<[f32; 63]>,
    dense_l: Box<[f32; STAGE3_EXT_FIR_CAP]>,
    dense_r: Box<[f32; STAGE3_EXT_FIR_CAP]>,
    write_pos: u32,
    pub onset_delay_samples: usize,
    mode: ColdFirMode,
}

impl Stage3HighSrFirExt {
    /// Pre-allocates all cold extended FIR scratch buffers (`0` subsequent `reconfigure` allocations).
    #[must_use]
    pub fn new_preallocated() -> Self {
        Self {
            hist_l: Box::new([0.0; STAGE3_EXT_FIR_CAP * 2]),
            hist_r: Box::new([0.0; STAGE3_EXT_FIR_CAP * 2]),
            bounce_taps: Box::new([BounceLagrangeTap::default(); SONA_FIR_MAX_TAPS]),
            diff_coeffs: Box::new([0.0; 63]),
            dense_l: Box::new([0.0; STAGE3_EXT_FIR_CAP]),
            dense_r: Box::new([0.0; STAGE3_EXT_FIR_CAP]),
            write_pos: 0,
            onset_delay_samples: 16,
            mode: ColdFirMode::SparseStride {
                stride: 1,
                num_taps: 16,
            },
        }
    }

    /// Allocates and configures cold extended FIR evaluator for `(base_l, base_r, num_base_taps, os_factor, is_bounce_4x, zoh)`.
    #[must_use]
    pub fn from_zoh(
        base_l: &[f32; SONA_FIR_MAX_TAPS],
        base_r: &[f32; SONA_FIR_MAX_TAPS],
        num_base_taps: usize,
        os_factor: usize,
        is_bounce_4x: bool,
        zoh: &ZohRescaledParams,
    ) -> Self {
        let mut out = Self::new_preallocated();
        out.reconfigure(base_l, base_r, num_base_taps, os_factor, is_bounce_4x, zoh);
        out
    }

    /// Reconfigures cold extended FIR taps & mode without reallocating existing history ring memory.
    pub fn reconfigure(
        &mut self,
        base_l: &[f32; SONA_FIR_MAX_TAPS],
        base_r: &[f32; SONA_FIR_MAX_TAPS],
        num_base_taps: usize,
        os_factor: usize,
        is_bounce_4x: bool,
        zoh: &ZohRescaledParams,
    ) {
        let active_base = num_base_taps.clamp(1, SONA_FIR_MAX_TAPS);
        let fs_eff = zoh.effective_sample_rate_hz();
        if let Some(q) = detect_integer_multiple_48k(fs_eff) {
            self.onset_delay_samples = (16 * q).clamp(14, STAGE3_EXT_FIR_MASK);
            self.dense_l[..SONA_FIR_MAX_TAPS].copy_from_slice(base_l);
            self.dense_r[..SONA_FIR_MAX_TAPS].copy_from_slice(base_r);
            self.mode = ColdFirMode::SparseStride {
                stride: q,
                num_taps: active_base,
            };
            return;
        }

        if os_factor == 1
            && !is_bounce_4x
            && (44_100.0..48_000.0).contains(&fs_eff)
            && let Some(diff_coeffs) = prepare_live_first_tap_difference_coeffs(fs_eff)
        {
            self.onset_delay_samples = compute_onset_delay_samples(zoh);
            *self.diff_coeffs = diff_coeffs;
            self.dense_l[..SONA_FIR_MAX_TAPS].fill(0.0);
            self.dense_r[..SONA_FIR_MAX_TAPS].fill(0.0);
            if active_base > 2 {
                let mut tail_l = [0.0_f32; SONA_FIR_MAX_TAPS + 1];
                let mut tail_r = [0.0_f32; SONA_FIR_MAX_TAPS + 1];
                tail_l[2..active_base].copy_from_slice(&base_l[2..active_base]);
                tail_r[2..active_base].copy_from_slice(&base_r[2..active_base]);
                let in_len = active_base + 1;
                zoh.rescale_fir_taps(&tail_l[..in_len], &mut self.dense_l[..active_base]);
                zoh.rescale_fir_taps(&tail_r[..in_len], &mut self.dense_r[..active_base]);
            }
            let rest_segs = if active_base > 2 {
                active_base.div_ceil(16).min(SONA_FIR_MAX_TAPS / 16)
            } else {
                0
            };
            self.mode = ColdFirMode::Live1xDiff {
                tap0_l: base_l[0],
                tap0_r: base_r[0],
                tap1_l: if active_base > 1 { base_l[1] } else { 0.0 },
                tap1_r: if active_base > 1 { base_r[1] } else { 0.0 },
                rest_segs,
            };
            return;
        }

        if is_bounce_4x && (176_400.0..=384_000.0).contains(&fs_eff) {
            self.onset_delay_samples = compute_onset_delay_samples(zoh);
            self.bounce_taps.fill(BounceLagrangeTap::default());
            for k in 1..active_base {
                if let Some((m, anchor_a, p_coeffs)) = prepare_bounce_6point_u128(k as u8, fs_eff) {
                    self.bounce_taps[k] = BounceLagrangeTap {
                        tap_l: base_l[k],
                        tap_r: base_r[k],
                        m: m as u16,
                        anchor_a: anchor_a as u8,
                        p_coeffs,
                    };
                }
            }
            self.mode = ColdFirMode::Bounce4xLagrange {
                tap0_l: base_l[0],
                tap0_r: base_r[0],
                num_taps: active_base,
            };
            return;
        }

        self.onset_delay_samples = compute_onset_delay_samples(zoh);
        let inv_rho = newton6_reciprocal_f64(zoh.rho_sr.max(1.0e-6));
        let target_taps = crate::math::ceil_f64((active_base as f64) * inv_rho) as usize;
        let out_len = target_taps.clamp(active_base, STAGE3_EXT_FIR_CAP);
        let num_segs = out_len.div_ceil(16).min(STAGE3_EXT_FIR_CAP / 16);
        self.dense_l.fill(0.0);
        self.dense_r.fill(0.0);
        zoh.rescale_fir_taps(&base_l[..active_base], &mut self.dense_l[..out_len]);
        zoh.rescale_fir_taps(&base_r[..active_base], &mut self.dense_r[..out_len]);
        self.mode = ColdFirMode::DenseResampled { num_segs };
    }

    /// Zeros history ring without changing FIR mode or taps.
    pub fn reset(&mut self) {
        self.hist_l.fill(0.0);
        self.hist_r.fill(0.0);
        self.write_pos = 0;
    }

    #[inline(always)]
    fn push_sample(&mut self, l: f32, r: f32) -> usize {
        let pos = (self.write_pos as usize).wrapping_sub(1) & STAGE3_EXT_FIR_MASK;
        self.write_pos = pos as u32;
        self.hist_l[pos] = l;
        self.hist_l[pos + STAGE3_EXT_FIR_CAP] = l;
        self.hist_r[pos] = r;
        self.hist_r[pos + STAGE3_EXT_FIR_CAP] = r;
        pos
    }

    /// Evaluates one frame across the active cold FIR mode, always advancing both L and R paths.
    #[inline(always)]
    pub fn step_frame<B: MathBackend, const MONO: bool>(
        &mut self,
        in_l: f32,
        in_r: f32,
    ) -> ([f32; 2], f32, f32) {
        let pos = self.push_sample(in_l, in_r);
        let del_l = self.hist_l[pos + self.onset_delay_samples];
        let del_r = self.hist_r[pos + self.onset_delay_samples];
        let fir = match &self.mode {
            ColdFirMode::SparseStride { stride, num_taps } => {
                let (mut al, mut ar) = (0.0_f32, 0.0_f32);
                let q = *stride;
                for k in 0..*num_taps {
                    let off = pos + k * q;
                    al += self.dense_l[k] * self.hist_l[off];
                    ar += self.dense_r[k] * self.hist_r[off];
                }
                [al, ar]
            }
            ColdFirMode::Live1xDiff {
                tap0_l,
                tap0_r,
                tap1_l,
                tap1_r,
                rest_segs,
            } => {
                let mut h64_l = [0.0_f32; 64];
                let mut h64_r = [0.0_f32; 64];
                h64_l.copy_from_slice(&self.hist_l[pos..pos + 64]);
                h64_r.copy_from_slice(&self.hist_r[pos..pos + 64]);
                let mut fl = *tap0_l * self.hist_l[pos]
                    + *tap1_l * eval_live_first_tap_difference(&h64_l, &self.diff_coeffs);
                let mut fr = *tap0_r * self.hist_r[pos]
                    + *tap1_r * eval_live_first_tap_difference(&h64_r, &self.diff_coeffs);
                if *rest_segs > 0 {
                    let (rl, rr) = dot_segments::<B>(
                        &self.hist_l[pos..],
                        &self.hist_r[pos..],
                        &self.dense_l[..SONA_FIR_MAX_TAPS],
                        &self.dense_r[..SONA_FIR_MAX_TAPS],
                        *rest_segs,
                    );
                    fl += rl;
                    fr += rr;
                }
                [fl, fr]
            }
            ColdFirMode::Bounce4xLagrange {
                tap0_l,
                tap0_r,
                num_taps,
            } => {
                let mut fl = *tap0_l * self.hist_l[pos];
                let mut fr = *tap0_r * self.hist_r[pos];
                let hl = &self.hist_l[pos..];
                let hr = &self.hist_r[pos..];
                for t in self.bounce_taps[1..*num_taps].iter() {
                    let m = t.m as usize;
                    let a = t.anchor_a as usize;
                    fl += t.tap_l * eval_bounce_6point_lagrange_tap(hl, m, a, &t.p_coeffs);
                    fr += t.tap_r * eval_bounce_6point_lagrange_tap(hr, m, a, &t.p_coeffs);
                }
                [fl, fr]
            }
            ColdFirMode::DenseResampled { num_segs } => {
                let (fl, fr) = dot_segments::<B>(
                    &self.hist_l[pos..],
                    &self.hist_r[pos..],
                    &self.dense_l[..],
                    &self.dense_r[..],
                    *num_segs,
                );
                [fl, fr]
            }
        };
        let _ = MONO;
        (fir, del_l, del_r)
    }
}

#[inline(always)]
fn dot_segments<B: MathBackend>(
    hist_l: &[f32],
    hist_r: &[f32],
    taps_l: &[f32],
    taps_r: &[f32],
    num_segs: usize,
) -> (f32, f32) {
    let mut acc_l = B::Vec4::zero();
    let mut acc_r = B::Vec4::zero();
    for seg in 0..num_segs {
        let s = seg << 4;
        let (hl, _) = hist_l[s..s + 16].as_chunks::<4>();
        let (tl, _) = taps_l[s..s + 16].as_chunks::<4>();
        let f0 = B::Vec4::load(&hl[0]).mul(B::Vec4::load(&tl[0]));
        let f1 = B::Vec4::load(&hl[1]).mul(B::Vec4::load(&tl[1]));
        acc_l = acc_l
            .add(f0.fma(B::Vec4::load(&hl[2]), B::Vec4::load(&tl[2])))
            .add(f1.fma(B::Vec4::load(&hl[3]), B::Vec4::load(&tl[3])));
        let (hr, _) = hist_r[s..s + 16].as_chunks::<4>();
        let (tr, _) = taps_r[s..s + 16].as_chunks::<4>();
        let g0 = B::Vec4::load(&hr[0]).mul(B::Vec4::load(&tr[0]));
        let g1 = B::Vec4::load(&hr[1]).mul(B::Vec4::load(&tr[1]));
        acc_r = acc_r
            .add(g0.fma(B::Vec4::load(&hr[2]), B::Vec4::load(&tr[2])))
            .add(g1.fma(B::Vec4::load(&hr[3]), B::Vec4::load(&tr[3])));
    }
    acc_l.horizontal_sum_pair(acc_r)
}

#[inline]
fn detect_integer_multiple_48k(fs_eff: f64) -> Option<usize> {
    [1usize, 2, 4, 8]
        .into_iter()
        .find(|&q| (fs_eff - (q as f64) * 48_000.0).abs() < 1.0e-3)
}

/// Computes `prepare_bounce_6point_lagrange_tap` via exact IEEE-754 `u64` decomposition and
/// an 11-bit shift-subtract division (`0 udivti3`, `0 udiv`, `0 fdiv`, `tap_k in 1..128`,
/// `<= 384 kHz`).
#[inline]
fn prepare_bounce_6point_u128(
    tap_k: u8,
    effective_rate_hz: f64,
) -> Option<(usize, usize, [f32; 5])> {
    if !(1..=127).contains(&tap_k)
        || !effective_rate_hz.is_finite()
        || !(176_400.0..=384_000.0).contains(&effective_rate_hz)
    {
        return None;
    }
    let bits = effective_rate_hz.to_bits();
    let exponent = (((bits >> 52) & 0x7ff) as u32).saturating_sub(1023);
    if exponent > 52 {
        return None;
    }
    // `exponent in {17, 18}` -> `significand < 2^53`, `denominator < 2^51`, `numerator < 2^60`.
    let significand = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let denominator = 48_000u64 << (52 - exponent);
    let numerator = u64::from(tap_k) * significand;
    let mut m = 0usize;
    let mut rem = numerator;
    let mut shift = 11usize;
    while shift > 0 {
        shift -= 1;
        let d_shifted = denominator << shift;
        if rem >= d_shifted {
            rem -= d_shifted;
            m |= 1usize << shift;
        }
    }
    let mu = (rem as f64) * newton6_reciprocal_f64(denominator as f64);
    let anchor_a = if 2 * rem <= denominator {
        2usize
    } else {
        3usize
    };

    const INV_120: f64 = 1.0 / 120.0;
    let nodes = [-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0];
    let scales = [-1.0_f64, 5.0, -10.0, 10.0, -5.0, 1.0];
    let mut h = [0.0_f64; 6];
    for i in 0..6 {
        let mut prod = scales[i];
        for (l, &vl) in nodes.iter().enumerate() {
            if l != i {
                prod *= mu - vl;
            }
        }
        h[i] = prod * INV_120;
    }
    let mut prefix = 0.0_f64;
    let mut p_coeffs = [0.0_f32; 5];
    for (j, slot) in p_coeffs.iter_mut().enumerate() {
        prefix += h[j];
        let sub = if j >= anchor_a { 1.0 } else { 0.0 };
        *slot = (prefix - sub) as f32;
    }
    Some((m, anchor_a, p_coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    #[test]
    fn test_stage3_high_sr_192k_384k_176k4_88k2_and_live_44k1() {
        let mut fir_l = [0.0_f32; SONA_FIR_MAX_TAPS];
        let mut fir_r = [0.0_f32; SONA_FIR_MAX_TAPS];
        fir_l[127] = 0.75;
        fir_r[0] = 0.5;
        fir_r[127] = -0.25;

        // Case (a) 4x = 192 kHz: Ultra tap 127 fires at exact sample 127 * 4 = 508 without truncation.
        let zoh_192 = ZohRescaledParams::new(48_000, 192_000);
        let mut ext_192 = Stage3HighSrFirExt::from_zoh(&fir_l, &fir_r, 128, 1, false, &zoh_192);
        assert_eq!(ext_192.onset_delay_samples, 64);
        let mut hit_508 = 0.0_f32;
        for n in 0..520 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let (out, _, _) = ext_192.step_frame::<DefaultMathBackend, false>(x, x);
            if n == 508 {
                hit_508 = out[0];
            }
        }
        assert!((hit_508 - 0.75).abs() < 1e-6);

        // Case (a) 8x = 384 kHz: tap 127 at sample 1016, onset = 128.
        let zoh_384 = ZohRescaledParams::new(48_000, 384_000);
        let mut ext_384 = Stage3HighSrFirExt::from_zoh(&fir_l, &fir_r, 128, 1, false, &zoh_384);
        assert_eq!(ext_384.onset_delay_samples, 128);
        let (o0, _, _) = ext_384.step_frame::<DefaultMathBackend, false>(1.0, 1.0);
        assert!((o0[1] - 0.5).abs() < 1e-6);

        // Case (c) 4x bounce at 176.4 kHz supports tap_k = 127 via u128 decomposition.
        let zoh_176 = ZohRescaledParams::new(48_000, 176_400);
        let mut ext_176 = Stage3HighSrFirExt::from_zoh(&fir_l, &fir_r, 128, 4, true, &zoh_176);
        let mut max_dc = 0.0_f32;
        for _ in 0..700 {
            let (out, _, _) = ext_176.step_frame::<DefaultMathBackend, false>(1.0, 1.0);
            max_dc = out[0];
        }
        assert!((max_dc - 0.75).abs() < 1e-4);

        // Case (d) 88.2 kHz 2x expands Ultra 128 taps cleanly beyond 128 without truncation.
        let zoh_88 = ZohRescaledParams::new(48_000, 88_200);
        let mut ext_88 = Stage3HighSrFirExt::from_zoh(&fir_l, &fir_r, 128, 2, false, &zoh_88);
        let mut dc_88 = 0.0_f32;
        for _ in 0..300 {
            let (out, _, _) = ext_88.step_frame::<DefaultMathBackend, false>(1.0, 1.0);
            dc_88 = out[0];
        }
        assert!((dc_88 - 0.75).abs() < 1e-4);

        // Case (b) live 1x 44.1 kHz preserves tap-0 exact impulse, tap-1 20 kHz tracking,
        // convergence toward 48 kHz, exact DC gain, and Full vs Lite prefix parity.
        let zoh_44 = ZohRescaledParams::new(48_000, 44_100);
        let mut base_l = [0.0_f32; SONA_FIR_MAX_TAPS];
        base_l[0] = 0.5;
        base_l[1] = 0.25;
        base_l[2] = 0.25;
        let mut ext_44 = Stage3HighSrFirExt::from_zoh(&base_l, &base_l, 32, 1, false, &zoh_44);
        let (imp0, _, _) = ext_44.step_frame::<DefaultMathBackend, false>(1.0, 1.0);
        assert!((imp0[0] - 0.5).abs() < 0.08);
        let mut sum_imp = imp0[0];
        for _ in 1..64 {
            let (out, _, _) = ext_44.step_frame::<DefaultMathBackend, false>(0.0, 0.0);
            sum_imp += out[0];
        }
        assert!((sum_imp - 1.0).abs() < 1e-4);

        let mut ext_44_dc = Stage3HighSrFirExt::from_zoh(&base_l, &base_l, 32, 1, false, &zoh_44);
        let mut ext_44_full = Stage3HighSrFirExt::from_zoh(&base_l, &base_l, 64, 1, false, &zoh_44);
        let mut dc_44 = 0.0_f32;
        for n in 0..96 {
            let (out, _, _) = ext_44_dc.step_frame::<DefaultMathBackend, false>(0.25, 0.25);
            let (out_f, _, _) = ext_44_full.step_frame::<DefaultMathBackend, false>(0.25, 0.25);
            assert!(
                (out[0] - out_f[0]).abs() < 1e-6,
                "Lite vs Full 44.1k live parity mismatch at n={n}: {} vs {}",
                out[0],
                out_f[0]
            );
            dc_44 = out[0];
        }
        assert!((dc_44 - 0.25).abs() < 1e-5);

        let mut tap0_only = [0.0_f32; SONA_FIR_MAX_TAPS];
        tap0_only[0] = 0.8;
        let mut ext_tap0 =
            Stage3HighSrFirExt::from_zoh(&tap0_only, &tap0_only, 32, 1, false, &zoh_44);
        let (t0, _, _) = ext_tap0.step_frame::<DefaultMathBackend, false>(1.0, 1.0);
        let (t1, _, _) = ext_tap0.step_frame::<DefaultMathBackend, false>(0.0, 0.0);
        assert!((t0[0] - 0.8).abs() < 1e-6);
        assert!(t1[0].abs() < 1e-6);

        let mut tap1_only = [0.0_f32; SONA_FIR_MAX_TAPS];
        tap1_only[1] = 1.0;
        let mut ext_tap1_44 =
            Stage3HighSrFirExt::from_zoh(&tap1_only, &tap1_only, 32, 1, false, &zoh_44);
        let zoh_47999 = ZohRescaledParams::new(48_000, 47_999);
        let mut ext_tap1_near48 =
            Stage3HighSrFirExt::from_zoh(&tap1_only, &tap1_only, 32, 1, false, &zoh_47999);
        let omega = core::f64::consts::TAU * 20_000.0 / 44_100.0;
        let mut max_err_20k = 0.0_f32;
        for n in 0..160 {
            let x = (omega * (n as f64)).sin() as f32;
            let (out, _, _) = ext_tap1_44.step_frame::<DefaultMathBackend, false>(x, x);
            if n >= 80 {
                let target = ((omega * (n as f64) - core::f64::consts::TAU * 20_000.0 / 48_000.0)
                    .sin()) as f32;
                max_err_20k = max_err_20k.max((out[0] - target).abs());
            }
        }
        assert!(
            max_err_20k < 0.08,
            "tap1 20kHz error too large: {max_err_20k}"
        );

        let (n0, _, _) = ext_tap1_near48.step_frame::<DefaultMathBackend, false>(1.0, 1.0);
        let (n1, _, _) = ext_tap1_near48.step_frame::<DefaultMathBackend, false>(0.0, 0.0);
        assert!(n0[0].abs() < 1e-3 && (n1[0] - 1.0).abs() < 1e-3);
    }
}
