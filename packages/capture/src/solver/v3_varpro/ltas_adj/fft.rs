#![forbid(unsafe_code)]
//! Radix-2 paired real-signal FFT plan, Hann Welch segmentation, and bin-to-band accumulation.

use std::f64::consts::PI;

use crate::metrics::companion::{LtasInterval, canonical_ltas_33_intervals_hz};

#[derive(Clone, Debug)]
pub(super) struct Radix2FftPlan {
    pub(super) n_fft: usize,
    bit_rev: Vec<usize>,
    tw_cos: Vec<f64>,
    tw_sin: Vec<f64>,
}

impl Radix2FftPlan {
    pub(super) fn new(n_fft: usize) -> Self {
        let mut bit_rev = vec![0usize; n_fft];
        let mut j = 0usize;
        for slot in bit_rev.iter_mut().skip(1) {
            let mut bit = n_fft >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            *slot = j;
        }
        let mut tw_cos = Vec::with_capacity(n_fft);
        let mut tw_sin = Vec::with_capacity(n_fft);
        let mut len = 2usize;
        while len <= n_fft {
            let half = len >> 1;
            for k in 0..half {
                let ang = -2.0 * PI * (k as f64) / (len as f64);
                let (s, c) = ang.sin_cos();
                tw_cos.push(c);
                tw_sin.push(s);
            }
            len <<= 1;
        }
        Self {
            n_fft,
            bit_rev,
            tw_cos,
            tw_sin,
        }
    }

    pub(super) fn fft_in_place(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n_fft;
        for (i, &j) in self.bit_rev.iter().enumerate() {
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut tw_base = 0usize;
        let mut len = 2usize;
        while len <= n {
            let half = len >> 1;
            let tw_c = &self.tw_cos[tw_base..tw_base + half];
            let tw_s = &self.tw_sin[tw_base..tw_base + half];
            let mut start = 0usize;
            while start < n {
                for k in 0..half {
                    let i0 = start + k;
                    let i1 = i0 + half;
                    let wr = tw_c[k];
                    let wi = tw_s[k];
                    let ur = re[i0];
                    let ui = im[i0];
                    let xr = re[i1];
                    let xi = im[i1];
                    let vr = xr * wr - xi * wi;
                    let vi = xr * wi + xi * wr;
                    re[i0] = ur + vr;
                    im[i0] = ui + vi;
                    re[i1] = ur - vr;
                    im[i1] = ui - vi;
                }
                start += len;
            }
            tw_base += half;
            len <<= 1;
        }
    }
}

/// Precomputed Welch-33 LTAS target band powers, bin mapping, and Hann window plan.
#[derive(Clone, Debug)]
pub struct Welch33TargetSummary {
    pub len: usize,
    pub warmup: usize,
    pub seg_len: usize,
    pub seg_starts: Vec<usize>,
    pub win: Vec<f64>,
    pub inv_win_norm: f64,
    pub inv_wins: f64,
    pub bin_band: Vec<u8>,
    pub active: [bool; 33],
    pub active_count: usize,
    pub tgt_band_p: [f64; 33],
    pub floor: f64,
    pub(super) plan: Radix2FftPlan,
}

/// Reusable scratch buffers avoiding per-call allocations in Welch-33 LTAS forward/adjoint passes.
#[derive(Clone, Debug)]
pub struct Welch33Scratch {
    pub(super) win_re: Vec<f64>,
    pub(super) win_im: Vec<f64>,
    pub(super) fft_re: Vec<f64>,
    pub(super) fft_im: Vec<f64>,
    pub(super) band_p: [f64; 33],
    pub(super) d_band: [f64; 33],
}

impl Default for Welch33Scratch {
    fn default() -> Self {
        Self {
            win_re: Vec::new(),
            win_im: Vec::new(),
            fft_re: Vec::new(),
            fft_im: Vec::new(),
            band_p: [0.0_f64; 33],
            d_band: [0.0_f64; 33],
        }
    }
}

fn compute_seg_starts(len_post: usize, win_cap: usize) -> (usize, Vec<usize>) {
    if len_post == 0 {
        return (0, Vec::new());
    }
    let seg_len = len_post.min(win_cap);
    let hop = (seg_len / 2).max(1);
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + seg_len <= len_post {
        starts.push(start);
        start += hop;
    }
    let last_start = len_post - seg_len;
    if last_start + hop != start {
        starts.push(last_start);
    }
    (seg_len, starts)
}

fn build_hann_window(seg_len: usize) -> (Vec<f64>, f64) {
    let mut win = vec![1.0_f64; seg_len];
    let mut norm = seg_len as f64;
    if seg_len >= 8 {
        norm = 0.0;
        let denom = (seg_len - 1) as f64;
        for (i, w) in win.iter_mut().enumerate() {
            let v = 0.5 * (1.0 - (2.0 * PI * (i as f64) / denom).cos());
            *w = v;
            norm += v * v;
        }
    }
    (win, 1.0 / norm.max(1.0e-12))
}

fn build_bin_band_map(
    n_fft: usize,
    sample_rate_hz: f64,
    max_freq_hz: f64,
    intervals: &[LtasInterval; 33],
) -> Vec<u8> {
    let half_plus_1 = n_fft / 2 + 1;
    let mut map = vec![u8::MAX; half_plus_1];
    let df = sample_rate_hz / (n_fft as f64);
    for (k, slot) in map.iter_mut().enumerate() {
        let f_hz = (k as f64) * df;
        if f_hz < 20.0 {
            continue;
        }
        if f_hz > max_freq_hz {
            break;
        }
        for (b, iv) in intervals.iter().enumerate() {
            if iv.high_hz <= max_freq_hz + 1.0e-9 && iv.contains(f_hz) {
                *slot = b as u8;
                break;
            }
        }
    }
    map
}

/// Builds a [`Welch33TargetSummary`] matching canonical `companion.rs` Welch-33 binning.
#[must_use]
pub fn build_welch33_target_summary(
    tgt: &[f64],
    warmup: usize,
    sample_rate_hz: f64,
) -> Welch33TargetSummary {
    let len = tgt.len();
    let w_clamped = warmup.min(len);
    let len_post = len.saturating_sub(w_clamped);
    let max_freq_hz = if sample_rate_hz >= 40_000.0 {
        20_000.0
    } else {
        (0.48 * sample_rate_hz).min(20_000.0)
    };
    let rate_scale = (sample_rate_hz / 48_000.0).ceil().max(1.0) as usize;
    let win_cap = 16_384usize.saturating_mul(rate_scale.next_power_of_two());
    let n_fft = (win_cap * 2).max((sample_rate_hz.ceil() as usize).next_power_of_two());
    let (seg_len, seg_starts) = compute_seg_starts(len_post, win_cap);
    let (win, inv_win_norm) = build_hann_window(seg_len);
    let intervals = canonical_ltas_33_intervals_hz();
    let bin_band = build_bin_band_map(n_fft, sample_rate_hz, max_freq_hz, &intervals);
    let plan = Radix2FftPlan::new(n_fft);
    let n_wins = seg_starts.len();
    let inv_wins = if n_wins > 0 {
        1.0 / (n_wins as f64)
    } else {
        0.0
    };

    let mut tgt_band_p = [0.0_f64; 33];
    if n_wins > 0 && seg_len > 0 {
        let mut fft_re = vec![0.0_f64; n_fft];
        let mut fft_im = vec![0.0_f64; n_fft];
        let scale = inv_wins * inv_win_norm;
        let mut w_idx = 0usize;
        while w_idx < n_wins {
            let st0 = seg_starts[w_idx];
            let seg0 = &tgt[w_clamped + st0..w_clamped + st0 + seg_len];
            let seg1 = if w_idx + 1 < n_wins {
                let st1 = seg_starts[w_idx + 1];
                Some(&tgt[w_clamped + st1..w_clamped + st1 + seg_len])
            } else {
                None
            };
            accum_paired_forward_fft(
                &plan,
                seg_len,
                &win,
                &bin_band,
                seg0,
                seg1,
                &mut fft_re,
                &mut fft_im,
                scale,
                &mut tgt_band_p,
                None,
                None,
            );
            w_idx += 2;
        }
    }

    let peak_band_p = tgt_band_p.iter().copied().fold(0.0_f64, f64::max);
    let fft_leak_floor = peak_band_p * 1.0e-18;
    let mut active = [false; 33];
    let mut active_count = 0usize;
    for b in 0..33 {
        if intervals[b].high_hz <= max_freq_hz + 1.0e-9 && tgt_band_p[b] > fft_leak_floor {
            active[b] = true;
            active_count += 1;
        }
    }
    let floor = (peak_band_p * 1.0e-15).max(1.0e-24);

    Welch33TargetSummary {
        len,
        warmup: w_clamped,
        seg_len,
        seg_starts,
        win,
        inv_win_norm,
        inv_wins,
        bin_band,
        active,
        active_count,
        tgt_band_p,
        floor,
        plan,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accum_paired_forward_fft(
    plan: &Radix2FftPlan,
    seg_len: usize,
    win: &[f64],
    bin_band: &[u8],
    s0: &[f64],
    s1_opt: Option<&[f64]>,
    fft_re: &mut [f64],
    fft_im: &mut [f64],
    scale: f64,
    band_p: &mut [f64; 33],
    mut win_out0: Option<(&mut [f64], &mut [f64])>,
    mut win_out1: Option<(&mut [f64], &mut [f64])>,
) {
    let n_fft = plan.n_fft;
    let half = n_fft / 2;
    fft_re.fill(0.0);
    fft_im.fill(0.0);
    for i in 0..seg_len {
        fft_re[i] = s0[i] * win[i];
    }
    if let Some(s1) = s1_opt {
        for i in 0..seg_len {
            fft_im[i] = s1[i] * win[i];
        }
    }
    plan.fft_in_place(fft_re, fft_im);
    for (k, &band_u8) in bin_band.iter().enumerate() {
        let (r0, i0, r1, i1, one_sided) = if k == 0 || k == half {
            (fft_re[k], 0.0, fft_im[k], 0.0, 1.0)
        } else {
            let nk = n_fft - k;
            (
                0.5 * (fft_re[k] + fft_re[nk]),
                0.5 * (fft_im[k] - fft_im[nk]),
                0.5 * (fft_im[k] + fft_im[nk]),
                0.5 * (fft_re[nk] - fft_re[k]),
                2.0,
            )
        };
        if let Some((ref mut wr0, ref mut wi0)) = win_out0 {
            wr0[k] = r0;
            wi0[k] = i0;
        }
        if let Some((ref mut wr1, ref mut wi1)) = win_out1 {
            wr1[k] = r1;
            wi1[k] = i1;
        }
        let b = band_u8 as usize;
        if b < 33 {
            let mag_sq = (r0 * r0 + i0 * i0)
                + if s1_opt.is_some() {
                    r1 * r1 + i1 * i1
                } else {
                    0.0
                };
            band_p[b] += mag_sq * scale * one_sided;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accum_paired_backward_fft(
    plan: &Radix2FftPlan,
    seg_len: usize,
    win: &[f64],
    bin_band: &[u8],
    active: &[bool; 33],
    d_band: &[f64; 33],
    power_scale: f64,
    win_re: &[f64],
    win_im: &[f64],
    off0: usize,
    off1_opt: Option<usize>,
    fft_re: &mut [f64],
    fft_im: &mut [f64],
    dy_full: &mut [f64],
    base0: usize,
    base1_opt: Option<usize>,
) {
    let n_fft = plan.n_fft;
    let half = n_fft / 2;
    let s_base = 2.0 * power_scale;
    fft_re.fill(0.0);
    fft_im.fill(0.0);
    for (k, &band_u8) in bin_band.iter().enumerate() {
        let b = band_u8 as usize;
        if b >= 33 || !active[b] {
            continue;
        }
        let c = s_base * d_band[b];
        if k == 0 || k == half {
            fft_re[k] = c * win_re[off0 + k];
            if let Some(off1) = off1_opt {
                fft_im[k] = c * win_re[off1 + k];
            }
        } else {
            let nk = n_fft - k;
            let h0_re = c * win_re[off0 + k];
            let h0_im = -c * win_im[off0 + k];
            let (h1_re, h1_im) = if let Some(off1) = off1_opt {
                (c * win_re[off1 + k], -c * win_im[off1 + k])
            } else {
                (0.0, 0.0)
            };
            fft_re[k] = h0_re - h1_im;
            fft_im[k] = h0_im + h1_re;
            fft_re[nk] = h0_re + h1_im;
            fft_im[nk] = -h0_im + h1_re;
        }
    }
    plan.fft_in_place(fft_re, fft_im);
    for (dst, (&w, &v)) in dy_full[base0..base0 + seg_len]
        .iter_mut()
        .zip(win.iter().zip(fft_re.iter()))
    {
        *dst += w * v;
    }
    if let Some(base1) = base1_opt {
        for (dst, (&w, &v)) in dy_full[base1..base1 + seg_len]
            .iter_mut()
            .zip(win.iter().zip(fft_im.iter()))
        {
            *dst += w * v;
        }
    }
}
