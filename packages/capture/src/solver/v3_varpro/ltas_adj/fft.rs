#![forbid(unsafe_code)]
//! Radix-2 paired real-signal FFT plan, Hann Welch segmentation, and bin-to-band accumulation.

use std::f64::consts::PI;

use crate::metrics::companion::{LtasInterval, canonical_ltas_33_intervals_hz};

mod radix4;
pub(super) use radix4::Radix2FftPlan;

const NYQ_STOPBAND_TAG: u8 = 33;

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
    pub tgt_nyq_p: f64,
    pub floor: f64,
    pub(super) plan: Radix2FftPlan,
}

/// Per-window-pair scratch buffers allowing lock-free parallel Welch-33 FFT forward and backward passes.
#[derive(Default, Debug)]
pub(super) struct WelchPairScratch {
    pub(super) fft_re: Vec<f64>,
    pub(super) fft_im: Vec<f64>,
    pub(super) win_re: Vec<f64>,
    pub(super) win_im: Vec<f64>,
    pub(super) dy_pair: Vec<f64>,
}

/// Reusable scratch buffers avoiding per-call allocations in Welch-33 LTAS forward/adjoint passes.
#[derive(Debug)]
pub struct Welch33Scratch {
    pub(super) win_re: Vec<f64>,
    pub(super) win_im: Vec<f64>,
    pub(super) fft_re: Vec<f64>,
    pub(super) fft_im: Vec<f64>,
    pub(in crate::solver) band_p: [f64; 33],
    pub(super) d_band: [f64; 33],
    pub(crate) last_max_abs_db: f64,
    pub(super) pair_pool: Vec<std::sync::Mutex<WelchPairScratch>>,
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
            last_max_abs_db: 0.0,
            pair_pool: Vec::new(),
        }
    }
}

impl Clone for Welch33Scratch {
    fn clone(&self) -> Self {
        Self {
            win_re: self.win_re.clone(),
            win_im: self.win_im.clone(),
            fft_re: self.fft_re.clone(),
            fft_im: self.fft_im.clone(),
            band_p: self.band_p,
            d_band: self.d_band,
            last_max_abs_db: self.last_max_abs_db,
            pair_pool: Vec::new(),
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
            *slot = NYQ_STOPBAND_TAG;
            continue;
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
    let mut tgt_nyq_p = 0.0_f64;
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
                Some(&mut tgt_nyq_p),
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
        tgt_nyq_p,
        floor,
        plan,
    }
}

impl Welch33TargetSummary {
    #[cfg(test)]
    pub(crate) fn n_fft(&self) -> usize {
        self.plan.n_fft
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
    mut nyq_p: Option<&mut f64>,
    mut win_out0: Option<(&mut [f64], &mut [f64])>,
    mut win_out1: Option<(&mut [f64], &mut [f64])>,
) {
    let n_fft = plan.n_fft;
    let half = n_fft / 2;
    let pruned_m = if let Some(ref sub) = plan.pruned_r4
        && seg_len <= sub.m
    {
        let m = sub.m;
        for r in 0..4 {
            let off = r * m;
            let re_r = &mut fft_re[off..off + m];
            let im_r = &mut fft_im[off..off + m];
            re_r[seg_len..m].fill(0.0);
            im_r[seg_len..m].fill(0.0);
            if r == 0 {
                for i in 0..seg_len {
                    re_r[i] = s0[i] * win[i];
                }
                if let Some(s1) = s1_opt {
                    for i in 0..seg_len {
                        im_r[i] = s1[i] * win[i];
                    }
                } else {
                    im_r[..seg_len].fill(0.0);
                }
            } else {
                let tc = &sub.tw_cos[r - 1];
                let ts = &sub.tw_sin[r - 1];
                if let Some(s1) = s1_opt {
                    for i in 0..seg_len {
                        let xr = s0[i] * win[i];
                        let xi = s1[i] * win[i];
                        re_r[i] = xr * tc[i] - xi * ts[i];
                        im_r[i] = xr * ts[i] + xi * tc[i];
                    }
                } else {
                    for i in 0..seg_len {
                        let xr = s0[i] * win[i];
                        re_r[i] = xr * tc[i];
                        im_r[i] = xr * ts[i];
                    }
                }
            }
            sub.sub_plan.fft_in_place(re_r, im_r);
        }
        Some(m)
    } else {
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
        None
    };

    let map_idx = |k: usize| -> usize {
        if let Some(m) = pruned_m {
            (k & 3) * m + (k >> 2)
        } else {
            k
        }
    };
    for (k, &band_u8) in bin_band.iter().enumerate() {
        let ki = map_idx(k);
        let (r0, i0, r1, i1, one_sided) = if k == 0 || k == half {
            (fft_re[ki], 0.0, fft_im[ki], 0.0, 1.0)
        } else {
            let nki = map_idx(n_fft - k);
            (
                0.5 * (fft_re[ki] + fft_re[nki]),
                0.5 * (fft_im[ki] - fft_im[nki]),
                0.5 * (fft_im[ki] + fft_im[nki]),
                0.5 * (fft_re[nki] - fft_re[ki]),
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
        let want_nyq = band_u8 == NYQ_STOPBAND_TAG && nyq_p.is_some();
        if scale != 0.0 && (b < 33 || want_nyq) {
            let mag_sq = (r0 * r0 + i0 * i0)
                + if s1_opt.is_some() {
                    r1 * r1 + i1 * i1
                } else {
                    0.0
                };
            let contrib = mag_sq * scale * one_sided;
            if b < 33 {
                band_p[b] += contrib;
            } else if let Some(ref mut np) = nyq_p {
                **np += contrib;
            }
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
    nyq_d: f64,
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
    let pruned_sub = plan.pruned_r4.as_ref().filter(|sub| seg_len <= sub.m);
    let pruned_m = pruned_sub.map(|s| s.m);
    let map_idx = |k: usize| -> usize {
        if let Some(m) = pruned_m {
            (k & 3) * m + (k >> 2)
        } else {
            k
        }
    };

    fft_re.fill(0.0);
    fft_im.fill(0.0);
    for (k, &band_u8) in bin_band.iter().enumerate() {
        let b = band_u8 as usize;
        let d_val = if b < 33 && active[b] {
            d_band[b]
        } else if band_u8 == NYQ_STOPBAND_TAG && nyq_d != 0.0 {
            nyq_d
        } else {
            continue;
        };
        let c = s_base * d_val;
        let ki = map_idx(k);
        if k == 0 || k == half {
            fft_re[ki] = c * win_re[off0 + k];
            if let Some(off1) = off1_opt {
                fft_im[ki] = c * win_re[off1 + k];
            }
        } else {
            let nki = map_idx(n_fft - k);
            let h0_re = c * win_re[off0 + k];
            let h0_im = -c * win_im[off0 + k];
            let (h1_re, h1_im) = if let Some(off1) = off1_opt {
                (c * win_re[off1 + k], -c * win_im[off1 + k])
            } else {
                (0.0, 0.0)
            };
            fft_re[ki] = h0_re - h1_im;
            fft_im[ki] = h0_im + h1_re;
            fft_re[nki] = h0_re + h1_im;
            fft_im[nki] = -h0_im + h1_re;
        }
    }

    if let Some(sub) = pruned_sub {
        let m = sub.m;
        for r in 0..4 {
            let off = r * m;
            let re_r = &mut fft_re[off..off + m];
            let im_r = &mut fft_im[off..off + m];
            sub.sub_plan.fft_in_place(re_r, im_r);
            if r == 0 {
                for n in 0..seg_len {
                    dy_full[base0 + n] += win[n] * re_r[n];
                }
                if let Some(base1) = base1_opt {
                    for n in 0..seg_len {
                        dy_full[base1 + n] += win[n] * im_r[n];
                    }
                }
            } else {
                let tc = &sub.tw_cos[r - 1];
                let ts = &sub.tw_sin[r - 1];
                for n in 0..seg_len {
                    let v0 = re_r[n] * tc[n] - im_r[n] * ts[n];
                    dy_full[base0 + n] += win[n] * v0;
                }
                if let Some(base1) = base1_opt {
                    for n in 0..seg_len {
                        let v1 = re_r[n] * ts[n] + im_r[n] * tc[n];
                        dy_full[base1 + n] += win[n] * v1;
                    }
                }
            }
        }
        return;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruned_radix4_matches_unpruned_65536_forward_and_backward() {
        let n_fft = 65_536;
        let seg_len = 16_384;
        let half_p1 = n_fft / 2 + 1;
        let plan_pruned = Radix2FftPlan::new(n_fft);
        assert!(plan_pruned.pruned_r4.is_some());
        let plan_ref = Radix2FftPlan::new_leaf(n_fft);

        let (win, _) = build_hann_window(seg_len);
        let intervals = canonical_ltas_33_intervals_hz();
        let bin_band = build_bin_band_map(n_fft, 48_000.0, 20_000.0, &intervals);
        let s0: Vec<f64> = (0..seg_len)
            .map(|i| 0.3 * ((i as f64) * 0.041).sin() + 0.15 * ((i as f64) * 0.27).cos())
            .collect();
        let s1: Vec<f64> = (0..seg_len)
            .map(|i| 0.25 * ((i as f64) * 0.083).cos() - 0.12 * ((i as f64) * 0.19).sin())
            .collect();

        let (mut re_p, mut im_p, mut bp_p) = (vec![0.0; n_fft], vec![0.0; n_fft], [0.0; 33]);
        let (mut re_r, mut im_r, mut bp_r) = (vec![0.0; n_fft], vec![0.0; n_fft], [0.0; 33]);
        let (mut wr0_p, mut wi0_p, mut wr1_p, mut wi1_p) = (
            vec![0.0; half_p1],
            vec![0.0; half_p1],
            vec![0.0; half_p1],
            vec![0.0; half_p1],
        );
        let (mut wr0_r, mut wi0_r, mut wr1_r, mut wi1_r) = (
            vec![0.0; half_p1],
            vec![0.0; half_p1],
            vec![0.0; half_p1],
            vec![0.0; half_p1],
        );

        accum_paired_forward_fft(
            &plan_pruned,
            seg_len,
            &win,
            &bin_band,
            &s0,
            Some(&s1),
            &mut re_p,
            &mut im_p,
            1.0e-4,
            &mut bp_p,
            None,
            Some((&mut wr0_p, &mut wi0_p)),
            Some((&mut wr1_p, &mut wi1_p)),
        );
        accum_paired_forward_fft(
            &plan_ref,
            seg_len,
            &win,
            &bin_band,
            &s0,
            Some(&s1),
            &mut re_r,
            &mut im_r,
            1.0e-4,
            &mut bp_r,
            None,
            Some((&mut wr0_r, &mut wi0_r)),
            Some((&mut wr1_r, &mut wi1_r)),
        );
        for b in 0..33 {
            assert!(
                (bp_p[b] - bp_r[b]).abs() < 1.0e-12,
                "band {b}: {} vs {}",
                bp_p[b],
                bp_r[b]
            );
        }

        let active = [true; 33];
        let d_band: [f64; 33] = std::array::from_fn(|b| ((b as f64) * 0.17).sin() * 0.25);
        let mut win_re = Vec::with_capacity(2 * half_p1);
        let mut win_im = Vec::with_capacity(2 * half_p1);
        win_re.extend_from_slice(&wr0_r);
        win_re.extend_from_slice(&wr1_r);
        win_im.extend_from_slice(&wi0_r);
        win_im.extend_from_slice(&wi1_r);
        let mut dy_p = vec![0.0_f64; 2 * seg_len];
        let mut dy_r = vec![0.0_f64; 2 * seg_len];

        accum_paired_backward_fft(
            &plan_pruned,
            seg_len,
            &win,
            &bin_band,
            &active,
            &d_band,
            0.0,
            1.0e-4,
            &win_re,
            &win_im,
            0,
            Some(half_p1),
            &mut re_p,
            &mut im_p,
            &mut dy_p,
            0,
            Some(seg_len),
        );
        accum_paired_backward_fft(
            &plan_ref,
            seg_len,
            &win,
            &bin_band,
            &active,
            &d_band,
            0.0,
            1.0e-4,
            &win_re,
            &win_im,
            0,
            Some(half_p1),
            &mut re_r,
            &mut im_r,
            &mut dy_r,
            0,
            Some(seg_len),
        );
        for i in 0..dy_r.len() {
            assert!(
                (dy_p[i] - dy_r[i]).abs() < 1.0e-11,
                "dy[{i}]: {} vs {}",
                dy_p[i],
                dy_r[i]
            );
        }
    }
}
