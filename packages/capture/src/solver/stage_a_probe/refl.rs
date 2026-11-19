#![forbid(unsafe_code)]
//! Stage IV Stereo Reflection & Reverberation Delay Anchor Identification (`solver::stage_a::refl`, `REQ-1.6`).

const WIENER_FFT_LEN: usize = 4_096;

/// Safe-Rust in-place radix-2 Cooley-Tukey FFT/IFFT (`fft_len` must be a power of two).
fn fft_radix2_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    debug_assert!(n.is_power_of_two());
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let angle = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let (wlen_sin, wlen_cos) = angle.sin_cos();
        for start in (0..n).step_by(len) {
            let (mut w_re, mut w_im) = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let u = start + k;
                let v = u + half;
                let vr = re[v] * w_re - im[v] * w_im;
                let vi = re[v] * w_im + im[v] * w_re;
                re[v] = re[u] - vr;
                im[v] = im[u] - vi;
                re[u] += vr;
                im[u] += vi;
                let next_re = w_re * wlen_cos - w_im * wlen_sin;
                w_im = w_re * wlen_sin + w_im * wlen_cos;
                w_re = next_re;
            }
        }
        len <<= 1;
    }
    if inverse {
        let inv_n = 1.0_f64 / (n as f64);
        for v in re.iter_mut() {
            *v *= inv_n;
        }
        for v in im.iter_mut() {
            *v *= inv_n;
        }
    }
}

/// 4096-pt Wiener impulse deconvolution `IFFT(conj(X) * R / (|X|^2 + reg))` with `<= 3072`
/// active samples so lags `0..1024` are strictly free of circular wrap-around.
fn wiener_deconv_abs(ds: &[f64], dr: &[f64]) -> Vec<f64> {
    let seg_len = ds.len().min(dr.len()).min(3_072);
    if seg_len < 64 {
        return vec![0.0_f64; 1_000];
    }
    let mut x_re = vec![0.0_f64; WIENER_FFT_LEN];
    let mut x_im = vec![0.0_f64; WIENER_FFT_LEN];
    let mut r_re = vec![0.0_f64; WIENER_FFT_LEN];
    let mut r_im = vec![0.0_f64; WIENER_FFT_LEN];
    x_re[..seg_len].copy_from_slice(&ds[..seg_len]);
    r_re[..seg_len].copy_from_slice(&dr[..seg_len]);
    // Gentle 16-sample right-edge fade before zero-padding avoids step-truncation sidelobes.
    for k in 0..16 {
        let w = 0.5 * (1.0 - (std::f64::consts::PI * ((k + 1) as f64) / 17.0).cos());
        let idx = seg_len - 1 - k;
        x_re[idx] *= w;
        r_re[idx] *= w;
    }

    fft_radix2_inplace(&mut x_re, &mut x_im, false);
    fft_radix2_inplace(&mut r_re, &mut r_im, false);

    let mut mean_px = 0.0_f64;
    for k in 0..WIENER_FFT_LEN {
        mean_px += x_re[k] * x_re[k] + x_im[k] * x_im[k];
    }
    mean_px /= WIENER_FFT_LEN as f64;
    if mean_px <= 1.0e-18 {
        return vec![0.0_f64; 1_000];
    }
    let reg = (3.5e-3 * mean_px).max(1.0e-18);

    let mut h_re = vec![0.0_f64; WIENER_FFT_LEN];
    let mut h_im = vec![0.0_f64; WIENER_FFT_LEN];
    for k in 0..WIENER_FFT_LEN {
        let px = x_re[k] * x_re[k] + x_im[k] * x_im[k];
        let denom = px + reg;
        h_re[k] = (x_re[k] * r_re[k] + x_im[k] * r_im[k]) / denom;
        h_im[k] = (x_re[k] * r_im[k] - x_im[k] * r_re[k]) / denom;
    }
    fft_radix2_inplace(&mut h_re, &mut h_im, true);
    h_re.into_iter().take(1_000).map(f64::abs).collect()
}

/// Identifies stereo reflection lag anchors via 4096-pt Wiener deconvolution combined with
/// pre-whitened cross-correlation of `(wet - s3)` against `s3`, mapping Left peaks to EVEN lanes
/// and Right peaks to ODD lanes across `Lite..Ultra` (`4..=1000` in V2), plus dispersive neighbor
/// taps (`[D, D-1, D+1, D±2]`) and diffuse loop gain (`r[train.pipeline.stage-a-poles]`).
/// Returns `(lite, full, pro, ultra, g_loop, has_reflections)`.
pub(crate) fn extract_stage4_reflection_delays(
    s3_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 8], f32, bool) {
    let def_lite = [53.0_f32, 89.0, 137.0, 199.0];
    let def_full = [257.0_f32, 313.0, 389.0, 463.0];
    let def_pro = [523.0_f32, 593.0, 661.0, 743.0];
    let def_ultra = [773.0_f32, 809.0, 839.0, 877.0, 907.0, 941.0, 971.0, 997.0];

    let n_all = s3_lr.len().min(wet_lr.len()).min(12_000);
    if n_all < 1_100 {
        return (def_lite, def_full, def_pro, def_ultra, 0.0, false);
    }
    let uhf_sq = |i: usize| {
        let d2l =
            f64::from(s3_lr[i][0]) - 2.0 * f64::from(s3_lr[i - 1][0]) + f64::from(s3_lr[i - 2][0]);
        let d2r =
            f64::from(s3_lr[i][1]) - 2.0 * f64::from(s3_lr[i - 1][1]) + f64::from(s3_lr[i - 2][1]);
        d2l * d2l + d2r * d2r
    };
    let tail_has_uhf = n_all > 8_744 && {
        let tot_uhf: f64 = (2..n_all).map(uhf_sq).sum();
        let tail_uhf: f64 = ((n_all - 3_256).max(2)..n_all).map(uhf_sq).sum();
        tail_uhf > 0.25 * tot_uhf
    };
    let (start_off, end_off) = if tail_has_uhf {
        (8_744, n_all)
    } else if n_all > 4_608 {
        let mut best_b = 0usize;
        let mut best_e = -1.0_f64;
        let mut b = 0usize;
        loop {
            let end_b = (b + 1_024).min(n_all);
            let e: f64 = (b.max(2)..end_b).map(uhf_sq).sum();
            if e > best_e {
                best_e = e;
                best_b = b;
            }
            if end_b == n_all {
                break;
            }
            b = (b + 512).min(n_all - 1_024);
        }
        let s = best_b.saturating_sub(1_536).min(n_all - 4_608);
        (s, s + 4_608)
    } else {
        (0, n_all)
    };
    let s3_lr = &s3_lr[start_off..end_off];
    let wet_lr = &wet_lr[start_off..end_off];
    let n = s3_lr.len();

    let (mut ds_l, mut ds_r) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    let (mut dr_l, mut dr_r) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    let (mut es_l, mut es_r, mut ew_tot) = (0.0, 0.0, 0.0);
    for i in 1..n {
        let sl = f64::from(s3_lr[i][0]) - 0.94 * f64::from(s3_lr[i - 1][0]);
        let sr = f64::from(s3_lr[i][1]) - 0.94 * f64::from(s3_lr[i - 1][1]);
        let rl_cur = f64::from(wet_lr[i][0]) - f64::from(s3_lr[i][0]);
        let rl_prv = f64::from(wet_lr[i - 1][0]) - f64::from(s3_lr[i - 1][0]);
        let rr_cur = f64::from(wet_lr[i][1]) - f64::from(s3_lr[i][1]);
        let rr_prv = f64::from(wet_lr[i - 1][1]) - f64::from(s3_lr[i - 1][1]);
        ds_l[i] = sl;
        ds_r[i] = sr;
        dr_l[i] = rl_cur - 0.94 * rl_prv;
        dr_r[i] = rr_cur - 0.94 * rr_prv;
        if i >= 64 {
            es_l += sl * sl;
            es_r += sr * sr;
            let wl = f64::from(wet_lr[i][0]) - 0.94 * f64::from(wet_lr[i - 1][0]);
            let wr = f64::from(wet_lr[i][1]) - 0.94 * f64::from(wet_lr[i - 1][1]);
            ew_tot += 0.5 * (wl * wl + wr * wr);
        }
    }
    let norm_l = (es_l * ew_tot).sqrt().max(1.0e-12);
    let norm_r = (es_r * ew_tot).sqrt().max(1.0e-12);
    let es_l_safe = es_l.max(1.0e-12);
    let es_r_safe = es_r.max(1.0e-12);

    let mut corr_l = vec![0.0_f64; 1_000];
    let mut corr_r = vec![0.0_f64; 1_000];
    for lag in 24..998 {
        let (mut sl, mut sr, mut al, mut ar) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for idx in lag.max(64)..n {
            sl += dr_l[idx] * ds_l[idx - lag];
            sr += dr_r[idx] * ds_r[idx - lag];
            al += ds_l[idx] * ds_l[idx - lag];
            ar += ds_r[idx] * ds_r[idx - lag];
        }
        let pen_l = (1.0 - 0.85 * (al / es_l_safe).powi(2)).clamp(0.0, 1.0);
        let pen_r = (1.0 - 0.85 * (ar / es_r_safe).powi(2)).clamp(0.0, 1.0);
        corr_l[lag] = (sl.abs() / norm_l) * pen_l;
        corr_r[lag] = (sr.abs() / norm_r) * pen_r;
    }

    let wiener_l = wiener_deconv_abs(&ds_l, &dr_l);
    let wiener_r = wiener_deconv_abs(&ds_r, &dr_r);
    let mut score_l = vec![0.0_f64; 1_000];
    let mut score_r = vec![0.0_f64; 1_000];
    for lag in 24..998 {
        score_l[lag] = 0.78 * wiener_l[lag] + 0.22 * corr_l[lag];
        score_r[lag] = 0.78 * wiener_r[lag] + 0.22 * corr_r[lag];
    }

    let collect_peaks =
        |score: &[f64], lo: usize, hi: usize, min_sep: usize, cap: usize| -> Vec<usize> {
            let mut peaks: Vec<(f64, usize)> = (lo + 1..hi)
                .filter(|&l| {
                    score[l] > 0.028 && score[l] >= score[l - 1] && score[l] >= score[l + 1]
                })
                .map(|l| (score[l], l))
                .collect();
            peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut out = Vec::with_capacity(cap);
            for (_, lag) in peaks {
                if out.iter().all(|&c: &usize| c.abs_diff(lag) >= min_sep) {
                    out.push(lag);
                    if out.len() == cap {
                        break;
                    }
                }
            }
            out
        };

    let peaks_l = collect_peaks(&score_l, 24, 996, 12, 8);
    let peaks_r = collect_peaks(&score_r, 24, 996, 12, 8);

    // Measure diffuse late reverb background excluding +/-3 samples around detected discrete peaks.
    let (mut late_w_sum, mut late_w_cnt) = (0.0_f64, 0usize);
    for l in 450..995 {
        let near_l = peaks_l.iter().any(|&p| p.abs_diff(l) <= 3);
        let near_r = peaks_r.iter().any(|&p| p.abs_diff(l) <= 3);
        if !near_l {
            late_w_sum += wiener_l[l];
            late_w_cnt += 1;
        }
        if !near_r {
            late_w_sum += wiener_r[l];
            late_w_cnt += 1;
        }
    }
    let late_w_mean = if late_w_cnt > 0 {
        late_w_sum / (late_w_cnt as f64)
    } else {
        0.0
    };
    let late_corr_mean: f64 = (450..995)
        .map(|l| 0.5 * (corr_l[l] + corr_r[l]))
        .sum::<f64>()
        / 545.0;
    let max_score = score_l[24..996]
        .iter()
        .chain(&score_r[24..996])
        .copied()
        .fold(0.0_f64, f64::max);
    let has_reflections = max_score > 0.045 || late_corr_mean > 0.018;
    let discrete_dominant = max_score > 4.0 * late_w_mean && late_w_mean < 0.012;
    let g_loop = if discrete_dominant || late_w_mean <= 0.012 {
        0.0_f32
    } else {
        (0.05 + 4.5 * (late_w_mean - 0.014).max(0.0)).clamp(0.05, 0.85) as f32
    };

    if !has_reflections {
        return (def_lite, def_full, def_pro, def_ultra, 0.0, false);
    }

    // Populate Lite (2 L + 2 R), Full, Pro, and Ultra with detected peaks + dispersive neighbor taps [D±1, D±2]
    let fill_ch = |peaks: &[usize]| -> ([usize; 2], [usize; 2], [usize; 2], [usize; 4]) {
        let d0 = peaks.first().copied().unwrap_or(def_lite[0] as usize);
        let d1 = peaks.get(1).copied().unwrap_or_else(|| (d0 + 19).min(996));
        let lite = [d0, d1];
        let full = [
            peaks
                .get(2)
                .copied()
                .unwrap_or_else(|| (d0.saturating_sub(1)).max(4)),
            peaks
                .get(3)
                .copied()
                .unwrap_or_else(|| (d1.saturating_sub(1)).max(4)),
        ];
        let pro = [
            peaks.get(4).copied().unwrap_or_else(|| (d0 + 1).min(996)),
            peaks.get(5).copied().unwrap_or_else(|| (d1 + 1).min(996)),
        ];
        let ultra = [
            peaks
                .get(6)
                .copied()
                .unwrap_or_else(|| (d0.saturating_sub(2)).max(4)),
            peaks.get(7).copied().unwrap_or_else(|| (d0 + 2).min(996)),
            (d1.saturating_sub(2)).max(4),
            (d1 + 2).min(996),
        ];
        (lite, full, pro, ultra)
    };

    let (ll, lf, lp, lu) = fill_ch(&peaks_l);
    let (rl, rf, rp, ru) = fill_ch(&peaks_r);
    let pair4 =
        |l: &[usize; 2], r: &[usize; 2]| [l[0] as f32, r[0] as f32, l[1] as f32, r[1] as f32];
    let u8_arr = std::array::from_fn(|k| {
        if k % 2 == 0 {
            lu[k / 2] as f32
        } else {
            ru[k / 2] as f32
        }
    });
    (
        pair4(&ll, &rl),
        pair4(&lf, &rf),
        pair4(&lp, &rp),
        u8_arr,
        g_loop,
        true,
    )
}
