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

pub(super) fn cancel_discrete_early_reflections(
    s3_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    lags_l: [usize; 4],
    lags_r: [usize; 4],
) -> Vec<[f32; 2]> {
    let n = s3_lr.len().min(wet_lr.len());
    let mut out = wet_lr[..n].to_vec();
    if n < 512 {
        return out;
    }
    for (ch, lags) in [(0usize, lags_l), (1usize, lags_r)] {
        let mut uniq: Vec<usize> = Vec::with_capacity(4);
        for &d in &lags {
            if (24..n / 2).contains(&d) && uniq.iter().all(|&u| u.abs_diff(d) >= 8) {
                uniq.push(d);
            }
        }
        let m = uniq.len();
        if m == 0 {
            continue;
        }
        let mut g = vec![0.0_f64; crate::solver::uplo_u_packed_len(m)];
        let mut c = vec![[0.0_f64; 2]; m];
        let max_d = *uniq.iter().max().unwrap_or(&24);
        for i in (max_d + 16)..n {
            let r_hp = (f64::from(wet_lr[i][ch]) - f64::from(s3_lr[i][ch]))
                - 0.94 * (f64::from(wet_lr[i - 1][ch]) - f64::from(s3_lr[i - 1][ch]));
            for col in 0..m {
                let dc = uniq[col];
                let sc = f64::from(s3_lr[i - dc][ch]) - 0.94 * f64::from(s3_lr[i - 1 - dc][ch]);
                let base = crate::solver::uplo_u_index(0, col);
                for row in 0..=col {
                    let dr = uniq[row];
                    let sr = f64::from(s3_lr[i - dr][ch]) - 0.94 * f64::from(s3_lr[i - 1 - dr][ch]);
                    g[base + row] += sr * sc;
                }
                c[col][0] += sc * r_hp;
            }
        }
        let tr: f64 = (0..m).map(|k| g[crate::solver::uplo_u_index(k, k)]).sum();
        if let Ok(sol) = crate::solver::solve_uplo_u_cholesky_2ch(&g, &c, (1.0e-3 * tr).max(1.0e-9))
        {
            for (k, &d) in uniq.iter().enumerate() {
                let alpha = (sol[k][0] as f32).clamp(-0.65, 0.65);
                if alpha.abs() >= 0.03 {
                    for i in d..n {
                        out[i][ch] -= alpha * s3_lr[i - d][ch];
                    }
                }
            }
        }
    }
    out
}

/// Blind V2 Stage IV (`ARENA0_STAGE4_EXT`) seeding + direct memory tank refinement (`REQ-1.6`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lift_v2_post_s3_stage4(
    manifest: &crate::proj::schema::DnamProjManifest,
    arena_64k: &mut [u8],
    s3_ref_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    base_g_loop: f32,
    has_reflections: bool,
    ir_peak_lag: usize,
    n_fit: usize,
) {
    use super::ext::{coord_refine_arena_scalars, star_has_meta, star_meta_f64, write_f32_le};
    use dnam_engine::stages::Stage4SpatialTank;
    use dnam_engine::stages::weights::ARENA0_STAGE4_EXT_OFFSET as ARENA0_STAGE4_EXT;

    if arena_64k.len() < 4_772 || n_fit < 512 {
        return;
    }
    let tail_ratio = star_meta_f64(manifest, "phys_tail_energy_ratio", 0.0) as f32;
    let bbd_dl = star_meta_f64(manifest, "phys_bbd_delay_l_samples", 0.0) as f32;
    let bbd_dr = star_meta_f64(manifest, "phys_bbd_delay_r_samples", 0.0) as f32;
    let bbd_gl = star_meta_f64(manifest, "phys_bbd_gain_l", 0.0) as f32;
    let bbd_fb = star_meta_f64(manifest, "phys_bbd_fb", 0.0) as f32;
    let bbd_cfb = star_meta_f64(manifest, "phys_bbd_cross_fb", 0.0) as f32;
    let bbd_lp = star_meta_f64(manifest, "phys_bbd_lpf_pole", 0.0) as f32;
    let rt60_ms = star_meta_f64(manifest, "phys_rt60_ms", 0.0) as f32;
    let shim = star_meta_f64(manifest, "phys_shimmer_score", 0.0) as f32;
    let lfo_hz = star_meta_f64(manifest, "phys_lfo_hz", 0.0) as f32;
    let lfo_depth = star_meta_f64(manifest, "phys_lfo_depth", 0.0) as f32;

    arena_64k[ARENA0_STAGE4_EXT..ARENA0_STAGE4_EXT + 68].fill(0);
    let discrete_refl_only = has_reflections && base_g_loop <= 1.0e-6;
    let has_phys_meta = star_has_meta(manifest, "phys_tail_energy_ratio");
    if !has_phys_meta {
        write_f32_le(arena_64k, 4_688, 0.0);
        if discrete_refl_only {
            write_f32_le(arena_64k, 4_696, 0.0);
        }
        write_f32_le(arena_64k, 4_700, 0.0);
        write_f32_le(
            arena_64k,
            ARENA0_STAGE4_EXT,
            if has_reflections || ir_peak_lag >= 16 {
                0.0
            } else {
                -1.0
            },
        );
        return;
    }
    if tail_ratio < 0.008 && bbd_gl < 0.02 && rt60_ms < 45.0 {
        write_f32_le(arena_64k, 4_688, 0.0);
        if discrete_refl_only {
            write_f32_le(arena_64k, 4_696, 0.0);
        }
        write_f32_le(arena_64k, 4_700, 0.0);
        if !(has_reflections || ir_peak_lag >= 16) {
            write_f32_le(arena_64k, ARENA0_STAGE4_EXT, -1.0);
        }
        return;
    }

    let mut refine_s4: Vec<(usize, f32, f32)> = Vec::new();
    let has_bbd = bbd_dl >= 480.0 && bbd_gl >= 0.025;
    let has_fdn =
        rt60_ms >= 90.0 && tail_ratio > 0.015 && (!has_bbd || shim > 0.25 || tail_ratio > 0.12);
    if has_bbd {
        write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 4, bbd_gl.clamp(0.05, 1.5));
        write_f32_le(
            arena_64k,
            ARENA0_STAGE4_EXT + 8,
            bbd_dl.clamp(48.0, 44_000.0),
        );
        let dr = if bbd_dr >= 48.0 { bbd_dr } else { bbd_dl };
        write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 12, dr.clamp(48.0, 44_000.0));
        write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 16, bbd_fb.clamp(0.0, 0.94));
        write_f32_le(
            arena_64k,
            ARENA0_STAGE4_EXT + 20,
            bbd_cfb.clamp(-0.92, 0.92),
        );
        write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 24, bbd_lp.clamp(0.0, 0.95));
        if lfo_hz > 0.15 {
            write_f32_le(
                arena_64k,
                ARENA0_STAGE4_EXT + 60,
                (lfo_depth * 24.0).clamp(0.0, 120.0),
            );
            write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 64, lfo_hz.clamp(0.1, 15.0));
        }
        refine_s4.extend([
            (ARENA0_STAGE4_EXT + 4, 0.0, 1.6),
            (ARENA0_STAGE4_EXT + 16, 0.0, 0.95),
            (ARENA0_STAGE4_EXT + 20, -0.92, 0.92),
            (ARENA0_STAGE4_EXT + 24, 0.0, 0.96),
        ]);
    }
    if has_fdn {
        let d_mean = 1_577.0_f64;
        let fdn_fb = (10.0_f64.powf(-3.0 * d_mean / (48.0 * f64::from(rt60_ms.max(80.0)))))
            .clamp(0.15, 0.975) as f32;
        let fdn_g = (tail_ratio.sqrt() * 0.55).clamp(0.08, 1.1);
        write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 28, fdn_g);
        for (i, d) in [1117.0_f32, 1399.0, 1747.0, 2053.0].into_iter().enumerate() {
            write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 32 + i * 4, d);
        }
        write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 48, fdn_fb);
        write_f32_le(arena_64k, ARENA0_STAGE4_EXT + 52, bbd_lp.clamp(0.18, 0.95));
        if shim > 0.25 {
            write_f32_le(
                arena_64k,
                ARENA0_STAGE4_EXT + 56,
                (shim * 0.45).clamp(0.12, 0.85),
            );
            refine_s4.push((ARENA0_STAGE4_EXT + 56, 0.0, 0.90));
        }
        refine_s4.extend([
            (ARENA0_STAGE4_EXT + 28, 0.0, 1.4),
            (ARENA0_STAGE4_EXT + 48, 0.1, 0.985),
            (ARENA0_STAGE4_EXT + 52, 0.0, 0.95),
        ]);
    }
    let tank_w = if has_bbd || has_fdn || discrete_refl_only {
        0.0
    } else {
        (tail_ratio * 2.0).clamp(0.05, 1.0)
    };
    write_f32_le(arena_64k, ARENA0_STAGE4_EXT, tank_w);
    if !has_bbd && !has_fdn && !discrete_refl_only {
        write_f32_le(arena_64k, 4_688, base_g_loop.clamp(0.0, 0.92));
    } else {
        write_f32_le(arena_64k, 4_688, 0.0);
        if discrete_refl_only {
            write_f32_le(arena_64k, 4_696, 0.0);
            write_f32_le(arena_64k, 4_700, 0.0);
        }
    }

    if !refine_s4.is_empty() {
        let eval_stereo = manifest.base.io_config.output_mode != "mono";
        let m = s3_ref_lr.len().min(wet_lr.len()).min(n_fit);
        let mut eval_s4 = |buf: &[u8]| -> f64 {
            let mut tank = Stage4SpatialTank::default();
            tank.load_from_arena(buf);
            let mut err = 0.0_f64;
            let mut tot = 0.0_f64;
            for i in 0..m {
                let (yl, yr) = tank.step_stereo(s3_ref_lr[i][0], s3_ref_lr[i][1]);
                let dl = f64::from(yl - wet_lr[i][0]);
                err += dl * dl;
                tot += f64::from(wet_lr[i][0]).powi(2);
                if eval_stereo {
                    let dr = f64::from(yr - wet_lr[i][1]);
                    err += dr * dr;
                    tot += f64::from(wet_lr[i][1]).powi(2);
                }
            }
            err / tot.max(1.0e-12)
        };
        let mut best_l = eval_s4(arena_64k);
        coord_refine_arena_scalars(&refine_s4, &mut best_l, arena_64k, &mut eval_s4);
    }
}

pub(super) fn solve_svf_weights_3x3(
    svf_basis: &[[f64; 3]],
    tgt: &[f64],
    start: usize,
    n_eval: usize,
) -> Option<([f64; 3], f64, f64)> {
    let mut gram = [0.0_f64; 6];
    let mut rhs = [[0.0_f64; 2]; 3];
    let mut y_e = 0.0_f64;
    for i in start..n_eval {
        let y = tgt[i];
        y_e += y * y;
        let b = svf_basis[i];
        for c in 0..3 {
            let col = super::super::uplo_u_index(0, c);
            for r in 0..=c {
                gram[col + r] += b[r] * b[c];
            }
            rhs[c][0] += b[c] * y;
        }
    }
    if y_e <= 1.0e-12 {
        return None;
    }
    let tr = gram[0] + gram[2] + gram[5];
    let h =
        super::super::solve_uplo_u_cholesky_2ch(&gram, &rhs, (1.0e-5 * tr).max(1.0e-12)).ok()?;
    let w = [h[0][0], h[1][0], h[2][0]];
    let mut err_e = 0.0_f64;
    for i in start..n_eval {
        let pred = w[0] * svf_basis[i][0] + w[1] * svf_basis[i][1] + w[2] * svf_basis[i][2];
        let d = tgt[i] - pred;
        err_e += d * d;
    }
    Some((w, err_e / y_e, y_e))
}

/// Pure-`f64` TPT SVF system-ID surrogate matching `Stage1Pre::step_stereo_ext` equations.
pub(super) fn fit_svf_tpt_surrogate(
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    eval_stereo: bool,
    has_post_nl: bool,
    n_eval: usize,
) -> Option<(f32, f32, f32, [f32; 3])> {
    let start = n_eval.min(512);
    if n_eval <= start + 32 || n_eval > dry_lr.len() || n_eval > wet_lr.len() {
        return None;
    }
    let (mut env_trace, mut x_in, mut tgt) = (
        Vec::with_capacity(n_eval),
        Vec::with_capacity(n_eval),
        Vec::with_capacity(n_eval),
    );
    let mut env = 0.0_f64;
    for i in 0..n_eval {
        let (dl, dr) = (f64::from(dry_lr[i][0]), f64::from(dry_lr[i][1]));
        let det = 0.5 * (dl.abs() + dr.abs());
        env += (if det > env { 0.032 } else { 0.0012 }) * (det - env);
        env_trace.push(env);
        x_in.push(if eval_stereo { 0.5 * (dl + dr) } else { dl });
        tgt.push(if eval_stereo {
            0.5 * f64::from(wet_lr[i][0] + wet_lr[i][1])
        } else {
            f64::from(wet_lr[i][0])
        });
    }
    let mut svf_basis = vec![[0.0_f64; 3]; n_eval];
    let mut u_buf = vec![0.0_f64; n_eval];
    let eval_cfg = |fc0: f64,
                    sens: f64,
                    k_damp: f64,
                    basis: &mut [[f64; 3]],
                    u: &mut [f64]|
     -> Option<(f64, [f64; 3])> {
        let (mut ic1, mut ic2) = (0.0_f64, 0.0_f64);
        for i in 0..n_eval {
            let fc =
                (fc0 * 2.0_f64.powf((sens * env_trace[i]).clamp(0.0, 3.8))).clamp(20.0, 18_000.0);
            let w = (std::f64::consts::PI * fc / 48_000.0).clamp(1.0e-4, 1.35);
            let g = w.sin() / w.cos().max(0.05);
            let a1 = 1.0 / (1.0 + g * (g + k_damp));
            let (a2, a3, xi) = (g * a1, g * g * a1, x_in[i]);
            let v3 = xi - ic2;
            let v1 = a1 * ic1 + a2 * v3;
            let v2 = ic2 + a2 * ic1 + a3 * v3;
            ic1 = 2.0 * v1 - ic1;
            ic2 = 2.0 * v2 - ic2;
            basis[i] = [v2, v1, xi - k_damp * v1 - v2];
        }
        let (w, mut best_rel, y_e) = solve_svf_weights_3x3(basis, &tgt, start, n_eval)?;
        if has_post_nl {
            let mut u_peak = 1.0e-9_f64;
            for i in 0..n_eval {
                u[i] = w[0] * basis[i][0] + w[1] * basis[i][1] + w[2] * basis[i][2];
                if i >= start {
                    u_peak = u_peak.max(u[i].abs());
                }
            }
            for drv in [1.2_f64, 2.0] {
                let (mut s_uu, mut s_uy) = (0.0_f64, 0.0_f64);
                for i in start..n_eval {
                    let z = (drv * (u[i] / u_peak)).tanh();
                    s_uu += z * z;
                    s_uy += z * tgt[i];
                }
                if s_uu > 1.0e-12 {
                    let g_sc = s_uy / s_uu;
                    let err_nl: f64 = (start..n_eval)
                        .map(|i| (tgt[i] - g_sc * (drv * (u[i] / u_peak)).tanh()).powi(2))
                        .sum();
                    best_rel = best_rel.min(err_nl / y_e);
                }
            }
        }
        Some((best_rel, w))
    };
    let (mut best_err, mut best_fc, mut best_sens, mut best_k) =
        (f64::INFINITY, 340.0_f64, 4.0_f64, 0.25_f64);
    for fc0 in [180.0_f64, 280.0, 420.0, 650.0, 1_000.0] {
        for sens in [1.5_f64, 4.0, 6.8, 9.5, 12.5] {
            for kd in [0.14_f64, 0.28, 0.50] {
                if let Some((err, _)) = eval_cfg(fc0, sens, kd, &mut svf_basis, &mut u_buf)
                    && err < best_err
                {
                    (best_err, best_fc, best_sens, best_k) = (err, fc0, sens, kd);
                }
            }
        }
    }
    if !best_err.is_finite() {
        return None;
    }
    for _pass in 0..2 {
        for &m_fc in &[0.78_f64, 1.28] {
            let fc = (best_fc * m_fc).clamp(120.0, 2_200.0);
            if let Some((e, _)) = eval_cfg(fc, best_sens, best_k, &mut svf_basis, &mut u_buf)
                && e < best_err
            {
                (best_err, best_fc) = (e, fc);
            }
        }
        for &d_sens in &[-1.2_f64, 1.2] {
            let s = (best_sens + d_sens).clamp(0.0, 15.0);
            if let Some((e, _)) = eval_cfg(best_fc, s, best_k, &mut svf_basis, &mut u_buf)
                && e < best_err
            {
                (best_err, best_sens) = (e, s);
            }
        }
        for &d_k in &[-0.06_f64, 0.06] {
            let kd = (best_k + d_k).clamp(0.06, 1.20);
            if let Some((e, _)) = eval_cfg(best_fc, best_sens, kd, &mut svf_basis, &mut u_buf)
                && e < best_err
            {
                (best_err, best_k) = (e, kd);
            }
        }
    }
    let static_err =
        eval_cfg(best_fc, 0.0, best_k, &mut svf_basis, &mut u_buf).map_or(1.0, |(e, _)| e);
    if !best_err.is_finite()
        || best_err >= 0.82
        || best_sens < 0.5
        || (has_post_nl && (best_err >= 0.52 || best_err >= 0.65 * static_err))
    {
        return None;
    }
    let (_, w) = eval_cfg(best_fc, best_sens, best_k, &mut svf_basis, &mut u_buf)?;
    let max_abs = w.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let sc = if max_abs > 1.0e-4 { max_abs } else { 1.0 };
    Some((
        best_fc as f32,
        best_sens as f32,
        best_k as f32,
        [(w[0] / sc) as f32, (w[1] / sc) as f32, (w[2] / sc) as f32],
    ))
}
