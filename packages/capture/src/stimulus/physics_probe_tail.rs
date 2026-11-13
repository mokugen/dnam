#![forbid(unsafe_code)]
//! Tail, Echo, RT60, BBD, and LFO Physics Helpers (`stimulus::physics_probe_tail`).

#[derive(Default)]
pub(super) struct TailDecayStats {
    pub(super) delays_ms: Vec<f32>,
    pub(super) decay_gains: Vec<f32>,
    pub(super) delay_l_samples: f64,
    pub(super) delay_r_samples: f64,
    pub(super) bbd_gain_l: f64,
    pub(super) bbd_gain_r: f64,
    pub(super) bbd_fb: f64,
    pub(super) bbd_cross_fb: f64,
    pub(super) bbd_lpf_pole: f64,
    pub(super) rt60_ms: f64,
    pub(super) tail_energy_ratio: f64,
    pub(super) bbd_hf_retain: f64,
    pub(super) shimmer_score: f64,
    pub(super) lfo_hz: f64,
    pub(super) lfo_depth: f64,
    pub(super) pingpong_asym: f64,
}

fn ac_couple_tail(buf: &[f32]) -> Vec<f32> {
    const R: f64 = 0.992;
    let mut out: Vec<f64> = buf.iter().map(|&v| f64::from(v)).collect();
    for _ in 0..2 {
        let (mut y, mut xp) = (0.0_f64, 0.0_f64);
        for slot in &mut out {
            let x = *slot;
            y = R * (y + x - xp);
            xp = x;
            *slot = y;
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn analyze_decay_tail(
    cs_dry_l: &[f32],
    cs_wet_l: &[f32],
    cs_wet_r: Option<&[f32]>,
    comb_dry: &[f32],
    comb_wet_l: &[f32],
    comb_wet_r: Option<&[f32]>,
    tail_wet_l: &[f32],
    tail_wet_r: Option<&[f32]>,
    comb_hf_ratio: f64,
) -> TailDecayStats {
    if tail_wet_l.len() < 1_024 || comb_wet_l.is_empty() {
        return TailDecayStats {
            bbd_hf_retain: 1.0,
            ..Default::default()
        };
    }
    let comb_rms = rms_energy(comb_wet_l).max(1.0e-6);
    let tail_ac_l = ac_couple_tail(tail_wet_l);
    let tail_ac_r = tail_wet_r.map(ac_couple_tail);
    let skip = (20 * 48).min(tail_ac_l.len() / 2);
    let tail_energy_ratio = (rms_energy(&tail_ac_l[skip..]) / comb_rms).clamp(0.0, 10.0);
    let blk = 48_usize;
    let n_ms = tail_ac_l.len() / blk;
    let env_l: Vec<f64> = (0..n_ms)
        .map(|m| rms_energy(&tail_ac_l[m * blk..(m + 1) * blk]))
        .collect();
    let env_r: Vec<f64> = tail_ac_r
        .as_deref()
        .map(|tr| {
            (0..(tr.len() / blk).min(n_ms))
                .map(|m| rms_energy(&tr[m * blk..(m + 1) * blk]))
                .collect()
        })
        .unwrap_or_default();
    let mut s = extract_echo_peaks(
        cs_dry_l,
        cs_wet_l,
        cs_wet_r,
        tail_energy_ratio,
        comb_dry,
        comb_wet_l,
        comb_wet_r,
        tail_wet_l,
        tail_wet_r,
        &env_l,
        comb_rms,
    );
    s.rt60_ms = estimate_rt60_ms(&env_l, comb_rms);
    s.tail_energy_ratio = tail_energy_ratio;
    let tail_late = &tail_wet_l[(60 * blk).min(tail_wet_l.len() / 2)..];
    s.bbd_hf_retain = if rms_energy(tail_late) > 5.0e-4 * comb_rms {
        (first_diff_hf_ratio(tail_late) / first_diff_hf_ratio(comb_wet_l).max(1.0e-4))
            .clamp(0.0, 4.0)
    } else {
        1.0
    };
    s.bbd_lpf_pole = if tail_energy_ratio > 0.012 && s.bbd_hf_retain < 0.96 {
        ((1.0 - s.bbd_hf_retain.max(0.05)) * 0.88)
            .sqrt()
            .clamp(0.0, 0.94)
    } else {
        0.0
    };
    s.shimmer_score = if s.bbd_hf_retain > 1.25 && tail_energy_ratio > 0.02 && comb_hf_ratio > 0.01
    {
        ((s.bbd_hf_retain - 1.0) * 0.8 + tail_energy_ratio.min(0.5)).clamp(0.0, 2.0)
    } else {
        0.0
    };
    (s.lfo_hz, s.lfo_depth) = estimate_lfo_modulation(&env_l);
    if env_r.len() == env_l.len() && env_l.len() > 80 {
        let (ds, ts) = (15..env_l.len().min(500)).fold((0.0_f64, 0.0_f64), |(d, t), m| {
            (d + (env_l[m] - env_r[m]).abs(), t + env_l[m] + env_r[m])
        });
        if ts > 1.0e-9 {
            s.pingpong_asym = (ds / ts).clamp(0.0, 1.0);
        }
    }
    s
}

fn corr_at_offset(
    motif: &[f32],
    motif_e: f64,
    tail: &[f32],
    off: usize,
    comb_rms: f64,
    step: usize,
) -> (f64, f64, f64) {
    let (mut dot, mut te) = (0.0_f64, 0.0_f64);
    for i in (0..motif.len()).step_by(step) {
        let (d, t) = (f64::from(motif[i]), f64::from(tail[off + i]));
        dot += d * t;
        te += t * t;
    }
    let scale = step as f64;
    let sc = dot / ((motif_e / scale) * (te + 1.0e-12)).sqrt().max(1.0e-12);
    let gain = ((te * scale / (motif.len() as f64)).sqrt() / comb_rms.max(1.0e-6)).clamp(0.0, 1.5);
    (sc.abs(), sc, gain)
}

#[allow(clippy::too_many_arguments)]
fn scan_peaks_shared(
    motif: &[f32],
    motif_e: f64,
    rms_m: f64,
    tgt: &[f32],
    start_off: usize,
    max_off: usize,
    c_step: usize,
    lag_base: usize,
    c_th: f64,
    f_th: f64,
    g_th: f64,
    dry_check: Option<(&[f32], f64, f64, &[f32])>,
) -> Vec<(f64, f64, f64, f64)> {
    if start_off + 16 >= max_off {
        return Vec::new();
    }
    let coarse: Vec<(usize, f64, f64)> = (start_off..max_off)
        .step_by(c_step)
        .map(|o| {
            let (a, _, g) = corr_at_offset(motif, motif_e, tgt, o, rms_m, 4);
            (o, a, g)
        })
        .collect();
    let mut out = Vec::new();
    for i in 1..coarse.len().saturating_sub(1) {
        if coarse[i].1 > c_th
            && coarse[i].2 > g_th
            && coarse[i].1 >= coarse[i - 1].1
            && coarse[i].1 >= coarse[i + 1].1
        {
            let (lo, hi) = (
                coarse[i].0.saturating_sub(c_step + 4).max(start_off),
                (coarse[i].0 + c_step + 4).min(max_off - 1),
            );
            let (mut bo, mut ba, mut bs, mut bg) = (lo, -1.0_f64, 0.0, 0.0);
            for o in lo..=hi {
                let (a, s, g) = corr_at_offset(motif, motif_e, tgt, o, rms_m, 1);
                if a > ba {
                    ba = a;
                    bo = o;
                    bs = s;
                    bg = g;
                }
            }
            if ba > f_th && bg > g_th {
                if let Some((dm, de, dr, dt)) = dry_check
                    && bo + dm.len() <= dt.len()
                    && corr_at_offset(dm, de, dt, bo, dr, 1).0 >= 0.55 * ba
                {
                    continue;
                }
                let frac = if bo > lo && bo < hi {
                    let y0 = corr_at_offset(motif, motif_e, tgt, bo - 1, rms_m, 1).0;
                    let y2 = corr_at_offset(motif, motif_e, tgt, bo + 1, rms_m, 1).0;
                    let den = 2.0 * (y0 - 2.0 * ba + y2);
                    if den.abs() > 1.0e-12 {
                        ((y0 - y2) / den).clamp(-0.5, 0.5)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let lag = (bo + lag_base) as f64 + frac;
                if out
                    .last()
                    .is_none_or(|p: &(f64, f64, f64, f64)| (lag - p.0).abs() > 720.0)
                {
                    out.push((lag, ba, bs, bg.clamp(0.05, 1.2)));
                    if out.len() >= 4 {
                        break;
                    }
                }
            }
        }
    }
    out
}

fn find_motif_peaks_fine(comb: &[f32], tail: &[f32], min_lag: usize) -> Vec<(f64, f64, f64, f64)> {
    let mlen = comb.len().min(1_024);
    if mlen < 256 || tail.len() < mlen + 64 {
        return Vec::new();
    }
    let motif = &comb[comb.len() - mlen..];
    let me: f64 = motif
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .max(1.0e-12);
    scan_peaks_shared(
        motif,
        me,
        (me / (mlen as f64)).sqrt().max(1.0e-6),
        tail,
        min_lag.saturating_sub(mlen),
        tail.len() - mlen,
        4,
        mlen,
        0.28,
        0.32,
        0.03,
        None,
    )
}

fn find_whitened_lag_peaks(dry_src: &[f32], src: &[f32], dst: &[f32]) -> Vec<(f64, f64, f64, f64)> {
    let n = src.len().min(dst.len()).min(dry_src.len());
    if n < 6_000 {
        return Vec::new();
    }
    let w: Vec<f32> = (1..n).map(|i| src[i] - 0.94 * src[i - 1]).collect();
    let wd: Vec<f32> = (1..n).map(|i| dst[i] - 0.94 * dst[i - 1]).collect();
    let wdry: Vec<f32> = (1..n).map(|i| dry_src[i] - 0.94 * dry_src[i - 1]).collect();
    let (mut bs, mut be) = (0usize, 0.0_f64);
    for st in (0..w.len().saturating_sub(29_800)).step_by(2_048) {
        let e: f64 = w[st..st + 4_096]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        if e > be {
            be = e;
            bs = st;
        }
    }
    let mlen = 4_096.min(w.len().saturating_sub(bs));
    let max_lag = 28_800.min(wd.len().saturating_sub(bs + mlen));
    if be <= 1.0e-8 || mlen < 2_048 || max_lag <= 760 {
        return Vec::new();
    }
    let dry_e: f64 = wdry[bs..bs + mlen]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .max(1.0e-12);
    let dry_rms = (dry_e / (mlen as f64)).sqrt().max(1.0e-6);
    scan_peaks_shared(
        &w[bs..bs + mlen],
        be,
        (be / (mlen as f64)).sqrt().max(1.0e-6),
        &wd[bs..],
        720,
        max_lag,
        8,
        0,
        0.14,
        0.18,
        0.02,
        Some((&wdry[bs..bs + mlen], dry_e, dry_rms, &wdry[bs..])),
    )
}

#[allow(clippy::too_many_arguments)]
fn extract_echo_peaks(
    cs_dry_l: &[f32],
    cs_wet_l: &[f32],
    cs_wet_r: Option<&[f32]>,
    tail_energy_ratio: f64,
    comb_dry: &[f32],
    comb_wet_l: &[f32],
    comb_wet_r: Option<&[f32]>,
    tail_wet_l: &[f32],
    tail_wet_r: Option<&[f32]>,
    env_ms: &[f64],
    comb_rms: f64,
) -> TailDecayStats {
    let mut fit = TailDecayStats {
        bbd_hf_retain: 1.0,
        ..Default::default()
    };
    if env_ms.len() < 40 {
        return fit;
    }
    let mut peaks_l = find_motif_peaks_fine(comb_wet_l, tail_wet_l, 720);
    if peaks_l.is_empty() {
        peaks_l = find_motif_peaks_fine(comb_dry, tail_wet_l, 720);
    }
    let mut peaks_r = tail_wet_r.map_or_else(Vec::new, |tr| {
        let pr = find_motif_peaks_fine(comb_wet_r.unwrap_or(comb_wet_l), tr, 720);
        if pr.is_empty() {
            find_motif_peaks_fine(comb_wet_l, tr, 720)
        } else {
            pr
        }
    });
    if peaks_l.is_empty() && tail_energy_ratio > 0.04 {
        peaks_l = find_whitened_lag_peaks(cs_dry_l, cs_wet_l, cs_wet_l);
        if peaks_r.is_empty()
            && let Some(csr) = cs_wet_r
        {
            peaks_r = find_whitened_lag_peaks(cs_dry_l, cs_wet_l, csr);
        }
    }
    if let Some(&(lag1, _, _, g1)) = peaks_l.first() {
        fit.delay_l_samples = lag1;
        fit.bbd_gain_l = g1;
        if let Some(&(_, _, _, g2)) = peaks_l.get(1) {
            fit.bbd_fb = (g2 / g1.max(1.0e-6)).clamp(0.0, 0.96);
        }
    }
    if let Some(&(lag_r1, _, _, g_r1)) = peaks_r.first() {
        fit.delay_r_samples = lag_r1;
        fit.bbd_gain_r = g_r1;
    } else if fit.delay_l_samples > 0.0 {
        fit.delay_r_samples = fit.delay_l_samples;
        fit.bbd_gain_r = fit.bbd_gain_l;
    }
    if let Some(tr) = tail_wet_r.filter(|_| fit.delay_l_samples > 1.0 && fit.bbd_gain_l > 0.02) {
        let mlen = comb_wet_l.len().min(1_024);
        let off_r = (fit.delay_l_samples.round() as usize).saturating_sub(mlen);
        if off_r + mlen <= tr.len() {
            let motif = &comb_wet_l[comb_wet_l.len() - mlen..];
            let me: f64 = motif
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .max(1.0e-12);
            let (_, s_lr, g_lr) = corr_at_offset(motif, me, tr, off_r, rms_energy(motif), 1);
            if s_lr.abs() > 0.25 {
                fit.bbd_cross_fb =
                    (s_lr.signum() * (g_lr / fit.bbd_gain_l.max(1.0e-6))).clamp(-0.92, 0.92);
            }
        }
    }
    for &(lag, _, _, g) in &peaks_l {
        fit.delays_ms.push((lag / 48.0) as f32);
        fit.decay_gains.push(g as f32);
    }
    if fit.delays_ms.is_empty() && env_ms.len() > 60 {
        let motif_len_ms = (comb_dry.len().min(1_024) as f32) / 96.0;
        for m in 35..env_ms.len().saturating_sub(5) {
            let cur = env_ms[m];
            let prev = env_ms[m - 8..m].iter().sum::<f64>() / 8.0;
            if cur > 0.05 * comb_rms
                && cur > 1.35 * prev
                && cur >= env_ms[m - 1]
                && cur >= env_ms[m + 1]
            {
                fit.delays_ms.push((m as f32) + motif_len_ms);
                fit.decay_gains
                    .push((cur / comb_rms).clamp(0.0, 1.2) as f32);
                if fit.delays_ms.len() >= 4 {
                    break;
                }
            }
        }
    }
    fit
}

fn estimate_rt60_ms(env_ms: &[f64], comb_rms: f64) -> f64 {
    if env_ms.len() < 120 {
        return 0.0;
    }
    let post_30_peak = env_ms.iter().skip(30).copied().fold(0.0_f64, f64::max);
    if post_30_peak <= 0.012 * comb_rms {
        return 0.0;
    }
    let thresh = (comb_rms * 1.0e-3).max(1.0e-6);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (m, &e) in env_ms.iter().enumerate().skip(30) {
        if e > thresh {
            xs.push((m as f64) * 1.0e-3);
            ys.push(20.0 * e.log10());
        }
    }
    if xs.len() < 40 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&x, &y) in xs.iter().zip(&ys) {
        num += (x - mx) * (y - my);
        den += (x - mx) * (x - mx);
    }
    if den <= 1.0e-12 {
        return 0.0;
    }
    let slope_db_per_s = num / den;
    if slope_db_per_s < -1.0 {
        ((-60.0 / slope_db_per_s) * 1_000.0).clamp(40.0, 30_000.0)
    } else if env_ms.last().copied().unwrap_or(0.0) > 0.02 * comb_rms {
        15_000.0 // nearly infinite / freeze / very long (> 15s) tail
    } else {
        0.0
    }
}

pub(super) fn estimate_lfo_modulation(env_ms: &[f64]) -> (f64, f64) {
    if env_ms.len() < 200 {
        return (0.0, 0.0);
    }
    let mean = (env_ms.iter().sum::<f64>() / (env_ms.len() as f64)).max(1.0e-9);
    let norm: Vec<f64> = env_ms.iter().map(|&e| (e / mean) - 1.0).collect();
    let var: f64 = norm.iter().map(|&x| x * x).sum::<f64>() / (norm.len() as f64);
    let depth = var.sqrt().clamp(0.0, 1.0);
    if depth < 0.04 {
        return (0.0, depth);
    }
    let (mut best_hz, mut best_pow, mut sum_pow, mut count, mut hz) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0usize, 0.4_f64);
    while hz <= 14.0 {
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        let w = 2.0 * std::f64::consts::PI * hz * 1.0e-3;
        for (t, &x) in norm.iter().enumerate() {
            let (s, c) = (w * (t as f64)).sin_cos();
            re += x * c;
            im += x * s;
        }
        let p = re * re + im * im;
        if p > best_pow {
            best_pow = p;
            best_hz = hz;
        }
        sum_pow += p;
        count += 1;
        hz += 0.25;
    }
    let avg_pow = sum_pow / (count.max(1) as f64);
    if best_pow > 2.8 * avg_pow {
        (best_hz, depth)
    } else {
        (0.0, depth)
    }
}

pub(super) fn rms_energy(buf: &[f32]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    (buf.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / (buf.len() as f64))
        .sqrt()
}

pub(super) fn first_diff_hf_ratio(buf: &[f32]) -> f64 {
    if buf.len() < 4 {
        return 0.0;
    }
    let (mut e, mut de, mut prev) = (0.0_f64, 0.0_f64, f64::from(buf[0]));
    for &s in &buf[1..] {
        let x = f64::from(s);
        e += x * x;
        de += (x - prev) * (x - prev);
        prev = x;
    }
    (de / (e + 1.0e-18)).sqrt()
}

pub(super) fn detect_short_delay_and_chorus_lfo(
    dc_l: &[f32],
    wc_l: &[f32],
    coreset_wet_r: Option<&[f32]>,
    comb_wet_l: &[f32],
    comb_wet_r: Option<&[f32]>,
) -> (f64, f64, f64, f64) {
    let n = dc_l.len().min(wc_l.len());
    let (mut short_ms, mut short_gain) = (0.0_f64, 0.0_f64);
    if n >= 2_400 {
        let (mut dx, mut dw) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        let (mut xx, mut xw, mut ww) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 1..n {
            dx[i] = f64::from(dc_l[i]) - 0.92 * f64::from(dc_l[i - 1]);
            dw[i] = f64::from(wc_l[i]) - 0.92 * f64::from(wc_l[i - 1]);
            xx += dx[i] * dx[i];
            xw += dx[i] * dw[i];
            ww += dw[i] * dw[i];
        }
        let (g0, res_norm) = (
            if xx > 1.0e-12 { xw / xx } else { 0.0 },
            (xx * ww).sqrt().max(1.0e-12),
        );
        let (mut best_lag, mut best_score) = (0usize, 0.0_f64);
        for lag in 72..1_056_usize.min(n / 2) {
            let mut dot = 0.0_f64;
            for idx in (lag + 16..n).step_by(2) {
                dot += (dw[idx] - g0 * dx[idx]) * dx[idx - lag];
            }
            let sc = (dot.abs() * 2.0) / res_norm;
            if sc > best_score {
                best_score = sc;
                best_lag = lag;
            }
        }
        if best_score > 0.045 && best_lag >= 72 {
            short_ms = (best_lag as f64) / 48.0;
            short_gain = best_score.clamp(0.0, 1.5);
        }
    }
    let mut side_env = Vec::new();
    let blk = 48_usize;
    for (wl, opt_r) in [(wc_l, coreset_wet_r), (comb_wet_l, comb_wet_r)] {
        if let Some(wr) = opt_r {
            for m in 0..(wl.len().min(wr.len()) / blk) {
                let de: f64 = (m * blk..(m + 1) * blk)
                    .map(|i| {
                        let d = f64::from(wl[i]) - f64::from(wr[i]);
                        d * d
                    })
                    .sum();
                side_env.push((de / (blk as f64)).sqrt());
            }
        }
    }
    let (lfo_hz, lfo_depth) = if side_env.len() >= 160 {
        estimate_lfo_modulation(&side_env)
    } else {
        (0.0, 0.0)
    };
    (short_ms, short_gain, lfo_hz, lfo_depth)
}
