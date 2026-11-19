#![forbid(unsafe_code)]
//! `4096`-FFT Hann Welch Wiener LFO/comb lobe detector, dyadic `1/3`-octave Kautz pole placement,
//! broadband RMS calibration, and unitary first-order FIR allpass helpers (`incoherent_rig/comb_detect.rs`).

use std::f64::consts::{PI, TAU};

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, tier_kautz_banks};
use dnam_engine::stages::stage4_tank::v3_fdn::{V3_TANK_MAX_DECLARED_DELAY, V3Stage4TankParams};
use dnam_engine::stages::{V3MatryoshkaModel, V3TierReadout};

use crate::solver::backends::TrainerBackend;
use crate::stimulus::fft::{Complex64, fft_radix2_in_place};

use super::super::super::model_f64::V3TrainStem;
use super::super::render_tier_stereo;
use super::lfo_detect::DetectedLfoSpec;

pub(super) const PRIME_QUARTETS: [[usize; 4]; 6] = [
    [53, 67, 79, 89],
    [97, 131, 173, 223],
    [197, 283, 389, 509],
    [373, 521, 733, 947],
    [461, 643, 829, 1_019],
    [1_499, 2_203, 3_109, 4_157],
];

#[derive(Clone, Copy)]
pub(super) struct FdnTankSpec {
    pub(super) delays: [usize; 4],
    pub(super) g_loop: f32,
    pub(super) base_psi: f32,
    pub(super) wet_ratio: f64,
    pub(super) tap_gains: [f32; 4],
    pub(super) lfo_rate_hz: f32,
    pub(super) lfo_depth_samples: f32,
    pub(super) lfo_stereo_phase: f32,
}

pub(super) fn clamp_lfo_depth_for_delays(delays: &[usize; 4], depth: f32) -> f32 {
    if depth <= 0.0 || !depth.is_finite() {
        return 0.0;
    }
    let min_d = *delays.iter().min().unwrap_or(&4);
    let max_d = *delays.iter().max().unwrap_or(&1_016);
    let head_lo = min_d.saturating_sub(4) as f32;
    let head_hi = (V3_TANK_MAX_DECLARED_DELAY - 4).saturating_sub(max_d) as f32;
    depth.min(head_lo.min(head_hi)).clamp(0.0, 64.0)
}

impl FdnTankSpec {
    pub(super) fn to_params(self) -> V3Stage4TankParams {
        let wr = self.wet_ratio.clamp(0.0, 0.96);
        let depth = clamp_lfo_depth_for_delays(&self.delays, self.lfo_depth_samples);
        let rate = if depth > 0.0 {
            self.lfo_rate_hz.clamp(0.0, 16.0)
        } else {
            0.0
        };
        let g2 = (f64::from(self.g_loop) * f64::from(self.g_loop)).clamp(0.0, 0.995);
        let inv_loop_gain = (1.0 - g2).sqrt();
        let inject_gain = (0.50 * inv_loop_gain).clamp(0.03, 0.50) as f32;
        let wet_gain_comp = 0.50 * inv_loop_gain / f64::from(inject_gain);
        let mut tank = V3Stage4TankParams {
            delay_lengths: self.delays,
            g_loop: self.g_loop,
            sigma_knee: 8.0,
            base_psi: self.base_psi,
            inject_gain,
            dry_mix: (1.0 - wr).sqrt() as f32,
            wet_mix: (wr.sqrt() * wet_gain_comp).clamp(0.0, 1.90) as f32,
            r_dc: 0.9999,
            vca_thresh: 2.0,
            vca_kappa: 0.0,
            vca_min_gain: 1.0,
            vca_att: 0.99,
            vca_rel: 0.9995,
            tap_gains: self.tap_gains,
            lfo_rate_hz: rate,
            lfo_depth_samples: depth,
            lfo_stereo_phase: self.lfo_stereo_phase,
        };
        tank.clamp_in_place();
        tank
    }
}

#[derive(Clone, Copy)]
pub(super) struct CombSpec {
    pub(super) dl: usize,
    pub(super) dr: usize,
    pub(super) sigma_l: usize,
    pub(super) sigma_r: usize,
    pub(super) sl: f32,
    pub(super) sr: f32,
    pub(super) wr_l: f64,
    pub(super) wr_r: f64,
}

fn find_comb_lobe_windowed(
    inp: &[f32],
    tgt: &[f32],
    start: usize,
    end: usize,
    n_fft: usize,
    min_lag: usize,
    max_lag: usize,
) -> Option<(usize, usize, f32, f64)> {
    if end <= start + 256 || start < 1_000 || min_lag > max_lag || max_lag >= n_fft / 2 {
        return None;
    }
    let scan_end = end.min(start + 96_000);
    let avail = scan_end - start;
    let win_len = avail.min(n_fft);
    let hop = (win_len / 2).max(256);
    let denom = (win_len.max(2) - 1) as f64;
    let win: Vec<f64> = (0..win_len)
        .map(|j| 0.5 * (1.0 - (TAU * (j as f64) / denom).cos()))
        .collect();

    let mut s_yx = vec![Complex64::new(0.0, 0.0); n_fft];
    let mut s_xx = vec![0.0_f64; n_fft];
    let mut bx = vec![Complex64::new(0.0, 0.0); n_fft];
    let mut by = vec![Complex64::new(0.0, 0.0); n_fft];

    let mut win_offsets = Vec::new();
    let mut w0 = 0_usize;
    while w0 + win_len <= avail {
        win_offsets.push(w0);
        w0 += hop;
    }
    if avail > win_len && win_offsets.last().copied() != Some(avail - win_len) {
        win_offsets.push(avail - win_len);
    }
    if win_offsets.is_empty() {
        return None;
    }
    for off in win_offsets {
        let base = start + off;
        for j in 0..win_len {
            let wj = win[j];
            bx[j] = Complex64::new(f64::from(inp[base + j]) * wj, 0.0);
            by[j] = Complex64::new(f64::from(tgt[base + j]) * wj, 0.0);
        }
        bx[win_len..n_fft].fill(Complex64::new(0.0, 0.0));
        by[win_len..n_fft].fill(Complex64::new(0.0, 0.0));
        fft_radix2_in_place(&mut bx, false);
        fft_radix2_in_place(&mut by, false);
        for k in 0..n_fft {
            let (xr, xi) = (bx[k].re, bx[k].im);
            let (yr, yi) = (by[k].re, by[k].im);
            s_yx[k].re += yr * xr + yi * xi;
            s_yx[k].im += yi * xr - yr * xi;
            s_xx[k] += xr * xr + xi * xi;
        }
    }

    let mean_s_xx = (s_xx.iter().sum::<f64>() / (n_fft as f64)).max(1.0e-24);
    let reg = 1.0e-3 * mean_s_xx;
    let mut h_fft: Vec<Complex64> = s_yx
        .iter()
        .zip(&s_xx)
        .map(|(yx, &xx)| {
            let d = xx + reg;
            Complex64::new(yx.re / d, yx.im / d)
        })
        .collect();
    fft_radix2_in_place(&mut h_fft, true);

    let e_dry: f64 = h_fft[..32].iter().map(|c| c.re * c.re).sum();
    let a_dry: f64 = h_fft[..32].iter().map(|c| c.re.abs()).sum();

    let mut best_env = (-1.0_f64, min_lag);
    for d in min_lag..=max_lag {
        let env: f64 = (d.saturating_sub(4).max(min_lag)..=(d + 4).min(max_lag))
            .map(|j| h_fft[j].re.abs())
            .sum();
        if env > best_env.0 {
            best_env = (env, d);
        }
    }
    let d_peak = best_env.1;
    let lo = d_peak.saturating_sub(64).max(min_lag);
    let hi = (d_peak + 64).min(max_lag);
    let e_lobe: f64 = (lo..=hi).map(|d| h_fft[d].re * h_fft[d].re).sum();
    let a_lobe_abs: f64 = (lo..=hi).map(|d| h_fft[d].re.abs()).sum();
    if e_lobe < 0.012 * e_dry.max(1.0e-18) && a_lobe_abs < 0.22 * a_dry.max(1.0e-9) {
        return None;
    }

    let mut h_pk = 0.0_f64;
    for d in d_peak.saturating_sub(8).max(min_lag)..=(d_peak + 8).min(max_lag) {
        let hd = h_fft[d].re;
        if hd.abs() > h_pk.abs() {
            h_pk = hd;
        }
    }
    let sign = if h_pk >= 0.0 { 1.0_f32 } else { -1.0_f32 };
    let s_f64 = f64::from(sign);
    let thresh = 0.12 * h_pk.abs();

    let (mut w_sum, mut dw_sum, mut a_lobe) = (0.0_f64, 0.0_f64, 0.0_f64);
    for d in lo..=hi {
        let aligned = s_f64 * h_fft[d].re;
        a_lobe += aligned.max(0.0);
        let w = (aligned - thresh).max(0.0);
        w_sum += w;
        dw_sum += (d as f64) * w;
    }
    let (d_mean, sigma_d) = if w_sum <= 1.0e-18 {
        (d_peak, 0)
    } else {
        let mean_f = dw_sum / w_sum;
        let mean_u = (mean_f.round() as usize).clamp(min_lag, max_lag);
        let mut var_sum = 0.0_f64;
        for d in lo..=hi {
            let aligned = s_f64 * h_fft[d].re;
            let w = (aligned - thresh).max(0.0);
            let diff = (d as f64) - mean_f;
            var_sum += diff * diff * w;
        }
        let sig = ((var_sum / w_sum).sqrt().round() as usize).min(256);
        (mean_u, sig)
    };
    let rho = (a_lobe / a_dry.max(1.0e-9)).clamp(0.15, 1.5);
    let r_e = (e_lobe / e_dry.max(1.0e-18)).clamp(0.02, 2.5);
    let wr_rho = rho * rho / (1.0 + rho * rho);
    let eff_re = 2.0 * r_e * (1.0 + 0.35 * (sigma_d as f64));
    let wr_re = eff_re / (1.0 + eff_re);
    let wr_est = (0.65 * wr_rho + 0.35 * wr_re).clamp(0.08, 0.68);
    Some((d_mean, sigma_d, sign, wr_est))
}

fn find_dominant_comb_lobe(
    inp: &[f32],
    tgt: &[f32],
    start: usize,
    end: usize,
) -> Option<(usize, usize, f32, f64)> {
    let short = find_comb_lobe_windowed(inp, tgt, start, end, 4_096, 32, 1_000);
    let long = if end.saturating_sub(start) >= 65_536 {
        find_comb_lobe_windowed(inp, tgt, start, end, 65_536, 1_001, 32_000)
    } else {
        None
    };
    match (short, long) {
        (Some(s), Some(l)) if l.3 > s.3 * 1.15 => Some(l),
        (Some(s), _) => Some(s),
        (None, Some(l)) => Some(l),
        (None, None) => None,
    }
}

pub(super) fn detect_comb_spec(
    stem: &V3TrainStem,
    warmup: usize,
    mono_out: bool,
) -> Option<CombSpec> {
    let n = stem.usable_len();
    let start = warmup.max(1_000);
    let (dl, sigma_l, sl, wr_l) = find_dominant_comb_lobe(&stem.in_l, &stem.tgt_l, start, n)?;
    let (dr, sigma_r, sr, wr_r) = if mono_out {
        (dl, sigma_l, sl, wr_l)
    } else {
        find_dominant_comb_lobe(&stem.in_r, &stem.tgt_r, start, n)
            .unwrap_or((dl, sigma_l, sl, wr_l))
    };
    Some(CombSpec {
        dl,
        dr,
        sigma_l,
        sigma_r,
        sl,
        sr,
        wr_l,
        wr_r,
    })
}

pub(super) fn build_comb_candidates(
    c: CombSpec,
    lfo_opt: Option<DetectedLfoSpec>,
) -> Vec<FdnTankSpec> {
    let mut quartets: Vec<[usize; 4]> = Vec::new();
    for &alpha in &[0.85_f64, 1.15, 0.45, 0.0] {
        let dl_off = ((c.sigma_l as f64) * alpha).round() as usize;
        let dr_off = ((c.sigma_r as f64) * alpha).round() as usize;
        let q = [
            c.dl.saturating_sub(dl_off)
                .clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            c.dr.saturating_sub(dr_off)
                .clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            (c.dl + dl_off).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            (c.dr + dr_off).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
        ];
        if !quartets.contains(&q) {
            quartets.push(q);
        }
    }
    if c.sigma_l < 4 && c.sigma_r < 4 {
        let q = [
            c.dl.saturating_sub(16).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            c.dr.saturating_sub(16).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            (c.dl + 16).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            (c.dr + 16).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
        ];
        if !quartets.contains(&q) {
            quartets.push(q);
        }
    }
    if c.dl > 2_500 || c.dr > 2_500 {
        for q in [
            [
                1_499,
                2_311,
                c.dl.clamp(8, V3_TANK_MAX_DECLARED_DELAY),
                c.dr.clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            ],
            [
                c.dl.clamp(8, V3_TANK_MAX_DECLARED_DELAY),
                c.dr.clamp(8, V3_TANK_MAX_DECLARED_DELAY),
                3_371,
                4_679,
            ],
        ] {
            if !quartets.contains(&q) {
                quartets.push(q);
            }
        }
    }
    let wr_mid = 0.5 * (c.wr_l + c.wr_r);
    let wr_cands = [
        (0.65 * wr_mid).clamp(0.08, 0.65),
        wr_mid,
        (1.35 * wr_mid).clamp(0.14, 0.72),
        (1.75 * wr_mid).clamp(0.20, 0.76),
    ];
    let c_taps = [c.sl, c.sr, c.sl, c.sr];
    let mut out = Vec::with_capacity(quartets.len() * wr_cands.len() + 18);

    if let Some(lfo) = lfo_opt {
        let d_l = (lfo.mean_delay_l.round() as usize).clamp(8, V3_TANK_MAX_DECLARED_DELAY);
        let d_r = (lfo.mean_delay_r.round() as usize).clamp(8, V3_TANK_MAX_DECLARED_DELAY);
        let d_1v = [d_l, d_r, d_l, d_r];
        let taps_1v = [2.0 * c.sl, 2.0 * c.sr, 0.0, 0.0];
        for &depth_scale in &[1.0_f32, 0.75] {
            let depth = clamp_lfo_depth_for_delays(&d_1v, depth_scale * lfo.depth_samples);
            if depth >= 1.5 {
                for &wr in &wr_cands {
                    out.push(FdnTankSpec {
                        delays: d_1v,
                        g_loop: 0.0,
                        base_psi: 0.0,
                        wet_ratio: wr,
                        tap_gains: taps_1v,
                        lfo_rate_hz: lfo.rate_hz,
                        lfo_depth_samples: depth,
                        lfo_stereo_phase: lfo.stereo_phase,
                    });
                }
            }
        }
        let d_2v = [
            d_l.saturating_sub(6).max(8),
            d_r.saturating_sub(6).max(8),
            (d_l + 6).min(V3_TANK_MAX_DECLARED_DELAY),
            (d_r + 6).min(V3_TANK_MAX_DECLARED_DELAY),
        ];
        let depth_2v = clamp_lfo_depth_for_delays(&d_2v, 0.85 * lfo.depth_samples);
        if depth_2v >= 1.5 {
            let taps_2v = [1.414 * c.sl, 1.414 * c.sr, 1.414 * c.sl, 1.414 * c.sr];
            for &wr in &wr_cands[1..3] {
                out.push(FdnTankSpec {
                    delays: d_2v,
                    g_loop: 0.0,
                    base_psi: 0.0,
                    wet_ratio: wr,
                    tap_gains: taps_2v,
                    lfo_rate_hz: lfo.rate_hz,
                    lfo_depth_samples: depth_2v,
                    lfo_stereo_phase: lfo.stereo_phase,
                });
            }
        }
    }

    let smear_rate = lfo_opt.map_or(1.45, |l| l.rate_hz.max(0.75));
    let smear_depth_raw = lfo_opt.map_or(10.0_f32, |l| l.depth_samples.max(8.0));
    for delays in &quartets {
        let smear_depth = clamp_lfo_depth_for_delays(delays, smear_depth_raw);
        for &wr in &wr_cands {
            if smear_depth >= 1.5 {
                out.push(FdnTankSpec {
                    delays: *delays,
                    g_loop: 0.0,
                    base_psi: 0.0,
                    wet_ratio: wr,
                    tap_gains: c_taps,
                    lfo_rate_hz: smear_rate,
                    lfo_depth_samples: smear_depth,
                    lfo_stereo_phase: std::f32::consts::FRAC_PI_2,
                });
            }
        }
    }
    for &delays in &quartets {
        for &wr in &wr_cands {
            out.push(FdnTankSpec {
                delays,
                g_loop: 0.0,
                base_psi: 0.0,
                wet_ratio: wr,
                tap_gains: c_taps,
                lfo_rate_hz: 0.0,
                lfo_depth_samples: 0.0,
                lfo_stereo_phase: 0.0,
            });
        }
    }
    let mod_cap = V3_TANK_MAX_DECLARED_DELAY.saturating_sub(10);
    let mod_delays = [
        c.dl.saturating_sub(37).clamp(16, mod_cap),
        c.dr.saturating_sub(19).clamp(16, mod_cap),
        (c.dl + 23).clamp(16, mod_cap),
        (c.dr + 53).clamp(16, mod_cap),
    ];
    let depth = clamp_lfo_depth_for_delays(&mod_delays, smear_depth_raw);
    if depth >= 1.5 {
        for &gl in &[0.0_f32, 0.25] {
            out.push(FdnTankSpec {
                delays: mod_delays,
                g_loop: gl,
                base_psi: std::f32::consts::FRAC_PI_4,
                wet_ratio: wr_mid.clamp(0.18, 0.65),
                tap_gains: [0.65 * c.sl, 0.65 * c.sr, -0.55 * c.sl, 0.55 * c.sr],
                lfo_rate_hz: smear_rate,
                lfo_depth_samples: depth,
                lfo_stereo_phase: std::f32::consts::FRAC_PI_2,
            });
        }
    }
    out
}

pub(super) fn compute_broadband_rms(
    out_l: &[f32],
    out_r: &[f32],
    warmup: usize,
    mono_out: bool,
) -> f64 {
    let n = out_l.len().min(out_r.len());
    if n <= warmup + 16 {
        return 0.0;
    }
    let mut ss = 0.0_f64;
    for i in warmup..n {
        let l = f64::from(out_l[i]);
        ss += if mono_out {
            l * l
        } else {
            let r = f64::from(out_r[i]);
            0.5 * (l * l + r * r)
        };
    }
    (ss / ((n - warmup) as f64)).sqrt()
}

fn snap_stage4_rms_mode<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    warmup: usize,
    target_rms: f64,
    mono_out: bool,
    skip_tol_db: f64,
) -> f64 {
    let t = tier as usize;
    let n = stem.usable_len();
    let cold = model.extract_tier(tier);
    let (ol, or) = render_tier_stereo::<B>(&cold, &stem.in_l[..n], &stem.in_r[..n]);
    let cur = compute_broadband_rms(&ol, &or, warmup, mono_out);
    if cur <= 1.0e-9
        || target_rms <= 1.0e-9
        || (skip_tol_db > 0.0 && (20.0 * (cur / target_rms).log10()).abs() <= skip_tol_db)
    {
        return cur;
    }
    let ratio = (target_rms / cur).clamp(0.05, 8.0);
    let old_dry = f64::from(model.stage4[t].dry_mix);
    let old_wet = f64::from(model.stage4[t].wet_mix);
    let max_mix = old_dry.abs().max(old_wet.abs());
    if (0.08..=1.92).contains(&(max_mix * ratio)) {
        model.stage4[t].dry_mix = (old_dry * ratio) as f32;
        model.stage4[t].wet_mix = (old_wet * ratio) as f32;
        model.stage4[t].clamp_in_place();
        let actual_scale = if max_mix > 1.0e-9 {
            let new_max = f64::from(
                model.stage4[t]
                    .dry_mix
                    .abs()
                    .max(model.stage4[t].wet_mix.abs()),
            );
            new_max / max_mix
        } else {
            ratio
        };
        return cur * actual_scale;
    }
    let pure_linear_feed = model.stage0.clean_sub_mix.abs() < 1.0e-7
        && model.readouts[t]
            .direct_s2_u_gain
            .iter()
            .chain(&model.readouts[t].direct_s2_hp_gain)
            .all(|&g| g.abs() < 1.0e-7)
        && model.readouts[t]
            .svf_direct_taps_l
            .iter()
            .chain(&model.readouts[t].svf_direct_taps_r)
            .all(|&v| v.abs() < 1.0e-7);
    if pure_linear_feed {
        let s3_scale = ((max_mix * ratio / 1.10).clamp(0.15, 12.0)) as f32;
        for tap in &mut model.readouts[t].fir_l {
            *tap *= s3_scale;
        }
        for tap in &mut model.readouts[t].fir_r {
            *tap *= s3_scale;
        }
        for (re, im) in &mut model.readouts[t].kautz_c_l {
            *re *= s3_scale;
            *im *= s3_scale;
        }
        for (re, im) in &mut model.readouts[t].kautz_c_r {
            *re *= s3_scale;
            *im *= s3_scale;
        }
        for w in &mut model.readouts[t].direct_s2_w[0] {
            *w *= s3_scale;
        }
        for w in &mut model.readouts[t].direct_s2_w[1] {
            *w *= s3_scale;
        }
        model.readouts[t].mask_to_tier(tier);
        let cold2 = model.extract_tier(tier);
        let (ol2, or2) = render_tier_stereo::<B>(&cold2, &stem.in_l[..n], &stem.in_r[..n]);
        let cur2 = compute_broadband_rms(&ol2, &or2, warmup, mono_out);
        if cur2 <= 1.0e-9 {
            return cur2;
        }
        let r2 = (target_rms / cur2).clamp(0.05, 8.0);
        let r2_uniform = r2.min(1.95 / max_mix.max(1.0e-9)) as f32;
        model.stage4[t].dry_mix *= r2_uniform;
        model.stage4[t].wet_mix *= r2_uniform;
        model.stage4[t].clamp_in_place();
        return cur2 * f64::from(r2_uniform);
    }
    let uniform_f = ratio.min(1.95 / max_mix.max(1.0e-9)) as f32;
    model.stage4[t].dry_mix *= uniform_f;
    model.stage4[t].wet_mix *= uniform_f;
    model.stage4[t].clamp_in_place();
    cur * f64::from(uniform_f)
}

pub(super) fn calibrate_stage4_rms<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    warmup: usize,
    target_rms: f64,
    mono_out: bool,
) -> f64 {
    snap_stage4_rms_mode::<B>(model, tier, stem, warmup, target_rms, mono_out, 0.0)
}

pub(super) fn snap_stage4_broadband_rms<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    warmup: usize,
    target_rms: f64,
    mono_out: bool,
) -> f64 {
    snap_stage4_rms_mode::<B>(model, tier, stem, warmup, target_rms, mono_out, 0.003)
}

fn apply_kautz_pyramid(
    model: &mut V3MatryoshkaModel,
    sr_hz: f64,
    pyramid: &[(f64, f64); MAX_KAUTZ_POLES],
) {
    for (slot, &(f, bw)) in pyramid.iter().enumerate() {
        let r = (-PI * (bw * f) / sr_hz).exp().clamp(0.55, 0.9985) as f32;
        let th = ((TAU * f / sr_hz) as f32).clamp(1.0e-4, (PI as f32) - 1.0e-4);
        model.kautz_poles[slot] = (r, th);
    }
}

pub(super) fn apply_lf_kautz_poles(model: &mut V3MatryoshkaModel, sr_hz: f64) {
    // 4-bank pyramid giving `Lite`/`Full` dense LF coverage alongside `Pro` midrange bridges and `1x` `s2_modal`.
    const LF_KAUTZ_PYRAMID: [(f64, f64); MAX_KAUTZ_POLES] = [
        (38.0, 0.30),
        (74.0, 0.30),
        (135.0, 0.30),
        (235.0, 0.30),
        (27.0, 0.28),
        (53.0, 0.28),
        (100.0, 0.28),
        (180.0, 0.28),
        (48.0, 0.26),
        (96.0, 0.24),
        (310.0, 0.27),
        (780.0, 0.27),
        (64.0, 0.25),
        (485.0, 0.27),
        (1_320.0, 0.26),
        (1_750.0, 0.26),
    ];
    apply_kautz_pyramid(model, sr_hz, &LF_KAUTZ_PYRAMID);
}

pub(super) fn apply_broadband_kautz_poles(model: &mut V3MatryoshkaModel, sr_hz: f64) {
    // Sub-bass priority + midrange/HF bridges for nonlinear GRU tracks without `s2_modal` at Pro/Ultra.
    const BROADBAND_KAUTZ_PYRAMID: [(f64, f64); MAX_KAUTZ_POLES] = [
        (36.0, 0.30),
        (92.0, 0.30),
        (225.0, 0.30),
        (560.0, 0.30),
        (25.0, 0.28),
        (58.0, 0.28),
        (145.0, 0.28),
        (355.0, 0.28),
        (45.0, 0.26),
        (108.0, 0.26),
        (880.0, 0.28),
        (1_380.0, 0.28),
        (29.5, 0.26),
        (72.0, 0.26),
        (180.0, 0.25),
        (1_120.0, 0.26),
    ];
    apply_kautz_pyramid(model, sr_hz, &BROADBAND_KAUTZ_PYRAMID);
}

pub(super) fn clear_readouts(model: &mut V3MatryoshkaModel) {
    for ro in &mut model.readouts {
        *ro = V3TierReadout::default();
    }
}

fn allpass_rotor_at_theta(a_c: f32, k_stages: usize, theta: f32, extra_phi: f32) -> (f32, f32) {
    let (st, ct) = f64::from(theta).sin_cos();
    let a = f64::from(a_c);
    let denom = (1.0 - 2.0 * a * ct + a * a).max(1.0e-12);
    let u1_re = (ct * (1.0 + a * a) - 2.0 * a) / denom;
    let u1_im = (-st * (1.0 - a * a)) / denom;
    let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
    for _ in 0..k_stages {
        (cr, ci) = (cr * u1_re - ci * u1_im, cr * u1_im + ci * u1_re);
    }
    let (sp, cp) = f64::from(extra_phi).sin_cos();
    ((cr * cp - ci * sp) as f32, (cr * sp + ci * cp) as f32)
}

pub(super) fn apply_channel_phase_detune(
    model: &mut V3MatryoshkaModel,
    tier: Tier,
    ch: usize,
    fir_alpha: f32,
    kautz_phi: f32,
) {
    let t = tier as usize;
    let fir_len = ((t + 1) * 16).min(64);
    let k_cnt = tier_kautz_banks(tier) * 4;
    let k_stages = ((fir_alpha.abs() / 0.38).ceil() as usize).clamp(1, 3);
    let a_c = (fir_alpha / (k_stages as f32)).clamp(-0.40, 0.40);
    let fir = if ch == 0 {
        &mut model.readouts[t].fir_l
    } else {
        &mut model.readouts[t].fir_r
    };
    for _ in 0..k_stages {
        let mut s = 0.0_f32;
        for tap in &mut fir[..fir_len] {
            let x = *tap;
            let y = -a_c * x + s;
            s = x + a_c * y;
            *tap = y;
        }
    }

    let extra_phi = 0.06 * kautz_phi;
    let poles = model.kautz_poles;
    let kc = if ch == 0 {
        &mut model.readouts[t].kautz_c_l
    } else {
        &mut model.readouts[t].kautz_c_r
    };
    for (m, (cr, ci)) in kc[..k_cnt].iter_mut().enumerate() {
        let (c_m, s_m) = allpass_rotor_at_theta(a_c, k_stages, poles[m].1, extra_phi);
        let nr = *cr * c_m - *ci * s_m;
        let ni = *cr * s_m + *ci * c_m;
        *cr = nr;
        *ci = ni;
    }

    if !model.tier_os2x[t]
        && model
            .rate_1x_cal
            .as_ref()
            .is_some_and(|c| (c.blocks[0].b_z[0] - 8.0).abs() < 1.0e-3)
    {
        const S2_FREQS_HZ: [f32; 6] = [380.0, 620.0, 1_020.0, 1_620.0, 2_550.0, 4_000.0];
        let sr = model.tone_svf.sample_rate_hz.max(44_100.0);
        for b in 0..3 {
            for pair in 0..2 {
                let th_p = core::f32::consts::TAU * S2_FREQS_HZ[b * 2 + pair] / sr;
                let (c_p, s_p) = allpass_rotor_at_theta(a_c, k_stages, th_p, extra_phi);
                let l0 = b * 12 + pair * 2;
                let l1 = l0 + 1;
                let w0 = model.readouts[t].direct_s2_w[ch][l0];
                let w1 = model.readouts[t].direct_s2_w[ch][l1];
                model.readouts[t].direct_s2_w[ch][l0] = w0 * c_p + w1 * s_p;
                model.readouts[t].direct_s2_w[ch][l1] = w1 * c_p - w0 * s_p;
            }
        }
    }

    if model.mono_out && ch == 0 {
        model.readouts[t].fir_r = model.readouts[t].fir_l;
        model.readouts[t].kautz_c_r = model.readouts[t].kautz_c_l;
        model.readouts[t].direct_s2_w[1] = model.readouts[t].direct_s2_w[0];
    }
}
