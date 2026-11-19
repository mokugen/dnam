#![forbid(unsafe_code)]
//! Discrete early room-reflection (`64..=960` sample lags) identification & causal `fp16`
//! Stage IV FDN deconvolution (`room_lti`).

mod wiener_echo;

use dnam_engine::stages::stage4_tank::v3_fdn::{
    V3_TANK_MAX_DECLARED_DELAY, V3Stage4TankHot, V3Stage4TankParams,
};
use dnam_engine::stages::{V3MatryoshkaModel, V3TierReadout};
use dnam_engine::{FtzDazGuard, Tier};

use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::backends::TrainerBackend;
use crate::solver::canonical_besr_warmup;

use super::super::adjoint::{AdjointWorkspace, calibrate_1x_tiers_with_backend};
use super::super::basis::solve_matryoshka_varpro_cholesky_with_backend;
use super::super::lbfgs::run_lbfgs_with_ws;
use super::super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset, V3TrainStem};
use super::super::omp::{
    repair_collided_kautz_poles, select_omp_kautz_poles_matryoshka_with_backend,
};
use super::super::polish::polish_f32_matryoshka_readouts_with_cert_stem;
use super::incoherent_rig::s2_modal;
use super::{Stem0CertCaches, render_tier_stereo, summaries_monotone, worst_ultra_besr_db};

const MIN_ECHO_LAG: usize = 64;
const MAX_ECHO_LAG: usize = 960;
const MIN_TAP_SEP: usize = 16;
const MIN_SCAN_SAMPLES: usize = 256;
const MAX_SCAN_SAMPLES: usize = 8_192;
const ECHO_R2_GATE: f64 = 0.12;
const MAX_CHANNEL_TAP_L1: f64 = 0.85;
const MIN_ROOM_BESR_IMPROVEMENT_DB: f64 = 2.0;

#[derive(Clone, Copy, Debug)]
struct ChannelEchoFit {
    d1: usize,
    d2: usize,
    g1: f64,
    g2: f64,
    r2: f64,
}

fn eval_2tap_chan(
    u: &[f32],
    res: &[f32],
    start: usize,
    end: usize,
    d1: usize,
    d2: usize,
) -> Option<(f64, f64, f64)> {
    let (mut s11, mut s12, mut s22) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut b1, mut b2, mut r_ss) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in start..end {
        let x1 = f64::from(u[i - d1]);
        let x2 = f64::from(u[i - d2]);
        let r = f64::from(res[i]);
        s11 += x1 * x1;
        s12 += x1 * x2;
        s22 += x2 * x2;
        b1 += x1 * r;
        b2 += x2 * r;
        r_ss += r * r;
    }
    if r_ss <= 1.0e-12 || s11 <= 1.0e-12 || s22 <= 1.0e-12 {
        return None;
    }
    let reg = 1.0e-9 * s11.max(s22);
    let a11 = s11 + reg;
    let a22 = s22 + reg;
    let det = a11 * a22 - s12 * s12;
    if !det.is_finite() || det.abs() <= 1.0e-18 {
        return None;
    }
    let g1 = (b1 * a22 - b2 * s12) / det;
    let g2 = (a11 * b2 - s12 * b1) / det;
    let mut sse = 0.0_f64;
    for i in start..end {
        let err = f64::from(res[i]) - g1 * f64::from(u[i - d1]) - g2 * f64::from(u[i - d2]);
        sse += err * err;
    }
    Some((g1, g2, (1.0 - sse / r_ss).clamp(0.0, 1.0)))
}

fn find_best_single_lag(
    u: &[f32],
    res: &[f64],
    start: usize,
    scan_end: usize,
    exclude_d: Option<usize>,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64, f64)> = None;
    for d in MIN_ECHO_LAG..=MAX_ECHO_LAG {
        if exclude_d.is_some_and(|ex| ex.abs_diff(d) < MIN_TAP_SEP) {
            continue;
        }
        let (mut xy, mut xx) = (0.0_f64, 0.0_f64);
        for i in start..scan_end {
            let x = f64::from(u[i - d]);
            xy += x * res[i];
            xx += x * x;
        }
        if xx <= 1.0e-12 {
            continue;
        }
        let explained = (xy * xy) / xx;
        if explained.is_finite() && best.is_none_or(|b| explained > b.1) {
            best = Some((d, explained, xy / xx));
        }
    }
    best.map(|(d, _, g)| (d, g))
}

fn fit_channel_echo_2tap(u: &[f32], tgt: &[f32], warmup: usize) -> Option<ChannelEchoFit> {
    let n = u.len().min(tgt.len());
    let start = warmup.max(MAX_ECHO_LAG + 18);
    if n <= start + MIN_SCAN_SAMPLES {
        return None;
    }
    let scan_end = n.min(start + MAX_SCAN_SAMPLES);
    let mut u_hp = vec![0.0_f32; n];
    let mut res_hp = vec![0.0_f32; n];
    let mut res_f64 = vec![0.0_f64; n];
    let mut r_ss_scan = 0.0_f64;
    let diff_at = |i: usize| f64::from(tgt[i] - u[i]);
    for i in 2..n {
        u_hp[i] = (f64::from(u[i]) - 2.0 * f64::from(u[i - 1]) + f64::from(u[i - 2])) as f32;
        let d_hp = diff_at(i) - 2.0 * diff_at(i - 1) + diff_at(i - 2);
        res_hp[i] = d_hp as f32;
        res_f64[i] = d_hp;
        if i >= start && i < scan_end {
            r_ss_scan += d_hp * d_hp;
        }
    }
    if r_ss_scan <= 1.0e-12 {
        return None;
    }

    let (d1, g1_init) = find_best_single_lag(&u_hp, &res_f64, start, scan_end, None)?;
    let mut res_perp = res_f64;
    for i in start..scan_end {
        res_perp[i] -= g1_init * f64::from(u_hp[i - d1]);
    }
    let d2_fallback = if d1 + 64 <= MAX_ECHO_LAG {
        d1 + 64
    } else {
        d1 - 64
    };
    let d2 = find_best_single_lag(&u_hp, &res_perp, start, scan_end, Some(d1))
        .map_or(d2_fallback, |(d, _)| d);

    let (mut g1, mut g2, mut r2) = eval_2tap_chan(&u_hp, &res_hp, start, n, d1, d2)?;
    if g2.abs() < 0.02 {
        // Check 1-tap energy vs 2-tap on HP pre-emphasized residuals.
        let (mut xy, mut xx, mut rr) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in start..n {
            let x = f64::from(u_hp[i - d1]);
            let r = f64::from(res_hp[i]);
            xy += x * r;
            xx += x * x;
            rr += r * r;
        }
        if xx > 1.0e-12 && rr > 1.0e-12 {
            g1 = xy / xx;
            g2 = 0.0;
            r2 = ((xy * xy) / (xx * rr)).clamp(0.0, 1.0);
        }
    }
    let l1 = g1.abs() + g2.abs();
    if l1 > MAX_CHANNEL_TAP_L1 {
        let scale = MAX_CHANNEL_TAP_L1 / l1;
        g1 *= scale;
        g2 *= scale;
    }
    Some(ChannelEchoFit { d1, d2, g1, g2, r2 })
}

fn build_feedforward_tank(
    fit_l: ChannelEchoFit,
    fit_r: Option<ChannelEchoFit>,
) -> V3Stage4TankParams {
    let fr = fit_r.unwrap_or(fit_l);
    let mut tank = V3Stage4TankParams {
        delay_lengths: [
            fit_l.d1.clamp(4, V3_TANK_MAX_DECLARED_DELAY),
            fr.d1.clamp(4, V3_TANK_MAX_DECLARED_DELAY),
            fit_l.d2.clamp(4, V3_TANK_MAX_DECLARED_DELAY),
            fr.d2.clamp(4, V3_TANK_MAX_DECLARED_DELAY),
        ],
        g_loop: 0.0,
        sigma_knee: 8.0,
        base_psi: 0.0,
        inject_gain: 1.0,
        dry_mix: 1.0,
        wet_mix: 1.0,
        r_dc: 0.9999,
        vca_thresh: 2.0,
        vca_kappa: 0.0,
        vca_min_gain: 1.0,
        vca_att: 0.99,
        vca_rel: 0.9995,
        tap_gains: [fit_l.g1 as f32, fr.g1 as f32, fit_l.g2 as f32, fr.g2 as f32],
        ..V3Stage4TankParams::identity()
    };
    tank.clamp_in_place();
    tank
}

fn estimate_tank_from_prediction(
    u_l: &[f32],
    u_r: &[f32],
    stem0: &V3TrainStem,
    mono_out: bool,
) -> Option<V3Stage4TankParams> {
    let warmup = stem0.warmup.max(canonical_besr_warmup(stem0.usable_len()));
    let fit_l = fit_channel_echo_2tap(u_l, &stem0.tgt_l, warmup)?;
    let fit_r = if mono_out {
        None
    } else {
        fit_channel_echo_2tap(u_r, &stem0.tgt_r, warmup)
    };
    let max_r2 = fit_l.r2.max(fit_r.map_or(0.0, |fr| fr.r2));
    if max_r2 < ECHO_R2_GATE {
        return None;
    }
    Some(build_feedforward_tank(fit_l, fit_r))
}

fn deconvolve_dataset_with_stage4<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    tank: &V3Stage4TankParams,
    mono_out: bool,
) -> V3TrainDataset {
    let _guard = FtzDazGuard::new();
    let tg = tank.tap_gains;
    let mut stems = Vec::with_capacity(dataset.stems.len());
    for (idx, st) in dataset.stems.iter().enumerate() {
        let n = st.usable_len();
        let mut hot = V3Stage4TankHot::from_cold(tank);
        let mut u_tgt_l = Vec::with_capacity(n);
        let mut u_tgt_r = Vec::with_capacity(n);
        for i in 0..n {
            let rf = hot.peek_integer_taps::<B>();
            let wet_l = tg[0] * rf[0] + tg[2] * rf[2];
            let wet_r = tg[1] * rf[1] + tg[3] * rf[3];
            let ul = (st.tgt_l[i] - wet_l).clamp(-32.0, 32.0);
            let ur = if mono_out {
                ul
            } else {
                (st.tgt_r[i] - wet_r).clamp(-32.0, 32.0)
            };
            let _ = hot.step_stereo::<B>(ul, ur);
            u_tgt_l.push(ul);
            u_tgt_r.push(ur);
        }
        stems.push((
            V3TrainStem {
                in_l: st.in_l[..n].to_vec(),
                in_r: st.in_r[..n].to_vec(),
                tgt_l: u_tgt_l,
                tgt_r: u_tgt_r,
                warmup: st.warmup,
            },
            dataset.stem_weights.get(idx).copied().unwrap_or(1.0),
        ));
    }
    V3TrainDataset::new_weighted(stems, dataset.sample_rate_hz)
}

fn eval_chan_mse_post_warmup(sig: &[f32], tgt: &[f32], warmup: usize) -> f64 {
    let n = sig.len().min(tgt.len());
    if n <= warmup + 8 {
        return f64::INFINITY;
    }
    let (mut err, mut den) = (0.0_f64, 0.0_f64);
    for i in warmup..n {
        let t = f64::from(tgt[i]);
        let d = t - f64::from(sig[i]);
        err += d * d;
        den += t * t;
    }
    err / den.max(1.0e-18)
}

fn eval_1x_max_mse<B: TrainerBackend>(model: &V3MatryoshkaModel, stem0: &V3TrainStem) -> f64 {
    let n = stem0.usable_len();
    let w = stem0.warmup.min(n.saturating_sub(1));
    let mut worst = 0.0_f64;
    for tier in [Tier::Lite, Tier::Full] {
        let (ol, or) = render_tier_stereo::<B>(
            &model.extract_tier(tier),
            &stem0.in_l[..n],
            &stem0.in_r[..n],
        );
        worst = worst.max(eval_chan_mse_post_warmup(&ol, &stem0.tgt_l, w));
        if !model.mono_out {
            worst = worst.max(eval_chan_mse_post_warmup(&or, &stem0.tgt_r, w));
        }
    }
    worst
}

fn refit_stage3_on_deconv<B: TrainerBackend>(
    seed_cold: &V3MatryoshkaModel,
    deconv_ds: &V3TrainDataset,
    mono_out: bool,
    pure_linear: bool,
) -> V3MatryoshkaModel {
    let mut base_no_s4 = seed_cold.clone();
    base_no_s4.stage4 = [V3Stage4TankParams::identity(); 4];
    let mut opt = Box::new(V3MatryoshkaOptF64::from_matryoshka_cold(&base_no_s4));
    opt.active_tier_weights = [0.18, 0.20, 0.26, 0.36];
    let ds_1x_len = if pure_linear { 24_576 } else { 10_240 };
    let ds_1x = deconv_ds
        .primary_subwindow_opts(ds_1x_len, false, false)
        .with_spectral_weights(0.0, 0.0, 0.0);
    let mut ws = AdjointWorkspace::default();
    let fit_ds_opt = (pure_linear && deconv_ds.stems.len() >= 2).then(|| {
        let pairs: Vec<(V3TrainStem, f64)> = deconv_ds
            .stems
            .iter()
            .enumerate()
            .map(|(si, st)| {
                let w = match si {
                    0 => 1.0,
                    1 => 8.0,
                    _ => deconv_ds.stem_weights.get(si).copied().unwrap_or(1.0),
                };
                (st.clone(), w)
            })
            .collect();
        V3TrainDataset::new_weighted_opts(pairs, deconv_ds.sample_rate_hz, false, false)
    });
    let stage3_ds = fit_ds_opt.as_ref().unwrap_or(deconv_ds);
    select_omp_kautz_poles_matryoshka_with_backend::<B>(&mut opt, stage3_ds);
    repair_collided_kautz_poles(&mut opt, deconv_ds.sample_rate_hz);
    solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut opt, stage3_ds, 2);
    if pure_linear {
        let ds_poles = stage3_ds
            .curriculum_subwindow_opts(24_576, 16_384, false, false)
            .with_spectral_weights(0.0, 0.0, 0.0);
        let _ = run_lbfgs_with_ws::<B>(&mut opt, &ds_poles, &mut ws, 18, false, true, true);
        repair_collided_kautz_poles(&mut opt, deconv_ds.sample_rate_hz);
        solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut opt, stage3_ds, 2);
    }
    calibrate_1x_tiers_with_backend::<B>(&mut opt, &ds_1x, &mut ws);
    let ds_polish = deconv_ds
        .for_stage3_polish(24_576)
        .with_spectral_weights(0.0, 0.0, 0.05);
    polish_f32_matryoshka_readouts_with_cert_stem::<B>(
        &mut opt,
        &ds_polish,
        deconv_ds.stems.first(),
    );
    let mut out = opt.to_matryoshka_cold();
    out.mono_out = mono_out;
    out.stage0 = seed_cold.stage0;
    if pure_linear && let Some(stem0) = deconv_ds.stems.first() {
        let mut modal_cand = out.clone();
        if fit_ds_opt.is_some() {
            s2_modal::fit_1x_modal_tier_linear_cholesky_dataset::<B>(
                &mut modal_cand,
                Tier::Lite,
                stage3_ds,
            );
            s2_modal::fit_1x_modal_tier_linear_cholesky_dataset::<B>(
                &mut modal_cand,
                Tier::Full,
                stage3_ds,
            );
        } else {
            s2_modal::fit_1x_modal_tier_linear_cholesky::<B>(&mut modal_cand, Tier::Lite, stem0);
            s2_modal::fit_1x_modal_tier_linear_cholesky::<B>(&mut modal_cand, Tier::Full, stem0);
        }
        if eval_1x_max_mse::<B>(&modal_cand, stem0) < eval_1x_max_mse::<B>(&out, stem0) {
            out = modal_cand;
        }
    }
    out
}

fn blend_tier_readouts(
    a: &V3TierReadout,
    b: &V3TierReadout,
    alpha: f32,
    tier: Tier,
) -> V3TierReadout {
    let om = 1.0 - alpha;
    let mut out = V3TierReadout::zeros();
    for i in 0..7 {
        out.svf_direct_taps_l[i] = alpha * a.svf_direct_taps_l[i] + om * b.svf_direct_taps_l[i];
        out.svf_direct_taps_r[i] = alpha * a.svf_direct_taps_r[i] + om * b.svf_direct_taps_r[i];
    }
    for ch in 0..2 {
        out.pre_cab_u_gain[ch] = alpha * a.pre_cab_u_gain[ch] + om * b.pre_cab_u_gain[ch];
        out.pre_cab_hp_gain[ch] = alpha * a.pre_cab_hp_gain[ch] + om * b.pre_cab_hp_gain[ch];
        out.direct_s2_u_gain[ch] = alpha * a.direct_s2_u_gain[ch] + om * b.direct_s2_u_gain[ch];
        out.direct_s2_hp_gain[ch] = alpha * a.direct_s2_hp_gain[ch] + om * b.direct_s2_hp_gain[ch];
        for l in 0..out.pre_cab_w[ch].len() {
            out.pre_cab_w[ch][l] = alpha * a.pre_cab_w[ch][l] + om * b.pre_cab_w[ch][l];
            out.direct_s2_w[ch][l] = alpha * a.direct_s2_w[ch][l] + om * b.direct_s2_w[ch][l];
        }
    }
    for k in 0..64 {
        out.fir_l[k] = alpha * a.fir_l[k] + om * b.fir_l[k];
        out.fir_r[k] = alpha * a.fir_r[k] + om * b.fir_r[k];
    }
    for m in 0..out.kautz_c_l.len() {
        out.kautz_b_gamma[m] = alpha * a.kautz_b_gamma[m] + om * b.kautz_b_gamma[m];
        out.kautz_c_l[m] = (
            alpha * a.kautz_c_l[m].0 + om * b.kautz_c_l[m].0,
            alpha * a.kautz_c_l[m].1 + om * b.kautz_c_l[m].1,
        );
        out.kautz_c_r[m] = (
            alpha * a.kautz_c_r[m].0 + om * b.kautz_c_r[m].0,
            alpha * a.kautz_c_r[m].1 + om * b.kautz_c_r[m].1,
        );
    }
    out.mask_to_tier(tier);
    out
}

fn pair_is_monotone(
    lo_l: &PerceptualCertificationSummary,
    hi_l: &PerceptualCertificationSummary,
    lo_r: Option<&PerceptualCertificationSummary>,
    hi_r: Option<&PerceptualCertificationSummary>,
) -> bool {
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut dl = [lo_l.clone(), hi_l.clone(), hi_l.clone(), hi_l.clone()];
    for k in 0..4 {
        dl[k].tier = tiers[k];
    }
    if !crate::metrics::certification::verify_monotone_4tier_summaries(&dl) {
        return false;
    }
    if let (Some(lr), Some(hr)) = (lo_r, hi_r) {
        let mut dr = [lr.clone(), hr.clone(), hr.clone(), hr.clone()];
        for k in 0..4 {
            dr[k].tier = tiers[k];
        }
        return crate::metrics::certification::verify_monotone_4tier_summaries(&dr);
    }
    true
}

fn repair_end_to_end_monotonicity<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    stem0: &V3TrainStem,
    caches: &mut Stem0CertCaches,
) -> Option<(
    [PerceptualCertificationSummary; 4],
    Option<[PerceptualCertificationSummary; 4]>,
)> {
    let (mut sums_l, mut sums_r) = caches.certify_matryoshka::<B>(model, stem0)?;
    if summaries_monotone(&sums_l, sums_r.as_ref()) {
        return Some((sums_l, sums_r));
    }
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    for tau in (0..3).rev() {
        if pair_is_monotone(
            &sums_l[tau],
            &sums_l[tau + 1],
            sums_r.as_ref().map(|r| &r[tau]),
            sums_r.as_ref().map(|r| &r[tau + 1]),
        ) {
            continue;
        }
        let mut fixed = false;
        if model.tier_os2x[tau] == model.tier_os2x[tau + 1] {
            let orig = model.readouts[tau].clone();
            let next_r = model.readouts[tau + 1].clone();
            for &alpha in &[0.85_f32, 0.55, 0.25, 0.0] {
                model.readouts[tau] = blend_tier_readouts(&orig, &next_r, alpha, tiers[tau]);
                let (cl, cr) = caches.certify_single_tier::<B>(
                    &model.extract_tier(tiers[tau]),
                    stem0,
                    None,
                    None,
                );
                if pair_is_monotone(
                    &cl,
                    &sums_l[tau + 1],
                    cr.as_ref(),
                    sums_r.as_ref().map(|r| &r[tau + 1]),
                ) {
                    sums_l[tau] = cl;
                    if let (Some(v), Some(sr)) = (cr, sums_r.as_mut()) {
                        sr[tau] = v;
                    }
                    fixed = true;
                    break;
                }
            }
            if !fixed {
                model.readouts[tau] = orig;
            }
        }
        if !fixed {
            let orig_taps = model.stage4[tau].tap_gains;
            for &scale in &[0.985_f32, 0.96, 0.92, 0.85, 0.72, 0.55] {
                model.stage4[tau].tap_gains = orig_taps.map(|g| g * scale);
                let (cl, cr) = caches.certify_single_tier::<B>(
                    &model.extract_tier(tiers[tau]),
                    stem0,
                    None,
                    None,
                );
                if pair_is_monotone(
                    &cl,
                    &sums_l[tau + 1],
                    cr.as_ref(),
                    sums_r.as_ref().map(|r| &r[tau + 1]),
                ) {
                    sums_l[tau] = cl;
                    if let (Some(v), Some(sr)) = (cr, sums_r.as_mut()) {
                        sr[tau] = v;
                    }
                    fixed = true;
                    break;
                }
            }
            if !fixed {
                model.stage4[tau].tap_gains = orig_taps;
            }
        }
    }
    caches.certify_matryoshka::<B>(model, stem0)
}

/// Identifies discrete early room reflections (`64..=960` lags) and jointly deconvolves +
/// refits Stage III over `2` alternating passes when end-to-end `Ultra` bESR improves cleanly.
pub(super) fn fit_room_lti_deconv<B: TrainerBackend>(
    baseline: &V3MatryoshkaModel,
    dataset: &V3TrainDataset,
    mono_out: bool,
    caches: &mut Stem0CertCaches,
    base_ultra_besr_db: f64,
) -> Option<V3MatryoshkaModel> {
    let stem0 = dataset.stems.first()?;
    let n = stem0.usable_len();
    let warmup = stem0.warmup.max(canonical_besr_warmup(n));
    let (wiener_tank, r2_lin) = wiener_echo::detect_wiener_room_2tap(stem0, warmup, mono_out);
    let pure_linear_flag = r2_lin >= 0.985;
    if !pure_linear_flag && r2_lin < 0.65 && base_ultra_besr_db > -12.0 {
        return None;
    }
    let cold_ultra = baseline.extract_tier(Tier::Ultra);
    let (u_l, u_r) = render_tier_stereo::<B>(&cold_ultra, &stem0.in_l[..n], &stem0.in_r[..n]);
    let pred_tank = estimate_tank_from_prediction(&u_l, &u_r, stem0, mono_out);

    let mut tank_cands = Vec::with_capacity(2);
    if let Some(t) = wiener_tank {
        tank_cands.push(t);
    }
    if let Some(t) = pred_tank {
        tank_cands.push(t);
    }
    if tank_cands.is_empty() {
        return None;
    }

    let mut lin_seed = V3MatryoshkaModel::dual_rate_default(mono_out);
    lin_seed.stage0 = baseline.stage0;
    let seeds: &[(&V3MatryoshkaModel, bool)] = if pure_linear_flag {
        &[(baseline, false), (&lin_seed, true)]
    } else {
        &[(baseline, false)]
    };

    let mut best_pass1: Option<(V3MatryoshkaModel, V3Stage4TankParams, f64, bool)> = None;
    for tank_try in tank_cands {
        let deconv1 = deconvolve_dataset_with_stage4::<B>(dataset, &tank_try, mono_out);
        for &(seed, is_lin) in seeds {
            let core_fit = refit_stage3_on_deconv::<B>(seed, &deconv1, mono_out, is_lin);
            let mut with_s4 = core_fit.clone();
            with_s4.stage4 = [tank_try; 4];
            let (sl, sr) = caches.certify_single_tier::<B>(
                &with_s4.extract_tier(Tier::Ultra),
                stem0,
                None,
                None,
            );
            let ultra_besr = sl
                .raw_besr_db
                .max(sr.as_ref().map_or(f64::NEG_INFINITY, |r| r.raw_besr_db));
            if ultra_besr.is_finite() && best_pass1.as_ref().is_none_or(|b| ultra_besr < b.2) {
                best_pass1 = Some((core_fit, tank_try, ultra_besr, is_lin));
            }
        }
    }
    let (core1, tank1, pass1_besr, used_pure_linear) = best_pass1?;
    if pass1_besr > -16.0 {
        return None;
    }

    // Pass 2: re-estimate room taps cleanly against the Stage III core prediction without echoes.
    let (u1_l, u1_r) = render_tier_stereo::<B>(
        &core1.extract_tier(Tier::Ultra),
        &stem0.in_l[..n],
        &stem0.in_r[..n],
    );
    let mut best_model = {
        let mut m = core1;
        m.stage4 = [tank1; 4];
        m
    };
    let mut best_besr = pass1_besr;
    if let Some(tank2) = estimate_tank_from_prediction(&u1_l, &u1_r, stem0, mono_out) {
        let deconv2 = deconvolve_dataset_with_stage4::<B>(dataset, &tank2, mono_out);
        let mut core2 =
            refit_stage3_on_deconv::<B>(&best_model, &deconv2, mono_out, used_pure_linear);
        core2.stage4 = [tank2; 4];
        let (sl2, sr2) =
            caches.certify_single_tier::<B>(&core2.extract_tier(Tier::Ultra), stem0, None, None);
        let besr2 = sl2
            .raw_besr_db
            .max(sr2.as_ref().map_or(f64::NEG_INFINITY, |r| r.raw_besr_db));
        if besr2.is_finite() && besr2 < best_besr {
            best_model = core2;
            best_besr = besr2;
        }
    }
    let _ = best_besr;

    let (final_l, final_r) = repair_end_to_end_monotonicity::<B>(&mut best_model, stem0, caches)?;
    let mono_ok = summaries_monotone(&final_l, final_r.as_ref());
    let final_ultra = worst_ultra_besr_db(&final_l, final_r.as_ref());
    (mono_ok
        && final_ultra <= -20.0
        && (final_ultra <= base_ultra_besr_db - MIN_ROOM_BESR_IMPROVEMENT_DB
            || (pure_linear_flag && final_ultra <= -35.0 && final_ultra < base_ultra_besr_db)))
        .then_some(best_model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backends::ScalarTrainerBackend;
    use std::f64::consts::TAU;

    #[test]
    fn test_room_lti_deconv_identifies_early_stereo_reflections() {
        let n = 8_192;
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let dry: Vec<f32> = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let white = ((seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
                let t = (i as f64) / 48_000.0;
                (0.25 * white + 0.20 * (TAU * 130.0 * t).sin()) as f32
            })
            .collect();
        let mut planted = V3MatryoshkaModel::dual_rate_default(false);
        for ro in &mut planted.readouts {
            ro.fir_l[0] = 0.72;
            ro.fir_l[1] = -0.18;
            ro.fir_r[0] = 0.72;
            ro.fir_r[1] = -0.18;
        }
        let tank_planted = build_feedforward_tank(
            ChannelEchoFit {
                d1: 89,
                d2: 233,
                g1: 0.24,
                g2: -0.14,
                r2: 1.0,
            },
            Some(ChannelEchoFit {
                d1: 144,
                d2: 377,
                g1: 0.24,
                g2: 0.14,
                r2: 1.0,
            }),
        );
        planted.stage4 = [tank_planted; 4];
        let (tgt_l, tgt_r) = render_tier_stereo::<ScalarTrainerBackend>(
            &planted.extract_tier(Tier::Ultra),
            &dry,
            &dry,
        );

        let ds = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: dry.clone(),
                in_r: dry,
                tgt_l,
                tgt_r,
                warmup: 512,
            }],
            48_000.0,
        );
        let mut base = planted.clone();
        base.stage4 = [V3Stage4TankParams::identity(); 4];
        let stem0 = &ds.stems[0];
        let mut caches = Stem0CertCaches::new(stem0, 48_000.0, false).unwrap();
        let (bl, br) = caches
            .certify_matryoshka::<ScalarTrainerBackend>(&base, stem0)
            .unwrap();
        let base_besr = worst_ultra_besr_db(&bl, br.as_ref());

        let recovered =
            fit_room_lti_deconv::<ScalarTrainerBackend>(&base, &ds, false, &mut caches, base_besr)
                .expect("Room LTI deconv recovers planted reflections");
        let s4 = recovered.stage4[3];
        assert!((80..=98).contains(&s4.delay_lengths[0]));
        assert!((134..=154).contains(&s4.delay_lengths[1]));
        assert!(s4.tap_gains[0] > 0.15);
    }
}
