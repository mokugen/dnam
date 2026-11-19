#![forbid(unsafe_code)]
//! Inter-tier loudness parity trims (`LUFS` & broadband `RMS`) and monotone `bESR` polish helpers.

use dnam_engine::Tier;
use dnam_engine::stages::V3TierReadout;
use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, tier_kautz_banks};
use dnam_engine::stages::weights::ARENA0_TIER_GAIN_TRIM_OFFSET;

use crate::metrics::certification::{
    PerceptualCertificationSummary, TargetChannelCertCache, verify_cross_tier_loudness_parity,
    verify_monotone_4tier_summaries,
};
use crate::metrics::{
    TargetRepeatabilityProfile, evaluate_besr_from_band_energies, split_bezout_3band_energies,
};

use super::compute_integrated_lufs;

pub(super) const ALL_TIERS: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

pub(super) struct StageDCertCache {
    cache_l: TargetChannelCertCache,
    cache_r: Option<TargetChannelCertCache>,
    n: usize,
    warmup: usize,
    mono_out: bool,
}

impl StageDCertCache {
    pub(super) fn new(wet_lr: &[[f32; 2]], n: usize, warmup: usize, mono_out: bool) -> Self {
        let len = n.min(wet_lr.len());
        let tl: Vec<f64> = wet_lr[..len].iter().map(|v| f64::from(v[0])).collect();
        let cache_l = TargetChannelCertCache::new(&tl, 48_000.0, warmup);
        let cache_r = (!mono_out).then(|| {
            let tr: Vec<f64> = wet_lr[..len].iter().map(|v| f64::from(v[1])).collect();
            TargetChannelCertCache::new(&tr, 48_000.0, warmup)
        });
        Self {
            cache_l,
            cache_r,
            n: len,
            warmup,
            mono_out,
        }
    }

    pub(super) fn eval_tier(
        &mut self,
        out_lr: &[[f32; 2]],
        trim: f32,
        tier: Tier,
    ) -> (
        PerceptualCertificationSummary,
        Option<PerceptualCertificationSummary>,
    ) {
        let g = f64::from(trim);
        let len = self.n.min(out_lr.len());
        let det_profile = TargetRepeatabilityProfile::deterministic();
        let ml: Vec<f64> = out_lr[..len].iter().map(|v| g * f64::from(v[0])).collect();
        let sl = self
            .cache_l
            .certify_universal_with_prescreen(tier, &ml, &det_profile, None);
        let sr = self.cache_r.as_mut().map(|cr| {
            let mr: Vec<f64> = out_lr[..len].iter().map(|v| g * f64::from(v[1])).collect();
            cr.certify_universal_with_prescreen(tier, &mr, &det_profile, None)
        });
        (sl, sr)
    }

    pub(super) fn eval_all_tiers_and_monotone(
        &mut self,
        rendered: &[Vec<[f32; 2]>; 4],
        trims: [f32; 4],
        use_broadband_rms: bool,
    ) -> (
        [PerceptualCertificationSummary; 4],
        Option<[PerceptualCertificationSummary; 4]>,
        [f32; 4],
        f64,
        bool,
    ) {
        let mut sums_l = Vec::with_capacity(4);
        let mut sums_r = Vec::with_capacity(4);
        let mut esr = [0.0_f32; 4];
        let mut rms_levels = [1.0e-12_f64; 4];
        for t in 0..4 {
            let (sl, sr) = self.eval_tier(&rendered[t], trims[t], ALL_TIERS[t]);
            let worst = sl
                .raw_besr_db
                .max(sr.as_ref().map_or(f64::NEG_INFINITY, |r| r.raw_besr_db));
            esr[t] = if worst.is_finite() { worst as f32 } else { 0.0 };
            sums_l.push(sl);
            if let Some(r) = sr {
                sums_r.push(r);
            }
            if rendered[t].len() > self.warmup {
                let db = measure_tier_loudness_db(
                    &rendered[t][self.warmup..],
                    self.mono_out,
                    use_broadband_rms,
                );
                let eff_db = db + 20.0 * f64::from(trims[t]).max(1.0e-6).log10();
                rms_levels[t] = 10.0_f64.powf(eff_db / 20.0).max(1.0e-12);
            }
        }
        let arr_l: [PerceptualCertificationSummary; 4] = sums_l.try_into().unwrap();
        let arr_r: Option<[PerceptualCertificationSummary; 4]> =
            self.cache_r.is_some().then(|| sums_r.try_into().unwrap());
        let (lu_step, _) = verify_cross_tier_loudness_parity(rms_levels);
        let mono = verify_monotone_4tier_summaries(&arr_l)
            && arr_r.as_ref().is_none_or(verify_monotone_4tier_summaries);
        (arr_l, arr_r, esr, lu_step, mono)
    }
}

pub(super) fn pair_is_strict_monotone(
    lo_l: &PerceptualCertificationSummary,
    hi_l: &PerceptualCertificationSummary,
    lo_r: Option<&PerceptualCertificationSummary>,
    hi_r: Option<&PerceptualCertificationSummary>,
) -> bool {
    let check_pair = |a: &PerceptualCertificationSummary, b: &PerceptualCertificationSummary| {
        let mut l_dl = a.clone();
        l_dl.tier = Tier::Lite;
        let mut l_df = b.clone();
        l_df.tier = Tier::Full;
        let mut l_dp = b.clone();
        l_dp.tier = Tier::Pro;
        let mut l_du = b.clone();
        l_du.tier = Tier::Ultra;
        verify_monotone_4tier_summaries(&[l_dl, l_df, l_dp, l_du])
    };
    check_pair(lo_l, hi_l)
        && match (lo_r, hi_r) {
            (Some(lr), Some(hr)) => check_pair(lr, hr),
            _ => true,
        }
}

pub(super) fn read_tier_gain_trims(arena: &[u8]) -> [f32; 4] {
    std::array::from_fn(|t| {
        let off = ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4;
        if off + 4 > arena.len() {
            1.0
        } else {
            let v = f32::from_le_bytes(arena[off..off + 4].try_into().unwrap());
            if v.is_finite() && (0.0625..=16.0).contains(&v) {
                v
            } else {
                1.0
            }
        }
    })
}

pub(super) fn write_tier_gain_trims(arena: &mut [u8], trims: [f32; 4]) {
    for (t, &val) in trims.iter().enumerate() {
        let off = ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4;
        if off + 4 <= arena.len() {
            let clean = if val.is_finite() {
                val.clamp(0.0625, 16.0)
            } else {
                1.0
            };
            arena[off..off + 4].copy_from_slice(&clean.to_le_bytes());
        }
    }
}

pub(super) fn blend_cold_readouts(
    cur: &V3TierReadout,
    prev: &V3TierReadout,
    alpha: f32,
    tier: Tier,
) -> V3TierReadout {
    if alpha <= 0.0 {
        let mut out = prev.clone();
        out.kautz_b_gamma = cur.kautz_b_gamma;
        out.mask_to_tier(tier);
        return out;
    }
    let om = 1.0 - alpha;
    let mut out = V3TierReadout::zeros();
    for i in 0..8 {
        out.svf_direct_taps_l[i] =
            alpha * cur.svf_direct_taps_l[i] + om * prev.svf_direct_taps_l[i];
        out.svf_direct_taps_r[i] =
            alpha * cur.svf_direct_taps_r[i] + om * prev.svf_direct_taps_r[i];
    }
    for ch in 0..2 {
        out.pre_cab_u_gain[ch] = alpha * cur.pre_cab_u_gain[ch] + om * prev.pre_cab_u_gain[ch];
        out.pre_cab_hp_gain[ch] = alpha * cur.pre_cab_hp_gain[ch] + om * prev.pre_cab_hp_gain[ch];
        out.direct_s2_u_gain[ch] =
            alpha * cur.direct_s2_u_gain[ch] + om * prev.direct_s2_u_gain[ch];
        out.direct_s2_hp_gain[ch] =
            alpha * cur.direct_s2_hp_gain[ch] + om * prev.direct_s2_hp_gain[ch];
        for i in 0..MAX_TOTAL_GRU_LANES {
            out.pre_cab_w[ch][i] = alpha * cur.pre_cab_w[ch][i] + om * prev.pre_cab_w[ch][i];
            out.direct_s2_w[ch][i] = alpha * cur.direct_s2_w[ch][i] + om * prev.direct_s2_w[ch][i];
        }
    }
    for i in 0..64 {
        out.fir_l[i] = alpha * cur.fir_l[i] + om * prev.fir_l[i];
        out.fir_r[i] = alpha * cur.fir_r[i] + om * prev.fir_r[i];
    }
    out.kautz_b_gamma = cur.kautz_b_gamma;
    for m in 0..MAX_KAUTZ_POLES {
        out.kautz_c_l[m] = (
            alpha * cur.kautz_c_l[m].0 + om * prev.kautz_c_l[m].0,
            alpha * cur.kautz_c_l[m].1 + om * prev.kautz_c_l[m].1,
        );
        out.kautz_c_r[m] = (
            alpha * cur.kautz_c_r[m].0 + om * prev.kautz_c_r[m].0,
            alpha * cur.kautz_c_r[m].1 + om * prev.kautz_c_r[m].1,
        );
    }
    out.mask_to_tier(tier);
    out
}

pub(super) fn apply_allpass_and_kautz_phase_rotation(
    r: &mut V3TierReadout,
    tier: Tier,
    theta: f32,
) {
    let fir_len = (((tier as usize) + 1) * 16).min(64);
    let k_cnt = tier_kautz_banks(tier) * 4;
    let a = -(0.5 * theta).tan();
    for fir in [&mut r.fir_l, &mut r.fir_r] {
        let mut s = 0.0_f32;
        for tap in &mut fir[..fir_len] {
            let x = *tap;
            let y = a * x + s;
            s = x - a * y;
            *tap = y;
        }
    }
    let (s_th, c_th) = theta.sin_cos();
    for kc in [&mut r.kautz_c_l, &mut r.kautz_c_r] {
        for (c0, c1) in &mut kc[..k_cnt] {
            let n0 = *c0 * c_th - *c1 * s_th;
            let n1 = *c0 * s_th + *c1 * c_th;
            *c0 = n0;
            *c1 = n1;
        }
    }
    r.mask_to_tier(tier);
}

pub(super) fn eval_trimmed_tier_esr(
    out_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    trim: f32,
    mono_out: bool,
    warmup: usize,
) -> f32 {
    let n = out_lr.len().min(wet_lr.len());
    if n <= warmup {
        return -120.0;
    }
    let g = f64::from(trim);
    let mut tgt_l = Vec::with_capacity(n);
    let mut mdl_l = Vec::with_capacity(n);
    let mut sig_sq = 0.0_f64;
    let mut err_sq = 0.0_f64;
    for t in 0..n {
        let tl = f64::from(wet_lr[t][0]);
        let ml = g * f64::from(out_lr[t][0]);
        tgt_l.push(tl);
        mdl_l.push(ml);
        if t >= warmup {
            let dl = tl - ml;
            sig_sq += tl * tl;
            err_sq += dl * dl;
        }
    }
    if sig_sq <= 1.0e-18 && mono_out {
        return if err_sq <= 1.0e-18 { -120.0 } else { 0.0 };
    }
    let (sl, el, nl) = split_bezout_3band_energies(&tgt_l, &mdl_l, None, 48_000.0, warmup);
    let rep_l = evaluate_besr_from_band_energies(sl, el, nl);
    if !rep_l.valid || !rep_l.besr_db.is_finite() {
        return if sig_sq <= 1.0e-18 {
            if err_sq <= 1.0e-18 { -120.0 } else { 0.0 }
        } else {
            (10.0 * (err_sq / sig_sq).max(1.0e-16).log10()) as f32
        };
    }
    if mono_out {
        return rep_l.besr_db as f32;
    }
    let mut tgt_r = Vec::with_capacity(n);
    let mut mdl_r = Vec::with_capacity(n);
    for t in 0..n {
        tgt_r.push(f64::from(wet_lr[t][1]));
        mdl_r.push(g * f64::from(out_lr[t][1]));
    }
    let (sr, er, nr) = split_bezout_3band_energies(&tgt_r, &mdl_r, None, 48_000.0, warmup);
    let rep_r = evaluate_besr_from_band_energies(sr, er, nr);
    if rep_r.valid && rep_r.besr_db.is_finite() {
        rep_l.besr_db.max(rep_r.besr_db) as f32
    } else {
        rep_l.besr_db as f32
    }
}

pub(super) fn tier_broadband_rms_db(samples_lr: &[[f32; 2]], mono_out: bool) -> f64 {
    if samples_lr.is_empty() {
        return -120.0;
    }
    let sum_sq: f64 = samples_lr
        .iter()
        .map(|&[l, r]| {
            let l64 = f64::from(l);
            if mono_out {
                l64 * l64
            } else {
                let r64 = f64::from(r);
                0.5 * (l64 * l64 + r64 * r64)
            }
        })
        .sum();
    let mean_sq = sum_sq / (samples_lr.len() as f64);
    if mean_sq <= 1.0e-20 || !mean_sq.is_finite() {
        -120.0
    } else {
        10.0 * mean_sq.log10()
    }
}

pub(super) fn measure_tier_loudness_db(
    samples_post_warmup: &[[f32; 2]],
    mono_out: bool,
    use_broadband_rms: bool,
) -> f64 {
    if use_broadband_rms {
        tier_broadband_rms_db(samples_post_warmup, mono_out)
    } else if mono_out {
        let mono_lr: Vec<[f32; 2]> = samples_post_warmup.iter().map(|&[l, _]| [l, l]).collect();
        compute_integrated_lufs(&mono_lr, 48_000)
    } else {
        compute_integrated_lufs(samples_post_warmup, 48_000)
    }
}

pub(super) fn solve_exact_tier_trim(
    samples_post_warmup: &[[f32; 2]],
    target_db: f64,
    mono_out: bool,
    use_broadband_rms: bool,
) -> f32 {
    if !target_db.is_finite() || target_db <= -100.0 || samples_post_warmup.is_empty() {
        return 1.0;
    }
    let lvl = measure_tier_loudness_db(samples_post_warmup, mono_out, use_broadband_rms);
    if !lvl.is_finite() || lvl <= -100.0 {
        return 1.0;
    }
    let mut g = 10.0_f64.powf((target_db - lvl) / 20.0).clamp(0.25, 4.0);
    if !use_broadband_rms {
        let gf = g as f32;
        let scaled: Vec<[f32; 2]> = samples_post_warmup
            .iter()
            .map(|&[l, r]| [l * gf, if mono_out { l * gf } else { r * gf }])
            .collect();
        let lu1 = compute_integrated_lufs(&scaled, 48_000);
        if lu1.is_finite() && lu1 > -100.0 && (lu1 - target_db).abs() > 1.0e-4 {
            g = (g * 10.0_f64.powf((target_db - lu1) / 20.0)).clamp(0.25, 4.0);
        }
    }
    g as f32
}

fn chan_pair_has_monotone_cushion(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
) -> bool {
    let ltas_lim = if lo.ltas_max_db <= 1.0e-4 {
        lo.ltas_max_db
    } else {
        (lo.ltas_max_db - 0.003).max(lo.ltas_max_db * 0.96)
    };
    let mr_lim = if lo.mr_iir <= 1.0e-5 {
        lo.mr_iir
    } else {
        (lo.mr_iir - 4.0e-4).min(lo.mr_iir * 0.975)
    };
    let esr_cush = if hi.raw_besr_db <= -35.0 { 0.26 } else { 0.08 };
    let band_cush = |b: usize| {
        if hi.band_esr_db[b] <= -35.0 {
            0.26
        } else {
            0.08
        }
    };
    let seg_cush = if hi.seg_esr_db <= -42.0 { 0.18 } else { 0.08 };
    hi.raw_besr_db <= lo.raw_besr_db - esr_cush
        && (0..3).all(|b| hi.band_esr_db[b] <= lo.band_esr_db[b] - band_cush(b))
        && hi.mr_iir <= mr_lim
        && hi.ltas_max_db <= ltas_lim
        && hi.seg_esr_db <= lo.seg_esr_db - seg_cush
}

pub(super) fn pair_has_monotone_cushion(
    lo_l: &PerceptualCertificationSummary,
    hi_l: &PerceptualCertificationSummary,
    lo_r: Option<&PerceptualCertificationSummary>,
    hi_r: Option<&PerceptualCertificationSummary>,
) -> bool {
    pair_is_strict_monotone(lo_l, hi_l, lo_r, hi_r)
        && chan_pair_has_monotone_cushion(lo_l, hi_l)
        && match (lo_r, hi_r) {
            (Some(lr), Some(hr)) => chan_pair_has_monotone_cushion(lr, hr),
            _ => true,
        }
}

fn spread_within_cap(levels: &[f64; 4], trims: &[f32; 4], max_spread_lu: f64) -> bool {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for k in 0..4 {
        if !levels[k].is_finite() || levels[k] <= -100.0 {
            return true;
        }
        let eff = levels[k] + 20.0 * f64::from(trims[k]).max(1.0e-6).log10();
        lo = lo.min(eff);
        hi = hi.max(eff);
    }
    (hi - lo) <= max_spread_lu
}

fn level_spread_lu(levels: &[f64; 4], trims: &[f32; 4]) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for k in 0..4 {
        if !levels[k].is_finite() || levels[k] <= -100.0 {
            return 0.0;
        }
        let eff = levels[k] + 20.0 * f64::from(trims[k]).max(1.0e-6).log10();
        lo = lo.min(eff);
        hi = hi.max(eff);
    }
    (hi - lo).max(0.0)
}

pub(super) fn fine_tune_monotone_trims(
    rendered: &[Vec<[f32; 2]>; 4],
    wet_slice: &[[f32; 2]],
    cand_trims: &mut [f32; 4],
    cand_esr: &mut [f32; 4],
    mono_out: bool,
    warmup: usize,
) -> bool {
    let n = rendered
        .iter()
        .map(Vec::len)
        .min()
        .unwrap_or(0)
        .min(wet_slice.len());
    if n <= warmup + 64 {
        return (0..3).all(|t| cand_esr[t + 1] <= cand_esr[t]);
    }
    let orig_trims = *cand_trims;
    let raw_lufs: [f64; 4] =
        std::array::from_fn(|t| measure_tier_loudness_db(&rendered[t][warmup..], mono_out, false));
    let raw_rms: [f64; 4] =
        std::array::from_fn(|t| measure_tier_loudness_db(&rendered[t][warmup..], mono_out, true));
    let lufs_cap = level_spread_lu(&raw_lufs, &orig_trims).max(0.045);
    let rms_cap = level_spread_lu(&raw_rms, &orig_trims).max(0.045);

    let mut cert = StageDCertCache::new(wet_slice, n, warmup, mono_out);
    let (mut sums_l, mut sums_r, esr_init, _, mono_init) =
        cert.eval_all_tiers_and_monotone(rendered, *cand_trims, true);
    *cand_esr = esr_init;
    let cushion_init = mono_init
        && (0..3).all(|t| {
            pair_has_monotone_cushion(
                &sums_l[t],
                &sums_l[t + 1],
                sums_r.as_ref().map(|r| &r[t]),
                sums_r.as_ref().map(|r| &r[t + 1]),
            )
        });
    if cushion_init {
        return true;
    }
    // Search +/- 0.040 LU while keeping adjacent steps and total LUFS/RMS spread <= 0.045 LU.
    for t in (0..3).rev() {
        let hi_l = sums_l[t + 1].clone();
        let hi_r = sums_r.as_ref().map(|r| r[t + 1].clone());
        let lo_l = &sums_l[t];
        let lo_r = sums_r.as_ref().map(|r| &r[t]);
        if pair_has_monotone_cushion(lo_l, &hi_l, lo_r, hi_r.as_ref()) {
            continue;
        }
        let g_base = f64::from(orig_trims[t]);
        let init_pass_l = sums_l[t].passes_target;
        let init_pass_r = sums_r.as_ref().is_none_or(|r| r[t].passes_target);
        let next_shift_lu =
            20.0 * (f64::from(cand_trims[t + 1]) / f64::from(orig_trims[t + 1])).log10();
        let mut best_cushion: Option<(
            f64,
            f32,
            f32,
            PerceptualCertificationSummary,
            Option<PerceptualCertificationSummary>,
        )> = None;
        let mut best_strict: Option<(
            f64,
            f32,
            f32,
            PerceptualCertificationSummary,
            Option<PerceptualCertificationSummary>,
        )> = None;
        let mut trial_trims = *cand_trims;
        for step in -20_i32..=20 {
            let delta_lu = f64::from(step) * 0.002;
            if (delta_lu - next_shift_lu).abs() > 0.045 {
                continue;
            }
            let g_try = (g_base * 10.0_f64.powf(delta_lu / 20.0)).clamp(0.25, 4.0) as f32;
            trial_trims[t] = g_try;
            if !spread_within_cap(&raw_lufs, &trial_trims, lufs_cap)
                || !spread_within_cap(&raw_rms, &trial_trims, rms_cap)
            {
                continue;
            }
            let (sl, sr) = cert.eval_tier(&rendered[t], g_try, ALL_TIERS[t]);
            let drop_l = init_pass_l && !sl.passes_target;
            let drop_r = init_pass_r && !sr.as_ref().is_none_or(|r| r.passes_target);
            if drop_l || drop_r {
                continue;
            }
            let worst = sl
                .raw_besr_db
                .max(sr.as_ref().map_or(f64::NEG_INFINITY, |r| r.raw_besr_db));
            if pair_has_monotone_cushion(&sl, &hi_l, sr.as_ref(), hi_r.as_ref())
                && best_cushion.as_ref().is_none_or(|b| delta_lu.abs() < b.0)
            {
                best_cushion = Some((delta_lu.abs(), g_try, worst as f32, sl, sr));
            } else if pair_is_strict_monotone(&sl, &hi_l, sr.as_ref(), hi_r.as_ref())
                && best_strict.as_ref().is_none_or(|b| delta_lu.abs() < b.0)
            {
                best_strict = Some((delta_lu.abs(), g_try, worst as f32, sl, sr));
            }
        }
        if let Some((_, g_best, e_best, sl_best, sr_best)) = best_cushion.or(best_strict) {
            cand_trims[t] = g_best;
            cand_esr[t] = e_best;
            sums_l[t] = sl_best;
            if let (Some(r_vec), Some(r_val)) = (sums_r.as_mut(), sr_best) {
                r_vec[t] = r_val;
            }
        }
    }
    verify_monotone_4tier_summaries(&sums_l)
        && sums_r.as_ref().is_none_or(verify_monotone_4tier_summaries)
}

pub(super) fn calibrate_trims_matching_ultra_lufs(
    rendered: &[Vec<[f32; 2]>; 4],
    wet_slice: &[[f32; 2]],
    mono_out: bool,
    warmup: usize,
    use_broadband_rms: bool,
) -> Option<([f32; 4], [f32; 4], f64, bool)> {
    let ultra_db = measure_tier_loudness_db(&rendered[3][warmup..], mono_out, use_broadband_rms);
    if !ultra_db.is_finite() || ultra_db <= -100.0 {
        return None;
    }

    let mut cand_trims = [1.0_f32; 4];
    for t in 0..3 {
        cand_trims[t] = solve_exact_tier_trim(
            &rendered[t][warmup..],
            ultra_db,
            mono_out,
            use_broadband_rms,
        );
    }
    let mut cand_esr: [f32; 4] = ALL_TIERS.map(|t| {
        let ti = t as usize;
        eval_trimmed_tier_esr(&rendered[ti], wet_slice, cand_trims[ti], mono_out, warmup)
    });
    let monotone = fine_tune_monotone_trims(
        rendered,
        wet_slice,
        &mut cand_trims,
        &mut cand_esr,
        mono_out,
        warmup,
    );
    Some((cand_trims, cand_esr, ultra_db, monotone))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn test_trims_exact_lufs_and_allpass_unitary_helpers() {
        let n = 4_096_usize;
        let base_lr: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let s = (0.30 * (TAU * 330.0 * t).sin() + 0.12 * (TAU * 2_400.0 * t).cos()) as f32;
                [s, s * 0.9]
            })
            .collect();
        let louder_lr: Vec<[f32; 2]> = base_lr.iter().map(|&[l, r]| [l * 1.15, r * 1.15]).collect();
        let quiet_lr: Vec<[f32; 2]> = base_lr.iter().map(|&[l, r]| [l * 0.85, r * 0.85]).collect();

        let target_lufs = measure_tier_loudness_db(&base_lr[512..], false, false);
        let trim_lufs = solve_exact_tier_trim(&louder_lr[512..], target_lufs, false, false);
        assert!(
            (trim_lufs - (1.0 / 1.15)).abs() < 1.0e-3,
            "trim={trim_lufs}"
        );
        let target_rms = measure_tier_loudness_db(&base_lr[512..], true, true);
        let trim_rms = solve_exact_tier_trim(&quiet_lr[512..], target_rms, true, true);
        assert!((trim_rms - (1.0 / 0.85)).abs() < 1.0e-3, "trim={trim_rms}");
        assert_eq!(solve_exact_tier_trim(&[], -20.0, false, false), 1.0);
        assert_eq!(
            solve_exact_tier_trim(&base_lr[512..], -110.0, false, false),
            1.0
        );

        let rendered = [
            louder_lr.clone(),
            quiet_lr.clone(),
            louder_lr,
            base_lr.clone(),
        ];
        let (trims, esr, udb, _) =
            calibrate_trims_matching_ultra_lufs(&rendered, &base_lr, false, 512, false)
                .expect("calibration should return Some");
        assert!(udb > -40.0);
        assert!(esr.iter().all(|v| v.is_finite()));
        assert!((trims[3] - 1.0).abs() < 1.0e-6);

        let mut arena = vec![0u8; ARENA0_TIER_GAIN_TRIM_OFFSET + 32];
        write_tier_gain_trims(&mut arena, [0.9, f32::NAN, 1.1, 1.0]);
        let read = read_tier_gain_trims(&arena);
        assert!((read[0] - 0.9).abs() < 1.0e-6);
        assert!((read[1] - 1.0).abs() < 1.0e-6);
        assert!((read[2] - 1.1).abs() < 1.0e-6);

        let mut ro = V3TierReadout::zeros();
        ro.fir_l[0] = 0.6;
        ro.fir_l[1] = -0.3;
        ro.kautz_c_l[0] = (0.4, 0.2);
        let e0: f32 = ro.kautz_c_l[0].0.powi(2) + ro.kautz_c_l[0].1.powi(2);
        apply_allpass_and_kautz_phase_rotation(&mut ro, Tier::Full, 0.18);
        let e1: f32 = ro.kautz_c_l[0].0.powi(2) + ro.kautz_c_l[0].1.powi(2);
        assert!((e0 - e1).abs() < 1.0e-6);
    }
}
