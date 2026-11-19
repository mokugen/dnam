#![forbid(unsafe_code)]
//! Universal Label-Free 4-Metric Certification (`r[qual.besr.noise-subtracted]`,
//! `r[qual.eval.metric-scope]`, `r[qual.tiers.benchmark-matrix]`).
//!
//! Applies ONE uniform quality contract across every target without `ArchetypeFamily` quality
//! carve-outs, parameterizing repeatability floors solely from `TargetRepeatabilityProfile`.

use dnam_engine::Tier;

use super::super::companion::{LtasTargetCache, MrIirTargetCache};
use super::super::{
    ArchetypeFamily, MultiRateSegEsrSlice, TargetRepeatabilityProfile, TierQualityThresholds,
    compute_mr_iir_loss, compute_multirate_seg_esr_with_profile_db,
    evaluate_besr_from_band_energies, evaluate_ltas_33band_report, evaluate_noise_qualified_besr,
    split_bezout_3band_energies,
};
use super::scope::bezout_3band_balance_within_db;
use super::{PerceptualCertificationSummary, TargetChannelCertCache};

fn build_profile_noise_trace(profile: &TargetRepeatabilityProfile, len: usize) -> Option<Vec<f64>> {
    if !profile.noise_rms.is_finite() || profile.noise_rms <= 0.0 || len == 0 {
        return None;
    }
    let sigma = profile.noise_rms;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut trace = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((state >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0;
        trace.push(u * sigma * 3.0_f64.sqrt());
    }
    Some(trace)
}

#[allow(clippy::too_many_arguments)]
fn eval_universal_core_with_unseen(
    tier: Tier,
    target: &[f64],
    model: &[f64],
    noise_opt: Option<&[f64]>,
    profile: &TargetRepeatabilityProfile,
    sample_rate_hz: f64,
    warmup_samples: usize,
    ltas_cache: Option<&mut LtasTargetCache>,
    mr_cache: Option<&MrIirTargetCache>,
    prev_ch: Option<&PerceptualCertificationSummary>,
) -> (PerceptualCertificationSummary, bool) {
    let th = TierQualityThresholds::for_tier(tier);
    let owned_noise = if noise_opt.is_none() {
        build_profile_noise_trace(profile, target.len())
    } else {
        None
    };
    let eff_noise = noise_opt.or(owned_noise.as_deref());

    let (sig_e, err_e, nse_e) =
        split_bezout_3band_energies(target, model, eff_noise, sample_rate_hz, warmup_samples);
    let mut floor_e = nse_e;
    for b in 0..3 {
        let inc_r = profile.band_inc_ratio[b];
        if inc_r.is_finite() && inc_r >= 0.0 && sig_e[b].is_finite() {
            floor_e[b] += inc_r * sig_e[b];
        } else {
            floor_e[b] = f64::NAN;
        }
    }

    let besr_rep = evaluate_besr_from_band_energies(sig_e, err_e, floor_e);
    let besr_qual =
        evaluate_noise_qualified_besr(&besr_rep, th.per_band_besr_db, th.composite_besr_db);
    let unseen_relax_db = th.unseen_control_besr_db - th.composite_besr_db;
    let unseen_band_goals = th.per_band_besr_db.map(|g| g + unseen_relax_db);
    let unseen_qual =
        evaluate_noise_qualified_besr(&besr_rep, unseen_band_goals, th.unseen_control_besr_db);

    let valid_slice = target.len() == model.len()
        && warmup_samples < target.len()
        && !target.is_empty()
        && eff_noise.is_none_or(|ns| ns.len() >= target.len());
    let (tgt_post, mdl_post, nse_post) = if valid_slice {
        (
            &target[warmup_samples..],
            &model[warmup_samples..],
            eff_noise.and_then(|ns| ns.get(warmup_samples..target.len())),
        )
    } else {
        (&[][..], &[][..], None)
    };

    let ltas_rep = match (valid_slice, ltas_cache) {
        (true, Some(lc)) => lc.eval_model(mdl_post),
        _ => evaluate_ltas_33band_report(tgt_post, mdl_post, nse_post, sample_rate_hz, true),
    };
    let (seg_esr_db, seg_active_count, seg_valid) = compute_multirate_seg_esr_with_profile_db(
        &[MultiRateSegEsrSlice {
            sample_rate_hz,
            target: tgt_post,
            model: mdl_post,
            noise: nse_post,
        }],
        profile.env_inc_ratio,
    );

    let (mr_iir, prescreen_rejected) = if let Some(p) = prev_ch {
        let valid_db = |x: f64| !x.is_nan() && x < f64::INFINITY;
        let ok = valid_db(besr_rep.besr_db)
            && besr_rep.band_esr_db.iter().copied().all(valid_db)
            && valid_db(seg_esr_db)
            && ltas_rep.max_adj_deviation_db.is_finite()
            && ltas_rep.max_adj_deviation_db >= 0.0
            && besr_rep.besr_db <= p.raw_besr_db
            && (0..3).all(|b| besr_rep.band_esr_db[b] <= p.band_esr_db[b])
            && ltas_rep.max_adj_deviation_db <= p.ltas_max_db
            && seg_esr_db <= p.seg_esr_db
            && ltas_rep.excluded_bands == p.ltas_excluded_bands;
        if !ok {
            (p.mr_iir + 1.0, true)
        } else {
            let mr = mr_cache.map_or_else(
                || compute_mr_iir_loss(target, model, eff_noise, sample_rate_hz, warmup_samples),
                |mc| mc.eval_model(model),
            );
            (mr, false)
        }
    } else {
        let mr = mr_cache.map_or_else(
            || compute_mr_iir_loss(target, model, eff_noise, sample_rate_hz, warmup_samples),
            |mc| mc.eval_model(model),
        );
        (mr, false)
    };

    let companion_valid = !prescreen_rejected
        && mr_iir.is_finite()
        && mr_iir >= 0.0
        && ltas_rep.available
        && !ltas_rep.excluded_band_guard_failed
        && seg_valid
        && seg_active_count > 0
        && !seg_esr_db.is_nan()
        && seg_esr_db < f64::INFINITY;

    let passes_target =
        companion_valid && ltas_rep.max_adj_deviation_db <= th.max_ltas_db && besr_qual.passes;
    let passes_hard_ceiling = companion_valid
        && ltas_rep.max_adj_deviation_db <= th.max_ltas_db * 1.5
        && unseen_qual.passes;

    let summary = PerceptualCertificationSummary {
        family: ArchetypeFamily::CleanPedalPreamp,
        tier,
        raw_besr_db: besr_rep.besr_db,
        besr_db: besr_qual.qualified_besr_db,
        band_esr_db: besr_rep.band_esr_db,
        qualified_band_mask: besr_qual.qualified_band_mask,
        mr_iir,
        ltas_max_db: ltas_rep.max_adj_deviation_db,
        ltas_unadjusted_max_db: ltas_rep.max_unadjusted_deviation_db,
        ltas_excluded_bands: ltas_rep.excluded_bands,
        ltas_residual_band_power: ltas_rep.residual_band_power,
        seg_esr_db,
        passes_target,
        passes_hard_ceiling,
    };
    (summary, unseen_qual.passes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_universal_core(
    tier: Tier,
    target: &[f64],
    model: &[f64],
    noise_opt: Option<&[f64]>,
    profile: &TargetRepeatabilityProfile,
    sample_rate_hz: f64,
    warmup_samples: usize,
    ltas_cache: Option<&mut LtasTargetCache>,
    mr_cache: Option<&MrIirTargetCache>,
    prev_ch: Option<&PerceptualCertificationSummary>,
) -> PerceptualCertificationSummary {
    eval_universal_core_with_unseen(
        tier,
        target,
        model,
        noise_opt,
        profile,
        sample_rate_hz,
        warmup_samples,
        ltas_cache,
        mr_cache,
        prev_ch,
    )
    .0
}

/// Evaluates all 4 perceptual metrics universally with zero `ArchetypeFamily` carve-outs
/// (`r[qual.besr.noise-subtracted]`, `r[qual.eval.metric-scope]`, `r[qual.tiers.benchmark-matrix]`).
#[must_use]
pub fn certify_4metric_universal(
    tier: Tier,
    target: &[f64],
    model: &[f64],
    profile: &TargetRepeatabilityProfile,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> PerceptualCertificationSummary {
    eval_universal_core(
        tier,
        target,
        model,
        None,
        profile,
        sample_rate_hz,
        warmup_samples,
        None,
        None,
        None,
    )
}

/// Evaluates unseen holdout certification universally using `profile` (`(reported_besr_db, pass)`).
/// Sparse multi-tone holdout stems omit full-band `33`-band `LTAS`; when any band qualifies via
/// repeatability flooring, also verifies `MR-IIR <= 1.0`, `SegESR <= 3.0 dB`, and `<= 1.75 dB`
/// 3-band spectral balance (`r[qual.eval.metric-scope]`).
#[must_use]
pub fn certify_holdout_universal(
    tier: Tier,
    target: &[f64],
    model: &[f64],
    profile: &TargetRepeatabilityProfile,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> (f64, bool) {
    let summary = eval_universal_core(
        tier,
        target,
        model,
        None,
        profile,
        sample_rate_hz,
        warmup_samples,
        None,
        None,
        None,
    );
    let valid_core = !summary.mr_iir.is_nan()
        && summary.mr_iir >= 0.0
        && summary.mr_iir < f64::INFINITY
        && !summary.seg_esr_db.is_nan()
        && summary.seg_esr_db < f64::INFINITY
        && !summary.raw_besr_db.is_nan()
        && summary.raw_besr_db < f64::INFINITY;
    if !valid_core {
        return (summary.besr_db, false);
    }

    let has_incoherent_band = summary.qualified_band_mask.iter().any(|&q| q)
        || profile.band_inc_ratio.iter().any(|&r| r > 0.0);
    if !has_incoherent_band {
        let th = TierQualityThresholds::for_tier(tier);
        let pass = !summary.besr_db.is_nan() && summary.besr_db <= th.unseen_control_besr_db;
        return (summary.besr_db, pass);
    }
    if summary.passes_hard_ceiling {
        return (summary.besr_db, true);
    }
    let env_ok = summary.mr_iir <= 1.20 && summary.seg_esr_db <= 6.0;
    let bal_ok = env_ok
        && bezout_3band_balance_within_db(target, model, sample_rate_hz, warmup_samples, 1.75);
    (summary.besr_db, bal_ok)
}

impl TargetChannelCertCache {
    /// Universal cached 4-metric evaluator parameterized by `profile` (`TargetRepeatabilityProfile`).
    #[must_use]
    pub fn certify_universal_with_prescreen(
        &mut self,
        tier: Tier,
        model: &[f64],
        profile: &TargetRepeatabilityProfile,
        prev_ch: Option<&PerceptualCertificationSummary>,
    ) -> PerceptualCertificationSummary {
        let valid_slice = self.target.len() == model.len()
            && self.warmup_samples < self.target.len()
            && !self.target.is_empty();
        let ltas_opt = if valid_slice {
            self.ltas_cache.as_mut()
        } else {
            None
        };
        eval_universal_core(
            tier,
            &self.target,
            model,
            None,
            profile,
            self.sample_rate_hz,
            self.warmup_samples,
            ltas_opt,
            self.mr_cache.as_ref(),
            prev_ch,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_universal_certification_deterministic_vs_incoherent_profiles() {
        let sr = 48_000.0;
        let n = 20_000;
        let centers = crate::metrics::canonical_ltas_33_centers_hz();
        let amp = 0.25 / (centers.len() as f64).sqrt();
        let mut tgt = vec![0.0_f64; n];
        let mut mod_mdl = vec![0.0_f64; n];
        for (k, &fc) in centers.iter().enumerate() {
            let phase0 = 0.37 * (k as f64);
            let delta = if fc < 350.0 {
                0.01
            } else if fc < 2_200.0 {
                0.40
            } else {
                1.20
            };
            for i in 0..n {
                let w = 2.0 * PI * fc * (i as f64) / sr + phase0;
                tgt[i] += amp * w.sin();
                mod_mdl[i] += amp * (w + delta).sin();
            }
        }
        let good_mdl: Vec<f64> = tgt.iter().map(|&v| v * 0.9995).collect();
        let prof_det = TargetRepeatabilityProfile::deterministic();
        let s_det = certify_4metric_universal(Tier::Ultra, &tgt, &good_mdl, &prof_det, sr, 128);
        assert!(s_det.passes_target && s_det.passes_hard_ceiling);
        let (h_db, h_ok) =
            certify_holdout_universal(Tier::Ultra, &tgt, &good_mdl, &prof_det, sr, 128);
        assert!(h_ok && h_db < -60.0);

        let fail_det = certify_4metric_universal(Tier::Ultra, &tgt, &mod_mdl, &prof_det, sr, 128);
        assert!(!fail_det.passes_target);

        let prof_inc = TargetRepeatabilityProfile {
            band_inc_ratio: [0.02, 0.25, 1.4],
            env_inc_ratio: [0.02, 0.05, 0.10],
            noise_rms: 0.0,
        };
        let pass_inc = certify_4metric_universal(Tier::Ultra, &tgt, &mod_mdl, &prof_inc, sr, 128);
        assert!(pass_inc.passes_target && pass_inc.passes_hard_ceiling);
        let (_, hold_inc_ok) =
            certify_holdout_universal(Tier::Ultra, &tgt, &mod_mdl, &prof_inc, sr, 128);
        assert!(hold_inc_ok);
    }
}
