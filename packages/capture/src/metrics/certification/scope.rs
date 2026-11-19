#![forbid(unsafe_code)]
//! Archetype metric-scope gating (`r[qual.eval.metric-scope]`) separating deterministic
//! sample-aligned `bESR` acceptance from free-running LFO / stochastic diffuse chains evaluated via
//! phase-invariant `MR-IIR`, `33`-band `LTAS`, and `SegESR`.

use super::{
    certify_4metric_signal_chain, evaluate_besr_from_band_energies, split_bezout_3band_energies,
};
use crate::metrics::companion::{LtasEvaluationReport, LtasTargetCache};
use crate::metrics::{ArchetypeFamily, TierQualityThresholds};
use dnam_engine::Tier;

/// Inputs to [`passes_for_family`] from `assemble_cert_summary`.
pub(super) struct ScopeEvalInputs<'a> {
    pub(super) family: ArchetypeFamily,
    pub(super) tier: Tier,
    pub(super) th: &'a TierQualityThresholds,
    pub(super) prescreen_rejected: bool,
    pub(super) besr_qual_passes: bool,
    pub(super) unseen_qual_passes: bool,
    pub(super) mr_iir: f64,
    pub(super) ltas_rep: &'a LtasEvaluationReport,
    pub(super) seg_valid: bool,
    pub(super) seg_active_count: usize,
    pub(super) seg_esr_db: f64,
}

#[inline]
const fn full_rig_mr_iir_ceiling(tier: Tier) -> f64 {
    match tier {
        Tier::Lite => 0.68,
        Tier::Full => 0.62,
        Tier::Pro => 0.56,
        Tier::Ultra => 0.52,
    }
}

/// Evaluates `(passes_target, passes_hard_ceiling)` under `r[qual.eval.metric-scope]`.
#[must_use]
pub(super) fn passes_for_family(inputs: &ScopeEvalInputs<'_>) -> (bool, bool) {
    let ltas = inputs.ltas_rep;
    let companion_base_valid = !inputs.prescreen_rejected
        && inputs.mr_iir.is_finite()
        && inputs.mr_iir >= 0.0
        && ltas.available
        && !ltas.excluded_band_guard_failed
        && inputs.seg_valid
        && inputs.seg_active_count > 0
        && !inputs.seg_esr_db.is_nan()
        && inputs.seg_esr_db < f64::INFINITY;

    match inputs.family {
        ArchetypeFamily::CleanPedalPreamp
        | ArchetypeFamily::CrunchHighGainAmp
        | ArchetypeFamily::FuzzOctaveChaos => {
            let ltas_ok = ltas.max_adj_deviation_db <= inputs.th.max_ltas_db;
            (
                companion_base_valid && ltas_ok && inputs.besr_qual_passes,
                companion_base_valid && ltas_ok && inputs.unseen_qual_passes,
            )
        }
        ArchetypeFamily::FullRigModDelayReverb => {
            let mr_max = full_rig_mr_iir_ceiling(inputs.tier);
            let mr_ceiling = mr_max * 1.5;
            let seg_ok = inputs.seg_esr_db <= 6.0;
            let pass_tgt = companion_base_valid
                && inputs.mr_iir <= mr_max
                && ltas.max_adj_deviation_db <= inputs.th.max_ltas_db
                && seg_ok;
            let pass_ceil = companion_base_valid
                && inputs.mr_iir <= mr_ceiling
                && ltas.max_adj_deviation_db <= inputs.th.max_ltas_db * 1.5
                && seg_ok;
            (pass_tgt, pass_ceil)
        }
    }
}

pub(super) fn is_sparse_holdout_spectrum(tgt_post: &[f64], sample_rate_hz: f64) -> bool {
    let Some(mut cache) = LtasTargetCache::new(tgt_post, sample_rate_hz, 20_000.0) else {
        return false;
    };
    let Some((p_y, _)) = cache.band_powers(tgt_post) else {
        return false;
    };
    let peak_p = p_y.iter().copied().fold(0.0_f64, f64::max);
    if !peak_p.is_finite() || peak_p <= 0.0 {
        return false;
    }
    let thresh = 5.0e-3 * peak_p;
    let excited_bands = p_y.iter().filter(|&&p| p > thresh).count();
    excited_bands > 0 && excited_bands < 28
}

pub(super) fn bezout_3band_balance_within_db(
    target: &[f64],
    model: &[f64],
    sample_rate_hz: f64,
    warmup_samples: usize,
    max_dev_db: f64,
) -> bool {
    let (sig_t, _, _) =
        split_bezout_3band_energies(target, target, None, sample_rate_hz, warmup_samples);
    let (sig_m, _, _) =
        split_bezout_3band_energies(model, model, None, sample_rate_hz, warmup_samples);
    let total_t = sig_t[0] + sig_t[1] + sig_t[2];
    let total_m = sig_m[0] + sig_m[1] + sig_m[2];
    if !total_t.is_finite()
        || total_t <= 0.0
        || !total_m.is_finite()
        || total_m <= 0.0
        || (10.0 * (total_m / total_t).log10()).abs() > 1.50
    {
        return false;
    }
    let reg = 0.02 * total_t;
    for b in 0..3 {
        if !sig_m[b].is_finite() || sig_m[b] < 0.0 {
            return false;
        }
        let diff_db = (10.0 * ((sig_m[b] + reg) / (sig_t[b] + reg)).log10()).abs();
        if !diff_db.is_finite() || diff_db > max_dev_db {
            return false;
        }
    }
    true
}

/// Evaluates unseen holdout certification `(reported_besr_db, pass)` for a given `family` and `tier`
/// (`ref[qual.eval.metric-scope]`), using sample-aligned `bESR <= unseen_control_besr_db` on
/// deterministic chains and phase-invariant `MR-IIR` + `33`-band `LTAS` (`* 1.5` ceiling) +
/// `SegESR` on [`ArchetypeFamily::FullRigModDelayReverb`].
#[must_use]
pub fn certify_holdout_for_family(
    family: ArchetypeFamily,
    tier: Tier,
    target: &[f64],
    model: &[f64],
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> (f64, bool) {
    match family {
        ArchetypeFamily::FullRigModDelayReverb => {
            let sum = certify_4metric_signal_chain(
                family,
                tier,
                target,
                model,
                None,
                sample_rate_hz,
                warmup_samples,
            );
            if sum.passes_hard_ceiling {
                return (sum.besr_db, true);
            }
            if target.len() == model.len()
                && warmup_samples < target.len()
                && is_sparse_holdout_spectrum(&target[warmup_samples..], sample_rate_hz)
            {
                let mr_ceiling = full_rig_mr_iir_ceiling(tier) * 1.5;
                let env_ok = sum.mr_iir.is_finite()
                    && sum.mr_iir >= 0.0
                    && sum.mr_iir <= mr_ceiling
                    && sum.seg_esr_db.is_finite()
                    && sum.seg_esr_db <= 6.0;
                let bal_ok = env_ok
                    && bezout_3band_balance_within_db(
                        target,
                        model,
                        sample_rate_hz,
                        warmup_samples,
                        1.75,
                    );
                return (sum.besr_db, bal_ok);
            }
            (sum.besr_db, false)
        }
        ArchetypeFamily::CleanPedalPreamp
        | ArchetypeFamily::CrunchHighGainAmp
        | ArchetypeFamily::FuzzOctaveChaos => {
            let (sig_e, err_e, nse_e) =
                split_bezout_3band_energies(target, model, None, sample_rate_hz, warmup_samples);
            let rep = evaluate_besr_from_band_energies(sig_e, err_e, nse_e);
            let th = TierQualityThresholds::for_tier(tier);
            let pass =
                rep.valid && !rep.besr_db.is_nan() && rep.besr_db <= th.unseen_control_besr_db;
            (rep.besr_db, pass)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_ltas_report(max_adj_db: f64) -> LtasEvaluationReport {
        LtasEvaluationReport {
            available: true,
            excluded_band_guard_failed: false,
            max_adj_deviation_db: max_adj_db,
            max_unadjusted_deviation_db: max_adj_db,
            excluded_bands: [false; 33],
            residual_band_power: [0.0; 33],
            active_band_count: 33,
        }
    }

    #[test]
    fn test_scope_branches_full_rig_from_deterministic_and_fails_closed() {
        let th = TierQualityThresholds::for_tier(Tier::Pro);
        let ltas_ok = make_ltas_report(0.10);
        let ltas_hold = make_ltas_report(0.20); // <= 0.15 * 1.5 = 0.225

        // FullRigModDelayReverb passes on MR-IIR/LTAS/SegESR even if sample-aligned bESR fails.
        let (pt, pc) = passes_for_family(&ScopeEvalInputs {
            family: ArchetypeFamily::FullRigModDelayReverb,
            tier: Tier::Pro,
            th: &th,
            prescreen_rejected: false,
            besr_qual_passes: false,
            unseen_qual_passes: false,
            mr_iir: 0.12,
            ltas_rep: &ltas_ok,
            seg_valid: true,
            seg_active_count: 10,
            seg_esr_db: -2.0,
        });
        assert!(pt && pc);

        // Relaxed * 1.5 holdout ceiling allows minor unseen LTAS drift only on ceiling, not target.
        let (pt_h, pc_h) = passes_for_family(&ScopeEvalInputs {
            family: ArchetypeFamily::FullRigModDelayReverb,
            tier: Tier::Pro,
            th: &th,
            prescreen_rejected: false,
            besr_qual_passes: false,
            unseen_qual_passes: false,
            mr_iir: 0.65, // > 0.56 target, <= 0.84 ceiling
            ltas_rep: &ltas_hold,
            seg_valid: true,
            seg_active_count: 10,
            seg_esr_db: 2.0,
        });
        assert!(!pt_h && pc_h);

        // Deterministic family still requires sample-aligned besr_qual/unseen_qual.
        let (det_t, det_c) = passes_for_family(&ScopeEvalInputs {
            family: ArchetypeFamily::CleanPedalPreamp,
            tier: Tier::Pro,
            th: &th,
            prescreen_rejected: false,
            besr_qual_passes: false,
            unseen_qual_passes: true,
            mr_iir: 2.5,
            ltas_rep: &ltas_ok,
            seg_valid: true,
            seg_active_count: 10,
            seg_esr_db: -40.0,
        });
        assert!(!det_t && det_c);

        // Fail-closed guards reject prescreen rejection, excluded guard failure, and runaway SegESR.
        let mut ltas_bad = make_ltas_report(0.05);
        ltas_bad.excluded_band_guard_failed = true;
        assert_eq!(
            passes_for_family(&ScopeEvalInputs {
                family: ArchetypeFamily::FullRigModDelayReverb,
                tier: Tier::Lite,
                th: &TierQualityThresholds::for_tier(Tier::Lite),
                prescreen_rejected: false,
                besr_qual_passes: true,
                unseen_qual_passes: true,
                mr_iir: 0.01,
                ltas_rep: &ltas_bad,
                seg_valid: true,
                seg_active_count: 10,
                seg_esr_db: 0.0,
            }),
            (false, false)
        );
        assert_eq!(
            passes_for_family(&ScopeEvalInputs {
                family: ArchetypeFamily::FullRigModDelayReverb,
                tier: Tier::Lite,
                th: &TierQualityThresholds::for_tier(Tier::Lite),
                prescreen_rejected: false,
                besr_qual_passes: true,
                unseen_qual_passes: true,
                mr_iir: 0.01,
                ltas_rep: &ltas_ok,
                seg_valid: true,
                seg_active_count: 10,
                seg_esr_db: 7.5,
            }),
            (false, false)
        );
    }

    #[test]
    fn test_certify_holdout_for_family_handles_phase_shifted_full_rig_and_deterministic() {
        let sr = 48_000.0;
        let n = 20_000usize;
        let centers = crate::metrics::canonical_ltas_33_centers_hz();
        let amp = 0.25 / (centers.len() as f64).sqrt();
        let mut tgt = vec![0.0_f64; n];
        let mut shifted = vec![0.0_f64; n];
        for (k, &fc) in centers.iter().enumerate() {
            let phase0 = 0.37 * (k as f64);
            for i in 0..n {
                let arg = 2.0 * PI * fc * (i as f64) / sr + phase0;
                tgt[i] += amp * arg.sin();
                shifted[i] += amp * (arg + 0.75).sin();
            }
        }
        let (_, pass_det_shift) = certify_holdout_for_family(
            ArchetypeFamily::CleanPedalPreamp,
            Tier::Full,
            &tgt,
            &shifted,
            sr,
            256,
        );
        assert!(!pass_det_shift);

        let (_, pass_rig_shift) = certify_holdout_for_family(
            ArchetypeFamily::FullRigModDelayReverb,
            Tier::Full,
            &tgt,
            &shifted,
            sr,
            256,
        );
        assert!(pass_rig_shift);

        let exact: Vec<f64> = tgt.iter().map(|&v| v * 0.999).collect();
        let (besr_exact, pass_det_exact) = certify_holdout_for_family(
            ArchetypeFamily::CrunchHighGainAmp,
            Tier::Ultra,
            &tgt,
            &exact,
            sr,
            256,
        );
        assert!(pass_det_exact && besr_exact < -50.0);

        // Sparse 6-tone musical holdout passes under FullRigModDelayReverb when envelope & 3-band balance match,
        // and fails closed on gain/EQ mismatch.
        let chord_freqs = [110.0, 164.81, 293.66, 587.33, 1_480.0, 3_720.0];
        let mut chord_tgt = vec![0.0_f64; n];
        let mut chord_mdl_ok = vec![0.0_f64; n];
        let mut chord_mdl_bad = vec![0.0_f64; n];
        for i in 0..n {
            let t = (i as f64) / sr;
            for (idx, &f) in chord_freqs.iter().enumerate() {
                let w = 0.12 / (idx as f64 + 1.0);
                let s0 = w * (2.0 * PI * f * t).sin();
                let s1 = w * (2.0 * PI * f * t + 0.6).sin();
                let s_bad = if f > 1_000.0 { 2.5 * s1 } else { s1 };
                chord_tgt[i] += s0;
                chord_mdl_ok[i] += s1;
                chord_mdl_bad[i] += s_bad;
            }
        }
        assert!(
            certify_holdout_for_family(
                ArchetypeFamily::FullRigModDelayReverb,
                Tier::Ultra,
                &chord_tgt,
                &chord_mdl_ok,
                sr,
                256,
            )
            .1
        );
        assert!(
            !certify_holdout_for_family(
                ArchetypeFamily::FullRigModDelayReverb,
                Tier::Ultra,
                &chord_tgt,
                &chord_mdl_bad,
                sr,
                256,
            )
            .1
        );
    }
}
