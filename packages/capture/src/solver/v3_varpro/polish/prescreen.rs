#![forbid(unsafe_code)]
//! Exact cold-model perceptual prescreen and `pre_cab` sharing predicate for `v3` polish.

use dnam_engine::Tier;
use dnam_engine::stages::v3_core::{V3CoreInstance, V3MatryoshkaModel};

use crate::metrics::certification::PerceptualCertificationSummary;
use crate::metrics::{
    ArchetypeFamily, MultiRateSegEsrSlice, TierQualityThresholds, compute_mr_iir_loss,
    compute_multirate_seg_esr_db, evaluate_besr_from_band_energies, evaluate_ltas_33band_report,
    evaluate_noise_qualified_besr, split_bezout_3band_energies,
};
use crate::solver::backends::TrainerBackend;

use super::super::model_f64::{V3TierReadoutF64, V3TrainStem};

#[must_use]
pub(super) fn shares_pre_cab(a: &V3TierReadoutF64, b: &V3TierReadoutF64, active_ch: usize) -> bool {
    (0..active_ch).all(|ch| {
        (a.pre_cab_u_gain[ch] - b.pre_cab_u_gain[ch]).abs() <= 1.0e-9
            && (a.pre_cab_hp_gain[ch] - b.pre_cab_hp_gain[ch]).abs() <= 1.0e-9
            && a.pre_cab_w[ch]
                .iter()
                .zip(&b.pre_cab_w[ch])
                .all(|(x, y)| (x - y).abs() <= 1.0e-9)
    })
}

fn fast_mono_prescreen_ok(
    besr_db: f64,
    band_esr_db: &[f64; 3],
    seg_esr_db: f64,
    ltas_max_db: f64,
    ltas_excluded_bands: [bool; 33],
    prev_ch: Option<&PerceptualCertificationSummary>,
) -> bool {
    prev_ch.is_none_or(|p| {
        let valid_db = |x: f64| !x.is_nan() && x < f64::INFINITY;
        valid_db(besr_db)
            && band_esr_db.iter().copied().all(valid_db)
            && valid_db(seg_esr_db)
            && ltas_max_db.is_finite()
            && ltas_max_db >= 0.0
            && besr_db <= p.raw_besr_db
            && (0..3).all(|b| band_esr_db[b] <= p.band_esr_db[b])
            && ltas_max_db <= p.ltas_max_db
            && seg_esr_db <= p.seg_esr_db
            && ltas_excluded_bands == p.ltas_excluded_bands
    })
}

fn eval_single_chan_summary(
    tier: Tier,
    target: &[f64],
    model: &[f64],
    sr_hz: f64,
    warmup: usize,
    prev_ch: Option<&PerceptualCertificationSummary>,
) -> PerceptualCertificationSummary {
    let th = TierQualityThresholds::for_tier(tier);
    let (sig_e, err_e, nse_e) = split_bezout_3band_energies(target, model, None, sr_hz, warmup);
    let besr_rep = evaluate_besr_from_band_energies(sig_e, err_e, nse_e);
    let besr_qual =
        evaluate_noise_qualified_besr(&besr_rep, th.per_band_besr_db, th.composite_besr_db);
    let unseen_relax_db = th.unseen_control_besr_db - th.composite_besr_db;
    let unseen_band_goals = th.per_band_besr_db.map(|g| g + unseen_relax_db);
    let unseen_qual =
        evaluate_noise_qualified_besr(&besr_rep, unseen_band_goals, th.unseen_control_besr_db);

    let valid_slice = target.len() == model.len() && warmup < target.len() && !target.is_empty();
    let (tgt_post, mdl_post) = if valid_slice {
        (&target[warmup..], &model[warmup..])
    } else {
        (&[][..], &[][..])
    };
    let ltas_rep = evaluate_ltas_33band_report(tgt_post, mdl_post, None, sr_hz, true);
    let (seg_esr_db, seg_active_count, seg_valid) =
        compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
            sample_rate_hz: sr_hz,
            target: tgt_post,
            model: mdl_post,
            noise: None,
        }]);

    if !fast_mono_prescreen_ok(
        besr_rep.besr_db,
        &besr_rep.band_esr_db,
        seg_esr_db,
        ltas_rep.max_adj_deviation_db,
        ltas_rep.excluded_bands,
        prev_ch,
    ) {
        return PerceptualCertificationSummary {
            family: ArchetypeFamily::CleanPedalPreamp,
            tier,
            raw_besr_db: besr_rep.besr_db,
            besr_db: besr_qual.qualified_besr_db,
            band_esr_db: besr_rep.band_esr_db,
            qualified_band_mask: besr_qual.qualified_band_mask,
            mr_iir: prev_ch.map_or(f64::INFINITY, |p| p.mr_iir + 1.0),
            ltas_max_db: ltas_rep.max_adj_deviation_db,
            ltas_unadjusted_max_db: ltas_rep.max_unadjusted_deviation_db,
            ltas_excluded_bands: ltas_rep.excluded_bands,
            ltas_residual_band_power: ltas_rep.residual_band_power,
            seg_esr_db,
            passes_target: false,
            passes_hard_ceiling: false,
        };
    }

    let mr_iir = compute_mr_iir_loss(target, model, None, sr_hz, warmup);
    let companion_ok = mr_iir.is_finite()
        && mr_iir >= 0.0
        && ltas_rep.available
        && !ltas_rep.excluded_band_guard_failed
        && ltas_rep.max_adj_deviation_db <= th.max_ltas_db
        && seg_valid
        && seg_active_count > 0
        && !seg_esr_db.is_nan()
        && seg_esr_db < f64::INFINITY;

    PerceptualCertificationSummary {
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
        passes_target: besr_qual.passes && companion_ok,
        passes_hard_ceiling: unseen_qual.passes && companion_ok,
    }
}

pub(super) fn certify_cold_tier_with_prescreen<B: TrainerBackend>(
    cold: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    tgt_l_f64: &[f64],
    tgt_r_f64: &[f64],
    sr_hz: f64,
    prev_req: Option<(
        &PerceptualCertificationSummary,
        Option<&PerceptualCertificationSummary>,
    )>,
) -> (
    PerceptualCertificationSummary,
    Option<PerceptualCertificationSummary>,
) {
    let model = cold.extract_tier(tier);
    let n = stem.usable_len();
    let warmup = stem.warmup.min(n);
    let mut inst = V3CoreInstance::<B>::from_cold_model(&model);
    let (mut out_l_f32, mut out_r_f32) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    let r_in = if cold.mono_out {
        &stem.in_l[..n]
    } else {
        &stem.in_r[..n]
    };
    inst.process_block_stereo(&stem.in_l[..n], r_in, &mut out_l_f32, &mut out_r_f32);
    let out_l: Vec<f64> = out_l_f32.iter().map(|&v| f64::from(v)).collect();
    let sl = eval_single_chan_summary(
        tier,
        &tgt_l_f64[..n],
        &out_l,
        sr_hz,
        warmup,
        prev_req.map(|(pl, _)| pl),
    );
    let sr = (!cold.mono_out).then(|| {
        let out_r: Vec<f64> = out_r_f32.iter().map(|&v| f64::from(v)).collect();
        eval_single_chan_summary(
            tier,
            &tgt_r_f64[..n],
            &out_r,
            sr_hz,
            warmup,
            prev_req.and_then(|(_, pr)| pr),
        )
    });
    (sl, sr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shares_pre_cab() {
        let a = V3TierReadoutF64::zeros();
        let mut b = V3TierReadoutF64::zeros();
        assert!(shares_pre_cab(&a, &b, 2));
        b.pre_cab_w[0][0] = 0.25;
        assert!(!shares_pre_cab(&a, &b, 1));
    }
}
