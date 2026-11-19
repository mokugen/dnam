#![forbid(unsafe_code)]
//! `1x`/`2x` rate-split monotonicity promotion helper for `v3` polish.

use crate::metrics::certification::{PerceptualCertificationSummary, TargetChannelCertCache};
use crate::solver::backends::TrainerBackend;

use super::super::super::basis::ALL_TIERS;
use super::super::super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset, V3TrainStem};
use super::super::polish_f32_matryoshka_readouts_with_cert_stem;
use super::super::prescreen::certify_cold_tier_with_prescreen;
use super::{ok_for_tau, passes_both};

fn eval_model_split_rank(
    mono_out: bool,
    sl: &[Option<PerceptualCertificationSummary>; 4],
    sr: &[Option<PerceptualCertificationSummary>; 4],
) -> (usize, bool, f64) {
    let mut pass_cnt = 0_usize;
    let mut sum_score = 0.0_f64;
    for t in 0..4 {
        let Some(l) = sl[t].as_ref() else {
            return (0, false, f64::NEG_INFINITY);
        };
        let r_opt = sr[t].as_ref();
        if passes_both(l, &r_opt.cloned()) {
            pass_cnt += 1;
        }
        let besr = r_opt.map_or(l.raw_besr_db, |r| l.raw_besr_db.max(r.raw_besr_db));
        let ltas = r_opt.map_or(l.ltas_max_db, |r| l.ltas_max_db.max(r.ltas_max_db));
        let fin = |v: f64| if v.is_finite() { v } else { 1.0e6 };
        sum_score -= fin(besr) + 2.0 * fin(ltas);
    }
    (pass_cnt, ok_for_tau(3, mono_out, sl, sr), sum_score)
}

/// Promotes a pure `1x` model when a `1x -> 2x` split introduces a `raw_besr_db` regression.
pub(in super::super) fn maybe_promote_1x_rate_split<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    cert_stem0: Option<&V3TrainStem>,
    summaries_l: &[Option<PerceptualCertificationSummary>; 4],
    summaries_r: &[Option<PerceptualCertificationSummary>; 4],
) {
    let has_rate_split_regression = model.os2x
        && model.rate_1x_cal.is_some()
        && !model.tier_os2x[1]
        && model.tier_os2x[2]
        && ({
            let reg_l = summaries_l[2]
                .as_ref()
                .zip(summaries_l[1].as_ref())
                .is_some_and(|(nxt, prv)| nxt.raw_besr_db > prv.raw_besr_db + 1.0e-3);
            let reg_r = summaries_r[2]
                .as_ref()
                .zip(summaries_r[1].as_ref())
                .is_some_and(|(nxt, prv)| nxt.raw_besr_db > prv.raw_besr_db + 1.0e-3);
            reg_l || reg_r
        });
    if !has_rate_split_regression {
        return;
    }
    let mut cold_1x = model.to_matryoshka_cold();
    let Some(cal) = cold_1x.rate_1x_cal.take() else {
        return;
    };
    cold_1x.os2x = false;
    cold_1x.tier_os2x = [false; 4];
    cold_1x.svf = cal.svf;
    cold_1x.feat_params = cal.feat_params;
    cold_1x.blocks = cal.blocks;
    let mut trial_1x = V3MatryoshkaOptF64::from_matryoshka_cold(&cold_1x);
    polish_f32_matryoshka_readouts_with_cert_stem::<B>(&mut trial_1x, dataset, cert_stem0);

    let stem0 = cert_stem0
        .filter(|s| s.usable_len() > 0)
        .or_else(|| dataset.stems.first());
    let Some(stem0) = stem0 else {
        return;
    };
    let n0 = stem0.usable_len();
    let w0 = stem0.warmup.min(n0);
    let tgt_l_f64: Vec<f64> = stem0.tgt_l[..n0].iter().map(|&v| f64::from(v)).collect();
    let mut cert_l = TargetChannelCertCache::new(&tgt_l_f64, dataset.sample_rate_hz, w0);
    let mut cert_r = (!trial_1x.mono_out).then(|| {
        let tgt_r_f64: Vec<f64> = stem0.tgt_r[..n0].iter().map(|&v| f64::from(v)).collect();
        TargetChannelCertCache::new(&tgt_r_f64, dataset.sample_rate_hz, w0)
    });
    let cold_eval = trial_1x.to_matryoshka_cold();
    let mut tr_l: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, None];
    let mut tr_r: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, None];
    for (tau, &tier) in ALL_TIERS.iter().enumerate() {
        let (sl, sr) = certify_cold_tier_with_prescreen::<B>(
            &cold_eval,
            tier,
            stem0,
            &mut cert_l,
            cert_r.as_mut(),
            None,
        );
        tr_l[tau] = Some(sl);
        tr_r[tau] = sr;
    }
    let orig_rank = eval_model_split_rank(model.mono_out, summaries_l, summaries_r);
    let trial_rank = eval_model_split_rank(trial_1x.mono_out, &tr_l, &tr_r);
    if trial_rank > orig_rank {
        *model = trial_1x;
    }
}
