#![forbid(unsafe_code)]
//! Black-box Stage 0 sub-harmonic divider (`Op-P`, `-1/-2 Oct` + Octavia) identification.
//!
//! Strictly signal-driven: probes input/target waveforms for sub-octave variance before running
//! a coarse-to-fine pole grid search and damped Gauss-Newton fit over `V3Stage0Hot`, accepting
//! the Stage 0 front-end only if end-to-end 4-tier certification improves strictly cleanly.

mod fit;

use dnam_engine::Tier;
use dnam_engine::stages::router::{V3Stage0Hot, V3Stage0Params};
use dnam_engine::stages::{V3CoreInstance, V3MatryoshkaModel};

use crate::metrics::TargetRepeatabilityProfile;
use crate::metrics::certification::{
    PerceptualCertificationSummary, TargetChannelCertCache, verify_monotone_4tier_summaries,
};
use crate::solver::backends::TrainerBackend;
use crate::solver::canonical_besr_warmup;

use super::model_f64::{V3TrainDataset, V3TrainStem};
use fit::{
    FIR_TAPS, MAX_PROBE_BODY_SAMPLES, SubOctFeatures, collect_stem_pair_features,
    fit_stage0_params_from_dataset, ridge_ls_r2,
};

const SUBOCTAVE_DELTA_R2_GATE: f64 = 0.025;
const MIN_STAGE0_BESR_IMPROVEMENT_DB: f64 = 1.0;

fn stem_has_suboctave_delta_r2(
    stem: &V3TrainStem,
    warmup_override: Option<usize>,
    max_cap: usize,
    lpf_pole: f32,
) -> f64 {
    let mut feat = SubOctFeatures {
        x_g: Vec::new(),
        abs_x: Vec::new(),
        env: Vec::new(),
        s1: Vec::new(),
        s2: Vec::new(),
        tgt: Vec::new(),
        valid_mask: Vec::new(),
    };
    collect_stem_pair_features(
        stem,
        warmup_override,
        max_cap,
        lpf_pole,
        0.98,
        false,
        &mut feat,
    );
    let n = feat.tgt.len();
    if n < 128 {
        return 0.0;
    }
    let mut max_delta = 0.0_f64;
    for lag in 0..FIR_TAPS {
        let mut rows_base = Vec::with_capacity(n);
        let mut rows_sub = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in FIR_TAPS..n {
            if !feat.valid_mask[i] {
                continue;
            }
            let j = i - lag;
            let x = feat.x_g[j];
            let xp = feat.x_g[j - 1];
            let ax = feat.abs_x[j];
            let axp = feat.abs_x[j - 1];
            let e = feat.env[j];
            let tx = (2.0 * x).tanh();
            rows_base.push([1.0, x, xp, ax, axp, e, tx]);
            rows_sub.push([
                1.0,
                x,
                xp,
                ax,
                axp,
                e,
                tx,
                feat.s1[j],
                feat.s1[j - 1],
                feat.s2[j],
                feat.s2[j - 1],
            ]);
            ys.push(feat.tgt[i]);
        }
        if let (Some((_, r2_base)), Some((_, r2_sub))) =
            (ridge_ls_r2(&rows_base, &ys), ridge_ls_r2(&rows_sub, &ys))
        {
            max_delta = max_delta.max(r2_sub - r2_base);
        }
    }
    max_delta
}

/// Returns `true` when `dataset` exhibits strong unique sub-harmonic (`f0/2`, `f0/4`) variance
/// across body or sweep stems under `0..8` sample group-delay lag compensation.
#[must_use]
pub fn probe_stage0_suboctave(dataset: &V3TrainDataset) -> bool {
    const PROBE_LPF_POLES: [f32; 3] = [0.90, 0.945, 0.965];
    if let Some(stem0) = dataset.stems.first()
        && PROBE_LPF_POLES.iter().any(|&lp| {
            stem_has_suboctave_delta_r2(stem0, None, MAX_PROBE_BODY_SAMPLES, lp)
                >= SUBOCTAVE_DELTA_R2_GATE
        })
    {
        return true;
    }
    if let Some(stem1) = dataset.stems.get(1)
        && PROBE_LPF_POLES.iter().any(|&lp| {
            stem_has_suboctave_delta_r2(stem1, Some(192), 5_952, lp) >= SUBOCTAVE_DELTA_R2_GATE
        })
    {
        return true;
    }
    false
}

fn transform_stem_with_stage0(st: &V3TrainStem, stage0: &V3Stage0Params) -> V3TrainStem {
    let n = st.usable_len();
    let mut hot = V3Stage0Hot::from_cold(stage0);
    let mut in_l = Vec::with_capacity(n);
    let mut in_r = Vec::with_capacity(n);
    for i in 0..n {
        let [dl, dr, _, _] = hot.step_stereo(st.in_l[i], st.in_r[i]);
        in_l.push(dl);
        in_r.push(dr);
    }
    V3TrainStem {
        in_l,
        in_r,
        tgt_l: st.tgt_l[..n].to_vec(),
        tgt_r: st.tgt_r[..n].to_vec(),
        warmup: st.warmup,
    }
}

fn transform_dataset_with_stage0(
    dataset: &V3TrainDataset,
    stage0: &V3Stage0Params,
) -> V3TrainDataset {
    let mut out = if dataset
        .stems
        .first()
        .is_some_and(|s| s.usable_len() >= 16_384)
        && dataset.stems.len() > 1
    {
        let stem0_s0 = transform_stem_with_stage0(&dataset.stems[0], stage0);
        V3TrainDataset::new_weighted(vec![(stem0_s0, 1.0)], dataset.sample_rate_hz)
    } else {
        let stems: Vec<(V3TrainStem, f64)> = dataset
            .stems
            .iter()
            .enumerate()
            .map(|(idx, st)| {
                (
                    transform_stem_with_stage0(st, stage0),
                    dataset.stem_weights.get(idx).copied().unwrap_or(1.0),
                )
            })
            .collect();
        let mut ds = V3TrainDataset::new_weighted_opts(stems, dataset.sample_rate_hz, false, false);
        ds.gabor_summaries = dataset.gabor_summaries.clone();
        ds.welch33_summaries = dataset.welch33_summaries.clone();
        ds.stem_band_weights = dataset.stem_band_weights.clone();
        ds
    };
    out.gabor_env_weight = dataset.gabor_env_weight;
    out.gabor_ltas_weight = dataset.gabor_ltas_weight;
    out.welch_ltas_weight = dataset.welch_ltas_weight;
    out
}

fn evaluate_4tier_summaries<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    stem: &V3TrainStem,
    sr_hz: f64,
    mono_out: bool,
) -> Option<(
    [PerceptualCertificationSummary; 4],
    Option<[PerceptualCertificationSummary; 4]>,
)> {
    let n = stem.usable_len();
    let warmup = stem.warmup.max(canonical_besr_warmup(n));
    if n <= warmup + 64 {
        return None;
    }
    let tl: Vec<f64> = stem.tgt_l[..n].iter().map(|&v| f64::from(v)).collect();
    let mut cache_l = TargetChannelCertCache::new(&tl, sr_hz, warmup);
    let mut cache_r = (!mono_out).then(|| {
        let tr: Vec<f64> = stem.tgt_r[..n].iter().map(|&v| f64::from(v)).collect();
        TargetChannelCertCache::new(&tr, sr_hz, warmup)
    });

    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];
    let mut mdl_64 = vec![0.0_f64; n];
    let mut sums_l = Vec::with_capacity(4);
    let mut sums_r = Vec::with_capacity(4);
    let det_profile = TargetRepeatabilityProfile::deterministic();

    for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let cold = model.extract_tier(tier);
        let mut inst = V3CoreInstance::<B>::from_cold_model(&cold);
        let keep_in_r = !cold.mono_out || cold.feat_params.key_link > 0.0;
        let in_r = if keep_in_r {
            &stem.in_r[..n]
        } else {
            &stem.in_l[..n]
        };
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(&stem.in_l[..n], in_r, &mut out_l, &mut out_r);
        for (dst, &src) in mdl_64.iter_mut().zip(&out_l) {
            *dst = f64::from(src);
        }
        let sl = cache_l.certify_universal_with_prescreen(tier, &mdl_64, &det_profile, None);
        sums_l.push(sl);
        if let Some(ref mut cr) = cache_r {
            for (dst, &src) in mdl_64.iter_mut().zip(&out_r) {
                *dst = f64::from(src);
            }
            let sr = cr.certify_universal_with_prescreen(tier, &mdl_64, &det_profile, None);
            sums_r.push(sr);
        }
    }
    let arr_l: [PerceptualCertificationSummary; 4] = sums_l.try_into().ok()?;
    let arr_r: Option<[PerceptualCertificationSummary; 4]> = if mono_out {
        None
    } else {
        Some(sums_r.try_into().ok()?)
    };
    Some((arr_l, arr_r))
}

/// Identifies `Stage 0` (`Op-P`) sub-octave parameters first (skipping redundant baseline training
/// when all 4 tiers pass cleanly and monotonically), falling back to a comparator `baseline` only if
/// needed.
#[inline(never)]
#[must_use]
pub fn fit_stage0_first_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
    core_fit_fn: fn(&V3TrainDataset, bool) -> V3MatryoshkaModel,
) -> Option<V3MatryoshkaModel> {
    if !probe_stage0_suboctave(dataset) {
        return None;
    }
    let s0_params = fit_stage0_params_from_dataset(dataset)?;
    let s0_dataset = transform_dataset_with_stage0(dataset, &s0_params);
    let mut cand = core_fit_fn(&s0_dataset, mono_out);
    cand.stage0 = s0_params;

    let stem0 = dataset.stems.first()?;
    let (cand_l, cand_r) =
        evaluate_4tier_summaries::<B>(&cand, stem0, dataset.sample_rate_hz, mono_out)?;
    let mono_ok = verify_monotone_4tier_summaries(&cand_l)
        && cand_r.as_ref().is_none_or(verify_monotone_4tier_summaries);
    let all_pass = cand_l.iter().all(|s| s.passes_target)
        && cand_r
            .as_ref()
            .is_none_or(|sr| sr.iter().all(|s| s.passes_target));
    if mono_ok && all_pass {
        return Some(cand);
    }
    let baseline = core_fit_fn(dataset, mono_out);
    let (base_l, base_r) =
        evaluate_4tier_summaries::<B>(&baseline, stem0, dataset.sample_rate_hz, mono_out)?;
    let base_ultra = base_l[3].raw_besr_db.max(
        base_r
            .as_ref()
            .map_or(f64::NEG_INFINITY, |r| r[3].raw_besr_db),
    );
    let cand_ultra = cand_l[3].raw_besr_db.max(
        cand_r
            .as_ref()
            .map_or(f64::NEG_INFINITY, |r| r[3].raw_besr_db),
    );
    if mono_ok && cand_ultra <= base_ultra - MIN_STAGE0_BESR_IMPROVEMENT_DB {
        Some(cand)
    } else {
        Some(baseline)
    }
}

/// Identifies `Stage 0` (`Op-P`) sub-octave parameters and fits a joint Matryoshka backbone
/// whenever raw stem waveforms pass the unique sub-harmonic variance gate and improve `Ultra` bESR.
#[inline(never)]
#[must_use]
pub fn identify_and_fit_stage0_matryoshka_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
    baseline: &V3MatryoshkaModel,
    core_fit_fn: fn(&V3TrainDataset, bool) -> V3MatryoshkaModel,
) -> Option<V3MatryoshkaModel> {
    if !probe_stage0_suboctave(dataset) {
        return None;
    }
    let s0_params = fit_stage0_params_from_dataset(dataset)?;
    let s0_dataset = transform_dataset_with_stage0(dataset, &s0_params);
    let mut cand = core_fit_fn(&s0_dataset, mono_out);
    cand.stage0 = s0_params;

    let stem0 = dataset.stems.first()?;
    let (base_l, base_r) =
        evaluate_4tier_summaries::<B>(baseline, stem0, dataset.sample_rate_hz, mono_out)?;
    let (cand_l, cand_r) =
        evaluate_4tier_summaries::<B>(&cand, stem0, dataset.sample_rate_hz, mono_out)?;

    let mono_ok = verify_monotone_4tier_summaries(&cand_l)
        && cand_r.as_ref().is_none_or(verify_monotone_4tier_summaries);
    let base_ultra = base_l[3].raw_besr_db.max(
        base_r
            .as_ref()
            .map_or(f64::NEG_INFINITY, |r| r[3].raw_besr_db),
    );
    let cand_ultra = cand_l[3].raw_besr_db.max(
        cand_r
            .as_ref()
            .map_or(f64::NEG_INFINITY, |r| r[3].raw_besr_db),
    );
    (mono_ok && cand_ultra <= base_ultra - MIN_STAGE0_BESR_IMPROVEMENT_DB).then_some(cand)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backends::ScalarTrainerBackend;
    use std::f64::consts::TAU;

    fn make_probe_input(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let env = 0.55 + 0.45 * (TAU * 3.1 * t).sin();
                (0.45 * env * (TAU * 130.0 * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn test_stage0_probe_rejects_asymmetric_clipping_and_detects_suboctave() {
        let n = 4_096;
        let dry = make_probe_input(n);
        let asym_tgt: Vec<f32> = dry
            .iter()
            .map(|&x| (2.5 * f64::from(x) + 0.8 * f64::from(x * x)).tanh() as f32)
            .collect();
        let ds_asym = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: dry.clone(),
                in_r: dry.clone(),
                tgt_l: asym_tgt.clone(),
                tgt_r: asym_tgt,
                warmup: 256,
            }],
            48_000.0,
        );
        assert!(!probe_stage0_suboctave(&ds_asym));

        let mut planted = V3Stage0Params::identity();
        planted.op_p_lpf_pole = 0.945;
        planted.op_p_env_pole = 0.982;
        planted.op_p_sub_mix = 1.25;
        planted.op_p_sub2_mix = 0.65;
        planted.op_p_oct_mix = 1.60;
        planted.op_p_env_mix = -0.80;
        let mut hot = V3Stage0Hot::from_cold(&planted);
        let sub_tgt: Vec<f32> = dry
            .iter()
            .map(|&x| {
                let [dl, _, _, _] = hot.step_stereo(x, x);
                (0.8 * f64::from(dl)).tanh() as f32
            })
            .collect();
        let ds_sub = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: dry.clone(),
                in_r: dry,
                tgt_l: sub_tgt.clone(),
                tgt_r: sub_tgt,
                warmup: 256,
            }],
            48_000.0,
        );
        assert!(probe_stage0_suboctave(&ds_sub));
        let fitted = fit_stage0_params_from_dataset(&ds_sub).expect("Stage 0 fit succeeds");
        assert!(fitted.op_p_sub_mix.abs() > 0.35);

        fn mock_scaled_core(ds: &V3TrainDataset, mono: bool) -> V3MatryoshkaModel {
            let mut m = V3MatryoshkaModel::identity(false, mono);
            let mut xy = 0.0_f64;
            let mut xx = 0.0_f64;
            if let Some(st) = ds.stems.first() {
                for i in st.warmup..st.usable_len() {
                    let x = f64::from(st.in_l[i]);
                    let y = f64::from(st.tgt_l[i]);
                    xy += x * y;
                    xx += x * x;
                }
            }
            let gain = if xx > 1.0e-12 {
                (xy / xx).clamp(-8.0, 8.0) as f32
            } else {
                1.0
            };
            for ro in &mut m.readouts {
                ro.fir_l[0] = gain;
                ro.fir_r[0] = gain;
            }
            m
        }
        let baseline = mock_scaled_core(&ds_sub, true);
        let accepted = identify_and_fit_stage0_matryoshka_with_backend::<ScalarTrainerBackend>(
            &ds_sub,
            true,
            &baseline,
            mock_scaled_core,
        )
        .expect("Stage 0 improves synthetic sub-octave divider");
        assert!(accepted.stage0.op_p_sub_mix.abs() > 0.35);
    }
}
