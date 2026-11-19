#![forbid(unsafe_code)]
//! Full-stem (`0..96,000`) Welch-33 LTAS target summary, trajectory cache, and subspace polish driver
//! so `has_sweep` stems optimize spectral uniformity on the exact `cert_stem0` certification domain.

use dnam_engine::stages::V3MatryoshkaModel;
use dnam_engine::stages::stage2_gru::{NUM_GRU_BLOCKS, tier_block_lanes_array, tier_total_gru_lanes};

use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::backends::TrainerBackend;

use super::super::basis::{
    ALL_TIERS, PreSplitStemChan, StemChanTraj, SubspaceGramData, build_tier_columns,
    eval_s2_from_readout, unpack_theta_to_readout,
};
use super::super::ltas_adj::{Welch33TargetSummary, build_welch33_target_summary};
use super::super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset, V3TrainStem};
use super::super::primitives::par_map_indexed;
use super::extract::extract_f32_shared_for_rate;
use super::subspace::subspace_lbfgs_polish_chan;

pub(super) struct CertWelchData<'a> {
    pub(super) stem0: &'a V3TrainStem,
    pub(super) sum_lr: [Welch33TargetSummary; 2],
    pub(super) trajs_1x: Option<[StemChanTraj; 2]>,
    pub(super) trajs_2x: Option<[StemChanTraj; 2]>,
}

impl<'a> CertWelchData<'a> {
    pub(super) fn build<B: TrainerBackend>(
        cold_init: &V3MatryoshkaModel,
        cert_stem0: Option<&'a V3TrainStem>,
        dataset: &V3TrainDataset,
        need_1x: bool,
        need_2x: bool,
        bg_1x: [usize; NUM_GRU_BLOCKS],
        bg_2x: [usize; NUM_GRU_BLOCKS],
    ) -> Option<Self> {
        let ds_n0 = dataset.stems.first().map_or(0, V3TrainStem::usable_len);
        let cs = cert_stem0.filter(|s| s.usable_len() > ds_n0)?;
        let nc = cs.usable_len();
        let wc = cs.warmup.min(nc);
        let tl: Vec<f64> = cs.tgt_l[..nc].iter().map(|&v| f64::from(v)).collect();
        let wl = build_welch33_target_summary(&tl, wc, dataset.sample_rate_hz);
        let wr = if cold_init.mono_out {
            wl.clone()
        } else {
            let tr: Vec<f64> = cs.tgt_r[..nc].iter().map(|&v| f64::from(v)).collect();
            build_welch33_target_summary(&tr, wc, dataset.sample_rate_hz)
        };
        let mut rate_jobs: Vec<(bool, [usize; NUM_GRU_BLOCKS])> = Vec::with_capacity(2);
        if need_1x {
            rate_jobs.push((false, bg_1x));
        }
        if need_2x {
            rate_jobs.push((true, bg_2x));
        }
        let extracted = par_map_indexed(rate_jobs.len(), |ji| {
            let (rate_os2x, block_groups) = rate_jobs[ji];
            (
                rate_os2x,
                extract_f32_shared_for_rate::<B>(
                    cold_init,
                    rate_os2x,
                    block_groups,
                    &cs.in_l[..nc],
                    &cs.in_r[..nc],
                ),
            )
        });
        let mut trajs_1x = None;
        let mut trajs_2x = None;
        for (rate_os2x, tpair) in extracted {
            if rate_os2x {
                trajs_2x = Some(tpair);
            } else {
                trajs_1x = Some(tpair);
            }
        }
        Some(Self {
            stem0: cs,
            sum_lr: [wl, wr],
            trajs_1x,
            trajs_2x,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_subspace_polish<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    tau: usize,
    prev_sum: [Option<&PerceptualCertificationSummary>; 2],
    ls_readout: &V3TierReadoutF64,
    grams: &[Option<SubspaceGramData>; 2],
    thetas: &[Vec<f64>; 2],
    trajs: &[Vec<StemChanTraj>; 2],
    stems_pre: &[Vec<PreSplitStemChan>; 2],
    dataset: &V3TrainDataset,
    cert_welch: Option<&CertWelchData<'_>>,
) -> V3TierReadoutF64 {
    let tier = ALL_TIERS[tau];
    let active_ch = if model.mono_out { 1 } else { 2 };
    let rate_os2x = model.tier_os2x[tau];
    model.readouts[tau] = ls_readout.clone();
    for ch in 0..active_ch {
        if let Some(ref g) = grams[ch] {
            let mut cols0_ch = Vec::new();
            let mut cw_sum: Option<&Welch33TargetSummary> = None;
            if let Some(cw) = cert_welch
                && let Some(c_pair) = if rate_os2x {
                    cw.trajs_2x.as_ref()
                } else {
                    cw.trajs_1x.as_ref()
                }
            {
                let nc = cw.stem0.usable_len();
                let tr0 = &c_pair[ch];
                let mut s2_0 = Vec::new();
                eval_s2_from_readout(
                    ls_readout,
                    ch,
                    tier_block_lanes_array(tier),
                    &tr0.u[..nc],
                    &tr0.hp[..nc],
                    &tr0.h[..nc],
                    &mut s2_0,
                );
                cols0_ch = build_tier_columns::<B>(
                    tier,
                    &model.kautz_poles,
                    &tr0.tap[..nc],
                    &tr0.u[..nc],
                    &tr0.hp[..nc],
                    &tr0.h[..nc],
                    &s2_0[..nc],
                );
                cw_sum = Some(&cw.sum_lr[ch]);
            } else if let Some(st0) = stems_pre[ch].first().filter(|s| s.n > 0) {
                let tr0 = &trajs[ch][0];
                let mut s2_0 = Vec::new();
                eval_s2_from_readout(
                    ls_readout,
                    ch,
                    tier_block_lanes_array(tier),
                    &tr0.u[..st0.n],
                    &tr0.hp[..st0.n],
                    &tr0.h[..st0.n],
                    &mut s2_0,
                );
                cols0_ch = build_tier_columns::<B>(
                    tier,
                    &model.kautz_poles,
                    &tr0.tap[..st0.n],
                    &tr0.u[..st0.n],
                    &tr0.hp[..st0.n],
                    &tr0.h[..st0.n],
                    &s2_0[..st0.n],
                );
            }
            let total_h = tier_total_gru_lanes(tier);
            let pre_cab_w_sq: f64 = model.readouts[tau].pre_cab_w[ch]
                .iter()
                .map(|w| w * w)
                .sum();
            let dir_w_budget_sq = (243.0 - pre_cab_w_sq).max(64.0);
            let mut theta = subspace_lbfgs_polish_chan::<B>(
                dataset,
                cw_sum,
                tier,
                prev_sum[ch],
                g,
                &cols0_ch,
                ch,
                total_h,
                dir_w_budget_sq,
                &thetas[ch],
            );
            let dir_w_sq: f64 = theta[9..9 + total_h].iter().map(|v| v * v).sum();
            if pre_cab_w_sq + dir_w_sq > 245.0 && thetas[ch].len() == theta.len() {
                for &alpha in &[0.75_f64, 0.5, 0.25, 0.0] {
                    let blended: Vec<f64> = theta
                        .iter()
                        .zip(&thetas[ch])
                        .map(|(&a, &b)| alpha * a + (1.0 - alpha) * b)
                        .collect();
                    let b_sq: f64 = blended[9..9 + total_h].iter().map(|v| v * v).sum();
                    if pre_cab_w_sq + b_sq <= 245.0 || alpha == 0.0 {
                        theta = blended;
                        break;
                    }
                }
            }
            unpack_theta_to_readout(tier, ch, &theta, &mut model.readouts[tau]);
        }
    }
    model.mirror_mono();
    model.readouts[tau].clone()
}
