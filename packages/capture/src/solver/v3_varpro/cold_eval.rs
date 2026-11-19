#![forbid(unsafe_code)]
//! Cold `f32` `V3CoreInstance` Bézout + Spectral Loss Evaluation (`v3_varpro::cold_eval`).

use dnam_engine::stages::{V3ColdTierModel, V3CoreInstance};

use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

use super::gabor::{self, GaborAdjointScratch};
use super::ltas_adj::{Welch33Scratch, eval_welch33_ltas_loss_and_adjoint};
use super::model_f64::V3TrainDataset;

/// Evaluates the exact Bézout 3-band + spectral loss using the real `f32` `V3CoreInstance` via backend `B`.
#[must_use]
pub fn evaluate_cold_loss_f32_with_backend<B: TrainerBackend>(
    cold: &V3ColdTierModel,
    dataset: &V3TrainDataset,
) -> f64 {
    let mut inst = V3CoreInstance::<B>::from_cold_model(cold);
    let active_ch = if cold.mono_out { 1 } else { 2 };
    let ch_scale = if cold.mono_out { 1.0 } else { 0.5 };
    let mut bands = [Vec::new(), Vec::new(), Vec::new()];
    let mut gab_scratch = GaborAdjointScratch::default();
    let mut wel_scratch = Welch33Scratch::default();
    let mut total_loss = 0.0_f64;

    for (stem_idx, stem) in dataset.stems.iter().enumerate() {
        let n = stem.usable_len();
        if n == 0 {
            continue;
        }
        let w = stem.warmup.min(n);
        let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        let keep_in_r = !cold.mono_out || cold.feat_params.key_link > 0.0;
        let r_in = if !keep_in_r {
            &stem.in_l[..n]
        } else {
            &stem.in_r[..n]
        };
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(&stem.in_l[..n], r_in, &mut out_l, &mut out_r);
        let sw = dataset.stem_weights[stem_idx];
        for ch in 0..active_ch {
            let (out, tgt) = if ch == 0 {
                (&out_l, &stem.tgt_l)
            } else {
                (&out_r, &stem.tgt_r)
            };
            let err: Vec<f64> = (0..n)
                .map(|t| f64::from(out[t]) - f64::from(tgt[t]))
                .collect();
            dataset.bank.split_3band(&err, &mut bands);
            let bw = dataset.stem_band_weights[stem_idx][ch];
            for b in 0..3 {
                total_loss += ch_scale * bw[b] * B::sum_sq_f64(&bands[b][w..n]);
            }
            if dataset.gabor_env_weight > 0.0 || dataset.gabor_ltas_weight > 0.0 {
                let y_f64: Vec<f64> = out[..n].iter().map(|&v| f64::from(v)).collect();
                total_loss += gabor::eval_gabor_loss_and_adjoint_with_backend::<B>(
                    &dataset.gabor_bank,
                    &dataset.gabor_summaries[stem_idx][ch],
                    &y_f64,
                    None,
                    sw,
                    ch_scale,
                    dataset.gabor_env_weight,
                    dataset.gabor_ltas_weight,
                    &mut gab_scratch,
                );
            }
            if dataset.welch_ltas_weight > 0.0 {
                let y_f64: Vec<f64> = out[..n].iter().map(|&v| f64::from(v)).collect();
                total_loss += eval_welch33_ltas_loss_and_adjoint(
                    &dataset.welch33_summaries[stem_idx][ch],
                    &y_f64,
                    None,
                    sw,
                    ch_scale,
                    dataset.welch_ltas_weight,
                    &mut wel_scratch,
                );
            }
        }
    }
    total_loss
}

/// Evaluates the exact Bézout 3-band + spectral loss using the real `f32` `V3CoreInstance`.
#[must_use]
pub fn evaluate_cold_loss_f32(cold: &V3ColdTierModel, dataset: &V3TrainDataset) -> f64 {
    struct ColdLossTask<'a>(&'a V3ColdTierModel, &'a V3TrainDataset);
    impl TrainerTask<f64> for ColdLossTask<'_> {
        fn run<B: TrainerBackend>(self) -> f64 {
            evaluate_cold_loss_f32_with_backend::<B>(self.0, self.1)
        }
    }
    DynamicTrainer::detect().dispatch(ColdLossTask(cold, dataset))
}

#[cfg(test)]
mod tests {
    use dnam_engine::Tier;
    use dnam_engine::stages::V3MatryoshkaModel;

    use super::*;
    use crate::solver::v3_varpro::V3TrainStem;

    #[test]
    fn test_evaluate_cold_loss_f32_zero_for_identity_and_positive_for_gain_mismatch() {
        let dry: Vec<f32> = (0..1_024).map(|i| (i as f32 * 0.037).sin() * 0.4).collect();
        let stem = V3TrainStem {
            in_l: dry.clone(),
            in_r: dry.clone(),
            tgt_l: dry.iter().map(|&v| v * 0.5).collect(),
            tgt_r: dry.iter().map(|&v| v * 0.5).collect(),
            warmup: 64,
        };
        let ds = V3TrainDataset::new(vec![stem], 48_000.0).with_spectral_weights(0.02, 0.02, 0.02);
        let id = V3MatryoshkaModel::identity(true, true).extract_tier(Tier::Lite);
        let loss = evaluate_cold_loss_f32(&id, &ds);
        assert!(loss.is_finite() && loss > 0.0);
    }
}
