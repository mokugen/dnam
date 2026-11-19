#![forbid(unsafe_code)]
//! Exact cold-model perceptual prescreen and `pre_cab` sharing predicate for `v3` polish.

use dnam_engine::Tier;
use dnam_engine::stages::{V3CoreInstance, V3MatryoshkaModel};

use crate::metrics::TargetRepeatabilityProfile;
use crate::metrics::certification::{PerceptualCertificationSummary, TargetChannelCertCache};
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

#[allow(clippy::too_many_arguments)]
pub(super) fn certify_cold_tier_with_prescreen<B: TrainerBackend>(
    cold: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
    cert_l: &mut TargetChannelCertCache,
    cert_r: Option<&mut TargetChannelCertCache>,
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
    let mut inst = V3CoreInstance::<B>::from_cold_model(&model);
    let (mut out_l_f32, mut out_r_f32) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    let keep_in_r = !cold.mono_out || model.feat_params.key_link > 0.0;
    let r_in = if !keep_in_r {
        &stem.in_l[..n]
    } else {
        &stem.in_r[..n]
    };
    inst.process_block_stereo(&stem.in_l[..n], r_in, &mut out_l_f32, &mut out_r_f32);
    let det_profile = TargetRepeatabilityProfile::deterministic();
    let out_l: Vec<f64> = out_l_f32.iter().map(|&v| f64::from(v)).collect();
    let sl = cert_l.certify_universal_with_prescreen(
        tier,
        &out_l,
        &det_profile,
        prev_req.map(|(pl, _)| pl),
    );
    let sr = if cold.mono_out {
        None
    } else {
        cert_r.map(|cr| {
            let out_r: Vec<f64> = out_r_f32.iter().map(|&v| f64::from(v)).collect();
            cr.certify_universal_with_prescreen(
                tier,
                &out_r,
                &det_profile,
                prev_req.and_then(|(_, pr)| pr),
            )
        })
    };
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
