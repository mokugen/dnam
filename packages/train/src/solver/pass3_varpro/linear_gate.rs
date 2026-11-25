//! Universal Pass-3 pure-linear parsimony gate omitting nonlinear `CoreQuad` readout columns
//! (`w_ro[..][6..12]`) and `2×` oversampling when pure-linear `ModalConvolver` + `Op-T` + `ro_pole`
//! bases already explain the target (`ref[train.pipeline.nested-varpro-schur]`, `REQ-1.9`).

use crate::error::TrainError;
use crate::runner::TrainRunner;

use super::super::model::{SonaModelF64, StereoTrainingPairs};
use super::{
    Pass3InnerOutcome, Pass3VarProOptions, Pass3VarProReport, run_pass3_varpro_readout_inner,
};

const PARSIMONY_FULL_MAX_NESR: f64 = 0.02;
const PARSIMONY_LIN_GOOD_NESR: f64 = 5.0e-3;
const PARSIMONY_LIN_DEEP_NESR: f64 = 5.0e-4;
const PARSIMONY_REL_IMPROVE_GOOD: f64 = 0.50;
const PARSIMONY_REL_IMPROVE_DEEP: f64 = 0.25;

const _: () = assert!(
    PARSIMONY_LIN_DEEP_NESR < PARSIMONY_LIN_GOOD_NESR
        && PARSIMONY_LIN_GOOD_NESR < PARSIMONY_FULL_MAX_NESR
        && PARSIMONY_REL_IMPROVE_DEEP < PARSIMONY_REL_IMPROVE_GOOD
        && PARSIMONY_REL_IMPROVE_GOOD < 1.0
);

/// Returns `true` when all `CoreQuad` nonlinear wave-shaping/SVF drives (`adaa_drive`,
/// `wavefolder_mix`, `svf_k_nl`) are zero.
#[must_use]
pub(super) fn has_zero_core_nonlinear_drives(model: &SonaModelF64) -> bool {
    model.core_quads.iter().all(|q| {
        q.adaa_drive.iter().all(|&v| v.abs() <= 1.0e-12)
            && q.wavefolder_mix.iter().all(|&v| v.abs() <= 1.0e-12)
            && q.svf_k_nl.iter().all(|&v| v.abs() <= 1.0e-12)
    })
}

/// Evaluates whether the pure-linear subspace (`suppress_post_core_lane_mask = u64::MAX`,
/// `oversample_upper_tiers_2x = false`) explains the target accurately without nonlinear `CoreQuad`
/// overfitting, replacing `model` and returning `Some((report, outcome))` when adopted.
pub(super) fn maybe_select_pure_linear_pass3(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    base_snapshot: &SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    w: usize,
    full_outcome: Pass3InnerOutcome,
) -> Result<Option<(Pass3VarProReport, Pass3InnerOutcome)>, TrainError> {
    if has_zero_core_nonlinear_drives(base_snapshot) {
        return Ok(None);
    }
    let n = pairs.aligned_len();
    let m = n.saturating_sub(w).max(1);
    let tgt_var = 0.5
        * (pairs.target_l[w..n].iter().map(|&v| v * v).sum::<f64>()
            + pairs.target_r[w..n].iter().map(|&v| v * v).sum::<f64>())
        / (m as f64);
    if !tgt_var.is_finite()
        || tgt_var <= 1.0e-18
        || !full_outcome.ultra_mse.is_finite()
        || full_outcome.ultra_mse / tgt_var >= PARSIMONY_FULL_MAX_NESR
    {
        return Ok(None);
    }

    let mut lin_model = base_snapshot.clone();
    lin_model.oversample_upper_tiers_2x = false;
    lin_model.tier_overrides = [None, None, None, None];
    lin_model.prefix_override_needed = [false; 4];
    let lin_opts = Pass3VarProOptions {
        suppress_direct_s2_linear: false,
        suppress_post_core_lane_mask: u64::MAX,
    };
    let (lin_rep, lin_out) =
        run_pass3_varpro_readout_inner(runner, &mut lin_model, pairs, w, lin_opts)?;
    let lin_nesr = lin_out.ultra_mse / tgt_var;
    let adopt = (lin_nesr < PARSIMONY_LIN_GOOD_NESR
        && full_outcome.ultra_mse >= PARSIMONY_REL_IMPROVE_GOOD * lin_out.ultra_mse)
        || (lin_nesr < PARSIMONY_LIN_DEEP_NESR
            && full_outcome.ultra_mse >= PARSIMONY_REL_IMPROVE_DEEP * lin_out.ultra_mse);
    if adopt {
        *model = lin_model;
        Ok(Some((lin_rep, lin_out)))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use core::f64::consts::TAU;

    use crate::solver::pass3_varpro::run_pass3_varpro_readout;

    use super::*;

    #[test]
    fn test_pure_linear_parsimony_gate_selects_linear_on_lti_and_keeps_nonlinear_on_tanh() {
        let runner = TrainRunner::cpu();
        let sr = 48_000.0_f64;
        let n = 1_536_usize;
        let in_l: Vec<f64> = (0..n)
            .map(|t| {
                let ts = (t as f64) / sr;
                0.35 * (TAU * 110.0 * ts).sin()
                    + 0.25 * (TAU * 620.0 * ts).cos()
                    + 0.15 * (TAU * 2_400.0 * ts).sin()
            })
            .collect();
        let in_r = in_l.clone();

        // Populate nonlinear CoreQuads on both tests to verify parsimony gating behavior.
        let mut seed_model = SonaModelF64::new(sr);
        seed_model.active_fir_taps = 64;
        seed_model.oversample_upper_tiers_2x = true;
        seed_model.core_quads[0].svf_g = [0.05, 0.12, 0.3, 0.8];
        seed_model.core_quads[0].svf_k = [1.2; 4];
        seed_model.core_quads[0].tone_dir = [0.75; 4];
        seed_model.core_quads[0].adaa_drive = [1.5, 2.5, 4.0, 6.0];
        assert!(!has_zero_core_nonlinear_drives(&seed_model));

        // Case A: Pure LTI FIR target -> parsimony gate must adopt pure-linear (w_ro == 0, 1x).
        let mut lti_tgt_l = vec![0.0_f64; n];
        for t in 0..n {
            lti_tgt_l[t] = 0.65 * in_l[t] + if t >= 6 { -0.25 * in_l[t - 6] } else { 0.0 };
        }
        let mut lti_model = seed_model.clone();
        let lti_rep = run_pass3_varpro_readout(
            &runner,
            &mut lti_model,
            &in_l,
            &in_r,
            &lti_tgt_l,
            &lti_tgt_l,
            64,
        )
        .unwrap();
        assert!(lti_rep.tier_mse[3] < 1.0e-5, "mse={}", lti_rep.tier_mse[3]);
        assert!(!lti_model.oversample_upper_tiers_2x);
        assert!(
            lti_model
                .w_ro
                .iter()
                .all(|row| row[6..12].iter().all(|&v| v == 0.0))
        );

        // Case B: Nonlinear distortion target -> must keep nonlinear CoreQuad readouts active.
        let nl_tgt_l: Vec<f64> = in_l
            .iter()
            .map(|&x| 0.40 * x + 0.35 * (3.5 * x).tanh())
            .collect();
        let mut nl_model = seed_model;
        let nl_rep = run_pass3_varpro_readout(
            &runner,
            &mut nl_model,
            &in_l,
            &in_r,
            &nl_tgt_l,
            &nl_tgt_l,
            64,
        )
        .unwrap();
        assert!(nl_rep.tier_mse[3] < 0.02);
        assert!(
            nl_model
                .w_ro
                .iter()
                .any(|row| row[6..12].iter().any(|&v| v.abs() > 1.0e-4)),
            "nonlinear target should retain CoreQuad w_ro columns"
        );
    }
}
