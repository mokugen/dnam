#![forbid(unsafe_code)]
//! `v3` Single-Run 4-Tier Matryoshka (`2x OS`) VarPro + Adjoint L-BFGS + Progressive-Slot OMP Solver.

pub mod adjoint;
pub mod alignment;
pub mod basis;
pub mod bezout7;
pub mod dataset;
pub mod gabor;
pub(crate) mod gru_core;
pub mod lbfgs;
pub mod ltas_adj;
pub mod model_f64;
pub mod omp;
pub mod polish;
pub(crate) mod primitives;
mod seed_init;
pub mod svf_id;

pub use alignment::{build_v3_causal_probe_excitation, estimate_v3_causal_deadtime_and_polarity};
pub use lbfgs::{
    run_lbfgs_core_matryoshka, run_lbfgs_core_matryoshka_with_backend, run_lbfgs_matryoshka,
    run_lbfgs_matryoshka_with_backend,
};
pub use model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset, V3TrainStem};
pub use omp::{
    repair_collided_kautz_poles, select_omp_kautz_poles_matryoshka,
    select_omp_kautz_poles_matryoshka_with_backend,
};
pub use polish::{polish_f32_matryoshka_readouts, polish_f32_matryoshka_readouts_with_backend};

use dnam_engine::Tier;
use dnam_engine::stages::v3_core::{V3ColdTierModel, V3CoreInstance, V3MatryoshkaModel};

#[cfg(test)]
use crate::metrics::{
    ArchetypeFamily, PerceptualCertificationSummary, certify_4metric_signal_chain,
};

use super::backends::{DynamicTrainer, TrainerBackend, TrainerTask};
use adjoint::{AdjointWorkspace, evaluate_matryoshka_loss_f64_with_backend};
use basis::{ALL_TIERS, solve_matryoshka_varpro_cholesky_with_backend};
use gabor::GaborAdjointScratch;
use ltas_adj::{Welch33Scratch, eval_welch33_ltas_loss_and_adjoint};
use primitives::par_map_vec;
use svf_id::prescreen_matryoshka_seeds_with_backend;

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
        let r_in = if cold.mono_out {
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
fn combine_stereo_summary(
    l: PerceptualCertificationSummary,
    r: PerceptualCertificationSummary,
) -> PerceptualCertificationSummary {
    PerceptualCertificationSummary {
        family: l.family,
        tier: l.tier,
        raw_besr_db: l.raw_besr_db.max(r.raw_besr_db),
        besr_db: l.besr_db.max(r.besr_db),
        band_esr_db: std::array::from_fn(|b| l.band_esr_db[b].max(r.band_esr_db[b])),
        qualified_band_mask: std::array::from_fn(|b| {
            l.qualified_band_mask[b] || r.qualified_band_mask[b]
        }),
        mr_iir: l.mr_iir.max(r.mr_iir),
        ltas_max_db: l.ltas_max_db.max(r.ltas_max_db),
        ltas_unadjusted_max_db: l.ltas_unadjusted_max_db.max(r.ltas_unadjusted_max_db),
        ltas_excluded_bands: std::array::from_fn(|b| {
            l.ltas_excluded_bands[b] && r.ltas_excluded_bands[b]
        }),
        ltas_residual_band_power: std::array::from_fn(|b| {
            l.ltas_residual_band_power[b].max(r.ltas_residual_band_power[b])
        }),
        seg_esr_db: l.seg_esr_db.max(r.seg_esr_db),
        passes_target: l.passes_target && r.passes_target,
        passes_hard_ceiling: l.passes_hard_ceiling && r.passes_hard_ceiling,
    }
}

#[cfg(test)]
fn signal_chain_summary_for_cold<B: TrainerBackend>(
    cold: &V3ColdTierModel,
    stem: &V3TrainStem,
    sr_hz: f64,
) -> Option<PerceptualCertificationSummary> {
    let n = stem.usable_len();
    if n <= stem.warmup + 32 {
        return None;
    }
    let mut inst = V3CoreInstance::<B>::from_cold_model(cold);
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    let r_in = if cold.mono_out {
        &stem.in_l[..n]
    } else {
        &stem.in_r[..n]
    };
    inst.reset_to_dc_equilibrium();
    inst.process_block_stereo(&stem.in_l[..n], r_in, &mut out_l, &mut out_r);
    let ol64: Vec<f64> = out_l.iter().map(|&v| f64::from(v)).collect();
    let tl64: Vec<f64> = stem.tgt_l[..n].iter().map(|&v| f64::from(v)).collect();
    let sum_l = certify_4metric_signal_chain(
        ArchetypeFamily::CleanPedalPreamp,
        cold.tier,
        &tl64,
        &ol64,
        None,
        sr_hz,
        stem.warmup,
    );
    if cold.mono_out {
        return Some(sum_l);
    }
    let or64: Vec<f64> = out_r.iter().map(|&v| f64::from(v)).collect();
    let tr64: Vec<f64> = stem.tgt_r[..n].iter().map(|&v| f64::from(v)).collect();
    let sum_r = certify_4metric_signal_chain(
        ArchetypeFamily::CleanPedalPreamp,
        cold.tier,
        &tr64,
        &or64,
        None,
        sr_hz,
        stem.warmup,
    );
    Some(combine_stereo_summary(sum_l, sum_r))
}

fn accept_varpro_if_improves<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    als_iters: usize,
    best_loss: &mut f64,
) {
    let backup = model.clone();
    solve_matryoshka_varpro_cholesky_with_backend::<B>(model, dataset, als_iters);
    let after = evaluate_matryoshka_loss_f64_with_backend::<B>(model, dataset, ws);
    if after.is_finite() && after <= *best_loss {
        *best_loss = after;
    } else {
        *model = backup;
    }
}

fn accept_omp_if_improves<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    best_loss: &mut f64,
) {
    let backup = model.clone();
    select_omp_kautz_poles_matryoshka_with_backend::<B>(model, dataset);
    let after = evaluate_matryoshka_loss_f64_with_backend::<B>(model, dataset, ws);
    if after.is_finite() && after <= *best_loss {
        *best_loss = after;
    } else {
        *model = backup;
    }
}

/// Backend-parameterized single-run 4-tier Matryoshka (`2x OS`) fitter (`VarPro` + L-BFGS + OMP + `f32` polish).
#[must_use]
pub fn fit_v3_matryoshka_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
) -> V3MatryoshkaModel {
    if dataset.stems.is_empty() || dataset.stems.iter().all(|s| s.usable_len() == 0) {
        return V3MatryoshkaModel::identity(true, mono_out);
    }

    // Prescreen 16 circuit templates across 4 archetypes in Lite (G=1, 4k -> 8k Gabor homotopy), keep top 2 (G=1).
    let mut winners = prescreen_matryoshka_seeds_with_backend::<B>(dataset, mono_out, 2);
    if winners.is_empty() {
        winners.push(V3MatryoshkaOptF64::from_matryoshka_cold(
            &V3MatryoshkaModel::identity(true, mono_out),
        ));
    }

    // Stage 1a (G=1, 10k+4k): lock phase with L-BFGS first, then OMP+VarPro+L-BFGS and prune 2 -> 1.
    let ds_s1a = dataset
        .curriculum_subwindow(10_240, 4_096)
        .with_spectral_weights(0.06, 0.03, 0.05);
    let s1a_cands = par_map_vec(winners, |_, mut cand| {
        let mut cws = AdjointWorkspace::default();
        let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut cand, &ds_s1a, 12);
        let mut cand_loss =
            evaluate_matryoshka_loss_f64_with_backend::<B>(&cand, &ds_s1a, &mut cws);
        if !cand_loss.is_finite() {
            cand_loss = f64::INFINITY;
        }
        accept_omp_if_improves::<B>(&mut cand, &ds_s1a, &mut cws, &mut cand_loss);
        accept_varpro_if_improves::<B>(&mut cand, &ds_s1a, &mut cws, 1, &mut cand_loss);
        let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut cand, &ds_s1a, 5);
        let mut s1a_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&cand, &ds_s1a, &mut cws);
        if !s1a_loss.is_finite() {
            s1a_loss = f64::INFINITY;
        }
        accept_varpro_if_improves::<B>(&mut cand, &ds_s1a, &mut cws, 1, &mut s1a_loss);
        (s1a_loss, cand)
    });
    let mut best_model: Option<V3MatryoshkaOptF64> = None;
    let mut best_loss = f64::INFINITY;
    for (s1a_loss, cand) in s1a_cands {
        if s1a_loss.is_finite() && s1a_loss < best_loss {
            best_loss = s1a_loss;
            best_model = Some(cand);
        }
    }
    let mut ws = AdjointWorkspace::default();
    let mut model = best_model.unwrap_or_else(|| {
        V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, mono_out))
    });

    // Stage 1b (G=1 long-horizon sag & RC bias lock: 28_672 + 8_192).
    let ds_s1b = dataset
        .curriculum_subwindow(28_672, 8_192)
        .with_spectral_weights(0.0, 0.0, 0.05);
    let mut s1b_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_s1b, &mut ws);
    if !s1b_loss.is_finite() {
        s1b_loss = f64::INFINITY;
    }
    accept_omp_if_improves::<B>(&mut model, &ds_s1b, &mut ws, &mut s1b_loss);
    accept_varpro_if_improves::<B>(&mut model, &ds_s1b, &mut ws, 1, &mut s1b_loss);
    let s1b_iters = if model.svf_env_oct_sens.abs() > 0.5 {
        16
    } else {
        12
    };
    let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut model, &ds_s1b, s1b_iters);
    let mut s1b_after = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_s1b, &mut ws);
    if !s1b_after.is_finite() {
        s1b_after = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s1b, &mut ws, 1, &mut s1b_after);

    // Stage 2a (G=4 unlock, 4-tier 14k+6k): joint co-adaptation across all 4 groups and tiers.
    svf_id::unlock_matryoshka_groups_after_prescreen(&mut model);
    model.active_tier_weights = [0.22, 0.20, 0.24, 0.34];
    let ds_s2a = dataset
        .curriculum_subwindow(14_336, 6_144)
        .with_spectral_weights(0.0, 0.0, 0.04);
    select_omp_kautz_poles_matryoshka_with_backend::<B>(&mut model, &ds_s2a);
    let mut s2a_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_s2a, &mut ws);
    if !s2a_loss.is_finite() {
        s2a_loss = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s2a, &mut ws, 1, &mut s2a_loss);
    let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut model, &ds_s2a, 10);

    // Stage 2b (G=4 4-tier medium-long refinement on 24_576 + 8_192).
    model.active_tier_weights = [0.18, 0.20, 0.25, 0.37];
    let ds_s2b = dataset
        .curriculum_subwindow(24_576, 8_192)
        .with_spectral_weights(0.0, 0.0, 0.05);
    let mut s2b_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_s2b, &mut ws);
    if !s2b_loss.is_finite() {
        s2b_loss = f64::INFINITY;
    }
    accept_omp_if_improves::<B>(&mut model, &ds_s2b, &mut ws, &mut s2b_loss);
    accept_varpro_if_improves::<B>(&mut model, &ds_s2b, &mut ws, 1, &mut s2b_loss);
    let _ = run_lbfgs_matryoshka_with_backend::<B>(&mut model, &ds_s2b, 8);

    // Stage 3 (100% dataset): full-length OMP + collision repair + exact f32 core trajectory polish.
    let ds_full = dataset.with_spectral_weights(0.0, 0.0, 0.05);
    select_omp_kautz_poles_matryoshka_with_backend::<B>(&mut model, &ds_full);
    repair_collided_kautz_poles(&mut model, dataset.sample_rate_hz);
    polish_f32_matryoshka_readouts_with_backend::<B>(&mut model, &ds_full);

    model.to_matryoshka_cold()
}

/// Dispatches [`fit_v3_matryoshka_with_backend`] via fastest CPU backend.
#[must_use]
pub fn fit_v3_matryoshka(dataset: &V3TrainDataset, mono_out: bool) -> V3MatryoshkaModel {
    struct FitTask<'a>(&'a V3TrainDataset, bool);
    impl TrainerTask<V3MatryoshkaModel> for FitTask<'_> {
        fn run<B: TrainerBackend>(self) -> V3MatryoshkaModel {
            fit_v3_matryoshka_with_backend::<B>(self.0, self.1)
        }
    }
    DynamicTrainer::detect().dispatch(FitTask(dataset, mono_out))
}

/// Backend-parameterized compatibility wrapper returning `[Lite, Full, Pro, Ultra]` cold models from [`fit_v3_matryoshka_with_backend`].
#[must_use]
pub fn fit_v3_core_four_tiers_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
) -> [V3ColdTierModel; 4] {
    fit_v3_matryoshka_with_backend::<B>(dataset, mono_out).to_four_tiers()
}

/// Dispatches [`fit_v3_core_four_tiers_with_backend`] via fastest CPU backend.
#[must_use]
pub fn fit_v3_core_four_tiers(dataset: &V3TrainDataset, mono_out: bool) -> [V3ColdTierModel; 4] {
    fit_v3_matryoshka(dataset, mono_out).to_four_tiers()
}

pub(super) fn live_switch_tier_safe<B: TrainerBackend>(
    cold: &V3MatryoshkaModel,
    stem: &V3TrainStem,
    max_tau: usize,
) -> bool {
    let n = stem.usable_len();
    if n < 8 {
        return true;
    }
    let peak_ref = stem.tgt_l[..n]
        .iter()
        .chain(&stem.tgt_r[..n])
        .fold(
            1.0_f32,
            |a, &v| if v.is_finite() { a.max(v.abs()) } else { a },
        );
    let bound = (peak_ref * 2.5).max(8.0);
    let mut inst = V3CoreInstance::<B>::from_cold_model(&cold.extract_tier(Tier::Lite));
    inst.reset_to_dc_equilibrium();
    let seg = n / (max_tau + 1);
    for t_idx in 0..=max_tau {
        inst.switch_tier(&cold.extract_tier(ALL_TIERS[t_idx]));
        for t in (t_idx * seg)..((t_idx + 1) * seg) {
            let xr = if cold.mono_out {
                stem.in_l[t]
            } else {
                stem.in_r[t]
            };
            let [yl, yr] = inst.step_stereo(stem.in_l[t], xr);
            if !yl.is_finite() || !yr.is_finite() || yl.abs() > bound || yr.abs() > bound {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use dnam_engine::Tier;
    use dnam_engine::math::DefaultMathBackend;
    use proptest::prelude::*;

    use super::*;
    use crate::metrics::{
        canonical_ltas_33_centers_hz, evaluate_besr_from_band_energies,
        split_bezout_3band_energies, verify_monotone_4tier_summaries,
    };
    use crate::solver::backends::DefaultTrainerBackend;

    fn make_synthetic_stem(n: usize, warmup: usize) -> V3TrainStem {
        let centers = canonical_ltas_33_centers_hz();
        let scale = 0.42 / (centers.len() as f64).sqrt();
        let mut x = vec![0.0_f32; n];
        for (t, v) in x.iter_mut().enumerate() {
            let tf = t as f64 / 48_000.0;
            let env = 0.45 + 0.55 * (0.5 - 0.5 * (2.0 * PI * 3.0 * tf).cos());
            let mut sum = 0.0_f64;
            for (k, &fc) in centers.iter().enumerate() {
                let phi = ((k as f64 + 1.0) * std::f64::consts::GOLDEN_RATIO).fract() * 2.0 * PI;
                sum += (2.0 * PI * fc * tf + phi).cos();
            }
            *v = (env * scale * sum) as f32;
        }
        let mut tgt = vec![0.0_f32; n];
        let mut lp = 0.0_f32;
        for t in 0..n {
            lp = 0.72 * lp + 0.28 * x[t];
            let d = (2.8 * lp + 0.22 * x[t] - 0.18).tanh();
            tgt[t] = 0.65 * d + if t >= 2 { 0.15 * x[t - 2] } else { 0.0 };
        }
        V3TrainStem {
            in_l: x.clone(),
            in_r: x,
            tgt_l: tgt.clone(),
            tgt_r: tgt,
            warmup,
        }
    }

    fn model_besr_db(cold: &V3ColdTierModel, stem: &V3TrainStem) -> f64 {
        let n = stem.in_l.len();
        let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(cold);
        let (mut ol, mut or_) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(&stem.in_l, &stem.in_r, &mut ol, &mut or_);
        let ol64: Vec<f64> = ol.iter().map(|&v| f64::from(v)).collect();
        let tgt64: Vec<f64> = stem.tgt_l.iter().map(|&v| f64::from(v)).collect();
        let (sig, err, noise) =
            split_bezout_3band_energies(&tgt64, &ol64, None, 48_000.0, stem.warmup);
        evaluate_besr_from_band_energies(sig, err, noise).besr_db
    }

    #[test]
    fn test_fit_v3_matryoshka_monotone_nonlinear_and_live_switch() {
        let stem = make_synthetic_stem(4_800, 128);
        let ds = V3TrainDataset::new(vec![stem.clone()], 48_000.0);
        let matryoshka = fit_v3_matryoshka(&ds, true);
        assert!(matryoshka.os2x);
        assert!(matryoshka.mono_out);

        let models = matryoshka.to_four_tiers();
        assert_eq!(models[0].tier, Tier::Lite);
        assert_eq!(models[1].tier, Tier::Full);
        assert_eq!(models[2].tier, Tier::Pro);
        assert_eq!(models[3].tier, Tier::Ultra);

        let b: Vec<f64> = models.iter().map(|m| model_besr_db(m, &stem)).collect();
        for w in b.windows(2) {
            assert!(w[1] <= w[0] + 1.0e-9, "non-monotone 4-tier bESR: {b:?}");
        }
        assert!(
            b[3] < -22.0,
            "expected strong synthetic nonlinear fit, got {b:?}"
        );

        let sums: [PerceptualCertificationSummary; 4] = models
            .iter()
            .map(|m| {
                signal_chain_summary_for_cold::<DefaultTrainerBackend>(m, &stem, 48_000.0).unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        assert!(verify_monotone_4tier_summaries(&sums));

        // Verify live single-instance switch_tier across all 4 tiers without glitches or NaN.
        let mut live = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&models[0]);
        live.reset_to_dc_equilibrium();
        let n_seg = stem.in_l.len() / 4;
        for (tau, tier_cold) in models.iter().enumerate() {
            live.switch_tier(tier_cold);
            let start = tau * n_seg;
            let end = start + n_seg;
            let (mut ol, mut or_) = (vec![0.0_f32; n_seg], vec![0.0_f32; n_seg]);
            live.process_block_stereo(
                &stem.in_l[start..end],
                &stem.in_l[start..end],
                &mut ol,
                &mut or_,
            );
            assert!(ol.iter().all(|v| v.is_finite()));
            assert_eq!(ol, or_);
        }
    }

    #[test]
    fn test_evaluate_cold_loss_f32_empty_and_identity() {
        let stem = make_synthetic_stem(512, 64);
        let ds = V3TrainDataset::new(vec![stem], 48_000.0).with_spectral_weights(0.04, 0.04, 0.06);
        let empty = V3TrainDataset::new(vec![], 48_000.0);
        let empty_fit = fit_v3_matryoshka(&empty, false);
        assert!(empty_fit.os2x);
        assert!(!empty_fit.mono_out);

        let id_cold = empty_fit.extract_tier(Tier::Lite);
        let loss = evaluate_cold_loss_f32(&id_cold, &ds);
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_v3_stereo_summary_core_wrappers_and_switch_safety() {
        let mut stereo_stem = make_synthetic_stem(4_800, 128);
        for (i, sample) in stereo_stem.in_r.iter_mut().enumerate() {
            *sample *= if i % 2 == 0 { 0.9 } else { 0.85 };
        }
        for (i, sample) in stereo_stem.tgt_r.iter_mut().enumerate() {
            *sample = 0.8 * stereo_stem.tgt_l[i] + 0.1 * stereo_stem.in_r[i];
        }
        let empty_stem = V3TrainStem {
            in_l: Vec::new(),
            in_r: Vec::new(),
            tgt_l: Vec::new(),
            tgt_r: Vec::new(),
            warmup: 0,
        };
        let ds_stereo =
            V3TrainDataset::new(vec![empty_stem.clone(), stereo_stem.clone()], 48_000.0)
                .with_spectral_weights(0.02, 0.02, 0.02);

        let mut stereo_model = V3MatryoshkaModel::identity(true, false);
        let lite_cold = stereo_model.extract_tier(Tier::Lite);
        let loss_stereo = evaluate_cold_loss_f32(&lite_cold, &ds_stereo);
        assert!(loss_stereo.is_finite() && loss_stereo > 0.0);

        // Short stem returns None; valid stereo stem combines L and R summaries
        assert!(
            signal_chain_summary_for_cold::<DefaultTrainerBackend>(
                &lite_cold,
                &empty_stem,
                48_000.0
            )
            .is_none()
        );
        let summary_stereo = signal_chain_summary_for_cold::<DefaultTrainerBackend>(
            &lite_cold,
            &stereo_stem,
            48_000.0,
        )
        .expect("stereo summary");
        assert!(summary_stereo.raw_besr_db.is_finite());

        // Live tier switch safety: short stem, valid stereo stem, and unstable gain guard
        let short_stem = make_synthetic_stem(4, 0);
        assert!(live_switch_tier_safe::<DefaultTrainerBackend>(
            &stereo_model,
            &short_stem,
            3
        ));
        assert!(live_switch_tier_safe::<DefaultTrainerBackend>(
            &stereo_model,
            &stereo_stem,
            3
        ));
        stereo_model.readouts[1].svf_direct_taps_r[0] = 1.0e6;
        assert!(!live_switch_tier_safe::<DefaultTrainerBackend>(
            &stereo_model,
            &stereo_stem,
            1
        ));

        // Compatibility 4-tier wrappers on empty dataset
        let empty_ds = V3TrainDataset::new(vec![], 48_000.0);
        let four_direct = fit_v3_core_four_tiers(&empty_ds, true);
        let four_backend =
            fit_v3_core_four_tiers_with_backend::<DefaultTrainerBackend>(&empty_ds, false);
        assert_eq!(four_direct.len(), 4);
        assert!(!four_backend[0].mono_out);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]
        #[test]
        fn prop_matryoshka_live_tier_switch_finite(
            freq_hz in 60.0_f64..1_200.0_f64,
            amp in 0.05_f32..0.45_f32,
            mono in any::<bool>(),
        ) {
            let n = 640_usize;
            let x: Vec<f32> = (0..n)
                .map(|i| amp * (2.0 * PI * freq_hz * (i as f64) / 48_000.0).sin() as f32)
                .collect();
            let mat = V3MatryoshkaModel::identity(true, mono);
            let tiers = mat.to_four_tiers();
            let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&tiers[0]);
            inst.reset_to_dc_equilibrium();
            for (tau, cold) in tiers.iter().enumerate() {
                inst.switch_tier(cold);
                let s = tau * 128;
                let e = s + 128;
                let (mut ol, mut or_) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
                inst.process_block_stereo(&x[s..e], &x[s..e], &mut ol, &mut or_);
                prop_assert!(ol.iter().all(|v| v.is_finite()));
                prop_assert!(or_.iter().all(|v| v.is_finite()));
                if mono {
                    prop_assert_eq!(ol, or_);
                }
            }
        }
    }
}
