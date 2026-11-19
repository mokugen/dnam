#![forbid(unsafe_code)]
//! `v3` Single-Run 4-Tier Matryoshka (`2x OS`) VarPro + Adjoint L-BFGS + Progressive-Slot OMP Solver.

pub mod adjoint;
pub mod alignment;
pub mod basis;
pub mod bezout7;
mod cold_eval;
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
pub mod stage0_id;
pub mod stage4_id;
pub mod svf_id;

pub use alignment::{build_v3_causal_probe_excitation, estimate_v3_causal_deadtime_and_polarity};
pub use cold_eval::{evaluate_cold_loss_f32, evaluate_cold_loss_f32_with_backend};
pub use lbfgs::{
    run_lbfgs_core_matryoshka, run_lbfgs_core_matryoshka_with_backend, run_lbfgs_matryoshka,
    run_lbfgs_matryoshka_with_backend,
};
pub use model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset, V3TrainStem};
pub use omp::{
    repair_collided_kautz_poles, select_omp_kautz_poles_matryoshka,
    select_omp_kautz_poles_matryoshka_with_backend,
};
pub use polish::{
    polish_f32_matryoshka_readouts, polish_f32_matryoshka_readouts_with_backend,
    polish_f32_matryoshka_readouts_with_cert_stem,
};

use dnam_engine::Tier;
use dnam_engine::stages::{V3ColdTierModel, V3MatryoshkaModel};

#[cfg(test)]
use crate::metrics::{
    ArchetypeFamily, PerceptualCertificationSummary, certify_4metric_signal_chain,
};
#[cfg(test)]
use dnam_engine::stages::V3CoreInstance;

use super::SolverQualityMode;
use super::backends::{DynamicTrainer, TrainerBackend, TrainerTask};
use super::progress::telemetry::{OptStepSpec, VisualProbeAnchor, emit_dataset_opt_step};
use super::progress::{TrainerProgressSnapshot, TrainingPhase, TrainingStatus};
use crate::stimulus::sealing::CORESET_A5_FRAMES_48K;
use adjoint::{AdjointWorkspace, evaluate_matryoshka_loss_f64_with_backend};
use basis::solve_matryoshka_varpro_cholesky_with_backend;
use lbfgs::run_lbfgs_with_ws;
use primitives::par_map_vec;
use svf_id::prescreen_matryoshka_seeds_with_backend;

/// Subwindow cap (`12,288` samples) covering the full `12,000`-sample `CORESET_A5` (`Q1..Q4` incl. `17.5 kHz` chirp).
const FAST_FIT_SUBWINDOW_SAMPLES: usize = 12_288;
const _: () = assert!(FAST_FIT_SUBWINDOW_SAMPLES >= CORESET_A5_FRAMES_48K);

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
    let keep_in_r = !cold.mono_out || cold.feat_params.key_link > 0.0;
    let r_in = if !keep_in_r {
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

#[inline(never)]
fn accept_varpro_if_improves<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    als_iters: usize,
    best_loss: &mut f64,
) {
    let backup = Box::new(model.clone());
    solve_matryoshka_varpro_cholesky_with_backend::<B>(model, dataset, als_iters);
    let after = evaluate_matryoshka_loss_f64_with_backend::<B>(model, dataset, ws);
    if after.is_finite() && after <= *best_loss {
        *best_loss = after;
    } else {
        model.clone_from(&backup);
    }
}

#[inline(never)]
fn accept_omp_if_improves<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    best_loss: &mut f64,
) {
    let backup = Box::new(model.clone());
    select_omp_kautz_poles_matryoshka_with_backend::<B>(model, dataset);
    let after = evaluate_matryoshka_loss_f64_with_backend::<B>(model, dataset, ws);
    if after.is_finite() && after <= *best_loss {
        *best_loss = after;
    } else {
        model.clone_from(&backup);
    }
}

/// Backend-parameterized single-run 4-tier Matryoshka (`2x OS`) fitter (`VarPro` + L-BFGS + OMP + `f32` polish),
/// augmented with signal-driven `Stage 0` sub-octave identification and `Stage IV` reflection/FDN refinement.
#[inline(never)]
#[must_use]
pub fn fit_v3_matryoshka_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
) -> V3MatryoshkaModel {
    if dataset.stems.is_empty() || dataset.stems.iter().all(|s| s.usable_len() == 0) {
        return V3MatryoshkaModel::dual_rate_default(mono_out);
    }
    let ds_core = dataset.with_progress(dataset.progress.subrange(0, 800));
    let after_s0 = if stage4_id::dataset_is_incoherent_candidate(&ds_core)
        && !stage0_id::probe_stage0_suboctave(&ds_core)
    {
        fit_v3_matryoshka_fast_with_backend::<B>(&ds_core, mono_out)
    } else {
        stage0_id::fit_stage0_first_with_backend::<B>(
            &ds_core,
            mono_out,
            fit_v3_matryoshka_core::<B>,
        )
        .unwrap_or_else(|| fit_v3_matryoshka_core::<B>(&ds_core, mono_out))
    };
    let ds_s4 = dataset.with_progress(dataset.progress.subrange(800, 990));
    let fitted = stage4_id::refine_stage4_matryoshka_with_backend::<B>(after_s0, &ds_s4, mono_out);
    if dataset.progress.is_active() {
        dataset.progress.emit(&TrainerProgressSnapshot {
            status: TrainingStatus::Completed,
            phase: TrainingPhase::Completed,
            active_tier: Some(Tier::Ultra),
            progress_permille: 1_000,
            ..Default::default()
        });
    }
    fitted
}

#[inline]
fn has_slow_envelope_or_sag(m: &V3MatryoshkaOptF64) -> bool {
    let slow_opt = m.feat_det_thresh > 0.0 && m.feat_slow_rel_beta < 8.0e-5 && m.feat_pinch_k < 0.25;
    let slow_sag = m.feat_slow_rel_beta < 5.0e-5
        && m.blocks
            .iter()
            .any(|b| b.k_sag[..4].iter().any(|&k| k > 0.25));
    slow_opt || slow_sag
}

#[inline]
fn has_stage1_crossover_split(m: &V3MatryoshkaOptF64) -> bool {
    m.svf_feat_lp2_mix > 0.9 && m.svf_env_oct_sens.abs() < 0.25
}

#[inline(never)]
fn fit_v3_matryoshka_core<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
) -> V3MatryoshkaModel {
    if dataset.stems.is_empty() || dataset.stems.iter().all(|s| s.usable_len() == 0) {
        return V3MatryoshkaModel::dual_rate_default(mono_out);
    }
    let anchor = VisualProbeAnchor::from_dataset(dataset);

    // Prescreen 16 circuit templates across 4 archetypes in Lite (G=1, 4k -> 5k Gabor homotopy), keep top 2 (G=1).
    let mut winners: Vec<Box<V3MatryoshkaOptF64>> =
        prescreen_matryoshka_seeds_with_backend::<B>(dataset, mono_out, 2)
            .into_iter()
            .map(Box::new)
            .collect();
    if winners.is_empty() {
        winners.push(Box::new(V3MatryoshkaOptF64::from_matryoshka_cold(
            &V3MatryoshkaModel::identity(true, mono_out),
        )));
    }
    let step_meta = |permille, step, tier, loss| OptStepSpec {
        phase: TrainingPhase::StageBBackbone,
        permille,
        step,
        total_steps: 6,
        tier,
        loss,
    };
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &winners[0],
        step_meta(100, 1, Tier::Lite, f64::NAN),
    );

    // Stage 1a (G=1 at 2x OS, 12k or 32k for slow optical / slow B+ sag): lock phase with L-BFGS first, then OMP+VarPro+L-BFGS and prune 2 -> 1.
    let s1a_len = if winners
        .iter()
        .any(|w| w.feat_key_link > 0.5 || has_slow_envelope_or_sag(w))
    {
        32_768
    } else {
        12_288
    };
    let ds_s1a = dataset
        .primary_subwindow_opts(s1a_len, true, false)
        .with_spectral_weights(0.06, 0.03, 0.0);
    let (s1a_iters, s1b_bonus, s2a_bonus, s2b_bonus, fast_cap) = match dataset.quality_mode {
        SolverQualityMode::BestUntilDiminishingReturns => {
            (11_usize, 4_usize, 4_usize, 4_usize, usize::MAX)
        }
        SolverQualityMode::ThresholdStop {
            max_polish_iters, ..
        } => {
            let cap = max_polish_iters.max(4);
            (7_usize.min(cap), 0, 0, 0, cap)
        }
    };
    let s1a_cands = par_map_vec(winners, |_, mut cand| {
        cand.tier_os2x = [true; 4];
        let mut cws = AdjointWorkspace::default();
        let mut cand_loss =
            run_lbfgs_with_ws::<B>(&mut cand, &ds_s1a, &mut cws, s1a_iters, true, true, true);
        if !cand_loss.is_finite() {
            cand_loss = f64::INFINITY;
        }
        accept_omp_if_improves::<B>(&mut cand, &ds_s1a, &mut cws, &mut cand_loss);
        accept_varpro_if_improves::<B>(&mut cand, &ds_s1a, &mut cws, 1, &mut cand_loss);
        let mut s1a_loss =
            run_lbfgs_with_ws::<B>(&mut cand, &ds_s1a, &mut cws, 2, true, true, true);
        if !s1a_loss.is_finite() {
            s1a_loss = f64::INFINITY;
        }
        accept_varpro_if_improves::<B>(&mut cand, &ds_s1a, &mut cws, 1, &mut s1a_loss);
        (s1a_loss, cand)
    });
    let mut best_model: Option<Box<V3MatryoshkaOptF64>> = None;
    let mut best_loss = f64::INFINITY;
    for (s1a_loss, cand) in s1a_cands {
        if s1a_loss.is_finite() && s1a_loss < best_loss {
            best_loss = s1a_loss;
            best_model = Some(cand);
        }
    }
    let mut ws = AdjointWorkspace::default();
    let mut model = best_model.unwrap_or_else(|| {
        Box::new(V3MatryoshkaOptF64::from_matryoshka_cold(
            &V3MatryoshkaModel::identity(true, mono_out),
        ))
    });
    model.os2x = true;
    model.tier_os2x = [true; 4];
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        step_meta(275, 2, Tier::Lite, best_loss),
    );

    // Stage 1b (G=1 at 2x OS contiguous sag & RC bias lock).
    let sweeping_svf = model.svf_env_oct_sens.abs() > 0.5;
    let slow_long = has_slow_envelope_or_sag(&model);
    let s1b_len = if slow_long {
        49_152
    } else if sweeping_svf {
        18_432
    } else {
        14_336
    };
    let ds_s1b = dataset
        .primary_subwindow_opts(s1b_len, false, false)
        .with_spectral_weights(0.0, 0.0, 0.0);
    let mut s1b_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_s1b, &mut ws);
    if !s1b_loss.is_finite() {
        s1b_loss = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s1b, &mut ws, 1, &mut s1b_loss);
    let s1b_iters = (if slow_long || sweeping_svf { 11 } else { 8 } + s1b_bonus).min(fast_cap);
    let mut s1b_after =
        run_lbfgs_with_ws::<B>(&mut model, &ds_s1b, &mut ws, s1b_iters, true, true, true);
    if !s1b_after.is_finite() {
        s1b_after = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s1b, &mut ws, 1, &mut s1b_after);
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        step_meta(450, 3, Tier::Lite, s1b_after),
    );

    // Stage 2a (pure 2x OS tapered unlock, [Lite, Pro, Ultra]): OMP + parallel co-adaptation on contiguous subwindow.
    svf_id::unlock_matryoshka_groups_after_prescreen(&mut model);
    model.tier_os2x = [true; 4];
    model.active_tier_weights = [0.22, 0.0, 0.33, 0.45];
    let sweeping_svf = model.svf_env_oct_sens.abs() > 0.5;
    let crossover_split = has_stage1_crossover_split(&model);
    let s2a_len = if slow_long {
        32_768
    } else if crossover_split {
        16_384
    } else if sweeping_svf {
        12_288
    } else {
        10_240
    };
    let ds_s2a = dataset
        .primary_subwindow_opts(s2a_len, false, false)
        .with_spectral_weights(0.0, 0.0, 0.0);
    select_omp_kautz_poles_matryoshka_with_backend::<B>(&mut model, &ds_s2a);
    let mut s2a_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_s2a, &mut ws);
    if !s2a_loss.is_finite() {
        s2a_loss = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s2a, &mut ws, 1, &mut s2a_loss);
    let s2a_iters =
        (if sweeping_svf || crossover_split { 10 } else { 7 } + s2a_bonus).min(fast_cap);
    let mut s2a_after =
        run_lbfgs_with_ws::<B>(&mut model, &ds_s2a, &mut ws, s2a_iters, true, true, true);
    if !s2a_after.is_finite() {
        s2a_after = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s2a, &mut ws, 1, &mut s2a_after);
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        step_meta(625, 4, Tier::Pro, s2a_after),
    );

    // Stage 2b (pure 2x OS contiguous refinement across primary + secondary subwindows).
    model.active_tier_weights = [0.18, 0.12, 0.32, 0.38];
    let (s2b_p, s2b_s) = if slow_long {
        (65_536, 20_480)
    } else if crossover_split {
        (24_576, 14_336)
    } else if sweeping_svf {
        (18_432, 14_336)
    } else {
        (14_336, 10_240)
    };
    let ds_s2b = dataset
        .curriculum_subwindow_opts(s2b_p, s2b_s, false, false)
        .with_spectral_weights(0.0, 0.0, 0.0);
    let mut s2b_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_s2b, &mut ws);
    if !s2b_loss.is_finite() {
        s2b_loss = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s2b, &mut ws, 1, &mut s2b_loss);
    let s2b_iters = (if slow_long
        || crossover_split
        || sweeping_svf
        || model.feat_pinch_k > 80.0
    {
        11
    } else {
        7
    } + s2b_bonus)
        .min(fast_cap);
    let mut s2b_after =
        run_lbfgs_with_ws::<B>(&mut model, &ds_s2b, &mut ws, s2b_iters, true, true, true);
    if !s2b_after.is_finite() {
        s2b_after = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_s2b, &mut ws, 1, &mut s2b_after);
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        step_meta(800, 5, Tier::Ultra, s2b_after),
    );

    // Stage 3: 2x OMP + collision repair, dedicated contiguous 1x calibration (S2c_1x), and aligned f32 polish.
    model.active_tier_weights = [0.18, 0.20, 0.26, 0.36];
    select_omp_kautz_poles_matryoshka_with_backend::<B>(&mut model, dataset);
    repair_collided_kautz_poles(&mut model, dataset.sample_rate_hz);
    let slow_long = has_slow_envelope_or_sag(&model);
    let crossover_split = has_stage1_crossover_split(&model);
    let ds_1x = if slow_long {
        dataset
            .curriculum_subwindow_opts(36_864, 16_384, false, false)
            .with_spectral_weights(0.0, 0.0, 0.0)
    } else if crossover_split {
        dataset
            .primary_subwindow_opts(16_384, false, false)
            .with_spectral_weights(0.0, 0.0, 0.0)
    } else {
        dataset
            .primary_subwindow_opts(10_240, false, false)
            .with_spectral_weights(0.0, 0.0, 0.0)
    };
    adjoint::calibrate_1x_tiers_with_backend::<B>(&mut model, &ds_1x, &mut ws);
    let ds_polish = dataset
        .for_stage3_polish(72_000)
        .with_spectral_weights(0.0, 0.0, 0.05);
    svf_id::apply_pre_polish_overdrive_recovery(&mut model);
    polish::polish_f32_matryoshka_readouts_with_cert_stem::<B>(
        &mut model,
        &ds_polish,
        dataset.full_cert_stem.as_ref().or_else(|| dataset.stems.first()),
    );
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        step_meta(975, 6, Tier::Ultra, f64::NAN),
    );

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

/// Fast-path `v3` Matryoshka (`2x OS`) fitter for short unit-test captures (`usable_len < 48_000`).
#[inline(never)]
#[must_use]
pub fn fit_v3_matryoshka_fast_with_backend<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    mono_out: bool,
) -> V3MatryoshkaModel {
    if dataset.stems.is_empty() || dataset.stems.iter().all(|s| s.usable_len() == 0) {
        return V3MatryoshkaModel::dual_rate_default(mono_out);
    }
    let anchor = VisualProbeAnchor::from_dataset(dataset);
    let fast_meta = |permille, step, tier, loss| OptStepSpec {
        phase: TrainingPhase::StageBBackbone,
        permille,
        step,
        total_steps: 4,
        tier,
        loss,
    };
    let mut model = Box::new(
        prescreen_matryoshka_seeds_with_backend::<B>(dataset, mono_out, 1)
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(
                    true, mono_out,
                ))
            }),
    );
    svf_id::unlock_matryoshka_groups_after_prescreen(&mut model);
    model.os2x = true;
    model.tier_os2x = [true; 4];
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        fast_meta(120, 1, Tier::Lite, f64::NAN),
    );
    model.active_tier_weights = [0.22, 0.22, 0.26, 0.30];

    let ds_lbfgs = dataset
        .primary_subwindow_opts(FAST_FIT_SUBWINDOW_SAMPLES, false, false)
        .with_spectral_weights(0.0, 0.0, 0.0);
    let is_incoh = stage4_id::dataset_is_incoherent_candidate(dataset);
    let ds_fast_full = if is_incoh {
        dataset
            .primary_subwindow_opts(FAST_FIT_SUBWINDOW_SAMPLES, false, false)
            .with_spectral_weights(0.0, 0.0, 0.0)
    } else {
        dataset.clone().with_spectral_weights(0.0, 0.0, 0.0)
    };
    let mut ws = AdjointWorkspace::default();
    select_omp_kautz_poles_matryoshka_with_backend::<B>(&mut model, &ds_fast_full);
    let mut init_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&model, &ds_lbfgs, &mut ws);
    if !init_loss.is_finite() {
        init_loss = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_fast_full, &mut ws, 1, &mut init_loss);
    let mut loss = run_lbfgs_with_ws::<B>(&mut model, &ds_lbfgs, &mut ws, 6, true, true, true);
    if !loss.is_finite() {
        loss = f64::INFINITY;
    }
    accept_varpro_if_improves::<B>(&mut model, &ds_fast_full, &mut ws, 1, &mut loss);
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        fast_meta(500, 2, Tier::Ultra, loss),
    );
    repair_collided_kautz_poles(&mut model, dataset.sample_rate_hz);
    solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut model, &ds_fast_full, 1);
    let ds_1x = dataset
        .primary_subwindow_opts(FAST_FIT_SUBWINDOW_SAMPLES, false, false)
        .with_spectral_weights(0.0, 0.0, 0.0);
    adjoint::calibrate_1x_tiers_with_backend::<B>(&mut model, &ds_1x, &mut ws);
    svf_id::apply_pre_polish_overdrive_recovery(&mut model);
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        fast_meta(780, 3, Tier::Full, f64::NAN),
    );
    if is_incoh {
        solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut model, &ds_fast_full, 1);
    } else {
        polish::polish_f32_matryoshka_readouts_with_cert_stem::<B>(
            &mut model,
            &ds_fast_full,
            dataset.stems.first(),
        );
    }
    emit_dataset_opt_step::<B>(
        dataset,
        anchor.as_ref(),
        &model,
        fast_meta(975, 4, Tier::Ultra, f64::NAN),
    );
    model.to_matryoshka_cold()
}

/// Dispatches [`fit_v3_matryoshka_fast_with_backend`] via fastest CPU backend.
#[must_use]
pub fn fit_v3_matryoshka_fast(dataset: &V3TrainDataset, mono_out: bool) -> V3MatryoshkaModel {
    struct FastFitTask<'a>(&'a V3TrainDataset, bool);
    impl TrainerTask<V3MatryoshkaModel> for FastFitTask<'_> {
        fn run<B: TrainerBackend>(self) -> V3MatryoshkaModel {
            fit_v3_matryoshka_fast_with_backend::<B>(self.0, self.1)
        }
    }
    DynamicTrainer::detect().dispatch(FastFitTask(dataset, mono_out))
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
        let stem = make_synthetic_stem(2_400, 128);
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
        assert!(
            verify_monotone_4tier_summaries(&sums),
            "non-monotone summaries: {sums:#?}"
        );

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
        let mut stereo_stem = make_synthetic_stem(2_048, 128);
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
        assert!(polish::live_switch_tier_safe::<DefaultTrainerBackend>(
            &stereo_model,
            &short_stem,
            3
        ));
        assert!(polish::live_switch_tier_safe::<DefaultTrainerBackend>(
            &stereo_model,
            &stereo_stem,
            3
        ));
        stereo_model.readouts[1].svf_direct_taps_r[0] = 1.0e6;
        assert!(!polish::live_switch_tier_safe::<DefaultTrainerBackend>(
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

    #[test]
    fn test_fit_v3_matryoshka_emits_hybrid_progress_visual_frames() {
        use crate::solver::{AtomicTrainerProgress, TrainerProgressHandle};
        use std::sync::{Arc, Mutex};

        let stem = make_synthetic_stem(4_096, 128);
        let slot = Arc::new(AtomicTrainerProgress::new());
        let events: Arc<Mutex<Vec<TrainerProgressSnapshot>>> = Arc::new(Mutex::new(Vec::new()));
        let ev_cb = Arc::clone(&events);
        let handle = TrainerProgressHandle::with_slot_and_callback(Arc::clone(&slot), move |s| {
            ev_cb.lock().unwrap().push(s.clone());
        });
        let ds = V3TrainDataset::new(vec![stem], 48_000.0).with_progress(handle);
        let _fitted = fit_v3_matryoshka(&ds, true);
        let done = slot.poll();
        assert_eq!(done.status, TrainingStatus::Completed);
        assert_eq!(done.progress_permille, 1_000);
        assert!(done.has_visual_frame);
        assert!(done.tone_match_percent().is_some_and(|pct| pct.is_finite()));
        assert!(events.lock().unwrap().len() >= 4);
    }
}
