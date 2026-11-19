#![forbid(unsafe_code)]
//! Candidate screening, linear diffuse FDN grid, and Welch-33 pre-polish selection for `incoherent_rig`.

use std::f64::consts::FRAC_PI_4;

use dnam_engine::Tier;
use dnam_engine::stages::V3MatryoshkaModel;
use dnam_engine::stages::stage2_gru::MAX_TOTAL_GRU_LANES;

use crate::metrics::certification::PerceptualCertificationSummary;
use crate::metrics::companion::LtasTargetCache;
use crate::solver::SolverQualityMode;
use crate::solver::backends::TrainerBackend;

use super::super::super::model_f64::V3TrainStem;
use super::super::super::primitives::par_map_indexed;
use super::super::render_tier_stereo;
use super::cascade::summary_passes_mode;
use super::cepstral_eq::{EqFreqGrid, PrecomputedTierEq};
use super::comb_detect::{
    FdnTankSpec, PRIME_QUARTETS, calibrate_stage4_rms, clamp_lfo_depth_for_delays,
};
use super::harmonic_cal::SweepHarmonicCheck;
use super::s2_modal;
use super::wave_hinge::WaveTargetCtx;
use super::welch_eq::{WelchPolishSpec, polish_tier_welch33};
use super::{CertCachePair, WelchSummaryPair, eval_tier_cert};

pub(super) const DIFFUSE_TAPS: [f32; 4] = [0.65, 0.65, -0.55, 0.55];

#[derive(Clone)]
pub(super) struct ClosedLoopEqCtx<'a> {
    pub(super) stem: &'a V3TrainStem,
    pub(super) warmup: usize,
    pub(super) sr_hz: f64,
    pub(super) grid: &'a EqFreqGrid,
    pub(super) ltas_l: LtasTargetCache,
    pub(super) ltas_r: Option<LtasTargetCache>,
}

pub(super) struct EqRunSpec<'a> {
    pub(super) tier: Tier,
    pub(super) target_rms: f64,
    pub(super) tilt_db: &'a [[f64; 33]; 2],
    pub(super) passes: usize,
    pub(super) step_gain: f64,
}

impl ClosedLoopEqCtx<'_> {
    pub(super) fn run_tier<B: TrainerBackend>(
        &mut self,
        model: &mut V3MatryoshkaModel,
        spec: &EqRunSpec<'_>,
        init_db: &mut [[f64; 33]; 2],
    ) {
        let tier = spec.tier;
        let t = tier as usize;
        let n = self.stem.usable_len();
        let mono = model.mono_out;
        let sr = self.sr_hz;
        let grid = self.grid;
        let stem = self.stem;
        let warmup = self.warmup;
        let target_rms = spec.target_rms;
        let use_s2 = s2_modal::has_1x_modal_basis(model, tier);
        let pre_eq = PrecomputedTierEq::new(&model.kautz_poles, tier, use_s2, sr, grid);
        let apply_eq_and_rms = |mdl: &mut V3MatryoshkaModel, db: &[[f64; 33]; 2]| {
            if use_s2 {
                mdl.readouts[t].direct_s2_w = [[0.0; MAX_TOTAL_GRU_LANES]; 2];
            }
            if let Some(eq) = pre_eq.as_ref() {
                if let Some(sol_l) = eq.solve(&db[0], sr, grid) {
                    mdl.readouts[t].fir_l = sol_l.fir;
                    mdl.readouts[t].kautz_c_l = sol_l.kautz_c;
                    if use_s2 {
                        s2_modal::write_direct_s2_weights(
                            &sol_l.direct_s2,
                            0,
                            &mut mdl.readouts[t],
                        );
                    }
                }
                if !mono {
                    if let Some(sol_r) = eq.solve(&db[1], sr, grid) {
                        mdl.readouts[t].fir_r = sol_r.fir;
                        mdl.readouts[t].kautz_c_r = sol_r.kautz_c;
                        if use_s2 {
                            s2_modal::write_direct_s2_weights(
                                &sol_r.direct_s2,
                                1,
                                &mut mdl.readouts[t],
                            );
                        }
                    }
                } else {
                    mdl.readouts[t].fir_r = mdl.readouts[t].fir_l;
                    mdl.readouts[t].kautz_c_r = mdl.readouts[t].kautz_c_l;
                    if use_s2 {
                        mdl.readouts[t].direct_s2_w[1] = mdl.readouts[t].direct_s2_w[0];
                    }
                }
            }
            mdl.readouts[t].mask_to_tier(tier);
            let _ = calibrate_stage4_rms::<B>(mdl, tier, stem, warmup, target_rms, mono);
        };
        for _ in 0..spec.passes {
            apply_eq_and_rms(model, init_db);
            let cold = model.extract_tier(tier);
            let (ol, or) = render_tier_stereo::<B>(&cold, &stem.in_l[..n], &stem.in_r[..n]);
            let ml: Vec<f64> = ol[warmup..n].iter().map(|&v| f64::from(v)).collect();
            if let Some((py, pm)) = self.ltas_l.band_powers(&ml) {
                for b in 0..33 {
                    let ratio = (py[b].max(1.0e-24) / pm[b].max(1.0e-24)).log10();
                    init_db[0][b] = (init_db[0][b]
                        + spec.step_gain * (10.0 * ratio + spec.tilt_db[0][b]))
                        .clamp(-30.0, 30.0);
                }
            }
            if let Some(lr) = self.ltas_r.as_mut() {
                let mr: Vec<f64> = or[warmup..n].iter().map(|&v| f64::from(v)).collect();
                if let Some((py, pm)) = lr.band_powers(&mr) {
                    for b in 0..33 {
                        let ratio = (py[b].max(1.0e-24) / pm[b].max(1.0e-24)).log10();
                        init_db[1][b] = (init_db[1][b]
                            + spec.step_gain * (10.0 * ratio + spec.tilt_db[1][b]))
                            .clamp(-30.0, 30.0);
                    }
                }
            } else {
                init_db[1] = init_db[0];
            }
        }
        apply_eq_and_rms(model, init_db);
    }

    pub(super) fn fast_screen_metrics<B: TrainerBackend>(
        &mut self,
        model: &V3MatryoshkaModel,
    ) -> (f64, f64) {
        let n = self.stem.usable_len();
        let warmup = self.warmup;
        let cold = model.extract_tier(Tier::Ultra);
        let (ol, or) = render_tier_stereo::<B>(&cold, &self.stem.in_l[..n], &self.stem.in_r[..n]);
        let mono = model.mono_out;
        let mut sig_sq = 0.0_f64;
        let mut err_sq = 0.0_f64;
        let mut ml = Vec::with_capacity(n.saturating_sub(warmup));
        for i in warmup..n {
            let yl = f64::from(ol[i]);
            ml.push(yl);
            let tl = f64::from(self.stem.tgt_l[i]);
            let dl = yl - tl;
            sig_sq += tl * tl;
            err_sq += dl * dl;
            if !mono {
                let tr = f64::from(self.stem.tgt_r[i]);
                let dr = f64::from(or[i]) - tr;
                sig_sq += tr * tr;
                err_sq += dr * dr;
            }
        }
        let err_l = self.ltas_l.band_powers(&ml).map_or(99.0, |(py, pm)| {
            (0..33)
                .map(|b| (10.0 * (py[b].max(1.0e-24) / pm[b].max(1.0e-24)).log10()).abs())
                .fold(0.0_f64, f64::max)
        });
        let err_r = self.ltas_r.as_mut().map_or(0.0, |lr| {
            let mr: Vec<f64> = or[warmup..n].iter().map(|&v| f64::from(v)).collect();
            lr.band_powers(&mr).map_or(99.0, |(py, pm)| {
                (0..33)
                    .map(|b| (10.0 * (py[b].max(1.0e-24) / pm[b].max(1.0e-24)).log10()).abs())
                    .fold(0.0_f64, f64::max)
            })
        });
        let esr_db = if sig_sq > 1.0e-18 {
            (10.0 * (err_sq / sig_sq).max(1.0e-12).log10()).clamp(-40.0, 6.0)
        } else {
            0.0
        };
        (err_l.max(err_r), esr_db)
    }

    fn fast_ltas_max_err<B: TrainerBackend>(&mut self, model: &V3MatryoshkaModel) -> f64 {
        self.fast_screen_metrics::<B>(model).0
    }
}

#[derive(Clone)]
pub(super) struct UltraSearchBest {
    pub(super) score: f64,
    pub(super) is_comb: bool,
    pub(super) tank: FdnTankSpec,
    pub(super) model: V3MatryoshkaModel,
    pub(super) track: V3MatryoshkaModel,
    pub(super) db: [[f64; 33]; 2],
}

pub(super) struct ScreenTargetCtx<'a, 'b> {
    pub(super) eq_ctx: &'b mut ClosedLoopEqCtx<'a>,
    pub(super) certs: &'b mut CertCachePair,
    pub(super) welch: &'b WelchSummaryPair<'a>,
    pub(super) wave_ctx: &'b WaveTargetCtx,
    pub(super) tgt_rms: f64,
    pub(super) sweep_check: Option<&'b mut SweepHarmonicCheck>,
}

fn score_ultra_summary(
    sl: &PerceptualCertificationSummary,
    sr: Option<&PerceptualCertificationSummary>,
    is_comb: bool,
    has_lfo: bool,
) -> f64 {
    let mr = sl.mr_iir.max(sr.map_or(0.0, |r| r.mr_iir));
    let ltas = sl.ltas_max_db.max(sr.map_or(0.0, |r| r.ltas_max_db));
    let seg = sl
        .seg_esr_db
        .max(sr.map_or(f64::NEG_INFINITY, |r| r.seg_esr_db));
    let besr = sl
        .raw_besr_db
        .max(sr.map_or(f64::NEG_INFINITY, |r| r.raw_besr_db));
    let ok_tank = summary_passes_mode(sl, SolverQualityMode::BestUntilDiminishingReturns)
        && sr
            .is_none_or(|r| summary_passes_mode(r, SolverQualityMode::BestUntilDiminishingReturns));
    let penalty = if ok_tank { 0.0 } else { 10.0 };
    let comb_bonus = if ok_tank && is_comb { 0.015 } else { 0.0 };
    let lfo_bonus = if ok_tank && has_lfo { 0.025 } else { 0.0 };
    let ltas_over = (ltas - 0.10).max(0.0);
    penalty
        + mr
        + 0.22 * ltas
        + 0.35 * ltas_over
        + 0.05 * seg.max(0.0)
        + 0.008 * besr.clamp(-45.0, 6.0)
        - comb_bonus
        - lfo_bonus
}

pub(super) fn screen_track_finalists<B: TrainerBackend>(
    track: &V3MatryoshkaModel,
    comb_cands: &[FdnTankSpec],
    ctx: &mut ScreenTargetCtx<'_, '_>,
) -> Vec<UltraSearchBest> {
    let zero_tilt = [[0.0_f64; 33]; 2];
    let warmup = ctx.eq_ctx.warmup;
    let mono_out = track.mono_out;
    let mut finalists: Vec<UltraSearchBest> = Vec::new();

    // 1. Fast LTAS pre-screen of comb/chorus candidates -> full MR-IIR cert on top <= 12.
    let comb_pre = par_map_indexed(comb_cands.len(), |idx| {
        let tank_spec = comb_cands[idx];
        let mut local_eq = ctx.eq_ctx.clone();
        let mut cand = track.clone();
        cand.stage4[3] = tank_spec.to_params();
        let mut db_try = [[0.0_f64; 33]; 2];
        local_eq.run_tier::<B>(
            &mut cand,
            &EqRunSpec {
                tier: Tier::Ultra,
                target_rms: ctx.tgt_rms,
                tilt_db: &zero_tilt,
                passes: 1,
                step_gain: 0.85,
            },
            &mut db_try,
        );
        let (ltas, esr_db) = local_eq.fast_screen_metrics::<B>(&cand);
        let sc = ltas + 0.08 * esr_db;
        (tank_spec.lfo_depth_samples >= 1.5, sc, tank_spec, cand)
    });
    let mut comb_mod_pre: Vec<(f64, FdnTankSpec, V3MatryoshkaModel)> = Vec::new();
    let mut comb_stat_pre: Vec<(f64, FdnTankSpec, V3MatryoshkaModel)> = Vec::new();
    for (is_mod, sc, tank_spec, cand) in comb_pre {
        if is_mod {
            comb_mod_pre.push((sc, tank_spec, cand));
        } else {
            comb_stat_pre.push((sc, tank_spec, cand));
        }
    }
    comb_mod_pre.sort_by(|a, b| a.0.total_cmp(&b.0));
    comb_stat_pre.sort_by(|a, b| a.0.total_cmp(&b.0));
    let top_comb: Vec<(FdnTankSpec, V3MatryoshkaModel)> = comb_mod_pre
        .into_iter()
        .take(8)
        .chain(comb_stat_pre.into_iter().take(4))
        .map(|(_, t, c)| (t, c))
        .collect();
    let mut comb_ranked: Vec<(f64, FdnTankSpec)> = par_map_indexed(top_comb.len(), |idx| {
        let (tank_spec, ref cand) = top_comb[idx];
        let mut local_certs = ctx.certs.clone();
        let (sl, sr_sum) =
            eval_tier_cert::<B>(cand, Tier::Ultra, ctx.eq_ctx.stem, &mut local_certs);
        (
            score_ultra_summary(
                &sl,
                sr_sum.as_ref(),
                true,
                tank_spec.lfo_depth_samples >= 1.5,
            ),
            tank_spec,
        )
    });
    comb_ranked.sort_by(|a, b| a.0.total_cmp(&b.0));

    // 2. Linear/LTI prime-FDN scan (`gl in [0.0, 0.55, 0.85]`, `lfo_depth = 0.0`).
    let mut seed_eq = track.clone();
    seed_eq.stage4[3] = FdnTankSpec {
        delays: PRIME_QUARTETS[1],
        g_loop: 0.0,
        base_psi: 0.0,
        wet_ratio: 0.0,
        tap_gains: DIFFUSE_TAPS,
        lfo_rate_hz: 0.0,
        lfo_depth_samples: 0.0,
        lfo_stereo_phase: 0.0,
    }
    .to_params();
    let mut eq_db = [[0.0_f64; 33]; 2];
    ctx.eq_ctx.run_tier::<B>(
        &mut seed_eq,
        &EqRunSpec {
            tier: Tier::Ultra,
            target_rms: ctx.tgt_rms,
            tilt_db: &zero_tilt,
            passes: 4,
            step_gain: 0.85,
        },
        &mut eq_db,
    );

    let mut coarse_grid = Vec::with_capacity(48);
    let quartets: &[[usize; 4]] = if ctx.eq_ctx.stem.usable_len() >= 64_000 {
        &[PRIME_QUARTETS[1], PRIME_QUARTETS[4], PRIME_QUARTETS[5]]
    } else {
        &[PRIME_QUARTETS[1]]
    };
    for &quartet in quartets {
        for &wr in &[0.14_f64, 0.26, 0.44, 0.52, 0.65] {
            for &gl in &[0.0_f32, 0.55, 0.85, 0.94, 0.975] {
                coarse_grid.push((quartet, wr, gl));
            }
        }
    }
    let mut top_coarse: Vec<(f64, [usize; 4], f64, f32)> =
        par_map_indexed(coarse_grid.len(), |idx| {
            let (quartet, wr, gl) = coarse_grid[idx];
            let mut cand = seed_eq.clone();
            cand.stage4[3] = FdnTankSpec {
                delays: quartet,
                g_loop: gl,
                base_psi: FRAC_PI_4 as f32,
                wet_ratio: wr,
                tap_gains: DIFFUSE_TAPS,
                lfo_rate_hz: 0.0,
                lfo_depth_samples: 0.0,
                lfo_stereo_phase: 0.0,
            }
            .to_params();
            let _ = calibrate_stage4_rms::<B>(
                &mut cand,
                Tier::Ultra,
                ctx.eq_ctx.stem,
                warmup,
                ctx.tgt_rms,
                mono_out,
            );
            let mut local_certs = ctx.certs.clone();
            let (sl, sr_sum) =
                eval_tier_cert::<B>(&cand, Tier::Ultra, ctx.eq_ctx.stem, &mut local_certs);
            (
                score_ultra_summary(&sl, sr_sum.as_ref(), false, false),
                quartet,
                wr,
                gl,
            )
        });
    top_coarse.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut diff_specs = Vec::with_capacity(36);
    for &(_, _best_q, wr, gl_seed) in top_coarse.iter().take(2) {
        for &quartet in &PRIME_QUARTETS {
            for &psi in &[0.0_f32, (0.5 * FRAC_PI_4) as f32, FRAC_PI_4 as f32] {
                diff_specs.push(FdnTankSpec {
                    delays: quartet,
                    g_loop: gl_seed,
                    base_psi: psi,
                    wet_ratio: wr,
                    tap_gains: DIFFUSE_TAPS,
                    lfo_rate_hz: 0.0,
                    lfo_depth_samples: 0.0,
                    lfo_stereo_phase: 0.0,
                });
            }
        }
        for &quartet in &[PRIME_QUARTETS[0], PRIME_QUARTETS[1], PRIME_QUARTETS[4]] {
            let depth = clamp_lfo_depth_for_delays(&quartet, 8.0);
            if depth >= 1.5 {
                diff_specs.push(FdnTankSpec {
                    delays: quartet,
                    g_loop: gl_seed,
                    base_psi: FRAC_PI_4 as f32,
                    wet_ratio: wr,
                    tap_gains: DIFFUSE_TAPS,
                    lfo_rate_hz: 1.35,
                    lfo_depth_samples: depth,
                    lfo_stereo_phase: FRAC_PI_4 as f32,
                });
            }
        }
    }
    let mut diffuse_cands: Vec<(f64, FdnTankSpec)> = par_map_indexed(diff_specs.len(), |idx| {
        let tank = diff_specs[idx];
        let mut cand = seed_eq.clone();
        cand.stage4[3] = tank.to_params();
        let _ = calibrate_stage4_rms::<B>(
            &mut cand,
            Tier::Ultra,
            ctx.eq_ctx.stem,
            warmup,
            ctx.tgt_rms,
            mono_out,
        );
        let mut local_eq = ctx.eq_ctx.clone();
        let sc = local_eq.fast_ltas_max_err::<B>(&cand);
        (sc, tank)
    });
    diffuse_cands.sort_by(|a, b| a.0.total_cmp(&b.0));

    // 3. Up to 6 finalists (`2` mod comb + `2` static comb + `2` diffuse) with `18`-iter Welch-33 polish.
    let mut comb_picks: Vec<FdnTankSpec> = comb_ranked
        .iter()
        .filter(|(_, t)| t.lfo_depth_samples >= 1.5)
        .take(2)
        .map(|(_, t)| *t)
        .collect();
    let static_take = if comb_picks.is_empty() { 4 } else { 2 };
    comb_picks.extend(
        comb_ranked
            .iter()
            .filter(|(_, t)| t.lfo_depth_samples < 1.5)
            .take(static_take)
            .map(|(_, t)| *t),
    );
    let diff_take = if comb_picks.is_empty() { 4 } else { 2 };
    let eval_jobs: Vec<(bool, FdnTankSpec)> = comb_picks
        .into_iter()
        .map(|t| (true, t))
        .chain(
            diffuse_cands
                .into_iter()
                .take(diff_take)
                .map(|(_, t)| (false, t)),
        )
        .take(6)
        .collect();

    let (wave_l, wave_r) = ctx.wave_ctx.make_specs(0.015, None, None);

    let raw_finalists = par_map_indexed(eval_jobs.len(), |idx| {
        let (is_comb, tank_spec) = eval_jobs[idx];
        let base_track = if is_comb { track } else { &seed_eq };
        let mut cand = base_track.clone();
        cand.stage4[3] = tank_spec.to_params();
        let mut db_try = if is_comb { [[0.0_f64; 33]; 2] } else { eq_db };
        let mut local_eq = ctx.eq_ctx.clone();
        local_eq.run_tier::<B>(
            &mut cand,
            &EqRunSpec {
                tier: Tier::Ultra,
                target_rms: ctx.tgt_rms,
                tilt_db: &zero_tilt,
                passes: if is_comb { 3 } else { 2 },
                step_gain: 0.90,
            },
            &mut db_try,
        );
        polish_tier_welch33::<B>(
            &mut cand,
            &WelchPolishSpec {
                tier: Tier::Ultra,
                stem: local_eq.stem,
                sum_l: ctx.welch.sum_l,
                sum_r: ctx.welch.sum_r,
                wave_l: Some(&wave_l),
                wave_r: (!mono_out).then_some(&wave_r),
                mr_l: None,
                mr_r: None,
                lock_rms_lr: None,
                goal_db: 0.015,
                iters: 26,
            },
        );
        let mut local_certs = ctx.certs.clone();
        let (sl, sr_sum) = eval_tier_cert::<B>(&cand, Tier::Ultra, local_eq.stem, &mut local_certs);
        let sc = score_ultra_summary(
            &sl,
            sr_sum.as_ref(),
            is_comb,
            tank_spec.lfo_depth_samples >= 1.5,
        );
        UltraSearchBest {
            score: sc,
            is_comb,
            tank: tank_spec,
            model: cand,
            track: base_track.clone(),
            db: db_try,
        }
    });
    for mut res in raw_finalists {
        if let Some(chk) = ctx.sweep_check.as_deref_mut() {
            res.score += chk.score_penalty::<B>(&res.model);
        }
        finalists.push(res);
    }
    finalists
}
