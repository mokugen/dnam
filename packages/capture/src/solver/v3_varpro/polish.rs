#![forbid(unsafe_code)]
//! Exact `f32` engine trajectory readout polish, subspace LTAS refinement, and monotonicity fallback.

mod cert_cols;
mod extract;
mod prescreen;
mod repair;
mod subspace;

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{
    MAX_KAUTZ_POLES, NUM_GRU_BLOCKS, tier_block_groups_array, tier_block_lanes_array,
    tier_kautz_banks, tier_total_gru_lanes,
};
use dnam_engine::stages::{V3CoreInstance, V3MatryoshkaModel};

use crate::metrics::TierQualityThresholds;
use crate::metrics::certification::{PerceptualCertificationSummary, TargetChannelCertCache};
use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

use super::basis::{
    ALL_TIERS, PreSplitStemChan, StemChanTraj, SubspaceGramData, als_update_pre_cab_chan,
    assemble_tier_band_grams, eval_bezout_loss_from_gram, eval_s2_from_readout,
    map_prev_tier_anchor, normalize_pre_cab_chan, normalize_pre_cab_gauge,
    solve_tier_readout_uplo_bounded, unpack_theta_to_readout,
};
use super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset, V3TrainStem};
use super::primitives::{ClampedCoreParams, par_map_indexed};
use cert_cols::{CertWelchData, apply_subspace_polish};
use extract::extract_f32_shared_for_rate;
use repair::{
    PolishBranchSpec, PolishEvalCtx, TierCandidate, blend_readout, candidate_score, eval_branch_ls,
    maybe_promote_1x_rate_split, ok_for_tau, passes_both, record_tier_candidate,
    refine_cross_rate_pro_ltas, repair_live_switch_safety, run_branch_subspace,
    try_corridor_repair, try_monotone_blends, try_strict_monotone_blends,
};

#[allow(clippy::too_many_arguments)]
fn solve_polish_chan_ls<B: TrainerBackend>(
    readout: &V3TierReadoutF64,
    tau: usize,
    prev_readout: Option<&V3TierReadoutF64>,
    poles: &[(f64, f64)],
    sweeping_svf: bool,
    ch: usize,
    dataset: &V3TrainDataset,
    trajs_ch: &[StemChanTraj],
    stems_pre: &[PreSplitStemChan],
) -> (V3TierReadoutF64, SubspaceGramData, Vec<f64>, f64) {
    let tier = ALL_TIERS[tau];
    let c_t = tier_block_lanes_array(tier);
    let total_h = tier_total_gru_lanes(tier);
    let k_t = tier_kautz_banks(tier) * 4;
    let mut s2_per_stem = vec![Vec::new(); stems_pre.len()];
    for (si, st) in stems_pre.iter().enumerate() {
        if st.seg > 0 {
            let tr = &trajs_ch[si];
            eval_s2_from_readout(
                readout,
                ch,
                c_t,
                &tr.u[..st.n],
                &tr.hp[..st.n],
                &tr.h[..st.n],
                &mut s2_per_stem[si],
            );
        }
    }
    let grams =
        assemble_tier_band_grams::<B>(tier, poles, k_t, &dataset.bank, stems_pre, &s2_per_stem);
    let anchor = prev_readout
        .map(|p| map_prev_tier_anchor(tau, p, ch))
        .unwrap_or_default();
    let mut bypass = Vec::with_capacity(9 + total_h);
    for idx in 0..7 {
        bypass.push((idx, 1.0e-5));
    }
    let svf_ridge = if sweeping_svf { 5.0e-4 } else { 2.0e-5 };
    for idx in 7..9 {
        bypass.push((idx, svf_ridge));
    }
    for idx in 9..9 + total_h {
        bypass.push((idx, 5.0e-5));
    }
    let pre_cab_w_sq: f64 = readout.pre_cab_w[ch].iter().map(|w| w * w).sum();
    let theta = solve_tier_readout_uplo_bounded(
        grams.p_dim,
        total_h,
        pre_cab_w_sq,
        &grams.gram_tot,
        &grams.rhs_tot,
        &bypass,
        (!anchor.is_empty()).then_some((anchor.as_slice(), 1.0e-6)),
    );
    let mut out_r = readout.clone();
    unpack_theta_to_readout(tier, ch, &theta, &mut out_r);
    let bez = eval_bezout_loss_from_gram(
        grams.p_dim,
        &grams.gram_tot,
        &grams.rhs_tot,
        grams.c_tot,
        &theta,
    );
    (out_r, grams, theta, bez)
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(super) fn fit_ls_candidate<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    tau: usize,
    prev_r: Option<&V3TierReadoutF64>,
    do_als: usize,
    use_prev_pre_cab: bool,
    kautz_pre: &[(f64, f64, f64, f64, f64, f64); MAX_KAUTZ_POLES],
    dataset: &V3TrainDataset,
    trajs: &[Vec<StemChanTraj>; 2],
    stems_pre: &[Vec<PreSplitStemChan>; 2],
) -> (
    V3TierReadoutF64,
    [Option<SubspaceGramData>; 2],
    [Vec<f64>; 2],
) {
    let active_ch = if model.mono_out { 1 } else { 2 };
    let sweeping_svf = model.svf_env_oct_sens.abs() > 0.5;
    let mut grams_all: [Option<SubspaceGramData>; 2] = [None, None];
    let mut thetas_all = [Vec::new(), Vec::new()];
    for ch in 0..active_ch {
        if let Some(pr) = prev_r.filter(|_| use_prev_pre_cab) {
            model.readouts[tau].pre_cab_u_gain[ch] = pr.pre_cab_u_gain[ch];
            model.readouts[tau].pre_cab_hp_gain[ch] = pr.pre_cab_hp_gain[ch];
            model.readouts[tau].pre_cab_w[ch].copy_from_slice(&pr.pre_cab_w[ch]);
        } else {
            normalize_pre_cab_chan(&mut model.readouts[tau], prev_r, tau, ch);
        }
        let (mut best_r, mut best_grams, mut best_theta, _) = solve_polish_chan_ls::<B>(
            &model.readouts[tau],
            tau,
            prev_r,
            &model.kautz_poles,
            sweeping_svf,
            ch,
            dataset,
            &trajs[ch],
            &stems_pre[ch],
        );
        model.readouts[tau] = best_r;
        for _ in 0..do_als {
            if als_update_pre_cab_chan::<B>(
                model,
                tau,
                ch,
                dataset,
                &trajs[ch],
                &stems_pre[ch],
                kautz_pre,
            ) {
                normalize_pre_cab_chan(&mut model.readouts[tau], prev_r, tau, ch);
                (best_r, best_grams, best_theta, _) = solve_polish_chan_ls::<B>(
                    &model.readouts[tau],
                    tau,
                    prev_r,
                    &model.kautz_poles,
                    sweeping_svf,
                    ch,
                    dataset,
                    &trajs[ch],
                    &stems_pre[ch],
                );
                model.readouts[tau] = best_r;
            } else {
                break;
            }
        }
        grams_all[ch] = Some(best_grams);
        thetas_all[ch] = best_theta;
    }
    model.mirror_mono();
    (model.readouts[tau].clone(), grams_all, thetas_all)
}

/// Exact `f32` engine trajectory readout polish + subspace LTAS refinement + 7-metric monotonicity fallback.
pub fn polish_f32_matryoshka_readouts_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
) {
    polish_f32_matryoshka_readouts_with_cert_stem::<B>(model, dataset, None);
}

/// Exact `f32` engine trajectory readout polish certifying against an optional full-length `cert_stem0`.
#[inline(never)]
pub fn polish_f32_matryoshka_readouts_with_cert_stem<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    cert_stem0: Option<&V3TrainStem>,
) {
    let active_ch = if model.mono_out { 1 } else { 2 };
    normalize_pre_cab_gauge(&mut model.readouts, active_ch);
    model.mirror_mono();
    if dataset.stems.is_empty() {
        return;
    }
    let cold_init = Box::new(model.to_matryoshka_cold());
    let cp = ClampedCoreParams::from_model(model);
    let rate_block_groups = |rate_os2x: bool| -> [usize; NUM_GRU_BLOCKS] {
        let mut bg = [1_usize; NUM_GRU_BLOCKS];
        for (tau, &tier) in ALL_TIERS.iter().enumerate() {
            if model.tier_os2x[tau] == rate_os2x {
                let tg = tier_block_groups_array(tier);
                for b in 0..NUM_GRU_BLOCKS {
                    bg[b] = bg[b].max(tg[b]);
                }
            }
        }
        bg
    };
    let need_1x = model.tier_os2x.contains(&false);
    let need_2x = model.tier_os2x.contains(&true);
    let bg_1x = rate_block_groups(false);
    let bg_2x = rate_block_groups(true);
    let n_stems = dataset.stems.len();
    let mut jobs: Vec<(bool, [usize; NUM_GRU_BLOCKS], usize)> = Vec::with_capacity(n_stems * 2);
    if need_1x {
        for si in 0..n_stems {
            jobs.push((false, bg_1x, si));
        }
    }
    if need_2x {
        for si in 0..n_stems {
            jobs.push((true, bg_2x, si));
        }
    }
    let mut job_results: Vec<(bool, [StemChanTraj; 2])> = if jobs.len() >= 2 {
        par_map_indexed(jobs.len(), |ji| {
            let (rate_os2x, block_groups, si) = jobs[ji];
            let st = &dataset.stems[si];
            let n = st.usable_len();
            (
                rate_os2x,
                extract_f32_shared_for_rate::<B>(
                    &cold_init,
                    rate_os2x,
                    block_groups,
                    &st.in_l[..n],
                    &st.in_r[..n],
                ),
            )
        })
    } else {
        jobs.iter()
            .map(|&(rate_os2x, block_groups, si)| {
                let st = &dataset.stems[si];
                let n = st.usable_len();
                (
                    rate_os2x,
                    extract_f32_shared_for_rate::<B>(
                        &cold_init,
                        rate_os2x,
                        block_groups,
                        &st.in_l[..n],
                        &st.in_r[..n],
                    ),
                )
            })
            .collect()
    };
    let mut trajs_1x: [Vec<StemChanTraj>; 2] = [Vec::new(), Vec::new()];
    let mut trajs_2x: [Vec<StemChanTraj>; 2] = [Vec::new(), Vec::new()];
    for (rate_os2x, ch_res) in job_results.drain(..) {
        let dst = if rate_os2x {
            &mut trajs_2x
        } else {
            &mut trajs_1x
        };
        for (ch, item) in ch_res.into_iter().enumerate().take(active_ch) {
            dst[ch].push(item);
        }
    }
    let build_pre_split = |trajs: [Vec<StemChanTraj>; 2], block_groups: [usize; NUM_GRU_BLOCKS]| {
        let c_max = block_groups.map(|g| g * 4);
        let stems_pre: [Vec<PreSplitStemChan>; 2] = std::array::from_fn(|ch| {
            if ch < active_ch {
                PreSplitStemChan::build_vec::<B>(dataset, &trajs[ch], ch, c_max)
            } else {
                Vec::new()
            }
        });
        (trajs, stems_pre)
    };
    let bundle_1x = need_1x.then(|| build_pre_split(trajs_1x, bg_1x));
    let bundle_2x = need_2x.then(|| build_pre_split(trajs_2x, bg_2x));
    let cert_welch = CertWelchData::build::<B>(
        &cold_init,
        cert_stem0,
        dataset,
        need_1x,
        need_2x,
        bg_1x,
        bg_2x,
    );

    let mut summaries_l: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, None];
    let mut summaries_r: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, None];
    let stem0 = cert_stem0
        .filter(|s| s.usable_len() > 0)
        .unwrap_or(&dataset.stems[0]);
    let n0 = stem0.usable_len();
    let w0 = stem0.warmup.min(n0);
    let tgt_l_f64: Vec<f64> = stem0.tgt_l[..n0].iter().map(|&v| f64::from(v)).collect();
    let cert_l = TargetChannelCertCache::new(&tgt_l_f64, dataset.sample_rate_hz, w0);
    let cert_r = (!model.mono_out).then(|| {
        let tgt_r_f64: Vec<f64> = stem0.tgt_r[..n0].iter().map(|&v| f64::from(v)).collect();
        TargetChannelCertCache::new(&tgt_r_f64, dataset.sample_rate_hz, w0)
    });
    let mut ctx = PolishEvalCtx {
        model,
        tau: 0,
        stem0,
        cert_l,
        cert_r,
        sl: &mut summaries_l,
        sr: &mut summaries_r,
    };
    let stage0_active = ctx.model.stage0.op_p_sub_mix.abs() > 1.0e-6
        || ctx.model.stage0.op_p_sub2_mix.abs() > 1.0e-6
        || ctx.model.stage0.op_p_oct_mix.abs() > 1.0e-6
        || ctx.model.stage0.op_p_env_mix.abs() > 1.0e-6;
    let mut incoherent_fast = false;
    for (tau, &tier) in ALL_TIERS.iter().enumerate() {
        ctx.tau = tau;
        let tier_rate = ctx.model.tier_os2x[tau];
        let (trajs, stems_pre) = if tier_rate {
            bundle_2x.as_ref().expect("2x polish bundle built")
        } else {
            bundle_1x.as_ref().expect("1x polish bundle built")
        };
        if tau == 0 {
            let thr0 = TierQualityThresholds::for_tier(tier);
            let orig_init_r = ctx.model.readouts[0].clone();
            let (mut ls_r, mut grams, mut thetas) = fit_ls_candidate::<B>(
                ctx.model,
                0,
                None,
                0,
                false,
                &cp.kautz_pre,
                dataset,
                trajs,
                stems_pre,
            );
            let (mut ls_sl, mut ls_sr, _) = ctx.eval::<B>(ls_r.clone());
            let worst_ls0 = ls_sr
                .as_ref()
                .map_or(ls_sl.raw_besr_db, |r| ls_sl.raw_besr_db.max(r.raw_besr_db));
            if !stage0_active && worst_ls0 > -5.5 {
                incoherent_fast = true;
                ctx.commit(ls_r, ls_sl, ls_sr);
                continue;
            }
            let lite_headroom = passes_both(&ls_sl, &ls_sr)
                && ls_sl.raw_besr_db <= thr0.composite_besr_db - 2.0
                && ls_sr
                    .as_ref()
                    .is_none_or(|r| r.raw_besr_db <= thr0.composite_besr_db - 2.0);
            if !lite_headroom {
                ctx.model.readouts[0] = orig_init_r;
                let (als_r, als_g, als_th) = fit_ls_candidate::<B>(
                    ctx.model,
                    0,
                    None,
                    1,
                    false,
                    &cp.kautz_pre,
                    dataset,
                    trajs,
                    stems_pre,
                );
                let (als_sl, als_sr, _) = ctx.eval::<B>(als_r.clone());
                if candidate_score(tier, &als_sl, &als_sr) > candidate_score(tier, &ls_sl, &ls_sr) {
                    ls_r = als_r;
                    grams = als_g;
                    thetas = als_th;
                    ls_sl = als_sl;
                    ls_sr = als_sr;
                }
            }
            let ltas_cap = if ls_sl.raw_besr_db <= -25.0
                && ls_sr.as_ref().is_none_or(|r| r.raw_besr_db <= -25.0)
            {
                0.30
            } else {
                thr0.max_ltas_db * 0.80
            };
            let ltas_ok = ls_sl.ltas_max_db <= ltas_cap
                && ls_sr.as_ref().is_none_or(|r| r.ltas_max_db <= ltas_cap);
            if !(passes_both(&ls_sl, &ls_sr) && ltas_ok) {
                let pol_r = apply_subspace_polish::<B>(
                    ctx.model,
                    0,
                    [None, None],
                    &ls_r,
                    &grams,
                    &thetas,
                    trajs,
                    stems_pre,
                    dataset,
                    cert_welch.as_ref(),
                );
                let (pol_sl, pol_sr, _) = ctx.eval::<B>(pol_r.clone());
                let mut best = if candidate_score(tier, &ls_sl, &ls_sr)
                    >= candidate_score(tier, &pol_sl, &pol_sr)
                {
                    (ls_r.clone(), ls_sl, ls_sr)
                } else {
                    (pol_r.clone(), pol_sl.clone(), pol_sr.clone())
                };
                if passes_both(&pol_sl, &pol_sr) {
                    let bl_ltas_cap = ltas_cap.min(thr0.max_ltas_db * 0.78);
                    let extra_headroom = pol_sl.besr_db <= thr0.composite_besr_db - 4.0
                        && pol_sr
                            .as_ref()
                            .is_none_or(|r| r.besr_db <= thr0.composite_besr_db - 4.0);
                    let alphas: &[f64] = if extra_headroom {
                        &[0.85, 0.60, 0.35, 0.15, 0.0]
                    } else {
                        &[0.85, 0.60, 0.35]
                    };
                    for &alpha in alphas {
                        let cand_r = blend_readout(&ls_r, &pol_r, alpha);
                        let (bl_sl, bl_sr, _) = ctx.eval::<B>(cand_r.clone());
                        let bl_ltas_ok = bl_sl.ltas_max_db <= bl_ltas_cap
                            && bl_sr
                                .as_ref()
                                .is_none_or(|r| r.ltas_max_db <= bl_ltas_cap);
                        if passes_both(&bl_sl, &bl_sr) && bl_ltas_ok {
                            best = (cand_r, bl_sl, bl_sr);
                            break;
                        }
                    }
                }
                ctx.commit(best.0, best.1, best.2);
            } else {
                ctx.commit(ls_r, ls_sl, ls_sr);
            }
            continue;
        }
        if incoherent_fast {
            let same_rate_prev = ctx.model.tier_os2x[tau] == ctx.model.tier_os2x[tau - 1];
            let prev_r = ctx.model.readouts[tau - 1].clone();
            let (ls_r, _, _) = fit_ls_candidate::<B>(
                ctx.model,
                tau,
                same_rate_prev.then_some(&prev_r),
                0,
                same_rate_prev,
                &cp.kautz_pre,
                dataset,
                trajs,
                stems_pre,
            );
            let (ls_sl, ls_sr, _) = ctx.eval::<B>(ls_r.clone());
            ctx.commit(ls_r, ls_sl, ls_sr);
            continue;
        }
        let orig_init_r = ctx.model.readouts[tau].clone();
        let prev_r = ctx.model.readouts[tau - 1].clone();
        let same_rate_prev = ctx.model.tier_os2x[tau] == ctx.model.tier_os2x[tau - 1];
        let prev_sl = ctx.sl[tau - 1].clone();
        let prev_sr = ctx.sr[tau - 1].clone();
        let prev_sum = [prev_sl.as_ref(), prev_sr.as_ref()];
        let mut best_ok: Option<TierCandidate> = None;
        let mut best_any: Option<TierCandidate> = None;
        let mut same_s2_best: Option<TierCandidate> = None;
        let mut done = false;
        let thr_tau = TierQualityThresholds::for_tier(tier);
        let has_besr_headroom = |cand: &Option<TierCandidate>| {
            cand.as_ref().is_some_and(|(_, bsl, bsr)| {
                bsl.raw_besr_db <= thr_tau.composite_besr_db - 2.0
                    && bsr
                        .as_ref()
                        .is_none_or(|r| r.raw_besr_db <= thr_tau.composite_besr_db - 2.0)
            })
        };
        let make_spec = |do_als: usize, use_prev: bool| PolishBranchSpec {
            prev_r: same_rate_prev.then_some(&prev_r),
            prev_sum,
            do_als,
            use_prev_pre_cab: use_prev,
            kautz_pre: &cp.kautz_pre,
            dataset,
            trajs,
            stems_pre,
            cert_welch: cert_welch.as_ref(),
        };
        ctx.model.readouts[tau] = orig_init_r.clone();
        let spec0 = make_spec(0, same_rate_prev);
        let (done0, ls_res0) = eval_branch_ls::<B>(
            &mut ctx,
            &spec0,
            &mut best_ok,
            &mut best_any,
            same_rate_prev.then_some(&mut same_s2_best),
        );
        if done0 && has_besr_headroom(&best_ok) {
            refine_cross_rate_pro_ltas::<B>(
                &mut ctx,
                same_rate_prev,
                &spec0,
                &ls_res0,
                prev_sl.as_ref(),
                prev_sr.as_ref(),
                &thr_tau,
                &mut best_ok,
            );
            done = true;
        } else {
            let spec0_has_headroom = ls_res0.worst_raw_besr_db <= thr_tau.composite_besr_db - 2.0;
            if spec0_has_headroom {
                if ctx.tau == 3
                    && same_rate_prev
                    && ls_res0.ls_passes
                    && !ls_res0.ls_ok
                    && ls_res0.worst_raw_besr_db <= thr_tau.composite_besr_db - 2.5
                {
                    try_monotone_blends::<B>(&mut ctx, &ls_res0.ls_r, &prev_r, &mut best_ok);
                    if has_besr_headroom(&best_ok)
                        && best_ok
                            .as_ref()
                            .is_some_and(|(_, bsl, bsr)| passes_both(bsl, bsr))
                    {
                        done = true;
                    }
                }
                if !done {
                    if run_branch_subspace::<B>(
                        &mut ctx,
                        &spec0,
                        &ls_res0,
                        &mut best_ok,
                        &mut best_any,
                        same_rate_prev.then_some(&mut same_s2_best),
                    ) && has_besr_headroom(&best_ok)
                    {
                        done = true;
                    } else if ctx.tau == 3
                        && same_rate_prev
                        && let Some(ref s2_cand) = same_s2_best
                        && s2_cand.1.raw_besr_db <= thr_tau.composite_besr_db - 3.0
                    {
                        try_monotone_blends::<B>(&mut ctx, &s2_cand.0, &prev_r, &mut best_ok);
                        if has_besr_headroom(&best_ok)
                            && best_ok
                                .as_ref()
                                .is_some_and(|(_, bsl, bsr)| passes_both(bsl, bsr))
                        {
                            done = true;
                        }
                    }
                }
            }
            if !done && same_rate_prev {
                ctx.model.readouts[tau] = orig_init_r.clone();
                let spec0_own = make_spec(0, false);
                let (done_own, ls_own) =
                    eval_branch_ls::<B>(&mut ctx, &spec0_own, &mut best_ok, &mut best_any, None);
                if done_own && has_besr_headroom(&best_ok) {
                    done = true;
                } else if (ls_own.worst_raw_besr_db <= thr_tau.composite_besr_db - 2.0
                    || ls_own.worst_raw_besr_db <= ls_res0.worst_raw_besr_db + 0.50)
                    && run_branch_subspace::<B>(
                        &mut ctx,
                        &spec0_own,
                        &ls_own,
                        &mut best_ok,
                        &mut best_any,
                        None,
                    )
                    && has_besr_headroom(&best_ok)
                {
                    done = true;
                }
            }
            if !done {
                ctx.model.readouts[tau] = orig_init_r.clone();
                let spec1 = make_spec(1, false);
                let (done1, ls_res1) =
                    eval_branch_ls::<B>(&mut ctx, &spec1, &mut best_ok, &mut best_any, None);
                if done1 {
                    done = true;
                } else if spec0_has_headroom {
                    if run_branch_subspace::<B>(
                        &mut ctx,
                        &spec1,
                        &ls_res1,
                        &mut best_ok,
                        &mut best_any,
                        None,
                    ) {
                        done = true;
                    }
                } else {
                    let prefer_delta = if same_rate_prev { 0.20 } else { 0.15 };
                    let prefer_1 =
                        ls_res1.worst_raw_besr_db + prefer_delta < ls_res0.worst_raw_besr_db;
                    let ordered = if prefer_1 {
                        [
                            (false, &spec1, &ls_res1),
                            (same_rate_prev, &spec0, &ls_res0),
                        ]
                    } else {
                        [
                            (same_rate_prev, &spec0, &ls_res0),
                            (false, &spec1, &ls_res1),
                        ]
                    };
                    for &(is_prev, spec, ls_res) in &ordered {
                        if run_branch_subspace::<B>(
                            &mut ctx,
                            spec,
                            ls_res,
                            &mut best_ok,
                            &mut best_any,
                            is_prev.then_some(&mut same_s2_best),
                        ) {
                            done = true;
                            break;
                        }
                    }
                }
            }
        }
        if done {
            if let Some((br, bsl, bsr)) = best_ok {
                ctx.commit(br, bsl, bsr);
            }
        } else {
            if same_rate_prev {
                if let Some(ref any_cand) = best_any
                    && best_ok.as_ref().is_none_or(|(_, bsl, bsr)| {
                        !passes_both(bsl, bsr)
                            && candidate_score(tier, &any_cand.1, &any_cand.2)
                                > candidate_score(tier, bsl, bsr)
                    })
                {
                    try_monotone_blends::<B>(&mut ctx, &any_cand.0, &prev_r, &mut best_ok);
                }
                if best_ok
                    .as_ref()
                    .is_none_or(|(_, bsl, bsr)| !passes_both(bsl, bsr))
                    && let Some(ref s2_cand) = same_s2_best
                {
                    try_monotone_blends::<B>(&mut ctx, &s2_cand.0, &prev_r, &mut best_ok);
                }
            }
            let want_better_any = !has_besr_headroom(&best_ok)
                && best_any.as_ref().is_some_and(|c| {
                    passes_both(&c.1, &c.2)
                        && best_ok
                            .as_ref()
                            .is_none_or(|(_, bsl, _)| c.1.raw_besr_db + 0.25 < bsl.raw_besr_db)
                });
            if ctx.tau >= 1
                && (best_ok
                    .as_ref()
                    .is_none_or(|(_, bsl, bsr)| !passes_both(bsl, bsr))
                    || want_better_any)
            {
                let need_pass = best_ok
                    .as_ref()
                    .is_some_and(|(_, bsl, bsr)| passes_both(bsl, bsr));
                let mut repair_cands: Vec<&V3TierReadoutF64> = Vec::new();
                if let Some(ref c) = best_any
                    && (!need_pass || passes_both(&c.1, &c.2))
                {
                    repair_cands.push(&c.0);
                }
                if !want_better_any
                    && let Some(ref c) = same_s2_best
                    && (!need_pass || passes_both(&c.1, &c.2))
                {
                    repair_cands.push(&c.0);
                }
                if !repair_cands.is_empty() {
                    let _ = try_corridor_repair::<B>(&mut ctx, &repair_cands, &mut best_ok);
                }
            }
            let ok_any = best_any.as_ref().is_some_and(|(_, asl, asr)| {
                ctx.sl[tau] = Some(asl.clone());
                ctx.sr[tau] = asr.clone();
                ok_for_tau(tau, ctx.model.mono_out, ctx.sl, ctx.sr)
            });
            let strict_blend_hit = if !ok_any && same_rate_prev && best_ok.is_none() {
                try_strict_monotone_blends::<B>(&mut ctx, &same_s2_best, &best_any, &prev_r)
            } else {
                None
            };
            if best_ok
                .as_ref()
                .is_none_or(|(_, bsl, bsr)| !passes_both(bsl, bsr))
                && !ok_any
                && strict_blend_hit.is_none()
                && same_rate_prev
            {
                let (psl, psr, pok) = ctx.eval::<B>(prev_r.clone());
                record_tier_candidate(tier, &mut best_ok, &mut best_any, &prev_r, &psl, &psr, pok);
            }
            if let Some((br, bsl, bsr)) = best_ok {
                ctx.commit(br, bsl, bsr);
            } else if ok_any {
                let (br, bsl, bsr) = best_any.unwrap();
                ctx.commit(br, bsl, bsr);
            } else if let Some((br, bsl, bsr)) = strict_blend_hit {
                ctx.commit(br, bsl, bsr);
            } else if same_rate_prev {
                let (sl, sr, _) = ctx.eval::<B>(prev_r.clone());
                ctx.commit(prev_r.clone(), sl, sr);
            } else if let Some((ar, asl, asr)) = best_any {
                ctx.commit(ar, asl, asr);
            } else {
                let (sl, sr, _) = ctx.eval::<B>(orig_init_r.clone());
                ctx.commit(orig_init_r, sl, sr);
            }
        }
        repair_live_switch_safety::<B>(&mut ctx, stem0, &prev_r, same_rate_prev);
    }

    // If 1x `Full` beats 2x `Pro`/`Ultra` (e.g., pure steep LTI high-pass above oversampler LPF cutoff),
    // promote the calibrated `1x` front-end across all four tiers so the hierarchy stays strictly monotone.
    maybe_promote_1x_rate_split::<B>(model, dataset, cert_stem0, &summaries_l, &summaries_r);
}

/// Dispatches [`polish_f32_matryoshka_readouts_with_backend`] via fastest CPU backend.
pub fn polish_f32_matryoshka_readouts(model: &mut V3MatryoshkaOptF64, dataset: &V3TrainDataset) {
    struct Task<'a>(&'a mut V3MatryoshkaOptF64, &'a V3TrainDataset);
    impl TrainerTask<()> for Task<'_> {
        fn run<B: TrainerBackend>(self) {
            polish_f32_matryoshka_readouts_with_backend::<B>(self.0, self.1);
        }
    }
    DynamicTrainer::detect().dispatch(Task(model, dataset));
}

#[inline(never)]
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
    let (mut out_l, mut out_r) = ([0.0_f32; 128], [0.0_f32; 128]);
    let seg = n / (max_tau + 1);
    for (t_idx, &tier) in ALL_TIERS.iter().enumerate().take(max_tau + 1) {
        inst.switch_tier(&cold.extract_tier(tier));
        let mut s = t_idx * seg;
        let end = (t_idx + 1) * seg;
        while s < end {
            let len = (end - s).min(128);
            let e = s + len;
            let keep_in_r = !cold.mono_out || cold.feat_params.key_link > 0.0;
            let r_slice = if !keep_in_r {
                &stem.in_l[s..e]
            } else {
                &stem.in_r[s..e]
            };
            inst.process_block_stereo(
                &stem.in_l[s..e],
                r_slice,
                &mut out_l[..len],
                &mut out_r[..len],
            );
            for (&yl, &yr) in out_l[..len].iter().zip(&out_r[..len]) {
                if !yl.is_finite() || !yr.is_finite() || yl.abs() > bound || yr.abs() > bound {
                    return false;
                }
            }
            s = e;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::backends::DefaultTrainerBackend;

    #[test]
    fn test_polish_empty_single_job_stereo_cert_and_als_subspace_fallback() {
        let mut m_empty =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(false, false));
        let empty_ds = V3TrainDataset::new_weighted(vec![], 48_000.0);
        polish_f32_matryoshka_readouts(&mut m_empty, &empty_ds);
        assert!(m_empty.readouts[0].pre_cab_u_gain[0].is_finite());

        let n = 1_024_usize;
        let mut in_l = vec![0.0_f32; n];
        let mut in_r = vec![0.0_f32; n];
        let mut tgt_l = vec![0.0_f32; n];
        let mut tgt_r = vec![0.0_f32; n];
        for i in 0..n {
            let t = i as f32;
            let xl = 0.45 * (t * 0.043).sin() + 0.15 * (t * 0.17).cos();
            let xr = 0.40 * (t * 0.051).cos() - 0.15 * (t * 0.19).sin();
            in_l[i] = xl;
            in_r[i] = xr;
            tgt_l[i] = 0.7 * (4.2 * xl).tanh() + 0.35 * xl * xl + 0.25 * (t * 0.61).sin();
            tgt_r[i] = 0.65 * (3.8 * xr).tanh() - 0.30 * xr * xr + 0.25 * (t * 0.73).cos();
        }
        let stem = V3TrainStem {
            in_l,
            in_r,
            tgt_l,
            tgt_r,
            warmup: 128,
        };
        let cert_stem = stem.clone();
        let ds = V3TrainDataset::new_weighted(vec![(stem, 1.0)], 48_000.0);
        let mut m =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(false, false));
        for b in 0..NUM_GRU_BLOCKS {
            m.blocks[b].w_in_n[0][0] = 1.4;
        }
        m.readouts[0].pre_cab_u_gain = [0.25, 0.20];
        m.readouts[0].pre_cab_w[0][0] = 0.15;
        m.readouts[0].pre_cab_w[1][0] = 0.15;
        polish_f32_matryoshka_readouts_with_cert_stem::<DefaultTrainerBackend>(
            &mut m,
            &ds,
            Some(&cert_stem),
        );
        for ro in &m.readouts {
            assert!(ro.pre_cab_u_gain.iter().all(|v| v.is_finite()));
            assert!(
                ro.fir_l
                    .iter()
                    .chain(ro.fir_r.iter())
                    .all(|v| v.is_finite())
            );
        }
    }
}
