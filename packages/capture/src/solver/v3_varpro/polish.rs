#![forbid(unsafe_code)]
//! Exact `f32` engine trajectory readout polish, subspace LTAS refinement, and monotonicity fallback.

mod extract;
mod prescreen;
mod repair;
mod subspace;

use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, tier_block_lanes, tier_kautz_banks,
};

use crate::metrics::TierQualityThresholds;
use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

use super::super::v3_varpro::live_switch_tier_safe;

use super::basis::{
    ALL_TIERS, PreSplitStemChan, StemChanTraj, SubspaceGramData, als_update_pre_cab_chan,
    assemble_tier_band_grams, build_tier_columns, eval_bezout_loss_from_gram, eval_s2_from_readout,
    map_prev_tier_anchor, normalize_pre_cab_chan, normalize_pre_cab_gauge,
    solve_uplo_with_bypass_and_anchor, unpack_theta_to_readout,
};
use super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset};
use super::primitives::{ClampedCoreParams, par_map_indexed};
use extract::extract_f32_shared_decimated;
use repair::{
    PolishBranchSpec, PolishEvalCtx, TierCandidate, candidate_score, passes_both,
    record_tier_candidate, try_corridor_repair, try_monotone_blends, try_polish_branch,
};
use subspace::subspace_lbfgs_polish_chan;

pub(super) fn blend_readout(
    a: &V3TierReadoutF64,
    b: &V3TierReadoutF64,
    alpha: f64,
) -> V3TierReadoutF64 {
    let om = 1.0 - alpha;
    let mut out = V3TierReadoutF64::zeros();
    for i in 0..7 {
        out.svf_direct_taps_l[i] = alpha * a.svf_direct_taps_l[i] + om * b.svf_direct_taps_l[i];
        out.svf_direct_taps_r[i] = alpha * a.svf_direct_taps_r[i] + om * b.svf_direct_taps_r[i];
    }
    for ch in 0..2 {
        out.pre_cab_u_gain[ch] = alpha * a.pre_cab_u_gain[ch] + om * b.pre_cab_u_gain[ch];
        out.pre_cab_hp_gain[ch] = alpha * a.pre_cab_hp_gain[ch] + om * b.pre_cab_hp_gain[ch];
        out.direct_s2_u_gain[ch] = alpha * a.direct_s2_u_gain[ch] + om * b.direct_s2_u_gain[ch];
        out.direct_s2_hp_gain[ch] = alpha * a.direct_s2_hp_gain[ch] + om * b.direct_s2_hp_gain[ch];
        for i in 0..MAX_TOTAL_GRU_LANES {
            out.pre_cab_w[ch][i] = alpha * a.pre_cab_w[ch][i] + om * b.pre_cab_w[ch][i];
            out.direct_s2_w[ch][i] = alpha * a.direct_s2_w[ch][i] + om * b.direct_s2_w[ch][i];
        }
    }
    for k in 0..64 {
        out.fir_l[k] = alpha * a.fir_l[k] + om * b.fir_l[k];
        out.fir_r[k] = alpha * a.fir_r[k] + om * b.fir_r[k];
    }
    for m in 0..MAX_KAUTZ_POLES {
        out.kautz_c_l[m] = (
            alpha * a.kautz_c_l[m].0 + om * b.kautz_c_l[m].0,
            alpha * a.kautz_c_l[m].1 + om * b.kautz_c_l[m].1,
        );
        out.kautz_c_r[m] = (
            alpha * a.kautz_c_r[m].0 + om * b.kautz_c_r[m].0,
            alpha * a.kautz_c_r[m].1 + om * b.kautz_c_r[m].1,
        );
    }
    out
}

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
    let c_t = tier_block_lanes(tier);
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
    let mut bypass = Vec::with_capacity(9 + 3 * c_t);
    for idx in 0..7 {
        bypass.push((idx, 1.0e-5));
    }
    let svf_ridge = if sweeping_svf { 5.0e-4 } else { 2.0e-5 };
    for idx in 7..9 {
        bypass.push((idx, svf_ridge));
    }
    for idx in 9..9 + 3 * c_t {
        bypass.push((idx, 5.0e-5));
    }
    let theta = solve_uplo_with_bypass_and_anchor(
        grams.p_dim,
        &grams.gram_tot,
        &grams.rhs_tot,
        &bypass,
        &anchor,
        1.0e-6,
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
    [Vec<Vec<f64>>; 2],
    [Vec<f64>; 2],
) {
    let active_ch = if model.mono_out { 1 } else { 2 };
    let sweeping_svf = model.svf_env_oct_sens.abs() > 0.5;
    let mut grams_all: [Option<SubspaceGramData>; 2] = [None, None];
    let mut cols0_all = [Vec::new(), Vec::new()];
    let mut thetas_all = [Vec::new(), Vec::new()];
    let tier = ALL_TIERS[tau];
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
        if let Some(st0) = stems_pre[ch].first().filter(|s| s.n > 0) {
            let tr0 = &trajs[ch][0];
            let mut s2_0 = Vec::new();
            eval_s2_from_readout(
                &model.readouts[tau],
                ch,
                tier_block_lanes(tier),
                &tr0.u[..st0.n],
                &tr0.hp[..st0.n],
                &tr0.h[..st0.n],
                &mut s2_0,
            );
            cols0_all[ch] = build_tier_columns::<B>(
                tier,
                &model.kautz_poles,
                &tr0.tap[..st0.n],
                &tr0.u[..st0.n],
                &tr0.hp[..st0.n],
                &tr0.h[..st0.n],
                &s2_0[..st0.n],
            );
        }
        grams_all[ch] = Some(best_grams);
        thetas_all[ch] = best_theta;
    }
    model.mirror_mono();
    (
        model.readouts[tau].clone(),
        grams_all,
        cols0_all,
        thetas_all,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_subspace_polish<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    tau: usize,
    prev_sum: [Option<&PerceptualCertificationSummary>; 2],
    ls_readout: &V3TierReadoutF64,
    grams: &[Option<SubspaceGramData>; 2],
    cols0: &[Vec<Vec<f64>>; 2],
    thetas: &[Vec<f64>; 2],
    dataset: &V3TrainDataset,
) -> V3TierReadoutF64 {
    let tier = ALL_TIERS[tau];
    let active_ch = if model.mono_out { 1 } else { 2 };
    model.readouts[tau] = ls_readout.clone();
    for ch in 0..active_ch {
        if let Some(ref g) = grams[ch] {
            let theta = subspace_lbfgs_polish_chan::<B>(
                dataset,
                tier,
                prev_sum[ch],
                g,
                &cols0[ch],
                ch,
                &thetas[ch],
            );
            unpack_theta_to_readout(tier, ch, &theta, &mut model.readouts[tau]);
        }
    }
    model.mirror_mono();
    model.readouts[tau].clone()
}

/// Exact `f32` engine trajectory readout polish + subspace LTAS refinement + 7-metric monotonicity fallback.
pub fn polish_f32_matryoshka_readouts_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
) {
    let active_ch = if model.mono_out { 1 } else { 2 };
    normalize_pre_cab_gauge(&mut model.readouts, active_ch);
    model.mirror_mono();
    if dataset.stems.is_empty() {
        return;
    }
    let cold_init = model.to_matryoshka_cold();
    let cp = ClampedCoreParams::from_model(model);
    let mut trajs = [Vec::new(), Vec::new()];
    let extract_stem = |si: usize| -> [StemChanTraj; 2] {
        let st = &dataset.stems[si];
        let n = st.usable_len();
        extract_f32_shared_decimated::<B>(&cold_init, &st.in_l[..n], &st.in_r[..n])
    };
    let per_stem_trajs: Vec<[StemChanTraj; 2]> = if dataset.stems.len() >= 2 {
        par_map_indexed(dataset.stems.len(), extract_stem)
    } else {
        (0..dataset.stems.len()).map(extract_stem).collect()
    };
    for ch_res in per_stem_trajs {
        for (ch, item) in ch_res.into_iter().enumerate().take(active_ch) {
            trajs[ch].push(item);
        }
    }
    let stems_pre: [Vec<PreSplitStemChan>; 2] = std::array::from_fn(|ch| {
        if ch < active_ch {
            PreSplitStemChan::build_vec::<B>(dataset, &trajs[ch], ch, MAX_BLOCK_LANES)
        } else {
            Vec::new()
        }
    });

    let mut summaries_l: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, None];
    let mut summaries_r: [Option<PerceptualCertificationSummary>; 4] = [None, None, None, None];
    let stem0 = &dataset.stems[0];
    let tgt_l_f64: Vec<f64> = stem0.tgt_l.iter().map(|&v| f64::from(v)).collect();
    let tgt_r_f64: Vec<f64> = stem0.tgt_r.iter().map(|&v| f64::from(v)).collect();
    let mut ctx = PolishEvalCtx {
        model,
        tau: 0,
        stem0,
        tgt_l_f64: &tgt_l_f64,
        tgt_r_f64: &tgt_r_f64,
        sr_hz: dataset.sample_rate_hz,
        sl: &mut summaries_l,
        sr: &mut summaries_r,
    };
    for (tau, &tier) in ALL_TIERS.iter().enumerate() {
        ctx.tau = tau;
        if tau == 0 {
            let (ls_r, grams, cols0, thetas) = fit_ls_candidate::<B>(
                ctx.model,
                0,
                None,
                3,
                false,
                &cp.kautz_pre,
                dataset,
                &trajs,
                &stems_pre,
            );
            let (ls_sl, ls_sr, _) = ctx.eval::<B>(ls_r.clone());
            let thr0 = TierQualityThresholds::for_tier(tier);
            let ltas_ok = ls_sl.ltas_max_db <= thr0.max_ltas_db * 0.85
                && ls_sr
                    .as_ref()
                    .is_none_or(|r| r.ltas_max_db <= thr0.max_ltas_db * 0.85);
            if !(passes_both(&ls_sl, &ls_sr) && ltas_ok) {
                let pol_r = apply_subspace_polish::<B>(
                    ctx.model,
                    0,
                    [None, None],
                    &ls_r,
                    &grams,
                    &cols0,
                    &thetas,
                    dataset,
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
                    for &alpha in &[0.85, 0.60, 0.35] {
                        let cand_r = blend_readout(&ls_r, &pol_r, alpha);
                        let (bl_sl, bl_sr, _) = ctx.eval::<B>(cand_r.clone());
                        if passes_both(&bl_sl, &bl_sr) {
                            best = (cand_r, bl_sl, bl_sr);
                            break;
                        }
                    }
                }
                ctx.commit(best.0, best.1, best.2);
            }
            continue;
        }
        let orig_init_r = ctx.model.readouts[tau].clone();
        let prev_r = ctx.model.readouts[tau - 1].clone();
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
                bsl.raw_besr_db <= thr_tau.composite_besr_db - 4.0
                    && bsr
                        .as_ref()
                        .is_none_or(|r| r.raw_besr_db <= thr_tau.composite_besr_db - 4.0)
            })
        };
        for &(do_als, use_prev) in &[(0_usize, true), (2_usize, false)] {
            ctx.model.readouts[tau] = orig_init_r.clone();
            let same_s2_dst = (do_als == 0 && use_prev).then_some(&mut same_s2_best);
            let branch_spec = PolishBranchSpec {
                prev_r: &prev_r,
                prev_sum,
                do_als,
                use_prev_pre_cab: use_prev,
                kautz_pre: &cp.kautz_pre,
                dataset,
                trajs: &trajs,
                stems_pre: &stems_pre,
            };
            if try_polish_branch::<B>(
                &mut ctx,
                &branch_spec,
                &mut best_ok,
                &mut best_any,
                same_s2_dst,
            ) && (!use_prev || has_besr_headroom(&best_ok))
            {
                done = true;
                break;
            }
        }
        if done {
            if let Some((br, bsl, bsr)) = best_ok {
                ctx.commit(br, bsl, bsr);
            }
        } else {
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
                let mut repair_cands: Vec<&V3TierReadoutF64> = Vec::new();
                if let Some(ref c) = best_any
                    && passes_both(&c.1, &c.2)
                {
                    repair_cands.push(&c.0);
                }
                if !want_better_any
                    && let Some(ref c) = same_s2_best
                    && passes_both(&c.1, &c.2)
                {
                    repair_cands.push(&c.0);
                }
                if !repair_cands.is_empty() {
                    let _ = try_corridor_repair::<B>(&mut ctx, &repair_cands, &mut best_ok);
                }
            }
            if best_ok
                .as_ref()
                .is_none_or(|(_, bsl, bsr)| !passes_both(bsl, bsr))
            {
                let (psl, psr, pok) = ctx.eval::<B>(prev_r.clone());
                record_tier_candidate(tier, &mut best_ok, &mut best_any, &prev_r, &psl, &psr, pok);
            }
            if let Some((br, bsl, bsr)) = best_ok {
                ctx.commit(br, bsl, bsr);
            } else {
                let (sl, sr, _) = ctx.eval::<B>(prev_r.clone());
                ctx.commit(prev_r.clone(), sl, sr);
            }
        }
        let cold_chk = ctx.model.to_matryoshka_cold();
        if !live_switch_tier_safe::<B>(&cold_chk, stem0, tau) {
            let cur_r = ctx.model.readouts[tau].clone();
            for &alpha in &[0.85, 0.65, 0.40, 0.15, 0.0] {
                let cand_r = blend_readout(&cur_r, &prev_r, alpha);
                let (sl, sr, ok) = ctx.eval::<B>(cand_r.clone());
                let cold_sw = ctx.model.to_matryoshka_cold();
                if (ok || alpha == 0.0) && live_switch_tier_safe::<B>(&cold_sw, stem0, tau) {
                    ctx.commit(cand_r, sl, sr);
                    break;
                }
            }
        }
    }
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
