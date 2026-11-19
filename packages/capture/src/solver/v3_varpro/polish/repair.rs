#![forbid(unsafe_code)]
//! Candidate scoring, monotonicity verification, and blend fallback for `v3` polish.

mod corridor;
mod rate_split;

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES};

use crate::metrics::TierQualityThresholds;
use crate::metrics::certification::{
    PerceptualCertificationSummary, TargetChannelCertCache, verify_monotone_4tier_summaries,
};
use crate::solver::backends::TrainerBackend;

use super::super::basis::{ALL_TIERS, PreSplitStemChan, StemChanTraj, SubspaceGramData};
use super::super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset, V3TrainStem};
use super::cert_cols::{CertWelchData, apply_subspace_polish};
use super::fit_ls_candidate;
use super::live_switch_tier_safe;
use super::prescreen::{certify_cold_tier_with_prescreen, shares_pre_cab};
pub(super) use corridor::{scale_linear_out, try_corridor_repair};
pub(super) use rate_split::maybe_promote_1x_rate_split;

pub(super) type TierCandidate = (
    V3TierReadoutF64,
    PerceptualCertificationSummary,
    Option<PerceptualCertificationSummary>,
);

fn mono_slice_ok(s: &[Option<PerceptualCertificationSummary>; 4], tau: usize) -> bool {
    let mut dummy: [_; 4] = std::array::from_fn(|i| s[i.min(tau)].clone().unwrap());
    for (k, &tier) in ALL_TIERS.iter().enumerate() {
        dummy[k].tier = tier;
    }
    verify_monotone_4tier_summaries(&dummy)
}

fn mono_pair_ok(s: &[Option<PerceptualCertificationSummary>; 4], tau: usize) -> bool {
    let a = s[tau - 1].clone().unwrap();
    let b = s[tau].clone().unwrap();
    let mut dummy = [a, b.clone(), b.clone(), b];
    for (k, &tier) in ALL_TIERS.iter().enumerate() {
        dummy[k].tier = tier;
    }
    verify_monotone_4tier_summaries(&dummy)
}

pub(super) fn ok_for_tau(
    tau: usize,
    mono_out: bool,
    sl: &[Option<PerceptualCertificationSummary>; 4],
    sr: &[Option<PerceptualCertificationSummary>; 4],
) -> bool {
    let full = mono_slice_ok(sl, tau) && (mono_out || mono_slice_ok(sr, tau));
    if full || tau < 2 {
        return full;
    }
    let prev_full = mono_slice_ok(sl, tau - 1) && (mono_out || mono_slice_ok(sr, tau - 1));
    if !prev_full {
        return mono_pair_ok(sl, tau) && (mono_out || mono_pair_ok(sr, tau));
    }
    false
}

fn chan_pair_meets_margin(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
    is_ultra: bool,
) -> bool {
    let (mr_fac, mr_abs) = if lo.mr_iir <= 0.006 && lo.ltas_max_db > 0.065 {
        (0.975, 1.0e-4)
    } else {
        (0.985, 1.5e-4)
    };
    let mr_lim = (lo.mr_iir - mr_abs)
        .max(lo.mr_iir * 0.96)
        .min(lo.mr_iir * mr_fac);
    let ltas_lim = if lo.ltas_max_db > 0.06 {
        if is_ultra {
            (lo.ltas_max_db - 0.004).min(lo.ltas_max_db * 0.94)
        } else {
            (lo.ltas_max_db - 0.007).min(lo.ltas_max_db * 0.90)
        }
    } else {
        (lo.ltas_max_db - 0.003).min(lo.ltas_max_db * 0.96)
    };
    let esr_cush = if hi.raw_besr_db <= -35.0 { 0.26 } else { 0.08 };
    let band_cush = |b: usize| {
        if hi.band_esr_db[b] <= -35.0 {
            0.26
        } else {
            0.08
        }
    };
    let seg_cush = if hi.seg_esr_db <= -42.0 { 0.18 } else { 0.08 };
    hi.raw_besr_db <= lo.raw_besr_db - esr_cush
        && (0..3).all(|b| hi.band_esr_db[b] <= lo.band_esr_db[b] - band_cush(b))
        && (lo.mr_iir <= 1.0e-5 || hi.mr_iir <= mr_lim)
        && (lo.ltas_max_db <= 1.0e-4 || hi.ltas_max_db <= ltas_lim)
        && hi.seg_esr_db <= lo.seg_esr_db - seg_cush
}

fn margin_pair_opt(
    lo: Option<&PerceptualCertificationSummary>,
    hi: Option<&PerceptualCertificationSummary>,
    is_ultra: bool,
) -> bool {
    lo.zip(hi)
        .is_some_and(|(a, b)| chan_pair_meets_margin(a, b, is_ultra))
}

pub(super) fn ok_for_tau_with_margin(
    tau: usize,
    mono_out: bool,
    sl: &[Option<PerceptualCertificationSummary>; 4],
    sr: &[Option<PerceptualCertificationSummary>; 4],
) -> bool {
    if !ok_for_tau(tau, mono_out, sl, sr) {
        return false;
    }
    if tau == 0 {
        return true;
    }
    let is_ultra = tau == 3;
    margin_pair_opt(sl[tau - 1].as_ref(), sl[tau].as_ref(), is_ultra)
        && (mono_out || margin_pair_opt(sr[tau - 1].as_ref(), sr[tau].as_ref(), is_ultra))
}

pub(super) fn passes_both(
    sl: &PerceptualCertificationSummary,
    sr: &Option<PerceptualCertificationSummary>,
) -> bool {
    sl.passes_target && sr.as_ref().is_none_or(|r| r.passes_target)
}

fn single_summary_violation(
    s: &PerceptualCertificationSummary,
    thr: &TierQualityThresholds,
) -> f64 {
    let fin = |v: f64| if v.is_finite() { v } else { 1.0e6 };
    let comp_v = (fin(s.besr_db) - thr.composite_besr_db).max(0.0);
    let band_v: f64 = (0..3)
        .map(|b| {
            if s.qualified_band_mask[b] {
                (fin(s.band_esr_db[b]) - thr.per_band_besr_db[b]).max(0.0)
            } else {
                0.0
            }
        })
        .sum();
    let ltas_v = 4.0 * (fin(s.ltas_max_db) - thr.max_ltas_db).max(0.0);
    comp_v + band_v + ltas_v
}

pub(super) fn candidate_score(
    tier: Tier,
    sl: &PerceptualCertificationSummary,
    sr: &Option<PerceptualCertificationSummary>,
) -> (bool, bool, f64, f64) {
    let fin = |v: f64| if v.is_finite() { v } else { 1.0e6 };
    let thr = TierQualityThresholds::for_tier(tier);
    let passes_t = passes_both(sl, sr);
    let passes_c = sl.passes_hard_ceiling && sr.as_ref().is_none_or(|r| r.passes_hard_ceiling);
    let worst_viol = sr.as_ref().map_or_else(
        || single_summary_violation(sl, &thr),
        |r| single_summary_violation(sl, &thr).max(single_summary_violation(r, &thr)),
    );
    let worst_besr = sr.as_ref().map_or_else(
        || fin(sl.raw_besr_db),
        |r| fin(sl.raw_besr_db).max(fin(r.raw_besr_db)),
    );
    let worst_ltas = sr.as_ref().map_or_else(
        || fin(sl.ltas_max_db),
        |r| fin(sl.ltas_max_db).max(fin(r.ltas_max_db)),
    );
    let ltas_tie = if passes_t && passes_c && worst_besr <= thr.composite_besr_db - 2.0 {
        2.5 * worst_ltas
    } else {
        0.0
    };
    (passes_t, passes_c, -worst_viol, -(worst_besr + ltas_tie))
}

pub(super) fn record_tier_candidate(
    tier: Tier,
    best_ok: &mut Option<TierCandidate>,
    best_any: &mut Option<TierCandidate>,
    r: &V3TierReadoutF64,
    sl: &PerceptualCertificationSummary,
    sr: &Option<PerceptualCertificationSummary>,
    ok: bool,
) {
    let sc = candidate_score(tier, sl, sr);
    if best_any
        .as_ref()
        .is_none_or(|(_, bsl, bsr)| sc > candidate_score(tier, bsl, bsr))
    {
        *best_any = Some((r.clone(), sl.clone(), sr.clone()));
    }
    if ok
        && best_ok
            .as_ref()
            .is_none_or(|(_, bsl, bsr)| sc > candidate_score(tier, bsl, bsr))
    {
        *best_ok = Some((r.clone(), sl.clone(), sr.clone()));
    }
}

pub(super) struct PolishEvalCtx<'a> {
    pub(super) model: &'a mut V3MatryoshkaOptF64,
    pub(super) tau: usize,
    pub(super) stem0: &'a V3TrainStem,
    pub(super) cert_l: TargetChannelCertCache,
    pub(super) cert_r: Option<TargetChannelCertCache>,
    pub(super) sl: &'a mut [Option<PerceptualCertificationSummary>; 4],
    pub(super) sr: &'a mut [Option<PerceptualCertificationSummary>; 4],
}

impl PolishEvalCtx<'_> {
    pub(super) fn eval_with_prescreen<B: TrainerBackend>(
        &mut self,
        readout: V3TierReadoutF64,
        fast_monotone: bool,
    ) -> (
        PerceptualCertificationSummary,
        Option<PerceptualCertificationSummary>,
        bool,
    ) {
        self.model.readouts[self.tau] = readout;
        self.model.mirror_mono();
        let cold_cur = Box::new(self.model.to_matryoshka_cold());
        let prev_req = if fast_monotone && self.tau > 0 {
            self.sl[self.tau - 1]
                .as_ref()
                .map(|pl| (pl, self.sr[self.tau - 1].as_ref()))
        } else {
            None
        };
        let (sl, sr) = certify_cold_tier_with_prescreen::<B>(
            &cold_cur,
            ALL_TIERS[self.tau],
            self.stem0,
            &mut self.cert_l,
            self.cert_r.as_mut(),
            prev_req,
        );
        self.sl[self.tau] = Some(sl.clone());
        self.sr[self.tau] = sr.clone();
        let ok = ok_for_tau_with_margin(self.tau, cold_cur.mono_out, self.sl, self.sr);
        (sl, sr, ok)
    }

    pub(super) fn eval<B: TrainerBackend>(
        &mut self,
        readout: V3TierReadoutF64,
    ) -> (
        PerceptualCertificationSummary,
        Option<PerceptualCertificationSummary>,
        bool,
    ) {
        self.eval_with_prescreen::<B>(readout, false)
    }

    pub(super) fn eval_fast_monotone<B: TrainerBackend>(
        &mut self,
        readout: V3TierReadoutF64,
    ) -> (
        PerceptualCertificationSummary,
        Option<PerceptualCertificationSummary>,
        bool,
    ) {
        self.eval_with_prescreen::<B>(readout, true)
    }

    pub(super) fn accept_with_mode<B: TrainerBackend>(
        &mut self,
        r: &V3TierReadoutF64,
        best_ok: &mut Option<TierCandidate>,
        best_any: &mut Option<TierCandidate>,
        same_s2_best: &mut Option<&mut Option<TierCandidate>>,
        fast_monotone: bool,
    ) -> (
        bool,
        PerceptualCertificationSummary,
        Option<PerceptualCertificationSummary>,
        bool,
    ) {
        let tier = ALL_TIERS[self.tau];
        let (sl, sr, ok) = self.eval_with_prescreen::<B>(r.clone(), fast_monotone);
        if let Some(slot) = same_s2_best.as_deref_mut() {
            record_tier_candidate(tier, &mut None, slot, r, &sl, &sr, ok);
        }
        record_tier_candidate(tier, best_ok, best_any, r, &sl, &sr, ok);
        let thr = TierQualityThresholds::for_tier(tier);
        let ltas_headroom_ok = sl.ltas_max_db <= thr.max_ltas_db * 0.68
            && sr
                .as_ref()
                .is_none_or(|v| v.ltas_max_db <= thr.max_ltas_db * 0.68);
        let band_headroom_ok = sl
            .band_esr_db
            .iter()
            .zip(&thr.per_band_besr_db)
            .all(|(&b, &lim)| b <= lim - 1.5)
            && sr.as_ref().is_none_or(|r| {
                r.band_esr_db
                    .iter()
                    .zip(&thr.per_band_besr_db)
                    .all(|(&b, &lim)| b <= lim - 1.5)
            });
        let same_rate_prev =
            self.tau == 0 || self.model.tier_os2x[self.tau] == self.model.tier_os2x[self.tau - 1];
        let margin_pair =
            |cur: &PerceptualCertificationSummary,
             prev: Option<&PerceptualCertificationSummary>| {
                prev.is_none_or(|p| {
                    let (mr_fac, ltas_fac) = if self.tau == 3 {
                        (0.95, 0.82)
                    } else {
                        (0.98, 0.92)
                    };
                    let deep_cross_rate = !same_rate_prev && p.raw_besr_db <= -35.0;
                    let gap = if p.raw_besr_db <= -44.0 || deep_cross_rate {
                        0.35
                    } else {
                        0.08
                    };
                    let seg_gap = if deep_cross_rate { 0.25 } else { 0.08 };
                    cur.raw_besr_db <= p.raw_besr_db - gap
                        && (0..3).all(|b| cur.band_esr_db[b] <= p.band_esr_db[b] - gap)
                        && cur.mr_iir <= p.mr_iir * mr_fac
                        && cur.ltas_max_db <= p.ltas_max_db * ltas_fac
                        && cur.seg_esr_db <= p.seg_esr_db - seg_gap
                })
            };
        let adj_margin_ok = self.tau == 0
            || (margin_pair(&sl, self.sl[self.tau - 1].as_ref())
                && sr
                    .as_ref()
                    .is_none_or(|r| margin_pair(r, self.sr[self.tau - 1].as_ref())));
        let headroom_ok = ltas_headroom_ok && band_headroom_ok && adj_margin_ok;
        if ok && passes_both(&sl, &sr) && headroom_ok {
            return (true, sl, sr, ok);
        }
        (false, sl, sr, ok)
    }

    pub(super) fn accept<B: TrainerBackend>(
        &mut self,
        r: &V3TierReadoutF64,
        best_ok: &mut Option<TierCandidate>,
        best_any: &mut Option<TierCandidate>,
        same_s2_best: &mut Option<&mut Option<TierCandidate>>,
    ) -> (
        bool,
        PerceptualCertificationSummary,
        Option<PerceptualCertificationSummary>,
        bool,
    ) {
        self.accept_with_mode::<B>(r, best_ok, best_any, same_s2_best, false)
    }

    pub(super) fn commit(
        &mut self,
        readout: V3TierReadoutF64,
        sl: PerceptualCertificationSummary,
        sr: Option<PerceptualCertificationSummary>,
    ) {
        self.model.readouts[self.tau] = readout;
        self.model.mirror_mono();
        self.sl[self.tau] = Some(sl);
        self.sr[self.tau] = sr;
    }
}

pub(super) struct PolishBranchSpec<'a> {
    pub(super) prev_r: Option<&'a V3TierReadoutF64>,
    pub(super) prev_sum: [Option<&'a PerceptualCertificationSummary>; 2],
    pub(super) do_als: usize,
    pub(super) use_prev_pre_cab: bool,
    pub(super) kautz_pre: &'a [(f64, f64, f64, f64, f64, f64); MAX_KAUTZ_POLES],
    pub(super) dataset: &'a V3TrainDataset,
    pub(super) trajs: &'a [Vec<StemChanTraj>; 2],
    pub(super) stems_pre: &'a [Vec<PreSplitStemChan>; 2],
    pub(super) cert_welch: Option<&'a CertWelchData<'a>>,
}

pub(super) struct LsBranchResult {
    pub(super) ls_r: V3TierReadoutF64,
    pub(super) grams: [Option<SubspaceGramData>; 2],
    pub(super) thetas: [Vec<f64>; 2],
    pub(super) ls_ok: bool,
    pub(super) ls_passes: bool,
    pub(super) worst_raw_besr_db: f64,
}

pub(super) fn eval_branch_ls<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    spec: &PolishBranchSpec<'_>,
    best_ok: &mut Option<TierCandidate>,
    best_any: &mut Option<TierCandidate>,
    mut same_s2_best: Option<&mut Option<TierCandidate>>,
) -> (bool, LsBranchResult) {
    let (ls_r, grams, thetas) = fit_ls_candidate::<B>(
        ctx.model,
        ctx.tau,
        spec.prev_r,
        spec.do_als,
        spec.use_prev_pre_cab,
        spec.kautz_pre,
        spec.dataset,
        spec.trajs,
        spec.stems_pre,
    );
    let (done, sl, sr, ls_ok) = ctx.accept::<B>(&ls_r, best_ok, best_any, &mut same_s2_best);
    let ls_passes = passes_both(&sl, &sr);
    let worst_raw_besr_db = sr
        .as_ref()
        .map_or(sl.raw_besr_db, |r| sl.raw_besr_db.max(r.raw_besr_db));
    (
        done,
        LsBranchResult {
            ls_r,
            grams,
            thetas,
            ls_ok,
            ls_passes,
            worst_raw_besr_db,
        },
    )
}

pub(super) fn run_branch_subspace<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    spec: &PolishBranchSpec<'_>,
    ls_res: &LsBranchResult,
    best_ok: &mut Option<TierCandidate>,
    best_any: &mut Option<TierCandidate>,
    mut same_s2_best: Option<&mut Option<TierCandidate>>,
) -> bool {
    let pol_r = apply_subspace_polish::<B>(
        ctx.model,
        ctx.tau,
        spec.prev_sum,
        &ls_res.ls_r,
        &ls_res.grams,
        &ls_res.thetas,
        spec.trajs,
        spec.stems_pre,
        spec.dataset,
        spec.cert_welch,
    );
    let (done, pol_sl, pol_sr, pol_ok) =
        ctx.accept::<B>(&pol_r, best_ok, best_any, &mut same_s2_best);
    if done {
        return true;
    }
    if !pol_ok || (!passes_both(&pol_sl, &pol_sr) && ls_res.ls_ok) {
        for &alpha in &[0.88, 0.65, 0.38, 0.15] {
            let cand_r = blend_readout(&pol_r, &ls_res.ls_r, alpha);
            if ctx
                .accept_with_mode::<B>(&cand_r, best_ok, best_any, &mut same_s2_best, true)
                .0
            {
                return true;
            }
        }
    }
    false
}

pub(super) fn try_monotone_blends<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    cand_r: &V3TierReadoutF64,
    prev_r: &V3TierReadoutF64,
    best_ok: &mut Option<TierCandidate>,
) {
    let active_ch = if ctx.model.mono_out { 1 } else { 2 };
    if !shares_pre_cab(cand_r, prev_r, active_ch) {
        return;
    }
    let tier = ALL_TIERS[ctx.tau];
    let thr = TierQualityThresholds::for_tier(tier);
    for &alpha in &[0.98, 0.92, 0.82, 0.68, 0.50, 0.30, 0.12] {
        let blended = blend_readout(cand_r, prev_r, alpha);
        let (sl, sr, ok) = ctx.eval_fast_monotone::<B>(blended.clone());
        if ok {
            let better = best_ok.as_ref().is_none_or(|(_, bsl, bsr)| {
                (!passes_both(bsl, bsr) && passes_both(&sl, &sr))
                    || (passes_both(&sl, &sr) == passes_both(bsl, bsr)
                        && candidate_score(tier, &sl, &sr) > candidate_score(tier, bsl, bsr))
            });
            if better {
                *best_ok = Some((blended, sl.clone(), sr.clone()));
            }
            let ltas_margin = sl.ltas_max_db <= thr.max_ltas_db * 0.80
                && sr
                    .as_ref()
                    .is_none_or(|v| v.ltas_max_db <= thr.max_ltas_db * 0.80);
            if passes_both(&sl, &sr) && ltas_margin {
                break;
            }
        }
    }
}

pub(super) fn try_strict_monotone_blends<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    same_s2_best: &Option<TierCandidate>,
    best_any: &Option<TierCandidate>,
    prev_r: &V3TierReadoutF64,
) -> Option<TierCandidate> {
    let tier = ALL_TIERS[ctx.tau];
    let active_ch = if ctx.model.mono_out { 1 } else { 2 };
    let mut hit: Option<TierCandidate> = None;
    for src_opt in [same_s2_best, best_any] {
        let Some(src) = src_opt.as_ref() else {
            continue;
        };
        if !shares_pre_cab(&src.0, prev_r, active_ch) {
            continue;
        }
        for &alpha in &[0.96, 0.88, 0.75, 0.58, 0.40, 0.22, 0.08] {
            let cand_r = blend_readout(&src.0, prev_r, alpha);
            let (bsl, bsr, _) = ctx.eval_fast_monotone::<B>(cand_r.clone());
            if ok_for_tau(ctx.tau, ctx.model.mono_out, ctx.sl, ctx.sr)
                && hit.as_ref().is_none_or(|(_, hsl, hsr)| {
                    candidate_score(tier, &bsl, &bsr) > candidate_score(tier, hsl, hsr)
                })
            {
                hit = Some((cand_r, bsl, bsr));
            }
        }
    }
    hit
}

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
pub(super) fn refine_cross_rate_pro_ltas<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    same_rate_prev: bool,
    spec0: &PolishBranchSpec<'_>,
    ls_res0: &LsBranchResult,
    prev_sl: Option<&PerceptualCertificationSummary>,
    prev_sr: Option<&PerceptualCertificationSummary>,
    thr_tau: &TierQualityThresholds,
    best_ok: &mut Option<TierCandidate>,
) {
    let cross_rate_tight_ltas = ctx.tau == 2
        && !same_rate_prev
        && best_ok.as_ref().is_some_and(|(_, bsl, bsr)| {
            let cur_ltas = bsr
                .as_ref()
                .map_or(bsl.ltas_max_db, |r| bsl.ltas_max_db.max(r.ltas_max_db));
            let prev_ltas = prev_sl.map_or(0.0, |p| {
                prev_sr.map_or(p.ltas_max_db, |pr| p.ltas_max_db.max(pr.ltas_max_db))
            });
            cur_ltas > 0.022
                && (prev_ltas <= 0.25
                    || cur_ltas > prev_ltas * 0.70
                    || cur_ltas > thr_tau.max_ltas_db * 0.65)
        });
    if !cross_rate_tight_ltas {
        return;
    }
    let mut sub_ok: Option<TierCandidate> = None;
    let mut sub_any: Option<TierCandidate> = None;
    let _ = run_branch_subspace::<B>(ctx, spec0, ls_res0, &mut sub_ok, &mut sub_any, None);
    if let Some((sr_r, sr_sl, sr_sr)) = sub_ok {
        let cur_ltas = best_ok.as_ref().map_or(f64::INFINITY, |(_, bsl, bsr)| {
            bsr.as_ref()
                .map_or(bsl.ltas_max_db, |r| bsl.ltas_max_db.max(r.ltas_max_db))
        });
        let new_ltas = sr_sr
            .as_ref()
            .map_or(sr_sl.ltas_max_db, |r| sr_sl.ltas_max_db.max(r.ltas_max_db));
        let new_besr_ok = sr_sl.raw_besr_db <= thr_tau.composite_besr_db - 2.0
            && sr_sr
                .as_ref()
                .is_none_or(|r| r.raw_besr_db <= thr_tau.composite_besr_db - 2.0);
        if new_besr_ok && passes_both(&sr_sl, &sr_sr) && new_ltas + 1.0e-4 < cur_ltas {
            *best_ok = Some((sr_r, sr_sl, sr_sr));
        }
    }
}

pub(super) fn repair_live_switch_safety<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    stem0: &V3TrainStem,
    prev_r: &V3TierReadoutF64,
    same_rate_prev: bool,
) {
    let tau = ctx.tau;
    let cold_chk = Box::new(ctx.model.to_matryoshka_cold());
    if live_switch_tier_safe::<B>(&cold_chk, stem0, tau) {
        return;
    }
    let cur_r = ctx.model.readouts[tau].clone();
    for &alpha in &[0.85, 0.65, 0.40, 0.15, 0.0] {
        let cand_r = if same_rate_prev {
            blend_readout(&cur_r, prev_r, alpha)
        } else {
            scale_linear_out(&cur_r, alpha.max(0.10))
        };
        let (sl, sr, ok) = ctx.eval::<B>(cand_r.clone());
        let cold_sw = ctx.model.to_matryoshka_cold();
        if (ok || alpha == 0.0) && live_switch_tier_safe::<B>(&cold_sw, stem0, tau) {
            ctx.commit(cand_r, sl, sr);
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use dnam_engine::stages::V3MatryoshkaModel;

    use super::*;
    use crate::metrics::ArchetypeFamily;
    use crate::solver::backends::DefaultTrainerBackend;

    fn make_summary(
        tier: Tier,
        raw_besr_db: f64,
        ltas_max_db: f64,
    ) -> PerceptualCertificationSummary {
        PerceptualCertificationSummary {
            family: ArchetypeFamily::CleanPedalPreamp,
            tier,
            raw_besr_db,
            besr_db: raw_besr_db,
            band_esr_db: [raw_besr_db; 3],
            qualified_band_mask: [true, false, true],
            mr_iir: 0.01,
            ltas_max_db,
            ltas_unadjusted_max_db: ltas_max_db,
            ltas_excluded_bands: [false; 33],
            ltas_residual_band_power: [0.0; 33],
            seg_esr_db: raw_besr_db,
            passes_target: raw_besr_db <= -15.0 && ltas_max_db <= 3.0,
            passes_hard_ceiling: raw_besr_db <= -12.0 && ltas_max_db <= 4.0,
        }
    }

    fn make_identity_stem(len: usize) -> V3TrainStem {
        use crate::metrics::companion::canonical_ltas_33_centers_hz;
        use dnam_engine::stages::{V3CoreInstance, V3MatryoshkaModel};
        use dnam_engine::types::Tier;

        let centers = canonical_ltas_33_centers_hz();
        let mut x = vec![0.0_f32; len];
        for (i, slot) in x.iter_mut().enumerate() {
            let t = (i as f64) / 48_000.0;
            let mut sum = 0.0_f64;
            for (k, &hz) in centers.iter().enumerate() {
                let phase = 0.37 * (k as f64);
                sum += (2.0 * PI * hz * t + phase).sin();
            }
            *slot = (sum * 0.04) as f32;
        }
        let cold = V3MatryoshkaModel::identity(true, true).extract_tier(Tier::Full);
        let mut inst = V3CoreInstance::<DefaultTrainerBackend>::from_cold_model(&cold);
        let mut tgt_l = vec![0.0_f32; len];
        let mut tgt_r = vec![0.0_f32; len];
        inst.process_block_stereo(&x, &x, &mut tgt_l, &mut tgt_r);
        V3TrainStem {
            in_l: x.clone(),
            in_r: x,
            tgt_l,
            tgt_r,
            warmup: 128,
        }
    }

    #[test]
    fn test_candidate_score_record_and_monotone_blends() {
        let mut r = V3TierReadoutF64 {
            svf_direct_taps_l: [1.0; 7],
            svf_direct_taps_r: [1.0; 7],
            direct_s2_u_gain: [0.5, -0.5],
            direct_s2_hp_gain: [0.25, -0.25],
            ..Default::default()
        };
        r.direct_s2_w[0][0] = 0.8;
        r.fir_l[0] = 1.0;
        r.fir_r[0] = 1.0;
        r.kautz_c_l[0] = (0.4, -0.2);

        let sl_good = make_summary(Tier::Full, -28.0, 1.0);
        let sr_bad = Some(make_summary(Tier::Full, -8.0, 6.0));
        let sl_nan = make_summary(Tier::Full, f64::NAN, 1.0);
        assert!(
            candidate_score(Tier::Full, &sl_good, &None)
                > candidate_score(Tier::Full, &sl_good, &sr_bad)
        );
        assert!(
            candidate_score(Tier::Full, &sl_good, &None)
                > candidate_score(Tier::Full, &sl_nan, &None)
        );

        let mut best_ok = None;
        let mut best_any = None;
        record_tier_candidate(
            Tier::Full,
            &mut best_ok,
            &mut best_any,
            &r,
            &sl_good,
            &sr_bad,
            true,
        );
        record_tier_candidate(
            Tier::Full,
            &mut best_ok,
            &mut best_any,
            &r,
            &sl_good,
            &None,
            true,
        );
        assert!(best_any.is_some());
        assert!(best_ok.is_some());
    }

    #[test]
    fn test_monotone_blends_and_margin_checks() {
        let stem = make_identity_stem(4_800);
        let n0 = stem.usable_len();
        let w0 = stem.warmup.min(n0);
        let tgt_l: Vec<f64> = stem.tgt_l[..n0].iter().map(|&v| f64::from(v)).collect();
        let mut model =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, true));
        let mut sl = [None, None, None, None];
        let mut sr = [None, None, None, None];
        let mut ctx = PolishEvalCtx {
            model: &mut model,
            tau: 0,
            stem0: &stem,
            cert_l: TargetChannelCertCache::new(&tgt_l, 48_000.0, w0),
            cert_r: None,
            sl: &mut sl,
            sr: &mut sr,
        };

        let id_r = V3TierReadoutF64::default();
        let mut tau0_r = id_r.clone();
        tau0_r.fir_l[0] = 0.98;
        let (s0_detuned, _, ok0) = ctx.eval::<DefaultTrainerBackend>(tau0_r.clone());
        assert!(ok0 && s0_detuned.passes_target);
        ctx.commit(tau0_r.clone(), s0_detuned, None);

        let mut diff_pre_cab = id_r.clone();
        diff_pre_cab.pre_cab_u_gain[0] = 2.0;
        ctx.tau = 1;
        let mut blend_ok = None;
        try_monotone_blends::<DefaultTrainerBackend>(&mut ctx, &diff_pre_cab, &id_r, &mut blend_ok);
        assert!(blend_ok.is_none());

        try_monotone_blends::<DefaultTrainerBackend>(&mut ctx, &id_r, &tau0_r, &mut blend_ok);
        assert!(blend_ok.is_some());
    }
}
