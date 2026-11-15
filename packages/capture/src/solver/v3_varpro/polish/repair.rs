#![forbid(unsafe_code)]
//! Candidate scoring, monotonicity verification, blend fallback, and corridor repair for `v3` polish.

use dnam_engine::Tier;
use dnam_engine::stages::v3_core::cascade::MAX_KAUTZ_POLES;

use crate::metrics::TierQualityThresholds;
use crate::metrics::certification::{
    PerceptualCertificationSummary, verify_monotone_4tier_summaries,
};
use crate::solver::backends::TrainerBackend;

use super::super::basis::{ALL_TIERS, PreSplitStemChan, StemChanTraj};
use super::super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset, V3TrainStem};
use super::prescreen::{certify_cold_tier_with_prescreen, shares_pre_cab};
use super::{apply_subspace_polish, blend_readout, fit_ls_candidate};

pub(super) type TierCandidate = (
    V3TierReadoutF64,
    PerceptualCertificationSummary,
    Option<PerceptualCertificationSummary>,
);

pub(super) fn ok_for_tau(
    tau: usize,
    mono_out: bool,
    sl: &[Option<PerceptualCertificationSummary>; 4],
    sr: &[Option<PerceptualCertificationSummary>; 4],
) -> bool {
    let mono_ok = |s: &[Option<PerceptualCertificationSummary>; 4]| {
        let mut dummy: [_; 4] = std::array::from_fn(|i| s[i.min(tau)].clone().unwrap());
        for (k, &tier) in ALL_TIERS.iter().enumerate() {
            dummy[k].tier = tier;
        }
        verify_monotone_4tier_summaries(&dummy)
    };
    mono_ok(sl) && (mono_out || mono_ok(sr))
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
    (passes_t, passes_c, -worst_viol, -worst_besr)
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
    pub(super) tgt_l_f64: &'a [f64],
    pub(super) tgt_r_f64: &'a [f64],
    pub(super) sr_hz: f64,
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
        let cold_cur = self.model.to_matryoshka_cold();
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
            self.tgt_l_f64,
            self.tgt_r_f64,
            self.sr_hz,
            prev_req,
        );
        self.sl[self.tau] = Some(sl.clone());
        self.sr[self.tau] = sr.clone();
        let ok = ok_for_tau(self.tau, cold_cur.mono_out, self.sl, self.sr);
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
        let headroom_ok = self.tau == 3
            || (sl.ltas_max_db <= thr.max_ltas_db * 0.78
                && sr
                    .as_ref()
                    .is_none_or(|v| v.ltas_max_db <= thr.max_ltas_db * 0.78));
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
    pub(super) prev_r: &'a V3TierReadoutF64,
    pub(super) prev_sum: [Option<&'a PerceptualCertificationSummary>; 2],
    pub(super) do_als: usize,
    pub(super) use_prev_pre_cab: bool,
    pub(super) kautz_pre: &'a [(f64, f64, f64, f64, f64, f64); MAX_KAUTZ_POLES],
    pub(super) dataset: &'a V3TrainDataset,
    pub(super) trajs: &'a [Vec<StemChanTraj>; 2],
    pub(super) stems_pre: &'a [Vec<PreSplitStemChan>; 2],
}

pub(super) fn try_polish_branch<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    spec: &PolishBranchSpec<'_>,
    best_ok: &mut Option<TierCandidate>,
    best_any: &mut Option<TierCandidate>,
    mut same_s2_best: Option<&mut Option<TierCandidate>>,
) -> bool {
    let (ls_r, grams, cols0, thetas) = fit_ls_candidate::<B>(
        ctx.model,
        ctx.tau,
        Some(spec.prev_r),
        spec.do_als,
        spec.use_prev_pre_cab,
        spec.kautz_pre,
        spec.dataset,
        spec.trajs,
        spec.stems_pre,
    );
    let (done, _, _, ls_ok) = ctx.accept::<B>(&ls_r, best_ok, best_any, &mut same_s2_best);
    if done {
        return true;
    }
    let pol_r = apply_subspace_polish::<B>(
        ctx.model,
        ctx.tau,
        spec.prev_sum,
        &ls_r,
        &grams,
        &cols0,
        &thetas,
        spec.dataset,
    );
    let (done, pol_sl, pol_sr, pol_ok) =
        ctx.accept::<B>(&pol_r, best_ok, best_any, &mut same_s2_best);
    if done {
        return true;
    }
    if !pol_ok || (!passes_both(&pol_sl, &pol_sr) && ls_ok) {
        for &alpha in &[0.88, 0.65, 0.38, 0.15] {
            let cand_r = blend_readout(&pol_r, &ls_r, alpha);
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
            if passes_both(&sl, &sr) {
                break;
            }
        }
    }
}

fn scale_linear_out(a: &V3TierReadoutF64, s: f64) -> V3TierReadoutF64 {
    let mut out = a.clone();
    for i in 0..7 {
        out.svf_direct_taps_l[i] *= s;
        out.svf_direct_taps_r[i] *= s;
    }
    for ch in 0..2 {
        out.direct_s2_u_gain[ch] *= s;
        out.direct_s2_hp_gain[ch] *= s;
        for v in &mut out.direct_s2_w[ch] {
            *v *= s;
        }
    }
    for k in 0..64 {
        out.fir_l[k] *= s;
        out.fir_r[k] *= s;
    }
    for m in 0..MAX_KAUTZ_POLES {
        out.kautz_c_l[m].0 *= s;
        out.kautz_c_l[m].1 *= s;
        out.kautz_c_r[m].0 *= s;
        out.kautz_c_r[m].1 *= s;
    }
    out
}

pub(super) fn try_corridor_repair<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    cands: &[&V3TierReadoutF64],
    best_ok: &mut Option<TierCandidate>,
) -> bool {
    let tau = ctx.tau;
    if tau < 1 || cands.is_empty() {
        return false;
    }
    let active_ch = if ctx.model.mono_out { 1 } else { 2 };
    let orig_prev = ctx.model.readouts[tau - 1].clone();
    let relaxed_prevs: Vec<V3TierReadoutF64> = if tau >= 2 {
        let lower_r = ctx.model.readouts[tau - 2].clone();
        if !shares_pre_cab(&orig_prev, &lower_r, active_ch) {
            return false;
        }
        [0.94, 0.86, 0.76, 0.64, 0.50, 0.35]
            .iter()
            .map(|&alpha| blend_readout(&orig_prev, &lower_r, alpha))
            .collect()
    } else {
        [0.992, 0.985, 0.975]
            .iter()
            .map(|&s| scale_linear_out(&orig_prev, s))
            .collect()
    };
    let orig_psl = ctx.sl[tau - 1].clone();
    let orig_psr = ctx.sr[tau - 1].clone();
    for relaxed_prev in relaxed_prevs {
        ctx.tau = tau - 1;
        let (psl, psr, pok) = ctx.eval_fast_monotone::<B>(relaxed_prev.clone());
        ctx.tau = tau;
        if !(pok && passes_both(&psl, &psr)) {
            continue;
        }
        for &cand in cands {
            let (sl, sr, ok) = ctx.eval_fast_monotone::<B>(cand.clone());
            if ok && passes_both(&sl, &sr) {
                *best_ok = Some((cand.clone(), sl, sr));
                return true;
            }
            let mut local_ok = None;
            try_monotone_blends::<B>(ctx, cand, &relaxed_prev, &mut local_ok);
            if let Some((br, bsl, bsr)) = local_ok
                && passes_both(&bsl, &bsr)
            {
                *best_ok = Some((br, bsl, bsr));
                return true;
            }
        }
    }
    ctx.model.readouts[tau - 1] = orig_prev;
    ctx.model.mirror_mono();
    ctx.sl[tau - 1] = orig_psl;
    ctx.sr[tau - 1] = orig_psr;
    false
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use dnam_engine::stages::v3_core::V3MatryoshkaModel;

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
        use dnam_engine::stages::v3_core::{V3CoreInstance, V3MatryoshkaModel};
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
    fn test_candidate_score_record_and_scale_linear_out() {
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
        r.kautz_c_r[0] = (-0.4, 0.2);
        let scaled = scale_linear_out(&r, 0.5);
        assert!((scaled.svf_direct_taps_l[0] - 0.5).abs() < 1.0e-12);
        assert!((scaled.direct_s2_u_gain[0] - 0.25).abs() < 1.0e-12);
        assert!((scaled.direct_s2_w[0][0] - 0.4).abs() < 1.0e-12);
        assert!((scaled.fir_r[0] - 0.5).abs() < 1.0e-12);
        assert!((scaled.kautz_c_l[0].0 - 0.2).abs() < 1.0e-12);

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
            &scaled,
            &sl_good,
            &None,
            true,
        );
        assert!(best_any.is_some());
        assert!(best_ok.is_some());
    }

    #[test]
    fn test_monotone_blends_and_corridor_repair_paths() {
        let stem = make_identity_stem(4_800);
        let tgt_l: Vec<f64> = stem.tgt_l.iter().map(|&v| f64::from(v)).collect();
        let tgt_r: Vec<f64> = stem.tgt_r.iter().map(|&v| f64::from(v)).collect();
        let mut model =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, true));
        let mut sl = [None, None, None, None];
        let mut sr = [None, None, None, None];
        let mut ctx = PolishEvalCtx {
            model: &mut model,
            tau: 0,
            stem0: &stem,
            tgt_l_f64: &tgt_l,
            tgt_r_f64: &tgt_r,
            sr_hz: 48_000.0,
            sl: &mut sl,
            sr: &mut sr,
        };

        let id_r = V3TierReadoutF64::default();
        let mut tau0_r = id_r.clone();
        tau0_r.fir_l[0] = 0.98;
        let (s0_detuned, _, ok0) = ctx.eval::<DefaultTrainerBackend>(tau0_r.clone());
        assert!(ok0 && s0_detuned.passes_target);
        ctx.commit(tau0_r.clone(), s0_detuned, None);

        // shares_pre_cab mismatch causes early return
        let mut diff_pre_cab = id_r.clone();
        diff_pre_cab.pre_cab_u_gain[0] = 2.0;
        ctx.tau = 1;
        let mut blend_ok = None;
        try_monotone_blends::<DefaultTrainerBackend>(&mut ctx, &diff_pre_cab, &id_r, &mut blend_ok);
        assert!(blend_ok.is_none());

        // Matching pre_cab blending from tau0 (0.98) toward id_r (1.0) improves over tau=0
        try_monotone_blends::<DefaultTrainerBackend>(&mut ctx, &id_r, &tau0_r, &mut blend_ok);
        assert!(blend_ok.is_some());

        // tau = 0 or empty cands returns false in try_corridor_repair
        ctx.tau = 0;
        let mut corr_ok = None;
        assert!(!try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&id_r],
            &mut corr_ok
        ));
        ctx.tau = 1;
        assert!(!try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[],
            &mut corr_ok
        ));

        // Commit exact id_r at tau=0, then test tau=1 corridor repair via blend & direct cand paths
        ctx.tau = 0;
        let (s_exact, _, _) = ctx.eval::<DefaultTrainerBackend>(id_r.clone());
        ctx.commit(id_r.clone(), s_exact, None);
        ctx.tau = 1;
        let mut overshoot = id_r.clone();
        overshoot.fir_l[0] = 1.02;
        assert!(try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&overshoot],
            &mut corr_ok
        ));

        ctx.tau = 0;
        let (s_exact2, _, _) = ctx.eval::<DefaultTrainerBackend>(id_r.clone());
        ctx.commit(id_r.clone(), s_exact2, None);
        ctx.tau = 1;
        assert!(try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&id_r],
            &mut corr_ok
        ));

        // tau = 2 pre_cab mismatch returns false; matching pre_cab with failing cand rolls back
        ctx.tau = 2;
        ctx.model.readouts[0] = id_r.clone();
        ctx.model.readouts[1] = diff_pre_cab;
        assert!(!try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&id_r],
            &mut corr_ok
        ));

        ctx.model.readouts[1] = id_r.clone();
        let mut bad_cand = id_r.clone();
        bad_cand.fir_l[0] = 0.0;
        let repaired =
            try_corridor_repair::<DefaultTrainerBackend>(&mut ctx, &[&bad_cand], &mut corr_ok);
        assert!(!repaired);
    }
}
