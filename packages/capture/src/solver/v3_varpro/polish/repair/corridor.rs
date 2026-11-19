#![forbid(unsafe_code)]
//! Scaling, HF perturbation, and monotone prefix-corridor repair helpers for `polish::repair`.

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, tier_kautz_banks};

use crate::metrics::TierQualityThresholds;
use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::backends::TrainerBackend;

use super::super::super::basis::ALL_TIERS;
use super::super::super::model_f64::V3TierReadoutF64;
use super::super::prescreen::shares_pre_cab;
use super::{
    PolishEvalCtx, TierCandidate, blend_readout, ok_for_tau, ok_for_tau_with_margin, passes_both,
    try_monotone_blends, try_strict_monotone_blends,
};

pub(in super::super) fn scale_linear_out(a: &V3TierReadoutF64, s: f64) -> V3TierReadoutF64 {
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

fn perturb_hf_and_scale(a: &V3TierReadoutF64, scale: f64, eps_hf: f64) -> V3TierReadoutF64 {
    let mut out = scale_linear_out(a, scale);
    out.fir_l[0] += eps_hf;
    out.fir_l[1] -= eps_hf;
    out.fir_r[0] += eps_hf;
    out.fir_r[1] -= eps_hf;
    out
}

/// Applies a causal 3-tap quadratic Lagrange fractional delay `z^{-d}` (`h0 + h1 z^{-1} + h2 z^{-2}`)
/// plus matching Kautz pole phase rotation `e^{-j theta_m d}` and uniform scaling so cross-rate
/// `1x -> 2x` monotone repair smoothly relaxes `bESR`, `MR-IIR`, and `SegESR` from exact `+1` (`d -> 0`)
/// with `< 0.005 dB` `LTAS` change.
fn frac_delay_and_scale(
    a: &V3TierReadoutF64,
    poles: &[(f64, f64); MAX_KAUTZ_POLES],
    tier: Tier,
    scale: f64,
    delay_samples: f64,
) -> V3TierReadoutF64 {
    let mut out = scale_linear_out(a, scale);
    let fir_len = ((tier as usize) + 1) * 16;
    let d = delay_samples.clamp(0.0, 0.35);
    let h0 = (1.0 - d) * (1.0 - 0.5 * d);
    let h1 = d * (2.0 - d);
    let h2 = -0.5 * d * (1.0 - d);
    for fir in [&mut out.fir_l[..fir_len], &mut out.fir_r[..fir_len]] {
        let (mut x1, mut x2) = (0.0_f64, 0.0_f64);
        for tap in fir.iter_mut() {
            let x0 = *tap;
            *tap = h0 * x0 + h1 * x1 + h2 * x2;
            x2 = x1;
            x1 = x0;
        }
    }
    let k_cnt = tier_kautz_banks(tier) * 4;
    for m in 0..k_cnt.min(MAX_KAUTZ_POLES) {
        let (st, ct) = (poles[m].1 * d).sin_cos();
        let (u_re, u_im) = (ct, -st);
        for kc in [&mut out.kautz_c_l[m], &mut out.kautz_c_r[m]] {
            let (cr, ci) = *kc;
            *kc = (cr * u_re - ci * u_im, cr * u_im + ci * u_re);
        }
    }
    out
}

fn has_2db_margin(
    tau: usize,
    sl: &PerceptualCertificationSummary,
    sr: &Option<PerceptualCertificationSummary>,
) -> bool {
    let lim = TierQualityThresholds::for_tier(ALL_TIERS[tau]).composite_besr_db - 2.0;
    sl.besr_db <= lim && sr.as_ref().is_none_or(|r| r.besr_db <= lim)
}

fn accept_repaired_cand<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    cand: &V3TierReadoutF64,
    prev_ref: &V3TierReadoutF64,
    require_pass: bool,
    require_margin: bool,
    best_ok: &mut Option<TierCandidate>,
) -> bool {
    let (sl, sr, ok_margin) = ctx.eval_fast_monotone::<B>(cand.clone());
    let ok = if require_margin {
        ok_margin
    } else {
        ok_for_tau(ctx.tau, ctx.model.mono_out, ctx.sl, ctx.sr)
    };
    if ok && (!require_pass || passes_both(&sl, &sr)) {
        *best_ok = Some((cand.clone(), sl, sr));
        return true;
    }
    if require_margin {
        let mut local_ok = None;
        try_monotone_blends::<B>(ctx, cand, prev_ref, &mut local_ok);
        if let Some((br, bsl, bsr)) = local_ok
            && (!require_pass || passes_both(&bsl, &bsr))
        {
            *best_ok = Some((br, bsl, bsr));
            return true;
        }
    } else if let Some((br, bsl, bsr)) =
        try_strict_monotone_blends::<B>(ctx, &None, &Some((cand.clone(), sl, sr)), prev_ref)
        && (!require_pass || passes_both(&bsl, &bsr))
    {
        *best_ok = Some((br, bsl, bsr));
        return true;
    }
    false
}

struct PrevRelaxCtx<'a> {
    orig_prev: &'a V3TierReadoutF64,
    orig_psl: &'a Option<PerceptualCertificationSummary>,
    orig_psr: &'a Option<PerceptualCertificationSummary>,
    prev_ltas_headroom: bool,
    prev_passed: bool,
    require_pass: bool,
    max_ltas_db: f64,
}

fn try_relaxed_prevs_pass<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    relaxed_prevs: &[V3TierReadoutF64],
    cands: &[&V3TierReadoutF64],
    prev_ctx: &PrevRelaxCtx<'_>,
    require_margin: bool,
    best_ok: &mut Option<TierCandidate>,
) -> bool {
    let tau = ctx.tau;
    let ltas_cap = if require_margin {
        prev_ctx.max_ltas_db * 0.90
    } else {
        prev_ctx.max_ltas_db
    };
    for relaxed_prev in relaxed_prevs {
        ctx.tau = tau - 1;
        let (psl, psr, pok_margin) = ctx.eval_fast_monotone::<B>(relaxed_prev.clone());
        ctx.tau = tau;
        let pok = if require_margin {
            pok_margin
        } else {
            ok_for_tau(tau - 1, ctx.model.mono_out, ctx.sl, ctx.sr)
        };
        let ltas_ok = !prev_ctx.prev_ltas_headroom
            || !prev_ctx.prev_passed
            || (psl.ltas_max_db <= ltas_cap
                && psr.as_ref().is_none_or(|r| r.ltas_max_db <= ltas_cap));
        if !(pok && ltas_ok && (!prev_ctx.prev_passed || passes_both(&psl, &psr))) {
            continue;
        }
        for &cand in cands {
            if accept_repaired_cand::<B>(
                ctx,
                cand,
                relaxed_prev,
                prev_ctx.require_pass,
                require_margin,
                best_ok,
            ) {
                return true;
            }
        }
    }
    ctx.model.readouts[tau - 1] = prev_ctx.orig_prev.clone();
    ctx.model.mirror_mono();
    ctx.sl[tau - 1] = prev_ctx.orig_psl.clone();
    ctx.sr[tau - 1] = prev_ctx.orig_psr.clone();
    false
}

enum PrefixStepKind {
    HfScale(f64, f64),
    FracDelayScale(f64, f64),
}

struct CascadeCtx<'a> {
    orig_prefix_r: &'a [V3TierReadoutF64],
    orig_prefix_sl: &'a [Option<PerceptualCertificationSummary>],
    orig_prefix_sr: &'a [Option<PerceptualCertificationSummary>],
    orig_passed: &'a [bool],
    orig_margin2: &'a [bool],
    orig_ltas_vals: &'a [f64],
    eval_cands: &'a [TierCandidate],
    require_pass: bool,
}

fn restore_cascade_prefix(ctx: &mut PolishEvalCtx<'_>, cc: &CascadeCtx<'_>, tau: usize) {
    ctx.model.readouts[..tau].clone_from_slice(&cc.orig_prefix_r[..tau]);
    ctx.sl[..tau].clone_from_slice(&cc.orig_prefix_sl[..tau]);
    ctx.sr[..tau].clone_from_slice(&cc.orig_prefix_sr[..tau]);
    ctx.tau = tau;
    ctx.model.mirror_mono();
}

fn try_cascade_schedule<B: TrainerBackend>(
    ctx: &mut PolishEvalCtx<'_>,
    cc: &CascadeCtx<'_>,
    steps: &[PrefixStepKind],
    require_margin: bool,
    best_ok: &mut Option<TierCandidate>,
) -> bool {
    let tau = ctx.tau;
    let poles = ctx.model.kautz_poles;
    for step in steps {
        restore_cascade_prefix(ctx, cc, tau);
        let mut prefix_ok = true;
        for (k, orig_rk) in cc.orig_prefix_r.iter().enumerate() {
            let cand_k = match *step {
                PrefixStepKind::HfScale(d_base, eps_base) => {
                    let f_k = 1.65_f64.powi((tau - 1 - k) as i32);
                    perturb_hf_and_scale(
                        orig_rk,
                        (1.0 - d_base * f_k).clamp(0.75, 0.9999),
                        eps_base * f_k,
                    )
                }
                PrefixStepKind::FracDelayScale(scale_base, delay_base) => {
                    let f_k = 1.45_f64.powi((tau - 1 - k) as i32);
                    let scale_k = (1.0 - (1.0 - scale_base) * f_k).clamp(0.85, 0.9999);
                    frac_delay_and_scale(
                        orig_rk,
                        &poles,
                        ALL_TIERS[k],
                        scale_k,
                        (delay_base * f_k).clamp(0.0, 0.35),
                    )
                }
            };
            ctx.tau = k;
            let (sl_k, sr_k, ok_margin) = ctx.eval_fast_monotone::<B>(cand_k.clone());
            let ok_k = if require_margin {
                ok_margin
            } else {
                ok_for_tau(k, ctx.model.mono_out, ctx.sl, ctx.sr)
            };
            let max_k = TierQualityThresholds::for_tier(ALL_TIERS[k]).max_ltas_db;
            let orig_k = cc.orig_ltas_vals[k];
            let cap_k = if require_margin {
                (orig_k + 0.025).max(max_k * 0.72).min(max_k * 0.85)
            } else {
                max_k
            };
            let ltas_k_ok = orig_k > max_k * 0.85
                || (sl_k.ltas_max_db <= cap_k
                    && sr_k.as_ref().is_none_or(|r| r.ltas_max_db <= cap_k));
            if !ok_k
                || !ltas_k_ok
                || (cc.orig_passed[k] && !passes_both(&sl_k, &sr_k))
                || (cc.orig_margin2[k] && !has_2db_margin(k, &sl_k, &sr_k))
            {
                prefix_ok = false;
                break;
            }
            ctx.commit(cand_k, sl_k, sr_k);
        }
        ctx.tau = tau;
        if prefix_ok {
            for (cand_r, sl_c, sr_c) in cc.eval_cands {
                if cc.require_pass && !passes_both(sl_c, sr_c) {
                    continue;
                }
                ctx.sl[tau] = Some(sl_c.clone());
                ctx.sr[tau] = sr_c.clone();
                let ok_tau = if require_margin {
                    ok_for_tau_with_margin(tau, ctx.model.mono_out, ctx.sl, ctx.sr)
                } else {
                    ok_for_tau(tau, ctx.model.mono_out, ctx.sl, ctx.sr)
                };
                if ok_tau {
                    ctx.model.readouts[tau] = cand_r.clone();
                    ctx.model.mirror_mono();
                    *best_ok = Some((cand_r.clone(), sl_c.clone(), sr_c.clone()));
                    return true;
                }
            }
            if ctx.model.tier_os2x[tau] == ctx.model.tier_os2x[tau - 1] {
                let prev_committed = ctx.model.readouts[tau - 1].clone();
                for (cand_r, sl_c, sr_c) in cc.eval_cands {
                    let hit = if require_margin {
                        let mut local_ok = None;
                        try_monotone_blends::<B>(ctx, cand_r, &prev_committed, &mut local_ok);
                        local_ok
                    } else {
                        try_strict_monotone_blends::<B>(
                            ctx,
                            &None,
                            &Some((cand_r.clone(), sl_c.clone(), sr_c.clone())),
                            &prev_committed,
                        )
                    };
                    if let Some((br, bsl, bsr)) = hit
                        && (!cc.require_pass || passes_both(&bsl, &bsr))
                    {
                        ctx.commit(br.clone(), bsl.clone(), bsr.clone());
                        *best_ok = Some((br, bsl, bsr));
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub(in super::super) fn try_corridor_repair<B: TrainerBackend>(
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
    let use_lower_blend = tau >= 2
        && ctx.model.tier_os2x[tau - 1] == ctx.model.tier_os2x[tau - 2]
        && shares_pre_cab(&orig_prev, &ctx.model.readouts[tau - 2], active_ch);
    const PERTURB_STEPS: [(f64, f64); 28] = [
        (0.9985, 0.0040),
        (0.9985, -0.0040),
        (0.9985, 0.0085),
        (0.9985, -0.0085),
        (0.9985, 0.0150),
        (0.9985, -0.0150),
        (0.9985, 0.0240),
        (0.9985, -0.0240),
        (0.9985, 0.0350),
        (0.9985, -0.0350),
        (0.998, 0.0012),
        (0.998, -0.0012),
        (0.996, 0.0024),
        (0.996, -0.0024),
        (0.993, 0.0042),
        (0.993, -0.0042),
        (0.988, 0.0065),
        (0.988, -0.0065),
        (0.983, 0.0105),
        (0.983, -0.0105),
        (0.983, 0.0160),
        (0.983, -0.0160),
        (0.983, 0.0240),
        (0.983, -0.0240),
        (0.984, 0.0340),
        (0.984, -0.0340),
        (0.984, 0.0450),
        (0.984, -0.0450),
    ];
    const FRAC_DELAY_STEPS: [(f64, f64); 13] = [
        (0.9985, 0.012),
        (0.9985, 0.025),
        (0.997, 0.045),
        (0.997, 0.070),
        (0.995, 0.100),
        (0.995, 0.135),
        (0.993, 0.175),
        (0.990, 0.220),
        (0.992, 0.265),
        (0.990, 0.315),
        (0.998, 0.018),
        (0.994, 0.060),
        (0.986, 0.160),
    ];
    let poles = ctx.model.kautz_poles;
    let prev_tier = ALL_TIERS[tau - 1];
    let mut relaxed_prevs: Vec<V3TierReadoutF64> = FRAC_DELAY_STEPS
        .iter()
        .map(|&(s, d)| frac_delay_and_scale(&orig_prev, &poles, prev_tier, s, d))
        .collect();
    if use_lower_blend {
        let lower_r = ctx.model.readouts[tau - 2].clone();
        relaxed_prevs.extend(
            [0.94, 0.86, 0.76, 0.64, 0.50, 0.35, 0.25, 0.15]
                .iter()
                .map(|&alpha| blend_readout(&orig_prev, &lower_r, alpha)),
        );
        relaxed_prevs.extend(
            PERTURB_STEPS
                .iter()
                .map(|&(s, eps)| perturb_hf_and_scale(&orig_prev, s, eps)),
        );
        relaxed_prevs.extend(
            [0.992, 0.980, 0.965, 0.950, 0.930, 0.905]
                .iter()
                .map(|&s| scale_linear_out(&orig_prev, s)),
        );
    } else {
        relaxed_prevs.extend(
            PERTURB_STEPS
                .iter()
                .map(|&(s, eps)| perturb_hf_and_scale(&orig_prev, s, eps)),
        );
        relaxed_prevs.extend(
            [0.996, 0.988, 0.975, 0.955, 0.925, 0.880]
                .iter()
                .map(|&s| scale_linear_out(&orig_prev, s)),
        );
    }
    let orig_psl = ctx.sl[tau - 1].clone();
    let orig_psr = ctx.sr[tau - 1].clone();
    let prev_thr = TierQualityThresholds::for_tier(prev_tier);
    let prev_ltas_headroom = orig_psl.as_ref().is_some_and(|p| {
        p.ltas_max_db <= prev_thr.max_ltas_db * 0.85
            && orig_psr
                .as_ref()
                .is_none_or(|r| r.ltas_max_db <= prev_thr.max_ltas_db * 0.85)
    });
    let prev_passed = orig_psl.as_ref().is_some_and(|p| passes_both(p, &orig_psr));
    let require_pass = best_ok
        .as_ref()
        .is_some_and(|(_, bsl, bsr)| passes_both(bsl, bsr));
    let prev_ctx = PrevRelaxCtx {
        orig_prev: &orig_prev,
        orig_psl: &orig_psl,
        orig_psr: &orig_psr,
        prev_ltas_headroom,
        prev_passed,
        require_pass,
        max_ltas_db: prev_thr.max_ltas_db,
    };
    if try_relaxed_prevs_pass::<B>(ctx, &relaxed_prevs, cands, &prev_ctx, true, best_ok) {
        return true;
    }

    const CASCADE_SCHEDULE: [(f64, f64); 15] = [
        (0.0015, 0.0015),
        (0.0015, -0.0015),
        (0.0025, 0.0035),
        (0.0025, -0.0035),
        (0.0030, 0.008),
        (0.0030, -0.008),
        (0.0080, 0.022),
        (0.0080, -0.022),
        (0.0160, 0.055),
        (0.0160, -0.055),
        (0.0010, 0.0),
        (0.0040, 0.0),
        (0.0120, 0.0),
        (0.0280, 0.0),
        (0.0480, 0.0),
    ];

    let orig_prefix_r: Vec<V3TierReadoutF64> =
        (0..tau).map(|k| ctx.model.readouts[k].clone()).collect();
    let orig_prefix_sl: Vec<Option<PerceptualCertificationSummary>> =
        (0..tau).map(|k| ctx.sl[k].clone()).collect();
    let orig_prefix_sr: Vec<Option<PerceptualCertificationSummary>> =
        (0..tau).map(|k| ctx.sr[k].clone()).collect();
    let orig_tau_r = ctx.model.readouts[tau].clone();
    let orig_tau_sl = ctx.sl[tau].clone();
    let orig_tau_sr = ctx.sr[tau].clone();

    let eval_cands: Vec<TierCandidate> = cands
        .iter()
        .map(|&c| {
            let (sl, sr, _) = ctx.eval::<B>(c.clone());
            (c.clone(), sl, sr)
        })
        .collect();

    let orig_passed: Vec<bool> = (0..tau)
        .map(|k| {
            orig_prefix_sl[k]
                .as_ref()
                .is_some_and(|p| passes_both(p, &orig_prefix_sr[k]))
        })
        .collect();
    let cand_clears_margin = |k: usize| {
        let lim = TierQualityThresholds::for_tier(ALL_TIERS[k]).composite_besr_db - 2.5;
        eval_cands.iter().any(|(_, sl_c, sr_c)| {
            passes_both(sl_c, sr_c)
                && sl_c.besr_db <= lim
                && sr_c.as_ref().is_none_or(|r| r.besr_db <= lim)
        })
    };
    let orig_margin2: Vec<bool> = (0..tau)
        .map(|k| {
            cand_clears_margin(k)
                && orig_prefix_sl[k]
                    .as_ref()
                    .is_some_and(|p| has_2db_margin(k, p, &orig_prefix_sr[k]))
        })
        .collect();
    let orig_ltas_vals: Vec<f64> = (0..tau)
        .map(|k| {
            orig_prefix_sl[k].as_ref().map_or(0.0, |p| {
                p.ltas_max_db
                    .max(orig_prefix_sr[k].as_ref().map_or(0.0, |r| r.ltas_max_db))
            })
        })
        .collect();
    let cascade_ctx = CascadeCtx {
        orig_prefix_r: &orig_prefix_r,
        orig_prefix_sl: &orig_prefix_sl,
        orig_prefix_sr: &orig_prefix_sr,
        orig_passed: &orig_passed,
        orig_margin2: &orig_margin2,
        orig_ltas_vals: &orig_ltas_vals,
        eval_cands: &eval_cands,
        require_pass,
    };
    let hf_steps: Vec<PrefixStepKind> = CASCADE_SCHEDULE
        .iter()
        .map(|&(d, e)| PrefixStepKind::HfScale(d, e))
        .collect();
    if try_cascade_schedule::<B>(ctx, &cascade_ctx, &hf_steps, true, best_ok) {
        return true;
    }

    if best_ok.is_none() {
        restore_cascade_prefix(ctx, &cascade_ctx, tau);
        if try_relaxed_prevs_pass::<B>(ctx, &relaxed_prevs, cands, &prev_ctx, false, best_ok) {
            return true;
        }
        let mut strict_steps: Vec<PrefixStepKind> =
            [(0.996, 0.06), (0.994, 0.12), (0.992, 0.18), (0.990, 0.24)]
                .iter()
                .map(|&(s, d)| PrefixStepKind::FracDelayScale(s, d))
                .collect();
        strict_steps.extend(
            CASCADE_SCHEDULE
                .iter()
                .map(|&(d, e)| PrefixStepKind::HfScale(d, e)),
        );
        if try_cascade_schedule::<B>(ctx, &cascade_ctx, &strict_steps, false, best_ok) {
            return true;
        }
    }

    restore_cascade_prefix(ctx, &cascade_ctx, tau);
    ctx.model.readouts[tau] = orig_tau_r;
    ctx.sl[tau] = orig_tau_sl;
    ctx.sr[tau] = orig_tau_sr;
    ctx.model.mirror_mono();
    false
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use dnam_engine::stages::{V3CoreInstance, V3MatryoshkaModel};
    use dnam_engine::types::Tier;

    use super::*;
    use crate::metrics::certification::TargetChannelCertCache;
    use crate::metrics::companion::canonical_ltas_33_centers_hz;
    use crate::solver::backends::DefaultTrainerBackend;
    use crate::solver::v3_varpro::model_f64::{V3MatryoshkaOptF64, V3TrainStem};

    fn make_stem(len: usize) -> V3TrainStem {
        let centers = canonical_ltas_33_centers_hz();
        let mut x = vec![0.0_f32; len];
        for (i, slot) in x.iter_mut().enumerate() {
            let t = (i as f64) / 48_000.0;
            let mut sum = 0.0_f64;
            for (k, &hz) in centers.iter().enumerate() {
                sum += (2.0 * PI * hz * t + 0.37 * (k as f64)).sin();
            }
            *slot = (sum * 0.04) as f32;
        }
        let cold = V3MatryoshkaModel::identity(true, true).extract_tier(Tier::Full);
        let mut inst = V3CoreInstance::<DefaultTrainerBackend>::from_cold_model(&cold);
        let (mut tgt_l, mut tgt_r) = (vec![0.0_f32; len], vec![0.0_f32; len]);
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
    fn test_scale_linear_out_and_corridor_repair_paths() {
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
        let scaled = scale_linear_out(&r, 0.5);
        assert!((scaled.svf_direct_taps_l[0] - 0.5).abs() < 1.0e-12);
        assert!((scaled.direct_s2_u_gain[0] - 0.25).abs() < 1.0e-12);
        assert!((scaled.direct_s2_w[0][0] - 0.4).abs() < 1.0e-12);
        assert!((scaled.fir_r[0] - 0.5).abs() < 1.0e-12);
        assert!((scaled.kautz_c_l[0].0 - 0.2).abs() < 1.0e-12);

        let stem = make_stem(4_800);
        let n0 = stem.usable_len();
        let tgt_l: Vec<f64> = stem.tgt_l[..n0].iter().map(|&v| f64::from(v)).collect();
        let mut model =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, true));
        let (mut sl, mut sr) = ([None, None, None, None], [None, None, None, None]);
        let mut ctx = PolishEvalCtx {
            model: &mut model,
            tau: 0,
            stem0: &stem,
            cert_l: TargetChannelCertCache::new(&tgt_l, 48_000.0, stem.warmup.min(n0)),
            cert_r: None,
            sl: &mut sl,
            sr: &mut sr,
        };
        let id_r = V3TierReadoutF64::default();
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

        for (t, r_t) in [(0, id_r.clone()), (1, id_r.clone())] {
            ctx.tau = t;
            let (s_t, _, _) = ctx.eval::<DefaultTrainerBackend>(r_t.clone());
            ctx.commit(r_t, s_t, None);
        }
        ctx.tau = 2;
        ctx.model.tier_os2x = [true, true, true, true];
        let mut pro_r = id_r.clone();
        pro_r.fir_l[0] = 0.996;
        assert!(try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&pro_r],
            &mut corr_ok
        ));
        assert!(corr_ok.is_some());

        ctx.tau = 0;
        let (s_e, _, _) = ctx.eval::<DefaultTrainerBackend>(id_r.clone());
        ctx.commit(id_r.clone(), s_e, None);
        ctx.tau = 1;
        let mut overshoot = id_r.clone();
        overshoot.fir_l[0] = 1.02;
        assert!(try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&overshoot],
            &mut corr_ok
        ));
        ctx.tau = 0;
        let (s_e2, _, _) = ctx.eval::<DefaultTrainerBackend>(id_r.clone());
        ctx.commit(id_r.clone(), s_e2, None);
        ctx.tau = 1;
        assert!(try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&id_r],
            &mut corr_ok
        ));
        let mut bad = id_r.clone();
        bad.fir_l[0] = 0.0;
        assert!(!try_corridor_repair::<DefaultTrainerBackend>(
            &mut ctx,
            &[&bad],
            &mut corr_ok
        ));
    }
}
