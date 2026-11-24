//! Bottom-up Stage B monotonic repair strategies for Pass-5 adjacent-tier enforcement
//! (`ref[train.pipeline.host-parity-and-export]`, `ref[qual.tiers.benchmark-matrix]`).

use super::super::super::model::SonaModelF64;
use super::helpers::{
    MonotoneEvalCtx, MonotoneState, RepairBaseline, blend_override_toward,
    blur_lower_tier_override, commit_dirty_eval, damp_lower_tier_detail, damp_lower_tier_override,
    others_intact, re_eval_dirty, strict_besr_preserved, tier_lufs_and_trim_ok,
};

pub(super) fn try_stage_b_bottom_up(
    model: &mut SonaModelF64,
    eval: &MonotoneEvalCtx<'_>,
    base: &RepairBaseline<'_>,
    state: &mut MonotoneState,
) -> bool {
    let MonotoneEvalCtx { config, .. } = *eval;
    const BETAS_BLEND: [f64; 11] = [
        0.97, 0.94, 0.90, 0.85, 0.78, 0.70, 0.60, 0.45, 0.30, 0.15, 0.0,
    ];
    const BETAS_DAMP_DETAIL: [f64; 12] = [
        0.97, 0.94, 0.90, 0.85, 0.78, 0.70, 0.60, 0.50, 0.40, 0.25, 0.10, 0.0,
    ];
    const BETAS_DAMP: [f64; 12] = [
        0.97, 0.94, 0.90, 0.85, 0.78, 0.70, 0.60, 0.50, 0.35, 0.20, 0.10, 0.0,
    ];
    const BETAS_COORD: [f64; 10] = [0.92, 0.85, 0.78, 0.70, 0.60, 0.45, 0.30, 0.18, 0.08, 0.0];
    const GAMMAS_BLUR: [f64; 8] = [0.04, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.92];
    let mut any_changed = false;

    for t in 1..=3 {
        let ultra_beats_prev = t < 3
            && state.cert.tier_reports[3].besr_db < state.cert.tier_reports[t - 1].besr_db - 1.0;
        if state.cert.pair_monotone[t]
            || (!ultra_beats_prev
                && state.cert.tier_reports[t].besr_db
                    >= state.cert.tier_reports[t - 1].besr_db + 1.5)
        {
            continue;
        }
        let mut repaired = false;

        // B1: Blend tier t-1 override toward t-2.
        if t >= 2 {
            let lo_cur = model.tier_overrides[t - 1]
                .as_deref()
                .cloned()
                .unwrap_or_else(|| base.overrides[t - 1].clone());
            let prev_lo = model.tier_overrides[t - 2]
                .as_deref()
                .cloned()
                .unwrap_or_else(|| base.overrides[t - 2].clone());
            for &beta in &BETAS_BLEND {
                let mut cand_model = model.clone();
                if let Some(ref mut dst) = cand_model.tier_overrides[t - 1] {
                    blend_override_toward(dst, &lo_cur, &prev_lo, beta);
                }
                let Ok(res) = re_eval_dirty(&mut cand_model, eval, state, 1_u8 << (t - 1)) else {
                    continue;
                };
                let c = &res.cert;
                if c.pair_monotone[t]
                    && others_intact(&state.cert, c, t)
                    && strict_besr_preserved(base.besr, c, t)
                    && strict_besr_preserved(base.besr, c, t - 1)
                    && tier_lufs_and_trim_ok(
                        &cand_model,
                        base.trims,
                        c,
                        config,
                        (t - 1)..=(t - 1),
                        true,
                    )
                    && commit_dirty_eval(state, res).is_ok()
                {
                    *model = cand_model;
                    any_changed = true;
                    repaired = true;
                    break;
                }
            }
        }

        // B2: Damp upper detail or full readout/FIR/Kautz weights on tier t-1.
        if !repaired {
            for (detail_only, strict_env, betas) in [
                (true, true, BETAS_DAMP_DETAIL.as_slice()),
                (false, false, BETAS_DAMP.as_slice()),
            ] {
                if repaired {
                    break;
                }
                for &beta in betas {
                    let mut cand_model = model.clone();
                    if let Some(ref mut dst) = cand_model.tier_overrides[t - 1] {
                        if detail_only {
                            damp_lower_tier_detail(dst, t - 1, beta);
                        } else {
                            damp_lower_tier_override(dst, t - 1, beta);
                        }
                    }
                    let Ok(res) = re_eval_dirty(&mut cand_model, eval, state, 1_u8 << (t - 1))
                    else {
                        continue;
                    };
                    let c = &res.cert;
                    if c.pair_monotone[t]
                        && others_intact(&state.cert, c, t)
                        && strict_besr_preserved(base.besr, c, t)
                        && (t == 1 || strict_besr_preserved(base.besr, c, t - 1))
                        && tier_lufs_and_trim_ok(
                            &cand_model,
                            base.trims,
                            c,
                            config,
                            (t - 1)..=(t - 1),
                            strict_env,
                        )
                        && commit_dirty_eval(state, res).is_ok()
                    {
                        *model = cand_model;
                        any_changed = true;
                        repaired = true;
                        break;
                    }
                }
            }
        }

        // B3: Coordinated multi-tier lower damping on 0..t.
        if !repaired && t >= 2 {
            for (detail_only, strict_env) in [(true, true), (false, false)] {
                if repaired {
                    break;
                }
                for &beta in &BETAS_COORD {
                    let mut cand_model = model.clone();
                    for k in 0..t {
                        let scale = beta.powi((t - k) as i32);
                        if let Some(ref mut dst) = cand_model.tier_overrides[k] {
                            if detail_only {
                                damp_lower_tier_detail(dst, k, scale);
                            } else {
                                damp_lower_tier_override(dst, k, scale);
                            }
                        }
                    }
                    let Ok(res) = re_eval_dirty(&mut cand_model, eval, state, (1_u8 << t) - 1)
                    else {
                        continue;
                    };
                    let c = &res.cert;
                    let prefix_monotone = c.pair_monotone[1..=t].iter().all(|&m| m);
                    let upper_ok =
                        ((t + 1)..=3).all(|u| !state.cert.pair_monotone[u] || c.pair_monotone[u]);
                    let strict_ok = (1..=t).all(|k| strict_besr_preserved(base.besr, c, k));
                    if prefix_monotone
                        && upper_ok
                        && strict_ok
                        && tier_lufs_and_trim_ok(
                            &cand_model,
                            base.trims,
                            c,
                            config,
                            0..=(t - 1),
                            strict_env,
                        )
                        && commit_dirty_eval(state, res).is_ok()
                    {
                        *model = cand_model;
                        any_changed = true;
                        repaired = true;
                        break;
                    }
                }
            }
        }

        // B4: Causal FIR lag blur + direct HPF tilt on lower tiers 0..t so scalar LUFS
        //     trimming cannot undo broadband attenuation while keeping Ultra 100% untouched.
        if !repaired {
            for &gamma in &GAMMAS_BLUR {
                let mut cand_model = model.clone();
                for k in 0..t {
                    let gamma_k = (gamma * ((t - k) as f64) / (t as f64)).min(0.95);
                    if let Some(ref mut dst) = cand_model.tier_overrides[k] {
                        blur_lower_tier_override(dst, k, gamma_k);
                    }
                }
                let Ok(res) = re_eval_dirty(&mut cand_model, eval, state, (1_u8 << t) - 1) else {
                    continue;
                };
                let c = &res.cert;
                let prefix_monotone = c.pair_monotone[1..=t].iter().all(|&m| m);
                let upper_ok =
                    ((t + 1)..=3).all(|u| !state.cert.pair_monotone[u] || c.pair_monotone[u]);
                let strict_ok = (1..=t).all(|k| strict_besr_preserved(base.besr, c, k));
                if prefix_monotone
                    && upper_ok
                    && strict_ok
                    && tier_lufs_and_trim_ok(&cand_model, base.trims, c, config, 0..=(t - 1), false)
                    && commit_dirty_eval(state, res).is_ok()
                {
                    *model = cand_model;
                    any_changed = true;
                    break;
                }
            }
        }
    }
    any_changed
}
