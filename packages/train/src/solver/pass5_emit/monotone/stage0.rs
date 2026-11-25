//! Stage-0 readout-only monotone repair preserving single-nested `f32` convolver precision
//! when adjusting `w_ro` / `direct_s2` suffices (`ref[train.pipeline.host-parity-and-export]`).

use super::super::super::model::{SonaModelF64, TierReadoutPrefixF64};
use super::super::certify::{StereoCertBundle, TierRenderBundle};
use super::helpers::{
    MonotoneEvalCtx, MonotoneState, blend_w_ro_and_direct_only, commit_dirty_eval,
    damp_w_ro_and_direct_only, materialize_w_ro_only_overrides, others_intact, re_eval,
    re_eval_dirty, strict_besr_preserved, tier_lufs_and_trim_ok,
};

pub(super) fn try_stage_0_fp32_w_ro_only(
    model: &mut SonaModelF64,
    eval: &MonotoneEvalCtx<'_>,
    orig_trims: &[f64; 4],
    cert: &StereoCertBundle,
) -> Option<(Vec<u8>, TierRenderBundle, StereoCertBundle)> {
    let MonotoneEvalCtx {
        in_l,
        in_r,
        target_l,
        target_r,
        warmup,
        config,
        ..
    } = *eval;
    if model.prefix_override_needed[1..].iter().any(|&v| v) {
        return None;
    }
    let mut cand = model.clone();
    materialize_w_ro_only_overrides(&mut cand);
    let Ok((c_bin, c_bdl, c_cert)) =
        re_eval(&mut cand, in_l, in_r, target_l, target_r, warmup, config)
    else {
        return None;
    };
    let mut c_state = MonotoneState {
        final_bin: c_bin,
        final_bundle: c_bdl,
        cert: c_cert,
    };
    if c_state.cert.pair_monotone[1..].iter().all(|&m| m)
        && c_state.cert.tier_reports[3].besr_db <= cert.tier_reports[3].besr_db + 0.1
        && tier_lufs_and_trim_ok(&cand, orig_trims, &c_state.cert, config, 0..=2, true)
    {
        *model = cand;
        return Some((c_state.final_bin, c_state.final_bundle, c_state.cert));
    }

    let init_besr: [f64; 4] = core::array::from_fn(|t| c_state.cert.tier_reports[t].besr_db);
    let init_ovs: [TierReadoutPrefixF64; 4] = core::array::from_fn(|t| {
        cand.tier_overrides[t]
            .as_deref()
            .cloned()
            .unwrap_or_default()
    });

    for t in 1..=3 {
        if c_state.cert.pair_monotone[t] {
            continue;
        }
        let hi_src = cand.tier_overrides[t]
            .as_deref()
            .cloned()
            .unwrap_or_else(|| init_ovs[t].clone());
        let lo_src = cand.tier_overrides[t - 1]
            .as_deref()
            .cloned()
            .unwrap_or_else(|| init_ovs[t - 1].clone());
        let strictly_better =
            c_state.cert.tier_reports[t].besr_db < c_state.cert.tier_reports[t - 1].besr_db;
        for &alpha in &[0.95, 0.88, 0.78, 0.65, 0.50] {
            let mut m_try = cand.clone();
            if let Some(ref mut dst) = m_try.tier_overrides[t] {
                blend_w_ro_and_direct_only(dst, &hi_src, &lo_src, alpha);
            }
            let Ok(res) = re_eval_dirty(&mut m_try, eval, &c_state, 1_u8 << t) else {
                continue;
            };
            let c = &res.cert;
            let acc_ok = !strictly_better
                || (c.tier_reports[t].besr_db <= init_besr[t] + 0.5
                    && c.tier_reports[t].besr_db < c.tier_reports[t - 1].besr_db);
            if c.pair_monotone[t]
                && others_intact(&c_state.cert, c, t)
                && acc_ok
                && tier_lufs_and_trim_ok(&m_try, orig_trims, c, config, t..=t, true)
                && commit_dirty_eval(&mut c_state, res).is_ok()
            {
                cand = m_try;
                break;
            }
        }
    }

    const W_RO_DETAIL_BETAS: [f64; 6] = [0.94, 0.85, 0.70, 0.50, 0.25, 0.0];
    const W_RO_FULL_BETAS: [f64; 9] = [0.95, 0.90, 0.82, 0.72, 0.60, 0.45, 0.30, 0.15, 0.0];
    const W_RO_COORD_BETAS: [f64; 8] = [0.92, 0.85, 0.75, 0.60, 0.45, 0.30, 0.15, 0.0];
    for t in 1..=3 {
        if c_state.cert.pair_monotone[t] {
            continue;
        }
        let mut repaired = false;
        for (detail_only, betas) in [
            (true, W_RO_DETAIL_BETAS.as_slice()),
            (false, W_RO_FULL_BETAS.as_slice()),
        ] {
            if repaired {
                break;
            }
            let strict_env = detail_only;
            for &beta in betas {
                let mut m_try = cand.clone();
                if let Some(ref mut dst) = m_try.tier_overrides[t - 1] {
                    damp_w_ro_and_direct_only(dst, t - 1, beta, detail_only);
                }
                let Ok(res) = re_eval_dirty(&mut m_try, eval, &c_state, 1_u8 << (t - 1)) else {
                    continue;
                };
                let c = &res.cert;
                if c.pair_monotone[t]
                    && others_intact(&c_state.cert, c, t)
                    && strict_besr_preserved(&init_besr, c, t)
                    && (t == 1 || strict_besr_preserved(&init_besr, c, t - 1))
                    && tier_lufs_and_trim_ok(
                        &m_try,
                        orig_trims,
                        c,
                        config,
                        (t - 1)..=(t - 1),
                        strict_env,
                    )
                    && commit_dirty_eval(&mut c_state, res).is_ok()
                {
                    cand = m_try;
                    repaired = true;
                    break;
                }
            }
        }
    }

    for t in 2..=3 {
        if c_state.cert.pair_monotone[t]
            || c_state.cert.tier_reports[t].besr_db
                >= c_state.cert.tier_reports[t - 1].besr_db + 1.5
        {
            continue;
        }
        let mut repaired = false;
        for (detail_only, strict_env) in [(true, true), (false, false)] {
            if repaired {
                break;
            }
            for &beta in &W_RO_COORD_BETAS {
                let mut m_try = cand.clone();
                for k in 0..t {
                    let scale = beta.powi((t - k) as i32);
                    if let Some(ref mut dst) = m_try.tier_overrides[k] {
                        damp_w_ro_and_direct_only(dst, k, scale, detail_only);
                    }
                }
                let Ok(res) = re_eval_dirty(&mut m_try, eval, &c_state, (1_u8 << t) - 1) else {
                    continue;
                };
                let c = &res.cert;
                let prefix_monotone = c.pair_monotone[1..=t].iter().all(|&m| m);
                let upper_ok =
                    ((t + 1)..=3).all(|u| !c_state.cert.pair_monotone[u] || c.pair_monotone[u]);
                let strict_ok = (1..=t).all(|k| strict_besr_preserved(&init_besr, c, k));
                if prefix_monotone
                    && upper_ok
                    && strict_ok
                    && tier_lufs_and_trim_ok(&m_try, orig_trims, c, config, 0..=(t - 1), strict_env)
                    && commit_dirty_eval(&mut c_state, res).is_ok()
                {
                    cand = m_try;
                    repaired = true;
                    break;
                }
            }
        }
    }

    if c_state.cert.pair_monotone[1..].iter().all(|&m| m)
        && c_state.cert.tier_reports[3].besr_db <= cert.tier_reports[3].besr_db + 0.1
        && tier_lufs_and_trim_ok(&cand, orig_trims, &c_state.cert, config, 0..=2, false)
    {
        *model = cand;
        return Some((c_state.final_bin, c_state.final_bundle, c_state.cert));
    }
    None
}
