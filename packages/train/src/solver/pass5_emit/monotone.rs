//! Non-destructive Pass-5 adjacent-tier monotone enforcement
//! (`ref[train.pipeline.host-parity-and-export]`, `ref[qual.tiers.benchmark-matrix]`).
//!
//! Preserves `f32` convolver precision when readout-only repairs suffice, isolates per-tier
//! overrides when strict sub-metric checks fail (e.g., 1 of 33 LTAS bands on `Full` vs `Lite`),
//! and protects `Pro`/`Ultra` Kautz poles, `FIR` taps, and inter-block tiles whenever higher
//! tiers strictly outperform lower tiers.

mod helpers;
mod stage0;
mod stage_b;

use crate::error::TrainError;

use super::super::model::{SonaModelF64, TierReadoutPrefixF64};
use super::certify::zero_tier_delta_weights;
use helpers::{
    DirtyEvalResult, RepairBaseline, blend_override_toward, commit_dirty_eval,
    materialize_tier_overrides, others_intact, pair_deficit_score, pass_count, re_eval,
    re_eval_dirty, snap_override_fp16_slice, tier_lufs_and_trim_ok, zero_tier_delta_readouts_only,
};
pub(super) use helpers::{MonotoneEvalCtx, MonotoneState};
use stage_b::try_stage_b_bottom_up;
use stage0::try_stage_0_fp32_w_ro_only;

fn try_stage_a_blend(
    model: &mut SonaModelF64,
    eval: &MonotoneEvalCtx<'_>,
    base: &RepairBaseline<'_>,
    state: &mut MonotoneState,
) -> bool {
    let MonotoneEvalCtx { config, .. } = *eval;
    let mut any_changed = false;
    for t in 1..=3 {
        if state.cert.pair_monotone[t] {
            continue;
        }
        let hi_src = model.tier_overrides[t]
            .as_deref()
            .cloned()
            .unwrap_or_else(|| base.overrides[t].clone());
        let lo_src = model.tier_overrides[t - 1]
            .as_deref()
            .cloned()
            .unwrap_or_else(|| base.overrides[t - 1].clone());
        let ultra_beats_prev = t < 3
            && state.cert.tier_reports[3].besr_db < state.cert.tier_reports[t - 1].besr_db - 1.0;
        let strictly_better =
            state.cert.tier_reports[t].besr_db < state.cert.tier_reports[t - 1].besr_db - 0.5;
        let alphas: &[f64] = if strictly_better {
            &[0.98, 0.95, 0.90, 0.85, 0.78]
        } else {
            &[0.92, 0.85, 0.70, 0.50, 0.25, 0.10, 0.0]
        };
        let mut solved = false;
        let mut best_deficit = pair_deficit_score(&state.cert, t);
        let mut best_partial: Option<DirtyEvalResult> = None;
        let mut best_partial_model: Option<SonaModelF64> = None;
        for &alpha in alphas {
            let mut cand_model = model.clone();
            if let Some(ref mut dst) = cand_model.tier_overrides[t] {
                blend_override_toward(dst, &hi_src, &lo_src, alpha);
            }
            let Ok(res) = re_eval_dirty(&mut cand_model, eval, state, 1_u8 << t) else {
                continue;
            };
            let c = &res.cert;
            let acc_ok = !strictly_better
                || (c.tier_reports[t].besr_db <= base.besr[t] + 0.5
                    && c.tier_reports[t].besr_db < c.tier_reports[t - 1].besr_db);
            if c.pair_monotone[t]
                && others_intact(&state.cert, c, t)
                && acc_ok
                && tier_lufs_and_trim_ok(&cand_model, base.trims, c, config, t..=t, true)
            {
                if commit_dirty_eval(state, res).is_ok() {
                    *model = cand_model;
                    any_changed = true;
                    solved = true;
                }
                break;
            }
            if !strictly_better && ultra_beats_prev {
                let def = pair_deficit_score(&res.cert, t);
                let ultra_ok = res.cert.tier_reports[3].besr_db <= base.besr[3] + 0.5
                    || (res.cert.tier_reports[3].besr_db < -50.0
                        && res.cert.tier_reports[3].besr_db
                            < res.cert.tier_reports[t - 1].besr_db - 1.0);
                if def + 1.0e-6 < best_deficit
                    && others_intact(&state.cert, &res.cert, t)
                    && ultra_ok
                    && tier_lufs_and_trim_ok(
                        &cand_model,
                        base.trims,
                        &res.cert,
                        config,
                        t..=t,
                        false,
                    )
                {
                    best_deficit = def;
                    best_partial_model = Some(cand_model);
                    best_partial = Some(res);
                }
            }
        }
        if !solved
            && let (Some(pm), Some(pres)) = (best_partial_model, best_partial)
            && commit_dirty_eval(state, pres).is_ok()
        {
            *model = pm;
            any_changed = true;
        }
    }
    any_changed
}

fn try_stage_c_readout_fallback(
    model: &mut SonaModelF64,
    eval: &MonotoneEvalCtx<'_>,
    orig_trims: &[f64; 4],
    orig_overrides: &[TierReadoutPrefixF64; 4],
    state: &mut MonotoneState,
) -> bool {
    let MonotoneEvalCtx {
        in_l,
        in_r,
        target_l,
        target_r,
        warmup,
        config,
        ..
    } = *eval;
    let mut any_changed = false;
    if state.cert.tier_reports[3].besr_db >= state.cert.tier_reports[0].besr_db - 0.5 {
        return false;
    }
    for t in 1..=2 {
        if state.cert.pair_monotone[t] {
            continue;
        }
        let mut lo_src = model.tier_overrides[t - 1]
            .as_deref()
            .cloned()
            .unwrap_or_else(|| orig_overrides[t - 1].clone());
        let mut cand_model = model.clone();
        if t == 1 {
            snap_override_fp16_slice(&mut lo_src, 16..32, 0..8, 4..8);
            if let Some(ref mut prev_mut) = cand_model.tier_overrides[0] {
                snap_override_fp16_slice(prev_mut, 16..32, 0..8, 4..8);
            }
        } else {
            snap_override_fp16_slice(&mut lo_src, 16..48, 0..12, 4..12);
            if let Some(ref mut prev_mut) = cand_model.tier_overrides[1] {
                snap_override_fp16_slice(prev_mut, 16..48, 0..12, 4..12);
            }
        }
        cand_model.tier_overrides[t] = Some(Box::new(lo_src));
        zero_tier_delta_readouts_only(&mut cand_model, t);

        let Ok((b, bdl, c)) = re_eval(
            &mut cand_model,
            in_l,
            in_r,
            target_l,
            target_r,
            warmup,
            config,
        ) else {
            continue;
        };
        let improves = pass_count(&c) > pass_count(&state.cert)
            || (pass_count(&c) == pass_count(&state.cert)
                && others_intact(&state.cert, &c, t)
                && pair_deficit_score(&c, t) + 1.0e-6 < pair_deficit_score(&state.cert, t));
        let ultra_ok = c.tier_reports[3].besr_db <= state.cert.tier_reports[3].besr_db + 0.1
            || (c.tier_reports[3].besr_db < -50.0
                && c.tier_reports[3].besr_db < c.tier_reports[t - 1].besr_db - 1.0);
        if improves
            && ultra_ok
            && tier_lufs_and_trim_ok(&cand_model, orig_trims, &c, config, t..=t, false)
        {
            *model = cand_model;
            state.final_bin = b;
            state.final_bundle = bdl;
            state.cert = c;
            any_changed = true;
        }
    }
    any_changed
}

/// Repairs non-monotone adjacent tier pairs without cascading weight destruction across higher
/// tiers whenever higher tiers outperform lower tiers, falling back to `zero_tier_delta_weights`
/// only when a tier remains irreparably regressed (`ref[train.pipeline.host-parity-and-export]`).
pub(super) fn enforce_monotone_tiers_non_destructive(
    model: &mut SonaModelF64,
    eval: &MonotoneEvalCtx<'_>,
    mut state: MonotoneState,
) -> Result<MonotoneState, TrainError> {
    if state.cert.pair_monotone[1..].iter().all(|&m| m) {
        return Ok(state);
    }

    let orig_trims = model.tier_gain_trim;
    if let Some((b0, bdl0, c0)) = try_stage_0_fp32_w_ro_only(model, eval, &orig_trims, &state.cert)
    {
        return Ok(MonotoneState {
            final_bin: b0,
            final_bundle: bdl0,
            cert: c0,
        });
    }

    let MonotoneEvalCtx {
        in_l,
        in_r,
        target_l,
        target_r,
        warmup,
        config,
        ..
    } = *eval;
    materialize_tier_overrides(model);
    let mut best_model = model.clone();
    if let Ok((b, bdl, c)) = re_eval(
        &mut best_model,
        in_l,
        in_r,
        target_l,
        target_r,
        warmup,
        config,
    ) {
        *model = best_model;
        state.final_bin = b;
        state.final_bundle = bdl;
        state.cert = c;
    }

    let orig_besr: [f64; 4] = core::array::from_fn(|t| state.cert.tier_reports[t].besr_db);
    let orig_trims = model.tier_gain_trim;
    let orig_overrides: [TierReadoutPrefixF64; 4] = core::array::from_fn(|t| {
        model.tier_overrides[t]
            .as_deref()
            .cloned()
            .unwrap_or_default()
    });
    let base = RepairBaseline {
        besr: &orig_besr,
        trims: &orig_trims,
        overrides: &orig_overrides,
    };

    for _round in 0..3 {
        if state.cert.pair_monotone[1..].iter().all(|&m| m) {
            return Ok(state);
        }

        let mut round_changed = try_stage_a_blend(model, eval, &base, &mut state);

        if state.cert.pair_monotone[1..].iter().all(|&m| m) {
            return Ok(state);
        }

        round_changed |= try_stage_b_bottom_up(model, eval, &base, &mut state);

        if state.cert.pair_monotone[1..].iter().all(|&m| m) {
            return Ok(state);
        }

        round_changed |=
            try_stage_c_readout_fallback(model, eval, &orig_trims, &orig_overrides, &mut state);
        if !round_changed {
            break;
        }
    }

    // Stage D: Protect Ultra tiles/Kautz before falling back to full zero_tier_delta_weights.
    for pass in 0..4 {
        let mut changed = false;
        for t in 1..=3 {
            if !state.cert.pair_monotone[t] {
                if pass == 0
                    && t < 3
                    && state.cert.tier_reports[3].besr_db
                        < state.cert.tier_reports[t - 1].besr_db - 1.0
                {
                    zero_tier_delta_readouts_only(model, t);
                } else {
                    zero_tier_delta_weights(model, t);
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let (b, bdl, c) = re_eval(model, in_l, in_r, target_l, target_r, warmup, config)?;
        state.final_bin = b;
        state.final_bundle = bdl;
        state.cert = c;
    }

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::super::Pass5EmitConfig;
    use super::super::certify::{StereoCertBundle, apply_full_quad_phase3_audit, render_all_tiers};
    use super::*;
    use crate::losses::check_adjacent_tier_monotone;
    use crate::solver::pass5_emit::encode_model_to_dnam_bin;
    use helpers::materialize_w_ro_only_overrides;

    fn run_monotone_test(
        model: &mut SonaModelF64,
        in_l: &[f64],
        in_r: &[f64],
        target_l: &[f64],
        target_r: &[f64],
        config: Pass5EmitConfig,
    ) -> StereoCertBundle {
        let gap_masks = [
            crate::losses::DrySilentGapMask::detect(in_l, target_l, 64, model.sample_rate_hz),
            crate::losses::DrySilentGapMask::detect(in_r, target_r, 64, model.sample_rate_hz),
        ];
        let eval = MonotoneEvalCtx {
            in_l,
            in_r,
            target_l,
            target_r,
            warmup: 64,
            gap_masks: &gap_masks,
            config,
        };
        let (final_bin, final_bundle, cert) =
            re_eval(model, in_l, in_r, target_l, target_r, 64, config).unwrap();
        enforce_monotone_tiers_non_destructive(
            model,
            &eval,
            MonotoneState {
                final_bin,
                final_bundle,
                cert,
            },
        )
        .unwrap()
        .cert
    }

    #[test]
    fn test_non_destructive_monotone_repair_preserves_ultra_kautz_and_recovers_exploding_tier() {
        let n = 1_024_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| 0.3 * ((i as f64) * 0.08).sin() + 0.15 * ((i as f64) * 0.35).cos())
            .collect();
        let in_r = in_l.clone();
        let mut z1 = 0.0_f64;
        let mut tgt_l = vec![0.0_f64; n];
        for (i, &x) in in_l.iter().enumerate() {
            z1 = 0.85 * z1 + 0.15 * x;
            let tap12 = if i >= 12 { 0.15 * in_l[i - 12] } else { 0.0 };
            tgt_l[i] = 0.55 * x + 0.35 * z1 + tap12;
        }
        let tgt_r = tgt_l.clone();

        let mut model = SonaModelF64::new(sr);
        model.active_fir_taps = 64;
        model.active_kautz_pairs = 16;
        model.direct_s2_u_gain = [0.55, 0.55];
        model.fir_l[12] = 0.15;
        model.fir_r[12] = 0.15;
        for k in 0..16 {
            model.kautz[k].r_low = 0.85;
            model.kautz[k].theta = 0.08 + 0.12 * (k as f64);
            model.kautz[k].b_gamma = 0.52;
            model.kautz[k].c_re_l = if k < 8 { 0.08 } else { 0.04 };
            model.kautz[k].c_re_r = model.kautz[k].c_re_l;
        }
        materialize_tier_overrides(&mut model);
        if let Some(ref mut full_ov) = model.tier_overrides[1] {
            full_ov.direct_s2_hp_gain = [0.65, 0.65];
        }
        let cfg = Pass5EmitConfig::default();
        let _ = apply_full_quad_phase3_audit(&mut model).unwrap();
        let (_init_bin, _init_bundle, init_cert) =
            re_eval(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, 64, cfg).unwrap();
        assert!(!init_cert.pair_monotone[1]);

        let final_cert = run_monotone_test(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, cfg);
        assert!(check_adjacent_tier_monotone(&final_cert.tier_reports));
        assert!(
            model.kautz[12].c_re_l.abs() > 0.01,
            "Ultra Kautz residues above Lite must be preserved: {}",
            model.kautz[12].c_re_l
        );

        let mut explode_model = SonaModelF64::new(sr);
        explode_model.oversample_upper_tiers_2x = false;
        explode_model.direct_s2_u_gain = [0.55, 0.55];
        explode_model.w_ro[32][6] = 3.5;
        let c3 = run_monotone_test(&mut explode_model, &in_l, &in_r, &tgt_l, &tgt_r, cfg);
        assert!(check_adjacent_tier_monotone(&c3.tier_reports));
    }

    #[test]
    fn test_multi_pair_lf_ltas_inversion_preserves_ultra_and_strict_besr_ordering() {
        let n = 1_536_usize;
        let sr = 48_000.0_f64;
        let centers = crate::losses::canonical_ltas_33_centers_hz();
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                let mut s = 0.0_f64;
                for (b, &fc) in centers.iter().enumerate() {
                    s += 0.045 * (core::f64::consts::TAU * fc * t + 0.37 * (b as f64)).sin();
                }
                s
            })
            .collect();
        let in_r = in_l.clone();

        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 128;
        model.active_kautz_pairs = 24;
        model.pre_conv_u_gain = [0.75, 0.75];
        model.direct_s2_u_gain = [0.60, 0.60];
        model.fir_l[10] = 0.10;
        model.fir_r[10] = 0.10;
        model.fir_l[40] = 0.12;
        model.fir_r[40] = 0.12;
        model.fir_l[64] = 0.10;
        model.fir_r[64] = 0.10;
        model.fir_l[100] = 0.10;
        model.fir_r[100] = 0.10;
        for k in 0..24 {
            model.kautz[k].r_low = 0.88;
            model.kautz[k].theta = 0.06 + 0.11 * (k as f64);
            model.kautz[k].b_gamma = 0.45;
            model.kautz[k].c_re_l = 0.025;
            model.kautz[k].c_re_r = 0.025;
        }
        let cfg = Pass5EmitConfig::default();
        let clean_bin = encode_model_to_dnam_bin(&model, cfg).unwrap();
        let clean_renders = render_all_tiers(&clean_bin, &in_l, &in_r, sr as f32).unwrap();
        let (tgt_l, tgt_r) = clean_renders.renders[3].clone();

        materialize_tier_overrides(&mut model);
        if let Some(ref mut full_ov) = model.tier_overrides[1] {
            full_ov.kautz_c_re_l[0] += 0.25;
            full_ov.kautz_c_re_r[0] += 0.25;
        }
        if let Some(ref mut pro_ov) = model.tier_overrides[2] {
            pro_ov.kautz_c_re_l[0] += 0.50;
            pro_ov.kautz_c_re_r[0] += 0.50;
        }
        let _ = apply_full_quad_phase3_audit(&mut model).unwrap();
        let (_init_bin, _init_bundle, init_cert) =
            re_eval(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, 64, cfg).unwrap();
        assert!(!init_cert.pair_monotone[1] && !init_cert.pair_monotone[2]);

        let final_cert = run_monotone_test(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, cfg);
        assert!(check_adjacent_tier_monotone(&final_cert.tier_reports));
        for t in 1..=3 {
            assert_eq!(pair_deficit_score(&final_cert, t), 0.0);
        }
        assert!(final_cert.tier_reports[3].besr_db < final_cert.tier_reports[0].besr_db - 2.0);
        assert!(final_cert.tier_reports[3].besr_db < final_cert.tier_reports[2].besr_db);
    }

    #[test]
    fn test_monotone_repair_preserves_lufs_parity_with_dominant_core_readout() {
        let n = 1_536_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                0.28 * (core::f64::consts::TAU * 220.0 * t).sin()
                    + 0.18 * (core::f64::consts::TAU * 880.0 * t).cos()
            })
            .collect();
        let in_r = in_l.clone();
        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 48;
        model.pre_conv_u_gain = [0.75, 0.75];
        model.direct_s2_u_gain = [0.02, 0.02];
        for l in 0..24 {
            model.w_ro[l][6] = 0.06;
            model.w_ro[l][7] = 0.06;
        }
        model.fir_l[38] = 0.25;
        model.fir_r[38] = 0.25;
        let cfg = Pass5EmitConfig::default();
        let clean_bin = encode_model_to_dnam_bin(&model, cfg).unwrap();
        let clean_renders = render_all_tiers(&clean_bin, &in_l, &in_r, sr as f32).unwrap();
        let (tgt_l, tgt_r) = clean_renders.renders[3].clone();

        materialize_tier_overrides(&mut model);
        if let Some(ref mut lite_ov) = model.tier_overrides[0] {
            lite_ov.fir_l[10] += 0.12;
            lite_ov.fir_r[10] += 0.12;
        }
        if let Some(ref mut full_ov) = model.tier_overrides[1] {
            full_ov.fir_l[40] += 0.35;
            full_ov.fir_r[40] += 0.35;
        }
        let (_init_bin, _init_bundle, init_cert) =
            re_eval(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, 64, cfg).unwrap();
        assert!(!init_cert.pair_monotone[1]);

        let final_cert = run_monotone_test(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, cfg);
        assert!(check_adjacent_tier_monotone(&final_cert.tier_reports));
        for k in 0..3 {
            assert!((final_cert.tier_lufs[k] - final_cert.tier_lufs[3]).abs() <= 0.05);
            assert!((0.25..=4.0).contains(&model.tier_gain_trim[k]));
        }
    }

    #[test]
    fn test_stage0_fp32_w_ro_repair_and_pro_inversion_preserves_ultra_tiles() {
        let n = 1_536_usize;
        let sr = 48_000.0_f64;
        let centers = crate::losses::canonical_ltas_33_centers_hz();
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                centers.iter().enumerate().fold(0.0_f64, |acc, (b, &fc)| {
                    acc + 0.04 * (core::f64::consts::TAU * fc * t + 0.29 * (b as f64)).sin()
                })
            })
            .collect();
        let in_r = in_l.clone();
        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 96;
        model.active_kautz_pairs = 28;
        model.pre_conv_u_gain = [0.7, 0.7];
        model.direct_s2_u_gain = [0.5, 0.5];
        model.fir_l[72] = 0.08;
        model.fir_r[72] = 0.08;
        for q in &mut model.core_quads {
            q.in_mid_weight = [0.3; 4];
            q.tone_dir = [0.4; 4];
        }
        model.w_ro[4][6] = 0.25;
        model.w_ro[4][7] = 0.25;
        let cfg = Pass5EmitConfig::default();
        let clean_bin = encode_model_to_dnam_bin(&model, cfg).unwrap();
        let clean_renders = render_all_tiers(&clean_bin, &in_l, &in_r, sr as f32).unwrap();
        let (tgt_l, tgt_r) = clean_renders.renders[3].clone();

        // Corrupt Lite w_ro only via Stage 0 path (prefix_override_needed = [false; 4]).
        let mut s0_model = model.clone();
        materialize_w_ro_only_overrides(&mut s0_model);
        if let Some(ref mut lite_ov) = s0_model.tier_overrides[0] {
            lite_ov.w_ro[4][6] = -0.35;
            lite_ov.w_ro[4][7] = -0.35;
        }
        let cf = run_monotone_test(&mut s0_model, &in_l, &in_r, &tgt_l, &tgt_r, cfg);
        assert!(check_adjacent_tier_monotone(&cf.tier_reports));
        assert_eq!(s0_model.prefix_override_needed, [false; 4]);

        // Verify 32-pole / 35-tile Full -> Pro inversion preserves Ultra tiles[13..23] and Kautz[12..32].
        let mut tile_model = model;
        tile_model.active_fir_taps = 128;
        tile_model.active_kautz_pairs = 32;
        for tile in &mut tile_model.tiles {
            tile.w_odd_4x4[0][0] = 0.04;
        }
        for k in 0..32 {
            tile_model.kautz[k].r_low = 0.87;
            tile_model.kautz[k].theta = 0.06 + 0.09 * (k as f64);
            tile_model.kautz[k].b_gamma = 0.5;
            tile_model.kautz[k].c_re_l = 0.03;
            tile_model.kautz[k].c_re_r = 0.03;
        }
        let ref_bin = encode_model_to_dnam_bin(&tile_model, cfg).unwrap();
        let ref_renders = render_all_tiers(&ref_bin, &in_l, &in_r, sr as f32).unwrap();
        let (tu_l, tu_r) = ref_renders.renders[3].clone();
        materialize_tier_overrides(&mut tile_model);
        if let Some(ref mut pro_ov) = tile_model.tier_overrides[2] {
            pro_ov.kautz_c_re_l[15] += 0.45;
            pro_ov.kautz_c_re_r[15] += 0.45;
        }
        let saved_tile15 = tile_model.tiles[15].w_odd_4x4[0][0];
        let cf2 = run_monotone_test(&mut tile_model, &in_l, &in_r, &tu_l, &tu_r, cfg);
        assert!(check_adjacent_tier_monotone(&cf2.tier_reports));
        assert_eq!(tile_model.tiles[15].w_odd_4x4[0][0], saved_tile15);
        assert!(tile_model.kautz[26].c_re_l.abs() > 0.01);
        assert!(cf2.tier_reports[3].besr_db < cf2.tier_reports[1].besr_db);
    }

    #[test]
    fn test_stage_b4_blur_resolves_flat_ltas_fir0_tie_and_preserves_ultra_kautz() {
        let n = 1_536_usize;
        let sr = 48_000.0_f64;
        let centers = crate::losses::canonical_ltas_33_centers_hz();
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                centers.iter().enumerate().fold(0.0_f64, |acc, (b, &fc)| {
                    acc + 0.04 * (core::f64::consts::TAU * fc * t + 0.31 * (b as f64)).sin()
                })
            })
            .collect();
        let in_r = in_l.clone();

        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 96;
        model.active_kautz_pairs = 28;
        model.pre_conv_u_gain = [1.0, 1.0];
        model.fir_l[0] = 0.85;
        model.fir_r[0] = 0.85;
        for k in 8..28 {
            model.kautz[k].r_low = 0.88;
            model.kautz[k].theta = 0.08 + 0.095 * (k as f64);
            model.kautz[k].b_gamma = 0.45;
            model.kautz[k].c_re_l = 0.004;
            model.kautz[k].c_re_r = 0.004;
        }
        let cfg = Pass5EmitConfig::default();
        let clean_bin = encode_model_to_dnam_bin(&model, cfg).unwrap();
        let clean_renders = render_all_tiers(&clean_bin, &in_l, &in_r, sr as f32).unwrap();
        let (mut tgt_l, mut tgt_r) = clean_renders.renders[3].clone();
        for ((tl, tr), (&xl, &xr)) in tgt_l.iter_mut().zip(&mut tgt_r).zip(in_l.iter().zip(&in_r)) {
            *tl += 0.15 * xl * xl * xl;
            *tr += 0.15 * xr * xr * xr;
        }

        materialize_tier_overrides(&mut model);
        if let Some(ref mut full_ov) = model.tier_overrides[1] {
            full_ov.kautz_c_im_l[8] = 0.025;
            full_ov.kautz_c_im_r[8] = 0.025;
        }
        let (_init_bin, _init_bundle, init_cert) =
            re_eval(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, 64, cfg).unwrap();
        assert!(!init_cert.pair_monotone[1]);

        let final_cert = run_monotone_test(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, cfg);
        assert!(check_adjacent_tier_monotone(&final_cert.tier_reports));
        for k in 8..28 {
            assert!(
                model.kautz[k].c_re_l.abs() > 1.0e-4,
                "Ultra kautz[{k}] must survive Stage B4 without Stage D wiping: {}",
                model.kautz[k].c_re_l
            );
        }
    }

    #[test]
    fn test_severe_intermediate_regression_preserves_ultra_tiles_and_kautz() {
        let n = 1_024_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| 0.25 * ((i as f64) * 0.07).sin() + 0.12 * ((i as f64) * 0.29).cos())
            .collect();
        let in_r = in_l.clone();
        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 128;
        model.active_kautz_pairs = 32;
        model.pre_conv_u_gain = [0.75, 0.75];
        model.direct_s2_u_gain = [0.25, 0.25];
        model.fir_l[100] = 0.08;
        model.fir_r[100] = 0.08;
        model.tiles[23].w_odd_4x4[0][0] = 0.05;
        model.w_ro[32][6] = 0.12;
        for k in 0..32 {
            model.kautz[k].r_low = 0.87;
            model.kautz[k].theta = 0.06 + 0.09 * (k as f64);
            model.kautz[k].b_gamma = 0.5;
            model.kautz[k].c_re_l = if k < 8 { 0.03 } else { 0.05 };
            model.kautz[k].c_re_r = model.kautz[k].c_re_l;
        }
        let cfg = Pass5EmitConfig::default();
        let clean_bin = encode_model_to_dnam_bin(&model, cfg).unwrap();
        let (tgt_l, tgt_r) = render_all_tiers(&clean_bin, &in_l, &in_r, sr as f32)
            .unwrap()
            .renders[3]
            .clone();

        materialize_tier_overrides(&mut model);
        if let Some(ref mut full_ov) = model.tier_overrides[1] {
            full_ov.kautz_c_re_l[0] += 0.45;
            full_ov.kautz_c_re_r[0] += 0.45;
            full_ov.fir_l[1] += 0.35;
            full_ov.fir_r[1] += 0.35;
        }
        let (_, _, init_cert) = re_eval(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, 64, cfg).unwrap();
        assert!(init_cert.tier_reports[1].besr_db >= init_cert.tier_reports[0].besr_db + 1.5);
        assert!(init_cert.tier_reports[3].besr_db < init_cert.tier_reports[0].besr_db - 1.0);

        let final_cert = run_monotone_test(&mut model, &in_l, &in_r, &tgt_l, &tgt_r, cfg);
        assert!(check_adjacent_tier_monotone(&final_cert.tier_reports));
        assert!(model.kautz[24].c_re_l.abs() > 0.0);
        assert!(model.tiles[23].w_odd_4x4[0][0].abs() > 0.0);
        assert!(final_cert.tier_reports[3].besr_db < final_cert.tier_reports[0].besr_db);
    }
}
