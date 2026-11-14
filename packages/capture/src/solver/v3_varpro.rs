#![forbid(unsafe_code)]
//! `v3` Cache-Coherent System Identification: Closed-Form `VarPro` + Exact Reverse-Mode Adjoint L-BFGS.

pub mod adjoint;
pub mod basis;
pub mod gabor;
pub mod lbfgs;
pub mod model_f64;
pub(crate) mod primitives;
pub mod svf_id;

pub use lbfgs::run_lbfgs;
pub use model_f64::{V3OptModelF64, V3TrainDataset, V3TrainStem};

use crate::metrics::{
    ArchetypeFamily, PerceptualCertificationSummary, certify_blackbox_coreset_chain,
};
use dnam_engine::Tier;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::stages::v3_core::cascade::{
    MAX_CASCADE_LANES, MAX_KAUTZ_POLES, NUM_CASCADE_BLOCKS, tier_cascade_lanes, tier_kautz_banks,
};
use dnam_engine::stages::v3_core::{V3ColdTierModel, V3CoreInstance};

use adjoint::{AdjointWorkspace, evaluate_loss_f64};
use basis::{
    select_omp_kautz_poles, solve_bezout_varpro_cholesky, solve_bezout_varpro_cholesky_f32,
};
use gabor::GaborAdjointScratch;
use svf_id::prescreen_lite_seeds;

/// Evaluates the exact Bézout 3-band `bESR` loss using the real `f32` `V3CoreInstance`.
#[must_use]
pub fn evaluate_cold_loss_f32(cold: &V3ColdTierModel, dataset: &V3TrainDataset) -> f64 {
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(cold);
    let active_ch = if cold.mono_out { 1 } else { 2 };
    let ch_scale = if cold.mono_out { 1.0 } else { 0.5 };
    let mut bands = [Vec::new(), Vec::new(), Vec::new()];
    let mut gab_scratch = GaborAdjointScratch::default();
    let mut total_loss = 0.0_f64;

    for (stem_idx, stem) in dataset.stems.iter().enumerate() {
        let n = stem
            .in_l
            .len()
            .min(stem.in_r.len())
            .min(stem.tgt_l.len())
            .min(stem.tgt_r.len());
        if n == 0 {
            continue;
        }
        let w = stem.warmup.min(n);
        let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        let r_in = if cold.mono_out {
            &stem.in_l[..n]
        } else {
            &stem.in_r[..n]
        };
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(&stem.in_l[..n], r_in, &mut out_l, &mut out_r);
        for ch in 0..active_ch {
            let (out, tgt) = if ch == 0 {
                (&out_l, &stem.tgt_l)
            } else {
                (&out_r, &stem.tgt_r)
            };
            let err: Vec<f64> = (0..n).map(|t| out[t] as f64 - tgt[t] as f64).collect();
            dataset.bank.split_3band(&err, &mut bands);
            let bw = dataset.stem_band_weights[stem_idx][ch];
            for b in 0..3 {
                let eb: f64 = bands[b][w..n].iter().map(|&v| v * v).sum();
                total_loss += ch_scale * bw[b] * eb;
            }
            if dataset.gabor_env_weight > 0.0 || dataset.gabor_ltas_weight > 0.0 {
                let y_f64: Vec<f64> = out[..n].iter().map(|&v| v as f64).collect();
                total_loss += dataset.eval_gabor_loss_and_adjoint(
                    stem_idx,
                    ch,
                    ch_scale,
                    &y_f64,
                    None,
                    &mut gab_scratch,
                );
            }
        }
    }
    total_loss
}

fn accept_varpro_if_improves(
    model: &mut V3OptModelF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    best_loss: &mut f64,
) {
    let backup = model.clone();
    solve_bezout_varpro_cholesky(model, dataset);
    let after = evaluate_loss_f64(model, dataset, ws);
    if after.is_finite() && after <= *best_loss {
        *best_loss = after;
    } else {
        *model = backup;
    }
}

fn lift_opt_model(prev: &V3OptModelF64, next_tier: Tier, next_os2x: bool) -> V3OptModelF64 {
    let mut out = prev.clone();
    out.tier = next_tier;
    out.os2x = next_os2x;
    if !prev.os2x && next_os2x {
        for b in 0..NUM_CASCADE_BLOCKS {
            out.blocks[b].atk_alpha = out.blocks[b].atk_alpha.clamp(0.001, 0.9999).sqrt();
            out.blocks[b].rel_alpha = out.blocks[b].rel_alpha.clamp(0.001, 0.99995).sqrt();
            for l in 0..MAX_CASCADE_LANES {
                if out.blocks[b].lp_pole[l] > 0.0 {
                    out.blocks[b].lp_pole[l] = out.blocks[b].lp_pole[l].clamp(0.02, 0.9985).sqrt();
                }
            }
        }
    }
    out
}

fn seed_new_cascade_group(model: &mut V3OptModelF64, start_lane: usize, end_lane: usize) {
    let end = end_lane.min(MAX_CASCADE_LANES);
    let poles = [0.42_f64, 0.68, 0.88, 0.978];
    for b in 0..NUM_CASCADE_BLOCKS {
        let sign_b = if b.is_multiple_of(2) { 1.0 } else { -1.0 };
        for l in start_lane..end {
            let idx = (l - start_lane) % 4;
            let pair_sign = if idx.is_multiple_of(2) { 1.0 } else { -1.0 };
            model.blocks[b].in_u_gain[l] = 0.04 * sign_b * pair_sign;
            model.blocks[b].lp_pole[l] = poles[idx];
            model.blocks[b].lp_mix[l] = 0.75;
            model.blocks[b].hp_mix[l] = 0.45;
            model.blocks[b].static_bias[l] = 0.12 * pair_sign;
            model.blocks[b].env_bias_sens[l] = 0.08 * pair_sign;
            model.blocks[b].skip_u_gain[l] = 0.06 * pair_sign;
            if b > 0 {
                model.blocks[b].prev_skip_gain[l] = 0.08;
                let src_prev = l - 4;
                model.blocks[b].inter_mat[l][src_prev] = 0.10 * pair_sign;
                model.blocks[b].inter_mat[l][l] = 0.08;
            }
        }
    }
}

fn polish_and_quantize(
    cand: &mut V3OptModelF64,
    ds_polish: &V3TrainDataset,
    ds_eval: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    prev_k: usize,
    next_k: usize,
) -> (V3ColdTierModel, f64) {
    select_omp_kautz_poles(cand, ds_polish, prev_k, next_k);
    let mut cur_loss = evaluate_loss_f64(cand, ds_polish, ws);
    accept_varpro_if_improves(cand, ds_polish, ws, &mut cur_loss);
    cur_loss = run_lbfgs(cand, ds_polish, ws, 55);
    accept_varpro_if_improves(cand, ds_polish, ws, &mut cur_loss);
    run_lbfgs(cand, ds_polish, ws, 30);

    let mut cold = cand.to_cold_model();
    let loss_before = evaluate_cold_loss_f32(&cold, ds_eval);
    let mut cold_vp = cold.clone();
    solve_bezout_varpro_cholesky_f32(&mut cold_vp, ds_eval);
    let loss_after = evaluate_cold_loss_f32(&cold_vp, ds_eval);
    if loss_after.is_finite() && loss_after <= loss_before {
        cold = cold_vp;
        (cold, loss_after)
    } else {
        (cold, loss_before)
    }
}

fn coreset_summary_for_cold(
    cold: &V3ColdTierModel,
    s0: &V3TrainStem,
    sr_hz: f64,
) -> Option<PerceptualCertificationSummary> {
    let n = s0
        .in_l
        .len()
        .min(s0.in_r.len())
        .min(s0.tgt_l.len())
        .min(s0.tgt_r.len());
    if n <= s0.warmup + 1 {
        return None;
    }
    let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(cold);
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    let r_in = if cold.mono_out {
        &s0.in_l[..n]
    } else {
        &s0.in_r[..n]
    };
    inst.reset_to_dc_equilibrium();
    inst.process_block_stereo(&s0.in_l[..n], r_in, &mut out_l, &mut out_r);
    let f64v = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    let sl = certify_blackbox_coreset_chain(
        ArchetypeFamily::CleanPedalPreamp,
        cold.tier,
        &f64v(&s0.tgt_l[..n]),
        &f64v(&out_l),
        sr_hz,
        s0.warmup,
    );
    if cold.mono_out {
        return Some(sl);
    }
    let sr = certify_blackbox_coreset_chain(
        ArchetypeFamily::CleanPedalPreamp,
        cold.tier,
        &f64v(&s0.tgt_r[..n]),
        &f64v(&out_r),
        sr_hz,
        s0.warmup,
    );
    Some(PerceptualCertificationSummary {
        family: sl.family,
        tier: sl.tier,
        raw_besr_db: sl.raw_besr_db.max(sr.raw_besr_db),
        besr_db: sl.besr_db.max(sr.besr_db),
        band_esr_db: std::array::from_fn(|b| sl.band_esr_db[b].max(sr.band_esr_db[b])),
        qualified_band_mask: std::array::from_fn(|b| {
            sl.qualified_band_mask[b] || sr.qualified_band_mask[b]
        }),
        mr_iir: sl.mr_iir.max(sr.mr_iir),
        ltas_max_db: sl.ltas_max_db.max(sr.ltas_max_db),
        ltas_unadjusted_max_db: sl.ltas_unadjusted_max_db.max(sr.ltas_unadjusted_max_db),
        ltas_excluded_bands: std::array::from_fn(|b| {
            sl.ltas_excluded_bands[b] && sr.ltas_excluded_bands[b]
        }),
        ltas_residual_band_power: std::array::from_fn(|b| {
            sl.ltas_residual_band_power[b].max(sr.ltas_residual_band_power[b])
        }),
        seg_esr_db: sl.seg_esr_db.max(sr.seg_esr_db),
        passes_target: sl.passes_target && sr.passes_target,
        passes_hard_ceiling: sl.passes_hard_ceiling && sr.passes_hard_ceiling,
    })
}

fn is_strict_summary_upgrade(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
) -> bool {
    hi.raw_besr_db <= lo.raw_besr_db
        && hi.mr_iir <= lo.mr_iir
        && hi.ltas_max_db <= lo.ltas_max_db
        && hi.seg_esr_db <= lo.seg_esr_db
        && hi.ltas_excluded_bands == lo.ltas_excluded_bands
        && (0..3).all(|b| hi.band_esr_db[b] <= lo.band_esr_db[b])
        && (0..33).all(|b| {
            !hi.ltas_excluded_bands[b]
                || hi.ltas_residual_band_power[b] <= lo.ltas_residual_band_power[b]
        })
}

fn blend_linear_readout(
    base: &V3ColdTierModel,
    target: &V3ColdTierModel,
    alpha: f32,
) -> V3ColdTierModel {
    let mut out = base.clone();
    let l = |a: f32, b: f32| a + alpha * (b - a);
    for i in 0..8 {
        out.svf.direct_taps_l[i] = l(base.svf.direct_taps_l[i], target.svf.direct_taps_l[i]);
        out.svf.direct_taps_r[i] = l(base.svf.direct_taps_r[i], target.svf.direct_taps_r[i]);
    }
    for (ch, row) in out.direct_s2_w.iter_mut().enumerate() {
        for (idx, dst) in row.iter_mut().enumerate() {
            *dst = l(base.direct_s2_w[ch][idx], target.direct_s2_w[ch][idx]);
        }
    }
    for k in 0..64 {
        out.fir_l[k] = l(base.fir_l[k], target.fir_l[k]);
        out.fir_r[k] = l(base.fir_r[k], target.fir_r[k]);
    }
    for m in 0..MAX_KAUTZ_POLES {
        out.kautz_c_l[m] = (
            l(base.kautz_c_l[m].0, target.kautz_c_l[m].0),
            l(base.kautz_c_l[m].1, target.kautz_c_l[m].1),
        );
        out.kautz_c_r[m] = (
            l(base.kautz_c_r[m].0, target.kautz_c_r[m].0),
            l(base.kautz_c_r[m].1, target.kautz_c_r[m].1),
        );
    }
    out
}

/// Fits nested monotone `[Lite, Full, Pro, Ultra]` `V3ColdTierModel`s via closed-form `VarPro`
/// + exact reverse-mode adjoint L-BFGS + 160-pole OMP Kautz selection + `DropGuard` rollback.
#[must_use]
pub fn fit_v3_core_four_tiers(dataset: &V3TrainDataset, mono_out: bool) -> [V3ColdTierModel; 4] {
    let ds_homotopy = dataset.with_gabor_weights(0.45, 0.03);
    let ds_polish = dataset.with_gabor_weights(0.0, 0.012);
    let mut ws = AdjointWorkspace::default();

    // --- Tier 0: Lite (C=4, 1x, FIR=16, Kautz=4) ---
    let mut top_seeds = prescreen_lite_seeds(dataset, mono_out);
    if top_seeds.is_empty() {
        top_seeds.push(V3OptModelF64::from_cold_model(
            &V3ColdTierModel::identity_for_tier(Tier::Lite, false, mono_out),
        ));
    }
    let mut best_lite_opt = top_seeds[0].clone();
    let mut best_lite_loss = f64::MAX;
    for mut seed in top_seeds {
        run_lbfgs(&mut seed, &ds_homotopy, &mut ws, 25);
        let loss = evaluate_loss_f64(&seed, &ds_polish, &mut ws);
        if loss.is_finite() && loss < best_lite_loss {
            best_lite_loss = loss;
            best_lite_opt = seed;
        }
    }
    let (lite_cold, lite_loss_f32) =
        polish_and_quantize(&mut best_lite_opt, &ds_polish, &ds_polish, &mut ws, 0, 4);
    let raw_lite_loss = evaluate_loss_f64(&best_lite_opt, &ds_polish, &mut ws);

    // --- Helper for synchronous 1x monotone lifted tiers (progressive raw_opt + strict 7-metric guard) ---
    let fit_lifted = |cert_cold: &V3ColdTierModel,
                      cert_loss_f32: f64,
                      raw_opt: &V3OptModelF64,
                      raw_loss: f64,
                      next_tier: Tier,
                      ws: &mut AdjointWorkspace|
     -> (V3ColdTierModel, f64, V3OptModelF64, f64) {
        let prev_c = tier_cascade_lanes(cert_cold.tier);
        let next_c = tier_cascade_lanes(next_tier);
        let prev_k = tier_kautz_banks(cert_cold.tier) * 4;
        let next_k = tier_kautz_banks(next_tier) * 4;

        let cert_opt = V3OptModelF64::from_cold_model(cert_cold);
        let exact_opt = lift_opt_model(&cert_opt, next_tier, false);
        let exact_cold = exact_opt.to_cold_model();
        let prev_sum = dataset
            .stems
            .first()
            .and_then(|s0| coreset_summary_for_cold(cert_cold, s0, dataset.sample_rate_hz));
        let summary_ok = |cold: &V3ColdTierModel| -> bool {
            let Some(ref lo) = prev_sum else {
                return true;
            };
            dataset
                .stems
                .first()
                .and_then(|s0| coreset_summary_for_cold(cold, s0, dataset.sample_rate_hz))
                .is_some_and(|hi| is_strict_summary_upgrade(lo, &hi))
        };

        let mut best_cold = exact_cold.clone();
        let mut best_loss = cert_loss_f32;
        let mut next_raw_opt = lift_opt_model(raw_opt, next_tier, false);
        let raw_cand = next_raw_opt.clone();
        let mut next_raw_loss = raw_loss;

        // Baseline 1: exact cert nonlinear + extended FIR/Kautz f32 VarPro with monotone blend fallback
        let mut ext_varpro_opt = exact_opt.clone();
        select_omp_kautz_poles(&mut ext_varpro_opt, &ds_polish, prev_k, next_k);
        let mut ext_varpro_cold = ext_varpro_opt.to_cold_model();
        solve_bezout_varpro_cholesky_f32(&mut ext_varpro_cold, &ds_polish);
        let mut ext_base_cold = exact_cold.clone();
        ext_base_cold.kautz_poles = ext_varpro_cold.kautz_poles;
        for alpha in [
            1.0_f32, 0.90, 0.80, 0.70, 0.55, 0.40, 0.25, 0.15, 0.08, 0.03, 0.01,
        ] {
            let b_cold = blend_linear_readout(&ext_base_cold, &ext_varpro_cold, alpha);
            let b_loss = evaluate_cold_loss_f32(&b_cold, &ds_polish);
            if b_loss.is_finite() && b_loss < best_loss && summary_ok(&b_cold) {
                best_loss = b_loss;
                best_cold = b_cold;
                break;
            }
        }

        let mut try_candidate =
            |mut cand: V3OptModelF64, is_raw_path: bool, ws: &mut AdjointWorkspace| -> bool {
                seed_new_cascade_group(&mut cand, prev_c, next_c);
                let (cold_c, loss_c) =
                    polish_and_quantize(&mut cand, &ds_polish, &ds_polish, ws, prev_k, next_k);
                if is_raw_path {
                    let raw_c_loss = evaluate_loss_f64(&cand, &ds_polish, ws);
                    if raw_c_loss.is_finite() && raw_c_loss < next_raw_loss {
                        next_raw_loss = raw_c_loss;
                        next_raw_opt = cand.clone();
                    }
                }
                if loss_c.is_finite() && loss_c < best_loss && summary_ok(&cold_c) {
                    best_loss = loss_c;
                    best_cold = cold_c;
                    return true;
                }
                let target_sync = V3OptModelF64::from_cold_model(&cold_c);
                for alpha in [0.90_f64, 0.75, 0.60, 0.45, 0.30, 0.15, 0.05] {
                    let b_opt = exact_opt.blend_same_os(&target_sync, alpha);
                    let mut b_cold = b_opt.to_cold_model();
                    let d_loss = evaluate_cold_loss_f32(&b_cold, &ds_polish);
                    if d_loss.is_finite() && d_loss < best_loss && summary_ok(&b_cold) {
                        best_loss = d_loss;
                        best_cold = b_cold;
                        return false;
                    }
                    solve_bezout_varpro_cholesky_f32(&mut b_cold, &ds_polish);
                    let v_loss = evaluate_cold_loss_f32(&b_cold, &ds_polish);
                    if v_loss.is_finite() && v_loss < best_loss && summary_ok(&b_cold) {
                        best_loss = v_loss;
                        best_cold = b_cold;
                        return false;
                    }
                }
                false
            };

        let raw_differs = (raw_opt.svf_base_g - cert_opt.svf_base_g).abs() > 1.0e-7
            || (raw_opt.pre_cab_u_gain[0] - cert_opt.pre_cab_u_gain[0]).abs() > 1.0e-6
            || (raw_opt.blocks[0].in_u_gain[0] - cert_opt.blocks[0].in_u_gain[0]).abs() > 1.0e-6;
        let direct_ok = try_candidate(raw_cand, true, ws);
        if !direct_ok && raw_differs {
            try_candidate(exact_opt.clone(), false, ws);
        }

        (best_cold, best_loss, next_raw_opt, next_raw_loss)
    };

    // --- Tier 1..3: Full -> Pro -> Ultra (1x synchronous) ---
    let (full_cold, full_loss_f32, full_raw, full_raw_loss) = fit_lifted(
        &lite_cold,
        lite_loss_f32,
        &best_lite_opt,
        raw_lite_loss,
        Tier::Full,
        &mut ws,
    );
    let (pro_cold, pro_loss_f32, pro_raw, pro_raw_loss) = fit_lifted(
        &full_cold,
        full_loss_f32,
        &full_raw,
        full_raw_loss,
        Tier::Pro,
        &mut ws,
    );
    let (ultra_cold, _, _, _) = fit_lifted(
        &pro_cold,
        pro_loss_f32,
        &pro_raw,
        pro_raw_loss,
        Tier::Ultra,
        &mut ws,
    );

    [lite_cold, full_cold, pro_cold, ultra_cold]
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::adjoint::evaluate_loss_and_grad_f64;
    use super::*;
    use crate::metrics::{evaluate_besr_from_band_energies, split_bezout_3band_energies};
    use dnam_engine::stages::v3_core::cascade::tier_fir_segments;

    type ParamMutFn = fn(&mut V3OptModelF64, f64);

    fn make_synthetic_stem(n: usize, warmup: usize) -> V3TrainStem {
        let mut x = vec![0.0_f32; n];
        for (t, v) in x.iter_mut().enumerate() {
            let tf = t as f64 / 48_000.0;
            let env = (0.3 + 0.7 * (2.0 * PI * 3.0 * tf).sin().abs()) as f32;
            *v = env
                * (0.45 * (2.0 * PI * 140.0 * tf).sin() + 0.25 * (2.0 * PI * 930.0 * tf).sin())
                    as f32;
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
        let ol64: Vec<f64> = ol.iter().map(|&v| v as f64).collect();
        let tgt64: Vec<f64> = stem.tgt_l.iter().map(|&v| v as f64).collect();
        let (sig, err, noise) =
            split_bezout_3band_energies(&tgt64, &ol64, None, 48_000.0, stem.warmup);
        evaluate_besr_from_band_energies(sig, err, noise).besr_db
    }

    fn sample_opt_model(tier: Tier, os2x: bool, mono_out: bool) -> V3OptModelF64 {
        let mut m = V3ColdTierModel::identity_for_tier(tier, os2x, mono_out);
        m.svf.base_g = 0.072;
        m.svf.g_min = 0.078;
        m.svf.g_max = 0.115;
        m.svf.sec2_g_ratio = 1.38;
        m.svf.env_oct_sens = 1.75;
        m.svf.env_comp = 1.8;
        m.svf.atk_alpha = 0.78;
        m.svf.rel_alpha = 0.96;
        m.svf.det_lp_pole = 0.85;
        m.svf.det_hp_mix = 0.42;
        m.svf.damp_k1 = 0.75;
        m.svf.damp_k2 = 0.95;
        m.svf.sec2_bp_feed = 0.28;
        m.svf.sec2_hp_feed = -0.14;
        m.svf.stereo_link = 0.4;
        m.svf.stage2_feed_taps[..7].copy_from_slice(&[0.55, 0.22, 0.31, -0.12, 0.19, 0.14, -0.07]);
        m.svf.direct_taps_l[..7].copy_from_slice(&[0.06, 0.02, -0.015, 0.01, 0.03, 0.01, -0.02]);
        m.svf.direct_taps_r[..7].copy_from_slice(&[0.05, 0.03, -0.02, 0.015, 0.02, -0.01, 0.01]);
        let c_cnt = tier_cascade_lanes(tier);
        for b in 0..NUM_CASCADE_BLOCKS {
            m.blocks[b].atk_alpha = 0.65;
            m.blocks[b].rel_alpha = 0.94;
            m.blocks[b].gate_mix = 0.32;
            m.blocks[b].gate_sharpness = 9.5;
            for l in 0..c_cnt {
                let lf = l as f32;
                m.blocks[b].in_u_gain[l] = 0.65 + 0.12 * lf;
                m.blocks[b].lp_pole[l] = 0.42 + 0.04 * lf;
                m.blocks[b].lp_mix[l] = 0.82;
                m.blocks[b].hp_mix[l] = 0.36;
                m.blocks[b].static_bias[l] = -0.28 + 0.11 * lf;
                m.blocks[b].env_bias_sens[l] = 0.21 - 0.04 * lf;
                m.blocks[b].skip_u_gain[l] = 0.19;
                m.blocks[b].prev_skip_gain[l] = 0.14;
                if b > 0 {
                    for s in 0..c_cnt {
                        m.blocks[b].inter_mat[l][s] = if l == s { 0.38 } else { -0.05 };
                    }
                }
            }
        }
        m.pre_cab_u_gain = [0.24, 0.21];
        for l in 0..c_cnt {
            m.pre_cab_w[0][l] = 0.18 + 0.03 * (l as f32);
            m.pre_cab_w[1][l] = 0.15 + 0.025 * (l as f32);
            m.direct_s2_w[0][l] = 0.04;
            m.direct_s2_w[1][l] = -0.03;
        }
        for k in 0..tier_fir_segments(tier) * 16 {
            m.fir_l[k] = 0.38 * (0.85_f32).powi(k as i32);
            m.fir_r[k] = 0.35 * (0.84_f32).powi(k as i32);
        }
        for k in 0..tier_kautz_banks(tier) * 4 {
            m.kautz_poles[k] = (0.86 + 0.008 * (k as f32), 0.16 + 0.08 * (k as f32));
            m.kautz_c_l[k] = (0.045, 0.02);
            m.kautz_c_r[k] = (-0.035, 0.018);
        }
        V3OptModelF64::from_cold_model(&m)
    }

    #[test]
    fn test_v3_bezout_loss_matches_metrics_besr() {
        let n = 512;
        let in_l: Vec<f32> = (0..n)
            .map(|i| 0.4 * (i as f32 * 0.07).sin() + 0.2 * (i as f32 * 0.41).cos())
            .collect();
        let tgt_l: Vec<f32> = in_l
            .iter()
            .enumerate()
            .map(|(i, &x)| 0.6 * x + 0.15 * (i as f32 * 0.13).sin())
            .collect();
        let warmup = 64;
        let ds = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l: in_l.clone(),
                in_r: in_l,
                tgt_l: tgt_l.clone(),
                tgt_r: tgt_l.clone(),
                warmup,
            }],
            48_000.0,
        );
        let mdl = sample_opt_model(Tier::Lite, false, true);
        let mut ws = AdjointWorkspace::default();
        let loss = evaluate_loss_f64(&mdl, &ds, &mut ws);
        let tgt64: Vec<f64> = tgt_l.iter().map(|&v| v as f64).collect();
        let (s, e, ns) = split_bezout_3band_energies(&tgt64, &ws.y_out[0], None, 48_000.0, warmup);
        let rep = evaluate_besr_from_band_energies(s, e, ns);
        let rel = (loss - rep.besr_lin).abs() / rep.besr_lin.max(1.0e-12);
        assert!(
            rel < 1.0e-12,
            "loss={loss} besr_lin={} rel={rel}",
            rep.besr_lin
        );
    }

    #[test]
    fn test_v3_adjoint_matches_central_finite_diff() {
        let n = 192;
        let (mut in_l, mut in_r, mut tgt_l, mut tgt_r) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let t = i as f32 * 0.097 + 0.37;
            in_l[i] = 0.55 * t.sin() + 0.31 * (2.3 * t + 0.5).cos() + 0.13;
            in_r[i] = 0.49 * (1.2 * t).cos() - 0.27 * (2.9 * t + 0.2).sin() - 0.09;
            tgt_l[i] = 0.35 * (t + 0.1).sin() + 0.18 * (2.1 * t).cos();
            tgt_r[i] = 0.30 * (1.1 * t + 0.2).cos() - 0.15 * (2.7 * t).sin();
        }
        let ds_base = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l,
                in_r,
                tgt_l,
                tgt_r,
                warmup: 24,
            }],
            48_000.0,
        );
        let ds_gabor = ds_base.with_gabor_weights(0.35, 0.05);
        let mut ws = AdjointWorkspace::default();
        for (tag, ds) in [("bezout", &ds_base), ("bezout+gabor", &ds_gabor)] {
            for os2x in [false, true] {
                let base = sample_opt_model(Tier::Lite, os2x, false);
                let (_, grad) = evaluate_loss_and_grad_f64(&base, ds, &mut ws);
                let h = 1.0e-6_f64;
                let checks: [(&str, f64, ParamMutFn); 24] = [
                    ("svf_base_g", grad.svf_base_g, |m, d| m.svf_base_g += d),
                    ("svf_g_min", grad.svf_g_min, |m, d| m.svf_g_min += d),
                    ("svf_g_max", grad.svf_g_max, |m, d| m.svf_g_max += d),
                    ("svf_sec2_g_ratio", grad.svf_sec2_g_ratio, |m, d| {
                        m.svf_sec2_g_ratio += d
                    }),
                    ("svf_env_oct_sens", grad.svf_env_oct_sens, |m, d| {
                        m.svf_env_oct_sens += d
                    }),
                    ("svf_env_comp", grad.svf_env_comp, |m, d| {
                        m.svf_env_comp += d
                    }),
                    ("svf_atk_alpha", grad.svf_atk_alpha, |m, d| {
                        m.svf_atk_alpha += d
                    }),
                    ("svf_rel_alpha", grad.svf_rel_alpha, |m, d| {
                        m.svf_rel_alpha += d
                    }),
                    ("svf_det_lp_pole", grad.svf_det_lp_pole, |m, d| {
                        m.svf_det_lp_pole += d
                    }),
                    ("svf_det_hp_mix", grad.svf_det_hp_mix, |m, d| {
                        m.svf_det_hp_mix += d
                    }),
                    ("svf_damp_k1", grad.svf_damp_k1, |m, d| m.svf_damp_k1 += d),
                    ("svf_damp_k2", grad.svf_damp_k2, |m, d| m.svf_damp_k2 += d),
                    ("svf_sec2_bp", grad.svf_sec2_bp_feed, |m, d| {
                        m.svf_sec2_bp_feed += d
                    }),
                    ("svf_sec2_hp", grad.svf_sec2_hp_feed, |m, d| {
                        m.svf_sec2_hp_feed += d
                    }),
                    ("svf_stereo_link", grad.svf_stereo_link, |m, d| {
                        m.svf_stereo_link += d
                    }),
                    ("svf_feed_1", grad.svf_stage2_feed_taps[1], |m, d| {
                        m.svf_stage2_feed_taps[1] += d
                    }),
                    ("svf_dir_r_2", grad.svf_direct_taps_r[2], |m, d| {
                        m.svf_direct_taps_r[2] += d
                    }),
                    ("b0_lp_pole_1", grad.blocks[0].lp_pole[1], |m, d| {
                        m.blocks[0].lp_pole[1] += d
                    }),
                    ("b0_static_bias", grad.blocks[0].static_bias[0], |m, d| {
                        m.blocks[0].static_bias[0] += d
                    }),
                    ("b1_inter_1_2", grad.blocks[1].inter_mat[1][2], |m, d| {
                        m.blocks[1].inter_mat[1][2] += d
                    }),
                    ("b0_gate_sharp", grad.blocks[0].gate_sharpness, |m, d| {
                        m.blocks[0].gate_sharpness += d
                    }),
                    ("pre_cab_w_l0", grad.pre_cab_w[0][0], |m, d| {
                        m.pre_cab_w[0][0] += d
                    }),
                    ("fir_l_3", grad.fir_l[3], |m, d| m.fir_l[3] += d),
                    ("kautz_th_0", grad.kautz_poles[0].1, |m, d| {
                        m.kautz_poles[0].1 += d
                    }),
                ];
                for (name, g_ana, mut_fn) in checks {
                    let (mut mp, mut mm) = (base.clone(), base.clone());
                    mut_fn(&mut mp, h);
                    mut_fn(&mut mm, -h);
                    let g_fd = (evaluate_loss_f64(&mp, ds, &mut ws)
                        - evaluate_loss_f64(&mm, ds, &mut ws))
                        / (2.0 * h);
                    let rel_err = (g_ana - g_fd).abs() / g_fd.abs().max(g_ana.abs()).max(1.0e-8);
                    assert!(
                        rel_err < 1.0e-5,
                        "{tag} os2x={os2x} {name}: ana={g_ana:.6e}, fd={g_fd:.6e}, rel={rel_err:.6e}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fit_v3_core_four_tiers_monotone_nonlinear() {
        let stem = make_synthetic_stem(2_400, 128);
        let ds = V3TrainDataset::new(vec![stem.clone()], 48_000.0);
        let models = fit_v3_core_four_tiers(&ds, true);
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
            .map(|m| coreset_summary_for_cold(m, &stem, 48_000.0).unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        assert!(crate::metrics::verify_monotone_4tier_summaries(&sums));
    }
}
