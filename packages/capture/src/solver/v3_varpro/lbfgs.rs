#![forbid(unsafe_code)]
//! Bounded parameter codecs and L-BFGS (`m = 12`) optimizer with `DropGuard` line-search rollback.

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::mem::DropGuard;

use dnam_engine::stages::v3_core::cascade::{
    NUM_CASCADE_BLOCKS, tier_cascade_lanes, tier_fir_segments, tier_kautz_banks,
};

use super::adjoint::{AdjointWorkspace, evaluate_loss_and_grad_f64, evaluate_loss_f64};
use super::model_f64::{V3OptModelF64, V3TrainDataset};

#[derive(Clone, Copy)]
enum MapKind {
    Identity,
    LinearLogit(f64, f64),
    LogLogit(f64, f64),
    ComplementLogLogit(f64, f64),
}

#[inline]
fn sigmoid(u: f64) -> f64 {
    if u >= 0.0 {
        let z = (-u).exp();
        1.0 / (1.0 + z)
    } else {
        let z = u.exp();
        z / (1.0 + z)
    }
}

#[inline]
fn encode_param(x: f64, kind: MapKind) -> f64 {
    match kind {
        MapKind::Identity => x,
        MapKind::LinearLogit(lo, hi) => {
            let p = ((x - lo) / (hi - lo)).clamp(1.0e-5, 1.0 - 1.0e-5);
            (p / (1.0 - p)).ln()
        }
        MapKind::LogLogit(lo, hi) => {
            let llo = lo.ln();
            let lhi = hi.ln();
            let lx = x.clamp(lo, hi).ln();
            let p = ((lx - llo) / (lhi - llo)).clamp(1.0e-5, 1.0 - 1.0e-5);
            (p / (1.0 - p)).ln()
        }
        MapKind::ComplementLogLogit(lo, hi) => {
            let c_min = (1.0 - hi).max(1.0e-9);
            let c_max = (1.0 - lo).max(c_min * 1.0001);
            let lc_min = c_min.ln();
            let lc_max = c_max.ln();
            let lc = (1.0 - x).clamp(c_min, c_max).ln();
            let p = ((lc - lc_min) / (lc_max - lc_min)).clamp(1.0e-5, 1.0 - 1.0e-5);
            (p / (1.0 - p)).ln()
        }
    }
}

#[inline]
fn decode_param(u: f64, kind: MapKind) -> (f64, f64) {
    match kind {
        MapKind::Identity => (u, 1.0),
        MapKind::LinearLogit(lo, hi) => {
            let s = sigmoid(u.clamp(-30.0, 30.0));
            let span = hi - lo;
            (lo + span * s, span * s * (1.0 - s))
        }
        MapKind::LogLogit(lo, hi) => {
            let llo = lo.ln();
            let span = hi.ln() - llo;
            let s = sigmoid(u.clamp(-30.0, 30.0));
            let x = (llo + span * s).exp();
            (x, x * span * s * (1.0 - s))
        }
        MapKind::ComplementLogLogit(lo, hi) => {
            let c_min = (1.0 - hi).max(1.0e-9);
            let c_max = (1.0 - lo).max(c_min * 1.0001);
            let lc_min = c_min.ln();
            let span = c_max.ln() - lc_min;
            let s = sigmoid(u.clamp(-30.0, 30.0));
            let c = (lc_min + span * s).exp();
            (1.0 - c, -c * span * s * (1.0 - s))
        }
    }
}

/// Visits every active parameter slot for `(model.tier, model.mono_out)` in canonical order
/// alongside its optional gradient value so encode/decode/chain-rule never drift out of sync.
fn visit_active_params<F>(model: &mut V3OptModelF64, grad: Option<&V3OptModelF64>, mut f: F)
where
    F: FnMut(&mut f64, f64, MapKind),
{
    let c_cnt = tier_cascade_lanes(model.tier);
    let f_cnt = tier_fir_segments(model.tier) * 16;
    let k_cnt = tier_kautz_banks(model.tier) * 4;
    let active_ch = if model.mono_out { 1 } else { 2 };

    macro_rules! p {
        ($field:expr, |$g:ident| $gfield:expr, $kind:expr) => {{
            let gv = match grad {
                Some($g) => $gfield,
                None => 0.0,
            };
            f(&mut $field, gv, $kind);
        }};
    }

    // Stage I nonlinear SVF + taps
    p!(
        model.svf_base_g,
        |g| g.svf_base_g,
        MapKind::LogLogit(1.0e-4, 1.45)
    );
    p!(
        model.svf_sec2_g_ratio,
        |g| g.svf_sec2_g_ratio,
        MapKind::LogLogit(0.25, 4.0)
    );
    p!(
        model.svf_env_oct_sens,
        |g| g.svf_env_oct_sens,
        MapKind::LinearLogit(-16.0, 16.0)
    );
    p!(
        model.svf_env_comp,
        |g| g.svf_env_comp,
        MapKind::LinearLogit(0.0, 16.0)
    );
    p!(
        model.svf_atk_alpha,
        |g| g.svf_atk_alpha,
        MapKind::ComplementLogLogit(0.001, 0.9999)
    );
    p!(
        model.svf_rel_alpha,
        |g| g.svf_rel_alpha,
        MapKind::ComplementLogLogit(0.001, 0.99995)
    );
    p!(
        model.svf_det_lp_pole,
        |g| g.svf_det_lp_pole,
        MapKind::ComplementLogLogit(0.05, 0.995)
    );
    p!(
        model.svf_det_hp_mix,
        |g| g.svf_det_hp_mix,
        MapKind::LinearLogit(-1.0, 2.0)
    );
    p!(
        model.svf_damp_k1,
        |g| g.svf_damp_k1,
        MapKind::LogLogit(0.05, 4.0)
    );
    p!(
        model.svf_damp_k2,
        |g| g.svf_damp_k2,
        MapKind::LogLogit(0.05, 4.0)
    );
    p!(
        model.svf_sec2_bp_feed,
        |g| g.svf_sec2_bp_feed,
        MapKind::LinearLogit(-2.0, 2.0)
    );
    p!(
        model.svf_sec2_hp_feed,
        |g| g.svf_sec2_hp_feed,
        MapKind::LinearLogit(-2.0, 2.0)
    );
    p!(
        model.svf_stereo_link,
        |g| g.svf_stereo_link,
        MapKind::LinearLogit(0.0, 1.0)
    );
    p!(
        model.svf_g_min,
        |g| g.svf_g_min,
        MapKind::LogLogit(1.0e-4, 1.45)
    );
    p!(
        model.svf_g_max,
        |g| g.svf_g_max,
        MapKind::LogLogit(1.0e-4, 1.45)
    );
    for i in 0..7 {
        p!(
            model.svf_stage2_feed_taps[i],
            |g| g.svf_stage2_feed_taps[i],
            MapKind::Identity
        );
        p!(
            model.svf_direct_taps_l[i],
            |g| g.svf_direct_taps_l[i],
            MapKind::Identity
        );
        if !model.mono_out {
            p!(
                model.svf_direct_taps_r[i],
                |g| g.svf_direct_taps_r[i],
                MapKind::Identity
            );
        }
    }

    // Stage II nonlinear cascade blocks (active lanes only)
    for b in 0..NUM_CASCADE_BLOCKS {
        p!(
            model.blocks[b].atk_alpha,
            |g| g.blocks[b].atk_alpha,
            MapKind::ComplementLogLogit(0.001, 0.9999)
        );
        p!(
            model.blocks[b].rel_alpha,
            |g| g.blocks[b].rel_alpha,
            MapKind::ComplementLogLogit(0.001, 0.99995)
        );
        p!(
            model.blocks[b].gate_mix,
            |g| g.blocks[b].gate_mix,
            MapKind::LinearLogit(0.0, 1.0)
        );
        p!(
            model.blocks[b].gate_sharpness,
            |g| g.blocks[b].gate_sharpness,
            MapKind::LogLogit(0.05, 1.0e4)
        );
        for l in 0..c_cnt {
            p!(
                model.blocks[b].in_u_gain[l],
                |g| g.blocks[b].in_u_gain[l],
                MapKind::Identity
            );
            p!(
                model.blocks[b].lp_pole[l],
                |g| g.blocks[b].lp_pole[l],
                MapKind::ComplementLogLogit(0.02, 0.9985)
            );
            p!(
                model.blocks[b].lp_mix[l],
                |g| g.blocks[b].lp_mix[l],
                MapKind::Identity
            );
            p!(
                model.blocks[b].hp_mix[l],
                |g| g.blocks[b].hp_mix[l],
                MapKind::Identity
            );
            p!(
                model.blocks[b].static_bias[l],
                |g| g.blocks[b].static_bias[l],
                MapKind::LinearLogit(-4.0, 4.0)
            );
            p!(
                model.blocks[b].env_bias_sens[l],
                |g| g.blocks[b].env_bias_sens[l],
                MapKind::LinearLogit(-4.0, 4.0)
            );
            p!(
                model.blocks[b].skip_u_gain[l],
                |g| g.blocks[b].skip_u_gain[l],
                MapKind::Identity
            );
            if b > 0 {
                p!(
                    model.blocks[b].prev_skip_gain[l],
                    |g| g.blocks[b].prev_skip_gain[l],
                    MapKind::Identity
                );
                for s in 0..c_cnt {
                    p!(
                        model.blocks[b].inter_mat[l][s],
                        |g| g.blocks[b].inter_mat[l][s],
                        MapKind::Identity
                    );
                }
            }
        }
    }

    // Stage II->III pre-cab + Stage III linear & Kautz poles
    for ch in 0..active_ch {
        p!(
            model.pre_cab_u_gain[ch],
            |g| g.pre_cab_u_gain[ch],
            MapKind::Identity
        );
        for l in 0..c_cnt {
            p!(
                model.pre_cab_w[ch][l],
                |g| g.pre_cab_w[ch][l],
                MapKind::Identity
            );
            p!(
                model.direct_s2_w[ch][l],
                |g| g.direct_s2_w[ch][l],
                MapKind::Identity
            );
        }
    }
    for k in 0..f_cnt {
        p!(model.fir_l[k], |g| g.fir_l[k], MapKind::Identity);
        if !model.mono_out {
            p!(model.fir_r[k], |g| g.fir_r[k], MapKind::Identity);
        }
    }
    for m in 0..k_cnt {
        p!(
            model.kautz_poles[m].0,
            |g| g.kautz_poles[m].0,
            MapKind::ComplementLogLogit(0.50, 0.997)
        );
        p!(
            model.kautz_poles[m].1,
            |g| g.kautz_poles[m].1,
            MapKind::LogLogit(1.0e-3, PI - 1.0e-3)
        );
        p!(
            model.kautz_c_l[m].0,
            |g| g.kautz_c_l[m].0,
            MapKind::Identity
        );
        p!(
            model.kautz_c_l[m].1,
            |g| g.kautz_c_l[m].1,
            MapKind::Identity
        );
        if !model.mono_out {
            p!(
                model.kautz_c_r[m].0,
                |g| g.kautz_c_r[m].0,
                MapKind::Identity
            );
            p!(
                model.kautz_c_r[m].1,
                |g| g.kautz_c_r[m].1,
                MapKind::Identity
            );
        }
    }
}

fn mirror_mono_channels(model: &mut V3OptModelF64) {
    if model.mono_out {
        model.svf_direct_taps_r = model.svf_direct_taps_l;
        model.pre_cab_u_gain[1] = model.pre_cab_u_gain[0];
        model.pre_cab_w[1] = model.pre_cab_w[0];
        model.direct_s2_w[1] = model.direct_s2_w[0];
        model.fir_r = model.fir_l;
        model.kautz_c_r = model.kautz_c_l;
    }
}

fn encode_model_u(model: &V3OptModelF64) -> Vec<f64> {
    let mut clone = model.clone();
    let mut u = Vec::with_capacity(256);
    visit_active_params(&mut clone, None, |val, _, kind| {
        u.push(encode_param(*val, kind));
    });
    u
}

fn decode_u_into_model(u: &[f64], model: &mut V3OptModelF64) {
    let mut idx = 0_usize;
    visit_active_params(model, None, |val, _, kind| {
        let (x, _) = decode_param(u[idx], kind);
        *val = x;
        idx += 1;
    });
    mirror_mono_channels(model);
}

fn map_model_grad_to_u(
    u: &[f64],
    template: &V3OptModelF64,
    grad_model: &V3OptModelF64,
) -> Vec<f64> {
    let mut scratch = template.clone();
    let mut g_u = Vec::with_capacity(u.len());
    let mut idx = 0_usize;
    visit_active_params(&mut scratch, Some(grad_model), |_, g_val, kind| {
        let (_, dx_du) = decode_param(u[idx], kind);
        g_u.push(g_val * dx_du);
        idx += 1;
    });
    g_u
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn lbfgs_direction(g: &[f64], history: &VecDeque<(Vec<f64>, Vec<f64>, f64)>) -> Vec<f64> {
    let mut q: Vec<f64> = g.to_vec();
    let k = history.len();
    let mut alpha = vec![0.0_f64; k];
    for (i, (s, y, rho)) in history.iter().enumerate().rev() {
        let a = rho * dot(s, &q);
        alpha[i] = a;
        for (qi, &yi) in q.iter_mut().zip(y) {
            *qi -= a * yi;
        }
    }
    let gamma = if let Some((s, y, _)) = history.back() {
        let yy = dot(y, y);
        if yy > 1.0e-18 {
            (dot(s, y) / yy).clamp(1.0e-6, 1.0e4)
        } else {
            1.0
        }
    } else {
        let gg = dot(g, g).sqrt();
        if gg > 1.0 {
            (1.0 / gg).clamp(1.0e-4, 1.0)
        } else {
            1.0
        }
    };
    let mut r: Vec<f64> = q.into_iter().map(|v| v * gamma).collect();
    for (i, (s, y, rho)) in history.iter().enumerate() {
        let beta = rho * dot(y, &r);
        let diff = alpha[i] - beta;
        for (ri, &si) in r.iter_mut().zip(s) {
            *ri += diff * si;
        }
    }
    for ri in &mut r {
        *ri = -*ri;
    }
    r
}

struct TrialRollback<'a> {
    model: &'a mut V3OptModelF64,
    u: &'a mut Vec<f64>,
    backup_model: V3OptModelF64,
    backup_u: Vec<f64>,
}

/// Evaluates a single backtracking line-search candidate wrapped in `core::mem::DropGuard`
/// so rejected, non-finite, or early-exit trials deterministically restore `(model, u)`.
#[allow(clippy::too_many_arguments)]
fn try_armijo_line_search(
    model: &mut V3OptModelF64,
    u: &mut Vec<f64>,
    g: &mut Vec<f64>,
    loss: &mut f64,
    dir: &[f64],
    slope: f64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut guard = DropGuard::new(
        TrialRollback {
            backup_model: model.clone(),
            backup_u: u.clone(),
            model,
            u,
        },
        |rb| {
            *rb.model = rb.backup_model;
            *rb.u = rb.backup_u;
        },
    );

    let orig_loss = *loss;
    let d_len = guard.u.len();
    let mut step_alpha = 1.0_f64;
    for _ in 0..14 {
        {
            let rb = &mut *guard;
            for i in 0..d_len {
                rb.u[i] = rb.backup_u[i] + step_alpha * dir[i];
            }
            decode_u_into_model(rb.u, rb.model);
        }
        let cand_loss = evaluate_loss_f64(guard.model, dataset, ws);
        if cand_loss.is_finite() && cand_loss <= orig_loss + 1.0e-4 * step_alpha * slope {
            let (exact_loss, grad_model) = evaluate_loss_and_grad_f64(guard.model, dataset, ws);
            let new_g = map_model_grad_to_u(guard.u, guard.model, &grad_model);
            if exact_loss.is_finite() && new_g.iter().all(|v| v.is_finite()) {
                let s_vec: Vec<f64> = guard
                    .u
                    .iter()
                    .zip(&guard.backup_u)
                    .map(|(&nu, &ou)| nu - ou)
                    .collect();
                let y_vec: Vec<f64> = new_g
                    .iter()
                    .zip(g.iter())
                    .map(|(&ng, &og)| ng - og)
                    .collect();
                *loss = exact_loss;
                *g = new_g;
                DropGuard::dismiss(guard);
                return Some((s_vec, y_vec));
            }
        }
        step_alpha *= 0.5;
    }
    None
}

/// Runs L-BFGS (`m = 12`) in smooth transformed parameter space with `DropGuard` rollback protection.
pub fn run_lbfgs(
    model: &mut V3OptModelF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    max_iters: usize,
) -> f64 {
    if max_iters == 0 {
        return evaluate_loss_f64(model, dataset, ws);
    }
    let mut u = encode_model_u(model);
    decode_u_into_model(&u, model);
    let (mut loss, grad_model) = evaluate_loss_and_grad_f64(model, dataset, ws);
    let mut g = map_model_grad_to_u(&u, model, &grad_model);
    let mut history: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::with_capacity(12);

    for _ in 0..max_iters {
        let g_norm = dot(&g, &g).sqrt();
        if !g_norm.is_finite() || g_norm < 1.0e-8 {
            break;
        }
        let mut dir = lbfgs_direction(&g, &history);
        let mut slope = dot(&dir, &g);
        if !slope.is_finite() || slope >= 0.0 {
            history.clear();
            dir = g.iter().map(|&v| -v).collect();
            slope = -dot(&g, &g);
        }

        if let Some((s_vec, y_vec)) =
            try_armijo_line_search(model, &mut u, &mut g, &mut loss, &dir, slope, dataset, ws)
        {
            let sy = dot(&s_vec, &y_vec);
            let sn = dot(&s_vec, &s_vec).sqrt();
            let yn = dot(&y_vec, &y_vec).sqrt();
            if sy.is_finite() && sy > 1.0e-10 * (sn * yn).max(1.0e-18) {
                if history.len() == 12 {
                    history.pop_front();
                }
                history.push_back((s_vec, y_vec, 1.0 / sy));
            }
        } else if !history.is_empty() {
            history.clear();
        } else {
            break;
        }
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::super::model_f64::V3TrainStem;
    use super::*;
    use dnam_engine::Tier;
    use dnam_engine::stages::v3_core::V3ColdTierModel;

    fn make_test_stem(n: usize, warmup: usize) -> V3TrainStem {
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

    #[test]
    fn test_drop_guard_line_search_rollback_and_dismiss() {
        let stem = make_test_stem(512, 64);
        let ds = V3TrainDataset::new(vec![stem], 48_000.0);
        let mut ws = AdjointWorkspace::default();
        let cold = V3ColdTierModel::identity_for_tier(Tier::Lite, false, true);
        let mut model = V3OptModelF64::from_cold_model(&cold);
        let orig_gain = model.pre_cab_u_gain[0];
        let mut u = encode_model_u(&model);
        let orig_u = u.clone();
        let (mut loss, grad_m) = evaluate_loss_and_grad_f64(&model, &ds, &mut ws);
        let mut g = map_model_grad_to_u(&u, &model, &grad_m);

        let bad_dir = vec![1.0e3_f64; u.len()];
        let accepted = try_armijo_line_search(
            &mut model, &mut u, &mut g, &mut loss, &bad_dir, 1.0, &ds, &mut ws,
        );
        assert!(accepted.is_none());
        assert_eq!(u, orig_u);
        assert!((model.pre_cab_u_gain[0] - orig_gain).abs() < 1.0e-12);

        let good_dir: Vec<f64> = g.iter().map(|&v| -v).collect();
        let slope = -dot(&g, &g);
        let prev_loss = loss;
        let step = try_armijo_line_search(
            &mut model, &mut u, &mut g, &mut loss, &good_dir, slope, &ds, &mut ws,
        );
        assert!(step.is_some());
        assert!(loss < prev_loss);
        assert_ne!(u, orig_u);
    }

    #[test]
    fn test_opt_codec_chain_rule_matches_central_finite_diff() {
        let stem = make_test_stem(400, 48);
        let ds = V3TrainDataset::new(vec![stem], 48_000.0);
        let mut ws = AdjointWorkspace::default();
        let mut cold = V3ColdTierModel::identity_for_tier(Tier::Lite, true, false);
        cold.svf.base_g = 0.08;
        cold.svf.g_min = 0.085;
        cold.svf.g_max = 0.125;
        cold.svf.env_oct_sens = 1.3;
        cold.svf.env_comp = 1.2;
        cold.blocks[0].gate_mix = 0.25;
        cold.blocks[0].gate_sharpness = 12.0;
        cold.blocks[0].in_u_gain = [
            1.4, -0.9, 0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut model = V3OptModelF64::from_cold_model(&cold);
        let u = encode_model_u(&model);
        decode_u_into_model(&u, &mut model);
        let (_, grad_m) = evaluate_loss_and_grad_f64(&model, &ds, &mut ws);
        let g_u = map_model_grad_to_u(&u, &model, &grad_m);

        let h = 1.0e-6_f64;
        for idx in (0..u.len())
            .step_by(9)
            .chain([2, 4, 5, 6, 13, 14, 36, 37, 41])
        {
            if idx >= u.len() {
                continue;
            }
            let (mut up, mut um) = (u.clone(), u.clone());
            up[idx] += h;
            um[idx] -= h;
            let (mut mp, mut mm) = (model.clone(), model.clone());
            decode_u_into_model(&up, &mut mp);
            decode_u_into_model(&um, &mut mm);
            let fd = (evaluate_loss_f64(&mp, &ds, &mut ws) - evaluate_loss_f64(&mm, &ds, &mut ws))
                / (2.0 * h);
            let err = (g_u[idx] - fd).abs();
            let scale = g_u[idx].abs().max(fd.abs()).max(1.0);
            assert!(
                err / scale < 5.0e-4,
                "codec u[{idx}] grad={}, fd={fd}",
                g_u[idx]
            );
        }
    }
}
