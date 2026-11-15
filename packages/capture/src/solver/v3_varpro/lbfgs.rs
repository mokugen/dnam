#![forbid(unsafe_code)]
//! Smooth Unconstrained Parameter Codec, `DropGuard` Armijo Line Search, and Two-Loop L-BFGS (`v3`).

use std::f64::consts::PI;

use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, MAX_GROUPS, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0, tier_block_lanes,
    tier_fir_segments, tier_groups, tier_kautz_banks,
};

use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

use super::adjoint::{
    AdjointWorkspace, evaluate_matryoshka_loss_and_grad_with_backend,
    evaluate_matryoshka_loss_f64_with_backend,
};
use super::basis::ALL_TIERS;
use super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset};

#[derive(Clone, Copy)]
enum ParamSpec {
    Free,
    Bounded(f64, f64),
    LogBounded(f64, f64),
    /// Encodes complement `c = 1.0 - v` on `(lo_c, hi_c)` in log space (`v = 1.0 - c`).
    ComplementLogBounded(f64, f64),
}

impl ParamSpec {
    #[inline(always)]
    fn encode(self, v: f64) -> f64 {
        match self {
            Self::Free => v,
            Self::Bounded(lo, hi) => {
                let span = (hi - lo).max(1.0e-12);
                let vc = v.clamp(lo + 1.0e-7 * span, hi - 1.0e-7 * span);
                let p = ((vc - lo) / span).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::LogBounded(lo, hi) => {
                let (la, lb) = (lo.ln(), hi.ln());
                let vc = v.clamp(lo * (1.0 + 1.0e-7), hi * (1.0 - 1.0e-7));
                let p = ((vc.ln() - la) / (lb - la).max(1.0e-12)).clamp(1.0e-7, 1.0 - 1.0e-7);
                (p / (1.0 - p)).ln()
            }
            Self::ComplementLogBounded(lo_c, hi_c) => Self::LogBounded(lo_c, hi_c).encode(1.0 - v),
        }
    }

    #[inline(always)]
    fn decode_and_deriv(self, u: f64) -> (f64, f64) {
        match self {
            Self::Free => (u, 1.0),
            Self::Bounded(lo, hi) => {
                let uc = u.clamp(-24.0, 24.0);
                let s = 1.0 / (1.0 + (-uc).exp());
                let span = hi - lo;
                (lo + span * s, span * s * (1.0 - s))
            }
            Self::LogBounded(lo, hi) => {
                let uc = u.clamp(-24.0, 24.0);
                let s = 1.0 / (1.0 + (-uc).exp());
                let (la, lb) = (lo.ln(), hi.ln());
                let lspan = lb - la;
                let v = (la + lspan * s).exp();
                (v, v * lspan * s * (1.0 - s))
            }
            Self::ComplementLogBounded(lo_c, hi_c) => {
                let (c, dc) = Self::LogBounded(lo_c, hi_c).decode_and_deriv(u);
                (1.0 - c, -dc)
            }
        }
    }
}

fn visit_active_params(
    m: &mut V3MatryoshkaOptF64,
    include_linear_out: bool,
    mut cb: impl FnMut(&mut f64, ParamSpec),
) {
    let g_cnt = m.max_active_groups.clamp(1, MAX_GROUPS);
    let c_blk = g_cnt * 4;
    let active_ch = if m.mono_out { 1 } else { 2 };

    if m.os2x {
        cb(&mut m.os_up_quad, ParamSpec::Bounded(-0.195, 0.045));
        cb(&mut m.os_dec_c_prev1, ParamSpec::Bounded(0.005, 0.275));
        cb(&mut m.os_dec_c0, ParamSpec::Bounded(0.005, 0.545));
        cb(&mut m.os_up_pre_emph, ParamSpec::Bounded(-0.27, 0.98));
        cb(&mut m.os_dec_aa_mix, ParamSpec::Bounded(1.0e-4, 0.9999));
        cb(&mut m.os_up_aa_mix, ParamSpec::Bounded(1.0e-4, 0.9999));
    }
    cb(&mut m.svf_base_g, ParamSpec::LogBounded(2.0e-4, 0.90));
    cb(&mut m.svf_sec2_g_ratio, ParamSpec::LogBounded(0.26, 3.90));
    cb(&mut m.svf_env_oct_sens, ParamSpec::Bounded(-23.5, 23.5));
    cb(&mut m.svf_env_comp, ParamSpec::LogBounded(1.0e-4, 15.5));
    cb(
        &mut m.svf_atk_alpha,
        ParamSpec::ComplementLogBounded(5.0e-5, 0.998),
    );
    cb(
        &mut m.svf_rel_alpha,
        ParamSpec::ComplementLogBounded(8.0e-6, 0.998),
    );
    cb(
        &mut m.svf_det_lp_pole,
        ParamSpec::ComplementLogBounded(6.0e-3, 0.94),
    );
    cb(&mut m.svf_det_hp_mix, ParamSpec::Bounded(-0.98, 1.98));
    cb(&mut m.svf_damp_k1, ParamSpec::LogBounded(0.06, 3.90));
    cb(&mut m.svf_damp_k2, ParamSpec::LogBounded(0.06, 3.90));
    cb(&mut m.svf_sec2_bp_feed, ParamSpec::Bounded(-1.95, 1.95));
    cb(&mut m.svf_sec2_hp_feed, ParamSpec::Bounded(-1.95, 1.95));
    if !m.mono_out {
        cb(&mut m.svf_stereo_link, ParamSpec::Bounded(0.001, 0.999));
    }
    cb(&mut m.svf_g_min, ParamSpec::LogBounded(1.5e-4, 0.15));
    cb(&mut m.svf_g_max, ParamSpec::LogBounded(0.02, 0.94));
    for v in &mut m.svf_stage2_feed_taps {
        cb(v, ParamSpec::Free);
    }

    cb(&mut m.feat_in_hp_beta, ParamSpec::LogBounded(1.2e-4, 0.48));
    cb(
        &mut m.feat_fast_atk_beta,
        ParamSpec::LogBounded(1.2e-4, 0.995),
    );
    cb(
        &mut m.feat_fast_rel_beta,
        ParamSpec::LogBounded(1.2e-5, 0.995),
    );
    cb(&mut m.feat_pinch_k, ParamSpec::LogBounded(1.0e-4, 24_500.0));
    cb(
        &mut m.feat_slow_atk_beta,
        ParamSpec::LogBounded(1.2e-5, 0.48),
    );
    cb(
        &mut m.feat_slow_rel_beta,
        ParamSpec::LogBounded(1.2e-6, 0.095),
    );
    cb(&mut m.feat_slow_det_comp, ParamSpec::Bounded(0.0, 63.0));

    for b in 0..NUM_GRU_BLOCKS {
        let blk = &mut m.blocks[b];
        for dst in 0..c_blk {
            let g_end = ((dst >> 2) + 1) << 2;
            for f in 0..NUM_IN_FEATS_B0 {
                cb(&mut blk.w_in_z[f][dst], ParamSpec::Free);
                cb(&mut blk.w_in_r[f][dst], ParamSpec::Free);
                cb(&mut blk.w_in_n[f][dst], ParamSpec::Free);
            }
            if b > 0 {
                for src in 0..g_end {
                    cb(&mut blk.w_prev_z[dst][src], ParamSpec::Free);
                    cb(&mut blk.w_prev_r[dst][src], ParamSpec::Free);
                    cb(&mut blk.w_prev_n[dst][src], ParamSpec::Free);
                }
            }
            for src in 0..g_end {
                cb(&mut blk.w_rec_z[dst][src], ParamSpec::Free);
                cb(&mut blk.w_rec_r[dst][src], ParamSpec::Free);
                cb(&mut blk.w_rec_n[dst][src], ParamSpec::Free);
            }
            cb(&mut blk.b_z[dst], ParamSpec::Bounded(-7.9, 8.1));
            cb(&mut blk.b_r[dst], ParamSpec::Bounded(-7.9, 8.1));
            cb(&mut blk.b_n[dst], ParamSpec::Bounded(-3.7, 3.7));
            cb(&mut blk.k_sag[dst], ParamSpec::Bounded(0.0, 7.8));
            cb(&mut blk.k_quad[dst], ParamSpec::Bounded(-0.44, 0.44));
        }
    }

    let mut max_k = 0_usize;
    for (tau, &tier) in ALL_TIERS.iter().enumerate() {
        if tier_groups(tier) <= g_cnt && m.active_tier_weights[tau] > 0.0 {
            max_k = max_k.max(tier_kautz_banks(tier) * 4);
        }
    }
    for pole in m.kautz_poles.iter_mut().take(max_k) {
        cb(&mut pole.0, ParamSpec::ComplementLogBounded(1.6e-3, 0.745));
        if pole.1 > 5.0e-4 {
            cb(&mut pole.1, ParamSpec::LogBounded(1.0e-3, PI - 0.002));
        }
    }

    for (tau, &tier) in ALL_TIERS.iter().enumerate() {
        if tier_groups(tier) > g_cnt || m.active_tier_weights[tau] <= 0.0 {
            continue;
        }
        let c_t = tier_block_lanes(tier);
        let f_t = tier_fir_segments(tier) * 16;
        let k_t = tier_kautz_banks(tier) * 4;
        let r = &mut m.readouts[tau];
        for ch in 0..active_ch {
            if include_linear_out {
                let taps = if ch == 0 {
                    &mut r.svf_direct_taps_l
                } else {
                    &mut r.svf_direct_taps_r
                };
                for v in taps.iter_mut() {
                    cb(v, ParamSpec::Free);
                }
            }
            cb(&mut r.pre_cab_u_gain[ch], ParamSpec::Free);
            cb(&mut r.pre_cab_hp_gain[ch], ParamSpec::Free);
            if include_linear_out {
                cb(&mut r.direct_s2_u_gain[ch], ParamSpec::Free);
                cb(&mut r.direct_s2_hp_gain[ch], ParamSpec::Free);
            }
            for b in 0..NUM_GRU_BLOCKS {
                let base = b * MAX_BLOCK_LANES;
                for l in 0..c_t {
                    cb(&mut r.pre_cab_w[ch][base + l], ParamSpec::Free);
                    if include_linear_out {
                        cb(&mut r.direct_s2_w[ch][base + l], ParamSpec::Free);
                    }
                }
            }
            if include_linear_out {
                let fir = if ch == 0 { &mut r.fir_l } else { &mut r.fir_r };
                for v in fir.iter_mut().take(f_t) {
                    cb(v, ParamSpec::Free);
                }
                let kc = if ch == 0 {
                    &mut r.kautz_c_l
                } else {
                    &mut r.kautz_c_r
                };
                for p in kc.iter_mut().take(k_t) {
                    cb(&mut p.0, ParamSpec::Free);
                    cb(&mut p.1, ParamSpec::Free);
                }
            }
        }
    }
}

fn pack_params(m: &mut V3MatryoshkaOptF64, include_linear_out: bool) -> Vec<f64> {
    let mut u = Vec::with_capacity(2048);
    visit_active_params(m, include_linear_out, |v, spec| u.push(spec.encode(*v)));
    u
}

fn unpack_and_derivs(
    m: &mut V3MatryoshkaOptF64,
    include_linear_out: bool,
    u: &[f64],
    jac_out: &mut [f64],
) {
    let mut i = 0_usize;
    visit_active_params(m, include_linear_out, |v, spec| {
        let (val, dv) = spec.decode_and_deriv(u[i]);
        *v = val;
        jac_out[i] = dv;
        i += 1;
    });
    m.mirror_mono();
}

fn pack_chain_grad(
    g: &mut V3MatryoshkaOptF64,
    include_linear_out: bool,
    jac: &[f64],
    out: &mut [f64],
) {
    let mut i = 0_usize;
    visit_active_params(g, include_linear_out, |gv, _| {
        out[i] = *gv * jac[i];
        i += 1;
    });
}

struct ModelBestGuard<'a> {
    target: &'a mut V3MatryoshkaOptF64,
    best: V3MatryoshkaOptF64,
    best_loss: f64,
    committed: bool,
}

impl Drop for ModelBestGuard<'_> {
    fn drop(&mut self) {
        if !self.committed {
            *self.target = self.best.clone();
            for blk in &mut self.target.blocks {
                blk.enforce_lower_triangular();
            }
            self.target.mirror_mono();
        }
    }
}

fn run_lbfgs_impl<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    max_iters: usize,
    include_linear_out: bool,
) -> f64 {
    if max_iters == 0 || dataset.stems.is_empty() {
        return 0.0;
    }
    let mut ws = AdjointWorkspace::default();
    let init_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(model, dataset, &mut ws);
    let mut guard = ModelBestGuard {
        best: model.clone(),
        best_loss: if init_loss.is_finite() {
            init_loss
        } else {
            f64::INFINITY
        },
        target: model,
        committed: false,
    };

    let mut u = pack_params(guard.target, include_linear_out);
    let dim = u.len();
    if dim == 0 {
        guard.committed = true;
        return guard.best_loss;
    }
    let (mut jac, mut g_u, mut g_model) =
        (vec![1.0; dim], vec![0.0; dim], guard.target.zero_like());
    unpack_and_derivs(guard.target, include_linear_out, &u, &mut jac);
    let mut loss = evaluate_matryoshka_loss_and_grad_with_backend::<B>(
        guard.target,
        dataset,
        &mut ws,
        Some(&mut g_model),
    );
    pack_chain_grad(&mut g_model, include_linear_out, &jac, &mut g_u);
    if loss.is_finite() && loss < guard.best_loss {
        guard.best_loss = loss;
        guard.best = guard.target.clone();
    }

    let (mut s_hist, mut y_hist, mut rho_hist) = (
        Vec::<Vec<f64>>::with_capacity(10),
        Vec::<Vec<f64>>::with_capacity(10),
        Vec::<f64>::with_capacity(10),
    );
    let (mut prev_u, mut prev_g, mut cand_u) = (vec![0.0; dim], vec![0.0; dim], vec![0.0; dim]);
    let mut small_improve_streak = 0_usize;

    for _iter in 0..max_iters {
        let g_norm2 = B::sum_sq_f64(&g_u);
        if !g_norm2.is_finite() || g_norm2.sqrt() < 1.0e-7 {
            break;
        }
        let mut q: Vec<f64> = g_u.iter().map(|&v| -v).collect();
        if !s_hist.is_empty() {
            let m = s_hist.len();
            let mut alpha = vec![0.0; m];
            for i in (0..m).rev() {
                alpha[i] = rho_hist[i] * B::dot_f64(&s_hist[i], &q);
                B::axpy_f64(&mut q, -alpha[i], &y_hist[i]);
            }
            if let (Some(sl), Some(yl)) = (s_hist.last(), y_hist.last()) {
                let yy = B::dot_f64(yl, yl).max(1.0e-18);
                let scale = (B::dot_f64(sl, yl) / yy).clamp(1.0e-6, 1.0e6);
                B::scale_f64(&mut q, scale);
            }
            for i in 0..m {
                let beta = rho_hist[i] * B::dot_f64(&y_hist[i], &q);
                B::axpy_f64(&mut q, alpha[i] - beta, &s_hist[i]);
            }
            let slope = B::dot_f64(&g_u, &q);
            if !slope.is_finite() || slope >= 0.0 {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                for j in 0..dim {
                    q[j] = -g_u[j];
                }
            }
        }
        prev_u.copy_from_slice(&u);
        prev_g.copy_from_slice(&g_u);
        let prev_accepted_loss = loss;

        let mut accepted = false;
        for attempt in 0..2 {
            if attempt == 1 {
                if s_hist.is_empty() {
                    break;
                }
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                for j in 0..dim {
                    q[j] = -prev_g[j];
                }
            }
            let q_max = q
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1.0e-12);
            let init_step = if attempt == 1 {
                (0.20 / q_max).min(0.15)
            } else if s_hist.is_empty() {
                (0.30 / q_max).min(0.25)
            } else {
                (2.0 / q_max).min(1.0)
            };
            let eff_slope = B::dot_f64(&prev_g, &q);
            if !eff_slope.is_finite() || eff_slope >= 0.0 {
                continue;
            }
            let mut step = init_step;
            let max_ls = if attempt == 0 { 5 } else { 3 };
            for ls_try in 0..max_ls {
                for j in 0..dim {
                    cand_u[j] = prev_u[j] + step * q[j];
                }
                unpack_and_derivs(guard.target, include_linear_out, &cand_u, &mut jac);
                let trial_l = if ls_try == 0 {
                    evaluate_matryoshka_loss_and_grad_with_backend::<B>(
                        guard.target,
                        dataset,
                        &mut ws,
                        Some(&mut g_model),
                    )
                } else {
                    evaluate_matryoshka_loss_f64_with_backend::<B>(guard.target, dataset, &mut ws)
                };
                if trial_l.is_finite() && trial_l <= loss + 1.0e-4 * step * eff_slope {
                    u.copy_from_slice(&cand_u);
                    loss = if ls_try == 0 {
                        trial_l
                    } else {
                        evaluate_matryoshka_loss_and_grad_with_backend::<B>(
                            guard.target,
                            dataset,
                            &mut ws,
                            Some(&mut g_model),
                        )
                    };
                    pack_chain_grad(&mut g_model, include_linear_out, &jac, &mut g_u);
                    if loss.is_finite() && loss < guard.best_loss {
                        guard.best_loss = loss;
                        guard.best = guard.target.clone();
                    }
                    accepted = true;
                    break;
                }
                step = if trial_l.is_finite() {
                    let denom = 2.0 * (trial_l - loss - eff_slope * step);
                    if denom > 1.0e-18 {
                        ((-eff_slope * step * step) / denom).clamp(0.15 * step, 0.60 * step)
                    } else {
                        0.35 * step
                    }
                } else {
                    0.25 * step
                };
            }
            if accepted {
                break;
            }
        }
        if !accepted {
            break;
        }
        if (prev_accepted_loss - loss).abs() < 5.0e-5 * prev_accepted_loss.abs().max(1.0e-12) {
            small_improve_streak += 1;
            if small_improve_streak >= 3 {
                break;
            }
        } else {
            small_improve_streak = 0;
        }
        let s: Vec<f64> = u.iter().zip(&prev_u).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_u.iter().zip(&prev_g).map(|(a, b)| a - b).collect();
        let sy = B::dot_f64(&s, &y);
        if sy.is_finite() && sy > 1.0e-18 {
            if s_hist.len() == 10 {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }
    }

    *guard.target = guard.best.clone();
    for blk in &mut guard.target.blocks {
        blk.enforce_lower_triangular();
    }
    guard.target.mirror_mono();
    guard.committed = true;
    guard.best_loss
}

/// Runs `max_iters` two-loop L-BFGS (`m = 10`) steps on unconstrained codec with `Drop` rollback safety via `B`.
pub fn run_lbfgs_matryoshka_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    max_iters: usize,
) -> f64 {
    run_lbfgs_impl::<B>(model, dataset, max_iters, true)
}

/// Runs `max_iters` two-loop L-BFGS (`m = 10`) steps optimizing core + `pre_cab` weights (holding Stage III linear output fixed) via `B`.
pub fn run_lbfgs_core_matryoshka_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    max_iters: usize,
) -> f64 {
    run_lbfgs_impl::<B>(model, dataset, max_iters, false)
}

/// Dispatches [`run_lbfgs_matryoshka_with_backend`] via fastest CPU backend.
pub fn run_lbfgs_matryoshka(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    max_iters: usize,
) -> f64 {
    struct Task<'a>(&'a mut V3MatryoshkaOptF64, &'a V3TrainDataset, usize);
    impl TrainerTask<f64> for Task<'_> {
        fn run<B: TrainerBackend>(self) -> f64 {
            run_lbfgs_matryoshka_with_backend::<B>(self.0, self.1, self.2)
        }
    }
    DynamicTrainer::detect().dispatch(Task(model, dataset, max_iters))
}

/// Dispatches [`run_lbfgs_core_matryoshka_with_backend`] via fastest CPU backend.
pub fn run_lbfgs_core_matryoshka(
    model: &mut V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    max_iters: usize,
) -> f64 {
    struct Task<'a>(&'a mut V3MatryoshkaOptF64, &'a V3TrainDataset, usize);
    impl TrainerTask<f64> for Task<'_> {
        fn run<B: TrainerBackend>(self) -> f64 {
            run_lbfgs_core_matryoshka_with_backend::<B>(self.0, self.1, self.2)
        }
    }
    DynamicTrainer::detect().dispatch(Task(model, dataset, max_iters))
}

#[cfg(test)]
mod tests {
    use dnam_engine::stages::v3_core::V3MatryoshkaModel;

    use super::*;
    use crate::solver::v3_varpro::model_f64::V3TrainStem;

    #[test]
    fn test_codec_roundtrip_and_chain_rule_matches_fd() {
        let mut m =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(true, false));
        m.max_active_groups = 1;
        m.active_tier_weights = [1.0, 0.0, 0.0, 0.0];
        m.svf_base_g = 0.11;
        m.feat_slow_rel_beta = 1.2e-5;
        m.feat_pinch_k = 12.0;
        m.feat_slow_det_comp = 3.5;
        m.blocks[0].w_in_n[0][0] = 0.7;
        m.blocks[0].k_sag[0] = 0.45;

        let u0 = pack_params(&mut m, true);
        let mut jac = vec![0.0; u0.len()];
        unpack_and_derivs(&mut m, true, &u0, &mut jac);
        let u1 = pack_params(&mut m, true);
        for (a, b) in u0.iter().zip(&u1) {
            assert!((a - b).abs() < 1.0e-6);
        }

        let n = 128;
        let in_l: Vec<f32> = (0..n).map(|i| 0.4 * ((i as f32) * 0.07).sin()).collect();
        let in_r: Vec<f32> = (0..n).map(|i| 0.35 * ((i as f32) * 0.11).cos()).collect();
        let tgt_l: Vec<f32> = in_l.iter().map(|&x| (1.2 * x).tanh()).collect();
        let tgt_r: Vec<f32> = in_r.iter().map(|&x| (1.1 * x).tanh()).collect();
        let ds = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l,
                in_r,
                tgt_l,
                tgt_r,
                warmup: 16,
            }],
            48_000.0,
        );

        let mut gm = m.zero_like();
        let mut ws = AdjointWorkspace::default();
        let _ = evaluate_matryoshka_loss_and_grad_with_backend::<
            crate::solver::backends::ScalarTrainerBackend,
        >(&m, &ds, &mut ws, Some(&mut gm));
        let mut gu = vec![0.0; u0.len()];
        pack_chain_grad(&mut gm, true, &jac, &mut gu);

        for idx in [0, 3, 10, 25, u0.len() - 5] {
            let mut up = u0.clone();
            let mut um = u0.clone();
            up[idx] += 1.0e-5;
            um[idx] -= 1.0e-5;
            let mut mp = m.clone();
            unpack_and_derivs(&mut mp, true, &up, &mut jac);
            let lp = evaluate_matryoshka_loss_f64_with_backend::<
                crate::solver::backends::ScalarTrainerBackend,
            >(&mp, &ds, &mut ws);
            let mut mm = m.clone();
            unpack_and_derivs(&mut mm, true, &um, &mut jac);
            let lm = evaluate_matryoshka_loss_f64_with_backend::<
                crate::solver::backends::ScalarTrainerBackend,
            >(&mm, &ds, &mut ws);
            let fd = (lp - lm) / 2.0e-5;
            let err = (gu[idx] - fd).abs() / gu[idx].abs().max(fd.abs()).max(1.0e-8);
            assert!(err < 1.0e-4, "idx={idx} gu={} fd={fd} err={err}", gu[idx]);
        }
    }

    #[test]
    fn test_drop_guard_restores_best_model_on_early_abort() {
        let m0 =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(false, true));
        let mut m = m0.clone();
        {
            let _guard = ModelBestGuard {
                best: m0.clone(),
                best_loss: 1.0,
                target: &mut m,
                committed: false,
            };
        }
        assert_eq!(m.svf_base_g, m0.svf_base_g);
    }
}
