//! Full-capture `Welch-33` SQP-KKT spectral tilt refinement for Pass-3 `VarPro`
//! (`ref[train.pipeline.nested-varpro-schur]`, `ref[qual.companion.ltas]`).

use super::ridge_solve::{
    MAX_CANDIDATE_ABS_BEFORE_SNAP, MAX_READOUT_L1_NORM, MIN_RIDGE_ABS_FLOOR, compute_channel_mse,
    within_6db_and_bounds,
};
use crate::linalg::spectral_kkt::{AffineBoxKktProblem, LtasKktConfig, solve_ltas_affine_box_kkt};
use crate::losses::Welch33TargetCache;
use crate::wide_gram::{uplo_u_index, uplo_u_packed_len};

const MAX_SQP_ITERS: usize = 3;
const SQP_RHO_SCHEDULE: &[f64] = &[0.25, 1.0, 4.0, 16.0, 64.0, 256.0];
const TRACK_FRAC_OF_AIM: f64 = 0.65;
/// Minimum target band power relative to `peak_p` (`-120 dB`) included in Pass-3 `f32`-feature
/// Welch-33 SQP-KKT refinement, staying `>= 24 dB` above the `f32` (`24`-bit) quantization noise
/// floor so unexcited rounding-noise bands never derail active signal-band tilt control.
const WELCH_SQP_MIN_REL_POWER: f64 = 1.0e-12;
const _: () = assert!(
    MAX_SQP_ITERS >= 1
        && SQP_RHO_SCHEDULE.len() >= 2
        && TRACK_FRAC_OF_AIM > 0.0
        && TRACK_FRAC_OF_AIM < 1.0
        && WELCH_SQP_MIN_REL_POWER > 0.0
        && WELCH_SQP_MIN_REL_POWER <= 1.0e-10
);

/// Inputs for full-capture `Welch-33` SQP-KKT polishing of one channel/tier linear readout.
pub(super) struct WelchSqpParams<'a> {
    pub base_w: &'a [f64],
    pub gram_sub: &'a [f64],
    pub rhs_sub: &'a [f64],
    pub lambda_diag: &'a [f64],
    pub feats_post: &'a [f32],
    pub tgts_post: &'a [f32],
    pub raw_target_post: &'a [f32],
    pub welch: &'a Welch33TargetCache,
    pub sr_hz: f64,
    pub ro_pole_48k: f64,
    pub budget_db: f64,
    pub polish_target_db: f64,
    pub base_mse: f64,
    pub mse_limit: f64,
}

#[inline]
fn is_sqp_band_active(welch: &Welch33TargetCache, b: usize) -> bool {
    welch.active[b] && welch.tgt_band_p[b] >= welch.peak_p * WELCH_SQP_MIN_REL_POWER
}

#[inline]
fn band_tilt_db(welch: &Welch33TargetCache, band_p: &[f64; 33], b: usize) -> f64 {
    if !is_sqp_band_active(welch, b) {
        return 0.0;
    }
    let ratio = (band_p[b] + welch.floor) / (welch.tgt_band_p[b] + welch.floor);
    welch.band_snr_w[b] * 10.0 * ratio.max(1.0e-30).log10().abs()
}

type WelchWindowSpec = (Vec<f64>, Vec<f64>);

/// Evaluates canonical full-capture `Welch-33` worst active band tilt (in dB) for
/// `total_pred = (raw_target_post - tgts_post) + F * weights`, optionally retaining Welch FFT
/// window spectra for exact band RMS adjoint computation.
pub(super) fn eval_total_pred_welch33_tilt_db(
    welch: &Welch33TargetCache,
    raw_target_post: &[f32],
    tgts_post: &[f32],
    feats_post: &[f32],
    weights: &[f64],
    retain_spectra: bool,
) -> (f64, [f64; 33], Vec<WelchWindowSpec>) {
    let dim = weights.len();
    let m = tgts_post.len().min(raw_target_post.len());
    if m == 0 || dim == 0 || welch.len != m || feats_post.len() < dim.saturating_mul(m) {
        return (0.0, [0.0; 33], Vec::new());
    }
    let mut pred = vec![0.0_f64; m];
    for t in 0..m {
        let mut p = f64::from(raw_target_post[t]) - f64::from(tgts_post[t]);
        for (c, &wc) in weights.iter().enumerate() {
            p += wc * f64::from(feats_post[c * m + t]);
        }
        pred[t] = p;
    }
    let (band_p, win_specs) = welch.eval_pred_band_powers_and_spectra(&pred, retain_spectra);
    let mut max_tilt = 0.0_f64;
    for b in 0..33 {
        max_tilt = max_tilt.max(band_tilt_db(welch, &band_p, b));
    }
    (max_tilt, band_p, win_specs)
}

struct WelchSqpLinearBox {
    band_rows: Vec<f64>,
    band_const: Vec<f64>,
    band_lo: Vec<f64>,
    band_hi: Vec<f64>,
}

fn build_welch_sqp_box_system(
    p: &WelchSqpParams<'_>,
    cur_w: &[f64],
    aim_db: f64,
    tracked: &mut [bool; 33],
    grad_y: &mut [f64],
) -> Option<WelchSqpLinearBox> {
    let dim = cur_w.len();
    let m = p.tgts_post.len();
    let (_cur_tilt, band_p, win_specs) = eval_total_pred_welch33_tilt_db(
        p.welch,
        p.raw_target_post,
        p.tgts_post,
        p.feats_post,
        cur_w,
        true,
    );
    let hold_floor = p.welch.peak_p * 1.0e-8;
    let mut active_idx = Vec::new();
    for b in 0..33 {
        if is_sqp_band_active(p.welch, b)
            && (tracked[b]
                || p.welch.tgt_band_p[b] >= hold_floor
                || band_tilt_db(p.welch, &band_p, b) > TRACK_FRAC_OF_AIM * aim_db)
        {
            tracked[b] = true;
            active_idx.push(b);
        }
    }
    if active_idx.is_empty() {
        return None;
    }

    let n_bands = active_idx.len();
    let mut band_rows = Vec::with_capacity(n_bands * dim);
    let mut band_const = Vec::with_capacity(n_bands);
    let mut band_lo = Vec::with_capacity(n_bands);
    let mut band_hi = Vec::with_capacity(n_bands);
    for &b in &active_idx {
        let tgt_r = (p.welch.tgt_band_p[b] + p.welch.floor).sqrt();
        let inv_tgt_r = 1.0 / tgt_r.max(1.0e-18);
        // Exact chain rule d/dy sqrt(P_b + F) = sqrt(P_b / (P_b + F)) * d/dy sqrt(P_b) (`= 1.0` when floor == 0).
        let floor_chain = if p.welch.floor > 0.0 && band_p[b] > 0.0 {
            (band_p[b] / (band_p[b] + p.welch.floor)).sqrt()
        } else {
            1.0
        };
        let row_scale = floor_chain * inv_tgt_r;
        p.welch
            .accum_band_rms_adjoint(&win_specs, b, band_p[b], grad_y);
        let mut dot_cur = 0.0_f64;
        for c in 0..dim {
            let col = &p.feats_post[c * m..(c + 1) * m];
            let g: f64 = col
                .iter()
                .zip(&*grad_y)
                .map(|(&fv, &gy)| f64::from(fv) * gy)
                .sum();
            let g_rel = g * row_scale;
            band_rows.push(g_rel);
            dot_cur += g_rel * cur_w[c];
        }
        let r_cur_rel = (band_p[b] + p.welch.floor).sqrt() * inv_tgt_r;
        band_const.push(r_cur_rel - dot_cur);

        let eff_db = (aim_db / p.welch.band_snr_w[b].max(1.0e-6)).max(1.0e-4);
        band_lo.push(10.0_f64.powf(-eff_db / 20.0));
        band_hi.push(10.0_f64.powf(eff_db / 20.0));
    }
    Some(WelchSqpLinearBox {
        band_rows,
        band_const,
        band_lo,
        band_hi,
    })
}

struct WelchSqpSearchState<'a> {
    aim_db: f64,
    l1_cap: f64,
    best_tilt: &'a mut f64,
    best_sol: &'a mut Option<(Vec<f64>, f64, f64)>,
}

fn search_welch_sqp_step(
    p: &WelchSqpParams<'_>,
    gram_reg: &[f64],
    lin: &WelchSqpLinearBox,
    cur_w: &[f64],
    st: &mut WelchSqpSearchState<'_>,
) -> Option<(Vec<f64>, f64)> {
    let dim = cur_w.len();
    let n_bands = lin.band_const.len();
    let prob = AffineBoxKktProblem {
        num_bands: n_bands,
        feature_dim: dim,
        gram_uplo: gram_reg,
        rhs: &p.rhs_sub[..dim],
        band_rows: &lin.band_rows,
        band_const: &lin.band_const,
        band_lo: &lin.band_lo,
        band_hi: &lin.band_hi,
    };
    let mut step_best: Option<(Vec<f64>, f64)> = None;
    for &rho in SQP_RHO_SCHEDULE {
        let Ok(kkt) = solve_ltas_affine_box_kkt(
            &prob,
            LtasKktConfig {
                ridge: MIN_RIDGE_ABS_FLOOR,
                max_outer_iters: 4,
                max_tilt_db: st.aim_db,
                penalty_rho: rho,
            },
        ) else {
            continue;
        };
        let mut hit_aim = false;
        for &alpha in &[1.0_f64, 0.5, 0.25] {
            let cand_w: Vec<f64> = if (alpha - 1.0).abs() <= f64::EPSILON {
                kkt.weights.clone()
            } else {
                cur_w
                    .iter()
                    .zip(&kkt.weights)
                    .map(|(&cw, &kw)| (1.0 - alpha) * cw + alpha * kw)
                    .collect()
            };
            if !within_6db_and_bounds(
                &cand_w,
                p.ro_pole_48k,
                p.sr_hz,
                MAX_CANDIDATE_ABS_BEFORE_SNAP,
            ) || cand_w.iter().map(|v| v.abs()).sum::<f64>() > st.l1_cap
            {
                continue;
            }
            let cand_mse = compute_channel_mse(&cand_w, p.feats_post, p.tgts_post, dim);
            if !cand_mse.is_finite() || cand_mse > p.mse_limit {
                continue;
            }
            let (cand_tilt, _, _) = eval_total_pred_welch33_tilt_db(
                p.welch,
                p.raw_target_post,
                p.tgts_post,
                p.feats_post,
                &cand_w,
                false,
            );
            if !cand_tilt.is_finite() {
                continue;
            }
            if step_best.as_ref().is_none_or(|(_, t)| cand_tilt < *t) {
                step_best = Some((cand_w.clone(), cand_tilt));
            }
            if cand_tilt < *st.best_tilt {
                *st.best_tilt = cand_tilt;
                *st.best_sol = Some((cand_w, cand_mse, cand_tilt));
                hit_aim = cand_tilt <= st.aim_db;
                break;
            }
        }
        if hit_aim {
            break;
        }
    }
    step_best
}

/// Runs sequential quadratic programming over linearized `Welch-33` band RMS bounds using exact
/// Euler-homogeneous Welch adjoints (`band_rows[b] = F^T d(sqrt(P_b))/dy`).
///
/// Returns `Some((weights, mse, tilt_db))` when a feasible `within_6db_and_bounds` candidate improves
/// `Welch-33` worst-band tilt within `p.mse_limit`.
pub(super) fn polish_weights_welch33_sqp_kkt(
    p: &WelchSqpParams<'_>,
) -> Option<(Vec<f64>, f64, f64)> {
    let dim = p.base_w.len();
    let m = p.tgts_post.len();
    let packed = uplo_u_packed_len(dim);
    if dim == 0
        || m == 0
        || p.welch.len != m
        || p.raw_target_post.len() != m
        || p.feats_post.len() < dim.saturating_mul(m)
        || p.gram_sub.len() < packed
        || p.rhs_sub.len() < dim
        || p.lambda_diag.len() < dim
    {
        return None;
    }

    let (init_tilt, _, _) = eval_total_pred_welch33_tilt_db(
        p.welch,
        p.raw_target_post,
        p.tgts_post,
        p.feats_post,
        p.base_w,
        false,
    );
    if !init_tilt.is_finite() || init_tilt <= p.polish_target_db {
        return None;
    }

    let mut gram_reg = p.gram_sub[..packed].to_vec();
    for (j, &lam) in p.lambda_diag.iter().enumerate().take(dim) {
        gram_reg[uplo_u_index(j, j)] += lam.max(0.0);
    }

    let box_db = (0.55 * p.budget_db)
        .max(p.polish_target_db)
        .min(0.85 * p.budget_db);
    let aim_db = if init_tilt > 0.80 * p.budget_db {
        box_db
    } else {
        p.polish_target_db
    };
    let base_l1: f64 = p.base_w.iter().map(|v| v.abs()).sum();
    let l1_cap = (base_l1 * 1.5 + 1.5).min(MAX_READOUT_L1_NORM);

    let mut cur_w = p.base_w.to_vec();
    let mut best_sol: Option<(Vec<f64>, f64, f64)> = None;
    let mut best_tilt = init_tilt;
    let mut tracked = [false; 33];
    let mut grad_y = vec![0.0_f64; m];

    for _sqp_iter in 0..MAX_SQP_ITERS {
        let Some(lin) = build_welch_sqp_box_system(p, &cur_w, aim_db, &mut tracked, &mut grad_y)
        else {
            break;
        };
        let mut search_st = WelchSqpSearchState {
            aim_db,
            l1_cap,
            best_tilt: &mut best_tilt,
            best_sol: &mut best_sol,
        };
        let Some((w, t)) = search_welch_sqp_step(p, &gram_reg, &lin, &cur_w, &mut search_st) else {
            break;
        };
        cur_w = w;
        if t <= aim_db {
            break;
        }
    }

    let _ = p.base_mse;
    best_sol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::pass3_varpro::spectral_refine::ridge_solve::build_scale_invariant_diag_ridge;

    #[test]
    fn test_welch_rms_adjoint_satisfies_euler_identity_and_fixes_tilt() {
        let sr = 48_000.0_f64;
        let m = 4_096_usize;
        let mut x0 = vec![0.0_f32; m];
        let mut x1 = vec![0.0_f32; m];
        let mut offset = vec![0.0_f64; m];
        let mut tgt = vec![0.0_f32; m];
        let mut raw = vec![0.0_f32; m];
        for t in 0..m {
            let tf = (t as f64) / sr;
            let low = 0.25 * (core::f64::consts::TAU * 250.0 * tf).sin();
            let high = 0.25 * (core::f64::consts::TAU * 5_000.0 * tf).sin();
            let off = 0.10 * (core::f64::consts::TAU * 1_000.0 * tf).sin();
            x0[t] = low as f32;
            x1[t] = high as f32;
            offset[t] = off;
            let target_total = off + 0.6 * low + 0.6 * high;
            raw[t] = target_total as f32;
            tgt[t] = (target_total - off) as f32;
        }
        let raw_f64: Vec<f64> = raw.iter().map(|&v| f64::from(v)).collect();
        let welch = Welch33TargetCache::new(&raw_f64, 0, sr).unwrap();

        // Check Euler identity <d sqrt(P_b)/dy, y> == sqrt(P_b(y)) on total prediction.
        let weights = [0.6_f64, 0.42_f64];
        let mut feats = Vec::with_capacity(2 * m);
        feats.extend_from_slice(&x0);
        feats.extend_from_slice(&x1);
        let (tilt0, band_p, win_specs) =
            eval_total_pred_welch33_tilt_db(&welch, &raw, &tgt, &feats, &weights, true);
        assert!(tilt0 > 2.0, "expected tilted 5kHz band > 2 dB, got {tilt0}");

        let pred: Vec<f64> = (0..m)
            .map(|t| {
                (f64::from(raw[t]) - f64::from(tgt[t]))
                    + weights[0] * f64::from(x0[t])
                    + weights[1] * f64::from(x1[t])
            })
            .collect();
        let mut grad = vec![0.0_f64; m];
        for b in 0..33 {
            if welch.active[b] && band_p[b] > 1.0e-12 {
                welch.accum_band_rms_adjoint(&win_specs, b, band_p[b], &mut grad);
                let dot: f64 = grad.iter().zip(&pred).map(|(&g, &y)| g * y).sum();
                let rms = band_p[b].sqrt();
                assert!(
                    (dot - rms).abs() <= 1.0e-9 * rms.max(1.0),
                    "Euler identity mismatch on band {b}: dot={dot}, rms={rms}"
                );
            }
        }

        // Verify SQP-KKT overrides a biased normal-equation RHS (tilted toward 0.50 on high tone)
        // using Welch-33 adjoints to pull `polished[1]` back toward 0.60 within <= 0.10 dB tilt.
        let tgt_biased: Vec<f32> = (0..m)
            .map(|t| (0.6 * f64::from(x0[t]) + 0.50 * f64::from(x1[t])) as f32)
            .collect();
        let mut gram = vec![0.0_f64; uplo_u_packed_len(2)];
        let mut rhs_biased = vec![0.0_f64; 2];
        for j in 0..2 {
            let fj = &feats[j * m..(j + 1) * m];
            rhs_biased[j] = fj
                .iter()
                .zip(&tgt_biased)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            for i in 0..=j {
                let fi = &feats[i * m..(i + 1) * m];
                gram[uplo_u_index(i, j)] = fi
                    .iter()
                    .zip(fj)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
            }
        }
        let biased_w = [0.6_f64, 0.50_f64];
        let (lambda_diag, _) = build_scale_invariant_diag_ridge(&gram, 2, 1.0e-5);
        let base_mse = compute_channel_mse(&biased_w, &feats, &tgt_biased, 2);
        let sig_mse = compute_channel_mse(&[0.0, 0.0], &feats, &tgt_biased, 2);
        let raw_biased_tgts: Vec<f32> = (0..m)
            .map(|t| (f64::from(raw[t]) - offset[t]) as f32)
            .collect();
        let (polished, _p_mse, p_tilt) = polish_weights_welch33_sqp_kkt(&WelchSqpParams {
            base_w: &biased_w,
            gram_sub: &gram,
            rhs_sub: &rhs_biased,
            lambda_diag: &lambda_diag,
            feats_post: &feats,
            tgts_post: &raw_biased_tgts,
            raw_target_post: &raw,
            welch: &welch,
            sr_hz: sr,
            ro_pole_48k: 0.0,
            budget_db: 0.15,
            polish_target_db: 0.03,
            base_mse,
            mse_limit: (base_mse * 2.0).max(0.08 * sig_mse),
        })
        .expect("SQP-KKT should polish biased normal-equation objective via Welch33 bounds");
        assert!(p_tilt <= 0.10, "expected p_tilt <= 0.10 dB, got {p_tilt}");
        assert!(
            (polished[0] - 0.6).abs() < 0.015 && (polished[1] - 0.6).abs() < 0.015,
            "polished={polished:?}"
        );
    }
}
