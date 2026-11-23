//! Stereo `ro_pole_48k` 1-D sweep and per-tier RHS extraction helpers for Pass-3 nested `VarPro`
//! (`ref[train.pipeline.nested-varpro-schur]`).

use crate::control::clamp::{MAX_GAIN_6DB_LIN, evaluate_readout_spectral_magnitudes};
use crate::control::stability::MAX_RO_POLE_48K_F64;
use crate::linalg::cholesky::solve_uplo_u_cholesky;
use crate::wide_gram::uplo_u_index;

use super::super::model::SonaModelF64;
use super::eval_hp_post_for_ro;
use super::features::{StereoFeatureSet, TIER_KAUTZ_CAPS, has_higher_tier_wet_op_t};
use super::spectral_refine::{compute_channel_mse, trace_scaled_ridge};

pub(super) struct SweepStereoResult {
    pub best_ro: f64,
    pub best_gram_l: Vec<f64>,
    pub best_rhs_l: Vec<f64>,
    pub best_hp_l: Vec<f32>,
    pub best_gram_r: Vec<f64>,
    pub best_rhs_r: Vec<f64>,
    pub best_hp_r: Vec<f32>,
}

pub(super) fn patch_col1_gram_and_rhs(
    base_gram: &[f64],
    base_rhs: &[f64],
    feats_post: &[f32],
    tgts_post: &[f32],
    hp_post: &[f32],
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let m = tgts_post.len();
    let mut g = base_gram.to_vec();
    let mut r = base_rhs.to_vec();
    g[uplo_u_index(0, 1)] = hp_post
        .iter()
        .zip(&feats_post[0..m])
        .map(|(&h, &f)| f64::from(h) * f64::from(f))
        .sum();
    g[uplo_u_index(1, 1)] = hp_post
        .iter()
        .map(|&h| {
            let hf = f64::from(h);
            hf * hf
        })
        .sum();
    for c in 2..dim {
        let col_slice = &feats_post[c * m..(c + 1) * m];
        g[uplo_u_index(1, c)] = hp_post
            .iter()
            .zip(col_slice)
            .map(|(&h, &f)| f64::from(h) * f64::from(f))
            .sum();
    }
    r[1] = hp_post
        .iter()
        .zip(tgts_post)
        .map(|(&h, &y)| f64::from(h) * f64::from(y))
        .sum();
    (g, r)
}

pub(super) fn sweep_ro_pole_48k_joint_stereo(
    model: &SonaModelF64,
    fset: &StereoFeatureSet,
    base_gram_l: &[f64],
    base_rhs_l: &[f64],
    base_gram_r: &[f64],
    base_rhs_r: &[f64],
    w: usize,
) -> SweepStereoResult {
    let dim = fset.dims[3];
    let m = fset.tgts_l_post.len();
    let candidates = [0.0, 0.5, 0.8, 0.92, 0.97, 0.992, MAX_RO_POLE_48K_F64];

    let mut best_ro = 0.0_f64;
    let mut best_err = f64::INFINITY;
    let mut err_zero = f64::INFINITY;
    let mut best_gram_l = base_gram_l.to_vec();
    let mut best_rhs_l = base_rhs_l.to_vec();
    let mut best_gram_r = base_gram_r.to_vec();
    let mut best_rhs_r = base_rhs_r.to_vec();
    let mut best_hp_l = vec![0.0_f32; m];
    let mut best_hp_r = vec![0.0_f32; m];

    for &ro in &candidates {
        let hp_l = eval_hp_post_for_ro(
            model,
            &fset.raw_l_col0,
            fset.raw_l_2x.as_deref(),
            ro,
            w,
            false,
            4,
        );
        let hp_r = eval_hp_post_for_ro(
            model,
            &fset.raw_r_col0,
            fset.raw_r_2x.as_deref(),
            ro,
            w,
            true,
            4,
        );
        let (gl, rl) = patch_col1_gram_and_rhs(
            base_gram_l,
            base_rhs_l,
            &fset.feats_l_post,
            &fset.tgts_l_post,
            &hp_l,
            dim,
        );
        let (gr, rr) = patch_col1_gram_and_rhs(
            base_gram_r,
            base_rhs_r,
            &fset.feats_r_post,
            &fset.tgts_r_post,
            &hp_r,
            dim,
        );
        let ridge_l = trace_scaled_ridge(&gl, dim, 1.0e-5);
        let ridge_r = trace_scaled_ridge(&gr, dim, 1.0e-5);
        let (Ok(wl), Ok(wr)) = (
            solve_uplo_u_cholesky(&gl, &rl, ridge_l),
            solve_uplo_u_cholesky(&gr, &rr, ridge_r),
        ) else {
            continue;
        };
        if ro > 0.0 {
            let fir0_off = 2 + 16 * 3;
            let fir0_l = if fset.active_fir_taps > 0 {
                wl.get(fir0_off).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            let fir0_r = if fset.active_fir_taps > 0 {
                wr.get(fir0_off).copied().unwrap_or(0.0)
            } else {
                0.0
            };
            let mags_l = evaluate_readout_spectral_magnitudes(
                &[wl[0] + fir0_l, wl[1]],
                ro,
                model.sample_rate_hz,
                16,
            );
            let mags_r = evaluate_readout_spectral_magnitudes(
                &[wr[0] + fir0_r, wr[1]],
                ro,
                model.sample_rate_hz,
                16,
            );
            if mags_l.iter().chain(&mags_r).any(|&v| v > MAX_GAIN_6DB_LIN) {
                continue;
            }
        }
        let mut fl_tmp = fset.feats_l_post.clone();
        fl_tmp[m..(2 * m)].copy_from_slice(&hp_l);
        let mut fr_tmp = fset.feats_r_post.clone();
        fr_tmp[m..(2 * m)].copy_from_slice(&hp_r);
        let err = compute_channel_mse(&wl, &fl_tmp, &fset.tgts_l_post, dim)
            + compute_channel_mse(&wr, &fr_tmp, &fset.tgts_r_post, dim);
        if ro == 0.0 {
            err_zero = err;
        }
        if (ro == 0.0 && err < best_err) || (ro > 0.0 && err < err_zero * 0.98 && err < best_err) {
            best_err = err;
            best_ro = ro;
            best_gram_l = gl;
            best_rhs_l = rl;
            best_hp_l = hp_l;
            best_gram_r = gr;
            best_rhs_r = rr;
            best_hp_r = hp_r;
        }
    }

    SweepStereoResult {
        best_ro,
        best_gram_l,
        best_rhs_l,
        best_hp_l,
        best_gram_r,
        best_rhs_r,
        best_hp_r,
    }
}

/// Computes per-tier RHS `feats^T y` vectors over post-warmup targets when `active_k` exceeds a
/// lower tier's Kautz pair cap (and higher `Op-T` quads are dormant); otherwise reuses the Ultra
/// RHS prefix (`ti == 0..3` with higher wet `Op-T` solve independently against tier features).
pub(super) fn build_per_tier_rhs(
    model: &SonaModelF64,
    feats_post: &[f32],
    fset: &StereoFeatureSet,
    ultra_rhs: &[f64],
    active_k: usize,
    is_right: bool,
) -> [Vec<f64>; 4] {
    let m = fset.tgts_l_post.len();
    core::array::from_fn(|ti| {
        let dim_t = fset.dims[ti];
        if ti == 3 || (active_k <= TIER_KAUTZ_CAPS[ti] && !has_higher_tier_wet_op_t(model, ti)) {
            ultra_rhs[..dim_t].to_vec()
        } else {
            let tgts = if is_right {
                fset.tier_tgts_r(ti)
            } else {
                fset.tier_tgts_l(ti)
            };
            (0..dim_t)
                .map(|c| {
                    let col = &feats_post[c * m..(c + 1) * m];
                    col.iter()
                        .zip(tgts)
                        .map(|(&f, &y)| f64::from(f) * f64::from(y))
                        .sum::<f64>()
                })
                .collect()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_col1_gram_and_rhs_exactness() {
        let m = 4_usize;
        let dim = 3_usize;
        let feats_post = [
            1.0_f32, 2.0, -1.0, 0.5, // col 0
            0.0, 0.0, 0.0, 0.0, // col 1 placeholder
            -0.5, 1.5, 0.25, -1.0, // col 2
        ];
        let tgts_post = [0.25_f32, -0.5, 1.0, 0.75];
        let hp_post = [0.5_f32, -0.25, 0.75, -0.5];
        let base_gram = vec![0.0_f64; 6];
        let base_rhs = vec![0.0_f64; 3];
        let (g, r) = patch_col1_gram_and_rhs(
            &base_gram,
            &base_rhs,
            &feats_post,
            &tgts_post,
            &hp_post,
            dim,
        );
        let g01: f64 = hp_post
            .iter()
            .zip(&feats_post[0..m])
            .map(|(&h, &f)| f64::from(h) * f64::from(f))
            .sum();
        let g11: f64 = hp_post.iter().map(|&h| f64::from(h) * f64::from(h)).sum();
        let g12: f64 = hp_post
            .iter()
            .zip(&feats_post[2 * m..3 * m])
            .map(|(&h, &f)| f64::from(h) * f64::from(f))
            .sum();
        let r1: f64 = hp_post
            .iter()
            .zip(&tgts_post)
            .map(|(&h, &y)| f64::from(h) * f64::from(y))
            .sum();
        assert!((g[uplo_u_index(0, 1)] - g01).abs() < 1.0e-12);
        assert!((g[uplo_u_index(1, 1)] - g11).abs() < 1.0e-12);
        assert!((g[uplo_u_index(1, 2)] - g12).abs() < 1.0e-12);
        assert!((r[1] - r1).abs() < 1.0e-12);
    }
}
