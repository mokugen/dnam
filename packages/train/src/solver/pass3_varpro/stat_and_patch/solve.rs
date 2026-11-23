//! Anchored `uplo='U'` Cholesky solver, exact-`f64` Ozaki PCG refinement, and quadratic
//! error helpers for [`super::patch_model_readouts_weighted_cholesky`]
//! (`ref[train.pipeline.nested-varpro-schur]`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};

use crate::error::TrainError;
use crate::linalg::cholesky::{
    MIN_CHOLESKY_RIDGE, PcgRefinementSpec, factorize_uplo_u_cholesky, refine_cholesky_lsq_pcg_f64,
    solve_factorized_uplo_u,
};
use crate::wide_gram::uplo_u_index;

use super::super::super::model::{SonaModelF64, TierReadoutPrefixF64};

#[inline]
fn snap_ro_fp16(v: f64) -> f64 {
    let x = if v.is_finite() {
        (v as f32).clamp(-64.0, 64.0)
    } else {
        0.0
    };
    let s = fp16_to_f32(f32_to_fp16(if x == 0.0 { 0.0 } else { x }));
    f64::from(if s == 0.0 { 0.0 } else { s })
}

#[inline]
fn snap_fir_f32(v: f64) -> f64 {
    let x = if v.is_finite() {
        (v as f32).clamp(-16.0, 16.0)
    } else {
        0.0
    };
    f64::from(if x == 0.0 { 0.0 } else { x })
}

pub(super) fn snap_prefix_readout_weights(pfx: &mut TierReadoutPrefixF64) {
    for c in 0..2 {
        pfx.pre_conv_u_gain[c] = snap_ro_fp16(pfx.pre_conv_u_gain[c]);
        pfx.pre_conv_hp_gain[c] = snap_ro_fp16(pfx.pre_conv_hp_gain[c]);
        pfx.direct_s2_u_gain[c] = snap_ro_fp16(pfx.direct_s2_u_gain[c]);
        pfx.direct_s2_hp_gain[c] = snap_ro_fp16(pfx.direct_s2_hp_gain[c]);
    }
    for row in &mut pfx.w_ro {
        for cell in row {
            *cell = snap_ro_fp16(*cell);
        }
    }
    for i in 0..pfx.fir_l.len() {
        pfx.fir_l[i] = snap_fir_f32(pfx.fir_l[i]);
        pfx.fir_r[i] = snap_fir_f32(pfx.fir_r[i]);
    }
}

const OZAKI_PCG_MAX_ITERS: usize = 16;
const _: () = assert!(OZAKI_PCG_MAX_ITERS > 0 && OZAKI_PCG_MAX_ITERS <= 16);

pub(super) struct AnchoredSolveResult {
    pub sol: Vec<f64>,
    pub lambda_diag: Vec<f64>,
    pub anchor_lam_diag: Vec<f64>,
    pub zero_ridge: f64,
    pub r_upper: Option<Vec<f64>>,
}

pub(super) struct OzakiStemChannel<'a> {
    pub feats: &'a [f32],
    pub tgts: &'a [f32],
    pub num_frames: usize,
    pub weight: f64,
}

pub(super) struct StereoOzakiRefineCtx<'a> {
    pub left: OzakiStemChannel<'a>,
    pub right: OzakiStemChannel<'a>,
    pub w0_l: &'a [f64],
    pub w0_r: &'a [f64],
    pub dim: usize,
}

fn refine_ozaki_channel(
    solved: &mut AnchoredSolveResult,
    stem: &OzakiStemChannel<'_>,
    anchor: &[f64],
    dim: usize,
) {
    let Some(ref r_upper) = solved.r_upper else {
        return;
    };
    let zero_anchor = anchor.iter().take(dim).all(|&v| v == 0.0);
    solved.sol = refine_cholesky_lsq_pcg_f64(
        &solved.sol,
        &PcgRefinementSpec {
            feats_col_major: stem.feats,
            targets: stem.tgts,
            num_frames: stem.num_frames,
            dim,
            stem_weight: stem.weight,
            anchor_lam_diag: &solved.anchor_lam_diag,
            zero_ridge: solved.zero_ridge,
            anchor: (!zero_anchor).then_some(anchor),
            r_upper,
            max_abs_w: 7.5,
            max_delta_w: (!zero_anchor).then_some(2.5),
            max_iters: OZAKI_PCG_MAX_ITERS,
        },
    );
}

pub(super) fn refine_stereo_ozaki_if_single_stem(
    sol_l: &mut AnchoredSolveResult,
    sol_r: &mut AnchoredSolveResult,
    ctx: &StereoOzakiRefineCtx<'_>,
) {
    refine_ozaki_channel(sol_l, &ctx.left, ctx.w0_l, ctx.dim);
    refine_ozaki_channel(sol_r, &ctx.right, ctx.w0_r, ctx.dim);
}

pub(super) fn pack_existing_weights_for_tier(
    model: &SonaModelF64,
    active_taps: usize,
    max_tier_idx: usize,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let pfx_opt = model.tier_overrides[max_tier_idx].as_deref();
    let dir_u = pfx_opt.map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain);
    let dir_hp = pfx_opt.map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain);
    let w_ro = pfx_opt.map_or(&model.w_ro, |p| &p.w_ro);
    let fir_l = pfx_opt.map_or(&model.fir_l, |p| &p.fir_l);
    let fir_r = pfx_opt.map_or(&model.fir_r, |p| &p.fir_r);

    let mut wl = vec![dir_u[0], dir_hp[0]];
    let mut wr = vec![dir_u[1], dir_hp[1]];
    let mut is_broadband_col = vec![true, true];
    let tier_defs = [
        (0..16_usize, 0..active_taps.min(32)),
        (
            16..24_usize,
            32..(32 + active_taps.saturating_sub(32).min(16)),
        ),
        (
            24..32_usize,
            48..(48 + active_taps.saturating_sub(48).min(48)),
        ),
        (
            32..40_usize,
            96..(96 + active_taps.saturating_sub(96).min(32)),
        ),
    ];
    for (ti, (lane_range, fir_range)) in tier_defs.into_iter().enumerate() {
        if ti > max_tier_idx {
            break;
        }
        for l in lane_range {
            wl.push(w_ro[l][6]);
            wr.push(w_ro[l][7]);
            wl.push(w_ro[l][8]);
            wr.push(w_ro[l][9]);
            wl.push(w_ro[l][10]);
            wr.push(w_ro[l][11]);
            is_broadband_col.extend_from_slice(&[false, false, false]);
        }
        for k in fir_range {
            wl.push(fir_l[k]);
            wr.push(fir_r[k]);
            is_broadband_col.push(ti == 0 && k == 0);
        }
    }
    (wl, wr, is_broadband_col)
}

pub(super) fn solve_anchored_uplo_u(
    gram: &[f64],
    rhs: &[f64],
    anchor: &[f64],
    is_broadband_col: &[bool],
    dim: usize,
) -> Result<AnchoredSolveResult, TrainError> {
    let mut trace = 0.0_f64;
    for j in 0..dim {
        trace += gram[uplo_u_index(j, j)].max(0.0);
    }
    let mean_diag = (trace / (dim.max(1) as f64)).max(1.0e-12);
    let zero_anchor = anchor.iter().take(dim).all(|&v| v == 0.0);
    let (anchor_lam_broadband, anchor_lam_detail) = if zero_anchor {
        (0.0, 0.0)
    } else {
        (1.0e-5 * mean_diag, 5.0e-2 * mean_diag)
    };
    let col_lam_at = |j: usize| -> f64 {
        if zero_anchor {
            0.0
        } else if is_broadband_col.get(j).copied().unwrap_or(false) {
            anchor_lam_broadband
        } else {
            anchor_lam_detail
        }
    };
    let schedule: &[f64] = if zero_anchor {
        &[1.0e-12, 1.0e-9, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    } else {
        &[1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    };

    let mut r_anchored = rhs[..dim].to_vec();
    if !zero_anchor {
        for (j, &w0) in anchor.iter().enumerate().take(dim) {
            r_anchored[j] += col_lam_at(j) * w0;
        }
    }

    let mut last_sol = vec![0.0_f64; dim];
    let init_diag = (schedule[0] * mean_diag).max(1.0e-15);
    let anchor_lam_diag: Vec<f64> = (0..dim).map(col_lam_at).collect();
    let mut last_zero_ridge = init_diag.max(MIN_CHOLESKY_RIDGE);
    let mut last_lambda_diag: Vec<f64> = anchor_lam_diag
        .iter()
        .map(|&a| (a + last_zero_ridge).max(1.0e-16))
        .collect();
    let mut g_work = vec![0.0_f64; gram.len()];

    for &rel in schedule {
        g_work.copy_from_slice(gram);
        let diag_reg = (rel * mean_diag).max(1.0e-15);
        for j in 0..dim {
            g_work[uplo_u_index(j, j)] += anchor_lam_diag[j] + diag_reg;
        }
        let chol_ridge = (0.01 * rel * mean_diag).max(1.0e-16);
        let eff_chol_ridge = chol_ridge.max(MIN_CHOLESKY_RIDGE);
        if let Ok(r_upper) = factorize_uplo_u_cholesky(dim, &g_work, chol_ridge)
            && let Ok(sol) = solve_factorized_uplo_u(dim, &r_upper, &r_anchored)
            && sol.iter().all(|v| v.is_finite())
        {
            last_sol = sol;
            last_zero_ridge = diag_reg + eff_chol_ridge;
            for j in 0..dim {
                last_lambda_diag[j] = anchor_lam_diag[j] + last_zero_ridge;
            }
            let bounds_ok = last_sol.iter().all(|&v| v.abs() <= 7.5);
            let delta_ok = zero_anchor
                || last_sol
                    .iter()
                    .zip(anchor.iter())
                    .take(dim)
                    .all(|(&s, &a)| (s - a).abs() <= 2.5);
            if bounds_ok && delta_ok {
                return Ok(AnchoredSolveResult {
                    sol: last_sol,
                    lambda_diag: last_lambda_diag,
                    anchor_lam_diag,
                    zero_ridge: last_zero_ridge,
                    r_upper: Some(r_upper),
                });
            }
        }
    }

    for v in &mut last_sol {
        *v = if v.is_finite() {
            v.clamp(-7.5, 7.5)
        } else {
            0.0
        };
    }
    Ok(AnchoredSolveResult {
        sol: last_sol,
        lambda_diag: last_lambda_diag,
        anchor_lam_diag,
        zero_ridge: last_zero_ridge,
        r_upper: None,
    })
}

pub(super) fn eval_quad_error_channel(
    w: &[f64],
    gram_sub: &[f64],
    rhs_sub: &[f64],
    y_sq: f64,
) -> f64 {
    let dim = rhs_sub.len().min(w.len());
    let mut quad = 0.0_f64;
    let mut dot_rhs = 0.0_f64;
    for j in 0..dim {
        let wj = w[j];
        dot_rhs += wj * rhs_sub[j];
        quad += wj * wj * gram_sub[uplo_u_index(j, j)];
        for i in 0..j {
            quad += 2.0 * w[i] * wj * gram_sub[uplo_u_index(i, j)];
        }
    }
    (y_sq - 2.0 * dot_rhs + quad).max(0.0)
}

pub(super) fn compute_col_rhs_f64(
    feats_post: &[f32],
    tgts: &[f32],
    m: usize,
    dim: usize,
) -> Vec<f64> {
    (0..dim)
        .map(|c| {
            feats_post[c * m..(c + 1) * m]
                .iter()
                .zip(tgts)
                .map(|(&f, &y)| f64::from(f) * f64::from(y))
                .sum::<f64>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_anchored_uplo_u_and_stereo_ozaki_pcg_paths() {
        let m = 64_usize;
        let dim = 2_usize;
        let mut feats = vec![0.0_f32; dim * m];
        for t in 0..m {
            feats[t] = ((t as f64) * 0.13).sin() as f32;
            feats[m + t] = ((t as f64) * 0.29).cos() as f32;
        }
        let tgts_l: Vec<f32> = (0..m)
            .map(|t| 0.5 * feats[t] - 0.25 * feats[m + t])
            .collect();
        let tgts_r: Vec<f32> = (0..m)
            .map(|t| -0.35 * feats[t] + 0.65 * feats[m + t])
            .collect();
        let rhs_l = compute_col_rhs_f64(&feats, &tgts_l, m, dim);
        let rhs_r = compute_col_rhs_f64(&feats, &tgts_r, m, dim);
        let g00: f64 = feats[..m]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        let g01: f64 = feats[..m]
            .iter()
            .zip(&feats[m..])
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let g11: f64 = feats[m..]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        // Inject artificial off-diagonal perturbation into preconditioner Gram.
        let gram_noisy = vec![g00, g01 + 2.0, g11];

        let zero = vec![0.0; dim];
        let broadband = vec![true, false];
        let mut sol_l = solve_anchored_uplo_u(&gram_noisy, &rhs_l, &zero, &broadband, dim).unwrap();
        let mut sol_r = solve_anchored_uplo_u(&gram_noisy, &rhs_r, &zero, &broadband, dim).unwrap();
        assert!(sol_l.r_upper.is_some() && sol_r.r_upper.is_some());

        let exact_gram = vec![g00, g01, g11];
        let y_sq_l: f64 = tgts_l.iter().map(|&y| f64::from(y) * f64::from(y)).sum();
        let pre_err = eval_quad_error_channel(&sol_l.sol, &exact_gram, &rhs_l, y_sq_l);
        refine_stereo_ozaki_if_single_stem(
            &mut sol_l,
            &mut sol_r,
            &StereoOzakiRefineCtx {
                left: OzakiStemChannel {
                    feats: &feats,
                    tgts: &tgts_l,
                    num_frames: m,
                    weight: 1.0,
                },
                right: OzakiStemChannel {
                    feats: &feats,
                    tgts: &tgts_r,
                    num_frames: m,
                    weight: 1.0,
                },
                w0_l: &zero,
                w0_r: &zero,
                dim,
            },
        );
        let post_err = eval_quad_error_channel(&sol_l.sol, &exact_gram, &rhs_l, y_sq_l);
        assert!(post_err < pre_err && (sol_l.sol[0] - 0.5).abs() < 1.0e-5);
        assert!((sol_r.sol[1] - 0.65).abs() < 1.0e-5);

        // Non-zero anchor solve and clamped fallback (`r_upper == None`) both succeed cleanly.
        let anchor = vec![0.4, -0.2];
        let anchored =
            solve_anchored_uplo_u(&exact_gram, &rhs_l, &anchor, &broadband, dim).unwrap();
        assert!(anchored.r_upper.is_some());
        let mut clamped =
            solve_anchored_uplo_u(&[1.0, 0.0, 1.0], &[1.0e6, 0.0], &zero, &broadband, dim).unwrap();
        assert!(clamped.r_upper.is_none());
        assert_eq!(clamped.sol[0], 7.5);
        refine_ozaki_channel(
            &mut clamped,
            &OzakiStemChannel {
                feats: &feats,
                tgts: &tgts_l,
                num_frames: m,
                weight: 1.0,
            },
            &zero,
            dim,
        );
        assert_eq!(clamped.sol[0], 7.5);

        let model = SonaModelF64::new(48_000.0);
        let (wl, wr, flags) = pack_existing_weights_for_tier(&model, 32, 0);
        assert_eq!(wl.len(), 2 + 16 * 3 + 32);
        assert_eq!(wr.len(), wl.len());
        assert!(flags[0] && flags[1] && !flags[2] && flags[2 + 16 * 3]);
    }
}
