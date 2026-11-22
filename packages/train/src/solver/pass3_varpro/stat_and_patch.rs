//! SSOT `STAT` `256 × 256` `uplo='U'` sufficient-statistics extraction and multi-stem
//! weighted Cholesky readout patching over `SonaModelF64` (`ref[train.pipeline.nested-varpro-schur]`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};

use crate::error::TrainError;
use crate::linalg::cholesky::solve_uplo_u_cholesky;
use crate::linalg::varpro::assemble_ozaki_normal_equations;
use crate::runner::{ActiveComputeBackend, TrainRunner};
use crate::wide_gram::{contract_256d_gram_tiled_wide192, uplo_u_index, uplo_u_packed_len};

use super::super::model::{SonaModelF64, StereoTrainingPairs, TierReadoutPrefixF64};
use super::eval_hp_post_for_ro;
use super::features::{TIER_KAUTZ_CAPS, build_pass3_feature_sets, unpack_weights_into_prefix};
use super::gram_snap::{build_tier_fp16_col_bounds, snap_weights_gram_coordinate};

/// Borrowed stereo stem tuple `(in_l, in_r, target_l, target_r, weight)` passed to
/// [`patch_model_readouts_weighted_cholesky`].
pub type WeightedReadoutStem<'a> = (&'a [f64], &'a [f64], &'a [f64], &'a [f64], f64);

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

fn snap_prefix_readout_weights(pfx: &mut TierReadoutPrefixF64) {
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

/// Contracts the `256 × 256` `uplo='U'` Gram matrix (`32,896` scalars) and stereo cross vector
/// (`[[f64; 2]; 256]`) over `model`'s post-warmup Pass-3 feature bus
/// (`ref[container.proj.stat-uplo-u]`, `ref[train.gram.wide192]`).
pub fn extract_stat_256d_gram(
    runner: &TrainRunner,
    model: &SonaModelF64,
    pairs: &StereoTrainingPairs<'_>,
    warmup: usize,
    stride: usize,
) -> Result<(Vec<f64>, [[f64; 2]; 256]), TrainError> {
    let StereoTrainingPairs {
        in_l,
        in_r,
        target_l,
        target_r,
    } = *pairs;
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let w = warmup.min(n / 4);
    let m = n.saturating_sub(w);
    if m < 32 {
        return Err(TrainError::InvalidDimensions {
            context: "extract_stat_256d_gram",
            expected: 32,
            actual: m,
        });
    }

    let mut m_tmp = model.clone();
    let mut fset = build_pass3_feature_sets(
        runner,
        &mut m_tmp,
        &in_l[..n],
        &in_r[..n],
        &target_l[..n],
        &target_r[..n],
        w,
    )?;
    let hp_l = eval_hp_post_for_ro(
        &m_tmp,
        &fset.raw_l_col0,
        fset.raw_l_2x.as_deref(),
        m_tmp.ro_pole_48k,
        w,
        false,
    );
    fset.feats_l_post[m..(2 * m)].copy_from_slice(&hp_l);

    let active_dim = fset.dims[3].min(256);
    let mut rows = vec![[0.0_f32; 256]; m];
    let mut targets = Vec::with_capacity(m);
    for t in 0..m {
        for c in 0..active_dim {
            rows[t][c] = fset.feats_l_post[c * m + t];
        }
        targets.push([
            f64::from(fset.tgts_l_post[t]),
            f64::from(fset.tgts_r_post[t]),
        ]);
    }
    let weights = vec![1.0_f64; m];
    let (gram_g, cross_vec) =
        contract_256d_gram_tiled_wide192(&rows, &targets, &weights, stride.max(1));
    let mut cross_256 = [[0.0_f64; 2]; 256];
    for (dst, &src) in cross_256.iter_mut().zip(cross_vec.iter()) {
        *dst = src;
    }
    Ok((gram_g, cross_256))
}

struct StereoGramRhs {
    gram_l: Vec<f64>,
    rhs_l: Vec<f64>,
    gram_r: Vec<f64>,
    rhs_r: Vec<f64>,
}

struct StereoGramFeatures<'a> {
    feats_l: &'a [f32],
    feats_r: &'a [f32],
    tgts_l: &'a [f32],
    tgts_r: &'a [f32],
    num_frames: usize,
    dim: usize,
    force_wide192: bool,
}

fn assemble_stereo_gram_rhs(
    runner: &TrainRunner,
    inputs: &StereoGramFeatures<'_>,
) -> Result<StereoGramRhs, TrainError> {
    let StereoGramFeatures {
        feats_l,
        feats_r,
        tgts_l,
        tgts_r,
        num_frames,
        dim,
        force_wide192,
    } = *inputs;
    if feats_l == feats_r {
        let mut tgts_lr = Vec::with_capacity(2 * num_frames);
        tgts_lr.extend_from_slice(&tgts_l[..num_frames]);
        tgts_lr.extend_from_slice(&tgts_r[..num_frames]);
        let (gram, rhs_v) = assemble_ozaki_normal_equations(
            runner,
            feats_l,
            &tgts_lr,
            num_frames,
            dim,
            2,
            force_wide192,
        )?;
        let rl: Vec<f64> = rhs_v.iter().map(|v| v[0]).collect();
        let rr: Vec<f64> = rhs_v.iter().map(|v| v[1]).collect();
        return Ok(StereoGramRhs {
            gram_l: gram.clone(),
            rhs_l: rl,
            gram_r: gram,
            rhs_r: rr,
        });
    }
    let (gl, rl_v) = assemble_ozaki_normal_equations(
        runner,
        feats_l,
        tgts_l,
        num_frames,
        dim,
        1,
        force_wide192,
    )?;
    let (gr, rr_v) = assemble_ozaki_normal_equations(
        runner,
        feats_r,
        tgts_r,
        num_frames,
        dim,
        1,
        force_wide192,
    )?;
    Ok(StereoGramRhs {
        gram_l: gl,
        rhs_l: rl_v.into_iter().map(|v| v[0]).collect(),
        gram_r: gr,
        rhs_r: rr_v.into_iter().map(|v| v[0]).collect(),
    })
}

fn pack_existing_weights_for_tier(
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

fn solve_anchored_uplo_u(
    gram: &[f64],
    rhs: &[f64],
    anchor: &[f64],
    is_broadband_col: &[bool],
    dim: usize,
) -> Result<(Vec<f64>, f64), TrainError> {
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
    let schedule: &[f64] = if zero_anchor {
        &[1.0e-12, 1.0e-9, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    } else {
        &[1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    };

    let mut r_anchored = rhs[..dim].to_vec();
    if !zero_anchor {
        for (j, &w0) in anchor.iter().enumerate().take(dim) {
            let col_lam = if is_broadband_col.get(j).copied().unwrap_or(false) {
                anchor_lam_broadband
            } else {
                anchor_lam_detail
            };
            r_anchored[j] += col_lam * w0;
        }
    }

    let mut last_sol = vec![0.0_f64; dim];
    let mut last_ridge = (anchor_lam_broadband + schedule[0] * mean_diag).max(1.0e-16);
    let mut g_work = vec![0.0_f64; gram.len()];

    for &rel in schedule {
        g_work.copy_from_slice(gram);
        let diag_reg = (rel * mean_diag).max(1.0e-15);
        for j in 0..dim {
            let col_lam = if zero_anchor {
                0.0
            } else if is_broadband_col.get(j).copied().unwrap_or(false) {
                anchor_lam_broadband
            } else {
                anchor_lam_detail
            };
            g_work[uplo_u_index(j, j)] += col_lam + diag_reg;
        }
        let chol_ridge = (0.01 * rel * mean_diag).max(1.0e-16);
        if let Ok(sol) = solve_uplo_u_cholesky(&g_work, &r_anchored, chol_ridge)
            && sol.iter().all(|v| v.is_finite())
        {
            last_sol = sol;
            last_ridge = anchor_lam_broadband + diag_reg + chol_ridge;
            let bounds_ok = last_sol.iter().all(|&v| v.abs() <= 7.5);
            let delta_ok = zero_anchor
                || last_sol
                    .iter()
                    .zip(anchor.iter())
                    .take(dim)
                    .all(|(&s, &a)| (s - a).abs() <= 2.5);
            if bounds_ok && delta_ok {
                return Ok((last_sol, last_ridge));
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
    Ok((last_sol, last_ridge))
}

fn eval_quad_error_channel(w: &[f64], gram_sub: &[f64], rhs_sub: &[f64], y_sq: f64) -> f64 {
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

fn compute_col_rhs_f64(feats_post: &[f32], tgts: &[f32], m: usize, dim: usize) -> Vec<f64> {
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

/// Solves a fast weighted multi-stem `uplo='U'` Cholesky readout patch across all 4 nested tiers
/// while leaving all non-readout core, Kautz, and `Op-T` parameters untouched.
pub fn patch_model_readouts_weighted_cholesky(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    stems: &[WeightedReadoutStem<'_>],
    warmup: usize,
) -> Result<(), TrainError> {
    if stems.is_empty() {
        return Err(TrainError::InvalidDimensions {
            context: "patch_model_readouts_weighted_cholesky",
            expected: 1,
            actual: 0,
        });
    }

    let force_wide192 = matches!(
        runner.active_backend(),
        ActiveComputeBackend::CpuSimd { .. }
    );
    let active_k = model.active_kautz_pairs.min(model.kautz.len());
    let mut total_dims: Option<[usize; 4]> = None;
    let mut active_fir_taps = 0_usize;
    let mut pc_gain = model.pre_conv_u_gain;
    let mut gram_l_acc = Vec::new();
    let mut rhs_l_acc = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut gram_r_acc = Vec::new();
    let mut rhs_r_acc = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut y_sq_l_acc = [0.0_f64; 4];
    let mut y_sq_r_acc = [0.0_f64; 4];
    let mut gram_l_1x_acc: Option<Vec<f64>> = None;
    let mut rhs_l_1x_acc: Option<Vec<f64>> = None;
    let mut gram_r_1x_acc: Option<Vec<f64>> = None;
    let mut rhs_r_1x_acc: Option<Vec<f64>> = None;
    let mut used_stems = 0_usize;

    for &(in_l, in_r, tgt_l, tgt_r, weight) in stems {
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        let n = in_l.len().min(in_r.len()).min(tgt_l.len()).min(tgt_r.len());
        let w = warmup.min(n / 4);
        let m = n.saturating_sub(w);
        if m < 32 {
            continue;
        }
        let mut fset = build_pass3_feature_sets(
            runner,
            model,
            &in_l[..n],
            &in_r[..n],
            &tgt_l[..n],
            &tgt_r[..n],
            w,
        )?;
        let hp_l = eval_hp_post_for_ro(
            model,
            &fset.raw_l_col0,
            fset.raw_l_2x.as_deref(),
            model.ro_pole_48k,
            w,
            false,
        );
        let hp_r = eval_hp_post_for_ro(
            model,
            &fset.raw_r_col0,
            fset.raw_r_2x.as_deref(),
            model.ro_pole_48k,
            w,
            true,
        );
        fset.feats_l_post[m..(2 * m)].copy_from_slice(&hp_l);
        fset.feats_r_post[m..(2 * m)].copy_from_slice(&hp_r);

        let dim = fset.dims[3];
        let StereoGramRhs {
            gram_l: gl,
            rhs_l: rl_v,
            gram_r: gr,
            rhs_r: rr_v,
        } = assemble_stereo_gram_rhs(
            runner,
            &StereoGramFeatures {
                feats_l: &fset.feats_l_post,
                feats_r: &fset.feats_r_post,
                tgts_l: &fset.tgts_l_post,
                tgts_r: &fset.tgts_r_post,
                num_frames: m,
                dim,
                force_wide192,
            },
        )?;

        if gram_l_acc.is_empty() {
            total_dims = Some(fset.dims);
            active_fir_taps = fset.active_fir_taps;
            pc_gain = fset.pc_gain;
            gram_l_acc = vec![0.0; gl.len()];
            gram_r_acc = vec![0.0; gr.len()];
            for ti in 0..4 {
                rhs_l_acc[ti] = vec![0.0; fset.dims[ti]];
                rhs_r_acc[ti] = vec![0.0; fset.dims[ti]];
            }
        }
        for (dst, &val) in gram_l_acc.iter_mut().zip(&gl) {
            *dst += weight * val;
        }
        for (dst, &val) in gram_r_acc.iter_mut().zip(&gr) {
            *dst += weight * val;
        }
        for (ti, &dim_t) in fset.dims.iter().enumerate() {
            let tl = fset.tier_tgts_l(ti);
            let tr = fset.tier_tgts_r(ti);
            let e_yl: f64 = tl.iter().map(|&yl| f64::from(yl) * f64::from(yl)).sum();
            let e_yr: f64 = tr.iter().map(|&yr| f64::from(yr) * f64::from(yr)).sum();
            y_sq_l_acc[ti] += weight * e_yl;
            y_sq_r_acc[ti] += weight * e_yr;

            if ti == 3 || active_k <= TIER_KAUTZ_CAPS[ti] {
                for (dst, &val) in rhs_l_acc[ti].iter_mut().zip(&rl_v[..dim_t]) {
                    *dst += weight * val;
                }
                for (dst, &val) in rhs_r_acc[ti].iter_mut().zip(&rr_v[..dim_t]) {
                    *dst += weight * val;
                }
            } else {
                let rl_ti = compute_col_rhs_f64(&fset.feats_l_post, tl, m, dim_t);
                let rr_ti = compute_col_rhs_f64(&fset.feats_r_post, tr, m, dim_t);
                for (dst, &val) in rhs_l_acc[ti].iter_mut().zip(&rl_ti) {
                    *dst += weight * val;
                }
                for (dst, &val) in rhs_r_acc[ti].iter_mut().zip(&rr_ti) {
                    *dst += weight * val;
                }
            }
        }

        if let (Some(fl), Some(fr), Some(tl), Some(tr)) = (
            fset.lite_1x_l_post.as_mut(),
            fset.lite_1x_r_post.as_mut(),
            fset.lite_1x_tgts_l_post.as_ref(),
            fset.lite_1x_tgts_r_post.as_ref(),
        ) {
            let d0 = fset.dims[0];
            let h1_l = eval_hp_post_for_ro(
                model,
                &fset.lite_1x_raw_l,
                None,
                model.ro_pole_48k,
                w,
                false,
            );
            let h1_r =
                eval_hp_post_for_ro(model, &fset.lite_1x_raw_r, None, model.ro_pole_48k, w, true);
            fl[m..(2 * m)].copy_from_slice(&h1_l);
            fr[m..(2 * m)].copy_from_slice(&h1_r);
            let StereoGramRhs {
                gram_l: g1l,
                rhs_l: r1l,
                gram_r: g1r,
                rhs_r: r1r,
            } = assemble_stereo_gram_rhs(
                runner,
                &StereoGramFeatures {
                    feats_l: fl,
                    feats_r: fr,
                    tgts_l: tl,
                    tgts_r: tr,
                    num_frames: m,
                    dim: d0,
                    force_wide192,
                },
            )?;
            let gl1_buf = gram_l_1x_acc.get_or_insert_with(|| vec![0.0; g1l.len()]);
            let rl1_buf = rhs_l_1x_acc.get_or_insert_with(|| vec![0.0; d0]);
            let gr1_buf = gram_r_1x_acc.get_or_insert_with(|| vec![0.0; g1r.len()]);
            let rr1_buf = rhs_r_1x_acc.get_or_insert_with(|| vec![0.0; d0]);
            for (dst, &val) in gl1_buf.iter_mut().zip(&g1l) {
                *dst += weight * val;
            }
            for (dst, &val) in rl1_buf.iter_mut().zip(&r1l) {
                *dst += weight * val;
            }
            for (dst, &val) in gr1_buf.iter_mut().zip(&g1r) {
                *dst += weight * val;
            }
            for (dst, &val) in rr1_buf.iter_mut().zip(&r1r) {
                *dst += weight * val;
            }
        }

        used_stems += 1;
    }

    let Some(dims) = total_dims else {
        return Err(TrainError::InvalidDimensions {
            context: "patch_model_readouts_weighted_cholesky",
            expected: 32,
            actual: used_stems,
        });
    };

    if !model.oversample_upper_tiers_2x {
        let d_ultra = dims[3];
        let (w0_l, w0_r, is_broadband) = pack_existing_weights_for_tier(model, active_fir_taps, 3);
        let zero_anchor_u = w0_l.iter().chain(&w0_r).all(|&v| v == 0.0);
        if zero_anchor_u {
            let (wl_u, ridge_l) =
                solve_anchored_uplo_u(&gram_l_acc, &rhs_l_acc[3], &w0_l, &is_broadband, d_ultra)?;
            let (wr_u, ridge_r) =
                solve_anchored_uplo_u(&gram_r_acc, &rhs_r_acc[3], &w0_r, &is_broadband, d_ultra)?;
            let fp16_bounds_nested = build_tier_fp16_col_bounds(3, active_fir_taps, false);
            let wl_snap =
                snap_weights_gram_coordinate(&wl_u, &gram_l_acc, ridge_l, &fp16_bounds_nested);
            let wr_snap =
                snap_weights_gram_coordinate(&wr_u, &gram_r_acc, ridge_r, &fp16_bounds_nested);

            let err_nested: [f64; 4] = core::array::from_fn(|ti| {
                let dt = dims[ti];
                let p_sub = uplo_u_packed_len(dt);
                eval_quad_error_channel(
                    &wl_snap[..dt],
                    &gram_l_acc[..p_sub],
                    &rhs_l_acc[ti],
                    y_sq_l_acc[ti],
                ) + eval_quad_error_channel(
                    &wr_snap[..dt],
                    &gram_r_acc[..p_sub],
                    &rhs_r_acc[ti],
                    y_sq_r_acc[ti],
                )
            });
            if err_nested[3] <= err_nested[2] + 1.0e-10
                && err_nested[2] <= err_nested[1] + 1.0e-10
                && err_nested[1] <= err_nested[0] + 1.0e-10
            {
                let mut pfx = unpack_weights_into_prefix(
                    &wl_snap,
                    &wr_snap,
                    active_fir_taps,
                    pc_gain,
                    3,
                    model,
                );
                snap_prefix_readout_weights(&mut pfx);
                model.direct_s2_u_gain = pfx.direct_s2_u_gain;
                model.direct_s2_hp_gain = pfx.direct_s2_hp_gain;
                model.pre_conv_u_gain = pfx.pre_conv_u_gain;
                model.pre_conv_hp_gain = pfx.pre_conv_hp_gain;
                for l in 0..40 {
                    model.w_ro[l][6..12].copy_from_slice(&pfx.w_ro[l][6..12]);
                }
                model.fir_l = pfx.fir_l;
                model.fir_r = pfx.fir_r;
                model.prefix_override_needed = [false; 4];
                model.tier_overrides = [None, None, None, None];
                return Ok(());
            }
        }
    }

    for (ti, &dim_t) in dims.iter().enumerate() {
        let packed_len = uplo_u_packed_len(dim_t);
        let (w0_l, w0_r, is_broadband) = pack_existing_weights_for_tier(model, active_fir_taps, ti);
        let ((wl, ridge_l), (wr, ridge_r), gl_use, gr_use) = if ti == 0
            && let (Some(g1l), Some(r1l), Some(g1r), Some(r1r)) = (
                gram_l_1x_acc.as_deref(),
                rhs_l_1x_acc.as_deref(),
                gram_r_1x_acc.as_deref(),
                rhs_r_1x_acc.as_deref(),
            ) {
            (
                solve_anchored_uplo_u(g1l, r1l, &w0_l, &is_broadband, dim_t)?,
                solve_anchored_uplo_u(g1r, r1r, &w0_r, &is_broadband, dim_t)?,
                g1l,
                g1r,
            )
        } else {
            (
                solve_anchored_uplo_u(
                    &gram_l_acc[..packed_len],
                    &rhs_l_acc[ti],
                    &w0_l,
                    &is_broadband,
                    dim_t,
                )?,
                solve_anchored_uplo_u(
                    &gram_r_acc[..packed_len],
                    &rhs_r_acc[ti],
                    &w0_r,
                    &is_broadband,
                    dim_t,
                )?,
                &gram_l_acc[..packed_len],
                &gram_r_acc[..packed_len],
            )
        };
        let fp16_bounds = build_tier_fp16_col_bounds(ti, active_fir_taps, true);
        let wl_snap = snap_weights_gram_coordinate(&wl, gl_use, ridge_l, &fp16_bounds);
        let wr_snap = snap_weights_gram_coordinate(&wr, gr_use, ridge_r, &fp16_bounds);
        let mut pfx =
            unpack_weights_into_prefix(&wl_snap, &wr_snap, active_fir_taps, pc_gain, ti, model);
        snap_prefix_readout_weights(&mut pfx);
        if ti == 3 {
            model.direct_s2_u_gain = pfx.direct_s2_u_gain;
            model.direct_s2_hp_gain = pfx.direct_s2_hp_gain;
            model.pre_conv_u_gain = pfx.pre_conv_u_gain;
            model.pre_conv_hp_gain = pfx.pre_conv_hp_gain;
            for l in 0..40 {
                model.w_ro[l][6..12].copy_from_slice(&pfx.w_ro[l][6..12]);
            }
            model.fir_l = pfx.fir_l;
            model.fir_r = pfx.fir_r;
        }
        model.tier_overrides[ti] = Some(Box::new(pfx));
    }
    if model.oversample_upper_tiers_2x || dims[3] > dims[0] {
        model.prefix_override_needed = [false, true, true, true];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};
    use dnam_vm::math::DefaultMathBackend;
    use dnam_vm::{SonaVm, Tier};

    #[test]
    fn test_extract_stat_256d_gram_and_weighted_cholesky_patch() {
        let runner = TrainRunner::cpu();
        let n = 512_usize;
        let x: Vec<f64> = (0..n)
            .map(|t| 0.25 * (core::f64::consts::TAU * 220.0 * (t as f64) / 48_000.0).sin())
            .collect();
        let tgt_l: Vec<f64> = x.iter().map(|&v| 0.75 * v).collect();
        let tgt_r: Vec<f64> = x.iter().map(|&v| -0.50 * v).collect();

        let mut model = SonaModelF64::new(48_000.0);
        model.oversample_upper_tiers_2x = false;
        assert!(
            extract_stat_256d_gram(
                &runner,
                &model,
                &StereoTrainingPairs {
                    in_l: &x[..16],
                    in_r: &x[..16],
                    target_l: &tgt_l[..16],
                    target_r: &tgt_r[..16],
                },
                0,
                1,
            )
            .is_err()
        );
        assert!(patch_model_readouts_weighted_cholesky(&runner, &mut model, &[], 32).is_err());

        let (gram, cross) = extract_stat_256d_gram(
            &runner,
            &model,
            &StereoTrainingPairs {
                in_l: &x,
                in_r: &x,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            32,
            2,
        )
        .unwrap();
        assert_eq!(gram.len(), uplo_u_packed_len(256));
        assert!(gram[0] > 0.0);
        assert!(cross[0][0] > 0.0 && cross[0][1] < 0.0);

        let orig_core = model.core_quads[0];
        patch_model_readouts_weighted_cholesky(
            &runner,
            &mut model,
            &[(&x, &x, &tgt_l, &tgt_r, 1.0)],
            32,
        )
        .unwrap();
        assert_eq!(model.core_quads[0], orig_core);
        assert!((model.direct_s2_u_gain[0] - 0.75).abs() < 0.05);
        assert!((model.direct_s2_u_gain[1] - (-0.50)).abs() < 0.05);

        let bin = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).unwrap();
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin, Tier::Ultra, 48_000.0).unwrap();
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let (mut ol, mut or) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        vm.process_block(&x32, &x32, &mut ol, &mut or).unwrap();
        let err: f64 = (64..n)
            .map(|t| (tgt_l[t] - f64::from(ol[t])).powi(2) + (tgt_r[t] - f64::from(or[t])).powi(2))
            .sum::<f64>();
        assert!(err < 1.0e-3, "patched err={err}");

        let mut model_2x = SonaModelF64::new(48_000.0);
        model_2x.oversample_upper_tiers_2x = true;
        patch_model_readouts_weighted_cholesky(
            &runner,
            &mut model_2x,
            &[(&x, &x, &tgt_l, &tgt_r, 1.0)],
            32,
        )
        .unwrap();
        assert!((model_2x.direct_s2_u_gain[0] - 0.75).abs() < 0.05);
        assert_eq!(model_2x.prefix_override_needed, [false, true, true, true]);
        let bin_2x = encode_model_to_dnam_bin(&model_2x, Pass5EmitConfig::default()).unwrap();
        for tier in [Tier::Lite, Tier::Ultra] {
            let mut vm2 =
                SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bin_2x, tier, 48_000.0).unwrap();
            let (mut ol2, mut or2) = (vec![0.0_f32; n], vec![0.0_f32; n]);
            vm2.process_block(&x32, &x32, &mut ol2, &mut or2).unwrap();
            let err2: f64 = (64..n)
                .map(|t| {
                    (tgt_l[t] - f64::from(ol2[t])).powi(2) + (tgt_r[t] - f64::from(or2[t])).powi(2)
                })
                .sum::<f64>();
            let sig2: f64 = (64..n)
                .map(|t| tgt_l[t] * tgt_l[t] + tgt_r[t] * tgt_r[t])
                .sum::<f64>();
            assert!(
                err2 / sig2.max(1.0e-12) < 1.0e-3,
                "patched 2x {tier:?} nesr={}",
                err2 / sig2.max(1.0e-12)
            );
        }
    }
}
