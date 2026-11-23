//! SSOT `STAT` `256 × 256` `uplo='U'` sufficient-statistics extraction and multi-stem
//! weighted Cholesky readout patching over `SonaModelF64` (`ref[train.pipeline.nested-varpro-schur]`).

mod solve;

use crate::error::TrainError;
use crate::linalg::varpro::assemble_ozaki_normal_equations;
use crate::runner::{ActiveComputeBackend, TrainRunner};
use crate::wide_gram::{contract_256d_gram_tiled_wide192, uplo_u_packed_len};

use super::super::model::{SonaModelF64, StereoTrainingPairs};
use super::eval_hp_post_for_ro;
use super::features::{
    StereoFeatureSet, TIER_KAUTZ_CAPS, build_pass3_feature_sets, has_higher_tier_wet_op_t,
    has_upper_tier_wet_op_t, unpack_weights_into_prefix,
};
use super::gram_snap::{build_tier_col_wire_specs, snap_weights_gram_coordinate};
use solve::{
    OzakiStemChannel, StereoOzakiRefineCtx, compute_col_rhs_f64, eval_quad_error_channel,
    pack_existing_weights_for_tier, refine_stereo_ozaki_if_single_stem,
    snap_prefix_readout_weights, solve_anchored_uplo_u,
};

/// Borrowed stereo stem tuple `(in_l, in_r, target_l, target_r, weight)` passed to
/// [`patch_model_readouts_weighted_cholesky`].
pub type WeightedReadoutStem<'a> = (&'a [f64], &'a [f64], &'a [f64], &'a [f64], f64);

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
        4,
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

fn accumulate_weighted_stereo_gram_rhs(
    acc: &mut Option<StereoGramRhs>,
    add: &StereoGramRhs,
    weight: f64,
) {
    let buf = acc.get_or_insert_with(|| StereoGramRhs {
        gram_l: vec![0.0; add.gram_l.len()],
        rhs_l: vec![0.0; add.rhs_l.len()],
        gram_r: vec![0.0; add.gram_r.len()],
        rhs_r: vec![0.0; add.rhs_r.len()],
    });
    for (dst, &val) in buf.gram_l.iter_mut().zip(&add.gram_l) {
        *dst += weight * val;
    }
    for (dst, &val) in buf.rhs_l.iter_mut().zip(&add.rhs_l) {
        *dst += weight * val;
    }
    for (dst, &val) in buf.gram_r.iter_mut().zip(&add.gram_r) {
        *dst += weight * val;
    }
    for (dst, &val) in buf.rhs_r.iter_mut().zip(&add.rhs_r) {
        *dst += weight * val;
    }
}

/// Solves a fast weighted multi-stem `uplo='U'` Cholesky readout patch across all 4 nested tiers
/// while leaving all non-readout core, Kautz, and `Op-T` parameters untouched.
pub fn patch_model_readouts_weighted_cholesky(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    stems: &[WeightedReadoutStem<'_>],
    warmup: usize,
) -> Result<(), TrainError> {
    let force_wide192 = matches!(
        runner.active_backend(),
        ActiveComputeBackend::CpuSimd { .. }
    );
    patch_model_readouts_weighted_cholesky_inner(runner, model, stems, warmup, force_wide192)
}

fn patch_model_readouts_weighted_cholesky_inner(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    stems: &[WeightedReadoutStem<'_>],
    warmup: usize,
    force_wide192: bool,
) -> Result<(), TrainError> {
    if stems.is_empty() {
        return Err(TrainError::InvalidDimensions {
            context: "patch_model_readouts_weighted_cholesky",
            expected: 1,
            actual: 0,
        });
    }

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
    let mut lite_1x_acc: Option<StereoGramRhs> = None;
    let mut tier_opt_acc: [Option<StereoGramRhs>; 3] = [None, None, None];
    let mut single_stem_fset: Option<(StereoFeatureSet, f64, usize)> = None;
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
            4,
        );
        let hp_r = eval_hp_post_for_ro(
            model,
            &fset.raw_r_col0,
            fset.raw_r_2x.as_deref(),
            model.ro_pole_48k,
            w,
            true,
            4,
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

            if ti == 3 || (active_k <= TIER_KAUTZ_CAPS[ti] && !has_higher_tier_wet_op_t(model, ti))
            {
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
                1,
            );
            let h1_r = eval_hp_post_for_ro(
                model,
                &fset.lite_1x_raw_r,
                None,
                model.ro_pole_48k,
                w,
                true,
                1,
            );
            fl[m..(2 * m)].copy_from_slice(&h1_l);
            fr[m..(2 * m)].copy_from_slice(&h1_r);
            let step_1x = assemble_stereo_gram_rhs(
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
            accumulate_weighted_stereo_gram_rhs(&mut lite_1x_acc, &step_1x, weight);
        }

        for ti in 0..3 {
            let dim_t = fset.dims[ti];
            let (raw_l_2x, raw_r_2x) = (fset.raw_l_2x.as_deref(), fset.raw_r_2x.as_deref());
            if let Some((ref mut fl_t, ref mut fr_t)) = fset.tier_opt_feats_post[ti] {
                let ht_l = eval_hp_post_for_ro(
                    model,
                    &fset.raw_l_col0,
                    raw_l_2x,
                    model.ro_pole_48k,
                    w,
                    false,
                    ti + 1,
                );
                let ht_r = eval_hp_post_for_ro(
                    model,
                    &fset.raw_r_col0,
                    raw_r_2x,
                    model.ro_pole_48k,
                    w,
                    true,
                    ti + 1,
                );
                fl_t[m..(2 * m)].copy_from_slice(&ht_l);
                fr_t[m..(2 * m)].copy_from_slice(&ht_r);
                let step_opt = assemble_stereo_gram_rhs(
                    runner,
                    &StereoGramFeatures {
                        feats_l: fl_t,
                        feats_r: fr_t,
                        tgts_l: &fset.tier_tgts_l_prefix[ti],
                        tgts_r: &fset.tier_tgts_r_prefix[ti],
                        num_frames: m,
                        dim: dim_t,
                        force_wide192,
                    },
                )?;
                accumulate_weighted_stereo_gram_rhs(&mut tier_opt_acc[ti], &step_opt, weight);
            }
        }

        used_stems += 1;
        if !force_wide192 {
            single_stem_fset = if used_stems == 1 {
                Some((fset, weight, m))
            } else {
                None
            };
        }
    }

    let Some(dims) = total_dims else {
        return Err(TrainError::InvalidDimensions {
            context: "patch_model_readouts_weighted_cholesky",
            expected: 32,
            actual: used_stems,
        });
    };

    let has_upper_opt = has_upper_tier_wet_op_t(model);
    if !model.oversample_upper_tiers_2x && !has_upper_opt {
        let d_ultra = dims[3];
        let (w0_l, w0_r, is_broadband) = pack_existing_weights_for_tier(model, active_fir_taps, 3);
        let zero_anchor_u = w0_l.iter().chain(&w0_r).all(|&v| v == 0.0);
        if zero_anchor_u {
            let mut sol_l =
                solve_anchored_uplo_u(&gram_l_acc, &rhs_l_acc[3], &w0_l, &is_broadband, d_ultra)?;
            let mut sol_r =
                solve_anchored_uplo_u(&gram_r_acc, &rhs_r_acc[3], &w0_r, &is_broadband, d_ultra)?;
            if let Some((ref fset, sw, sm)) = single_stem_fset {
                refine_stereo_ozaki_if_single_stem(
                    &mut sol_l,
                    &mut sol_r,
                    &StereoOzakiRefineCtx {
                        left: OzakiStemChannel {
                            feats: &fset.feats_l_post,
                            tgts: &fset.tgts_l_post,
                            num_frames: sm,
                            weight: sw,
                        },
                        right: OzakiStemChannel {
                            feats: &fset.feats_r_post,
                            tgts: &fset.tgts_r_post,
                            num_frames: sm,
                            weight: sw,
                        },
                        w0_l: &w0_l,
                        w0_r: &w0_r,
                        dim: d_ultra,
                    },
                );
            }
            let col_specs_nested = build_tier_col_wire_specs(3, active_fir_taps, false);
            let wl_snap = snap_weights_gram_coordinate(
                &sol_l.sol,
                &gram_l_acc,
                &sol_l.lambda_diag,
                &col_specs_nested,
            );
            let wr_snap = snap_weights_gram_coordinate(
                &sol_r.sol,
                &gram_r_acc,
                &sol_r.lambda_diag,
                &col_specs_nested,
            );

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
        let use_1x = ti == 0 && lite_1x_acc.is_some();
        let use_tier_opt = !use_1x && ti < 3 && tier_opt_acc[ti].is_some();
        let (mut sol_l, mut sol_r, gl_use, gr_use) = if use_1x && let Some(ref acc_1x) = lite_1x_acc
        {
            (
                solve_anchored_uplo_u(&acc_1x.gram_l, &acc_1x.rhs_l, &w0_l, &is_broadband, dim_t)?,
                solve_anchored_uplo_u(&acc_1x.gram_r, &acc_1x.rhs_r, &w0_r, &is_broadband, dim_t)?,
                acc_1x.gram_l.as_slice(),
                acc_1x.gram_r.as_slice(),
            )
        } else if use_tier_opt && let Some(ref acc_opt) = tier_opt_acc[ti] {
            (
                solve_anchored_uplo_u(
                    &acc_opt.gram_l,
                    &acc_opt.rhs_l,
                    &w0_l,
                    &is_broadband,
                    dim_t,
                )?,
                solve_anchored_uplo_u(
                    &acc_opt.gram_r,
                    &acc_opt.rhs_r,
                    &w0_r,
                    &is_broadband,
                    dim_t,
                )?,
                acc_opt.gram_l.as_slice(),
                acc_opt.gram_r.as_slice(),
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
        if let Some((ref fset, sw, sm)) = single_stem_fset {
            let (fl, tl, fr, tr) = if use_1x
                && let (Some(f1l), Some(t1l), Some(f1r), Some(t1r)) = (
                    fset.lite_1x_l_post.as_deref(),
                    fset.lite_1x_tgts_l_post.as_deref(),
                    fset.lite_1x_r_post.as_deref(),
                    fset.lite_1x_tgts_r_post.as_deref(),
                ) {
                (f1l, t1l, f1r, t1r)
            } else if use_tier_opt && let Some((ref fl_t, ref fr_t)) = fset.tier_opt_feats_post[ti]
            {
                (
                    fl_t.as_slice(),
                    fset.tier_tgts_l(ti),
                    fr_t.as_slice(),
                    fset.tier_tgts_r(ti),
                )
            } else {
                (
                    fset.feats_l_post.as_slice(),
                    fset.tier_tgts_l(ti),
                    fset.feats_r_post.as_slice(),
                    fset.tier_tgts_r(ti),
                )
            };
            refine_stereo_ozaki_if_single_stem(
                &mut sol_l,
                &mut sol_r,
                &StereoOzakiRefineCtx {
                    left: OzakiStemChannel {
                        feats: fl,
                        tgts: tl,
                        num_frames: sm,
                        weight: sw,
                    },
                    right: OzakiStemChannel {
                        feats: fr,
                        tgts: tr,
                        num_frames: sm,
                        weight: sw,
                    },
                    w0_l: &w0_l,
                    w0_r: &w0_r,
                    dim: dim_t,
                },
            );
        }
        let col_specs = build_tier_col_wire_specs(ti, active_fir_taps, true);
        let wl_snap =
            snap_weights_gram_coordinate(&sol_l.sol, gl_use, &sol_l.lambda_diag, &col_specs);
        let wr_snap =
            snap_weights_gram_coordinate(&sol_r.sol, gr_use, &sol_r.lambda_diag, &col_specs);
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
    if model.oversample_upper_tiers_2x || dims[3] > dims[0] || has_upper_opt {
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

    #[test]
    fn test_patch_ozaki_pcg_true_stereo_multi_quad_and_multistem_paths() {
        let runner = TrainRunner::cpu();
        let n = 512_usize;
        let xl: Vec<f64> = (0..n)
            .map(|t| 0.30 * (core::f64::consts::TAU * 220.0 * (t as f64) / 48_000.0).sin())
            .collect();
        let xr: Vec<f64> = (0..n)
            .map(|t| 0.25 * (core::f64::consts::TAU * 370.0 * (t as f64) / 48_000.0).cos())
            .collect();
        let tl: Vec<f64> = xl.iter().map(|&v| 0.65 * v).collect();
        let tr: Vec<f64> = xr.iter().map(|&v| -0.45 * v).collect();

        // 1. True-stereo + multi-quad wet Op-T + active_k = 16 via CPU Ozaki (`force_wide192 = false`).
        let mut m1 = SonaModelF64::new(48_000.0);
        m1.oversample_upper_tiers_2x = false;
        m1.active_kautz_pairs = 16;
        for k in 0..16 {
            m1.kautz[k].r_low = 0.85;
            m1.kautz[k].theta = 0.1 + 0.08 * (k as f64);
            m1.kautz[k].b_gamma = 0.2;
        }
        for q in 0..2 {
            m1.transport_quads[q].input_gains = [0.25; 4];
            m1.transport_quads[q].wet_tap_l = [0.15, 0.0, 0.0, 0.0];
            m1.transport_quads[q].wet_tap_r = [0.0, 0.15, 0.0, 0.0];
        }
        patch_model_readouts_weighted_cholesky_inner(
            &runner,
            &mut m1,
            &[(&xl, &xr, &tl, &tr, 1.0)],
            32,
            false,
        )
        .unwrap();
        assert_eq!(m1.prefix_override_needed, [false, true, true, true]);
        assert!(m1.tier_overrides.iter().all(Option::is_some));

        // 2. 2x oversampled Ozaki + single-stem PCG path (`use_1x = true`), plus multi-stem + degenerate skips.
        let mut m2 = SonaModelF64::new(48_000.0);
        m2.oversample_upper_tiers_2x = true;
        patch_model_readouts_weighted_cholesky_inner(
            &runner,
            &mut m2,
            &[(&xl, &xr, &tl, &tr, 1.0)],
            32,
            false,
        )
        .unwrap();
        assert!((m2.direct_s2_u_gain[0] - 0.65).abs() < 0.1);

        let mut m3 = SonaModelF64::new(48_000.0);
        m3.oversample_upper_tiers_2x = false;
        patch_model_readouts_weighted_cholesky_inner(
            &runner,
            &mut m3,
            &[
                (&xl, &xr, &tl, &tr, 0.0), // skipped: non-positive weight
                (&xl[..32], &xr[..32], &tl[..32], &tr[..32], 1.0), // skipped: m < 32
                (&xl, &xr, &tl, &tr, 0.5),
                (&xl, &xr, &tl, &tr, 0.5),
            ],
            32,
            false,
        )
        .unwrap();
        assert!((m3.direct_s2_u_gain[0] - 0.65).abs() < 0.05);
        assert!(
            patch_model_readouts_weighted_cholesky_inner(
                &runner,
                &mut m3,
                &[(&xl, &xr, &tl, &tr, 0.0)],
                32,
                false,
            )
            .is_err()
        );

        // 3. Single-stem 1x zero-anchor Ozaki nested shortcut (mono features), then non-zero-anchor fallback.
        let xm: Vec<f64> = (0..n)
            .map(|t| (((t * 1_664_525 + 1_013_904_223) % 997) as f64 / 997.0) - 0.5)
            .collect();
        let tl_n: Vec<f64> = (0..n)
            .map(|t| {
                0.4 * xm[t]
                    + if t >= 36 { 0.25 * xm[t - 36] } else { 0.0 }
                    + if t >= 56 { 0.20 * xm[t - 56] } else { 0.0 }
                    + if t >= 100 { 0.15 * xm[t - 100] } else { 0.0 }
            })
            .collect();
        let mut m4 = SonaModelF64::new(48_000.0);
        m4.oversample_upper_tiers_2x = false;
        m4.direct_s2_u_gain = [0.0, 0.0];
        m4.active_fir_taps = 128;
        patch_model_readouts_weighted_cholesky_inner(
            &runner,
            &mut m4,
            &[(&xm, &xm, &tl_n, &tl_n, 1.0)],
            32,
            false,
        )
        .unwrap();
        assert_eq!(m4.prefix_override_needed, [false; 4]);
        assert!(m4.fir_l[36].abs() > 0.15);
        patch_model_readouts_weighted_cholesky_inner(
            &runner,
            &mut m4,
            &[(&xm, &xm, &tl_n, &tl_n, 1.0)],
            32,
            false,
        )
        .unwrap();
        assert!(m4.tier_overrides[3].is_some());
    }
}
