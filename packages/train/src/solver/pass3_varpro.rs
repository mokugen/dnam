//! Pass-3 joint 4-tier causally nested `VarPro` + Schur-complement readout re-balancing,
//! `1×` Lite / `2×` `Full..Ultra` adaptation, stereo `ro_pole_48k` 1-D sweep, EMR `MM-Cholesky`
//! & 33-band stereo `LTAS` Newton-KKT solver (`ref[train.pipeline.nested-varpro-schur]`).

pub(crate) mod features;
mod gram_snap;
mod pre_conv_proj;
mod spectral_refine;
mod stat_and_patch;

pub use stat_and_patch::{
    WeightedReadoutStem, extract_stat_256d_gram, patch_model_readouts_weighted_cholesky,
};

use crate::control::clamp::{MAX_GAIN_6DB_LIN, evaluate_readout_spectral_magnitudes};
use crate::control::stability::MAX_RO_POLE_48K_F64;
use crate::error::TrainError;
use crate::linalg::cholesky::solve_uplo_u_cholesky;
use crate::linalg::varpro::assemble_ozaki_normal_equations;
use crate::runner::{ActiveComputeBackend, TrainRunner};
use crate::wide_gram::{hash_gram_input_v3_f32, uplo_u_index};

use super::model::SonaModelF64;
use super::model::cascade::downsample_lane_2x_f64;
use features::{
    StereoFeatureSet, TIER_KAUTZ_CAPS, apply_op_t_series_channel, build_pass3_feature_sets,
    eval_highpass_pole, unpack_weights_into_prefix,
};
use spectral_refine::{
    ChannelSpectralCtx, compute_channel_hf_mse, compute_channel_mse,
    refine_channel_tier_emr_and_ltas_kkt, solve_bounded_uplo_u, trace_scaled_ridge,
};

/// Result returned by Pass-3 nested 4-tier `VarPro` readout identification.
#[derive(Clone, Debug, PartialEq)]
pub struct Pass3VarProReport {
    /// Selected `48 kHz` `LinearReadout` leak pole `ro_pole_48k` in `[0.0, MAX_RO_POLE_48K_F64]`.
    pub ro_pole_48k: f64,
    /// Per-tier normalized stereo readout MSE `[Lite, Full, Pro, Ultra]` over fitted feature subspace.
    pub tier_mse: [f64; 4],
    /// Flags indicating which upper tiers (`1..=3`) triggered prefix overrides.
    pub prefix_overrides: [bool; 4],
    /// Canonical `STAT` digest computed over row-major `total_dim` post-warmup feature matrix.
    pub stat_gram_digest: [u8; 32],
}

fn eval_hp_post_for_ro(
    model: &SonaModelF64,
    raw_1x: &[f64],
    raw_2x: Option<&[f64]>,
    ro_48k: f64,
    w: usize,
    is_right: bool,
) -> Vec<f32> {
    let n = raw_1x.len();
    let m = n - w;
    if ro_48k <= 0.0 {
        return vec![0.0_f32; m];
    }
    let rho_base = 48_000.0 / model.sample_rate_hz.max(8_000.0);
    let hp_sig = if let Some(r2x) = raw_2x {
        let ro_2x = ro_48k.powf(0.5 * rho_base).clamp(0.0, 0.999_995);
        let hp_2x = eval_highpass_pole(r2x, ro_2x);
        downsample_lane_2x_f64(&hp_2x, model.os_dec_c_prev1, model.os_dec_c0)
    } else {
        let ro_1x = ro_48k.powf(rho_base).clamp(0.0, 0.999_995);
        eval_highpass_pole(raw_1x, ro_1x)
    };
    let eff = apply_op_t_series_channel(model, &hp_sig, is_right);
    eff[w..n].iter().map(|&v| v as f32).collect()
}

fn patch_col1_gram_and_rhs(
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

struct SweepStereoResult {
    best_ro: f64,
    best_gram_l: Vec<f64>,
    best_rhs_l: Vec<f64>,
    best_hp_l: Vec<f32>,
    best_gram_r: Vec<f64>,
    best_rhs_r: Vec<f64>,
    best_hp_r: Vec<f32>,
}

fn sweep_ro_pole_48k_joint_stereo(
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
        );
        let hp_r = eval_hp_post_for_ro(
            model,
            &fset.raw_r_col0,
            fset.raw_r_2x.as_deref(),
            ro,
            w,
            true,
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
/// lower tier's Kautz pair cap; otherwise reuses the Ultra RHS prefix.
fn build_per_tier_rhs(
    feats_post: &[f32],
    fset: &StereoFeatureSet,
    ultra_rhs: &[f64],
    active_k: usize,
    is_right: bool,
) -> [Vec<f64>; 4] {
    let m = fset.tgts_l_post.len();
    core::array::from_fn(|ti| {
        let dim_t = fset.dims[ti];
        if ti == 3 || active_k <= TIER_KAUTZ_CAPS[ti] {
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

#[derive(Clone, Copy)]
pub(super) struct StereoVarProPairs<'a> {
    pub in_l: &'a [f64],
    pub in_r: &'a [f64],
    pub target_l: &'a [f64],
    pub target_r: &'a [f64],
}

#[derive(Clone, Copy)]
pub(super) struct Pass3InnerOutcome {
    report_lite_1x: bool,
    lite_mse: f64,
    full_mse: f64,
    ultra_mse: f64,
    lite_hf_mse: f64,
    full_hf_mse: f64,
}

fn run_pass3_varpro_readout_inner(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    w: usize,
) -> Result<(Pass3VarProReport, Pass3InnerOutcome), TrainError> {
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let m = n - w;

    model.ro_pole_48k = 0.0;
    let mut fset = build_pass3_feature_sets(
        runner,
        model,
        &in_l[..n],
        &in_r[..n],
        &target_l[..n],
        &target_r[..n],
        w,
    )?;
    let total_dim = fset.dims[3];
    let force_wide192 = matches!(
        runner.active_backend(),
        ActiveComputeBackend::CpuSimd { .. }
    );

    let (base_gl, base_rl_v) = assemble_ozaki_normal_equations(
        runner,
        &fset.feats_l_post,
        &fset.tgts_l_post,
        m,
        total_dim,
        1,
        force_wide192,
    )?;
    let (base_gr, base_rr_v) = assemble_ozaki_normal_equations(
        runner,
        &fset.feats_r_post,
        &fset.tgts_r_post,
        m,
        total_dim,
        1,
        force_wide192,
    )?;
    let base_rl: Vec<f64> = base_rl_v.into_iter().map(|v| v[0]).collect();
    let base_rr: Vec<f64> = base_rr_v.into_iter().map(|v| v[0]).collect();

    let sweep =
        sweep_ro_pole_48k_joint_stereo(model, &fset, &base_gl, &base_rl, &base_gr, &base_rr, w);
    model.ro_pole_48k = sweep.best_ro.clamp(0.0, MAX_RO_POLE_48K_F64);
    fset.feats_l_post[m..(2 * m)].copy_from_slice(&sweep.best_hp_l);
    fset.feats_r_post[m..(2 * m)].copy_from_slice(&sweep.best_hp_r);

    let active_k = model.active_kautz_pairs.min(model.kautz.len());
    let rhs_l_tiers = build_per_tier_rhs(
        &fset.feats_l_post,
        &fset,
        &sweep.best_rhs_l,
        active_k,
        false,
    );
    let rhs_r_tiers =
        build_per_tier_rhs(&fset.feats_r_post, &fset, &sweep.best_rhs_r, active_k, true);

    // Solve `1x` Lite independently whenever `oversample_upper_tiers_2x` is enabled.
    let lite_1x_sol = if let (Some(fl), Some(fr), Some(tl), Some(tr)) = (
        fset.lite_1x_l_post.as_mut(),
        fset.lite_1x_r_post.as_mut(),
        fset.lite_1x_tgts_l_post.as_ref(),
        fset.lite_1x_tgts_r_post.as_ref(),
    ) {
        let d_lite = fset.dims[0];
        let hp_l = eval_hp_post_for_ro(
            model,
            &fset.lite_1x_raw_l,
            None,
            model.ro_pole_48k,
            w,
            false,
        );
        let hp_r =
            eval_hp_post_for_ro(model, &fset.lite_1x_raw_r, None, model.ro_pole_48k, w, true);
        fl[m..(2 * m)].copy_from_slice(&hp_l);
        fr[m..(2 * m)].copy_from_slice(&hp_r);
        let (gl_1x, rl_1x_v) =
            assemble_ozaki_normal_equations(runner, fl, tl, m, d_lite, 1, force_wide192)?;
        let (gr_1x, rr_1x_v) =
            assemble_ozaki_normal_equations(runner, fr, tr, m, d_lite, 1, force_wide192)?;
        let rl_1x: Vec<f64> = rl_1x_v.into_iter().map(|v| v[0]).collect();
        let rr_1x: Vec<f64> = rr_1x_v.into_iter().map(|v| v[0]).collect();
        let (wl_1x, rl_ridge) = solve_bounded_uplo_u(
            &gl_1x,
            &rl_1x,
            d_lite,
            model.ro_pole_48k,
            model.sample_rate_hz,
        )?;
        let (wr_1x, rr_ridge) = solve_bounded_uplo_u(
            &gr_1x,
            &rr_1x,
            d_lite,
            model.ro_pole_48k,
            model.sample_rate_hz,
        )?;
        Some((gl_1x, rl_1x, wl_1x, rl_ridge, gr_1x, rr_1x, wr_1x, rr_ridge))
    } else {
        None
    };

    let mut wl_tiers = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut wr_tiers = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut ridge_l_tiers = [1.0e-5_f64; 4];
    let mut ridge_r_tiers = [1.0e-5_f64; 4];
    for (ti, &dim_t) in fset.dims.iter().enumerate() {
        let (wl, rdl) = solve_bounded_uplo_u(
            &sweep.best_gram_l,
            &rhs_l_tiers[ti],
            dim_t,
            model.ro_pole_48k,
            model.sample_rate_hz,
        )?;
        let (wr, rdr) = solve_bounded_uplo_u(
            &sweep.best_gram_r,
            &rhs_r_tiers[ti],
            dim_t,
            model.ro_pole_48k,
            model.sample_rate_hz,
        )?;
        wl_tiers[ti] = wl;
        wr_tiers[ti] = wr;
        ridge_l_tiers[ti] = rdl;
        ridge_r_tiers[ti] = rdr;
    }

    for ti in 0..4 {
        if ti == 0 {
            model.prefix_override_needed[0] = false;
            continue;
        }
        let dim_t = fset.dims[ti];
        let diff_l = (0..dim_t)
            .map(|c| (wl_tiers[ti][c] - wl_tiers[3][c]).abs())
            .fold(0.0_f64, f64::max);
        let diff_r = (0..dim_t)
            .map(|c| (wr_tiers[ti][c] - wr_tiers[3][c]).abs())
            .fold(0.0_f64, f64::max);
        let norm_t = wl_tiers[ti]
            .iter()
            .chain(&wr_tiers[ti])
            .map(|&v| v.abs())
            .fold(1.0_f64, f64::max);
        model.prefix_override_needed[ti] = model.oversample_upper_tiers_2x
            || active_k > TIER_KAUTZ_CAPS[ti - 1]
            || diff_l.max(diff_r) > 1.0e-3 * norm_t;
    }
    let emit_prefix_overrides = model.prefix_override_needed[1..].iter().any(|&v| v);

    let mut tier_mse = [0.0_f64; 4];
    let mut tier_hf_mse = [0.0_f64; 2];
    for (ti, &dim_t) in fset.dims.iter().enumerate() {
        let (wl_refined, wr_refined) = if ti == 0
            && let Some((
                ref gl_1x,
                ref rl_1x,
                ref wl_1x,
                rl_ridge,
                ref gr_1x,
                ref rr_1x,
                ref wr_1x,
                rr_ridge,
            )) = lite_1x_sol
            && let (Some(fl), Some(fr), Some(tl), Some(tr)) = (
                fset.lite_1x_l_post.as_deref(),
                fset.lite_1x_r_post.as_deref(),
                fset.lite_1x_tgts_l_post.as_deref(),
                fset.lite_1x_tgts_r_post.as_deref(),
            ) {
            let wl_r = refine_channel_tier_emr_and_ltas_kkt(
                wl_1x,
                dim_t,
                0,
                &ChannelSpectralCtx {
                    gram: gl_1x,
                    rhs: rl_1x,
                    feats_post: fl,
                    tgts_post: tl,
                    sr_hz: model.sample_rate_hz,
                    ro_pole_48k: model.ro_pole_48k,
                    active_fir_taps: fset.active_fir_taps,
                    emit_prefix_overrides,
                    used_ridge: rl_ridge,
                },
            );
            let wr_r = refine_channel_tier_emr_and_ltas_kkt(
                wr_1x,
                dim_t,
                0,
                &ChannelSpectralCtx {
                    gram: gr_1x,
                    rhs: rr_1x,
                    feats_post: fr,
                    tgts_post: tr,
                    sr_hz: model.sample_rate_hz,
                    ro_pole_48k: model.ro_pole_48k,
                    active_fir_taps: fset.active_fir_taps,
                    emit_prefix_overrides,
                    used_ridge: rr_ridge,
                },
            );
            tier_mse[0] = 0.5
                * (compute_channel_mse(&wl_r, fl, tl, dim_t)
                    + compute_channel_mse(&wr_r, fr, tr, dim_t));
            tier_hf_mse[0] = 0.5
                * (compute_channel_hf_mse(&wl_r, fl, tl, dim_t)
                    + compute_channel_hf_mse(&wr_r, fr, tr, dim_t));
            (wl_r, wr_r)
        } else {
            let tl = fset.tier_tgts_l(ti);
            let tr = fset.tier_tgts_r(ti);
            let wl_r = refine_channel_tier_emr_and_ltas_kkt(
                &wl_tiers[ti],
                dim_t,
                ti,
                &ChannelSpectralCtx {
                    gram: &sweep.best_gram_l,
                    rhs: &rhs_l_tiers[ti],
                    feats_post: &fset.feats_l_post,
                    tgts_post: tl,
                    sr_hz: model.sample_rate_hz,
                    ro_pole_48k: model.ro_pole_48k,
                    active_fir_taps: fset.active_fir_taps,
                    emit_prefix_overrides,
                    used_ridge: ridge_l_tiers[ti],
                },
            );
            let wr_r = refine_channel_tier_emr_and_ltas_kkt(
                &wr_tiers[ti],
                dim_t,
                ti,
                &ChannelSpectralCtx {
                    gram: &sweep.best_gram_r,
                    rhs: &rhs_r_tiers[ti],
                    feats_post: &fset.feats_r_post,
                    tgts_post: tr,
                    sr_hz: model.sample_rate_hz,
                    ro_pole_48k: model.ro_pole_48k,
                    active_fir_taps: fset.active_fir_taps,
                    emit_prefix_overrides,
                    used_ridge: ridge_r_tiers[ti],
                },
            );
            tier_mse[ti] = 0.5
                * (compute_channel_mse(&wl_r, &fset.feats_l_post, tl, dim_t)
                    + compute_channel_mse(&wr_r, &fset.feats_r_post, tr, dim_t));
            if ti <= 1 {
                tier_hf_mse[ti] = 0.5
                    * (compute_channel_hf_mse(&wl_r, &fset.feats_l_post, tl, dim_t)
                        + compute_channel_hf_mse(&wr_r, &fset.feats_r_post, tr, dim_t));
            }
            (wl_r, wr_r)
        };

        let pfx = unpack_weights_into_prefix(
            &wl_refined,
            &wr_refined,
            fset.active_fir_taps,
            fset.pc_gain,
            ti,
            model,
        );
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

    let mut feats_row_major = vec![0.0_f32; m * total_dim];
    for t in 0..m {
        for c in 0..total_dim {
            feats_row_major[t * total_dim + c] = fset.feats_l_post[c * m + t];
        }
    }
    let ones = vec![1.0_f64; m];
    let stat_gram_digest = hash_gram_input_v3_f32(total_dim, &feats_row_major, &ones, 1);

    let outcome = Pass3InnerOutcome {
        report_lite_1x: lite_1x_sol.is_some(),
        lite_mse: tier_mse[0],
        full_mse: tier_mse[1],
        ultra_mse: tier_mse[3],
        lite_hf_mse: tier_hf_mse[0],
        full_hf_mse: tier_hf_mse[1],
    };
    Ok((
        Pass3VarProReport {
            ro_pole_48k: model.ro_pole_48k,
            tier_mse,
            prefix_overrides: model.prefix_override_needed,
            stat_gram_digest,
        },
        outcome,
    ))
}

pub(super) fn run_pass3_varpro_readout_with_os2x_fallback(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    w: usize,
) -> Result<(Pass3VarProReport, Pass3InnerOutcome), TrainError> {
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let m = n - w;
    let (report_2x, outcome_2x) =
        run_pass3_varpro_readout_inner(runner, model, in_l, in_r, target_l, target_r, w)?;
    let inv_cnt = 1.0 / (m as f64);
    let tgt_var = 0.5
        * (target_l[w..n].iter().map(|&v| v * v).sum::<f64>()
            + target_r[w..n].iter().map(|&v| v * v).sum::<f64>())
        * inv_cnt;
    let eps_floor = (1.0e-7 * tgt_var).max(1.0e-12);
    if model.oversample_upper_tiers_2x
        && outcome_2x.report_lite_1x
        && (outcome_2x.full_mse > outcome_2x.lite_mse * 1.05 + eps_floor
            || outcome_2x.ultra_mse > outcome_2x.lite_mse * 1.05 + eps_floor
            || outcome_2x.full_hf_mse > outcome_2x.lite_hf_mse * 1.10 + eps_floor)
    {
        model.oversample_upper_tiers_2x = false;
        model.tier_overrides = [None, None, None, None];
        return run_pass3_varpro_readout_inner(runner, model, in_l, in_r, target_l, target_r, w);
    }
    Ok((report_2x, outcome_2x))
}

/// Runs Pass-3 joint 4-tier causally nested `VarPro` readout identification across `Lite..Ultra`
/// including stereo `ro_pole_48k` 1-D sweep, automatic `1×` fallback when `2×` fails to improve
/// `Full`/`Ultra` over `1×` Lite, EMR `MM-Cholesky`, and 33-band stereo `LTAS` KKT refinement.
pub fn run_pass3_varpro_readout(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    warmup: usize,
) -> Result<Pass3VarProReport, TrainError> {
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let w = warmup.min(n / 4);
    let m = n - w;
    if m < 32 {
        return Err(TrainError::InvalidDimensions {
            context: "run_pass3_varpro_readout",
            expected: 32,
            actual: m,
        });
    }

    let (report, outcome) = run_pass3_varpro_readout_with_os2x_fallback(
        runner,
        model,
        &in_l[..n],
        &in_r[..n],
        &target_l[..n],
        &target_r[..n],
        w,
    )?;
    pre_conv_proj::maybe_refine_with_pre_conv_projection(
        runner,
        model,
        StereoVarProPairs {
            in_l: &in_l[..n],
            in_r: &in_r[..n],
            target_l: &target_l[..n],
            target_r: &target_r[..n],
        },
        w,
        report,
        outcome,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass3_varpro_joint_4tier_monotone_mse_and_ro_pole_sweep() {
        let runner = TrainRunner::cpu();
        let n = 1_024_usize;
        let sr = 48_000.0_f64;
        let (mut s_l, mut s_r) = (0x1234_5678_9ABC_DEF0_u64, 0xFEDC_BA98_7654_3210_u64);
        let in_l: Vec<f64> = (0..n)
            .map(|_| {
                s_l = s_l.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((s_l >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5
            })
            .collect();
        let in_r: Vec<f64> = (0..n)
            .map(|_| {
                s_r = s_r.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                0.8 * (((s_r >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5)
            })
            .collect();

        // Distinct L/R targets with direct, 0.92 HP shelf, and FIR taps across Lite/Full/Pro.
        let hp_l = eval_highpass_pole(&in_l, 0.92);
        let hp_r = eval_highpass_pole(&in_r, 0.92);
        let (mut tgt_l, mut tgt_r) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        for t in 0..n {
            let f1_l = if t >= 5 { 0.35 * in_l[t - 5] } else { 0.0 };
            let f2_l = if t >= 35 { 0.20 * in_l[t - 35] } else { 0.0 };
            let f3_l = if t >= 55 { 0.12 * in_l[t - 55] } else { 0.0 };
            tgt_l[t] = 0.70 * in_l[t] + 0.45 * hp_l[t] + f1_l + f2_l + f3_l;

            let f1_r = if t >= 7 { -0.28 * in_r[t - 7] } else { 0.0 };
            let f2_r = if t >= 40 { 0.18 * in_r[t - 40] } else { 0.0 };
            tgt_r[t] = 0.45 * in_r[t] + 0.30 * hp_r[t] + f1_r + f2_r;
        }
        let mut model = SonaModelF64::new(sr);
        model.active_fir_taps = 64;
        model.oversample_upper_tiers_2x = false;
        let rep = run_pass3_varpro_readout(&runner, &mut model, &in_l, &in_r, &tgt_l, &tgt_r, 64)
            .unwrap();
        assert!(
            rep.ro_pole_48k > 0.8 && rep.ro_pole_48k <= MAX_RO_POLE_48K_F64,
            "ro={}",
            rep.ro_pole_48k
        );
        assert!(rep.tier_mse[1] < rep.tier_mse[0], "mse={:?}", rep.tier_mse);
        assert!(rep.tier_mse[2] < rep.tier_mse[1], "mse={:?}", rep.tier_mse);
        assert!(
            rep.tier_mse[3] <= rep.tier_mse[2] + 1.0e-6,
            "mse={:?}",
            rep.tier_mse
        );
        assert!(rep.tier_mse[3] < 1.0e-3, "mse={:?}", rep.tier_mse);
        assert!((model.fir_l[5] - model.fir_r[5]).abs() > 0.15);
        assert!((model.fir_r[7] - (-0.28)).abs() < 0.05);
        assert!(model.tier_overrides.iter().all(|o| o.is_some()));
        assert_ne!(rep.stat_gram_digest, [0_u8; 32]);

        // Verify 2x oversampled upper tiers mode also populates 1x Lite + 2x Full/Pro/Ultra prefixes
        // cleanly on full audio-bandwidth excitation (<= 10 kHz).
        let audio_in_l: Vec<f64> = (0..n)
            .map(|t| {
                let ts = (t as f64) / sr;
                0.30 * (core::f64::consts::TAU * 180.0 * ts).sin()
                    + 0.25 * (core::f64::consts::TAU * 1_150.0 * ts).cos()
                    + 0.20 * (core::f64::consts::TAU * 4_200.0 * ts).sin()
                    + 0.15 * (core::f64::consts::TAU * 8_500.0 * ts).cos()
            })
            .collect();
        let audio_in_r: Vec<f64> = (0..n)
            .map(|t| {
                let ts = (t as f64) / sr;
                0.28 * (core::f64::consts::TAU * 260.0 * ts).cos()
                    + 0.22 * (core::f64::consts::TAU * 1_900.0 * ts).sin()
                    + 0.18 * (core::f64::consts::TAU * 6_300.0 * ts).sin()
            })
            .collect();
        let hp_al = eval_highpass_pole(&audio_in_l, 0.92);
        let hp_ar = eval_highpass_pole(&audio_in_r, 0.92);
        let (mut at_l, mut at_r) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        for t in 0..n {
            let f5 = if t >= 5 {
                0.35 * audio_in_l[t - 5]
            } else {
                0.0
            };
            let f35 = if t >= 35 {
                0.20 * audio_in_l[t - 35]
            } else {
                0.0
            };
            at_l[t] = 0.70 * audio_in_l[t] + 0.45 * hp_al[t] + f5 + f35;
            let f7 = if t >= 7 {
                -0.28 * audio_in_r[t - 7]
            } else {
                0.0
            };
            at_r[t] = 0.45 * audio_in_r[t] + 0.30 * hp_ar[t] + f7;
        }
        let mut model_2x = SonaModelF64::new(sr);
        model_2x.active_fir_taps = 64;
        model_2x.oversample_upper_tiers_2x = true;
        let rep_2x = run_pass3_varpro_readout(
            &runner,
            &mut model_2x,
            &audio_in_l,
            &audio_in_r,
            &at_l,
            &at_r,
            64,
        )
        .unwrap();
        assert!(
            rep_2x.tier_mse[3] < 1.0e-3,
            "2x tier_mse={:?}",
            rep_2x.tier_mse
        );
        assert!(
            rep_2x.prefix_overrides[1] && rep_2x.prefix_overrides[2] && rep_2x.prefix_overrides[3]
        );
    }
}
