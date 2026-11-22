//! Pass-3 joint 4-tier causally nested `VarPro` + Schur-complement readout re-balancing,
//! `1×` Lite / `2×` `Full..Ultra` adaptation, stereo `ro_pole_48k` 1-D sweep, EMR `MM-Cholesky`
//! & 33-band stereo `LTAS` Newton-KKT solver (`ref[train.pipeline.nested-varpro-schur]`).

pub(crate) mod features;
mod gram_snap;

use crate::control::clamp::{MAX_GAIN_6DB_LIN, evaluate_readout_spectral_magnitudes};
use crate::control::stability::MAX_RO_POLE_48K_F64;
use crate::error::TrainError;
use crate::linalg::cholesky::solve_uplo_u_cholesky;
use crate::linalg::spectral_kkt::{
    EmrMmConfig, EmrSpectralProblem, LtasKktConfig, solve_emr_mm_cholesky, solve_ltas_newton_kkt,
};
use crate::linalg::varpro::{assemble_ozaki_normal_equations, solve_nested_tier_prefix_schur};
use crate::losses::canonical_ltas_33_centers_hz;
use crate::runner::TrainRunner;
use crate::wide_gram::{hash_gram_input_v3_f32, uplo_u_index};

use super::model::SonaModelF64;
use super::model::cascade::downsample_lane_2x_f64;
use features::{
    StereoFeatureSet, apply_op_t_series_channel, build_pass3_feature_sets, eval_highpass_pole,
    unpack_weights_into_prefix,
};
use gram_snap::{
    build_tier_fp16_col_bounds, snap_fp16_val, snap_weights_direct, snap_weights_gram_coordinate,
};

/// Canonical per-tier 33-band LTAS max tilt limits `[Lite, Full, Pro, Ultra]` in dB
/// (`ref[qual.tiers.floors-not-stopping-criteria]`).
const TIER_LTAS_TILT_BUDGET_DB: [f64; 4] = [0.75, 0.35, 0.15, 0.10];

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

fn compute_channel_mse(weights: &[f64], feats_post: &[f32], tgts_post: &[f32], dim: usize) -> f64 {
    let m = tgts_post.len().max(1);
    let mut sq = 0.0_f64;
    for t in 0..tgts_post.len() {
        let mut pred = 0.0_f64;
        for c in 0..dim {
            pred += weights[c] * f64::from(feats_post[c * tgts_post.len() + t]);
        }
        let d = f64::from(tgts_post[t]) - pred;
        sq += d * d;
    }
    sq / (m as f64)
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
        let (Ok(wl), Ok(wr)) = (
            solve_uplo_u_cholesky(&gl, &rl, 1.0e-5),
            solve_uplo_u_cholesky(&gr, &rr, 1.0e-5),
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

struct ChannelSpectralCtx<'a> {
    gram: &'a [f64],
    rhs: &'a [f64],
    feats_post: &'a [f32],
    tgts_post: &'a [f32],
    sr_hz: f64,
    ro_pole_48k: f64,
    active_fir_taps: usize,
    emit_prefix_overrides: bool,
}

fn refine_channel_tier_emr_and_ltas_kkt(
    base_weights: &[f64],
    dim: usize,
    tier_idx: usize,
    ctx: &ChannelSpectralCtx<'_>,
) -> Vec<f64> {
    let m = ctx.tgts_post.len();
    let dft_len = m.min(512);
    let centers = canonical_ltas_33_centers_hz();
    let n_bands = centers.len();
    let packed_sub = dim * (dim + 1) / 2;
    let gram_sub = &ctx.gram[..packed_sub];
    let rhs_sub = &ctx.rhs[..dim];

    let mut basis_re = vec![0.0_f64; n_bands * dim];
    let mut basis_im = vec![0.0_f64; n_bands * dim];
    let mut tgt_re = vec![0.0_f64; n_bands];
    let mut tgt_im = vec![0.0_f64; n_bands];
    let mut band_rows = vec![0.0_f64; n_bands * dim];
    let mut band_amps = vec![0.0_f64; n_bands];

    for (b, &fc_hz) in centers.iter().enumerate() {
        let omega = core::f64::consts::TAU * fc_hz / ctx.sr_hz.max(8_000.0);
        let (mut yr, mut yi) = (0.0_f64, 0.0_f64);
        for t in 0..dft_len {
            let (s, c) = (omega * (t as f64)).sin_cos();
            let yt = f64::from(ctx.tgts_post[t]);
            yr += yt * c;
            yi -= yt * s;
            for col in 0..dim {
                let fv = f64::from(ctx.feats_post[col * m + t]);
                basis_re[b * dim + col] += fv * c;
                basis_im[b * dim + col] -= fv * s;
            }
        }
        let amp = (yr * yr + yi * yi).sqrt();
        tgt_re[b] = yr;
        tgt_im[b] = yi;
        band_amps[b] = amp.max(1.0e-6);
        let (ur, ui) = if amp > 1.0e-9 {
            (yr / amp, yi / amp)
        } else {
            (1.0, 0.0)
        };
        for col in 0..dim {
            band_rows[b * dim + col] = ur * basis_re[b * dim + col] + ui * basis_im[b * dim + col];
        }
    }

    let eval_max_tilt_db = |w: &[f64]| -> f64 {
        let mut mx = 0.0_f64;
        for b in 0..n_bands {
            let mut pr = 0.0_f64;
            let mut pi = 0.0_f64;
            for col in 0..dim {
                pr += w[col] * basis_re[b * dim + col];
                pi += w[col] * basis_im[b * dim + col];
            }
            let p_amp = (pr * pr + pi * pi).sqrt().max(1.0e-12);
            let t_amp = band_amps[b];
            mx = mx.max((20.0 * (p_amp / t_amp).log10()).abs());
        }
        mx
    };

    let within_6db = |w: &[f64]| -> bool {
        let mags =
            evaluate_readout_spectral_magnitudes(&[w[0], w[1]], ctx.ro_pole_48k, ctx.sr_hz, 16);
        mags.iter().all(|&v| v.is_finite() && v <= MAX_GAIN_6DB_LIN)
    };

    let base_mse = compute_channel_mse(base_weights, ctx.feats_post, ctx.tgts_post, dim);
    let mut best_w = base_weights[..dim].to_vec();
    let mut best_mse = base_mse;
    let mut best_tilt = eval_max_tilt_db(&best_w);

    if let Ok(emr) = solve_emr_mm_cholesky(
        &EmrSpectralProblem {
            num_bands: n_bands,
            feature_dim: dim,
            band_freqs_hz: &centers,
            basis_re: &basis_re,
            basis_im: &basis_im,
            target_re: &tgt_re,
            target_im: &tgt_im,
        },
        EmrMmConfig::default(),
    ) && emr.weights.iter().all(|v| v.is_finite())
        && within_6db(&emr.weights)
    {
        let emr_mse = compute_channel_mse(&emr.weights, ctx.feats_post, ctx.tgts_post, dim);
        let emr_tilt = eval_max_tilt_db(&emr.weights);
        if emr_mse < best_mse * 0.99 && emr_tilt <= best_tilt + 0.05 {
            best_w = emr.weights;
            best_mse = emr_mse;
            best_tilt = emr_tilt;
        }
    }

    if best_tilt > TIER_LTAS_TILT_BUDGET_DB[tier_idx]
        && let Ok(kkt) = solve_ltas_newton_kkt(
            dim,
            n_bands,
            gram_sub,
            rhs_sub,
            &band_rows,
            &band_amps,
            LtasKktConfig {
                ridge: 1.0e-5,
                max_outer_iters: 8,
                max_tilt_db: TIER_LTAS_TILT_BUDGET_DB[tier_idx],
                penalty_rho: 2.0,
            },
        )
        && kkt.weights.iter().all(|v| v.is_finite())
        && within_6db(&kkt.weights)
    {
        let kkt_mse = compute_channel_mse(&kkt.weights, ctx.feats_post, ctx.tgts_post, dim);
        let kkt_tilt = eval_max_tilt_db(&kkt.weights);
        if kkt_mse <= best_mse * 1.5 && kkt_tilt < best_tilt {
            best_w = kkt.weights;
            best_tilt = kkt_tilt;
        }
    }

    let base_within_6db = within_6db(&best_w);
    let fp16_bounds =
        build_tier_fp16_col_bounds(tier_idx, ctx.active_fir_taps, ctx.emit_prefix_overrides);
    let snapped = snap_weights_gram_coordinate(&best_w, gram_sub, 1.0e-5, &fp16_bounds);
    let tilt_budget = best_tilt.max(TIER_LTAS_TILT_BUDGET_DB[tier_idx]) + 0.05;
    if snapped.iter().all(|v| v.is_finite())
        && (!base_within_6db || within_6db(&snapped))
        && eval_max_tilt_db(&snapped) <= tilt_budget
    {
        return snapped;
    }

    let mut fallback = snap_weights_direct(&best_w, &fp16_bounds);
    if base_within_6db && !within_6db(&fallback) {
        let (b0, b1) = (fallback[0], fallback[1]);
        for &scale_k in &[0.98_f64, 0.90, 0.75, 0.5, 0.0] {
            let mags =
                evaluate_readout_spectral_magnitudes(&[b0, b1], ctx.ro_pole_48k, ctx.sr_hz, 16);
            let peak = mags.iter().copied().fold(0.0_f64, f64::max);
            let adj = if peak > MAX_GAIN_6DB_LIN && peak.is_finite() {
                scale_k * (MAX_GAIN_6DB_LIN / peak)
            } else {
                scale_k
            };
            fallback[0] = snap_fp16_val(b0 * adj, -8.0, 8.0);
            fallback[1] = snap_fp16_val(b1 * adj, -8.0, 8.0);
            if within_6db(&fallback) {
                break;
            }
        }
    }
    fallback
}

/// Runs Pass-3 joint 4-tier causally nested `VarPro` readout identification across `Lite..Ultra`
/// including stereo `ro_pole_48k` 1-D sweep, optional `1×` Lite vs `2×` `Full..Ultra` adaptation,
/// EMR `MM-Cholesky`, and 33-band stereo `LTAS` KKT constraint enforcement.
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

    let (base_gl, base_rl_v) = assemble_ozaki_normal_equations(
        runner,
        &fset.feats_l_post,
        &fset.tgts_l_post,
        m,
        total_dim,
        1,
        false,
    )?;
    let (base_gr, base_rr_v) = assemble_ozaki_normal_equations(
        runner,
        &fset.feats_r_post,
        &fset.tgts_r_post,
        m,
        total_dim,
        1,
        false,
    )?;
    let base_rl: Vec<f64> = base_rl_v.into_iter().map(|v| v[0]).collect();
    let base_rr: Vec<f64> = base_rr_v.into_iter().map(|v| v[0]).collect();

    let sweep =
        sweep_ro_pole_48k_joint_stereo(model, &fset, &base_gl, &base_rl, &base_gr, &base_rr, w);
    model.ro_pole_48k = sweep.best_ro.clamp(0.0, MAX_RO_POLE_48K_F64);
    fset.feats_l_post[m..(2 * m)].copy_from_slice(&sweep.best_hp_l);
    fset.feats_r_post[m..(2 * m)].copy_from_slice(&sweep.best_hp_r);

    let schur_l = solve_nested_tier_prefix_schur(
        &sweep.best_gram_l,
        &sweep.best_rhs_l,
        &sweep.best_rhs_l,
        fset.dims,
        1.0e-5,
        1.0e-3,
    )?;
    let schur_r = solve_nested_tier_prefix_schur(
        &sweep.best_gram_r,
        &sweep.best_rhs_r,
        &sweep.best_rhs_r,
        fset.dims,
        1.0e-5,
        1.0e-3,
    )?;

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
            assemble_ozaki_normal_equations(runner, fl, tl, m, d_lite, 1, false)?;
        let (gr_1x, rr_1x_v) =
            assemble_ozaki_normal_equations(runner, fr, tr, m, d_lite, 1, false)?;
        let rl_1x: Vec<f64> = rl_1x_v.into_iter().map(|v| v[0]).collect();
        let rr_1x: Vec<f64> = rr_1x_v.into_iter().map(|v| v[0]).collect();
        let wl_1x = solve_uplo_u_cholesky(&gl_1x, &rl_1x, 1.0e-5)?;
        let wr_1x = solve_uplo_u_cholesky(&gr_1x, &rr_1x, 1.0e-5)?;
        Some((gl_1x, rl_1x, wl_1x, gr_1x, rr_1x, wr_1x))
    } else {
        None
    };

    for ti in 0..4 {
        model.prefix_override_needed[ti] = (ti >= 1)
            && (schur_l.prefix_override_needed[ti]
                || schur_r.prefix_override_needed[ti]
                || model.oversample_upper_tiers_2x);
    }
    let emit_prefix_overrides = model.prefix_override_needed[1..].iter().any(|&v| v);

    let mut tier_mse = [0.0_f64; 4];
    for (ti, &dim_t) in fset.dims.iter().enumerate() {
        let (wl_refined, wr_refined) = if ti == 0
            && let Some((ref gl_1x, ref rl_1x, ref wl_1x, ref gr_1x, ref rr_1x, ref wr_1x)) =
                lite_1x_sol
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
                },
            );
            tier_mse[0] = 0.5
                * (compute_channel_mse(&wl_r, fl, tl, dim_t)
                    + compute_channel_mse(&wr_r, fr, tr, dim_t));
            (wl_r, wr_r)
        } else {
            let wl_raw: Vec<f64> = schur_l.joint_prefix_weights[ti]
                .iter()
                .map(|row| row[0])
                .collect();
            let wr_raw: Vec<f64> = schur_r.joint_prefix_weights[ti]
                .iter()
                .map(|row| row[0])
                .collect();
            let wl_r = refine_channel_tier_emr_and_ltas_kkt(
                &wl_raw,
                dim_t,
                ti,
                &ChannelSpectralCtx {
                    gram: &sweep.best_gram_l,
                    rhs: &sweep.best_rhs_l,
                    feats_post: &fset.feats_l_post,
                    tgts_post: &fset.tgts_l_post,
                    sr_hz: model.sample_rate_hz,
                    ro_pole_48k: model.ro_pole_48k,
                    active_fir_taps: fset.active_fir_taps,
                    emit_prefix_overrides,
                },
            );
            let wr_r = refine_channel_tier_emr_and_ltas_kkt(
                &wr_raw,
                dim_t,
                ti,
                &ChannelSpectralCtx {
                    gram: &sweep.best_gram_r,
                    rhs: &sweep.best_rhs_r,
                    feats_post: &fset.feats_r_post,
                    tgts_post: &fset.tgts_r_post,
                    sr_hz: model.sample_rate_hz,
                    ro_pole_48k: model.ro_pole_48k,
                    active_fir_taps: fset.active_fir_taps,
                    emit_prefix_overrides,
                },
            );
            tier_mse[ti] = 0.5
                * (compute_channel_mse(&wl_r, &fset.feats_l_post, &fset.tgts_l_post, dim_t)
                    + compute_channel_mse(&wr_r, &fset.feats_r_post, &fset.tgts_r_post, dim_t));
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

    Ok(Pass3VarProReport {
        ro_pole_48k: model.ro_pole_48k,
        tier_mse,
        prefix_overrides: model.prefix_override_needed,
        stat_gram_digest,
    })
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
