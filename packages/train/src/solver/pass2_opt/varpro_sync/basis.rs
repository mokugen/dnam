//! Linear basis column builders and `uplo='U'` Ozaki-Cholesky solve helpers for
//! [`super::varpro_sync`].

use crate::error::TrainError;
use crate::linalg::cholesky::solve_uplo_adaptive_ridge;
use crate::linalg::varpro::assemble_ozaki_normal_equations;
use crate::runner::{ActiveComputeBackend, TrainRunner};
use crate::wide_gram::uplo_u_index;

const VARPRO_MAX_READOUT_ABS: f64 = 7.5;
const VARPRO_MAX_READOUT_L1_NORM: f64 = 48.0;
const _: () = assert!(
    VARPRO_MAX_READOUT_ABS > 0.0
        && VARPRO_MAX_READOUT_ABS < 8.0
        && VARPRO_MAX_READOUT_L1_NORM > 0.0
        && VARPRO_MAX_READOUT_L1_NORM < 100.0
);

use super::super::super::model::cascade::sanitize_unit_f64;
use super::super::super::model::{SonaModelF64, TierReadoutPrefixF64};
use super::super::adjoints::KautzChannelStepper;

#[derive(Clone, Debug)]
pub(super) struct ChannelReadoutSol {
    pub direct_s2_u: f64,
    pub direct_s2_hp: f64,
    pub w_ro_6_to_11: Vec<[f64; 3]>,
    pub fir: Vec<f64>,
    pub kautz_indices: Vec<usize>,
    pub kautz_c_re: Vec<f64>,
    pub kautz_c_im: Vec<f64>,
}

fn snapshot_prefix_from_model(model: &SonaModelF64) -> TierReadoutPrefixF64 {
    let mut ovr = TierReadoutPrefixF64 {
        pre_conv_u_gain: model.pre_conv_u_gain,
        pre_conv_hp_gain: model.pre_conv_hp_gain,
        direct_s2_u_gain: model.direct_s2_u_gain,
        direct_s2_hp_gain: model.direct_s2_hp_gain,
        w_ro: model.w_ro,
        fir_l: model.fir_l,
        fir_r: model.fir_r,
        ..TierReadoutPrefixF64::default()
    };
    for (k, kp) in model.kautz.iter().enumerate() {
        ovr.kautz_b_gamma[k] = kp.b_gamma;
        ovr.kautz_c_re_l[k] = kp.c_re_l;
        ovr.kautz_c_im_l[k] = kp.c_im_l;
        ovr.kautz_c_re_r[k] = kp.c_re_r;
        ovr.kautz_c_im_r[k] = kp.c_im_r;
    }
    ovr
}

#[inline]
fn blend_readout(old: f64, new: f64, alpha: f64) -> f64 {
    if alpha >= 1.0 {
        new
    } else {
        old + alpha * (new - old)
    }
}

pub(super) fn apply_readout_solution(
    model: &mut SonaModelF64,
    base_view: &SonaModelF64,
    tier_idx: usize,
    sol_lr: (&ChannelReadoutSol, &ChannelReadoutSol),
    sync_top_level: bool,
    blend_alpha: f64,
) {
    let (sol_l, sol_r) = sol_lr;
    let ti = tier_idx.min(3);
    let mut ovr = model.tier_overrides[ti]
        .take()
        .unwrap_or_else(|| Box::new(snapshot_prefix_from_model(model)));
    ovr.pre_conv_u_gain = model.pre_conv_u_gain;
    ovr.pre_conv_hp_gain = model.pre_conv_hp_gain;
    ovr.direct_s2_u_gain = [
        blend_readout(
            base_view.direct_s2_u_gain[0],
            sol_l.direct_s2_u,
            blend_alpha,
        ),
        blend_readout(
            base_view.direct_s2_u_gain[1],
            sol_r.direct_s2_u,
            blend_alpha,
        ),
    ];
    ovr.direct_s2_hp_gain = [
        blend_readout(
            base_view.direct_s2_hp_gain[0],
            sol_l.direct_s2_hp,
            blend_alpha,
        ),
        blend_readout(
            base_view.direct_s2_hp_gain[1],
            sol_r.direct_s2_hp,
            blend_alpha,
        ),
    ];

    let n_rows = sol_l.w_ro_6_to_11.len().min(sol_r.w_ro_6_to_11.len());
    for row in 0..n_rows {
        ovr.w_ro[row][0..6].copy_from_slice(&model.w_ro[row][0..6]);
        ovr.w_ro[row][6] = blend_readout(
            base_view.w_ro[row][6],
            sol_l.w_ro_6_to_11[row][0],
            blend_alpha,
        );
        ovr.w_ro[row][7] = blend_readout(
            base_view.w_ro[row][7],
            sol_r.w_ro_6_to_11[row][0],
            blend_alpha,
        );
        ovr.w_ro[row][8] = blend_readout(
            base_view.w_ro[row][8],
            sol_l.w_ro_6_to_11[row][1],
            blend_alpha,
        );
        ovr.w_ro[row][9] = blend_readout(
            base_view.w_ro[row][9],
            sol_r.w_ro_6_to_11[row][1],
            blend_alpha,
        );
        ovr.w_ro[row][10] = blend_readout(
            base_view.w_ro[row][10],
            sol_l.w_ro_6_to_11[row][2],
            blend_alpha,
        );
        ovr.w_ro[row][11] = blend_readout(
            base_view.w_ro[row][11],
            sol_r.w_ro_6_to_11[row][2],
            blend_alpha,
        );
    }
    let f_cap = sol_l.fir.len().min(sol_r.fir.len());
    for i in 0..f_cap {
        ovr.fir_l[i] = blend_readout(base_view.fir_l[i], sol_l.fir[i], blend_alpha);
        ovr.fir_r[i] = blend_readout(base_view.fir_r[i], sol_r.fir[i], blend_alpha);
    }

    for (pos, &k) in sol_l.kautz_indices.iter().enumerate() {
        ovr.kautz_b_gamma[k] = model.kautz[k].b_gamma;
        ovr.kautz_c_re_l[k] = blend_readout(
            base_view.kautz[k].c_re_l,
            sol_l.kautz_c_re[pos],
            blend_alpha,
        );
        ovr.kautz_c_im_l[k] = blend_readout(
            base_view.kautz[k].c_im_l,
            sol_l.kautz_c_im[pos],
            blend_alpha,
        );
        ovr.kautz_c_re_r[k] = blend_readout(
            base_view.kautz[k].c_re_r,
            sol_r.kautz_c_re[pos],
            blend_alpha,
        );
        ovr.kautz_c_im_r[k] = blend_readout(
            base_view.kautz[k].c_im_r,
            sol_r.kautz_c_im[pos],
            blend_alpha,
        );
    }

    if sync_top_level {
        model.direct_s2_u_gain = ovr.direct_s2_u_gain;
        model.direct_s2_hp_gain = ovr.direct_s2_hp_gain;
        for row in 0..n_rows {
            model.w_ro[row][6..12].copy_from_slice(&ovr.w_ro[row][6..12]);
        }
        model.fir_l[..f_cap].copy_from_slice(&ovr.fir_l[..f_cap]);
        model.fir_r[..f_cap].copy_from_slice(&ovr.fir_r[..f_cap]);
        for &k in &sol_l.kautz_indices {
            model.kautz[k].c_re_l = ovr.kautz_c_re_l[k];
            model.kautz[k].c_im_l = ovr.kautz_c_im_l[k];
            model.kautz[k].c_re_r = ovr.kautz_c_re_r[k];
            model.kautz[k].c_im_r = ovr.kautz_c_im_r[k];
        }
    }
    model.tier_overrides[ti] = Some(ovr);
}

pub(super) fn append_fir_cols(
    feats: &mut Vec<f32>,
    anchors: &mut Vec<(usize, f64)>,
    pre_conv_ch: &[f64],
    existing_fir: &[f64],
    tap_count: usize,
    spec: (usize, usize, f64),
) {
    let (w, n, dry_mix) = spec;
    let mut shifted = vec![0.0_f64; n];
    for (k, &old_tap) in existing_fir.iter().enumerate().take(tap_count) {
        shifted[..k.min(n)].fill(0.0);
        if k < n {
            shifted[k..n].copy_from_slice(&pre_conv_ch[..(n - k)]);
        }
        let col_idx = anchors.len();
        for &v in &shifted[w..n] {
            feats.push((dry_mix * v) as f32);
        }
        anchors.push((col_idx, old_tap));
    }
}

pub(super) fn extract_kautz_basis_modes(
    view: &SonaModelF64,
    onset: &[f64],
    active_k_cap: usize,
    active_k_indices: &[usize],
    is_right: bool,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = onset.len();
    let mut stepper = KautzChannelStepper::new(&view.kautz[..active_k_cap], is_right, n);
    for &ut in onset {
        let _ = stepper.step_forward_frame(ut, true);
    }
    let (sr_hist, si_hist) = stepper.state_history();
    active_k_indices
        .iter()
        .map(|&k| {
            let mut mode_re = Vec::with_capacity(n);
            let mut mode_im = Vec::with_capacity(n);
            for t in 0..n {
                mode_re.push(sr_hist[t * active_k_cap + k]);
                mode_im.push(si_hist[t * active_k_cap + k]);
            }
            (mode_re, mode_im)
        })
        .collect()
}

pub(super) fn append_kautz_cols(
    feats: &mut Vec<f32>,
    anchors: &mut Vec<(usize, f64)>,
    view: &SonaModelF64,
    modes: &[(Vec<f64>, Vec<f64>)],
    active_k_indices: &[usize],
    spec: (usize, usize, f64, bool),
) {
    let (w, n, dry_mix, is_right) = spec;
    for (pos, &k) in active_k_indices.iter().enumerate() {
        let (old_re, old_im) = if is_right {
            (view.kautz[k].c_re_r, view.kautz[k].c_im_r)
        } else {
            (view.kautz[k].c_re_l, view.kautz[k].c_im_l)
        };
        let (ref mode_re, ref mode_im) = modes[pos];
        let col_re = anchors.len();
        for &v in &mode_re[w..n] {
            feats.push((2.0 * dry_mix * v) as f32);
        }
        anchors.push((col_re, old_re));
        let col_im = anchors.len();
        for &v in &mode_im[w..n] {
            feats.push((-2.0 * dry_mix * v) as f32);
        }
        anchors.push((col_im, old_im));
    }
}

fn is_within_readout_bounds(sol: &[f64]) -> bool {
    let mut l1_norm = 0.0_f64;
    for &v in sol {
        if !v.is_finite() || v.abs() > VARPRO_MAX_READOUT_ABS {
            return false;
        }
        l1_norm += v.abs();
    }
    l1_norm <= VARPRO_MAX_READOUT_L1_NORM
}

pub(super) fn solve_ls_column_system(
    runner: &TrainRunner,
    feats: &[f32],
    tgts_f32: &[f32],
    anchors: &[(usize, f64)],
    m: usize,
    feat_dim: usize,
) -> Result<Vec<f64>, TrainError> {
    let force_wide192 = matches!(
        runner.active_backend(),
        ActiveComputeBackend::CpuSimd { .. }
    );
    let (gram, rhs_v) =
        assemble_ozaki_normal_equations(runner, feats, tgts_f32, m, feat_dim, 1, force_wide192)?;
    let rhs: Vec<f64> = rhs_v.into_iter().map(|v| v[0]).collect();

    let mut last_sol = vec![0.0_f64; feat_dim];
    if let Ok((sol, _)) =
        solve_uplo_adaptive_ridge(feat_dim, &gram, &rhs, &[], Some((anchors, 1.0e-6)), 1.0e-8)
    {
        if is_within_readout_bounds(&sol) {
            return Ok(sol);
        }
        last_sol = sol;
    }

    let mut trace = 0.0_f64;
    for j in 0..feat_dim {
        trace += gram[uplo_u_index(j, j)].max(0.0);
    }
    let mean_diag = (trace / (feat_dim.max(1) as f64)).max(1.0e-12);
    let anchor_w = (1.0e-6 * mean_diag).clamp(1.0e-8, 1.0e-4);

    for &rel in &[1.0e-5_f64, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0] {
        let ridge = (rel * mean_diag).max(1.0e-5);
        if let Ok((sol, _)) =
            solve_uplo_adaptive_ridge(feat_dim, &gram, &rhs, &[], Some((anchors, anchor_w)), ridge)
        {
            let within = is_within_readout_bounds(&sol);
            last_sol = sol;
            if within {
                return Ok(last_sol);
            }
        }
    }

    let mut max_abs = 0.0_f64;
    let mut l1_norm = 0.0_f64;
    for v in &mut last_sol {
        if !v.is_finite() {
            *v = 0.0;
        } else {
            let av = v.abs();
            max_abs = max_abs.max(av);
            l1_norm += av;
        }
    }
    let scale = (VARPRO_MAX_READOUT_ABS / max_abs.max(1.0e-12))
        .min(VARPRO_MAX_READOUT_L1_NORM / l1_norm.max(1.0e-12))
        .min(1.0);
    for v in &mut last_sol {
        *v *= scale;
    }
    Ok(last_sol)
}

pub(super) fn unpack_core_and_fir_sol(
    sol: &[f64],
    n_rows: usize,
    fir_taps: usize,
) -> (f64, f64, Vec<[f64; 3]>, Vec<f64>) {
    let direct_s2_u = sanitize_unit_f64(sol[0], -8.0, 8.0, 0.0);
    let direct_s2_hp = sanitize_unit_f64(sol[1], -8.0, 8.0, 0.0);
    let mut cursor = 2_usize;
    let mut w_ro_6_to_11 = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        w_ro_6_to_11.push([
            sanitize_unit_f64(sol[cursor], -8.0, 8.0, 0.0),
            sanitize_unit_f64(sol[cursor + 1], -8.0, 8.0, 0.0),
            sanitize_unit_f64(sol[cursor + 2], -8.0, 8.0, 0.0),
        ]);
        cursor += 3;
    }
    let fir: Vec<f64> = sol[cursor..(cursor + fir_taps)]
        .iter()
        .map(|&v| sanitize_unit_f64(v, -16.0, 16.0, 0.0))
        .collect();
    (direct_s2_u, direct_s2_hp, w_ro_6_to_11, fir)
}
