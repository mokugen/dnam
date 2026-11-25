//! True-stereo Pass-3 VarPro feature matrix assembly across `Op-N[Pre]`, `1×` Lite and
//! `1×`/`2×` `Full..Ultra` causal core cascades, FIR taps, ZOH-rescaled `ro_pole_48k`, and
//! `tank_dry_mix * I + Op-T` filtering (`ref[train.pipeline.nested-varpro-schur]`).

use crate::control::stability::MAX_RO_POLE_48K_F64;
use crate::error::TrainError;
use crate::runner::TrainRunner;

use super::super::model::cascade::{
    CascadeCrossbarSpec, downsample_lane_2x_f64, run_causal_core_cascade_with_crossbar,
    sanitize_unit_f64, upsample_stereo_2x_f64,
};
use super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64, TierReadoutPrefixF64};
use super::super::pass2_opt::crossbar::{eval_crossbar_forward, eval_crossbar_forward_os2x};
use super::super::pass2_opt::op_t_quad::{eval_op_t_bank_stereo_and_adjoint, has_wet_op_t_output};
use super::super::pass2_opt::sensory::eval_sensory_quad_and_adjoint;
use super::super::pass2_opt::{eval_kautz_modal_channel_and_adjoint, eval_op_n_pre_and_adjoint};

/// Shared conjugate Kautz pair capacity per tier `[Lite, Full, Pro, Ultra]`.
pub(super) const TIER_KAUTZ_CAPS: [usize; 4] = [8, 12, 24, 32];
const _: () = assert!(TIER_KAUTZ_CAPS[0] == 8 && TIER_KAUTZ_CAPS[3] == 32);

/// Returns `true` when any upper-tier transport quad (`Full..Ultra`, indices `1..4`) has wet
/// `Op-T` output so lower tiers (`0..ti`) differ from Ultra's `4`-quad feature matrix.
#[inline]
pub(super) fn has_upper_tier_wet_op_t(model: &SonaModelF64) -> bool {
    model.transport_quads[1..].iter().any(has_wet_op_t_output)
}

/// Returns `true` when any transport quad strictly above `ti` (`(ti + 1)..4`) has wet `Op-T`.
#[inline]
pub(super) fn has_higher_tier_wet_op_t(model: &SonaModelF64, ti: usize) -> bool {
    ti < 3
        && model.transport_quads[(ti + 1)..]
            .iter()
            .any(has_wet_op_t_output)
}

/// True-stereo column-major post-warmup feature matrices and target vectors for Pass-3 VarPro.
pub(super) struct StereoFeatureSet {
    pub dims: [usize; 4],
    pub active_fir_taps: usize,
    pub pc_gain: [f64; 2],
    pub raw_l_col0: Vec<f64>,
    pub raw_r_col0: Vec<f64>,
    pub raw_l_2x: Option<Vec<f64>>,
    pub raw_r_2x: Option<Vec<f64>>,
    pub feats_l_post: Vec<f32>,
    pub feats_r_post: Vec<f32>,
    pub tgts_l_post: Vec<f32>,
    pub tgts_r_post: Vec<f32>,
    /// Post-warmup targets with tier-specific `Lite/Full/Pro` Kautz counts `[8, 12, 24]` subtracted.
    pub tier_tgts_l_prefix: [Vec<f32>; 3],
    pub tier_tgts_r_prefix: [Vec<f32>; 3],
    /// Tier-specific `[Lite, Full, Pro]` feature matrices when upper-tier `Op-T` quads have wet
    /// output (`active_opt = ti + 1 < 4`), ensuring exact feature/target `Op-T` parity per tier.
    pub tier_opt_feats_post: [Option<(Vec<f32>, Vec<f32>)>; 3],
    pub lite_1x_l_post: Option<Vec<f32>>,
    pub lite_1x_r_post: Option<Vec<f32>>,
    pub lite_1x_tgts_l_post: Option<Vec<f32>>,
    pub lite_1x_tgts_r_post: Option<Vec<f32>>,
    pub lite_1x_raw_l: Vec<f64>,
    pub lite_1x_raw_r: Vec<f64>,
}

impl StereoFeatureSet {
    #[inline]
    pub(super) fn tier_tgts_l(&self, ti: usize) -> &[f32] {
        if ti < 3 {
            &self.tier_tgts_l_prefix[ti]
        } else {
            &self.tgts_l_post
        }
    }

    #[inline]
    pub(super) fn tier_tgts_r(&self, ti: usize) -> &[f32] {
        if ti < 3 {
            &self.tier_tgts_r_prefix[ti]
        } else {
            &self.tgts_r_post
        }
    }
}

/// Evaluates the scalar 1-pole high-pass branch `u[t] - lp(u; pole)[t]`.
#[must_use]
pub(super) fn eval_highpass_pole(sig: &[f64], pole: f64) -> Vec<f64> {
    let mut hp = vec![0.0_f64; sig.len()];
    if pole <= 0.0 || !pole.is_finite() {
        return hp;
    }
    let p = pole.clamp(0.0, MAX_RO_POLE_48K_F64);
    let one_m_p = 1.0 - p;
    let mut lp = 0.0_f64;
    for (i, &u) in sig.iter().enumerate() {
        lp = p * lp + one_m_p * u;
        hp[i] = u - lp;
    }
    hp
}

#[inline]
pub(super) fn is_op_t_active_up_to(model: &SonaModelF64, active_opt: usize) -> bool {
    model.transport_quads[..active_opt.min(model.transport_quads.len())]
        .iter()
        .any(has_wet_op_t_output)
}

pub(super) fn apply_op_t_series_channel_bank(
    model: &SonaModelF64,
    sig: &[f64],
    is_right: bool,
    active_opt: usize,
) -> Vec<f64> {
    let dry = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let q_opt = active_opt.clamp(1, model.transport_quads.len());
    if !is_op_t_active_up_to(model, q_opt) {
        return sig.iter().map(|&v| dry * v).collect();
    }
    let zeros = vec![0.0_f64; sig.len()];
    let (xl, xr) = if is_right {
        (zeros.as_slice(), sig)
    } else {
        (sig, zeros.as_slice())
    };
    let (tl, tr, _) = eval_op_t_bank_stereo_and_adjoint(
        xl,
        xr,
        &model.transport_quads,
        q_opt,
        model.sample_rate_hz,
        None,
    );
    let wet = if is_right { &tr } else { &tl };
    sig.iter().zip(wet).map(|(&s, &w)| dry * s + w).collect()
}

pub(super) fn apply_op_t_stereo_bank(
    model: &SonaModelF64,
    sig_l: &[f64],
    sig_r: &[f64],
    active_opt: usize,
) -> (Vec<f64>, Vec<f64>) {
    let dry = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let q_opt = active_opt.clamp(1, model.transport_quads.len());
    if !is_op_t_active_up_to(model, q_opt) {
        return (
            sig_l.iter().map(|&v| dry * v).collect(),
            sig_r.iter().map(|&v| dry * v).collect(),
        );
    }
    let (tl, tr, _) = eval_op_t_bank_stereo_and_adjoint(
        sig_l,
        sig_r,
        &model.transport_quads,
        q_opt,
        model.sample_rate_hz,
        None,
    );
    let out_l = sig_l.iter().zip(&tl).map(|(&s, &w)| dry * s + w).collect();
    let out_r = sig_r.iter().zip(&tr).map(|(&s, &w)| dry * s + w).collect();
    (out_l, out_r)
}

fn subtract_modal_contributions_per_tier(
    model: &SonaModelF64,
    pre_conv_l: &[f64],
    pre_conv_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
) -> ([Vec<f64>; 4], [Vec<f64>; 4]) {
    let n = pre_conv_l
        .len()
        .min(pre_conv_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let active_k = model.active_kautz_pairs.min(model.kautz.len());
    if active_k == 0 {
        let base_l = target_l[..n].to_vec();
        let base_r = target_r[..n].to_vec();
        return (
            core::array::from_fn(|_| base_l.clone()),
            core::array::from_fn(|_| base_r.clone()),
        );
    }

    let mut on_l = vec![0.0_f64; n];
    let mut on_r = vec![0.0_f64; n];
    if n > 16 {
        on_l[16..n].copy_from_slice(&pre_conv_l[..(n - 16)]);
        on_r[16..n].copy_from_slice(&pre_conv_r[..(n - 16)]);
    }

    let mut raw_l = [
        vec![0.0_f64; n],
        vec![0.0_f64; n],
        vec![0.0_f64; n],
        vec![0.0_f64; n],
    ];
    let mut raw_r = [
        vec![0.0_f64; n],
        vec![0.0_f64; n],
        vec![0.0_f64; n],
        vec![0.0_f64; n],
    ];
    let mut prev_cap = 0_usize;
    for ti in 0..4 {
        let cur_cap = active_k.min(TIER_KAUTZ_CAPS[ti]);
        if ti > 0 {
            let (lo_l, hi_l) = raw_l.split_at_mut(ti);
            hi_l[0].copy_from_slice(&lo_l[ti - 1]);
            let (lo_r, hi_r) = raw_r.split_at_mut(ti);
            hi_r[0].copy_from_slice(&lo_r[ti - 1]);
        }
        if cur_cap > prev_cap {
            let (dl, _) = eval_kautz_modal_channel_and_adjoint(
                &on_l,
                &model.kautz[prev_cap..cur_cap],
                false,
                None,
                None,
            );
            let (dr, _) = eval_kautz_modal_channel_and_adjoint(
                &on_r,
                &model.kautz[prev_cap..cur_cap],
                true,
                None,
                None,
            );
            for t in 0..n {
                raw_l[ti][t] += dl[t];
                raw_r[ti][t] += dr[t];
            }
            prev_cap = cur_cap;
        }
    }

    let mut adj_l: [Vec<f64>; 4] = core::array::from_fn(|_| target_l[..n].to_vec());
    let mut adj_r: [Vec<f64>; 4] = core::array::from_fn(|_| target_r[..n].to_vec());
    let caps = TIER_KAUTZ_CAPS.map(|c| active_k.min(c));
    let mut last_eff: Option<(Vec<f64>, Vec<f64>)> = None;
    for ti in 0..4 {
        let active_opt = ti + 1;
        if ti == 0 || caps[ti] != caps[ti - 1] || has_wet_op_t_output(&model.transport_quads[ti]) {
            last_eff = Some(apply_op_t_stereo_bank(
                model, &raw_l[ti], &raw_r[ti], active_opt,
            ));
        }
        if let Some((ref eff_l, ref eff_r)) = last_eff {
            for t in 0..n {
                adj_l[ti][t] -= eff_l[t];
                adj_r[ti][t] -= eff_r[t];
            }
        }
    }
    (adj_l, adj_r)
}

pub(super) fn build_pre_conv_bus(
    model: &SonaModelF64,
    u_l: &[f64],
    u_r: &[f64],
    cascade_feats: &[[f32; 4]],
    active_quads: usize,
    rho_core: f64,
    is_2x: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n_sub = u_l.len().min(u_r.len());
    let (pc_l, pc_r) = (model.pre_conv_u_gain[0], model.pre_conv_u_gain[1]);
    let (hp_gain_l, hp_gain_r) = (model.pre_conv_hp_gain[0], model.pre_conv_hp_gain[1]);
    let ro_core = if model.ro_pole_48k > 0.0 && model.ro_pole_48k.is_finite() {
        model
            .ro_pole_48k
            .powf(rho_core)
            .clamp(0.0, MAX_RO_POLE_48K_F64)
    } else {
        0.0
    };
    let hp_l = eval_highpass_pole(&u_l[..n_sub], ro_core);
    let hp_r = eval_highpass_pole(&u_r[..n_sub], ro_core);
    let mut pre_l = vec![0.0_f64; n_sub];
    let mut pre_r = vec![0.0_f64; n_sub];
    let active_lanes = (active_quads * 4).min(40);
    for t in 0..n_sub {
        let mut bus_l = pc_l * u_l[t] + hp_gain_l * hp_l[t];
        let mut bus_r = pc_r * u_r[t] + hp_gain_r * hp_r[t];
        for l in 0..active_lanes {
            let r = &model.w_ro[l];
            if r[0] == 0.0
                && r[1] == 0.0
                && r[2] == 0.0
                && r[3] == 0.0
                && r[4] == 0.0
                && r[5] == 0.0
            {
                continue;
            }
            let q = l / 4;
            let k = l % 4;
            let fbase = (q * n_sub + t) * 6;
            let v_eff_l = f64::from(cascade_feats[fbase][k]);
            let v_eff_r = f64::from(cascade_feats[fbase + 1][k]);
            let odd_l = f64::from(cascade_feats[fbase + 2][k]);
            let odd_r = f64::from(cascade_feats[fbase + 3][k]);
            let even_l = f64::from(cascade_feats[fbase + 4][k]);
            let even_r = f64::from(cascade_feats[fbase + 5][k]);
            bus_l += r[0] * odd_l + r[2] * even_l + r[4] * v_eff_l;
            bus_r += r[1] * odd_r + r[3] * even_r + r[5] * v_eff_r;
        }
        pre_l[t] = bus_l;
        pre_r[t] = bus_r;
    }
    if is_2x {
        (
            downsample_lane_2x_f64(&pre_l, model.os_dec_c_prev1, model.os_dec_c0),
            downsample_lane_2x_f64(&pre_r, model.os_dec_c_prev1, model.os_dec_c0),
        )
    } else {
        (pre_l, pre_r)
    }
}

struct ExtractMatrixSpec<'a> {
    ux_base: &'a [f64],
    pre_conv_chan: &'a [f64],
    cascade_feats: &'a [[f32; 4]],
    cascade_frames: usize,
    is_2x: bool,
    is_right: bool,
    tier_dims: [usize; 4],
    max_tier_idx: usize,
    w: usize,
    active_taps: usize,
    options: super::Pass3VarProOptions,
}

fn extract_channel_tier_matrix(model: &SonaModelF64, spec: &ExtractMatrixSpec<'_>) -> Vec<f32> {
    let n = spec.ux_base.len();
    let m = n - spec.w;
    let total_dim = spec.tier_dims[spec.max_tier_idx];
    let active_opt = (spec.max_tier_idx + 1).clamp(1, 4);
    let op_t = is_op_t_active_up_to(model, active_opt);
    let (dec_p1, dec_c0) = (model.os_dec_c_prev1, model.os_dec_c0);
    let mut feats_post = vec![0.0_f32; total_dim * m];

    let mut write_col = |col_idx: usize, full_col: &[f64]| {
        if col_idx >= total_dim {
            return;
        }
        let filtered = if op_t || (model.tank_dry_mix - 1.0).abs() > 1.0e-12 {
            apply_op_t_series_channel_bank(model, full_col, spec.is_right, active_opt)
        } else {
            full_col.to_vec()
        };
        for t in 0..m {
            feats_post[col_idx * m + t] = filtered[spec.w + t] as f32;
        }
    };

    if !spec.options.suppress_direct_s2_linear {
        write_col(0, spec.ux_base);
    }
    // Col 1 is populated cleanly during `ro_pole_48k` joint sweep unless suppressed.

    let lite_fir = spec.active_taps.min(32);
    let full_fir = spec.active_taps.saturating_sub(32).min(16);
    let pro_fir = spec.active_taps.saturating_sub(48).min(48);
    let ultra_fir = spec.active_taps.saturating_sub(96).min(32);
    let tier_defs = [
        (0..16_usize, 0..lite_fir),
        (16..24_usize, 32..(32 + full_fir)),
        (24..32_usize, 48..(48 + pro_fir)),
        (32..40_usize, 96..(96 + ultra_fir)),
    ];

    let mut col = 2_usize;
    let lr_off = usize::from(spec.is_right);
    for (ti, (lane_range, fir_range)) in tier_defs.into_iter().enumerate() {
        if ti > spec.max_tier_idx {
            break;
        }
        for l in lane_range {
            if l < 64 && (spec.options.suppress_post_core_lane_mask & (1_u64 << l)) != 0 {
                col += 3;
                continue;
            }
            let q = l / 4;
            let k = l % 4;
            for sub_off in [2_usize, 4_usize, 0_usize] {
                let raw_sig: Vec<f64> = if spec.is_2x {
                    let s2x: Vec<f64> = (0..spec.cascade_frames)
                        .map(|t| {
                            f64::from(
                                spec.cascade_feats
                                    [(q * spec.cascade_frames + t) * 6 + sub_off + lr_off][k],
                            )
                        })
                        .collect();
                    downsample_lane_2x_f64(&s2x, dec_p1, dec_c0)
                } else {
                    (0..n)
                        .map(|t| {
                            f64::from(spec.cascade_feats[(q * n + t) * 6 + sub_off + lr_off][k])
                        })
                        .collect()
                };
                write_col(col, &raw_sig);
                col += 1;
            }
        }
        for k in fir_range {
            let mut tap_col = vec![0.0_f64; n];
            tap_col[k..n].copy_from_slice(&spec.pre_conv_chan[..(n - k)]);
            write_col(col, &tap_col);
            col += 1;
        }
    }
    feats_post
}

struct StereoExtractBase<'a> {
    ux_l: &'a [f64],
    ux_r: &'a [f64],
    pre_l: &'a [f64],
    pre_r: &'a [f64],
    cascade_feats: &'a [[f32; 4]],
    cascade_frames: usize,
    is_2x: bool,
    tier_dims: [usize; 4],
    w: usize,
    active_taps: usize,
    options: super::Pass3VarProOptions,
}

fn extract_stereo_tier_matrices(
    model: &SonaModelF64,
    base: &StereoExtractBase<'_>,
    max_tier_idx: usize,
) -> (Vec<f32>, Vec<f32>) {
    let fl = extract_channel_tier_matrix(
        model,
        &ExtractMatrixSpec {
            ux_base: base.ux_l,
            pre_conv_chan: base.pre_l,
            cascade_feats: base.cascade_feats,
            cascade_frames: base.cascade_frames,
            is_2x: base.is_2x,
            is_right: false,
            tier_dims: base.tier_dims,
            max_tier_idx,
            w: base.w,
            active_taps: base.active_taps,
            options: base.options,
        },
    );
    let fr = extract_channel_tier_matrix(
        model,
        &ExtractMatrixSpec {
            ux_base: base.ux_r,
            pre_conv_chan: base.pre_r,
            cascade_feats: base.cascade_feats,
            cascade_frames: base.cascade_frames,
            is_2x: base.is_2x,
            is_right: true,
            tier_dims: base.tier_dims,
            max_tier_idx,
            w: base.w,
            active_taps: base.active_taps,
            options: base.options,
        },
    );
    (fl, fr)
}

fn build_tier_opt_feats_post(
    model: &SonaModelF64,
    base: &StereoExtractBase<'_>,
) -> [Option<(Vec<f32>, Vec<f32>)>; 3] {
    if !has_upper_tier_wet_op_t(model) {
        return [None, None, None];
    }
    core::array::from_fn(|ti| {
        if (ti == 0 && model.oversample_upper_tiers_2x) || !has_higher_tier_wet_op_t(model, ti) {
            None
        } else {
            Some(extract_stereo_tier_matrices(model, base, ti))
        }
    })
}

/// Evaluates the shared `1x` Pass-3 front-end (`SensoryQuad` -> `Op-N[Pre]`) returning
/// `(u_sense_l, u_sense_r, ux_l, ux_r)`.
pub(super) fn eval_pass3_frontend_1x(
    model: &SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = in_l.len().min(in_r.len());
    let (u_sense_l, u_sense_r, _) =
        eval_sensory_quad_and_adjoint(model, &in_l[..n], &in_r[..n], None, None);
    let (ux_l, _) = eval_op_n_pre_and_adjoint(
        &u_sense_l,
        model.op_n_pre_gamma_bc,
        model.op_n_pre_zoh_hold_depth,
        model.op_n_pre_beta_q,
        model.sample_rate_hz,
        None,
        None,
    );
    let (ux_r, _) = eval_op_n_pre_and_adjoint(
        &u_sense_r,
        model.op_n_pre_gamma_bc,
        model.op_n_pre_zoh_hold_depth,
        model.op_n_pre_beta_q,
        model.sample_rate_hz,
        None,
        None,
    );
    (u_sense_l, u_sense_r, ux_l, ux_r)
}

pub(super) fn build_pass3_feature_sets(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: &super::super::model::StereoTrainingPairs<'_>,
    w: usize,
    options: super::Pass3VarProOptions,
) -> Result<StereoFeatureSet, TrainError> {
    let (in_l, in_r, target_l, target_r) = (pairs.in_l, pairs.in_r, pairs.target_l, pairs.target_r);
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let active_taps = model.active_fir_taps.min(NUM_FIR_TAPS_ULTRA);
    let snap_pc_fp16 = |v: f64| -> f64 {
        let clamped = if v.is_finite() {
            v.clamp(-8.0, 8.0)
        } else {
            0.0
        };
        let s = f64::from(dnam_kernels::fp16_to_f32(dnam_kernels::f32_to_fp16(
            clamped as f32,
        )));
        if s == 0.0 { 0.0 } else { s }
    };
    let default_pc = |v: f64| -> f64 {
        if v.is_finite() && v.abs() >= 1.0e-6 {
            snap_pc_fp16(v)
        } else {
            1.0
        }
    };
    let any_w_ro_pre = model
        .w_ro
        .iter()
        .any(|r| r[0..6].iter().any(|&v| v.abs() >= 1.0e-6));
    let pc_gain = if any_w_ro_pre {
        [
            snap_pc_fp16(model.pre_conv_u_gain[0]),
            snap_pc_fp16(model.pre_conv_u_gain[1]),
        ]
    } else if active_taps > 0 || model.active_kautz_pairs > 0 {
        [
            default_pc(model.pre_conv_u_gain[0]),
            default_pc(model.pre_conv_u_gain[1]),
        ]
    } else {
        [0.0, 0.0]
    };
    model.pre_conv_u_gain = pc_gain;
    model.pre_conv_hp_gain = [
        snap_pc_fp16(model.pre_conv_hp_gain[0]),
        snap_pc_fp16(model.pre_conv_hp_gain[1]),
    ];

    let (u_sense_l, u_sense_r, ux_l, ux_r) = eval_pass3_frontend_1x(model, &in_l[..n], &in_r[..n]);

    let lite_fir = active_taps.min(32);
    let full_fir = active_taps.saturating_sub(32).min(16);
    let pro_fir = active_taps.saturating_sub(48).min(48);
    let ultra_fir = active_taps.saturating_sub(96).min(32);
    let d_lite = 2 + 16 * 3 + lite_fir;
    let d_full = d_lite + 8 * 3 + full_fir;
    let d_pro = d_full + 8 * 3 + pro_fir;
    let d_ultra = d_pro + 8 * 3 + ultra_fir;
    let dims = [d_lite, d_full, d_pro, d_ultra];

    let tgts_post_fn = |adj: &[f64]| adj[w..n].iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let rho_1x = 48_000.0 / model.sample_rate_hz.max(8_000.0);

    if model.oversample_upper_tiers_2x {
        let cb_1x = eval_crossbar_forward(model, &u_sense_l, &u_sense_r, 4, false);
        let cas_1x = run_causal_core_cascade_with_crossbar(
            runner,
            model,
            &ux_l,
            &ux_r,
            CascadeCrossbarSpec {
                active_quads: 4,
                rho_core: rho_1x,
                write_checkpoints: false,
                crossbar_add: cb_1x
                    .active
                    .then_some((cb_1x.cb_add_l.as_slice(), cb_1x.cb_add_r.as_slice())),
            },
        )?;
        let (pre_1x_l, pre_1x_r) = build_pre_conv_bus(
            model,
            &ux_l,
            &ux_r,
            &cas_1x.feature_outputs,
            4,
            rho_1x,
            false,
        );
        let (adj_1x_l, adj_1x_r) =
            subtract_modal_contributions_per_tier(model, &pre_1x_l, &pre_1x_r, target_l, target_r);
        let lite_1x_l_post = Some(extract_channel_tier_matrix(
            model,
            &ExtractMatrixSpec {
                ux_base: &ux_l,
                pre_conv_chan: &pre_1x_l,
                cascade_feats: &cas_1x.feature_outputs,
                cascade_frames: n,
                is_2x: false,
                is_right: false,
                tier_dims: dims,
                max_tier_idx: 0,
                w,
                active_taps,
                options,
            },
        ));
        let lite_1x_r_post = Some(extract_channel_tier_matrix(
            model,
            &ExtractMatrixSpec {
                ux_base: &ux_r,
                pre_conv_chan: &pre_1x_r,
                cascade_feats: &cas_1x.feature_outputs,
                cascade_frames: n,
                is_2x: false,
                is_right: true,
                tier_dims: dims,
                max_tier_idx: 0,
                w,
                active_taps,
                options,
            },
        ));

        let (up_l, up_r) = upsample_stereo_2x_f64(&ux_l, &ux_r, model.os_up_quad_coeff);
        let dec_ux_l = downsample_lane_2x_f64(&up_l, model.os_dec_c_prev1, model.os_dec_c0);
        let dec_ux_r = downsample_lane_2x_f64(&up_r, model.os_dec_c_prev1, model.os_dec_c0);
        let rho_2x = 0.5 * rho_1x;
        let cb_2x = eval_crossbar_forward_os2x(model, &u_sense_l, &u_sense_r, 10, true);
        let cas_2x = run_causal_core_cascade_with_crossbar(
            runner,
            model,
            &up_l,
            &up_r,
            CascadeCrossbarSpec {
                active_quads: 10,
                rho_core: rho_2x,
                write_checkpoints: false,
                crossbar_add: cb_2x
                    .active
                    .then_some((cb_2x.cb_add_l.as_slice(), cb_2x.cb_add_r.as_slice())),
            },
        )?;
        let (pre_2x_l, pre_2x_r) = build_pre_conv_bus(
            model,
            &up_l,
            &up_r,
            &cas_2x.feature_outputs,
            10,
            rho_2x,
            true,
        );
        let (adj_tgt_l, adj_tgt_r) =
            subtract_modal_contributions_per_tier(model, &pre_2x_l, &pre_2x_r, target_l, target_r);

        let upper_base = StereoExtractBase {
            ux_l: &dec_ux_l,
            ux_r: &dec_ux_r,
            pre_l: &pre_2x_l,
            pre_r: &pre_2x_r,
            cascade_feats: &cas_2x.feature_outputs,
            cascade_frames: 2 * n,
            is_2x: true,
            tier_dims: dims,
            w,
            active_taps,
            options,
        };
        let (feats_l_post, feats_r_post) = extract_stereo_tier_matrices(model, &upper_base, 3);
        let tier_opt_feats_post = build_tier_opt_feats_post(model, &upper_base);
        Ok(StereoFeatureSet {
            dims,
            active_fir_taps: active_taps,
            pc_gain,
            raw_l_col0: dec_ux_l,
            raw_r_col0: dec_ux_r,
            raw_l_2x: Some(up_l),
            raw_r_2x: Some(up_r),
            feats_l_post,
            feats_r_post,
            tgts_l_post: tgts_post_fn(&adj_tgt_l[3]),
            tgts_r_post: tgts_post_fn(&adj_tgt_r[3]),
            tier_tgts_l_prefix: [
                tgts_post_fn(&adj_1x_l[0]),
                tgts_post_fn(&adj_tgt_l[1]),
                tgts_post_fn(&adj_tgt_l[2]),
            ],
            tier_tgts_r_prefix: [
                tgts_post_fn(&adj_1x_r[0]),
                tgts_post_fn(&adj_tgt_r[1]),
                tgts_post_fn(&adj_tgt_r[2]),
            ],
            tier_opt_feats_post,
            lite_1x_l_post,
            lite_1x_r_post,
            lite_1x_tgts_l_post: Some(tgts_post_fn(&adj_1x_l[0])),
            lite_1x_tgts_r_post: Some(tgts_post_fn(&adj_1x_r[0])),
            lite_1x_raw_l: ux_l,
            lite_1x_raw_r: ux_r,
        })
    } else {
        let cb = eval_crossbar_forward(model, &u_sense_l, &u_sense_r, 10, true);
        let cas = run_causal_core_cascade_with_crossbar(
            runner,
            model,
            &ux_l,
            &ux_r,
            CascadeCrossbarSpec {
                active_quads: 10,
                rho_core: rho_1x,
                write_checkpoints: false,
                crossbar_add: cb
                    .active
                    .then_some((cb.cb_add_l.as_slice(), cb.cb_add_r.as_slice())),
            },
        )?;
        let (pre_l, pre_r) =
            build_pre_conv_bus(model, &ux_l, &ux_r, &cas.feature_outputs, 10, rho_1x, false);
        let (adj_tgt_l, adj_tgt_r) =
            subtract_modal_contributions_per_tier(model, &pre_l, &pre_r, target_l, target_r);
        let base_1x = StereoExtractBase {
            ux_l: &ux_l,
            ux_r: &ux_r,
            pre_l: &pre_l,
            pre_r: &pre_r,
            cascade_feats: &cas.feature_outputs,
            cascade_frames: n,
            is_2x: false,
            tier_dims: dims,
            w,
            active_taps,
            options,
        };
        let (feats_l_post, feats_r_post) = extract_stereo_tier_matrices(model, &base_1x, 3);
        let tier_opt_feats_post = build_tier_opt_feats_post(model, &base_1x);
        Ok(StereoFeatureSet {
            dims,
            active_fir_taps: active_taps,
            pc_gain,
            raw_l_col0: ux_l.clone(),
            raw_r_col0: ux_r.clone(),
            raw_l_2x: None,
            raw_r_2x: None,
            feats_l_post,
            feats_r_post,
            tgts_l_post: tgts_post_fn(&adj_tgt_l[3]),
            tgts_r_post: tgts_post_fn(&adj_tgt_r[3]),
            tier_tgts_l_prefix: [
                tgts_post_fn(&adj_tgt_l[0]),
                tgts_post_fn(&adj_tgt_l[1]),
                tgts_post_fn(&adj_tgt_l[2]),
            ],
            tier_tgts_r_prefix: [
                tgts_post_fn(&adj_tgt_r[0]),
                tgts_post_fn(&adj_tgt_r[1]),
                tgts_post_fn(&adj_tgt_r[2]),
            ],
            tier_opt_feats_post,
            lite_1x_l_post: None,
            lite_1x_r_post: None,
            lite_1x_tgts_l_post: None,
            lite_1x_tgts_r_post: None,
            lite_1x_raw_l: ux_l,
            lite_1x_raw_r: ux_r,
        })
    }
}

pub(super) fn unpack_weights_into_prefix(
    weights_l: &[f64],
    weights_r: &[f64],
    active_taps: usize,
    pc_gain: [f64; 2],
    max_tier_idx: usize,
    base_model: &SonaModelF64,
) -> TierReadoutPrefixF64 {
    let mut pfx = TierReadoutPrefixF64 {
        pre_conv_u_gain: pc_gain,
        pre_conv_hp_gain: base_model.pre_conv_hp_gain,
        direct_s2_u_gain: [weights_l[0].clamp(-8.0, 8.0), weights_r[0].clamp(-8.0, 8.0)],
        direct_s2_hp_gain: [weights_l[1].clamp(-8.0, 8.0), weights_r[1].clamp(-8.0, 8.0)],
        ..TierReadoutPrefixF64::default()
    };
    for l in 0..40 {
        pfx.w_ro[l][0..6].copy_from_slice(&base_model.w_ro[l][0..6]);
    }
    for (k, kz) in base_model.kautz.iter().enumerate() {
        pfx.kautz_b_gamma[k] = kz.b_gamma;
        pfx.kautz_c_re_l[k] = kz.c_re_l;
        pfx.kautz_c_im_l[k] = kz.c_im_l;
        pfx.kautz_c_re_r[k] = kz.c_re_r;
        pfx.kautz_c_im_r[k] = kz.c_im_r;
    }
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
    let mut col = 2_usize;
    for (ti, (lane_range, fir_range)) in tier_defs.into_iter().enumerate() {
        if ti > max_tier_idx {
            break;
        }
        for l in lane_range {
            pfx.w_ro[l][6] = weights_l[col].clamp(-8.0, 8.0);
            pfx.w_ro[l][7] = weights_r[col].clamp(-8.0, 8.0);
            pfx.w_ro[l][8] = weights_l[col + 1].clamp(-8.0, 8.0);
            pfx.w_ro[l][9] = weights_r[col + 1].clamp(-8.0, 8.0);
            pfx.w_ro[l][10] = weights_l[col + 2].clamp(-8.0, 8.0);
            pfx.w_ro[l][11] = weights_r[col + 2].clamp(-8.0, 8.0);
            col += 3;
        }
        for k in fir_range {
            pfx.fir_l[k] = weights_l[col].clamp(-8.0, 8.0);
            pfx.fir_r[k] = weights_r[col].clamp(-8.0, 8.0);
            col += 1;
        }
    }
    pfx
}
