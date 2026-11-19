#![forbid(unsafe_code)]
//! Differentiable `2x -> 1x` ZOH Parameter Map and Exact Reverse-Mode Jacobian Pullback (`1x -> 2x`).

use dnam_engine::stages::stage1_svf::SVF_W_MAX;
use dnam_engine::stages::stage2_gru::{MAX_BLOCK_LANES, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0};
use dnam_engine::stages::stage3_cab::V3Rate1xCalibration;

use crate::solver::backends::TrainerBackend;

use super::super::basis::solve_matryoshka_varpro_cholesky_with_backend;
use super::super::lbfgs::run_lbfgs_with_ws;
use super::super::model_f64::{V3GruBlockF64, V3MatryoshkaOptF64, V3TrainDataset};
use super::super::primitives::{sigmoid_pade5_with_deriv, tanh_pade5_with_deriv};
use super::{AdjointWorkspace, evaluate_matryoshka_loss_f64_with_backend};

#[inline]
fn inv_tanh_pade5_pos_f64(y: f64) -> f64 {
    if y <= 0.0 {
        return 0.0;
    }
    if y >= 1.0 {
        return 4.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 4.0_f64);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if tanh_pade5_with_deriv(mid).0 < y {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[inline]
fn inv_sigmoid_pade5_f64(p: f64) -> f64 {
    if p <= 0.5 {
        -2.0 * inv_tanh_pade5_pos_f64((1.0 - 2.0 * p).clamp(0.0, 1.0))
    } else {
        2.0 * inv_tanh_pade5_pos_f64((2.0 * p - 1.0).clamp(0.0, 1.0))
    }
}

#[inline]
fn rescale_w_with_deriv(w: f64, lo: f64, hi: f64, w_max: f64) -> (f64, f64, f64) {
    let wc = w.clamp(1.0e-4, w_max);
    let d_wc = if w > 1.0e-4 && w < w_max { 1.0 } else { 0.0 };
    let rad = 1.0 + 1.5625 * wc * wc;
    let s = rad.sqrt();
    let raw = 2.0 * wc / s;
    let val = raw.clamp(lo, hi);
    let d_raw = d_wc * (2.0 / (s * rad));
    let d_val = if raw > lo && raw < hi { d_raw } else { 0.0 };
    (val, raw, d_val)
}

#[inline]
fn rescale_pole_sq_with_deriv(p: f64, lo: f64, hi: f64) -> (f64, f64) {
    let pc = p.clamp(lo, hi);
    let dp_c = if p > lo && p < hi { 1.0 } else { 0.0 };
    let raw = pc * pc;
    let val = raw.clamp(lo, hi);
    let d_val = if raw > lo && raw < hi {
        dp_c * 2.0 * pc
    } else {
        0.0
    };
    (val, d_val)
}

#[inline]
fn rescale_feat_beta_with_deriv(beta: f64, lo: f64, hi: f64) -> (f64, f64) {
    let bc = beta.clamp(lo, hi);
    let db_c = if beta > lo && beta < hi { 1.0 } else { 0.0 };
    let raw = bc * (2.0 - bc);
    let val = raw.clamp(lo, hi);
    let d_val = if raw > lo && raw < hi {
        db_c * 2.0 * (1.0 - bc)
    } else {
        0.0
    };
    (val, d_val)
}

#[inline]
fn lane_w_r_max(blk: &V3GruBlockF64, dst: usize) -> f64 {
    let mut w_r_max = 0.0_f64;
    for f in 0..NUM_IN_FEATS_B0 {
        w_r_max = w_r_max.max(blk.w_in_r[f][dst].abs());
    }
    for s in 0..MAX_BLOCK_LANES {
        w_r_max = w_r_max
            .max(blk.w_prev_r[dst][s].abs())
            .max(blk.w_rec_r[dst][s].abs());
    }
    w_r_max
}

/// Builds the exact `f64` ZOH-rescaled `1x` (`48 kHz`) image of a `2x` (`96 kHz`) Matryoshka model.
#[must_use]
#[inline(never)]
pub(in crate::solver::v3_varpro) fn rescale_model_2x_to_1x_f64(
    src: &V3MatryoshkaOptF64,
) -> V3MatryoshkaOptF64 {
    let mut out = src.clone();
    out.os2x = false;
    out.tier_os2x = [false; 4];
    out.os_up_zoh_mix = 0.0;
    out.rate_1x_cal = None;

    let w_max = f64::from(SVF_W_MAX);
    let (bg_1x, _, _) = rescale_w_with_deriv(src.svf_base_g, 1.0e-4, w_max, w_max);
    let (gmin_1x, _, _) = rescale_w_with_deriv(src.svf_g_min, 1.0e-4, w_max, w_max);
    let (_, raw_gmax, _) = rescale_w_with_deriv(src.svf_g_max, 1.0e-4, w_max, w_max);
    out.svf_base_g = bg_1x;
    out.svf_g_min = gmin_1x;
    out.svf_g_max = raw_gmax.clamp(gmin_1x, w_max);
    out.svf_det_lp_pole = rescale_pole_sq_with_deriv(src.svf_det_lp_pole, 0.05, 0.995).0;
    out.svf_atk_alpha = rescale_pole_sq_with_deriv(src.svf_atk_alpha, 0.001, 0.9999).0;
    out.svf_rel_alpha = rescale_pole_sq_with_deriv(src.svf_rel_alpha, 0.001, 0.99995).0;

    out.feat_in_hp_beta = rescale_feat_beta_with_deriv(src.feat_in_hp_beta, 1.0e-4, 0.50).0;
    out.feat_fast_atk_beta = rescale_feat_beta_with_deriv(src.feat_fast_atk_beta, 1.0e-4, 0.999).0;
    out.feat_fast_rel_beta = rescale_feat_beta_with_deriv(src.feat_fast_rel_beta, 1.0e-5, 0.999).0;
    out.feat_slow_atk_beta = rescale_feat_beta_with_deriv(src.feat_slow_atk_beta, 1.0e-5, 0.50).0;
    out.feat_slow_rel_beta = rescale_feat_beta_with_deriv(src.feat_slow_rel_beta, 1.0e-6, 0.10).0;

    for b in 0..NUM_GRU_BLOCKS {
        let sblk = &src.blocks[b];
        let dblk = &mut out.blocks[b];
        for dst in 0..MAX_BLOCK_LANES {
            let bz_c = sblk.b_z[dst].clamp(-7.90, 7.90);
            let z2x = sigmoid_pade5_with_deriv(bz_c).0;
            let mult = 2.0 - z2x;
            let z1x = (z2x * mult).clamp(1.0e-6, 0.9997);
            dblk.b_z[dst] = inv_sigmoid_pade5_f64(z1x).clamp(-7.90, 7.90);
            let z_slope = 2.0 / mult;
            for f in 0..NUM_IN_FEATS_B0 {
                dblk.w_in_z[f][dst] *= z_slope;
            }
            for s in 0..MAX_BLOCK_LANES {
                dblk.w_prev_z[dst][s] *= z_slope;
                dblk.w_rec_z[dst][s] *= z_slope;
            }

            if lane_w_r_max(sblk, dst) > 6.0 {
                if sblk.b_n[dst].abs() > 2.0 {
                    let ab = sblk.b_n[dst].abs().clamp(2.0, 3.8);
                    let gap_2x = (1.0 - tanh_pade5_with_deriv(ab).0).max(1.0e-9);
                    let gap_1x = (mult * gap_2x).clamp(1.0e-6, 0.5);
                    let b_n_mag = inv_tanh_pade5_pos_f64(1.0 - gap_1x);
                    dblk.b_n[dst] = (sblk.b_n[dst].signum() * b_n_mag).clamp(-3.8, 3.8);
                } else {
                    for f in 0..NUM_IN_FEATS_B0 {
                        dblk.w_in_n[f][dst] *= mult;
                    }
                    for s in 0..MAX_BLOCK_LANES {
                        dblk.w_prev_n[dst][s] *= mult;
                        dblk.w_rec_n[dst][s] *= mult;
                    }
                }
            } else {
                let br_c = sblk.b_r[dst].clamp(-7.90, 7.90);
                let r2x = sigmoid_pade5_with_deriv(br_c).0;
                let r1x = (r2x * mult).clamp(1.0e-4, 0.995);
                dblk.b_r[dst] = inv_sigmoid_pade5_f64(r1x).clamp(-7.90, 7.90);
                let r_slope = ((1.0 - r2x) / (1.0 - r1x)).clamp(0.5, 2.5);
                for f in 0..NUM_IN_FEATS_B0 {
                    dblk.w_in_r[f][dst] *= r_slope;
                }
                for s in 0..MAX_BLOCK_LANES {
                    dblk.w_prev_r[dst][s] *= r_slope;
                    dblk.w_rec_r[dst][s] *= r_slope;
                }
            }
        }
        dblk.enforce_lower_triangular();
    }
    out.mirror_mono();
    out
}

/// Accumulates the exact reverse-mode Jacobian pullback `grad_2x += J^T * grad_1x`.
pub(super) fn pullback_grad_1x_to_2x(
    src_2x: &V3MatryoshkaOptF64,
    grad_1x: &V3MatryoshkaOptF64,
    grad_2x: &mut V3MatryoshkaOptF64,
) {
    let w_max = f64::from(SVF_W_MAX);
    let (_, _, d_bg) = rescale_w_with_deriv(src_2x.svf_base_g, 1.0e-4, w_max, w_max);
    let (gmin_1x, _, d_gmin) = rescale_w_with_deriv(src_2x.svf_g_min, 1.0e-4, w_max, w_max);
    let (_, raw_gmax, d_raw_gmax) = rescale_w_with_deriv(src_2x.svf_g_max, 1.0e-4, w_max, w_max);

    grad_2x.svf_base_g += d_bg * grad_1x.svf_base_g;
    grad_2x.svf_g_min += d_gmin * grad_1x.svf_g_min;
    if raw_gmax > gmin_1x && raw_gmax < w_max {
        grad_2x.svf_g_max += d_raw_gmax * grad_1x.svf_g_max;
    } else if raw_gmax <= gmin_1x {
        grad_2x.svf_g_min += d_gmin * grad_1x.svf_g_max;
    }
    grad_2x.svf_det_lp_pole +=
        rescale_pole_sq_with_deriv(src_2x.svf_det_lp_pole, 0.05, 0.995).1 * grad_1x.svf_det_lp_pole;
    grad_2x.svf_atk_alpha +=
        rescale_pole_sq_with_deriv(src_2x.svf_atk_alpha, 0.001, 0.9999).1 * grad_1x.svf_atk_alpha;
    grad_2x.svf_rel_alpha +=
        rescale_pole_sq_with_deriv(src_2x.svf_rel_alpha, 0.001, 0.99995).1 * grad_1x.svf_rel_alpha;

    grad_2x.svf_sec2_g_ratio += grad_1x.svf_sec2_g_ratio;
    grad_2x.svf_env_oct_sens += grad_1x.svf_env_oct_sens;
    grad_2x.svf_env_comp += grad_1x.svf_env_comp;
    grad_2x.svf_det_hp_mix += grad_1x.svf_det_hp_mix;
    grad_2x.svf_damp_k1 += grad_1x.svf_damp_k1;
    grad_2x.svf_damp_k2 += grad_1x.svf_damp_k2;
    grad_2x.svf_sec2_bp_feed += grad_1x.svf_sec2_bp_feed;
    grad_2x.svf_sec2_hp_feed += grad_1x.svf_sec2_hp_feed;
    grad_2x.svf_stereo_link += grad_1x.svf_stereo_link;
    for i in 0..7 {
        grad_2x.svf_stage2_feed_taps[i] += grad_1x.svf_stage2_feed_taps[i];
    }

    grad_2x.feat_in_hp_beta += rescale_feat_beta_with_deriv(src_2x.feat_in_hp_beta, 1.0e-4, 0.50).1
        * grad_1x.feat_in_hp_beta;
    grad_2x.feat_fast_atk_beta +=
        rescale_feat_beta_with_deriv(src_2x.feat_fast_atk_beta, 1.0e-4, 0.999).1
            * grad_1x.feat_fast_atk_beta;
    grad_2x.feat_fast_rel_beta +=
        rescale_feat_beta_with_deriv(src_2x.feat_fast_rel_beta, 1.0e-5, 0.999).1
            * grad_1x.feat_fast_rel_beta;
    grad_2x.feat_slow_atk_beta +=
        rescale_feat_beta_with_deriv(src_2x.feat_slow_atk_beta, 1.0e-5, 0.50).1
            * grad_1x.feat_slow_atk_beta;
    grad_2x.feat_slow_rel_beta +=
        rescale_feat_beta_with_deriv(src_2x.feat_slow_rel_beta, 1.0e-6, 0.10).1
            * grad_1x.feat_slow_rel_beta;
    grad_2x.feat_pinch_k += grad_1x.feat_pinch_k;
    grad_2x.feat_slow_det_comp += grad_1x.feat_slow_det_comp;
    grad_2x.svf_feat_lp2_mix += grad_1x.svf_feat_lp2_mix;
    grad_2x.feat_key_link += grad_1x.feat_key_link;
    grad_2x.feat_det_thresh += grad_1x.feat_det_thresh;

    for b in 0..NUM_GRU_BLOCKS {
        let sb = &src_2x.blocks[b];
        let g1 = &grad_1x.blocks[b];
        let g2 = &mut grad_2x.blocks[b];
        for dst in 0..MAX_BLOCK_LANES {
            g2.k_sag[dst] += g1.k_sag[dst];
            g2.k_quad[dst] += g1.k_quad[dst];

            let bz_c = sb.b_z[dst].clamp(-7.90, 7.90);
            let (z2x, dz2x_dbz) = sigmoid_pade5_with_deriv(bz_c);
            let mult = 2.0 - z2x;
            let z1x_raw = z2x * mult;
            let z1x = z1x_raw.clamp(1.0e-6, 0.9997);
            let bz1x_raw = inv_sigmoid_pade5_f64(z1x);
            let z_slope = 2.0 / mult;

            let mut d_z_slope = 0.0_f64;
            for f in 0..NUM_IN_FEATS_B0 {
                g2.w_in_z[f][dst] += g1.w_in_z[f][dst] * z_slope;
                d_z_slope += g1.w_in_z[f][dst] * sb.w_in_z[f][dst];
            }
            let s_end = ((dst >> 2) + 1) << 2;
            for s in 0..s_end {
                g2.w_prev_z[dst][s] += g1.w_prev_z[dst][s] * z_slope;
                d_z_slope += g1.w_prev_z[dst][s] * sb.w_prev_z[dst][s];
                g2.w_rec_z[dst][s] += g1.w_rec_z[dst][s] * z_slope;
                d_z_slope += g1.w_rec_z[dst][s] * sb.w_rec_z[dst][s];
            }

            let mut d_mult = (-2.0 / (mult * mult)) * d_z_slope;
            if lane_w_r_max(sb, dst) > 6.0 {
                g2.b_r[dst] += g1.b_r[dst];
                for f in 0..NUM_IN_FEATS_B0 {
                    g2.w_in_r[f][dst] += g1.w_in_r[f][dst];
                }
                for s in 0..s_end {
                    g2.w_prev_r[dst][s] += g1.w_prev_r[dst][s];
                    g2.w_rec_r[dst][s] += g1.w_rec_r[dst][s];
                }
                if sb.b_n[dst].abs() > 2.0 {
                    let sgn = sb.b_n[dst].signum();
                    let ab_raw = sb.b_n[dst].abs();
                    let ab = ab_raw.clamp(2.0, 3.8);
                    let (th2x, dth2x) = tanh_pade5_with_deriv(ab);
                    let gap_2x_raw = 1.0 - th2x;
                    let gap_2x = gap_2x_raw.max(1.0e-9);
                    let gap_1x_raw = mult * gap_2x;
                    let gap_1x = gap_1x_raw.clamp(1.0e-6, 0.5);
                    let bn_mag_raw = inv_tanh_pade5_pos_f64(1.0 - gap_1x);
                    let bn1x_raw = sgn * bn_mag_raw;
                    let d_bn_mag = if bn1x_raw > -3.8 && bn1x_raw < 3.8 {
                        sgn * g1.b_n[dst]
                    } else {
                        0.0
                    };
                    let dth1 = tanh_pade5_with_deriv(bn_mag_raw).1;
                    let d_gap1x = if gap_1x_raw > 1.0e-6 && gap_1x_raw < 0.5 && dth1 > 1.0e-12 {
                        -d_bn_mag / dth1
                    } else {
                        0.0
                    };
                    d_mult += gap_2x * d_gap1x;
                    if ab_raw > 2.0 && ab_raw < 3.8 && gap_2x_raw > 1.0e-9 {
                        g2.b_n[dst] += sgn * (-dth2x) * mult * d_gap1x;
                    }
                    for f in 0..NUM_IN_FEATS_B0 {
                        g2.w_in_n[f][dst] += g1.w_in_n[f][dst];
                    }
                    for s in 0..s_end {
                        g2.w_prev_n[dst][s] += g1.w_prev_n[dst][s];
                        g2.w_rec_n[dst][s] += g1.w_rec_n[dst][s];
                    }
                } else {
                    g2.b_n[dst] += g1.b_n[dst];
                    for f in 0..NUM_IN_FEATS_B0 {
                        g2.w_in_n[f][dst] += g1.w_in_n[f][dst] * mult;
                        d_mult += g1.w_in_n[f][dst] * sb.w_in_n[f][dst];
                    }
                    for s in 0..s_end {
                        g2.w_prev_n[dst][s] += g1.w_prev_n[dst][s] * mult;
                        d_mult += g1.w_prev_n[dst][s] * sb.w_prev_n[dst][s];
                        g2.w_rec_n[dst][s] += g1.w_rec_n[dst][s] * mult;
                        d_mult += g1.w_rec_n[dst][s] * sb.w_rec_n[dst][s];
                    }
                }
            } else {
                g2.b_n[dst] += g1.b_n[dst];
                for f in 0..NUM_IN_FEATS_B0 {
                    g2.w_in_n[f][dst] += g1.w_in_n[f][dst];
                }
                for s in 0..s_end {
                    g2.w_prev_n[dst][s] += g1.w_prev_n[dst][s];
                    g2.w_rec_n[dst][s] += g1.w_rec_n[dst][s];
                }

                let br_c = sb.b_r[dst].clamp(-7.90, 7.90);
                let (r2x, dr2x_dbr) = sigmoid_pade5_with_deriv(br_c);
                let r1x_raw = r2x * mult;
                let r1x = r1x_raw.clamp(1.0e-4, 0.995);
                let br1x_raw = inv_sigmoid_pade5_f64(r1x);
                let r_slope_raw = (1.0 - r2x) / (1.0 - r1x);
                let r_slope = r_slope_raw.clamp(0.5, 2.5);

                let mut d_r_slope = 0.0_f64;
                for f in 0..NUM_IN_FEATS_B0 {
                    g2.w_in_r[f][dst] += g1.w_in_r[f][dst] * r_slope;
                    d_r_slope += g1.w_in_r[f][dst] * sb.w_in_r[f][dst];
                }
                for s in 0..s_end {
                    g2.w_prev_r[dst][s] += g1.w_prev_r[dst][s] * r_slope;
                    d_r_slope += g1.w_prev_r[dst][s] * sb.w_prev_r[dst][s];
                    g2.w_rec_r[dst][s] += g1.w_rec_r[dst][s] * r_slope;
                    d_r_slope += g1.w_rec_r[dst][s] * sb.w_rec_r[dst][s];
                }

                let d_r_slope_raw = if r_slope_raw > 0.5 && r_slope_raw < 2.5 {
                    d_r_slope
                } else {
                    0.0
                };
                let om_r1x = 1.0 - r1x;
                let mut d_r2x = (-1.0 / om_r1x) * d_r_slope_raw;
                let mut d_r1x = ((1.0 - r2x) / (om_r1x * om_r1x)) * d_r_slope_raw;
                if br1x_raw > -7.90 && br1x_raw < 7.90 {
                    let dsig_r1 = sigmoid_pade5_with_deriv(br1x_raw).1;
                    if dsig_r1 > 1.0e-12 {
                        d_r1x += g1.b_r[dst] / dsig_r1;
                    }
                }
                if r1x_raw > 1.0e-4 && r1x_raw < 0.995 {
                    d_r2x += mult * d_r1x;
                    d_mult += r2x * d_r1x;
                }
                if sb.b_r[dst] > -7.90 && sb.b_r[dst] < 7.90 {
                    g2.b_r[dst] += dr2x_dbr * d_r2x;
                }
            }

            let mut d_z2x = -d_mult;
            if z1x_raw > 1.0e-6 && z1x_raw < 0.9997 {
                let dsig1 = sigmoid_pade5_with_deriv(bz1x_raw).1;
                let d_z1x = if bz1x_raw > -7.90 && bz1x_raw < 7.90 && dsig1 > 1.0e-12 {
                    g1.b_z[dst] / dsig1
                } else {
                    0.0
                };
                d_z2x += 2.0 * (1.0 - z2x) * d_z1x;
            }
            if src_2x.blocks[b].b_z[dst] > -7.90 && src_2x.blocks[b].b_z[dst] < 7.90 {
                g2.b_z[dst] += dz2x_dbz * d_z2x;
            }
        }
        g2.enforce_lower_triangular();
    }

    for (dst, src) in grad_2x.kautz_poles.iter_mut().zip(&grad_1x.kautz_poles) {
        dst.0 += src.0;
        dst.1 += src.1;
    }
    for tau in 0..4 {
        grad_2x.readouts[tau].add_assign(&grad_1x.readouts[tau]);
    }
    grad_2x.mirror_mono();
}

/// Calibrates dedicated `1x` (`48 kHz`) front-end/Stage II parameters and `[Lite, Full]` readouts
/// starting from the Padé-exact ZOH-rescaled `2x` backbone with shared `kautz_poles` held fixed.
#[inline(never)]
pub(in crate::solver::v3_varpro) fn calibrate_1x_tiers_with_backend<B: TrainerBackend>(
    model: &mut V3MatryoshkaOptF64,
    ds_1x: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
) {
    if ds_1x.stems.is_empty() {
        return;
    }
    let mut m_start = Box::new(rescale_model_2x_to_1x_f64(model));
    m_start.os2x = false;
    m_start.tier_os2x = [false; 4];
    m_start.rate_1x_cal = None;
    m_start.max_active_groups = 2;
    m_start.active_tier_weights = [0.45, 0.55, 0.0, 0.0];
    m_start.kautz_poles = model.kautz_poles;

    let pre_vp = evaluate_matryoshka_loss_f64_with_backend::<B>(&m_start, ds_1x, ws);
    let mut base_loss = if pre_vp.is_finite() {
        pre_vp
    } else {
        f64::INFINITY
    };
    let mut m_vp = m_start.clone();
    solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut m_vp, ds_1x, 1);
    let post_vp = evaluate_matryoshka_loss_f64_with_backend::<B>(&m_vp, ds_1x, ws);
    if post_vp.is_finite() && post_vp <= base_loss {
        m_start = m_vp;
        base_loss = post_vp;
    }

    if m_start.svf_env_oct_sens.abs() < 0.25 && m_start.svf_feat_lp2_mix < 0.5 {
        if m_start.svf_base_g > 0.80 {
            let mut m_aa = m_start.clone();
            m_aa.svf_base_g = 0.58;
            m_aa.svf_g_max = m_aa.svf_g_max.max(0.95);
            solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut m_aa, ds_1x, 1);
            let aa_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&m_aa, ds_1x, ws);
            if aa_loss.is_finite() && aa_loss < base_loss {
                m_start = m_aa;
                base_loss = aa_loss;
            }
        }
        let mut m_lp2 = m_start.clone();
        m_lp2.svf_base_g = 0.56;
        m_lp2.svf_damp_k1 = std::f64::consts::SQRT_2;
        m_lp2.svf_damp_k2 = std::f64::consts::SQRT_2;
        m_lp2.svf_g_max = m_lp2.svf_g_max.max(0.95);
        m_lp2.svf_stage2_feed_taps = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut m_lp2, ds_1x, 1);
        let lp2_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&m_lp2, ds_1x, ws);
        if lp2_loss.is_finite() && lp2_loss < base_loss {
            m_start = m_lp2;
            base_loss = lp2_loss;
        }
    }

    let mut m_opt = m_start.clone();
    let stem0_len = ds_1x.stems.first().map_or(0, |s| s.usable_len());
    let lbfgs_iters = if stem0_len < 4_096 {
        5
    } else if stem0_len >= 24_576 {
        10
    } else {
        7
    };
    let mut opt_loss =
        run_lbfgs_with_ws::<B>(&mut m_opt, ds_1x, ws, lbfgs_iters, true, true, false);
    if !opt_loss.is_finite() {
        opt_loss = f64::INFINITY;
    }
    let backup = m_opt.clone();
    solve_matryoshka_varpro_cholesky_with_backend::<B>(&mut m_opt, ds_1x, 1);
    let next_loss = evaluate_matryoshka_loss_f64_with_backend::<B>(&m_opt, ds_1x, ws);
    if next_loss.is_finite() && next_loss <= opt_loss {
        opt_loss = next_loss;
    } else {
        m_opt = backup;
    }

    let mut winner = if opt_loss.is_finite() && (!base_loss.is_finite() || opt_loss <= base_loss) {
        m_opt
    } else {
        m_start
    };
    winner.mirror_mono();

    let cold_1x = Box::new(winner.to_matryoshka_cold());
    model.rate_1x_cal = Some(V3Rate1xCalibration {
        svf: cold_1x.svf,
        feat_params: cold_1x.feat_params,
        blocks: cold_1x.blocks,
    });
    model.readouts[0] = winner.readouts[0].clone();
    model.readouts[1] = winner.readouts[1].clone();
    model.tier_os2x = [false, false, true, true];
    model.mirror_mono();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    use dnam_engine::Tier;
    use dnam_engine::stages::{V3ColdTierModel, V3MatryoshkaModel};

    use super::super::super::model_f64::{V3TrainDataset, V3TrainStem};
    use super::super::super::seed_init::build_matryoshka_seed_bank;
    use super::super::{evaluate_matryoshka_loss_and_grad, evaluate_matryoshka_loss_f64};

    fn assert_blocks_close(a: &V3ColdTierModel, b: &V3ColdTierModel) {
        assert!((a.svf.base_g - b.svf.base_g).abs() < 1.0e-5);
        assert!((a.svf.g_min - b.svf.g_min).abs() < 1.0e-5);
        assert!((a.svf.g_max - b.svf.g_max).abs() < 1.0e-5);
        assert!((a.feat_params.in_hp_beta - b.feat_params.in_hp_beta).abs() < 1.0e-6);
        assert!((a.feat_params.fast_atk_beta - b.feat_params.fast_atk_beta).abs() < 1.0e-6);
        for blk_i in 0..NUM_GRU_BLOCKS {
            for l in 0..MAX_BLOCK_LANES {
                assert!((a.blocks[blk_i].b_z[l] - b.blocks[blk_i].b_z[l]).abs() < 1.0e-4);
                assert!((a.blocks[blk_i].b_r[l] - b.blocks[blk_i].b_r[l]).abs() < 1.0e-4);
                assert!((a.blocks[blk_i].b_n[l] - b.blocks[blk_i].b_n[l]).abs() < 1.0e-4);
                assert!(
                    (a.blocks[blk_i].w_in_n[0][l] - b.blocks[blk_i].w_in_n[0][l]).abs() < 1.0e-4
                );
            }
        }
    }

    #[test]
    fn test_rescale_2x_to_1x_f64_matches_cold_extract() {
        let seeds = build_matryoshka_seed_bank(true);
        for idx in [0_usize, 4, 9, 13] {
            let mut s = seeds[idx].clone();
            s.tier_os2x = [false, false, true, true];
            let c_ext = s.to_matryoshka_cold().extract_tier(Tier::Lite);
            let r_f64 = rescale_model_2x_to_1x_f64(&s)
                .to_matryoshka_cold()
                .extract_tier(Tier::Lite);
            assert_blocks_close(&c_ext, &r_f64);
        }
    }

    #[test]
    fn test_dual_rate_pullback_grad_matches_central_finite_diff() {
        let n = 384_usize;
        let (mut in_l, mut in_r, mut tgt_l, mut tgt_r) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let t = (i as f64) / 48_000.0;
            let xl = 0.35 * (2.0 * PI * 220.0 * t).sin() + 0.18 * (2.0 * PI * 3_100.0 * t).cos();
            let xr = 0.30 * (2.0 * PI * 310.0 * t + 0.4).sin();
            in_l.push(xl as f32);
            in_r.push(xr as f32);
            tgt_l.push((0.8 * xl + 0.25 * (xl * 2.5).tanh()) as f32);
            tgt_r.push((0.75 * xr + 0.2 * (xr * 2.2).tanh()) as f32);
        }
        let ds = V3TrainDataset::new(
            vec![V3TrainStem {
                in_l,
                in_r,
                tgt_l,
                tgt_r,
                warmup: 48,
            }],
            48_000.0,
        )
        .with_spectral_weights(0.15, 0.10, 0.0);

        let mut seeds = build_matryoshka_seed_bank(false);
        let mut m = seeds.swap_remove(4); // SeriesTubeSag seed (exercises diode switch + LTI poles)
        m.tier_os2x = [false, false, true, true];
        m.max_active_groups = 3;
        m.active_tier_weights = [0.25, 0.25, 0.25, 0.25];
        m.blocks[0].w_in_r[0][2] = 0.4; // LTI r lane
        m.blocks[0].b_z[2] = -0.35;
        m.blocks[0].b_r[2] = -0.25;
        m.blocks[0].b_n[0] = 0.05;
        m.svf_base_g = 0.45;
        m.svf_g_max = 0.72;
        m.feat_in_hp_beta = 0.08;
        m.readouts[0].pre_cab_w[0][0] = 0.12;
        m.readouts[0].pre_cab_w[0][1] = 0.15;
        m.readouts[0].pre_cab_w[0][2] = 0.22;
        m.readouts[1] = m.readouts[0].clone();
        m.readouts[2] = m.readouts[0].clone();
        m.readouts[3] = m.readouts[0].clone();

        let (_, grad) = evaluate_matryoshka_loss_and_grad(&m, &ds);
        let mut check =
            |name: &str, eps: f64, ana: f64, get_set: fn(&mut V3MatryoshkaOptF64) -> &mut f64| {
                let orig = *get_set(&mut m);
                *get_set(&mut m) = orig + eps;
                let lp = evaluate_matryoshka_loss_f64(&m, &ds);
                *get_set(&mut m) = orig - eps;
                let lm = evaluate_matryoshka_loss_f64(&m, &ds);
                *get_set(&mut m) = orig;
                let fd = (lp - lm) / (2.0 * eps);
                let rel = (ana - fd).abs() / ana.abs().max(fd.abs()).max(1.0e-4);
                assert!(rel < 1.0e-4, "{name}: ana={ana} fd={fd} rel={rel}");
            };
        check("svf_base_g", 1e-6, grad.svf_base_g, |x| &mut x.svf_base_g);
        check("svf_g_max", 1e-6, grad.svf_g_max, |x| &mut x.svf_g_max);
        check("svf_det_lp_pole", 1e-6, grad.svf_det_lp_pole, |x| {
            &mut x.svf_det_lp_pole
        });
        check("feat_in_hp_beta", 1e-6, grad.feat_in_hp_beta, |x| {
            &mut x.feat_in_hp_beta
        });
        check("b0.b_z[0] (diode w_n)", 1e-6, grad.blocks[0].b_z[0], |x| {
            &mut x.blocks[0].b_z[0]
        });
        check(
            "b0.b_z[1] (diode b_n>2)",
            1e-6,
            grad.blocks[0].b_z[1],
            |x| &mut x.blocks[0].b_z[1],
        );
        check(
            "b0.b_n[1] (diode b_n>2)",
            1e-6,
            grad.blocks[0].b_n[1],
            |x| &mut x.blocks[0].b_n[1],
        );
        check("b0.w_in_n[0][0]", 1e-6, grad.blocks[0].w_in_n[0][0], |x| {
            &mut x.blocks[0].w_in_n[0][0]
        });
        check("b0.b_z[2] (LTI)", 1e-6, grad.blocks[0].b_z[2], |x| {
            &mut x.blocks[0].b_z[2]
        });
        check("b0.b_r[2] (LTI)", 1e-6, grad.blocks[0].b_r[2], |x| {
            &mut x.blocks[0].b_r[2]
        });
        check(
            "b0.w_in_r[0][2] (LTI)",
            1e-6,
            grad.blocks[0].w_in_r[0][2],
            |x| &mut x.blocks[0].w_in_r[0][2],
        );
        check(
            "r0.fir_l[0] (1x tier)",
            1e-6,
            grad.readouts[0].fir_l[0],
            |x| &mut x.readouts[0].fir_l[0],
        );
        check(
            "r2.fir_l[0] (2x tier)",
            1e-6,
            grad.readouts[2].fir_l[0],
            |x| &mut x.readouts[2].fir_l[0],
        );

        let poles_before = m.kautz_poles;
        let mut cal_ws = AdjointWorkspace::default();
        calibrate_1x_tiers_with_backend::<crate::solver::backends::ScalarTrainerBackend>(
            &mut m,
            &ds,
            &mut cal_ws,
        );
        assert!(m.rate_1x_cal.is_some());
        assert_eq!(m.tier_os2x, [false, false, true, true]);
        assert_eq!(m.kautz_poles, poles_before);
        let _ = V3MatryoshkaModel::dual_rate_default(false);
    }
}
