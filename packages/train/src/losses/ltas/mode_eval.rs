//! Welch 33-band LTAS mode objectives and ultrasonic `[20 kHz, fs/2]` hinge derivatives
//! (`ref[qual.companion.ltas]`, `ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::LN_10;

use super::{LtasLossMode, Welch33TargetCache};

const DB_PER_LN: f64 = 10.0 / LN_10;

#[inline]
fn c2_hinge(u: f64, scale: f64, width: f64) -> (f64, f64) {
    if u <= 0.0 {
        return (0.0, 0.0);
    }
    let c = 0.25 * width * width;
    let u2 = u * u;
    let d = u2 + c;
    (scale * u2 * u / d, scale * u2 * (u2 + 3.0 * c) / (d * d))
}

#[inline]
fn pseudo_huber(s: f64, delta: f64) -> (f64, f64) {
    let r = s / delta;
    let root = (1.0 + r * r).sqrt();
    (delta * delta * (root - 1.0), s / root)
}

#[inline]
pub(super) fn nyquist_hinge_loss_and_deriv(
    cache: &Welch33TargetCache,
    nyq_p: f64,
    weight: f64,
) -> (f64, f64) {
    if weight <= 0.0 {
        return (0.0, 0.0);
    }
    let hf_ref = cache
        .tgt_nyq_p
        .max(0.5 * cache.tgt_band_p[32])
        .max(0.25 * cache.tgt_band_p[31]);
    let tot_tgt = cache.tgt_band_p.iter().sum::<f64>().max(cache.floor);
    let nyq_ceil = (2.2 * hf_ref + 2.0e-3 * tot_tgt).max(cache.floor);
    if nyq_p <= nyq_ceil {
        return (0.0, 0.0);
    }
    let u_db = DB_PER_LN * (nyq_p / nyq_ceil).ln();
    (
        weight * 2.0 * u_db * u_db,
        weight * 4.0 * u_db * DB_PER_LN / nyq_p,
    )
}

#[inline]
fn eval_ltas_band_error_and_snr(
    cache: &Welch33TargetCache,
    band_p: &[f64; 33],
    b: usize,
) -> (f64, f64, f64) {
    let pb = band_p[b] + cache.floor;
    let snr = cache.band_snr_w[b];
    let e = snr * DB_PER_LN * (pb.ln() - (cache.tgt_band_p[b] + cache.floor).ln());
    (pb, snr, e)
}

pub(super) fn eval_ltas_mode_terms(
    cache: &Welch33TargetCache,
    band_p: &[f64; 33],
    mode: LtasLossMode,
    need_grad: bool,
) -> (f64, [f64; 33]) {
    let mut d_band = [0.0_f64; 33];
    let inv_act = 1.0 / (cache.active_count.max(1) as f64);
    let loss = match mode {
        LtasLossMode::LogSumExpMinimax { beta, goal_db } => {
            let b_eff = beta.max(1.0);
            let mut errs = [0.0_f64; 33];
            let mut max_db = 0.0_f64;
            let mut mean_sq = 0.0_f64;
            for b in 0..33 {
                if cache.active[b] {
                    let (_, _, e) = eval_ltas_band_error_and_snr(cache, band_p, b);
                    errs[b] = e;
                    max_db = max_db.max(e.abs());
                    mean_sq += inv_act * e * e;
                }
            }
            let z0 = (-b_eff * max_db).exp();
            let (mut wp, mut wn, mut excess) = ([0.0_f64; 33], [0.0_f64; 33], 0.0_f64);
            for b in 0..33 {
                if cache.active[b] {
                    wp[b] = (b_eff * (errs[b] - max_db)).exp();
                    wn[b] = (b_eff * (-errs[b] - max_db)).exp();
                    excess += (wp[b] - z0) + (wn[b] - z0);
                }
            }
            let denom = (z0 + excess).max(1.0e-30);
            let f_lse = (max_db + denom.ln() / b_eff).max(0.0);
            let (sp, dsp) = if goal_db > 0.0 {
                let u = f_lse - goal_db;
                let r = (u * u + 1.0e-6).sqrt();
                (0.5 * (u + r), 0.5 * (1.0 + u / r))
            } else {
                (f_lse, 1.0)
            };
            if need_grad {
                let sl = if f_lse > 0.0 {
                    2.0 * sp * dsp + 0.5 * f_lse
                } else {
                    0.0
                };
                for b in 0..33 {
                    if cache.active[b] {
                        let (pb, snr, e) = eval_ltas_band_error_and_snr(cache, band_p, b);
                        d_band[b] = (sl * (wp[b] - wn[b]) / denom + 0.5 * inv_act * e)
                            * (snr * DB_PER_LN)
                            / pb;
                    }
                }
            }
            sp * sp + 0.25 * f_lse * f_lse + 0.25 * mean_sq
        }
        LtasLossMode::HingeMaxDb(tmax) => {
            let mut acc = 0.0_f64;
            let tau = (0.72 * tmax).max(0.015);
            let tw = (0.22 * tmax).max(0.010);
            let eps = 0.003_f64;
            for b in 0..33 {
                if !cache.active[b] {
                    continue;
                }
                let (pb, snr, e) = eval_ltas_band_error_and_snr(cache, band_p, b);
                let root = (e * e + eps * eps).sqrt();
                let u = (root - eps) - tau;
                let (hv, hd) = pseudo_huber(e, 1.5);
                let (gv, gd) = c2_hinge(u, 1.5, tw);
                acc += 0.10 * inv_act * hv + 2.0 * gv;
                if need_grad {
                    d_band[b] =
                        (0.10 * inv_act * hd + 2.0 * gd * (e / root)) * (snr * DB_PER_LN) / pb;
                }
            }
            acc
        }
        LtasLossMode::L2Huber => {
            let mut acc = 0.0_f64;
            for b in 0..33 {
                if !cache.active[b] {
                    continue;
                }
                let (pb, snr, e) = eval_ltas_band_error_and_snr(cache, band_p, b);
                let (hv, hd) = pseudo_huber(e, 2.0);
                acc += 2.0 * inv_act * hv;
                if need_grad {
                    d_band[b] = 2.0 * inv_act * hd * (snr * DB_PER_LN) / pb;
                }
            }
            acc
        }
    };
    (loss, d_band)
}
