#![forbid(unsafe_code)]
//! Causal Bézout 3-band ESR and `20 ms` 3-band Bézout RMS energy-envelope `SegESR` hinge loss &
//! adjoint (`incoherent_rig/wave_hinge.rs`).
//!
//! Supplies a weak waveform attraction term on `Ultra` to break LFO/comb phase degeneracy and
//! one-sided lower-bound hinges (`>= hi + margin`) on `[Pro, Full, Lite]` to guarantee strict
//! monotone ordering during top-down Welch-33 EQ polishing.

use crate::metrics::MAX_METRIC_SAMPLE_RATE_HZ;
use crate::solver::v3_varpro::bezout7::Bezout7FilterBank;

#[derive(Clone, Copy)]
pub(super) struct SegEnvFrame {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) w_f: f64,
    pub(super) y_sq: [f64; 3],
    pub(super) y_rms: [f64; 3],
}

/// Precomputes active `20 ms` (`L = round(fs / 50)`) post-warmup 3-band Bézout RMS energy frames
/// matching `metrics::companion::compute_multirate_seg_esr_db` for zero-noise deterministic targets.
pub(super) fn build_seg_env_frames(
    bezout: Bezout7FilterBank,
    tgt: &[f64],
    warmup: usize,
    sr_hz: f64,
) -> Vec<SegEnvFrame> {
    let n = tgt.len();
    if n <= warmup {
        return Vec::new();
    }
    let fs = sr_hz.clamp(8_000.0, MAX_METRIC_SAMPLE_RATE_HZ);
    let full_len = (fs / 50.0).round().max(1.0) as usize;
    let tgt_post = &tgt[warmup..];
    let n_post = tgt_post.len();
    let mut t_bands: [Vec<f64>; 3] = Default::default();
    bezout.split_3band_and_energies(tgt_post, n_post, &mut t_bands);

    let mut frames = Vec::new();
    let mut start = 0_usize;
    while start < n_post {
        let end = (start + full_len).min(n_post);
        let e_sig: f64 = tgt_post[start..end].iter().map(|&v| v * v).sum();
        if e_sig > 0.0 {
            let inv_nf = 1.0 / ((end - start) as f64);
            let mut y_sq = [0.0_f64; 3];
            let mut y_rms = [0.0_f64; 3];
            for b in 0..3 {
                let sq = t_bands[b][start..end].iter().map(|&v| v * v).sum::<f64>() * inv_nf;
                y_sq[b] = sq;
                y_rms[b] = sq.sqrt();
            }
            frames.push(SegEnvFrame {
                start,
                end,
                w_f: ((end - start) as f64) / fs,
                y_sq,
                y_rms,
            });
        }
        start = end;
    }
    frames
}

pub(super) struct WaveFloorSpec<'a> {
    pub(super) bezout: Bezout7FilterBank,
    pub(super) tgt: &'a [f64],
    pub(super) tgt_band_e: [f64; 3],
    pub(super) seg_frames: &'a [SegEnvFrame],
    pub(super) pull_weight: f64,
    pub(super) min_band_esr_lin: Option<[f64; 3]>,
    pub(super) min_raw_besr_lin: Option<f64>,
    pub(super) min_seg_esr_lin: Option<f64>,
}

pub(super) struct WaveTargetCtx {
    bezout: Bezout7FilterBank,
    tl_full: Vec<f64>,
    tr_full: Vec<f64>,
    tgt_band_e_l: [f64; 3],
    tgt_band_e_r: [f64; 3],
    seg_frames_l: Vec<SegEnvFrame>,
    seg_frames_r: Vec<SegEnvFrame>,
}

impl WaveTargetCtx {
    pub(super) fn new(tl_full: Vec<f64>, tr_full: Vec<f64>, warmup: usize, sr_hz: f64) -> Self {
        let bezout = Bezout7FilterBank::new(sr_hz);
        let tgt_band_e_l = bezout.band_energies(&tl_full, warmup);
        let tgt_band_e_r = bezout.band_energies(&tr_full, warmup);
        let seg_frames_l = build_seg_env_frames(bezout, &tl_full, warmup, sr_hz);
        let seg_frames_r = build_seg_env_frames(bezout, &tr_full, warmup, sr_hz);
        Self {
            bezout,
            tl_full,
            tr_full,
            tgt_band_e_l,
            tgt_band_e_r,
            seg_frames_l,
            seg_frames_r,
        }
    }

    pub(super) fn make_specs(
        &self,
        pull_weight: f64,
        lo_l: Option<([f64; 3], f64, f64)>,
        lo_r: Option<([f64; 3], f64, f64)>,
    ) -> (WaveFloorSpec<'_>, WaveFloorSpec<'_>) {
        let spec_l = WaveFloorSpec {
            bezout: self.bezout,
            tgt: &self.tl_full,
            tgt_band_e: self.tgt_band_e_l,
            seg_frames: &self.seg_frames_l,
            pull_weight,
            min_band_esr_lin: lo_l.map(|(b, _, _)| b),
            min_raw_besr_lin: lo_l.map(|(_, r, _)| r),
            min_seg_esr_lin: lo_l.map(|(_, _, s)| s),
        };
        let spec_r = WaveFloorSpec {
            bezout: self.bezout,
            tgt: &self.tr_full,
            tgt_band_e: self.tgt_band_e_r,
            seg_frames: &self.seg_frames_r,
            pull_weight,
            min_band_esr_lin: lo_r.map(|(b, _, _)| b),
            min_raw_besr_lin: lo_r.map(|(_, r, _)| r),
            min_seg_esr_lin: lo_r.map(|(_, _, s)| s),
        };
        (spec_l, spec_r)
    }
}

#[derive(Default)]
pub(super) struct WaveHingeScratch {
    err: Vec<f64>,
    bands: [Vec<f64>; 3],
    m_bands: [Vec<f64>; 3],
    d_m_bands: [Vec<f64>; 3],
    frame_m_sq: Vec<[f64; 3]>,
}

fn eval_seg_env_hinge_and_adjoint(
    spec: &WaveFloorSpec<'_>,
    y_post: &[f64],
    g_seg: f64,
    dy_post: Option<&mut [f64]>,
    sc: &mut WaveHingeScratch,
) -> f64 {
    let n_post = y_post.len();
    if n_post == 0 || spec.seg_frames.is_empty() {
        return 0.0;
    }
    let _ = spec
        .bezout
        .split_3band_and_energies(y_post, n_post, &mut sc.m_bands);

    sc.frame_m_sq.clear();
    let mut weighted_lin_sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for f in spec.seg_frames {
        if f.end > n_post {
            break;
        }
        let inv_nf = 1.0 / ((f.end - f.start) as f64);
        let mut m_sq = [0.0_f64; 3];
        let mut r_sum = 0.0_f64;
        for b in 0..3 {
            let sq = sc.m_bands[b][f.start..f.end]
                .iter()
                .map(|&v| v * v)
                .sum::<f64>()
                * inv_nf;
            m_sq[b] = sq;
            let r_b = if f.y_sq[b] > 0.0 {
                let d_rms = sq.sqrt() - f.y_rms[b];
                (d_rms * d_rms) / f.y_sq[b]
            } else if sq == 0.0 {
                0.0
            } else {
                1.0
            };
            r_sum += r_b;
        }
        sc.frame_m_sq.push(m_sq);
        weighted_lin_sum += f.w_f * (r_sum / 3.0);
        weight_sum += f.w_f;
    }
    if weight_sum <= 0.0 {
        return 0.0;
    }

    let r_seg = weighted_lin_sum / weight_sum;
    let g = g_seg.max(1.0e-15);
    if !(r_seg < g && r_seg > 1.0e-24) {
        return 0.0;
    }

    let deficit = (g - r_seg) / g;
    if let Some(dy_buf) = dy_post {
        let d_loss_d_rseg = -192.0 * deficit / g;
        for b in 0..3 {
            sc.d_m_bands[b].fill(0.0);
            sc.d_m_bands[b].resize(n_post, 0.0);
        }
        for (idx, f) in spec.seg_frames.iter().enumerate() {
            if f.end > n_post {
                break;
            }
            let n_f = (f.end - f.start) as f64;
            let scale_base = d_loss_d_rseg * (f.w_f / (weight_sum * 3.0)) * (2.0 / n_f);
            for b in 0..3 {
                let m_sq_b = sc.frame_m_sq[idx][b];
                if f.y_sq[b] > 0.0 && m_sq_b > 1.0e-30 {
                    let m_rms = m_sq_b.sqrt();
                    let coeff = scale_base * (m_rms - f.y_rms[b]) / (f.y_sq[b] * m_rms);
                    for t in f.start..f.end {
                        sc.d_m_bands[b][t] += coeff * sc.m_bands[b][t];
                    }
                }
            }
        }
        spec.bezout.accum_3band_band_grads(&sc.d_m_bands, dy_buf);
    }
    96.0 * deficit * deficit
}

pub(super) fn eval_wave_hinge_and_adjoint(
    spec: &WaveFloorSpec<'_>,
    y_hat: &[f64],
    warmup: usize,
    mut dy: Option<&mut [f64]>,
    sc: &mut WaveHingeScratch,
) -> f64 {
    let n = y_hat
        .len()
        .min(spec.tgt.len())
        .min(dy.as_ref().map_or(usize::MAX, |d| d.len()));
    if n <= warmup {
        return 0.0;
    }
    sc.err.resize(n, 0.0);
    for i in 0..n {
        sc.err[i] = y_hat[i] - spec.tgt[i];
    }

    let err_band_e = if dy.is_some() {
        spec.bezout
            .split_3band_and_energies(&sc.err[..n], warmup, &mut sc.bands)
    } else {
        spec.bezout.band_energies(&sc.err[..n], warmup)
    };
    let mut loss = 0.0_f64;
    let mut bw = [0.0_f64; 3];
    let mut esr_bands = [0.0_f64; 3];
    for b in 0..3 {
        let tgt_e = spec.tgt_band_e[b].max(1.0e-18);
        let esr_b = err_band_e[b] / tgt_e;
        esr_bands[b] = esr_b;
        if spec.pull_weight > 0.0 {
            loss += spec.pull_weight * esr_b / 3.0;
            bw[b] += spec.pull_weight / (3.0 * tgt_e);
        }
        if let Some(g_bands) = spec.min_band_esr_lin {
            let g = g_bands[b].max(1.0e-15);
            if esr_b < g && err_band_e[b] > 1.0e-24 {
                let deficit = (g - esr_b) / g;
                loss += 96.0 * deficit * deficit;
                bw[b] += -192.0 * deficit / (g * tgt_e);
            }
        }
    }
    if let Some(g_raw) = spec.min_raw_besr_lin {
        let g = g_raw.max(1.0e-15);
        let raw_esr = (esr_bands[0] + esr_bands[1] + esr_bands[2]) / 3.0;
        if raw_esr < g && raw_esr > 1.0e-24 {
            let deficit = (g - raw_esr) / g;
            loss += 96.0 * deficit * deficit;
            let d_raw = -192.0 * deficit / (3.0 * g);
            for b in 0..3 {
                bw[b] += d_raw / spec.tgt_band_e[b].max(1.0e-18);
            }
        }
    }

    if let Some(dy_buf) = dy.as_deref_mut()
        && bw.iter().any(|&w| w != 0.0)
    {
        spec.bezout
            .accum_3band_adjoint(&sc.bands, warmup, bw, &mut dy_buf[..n]);
    }

    if let Some(g_seg) = spec.min_seg_esr_lin {
        loss += eval_seg_env_hinge_and_adjoint(
            spec,
            &y_hat[warmup..n],
            g_seg,
            dy.map(|buf| &mut buf[warmup..n]),
            sc,
        );
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{
        MultiRateSegEsrSlice, compute_multirate_seg_esr_db, evaluate_besr_from_band_energies,
        split_bezout_3band_energies,
    };
    use std::f64::consts::TAU;

    fn eval_seg_env_lin(bezout: Bezout7FilterBank, frames: &[SegEnvFrame], y_post: &[f64]) -> f64 {
        let mut m_bands: [Vec<f64>; 3] = Default::default();
        bezout.split_3band_and_energies(y_post, y_post.len(), &mut m_bands);
        let mut weighted_lin_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for f in frames {
            let inv_nf = 1.0 / ((f.end - f.start) as f64);
            let mut r_sum = 0.0_f64;
            for b in 0..3 {
                let sq = m_bands[b][f.start..f.end]
                    .iter()
                    .map(|&v| v * v)
                    .sum::<f64>()
                    * inv_nf;
                let d_rms = sq.sqrt() - f.y_rms[b];
                r_sum += (d_rms * d_rms) / f.y_sq[b];
            }
            weighted_lin_sum += f.w_f * (r_sum / 3.0);
            weight_sum += f.w_f;
        }
        weighted_lin_sum / weight_sum
    }

    #[test]
    fn test_wave_hinge_metric_parity_and_finite_diff_adjoint() {
        let n = 4_800;
        let warmup = 384;
        let sr = 48_000.0;
        let tgt: Vec<f64> = (0..n)
            .map(|i| 0.3 * (TAU * 190.0 * (i as f64) / sr).sin() + 0.05 * ((i % 7) as f64 - 3.0))
            .collect();
        let y_hat: Vec<f64> = tgt
            .iter()
            .enumerate()
            .map(|(i, &x)| x + 0.03 * (TAU * 910.0 * (i as f64) / sr).cos())
            .collect();

        let bezout = Bezout7FilterBank::new(sr);
        let seg_frames = build_seg_env_frames(bezout, &tgt, warmup, sr);
        let r_seg = eval_seg_env_lin(bezout, &seg_frames, &y_hat[warmup..]);
        let (seg_db, seg_cnt, seg_ok) = compute_multirate_seg_esr_db(&[MultiRateSegEsrSlice {
            sample_rate_hz: sr,
            target: &tgt[warmup..],
            model: &y_hat[warmup..],
            noise: None,
        }]);
        assert!(seg_ok);
        assert_eq!(seg_cnt, seg_frames.len());
        assert!(
            (10.0 * r_seg.log10() - seg_db).abs() < 1.0e-9,
            "local={} pub={seg_db}",
            10.0 * r_seg.log10()
        );

        let tgt_band_e = bezout.band_energies(&tgt, warmup);
        let err: Vec<f64> = y_hat.iter().zip(&tgt).map(|(a, b)| a - b).collect();
        let err_e = bezout.band_energies(&err, warmup);
        let (sig_e, rep_err_e, nse_e) = split_bezout_3band_energies(&tgt, &y_hat, None, sr, warmup);
        let rep = evaluate_besr_from_band_energies(sig_e, rep_err_e, nse_e);
        let esr_lin = [
            err_e[0] / tgt_band_e[0],
            err_e[1] / tgt_band_e[1],
            err_e[2] / tgt_band_e[2],
        ];
        for b in 0..3 {
            assert!((10.0 * esr_lin[b].log10() - rep.band_esr_db[b]).abs() < 1.0e-9);
        }

        // Verify analytical adjoint matches central finite differences when pull + both hinges fire.
        let spec = WaveFloorSpec {
            bezout,
            tgt: &tgt,
            tgt_band_e,
            seg_frames: &seg_frames,
            pull_weight: 0.025,
            min_band_esr_lin: Some(esr_lin.map(|e| e * 1.4)),
            min_raw_besr_lin: Some((esr_lin[0] + esr_lin[1] + esr_lin[2]) * 1.4 / 3.0),
            min_seg_esr_lin: Some(r_seg * 1.35),
        };
        let mut dy = vec![0.0_f64; n];
        let mut sc = WaveHingeScratch::default();
        let base_loss = eval_wave_hinge_and_adjoint(&spec, &y_hat, warmup, Some(&mut dy), &mut sc);
        assert!(base_loss > 0.0);

        let eps = 1.0e-6;
        for &idx in &[200_usize, 700, 2_100, 4_790] {
            let mut yp = y_hat.clone();
            let mut ym = y_hat.clone();
            yp[idx] += eps;
            ym[idx] -= eps;
            let lp = eval_wave_hinge_and_adjoint(&spec, &yp, warmup, None, &mut sc);
            let lm = eval_wave_hinge_and_adjoint(&spec, &ym, warmup, None, &mut sc);
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (fd - dy[idx]).abs() < 1.0e-5,
                "idx={idx}: fd={fd} vs adj={}",
                dy[idx]
            );
        }
    }
}
