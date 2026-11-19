#![forbid(unsafe_code)]
//! Out-of-band `[20 kHz, Nyquist]` stopband hinge loss & paired FFT adjoint (`ltas_adj/nyquist.rs`).
//!
//! Prevents Stage III FIR/Kautz polish from dumping unconstrained ultrasonic energy into
//! `(20 kHz, Nyquist]` to game broadband RMS before down-tier `snap_stage4_broadband_rms` scaling.

use std::f64::consts::LN_10;

use super::fft::Welch33TargetSummary;
#[cfg(test)]
use super::fft::{Welch33Scratch, accum_paired_backward_fft, accum_paired_forward_fft};

#[inline]
pub(crate) fn nyq_ceil_power(summary: &Welch33TargetSummary) -> f64 {
    let hf_ref = summary
        .tgt_nyq_p
        .max(0.5 * summary.tgt_band_p[32])
        .max(0.25 * summary.tgt_band_p[31]);
    let tot_tgt: f64 = summary.tgt_band_p.iter().sum::<f64>().max(summary.floor);
    (2.2 * hf_ref + 2.0e-3 * tot_tgt).max(summary.floor)
}

#[cfg(test)]
#[must_use]
pub(crate) fn compute_nyquist_stopband_power(summary: &Welch33TargetSummary, sig: &[f64]) -> f64 {
    let n_wins = summary.seg_starts.len();
    if n_wins == 0 || summary.seg_len == 0 || sig.len() < summary.warmup + summary.seg_len {
        return 0.0;
    }
    let mut fft_re = vec![0.0_f64; summary.plan.n_fft];
    let mut fft_im = vec![0.0_f64; summary.plan.n_fft];
    let mut dummy = [0.0_f64; 33];
    let mut nyq_p = 0.0_f64;
    let scale = summary.inv_wins * summary.inv_win_norm;
    let mut w_idx = 0_usize;
    while w_idx < n_wins {
        let st0 = summary.seg_starts[w_idx];
        let seg0 = &sig[summary.warmup + st0..summary.warmup + st0 + summary.seg_len];
        let seg1 = (w_idx + 1 < n_wins).then(|| {
            let st1 = summary.seg_starts[w_idx + 1];
            &sig[summary.warmup + st1..summary.warmup + st1 + summary.seg_len]
        });
        accum_paired_forward_fft(
            &summary.plan,
            summary.seg_len,
            &summary.win,
            &summary.bin_band,
            seg0,
            seg1,
            &mut fft_re,
            &mut fft_im,
            scale,
            &mut dummy,
            Some(&mut nyq_p),
            None,
            None,
        );
        w_idx += 2;
    }
    nyq_p
}

#[inline]
pub(crate) fn eval_nyquist_stopband_hinge(
    summary: &Welch33TargetSummary,
    p_nyq: f64,
) -> (f64, f64) {
    let nyq_ceil = nyq_ceil_power(summary);
    if p_nyq <= nyq_ceil {
        return (0.0, 0.0);
    }
    let db_per_ln = 10.0 / LN_10;
    let u_db = db_per_ln * (p_nyq / nyq_ceil).ln();
    let loss = 2.0 * u_db * u_db;
    let d_loss = 4.0 * u_db * db_per_ln / p_nyq;
    (loss, d_loss)
}

#[cfg(test)]
pub(crate) fn eval_nyquist_hinge_and_adjoint(
    summary: &Welch33TargetSummary,
    y: &[f64],
    mut grad_dy: Option<&mut [f64]>,
    scratch: &mut Welch33Scratch,
) -> f64 {
    let n_wins = summary.seg_starts.len();
    if n_wins == 0 || summary.seg_len == 0 || y.len() < summary.warmup + summary.seg_len {
        return 0.0;
    }
    let n_fft = summary.n_fft();
    let half_p1 = n_fft / 2 + 1;
    let need_grad = grad_dy.is_some();
    scratch.fft_re.resize(n_fft, 0.0);
    scratch.fft_im.resize(n_fft, 0.0);
    if need_grad {
        let total = n_wins * half_p1;
        if scratch.win_re.len() < total {
            scratch.win_re.resize(total, 0.0);
            scratch.win_im.resize(total, 0.0);
        }
    }
    let scale = summary.inv_wins * summary.inv_win_norm;
    let mut dummy_bp = [0.0_f64; 33];
    let mut p_nyq = 0.0_f64;
    let mut w_idx = 0_usize;
    while w_idx < n_wins {
        let st0 = summary.seg_starts[w_idx];
        let seg0 = &y[summary.warmup + st0..summary.warmup + st0 + summary.seg_len];
        let seg1 = (w_idx + 1 < n_wins).then(|| {
            let st1 = summary.seg_starts[w_idx + 1];
            &y[summary.warmup + st1..summary.warmup + st1 + summary.seg_len]
        });
        if need_grad {
            let off0 = w_idx * half_p1;
            let (wr_h, wr_t) = scratch.win_re.split_at_mut(off0 + half_p1);
            let (wi_h, wi_t) = scratch.win_im.split_at_mut(off0 + half_p1);
            let out0 = Some((&mut wr_h[off0..], &mut wi_h[off0..]));
            let out1 = seg1.map(|_| (&mut wr_t[..half_p1], &mut wi_t[..half_p1]));
            accum_paired_forward_fft(
                &summary.plan,
                summary.seg_len,
                &summary.win,
                &summary.bin_band,
                seg0,
                seg1,
                &mut scratch.fft_re,
                &mut scratch.fft_im,
                scale,
                &mut dummy_bp,
                Some(&mut p_nyq),
                out0,
                out1,
            );
        } else {
            accum_paired_forward_fft(
                &summary.plan,
                summary.seg_len,
                &summary.win,
                &summary.bin_band,
                seg0,
                seg1,
                &mut scratch.fft_re,
                &mut scratch.fft_im,
                scale,
                &mut dummy_bp,
                Some(&mut p_nyq),
                None,
                None,
            );
        }
        w_idx += 2;
    }
    let (loss, nyq_d) = eval_nyquist_stopband_hinge(summary, p_nyq);
    if let Some(dy) = grad_dy.as_deref_mut()
        && nyq_d != 0.0
    {
        let (inactive, zero_d) = ([false; 33], [0.0_f64; 33]);
        let mut w_idx = 0_usize;
        while w_idx < n_wins {
            let base0 = summary.warmup + summary.seg_starts[w_idx];
            let off0 = w_idx * half_p1;
            let (off1_opt, base1_opt) = if w_idx + 1 < n_wins {
                (
                    Some((w_idx + 1) * half_p1),
                    Some(summary.warmup + summary.seg_starts[w_idx + 1]),
                )
            } else {
                (None, None)
            };
            accum_paired_backward_fft(
                &summary.plan,
                summary.seg_len,
                &summary.win,
                &summary.bin_band,
                &inactive,
                &zero_d,
                nyq_d,
                scale,
                &scratch.win_re,
                &scratch.win_im,
                off0,
                off1_opt,
                &mut scratch.fft_re,
                &mut scratch.fft_im,
                dy,
                base0,
                base1_opt,
            );
            w_idx += 2;
        }
    }
    loss
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;
    use crate::solver::v3_varpro::ltas_adj::{
        build_welch33_target_summary, eval_welch33_lse_minimax_loss_and_adjoint,
        eval_welch33_lse_minimax_with_nyq_and_adjoint,
    };

    #[test]
    fn test_nyquist_hinge_power_and_adjoint_match_central_finite_diff() {
        let n = 4_096;
        let warmup = 256;
        let sr = 48_000.0;
        let tgt: Vec<f64> = (0..n)
            .map(|i| 0.25 * (TAU * 420.0 * (i as f64) / sr).sin())
            .collect();
        let sum = build_welch33_target_summary(&tgt, warmup, sr);
        assert!(sum.tgt_nyq_p < 1.0e-6);

        let y: Vec<f64> = (0..n)
            .map(|i| tgt[i] + 0.08 * (TAU * 21_700.0 * (i as f64) / sr).sin())
            .collect();
        let p_nyq = compute_nyquist_stopband_power(&sum, &y);
        assert!(p_nyq > nyq_ceil_power(&sum));

        let mut sc = Welch33Scratch::default();
        let mut dy = vec![0.0_f64; n];
        let loss = eval_nyquist_hinge_and_adjoint(&sum, &y, Some(&mut dy), &mut sc);
        assert!(
            loss > 0.1 && loss < 5_000.0,
            "well-conditioned log-dB loss={loss}"
        );

        let dy_peak = dy.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let eps = 1.0e-6;
        for &idx in &[warmup + 41, warmup + 517, warmup + 1_803, n - 97] {
            let mut yp = y.clone();
            let mut ym = y.clone();
            yp[idx] += eps;
            ym[idx] -= eps;
            let lp = eval_nyquist_hinge_and_adjoint(&sum, &yp, None, &mut sc);
            let lm = eval_nyquist_hinge_and_adjoint(&sum, &ym, None, &mut sc);
            let fd = (lp - lm) / (2.0 * eps);
            let denom = fd
                .abs()
                .max(dy[idx].abs())
                .max(dy_peak * 1.0e-2)
                .max(1.0e-9);
            assert!(
                (dy[idx] - fd).abs() / denom < 1.0e-5,
                "idx={idx}: {} vs {fd} (denom={denom})",
                dy[idx]
            );
        }

        let mut dy_sep = vec![0.0_f64; n];
        let l_sep = eval_welch33_lse_minimax_loss_and_adjoint(
            &sum,
            &y,
            Some(&mut dy_sep),
            16.0,
            28.0,
            0.01,
            &mut sc,
        ) + eval_nyquist_hinge_and_adjoint(&sum, &y, Some(&mut dy_sep), &mut sc);
        let mut dy_fused = vec![0.0_f64; n];
        let l_fused = eval_welch33_lse_minimax_with_nyq_and_adjoint(
            &sum,
            &y,
            Some(&mut dy_fused),
            16.0,
            28.0,
            0.01,
            &mut sc,
        );
        assert!(
            (l_fused - l_sep).abs() / l_sep.abs().max(1.0) < 1.0e-12,
            "fused={l_fused} vs sep={l_sep}"
        );
        let dy_f_peak = dy_fused.iter().fold(1.0_f64, |m, &v| m.max(v.abs()));
        for i in 0..n {
            let denom = dy_sep[i].abs().max(dy_fused[i].abs()).max(dy_f_peak);
            assert!(
                (dy_fused[i] - dy_sep[i]).abs() / denom < 1.0e-12,
                "dy[{i}]: {} vs {} (peak={dy_f_peak})",
                dy_fused[i],
                dy_sep[i]
            );
        }
    }
}
