#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Zero-mean analytic envelope extraction using `dnam_kernels::HilbertH1StereoKernel` (6-pair
//! Jacobi elliptic `H1` quadrature lattice, `align::regimes::hilbert_env`).

use dnam_kernels::{HilbertH1StereoKernel, interpolate_hilbert_h1_alphas_f64};

/// 1st-order sub-bass high-pass cutoff stripping DC/rectifier envelope offset (`ref[train.capture.alignment.multi-regime]`).
const SUB_BASS_HPF_CUTOFF_HZ: f64 = 160.0;

/// Extracts zero-mean instantaneous analytic envelopes `(env_dry, env_wet)` simultaneously using
/// a stereo 6-pair Jacobi elliptic `H1` Hilbert lattice rate-adapted to `sample_rate_hz`.
#[must_use]
pub(super) fn compute_analytic_envelopes_stereo_pair(
    dry: &[f32],
    wet: &[f32],
    sample_rate_hz: u32,
) -> (Vec<f32>, Vec<f32>) {
    let n = dry.len().min(wet.len());
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let inv_n = 1.0 / (n as f64);
    let mean_d = dry[..n].iter().map(|&x| f64::from(x)).sum::<f64>() * inv_n;
    let mean_w = wet[..n].iter().map(|&x| f64::from(x)).sum::<f64>() * inv_n;

    let sr = f64::from(sample_rate_hz.clamp(8_000, 384_000));
    let (alpha_i, alpha_q) = interpolate_hilbert_h1_alphas_f64(sr);
    let mut hilbert_fwd = HilbertH1StereoKernel::with_coeffs(alpha_i, alpha_q);
    // Zero-phase forward-backward sub-bass high-pass strips DC/baseband rectifier terms
    // with zero group-delay distortion so Jacobi H1 operates strictly on AC carrier bands.
    let a_hp = (-std::f64::consts::TAU * SUB_BASS_HPF_CUTOFF_HZ / sr).exp();
    let mut hp_d = vec![0.0_f64; n];
    let mut hp_w = vec![0.0_f64; n];
    let (mut xd_prev, mut yd_f, mut xw_prev, mut yw_f) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        let xd = f64::from(dry[i]) - mean_d;
        let xw = f64::from(wet[i]) - mean_w;
        yd_f = a_hp * (yd_f + xd - xd_prev);
        xd_prev = xd;
        yw_f = a_hp * (yw_f + xw - xw_prev);
        xw_prev = xw;
        hp_d[i] = yd_f;
        hp_w[i] = yw_f;
    }
    let (mut bd_prev, mut yd_b, mut bw_prev, mut yw_b) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for idx in (0..n).rev() {
        let xd = hp_d[idx];
        let xw = hp_w[idx];
        yd_b = a_hp * (yd_b + xd - bd_prev);
        bd_prev = xd;
        yw_b = a_hp * (yw_b + xw - bw_prev);
        bw_prev = xw;
        hp_d[idx] = yd_b;
        hp_w[idx] = yw_b;
    }

    let mut i_d = Vec::with_capacity(n);
    let mut q_d = Vec::with_capacity(n);
    let mut i_w = Vec::with_capacity(n);
    let mut q_w = Vec::with_capacity(n);

    for i in 0..n {
        let ((id, qd), (iw, qw)) = hilbert_fwd.step_stereo(hp_d[i] as f32, hp_w[i] as f32);
        i_d.push(id);
        q_d.push(qd);
        i_w.push(iw);
        q_w.push(qw);
    }

    // Time-reversed `alpha_i` all-pass equalization cancels the IIR Hilbert lattice's frequency-
    // dependent group delay across all carrier bands (`H_I(e^{j w}) * H_I^*(e^{j w}) == 1`),
    // preserving true zero-phase analytic envelopes for asymmetric dry vs nonlinear harmonic carriers.
    let mut eq_i = HilbertH1StereoKernel::with_coeffs(alpha_i, alpha_i);
    let mut eq_q = HilbertH1StereoKernel::with_coeffs(alpha_i, alpha_i);
    let mut env_d = vec![0.0_f64; n];
    let mut env_w = vec![0.0_f64; n];
    let mut sum_d = 0.0_f64;
    let mut sum_w = 0.0_f64;

    for idx in (0..n).rev() {
        let ((zi_d, _), (zi_w, _)) = eq_i.step_stereo(i_d[idx], i_w[idx]);
        let ((zq_d, _), (zq_w, _)) = eq_q.step_stereo(q_d[idx], q_w[idx]);
        let md = f64::from(zi_d).hypot(f64::from(zq_d));
        let mw = f64::from(zi_w).hypot(f64::from(zq_w));
        sum_d += md;
        sum_w += mw;
        env_d[idx] = md;
        env_w[idx] = mw;
    }

    let env_mean_d = sum_d * inv_n;
    let env_mean_w = sum_w * inv_n;
    let out_d = env_d.into_iter().map(|e| (e - env_mean_d) as f32).collect();
    let out_w = env_w.into_iter().map(|e| (e - env_mean_w) as f32).collect();
    (out_d, out_w)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;

    #[test]
    fn test_hilbert_h1_stereo_analytic_envelope_recovers_delayed_rectifier_envelope() {
        assert_eq!(
            compute_analytic_envelopes_stereo_pair(&[], &[], 48_000),
            (Vec::new(), Vec::new())
        );

        let (dc_a, dc_b) = compute_analytic_envelopes_stereo_pair(&[0.5; 256], &[0.5; 256], 48_000);
        assert_eq!(dc_a.len(), 256);
        assert!(dc_a.iter().chain(&dc_b).all(|&v| v.abs() < 1.0e-6));

        let shift = 29_usize;
        for test_sr in [48_000_u32, 96_000_u32] {
            let n = if test_sr == 48_000 { 8_192 } else { 16_384 };
            let dry: Vec<f32> = (0..n)
                .map(|i| {
                    let t = (i as f64) / f64::from(test_sr);
                    let mod_env: f64 = [0.03, 0.07, 0.11, 0.14]
                        .iter()
                        .map(|&tc| (-((t - tc) / 0.005).powi(2)).exp())
                        .sum();
                    (mod_env * (TAU * 1_100.0 * t).sin()) as f32
                })
                .collect();
            let mut rect_wet = vec![0.0_f32; n];
            for i in shift..n {
                let x = dry[i - shift];
                rect_wet[i] = 2.0 * x * x;
            }
            let (ed, ew) = compute_analytic_envelopes_stereo_pair(&dry, &rect_wet, test_sr);
            assert_eq!(ed.len(), n);
            assert_eq!(ew.len(), n);
            let est = crate::align::estimate_subsample_lag_and_gain(&ed, &ew, 64).unwrap();
            assert!(
                (est.total_lag_samples - (shift as f64)).abs() <= 1.5,
                "sr={test_sr}: expected ~{shift}, got {}",
                est.total_lag_samples
            );
        }
    }
}
