#![forbid(unsafe_code)]
//! Causal degree-7 Binomial/Bézout 3-band crossover bank (`<= 250 Hz`, MF, `>= 2.5 kHz`).

use crate::metrics::BEZOUT7_LP_COEFFS;
use std::f64::consts::PI;

/// Causal degree-7 Binomial/Bézout LF (`<=250 Hz`), MF, HF (`>=2.5 kHz`) energy-preserving crossover.
#[derive(Clone, Copy, Debug)]
pub struct Bezout7FilterBank {
    pub r1: f64,
    pub omr1: f64,
    pub r2: f64,
    pub omr2: f64,
}

impl Bezout7FilterBank {
    /// Constructs a degree-7 Bézout 3-band crossover bank matching `metrics::split_bezout_3band_energies`.
    #[must_use]
    pub fn new(sample_rate_hz: f64) -> Self {
        let sr = sample_rate_hz.max(8_000.0);
        let r1 = (-2.0 * PI * 250.0 / sr).exp();
        let r2 = (-2.0 * PI * 2_500.0 / sr).exp();
        Self {
            r1,
            omr1: 1.0 - r1,
            r2,
            omr2: 1.0 - r2,
        }
    }

    #[inline(always)]
    fn step_lp7(x: f64, r: f64, omr: f64, u: &mut [f64; 7]) -> f64 {
        let mut prev = x;
        for stage in u.iter_mut() {
            *stage = omr * prev + r * *stage;
            prev = *stage;
        }
        BEZOUT7_LP_COEFFS[0] * u[3]
            + BEZOUT7_LP_COEFFS[1] * u[4]
            + BEZOUT7_LP_COEFFS[2] * u[5]
            + BEZOUT7_LP_COEFFS[3] * u[6]
    }

    #[inline(always)]
    fn rev_lp7(d_lp: f64, r: f64, omr: f64, r_u: &mut [f64; 7]) -> f64 {
        r_u[3] += BEZOUT7_LP_COEFFS[0] * d_lp;
        r_u[4] += BEZOUT7_LP_COEFFS[1] * d_lp;
        r_u[5] += BEZOUT7_LP_COEFFS[2] * d_lp;
        r_u[6] += BEZOUT7_LP_COEFFS[3] * d_lp;
        for s in (1..7).rev() {
            r_u[s - 1] += omr * r_u[s];
            r_u[s] *= r;
        }
        let dx = omr * r_u[0];
        r_u[0] *= r;
        dx
    }

    /// Splits `sig` into `[LF, MF, HF]` degree-7 complementary signals.
    pub fn split_3band(&self, sig: &[f64], out: &mut [Vec<f64>; 3]) {
        let _ = self.split_3band_and_energies(sig, sig.len(), out);
    }

    /// Splits `sig` into `[LF, MF, HF]` buffers and returns `[sum(LF^2), sum(MF^2), sum(HF^2)]` over `warmup..n`.
    pub fn split_3band_and_energies(
        &self,
        sig: &[f64],
        warmup: usize,
        out: &mut [Vec<f64>; 3],
    ) -> [f64; 3] {
        let n = sig.len();
        for b in out.iter_mut() {
            b.resize(n, 0.0);
        }
        let (mut u1, mut u2) = ([0.0_f64; 7], [0.0_f64; 7]);
        let mut e = [0.0_f64; 3];
        for (t, &x) in sig.iter().enumerate() {
            let lf = Self::step_lp7(x, self.r1, self.omr1, &mut u1);
            let l2 = Self::step_lp7(x, self.r2, self.omr2, &mut u2);
            let mf = l2 - lf;
            let hf = x - l2;
            out[0][t] = lf;
            out[1][t] = mf;
            out[2][t] = hf;
            if t >= warmup {
                e[0] += lf * lf;
                e[1] += mf * mf;
                e[2] += hf * hf;
            }
        }
        e
    }

    #[inline(always)]
    fn rev_3band_step(
        &self,
        d_lf: f64,
        d_mf: f64,
        d_hf: f64,
        r_u1: &mut [f64; 7],
        r_u2: &mut [f64; 7],
    ) -> f64 {
        let d_lp1 = d_lf - d_mf;
        let d_lp2 = d_mf - d_hf;
        d_hf + Self::rev_lp7(d_lp1, self.r1, self.omr1, r_u1)
            + Self::rev_lp7(d_lp2, self.r2, self.omr2, r_u2)
    }

    /// Accumulates reverse-time Bézout 3-band adjoint `dL/dy` from explicit `[LF, MF, HF]`
    /// per-sample band gradients `d_bands` directly into `grad`.
    pub fn accum_3band_band_grads(&self, d_bands: &[Vec<f64>; 3], grad: &mut [f64]) {
        let n = d_bands[0]
            .len()
            .min(d_bands[1].len())
            .min(d_bands[2].len())
            .min(grad.len());
        let (mut r_u1, mut r_u2) = ([0.0_f64; 7], [0.0_f64; 7]);
        for t in (0..n).rev() {
            grad[t] += self.rev_3band_step(
                d_bands[0][t],
                d_bands[1][t],
                d_bands[2][t],
                &mut r_u1,
                &mut r_u2,
            );
        }
    }

    /// Accumulates reverse-time Bézout 3-band adjoint `dL/dy` from precomputed `[LF, MF, HF]` buffers.
    pub fn accum_3band_adjoint(
        &self,
        scratch: &[Vec<f64>; 3],
        warmup: usize,
        bw: [f64; 3],
        grad: &mut [f64],
    ) {
        let n = scratch[0]
            .len()
            .min(scratch[1].len())
            .min(scratch[2].len())
            .min(grad.len());
        let w = warmup.min(n);
        let (mut r_u1, mut r_u2) = ([0.0_f64; 7], [0.0_f64; 7]);
        let two_bw = [2.0 * bw[0], 2.0 * bw[1], 2.0 * bw[2]];
        for t in (0..n).rev() {
            let (d_lf, d_mf, d_hf) = if t >= w {
                (
                    two_bw[0] * scratch[0][t],
                    two_bw[1] * scratch[1][t],
                    two_bw[2] * scratch[2][t],
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            grad[t] += self.rev_3band_step(d_lf, d_mf, d_hf, &mut r_u1, &mut r_u2);
        }
    }

    /// Evaluates weighted Bézout 3-band loss `sum_b bw[b] * sum_{t>=warmup} err_b[t]^2` and accumulates `dL/dy` into `dy`.
    pub fn eval_loss_and_adjoint(
        &self,
        err: &[f64],
        warmup: usize,
        bw: [f64; 3],
        dy: Option<&mut [f64]>,
        scratch: &mut [Vec<f64>; 3],
    ) -> f64 {
        let n = err.len();
        let e = self.split_3band_and_energies(err, warmup, scratch);
        let loss = bw[0] * e[0] + bw[1] * e[1] + bw[2] * e[2];
        if let Some(grad) = dy {
            self.accum_3band_adjoint(scratch, warmup, bw, &mut grad[..n]);
        }
        loss
    }

    /// Splits `sig` and computes `[sum(LF^2), sum(MF^2), sum(HF^2)]` over `warmup..n`.
    #[must_use]
    pub fn band_energies(&self, sig: &[f64], warmup: usize) -> [f64; 3] {
        let (mut u1, mut u2) = ([0.0_f64; 7], [0.0_f64; 7]);
        let mut e = [0.0_f64; 3];
        for (t, &x) in sig.iter().enumerate() {
            let lf = Self::step_lp7(x, self.r1, self.omr1, &mut u1);
            let l2 = Self::step_lp7(x, self.r2, self.omr2, &mut u2);
            let mf = l2 - lf;
            let hf = x - l2;
            if t >= warmup {
                e[0] += lf * lf;
                e[1] += mf * mf;
                e[2] += hf * hf;
            }
        }
        e
    }
}
