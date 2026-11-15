#![forbid(unsafe_code)]
//! Causal degree-7 Binomial/Bézout 3-band crossover bank (`<= 250 Hz`, MF, `>= 2.5 kHz`).

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
        35.0 * u[3] - 84.0 * u[4] + 70.0 * u[5] - 20.0 * u[6]
    }

    #[inline(always)]
    fn rev_lp7(d_lp: f64, r: f64, omr: f64, r_u: &mut [f64; 7]) -> f64 {
        r_u[3] += 35.0 * d_lp;
        r_u[4] -= 84.0 * d_lp;
        r_u[5] += 70.0 * d_lp;
        r_u[6] -= 20.0 * d_lp;
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
        let n = sig.len();
        for b in out.iter_mut() {
            b.resize(n, 0.0);
        }
        let (mut u1, mut u2) = ([0.0_f64; 7], [0.0_f64; 7]);
        for (t, &x) in sig.iter().enumerate() {
            let lf = Self::step_lp7(x, self.r1, self.omr1, &mut u1);
            let l2 = Self::step_lp7(x, self.r2, self.omr2, &mut u2);
            out[0][t] = lf;
            out[1][t] = l2 - lf;
            out[2][t] = x - l2;
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
        self.split_3band(err, scratch);
        let w = warmup.min(n);
        let mut loss = 0.0_f64;
        for t in w..n {
            for b in 0..3 {
                let eb = scratch[b][t];
                loss += bw[b] * eb * eb;
            }
        }
        if let Some(grad) = dy {
            let (mut r_u1, mut r_u2) = ([0.0_f64; 7], [0.0_f64; 7]);
            for t in (0..n).rev() {
                let (d_lf, d_mf, d_hf) = if t >= w {
                    (
                        2.0 * bw[0] * scratch[0][t],
                        2.0 * bw[1] * scratch[1][t],
                        2.0 * bw[2] * scratch[2][t],
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };
                let d_lp1 = d_lf - d_mf;
                let d_lp2 = d_mf - d_hf;
                grad[t] += d_hf
                    + Self::rev_lp7(d_lp1, self.r1, self.omr1, &mut r_u1)
                    + Self::rev_lp7(d_lp2, self.r2, self.omr2, &mut r_u2);
            }
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
