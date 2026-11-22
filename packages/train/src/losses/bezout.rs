//! Causal degree-7 Binomial/Bézout 3-band `bESR` and `20 ms` `SegESR` metrics + exact
//! reverse-time IIR adjoints (`ref[qual.besr.bezout-crossover]`, `ref[qual.besr.noise-subtracted]`,
//! `ref[qual.besr.noise-qualified]`, `ref[qual.besr.uncertain-floor]`, `ref[qual.companion.seg-esr]`).

use core::f64::consts::PI;

mod metrics;

pub use metrics::{
    BesrIntervalBounds, BesrIntervalDecision, BesrJointIntervalBounds, compute_seg_esr_channel_db,
    eval_seg_esr_loss_and_adjoint, evaluate_joint_interval_besr, evaluate_uncertain_floor_besr,
};

use crate::math::trainer::{BEZOUT7_C4, BEZOUT7_C5, BEZOUT7_C6, BEZOUT7_C7};

const BEZOUT7_SIGNED: [f64; 4] = [BEZOUT7_C4, -BEZOUT7_C5, BEZOUT7_C6, -BEZOUT7_C7];

#[inline]
const fn const_abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

const fn const_ln_f64_near_unity(x: f64) -> f64 {
    let t = (x - 1.0) / (x + 1.0);
    let t2 = t * t;
    let mut term = t;
    let mut sum = 0.0_f64;
    let mut k = 0_u32;
    while k < 32 {
        sum += term / ((2 * k + 1) as f64);
        term *= t2;
        k += 1;
    }
    2.0 * sum
}

const fn const_exp_f64(x: f64) -> f64 {
    let scaled = x * 0.0625;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    let mut n = 1_u32;
    while n <= 20 {
        term *= scaled / (n as f64);
        sum += term;
        n += 1;
    }
    let s2 = sum * sum;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    s8 * s8
}

/// Canonical `+1.0 dB` power factor $10^{0.1}$ (`ref[qual.besr.noise-qualified]`).
pub const ONE_DB_POWER_FACTOR: f64 = const_exp_f64(0.1 * core::f64::consts::LN_10);
const _: () = assert!(
    const_abs_f64(
        ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            * ONE_DB_POWER_FACTOR
            - 10.0
    ) < 1.0e-12
);

/// Worst-band composite ceiling offset $10 \log_{10}(3)$ dB (`ref[qual.besr.noise-subtracted]`).
pub const BESR_WORST_BAND_OFFSET_DB: f64 =
    10.0 * const_ln_f64_near_unity(3.0) / core::f64::consts::LN_10;
const _: () = assert!(const_abs_f64(BESR_WORST_BAND_OFFSET_DB - 4.771_212_547_196_624) < 1.0e-14);
const _: () = assert!(
    const_abs_f64(
        BEZOUT7_SIGNED[0] + BEZOUT7_SIGNED[1] + BEZOUT7_SIGNED[2] + BEZOUT7_SIGNED[3] - 1.0
    ) < 1.0e-15
);

/// Causal degree-7 Binomial/Bézout 3-band crossover bank (`250 Hz` / `2.5 kHz`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bezout7FilterBank {
    /// Pole radius at `250 Hz`.
    pub r1: f64,
    /// `1 - r1`.
    pub omr1: f64,
    /// Pole radius at `2,500 Hz`.
    pub r2: f64,
    /// `1 - r2`.
    pub omr2: f64,
}

impl Bezout7FilterBank {
    /// Builds a Bézout-7 crossover bank calibrated to `sample_rate_hz`.
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
        BEZOUT7_SIGNED[0] * u[3]
            + BEZOUT7_SIGNED[1] * u[4]
            + BEZOUT7_SIGNED[2] * u[5]
            + BEZOUT7_SIGNED[3] * u[6]
    }

    #[inline(always)]
    fn rev_lp7(d_lp: f64, r: f64, omr: f64, r_u: &mut [f64; 7]) -> f64 {
        r_u[3] += BEZOUT7_SIGNED[0] * d_lp;
        r_u[4] += BEZOUT7_SIGNED[1] * d_lp;
        r_u[5] += BEZOUT7_SIGNED[2] * d_lp;
        r_u[6] += BEZOUT7_SIGNED[3] * d_lp;
        for s in (1..7).rev() {
            r_u[s - 1] += omr * r_u[s];
            r_u[s] *= r;
        }
        let dx = omr * r_u[0];
        r_u[0] *= r;
        dx
    }

    /// Splits `sig` from exact zero state into `[LF, MF, HF]` with partition-of-unity sum.
    #[must_use]
    pub fn split_3band(&self, sig: &[f64]) -> [Vec<f64>; 3] {
        let n = sig.len();
        let mut out = [vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]];
        let _ = self.split_3band_and_energies(sig, n, &mut out);
        out
    }

    /// Splits `sig` over the entire timeline and returns `[E_LF, E_MF, E_HF]` accumulated over `warmup..n`.
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

    /// Propagates per-sample band gradients `[d_lf, d_mf, d_hf]` backward into `grad` (`dL/dy`).
    pub fn accum_3band_adjoint(&self, d_bands: &[Vec<f64>; 3], grad: &mut [f64]) {
        let n = d_bands[0]
            .len()
            .min(d_bands[1].len())
            .min(d_bands[2].len())
            .min(grad.len());
        let (mut r_u1, mut r_u2) = ([0.0_f64; 7], [0.0_f64; 7]);
        for t in (0..n).rev() {
            let (d_lf, d_mf, d_hf) = (d_bands[0][t], d_bands[1][t], d_bands[2][t]);
            let d_lp1 = d_lf - d_mf;
            let d_lp2 = d_mf - d_hf;
            grad[t] += d_hf
                + Self::rev_lp7(d_lp1, self.r1, self.omr1, &mut r_u1)
                + Self::rev_lp7(d_lp2, self.r2, self.omr2, &mut r_u2);
        }
    }

    /// Evaluates differentiable normalized `bESR = (1/3) sum_b E_b(pred - target) / (E_b(target) + eps_floor[b])`
    /// and accumulates exact reverse-time gradients into `grad_opt`.
    pub fn eval_besr_loss_and_adjoint(
        &self,
        pred: &[f64],
        target: &[f64],
        target_band_energies: [f64; 3],
        eps_floor: [f64; 3],
        warmup: usize,
        grad_opt: Option<&mut [f64]>,
    ) -> f64 {
        let n = pred.len().min(target.len());
        let err: Vec<f64> = (0..n).map(|t| pred[t] - target[t]).collect();
        let mut bands = [Vec::new(), Vec::new(), Vec::new()];
        let err_e = self.split_3band_and_energies(&err, warmup.min(n), &mut bands);

        let inv_denom = [
            1.0 / (3.0 * (target_band_energies[0] + eps_floor[0]).max(1.0e-18)),
            1.0 / (3.0 * (target_band_energies[1] + eps_floor[1]).max(1.0e-18)),
            1.0 / (3.0 * (target_band_energies[2] + eps_floor[2]).max(1.0e-18)),
        ];
        let loss = err_e[0] * inv_denom[0] + err_e[1] * inv_denom[1] + err_e[2] * inv_denom[2];

        if let Some(grad) = grad_opt {
            let w0 = warmup.min(n);
            let mut d_bands = [vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]];
            for b in 0..3 {
                let s = 2.0 * inv_denom[b];
                for t in w0..n {
                    d_bands[b][t] = s * bands[b][t];
                }
            }
            self.accum_3band_adjoint(&d_bands, &mut grad[..n]);
        }
        loss
    }
}

/// Noise-subtracted `bESR` report for one output channel (`ref[qual.besr.noise-subtracted]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BesrChannelReport {
    /// Whether inputs were finite and zero-denominator collapse was absent.
    pub valid: bool,
    /// Target band energies `Y_b`.
    pub sig_energy: [f64; 3],
    /// Residual band energies `R_b`.
    pub err_energy: [f64; 3],
    /// Expected noise band energies `F_b`.
    pub noise_energy: [f64; 3],
    /// Noise-subtracted linear band ESR `max(0, R_b - F_b) / (Y_b + F_b)`.
    pub band_esr_lin: [f64; 3],
    /// Per-band `10 log10(band_esr_lin[b])` in dB.
    pub band_esr_db: [f64; 3],
    /// Composite linear average `(ESR_LF + ESR_MF + ESR_HF) / 3`.
    pub besr_lin: f64,
    /// Composite `bESR` in dB.
    pub besr_db: f64,
}

/// Exact noise-qualified `bESR` acceptance evaluation (`ref[qual.besr.noise-qualified]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseQualification {
    /// True for bands meeting all 3 qualification conditions (`F > 0`, `Y/F < 10^{-g/10}`, `R^10 <= 10 F^10`).
    pub qualified_bands: [bool; 3],
    /// Composite linear `bESR` with qualifying bands zeroed (still dividing by **3**).
    pub qualified_besr_lin: f64,
    /// Composite noise-qualified `bESR` in dB.
    pub qualified_besr_db: f64,
    /// Overall pass against `per_band_ceil_db` and `composite_ceil_db`.
    pub passes: bool,
}

pub(super) fn esr_ratio(y: f64, r: f64, f: f64) -> (f64, bool) {
    if !y.is_finite() || !r.is_finite() || !f.is_finite() || y < 0.0 || r < 0.0 || f < 0.0 {
        return (f64::INFINITY, false);
    }
    let d = y + f;
    if d > 0.0 {
        ((r - f).max(0.0) / d, true)
    } else if r == 0.0 {
        (0.0, true)
    } else {
        (f64::INFINITY, false)
    }
}

pub(super) fn linear_to_db(val: f64) -> f64 {
    if val == 0.0 {
        f64::NEG_INFINITY
    } else if val.is_infinite() {
        f64::INFINITY
    } else {
        10.0 * val.log10()
    }
}

/// Computes `BesrChannelReport` on one channel across `[warmup..n]` after full-timeline zero-init
/// filtering (`ref[qual.besr.finite-timeline]`, `ref[qual.besr.zero-denominator]`).
#[must_use]
pub fn compute_besr_channel(
    pred: &[f64],
    target: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup: usize,
) -> BesrChannelReport {
    let n = pred.len().min(target.len());
    let bank = Bezout7FilterBank::new(sample_rate_hz);
    let mut tmp = [Vec::new(), Vec::new(), Vec::new()];
    let err: Vec<f64> = (0..n).map(|t| pred[t] - target[t]).collect();
    let w = warmup.min(n);
    let sig_e = bank.split_3band_and_energies(&target[..n], w, &mut tmp);
    let err_e = bank.split_3band_and_energies(&err, w, &mut tmp);
    let nse_e = match noise {
        Some(ns) if ns.len() >= n => bank.split_3band_and_energies(&ns[..n], w, &mut tmp),
        _ => [0.0; 3],
    };

    let mut band_lin = [0.0_f64; 3];
    let mut band_db = [f64::NEG_INFINITY; 3];
    let mut valid = true;
    for b in 0..3 {
        let (lin, ok) = esr_ratio(sig_e[b], err_e[b], nse_e[b]);
        valid &= ok;
        band_lin[b] = lin;
        band_db[b] = linear_to_db(lin);
    }
    let besr_lin = (band_lin[0] + band_lin[1] + band_lin[2]) / 3.0;
    BesrChannelReport {
        valid,
        sig_energy: sig_e,
        err_energy: err_e,
        noise_energy: nse_e,
        band_esr_lin: band_lin,
        band_esr_db: band_db,
        besr_lin,
        besr_db: linear_to_db(besr_lin),
    }
}

/// Evaluates `ref[qual.besr.noise-qualified]` using exact rational `R^10 <= 10 F^10` residual check.
#[must_use]
pub fn evaluate_noise_qualified_besr(
    report: &BesrChannelReport,
    per_band_ceil_db: [f64; 3],
    composite_ceil_db: f64,
) -> NoiseQualification {
    let mut qual = [false; 3];
    let mut band_ok = [false; 3];
    let mut qual_lin = [0.0_f64; 3];
    for b in 0..3 {
        let y = report.sig_energy[b];
        let r = report.err_energy[b];
        let f = report.noise_energy[b];
        let g = 10.0_f64.powf(per_band_ceil_db[b] / 10.0);
        let cond1 = f.is_finite() && f > 0.0;
        let cond2 = cond1 && (y / f) < (1.0 / g);
        // Exact rational R <= 10^{0.1} F => (R / F)^10 <= 10.0.
        let cond3 = cond1 && (r / f).powi(10) <= 10.0 + 1.0e-12;
        if cond1 && cond2 && cond3 {
            qual[b] = true;
            band_ok[b] = true;
            qual_lin[b] = 0.0;
        } else {
            band_ok[b] = report.band_esr_lin[b] <= g;
            qual_lin[b] = report.band_esr_lin[b];
        }
    }
    let q_comp = (qual_lin[0] + qual_lin[1] + qual_lin[2]) / 3.0;
    let g_comp = 10.0_f64.powf(composite_ceil_db / 10.0);
    NoiseQualification {
        qualified_bands: qual,
        qualified_besr_lin: q_comp,
        qualified_besr_db: linear_to_db(q_comp),
        passes: report.valid && band_ok.iter().all(|&p| p) && q_comp <= g_comp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezout7_partition_and_besr_fd_adjoint() {
        let bank = Bezout7FilterBank::new(48_000.0);
        let n = 1_200_usize;
        let target: Vec<f64> = (0..n)
            .map(|t| {
                0.4 * ((t as f64) * 0.025).sin()
                    + 0.35 * ((t as f64) * 0.14).cos()
                    + 0.25 * ((t as f64) * 0.95).sin()
            })
            .collect();
        let bands = bank.split_3band(&target);
        for t in 0..n {
            let s = bands[0][t] + bands[1][t] + bands[2][t];
            assert!((s - target[t]).abs() <= 1.0e-13);
        }

        let pred: Vec<f64> = target
            .iter()
            .enumerate()
            .map(|(i, &x)| 0.9 * x + 0.02 * (i as f64 * 0.3).cos())
            .collect();
        let target_e = bank.split_3band_and_energies(&target, 128, &mut [vec![], vec![], vec![]]);
        let mut g_besr = vec![0.0_f64; n];
        let _ = bank.eval_besr_loss_and_adjoint(
            &pred,
            &target,
            target_e,
            [1.0e-6; 3],
            128,
            Some(&mut g_besr),
        );
        let eps = 1.0e-6_f64;
        for idx in [20, 150, 500, 990] {
            let mut pp = pred.clone();
            let mut pm = pred.clone();
            pp[idx] += eps;
            pm[idx] -= eps;
            let fd_b =
                (bank.eval_besr_loss_and_adjoint(&pp, &target, target_e, [1.0e-6; 3], 128, None)
                    - bank.eval_besr_loss_and_adjoint(
                        &pm,
                        &target,
                        target_e,
                        [1.0e-6; 3],
                        128,
                        None,
                    ))
                    / (2.0 * eps);
            assert!(
                (g_besr[idx] - fd_b).abs() <= 1.0e-6 * g_besr[idx].abs().max(1.0e-3),
                "bESR idx={idx} ana={} fd={fd_b}",
                g_besr[idx]
            );
        }
        let q_rep = BesrChannelReport {
            valid: true,
            sig_energy: [0.5, 100.0, 100.0],
            err_energy: [1.2, 0.01, 0.01],
            noise_energy: [1.0, 0.0, 0.0],
            band_esr_lin: [0.2 / 1.5, 1.0e-4, 1.0e-4],
            band_esr_db: [-8.75, -40.0, -40.0],
            besr_lin: 0.044,
            besr_db: -13.5,
        };
        let nq = evaluate_noise_qualified_besr(&q_rep, [-23.0, -20.0, -15.0], -18.0);
        assert!(nq.qualified_bands[0] && nq.passes);
    }

    #[test]
    fn test_bezout_const_helpers_and_ratio_edges() {
        assert_eq!(const_abs_f64(-2.5), 2.5);
        assert_eq!(const_abs_f64(2.5), 2.5);
        assert!((const_ln_f64_near_unity(1.5) - 1.5_f64.ln()).abs() <= 1.0e-12);
        assert!((const_exp_f64(0.1) - 0.1_f64.exp()).abs() <= 1.0e-12);
        assert_eq!(esr_ratio(-1.0, 0.0, 0.0), (f64::INFINITY, false));
        assert_eq!(esr_ratio(0.0, 0.0, 0.0), (0.0, true));
        assert_eq!(esr_ratio(0.0, 1.0, 0.0), (f64::INFINITY, false));
        assert_eq!(linear_to_db(0.0), f64::NEG_INFINITY);
        assert_eq!(linear_to_db(f64::INFINITY), f64::INFINITY);
    }
}
