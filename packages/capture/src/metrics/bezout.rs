#![forbid(unsafe_code)]
//! Compile-time Bézout / exponential constants & exact degree-7 Binomial/Bézout 3-band crossover
//! (`250 Hz` / `2.5 kHz`, `r[qual.besr.bezout-crossover]`, `r[repo.code.constant-eval-and-derivations]`).

use std::f64::consts::PI;

/// Branch-free compile-time `|x|` for `f64`.
#[inline]
#[must_use]
pub const fn const_abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

/// Compile-time `ln(x)` via `atanh` transform for `x > 0` near `[0.5, 4.0]`.
#[must_use]
pub const fn const_ln_f64_near_unity(x: f64) -> f64 {
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

/// Compile-time `exp(x)` via range-halving + degree-24 Taylor evaluation (`|x| <= 4.0`).
#[must_use]
pub const fn const_exp_f64(x: f64) -> f64 {
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

const fn const_bezout7_lp_coeffs() -> [f64; 4] {
    const fn binom(n: u32, k: u32) -> u32 {
        let mut num = 1_u32;
        let mut den = 1_u32;
        let mut i = 0_u32;
        while i < k {
            num *= n - i;
            den *= i + 1;
            i += 1;
        }
        num / den
    }
    // 1 / B(4, 4) = 7! / (3! * 3!) = 140
    let inv_beta44 = (7 * binom(6, 3)) as f64;
    [
        inv_beta44 * (binom(3, 0) as f64) / 4.0,
        -inv_beta44 * (binom(3, 1) as f64) / 5.0,
        inv_beta44 * (binom(3, 2) as f64) / 6.0,
        -inv_beta44 * (binom(3, 3) as f64) / 7.0,
    ]
}

/// Degree-7 Binomial/Bézout regularized incomplete-beta coefficients `[35, -84, 70, -20]`.
pub const BEZOUT7_LP_COEFFS: [f64; 4] = const_bezout7_lp_coeffs();
const _: () = assert!(
    const_abs_f64(
        BEZOUT7_LP_COEFFS[0] + BEZOUT7_LP_COEFFS[1] + BEZOUT7_LP_COEFFS[2] + BEZOUT7_LP_COEFFS[3]
            - 1.0
    ) < 1.0e-15
);

/// Worst-band bound offset $10 \log_{10}(3) \approx 4.771212547\text{ dB}$ (`ref[qual.besr.noise-subtracted]`).
pub const BESR_WORST_BAND_OFFSET_DB: f64 =
    10.0 * const_ln_f64_near_unity(3.0) / std::f64::consts::LN_10;
const _: () = assert!(const_abs_f64(BESR_WORST_BAND_OFFSET_DB - 4.771_212_547_196_624) < 1.0e-14);

#[inline]
fn eval_bezout_lowpass_complex(freq_hz: f64, cutoff_hz: f64, sample_rate_hz: f64) -> (f64, f64) {
    let r = (-2.0 * PI * cutoff_hz / sample_rate_hz).exp();
    let omega = 2.0 * PI * freq_hz / sample_rate_hz;
    let (sin_w, cos_w) = omega.sin_cos();
    let den_re = 1.0 - r * cos_w;
    let den_im = r * sin_w;
    let den_norm = den_re * den_re + den_im * den_im;
    let num = 1.0 - r;
    let h_re = num * den_re / den_norm;
    let h_im = -num * den_im / den_norm;

    let c_mul = |(ar, ai): (f64, f64), (br, bi): (f64, f64)| (ar * br - ai * bi, ar * bi + ai * br);
    let h2 = c_mul((h_re, h_im), (h_re, h_im));
    let h4 = c_mul(h2, h2);
    let h5 = c_mul(h4, (h_re, h_im));
    let h6 = c_mul(h5, (h_re, h_im));
    let h7 = c_mul(h6, (h_re, h_im));

    (
        BEZOUT7_LP_COEFFS[0] * h4.0
            + BEZOUT7_LP_COEFFS[1] * h5.0
            + BEZOUT7_LP_COEFFS[2] * h6.0
            + BEZOUT7_LP_COEFFS[3] * h7.0,
        BEZOUT7_LP_COEFFS[0] * h4.1
            + BEZOUT7_LP_COEFFS[1] * h5.1
            + BEZOUT7_LP_COEFFS[2] * h6.1
            + BEZOUT7_LP_COEFFS[3] * h7.1,
    )
}

/// Evaluates the exact complex frequency responses `[H_LF(e^{jω}), H_MF(e^{jω}), H_HF(e^{jω})]`
/// of the causal degree-7 Binomial/Bézout crossover at `freq_hz` and `sample_rate_hz`
/// (`ref[qual.besr.bezout-crossover]`, `ref[qual.besr.filter-response]`).
#[must_use]
pub fn eval_bezout_3band_complex(freq_hz: f64, sample_rate_hz: f64) -> [(f64, f64); 3] {
    let fs = if (8_000.0..=1_536_000.0).contains(&sample_rate_hz) {
        sample_rate_hz
    } else {
        48_000.0
    };
    let f = if freq_hz.is_finite() {
        freq_hz.abs().clamp(0.0, 0.5 * fs)
    } else {
        0.0
    };
    let l_250 = eval_bezout_lowpass_complex(f, 250.0, fs);
    let l_2500 = eval_bezout_lowpass_complex(f, 2_500.0, fs);
    [
        l_250,
        (l_2500.0 - l_250.0, l_2500.1 - l_250.1),
        (1.0 - l_2500.0, -l_2500.1),
    ]
}

/// Evaluates the real-part coherent partition `[Re(H_LF), Re(H_MF), Re(H_HF)]` at `48 kHz`.
#[must_use]
pub fn eval_bezout_3band_weights(freq_hz: f64) -> [f64; 3] {
    let [lf, mf, hf] = eval_bezout_3band_complex(freq_hz, 48_000.0);
    [lf.0, mf.0, hf.0]
}

#[inline]
pub(crate) fn step_bezout7_lp(x: f64, r: f64, one_minus_r: f64, u: &mut [f64; 7]) -> f64 {
    let mut prev = x;
    for stage in u.iter_mut() {
        *stage = one_minus_r * prev + r * *stage;
        prev = *stage;
    }
    BEZOUT7_LP_COEFFS[0] * u[3]
        + BEZOUT7_LP_COEFFS[1] * u[4]
        + BEZOUT7_LP_COEFFS[2] * u[5]
        + BEZOUT7_LP_COEFFS[3] * u[6]
}

/// Filters `sig` from exact zero initial state into `[LF, MF, HF]` degree-7 Bézout time series.
#[must_use]
pub(crate) fn split_bezout_3band_series(sig: &[f64], sample_rate_hz: f64) -> Option<[Vec<f64>; 3]> {
    if !(8_000.0..=1_536_000.0).contains(&sample_rate_hz)
        || sig.is_empty()
        || !sig.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let r1 = (-2.0 * PI * 250.0 / sample_rate_hz).exp();
    let omr1 = 1.0 - r1;
    let r2 = (-2.0 * PI * 2_500.0 / sample_rate_hz).exp();
    let omr2 = 1.0 - r2;
    let n = sig.len();
    let (mut u1, mut u2) = ([0.0_f64; 7], [0.0_f64; 7]);
    let mut out = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for &x in sig {
        let lf = step_bezout7_lp(x, r1, omr1, &mut u1);
        let l2 = step_bezout7_lp(x, r2, omr2, &mut u2);
        out[0].push(lf);
        out[1].push(l2 - lf);
        out[2].push(x - l2);
    }
    Some(out)
}

/// Splits a finite-timeline signal into 3 exact complementary Binomial/Bézout bands (`LF`, `MF`, `HF`),
/// stepping zero-initialized filters across the **entire** aligned timeline `0..n_total` before accumulating
/// energies over `warmup_samples..n_total` (`ref[qual.besr.bezout-crossover]`, `ref[qual.besr.finite-timeline]`).
#[must_use]
pub fn try_split_bezout_3band_energies(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    if !(8_000.0..=1_536_000.0).contains(&sample_rate_hz)
        || target.is_empty()
        || target.len() != model.len()
        || warmup_samples >= target.len()
        || !target.iter().all(|v| v.is_finite())
        || !model.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let n_total = target.len();
    if let Some(ns) = noise
        && (ns.len() < n_total || !ns[..n_total].iter().all(|v| v.is_finite()))
    {
        return None;
    }

    let r1 = (-2.0 * PI * 250.0 / sample_rate_hz).exp();
    let omr1 = 1.0 - r1;
    let r2 = (-2.0 * PI * 2_500.0 / sample_rate_hz).exp();
    let omr2 = 1.0 - r2;

    let (mut u1_t, mut u2_t) = ([0.0_f64; 7], [0.0_f64; 7]);
    let (mut u1_e, mut u2_e) = ([0.0_f64; 7], [0.0_f64; 7]);
    let (mut u1_n, mut u2_n) = ([0.0_f64; 7], [0.0_f64; 7]);

    let (mut sig_e, mut err_e, mut nse_e) = ([0.0_f64; 3], [0.0_f64; 3], [0.0_f64; 3]);
    let (mut nz_sig, mut nz_err, mut nz_nse) = ([false; 3], [false; 3], [false; 3]);

    for i in 0..n_total {
        let y = target[i];
        let e = model[i] - y;
        let eta = noise.map_or(0.0, |ns| ns[i]);

        let l250_t = step_bezout7_lp(y, r1, omr1, &mut u1_t);
        let l2500_t = step_bezout7_lp(y, r2, omr2, &mut u2_t);
        let (lf_t, mf_t, hf_t) = (l250_t, l2500_t - l250_t, y - l2500_t);

        let l250_e = step_bezout7_lp(e, r1, omr1, &mut u1_e);
        let l2500_e = step_bezout7_lp(e, r2, omr2, &mut u2_e);
        let (lf_e, mf_e, hf_e) = (l250_e, l2500_e - l250_e, e - l2500_e);

        let l250_n = step_bezout7_lp(eta, r1, omr1, &mut u1_n);
        let l2500_n = step_bezout7_lp(eta, r2, omr2, &mut u2_n);
        let (lf_n, mf_n, hf_n) = (l250_n, l2500_n - l250_n, eta - l2500_n);

        if i >= warmup_samples {
            for (b, (st, (er, ns))) in [lf_t, mf_t, hf_t]
                .into_iter()
                .zip([lf_e, mf_e, hf_e].into_iter().zip([lf_n, mf_n, hf_n]))
                .enumerate()
            {
                sig_e[b] += st * st;
                err_e[b] += er * er;
                nse_e[b] += ns * ns;
                nz_sig[b] |= st != 0.0;
                nz_err[b] |= er != 0.0;
                nz_nse[b] |= ns != 0.0;
            }
        }
    }

    for b in 0..3 {
        if !sig_e[b].is_finite()
            || !err_e[b].is_finite()
            || !nse_e[b].is_finite()
            || (nz_sig[b] && sig_e[b] == 0.0)
            || (nz_err[b] && err_e[b] == 0.0)
            || (nz_nse[b] && nse_e[b] == 0.0)
        {
            return None;
        }
    }
    Some((sig_e, err_e, nse_e))
}

/// Convenience wrapper around [`try_split_bezout_3band_energies`] returning `[f64::NAN; 3]` on invalid inputs.
#[must_use]
pub fn split_bezout_3band_energies(
    target: &[f64],
    model: &[f64],
    noise: Option<&[f64]>,
    sample_rate_hz: f64,
    warmup_samples: usize,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    try_split_bezout_3band_energies(target, model, noise, sample_rate_hz, warmup_samples)
        .unwrap_or(([f64::NAN; 3], [f64::NAN; 3], [f64::NAN; 3]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezout7_exact_complex_sum_crossover_gain_and_finite_timeline() {
        for fs in [44_100.0, 48_000.0, 88_200.0, 96_000.0] {
            for f in [0.0, 20.0, 250.0, 1_000.0, 2_500.0, 10_000.0, 20_000.0] {
                let [lf, mf, hf] = eval_bezout_3band_complex(f, fs);
                assert!((lf.0 + mf.0 + hf.0 - 1.0).abs() < 1.0e-13);
                assert!((lf.1 + mf.1 + hf.1).abs() < 1.0e-13);
            }
        }
        let [lf_250, _, _] = eval_bezout_3band_complex(250.0, 48_000.0);
        let gain_db_250 = 10.0 * (lf_250.0 * lf_250.0 + lf_250.1 * lf_250.1).log10();
        assert!((gain_db_250 - 9.33).abs() < 0.05);

        let sig: Vec<f64> = (0..1_024).map(|i| (0.05 * (i as f64)).sin()).collect();
        let mdl: Vec<f64> = sig.iter().map(|&v| 0.98 * v).collect();
        let full_tl = try_split_bezout_3band_energies(&sig, &mdl, None, 48_000.0, 256).unwrap();
        let reset_at_boundary =
            try_split_bezout_3band_energies(&sig[256..], &mdl[256..], None, 48_000.0, 0).unwrap();
        assert_ne!(full_tl.0, reset_at_boundary.0);

        let series = split_bezout_3band_series(&sig, 48_000.0).unwrap();
        let sum_series: f64 = series[0][256..].iter().map(|v| v * v).sum();
        assert!((sum_series - full_tl.0[0]).abs() < 1.0e-10);
    }
}
