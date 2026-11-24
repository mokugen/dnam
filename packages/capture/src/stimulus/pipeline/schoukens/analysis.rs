#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! `Schoukens–Pintelon` Frequency-Domain `BLA` (`bla_mag_33_db`), Even/Odd Nonlinear Distortion
//! Separation (`even_dist_ratio_db`, `odd_dist_ratio_db`), `Law 0.A`-Gated Noise Variance
//! (`noise_var_33_lin`, `law0a_ltv_flag`), and Schroeder `EDC` $T_{60}$ Analyzer
//! (`stimulus::pipeline::schoukens::analysis`).
//!
//! Canonical Reference: `docs/specs/training/capture_and_alignment.md`
//! (`ref[train.capture.excitation.codesigned]`) & `docs/specs/containers/project_container_format.md`
//! (`ref[container.proj.jcs-manifest]`).

use std::collections::BTreeSet;

use crate::CANONICAL_TRAINER_SAMPLE_RATE_HZ;
use crate::metrics::canonical_ltas_33_intervals_hz;
use crate::proj::schema::ExtractedPhysics;
use crate::stimulus::fft::{Complex64, rfft_bins};

use super::edc::compute_schroeder_edc_t60_s;
use super::regimes::OddRandomMultisinePlan;

/// Minimum inter-period coherence required for `Law 0.A` time-invariance (`< 0.90 => law0a_ltv_flag`).
pub const LAW0A_MIN_PERIOD_COHERENCE: f64 = 0.90;

/// Maximum baseline normalized period mismatch fraction under `Law 0.A` (`> max(0.01, 8*E_calb/E_y1)`).
pub const LAW0A_REL_MISMATCH_FLOOR: f64 = 0.01;

/// Exact unbiased estimator factor `1 / (1 - ln 2) ~= 3.258891` recovering mean complex-Gaussian
/// bin noise power `E[|W|^2]` from the lower half of sorted `Exp(1)` guard-bin powers
/// (`E[Z | Z <= ln 2] = 1 - ln 2`).
const EXP_LOWER_HALF_MEAN_INV: f64 = 1.0 / (1.0 - std::f64::consts::LN_2);
const _: () = assert!(EXP_LOWER_HALF_MEAN_INV > 3.25 && EXP_LOWER_HALF_MEAN_INV < 3.27);

/// Aligned `Regime 2` wet audio and quiet decay gap intervals for [`analyze_schoukens_pintelon_physics`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Regime2DecayInput<'a> {
    /// Aligned left-channel `Regime 2` wet samples.
    pub wet_l: &'a [f32],
    /// Optional aligned right-channel `Regime 2` wet samples.
    pub wet_r: Option<&'a [f32]>,
    /// `(gap_start, gap_end)` sample index intervals for `Regime 2` quiet decay tails.
    pub gap_ranges: &'a [(usize, usize)],
}

/// Configuration for [`analyze_schoukens_pintelon_physics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchoukensAnalysisConfig {
    /// Calibrated quiescent hardware noise floor in dBFS (`default = -100.0`).
    pub calb_noise_floor_dbfs: f32,
    /// Condensed coreset regime boundaries `[0, r2_start, r3_start, r4_start, total_frames]`.
    pub coreset_regime_offsets: [u32; 5],
}

impl Default for SchoukensAnalysisConfig {
    fn default() -> Self {
        Self {
            calb_noise_floor_dbfs: -100.0,
            coreset_regime_offsets: [0; 5],
        }
    }
}

/// Physical descriptors extracted from Two-Tier `Schoukens–Pintelon` capture responses.
#[derive(Debug, Clone, PartialEq)]
pub struct SchoukensPhysicsResult {
    /// Best Linear Approximation 1/3-octave magnitude profile (`33` bands, dB).
    pub bla_mag_33_db: Vec<f32>,
    /// Broadband even-order distortion ratio relative to fundamental (dB).
    pub even_dist_ratio_db: f32,
    /// Broadband odd-order distortion ratio relative to fundamental (dB).
    pub odd_dist_ratio_db: f32,
    /// `Law 0.A`-gated 1/3-octave repeatability noise variance (`33` non-negative values, linear power).
    pub noise_var_33_lin: Vec<f32>,
    /// Schroeder Energy Decay Curve `60 dB` reverberation time in seconds (`>= 0.0`).
    pub edc_t60_s: f32,
    /// True when `Law 0.A` detected time-varying modulation across settled `Regime 1` periods.
    pub law0a_ltv_flag: bool,
    /// Monotonic coreset regime sample boundaries `[0, r2_start, r3_start, r4_start, total_frames]`.
    pub coreset_regime_offsets: [u32; 5],
}

impl SchoukensPhysicsResult {
    /// Merges the 7 `Schoukens–Pintelon` physics fields into `target` [`ExtractedPhysics`].
    pub fn apply_to_extracted_physics(&self, target: &mut ExtractedPhysics) {
        target.bla_mag_33_db = self.bla_mag_33_db.clone();
        target.even_dist_ratio_db = self.even_dist_ratio_db;
        target.odd_dist_ratio_db = self.odd_dist_ratio_db;
        target.noise_var_33_lin = self.noise_var_33_lin.clone();
        target.edc_t60_s = self.edc_t60_s;
        target.law0a_ltv_flag = self.law0a_ltv_flag;
        if self.coreset_regime_offsets != [0; 5] {
            target.coreset_regime_offsets = self.coreset_regime_offsets;
        }
    }
}

fn check_law0a_periods_ltv(
    specs: &[Vec<Complex64>],
    plan: &OddRandomMultisinePlan,
    parseval_scale: f64,
    calb_total_lin: f64,
) -> bool {
    if specs.len() < 2 {
        return false;
    }
    let y1 = &specs[0];
    let n_bins = y1.len().saturating_sub(1);
    let mut e_y1_raw = 0.0_f64;
    for (k, c) in y1.iter().enumerate().take(n_bins).skip(1) {
        if !plan.masked_bins.contains(&k) {
            e_y1_raw += c.re * c.re + c.im * c.im;
        }
    }
    let e_y1 = e_y1_raw * parseval_scale;
    if e_y1 <= 1.0e-12 {
        return false;
    }
    let thresh = ((8.0 * calb_total_lin) / e_y1).max(LAW0A_REL_MISMATCH_FLOOR);

    for yp in &specs[1..] {
        let mut diff_e_raw = 0.0_f64;
        let mut dot_raw = 0.0_f64;
        let mut e_yp_raw = 0.0_f64;
        for k in 1..n_bins {
            if plan.masked_bins.contains(&k) {
                continue;
            }
            let a = y1[k];
            let b = yp[k];
            let dr = b.re - a.re;
            let di = b.im - a.im;
            diff_e_raw += dr * dr + di * di;
            dot_raw += a.re * b.re + a.im * b.im;
            e_yp_raw += b.re * b.re + b.im * b.im;
        }
        let mismatch = (diff_e_raw * parseval_scale) / e_y1;
        let e_yp = e_yp_raw * parseval_scale;
        let coh = ((dot_raw * parseval_scale) / (e_y1 * e_yp).sqrt().max(1.0e-18)).abs();
        if mismatch > thresh || coh < LAW0A_MIN_PERIOD_COHERENCE {
            return true;
        }
    }
    false
}

fn estimate_unexcited_guard_bin_noise_floor(
    plan: &OddRandomMultisinePlan,
    y0: &[f32],
    parseval_scale: f64,
    calb_bin_floor: f64,
) -> f64 {
    let n_per = plan.period_samples;
    if y0.len() < n_per || (plan.omitted_odd_bins.len() < 16 && plan.unexcited_even_bins.len() < 16)
    {
        return calb_bin_floor;
    }
    // Energy-normalized Hann window (`sqrt(8/3) * hann`) suppresses fractional-cycle LTV AM/LFO
    // rectangular-window edge leakage on isolated bins separated from excited carriers (`|dk| >= 2`).
    let hann_scale = (8.0_f64 / 3.0_f64).sqrt() * 0.5;
    let inv_n = 1.0 / (n_per as f64);
    let windowed: Vec<f32> = (0..n_per)
        .map(|i| {
            let w = hann_scale * (1.0 - (std::f64::consts::TAU * (i as f64) * inv_n).cos());
            (f64::from(y0[i]) * w) as f32
        })
        .collect();
    let w_spec = rfft_bins(&windowed);

    let lower_half_est = |bins: &[usize]| -> Option<f64> {
        let mut vals: Vec<f64> = bins
            .iter()
            .filter(|&&b| b < w_spec.len() && !plan.masked_bins.contains(&b))
            .map(|&b| w_spec[b].norm_sqr() * parseval_scale)
            .collect();
        if vals.len() < 8 {
            return None;
        }
        vals.sort_by(f64::total_cmp);
        let q1_end = (vals.len() / 2).max(1);
        let mean_low = vals[..q1_end].iter().sum::<f64>() / (q1_end as f64);
        Some(mean_low * EXP_LOWER_HALF_MEAN_INV)
    };

    // Prefer isolated even bins whose odd +/-1 neighbors are BOTH omitted/unexcited (zero Hann
    // carrier bleed), so pure odd nonlinear distortion cannot inflate the noise-floor estimate.
    let excited_set: BTreeSet<usize> = plan.excited_odd_bins.iter().map(|&(b, _)| b).collect();
    let isolated_evens: Vec<usize> = plan
        .unexcited_even_bins
        .iter()
        .copied()
        .filter(|&b| {
            !excited_set.contains(&(b.saturating_sub(1))) && !excited_set.contains(&(b + 1))
        })
        .collect();

    let odd_est = lower_half_est(&plan.omitted_odd_bins);
    let even_est = lower_half_est(&isolated_evens);
    let combined = match (even_est, odd_est) {
        (Some(ee), Some(oe)) => ee.min(oe),
        (None, Some(oe)) => oe,
        (Some(ee), None) => ee,
        (None, None) => calb_bin_floor,
    };
    combined.max(calb_bin_floor)
}

/// Analyzes settled `Regime 1` multisine periods (with Period 0 dropped) and aligned `Regime 2`
/// decay gaps to extract `BLA`, even/odd distortion ratios, `Law 0.A`-gated noise variance,
/// `edc_t60_s`, and `law0a_ltv_flag` (`ref[train.capture.excitation.codesigned]`).
#[must_use]
pub fn analyze_schoukens_pintelon_physics(
    plan: &OddRandomMultisinePlan,
    dry_r1_period_l: &[f32],
    settled_r1_periods_l: &[&[f32]],
    settled_r1_periods_r: Option<&[&[f32]]>,
    regime2: Regime2DecayInput<'_>,
    config: SchoukensAnalysisConfig,
) -> SchoukensPhysicsResult {
    let n_per = plan.period_samples;
    let calb_db = f64::from(config.calb_noise_floor_dbfs).clamp(-160.0, -20.0);
    let calb_total_lin = 10.0_f64.powf(calb_db / 10.0);

    if dry_r1_period_l.len() < n_per
        || settled_r1_periods_l.is_empty()
        || settled_r1_periods_l.iter().any(|p| p.len() < n_per)
    {
        return SchoukensPhysicsResult {
            bla_mag_33_db: vec![0.0; 33],
            even_dist_ratio_db: -120.0,
            odd_dist_ratio_db: -120.0,
            noise_var_33_lin: vec![(calb_total_lin / 33.0) as f32; 33],
            edc_t60_s: compute_schroeder_edc_t60_s(
                regime2.wet_l,
                regime2.wet_r,
                regime2.gap_ranges,
                CANONICAL_TRAINER_SAMPLE_RATE_HZ,
                calb_total_lin,
            ),
            law0a_ltv_flag: false,
            coreset_regime_offsets: config.coreset_regime_offsets,
        };
    }

    let x_spec = rfft_bins(&dry_r1_period_l[..n_per]);
    let specs_l: Vec<Vec<Complex64>> = settled_r1_periods_l
        .iter()
        .map(|p| rfft_bins(&p[..n_per]))
        .collect();
    let n_bins = x_spec.len();
    let p_count = specs_l.len() as f64;
    let parseval_scale = 2.0 / ((n_per as f64) * (n_per as f64));

    let mut law0a_ltv_flag =
        check_law0a_periods_ltv(&specs_l, plan, parseval_scale, calb_total_lin);
    if !law0a_ltv_flag
        && let Some(periods_r) = settled_r1_periods_r
        && periods_r.len() >= 2
        && periods_r.iter().all(|p| p.len() >= n_per)
    {
        let specs_r: Vec<Vec<Complex64>> =
            periods_r.iter().map(|p| rfft_bins(&p[..n_per])).collect();
        law0a_ltv_flag |= check_law0a_periods_ltv(&specs_r, plan, parseval_scale, calb_total_lin);
    }

    let mut y_mean = vec![Complex64::default(); n_bins];
    for k in 0..n_bins {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for sp in &specs_l {
            re += sp[k].re;
            im += sp[k].im;
        }
        y_mean[k] = Complex64::new(re / p_count, im / p_count);
    }

    let n_pos = (n_per / 2).saturating_sub(1).max(1);
    let calb_bin_floor = calb_total_lin / (n_pos as f64);
    let guard_bin_est = estimate_unexcited_guard_bin_noise_floor(
        plan,
        settled_r1_periods_l[0],
        parseval_scale,
        calb_bin_floor,
    );

    let mut per_bin_noise = vec![calb_bin_floor; n_bins];
    if !law0a_ltv_flag && specs_l.len() >= 2 {
        let p_denom = (specs_l.len() - 1) as f64;
        for k in 1..n_bins.saturating_sub(1) {
            if plan.masked_bins.contains(&k) {
                per_bin_noise[k] = calb_bin_floor;
                continue;
            }
            let mut sum_var = 0.0_f64;
            for sp in &specs_l {
                let dr = sp[k].re - y_mean[k].re;
                let di = sp[k].im - y_mean[k].im;
                sum_var += dr * dr + di * di;
            }
            let var_k = (sum_var * parseval_scale) / p_denom;
            per_bin_noise[k] = var_k.max(calb_bin_floor);
        }
    } else {
        for (k, slot) in per_bin_noise.iter_mut().enumerate().take(n_bins) {
            if !plan.masked_bins.contains(&k) {
                *slot = guard_bin_est;
            }
        }
    }

    let intervals = canonical_ltas_33_intervals_hz();
    let mut noise_var_33_lin = vec![calb_bin_floor as f32; 33];
    let mut band_idx_for_bin = vec![usize::MAX; n_bins];
    let df = f64::from(CANONICAL_TRAINER_SAMPLE_RATE_HZ) / (n_per as f64);
    for (k, slot) in band_idx_for_bin.iter_mut().enumerate().skip(1) {
        let f_hz = (k as f64) * df;
        for (b, iv) in intervals.iter().enumerate() {
            if iv.contains(f_hz) {
                *slot = b;
                break;
            }
        }
    }

    let mut band_noise_sum = [0.0_f64; 33];
    let mut band_bin_count = [0_usize; 33];
    for k in 1..n_bins.saturating_sub(1) {
        let b = band_idx_for_bin[k];
        if b < 33 && !plan.masked_bins.contains(&k) {
            band_noise_sum[b] += per_bin_noise[k];
            band_bin_count[b] += 1;
        }
    }
    for (b, &ns) in band_noise_sum.iter().enumerate() {
        let proportional_floor = (band_bin_count[b].max(1) as f64) * calb_bin_floor;
        noise_var_33_lin[b] = ns.max(proportional_floor) as f32;
    }

    // Best Linear Approximation magnitude across 33 1/3-octave LTAS bands on excited odd bins.
    // `y_mean` averages `p_count` periods, so its residual bin noise variance is `per_bin_noise / p_count`.
    let mut band_x_pow = [0.0_f64; 33];
    let mut band_y_pow = [0.0_f64; 33];
    for &(bin, _) in &plan.excited_odd_bins {
        if bin < n_bins {
            let b = band_idx_for_bin[bin];
            if b < 33 {
                band_x_pow[b] += x_spec[bin].norm_sqr() * parseval_scale;
                let mean_noise = per_bin_noise[bin] / p_count;
                let yp = (y_mean[bin].norm_sqr() * parseval_scale - mean_noise).max(0.0);
                band_y_pow[b] += yp;
            }
        }
    }
    let mut bla_db_opt = [None::<f64>; 33];
    for b in 0..33 {
        if band_x_pow[b] > 1.0e-18 {
            let mag_db = 10.0 * ((band_y_pow[b] / band_x_pow[b]).max(1.0e-12)).log10();
            bla_db_opt[b] = Some(mag_db.clamp(-120.0, 40.0));
        }
    }
    let mut bla_mag_33_db = vec![0.0_f32; 33];
    for b in 0..33 {
        let val = if let Some(v) = bla_db_opt[b] {
            v
        } else {
            let left = (0..b).rev().find_map(|i| bla_db_opt[i].map(|v| (i, v)));
            let right = ((b + 1)..33).find_map(|i| bla_db_opt[i].map(|v| (i, v)));
            match (left, right) {
                (Some((il, vl)), Some((ir, vr))) => {
                    let t = ((b - il) as f64) / ((ir - il) as f64);
                    vl + t * (vr - vl)
                }
                (Some((_, vl)), None) => vl,
                (None, Some((_, vr))) => vr,
                (None, None) => 0.0,
            }
        };
        bla_mag_33_db[b] = val as f32;
    }

    // Even vs Odd nonlinear distortion power separated across unexcited even vs omitted odd bins.
    let mut p_fund = 0.0_f64;
    for &(bin, _) in &plan.excited_odd_bins {
        if bin < n_bins {
            let mean_noise = per_bin_noise[bin] / p_count;
            p_fund += (y_mean[bin].norm_sqr() * parseval_scale - mean_noise).max(0.0);
        }
    }
    let mut p_even_signed = 0.0_f64;
    for &bin in &plan.unexcited_even_bins {
        if bin < n_bins && !plan.masked_bins.contains(&bin) {
            let noise_sub = per_bin_noise[bin] / p_count;
            p_even_signed += y_mean[bin].norm_sqr() * parseval_scale - noise_sub;
        }
    }
    let p_even = p_even_signed.max(0.0);
    let mut p_odd_signed = 0.0_f64;
    for &bin in &plan.omitted_odd_bins {
        if bin < n_bins && !plan.masked_bins.contains(&bin) {
            let noise_sub = per_bin_noise[bin] / p_count;
            p_odd_signed += y_mean[bin].norm_sqr() * parseval_scale - noise_sub;
        }
    }
    let p_odd_omitted = p_odd_signed.max(0.0);
    let n_exc = plan.excited_odd_bins.len() as f64;
    let n_omit = plan.omitted_odd_bins.len().max(1) as f64;
    let p_odd_total = p_odd_omitted * ((n_exc + n_omit) / n_omit);
    let fund_safe = p_fund.max(1.0e-18);
    let even_dist_ratio_db =
        (10.0 * (p_even / fund_safe).max(1.0e-12).log10()).clamp(-120.0, 20.0) as f32;
    let odd_dist_ratio_db =
        (10.0 * (p_odd_total / fund_safe).max(1.0e-12).log10()).clamp(-120.0, 20.0) as f32;

    let total_noise_lin = noise_var_33_lin
        .iter()
        .map(|&v| f64::from(v))
        .sum::<f64>()
        .max(calb_total_lin);
    let edc_t60_s = compute_schroeder_edc_t60_s(
        regime2.wet_l,
        regime2.wet_r,
        regime2.gap_ranges,
        CANONICAL_TRAINER_SAMPLE_RATE_HZ,
        total_noise_lin,
    );

    SchoukensPhysicsResult {
        bla_mag_33_db,
        even_dist_ratio_db,
        odd_dist_ratio_db,
        noise_var_33_lin,
        edc_t60_s,
        law0a_ltv_flag,
        coreset_regime_offsets: config.coreset_regime_offsets,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::f64::consts::TAU;

    use super::*;
    use crate::stimulus::fft::ADC_PEAK_GOVERNOR_FS;
    use crate::stimulus::pipeline::schoukens::regimes::{
        SCHOUKENS_CORESET_25S_OFFSETS, build_odd_random_multisine_plan,
        synthesize_odd_random_multisine_period,
    };

    #[test]
    fn test_schoukens_bla_even_odd_separation_law0a_ltv_and_hum_exclusion() {
        let seed = [0x21_u8; 256];
        let plan = build_odd_random_multisine_plan(&seed, &BTreeSet::new());
        let (x_per, _) =
            synthesize_odd_random_multisine_period(&plan, &seed, 0, ADC_PEAK_GOVERNOR_FS);
        let cfg = SchoukensAnalysisConfig {
            calb_noise_floor_dbfs: -100.0,
            coreset_regime_offsets: SCHOUKENS_CORESET_25S_OFFSETS,
        };

        let no_r2 = Regime2DecayInput::default();

        // 1. Clean linear gain = 0.5 (-6.02 dB)
        let y_lin: Vec<f32> = x_per.iter().map(|&x| 0.5 * x).collect();
        let lin_res = analyze_schoukens_pintelon_physics(
            &plan,
            &x_per,
            &[&y_lin, &y_lin, &y_lin],
            None,
            no_r2,
            cfg,
        );
        assert!(!lin_res.law0a_ltv_flag);
        assert!(
            lin_res.even_dist_ratio_db < -60.0,
            "even={}",
            lin_res.even_dist_ratio_db
        );
        assert!(
            lin_res.odd_dist_ratio_db < -60.0,
            "odd={}",
            lin_res.odd_dist_ratio_db
        );
        for (b, &m_db) in lin_res.bla_mag_33_db.iter().enumerate() {
            assert!((m_db + 6.02).abs() < 0.25, "band {b}: bla={m_db}");
        }

        // 2. Odd symmetric saturator tanh(3.5x) -> dominant odd distortion, suppressed even distortion
        let y_odd: Vec<f32> = x_per
            .iter()
            .map(|&x| (3.5 * f64::from(x)).tanh() as f32)
            .collect();
        let odd_res =
            analyze_schoukens_pintelon_physics(&plan, &x_per, &[&y_odd, &y_odd], None, no_r2, cfg);
        assert!(odd_res.odd_dist_ratio_db > -30.0);
        assert!(odd_res.even_dist_ratio_db < odd_res.odd_dist_ratio_db - 20.0);

        // 3. Even asymmetric shaper x + 0.35 * x^2 -> dominant even distortion, suppressed odd
        let y_even: Vec<f32> = x_per
            .iter()
            .map(|&x| {
                let xf = f64::from(x);
                (xf + 0.35 * xf * xf) as f32
            })
            .collect();
        let even_res = analyze_schoukens_pintelon_physics(
            &plan,
            &x_per,
            &[&y_even, &y_even],
            None,
            no_r2,
            cfg,
        );
        assert!(even_res.even_dist_ratio_db > -30.0);
        assert!(even_res.odd_dist_ratio_db < even_res.even_dist_ratio_db - 20.0);

        // 4. Law 0.A LTV detection: continuous pitch/delay modulation breaks period repeatability
        let p0 = x_per.clone();
        let p1: Vec<f32> = (0..x_per.len())
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let g = 0.65 + 0.35 * (TAU * 0.7 * t).sin();
                x_per[i] * (g as f32)
            })
            .collect();
        let ltv_res =
            analyze_schoukens_pintelon_physics(&plan, &x_per, &[&p0, &p1], None, no_r2, cfg);
        assert!(ltv_res.law0a_ltv_flag);
        // Fallback noise floor stays bounded near CALB/guard floor instead of blowing up from LTV diff
        let total_nv: f64 = ltv_res.noise_var_33_lin.iter().map(|&v| f64::from(v)).sum();
        assert!(
            total_nv < 1.0e-5,
            "unexpected LTV noise variance blowup {total_nv}"
        );

        // 5. Law 0.C continuous-phase -35 dBFS 60 Hz hum across multiple periods does NOT trip LTV
        // (even though N_per = 16,384 spans 20.48 cycles / ~173 deg phase rotation per period!).
        let hum_amp = 10.0_f64.powf(-35.0 / 20.0);
        let synth_hum_period = |p_idx: usize| -> Vec<f32> {
            (0..x_per.len())
                .map(|i| {
                    let g = (p_idx * x_per.len() + i) as f64;
                    x_per[i] + (hum_amp * (TAU * 60.0 * g / 48_000.0).sin()) as f32
                })
                .collect()
        };
        let p_hum_0 = synth_hum_period(0);
        let p_hum_1 = synth_hum_period(1);
        let p_hum_2 = synth_hum_period(2);
        let hum_res = analyze_schoukens_pintelon_physics(
            &plan,
            &x_per,
            &[&p_hum_0, &p_hum_1, &p_hum_2],
            None,
            no_r2,
            cfg,
        );
        assert!(!hum_res.law0a_ltv_flag);
        assert!(f64::from(hum_res.noise_var_33_lin[4]) < 1.0e-6);

        // 6. LTI noisy target (-55 dBFS white hiss across 3 periods): subtracted properly so
        // quiescent broadband hiss is NOT misclassified as even/odd distortion (Critic C7).
        let mut rng_state = 0x1357_2468_9BDF_ECA0_u64;
        let mut synth_noisy_period = || -> Vec<f32> {
            let mut out = Vec::with_capacity(x_per.len());
            for &x in &x_per {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 32) as f64) / f64::from(u32::MAX) - 0.5;
                out.push(x + (0.004 * u) as f32);
            }
            out
        };
        let pn0 = synth_noisy_period();
        let pn1 = synth_noisy_period();
        let pn2 = synth_noisy_period();
        let noisy_lti = analyze_schoukens_pintelon_physics(
            &plan,
            &x_per,
            &[&pn0, &pn1, &pn2],
            None,
            no_r2,
            cfg,
        );
        assert!(!noisy_lti.law0a_ltv_flag);
        assert!(
            noisy_lti.even_dist_ratio_db < -50.0 && noisy_lti.odd_dist_ratio_db < -50.0,
            "noise leaked into distortion: even={} odd={}",
            noisy_lti.even_dist_ratio_db,
            noisy_lti.odd_dist_ratio_db
        );

        // 7. Single-period nonlinear odd saturator still preserves odd distortion without cancelling.
        let single_odd =
            analyze_schoukens_pintelon_physics(&plan, &x_per, &[&y_odd], None, no_r2, cfg);
        assert!(single_odd.odd_dist_ratio_db > -30.0);
        assert!(single_odd.even_dist_ratio_db < single_odd.odd_dist_ratio_db - 15.0);

        // 8. Default config, short-period fallback, and all 4 BLA sparse interpolation arms.
        let def_cfg = SchoukensAnalysisConfig::default();
        assert_eq!(def_cfg.calb_noise_floor_dbfs, -100.0);
        let mut short_plan = plan.clone();
        short_plan.excited_odd_bins.clear();
        let all_none =
            analyze_schoukens_pintelon_physics(&short_plan, &[], &[], None, no_r2, def_cfg);
        assert_eq!(all_none.bla_mag_33_db, vec![0.0_f32; 33]);

        // Sparse excited bins only around ~500 Hz (b~14) & ~2 kHz (b~20) cover Left-None, Mid, Right-None.
        let mut sparse_plan = plan.clone();
        sparse_plan.excited_odd_bins = plan
            .excited_odd_bins
            .iter()
            .copied()
            .filter(|&(b, _)| (b > 165 && b < 180) || (b > 675 && b < 695))
            .collect();
        let sparse_res = analyze_schoukens_pintelon_physics(
            &sparse_plan,
            &x_per,
            &[&x_per],
            None,
            no_r2,
            def_cfg,
        );
        assert_eq!(sparse_res.bla_mag_33_db.len(), 33);
        assert!(sparse_res.bla_mag_33_db.iter().all(|v| v.is_finite()));

        // 9. Unbiased Exp(1) guard-bin noise estimator recovers true variance within +/-2.0 dB
        // and 3-period LTI -40 dBFS white hiss cancels without distortion leakage.
        let mut gauss_rng = 0xD1A3_1234_5678_9ABC_u64;
        let next_gauss = |st: &mut u64, sigma: f64| -> f32 {
            let mut next_u = || {
                *st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((*st >> 32) as f64) + 0.5) / (f64::from(u32::MAX) + 1.0)
            };
            let u1 = next_u().max(1.0e-12);
            let u2 = next_u();
            (sigma * (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()) as f32
        };
        let n_per = plan.period_samples;
        let parseval_scale = 2.0 / ((n_per as f64) * (n_per as f64));
        let sigma_n = 0.01_f64; // -40 dBFS RMS white noise
        let true_bin_var = (2.0 * sigma_n * sigma_n) / (n_per as f64);
        let gauss_per: Vec<f32> = (0..n_per)
            .map(|_| next_gauss(&mut gauss_rng, sigma_n))
            .collect();
        let est_bin_var =
            estimate_unexcited_guard_bin_noise_floor(&plan, &gauss_per, parseval_scale, 1.0e-18);
        let err_db = 10.0 * (est_bin_var / true_bin_var).log10();
        assert!(err_db.abs() < 2.0, "guard noise bias={err_db} dB");

        let g0: Vec<f32> = x_per
            .iter()
            .map(|&x| x + next_gauss(&mut gauss_rng, sigma_n))
            .collect();
        let g1: Vec<f32> = x_per
            .iter()
            .map(|&x| x + next_gauss(&mut gauss_rng, sigma_n))
            .collect();
        let g2: Vec<f32> = x_per
            .iter()
            .map(|&x| x + next_gauss(&mut gauss_rng, sigma_n))
            .collect();
        let hiss_res =
            analyze_schoukens_pintelon_physics(&plan, &x_per, &[&g0, &g1, &g2], None, no_r2, cfg);
        assert!(!hiss_res.law0a_ltv_flag);
        assert!(
            hiss_res.even_dist_ratio_db < -50.0 && hiss_res.odd_dist_ratio_db < -50.0,
            "-40 dBFS hiss leaked: even={} odd={}",
            hiss_res.even_dist_ratio_db,
            hiss_res.odd_dist_ratio_db
        );
    }
}
