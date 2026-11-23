//! Pass-1 dry `Op-P` quadrature probe, data-driven [`Pass1PhysicsHint`], signed dry-vs-wet
//! `HilbertH1StereoKernel` SSB demodulation (`[-0.49*fs, +0.49*fs]`), `Op-N[Pre]` quantization /
//! ZOH plateau estimation, `SensoryQuad` envelope sensitivity, and even-order `rect_sign` seeding
//! (`ref[train.pipeline.physics-probe]`).

use core::f64::consts::TAU;

use dnam_kernels::{HilbertH1StereoKernel, PhasorQ32Bank4, T1DcBlockerStereo, T1SubHarmonicKernel};

use super::super::model::SonaModelF64;

/// Summary returned by the `< 0.5 ms` CPU `Op-P` quadrature & pitch probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpPProbeSummary {
    /// Detected narrowband dominant carrier frequency in Hz (`0.0` if broadband).
    pub dominant_freq_hz: f64,
    /// Normalized pitch periodicity confidence `[0.0, 1.0]`.
    pub pitch_confidence: f64,
    /// Mean stereo quadrature analytic energy `mean(I^2 + Q^2)`.
    pub analytic_energy: f64,
    /// Detected sub-harmonic octaver activity ratio `[0.0, 1.0]`.
    pub subharmonic_activity: f64,
}

/// Optional 100% data-driven scalar physics hints passed from Stage A or manifest metadata.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pass1PhysicsHint {
    /// Estimated reverb / tail RT60 decay time in seconds.
    pub rt60_s: Option<f64>,
    /// Dominant echo / delay time in milliseconds.
    pub delay_ms: Option<f64>,
    /// Estimated total harmonic distortion ratio.
    pub thd_ratio: Option<f64>,
    /// Signed even-vs-odd harmonic asymmetry (`+` positive half-wave bias, `-` negative).
    pub even_odd_asymmetry: Option<f64>,
    /// Detected LFO / chorus modulation rate in Hz.
    pub modulation_rate_hz: Option<f64>,
    /// Estimated repeatability noise floor in dBFS.
    pub noise_floor_db: Option<f64>,
    /// Whether significant true-stereo decorrelation was measured.
    pub has_stereo_width: Option<bool>,
}

/// Executes the `< 0.5 ms` CPU `Op-P` probe across up to `4,096` dry stereo samples using
/// `dnam-kernels` `T1DcBlockerStereo`, `HilbertH1StereoKernel`, `T1SubHarmonicKernel`, and
/// `PhasorQ32Bank4` (`ref[train.pipeline.physics-probe]`).
#[must_use]
pub fn run_cpu_op_p_probe(
    input_l: &[f64],
    input_r: &[f64],
    sample_rate_hz: f64,
) -> OpPProbeSummary {
    let n = input_l.len().min(input_r.len()).min(4_096);
    if n < 32 {
        return OpPProbeSummary {
            dominant_freq_hz: 0.0,
            pitch_confidence: 0.0,
            analytic_energy: 0.0,
            subharmonic_activity: 0.0,
        };
    }
    let mut dc = T1DcBlockerStereo::new();
    let mut hilb = HilbertH1StereoKernel::new();
    let mut t1 = T1SubHarmonicKernel::new();
    let mut phasor = PhasorQ32Bank4::from_freqs_hz(
        [110.0, 220.0, 440.0, 880.0],
        1.0 / (sample_rate_hz.max(8_000.0) as f32),
    );

    let mut e_analytic = 0.0_f64;
    let mut e_sub = 0.0_f64;
    let mut dphase_cos_acc = 0.0_f64;
    let mut dphase_sin_acc = 0.0_f64;
    let mut prev_re = 1.0_f64;
    let mut prev_im = 0.0_f64;

    for i in 0..n {
        let (cl, cr) = dc.step_stereo(input_l[i] as f32, input_r[i] as f32);
        let ((il, ql), (ir, qr)) = hilb.step_stereo(cl, cr);
        let im = 0.5 * (il + ir);
        let qm = 0.5 * (ql + qr);
        let feat = t1.step(im, qm);
        let (p_cos, _p_sin) = phasor.step_quadrature();
        let _carrier_probe = f64::from(p_cos[0]);

        let il64 = f64::from(il);
        let ql64 = f64::from(ql);
        let ir64 = f64::from(ir);
        let qr64 = f64::from(qr);
        e_analytic += 0.5 * (il64 * il64 + ql64 * ql64 + ir64 * ir64 + qr64 * qr64);
        e_sub += f64::from(feat[0] * feat[0] + feat[1] * feat[1]);

        let im64 = f64::from(im);
        let qm64 = f64::from(qm);
        let mid_norm = (im64 * im64 + qm64 * qm64).sqrt().max(1.0e-12);
        let cur_re = im64 / mid_norm;
        let cur_im = qm64 / mid_norm;
        if i > 0 {
            dphase_cos_acc += cur_re * prev_re + cur_im * prev_im;
            dphase_sin_acc += cur_im * prev_re - cur_re * prev_im;
        }
        prev_re = cur_re;
        prev_im = cur_im;
    }
    let mean_an = e_analytic / (n as f64);
    let mean_sub = e_sub / (n as f64);
    let r_vec = (dphase_cos_acc * dphase_cos_acc + dphase_sin_acc * dphase_sin_acc).sqrt()
        / ((n - 1) as f64);
    let pitch_confidence = r_vec.clamp(0.0, 1.0);
    let omega = dphase_sin_acc.atan2(dphase_cos_acc).abs();
    let dominant_freq_hz = if pitch_confidence > 0.35 {
        omega * sample_rate_hz / TAU
    } else {
        0.0
    };
    OpPProbeSummary {
        dominant_freq_hz,
        pitch_confidence,
        analytic_energy: mean_an,
        subharmonic_activity: (mean_sub / mean_an.max(1.0e-12)).clamp(0.0, 1.0),
    }
}

/// Estimates signed SSB frequency translation (`[-0.49 * fs, +0.49 * fs]`) between dry and wet
/// stereo waveforms using `HilbertH1StereoKernel` analytic cross-product `z_wet * conj(z_dry)`.
#[must_use]
pub(super) fn estimate_signed_ssb_shift_hz(
    input_l: &[f64],
    input_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    sample_rate_hz: f64,
) -> Option<f64> {
    let n = input_l
        .len()
        .min(input_r.len())
        .min(target_l.len())
        .min(target_r.len())
        .min(16_384);
    if n < 64 {
        return None;
    }
    let mut dc_x = T1DcBlockerStereo::new();
    let mut dc_y = T1DcBlockerStereo::new();
    let mut hilb_x = HilbertH1StereoKernel::new();
    let mut hilb_y = HilbertH1StereoKernel::new();

    let (mut prev_re, mut prev_im, mut prev_mag) = (1.0_f64, 0.0_f64, 0.0_f64);
    let (mut acc_re, mut acc_im, mut weight_sum) = (0.0_f64, 0.0_f64, 0.0_f64);
    let warmup = 32_usize.min(n / 4);

    for t in 0..n {
        let (cxl, cxr) = dc_x.step_stereo(input_l[t] as f32, input_r[t] as f32);
        let (cyl, cyr) = dc_y.step_stereo(target_l[t] as f32, target_r[t] as f32);
        let ((ixl, qxl), (ixr, qxr)) = hilb_x.step_stereo(cxl, cxr);
        let ((iyl, qyl), (iyr, qyr)) = hilb_y.step_stereo(cyl, cyr);

        let zx_r = 0.5 * f64::from(ixl + ixr);
        let zx_i = 0.5 * f64::from(qxl + qxr);
        let zy_r = 0.5 * f64::from(iyl + iyr);
        let zy_i = 0.5 * f64::from(qyl + qyr);

        // z_wet * conj(z_dry) = (zy_r + j zy_i)(zx_r - j zx_i)
        let cr = zy_r * zx_r + zy_i * zx_i;
        let ci = zy_i * zx_r - zy_r * zx_i;
        let mag = cr.hypot(ci);
        if t > warmup && mag > 1.0e-12 && prev_mag > 1.0e-12 {
            // c[t] * conj(c[t-1]) accumulates phase increment exp(j * delta_omega).
            acc_re += cr * prev_re + ci * prev_im;
            acc_im += ci * prev_re - cr * prev_im;
            weight_sum += mag * prev_mag;
        }
        prev_re = cr;
        prev_im = ci;
        prev_mag = mag;
    }

    if weight_sum <= 1.0e-12 {
        return None;
    }
    let coh = acc_re.hypot(acc_im) / weight_sum;
    let delta_hz = acc_im.atan2(acc_re) * sample_rate_hz / TAU;
    let max_hz = 0.49 * sample_rate_hz;
    // Hilbert H1 has +-90 deg quadrature only across audio passband; ignore near-zero static shifts.
    if coh > 0.55 && delta_hz.abs() >= 1.5 && delta_hz.abs() <= max_hz {
        Some(delta_hz.clamp(-max_hz, max_hz))
    } else {
        None
    }
}

/// Borrowed dry/wet/residual stereo slices and post-warmup start index for front-end seeding.
pub(super) struct TelemetrySeedSlices<'a> {
    pub input_l: &'a [f64],
    pub input_r: &'a [f64],
    pub target_l: &'a [f64],
    pub target_r: &'a [f64],
    pub res_l: &'a [f64],
    pub res_r: &'a [f64],
    pub start: usize,
}

/// Seeds `Op-N[Pre]`, `SensoryQuad`, `phasor_freq_hz`, and even-order `rect_sign` polarity from
/// aligned dry/wet telemetry without inventing spurious quantization on clean signals.
pub(super) fn seed_telemetry_and_front_end(
    model: &mut SonaModelF64,
    op_p: &OpPProbeSummary,
    slices: &TelemetrySeedSlices<'_>,
    hint: Option<&Pass1PhysicsHint>,
) {
    let TelemetrySeedSlices {
        input_l,
        input_r,
        target_l,
        target_r,
        res_l,
        res_r,
        start,
    } = *slices;
    let n = input_l
        .len()
        .min(input_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let sr = model.sample_rate_hz.max(8_000.0);
    let max_phasor_hz = 0.49 * sr;

    // 1. Signed phasor frequency from SSB demodulation, falling back to Op-P dominant frequency.
    if let Some(ssb_hz) = estimate_signed_ssb_shift_hz(
        &input_l[..n],
        &input_r[..n],
        &target_l[..n],
        &target_r[..n],
        sr,
    ) {
        model.phasor_freq_hz[0] = ssb_hz;
    } else if op_p.dominant_freq_hz > 20.0 {
        model.phasor_freq_hz[0] = op_p.dominant_freq_hz.clamp(-max_phasor_hz, max_phasor_hz);
    }

    // 2. Even-order polarity -> discrete CoreQuad rect_sign seeding.
    let asym = if let Some(h) = hint.and_then(|v| v.even_odd_asymmetry) {
        h
    } else if n > start + 8 {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for t in start..n {
            let xm = 0.5 * (input_l[t] + input_r[t]);
            let ym_res = 0.5 * (res_l[t] + res_r[t]);
            let x2 = xm * xm;
            num += ym_res * x2;
            den += x2 * x2;
        }
        if den > 1.0e-14 { num / den } else { 0.0 }
    } else {
        0.0
    };
    if asym.abs() > 1.0e-3 {
        let sgn = if asym < 0.0 { -1.0 } else { 1.0 };
        model.core_quads[0].rect_sign = [sgn, sgn, 1.0, -1.0];
    }

    // 3. Zero-bias data-driven Op-N[Pre] ZOH hold & bit-depth telemetry (0.0 on clean audio).
    model.op_n_pre_gamma_bc = 0.0;
    model.op_n_pre_zoh_hold_depth = 0.0;
    model.op_n_pre_beta_q = 0.0;
    if n > start + 64 {
        let m = (n - start).min(8_192);
        let y_slice = &target_l[start..start + m];
        let x_slice = &input_l[start..start + m];
        let y_rms = (y_slice.iter().map(|&v| v * v).sum::<f64>() / (m as f64)).sqrt();
        let x_rms = (x_slice.iter().map(|&v| v * v).sum::<f64>() / (m as f64)).sqrt();
        if y_rms > 1.0e-6 && x_rms > 1.0e-6 {
            let eps_flat = 1.0e-6 * y_rms;
            let y_flat_runs = y_slice
                .windows(2)
                .filter(|w| (w[1] - w[0]).abs() <= eps_flat)
                .count() as f64
                / ((m - 1) as f64);
            let x_flat_runs = x_slice
                .windows(2)
                .filter(|w| (w[1] - w[0]).abs() <= 1.0e-6 * x_rms)
                .count() as f64
                / ((m - 1) as f64);
            let plateau_excess = y_flat_runs - x_flat_runs;
            if plateau_excess > 0.35 && y_flat_runs > 0.40 {
                // Only seed ZOH hold depth when crisp sample-and-hold plateaus occur in wet vs dry.
                let hold_ratio =
                    (1.0 - (1.0 - plateau_excess).clamp(1.0e-4, 1.0)).clamp(0.1, 0.999);
                model.op_n_pre_zoh_hold_depth = hold_ratio;
                model.op_n_pre_gamma_bc = 0.5;
            }
            // Detect coarse staircase quantization only when zero-diff plateaus ALSO accompany discrete step clusters.
            if plateau_excess > 0.25 && y_flat_runs > 0.30 {
                let mut active_diffs = Vec::with_capacity(m);
                for w in y_slice.windows(2) {
                    let d = (w[1] - w[0]).abs();
                    if d > eps_flat {
                        active_diffs.push(d);
                    }
                }
                if active_diffs.len() > 16 {
                    active_diffs.sort_by(f64::total_cmp);
                    let step_est = active_diffs[active_diffs.len() / 8];
                    if step_est > 0.02 * y_rms {
                        let bits_est = (1.0 / step_est.max(1.0e-6)).log2().clamp(1.0, 12.0);
                        model.op_n_pre_beta_q = bits_est;
                        model.op_n_pre_gamma_bc = model.op_n_pre_gamma_bc.max(0.5);
                    }
                }
            }
        }
    }

    // 4. Data-driven SensoryQuad envelope/filter init (only on genuine slow dynamic range
    // modulation so static waveshapers never throttle input gain before Pass 2 optimizes VCA).
    if n > start + 256 {
        let m_win = n.min(start + 16_384);
        let mut dry_env = 0.0_f64;
        let mut wet_env = 0.0_f64;
        let mut dry_samples = Vec::with_capacity((m_win - start) / 64 + 1);
        let (mut sum_x, mut sum_x2, mut sum_xy, mut sum_y, mut cnt) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for t in start..m_win {
            let ax = 0.5 * (input_l[t].abs() + input_r[t].abs());
            let ay = 0.5 * (target_l[t].abs() + target_r[t].abs());
            dry_env = 0.99 * dry_env + 0.01 * ax;
            wet_env = 0.99 * wet_env + 0.01 * ay;
            if t >= start + 64 && dry_env > 1.0e-4 {
                if (t - start).is_multiple_of(64) {
                    dry_samples.push(dry_env);
                }
                let log_g = (wet_env / dry_env).clamp(0.05, 4.0) - 1.0;
                sum_x += dry_env;
                sum_x2 += dry_env * dry_env;
                sum_xy += dry_env * log_g;
                sum_y += log_g;
                cnt += 1.0;
            }
        }
        dry_samples.sort_by(f64::total_cmp);
        let dynamic_ratio = if dry_samples.len() >= 10 {
            let p10 = dry_samples[dry_samples.len() / 10].max(1.0e-6);
            let p90 = dry_samples[(dry_samples.len() * 9) / 10];
            p90 / p10
        } else {
            1.0
        };
        if cnt > 64.0 && dynamic_ratio > 1.6 {
            let mx = sum_x / cnt;
            let my = sum_y / cnt;
            let var_x = (sum_x2 / cnt - mx * mx).max(1.0e-12);
            let std_x = var_x.sqrt();
            let cov = sum_xy / cnt - mx * my;
            let slope = cov / var_x;
            let frac_swing = (slope * std_x).abs();
            if frac_swing > 0.12 && slope.is_finite() {
                let freqs = [120.0, 500.0, 2_000.0, 7_500.0];
                for (i, &f_hz) in freqs.iter().enumerate() {
                    model.sense_svf_g[i] =
                        (core::f64::consts::PI * f_hz / sr).tan().clamp(1.0e-4, 8.0);
                    model.sense_tone_lp[i] = 1.0;
                    model.sense_tone_bp[i] = 0.25;
                }
                model.sense_vca_w = [(0.25 * slope).clamp(-0.35, 0.35), 0.0, 0.0, 0.0];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssb_signed_demodulation_and_op_n_pre_and_rect_sign_telemetry() {
        let n = 4_096_usize;
        let sr = 48_000.0_f64;
        let dry_l: Vec<f64> = (0..n)
            .map(|t| 0.4 * (TAU * 1_000.0 * (t as f64) / sr).cos())
            .collect();
        let dry_r: Vec<f64> = dry_l.clone();

        // Positive +150 Hz and negative -220 Hz single-sideband frequency shift.
        for shift_hz in [150.0_f64, -220.0_f64] {
            let wet_l: Vec<f64> = (0..n)
                .map(|t| 0.4 * (TAU * (1_000.0 + shift_hz) * (t as f64) / sr).cos())
                .collect();
            let est = estimate_signed_ssb_shift_hz(&dry_l, &dry_r, &wet_l, &wet_l, sr)
                .expect("expected signed SSB shift estimate");
            assert!(
                (est - shift_hz).abs() < 8.0,
                "SSB shift mismatch: est={est} expected={shift_hz}"
            );
        }

        // Static non-shifted signal yields None (leaves phasor_freq_hz untouched or 0).
        assert!(estimate_signed_ssb_shift_hz(&dry_l, &dry_r, &dry_l, &dry_r, sr).is_none());

        // ZOH staircase signal triggers op_n_pre_zoh_hold_depth > 0, clean audio leaves it 0.0.
        let mut zoh_y = vec![0.0_f64; n];
        for t in 0..n {
            zoh_y[t] = (4.0 * dry_l[(t / 8) * 8]).round() / 4.0;
        }
        let mut mdl_zoh = SonaModelF64::new(sr);
        let op_p = run_cpu_op_p_probe(&dry_l, &dry_r, sr);
        seed_telemetry_and_front_end(
            &mut mdl_zoh,
            &op_p,
            &TelemetrySeedSlices {
                input_l: &dry_l,
                input_r: &dry_r,
                target_l: &zoh_y,
                target_r: &zoh_y,
                res_l: &zoh_y,
                res_r: &zoh_y,
                start: 32,
            },
            None,
        );
        assert!(mdl_zoh.op_n_pre_zoh_hold_depth > 0.2);
        assert!(mdl_zoh.op_n_pre_gamma_bc > 0.0);

        let mut mdl_clean = SonaModelF64::new(sr);
        let zero_res = vec![0.0_f64; n];
        seed_telemetry_and_front_end(
            &mut mdl_clean,
            &op_p,
            &TelemetrySeedSlices {
                input_l: &dry_l,
                input_r: &dry_r,
                target_l: &dry_l,
                target_r: &dry_r,
                res_l: &zero_res,
                res_r: &zero_res,
                start: 32,
            },
            None,
        );
        assert_eq!(mdl_clean.op_n_pre_gamma_bc, 0.0);
        assert_eq!(mdl_clean.op_n_pre_zoh_hold_depth, 0.0);
        assert_eq!(mdl_clean.op_n_pre_beta_q, 0.0);

        // Even-order negative polarity detection sets rect_sign[0] = -1.0.
        let mut even_res = vec![0.0_f64; n];
        for t in 0..n {
            even_res[t] = -0.35 * dry_l[t] * dry_l[t];
        }
        let mut mdl_neg = SonaModelF64::new(sr);
        seed_telemetry_and_front_end(
            &mut mdl_neg,
            &op_p,
            &TelemetrySeedSlices {
                input_l: &dry_l,
                input_r: &dry_r,
                target_l: &dry_l,
                target_r: &dry_r,
                res_l: &even_res,
                res_r: &even_res,
                start: 32,
            },
            None,
        );
        assert_eq!(mdl_neg.core_quads[0].rect_sign[0], -1.0);
    }
}
