//! Phase-3 `[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`, and `[NUMERIC-ENVELOPE]` audit
//! (`ref[train.audit.phase3.clamp-dynamic]`, `ref[train.audit.phase3.clamp-head-gain]`,
//! `ref[train.audit.phase3.numeric-envelope]`).

use core::f64::consts::PI;

use dnam_kernels::{MAX_PARALRU_R_F32, TANK_MAX_G_LIN_F32};

use crate::error::TrainError;

use super::super::stability::{
    MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP, MIN_DELAY_STANDALONE_SAMP, RAIL_OUTPUT_CEIL,
};
use super::{MAX_GAIN_6DB_LIN, MAX_GAIN_IDEMPOTENT, Phase3AuditInput, Phase3AuditReport};

/// Evaluates L/R spectral magnitude response of the unity-normalized `ro_pole_48k` readout
/// transfer `H(e^{jw}) = g_dir + g_hp * ro * (1 - e^{-jw}) / (1 - ro * e^{-jw})` over `[20 Hz..Nyquist*0.9]`
/// so callers can feed exact `freq_mag_{l,r}` into [`audit_phase3_dynamic_and_head_gain`].
#[must_use]
pub fn evaluate_readout_spectral_magnitudes(
    gains: &[f64],
    ro_pole_48k: f64,
    sample_rate_hz: f64,
    num_freqs: usize,
) -> Vec<f64> {
    let ro = if ro_pole_48k.is_finite() {
        ro_pole_48k.clamp(0.0, MAX_RO_POLE_48K_F64)
    } else {
        0.0
    };
    let g_dir = gains.first().copied().unwrap_or(0.0)
        + gains.get(2..).map_or(0.0, |tail| tail.iter().sum::<f64>());
    let g_hp = gains.get(1).copied().unwrap_or(0.0);
    let n_f = num_freqs.max(8);
    let mut mags = Vec::with_capacity(n_f);
    for k in 0..n_f {
        let frac = (k as f64) / ((n_f - 1) as f64);
        let f_hz = 20.0 * (sample_rate_hz * 0.45 / 20.0).powf(frac);
        let w = 2.0 * PI * f_hz / sample_rate_hz;
        let (s, c) = w.sin_cos();
        let nr = ro * (1.0 - c);
        let ni = ro * s;
        let dr = 1.0 - ro * c;
        let di = ro * s;
        let denom = (dr * dr + di * di).max(1.0e-18);
        let hp_re = (nr * dr + ni * di) / denom;
        let hp_im = (ni * dr - nr * di) / denom;
        let h_re = g_dir + g_hp * hp_re;
        let h_im = g_hp * hp_im;
        mags.push((h_re * h_re + h_im * h_im).sqrt());
    }
    mags
}

/// Runs the Phase-3 `[CLAMP-DYNAMIC]`, `[CLAMP-HEAD-GAIN]`, and `[NUMERIC-ENVELOPE]` audit
/// (`ref[train.audit.phase3.clamp-dynamic]`, `ref[train.audit.phase3.clamp-head-gain]`), failing
/// closed if any field is non-finite or `burst_peak_abs > 100.0`.
pub fn audit_phase3_dynamic_and_head_gain(
    input: &mut Phase3AuditInput,
) -> Result<Phase3AuditReport, TrainError> {
    if !input.burst_peak_abs.is_finite()
        || input.burst_peak_abs.abs() > f64::from(RAIL_OUTPUT_CEIL)
        || !input.min_delay_samples.is_finite()
        || !input.max_delay_excursion.is_finite()
        || !input.max_paralru_r_excursion.is_finite()
        || !input.max_tank_g_excursion.is_finite()
        || !input.ro_pole_48k.is_finite()
    {
        return Err(TrainError::AuditViolation {
            context: "Phase-3 [NUMERIC-ENVELOPE]",
            detail: "non-finite Phase-3 parameter or burst peak exceeding +/-100.0",
        });
    }
    if !input.readout_gains_l.iter().all(|v| v.is_finite())
        || !input.readout_gains_r.iter().all(|v| v.is_finite())
    {
        return Err(TrainError::AuditViolation {
            context: "Phase-3 [CLAMP-HEAD-GAIN]",
            detail: "non-finite readout gain coefficient",
        });
    }

    let mut warnings = Vec::new();
    let delay_floor = if input.crossbar_coupled {
        MIN_DELAY_CROSSBAR_SAMP
    } else {
        MIN_DELAY_STANDALONE_SAMP
    };
    let mut tightened_min = input.min_delay_samples.clamp(delay_floor, 48_000.0);
    let mut exc = input.max_delay_excursion.abs();
    if tightened_min - exc < delay_floor {
        let req_min = delay_floor + exc;
        if req_min <= 48_000.0 {
            tightened_min = req_min;
        } else {
            exc = (tightened_min - delay_floor).max(0.0);
        }
        warnings.push(format!(
            "[CLAMP-DYNAMIC] delay tightened min -> {tightened_min:.2}, exc -> {exc:.2} (trough >= {delay_floor:.1})"
        ));
    }
    input.min_delay_samples = tightened_min;
    input.max_delay_excursion = exc;
    input.max_paralru_r_excursion = input
        .max_paralru_r_excursion
        .clamp(0.0, f64::from(MAX_PARALRU_R_F32));
    input.max_tank_g_excursion = input
        .max_tank_g_excursion
        .clamp(0.0, f64::from(TANK_MAX_G_LIN_F32));
    input.ro_pole_48k = input.ro_pole_48k.clamp(0.0, MAX_RO_POLE_48K_F64 as f32);

    let mut peak_before = 0.0_f64;
    for &m in input.freq_mag_l.iter().chain(&input.freq_mag_r) {
        if !m.is_finite() {
            return Err(TrainError::AuditViolation {
                context: "Phase-3 [CLAMP-HEAD-GAIN]",
                detail: "non-finite spectral gain sample",
            });
        }
        peak_before = peak_before.max(m.abs());
    }

    let mut scale = 1.0_f64;
    let peak_after = if peak_before > MAX_GAIN_6DB_LIN {
        scale = MAX_GAIN_IDEMPOTENT / peak_before;
        for g in input
            .readout_gains_l
            .iter_mut()
            .chain(&mut input.readout_gains_r)
        {
            *g *= scale;
        }
        for m in input.freq_mag_l.iter_mut().chain(&mut input.freq_mag_r) {
            *m *= scale;
        }
        let recomputed = input
            .freq_mag_l
            .iter()
            .chain(&input.freq_mag_r)
            .copied()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        warnings.push(format!(
            "[CLAMP-HEAD-GAIN] peak {peak_before:.6} > +6 dB ({MAX_GAIN_6DB_LIN:.6}) -> scaled x{scale:.8} (after {recomputed:.6})"
        ));
        recomputed
    } else {
        peak_before
    };

    Ok(Phase3AuditReport {
        tightened_min_delay: tightened_min,
        clamped_delay_excursion: exc,
        ro_pole_48k: input.ro_pole_48k,
        head_gain_scale: scale,
        peak_gain_before: peak_before,
        peak_gain_after: peak_after,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase3_head_gain_idempotence_and_nan_rejection() {
        let gains = vec![2.4_f64, 0.2_f64];
        let mags_l = evaluate_readout_spectral_magnitudes(&gains, 0.1, 48_000.0, 64);
        let mut p3 = Phase3AuditInput {
            min_delay_samples: 6.0,
            max_delay_excursion: 2.5,
            max_paralru_r_excursion: 0.99999,
            max_tank_g_excursion: 0.99,
            crossbar_coupled: true,
            ro_pole_48k: 1.0,
            readout_gains_l: gains,
            readout_gains_r: vec![1.5],
            freq_mag_l: mags_l,
            freq_mag_r: vec![1.5; 64],
            burst_peak_abs: 12.0,
        };
        let r1 = audit_phase3_dynamic_and_head_gain(&mut p3).unwrap();
        assert!(r1.head_gain_scale < 1.0 && r1.peak_gain_after <= MAX_GAIN_6DB_LIN);
        assert!(r1.tightened_min_delay >= 7.5 && r1.ro_pole_48k < 1.0);

        let gains_snapshot = p3.readout_gains_l.clone();
        let mags_snapshot = p3.freq_mag_l.clone();
        let r2 = audit_phase3_dynamic_and_head_gain(&mut p3).unwrap();
        assert_eq!(r2.head_gain_scale, 1.0);
        for (a, b) in gains_snapshot.iter().zip(&p3.readout_gains_l) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in mags_snapshot.iter().zip(&p3.freq_mag_l) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        p3.max_delay_excursion = f64::NAN;
        assert!(matches!(
            audit_phase3_dynamic_and_head_gain(&mut p3),
            Err(TrainError::AuditViolation { .. })
        ));
    }
}
