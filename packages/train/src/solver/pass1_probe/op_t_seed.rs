//! Soft FFT GCC-PHAT & power-envelope cepstrum multi-delay `Op-T` seeding across `1..=4`
//! symmetric pure-wet `TransportQuad`s (`>= 5.0` samples, zero lockout on diffuse reverbs or
//! subtle chorus, silent target guard) (`ref[train.pipeline.physics-probe]`).

mod peaks;

use crate::control::stability::MIN_DELAY_CROSSBAR_SAMP;

use super::super::model::TransportQuadF64;
use super::super::pass2_opt::op_t_quad::{
    eval_op_t_bank_stereo_and_adjoint, eval_op_t_quad_stereo_and_adjoint,
};
use super::telemetry_init::Pass1PhysicsHint;
use peaks::{GCC_WIN_CAP, GccPeakStats, detect_gcc_phat_cepstrum_peaks};

const OPT_MAX_QUADS: usize = 4;
const _: () = assert!(OPT_MAX_QUADS == 4);
const _: () = assert!(GCC_WIN_CAP.is_power_of_two());

/// Seeded `Op-T` diagnostics + updated target with confirmed comb contribution removed.
pub(super) struct OpTSeedOutcome {
    pub dominant_delay_samples: f64,
    pub estimated_loop_gain: f64,
    pub incoherent_echo_only_soft_seed: bool,
    pub tgt_no_opt_l: Vec<f64>,
    pub tgt_no_opt_r: Vec<f64>,
}

fn normalized_res_autocorr_peak(residual: &[f64], max_lag: usize) -> (f64, f64) {
    let n = residual.len();
    let lag_lo = MIN_DELAY_CROSSBAR_SAMP as usize;
    let lag_hi = max_lag.min(n / 2).max(lag_lo);
    if n < lag_lo + 8 {
        return (0.0, 0.0);
    }
    let mean = residual.iter().sum::<f64>() / (n as f64);
    let centered: Vec<f64> = residual.iter().map(|&v| v - mean).collect();
    let e0: f64 = centered.iter().map(|v| v * v).sum::<f64>().max(1.0e-12);
    let mut best_lag = 0_usize;
    let mut best_rho = 0.0_f64;
    for lag in lag_lo..=lag_hi {
        let mut dot = 0.0_f64;
        for t in lag..n {
            dot += centered[t] * centered[t - lag];
        }
        let rho = dot.abs() / e0;
        if rho > best_rho {
            best_rho = rho;
            best_lag = lag;
        }
    }
    (best_lag as f64, best_rho)
}

fn estimate_comb_wet_tap(
    target: &[f64],
    residual: &[f64],
    start: usize,
    d: usize,
    g_init: f64,
) -> f64 {
    let n = target.len().min(residual.len());
    let mut comb = vec![0.0_f64; n];
    let (mut num_wet, mut den_wet) = (0.0_f64, 0.0_f64);
    for t in 0..n {
        let base_t = target[t] - residual[t];
        let tap = if t >= d { comb[t - d] } else { 0.0 };
        comb[t] = base_t + g_init * tap;
        if t >= start {
            num_wet += residual[t] * tap;
            den_wet += tap * tap;
        }
    }
    if den_wet <= 1.0e-12 {
        0.0
    } else {
        (num_wet / den_wet).clamp(-0.90, 0.90)
    }
}

/// Inputs and residual calibration metrics passed into [`seed_soft_op_t_quads`].
pub(super) struct OpTSeedInput<'a> {
    pub input_l: &'a [f64],
    pub input_r: &'a [f64],
    pub target_l: &'a [f64],
    pub target_r: &'a [f64],
    pub res_fir_l: &'a [f64],
    pub res_fir_r: &'a [f64],
    pub start: usize,
    pub tgt_tot_e: f64,
    pub res_fir_tot_e: f64,
    pub sample_rate_hz: f64,
    pub hint: Option<&'a Pass1PhysicsHint>,
}

/// Seeds `model.transport_quads[0..4]` cleanly from dry/wet residuals and optional physics hints
/// (`ref[train.pipeline.physics-probe]`).
pub(super) fn seed_soft_op_t_quads(
    model: &mut super::super::model::SonaModelF64,
    inp: &OpTSeedInput<'_>,
) -> OpTSeedOutcome {
    let OpTSeedInput {
        input_l,
        input_r,
        target_l,
        target_r,
        res_fir_l,
        res_fir_r,
        start,
        tgt_tot_e,
        res_fir_tot_e,
        sample_rate_hz,
        hint,
    } = *inp;
    let n = input_l
        .len()
        .min(input_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let mut tgt_no_opt_l = target_l[..n].to_vec();
    let mut tgt_no_opt_r = target_r[..n].to_vec();

    // Preserve strict silence safety guard when wet target energy is negligible.
    if tgt_tot_e <= 1.0e-12 || res_fir_tot_e <= 1.0e-10 {
        return OpTSeedOutcome {
            dominant_delay_samples: 0.0,
            estimated_loop_gain: 0.0,
            incoherent_echo_only_soft_seed: false,
            tgt_no_opt_l,
            tgt_no_opt_r,
        };
    }

    let max_lag = ((n.saturating_sub(start)) / 2).min(4_096);
    let res_fir_l_e: f64 = res_fir_l[start..n].iter().map(|v| v * v).sum();
    let res_fir_r_e: f64 = res_fir_r[start..n].iter().map(|v| v * v).sum();

    let (lag_l, rho_l) = if res_fir_l_e > 1.0e-6 * tgt_tot_e {
        normalized_res_autocorr_peak(&res_fir_l[start..n], max_lag)
    } else {
        (0.0, 0.0)
    };
    let (lag_r, rho_r) = if res_fir_r_e > 1.0e-6 * tgt_tot_e {
        normalized_res_autocorr_peak(&res_fir_r[start..n], max_lag)
    } else {
        (0.0, 0.0)
    };

    let (crisp_lag, crisp_rho) = if rho_r > rho_l && lag_r >= MIN_DELAY_CROSSBAR_SAMP {
        (lag_r, rho_r)
    } else {
        (lag_l, rho_l)
    };

    if crisp_rho > 0.25 && crisp_lag >= MIN_DELAY_CROSSBAR_SAMP {
        let g_init = crisp_rho.clamp(0.0, 0.85);
        let delay_l = if lag_l >= MIN_DELAY_CROSSBAR_SAMP {
            lag_l
        } else {
            crisp_lag
        };
        let delay_r = if lag_r >= MIN_DELAY_CROSSBAR_SAMP {
            lag_r
        } else {
            crisp_lag
        };
        let d_l = (delay_l.round() as usize).min(n.saturating_sub(1));
        let d_r = (delay_r.round() as usize).min(n.saturating_sub(1));
        let wl = estimate_comb_wet_tap(target_l, res_fir_l, start, d_l, g_init);
        let wr = estimate_comb_wet_tap(target_r, res_fir_r, start, d_r, g_init);
        let wl_init = if lag_l >= MIN_DELAY_CROSSBAR_SAMP && wl.abs() >= 0.05 {
            wl
        } else {
            0.0
        };
        let wr_init = if lag_r >= MIN_DELAY_CROSSBAR_SAMP && wr.abs() >= 0.05 {
            wr
        } else {
            0.0
        };
        if wl_init != 0.0 || wr_init != 0.0 {
            let cand_tq = TransportQuadF64 {
                delay_samples: [delay_l, delay_r, delay_l, delay_r],
                g_loop: g_init,
                input_gains: [0.5, 0.5, -0.5, -0.5],
                wet_tap_l: [wl_init, 0.0, -wl_init, 0.0],
                wet_tap_r: [0.0, wr_init, 0.0, -wr_init],
                ext_f32: delay_l > 120.0 || delay_r > 120.0,
                ..TransportQuadF64::default()
            };
            let base_l: Vec<f64> = (0..n).map(|t| target_l[t] - res_fir_l[t]).collect();
            let base_r: Vec<f64> = (0..n).map(|t| target_r[t] - res_fir_r[t]).collect();
            let (op_l, op_r, _) =
                eval_op_t_quad_stereo_and_adjoint(&base_l, &base_r, &cand_tq, sample_rate_hz, None);
            let with_op_t_e: f64 = (start..n)
                .map(|t| {
                    let dl = res_fir_l[t] - op_l[t];
                    let dr = res_fir_r[t] - op_r[t];
                    dl * dl + dr * dr
                })
                .sum();
            if with_op_t_e < 0.85 * res_fir_tot_e {
                model.transport_quads[0] = cand_tq;
                for t in 0..n {
                    tgt_no_opt_l[t] -= op_l[t];
                    tgt_no_opt_r[t] -= op_r[t];
                }
                return OpTSeedOutcome {
                    dominant_delay_samples: crisp_lag,
                    estimated_loop_gain: g_init,
                    incoherent_echo_only_soft_seed: false,
                    tgt_no_opt_l,
                    tgt_no_opt_r,
                };
            }
        }
    }

    // Soft GCC-PHAT + cepstrum multi-delay fallback: gate on genuine MAD-calibrated delay
    // prominence (excluding periodic dry grating lobes), excess LFO modulation, trailing reverb
    // tail, or explicit Stage-A physics hint so memoryless waveshaping never triggers Op-T tanks.
    let GccPeakStats {
        peaks,
        top_lag,
        phat_z,
        dry_rho_at_top_lag,
        res_xy_rho_at_top_lag,
        excess_mod_cv,
        env_corr,
        hf_ratio,
    } = detect_gcc_phat_cepstrum_peaks(input_l, input_r, res_fir_l, res_fir_r, start, max_lag);
    let res_frac = (res_fir_tot_e / tgt_tot_e).clamp(0.0, 1.0);
    let pos_hint = |opt: Option<f64>| opt.filter(|&v| v.is_finite() && v > 0.0);
    let has_hint = hint
        .and_then(|h| {
            pos_hint(h.delay_ms)
                .or_else(|| pos_hint(h.rt60_s))
                .or_else(|| pos_hint(h.modulation_rate_hz))
        })
        .is_some();
    let has_echo_prominence = top_lag >= 32
        && phat_z >= 6.5
        && dry_rho_at_top_lag < 0.35
        && res_xy_rho_at_top_lag >= 0.12
        && res_frac >= 1.0e-3;
    let has_mod_transport = excess_mod_cv > 0.18 && env_corr < 0.65 && res_frac >= 5.0e-4;
    let has_decay_tail = if n >= start + 256 {
        let mid = start + (n - start) / 2;
        let tail_start = start + ((n - start) * 3) / 4;
        let head_dry_e: f64 = input_l[start..mid]
            .iter()
            .zip(&input_r[start..mid])
            .map(|(a, b)| a * a + b * b)
            .sum();
        let tail_dry_e: f64 = input_l[tail_start..n]
            .iter()
            .zip(&input_r[tail_start..n])
            .map(|(a, b)| a * a + b * b)
            .sum();
        let tail_res_e: f64 = res_fir_l[tail_start..n]
            .iter()
            .zip(&res_fir_r[tail_start..n])
            .map(|(a, b)| a * a + b * b)
            .sum();
        head_dry_e > 1.0e-12 && tail_dry_e < 1.0e-4 * head_dry_e && tail_res_e > 1.0e-3 * tgt_tot_e
    } else {
        false
    };
    if !has_hint && !has_echo_prominence && !has_mod_transport && !has_decay_tail {
        return OpTSeedOutcome {
            dominant_delay_samples: 0.0,
            estimated_loop_gain: 0.0,
            incoherent_echo_only_soft_seed: false,
            tgt_no_opt_l,
            tgt_no_opt_r,
        };
    }

    let hint_delay_samp = hint
        .and_then(|h| h.delay_ms)
        .filter(|&d| d.is_finite() && d > 0.0)
        .map(|d| (d * 1.0e-3 * sample_rate_hz).clamp(MIN_DELAY_CROSSBAR_SAMP, 48_000.0));
    let hint_rt60 = hint
        .and_then(|h| h.rt60_s)
        .filter(|&t| t.is_finite() && t > 0.0);
    let hint_mod = hint
        .and_then(|h| h.modulation_rate_hz)
        .filter(|&r| r.is_finite() && r > 0.0)
        .map(|r| r.clamp(0.0, 16.0));

    let default_delays = [
        37.0_f64, 53.0, 79.0, 107.0, 151.0, 211.0, 293.0, 389.0, 521.0, 709.0, 953.0, 1_297.0,
        1_741.0, 2_311.0, 3_079.0, 4_099.0,
    ];
    let mut delays = [0.0_f64; 16];
    let mut d_idx = 0_usize;
    if let Some(hd) = hint_delay_samp {
        delays[d_idx] = hd;
        d_idx += 1;
    }
    for &(lag, _) in &peaks {
        if d_idx >= 16 {
            break;
        }
        if delays[..d_idx].iter().all(|&d| (d - lag).abs() >= 3.0) {
            delays[d_idx] = lag.max(MIN_DELAY_CROSSBAR_SAMP);
            d_idx += 1;
        }
    }
    for &def_d in &default_delays {
        if d_idx >= 16 {
            break;
        }
        let d_clamped = def_d
            .min(max_lag.max(64) as f64)
            .max(MIN_DELAY_CROSSBAR_SAMP);
        if delays[..d_idx]
            .iter()
            .all(|&d| (d - d_clamped).abs() >= 3.0)
        {
            delays[d_idx] = d_clamped;
            d_idx += 1;
        }
    }

    let soft_g = if let Some(rt60) = hint_rt60 {
        let mean_d = delays[..4].iter().sum::<f64>() / 4.0;
        10.0_f64
            .powf(-3.0 * mean_d / (rt60 * sample_rate_hz).max(1.0))
            .clamp(0.15, 0.92)
    } else {
        (0.25 + 0.55 * res_frac.sqrt()).clamp(0.15, 0.85)
    };
    let mod_detected = hint_mod.is_some() || (excess_mod_cv > 0.18 && env_corr < 0.65);
    let lfo_rate = hint_mod.unwrap_or(if mod_detected { 0.85 } else { 0.0 });
    let lfo_depth = if mod_detected { 3.5 } else { 0.0 };
    let shimmer_detected = (has_decay_tail || hint_rt60.is_some())
        && hf_ratio > 0.35
        && crisp_rho < 0.22
        && res_frac > 0.05;

    let mut cand_quads = [TransportQuadF64::default(); OPT_MAX_QUADS];
    for q in 0..OPT_MAX_QUADS {
        let d4 = [
            delays[q * 4],
            delays[q * 4 + 1],
            delays[q * 4 + 2],
            delays[q * 4 + 3],
        ];
        let sign = if q % 2 == 0 { 1.0 } else { -1.0 };
        let tq = &mut cand_quads[q];
        tq.delay_samples = d4;
        tq.g_loop = (soft_g * (1.0 - 0.05 * (q as f64))).clamp(0.05, 0.92);
        tq.damp_pole = (0.18 + 0.25 * (1.0 - hf_ratio.clamp(0.0, 1.0))).clamp(0.05, 0.75);
        tq.input_gains = [0.25, 0.25, -0.25, -0.25];
        tq.wet_tap_l = [sign, 0.5 * sign, -sign, -0.5 * sign];
        tq.wet_tap_r = if hint.and_then(|h| h.has_stereo_width) == Some(false) {
            tq.wet_tap_l
        } else {
            [0.5 * sign, -sign, -0.5 * sign, sign]
        };
        tq.allpass_delays = [7, 13, 19, 29];
        tq.allpass_g = [0.22, -0.18, 0.16, -0.12];
        tq.scat_psi = sign * 0.20;
        tq.lfo_phase_off_q16 = [0, 16_384, 32_768, 49_152];
        tq.lfo_rate_hz = lfo_rate;
        tq.lfo_depths = [
            lfo_depth,
            1.15 * lfo_depth,
            0.9 * lfo_depth,
            1.25 * lfo_depth,
        ];
        tq.ramp_phase_off_q16 = [0, 32_768, 16_384, 49_152];
        if shimmer_detected {
            tq.ramp_rate_hz = 1.5;
            tq.ramp_spans_samples = [64.0, 64.0, 64.0, 64.0];
            tq.window_depth = 0.5;
        }
        tq.ext_f32 = d4.iter().any(|&d| d > 120.0);
    }

    let win_eval = n.min(start + GCC_WIN_CAP);
    let (op_l, op_r, _) = eval_op_t_bank_stereo_and_adjoint(
        &input_l[..win_eval],
        &input_r[..win_eval],
        &cand_quads,
        OPT_MAX_QUADS,
        sample_rate_hz,
        None,
    );
    let (mut num, mut den, mut res_eval_e) = (0.0_f64, 0.0_f64, 0.0_f64);
    for t in start..win_eval {
        num += res_fir_l[t] * op_l[t] + res_fir_r[t] * op_r[t];
        den += op_l[t] * op_l[t] + op_r[t] * op_r[t];
        res_eval_e += res_fir_l[t] * res_fir_l[t] + res_fir_r[t] * res_fir_r[t];
    }
    let alpha_ls = if den > 1.0e-12 {
        (num / den).clamp(-0.35, 0.35)
    } else {
        0.0
    };
    let mut err_after = 0.0_f64;
    for t in start..win_eval {
        let el = res_fir_l[t] - alpha_ls * op_l[t];
        let er = res_fir_r[t] - alpha_ls * op_r[t];
        err_after += el * el + er * er;
    }

    let coherent = alpha_ls.abs() >= 0.01 && err_after < 0.92 * res_eval_e;
    let allow_incoherent_soft_seed =
        has_echo_prominence || has_decay_tail || hint_rt60.is_some() || mod_detected;
    if !coherent && !allow_incoherent_soft_seed {
        return OpTSeedOutcome {
            dominant_delay_samples: 0.0,
            estimated_loop_gain: 0.0,
            incoherent_echo_only_soft_seed: false,
            tgt_no_opt_l,
            tgt_no_opt_r,
        };
    }
    let tap_scale = if coherent { alpha_ls } else { 0.005 };
    for tq in &mut cand_quads {
        for v in &mut tq.wet_tap_l {
            *v *= tap_scale;
        }
        for v in &mut tq.wet_tap_r {
            *v *= tap_scale;
        }
    }
    model.transport_quads = cand_quads;
    if coherent {
        let (full_op_l, full_op_r, _) = eval_op_t_bank_stereo_and_adjoint(
            &input_l[..n],
            &input_r[..n],
            &model.transport_quads,
            OPT_MAX_QUADS,
            sample_rate_hz,
            None,
        );
        for t in 0..n {
            tgt_no_opt_l[t] -= full_op_l[t];
            tgt_no_opt_r[t] -= full_op_r[t];
        }
    }

    OpTSeedOutcome {
        dominant_delay_samples: delays[0],
        estimated_loop_gain: soft_g,
        incoherent_echo_only_soft_seed: !coherent && !has_decay_tail && !mod_detected && !has_hint,
        tgt_no_opt_l,
        tgt_no_opt_r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::SonaModelF64;

    #[test]
    fn test_soft_op_t_seed_non_lockout_for_diffuse_reverb_and_subtle_chorus() {
        let n = 2_048_usize;
        let sr = 48_000.0_f64;
        let mut x = vec![0.0_f64; n];
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        for v in &mut x {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = 0.5 * (((seed >> 33) as f64) / ((1_u64 << 31) as f64) - 1.0);
        }

        // 1. Diffuse reverb residual with low linear autocorrelation (< 0.15) is NOT locked out.
        let mut diff_y = vec![0.0_f64; n];
        for t in 180..n {
            diff_y[t] =
                0.14 * x[t - 73] - 0.11 * x[t - 109] + 0.09 * x[t - 157] - 0.07 * diff_y[t - 179];
        }
        let tot_e: f64 = 2.0 * diff_y[64..].iter().map(|v| v * v).sum::<f64>();
        let mut mdl_rev = SonaModelF64::new(sr);
        let out_rev = seed_soft_op_t_quads(
            &mut mdl_rev,
            &OpTSeedInput {
                input_l: &x,
                input_r: &x,
                target_l: &diff_y,
                target_r: &diff_y,
                res_fir_l: &diff_y,
                res_fir_r: &diff_y,
                start: 64,
                tgt_tot_e: tot_e,
                res_fir_tot_e: tot_e,
                sample_rate_hz: sr,
                hint: None,
            },
        );
        assert!(out_rev.dominant_delay_samples >= MIN_DELAY_CROSSBAR_SAMP);
        assert!(out_rev.estimated_loop_gain > 0.1);
        assert!(mdl_rev.transport_quads.iter().all(|q| q.g_loop > 0.0));
        assert!(
            mdl_rev.transport_quads[0]
                .wet_tap_l
                .iter()
                .any(|&w| w.abs() > 0.0)
        );

        // 2. Subtle time-varying chorus residual triggers quadrature LFO seeding without lockout.
        let mut dry = vec![0.0_f64; n];
        let mut res_chorus = vec![0.0_f64; n];
        for t in 0..n {
            let tm = (t as f64) / sr;
            dry[t] = 0.5 * (core::f64::consts::TAU * 330.0 * tm).sin();
            let env = 0.5 * (1.0 + (core::f64::consts::TAU * 3.5 * tm).sin());
            res_chorus[t] = 0.08 * env * dry[t.saturating_sub(140)];
        }
        let c_res_e: f64 = 2.0 * res_chorus[64..].iter().map(|v| v * v).sum::<f64>();
        let c_tot_e: f64 = 2.0 * dry[64..].iter().map(|v| v * v).sum::<f64>();
        let mut mdl_ch = SonaModelF64::new(sr);
        let _ = seed_soft_op_t_quads(
            &mut mdl_ch,
            &OpTSeedInput {
                input_l: &dry,
                input_r: &dry,
                target_l: &dry,
                target_r: &dry,
                res_fir_l: &res_chorus,
                res_fir_r: &res_chorus,
                start: 64,
                tgt_tot_e: c_tot_e,
                res_fir_tot_e: c_res_e,
                sample_rate_hz: sr,
                hint: None,
            },
        );
        assert!(mdl_ch.transport_quads[0].lfo_rate_hz > 0.0);
        assert_eq!(
            mdl_ch.transport_quads[0].lfo_phase_off_q16,
            [0, 16_384, 32_768, 49_152]
        );

        // 3. Silent target leaves Op-T completely muted.
        let mut mdl_sil = SonaModelF64::new(sr);
        let zeros = vec![0.0_f64; n];
        let out_sil = seed_soft_op_t_quads(
            &mut mdl_sil,
            &OpTSeedInput {
                input_l: &x,
                input_r: &x,
                target_l: &zeros,
                target_r: &zeros,
                res_fir_l: &zeros,
                res_fir_r: &zeros,
                start: 64,
                tgt_tot_e: 0.0,
                res_fir_tot_e: 0.0,
                sample_rate_hz: sr,
                hint: None,
            },
        );
        assert_eq!(out_sil.estimated_loop_gain, 0.0);
        assert_eq!(mdl_sil.transport_quads[0].g_loop, 0.0);

        // 4. Memoryless symmetric overdrive residual leaves Op-T & shimmer completely inactive.
        let od_y: Vec<f64> = x.iter().map(|&v| (3.0 * v).tanh()).collect();
        let dot_xy: f64 = x[64..].iter().zip(&od_y[64..]).map(|(a, b)| a * b).sum();
        let dot_xx: f64 = x[64..].iter().map(|a| a * a).sum::<f64>().max(1.0e-12);
        let gain_lin = dot_xy / dot_xx;
        let od_res: Vec<f64> = od_y
            .iter()
            .zip(&x)
            .map(|(&y, &xv)| y - gain_lin * xv)
            .collect();
        let od_tot_e: f64 = 2.0 * od_y[64..].iter().map(|v| v * v).sum::<f64>();
        let od_res_e: f64 = 2.0 * od_res[64..].iter().map(|v| v * v).sum::<f64>();
        let mut mdl_od = SonaModelF64::new(sr);
        let out_od = seed_soft_op_t_quads(
            &mut mdl_od,
            &OpTSeedInput {
                input_l: &x,
                input_r: &x,
                target_l: &od_y,
                target_r: &od_y,
                res_fir_l: &od_res,
                res_fir_r: &od_res,
                start: 64,
                tgt_tot_e: od_tot_e,
                res_fir_tot_e: od_res_e,
                sample_rate_hz: sr,
                hint: None,
            },
        );
        assert_eq!(out_od.estimated_loop_gain, 0.0);
        assert_eq!(mdl_od.transport_quads[0].g_loop, 0.0);
        assert_eq!(mdl_od.transport_quads[0].ramp_rate_hz, 0.0);

        // 5. Level-stepped memoryless overdrive residual must not trigger Op-T LFO modulation.
        let x_mean = x.iter().sum::<f64>() / (n as f64);
        let levels = [0.05_f64, 0.2, 0.5, 0.9];
        let x_step: Vec<f64> = (0..n)
            .map(|t| levels[(t * 4) / n] * (x[t] - x_mean))
            .collect();
        let step_y: Vec<f64> = x_step.iter().map(|&v| (3.0 * v).tanh()).collect();
        let s_xy: f64 = x_step[64..]
            .iter()
            .zip(&step_y[64..])
            .map(|(a, b)| a * b)
            .sum();
        let s_xx: f64 = x_step[64..].iter().map(|a| a * a).sum::<f64>().max(1.0e-12);
        let s_gain = s_xy / s_xx;
        let step_res: Vec<f64> = step_y
            .iter()
            .zip(&x_step)
            .map(|(&y, &xv)| y - s_gain * xv)
            .collect();
        let step_tot_e: f64 = 2.0 * step_y[64..].iter().map(|v| v * v).sum::<f64>();
        let step_res_e: f64 = 2.0 * step_res[64..].iter().map(|v| v * v).sum::<f64>();
        let mut mdl_step = SonaModelF64::new(sr);
        let out_step = seed_soft_op_t_quads(
            &mut mdl_step,
            &OpTSeedInput {
                input_l: &x_step,
                input_r: &x_step,
                target_l: &step_y,
                target_r: &step_y,
                res_fir_l: &step_res,
                res_fir_r: &step_res,
                start: 64,
                tgt_tot_e: step_tot_e,
                res_fir_tot_e: step_res_e,
                sample_rate_hz: sr,
                hint: None,
            },
        );
        assert_eq!(out_step.estimated_loop_gain, 0.0);
        assert_eq!(mdl_step.transport_quads[0].g_loop, 0.0);
        assert_eq!(mdl_step.transport_quads[0].lfo_rate_hz, 0.0);

        // 6. Nonlinear waveshaper + 16-sample-delayed resonant FIR/cabinet (no Op-T) must NOT lock Op-T.
        let mut x_zc = vec![0.0_f64; n];
        let mut s6 = 0x517C_C1B7_2722_0A95_u64;
        for v in &mut x_zc {
            s6 = s6
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((s6 >> 32) as f64) / ((1_u64 << 32) as f64) - 0.5;
        }
        let mut shaper_y = vec![0.0_f64; n];
        let mut lin_y = vec![0.0_f64; n];
        for t in 16..n {
            let nl = 0.7 * (2.5 * x_zc[t - 16]).tanh();
            let lin = 0.7 * 2.5 * x_zc[t - 16];
            shaper_y[t] = 0.45 * nl + 0.55 * shaper_y[t - 1] - 0.15 * x_zc[t.saturating_sub(24)];
            lin_y[t] = 0.45 * lin + 0.55 * lin_y[t - 1] - 0.15 * x_zc[t.saturating_sub(24)];
        }
        let d_lin: f64 = lin_y[64..]
            .iter()
            .zip(&shaper_y[64..])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / lin_y[64..].iter().map(|a| a * a).sum::<f64>().max(1.0e-12);
        let shaper_res: Vec<f64> = shaper_y
            .iter()
            .zip(&lin_y)
            .map(|(&y, &lv)| y - d_lin * lv)
            .collect();
        let shaper_tot_e: f64 = 2.0 * shaper_y[64..].iter().map(|v| v * v).sum::<f64>();
        let shaper_res_e: f64 = 2.0 * shaper_res[64..].iter().map(|v| v * v).sum::<f64>();
        let mut mdl_cab = SonaModelF64::new(sr);
        let out_cab = seed_soft_op_t_quads(
            &mut mdl_cab,
            &OpTSeedInput {
                input_l: &x_zc,
                input_r: &x_zc,
                target_l: &shaper_y,
                target_r: &shaper_y,
                res_fir_l: &shaper_res,
                res_fir_r: &shaper_res,
                start: 64,
                tgt_tot_e: shaper_tot_e,
                res_fir_tot_e: shaper_res_e,
                sample_rate_hz: sr,
                hint: None,
            },
        );
        assert_eq!(out_cab.estimated_loop_gain, 0.0);
        assert_eq!(mdl_cab.transport_quads[0].g_loop, 0.0);
        assert_eq!(mdl_cab.transport_quads[0].lfo_rate_hz, 0.0);

        // Case 7: Nonlinear shaper + 16-sample-delayed resonant cabinet with mono physics hint
        // (delay_ms: Some(3.5), has_stereo_width: Some(false)) must not falsely lock Op-T.
        let mut mdl_res_cab = SonaModelF64::new(sr);
        let n_c7 = 8192_usize;
        let mut in_c7 = vec![0.0_f64; n_c7];
        let mut rng_c7 = 0x9e37_79b9_7f4a_7c15_u64;
        for (t, dst) in in_c7.iter_mut().enumerate() {
            rng_c7 = rng_c7
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((rng_c7 >> 32) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            let amp = 0.25
                + 0.75
                    * (2.0 * std::f64::consts::PI * 2.0 * t as f64 / sr)
                        .sin()
                        .abs();
            *dst = 0.6 * amp * noise;
        }
        let mut tgt_c7 = vec![0.0_f64; n_c7];
        let mut ring_c7 = [0.0_f64; 32];
        let (mut st_re, mut st_im) = (0.0_f64, 0.0_f64);
        let r_pole = 0.96_f64;
        let th_pole = 0.18_f64;
        let b_g = (1.0 - r_pole * r_pole).sqrt();
        let (cr_p, st_p) = (th_pole.cos(), th_pole.sin());
        for t in 0..n_c7 {
            let nl = (1.8 * in_c7[t]).tanh();
            let delayed = ring_c7[(t + 16) & 31];
            ring_c7[t & 31] = nl;
            let next_re = r_pole * (cr_p * st_re - st_p * st_im) + b_g * delayed;
            let next_im = r_pole * (st_p * st_re + cr_p * st_im);
            (st_re, st_im) = (next_re, next_im);
            tgt_c7[t] = 0.35 * nl + 0.45 * next_re;
        }
        let mut res_c7 = vec![0.0_f64; n_c7];
        let (mut tot_c7, mut res_e_c7) = (0.0_f64, 0.0_f64);
        for t in 0..n_c7 {
            // Simulate FIR-cancelled fundamental leaving higher-order shaper + delayed cabinet residue.
            res_c7[t] = tgt_c7[t] - 0.55 * in_c7[t];
            if t >= 64 {
                tot_c7 += 2.0 * tgt_c7[t] * tgt_c7[t];
                res_e_c7 += 2.0 * res_c7[t] * res_c7[t];
            }
        }
        let out_c7 = seed_soft_op_t_quads(
            &mut mdl_res_cab,
            &OpTSeedInput {
                input_l: &in_c7,
                input_r: &in_c7,
                target_l: &tgt_c7,
                target_r: &tgt_c7,
                res_fir_l: &res_c7,
                res_fir_r: &res_c7,
                start: 64,
                tgt_tot_e: tot_c7,
                res_fir_tot_e: res_e_c7,
                sample_rate_hz: sr,
                hint: Some(&Pass1PhysicsHint {
                    delay_ms: Some(3.5),
                    has_stereo_width: Some(false),
                    ..Default::default()
                }),
            },
        );
        assert_eq!(out_c7.estimated_loop_gain, 0.0);
        assert_eq!(mdl_res_cab.transport_quads[0].g_loop, 0.0);
        assert_eq!(mdl_res_cab.transport_quads[0].lfo_rate_hz, 0.0);
    }
}
