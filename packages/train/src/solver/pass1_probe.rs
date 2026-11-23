//! Pass-1 universal black-box physics & linear-modal identification probe
//! (`< 0.5 ms` CPU `Op-P` quadrature probe, `O(N log N)` FFT Toeplitz FIR `32/48/96/128`,
//! Steiglitz–McBride `(B, A)` + 2-pass OMP stereo Kautz `8/12/24/32` up to `0.9992`,
//! soft GCC-PHAT/cepstrum multi-quad `Op-T`, and data-driven front-end telemetry,
//! `ref[train.pipeline.physics-probe]`).

pub(crate) mod fir_toeplitz;
pub(crate) mod kautz_fit;
mod op_t_seed;
mod steiglitz_mcbride;
mod telemetry_init;

pub use kautz_fit::synth_kautz_unit_mode;
pub use steiglitz_mcbride::steiglitz_mcbride_poles;
pub use telemetry_init::{OpPProbeSummary, Pass1PhysicsHint, run_cpu_op_p_probe};

use crate::error::TrainError;

use super::model::{
    NUM_FIR_TAPS_ULTRA, NUM_KAUTZ_PAIRS_ULTRA, NUM_TRANSPORT_QUADS_ULTRA, SonaModelF64,
    TransportQuadF64,
};
use fir_toeplitz::{fit_early_fir_taps, residual_after_fir};
use kautz_fit::{
    KautzFitInputs, append_log_octave_h2_candidates, fit_joint_kautz_bank, poles_too_close,
};
use op_t_seed::{OpTSeedInput, seed_soft_op_t_quads};
use steiglitz_mcbride::steiglitz_mcbride_poles as extract_sm_poles;
use telemetry_init::{TelemetrySeedSlices, seed_telemetry_and_front_end};

/// Configuration passed into [`run_pass1_probe_with_hint`].
#[derive(Clone, Copy, Debug)]
pub struct Pass1ProbeConfig<'a> {
    /// Target device/project sample rate in Hz.
    pub sample_rate_hz: f64,
    /// Maximum early FIR taps to fit (`32..=128`).
    pub max_fir_taps: usize,
    /// Maximum Kautz modal pairs to allocate (`<= 32`).
    pub max_kautz_pairs: usize,
    /// Optional 100% data-driven scalar physics hints from Stage A.
    pub hint: Option<&'a Pass1PhysicsHint>,
}

/// Complete Pass-1 initialization diagnostic report.
#[derive(Clone, Debug, PartialEq)]
pub struct Pass1ProbeReport {
    /// CPU `Op-P` quadrature summary on dry stereo excitation.
    pub op_p: OpPProbeSummary,
    /// Count of recovered complex conjugate Kautz modal pairs (`<= 32`).
    pub recovered_kautz_pairs: usize,
    /// Dominant `Op-T` feedback lag detected in samples (`>= 5.0`, or `0.0` if non-resonant).
    pub dominant_transport_delay_samples: f64,
    /// Estimated `Op-T` feedback loop gain `[0.0, 0.99]`.
    pub estimated_loop_gain: f64,
    /// Initial normalized stereo time-domain residual energy ratio vs target.
    pub initial_esr_lin: f64,
}

/// Complete Pass-1 output bundling diagnostics ([`Pass1ProbeReport`]) and the seeded
/// [`SonaModelF64`] for Stage A -> Stage B handoff without redundant recomputation.
#[derive(Clone, Debug)]
pub struct Pass1ProbeResult {
    /// Diagnostic summary emitted by Pass 1.
    pub report: Pass1ProbeReport,
    /// Initialized 4-tier `SONA v3.0` model ready for Pass 2.
    pub model: SonaModelF64,
}

/// Executes Pass-1 physics & basis probing with [`Pass1ProbeConfig`] (including optional scalar
/// [`Pass1PhysicsHint`] priors) and returns [`Pass1ProbeResult`] (`ref[train.pipeline.physics-probe]`).
pub fn run_pass1_probe_with_hint(
    input_l: &[f64],
    input_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    cfg: Pass1ProbeConfig<'_>,
) -> Result<Pass1ProbeResult, TrainError> {
    let Pass1ProbeConfig {
        sample_rate_hz,
        max_fir_taps,
        max_kautz_pairs,
        hint,
    } = cfg;
    let n = input_l
        .len()
        .min(input_r.len())
        .min(target_l.len())
        .min(target_r.len());
    if n < 64 {
        return Err(TrainError::InvalidDimensions {
            context: "run_pass1_probe",
            expected: 64,
            actual: n,
        });
    }

    let op_p = run_cpu_op_p_probe(&input_l[..n], &input_r[..n], sample_rate_hz);
    let mut model = SonaModelF64::new(sample_rate_hz);

    let fir_n = max_fir_taps.clamp(32, NUM_FIR_TAPS_ULTRA);
    let start = fir_n.min(n / 4);
    let tgt_l_e: f64 = target_l[start..n].iter().map(|v| v * v).sum::<f64>();
    let tgt_r_e: f64 = target_r[start..n].iter().map(|v| v * v).sum::<f64>();
    let tgt_tot_raw = tgt_l_e + tgt_r_e;
    let tgt_tot_e = tgt_tot_raw.max(1.0e-12);

    // Step 1: O(N log N) FFT Toeplitz early FIR baseline + soft Op-T multi-quad seeding.
    let tmp_fir_l = fit_early_fir_taps(&input_l[..n], &target_l[..n], fir_n);
    let tmp_fir_r = fit_early_fir_taps(&input_r[..n], &target_r[..n], fir_n);
    let res_fir_l = residual_after_fir(&input_l[..n], &target_l[..n], &tmp_fir_l, fir_n);
    let res_fir_r = residual_after_fir(&input_r[..n], &target_r[..n], &tmp_fir_r, fir_n);
    let res_fir_l_e: f64 = res_fir_l[start..n].iter().map(|v| v * v).sum();
    let res_fir_r_e: f64 = res_fir_r[start..n].iter().map(|v| v * v).sum();
    let res_fir_tot_e = res_fir_l_e + res_fir_r_e;

    // Seed signed SSB phasor_freq_hz, Op-N[Pre], SensoryQuad, and even-order rect_sign.
    seed_telemetry_and_front_end(
        &mut model,
        &op_p,
        &TelemetrySeedSlices {
            input_l,
            input_r,
            target_l,
            target_r,
            res_l: &res_fir_l,
            res_r: &res_fir_r,
            start,
        },
        hint,
    );

    let op_t_out = seed_soft_op_t_quads(
        &mut model,
        &OpTSeedInput {
            input_l: &input_l[..n],
            input_r: &input_r[..n],
            target_l: &target_l[..n],
            target_r: &target_r[..n],
            res_fir_l: &res_fir_l,
            res_fir_r: &res_fir_r,
            start,
            tgt_tot_e: tgt_tot_raw,
            res_fir_tot_e,
            sample_rate_hz,
            hint,
        },
    );
    let mut delay_samp = op_t_out.dominant_delay_samples;
    let mut loop_g = op_t_out.estimated_loop_gain;
    let tgt_no_opt_l = op_t_out.tgt_no_opt_l;
    let tgt_no_opt_r = op_t_out.tgt_no_opt_r;

    // Step 2: Fit 16-tap onset FIR (`CONVOLVER_KAUTZ_ONSET_DELAY_48K = 16`) before 2-pass OMP +
    // Steiglitz-McBride Kautz fitting (`r <= 0.9992`, `K <= 32`).
    let fir_onset_n = fir_n.min(16);
    let onset_fir_l = fit_early_fir_taps(&input_l[..n], &tgt_no_opt_l, fir_onset_n);
    let onset_fir_r = fit_early_fir_taps(&input_r[..n], &tgt_no_opt_r, fir_onset_n);
    let res_16_l = residual_after_fir(&input_l[..n], &tgt_no_opt_l, &onset_fir_l, fir_onset_n);
    let res_16_r = residual_after_fir(&input_r[..n], &tgt_no_opt_r, &onset_fir_r, fir_onset_n);
    let start16 = fir_onset_n.min(n / 4);
    let res_16_l_e: f64 = res_16_l[start16..n].iter().map(|v| v * v).sum();
    let res_16_r_e: f64 = res_16_r[start16..n].iter().map(|v| v * v).sum();
    let res_16_tot_e = res_16_l_e + res_16_r_e;

    let k_cap = max_kautz_pairs.min(NUM_KAUTZ_PAIRS_ULTRA);
    let sm_end = n.min(start16 + 16_384);
    let mut poles: Vec<(f64, f64)> = Vec::new();
    if res_16_l_e > 1.0e-6 * tgt_tot_e && sm_end > start16 + 24 {
        poles.extend(extract_sm_poles(
            &input_l[start16..sm_end],
            &res_16_l[start16..sm_end],
            16,
            4,
        ));
    }
    if res_16_r_e > 1.0e-6 * tgt_tot_e && sm_end > start16 + 24 {
        for &(r, th) in
            &extract_sm_poles(&input_r[start16..sm_end], &res_16_r[start16..sm_end], 16, 4)
        {
            if !poles.iter().any(|&(pr, pt)| poles_too_close(pr, pt, r, th)) {
                poles.push((r, th));
            }
        }
    }
    poles.sort_by(|x, y| y.0.total_cmp(&x.0));
    if res_16_tot_e > 1.0e-5 * tgt_tot_e {
        append_log_octave_h2_candidates(&mut poles, sample_rate_hz, k_cap);
    }

    let (accepted_kautz, res_after_k_l, res_after_k_r) = fit_joint_kautz_bank(&KautzFitInputs {
        candidates: &poles,
        input_l: &input_l[..n],
        input_r: &input_r[..n],
        res_in_l: &res_16_l,
        res_in_r: &res_16_r,
        start: start16,
        k_limit: k_cap,
    });
    let pair_idx = accepted_kautz.len().min(NUM_KAUTZ_PAIRS_ULTRA);
    for (idx, pair) in accepted_kautz.into_iter().take(pair_idx).enumerate() {
        model.kautz[idx] = pair;
    }
    model.active_kautz_pairs = pair_idx;

    // Step 3: Reconstruct `(tgt_no_opt - kautz_out)` for final early FIR fitting.
    let mut tgt_for_fir_l = vec![0.0_f64; n];
    let mut tgt_for_fir_r = vec![0.0_f64; n];
    for t in 0..n {
        let mut fl_16 = 0.0_f64;
        let mut fr_16 = 0.0_f64;
        for k in 0..fir_onset_n.min(t + 1) {
            fl_16 += onset_fir_l[k] * input_l[t - k];
            fr_16 += onset_fir_r[k] * input_r[t - k];
        }
        tgt_for_fir_l[t] = res_after_k_l[t] + fl_16;
        tgt_for_fir_r[t] = res_after_k_r[t] + fr_16;
    }

    model.fir_l = fit_early_fir_taps(&input_l[..n], &tgt_for_fir_l, fir_n);
    model.fir_r = fit_early_fir_taps(&input_r[..n], &tgt_for_fir_r, fir_n);
    model.active_fir_taps = fir_n;

    let final_res_l = residual_after_fir(&input_l[..n], &tgt_for_fir_l, &model.fir_l, fir_n);
    let final_res_r = residual_after_fir(&input_r[..n], &tgt_for_fir_r, &model.fir_r, fir_n);
    let res_tot_e: f64 = final_res_l[start..n]
        .iter()
        .zip(&final_res_r[start..n])
        .map(|(a, b)| a * a + b * b)
        .sum();
    if op_t_out.incoherent_echo_only_soft_seed
        && res_tot_e < 0.03 * tgt_tot_e
        && res_tot_e < 0.5 * res_fir_tot_e
    {
        model.transport_quads = [TransportQuadF64::default(); NUM_TRANSPORT_QUADS_ULTRA];
        delay_samp = 0.0;
        loop_g = 0.0;
    }

    model.pre_conv_u_gain = if model.active_fir_taps > 0 || model.active_kautz_pairs > 0 {
        [1.0, 1.0]
    } else {
        [0.0, 0.0]
    };
    model.direct_s2_u_gain = [0.0, 0.0];
    model.clamp_all_stability_bounds(true);

    Ok(Pass1ProbeResult {
        report: Pass1ProbeReport {
            op_p,
            recovered_kautz_pairs: pair_idx,
            dominant_transport_delay_samples: delay_samp,
            estimated_loop_gain: loop_g,
            initial_esr_lin: res_tot_e / tgt_tot_e,
        },
        model,
    })
}

/// Runs Pass-1 universal physics probe + pole/FIR/comb initialization from aligned stereo inputs
/// and targets (`ref[train.pipeline.physics-probe]`).
pub fn run_pass1_probe(
    input_l: &[f64],
    input_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    sample_rate_hz: f64,
    max_fir_taps: usize,
    max_kautz_pairs: usize,
) -> Result<(Pass1ProbeReport, SonaModelF64), TrainError> {
    let res = run_pass1_probe_with_hint(
        input_l,
        input_r,
        target_l,
        target_r,
        Pass1ProbeConfig {
            sample_rate_hz,
            max_fir_taps,
            max_kautz_pairs,
            hint: None,
        },
    )?;
    Ok((res.report, res.model))
}

#[cfg(test)]
mod tests {
    use core::f64::consts::TAU;

    use super::*;

    #[test]
    fn test_pass1_probe_recovers_fir_resonant_poles_and_comb_delay() {
        let n = 1_024_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                0.4 * (TAU * 220.0 * t).sin()
                    + 0.3 * (TAU * 1_100.0 * t).cos()
                    + if i == 40 { 0.8 } else { 0.0 }
            })
            .collect();
        let in_r = in_l.clone();

        let mut tgt_l = vec![0.0_f64; n];
        let (r_true, th_true) = (0.92_f64, 0.28_f64);
        let (mr, mi) = synth_kautz_unit_mode(r_true, th_true, &in_l);
        for t in 0..n {
            let fir = 0.6 * in_l[t] + if t >= 3 { 0.35 * in_l[t - 3] } else { 0.0 };
            let modal = 0.05 * mr[t] - 0.02 * mi[t];
            let comb = if t >= 12 { 0.25 * tgt_l[t - 12] } else { 0.0 };
            tgt_l[t] = fir + modal + comb;
        }
        let tgt_r = tgt_l.clone();

        let (rep, mdl) = run_pass1_probe(&in_l, &in_r, &tgt_l, &tgt_r, sr, 32, 8).unwrap();
        assert!(rep.op_p.analytic_energy > 0.0);
        assert!(rep.recovered_kautz_pairs > 0);
        assert!(mdl.fir_l[0] > 0.45 && mdl.fir_l[3] > 0.2);
        assert_eq!(mdl.op_n_pre_gamma_bc, 0.0);
        assert_eq!(mdl.op_n_pre_beta_q, 0.0);
        assert!(rep.initial_esr_lin < 0.25);

        // Steiglitz-McBride recovers plant poles cleanly under heavily colored input excitation.
        let mut col_in = vec![0.0_f64; n];
        let (mut s_c, mut seed) = (0.0_f64, 0x9E37_79B9_7F4A_7C15_u64);
        for v in &mut col_in {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let w = ((seed >> 33) as f64) / ((1_u64 << 31) as f64) - 0.5;
            s_c = 0.85 * s_c + w;
            *v = s_c;
        }
        let mut plant_y = vec![0.0_f64; n];
        let a1 = -2.0 * 0.90 * (0.45_f64).cos();
        let a2 = 0.90 * 0.90;
        for t in 2..n {
            plant_y[t] = col_in[t] - a1 * plant_y[t - 1] - a2 * plant_y[t - 2];
        }
        let sm_poles = steiglitz_mcbride_poles(&col_in, &plant_y, 2, 4);
        assert!(!sm_poles.is_empty());
        assert!((sm_poles[0].0 - 0.90).abs() < 0.05, "r={:?}", sm_poles[0]);
        assert!((sm_poles[0].1 - 0.45).abs() < 0.05, "th={:?}", sm_poles[0]);

        // Right-channel-only late delayed comb recovery (`42` > `32` early FIR taps, zero Left comb).
        let mut right_only_tgt = vec![0.0_f64; n];
        let left_flat: Vec<f64> = col_in.iter().map(|&v| 0.5 * v).collect();
        for t in 0..n {
            right_only_tgt[t] = 0.5 * col_in[t]
                + if t >= 42 {
                    0.65 * right_only_tgt[t - 42]
                } else {
                    0.0
                };
        }
        let (rep_r, _) =
            run_pass1_probe(&col_in, &col_in, &left_flat, &right_only_tgt, sr, 32, 8).unwrap();
        assert_eq!(rep_r.dominant_transport_delay_samples, 42.0);
        assert!(
            rep_r.estimated_loop_gain > 0.35,
            "g={}",
            rep_r.estimated_loop_gain
        );

        // Steep narrowband pure-LTI filter (`450..800 Hz` cascaded poles) must release any
        // incoherent `Op-T` boundary seed once `FIR + Kautz` solves the linear tail (`< 0.03`).
        let n_bp = 8_192_usize;
        let mut bp_in = vec![0.0_f64; n_bp];
        let mut rng = 0x1357_2468_9ACE_BDF0_u64;
        for (i, dst) in bp_in.iter_mut().enumerate() {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let white = ((rng >> 33) as f64) / ((1_u64 << 31) as f64) - 1.0;
            let t = (i as f64) / sr;
            let tone = 0.35 * (TAU * 550.0 * t).sin() + 0.25 * (TAU * 700.0 * t).cos();
            *dst = 0.45 * white + tone + if i == 64 { 0.7 } else { 0.0 };
        }
        let r_hp450 = (-TAU * 450.0 / sr).exp();
        let r_lp800 = (-TAU * 800.0 / sr).exp();
        let (mut s_hp, mut s_lp) = ([0.0_f64; 8], [0.0_f64; 8]);
        let mut bp_tgt = vec![0.0_f64; n_bp];
        for (i, &x) in bp_in.iter().enumerate() {
            let mut v = x;
            for st in &mut s_hp {
                let lp = (1.0 - r_hp450) * v + r_hp450 * *st;
                *st = lp;
                v -= lp;
            }
            for st in &mut s_lp {
                *st = (1.0 - r_lp800) * v + r_lp800 * *st;
                v = *st;
            }
            bp_tgt[i] = 3.5 * v;
        }
        let (rep_bp, mdl_bp) =
            run_pass1_probe(&bp_in, &bp_in, &bp_tgt, &bp_tgt, sr, 128, 32).unwrap();
        assert_eq!(rep_bp.dominant_transport_delay_samples, 0.0);
        assert_eq!(rep_bp.estimated_loop_gain, 0.0);
        assert!(mdl_bp.transport_quads.iter().all(|tq| tq.g_loop == 0.0));
        assert!(
            rep_bp.initial_esr_lin < 0.03,
            "steep LTI bandpass esr={}",
            rep_bp.initial_esr_lin
        );

        let bad_hint = Pass1PhysicsHint {
            delay_ms: Some(0.0),
            rt60_s: Some(f64::NAN),
            modulation_rate_hz: Some(-1.0),
            ..Pass1PhysicsHint::default()
        };
        let res_bp_h = run_pass1_probe_with_hint(
            &bp_in,
            &bp_in,
            &bp_tgt,
            &bp_tgt,
            Pass1ProbeConfig {
                sample_rate_hz: sr,
                max_fir_taps: 128,
                max_kautz_pairs: 32,
                hint: Some(&bad_hint),
            },
        )
        .unwrap();
        assert_eq!(res_bp_h.report.estimated_loop_gain, 0.0);

        // Subtle unhinted late feedforward echo (`-17 dB`, lag 220 > 128 FIR taps, white dry input
        // so `dry_rho < 0.35` and `crisp_rho < 0.25`) must NOT have Op-T wiped because Kautz does
        // not attenuate its post-FIR residual (`res_tot_e >= 0.5 * res_fir_tot_e`).
        let mut echo_in = vec![0.0_f64; n_bp];
        let mut erng = 0x2468_ACE0_1357_9BDF_u64;
        for dst in &mut echo_in {
            erng = erng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *dst = 0.5 * (((erng >> 33) as f64) / ((1_u64 << 31) as f64) - 1.0);
        }
        let echo_tgt: Vec<f64> = (0..n_bp)
            .map(|i| {
                echo_in[i]
                    + if i >= 220 {
                        0.14 * echo_in[i - 220]
                    } else {
                        0.0
                    }
            })
            .collect();
        let (rep_echo, mdl_echo) =
            run_pass1_probe(&echo_in, &echo_in, &echo_tgt, &echo_tgt, sr, 128, 32).unwrap();
        assert!(
            rep_echo.estimated_loop_gain > 0.0
                && mdl_echo.transport_quads.iter().any(|tq| tq.g_loop > 0.0),
            "subtle late echo must keep Op-T seeded (g={})",
            rep_echo.estimated_loop_gain
        );
    }
}
