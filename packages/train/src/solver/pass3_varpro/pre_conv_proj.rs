//! Alternating Hammerstein–Wiener `pre_conv` nonlinear bus projection + Kautz re-fit for
//! cascaded nonlinear + resonant/dispersive black-box targets (`ref[train.pipeline.nested-varpro-schur]`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};

use crate::error::TrainError;
use crate::runner::TrainRunner;

use super::super::model::cascade::{
    CascadeCrossbarSpec, run_causal_core_cascade_with_crossbar, upsample_stereo_2x_f64,
};
use super::super::model::{KautzPairF64, NUM_KAUTZ_PAIRS_ULTRA, SonaModelF64};
use super::super::pass1_probe::kautz_fit::{
    KautzFitInputs, append_log_octave_h2_candidates, fit_early_fir_taps, fit_joint_kautz_bank,
    residual_after_fir,
};
use super::super::pass2_opt::crossbar::{eval_crossbar_forward, eval_crossbar_forward_os2x};
use super::features::{build_pre_conv_bus, eval_pass3_frontend_1x};
use super::{
    Pass3InnerOutcome, Pass3VarProReport, StereoVarProPairs,
    run_pass3_varpro_readout_with_os2x_fallback,
};

#[inline]
fn snap_fp16_bounded(v: f64, lo: f64, hi: f64) -> f64 {
    let clamped = if v.is_finite() {
        (v as f32).clamp(lo as f32, hi as f32)
    } else {
        0.0
    };
    let s = f64::from(fp16_to_f32(f32_to_fp16(clamped)));
    if s == 0.0 { 0.0 } else { s }
}

fn post_warmup_stereo_rms(l: &[f64], r: &[f64], w: usize, n: usize) -> f64 {
    let m = n.saturating_sub(w);
    if m == 0 {
        return 0.0;
    }
    let e: f64 = l[w..n]
        .iter()
        .zip(&r[w..n])
        .map(|(&a, &b)| a * a + b * b)
        .sum();
    (0.5 * e / (m as f64)).sqrt()
}

/// Attempts one non-destructive alternating Hammerstein–Wiener Pass-3 projection onto the shared
/// `pre_conv` bus (`pre_conv_u_gain` + `w_ro[0..16][0..6]`) whenever Ultra normalized MSE remains
/// above `-18.2 dB`, committing only when Ultra MSE strictly drops by `> 8%` without regressing Lite.
pub(super) fn maybe_refine_with_pre_conv_projection(
    runner: &TrainRunner,
    model: &mut SonaModelF64,
    pairs: StereoVarProPairs<'_>,
    w: usize,
    cur_report: Pass3VarProReport,
    cur_outcome: Pass3InnerOutcome,
) -> Result<Pass3VarProReport, TrainError> {
    let StereoVarProPairs {
        in_l,
        in_r,
        target_l,
        target_r,
    } = pairs;
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let m = n.saturating_sub(w).max(1);
    let tgt_var = 0.5
        * (target_l[w..n].iter().map(|&v| v * v).sum::<f64>()
            + target_r[w..n].iter().map(|&v| v * v).sum::<f64>())
        / (m as f64);
    if model.active_fir_taps == 0
        || !tgt_var.is_finite()
        || tgt_var <= 1.0e-12
        || cur_outcome.ultra_mse / tgt_var <= 0.015
    {
        return Ok(cur_report);
    }

    let (lite_dir, lite_w_ro) = model.tier_overrides[0]
        .as_deref()
        .map_or((model.direct_s2_u_gain, &model.w_ro), |p| {
            (p.direct_s2_u_gain, &p.w_ro)
        });
    let mut cand = model.clone();
    cand.pre_conv_u_gain = lite_dir;
    for l in 0..16 {
        cand.w_ro[l][0..6].copy_from_slice(&lite_w_ro[l][6..12]);
    }
    for l in 16..40 {
        cand.w_ro[l][0..6].fill(0.0);
    }
    let has_signal = cand.pre_conv_u_gain.iter().any(|&v| v.abs() > 1.0e-9)
        || cand.w_ro[..16]
            .iter()
            .any(|row| row[0..6].iter().any(|&v| v.abs() > 1.0e-9));
    if !has_signal {
        return Ok(cur_report);
    }

    let (u_sense_l, u_sense_r, ux_l, ux_r) = eval_pass3_frontend_1x(&cand, &in_l[..n], &in_r[..n]);
    let rho_1x = 48_000.0 / cand.sample_rate_hz.max(8_000.0);
    let cb_1x = eval_crossbar_forward(&cand, &u_sense_l, &u_sense_r, 4, false);
    let cas_1x = run_causal_core_cascade_with_crossbar(
        runner,
        &cand,
        &ux_l,
        &ux_r,
        CascadeCrossbarSpec {
            active_quads: 4,
            rho_core: rho_1x,
            write_checkpoints: false,
            crossbar_add: cb_1x
                .active
                .then_some((cb_1x.cb_add_l.as_slice(), cb_1x.cb_add_r.as_slice())),
        },
    )?;
    cand.pre_conv_u_gain = [0.0, 0.0];
    let saved_pc_hp = cand.pre_conv_hp_gain;
    cand.pre_conv_hp_gain = [0.0, 0.0];
    let (core_only_l, core_only_r) = build_pre_conv_bus(
        &cand,
        &ux_l,
        &ux_r,
        &cas_1x.feature_outputs,
        4,
        rho_1x,
        false,
    );
    let rms_core_only = post_warmup_stereo_rms(&core_only_l, &core_only_r, w, n);
    if rms_core_only < 0.15 * tgt_var.sqrt() {
        return Ok(cur_report);
    }
    cand.pre_conv_u_gain = lite_dir;
    cand.pre_conv_hp_gain = saved_pc_hp;

    let (bus_ux_l, bus_ux_r, bus_feats, bus_quads, bus_rho, bus_is_2x) =
        if cand.oversample_upper_tiers_2x {
            let (up_l, up_r) = upsample_stereo_2x_f64(&ux_l, &ux_r, cand.os_up_quad_coeff);
            let rho_2x = 0.5 * rho_1x;
            let cb_2x = eval_crossbar_forward_os2x(&cand, &u_sense_l, &u_sense_r, 10, true);
            let cas_2x = run_causal_core_cascade_with_crossbar(
                runner,
                &cand,
                &up_l,
                &up_r,
                CascadeCrossbarSpec {
                    active_quads: 10,
                    rho_core: rho_2x,
                    write_checkpoints: false,
                    crossbar_add: cb_2x
                        .active
                        .then_some((cb_2x.cb_add_l.as_slice(), cb_2x.cb_add_r.as_slice())),
                },
            )?;
            (up_l, up_r, cas_2x.feature_outputs, 10_usize, rho_2x, true)
        } else {
            (
                ux_l.clone(),
                ux_r.clone(),
                cas_1x.feature_outputs,
                4_usize,
                rho_1x,
                false,
            )
        };
    let (pre_raw_l, pre_raw_r) = build_pre_conv_bus(
        &cand, &bus_ux_l, &bus_ux_r, &bus_feats, bus_quads, bus_rho, bus_is_2x,
    );

    let rms_ux = post_warmup_stereo_rms(&ux_l, &ux_r, w, n);
    let rms_pre = post_warmup_stereo_rms(&pre_raw_l, &pre_raw_r, w, n);
    if rms_ux <= 1.0e-12 || rms_pre <= 1.0e-12 {
        return Ok(cur_report);
    }
    let scale = (rms_ux / rms_pre).clamp(0.0625, 16.0);
    for c in 0..2 {
        cand.pre_conv_u_gain[c] = snap_fp16_bounded(cand.pre_conv_u_gain[c] * scale, -8.0, 8.0);
    }
    for l in 0..16 {
        for col in 0..6 {
            cand.w_ro[l][col] = snap_fp16_bounded(cand.w_ro[l][col] * scale, -4.0, 4.0);
        }
    }

    let (pre_l, pre_r) = build_pre_conv_bus(
        &cand, &bus_ux_l, &bus_ux_r, &bus_feats, bus_quads, bus_rho, bus_is_2x,
    );
    if post_warmup_stereo_rms(&pre_l, &pre_r, w, n) <= 1.0e-12 {
        return Ok(cur_report);
    }

    let fir_onset_n = cand.active_fir_taps.min(16);
    let onset_fir_l = fit_early_fir_taps(&pre_l, &target_l[..n], fir_onset_n);
    let onset_fir_r = fit_early_fir_taps(&pre_r, &target_r[..n], fir_onset_n);
    let res_16_l = residual_after_fir(&pre_l, &target_l[..n], &onset_fir_l, fir_onset_n);
    let res_16_r = residual_after_fir(&pre_r, &target_r[..n], &onset_fir_r, fir_onset_n);

    let active_k = cand.active_kautz_pairs.min(cand.kautz.len());
    let mut poles: Vec<(f64, f64)> = cand.kautz[..active_k]
        .iter()
        .filter(|kp| kp.r_low.is_finite() && kp.theta.is_finite() && kp.r_low > 0.0)
        .map(|kp| (kp.r_low, kp.theta))
        .collect();
    append_log_octave_h2_candidates(&mut poles, cand.sample_rate_hz, NUM_KAUTZ_PAIRS_ULTRA);
    let start16 = w.max(fir_onset_n).min(n / 4);
    let (accepted_kautz, _, _) = fit_joint_kautz_bank(&KautzFitInputs {
        candidates: &poles,
        input_l: &pre_l,
        input_r: &pre_r,
        res_in_l: &res_16_l,
        res_in_r: &res_16_r,
        start: start16,
        k_limit: NUM_KAUTZ_PAIRS_ULTRA,
    });
    cand.kautz = [KautzPairF64::default(); NUM_KAUTZ_PAIRS_ULTRA];
    if accepted_kautz.is_empty() {
        cand.active_kautz_pairs = 0;
    } else {
        let k_len = accepted_kautz.len().min(NUM_KAUTZ_PAIRS_ULTRA);
        for (idx, pair) in accepted_kautz.into_iter().take(k_len).enumerate() {
            cand.kautz[idx] = pair;
        }
        cand.active_kautz_pairs = k_len;
    }

    cand.tier_overrides = [None, None, None, None];
    cand.prefix_override_needed = [false; 4];
    let (cand_report, cand_outcome) = run_pass3_varpro_readout_with_os2x_fallback(
        runner,
        &mut cand,
        &super::super::model::StereoTrainingPairs {
            in_l,
            in_r,
            target_l,
            target_r,
        },
        w,
        super::Pass3VarProOptions::default(),
    )?;

    if cand_outcome.ultra_mse < cur_outcome.ultra_mse * 0.90
        && cand_outcome.lite_mse <= cur_outcome.lite_mse * 1.10
    {
        *model = cand;
        Ok(cand_report)
    } else {
        Ok(cur_report)
    }
}

#[cfg(test)]
mod tests {
    use core::f64::consts::TAU;

    use dnam_vm::bin::strip_dnam_bin_to_tier;
    use dnam_vm::math::DefaultMathBackend;
    use dnam_vm::{SonaVm, Tier};

    use crate::solver::pass1_probe::kautz_fit::synth_kautz_unit_mode;
    use crate::solver::pass3_varpro::run_pass3_varpro_readout;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};

    use super::*;

    fn vm_post_warmup_mse(
        model: &SonaModelF64,
        tier: Tier,
        in_l: &[f64],
        tgt_l: &[f64],
        w: usize,
    ) -> f64 {
        let full_bin = encode_model_to_dnam_bin(model, Pass5EmitConfig::default()).unwrap();
        let slice = strip_dnam_bin_to_tier(&full_bin, tier).unwrap();
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(slice, tier, 48_000.0).unwrap();
        let in32: Vec<f32> = in_l.iter().map(|&x| x as f32).collect();
        let (mut ol, mut or) = (vec![0.0_f32; in32.len()], vec![0.0_f32; in32.len()]);
        vm.process_block(&in32, &in32, &mut ol, &mut or).unwrap();
        let sq: f64 = ol[w..]
            .iter()
            .zip(&tgt_l[w..])
            .map(|(&y, &t)| {
                let d = t - f64::from(y);
                d * d
            })
            .sum();
        sq / (in32.len().saturating_sub(w).max(1) as f64)
    }

    #[test]
    fn test_pre_conv_proj_improves_nonlinear_then_resonant_filter_target() {
        let runner = TrainRunner::cpu();
        let n = 2_048_usize;
        let w = 64_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|t| {
                let ts = (t as f64) / sr;
                0.28 * (TAU * 170.0 * ts).sin()
                    + 0.22 * (TAU * 430.0 * ts).cos()
                    + 0.16 * (TAU * 1_150.0 * ts).sin()
            })
            .collect();
        let in_r = in_l.clone();
        let nl: Vec<f64> = in_l.iter().map(|&x| (4.0 * x).tanh()).collect();
        let (mr, _mi) = synth_kautz_unit_mode(0.90, 0.14, &nl);
        let mut tgt_l = vec![0.0_f64; n];
        for t in 0..n {
            let tap24 = if t >= 24 { 0.55 * nl[t - 24] } else { 0.0 };
            let tap40 = if t >= 40 { -0.35 * nl[t - 40] } else { 0.0 };
            let res16 = if t >= 16 { 0.18 * mr[t - 16] } else { 0.0 };
            tgt_l[t] = 0.65 * nl[t] + tap24 + tap40 + res16;
        }
        let tgt_r = tgt_l.clone();

        let make_base_model = || {
            let mut m = SonaModelF64::new(sr);
            m.oversample_upper_tiers_2x = false;
            m.active_fir_taps = 64;
            m.active_kautz_pairs = 24;
            for k in 0..24 {
                m.kautz[k].r_low = 0.85;
                m.kautz[k].theta = 0.1 + 0.08 * (k as f64);
                m.kautz[k].b_gamma = 0.5;
                m.kautz[k].c_re_l = 0.12;
                m.kautz[k].c_re_r = 0.12;
            }
            m.core_quads[0].svf_g = [0.22; 4];
            m.core_quads[0].tone_dir = [1.0; 4];
            m.core_quads[0].in_mid_weight = [1.0; 4];
            m.core_quads[0].in_side_weight = [1.0; 4];
            m.core_quads[0].adaa_drive = [4.0, 2.0, 6.0, 8.0];
            m
        };

        let mut baseline = make_base_model();
        let base_rep = run_pass3_varpro_readout_with_os2x_fallback(
            &runner,
            &mut baseline,
            &crate::solver::model::StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_l,
                target_r: &tgt_r,
            },
            w,
            crate::Pass3VarProOptions::default(),
        )
        .unwrap()
        .0;

        let mut projected = make_base_model();
        let proj_rep =
            run_pass3_varpro_readout(&runner, &mut projected, &in_l, &in_r, &tgt_l, &tgt_r, w)
                .unwrap();

        let has_pre_conv_weights = projected.w_ro[..16]
            .iter()
            .any(|row| row[0..6].iter().any(|&v| v.abs() > 1.0e-4));
        assert!(
            has_pre_conv_weights,
            "expected non-zero w_ro[..][0..6] after projection"
        );
        for kp in &projected.kautz[projected.active_kautz_pairs..] {
            assert_eq!(kp.c_re_l, 0.0);
            assert_eq!(kp.c_im_l, 0.0);
        }
        let gain_db = 10.0 * (base_rep.tier_mse[3] / proj_rep.tier_mse[3].max(1.0e-18)).log10();
        assert!(
            gain_db >= 3.0,
            "expected >= 3 dB Ultra MSE improvement from pre_conv projection, got {gain_db:.2} dB (base={}, proj={})",
            base_rep.tier_mse[3],
            proj_rep.tier_mse[3]
        );
        let vm_base_ultra = vm_post_warmup_mse(&baseline, Tier::Ultra, &in_l, &tgt_l, w);
        let vm_proj_ultra = vm_post_warmup_mse(&projected, Tier::Ultra, &in_l, &tgt_l, w);
        let vm_gain_db = 10.0 * (vm_base_ultra / vm_proj_ultra.max(1.0e-18)).log10();
        assert!(
            vm_gain_db >= 3.0,
            "expected >= 3 dB SonaVm Ultra MSE improvement, got {vm_gain_db:.2} dB (base={vm_base_ultra}, proj={vm_proj_ultra})"
        );
        let _vm_proj_lite = vm_post_warmup_mse(&projected, Tier::Lite, &in_l, &tgt_l, w);
    }
}
