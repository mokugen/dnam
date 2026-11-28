//! Bridge applying `GRA` outputs onto `SONA v4.0` [`CoreQuadF64`] & [`TransportQuadF64`]
//! targets without modifying `v3` `hoist_stage_targets_onto_model`, plus clamp-accurate reverse
//! pullback (`ref[dsp.ctrl.gra.rational-and-entmax]`).

use dnam_kernels::{MAX_OUT_GAIN, MAX_PARALRU_DUHEM_GAMMA_F32, MIN_TPT_G, TANK_MAX_G_LIN_F32};

use crate::solver::model::{CoreQuadF64, TransportQuadF64};

use super::adjoint::GraOutputsGrad;
use super::forward::GraOutputs;

/// Bridge applying `GRA` outputs onto `SONA v4.0` `CoreQuadF64` & `TransportQuadF64` targets
/// without modifying `v3` `hoist_stage_targets_onto_model`, plus reverse pullback.
#[derive(Clone, Copy, Debug, Default)]
pub struct GraCoreOpTBridge;

impl GraCoreOpTBridge {
    /// Applies `GRA` `stage_targets` (`0..32`) and `op_t_q` (`0..16`) onto `core_quads` and
    /// `transport_quads` (preserving `StarCenter` 0-ULP identity when `gra` is zero).
    pub fn apply_to_core_and_opt(
        gra: &GraOutputs,
        base_core: &[CoreQuadF64; 10],
        base_opt: &[TransportQuadF64; 4],
        out_core: &mut [CoreQuadF64; 10],
        out_opt: &mut [TransportQuadF64; 4],
    ) {
        *out_core = *base_core;
        *out_opt = *base_opt;
        let g_oct = gra.stage_targets[0].clamp(-3.0, 3.0);
        let g_scale = 2.0_f64.powf(g_oct);
        let k_delta = gra.stage_targets[1].clamp(-0.8, 1.5);
        for (q, cq) in out_core.iter_mut().enumerate() {
            let block_idx = dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0;
            let adaa_oct = gra.stage_targets[2 + block_idx].clamp(-1.5, 1.5);
            let adaa_scale = 2.0_f64.powf(adaa_oct);
            let lru_bias_d = gra.stage_targets[6 + block_idx].clamp(-2.0, 2.0);
            let gain_mod = if q < 6 {
                gra.g_edge[q].clamp(-1.0, 1.0)
            } else {
                0.0
            };
            for l in 0..4 {
                cq.svf_g[l] = (cq.svf_g[l] * g_scale).clamp(f64::from(MIN_TPT_G), 64.0);
                cq.svf_k[l] = (cq.svf_k[l] + k_delta).clamp(0.0, 8.0);
                cq.adaa_drive[l] = (cq.adaa_drive[l] * adaa_scale).clamp(-32.0, 32.0);
                cq.lru_bias_drive[l] = (cq.lru_bias_drive[l] + lru_bias_d).clamp(-16.0, 16.0);
                cq.out_gain[l] = (cq.out_gain[l] * (1.0 + gain_mod))
                    .clamp(-f64::from(MAX_OUT_GAIN), f64::from(MAX_OUT_GAIN));
            }
        }

        let g_loop_max = f64::from(TANK_MAX_G_LIN_F32);
        let damp_max = f64::from(MAX_PARALRU_DUHEM_GAMMA_F32).min(0.999_95);
        let delay_scale = 2.0_f64.powf(gra.stage_targets[17].clamp(-3.0, 3.0));
        let lfo_depth_scale = 2.0_f64.powf(gra.stage_targets[20].clamp(-4.0, 4.0));
        let in_gain_scale = 2.0_f64.powf(gra.stage_targets[23].clamp(-4.0, 4.0));
        let g_delta_13 = gra.stage_targets[13].clamp(-1.998, 1.998);
        for (q, tq) in out_opt.iter_mut().enumerate() {
            let q_base = q * 4;
            let q_delay_scale = 2.0_f64.powf(gra.op_t_q[q_base].clamp(-1.5, 1.5));
            let q_g_delta = gra.op_t_q[q_base + 1];
            let q_damp_delta = gra.op_t_q[q_base + 2];
            let q_lfo_delta = gra.op_t_q[q_base + 3];
            tq.g_loop = (tq.g_loop + g_delta_13 + q_g_delta).clamp(-g_loop_max, g_loop_max);
            tq.fb_psi = (tq.fb_psi + gra.stage_targets[14])
                .clamp(-core::f64::consts::PI, core::f64::consts::PI);
            tq.freeze_mix = (tq.freeze_mix + gra.stage_targets[16]).clamp(0.0, 0.999);
            tq.ramp_rate_hz = (tq.ramp_rate_hz + gra.stage_targets[18]).clamp(-64.0, 64.0);
            tq.lfo_rate_hz =
                (tq.lfo_rate_hz + gra.stage_targets[19] + q_lfo_delta).clamp(0.0, 16.0);
            tq.damp_pole =
                (tq.damp_pole + gra.stage_targets[21] + q_damp_delta).clamp(0.0, damp_max);
            tq.window_depth = (tq.window_depth + gra.stage_targets[24]).clamp(0.0, 1.0);
            for l in 0..4 {
                tq.delay_samples[l] = (tq.delay_samples[l] * delay_scale * q_delay_scale
                    + gra.stage_targets[12])
                    .clamp(5.0, 48_000.0);
                tq.lfo_depths[l] = (tq.lfo_depths[l] * lfo_depth_scale).clamp(0.0, 512.0);
                tq.input_gains[l] = (tq.input_gains[l] * in_gain_scale).clamp(-2.0, 2.0);
                tq.allpass_g[l] = (tq.allpass_g[l] + gra.stage_targets[25]).clamp(-0.75, 0.75);
            }
        }
    }

    /// Pulls back physical `(d_core, d_opt)` gradients onto [`GraOutputsGrad`] along the
    /// exact bridge mapping in [`Self::apply_to_core_and_opt`], masking every input and output
    /// clamp strictly on its open interior interval.
    #[must_use]
    pub fn pullback_from_core_and_opt(
        gra: &GraOutputs,
        base_core: &[CoreQuadF64; 10],
        base_opt: &[TransportQuadF64; 4],
        d_core: &[CoreQuadF64; 10],
        d_opt: &[TransportQuadF64; 4],
    ) -> GraOutputsGrad {
        let mut grad = GraOutputsGrad::default();
        let ln2 = core::f64::consts::LN_2;
        let min_tpt_g = f64::from(MIN_TPT_G);
        let max_out_gain = f64::from(MAX_OUT_GAIN);
        let g_oct = gra.stage_targets[0].clamp(-3.0, 3.0);
        let g_scale = 2.0_f64.powf(g_oct);
        let g_oct_open = gra.stage_targets[0] > -3.0 && gra.stage_targets[0] < 3.0;
        let k_delta = gra.stage_targets[1].clamp(-0.8, 1.5);
        let k_delta_open = gra.stage_targets[1] > -0.8 && gra.stage_targets[1] < 1.5;

        for (q, (bc, dc)) in base_core.iter().zip(d_core).enumerate() {
            let block_idx = dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0;
            let adaa_oct = gra.stage_targets[2 + block_idx].clamp(-1.5, 1.5);
            let adaa_scale = 2.0_f64.powf(adaa_oct);
            let adaa_open =
                gra.stage_targets[2 + block_idx] > -1.5 && gra.stage_targets[2 + block_idx] < 1.5;
            let lru_bias_d = gra.stage_targets[6 + block_idx].clamp(-2.0, 2.0);
            let lru_open =
                gra.stage_targets[6 + block_idx] > -2.0 && gra.stage_targets[6 + block_idx] < 2.0;
            let (gain_mod, g_edge_open) = if q < 6 {
                (
                    gra.g_edge[q].clamp(-1.0, 1.0),
                    gra.g_edge[q] > -1.0 && gra.g_edge[q] < 1.0,
                )
            } else {
                (0.0, false)
            };
            for l in 0..4 {
                let raw_svf_g = bc.svf_g[l] * g_scale;
                if g_oct_open && raw_svf_g > min_tpt_g && raw_svf_g < 64.0 {
                    grad.d_stage_targets[0] += dc.svf_g[l] * raw_svf_g * ln2;
                }
                let raw_svf_k = bc.svf_k[l] + k_delta;
                if k_delta_open && raw_svf_k > 0.0 && raw_svf_k < 8.0 {
                    grad.d_stage_targets[1] += dc.svf_k[l];
                }
                let raw_adaa = bc.adaa_drive[l] * adaa_scale;
                if adaa_open && raw_adaa > -32.0 && raw_adaa < 32.0 {
                    grad.d_stage_targets[2 + block_idx] += dc.adaa_drive[l] * raw_adaa * ln2;
                }
                let raw_lru = bc.lru_bias_drive[l] + lru_bias_d;
                if lru_open && raw_lru > -16.0 && raw_lru < 16.0 {
                    grad.d_stage_targets[6 + block_idx] += dc.lru_bias_drive[l];
                }
                let raw_out_gain = bc.out_gain[l] * (1.0 + gain_mod);
                if g_edge_open && raw_out_gain > -max_out_gain && raw_out_gain < max_out_gain {
                    grad.d_g_edge[q] += dc.out_gain[l] * bc.out_gain[l];
                }
            }
        }

        let g_loop_max = f64::from(TANK_MAX_G_LIN_F32);
        let damp_max = f64::from(MAX_PARALRU_DUHEM_GAMMA_F32).min(0.999_95);
        let pi = core::f64::consts::PI;

        let t13_clamped = gra.stage_targets[13].clamp(-1.998, 1.998);
        let t13_open = gra.stage_targets[13] > -1.998 && gra.stage_targets[13] < 1.998;
        let t17_open = gra.stage_targets[17] > -3.0 && gra.stage_targets[17] < 3.0;
        let t20_open = gra.stage_targets[20] > -4.0 && gra.stage_targets[20] < 4.0;
        let t23_open = gra.stage_targets[23] > -4.0 && gra.stage_targets[23] < 4.0;

        let delay_scale = 2.0_f64.powf(gra.stage_targets[17].clamp(-3.0, 3.0));
        let lfo_depth_scale = 2.0_f64.powf(gra.stage_targets[20].clamp(-4.0, 4.0));
        let in_gain_scale = 2.0_f64.powf(gra.stage_targets[23].clamp(-4.0, 4.0));

        for (q, (bo, do_)) in base_opt.iter().zip(d_opt).enumerate() {
            let q_base = q * 4;
            let q_delay_oct = gra.op_t_q[q_base].clamp(-1.5, 1.5);
            let q_delay_scale = 2.0_f64.powf(q_delay_oct);
            let q_delay_open = gra.op_t_q[q_base] > -1.5 && gra.op_t_q[q_base] < 1.5;

            let raw_g_loop = bo.g_loop + t13_clamped + gra.op_t_q[q_base + 1];
            if raw_g_loop > -g_loop_max && raw_g_loop < g_loop_max {
                if t13_open {
                    grad.d_stage_targets[13] += do_.g_loop;
                }
                grad.d_op_t_q[q_base + 1] += do_.g_loop;
            }
            let raw_fb_psi = bo.fb_psi + gra.stage_targets[14];
            if raw_fb_psi > -pi && raw_fb_psi < pi {
                grad.d_stage_targets[14] += do_.fb_psi;
            }
            let raw_freeze = bo.freeze_mix + gra.stage_targets[16];
            if raw_freeze > 0.0 && raw_freeze < 0.999 {
                grad.d_stage_targets[16] += do_.freeze_mix;
            }
            let raw_ramp = bo.ramp_rate_hz + gra.stage_targets[18];
            if raw_ramp > -64.0 && raw_ramp < 64.0 {
                grad.d_stage_targets[18] += do_.ramp_rate_hz;
            }
            let raw_lfo_rate = bo.lfo_rate_hz + gra.stage_targets[19] + gra.op_t_q[q_base + 3];
            if raw_lfo_rate > 0.0 && raw_lfo_rate < 16.0 {
                grad.d_stage_targets[19] += do_.lfo_rate_hz;
                grad.d_op_t_q[q_base + 3] += do_.lfo_rate_hz;
            }
            let raw_damp = bo.damp_pole + gra.stage_targets[21] + gra.op_t_q[q_base + 2];
            if raw_damp > 0.0 && raw_damp < damp_max {
                grad.d_stage_targets[21] += do_.damp_pole;
                grad.d_op_t_q[q_base + 2] += do_.damp_pole;
            }
            let raw_win = bo.window_depth + gra.stage_targets[24];
            if raw_win > 0.0 && raw_win < 1.0 {
                grad.d_stage_targets[24] += do_.window_depth;
            }

            for l in 0..4 {
                let scaled_d = bo.delay_samples[l] * delay_scale * q_delay_scale;
                let raw_delay = scaled_d + gra.stage_targets[12];
                if raw_delay > 5.0 && raw_delay < 48_000.0 {
                    if t17_open {
                        grad.d_stage_targets[17] += do_.delay_samples[l] * scaled_d * ln2;
                    }
                    if q_delay_open {
                        grad.d_op_t_q[q_base] += do_.delay_samples[l] * scaled_d * ln2;
                    }
                    grad.d_stage_targets[12] += do_.delay_samples[l];
                }
                let raw_lfo_depth = bo.lfo_depths[l] * lfo_depth_scale;
                if t20_open && raw_lfo_depth > 0.0 && raw_lfo_depth < 512.0 {
                    grad.d_stage_targets[20] += do_.lfo_depths[l] * raw_lfo_depth * ln2;
                }
                let raw_in_gain = bo.input_gains[l] * in_gain_scale;
                if t23_open && raw_in_gain > -2.0 && raw_in_gain < 2.0 {
                    grad.d_stage_targets[23] += do_.input_gains[l] * raw_in_gain * ln2;
                }
                let raw_ap = bo.allpass_g[l] + gra.stage_targets[25];
                if raw_ap > -0.75 && raw_ap < 0.75 {
                    grad.d_stage_targets[25] += do_.allpass_g[l];
                }
            }
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones_core_grad() -> CoreQuadF64 {
        CoreQuadF64 {
            svf_g: [0.3; 4],
            svf_k: [0.4; 4],
            svf_k_nl: [0.0; 4],
            tone_lp: [0.0; 4],
            tone_bp: [0.0; 4],
            tone_hp: [0.0; 4],
            tone_ap: [0.0; 4],
            tone_dir: [0.0; 4],
            lru_alpha_rel: [0.0; 4],
            lru_alpha_atk: [0.0; 4],
            lru_kappa: [0.0; 4],
            lru_duhem_gamma: [0.0; 4],
            lru_signed_mix: [0.0; 4],
            lru_stereo_link: [0.0; 4],
            rect_sign: [0.0; 4],
            sag_depth: [0.0; 4],
            adaa_drive: [-0.2; 4],
            lru_bias_drive: [0.5; 4],
            wavefolder_mix: [0.0; 4],
            glu_mix: [0.0; 4],
            g0: [0.0; 4],
            g_inf: [0.0; 4],
            d_pre: [0.0; 4],
            se_w_sq: [0.0; 4],
            se_d_bus: [0.0; 4],
            lin_mix: [0.0; 4],
            out_gain: [0.75; 4],
            in_mid_weight: [0.0; 4],
            in_side_weight: [0.0; 4],
        }
    }

    fn ones_opt_grad() -> TransportQuadF64 {
        TransportQuadF64 {
            delay_samples: [0.2; 4],
            ramp_spans_samples: [0.0; 4],
            g_loop: -0.4,
            r_dc: 0.0,
            damp_pole: 0.35,
            lfo_rate_hz: 0.45,
            ramp_rate_hz: -0.25,
            lfo_depths: [0.15; 4],
            input_gains: [0.55; 4],
            wet_tap_l: [0.0; 4],
            wet_tap_r: [0.0; 4],
            allpass_g: [-0.3; 4],
            allpass_delays: [0; 4],
            fb_psi: 0.25,
            sigma_knee: 0.0,
            freeze_mix: 0.6,
            window_depth: -0.5,
            scat_psi: 0.0,
            lfo_phase_off_q16: [0; 4],
            ramp_phase_off_q16: [0; 4],
            ext_f32: false,
        }
    }

    fn bridge_loss(
        gra: &GraOutputs,
        base_core: &[CoreQuadF64; 10],
        base_opt: &[TransportQuadF64; 4],
        d_core: &[CoreQuadF64; 10],
        d_opt: &[TransportQuadF64; 4],
    ) -> f64 {
        let mut oc = *base_core;
        let mut oo = *base_opt;
        GraCoreOpTBridge::apply_to_core_and_opt(gra, base_core, base_opt, &mut oc, &mut oo);
        let mut acc = 0.0_f64;
        for (c, dc) in oc.iter().zip(d_core) {
            for l in 0..4 {
                acc += c.svf_g[l] * dc.svf_g[l]
                    + c.svf_k[l] * dc.svf_k[l]
                    + c.adaa_drive[l] * dc.adaa_drive[l]
                    + c.lru_bias_drive[l] * dc.lru_bias_drive[l]
                    + c.out_gain[l] * dc.out_gain[l];
            }
        }
        for (o, do_) in oo.iter().zip(d_opt) {
            acc += o.g_loop * do_.g_loop
                + o.fb_psi * do_.fb_psi
                + o.freeze_mix * do_.freeze_mix
                + o.ramp_rate_hz * do_.ramp_rate_hz
                + o.lfo_rate_hz * do_.lfo_rate_hz
                + o.damp_pole * do_.damp_pole
                + o.window_depth * do_.window_depth;
            for l in 0..4 {
                acc += o.delay_samples[l] * do_.delay_samples[l]
                    + o.lfo_depths[l] * do_.lfo_depths[l]
                    + o.input_gains[l] * do_.input_gains[l]
                    + o.allpass_g[l] * do_.allpass_g[l];
            }
        }
        acc
    }

    #[test]
    fn test_gra_core_opt_bridge_star_center_and_interior_fd() {
        let base_core = [CoreQuadF64 {
            adaa_drive: [2.5; 4],
            lru_bias_drive: [-1.2; 4],
            ..CoreQuadF64::default()
        }; 10];
        let base_opt = [TransportQuadF64 {
            delay_samples: [240.0; 4],
            g_loop: 0.25,
            damp_pole: 0.3,
            lfo_rate_hz: 1.5,
            ramp_rate_hz: 2.0,
            lfo_depths: [4.0; 4],
            input_gains: [0.5; 4],
            allpass_g: [0.1; 4],
            fb_psi: 0.3,
            freeze_mix: 0.2,
            window_depth: 0.35,
            ..TransportQuadF64::default()
        }; 4];

        let mut out_core = base_core;
        let mut out_opt = base_opt;
        GraCoreOpTBridge::apply_to_core_and_opt(
            &GraOutputs::default(),
            &base_core,
            &base_opt,
            &mut out_core,
            &mut out_opt,
        );
        assert_eq!(out_core, base_core);
        assert_eq!(out_opt, base_opt);

        let mut gra = GraOutputs::default();
        gra.stage_targets[0] = 0.25;
        gra.stage_targets[1] = 0.15;
        gra.stage_targets[2] = -0.35;
        gra.stage_targets[6] = 0.45;
        gra.stage_targets[12] = 12.0;
        gra.stage_targets[13] = -0.3;
        gra.stage_targets[14] = 0.4;
        gra.stage_targets[16] = 0.15;
        gra.stage_targets[17] = 0.2;
        gra.stage_targets[18] = -5.0;
        gra.stage_targets[19] = 0.8;
        gra.stage_targets[20] = 0.5;
        gra.stage_targets[21] = 0.12;
        gra.stage_targets[23] = -0.4;
        gra.stage_targets[24] = 0.2;
        gra.stage_targets[25] = -0.15;
        gra.g_edge = [0.1, -0.2, 0.3, -0.1, 0.25, -0.35];
        gra.op_t_q = [
            0.15, 0.05, -0.04, 0.2, -0.1, 0.08, 0.03, -0.15, 0.2, -0.02, 0.05, 0.1, -0.25, 0.04,
            -0.02, 0.05,
        ];

        let d_core = [ones_core_grad(); 10];
        let d_opt = [ones_opt_grad(); 4];
        let ana = GraCoreOpTBridge::pullback_from_core_and_opt(
            &gra, &base_core, &base_opt, &d_core, &d_opt,
        );

        let eps = 1.0e-5_f64;
        for idx in [0, 1, 2, 6, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 24, 25] {
            let (mut gp, mut gm) = (gra.clone(), gra.clone());
            gp.stage_targets[idx] += eps;
            gm.stage_targets[idx] -= eps;
            let fd = (bridge_loss(&gp, &base_core, &base_opt, &d_core, &d_opt)
                - bridge_loss(&gm, &base_core, &base_opt, &d_core, &d_opt))
                / (2.0 * eps);
            let a = ana.d_stage_targets[idx];
            assert!(
                (a - fd).abs() < 1.0e-8 * a.abs().max(fd.abs()).max(1.0),
                "interior stage_targets[{idx}]: ana={a} fd={fd}"
            );
        }
        for idx in 0..6 {
            let (mut gp, mut gm) = (gra.clone(), gra.clone());
            gp.g_edge[idx] += eps;
            gm.g_edge[idx] -= eps;
            let fd = (bridge_loss(&gp, &base_core, &base_opt, &d_core, &d_opt)
                - bridge_loss(&gm, &base_core, &base_opt, &d_core, &d_opt))
                / (2.0 * eps);
            let a = ana.d_g_edge[idx];
            assert!(
                (a - fd).abs() < 1.0e-8 * a.abs().max(fd.abs()).max(1.0),
                "interior g_edge[{idx}]: ana={a} fd={fd}"
            );
        }
        for idx in 0..16 {
            let (mut gp, mut gm) = (gra.clone(), gra.clone());
            gp.op_t_q[idx] += eps;
            gm.op_t_q[idx] -= eps;
            let fd = (bridge_loss(&gp, &base_core, &base_opt, &d_core, &d_opt)
                - bridge_loss(&gm, &base_core, &base_opt, &d_core, &d_opt))
                / (2.0 * eps);
            let a = ana.d_op_t_q[idx];
            assert!(
                (a - fd).abs() < 1.0e-8 * a.abs().max(fd.abs()).max(1.0),
                "interior op_t_q[{idx}]: ana={a} fd={fd}"
            );
        }
    }

    #[test]
    fn test_gra_core_opt_bridge_clamped_input_and_output_gradients_zero() {
        // 1. Input clamps on `stage_targets[13, 17, 20, 23]` and `op_t_q[0]` with interior outputs.
        let base_core = [CoreQuadF64 {
            svf_g: [100.0; 4],          // clamps > 64.0
            svf_k: [-1.0; 4],           // clamps < 0.0
            adaa_drive: [40.0; 4],      // clamps > 32.0
            lru_bias_drive: [-20.0; 4], // clamps < -16.0
            out_gain: [10.0; 4],        // clamps > MAX_OUT_GAIN
            ..CoreQuadF64::default()
        }; 10];
        let base_opt = [TransportQuadF64 {
            delay_samples: [200.0; 4],
            g_loop: 0.0,
            lfo_depths: [4.0; 4],
            input_gains: [0.5; 4],
            ..TransportQuadF64::default()
        }; 4];
        let d_core = [ones_core_grad(); 10];
        let d_opt = [ones_opt_grad(); 4];

        let mut gra_in_clamp = GraOutputs::default();
        gra_in_clamp.stage_targets[13] = 2.5; // > 1.998 (out_opt.g_loop=0.999? Wait: 0+1.998=1.998 > g_loop_max too, test separately below)
        gra_in_clamp.stage_targets[17] = 5.0; // > 3.0 (delay=200*8=1600 < 48_000 -> interior output)
        gra_in_clamp.stage_targets[20] = -5.0; // < -4.0 (lfo_depth=4/16=0.25 -> interior output)
        gra_in_clamp.stage_targets[23] = 5.0; // > 4.0 (wait: 0.5*16=8 > 2 -> use base input_gains=0.05 -> 0.8)
        gra_in_clamp.op_t_q[0] = 2.5; // > 1.5 (delay=1600*2.828=4525 < 48_000 -> interior output)
        gra_in_clamp.op_t_q[4] = 2.5;

        let mut opt_in = base_opt;
        opt_in[0].input_gains = [0.05; 4];
        opt_in[0].g_loop = -1.5; // -1.5 + 1.998 = 0.498 -> interior output while stage_targets[13]=2.5 is input-clamped!
        let ana_in = GraCoreOpTBridge::pullback_from_core_and_opt(
            &gra_in_clamp,
            &base_core,
            &opt_in,
            &d_core,
            &d_opt,
        );
        // Core outputs are all saturated -> zero gradients on core slots + g_edge.
        assert_eq!(ana_in.d_stage_targets[0], 0.0);
        assert_eq!(ana_in.d_stage_targets[1], 0.0);
        assert_eq!(ana_in.d_stage_targets[2], 0.0);
        assert_eq!(ana_in.d_stage_targets[6], 0.0);
        assert!(ana_in.d_g_edge.iter().all(|&g| g == 0.0));
        // Input-clamped transport slots yield 0.0 even though outputs are interior!
        assert_eq!(ana_in.d_stage_targets[13], 0.0);
        assert_eq!(ana_in.d_stage_targets[17], 0.0);
        assert_eq!(ana_in.d_stage_targets[20], 0.0);
        assert_eq!(ana_in.d_stage_targets[23], 0.0);
        assert_eq!(ana_in.d_op_t_q[0], 0.0);
        assert_eq!(ana_in.d_op_t_q[4], 0.0);
        // Unclamped `op_t_q[1]` (added to interior g_loop on quad 0) stays active!
        assert!((ana_in.d_op_t_q[1] - d_opt[0].g_loop).abs() < 1.0e-12);

        // 2. Output clamps on `out_opt` while `stage_targets` / `op_t_q` are strictly interior!
        let opt_out_clamp = [TransportQuadF64 {
            delay_samples: [50_000.0; 4], // > 48_000.0
            g_loop: 0.9,                  // + 0.3 = 1.2 > g_loop_max
            fb_psi: 3.1,                  // + 0.5 > PI
            freeze_mix: 0.95,             // + 0.2 > 0.999
            ramp_rate_hz: -60.0,          // - 10.0 < -64.0
            lfo_rate_hz: 15.5,            // + 1.0 > 16.0
            damp_pole: 0.98,              // + 0.1 > damp_max
            window_depth: 0.95,           // + 0.2 > 1.0
            lfo_depths: [600.0; 4],       // > 512.0
            input_gains: [-2.5; 4],       // < -2.0
            allpass_g: [0.7, -0.7, 0.7, -0.7],
            ..TransportQuadF64::default()
        }; 4];
        let mut gra_out_clamp = GraOutputs::default();
        gra_out_clamp.stage_targets[12] = 1.0;
        gra_out_clamp.stage_targets[13] = 0.3;
        gra_out_clamp.stage_targets[14] = 0.5;
        gra_out_clamp.stage_targets[16] = 0.2;
        gra_out_clamp.stage_targets[17] = 0.1;
        gra_out_clamp.stage_targets[18] = -10.0;
        gra_out_clamp.stage_targets[19] = 1.0;
        gra_out_clamp.stage_targets[20] = 0.1;
        gra_out_clamp.stage_targets[21] = 0.1;
        gra_out_clamp.stage_targets[23] = 0.1;
        gra_out_clamp.stage_targets[24] = 0.2;
        gra_out_clamp.stage_targets[25] = 0.2; // allpass_g lanes 0,2 > 0.75 (clamped), lanes 1,3 = -0.5 (open)
        gra_out_clamp.op_t_q[0] = 0.1;
        gra_out_clamp.op_t_q[1] = 0.05;
        gra_out_clamp.op_t_q[2] = 0.05;
        gra_out_clamp.op_t_q[3] = 0.05;

        let ana_out = GraCoreOpTBridge::pullback_from_core_and_opt(
            &gra_out_clamp,
            &base_core,
            &opt_out_clamp,
            &d_core,
            &d_opt,
        );
        for idx in [12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 24] {
            assert_eq!(
                ana_out.d_stage_targets[idx], 0.0,
                "clamped out_opt slot {idx} should be 0.0"
            );
        }
        assert_eq!(ana_out.d_op_t_q[0], 0.0);
        assert_eq!(ana_out.d_op_t_q[1], 0.0);
        assert_eq!(ana_out.d_op_t_q[2], 0.0);
        assert_eq!(ana_out.d_op_t_q[3], 0.0);

        // Partially clamped `allpass_g`: only the 2 unclamped lanes per quad contribute, matching FD!
        let d_opt_ap = [TransportQuadF64 {
            delay_samples: [0.0; 4],
            ramp_spans_samples: [0.0; 4],
            g_loop: 0.0,
            r_dc: 0.0,
            damp_pole: 0.0,
            lfo_rate_hz: 0.0,
            ramp_rate_hz: 0.0,
            lfo_depths: [0.0; 4],
            input_gains: [0.0; 4],
            wet_tap_l: [0.0; 4],
            wet_tap_r: [0.0; 4],
            allpass_g: [-0.3; 4],
            allpass_delays: [0; 4],
            fb_psi: 0.0,
            sigma_knee: 0.0,
            freeze_mix: 0.0,
            window_depth: 0.0,
            scat_psi: 0.0,
            lfo_phase_off_q16: [0; 4],
            ramp_phase_off_q16: [0; 4],
            ext_f32: false,
        }; 4];
        let d_core_zero = [CoreQuadF64 {
            svf_g: [0.0; 4],
            svf_k: [0.0; 4],
            svf_k_nl: [0.0; 4],
            tone_lp: [0.0; 4],
            tone_bp: [0.0; 4],
            tone_hp: [0.0; 4],
            tone_ap: [0.0; 4],
            tone_dir: [0.0; 4],
            lru_alpha_rel: [0.0; 4],
            lru_alpha_atk: [0.0; 4],
            lru_kappa: [0.0; 4],
            lru_duhem_gamma: [0.0; 4],
            lru_signed_mix: [0.0; 4],
            lru_stereo_link: [0.0; 4],
            rect_sign: [0.0; 4],
            sag_depth: [0.0; 4],
            adaa_drive: [0.0; 4],
            lru_bias_drive: [0.0; 4],
            wavefolder_mix: [0.0; 4],
            glu_mix: [0.0; 4],
            g0: [0.0; 4],
            g_inf: [0.0; 4],
            d_pre: [0.0; 4],
            se_w_sq: [0.0; 4],
            se_d_bus: [0.0; 4],
            lin_mix: [0.0; 4],
            out_gain: [0.0; 4],
            in_mid_weight: [0.0; 4],
            in_side_weight: [0.0; 4],
        }; 10];
        let eps = 1.0e-4_f64;
        let (mut gp, mut gm) = (gra_out_clamp.clone(), gra_out_clamp.clone());
        gp.stage_targets[25] += eps;
        gm.stage_targets[25] -= eps;
        let fd_25 = (bridge_loss(&gp, &base_core, &opt_out_clamp, &d_core_zero, &d_opt_ap)
            - bridge_loss(&gm, &base_core, &opt_out_clamp, &d_core_zero, &d_opt_ap))
            / (2.0 * eps);
        assert!(
            (ana_out.d_stage_targets[25] - fd_25).abs() < 1.0e-11,
            "ana={} fd={fd_25}",
            ana_out.d_stage_targets[25]
        );
    }
}
