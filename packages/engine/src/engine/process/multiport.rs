//! 4-port (`4CM`, `WDW`, `Sidechain`) stereo processing loops for [`DnamEngine`].

use super::{DnamAudioIo, micro_ramp_advance_clamped};
use crate::engine::{DnamEngine, EngineBackend};
use crate::math::{
    CERTIFIED_INPUT_ENVELOPE_MIN_F32, flush_recursive_state_f32, sanitize_input_sample,
    sanitize_output_sample,
};
use crate::types::FtzDazGuard;

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Low-level 4-port (`[inst_l, inst_r, ret_l, ret_r] -> [main_l, main_r, send_or_wdw_3, send_4]`)
    /// `SONA` stepping preserving strict `S_{ret->send} == 0` (`ref[dsp.router.ports]`).
    #[inline(never)]
    pub(crate) fn step_frame_4port_sona(
        &mut self,
        in_l: f32,
        in_r: f32,
        ret_l: f32,
        ret_r: f32,
        has_return: bool,
        mono_only: bool,
    ) -> [f32; 4] {
        self.sync_hot_core_tier();
        let in_r = if self.single_input { in_l } else { in_r };
        let in_l = sanitize_input_sample(
            in_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let in_r = sanitize_input_sample(
            in_r,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_l = sanitize_input_sample(
            ret_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_r = sanitize_input_sample(
            ret_r,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        if self.bypassed && !self.trails_enabled {
            return if mono_only {
                [in_l, in_l, 0.0, 0.0]
            } else {
                [in_l, in_r, 0.0, 0.0]
            };
        }

        let is_frozen = self.hot_core.tank_freeze();
        let live_scale = if !self.bypassed && self.freeze_mutes_dry && is_frozen {
            0.0
        } else {
            1.0
        };
        let (eff_in_l, eff_in_r, eff_ret_l, eff_ret_r) = if self.bypassed {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (
                in_l * live_scale,
                in_r * live_scale,
                ret_l * live_scale,
                ret_r * live_scale,
            )
        };

        let k_ctrl = self.zoh.control_decimation_host.max(1);
        self.ctrl_energy_accum = flush_recursive_state_f32(
            self.ctrl_energy_accum + 0.5 * (eff_in_l.abs() + eff_in_r.abs()),
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= k_ctrl {
            self.step_control_tick_3khz();
        }
        micro_ramp_advance_clamped(
            &mut self.current_stage_params,
            &self.stage_ramp_deltas,
            &self.control_bus.stage_targets,
            1.0,
        );
        self.hot_core
            .apply_control_bus_with_z_stage(&self.current_stage_params, &self.control_bus.z_stage);

        let prev_mono = self.hot_core.mono_out();
        if mono_only {
            self.hot_core.set_mono_out(true);
        }
        let ([mut wet_l, mut wet_r], [core_l, core_r], [send_l, send_r]) =
            self.hot_core.step_4port_frame(
                &mut self.io_router,
                eff_in_l,
                eff_in_r,
                eff_ret_l,
                eff_ret_r,
                has_return,
            );
        self.hot_core.set_mono_out(prev_mono);

        if !mono_only
            && !self.hot_core.mono_out()
            && !self.io_router.wdw_mode
            && (self.stereo_width - 1.0).abs() > 1e-6
        {
            let mid = 0.5 * (wet_l + wet_r);
            let side = 0.5 * (wet_l - wet_r);
            wet_l = mid + self.stereo_width * side;
            wet_r = mid - self.stereo_width * side;
        }

        let tier_trim = self.cached_tier_gain_trim[self.effective_tier() as usize];
        let eff_dry_g = if self.bypassed {
            1.0
        } else {
            self.mix_dry_gain * tier_trim
        };
        let eff_wet_g = if self.bypassed && !self.trails_enabled {
            0.0
        } else if self.bypassed {
            self.mix_wet_gain
        } else {
            self.mix_wet_gain * tier_trim
        };

        if self.io_router.wdw_mode {
            let dry_mix = self.hot_core.tank_dry_mix();
            let w_wet_l = wet_l - dry_mix * core_l;
            let w_wet_r = wet_r - dry_mix * core_r;
            let core_mid = 0.5 * (core_l + core_r);
            let wet_mid = 0.5 * (w_wet_l + w_wet_r);
            let wet_side = 0.5 * (w_wet_l - w_wet_r);
            let (wdw_dry_c, wdw_wet_l, wdw_wet_r) = crate::graph::eval_wdw_routing_outputs(
                self.io_router.wdw_tap_mode,
                core_mid,
                wet_mid,
                wet_side,
                self.stereo_width,
                in_l,
                in_r,
            );
            let out_l = sanitize_output_sample(eff_wet_g * wdw_wet_l, &mut self.saturation_flags);
            let out_r = if mono_only {
                out_l
            } else {
                sanitize_output_sample(eff_wet_g * wdw_wet_r, &mut self.saturation_flags)
            };
            let out_3 = sanitize_output_sample(wdw_dry_c, &mut self.saturation_flags);
            return [out_l, out_r, out_3, 0.0];
        }

        let eff_dry_in = if mono_only { 0.5 * (in_l + in_r) } else { in_l };
        let out_l = sanitize_output_sample(
            eff_dry_g * eff_dry_in + eff_wet_g * wet_l,
            &mut self.saturation_flags,
        );
        let out_r = if mono_only {
            out_l
        } else {
            sanitize_output_sample(
                eff_dry_g * in_r + eff_wet_g * wet_r,
                &mut self.saturation_flags,
            )
        };
        let out_3 = sanitize_output_sample(send_l, &mut self.saturation_flags);
        let out_4 = sanitize_output_sample(send_r, &mut self.saturation_flags);
        [out_l, out_r, out_3, out_4]
    }

    /// Internal 4x oversampled 4-port stepping using `self.oversampler` and `self.oversampler_aux`.
    #[inline(never)]
    pub(crate) fn step_frame_4x_4port_sona(
        &mut self,
        in_l: f32,
        in_r: f32,
        ret_l: f32,
        ret_r: f32,
        has_return: bool,
        mono_only: bool,
    ) -> [f32; 4] {
        self.sync_hot_core_tier();
        let in_r = if self.single_input { in_l } else { in_r };
        let in_l = sanitize_input_sample(
            in_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let in_r = sanitize_input_sample(
            in_r,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_l = sanitize_input_sample(
            ret_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let ret_r = sanitize_input_sample(
            ret_r,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        );
        let up_main = self.oversampler.upsample_4x(in_l, in_r);
        let up_aux = if has_return {
            self.oversampler_aux.upsample_4x(ret_l, ret_r)
        } else {
            [[0.0_f32; 2]; 4]
        };
        let mut proc_main = [[0.0_f32; 2]; 4];
        let mut proc_aux = [[0.0_f32; 2]; 4];
        for m in 0..4 {
            let [ol, or, o3, o4] = self.step_frame_4port_sona(
                up_main[m][0],
                up_main[m][1],
                up_aux[m][0],
                up_aux[m][1],
                has_return,
                mono_only,
            );
            proc_main[m] = [ol, or];
            proc_aux[m] = [o3, o4];
        }
        let [dl, dr] = self.oversampler.downsample_4x(proc_main);
        let [d3, d4] = self.oversampler_aux.downsample_4x(proc_aux);
        [
            sanitize_output_sample(dl, &mut self.saturation_flags),
            sanitize_output_sample(dr, &mut self.saturation_flags),
            sanitize_output_sample(d3, &mut self.saturation_flags),
            sanitize_output_sample(d4, &mut self.saturation_flags),
        ]
    }

    /// Processes a multi-port audio block (`Stage 0` 8-port router contract: `in_1..4`, `out_1..4`).
    pub fn process_block_io(&mut self, io: DnamAudioIo<'_>) {
        let _ftz = FtzDazGuard::new();
        self.hot_core.cold.max_op_n_drive = 0.0;
        let mut num_frames = io.in_1.len().min(io.out_1.len());
        if let Some(s) = io.in_2 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(s) = io.in_3 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(s) = io.in_4 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(ref s) = io.out_2 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(ref s) = io.out_3 {
            num_frames = num_frames.min(s.len());
        }
        if let Some(ref s) = io.out_4 {
            num_frames = num_frames.min(s.len());
        }
        self.tempo_state.advance_frames(num_frames);

        let has_return = io.in_3.is_some() || io.in_4.is_some();
        let mono_only = io.out_2.is_none() && io.out_3.is_none() && io.out_4.is_none();
        let DnamAudioIo {
            in_1,
            in_2,
            in_3,
            in_4,
            out_1,
            mut out_2,
            mut out_3,
            mut out_4,
        } = io;

        for i in 0..num_frames {
            let il = in_1[i];
            let ir = in_2.map_or(il, |s| s[i]);
            let rl = in_3.map_or(0.0, |s| s[i]);
            let rr = in_4.map_or(rl, |s| s[i]);
            let [ol, or, o3, o4] = if self.offline_bounce {
                self.step_frame_4x_4port_sona(il, ir, rl, rr, has_return, mono_only)
            } else {
                self.step_frame_4port_sona(il, ir, rl, rr, has_return, mono_only)
            };
            out_1[i] = ol;
            if let Some(ref mut s2) = out_2 {
                s2[i] = or;
            }
            if let Some(ref mut s3) = out_3 {
                s3[i] = o3;
            }
            if let Some(ref mut s4) = out_4 {
                s4[i] = o4;
            }
        }
    }

    /// Processes a 4-Cable Method (`4CM`) block (`[inst_l, inst_r, ret_l, ret_r] -> [main_l, main_r, send_l, send_r]`).
    #[expect(
        clippy::too_many_arguments,
        reason = "4-Cable Method exposes explicit 4 stereo I/O port buffers"
    )]
    pub fn process_block_4cm(
        &mut self,
        inst_l: &[f32],
        inst_r: &[f32],
        ret_l: &[f32],
        ret_r: &[f32],
        main_l: &mut [f32],
        main_r: &mut [f32],
        send_l: &mut [f32],
        send_r: &mut [f32],
    ) {
        let prev_wdw = self.io_router.wdw_mode;
        self.io_router.wdw_mode = false;
        self.process_block_io(DnamAudioIo {
            in_1: inst_l,
            in_2: Some(inst_r),
            in_3: Some(ret_l),
            in_4: Some(ret_r),
            out_1: main_l,
            out_2: Some(main_r),
            out_3: Some(send_l),
            out_4: Some(send_r),
        });
        self.io_router.wdw_mode = prev_wdw;
    }

    /// Processes a Wet/Dry/Wet (`WDW`) 3-output block (`[in_l, in_r] -> [wet_l, wet_r, dry_center]`).
    pub fn process_block_wdw(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        wet_l: &mut [f32],
        wet_r: &mut [f32],
        dry_center: &mut [f32],
    ) {
        let prev_wdw = self.io_router.wdw_mode;
        self.io_router.wdw_mode = true;
        self.process_block_io(DnamAudioIo {
            in_1: in_l,
            in_2: Some(in_r),
            in_3: None,
            in_4: None,
            out_1: wet_l,
            out_2: Some(wet_r),
            out_3: Some(dry_center),
            out_4: None,
        });
        self.io_router.wdw_mode = prev_wdw;
    }

    /// Processes a sidechain / auxiliary return stereo block (`[main_l, main_r, aux_l, aux_r] -> [out_l, out_r]`).
    pub fn process_block_sidechain(
        &mut self,
        main_l: &[f32],
        main_r: &[f32],
        aux_l: &[f32],
        aux_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        self.process_block_io(DnamAudioIo {
            in_1: main_l,
            in_2: Some(main_r),
            in_3: Some(aux_l),
            in_4: Some(aux_r),
            out_1: out_l,
            out_2: Some(out_r),
            out_3: None,
            out_4: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::engine::DnamEngine;
    use crate::graph::SonaMatryoshkaModel;
    use crate::graph::weights::pack_sona_matryoshka_to_arena;
    use crate::math::ScalarBackend;
    use crate::schema::DnamJsonHeader;
    use crate::types::Tier;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_4cm_wdw_sidechain_and_bypass_routing() {
        let _ftz = crate::types::FtzDazGuard::new();
        let mut header = DnamJsonHeader::default_reference();
        let mut mat = SonaMatryoshkaModel::dual_rate_default(false);
        for t in 0..4 {
            mat.readouts[t].pre_cab_u_gain = [0.25, 0.25];
            mat.readouts[t].direct_s2_u_gain = [0.5, 0.5];
            mat.readouts[t].fir_l[t * 16] = 0.5;
            mat.readouts[t].fir_r[t * 16] = 0.5;
            mat.stage4[t].wet_mix = 0.25;
        }
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        pack_sona_matryoshka_to_arena(&mat, &mut arena);
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let mut eng = DnamEngine::<ScalarBackend>::from_bytes_with_max_tier(
                &container,
                tier,
                Tier::Ultra,
                48_000,
            )
            .unwrap();
            let y = eng.step_frame_sona(0.25, -0.20, false);
            assert!(y[0].is_finite() && y[1].is_finite());
        }

        // Verify 4CM return-to-send isolation (S_{ret->send} == 0) and Return -> Stage II injection
        let mut eng_4cm_a =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_4cm_b =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let inst_l = [0.2f32; 64];
        let inst_r = [-0.15f32; 64];
        let ret_zero = [0.0f32; 64];
        let ret_sig = [0.4f32; 64];
        let (mut main_l_a, mut main_r_a, mut send_l_a, mut send_r_a) =
            ([0.0f32; 64], [0.0f32; 64], [0.0f32; 64], [0.0f32; 64]);
        let (mut main_l_b, mut main_r_b, mut send_l_b, mut send_r_b) =
            ([0.0f32; 64], [0.0f32; 64], [0.0f32; 64], [0.0f32; 64]);
        eng_4cm_a.process_block_4cm(
            &inst_l,
            &inst_r,
            &ret_zero,
            &ret_zero,
            &mut main_l_a,
            &mut main_r_a,
            &mut send_l_a,
            &mut send_r_a,
        );
        eng_4cm_b.process_block_4cm(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut main_l_b,
            &mut main_r_b,
            &mut send_l_b,
            &mut send_r_b,
        );
        // Send output must be 100% invariant to Return input (S_{ret->send} == 0)
        assert_eq!(send_l_a, send_l_b);
        assert_eq!(send_r_a, send_r_b);
        // Main output must reflect Return injection after alignment delay
        assert_ne!(main_l_a, main_l_b);

        // Verify WDW routing outputs
        let mut eng_wdw =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let (mut wdw_l, mut wdw_r, mut wdw_dry) = ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
        eng_wdw.process_block_wdw(&inst_l, &inst_r, &mut wdw_l, &mut wdw_r, &mut wdw_dry);
        assert!(wdw_dry.iter().any(|&v| v.abs() > 1e-5));

        let (mut sc_out_l, mut sc_out_r) = ([0.0f32; 16], [0.0f32; 16]);
        eng_wdw.process_block_sidechain(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut sc_out_l,
            &mut sc_out_r,
        );
        assert!(sc_out_l[0].is_finite());

        eng_wdw.set_offline_bounce(true);
        eng_wdw.process_block_4cm(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut main_l_a,
            &mut main_r_a,
            &mut send_l_a,
            &mut send_r_a,
        );
        eng_wdw.process_block_wdw(&inst_l, &inst_r, &mut wdw_l, &mut wdw_r, &mut wdw_dry);
        eng_wdw.process_block_sidechain(
            &inst_l,
            &inst_r,
            &ret_sig,
            &ret_sig,
            &mut sc_out_l,
            &mut sc_out_r,
        );
        assert!(sc_out_l[0].is_finite());

        eng_wdw.set_offline_bounce(false);
        eng_wdw.set_bypass(true, false);
        let [b_l, b_r, b_3, b_4] =
            eng_wdw.step_frame_4port_sona(0.25, -0.25, 0.0, 0.0, false, false);
        assert_eq!([b_l, b_r, b_3, b_4], [0.25, -0.25, 0.0, 0.0]);

        // Verify WDW stereo_width applies linear width scaling (no pre-widening double width)
        let mut wdw_w1 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut wdw_w2 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        wdw_w1.set_stereo_width(1.0);
        wdw_w2.set_stereo_width(2.0);
        let in_l_asym = [0.6_f32; 16];
        let in_r_asym = [-0.2_f32; 16];
        let (mut l1, mut r1, mut d1) = ([0.0_f32; 16], [0.0_f32; 16], [0.0_f32; 16]);
        let (mut l2, mut r2, mut d2) = ([0.0_f32; 16], [0.0_f32; 16], [0.0_f32; 16]);
        wdw_w1.process_block_wdw(&in_l_asym, &in_r_asym, &mut l1, &mut r1, &mut d1);
        wdw_w2.process_block_wdw(&in_l_asym, &in_r_asym, &mut l2, &mut r2, &mut d2);
        let max_side_1 = l1
            .iter()
            .zip(r1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_side_1 > 1e-4, "expected stereo difference in WDW wet");
        for i in 0..16 {
            let mid1 = 0.5 * (l1[i] + r1[i]);
            let mid2 = 0.5 * (l2[i] + r2[i]);
            let side1 = 0.5 * (l1[i] - r1[i]);
            let side2 = 0.5 * (l2[i] - r2[i]);
            assert!((mid2 - mid1).abs() < 1e-5, "WDW mid contaminated at {i}");
            assert!(
                (side2 - 2.0 * side1).abs() < 1e-5,
                "WDW side double-widened at {i}: got {side2}, expected {}",
                2.0 * side1
            );
            assert!((d2[i] - d1[i]).abs() < 1e-6);
        }

        // Verify 4x 4-port downsamples out_3/out_4 through oversampler_aux cleanly (DirectDry tap -> 0.5)
        let mut eng_4x =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng_4x.io_router.wdw_tap_mode = crate::graph::WdwDryTapMode::DirectDry;
        eng_4x.set_offline_bounce(true);
        let dc_in = [0.5_f32; 32];
        let (mut wdw_l_4x, mut wdw_r_4x, mut wdw_dry_4x) =
            ([0.0_f32; 32], [0.0_f32; 32], [0.0_f32; 32]);
        eng_4x.process_block_wdw(
            &dc_in,
            &dc_in,
            &mut wdw_l_4x,
            &mut wdw_r_4x,
            &mut wdw_dry_4x,
        );
        assert!(wdw_dry_4x.iter().all(|v| v.is_finite()));
        assert!(wdw_dry_4x[0].abs() < wdw_dry_4x[31].abs());
        assert!((wdw_dry_4x[31] - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_live_telemetry_peaks_op_n_drive_pole_radius_and_multi_quad_loop_gain() {
        let mut header = DnamJsonHeader::default_reference();
        let mut mat = SonaMatryoshkaModel::identity(false, false);
        mat.tier_os2x = [false, true, false, false];
        mat.groups[0].in_x_weight = [0.8, 0.0, 0.0, 0.0];
        mat.groups[0].tone_dir = [1.0; 4];
        mat.groups[0].tone_lp = [0.0; 4];
        mat.groups[0].adaa_drive = [1.5, 0.0, 0.0, 0.0];
        mat.groups[0].lru_alpha_rel = [0.85, 0.90, 0.70, 0.50];
        mat.kautz_poles[16] = (0.96, 0.35);
        for t in 0..4 {
            mat.stage4[t].g_loop = 0.25;
            mat.stage4[t].quads[0].g_loop = 0.25;
            mat.stage4[t].quads[1].g_loop = 0.60;
            mat.stage4[t].quads[2].g_loop = 0.88;
            mat.stage4[t].quads[3].g_loop = 0.94;
        }
        let mut arena = alloc::vec![0_u8; ARENA_OFFSET_ULTRA_END as usize];
        pack_sona_matryoshka_to_arena(&mat, &mut arena);
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Pro, 48_000).unwrap();
        let init_snap = engine.telemetry_snapshot();
        assert_eq!(init_snap.max_op_n_drive, 0.0);
        assert!((init_snap.max_pole_radius - 0.96).abs() < 1.0e-4);
        assert!((init_snap.max_effective_loop_gain - 0.88).abs() < 1.0e-6);

        let zeros = [0.0_f32; 64];
        let (mut out_l, mut out_r) = ([0.0_f32; 64], [0.0_f32; 64]);
        engine.process_block(&zeros, &zeros, &mut out_l, &mut out_r);
        assert_eq!(engine.telemetry_snapshot().max_op_n_drive, 0.0);

        let drive_in: [f32; 64] = core::array::from_fn(|i| 0.5 * ((i as f32) * 0.2).sin());
        engine.process_block(&drive_in, &drive_in, &mut out_l, &mut out_r);
        let driven_snap = engine.telemetry_snapshot();
        assert!(driven_snap.max_op_n_drive > 0.05 && driven_snap.max_op_n_drive <= 1.0);

        engine.set_bypass(true, false);
        engine.process_block(&drive_in, &drive_in, &mut out_l, &mut out_r);
        assert_eq!(engine.telemetry_snapshot().max_op_n_drive, 0.0);

        engine.set_tier(Tier::Ultra);
        engine.set_bypass(false, false);
        engine.process_block(&zeros, &zeros, &mut out_l, &mut out_r);
        assert!((engine.telemetry_snapshot().max_effective_loop_gain - 0.94).abs() < 1.0e-6);

        // Verify frame-0 impulse peaks inside short (<16) and multi-subblock (32 = 2x16) blocks match step_stereo across 1x & OS2X.
        let mut imp_short = [0.0_f32; 12];
        imp_short[0] = 0.85;
        let mut imp_multi = [0.0_f32; 32];
        imp_multi[0] = 0.85;
        let (mut out_short_l, mut out_short_r) = ([0.0_f32; 12], [0.0_f32; 12]);
        let (mut out_multi_l, mut out_multi_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        for (tier, expected_os2x) in [(Tier::Lite, false), (Tier::Full, true)] {
            let mut short_eng =
                DnamEngine::<ScalarBackend>::from_bytes(&container, tier, 48_000).unwrap();
            assert_eq!(short_eng.hot_core.os2x(), expected_os2x);
            short_eng.process_block(&imp_short, &imp_short, &mut out_short_l, &mut out_short_r);
            let short_peak = short_eng.telemetry_snapshot().max_op_n_drive;

            let mut multi_eng =
                DnamEngine::<ScalarBackend>::from_bytes(&container, tier, 48_000).unwrap();
            assert_eq!(multi_eng.hot_core.os2x(), expected_os2x);
            multi_eng.process_block(&imp_multi, &imp_multi, &mut out_multi_l, &mut out_multi_r);
            let multi_peak = multi_eng.telemetry_snapshot().max_op_n_drive;

            let mut step_eng =
                DnamEngine::<ScalarBackend>::from_bytes(&container, tier, 48_000).unwrap();
            assert_eq!(step_eng.hot_core.os2x(), expected_os2x);
            for &x in &imp_multi {
                let _ = step_eng.hot_core.step_stereo(x, x);
            }
            let step_peak = step_eng.telemetry_snapshot().max_op_n_drive;
            assert!(
                short_peak > 0.1 && multi_peak > 0.1,
                "tier {:?} (os2x={}) missed frame-0 impulse peak",
                tier,
                expected_os2x
            );
            assert!(
                (short_peak - step_peak).abs() < 1.0e-6 && (multi_peak - step_peak).abs() < 1.0e-6,
                "tier {:?} (os2x={}) subblock vs step_stereo drive peak mismatch: short={}, multi={}, step={}",
                tier,
                expected_os2x,
                short_peak,
                multi_peak,
                step_peak
            );
        }
    }
}
