//! 4-port (`4CM`, `WDW`, `Sidechain`) stereo processing loops over [`crate::engine::HostShellPatchbay`].

use dnam_kernels::scrub_and_flush_f32_bits;
use dnam_vm::math::{
    CERTIFIED_INPUT_ENVELOPE_MIN_F32, sanitize_input_sample, sanitize_output_sample,
};

use super::DnamAudioIo;
use crate::engine::{DnamEngine, EngineBackend};
use crate::types::FtzDazGuard;

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Low-level 4-port (`[inst_l, inst_r, ret_l, ret_r] -> [main_l, main_r, send_or_wdw_dry, send_r]`)
    /// `SONA` stepping preserving structural `S_{ret->send} == 0` (`ref[dsp.router.ports]`).
    #[inline(never)]
    pub(crate) fn step_frame_4port_sona(
        &mut self,
        in_l: f32,
        in_r: f32,
        ret_l: f32,
        ret_r: f32,
        mono_only: bool,
    ) -> [f32; 4] {
        self.sync_hot_core_tier();
        let in_l = scrub_and_flush_f32_bits(sanitize_input_sample(
            in_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        ));
        let in_r = if self.single_input {
            in_l
        } else {
            scrub_and_flush_f32_bits(sanitize_input_sample(
                in_r,
                CERTIFIED_INPUT_ENVELOPE_MIN_F32,
                &mut self.saturation_flags,
            ))
        };
        let ret_l = scrub_and_flush_f32_bits(sanitize_input_sample(
            ret_l,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        ));
        let ret_r = scrub_and_flush_f32_bits(sanitize_input_sample(
            ret_r,
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut self.saturation_flags,
        ));

        let is_wdw = self.patchbay.wdw_enabled
            || self.patchbay.routing_mode == crate::engine::IoRoutingMode::WetDryWet;

        if self.bypassed && !self.trails_enabled {
            if is_wdw {
                let (_, _, dry_tap) = self.patchbay.route_pre_vm([in_l, in_r, ret_l, ret_r]);
                return [
                    0.0,
                    0.0,
                    sanitize_output_sample(dry_tap, &mut self.saturation_flags),
                    0.0,
                ];
            }
            return if mono_only {
                let m = 0.5 * (in_l + in_r);
                [m, m, 0.0, 0.0]
            } else {
                [in_l, in_r, 0.0, 0.0]
            };
        }

        let (vm_in_raw, send_out, dry_tap) = self.patchbay.route_pre_vm([in_l, in_r, ret_l, ret_r]);
        let vm_in = if self.bypassed {
            [0.0_f32; 2]
        } else {
            let peak = vm_in_raw[0].abs().max(vm_in_raw[1].abs()).min(1.0);
            if peak > self.last_block_max_drive {
                self.last_block_max_drive = peak;
            }
            vm_in_raw
        };
        let (vm_l, vm_r) = self.vm.step_frame(vm_in[0], vm_in[1]).unwrap_or((0.0, 0.0));
        let (vm_core_lr, vm_wet_lr) = self.vm.last_wdw_partition();
        let [send_l, port1_send_r_or_dry, main_raw_l, main_raw_r] = self
            .patchbay
            .route_post_vm_wdw([vm_l, vm_r], vm_core_lr, vm_wet_lr, dry_tap, send_out);

        let vm_wet_pol_inv = self.vm.slab().wet_polarity_inv();
        let host_pol_sign = if !vm_wet_pol_inv && self.json_wet_polarity_inverted() {
            -1.0_f32
        } else {
            1.0_f32
        };
        let dry_g = if is_wdw {
            0.0_f32
        } else if self.bypassed {
            1.0_f32
        } else {
            self.mix_dry_gain
        };
        let wet_g = host_pol_sign * self.mix_wet_gain;

        let mut main_l = dry_g * in_l + wet_g * main_raw_l;
        let mut main_r = dry_g * in_r + wet_g * main_raw_r;
        main_l = sanitize_output_sample(main_l, &mut self.saturation_flags);
        main_r = sanitize_output_sample(main_r, &mut self.saturation_flags);
        if self.preview_mode {
            (main_l, main_r) = self.preview_guard.process_sample_lr(main_l, main_r);
        }
        if mono_only {
            let m = 0.5 * (main_l + main_r);
            main_l = m;
            main_r = m;
        }

        let (o3, o4) = if is_wdw {
            let dry_center =
                if self.patchbay.wdw_mode == crate::engine::WdwDryTapMode::PreSpatialCore {
                    let core_mid = host_pol_sign * port1_send_r_or_dry;
                    if self.bypassed {
                        dry_tap + core_mid
                    } else {
                        core_mid
                    }
                } else {
                    port1_send_r_or_dry
                };
            (
                sanitize_output_sample(dry_center, &mut self.saturation_flags),
                0.0,
            )
        } else if self.bypassed {
            (0.0, 0.0)
        } else {
            (
                sanitize_output_sample(send_l, &mut self.saturation_flags),
                sanitize_output_sample(port1_send_r_or_dry, &mut self.saturation_flags),
            )
        };
        [main_l, main_r, o3, o4]
    }

    /// Processes a multi-port audio block (`Stage 0` 8-port router contract: `in_1..4`, `out_1..4`).
    pub fn process_block_io(&mut self, io: DnamAudioIo<'_>) {
        let _ftz = FtzDazGuard::new();
        self.last_block_max_drive = 0.0;
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
            let [ol, or, o3, o4] = self.step_frame_4port_sona(il, ir, rl, rr, mono_only);
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
        let prev_wdw = self.patchbay.wdw_enabled;
        self.patchbay.wdw_enabled = false;
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
        self.patchbay.wdw_enabled = prev_wdw;
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
        let prev_wdw = self.patchbay.wdw_enabled;
        self.patchbay.wdw_enabled = true;
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
        self.patchbay.wdw_enabled = prev_wdw;
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
    use crate::container::build_dnam_container_from_dnam_bin;
    use crate::engine::{DnamEngine, WdwDryTapMode};
    use crate::schema::DnamJsonHeader;
    use crate::test_support::{build_spatial_test_dnam_container, build_test_dnam_container};
    use crate::types::Tier;
    use dnam_vm::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DNBN_FLAG_STEREO_OUT,
        DnamBinBuildInput, SONA_META_TIER_OS_EXPLICIT, encode_dnam_bin,
    };
    use dnam_vm::math::ScalarBackend;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16, U32};

    #[test]
    fn test_4cm_wdw_sidechain_and_bypass_routing() {
        let _ftz = crate::types::FtzDazGuard::new();
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Ultra, true);

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
        assert_eq!(send_l_a, send_l_b);
        assert_eq!(send_r_a, send_r_b);
        assert_ne!(main_l_a, main_l_b);

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

        eng_wdw.set_offline_bounce(false);
        eng_wdw.set_bypass(true, false);
        let [b_l, b_r, b_3, b_4] = eng_wdw.step_frame_4port_sona(0.25, -0.25, 0.0, 0.0, false);
        assert_eq!([b_l, b_r, b_3, b_4], [0.25, -0.25, 0.0, 0.0]);

        let mut wdw_w1 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut wdw_w2 =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        wdw_w1.set_wdw_mode(WdwDryTapMode::DirectDry);
        wdw_w2.set_wdw_mode(WdwDryTapMode::DirectDry);
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
            assert!((side2 - 2.0 * side1).abs() < 1e-5);
            assert!((d2[i] - d1[i]).abs() < 1e-6);
        }

        let mut eng_4x =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng_4x.set_wdw_mode(WdwDryTapMode::DirectDry);
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
        assert!((wdw_dry_4x[31] - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_live_telemetry_peaks_op_n_drive_pole_radius_and_multi_quad_loop_gain() {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
        a0.crossbar.sona_meta_flags = U32::new(
            (a0.crossbar.sona_meta_flags.get() & !0x0F) | SONA_META_TIER_OS_EXPLICIT | 0b0010,
        );
        a0.crossbar.cell_desc.cell_flags = a0.crossbar.sona_meta_flags;
        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
        a0.transport_lite.g_loop = F32::new(0.25);

        let mut a1 = Arena1FullDeltaWire::new_zeroed();
        a1.transport_full.g_loop = F32::new(0.60);

        let mut a2 = Arena2ProDeltaWire::new_zeroed();
        a2.conv_kautz_pro[0].r_low[0] = F32::new(0.96);
        a2.conv_kautz_pro[0].theta[0] = F32::new(0.35);
        a2.transport_pro[0].g_loop = F32::new(0.88);

        let mut a3 = Arena3UltraDeltaWire::new_zeroed();
        a3.transport_ultra[0].g_loop = F32::new(0.94);

        let dbin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: 0.995,
            tail_time_ms: 150.0,
            ro_pole_48k: 0.0,
            switch_xfade_samples: 128,
            content_hash_prefix: [0; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: Some(&a1),
            arena2: Some(&a2),
            arena3: Some(&a3),
        })
        .unwrap();

        let mut header = DnamJsonHeader::default_reference();
        let container = build_dnam_container_from_dnam_bin(&mut header, &dbin).unwrap();

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
    }

    /// verify[dsp.router.wdw-contract]
    #[test]
    fn test_wdw_pre_spatial_core_vs_direct_dry_and_trails_continuity() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_spatial_test_dnam_container(&mut header, Tier::Lite);

        let in_l = [0.6_f32; 24];
        let in_r = [-0.2_f32; 24];
        let mut core_l = [0.0_f32; 24];
        let mut core_r = [0.0_f32; 24];
        let mut core_dry = [0.0_f32; 24];
        let mut dir_l = [0.0_f32; 24];
        let mut dir_r = [0.0_f32; 24];
        let mut dir_dry = [0.0_f32; 24];

        let mut eng_core =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_dir =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut eng_step =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        let alpha = 1.6_f32;
        eng_core.set_wdw_mode(WdwDryTapMode::PreSpatialCore);
        eng_core.set_stereo_width(alpha);
        eng_dir.set_wdw_mode(WdwDryTapMode::DirectDry);
        eng_dir.set_stereo_width(alpha);
        eng_step.set_wdw_mode(WdwDryTapMode::PreSpatialCore);
        eng_step.set_wdw_enabled(true);
        eng_step.set_stereo_width(alpha);

        eng_core.process_block_wdw(&in_l, &in_r, &mut core_l, &mut core_r, &mut core_dry);
        eng_dir.process_block_wdw(&in_l, &in_r, &mut dir_l, &mut dir_r, &mut dir_dry);

        for i in 0..24 {
            let [wl, wr, dc, _] = eng_step.step_frame_4port_sona(in_l[i], in_r[i], 0.0, 0.0, false);
            let (vm_core, vm_wet) = eng_step.vm.last_wdw_partition();
            let m_c = 0.5 * (vm_core[0] + vm_core[1]);
            let m_w = 0.5 * (vm_wet[0] + vm_wet[1]);
            let s_w = 0.5 * (vm_wet[0] - vm_wet[1]);
            assert!((core_dry[i] - m_c).abs() < 1.0e-6);
            assert!((core_l[i] - (m_w + alpha * s_w)).abs() < 1.0e-6);
            assert!((core_r[i] - (m_w - alpha * s_w)).abs() < 1.0e-6);
            assert!((dc - core_dry[i]).abs() < 1.0e-6);
            assert!((wl - core_l[i]).abs() < 1.0e-6);
            assert!((wr - core_r[i]).abs() < 1.0e-6);
            assert!((dir_dry[i] - 0.5 * (in_l[i] + in_r[i])).abs() < 1.0e-6);
        }
        assert!(
            core_l
                .iter()
                .zip(core_r.iter())
                .any(|(l, r)| (l - r).abs() > 1.0e-4)
        );
        assert!(
            core_dry
                .iter()
                .zip(dir_dry.iter())
                .any(|(a, b)| (a - b).abs() > 1.0e-4)
        );

        // Trails bypass in 4-port preserves wet_polarity_inverted and mix_wet_gain trim.
        let mut pol_hdr = DnamJsonHeader::default_reference();
        pol_hdr.io_config.capabilities.wet_polarity_inverted = true;
        pol_hdr.io_config.leveling.auto_unity_enabled = true;
        pol_hdr.io_config.leveling.calibrated_trim_db = -6.0206;
        let pol_c = build_spatial_test_dnam_container(&mut pol_hdr, Tier::Lite);
        let mut ref_hdr = DnamJsonHeader::default_reference();
        let ref_c = build_spatial_test_dnam_container(&mut ref_hdr, Tier::Lite);

        let mut eng_pol =
            DnamEngine::<ScalarBackend>::from_bytes(&pol_c, Tier::Lite, 48_000).unwrap();
        let mut eng_ref =
            DnamEngine::<ScalarBackend>::from_bytes(&ref_c, Tier::Lite, 48_000).unwrap();
        for _ in 0..32 {
            let _ = eng_pol.step_frame_4port_sona(0.5, 0.5, 0.0, 0.0, false);
            let _ = eng_ref.step_frame_4port_sona(0.5, 0.5, 0.0, 0.0, false);
        }
        eng_pol.set_bypass(true, true);
        eng_ref.set_bypass(true, true);
        let [trail_pol_l, _, _, _] = eng_pol.step_frame_4port_sona(0.0, 0.0, 0.0, 0.0, false);
        let [trail_ref_l, _, _, _] = eng_ref.step_frame_4port_sona(0.0, 0.0, 0.0, 0.0, false);
        assert!(trail_ref_l.abs() > 1.0e-5);
        assert!((trail_pol_l + 0.5 * trail_ref_l).abs() < 1.0e-3);
    }

    #[test]
    fn test_wdw_4x_bounce_dirac_pdc_alignment() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Lite, true);

        let mut eng_unaligned =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng_unaligned.set_wdw_mode(WdwDryTapMode::DirectDry);
        eng_unaligned.set_offline_bounce(true);
        assert_eq!(eng_unaligned.latency_samples(), 12);

        let mut imp_l = [0.0_f32; 64];
        let mut imp_r = [0.0_f32; 64];
        imp_l[0] = 1.0;
        imp_r[0] = 1.0;
        let (mut wl, mut wr, mut dry) = ([0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64]);
        eng_unaligned.process_block_wdw(&imp_l, &imp_r, &mut wl, &mut wr, &mut dry);
        assert_eq!(dry[12], 1.0);
        assert!(dry.iter().enumerate().all(|(i, &v)| (i == 12) || v == 0.0));

        let mut eng_aligned =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        eng_aligned.set_wdw_mode(WdwDryTapMode::DirectDry);
        eng_aligned.set_offline_bounce(true);
        eng_aligned.configure_parallel_alignment(
            crate::engine::ParallelAlignmentMode::FullBand64Tap,
            true,
            0.0,
            0.0,
        );
        assert_eq!(eng_aligned.latency_samples(), 43);
        let (mut wl2, mut wr2, mut dry2) = ([0.0_f32; 64], [0.0_f32; 64], [0.0_f32; 64]);
        eng_aligned.process_block_wdw(&imp_l, &imp_r, &mut wl2, &mut wr2, &mut dry2);
        assert_eq!(dry2[43], 1.0);
        assert!(dry2.iter().enumerate().all(|(i, &v)| (i == 43) || v == 0.0));
    }

    #[test]
    fn test_wdw_isolation_under_hard_bypass_trails_and_host_parallel() {
        use crate::schema::BlendMode;

        let mut hdr_full = DnamJsonHeader::default_reference();
        hdr_full.io_config.capabilities.blend_mode = BlendMode::HostParallel;
        hdr_full.io_config.capabilities.default_blend = 1.0;
        let c_full = build_spatial_test_dnam_container(&mut hdr_full, Tier::Lite);

        let mut hdr_half = DnamJsonHeader::default_reference();
        hdr_half.io_config.capabilities.blend_mode = BlendMode::HostParallel;
        hdr_half.io_config.capabilities.default_blend = 0.4;
        let c_half = build_spatial_test_dnam_container(&mut hdr_half, Tier::Lite);

        let in_l = [0.7_f32; 24];
        let in_r = [-0.3_f32; 24];
        let zeros = [0.0_f32; 24];

        for mode in [WdwDryTapMode::PreSpatialCore, WdwDryTapMode::DirectDry] {
            let mut e_full =
                DnamEngine::<ScalarBackend>::from_bytes(&c_full, Tier::Lite, 48_000).unwrap();
            let mut e_half =
                DnamEngine::<ScalarBackend>::from_bytes(&c_half, Tier::Lite, 48_000).unwrap();
            e_full.set_wdw_mode(mode);
            e_half.set_wdw_mode(mode);
            let (mut l1, mut r1, mut d1) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            let (mut l2, mut r2, mut d2) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            e_full.process_block_wdw(&in_l, &in_r, &mut l1, &mut r1, &mut d1);
            e_half.process_block_wdw(&in_l, &in_r, &mut l2, &mut r2, &mut d2);
            assert!(l1.iter().any(|&v| v.abs() > 1.0e-4));
            for i in 0..24 {
                assert!((l2[i] - 0.4 * l1[i]).abs() < 1.0e-5);
                assert!((r2[i] - 0.4 * r1[i]).abs() < 1.0e-5);
                if mode == WdwDryTapMode::DirectDry {
                    assert!((d2[i] - 0.2).abs() < 1.0e-6);
                }
            }

            // Trails bypass: non-zero input must not bleed into wet L/R, while DirectDry stays live.
            let mut e_zero_trails =
                DnamEngine::<ScalarBackend>::from_bytes(&c_half, Tier::Lite, 48_000).unwrap();
            e_zero_trails.set_wdw_mode(mode);
            e_zero_trails.process_block_wdw(&in_l, &in_r, &mut l1, &mut r1, &mut d1);

            e_half.set_bypass(true, true);
            e_zero_trails.set_bypass(true, true);
            let (mut tl_in, mut tr_in, mut td_in) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            let (mut tl_z, mut tr_z, mut td_z) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            e_half.process_block_wdw(&in_l, &in_r, &mut tl_in, &mut tr_in, &mut td_in);
            e_zero_trails.process_block_wdw(&zeros, &zeros, &mut tl_z, &mut tr_z, &mut td_z);
            assert!(tl_z.iter().any(|&v| v.abs() > 1.0e-5));
            for i in 0..24 {
                assert_eq!(tl_in[i].to_bits(), tl_z[i].to_bits());
                assert_eq!(tr_in[i].to_bits(), tr_z[i].to_bits());
                if mode == WdwDryTapMode::DirectDry {
                    assert!((td_in[i] - 0.2).abs() < 1.0e-6);
                } else {
                    assert!((td_in[i] - (0.2 + td_z[i])).abs() < 1.0e-5);
                }
            }

            // Hard bypass: wet L/R are strictly 0.0 and dry_center carries aligned dry signal
            // (draining the 12-sample prior block tail before the frame-12 impulse peak).
            e_half.set_bypass(true, false);
            e_half.set_offline_bounce(true);
            let mut imp = [0.0_f32; 24];
            imp[0] = 1.0;
            let (mut hl, mut hr, mut hd) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            e_half.process_block_wdw(&imp, &imp, &mut hl, &mut hr, &mut hd);
            assert!(hl.iter().all(|&v| v == 0.0));
            assert!(hr.iter().all(|&v| v == 0.0));
            assert!(hd[..12].iter().all(|&v| (v - 0.2).abs() < 1.0e-6));
            assert_eq!(hd[12], 1.0);
            assert!(hd[13..].iter().all(|&v| v == 0.0));

            e_half.reset_to_dc_equilibrium();
            e_half.process_block_wdw(&imp, &imp, &mut hl, &mut hr, &mut hd);
            assert_eq!(hd[12], 1.0);
            assert!(hd.iter().enumerate().all(|(i, &v)| (i == 12) || v == 0.0));
        }

        // DirectDry delay ring advances during trails bypass at 4x bounce.
        let mut e_trails_bounce =
            DnamEngine::<ScalarBackend>::from_bytes(&c_half, Tier::Lite, 48_000).unwrap();
        e_trails_bounce.set_wdw_mode(WdwDryTapMode::DirectDry);
        e_trails_bounce.set_offline_bounce(true);
        e_trails_bounce.set_bypass(true, true);
        let mut imp = [0.0_f32; 24];
        imp[0] = 1.0;
        let (mut wl, mut wr, mut dry) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
        e_trails_bounce.process_block_wdw(&imp, &imp, &mut wl, &mut wr, &mut dry);
        assert_eq!(dry[12], 1.0);
    }

    #[test]
    fn test_wdw_all_four_polarity_truth_table_single_source() {
        use crate::test_support::build_spatial_test_dnam_container_with_dnbn_polarity;

        let build_case = |dnbn_inv: bool, json_inv: bool| {
            let mut hdr = DnamJsonHeader::default_reference();
            hdr.io_config.capabilities.wet_polarity_inverted = json_inv;
            let bytes = build_spatial_test_dnam_container_with_dnbn_polarity(
                &mut hdr,
                Tier::Lite,
                dnbn_inv,
            );
            DnamEngine::<ScalarBackend>::from_bytes_owned(&bytes, Tier::Lite, 48_000).unwrap()
        };

        let in_l = [0.55_f32; 24];
        let in_r = [-0.15_f32; 24];

        for mode in [WdwDryTapMode::PreSpatialCore, WdwDryTapMode::DirectDry] {
            let mut e_ff = build_case(false, false);
            let mut e_tf = build_case(true, false);
            let mut e_ft = build_case(false, true);
            let mut e_tt = build_case(true, true);
            for eng in [&mut e_ff, &mut e_tf, &mut e_ft, &mut e_tt] {
                eng.set_wdw_mode(mode);
            }

            let (mut l_ff, mut r_ff, mut d_ff) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            let (mut l_tf, mut r_tf, mut d_tf) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            let (mut l_ft, mut r_ft, mut d_ft) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            let (mut l_tt, mut r_tt, mut d_tt) = ([0.0_f32; 24], [0.0_f32; 24], [0.0_f32; 24]);
            e_ff.process_block_wdw(&in_l, &in_r, &mut l_ff, &mut r_ff, &mut d_ff);
            e_tf.process_block_wdw(&in_l, &in_r, &mut l_tf, &mut r_tf, &mut d_tf);
            e_ft.process_block_wdw(&in_l, &in_r, &mut l_ft, &mut r_ft, &mut d_ft);
            e_tt.process_block_wdw(&in_l, &in_r, &mut l_tt, &mut r_tt, &mut d_tt);

            assert!(l_ff.iter().any(|&v| v.abs() > 1.0e-4));
            assert!(d_ff.iter().any(|&v| v.abs() > 1.0e-4));
            for i in 0..24 {
                for (li, ri, di) in [
                    (&l_tf, &r_tf, &d_tf),
                    (&l_ft, &r_ft, &d_ft),
                    (&l_tt, &r_tt, &d_tt),
                ] {
                    assert!((li[i] + l_ff[i]).abs() < 1.0e-5);
                    assert!((ri[i] + r_ff[i]).abs() < 1.0e-5);
                    if mode == WdwDryTapMode::PreSpatialCore {
                        assert!((di[i] + d_ff[i]).abs() < 1.0e-5);
                    } else {
                        assert!((di[i] - d_ff[i]).abs() < 1.0e-6);
                    }
                }
            }
        }
    }
}
