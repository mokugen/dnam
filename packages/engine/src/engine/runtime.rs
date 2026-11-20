//! State lifecycle, model reloading, and offline bounce rendering for [`DnamEngine`].

use super::{DnamEngine, EngineBackend};
use crate::container::{DnamContainerView, DnamPreamble, ZohRescaledParams};
use crate::schema::DnamError;
use crate::types::Tier;

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    pub(crate) fn init_from_view(
        &mut self,
        view: DnamContainerView<'a>,
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        if (view.preamble.version_major, view.preamble.version_minor)
            != (
                crate::container::CONTAINER_VERSION_MAJOR,
                crate::container::CONTAINER_VERSION_MINOR,
            )
            || view.header.architecture != "sona"
        {
            return Err(DnamError::UnsupportedVersion {
                major: view.preamble.version_major,
                minor: view.preamble.version_minor,
            });
        }
        let arena_len = view.arena_storage.as_bytes().len();
        let max_available_tier = Self::detect_max_tier(&view.preamble, arena_len);
        let active_tier = tier.min(max_available_tier).min(max_supported_tier);

        let zoh = ZohRescaledParams::compute(
            view.preamble.sample_rate_train_hz,
            sample_rate_hz,
            0.9993,
            0.995,
            64.0,
            128,
        )?;

        let freeze_mutes_dry = view.header.io_config.capabilities.freeze_mutes_dry;
        let single_input = view.header.io_config.inputs == 1;
        self.preamble = view.preamble;
        self.header = Some(view.header);
        self.header_active = true;
        self.reloaded_header_flags = None;
        self.verification_status = view.verification_status;
        self.verification_status_active = true;
        self.protection_profile = view.protection_profile;
        self.arena_storage = view.arena_storage;
        self.control_bus.num_contributors = 0;
        self.control_bus.load_control_records(&view.control_records);
        let active_hdr = if self.header_active {
            self.header.as_ref()
        } else {
            None
        };
        if !view.clamp_descriptors.is_empty() {
            let n_desc = view.clamp_descriptors.len();
            self.control_bus.ensure_control_capacity(n_desc);
            for (i, d) in view.clamp_descriptors.iter().enumerate() {
                self.control_bus.descriptors[i] = *d;
            }
            if self.control_bus.num_controls < n_desc {
                self.control_bus.num_controls = n_desc;
            }
        } else if view.control_records.is_empty()
            && let Some(hdr) = active_hdr
            && !hdr.controls_schema.is_empty()
            && let Ok(compiled) = hdr.compile_clamp_descriptors()
        {
            let n_desc = compiled.len();
            self.control_bus.ensure_control_capacity(n_desc);
            for (i, d) in compiled.into_iter().enumerate() {
                self.control_bus.descriptors[i] = d;
            }
            if self.control_bus.num_controls < n_desc {
                self.control_bus.num_controls = n_desc;
            }
        }
        if !view.cp1_contributors.is_empty() {
            self.control_bus
                .load_cp1_contributors(&view.cp1_contributors);
        }
        if let Some(hdr) = active_hdr {
            self.control_bus
                .ensure_control_capacity(hdr.controls_schema.len());
        }
        self.tier = active_tier;
        self.max_available_tier = max_available_tier;
        self.max_supported_tier = max_supported_tier;
        self.offline_bounce = false;
        self.control_bus.offline_bounce = false;
        self.maximize_tier_on_bounce = true;
        self.control_bus.ultra_tier_active = self.effective_tier() == Tier::Ultra;
        self.oversampler.reset();
        self.oversampler_aux.reset();
        self.tempo_state.sample_rate_hz = sample_rate_hz;
        self.zoh = zoh;
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        self.bypassed = false;
        self.trails_enabled = false;
        self.freeze_mutes_dry = freeze_mutes_dry;
        self.single_input = single_input;
        self.stereo_width = 1.0;

        self.preallocate_cold_tank_for(&view.preamble, active_tier);
        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
        Ok(())
    }

    /// Pre-allocates Stage IV extended delay ring coldly outside RT / `reconfigure_*` call graphs.
    pub(crate) fn preallocate_cold_tank_for(&mut self, preamble: &DnamPreamble, target_tier: Tier) {
        let arena = self.arena_storage.as_bytes();
        let max_tier_idx = (Self::detect_max_tier(preamble, arena.len()) as usize).min(3);
        let declared_48k = self.header().map_or(0usize, |hdr| {
            hdr.delay_histories
                .iter()
                .map(|d| d.max_samples_48k() as usize)
                .max()
                .unwrap_or(0)
        });
        let mono_out = !preamble.is_stereo_out();
        let mat = crate::graph::weights::unpack_sona_matryoshka_from_arena(arena, mono_out);
        *self.cold_tiers = mat.to_four_tiers();
        let mut tier_max_req = 0usize;
        let mut needs_ext_or_diffuser = false;
        for m in &self.cold_tiers[..=max_tier_idx] {
            let aq = m.stage4.active_quads.clamp(1, 4);
            if aq > 1 {
                needs_ext_or_diffuser = true;
            }
            for q in 0..aq {
                let qp = m.stage4.effective_quad_params(q);
                if qp.ext_format == crate::graph::tank::fdn::TankExtFormat::F32
                    || qp.allpass_delays.iter().any(|&d| d > 0)
                {
                    needs_ext_or_diffuser = true;
                }
                let q_req = qp.delay_lengths.iter().copied().max().unwrap_or(0)
                    + (qp.lfo_depth_samples as usize)
                    + 6;
                tier_max_req = tier_max_req.max(q_req);
            }
        }
        let max_ref = declared_48k.max(tier_max_req);
        let bounce_min = max_ref.saturating_mul(16).saturating_add(32).max(2_048);
        if max_ref > crate::graph::tank::fdn::V3_TANK_CAP
            || bounce_min > crate::graph::tank::fdn::V3_TANK_CAP
            || needs_ext_or_diffuser
        {
            self.hot_core
                .ensure_cold_tank_cap(&self.cold_tiers[max_tier_idx].stage4, bounce_min);
            self.bounce_scratch
                .hot_core
                .ensure_cold_tank_cap(&self.cold_tiers[max_tier_idx].stage4, bounce_min);
            let eff = (target_tier as usize).min(max_tier_idx);
            self.hot_core
                .ensure_cold_tank_cap(&self.cold_tiers[eff].stage4, bounce_min);
            self.bounce_scratch
                .hot_core
                .ensure_cold_tank_cap(&self.cold_tiers[eff].stage4, bounce_min);
        }
    }

    pub(crate) fn detect_max_tier(preamble: &DnamPreamble, arena_len: usize) -> Tier {
        if (preamble.flags & crate::container::FLAG_HAS_ULTRA_DELTA) != 0
            && preamble.arena_ultra_end_cl > 0
            && arena_len >= (preamble.arena_offset_ultra_end() as usize)
        {
            Tier::Ultra
        } else if (preamble.flags & crate::container::FLAG_HAS_PRO_DELTA) != 0
            && preamble.arena_pro_end_cl > 0
            && arena_len >= (preamble.arena_offset_pro_end() as usize)
        {
            Tier::Pro
        } else if (preamble.flags & crate::container::FLAG_HAS_FULL_DELTA) != 0
            && preamble.arena_full_end_cl > 0
            && arena_len >= (preamble.arena_offset_full_end() as usize)
        {
            Tier::Full
        } else {
            Tier::Lite
        }
    }

    pub(crate) fn reconfigure_for_sample_rate(
        &mut self,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, false)
    }

    pub(crate) fn load_v3_hot_core_from_arena(&mut self, reset_state: bool) {
        let arena = self.arena_storage.as_bytes();
        let mono_out = !self.preamble.is_stereo_out();
        let mat = crate::graph::weights::unpack_sona_matryoshka_from_arena(arena, mono_out);
        if arena.len() >= 64 {
            let mut has_s = false;
            let mut s_mat = [[0.0_f32; 4]; 4];
            for (r, row) in s_mat.iter_mut().enumerate() {
                for (c, cell) in row.iter_mut().enumerate() {
                    let v = crate::graph::weights::read_f32_le(arena, (r * 4 + c) * 4);
                    if v.is_finite() {
                        *cell = v.clamp(-4.0, 4.0);
                        if v.abs() > 1.0e-12 {
                            has_s = true;
                        }
                    }
                }
            }
            if has_s {
                self.io_router.set_scattering_matrix(s_mat);
            } else if mat
                .stage0
                .s_router
                .iter()
                .flatten()
                .any(|v| v.is_finite() && v.abs() > 1.0e-12)
            {
                self.io_router.set_scattering_matrix(mat.stage0.s_router);
            } else {
                self.io_router.s_matrix = crate::graph::IoRouter::DEFAULT_S_MATRIX;
            }
        }
        if let Some(hdr) = self.header() {
            self.io_router
                .set_crossover_hz(hdr.io_router.sub_crossover_hz);
        } else if let Some(flags) = self.reloaded_header_flags {
            self.io_router.set_crossover_hz(flags.sub_crossover_hz);
        }
        self.io_router.reconfigure_sample_rate(&self.zoh);

        for t in 0..4 {
            let trim_off = crate::graph::weights::ARENA0_TIER_GAIN_TRIM_OFFSET + t * 4;
            let raw_trim = crate::graph::weights::read_f32_le(arena, trim_off);
            self.cached_tier_gain_trim[t] = if (0.0625..=16.0).contains(&raw_trim) {
                raw_trim
            } else {
                1.0
            };
        }

        *self.cold_tiers = mat.to_four_tiers();
        let eff = self.effective_tier() as usize;
        let was_frozen = self.hot_core.tank_freeze();
        if reset_state {
            self.hot_core.reload_in_place(&self.cold_tiers[eff]);
        } else {
            self.hot_core.switch_tier(&self.cold_tiers[eff]);
            self.hot_core.set_tank_freeze(was_frozen);
        }
        let target_os2x = self.cold_tiers[eff].os2x && !self.offline_bounce;
        self.hot_core.reconfigure_with_zoh_mode(
            &self.cold_tiers[eff],
            &self.zoh,
            if target_os2x { 2 } else { 1 },
            self.offline_bounce,
        );
        self.hot_core.set_os2x(target_os2x);
        self.hot_core.set_s0_s_router(self.io_router.s_matrix);
    }

    pub(crate) fn training_zoh_poles(&self) -> (f32, f32, f32, u32) {
        let arena = self.arena_storage.as_bytes();
        let v_ac =
            crate::graph::weights::read_f32_le(arena, crate::graph::weights::ARENA0_R_AC_OFFSET);
        let r_ac_tr = if v_ac > 0.9 && v_ac < 1.0 {
            v_ac
        } else {
            0.9993
        };
        let (hdr_dc, hdr_tau, hdr_xfade) = if let Some(h) = self.header() {
            (
                h.spatial_tank.dc_blocker_pole,
                h.spatial_tank.tzf_reference_tap_samples,
                h.spatial_tank.switch_xfade_samples,
            )
        } else if let Some(flags) = self.reloaded_header_flags {
            (
                flags.dc_blocker_pole,
                flags.tzf_reference_tap_samples,
                flags.switch_xfade_samples,
            )
        } else {
            (0.995, 64.0, 128)
        };
        let r_dc_tr = if hdr_dc > 0.8 && hdr_dc < 1.0 {
            hdr_dc
        } else {
            0.995
        };
        let zero_op_t = crate::graph::weights::has_pipeline_flag(
            self.arena_storage.as_bytes(),
            crate::graph::weights::PIPE_FLAG_ZERO_OP_T_PRE,
        );
        let tau_ref_tr = if zero_op_t {
            0.0
        } else if hdr_tau > 0.0 {
            hdr_tau
        } else {
            64.0
        };
        let switch_xfade_tr = if hdr_xfade > 0 { hdr_xfade as u32 } else { 128 };
        (r_ac_tr, r_dc_tr, tau_ref_tr, switch_xfade_tr)
    }

    /// Zero-drop-glue (`Copy` error) sample-rate reconfiguration for real-time model reloaders.
    pub(crate) fn reconfigure_for_sample_rate_rt(
        &mut self,
        sample_rate_hz: u32,
        reset_controls: bool,
    ) -> Result<(), crate::container::ContainerWireError> {
        let (r_ac_tr, r_dc_tr, tau_ref_tr, switch_xfade_tr) = self.training_zoh_poles();

        let oversample_factor = if self.offline_bounce { 4 } else { 1 };
        let Some(new_zoh) = ZohRescaledParams::compute_oversampled_opt(
            self.preamble.sample_rate_train_hz,
            sample_rate_hz,
            oversample_factor,
            r_ac_tr,
            r_dc_tr,
            tau_ref_tr,
            switch_xfade_tr,
        ) else {
            return Err(crate::container::ContainerWireError::InvalidSampleRate(
                sample_rate_hz,
            ));
        };
        self.zoh = new_zoh;

        self.control_bus.ultra_tier_active = self.effective_tier() == Tier::Ultra;
        let eff_len = self
            .effective_tier()
            .expected_arena_bytes()
            .min(self.arena_storage.as_bytes().len());
        if reset_controls {
            if let Some(hdr) = self.header() {
                let default_blend = hdr.io_config.capabilities.default_blend;
                self.configure_host_mix_and_leveling(default_blend, 0.0, true);
            } else if let Some(flags) = self.reloaded_header_flags {
                self.configure_host_mix_and_leveling(flags.default_blend, 0.0, true);
            } else {
                self.mix_dry_gain = 0.0;
                self.mix_wet_gain = 1.0;
            }
            {
                let arena = self.arena_storage.as_bytes();
                self.control_bus.load_from_arena0(
                    arena,
                    self.preamble.num_controls as usize,
                    self.zoh.rho_ctrl,
                );
            }
            if let Some(hdr) = self.header() {
                let mh = hdr.control_bus.head_trust_region_radius;
                if mh > 1e-3 {
                    self.control_bus.trust_radius_mh = mh;
                    self.control_bus.inv_trust_radius_mh2 =
                        crate::math::newton6_reciprocal_f64((mh as f64) * (mh as f64)) as f32;
                }
            }
            let eff_arena = &self.arena_storage.as_bytes()[..eff_len];
            self.control_bus.refresh_static_state(eff_arena);
        } else {
            {
                let eff_arena = &self.arena_storage.as_bytes()[..eff_len];
                self.control_bus
                    .update_sample_rate_params(eff_arena, self.zoh.rho_ctrl);
            }
            if let Some(hdr) = self.header() {
                let mh = hdr.control_bus.head_trust_region_radius;
                if mh > 1e-3 {
                    self.control_bus.trust_radius_mh = mh;
                    self.control_bus.inv_trust_radius_mh2 =
                        crate::math::newton6_reciprocal_f64((mh as f64) * (mh as f64)) as f32;
                }
            }
            self.control_bus.reset_dynamic_state();
            let eff_arena = &self.arena_storage.as_bytes()[..eff_len];
            self.control_bus.refresh_static_state(eff_arena);
        }

        self.load_v3_hot_core_from_arena(reset_controls);
        self.current_stage_params = self.control_bus.stage_targets;
        self.stage_ramp_deltas = [0.0; 16];
        self.hot_core
            .snap_control_bus_with_z_stage(&self.current_stage_params, &self.control_bus.z_stage);

        let num_ctrls = self.control_bus.num_controls;
        for ctrl_idx in 0..num_ctrls {
            let raw_val = self.control_bus.raw_values[ctrl_idx];
            self.refresh_calibrated_control_effects(ctrl_idx, raw_val);
        }

        self.oversampler.reset();
        self.oversampler_aux.reset();
        self.tempo_state.sample_rate_hz = sample_rate_hz;
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        Ok(())
    }

    pub(crate) fn reconfigure_for_sample_rate_with_mode(
        &mut self,
        sample_rate_hz: u32,
        reset_controls: bool,
    ) -> Result<(), DnamError> {
        self.reconfigure_for_sample_rate_rt(sample_rate_hz, reset_controls)
            .map_err(loader::wire_err_to_dnam_rt)
    }

    /// Resets all states to analytical DC resting equilibrium.
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.control_bus.ultra_tier_active = self.effective_tier() == Tier::Ultra;
        self.control_bus.reset_dynamic_state();
        let eff_len = self
            .effective_tier()
            .expected_arena_bytes()
            .min(self.arena_storage.as_bytes().len());
        let arena = &self.arena_storage.as_bytes()[..eff_len];
        self.control_bus.refresh_static_state(arena);
        self.current_stage_params = self.control_bus.stage_targets;
        self.stage_ramp_deltas = [0.0; 16];
        self.hot_core.reset_to_dc_equilibrium();
        self.io_router.reset();
        self.hot_core
            .snap_control_bus_with_z_stage(&self.current_stage_params, &self.control_bus.z_stage);
        let num_ctrls = self.control_bus.num_controls;
        for ctrl_idx in 0..num_ctrls {
            let raw_val = self.control_bus.raw_values[ctrl_idx];
            self.refresh_calibrated_control_effects(ctrl_idx, raw_val);
        }
        self.oversampler.reset();
        self.oversampler_aux.reset();
        self.tempo_state.reset();
        self.current_stage_params = self.control_bus.stage_targets;
        self.stage_ramp_deltas = [0.0; 16];
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
    }
}

pub mod bounce;
pub mod loader;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        ARENA_OFFSET_ULTRA_END, DnamPreamble, FLAG_HAS_FULL_DELTA, FLAG_HAS_PRO_DELTA,
        FLAG_HAS_ULTRA_DELTA, build_dnam_container,
    };
    use crate::math::ScalarBackend;
    use crate::schema::DnamJsonHeader;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn test_detect_max_tier() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut preamble = DnamPreamble::from_bytes(&container).unwrap();
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 4_608),
            Tier::Lite
        );

        preamble.flags |= FLAG_HAS_FULL_DELTA;
        preamble.arena_full_end_cl = 124;
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 7_936),
            Tier::Full
        );

        preamble.flags |= FLAG_HAS_PRO_DELTA;
        preamble.arena_pro_end_cl = 248;
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 15_872),
            Tier::Pro
        );

        preamble.flags |= FLAG_HAS_ULTRA_DELTA;
        preamble.arena_ultra_end_cl = 480;
        assert_eq!(
            DnamEngine::<ScalarBackend>::detect_max_tier(&preamble, 30_720),
            Tier::Ultra
        );
    }

    #[test]
    fn test_runtime_render_offline_bounce_and_reloads() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine = DnamEngine::<ScalarBackend>::from_bytes_with_max_tier(
            &container,
            Tier::Ultra,
            Tier::Ultra,
            48000,
        )
        .unwrap();

        // 1. Reset to DC equilibrium
        engine.reset_to_dc_equilibrium();

        // 2. Offline bounce with small buffer (n <= 12 PDC)
        let short_in = [0.1f32; 8];
        let mut short_l = [0.0f32; 8];
        let mut short_r = [0.0f32; 8];
        engine.render_offline_bounce(&short_in, &short_in, &mut short_l, &mut short_r);

        // 3. Offline bounce with longer buffer (n > 12 PDC)
        let long_in = [0.1f32; 32];
        let mut long_l = [0.0f32; 32];
        let mut long_r = [0.0f32; 32];
        engine.render_offline_bounce(&long_in, &long_in, &mut long_l, &mut long_r);

        // 4. Reload in place owned success and error
        let mut owned_engine =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&container, Tier::Lite, 48000).unwrap();
        assert!(
            owned_engine
                .reload_model_in_place_owned(&container, Tier::Lite, 48000)
                .is_ok()
        );

        // Error: corrupt preamble
        assert!(
            owned_engine
                .reload_model_in_place_owned(&[0u8; 32], Tier::Lite, 48000)
                .is_err()
        );

        // Error: buffer too small
        assert!(
            owned_engine
                .reload_model_in_place_owned(&container[..container.len() - 100], Tier::Lite, 48000)
                .is_err()
        );

        // Error: header crc mismatch (corrupt a byte in Region 3 JCS header)
        let mut bad_crc = container.clone();
        bad_crc[192] ^= 0xAA;
        assert!(matches!(
            owned_engine.reload_model_in_place_owned(&bad_crc, Tier::Lite, 48000),
            Err(DnamError::Crc32cMismatch {
                region: "header",
                ..
            })
        ));
        assert!(matches!(
            engine.reload_model_in_place(&bad_crc, Tier::Lite, 48000),
            Err(DnamError::Crc32cMismatch {
                region: "header",
                ..
            })
        ));

        // Error: arena crc mismatch in reload_model_in_place and reload_model_in_place_owned
        let preamble = DnamPreamble::from_bytes(&container).unwrap();
        let arena_off = preamble.tensor_arena_offset as usize;
        let mut bad_arena = container.clone();
        bad_arena[arena_off] ^= 0xFF;
        assert!(
            owned_engine
                .reload_model_in_place_owned(&bad_arena, Tier::Lite, 48000)
                .is_err()
        );
        assert!(
            engine
                .reload_model_in_place(&bad_arena, Tier::Lite, 48000)
                .is_err()
        );

        // Verify reload_model_in_place_owned preserves parsed JSON header
        assert!(owned_engine.header().is_some());

        // Verify process_block advances tempo_state.total_frames and reset_to_dc_equilibrium preserves ZOH delays
        owned_engine.set_sample_rate(96_000).unwrap();
        let _ = owned_engine.register_tap_tempo(None, true);
        owned_engine.process_block(&short_in, &short_in, &mut short_l, &mut short_r);
        assert_eq!(owned_engine.tempo_state.total_frames, 8);
        let (bpm_tap, _) = owned_engine.register_tap_tempo(None, true);
        assert!((20.0..=400.0).contains(&bpm_tap));
        let d_before = owned_engine.io_router.return_aligner_l.delay_samples;
        assert!((d_before - 62.0).abs() < 1e-5);
        owned_engine.reset_to_dc_equilibrium();
        assert_eq!(
            owned_engine.io_router.return_aligner_l.delay_samples,
            d_before
        );

        // 5. Open v3.0 wire directory reload in-place and owned
        let hdr_json = serde_json::to_vec(&header).unwrap();
        let open_v3 = crate::container::build_v3_open_dnam_container(
            &hdr_json,
            &arena[..crate::container::ARENA_OFFSET_LITE_END as usize],
            &[0xAA; 64],
            &[
                &arena[..crate::container::ARENA_OFFSET_LITE_END as usize],
                &arena[crate::container::ARENA_OFFSET_LITE_END as usize
                    ..crate::container::ARENA_OFFSET_FULL_END as usize],
                &arena[crate::container::ARENA_OFFSET_FULL_END as usize
                    ..crate::container::ARENA_OFFSET_PRO_END as usize],
                &arena[crate::container::ARENA_OFFSET_PRO_END as usize
                    ..crate::container::ARENA_OFFSET_ULTRA_END as usize],
            ],
            None,
            0,
        )
        .unwrap();
        assert!(
            engine
                .reload_model_in_place(&open_v3, Tier::Full, 48_000)
                .is_ok()
        );
        assert!(
            owned_engine
                .reload_model_in_place_owned(&open_v3, Tier::Full, 48_000)
                .is_ok()
        );
    }

    #[test]
    fn test_init_from_view_stage4_zoh_and_reconfigure_edge_branches() {
        use crate::container::wire::DelayHistoryEntry;
        use crate::graph::weights::{
            Arena0LiteWire, PIPE_FLAG_SONA, PIPE_FLAG_ZERO_OP_T_PRE, PIPE_MAGIC_HI,
            SONA_CELL_MAGIC, V3_META_STAGE0_EXT,
        };
        use crate::provenance::OpenArenaCryptoProvider;
        use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints};
        use alloc::collections::BTreeMap;
        use alloc::string::String;
        use zerocopy::FromBytes;
        use zerocopy::byteorder::little_endian::{F32, U32};

        let mut hdr = DnamJsonHeader::default_reference();
        hdr.controls_schema.push(ControlSchemaDef {
            id: String::from("drive"),
            name: String::from("Drive"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(1),
            anchor_nodes_norm: Some(vec![0.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: alloc::vec![],
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        });
        hdr.delay_histories.push(DelayHistoryEntry {
            id: String::from("tank.q0"),
            owner: String::from("spatial_tank"),
            kind: String::from("fractional"),
            introduced_tier: String::from("ultra"),
            sample_format: String::from("f32"),
            reference_samples: [(String::from("ultra"), String::from("256"))]
                .into_iter()
                .collect(),
        });
        hdr.spatial_tank.dc_blocker_pole = 0.5;
        hdr.spatial_tank.tzf_reference_tap_samples = 0.0;
        hdr.control_bus.head_trust_region_radius = 0.0;

        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        {
            let (a0, _) = Arena0LiteWire::mut_from_prefix(&mut arena).unwrap();
            a0.stage0.cell_phasors.cell_desc.magic = SONA_CELL_MAGIC;
            a0.stage0.meta.pipeline_flags = U32::new(PIPE_MAGIC_HI | PIPE_FLAG_SONA);
            a0.stage0.meta.v3_meta_flags = U32::new(V3_META_STAGE0_EXT);
            a0.stage0.s_router[0][0] = F32::new(f32::NAN);
            a0.stage4_lite.ext_format = U32::new(1);
            a0.stage4_lite.allpass_delays[0] = U32::new(19);
        }
        let c = build_dnam_container(&mut hdr, &arena, &arena, false).unwrap();

        let mut bad_view = DnamContainerView::from_bytes(&c, &OpenArenaCryptoProvider).unwrap();
        bad_view.preamble.version_major = 2;
        let mut e_bad = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        assert!(matches!(
            e_bad.init_from_view(bad_view, Tier::Lite, Tier::Ultra, 48_000),
            Err(DnamError::UnsupportedVersion { major: 2, .. })
        ));

        let mut view_desc = DnamContainerView::from_bytes(&c, &OpenArenaCryptoProvider).unwrap();
        view_desc.control_records.clear();
        let mut e_desc = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        e_desc
            .init_from_view(view_desc, Tier::Lite, Tier::Ultra, 48_000)
            .unwrap();
        assert_eq!(e_desc.control_bus.num_controls, 1);

        let mut view_comp = DnamContainerView::from_bytes(&c, &OpenArenaCryptoProvider).unwrap();
        view_comp.control_records.clear();
        view_comp.clamp_descriptors.clear();
        let mut e_comp = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        e_comp
            .init_from_view(view_comp, Tier::Lite, Tier::Ultra, 48_000)
            .unwrap();
        assert_eq!(e_comp.control_bus.num_controls, 1);
        e_comp.set_sample_rate(96_000).unwrap();

        {
            let (a0, _) = Arena0LiteWire::mut_from_prefix(&mut arena).unwrap();
            a0.stage0.meta.pipeline_flags = U32::new(PIPE_MAGIC_HI | PIPE_FLAG_ZERO_OP_T_PRE);
        }
        let c_zop = build_dnam_container(&mut hdr, &arena, &arena, false).unwrap();
        let e_zop = DnamEngine::<ScalarBackend>::from_bytes(&c_zop, Tier::Lite, 48_000).unwrap();
        assert!(e_zop.training_zoh_poles().2.abs() < 1e-6);

        let mut empty = DnamEngine::<ScalarBackend>::new_empty(Tier::Lite);
        empty.header_active = false;
        empty.reloaded_header_flags = None;
        let _ = empty.training_zoh_poles();
        empty.reconfigure_for_sample_rate_rt(48_000, true).unwrap();
        empty.reconfigure_for_sample_rate_rt(48_000, false).unwrap();
    }
}
