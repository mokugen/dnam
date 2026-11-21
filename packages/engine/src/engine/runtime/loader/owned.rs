//! Owned container hot-swapping (`reload_model_in_place_owned`) for [`DnamEngine`].

use dnam_vm::bin::DnamBinView;
use dnam_vm::graph::slab::ctrl_latch::latch_immediate_at_init_or_reset;

use crate::container::{ContainerWireError, extract_dnam_bin_rt, parse_and_validate_dmu1_frame};
use crate::engine::{DnamEngine, EngineBackend};
use crate::provenance::VerificationStatus;
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile};
use crate::types::Tier;

use super::validate_reload_frame_rt;

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Hot-swaps a new canonical `v3.0` `"sona"` `.dnam` model in-place into an owned engine
    /// (`r[sec.preview.containment]`, fail-closed JSON, `DMU1`, and `.dnam-bin` validation).
    pub fn reload_model_in_place_owned(
        &mut self,
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        if !(44_100..=384_000).contains(&sample_rate_hz) {
            return Err(DnamError::InvalidSampleRate(sample_rate_hz));
        }
        let frame = validate_reload_frame_rt(bytes).map_err(ContainerWireError::into_dnam_error)?;
        let mut hdr = DnamJsonHeader::from_json_slice(frame.header_slice)
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
        if hdr.architecture != "sona" {
            return Err(DnamError::UnsupportedVersion {
                major: frame.preamble.version_major,
                minor: frame.preamble.version_minor,
            });
        }
        hdr.ensure_not_preview_for_daw_host()?;
        let dmu_val = parse_and_validate_dmu1_frame(frame.mutable_slice)?;
        crate::container::validate_mutable_json_schema(&dmu_val)?;
        if let Some(presets_val) = dmu_val.get("presets") {
            hdr.presets = serde_json::from_value(presets_val.clone())
                .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
        }
        if let Some(um_val) = dmu_val.get("user_metadata") {
            hdr.user_metadata = serde_json::from_value(um_val.clone())
                .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
        }
        if let Some(ctrl_um_obj) = dmu_val
            .get("control_user_metadata")
            .and_then(|v| v.as_object())
        {
            for ctrl in &mut hdr.controls_schema {
                if let Some(entry_val) = ctrl_um_obj.get(&ctrl.id) {
                    if let Some(map) = entry_val.as_object() {
                        for (k, v) in map {
                            ctrl.user_metadata.insert(k.clone(), v.clone());
                        }
                    } else {
                        ctrl.user_metadata
                            .insert(alloc::string::String::from("value"), entry_val.clone());
                    }
                }
            }
        }
        if let Some(stem_um_obj) = dmu_val
            .get("stem_user_metadata")
            .and_then(|v| v.as_object())
        {
            for (k, v) in stem_um_obj {
                hdr.stem_user_metadata.insert(k.clone(), v.clone());
            }
        }

        let (dbin_slice, _) =
            extract_dnam_bin_rt(bytes).map_err(ContainerWireError::into_dnam_error)?;
        let bin_view = DnamBinView::parse(dbin_slice).map_err(crate::container::map_vm_error)?;
        let max_avail = bin_view.retained_tier;
        let effective_ceiling = if self.preview_mode {
            Tier::Lite
        } else {
            self.max_supported_tier
        };
        let active_tier = tier.min(max_avail).min(effective_ceiling);
        let bounce_tier = max_avail.min(effective_ceiling);
        let bounce_sr = sample_rate_hz.min(96_000) as f32;

        // Pre-validate or construct replacement VMs before mutating `self` so failures remain atomic.
        let headroom = dnam_vm::graph::VmReloadHeadroom::for_live_engine()
            .with_transport_headroom(16_384, 16_384, 4_096);
        let live_upgrade = if self
            .vm
            .check_reload_capacity(
                &bin_view,
                active_tier,
                sample_rate_hz as f32,
                self.offline_bounce,
            )
            .is_err()
        {
            Some(alloc::boxed::Box::new(
                dnam_vm::SonaVm::from_view_owned_with_headroom(
                    &bin_view,
                    active_tier,
                    sample_rate_hz as f32,
                    self.offline_bounce,
                    headroom,
                )
                .map_err(crate::container::map_vm_error)?,
            ))
        } else {
            None
        };
        let bounce_upgrade = if self
            .bounce_vm
            .check_reload_capacity(&bin_view, bounce_tier, bounce_sr, true)
            .is_err()
        {
            Some(alloc::boxed::Box::new(
                dnam_vm::SonaVm::from_view_owned_with_headroom(
                    &bin_view,
                    bounce_tier,
                    bounce_sr,
                    true,
                    headroom,
                )
                .map_err(crate::container::map_vm_error)?,
            ))
        } else {
            None
        };

        if let Some(new_vm) = live_upgrade {
            self.vm = new_vm;
            self.preallocated_max_tier = self.preallocated_max_tier.max(max_avail);
        } else {
            self.vm
                .reload_from_view_in_place(
                    &bin_view,
                    active_tier,
                    sample_rate_hz as f32,
                    self.offline_bounce,
                )
                .map_err(crate::container::map_vm_error)?;
        }
        if let Some(new_bounce_vm) = bounce_upgrade {
            self.bounce_vm = new_bounce_vm;
        } else {
            self.bounce_vm
                .reload_from_view_in_place(&bin_view, bounce_tier, bounce_sr, true)
                .map_err(crate::container::map_vm_error)?;
        }

        latch_immediate_at_init_or_reset(self.vm.slab_mut());
        latch_immediate_at_init_or_reset(self.bounce_vm.slab_mut());

        self.single_input = hdr.io_config.inputs == 1;
        let default_blend = hdr.io_config.capabilities.default_blend;
        let mut preamble = frame.preamble;
        if hdr.io_config.output_mode == "mono" {
            preamble.flags &= !crate::container::FLAG_STEREO_OUT;
        } else {
            preamble.flags |= crate::container::FLAG_STEREO_OUT;
        }
        self.preamble = preamble;
        self.tail_time_ms = hdr.io_config.tail_time_ms;
        self.header = Some(hdr);
        self.header_active = true;
        self.verification_status = VerificationStatus::Unsigned;
        self.verification_status_active = true;
        self.protection_profile = DnamProtectionProfile::Open;
        self.reloaded_header_flags = None;
        self.tier = active_tier;
        self.max_available_tier = max_avail;
        self.last_block_max_drive = 0.0;
        self.extra_clamp_mask = 0;

        let live_zoh = self.vm.slab().cold().zoh;
        self.patchbay
            .reconfigure_sample_rate(live_zoh.f_s_vm, live_zoh.is_bounce_4x);
        self.patchbay.set_ms_width(1.0);
        self.patchbay.reset();
        let bounce_zoh = self.bounce_vm.slab().cold().zoh;
        self.bounce_patchbay
            .reconfigure_sample_rate(bounce_zoh.f_s_vm, bounce_zoh.is_bounce_4x);
        self.bounce_patchbay.set_ms_width(1.0);
        self.bounce_patchbay.reset();
        self.preview_guard.prepare(sample_rate_hz as f32);
        self.tempo_state.sample_rate_hz = sample_rate_hz;
        self.configure_host_mix_and_leveling(default_blend, 0.0, true);
        let num_ctrls = self.num_controls();
        for idx in 0..num_ctrls {
            if let Some(&val) = self.clamped_control_values().get(idx) {
                self.refresh_calibrated_control_effects(idx, val);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        build_dnam_container_from_dnam_bin, build_v3_open_dnam_container, strip_container_to_tier,
    };
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, PresetDef};
    use crate::test_support::build_test_dnam_container;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use dnam_vm::control::Cp1ContributorSlot;
    use dnam_vm::math::ScalarBackend;

    #[test]
    fn test_owned_reload_presets_flags_metadata_and_validation() {
        let mut base_hdr = DnamJsonHeader::default_reference();
        let full_c = build_test_dnam_container(&mut base_hdr, Tier::Ultra, true);
        let lite_c = strip_container_to_tier(&full_c, Tier::Lite).unwrap();

        let mut ext_header = DnamJsonHeader::default_reference();
        let mut p_ctrls = BTreeMap::new();
        p_ctrls.insert(String::from("drive"), serde_json::json!(6.0));
        ext_header.presets.push(PresetDef {
            id: String::from("reloaded_preset"),
            name: String::from("Reloaded Preset"),
            description: None,
            controls: p_ctrls,
            user_metadata: BTreeMap::new(),
        });
        ext_header
            .user_metadata
            .insert(String::from("author_note"), serde_json::json!("v2"));
        let ext_c = build_test_dnam_container(&mut ext_header, Tier::Ultra, true);

        let mut owned =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&lite_c, Tier::Lite, 48_000).unwrap();
        owned.set_stereo_width(1.85);
        owned
            .reload_model_in_place_owned(&ext_c, Tier::Ultra, 44_100)
            .unwrap();
        assert!((owned.stereo_width() - 1.0).abs() < 1.0e-6);
        let hdr = owned.header().expect("owned reload keeps header");
        assert!(hdr.presets.iter().any(|p| p.id == "reloaded_preset"));
        assert_eq!(
            hdr.user_metadata.get("author_note"),
            Some(&serde_json::json!("v2"))
        );

        let mut prev_hdr = DnamJsonHeader::default_reference();
        prev_hdr.container_kind = String::from("preview");
        let prev_c = build_test_dnam_container(&mut prev_hdr, Tier::Lite, true);
        owned.protection_profile = DnamProtectionProfile::DeviceAllowlisted;
        assert!(matches!(
            owned.reload_model_in_place_owned(&prev_c, Tier::Lite, 48_000),
            Err(DnamError::PreviewContainerRejected)
        ));

        let mut flags_hdr = DnamJsonHeader::default_reference();
        flags_hdr.io_config.inputs = 1;
        flags_hdr.io_config.capabilities.default_blend = 0.25;
        flags_hdr.io_config.capabilities.wet_polarity_inverted = true;
        flags_hdr.io_config.leveling.auto_unity_enabled = false;
        flags_hdr.io_config.leveling.calibrated_trim_db = 3.0;
        let flags_c = build_test_dnam_container(&mut flags_hdr, Tier::Lite, true);
        owned
            .reload_model_in_place_owned(&flags_c, Tier::Lite, 48_000)
            .unwrap();
        assert_eq!(owned.verification_status(), &VerificationStatus::Unsigned);
        assert_eq!(owned.protection_profile(), DnamProtectionProfile::Open);
        assert!((owned.mix_dry_gain - 0.75).abs() < 1.0e-4);

        let bad_arena = vec![0u8; crate::container::ARENA_OFFSET_LITE_END as usize];
        let bad_json_container = build_v3_open_dnam_container(
            b"{not valid json",
            &[0u8; 64],
            &[0u8; 64],
            &[&bad_arena],
            None,
            0,
        )
        .unwrap();
        assert!(matches!(
            owned.reload_model_in_place_owned(&bad_json_container, Tier::Lite, 48_000),
            Err(DnamError::InvalidJsonHeader(_))
        ));

        let mut mono_hdr = DnamJsonHeader::default_reference();
        mono_hdr.io_config.output_mode = String::from("mono");
        let mono_c = build_test_dnam_container(&mut mono_hdr, Tier::Lite, true);
        owned
            .reload_model_in_place_owned(&mono_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(!owned.preamble.is_stereo_out());
        assert_eq!(owned.telemetry_snapshot().stereo_out, 0);

        let mut cp1_hdr = DnamJsonHeader::default_reference();
        let mut ctrl_um = BTreeMap::new();
        ctrl_um.insert(String::from("color"), serde_json::json!("red"));
        cp1_hdr.controls_schema.push(ControlSchemaDef {
            id: String::from("drive"),
            name: String::from("Drive"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(1),
            anchor_nodes_norm: Some(vec![0.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: alloc::vec::Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: ctrl_um,
        });
        cp1_hdr
            .stem_user_metadata
            .insert(String::from("s0"), serde_json::json!({"gain": 1.0}));
        let descriptors = cp1_hdr.compile_clamp_descriptors().unwrap();
        let wire_ctrls: vec::Vec<_> = descriptors.iter().map(|d| d.to_wire()).collect();
        let mut slot = Cp1ContributorSlot::zero();
        slot.lanes = 1;
        slot.control_idx = 0;
        slot.main_base[0][0] = 0.25;
        slot.emb[0] = 1.0;
        slot.v_base[0] = 0.1;
        let cp1_stream = crate::container::encode_cp1_shared_stream(&[slot]).unwrap();
        let a0 = dnam_vm::bin::DnamBinBuildInput::default_arena0_lite();
        let dbin = dnam_vm::bin::encode_dnam_bin(&dnam_vm::bin::DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: dnam_vm::bin::DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: 0.995,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.0,
            switch_xfade_samples: 128,
            content_hash_prefix: [0; 8],
            controls: &wire_ctrls,
            cp1_overflow: &cp1_stream,
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap();
        let c_cp1 = build_dnam_container_from_dnam_bin(&mut cp1_hdr, &dbin).unwrap();
        owned
            .reload_model_in_place_owned(&c_cp1, Tier::Lite, 48_000)
            .unwrap();
        let reloaded = owned.header().unwrap();
        assert!(
            reloaded.controls_schema[0]
                .user_metadata
                .contains_key("color")
        );
        assert!(reloaded.stem_user_metadata.contains_key("s0"));

        let mut mlp_hdr = DnamJsonHeader::default_reference();
        mlp_hdr.architecture = String::from("mlp");
        let mlp_json = serde_json::to_vec(&mlp_hdr).unwrap();
        let mlp_arena = vec![0u8; crate::container::ARENA_OFFSET_LITE_END as usize];
        let mlp_c =
            build_v3_open_dnam_container(&mlp_json, &[0u8; 64], &[0u8; 64], &[&mlp_arena], None, 0)
                .unwrap();
        assert!(matches!(
            owned.reload_model_in_place_owned(&mlp_c, Tier::Lite, 48_000),
            Err(DnamError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn test_owned_reload_at_192khz_and_384khz_reloads_bounce_vm_and_renders_cleanly() {
        let mut lite_hdr = DnamJsonHeader::default_reference();
        let full_c = build_test_dnam_container(&mut lite_hdr, Tier::Ultra, true);
        let lite_c = strip_container_to_tier(&full_c, Tier::Lite).unwrap();

        let mut ultra_hdr = DnamJsonHeader::default_reference();
        ultra_hdr.controls_schema.push(ControlSchemaDef {
            id: String::from("drive"),
            name: String::from("Drive"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(1),
            anchor_nodes_norm: Some(vec![0.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(4.0),
            conditional_bounds: alloc::vec::Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        });
        let ultra_c = build_test_dnam_container(&mut ultra_hdr, Tier::Ultra, true);

        for sr in [192_000_u32, 384_000_u32] {
            let mut owned =
                DnamEngine::<ScalarBackend>::from_bytes_owned(&lite_c, Tier::Lite, 48_000).unwrap();
            owned
                .reload_model_in_place_owned(&ultra_c, Tier::Pro, sr)
                .unwrap();

            assert_eq!(owned.sample_rate_hz(), sr);
            assert_eq!(owned.tier(), Tier::Pro);
            assert_eq!(owned.vm.slab().cold().zoh.host_rate_hz, sr as f32);
            assert_eq!(owned.bounce_vm.slab().cold().zoh.host_rate_hz, 96_000.0);
            assert_eq!(owned.bounce_vm.slab().cold().zoh.f_s_vm, 384_000.0);
            assert!(owned.bounce_vm.slab().cold().zoh.is_bounce_4x);

            let in_l = [0.25_f32; 32];
            let in_r = [-0.25_f32; 32];
            let (mut out_l, mut out_r) = ([0.0_f32; 32], [0.0_f32; 32]);
            owned.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
            assert!(out_l.iter().all(|v| v.is_finite()) && out_r.iter().all(|v| v.is_finite()));
        }
    }
}
