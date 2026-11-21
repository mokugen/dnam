//! In-place zero-allocation hot-swap container validation and loading for [`DnamEngine`].

mod json_scan;
mod owned;

use dnam_vm::graph::slab::ctrl_latch::latch_immediate_at_init_or_reset;

use crate::container::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, ContainerWireError, DNAM_MAGIC, DnamPreamble,
    extract_dnam_bin_rt, parse_dcrc_record_rt, validate_v3_dnam_layout_no_alloc_rt,
};
use crate::engine::{DnamEngine, EngineBackend};
use crate::schema::{DnamError, DnamProtectionProfile};
use crate::types::Tier;

use json_scan::{RtReloadError, scan_header_flags_rt};

pub(super) struct ValidatedReloadFrame<'b> {
    pub(super) preamble: DnamPreamble,
    pub(super) header_slice: &'b [u8],
    pub(super) mutable_slice: &'b [u8],
}

pub(super) fn validate_reload_frame_rt(
    bytes: &[u8],
) -> Result<ValidatedReloadFrame<'_>, ContainerWireError> {
    let preamble = DnamPreamble::from_bytes_rt(bytes)?;
    if preamble.magic != DNAM_MAGIC {
        return Err(ContainerWireError::InvalidMagic(preamble.magic));
    }
    if (preamble.version_major, preamble.version_minor)
        != (CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR)
    {
        return Err(ContainerWireError::UnsupportedVersion {
            major: preamble.version_major,
            minor: preamble.version_minor,
        });
    }
    if preamble.is_encrypted_arena() || bytes[8] > 0 {
        return Err(ContainerWireError::EncryptedArenaRequiresDeviceKey);
    }
    let layout = validate_v3_dnam_layout_no_alloc_rt(bytes)?;
    let auth_slice = &bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    let has_dcrc = if let Some(dcrc) = parse_dcrc_record_rt(auth_slice)? {
        let calc_hdr_crc = crate::container::crc32c_update(0, &bytes[..layout.auth_extent.0]);
        if calc_hdr_crc != dcrc.crc_prefix {
            return Err(ContainerWireError::Crc32cMismatch {
                region: "header",
                expected: dcrc.crc_prefix,
                computed: calc_hdr_crc,
            });
        }
        true
    } else {
        false
    };
    if has_dcrc {
        let arena_slice =
            &bytes[layout.tier_start..layout.tier_ends[layout.retained_tier as usize]];
        crate::container::DnamContainerView::verify_arena_prefix_crcs_rt(&preamble, arena_slice)?;
    }

    Ok(ValidatedReloadFrame {
        preamble,
        header_slice: &bytes
            [layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1],
        mutable_slice: &bytes
            [layout.mutable_extent.0..layout.mutable_extent.0 + layout.mutable_extent.1],
    })
}

/// Converts [`ContainerWireError`] to [`DnamError`] with zero heap allocation (`String::new()`).
#[cold]
#[inline(never)]
pub(super) fn wire_err_to_dnam_rt(err: ContainerWireError) -> DnamError {
    match err {
        ContainerWireError::BufferTooSmall { expected, actual } => {
            DnamError::BufferTooSmall { expected, actual }
        }
        ContainerWireError::InvalidMagic(m) => DnamError::InvalidMagic(m),
        ContainerWireError::UnsupportedVersion { major, minor } => {
            DnamError::UnsupportedVersion { major, minor }
        }
        ContainerWireError::UnalignedOffset { field, offset } => {
            DnamError::UnalignedOffset { field, offset }
        }
        ContainerWireError::Crc32cMismatch {
            region,
            expected,
            computed,
        } => DnamError::Crc32cMismatch {
            region,
            expected,
            computed,
        },
        ContainerWireError::SchemaStatic(_) => {
            DnamError::SchemaValidationError(alloc::string::String::new())
        }
        ContainerWireError::EncryptedArenaRequiresDeviceKey => {
            DnamError::EncryptedArenaRequiresDeviceKey
        }
        ContainerWireError::InvalidSampleRate(sr) => DnamError::InvalidSampleRate(sr),
    }
}

/// Converts [`RtReloadError`] to [`DnamError`] strictly at the public API boundary with
/// zero heap allocation (`String::new()` only) and zero `Drop` glue.
#[cold]
#[inline(never)]
fn rt_err_to_dnam(err: RtReloadError) -> DnamError {
    match err {
        RtReloadError::Wire(wire) => wire_err_to_dnam_rt(wire),
        RtReloadError::ControlCapacityExceeded { required, capacity } => {
            DnamError::BufferTooSmall {
                expected: required,
                actual: capacity,
            }
        }
        RtReloadError::InvalidJson => DnamError::InvalidJsonHeader(alloc::string::String::new()),
        RtReloadError::UnsupportedArchitecture => DnamError::UnsupportedVersion {
            major: CONTAINER_VERSION_MAJOR,
            minor: CONTAINER_VERSION_MINOR,
        },
        RtReloadError::PreviewContainerRejected => DnamError::PreviewContainerRejected,
        RtReloadError::RejectedLegacyField => {
            DnamError::SchemaValidationError(alloc::string::String::new())
        }
    }
}

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    fn reload_model_in_place_rt(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), RtReloadError> {
        if !(44_100..=384_000).contains(&sample_rate_hz) {
            return Err(RtReloadError::Wire(ContainerWireError::InvalidSampleRate(
                sample_rate_hz,
            )));
        }
        let frame = validate_reload_frame_rt(bytes).map_err(RtReloadError::Wire)?;
        let flags = scan_header_flags_rt(frame.header_slice)?;
        let (_, bin_view) = extract_dnam_bin_rt(bytes).map_err(RtReloadError::Wire)?;

        let p_ctrls = bin_view.header.num_controls.get() as usize;
        let ctrl_cap = self
            .vm
            .slab()
            .cold()
            .ctrl_bus
            .as_ref()
            .map_or(0, |b| b.backing.capacity_limits().0);
        if p_ctrls > ctrl_cap {
            return Err(RtReloadError::ControlCapacityExceeded {
                required: p_ctrls,
                capacity: ctrl_cap,
            });
        }

        let max_avail = bin_view.retained_tier;
        let capped_max_avail = max_avail.min(self.max_supported_tier);
        let active_tier = if self.preview_mode {
            Tier::Lite
        } else {
            tier.min(capped_max_avail)
        };
        if active_tier > self.preallocated_max_tier
            || (max_avail.min(self.max_supported_tier) > self.preallocated_max_tier
                && tier > self.preallocated_max_tier)
        {
            return Err(RtReloadError::Wire(ContainerWireError::BufferTooSmall {
                expected: max_avail.expected_arena_bytes(),
                actual: self.preallocated_max_tier.expected_arena_bytes(),
            }));
        }
        let bounce_tier = capped_max_avail.min(self.preallocated_max_tier);
        let bounce_ctrl_cap = self
            .bounce_vm
            .slab()
            .cold()
            .ctrl_bus
            .as_ref()
            .map_or(0, |b| b.backing.capacity_limits().0);
        let eff_ctrl_cap = ctrl_cap.min(bounce_ctrl_cap);
        let map_vm_reload_err = |err: dnam_vm::VmError| match err {
            dnam_vm::VmError::InsufficientBackingCapacity => {
                RtReloadError::ControlCapacityExceeded {
                    required: p_ctrls.max(1),
                    capacity: eff_ctrl_cap,
                }
            }
            other => RtReloadError::Wire(crate::container::map_vm_error_rt(other)),
        };

        // Pre-validate capacity on BOTH live and bounce VMs before mutating either.
        self.vm
            .check_reload_capacity(
                &bin_view,
                active_tier,
                sample_rate_hz as f32,
                self.offline_bounce,
            )
            .map_err(map_vm_reload_err)?;
        let bounce_sr = sample_rate_hz.min(96_000) as f32;
        self.bounce_vm
            .check_reload_capacity(&bin_view, bounce_tier, bounce_sr, true)
            .map_err(map_vm_reload_err)?;

        self.vm
            .reload_from_view_in_place(
                &bin_view,
                active_tier,
                sample_rate_hz as f32,
                self.offline_bounce,
            )
            .map_err(map_vm_reload_err)?;
        self.bounce_vm
            .reload_from_view_in_place(&bin_view, bounce_tier, bounce_sr, true)
            .map_err(map_vm_reload_err)?;

        latch_immediate_at_init_or_reset(self.vm.slab_mut());
        latch_immediate_at_init_or_reset(self.bounce_vm.slab_mut());

        // Clear header & verification visibility without dropping heap Strings on the RT thread (`0` frees).
        self.header_active = false;
        self.verification_status_active = false;
        self.protection_profile = DnamProtectionProfile::Open;
        self.reloaded_header_flags = Some(flags);
        self.single_input = flags.single_input;
        self.tier = active_tier;
        self.max_available_tier = max_avail;
        self.tail_time_ms = frame.preamble.tail_time_ms;
        self.last_block_max_drive = 0.0;
        self.extra_clamp_mask = 0;
        let mut preamble = frame.preamble;
        if flags.stereo_out {
            preamble.flags |= crate::container::FLAG_STEREO_OUT;
        } else {
            preamble.flags &= !crate::container::FLAG_STEREO_OUT;
        }
        self.preamble = preamble;

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
        self.configure_host_mix_and_leveling(flags.default_blend, 0.0, true);
        for idx in 0..p_ctrls {
            if let Some(&val) = self.clamped_control_values().get(idx) {
                self.refresh_calibrated_control_effects(idx, val);
            }
        }
        Ok(())
    }

    /// Hot-swaps a new canonical `v3.0` `"sona"` `.dnam` model in-place with zero heap allocations
    /// or deallocations (`0` `DnamError` / `DnamJsonHeader` drop glue on real-time paths).
    ///
    /// Calibrated host control roles (`StereoWidth`, `StereoMode`, `Mix`), clamps, and binary
    /// `TSYN` tempo sync stay live across RT reloads, while named JSON `presets` require owned
    /// reload (`reload_model_in_place_owned`) to preserve zero heap allocations/drops on the
    /// audio thread.
    pub fn reload_model_in_place(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        self.reload_model_in_place_rt(bytes, tier, sample_rate_hz)
            .map_err(rt_err_to_dnam)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        DnamContainerView, build_dnam_container_from_dnam_bin, build_v3_open_dnam_container,
        strip_container_to_tier,
    };
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{
        BlendMode, ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef,
        ControlUiHints, DnamJsonHeader, PresetDef,
    };
    use crate::test_support::{build_test_dnam_bin, build_test_dnam_container};
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use dnam_vm::bin::{DnamBinView, PIPE_CANONICAL_V3_SONA};
    use dnam_vm::control::Cp1ContributorSlot;
    use dnam_vm::math::ScalarBackend;
    use dnam_vm::types::Tier as VmTier;

    #[test]
    fn test_reload_borrowed_ext_flags_and_errors() {
        let mut lite_header = DnamJsonHeader::default_reference();
        let full_c = build_test_dnam_container(&mut lite_header, Tier::Ultra, true);
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

        let mut engine = DnamEngine::<ScalarBackend>::with_max_capacity();
        engine
            .reload_model_in_place(&lite_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(matches!(
            engine.reload_model_in_place(&[0u8; 256], Tier::Lite, 48_000),
            Err(DnamError::InvalidMagic(_))
        ));
        let mut enc = lite_c.clone();
        enc[8] = 1;
        assert!(matches!(
            engine.reload_model_in_place(&enc, Tier::Lite, 48_000),
            Err(DnamError::EncryptedArenaRequiresDeviceKey)
        ));
        let mut bad_crc = lite_c.clone();
        bad_crc[192] ^= 0x5A;
        assert!(matches!(
            engine.reload_model_in_place(&bad_crc, Tier::Lite, 48_000),
            Err(DnamError::Crc32cMismatch {
                region: "header",
                ..
            })
        ));

        engine
            .reload_model_in_place(&ext_c, Tier::Ultra, 44_100)
            .unwrap();
        assert!(engine.header().is_none());
        assert_eq!(engine.tier(), Tier::Ultra);
        assert_eq!(
            engine.vm.slab().cold().wire_crossbar.pipeline_flags.get(),
            PIPE_CANONICAL_V3_SONA
        );

        let mut small_engine =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&lite_c, Tier::Lite, 48_000).unwrap();
        assert!(matches!(
            small_engine.reload_model_in_place(&ext_c, Tier::Ultra, 44_100),
            Err(DnamError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_reload_rejects_preview_malformed_json_and_propagates_dsp_flags() {
        let mut base_hdr = DnamJsonHeader::default_reference();
        let base_c = build_test_dnam_container(&mut base_hdr, Tier::Ultra, true);

        let mut prev_hdr = DnamJsonHeader::default_reference();
        prev_hdr.container_kind = String::from("preview");
        let prev_c = build_test_dnam_container(&mut prev_hdr, Tier::Lite, true);

        let mut flags_hdr = DnamJsonHeader::default_reference();
        flags_hdr.io_config.inputs = 1;
        flags_hdr.io_config.capabilities.default_blend = 0.25;
        flags_hdr.io_config.capabilities.wet_polarity_inverted = true;
        flags_hdr.io_config.leveling.auto_unity_enabled = false;
        flags_hdr.io_config.leveling.calibrated_trim_db = 3.0;
        let flags_c = build_test_dnam_container(&mut flags_hdr, Tier::Lite, true);

        let mut borrowed = DnamEngine::<ScalarBackend>::with_max_capacity();
        borrowed
            .reload_model_in_place(&base_c, Tier::Lite, 48_000)
            .unwrap();
        borrowed.protection_profile = DnamProtectionProfile::DeviceAllowlisted;

        assert!(matches!(
            borrowed.reload_model_in_place(&prev_c, Tier::Lite, 48_000),
            Err(DnamError::PreviewContainerRejected)
        ));
        borrowed
            .reload_model_in_place(&flags_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(borrowed.header().is_none());
        assert_eq!(borrowed.protection_profile(), DnamProtectionProfile::Open);
        assert!(borrowed.single_input);
        assert!((borrowed.mix_dry_gain - 0.75).abs() < 1e-4);
        assert!((borrowed.mix_wet_gain - 0.25).abs() < 1e-4);
        assert!((borrowed.telemetry_snapshot().mix_wet_gain - (-0.25)).abs() < 1e-4);

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
            borrowed.reload_model_in_place(&bad_json_container, Tier::Lite, 48_000),
            Err(DnamError::InvalidJsonHeader(_))
        ));

        let mut mono_hdr = DnamJsonHeader::default_reference();
        mono_hdr.io_config.output_mode = String::from("mono");
        let mono_c = build_test_dnam_container(&mut mono_hdr, Tier::Lite, true);
        borrowed
            .reload_model_in_place(&mono_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(!borrowed.preamble.is_stereo_out());
        assert_eq!(borrowed.telemetry_snapshot().stereo_out, 0);
        borrowed
            .reload_model_in_place(&base_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(borrowed.preamble.is_stereo_out());
        assert_eq!(borrowed.telemetry_snapshot().stereo_out, 1);
    }

    #[test]
    fn test_reload_cp1_control_capacity_and_sample_rate_paths() {
        assert!(matches!(
            wire_err_to_dnam_rt(ContainerWireError::UnalignedOffset {
                field: "f",
                offset: 3,
            }),
            DnamError::UnalignedOffset {
                field: "f",
                offset: 3,
            }
        ));

        let mut hdr = DnamJsonHeader::default_reference();
        let mut ctrl_um = BTreeMap::new();
        ctrl_um.insert(String::from("color"), serde_json::json!("red"));
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
            conditional_bounds: alloc::vec::Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: None,
            ui_hints: ControlUiHints::default(),
            user_metadata: ctrl_um,
        });
        hdr.stem_user_metadata
            .insert(String::from("s0"), serde_json::json!({"gain": 1.0}));

        let descriptors = hdr.compile_clamp_descriptors().unwrap();
        let wire_ctrls: vec::Vec<_> = descriptors.iter().map(|d| d.to_wire()).collect();
        let mut slot = Cp1ContributorSlot::zero();
        slot.lanes = 1;
        slot.control_idx = 0;
        slot.main_base[0][0] = 0.25;
        slot.emb[0] = 1.0;
        slot.v_base[0] = 0.1;

        let cp1_stream = crate::container::encode_cp1_shared_stream(&[slot]).unwrap();
        let dbin = {
            let a0 = dnam_vm::bin::DnamBinBuildInput::default_arena0_lite();
            dnam_vm::bin::encode_dnam_bin(&dnam_vm::bin::DnamBinBuildInput {
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
            .unwrap()
        };
        let c_cp1 = build_dnam_container_from_dnam_bin(&mut hdr, &dbin).unwrap();

        let view =
            DnamContainerView::from_bytes(&c_cp1, &DefaultCryptoProvider::default()).unwrap();
        assert_eq!(view.cp1_contributors.len(), 1);

        let mut eng = DnamEngine::<ScalarBackend>::with_max_capacity();
        assert!(matches!(
            eng.reload_model_in_place(&c_cp1, Tier::Lite, 22_050),
            Err(DnamError::InvalidSampleRate(22_050))
        ));
        eng.reload_model_in_place(&c_cp1, Tier::Lite, 48_000)
            .unwrap();
        assert_eq!(
            eng.vm
                .slab()
                .cold()
                .ctrl_bus
                .as_ref()
                .unwrap()
                .num_contributors,
            1
        );

        let base_zero_dbin = build_test_dnam_bin(
            Tier::Lite,
            48_000,
            dnam_vm::bin::DNBN_FLAG_STEREO_OUT,
            &[],
            &[],
            true,
        );
        let mut base_zero_hdr = DnamJsonHeader::default_reference();
        let base_zero_c =
            build_dnam_container_from_dnam_bin(&mut base_zero_hdr, &base_zero_dbin).unwrap();
        let mut tight =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&base_zero_c, Tier::Lite, 48_000)
                .unwrap();
        let mut two_hdr = DnamJsonHeader::default_reference();
        two_hdr.controls_schema = alloc::vec![
            ControlSchemaDef {
                id: String::from("c0"),
                name: String::from("C0"),
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
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("c1"),
                name: String::from("C1"),
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
                user_metadata: BTreeMap::new(),
            },
        ];
        let c_two_ctrls = build_test_dnam_container(&mut two_hdr, Tier::Lite, false);
        assert!(matches!(
            tight.reload_model_in_place(&c_two_ctrls, Tier::Lite, 48_000),
            Err(DnamError::BufferTooSmall {
                expected: 2,
                actual: 1,
            })
        ));
        let _ = VmTier::Lite;
    }

    #[test]
    fn test_rt_reload_atomic_failure_when_bounce_vm_lacks_capacity_preserves_live_audio() {
        let mut base_hdr = DnamJsonHeader::default_reference();
        let base_c = build_test_dnam_container(&mut base_hdr, Tier::Lite, true);

        let mut two_hdr = DnamJsonHeader::default_reference();
        two_hdr.controls_schema = alloc::vec![
            ControlSchemaDef {
                id: String::from("c0"),
                name: String::from("C0"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(1),
                anchor_nodes_norm: Some(vec![0.0]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(3.0),
                conditional_bounds: alloc::vec::Vec::new(),
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("c1"),
                name: String::from("C1"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(1),
                anchor_nodes_norm: Some(vec![0.0]),
                declared_range: Some([0.0, 10.0]),
                safe_clamped_range: Some([0.0, 10.0]),
                default: serde_json::json!(7.0),
                conditional_bounds: alloc::vec::Vec::new(),
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: None,
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let two_c = build_test_dnam_container(&mut two_hdr, Tier::Lite, false);

        let mut eng = DnamEngine::<ScalarBackend>::with_max_capacity();
        eng.reload_model_in_place(&base_c, Tier::Lite, 48_000)
            .unwrap();
        // Replace `bounce_vm` with a tight 0-control VM while `vm` retains full headroom.
        let zero_dbin = build_test_dnam_bin(
            Tier::Lite,
            48_000,
            dnam_vm::bin::DNBN_FLAG_STEREO_OUT,
            &[],
            &[],
            true,
        );
        let zero_view = DnamBinView::parse(&zero_dbin).unwrap();
        *eng.bounce_vm = dnam_vm::SonaVm::from_view_owned_with_headroom(
            &zero_view,
            Tier::Lite,
            48_000.0,
            true,
            dnam_vm::graph::VmReloadHeadroom::zero(),
        )
        .unwrap();

        let mut ref_eng = DnamEngine::<ScalarBackend>::with_max_capacity();
        ref_eng
            .reload_model_in_place(&base_c, Tier::Lite, 48_000)
            .unwrap();

        let in_l = [0.3_f32; 32];
        let in_r = [-0.2_f32; 32];
        let (mut pre_l, mut pre_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        let (mut ref_l, mut ref_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        eng.process_block(&in_l, &in_r, &mut pre_l, &mut pre_r);
        ref_eng.process_block(&in_l, &in_r, &mut ref_l, &mut ref_r);
        assert_eq!(pre_l, ref_l);
        assert_eq!(pre_r, ref_r);

        assert!(matches!(
            eng.reload_model_in_place(&two_c, Tier::Lite, 48_000),
            Err(DnamError::BufferTooSmall { .. })
        ));
        assert_eq!(eng.num_controls(), 0);

        let (mut post_l, mut post_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        let (mut ref2_l, mut ref2_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        eng.process_block(&in_l, &in_r, &mut post_l, &mut post_r);
        ref_eng.process_block(&in_l, &in_r, &mut ref2_l, &mut ref2_r);
        assert_eq!(post_l, ref2_l);
        assert_eq!(post_r, ref2_r);
    }

    #[test]
    fn test_rt_reload_resets_ms_width_and_preserves_calibrated_host_roles() {
        let mut base_hdr = DnamJsonHeader::default_reference();
        let base_c = build_test_dnam_container(&mut base_hdr, Tier::Lite, true);

        let mut cal_hdr = DnamJsonHeader::default_reference();
        cal_hdr.io_config.capabilities.blend_mode = BlendMode::HostParallel;
        cal_hdr.io_config.capabilities.default_blend = 0.5;
        cal_hdr.io_config.capabilities.wet_polarity_inverted = true;
        cal_hdr.io_config.leveling.auto_unity_enabled = true;
        cal_hdr.io_config.leveling.calibrated_trim_db = -6.0206;
        cal_hdr.controls_schema = alloc::vec![
            ControlSchemaDef {
                id: String::from("width"),
                name: String::from("Width"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 2.0]),
                safe_clamped_range: Some([0.0, 2.0]),
                default: serde_json::json!(1.4),
                conditional_bounds: alloc::vec::Vec::new(),
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoWidth,
                    lobatto_physical_anchors: Some([0.0, 1.0, 2.0]),
                    conditional_physical_anchors: alloc::vec::Vec::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("stereo_mode"),
                name: String::from("Stereo Mode"),
                kind: ControlKind::Switch,
                unit: None,
                lobatto_degree: None,
                anchor_nodes_norm: None,
                declared_range: None,
                safe_clamped_range: None,
                default: serde_json::json!("Stereo"),
                conditional_bounds: alloc::vec::Vec::new(),
                represented_switch_options: None,
                declared_options: Some(vec![String::from("Mono"), String::from("Stereo")]),
                active_options: Some(vec![String::from("Mono"), String::from("Stereo")]),
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::StereoMode,
                    lobatto_physical_anchors: None,
                    conditional_physical_anchors: alloc::vec::Vec::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
            ControlSchemaDef {
                id: String::from("mix"),
                name: String::from("Mix"),
                kind: ControlKind::Range,
                unit: None,
                lobatto_degree: Some(2),
                anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
                declared_range: Some([0.0, 100.0]),
                safe_clamped_range: Some([0.0, 100.0]),
                default: serde_json::json!(40.0),
                conditional_bounds: alloc::vec::Vec::new(),
                represented_switch_options: None,
                declared_options: None,
                active_options: None,
                clamp_warning: None,
                calibration: Some(ControlCalibration {
                    role: ControlCalibrationRole::Mix,
                    lobatto_physical_anchors: Some([0.0, 0.5, 1.0]),
                    conditional_physical_anchors: alloc::vec::Vec::new(),
                    tempo_sync_default_subdivision: None,
                }),
                ui_hints: ControlUiHints::default(),
                user_metadata: BTreeMap::new(),
            },
        ];
        let cal_c = build_test_dnam_container(&mut cal_hdr, Tier::Lite, true);

        let mut eng = DnamEngine::<ScalarBackend>::with_max_capacity();
        eng.reload_model_in_place(&base_c, Tier::Lite, 48_000)
            .unwrap();
        eng.set_stereo_width(1.8);
        assert!((eng.stereo_width() - 1.8).abs() < 1.0e-6);

        // Reloading back to `base_c` (which has no width control) resets stale `ms_width` to 1.0.
        eng.reload_model_in_place(&base_c, Tier::Lite, 48_000)
            .unwrap();
        assert!((eng.stereo_width() - 1.0).abs() < 1.0e-6);

        // Reloading `cal_c` applies default calibrated Mix (40% -> 0.4) and StereoMode (`Stereo` -> 1.0).
        eng.reload_model_in_place(&cal_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(eng.header().is_none());
        assert!((eng.stereo_width() - 1.0).abs() < 1e-6);
        assert!((eng.mix_dry_gain - 0.6).abs() < 1e-4);
        assert!((eng.mix_wet_gain - 0.2).abs() < 1.0e-3);
        assert!((eng.telemetry_snapshot().mix_wet_gain - (-0.2)).abs() < 1.0e-3);

        // Live control updates continue driving StereoWidth, StereoMode, and Mix without DnamJsonHeader!
        eng.set_control_value(0, 1.65);
        assert!((eng.stereo_width() - 1.65).abs() < 1.0e-5);
        eng.set_control_immediate(1, 0.0);
        assert!((eng.stereo_width() - 0.0).abs() < 1.0e-6);
        eng.set_control_value(2, 80.0);
        assert!((eng.mix_dry_gain - 0.2).abs() < 1.0e-4);
        assert!((eng.mix_wet_gain - 0.4).abs() < 1.0e-3);
        assert!((eng.telemetry_snapshot().mix_wet_gain - (-0.4)).abs() < 1.0e-3);
    }
}
