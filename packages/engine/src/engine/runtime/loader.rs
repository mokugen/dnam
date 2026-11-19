//! In-place hot-swap container validation and loading for [`DnamEngine`].

mod json_scan;

use crate::container::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, CP1_MAGIC, ContainerWireError, DNAM_MAGIC,
    DnamContainerView, DnamPreamble, crc32c_update, decode_cp1_shared_stream_into_slice_rt,
    parse_and_validate_dmu1_frame, parse_dcrc_record_rt, validate_v3_dnam_layout_no_alloc_rt,
};
use crate::control::ControlClampDescriptor;
use crate::engine::{DnamEngine, EngineBackend};
use crate::memory::ArenaStorage;
use crate::provenance::VerificationStatus;
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile};
use crate::types::Tier;

use json_scan::{RtReloadError, scan_header_flags_rt};

struct ValidatedReloadFrame<'b> {
    preamble: DnamPreamble,
    arena_slice: &'b [u8],
    shared_extent_slice: &'b [u8],
    header_slice: &'b [u8],
    mutable_slice: &'b [u8],
}

fn validate_reload_frame_rt(bytes: &[u8]) -> Result<ValidatedReloadFrame<'_>, ContainerWireError> {
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
        let calc_hdr_crc = crc32c_update(0, &bytes[..layout.auth_extent.0]);
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

    let arena_slice = &bytes[layout.tier_start..layout.tier_ends[layout.retained_tier as usize]];
    if has_dcrc {
        DnamContainerView::verify_arena_prefix_crcs_rt(&preamble, arena_slice)?;
    }

    Ok(ValidatedReloadFrame {
        preamble,
        arena_slice,
        shared_extent_slice: &bytes
            [layout.shared_extent.0..layout.shared_extent.0 + layout.shared_extent.1],
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
    }
}

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    #[inline]
    fn preallocated_control_capacity(&self) -> usize {
        self.control_bus
            .descriptors
            .len()
            .min(self.control_bus.projections.len())
            .min(self.control_bus.raw_values.len())
            .min(self.control_bus.clamped_values.len())
            .min(self.control_bus.target_u_norm.len())
            .min(self.control_bus.current_u_norm.len())
            .min(self.control_bus.prev_eval_u_norm.len())
            .min(self.control_bus.prev_eval_psi.len())
            .min(self.control_bus.active_switch_options.len())
    }

    fn apply_validated_reload_rt(
        &mut self,
        preamble: DnamPreamble,
        arena_len: usize,
        shared_extent_slice: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), RtReloadError> {
        let max_avail = Self::detect_max_tier(&preamble, arena_len);
        let active_tier = tier.min(max_avail).min(self.max_supported_tier);
        let p_ctrls = preamble.num_controls as usize;
        self.control_bus.num_contributors = 0;
        self.control_bus.num_controls = p_ctrls;
        if let Some(desc_bytes_len) = p_ctrls.checked_mul(64)
            && shared_extent_slice.len() >= desc_bytes_len
        {
            for p in 0..p_ctrls {
                let off = p * 64;
                let mut db = [0u8; 64];
                db.copy_from_slice(&shared_extent_slice[off..off + 64]);
                self.control_bus.descriptors[p] = ControlClampDescriptor::from_bytes(&db);
            }
            let cp1_tail = &shared_extent_slice[desc_bytes_len..];
            if cp1_tail.len() >= 8
                && cp1_tail[0..4] == CP1_MAGIC
                && let Ok(cnt) = decode_cp1_shared_stream_into_slice_rt(
                    cp1_tail,
                    &mut self.control_bus.contributors,
                )
            {
                self.control_bus.num_contributors = cnt;
            }
        }
        self.preamble = preamble;
        self.tier = active_tier;
        self.max_available_tier = max_avail;
        self.reconfigure_for_sample_rate_rt(sample_rate_hz, true)
            .map_err(RtReloadError::Wire)
    }

    fn reload_model_in_place_rt(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), RtReloadError> {
        if !(44_100..=192_000).contains(&sample_rate_hz) {
            return Err(RtReloadError::Wire(ContainerWireError::InvalidSampleRate(
                sample_rate_hz,
            )));
        }
        let frame = validate_reload_frame_rt(bytes).map_err(RtReloadError::Wire)?;
        let flags = scan_header_flags_rt(frame.header_slice)?;
        let p_ctrls = frame.preamble.num_controls as usize;
        let ctrl_cap = self.preallocated_control_capacity();
        if p_ctrls > ctrl_cap {
            return Err(RtReloadError::ControlCapacityExceeded {
                required: p_ctrls,
                capacity: ctrl_cap,
            });
        }
        let max_avail = Self::detect_max_tier(&frame.preamble, frame.arena_slice.len());
        let capped_max_avail = max_avail.min(self.max_supported_tier);
        let capped_arena_len = capped_max_avail
            .expected_arena_bytes()
            .min(frame.arena_slice.len());
        let target_arena = &frame.arena_slice[..capped_arena_len];
        if !self.arena_storage.reload_in_place_or_reborrow(target_arena) {
            return Err(RtReloadError::Wire(ContainerWireError::BufferTooSmall {
                expected: target_arena.len(),
                actual: self.arena_storage.capacity_bytes(),
            }));
        }
        // Clear header & verification visibility without dropping heap Strings on the RT thread (`0` frees).
        self.header_active = false;
        self.verification_status_active = false;
        self.protection_profile = DnamProtectionProfile::Open;
        self.reloaded_header_flags = Some(flags);
        self.single_input = flags.single_input;
        self.freeze_mutes_dry = flags.freeze_mutes_dry;
        let mut preamble = frame.preamble;
        if flags.stereo_out {
            preamble.flags |= crate::container::FLAG_STEREO_OUT;
        } else {
            preamble.flags &= !crate::container::FLAG_STEREO_OUT;
        }
        self.apply_validated_reload_rt(
            preamble,
            capped_arena_len,
            frame.shared_extent_slice,
            tier,
            sample_rate_hz,
        )?;
        self.configure_host_mix_and_leveling(flags.default_blend, 0.0, true);
        Ok(())
    }

    /// Hot-swaps a new canonical `v3.0` `"sona"` `.dnam` model in-place with zero heap allocations
    /// or deallocations (`0` `DnamError` / `ArenaStorage` drop glue on real-time paths).
    pub fn reload_model_in_place(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        self.reload_model_in_place_rt(bytes, tier, sample_rate_hz)
            .map_err(rt_err_to_dnam)
    }

    /// Hot-swaps a new canonical `v3.0` `"sona"` `.dnam` model in-place into an owned engine
    /// (`r[sec.preview.containment]`, fail-closed JSON & `DMU1` validation).
    pub fn reload_model_in_place_owned(
        &mut self,
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
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
                    ctrl.user_metadata = serde_json::from_value(entry_val.clone())
                        .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
                }
            }
        }
        if let Some(stem_um_val) = dmu_val.get("stem_user_metadata") {
            hdr.stem_user_metadata = serde_json::from_value(stem_um_val.clone())
                .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
        }
        hdr.validate()?;
        if !self.arena_storage.reload_in_place(frame.arena_slice) {
            self.arena_storage = ArenaStorage::from_slice_owned(frame.arena_slice);
        }
        self.control_bus
            .ensure_control_capacity(frame.preamble.num_controls as usize);
        self.freeze_mutes_dry = hdr.io_config.capabilities.freeze_mutes_dry;
        self.single_input = hdr.io_config.inputs == 1;
        self.reloaded_header_flags = None;
        self.verification_status = VerificationStatus::Unsigned;
        self.verification_status_active = true;
        self.protection_profile = DnamProtectionProfile::Open;
        self.header = Some(hdr);
        self.header_active = true;
        let max_avail = Self::detect_max_tier(&frame.preamble, frame.arena_slice.len());
        let next_tier = tier.min(max_avail).min(self.max_supported_tier);
        let mut preamble = frame.preamble;
        if self
            .header
            .as_ref()
            .is_some_and(|h| h.io_config.output_mode == "mono")
        {
            preamble.flags &= !crate::container::FLAG_STEREO_OUT;
        } else {
            preamble.flags |= crate::container::FLAG_STEREO_OUT;
        }
        self.preallocate_cold_stage4_for(&preamble, next_tier);
        self.apply_validated_reload_rt(
            preamble,
            frame.arena_slice.len(),
            frame.shared_extent_slice,
            tier,
            sample_rate_hz,
        )
        .map_err(rt_err_to_dnam)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container, strip_container_to_tier};
    use crate::engine::ReloadHeaderFlags;
    use crate::math::ScalarBackend;
    use crate::provenance::OpenArenaCryptoProvider;
    use crate::schema::{
        ControlCalibration, ControlCalibrationRole, ControlKind, ControlSchemaDef, ControlUiHints,
        DnamJsonHeader, PresetDef,
    };
    use crate::stages::weights::{
        ARENA0_PIPELINE_FLAGS_OFFSET, PIPE_FLAG_STAGE1_EXT, PIPE_FLAG_STAGE4_EXT, PIPE_MAGIC_HI,
    };
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;

    fn scan_header_flags_no_alloc(header_slice: &[u8]) -> Result<ReloadHeaderFlags, DnamError> {
        scan_header_flags_rt(header_slice).map_err(rt_err_to_dnam)
    }

    #[test]
    fn test_reload_borrowed_and_owned_ext_flags_dmu1_and_errors() {
        let mut lite_header = DnamJsonHeader::default_reference();
        let lite_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let full_c =
            build_dnam_container(&mut lite_header, &lite_arena, &lite_arena, false).unwrap();
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
        let mut ext_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let flags = PIPE_MAGIC_HI | PIPE_FLAG_STAGE1_EXT | PIPE_FLAG_STAGE4_EXT;
        let flags_le = zerocopy::byteorder::little_endian::U32::new(flags);
        ext_arena[ARENA0_PIPELINE_FLAGS_OFFSET..ARENA0_PIPELINE_FLAGS_OFFSET + 4]
            .copy_from_slice(zerocopy::IntoBytes::as_bytes(&flags_le));
        let ext_c = build_dnam_container(&mut ext_header, &ext_arena, &ext_arena, false).unwrap();

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

        // Pre-allocated Ultra engine reloads into larger Ultra container with PIPE_FLAG_STAGE1_EXT | STAGE4_EXT
        engine
            .reload_model_in_place(&ext_c, Tier::Ultra, 44_100)
            .unwrap();
        assert!(engine.header().is_none());

        // Under-capacity Lite-sized engine rejects Ultra container reload without heap reallocation
        let mut small_engine =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&lite_c, Tier::Lite, 48_000).unwrap();
        assert!(matches!(
            small_engine.reload_model_in_place(&ext_c, Tier::Ultra, 44_100),
            Err(DnamError::BufferTooSmall { .. })
        ));

        // Owned reload into larger Ultra container overlays DMU1 presets & user_metadata
        let mut owned =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&lite_c, Tier::Lite, 48_000).unwrap();
        owned
            .reload_model_in_place_owned(&ext_c, Tier::Ultra, 44_100)
            .unwrap();
        let hdr = owned.header().expect("owned reload keeps header");
        assert!(hdr.presets.iter().any(|p| p.id == "reloaded_preset"));
        assert_eq!(
            hdr.user_metadata.get("author_note"),
            Some(&serde_json::json!("v2"))
        );
    }

    #[test]
    fn test_init_from_view_compiles_clamp_descriptors_when_empty() {
        let mut header = DnamJsonHeader::default_reference();
        header.controls_schema.push(ControlSchemaDef {
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
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::Mix,
                lobatto_physical_anchors: None,
                conditional_physical_anchors: alloc::vec::Vec::new(),
                tempo_sync_default_subdivision: None,
            }),
            ui_hints: ControlUiHints::default(),
            user_metadata: BTreeMap::new(),
        });
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let c = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut engine = DnamEngine::<ScalarBackend>::from_bytes(&c, Tier::Lite, 48_000).unwrap();
        let mut view = DnamContainerView::from_bytes(&c, &OpenArenaCryptoProvider).unwrap();
        view.clamp_descriptors.clear();
        view.control_records.clear();
        engine
            .init_from_view(view, Tier::Lite, Tier::Ultra, 48_000)
            .unwrap();
        assert_eq!(engine.control_bus.num_controls, 1);
        assert_eq!(
            engine.control_descriptors().len(),
            header.controls_schema.len()
        );
    }

    #[test]
    fn test_scan_header_flags_no_alloc_and_preview_containment() {
        let absent_sf = br#"{"architecture":"sona","container_kind":"dnam","io_config":{"inputs":1,"capabilities":{"freeze_mutes_dry":true,"default_blend":0.25,"wet_polarity_inverted":true},"leveling":{"auto_unity_enabled":true,"calibrated_trim_db":-3.5}},"stage0_router":{"sub_crossover_hz":150.0},"stage4_tank":{"dc_blocker_pole":0.992,"tzf_reference_tap_samples":32.0,"switch_xfade_samples":64}}"#;
        let flags = scan_header_flags_no_alloc(absent_sf).unwrap();
        assert!(flags.single_input);
        assert!(flags.freeze_mutes_dry);
        assert!(flags.supports_freeze);
        assert!((flags.default_blend - 0.25).abs() < 1e-6);
        assert!(flags.wet_polarity_inverted);
        assert!(flags.auto_unity_enabled);
        assert!((flags.calibrated_trim_db + 3.5).abs() < 1e-6);
        assert!((flags.sub_crossover_hz - 150.0).abs() < 1e-6);
        assert!((flags.dc_blocker_pole - 0.992).abs() < 1e-6);
        assert!((flags.tzf_reference_tap_samples - 32.0).abs() < 1e-6);
        assert_eq!(flags.switch_xfade_samples, 64);

        let explicit_sf = br#"{"architecture":"sona","io_config":{"inputs":2,"capabilities":{"supports_freeze":false}}}"#;
        let flags2 = scan_header_flags_no_alloc(explicit_sf).unwrap();
        assert!(!flags2.single_input);
        assert!(!flags2.supports_freeze);

        // Duplicate keys must remain sticky so trailing values cannot bypass containment.
        assert!(matches!(
            scan_header_flags_no_alloc(br#"{"architecture":"sona","architecture":"mlp"}"#),
            Err(DnamError::UnsupportedVersion { .. })
        ));
        assert!(matches!(
            scan_header_flags_no_alloc(
                br#"{"architecture":"sona","container_kind":"preview","container_kind":"dnam"}"#
            ),
            Err(DnamError::PreviewContainerRejected)
        ));
        assert!(matches!(
            scan_header_flags_no_alloc(br#"{"architecture":"sona","container_kind":"preview"}"#),
            Err(DnamError::PreviewContainerRejected)
        ));
        assert!(matches!(
            scan_header_flags_no_alloc(
                br#"{"architecture":"sona","container_kind":"pre\u0076iew"}"#
            ),
            Err(DnamError::PreviewContainerRejected)
        ));
        assert!(matches!(
            scan_header_flags_no_alloc(br#"{"architecture":"mlp"}"#),
            Err(DnamError::UnsupportedVersion { .. })
        ));
        assert!(matches!(
            scan_header_flags_no_alloc(&[0xFF, 0xFE, 0xFD]),
            Err(DnamError::InvalidJsonHeader(_))
        ));
        assert!(matches!(
            scan_header_flags_no_alloc(br#"{"architecture":"sona" trailing}"#),
            Err(DnamError::InvalidJsonHeader(_))
        ));
    }

    #[test]
    fn test_reload_rejects_preview_malformed_json_and_propagates_dsp_flags() {
        let mut base_hdr = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let base_c = build_dnam_container(&mut base_hdr, &arena, &arena, false).unwrap();

        let mut prev_hdr = DnamJsonHeader::default_reference();
        prev_hdr.container_kind = String::from("preview");
        let prev_c = build_dnam_container(&mut prev_hdr, &arena, &arena, false).unwrap();

        let mut flags_hdr = DnamJsonHeader::default_reference();
        flags_hdr.io_config.inputs = 1;
        flags_hdr.io_config.capabilities.supports_freeze = true;
        flags_hdr.io_config.capabilities.freeze_mutes_dry = true;
        flags_hdr.io_config.capabilities.default_blend = 0.4;
        flags_hdr.io_config.capabilities.wet_polarity_inverted = true;
        flags_hdr.io_config.leveling.auto_unity_enabled = true;
        flags_hdr.io_config.leveling.calibrated_trim_db = -6.020_6;
        flags_hdr.stage0_router.sub_crossover_hz = 160.0;
        flags_hdr.stage4_tank.dc_blocker_pole = 0.992;
        flags_hdr.stage4_tank.tzf_reference_tap_samples = 32.0;
        flags_hdr.stage4_tank.switch_xfade_samples = 64;
        let flags_c = build_dnam_container(&mut flags_hdr, &arena, &arena, false).unwrap();

        let mut borrowed =
            DnamEngine::<ScalarBackend>::from_bytes(&base_c, Tier::Lite, 48_000).unwrap();
        // Seed non-default verification status/profile before RT reload to ensure they reset cleanly.
        borrowed.verification_status = VerificationStatus::CreatorVerified {
            signer_id: String::from("stale_creator"),
            pubkey_ed25519: [0x11; 32],
        };
        borrowed.protection_profile = DnamProtectionProfile::DeviceAllowlisted;

        assert!(matches!(
            borrowed.reload_model_in_place(&prev_c, Tier::Lite, 48_000),
            Err(DnamError::PreviewContainerRejected)
        ));
        borrowed
            .reload_model_in_place(&flags_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(borrowed.header().is_none());
        assert_eq!(
            borrowed.verification_status(),
            &VerificationStatus::Unsigned
        );
        assert_eq!(borrowed.protection_profile(), DnamProtectionProfile::Open);
        assert!(borrowed.supports_freeze());
        assert!(borrowed.single_input);
        assert!(borrowed.freeze_mutes_dry);
        assert!((borrowed.mix_dry_gain - 0.6).abs() < 1e-5);
        assert!((borrowed.mix_wet_gain + 0.2).abs() < 1e-4);
        assert!((borrowed.stage0_router.crossover_hz - 160.0).abs() < 1e-4);
        assert!((borrowed.zoh().r_dc_host - 0.992).abs() < 1e-6);
        assert!((borrowed.zoh().tau_ref_samples_host - 32.0).abs() < 1e-6);
        assert_eq!(borrowed.zoh().switch_xfade_samples_host, 64);

        borrowed.configure_host_mix_and_leveling(0.5, 0.0, false);
        assert!((borrowed.mix_dry_gain - 0.5).abs() < 1e-5);
        assert!((borrowed.mix_wet_gain + 0.5).abs() < 1e-5);

        let mut owned =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&base_c, Tier::Lite, 48_000).unwrap();
        owned.verification_status = VerificationStatus::CreatorVerified {
            signer_id: String::from("stale_creator"),
            pubkey_ed25519: [0x22; 32],
        };
        owned.protection_profile = DnamProtectionProfile::DeviceAllowlisted;
        assert!(matches!(
            owned.reload_model_in_place_owned(&prev_c, Tier::Lite, 48_000),
            Err(DnamError::PreviewContainerRejected)
        ));
        owned
            .reload_model_in_place_owned(&flags_c, Tier::Lite, 48_000)
            .unwrap();
        assert_eq!(owned.verification_status(), &VerificationStatus::Unsigned);
        assert_eq!(owned.protection_profile(), DnamProtectionProfile::Open);

        let bad_json_container = crate::container::build_v3_open_dnam_container(
            br#"{"broken": "#,
            &[0u8; 64],
            &[0u8; 64],
            &[&arena[..crate::container::ARENA_OFFSET_LITE_END as usize]],
            None,
            0,
        )
        .unwrap();
        assert!(matches!(
            borrowed.reload_model_in_place(&bad_json_container, Tier::Lite, 48_000),
            Err(DnamError::InvalidJsonHeader(_))
        ));
        assert!(matches!(
            owned.reload_model_in_place_owned(&bad_json_container, Tier::Lite, 48_000),
            Err(DnamError::InvalidJsonHeader(_))
        ));

        let mut mono_hdr = DnamJsonHeader::default_reference();
        mono_hdr.io_config.output_mode = String::from("mono");
        let mono_c = build_dnam_container(&mut mono_hdr, &arena, &arena, false).unwrap();
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

        owned
            .reload_model_in_place_owned(&mono_c, Tier::Lite, 48_000)
            .unwrap();
        assert!(!owned.preamble.is_stereo_out());
        assert_eq!(owned.telemetry_snapshot().stereo_out, 0);
    }
}
