//! In-place hot-swap container validation and loading for [`DnamEngine`].

use crate::container::{
    CP1_MAGIC, DNAM_MAGIC, DnamContainerView, DnamPreamble, crc32c_update,
    decode_cp1_shared_stream_into_slice, parse_and_validate_dmu1_frame, parse_dcrc_record,
    validate_v2_dnam_layout_no_alloc,
};
use crate::control::ControlClampDescriptor;
use crate::engine::{DnamEngine, EngineBackend};
use crate::memory::ArenaStorage;
use crate::schema::{DnamError, DnamJsonHeader};
use crate::types::Tier;

struct ValidatedReloadFrame<'b> {
    preamble: DnamPreamble,
    arena_slice: &'b [u8],
    shared_extent_slice: &'b [u8],
    header_slice: &'b [u8],
    mutable_slice: &'b [u8],
}

fn validate_reload_frame(bytes: &[u8]) -> Result<ValidatedReloadFrame<'_>, DnamError> {
    let preamble = DnamPreamble::from_bytes(bytes)?;
    if preamble.magic != DNAM_MAGIC {
        return Err(DnamError::InvalidMagic(preamble.magic));
    }
    if preamble.is_encrypted_arena() || bytes[8] > 0 {
        return Err(DnamError::EncryptedArenaRequiresDeviceKey);
    }
    let layout = validate_v2_dnam_layout_no_alloc(bytes)?;
    let auth_slice = &bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    let has_dcrc = if let Some(dcrc) = parse_dcrc_record(auth_slice)? {
        let calc_hdr_crc = crc32c_update(0, &bytes[..layout.auth_extent.0]);
        if calc_hdr_crc != dcrc.crc_prefix {
            return Err(DnamError::Crc32cMismatch {
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
        DnamContainerView::verify_arena_prefix_crcs(&preamble, arena_slice)?;
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

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    fn apply_validated_reload(
        &mut self,
        preamble: DnamPreamble,
        arena_len: usize,
        shared_extent_slice: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let max_avail = Self::detect_max_tier(&preamble, arena_len);
        let active_tier = tier.min(max_avail).min(self.max_supported_tier);
        self.control_bus.num_contributors = 0;
        let p_ctrls = preamble.num_controls as usize;
        self.control_bus.ensure_control_capacity(p_ctrls);
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
                && let Ok(cnt) = decode_cp1_shared_stream_into_slice(
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
        self.reconfigure_for_sample_rate_with_mode(sample_rate_hz, true)?;
        Ok(())
    }

    /// Hot-swaps a new canonical `v2.0` `.dnam` model in-place with zero heap allocations.
    pub fn reload_model_in_place(
        &mut self,
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let frame = validate_reload_frame(bytes)?;
        if !self.arena_storage.reload_in_place(frame.arena_slice) {
            self.arena_storage = ArenaStorage::from_slice_or_copy(frame.arena_slice);
        }
        self.header = None;
        self.apply_validated_reload(
            frame.preamble,
            frame.arena_slice.len(),
            frame.shared_extent_slice,
            tier,
            sample_rate_hz,
        )
    }

    /// Hot-swaps a canonical `v2.0` `.dnam` model in-place into an owned engine.
    pub fn reload_model_in_place_owned(
        &mut self,
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<(), DnamError> {
        let frame = validate_reload_frame(bytes)?;
        if !self.arena_storage.reload_in_place(frame.arena_slice) {
            self.arena_storage = ArenaStorage::from_slice_owned(frame.arena_slice);
        }
        let mut parsed_header = DnamJsonHeader::from_json_slice(frame.header_slice).ok();
        if let Some(ref mut hdr) = parsed_header {
            if let Ok(dmu_val) = parse_and_validate_dmu1_frame(frame.mutable_slice) {
                if let Some(presets_val) = dmu_val.get("presets")
                    && let Ok(p) = serde_json::from_value(presets_val.clone())
                {
                    hdr.presets = p;
                }
                if let Some(um_val) = dmu_val.get("user_metadata")
                    && let Ok(um) = serde_json::from_value(um_val.clone())
                {
                    hdr.user_metadata = um;
                }
            }
            self.freeze_mutes_dry = hdr.io_config.capabilities.freeze_mutes_dry;
            self.single_input = hdr.io_config.inputs == 1;
        }
        self.header = parsed_header;
        let max_avail = Self::detect_max_tier(&frame.preamble, frame.arena_slice.len());
        let next_tier = tier.min(max_avail).min(self.max_supported_tier);
        self.preallocate_cold_stage4_for(&frame.preamble, next_tier);
        self.apply_validated_reload(
            frame.preamble,
            frame.arena_slice.len(),
            frame.shared_extent_slice,
            tier,
            sample_rate_hz,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container, strip_container_to_tier};
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
        ext_arena[ARENA0_PIPELINE_FLAGS_OFFSET..ARENA0_PIPELINE_FLAGS_OFFSET + 4]
            .copy_from_slice(&flags.to_le_bytes());
        let ext_c = build_dnam_container(&mut ext_header, &ext_arena, &ext_arena, false).unwrap();

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&lite_c, Tier::Lite, 48_000).unwrap();
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

        // Borrowed reload from Lite into larger Ultra container with PIPE_FLAG_STAGE1_EXT | STAGE4_EXT
        engine
            .reload_model_in_place(&ext_c, Tier::Ultra, 44_100)
            .unwrap();
        assert!(engine.header.is_none());

        // Owned reload into larger Ultra container overlays DMU1 presets & user_metadata
        let mut owned =
            DnamEngine::<ScalarBackend>::from_bytes_owned(&lite_c, Tier::Lite, 48_000).unwrap();
        owned
            .reload_model_in_place_owned(&ext_c, Tier::Ultra, 44_100)
            .unwrap();
        let hdr = owned.header.as_ref().expect("owned reload keeps header");
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
            conditional_min: BTreeMap::new(),
            conditional_max: BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            calibration: Some(ControlCalibration {
                role: ControlCalibrationRole::Mix,
                lobatto_physical_anchors: None,
                conditional_physical_anchors: BTreeMap::new(),
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
}
