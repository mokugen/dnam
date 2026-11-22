//! Zero-copy Layer 0 `.dnam-bin` (`"DNBN"`) extraction and validation for hybrid `.dnam`
//! containers (`extract_dnam_bin`, `ref[container.dnam.vm-binary]`).

use alloc::borrow::Cow;
use alloc::vec::Vec;

use dnam_vm::VmError;
use dnam_vm::bin::DnamBinView;
use dnam_vm::control::{ControlClampDescriptor, Cp1ContributorSlot};

use super::super::wire::{
    ContainerWireError, V3_PREAMBLE_SIZE, decode_cp1_shared_stream_into_slice,
    parse_dcrc_record_rt, validate_v3_dnam_layout_no_alloc_rt,
};
use super::super::{DNAM_MAGIC, FLAG_ENCRYPTED_ARENA, crc32c_update};
use crate::schema::DnamError;

#[cold]
#[inline(never)]
pub(crate) fn map_vm_error(err: VmError) -> DnamError {
    match err {
        VmError::TruncatedPrefix => DnamError::BufferTooSmall {
            expected: 128,
            actual: 0,
        },
        VmError::InvalidMagic => DnamError::InvalidMagic([0; 4]),
        VmError::UnsupportedVersion => DnamError::UnsupportedVersion { major: 0, minor: 0 },
        VmError::PrefixCrcMismatch => DnamError::Crc32cMismatch {
            region: "dbin_prefix",
            expected: 0,
            computed: 1,
        },
        VmError::TierArenaCrcMismatch => DnamError::Crc32cMismatch {
            region: "dbin_arena",
            expected: 0,
            computed: 1,
        },
        VmError::InvalidSampleRate => DnamError::InvalidSampleRate(0),
        _ => DnamError::SchemaValidationError(alloc::format!("{err}")),
    }
}

#[cold]
#[inline(never)]
pub(crate) fn map_vm_error_rt(err: VmError) -> ContainerWireError {
    match err {
        VmError::TruncatedPrefix => ContainerWireError::BufferTooSmall {
            expected: 128,
            actual: 0,
        },
        VmError::InvalidMagic => ContainerWireError::InvalidMagic([0; 4]),
        VmError::UnsupportedVersion => {
            ContainerWireError::UnsupportedVersion { major: 0, minor: 0 }
        }
        VmError::PrefixCrcMismatch => ContainerWireError::Crc32cMismatch {
            region: "dbin_prefix",
            expected: 0,
            computed: 1,
        },
        VmError::TierArenaCrcMismatch => ContainerWireError::Crc32cMismatch {
            region: "dbin_arena",
            expected: 0,
            computed: 1,
        },
        VmError::InvalidSampleRate => ContainerWireError::InvalidSampleRate(0),
        _ => ContainerWireError::SchemaStatic("invalid embedded .dnam-bin"),
    }
}

/// Zero-allocation Layer 0 `.dnam-bin` extractor used by both [`extract_dnam_bin`] and
/// real-time [`crate::DnamEngine::reload_model_in_place`].
pub(crate) fn extract_dnam_bin_rt(
    bytes: &[u8],
) -> Result<(&[u8], DnamBinView<'_>), ContainerWireError> {
    if bytes.len() < V3_PREAMBLE_SIZE {
        return Err(ContainerWireError::BufferTooSmall {
            expected: V3_PREAMBLE_SIZE,
            actual: bytes.len(),
        });
    }
    if bytes[0..4] != DNAM_MAGIC {
        return Err(ContainerWireError::InvalidMagic([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]));
    }
    if bytes[8] != 0 {
        return Err(ContainerWireError::EncryptedArenaRequiresDeviceKey);
    }
    let layout = validate_v3_dnam_layout_no_alloc_rt(bytes)?;
    let (dbin_off, dbin_prefix_len) = layout.dbin_extent;
    if dbin_prefix_len == 0 {
        return Err(ContainerWireError::SchemaStatic(
            "container does not carry an embedded .dnam-bin prefix",
        ));
    }
    let auth_slice = &bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    if let Some(dcrc) = parse_dcrc_record_rt(auth_slice)? {
        if (dcrc.flags_hint & FLAG_ENCRYPTED_ARENA) != 0 {
            return Err(ContainerWireError::EncryptedArenaRequiresDeviceKey);
        }
        let calc_prefix = crc32c_update(0, &bytes[..layout.auth_extent.0]);
        if calc_prefix != dcrc.crc_prefix {
            return Err(ContainerWireError::Crc32cMismatch {
                region: "header",
                expected: dcrc.crc_prefix,
                computed: calc_prefix,
            });
        }
        let calc_dbin = crc32c_update(0, &bytes[dbin_off..layout.tier_start]);
        if calc_dbin != dcrc.crc_dbin_prefix {
            return Err(ContainerWireError::Crc32cMismatch {
                region: "dbin_prefix",
                expected: dcrc.crc_dbin_prefix,
                computed: calc_dbin,
            });
        }
    }

    let retained_end = layout.tier_ends[layout.retained_tier as usize];
    let dbin_slice = &bytes[dbin_off..retained_end];
    let dbin_view = DnamBinView::parse(dbin_slice).map_err(map_vm_error_rt)?;
    if (dbin_view.retained_tier as u8) != layout.retained_tier
        || (dbin_view.header.arena0_offset.get() as usize) != dbin_prefix_len
    {
        return Err(ContainerWireError::SchemaStatic(
            "embedded .dnam-bin layout disagrees with outer directory",
        ));
    }
    Ok((dbin_slice, dbin_view))
}

/// Validates an Open `v3.0` (`"sona"`) `.dnam` container with zero heap allocation and returns a
/// zero-copy slice pointing directly at the embedded Layer 0 `.dnam-bin` bytes
/// `[dbin_offset .. retained_tier_end]` (`ref[container.dnam.vm-binary]`).
pub fn extract_dnam_bin(bytes: &[u8]) -> Result<&[u8], DnamError> {
    extract_dnam_bin_rt(bytes)
        .map(|(slice, _)| slice)
        .map_err(ContainerWireError::into_dnam_error)
}

/// Cold loader helper returning borrowed Open `.dnam-bin` bytes or assembling decrypted
/// `DeviceAllowlisted` `dbin_prefix + plain_arena`.
pub(crate) fn extract_or_assemble_dnam_bin_cold<'b>(
    bytes: &'b [u8],
    view: &super::DnamContainerView<'_>,
) -> Result<Cow<'b, [u8]>, DnamError> {
    if !view.preamble.is_encrypted_arena() {
        return extract_dnam_bin(bytes).map(Cow::Borrowed);
    }
    let prefix_len = usize::try_from(view.preamble.dbin_prefix_bytes)
        .map_err(|_| DnamError::SchemaValidationError("dbin_prefix length overflow".into()))?;
    if prefix_len == 0 {
        return Err(DnamError::SchemaValidationError(
            alloc::string::String::from("container does not carry an embedded .dnam-bin prefix"),
        ));
    }
    let prefix_off = usize::try_from(view.preamble.dbin_offset)
        .map_err(|_| DnamError::SchemaValidationError("dbin_offset overflow".into()))?;
    let prefix_end = prefix_off
        .checked_add(prefix_len)
        .ok_or_else(|| DnamError::SchemaValidationError("dbin_prefix end overflow".into()))?;
    let prefix = bytes
        .get(prefix_off..prefix_end)
        .ok_or(DnamError::BufferTooSmall {
            expected: prefix_end,
            actual: bytes.len(),
        })?;
    let plain_arena = view.arena_storage.as_bytes();
    let mut assembled = Vec::with_capacity(prefix.len() + plain_arena.len());
    assembled.extend_from_slice(prefix);
    assembled.extend_from_slice(plain_arena);
    let _ = DnamBinView::parse(&assembled).map_err(map_vm_error)?;
    Ok(Cow::Owned(assembled))
}

/// Validates embedded `.dnam-bin` framing and decodes control descriptors and `CP1\0` slots
/// from its `shared_controls` section.
pub(super) fn decode_controls_from_embedded_dbin(
    dbin_prefix: &[u8],
    plain_arena: &[u8],
    expected_retained_tier: u8,
) -> Result<(Vec<ControlClampDescriptor>, Vec<Cp1ContributorSlot>), DnamError> {
    let mut assembled = Vec::with_capacity(dbin_prefix.len() + plain_arena.len());
    assembled.extend_from_slice(dbin_prefix);
    assembled.extend_from_slice(plain_arena);
    let dbin_view = DnamBinView::parse(&assembled).map_err(map_vm_error)?;
    if (dbin_view.retained_tier as u8) != expected_retained_tier
        || (dbin_view.header.arena0_offset.get() as usize) != dbin_prefix.len()
    {
        return Err(DnamError::SchemaValidationError(
            alloc::string::String::from(
                "embedded .dnam-bin prefix disagrees with container arena layout",
            ),
        ));
    }
    let clamp_descriptors = dbin_view
        .control_descriptors
        .iter()
        .map(ControlClampDescriptor::from_wire)
        .collect();
    let mut cp1_contributors = Vec::new();
    if dbin_view.cp1_overflow.len() >= 8 {
        let count = zerocopy::FromBytes::ref_from_bytes(&dbin_view.cp1_overflow[4..6])
            .map_or(0_usize, |v: &zerocopy::byteorder::little_endian::U16| {
                v.get() as usize
            });
        cp1_contributors.resize(count, Cp1ContributorSlot::zero());
        let decoded =
            decode_cp1_shared_stream_into_slice(dbin_view.cp1_overflow, &mut cp1_contributors)?;
        cp1_contributors.truncate(decoded);
    } else {
        let count = usize::from(
            dbin_view
                .arena0
                .control_bus_lite
                .cp1_dir
                .cp1_dir_count
                .get(),
        );
        if count > 0 {
            cp1_contributors.resize(count, Cp1ContributorSlot::zero());
            let decoded = dnam_vm::control::cp1::decode_cp1_contributors(
                &dbin_view.arena0.control_bus_lite,
                dbin_view.arena3.map(|w| &w.control_bus_ultra_delta),
                &[],
                dbin_view.control_descriptors.len(),
                &mut cp1_contributors,
            )
            .map_err(map_vm_error)?;
            cp1_contributors.truncate(decoded);
        }
    }
    Ok((clamp_descriptors, cp1_contributors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        DnamContainerView, build_dnam_container, build_dnam_container_from_dnam_bin,
        strip_container_to_tier,
    };
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
    use crate::types::Tier;
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use dnam_vm::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, ControlClampDescriptorWire,
        DnamBinBuildInput, encode_dnam_bin, strip_dnam_bin_to_tier,
    };
    use dnam_vm::types::Tier as VmTier;
    use proptest::prelude::*;
    use zerocopy::FromZeros;

    fn make_test_dnam_bin(tier: VmTier, ro_pole_48k: f32) -> Vec<u8> {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        let tier_mask = (1_u16 << (tier.index() + 1)) - 1;
        a0.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(tier_mask);
        let a1 = Arena1FullDeltaWire::new_zeroed();
        let a2 = Arena2ProDeltaWire::new_zeroed();
        let a3 = Arena3UltraDeltaWire::new_zeroed();
        let ctrl = ControlClampDescriptor::range(0.0, 10.0, 5.0, 2, [0.0, -1.0, 1.0]).to_wire();
        encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 64,
            dc_blocker_pole_5hz: 0.995,
            tail_time_ms: 120.0,
            ro_pole_48k,
            switch_xfade_samples: 128,
            content_hash_prefix: [1, 2, 3, 4, 5, 6, 7, 8],
            controls: &[ctrl],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: (tier >= VmTier::Full).then_some(&a1),
            arena2: (tier >= VmTier::Pro).then_some(&a2),
            arena3: (tier >= VmTier::Ultra).then_some(&a3),
        })
        .unwrap()
    }

    fn header_with_one_control(ro_pole_48k: f32) -> DnamJsonHeader {
        let mut hdr = DnamJsonHeader::default_reference();
        hdr.linear_readout.ro_pole_48k = ro_pole_48k;
        hdr.controls_schema.push(ControlSchemaDef {
            id: String::from("drive"),
            name: String::from("Drive"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(alloc::vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        });
        hdr
    }

    #[test]
    fn test_extract_dnam_bin_zero_copy_strip_and_fail_closed() {
        let ultra_bin = make_test_dnam_bin(VmTier::Ultra, 0.25);
        let mut hdr = header_with_one_control(0.25);
        let hybrid = build_dnam_container_from_dnam_bin(&mut hdr, &ultra_bin).unwrap();

        let extracted = extract_dnam_bin(&hybrid).unwrap();
        assert_eq!(extracted, ultra_bin.as_slice());

        let crypto = DefaultCryptoProvider::default();
        let view = DnamContainerView::from_bytes(&hybrid, &crypto).unwrap();
        assert_eq!(view.clamp_descriptors.len(), 1);
        assert_eq!(view.preamble.dbin_prefix_bytes, 192);

        // Stripping the hybrid .dnam to Lite yields exact strip_dnam_bin_to_tier parity.
        let stripped = strip_container_to_tier(&hybrid, Tier::Lite).unwrap();
        let stripped_bin = extract_dnam_bin(&stripped).unwrap();
        assert_eq!(
            stripped_bin,
            strip_dnam_bin_to_tier(&ultra_bin, VmTier::Lite).unwrap()
        );

        // Legacy container without dbin_prefix rejects extract_dnam_bin cleanly.
        let mut legacy_hdr = header_with_one_control(0.0);
        let arena = alloc::vec![0u8; crate::container::ARENA_OFFSET_LITE_END as usize];
        let legacy = build_dnam_container(&mut legacy_hdr, &arena, &arena, false).unwrap();
        assert!(extract_dnam_bin(&legacy).is_err());

        // Corrupting dbin_prefix byte triggers Crc32cMismatch on extract_dnam_bin.
        let mut corrupt_dbin = hybrid.clone();
        let doff = view.preamble.dbin_offset as usize;
        corrupt_dbin[doff + 40] ^= 0xFF;
        assert!(matches!(
            extract_dnam_bin(&corrupt_dbin),
            Err(DnamError::Crc32cMismatch {
                region: "dbin_prefix",
                ..
            })
        ));
    }

    #[test]
    fn test_inline_cp1_dir_decodes_into_container_view() {
        use zerocopy::IntoBytes;
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        let mut a1 = Arena1FullDeltaWire::new_zeroed();
        let mut a2 = Arena2ProDeltaWire::new_zeroed();
        let mut a3 = Arena3UltraDeltaWire::new_zeroed();
        a0.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(0x000F);
        a0.control_bus_lite.cp1_dir.cp1_dir_count = zerocopy::byteorder::little_endian::U16::new(1);
        a0.control_bus_lite.cp1_dir.entries[0] = [0, 3, 0xFF];
        let mut w = [zerocopy::byteorder::little_endian::F32::new(0.0); 34];
        w[0] = zerocopy::byteorder::little_endian::F32::new(0.25);
        a0.control_bus_lite.cp1_base[..w.as_bytes().len()].copy_from_slice(w.as_bytes());
        let ctrl = ControlClampDescriptor::range(0.0, 10.0, 5.0, 3, [0.0, -1.0, 1.0]).to_wire();
        let dbin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 64,
            dc_blocker_pole_5hz: 0.995,
            tail_time_ms: 10.0,
            ro_pole_48k: 0.0,
            switch_xfade_samples: 64,
            content_hash_prefix: [5; 8],
            controls: &[ctrl],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: Some(&a1),
            arena2: Some(&a2),
            arena3: Some(&a3),
        })
        .unwrap();
        let _ = (&mut a1, &mut a2, &mut a3);
        let mut hdr = header_with_one_control(0.0);
        hdr.controls_schema[0].lobatto_degree = Some(3);
        let hybrid = build_dnam_container_from_dnam_bin(&mut hdr, &dbin).unwrap();
        let view =
            DnamContainerView::from_bytes(&hybrid, &DefaultCryptoProvider::default()).unwrap();
        assert_eq!(view.cp1_contributors.len(), 1);
        assert_eq!(view.cp1_contributors[0].control_idx, 0);
        assert_eq!(view.cp1_contributors[0].lanes, 3);
        assert_eq!(view.cp1_contributors[0].main_base[0][0], 0.25);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_hybrid_dnam_bin_roundtrip_and_strip(
            tier_idx in 0usize..4usize,
            ro_raw in 0.0f32..=0.95f32,
            ctrl_val in 0.0f32..=10.0f32,
        ) {
            let vm_tier = VmTier::from_u8(tier_idx as u8);
            let eng_tier = Tier::from_u8(tier_idx as u8);
            let mut a0 = DnamBinBuildInput::default_arena0_lite();
            let tier_mask = (1_u16 << (vm_tier.index() + 1)) - 1;
            a0.crossbar.cell_desc.tier_flags =
                zerocopy::byteorder::little_endian::U16::new(tier_mask);
            let a1 = Arena1FullDeltaWire::new_zeroed();
            let a2 = Arena2ProDeltaWire::new_zeroed();
            let a3 = Arena3UltraDeltaWire::new_zeroed();
            let ctrl = ControlClampDescriptor::range(0.0, 10.0, ctrl_val, 2, [0.0, -1.0, 1.0]).to_wire();
            let controls_slice: &[ControlClampDescriptorWire] = &[ctrl];
            let dbin = encode_dnam_bin(&DnamBinBuildInput {
                reference_sample_rate_hz: 48_000,
                header_flags: 0,
                ext_ring_pow2_frames: 0,
                ext_ring_lines: 0,
                ext_ring_format: 0,
                declared_max_delay_48k: 64,
                dc_blocker_pole_5hz: 0.995,
                tail_time_ms: 10.0,
                ro_pole_48k: ro_raw,
                switch_xfade_samples: 64,
                content_hash_prefix: [9; 8],
                controls: controls_slice,
                cp1_overflow: &[],
                tsyn: &[],
                cext: None,
                arena0: &a0,
                arena1: (vm_tier >= VmTier::Full).then_some(&a1),
                arena2: (vm_tier >= VmTier::Pro).then_some(&a2),
                arena3: (vm_tier >= VmTier::Ultra).then_some(&a3),
            }).unwrap();
            let mut hdr = header_with_one_control(ro_raw);
            let hybrid = build_dnam_container_from_dnam_bin(&mut hdr, &dbin).unwrap();
            let sliced = extract_dnam_bin(&hybrid).unwrap();
            prop_assert_eq!(sliced, dbin.as_slice());

            let stripped_lite = strip_container_to_tier(&hybrid, Tier::Lite).unwrap();
            let extracted_lite = extract_dnam_bin(&stripped_lite).unwrap();
            let expected_lite = strip_dnam_bin_to_tier(&dbin, VmTier::Lite).unwrap();
            prop_assert_eq!(extracted_lite, expected_lite);
            let _ = eng_tier;
        }
    }
}
