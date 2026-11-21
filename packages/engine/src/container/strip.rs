//! Zero-re-signing canonical `v3.0` (`"sona"`) `.dnam` container tier stripper (`strip_container_to_tier`).
//!
//! Canonical Reference: `docs/specs/containers/dnam_binary_format.md`.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::schema::DnamError;
use crate::types::Tier;

use dnam_vm::bin::DnamBinView;

use super::view::map_vm_error;
use super::wire::{V3_PREAMBLE_SIZE, parse_dcrc_record, validate_v3_dnam_layout_with_max_profile};
use super::{DNAM_MAGIC, FLAG_ENCRYPTED_ARENA, crc32c_update};

const fn tier_region_name(tier_idx: usize) -> &'static str {
    match tier_idx {
        0 => "arena_lite",
        1 => "arena_full",
        2 => "arena_pro",
        _ => "arena_ultra",
    }
}

/// Strips a canonical `v3.0` (`"sona"`) `.dnam` container down to `max_tier` (`Lite = 4,608 B`,
/// `Full = 7,936 B`, `Pro = 15,872 B`, or `Ultra = 30,720 B` arenas) **without modifying the
/// preamble, directory, Region 3 JCS header, Region 4 shared extent, `auth_extent`, or optional
/// embedded `dbin_prefix`**, preserving `DAU1` / `metadata.signatures` and `DMU1` mutable metadata
/// with zero re-signing (`ref[container.dnam.tier-prefix]`, `REQ-5.6`).
pub fn strip_container_to_tier(
    container_bytes: &[u8],
    max_tier: Tier,
) -> Result<Vec<u8>, DnamError> {
    if container_bytes.len() < V3_PREAMBLE_SIZE {
        return Err(DnamError::BufferTooSmall {
            expected: V3_PREAMBLE_SIZE,
            actual: container_bytes.len(),
        });
    }
    if container_bytes[0..4] != DNAM_MAGIC {
        return Err(DnamError::SchemaValidationError(
            "strip_container_to_tier only applies to runtime .dnam containers".to_string(),
        ));
    }
    let layout = validate_v3_dnam_layout_with_max_profile(container_bytes, 1)?;
    let auth_slice =
        &container_bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    let maybe_dcrc = parse_dcrc_record(auth_slice)?;
    if let Some(dcrc) = maybe_dcrc {
        let calc_crc = crc32c_update(0, &container_bytes[..layout.auth_extent.0]);
        if calc_crc != dcrc.crc_prefix {
            return Err(DnamError::Crc32cMismatch {
                region: "header",
                expected: dcrc.crc_prefix,
                computed: calc_crc,
            });
        }
        if layout.dbin_extent.1 > 0 {
            let calc_dbin =
                crc32c_update(0, &container_bytes[layout.dbin_extent.0..layout.tier_start]);
            if calc_dbin != dcrc.crc_dbin_prefix {
                return Err(DnamError::Crc32cMismatch {
                    region: "dbin_prefix",
                    expected: dcrc.crc_dbin_prefix,
                    computed: calc_dbin,
                });
            }
        } else if dcrc.crc_dbin_prefix != 0 {
            return Err(DnamError::SchemaValidationError(
                "non-zero crc_dbin_prefix when dbin_prefix_bytes == 0".to_string(),
            ));
        }
    }

    let mut is_encrypted = maybe_dcrc.map_or(layout.protection_profile != 0, |d| {
        (d.flags_hint & FLAG_ENCRYPTED_ARENA) != 0
    });
    if !is_encrypted && layout.protection_profile == 1 && maybe_dcrc.is_some() {
        let json_slice = &container_bytes
            [layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
        let raw_val = super::wire::parse_strict_json_object_bytes(json_slice)?;
        let header: crate::schema::DnamJsonHeader = serde_json::from_value(raw_val)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        let receipt_has_nonce = header
            .metadata
            .signatures
            .as_ref()
            .and_then(|s| s.marketplace_receipt.as_ref())
            .is_some_and(|r| r.arena_nonce_96.is_some());
        if receipt_has_nonce {
            is_encrypted = true;
        }
    }
    let effective_tier = (max_tier as u8).min(layout.retained_tier);
    let tier_end = layout.tier_ends[effective_tier as usize];
    if let Some(dcrc) = maybe_dcrc
        && !is_encrypted
    {
        let mut crc = 0_u32;
        let mut prev = layout.tier_start;
        for t in 0..=(effective_tier as usize) {
            let end = layout.tier_ends[t];
            crc = crc32c_update(crc, &container_bytes[prev..end]);
            if crc != dcrc.crc_tiers[t] {
                return Err(DnamError::Crc32cMismatch {
                    region: tier_region_name(t),
                    expected: dcrc.crc_tiers[t],
                    computed: crc,
                });
            }
            prev = end;
        }
    }
    if layout.dbin_extent.1 > 0 && !is_encrypted {
        let dbin_slice = &container_bytes[layout.dbin_extent.0..tier_end];
        let _ = DnamBinView::parse(dbin_slice).map_err(map_vm_error)?;
    }

    let mutable_offset = layout.mutable_extent.0;
    let mut stripped = Vec::with_capacity(tier_end + (container_bytes.len() - mutable_offset));
    stripped.extend_from_slice(&container_bytes[..tier_end]);
    stripped.extend_from_slice(&container_bytes[mutable_offset..]);
    Ok(stripped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        ARENA_OFFSET_ULTRA_END, DcrcAuthRecordWire, DnamPreamble, build_dnam_container,
        build_dnam_container_from_dnam_bin,
    };
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
    use alloc::collections::BTreeMap;
    use alloc::string::String;
    use alloc::vec;
    use dnam_vm::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DnamBinBuildInput,
        encode_dnam_bin,
    };
    use zerocopy::{FromBytes, FromZeros};

    fn make_non_dbin_shared_header() -> DnamJsonHeader {
        let mut hdr = DnamJsonHeader::default_reference();
        hdr.controls_schema.push(ControlSchemaDef {
            id: String::from("gain"),
            name: String::from("Gain"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
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

    fn make_hybrid_ultra_container() -> Vec<u8> {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(0x000F);
        let a1 = Arena1FullDeltaWire::new_zeroed();
        let a2 = Arena2ProDeltaWire::new_zeroed();
        let a3 = Arena3UltraDeltaWire::new_zeroed();
        let dnam_bin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 64,
            dc_blocker_pole_5hz: dnam_vm::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 64,
            content_hash_prefix: [7; 8],
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
        let mut hdr = DnamJsonHeader::default_reference();
        build_dnam_container_from_dnam_bin(&mut hdr, &dnam_bin).unwrap()
    }

    fn assert_dcrc_tier_tamper_fails_closed(container: &[u8]) {
        let layout = validate_v3_dnam_layout_with_max_profile(container, 1).unwrap();
        let expected_regions = ["arena_lite", "arena_full", "arena_pro", "arena_ultra"];
        let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

        for t in 0..4 {
            // 1. Corrupting dcrc.crc_tiers[t] must fail closed with Crc32cMismatch.
            let mut bad_dcrc = container.to_vec();
            let (dcrc_wire, _) =
                DcrcAuthRecordWire::mut_from_prefix(&mut bad_dcrc[layout.auth_extent.0..]).unwrap();
            let orig = dcrc_wire.crc_tiers[t].get();
            dcrc_wire.crc_tiers[t] = zerocopy::byteorder::little_endian::U32::new(orig ^ 0xA55A);
            assert!(matches!(
                strip_container_to_tier(&bad_dcrc, tiers[t]),
                Err(DnamError::Crc32cMismatch { region, .. }) if region == expected_regions[t]
            ));

            // 2. Corrupting arena bytes inside tier `t` must fail closed with Crc32cMismatch.
            let mut bad_arena = container.to_vec();
            let tier_byte_off = if t == 0 {
                layout.tier_start
            } else {
                layout.tier_ends[t - 1]
            };
            bad_arena[tier_byte_off] ^= 0x80;
            assert!(matches!(
                strip_container_to_tier(&bad_arena, tiers[t]),
                Err(DnamError::Crc32cMismatch { region, .. }) if region == expected_regions[t]
            ));
        }
    }

    #[test]
    fn test_strip_container_error_branches_and_tier_flags() {
        let mut hdr = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut hdr, &arena, &arena, false).unwrap();

        // Truncated buffer must fail
        assert!(strip_container_to_tier(&container[..32], Tier::Lite).is_err());
        assert!(strip_container_to_tier(&container[..512], Tier::Lite).is_err());

        // Non-DNAM magic (e.g. DNPJ project container) must be rejected
        let mut proj_magic = container.clone();
        proj_magic[0..4].copy_from_slice(b"DNPJ");
        assert!(strip_container_to_tier(&proj_magic, Tier::Lite).is_err());

        // Stripping to Ultra preserves all flags and 8 SVD modes
        let ultra_stripped = strip_container_to_tier(&container, Tier::Ultra).unwrap();
        let pr_ultra = DnamPreamble::from_bytes(&ultra_stripped).unwrap();
        assert_eq!(pr_ultra.highest_tier(), Tier::Ultra);

        // Stripping to Lite clears Full/Pro/Ultra flags and sets highest_tier to Lite (4 SVD modes)
        let lite_stripped = strip_container_to_tier(&container, Tier::Lite).unwrap();
        let pr_lite = DnamPreamble::from_bytes(&lite_stripped).unwrap();
        assert_eq!(pr_lite.highest_tier(), Tier::Lite);
        assert_eq!(pr_lite.highest_tier().svd_control_modes(), 4);

        // Stripping an already-Lite container to Pro clamps cleanly to Lite
        let clamped_lite = strip_container_to_tier(&lite_stripped, Tier::Pro).unwrap();
        assert_eq!(clamped_lite, lite_stripped);

        // Corrupted header prefix must trigger header Crc32cMismatch
        let mut corrupt_hdr = container.clone();
        corrupt_hdr[crate::container::V3_HEADER_OFFSET] ^= 0x01;
        assert!(matches!(
            strip_container_to_tier(&corrupt_hdr, Tier::Lite),
            Err(DnamError::Crc32cMismatch {
                region: "header",
                ..
            })
        ));
    }

    #[test]
    fn test_strip_container_verifies_dcrc_tiers_on_shared_and_hybrid_containers() {
        // 1. Non-dbin container carrying Region 4 shared_bytes (`dbin_prefix_bytes == 0`)
        let mut shared_hdr = make_non_dbin_shared_header();
        let arena = vec![0x2Cu8; ARENA_OFFSET_ULTRA_END as usize];
        let shared_container =
            build_dnam_container(&mut shared_hdr, &arena, &arena, false).unwrap();
        let shared_layout = validate_v3_dnam_layout_with_max_profile(&shared_container, 0).unwrap();
        assert!(shared_layout.shared_extent.1 > 0);
        assert_eq!(shared_layout.dbin_extent.1, 0);
        assert_dcrc_tier_tamper_fails_closed(&shared_container);

        // 2. Hybrid container carrying embedded `.dnam-bin` (`dbin_prefix_bytes > 0`)
        let hybrid_container = make_hybrid_ultra_container();
        let hybrid_layout = validate_v3_dnam_layout_with_max_profile(&hybrid_container, 0).unwrap();
        assert_eq!(hybrid_layout.shared_extent.1, 0);
        assert!(hybrid_layout.dbin_extent.1 > 0);
        assert!(strip_container_to_tier(&hybrid_container, Tier::Lite).is_ok());
        assert_dcrc_tier_tamper_fails_closed(&hybrid_container);

        // 3. Encrypted arena (`FLAG_ENCRYPTED_ARENA`) skips plaintext tier CRC comparison against ciphertext.
        let mut enc_hdr = DnamJsonHeader::default_reference();
        let enc_arena = vec![0x7Eu8; ARENA_OFFSET_ULTRA_END as usize];
        let enc_container = build_dnam_container(&mut enc_hdr, &arena, &enc_arena, true).unwrap();
        let stripped_enc = strip_container_to_tier(&enc_container, Tier::Lite).unwrap();
        assert_eq!(
            DnamPreamble::from_bytes(&stripped_enc)
                .unwrap()
                .highest_tier(),
            Tier::Lite
        );
    }

    #[test]
    fn test_strip_container_profile1_nonce_without_dcrc_flag_and_malformed_json_fail_closed() {
        use crate::container::FLAG_HAS_SIGNATURES;
        use crate::schema::{CreatorSignature, MarketplaceReceipt, ModelSignatures};

        let mut hdr = DnamJsonHeader::default_reference();
        hdr.metadata.signatures = Some(ModelSignatures {
            immutable_header_sha256: "00".repeat(32),
            master_arena_sha256: "00".repeat(32),
            tier_arena_sha256: None,
            creator_signature: CreatorSignature {
                signer_id: String::from("creator:test"),
                pubkey_ed25519: "11".repeat(32),
                sig_ed25519: "22".repeat(64),
            },
            marketplace_receipt: Some(MarketplaceReceipt {
                marketplace_id: String::from("dnam.store"),
                license_id: String::from("lic-1"),
                buyer_hash: "33".repeat(32),
                license_tier: String::from("personal"),
                issued_at: String::from("2026-01-01T00:00:00Z"),
                buyer_arena_sha256: "44".repeat(32),
                buyer_tier_arena_sha256: None,
                encryption_mode: String::from("x25519_chacha20_arena"),
                arena_nonce_96: Some(String::from("00112233445566778899aabb")),
                ephemeral_pubkey_x25519: Some("55".repeat(32)),
                authorized_devices: Vec::new(),
                royalty_splits: Vec::new(),
                receipt_sig_ed25519: "66".repeat(64),
            }),
        });
        let plaintext_arena = vec![0x11u8; ARENA_OFFSET_ULTRA_END as usize];
        let enc_arena = vec![0x7Eu8; ARENA_OFFSET_ULTRA_END as usize];
        let mut c = build_dnam_container(&mut hdr, &plaintext_arena, &enc_arena, true).unwrap();
        let layout = validate_v3_dnam_layout_with_max_profile(&c, 1).unwrap();
        assert_eq!(layout.protection_profile, 1);

        // Clear FLAG_ENCRYPTED_ARENA from DCRC flags_hint; arena_nonce_96 in Region 3 must still engage encrypted mode.
        let (dcrc_wire, _) =
            DcrcAuthRecordWire::mut_from_prefix(&mut c[layout.auth_extent.0..]).unwrap();
        dcrc_wire.flags_hint = zerocopy::byteorder::little_endian::U16::new(FLAG_HAS_SIGNATURES);
        let stripped = strip_container_to_tier(&c, Tier::Lite).unwrap();
        assert_eq!(
            DnamPreamble::from_bytes(&stripped).unwrap().highest_tier(),
            Tier::Lite
        );

        // Malformed Region 3 JSON with matching prefix CRC under protection_profile == 1 fails closed.
        let mut bad_json_c = c.clone();
        bad_json_c[layout.header_extent.0] = b'[';
        let new_prefix_crc = crc32c_update(0, &bad_json_c[..layout.auth_extent.0]);
        let (dcrc_bad, _) =
            DcrcAuthRecordWire::mut_from_prefix(&mut bad_json_c[layout.auth_extent.0..]).unwrap();
        dcrc_bad.crc_prefix = zerocopy::byteorder::little_endian::U32::new(new_prefix_crc);
        assert!(strip_container_to_tier(&bad_json_c, Tier::Lite).is_err());
    }
}
