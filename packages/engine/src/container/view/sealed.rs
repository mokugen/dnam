//! Hardware enclave `.dnam-seal` (`b"DNSL"`) container decryption and view construction.

use alloc::string::ToString;

use super::super::preamble::DnamPreamble;
use super::super::wire::parse_strict_json_object_bytes;
use super::super::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DNAM_MAGIC, FLAG_ENCRYPTED_ARENA,
    FLAG_HAS_FULL_DELTA, FLAG_HAS_LITE, FLAG_HAS_PRO_DELTA, FLAG_HAS_SIGNATURES,
    FLAG_HAS_ULTRA_DELTA, FLAG_STEREO_OUT,
};
use super::{DnamContainerView, auth, decode_shared_controls_extent};
use crate::memory::ArenaStorage;
use crate::provenance::{
    CryptoProvider, VerificationStatus, unseal_dnsl_up_to_tier, verify_container_provenance,
    verify_container_provenance_for_dai1_prefix,
};
use crate::schema::{DnamError, DnamJsonHeader};
use crate::types::Tier;

impl<'a> DnamContainerView<'a> {
    /// Loads, cryptographically verifies, and decrypts a canonical `v3.0` `.dnam-seal` (`b"DNSL"`)
    /// container directly via `DnslPreambleWire`, `DnslDirectoryWire`, `DSA1`, `DNPL`, `DAI1`, and
    /// `DME1` AEAD records (`ref[sec.drm.dnsl-public-framing]`).
    pub fn from_sealed_bytes<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        allow_preview: bool,
        max_supported_tier: Tier,
    ) -> Result<Self, DnamError> {
        Self::from_sealed_bytes_with_dbin(bytes, crypto, allow_preview, max_supported_tier)
            .map(|(view, _)| view)
    }

    pub(crate) fn from_sealed_bytes_with_dbin<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        allow_preview: bool,
        max_supported_tier: Tier,
    ) -> Result<(Self, alloc::vec::Vec<u8>), DnamError> {
        dnam_vm::math::warm_simd_feature_caches();
        let unsealed = unseal_dnsl_up_to_tier(bytes, crypto, max_supported_tier)?;
        let raw_val = parse_strict_json_object_bytes(&unsealed.header_json_bytes)?;
        let mut header: DnamJsonHeader = serde_json::from_value(raw_val)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        if !allow_preview {
            header.ensure_not_preview_for_daw_host()?;
        }
        if let Some(ref mut_val) = unsealed.mutable_json {
            auth::merge_mutable_doc_into_header(&mut header, mut_val)?;
        }
        header.validate()?;

        let protection_profile =
            auth::resolve_protection_profile_and_policy(&header, 2, true, true)?;
        let verification_status = if let Some(ref sigs) = header.metadata.signatures {
            let is_full_ultra =
                unsealed.arena_bytes.len() == crate::container::ARENA_OFFSET_ULTRA_END as usize;
            let has_tier_shas = sigs.tier_arena_sha256.is_some()
                && sigs
                    .marketplace_receipt
                    .as_ref()
                    .is_none_or(|r| r.buyer_tier_arena_sha256.is_some());
            if is_full_ultra || has_tier_shas {
                verify_container_provenance(&header, true, &unsealed.arena_bytes, crypto)?
            } else {
                verify_container_provenance_for_dai1_prefix(&header, crypto)?
            }
        } else {
            VerificationStatus::MarketplaceVerified {
                signer_id: header.metadata.author.clone().unwrap_or_default(),
                marketplace_id: alloc::string::String::from("dnam-marketplace"),
                license_id: alloc::string::String::from("dai1"),
                license_tier: unsealed.retained_tier.as_str().to_string(),
            }
        };

        let (clamp_descriptors, cp1_contributors) = if !unsealed.dbin_prefix.is_empty() {
            super::dbin::decode_controls_from_embedded_dbin(
                &unsealed.dbin_prefix,
                &unsealed.arena_bytes,
                unsealed.retained_tier as u8,
            )?
        } else {
            decode_shared_controls_extent(&unsealed.shared_bytes, &header)?
        };

        let mut flags = FLAG_HAS_LITE | FLAG_HAS_SIGNATURES | FLAG_ENCRYPTED_ARENA;
        if unsealed.retained_tier >= Tier::Full {
            flags |= FLAG_HAS_FULL_DELTA;
        }
        if unsealed.retained_tier >= Tier::Pro {
            flags |= FLAG_HAS_PRO_DELTA;
        }
        if unsealed.retained_tier >= Tier::Ultra {
            flags |= FLAG_HAS_ULTRA_DELTA;
        }
        if header.io_config.output_mode != "mono" {
            flags |= FLAG_STEREO_OUT;
        }
        let retained_idx = unsealed.retained_tier as usize;
        let rel0 = unsealed.tier_ends[0] as u32;
        let rel1 = if retained_idx >= 1 {
            unsealed.tier_ends[1] as u32
        } else {
            rel0
        };
        let rel2 = if retained_idx >= 2 {
            unsealed.tier_ends[2] as u32
        } else {
            rel1
        };
        let rel3 = if retained_idx >= 3 {
            unsealed.tier_ends[3] as u32
        } else {
            rel2
        };
        let to_cl = |bytes_len: u32| -> u16 { (bytes_len / 64).min(u16::MAX as u32) as u16 };
        let preamble = DnamPreamble {
            magic: DNAM_MAGIC,
            version_major: CONTAINER_VERSION_MAJOR,
            version_minor: CONTAINER_VERSION_MINOR,
            flags,
            num_controls: header.controls_schema.len().min(u16::MAX as usize) as u16,
            sample_rate_train_hz: 48_000,
            json_header_byte_len: unsealed.header_json_bytes.len() as u32,
            tensor_arena_offset: unsealed.logical_dnam_dir.tier_start.get() as u32,
            tensor_arena_byte_len: unsealed.arena_bytes.len() as u32,
            arena_lite_end_cl: to_cl(rel0),
            arena_full_end_cl: to_cl(rel1),
            arena_pro_end_cl: to_cl(rel2),
            arena_ultra_end_cl: to_cl(rel3),
            arena_crc32c_lite: 0,
            arena_crc32c_full: 0,
            arena_crc32c_pro: 0,
            arena_crc32c_ultra: 0,
            _reserved_stems: 0,
            _reserved: 0,
            tail_time_ms: header.io_config.tail_time_ms,
            header_crc32c: 0,
            crc_dbin_prefix: 0,
            dbin_offset: unsealed.logical_dnam_dir.dbin_offset.get(),
            dbin_prefix_bytes: unsealed.logical_dnam_dir.dbin_prefix_bytes.get(),
        };
        let arena_storage = ArenaStorage::from_slice_owned(&unsealed.arena_bytes);
        let assembled_dbin = if unsealed.dbin_prefix.is_empty() {
            alloc::vec::Vec::new()
        } else {
            let mut buf = alloc::vec::Vec::with_capacity(
                unsealed.dbin_prefix.len() + unsealed.arena_bytes.len(),
            );
            buf.extend_from_slice(&unsealed.dbin_prefix);
            buf.extend_from_slice(&unsealed.arena_bytes);
            buf
        };
        let view = Self {
            preamble,
            header,
            verification_status,
            protection_profile,
            clamp_descriptors,
            cp1_contributors,
            arena_storage,
        };
        Ok((view, assembled_dbin))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "crypto")]
    #[test]
    fn test_sealed_hybrid_container_decodes_embedded_dbin_controls_and_cp1() {
        use super::*;
        use crate::container::{build_dnam_container_from_dnam_bin, encode_cp1_shared_stream};
        use crate::provenance::{RustCryptoProvider, seal_dnam_container};
        use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints};
        use alloc::collections::BTreeMap;
        use alloc::string::String;
        use alloc::vec::Vec;
        use dnam_vm::bin::{
            Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
            ControlClampDescriptorWire, DnamBinBuildInput, DnamBinView, encode_dnam_bin,
        };
        use dnam_vm::control::{ControlClampDescriptor, Cp1ContributorSlot};
        use zerocopy::FromZeros;

        let mut hdr = DnamJsonHeader::default_reference();
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

        let expected_descs = hdr.compile_clamp_descriptors().unwrap();
        let wire_descs: Vec<ControlClampDescriptorWire> = expected_descs
            .iter()
            .map(ControlClampDescriptor::to_wire)
            .collect();

        let mut cp1_slot = Cp1ContributorSlot::zero();
        cp1_slot.control_idx = 0;
        cp1_slot.ctrl_kind = 0;
        cp1_slot.lanes = 2;
        cp1_slot.aux = 0xFF;
        cp1_slot.has_ultra = true;
        cp1_slot.main_base[0][0] = 0.45;
        cp1_slot.main_base[1][1] = -0.35;
        cp1_slot.emb[0] = 0.75;
        cp1_slot.emb[1] = -0.25;
        cp1_slot.v_base = [0.1, 0.2, 0.3, 0.4];
        cp1_slot.main_ultra[0][0] = 0.05;
        cp1_slot.v_ultra = [0.01, 0.02, 0.03, 0.04];
        let cp1_overflow = encode_cp1_shared_stream(&[cp1_slot]).unwrap();

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
            content_hash_prefix: [5; 8],
            controls: &wire_descs,
            cp1_overflow: &cp1_overflow,
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: Some(&a1),
            arena2: Some(&a2),
            arena3: Some(&a3),
        })
        .unwrap();
        let dbin_view = DnamBinView::parse(&dnam_bin).unwrap();
        assert_eq!(dbin_view.control_descriptors.len(), 1);
        assert_eq!(dbin_view.cp1_overflow, cp1_overflow.as_slice());

        let hybrid = build_dnam_container_from_dnam_bin(&mut hdr, &dnam_bin).unwrap();

        let oem_seed = [41u8; 32];
        let oem_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_seed);
        let dev_sk = [42u8; 32];
        let dev_pk = RustCryptoProvider::x25519_pubkey_from_secret(&dev_sk);
        let dev_id = "2122232425262728292a2b2c2d2e2f30";
        let cert = RustCryptoProvider::sign_oem_device_certificate(
            &oem_seed,
            "darkglass",
            "anagram",
            dev_id,
            &dev_pk,
        )
        .unwrap();
        let mp_seed = [51u8; 32];
        let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);
        let crypto = RustCryptoProvider::new()
            .with_device_key(dev_id, dev_sk)
            .with_marketplace_pubkey(mp_pk);

        let sealed = seal_dnam_container(
            &hybrid,
            &[cert],
            &[("darkglass", oem_pk)],
            &mp_seed,
            &[52u8; 32],
            &[54u8; 32],
            &[53u8; 12],
        )
        .unwrap();

        let view = DnamContainerView::from_bytes(&sealed, &crypto).unwrap();
        assert_eq!(view.clamp_descriptors, expected_descs);
        assert_eq!(view.cp1_contributors, alloc::vec![cp1_slot]);
        assert!(view.preamble.dbin_prefix_bytes > 0);

        let lite_view =
            DnamContainerView::from_bytes_with_max_tier(&sealed, &crypto, Tier::Lite).unwrap();
        assert_eq!(lite_view.preamble.highest_tier(), Tier::Lite);
        assert_eq!(lite_view.clamp_descriptors, expected_descs);
        assert_eq!(lite_view.cp1_contributors, alloc::vec![cp1_slot]);
    }
}
