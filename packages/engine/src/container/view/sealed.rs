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
        crate::math::warm_simd_feature_caches();
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

        let (clamp_descriptors, cp1_contributors, control_records) =
            decode_shared_controls_extent(&unsealed.shared_bytes, &header)?;

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
        };
        let arena_storage = ArenaStorage::from_slice_owned(&unsealed.arena_bytes);
        Ok(Self {
            preamble,
            header,
            verification_status,
            protection_profile,
            clamp_descriptors,
            cp1_contributors,
            control_records,
            arena_storage,
        })
    }
}
