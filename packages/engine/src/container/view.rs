//! Zero-copy canonical `v3.0` (`"sona"`) container view (`DnamContainerView`).

pub mod auth;
mod sealed;

use alloc::string::ToString;
use alloc::vec::Vec;

use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::U16;

use crate::control::{
    ControlChannelRecord, ControlClampDescriptor, ControlLatentProjection, Cp1ContributorSlot,
};
use crate::memory::{AlignedBuffer, ArenaStorage, CacheLine64};
use crate::provenance::{
    CryptoProvider, DNSL_MAGIC, VerificationStatus, chacha20_xor_arena_blocks, decode_hex_array,
};
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile};
use crate::types::Tier;

use super::preamble::DnamPreamble;
use super::wire::{
    CP1_MAGIC, DnamPreambleWire, V3_PREAMBLE_SIZE, decode_cp1_shared_stream_into_slice,
    parse_and_validate_dmu1_frame, parse_dcrc_record, parse_strict_json_object_bytes,
    validate_v3_dnam_layout_with_max_profile_no_alloc,
};
use super::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DNAM_MAGIC, FLAG_ENCRYPTED_ARENA,
    FLAG_HAS_FULL_DELTA, FLAG_HAS_PRO_DELTA, FLAG_HAS_SIGNATURES, FLAG_HAS_ULTRA_DELTA,
    crc32c_update,
};

const FORBIDDEN_DCRC_RESERVED_FLAG: u16 = 1 << 12;

/// Decoded shared control descriptors, `CP1\0` contributor slots, and unified 128-byte control records.
pub type SharedControlsExtent = (
    Vec<ControlClampDescriptor>,
    Vec<Cp1ContributorSlot>,
    Vec<ControlChannelRecord>,
);

/// Decodes the shared controls extent (`ControlClampDescriptor[0..P]` + optional `CP1\0` stream)
/// for a runtime `.dnam` header.
pub fn decode_shared_controls_extent(
    shared_slice: &[u8],
    header: &DnamJsonHeader,
) -> Result<SharedControlsExtent, DnamError> {
    let p_ctrls = header.controls_schema.len();
    let desc_bytes_len = p_ctrls.checked_mul(64).ok_or_else(|| {
        DnamError::SchemaValidationError("controls_schema count overflow".to_string())
    })?;
    let starts_with_cp1 = shared_slice.len() >= 4 && shared_slice[0..4] == CP1_MAGIC;
    let (clamp_descriptors, cp1_slice) =
        if p_ctrls > 0 && shared_slice.len() >= desc_bytes_len && !starts_with_cp1 {
            let mut descs = Vec::with_capacity(p_ctrls);
            for p in 0..p_ctrls {
                let off = p * 64;
                let mut desc_bytes = [0u8; 64];
                desc_bytes.copy_from_slice(&shared_slice[off..off + 64]);
                descs.push(ControlClampDescriptor::from_bytes(&desc_bytes));
            }
            (descs, &shared_slice[desc_bytes_len..])
        } else {
            (
                header.compile_clamp_descriptors().unwrap_or_default(),
                shared_slice,
            )
        };
    let mut cp1_contributors = Vec::new();
    if cp1_slice.len() >= 8 && cp1_slice[0..4] == CP1_MAGIC {
        let count = U16::ref_from_bytes(&cp1_slice[4..6]).map_or(0, |v| v.get()) as usize;
        cp1_contributors.resize(count, Cp1ContributorSlot::zero());
        decode_cp1_shared_stream_into_slice(cp1_slice, &mut cp1_contributors)?;
    }
    let control_records = clamp_descriptors
        .iter()
        .copied()
        .map(|clamp| ControlChannelRecord {
            clamp,
            projection: ControlLatentProjection::new(),
        })
        .collect();
    Ok((clamp_descriptors, cp1_contributors, control_records))
}

/// Zero-copy or decrypted view over a validated canonical `v3.0` (`"sona"`) `.dnam` or
/// `.dnam-seal` (`b"DNSL"`) container.
#[derive(Debug)]
pub struct DnamContainerView<'a> {
    /// Logical 64-byte binary preamble summary.
    pub preamble: DnamPreamble,
    /// Parsed UTF-8 JSON header (`no_std` `serde` + `alloc`) with `DMU1` mutable fields merged.
    pub header: DnamJsonHeader,
    /// Cryptographic provenance status (`Unsigned`, `CreatorVerified`, or `MarketplaceVerified`).
    pub verification_status: VerificationStatus,
    /// Resolved container protection profile (`Open`, `DeviceAllowlisted`, or `EnclaveSealed`).
    pub protection_profile: DnamProtectionProfile,
    /// Compiled 64-byte `ControlClampDescriptor[0..P]` table from Region 4 `shared_extent`.
    pub clamp_descriptors: Vec<ControlClampDescriptor>,
    /// Decoded `CP1\0` variable-length contributor slots from Region 4 `shared_extent` (if present).
    pub cp1_contributors: Vec<Cp1ContributorSlot>,
    /// Unified 128-byte `ControlChannelRecord[0..P]` table derived from `clamp_descriptors`.
    pub control_records: Vec<ControlChannelRecord>,
    /// 64-byte aligned tensor arena storage (`Borrowed` on aligned `mmap`, or `Owned` if decrypted/copied).
    pub arena_storage: ArenaStorage<'a>,
}

impl<'a> DnamContainerView<'a> {
    /// Returns per-stem mutable user metadata merged from `DMU1`/`DME1`.
    #[inline]
    pub fn stem_user_metadata(
        &self,
    ) -> &alloc::collections::BTreeMap<alloc::string::String, serde_json::Value> {
        &self.header.stem_user_metadata
    }

    /// Full container loader for DAW/hardware hosts (rejects `"container_kind": "preview"`).
    #[inline]
    pub fn from_bytes<C: CryptoProvider>(bytes: &'a [u8], crypto: &C) -> Result<Self, DnamError> {
        Self::from_bytes_with_max_tier(bytes, crypto, Tier::Ultra)
    }

    /// Storefront browser preview loader allowing `"container_kind": "preview"` (`r[sec.preview.containment]`).
    #[inline]
    pub fn from_bytes_allowing_preview<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
    ) -> Result<Self, DnamError> {
        Self::from_bytes_with_max_tier_allowing_preview(bytes, crypto, Tier::Ultra)
    }

    /// Tier-capped storefront preview loader allowing `"container_kind": "preview"`
    /// (`r[sec.preview.containment]`).
    pub fn from_bytes_with_max_tier_allowing_preview<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        max_supported_tier: Tier,
    ) -> Result<Self, DnamError> {
        Self::from_bytes_with_max_tier_inner(bytes, crypto, true, max_supported_tier)
    }

    /// Tier-capped container loader for DAW/hardware hosts: loads, decrypts, and verifies only up to
    /// `max_supported_tier`, rejecting `"container_kind": "preview"` fail-closed (`r[sec.preview.containment]`).
    pub fn from_bytes_with_max_tier<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        max_supported_tier: Tier,
    ) -> Result<Self, DnamError> {
        Self::from_bytes_with_max_tier_inner(bytes, crypto, false, max_supported_tier)
    }

    fn from_bytes_with_max_tier_inner<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        allow_preview: bool,
        max_supported_tier: Tier,
    ) -> Result<Self, DnamError> {
        crate::math::warm_simd_feature_caches();
        if bytes.len() >= 4 && bytes[0..4] == DNSL_MAGIC {
            return Self::from_sealed_bytes(bytes, crypto, allow_preview, max_supported_tier);
        }
        Self::from_inner_dnam_bytes(bytes, crypto, false, allow_preview, max_supported_tier)
    }

    fn from_inner_dnam_bytes<C: CryptoProvider>(
        bytes: &'a [u8],
        crypto: &C,
        is_enclave_sealed_wrapper: bool,
        allow_preview: bool,
        max_supported_tier: Tier,
    ) -> Result<Self, DnamError> {
        let Ok((pw, _)) = DnamPreambleWire::read_from_prefix(bytes) else {
            return Err(DnamError::BufferTooSmall {
                expected: V3_PREAMBLE_SIZE,
                actual: bytes.len(),
            });
        };
        let magic = pw.magic;
        if magic != DNAM_MAGIC {
            return Err(DnamError::InvalidMagic(magic));
        }
        let version_major = pw.version_major.get();
        let version_minor = pw.version_minor.get();
        if version_major != CONTAINER_VERSION_MAJOR || version_minor != CONTAINER_VERSION_MINOR {
            return Err(DnamError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }

        let layout = validate_v3_dnam_layout_with_max_profile_no_alloc(bytes, 2)?;
        let auth_slice = &bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
        let maybe_dcrc = parse_dcrc_record(auth_slice)?;
        if let Some(dcrc) = maybe_dcrc {
            let calc_prefix_crc = crc32c_update(0, &bytes[..layout.auth_extent.0]);
            if calc_prefix_crc != dcrc.crc_prefix {
                return Err(DnamError::Crc32cMismatch {
                    region: "header",
                    expected: dcrc.crc_prefix,
                    computed: calc_prefix_crc,
                });
            }
            if (dcrc.flags_hint & FORBIDDEN_DCRC_RESERVED_FLAG) != 0 {
                return Err(DnamError::SchemaValidationError(
                    "reserved flags_hint bit 12 is forbidden in runtime .dnam containers"
                        .to_string(),
                ));
            }
        }

        let json_slice =
            &bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
        let raw_val = parse_strict_json_object_bytes(json_slice)?;
        let mut header: DnamJsonHeader = serde_json::from_value(raw_val)
            .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
        if !allow_preview {
            header.ensure_not_preview_for_daw_host()?;
        }

        let mutable_slice =
            &bytes[layout.mutable_extent.0..layout.mutable_extent.0 + layout.mutable_extent.1];
        if mutable_slice.starts_with(b"DME1") {
            return Err(DnamError::CryptoVerificationFailed(
                "DME1 mutable frame without DNPL context requires .dnam-seal",
            ));
        }
        let dmu_val = parse_and_validate_dmu1_frame(mutable_slice)?;
        auth::merge_mutable_doc_into_header(&mut header, &dmu_val)?;

        header.validate()?;
        let schema_ver = header.schema_version()?;
        schema_ver.ensure_matches_preamble(version_major, version_minor)?;

        let mut preamble = DnamPreamble::from_bytes(bytes)?;
        let retained_tier = Tier::from_u8(layout.retained_tier).min(max_supported_tier);
        let tier_end = layout.tier_ends[retained_tier as usize];
        let raw_arena = &bytes[layout.tier_start..tier_end];

        if retained_tier < Tier::from_u8(layout.retained_tier) {
            if retained_tier < Tier::Ultra {
                preamble.flags &= !FLAG_HAS_ULTRA_DELTA;
            }
            if retained_tier < Tier::Pro {
                preamble.flags &= !FLAG_HAS_PRO_DELTA;
            }
            if retained_tier < Tier::Full {
                preamble.flags &= !FLAG_HAS_FULL_DELTA;
            }
            preamble.tensor_arena_byte_len = raw_arena.len() as u32;
        }

        let dcrc_encrypted = maybe_dcrc.is_some_and(|d| (d.flags_hint & FLAG_ENCRYPTED_ARENA) != 0);
        let receipt_has_nonce = header
            .metadata
            .signatures
            .as_ref()
            .and_then(|s| s.marketplace_receipt.as_ref())
            .is_some_and(|r| r.arena_nonce_96.is_some());
        let needs_arena_decrypt = dcrc_encrypted
            || (!is_enclave_sealed_wrapper && layout.protection_profile == 1 && receipt_has_nonce);

        let arena_storage =
            if needs_arena_decrypt {
                let sigs = header.metadata.signatures.as_ref().ok_or(
                    DnamError::CryptoVerificationFailed(
                        "ENCRYPTED_ARENA=1 requires metadata.signatures",
                    ),
                )?;
                let receipt = sigs.marketplace_receipt.as_ref().ok_or(
                    DnamError::CryptoVerificationFailed(
                        "ENCRYPTED_ARENA=1 requires metadata.signatures.marketplace_receipt",
                    ),
                )?;
                let nonce_hex =
                    receipt
                        .arena_nonce_96
                        .as_ref()
                        .ok_or(DnamError::CryptoVerificationFailed(
                            "missing arena_nonce_96 in receipt",
                        ))?;
                let nonce_96 = decode_hex_array::<12>(nonce_hex)?;

                let target_device = if let Some(local_id) = crypto.local_device_id_hash() {
                    receipt
                        .authorized_devices
                        .iter()
                        .find(|d| d.device_id_hash == local_id)
                } else {
                    receipt.authorized_devices.first()
                }
                .ok_or(DnamError::EncryptedArenaRequiresDeviceKey)?;

                let cek = crypto.unwrap_device_cek(receipt, target_device)?;
                let mut owned = AlignedBuffer::<CacheLine64>::new_zeroed(raw_arena.len() / 64);
                owned.as_bytes_mut().copy_from_slice(raw_arena);
                chacha20_xor_arena_blocks(owned.as_bytes_mut(), &cek, &nonce_96, 0)?;
                ArenaStorage::from_owned_buffer(owned)
            } else if is_enclave_sealed_wrapper {
                ArenaStorage::from_slice_owned(raw_arena)
            } else {
                ArenaStorage::from_slice_or_copy(raw_arena)
            };

        let protection_profile = auth::resolve_protection_profile_and_policy(
            &header,
            layout.protection_profile,
            dcrc_encrypted,
            is_enclave_sealed_wrapper,
        )?;

        let plain_bytes = arena_storage.as_bytes();
        if maybe_dcrc.is_some() {
            Self::verify_arena_prefix_crcs(&preamble, plain_bytes)?;
        }

        let shared_slice =
            &bytes[layout.shared_extent.0..layout.shared_extent.0 + layout.shared_extent.1];
        let (clamp_descriptors, cp1_contributors, control_records) =
            decode_shared_controls_extent(shared_slice, &header)?;

        let verification_status = auth::verify_dau1_or_legacy_provenance(
            bytes,
            auth_slice,
            json_slice,
            shared_slice,
            &layout,
            &header,
            preamble.has_signatures(),
            retained_tier,
            plain_bytes,
            crypto,
        )?;

        if !matches!(verification_status, VerificationStatus::Unsigned) {
            preamble.flags |= FLAG_HAS_SIGNATURES;
        }
        if header.io_config.output_mode == "mono" {
            preamble.flags &= !crate::container::FLAG_STEREO_OUT;
        } else {
            preamble.flags |= crate::container::FLAG_STEREO_OUT;
        }
        preamble.num_controls = header.controls_schema.len().min(u16::MAX as usize) as u16;
        preamble.tail_time_ms = header.io_config.tail_time_ms;

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

pub use super::builder::{build_dnam_container, update_header_in_place};
pub use super::strip::strip_container_to_tier;

#[cfg(test)]
mod tests {
    use super::super::{
        ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    };
    use super::*;
    use crate::container::validate_v3_open_dnam_layout;
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{
        ControlBusConfig, DatasetProvenance, IoConfig, ModelMetadata, PresetDef, RouterConfig,
        SlimmableTiersConfig, Stage3CabinetKautzConfig, Stage4SpatialTankConfig, TierProfile,
    };
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;
    use alloc::{format, vec};
    use zerocopy::IntoBytes;
    use zerocopy::byteorder::little_endian::U32;

    fn make_base_header(name: &str) -> DnamJsonHeader {
        let mk_tier = |ch: u8, fir: u8, kautz: u8, tank: u8, rank: u8, end_off: u32| TierProfile {
            feature_bus_dim: (ch as u16) * 4,
            channels: ch,
            early_fir_taps: fir,
            kautz_modes: kautz,
            tank_lines: tank,
            head_svd_rank: rank,
            loudness_lufs: -20.5,
            output_trim_db: 0.0,
            validation_esr_db: -42.0,
            arena_end_byte_offset: end_off,
            prefix_crc32c: 0,
        };
        DnamJsonHeader {
            version: "3.0.0".to_string(),
            container_kind: "dnam".to_string(),
            architecture: "sona".to_string(),
            sample_rate_train_hz: 48_000,
            metadata: ModelMetadata {
                name: name.to_string(),
                author: Some("UnitTest".to_string()),
                description: None,
                license: Some("MIT".to_string()),
                created_at: "2026-10-23T00:00:00Z".to_string(),
                input_reference_dbfs: -20.5,
                provenance: DatasetProvenance::default(),
                distribution_policy: None,
                signatures: None,
            },
            user_metadata: BTreeMap::new(),
            stem_user_metadata: BTreeMap::new(),
            io_config: IoConfig::default(),
            slimmable_tiers: SlimmableTiersConfig {
                lite: mk_tier(16, 32, 8, 4, 4, ARENA_OFFSET_LITE_END),
                full: mk_tier(24, 48, 12, 8, 4, ARENA_OFFSET_FULL_END),
                pro: mk_tier(32, 96, 24, 12, 4, ARENA_OFFSET_PRO_END),
                ultra: mk_tier(40, 128, 32, 16, 8, ARENA_OFFSET_ULTRA_END),
            },
            controls_schema: Vec::new(),
            presets: Vec::new(),
            control_bus: ControlBusConfig::default(),
            stage0_router: RouterConfig::default(),
            stage3_cabinet: Stage3CabinetKautzConfig::default(),
            stage4_tank: Stage4SpatialTankConfig::default(),
            delay_histories: Vec::new(),
            tensor_directory: BTreeMap::new(),
        }
    }

    #[test]
    fn test_container_build_view_strip_and_in_place_update() {
        let mut hdr = make_base_header("Test");
        assert!(build_dnam_container(&mut hdr, &[0u8; 128], &[0u8; 128], false).is_err());

        let lite_arena = vec![0u8; ARENA_OFFSET_LITE_END as usize];
        let full_arena = vec![0u8; ARENA_OFFSET_FULL_END as usize];
        let pro_arena = vec![0u8; ARENA_OFFSET_PRO_END as usize];
        let ultra_arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];

        let lite_bytes = build_dnam_container(&mut hdr, &lite_arena, &lite_arena, false).unwrap();
        let full_bytes = build_dnam_container(&mut hdr, &full_arena, &full_arena, false).unwrap();
        let pro_bytes = build_dnam_container(&mut hdr, &pro_arena, &pro_arena, false).unwrap();
        let mut ultra_bytes =
            build_dnam_container(&mut hdr, &ultra_arena, &ultra_arena, false).unwrap();
        assert_eq!(ultra_bytes.len() % 64, 0);

        let crypto = DefaultCryptoProvider::default();
        assert!(DnamContainerView::from_bytes(&lite_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&full_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&pro_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&ultra_bytes, &crypto).is_ok());
        assert!(DnamContainerView::from_bytes(&ultra_bytes[..100], &crypto).is_err());

        // Test max_supported_tier prefix loading on unstripped Ultra container
        let capped_pro_view =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Pro).unwrap();
        assert_eq!(capped_pro_view.preamble.highest_tier(), Tier::Pro);
        assert_eq!(
            capped_pro_view.arena_storage.as_bytes().len(),
            ARENA_OFFSET_PRO_END as usize
        );

        // Test strip_container_to_tier
        let stripped_pro = strip_container_to_tier(&ultra_bytes, Tier::Pro).unwrap();
        let stripped_view = DnamContainerView::from_bytes(&stripped_pro, &crypto).unwrap();
        assert_eq!(stripped_view.preamble.highest_tier(), Tier::Pro);
        assert_eq!(
            stripped_view.arena_storage.as_bytes().len(),
            ARENA_OFFSET_PRO_END as usize
        );

        let layout = validate_v3_open_dnam_layout(&ultra_bytes).unwrap();
        ultra_bytes[layout.header_extent.0] ^= 0xFF;
        assert!(DnamContainerView::from_bytes(&ultra_bytes, &crypto).is_err());
        ultra_bytes[layout.header_extent.0] ^= 0xFF;

        let mut preamble = DnamPreamble::from_bytes(&ultra_bytes).unwrap();
        let arena_off = preamble.tensor_arena_offset as usize;
        let raw_arena = &ultra_bytes[arena_off..layout.tier_ends[3]];
        preamble.arena_crc32c_lite ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
        preamble.arena_crc32c_lite ^= 1;
        preamble.arena_crc32c_full ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
        preamble.arena_crc32c_full ^= 1;
        preamble.arena_crc32c_pro ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());
        preamble.arena_crc32c_pro ^= 1;
        preamble.arena_crc32c_ultra ^= 1;
        assert!(DnamContainerView::verify_arena_prefix_crcs(&preamble, raw_arena).is_err());

        assert!(update_header_in_place(&mut ultra_bytes[..64], &hdr).is_err());
        let mut huge_hdr = hdr.clone();
        for i in 0..200 {
            huge_hdr
                .user_metadata
                .insert(format!("key_{i}"), serde_json::json!("x".repeat(50)));
        }
        assert!(update_header_in_place(&mut ultra_bytes, &huge_hdr).is_err());

        // Immutable field edits must be rejected by update_header_in_place
        let mut immut_edit = hdr.clone();
        immut_edit.metadata.name = "TamperedName".to_string();
        assert!(update_header_in_place(&mut ultra_bytes, &immut_edit).is_err());

        // Valid mutable DMU1 in-place edit (presets + user_metadata)
        let mut mut_hdr = hdr.clone();
        mut_hdr
            .user_metadata
            .insert("tag".to_string(), serde_json::json!("studio"));
        mut_hdr.presets.push(PresetDef {
            id: "clean".to_string(),
            name: "Clean".to_string(),
            description: None,
            controls: BTreeMap::new(),
            user_metadata: BTreeMap::new(),
        });
        update_header_in_place(&mut ultra_bytes, &mut_hdr).unwrap();
        let updated_v = DnamContainerView::from_bytes(&ultra_bytes, &crypto).unwrap();
        assert_eq!(
            updated_v.header.user_metadata.get("tag"),
            Some(&serde_json::json!("studio"))
        );
        assert_eq!(updated_v.header.presets.len(), 1);

        // Test stripping to Full and Lite preserves DMU1 mutable edits
        let stripped_full = strip_container_to_tier(&ultra_bytes, Tier::Full).unwrap();
        let full_v = DnamContainerView::from_bytes(&stripped_full, &crypto).unwrap();
        assert_eq!(full_v.preamble.highest_tier(), Tier::Full);
        assert_eq!(full_v.header.presets.len(), 1);

        let stripped_lite = strip_container_to_tier(&ultra_bytes, Tier::Lite).unwrap();
        let lite_v = DnamContainerView::from_bytes(&stripped_lite, &crypto).unwrap();
        assert_eq!(lite_v.preamble.highest_tier(), Tier::Lite);
        assert_eq!(lite_v.header.presets.len(), 1);

        // Test capped view at Full and Lite
        let capped_full =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Full).unwrap();
        assert_eq!(capped_full.preamble.highest_tier(), Tier::Full);
        let capped_lite =
            DnamContainerView::from_bytes_with_max_tier(&ultra_bytes, &crypto, Tier::Lite).unwrap();
        assert_eq!(capped_lite.preamble.highest_tier(), Tier::Lite);

        // Error branch tests:
        // 1. Forbidden reserved flag in DCRC
        let mut bad_calib = ultra_bytes.clone();
        let flags_off = layout.auth_extent.0 + 26;
        let (flags_wire, _) = U16::mut_from_prefix(&mut bad_calib[flags_off..]).unwrap();
        *flags_wire = U16::new(flags_wire.get() | FORBIDDEN_DCRC_RESERVED_FLAG);
        assert!(DnamContainerView::from_bytes(&bad_calib, &crypto).is_err());

        // 2. Truncated buffer
        let truncated_len = ultra_bytes.len() - 100;
        assert!(DnamContainerView::from_bytes(&ultra_bytes[..truncated_len], &crypto).is_err());

        // 3. DCRC prefix CRC mismatch
        let mut bad_crc = ultra_bytes.clone();
        bad_crc[layout.auth_extent.0 + 4] ^= 0x55;
        assert!(DnamContainerView::from_bytes(&bad_crc, &crypto).is_err());

        #[cfg(feature = "crypto")]
        {
            use crate::provenance::{
                MarketplaceReceiptSigningSpec, RustCryptoProvider, seal_dnam_container,
                sign_creator_header_json, sign_marketplace_receipt_json,
            };

            let oem_seed = [21u8; 32];
            let oem_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&oem_seed);
            let dev_sk = [22u8; 32];
            let dev_pk = RustCryptoProvider::x25519_pubkey_from_secret(&dev_sk);
            let dev_id = "0102030405060708090a0b0c0d0e0f10";
            let cert = RustCryptoProvider::sign_oem_device_certificate(
                &oem_seed,
                "darkglass",
                "anagram",
                dev_id,
                &dev_pk,
            )
            .unwrap();
            let mp_seed = [1u8; 32];
            let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);
            let seal_with = |open: &[u8]| {
                seal_dnam_container(
                    open,
                    core::slice::from_ref(&cert),
                    &[("darkglass", oem_pk)],
                    &mp_seed,
                    &[2u8; 32],
                    &[4u8; 32],
                    &[3u8; 12],
                )
                .unwrap()
            };
            let sealed = seal_with(&ultra_bytes);
            let dev_crypto = RustCryptoProvider::new()
                .with_device_key(dev_id, dev_sk)
                .with_marketplace_pubkey(mp_pk);
            let owned_view = DnamContainerView::from_bytes(&sealed, &dev_crypto).unwrap();
            assert_eq!(
                owned_view.protection_profile,
                DnamProtectionProfile::EnclaveSealed
            );
            // Missing trusted marketplace_pubkey must fail closed!
            let no_mp_crypto = RustCryptoProvider::new().with_device_key(dev_id, dev_sk);
            assert!(DnamContainerView::from_bytes(&sealed, &no_mp_crypto).is_err());

            // Tier-capped .dnam-seal with omitted tier_arena_sha256 succeeds & keeps DME1 presets
            let c_json = sign_creator_header_json(
                &mut_hdr.to_json_string().unwrap(),
                &ultra_arena,
                &[9u8; 32],
                "did:web:creator",
                false,
            )
            .unwrap();
            let spec: MarketplaceReceiptSigningSpec = serde_json::from_str(
                r#"{"marketplace_id":"dnam.store","license_id":"lic-cap","buyer_hash":"0000000000000000000000000000000000000000000000000000000000000000","license_tier":"full"}"#,
            )
            .unwrap();
            let m_json =
                sign_marketplace_receipt_json(&c_json, &ultra_arena, &mp_seed, spec).unwrap();
            let mut s_hdr = DnamJsonHeader::from_json_str(&m_json).unwrap();
            let s_open =
                build_dnam_container(&mut s_hdr, &ultra_arena, &ultra_arena, false).unwrap();
            let s_seal = seal_with(&s_open);
            let cap_full =
                DnamContainerView::from_bytes_with_max_tier(&s_seal, &dev_crypto, Tier::Full)
                    .unwrap();
            assert_eq!(cap_full.preamble.highest_tier(), Tier::Full);
            assert!(matches!(
                cap_full.verification_status,
                VerificationStatus::MarketplaceVerified { .. }
            ));
            assert!(cap_full.header.presets.iter().any(|p| p.id == "clean"));
        }

        // Preview container containment: rejected by DAW loader, permitted by storefront preview loader!
        let mut prev_hdr = make_base_header("PreviewModel");
        prev_hdr.container_kind = "preview".to_string();
        prev_hdr
            .stem_user_metadata
            .insert("stem0".to_string(), serde_json::json!({"role": "dry"}));
        let prev_arena = alloc::vec![0u8; ARENA_OFFSET_LITE_END as usize];
        let prev_bytes =
            build_dnam_container(&mut prev_hdr, &prev_arena, &prev_arena, false).unwrap();
        assert!(matches!(
            DnamContainerView::from_bytes(&prev_bytes, &crypto),
            Err(DnamError::PreviewContainerRejected)
        ));
        let prev_view =
            DnamContainerView::from_bytes_allowing_preview(&prev_bytes, &crypto).unwrap();
        assert_eq!(prev_view.header.container_kind, "preview");
        assert_eq!(
            prev_view.stem_user_metadata().get("stem0"),
            Some(&serde_json::json!({"role": "dry"}))
        );

        // 4. Invalid JSON header (with recomputed DCRC prefix CRC)
        let mut bad_json = ultra_bytes.clone();
        bad_json[layout.header_extent.0..layout.header_extent.0 + 8].copy_from_slice(b"invalid_");
        let recrc = crc32c_update(0, &bad_json[..layout.auth_extent.0]);
        bad_json[layout.auth_extent.0 + 4..layout.auth_extent.0 + 8]
            .copy_from_slice(U32::new(recrc).as_bytes());
        assert!(DnamContainerView::from_bytes(&bad_json, &crypto).is_err());

        // 5. Unsupported container version (including legacy v2.0)
        let mut ver_v2 = ultra_bytes.clone();
        ver_v2[4] = 2;
        assert!(matches!(
            DnamContainerView::from_bytes(&ver_v2, &crypto),
            Err(DnamError::UnsupportedVersion { major: 2, minor: 0 })
        ));
        let mut ver_mismatch = ultra_bytes.clone();
        ver_mismatch[4] = 99;
        assert!(matches!(
            DnamContainerView::from_bytes(&ver_mismatch, &crypto),
            Err(DnamError::UnsupportedVersion {
                major: 99,
                minor: 0
            })
        ));

        // 6. DNSL sealed magic with corrupt container
        let mut corrupt_dnsl = [0u8; 128];
        corrupt_dnsl[0..4].copy_from_slice(&DNSL_MAGIC);
        assert!(DnamContainerView::from_bytes(&corrupt_dnsl, &crypto).is_err());

        // 7. v3 DAU1 auth_kind=1 enforces pinned marketplace_pubkey on CryptoProvider
        #[cfg(feature = "crypto")]
        {
            use crate::container::wire::build_v3_open_dnam_container;
            use crate::provenance::{
                Dau1Envelope, RustCryptoProvider, build_buyer_receipt_v1,
                build_creator_manifest_v1, build_plaintext_shared_v1,
                compute_creator_signed_manifest_sha256, compute_license_policy_sha256,
                compute_plaintext_prefixes_v1, sha256_digest,
            };

            let v3_hdr_json = serde_json::to_vec(&make_base_header("Dau1Market")).unwrap();
            let hdr_sha = sha256_digest(&v3_hdr_json);
            let t0 = [0u8; ARENA_OFFSET_LITE_END as usize];
            let tiers: [&[u8]; 1] = [&t0];
            let c_seed = [0x31u8; 32];
            let m_seed = [0x42u8; 32];
            let c_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&c_seed);
            let m_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&m_seed);
            let policy = b"tier=lite;offline=1".to_vec();
            let dummy_dau = Dau1Envelope {
                auth_kind: 1,
                creator_pubkey: c_pk,
                creator_max_tier: 0,
                master_prefixes: alloc::vec![[0u8; 32]],
                creator_signature: [0u8; 64],
                marketplace_pubkey: Some(m_pk),
                policy_bytes: policy.clone(),
                buyer_max_tier: Some(0),
                buyer_prefixes: alloc::vec![[0u8; 32]],
                marketplace_signature: Some([0u8; 64]),
            }
            .to_bytes()
            .unwrap();
            let probe =
                build_v3_open_dnam_container(&v3_hdr_json, &[0u8; 64], &dummy_dau, &tiers, None, 1)
                    .unwrap();
            let shared_f = build_plaintext_shared_v1(&probe[0..64], &probe[64..192], &[0u8; 64]);
            let m_pfxs = compute_plaintext_prefixes_v1(
                b"DNAM-MASTER-PREFIX-V1",
                &hdr_sha,
                &shared_f,
                &tiers,
            )
            .unwrap();
            let b_pfxs =
                compute_plaintext_prefixes_v1(b"DNAM-BUYER-PREFIX-V1", &hdr_sha, &shared_f, &tiers)
                    .unwrap();
            let c_man = build_creator_manifest_v1(&c_pk, &hdr_sha, &m_pfxs).unwrap();
            let c_sig = RustCryptoProvider::ed25519_sign(&c_seed, &c_man);
            let c_id = compute_creator_signed_manifest_sha256(&c_man, &c_sig);
            let pol_sha = compute_license_policy_sha256(&policy);
            let b_rec =
                build_buyer_receipt_v1(&m_pk, &c_id, &pol_sha, 0, &b_pfxs, None, &[]).unwrap();
            let m_sig = RustCryptoProvider::ed25519_sign(&m_seed, &b_rec);
            let real_dau = Dau1Envelope {
                auth_kind: 1,
                creator_pubkey: c_pk,
                creator_max_tier: 0,
                master_prefixes: m_pfxs,
                creator_signature: c_sig,
                marketplace_pubkey: Some(m_pk),
                policy_bytes: policy.clone(),
                buyer_max_tier: Some(0),
                buyer_prefixes: b_pfxs,
                marketplace_signature: Some(m_sig),
            }
            .to_bytes()
            .unwrap();
            let signed_v3 =
                build_v3_open_dnam_container(&v3_hdr_json, &[0u8; 64], &real_dau, &tiers, None, 1)
                    .unwrap();
            let pinned_ok = RustCryptoProvider::new().with_marketplace_pubkey(m_pk);
            let pinned_bad = RustCryptoProvider::new().with_marketplace_pubkey([0x99u8; 32]);
            let unpinned = RustCryptoProvider::new();
            assert!(DnamContainerView::from_bytes(&signed_v3, &pinned_ok).is_ok());
            assert!(matches!(
                DnamContainerView::from_bytes(&signed_v3, &pinned_bad),
                Err(DnamError::CryptoVerificationFailed(_))
            ));
            assert!(matches!(
                DnamContainerView::from_bytes(&signed_v3, &unpinned),
                Err(DnamError::CryptoVerificationFailed(_))
            ));

            // DAU1 with protection_profile != 0 must be rejected fail-closed
            let mut bad_prof = signed_v3.clone();
            bad_prof[8] = 1;
            assert!(matches!(
                DnamContainerView::from_bytes(&bad_prof, &pinned_ok),
                Err(DnamError::CryptoVerificationFailed(_))
            ));

            // Mono DAU1 without DCRC must clear FLAG_STEREO_OUT from header.io_config.output_mode
            let mut mono_hdr = make_base_header("Dau1Mono");
            mono_hdr.io_config.output_mode = alloc::string::String::from("mono");
            let mono_json = serde_json::to_vec(&mono_hdr).unwrap();
            let mono_hdr_sha = sha256_digest(&mono_json);
            let mono_probe =
                build_v3_open_dnam_container(&mono_json, &[0u8; 64], &dummy_dau, &tiers, None, 1)
                    .unwrap();
            let mono_shared =
                build_plaintext_shared_v1(&mono_probe[0..64], &mono_probe[64..192], &[0u8; 64]);
            let mono_m_pfxs = compute_plaintext_prefixes_v1(
                b"DNAM-MASTER-PREFIX-V1",
                &mono_hdr_sha,
                &mono_shared,
                &tiers,
            )
            .unwrap();
            let mono_b_pfxs = compute_plaintext_prefixes_v1(
                b"DNAM-BUYER-PREFIX-V1",
                &mono_hdr_sha,
                &mono_shared,
                &tiers,
            )
            .unwrap();
            let mono_c_man = build_creator_manifest_v1(&c_pk, &mono_hdr_sha, &mono_m_pfxs).unwrap();
            let mono_c_sig = RustCryptoProvider::ed25519_sign(&c_seed, &mono_c_man);
            let mono_c_id = compute_creator_signed_manifest_sha256(&mono_c_man, &mono_c_sig);
            let mono_b_rec =
                build_buyer_receipt_v1(&m_pk, &mono_c_id, &pol_sha, 0, &mono_b_pfxs, None, &[])
                    .unwrap();
            let mono_m_sig = RustCryptoProvider::ed25519_sign(&m_seed, &mono_b_rec);
            let mono_dau = Dau1Envelope {
                auth_kind: 1,
                creator_pubkey: c_pk,
                creator_max_tier: 0,
                master_prefixes: mono_m_pfxs,
                creator_signature: mono_c_sig,
                marketplace_pubkey: Some(m_pk),
                policy_bytes: policy.clone(),
                buyer_max_tier: Some(0),
                buyer_prefixes: mono_b_pfxs,
                marketplace_signature: Some(mono_m_sig),
            }
            .to_bytes()
            .unwrap();
            let mono_v3 =
                build_v3_open_dnam_container(&mono_json, &[0u8; 64], &mono_dau, &tiers, None, 1)
                    .unwrap();
            let mono_view = DnamContainerView::from_bytes(&mono_v3, &pinned_ok).unwrap();
            assert!(!mono_view.preamble.is_stereo_out());
        }
    }
}
