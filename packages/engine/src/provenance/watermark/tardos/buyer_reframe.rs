//! Buyer Container Re-Framing for `v3.0` (`"sona"`) `.dnam` Containers with Embedded `dbin_prefix`
//! (`ref[sec.sig.prefix-delegation]`, `ref[sec.watermark.tardos-code]`).
//!
//! When a buyer watermark (Mechanism A1/A2/B or Layer 2 phase coding) modifies `Arena 0..=max` inside
//! an open (`DCRC`) or split-signature (`DAU1`) `.dnam` container carrying an embedded Layer 0
//! `.dnam-bin` prefix (`dbin_prefix_bytes > 0`):
//! - Recomputes `.dnam-bin` `tier_arena_crc32c[0..=max]` and `prefix_crc32c` inside `dbin_prefix`
//!   (while keeping creator-authored `dbin_prefix` register & control bytes intact).
//! - Recomputes outer `DCRC` (`crc_prefix`, `crc_dbin_prefix`, `crc_tiers`) for `DCRC` containers.
//! - Recomputes buyer `L_buyer` prefix digests committing to `F("dbin_prefix", exact_dbin_prefix)`
//!   and signs the marketplace receipt while preserving `L_master` and `creator_signature` verbatim
//!   on `DAU1` buyer containers.

#[cfg(feature = "crypto")]
use alloc::vec;
use alloc::vec::Vec;

use dnam_vm::bin::{ARENA_TIER_BYTES, DnamBinHeaderWire};
use zerocopy::byteorder::little_endian::U32;
use zerocopy::{FromBytes, IntoBytes};

use crate::container::{
    DCRC_RECORD_SIZE, DcrcAuthRecord, V3ContainerBuildSpec, build_v3_dnam_container_with_dbin,
    crc32c_update, parse_dcrc_record, validate_v3_dnam_layout_with_max_profile,
};
#[cfg(feature = "crypto")]
use crate::container::{V3_HEADER_OFFSET, V3_PREAMBLE_SIZE};
#[cfg(feature = "crypto")]
use crate::provenance::RustCryptoProvider;
#[cfg(feature = "crypto")]
use crate::provenance::security::{
    Dau1Envelope, build_buyer_receipt_v1, build_creator_manifest_v1,
    build_plaintext_shared_v1_with_dbin, compute_creator_signed_manifest_sha256,
    compute_license_policy_sha256, compute_plaintext_prefixes_v1,
};
#[cfg(feature = "crypto")]
use crate::provenance::sha256_digest;
use crate::schema::{DnamError, reject_if};

fn recompute_dbin_prefix_crcs(
    dbin_prefix: &mut [u8],
    watermarked_arena: &[u8],
    retained_tier: usize,
) -> Result<(), DnamError> {
    reject_if(
        retained_tier > 3 || watermarked_arena.len() != ARENA_TIER_BYTES[retained_tier],
        "watermarked_arena length mismatch for retained_tier",
    )?;
    let mut hdr = DnamBinHeaderWire::read_from_prefix(dbin_prefix)
        .map_err(|_| DnamError::SchemaValidationError("invalid dbin_prefix DNBN header".into()))?
        .0;
    reject_if(&hdr.magic != b"DNBN", "invalid dbin_prefix DNBN magic")?;
    let arena0_off = hdr.arena0_offset.get() as usize;
    reject_if(
        arena0_off != dbin_prefix.len(),
        "dbin_prefix length mismatch against DNBN arena0_offset",
    )?;
    let max_present = (hdr.max_present_tier as usize).min(retained_tier);
    for t in 0..=max_present {
        let tier_end = ARENA_TIER_BYTES[t];
        hdr.tier_end_offsets[t] = U32::new((arena0_off + tier_end) as u32);
        hdr.tier_arena_crc32c[t] = U32::new(crc32c_update(0, &watermarked_arena[..tier_end]));
    }
    dbin_prefix[0..64].copy_from_slice(hdr.as_bytes());
    let prefix_crc = crc32c_update(0, &dbin_prefix[0..20]);
    let prefix_crc = crc32c_update(prefix_crc, &dbin_prefix[24..]);
    hdr.prefix_crc32c = U32::new(prefix_crc);
    dbin_prefix[0..64].copy_from_slice(hdr.as_bytes());
    Ok(())
}

fn split_arena_tier_slices(arena: &[u8], retained_tier: usize) -> Result<Vec<&[u8]>, DnamError> {
    reject_if(
        retained_tier > 3 || arena.len() != ARENA_TIER_BYTES[retained_tier],
        "arena length mismatch for retained_tier",
    )?;
    let mut slices = Vec::with_capacity(retained_tier + 1);
    let mut prev = 0_usize;
    for &end in &ARENA_TIER_BYTES[..=retained_tier] {
        slices.push(&arena[prev..end]);
        prev = end;
    }
    Ok(slices)
}

fn stamp_dcrc_after_build(
    out: &mut [u8],
    template: &DcrcAuthRecord,
    auth_off: usize,
    dbin_off: usize,
    dbin_len: usize,
    tier_slices: &[&[u8]],
) {
    let crc_prefix = crc32c_update(0, &out[..auth_off]);
    let crc_dbin_prefix = crc32c_update(0, &out[dbin_off..dbin_off + dbin_len]);
    let mut crc = 0_u32;
    let mut crc_tiers = [0_u32; 4];
    for (idx, slice) in tier_slices.iter().enumerate() {
        crc = crc32c_update(crc, slice);
        crc_tiers[idx] = crc;
    }
    let last_crc = crc_tiers[tier_slices.len() - 1];
    for slot in crc_tiers.iter_mut().skip(tier_slices.len()) {
        *slot = last_crc;
    }
    let dcrc = DcrcAuthRecord {
        crc_prefix,
        crc_tiers,
        num_controls: template.num_controls,
        flags_hint: template.flags_hint,
        tail_time_ms_bits: template.tail_time_ms_bits,
        crc_dbin_prefix,
    };
    out[auth_off..auth_off + DCRC_RECORD_SIZE].copy_from_slice(&dcrc.to_bytes());
}

/// Re-frames an unsigned `DCRC`-backed `v3.0` `.dnam` container (`dbin_prefix_bytes > 0`) after
/// applying buyer watermarks to `watermarked_arena`:
/// - Recomputes Layer 0 `.dnam-bin` `tier_arena_crc32c[0..=max]` and `prefix_crc32c`.
/// - Recomputes outer `DCRC` `crc_prefix`, `crc_dbin_prefix`, and `crc_tiers`.
pub fn reframe_buyer_container_with_dbin(
    container_bytes: &[u8],
    watermarked_arena: &[u8],
) -> Result<Vec<u8>, DnamError> {
    let layout = validate_v3_dnam_layout_with_max_profile(container_bytes, 0)?;
    reject_if(
        layout.dbin_extent.1 < 128,
        "reframe_buyer_container_with_dbin requires non-empty dbin_prefix",
    )?;
    let auth_slice =
        &container_bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    reject_if(
        !auth_slice.starts_with(b"DCRC"),
        "DCRC authentication envelope required; use reframe_buyer_dau1_container_with_dbin for DAU1",
    )?;
    let template = parse_dcrc_record(auth_slice)?.ok_or(DnamError::SchemaValidationError(
        "DCRC record missing".into(),
    ))?;
    reject_if(
        (template.flags_hint & crate::container::FLAG_ENCRYPTED_ARENA) != 0,
        "reframe_buyer_container_with_dbin requires unencrypted open arena; strip or decrypt before watermarking",
    )?;
    let retained_tier = layout.retained_tier as usize;
    let tier_slices = split_arena_tier_slices(watermarked_arena, retained_tier)?;
    let mut dbin_prefix =
        container_bytes[layout.dbin_extent.0..layout.dbin_extent.0 + layout.dbin_extent.1].to_vec();
    recompute_dbin_prefix_crcs(&mut dbin_prefix, watermarked_arena, retained_tier)?;

    let hdr_slice =
        &container_bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
    let shared_slice =
        &container_bytes[layout.shared_extent.0..layout.shared_extent.0 + layout.shared_extent.1];
    let mutable_frame = (layout.mutable_extent.1 > 0).then(|| {
        &container_bytes[layout.mutable_extent.0..layout.mutable_extent.0 + layout.mutable_extent.1]
    });

    let placeholder = [0u8; DCRC_RECORD_SIZE];
    let mut out = build_v3_dnam_container_with_dbin(&V3ContainerBuildSpec {
        protection_profile: layout.protection_profile,
        header_bytes: hdr_slice,
        shared_bytes: shared_slice,
        auth_bytes: &placeholder,
        dbin_prefix_bytes: &dbin_prefix,
        tier_slices: &tier_slices,
        mutable_frame_bytes: mutable_frame,
        auth_kind: layout.auth_kind,
    })?;
    let out_layout = validate_v3_dnam_layout_with_max_profile(&out, 0)?;
    stamp_dcrc_after_build(
        &mut out,
        &template,
        out_layout.auth_extent.0,
        out_layout.dbin_extent.0,
        out_layout.dbin_extent.1,
        &tier_slices,
    );
    Ok(out)
}

/// Re-frames a `DAU1`-authenticated (or `DCRC`-backed) `v3.0` `.dnam` container with embedded
/// `dbin_prefix` after buyer watermarking (`ref[sec.sig.prefix-delegation]`):
/// - Recomputes Layer 0 `.dnam-bin` `tier_arena_crc32c[0..=max]` and `prefix_crc32c`.
/// - Preserves creator `L_master` (`master_prefixes`) and `creator_signature` verbatim.
/// - Recomputes `L_buyer` with `F("dbin_prefix", exact_dbin_prefix)` and signs the marketplace
///   receipt (`auth_kind = 1`).
#[cfg(feature = "crypto")]
pub fn reframe_buyer_dau1_container_with_dbin(
    container_bytes: &[u8],
    watermarked_arena: &[u8],
    marketplace_seed_ed25519: &[u8; 32],
    policy_bytes: &[u8],
) -> Result<Vec<u8>, DnamError> {
    let layout = validate_v3_dnam_layout_with_max_profile(container_bytes, 1)?;
    reject_if(
        layout.dbin_extent.1 < 128,
        "reframe_buyer_dau1_container_with_dbin requires non-empty dbin_prefix",
    )?;
    let auth_slice =
        &container_bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    if auth_slice.starts_with(b"DCRC") {
        return reframe_buyer_container_with_dbin(container_bytes, watermarked_arena);
    }
    reject_if(
        layout.protection_profile != 0 || !auth_slice.starts_with(b"DAU1"),
        "reframe_buyer_dau1_container_with_dbin requires Open-profile DAU1 container",
    )?;

    let mut dau = Dau1Envelope::from_bytes(auth_slice)?;
    let retained_tier = layout.retained_tier as usize;
    let tier_slices = split_arena_tier_slices(watermarked_arena, retained_tier)?;
    let mut dbin_prefix =
        container_bytes[layout.dbin_extent.0..layout.dbin_extent.0 + layout.dbin_extent.1].to_vec();
    recompute_dbin_prefix_crcs(&mut dbin_prefix, watermarked_arena, retained_tier)?;

    let mp_pubkey = RustCryptoProvider::ed25519_pubkey_from_seed(marketplace_seed_ed25519);
    let m_tiers = retained_tier + 1;
    dau.auth_kind = 1;
    dau.marketplace_pubkey = Some(mp_pubkey);
    dau.policy_bytes = policy_bytes.to_vec();
    dau.buyer_max_tier = Some(layout.retained_tier);
    dau.buyer_prefixes = vec![[0u8; 32]; m_tiers];
    dau.marketplace_signature = Some([0u8; 64]);
    let placeholder_auth = dau.to_bytes()?;

    let hdr_slice =
        &container_bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
    let shared_slice =
        &container_bytes[layout.shared_extent.0..layout.shared_extent.0 + layout.shared_extent.1];
    let mutable_frame = (layout.mutable_extent.1 > 0).then(|| {
        &container_bytes[layout.mutable_extent.0..layout.mutable_extent.0 + layout.mutable_extent.1]
    });

    let mut out = build_v3_dnam_container_with_dbin(&V3ContainerBuildSpec {
        protection_profile: layout.protection_profile,
        header_bytes: hdr_slice,
        shared_bytes: shared_slice,
        auth_bytes: &placeholder_auth,
        dbin_prefix_bytes: &dbin_prefix,
        tier_slices: &tier_slices,
        mutable_frame_bytes: mutable_frame,
        auth_kind: 1,
    })?;
    let out_layout = validate_v3_dnam_layout_with_max_profile(&out, 1)?;
    let out_dbin_off = out_layout.dbin_extent.0;
    let out_dbin_len = out_layout.dbin_extent.1;
    let shared_framing = build_plaintext_shared_v1_with_dbin(
        &out[..V3_PREAMBLE_SIZE],
        &out[V3_PREAMBLE_SIZE..V3_HEADER_OFFSET],
        shared_slice,
        &out[out_dbin_off..out_dbin_off + out_dbin_len],
    );
    let hdr_sha = sha256_digest(hdr_slice);
    let buyer_prefixes = compute_plaintext_prefixes_v1(
        b"DNAM-BUYER-PREFIX-V1",
        &hdr_sha,
        &shared_framing,
        &tier_slices,
    )?;

    let creator_manifest =
        build_creator_manifest_v1(&dau.creator_pubkey, &hdr_sha, &dau.master_prefixes)?;
    let creator_signed_sha =
        compute_creator_signed_manifest_sha256(&creator_manifest, &dau.creator_signature);
    let policy_sha = compute_license_policy_sha256(&dau.policy_bytes);
    let buyer_receipt = build_buyer_receipt_v1(
        &mp_pubkey,
        &creator_signed_sha,
        &policy_sha,
        layout.protection_profile,
        &buyer_prefixes,
        None,
        &[],
    )?;
    let mp_sig = RustCryptoProvider::ed25519_sign(marketplace_seed_ed25519, &buyer_receipt);
    dau.buyer_prefixes = buyer_prefixes;
    dau.marketplace_signature = Some(mp_sig);
    let final_auth = dau.to_bytes()?;
    reject_if(
        final_auth.len() != placeholder_auth.len(),
        "DAU1 buyer envelope length changed unexpectedly",
    )?;
    let auth_off = out_layout.auth_extent.0;
    out[auth_off..auth_off + final_auth.len()].copy_from_slice(&final_auth);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        DnamContainerView, build_dnam_container_from_dnam_bin, extract_dnam_bin,
    };
    use crate::provenance::watermark::tardos::{
        TARDOS_FP16_SPANS, apply_tardos_watermark_7680, generate_tardos_code_7680,
    };
    use crate::provenance::{DefaultCryptoProvider, VerificationStatus};
    use crate::schema::DnamJsonHeader;
    use dnam_vm::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, DnamBinBuildInput,
        DnamBinView, encode_dnam_bin,
    };
    use zerocopy::FromZeros;

    fn build_ultra_dnam_bin() -> Vec<u8> {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(0x000F);
        let a1 = Arena1FullDeltaWire::new_zeroed();
        let a2 = Arena2ProDeltaWire::new_zeroed();
        let a3 = Arena3UltraDeltaWire::new_zeroed();
        encode_dnam_bin(&DnamBinBuildInput {
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
        .unwrap()
    }

    #[test]
    fn test_tardos_excludes_ou_to_z_ultra_and_reframes_buyer_dcrc_with_dbin() {
        let master_bin = build_ultra_dnam_bin();
        let mut hdr = DnamJsonHeader::default_reference();
        let master_container = build_dnam_container_from_dnam_bin(&mut hdr, &master_bin).unwrap();
        let layout = validate_v3_dnam_layout_with_max_profile(&master_container, 1).unwrap();
        let mut arena = master_container[layout.tier_start..layout.tier_ends[3]].to_vec();

        // Populate `ou_to_z_stage_ultra_fp16[4]` at 29_952..29_960 and confirm Tardos leaves it unchanged.
        let ou_z_bytes = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        arena[29_952..29_960].copy_from_slice(&ou_z_bytes);
        let master_dbin_prefix = master_container
            [layout.dbin_extent.0..layout.dbin_extent.0 + layout.dbin_extent.1]
            .to_vec();

        let code = generate_tardos_code_7680(&[0x42; 32], "m-reframe", "lic-reframe");
        apply_tardos_watermark_7680(&mut arena, &code, 1e-3).unwrap();
        assert_eq!(&arena[29_952..29_960], &ou_z_bytes);

        let reframed = reframe_buyer_container_with_dbin(&master_container, &arena).unwrap();
        let out_layout = validate_v3_dnam_layout_with_max_profile(&reframed, 1).unwrap();
        // Buyer dbin_prefix payload outside header CRCs is verbatim equal to master_dbin_prefix.
        assert_eq!(
            &reframed[out_layout.dbin_extent.0 + 64
                ..out_layout.dbin_extent.0 + out_layout.dbin_extent.1],
            &master_dbin_prefix[64..]
        );

        let extracted_bin = extract_dnam_bin(&reframed).unwrap();
        let bin_view = DnamBinView::parse(extracted_bin).unwrap();
        assert_eq!(
            &extracted_bin[bin_view.header.arena0_offset.get() as usize..],
            arena.as_slice()
        );

        let view =
            DnamContainerView::from_bytes(&reframed, &DefaultCryptoProvider::default()).unwrap();
        assert_eq!(view.verification_status, VerificationStatus::Unsigned);
        assert_eq!(
            TARDOS_FP16_SPANS.iter().map(|s| s.len()).sum::<usize>(),
            3_360
        );
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_reframe_buyer_dau1_with_dbin_preserves_creator_signature_and_passes_view() {
        let creator_seed = [0x15u8; 32];
        let mp_seed = [0x29u8; 32];
        let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
        let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);

        let master_bin = build_ultra_dnam_bin();
        let dbin_view = DnamBinView::parse(&master_bin).unwrap();
        let dbin_prefix_bytes = &master_bin[..dbin_view.header.arena0_offset.get() as usize];
        let arena_master = &master_bin[dbin_view.header.arena0_offset.get() as usize..];
        let tier_slices = split_arena_tier_slices(arena_master, 3).unwrap();

        let hdr = DnamJsonHeader::default_reference();
        let hdr_json =
            DnamJsonHeader::validate_and_normalize_json_str(&serde_json::to_string(&hdr).unwrap())
                .unwrap();
        let hdr_bytes = hdr_json.into_bytes();

        // 1. Creator builds auth_kind == 0 DAU1 container with dbin_prefix.
        let dummy_dau = Dau1Envelope {
            auth_kind: 0,
            creator_pubkey: creator_pk,
            creator_max_tier: 3,
            master_prefixes: vec![[0u8; 32]; 4],
            creator_signature: [0u8; 64],
            marketplace_pubkey: None,
            policy_bytes: Vec::new(),
            buyer_max_tier: None,
            buyer_prefixes: Vec::new(),
            marketplace_signature: None,
        };
        let dummy_auth = dummy_dau.to_bytes().unwrap();
        let mut master_container = build_v3_dnam_container_with_dbin(&V3ContainerBuildSpec {
            protection_profile: 0,
            header_bytes: &hdr_bytes,
            shared_bytes: &[],
            auth_bytes: &dummy_auth,
            dbin_prefix_bytes,
            tier_slices: &tier_slices,
            mutable_frame_bytes: None,
            auth_kind: 0,
        })
        .unwrap();
        let m_layout = validate_v3_dnam_layout_with_max_profile(&master_container, 1).unwrap();
        let shared_master = build_plaintext_shared_v1_with_dbin(
            &master_container[..V3_PREAMBLE_SIZE],
            &master_container[V3_PREAMBLE_SIZE..V3_HEADER_OFFSET],
            &[],
            &master_container
                [m_layout.dbin_extent.0..m_layout.dbin_extent.0 + m_layout.dbin_extent.1],
        );
        let hdr_sha = sha256_digest(&hdr_bytes);
        let l_master = compute_plaintext_prefixes_v1(
            b"DNAM-MASTER-PREFIX-V1",
            &hdr_sha,
            &shared_master,
            &tier_slices,
        )
        .unwrap();
        let creator_manifest = build_creator_manifest_v1(&creator_pk, &hdr_sha, &l_master).unwrap();
        let creator_sig = RustCryptoProvider::ed25519_sign(&creator_seed, &creator_manifest);
        let signed_master_auth = Dau1Envelope {
            master_prefixes: l_master.clone(),
            creator_signature: creator_sig,
            ..dummy_dau
        }
        .to_bytes()
        .unwrap();
        master_container[m_layout.auth_extent.0..m_layout.auth_extent.0 + signed_master_auth.len()]
            .copy_from_slice(&signed_master_auth);

        // 2. Watermark arena & buyer re-frame with auth_kind == 1.
        let mut arena_buyer = arena_master.to_vec();
        let code = generate_tardos_code_7680(&[0x77; 32], "m-dau1", "buyer-99");
        apply_tardos_watermark_7680(&mut arena_buyer, &code, 1e-3).unwrap();
        let policy = br#"{"license_id":"buyer-99","tier":"ultra"}"#;
        let buyer_container = reframe_buyer_dau1_container_with_dbin(
            &master_container,
            &arena_buyer,
            &mp_seed,
            policy,
        )
        .unwrap();

        let b_layout = validate_v3_dnam_layout_with_max_profile(&buyer_container, 1).unwrap();
        let b_dau = Dau1Envelope::from_bytes(
            &buyer_container
                [b_layout.auth_extent.0..b_layout.auth_extent.0 + b_layout.auth_extent.1],
        )
        .unwrap();
        // L_master and creator_signature are preserved verbatim.
        assert_eq!(b_dau.master_prefixes, l_master);
        assert_eq!(b_dau.creator_signature, creator_sig);

        let provider = RustCryptoProvider::new().with_marketplace_pubkey(mp_pk);
        let view = DnamContainerView::from_bytes(&buyer_container, &provider).unwrap();
        assert!(matches!(
            view.verification_status,
            VerificationStatus::MarketplaceVerified { .. }
        ));
        let extracted = extract_dnam_bin(&buyer_container).unwrap();
        let extracted_view = DnamBinView::parse(extracted).unwrap();
        assert_eq!(
            &extracted[extracted_view.header.arena0_offset.get() as usize..],
            arena_buyer.as_slice()
        );
    }

    #[test]
    fn test_reframe_buyer_container_rejects_profile1_and_encrypted_arena() {
        use crate::container::{
            DcrcAuthRecordWire, FLAG_ENCRYPTED_ARENA, build_dnam_container_with_dbin,
        };

        let master_bin = build_ultra_dnam_bin();
        let bin_view = DnamBinView::parse(&master_bin).unwrap();
        let arena0_off = bin_view.header.arena0_offset.get() as usize;
        let dbin_prefix = &master_bin[..arena0_off];
        let arena = &master_bin[arena0_off..];
        let mut hdr = DnamJsonHeader::default_reference();

        // 1. Encrypted hybrid container (`protection_profile == 1` + `FLAG_ENCRYPTED_ARENA`) fails closed.
        let enc_container =
            build_dnam_container_with_dbin(&mut hdr, arena, arena, dbin_prefix, true).unwrap();
        assert_eq!(enc_container[8], 1);
        assert!(matches!(
            reframe_buyer_container_with_dbin(&enc_container, arena),
            Err(DnamError::SchemaValidationError(_))
        ));

        // 2. Open-built hybrid container with `protection_profile == 1` alone fails closed.
        let open_container = build_dnam_container_from_dnam_bin(&mut hdr, &master_bin).unwrap();
        let mut profile1_only = open_container.clone();
        profile1_only[8] = 1;
        assert!(matches!(
            reframe_buyer_container_with_dbin(&profile1_only, arena),
            Err(DnamError::SchemaValidationError(_))
        ));

        // 3. `protection_profile == 0` with `FLAG_ENCRYPTED_ARENA` in DCRC `flags_hint` fails closed.
        let mut flag_enc_only = open_container;
        let layout = validate_v3_dnam_layout_with_max_profile(&flag_enc_only, 0).unwrap();
        let (dcrc_wire, _) =
            DcrcAuthRecordWire::mut_from_prefix(&mut flag_enc_only[layout.auth_extent.0..])
                .unwrap();
        let orig_flags = dcrc_wire.flags_hint.get();
        dcrc_wire.flags_hint =
            zerocopy::byteorder::little_endian::U16::new(orig_flags | FLAG_ENCRYPTED_ARENA);
        assert!(matches!(
            reframe_buyer_container_with_dbin(&flag_enc_only, arena),
            Err(DnamError::SchemaValidationError(_))
        ));
    }
}
