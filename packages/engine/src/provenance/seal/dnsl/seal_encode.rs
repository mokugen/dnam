//! Canonical `v3.0` `.dnam-seal` (`b"DNSL"`) writer (`seal_dnam_container`, feature `"crypto"`).

use alloc::vec::Vec;

use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::{U32, U64};

use super::super::enforce_distribution_policy;
use super::super::hpke_base::{aead_seal_record, hpke_seal_cek_base};
use super::{
    DSA1_MAGIC, Dsa1ManifestHeaderWire, MAX_SEAL_RECIPIENTS, align64_usize,
    extract_mutable_json_bytes, merge_mutable_fields, pad_to_64,
};
use crate::container::{
    DEFAULT_DMU1_TRAILER_CAPACITY, DmueFooterWire, DnamDirectoryWire, align64_u64,
    build_dmu1_mutable_frame, encode_domain_field, parse_and_validate_dmu1_frame,
    validate_v3_dnam_layout_with_max_profile,
};
use crate::provenance::provider::RustCryptoProvider;
use crate::provenance::security::{
    Dai1Envelope, Dau1Envelope, DnplPreambleWire, DnslDirectoryWire, DnslPreambleWire,
    HpkeSlotPair, InnerDirectoryWire, build_buyer_receipt_v1, build_creator_manifest_v1,
    build_dme1_header, build_hpke_slot_binding_v1, build_mutable_content_aad_v1,
    build_mutable_hpke_binding_v1, build_outer_manifest_v1, build_plaintext_shared_v1,
    build_prewrap_plan_v1, build_record_aad_v1, build_record_nonce_v1,
    compute_ciphertext_prefixes_v1, compute_creator_signed_manifest_sha256,
    compute_encrypted_structural_metadata_sha256, compute_license_policy_sha256,
    compute_logical_metadata_sha256, compute_mutable_wrap_sha256,
    compute_outer_public_metadata_sha256, compute_plaintext_prefixes_v1,
    compute_record_context_sha256, compute_wrap_plan_sha256, encode_hpke_slot_vector,
    encode_prefix_list,
};
use crate::provenance::sha256::sha256_digest;
use crate::provenance::verify::verify_oem_device_certificate;
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile, OemDeviceCertificate};

struct LogicalDnplFraming {
    dnpl_preamble_bytes: Vec<u8>,
    logical_prefix: Vec<u8>,
    logical_meta_sha: [u8; 32],
    hdr_sha: [u8; 32],
    shared_framing: Vec<u8>,
}

struct InnerSealedResult {
    outer_rec0_plain: Vec<u8>,
    inner_ct_tier_padded: Vec<Vec<u8>>,
}

struct SealKeys<'a> {
    mp_pubkey: [u8; 32],
    marketplace_seed_ed25519: &'a [u8; 32],
    ephemeral_secret_x25519: &'a [u8; 32],
    issuance_id: [u8; 16],
    cek_outer: &'a [u8; 32],
    cek_inner: [u8; 32],
    cek_mutable: [u8; 32],
}

fn derive_secret_32(tag: &[u8], a: &[u8], b: &[u8], c: &[u8], idx: u32) -> [u8; 32] {
    let mut msg = Vec::with_capacity(64 + a.len() + b.len() + c.len());
    msg.extend_from_slice(&encode_domain_field("tag", tag));
    msg.extend_from_slice(&encode_domain_field("a", a));
    msg.extend_from_slice(&encode_domain_field("b", b));
    msg.extend_from_slice(&encode_domain_field("c", c));
    msg.extend_from_slice(&encode_domain_field("idx", U32::new(idx).as_bytes()));
    sha256_digest(&msg)
}

type RecipientKeyLists = (Vec<[u8; 32]>, Vec<[u8; 32]>);

fn resolve_sorted_recipients(
    recipient_certs: &[OemDeviceCertificate],
    trusted_oem_pubkeys: &[(&str, [u8; 32])],
    inner_header: &DnamJsonHeader,
) -> Result<RecipientKeyLists, DnamError> {
    if let Some(ref policy) = inner_header.metadata.distribution_policy {
        for cert in recipient_certs {
            if !policy.is_oem_device_allowed(&cert.oem_vendor, &cert.model_code) {
                return Err(DnamError::CryptoVerificationFailed(
                    "recipient OemDeviceCertificate vendor/model is not permitted by creator distribution_policy",
                ));
            }
        }
    }

    let verify_provider = RustCryptoProvider::new();
    let mut recipients: Vec<([u8; 32], [u8; 32])> = Vec::with_capacity(recipient_certs.len());
    for cert in recipient_certs {
        let (_, oem_pk) = trusted_oem_pubkeys
            .iter()
            .find(|(vendor, _)| vendor.eq_ignore_ascii_case(&cert.oem_vendor))
            .ok_or(DnamError::CryptoVerificationFailed(
                "unknown or untrusted OEM vendor in OemDeviceCertificate",
            ))?;
        let pk_x25519 = verify_oem_device_certificate(cert, oem_pk, &verify_provider)?;
        recipients.push((sha256_digest(&pk_x25519), pk_x25519));
    }
    recipients.sort_by_key(|a| a.0);
    for pair in recipients.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(DnamError::CryptoVerificationFailed(
                "duplicate recipient X25519 key in .dnam-seal recipients",
            ));
        }
    }
    Ok((
        recipients.iter().map(|r| r.0).collect(),
        recipients.iter().map(|r| r.1).collect(),
    ))
}

fn build_logical_dnpl_framing(
    header_slice: &[u8],
    shared_slice: &[u8],
    auth_slice: &[u8],
    tier_slices: &[&[u8]],
    max_tier: u8,
) -> Result<LogicalDnplFraming, DnamError> {
    let dnpl_preamble = DnplPreambleWire::new(2, 1, max_tier)?;
    let dnpl_preamble_bytes = dnpl_preamble.to_bytes();
    let logical_header_off = 192u64;
    let logical_header_len = header_slice.len() as u64;
    let logical_shared_off = align64_u64(logical_header_off + logical_header_len).ok_or(
        DnamError::CryptoVerificationFailed("logical offset overflow"),
    )?;
    let logical_shared_len = shared_slice.len() as u64;
    let logical_auth_off = align64_u64(logical_shared_off + logical_shared_len).ok_or(
        DnamError::CryptoVerificationFailed("logical offset overflow"),
    )?;
    let logical_auth_len = auth_slice.len() as u64;
    let logical_tier_start = align64_u64(logical_auth_off + logical_auth_len).ok_or(
        DnamError::CryptoVerificationFailed("logical offset overflow"),
    )?;
    let mut logical_tier_ends = [U64::new(0); 4];
    let mut l_cur = logical_tier_start;
    for (t, t_slice) in tier_slices.iter().enumerate() {
        l_cur += t_slice.len() as u64;
        logical_tier_ends[t] = U64::new(l_cur);
    }
    let logical_dir = DnamDirectoryWire {
        immutable_header_offset: U64::new(logical_header_off),
        immutable_header_length: U64::new(logical_header_len),
        shared_binary_offset: U64::new(logical_shared_off),
        shared_binary_length: U64::new(logical_shared_len),
        authentication_envelope_offset: U64::new(logical_auth_off),
        authentication_envelope_length: U64::new(logical_auth_len),
        reserved_offset: U64::new(0),
        reserved_capacity: U64::new(0),
        tier_start: U64::new(logical_tier_start),
        original_tier_ends: logical_tier_ends,
        reserved_tail: [0u8; 24],
    };
    let mut logical_prefix = Vec::with_capacity(align64_usize(192 + header_slice.len()));
    logical_prefix.extend_from_slice(&dnpl_preamble_bytes);
    logical_prefix.extend_from_slice(logical_dir.as_bytes());
    logical_prefix.extend_from_slice(header_slice);
    pad_to_64(&mut logical_prefix);
    let logical_meta_sha = compute_logical_metadata_sha256(&logical_prefix);
    let hdr_sha = sha256_digest(header_slice);
    let shared_framing =
        build_plaintext_shared_v1(&dnpl_preamble_bytes, logical_dir.as_bytes(), shared_slice);

    Ok(LogicalDnplFraming {
        dnpl_preamble_bytes: dnpl_preamble_bytes.to_vec(),
        logical_prefix,
        logical_meta_sha,
        hdr_sha,
        shared_framing,
    })
}

struct SealRecipients<'a> {
    key_ids: &'a [[u8; 32]],
    pubkeys: &'a [[u8; 32]],
}

fn seal_inner_dnpl_layer(
    logical: &LogicalDnplFraming,
    shared_slice: &[u8],
    auth_slice: &[u8],
    tier_slices: &[&[u8]],
    max_tier: u8,
    recipients: &SealRecipients<'_>,
    keys: &SealKeys<'_>,
) -> Result<InnerSealedResult, DnamError> {
    let rec_key_ids = recipients.key_ids;
    let rec_pubkeys = recipients.pubkeys;
    let m_tiers = tier_slices.len();
    let (creator_pubkey, creator_max_tier, master_prefixes, creator_signature) =
        if auth_slice.starts_with(b"DAU1") {
            let dau = Dau1Envelope::from_bytes(auth_slice)?;
            (
                dau.creator_pubkey,
                dau.creator_max_tier,
                dau.master_prefixes,
                dau.creator_signature,
            )
        } else {
            let pfxs = compute_plaintext_prefixes_v1(
                b"DNAM-MASTER-PREFIX-V1",
                &logical.hdr_sha,
                &logical.shared_framing,
                tier_slices,
            )?;
            let man = build_creator_manifest_v1(&keys.mp_pubkey, &logical.hdr_sha, &pfxs)?;
            let sig = RustCryptoProvider::ed25519_sign(keys.marketplace_seed_ed25519, &man);
            (keys.mp_pubkey, max_tier, pfxs, sig)
        };

    let creator_manifest =
        build_creator_manifest_v1(&creator_pubkey, &logical.hdr_sha, &master_prefixes)?;
    let creator_signed_sha =
        compute_creator_signed_manifest_sha256(&creator_manifest, &creator_signature);
    let buyer_prefixes = compute_plaintext_prefixes_v1(
        b"DNAM-BUYER-PREFIX-V1",
        &logical.hdr_sha,
        &logical.shared_framing,
        tier_slices,
    )?;
    let policy_bytes = b"{\"profile\":\"enclave_sealed\"}".to_vec();
    let policy_sha = compute_license_policy_sha256(&policy_bytes);

    let inner_plan_off = 192 + logical.logical_prefix.len();
    let inner_ct_lens: Vec<usize> = core::iter::once(shared_slice.len() + 16)
        .chain(tier_slices.iter().map(|s| s.len() + 16))
        .collect();
    let mut d_off = 64u64;
    let dummy_recs: Vec<(u64, u64)> = inner_ct_lens
        .iter()
        .map(|&l| {
            let rec = (d_off, l as u64);
            d_off += align64_usize(l) as u64;
            rec
        })
        .collect();
    let dummy_plan = build_prewrap_plan_v1(
        &keys.issuance_id,
        1,
        2,
        max_tier,
        &[0u8; 32],
        rec_key_ids,
        &dummy_recs,
    )?;
    let inner_slots_off = align64_usize(inner_plan_off + dummy_plan.len());
    let slots_len = 4 + 80 * rec_key_ids.len();
    let inner_auth_off = align64_usize(inner_slots_off + slots_len);
    let dai1_len = 209 + 33 * ((creator_max_tier as usize) + 1) + 66 * m_tiers + policy_bytes.len();
    let inner_rec0_off = align64_usize(inner_auth_off + dai1_len);
    let mut inner_recs: Vec<(u64, u64)> = Vec::with_capacity(1 + m_tiers);
    inner_recs.push((inner_rec0_off as u64, inner_ct_lens[0] as u64));
    let mut i_cursor = align64_usize(inner_rec0_off + inner_ct_lens[0]);
    let mut inner_tier_ends = [U64::new(0); 4];
    for (t, &ct_len) in inner_ct_lens[1..].iter().enumerate() {
        inner_recs.push((i_cursor as u64, ct_len as u64));
        i_cursor = align64_usize(i_cursor + ct_len);
        inner_tier_ends[t] = U64::new(i_cursor as u64);
    }

    let inner_dir = InnerDirectoryWire {
        plan_offset: U64::new(inner_plan_off as u64),
        plan_length: U64::new(dummy_plan.len() as u64),
        slots_offset: U64::new(inner_slots_off as u64),
        slots_length: U64::new(slots_len as u64),
        authentication_offset: U64::new(inner_auth_off as u64),
        authentication_length: U64::new(dai1_len as u64),
        shared_record_offset: U64::new(inner_rec0_off as u64),
        shared_ciphertext_length: U64::new(inner_ct_lens[0] as u64),
        original_tier_ends: inner_tier_ends,
        logical_metadata_sha256: logical.logical_meta_sha,
    };
    let inner_struct_sha = compute_encrypted_structural_metadata_sha256(
        1,
        &logical.dnpl_preamble_bytes,
        inner_dir.as_bytes(),
    )?;
    let inner_plan = build_prewrap_plan_v1(
        &keys.issuance_id,
        1,
        2,
        max_tier,
        &inner_struct_sha,
        rec_key_ids,
        &inner_recs,
    )?;
    let inner_wrap_sha = compute_wrap_plan_sha256(&inner_plan);
    let inner_slots: Vec<HpkeSlotPair> = rec_key_ids
        .iter()
        .zip(rec_pubkeys.iter())
        .enumerate()
        .map(|(i, (kid, pk))| {
            let slot_esk = derive_secret_32(
                b"DNAM-DNSL-HPKE-INNER-ESK",
                keys.ephemeral_secret_x25519,
                kid,
                &inner_wrap_sha,
                i as u32,
            );
            let info =
                build_hpke_slot_binding_v1(b"DNAM-HPKE-INFO-V1", &inner_wrap_sha, i as u32, kid)?;
            let aad =
                build_hpke_slot_binding_v1(b"DNAM-HPKE-AAD-V1", &inner_wrap_sha, i as u32, kid)?;
            hpke_seal_cek_base(&slot_esk, pk, &info, &aad, &keys.cek_inner)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let inner_slots_vec = encode_hpke_slot_vector(&inner_slots)?;
    let inner_rec_ctx_sha = compute_record_context_sha256(&inner_wrap_sha, &inner_slots_vec);

    let mut inner_padded_cts: Vec<Vec<u8>> = Vec::with_capacity(1 + m_tiers);
    for (idx, pt) in core::iter::once(shared_slice)
        .chain(tier_slices.iter().copied())
        .enumerate()
    {
        let ct_len = (pt.len() + 16) as u64;
        let nonce = build_record_nonce_v1(idx as u32)?;
        let aad = build_record_aad_v1(&inner_rec_ctx_sha, idx as u32, ct_len)?;
        let mut ct = aead_seal_record(&keys.cek_inner, &nonce, &aad, pt)?;
        pad_to_64(&mut ct);
        inner_padded_cts.push(ct);
    }
    let inner_ct_tier_padded = inner_padded_cts[1..].to_vec();
    let inner_ct_tier_refs: Vec<&[u8]> = inner_ct_tier_padded.iter().map(Vec::as_slice).collect();
    let inner_ct_prefixes = compute_ciphertext_prefixes_v1(
        &inner_rec_ctx_sha,
        &inner_padded_cts[0],
        &inner_ct_tier_refs,
    )?;

    let buyer_msg = build_buyer_receipt_v1(
        &keys.mp_pubkey,
        &creator_signed_sha,
        &policy_sha,
        2,
        &buyer_prefixes,
        Some(&inner_rec_ctx_sha),
        &inner_ct_prefixes,
    )?;
    let dai1_mp_sig = RustCryptoProvider::ed25519_sign(keys.marketplace_seed_ed25519, &buyer_msg);
    let dai1 = Dai1Envelope {
        creator_pubkey,
        creator_max_tier,
        master_prefixes,
        creator_signature,
        marketplace_pubkey: keys.mp_pubkey,
        policy_bytes,
        buyer_max_tier: max_tier,
        buyer_prefixes,
        inner_ciphertext_prefixes: inner_ct_prefixes,
        marketplace_signature: dai1_mp_sig,
    };
    let dai1_bytes = dai1.to_bytes()?;
    debug_assert_eq!(dai1_bytes.len(), dai1_len);

    let mut outer_rec0_plain =
        Vec::with_capacity(align64_usize(inner_rec0_off + inner_padded_cts[0].len()));
    outer_rec0_plain.extend_from_slice(&logical.dnpl_preamble_bytes);
    outer_rec0_plain.extend_from_slice(inner_dir.as_bytes());
    outer_rec0_plain.extend_from_slice(&logical.logical_prefix);
    pad_to_64(&mut outer_rec0_plain);
    outer_rec0_plain.extend_from_slice(&inner_plan);
    pad_to_64(&mut outer_rec0_plain);
    outer_rec0_plain.extend_from_slice(&inner_slots_vec);
    pad_to_64(&mut outer_rec0_plain);
    outer_rec0_plain.extend_from_slice(&dai1_bytes);
    pad_to_64(&mut outer_rec0_plain);
    outer_rec0_plain.extend_from_slice(&inner_padded_cts[0]);
    pad_to_64(&mut outer_rec0_plain);

    Ok(InnerSealedResult {
        outer_rec0_plain,
        inner_ct_tier_padded,
    })
}

fn append_dme1_trailer(
    out: &mut Vec<u8>,
    inner_header: &DnamJsonHeader,
    rec_key_ids: &[[u8; 32]],
    rec_pubkeys: &[[u8; 32]],
    outer_rec_ctx_sha: &[u8; 32],
    keys: &SealKeys<'_>,
) -> Result<(), DnamError> {
    let mut_json = extract_mutable_json_bytes(inner_header)?;
    let fixed_overhead = 20 + 80 * rec_pubkeys.len() + 4 + 32 * rec_pubkeys.len() + 16 + 16;
    let min_cap = DEFAULT_DMU1_TRAILER_CAPACITY.min(mut_json.len().max(2) + 16);
    let dmu1_cap = {
        let rem = (fixed_overhead + min_cap) % 64;
        if rem == 0 {
            min_cap
        } else {
            min_cap + (64 - rem)
        }
    };
    let dmu1_frame = build_dmu1_mutable_frame(&mut_json, dmu1_cap)?;
    let mut dme1_plain = Vec::with_capacity(4 + 32 * rec_pubkeys.len() + dmu1_frame.len());
    dme1_plain.extend_from_slice(U32::new(rec_pubkeys.len() as u32).as_bytes());
    for pk in rec_pubkeys {
        dme1_plain.extend_from_slice(pk);
    }
    dme1_plain.extend_from_slice(&dmu1_frame);
    let dme1_ct_len = (dme1_plain.len() + 16) as u64;
    let dme1_hdr = build_dme1_header(2, rec_pubkeys.len() as u32, dme1_ct_len)?;
    let mut_wrap_sha = compute_mutable_wrap_sha256(outer_rec_ctx_sha, &dme1_hdr, rec_key_ids)?;
    let dme1_slots: Vec<HpkeSlotPair> = rec_key_ids
        .iter()
        .zip(rec_pubkeys.iter())
        .enumerate()
        .map(|(i, (kid, pk))| {
            let slot_esk = derive_secret_32(
                b"DNAM-DNSL-HPKE-MUT-ESK",
                keys.ephemeral_secret_x25519,
                kid,
                &mut_wrap_sha,
                i as u32,
            );
            let info = build_mutable_hpke_binding_v1(
                b"DNAM-MUTABLE-HPKE-INFO-V1",
                &mut_wrap_sha,
                i as u32,
                kid,
            )?;
            let aad = build_mutable_hpke_binding_v1(
                b"DNAM-MUTABLE-HPKE-AAD-V1",
                &mut_wrap_sha,
                i as u32,
                kid,
            )?;
            hpke_seal_cek_base(&slot_esk, pk, &info, &aad, &keys.cek_mutable)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dme1_slots_vec = encode_hpke_slot_vector(&dme1_slots)?;
    let dme1_aad = build_mutable_content_aad_v1(&mut_wrap_sha, &dme1_slots)?;
    let dme1_ct = aead_seal_record(&keys.cek_mutable, &[0u8; 12], &dme1_aad, &dme1_plain)?;
    let dme1_frame_len = (20 + 80 * rec_pubkeys.len() + dme1_ct.len()) as u64;
    let dmue_footer = DmueFooterWire {
        magic: *b"DMUE",
        frame_len: U64::new(dme1_frame_len),
        reserved: U32::new(0),
    };

    out.extend_from_slice(&dme1_hdr);
    out.extend_from_slice(&dme1_slots_vec[4..]);
    out.extend_from_slice(&dme1_ct);
    out.extend_from_slice(dmue_footer.as_bytes());
    Ok(())
}

fn assemble_outer_dnsl_envelope(
    inner: &InnerSealedResult,
    inner_header: &DnamJsonHeader,
    max_tier: u8,
    rec_key_ids: &[[u8; 32]],
    rec_pubkeys: &[[u8; 32]],
    keys: &SealKeys<'_>,
) -> Result<Vec<u8>, DnamError> {
    let m_tiers = (max_tier as usize) + 1;
    let slots_len = 4 + 80 * rec_key_ids.len();
    let outer_plains: Vec<&[u8]> = core::iter::once(inner.outer_rec0_plain.as_slice())
        .chain(inner.inner_ct_tier_padded.iter().map(Vec::as_slice))
        .collect();
    let outer_ct_lens: Vec<usize> = outer_plains.iter().map(|p| p.len() + 16).collect();

    let dnsl_preamble = DnslPreambleWire::new(max_tier)?;
    let dnsl_preamble_bytes = dnsl_preamble.to_bytes();
    let mut do_off = 640u64;
    let dummy_outer_recs: Vec<(u64, u64)> = outer_ct_lens
        .iter()
        .map(|&l| {
            let rec = (do_off, l as u64);
            do_off += align64_usize(l) as u64;
            rec
        })
        .collect();
    let dummy_outer_plan = build_prewrap_plan_v1(
        &keys.issuance_id,
        0,
        2,
        max_tier,
        &[0u8; 32],
        rec_key_ids,
        &dummy_outer_recs,
    )?;
    let outer_plan_off = 192usize;
    let outer_slots_off = align64_usize(outer_plan_off + dummy_outer_plan.len());
    let outer_manifest_off = align64_usize(outer_slots_off + slots_len);
    let dsa1_len = 105 + 33 * m_tiers;
    let outer_rec0_off = align64_usize(outer_manifest_off + dsa1_len);
    let mut outer_recs: Vec<(u64, u64)> = Vec::with_capacity(1 + m_tiers);
    outer_recs.push((outer_rec0_off as u64, outer_ct_lens[0] as u64));
    let mut o_cursor = align64_usize(outer_rec0_off + outer_ct_lens[0]);
    let mut outer_tier_ends = [U64::new(0); 4];
    for (t, &ct_len) in outer_ct_lens[1..].iter().enumerate() {
        outer_recs.push((o_cursor as u64, ct_len as u64));
        o_cursor = align64_usize(o_cursor + ct_len);
        outer_tier_ends[t] = U64::new(o_cursor as u64);
    }

    let dnsl_dir = DnslDirectoryWire {
        plan_offset: U64::new(outer_plan_off as u64),
        plan_length: U64::new(dummy_outer_plan.len() as u64),
        slots_offset: U64::new(outer_slots_off as u64),
        slots_length: U64::new(slots_len as u64),
        manifest_offset: U64::new(outer_manifest_off as u64),
        manifest_length: U64::new(dsa1_len as u64),
        shared_record_offset: U64::new(outer_rec0_off as u64),
        shared_ciphertext_length: U64::new(outer_ct_lens[0] as u64),
        original_tier_ends: outer_tier_ends,
        reserved: [0u8; 32],
    };
    let outer_struct_sha =
        compute_encrypted_structural_metadata_sha256(0, &dnsl_preamble_bytes, dnsl_dir.as_bytes())?;
    let outer_plan = build_prewrap_plan_v1(
        &keys.issuance_id,
        0,
        2,
        max_tier,
        &outer_struct_sha,
        rec_key_ids,
        &outer_recs,
    )?;
    let outer_wrap_sha = compute_wrap_plan_sha256(&outer_plan);
    let outer_slots: Vec<HpkeSlotPair> = rec_key_ids
        .iter()
        .zip(rec_pubkeys.iter())
        .enumerate()
        .map(|(i, (kid, pk))| {
            let slot_esk = derive_secret_32(
                b"DNAM-DNSL-HPKE-OUTER-ESK",
                keys.ephemeral_secret_x25519,
                kid,
                &outer_wrap_sha,
                i as u32,
            );
            let info =
                build_hpke_slot_binding_v1(b"DNAM-HPKE-INFO-V1", &outer_wrap_sha, i as u32, kid)?;
            let aad =
                build_hpke_slot_binding_v1(b"DNAM-HPKE-AAD-V1", &outer_wrap_sha, i as u32, kid)?;
            hpke_seal_cek_base(&slot_esk, pk, &info, &aad, keys.cek_outer)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outer_slots_vec = encode_hpke_slot_vector(&outer_slots)?;
    let outer_rec_ctx_sha = compute_record_context_sha256(&outer_wrap_sha, &outer_slots_vec);

    let mut outer_padded_cts: Vec<Vec<u8>> = Vec::with_capacity(1 + m_tiers);
    for (idx, pt) in outer_plains.iter().enumerate() {
        let ct_len = (pt.len() + 16) as u64;
        let nonce = build_record_nonce_v1(idx as u32)?;
        let aad = build_record_aad_v1(&outer_rec_ctx_sha, idx as u32, ct_len)?;
        let mut ct = aead_seal_record(keys.cek_outer, &nonce, &aad, pt)?;
        pad_to_64(&mut ct);
        outer_padded_cts.push(ct);
    }
    let outer_ct_tier_refs: Vec<&[u8]> = outer_padded_cts[1..].iter().map(Vec::as_slice).collect();
    let outer_ct_prefixes = compute_ciphertext_prefixes_v1(
        &outer_rec_ctx_sha,
        &outer_padded_cts[0],
        &outer_ct_tier_refs,
    )?;

    let mut public_aux = Vec::new();
    public_aux.extend_from_slice(&outer_plan);
    pad_to_64(&mut public_aux);
    public_aux.extend_from_slice(&outer_slots_vec);
    let outer_pub_meta_sha = compute_outer_public_metadata_sha256(
        &dnsl_preamble_bytes,
        dnsl_dir.as_bytes(),
        &public_aux,
    );
    let outer_manifest = build_outer_manifest_v1(
        &keys.mp_pubkey,
        &outer_pub_meta_sha,
        &outer_rec_ctx_sha,
        &outer_ct_prefixes,
    )?;
    let outer_sig =
        RustCryptoProvider::ed25519_sign(keys.marketplace_seed_ed25519, &outer_manifest);
    let dsa1_hdr = Dsa1ManifestHeaderWire {
        magic: DSA1_MAGIC,
        original_maximum_tier: max_tier,
        reserved: [0u8; 3],
        outer_signer_public_key: keys.mp_pubkey,
    };
    let mut dsa1_bytes = Vec::with_capacity(dsa1_len);
    dsa1_bytes.extend_from_slice(dsa1_hdr.as_bytes());
    dsa1_bytes.extend_from_slice(&encode_prefix_list(&outer_ct_prefixes)?);
    dsa1_bytes.extend_from_slice(&outer_sig);

    let total_outer = outer_tier_ends[max_tier as usize].get() as usize;
    let mut out = Vec::with_capacity(total_outer + DEFAULT_DMU1_TRAILER_CAPACITY + 256);
    out.extend_from_slice(&dnsl_preamble_bytes);
    out.extend_from_slice(dnsl_dir.as_bytes());
    out.extend_from_slice(&outer_plan);
    pad_to_64(&mut out);
    out.extend_from_slice(&outer_slots_vec);
    pad_to_64(&mut out);
    out.extend_from_slice(&dsa1_bytes);
    pad_to_64(&mut out);
    for ct in &outer_padded_cts {
        out.extend_from_slice(ct);
    }
    debug_assert_eq!(out.len(), total_outer);

    append_dme1_trailer(
        &mut out,
        inner_header,
        rec_key_ids,
        rec_pubkeys,
        &outer_rec_ctx_sha,
        keys,
    )?;
    Ok(out)
}

/// Seals a validated `v3.0` `.dnam` container into the canonical `v3.0` `.dnam-seal` (`b"DNSL"`)
/// envelope (`protection_profile = 2`) with `DNPL` inner framing, `DAI1` + `DSA1` signatures,
/// RFC 9180 Base HPKE slots, `ChaCha20-Poly1305` AEAD records, and a `DME1` + `DMUE` trailer.
pub fn seal_dnam_container(
    inner_dnam_bytes: &[u8],
    recipient_certs: &[OemDeviceCertificate],
    trusted_oem_pubkeys: &[(&str, [u8; 32])],
    marketplace_seed_ed25519: &[u8; 32],
    ephemeral_secret_x25519: &[u8; 32],
    cek_32: &[u8; 32],
    chacha20_nonce_96: &[u8; 12],
) -> Result<Vec<u8>, DnamError> {
    if inner_dnam_bytes.len() < 256 || !inner_dnam_bytes.len().is_multiple_of(64) {
        return Err(DnamError::UnalignedOffset {
            field: "inner_dnam_bytes.len",
            offset: inner_dnam_bytes.len() as u32,
        });
    }
    if recipient_certs.is_empty() || recipient_certs.len() > MAX_SEAL_RECIPIENTS {
        return Err(DnamError::CryptoVerificationFailed(
            "seal_dnam_container requires 1..=16 recipient OemDeviceCertificates",
        ));
    }

    let layout = validate_v3_dnam_layout_with_max_profile(inner_dnam_bytes, 2)?;
    let header_slice =
        &inner_dnam_bytes[layout.header_extent.0..layout.header_extent.0 + layout.header_extent.1];
    let mut inner_header = DnamJsonHeader::from_json_slice(header_slice)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    let mutable_slice = &inner_dnam_bytes
        [layout.mutable_extent.0..layout.mutable_extent.0 + layout.mutable_extent.1];
    let dmu_val = parse_and_validate_dmu1_frame(mutable_slice)?;
    merge_mutable_fields(&mut inner_header, &dmu_val)?;
    inner_header.validate()?;

    enforce_distribution_policy(&inner_header, DnamProtectionProfile::EnclaveSealed)?;
    let (rec_key_ids, rec_pubkeys) =
        resolve_sorted_recipients(recipient_certs, trusted_oem_pubkeys, &inner_header)?;

    let max_tier = layout.retained_tier;
    let m_tiers = (max_tier as usize) + 1;
    let shared_slice =
        &inner_dnam_bytes[layout.shared_extent.0..layout.shared_extent.0 + layout.shared_extent.1];
    let auth_slice =
        &inner_dnam_bytes[layout.auth_extent.0..layout.auth_extent.0 + layout.auth_extent.1];
    let mut tier_slices: Vec<&[u8]> = Vec::with_capacity(m_tiers);
    let mut prev_t = layout.tier_start;
    for t in 0..m_tiers {
        let end_t = layout.tier_ends[t];
        tier_slices.push(&inner_dnam_bytes[prev_t..end_t]);
        prev_t = end_t;
    }

    let issuance_full = derive_secret_32(
        b"DNAM-DNSL-ISSUANCE-V1",
        ephemeral_secret_x25519,
        cek_32,
        chacha20_nonce_96,
        0,
    );
    let mut issuance_id = [0u8; 16];
    issuance_id.copy_from_slice(&issuance_full[0..16]);
    let keys = SealKeys {
        mp_pubkey: RustCryptoProvider::ed25519_pubkey_from_seed(marketplace_seed_ed25519),
        marketplace_seed_ed25519,
        ephemeral_secret_x25519,
        issuance_id,
        cek_outer: cek_32,
        cek_inner: derive_secret_32(
            b"DNAM-DNSL-INNER-CEK-V1",
            ephemeral_secret_x25519,
            cek_32,
            chacha20_nonce_96,
            1,
        ),
        cek_mutable: derive_secret_32(
            b"DNAM-DNSL-MUTABLE-CEK-V1",
            ephemeral_secret_x25519,
            cek_32,
            chacha20_nonce_96,
            2,
        ),
    };

    let logical = build_logical_dnpl_framing(
        header_slice,
        shared_slice,
        auth_slice,
        &tier_slices,
        max_tier,
    )?;
    let recipients = SealRecipients {
        key_ids: &rec_key_ids,
        pubkeys: &rec_pubkeys,
    };
    let inner_sealed = seal_inner_dnpl_layer(
        &logical,
        shared_slice,
        auth_slice,
        &tier_slices,
        max_tier,
        &recipients,
        &keys,
    )?;
    assemble_outer_dnsl_envelope(
        &inner_sealed,
        &inner_header,
        max_tier,
        &rec_key_ids,
        &rec_pubkeys,
        &keys,
    )
}
