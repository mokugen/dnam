//! Canonical `v3.0` `.dnam-seal` (`b"DNSL"`) reader, verifier, and unsealer.

use alloc::vec::Vec;

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, IntoBytes};

use super::{
    DSA1_MAGIC, Dsa1ManifestHeaderWire, MAX_SEAL_RECIPIENTS, UnsealedDnslInner, align64_usize,
    extract_mutable_json_bytes, extract_section_slice, merge_mutable_fields, pad_to_64,
};
use crate::container::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DEFAULT_DMU1_TRAILER_CAPACITY, DNAM_MAGIC,
    DmueFooterWire, DnamDirectoryWire, DnamPreambleWire, align64_u64, build_dmu1_mutable_frame,
};
use crate::provenance::provider::CryptoProvider;
use crate::provenance::security::{
    Dai1Envelope, Dme1HeaderWire, DnplPreambleWire, DnslDirectoryWire, DnslPreambleWire,
    InnerDirectoryWire, build_buyer_receipt_v1, build_creator_manifest_v1,
    build_hpke_slot_binding_v1, build_mutable_content_aad_v1, build_mutable_hpke_binding_v1,
    build_outer_manifest_v1, build_plaintext_shared_v1, build_prewrap_plan_v1, build_record_aad_v1,
    build_record_nonce_v1, compute_ciphertext_prefixes_v1, compute_creator_signed_manifest_sha256,
    compute_encrypted_structural_metadata_sha256, compute_license_policy_sha256,
    compute_logical_metadata_sha256, compute_mutable_wrap_sha256,
    compute_outer_public_metadata_sha256, compute_plaintext_prefixes_v1,
    compute_record_context_sha256, compute_wrap_plan_sha256, parse_dme1_decrypted_plaintext,
    parse_hpke_slot_vector, parse_prefix_list, parse_recipient_key_ids, parse_record_directory,
};
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile, reject_if};
use crate::types::Tier;

struct VerifiedOuterDnsl {
    maximum_tier: u8,
    retained_tier: Tier,
    outer_tier_ends: [u64; 4],
    outer_rec_ctx_sha: [u8; 32],
    rec_key_ids: Vec<[u8; 32]>,
    local_kid: [u8; 32],
    slot_idx: usize,
    outer_plains: Vec<Vec<u8>>,
}

struct VerifiedInnerDnpl {
    logical_dnam_preamble: DnamPreambleWire,
    logical_dnam_dir: DnamDirectoryWire,
    header_json_bytes: Vec<u8>,
    shared_bytes: Vec<u8>,
    arena_bytes: Vec<u8>,
    tier_ends: [usize; 4],
    dai1_bytes: Vec<u8>,
    dai1: Dai1Envelope,
}

fn extract_plan_domain_bytes<'a>(plan: &'a [u8], tag: &[u8]) -> Result<&'a [u8], DnamError> {
    let tag_len_wire = U32::new(tag.len() as u32);
    let hdr_len = 4 + tag.len();
    let pos = plan
        .windows(hdr_len)
        .position(|w| &w[..4] == tag_len_wire.as_bytes() && &w[4..] == tag)
        .ok_or(DnamError::CryptoVerificationFailed(
            "missing domain field in pre-wrap plan",
        ))?;
    let (val_len, rest) = U64::read_from_prefix(&plan[pos + hdr_len..])
        .map_err(|_| DnamError::CryptoVerificationFailed("corrupt pre-wrap plan field length"))?;
    let len_usize = usize::try_from(val_len.get())
        .map_err(|_| DnamError::CryptoVerificationFailed("pre-wrap plan field length overflow"))?;
    rest.get(..len_usize)
        .ok_or(DnamError::CryptoVerificationFailed(
            "truncated pre-wrap plan field",
        ))
}

fn parse_outer_records_from_plan(
    plan: &[u8],
    maximum_tier: u8,
) -> Result<Vec<(u64, u64)>, DnamError> {
    let slice = extract_plan_domain_bytes(plan, b"record_descriptors")?;
    parse_record_directory(slice, maximum_tier)
}

fn build_prewrap_plan_with_same_issuance(
    plan: &[u8],
    layer: u8,
    profile: u8,
    maximum_tier: u8,
    structural_sha: &[u8; 32],
    recipients: &[[u8; 32]],
    records: &[(u64, u64)],
) -> Result<Vec<u8>, DnamError> {
    let id_slice = extract_plan_domain_bytes(plan, b"issuance_id")?;
    let mut issuance_id = [0u8; 16];
    reject_if(
        id_slice.len() != 16,
        "invalid issuance_id length in pre-wrap plan",
    )?;
    issuance_id.copy_from_slice(id_slice);
    build_prewrap_plan_v1(
        &issuance_id,
        layer,
        profile,
        maximum_tier,
        structural_sha,
        recipients,
        records,
    )
}

/// Parses and decrypts a `DME1` frame given the signed outermost `record_context_sha256`.
pub fn parse_and_decrypt_dme1_frame<C: CryptoProvider>(
    frame_bytes: &[u8],
    record_context_sha256: &[u8; 32],
    recipient_key_ids: &[[u8; 32]],
    crypto: &C,
) -> Result<serde_json::Value, DnamError> {
    let hdr = Dme1HeaderWire::from_bytes(frame_bytes)?;
    let n = hdr.recipient_count.get() as usize;
    let c_len = hdr.ciphertext_length.get() as usize;
    reject_if(
        n != recipient_key_ids.len(),
        "DME1 recipient count mismatch",
    )?;
    let expected_len = 20 + 80 * n + c_len;
    reject_if(
        frame_bytes.len() != expected_len,
        "invalid DME1 frame length",
    )?;

    let mut slots_vec = Vec::with_capacity(4 + 80 * n);
    slots_vec.extend_from_slice(U32::new(n as u32).as_bytes());
    slots_vec.extend_from_slice(&frame_bytes[20..20 + 80 * n]);
    let slots = parse_hpke_slot_vector(&slots_vec, n, MAX_SEAL_RECIPIENTS)?;

    let local_pk = crypto
        .device_pubkey_x25519()
        .ok_or(DnamError::EncryptedArenaRequiresDeviceKey)?;
    let local_kid = sha256_digest(&local_pk);
    let slot_idx = recipient_key_ids
        .iter()
        .position(|k| k == &local_kid)
        .ok_or(DnamError::CryptoVerificationFailed(
            "local X25519 key not present in DME1 recipient slots",
        ))?;

    let (hdr_20b, _) = <[u8; 20]>::read_from_prefix(frame_bytes)
        .map_err(|_| DnamError::CryptoVerificationFailed("truncated DME1 header"))?;
    let mut_wrap_sha =
        compute_mutable_wrap_sha256(record_context_sha256, &hdr_20b, recipient_key_ids)?;
    let info = build_mutable_hpke_binding_v1(
        b"DNAM-MUTABLE-HPKE-INFO-V1",
        &mut_wrap_sha,
        slot_idx as u32,
        &local_kid,
    )?;
    let aad = build_mutable_hpke_binding_v1(
        b"DNAM-MUTABLE-HPKE-AAD-V1",
        &mut_wrap_sha,
        slot_idx as u32,
        &local_kid,
    )?;
    let cek_mut = crypto.hpke_open_slot_cek(&slots[slot_idx].0, &info, &aad, &slots[slot_idx].1)?;
    let content_aad = build_mutable_content_aad_v1(&mut_wrap_sha, &slots)?;
    let ct_slice = &frame_bytes[20 + 80 * n..];
    let pt = crypto.aead_chacha20poly1305_open(&cek_mut, &[0u8; 12], &content_aad, ct_slice)?;
    parse_dme1_decrypted_plaintext(&pt, recipient_key_ids)
}

fn verify_and_decrypt_outer_dnsl<C: CryptoProvider>(
    sealed_bytes: &[u8],
    pinned_mp_pk: &[u8; 32],
    crypto: &C,
    max_supported_tier: Tier,
) -> Result<VerifiedOuterDnsl, DnamError> {
    let outer_preamble = DnslPreambleWire::from_bytes(sealed_bytes)?;
    let outer_dir = DnslDirectoryWire::from_bytes(sealed_bytes.get(64..192).ok_or(
        DnamError::BufferTooSmall {
            expected: 192,
            actual: sealed_bytes.len(),
        },
    )?)?;
    let m_tiers = (outer_preamble.maximum_tier as usize) + 1;
    let retained_tier_idx = (outer_preamble.maximum_tier.min(max_supported_tier as u8)) as usize;
    let retained_tier = Tier::from_u8(retained_tier_idx as u8);

    let plan_slice = extract_section_slice(
        sealed_bytes,
        outer_dir.plan_offset.get(),
        outer_dir.plan_length.get(),
    )?;
    let slots_slice = extract_section_slice(
        sealed_bytes,
        outer_dir.slots_offset.get(),
        outer_dir.slots_length.get(),
    )?;
    let manifest_slice = extract_section_slice(
        sealed_bytes,
        outer_dir.manifest_offset.get(),
        outer_dir.manifest_length.get(),
    )?;
    reject_if(
        !outer_dir.slots_offset.get().is_multiple_of(64)
            || !outer_dir.manifest_offset.get().is_multiple_of(64)
            || !outer_dir.shared_record_offset.get().is_multiple_of(64)
            || outer_dir.manifest_length.get() != (105 + 33 * m_tiers) as u64,
        "invalid DNSL alignment or manifest length",
    )?;
    let outer_tier_ends = outer_dir.original_tier_ends.map(U64::get);
    let mut prev_end = align64_u64(
        outer_dir.shared_record_offset.get() + outer_dir.shared_ciphertext_length.get(),
    )
    .ok_or(DnamError::CryptoVerificationFailed("DNSL offset overflow"))?;
    for (idx, &end) in outer_tier_ends.iter().enumerate() {
        if idx < m_tiers {
            reject_if(
                !end.is_multiple_of(64) || end <= prev_end,
                "invalid DNSL tier_ends",
            )?;
            prev_end = end;
        } else {
            reject_if(end != 0, "unexpected non-zero DNSL tier end")?;
        }
    }

    let rec_key_ids_slice = extract_plan_domain_bytes(plan_slice, b"recipient_key_ids")?;
    let rec_key_ids = parse_recipient_key_ids(rec_key_ids_slice, MAX_SEAL_RECIPIENTS)?;
    let outer_slots = parse_hpke_slot_vector(slots_slice, rec_key_ids.len(), MAX_SEAL_RECIPIENTS)?;

    let (dsa1_hdr, dsa1_rest) = Dsa1ManifestHeaderWire::read_from_prefix(manifest_slice)
        .map_err(|_| DnamError::CryptoVerificationFailed("truncated DSA1 manifest"))?;
    reject_if(
        dsa1_hdr.magic != DSA1_MAGIC
            || dsa1_hdr.original_maximum_tier != outer_preamble.maximum_tier
            || dsa1_hdr.reserved != [0, 0, 0]
            || &dsa1_hdr.outer_signer_public_key != pinned_mp_pk,
        "DSA1 manifest header or signer mismatch",
    )?;
    let pfxs_end = 1 + 33 * m_tiers;
    reject_if(
        dsa1_rest.len() != pfxs_end + 64,
        "invalid DSA1 manifest length",
    )?;
    let outer_ct_prefixes = parse_prefix_list(&dsa1_rest[..pfxs_end], outer_preamble.maximum_tier)?;
    let mut outer_sig = [0u8; 64];
    outer_sig.copy_from_slice(&dsa1_rest[pfxs_end..]);

    let outer_wrap_sha = compute_wrap_plan_sha256(plan_slice);
    let outer_rec_ctx_sha = compute_record_context_sha256(&outer_wrap_sha, slots_slice);

    let shared_ct = extract_section_slice(
        sealed_bytes,
        outer_dir.shared_record_offset.get(),
        outer_dir.shared_ciphertext_length.get(),
    )?;
    let shared_padded_end =
        align64_usize(outer_dir.shared_record_offset.get() as usize + shared_ct.len());
    let shared_ct_padded = sealed_bytes
        .get(outer_dir.shared_record_offset.get() as usize..shared_padded_end)
        .ok_or(DnamError::BufferTooSmall {
            expected: shared_padded_end,
            actual: sealed_bytes.len(),
        })?;

    let outer_recs_parsed = parse_outer_records_from_plan(plan_slice, outer_preamble.maximum_tier)?;
    reject_if(
        outer_recs_parsed[0]
            != (
                outer_dir.shared_record_offset.get(),
                outer_dir.shared_ciphertext_length.get(),
            ),
        "DNSL shared record mismatch with plan",
    )?;

    let mut outer_rec_slices: Vec<&[u8]> = Vec::with_capacity(2 + retained_tier_idx);
    outer_rec_slices.push(shared_ct);
    let mut tier_ct_padded_slices: Vec<&[u8]> = Vec::with_capacity(retained_tier_idx + 1);
    let mut r_start = shared_padded_end;
    for (t_idx, &tier_end) in outer_tier_ends
        .iter()
        .enumerate()
        .take(retained_tier_idx + 1)
    {
        let (exp_off, exp_len) = outer_recs_parsed[1 + t_idx];
        reject_if(
            exp_off != r_start as u64 || align64_u64(exp_off + exp_len) != Some(tier_end),
            "DNSL tier record mismatch with plan",
        )?;
        let end_u = tier_end as usize;
        let rec_padded = sealed_bytes
            .get(r_start..end_u)
            .ok_or(DnamError::BufferTooSmall {
                expected: end_u,
                actual: sealed_bytes.len(),
            })?;
        tier_ct_padded_slices.push(rec_padded);
        outer_rec_slices.push(extract_section_slice(sealed_bytes, exp_off, exp_len)?);
        r_start = end_u;
    }

    let mut public_aux = Vec::new();
    public_aux.extend_from_slice(plan_slice);
    pad_to_64(&mut public_aux);
    public_aux.extend_from_slice(slots_slice);
    let outer_pub_meta = compute_outer_public_metadata_sha256(
        &sealed_bytes[0..64],
        &sealed_bytes[64..192],
        &public_aux,
    );
    let expected_outer_manifest = build_outer_manifest_v1(
        pinned_mp_pk,
        &outer_pub_meta,
        &outer_rec_ctx_sha,
        &outer_ct_prefixes,
    )?;
    crypto.ed25519_verify(pinned_mp_pk, &expected_outer_manifest, &outer_sig)?;

    let computed_outer_ct_pfxs = compute_ciphertext_prefixes_v1(
        &outer_rec_ctx_sha,
        shared_ct_padded,
        &tier_ct_padded_slices,
    )?;
    for (t_idx, pfx) in computed_outer_ct_pfxs.iter().enumerate() {
        if outer_ct_prefixes.get(t_idx) != Some(pfx) {
            return Err(DnamError::CryptoVerificationFailed(
                "DSA1 outer ciphertext prefix digest mismatch",
            ));
        }
    }

    let expected_outer_struct = compute_encrypted_structural_metadata_sha256(
        0,
        &sealed_bytes[0..64],
        &sealed_bytes[64..192],
    )?;
    let expected_outer_plan = build_prewrap_plan_with_same_issuance(
        plan_slice,
        0,
        2,
        outer_preamble.maximum_tier,
        &expected_outer_struct,
        &rec_key_ids,
        &outer_recs_parsed,
    )?;
    reject_if(
        plan_slice != expected_outer_plan.as_slice(),
        "DNSL pre-wrap plan integrity check failed",
    )?;

    let local_pk = crypto
        .device_pubkey_x25519()
        .ok_or(DnamError::EncryptedArenaRequiresDeviceKey)?;
    let local_kid = sha256_digest(&local_pk);
    let slot_idx = rec_key_ids.iter().position(|k| k == &local_kid).ok_or(
        DnamError::CryptoVerificationFailed(
            "local hardware X25519 key is not in .dnam-seal recipient slots",
        ),
    )?;
    let outer_info = build_hpke_slot_binding_v1(
        b"DNAM-HPKE-INFO-V1",
        &outer_wrap_sha,
        slot_idx as u32,
        &local_kid,
    )?;
    let outer_aad = build_hpke_slot_binding_v1(
        b"DNAM-HPKE-AAD-V1",
        &outer_wrap_sha,
        slot_idx as u32,
        &local_kid,
    )?;
    let cek_outer = crypto.hpke_open_slot_cek(
        &outer_slots[slot_idx].0,
        &outer_info,
        &outer_aad,
        &outer_slots[slot_idx].1,
    )?;

    let mut outer_plains = Vec::with_capacity(outer_rec_slices.len());
    for (idx, ct) in outer_rec_slices.iter().enumerate() {
        let nonce = build_record_nonce_v1(idx as u32)?;
        let aad = build_record_aad_v1(&outer_rec_ctx_sha, idx as u32, ct.len() as u64)?;
        outer_plains.push(crypto.aead_chacha20poly1305_open(&cek_outer, &nonce, &aad, ct)?);
    }

    Ok(VerifiedOuterDnsl {
        maximum_tier: outer_preamble.maximum_tier,
        retained_tier,
        outer_tier_ends,
        outer_rec_ctx_sha,
        rec_key_ids,
        local_kid,
        slot_idx,
        outer_plains,
    })
}

fn verify_and_decrypt_inner_dnpl<C: CryptoProvider>(
    outer: &VerifiedOuterDnsl,
    pinned_mp_pk: &[u8; 32],
    crypto: &C,
) -> Result<VerifiedInnerDnpl, DnamError> {
    let inner_rec0 = &outer.outer_plains[0];
    let inner_preamble = DnplPreambleWire::from_bytes(inner_rec0)?;
    let inner_dir = InnerDirectoryWire::from_bytes(inner_rec0.get(64..192).ok_or(
        DnamError::BufferTooSmall {
            expected: 192,
            actual: inner_rec0.len(),
        },
    )?)?;
    reject_if(
        inner_preamble.protection_profile != 2
            || inner_preamble.buyer_max_tier != outer.maximum_tier,
        "DNPL inner preamble mismatch with DNSL",
    )?;
    let logical_prefix_end = usize::try_from(inner_dir.plan_offset.get())
        .map_err(|_| DnamError::CryptoVerificationFailed("DNPL plan_offset overflow"))?;
    let logical_prefix =
        inner_rec0
            .get(192..logical_prefix_end)
            .ok_or(DnamError::CryptoVerificationFailed(
                "truncated DNPL logical_prefix",
            ))?;
    reject_if(
        compute_logical_metadata_sha256(logical_prefix) != inner_dir.logical_metadata_sha256,
        "DNPL logical_metadata_sha256 mismatch",
    )?;
    let logical_dnam_preamble = DnamPreambleWire::read_from_prefix(logical_prefix)
        .map(|(p, _)| p)
        .map_err(|_| DnamError::CryptoVerificationFailed("invalid logical preamble"))?;
    let logical_dnam_dir = logical_prefix
        .get(64..192)
        .and_then(|s| DnamDirectoryWire::read_from_bytes(s).ok())
        .ok_or(DnamError::CryptoVerificationFailed(
            "invalid logical directory",
        ))?;
    let l_hdr_len = usize::try_from(logical_dnam_dir.immutable_header_length.get())
        .map_err(|_| DnamError::CryptoVerificationFailed("logical header length overflow"))?;
    let header_json_bytes = logical_prefix
        .get(192..192 + l_hdr_len)
        .ok_or(DnamError::CryptoVerificationFailed(
            "truncated logical JSON header",
        ))?
        .to_vec();

    let inner_plan = extract_section_slice(
        inner_rec0,
        inner_dir.plan_offset.get(),
        inner_dir.plan_length.get(),
    )?;
    let inner_slots_slice = extract_section_slice(
        inner_rec0,
        inner_dir.slots_offset.get(),
        inner_dir.slots_length.get(),
    )?;
    let dai1_slice = extract_section_slice(
        inner_rec0,
        inner_dir.authentication_offset.get(),
        inner_dir.authentication_length.get(),
    )?;
    let inner_shared_ct = extract_section_slice(
        inner_rec0,
        inner_dir.shared_record_offset.get(),
        inner_dir.shared_ciphertext_length.get(),
    )?;

    let inner_struct_sha =
        compute_encrypted_structural_metadata_sha256(1, &inner_rec0[0..64], &inner_rec0[64..192])?;
    let inner_recs_parsed =
        parse_outer_records_from_plan(inner_plan, inner_preamble.buyer_max_tier)?;
    let expected_inner_plan = build_prewrap_plan_with_same_issuance(
        inner_plan,
        1,
        2,
        inner_preamble.buyer_max_tier,
        &inner_struct_sha,
        &outer.rec_key_ids,
        &inner_recs_parsed,
    )?;
    reject_if(
        inner_plan != expected_inner_plan.as_slice(),
        "DNPL inner pre-wrap plan integrity check failed",
    )?;

    let inner_slots = parse_hpke_slot_vector(
        inner_slots_slice,
        outer.rec_key_ids.len(),
        MAX_SEAL_RECIPIENTS,
    )?;
    let inner_wrap_sha = compute_wrap_plan_sha256(inner_plan);
    let inner_rec_ctx_sha = compute_record_context_sha256(&inner_wrap_sha, inner_slots_slice);

    let dai1 = Dai1Envelope::from_bytes(dai1_slice)?;
    if &dai1.marketplace_pubkey != pinned_mp_pk {
        return Err(DnamError::CryptoVerificationFailed(
            "DAI1 marketplace_pubkey mismatch against pinned trusted marketplace key",
        ));
    }
    let inner_info = build_hpke_slot_binding_v1(
        b"DNAM-HPKE-INFO-V1",
        &inner_wrap_sha,
        outer.slot_idx as u32,
        &outer.local_kid,
    )?;
    let inner_aad = build_hpke_slot_binding_v1(
        b"DNAM-HPKE-AAD-V1",
        &inner_wrap_sha,
        outer.slot_idx as u32,
        &outer.local_kid,
    )?;
    let cek_inner = crypto.hpke_open_slot_cek(
        &inner_slots[outer.slot_idx].0,
        &inner_info,
        &inner_aad,
        &inner_slots[outer.slot_idx].1,
    )?;

    let shared_bytes = crypto.aead_chacha20poly1305_open(
        &cek_inner,
        &build_record_nonce_v1(0)?,
        &build_record_aad_v1(&inner_rec_ctx_sha, 0, inner_shared_ct.len() as u64)?,
        inner_shared_ct,
    )?;

    let retained_tier_idx = outer.retained_tier as usize;
    let mut inner_shared_ct_padded = inner_shared_ct.to_vec();
    pad_to_64(&mut inner_shared_ct_padded);
    let mut inner_tier_ct_padded = Vec::with_capacity(retained_tier_idx + 1);
    let mut arena_bytes = Vec::new();
    let mut tier_ends = [0usize; 4];
    for (t_idx, outer_inner_ct_slice) in outer.outer_plains[1..].iter().enumerate() {
        let (_, exp_ct_len) = inner_recs_parsed[1 + t_idx];
        let mut padded_ct = outer_inner_ct_slice.to_vec();
        pad_to_64(&mut padded_ct);
        inner_tier_ct_padded.push(padded_ct);
        let raw_ct = outer_inner_ct_slice.get(..exp_ct_len as usize).ok_or(
            DnamError::CryptoVerificationFailed("inner tier ciphertext length mismatch"),
        )?;
        let plain_t = crypto.aead_chacha20poly1305_open(
            &cek_inner,
            &build_record_nonce_v1((1 + t_idx) as u32)?,
            &build_record_aad_v1(&inner_rec_ctx_sha, (1 + t_idx) as u32, exp_ct_len)?,
            raw_ct,
        )?;
        arena_bytes.extend_from_slice(&plain_t);
        tier_ends[t_idx] = arena_bytes.len();
    }

    let inner_tier_ct_refs: Vec<&[u8]> = inner_tier_ct_padded.iter().map(Vec::as_slice).collect();
    let computed_inner_ct_pfxs = compute_ciphertext_prefixes_v1(
        &inner_rec_ctx_sha,
        &inner_shared_ct_padded,
        &inner_tier_ct_refs,
    )?;
    for (t_idx, pfx) in computed_inner_ct_pfxs.iter().enumerate() {
        if dai1.inner_ciphertext_prefixes.get(t_idx) != Some(pfx) {
            return Err(DnamError::CryptoVerificationFailed(
                "DAI1 inner ciphertext prefix digest mismatch",
            ));
        }
    }

    let hdr_sha = sha256_digest(&header_json_bytes);
    let shared_framing = build_plaintext_shared_v1(
        &logical_prefix[0..64],
        &logical_prefix[64..192],
        &shared_bytes,
    );
    let mut tier_slices: Vec<&[u8]> = Vec::with_capacity(retained_tier_idx + 1);
    let mut a_prev = 0usize;
    for end in &tier_ends[..=retained_tier_idx] {
        tier_slices.push(&arena_bytes[a_prev..*end]);
        a_prev = *end;
    }
    let buyer_pfxs = compute_plaintext_prefixes_v1(
        b"DNAM-BUYER-PREFIX-V1",
        &hdr_sha,
        &shared_framing,
        &tier_slices,
    )?;
    for (t_idx, pfx) in buyer_pfxs.iter().enumerate() {
        if dai1.buyer_prefixes.get(t_idx) != Some(pfx) {
            return Err(DnamError::CryptoVerificationFailed(
                "DAI1 buyer plaintext prefix digest mismatch",
            ));
        }
    }
    let creator_manifest =
        build_creator_manifest_v1(&dai1.creator_pubkey, &hdr_sha, &dai1.master_prefixes)?;
    crypto.ed25519_verify(
        &dai1.creator_pubkey,
        &creator_manifest,
        &dai1.creator_signature,
    )?;
    let creator_signed_sha =
        compute_creator_signed_manifest_sha256(&creator_manifest, &dai1.creator_signature);
    let policy_sha = compute_license_policy_sha256(&dai1.policy_bytes);
    let expected_buyer_msg = build_buyer_receipt_v1(
        &dai1.marketplace_pubkey,
        &creator_signed_sha,
        &policy_sha,
        2,
        &dai1.buyer_prefixes,
        Some(&inner_rec_ctx_sha),
        &dai1.inner_ciphertext_prefixes,
    )?;
    crypto.ed25519_verify(
        pinned_mp_pk,
        &expected_buyer_msg,
        &dai1.marketplace_signature,
    )?;

    Ok(VerifiedInnerDnpl {
        logical_dnam_preamble,
        logical_dnam_dir,
        header_json_bytes,
        shared_bytes,
        arena_bytes,
        tier_ends,
        dai1_bytes: dai1_slice.to_vec(),
        dai1,
    })
}

fn maybe_decrypt_dme1_trailer<C: CryptoProvider>(
    sealed_bytes: &[u8],
    outer: &VerifiedOuterDnsl,
    crypto: &C,
) -> Result<Option<serde_json::Value>, DnamError> {
    let retained_end = outer.outer_tier_ends[outer.retained_tier as usize] as usize;
    let max_end = outer.outer_tier_ends[outer.maximum_tier as usize] as usize;
    if sealed_bytes.len() == max_end || sealed_bytes.len() == retained_end {
        return Ok(None);
    }
    if sealed_bytes.len() < retained_end.saturating_add(16) {
        return Err(DnamError::CryptoVerificationFailed("truncated DMUE footer"));
    }
    let footer = DmueFooterWire::ref_from_bytes(&sealed_bytes[sealed_bytes.len() - 16..])
        .map_err(|_| DnamError::CryptoVerificationFailed("invalid DMUE footer"))?;
    reject_if(
        &footer.magic != b"DMUE" || footer.reserved.get() != 0,
        "invalid DMUE footer",
    )?;
    let trailer_total = (footer.frame_len.get() as usize)
        .checked_add(16)
        .ok_or(DnamError::CryptoVerificationFailed("DME1 length overflow"))?;
    let trailer_start = if max_end.checked_add(trailer_total) == Some(sealed_bytes.len()) {
        max_end
    } else if retained_end.checked_add(trailer_total) == Some(sealed_bytes.len()) {
        retained_end
    } else {
        return Err(DnamError::CryptoVerificationFailed(
            "DME1 trailer size mismatch",
        ));
    };
    let dme1_slice = &sealed_bytes[trailer_start..sealed_bytes.len() - 16];
    let decrypted = parse_and_decrypt_dme1_frame(
        dme1_slice,
        &outer.outer_rec_ctx_sha,
        &outer.rec_key_ids,
        crypto,
    )?;
    Ok(Some(decrypted))
}

/// Verifies and decrypts a canonical `v3.0` `.dnam-seal` (`b"DNSL"`) envelope up to `max_supported_tier`
/// fail-closed against `crypto.marketplace_pubkey()` and `crypto.device_pubkey_x25519()`.
pub fn unseal_dnsl_up_to_tier<C: CryptoProvider>(
    sealed_bytes: &[u8],
    crypto: &C,
    max_supported_tier: Tier,
) -> Result<UnsealedDnslInner, DnamError> {
    let pinned_mp_pk = crypto
        .marketplace_pubkey()
        .ok_or(DnamError::CryptoVerificationFailed(
            "trusted marketplace_pubkey required to unseal .dnam-seal",
        ))?;
    let outer =
        verify_and_decrypt_outer_dnsl(sealed_bytes, pinned_mp_pk, crypto, max_supported_tier)?;
    let inner = verify_and_decrypt_inner_dnpl(&outer, pinned_mp_pk, crypto)?;
    let inner_header = DnamJsonHeader::from_json_slice(&inner.header_json_bytes)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    super::super::enforce_distribution_policy(&inner_header, DnamProtectionProfile::EnclaveSealed)?;
    if let Some(receipt) = inner_header
        .metadata
        .signatures
        .as_ref()
        .and_then(|s| s.marketplace_receipt.as_ref())
        && !receipt.authorized_devices.is_empty()
        && let Some(local_id) = crypto.local_device_id_hash()
        && !receipt
            .authorized_devices
            .iter()
            .any(|d| d.device_id_hash == local_id)
    {
        return Err(DnamError::CryptoVerificationFailed(
            "local hardware device_id_hash is not in receipt.authorized_devices",
        ));
    }
    let mutable_json = maybe_decrypt_dme1_trailer(sealed_bytes, &outer, crypto)?;

    Ok(UnsealedDnslInner {
        maximum_tier: outer.maximum_tier,
        retained_tier: outer.retained_tier,
        logical_dnam_preamble: inner.logical_dnam_preamble,
        logical_dnam_dir: inner.logical_dnam_dir,
        header_json_bytes: inner.header_json_bytes,
        shared_bytes: inner.shared_bytes,
        auth_bytes: inner.dai1_bytes,
        arena_bytes: inner.arena_bytes,
        tier_ends: inner.tier_ends,
        mutable_json,
        dai1: inner.dai1,
    })
}

/// Unseals a canonical `v3.0` `.dnam-seal` (`b"DNSL"`) envelope into a validated canonical
/// `v3.0` plaintext `.dnam` container (`DnamPreambleWire`, `DnamDirectoryWire`, JCS header,
/// `shared_extent`, `DCRC`, `Arena 0..retained`, and `DMU1`/`DMUE`), failing closed when
/// `crypto.marketplace_pubkey()` is absent or verification fails.
pub fn unseal_dnam_container<C: CryptoProvider>(
    sealed_bytes: &[u8],
    crypto: &C,
) -> Result<Vec<u8>, DnamError> {
    let unsealed = unseal_dnsl_up_to_tier(sealed_bytes, crypto, Tier::Ultra)?;
    let mut header = DnamJsonHeader::from_json_slice(&unsealed.header_json_bytes)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    if let Some(ref mut_val) = unsealed.mutable_json {
        merge_mutable_fields(&mut header, mut_val)?;
    }
    let mut_json = extract_mutable_json_bytes(&header)?;
    let dmu1 = build_dmu1_mutable_frame(&mut_json, DEFAULT_DMU1_TRAILER_CAPACITY)?;

    let auth_len = 64usize;
    let hdr_off = 192u64;
    let hdr_len = unsealed.header_json_bytes.len() as u64;
    let shared_off = align64_u64(hdr_off + hdr_len).ok_or(DnamError::CryptoVerificationFailed(
        "DNSL unseal shared_off overflow",
    ))?;
    let shared_len = unsealed.shared_bytes.len() as u64;
    let auth_off = align64_u64(shared_off + shared_len).ok_or(
        DnamError::CryptoVerificationFailed("DNSL unseal auth_off overflow"),
    )?;
    let tier_start = align64_u64(auth_off + auth_len as u64).ok_or(
        DnamError::CryptoVerificationFailed("DNSL unseal tier_start overflow"),
    )?;
    let mut tier_ends = [U64::new(0); 4];
    for t in 0..=(unsealed.retained_tier as usize) {
        tier_ends[t] = U64::new(tier_start + unsealed.tier_ends[t] as u64);
    }
    let preamble = DnamPreambleWire {
        magic: DNAM_MAGIC,
        version_major: U16::new(CONTAINER_VERSION_MAJOR),
        version_minor: U16::new(CONTAINER_VERSION_MINOR),
        protection_profile: 0,
        auth_kind: 0,
        maximum_tier: unsealed.retained_tier as u8,
        reserved_flag: 0,
        sample_rate_train_hz: U32::new(48_000),
        reserved_pad: [0u8; 48],
    };
    let dir = DnamDirectoryWire {
        immutable_header_offset: U64::new(hdr_off),
        immutable_header_length: U64::new(hdr_len),
        shared_binary_offset: U64::new(shared_off),
        shared_binary_length: U64::new(shared_len),
        authentication_envelope_offset: U64::new(auth_off),
        authentication_envelope_length: U64::new(auth_len as u64),
        reserved_offset: U64::new(0),
        reserved_capacity: U64::new(0),
        tier_start: U64::new(tier_start),
        original_tier_ends: tier_ends,
        reserved_tail: [0u8; 24],
    };
    let total_end = (tier_start as usize) + unsealed.arena_bytes.len();
    let mut out = Vec::with_capacity(total_end + dmu1.len() + 16);
    out.extend_from_slice(preamble.as_bytes());
    out.extend_from_slice(dir.as_bytes());
    out.extend_from_slice(&unsealed.header_json_bytes);
    pad_to_64(&mut out);
    out.extend_from_slice(&unsealed.shared_bytes);
    pad_to_64(&mut out);
    out.resize(out.len() + 64, 0);
    let mut crc = 0_u32;
    let mut crc_tiers = [0_u32; 4];
    let mut prev_t = 0_usize;
    for t in 0..=(unsealed.retained_tier as usize) {
        let end_t = unsealed.tier_ends[t];
        crc = crate::container::crc32c_update(crc, &unsealed.arena_bytes[prev_t..end_t]);
        crc_tiers[t] = crc;
        prev_t = end_t;
    }
    let last_crc = crc_tiers[unsealed.retained_tier as usize];
    for slot in crc_tiers
        .iter_mut()
        .skip((unsealed.retained_tier as usize) + 1)
    {
        *slot = last_crc;
    }
    let mut flags_hint = 0_u16;
    if header.io_config.output_mode != "mono" {
        flags_hint |= crate::container::FLAG_STEREO_OUT;
    }
    if header.metadata.signatures.is_some() {
        flags_hint |= crate::container::FLAG_HAS_SIGNATURES;
    }
    let dcrc = crate::container::DcrcAuthRecord {
        crc_prefix: crate::container::crc32c_update(0, &out[..auth_off as usize]),
        crc_tiers,
        num_controls: header.controls_schema.len().min(u16::MAX as usize) as u16,
        flags_hint,
        tail_time_ms_bits: header.io_config.tail_time_ms.to_bits(),
    };
    out[auth_off as usize..auth_off as usize + 64].copy_from_slice(&dcrc.to_bytes());
    out.extend_from_slice(&unsealed.arena_bytes);
    out.extend_from_slice(&dmu1);
    let dmue = DmueFooterWire {
        magic: *b"DMUE",
        frame_len: U64::new(dmu1.len() as u64),
        reserved: U32::new(0),
    };
    out.extend_from_slice(dmue.as_bytes());
    Ok(out)
}
