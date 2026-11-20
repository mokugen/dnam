//! 2-Certificate `Ed25519` Provenance Verification & Marketplace Receipt Signing.

mod jcs;

pub(crate) use jcs::canonicalize_jcs_value;
pub use jcs::{
    compute_immutable_header_sha256, compute_immutable_header_sha256_for_version,
    project_immutable_header_for_schema_version,
};

use alloc::string::String;
use alloc::vec::Vec;

use zerocopy::IntoBytes;
use zerocopy::byteorder::little_endian::{U16, U32};

use crate::container::encode_domain_field;
use crate::schema::{
    DnamError, DnamJsonHeader, MarketplaceReceipt, ModelSignatures, OemDeviceCertificate,
};

use super::provider::{CryptoProvider, VerificationStatus};
use super::security::{build_oem_cert_v2_signing_message, encode_prefix_list};
use super::sha256::{decode_hex_array, sha256_digest};

/// Builds the domain-separated creator signing message binding `immutable_header_sha256`,
/// `master_arena_sha256`, and optional 4-tier `tier_arena_sha256` prefix digests strictly via
/// length-prefixed `encode_domain_field` (`"DNAM-CREATOR-MANIFEST-V1"`, `ref[sec.sig.prefix-delegation]`).
pub fn build_creator_signing_message_for_tiers(
    immutable_header_sha256: &[u8; 32],
    master_arena_sha256: &[u8; 32],
    tier_arena_sha256: Option<&[String; 4]>,
) -> Result<Vec<u8>, DnamError> {
    let mut msg = Vec::with_capacity(288);
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-CREATOR-MANIFEST-V1"));
    msg.extend_from_slice(&encode_domain_field(
        "immutable_header_sha256",
        immutable_header_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "master_arena_sha256",
        master_arena_sha256,
    ));
    if let Some(tier_shas) = tier_arena_sha256 {
        let mut decoded = [[0u8; 32]; 4];
        for (slot, hex) in decoded.iter_mut().zip(tier_shas.iter()) {
            *slot = decode_hex_array::<32>(hex)?;
        }
        let prefix_list = encode_prefix_list(&decoded)?;
        msg.extend_from_slice(&encode_domain_field("tier_arena_sha256", &prefix_list));
    }
    Ok(msg)
}

/// Builds the domain-separated creator signing message (`tier_arena_sha256 = None`) via
/// length-prefixed `encode_domain_field` (`"DNAM-CREATOR-MANIFEST-V1"`, `ref[sec.sig.prefix-delegation]`).
#[inline]
pub fn build_creator_signing_message(
    immutable_header_sha256: &[u8; 32],
    master_arena_sha256: &[u8; 32],
) -> Vec<u8> {
    build_creator_signing_message_for_tiers(immutable_header_sha256, master_arena_sha256, None)
        .expect("tierless creator signing message cannot fail")
}

/// Builds the domain-separated byte payload signed by the marketplace's `Ed25519` key for a
/// [`MarketplaceReceipt`] using strict `encode_domain_field` framing across all receipt fields
/// and prefix digests (`"DNAM-BUYER-RECEIPT-V1"`, `ref[sec.sig.prefix-delegation]`).
pub fn build_marketplace_receipt_signing_message(
    immutable_header_sha256: &[u8; 32],
    master_arena_sha256: &[u8; 32],
    receipt: &MarketplaceReceipt,
) -> Result<Vec<u8>, DnamError> {
    let buyer_arena_sha256 = decode_hex_array::<32>(&receipt.buyer_arena_sha256)?;
    let mut msg = Vec::with_capacity(512);
    msg.extend_from_slice(&encode_domain_field("domain", b"DNAM-BUYER-RECEIPT-V1"));
    msg.extend_from_slice(&encode_domain_field(
        "immutable_header_sha256",
        immutable_header_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "master_arena_sha256",
        master_arena_sha256,
    ));
    msg.extend_from_slice(&encode_domain_field(
        "buyer_arena_sha256",
        &buyer_arena_sha256,
    ));
    if let Some(ref tier_shas) = receipt.buyer_tier_arena_sha256 {
        let mut decoded = [[0u8; 32]; 4];
        for (slot, hex) in decoded.iter_mut().zip(tier_shas.iter()) {
            *slot = decode_hex_array::<32>(hex)?;
        }
        let prefix_list = encode_prefix_list(&decoded)?;
        msg.extend_from_slice(&encode_domain_field(
            "buyer_tier_arena_sha256",
            &prefix_list,
        ));
    }
    msg.extend_from_slice(&encode_domain_field(
        "marketplace_id",
        receipt.marketplace_id.as_bytes(),
    ));
    msg.extend_from_slice(&encode_domain_field(
        "license_id",
        receipt.license_id.as_bytes(),
    ));
    msg.extend_from_slice(&encode_domain_field(
        "buyer_hash",
        receipt.buyer_hash.as_bytes(),
    ));
    msg.extend_from_slice(&encode_domain_field(
        "license_tier",
        receipt.license_tier.as_bytes(),
    ));
    msg.extend_from_slice(&encode_domain_field(
        "issued_at",
        receipt.issued_at.as_bytes(),
    ));
    msg.extend_from_slice(&encode_domain_field(
        "encryption_mode",
        receipt.encryption_mode.as_bytes(),
    ));
    if let Some(ref nonce) = receipt.arena_nonce_96 {
        msg.extend_from_slice(&encode_domain_field("arena_nonce_96", nonce.as_bytes()));
    }
    if let Some(ref epk) = receipt.ephemeral_pubkey_x25519 {
        msg.extend_from_slice(&encode_domain_field(
            "ephemeral_pubkey_x25519",
            epk.as_bytes(),
        ));
    }
    msg.extend_from_slice(&encode_domain_field(
        "authorized_device_count",
        U32::new(receipt.authorized_devices.len() as u32).as_bytes(),
    ));
    for dev in &receipt.authorized_devices {
        msg.extend_from_slice(&encode_domain_field(
            "device_oem_vendor",
            dev.oem_vendor.as_bytes(),
        ));
        msg.extend_from_slice(&encode_domain_field(
            "device_model_code",
            dev.model_code.as_bytes(),
        ));
        msg.extend_from_slice(&encode_domain_field(
            "device_id_hash",
            dev.device_id_hash.as_bytes(),
        ));
        if let Some(ref wrapped) = dev.wrapped_cek {
            msg.extend_from_slice(&encode_domain_field(
                "device_wrapped_cek",
                wrapped.as_bytes(),
            ));
        }
    }
    msg.extend_from_slice(&encode_domain_field(
        "royalty_split_count",
        U32::new(receipt.royalty_splits.len() as u32).as_bytes(),
    ));
    for split in &receipt.royalty_splits {
        msg.extend_from_slice(&encode_domain_field(
            "royalty_recipient_id",
            split.recipient_id.as_bytes(),
        ));
        msg.extend_from_slice(&encode_domain_field("royalty_role", split.role.as_bytes()));
        msg.extend_from_slice(&encode_domain_field(
            "royalty_share_bps",
            U16::new(split.share_bps).as_bytes(),
        ));
    }
    Ok(msg)
}

const TIER_END_OFFSETS: [usize; 4] = [4_608, 7_936, 15_872, 30_720];

type VerifiedCreatorDigests = ([u8; 32], [u8; 32], [u8; 32]);

fn verify_header_and_creator_signature<C: CryptoProvider>(
    header: &DnamJsonHeader,
    sigs: &ModelSignatures,
    crypto: &C,
) -> Result<VerifiedCreatorDigests, DnamError> {
    let expected_header_sha = decode_hex_array::<32>(&sigs.immutable_header_sha256)?;
    let computed_header_sha = compute_immutable_header_sha256(header)?;
    if expected_header_sha != computed_header_sha {
        return Err(DnamError::CryptoVerificationFailed(
            "immutable_header_sha256 mismatch (header was tampered with)",
        ));
    }

    let master_arena_sha = decode_hex_array::<32>(&sigs.master_arena_sha256)?;
    let creator_pubkey = decode_hex_array::<32>(&sigs.creator_signature.pubkey_ed25519)?;
    let creator_sig = decode_hex_array::<64>(&sigs.creator_signature.sig_ed25519)?;
    let creator_msg = build_creator_signing_message_for_tiers(
        &expected_header_sha,
        &master_arena_sha,
        sigs.tier_arena_sha256.as_ref(),
    )?;
    crypto.ed25519_verify(&creator_pubkey, &creator_msg, &creator_sig)?;
    Ok((expected_header_sha, master_arena_sha, creator_pubkey))
}

fn verify_arena_and_tier_shas(
    plaintext_arena: &[u8],
    expected_full_sha: &[u8; 32],
    tier_shas: Option<&[String; 4]>,
    full_mismatch_msg: &'static str,
    tier_mismatch_msg: &'static str,
) -> Result<(), DnamError> {
    let computed_arena_sha = sha256_digest(plaintext_arena);
    let prefix_tier_idx = TIER_END_OFFSETS
        .iter()
        .position(|&off| off == plaintext_arena.len());
    let mut sha_matched = computed_arena_sha == *expected_full_sha;
    if let (Some(t_idx), Some(tier_shas)) = (prefix_tier_idx, tier_shas) {
        for (k, end_off) in TIER_END_OFFSETS.iter().copied().enumerate().take(t_idx + 1) {
            let expected_tier_sha = decode_hex_array::<32>(&tier_shas[k])?;
            let actual_tier_sha = if end_off == plaintext_arena.len() {
                computed_arena_sha
            } else {
                sha256_digest(&plaintext_arena[..end_off])
            };
            if actual_tier_sha != expected_tier_sha {
                return Err(DnamError::CryptoVerificationFailed(tier_mismatch_msg));
            }
        }
        if t_idx < 3 {
            sha_matched = true;
        }
    }
    if !sha_matched {
        return Err(DnamError::CryptoVerificationFailed(full_mismatch_msg));
    }
    Ok(())
}

fn verify_marketplace_receipt_sig<C: CryptoProvider>(
    sigs: &ModelSignatures,
    receipt: &MarketplaceReceipt,
    expected_header_sha: &[u8; 32],
    master_arena_sha: &[u8; 32],
    crypto: &C,
) -> Result<VerificationStatus, DnamError> {
    if !receipt.authorized_devices.is_empty()
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
    let mp_pubkey = crypto
        .marketplace_pubkey()
        .ok_or(DnamError::CryptoVerificationFailed(
            "trusted marketplace_pubkey required to verify MarketplaceReceipt",
        ))?;
    let receipt_sig = decode_hex_array::<64>(&receipt.receipt_sig_ed25519)?;
    let receipt_msg =
        build_marketplace_receipt_signing_message(expected_header_sha, master_arena_sha, receipt)?;
    crypto.ed25519_verify(mp_pubkey, &receipt_msg, &receipt_sig)?;
    Ok(VerificationStatus::MarketplaceVerified {
        signer_id: sigs.creator_signature.signer_id.clone(),
        marketplace_id: receipt.marketplace_id.clone(),
        license_id: receipt.license_id.clone(),
        license_tier: receipt.license_tier.clone(),
    })
}

/// Verifies the split-payload `Ed25519` signature chain (`creator_signature` and optional
/// `marketplace_receipt`) and plaintext `Arena 0..3` SHA-256 integrity.
pub fn verify_container_provenance<C: CryptoProvider>(
    header: &DnamJsonHeader,
    has_signatures_flag: bool,
    plaintext_arena: &[u8],
    crypto: &C,
) -> Result<VerificationStatus, DnamError> {
    let Some(ref sigs) = header.metadata.signatures else {
        if has_signatures_flag {
            return Err(DnamError::CryptoVerificationFailed(
                "HAS_SIGNATURES flag is set in preamble, but metadata.signatures is missing",
            ));
        }
        return Ok(VerificationStatus::Unsigned);
    };

    let (expected_header_sha, master_arena_sha, creator_pubkey) =
        verify_header_and_creator_signature(header, sigs, crypto)?;

    if let Some(ref receipt) = sigs.marketplace_receipt {
        let buyer_arena_sha = decode_hex_array::<32>(&receipt.buyer_arena_sha256)?;
        verify_arena_and_tier_shas(
            plaintext_arena,
            &buyer_arena_sha,
            receipt.buyer_tier_arena_sha256.as_ref(),
            "buyer_arena_sha256 mismatch against plaintext tensor arena",
            "buyer_tier_arena_sha256 mismatch against plaintext tensor arena prefix",
        )?;
        verify_marketplace_receipt_sig(
            sigs,
            receipt,
            &expected_header_sha,
            &master_arena_sha,
            crypto,
        )
    } else {
        verify_arena_and_tier_shas(
            plaintext_arena,
            &master_arena_sha,
            sigs.tier_arena_sha256.as_ref(),
            "master_arena_sha256 mismatch against plaintext tensor arena",
            "tier_arena_sha256 mismatch against plaintext tensor arena prefix",
        )?;
        Ok(VerificationStatus::CreatorVerified {
            signer_id: sigs.creator_signature.signer_id.clone(),
            pubkey_ed25519: creator_pubkey,
        })
    }
}

/// Verifies `immutable_header_sha256`, `creator_signature` (against `master_arena_sha256`), and
/// optional `marketplace_receipt` (`receipt_sig_ed25519` + `authorized_devices`) for a tier-capped
/// `DAI1`-authenticated `.dnam-seal` prefix (`retained_tier < Tier::Ultra`) when optional per-tier
/// JSON SHA-256 arrays are absent.
pub fn verify_container_provenance_for_dai1_prefix<C: CryptoProvider>(
    header: &DnamJsonHeader,
    crypto: &C,
) -> Result<VerificationStatus, DnamError> {
    let Some(ref sigs) = header.metadata.signatures else {
        return Err(DnamError::CryptoVerificationFailed(
            "HAS_SIGNATURES flag is set in preamble, but metadata.signatures is missing",
        ));
    };
    let (expected_header_sha, master_arena_sha, creator_pubkey) =
        verify_header_and_creator_signature(header, sigs, crypto)?;
    if let Some(ref receipt) = sigs.marketplace_receipt {
        verify_marketplace_receipt_sig(
            sigs,
            receipt,
            &expected_header_sha,
            &master_arena_sha,
            crypto,
        )
    } else {
        Ok(VerificationStatus::CreatorVerified {
            signer_id: sigs.creator_signature.signer_id.clone(),
            pubkey_ed25519: creator_pubkey,
        })
    }
}

/// Verifies a factory-provisioned V2 [`OemDeviceCertificate`] (`"DNAM-OEM-CERT-V2"`) against the
/// manufacturer's pinned 32-byte `Ed25519` public key, returning the verified 32-byte TrustZone `X25519` public key.
pub fn verify_oem_device_certificate<C: CryptoProvider>(
    cert: &OemDeviceCertificate,
    trusted_oem_pubkey_ed25519: &[u8; 32],
    crypto: &C,
) -> Result<[u8; 32], DnamError> {
    let raw_id = decode_hex_array::<16>(&cert.device_id_hash)?;
    let pubkey_x25519 = decode_hex_array::<32>(&cert.pubkey_x25519)?;
    let oem_sig = decode_hex_array::<64>(&cert.oem_sig_ed25519)?;
    let assurance = cert.assurance.unwrap_or(1);
    let policy_sha = match cert.attestation_policy_sha256 {
        Some(ref hex) => decode_hex_array::<32>(hex)?,
        None => [0u8; 32],
    };
    let evidence_sha = match cert.attestation_evidence_sha256 {
        Some(ref hex) => decode_hex_array::<32>(hex)?,
        None => [0u8; 32],
    };
    let msg = build_oem_cert_v2_signing_message(
        &cert.oem_vendor,
        &cert.model_code,
        &raw_id,
        &pubkey_x25519,
        trusted_oem_pubkey_ed25519,
        assurance,
        &policy_sha,
        &evidence_sha,
    )?;
    crypto.ed25519_verify(trusted_oem_pubkey_ed25519, &msg, &oem_sig)?;
    Ok(pubkey_x25519)
}

pub use super::seal::{DNSL_MAGIC, MAX_SEAL_RECIPIENTS, enforce_distribution_policy};

#[cfg(feature = "crypto")]
pub use super::seal::seal_dnam_container;

pub use super::seal::unseal_dnam_container;

/// Specification for signing and attaching a [`MarketplaceReceipt`] to a creator-signed
/// [`DnamJsonHeader`] (`ref[sec.sig.prefix-delegation]`).
#[cfg(feature = "crypto")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MarketplaceReceiptSigningSpec {
    #[serde(alias = "marketplaceId")]
    pub marketplace_id: String,
    #[serde(alias = "licenseId")]
    pub license_id: String,
    #[serde(alias = "buyerHash")]
    pub buyer_hash: String,
    #[serde(alias = "licenseTier")]
    pub license_tier: String,
    #[serde(default, alias = "issuedAt")]
    pub issued_at: Option<String>,
    #[serde(default, alias = "encryptionMode")]
    pub encryption_mode: Option<String>,
    #[serde(default, alias = "arenaNonce96")]
    pub arena_nonce_96: Option<String>,
    #[serde(default, alias = "ephemeralPubkeyX25519")]
    pub ephemeral_pubkey_x25519: Option<String>,
    #[serde(default, alias = "authorizedDevices")]
    pub authorized_devices: Vec<crate::schema::AuthorizedDevice>,
    #[serde(default, alias = "royaltySplits")]
    pub royalty_splits: Vec<crate::schema::RoyaltySplit>,
    #[serde(default, alias = "includeBuyerTierPrefixes")]
    pub include_buyer_tier_prefixes: bool,
}

/// Signs a [`DnamJsonHeader`] JSON string and `master_arena` using the creator's 32-byte `Ed25519`
/// seed, returning the updated canonical wire JSON header string (`ref[sec.sig.prefix-delegation]`).
#[cfg(feature = "crypto")]
pub fn sign_creator_header_json(
    header_json: &str,
    master_arena: &[u8],
    creator_seed_32: &[u8; 32],
    signer_id: &str,
    include_tier_prefixes: bool,
) -> Result<String, DnamError> {
    use crate::container::{
        ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    };
    use crate::provenance::RustCryptoProvider;
    use crate::provenance::sha256::encode_hex;
    use crate::schema::{CreatorSignature, ModelSignatures};

    let mut header: DnamJsonHeader = DnamJsonHeader::from_json_str(header_json)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    let hdr_sha_bytes = compute_immutable_header_sha256(&header)?;
    let arena_sha_bytes = sha256_digest(master_arena);
    let pubkey = RustCryptoProvider::ed25519_pubkey_from_seed(creator_seed_32);
    let tier_arena_sha256 =
        if include_tier_prefixes && master_arena.len() >= ARENA_OFFSET_ULTRA_END as usize {
            Some([
                encode_hex(&sha256_digest(
                    &master_arena[..ARENA_OFFSET_LITE_END as usize],
                )),
                encode_hex(&sha256_digest(
                    &master_arena[..ARENA_OFFSET_FULL_END as usize],
                )),
                encode_hex(&sha256_digest(
                    &master_arena[..ARENA_OFFSET_PRO_END as usize],
                )),
                encode_hex(&sha256_digest(
                    &master_arena[..ARENA_OFFSET_ULTRA_END as usize],
                )),
            ])
        } else {
            None
        };
    let msg = build_creator_signing_message_for_tiers(
        &hdr_sha_bytes,
        &arena_sha_bytes,
        tier_arena_sha256.as_ref(),
    )?;
    let sig = RustCryptoProvider::ed25519_sign(creator_seed_32, &msg);

    header.metadata.signatures = Some(ModelSignatures {
        immutable_header_sha256: encode_hex(&hdr_sha_bytes),
        master_arena_sha256: encode_hex(&arena_sha_bytes),
        tier_arena_sha256,
        creator_signature: CreatorSignature {
            signer_id: String::from(signer_id),
            pubkey_ed25519: encode_hex(&pubkey),
            sig_ed25519: encode_hex(&sig),
        },
        marketplace_receipt: None,
    });
    header
        .to_json_string()
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))
}

/// Signs a [`MarketplaceReceipt`] and attaches it to a creator-signed [`DnamJsonHeader`] JSON
/// string, returning the updated canonical wire JSON header string (`ref[sec.sig.prefix-delegation]`).
#[cfg(feature = "crypto")]
pub fn sign_marketplace_receipt_json(
    header_json: &str,
    buyer_arena: &[u8],
    marketplace_seed_32: &[u8; 32],
    spec: MarketplaceReceiptSigningSpec,
) -> Result<String, DnamError> {
    use crate::container::{
        ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    };
    use crate::provenance::RustCryptoProvider;
    use crate::provenance::sha256::encode_hex;

    let mut header: DnamJsonHeader = DnamJsonHeader::from_json_str(header_json)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    let sigs = header.metadata.signatures.as_mut().ok_or_else(|| {
        DnamError::SchemaValidationError(String::from(
            "sign_marketplace_receipt requires a creator-signed header",
        ))
    })?;

    let hdr_sha = decode_hex_array::<32>(&sigs.immutable_header_sha256)?;
    let master_sha = decode_hex_array::<32>(&sigs.master_arena_sha256)?;
    let buyer_arena_sha256 = encode_hex(&sha256_digest(buyer_arena));

    let buyer_tier_arena_sha256 = if spec.include_buyer_tier_prefixes
        && buyer_arena.len() >= ARENA_OFFSET_ULTRA_END as usize
    {
        Some([
            encode_hex(&sha256_digest(
                &buyer_arena[..ARENA_OFFSET_LITE_END as usize],
            )),
            encode_hex(&sha256_digest(
                &buyer_arena[..ARENA_OFFSET_FULL_END as usize],
            )),
            encode_hex(&sha256_digest(
                &buyer_arena[..ARENA_OFFSET_PRO_END as usize],
            )),
            encode_hex(&sha256_digest(
                &buyer_arena[..ARENA_OFFSET_ULTRA_END as usize],
            )),
        ])
    } else {
        None
    };

    let mut receipt = MarketplaceReceipt {
        marketplace_id: spec.marketplace_id,
        license_id: spec.license_id,
        buyer_hash: spec.buyer_hash,
        license_tier: spec.license_tier,
        issued_at: spec
            .issued_at
            .unwrap_or_else(|| String::from("2026-11-04T00:00:00Z")),
        buyer_arena_sha256,
        buyer_tier_arena_sha256,
        encryption_mode: spec.encryption_mode.unwrap_or_else(|| String::from("none")),
        arena_nonce_96: spec.arena_nonce_96,
        ephemeral_pubkey_x25519: spec.ephemeral_pubkey_x25519,
        authorized_devices: spec.authorized_devices,
        royalty_splits: spec.royalty_splits,
        receipt_sig_ed25519: String::new(),
    };

    let msg = build_marketplace_receipt_signing_message(&hdr_sha, &master_sha, &receipt)?;
    let sig = RustCryptoProvider::ed25519_sign(marketplace_seed_32, &msg);
    receipt.receipt_sig_ed25519 = encode_hex(&sig);
    sigs.marketplace_receipt = Some(receipt);

    header
        .to_json_string()
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::{PresetDef, SchemaVersion};
    use alloc::collections::BTreeMap;

    #[test]
    fn test_immutable_header_jcs_excludes_mutable_presets_and_user_metadata() {
        let mut h1 = DnamJsonHeader::default_reference();
        let sha1 = compute_immutable_header_sha256(&h1).unwrap();

        h1.presets.push(PresetDef {
            id: String::from("user_custom"),
            name: String::from("User Custom Preset"),
            description: Some(String::from("Player")),
            controls: BTreeMap::from([(String::from("gain"), serde_json::json!(9.5))]),
            user_metadata: BTreeMap::new(),
        });
        h1.user_metadata
            .insert(String::from("favorite"), serde_json::json!(true));
        h1.user_metadata
            .insert(String::from("null_val"), serde_json::json!(null));
        let sha2 = compute_immutable_header_sha256(&h1).unwrap();
        assert_eq!(sha1, sha2);

        // Missing signatures when HAS_SIGNATURES is set must fail
        let crypto = DefaultCryptoProvider::default();
        assert!(verify_container_provenance(&h1, true, &[0u8; 64], &crypto).is_err());
        assert_eq!(
            verify_container_provenance(&h1, false, &[0u8; 64], &crypto).unwrap(),
            VerificationStatus::Unsigned
        );

        #[cfg(feature = "crypto")]
        {
            use crate::provenance::RustCryptoProvider;
            use crate::provenance::sha256::encode_hex;
            use crate::schema::{CreatorSignature, ModelSignatures};

            let imm_sha = compute_immutable_header_sha256(&h1).unwrap();
            let arena = [0x42u8; 64];
            let arena_sha = sha256_digest(&arena);
            let seed = [11u8; 32];
            let pk = RustCryptoProvider::ed25519_pubkey_from_seed(&seed);
            let msg = build_creator_signing_message(&imm_sha, &arena_sha);
            let sig = RustCryptoProvider::ed25519_sign(&seed, &msg);

            h1.metadata.signatures = Some(ModelSignatures {
                immutable_header_sha256: encode_hex(&imm_sha),
                master_arena_sha256: encode_hex(&arena_sha),
                tier_arena_sha256: None,
                creator_signature: CreatorSignature {
                    signer_id: String::from("creator-1"),
                    pubkey_ed25519: encode_hex(&pk),
                    sig_ed25519: encode_hex(&sig),
                },
                marketplace_receipt: None,
            });
            let status = verify_container_provenance(&h1, true, &arena, &crypto).unwrap();
            assert_eq!(
                status,
                VerificationStatus::CreatorVerified {
                    signer_id: String::from("creator-1"),
                    pubkey_ed25519: pk,
                }
            );

            // Tampered arena must fail
            assert!(verify_container_provenance(&h1, true, &[0u8; 64], &crypto).is_err());

            // Test tier_arena_sha256 prefix verification and mismatch on creator-only signed container
            let full_arena = [0x42u8; 30_720];
            let full_arena_sha = sha256_digest(&full_arena);
            let tier_shas = [
                encode_hex(&sha256_digest(&full_arena[..4_608])),
                encode_hex(&sha256_digest(&full_arena[..7_936])),
                encode_hex(&sha256_digest(&full_arena[..15_872])),
                encode_hex(&full_arena_sha),
            ];
            let msg_full = build_creator_signing_message_for_tiers(
                &imm_sha,
                &full_arena_sha,
                Some(&tier_shas),
            )
            .unwrap();
            let sig_full = RustCryptoProvider::ed25519_sign(&seed, &msg_full);
            h1.metadata.signatures = Some(ModelSignatures {
                immutable_header_sha256: encode_hex(&imm_sha),
                master_arena_sha256: encode_hex(&full_arena_sha),
                tier_arena_sha256: Some(tier_shas.clone()),
                creator_signature: CreatorSignature {
                    signer_id: String::from("creator-1"),
                    pubkey_ed25519: encode_hex(&pk),
                    sig_ed25519: encode_hex(&sig_full),
                },
                marketplace_receipt: None,
            });
            assert!(verify_container_provenance(&h1, true, &full_arena[..7_936], &crypto).is_ok());
            assert!(verify_container_provenance(&h1, true, &[0u8; 7_936], &crypto).is_err());

            // Test marketplace_receipt error branches: buyer_tier_arena_sha256 mismatch,
            // buyer_arena_sha256 mismatch, and unauthorized local_device_id_hash
            let mut rcpt = crate::schema::MarketplaceReceipt {
                marketplace_id: String::from("mkt-1"),
                license_id: String::from("lic-1"),
                buyer_hash: encode_hex(&[0xAA; 32]),
                license_tier: String::from("pro"),
                issued_at: String::from("2026-03-31T00:00:00Z"),
                buyer_arena_sha256: encode_hex(&full_arena_sha),
                buyer_tier_arena_sha256: Some(tier_shas.clone()),
                encryption_mode: String::from("none"),
                arena_nonce_96: None,
                ephemeral_pubkey_x25519: None,
                authorized_devices: alloc::vec![crate::schema::AuthorizedDevice {
                    oem_vendor: String::from("darkglass"),
                    model_code: String::from("anagram"),
                    device_id_hash: String::from("0102030405060708090a0b0c0d0e0f10"),
                    wrapped_cek: None,
                }],
                royalty_splits: alloc::vec![],
                receipt_sig_ed25519: encode_hex(&[0u8; 64]),
            };
            h1.metadata.signatures.as_mut().unwrap().marketplace_receipt = Some(rcpt.clone());

            // 1. buyer_tier_arena_sha256 mismatch on prefix
            assert!(verify_container_provenance(&h1, true, &[0u8; 7_936], &crypto).is_err());

            // 2. buyer_arena_sha256 mismatch when buyer_tier_arena_sha256 is None
            rcpt.buyer_tier_arena_sha256 = None;
            h1.metadata.signatures.as_mut().unwrap().marketplace_receipt = Some(rcpt.clone());
            assert!(verify_container_provenance(&h1, true, &[0u8; 64], &crypto).is_err());

            // 3. Unauthorized local_device_id_hash
            let unauth_crypto = RustCryptoProvider::new()
                .with_device_key("ffffffffffffffffffffffffffffffff", [1u8; 32]);
            assert!(verify_container_provenance(&h1, true, &full_arena, &unauth_crypto).is_err());

            // Tampered header must fail
            h1.metadata.name = String::from("Tampered");
            assert!(verify_container_provenance(&h1, true, &arena, &crypto).is_err());

            // Unsupported future schema version must fail validation and JCS projection
            let mut h_future = DnamJsonHeader::default_reference();
            h_future.version = String::from("1.99.0");
            assert!(h_future.validate().is_err());
            assert!(compute_immutable_header_sha256(&h_future).is_err());
            assert!(
                compute_immutable_header_sha256_for_version(&h1, SchemaVersion::new(4, 0, 0))
                    .is_err()
            );

            // SchemaVersion parsing and preamble matching
            let v3 = SchemaVersion::parse("3.0.0").unwrap();
            assert_eq!(v3, SchemaVersion::V3_0_0);
            assert!(v3.is_supported());
            assert!(v3.ensure_matches_preamble(3, 0).is_ok());
            assert!(v3.ensure_matches_preamble(3, 1).is_err());
            assert!(SchemaVersion::parse("invalid").is_err());
            assert!(SchemaVersion::parse("3.0.0.1").is_err());

            // Test sign_creator_header_json and sign_marketplace_receipt_json
            let base_hdr = DnamJsonHeader::default_reference();
            let base_json = serde_json::to_string(&base_hdr).unwrap();
            assert!(
                sign_marketplace_receipt_json(
                    &base_json,
                    &full_arena,
                    &[22u8; 32],
                    MarketplaceReceiptSigningSpec {
                        marketplace_id: String::from("mkt-1"),
                        license_id: String::from("lic-1"),
                        buyer_hash: encode_hex(&[0xBB; 32]),
                        license_tier: String::from("lite"),
                        issued_at: None,
                        encryption_mode: None,
                        arena_nonce_96: None,
                        ephemeral_pubkey_x25519: None,
                        authorized_devices: alloc::vec![],
                        royalty_splits: alloc::vec![],
                        include_buyer_tier_prefixes: false,
                    },
                )
                .is_err()
            );
            let signed_creator_json =
                sign_creator_header_json(&base_json, &full_arena, &seed, "creator-1", true)
                    .unwrap();
            let mkt_seed = [22u8; 32];
            let mkt_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mkt_seed);
            let signed_mkt_json = sign_marketplace_receipt_json(
                &signed_creator_json,
                &full_arena,
                &mkt_seed,
                MarketplaceReceiptSigningSpec {
                    marketplace_id: String::from("mkt-1"),
                    license_id: String::from("lic-1"),
                    buyer_hash: encode_hex(&[0xBB; 32]),
                    license_tier: String::from("pro"),
                    issued_at: None,
                    encryption_mode: None,
                    arena_nonce_96: None,
                    ephemeral_pubkey_x25519: None,
                    authorized_devices: alloc::vec![],
                    royalty_splits: alloc::vec![],
                    include_buyer_tier_prefixes: true,
                },
            )
            .unwrap();
            let signed_mkt_hdr: DnamJsonHeader = serde_json::from_str(&signed_mkt_json).unwrap();
            let mkt_crypto = RustCryptoProvider::new().with_marketplace_pubkey(mkt_pk);
            let mkt_status =
                verify_container_provenance(&signed_mkt_hdr, true, &full_arena, &mkt_crypto)
                    .unwrap();
            assert_eq!(
                mkt_status,
                VerificationStatus::MarketplaceVerified {
                    signer_id: String::from("creator-1"),
                    marketplace_id: String::from("mkt-1"),
                    license_id: String::from("lic-1"),
                    license_tier: String::from("pro"),
                }
            );

            // Fail closed when marketplace_pubkey is None even if MarketplaceReceipt signatures match!
            assert!(
                verify_container_provenance(
                    &signed_mkt_hdr,
                    true,
                    &full_arena,
                    &RustCryptoProvider::new(),
                )
                .is_err()
            );

            // Receipt field tampering (issued_at / royalty_splits / tier_arena_sha256) must fail!
            let mut tampered_issued = signed_mkt_hdr.clone();
            tampered_issued
                .metadata
                .signatures
                .as_mut()
                .unwrap()
                .marketplace_receipt
                .as_mut()
                .unwrap()
                .issued_at = String::from("2027-01-01T00:00:00Z");
            assert!(
                verify_container_provenance(&tampered_issued, true, &full_arena, &mkt_crypto)
                    .is_err()
            );
            let mut tampered_royalty = signed_mkt_hdr.clone();
            tampered_royalty
                .metadata
                .signatures
                .as_mut()
                .unwrap()
                .marketplace_receipt
                .as_mut()
                .unwrap()
                .royalty_splits
                .push(crate::schema::RoyaltySplit {
                    recipient_id: String::from("attacker"),
                    role: String::from("amp_capture"),
                    share_bps: 10_000,
                });
            assert!(
                verify_container_provenance(&tampered_royalty, true, &full_arena, &mkt_crypto)
                    .is_err()
            );
            let mut tampered_tier_sha = signed_mkt_hdr;
            tampered_tier_sha
                .metadata
                .signatures
                .as_mut()
                .unwrap()
                .tier_arena_sha256
                .as_mut()
                .unwrap()[0] = encode_hex(&[0xEEu8; 32]);
            assert!(
                verify_container_provenance(&tampered_tier_sha, true, &full_arena, &mkt_crypto)
                    .is_err()
            );
        }
    }
}
