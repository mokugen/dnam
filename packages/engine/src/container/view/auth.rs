//! Container protection profile resolution, `DAU1` two-certificate verification,
//! and incremental arena prefix `CRC32C` verification.

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::provenance::{
    CryptoProvider, VerificationStatus, enforce_distribution_policy, verify_container_provenance,
};
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile};
use crate::types::Tier;

use super::super::preamble::DnamPreamble;
use super::super::wire::V2ContainerLayout;
use super::DnamContainerView;
use super::{FLAG_HAS_FULL_DELTA, FLAG_HAS_PRO_DELTA, FLAG_HAS_ULTRA_DELTA, crc32c_update};

pub(super) fn resolve_protection_profile_and_policy(
    header: &DnamJsonHeader,
    layout_protection_profile: u8,
    dcrc_encrypted: bool,
    is_enclave_sealed_wrapper: bool,
) -> Result<DnamProtectionProfile, DnamError> {
    let has_authorized_devices = header
        .metadata
        .signatures
        .as_ref()
        .and_then(|s| s.marketplace_receipt.as_ref())
        .is_some_and(|r| !r.authorized_devices.is_empty());
    let protection_profile = if is_enclave_sealed_wrapper {
        DnamProtectionProfile::EnclaveSealed
    } else {
        match layout_protection_profile {
            0 if dcrc_encrypted || has_authorized_devices => {
                DnamProtectionProfile::DeviceAllowlisted
            }
            0 => DnamProtectionProfile::Open,
            1 => DnamProtectionProfile::DeviceAllowlisted,
            2 => DnamProtectionProfile::EnclaveSealed,
            _ => {
                return Err(DnamError::SchemaValidationError(
                    "invalid v2 protection_profile".to_string(),
                ));
            }
        }
    };
    enforce_distribution_policy(header, protection_profile)?;
    Ok(protection_profile)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_dau1_or_legacy_provenance<C: CryptoProvider>(
    bytes: &[u8],
    auth_slice: &[u8],
    json_slice: &[u8],
    shared_slice: &[u8],
    layout: &V2ContainerLayout,
    header: &DnamJsonHeader,
    preamble_has_signatures: bool,
    retained_tier: Tier,
    plain_bytes: &[u8],
    crypto: &C,
) -> Result<VerificationStatus, DnamError> {
    if !(auth_slice.starts_with(b"DAU1") || layout.auth_kind > 0) {
        return verify_container_provenance(header, preamble_has_signatures, plain_bytes, crypto);
    }

    let env = crate::provenance::Dau1Envelope::from_bytes(auth_slice)?;
    if env.auth_kind != layout.auth_kind {
        return Err(DnamError::CryptoVerificationFailed(
            "DAU1 auth_kind mismatch with preamble",
        ));
    }
    let shared_framing =
        crate::provenance::build_plaintext_shared_v1(&bytes[0..64], &bytes[64..192], shared_slice);
    let hdr_sha = crate::provenance::sha256_digest(json_slice);
    let mut tier_slices_vec = Vec::with_capacity((retained_tier as usize) + 1);
    let mut prev_t = layout.tier_start;
    for t_idx in 0..=(retained_tier as usize) {
        let end_t = layout.tier_ends[t_idx];
        tier_slices_vec.push(&bytes[prev_t..end_t]);
        prev_t = end_t;
    }
    if env.auth_kind == 0 {
        if env.creator_max_tier != layout.maximum_tier {
            return Err(DnamError::CryptoVerificationFailed(
                "DAU1 creator_max_tier mismatch with preamble",
            ));
        }
        let prefixes = crate::provenance::compute_plaintext_prefixes_v1(
            b"DNAM-MASTER-PREFIX-V1",
            &hdr_sha,
            &shared_framing,
            &tier_slices_vec,
        )?;
        for (t_idx, pfx) in prefixes.iter().enumerate() {
            if env.master_prefixes.get(t_idx) != Some(pfx) {
                return Err(DnamError::CryptoVerificationFailed(
                    "DAU1 master tier prefix digest mismatch",
                ));
            }
        }
    }
    let creator_manifest = crate::provenance::build_creator_manifest_v1(
        &env.creator_pubkey,
        &hdr_sha,
        &env.master_prefixes,
    )?;
    crypto.ed25519_verify(
        &env.creator_pubkey,
        &creator_manifest,
        &env.creator_signature,
    )?;
    if env.auth_kind == 1 {
        let m_pub = env
            .marketplace_pubkey
            .ok_or(DnamError::CryptoVerificationFailed(
                "missing DAU1 marketplace_pubkey",
            ))?;
        if let Some(pinned_mp) = crypto.marketplace_pubkey()
            && &m_pub != pinned_mp
        {
            return Err(DnamError::CryptoVerificationFailed(
                "DAU1 marketplace_pubkey mismatch against pinned trusted marketplace key",
            ));
        }
        let b_max = env
            .buyer_max_tier
            .ok_or(DnamError::CryptoVerificationFailed(
                "missing DAU1 buyer_max_tier",
            ))?;
        if b_max != layout.maximum_tier || b_max > env.creator_max_tier {
            return Err(DnamError::CryptoVerificationFailed(
                "DAU1 buyer_max_tier mismatch with preamble or creator_max_tier",
            ));
        }
        let m_sig = env
            .marketplace_signature
            .ok_or(DnamError::CryptoVerificationFailed(
                "missing DAU1 marketplace_signature",
            ))?;
        let buyer_prefixes = crate::provenance::compute_plaintext_prefixes_v1(
            b"DNAM-BUYER-PREFIX-V1",
            &hdr_sha,
            &shared_framing,
            &tier_slices_vec,
        )?;
        for (t_idx, bpfx) in buyer_prefixes.iter().enumerate() {
            if env.buyer_prefixes.get(t_idx) != Some(bpfx) {
                return Err(DnamError::CryptoVerificationFailed(
                    "DAU1 buyer tier prefix digest mismatch",
                ));
            }
        }
        let creator_signed_sha = crate::provenance::compute_creator_signed_manifest_sha256(
            &creator_manifest,
            &env.creator_signature,
        );
        let policy_sha = crate::provenance::compute_license_policy_sha256(&env.policy_bytes);
        let buyer_receipt = crate::provenance::build_buyer_receipt_v1(
            &m_pub,
            &creator_signed_sha,
            &policy_sha,
            layout.protection_profile,
            &env.buyer_prefixes,
            None,
            &[],
        )?;
        crypto.ed25519_verify(&m_pub, &buyer_receipt, &m_sig)?;
        Ok(VerificationStatus::MarketplaceVerified {
            signer_id: header.metadata.author.clone().unwrap_or_default(),
            marketplace_id: alloc::string::String::from("dnam-marketplace"),
            license_id: alloc::string::String::from("dau1"),
            license_tier: Tier::from_u8(b_max).as_str().to_string(),
        })
    } else {
        Ok(VerificationStatus::CreatorVerified {
            signer_id: header.metadata.author.clone().unwrap_or_default(),
            pubkey_ed25519: env.creator_pubkey,
        })
    }
}

impl<'a> DnamContainerView<'a> {
    /// Verifies `arena_crc32c_lite`, `arena_crc32c_full`, `arena_crc32c_pro`, and `arena_crc32c_ultra`
    /// over the available prefix of `plain_arena` in a single incremental pass.
    pub fn verify_arena_prefix_crcs(
        preamble: &DnamPreamble,
        plain_arena: &[u8],
    ) -> Result<(), DnamError> {
        let lite_end = preamble.arena_offset_lite_end() as usize;
        let mut crc = 0_u32;
        if plain_arena.len() >= lite_end && lite_end > 0 {
            crc = crc32c_update(0, &plain_arena[..lite_end]);
            if crc != preamble.arena_crc32c_lite {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_lite",
                    expected: preamble.arena_crc32c_lite,
                    computed: crc,
                });
            }
        }
        let full_end = preamble.arena_offset_full_end() as usize;
        if (preamble.flags & FLAG_HAS_FULL_DELTA) != 0
            && plain_arena.len() >= full_end
            && full_end > lite_end
        {
            crc = crc32c_update(crc, &plain_arena[lite_end..full_end]);
            if crc != preamble.arena_crc32c_full {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_full",
                    expected: preamble.arena_crc32c_full,
                    computed: crc,
                });
            }
        }
        let pro_end = preamble.arena_offset_pro_end() as usize;
        if (preamble.flags & FLAG_HAS_PRO_DELTA) != 0
            && plain_arena.len() >= pro_end
            && pro_end > full_end
        {
            crc = crc32c_update(crc, &plain_arena[full_end..pro_end]);
            if crc != preamble.arena_crc32c_pro {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_pro",
                    expected: preamble.arena_crc32c_pro,
                    computed: crc,
                });
            }
        }
        let ultra_end = preamble.arena_offset_ultra_end() as usize;
        if (preamble.flags & FLAG_HAS_ULTRA_DELTA) != 0
            && plain_arena.len() >= ultra_end
            && ultra_end > pro_end
        {
            crc = crc32c_update(crc, &plain_arena[pro_end..ultra_end]);
            if crc != preamble.arena_crc32c_ultra {
                return Err(DnamError::Crc32cMismatch {
                    region: "arena_ultra",
                    expected: preamble.arena_crc32c_ultra,
                    computed: crc,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::DefaultCryptoProvider;
    use crate::schema::DnamJsonHeader;

    #[test]
    fn test_protection_profile_and_dau1_error_branches() {
        let hdr = DnamJsonHeader::default_reference();
        assert_eq!(
            resolve_protection_profile_and_policy(&hdr, 0, false, true).unwrap(),
            DnamProtectionProfile::EnclaveSealed
        );
        assert_eq!(
            resolve_protection_profile_and_policy(&hdr, 0, true, false).unwrap(),
            DnamProtectionProfile::DeviceAllowlisted
        );
        assert_eq!(
            resolve_protection_profile_and_policy(&hdr, 1, false, false).unwrap(),
            DnamProtectionProfile::DeviceAllowlisted
        );
        assert_eq!(
            resolve_protection_profile_and_policy(&hdr, 2, false, false).unwrap(),
            DnamProtectionProfile::EnclaveSealed
        );
        assert!(resolve_protection_profile_and_policy(&hdr, 3, false, false).is_err());

        #[cfg(feature = "crypto")]
        {
            use crate::provenance::{
                Dau1Envelope, RustCryptoProvider, build_creator_manifest_v1,
                compute_plaintext_prefixes_v1, sha256_digest,
            };

            let mut bytes = alloc::vec![0u8; 512];
            let json = br#"{}"#;
            let shared = b"";
            let tier0 = &bytes[256..320];
            let hdr_sha = sha256_digest(json);
            let shared_framing = crate::provenance::build_plaintext_shared_v1(
                &bytes[0..64],
                &bytes[64..192],
                shared,
            );
            let master_pfx = compute_plaintext_prefixes_v1(
                b"DNAM-MASTER-PREFIX-V1",
                &hdr_sha,
                &shared_framing,
                &[tier0],
            )
            .unwrap();
            let creator_seed = [9u8; 32];
            let creator_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&creator_seed);
            let creator_manifest =
                build_creator_manifest_v1(&creator_pk, &hdr_sha, &master_pfx).unwrap();
            let creator_sig = RustCryptoProvider::ed25519_sign(&creator_seed, &creator_manifest);

            let env_creator = Dau1Envelope {
                auth_kind: 0,
                creator_pubkey: creator_pk,
                creator_max_tier: 0,
                master_prefixes: master_pfx.clone(),
                creator_signature: creator_sig,
                marketplace_pubkey: None,
                policy_bytes: alloc::vec![],
                buyer_max_tier: None,
                buyer_prefixes: alloc::vec![],
                marketplace_signature: None,
            };
            let env_creator_bytes = env_creator.to_bytes().unwrap();

            let mut layout = V2ContainerLayout {
                magic: *b"DNAM",
                protection_profile: 0,
                auth_kind: 1, // mismatch with env.auth_kind = 0
                maximum_tier: 0,
                retained_tier: 0,
                sample_rate_train_hz: 48_000,
                header_extent: (192, json.len()),
                shared_extent: (256, 0),
                auth_extent: (256, env_creator_bytes.len()),
                tier_start: 256,
                tier_ends: [320, 320, 320, 320],
                mutable_extent: (320, 32),
            };
            let crypto = DefaultCryptoProvider::default();
            assert!(
                verify_dau1_or_legacy_provenance(
                    &bytes,
                    &env_creator_bytes,
                    json,
                    shared,
                    &layout,
                    &hdr,
                    true,
                    Tier::Lite,
                    tier0,
                    &crypto,
                )
                .is_err()
            );

            // creator_max_tier mismatch
            layout.auth_kind = 0;
            layout.maximum_tier = 1;
            assert!(
                verify_dau1_or_legacy_provenance(
                    &bytes,
                    &env_creator_bytes,
                    json,
                    shared,
                    &layout,
                    &hdr,
                    true,
                    Tier::Lite,
                    tier0,
                    &crypto,
                )
                .is_err()
            );

            // Valid auth_kind == 1 envelope, trigger buyer_max_tier & buyer_prefix mismatches
            let mkt_seed = [10u8; 32];
            let mkt_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mkt_seed);
            let mut env_buyer = env_creator;
            env_buyer.auth_kind = 1;
            env_buyer.marketplace_pubkey = Some(mkt_pk);
            env_buyer.buyer_max_tier = Some(0);
            env_buyer.buyer_prefixes = alloc::vec![[0xAA; 32]]; // mismatched digest
            env_buyer.marketplace_signature = Some([0u8; 64]);
            let env_buyer_bytes = env_buyer.to_bytes().unwrap();

            layout.auth_kind = 1;
            layout.maximum_tier = 1; // b_max (0) != maximum_tier (1)
            assert!(
                verify_dau1_or_legacy_provenance(
                    &bytes,
                    &env_buyer_bytes,
                    json,
                    shared,
                    &layout,
                    &hdr,
                    true,
                    Tier::Lite,
                    tier0,
                    &crypto,
                )
                .is_err()
            );

            layout.maximum_tier = 0; // buyer_prefixes mismatch
            assert!(
                verify_dau1_or_legacy_provenance(
                    &bytes,
                    &env_buyer_bytes,
                    json,
                    shared,
                    &layout,
                    &hdr,
                    true,
                    Tier::Lite,
                    tier0,
                    &crypto,
                )
                .is_err()
            );
            bytes.clear();
        }
    }
}
