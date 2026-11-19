//! Canonical `v3.0` `.dnam-seal` (`b"DNSL"`) hardware enclave framing and `DistributionPolicy` enforcement.
//!
//! Canonical Reference: `docs/specs/security/provenance_and_enclave_drm.md` (`ref[sec.policy.distribution-policy]`,
//! `ref[sec.drm.oem-certificate]`, `ref[sec.drm.hpke-record-encryption]`, `ref[sec.drm.dnsl-public-framing]`).

pub mod hpke_base;
pub mod v3_dnsl;

#[cfg(feature = "crypto")]
pub use v3_dnsl::seal_dnam_container;
pub use v3_dnsl::{
    Dsa1ManifestHeaderWire, UnsealedDnslInner, parse_and_decrypt_dme1_frame, unseal_dnam_container,
    unseal_dnsl_up_to_tier,
};

pub use crate::provenance::security::DNSL_WIRE_MAGIC as DNSL_MAGIC;
use crate::schema::{DnamError, DnamJsonHeader, DnamProtectionProfile};

/// Maximum number of hardware recipient slots in a `.dnam-seal` (`b"DNSL"`) envelope (`16`).
pub const MAX_SEAL_RECIPIENTS: usize = 16;

/// Enforces the creator-signed [`crate::schema::DistributionPolicy`] (`metadata.distribution_policy`)
/// against the container's active [`DnamProtectionProfile`] and `marketplace_receipt.authorized_devices`.
pub fn enforce_distribution_policy(
    header: &DnamJsonHeader,
    active_profile: DnamProtectionProfile,
) -> Result<(), DnamError> {
    let Some(ref policy) = header.metadata.distribution_policy else {
        return Ok(());
    };

    if !policy.is_profile_allowed(active_profile) {
        return Err(DnamError::CryptoVerificationFailed(
            "container protection profile is not permitted by creator distribution_policy",
        ));
    }

    if !policy.allowed_oem_vendors.is_empty() {
        if active_profile == DnamProtectionProfile::Open {
            return Err(DnamError::CryptoVerificationFailed(
                "creator distribution_policy restricts model to specific OEM hardware vendors",
            ));
        }

        let maybe_receipt = header
            .metadata
            .signatures
            .as_ref()
            .and_then(|s| s.marketplace_receipt.as_ref());

        if let Some(receipt) = maybe_receipt {
            if receipt.authorized_devices.is_empty()
                && active_profile == DnamProtectionProfile::DeviceAllowlisted
            {
                return Err(DnamError::CryptoVerificationFailed(
                    "creator distribution_policy requires authorized OEM devices",
                ));
            }
            for dev in &receipt.authorized_devices {
                if !policy.is_oem_device_allowed(&dev.oem_vendor, &dev.model_code) {
                    return Err(DnamError::CryptoVerificationFailed(
                        "authorized device OEM vendor/model is not permitted by creator distribution_policy",
                    ));
                }
            }
        } else if active_profile == DnamProtectionProfile::DeviceAllowlisted {
            return Err(DnamError::CryptoVerificationFailed(
                "creator distribution_policy requires authorized OEM devices",
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        AuthorizedDevice, DistributionPolicy, DnamJsonHeader, MarketplaceReceipt, ModelSignatures,
    };
    use alloc::string::ToString;

    #[test]
    fn test_distribution_policy_enforcement_branches() {
        let mut header = DnamJsonHeader::default_reference();

        // 1. No distribution policy: always Ok
        header.metadata.distribution_policy = None;
        assert!(enforce_distribution_policy(&header, DnamProtectionProfile::Open).is_ok());

        // 2. Profile not allowed
        header.metadata.distribution_policy = Some(DistributionPolicy {
            allowed_protection_profiles: alloc::vec![DnamProtectionProfile::EnclaveSealed],
            allowed_oem_vendors: alloc::vec![],
        });
        assert!(enforce_distribution_policy(&header, DnamProtectionProfile::Open).is_err());

        // 3. Vendor restricted + Open profile -> Err
        header.metadata.distribution_policy = Some(DistributionPolicy {
            allowed_protection_profiles: alloc::vec![
                DnamProtectionProfile::Open,
                DnamProtectionProfile::DeviceAllowlisted
            ],
            allowed_oem_vendors: alloc::vec!["darkglass".to_string()],
        });
        assert!(enforce_distribution_policy(&header, DnamProtectionProfile::Open).is_err());

        // 4. DeviceAllowlisted without receipt -> Err
        header.metadata.signatures = None;
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_err()
        );

        // 5. DeviceAllowlisted with empty authorized_devices -> Err
        header.metadata.signatures = Some(ModelSignatures {
            immutable_header_sha256: "0".repeat(64),
            master_arena_sha256: "0".repeat(64),
            tier_arena_sha256: None,
            creator_signature: crate::schema::CreatorSignature {
                signer_id: "creator".to_string(),
                pubkey_ed25519: "0".repeat(64),
                sig_ed25519: "0".repeat(128),
            },
            marketplace_receipt: Some(MarketplaceReceipt {
                marketplace_id: "mkt".to_string(),
                license_id: "lic".to_string(),
                buyer_hash: "0".repeat(64),
                license_tier: "oem".to_string(),
                issued_at: "2026-01-01T00:00:00Z".to_string(),
                buyer_arena_sha256: "0".repeat(64),
                buyer_tier_arena_sha256: None,
                encryption_mode: "none".to_string(),
                arena_nonce_96: None,
                ephemeral_pubkey_x25519: None,
                authorized_devices: alloc::vec![],
                royalty_splits: alloc::vec![],
                receipt_sig_ed25519: "0".repeat(128),
            }),
        });
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_err()
        );

        // 6. DeviceAllowlisted with matching authorized device -> Ok
        if let Some(ref mut sigs) = header.metadata.signatures
            && let Some(ref mut rcpt) = sigs.marketplace_receipt
        {
            rcpt.authorized_devices.push(AuthorizedDevice {
                oem_vendor: "darkglass".to_string(),
                model_code: "anagram".to_string(),
                device_id_hash: "0".repeat(32),
                wrapped_cek: None,
            });
        }
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_ok()
        );

        // 7. Disallowed device vendor -> Err
        if let Some(ref mut sigs) = header.metadata.signatures
            && let Some(ref mut rcpt) = sigs.marketplace_receipt
        {
            rcpt.authorized_devices.push(AuthorizedDevice {
                oem_vendor: "disallowed".to_string(),
                model_code: "other".to_string(),
                device_id_hash: "0".repeat(32),
                wrapped_cek: None,
            });
        }
        assert!(
            enforce_distribution_policy(&header, DnamProtectionProfile::DeviceAllowlisted).is_err()
        );
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn test_v3_dnsl_seal_unseal_fail_closed_and_tamper_resistance() {
        use crate::container::{ARENA_OFFSET_ULTRA_END, DnamContainerView, build_dnam_container};
        use crate::provenance::RustCryptoProvider;
        use crate::provenance::security::{DnslDirectoryWire, DnslPreambleWire};
        use alloc::string::String;

        assert!(DnslPreambleWire::from_bytes(&[0u8; 16]).is_err());
        assert!(DnslDirectoryWire::from_bytes(&[0u8; 16]).is_err());

        let mut header = DnamJsonHeader::default_reference();
        header
            .user_metadata
            .insert(String::from("tag"), serde_json::json!("studio"));
        header.presets.push(crate::schema::PresetDef {
            id: String::from("p1"),
            name: String::from("Preset 1"),
            description: None,
            controls: alloc::collections::BTreeMap::new(),
            user_metadata: alloc::collections::BTreeMap::new(),
        });
        let arena = alloc::vec![0x3Au8; ARENA_OFFSET_ULTRA_END as usize];
        let dnam = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

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
        let trusted = [("darkglass", oem_pk)];
        let mp_seed = [1u8; 32];
        let mp_pk = RustCryptoProvider::ed25519_pubkey_from_seed(&mp_seed);
        let crypto = RustCryptoProvider::new()
            .with_device_key(dev_id, dev_sk)
            .with_marketplace_pubkey(mp_pk);

        // Invalid seal inputs
        assert!(
            seal_dnam_container(
                &[0u8; 4095],
                core::slice::from_ref(&cert),
                &trusted,
                &mp_seed,
                &[2u8; 32],
                &[4u8; 32],
                &[3u8; 12],
            )
            .is_err()
        );

        let sealed = seal_dnam_container(
            &dnam,
            &[cert],
            &trusted,
            &mp_seed,
            &[2u8; 32],
            &[4u8; 32],
            &[3u8; 12],
        )
        .unwrap();
        assert_eq!(&sealed[0..4], &DNSL_MAGIC);
        assert_eq!(sealed[8], 2); // canonical protection_profile = 2 at byte offset 8
        assert_eq!(sealed.len() % 64, 0);

        // Missing marketplace_pubkey MUST fail closed!
        let missing_mp = RustCryptoProvider::new().with_device_key(dev_id, dev_sk);
        assert!(unseal_dnam_container(&sealed, &missing_mp).is_err());

        // Missing/wrong device X25519 key MUST fail closed!
        let wrong_sk_crypto = RustCryptoProvider::new()
            .with_device_key(dev_id, [99u8; 32])
            .with_marketplace_pubkey(mp_pk);
        assert!(unseal_dnam_container(&sealed, &wrong_sk_crypto).is_err());

        // Valid unseal recovers matching canonical plaintext .dnam
        let unsealed = unseal_dnam_container(&sealed, &crypto).unwrap();
        let view = DnamContainerView::from_bytes(&unsealed, &crypto).unwrap();
        assert_eq!(view.arena_storage.as_bytes(), arena.as_slice());
        assert_eq!(view.header.presets.len(), 1);
        assert_eq!(
            view.header.user_metadata.get("tag"),
            Some(&serde_json::json!("studio"))
        );

        // Truncation and AEAD tag/ciphertext tampering fail closed!
        assert!(unseal_dnam_container(&sealed[..256], &crypto).is_err());
        let mut bad_payload = sealed;
        let last_record_byte = bad_payload.len() - 64;
        bad_payload[last_record_byte] ^= 0xFF;
        assert!(unseal_dnam_container(&bad_payload, &crypto).is_err());
    }
}
