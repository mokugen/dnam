//! OEM Device Certificate V2 & RFC 9180 HPKE + `ChaCha20-Poly1305` Record Framing.
//!
//! Canonical Reference: `docs/specs/security/provenance_and_enclave_drm.md` (`ref[sec.drm.oem-certificate]`, `ref[sec.drm.trustzone-flow]`, `ref[sec.drm.hpke-record-encryption]`, `ref[sec.drm.dnsl-public-framing]`, `ref[sec.drm.protected-inner-framing]`, `ref[sec.drm.protection-profiles]`).

pub mod framing;
pub mod wire;

pub use framing::{
    HpkeSlotPair, build_hpke_slot_binding_v1, build_prewrap_plan_v1, build_record_aad_v1,
    build_record_nonce_v1, compute_record_context_sha256, compute_wrap_plan_sha256,
    encode_hpke_slot_vector, encode_recipient_key_ids, encode_record_directory,
    parse_hpke_slot_vector, parse_recipient_key_ids, parse_record_directory,
};
pub use wire::{
    DME1_WIRE_MAGIC, DNPL_WIRE_MAGIC, DNSL_WIRE_MAGIC, DRM_WIRE_VERSION_MAJOR,
    DRM_WIRE_VERSION_MINOR, Dme1HeaderWire, DnplPreambleWire, DnslDirectoryWire, DnslPreambleWire,
    HpkeSlotWire, InnerDirectoryWire, RecordDirEntryWire,
};

use alloc::vec::Vec;

use crate::container::encode_domain_field;
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, reject_if};

/// Maximum plaintext byte length for a single encrypted container record (`(2^32 - 1) * 64`).
pub const MAX_RECORD_PLAINTEXT_BYTES: u64 = ((1u64 << 32) - 1) * 64;
/// Maximum ciphertext byte length (`MAX_RECORD_PLAINTEXT_BYTES + 16`).
pub const MAX_RECORD_CIPHERTEXT_BYTES: u64 = MAX_RECORD_PLAINTEXT_BYTES + 16;
/// RFC 9180 HPKE cipher suite identifier (`DHKEM(X25519, HKDF-SHA256) = 0x0020`,
/// `HKDF-SHA256 = 0x0001`, `ChaCha20Poly1305 = 0x0003`, `ref[sec.drm.hpke-record-encryption]`).
pub const HPKE_SUITE_X25519_SHA256_CHACHA20POLY1305: [u8; 6] = [0x00, 0x20, 0x00, 0x01, 0x00, 0x03];

fn is_canonical_oem_id(s: &str, max_len: usize) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > max_len {
        return false;
    }
    if !(b[0].is_ascii_lowercase() || b[0].is_ascii_digit()) {
        return false;
    }
    b[1..]
        .iter()
        .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'.' | b'_' | b'-'))
}

/// Builds the domain-separated V2 `OemDeviceCertificate` signing message (`"DNAM-OEM-CERT-V2"`,
/// `ref[sec.drm.oem-certificate]`).
#[allow(clippy::too_many_arguments)]
pub fn build_oem_cert_v2_signing_message(
    vendor: &str,
    model: &str,
    device_id_hash: &[u8],
    public_key_x25519: &[u8],
    issuer_public_key_ed25519: &[u8],
    assurance: u8,
    attestation_policy_sha256: &[u8],
    attestation_evidence_sha256: &[u8],
) -> Result<Vec<u8>, DnamError> {
    reject_if(!is_canonical_oem_id(vendor, 32), "invalid vendor")?;
    reject_if(!is_canonical_oem_id(model, 64), "invalid model")?;
    reject_if(
        device_id_hash.len() != 16
            || public_key_x25519.len() != 32
            || issuer_public_key_ed25519.len() != 32
            || attestation_policy_sha256.len() != 32
            || attestation_evidence_sha256.len() != 32
            || !matches!(assurance, 1 | 2),
        "invalid certificate field",
    )?;
    let issuer_sha = sha256_digest(issuer_public_key_ed25519);
    let mut out = Vec::new();
    out.extend_from_slice(&encode_domain_field("domain", b"DNAM-OEM-CERT-V2"));
    out.extend_from_slice(&encode_domain_field("oem_vendor", vendor.as_bytes()));
    out.extend_from_slice(&encode_domain_field("model_code", model.as_bytes()));
    out.extend_from_slice(&encode_domain_field("device_id_hash", device_id_hash));
    out.extend_from_slice(&encode_domain_field("public_key_x25519", public_key_x25519));
    out.extend_from_slice(&encode_domain_field("issuer_key_sha256", &issuer_sha));
    out.extend_from_slice(&encode_domain_field("assurance", &[assurance]));
    out.extend_from_slice(&encode_domain_field(
        "attestation_policy_sha256",
        attestation_policy_sha256,
    ));
    out.extend_from_slice(&encode_domain_field(
        "attestation_evidence_sha256",
        attestation_evidence_sha256,
    ));
    Ok(out)
}

/// Computes `structural_metadata_sha256` under `"DNAM-ENCRYPTED-STRUCTURE-V1"` (`layer` in `0..=1`,
/// `ref[sec.drm.hpke-record-encryption]`).
pub fn compute_encrypted_structural_metadata_sha256(
    layer: u8,
    preamble: &[u8],
    directory: &[u8],
) -> Result<[u8; 32], DnamError> {
    reject_if(
        layer > 1 || preamble.is_empty() || directory.is_empty(),
        "invalid encrypted structure",
    )?;
    let mut msg = Vec::new();
    msg.extend_from_slice(&encode_domain_field(
        "domain",
        b"DNAM-ENCRYPTED-STRUCTURE-V1",
    ));
    msg.extend_from_slice(&encode_domain_field("layer", &[layer]));
    msg.extend_from_slice(&encode_domain_field("preamble", preamble));
    msg.extend_from_slice(&encode_domain_field("directory", directory));
    Ok(sha256_digest(&msg))
}

#[cfg(test)]
mod spec_tests {
    use crate::provenance::{
        build_oem_cert_v2_signing_message, compute_encrypted_structural_metadata_sha256,
    };
    use alloc::vec::Vec;

    /// verify[sec.drm.oem-certificate]
    /// verify[sec.drm.trustzone-flow]
    #[test]
    fn test_oem_certificate_v2_signing_message_and_identifier_validation() {
        let dev_id: Vec<u8> = (0..16u8).collect();
        let pk_x25519: Vec<u8> = (0..32u8).collect();
        let issuer_pk = [0u8; 32];
        let policy = [b'P'; 32];
        let evidence = [b'E'; 32];
        let msg = build_oem_cert_v2_signing_message(
            "acme", "pedal-v1", &dev_id, &pk_x25519, &issuer_pk, 1, &policy, &evidence,
        )
        .unwrap();

        let expected_issuer_sha = [
            0x66, 0x68, 0x7a, 0xad, 0xf8, 0x62, 0xbd, 0x77, 0x6c, 0x8f, 0xc1, 0x8b, 0x8e, 0x9f,
            0x8e, 0x20, 0x08, 0x97, 0x14, 0x85, 0x6e, 0xe2, 0x33, 0xb3, 0x90, 0x2a, 0x59, 0x1d,
            0x0d, 0x5f, 0x29, 0x25,
        ];
        let mut expected = Vec::new();
        expected.extend_from_slice(
            b"\x06\x00\x00\x00domain\x10\x00\x00\x00\x00\x00\x00\x00DNAM-OEM-CERT-V2",
        );
        expected
            .extend_from_slice(b"\x0a\x00\x00\x00oem_vendor\x04\x00\x00\x00\x00\x00\x00\x00acme");
        expected.extend_from_slice(
            b"\x0a\x00\x00\x00model_code\x08\x00\x00\x00\x00\x00\x00\x00pedal-v1",
        );
        expected
            .extend_from_slice(b"\x0e\x00\x00\x00device_id_hash\x10\x00\x00\x00\x00\x00\x00\x00");
        expected.extend_from_slice(&dev_id);
        expected.extend_from_slice(
            b"\x11\x00\x00\x00public_key_x25519\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&pk_x25519);
        expected.extend_from_slice(
            b"\x11\x00\x00\x00issuer_key_sha256\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&expected_issuer_sha);
        expected
            .extend_from_slice(b"\x09\x00\x00\x00assurance\x01\x00\x00\x00\x00\x00\x00\x00\x01");
        expected.extend_from_slice(
            b"\x19\x00\x00\x00attestation_policy_sha256\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&policy);
        expected.extend_from_slice(
            b"\x1b\x00\x00\x00attestation_evidence_sha256\x20\x00\x00\x00\x00\x00\x00\x00",
        );
        expected.extend_from_slice(&evidence);
        assert_eq!(msg, expected);

        for (bad_vendor, bad_model) in [
            ("", "pedal"),
            ("Acme", "pedal"),
            ("acme|pedal", "mk2"),
            ("acme", "Pedal"),
            ("acme", "pedal|mk2"),
        ] {
            assert!(
                build_oem_cert_v2_signing_message(
                    bad_vendor, bad_model, &dev_id, &pk_x25519, &issuer_pk, 1, &policy, &evidence,
                )
                .is_err()
            );
        }
        assert!(
            build_oem_cert_v2_signing_message(
                "acme", "pedal", &dev_id, &pk_x25519, &issuer_pk, 0, &policy, &evidence,
            )
            .is_err()
        );
    }

    #[test]
    fn encrypted_structure_framing_has_distinct_layer_binding() {
        let original = compute_encrypted_structural_metadata_sha256(0, b"p", b"dd").unwrap();
        assert_ne!(
            original,
            compute_encrypted_structural_metadata_sha256(1, b"p", b"dd").unwrap()
        );
        assert_ne!(
            original,
            compute_encrypted_structural_metadata_sha256(0, b"q", b"dd").unwrap()
        );
        assert_ne!(
            original,
            compute_encrypted_structural_metadata_sha256(0, b"p", b"de").unwrap()
        );
        assert_ne!(
            original,
            compute_encrypted_structural_metadata_sha256(0, b"pd", b"d").unwrap()
        );
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn encrypted_structure_framing_has_fixed_bytes_and_distinct_layer_binding() {
        use sha2::{Digest, Sha256};

        let expected = [
            b"\x06\0\0\0domain\x1b\0\0\0\0\0\0\0DNAM-ENCRYPTED-STRUCTURE-V1".as_slice(),
            b"\x05\0\0\0layer\x01\0\0\0\0\0\0\0\0",
            b"\x08\0\0\0preamble\x01\0\0\0\0\0\0\0p",
            b"\x09\0\0\0directory\x02\0\0\0\0\0\0\0dd",
        ]
        .concat();
        let original = compute_encrypted_structural_metadata_sha256(0, b"p", b"dd").unwrap();
        let expected_digest: [u8; 32] = Sha256::digest(expected).into();
        assert_eq!(original, expected_digest);
    }
}
