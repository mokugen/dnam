//! Audited RFC 9180 Base HPKE (`DHKEM(X25519, HKDF-SHA256)` + `HKDF-SHA256` + `ChaCha20Poly1305`)
//! and `ChaCha20-Poly1305` record AEAD helpers (`#![forbid(unsafe_code)]`, `ref[sec.drm.hpke-record-encryption]`).

#[cfg(feature = "crypto")]
use alloc::vec::Vec;
#[cfg(feature = "crypto")]
use chacha20poly1305::aead::{Aead, Payload};
#[cfg(feature = "crypto")]
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce};
#[cfg(feature = "crypto")]
use hkdf::Hkdf;
#[cfg(feature = "crypto")]
use sha2::Sha256;
#[cfg(feature = "crypto")]
use x25519_dalek::{PublicKey, StaticSecret};
#[cfg(feature = "crypto")]
use zerocopy::IntoBytes;
#[cfg(feature = "crypto")]
use zerocopy::byteorder::big_endian::U16 as BeU16;
#[cfg(feature = "crypto")]
use zeroize::Zeroizing;

#[cfg(feature = "crypto")]
use crate::provenance::security::HpkeSlotPair;
#[cfg(feature = "crypto")]
use crate::schema::DnamError;

#[cfg(feature = "crypto")]
const SUITE_ID_KEM: [u8; 5] = *b"KEM\x00\x20";
#[cfg(feature = "crypto")]
const SUITE_ID_HPKE: [u8; 10] = [b'H', b'P', b'K', b'E', 0x00, 0x20, 0x00, 0x01, 0x00, 0x03];

#[cfg(feature = "crypto")]
fn labeled_extract(salt: &[u8], suite_id: &[u8], label: &[u8], ikm: &[u8]) -> [u8; 32] {
    let mut full_ikm = Vec::with_capacity(7 + suite_id.len() + label.len() + ikm.len());
    full_ikm.extend_from_slice(b"HPKE-v1");
    full_ikm.extend_from_slice(suite_id);
    full_ikm.extend_from_slice(label);
    full_ikm.extend_from_slice(ikm);
    let (prk, _) = Hkdf::<Sha256>::extract(Some(salt), &full_ikm);
    let mut out = [0u8; 32];
    out.copy_from_slice(prk.as_slice());
    out
}

#[cfg(feature = "crypto")]
fn labeled_expand(
    prk_bytes: &[u8; 32],
    suite_id: &[u8],
    label: &[u8],
    info: &[u8],
    out: &mut [u8],
) -> Result<(), DnamError> {
    let hk = Hkdf::<Sha256>::from_prk(prk_bytes)
        .map_err(|_| DnamError::CryptoVerificationFailed("invalid HKDF PRK length"))?;
    let len_u16 = u16::try_from(out.len())
        .map_err(|_| DnamError::CryptoVerificationFailed("HPKE LabeledExpand length overflow"))?;
    let mut full_info = Vec::with_capacity(2 + 7 + suite_id.len() + label.len() + info.len());
    full_info.extend_from_slice(BeU16::new(len_u16).as_bytes());
    full_info.extend_from_slice(b"HPKE-v1");
    full_info.extend_from_slice(suite_id);
    full_info.extend_from_slice(label);
    full_info.extend_from_slice(info);
    hk.expand(&full_info, out)
        .map_err(|_| DnamError::CryptoVerificationFailed("HPKE LabeledExpand failed"))
}

#[cfg(feature = "crypto")]
fn extract_and_expand_kem(
    dh: &[u8; 32],
    enc: &[u8; 32],
    pk_r: &[u8; 32],
) -> Result<Zeroizing<[u8; 32]>, DnamError> {
    let mut kem_ctx = [0u8; 64];
    kem_ctx[0..32].copy_from_slice(enc);
    kem_ctx[32..64].copy_from_slice(pk_r);
    let eae_prk = Zeroizing::new(labeled_extract(&[], &SUITE_ID_KEM, b"eae_prk", dh));
    let mut shared_secret = Zeroizing::new([0u8; 32]);
    labeled_expand(
        &eae_prk,
        &SUITE_ID_KEM,
        b"shared_secret",
        &kem_ctx,
        &mut *shared_secret,
    )?;
    Ok(shared_secret)
}

#[cfg(feature = "crypto")]
type HpkeKeyAndNonce = (Zeroizing<[u8; 32]>, Zeroizing<[u8; 12]>);

#[cfg(feature = "crypto")]
fn key_schedule_base(shared_secret: &[u8; 32], info: &[u8]) -> Result<HpkeKeyAndNonce, DnamError> {
    let psk_id_hash = labeled_extract(&[], &SUITE_ID_HPKE, b"psk_id_hash", &[]);
    let info_hash = labeled_extract(&[], &SUITE_ID_HPKE, b"info_hash", info);
    let mut ks_context = [0u8; 65];
    ks_context[0] = 0x00; // mode_base
    ks_context[1..33].copy_from_slice(&psk_id_hash);
    ks_context[33..65].copy_from_slice(&info_hash);

    let secret = Zeroizing::new(labeled_extract(
        shared_secret,
        &SUITE_ID_HPKE,
        b"secret",
        &[],
    ));
    let mut key = Zeroizing::new([0u8; 32]);
    let mut nonce = Zeroizing::new([0u8; 12]);
    labeled_expand(&secret, &SUITE_ID_HPKE, b"key", &ks_context, &mut *key)?;
    labeled_expand(
        &secret,
        &SUITE_ID_HPKE,
        b"base_nonce",
        &ks_context,
        &mut *nonce,
    )?;
    Ok((key, nonce))
}

/// Seals a 32-byte CEK via RFC 9180 Base HPKE (`DHKEM(X25519, HKDF-SHA256)` + `ChaCha20Poly1305`),
/// rejecting non-contributory shared secrets fail-closed.
#[cfg(feature = "crypto")]
pub fn hpke_seal_cek_base(
    ephemeral_secret_x25519: &[u8; 32],
    recipient_pk_x25519: &[u8; 32],
    info: &[u8],
    aad: &[u8],
    cek_32: &[u8; 32],
) -> Result<HpkeSlotPair, DnamError> {
    let sk_e = StaticSecret::from(*ephemeral_secret_x25519);
    let enc = PublicKey::from(&sk_e).to_bytes();
    let pk_r = PublicKey::from(*recipient_pk_x25519);
    let dh = sk_e.diffie_hellman(&pk_r);
    if !dh.was_contributory() {
        return Err(DnamError::CryptoVerificationFailed(
            "non-contributory X25519 shared secret during HPKE seal",
        ));
    }
    let shared_secret = extract_and_expand_kem(dh.as_bytes(), &enc, recipient_pk_x25519)?;
    let (key, nonce) = key_schedule_base(&shared_secret, info)?;
    let cipher = ChaCha20Poly1305::new((&*key).into());
    let ct_and_tag = cipher
        .encrypt(Nonce::from_slice(&*nonce), Payload { msg: cek_32, aad })
        .map_err(|_| DnamError::CryptoVerificationFailed("HPKE ChaCha20-Poly1305 seal failed"))?;
    let mut wrapped = [0u8; 48];
    wrapped.copy_from_slice(&ct_and_tag);
    Ok((enc, wrapped))
}

/// Opens a 48-byte wrapped CEK via RFC 9180 Base HPKE (`DHKEM(X25519, HKDF-SHA256)` + `ChaCha20Poly1305`),
/// rejecting non-contributory shared secrets and tag mismatches fail-closed.
#[cfg(feature = "crypto")]
pub fn hpke_open_cek_base(
    recipient_secret_x25519: &[u8; 32],
    enc: &[u8; 32],
    info: &[u8],
    aad: &[u8],
    wrapped_cek_and_tag: &[u8; 48],
) -> Result<Zeroizing<[u8; 32]>, DnamError> {
    let sk_r = StaticSecret::from(*recipient_secret_x25519);
    let pk_r = PublicKey::from(&sk_r).to_bytes();
    let enc_pk = PublicKey::from(*enc);
    let dh = sk_r.diffie_hellman(&enc_pk);
    if !dh.was_contributory() {
        return Err(DnamError::CryptoVerificationFailed(
            "non-contributory X25519 shared secret during HPKE open",
        ));
    }
    let shared_secret = extract_and_expand_kem(dh.as_bytes(), enc, &pk_r)?;
    let (key, nonce) = key_schedule_base(&shared_secret, info)?;
    let cipher = ChaCha20Poly1305::new((&*key).into());
    let pt = cipher
        .decrypt(
            Nonce::from_slice(&*nonce),
            Payload {
                msg: wrapped_cek_and_tag,
                aad,
            },
        )
        .map_err(|_| {
            DnamError::CryptoVerificationFailed("HPKE ChaCha20-Poly1305 tag verification failed")
        })?;
    let mut cek = Zeroizing::new([0u8; 32]);
    cek.copy_from_slice(&pt);
    Ok(cek)
}

/// Encrypts a plaintext record using `ChaCha20-Poly1305` (`ct || tag`, `+16 B`).
#[cfg(feature = "crypto")]
pub fn aead_seal_record(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, DnamError> {
    let cipher = ChaCha20Poly1305::new(key.into());
    cipher
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| DnamError::CryptoVerificationFailed("ChaCha20-Poly1305 record seal failed"))
}

/// Decrypts and authenticates a `ChaCha20-Poly1305` record (`ct || tag`).
#[cfg(feature = "crypto")]
pub fn aead_open_record(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    ciphertext_and_tag: &[u8],
) -> Result<Vec<u8>, DnamError> {
    if ciphertext_and_tag.len() < 16 {
        return Err(DnamError::CryptoVerificationFailed(
            "truncated ChaCha20-Poly1305 ciphertext record",
        ));
    }
    let cipher = ChaCha20Poly1305::new(key.into());
    cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext_and_tag,
                aad,
            },
        )
        .map_err(|_| {
            DnamError::CryptoVerificationFailed(
                "ChaCha20-Poly1305 record authentication tag mismatch",
            )
        })
}

#[cfg(all(test, feature = "crypto"))]
mod tests {
    use super::*;

    #[test]
    fn test_rfc9180_hpke_base_cek_roundtrip_and_non_contributory_rejection() {
        let sk_r = [0x42u8; 32];
        let pk_r = PublicKey::from(&StaticSecret::from(sk_r)).to_bytes();
        let sk_e = [0x24u8; 32];
        let cek = [0xA5u8; 32];

        let (enc, wrapped) = hpke_seal_cek_base(&sk_e, &pk_r, b"info-v1", b"aad-v1", &cek).unwrap();
        let opened = hpke_open_cek_base(&sk_r, &enc, b"info-v1", b"aad-v1", &wrapped).unwrap();
        assert_eq!(*opened, cek);

        // Wrong info, wrong AAD, tampered tag, and non-contributory all-zero key must fail closed
        assert!(hpke_open_cek_base(&sk_r, &enc, b"info-v2", b"aad-v1", &wrapped).is_err());
        assert!(hpke_open_cek_base(&sk_r, &enc, b"info-v1", b"aad-v2", &wrapped).is_err());
        let mut tampered = wrapped;
        tampered[47] ^= 1;
        assert!(hpke_open_cek_base(&sk_r, &enc, b"info-v1", b"aad-v1", &tampered).is_err());
        assert!(hpke_seal_cek_base(&sk_e, &[0u8; 32], b"info", b"aad", &cek).is_err());
        assert!(hpke_open_cek_base(&sk_r, &[0u8; 32], b"info", b"aad", &wrapped).is_err());

        let ct = aead_seal_record(&cek, &[0u8; 12], b"rec-aad", b"hello sona").unwrap();
        assert_eq!(
            aead_open_record(&cek, &[0u8; 12], b"rec-aad", &ct).unwrap(),
            b"hello sona"
        );
        assert!(aead_open_record(&cek, &[0u8; 12], b"bad-aad", &ct).is_err());
        assert!(aead_open_record(&cek, &[0u8; 12], b"rec-aad", &[0u8; 8]).is_err());
    }
}
