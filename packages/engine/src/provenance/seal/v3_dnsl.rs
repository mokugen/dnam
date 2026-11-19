//! Canonical `v3.0` `.dnam-seal` (`b"DNSL"` + `DNPL` + `DAI1` + `DSA1` + `DME1`) HPKE &
//! `ChaCha20-Poly1305` AEAD sealing and unsealing (`#![forbid(unsafe_code)]`,
//! `ref[sec.drm.hpke-record-encryption]`, `ref[sec.drm.dnsl-public-framing]`,
//! `ref[sec.drm.protected-inner-framing]`, `ref[sec.drm.dme1-mutable-suffix]`).

#[cfg(feature = "crypto")]
pub mod seal_encode;
pub mod seal_unseal;

#[cfg(feature = "crypto")]
pub use seal_encode::seal_dnam_container;
pub use seal_unseal::{
    parse_and_decrypt_dme1_frame, unseal_dnam_container, unseal_dnsl_up_to_tier,
};

use alloc::vec::Vec;
use core::mem::{align_of, offset_of, size_of};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::container::{DnamDirectoryWire, DnamPreambleWire, validate_mutable_json_schema};
use crate::provenance::security::Dai1Envelope;
use crate::schema::{DnamError, DnamJsonHeader};
use crate::types::Tier;

pub(crate) const DSA1_MAGIC: [u8; 4] = *b"DSA1";
pub(crate) const MAX_SEAL_RECIPIENTS: usize = 16;

/// Public `DSA1` outer manifest fixed 40-byte header (`ref[sec.drm.dnsl-public-framing]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Dsa1ManifestHeaderWire {
    pub magic: [u8; 4],
    pub original_maximum_tier: u8,
    pub reserved: [u8; 3],
    pub outer_signer_public_key: [u8; 32],
}

const _: () = assert!(size_of::<Dsa1ManifestHeaderWire>() == 40);
const _: () = assert!(align_of::<Dsa1ManifestHeaderWire>() == 1);
const _: () = assert!(offset_of!(Dsa1ManifestHeaderWire, outer_signer_public_key) == 8);

/// Decrypted and cryptographically verified payload extracted from a `v3.0` `.dnam-seal` (`b"DNSL"`).
#[derive(Debug)]
pub struct UnsealedDnslInner {
    pub maximum_tier: u8,
    pub retained_tier: Tier,
    pub logical_dnam_preamble: DnamPreambleWire,
    pub logical_dnam_dir: DnamDirectoryWire,
    pub header_json_bytes: Vec<u8>,
    pub shared_bytes: Vec<u8>,
    pub auth_bytes: Vec<u8>,
    pub arena_bytes: Vec<u8>,
    pub tier_ends: [usize; 4],
    pub mutable_json: Option<serde_json::Value>,
    pub dai1: Dai1Envelope,
}

#[inline]
pub(crate) const fn align64_usize(len: usize) -> usize {
    (len + 63) & !63
}

pub(crate) fn pad_to_64(buf: &mut Vec<u8>) {
    let target = align64_usize(buf.len());
    buf.resize(target, 0);
}

pub(crate) fn extract_section_slice(buf: &[u8], offset: u64, len: u64) -> Result<&[u8], DnamError> {
    let off = usize::try_from(offset)
        .map_err(|_| DnamError::CryptoVerificationFailed("DNSL offset overflow"))?;
    let l = usize::try_from(len)
        .map_err(|_| DnamError::CryptoVerificationFailed("DNSL length overflow"))?;
    let end = off
        .checked_add(l)
        .ok_or(DnamError::CryptoVerificationFailed("DNSL range overflow"))?;
    buf.get(off..end).ok_or(DnamError::BufferTooSmall {
        expected: end,
        actual: buf.len(),
    })
}

pub(crate) fn extract_mutable_json_bytes(header: &DnamJsonHeader) -> Result<Vec<u8>, DnamError> {
    let mut dmu_obj = serde_json::Map::new();
    if !header.presets.is_empty() {
        dmu_obj.insert(
            "presets".into(),
            serde_json::to_value(&header.presets)
                .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?,
        );
    }
    if !header.user_metadata.is_empty() {
        dmu_obj.insert(
            "user_metadata".into(),
            serde_json::to_value(&header.user_metadata)
                .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?,
        );
    }
    let mut ctrl_um = serde_json::Map::new();
    for ctrl in &header.controls_schema {
        if !ctrl.user_metadata.is_empty() {
            ctrl_um.insert(
                ctrl.id.clone(),
                serde_json::to_value(&ctrl.user_metadata)
                    .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?,
            );
        }
    }
    if !ctrl_um.is_empty() {
        dmu_obj.insert(
            "control_user_metadata".into(),
            serde_json::Value::Object(ctrl_um),
        );
    }
    if !header.stem_user_metadata.is_empty() {
        dmu_obj.insert(
            "stem_user_metadata".into(),
            serde_json::to_value(&header.stem_user_metadata)
                .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?,
        );
    }
    let doc = serde_json::Value::Object(dmu_obj);
    validate_mutable_json_schema(&doc)?;
    serde_json::to_vec(&doc).map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))
}

pub(crate) fn merge_mutable_fields(
    header: &mut DnamJsonHeader,
    dmu_val: &serde_json::Value,
) -> Result<(), DnamError> {
    validate_mutable_json_schema(dmu_val)?;
    if let Some(presets_val) = dmu_val.get("presets") {
        header.presets = serde_json::from_value(presets_val.clone())
            .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
    }
    if let Some(um_val) = dmu_val.get("user_metadata") {
        header.user_metadata = serde_json::from_value(um_val.clone())
            .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
    }
    if let Some(ctrl_um_obj) = dmu_val
        .get("control_user_metadata")
        .and_then(|v| v.as_object())
    {
        for ctrl in &mut header.controls_schema {
            if let Some(entry_val) = ctrl_um_obj.get(&ctrl.id) {
                ctrl.user_metadata = serde_json::from_value(entry_val.clone())
                    .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
            }
        }
    }
    if let Some(stem_um_val) = dmu_val.get("stem_user_metadata") {
        header.stem_user_metadata = serde_json::from_value(stem_um_val.clone())
            .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
    }
    Ok(())
}
