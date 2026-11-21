//! Mutable `DMU1` / `DME1` trailer framing and strict JSON validation
//! (`ref[container.dnam.mutable-frame]`, `ref[container.dnam.mutable-json]`).

use alloc::collections::BTreeSet;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use serde_json::Value;
use zerocopy::byteorder::little_endian::{U32, U64};
use zerocopy::{FromBytes, IntoBytes};

use super::super::jcs::parse_strict_json_object_bytes;
use super::super::structs::{DMU1_HEADER_SIZE, Dmu1HeaderWire};
use super::ContainerWireError;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};
use dnam_vm::math::{crc32c, crc32c_slice};

/// Builds a `DMU1` open mutable frame of total capacity `capacity >= 16 + payload.len()`
/// (`ref[container.dnam.mutable-frame]`).
pub fn build_dmu1_mutable_frame(payload: &[u8], capacity: usize) -> Result<Vec<u8>, DnamError> {
    reject_if(
        payload.len() < 2 || capacity < DMU1_HEADER_SIZE + payload.len(),
        "invalid DMU1 payload or capacity",
    )?;
    let len_le = U64::new(payload.len() as u64);
    let mut crc_input = Vec::with_capacity(12 + payload.len());
    crc_input.extend_from_slice(b"DMU1");
    crc_input.extend_from_slice(len_le.as_bytes());
    crc_input.extend_from_slice(payload);
    let checksum = crc32c(&crc_input);

    let mut frame = vec![0u8; capacity];
    let hdr = Dmu1HeaderWire {
        magic: *b"DMU1",
        crc32c: U32::new(checksum),
        json_len: len_le,
    };
    let _ = hdr.write_to_prefix(&mut frame);
    frame[DMU1_HEADER_SIZE..DMU1_HEADER_SIZE + payload.len()].copy_from_slice(payload);
    Ok(frame)
}

/// Zero-drop-glue (`Copy` error) `DMU1` mutable frame validator for real-time paths.
pub(crate) fn validate_dmu1_frame_rt(frame: &[u8]) -> Result<(), ContainerWireError> {
    let (hdr, body) = Dmu1HeaderWire::read_from_prefix(frame)
        .map_err(|_| ContainerWireError::SchemaStatic("invalid DMU1 magic or short frame"))?;
    if &hdr.magic != b"DMU1" || body.len() < 2 {
        return Err(ContainerWireError::SchemaStatic(
            "invalid DMU1 magic or short frame",
        ));
    }
    let stored_crc = hdr.crc32c.get();
    let n_len = hdr.json_len.get() as usize;
    if n_len < 2 || n_len > body.len() {
        return Err(ContainerWireError::SchemaStatic(
            "invalid DMU1 JSON payload length",
        ));
    }
    let (payload, padding) = body.split_at(n_len);
    if padding.iter().any(|&b| b != 0) {
        return Err(ContainerWireError::SchemaStatic(
            "non-zero DMU1 padding bytes",
        ));
    }
    let c0 = crc32c_slice(0, b"DMU1");
    let c1 = crc32c_slice(c0, hdr.json_len.as_bytes());
    let calc_crc = crc32c_slice(c1, payload);
    if stored_crc != calc_crc {
        return Err(ContainerWireError::Crc32cMismatch {
            region: "DMU1",
            expected: stored_crc,
            computed: calc_crc,
        });
    }
    if payload.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(ContainerWireError::SchemaStatic("UTF-8 BOM is forbidden"));
    }
    let text = core::str::from_utf8(payload)
        .map_err(|_| ContainerWireError::SchemaStatic("invalid UTF-8 JSON"))?;
    let trimmed = text.trim_ascii();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err(ContainerWireError::SchemaStatic(
            "DMU1 payload must be a JSON object",
        ));
    }
    Ok(())
}

/// Validates a `DMU1` mutable frame (CRC32C, zero padding, and UTF-8 `{...}` framing) without heap allocation.
pub fn validate_dmu1_frame_no_alloc(frame: &[u8]) -> Result<(), DnamError> {
    validate_dmu1_frame_rt(frame).map_err(ContainerWireError::into_dnam_error)
}

/// Validates a `DME1` encrypted mutable frame without heap allocation (`ref[sec.drm.protected-inner-framing]`).
pub fn validate_dme1_frame_no_alloc(frame: &[u8], expected_profile: u8) -> Result<(), DnamError> {
    let wire = crate::provenance::Dme1HeaderWire::from_bytes(frame)?;
    reject_if(
        wire.protection_profile != expected_profile,
        "DME1 protection_profile mismatch",
    )?;
    let slots_bytes = (wire.recipient_count.get() as usize)
        .checked_mul(80)
        .or_schema_err("DME1 slot count overflow")?;
    let ct_bytes = usize::try_from(wire.ciphertext_length.get())
        .ok()
        .or_schema_err("DME1 ciphertext_length overflow")?;
    let used = 20usize
        .checked_add(slots_bytes)
        .and_then(|v| v.checked_add(ct_bytes))
        .or_schema_err("DME1 length overflow")?;
    reject_if(used > frame.len(), "truncated DME1 frame")?;
    reject_if(
        frame[used..].iter().any(|&b| b != 0),
        "non-zero DME1 padding bytes",
    )?;
    Ok(())
}

/// Validates a `DMU1` mutable frame (CRC32C, zero padding, and strict RFC 8259 UTF-8 JSON object without
/// BOM or duplicate keys) and returns the parsed JSON `Value` (`ref[container.dnam.mutable-frame]`).
pub fn parse_and_validate_dmu1_frame(frame: &[u8]) -> Result<Value, DnamError> {
    validate_dmu1_frame_no_alloc(frame)?;
    let (hdr, body) = Dmu1HeaderWire::read_from_prefix(frame)
        .ok()
        .or_schema_err("invalid DMU1 header")?;
    let n_len = hdr.json_len.get() as usize;
    parse_strict_json_object_bytes(&body[..n_len])
}

/// Validates a mutable JSON document against the normative schema (`version`, `presets`,
/// `user_metadata`, `control_user_metadata`, `stem_user_metadata`, `ref[container.dnam.mutable-json]`).
pub fn validate_mutable_json_schema(doc: &Value) -> Result<(), DnamError> {
    let obj = doc
        .as_object()
        .or_schema_err("mutable JSON must be an object")?;
    for key in obj.keys() {
        if !matches!(
            key.as_str(),
            "version"
                | "presets"
                | "user_metadata"
                | "control_user_metadata"
                | "stem_user_metadata"
        ) {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "unknown mutable member: {key}"
            )));
        }
    }
    if let Some(ver) = obj.get("version")
        && ver.as_i64() != Some(1)
        && ver.as_u64() != Some(1)
    {
        return Err(DnamError::SchemaValidationError(
            "unsupported version".to_string(),
        ));
    }
    for meta_key in [
        "user_metadata",
        "control_user_metadata",
        "stem_user_metadata",
    ] {
        if let Some(v) = obj.get(meta_key) {
            let map = v.as_object().or_schema_err("invalid metadata object")?;
            if meta_key != "user_metadata" {
                for item in map.values() {
                    reject_if(!item.is_object(), "invalid keyed metadata")?;
                }
            }
        }
    }
    if let Some(presets_val) = obj.get("presets") {
        let presets = presets_val.as_array().or_schema_err("invalid presets")?;
        let mut seen_ids = BTreeSet::new();
        for preset_val in presets {
            let p_obj = preset_val.as_object().or_schema_err("invalid preset")?;
            for k in p_obj.keys() {
                reject_if(
                    !matches!(
                        k.as_str(),
                        "id" | "name" | "description" | "controls" | "user_metadata"
                    ),
                    "invalid preset",
                )?;
            }
            let id = p_obj.get("id").and_then(Value::as_str).unwrap_or("");
            reject_if(
                id.is_empty() || !seen_ids.insert(id.to_string()),
                "invalid preset id",
            )?;
            let name = p_obj.get("name").and_then(Value::as_str).unwrap_or("");
            reject_if(name.is_empty(), "invalid preset name")?;
            if let Some(desc) = p_obj.get("description")
                && !desc.is_null()
                && !desc.is_string()
            {
                return Err(DnamError::SchemaValidationError(
                    "invalid description".to_string(),
                ));
            }
            if let Some(um) = p_obj.get("user_metadata")
                && !um.is_object()
            {
                return Err(DnamError::SchemaValidationError(
                    "invalid preset metadata".to_string(),
                ));
            }
            let controls = p_obj
                .get("controls")
                .and_then(Value::as_object)
                .or_schema_err("invalid control map")?;
            for cv in controls.values() {
                let valid_val =
                    cv.is_string() || (cv.is_number() && cv.as_f64().is_some_and(f64::is_finite));
                reject_if(!valid_val, "invalid control value")?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::Dme1HeaderWire;
    use serde_json::json;

    #[test]
    fn test_dmu1_dme1_framing_and_mutable_schema_error_paths() {
        assert!(build_dmu1_mutable_frame(b"{", 32).is_err());
        assert!(build_dmu1_mutable_frame(b"{}", 16).is_err());

        assert!(validate_dmu1_frame_no_alloc(b"DMU1").is_err());
        assert!(validate_dmu1_frame_no_alloc(b"XXXX000000000000{}").is_err());
        assert!(validate_dmu1_frame_no_alloc(b"DMU1000000000000{").is_err());

        let mut bad_len_small = [0u8; 20];
        bad_len_small[..4].copy_from_slice(b"DMU1");
        bad_len_small[8..16].copy_from_slice(U64::new(1).as_bytes());
        assert!(validate_dmu1_frame_no_alloc(&bad_len_small).is_err());
        bad_len_small[8..16].copy_from_slice(U64::new(99).as_bytes());
        assert!(validate_dmu1_frame_no_alloc(&bad_len_small).is_err());

        let bom_frame = build_dmu1_mutable_frame(b"\xEF\xBB\xBF{}", 24).unwrap();
        assert!(validate_dmu1_frame_no_alloc(&bom_frame).is_err());
        let bad_utf8 = build_dmu1_mutable_frame(b"\xFF\xFE{}", 24).unwrap();
        assert!(validate_dmu1_frame_no_alloc(&bad_utf8).is_err());
        let array_frame = build_dmu1_mutable_frame(b"[]", 24).unwrap();
        assert!(validate_dmu1_frame_no_alloc(&array_frame).is_err());

        let mut dme1 = vec![0u8; 192];
        let dme_hdr = Dme1HeaderWire::new(1, 1, 70).unwrap().to_bytes();
        dme1[..20].copy_from_slice(&dme_hdr);
        validate_dme1_frame_no_alloc(&dme1, 1).unwrap();
        assert!(validate_dme1_frame_no_alloc(&dme1, 2).is_err());
        assert!(validate_dme1_frame_no_alloc(&dme1[..100], 1).is_err());
        let mut dme1_pad = dme1.clone();
        dme1_pad[180] = 1;
        assert!(validate_dme1_frame_no_alloc(&dme1_pad, 1).is_err());
        assert!(validate_dme1_frame_no_alloc(&[0u8; 32], 1).is_err());

        let bad_desc = json!({
            "presets": [{"id": "a", "name": "A", "description": 123, "controls": {}}]
        });
        assert!(validate_mutable_json_schema(&bad_desc).is_err());
        let bad_um = json!({
            "presets": [{"id": "a", "name": "A", "user_metadata": [], "controls": {}}]
        });
        assert!(validate_mutable_json_schema(&bad_um).is_err());
    }
}
