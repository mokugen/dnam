//! RFC 8785 JCS canonicalization and preset-safe `immutable_header_sha256` projection.

use alloc::string::String;
use alloc::vec::Vec;

use super::super::sha256::sha256_digest;
use crate::schema::{DnamError, DnamJsonHeader, SchemaVersion};

/// Recursively canonicalizes a `serde_json::Value` per RFC 8785 JSON Canonicalization Scheme (JCS).
pub(crate) fn canonicalize_jcs_value(val: &serde_json::Value, out: &mut String) {
    match val {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64()
                && f == 0.0
            {
                out.push('0');
                return;
            }
            out.push_str(&serde_json::to_string(n).unwrap_or_default());
        }
        serde_json::Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_default());
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonicalize_jcs_value(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            // Sort keys by RFC 8785 UTF-16 code-unit lexicographical order
            let mut sorted: Vec<(&str, &serde_json::Value)> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            sorted.sort_by(|(a, _), (b, _)| a.encode_utf16().cmp(b.encode_utf16()));
            out.push('{');
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                canonicalize_jcs_value(v, out);
            }
            out.push('}');
        }
    }
}

/// Projects a serialized [`DnamJsonHeader`] `serde_json::Value` onto the canonical immutable
/// field set for `schema_version`, stripping mutable DAW/user fields and any additive fields
/// introduced in minor schema versions newer than `schema_version`.
pub fn project_immutable_header_for_schema_version(
    val: &mut serde_json::Value,
    schema_version: SchemaVersion,
) -> Result<(), DnamError> {
    schema_version.ensure_supported()?;

    if let Some(obj) = val.as_object_mut() {
        obj.remove("presets");
        obj.remove("user_metadata");

        if let Some(meta) = obj.get_mut("metadata").and_then(|m| m.as_object_mut()) {
            meta.remove("signatures");
        }
        if let Some(tiers) = obj
            .get_mut("slimmable_tiers")
            .and_then(|t| t.as_object_mut())
        {
            for tier_key in ["lite", "full", "pro", "ultra"] {
                if let Some(t_obj) = tiers.get_mut(tier_key).and_then(|t| t.as_object_mut()) {
                    t_obj.remove("prefix_crc32c");
                }
            }
        }
        if let Some(ctrls) = obj
            .get_mut("controls_schema")
            .and_then(|c| c.as_array_mut())
        {
            for c_item in ctrls {
                if let Some(c_obj) = c_item.as_object_mut() {
                    c_obj.remove("user_metadata");
                }
            }
        }
        if let Some(stems) = obj.get_mut("stems_registry").and_then(|s| s.as_array_mut()) {
            for s_item in stems {
                if let Some(s_obj) = s_item.as_object_mut() {
                    s_obj.remove("user_metadata");
                }
            }
        }

        // Future minor-version projection hooks (e.g. `if schema_version.minor < 1 { ... }`)
        // strip additive `#[serde(default)]` fields introduced after `schema_version` so
        // containers signed under an earlier `1.x` minor version retain their exact JCS digest.
    }

    Ok(())
}

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` digest over a [`DnamJsonHeader`]
/// projected to a specific [`SchemaVersion`].
pub fn compute_immutable_header_sha256_for_version(
    header: &DnamJsonHeader,
    schema_version: SchemaVersion,
) -> Result<[u8; 32], DnamError> {
    let hdr_bytes = header
        .to_json_vec()
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    let mut val: serde_json::Value = serde_json::from_slice(&hdr_bytes)
        .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
    project_immutable_header_for_schema_version(&mut val, schema_version)?;

    let mut canonical = String::with_capacity(2048);
    canonicalize_jcs_value(&val, &mut canonical);
    Ok(sha256_digest(canonical.as_bytes()))
}

/// Computes the preset-safe RFC 8785 JCS `immutable_header_sha256` digest over a [`DnamJsonHeader`]
/// using the header's declared [`SchemaVersion`].
///
/// Excludes mutable fields (`presets`, `user_metadata`, `metadata.signatures`,
/// `slimmable_tiers.*.prefix_crc32c`, `controls_schema[*].user_metadata`, and
/// `stems_registry[*].user_metadata`) so users can save DAW presets and tags in-place
/// (`0x40..0x1000`) without invalidating `creator_signature` or `marketplace_receipt`.
pub fn compute_immutable_header_sha256(header: &DnamJsonHeader) -> Result<[u8; 32], DnamError> {
    let schema_version = header.schema_version()?;
    compute_immutable_header_sha256_for_version(header, schema_version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_immutable_header_strips_controls_and_stems_user_metadata() {
        let mut val = serde_json::json!({
            "presets": [],
            "user_metadata": {"a": -0.0},
            "controls_schema": [{"id": "drive", "user_metadata": {"note": 1}}],
            "stems_registry": [{"id": "s1", "user_metadata": {"tag": "warm"}}],
        });
        project_immutable_header_for_schema_version(&mut val, SchemaVersion::V2_0_0).unwrap();
        assert!(val.get("presets").is_none());
        assert!(val.get("user_metadata").is_none());
        assert!(val["controls_schema"][0].get("user_metadata").is_none());
        assert!(val["stems_registry"][0].get("user_metadata").is_none());

        let mut canonical = String::new();
        canonicalize_jcs_value(
            &serde_json::json!({"z": -0.0, "a": [null, true, false]}),
            &mut canonical,
        );
        assert_eq!(canonical, r#"{"a":[null,true,false],"z":0}"#);
    }
}
