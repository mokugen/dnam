//! Unknown-key rejection and structural schema parity checks for [`super::DnamJsonHeader`].

use alloc::vec::Vec;

use serde_json::Value;

use crate::schema::DnamError;

const USER_DICT_KEYS: &[&str] = &[
    "user_metadata",
    "stem_user_metadata",
    "controls",
    "controls_meta",
    "reference_samples",
    "scale_labels",
    "option_labels",
];

const REJECTED_LEGACY_KEYS: &[&str] = &[
    "io_router",
    "sub_crossover_hz",
    "supports_freeze",
    "freeze_mutes_dry",
    "freeze_latch",
];

fn is_skipped_optional_field(parent_key: Option<&str>, key: &str, v_in: &Value) -> bool {
    match (parent_key, key) {
        (None, "stems_registry" | "audit_report")
        | (Some("metadata"), "distribution_policy")
        | (Some("signatures"), "tier_arena_sha256")
        | (Some("marketplace_receipt"), "buyer_tier_arena_sha256")
        | (Some("controls_schema"), "calibration" | "represented_switch_options")
        | (Some("calibration"), "lobatto_physical_anchors" | "tempo_sync_default_subdivision")
        | (Some("stems_registry"), "extracted_physics")
        | (
            Some("tensor_directory_entry" | "g_audio_uplo_u_f64" | "c_audio_tier_f64"),
            "packing" | "feature_dim" | "packed_len" | "shape",
        ) => v_in.is_null(),
        (None, "stem_user_metadata") => {
            v_in.as_object().is_some_and(serde_json::Map::is_empty) || v_in.is_null()
        }
        (Some("controls_schema"), "conditional_bounds") => {
            v_in.as_array().is_some_and(Vec::is_empty) || v_in.is_null()
        }
        (Some("calibration"), "conditional_physical_anchors") => {
            v_in.as_array().is_some_and(Vec::is_empty) || v_in.is_null()
        }
        (None, "delay_histories") => v_in.as_array().is_some_and(Vec::is_empty) || v_in.is_null(),
        _ => false,
    }
}

/// Verifies that no unknown JSON schema fields exist in `input` relative to `normalized`.
pub fn check_no_unknown_header_keys(
    input: &Value,
    normalized: &Value,
    parent_key: Option<&str>,
) -> Result<(), DnamError> {
    if let Some(pk) = parent_key
        && USER_DICT_KEYS.contains(&pk)
    {
        return Ok(());
    }
    match (input, normalized) {
        (Value::Object(in_map), Value::Object(norm_map)) => {
            if parent_key == Some("tensor_directory") {
                for (k, v_in) in in_map {
                    if let Some(v_norm) = norm_map.get(k) {
                        check_no_unknown_header_keys(v_in, v_norm, Some("tensor_directory_entry"))?;
                    }
                }
                return Ok(());
            }
            for (k, v_in) in in_map {
                if REJECTED_LEGACY_KEYS.contains(&k.as_str()) {
                    return Err(DnamError::SchemaValidationError(alloc::format!(
                        "Legacy v3 schema field '{k}' is rejected"
                    )));
                }
                if let Some(v_norm) = norm_map.get(k) {
                    check_no_unknown_header_keys(v_in, v_norm, Some(k.as_str()))?;
                } else if !is_skipped_optional_field(parent_key, k.as_str(), v_in) {
                    return Err(DnamError::SchemaValidationError(alloc::format!(
                        "Unknown schema field '{k}'"
                    )));
                }
            }
            Ok(())
        }
        (Value::Array(in_arr), Value::Array(norm_arr)) => {
            for (v_in, v_norm) in in_arr.iter().zip(norm_arr.iter()) {
                check_no_unknown_header_keys(v_in, v_norm, parent_key)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_conditional_physical_anchors_unknown_keys_and_empty_array() {
        let in_empty = json!({ "calibration": { "conditional_physical_anchors": [] } });
        let norm_empty = json!({ "calibration": {} });
        assert!(check_no_unknown_header_keys(&in_empty, &norm_empty, None).is_ok());

        let in_null = json!({ "calibration": { "conditional_physical_anchors": null } });
        assert!(check_no_unknown_header_keys(&in_null, &norm_empty, None).is_ok());

        let in_bad_curve = json!({
            "calibration": {
                "conditional_physical_anchors": [
                    { "when": [{ "control_id": "sw", "option_id": "a" }], "anchors": [1.0, 2.0, 3.0], "extra": 1 }
                ]
            }
        });
        let norm_curve = json!({
            "calibration": {
                "conditional_physical_anchors": [
                    { "when": [{ "control_id": "sw", "option_id": "a" }], "anchors": [1.0, 2.0, 3.0] }
                ]
            }
        });
        assert!(check_no_unknown_header_keys(&in_bad_curve, &norm_curve, None).is_err());
        assert!(check_no_unknown_header_keys(&norm_curve, &norm_curve, None).is_ok());
    }

    #[test]
    fn test_rejected_legacy_keys_fail_closed() {
        let norm = json!({});
        for legacy_key in REJECTED_LEGACY_KEYS {
            let input = json!({ (*legacy_key): true });
            assert!(check_no_unknown_header_keys(&input, &norm, None).is_err());
        }
    }
}
