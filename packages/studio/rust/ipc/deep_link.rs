#![forbid(unsafe_code)]
//! `dnam://` deep-link parsing and Tauri v2 IPC payload isolation validators.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{StudioError, StudioResult};

use super::StudioIpcRequest;

/// Parsed `dnam://` custom URI deep-link route (`tauri-plugin-deep-link` + `tauri-plugin-single-instance`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DnamDeepLinkRoute {
    /// Raw URI string (e.g. `"dnam://models/b7k_ultra"`).
    #[serde(alias = "raw_uri")]
    pub raw_uri: String,
    /// Route kind (`"model"`, `"marketplace"`, `"open"`).
    pub kind: String,
    /// Target identifier or path extracted from the URI.
    pub target: String,
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn decode_percent_pass(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut changed = false;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2]))
        {
            let decoded_byte = (hi << 4) | lo;
            if decoded_byte <= 0x1f || decoded_byte == 0x7f {
                return None;
            }
            out.push(decoded_byte);
            changed = true;
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    if !changed {
        return Some(input.to_string());
    }
    let decoded = String::from_utf8(out).ok()?;
    if decoded
        .chars()
        .any(|c| matches!(c as u32, 0x80..=0x9f) && !input.contains(c))
    {
        return None;
    }
    Some(decoded)
}

fn decoded_contains_backslash(s: &str) -> bool {
    if s.contains('\\') {
        return true;
    }
    let mut cur = s.to_string();
    for _ in 0..4 {
        let Some(next) = decode_percent_pass(&cur) else {
            return true;
        };
        if next.contains('\\') {
            return true;
        }
        if next == cur {
            return false;
        }
        cur = next;
    }
    decode_percent_pass(&cur).is_none_or(|probe| probe != cur)
}

/// Parses a `dnam://` URI into a structured [`DnamDeepLinkRoute`].
pub fn parse_dnam_deep_link(uri: &str) -> StudioResult<DnamDeepLinkRoute> {
    use std::path::Component;
    let trimmed = uri.trim();
    if trimmed.chars().any(char::is_control) || string_contains_unsafe_token(trimmed) {
        return Err(StudioError::Validation(format!(
            "Unsafe traversal or control sequence in dnam:// URI: {trimmed}"
        )));
    }
    let rest = trimmed.strip_prefix("dnam://").ok_or_else(|| {
        StudioError::Validation(format!("Deep link must start with 'dnam://': {trimmed}"))
    })?;
    let rest = rest.trim_matches('/');
    let is_safe_slug = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    };
    if let Some(model_id) = rest.strip_prefix("models/") {
        let cleaned = model_id.trim_matches('/');
        if !is_safe_slug(cleaned) {
            return Err(StudioError::Validation(
                "dnam://models/ requires a valid single-segment model ID".to_string(),
            ));
        }
        return Ok(DnamDeepLinkRoute {
            raw_uri: trimmed.to_string(),
            kind: "model".to_string(),
            target: cleaned.to_string(),
        });
    }
    if let Some(market_id) = rest.strip_prefix("marketplace/") {
        let cleaned = market_id.trim_matches('/');
        if !is_safe_slug(cleaned) {
            return Err(StudioError::Validation(
                "dnam://marketplace/ requires a valid single-segment listing ID".to_string(),
            ));
        }
        return Ok(DnamDeepLinkRoute {
            raw_uri: trimmed.to_string(),
            kind: "marketplace".to_string(),
            target: cleaned.to_string(),
        });
    }
    if let Some(query) = rest.strip_prefix("open?path=") {
        if query.is_empty()
            || decoded_contains_backslash(query)
            || Path::new(query)
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(StudioError::Validation(
                "dnam://open?path= requires a non-empty traversal-safe file path".to_string(),
            ));
        }
        return Ok(DnamDeepLinkRoute {
            raw_uri: trimmed.to_string(),
            kind: "open".to_string(),
            target: query.to_string(),
        });
    }
    Err(StudioError::Validation(format!(
        "Unsupported dnam:// route: {trimmed}"
    )))
}

fn string_contains_unsafe_token(s: &str) -> bool {
    if s.contains("..") || s.contains('\0') {
        return true;
    }
    let mut cur = s.to_string();
    for _ in 0..4 {
        let Some(next) = decode_percent_pass(&cur) else {
            return true;
        };
        if next.contains("..") || next.contains('\0') {
            return true;
        }
        if next == cur {
            return false;
        }
        cur = next;
    }
    decode_percent_pass(&cur).is_none_or(|probe| probe != cur)
}

fn value_contains_unsafe_string(val: &Value) -> bool {
    match val {
        Value::String(s) => string_contains_unsafe_token(s),
        Value::Array(items) => items.iter().any(value_contains_unsafe_string),
        Value::Object(map) => map
            .iter()
            .any(|(k, v)| string_contains_unsafe_token(k) || value_contains_unsafe_string(v)),
        _ => false,
    }
}

/// Validates an incoming [`StudioIpcRequest`] against the Tauri v2 Isolation Pattern invariants.
pub fn validate_isolated_ipc_request(req: &StudioIpcRequest) -> StudioResult<()> {
    if req.command.trim().is_empty() {
        return Err(StudioError::Validation(
            "Tauri Isolation blocked IPC request with empty command".to_string(),
        ));
    }
    if !req.args.is_object() {
        return Err(StudioError::Validation(
            "Tauri Isolation blocked IPC request with non-object args".to_string(),
        ));
    }
    if string_contains_unsafe_token(&req.command) || value_contains_unsafe_string(&req.args) {
        return Err(StudioError::Validation(
            "Tauri Isolation blocked path traversal or null byte in IPC payload".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_ipc_isolation_and_deep_link_roundtrip(
            slug in "[a-zA-Z0-9_-]{1,24}",
            prefix in "[a-zA-Z0-9_]{0,12}",
            suffix in "[a-zA-Z0-9_]{0,12}",
        ) {
            let model_uri = format!("dnam://models/{slug}");
            let parsed_m = parse_dnam_deep_link(&model_uri).expect("valid model uri");
            prop_assert_eq!(parsed_m.kind, "model");
            prop_assert_eq!(parsed_m.target, slug.as_str());

            let market_uri = format!("dnam://marketplace/{slug}");
            let parsed_mk = parse_dnam_deep_link(&market_uri).expect("valid market uri");
            prop_assert_eq!(parsed_mk.kind, "marketplace");
            prop_assert_eq!(parsed_mk.target, slug.as_str());

            let safe_req = StudioIpcRequest {
                command: "set_parameter".to_string(),
                args: json!({ "id": slug, "nested": [prefix.clone(), suffix.clone()] }),
            };
            prop_assert!(validate_isolated_ipc_request(&safe_req).is_ok());

            let traversal_req = StudioIpcRequest {
                command: "get_workspace".to_string(),
                args: json!({ "path": format!("{prefix}..{suffix}") }),
            };
            prop_assert!(validate_isolated_ipc_request(&traversal_req).is_err());

            let null_req = StudioIpcRequest {
                command: "get_workspace".to_string(),
                args: json!({ "nested": [format!("{prefix}\0{suffix}")] }),
            };
            prop_assert!(validate_isolated_ipc_request(&null_req).is_err());
        }
    }

    #[test]
    fn deep_links_and_ipc_isolation_reject_traversal_and_control_bytes() {
        assert_eq!(
            parse_dnam_deep_link("dnam://models/b7k")
                .expect("model")
                .target,
            "b7k"
        );
        assert_eq!(
            parse_dnam_deep_link("dnam://marketplace/item1")
                .expect("market")
                .target,
            "item1"
        );
        assert_eq!(
            parse_dnam_deep_link("dnam://open?path=/tmp/a.dnam")
                .expect("open")
                .target,
            "/tmp/a.dnam"
        );
        assert!(parse_dnam_deep_link("http://bad").is_err());
        assert!(parse_dnam_deep_link("dnam://models/").is_err());
        assert!(parse_dnam_deep_link("dnam://marketplace/").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=").is_err());
        assert!(parse_dnam_deep_link("dnam://models/../../etc/passwd").is_err());
        assert!(parse_dnam_deep_link("dnam://models/foo/bar").is_err());
        assert!(parse_dnam_deep_link("dnam://marketplace/%2e%2e").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=../../secret\0").is_err());
        assert!(parse_dnam_deep_link("dnam://unknown/route").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/.%2e/etc/passwd").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/%252e%252e/etc/passwd").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/bad%5cwin.dnam").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=C:\\win.dnam").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/bad%00null").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/bad%C2%85c1").is_err());
        assert!(parse_dnam_deep_link("dnam://open?path=/tmp/ok%20file.dnam").is_ok());
        assert!(parse_dnam_deep_link("dnam://open?path=C:%252525255cwin.dnam").is_err());

        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "   ".to_string(),
                args: json!({}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"arr": ["../secret"]}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"../bad_key": "ok"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"path": "%2E%2E/secret"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "../evil_cmd".to_string(),
                args: json!({}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!(null),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"path": ".%2e/secret"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({"path": "%252e%252e/secret"}),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "send_notification".to_string(),
                args: json!({"title": "100% Complete", "body": "SNR 78 dB"}),
            })
            .is_ok()
        );
        assert!(
            serde_json::from_value::<StudioIpcRequest>(json!({
                "cmd": "get_telemetry",
                "command": "plugin:shell|execute",
                "args": {}
            }))
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({ "path": "%252525252e%252525252e/etc/passwd" }),
            })
            .is_err()
        );
        assert!(
            validate_isolated_ipc_request(&StudioIpcRequest {
                command: "get_telemetry".to_string(),
                args: json!({ "nul": "safe%2525252500tail" }),
            })
            .is_err()
        );
    }
}
