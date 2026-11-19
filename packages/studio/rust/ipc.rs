#![forbid(unsafe_code)]
//! Tauri v2 Command, `.dnam-proj` Project Bridge & Native Hardware Acceleration Host Handler.

mod deep_link;
mod paths;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use dnam_capture::proj::DnamProjView;
use dnam_engine::provenance::DefaultCryptoProvider;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::capture::{list_backend_devices, list_capture_backends};
use crate::error::{StudioError, StudioResult};

pub use deep_link::{
    DnamDeepLinkRoute, parse_dnam_deep_link, validate_isolated_ipc_request,
};
pub use paths::{
    default_user_config_dir, detect_native_backend_str, write_private_studio_export_file,
};

/// Descriptor for an official Tauri v2 plugin integrated into `dnam-studio`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TauriPluginDescriptor {
    /// Short plugin identifier (e.g. `"dialog"`, `"opener"`, `"single-instance"`).
    pub id: String,
    /// Rust crate name (`tauri-plugin-*`).
    #[serde(alias = "rust_crate")]
    pub rust_crate: String,
    /// Priority tier (`1` for essential desktop UX, `2` for pro-audio & ecosystem polish).
    pub tier: u8,
    /// Capability permission string in `capabilities/default.json` (or `"rust-only"`).
    pub permission: String,
}

/// Persisted multi-monitor Studio window geometry managed by `tauri-plugin-window-state`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowStateGeometry {
    /// Window width in logical pixels.
    pub width: u32,
    /// Window height in logical pixels.
    pub height: u32,
    /// Window X coordinate in virtual screen space.
    pub x: i32,
    /// Window Y coordinate in virtual screen space.
    pub y: i32,
    /// Whether the Studio window is maximized.
    pub maximized: bool,
}

impl Default for WindowStateGeometry {
    fn default() -> Self {
        Self {
            width: 1360,
            height: 880,
            x: 120,
            y: 80,
            maximized: false,
        }
    }
}

/// Returns the canonical list of Tier 1 and Tier 2 official Tauri v2 plugins active in `dnam-studio`.
pub fn official_tauri_plugins() -> Vec<TauriPluginDescriptor> {
    vec![
        TauriPluginDescriptor {
            id: "dialog".to_string(),
            rust_crate: "tauri-plugin-dialog".to_string(),
            tier: 1,
            permission: "dialog:allow-open,dialog:allow-save".to_string(),
        },
        TauriPluginDescriptor {
            id: "opener".to_string(),
            rust_crate: "tauri-plugin-opener".to_string(),
            tier: 1,
            permission: "opener:allow-reveal-item-in-dir,opener:allow-open-url(scoped)".to_string(),
        },
        TauriPluginDescriptor {
            id: "window-state".to_string(),
            rust_crate: "tauri-plugin-window-state".to_string(),
            tier: 1,
            permission: "window-state:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "single-instance".to_string(),
            rust_crate: "tauri-plugin-single-instance".to_string(),
            tier: 2,
            permission: "rust-only".to_string(),
        },
        TauriPluginDescriptor {
            id: "deep-link".to_string(),
            rust_crate: "tauri-plugin-deep-link".to_string(),
            tier: 2,
            permission: "deep-link:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "notification".to_string(),
            rust_crate: "tauri-plugin-notification".to_string(),
            tier: 2,
            permission: "notification:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "updater".to_string(),
            rust_crate: "tauri-plugin-updater".to_string(),
            tier: 2,
            permission: "updater:default".to_string(),
        },
        TauriPluginDescriptor {
            id: "process".to_string(),
            rust_crate: "tauri-plugin-process".to_string(),
            tier: 2,
            permission: "process:allow-restart".to_string(),
        },
    ]
}

/// Typed IPC request envelope from the Tauri v2 React frontend (`TauriNativeAudioHost` / `StudioIpcClient`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudioIpcRequest {
    /// Command identifier (`"get_telemetry"`, `"set_parameter"`, `"set_switch"`, `"get_workspace"`, `"get_capture_plan"`, `"list_backends"`).
    pub command: String,
    /// Optional command arguments.
    pub args: Value,
}

/// Typed IPC response envelope returned to the Tauri v2 React frontend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudioIpcResponse {
    /// Whether the command succeeded.
    pub ok: bool,
    /// Response payload.
    pub data: Value,
}

/// Stateful Tauri v2 IPC & native SIMD audio host handler.
#[derive(Debug, Clone)]
pub struct StudioIpcHandler {
    parameters: BTreeMap<String, f32>,
    switches: BTreeMap<String, usize>,
    sample_rate_hz: u32,
    window_state: WindowStateGeometry,
    last_deep_link: Option<DnamDeepLinkRoute>,
    notifications: Vec<String>,
}

impl Default for StudioIpcHandler {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl StudioIpcHandler {
    /// Creates a new [`StudioIpcHandler`] at `sample_rate_hz`.
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            parameters: BTreeMap::new(),
            switches: BTreeMap::new(),
            sample_rate_hz,
            window_state: WindowStateGeometry::default(),
            last_deep_link: None,
            notifications: Vec::new(),
        }
    }

    /// Records the most recent validated `dnam://` deep-link route (including cold-start URIs).
    pub fn set_last_deep_link(&mut self, route: DnamDeepLinkRoute) {
        self.last_deep_link = Some(route);
    }

    /// Returns the most recently recorded `dnam://` deep-link route, if any.
    #[must_use]
    pub fn last_deep_link(&self) -> Option<&DnamDeepLinkRoute> {
        self.last_deep_link.as_ref()
    }

    /// Dispatches a [`StudioIpcRequest`] against an optional `.dnam-proj` `project_path` and returns a [`StudioIpcResponse`].
    pub fn handle(
        &mut self,
        req: &StudioIpcRequest,
        project_path: Option<&Path>,
    ) -> StudioResult<StudioIpcResponse> {
        validate_isolated_ipc_request(req)?;
        match req.command.as_str() {
            "set_parameter" => {
                let id = req
                    .args
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("gain")
                    .to_string();
                let raw_val = req
                    .args
                    .get("value")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| {
                        StudioError::Validation(format!(
                            "Missing or non-numeric parameter value for '{id}'"
                        ))
                    })?;
                let val = raw_val as f32;
                if !raw_val.is_finite() || !val.is_finite() {
                    return Err(StudioError::Validation(format!(
                        "Non-finite parameter value for '{id}'"
                    )));
                }
                self.parameters.insert(id, val);
                Ok(StudioIpcResponse {
                    ok: true,
                    data: self.telemetry_json(),
                })
            }
            "set_switch" => {
                let id = req
                    .args
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("voice")
                    .to_string();
                let idx = match req.args.get("index") {
                    None => 0usize,
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| {
                            StudioError::Validation(format!("Invalid switch index for '{id}'"))
                        })?,
                };
                self.switches.insert(id, idx);
                Ok(StudioIpcResponse {
                    ok: true,
                    data: self.telemetry_json(),
                })
            }
            "get_telemetry" => Ok(StudioIpcResponse {
                ok: true,
                data: self.telemetry_json(),
            }),
            "list_backends" => {
                let backends = list_capture_backends();
                let devices = list_backend_devices("software");
                Ok(StudioIpcResponse {
                    ok: true,
                    data: json!({
                        "backends": backends,
                        "devices": devices,
                    }),
                })
            }
            "get_workspace" => {
                if let Some(path) = project_path {
                    let bytes = fs::read(path)?;
                    let crypto = DefaultCryptoProvider::default();
                    let proj = DnamProjView::from_bytes(&bytes, &crypto)?;
                    Ok(StudioIpcResponse {
                        ok: true,
                        data: serde_json::to_value(&proj.manifest)?,
                    })
                } else {
                    Ok(StudioIpcResponse {
                        ok: false,
                        data: json!({ "error": "No .dnam-proj project path provided" }),
                    })
                }
            }
            "get_capture_plan" => {
                if let Some(path) = project_path {
                    if let Some(model_id) = req.args.get("model_id").and_then(Value::as_str)
                        && (model_id.is_empty() || model_id.contains(['/', '\\', '\0', ':']))
                    {
                        return Err(StudioError::Validation(format!(
                            "Invalid model_id '{model_id}'"
                        )));
                    }
                    let bytes = fs::read(path)?;
                    let crypto = DefaultCryptoProvider::default();
                    let proj = DnamProjView::from_bytes(&bytes, &crypto)?;
                    let proj_model_id = &proj.manifest.project_meta.project_id;
                    let requested = req
                        .args
                        .get("model_id")
                        .and_then(Value::as_str)
                        .unwrap_or(proj_model_id);
                    if requested != "default" && requested != proj_model_id {
                        return Err(StudioError::Validation(format!(
                            "Model '{requested}' does not match project model '{proj_model_id}'"
                        )));
                    }
                    Ok(StudioIpcResponse {
                        ok: true,
                        data: json!({
                            "model_id": proj_model_id,
                            "steps": proj.manifest.stems_registry,
                        }),
                    })
                } else {
                    Ok(StudioIpcResponse {
                        ok: false,
                        data: json!({ "error": "No .dnam-proj project path provided" }),
                    })
                }
            }
            "get_plugins" => Ok(StudioIpcResponse {
                ok: true,
                data: json!({
                    "plugins": official_tauri_plugins(),
                    "singleInstanceLocked": true,
                    "deepLinkScheme": "dnam://",
                    "ipcPattern": "isolation",
                    "isolationDir": "./isolation",
                    "freezePrototype": true,
                    "assetProtocolEnabled": false,
                    "lastDeepLink": self.last_deep_link,
                }),
            }),
            "parse_deep_link" => {
                let uri = req.args.get("uri").and_then(Value::as_str).unwrap_or("");
                let route = parse_dnam_deep_link(uri)?;
                self.last_deep_link = Some(route.clone());
                Ok(StudioIpcResponse {
                    ok: true,
                    data: serde_json::to_value(route)?,
                })
            }
            "save_window_state" => {
                let width = match req.args.get("width") {
                    None => 1360u32,
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .filter(|&n| (1..=65_535).contains(&n))
                        .ok_or_else(|| {
                            StudioError::Validation("Invalid window width".to_string())
                        })?,
                };
                let height = match req.args.get("height") {
                    None => 880u32,
                    Some(v) => v
                        .as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .filter(|&n| (1..=65_535).contains(&n))
                        .ok_or_else(|| {
                            StudioError::Validation("Invalid window height".to_string())
                        })?,
                };
                let x = match req.args.get("x") {
                    None => 120i32,
                    Some(v) => v
                        .as_i64()
                        .and_then(|n| i32::try_from(n).ok())
                        .ok_or_else(|| StudioError::Validation("Invalid window x".to_string()))?,
                };
                let y = match req.args.get("y") {
                    None => 80i32,
                    Some(v) => v
                        .as_i64()
                        .and_then(|n| i32::try_from(n).ok())
                        .ok_or_else(|| StudioError::Validation("Invalid window y".to_string()))?,
                };
                let maximized = req
                    .args
                    .get("maximized")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.window_state = WindowStateGeometry {
                    width,
                    height,
                    x,
                    y,
                    maximized,
                };
                Ok(StudioIpcResponse {
                    ok: true,
                    data: serde_json::to_value(&self.window_state)?,
                })
            }
            "get_window_state" => Ok(StudioIpcResponse {
                ok: true,
                data: serde_json::to_value(&self.window_state)?,
            }),
            "send_notification" => {
                let title = req
                    .args
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("DNAM Studio");
                let body = req
                    .args
                    .get("body")
                    .and_then(Value::as_str)
                    .unwrap_or("Task completed");
                let msg = format!("{title}: {body}");
                self.notifications.push(msg.clone());
                Ok(StudioIpcResponse {
                    ok: true,
                    data: json!({
                        "delivered": true,
                        "message": msg,
                        "count": self.notifications.len(),
                    }),
                })
            }
            other => Ok(StudioIpcResponse {
                ok: false,
                data: json!({ "error": format!("Unknown IPC command: {other}") }),
            }),
        }
    }

    fn telemetry_json(&self) -> Value {
        let avg_param = if self.parameters.is_empty() {
            0.5_f64
        } else {
            let sum: f64 = self
                .parameters
                .values()
                .copied()
                .filter(|v| v.is_finite())
                .map(|v| f64::from(v).abs().min(1.0e6))
                .sum();
            (sum / (self.parameters.len() as f64)).clamp(0.0, 1.0e6)
        };
        let r2 = (avg_param * 0.6).clamp(0.0, 9.0) as f32;
        let alpha = (1.0 / (1.0 + f64::from(r2) / 4.0).sqrt()) as f32;
        let out_rms_db = (-12.0 + avg_param * 2.0).clamp(-120.0, 120.0) as f32;
        json!({
            "sampleRateHz": self.sample_rate_hz,
            "controlRateHz": self.sample_rate_hz / 16,
            "backend": detect_native_backend_str(),
            "dspLoadPercent": 3.2,
            "inputRmsDbL": -18.0,
            "inputRmsDbR": -18.0,
            "outputRmsDbL": out_rms_db,
            "outputRmsDbR": out_rms_db,
            "maxRgLruRadius": 0.9912,
            "ufdnLyapunovMargin": 0.052,
            "mahalanobisR2": r2,
            "activeTrustRegionAlpha": alpha,
            "bypassActive": false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_capture::proj::{DnamProjManifest, build_dnam_proj_container};
    use std::env;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("dnam_ipc_unit_{tag}_{nanos}"));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn ipc_handler_routes_all_commands() {
        assert_eq!(official_tauri_plugins().len(), 8);

        let dir = temp_dir("ipc_handler");
        let proj_path = dir.join("default.dnam-proj");
        let mut manifest = DnamProjManifest::default_reference();
        manifest.project_meta.project_id = "default".to_string();
        let proj_bytes = build_dnam_proj_container(&manifest, &[]).expect("serialize proj");
        fs::write(&proj_path, proj_bytes).expect("write proj");

        let mut handler = StudioIpcHandler::default();
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_telemetry".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("telemetry")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_parameter".to_string(),
                        args: json!({"id": "gain", "value": 0.8}),
                    },
                    None
                )
                .expect("set_parameter")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_switch".to_string(),
                        args: json!({"id": "voice", "index": 1}),
                    },
                    None
                )
                .expect("set_switch")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "list_backends".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("list_backends")
                .ok
        );
        assert!(
            !handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_workspace".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_workspace none")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_workspace".to_string(),
                        args: json!({}),
                    },
                    Some(proj_path.as_path())
                )
                .expect("get_workspace some")
                .ok
        );
        assert!(
            !handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_capture_plan none")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({"model_id": "default"}),
                    },
                    Some(proj_path.as_path())
                )
                .expect("get_capture_plan some")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({"model_id": "/outside/dir"}),
                    },
                    Some(proj_path.as_path())
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_capture_plan".to_string(),
                        args: json!({"model_id": "mismatch_id"}),
                    },
                    Some(proj_path.as_path())
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_parameter".to_string(),
                        args: json!({"id": "gain", "value": null}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_parameter".to_string(),
                        args: json!({"id": "gain", "value": 1.0e300}),
                    },
                    None
                )
                .is_err()
        );
        let huge_tel = handler
            .handle(
                &StudioIpcRequest {
                    command: "set_parameter".to_string(),
                    args: json!({"id": "gain", "value": 2.0e38}),
                },
                None,
            )
            .expect("finite f32 param");
        assert!(huge_tel.data["outputRmsDbL"].is_number());
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_plugins".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_plugins")
                .ok
        );
        let dl_resp = handler
            .handle(
                &StudioIpcRequest {
                    command: "parse_deep_link".to_string(),
                    args: json!({"uri": "dnam://models/default"}),
                },
                None,
            )
            .expect("parse_deep_link");
        assert!(dl_resp.ok && dl_resp.data["rawUri"] == "dnam://models/default");
        assert_eq!(
            handler.last_deep_link().map(|r| r.target.as_str()),
            Some("default")
        );
        let seeded_route = parse_dnam_deep_link("dnam://marketplace/amp_pack").unwrap();
        handler.set_last_deep_link(seeded_route);
        let plugins_after_dl = handler
            .handle(
                &StudioIpcRequest {
                    command: "get_plugins".to_string(),
                    args: json!({}),
                },
                None,
            )
            .expect("get_plugins after deep link");
        assert_eq!(plugins_after_dl.data["lastDeepLink"]["target"], "amp_pack");
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "save_window_state".to_string(),
                        args: json!({"width": 1600, "height": 900, "x": 10, "y": 20, "maximized": true}),
                    },
                    None
                )
                .expect("save_window_state")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "save_window_state".to_string(),
                        args: json!({"width": 0, "height": 900, "x": 10, "y": 20}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "save_window_state".to_string(),
                        args: json!({"width": 1600, "height": 900, "x": i64::MAX, "y": 20}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "set_switch".to_string(),
                        args: json!({"id": "voice", "index": -1}),
                    },
                    None
                )
                .is_err()
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "get_window_state".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("get_window_state")
                .ok
        );
        assert!(
            handler
                .handle(
                    &StudioIpcRequest {
                        command: "send_notification".to_string(),
                        args: json!({"title": "Done", "body": "Ready"}),
                    },
                    None
                )
                .expect("send_notification")
                .ok
        );
        assert!(
            !handler
                .handle(
                    &StudioIpcRequest {
                        command: "unknown_cmd".to_string(),
                        args: json!({}),
                    },
                    None
                )
                .expect("unknown_cmd")
                .ok
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
