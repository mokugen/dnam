//! Canonical `.dnam` UTF-8 JSON Header (`DnamJsonHeader`), subsystem configurations,
//! tensor directory (`TensorSliceEntry`), and binary clamp compiler.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

pub use crate::container::wire::DelayHistoryEntry;

use super::controls::{ControlKind, ControlSchemaDef, PresetDef};
use super::error::DnamError;
use super::signatures::{DatasetProvenance, DistributionPolicy, ModelSignatures};

mod clamp_compiler;
mod unknown_keys;

pub use unknown_keys::check_no_unknown_header_keys;

pub use super::io_and_tiers::{
    BlendMode, IoConfig, LevelingConfig, LevelingMetric, ModelCapabilities, SlimmableTiersConfig,
    TierProfile, default_blend_one, default_head_svd_rank, default_input_ref_dbfs, default_inputs,
    default_output_mode, default_pro_tier_profile, default_true,
};
pub use super::json_wire::TensorSliceEntry;
pub use super::topology::{
    ControlBusConfig, LinearReadoutConfig, ModalConvolverConfig, TransportConfig,
    default_allpass_delays, default_dc_pole, default_delay_storage, default_envelopes,
    default_fir_taps, default_fm_rank, default_interp, default_kautz_modals, default_lfo_harmonics,
    default_lfos, default_loop_ceiling, default_modulation_bus_dim, default_ro_pole_48k,
    default_switch_xfade, default_trust_radius, default_tzf_tap, deserialize_allpass_delays,
};

/// Semantic version (`major.minor.patch`) of the `.dnam` container schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SchemaVersion {
    /// Breaking container wire format or tensor layout version (`CONTAINER_VERSION_MAJOR`).
    pub major: u16,
    /// Backwards-compatible additive schema version (`CONTAINER_VERSION_MINOR`).
    pub minor: u16,
    /// Backwards-compatible patch version (`0`).
    pub patch: u16,
}

impl SchemaVersion {
    /// Canonical `3.0.0` (`"sona"`) schema release (`ref[container.dnam.formats]`).
    pub const V3_0_0: Self = Self::new(3, 0, 0);

    /// Current container schema version supported by this engine build.
    pub const CURRENT: Self = Self::V3_0_0;

    /// Constructs a new [`SchemaVersion`].
    #[inline]
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parses a semantic version string (`"major.minor.patch"` or `"major.minor"`).
    pub fn parse(version_str: &str) -> Result<Self, DnamError> {
        let mut parts = version_str.trim().split('.');
        let major_str = parts.next().ok_or_else(|| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema version '{version_str}': expected 'major.minor.patch'"
            ))
        })?;
        let minor_str = parts.next().ok_or_else(|| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema version '{version_str}': expected 'major.minor.patch'"
            ))
        })?;
        let patch_str = parts.next().unwrap_or("0");
        if parts.next().is_some() {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "invalid schema version '{version_str}': too many dot-separated segments"
            )));
        }
        let major = major_str.parse::<u16>().map_err(|_| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema major version in '{version_str}'"
            ))
        })?;
        let minor = minor_str.parse::<u16>().map_err(|_| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema minor version in '{version_str}'"
            ))
        })?;
        let patch = patch_str.parse::<u16>().map_err(|_| {
            DnamError::SchemaValidationError(alloc::format!(
                "invalid schema patch version in '{version_str}'"
            ))
        })?;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Formats the schema version as `"major.minor.patch"`.
    pub fn as_semver_string(&self) -> String {
        alloc::format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    /// Returns `true` if this schema version is supported by the current engine build.
    #[inline]
    pub const fn is_supported(&self) -> bool {
        self.major == crate::container::CONTAINER_VERSION_MAJOR
            && self.minor == crate::container::CONTAINER_VERSION_MINOR
    }

    /// Fails with [`DnamError::UnsupportedVersion`] if this schema version is not supported.
    #[inline]
    pub fn ensure_supported(&self) -> Result<(), DnamError> {
        if !self.is_supported() {
            return Err(DnamError::UnsupportedVersion {
                major: self.major,
                minor: self.minor,
            });
        }
        Ok(())
    }

    /// Verifies that the 64-byte binary preamble `(version_major, version_minor)` matches this JSON header version.
    pub fn ensure_matches_preamble(
        &self,
        preamble_major: u16,
        preamble_minor: u16,
    ) -> Result<(), DnamError> {
        if self.major != preamble_major || self.minor != preamble_minor {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "JSON header version '{}.{}.{}' does not match binary preamble version '{}.{}'",
                self.major,
                self.minor,
                self.patch,
                preamble_major,
                preamble_minor
            )));
        }
        Ok(())
    }
}

const _: () = assert!(
    SchemaVersion::CURRENT.major == crate::container::CONTAINER_VERSION_MAJOR
        && SchemaVersion::CURRENT.minor == crate::container::CONTAINER_VERSION_MINOR
);

fn default_version() -> String {
    SchemaVersion::V3_0_0.as_semver_string()
}

fn default_container_kind() -> String {
    "dnam".to_string()
}

fn default_architecture() -> String {
    "sona".to_string()
}

const fn default_sample_rate() -> u32 {
    48_000
}

/// Top-level UTF-8 JSON header in Region 3 of a canonical `v3.0` `.dnam` container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DnamJsonHeader {
    /// Semantic version of the container schema (`"3.0.0"`).
    #[serde(default = "default_version")]
    pub version: String,
    /// Container kind: `"dnam"` or `"preview"`.
    #[serde(default = "default_container_kind")]
    pub container_kind: String,
    /// Target engine architecture tag (`"sona"`).
    #[serde(default = "default_architecture")]
    pub architecture: String,
    /// Sampling rate at which the neural model was trained (`48000`).
    #[serde(default = "default_sample_rate")]
    pub sample_rate_train_hz: u32,
    /// Essential model identity, calibration, dataset provenance, and cryptographic signatures.
    pub metadata: ModelMetadata,
    /// Open dictionary for arbitrary application/user-defined metadata.
    #[serde(default)]
    pub user_metadata: BTreeMap<String, serde_json::Value>,
    /// Open per-stem mutable dictionary merged from `DMU1`/`DME1` (`stem_user_metadata`, stripped from JCS digest).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stem_user_metadata: BTreeMap<String, serde_json::Value>,
    /// Audio I/O topology, PDC latency, estimated tail time, capabilities, and auto-unity leveling.
    pub io_config: IoConfig,
    /// Slimmable tier profiles (`lite`, `full`, `pro`, `ultra`).
    pub slimmable_tiers: SlimmableTiersConfig,
    /// Human-readable definitions of all exposed macro controls ($P \in [0, 16]$).
    #[serde(default)]
    pub controls_schema: Vec<ControlSchemaDef>,
    /// Creator factory presets ($K \in [0, 64]$).
    #[serde(default)]
    pub presets: Vec<PresetDef>,
    /// Control-Bus parameter definitions.
    #[serde(default)]
    pub control_bus: ControlBusConfig,
    /// Early FIR + Orthonormal Kautz Modal Bank convolver configuration (`modal_convolver`).
    #[serde(default)]
    pub modal_convolver: ModalConvolverConfig,
    /// Learned 48 kHz linear readout leak pole configuration (`linear_readout`).
    #[serde(default)]
    pub linear_readout: LinearReadoutConfig,
    /// Symmetric coercive `Op-T` Transport configuration (`transport`).
    #[serde(default)]
    pub transport: TransportConfig,
    /// Optional delay-history capacity declarations for non-RT BBD/FDN buffer pre-allocation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delay_histories: Vec<DelayHistoryEntry>,
    /// Tensor directory mapping symbolic tensor names to byte slices in `Arena 0..3`.
    #[serde(default)]
    pub tensor_directory: BTreeMap<String, TensorSliceEntry>,
}

/// Essential model identity, calibration, dataset provenance, and cryptographic signatures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    /// Human-readable model name.
    pub name: String,
    /// Optional creator / studio name.
    #[serde(default)]
    pub author: Option<String>,
    /// Optional description of the modeled hardware gear.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional SPDX or custom license identifier.
    #[serde(default)]
    pub license: Option<String>,
    /// ISO-8601 UTC creation timestamp.
    pub created_at: String,
    /// Reference input calibration level in dBFS (`1 dBu = 0.869 Vrms = -20.5 dBFS` default).
    #[serde(default = "default_input_ref_dbfs")]
    pub input_reference_dbfs: f32,
    /// Content-addressable training dataset provenance.
    #[serde(default)]
    pub provenance: DatasetProvenance,
    /// Optional creator-signed distribution and OEM hardware exclusivity policy (covered by `immutable_header_sha256`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distribution_policy: Option<DistributionPolicy>,
    /// Optional split-payload `Ed25519` creator signature and marketplace receipt.
    #[serde(default)]
    pub signatures: Option<ModelSignatures>,
}

impl DnamJsonHeader {
    /// Constructs a canonical reference [`DnamJsonHeader`] (used by tests and container builders).
    pub fn default_reference() -> Self {
        Self {
            version: SchemaVersion::V3_0_0.as_semver_string(),
            container_kind: "dnam".to_string(),
            architecture: default_architecture(),
            sample_rate_train_hz: 48_000,
            metadata: ModelMetadata {
                name: "Reference DNAM Model".to_string(),
                author: None,
                description: None,
                license: None,
                created_at: "2026-10-23T00:00:00Z".to_string(),
                input_reference_dbfs: -20.5,
                provenance: DatasetProvenance::default(),
                distribution_policy: None,
                signatures: None,
            },
            user_metadata: BTreeMap::new(),
            stem_user_metadata: BTreeMap::new(),
            io_config: IoConfig::default(),
            slimmable_tiers: SlimmableTiersConfig {
                lite: TierProfile {
                    feature_bus_dim: 64,
                    channels: 16,
                    early_fir_taps: 32,
                    kautz_modes: 8,
                    tank_lines: 4,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -40.0,
                    arena_end_byte_offset: crate::container::ARENA_OFFSET_LITE_END,
                    prefix_crc32c: 0,
                },
                full: TierProfile {
                    feature_bus_dim: 128,
                    channels: 24,
                    early_fir_taps: 48,
                    kautz_modes: 12,
                    tank_lines: 8,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -45.0,
                    arena_end_byte_offset: crate::container::ARENA_OFFSET_FULL_END,
                    prefix_crc32c: 0,
                },
                pro: TierProfile {
                    feature_bus_dim: 192,
                    channels: 32,
                    early_fir_taps: 96,
                    kautz_modes: 24,
                    tank_lines: 12,
                    head_svd_rank: 4,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -48.0,
                    arena_end_byte_offset: crate::container::ARENA_OFFSET_PRO_END,
                    prefix_crc32c: 0,
                },
                ultra: TierProfile {
                    feature_bus_dim: 256,
                    channels: 40,
                    early_fir_taps: 128,
                    kautz_modes: 32,
                    tank_lines: 16,
                    head_svd_rank: 8,
                    loudness_lufs: -20.5,
                    output_trim_db: 0.0,
                    validation_esr_db: -52.0,
                    arena_end_byte_offset: crate::container::ARENA_OFFSET_ULTRA_END,
                    prefix_crc32c: 0,
                },
            },
            controls_schema: Vec::new(),
            presets: Vec::new(),
            control_bus: ControlBusConfig::default(),
            modal_convolver: ModalConvolverConfig::default(),
            linear_readout: LinearReadoutConfig::default(),
            transport: TransportConfig::default(),
            delay_histories: Vec::new(),
            tensor_directory: BTreeMap::new(),
        }
    }

    /// Parses and validates the header's semantic [`SchemaVersion`].
    pub fn schema_version(&self) -> Result<SchemaVersion, DnamError> {
        let ver = SchemaVersion::parse(&self.version)?;
        ver.ensure_supported()?;
        Ok(ver)
    }

    /// Rejects storefront `.dnam-prev` (`container_kind == "preview"`) containers when loaded on a
    /// DAW or hardware host (`ref[sec.preview.containment]`).
    pub fn ensure_not_preview_for_daw_host(&self) -> Result<(), DnamError> {
        if self.container_kind == "preview" {
            return Err(DnamError::PreviewContainerRejected);
        }
        Ok(())
    }

    /// Validates all structural, alignment, and safety invariants of the JSON header.
    pub fn validate(&self) -> Result<(), DnamError> {
        let ver = self.schema_version()?;
        if self.architecture != "sona" {
            return Err(DnamError::UnsupportedVersion {
                major: ver.major,
                minor: ver.minor,
            });
        }
        if self.container_kind != "dnam" && self.container_kind != "preview" {
            return Err(DnamError::SchemaValidationError(
                "container_kind must be 'dnam' or 'preview'".to_string(),
            ));
        }
        if self.sample_rate_train_hz != 48_000 {
            return Err(DnamError::InvalidSampleRate(self.sample_rate_train_hz));
        }
        let ro_pole = self.linear_readout.ro_pole_48k;
        if !ro_pole.is_finite() || !(0.0..=dnam_kernels::MAX_PARALRU_R_F32).contains(&ro_pole) {
            return Err(DnamError::SchemaValidationError(alloc::format!(
                "linear_readout.ro_pole_48k ({ro_pole}) must be finite and in [0.0, {}]",
                dnam_kernels::MAX_PARALRU_R_F32
            )));
        }
        if self.controls_schema.len() > u8::MAX as usize {
            return Err(DnamError::SchemaValidationError(
                "controls_schema exceeds u8::MAX (255) controls in binary preamble".to_string(),
            ));
        }
        let mut seen_delay_ids = alloc::collections::BTreeSet::new();
        for decl in &self.delay_histories {
            let _ = decl.validate_standalone(&mut seen_delay_ids)?;
        }
        let tier_ends = [
            self.slimmable_tiers.lite.arena_end_byte_offset,
            self.slimmable_tiers.full.arena_end_byte_offset,
            self.slimmable_tiers.pro.arena_end_byte_offset,
            self.slimmable_tiers.ultra.arena_end_byte_offset,
        ];
        for entry in self.tensor_directory.values() {
            if entry.arena > 3 {
                return Err(DnamError::SchemaValidationError(alloc::format!(
                    "tensor_directory arena {} exceeds maximum tier 3",
                    entry.arena
                )));
            }
            if (entry.byte_offset & 0x3F) != 0 {
                return Err(DnamError::UnalignedOffset {
                    field: "tensor_directory.byte_offset",
                    offset: entry.byte_offset,
                });
            }
            if (entry.byte_length & 0x3F) != 0 {
                return Err(DnamError::UnalignedOffset {
                    field: "tensor_directory.byte_length",
                    offset: entry.byte_length,
                });
            }
            let end = entry
                .byte_offset
                .checked_add(entry.byte_length)
                .ok_or_else(|| {
                    DnamError::SchemaValidationError(
                        "tensor_directory byte_offset + byte_length overflows u32".to_string(),
                    )
                })?;
            let tier_limit = tier_ends[entry.arena as usize];
            if end > tier_limit {
                return Err(DnamError::SchemaValidationError(alloc::format!(
                    "tensor_directory slice [{}..{}) exceeds arena {} end bound {}",
                    entry.byte_offset,
                    end,
                    entry.arena,
                    tier_limit
                )));
            }
        }
        let _ = self.compile_clamp_descriptors()?;
        Ok(())
    }

    /// Compiles `controls_schema` ($P \in [0, 255]$) into a vector of 64-byte binary
    /// [`dnam_vm::control::ControlClampDescriptor`] records (`CL 0` of each 128-byte `ControlChannelRecord[p]`)
    /// used at $3\text{ kHz}$ with zero string parsing and zero `FDIV`.
    pub fn compile_clamp_descriptors(
        &self,
    ) -> Result<Vec<dnam_vm::control::ControlClampDescriptor>, DnamError> {
        clamp_compiler::compile_controls_schema_descriptors(&self.controls_schema)
    }

    /// Deserializes a [`DnamJsonHeader`] from UTF-8 JSON bytes in a single non-generic compilation unit.
    #[inline(never)]
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Deserializes a [`DnamJsonHeader`] from a UTF-8 JSON string (delegating to [`Self::from_json_slice`]).
    #[inline]
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        Self::from_json_slice(s.as_bytes())
    }

    /// Serializes this [`DnamJsonHeader`] to a UTF-8 JSON byte vector in a single non-generic compilation unit.
    #[inline(never)]
    pub fn to_json_vec(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Serializes this [`DnamJsonHeader`] to a UTF-8 JSON string (delegating to [`Self::to_json_vec`]).
    #[inline]
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        let bytes = self.to_json_vec()?;
        // SAFETY: `serde_json::to_vec` guarantees valid UTF-8 output.
        Ok(String::from_utf8(bytes).expect("serde_json::to_vec must emit valid UTF-8"))
    }

    /// Parses, normalizes numeric range defaults/presets/stems, validates structural invariants,
    /// rejects unknown schema fields, and serializes a [`DnamJsonHeader`] to canonical wire JSON.
    pub fn validate_and_normalize_json_str(header_json: &str) -> Result<String, DnamError> {
        let input_val: serde_json::Value = serde_json::from_slice(header_json.as_bytes())
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
        let mut header: Self = Self::from_json_slice(header_json.as_bytes())
            .map_err(|e| DnamError::SchemaValidationError(alloc::format!("{e}")))?;
        for ctrl in &mut header.controls_schema {
            if ctrl.kind == ControlKind::Range
                && let Some(f) = ctrl.default.as_f64()
            {
                ctrl.default = serde_json::Value::from(f);
            }
        }
        let range_ids: Vec<String> = header
            .controls_schema
            .iter()
            .filter(|c| c.kind == ControlKind::Range)
            .map(|c| c.id.clone())
            .collect();
        for preset in &mut header.presets {
            for (k, v) in &mut preset.controls {
                if range_ids.contains(k)
                    && let Some(f) = v.as_f64()
                {
                    *v = serde_json::Value::from(f);
                }
            }
        }
        header.validate()?;
        let out_bytes = header
            .to_json_vec()
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
        let norm_val: serde_json::Value = serde_json::from_slice(&out_bytes)
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))?;
        check_no_unknown_header_keys(&input_val, &norm_val, None)?;
        String::from_utf8(out_bytes)
            .map_err(|e| DnamError::InvalidJsonHeader(alloc::format!("{e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ControlUiHints;
    use alloc::vec;

    #[test]
    fn test_minimal_json_serde_defaults_and_validation_errors() {
        let minimal_json = r#"{
            "metadata": {
                "name": "Minimal",
                "created_at": "2026-10-23T00:00:00Z",
                "signatures": {
                    "immutable_header_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "master_arena_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "creator_signature": {
                        "signer_id": "did:web:test",
                        "pubkey_ed25519": "0000000000000000000000000000000000000000000000000000000000000000",
                        "sig_ed25519": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                    },
                    "marketplace_receipt": {
                        "marketplace_id": "dnam.store",
                        "license_id": "lic1",
                        "buyer_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                        "license_tier": "personal",
                        "issued_at": "2026-10-23T00:00:00Z",
                        "buyer_arena_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                        "receipt_sig_ed25519": "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                    }
                }
            },
            "io_config": {},
            "control_bus": {},
            "modal_convolver": {},
            "linear_readout": {},
            "transport": {},
            "slimmable_tiers": {
                "lite": { "channels": 16, "tank_lines": 4, "arena_end_byte_offset": 4608, "prefix_crc32c": 0 },
                "full": { "channels": 24, "tank_lines": 8, "arena_end_byte_offset": 7936, "prefix_crc32c": 0 },
                "ultra": { "channels": 40, "tank_lines": 16, "arena_end_byte_offset": 30720, "prefix_crc32c": 0 }
            },
            "controls_schema": [
                {
                    "id": "Gain",
                    "name": "Gain",
                    "type": "range",
                    "declared_range": [0.0, 10.0],
                    "default": 5.0,
                    "conditional_bounds": [
                        { "when": [{ "control_id": "Sw", "option_id": "on" }], "min": 2.0 },
                        { "when": [{ "control_id": "UnknownSw", "option_id": "on" }], "min": 1.0 },
                        { "when": [{ "control_id": "Sw", "option_id": "off" }], "max": 8.0 }
                    ]
                },
                {
                    "id": "Sw",
                    "name": "Sw",
                    "type": "switch",
                    "declared_options": ["off", "on"],
                    "active_options": ["nonexistent"],
                    "default": "off"
                }
            ]
        }"#;
        let mut h: DnamJsonHeader = serde_json::from_str(minimal_json).unwrap();
        assert_eq!(h.version, "3.0.0");
        assert_eq!(h.container_kind, "dnam");
        assert_eq!(h.architecture, "sona");
        assert_eq!(h.sample_rate_train_hz, 48_000);
        assert_eq!(h.metadata.input_reference_dbfs, -20.5);
        assert_eq!(h.io_config.inputs, 2);
        assert_eq!(h.slimmable_tiers.lite.head_svd_rank, 4);
        assert_eq!(h.slimmable_tiers.pro.arena_end_byte_offset, 15_872);
        assert_eq!(h.control_bus.envelopes, 4);
        assert_eq!(h.modal_convolver.fir_taps, 32);
        assert_eq!(h.linear_readout.ro_pole_48k, 0.0);
        assert_eq!(h.transport.dc_blocker_pole, 0.995);
        assert!(h.validate().is_ok());

        h.linear_readout.ro_pole_48k = dnam_kernels::MAX_PARALRU_R_F32;
        assert!(h.validate().is_ok());
        h.linear_readout.ro_pole_48k = -0.01;
        assert!(h.validate().is_err());
        h.linear_readout.ro_pole_48k = 1.0;
        assert!(h.validate().is_err());
        h.linear_readout.ro_pole_48k = f32::NAN;
        assert!(h.validate().is_err());
        h.linear_readout.ro_pole_48k = 0.0;

        // Legacy v2 or non-"sona" architecture must fail fast with UnsupportedVersion.
        h.version = "2.0.0".to_string();
        assert!(matches!(
            h.validate(),
            Err(DnamError::UnsupportedVersion { major: 2, minor: 0 })
        ));
        h.version = "3.0.0".to_string();

        h.architecture = "DNAM_TORA".to_string();
        assert!(matches!(
            h.validate(),
            Err(DnamError::UnsupportedVersion { major: 3, minor: 0 })
        ));
        h.architecture = "sona".to_string();

        // delay_histories uses canonical DelayHistoryEntry schema.
        let decl_default: DelayHistoryEntry = serde_json::from_str(
            r#"{
                "id": "tank.quad0.line0",
                "owner": "transport",
                "kind": "fractional",
                "introduced_tier": "lite",
                "sample_format": "fp16",
                "reference_samples": { "lite": "1024" }
            }"#,
        )
        .unwrap();
        assert_eq!(decl_default.sample_format, "fp16");
        assert_eq!(decl_default.max_samples_48k(), 1024);
        h.delay_histories.push(decl_default);
        assert!(h.validate().is_ok());
        h.delay_histories[0].sample_format = "f32".to_string();
        assert!(h.validate().is_ok());
        h.delay_histories[0].sample_format = "f64".to_string();
        assert!(h.validate().is_err());
        h.delay_histories.clear();

        let compiled = h.compile_clamp_descriptors().unwrap();
        assert_eq!(compiled[0].cond_rules[0].rule_op, 2);
        assert_eq!(compiled[0].cond_rules[1].rule_op, 1);
        assert_eq!(compiled[1].active_options_mask, 1);

        // Validation error branches
        h.container_kind = "invalid".to_string();
        assert!(h.validate().is_err());
        h.container_kind = "preview".to_string();
        assert!(h.validate().is_ok());
        h.container_kind = "dnam".to_string();

        h.sample_rate_train_hz = 10_000;
        assert!(h.validate().is_err());
        h.sample_rate_train_hz = 44_100;
        assert!(h.validate().is_err());
        h.sample_rate_train_hz = 48_000;

        let mut bad_slice = TensorSliceEntry {
            arena: 0,
            dtype: "f32".to_string(),
            packing: None,
            feature_dim: None,
            packed_len: None,
            shape: None,
            byte_offset: 3,
            byte_length: 64,
        };
        h.tensor_directory
            .insert("bad_off".to_string(), bad_slice.clone());
        assert!(h.validate().is_err());
        bad_slice.byte_offset = 64;
        bad_slice.byte_length = 63;
        h.tensor_directory
            .insert("bad_off".to_string(), bad_slice.clone());
        assert!(h.validate().is_err());
        bad_slice.byte_offset = 4608;
        bad_slice.byte_length = 64;
        h.tensor_directory
            .insert("bad_off".to_string(), bad_slice.clone());
        assert!(h.validate().is_err());
        bad_slice.arena = 4;
        bad_slice.byte_offset = 0;
        h.tensor_directory.insert("bad_off".to_string(), bad_slice);
        assert!(h.validate().is_err());
        h.tensor_directory.clear();

        // Missing declared_range on range control
        h.controls_schema = vec![ControlSchemaDef {
            id: "Gain".to_string(),
            name: "Gain".to_string(),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: None,
            anchor_nodes_norm: Some(vec![0.0, -1.0, 1.0]),
            declared_range: None,
            safe_clamped_range: None,
            default: serde_json::json!(5.0),
            conditional_bounds: Vec::new(),
            represented_switch_options: None,
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: BTreeMap::new(),
        }];
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_range = Some([5.0, 5.0]);
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_range = Some([0.0, 10.0]);
        h.controls_schema[0].default = serde_json::json!("bad");
        assert!(h.validate().is_err());

        h.controls_schema[0].kind = ControlKind::Switch;
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_options = Some(vec![]);
        assert!(h.validate().is_err());

        h.controls_schema[0].declared_options = Some(vec!["off".to_string(), "on".to_string()]);
        h.controls_schema = vec![h.controls_schema[0].clone(); 256];
        assert!(h.validate().is_err());
        assert!(h.compile_clamp_descriptors().is_err());

        // Capabilities & Leveling defaults and validation
        assert_eq!(h.io_config.capabilities.blend_mode, BlendMode::None);
        assert_eq!(h.io_config.capabilities.default_blend, 1.0);
        assert!(!h.io_config.capabilities.wet_polarity_inverted);
        assert_eq!(
            h.io_config.leveling.leveling_metric,
            LevelingMetric::RmsDbfs
        );
        assert!(h.io_config.leveling.auto_unity_enabled);

        assert!(SchemaVersion::parse("a.0.0").is_err());
        assert!(SchemaVersion::parse("1.b.0").is_err());
        assert!(SchemaVersion::parse("1.0.c").is_err());
        assert_eq!(SchemaVersion::V3_0_0.as_semver_string(), "3.0.0");
        assert_eq!(SchemaVersion::CURRENT, SchemaVersion::V3_0_0);

        let norm_json = DnamJsonHeader::validate_and_normalize_json_str(minimal_json).unwrap();
        assert!(norm_json.contains("\"version\":\"3.0.0\""));
        assert!(norm_json.contains("\"architecture\":\"sona\""));
        assert!(DnamJsonHeader::validate_and_normalize_json_str("not json").is_err());
        let bad_unknown = minimal_json.replace(
            "\"metadata\": {",
            "\"unknown_root_field\": 123, \"metadata\": {",
        );
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&bad_unknown).is_err());

        // Misplaced optional field ("shape" inside "modal_convolver" or "distribution_policy" at root) must fail
        let bad_misplaced = minimal_json.replace(
            "\"modal_convolver\": {}",
            "\"modal_convolver\": {\"shape\": \"bad\"}",
        );
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&bad_misplaced).is_err());
        let bad_root_opt = minimal_json.replace(
            "\"metadata\": {",
            "\"distribution_policy\": null, \"metadata\": {",
        );
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&bad_root_opt).is_err());

        // Legacy fields evicted from the canonical v3.0 SONA schema must fail closed.
        for legacy_patch in [
            ("\"metadata\": {", "\"io_router\": {}, \"metadata\": {"),
            (
                "\"metadata\": {",
                "\"sub_crossover_hz\": 200.0, \"metadata\": {",
            ),
            (
                "\"io_config\": {}",
                "\"io_config\": {\"capabilities\": {\"supports_freeze\": true}}",
            ),
            (
                "\"io_config\": {}",
                "\"io_config\": {\"capabilities\": {\"freeze_mutes_dry\": true}}",
            ),
            (
                "\"controls_schema\": [",
                "\"controls_schema\": [{ \"id\": \"F\", \"name\": \"F\", \"type\": \"switch\", \"declared_options\": [\"off\", \"on\"], \"default\": \"off\", \"calibration\": { \"role\": \"freeze_latch\" } },",
            ),
        ] {
            let legacy_json = minimal_json.replace(legacy_patch.0, legacy_patch.1);
            assert!(DnamJsonHeader::from_json_str(&legacy_json).is_err());
            assert!(DnamJsonHeader::validate_and_normalize_json_str(&legacy_json).is_err());
        }

        // Explicit null on valid skip_serializing_if fields must succeed
        let ok_explicit_nulls = minimal_json.replace(
            "\"metadata\": {",
            "\"delay_histories\": [], \"stem_user_metadata\": {}, \"metadata\": {\"distribution_policy\": null,",
        );
        assert!(DnamJsonHeader::validate_and_normalize_json_str(&ok_explicit_nulls).is_ok());
    }
}
