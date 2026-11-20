//! Model-declared delay histories and Stage III 64B-aligned tier slice validation
//! (`ref[container.dnam.delay-histories]`, `ref[container.dnam.cabinet-slices]`).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::F32;

use crate::schema::{DnamError, SchemaOptionExt, reject_if};

/// Per-tier Stage III early FIR slice bytes (`[256, 128, 384, 256] B` = `32 / +16 / +48 / +32` taps/ch × 2ch × 4B).
pub(super) const STAGE3_FIR_SLICE_BYTES_BY_TIER: [usize; 4] = [256, 128, 384, 256];

const _: () = assert!(
    STAGE3_FIR_SLICE_BYTES_BY_TIER[0]
        + STAGE3_FIR_SLICE_BYTES_BY_TIER[1]
        + STAGE3_FIR_SLICE_BYTES_BY_TIER[2]
        + STAGE3_FIR_SLICE_BYTES_BY_TIER[3]
        == 1_024
);
const _: () = assert!(
    crate::stages::weights::STAGE3_KAUTZ_BLOCKS_PER_TIER[0] == 2
        && crate::stages::weights::STAGE3_KAUTZ_BLOCKS_PER_TIER[1] == 1
        && crate::stages::weights::STAGE3_KAUTZ_BLOCKS_PER_TIER[2] == 3
        && crate::stages::weights::STAGE3_KAUTZ_BLOCKS_PER_TIER[3] == 2
);

const TIER_NAMES: [&str; 4] = ["lite", "full", "pro", "ultra"];

fn default_delay_sample_format() -> String {
    String::from("fp16")
}

/// Structured `delay_histories` JSON entry serialized in the canonical `v3.0` immutable header
/// (`ref[container.dnam.delay-histories]`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DelayHistoryEntry {
    pub id: String,
    pub stage: String,
    pub kind: String,
    pub introduced_tier: String,
    #[serde(default = "default_delay_sample_format")]
    pub sample_format: String,
    #[serde(default)]
    pub reference_samples: BTreeMap<String, String>,
}

impl DelayHistoryEntry {
    /// Returns the maximum declared `48 kHz` reference sample count across tiers (saturating to `u32::MAX`).
    pub fn max_samples_48k(&self) -> u32 {
        self.reference_samples
            .values()
            .filter_map(|s| parse_canonical_reference_samples(s).ok())
            .max()
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32
    }

    /// Validates this entry standalone (accepting any contiguous `[introduced_tier..=max_present]` tier keys).
    pub fn validate_standalone(
        &self,
        seen_ids: &mut BTreeSet<String>,
    ) -> Result<DeclaredDelayHistory, DnamError> {
        let intro =
            tier_name_to_index(&self.introduced_tier).or_schema_err("invalid introduced_tier")?;
        let mut max_present = intro;
        for (idx, name) in TIER_NAMES.iter().enumerate() {
            if self.reference_samples.contains_key(*name) {
                let u = idx as u8;
                reject_if(u < intro, "mismatched reference_samples tier keys")?;
                max_present = max_present.max(u);
            }
        }
        self.to_declared_for_max_tier(max_present, seen_ids)
    }

    /// Validates this entry against an explicit `max_tier` (`0..=3`) and returns [`DeclaredDelayHistory`].
    pub fn to_declared_for_max_tier(
        &self,
        max_tier: u8,
        seen_ids: &mut BTreeSet<String>,
    ) -> Result<DeclaredDelayHistory, DnamError> {
        validate_delay_history_fields(
            &self.id,
            &self.stage,
            &self.kind,
            &self.introduced_tier,
            &self.sample_format,
            self.reference_samples.len(),
            |tier_name| self.reference_samples.get(tier_name).map(String::as_str),
            max_tier,
            seen_ids,
        )
    }
}

/// Validated `delay_histories` declaration (`ref[container.dnam.delay-histories]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeclaredDelayHistory {
    pub id: String,
    pub stage: String,
    pub kind: String,
    pub introduced_tier: u8,
    pub sample_format: String,
    /// Per-tier reference sample counts at `48 kHz` (`0` for tiers below `introduced_tier`).
    pub reference_samples_by_tier: [u64; 4],
}

/// Parses a canonical decimal string (`[1-9][0-9]*`) as a non-zero `u64` (`ref[container.dnam.delay-histories]`).
pub fn parse_canonical_reference_samples(text: &str) -> Result<u64, DnamError> {
    let bytes = text.as_bytes();
    reject_if(
        bytes.is_empty() || !bytes[0].is_ascii_digit() || bytes[0] == b'0',
        "noncanonical unsigned count",
    )?;
    reject_if(
        bytes.iter().any(|b| !b.is_ascii_digit()),
        "noncanonical unsigned count",
    )?;
    text.parse::<u64>()
        .map_err(|_| DnamError::SchemaValidationError("count exceeds u64".to_string()))
}

fn tier_name_to_index(name: &str) -> Option<u8> {
    match name {
        "lite" => Some(0),
        "full" => Some(1),
        "pro" => Some(2),
        "ultra" => Some(3),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_delay_history_fields<'a, F>(
    id: &str,
    stage: &str,
    kind: &str,
    intro_str: &str,
    sample_format: &str,
    ref_len: usize,
    get_ref: F,
    max_tier: u8,
    seen_ids: &mut BTreeSet<String>,
) -> Result<DeclaredDelayHistory, DnamError>
where
    F: Fn(&str) -> Option<&'a str>,
{
    reject_if(max_tier > 3, "invalid max_tier")?;
    reject_if(
        id.is_empty() || !seen_ids.insert(id.to_string()),
        "duplicate or empty delay history id",
    )?;
    reject_if(
        !matches!(stage, "op_t_tank" | "stage4_tank" | "stage1_pre"),
        "invalid delay history stage",
    )?;
    reject_if(
        !matches!(kind, "fractional" | "allpass"),
        "invalid delay history kind",
    )?;
    let introduced_tier = tier_name_to_index(intro_str).or_schema_err("invalid introduced_tier")?;
    reject_if(
        introduced_tier > max_tier,
        "introduced_tier exceeds max_tier",
    )?;
    reject_if(
        !matches!(sample_format, "fp16" | "f32"),
        "invalid sample_format",
    )?;
    let expected_tier_count = (max_tier - introduced_tier + 1) as usize;
    reject_if(
        ref_len != expected_tier_count,
        "mismatched reference_samples tier keys",
    )?;
    let mut ref_by_tier = [0u64; 4];
    for t in (introduced_tier as usize)..=(max_tier as usize) {
        let s = get_ref(TIER_NAMES[t]).or_schema_err("missing tier sample count")?;
        let count = parse_canonical_reference_samples(s)?;
        reject_if(
            kind == "fractional" && count < 4,
            "fractional delay reference_samples must be >= 4",
        )?;
        ref_by_tier[t] = count;
    }
    Ok(DeclaredDelayHistory {
        id: id.to_string(),
        stage: stage.to_string(),
        kind: kind.to_string(),
        introduced_tier,
        sample_format: sample_format.to_string(),
        reference_samples_by_tier: ref_by_tier,
    })
}

/// Validates structured [`DelayHistoryEntry`] slice for a container with `max_tier` (`0..=3`).
pub fn validate_delay_history_entries(
    entries: &[DelayHistoryEntry],
    max_tier: u8,
) -> Result<Vec<DeclaredDelayHistory>, DnamError> {
    reject_if(max_tier > 3, "invalid max_tier")?;
    let mut seen_ids = BTreeSet::new();
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        out.push(entry.to_declared_for_max_tier(max_tier, &mut seen_ids)?);
    }
    Ok(out)
}

/// Validates and parses the `delay_histories` JSON array for a container with `max_tier` (`0..=3`,
/// `ref[container.dnam.delay-histories]`, `ref[dsp.op-t.ring.declaration]`).
pub fn validate_delay_histories(
    entries: &[Value],
    max_tier: u8,
) -> Result<Vec<DeclaredDelayHistory>, DnamError> {
    reject_if(max_tier > 3, "invalid max_tier")?;
    let mut seen_ids = BTreeSet::new();
    let mut out = Vec::with_capacity(entries.len());

    for item in entries {
        let obj = item
            .as_object()
            .or_schema_err("delay history must be an object")?;
        reject_if(obj.len() != 6, "invalid delay history fields")?;
        let id = obj.get("id").and_then(Value::as_str).unwrap_or("");
        let stage = obj.get("stage").and_then(Value::as_str).unwrap_or("");
        let kind = obj.get("kind").and_then(Value::as_str).unwrap_or("");
        let intro_str = obj
            .get("introduced_tier")
            .and_then(Value::as_str)
            .unwrap_or("");
        let sample_format = obj
            .get("sample_format")
            .and_then(Value::as_str)
            .unwrap_or("");
        let ref_map = obj
            .get("reference_samples")
            .and_then(Value::as_object)
            .or_schema_err("invalid reference_samples")?;
        out.push(validate_delay_history_fields(
            id,
            stage,
            kind,
            intro_str,
            sample_format,
            ref_map.len(),
            |tier_name| ref_map.get(tier_name).and_then(Value::as_str),
            max_tier,
            &mut seen_ids,
        )?);
    }
    Ok(out)
}

/// Validates that a Stage III early FIR slice (`[256, 128, 384, 256] B` across `Lite..Ultra`) and
/// Kautz modal slice (`[2, 1, 3, 2]` `128 B` blocks = `[256, 128, 384, 256] B` in `v3.0`) are
/// 64-byte aligned, non-overlapping, within their tier bounds, and contain only finite `f32`
/// values (`ref[container.dnam.cabinet-slices]`, `ref[container.dnam.load-validation]`).
pub fn validate_stage3_tier_slices(
    tier_index: u8,
    tier_bytes: &[u8],
    early_fir_offset_in_tier: usize,
    early_fir_len: usize,
    kautz_offset_in_tier: usize,
    kautz_len: usize,
) -> Result<(), DnamError> {
    let idx = tier_index as usize;
    if idx >= STAGE3_FIR_SLICE_BYTES_BY_TIER.len() {
        return Err(DnamError::SchemaValidationError(
            "invalid tier_index".to_string(),
        ));
    }
    let expected_fir_len = STAGE3_FIR_SLICE_BYTES_BY_TIER[idx];
    let expected_kautz_len = crate::stages::weights::STAGE3_KAUTZ_BLOCKS_PER_TIER[idx] * 128;
    reject_if(
        early_fir_len != expected_fir_len || kautz_len != expected_kautz_len,
        "invalid Stage III slice byte length",
    )?;
    reject_if(
        !early_fir_offset_in_tier.is_multiple_of(64) || !kautz_offset_in_tier.is_multiple_of(64),
        "misaligned Stage III slice offset",
    )?;
    let fir_end = early_fir_offset_in_tier + early_fir_len;
    let kautz_end = kautz_offset_in_tier + kautz_len;
    reject_if(
        fir_end > tier_bytes.len() || kautz_end > tier_bytes.len(),
        "Stage III slice outside tier",
    )?;
    reject_if(
        early_fir_offset_in_tier < kautz_end && kautz_offset_in_tier < fir_end,
        "overlapping Stage III slices",
    )?;
    for slice in [
        &tier_bytes[early_fir_offset_in_tier..fir_end],
        &tier_bytes[kautz_offset_in_tier..kautz_end],
    ] {
        let words = <[F32]>::ref_from_bytes(slice)
            .ok()
            .or_schema_err("invalid f32 slice")?;
        for w in words {
            reject_if(!w.get().is_finite(), "nonfinite coefficient")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_history_entry_serde_and_validation() {
        let json = r#"{
            "id": "tank.quad0.line0",
            "stage": "op_t_tank",
            "kind": "fractional",
            "introduced_tier": "lite",
            "sample_format": "fp16",
            "reference_samples": { "lite": "432000", "full": "432000" }
        }"#;
        let entry: DelayHistoryEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.max_samples_48k(), 432_000);
        let parsed = validate_delay_history_entries(&[entry], 1).unwrap();
        assert_eq!(parsed[0].reference_samples_by_tier[..2], [432_000, 432_000]);
    }
}
