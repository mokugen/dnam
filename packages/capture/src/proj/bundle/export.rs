#![forbid(unsafe_code)]
//! Runtime `.dnam` / `.dnam-bin` (`< 0.2 ms`) exporting, `DMU1` mutable metadata suffix
//! growth, and `DAU1` tier slimming from `proj::bundle` (`ref[container.proj.wght-and-export]`).

use std::collections::BTreeMap;

use dnam_engine::Tier;
use dnam_engine::container::{
    DEFAULT_DMU1_TRAILER_CAPACITY, DMU1_HEADER_SIZE, DnamContainerView, align64_u64,
    append_dmue_mutable_suffix, build_dmu1_mutable_frame, extract_dnam_bin,
    strip_container_to_tier, update_header_in_place, validate_v3_dnam_layout_with_max_profile,
};
use dnam_engine::provenance::{CryptoProvider, DefaultCryptoProvider, enforce_distribution_policy};
use dnam_engine::schema::{DnamError, DnamJsonHeader};
use dnam_vm::bin::{DnamBinView, strip_dnam_bin_to_tier};

use super::super::directory::ProjChunkKind;
use super::super::schema::DnamProjManifest;
use super::DnamProjView;
use super::dbin::map_vm_err;

fn grow_open_dnam_dmu1_suffix<C: CryptoProvider>(
    out_dnam: &mut Vec<u8>,
    merged_hdr: &DnamJsonHeader,
    crypto: &C,
) -> Result<(), DnamError> {
    merged_hdr.validate()?;
    let layout = validate_v3_dnam_layout_with_max_profile(out_dnam, 1)?;
    let mut dmu_obj = serde_json::Map::new();
    if !merged_hdr.presets.is_empty() {
        dmu_obj.insert(
            "presets".to_string(),
            serde_json::to_value(&merged_hdr.presets)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    if !merged_hdr.user_metadata.is_empty() {
        dmu_obj.insert(
            "user_metadata".to_string(),
            serde_json::to_value(&merged_hdr.user_metadata)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    let mut ctrl_um = serde_json::Map::new();
    for ctrl in &merged_hdr.controls_schema {
        if !ctrl.user_metadata.is_empty() {
            ctrl_um.insert(
                ctrl.id.clone(),
                serde_json::to_value(&ctrl.user_metadata)
                    .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
            );
        }
    }
    if !ctrl_um.is_empty() {
        dmu_obj.insert(
            "control_user_metadata".to_string(),
            serde_json::Value::Object(ctrl_um),
        );
    }
    if !merged_hdr.stem_user_metadata.is_empty() {
        dmu_obj.insert(
            "stem_user_metadata".to_string(),
            serde_json::to_value(&merged_hdr.stem_user_metadata)
                .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?,
        );
    }
    let payload_bytes = serde_json::to_vec(&serde_json::Value::Object(dmu_obj))
        .map_err(|e| DnamError::InvalidJsonHeader(e.to_string()))?;
    let required = DMU1_HEADER_SIZE + payload_bytes.len();
    if required > DEFAULT_DMU1_TRAILER_CAPACITY {
        return Err(DnamError::HeaderCapacityExceeded {
            capacity: DEFAULT_DMU1_TRAILER_CAPACITY,
            required,
        });
    }
    let grown_cap = align64_u64((required + 16) as u64)
        .and_then(|n| usize::try_from(n).ok())
        .map(|n| (n - 16).min(DEFAULT_DMU1_TRAILER_CAPACITY))
        .ok_or_else(|| {
            DnamError::SchemaValidationError("DMU1 grown capacity overflow".to_string())
        })?;
    let dmu1 = build_dmu1_mutable_frame(&payload_bytes, grown_cap)?;
    out_dnam.truncate(layout.mutable_extent.0);
    append_dmue_mutable_suffix(out_dnam, &dmu1);
    let _ = DnamContainerView::from_bytes(out_dnam, crypto)?;
    Ok(())
}

fn build_merged_stem_user_metadata(
    manifest: &DnamProjManifest,
) -> BTreeMap<String, serde_json::Value> {
    let mut merged = manifest.base.stem_user_metadata.clone();
    for stem in &manifest.stems_registry {
        if !stem.user_metadata.is_empty() {
            let map: serde_json::Map<String, serde_json::Value> = stem
                .user_metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            merged.insert(stem.stem_id.clone(), serde_json::Value::Object(map));
        }
    }
    merged
}

/// Exports a validated runtime `v3.0` `.dnam` container from the project's `WGHT` chunk in `< 0.2 ms`
/// (`r[container.proj.wght-and-export]`).
pub fn export_runtime_dnam(
    proj_bytes: &[u8],
    target_tier: Option<Tier>,
) -> Result<Vec<u8>, DnamError> {
    export_runtime_dnam_with_crypto(proj_bytes, target_tier, &DefaultCryptoProvider::default())
}

/// Exports a validated bare Layer 0 `.dnam-bin` (`"DNBN"`) register program directly from the
/// project's `WGHT` chunk (`ref[container.proj.wght-and-export]`).
pub fn export_runtime_dnam_bin(
    proj_bytes: &[u8],
    target_tier: Option<Tier>,
) -> Result<Vec<u8>, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamProjView::from_bytes(proj_bytes, &crypto)?;
    let Some(wght_bytes) = view.chunk_slice(ProjChunkKind::Wght, 0) else {
        return Err(DnamError::SchemaValidationError(
            "cannot export runtime .dnam-bin: project has no solved WGHT chunk".to_string(),
        ));
    };
    let dbin = extract_dnam_bin(wght_bytes)?;
    let bin_view = DnamBinView::parse(dbin).map_err(map_vm_err)?;
    if let Some(max_tier) = target_tier
        && (max_tier as u8) < (bin_view.retained_tier as u8)
    {
        return strip_dnam_bin_to_tier(dbin, max_tier)
            .map(Vec::from)
            .map_err(map_vm_err);
    }
    Ok(dbin.to_owned())
}

/// Exports a validated runtime `v3.0` `.dnam` container from `WGHT` using `crypto`, enforcing
/// `[STAB-ZERO-DC]`, `DistributionPolicy`, and `DAU1` buyer tier-slimming rules (`r[container.proj.wght-and-export]`).
pub fn export_runtime_dnam_with_crypto<C: CryptoProvider>(
    proj_bytes: &[u8],
    target_tier: Option<Tier>,
    crypto: &C,
) -> Result<Vec<u8>, DnamError> {
    let (view, wght_pair_opt) = DnamProjView::from_bytes_with_wght_view(proj_bytes, crypto, true)?;
    let Some((wght_bytes, rt_view)) = wght_pair_opt else {
        return Err(DnamError::SchemaValidationError(
            "cannot export runtime .dnam: project has no solved WGHT chunk".to_string(),
        ));
    };

    enforce_distribution_policy(&rt_view.header, rt_view.protection_profile)?;

    let mut out_dnam = wght_bytes.to_vec();
    let merged_stem_meta = build_merged_stem_user_metadata(&view.manifest);
    let ctrl_meta_differs = rt_view.header.controls_schema.len()
        == view.manifest.base.controls_schema.len()
        && rt_view
            .header
            .controls_schema
            .iter()
            .zip(&view.manifest.base.controls_schema)
            .any(|(a, b)| a.id == b.id && a.user_metadata != b.user_metadata);
    if rt_view.header.presets != view.manifest.base.presets
        || rt_view.header.user_metadata != view.manifest.base.user_metadata
        || rt_view.header.stem_user_metadata != merged_stem_meta
        || ctrl_meta_differs
    {
        let mut merged_hdr = rt_view.header.clone();
        merged_hdr.presets = view.manifest.base.presets.clone();
        merged_hdr.user_metadata = view.manifest.base.user_metadata.clone();
        merged_hdr.stem_user_metadata = merged_stem_meta;
        if merged_hdr.controls_schema.len() == view.manifest.base.controls_schema.len() {
            for (dst, src) in merged_hdr
                .controls_schema
                .iter_mut()
                .zip(&view.manifest.base.controls_schema)
            {
                if dst.id == src.id {
                    dst.user_metadata = src.user_metadata.clone();
                }
            }
        }
        match update_header_in_place(&mut out_dnam, &merged_hdr) {
            Ok(()) => {}
            Err(DnamError::HeaderCapacityExceeded { .. })
                if !rt_view.preamble.is_encrypted_arena() =>
            {
                grow_open_dnam_dmu1_suffix(&mut out_dnam, &merged_hdr, crypto)?;
            }
            Err(err) => return Err(err),
        }
    }

    if let Some(max_tier) = target_tier
        && (max_tier as u8) < (rt_view.preamble.highest_tier() as u8)
    {
        if let Some(ref sigs) = rt_view.header.metadata.signatures
            && let Some(ref rcpt) = sigs.marketplace_receipt
            && rcpt.buyer_tier_arena_sha256.is_none()
        {
            return Err(DnamError::CryptoVerificationFailed(
                "cannot strip buyer-signed WGHT without buyer_tier_arena_sha256",
            ));
        }
        out_dnam = strip_container_to_tier(&out_dnam, max_tier)?;
        let _ = DnamContainerView::from_bytes(&out_dnam, crypto)?;
    }

    Ok(out_dnam)
}
