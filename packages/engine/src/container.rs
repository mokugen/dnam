//! `.dnam` Hybrid Binary Container Parser, Builder, Stripper & Layer 0 Extractor (`container`).
//!
//! Implements the canonical `v3.0` (`"sona"`) `64 B`-aligned `zerocopy` wire framing, zero-copy `mmap`
//! loader (`< 50 ns` Layer 0 `.dnam-bin` extraction), `DMU1`/`DMUE` in-place mutable suffix editor,
//! and Castagnoli `CRC32C` verification from
//! [`docs/specs/containers/dnam_binary_format.md`](../../../docs/specs/containers/dnam_binary_format.md).

pub mod builder;
pub mod preamble;
pub mod strip;
pub mod view;
pub mod wire;

pub use wire::{
    CP1_MAGIC, Cp1DecodedRecord, DCRC_MAGIC, DCRC_RECORD_SIZE, DEFAULT_DMU1_TRAILER_CAPACITY,
    DEFAULT_MAX_MUTABLE_CAPACITY, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE, DcrcAuthRecord,
    DcrcAuthRecordWire, DeclaredDelayHistory, Dmu1HeaderWire, DmueFooterWire, DnamDirectoryWire,
    DnamPreambleWire, SonaCellDescWire, StructuredConditionalBound, StructuredConditionalCurve,
    SwitchControlSpec, V3_DIRECTORY_SIZE, V3_HEADER_OFFSET, V3_PREAMBLE_SIZE, V3ContainerBuildSpec,
    V3ContainerLayout, align64_u64, append_dmue_mutable_suffix, build_dmu1_mutable_frame,
    build_v3_dnam_container_with_dbin, build_v3_dnam_container_with_profile,
    build_v3_open_dnam_container, compute_v3_open_retained_prefix_sha256,
    cp1_contributor_record_bytes, decode_cp1_contributor_stream,
    decode_cp1_contributor_stream_into_slice, decode_cp1_shared_stream_into_slice,
    encode_cp1_shared_stream, encode_domain_field, eval_ui_midpoint_power_skew,
    is_valid_jcs_unicode_text, parse_and_validate_dmu1_frame, parse_canonical_reference_samples,
    parse_dcrc_record, parse_strict_json_object_bytes, parse_structured_conditional_bounds,
    parse_structured_conditional_curves, project_immutable_header_value,
    resolve_conditional_safe_interval, select_conditional_calibration_curve, serialize_jcs_value,
    strip_v3_open_dnam_container, validate_canonical_jcs_header_bytes,
    validate_convolver_tier_slices, validate_delay_histories, validate_delay_history_entries,
    validate_dmu1_frame_no_alloc, validate_mutable_json_schema, validate_v3_dnam_layout_no_alloc,
    validate_v3_dnam_layout_with_max_profile, validate_v3_dnam_layout_with_max_profile_no_alloc,
    validate_v3_open_dnam_layout,
};
pub(crate) use wire::{
    ContainerWireError, parse_dcrc_record_rt, validate_v3_dnam_layout_no_alloc_rt,
};

#[cfg(feature = "crypto")]
pub use crate::provenance::seal_dnam_container;
pub use crate::provenance::{DNSL_MAGIC, MAX_SEAL_RECIPIENTS, unseal_dnam_container};
pub use builder::{
    build_dnam_container, build_dnam_container_from_dnam_bin, build_dnam_container_with_cp1,
    build_dnam_container_with_dbin, update_header_in_place,
};
pub use preamble::DnamPreamble;
pub use strip::strip_container_to_tier;
pub use view::{
    DnamContainerView, SharedControlsExtent, decode_shared_controls_extent, extract_dnam_bin,
};
pub(crate) use view::{extract_dnam_bin_rt, map_vm_error, map_vm_error_rt};

/// Magic bytes for a `.dnam` deployment model container (`0x4D414E44` little-endian).
pub const DNAM_MAGIC: [u8; 4] = *b"DNAM";

/// Current container major version (`3`).
pub const CONTAINER_VERSION_MAJOR: u16 = 3;

/// Current container minor version (`0`).
pub const CONTAINER_VERSION_MINOR: u16 = 0;

/// Fixed byte length of `Arena 0 (Lite)` (`72 × 64 B = 4,608 B`).
pub const ARENA_OFFSET_LITE_END: u32 = dnam_vm::bin::ARENA_TIER_BYTES[0] as u32;

/// Fixed cumulative byte end of `Arena 0..1 (Full)` (`124 × 64 B = 7,936 B`).
pub const ARENA_OFFSET_FULL_END: u32 = dnam_vm::bin::ARENA_TIER_BYTES[1] as u32;

/// Fixed cumulative byte end of `Arena 0..2 (Pro)` (`248 × 64 B = 15,872 B`).
pub const ARENA_OFFSET_PRO_END: u32 = dnam_vm::bin::ARENA_TIER_BYTES[2] as u32;

/// Fixed cumulative byte end of `Arena 0..3 (Ultra)` (`480 × 64 B = 30,720 B`).
pub const ARENA_OFFSET_ULTRA_END: u32 = dnam_vm::bin::ARENA_TIER_BYTES[3] as u32;

/// End of `Arena 0 (Lite)` in 64-byte cache lines (`72`).
pub const ARENA_LITE_END_CL: u16 = (ARENA_OFFSET_LITE_END / 64) as u16;

/// End of `Arena 0..1 (Full)` in 64-byte cache lines (`124`).
pub const ARENA_FULL_END_CL: u16 = (ARENA_OFFSET_FULL_END / 64) as u16;

/// End of `Arena 0..2 (Pro)` in 64-byte cache lines (`248`).
pub const ARENA_PRO_END_CL: u16 = (ARENA_OFFSET_PRO_END / 64) as u16;

/// End of `Arena 0..3 (Ultra)` in 64-byte cache lines (`480`).
pub const ARENA_ULTRA_END_CL: u16 = (ARENA_OFFSET_ULTRA_END / 64) as u16;

const _: () = assert!(ARENA_LITE_END_CL == 72);
const _: () = assert!(ARENA_FULL_END_CL == 124);
const _: () = assert!(ARENA_PRO_END_CL == 248);
const _: () = assert!(ARENA_ULTRA_END_CL == 480);
const _: () = assert!(ARENA_OFFSET_LITE_END == 4_608);
const _: () = assert!(ARENA_OFFSET_FULL_END == 7_936);
const _: () = assert!(ARENA_OFFSET_PRO_END == 15_872);
const _: () = assert!(ARENA_OFFSET_ULTRA_END == 30_720);
const _: () = assert!(ARENA_FULL_END_CL == ARENA_LITE_END_CL + 52);
const _: () = assert!(ARENA_PRO_END_CL == ARENA_FULL_END_CL + 124);
const _: () = assert!(ARENA_ULTRA_END_CL == ARENA_PRO_END_CL + 232);

// ============================================================================
// Preamble Flag Bits (`flags: u16`)
// ============================================================================

/// Bit 0 (`1`): `Arena 0 (Lite)` is present.
pub const FLAG_HAS_LITE: u16 = 1 << 0;
/// Bit 1 (`2`): `Arena 1 (Full Delta)` is present.
pub const FLAG_HAS_FULL_DELTA: u16 = 1 << 1;
/// Bit 2 (`4`): `Arena 2 (Pro Delta)` is present.
pub const FLAG_HAS_PRO_DELTA: u16 = 1 << 2;
/// Bit 3 (`8`): `Arena 3 (Ultra Delta)` is present.
pub const FLAG_HAS_ULTRA_DELTA: u16 = 1 << 3;
/// Bit 9 (`512`): Stereo output head enabled (`1 = stereo`, `0 = mono` saving 50% head MACs).
pub const FLAG_STEREO_OUT: u16 = 1 << 9;
/// Bit 10 (`1024`): Split-payload `Ed25519` signatures are present in `metadata.signatures`.
pub const FLAG_HAS_SIGNATURES: u16 = 1 << 10;
/// Bit 11 (`2048`): `Arena 0..3` is `ChaCha20` 64-byte block encrypted for `authorized_devices`.
pub const FLAG_ENCRYPTED_ARENA: u16 = 1 << 11;

/// Updates an incremental Castagnoli `CRC32C` state with `data` using hardware `CRC32C` when available.
#[inline]
pub fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
    dnam_vm::math::crc32c_slice(crc, data)
}

/// Computes the Castagnoli `CRC32C` (`0x1EDC6F41`) checksum of `data`.
#[inline]
pub fn crc32c_compute(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32c_compute_matches_standard_vector() {
        assert_eq!(crc32c_compute(b"123456789"), 0xe306_9283);
    }
}
