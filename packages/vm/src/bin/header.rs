//! Declarative `#[repr(C)]` `zerocopy` prefix records (`DnamBinHeaderWire`,
//! `DnamBinRuntimeMetaWire`, `ControlClampDescriptorWire`, `ConditionalClampRuleWire`,
//! `TempoSyncEntryWire`, `CextDispatchWire`, and `CP1` directory wire structs).

use core::mem::{align_of, offset_of, size_of};

use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::arena_wire::{
    ARENA0_LITE_BYTES, ARENA1_FULL_CUMULATIVE_BYTES, ARENA2_PRO_CUMULATIVE_BYTES,
    ARENA3_ULTRA_CUMULATIVE_BYTES,
};

/// `.dnam-bin` File Header magic (`"DNBN"`).
pub const DNBN_MAGIC: [u8; 4] = *b"DNBN";
/// `.dnam-bin` Runtime Metadata Record magic (`"DBRM"`).
pub const DBRM_MAGIC: [u8; 4] = *b"DBRM";
/// Overflow `CP1` stream header magic (`"CP1\0"`).
pub const CP1_STREAM_MAGIC: [u8; 4] = *b"CP1\0";
/// Embedded SONA Cell Descriptor magic (`"SONA"`).
pub const SONA_CELL_MAGIC: [u8; 4] = *b"SONA";

/// Supported container major version (`4`).
pub const DNBN_VERSION_MAJOR: u16 = 4;
/// Supported container minor version (`0`).
pub const DNBN_VERSION_MINOR: u16 = 0;

/// Fixed cumulative byte footprints (`[Lite, Full, Pro, Ultra]`) per `ref[container.dnam.tier-prefix]`.
pub const ARENA_TIER_BYTES: [usize; 4] = [
    ARENA0_LITE_BYTES,
    ARENA1_FULL_CUMULATIVE_BYTES,
    ARENA2_PRO_CUMULATIVE_BYTES,
    ARENA3_ULTRA_CUMULATIVE_BYTES,
];
/// Byte size of the optional unwatermarked `"CEXT"` dispatch extension (`1,280 B`).
pub const CEXT_DISPATCH_BYTES: usize = 1_280;
/// Byte size of one `"TSYN"` physical tempo-sync entry (`40 B`).
pub const TSYN_ENTRY_BYTES: usize = 40;
/// Maximum physical controls `P` supported in `.dnam-bin` (`255`).
pub const MAX_NUM_CONTROLS: usize = 255;

/// `DnamBinHeaderWire::flags` bit 4: single-input / mono-in capability.
pub const DNBN_FLAG_SINGLE_INPUT: u16 = 1 << 4;
/// `DnamBinHeaderWire::flags` bits 5..=6 (evicted Freeze bits; must always equal `0`).
pub const DNBN_FLAG_FREEZE_EVICTED_MASK: u16 = 0b0110_0000;
/// `DnamBinHeaderWire::flags` bit 7: wet polarity inversion (`-eff_trim`).
pub const DNBN_FLAG_WET_POLARITY_INV: u16 = 1 << 7;
/// `DnamBinHeaderWire::flags` bit 8: auto-unity leveling enabled.
pub const DNBN_FLAG_AUTO_UNITY: u16 = 1 << 8;
/// `DnamBinHeaderWire::flags` bit 9: stereo-out topology.
pub const DNBN_FLAG_STEREO_OUT: u16 = 1 << 9;
/// `DnamBinHeaderWire::flags` reserved high bits `10..=15` (must equal `0`).
pub const DNBN_FLAG_RESERVED_HI_MASK: u16 = 0xFC00;

/// `"TSYN"` `flags` bit 0: clamp directly to physical bounds without octave folding.
pub const TSYN_FLAG_CLAMP_ONLY: u32 = 1 << 0;
/// `"CP1\0"` header `flags` bit 0: overflow stream carries `ultra_bytes` slices.
pub const CP1_STREAM_FLAG_HAS_ULTRA: u8 = 1 << 0;

/// Rounds `len` upward to the next `64 B` cache-line boundary (`(len + 63) & !63`).
#[inline]
#[must_use]
pub const fn align64(len: usize) -> usize {
    (len + 63) & !63
}

/// Fixed `64 B` `.dnam-bin` header (`"DNBN"`, bytes `0..64`, `ref[container.dnam.vm-binary]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DnamBinHeaderWire {
    /// Magic tag: `"DNBN"`.
    pub magic: [u8; 4],
    /// Major format version (`4`).
    pub version_major: U16,
    /// Minor format version (`0`).
    pub version_minor: U16,
    /// Original emitted ceiling tier index (`0..=3`, immutable across `REQ-5.6` truncation).
    pub max_present_tier: u8,
    /// Reserved byte (`0`).
    pub reserved0: u8,
    /// Bitmask: bits `0..=3` tier mask, bit 4 `SINGLE_INPUT`, bits `5..=6` strictly 0,
    /// bit 7 `WET_POLARITY_INV`, bit 8 `AUTO_UNITY`, bit 9 `STEREO_OUT`, bits `10..=15` strictly 0.
    pub flags: U16,
    /// Total physical controls `P` (`0..=255`).
    pub num_controls: U16,
    /// Count of `40 B` `"TSYN"` rows (`tsyn_len == tsyn_count * 40`).
    pub tsyn_count: U16,
    /// Canonical training/reference sample rate in Hz (`44_100..=96_000`).
    pub reference_sample_rate_hz: U32,
    /// Castagnoli `CRC-32C` over `[0..20) || [24..arena0_offset)`.
    pub prefix_crc32c: U32,
    /// `64 B`-aligned offset of Arena 0 relative to byte 0 of `.dnam-bin` (`>= 128`).
    pub arena0_offset: U32,
    /// Cumulative end offsets (`arena0_offset + ARENA_TIER_BYTES[t]`) for `t <= max_present_tier`, else `0`.
    pub tier_end_offsets: [U32; 4],
    /// Cumulative `CRC-32C` over `[arena0_offset..tier_end_offsets[t])` for `t <= max_present_tier`, else `0`.
    pub tier_arena_crc32c: [U32; 4],
    /// Byte length of the immediately following `"DBRM"` record (`64`).
    pub runtime_meta_bytes: U16,
    /// Reserved high flags word (`0`).
    pub reserved_flags_hi: U16,
}

const _: () = assert!(size_of::<DnamBinHeaderWire>() == 64);
const _: () = assert!(align_of::<DnamBinHeaderWire>() == 1);
const _: () = assert!(offset_of!(DnamBinHeaderWire, prefix_crc32c) == 20);
const _: () = assert!(offset_of!(DnamBinHeaderWire, arena0_offset) == 24);
const _: () = assert!(offset_of!(DnamBinHeaderWire, tier_end_offsets) == 28);
const _: () = assert!(offset_of!(DnamBinHeaderWire, tier_arena_crc32c) == 44);
const _: () = assert!(offset_of!(DnamBinHeaderWire, runtime_meta_bytes) == 60);

/// Fixed `64 B` `.dnam-bin` Runtime Metadata Record (`"DBRM"`, bytes `64..128`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DnamBinRuntimeMetaWire {
    /// Magic tag: `"DBRM"`.
    pub magic: [u8; 4],
    /// `64 B`-aligned offset of `shared_controls`, or `0` iff `shared_controls_len == 0`.
    pub shared_controls_offset: U32,
    /// `num_controls * 64 + cp1_overflow_bytes`.
    pub shared_controls_len: U32,
    /// `64 B`-aligned offset of `"TSYN"`, or `0` iff `tsyn_len == 0`.
    pub tsyn_offset: U32,
    /// `tsyn_count * 40`.
    pub tsyn_len: U32,
    /// `64 B`-aligned offset of `"CEXT"`, or `0` iff `cext_len == 0`.
    pub cext_offset: U32,
    /// Strictly `0` or `1,280`.
    pub cext_len: U32,
    /// External delay ring log2 frames (`0` or `7..=19`).
    pub ext_ring_pow2_frames: u8,
    /// External delay line count (`0..=16`).
    pub ext_ring_lines: u8,
    /// External delay sample format (`0 = fp16`, `1 = f32`).
    pub ext_ring_format: u8,
    /// Reserved byte (`0`).
    pub reserved0: u8,
    /// Declared `48 kHz` `Op-T` transport tank delay ceiling (`<= 48_000`).
    pub declared_max_delay_48k: U32,
    /// `5 Hz` wet DC blocker pole (`48 kHz`).
    pub dc_blocker_pole_5hz: F32,
    /// Declared decay tail time in milliseconds (`>= 0.0`).
    pub tail_time_ms: F32,
    /// Learned `48 kHz` `LinearReadout` leak pole (`[0.0, MAX_PARALRU_R_F32]`).
    pub ro_pole_48k: F32,
    /// Switch crossfade duration in samples.
    pub switch_xfade_samples: U16,
    /// Reserved conditional rule count (`0`; conditional rules live inline in descriptors).
    pub reserved_cond_zero: U16,
    /// First `8 B` of SHA-256 content hash prefix.
    pub content_hash_prefix: [u8; 8],
    /// Trailing `"CP1\0"` overflow stream byte length (`0` or `>= 8`).
    pub cp1_overflow_bytes: U32,
}

const _: () = assert!(size_of::<DnamBinRuntimeMetaWire>() == 64);
const _: () = assert!(align_of::<DnamBinRuntimeMetaWire>() == 1);
const _: () = assert!(offset_of!(DnamBinRuntimeMetaWire, shared_controls_offset) == 4);
const _: () = assert!(offset_of!(DnamBinRuntimeMetaWire, ext_ring_pow2_frames) == 28);
const _: () = assert!(offset_of!(DnamBinRuntimeMetaWire, declared_max_delay_48k) == 32);
const _: () = assert!(offset_of!(DnamBinRuntimeMetaWire, ro_pole_48k) == 44);
const _: () = assert!(offset_of!(DnamBinRuntimeMetaWire, cp1_overflow_bytes) == 60);

/// Switch-coupled conditional clamp rule (`8 B` LE wire struct) embedded inside
/// [`ControlClampDescriptorWire`].
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConditionalClampRuleWire {
    /// Conditioning `switch` control index (`0..num_controls`).
    pub switch_ctrl_idx: U16,
    /// Matching switch option index (`0..=15`).
    pub switch_opt_idx: u8,
    /// Rule opcode: `0 = none`, `1 = max_ceil`, `2 = min_floor`, `3 = and_prefix`,
    /// `5 = and_max_keep`, `6 = and_min_end`.
    pub rule_op: u8,
    /// Physical bound value for this conditional rule.
    pub bound_val: F32,
}

const _: () = assert!(size_of::<ConditionalClampRuleWire>() == 8);
const _: () = assert!(align_of::<ConditionalClampRuleWire>() == 1);

/// Fixed `64 B` wire layout for a control clamp descriptor (`ref[dsp.ctrl.range.normalize-clamp]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ControlClampDescriptorWire {
    /// Control kind (`0 = range`, `1 = switch`, `0xFF = unused`).
    pub ctrl_kind: u8,
    /// Lobatto degree (`range`, `0..=3`) or `num_options - 1` (`switch`, `0..=15`).
    pub lobatto_degree: u8,
    /// Bitmask of represented/valid switch options (`!= 0` for `switch`).
    pub active_options_mask: U16,
    /// Default physical value (`range`) or option index (`switch`).
    pub default_val: F32,
    /// Base lower physical bound.
    pub safe_min_base: F32,
    /// Base upper physical bound.
    pub safe_max_base: F32,
    /// Precomputed normalization scale (`binary64`-derived `2.0 / (max - min)`).
    pub scale_norm: F32,
    /// Precomputed normalization offset.
    pub offset_norm: F32,
    /// Normalized Lobatto anchors `[a1, a2, a3]` (`a1 = StarCenter`).
    pub anchor_nodes_norm: [F32; 3],
    /// Reserved alignment field (`0`).
    pub _reserved: U32,
    /// Three inline switch-conditional clamp slots (`24 B`).
    pub cond_rules: [ConditionalClampRuleWire; 3],
}

const _: () = assert!(size_of::<ControlClampDescriptorWire>() == 64);
const _: () = assert!(align_of::<ControlClampDescriptorWire>() == 1);
const _: () = assert!(offset_of!(ControlClampDescriptorWire, cond_rules) == 40);

/// Pure-binary `40 B` `"TSYN"` tempo-sync row (`ref[dsp.tempo.subdivisions]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TempoSyncEntryWire {
    /// Target physical control index (`< num_controls`).
    pub ctrl_idx: U16,
    /// Conditioning switch control index when `has_switch_cond == 1`.
    pub switch_ctrl_idx: U16,
    /// Conditioning switch option index (`0..=15`).
    pub switch_opt_idx: u8,
    /// `0 = unconditional row`, `1 = switch-conditional row`.
    pub has_switch_cond: u8,
    /// Physical unit mode: `0 = PeriodMs`, `1 = FrequencyHz`.
    pub mode: u8,
    /// Default subdivision index (`0..=11`).
    pub default_subdiv: u8,
    /// Bit 0: `0 = OCTAVE_FOLD`, `1 = CLAMP_ONLY`; bits `1..=31` strictly `0`.
    pub flags: U32,
    /// Precomputed rational subdivision ratio (`> 0.0`).
    pub div_ratio: F32,
    /// Finite unit scaling factor (`> 0.0`).
    pub bpm_factor: F32,
    /// Three finite Lobatto physical calibration anchors `[v_min, v_mid, v_max]`.
    pub lobatto_anchors: [F32; 3],
    /// Finite lower physical control bound (`<= declared_max`).
    pub declared_min: F32,
    /// Finite upper physical control bound (`>= declared_min`).
    pub declared_max: F32,
}

const _: () = assert!(size_of::<TempoSyncEntryWire>() == TSYN_ENTRY_BYTES);
const _: () = assert!(align_of::<TempoSyncEntryWire>() == 1);

/// Unwatermarked `1,280 B` extended 3 kHz dispatch matrices (`"CEXT"`, `stage_targets[16..32]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CextDispatchWire {
    /// Row-major `w_ctrl_ext[row - 16][0..16]` (`1,024 B`).
    pub w_ctrl_ext: [[F32; 16]; 16],
    /// Row-major Ultra extension `w_ctrl_ultra_ext[row - 16][0..4]` (`256 B`).
    pub w_ctrl_ultra_ext: [[F32; 4]; 16],
}

const _: () = assert!(size_of::<CextDispatchWire>() == CEXT_DISPATCH_BYTES);
const _: () = assert!(align_of::<CextDispatchWire>() == 1);

/// Inline `CP1` contributor directory inside Arena 0 Control Plane (`28 B`, `0..=8` entries).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Cp1InlineDirWire {
    /// Active inline slot count (`0..=8`).
    pub cp1_dir_count: U16,
    /// Directory flags (must equal `0`).
    pub flags: U16,
    /// Up to 8 packed 3-byte records `[ctrl_idx, (ctrl_kind << 4) | lanes, aux]`.
    pub entries: [[u8; 3]; 8],
}

const _: () = assert!(size_of::<Cp1InlineDirWire>() == 28);
const _: () = assert!(align_of::<Cp1InlineDirWire>() == 1);

/// Overflow `"CP1\0"` binary stream header (`8 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Cp1StreamHeaderWire {
    /// Magic tag: `"CP1\0"`.
    pub magic: [u8; 4],
    /// Contributor slot count `C`.
    pub count: U16,
    /// Bit 0 = `has_ultra`; remaining bits `0`.
    pub flags: u8,
    /// Reserved byte (`0`).
    pub reserved: u8,
}

const _: () = assert!(size_of::<Cp1StreamHeaderWire>() == 8);
const _: () = assert!(align_of::<Cp1StreamHeaderWire>() == 1);

/// Overflow `"CP1\0"` directory entry (`4 B` per contributor slot).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Cp1StreamDirEntryWire {
    /// Target physical control index (`< num_controls`).
    pub control_idx: U16,
    /// Packed `((ctrl_kind & 3) << 2) | (lanes & 3)`.
    pub raw_kl: u8,
    /// Option index (`Switch`), partner control ID (`BilinearCrossTerm`), or `0xFF`.
    pub aux: u8,
}

const _: () = assert!(size_of::<Cp1StreamDirEntryWire>() == 4);
const _: () = assert!(align_of::<Cp1StreamDirEntryWire>() == 1);

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use zerocopy::FromZeros;

    use super::*;

    #[test]
    fn test_prefix_wire_structs_roundtrip() {
        let mut hdr = DnamBinHeaderWire::new_zeroed();
        hdr.magic = DNBN_MAGIC;
        hdr.version_major = U16::new(DNBN_VERSION_MAJOR);
        let parsed = DnamBinHeaderWire::ref_from_bytes(hdr.as_bytes()).unwrap();
        assert_eq!(parsed.magic, DNBN_MAGIC);
        assert_eq!(parsed.version_major.get(), 4);

        let meta = DnamBinRuntimeMetaWire::new_zeroed();
        assert_eq!(meta.as_bytes().len(), 64);

        let desc = ControlClampDescriptorWire::new_zeroed();
        assert_eq!(desc.as_bytes().len(), 64);

        let tsyn = TempoSyncEntryWire::new_zeroed();
        assert_eq!(tsyn.as_bytes().len(), 40);

        let cext = CextDispatchWire::new_zeroed();
        assert_eq!(cext.as_bytes().len(), 1_280);
    }

    proptest! {
        #[test]
        fn prop_wire_header_and_tsyn_zerocopy_roundtrip(
            max_tier in 0_u8..=3,
            sample_rate in 44_100_u32..=96_000,
            div_ratio in 0.1_f32..8.0,
            bpm_factor in 0.25_f32..4.0,
        ) {
            let mut hdr = DnamBinHeaderWire::new_zeroed();
            hdr.magic = DNBN_MAGIC;
            hdr.version_major = U16::new(DNBN_VERSION_MAJOR);
            hdr.max_present_tier = max_tier;
            hdr.reference_sample_rate_hz = U32::new(sample_rate);
            let hdr_back = DnamBinHeaderWire::ref_from_bytes(hdr.as_bytes()).unwrap();
            prop_assert_eq!(hdr_back.max_present_tier, max_tier);
            prop_assert_eq!(hdr_back.reference_sample_rate_hz.get(), sample_rate);

            let mut tsyn = TempoSyncEntryWire::new_zeroed();
            tsyn.div_ratio = F32::new(div_ratio);
            tsyn.bpm_factor = F32::new(bpm_factor);
            let tsyn_back = TempoSyncEntryWire::ref_from_bytes(tsyn.as_bytes()).unwrap();
            prop_assert_eq!(tsyn_back.div_ratio.get().to_bits(), div_ratio.to_bits());
            prop_assert_eq!(tsyn_back.bpm_factor.get().to_bits(), bpm_factor.to_bits());
        }
    }
}
