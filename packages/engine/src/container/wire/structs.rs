//! Declarative `#[repr(C)]` `zerocopy` wire structs for `.dnam` `v3.0` (`"sona"`) binary headers,
//! directories, mutable frames, and `DCRC` records (`ref[sdk.rust.zerocopy-wire-structs]`,
//! `ref[container.dnam.preamble-directory]`, `ref[container.dnam.mutable-frame]`).

use core::mem::{align_of, size_of};

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub use crate::stages::weights::SonaCellDescWire;

/// Fixed 64-byte preamble size (`0..64`, `ref[container.dnam.preamble-directory]`).
pub const V3_PREAMBLE_SIZE: usize = 64;
/// Fixed 128-byte directory size (`64..192`, `ref[container.dnam.preamble-directory]`).
pub const V3_DIRECTORY_SIZE: usize = 128;
/// Fixed offset where the immutable JSON header begins (`192`, `ref[container.dnam.preamble-directory]`).
pub const V3_HEADER_OFFSET: usize = V3_PREAMBLE_SIZE + V3_DIRECTORY_SIZE;
/// Fixed 16-byte `DMU1` mutable frame header size (`ref[container.dnam.mutable-frame]`).
pub const DMU1_HEADER_SIZE: usize = 16;
/// Fixed 16-byte `DMUE` mutable footer size (`ref[container.dnam.mutable-frame]`).
pub const DMUE_FOOTER_SIZE: usize = 16;
/// Magic bytes for a 64-byte local Castagnoli `CRC32C` integrity record in an unsigned `auth_extent`.
pub const DCRC_MAGIC: [u8; 4] = *b"DCRC";
/// Fixed byte size of a `DCRC` local integrity record (`64 B`).
pub const DCRC_RECORD_SIZE: usize = 64;

/// Region 1: Fixed `64 B` cache-line-aligned container preamble (`0..64`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct DnamPreambleWire {
    /// ASCII container magic (`b"DNAM"` or `b"DNSL"`).
    pub magic: [u8; 4],
    /// Wire container major version (`3`).
    pub version_major: U16,
    /// Wire container minor version (`0`).
    pub version_minor: U16,
    /// Protection profile (`0 = Open`, `1 = DeviceAllowlisted`, `2 = EnclaveSealed`).
    pub protection_profile: u8,
    /// Authentication envelope kind (`0 = creator-only`, `1 = creator + marketplace`).
    pub auth_kind: u8,
    /// Maximum embedded tier index (`0 = Lite..3 = Ultra`).
    pub maximum_tier: u8,
    /// Reserved flags byte (must be `0`).
    pub reserved_flag: u8,
    /// Training sample rate in Hz (`48_000`).
    pub sample_rate_train_hz: U32,
    /// Reserved zero-filled tail padding (`16..64`).
    pub reserved_pad: [u8; 48],
}

const _: () = assert!(size_of::<DnamPreambleWire>() == V3_PREAMBLE_SIZE);
const _: () = assert!(align_of::<DnamPreambleWire>() == 64);

/// Region 2: Fixed `128 B` directory table (`64..192`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DnamDirectoryWire {
    /// File offset of Region 3 immutable JCS JSON header (`192`).
    pub immutable_header_offset: U64,
    /// Byte length of unpadded UTF-8 JCS header.
    pub immutable_header_length: U64,
    /// File offset of `64 B`-aligned Region 4 shared extent.
    pub shared_binary_offset: U64,
    /// Byte length of Region 4 shared extent.
    pub shared_binary_length: U64,
    /// File offset of `64 B`-aligned Region 5 authentication envelope.
    pub authentication_envelope_offset: U64,
    /// Byte length of Region 5 authentication envelope.
    pub authentication_envelope_length: U64,
    /// Reserved offset slot (must be `0`).
    pub reserved_offset: U64,
    /// Reserved capacity slot (must be `0`).
    pub reserved_capacity: U64,
    /// File offset where `Arena 0` begins (`64 B`-aligned).
    pub tier_start: U64,
    /// Cumulative original tier boundary offsets `[lite_end, full_end, pro_end, ultra_end]`.
    pub original_tier_ends: [U64; 4],
    /// Reserved zero-filled tail padding (`24 B`).
    pub reserved_tail: [u8; 24],
}

const _: () = assert!(size_of::<DnamDirectoryWire>() == V3_DIRECTORY_SIZE);
const _: () = assert!(align_of::<DnamDirectoryWire>() == 1);

/// Fixed `16 B` `DMU1` mutable JSON frame header (`ref[container.dnam.mutable-frame]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct Dmu1HeaderWire {
    /// ASCII frame magic (`b"DMU1"`).
    pub magic: [u8; 4],
    /// Castagnoli `CRC-32C` over `"DMU1" || u64le(json_len) || JSON bytes`.
    pub crc32c: U32,
    /// Unpadded UTF-8 JSON object byte length (`>= 2`).
    pub json_len: U64,
}

const _: () = assert!(size_of::<Dmu1HeaderWire>() == DMU1_HEADER_SIZE);
const _: () = assert!(align_of::<Dmu1HeaderWire>() == 1);

/// Fixed `16 B` `DMUE` end-of-container mutable footer (`ref[container.dnam.mutable-frame]`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DmueFooterWire {
    /// ASCII footer magic (`b"DMUE"`).
    pub magic: [u8; 4],
    /// Byte span of the mutable frame immediately preceding this footer.
    pub frame_len: U64,
    /// Reserved footer field (must be `0`).
    pub reserved: U32,
}

const _: () = assert!(size_of::<DmueFooterWire>() == DMUE_FOOTER_SIZE);
const _: () = assert!(align_of::<DmueFooterWire>() == 1);

/// Fixed `64 B` local `DCRC` Castagnoli `CRC-32C` integrity record in `auth_extent`.
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DcrcAuthRecordWire {
    /// ASCII integrity record magic (`b"DCRC"`).
    pub magic: [u8; 4],
    /// `CRC-32C` over container bytes `[0..auth_offset]`.
    pub crc_prefix: U32,
    /// Cumulative tier arena `CRC-32C` digests `[lite, full, pro, ultra]`.
    pub crc_tiers: [U32; 4],
    /// Total control count `P`.
    pub num_controls: U16,
    /// Encoded `FLAG_*` bitmask hint.
    pub flags_hint: U16,
    /// `f32::to_bits()` of `io_config.tail_time_ms`.
    pub tail_time_ms_bits: U32,
    /// Reserved zero-filled tail padding (`32 B`).
    pub reserved_pad: [u8; 32],
}

const _: () = assert!(size_of::<DcrcAuthRecordWire>() == DCRC_RECORD_SIZE);
const _: () = assert!(align_of::<DcrcAuthRecordWire>() == 1);

const _: () = assert!(size_of::<SonaCellDescWire>() == 16);
const _: () = assert!(align_of::<SonaCellDescWire>() == 1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zerocopy_container_wire_struct_roundtrips() {
        let preamble = DnamPreambleWire {
            magic: *b"DNAM",
            version_major: U16::new(3),
            version_minor: U16::new(0),
            protection_profile: 0,
            auth_kind: 1,
            maximum_tier: 3,
            reserved_flag: 0,
            sample_rate_train_hz: U32::new(48_000),
            reserved_pad: [0; 48],
        };
        let (decoded, tail) = DnamPreambleWire::read_from_prefix(preamble.as_bytes()).unwrap();
        assert!(tail.is_empty());
        assert_eq!(decoded, preamble);

        let dir = DnamDirectoryWire {
            immutable_header_offset: U64::new(V3_HEADER_OFFSET as u64),
            immutable_header_length: U64::new(256),
            shared_binary_offset: U64::new(448),
            shared_binary_length: U64::new(64),
            authentication_envelope_offset: U64::new(512),
            authentication_envelope_length: U64::new(64),
            reserved_offset: U64::new(0),
            reserved_capacity: U64::new(0),
            tier_start: U64::new(576),
            original_tier_ends: [
                U64::new(5_184),
                U64::new(8_512),
                U64::new(16_448),
                U64::new(31_296),
            ],
            reserved_tail: [0; 24],
        };
        assert_eq!(
            DnamDirectoryWire::ref_from_bytes(dir.as_bytes()).unwrap(),
            &dir
        );
    }
}
