//! Zero-allocation `v3.0` (`"sona"`) directory bounds validation, `dbin_prefix` alignment checks,
//! and `DCRC` authentication record encoding/parsing (`ref[container.dnam.preamble-directory]`).

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, IntoBytes};

use super::super::structs::{
    DCRC_MAGIC, DCRC_RECORD_SIZE, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE, DcrcAuthRecordWire,
    DmueFooterWire, DnamDirectoryWire, V3_HEADER_OFFSET, V3_PREAMBLE_SIZE,
};
use super::{DEFAULT_MAX_MUTABLE_CAPACITY, align64_u64};
use crate::schema::DnamError;

/// Allocation-free (`Copy`) container wire error avoiding `DnamError` `String` drop glue on RT paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContainerWireError {
    BufferTooSmall {
        expected: usize,
        actual: usize,
    },
    InvalidMagic([u8; 4]),
    UnsupportedVersion {
        major: u16,
        minor: u16,
    },
    UnalignedOffset {
        field: &'static str,
        offset: u32,
    },
    Crc32cMismatch {
        region: &'static str,
        expected: u32,
        computed: u32,
    },
    SchemaStatic(&'static str),
    EncryptedArenaRequiresDeviceKey,
    InvalidSampleRate(u32),
}

impl ContainerWireError {
    #[cold]
    #[inline(never)]
    pub(crate) fn into_dnam_error(self) -> DnamError {
        match self {
            Self::BufferTooSmall { expected, actual } => {
                DnamError::BufferTooSmall { expected, actual }
            }
            Self::InvalidMagic(m) => DnamError::InvalidMagic(m),
            Self::UnsupportedVersion { major, minor } => {
                DnamError::UnsupportedVersion { major, minor }
            }
            Self::UnalignedOffset { field, offset } => DnamError::UnalignedOffset { field, offset },
            Self::Crc32cMismatch {
                region,
                expected,
                computed,
            } => DnamError::Crc32cMismatch {
                region,
                expected,
                computed,
            },
            Self::SchemaStatic(msg) => {
                DnamError::SchemaValidationError(alloc::string::String::from(msg))
            }
            Self::EncryptedArenaRequiresDeviceKey => DnamError::EncryptedArenaRequiresDeviceKey,
            Self::InvalidSampleRate(sr) => DnamError::InvalidSampleRate(sr),
        }
    }
}

/// Local 64-byte Castagnoli `CRC32C` integrity record stored in `auth_extent` for non-`DAU1` v3 `.dnam` containers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DcrcAuthRecord {
    pub crc_prefix: u32,
    pub crc_tiers: [u32; 4],
    pub num_controls: u16,
    pub flags_hint: u16,
    pub tail_time_ms_bits: u32,
    pub crc_dbin_prefix: u32,
}

impl DcrcAuthRecord {
    /// Serializes this 64-byte `DCRC` record into little-endian bytes via [`DcrcAuthRecordWire`].
    #[inline]
    pub fn to_bytes(&self) -> [u8; DCRC_RECORD_SIZE] {
        let wire = DcrcAuthRecordWire {
            magic: DCRC_MAGIC,
            crc_prefix: U32::new(self.crc_prefix),
            crc_tiers: [
                U32::new(self.crc_tiers[0]),
                U32::new(self.crc_tiers[1]),
                U32::new(self.crc_tiers[2]),
                U32::new(self.crc_tiers[3]),
            ],
            num_controls: U16::new(self.num_controls),
            flags_hint: U16::new(self.flags_hint),
            tail_time_ms_bits: U32::new(self.tail_time_ms_bits),
            crc_dbin_prefix: U32::new(self.crc_dbin_prefix),
            reserved_pad: [0u8; 28],
        };
        let mut out = [0u8; DCRC_RECORD_SIZE];
        out.copy_from_slice(wire.as_bytes());
        out
    }
}

/// Zero-drop-glue (`Copy` error) `DCRC` record parser for real-time container reloaders.
pub(crate) fn parse_dcrc_record_rt(
    auth_slice: &[u8],
) -> Result<Option<DcrcAuthRecord>, ContainerWireError> {
    if auth_slice.len() < 4 || auth_slice[0..4] != DCRC_MAGIC {
        return Ok(None);
    }
    let wire = DcrcAuthRecordWire::ref_from_bytes(auth_slice)
        .map_err(|_| ContainerWireError::SchemaStatic("invalid DCRC auth record"))?;
    if wire.reserved_pad.iter().any(|&b| b != 0) {
        return Err(ContainerWireError::SchemaStatic("invalid DCRC auth record"));
    }
    Ok(Some(DcrcAuthRecord {
        crc_prefix: wire.crc_prefix.get(),
        crc_tiers: [
            wire.crc_tiers[0].get(),
            wire.crc_tiers[1].get(),
            wire.crc_tiers[2].get(),
            wire.crc_tiers[3].get(),
        ],
        num_controls: wire.num_controls.get(),
        flags_hint: wire.flags_hint.get(),
        tail_time_ms_bits: wire.tail_time_ms_bits.get(),
        crc_dbin_prefix: wire.crc_dbin_prefix.get(),
    }))
}

/// Parses a `DCRC` local integrity record if `auth_slice` starts with `b"DCRC"`.
pub fn parse_dcrc_record(auth_slice: &[u8]) -> Result<Option<DcrcAuthRecord>, DnamError> {
    parse_dcrc_record_rt(auth_slice).map_err(ContainerWireError::into_dnam_error)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidatedDirectoryExtents {
    pub retained_tier: u8,
    pub header_extent: (usize, usize),
    pub shared_extent: (usize, usize),
    pub auth_extent: (usize, usize),
    pub dbin_extent: (usize, usize),
    pub tier_start: usize,
    pub tier_ends: [usize; 4],
    pub mutable_extent: (usize, usize),
}

fn check_zero_pad(data: &[u8], start: u64, end: u64) -> Result<(), ContainerWireError> {
    let s = start as usize;
    let e = end as usize;
    if data[s..e].iter().any(|&b| b != 0) {
        return Err(ContainerWireError::SchemaStatic("nonzero padding"));
    }
    Ok(())
}

/// Validates a 128-byte [`DnamDirectoryWire`], region monotonicity, optional `dbin_prefix`, zero
/// alignment padding, tier bounds, and trailing `DMUE` footer without heap allocation.
pub(crate) fn validate_v3_directory_extents_rt(
    data: &[u8],
    maximum: u8,
) -> Result<ValidatedDirectoryExtents, ContainerWireError> {
    if data.len() < V3_HEADER_OFFSET {
        return Err(ContainerWireError::SchemaStatic("short structure"));
    }
    let dir = DnamDirectoryWire::ref_from_bytes(&data[V3_PREAMBLE_SIZE..V3_HEADER_OFFSET])
        .map_err(|_| ContainerWireError::SchemaStatic("invalid directory"))?;
    let header_offset = dir.immutable_header_offset.get();
    let header_length = dir.immutable_header_length.get();
    let shared_offset = dir.shared_binary_offset.get();
    let shared_length = dir.shared_binary_length.get();
    let auth_offset = dir.authentication_envelope_offset.get();
    let auth_length = dir.authentication_envelope_length.get();
    let dbin_offset = dir.dbin_offset.get();
    let dbin_prefix_bytes = dir.dbin_prefix_bytes.get();
    let tier_start = dir.tier_start.get();
    let ends_u64 = dir.original_tier_ends.map(U64::get);

    if dir.reserved_tail.iter().any(|&b| b != 0)
        || header_offset != (V3_HEADER_OFFSET as u64)
        || header_length == 0
        || auth_length == 0
    {
        return Err(ContainerWireError::SchemaStatic("invalid directory"));
    }

    let header_end = header_offset
        .checked_add(header_length)
        .ok_or(ContainerWireError::SchemaStatic("header extent overflow"))?;
    if header_end > (u32::MAX as u64) || Some(shared_offset) != align64_u64(header_end) {
        return Err(ContainerWireError::SchemaStatic("invalid directory"));
    }
    let shared_end = shared_offset
        .checked_add(shared_length)
        .ok_or(ContainerWireError::SchemaStatic("shared extent overflow"))?;
    if shared_end > (u32::MAX as u64) || Some(auth_offset) != align64_u64(shared_end) {
        return Err(ContainerWireError::SchemaStatic("invalid directory"));
    }
    let auth_end = auth_offset
        .checked_add(auth_length)
        .ok_or(ContainerWireError::SchemaStatic("auth extent overflow"))?;
    if auth_end > (u32::MAX as u64) {
        return Err(ContainerWireError::SchemaStatic("auth extent overflow"));
    }

    let dbin_end = if dbin_prefix_bytes > 0 {
        if shared_length > 0 {
            return Err(ContainerWireError::SchemaStatic(
                "shared_binary_length must equal 0 when dbin_prefix_bytes > 0",
            ));
        }
        if (dbin_offset & 63) != 0 || (dbin_prefix_bytes & 63) != 0 {
            return Err(ContainerWireError::SchemaStatic(
                "misaligned dbin_offset or dbin_prefix_bytes",
            ));
        }
        let auth_aligned = align64_u64(auth_end)
            .ok_or(ContainerWireError::SchemaStatic("auth extent overflow"))?;
        if dbin_offset < auth_aligned || dbin_offset > (u32::MAX as u64) {
            return Err(ContainerWireError::SchemaStatic(
                "dbin_offset precedes aligned auth_extent",
            ));
        }
        let end = dbin_offset
            .checked_add(dbin_prefix_bytes)
            .ok_or(ContainerWireError::SchemaStatic("dbin extent overflow"))?;
        if end > (u32::MAX as u64)
            || Some(tier_start) != align64_u64(end)
            || tier_start > (u32::MAX as u64)
        {
            return Err(ContainerWireError::SchemaStatic(
                "invalid tier_start after dbin_prefix",
            ));
        }
        end
    } else {
        if dbin_offset != 0 {
            return Err(ContainerWireError::SchemaStatic(
                "non-zero dbin_offset when dbin_prefix_bytes == 0",
            ));
        }
        if Some(tier_start) != align64_u64(auth_end) || tier_start > (u32::MAX as u64) {
            return Err(ContainerWireError::SchemaStatic("invalid tier start"));
        }
        0
    };

    let mut previous = tier_start;
    for (index, &end) in ends_u64.iter().enumerate() {
        if index <= (maximum as usize) {
            if !end.is_multiple_of(64) {
                return Err(ContainerWireError::UnalignedOffset {
                    field: "tier_ends",
                    offset: end as u32,
                });
            }
            if end <= previous || end > (u32::MAX as u64) {
                return Err(ContainerWireError::SchemaStatic("invalid tier end"));
            }
            previous = end;
        } else if end != 0 {
            return Err(ContainerWireError::SchemaStatic("unexpected tier end"));
        }
    }

    if data.len() < V3_HEADER_OFFSET + DMUE_FOOTER_SIZE {
        return Err(ContainerWireError::SchemaStatic("invalid mutable trailer"));
    }
    let footer = DmueFooterWire::ref_from_bytes(&data[data.len() - DMUE_FOOTER_SIZE..])
        .map_err(|_| ContainerWireError::SchemaStatic("invalid mutable trailer"))?;
    let mutable_capacity = footer.frame_len.get();
    let footer_reserved = footer.reserved.get();
    if &footer.magic != b"DMUE"
        || footer_reserved != 0
        || mutable_capacity < (DMU1_HEADER_SIZE as u64 + 2)
        || mutable_capacity > (DEFAULT_MAX_MUTABLE_CAPACITY as u64)
        || mutable_capacity > ((data.len() - DMUE_FOOTER_SIZE) as u64)
    {
        return Err(ContainerWireError::SchemaStatic("invalid mutable trailer"));
    }
    let mutable_offset = data.len() - DMUE_FOOTER_SIZE - (mutable_capacity as usize);
    let mut retained_opt = None;
    for (idx, &end) in ends_u64[..=(maximum as usize)].iter().enumerate() {
        if end == (mutable_offset as u64) {
            retained_opt = Some(idx as u8);
            break;
        }
    }
    let retained_tier =
        retained_opt.ok_or(ContainerWireError::SchemaStatic("invalid retained end"))?;

    check_zero_pad(data, header_end, shared_offset)?;
    check_zero_pad(data, shared_end, auth_offset)?;
    if dbin_prefix_bytes > 0 {
        check_zero_pad(data, auth_end, dbin_offset)?;
        check_zero_pad(data, dbin_end, tier_start)?;
    } else {
        check_zero_pad(data, auth_end, tier_start)?;
    }

    Ok(ValidatedDirectoryExtents {
        retained_tier,
        header_extent: (header_offset as usize, header_length as usize),
        shared_extent: (shared_offset as usize, shared_length as usize),
        auth_extent: (auth_offset as usize, auth_length as usize),
        dbin_extent: (dbin_offset as usize, dbin_prefix_bytes as usize),
        tier_start: tier_start as usize,
        tier_ends: [
            ends_u64[0] as usize,
            ends_u64[1] as usize,
            ends_u64[2] as usize,
            ends_u64[3] as usize,
        ],
        mutable_extent: (mutable_offset, mutable_capacity as usize),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcrc_crc_dbin_prefix_roundtrip() {
        let mut wire = DcrcAuthRecord {
            crc_prefix: 11,
            crc_tiers: [1, 2, 3, 4],
            num_controls: 5,
            flags_hint: 0x060F,
            tail_time_ms_bits: 0x43FA_0000,
            crc_dbin_prefix: 0xDEAD_BEEF,
        }
        .to_bytes();
        let parsed = parse_dcrc_record(&wire).unwrap().unwrap();
        assert_eq!(parsed.crc_dbin_prefix, 0xDEAD_BEEF);
        wire[63] = 1;
        assert!(parse_dcrc_record(&wire).is_err());
    }
}
