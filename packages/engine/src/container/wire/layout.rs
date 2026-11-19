//! Canonical v3.0 (`"sona"`) `.dnam` (`b"DNAM"`) Open Layout, 64B Preamble, 128B Directory,
//! and `DMU1` / `DMUE` Mutable Suffix Framing (`ref[container.dnam.formats]`,
//! `ref[container.dnam.wire-framing]`, `ref[container.dnam.preamble-directory]`,
//! `ref[container.dnam.open-layout]`, `ref[container.dnam.mutable-frame]`,
//! `ref[container.dnam.mutable-json]`, `ref[container.dnam.tier-prefix]`,
//! `ref[container.dnam.sample-rate-anchor]`).

mod mutable_frame;

pub(crate) use mutable_frame::validate_dmu1_frame_rt;
pub use mutable_frame::{
    build_dmu1_mutable_frame, parse_and_validate_dmu1_frame, validate_dme1_frame_no_alloc,
    validate_dmu1_frame_no_alloc, validate_mutable_json_schema,
};

use alloc::vec;
use alloc::vec::Vec;

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, IntoBytes};

use super::structs::{
    DCRC_MAGIC, DCRC_RECORD_SIZE, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE, DcrcAuthRecordWire,
    DmueFooterWire, DnamDirectoryWire, DnamPreambleWire, V3_HEADER_OFFSET, V3_PREAMBLE_SIZE,
};
use crate::container::{CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DNAM_MAGIC};
use crate::provenance::sha256_digest;
use crate::schema::{DnamError, SchemaOptionExt, reject_if};

/// Maximum default mutable frame capacity for bounded structural validation (`4,096 B`).
pub const DEFAULT_MAX_MUTABLE_CAPACITY: usize = 4_096;
/// Default `DMU1` mutable frame capacity (`4,080 B`) so `DMU1 (4,080 B) + DMUE (16 B) == 4,096 B` (`% 64 == 0`).
pub const DEFAULT_DMU1_TRAILER_CAPACITY: usize = DEFAULT_MAX_MUTABLE_CAPACITY - DMUE_FOOTER_SIZE;

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
            reserved_pad: [0u8; 32],
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
    }))
}

/// Parses a `DCRC` local integrity record if `auth_slice` starts with `b"DCRC"`.
pub fn parse_dcrc_record(auth_slice: &[u8]) -> Result<Option<DcrcAuthRecord>, DnamError> {
    parse_dcrc_record_rt(auth_slice).map_err(ContainerWireError::into_dnam_error)
}

/// Domain-separated binary field encoder `F(label, value) = u32le(len(label)) || label || u64le(len(value)) || value`
/// (`ref[container.dnam.wire-framing]`, `ref[sec.sig.prefix-delegation]`).
pub fn encode_domain_field(label: &str, value: &[u8]) -> Vec<u8> {
    let label_bytes = label.as_bytes();
    let label_len = U32::new(label_bytes.len() as u32);
    let value_len = U64::new(value.len() as u64);
    let mut out = Vec::with_capacity(4 + label_bytes.len() + 8 + value.len());
    out.extend_from_slice(label_len.as_bytes());
    out.extend_from_slice(label_bytes);
    out.extend_from_slice(value_len.as_bytes());
    out.extend_from_slice(value);
    out
}

/// Aligns `value` up to a 64-byte boundary with `u64` overflow protection (`ref[container.dnam.preamble-directory]`).
#[inline]
pub const fn align64_u64(value: u64) -> Option<u64> {
    if value > u64::MAX - 63 {
        None
    } else {
        Some((value + 63) & !63)
    }
}

/// Validated `v3.0` (`"sona"`) `.dnam` layout (`ref[container.dnam.open-layout]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V3ContainerLayout {
    pub magic: [u8; 4],
    pub protection_profile: u8,
    pub auth_kind: u8,
    pub maximum_tier: u8,
    pub retained_tier: u8,
    pub sample_rate_train_hz: u32,
    pub header_extent: (usize, usize),
    pub shared_extent: (usize, usize),
    pub auth_extent: (usize, usize),
    pub tier_start: usize,
    pub tier_ends: [usize; 4],
    pub mutable_extent: (usize, usize),
}

/// Builds a canonical `v3.0` (`"sona"`) `.dnam` container (`magic = b"DNAM"`, `version = (3, 0)`)
/// with an explicit `protection_profile` (`0 = Open`, `1 = DeviceAllowlisted`, `2 = EnclaveSealed`).
pub fn build_v3_dnam_container_with_profile(
    protection_profile: u8,
    header_bytes: &[u8],
    shared_bytes: &[u8],
    auth_bytes: &[u8],
    tier_slices: &[&[u8]],
    mutable_frame_bytes: Option<&[u8]>,
    auth_kind: u8,
) -> Result<Vec<u8>, DnamError> {
    reject_if(
        protection_profile > 2
            || header_bytes.is_empty()
            || auth_bytes.is_empty()
            || tier_slices.is_empty()
            || tier_slices.len() > 4
            || auth_kind > 1,
        "invalid v3 container inputs",
    )?;
    for slice in tier_slices {
        reject_if(
            slice.is_empty() || !slice.len().is_multiple_of(64),
            "tier slice must be non-empty and 64-byte aligned",
        )?;
    }
    let header_offset = V3_HEADER_OFFSET as u64;
    let header_len = header_bytes.len() as u64;
    let shared_offset = header_offset
        .checked_add(header_len)
        .and_then(align64_u64)
        .or_schema_err("header extent overflow")?;
    let shared_len = shared_bytes.len() as u64;
    let auth_offset = shared_offset
        .checked_add(shared_len)
        .and_then(align64_u64)
        .or_schema_err("shared extent overflow")?;
    let auth_len = auth_bytes.len() as u64;
    let tier_start = auth_offset
        .checked_add(auth_len)
        .and_then(align64_u64)
        .or_schema_err("auth extent overflow")?;

    let mut ends = [0u64; 4];
    let mut cur_end = tier_start;
    for (i, slice) in tier_slices.iter().enumerate() {
        cur_end = cur_end
            .checked_add(slice.len() as u64)
            .or_schema_err("tier extent overflow")?;
        ends[i] = cur_end;
    }
    reject_if(cur_end > (u32::MAX as u64), "container exceeds u32::MAX")?;

    let mut out = vec![0u8; cur_end as usize];
    let preamble = DnamPreambleWire {
        magic: DNAM_MAGIC,
        version_major: U16::new(CONTAINER_VERSION_MAJOR),
        version_minor: U16::new(CONTAINER_VERSION_MINOR),
        protection_profile,
        auth_kind,
        maximum_tier: (tier_slices.len() - 1) as u8,
        reserved_flag: 0,
        sample_rate_train_hz: U32::new(48_000),
        reserved_pad: [0u8; 48],
    };
    let _ = preamble.write_to_prefix(&mut out);

    let directory = DnamDirectoryWire {
        immutable_header_offset: U64::new(header_offset),
        immutable_header_length: U64::new(header_len),
        shared_binary_offset: U64::new(shared_offset),
        shared_binary_length: U64::new(shared_len),
        authentication_envelope_offset: U64::new(auth_offset),
        authentication_envelope_length: U64::new(auth_len),
        reserved_offset: U64::new(0),
        reserved_capacity: U64::new(0),
        tier_start: U64::new(tier_start),
        original_tier_ends: [
            U64::new(ends[0]),
            U64::new(ends[1]),
            U64::new(ends[2]),
            U64::new(ends[3]),
        ],
        reserved_tail: [0u8; 24],
    };
    let _ = directory.write_to_prefix(&mut out[V3_PREAMBLE_SIZE..]);

    // 3. Regions
    let h_off = header_offset as usize;
    out[h_off..h_off + header_bytes.len()].copy_from_slice(header_bytes);
    let s_off = shared_offset as usize;
    out[s_off..s_off + shared_bytes.len()].copy_from_slice(shared_bytes);
    let a_off = auth_offset as usize;
    out[a_off..a_off + auth_bytes.len()].copy_from_slice(auth_bytes);

    let mut t_off = tier_start as usize;
    for slice in tier_slices {
        out[t_off..t_off + slice.len()].copy_from_slice(slice);
        t_off += slice.len();
    }

    let default_dmu1;
    let m_bytes = match mutable_frame_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            default_dmu1 = build_dmu1_mutable_frame(b"{}", 48)?;
            &default_dmu1
        }
    };
    append_dmue_mutable_suffix(&mut out, m_bytes);
    Ok(out)
}

/// Builds a canonical `v3.0` (`"sona"`) Open `.dnam` container (`magic = b"DNAM"`, `version = (3, 0)`,
/// `ref[container.dnam.open-layout]`).
pub fn build_v3_open_dnam_container(
    header_bytes: &[u8],
    shared_bytes: &[u8],
    auth_bytes: &[u8],
    tier_slices: &[&[u8]],
    mutable_frame_bytes: Option<&[u8]>,
    auth_kind: u8,
) -> Result<Vec<u8>, DnamError> {
    build_v3_dnam_container_with_profile(
        0,
        header_bytes,
        shared_bytes,
        auth_bytes,
        tier_slices,
        mutable_frame_bytes,
        auth_kind,
    )
}

/// Appends `mutable_frame` and the 16-byte `DMUE` footer to `container_prefix` (`ref[container.dnam.mutable-frame]`).
pub fn append_dmue_mutable_suffix(container_prefix: &mut Vec<u8>, mutable_frame: &[u8]) {
    let footer = DmueFooterWire {
        magic: *b"DMUE",
        frame_len: U64::new(mutable_frame.len() as u64),
        reserved: U32::new(0),
    };
    container_prefix.extend_from_slice(mutable_frame);
    container_prefix.extend_from_slice(footer.as_bytes());
}

fn validate_v3_prefix_and_dir_rt(
    data: &[u8],
    max_profile: u8,
) -> Result<(V3ContainerLayout, usize, usize), ContainerWireError> {
    if max_profile > 2 {
        return Err(ContainerWireError::SchemaStatic("invalid max_profile"));
    }
    if data.len() < V3_HEADER_OFFSET {
        return Err(ContainerWireError::SchemaStatic("short structure"));
    }
    let (preamble, _) = DnamPreambleWire::read_from_prefix(data)
        .map_err(|_| ContainerWireError::SchemaStatic("short structure"))?;
    if preamble.magic != DNAM_MAGIC {
        return Err(ContainerWireError::InvalidMagic(preamble.magic));
    }
    let major = preamble.version_major.get();
    let minor = preamble.version_minor.get();
    if major != CONTAINER_VERSION_MAJOR || minor != CONTAINER_VERSION_MINOR {
        return Err(ContainerWireError::UnsupportedVersion { major, minor });
    }
    let profile = preamble.protection_profile;
    let auth_kind = preamble.auth_kind;
    let maximum = preamble.maximum_tier;
    let reserved = preamble.reserved_flag;
    let rate = preamble.sample_rate_train_hz.get();
    if profile > max_profile
        || auth_kind > 1
        || maximum > 3
        || reserved != 0
        || rate != 48_000
        || preamble.reserved_pad.iter().any(|&b| b != 0)
    {
        return Err(ContainerWireError::SchemaStatic("invalid preamble"));
    }

    let dir = DnamDirectoryWire::ref_from_bytes(&data[V3_PREAMBLE_SIZE..V3_HEADER_OFFSET])
        .map_err(|_| ContainerWireError::SchemaStatic("invalid directory"))?;
    let header_offset = dir.immutable_header_offset.get();
    let header_length = dir.immutable_header_length.get();
    let shared_offset = dir.shared_binary_offset.get();
    let shared_length = dir.shared_binary_length.get();
    let auth_offset = dir.authentication_envelope_offset.get();
    let auth_length = dir.authentication_envelope_length.get();
    let reserved_offset = dir.reserved_offset.get();
    let reserved_capacity = dir.reserved_capacity.get();
    let tier_start = dir.tier_start.get();
    let ends_u64 = dir.original_tier_ends.map(U64::get);

    if dir.reserved_tail.iter().any(|&b| b != 0)
        || reserved_offset != 0
        || reserved_capacity != 0
        || header_offset != (V3_HEADER_OFFSET as u64)
        || header_length == 0
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
    if shared_end > (u32::MAX as u64)
        || Some(auth_offset) != align64_u64(shared_end)
        || auth_length == 0
    {
        return Err(ContainerWireError::SchemaStatic("invalid directory"));
    }
    let auth_end = auth_offset
        .checked_add(auth_length)
        .ok_or(ContainerWireError::SchemaStatic("auth extent overflow"))?;
    if auth_end > (u32::MAX as u64)
        || Some(tier_start) != align64_u64(auth_end)
        || tier_start > (u32::MAX as u64)
    {
        return Err(ContainerWireError::SchemaStatic("invalid tier start"));
    }

    let mut previous = tier_start;
    for (index, &end) in ends_u64.iter().enumerate() {
        if index <= (maximum as usize) {
            if end <= previous || !end.is_multiple_of(64) || end > (u32::MAX as u64) {
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

    for (start, end) in [
        (header_end as usize, shared_offset as usize),
        (shared_end as usize, auth_offset as usize),
        (auth_end as usize, tier_start as usize),
    ] {
        if data[start..end].iter().any(|&b| b != 0) {
            return Err(ContainerWireError::SchemaStatic("nonzero padding"));
        }
    }

    Ok((
        V3ContainerLayout {
            magic: preamble.magic,
            protection_profile: profile,
            auth_kind,
            maximum_tier: maximum,
            retained_tier,
            sample_rate_train_hz: rate,
            header_extent: (header_offset as usize, header_length as usize),
            shared_extent: (shared_offset as usize, shared_length as usize),
            auth_extent: (auth_offset as usize, auth_length as usize),
            tier_start: tier_start as usize,
            tier_ends: [
                ends_u64[0] as usize,
                ends_u64[1] as usize,
                ends_u64[2] as usize,
                ends_u64[3] as usize,
            ],
            mutable_extent: (mutable_offset, mutable_capacity as usize),
        },
        mutable_offset,
        data.len() - DMUE_FOOTER_SIZE,
    ))
}

/// Zero-drop-glue (`Copy` error) `v3.0` Open `.dnam` physical layout validator for real-time paths.
pub(crate) fn validate_v3_dnam_layout_no_alloc_rt(
    data: &[u8],
) -> Result<V3ContainerLayout, ContainerWireError> {
    let (layout, mut_start, mut_end) = validate_v3_prefix_and_dir_rt(data, 0)?;
    validate_dmu1_frame_rt(&data[mut_start..mut_end])?;
    Ok(layout)
}

/// Validates a `v3.0` (`"sona"`) `.dnam` container's physical layout up to `max_profile` (`0..=2`) without heap allocation.
pub fn validate_v3_dnam_layout_with_max_profile_no_alloc(
    data: &[u8],
    max_profile: u8,
) -> Result<V3ContainerLayout, DnamError> {
    let (layout, mut_start, mut_end) = validate_v3_prefix_and_dir_rt(data, max_profile)
        .map_err(ContainerWireError::into_dnam_error)?;
    let mutable_frame = &data[mut_start..mut_end];
    if layout.protection_profile == 0 {
        validate_dmu1_frame_no_alloc(mutable_frame)?;
    } else if mutable_frame.starts_with(b"DME1") {
        validate_dme1_frame_no_alloc(mutable_frame, layout.protection_profile)?;
    } else {
        validate_dmu1_frame_no_alloc(mutable_frame)?;
    }
    Ok(layout)
}

/// Validates a `v3.0` Open `.dnam` container's physical layout without heap allocation (`ref[container.dnam.open-layout]`,
/// `ref[container.dnam.preamble-directory]`, `ref[container.dnam.mutable-frame]`).
pub fn validate_v3_dnam_layout_no_alloc(data: &[u8]) -> Result<V3ContainerLayout, DnamError> {
    validate_v3_dnam_layout_no_alloc_rt(data).map_err(ContainerWireError::into_dnam_error)
}

/// Validates a `v3.0` `.dnam` container's physical layout and strict RFC 8259 `DMU1` mutable JSON object up to `max_profile`.
pub fn validate_v3_dnam_layout_with_max_profile(
    data: &[u8],
    max_profile: u8,
) -> Result<V3ContainerLayout, DnamError> {
    let layout = validate_v3_dnam_layout_with_max_profile_no_alloc(data, max_profile)?;
    parse_and_validate_dmu1_frame(&data[layout.mutable_extent.0..data.len() - DMUE_FOOTER_SIZE])?;
    Ok(layout)
}

/// Validates a `v3.0` Open `.dnam` container's physical layout (`ref[container.dnam.open-layout]`,
/// `ref[container.dnam.preamble-directory]`, `ref[container.dnam.mutable-frame]`).
pub fn validate_v3_open_dnam_layout(data: &[u8]) -> Result<V3ContainerLayout, DnamError> {
    validate_v3_dnam_layout_with_max_profile(data, 0)
}

/// Strips a validated `v3.0` Open `.dnam` container to `target_tier <= retained_tier` without re-signing
/// (`ref[container.dnam.open-layout]`, `ref[container.dnam.tier-prefix]`).
pub fn strip_v3_open_dnam_container(data: &[u8], target_tier: u8) -> Result<Vec<u8>, DnamError> {
    let layout = validate_v3_open_dnam_layout(data)?;
    reject_if(target_tier > layout.retained_tier, "invalid strip tier")?;
    let tier_end = layout.tier_ends[target_tier as usize];
    let mutable_offset = layout.mutable_extent.0;
    let mut out = Vec::with_capacity(tier_end + (data.len() - mutable_offset));
    out.extend_from_slice(&data[..tier_end]);
    out.extend_from_slice(&data[mutable_offset..]);
    Ok(out)
}

/// Computes the cumulative SHA-256 prefix digest of the retained tier in a `v3.0` Open `.dnam` container
/// under `domain` (e.g. `b"DNAM-MASTER-PREFIX-V1"`, `ref[container.dnam.open-layout]`, `ref[sec.sig.prefix-delegation]`).
pub fn compute_v3_open_retained_prefix_sha256(
    data: &[u8],
    domain: &[u8],
) -> Result<[u8; 32], DnamError> {
    let layout = validate_v3_open_dnam_layout(data)?;
    let (h_off, h_len) = layout.header_extent;
    let (s_off, s_len) = layout.shared_extent;
    let header_sha = sha256_digest(&data[h_off..h_off + h_len]);

    let mut shared = Vec::new();
    shared.extend_from_slice(&encode_domain_field("domain", b"DNAM-PLAINTEXT-SHARED-V1"));
    shared.extend_from_slice(&encode_domain_field("preamble", &data[..V3_PREAMBLE_SIZE]));
    shared.extend_from_slice(&encode_domain_field(
        "directory",
        &data[V3_PREAMBLE_SIZE..V3_HEADER_OFFSET],
    ));
    shared.extend_from_slice(&encode_domain_field(
        "shared_model_binary",
        &data[s_off..s_off + s_len],
    ));

    let mut transcript = Vec::new();
    transcript.extend_from_slice(&encode_domain_field("domain", domain));
    transcript.extend_from_slice(&encode_domain_field("immutable_header_sha256", &header_sha));
    transcript.extend_from_slice(&encode_domain_field("shared_model_data", &shared));

    let labels = ["tier_lite", "tier_full", "tier_pro", "tier_ultra"];
    let mut start = layout.tier_start;
    for t in 0..=(layout.retained_tier as usize) {
        let end = layout.tier_ends[t];
        transcript.extend_from_slice(&encode_domain_field(labels[t], &data[start..end]));
        start = end;
    }
    Ok(sha256_digest(&transcript))
}

#[cfg(test)]
mod spec_tests {
    use super::*;
    use crate::container::{DNAM_MAGIC, DNSL_MAGIC};
    use crate::math::crc32c;

    /// verify[container.dnam.formats]
    /// verify[container.dnam.wire-framing]
    /// verify[container.dnam.preamble-directory]
    /// verify[container.dnam.open-layout]
    /// verify[container.dnam.tier-prefix]
    /// verify[container.dnam.sample-rate-anchor]
    /// Verifies `.dnam` `v3.0` (`"sona"`) 64B preamble, 128B directory, 64B-aligned regions, tier
    /// stripping across `Lite..Ultra`, and prefix commitment invariance under mutable suffix edits.
    #[test]
    fn test_v3_open_dnam_framing_preamble_directory_and_tier_stripping() {
        assert_eq!(&DNAM_MAGIC, b"DNAM");
        assert_eq!(&DNSL_MAGIC, b"DNSL");

        let header = br#"{"version":1}"#;
        let shared = [b'S'; 64];
        let auth = [b'A'; 32];
        let tiers: [&[u8]; 4] = [&[0x41; 64], &[0x42; 64], &[0x43; 64], &[0x44; 64]];
        let dmu1 = build_dmu1_mutable_frame(br#"{"user_metadata":{"tag":"v1"}}"#, 64).unwrap();

        let container =
            build_v3_open_dnam_container(header, &shared, &auth, &tiers, Some(&dmu1), 0).unwrap();
        let layout = validate_v3_open_dnam_layout(&container).unwrap();
        assert_eq!(layout.magic, *b"DNAM");
        assert_eq!(layout.protection_profile, 0);
        assert_eq!(layout.auth_kind, 0);
        assert_eq!(layout.maximum_tier, 3);
        assert_eq!(layout.retained_tier, 3);
        assert_eq!(layout.sample_rate_train_hz, 48_000);
        assert_eq!(layout.header_extent, (192, header.len()));
        assert_eq!(layout.shared_extent.0 % 64, 0);
        assert_eq!(layout.auth_extent.0 % 64, 0);
        assert_eq!(layout.tier_start % 64, 0);

        // Stripping to each tier 0..=3 preserves the original preamble, directory, and tier prefix digest
        for target_tier in 0..=3u8 {
            let stripped = strip_v3_open_dnam_container(&container, target_tier).unwrap();
            let s_layout = validate_v3_open_dnam_layout(&stripped).unwrap();
            assert_eq!(s_layout.maximum_tier, 3);
            assert_eq!(s_layout.retained_tier, target_tier);
            assert_eq!(s_layout.tier_ends, layout.tier_ends);

            // Replacing mutable suffix does not alter the retained prefix digest
            let mut with_new_mutable =
                stripped[..s_layout.tier_ends[target_tier as usize]].to_vec();
            let dmu2 = build_dmu1_mutable_frame(br#"{"user_metadata":{"tag":"v3"}}"#, 64).unwrap();
            append_dmue_mutable_suffix(&mut with_new_mutable, &dmu2);
            let d1 = compute_v3_open_retained_prefix_sha256(&stripped, b"DNAM-MASTER-PREFIX-V1")
                .unwrap();
            let d2 =
                compute_v3_open_retained_prefix_sha256(&with_new_mutable, b"DNAM-MASTER-PREFIX-V1")
                    .unwrap();
            assert_eq!(d1, d2);
        }

        // Legacy v2 version fails fast with UnsupportedVersion
        let mut legacy_v2 = container.clone();
        legacy_v2[4..6].copy_from_slice(U16::new(2).as_bytes());
        assert!(matches!(
            validate_v3_open_dnam_layout(&legacy_v2),
            Err(DnamError::UnsupportedVersion { major: 2, minor: 0 })
        ));

        // Tampering with preamble padding, sample rate, or alignment padding is rejected
        let mut bad_rate = container.clone();
        bad_rate[12..16].copy_from_slice(U32::new(44_100).as_bytes());
        assert!(validate_v3_open_dnam_layout(&bad_rate).is_err());

        let mut bad_pad = container.clone();
        bad_pad[192 + header.len()] = 1;
        assert!(validate_v3_open_dnam_layout(&bad_pad).is_err());
    }

    /// verify[container.dnam.mutable-frame]
    /// verify[container.dnam.mutable-json]
    /// Verifies `DMU1` wire framing, Castagnoli CRC-32C vectors (`0xE3069283`, `0xA1CCB48F`),
    /// `DMUE` footer validation, and mutable JSON member schema validation.
    #[test]
    fn test_dmu1_mutable_frame_and_mutable_json_schema() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        let empty_frame = build_dmu1_mutable_frame(b"{}", 18).unwrap();
        assert_eq!(empty_frame, b"DMU1\x8f\xb4\xcc\xa1\x02\0\0\0\0\0\0\0{}");
        let doc = parse_and_validate_dmu1_frame(&empty_frame).unwrap();
        validate_mutable_json_schema(&doc).unwrap();

        let rich_payload = br#"{"version":1,"presets":[{"id":"warm","name":"Warm","controls":{"gain":0.4,"mode":"lead"},"user_metadata":{"note":"ok"}}],"user_metadata":{"tags":["amp"]},"control_user_metadata":{"gain":{"color":"red"}},"stem_user_metadata":{"stem-1":{"tag":"keep"}}}"#;
        let rich_frame =
            build_dmu1_mutable_frame(rich_payload, 16 + rich_payload.len() + 16).unwrap();
        let rich_doc = parse_and_validate_dmu1_frame(&rich_frame).unwrap();
        validate_mutable_json_schema(&rich_doc).unwrap();

        // Non-zero padding or corrupted CRC is rejected
        let mut bad_pad = rich_frame.clone();
        *bad_pad.last_mut().unwrap() = 1;
        assert!(parse_and_validate_dmu1_frame(&bad_pad).is_err());

        let mut bad_crc = empty_frame.clone();
        bad_crc[4] ^= 0xff;
        assert!(parse_and_validate_dmu1_frame(&bad_crc).is_err());

        // Invalid mutable member shapes are rejected
        for bad_json in [
            br#"{"version":2}"#.as_slice(),
            br#"{"unknown":{}}"#.as_slice(),
            br#"{"presets":{}}"#.as_slice(),
            br#"{"control_user_metadata":{"gain":2}}"#.as_slice(),
            br#"{"presets":[{"id":"p","name":"P","controls":{},"unknown":0}]}"#.as_slice(),
            br#"{"presets":[{"id":"p","name":"P","controls":{}},{"id":"p","name":"Q","controls":{}}]}"#
                .as_slice(),
            br#"{"presets":[{"id":"p","name":"P","controls":{"gain":true}}]}"#.as_slice(),
            br#"{"presets":[{"id":"p","name":"P"}]}"#.as_slice(),
        ] {
            let f = build_dmu1_mutable_frame(bad_json, 16 + bad_json.len()).unwrap();
            let v = parse_and_validate_dmu1_frame(&f).unwrap();
            assert!(validate_mutable_json_schema(&v).is_err());
        }
    }
}

#[cfg(test)]
mod mutable_reference_tests {
    // Byte-level mutable-frame examples, not a production container parser.

    use crate::math::crc32c;
    use zerocopy::{FromBytes, IntoBytes};

    #[test]
    fn open_mutable_empty_object_has_a_fixed_crc32c_and_wire_frame() {
        use zerocopy::byteorder::little_endian::{U32, U64};
        let payload = b"{}";
        let length = U64::new(u64::try_from(payload.len()).unwrap());
        let checksum_input = [b"DMU1".as_slice(), length.as_bytes(), payload].concat();
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(crc32c(&checksum_input), 0xa1cc_b48f);

        let frame = [
            b"DMU1".as_slice(),
            U32::new(crc32c(&checksum_input)).as_bytes(),
            length.as_bytes(),
            payload,
        ]
        .concat();
        assert_eq!(frame, b"DMU1\x8f\xb4\xcc\xa1\x02\0\0\0\0\0\0\0{}");
    }

    #[test]
    fn recomputing_mutable_crc_does_not_authenticate_the_content() {
        use zerocopy::byteorder::little_endian::U32;
        let first = b"DMU1\x07\0\0\0\0\0\0\0{\"x\":0}";
        let mut frame = [
            b"DMU1".as_slice(),
            U32::new(crc32c(first)).as_bytes(),
            &first[4..],
        ]
        .concat();
        frame[21] = b'1';
        let altered_checksum = crc32c(&[&frame[..4], &frame[8..16], &frame[16..]].concat());
        let stored_checksum = U32::ref_from_bytes(&frame[4..8]).unwrap().get();
        assert_ne!(stored_checksum, altered_checksum);
        frame[4..8].copy_from_slice(U32::new(altered_checksum).as_bytes());
        assert_eq!(
            U32::ref_from_bytes(&frame[4..8]).unwrap().get(),
            crc32c(&[&frame[..4], &frame[8..16], &frame[16..]].concat())
        );
        assert_eq!(&frame[16..], br#"{"x":1}"#);
    }
}
