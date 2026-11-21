//! Canonical `v3.0` (`"sona"`) preamble & 128-byte region directory view (`DnamPreamble`).

use core::mem::{align_of, size_of};

use dnam_vm::bin::{ControlClampDescriptorWire, DnamBinHeaderWire};
use dnam_vm::control::ControlClampDescriptor;
use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, IntoBytes};

use crate::schema::DnamError;
use crate::types::Tier;

use super::wire::{
    CP1_MAGIC, ContainerWireError, DMU1_HEADER_SIZE, DMUE_FOOTER_SIZE, DnamPreambleWire,
    V3_HEADER_OFFSET, V3_PREAMBLE_SIZE, parse_dcrc_record_rt, validate_v3_directory_extents_rt,
};
use super::{
    CONTAINER_VERSION_MAJOR, CONTAINER_VERSION_MINOR, DNAM_MAGIC, FLAG_ENCRYPTED_ARENA,
    FLAG_HAS_FULL_DELTA, FLAG_HAS_LITE, FLAG_HAS_PRO_DELTA, FLAG_HAS_SIGNATURES,
    FLAG_HAS_ULTRA_DELTA, FLAG_STEREO_OUT,
};

/// Minimum byte length of a valid `v3.0` `.dnam` container (`192 + 18 + 16 = 226 B`).
const MIN_V3_CONTAINER_BYTES: usize = V3_HEADER_OFFSET + DMU1_HEADER_SIZE + 2 + DMUE_FOOTER_SIZE;

/// Logical preamble descriptor (`#[repr(C, align(64))]`) summarizing a validated
/// `v3.0` (`"sona"`) `.dnam` container without heap allocation (`2` cache lines = `128 B`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnamPreamble {
    /// `b"DNAM"` (`0x4D414E44`).
    pub magic: [u8; 4],
    /// Container major version (`3`).
    pub version_major: u16,
    /// Container minor version (`0`).
    pub version_minor: u16,
    /// Bitmask of `FLAG_*` constants.
    pub flags: u16,
    /// Total user controls $P \in [0, 65535]$.
    pub num_controls: u16,
    /// Reference training sample rate in Hz (`48000`).
    pub sample_rate_train_hz: u32,
    /// Unpadded UTF-8 byte length of Region 3 JCS JSON header starting at `0x00C0` (`192`).
    pub json_header_byte_len: u32,
    /// File byte offset of `tier_start` (`% 64 == 0`).
    pub tensor_arena_offset: u32,
    /// Total retained byte length of tier slices (`% 64 == 0`).
    pub tensor_arena_byte_len: u32,
    /// Arena-relative end of `Lite` (`tier 0`) in 64B cache lines (`72` $\to$ `4,608 B`).
    pub arena_lite_end_cl: u16,
    /// Arena-relative end of `Full` (`tier 0..=1`) in 64B cache lines (`124` $\to$ `7,936 B`).
    pub arena_full_end_cl: u16,
    /// Arena-relative end of `Pro` (`tier 0..=2`) in 64B cache lines (`248` $\to$ `15,872 B`).
    pub arena_pro_end_cl: u16,
    /// Arena-relative end of `Ultra` (`tier 0..=3`) in 64B cache lines (`480` $\to$ `30,720 B`).
    pub arena_ultra_end_cl: u16,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_lite_end]`.
    pub arena_crc32c_lite: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_full_end]`.
    pub arena_crc32c_full: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_pro_end]`.
    pub arena_crc32c_pro: u32,
    /// Castagnoli `CRC32C` over `arena[0..arena_offset_ultra_end]`.
    pub arena_crc32c_ultra: u32,
    /// Reserved 16-bit alignment slot (`0`).
    pub _reserved_stems: u16,
    /// Reserved 16-bit alignment gap (`0`).
    pub _reserved: u16,
    /// Estimated $T_{60}$ decay tail time in milliseconds for DAW sleep management.
    pub tail_time_ms: f32,
    /// Castagnoli `CRC32C` over `bytes[0..auth_offset]` when a `DCRC` record is present.
    pub header_crc32c: u32,
    /// Castagnoli `CRC32C` over `[dbin_offset..tier_start)` when a `DCRC` record and `.dnam-bin` prefix are present.
    pub crc_dbin_prefix: u32,
    /// Optional `64 B`-aligned byte offset of the embedded Layer 0 `.dnam-bin` prefix (`0` when absent).
    pub dbin_offset: u64,
    /// Byte length of the embedded Layer 0 `.dnam-bin` prefix (`% 64 == 0`, `0` when absent).
    pub dbin_prefix_bytes: u64,
}

const _: () = assert!(size_of::<DnamPreamble>() == 128);
const _: () = assert!(align_of::<DnamPreamble>() == 64);

fn scan_shared_control_count(shared_slice: &[u8]) -> u16 {
    if shared_slice.len() >= 4 && shared_slice[0..4] == CP1_MAGIC {
        return 0;
    }
    let mut count = 0usize;
    let mut off = 0usize;
    while off + 64 <= shared_slice.len() {
        if shared_slice[off..off + 4] == CP1_MAGIC {
            break;
        }
        let Ok(wire) = ControlClampDescriptorWire::ref_from_bytes(&shared_slice[off..off + 64])
        else {
            break;
        };
        let desc = ControlClampDescriptor::from_wire(wire);
        let populated = desc.ctrl_kind <= 1
            && (desc.safe_max_base > desc.safe_min_base
                || desc.scale_norm != 0.0
                || desc.active_options_mask != 0
                || desc.lobatto_degree > 0);
        if !populated {
            break;
        }
        count += 1;
        off += 64;
    }
    count.min(u16::MAX as usize) as u16
}

impl DnamPreamble {
    /// Zero-drop-glue (`Copy` error) [`DnamPreamble`] parser for real-time container reloaders.
    pub(crate) fn from_bytes_rt(bytes: &[u8]) -> Result<Self, ContainerWireError> {
        let Ok((pw, _)) = DnamPreambleWire::read_from_prefix(bytes) else {
            return Err(ContainerWireError::BufferTooSmall {
                expected: V3_PREAMBLE_SIZE,
                actual: bytes.len(),
            });
        };
        let magic = pw.magic;
        if magic != DNAM_MAGIC {
            return Err(ContainerWireError::InvalidMagic(magic));
        }
        let version_major = pw.version_major.get();
        let version_minor = pw.version_minor.get();
        if version_major != CONTAINER_VERSION_MAJOR || version_minor != CONTAINER_VERSION_MINOR {
            return Err(ContainerWireError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }
        if bytes.len() < MIN_V3_CONTAINER_BYTES {
            return Err(ContainerWireError::BufferTooSmall {
                expected: MIN_V3_CONTAINER_BYTES,
                actual: bytes.len(),
            });
        }

        let profile = pw.protection_profile;
        let auth_kind = pw.auth_kind;
        let maximum = pw.maximum_tier;
        let reserved = pw.reserved_flag;
        let sample_rate_train_hz = pw.sample_rate_train_hz.get();
        let max_profile = if magic == DNAM_MAGIC { 2 } else { 0 };
        let max_auth_kind = if magic == DNAM_MAGIC { 1 } else { 0 };
        if profile > max_profile
            || auth_kind > max_auth_kind
            || maximum > 3
            || reserved != 0
            || sample_rate_train_hz != 48_000
            || pw.reserved_pad.iter().any(|&b| b != 0)
        {
            return Err(ContainerWireError::SchemaStatic("invalid v3 preamble"));
        }

        let dir = validate_v3_directory_extents_rt(bytes, maximum)?;
        let retained_tier = dir.retained_tier;
        let tier_start = dir.tier_start;
        let retained_end = dir.tier_ends[retained_tier as usize];
        let tensor_arena_offset = tier_start as u32;
        let tensor_arena_byte_len = (retained_end - tier_start) as u32;

        let rel0 = (dir.tier_ends[0] - tier_start) as u32;
        let rel1 = if maximum >= 1 {
            (dir.tier_ends[1] - tier_start) as u32
        } else {
            rel0
        };
        let rel2 = if maximum >= 2 {
            (dir.tier_ends[2] - tier_start) as u32
        } else {
            rel1
        };
        let rel3 = if maximum >= 3 {
            (dir.tier_ends[3] - tier_start) as u32
        } else {
            rel2
        };

        let to_cl = |rel_bytes: u32| -> u16 { (rel_bytes / 64).min(u16::MAX as u32) as u16 };

        let mut flags = FLAG_HAS_LITE | FLAG_STEREO_OUT;
        if retained_tier >= 1 {
            flags |= FLAG_HAS_FULL_DELTA;
        }
        if retained_tier >= 2 {
            flags |= FLAG_HAS_PRO_DELTA;
        }
        if retained_tier >= 3 {
            flags |= FLAG_HAS_ULTRA_DELTA;
        }
        if profile > 0 {
            flags |= FLAG_ENCRYPTED_ARENA;
        }
        if auth_kind > 0 {
            flags |= FLAG_HAS_SIGNATURES;
        }

        let shared_slice = &bytes[dir.shared_extent.0..dir.shared_extent.0 + dir.shared_extent.1];
        let auth_slice = &bytes[dir.auth_extent.0..dir.auth_extent.0 + dir.auth_extent.1];
        let dbin_slice = if dir.dbin_extent.1 > 0 {
            &bytes[dir.dbin_extent.0..dir.dbin_extent.0 + dir.dbin_extent.1]
        } else {
            &[]
        };

        let fallback_num_controls = || -> u16 {
            if !dbin_slice.is_empty() {
                DnamBinHeaderWire::read_from_prefix(dbin_slice)
                    .map_or(0, |(h, _)| h.num_controls.get())
            } else {
                scan_shared_control_count(shared_slice)
            }
        };

        let num_controls: u16;
        let mut tail_time_ms = 0.0f32;
        let mut header_crc32c = 0u32;
        let mut crc_dbin_prefix = 0u32;
        let mut crc_tiers = [0u32; 4];

        if auth_slice.starts_with(b"DAU1") {
            flags |= FLAG_HAS_SIGNATURES;
            num_controls = fallback_num_controls();
        } else if let Some(dcrc) = parse_dcrc_record_rt(auth_slice)? {
            if dir.dbin_extent.1 == 0 && dcrc.crc_dbin_prefix != 0 {
                return Err(ContainerWireError::SchemaStatic(
                    "non-zero crc_dbin_prefix when dbin_prefix_bytes == 0",
                ));
            }
            let hint_mask = FLAG_STEREO_OUT | FLAG_ENCRYPTED_ARENA | FLAG_HAS_SIGNATURES;
            flags = (flags & !FLAG_STEREO_OUT) | (dcrc.flags_hint & hint_mask);
            num_controls = dcrc.num_controls;
            tail_time_ms = f32::from_bits(dcrc.tail_time_ms_bits);
            header_crc32c = dcrc.crc_prefix;
            crc_dbin_prefix = dcrc.crc_dbin_prefix;
            crc_tiers = dcrc.crc_tiers;
        } else {
            num_controls = fallback_num_controls();
        }

        Ok(Self {
            magic,
            version_major,
            version_minor,
            flags,
            num_controls,
            sample_rate_train_hz,
            json_header_byte_len: dir.header_extent.1 as u32,
            tensor_arena_offset,
            tensor_arena_byte_len,
            arena_lite_end_cl: to_cl(rel0),
            arena_full_end_cl: to_cl(rel1),
            arena_pro_end_cl: to_cl(rel2),
            arena_ultra_end_cl: to_cl(rel3),
            arena_crc32c_lite: crc_tiers[0],
            arena_crc32c_full: crc_tiers[1],
            arena_crc32c_pro: crc_tiers[2],
            arena_crc32c_ultra: crc_tiers[3],
            _reserved_stems: 0,
            _reserved: 0,
            tail_time_ms,
            header_crc32c,
            crc_dbin_prefix,
            dbin_offset: dir.dbin_extent.0 as u64,
            dbin_prefix_bytes: dir.dbin_extent.1 as u64,
        })
    }

    /// Parses a [`DnamPreamble`] from a canonical `v3.0` (`"sona"`) `.dnam` container buffer
    /// without heap allocation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DnamError> {
        Self::from_bytes_rt(bytes).map_err(ContainerWireError::into_dnam_error)
    }

    /// Serializes the canonical 64-byte `v3.0` wire preamble via [`DnamPreambleWire`].
    pub fn to_bytes(&self) -> [u8; 64] {
        let wire = DnamPreambleWire {
            magic: self.magic,
            version_major: U16::new(self.version_major),
            version_minor: U16::new(self.version_minor),
            protection_profile: u8::from(self.is_encrypted_arena()),
            auth_kind: 0,
            maximum_tier: self.highest_tier() as u8,
            reserved_flag: 0,
            sample_rate_train_hz: U32::new(self.sample_rate_train_hz),
            reserved_pad: [0u8; 48],
        };
        let mut out = [0u8; 64];
        let _ = wire.write_to_prefix(&mut out);
        out
    }

    /// Arena-relative byte end of `Arena 0 (Lite)` (`arena_lite_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_lite_end(&self) -> u32 {
        (self.arena_lite_end_cl as u32) * 64
    }

    /// Arena-relative byte end of `Arena 0..1 (Full)` (`arena_full_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_full_end(&self) -> u32 {
        (self.arena_full_end_cl as u32) * 64
    }

    /// Arena-relative byte end of `Arena 0..2 (Pro)` (`arena_pro_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_pro_end(&self) -> u32 {
        (self.arena_pro_end_cl as u32) * 64
    }

    /// Arena-relative byte end of `Arena 0..3 (Ultra)` (`arena_ultra_end_cl * 64`).
    #[inline]
    pub const fn arena_offset_ultra_end(&self) -> u32 {
        (self.arena_ultra_end_cl as u32) * 64
    }

    /// Returns the highest tier present in the container based on `flags` (`FLAG_HAS_*_DELTA`).
    #[inline]
    pub const fn highest_tier(&self) -> Tier {
        if (self.flags & FLAG_HAS_ULTRA_DELTA) != 0 {
            Tier::Ultra
        } else if (self.flags & FLAG_HAS_PRO_DELTA) != 0 {
            Tier::Pro
        } else if (self.flags & FLAG_HAS_FULL_DELTA) != 0 {
            Tier::Full
        } else {
            Tier::Lite
        }
    }

    /// Returns the cumulative byte end offset for `tier`.
    #[inline]
    pub const fn arena_end_for_tier(&self, tier: Tier) -> usize {
        match tier {
            Tier::Lite => self.arena_offset_lite_end() as usize,
            Tier::Full => self.arena_offset_full_end() as usize,
            Tier::Pro => self.arena_offset_pro_end() as usize,
            Tier::Ultra => self.arena_offset_ultra_end() as usize,
        }
    }

    /// Returns the cumulative Castagnoli `CRC32C` for `tier`.
    #[inline]
    pub const fn crc32c_for_tier(&self, tier: Tier) -> u32 {
        match tier {
            Tier::Lite => self.arena_crc32c_lite,
            Tier::Full => self.arena_crc32c_full,
            Tier::Pro => self.arena_crc32c_pro,
            Tier::Ultra => self.arena_crc32c_ultra,
        }
    }

    /// Returns `true` if `STEREO_OUT` (`bit 9`) is set.
    #[inline]
    pub const fn is_stereo_out(&self) -> bool {
        (self.flags & FLAG_STEREO_OUT) != 0
    }

    /// Returns `true` if `HAS_SIGNATURES` (`bit 10`) is set.
    #[inline]
    pub const fn has_signatures(&self) -> bool {
        (self.flags & FLAG_HAS_SIGNATURES) != 0
    }

    /// Returns `true` if `ENCRYPTED_ARENA` (`bit 11`) is set.
    #[inline]
    pub const fn is_encrypted_arena(&self) -> bool {
        (self.flags & FLAG_ENCRYPTED_ARENA) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, DnamDirectoryWire, build_dnam_container};
    use crate::schema::DnamJsonHeader;
    use alloc::vec;

    #[test]
    fn test_preamble_roundtrip_and_error_branches() {
        assert!(DnamPreamble::from_bytes(&[0u8; 10]).is_err());
        let mut bad_preamble = [0u8; 64];
        bad_preamble[0..4].copy_from_slice(b"NOPE");
        assert!(matches!(
            DnamPreamble::from_bytes(&bad_preamble),
            Err(DnamError::InvalidMagic(_))
        ));
        bad_preamble[0..4].copy_from_slice(b"DNAM");
        bad_preamble[4..6].copy_from_slice(zerocopy::IntoBytes::as_bytes(
            &zerocopy::byteorder::little_endian::U16::new(2),
        ));
        assert!(matches!(
            DnamPreamble::from_bytes(&bad_preamble),
            Err(DnamError::UnsupportedVersion { major: 2, minor: 0 })
        ));
        bad_preamble[4..6].copy_from_slice(zerocopy::IntoBytes::as_bytes(
            &zerocopy::byteorder::little_endian::U16::new(CONTAINER_VERSION_MAJOR),
        ));
        assert!(matches!(
            DnamPreamble::from_bytes(&bad_preamble),
            Err(DnamError::BufferTooSmall { .. })
        ));

        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();
        let mut p = DnamPreamble::from_bytes(&container).unwrap();
        assert_eq!(size_of::<DnamPreamble>(), 128);
        assert_eq!(align_of::<DnamPreamble>(), 64);
        assert_eq!(p.version_major, 3);
        assert_eq!(p.version_minor, 0);
        assert_eq!(p.highest_tier(), Tier::Ultra);
        assert_eq!(p.dbin_offset, 0);
        assert_eq!(p.dbin_prefix_bytes, 0);
        assert_eq!(p.crc_dbin_prefix, 0);
        assert_eq!(p.arena_lite_end_cl, 72);
        assert_eq!(p.arena_full_end_cl, 124);
        assert_eq!(p.arena_pro_end_cl, 248);
        assert_eq!(p.arena_ultra_end_cl, 480);
        assert_eq!(p.to_bytes()[0..4], DNAM_MAGIC);

        // Non-zero crc_dbin_prefix when dbin_prefix_bytes == 0 must fail closed.
        let mut bad_dcrc = container.clone();
        let auth_off = DnamDirectoryWire::ref_from_bytes(&container[64..192])
            .unwrap()
            .authentication_envelope_offset
            .get() as usize;
        bad_dcrc[auth_off + 32..auth_off + 36].copy_from_slice(U32::new(1).as_bytes());
        assert!(DnamPreamble::from_bytes(&bad_dcrc).is_err());

        p.arena_crc32c_lite = 11;
        p.arena_crc32c_full = 22;
        p.arena_crc32c_pro = 33;
        p.arena_crc32c_ultra = 44;
        assert_eq!(p.crc32c_for_tier(Tier::Lite), 11);
        assert_eq!(p.crc32c_for_tier(Tier::Full), 22);
        assert_eq!(p.crc32c_for_tier(Tier::Pro), 33);
        assert_eq!(p.crc32c_for_tier(Tier::Ultra), 44);
        assert_eq!(p.arena_end_for_tier(Tier::Lite), 72 * 64);
        assert_eq!(p.arena_end_for_tier(Tier::Full), 124 * 64);
        assert_eq!(p.arena_end_for_tier(Tier::Pro), 248 * 64);
        assert_eq!(p.arena_end_for_tier(Tier::Ultra), 480 * 64);

        // Scan CP1 shared controls until sentinel or CP1 magic
        let mut shared_buf = vec![0u8; 64];
        shared_buf[0..4].copy_from_slice(&crate::container::CP1_MAGIC);
        assert_eq!(scan_shared_control_count(&shared_buf), 0);
        let mut one_desc = vec![0u8; 128];
        one_desc[12..16].copy_from_slice(zerocopy::IntoBytes::as_bytes(
            &zerocopy::byteorder::little_endian::F32::new(1.0),
        )); // safe_max_base (offset 12..16) > safe_min_base
        assert_eq!(scan_shared_control_count(&one_desc), 1);
        one_desc[64..68].copy_from_slice(&crate::container::CP1_MAGIC);
        assert_eq!(scan_shared_control_count(&one_desc), 1);

        // Directory invariants & unaligned tier_end rejections
        let mut bad_dir = container.clone();
        bad_dir[64 + 8] = 0; // header_length = 0
        assert!(DnamPreamble::from_bytes(&bad_dir).is_err());

        let mut bad_align = container.clone();
        bad_align[64 + 2 * 8] += 1; // shared_offset unaligned
        assert!(DnamPreamble::from_bytes(&bad_align).is_err());

        let mut bad_tier_end = container.clone();
        bad_tier_end[64 + 9 * 8..64 + 10 * 8].copy_from_slice(zerocopy::IntoBytes::as_bytes(
            &zerocopy::byteorder::little_endian::U64::new(u64::MAX - 63),
        ));
        assert!(DnamPreamble::from_bytes(&bad_tier_end).is_err());

        // Unaligned tier_end in directory must fail with UnalignedOffset
        container[64 + 9 * 8] = 65;
        assert!(matches!(
            DnamPreamble::from_bytes(&container),
            Err(DnamError::UnalignedOffset { .. })
        ));
    }
}
