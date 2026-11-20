//! Zero-copy `.dnam-bin` (`"DNBN"`) parser/validator ([`DnamBinView`]) and zero-mutation
//! `REQ-5.6` tier stripper ([`strip_dnam_bin_to_tier`]).

mod validate;

use zerocopy::FromBytes;

use self::validate::{
    ArenaValidationInput, validate_arena0_and_quads, validate_cp1_overflow, validate_descriptors,
    validate_prefix_header, validate_region_layout_and_padding, validate_tsyn_entries,
};
use super::arena_wire::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
};
use super::header::{
    ARENA_TIER_BYTES, CEXT_DISPATCH_BYTES, CextDispatchWire, ControlClampDescriptorWire,
    DBRM_MAGIC, DNBN_MAGIC, DnamBinHeaderWire, DnamBinRuntimeMetaWire, TSYN_ENTRY_BYTES,
    TempoSyncEntryWire,
};
use crate::math::{Crc32cBackend, DefaultMathBackend};
use crate::types::{Tier, VmError};

/// Zero-copy validated view over an in-memory `.dnam-bin` buffer (`ref[container.dnam.vm-binary]`).
#[derive(Debug, Clone, Copy)]
pub struct DnamBinView<'a> {
    /// Fixed `64 B` `"DNBN"` file header (`0..64`).
    pub header: &'a DnamBinHeaderWire,
    /// Fixed `64 B` `"DBRM"` runtime metadata record (`64..128`).
    pub runtime_meta: &'a DnamBinRuntimeMetaWire,
    /// Borrowed slice of `P` control clamp descriptors (`P <= 255`).
    pub control_descriptors: &'a [ControlClampDescriptorWire],
    /// Trailing `"CP1\0"` overflow stream bytes inside `shared_controls` (empty if inline-only).
    pub cp1_overflow: &'a [u8],
    /// Borrowed slice of `tsyn_count` pure-binary `"TSYN"` tempo-sync rows.
    pub tsyn_entries: &'a [TempoSyncEntryWire],
    /// Optional unwatermarked `1,280 B` `"CEXT"` dispatch matrices (`stage_targets[16..32]`).
    pub cext: Option<&'a CextDispatchWire>,
    /// Retained SONA tier determined by `bytes.len() == tier_end_offsets[t]`.
    pub retained_tier: Tier,
    /// Self-contained Arena 0 Lite payload (`4,608 B`).
    pub arena0: &'a Arena0LiteWire,
    /// Optional Arena 1 Full delta (`3,328 B`, present when `retained_tier >= Full`).
    pub arena1: Option<&'a Arena1FullDeltaWire>,
    /// Optional Arena 2 Pro delta (`7,936 B`, present when `retained_tier >= Pro`).
    pub arena2: Option<&'a Arena2ProDeltaWire>,
    /// Optional Arena 3 Ultra delta (`14,848 B`, present when `retained_tier == Ultra`).
    pub arena3: Option<&'a Arena3UltraDeltaWire>,
}

impl<'a> DnamBinView<'a> {
    /// Binds a `.dnam-bin` view in `< 50 ns` after validating fixed header/meta bounds and
    /// tier end offsets, for hot/re-entrant paths whose buffer CRC-32C was already authenticated
    /// (for cold untrusted input, call [`Self::parse`] which verifies prefix & arena CRC-32C).
    #[inline]
    pub fn bind_prevalidated(bytes: &'a [u8]) -> Result<Self, VmError> {
        if bytes.len() < 128 {
            return Err(VmError::TruncatedPrefix);
        }
        let header = bytes
            .get(0..64)
            .and_then(|s| DnamBinHeaderWire::ref_from_bytes(s).ok())
            .ok_or(VmError::TruncatedPrefix)?;
        let meta = bytes
            .get(64..128)
            .and_then(|s| DnamBinRuntimeMetaWire::ref_from_bytes(s).ok())
            .ok_or(VmError::TruncatedPrefix)?;
        if header.magic != DNBN_MAGIC || meta.magic != DBRM_MAGIC {
            return Err(VmError::InvalidMagic);
        }
        let max_tier = header.max_present_tier as usize;
        if max_tier > 3 {
            return Err(VmError::InvalidTier);
        }
        let arena0_off = header.arena0_offset.get() as usize;
        if arena0_off < 128 || (arena0_off & 63) != 0 || arena0_off > bytes.len() {
            return Err(VmError::InvalidRegionLayout);
        }
        let mut retained_idx = None;
        for t in 0..=max_tier {
            if bytes.len() == header.tier_end_offsets[t].get() as usize {
                retained_idx = Some(t);
                break;
            }
        }
        let retained_idx = retained_idx.ok_or(VmError::LengthTierMismatch)?;
        let retained_tier = Tier::from_u8(retained_idx as u8);

        let num_controls = header.num_controls.get() as usize;
        let sc_off = meta.shared_controls_offset.get() as usize;
        let sc_bytes_len = num_controls
            .checked_mul(64)
            .ok_or(VmError::InvalidControlDescriptor)?;
        let sc_end = sc_off
            .checked_add(sc_bytes_len)
            .ok_or(VmError::InvalidControlDescriptor)?;
        let control_descriptors = if num_controls > 0 {
            let desc_slice = bytes
                .get(sc_off..sc_end)
                .ok_or(VmError::InvalidControlDescriptor)?;
            <[ControlClampDescriptorWire]>::ref_from_bytes(desc_slice)
                .map_err(|_| VmError::InvalidControlDescriptor)?
        } else {
            &[]
        };
        let cp1_over_len = meta.cp1_overflow_bytes.get() as usize;
        let cp1_overflow = if cp1_over_len > 0 {
            let cp1_end = sc_end
                .checked_add(cp1_over_len)
                .ok_or(VmError::InvalidCp1Stream)?;
            bytes
                .get(sc_end..cp1_end)
                .ok_or(VmError::InvalidCp1Stream)?
        } else {
            &[]
        };
        let tsyn_count = header.tsyn_count.get() as usize;
        let tsyn_off = meta.tsyn_offset.get() as usize;
        let tsyn_entries = if tsyn_count > 0 {
            let tsyn_bytes_len = tsyn_count
                .checked_mul(TSYN_ENTRY_BYTES)
                .ok_or(VmError::InvalidTempoSyncEntry)?;
            let tsyn_end = tsyn_off
                .checked_add(tsyn_bytes_len)
                .ok_or(VmError::InvalidTempoSyncEntry)?;
            let tsyn_slice = bytes
                .get(tsyn_off..tsyn_end)
                .ok_or(VmError::InvalidTempoSyncEntry)?;
            <[TempoSyncEntryWire]>::ref_from_bytes(tsyn_slice)
                .map_err(|_| VmError::InvalidTempoSyncEntry)?
        } else {
            &[]
        };
        let cext_off = meta.cext_offset.get() as usize;
        let cext = if meta.cext_len.get() == CEXT_DISPATCH_BYTES as u32 {
            let cext_end = cext_off
                .checked_add(CEXT_DISPATCH_BYTES)
                .ok_or(VmError::InvalidRegionLayout)?;
            let cext_slice = bytes
                .get(cext_off..cext_end)
                .ok_or(VmError::InvalidRegionLayout)?;
            Some(
                CextDispatchWire::ref_from_bytes(cext_slice)
                    .map_err(|_| VmError::InvalidRegionLayout)?,
            )
        } else {
            None
        };
        let a0_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[0])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena0 = bytes
            .get(arena0_off..a0_end)
            .and_then(|s| Arena0LiteWire::ref_from_bytes(s).ok())
            .ok_or(VmError::InvalidArenaHeader)?;
        let a1_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[1])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena1 = if retained_idx >= 1 {
            Some(
                bytes
                    .get(a0_end..a1_end)
                    .and_then(|s| Arena1FullDeltaWire::ref_from_bytes(s).ok())
                    .ok_or(VmError::InvalidArenaHeader)?,
            )
        } else {
            None
        };
        let a2_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[2])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena2 = if retained_idx >= 2 {
            Some(
                bytes
                    .get(a1_end..a2_end)
                    .and_then(|s| Arena2ProDeltaWire::ref_from_bytes(s).ok())
                    .ok_or(VmError::InvalidArenaHeader)?,
            )
        } else {
            None
        };
        let a3_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[3])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena3 = if retained_idx >= 3 {
            Some(
                bytes
                    .get(a2_end..a3_end)
                    .and_then(|s| Arena3UltraDeltaWire::ref_from_bytes(s).ok())
                    .ok_or(VmError::InvalidArenaHeader)?,
            )
        } else {
            None
        };
        Ok(Self {
            header,
            runtime_meta: meta,
            control_descriptors,
            cp1_overflow,
            tsyn_entries,
            cext,
            retained_tier,
            arena0,
            arena1,
            arena2,
            arena3,
        })
    }

    /// Parses and validates an immutable `.dnam-bin` byte slice using [`DefaultMathBackend`] CRC-32C.
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self, VmError> {
        Self::parse_with_crc::<DefaultMathBackend>(bytes)
    }

    /// Parses and validates an immutable `.dnam-bin` byte slice using a specified [`Crc32cBackend`].
    pub fn parse_with_crc<C: Crc32cBackend>(bytes: &'a [u8]) -> Result<Self, VmError> {
        if bytes.len() < 128 {
            return Err(VmError::TruncatedPrefix);
        }
        let header = bytes
            .get(0..64)
            .and_then(|s| DnamBinHeaderWire::ref_from_bytes(s).ok())
            .ok_or(VmError::TruncatedPrefix)?;
        let meta = bytes
            .get(64..128)
            .and_then(|s| DnamBinRuntimeMetaWire::ref_from_bytes(s).ok())
            .ok_or(VmError::TruncatedPrefix)?;
        let (max_tier, num_controls, tsyn_count) = validate_prefix_header(header, meta)?;
        let (sc_off, tsyn_off, cext_off, arena0_off) =
            validate_region_layout_and_padding(bytes, header, meta, num_controls, tsyn_count)?;

        let mut retained_tier_idx: Option<usize> = None;
        for t in 0..4 {
            let end_off = header.tier_end_offsets[t].get() as usize;
            let crc_t = header.tier_arena_crc32c[t].get();
            if t <= max_tier {
                let expected_end = arena0_off
                    .checked_add(ARENA_TIER_BYTES[t])
                    .ok_or(VmError::LengthTierMismatch)?;
                if end_off != expected_end {
                    return Err(VmError::LengthTierMismatch);
                }
                if bytes.len() == end_off {
                    retained_tier_idx = Some(t);
                }
            } else if end_off != 0 || crc_t != 0 {
                return Err(VmError::LengthTierMismatch);
            }
        }
        let retained_idx = retained_tier_idx.ok_or(VmError::LengthTierMismatch)?;
        let retained_tier = Tier::from_u8(retained_idx as u8);

        let prefix_hdr_slice = bytes.get(0..20).ok_or(VmError::InvalidRegionLayout)?;
        let prefix_body_slice = bytes
            .get(24..arena0_off)
            .ok_or(VmError::InvalidRegionLayout)?;
        let prefix_crc = C::crc32c_update(0, prefix_hdr_slice);
        let prefix_crc = C::crc32c_update(prefix_crc, prefix_body_slice);
        if prefix_crc != header.prefix_crc32c.get() {
            return Err(VmError::PrefixCrcMismatch);
        }
        let arena_slice = bytes
            .get(arena0_off..)
            .ok_or(VmError::InvalidRegionLayout)?;
        let arena_crc = C::crc32c_update(0, arena_slice);
        if arena_crc != header.tier_arena_crc32c[retained_idx].get() {
            return Err(VmError::TierArenaCrcMismatch);
        }

        let desc_bytes_len = num_controls
            .checked_mul(64)
            .ok_or(VmError::InvalidControlDescriptor)?;
        let sc_desc_end = sc_off
            .checked_add(desc_bytes_len)
            .ok_or(VmError::InvalidControlDescriptor)?;
        let control_descriptors = if num_controls > 0 {
            let desc_bytes = bytes
                .get(sc_off..sc_desc_end)
                .ok_or(VmError::InvalidControlDescriptor)?;
            <[ControlClampDescriptorWire]>::ref_from_bytes(desc_bytes)
                .map_err(|_| VmError::InvalidControlDescriptor)?
        } else {
            &[]
        };
        let cp1_over_len = meta.cp1_overflow_bytes.get() as usize;
        let cp1_overflow = if cp1_over_len > 0 {
            let cp1_start = if num_controls > 0 {
                sc_desc_end
            } else {
                sc_off
            };
            let cp1_end = cp1_start
                .checked_add(cp1_over_len)
                .ok_or(VmError::InvalidCp1Stream)?;
            bytes
                .get(cp1_start..cp1_end)
                .ok_or(VmError::InvalidCp1Stream)?
        } else {
            &[]
        };
        validate_descriptors(control_descriptors, num_controls)?;
        validate_cp1_overflow(cp1_overflow, num_controls)?;

        let tsyn_bytes_len = tsyn_count
            .checked_mul(TSYN_ENTRY_BYTES)
            .ok_or(VmError::InvalidTempoSyncEntry)?;
        let tsyn_end = tsyn_off
            .checked_add(tsyn_bytes_len)
            .ok_or(VmError::InvalidTempoSyncEntry)?;
        let tsyn_entries = if tsyn_count > 0 {
            let tsyn_bytes = bytes
                .get(tsyn_off..tsyn_end)
                .ok_or(VmError::InvalidTempoSyncEntry)?;
            <[TempoSyncEntryWire]>::ref_from_bytes(tsyn_bytes)
                .map_err(|_| VmError::InvalidTempoSyncEntry)?
        } else {
            &[]
        };
        validate_tsyn_entries(tsyn_entries, control_descriptors, num_controls)?;

        let cext = if meta.cext_len.get() == CEXT_DISPATCH_BYTES as u32 {
            let cext_end = cext_off
                .checked_add(CEXT_DISPATCH_BYTES)
                .ok_or(VmError::InvalidRegionLayout)?;
            let cext_bytes = bytes
                .get(cext_off..cext_end)
                .ok_or(VmError::InvalidRegionLayout)?;
            Some(
                CextDispatchWire::ref_from_bytes(cext_bytes)
                    .map_err(|_| VmError::InvalidRegionLayout)?,
            )
        } else {
            None
        };

        let a0_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[0])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena0 = bytes
            .get(arena0_off..a0_end)
            .and_then(|s| Arena0LiteWire::ref_from_bytes(s).ok())
            .ok_or(VmError::InvalidArenaHeader)?;
        let a1_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[1])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena1 = if retained_idx >= 1 {
            Some(
                bytes
                    .get(a0_end..a1_end)
                    .and_then(|s| Arena1FullDeltaWire::ref_from_bytes(s).ok())
                    .ok_or(VmError::InvalidArenaHeader)?,
            )
        } else {
            None
        };
        let a2_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[2])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena2 = if retained_idx >= 2 {
            Some(
                bytes
                    .get(a1_end..a2_end)
                    .and_then(|s| Arena2ProDeltaWire::ref_from_bytes(s).ok())
                    .ok_or(VmError::InvalidArenaHeader)?,
            )
        } else {
            None
        };
        let a3_end = arena0_off
            .checked_add(ARENA_TIER_BYTES[3])
            .ok_or(VmError::InvalidArenaHeader)?;
        let arena3 = if retained_idx >= 3 {
            Some(
                bytes
                    .get(a2_end..a3_end)
                    .and_then(|s| Arena3UltraDeltaWire::ref_from_bytes(s).ok())
                    .ok_or(VmError::InvalidArenaHeader)?,
            )
        } else {
            None
        };

        validate_arena0_and_quads(ArenaValidationInput {
            arena0,
            arena1,
            arena2,
            arena3,
            cp1_overflow_empty: cp1_overflow.is_empty(),
            num_controls,
            max_present_tier: max_tier,
            retained_tier,
            meta,
        })?;

        Ok(Self {
            header,
            runtime_meta: meta,
            control_descriptors,
            cp1_overflow,
            tsyn_entries,
            cext,
            retained_tier,
            arena0,
            arena1,
            arena2,
            arena3,
        })
    }
}

/// Performs zero-mutation `REQ-5.6` tier truncation by returning the exact slice
/// `&bytes[..tier_end_offsets[target_tier]]` after verifying `bytes` is a valid `.dnam-bin`
/// container and `target_tier <= retained_tier`.
#[inline]
pub fn strip_dnam_bin_to_tier(bytes: &[u8], target_tier: Tier) -> Result<&[u8], VmError> {
    let view = DnamBinView::parse(bytes)?;
    if target_tier > view.retained_tier {
        return Err(VmError::InvalidTier);
    }
    let end_off = view.header.tier_end_offsets[target_tier.index()].get() as usize;
    Ok(&bytes[..end_off])
}

/// Non-inlined wrapper over [`DnamBinView::bind_prevalidated`] for release disassembly auditing
/// (`ref[sdk.rust.realtime-disassembly-audit]`).
#[inline(never)]
pub fn bind_dnam_bin_prevalidated(bytes: &[u8]) -> Result<DnamBinView<'_>, VmError> {
    DnamBinView::bind_prevalidated(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rejects_short_buffer_and_bad_magic() {
        assert_eq!(
            DnamBinView::parse(&[0u8; 64]).unwrap_err(),
            VmError::TruncatedPrefix
        );
        let buf = [0u8; 128];
        assert_eq!(DnamBinView::parse(&buf).unwrap_err(), VmError::InvalidMagic);
    }
}
