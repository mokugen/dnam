//! Structural, region-layout, and schema validation helpers for [`DnamBinView`].

mod arena;

pub(super) use arena::{ArenaValidationInput, validate_arena0_and_quads};
use zerocopy::FromBytes;

use crate::bin::header::{
    CP1_STREAM_FLAG_HAS_ULTRA, CP1_STREAM_MAGIC, ControlClampDescriptorWire, Cp1StreamDirEntryWire,
    Cp1StreamHeaderWire, DBRM_MAGIC, DNBN_FLAG_FREEZE_EVICTED_MASK, DNBN_FLAG_RESERVED_HI_MASK,
    DNBN_MAGIC, DNBN_VERSION_MAJOR, DNBN_VERSION_MINOR, DnamBinHeaderWire, DnamBinRuntimeMetaWire,
    MAX_NUM_CONTROLS, TSYN_ENTRY_BYTES, TempoSyncEntryWire,
};
use crate::control::MAX_EXACT_CONTRIBUTORS;
use crate::types::VmError;

#[inline]
fn checked_align64(v: usize) -> Option<usize> {
    v.checked_add(63).map(|x| x & !63)
}

fn ensure_all_zero(bytes: &[u8]) -> Result<(), VmError> {
    if bytes.iter().all(|&b| b == 0) {
        Ok(())
    } else {
        Err(VmError::NonZeroAlignmentPadding)
    }
}

pub(super) fn validate_prefix_header(
    header: &DnamBinHeaderWire,
    meta: &DnamBinRuntimeMetaWire,
) -> Result<(usize, usize, usize), VmError> {
    if header.magic != DNBN_MAGIC || meta.magic != DBRM_MAGIC {
        return Err(VmError::InvalidMagic);
    }
    if header.version_major.get() != DNBN_VERSION_MAJOR
        || header.version_minor.get() != DNBN_VERSION_MINOR
    {
        return Err(VmError::UnsupportedVersion);
    }
    if header.reserved0 != 0
        || header.runtime_meta_bytes.get() != 64
        || header.reserved_flags_hi.get() != 0
        || meta.reserved0 != 0
        || meta.reserved_cond_zero.get() != 0
    {
        return Err(VmError::InvalidHeaderFlagsOrReserved);
    }
    let max_tier = header.max_present_tier as usize;
    if max_tier > 3 {
        return Err(VmError::InvalidTier);
    }
    let expected_tier_mask = (1_u16 << (max_tier + 1)) - 1;
    let flags = header.flags.get();
    if (flags & (DNBN_FLAG_FREEZE_EVICTED_MASK | DNBN_FLAG_RESERVED_HI_MASK)) != 0
        || (flags & 0x000F) != expected_tier_mask
    {
        return Err(VmError::InvalidHeaderFlagsOrReserved);
    }
    let sr = header.reference_sample_rate_hz.get();
    if !(44_100..=96_000).contains(&sr) {
        return Err(VmError::InvalidSampleRate);
    }
    let num_controls = header.num_controls.get() as usize;
    if num_controls > MAX_NUM_CONTROLS {
        return Err(VmError::InvalidControlDescriptor);
    }
    if !(meta.ext_ring_pow2_frames == 0 || (7..=19).contains(&meta.ext_ring_pow2_frames))
        || meta.ext_ring_lines > 16
        || meta.ext_ring_format > 1
        || meta.declared_max_delay_48k.get() > 48_000
    {
        return Err(VmError::InvalidHeaderFlagsOrReserved);
    }
    let dc_pole = meta.dc_blocker_pole_5hz.get();
    let tail_ms = meta.tail_time_ms.get();
    if !dc_pole.is_finite() || !tail_ms.is_finite() || tail_ms < 0.0 {
        return Err(VmError::InvalidHeaderFlagsOrReserved);
    }
    let tsyn_count = header.tsyn_count.get() as usize;
    Ok((max_tier, num_controls, tsyn_count))
}

pub(super) fn validate_region_layout_and_padding(
    bytes: &[u8],
    header: &DnamBinHeaderWire,
    meta: &DnamBinRuntimeMetaWire,
    num_controls: usize,
    tsyn_count: usize,
) -> Result<(usize, usize, usize, usize), VmError> {
    let arena0_off = header.arena0_offset.get() as usize;
    if arena0_off < 128 || (arena0_off & 63) != 0 || arena0_off > bytes.len() {
        return Err(VmError::InvalidRegionLayout);
    }
    let sc_off = meta.shared_controls_offset.get() as usize;
    let sc_len = meta.shared_controls_len.get() as usize;
    let cp1_over_len = meta.cp1_overflow_bytes.get() as usize;
    if !(cp1_over_len == 0 || cp1_over_len >= 8) {
        return Err(VmError::InvalidCp1Stream);
    }
    let desc_bytes = num_controls
        .checked_mul(64)
        .ok_or(VmError::InvalidRegionLayout)?;
    let expected_sc_len = desc_bytes
        .checked_add(cp1_over_len)
        .ok_or(VmError::InvalidRegionLayout)?;
    if sc_len != expected_sc_len
        || (sc_len == 0 && sc_off != 0)
        || (sc_len > 0 && (sc_off < 128 || (sc_off & 63) != 0))
    {
        return Err(VmError::InvalidRegionLayout);
    }
    let sc_end = sc_off
        .checked_add(sc_len)
        .ok_or(VmError::InvalidRegionLayout)?;
    let sc_end_aligned = if sc_len > 0 {
        checked_align64(sc_end).ok_or(VmError::InvalidRegionLayout)?
    } else {
        128
    };

    let tsyn_off = meta.tsyn_offset.get() as usize;
    let tsyn_len = meta.tsyn_len.get() as usize;
    let expected_tsyn_len = tsyn_count
        .checked_mul(TSYN_ENTRY_BYTES)
        .ok_or(VmError::InvalidRegionLayout)?;
    if tsyn_len != expected_tsyn_len
        || (tsyn_len == 0 && tsyn_off != 0)
        || (tsyn_len > 0 && ((tsyn_off & 63) != 0 || tsyn_off < sc_end_aligned))
    {
        return Err(VmError::InvalidRegionLayout);
    }
    let tsyn_end = tsyn_off
        .checked_add(tsyn_len)
        .ok_or(VmError::InvalidRegionLayout)?;
    let tsyn_end_aligned = if tsyn_len > 0 {
        checked_align64(tsyn_end).ok_or(VmError::InvalidRegionLayout)?
    } else {
        sc_end_aligned
    };

    let cext_off = meta.cext_offset.get() as usize;
    let cext_len = meta.cext_len.get() as usize;
    if cext_off != 0 || cext_len != 0 {
        return Err(VmError::InvalidRegionLayout);
    }
    let cext_end_aligned = tsyn_end_aligned;
    if cext_end_aligned > arena0_off {
        return Err(VmError::InvalidRegionLayout);
    }

    let mut cursor = 128usize;
    if sc_len > 0 {
        ensure_all_zero(
            bytes
                .get(cursor..sc_off)
                .ok_or(VmError::InvalidRegionLayout)?,
        )?;
        cursor = sc_end;
    }
    if tsyn_len > 0 {
        ensure_all_zero(
            bytes
                .get(cursor..tsyn_off)
                .ok_or(VmError::InvalidRegionLayout)?,
        )?;
        cursor = tsyn_end;
    }
    ensure_all_zero(
        bytes
            .get(cursor..arena0_off)
            .ok_or(VmError::InvalidRegionLayout)?,
    )?;
    Ok((sc_off, tsyn_off, cext_off, arena0_off))
}

pub(super) fn validate_descriptors(
    descriptors: &[ControlClampDescriptorWire],
    num_controls: usize,
) -> Result<(), VmError> {
    for desc in descriptors {
        if desc._reserved.get() != 0 {
            return Err(VmError::InvalidControlDescriptor);
        }
        match desc.ctrl_kind {
            0 => {
                if desc.lobatto_degree > 3 {
                    return Err(VmError::InvalidControlDescriptor);
                }
                let smin = desc.safe_min_base.get();
                let smax = desc.safe_max_base.get();
                if !desc.default_val.get().is_finite()
                    || !smin.is_finite()
                    || !smax.is_finite()
                    || smin > smax
                    || !desc.scale_norm.get().is_finite()
                    || !desc.offset_norm.get().is_finite()
                    || !desc.anchor_nodes_norm.iter().all(|v| v.get().is_finite())
                {
                    return Err(VmError::InvalidControlDescriptor);
                }
            }
            1 => {
                let deg = desc.lobatto_degree;
                let mask = desc.active_options_mask.get();
                if !desc.default_val.get().is_finite()
                    || deg > 15
                    || mask == 0
                    || (u32::from(mask) >> (u32::from(deg) + 1)) != 0
                {
                    return Err(VmError::InvalidControlDescriptor);
                }
            }
            0xFF => {}
            _ => return Err(VmError::InvalidControlDescriptor),
        }
        for rule in &desc.cond_rules {
            if !matches!(rule.rule_op, 0 | 1 | 2 | 3 | 5 | 6) {
                return Err(VmError::InvalidControlDescriptor);
            }
            if rule.rule_op != 0 {
                let sw_idx = usize::from(rule.switch_ctrl_idx.get());
                let valid_target = sw_idx < num_controls
                    && descriptors.get(sw_idx).is_some_and(|sw| sw.ctrl_kind == 1);
                if !valid_target || rule.switch_opt_idx > 15 || !rule.bound_val.get().is_finite() {
                    return Err(VmError::InvalidControlDescriptor);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn validate_cp1_overflow(bytes: &[u8], num_controls: usize) -> Result<(), VmError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let (hdr, rem) =
        Cp1StreamHeaderWire::read_from_prefix(bytes).or(Err(VmError::InvalidCp1Stream))?;
    if hdr.magic != CP1_STREAM_MAGIC
        || (hdr.flags & !CP1_STREAM_FLAG_HAS_ULTRA) != 0
        || hdr.reserved != 0
    {
        return Err(VmError::InvalidCp1Stream);
    }
    let count = hdr.count.get() as usize;
    if count > MAX_EXACT_CONTRIBUTORS {
        return Err(VmError::InvalidCp1Stream);
    }
    let dir_bytes_len = count.checked_mul(4).ok_or(VmError::InvalidCp1Stream)?;
    let (dir_bytes, payload) = rem
        .split_at_checked(dir_bytes_len)
        .ok_or(VmError::InvalidCp1Stream)?;
    let entries =
        <[Cp1StreamDirEntryWire]>::ref_from_bytes(dir_bytes).or(Err(VmError::InvalidCp1Stream))?;
    let mut base_bytes = 0usize;
    let mut ultra_bytes = 0usize;
    for entry in entries {
        if (entry.raw_kl & 0xF0) != 0 {
            return Err(VmError::InvalidCp1Stream);
        }
        let kind = (entry.raw_kl >> 2) & 0b11;
        let lanes = usize::from(entry.raw_kl & 0b11);
        if usize::from(entry.control_idx.get()) >= num_controls || kind > 2 || lanes == 0 {
            return Err(VmError::InvalidCp1Stream);
        }
        if (kind == 1 && lanes != 1) || (kind == 2 && usize::from(entry.aux) >= num_controls) {
            return Err(VmError::InvalidCp1Stream);
        }
        let rec_base = 4usize
            .checked_mul(9 * lanes + 4)
            .ok_or(VmError::InvalidCp1Stream)?;
        base_bytes = base_bytes
            .checked_add(rec_base)
            .ok_or(VmError::InvalidCp1Stream)?;
        if (hdr.flags & CP1_STREAM_FLAG_HAS_ULTRA) != 0 {
            let rec_ultra = 4usize
                .checked_mul(8 * lanes + 4)
                .ok_or(VmError::InvalidCp1Stream)?;
            ultra_bytes = ultra_bytes
                .checked_add(rec_ultra)
                .ok_or(VmError::InvalidCp1Stream)?;
        }
    }
    let total_payload = base_bytes
        .checked_add(ultra_bytes)
        .ok_or(VmError::InvalidCp1Stream)?;
    if payload.len() != total_payload {
        return Err(VmError::InvalidCp1Stream);
    }
    Ok(())
}

pub(super) fn validate_tsyn_entries(
    entries: &[TempoSyncEntryWire],
    descriptors: &[ControlClampDescriptorWire],
    num_controls: usize,
) -> Result<(), VmError> {
    for row in entries {
        let ctrl_idx = usize::from(row.ctrl_idx.get());
        let sw_idx = usize::from(row.switch_ctrl_idx.get());
        let valid_target =
            ctrl_idx < num_controls && descriptors.get(ctrl_idx).is_some_and(|d| d.ctrl_kind == 0);
        let valid_switch = row.has_switch_cond == 0
            || (row.has_switch_cond == 1
                && sw_idx < num_controls
                && descriptors.get(sw_idx).is_some_and(|d| d.ctrl_kind == 1)
                && row.switch_opt_idx <= 15);
        if !valid_target
            || !valid_switch
            || row.has_switch_cond > 1
            || row.mode > 1
            || row.default_subdiv >= 12
            || (row.flags.get() & !1) != 0
        {
            return Err(VmError::InvalidTempoSyncEntry);
        }
        let div_ratio = row.div_ratio.get();
        let bpm_factor = row.bpm_factor.get();
        let d_min = row.declared_min.get();
        let d_max = row.declared_max.get();
        if !div_ratio.is_finite()
            || div_ratio <= 0.0
            || !bpm_factor.is_finite()
            || bpm_factor <= 0.0
            || !d_min.is_finite()
            || !d_max.is_finite()
            || d_min > d_max
            || !row.lobatto_anchors.iter().all(|v| v.get().is_finite())
        {
            return Err(VmError::InvalidTempoSyncEntry);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::arena_wire::{SONA_META_RESERVED_MASK, TRANSPORT_FLAG_EXT_F32};
    use crate::bin::header::{
        ConditionalClampRuleWire, Cp1StreamDirEntryWire, Cp1StreamHeaderWire,
    };
    use crate::bin::{DnamBinBuildInput, DnamBinView, encode_dnam_bin_into};
    use crate::types::Tier;
    use zerocopy::byteorder::little_endian::{F32, U16, U32};
    use zerocopy::{FromZeros, IntoBytes};

    fn make_lite_bytes() -> [u8; 128 + crate::bin::ARENA0_LITE_BYTES] {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let mut buf = [0u8; 128 + crate::bin::ARENA0_LITE_BYTES];
        encode_dnam_bin_into(&input, &mut buf).unwrap();
        buf
    }

    /// verify[container.dnam.vm-binary]
    #[test]
    fn test_prefix_and_region_validation_error_variants() {
        let base = make_lite_bytes();
        let hdr = DnamBinHeaderWire::ref_from_bytes(&base[0..64]).unwrap();
        let meta = DnamBinRuntimeMetaWire::ref_from_bytes(&base[64..128]).unwrap();

        let mut bad_hdr = *hdr;
        bad_hdr.version_major = U16::new(9);
        assert_eq!(
            validate_prefix_header(&bad_hdr, meta).unwrap_err(),
            VmError::UnsupportedVersion
        );

        let mut bad_res = *hdr;
        bad_res.reserved0 = 1;
        assert_eq!(
            validate_prefix_header(&bad_res, meta).unwrap_err(),
            VmError::InvalidHeaderFlagsOrReserved
        );

        let mut bad_tier = *hdr;
        bad_tier.max_present_tier = 4;
        assert_eq!(
            validate_prefix_header(&bad_tier, meta).unwrap_err(),
            VmError::InvalidTier
        );

        let mut bad_flags = *hdr;
        bad_flags.flags = U16::new(0x020F);
        assert_eq!(
            validate_prefix_header(&bad_flags, meta).unwrap_err(),
            VmError::InvalidHeaderFlagsOrReserved
        );

        let mut bad_sr = *hdr;
        bad_sr.reference_sample_rate_hz = U32::new(22_050);
        assert_eq!(
            validate_prefix_header(&bad_sr, meta).unwrap_err(),
            VmError::InvalidSampleRate
        );

        let mut bad_ctrl = *hdr;
        bad_ctrl.num_controls = U16::new(300);
        assert_eq!(
            validate_prefix_header(&bad_ctrl, meta).unwrap_err(),
            VmError::InvalidControlDescriptor
        );

        let mut bad_meta = *meta;
        bad_meta.ext_ring_lines = 99;
        assert_eq!(
            validate_prefix_header(hdr, &bad_meta).unwrap_err(),
            VmError::InvalidHeaderFlagsOrReserved
        );
        bad_meta = *meta;
        bad_meta.tail_time_ms = F32::new(-1.0);
        assert_eq!(
            validate_prefix_header(hdr, &bad_meta).unwrap_err(),
            VmError::InvalidHeaderFlagsOrReserved
        );
        bad_meta = *meta;
        bad_meta.dc_blocker_pole_5hz = F32::new(f32::NAN);
        assert_eq!(
            validate_prefix_header(hdr, &bad_meta).unwrap_err(),
            VmError::InvalidHeaderFlagsOrReserved
        );
        // Out-of-range or non-finite ro_pole_48k passes header validation for Table 4 cold clamping.
        for ro in [-0.5_f32, 1.0, f32::NAN, f32::INFINITY] {
            let mut ro_meta = *meta;
            ro_meta.ro_pole_48k = F32::new(ro);
            assert!(validate_prefix_header(hdr, &ro_meta).is_ok());
        }

        // Region layout errors: unaligned arena0, non-zero padding, bad cp1_overflow size.
        let mut unaligned_hdr = *hdr;
        unaligned_hdr.arena0_offset = U32::new(130);
        assert_eq!(
            validate_region_layout_and_padding(&base, &unaligned_hdr, meta, 0, 0).unwrap_err(),
            VmError::InvalidRegionLayout
        );

        let mut dirty_buf = [0u8; 256 + 4_608];
        dirty_buf[128] = 0xAA;
        let mut off_hdr = *hdr;
        off_hdr.arena0_offset = U32::new(192);
        assert_eq!(
            validate_region_layout_and_padding(&dirty_buf, &off_hdr, meta, 0, 0).unwrap_err(),
            VmError::NonZeroAlignmentPadding
        );

        let mut bad_cp1_meta = *meta;
        bad_cp1_meta.cp1_overflow_bytes = U32::new(4);
        assert_eq!(
            validate_region_layout_and_padding(&base, hdr, &bad_cp1_meta, 0, 0).unwrap_err(),
            VmError::InvalidCp1Stream
        );
    }

    /// verify[container.dnam.vm-binary]
    #[test]
    fn test_descriptor_cp1_tsyn_and_arena_validation_errors() {
        let mut d = ControlClampDescriptorWire::new_zeroed();
        d._reserved = U32::new(1);
        assert_eq!(
            validate_descriptors(&[d], 1).unwrap_err(),
            VmError::InvalidControlDescriptor
        );

        d = ControlClampDescriptorWire::new_zeroed();
        d.ctrl_kind = 0;
        d.lobatto_degree = 9;
        assert_eq!(
            validate_descriptors(&[d], 1).unwrap_err(),
            VmError::InvalidControlDescriptor
        );

        d = ControlClampDescriptorWire::new_zeroed();
        d.ctrl_kind = 1;
        d.lobatto_degree = 1;
        d.active_options_mask = U16::new(0);
        assert_eq!(
            validate_descriptors(&[d], 1).unwrap_err(),
            VmError::InvalidControlDescriptor
        );

        d = ControlClampDescriptorWire::new_zeroed();
        d.ctrl_kind = 0x7F;
        assert_eq!(
            validate_descriptors(&[d], 1).unwrap_err(),
            VmError::InvalidControlDescriptor
        );

        d = ControlClampDescriptorWire::new_zeroed();
        d.ctrl_kind = 0xFF;
        d.cond_rules[0] = ConditionalClampRuleWire {
            switch_ctrl_idx: U16::new(5),
            switch_opt_idx: 0,
            rule_op: 1,
            bound_val: F32::new(1.0),
        };
        assert_eq!(
            validate_descriptors(&[d], 1).unwrap_err(),
            VmError::InvalidControlDescriptor
        );

        // Reject cond_rules pointing to non-switch control or switch_opt_idx > 15.
        let mut range_d = ControlClampDescriptorWire::new_zeroed();
        range_d.ctrl_kind = 0;
        range_d.cond_rules[0] = ConditionalClampRuleWire {
            switch_ctrl_idx: U16::new(0), // points at self (Range, not Switch!)
            switch_opt_idx: 0,
            rule_op: 1,
            bound_val: F32::new(1.0),
        };
        assert_eq!(
            validate_descriptors(&[range_d], 1).unwrap_err(),
            VmError::InvalidControlDescriptor
        );
        let mut sw_d = ControlClampDescriptorWire::new_zeroed();
        sw_d.ctrl_kind = 1;
        sw_d.lobatto_degree = 1;
        sw_d.active_options_mask = U16::new(3);
        range_d.cond_rules[0] = ConditionalClampRuleWire {
            switch_ctrl_idx: U16::new(1),
            switch_opt_idx: 16, // > 15
            rule_op: 1,
            bound_val: F32::new(1.0),
        };
        assert_eq!(
            validate_descriptors(&[range_d, sw_d], 2).unwrap_err(),
            VmError::InvalidControlDescriptor
        );

        // CP1 stream rejections (including upper nibble != 0 and switch lanes != 1).
        assert_eq!(
            validate_cp1_overflow(&[0u8; 8], 1).unwrap_err(),
            VmError::InvalidCp1Stream
        );
        let mut cp1_buf = [0u8; 16];
        let hdr = Cp1StreamHeaderWire {
            magic: CP1_STREAM_MAGIC,
            count: U16::new(1),
            flags: 0,
            reserved: 0,
        };
        hdr.write_to_prefix(&mut cp1_buf[0..8]).unwrap();
        let entry = Cp1StreamDirEntryWire {
            control_idx: U16::new(0),
            raw_kl: (2 << 2) | 1,
            aux: 9,
        };
        entry.write_to_prefix(&mut cp1_buf[8..12]).unwrap();
        assert_eq!(
            validate_cp1_overflow(&cp1_buf, 1).unwrap_err(),
            VmError::InvalidCp1Stream
        );
        let hi_nibble = Cp1StreamDirEntryWire {
            control_idx: U16::new(0),
            raw_kl: 0x11,
            aux: 0xFF,
        };
        hi_nibble.write_to_prefix(&mut cp1_buf[8..12]).unwrap();
        assert_eq!(
            validate_cp1_overflow(&cp1_buf, 1).unwrap_err(),
            VmError::InvalidCp1Stream
        );
        let sw_two_lanes = Cp1StreamDirEntryWire {
            control_idx: U16::new(0),
            raw_kl: (1 << 2) | 2,
            aux: 0,
        };
        sw_two_lanes.write_to_prefix(&mut cp1_buf[8..12]).unwrap();
        assert_eq!(
            validate_cp1_overflow(&cp1_buf, 1).unwrap_err(),
            VmError::InvalidCp1Stream
        );

        // TSYN rejections (requires Range target & valid Switch condition).
        let bad_row = TempoSyncEntryWire::new_zeroed();
        assert_eq!(
            validate_tsyn_entries(&[bad_row], &[sw_d], 1).unwrap_err(),
            VmError::InvalidTempoSyncEntry
        );

        // Arena / quad rejections.
        let base = make_lite_bytes();
        let view = DnamBinView::parse(&base).unwrap();
        let mut bad_a0 = *view.arena0;
        bad_a0.crossbar.pipeline_flags = U32::new(0);
        let check_lite_bad = |a0: &crate::bin::arena_wire::Arena0LiteWire| {
            validate_arena0_and_quads(ArenaValidationInput {
                arena0: a0,
                arena1: None,
                arena2: None,
                arena3: None,
                cp1_overflow_empty: true,
                num_controls: 0,
                max_present_tier: 0,
                retained_tier: Tier::Lite,
                meta: view.runtime_meta,
            })
            .unwrap_err()
        };
        assert_eq!(check_lite_bad(&bad_a0), VmError::InvalidArenaHeader);
        bad_a0 = *view.arena0;
        bad_a0.crossbar.sona_meta_flags = U32::new(SONA_META_RESERVED_MASK);
        assert_eq!(check_lite_bad(&bad_a0), VmError::InvalidArenaHeader);
        bad_a0 = *view.arena0;
        bad_a0.transport_lite.flags = 0x80;
        assert_eq!(check_lite_bad(&bad_a0), VmError::InvalidArenaHeader);
        bad_a0 = *view.arena0;
        bad_a0.transport_lite.flags = TRANSPORT_FLAG_EXT_F32;
        assert_eq!(check_lite_bad(&bad_a0), VmError::InvalidArenaHeader);
    }
}
