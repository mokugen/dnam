//! Deterministic `.dnam-bin` (`"DNBN"`) encoder supporting both `#![no_std]` `!alloc`
//! fixed-buffer writes ([`encode_dnam_bin_into`]) and optional `alloc` `Vec<u8>` output
//! ([`encode_dnam_bin`]).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use zerocopy::{FromZeros, IntoBytes};

use super::arena_wire::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    PIPE_CANONICAL_V3_SONA, SONA_META_OS_EXT_PRESENT, SONA_META_ROUTER_EXT, SONA_META_TANK_LFO,
    SONA_META_TANK_TAP_GAINS, TANK_FLAG_EXT_F32,
};
use super::header::{
    ARENA_TIER_BYTES, CEXT_DISPATCH_BYTES, CextDispatchWire, ControlClampDescriptorWire,
    DBRM_MAGIC, DNBN_FLAG_FREEZE_EVICTED_MASK, DNBN_FLAG_RESERVED_HI_MASK, DNBN_FLAG_STEREO_OUT,
    DNBN_MAGIC, DNBN_VERSION_MAJOR, DNBN_VERSION_MINOR, DnamBinHeaderWire, DnamBinRuntimeMetaWire,
    MAX_NUM_CONTROLS, SONA_CELL_MAGIC, TSYN_ENTRY_BYTES, TempoSyncEntryWire, align64,
};
use super::view::DnamBinView;
use crate::math::{Crc32cBackend, DefaultMathBackend};
use crate::types::{Tier, VmError};

/// Allocation error alias returned by [`encode_dnam_bin`].
#[cfg(feature = "alloc")]
pub type DnamBinBuildOwnedError = VmError;

/// Inputs for deterministic `.dnam-bin` serialization.
#[derive(Debug, Clone, Copy)]
pub struct DnamBinBuildInput<'a> {
    /// Reference training sample rate (`44_100..=96_000`, default `48_000`).
    pub reference_sample_rate_hz: u32,
    /// Optional `DNBN` header flags (bits `0..=3` overwritten from highest present tier).
    pub header_flags: u16,
    /// External delay ring `log2` frames (`0` or `7..=19`).
    pub ext_ring_pow2_frames: u8,
    /// External delay line count (`0..=16`).
    pub ext_ring_lines: u8,
    /// External sample format (`0 = fp16`, `1 = f32`).
    pub ext_ring_format: u8,
    /// Declared `48 kHz` max spatial tank delay (`<= 48_000`).
    pub declared_max_delay_48k: u32,
    /// `5 Hz` wet DC blocker pole (`48 kHz`).
    pub dc_blocker_pole_5hz: f32,
    /// Decay tail duration (`>= 0.0`).
    pub tail_time_ms: f32,
    /// Default anti-creep AC pole (`[0.0, 1.0)`).
    pub r_ac_default: f32,
    /// Switch crossfade samples.
    pub switch_xfade_samples: u16,
    /// First `8 B` of content SHA-256 hash prefix.
    pub content_hash_prefix: [u8; 8],
    /// Parametric control clamp descriptors (`0..=255`).
    pub controls: &'a [ControlClampDescriptorWire],
    /// Optional `"CP1\0"` overflow stream bytes (empty when inline-only).
    pub cp1_overflow: &'a [u8],
    /// Pure-binary `"TSYN"` entries.
    pub tsyn: &'a [TempoSyncEntryWire],
    /// Optional `1,280 B` `"CEXT"` dispatch matrices.
    pub cext: Option<&'a CextDispatchWire>,
    /// Required Arena 0 Lite payload (`4,608 B`).
    pub arena0: &'a Arena0LiteWire,
    /// Optional Arena 1 Full delta (`3,328 B`).
    pub arena1: Option<&'a Arena1FullDeltaWire>,
    /// Optional Arena 2 Pro delta (`7,936 B`, requires `arena1.is_some()`).
    pub arena2: Option<&'a Arena2ProDeltaWire>,
    /// Optional Arena 3 Ultra delta (`14,848 B`, requires `arena1` and `arena2`).
    pub arena3: Option<&'a Arena3UltraDeltaWire>,
}

impl DnamBinBuildInput<'_> {
    /// Creates a valid default Lite [`Arena0LiteWire`] carrying canonical `SONA v3.0` headers.
    pub fn default_arena0_lite() -> Arena0LiteWire {
        let mut a0 = Arena0LiteWire::new_zeroed();
        for i in 0..4 {
            a0.crossbar.s_router[i][i] = zerocopy::byteorder::little_endian::F32::new(1.0);
            a0.crossbar.tier_gain_trim[i] = zerocopy::byteorder::little_endian::F32::new(1.0);
        }
        a0.crossbar.pipeline_flags =
            zerocopy::byteorder::little_endian::U32::new(PIPE_CANONICAL_V3_SONA);
        let meta_flags = SONA_META_OS_EXT_PRESENT
            | SONA_META_ROUTER_EXT
            | SONA_META_TANK_TAP_GAINS
            | SONA_META_TANK_LFO;
        a0.crossbar.sona_meta_flags = zerocopy::byteorder::little_endian::U32::new(meta_flags);
        a0.crossbar.anti_zipper_ramp_scale = zerocopy::byteorder::little_endian::F32::new(0.0625);
        a0.crossbar.router_gain = [
            zerocopy::byteorder::little_endian::F32::new(1.0),
            zerocopy::byteorder::little_endian::F32::new(1.0),
        ];
        a0.crossbar.router_mix_matrix[0][0] = zerocopy::byteorder::little_endian::F32::new(1.0);
        a0.crossbar.router_mix_matrix[1][1] = zerocopy::byteorder::little_endian::F32::new(1.0);
        a0.crossbar.cell_desc.magic = SONA_CELL_MAGIC;
        a0.crossbar.cell_desc.version_major =
            zerocopy::byteorder::little_endian::U16::new(DNBN_VERSION_MAJOR);
        a0.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(0x0001);
        a0.crossbar.cell_desc.oversample_factor = zerocopy::byteorder::little_endian::U16::new(1);
        a0.crossbar.cell_desc.cell_flags = zerocopy::byteorder::little_endian::U32::new(meta_flags);
        a0.spatial_tank_lite.r_dc =
            zerocopy::byteorder::little_endian::F32::new(crate::math::TANK_DC_BLOCKER_48K_F32);
        a0.control_bus_lite.rho_drift_train = zerocopy::byteorder::little_endian::F32::new(0.999);
        a0.control_bus_lite.env_poles = [
            zerocopy::byteorder::little_endian::F32::new(0.95),
            zerocopy::byteorder::little_endian::F32::new(0.99),
            zerocopy::byteorder::little_endian::F32::new(0.998),
        ];
        a0.control_bus_lite.trust_radius_mh = zerocopy::byteorder::little_endian::F32::new(2.5);
        a0.control_bus_lite.rho_head_and_ctrl_tail[0] =
            zerocopy::byteorder::little_endian::F32::new(2.5);
        a0
    }
}

fn determine_max_tier_and_offsets(
    input: &DnamBinBuildInput<'_>,
) -> Result<(Tier, usize, usize, usize, usize, usize), VmError> {
    if input.controls.len() > MAX_NUM_CONTROLS {
        return Err(VmError::InvalidControlDescriptor);
    }
    if !(input.cp1_overflow.is_empty() || input.cp1_overflow.len() >= 8) {
        return Err(VmError::InvalidCp1Stream);
    }
    let max_tier = match (input.arena1, input.arena2, input.arena3) {
        (None, None, None) => Tier::Lite,
        (Some(_), None, None) => Tier::Full,
        (Some(_), Some(_), None) => Tier::Pro,
        (Some(_), Some(_), Some(_)) => Tier::Ultra,
        _ => return Err(VmError::InvalidTier),
    };
    let mut cursor = 128usize;
    let sc_len = input.controls.len() * 64 + input.cp1_overflow.len();
    let sc_off = if sc_len > 0 {
        let off = cursor;
        cursor = align64(off + sc_len);
        off
    } else {
        0
    };
    let tsyn_len = input.tsyn.len() * TSYN_ENTRY_BYTES;
    let tsyn_off = if tsyn_len > 0 {
        let off = cursor;
        cursor = align64(off + tsyn_len);
        off
    } else {
        0
    };
    let cext_off = if input.cext.is_some() {
        let off = cursor;
        cursor = align64(off + CEXT_DISPATCH_BYTES);
        off
    } else {
        0
    };
    let arena0_off = align64(cursor);
    let total_len = arena0_off + ARENA_TIER_BYTES[max_tier.index()];
    Ok((max_tier, sc_off, tsyn_off, cext_off, arena0_off, total_len))
}

/// Computes the exact byte length of the serialized `.dnam-bin` payload without allocation.
#[inline]
pub fn computed_dnam_bin_byte_len(input: &DnamBinBuildInput<'_>) -> Result<usize, VmError> {
    let (_, _, _, _, _, total_len) = determine_max_tier_and_offsets(input)?;
    Ok(total_len)
}

/// Encodes `input` into caller-supplied slice `out` (zero heap allocation) and returns byte count.
pub fn encode_dnam_bin_into(
    input: &DnamBinBuildInput<'_>,
    out: &mut [u8],
) -> Result<usize, VmError> {
    encode_dnam_bin_into_with_crc::<DefaultMathBackend>(input, out)
}

/// Encodes `input` into caller-supplied slice `out` using a specified [`Crc32cBackend`].
pub fn encode_dnam_bin_into_with_crc<C: Crc32cBackend>(
    input: &DnamBinBuildInput<'_>,
    out: &mut [u8],
) -> Result<usize, VmError> {
    if (input.header_flags & (DNBN_FLAG_FREEZE_EVICTED_MASK | DNBN_FLAG_RESERVED_HI_MASK)) != 0 {
        return Err(VmError::InvalidHeaderFlagsOrReserved);
    }
    let (max_tier, sc_off, tsyn_off, cext_off, arena0_off, total_len) =
        determine_max_tier_and_offsets(input)?;
    if out.len() < total_len {
        return Err(VmError::InsufficientBackingCapacity);
    }
    let dst = &mut out[..total_len];
    dst.fill(0);

    let max_idx = max_tier.index();
    let tier_mask = (1_u16 << (max_idx + 1)) - 1;
    let flags = (input.header_flags & !0x000F) | tier_mask | DNBN_FLAG_STEREO_OUT;
    let sc_len = input.controls.len() * 64 + input.cp1_overflow.len();
    let tsyn_len = input.tsyn.len() * TSYN_ENTRY_BYTES;
    let cext_len = if input.cext.is_some() {
        CEXT_DISPATCH_BYTES
    } else {
        0
    };

    if sc_len > 0 {
        let mut cursor = sc_off;
        for ctrl in input.controls {
            ctrl.write_to_prefix(&mut dst[cursor..])
                .map_err(|_| VmError::InsufficientBackingCapacity)?;
            cursor += 64;
        }
        if !input.cp1_overflow.is_empty() {
            dst[cursor..cursor + input.cp1_overflow.len()].copy_from_slice(input.cp1_overflow);
        }
    }
    if tsyn_len > 0 {
        let mut cursor = tsyn_off;
        for row in input.tsyn {
            row.write_to_prefix(&mut dst[cursor..])
                .map_err(|_| VmError::InsufficientBackingCapacity)?;
            cursor += TSYN_ENTRY_BYTES;
        }
    }
    if let Some(cext) = input.cext {
        cext.write_to_prefix(&mut dst[cext_off..])
            .map_err(|_| VmError::InsufficientBackingCapacity)?;
    }

    input
        .arena0
        .write_to_prefix(&mut dst[arena0_off..])
        .map_err(|_| VmError::InsufficientBackingCapacity)?;
    if let Some(a1) = input.arena1 {
        a1.write_to_prefix(&mut dst[arena0_off + ARENA_TIER_BYTES[0]..])
            .map_err(|_| VmError::InsufficientBackingCapacity)?;
    }
    if let Some(a2) = input.arena2 {
        a2.write_to_prefix(&mut dst[arena0_off + ARENA_TIER_BYTES[1]..])
            .map_err(|_| VmError::InsufficientBackingCapacity)?;
    }
    if let Some(a3) = input.arena3 {
        a3.write_to_prefix(&mut dst[arena0_off + ARENA_TIER_BYTES[2]..])
            .map_err(|_| VmError::InsufficientBackingCapacity)?;
    }

    let mut any_ext = (input.arena0.spatial_tank_lite.flags & TANK_FLAG_EXT_F32) != 0;
    if let Some(a1) = input.arena1 {
        any_ext |= (a1.spatial_tank_full.flags & TANK_FLAG_EXT_F32) != 0;
    }
    if let Some(a2) = input.arena2 {
        any_ext |= (a2.spatial_tank_pro[0].flags & TANK_FLAG_EXT_F32) != 0;
    }
    if let Some(a3) = input.arena3 {
        any_ext |= (a3.spatial_tank_ultra[0].flags & TANK_FLAG_EXT_F32) != 0;
    }

    if input.ext_ring_format > 1 || (input.ext_ring_format == 1 && !any_ext) {
        return Err(VmError::InvalidHeaderFlagsOrReserved);
    }
    let ext_ring_format = u8::from(any_ext);

    let mut hdr = DnamBinHeaderWire::new_zeroed();
    hdr.magic = DNBN_MAGIC;
    hdr.version_major = zerocopy::byteorder::little_endian::U16::new(DNBN_VERSION_MAJOR);
    hdr.version_minor = zerocopy::byteorder::little_endian::U16::new(DNBN_VERSION_MINOR);
    hdr.max_present_tier = max_idx as u8;
    hdr.flags = zerocopy::byteorder::little_endian::U16::new(flags);
    hdr.num_controls = zerocopy::byteorder::little_endian::U16::new(input.controls.len() as u16);
    hdr.tsyn_count = zerocopy::byteorder::little_endian::U16::new(input.tsyn.len() as u16);
    hdr.reference_sample_rate_hz =
        zerocopy::byteorder::little_endian::U32::new(input.reference_sample_rate_hz);
    hdr.arena0_offset = zerocopy::byteorder::little_endian::U32::new(arena0_off as u32);
    hdr.runtime_meta_bytes = zerocopy::byteorder::little_endian::U16::new(64);
    for t in 0..=max_idx {
        let end_off = arena0_off + ARENA_TIER_BYTES[t];
        let arena_crc = C::crc32c_update(0, &dst[arena0_off..end_off]);
        hdr.tier_end_offsets[t] = zerocopy::byteorder::little_endian::U32::new(end_off as u32);
        hdr.tier_arena_crc32c[t] = zerocopy::byteorder::little_endian::U32::new(arena_crc);
    }

    let meta = DnamBinRuntimeMetaWire {
        magic: DBRM_MAGIC,
        shared_controls_offset: zerocopy::byteorder::little_endian::U32::new(sc_off as u32),
        shared_controls_len: zerocopy::byteorder::little_endian::U32::new(sc_len as u32),
        tsyn_offset: zerocopy::byteorder::little_endian::U32::new(tsyn_off as u32),
        tsyn_len: zerocopy::byteorder::little_endian::U32::new(tsyn_len as u32),
        cext_offset: zerocopy::byteorder::little_endian::U32::new(cext_off as u32),
        cext_len: zerocopy::byteorder::little_endian::U32::new(cext_len as u32),
        ext_ring_pow2_frames: input.ext_ring_pow2_frames,
        ext_ring_lines: input.ext_ring_lines,
        ext_ring_format,
        reserved0: 0,
        declared_max_delay_48k: zerocopy::byteorder::little_endian::U32::new(
            input.declared_max_delay_48k,
        ),
        dc_blocker_pole_5hz: zerocopy::byteorder::little_endian::F32::new(
            input.dc_blocker_pole_5hz,
        ),
        tail_time_ms: zerocopy::byteorder::little_endian::F32::new(input.tail_time_ms),
        r_ac_default: zerocopy::byteorder::little_endian::F32::new(input.r_ac_default),
        switch_xfade_samples: zerocopy::byteorder::little_endian::U16::new(
            input.switch_xfade_samples,
        ),
        reserved_cond_zero: zerocopy::byteorder::little_endian::U16::new(0),
        content_hash_prefix: input.content_hash_prefix,
        cp1_overflow_bytes: zerocopy::byteorder::little_endian::U32::new(
            input.cp1_overflow.len() as u32
        ),
    };

    hdr.write_to_prefix(&mut dst[0..64])
        .map_err(|_| VmError::InsufficientBackingCapacity)?;
    meta.write_to_prefix(&mut dst[64..128])
        .map_err(|_| VmError::InsufficientBackingCapacity)?;
    let prefix_crc = C::crc32c_update(0, &dst[0..20]);
    let prefix_crc = C::crc32c_update(prefix_crc, &dst[24..arena0_off]);
    hdr.prefix_crc32c = zerocopy::byteorder::little_endian::U32::new(prefix_crc);
    hdr.write_to_prefix(&mut dst[0..64])
        .map_err(|_| VmError::InsufficientBackingCapacity)?;

    DnamBinView::parse_with_crc::<C>(dst)?;
    Ok(total_len)
}

/// Allocates and encodes a complete validated `.dnam-bin` payload into a `Vec<u8>`.
#[cfg(feature = "alloc")]
pub fn encode_dnam_bin(input: &DnamBinBuildInput<'_>) -> Result<Vec<u8>, DnamBinBuildOwnedError> {
    let total_len = computed_dnam_bin_byte_len(input)?;
    let mut out = alloc::vec![0u8; total_len];
    let written = encode_dnam_bin_into(input, &mut out)?;
    out.truncate(written);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_lite_dnam_bin_no_alloc_and_parse() {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 500.0,
            r_ac_default: 0.99,
            switch_xfade_samples: 64,
            content_hash_prefix: [1, 2, 3, 4, 5, 6, 7, 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: None,
            arena2: None,
            arena3: None,
        };
        let len = computed_dnam_bin_byte_len(&input).unwrap();
        assert_eq!(len, 128 + 4_608);
        let mut stack_buf = [0u8; 128 + 4_608];
        let written = encode_dnam_bin_into(&input, &mut stack_buf).unwrap();
        assert_eq!(written, len);
        let view = DnamBinView::parse(&stack_buf).unwrap();
        assert_eq!(view.retained_tier, Tier::Lite);
        assert_eq!(view.header.flags.get(), 0x0201);

        let mut short_buf = [0u8; 64];
        assert_eq!(
            encode_dnam_bin_into(&input, &mut short_buf).unwrap_err(),
            VmError::InsufficientBackingCapacity
        );

        let a1 = Arena1FullDeltaWire::new_zeroed();
        let a2 = Arena2ProDeltaWire::new_zeroed();
        let mut a0_full = a0;
        a0_full.crossbar.cell_desc.tier_flags =
            zerocopy::byteorder::little_endian::U16::new(0b0011);
        let full_input = DnamBinBuildInput {
            arena0: &a0_full,
            arena1: Some(&a1),
            ..input
        };
        let mut full_buf = [0u8; 128 + 7_936];
        let full_written = encode_dnam_bin_into(&full_input, &mut full_buf).unwrap();
        assert_eq!(
            DnamBinView::parse(&full_buf[..full_written])
                .unwrap()
                .retained_tier,
            Tier::Full
        );

        let mut a0_pro = a0;
        a0_pro.crossbar.cell_desc.tier_flags = zerocopy::byteorder::little_endian::U16::new(0b0111);
        let pro_input = DnamBinBuildInput {
            arena0: &a0_pro,
            arena1: Some(&a1),
            arena2: Some(&a2),
            ..input
        };
        let mut pro_buf = [0u8; 128 + 15_872];
        let pro_written = encode_dnam_bin_into(&pro_input, &mut pro_buf).unwrap();
        assert_eq!(
            DnamBinView::parse(&pro_buf[..pro_written])
                .unwrap()
                .retained_tier,
            Tier::Pro
        );
    }
}
