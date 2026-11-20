//! Arena `0..=3`, `SonaCrossbarBlockWire`, `TankQuadWire`, and inline `cp1_dir` validation helpers.

use crate::bin::arena_wire::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, PIPE_MAGIC_MASK,
    PIPE_REQUIRED_SONA_MASK, SONA_META_CORE_MODS_PRESENT, SONA_META_RESERVED_MASK,
    SonaCrossbarBlockWire, TANK_FLAG_EXT_F32, TANK_FLAG_SCHROEDER_ENABLE, TankQuadWire,
};
use crate::bin::header::{
    DNBN_VERSION_MAJOR, DNBN_VERSION_MINOR, DnamBinRuntimeMetaWire, SONA_CELL_MAGIC,
};
use crate::types::{Tier, VmError};

fn validate_quad(quad: &TankQuadWire) -> Result<bool, VmError> {
    let valid_mask = TANK_FLAG_SCHROEDER_ENABLE | TANK_FLAG_EXT_F32;
    if (quad.flags & !valid_mask) != 0 || quad.reserved != 0 {
        return Err(VmError::InvalidArenaHeader);
    }
    Ok((quad.flags & TANK_FLAG_EXT_F32) != 0)
}

fn validate_crossbar_block_reserved(block: &SonaCrossbarBlockWire) -> Result<(), VmError> {
    if !block.phasor_stereo_phase_fp16[4..8]
        .iter()
        .all(|v| v.get() == 0)
        || !block.crossbar_reserved_fp16.iter().all(|v| v.get() == 0)
    {
        return Err(VmError::InvalidArenaHeader);
    }
    Ok(())
}

fn validate_inline_cp1_dir(
    arena0: &Arena0LiteWire,
    arena3: Option<&Arena3UltraDeltaWire>,
    num_controls: usize,
) -> Result<(), VmError> {
    let dir_count = arena0.control_bus_lite.cp1_dir.cp1_dir_count.get() as usize;
    if dir_count > 8 || arena0.control_bus_lite.cp1_dir.flags.get() != 0 {
        return Err(VmError::InvalidCp1Stream);
    }
    let mut base_bytes = 0usize;
    let mut ultra_bytes = 0usize;
    for &[c_idx, kl, aux] in &arena0.control_bus_lite.cp1_dir.entries[..dir_count] {
        let ctrl_kind = (kl >> 4) & 0x0F;
        let lanes = usize::from(kl & 0x0F);
        if usize::from(c_idx) >= num_controls || ctrl_kind > 2 || !(1..=3).contains(&lanes) {
            return Err(VmError::InvalidCp1Stream);
        }
        if (ctrl_kind == 1 && lanes != 1) || (ctrl_kind == 2 && usize::from(aux) >= num_controls) {
            return Err(VmError::InvalidCp1Stream);
        }
        base_bytes += 4 * (9 * lanes + 4);
        if arena3.is_some() {
            ultra_bytes += 4 * (8 * lanes + 4);
        }
    }
    if base_bytes > 576 || ultra_bytes > 2_112 {
        return Err(VmError::InvalidCp1Stream);
    }
    Ok(())
}

/// Cohesive input bundle for validating arena headers, crossbar blocks, quads, and inline `cp1_dir`.
#[derive(Clone, Copy)]
pub(in crate::bin::view) struct ArenaValidationInput<'a> {
    pub arena0: &'a Arena0LiteWire,
    pub arena1: Option<&'a Arena1FullDeltaWire>,
    pub arena2: Option<&'a Arena2ProDeltaWire>,
    pub arena3: Option<&'a Arena3UltraDeltaWire>,
    pub cp1_overflow_empty: bool,
    pub num_controls: usize,
    pub max_present_tier: usize,
    pub retained_tier: Tier,
    pub meta: &'a DnamBinRuntimeMetaWire,
}

pub(in crate::bin::view) fn validate_arena0_and_quads(
    input: ArenaValidationInput<'_>,
) -> Result<(), VmError> {
    let ArenaValidationInput {
        arena0,
        arena1,
        arena2,
        arena3,
        cp1_overflow_empty,
        num_controls,
        max_present_tier,
        retained_tier,
        meta,
    } = input;
    let pipe = arena0.crossbar.pipeline_flags.get();
    if (pipe & PIPE_MAGIC_MASK) != 0x444E_0000
        || (pipe & PIPE_REQUIRED_SONA_MASK) != PIPE_REQUIRED_SONA_MASK
        || (pipe & 0xFF07) != 0
    {
        return Err(VmError::InvalidArenaHeader);
    }
    let meta_flags = arena0.crossbar.sona_meta_flags.get();
    if (meta_flags & SONA_META_RESERVED_MASK) != 0 {
        return Err(VmError::InvalidArenaHeader);
    }
    let cell = &arena0.crossbar.cell_desc;
    let expected_tier_flags = (1_u16 << (max_present_tier + 1)) - 1;
    if cell.magic != SONA_CELL_MAGIC
        || cell.version_major.get() != DNBN_VERSION_MAJOR
        || cell.version_minor.get() != DNBN_VERSION_MINOR
        || cell.tier_flags.get() != expected_tier_flags
        || !matches!(cell.oversample_factor.get(), 1 | 2)
        || cell.cell_flags.get() != meta_flags
    {
        return Err(VmError::InvalidArenaHeader);
    }
    if cp1_overflow_empty {
        validate_inline_cp1_dir(arena0, arena3, num_controls)?;
    } else if arena0.control_bus_lite.cp1_dir.flags.get() != 0
        || arena0.control_bus_lite.cp1_dir.cp1_dir_count.get() > 8
    {
        return Err(VmError::InvalidCp1Stream);
    }

    let mut any_ext_f32 = validate_quad(&arena0.spatial_tank_lite)?;
    if let Some(a1) = arena1 {
        any_ext_f32 |= validate_quad(&a1.spatial_tank_full)?;
        validate_crossbar_block_reserved(&a1.crossbar_full)?;
    }
    if let Some(a2) = arena2 {
        any_ext_f32 |= validate_quad(&a2.spatial_tank_pro[0])?;
        validate_crossbar_block_reserved(&a2.crossbar_pro)?;
    }
    if let Some(a3) = arena3 {
        any_ext_f32 |= validate_quad(&a3.spatial_tank_ultra[0])?;
        validate_crossbar_block_reserved(&a3.crossbar_ultra)?;
        if !a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[160..256]
            .iter()
            .all(|v| v.get() == 0)
        {
            return Err(VmError::InvalidArenaHeader);
        }
        for k in 0..8 {
            if !a3.stage_latent_core_mods_fp16[k * 128 + 120..k * 128 + 128]
                .iter()
                .all(|v| v.get() == 0)
            {
                return Err(VmError::InvalidArenaHeader);
            }
        }
        if (meta_flags & SONA_META_CORE_MODS_PRESENT) == 0
            && !a3.stage_latent_core_mods_fp16.iter().all(|v| v.get() == 0)
        {
            return Err(VmError::InvalidArenaHeader);
        }
    }
    if (any_ext_f32 && meta.ext_ring_format == 0)
        || (retained_tier.index() == max_present_tier
            && meta.ext_ring_format != u8::from(any_ext_f32))
    {
        return Err(VmError::InvalidArenaHeader);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::{DnamBinBuildInput, DnamBinView, encode_dnam_bin_into};
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{U16, U32};

    #[test]
    fn test_crossbar_and_arena3_reserved_zero_checks() {
        let a0 = DnamBinBuildInput::default_arena0_lite();
        let a1 = Arena1FullDeltaWire::new_zeroed();
        let a2 = Arena2ProDeltaWire::new_zeroed();
        let a3 = Arena3UltraDeltaWire::new_zeroed();
        let mut a0_ultra = a0;
        a0_ultra.crossbar.cell_desc.tier_flags = U16::new(0x000F);
        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: 0,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 0.0,
            r_ac_default: 0.99,
            switch_xfade_samples: 0,
            content_hash_prefix: [0; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0_ultra,
            arena1: Some(&a1),
            arena2: Some(&a2),
            arena3: Some(&a3),
        };
        let mut buf = [0u8; 128 + 30_720];
        let n = encode_dnam_bin_into(&input, &mut buf).unwrap();
        let view = DnamBinView::parse(&buf[..n]).unwrap();

        let mut bad_a1 = a1;
        bad_a1.crossbar_full.crossbar_reserved_fp16[0] = U16::new(1);
        assert_eq!(
            validate_arena0_and_quads(ArenaValidationInput {
                arena0: view.arena0,
                arena1: Some(&bad_a1),
                arena2: None,
                arena3: None,
                cp1_overflow_empty: true,
                num_controls: 0,
                max_present_tier: 3,
                retained_tier: Tier::Full,
                meta: view.runtime_meta,
            })
            .unwrap_err(),
            VmError::InvalidArenaHeader
        );

        let mut bad_a3 = a3;
        bad_a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[160] = U16::new(1);
        assert_eq!(
            validate_arena0_and_quads(ArenaValidationInput {
                arena0: view.arena0,
                arena1: Some(&a1),
                arena2: Some(&a2),
                arena3: Some(&bad_a3),
                cp1_overflow_empty: true,
                num_controls: 0,
                max_present_tier: 3,
                retained_tier: Tier::Ultra,
                meta: view.runtime_meta,
            })
            .unwrap_err(),
            VmError::InvalidArenaHeader
        );

        bad_a3 = a3;
        bad_a3.stage_latent_core_mods_fp16[120] = U16::new(1);
        assert_eq!(
            validate_arena0_and_quads(ArenaValidationInput {
                arena0: view.arena0,
                arena1: Some(&a1),
                arena2: Some(&a2),
                arena3: Some(&bad_a3),
                cp1_overflow_empty: true,
                num_controls: 0,
                max_present_tier: 3,
                retained_tier: Tier::Ultra,
                meta: view.runtime_meta,
            })
            .unwrap_err(),
            VmError::InvalidArenaHeader
        );

        bad_a3 = a3;
        bad_a3.stage_latent_core_mods_fp16[0] = U16::new(1);
        assert_eq!(
            validate_arena0_and_quads(ArenaValidationInput {
                arena0: view.arena0,
                arena1: Some(&a1),
                arena2: Some(&a2),
                arena3: Some(&bad_a3),
                cp1_overflow_empty: true,
                num_controls: 0,
                max_present_tier: 3,
                retained_tier: Tier::Ultra,
                meta: view.runtime_meta,
            })
            .unwrap_err(),
            VmError::InvalidArenaHeader
        );

        let mut a0_core_mods = *view.arena0;
        let mf = a0_core_mods.crossbar.sona_meta_flags.get() | SONA_META_CORE_MODS_PRESENT;
        a0_core_mods.crossbar.sona_meta_flags = U32::new(mf);
        a0_core_mods.crossbar.cell_desc.cell_flags = U32::new(mf);
        assert!(
            validate_arena0_and_quads(ArenaValidationInput {
                arena0: &a0_core_mods,
                arena1: Some(&a1),
                arena2: Some(&a2),
                arena3: Some(&bad_a3),
                cp1_overflow_empty: true,
                num_controls: 0,
                max_present_tier: 3,
                retained_tier: Tier::Ultra,
                meta: view.runtime_meta,
            })
            .is_ok()
        );
    }
}
