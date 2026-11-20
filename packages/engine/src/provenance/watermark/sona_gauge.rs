//! `0-ULP` Odd-Symmetric `SONA` (`PolyADAA1` / `ParaLRU`) Hidden-Lane Sign-Flip Buyer
//! Watermarking (`ref[sec.watermark.reciprocal-gauge]`).
//!
//! Because odd-symmetric `PolyADAA1` and optional `C^2` wavefolder satisfy $g(-x) = -g(x)$
//! at analytical equilibrium $+0.0$, even-activation `adaa_even` satisfies $g_e(-x) = +g_e(x)$,
//! and decoupled `1D` `ParaLRU` commutes with input polarity conjugation when `rect_sign` flips,
//! flipping any active Matryoshka lane $i \in \{0,\dots,39\}$ by $s_i \in \{-1, +1\}$ across input
//! weights, crossbar (`op_p_to_group`, `phasor_to_group`), inter-block `4x4` spatial tiles,
//! `lru_bias_drive`, `rect_sign_fp16`, and `VarPro` linear `odd` readout weights preserves
//! **bit-exact `0.000000 ULP`** audio parity across `Lite..Ultra` and `strip_container_to_tier`.

use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::{F32, U16};

use super::max_tiers_for_arena_len;
use crate::graph::sona_core::model::SONA_INTERBLOCK_TILES;
use crate::graph::weights::{
    ARENA_FULL_BYTES, ARENA_LITE_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES, Arena0LiteWire,
    Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, SonaCrossbarBlockWire,
    SonaGroupWire, SonaInterBlockTilePairWire,
};
use crate::provenance::sha256::sha256_digest;
use crate::schema::DnamError;

/// Cumulative Matryoshka active `SONA` hidden lanes per tier (`16 / 24 / 32 / 40`).
pub const SONA_SIGN_LANES_PER_TIER: [usize; 4] = [16, 24, 32, 40];
/// Maximum active Matryoshka `SONA` hidden lanes across Blocks `0..3` (`[4, 3, 2, 1]` groups × 4 = `40`).
pub const SONA_TOTAL_LANES: usize = 40;

const SIGN_BIT_FP16: u16 = 0x8000;

#[inline]
fn flip_f32_wire(slot: &mut F32, sign: f32) {
    if sign < 0.0 {
        let v = slot.get();
        if v != 0.0 {
            *slot = F32::new(-v);
        }
    }
}

#[inline]
fn flip_fp16_wire(slot: &mut U16, sign: f32) {
    if sign < 0.0 {
        let bits = slot.get();
        if (bits & 0x7FFF) != 0 {
            *slot = U16::new(bits ^ SIGN_BIT_FP16);
        }
    }
}

fn flip_sona_group_wire(gw: &mut SonaGroupWire, global_g: usize, s: &[f32; SONA_TOTAL_LANES]) {
    for l in 0..4 {
        let sign = s[global_g * 4 + l];
        flip_f32_wire(&mut gw.in_x_weight[l], sign);
        flip_f32_wire(&mut gw.lru_bias_drive[l], sign);
        flip_fp16_wire(&mut gw.rect_sign_fp16[l], sign);
    }
}

fn flip_sona_tile_wire(
    tw: &mut SonaInterBlockTilePairWire,
    tile_idx: usize,
    s: &[f32; SONA_TOTAL_LANES],
) {
    let (sg, dg) = SONA_INTERBLOCK_TILES[tile_idx];
    for col in 0..4 {
        let s_src = s[sg * 4 + col];
        for row in 0..4 {
            let s_dst = s[dg * 4 + row];
            flip_f32_wire(&mut tw.w_odd_4x4[col][row], s_dst * s_src);
            flip_fp16_wire(&mut tw.w_even_4x4_fp16[col][row], s_dst);
            flip_fp16_wire(&mut tw.w_lru_4x4_fp16[col][row], s_dst);
        }
    }
}

fn flip_crossbar_active_groups(
    cb: &mut SonaCrossbarBlockWire,
    active_groups: usize,
    s: &[f32; SONA_TOTAL_LANES],
) {
    for g in 0..active_groups {
        for l in 0..4 {
            let sign = s[g * 4 + l];
            flip_fp16_wire(&mut cb.op_p_to_group_fp16[g][l], sign);
            flip_fp16_wire(&mut cb.phasor_to_group_fp16[g][l], sign);
        }
    }
}

fn flip_ro_odd_lanes(w_ro: &mut [U16], lane_count: usize, s: &[f32; SONA_TOTAL_LANES]) {
    for l in 0..lane_count {
        let sign = s[l];
        let base = l * 12;
        flip_fp16_wire(&mut w_ro[base], sign);
        flip_fp16_wire(&mut w_ro[base + 1], sign);
        flip_fp16_wire(&mut w_ro[base + 6], sign);
        flip_fp16_wire(&mut w_ro[base + 7], sign);
    }
}

/// Deterministically derives a prefix-independent 40-lane buyer sign-flip key
/// (`±1.0_f32` per Matryoshka lane, `SHA-256("DNAM-WM-SONA-SIGN-V1:" || buyer_seed || [tier])`).
#[must_use]
pub fn derive_sona_sign_gauge_40(buyer_seed: &[u8; 32]) -> [f32; SONA_TOTAL_LANES] {
    let mut signs = [1.0_f32; SONA_TOTAL_LANES];
    let mut msg = [0u8; 21 + 32 + 1];
    msg[0..21].copy_from_slice(b"DNAM-WM-SONA-SIGN-V1:");
    msg[21..53].copy_from_slice(buyer_seed);

    let ranges = [(0usize, 16usize), (16, 24), (24, 32), (32, 40)];
    for (tier, &(start, end)) in ranges.iter().enumerate() {
        msg[53] = tier as u8;
        let digest = sha256_digest(&msg);
        for (bit_idx, idx) in (start..end).enumerate() {
            let bit = (digest[bit_idx >> 3] >> (bit_idx & 7)) & 1;
            signs[idx] = if bit == 1 { 1.0 } else { -1.0 };
        }
    }
    signs
}

/// Applies the **`0-ULP` Odd-Symmetric `SONA` Hidden-Lane Sign-Flip Gauge** across all active
/// Matryoshka `SonaGroupWire`, `SonaInterBlockTilePairWire`, `SonaCrossbarBlockWire`,
/// and `VarPro` linear `odd` readout weights present in `arena`
/// (`4,608 / 7,936 / 15,872 / 30,720 B`), commuting losslessly with `strip_container_to_tier(Lite..Ultra)`.
pub fn apply_sona_sign_gauge_watermark(arena: &mut [u8], signs: &[f32]) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
            actual: arena.len(),
        });
    }
    let active_lanes = SONA_SIGN_LANES_PER_TIER[max_tiers - 1];
    if signs.len() < active_lanes {
        return Err(DnamError::BufferTooSmall {
            expected: active_lanes,
            actual: signs.len(),
        });
    }

    let mut s = [1.0_f32; SONA_TOTAL_LANES];
    for (dst, &src) in s.iter_mut().zip(signs.iter()).take(active_lanes) {
        *dst = if src < 0.0 { -1.0 } else { 1.0 };
    }

    let actual_len = arena.len();

    // 1. Arena 0 (`Lite`): groups 0..4, tiles 0..3, Lite crossbar slots, Lite readout odd weights
    {
        let (a0, _) =
            Arena0LiteWire::mut_from_prefix(arena).map_err(|_| DnamError::BufferTooSmall {
                expected: ARENA_LITE_BYTES,
                actual: actual_len,
            })?;
        for g in 0..4 {
            flip_sona_group_wire(&mut a0.sona_core.groups[g], g, &s);
            for l in 0..4 {
                let sign = s[g * 4 + l];
                flip_fp16_wire(
                    &mut a0.readout_lite.stage_latent_mods_fp16[80 + g * 4 + l],
                    sign,
                );
                flip_fp16_wire(&mut a0.readout_lite.w_ro_fp16[192 + g * 4 + l], sign);
            }
        }
        for t in 0..3 {
            flip_sona_tile_wire(&mut a0.sona_core.tiles[t], t, &s);
        }
        flip_ro_odd_lanes(&mut a0.readout_lite.w_ro_fp16[..192], 16, &s);
    }

    // 2. Arena 1 (`Full`): groups 4..6, tiles 3..13, Full crossbar (0..6), Full readout (0..24)
    if max_tiers >= 2 {
        let (a1, _) = Arena1FullDeltaWire::mut_from_prefix(&mut arena[ARENA_LITE_BYTES..])
            .map_err(|_| DnamError::BufferTooSmall {
                expected: ARENA_FULL_BYTES,
                actual: actual_len,
            })?;
        for i in 0..2 {
            flip_sona_group_wire(&mut a1.sona_core_full.groups[i], 4 + i, &s);
        }
        for i in 0..10 {
            flip_sona_tile_wire(&mut a1.sona_core_full.tiles[i], 3 + i, &s);
        }
        flip_crossbar_active_groups(&mut a1.sona_core_full.crossbar, 6, &s);
        flip_ro_odd_lanes(
            &mut a1.varpro_readout_full.w_ro_and_mods_fp16[..288],
            24,
            &s,
        );
    }

    // 3. Arena 2 (`Pro`): groups 6..8, tiles 13..23, Pro crossbar (0..8), Pro readout (0..32)
    if max_tiers >= 3 {
        let (a2, _) =
            Arena2ProDeltaWire::mut_from_prefix(&mut arena[ARENA_FULL_BYTES..]).map_err(|_| {
                DnamError::BufferTooSmall {
                    expected: ARENA_PRO_BYTES,
                    actual: actual_len,
                }
            })?;
        for i in 0..2 {
            flip_sona_group_wire(&mut a2.sona_core_pro.groups[i], 6 + i, &s);
        }
        for i in 0..10 {
            flip_sona_tile_wire(&mut a2.sona_core_pro.tiles[i], 13 + i, &s);
        }
        flip_crossbar_active_groups(&mut a2.sona_core_pro.crossbar_pro, 8, &s);
        flip_ro_odd_lanes(&mut a2.varpro_readout_pro.w_ro_and_mods_fp16[..384], 32, &s);
    }

    // 4. Arena 3 (`Ultra`): groups 8..10, tiles 23..35, Ultra crossbar (0..10), Ultra readout (0..40)
    if max_tiers >= 4 {
        let (a3, _) = Arena3UltraDeltaWire::mut_from_prefix(&mut arena[ARENA_PRO_BYTES..])
            .map_err(|_| DnamError::BufferTooSmall {
                expected: ARENA_ULTRA_BYTES,
                actual: actual_len,
            })?;
        for i in 0..2 {
            flip_sona_group_wire(&mut a3.sona_core_ultra.groups[i], 8 + i, &s);
        }
        for i in 0..12 {
            flip_sona_tile_wire(&mut a3.sona_core_ultra.tiles[i], 23 + i, &s);
        }
        flip_crossbar_active_groups(&mut a3.sona_core_ultra.crossbar_ultra, 10, &s);
        flip_ro_odd_lanes(
            &mut a3.varpro_readout_ultra.w_ro_and_mods_fp16[..480],
            40,
            &s,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{build_dnam_container, strip_container_to_tier};
    use crate::engine::DnamEngine;
    use crate::graph::sona_core::model::SonaMatryoshkaModel;
    use crate::graph::weights::pack_sona_matryoshka_to_arena;
    use crate::math::DefaultMathBackend;
    use crate::provenance::{extract_structural_hash_224d, structural_cosine_similarity};
    use crate::schema::{ControlKind, ControlSchemaDef, ControlUiHints, DnamJsonHeader};
    use crate::types::Tier;
    use proptest::prelude::*;

    fn build_rich_test_sona_arena() -> alloc::vec::Vec<u8> {
        let mut model = SonaMatryoshkaModel::dual_rate_default(false);
        for (g, grp) in model.groups.iter_mut().enumerate() {
            for l in 0..4 {
                let s = if ((g + l) & 1) == 0 { 1.0 } else { -1.0 };
                grp.svf_g[l] = 0.14 + 0.02 * ((g + l) as f32);
                grp.svf_k[l] = 0.65;
                grp.lru_alpha_rel[l] = 0.90;
                grp.lru_alpha_atk[l] = 0.20;
                grp.lru_kappa[l] = 0.35;
                grp.adaa_drive[l] = 0.8 + 0.1 * (l as f32);
                grp.lru_bias_drive[l] = 0.15 * s;
                grp.in_x_weight[l] = (0.35 - 0.02 * (g as f32)) * s;
                grp.tone_lp[l] = 0.45;
                grp.tone_bp[l] = 0.25 * s;
                grp.tone_dir[l] = 0.30;
                grp.sag_depth[l] = 0.18;
                grp.wavefolder_mix[l] = if (l & 1) == 0 { 0.25 } else { 0.0 };
                grp.rect_sign[l] = s;
            }
        }
        for (ti, tile) in model.tiles.iter_mut().enumerate() {
            for col in 0..4 {
                for row in 0..4 {
                    let s = if ((ti + col + row) & 1) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    tile.w_odd_4x4[col][row] = 0.08 * s * (1.0 + 0.1 * (row as f32));
                    tile.w_even_4x4[col][row] = 0.05 * s;
                    tile.w_lru_4x4[col][row] = 0.04 * s;
                }
            }
        }
        model.crossbar.op_p_crossbar_gain = 0.25;
        model.crossbar.op_t_crossbar_gain = 0.15;
        model.crossbar.phasor_freq_hz = [1.5, 3.5, 6.0, 12.0];
        for g in 0..model.groups.len() {
            for l in 0..4 {
                let s = if ((g + l) & 1) == 0 { 1.0 } else { -1.0 };
                model.crossbar.op_p_to_group[g][l] = 0.12 * s;
                model.crossbar.phasor_to_group[g][l] = 0.09 * s;
            }
        }
        model.stage0.clean_sub_mix = 0.18;
        model.stage0.xo_low_gain = 1.05;
        model.stage0.xo_high_gain = 0.95;
        model.ctrl.w_svf_g_oct[0] = 0.25;
        model.ctrl.w_tank_psi[0] = 0.15;
        model.ctrl.w_adaa_drive_scale[0][0] = 0.20;
        for (tier_idx, ro) in model.readouts.iter_mut().enumerate() {
            for ch in 0..2 {
                for l in 0..SONA_TOTAL_LANES {
                    let s = if ((ch + l + tier_idx) & 1) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    ro.pre_cab_w_odd[ch][l] = 0.08 * s * (1.0 + 0.02 * (l as f32));
                    ro.pre_cab_w_even[ch][l] = 0.03 * s;
                    ro.pre_cab_w_lru[ch][l] = 0.02 * s;
                    ro.direct_w_odd[ch][l] = 0.05 * s;
                    ro.direct_w_even[ch][l] = 0.025 * s;
                    ro.direct_w_lru[ch][l] = 0.015 * s;
                }
            }
            for tap in 0..128 {
                ro.fir_l[tap] = if tap % 4 == 0 { 0.08 } else { 0.005 };
                ro.fir_r[tap] = if tap % 4 == 0 { 0.075 } else { 0.004 };
            }
            for m in 0..32 {
                ro.kautz_b_gamma[m] = 0.35;
                ro.kautz_c_l[m] = (0.14, -0.08);
                ro.kautz_c_r[m] = (0.12, -0.06);
            }
        }
        for s4 in &mut model.stage4 {
            s4.g_loop = 0.72;
            s4.wet_mix = 0.22;
        }
        let mut arena = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        pack_sona_matryoshka_to_arena(&model, &mut arena);
        arena
    }

    #[test]
    fn test_sona_sign_gauge_errors_and_prefix_commutativity() {
        let mut short = [0u8; 512];
        assert!(apply_sona_sign_gauge_watermark(&mut short, &[1.0; SONA_TOTAL_LANES]).is_err());
        let mut lite = [0u8; ARENA_LITE_BYTES];
        assert!(apply_sona_sign_gauge_watermark(&mut lite, &[1.0; 8]).is_err());

        let master = build_rich_test_sona_arena();
        let seed_a = [0x2Au8; 32];
        let signs_a = derive_sona_sign_gauge_40(&seed_a);
        let mut ultra_gauged = master.clone();
        apply_sona_sign_gauge_watermark(&mut ultra_gauged, &signs_a).unwrap();

        let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
        let mut hdr = DnamJsonHeader::default_reference();
        hdr.controls_schema.push(ControlSchemaDef {
            id: alloc::string::String::from("gain"),
            name: alloc::string::String::from("Gain"),
            kind: ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(alloc::vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_bounds: alloc::vec![],
            declared_options: None,
            active_options: None,
            represented_switch_options: None,
            clamp_warning: None,
            ui_hints: ControlUiHints::default(),
            calibration: None,
            user_metadata: alloc::collections::BTreeMap::new(),
        });
        let master_container = build_dnam_container(&mut hdr, &master, &master, false).unwrap();
        let gauged_container =
            build_dnam_container(&mut hdr.clone(), &ultra_gauged, &ultra_gauged, false).unwrap();

        let hash_master = extract_structural_hash_224d(&master).unwrap();
        let hash_gauged = extract_structural_hash_224d(&ultra_gauged).unwrap();
        assert!((structural_cosine_similarity(&hash_master, &hash_gauged) - 1.0).abs() < 1e-12);

        for (t_idx, &tier) in tiers.iter().enumerate() {
            let prefix_len = tier.expected_arena_bytes();
            let mut sliced_then_gauged = master[..prefix_len].to_vec();
            let prefix_signs = &signs_a[..SONA_SIGN_LANES_PER_TIER[t_idx]];
            apply_sona_sign_gauge_watermark(&mut sliced_then_gauged, prefix_signs).unwrap();
            assert_eq!(sliced_then_gauged, ultra_gauged[..prefix_len]);

            let stripped_m = strip_container_to_tier(&master_container, tier).unwrap();
            let stripped_g = strip_container_to_tier(&gauged_container, tier).unwrap();
            let mut em =
                DnamEngine::<DefaultMathBackend>::from_bytes(&stripped_m, tier, 48_000).unwrap();
            let mut eg =
                DnamEngine::<DefaultMathBackend>::from_bytes(&stripped_g, tier, 48_000).unwrap();
            em.set_control_value(0, 7.2);
            eg.set_control_value(0, 7.2);

            let in_l: alloc::vec::Vec<f32> =
                (0..128).map(|i| 0.35 * ((i as f32) * 0.13).sin()).collect();
            let in_r: alloc::vec::Vec<f32> =
                (0..128).map(|i| 0.25 * ((i as f32) * 0.09).cos()).collect();
            let (mut ml, mut mr) = (alloc::vec![0.0_f32; 128], alloc::vec![0.0_f32; 128]);
            let (mut gl, mut gr) = (alloc::vec![0.0_f32; 128], alloc::vec![0.0_f32; 128]);
            em.process_block(&in_l, &in_r, &mut ml, &mut mr);
            eg.process_block(&in_l, &in_r, &mut gl, &mut gr);

            for i in 0..128 {
                assert_eq!(
                    ml[i].to_bits(),
                    gl[i].to_bits(),
                    "tier {:?} sample {}: L {} vs {}",
                    tier,
                    i,
                    ml[i],
                    gl[i]
                );
                assert_eq!(
                    mr[i].to_bits(),
                    gr[i].to_bits(),
                    "tier {:?} sample {}: R {} vs {}",
                    tier,
                    i,
                    mr[i],
                    gr[i]
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_sona_sign_gauge_random_keys_preserve_zero_ulp_audio(bits in 0u64..(1u64 << 40)) {
            let mut signs = [1.0_f32; SONA_TOTAL_LANES];
            for (i, slot) in signs.iter_mut().enumerate() {
                if ((bits >> i) & 1) == 1 {
                    *slot = -1.0;
                }
            }
            let master = build_rich_test_sona_arena();
            let mut gauged = master.clone();
            apply_sona_sign_gauge_watermark(&mut gauged, &signs).unwrap();

            let mut hdr = DnamJsonHeader::default_reference();
            let mc = build_dnam_container(&mut hdr, &master, &master, false).unwrap();
            let gc = build_dnam_container(&mut hdr.clone(), &gauged, &gauged, false).unwrap();
            let mut em = DnamEngine::<DefaultMathBackend>::from_bytes(&mc, Tier::Ultra, 48_000).unwrap();
            let mut eg = DnamEngine::<DefaultMathBackend>::from_bytes(&gc, Tier::Ultra, 48_000).unwrap();

            let in_l: alloc::vec::Vec<f32> = (0..64).map(|i| 0.4 * ((i as f32) * 0.17).sin()).collect();
            let in_r: alloc::vec::Vec<f32> = (0..64).map(|i| 0.3 * ((i as f32) * 0.11).cos()).collect();
            let (mut ml, mut mr) = (alloc::vec![0.0_f32; 64], alloc::vec![0.0_f32; 64]);
            let (mut gl, mut gr) = (alloc::vec![0.0_f32; 64], alloc::vec![0.0_f32; 64]);
            em.process_block(&in_l, &in_r, &mut ml, &mut mr);
            eg.process_block(&in_l, &in_r, &mut gl, &mut gr);
            for i in 0..64 {
                prop_assert_eq!(ml[i].to_bits(), gl[i].to_bits());
                prop_assert_eq!(mr[i].to_bits(), gr[i].to_bits());
            }
        }
    }
}
