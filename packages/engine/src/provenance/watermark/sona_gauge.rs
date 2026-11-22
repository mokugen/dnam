//! `0-ULP` Odd-Symmetric `SONA` (`PolyADAA1` / `ParaLRU` / `PHS`) Hidden-Lane Sign-Flip Buyer
//! Watermarking (`ref[sec.watermark.reciprocal-gauge]`).
//!
//! Because odd-symmetric `PolyADAA1`, `PHS` (`i_L^3` is strictly odd so flipping hidden state
//! polarity requires no change to `svf_k_nl_fp16`), and optional `C^2` wavefolder satisfy
//! $g(-x) = -g(x)$ at analytical equilibrium $+0.0$, even-activation `adaa_even` satisfies
//! $g_e(-x) = +g_e(x)$, and decoupled `1D` `ParaLRU` commutes with input polarity conjugation when
//! `rect_sign` (bit 15 of `wavefolder_mix_fp16`) flips, flipping any active Matryoshka lane
//! $i \in \{0,\dots,39\}$ by $s_i \in \{-1, +1\}$ across `in_mid_weight_fp16`,
//! `in_side_weight_fp16`, `lru_bias_drive_fp16`, `wavefolder_mix_fp16` bit-15 `rect_sign`, cyclic
//! `glu_mix_fp16`, crossbar (`op_p_to_core`, `phasor_to_core`), inter-block `4x4` spatial tiles,
//! and `LinearReadout` `odd` weights preserves **bit-exact `0.000000 ULP`** audio parity across
//! `Lite..Ultra`.

use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::{F32, U16};

use super::max_tiers_for_arena_len;
use crate::provenance::sha256::sha256_digest;
use crate::schema::DnamError;
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire,
    Arena3UltraDeltaWire, CoreQuadWire, SonaCrossbarBlockWire, SonaInterBlockTilePairWire,
};
use dnam_vm::graph::SONA_INTERBLOCK_TILES;

const ARENA_LITE_BYTES: usize = ARENA_TIER_BYTES[0];
const ARENA_FULL_BYTES: usize = ARENA_TIER_BYTES[1];
const ARENA_PRO_BYTES: usize = ARENA_TIER_BYTES[2];
const ARENA_ULTRA_BYTES: usize = ARENA_TIER_BYTES[3];

/// Cumulative Matryoshka active `SONA` hidden lanes per tier (`16 / 24 / 32 / 40`).
pub const SONA_SIGN_LANES_PER_TIER: [usize; 4] = [16, 24, 32, 40];
/// Maximum active Matryoshka `SONA` hidden lanes across `CoreQuad`s `0..10` (`40`).
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

/// Unconditionally toggles bit 15 (`rect_sign`) of `wavefolder_mix_fp16` (including
/// `0x0000 <-> 0x8000`) when `sign < 0.0`, preserving zero-magnitude rectification conjugation at
/// `0-ULP` without touching `svf_k_nl_fp16`.
#[inline]
fn flip_wavefolder_rect_sign_bit_fp16(slot: &mut U16, sign: f32) {
    if sign < 0.0 {
        *slot = U16::new(slot.get() ^ SIGN_BIT_FP16);
    }
}

fn flip_core_quad_wire(qw: &mut CoreQuadWire, global_g: usize, s: &[f32; SONA_TOTAL_LANES]) {
    for l in 0..4 {
        let sign = s[global_g * 4 + l];
        let sign_next = s[global_g * 4 + ((l + 1) & 3)];
        flip_fp16_wire(&mut qw.in_mid_weight_fp16[l], sign);
        flip_fp16_wire(&mut qw.in_side_weight_fp16[l], sign);
        flip_fp16_wire(&mut qw.lru_bias_drive_fp16[l], sign);
        flip_wavefolder_rect_sign_bit_fp16(&mut qw.wavefolder_mix_fp16[l], sign);
        flip_fp16_wire(&mut qw.glu_mix_fp16[l], sign * sign_next);
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
            flip_fp16_wire(&mut cb.op_p_to_core_fp16[g][l], sign);
            flip_fp16_wire(&mut cb.phasor_to_core_fp16[g][l], sign);
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
/// Matryoshka [`CoreQuadWire`], [`SonaInterBlockTilePairWire`], [`SonaCrossbarBlockWire`],
/// and `LinearReadout` `odd` weights present in `arena`
/// (`4,608 / 7,936 / 15,872 / 30,720 B`), commuting losslessly with tier truncation (`Lite..Ultra`).
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

    // 1. Arena 0 (`Lite`): core quads 0..4, tiles 0..3, Lite crossbar slots, Lite readout odd weights
    {
        let (a0, _) =
            Arena0LiteWire::mut_from_prefix(arena).map_err(|_| DnamError::BufferTooSmall {
                expected: ARENA_LITE_BYTES,
                actual: actual_len,
            })?;
        for g in 0..4 {
            flip_core_quad_wire(&mut a0.core_quads[g], g, &s);
            for l in 0..4 {
                let sign = s[g * 4 + l];
                flip_fp16_wire(
                    &mut a0.linear_readout_lite.stage_latent_mods_fp16[80 + g * 4 + l],
                    sign,
                );
                flip_fp16_wire(&mut a0.linear_readout_lite.w_ro_fp16[192 + g * 4 + l], sign);
            }
        }
        for t in 0..3 {
            flip_sona_tile_wire(&mut a0.tiles[t], t, &s);
        }
        flip_ro_odd_lanes(&mut a0.linear_readout_lite.w_ro_fp16[..192], 16, &s);
    }

    // 2. Arena 1 (`Full`): core quads 4..6, tiles 3..13, Full crossbar (0..6), Full readout (0..24)
    if max_tiers >= 2 {
        let (a1, _) = Arena1FullDeltaWire::mut_from_prefix(&mut arena[ARENA_LITE_BYTES..])
            .map_err(|_| DnamError::BufferTooSmall {
                expected: ARENA_FULL_BYTES,
                actual: actual_len,
            })?;
        for i in 0..2 {
            flip_core_quad_wire(&mut a1.core_quads[i], 4 + i, &s);
        }
        for i in 0..10 {
            flip_sona_tile_wire(&mut a1.tiles[i], 3 + i, &s);
        }
        flip_crossbar_active_groups(&mut a1.crossbar_full, 6, &s);
        flip_ro_odd_lanes(
            &mut a1.linear_readout_full.w_ro_and_mods_fp16[..288],
            24,
            &s,
        );
    }

    // 3. Arena 2 (`Pro`): core quads 6..8, tiles 13..23, Pro crossbar (0..8), Pro readout (0..32)
    if max_tiers >= 3 {
        let (a2, _) =
            Arena2ProDeltaWire::mut_from_prefix(&mut arena[ARENA_FULL_BYTES..]).map_err(|_| {
                DnamError::BufferTooSmall {
                    expected: ARENA_PRO_BYTES,
                    actual: actual_len,
                }
            })?;
        for i in 0..2 {
            flip_core_quad_wire(&mut a2.core_quads[i], 6 + i, &s);
        }
        for i in 0..10 {
            flip_sona_tile_wire(&mut a2.tiles[i], 13 + i, &s);
        }
        flip_crossbar_active_groups(&mut a2.crossbar_pro, 8, &s);
        flip_ro_odd_lanes(&mut a2.linear_readout_pro.w_ro_and_mods_fp16[..384], 32, &s);
    }

    // 4. Arena 3 (`Ultra`): core quads 8..10, tiles 23..35, Ultra crossbar (0..10), Ultra readout (0..40)
    if max_tiers >= 4 {
        let (a3, _) = Arena3UltraDeltaWire::mut_from_prefix(&mut arena[ARENA_PRO_BYTES..])
            .map_err(|_| DnamError::BufferTooSmall {
                expected: ARENA_ULTRA_BYTES,
                actual: actual_len,
            })?;
        for i in 0..2 {
            flip_core_quad_wire(&mut a3.core_quads[i], 8 + i, &s);
        }
        for i in 0..12 {
            flip_sona_tile_wire(&mut a3.tiles[i], 23 + i, &s);
        }
        flip_crossbar_active_groups(&mut a3.crossbar_ultra, 10, &s);
        flip_ro_odd_lanes(
            &mut a3.linear_readout_ultra.w_ro_and_mods_fp16[..480],
            40,
            &s,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{extract_structural_hash_224d, structural_cosine_similarity};
    use dnam_kernels::f32_to_fp16;
    use dnam_vm::SonaVm;
    use dnam_vm::bin::{
        ControlClampDescriptorWire, DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, KautzSoaBlockWire,
        TransportQuadWire, encode_dnam_bin,
    };
    use dnam_vm::types::Tier;
    use proptest::prelude::*;
    use zerocopy::{FromZeros, IntoBytes};

    fn populate_test_core_quad(qw: &mut CoreQuadWire, g: usize) {
        for l in 0..4 {
            let s = if ((g + l) & 1) == 0 { 1.0 } else { -1.0 };
            qw.svf_g[l] = F32::new(0.14 + 0.02 * ((g + l) as f32));
            qw.svf_k_fp16[l] = U16::new(f32_to_fp16(0.65));
            qw.lru_alpha_rel[l] = F32::new(0.90);
            qw.lru_alpha_atk[l] = F32::new(0.20);
            qw.lru_kappa_fp16[l] = U16::new(f32_to_fp16(0.35));
            qw.adaa_drive_fp16[l] = U16::new(f32_to_fp16(0.8 + 0.1 * (l as f32)));
            qw.lru_bias_drive_fp16[l] = U16::new(f32_to_fp16(0.15 * s));
            qw.tone_lp_fp16[l] = U16::new(f32_to_fp16(0.45));
            qw.tone_bp_fp16[l] = U16::new(f32_to_fp16(0.25 * s));
            qw.tone_dir_fp16[l] = U16::new(f32_to_fp16(0.30));
            qw.sag_depth_fp16[l] = U16::new(f32_to_fp16(0.18));
            let wf = if (l & 1) == 0 { 0.25 } else { 0.0 };
            qw.wavefolder_mix_fp16[l] =
                U16::new(dnam_kernels::pack_wf_mix_and_rect_sign_fp16(wf, s));
            if matches!(g, 0 | 4 | 6 | 8) {
                qw.svf_k_nl_fp16[l] = U16::new(f32_to_fp16(1.2 + 0.15 * (l as f32)));
            }
            qw.in_mid_weight_fp16[l] = U16::new(f32_to_fp16((0.35 - 0.02 * (g as f32)) * s));
            qw.in_side_weight_fp16[l] = U16::new(f32_to_fp16((0.12 - 0.01 * (g as f32)) * s));
            qw.glu_mix_fp16[l] =
                U16::new(f32_to_fp16(if (l & 1) == 0 { 0.35 * s } else { -0.25 * s }));
        }
    }

    fn populate_test_tile(tw: &mut SonaInterBlockTilePairWire, ti: usize) {
        for col in 0..4 {
            for row in 0..4 {
                let s = if ((ti + col + row) & 1) == 0 {
                    1.0
                } else {
                    -1.0
                };
                tw.w_odd_4x4[col][row] = F32::new(0.08 * s * (1.0 + 0.1 * (row as f32)));
                tw.w_even_4x4_fp16[col][row] = U16::new(f32_to_fp16(0.05 * s));
                tw.w_lru_4x4_fp16[col][row] = U16::new(f32_to_fp16(0.04 * s));
            }
        }
    }

    fn populate_test_ro_lanes(w_ro: &mut [U16], lane_count: usize, tier_idx: usize) {
        for l in 0..lane_count {
            let s = if ((l + tier_idx) & 1) == 0 { 1.0 } else { -1.0 };
            let base = l * 12;
            w_ro[base] = U16::new(f32_to_fp16(0.08 * s * (1.0 + 0.02 * (l as f32))));
            w_ro[base + 1] = U16::new(f32_to_fp16(0.05 * s));
            w_ro[base + 2] = U16::new(f32_to_fp16(0.03 * s));
            w_ro[base + 3] = U16::new(f32_to_fp16(0.02 * s));
            w_ro[base + 6] = U16::new(f32_to_fp16(0.075 * s));
            w_ro[base + 7] = U16::new(f32_to_fp16(0.045 * s));
            w_ro[base + 8] = U16::new(f32_to_fp16(0.025 * s));
            w_ro[base + 9] = U16::new(f32_to_fp16(0.015 * s));
        }
    }

    fn populate_test_kautz(blk: &mut KautzSoaBlockWire) {
        for l in 0..4 {
            blk.r_low[l] = F32::new(0.90);
            blk.dr[l] = F32::new(0.02);
            blk.theta[l] = F32::new(0.25 + 0.1 * (l as f32));
            blk.b_gamma[l] = F32::new(0.35);
            blk.c_re_l[l] = F32::new(0.14);
            blk.c_im_l[l] = F32::new(-0.08);
            blk.c_re_r[l] = F32::new(0.12);
            blk.c_im_r[l] = F32::new(-0.06);
        }
    }

    fn populate_test_transport(tq: &mut TransportQuadWire) {
        for l in 0..4 {
            tq.delay_samples[l] = F32::new(128.0 + 32.0 * (l as f32));
            tq.input_gains_fp16[l] = U16::new(f32_to_fp16(0.5));
            tq.wet_tap_l_fp16[l] = U16::new(f32_to_fp16(0.3));
            tq.wet_tap_r_fp16[l] = U16::new(f32_to_fp16(0.25));
        }
        tq.fb_psi_fp16 = U16::new(f32_to_fp16(0.2));
        tq.sigma_knee_fp16 = U16::new(f32_to_fp16(0.9));
    }

    fn build_rich_test_sona_arena() -> alloc::vec::Vec<u8> {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        a0.crossbar.crossbar_op_gains_fp16 =
            [U16::new(f32_to_fp16(0.25)), U16::new(f32_to_fp16(0.15))];
        a0.crossbar.phasor_freq_hz = [F32::new(1.5), F32::new(3.5), F32::new(6.0), F32::new(12.0)];
        for g in 0..4 {
            populate_test_core_quad(&mut a0.core_quads[g], g);
            for l in 0..4 {
                let s = if ((g + l) & 1) == 0 { 1.0 } else { -1.0 };
                a0.linear_readout_lite.stage_latent_mods_fp16[80 + g * 4 + l] =
                    U16::new(f32_to_fp16(0.12 * s));
                a0.linear_readout_lite.w_ro_fp16[192 + g * 4 + l] = U16::new(f32_to_fp16(0.09 * s));
            }
        }
        for t in 0..3 {
            populate_test_tile(&mut a0.tiles[t], t);
        }
        for tap in 0..32 {
            let v = if tap % 4 == 0 { 0.08 } else { 0.005 };
            a0.conv_fir_lite.fir_l[tap] = F32::new(v);
            a0.conv_fir_lite.fir_r[tap] = F32::new(v * 0.9);
        }
        for blk in &mut a0.conv_kautz_lite {
            populate_test_kautz(blk);
        }
        populate_test_transport(&mut a0.transport_lite);
        populate_test_ro_lanes(&mut a0.linear_readout_lite.w_ro_fp16[..192], 16, 0);

        let mut a1 = Arena1FullDeltaWire::new_zeroed();
        for i in 0..2 {
            populate_test_core_quad(&mut a1.core_quads[i], 4 + i);
        }
        for i in 0..10 {
            populate_test_tile(&mut a1.tiles[i], 3 + i);
        }
        for g in 0..6 {
            for l in 0..4 {
                let s = if ((g + l) & 1) == 0 { 1.0 } else { -1.0 };
                a1.crossbar_full.op_p_to_core_fp16[g][l] = U16::new(f32_to_fp16(0.12 * s));
                a1.crossbar_full.phasor_to_core_fp16[g][l] = U16::new(f32_to_fp16(0.09 * s));
            }
        }
        populate_test_kautz(&mut a1.conv_kautz_full);
        populate_test_transport(&mut a1.transport_full);
        populate_test_ro_lanes(&mut a1.linear_readout_full.w_ro_and_mods_fp16[..288], 24, 1);

        let mut a2 = Arena2ProDeltaWire::new_zeroed();
        for i in 0..2 {
            populate_test_core_quad(&mut a2.core_quads[i], 6 + i);
        }
        for i in 0..10 {
            populate_test_tile(&mut a2.tiles[i], 13 + i);
        }
        for g in 0..8 {
            for l in 0..4 {
                let s = if ((g + l) & 1) == 0 { 1.0 } else { -1.0 };
                a2.crossbar_pro.op_p_to_core_fp16[g][l] = U16::new(f32_to_fp16(0.11 * s));
                a2.crossbar_pro.phasor_to_core_fp16[g][l] = U16::new(f32_to_fp16(0.08 * s));
            }
        }
        for blk in &mut a2.conv_kautz_pro {
            populate_test_kautz(blk);
        }
        populate_test_transport(&mut a2.transport_pro[0]);
        populate_test_ro_lanes(&mut a2.linear_readout_pro.w_ro_and_mods_fp16[..384], 32, 2);

        let mut a3 = Arena3UltraDeltaWire::new_zeroed();
        for i in 0..2 {
            populate_test_core_quad(&mut a3.core_quads[i], 8 + i);
        }
        for i in 0..12 {
            populate_test_tile(&mut a3.tiles[i], 23 + i);
        }
        for g in 0..10 {
            for l in 0..4 {
                let s = if ((g + l) & 1) == 0 { 1.0 } else { -1.0 };
                a3.crossbar_ultra.op_p_to_core_fp16[g][l] = U16::new(f32_to_fp16(0.10 * s));
                a3.crossbar_ultra.phasor_to_core_fp16[g][l] = U16::new(f32_to_fp16(0.07 * s));
            }
        }
        for blk in &mut a3.conv_kautz_ultra {
            populate_test_kautz(blk);
        }
        populate_test_transport(&mut a3.transport_ultra[0]);
        populate_test_ro_lanes(
            &mut a3.linear_readout_ultra.w_ro_and_mods_fp16[..480],
            40,
            3,
        );

        let mut out = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        out[0..ARENA_LITE_BYTES].copy_from_slice(a0.as_bytes());
        out[ARENA_LITE_BYTES..ARENA_FULL_BYTES].copy_from_slice(a1.as_bytes());
        out[ARENA_FULL_BYTES..ARENA_PRO_BYTES].copy_from_slice(a2.as_bytes());
        out[ARENA_PRO_BYTES..ARENA_ULTRA_BYTES].copy_from_slice(a3.as_bytes());
        out
    }

    fn encode_dnam_bin_for_tier(
        arena: &[u8],
        tier: Tier,
        with_control: bool,
    ) -> alloc::vec::Vec<u8> {
        let mut a0 = *Arena0LiteWire::ref_from_prefix(arena).unwrap().0;
        let tier_mask = (1_u16 << (tier.index() + 1)) - 1;
        a0.crossbar.cell_desc.tier_flags = U16::new(tier_mask);
        let a1 = (tier >= Tier::Full).then(|| {
            Arena1FullDeltaWire::ref_from_prefix(&arena[ARENA_LITE_BYTES..])
                .unwrap()
                .0
        });
        let a2 = (tier >= Tier::Pro).then(|| {
            Arena2ProDeltaWire::ref_from_prefix(&arena[ARENA_FULL_BYTES..])
                .unwrap()
                .0
        });
        let a3 = (tier >= Tier::Ultra).then(|| {
            Arena3UltraDeltaWire::ref_from_prefix(&arena[ARENA_PRO_BYTES..])
                .unwrap()
                .0
        });

        let mut ctrl = ControlClampDescriptorWire::new_zeroed();
        ctrl.ctrl_kind = 0;
        ctrl.lobatto_degree = 2;
        ctrl.default_val = F32::new(5.0);
        ctrl.safe_min_base = F32::new(0.0);
        ctrl.safe_max_base = F32::new(10.0);
        ctrl.scale_norm = F32::new(0.2);
        ctrl.offset_norm = F32::new(-1.0);
        ctrl.anchor_nodes_norm = [F32::new(0.0), F32::new(-1.0), F32::new(1.0)];
        let ctrls = if with_control {
            alloc::vec![ctrl]
        } else {
            alloc::vec::Vec::new()
        };

        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: dnam_vm::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 250.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 64,
            content_hash_prefix: [1, 2, 3, 4, 5, 6, 7, 8],
            controls: &ctrls,
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: a1,
            arena2: a2,
            arena3: a3,
        };
        encode_dnam_bin(&input).unwrap()
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
        let hash_master = extract_structural_hash_224d(&master).unwrap();
        let hash_gauged = extract_structural_hash_224d(&ultra_gauged).unwrap();
        assert!((structural_cosine_similarity(&hash_master, &hash_gauged) - 1.0).abs() < 1e-12);

        for (t_idx, &tier) in tiers.iter().enumerate() {
            let prefix_len = ARENA_TIER_BYTES[t_idx];
            let mut sliced_then_gauged = master[..prefix_len].to_vec();
            let prefix_signs = &signs_a[..SONA_SIGN_LANES_PER_TIER[t_idx]];
            apply_sona_sign_gauge_watermark(&mut sliced_then_gauged, prefix_signs).unwrap();
            assert_eq!(sliced_then_gauged, ultra_gauged[..prefix_len]);

            let bin_m = encode_dnam_bin_for_tier(&master, tier, true);
            let bin_g = encode_dnam_bin_for_tier(&ultra_gauged, tier, true);
            let mut vm_m =
                SonaVm::<dnam_vm::math::ScalarBackend>::open_dnam_bin_owned(&bin_m, tier, 48_000.0)
                    .unwrap();
            let mut vm_g =
                SonaVm::<dnam_vm::math::ScalarBackend>::open_dnam_bin_owned(&bin_g, tier, 48_000.0)
                    .unwrap();
            vm_m.set_control_physical(0, 7.2).unwrap();
            vm_g.set_control_physical(0, 7.2).unwrap();

            let in_l: alloc::vec::Vec<f32> =
                (0..128).map(|i| 0.35 * ((i as f32) * 0.13).sin()).collect();
            let in_r: alloc::vec::Vec<f32> =
                (0..128).map(|i| 0.25 * ((i as f32) * 0.09).cos()).collect();
            let (mut ml, mut mr) = (alloc::vec![0.0_f32; 128], alloc::vec![0.0_f32; 128]);
            let (mut gl, mut gr) = (alloc::vec![0.0_f32; 128], alloc::vec![0.0_f32; 128]);
            vm_m.process_block(&in_l, &in_r, &mut ml, &mut mr).unwrap();
            vm_g.process_block(&in_l, &in_r, &mut gl, &mut gr).unwrap();

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

            let bin_m = encode_dnam_bin_for_tier(&master, Tier::Ultra, false);
            let bin_g = encode_dnam_bin_for_tier(&gauged, Tier::Ultra, false);
            let mut vm_m = SonaVm::<dnam_vm::math::ScalarBackend>::open_dnam_bin_owned(
                &bin_m,
                Tier::Ultra,
                48_000.0,
            )
            .unwrap();
            let mut vm_g = SonaVm::<dnam_vm::math::ScalarBackend>::open_dnam_bin_owned(
                &bin_g,
                Tier::Ultra,
                48_000.0,
            )
            .unwrap();

            let in_l: alloc::vec::Vec<f32> = (0..64).map(|i| 0.4 * ((i as f32) * 0.17).sin()).collect();
            let in_r: alloc::vec::Vec<f32> = (0..64).map(|i| 0.3 * ((i as f32) * 0.11).cos()).collect();
            let (mut ml, mut mr) = (alloc::vec![0.0_f32; 64], alloc::vec![0.0_f32; 64]);
            let (mut gl, mut gr) = (alloc::vec![0.0_f32; 64], alloc::vec![0.0_f32; 64]);
            vm_m.process_block(&in_l, &in_r, &mut ml, &mut mr).unwrap();
            vm_g.process_block(&in_l, &in_r, &mut gl, &mut gr).unwrap();
            for i in 0..64 {
                prop_assert_eq!(ml[i].to_bits(), gl[i].to_bits());
                prop_assert_eq!(mr[i].to_bits(), gr[i].to_bits());
            }
        }
    }

    fn populate_zero_rect_core_quad(qw: &mut CoreQuadWire, g: usize) {
        for l in 0..4 {
            qw.svf_g[l] = F32::new(0.16 + 0.03 * (l as f32));
            qw.svf_k_fp16[l] = U16::new(f32_to_fp16(0.7));
            qw.lru_alpha_rel[l] = F32::new(0.92);
            qw.lru_alpha_atk[l] = F32::new(0.25);
            qw.lru_kappa_fp16[l] = U16::new(f32_to_fp16(0.4));
            qw.lru_duhem_gamma_fp16[l] = U16::new(f32_to_fp16(2.5));
            qw.lru_signed_mix_fp16[l] = U16::new(f32_to_fp16(0.5));
            qw.in_mid_weight_fp16[l] = U16::new(f32_to_fp16(0.35 - 0.02 * (g as f32)));
            qw.adaa_drive_fp16[l] = U16::new(f32_to_fp16(1.2));
            qw.sag_depth_fp16[l] = U16::new(f32_to_fp16(0.3));
            qw.tone_dir_fp16[l] = U16::new(f32_to_fp16(0.6));
            if matches!(g, 0 | 4 | 6 | 8) {
                qw.svf_k_nl_fp16[l] = U16::new(f32_to_fp16(1.5));
            }
            assert_eq!(qw.wavefolder_mix_fp16[l].get(), 0x0000);
        }
    }

    #[test]
    fn test_sona_sign_gauge_zero_rect_sign_preserves_zero_ulp_across_tiers_and_double_flip() {
        for bits in [0x0000_u16, 0x8000, 0x3400, 0xB400] {
            let mut slot = U16::new(bits);
            flip_wavefolder_rect_sign_bit_fp16(&mut slot, -1.0);
            assert_eq!(slot.get(), bits ^ SIGN_BIT_FP16);
            flip_wavefolder_rect_sign_bit_fp16(&mut slot, -1.0);
            assert_eq!(slot.get(), bits);
            flip_wavefolder_rect_sign_bit_fp16(&mut slot, -1.0);
            assert_eq!(slot.get(), bits ^ SIGN_BIT_FP16);
        }

        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        for g in 0..4 {
            populate_zero_rect_core_quad(&mut a0.core_quads[g], g);
        }
        populate_test_ro_lanes(&mut a0.linear_readout_lite.w_ro_fp16[..192], 16, 0);

        let mut a1 = Arena1FullDeltaWire::new_zeroed();
        for i in 0..2 {
            populate_zero_rect_core_quad(&mut a1.core_quads[i], 4 + i);
        }
        populate_test_ro_lanes(&mut a1.linear_readout_full.w_ro_and_mods_fp16[..288], 24, 1);

        let mut a2 = Arena2ProDeltaWire::new_zeroed();
        for i in 0..2 {
            populate_zero_rect_core_quad(&mut a2.core_quads[i], 6 + i);
        }
        populate_test_ro_lanes(&mut a2.linear_readout_pro.w_ro_and_mods_fp16[..384], 32, 2);

        let mut a3 = Arena3UltraDeltaWire::new_zeroed();
        for i in 0..2 {
            populate_zero_rect_core_quad(&mut a3.core_quads[i], 8 + i);
        }
        populate_test_ro_lanes(
            &mut a3.linear_readout_ultra.w_ro_and_mods_fp16[..480],
            40,
            3,
        );

        let mut master = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        master[0..ARENA_LITE_BYTES].copy_from_slice(a0.as_bytes());
        master[ARENA_LITE_BYTES..ARENA_FULL_BYTES].copy_from_slice(a1.as_bytes());
        master[ARENA_FULL_BYTES..ARENA_PRO_BYTES].copy_from_slice(a2.as_bytes());
        master[ARENA_PRO_BYTES..ARENA_ULTRA_BYTES].copy_from_slice(a3.as_bytes());

        let signs = derive_sona_sign_gauge_40(&[0x5Cu8; 32]);
        assert!(signs.iter().any(|&s| s < 0.0));

        let mut flip1 = master.clone();
        apply_sona_sign_gauge_watermark(&mut flip1, &signs).unwrap();
        let mut flip2 = flip1.clone();
        apply_sona_sign_gauge_watermark(&mut flip2, &signs).unwrap();

        let (f1_a0, _) = Arena0LiteWire::ref_from_prefix(&flip1).unwrap();
        let (f2_a0, _) = Arena0LiteWire::ref_from_prefix(&flip2).unwrap();
        for l in 0..4 {
            let expected1 = if signs[l] < 0.0 {
                SIGN_BIT_FP16
            } else {
                0x0000
            };
            assert_eq!(f1_a0.core_quads[0].wavefolder_mix_fp16[l].get(), expected1);
            assert_eq!(f2_a0.core_quads[0].wavefolder_mix_fp16[l].get(), 0x0000);
            assert_eq!(f1_a0.core_quads[0].svf_k_nl_fp16[l].get(), f32_to_fp16(1.5));
            assert_eq!(f2_a0.core_quads[0].svf_k_nl_fp16[l].get(), f32_to_fp16(1.5));
        }

        let in_l: alloc::vec::Vec<f32> =
            (0..64).map(|i| 0.35 * ((i as f32) * 0.13).sin()).collect();
        let in_r: alloc::vec::Vec<f32> =
            (0..64).map(|i| 0.25 * ((i as f32) * 0.09).cos()).collect();
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let bin_m = encode_dnam_bin_for_tier(&master, tier, false);
            let bin_1 = encode_dnam_bin_for_tier(&flip1, tier, false);
            let bin_2 = encode_dnam_bin_for_tier(&flip2, tier, false);
            let mut vm_m =
                SonaVm::<dnam_vm::math::ScalarBackend>::open_dnam_bin_owned(&bin_m, tier, 48_000.0)
                    .unwrap();
            let mut vm_1 =
                SonaVm::<dnam_vm::math::ScalarBackend>::open_dnam_bin_owned(&bin_1, tier, 48_000.0)
                    .unwrap();
            let mut vm_2 =
                SonaVm::<dnam_vm::math::ScalarBackend>::open_dnam_bin_owned(&bin_2, tier, 48_000.0)
                    .unwrap();
            let (mut ml, mut mr) = (alloc::vec![0.0_f32; 64], alloc::vec![0.0_f32; 64]);
            let (mut l1, mut r1) = (alloc::vec![0.0_f32; 64], alloc::vec![0.0_f32; 64]);
            let (mut l2, mut r2) = (alloc::vec![0.0_f32; 64], alloc::vec![0.0_f32; 64]);
            vm_m.process_block(&in_l, &in_r, &mut ml, &mut mr).unwrap();
            vm_1.process_block(&in_l, &in_r, &mut l1, &mut r1).unwrap();
            vm_2.process_block(&in_l, &in_r, &mut l2, &mut r2).unwrap();
            for i in 0..64 {
                assert_eq!(ml[i].to_bits(), l1[i].to_bits());
                assert_eq!(mr[i].to_bits(), r1[i].to_bits());
                assert_eq!(ml[i].to_bits(), l2[i].to_bits());
                assert_eq!(mr[i].to_bits(), r2[i].to_bits());
            }
        }
    }
}
