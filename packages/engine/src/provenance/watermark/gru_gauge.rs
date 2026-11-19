//! `0-ULP` Odd-Activation `GRU` Hidden-Lane Sign-Flip Buyer Watermarking
//! (`ref[sec.watermark.reciprocal-gauge]`).
//!
//! Because Stage II candidate activations use odd `NR2` Padé $\tanh(-x) = -\tanh(x)$ with
//! $k_{\text{quad}} a_{\text{sig}}^2$ carrying even square-law curvature (so flipping both
//! `a_sig` and `k_quad` yields $-(a_{\text{sig}} + k_{\text{quad}} a_{\text{sig}}^2)$) and
//! `z`/`r` gates left invariant, flipping active hidden lane $(b, g, l)$ by $s_{b,g,l} \in \{-1, +1\}$
//! yields **bit-exact `0-ULP`** audio equivalence while surviving `strip_container_to_tier(Lite..Ultra)`.

use super::{max_tiers_for_arena_len, read_f32_le, write_f32_le};
use crate::provenance::sha256::sha256_digest;
use crate::schema::DnamError;
use crate::stages::stage2_gru::film::{MAX_CTRL_BLOCK_LANES, NUM_CTRL_BLOCKS, NUM_V3_MACRO_KNOBS};
use crate::stages::stage2_gru::{NUM_GRU_BLOCKS, NUM_IN_FEATS_B0, tier_block_groups};
use crate::stages::weights::{
    ARENA_LITE_BYTES, ARENA0_DIRECT_READOUT_OFFSET, ARENA0_STAGE2_GRU_OFFSET,
    ARENA0_TONE_SVF_PARAMS_OFFSET, ARENA1_CTRL_PARAMS_BYTES, ARENA1_CTRL_PARAMS_OFFSET,
    ARENA1_DIRECT_READOUT_OFFSET, ARENA1_STAGE2_GRU_OFFSET, ARENA2_DIRECT_READOUT_OFFSET,
    ARENA2_STAGE2_GRU_OFFSET, ARENA3_DIRECT_READOUT_OFFSET, ARENA3_STAGE2_GRU_OFFSET,
    GRU_FEED_BYTES, GRU_TILE_TRIPLE_BYTES,
};
use crate::types::Tier;

/// Cumulative Matryoshka active GRU hidden lanes per tier (`12 / 16 / 20 / 24`).
pub const GRU_SIGN_LANES_PER_TIER: [usize; 4] = [12, 16, 20, 24];
/// Maximum active Matryoshka GRU hidden lanes across Blocks `0..3` (`[3, 2, 1]` groups × 4 = `24`).
pub const GRU_TOTAL_LANES: usize = 24;

const _: () = assert!(3 * NUM_IN_FEATS_B0 * 16 + 80 == GRU_FEED_BYTES);
const W_FILM_BN_OFFSET: usize = ARENA1_CTRL_PARAMS_OFFSET
    + 256
    + NUM_V3_MACRO_KNOBS * NUM_CTRL_BLOCKS * MAX_CTRL_BLOCK_LANES * 4;
const _: () = assert!(
    W_FILM_BN_OFFSET + NUM_V3_MACRO_KNOBS * NUM_CTRL_BLOCKS * MAX_CTRL_BLOCK_LANES * 4
        == ARENA1_CTRL_PARAMS_OFFSET + ARENA1_CTRL_PARAMS_BYTES
);

#[inline]
const fn matryoshka_lane_idx(blk: usize, grp: usize, lane: usize) -> Option<usize> {
    let base = match (blk, grp) {
        (0, 0) => 0,
        (1, 0) => 4,
        (2, 0) => 8,
        (0, 1) => 12,
        (1, 1) => 16,
        (0, 2) => 20,
        _ => return None,
    };
    Some(base + (lane & 3))
}

#[inline]
fn flip_f32_if_neg(buf: &mut [u8], off: usize, sign: f32) {
    if sign < 0.0 {
        let v = read_f32_le(buf, off);
        if v != 0.0 {
            write_f32_le(buf, off, -v);
        }
    }
}

fn flip_gru_group_feed(buf: &mut [u8], base: usize, dst_blk: usize, dst_g: usize, s: &[f32; 24]) {
    // `w_in_n` sits at `+160..+240` (`NUM_IN_FEATS_B0=5` × 4 lanes), `b_n` at `+272..+288`,
    // and `k_quad` at `+304..+320` (because `a_sig + k_quad * a_sig^2` negates iff `a_sig` and `k_quad` both flip).
    // Leave `w_in_z`, `w_in_r`, `b_z`, `b_r`, and `k_sag` untouched.
    for l in 0..4 {
        let sign = matryoshka_lane_idx(dst_blk, dst_g, l).map_or(1.0, |i| s[i]);
        if sign < 0.0 {
            for f in 0..NUM_IN_FEATS_B0 {
                flip_f32_if_neg(buf, base + 160 + (f * 4 + l) * 4, sign);
            }
            flip_f32_if_neg(buf, base + 272 + l * 4, sign);
            flip_f32_if_neg(buf, base + 304 + l * 4, sign);
        }
    }
}

fn flip_gru_tile_triple(
    buf: &mut [u8],
    base: usize,
    dst_blk: usize,
    dst_g: usize,
    src_blk: usize,
    src_g: usize,
    s: &[f32; 24],
) {
    for row in 0..4 {
        let s_dst = matryoshka_lane_idx(dst_blk, dst_g, row).map_or(1.0, |i| s[i]);
        for col in 0..4 {
            let s_src = matryoshka_lane_idx(src_blk, src_g, col).map_or(1.0, |i| s[i]);
            let cell = (row * 4 + col) * 4;
            // Gates `z` (`+0`) and `r` (`+64`) stay invariant by canceling `s_src`;
            // candidate `n` (`+128`) carries `s_dst * s_src`.
            flip_f32_if_neg(buf, base + cell, s_src);
            flip_f32_if_neg(buf, base + 64 + cell, s_src);
            flip_f32_if_neg(buf, base + 128 + cell, s_dst * s_src);
        }
    }
}

/// Deterministically derives a prefix-independent 24-lane buyer sign-flip key
/// (`±1.0_f32` per Matryoshka lane, `SHA-256("DNAM-WM-GRU-SIGN-V1:" || buyer_seed || [tier])`).
#[must_use]
pub fn derive_gru_sign_gauge_24(buyer_seed: &[u8; 32]) -> [f32; GRU_TOTAL_LANES] {
    let mut signs = [1.0_f32; GRU_TOTAL_LANES];
    let mut msg = [0u8; 20 + 32 + 1];
    msg[0..20].copy_from_slice(b"DNAM-WM-GRU-SIGN-V1:");
    msg[20..52].copy_from_slice(buyer_seed);

    let ranges = [(0usize, 12usize), (12, 16), (16, 20), (20, 24)];
    for (tier, &(start, end)) in ranges.iter().enumerate() {
        msg[52] = tier as u8;
        let digest = sha256_digest(&msg);
        for (bit_idx, idx) in (start..end).enumerate() {
            let bit = (digest[bit_idx >> 3] >> (bit_idx & 7)) & 1;
            signs[idx] = if bit == 1 { 1.0 } else { -1.0 };
        }
    }
    signs
}

/// Applies the **`0-ULP` Odd-Activation GRU Hidden-Lane Sign-Flip Gauge** across all active
/// Matryoshka Stage II GRU, Tone SVF, ControlBus `w_film_bn`, and direct readout weights present
/// in `arena` (`4,608 / 10,240 / 18,432 / 28,672 B`), commuting losslessly with
/// `strip_container_to_tier(Lite..Ultra)`.
pub fn apply_gru_sign_gauge_watermark(arena: &mut [u8], signs: &[f32]) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
            actual: arena.len(),
        });
    }
    let active_lanes = GRU_SIGN_LANES_PER_TIER[max_tiers - 1];
    if signs.len() < active_lanes {
        return Err(DnamError::BufferTooSmall {
            expected: active_lanes,
            actual: signs.len(),
        });
    }

    let mut s = [1.0_f32; GRU_TOTAL_LANES];
    for (dst, &src) in s.iter_mut().zip(signs.iter()).take(active_lanes) {
        *dst = if src < 0.0 { -1.0 } else { 1.0 };
    }

    // 1. Arena 0: 1x calibration [1, 1, 1] GRU (`Lite`, always present)
    let mut off = ARENA0_STAGE2_GRU_OFFSET;
    flip_gru_group_feed(arena, off, 0, 0, &s);
    off += GRU_FEED_BYTES;
    flip_gru_tile_triple(arena, off, 0, 0, 0, 0, &s);
    off += GRU_TILE_TRIPLE_BYTES;
    for b in 1..NUM_GRU_BLOCKS {
        flip_gru_group_feed(arena, off, b, 0, &s);
        off += GRU_FEED_BYTES;
        flip_gru_tile_triple(arena, off, b, 0, b - 1, 0, &s);
        off += GRU_TILE_TRIPLE_BYTES;
        flip_gru_tile_triple(arena, off, b, 0, b, 0, &s);
        off += GRU_TILE_TRIPLE_BYTES;
    }

    // Inter-Block Tone SVF projection `h0_proj_w[0..4]` sits at `ARENA0_TONE_SVF_PARAMS_OFFSET + 28`.
    for l in 0..4 {
        let sign = matryoshka_lane_idx(0, 0, l).map_or(1.0, |i| s[i]);
        flip_f32_if_neg(arena, ARENA0_TONE_SVF_PARAMS_OFFSET + 28 + l * 4, sign);
    }

    // 2. Arena 1 (`Full`): 1x Block 0 group 1 + ControlBus `w_film_bn`
    if max_tiers >= 2 {
        let mut f_off = ARENA1_STAGE2_GRU_OFFSET;
        flip_gru_group_feed(arena, f_off, 0, 1, &s);
        f_off += GRU_FEED_BYTES;
        for src_g in 0..2 {
            flip_gru_tile_triple(arena, f_off, 0, 1, 0, src_g, &s);
            f_off += GRU_TILE_TRIPLE_BYTES;
        }
        for k in 0..NUM_V3_MACRO_KNOBS {
            for b in 0..NUM_CTRL_BLOCKS {
                for c in 0..MAX_CTRL_BLOCK_LANES {
                    if let Some(lane_idx) = matryoshka_lane_idx(b, c >> 2, c & 3)
                        && lane_idx < 16
                    {
                        let w_off = W_FILM_BN_OFFSET + ((k * NUM_CTRL_BLOCKS + b) * 12 + c) * 4;
                        flip_f32_if_neg(arena, w_off, s[lane_idx]);
                    }
                }
            }
        }
    }

    // 3. Arena 2 (`Pro`): self-contained 2x [2, 2, 1] GRU + Pro `w_film_bn` tail
    if max_tiers >= 3 {
        let mut p_off = ARENA2_STAGE2_GRU_OFFSET;
        let pro_bg = [2usize, 2, 1];
        for (b, &g_cnt) in pro_bg.iter().enumerate() {
            for g in 0..g_cnt {
                flip_gru_group_feed(arena, p_off, b, g, &s);
                p_off += GRU_FEED_BYTES;
                if b > 0 {
                    for src_g in 0..=g {
                        flip_gru_tile_triple(arena, p_off, b, g, b - 1, src_g, &s);
                        p_off += GRU_TILE_TRIPLE_BYTES;
                    }
                }
                for src_g in 0..=g {
                    flip_gru_tile_triple(arena, p_off, b, g, b, src_g, &s);
                    p_off += GRU_TILE_TRIPLE_BYTES;
                }
            }
        }
        for k in 0..NUM_V3_MACRO_KNOBS {
            for l in 0..4 {
                let w_off = ARENA2_STAGE2_GRU_OFFSET + 3_712 + ((k << 2) | l) * 4;
                flip_f32_if_neg(arena, w_off, s[16 + l]);
            }
        }
    }

    // 4. Arena 3 (`Ultra`): 2x Block 0 group 2 + Ultra `w_film_bn` tail
    if max_tiers >= 4 {
        let mut u_off = ARENA3_STAGE2_GRU_OFFSET;
        flip_gru_group_feed(arena, u_off, 0, 2, &s);
        u_off += GRU_FEED_BYTES;
        for src_g in 0..3 {
            flip_gru_tile_triple(arena, u_off, 0, 2, 0, src_g, &s);
            u_off += GRU_TILE_TRIPLE_BYTES;
        }
        for k in 0..NUM_V3_MACRO_KNOBS {
            for l in 0..4 {
                let w_off = ARENA3_STAGE2_GRU_OFFSET + 896 + ((k << 2) | l) * 4;
                flip_f32_if_neg(arena, w_off, s[20 + l]);
            }
        }
    }

    // 5. Direct Readout (`pre_cab_w` and `direct_s2_w`) for each tier present
    let readout_bases = [
        ARENA0_DIRECT_READOUT_OFFSET,
        ARENA1_DIRECT_READOUT_OFFSET,
        ARENA2_DIRECT_READOUT_OFFSET,
        ARENA3_DIRECT_READOUT_OFFSET,
    ];
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    for (&tier, &ro_base) in tiers.iter().zip(readout_bases.iter()).take(max_tiers) {
        let mut w_off = ro_base + 96;
        for _kind in 0..2 {
            for _ch in 0..2 {
                for b in 0..NUM_GRU_BLOCKS {
                    for g in 0..tier_block_groups(tier, b) {
                        for l in 0..4 {
                            let sign = matryoshka_lane_idx(b, g, l).map_or(1.0, |i| s[i]);
                            flip_f32_if_neg(arena, w_off, sign);
                            w_off += 4;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{build_dnam_container, strip_container_to_tier};
    use crate::engine::DnamEngine;
    use crate::math::DefaultMathBackend;
    use crate::schema::DnamJsonHeader;
    use crate::stages::stage2_gru::{V3ControlBusParams, V3InterBlockToneSvfParams};
    use crate::stages::stage3_cab::{V3MatryoshkaModel, V3TierReadout};
    use crate::stages::weights::{ARENA_ULTRA_BYTES, pack_v3_matryoshka_to_arena};
    use proptest::prelude::*;

    fn build_rich_test_v3_arena() -> alloc::vec::Vec<u8> {
        let mut model = V3MatryoshkaModel::dual_rate_default(false);
        for b in 0..NUM_GRU_BLOCKS {
            let blk = &mut model.blocks[b];
            for f in 0..NUM_IN_FEATS_B0 {
                for l in 0..12 {
                    let s = if ((b + f + l) & 1) == 0 { 1.0 } else { -1.0 };
                    blk.w_in_z[f][l] = 0.08 * s;
                    blk.w_in_r[f][l] = 0.06 * s;
                    blk.w_in_n[f][l] = 0.14 * s * (1.0 + 0.03 * (l as f32));
                }
            }
            for l in 0..12 {
                blk.b_z[l] = -0.25;
                blk.b_r[l] = 0.15;
                blk.b_n[l] = 0.07 * if (l & 1) == 0 { 1.0 } else { -1.0 };
                blk.k_sag[l] = 0.25;
                blk.k_quad[l] = 0.12;
                for src in 0..=l {
                    let v = 0.02 * (1 + ((l + src) % 5)) as f32;
                    blk.w_rec_z[l][src] = v;
                    blk.w_rec_r[l][src] = -0.8 * v;
                    blk.w_rec_n[l][src] = 1.1 * v;
                    if b > 0 {
                        blk.w_prev_z[l][src] = 0.9 * v;
                        blk.w_prev_r[l][src] = -0.7 * v;
                        blk.w_prev_n[l][src] = 1.25 * v;
                    }
                }
            }
        }
        model.tone_svf = V3InterBlockToneSvfParams {
            h0_proj_w: [0.35, -0.22, 0.18, -0.41],
            s1_proj_w: 0.4,
            cab_inject_w: 0.3,
            b1_inject_w: 0.25,
            ..V3InterBlockToneSvfParams::identity()
        };
        let mut ctrl = V3ControlBusParams::identity();
        for k in 0..NUM_V3_MACRO_KNOBS {
            for b in 0..NUM_GRU_BLOCKS {
                for c in 0..12 {
                    ctrl.w_film_gain[k][b][c] = 0.04;
                    ctrl.w_film_bn[k][b][c] =
                        0.05 * if ((k + b + c) & 1) == 0 { 1.0 } else { -1.0 };
                }
            }
        }
        model.ctrl = ctrl;
        for t in 0..4 {
            let mut ro = V3TierReadout::default();
            for ch in 0..2 {
                for l in 0..24 {
                    ro.pre_cab_w[ch][l] = 0.05 * ((l % 4 + 1) as f32);
                    ro.direct_s2_w[ch][l] = 0.03 * ((l % 3 + 1) as f32);
                }
            }
            model.readouts[t] = ro;
        }
        let mut arena = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        pack_v3_matryoshka_to_arena(&model, &mut arena);
        arena
    }

    #[test]
    fn test_gru_sign_gauge_errors_and_prefix_commutativity() {
        let mut short = [0u8; 512];
        assert!(apply_gru_sign_gauge_watermark(&mut short, &[1.0; 24]).is_err());
        let mut lite = [0u8; ARENA_LITE_BYTES];
        assert!(apply_gru_sign_gauge_watermark(&mut lite, &[1.0; 8]).is_err());

        let master = build_rich_test_v3_arena();
        let seed_a = [0x2Au8; 32];
        let signs_a = derive_gru_sign_gauge_24(&seed_a);
        let mut ultra_gauged = master.clone();
        apply_gru_sign_gauge_watermark(&mut ultra_gauged, &signs_a).unwrap();

        let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
        let mut hdr = DnamJsonHeader::default_reference();
        hdr.controls_schema.push(crate::schema::ControlSchemaDef {
            id: alloc::string::String::from("gain"),
            name: alloc::string::String::from("Gain"),
            kind: crate::schema::ControlKind::Range,
            unit: None,
            lobatto_degree: Some(2),
            anchor_nodes_norm: Some(alloc::vec![0.0, -1.0, 1.0]),
            declared_range: Some([0.0, 10.0]),
            safe_clamped_range: Some([0.0, 10.0]),
            default: serde_json::json!(5.0),
            conditional_min: alloc::collections::BTreeMap::new(),
            conditional_max: alloc::collections::BTreeMap::new(),
            declared_options: None,
            active_options: None,
            clamp_warning: None,
            ui_hints: crate::schema::ControlUiHints::default(),
            calibration: None,
            user_metadata: alloc::collections::BTreeMap::new(),
        });
        let master_container = build_dnam_container(&mut hdr, &master, &master, false).unwrap();
        let gauged_container =
            build_dnam_container(&mut hdr.clone(), &ultra_gauged, &ultra_gauged, false).unwrap();

        for (t_idx, &tier) in tiers.iter().enumerate() {
            let prefix_len = tier.expected_arena_bytes();
            let mut sliced_then_gauged = master[..prefix_len].to_vec();
            let prefix_signs = &signs_a[..GRU_SIGN_LANES_PER_TIER[t_idx]];
            apply_gru_sign_gauge_watermark(&mut sliced_then_gauged, prefix_signs).unwrap();
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
                assert!(
                    (ml[i] - gl[i]).abs() <= 1e-6 && (mr[i] - gr[i]).abs() <= 1e-6,
                    "tier {:?} sample {}: L {} vs {}, R {} vs {}",
                    tier,
                    i,
                    ml[i],
                    gl[i],
                    mr[i],
                    gr[i]
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_gru_sign_gauge_random_keys_preserve_zero_ulp_audio(bits in 0u32..(1u32 << 24)) {
            let mut signs = [1.0_f32; GRU_TOTAL_LANES];
            for (i, slot) in signs.iter_mut().enumerate() {
                if ((bits >> i) & 1) == 1 {
                    *slot = -1.0;
                }
            }
            let master = build_rich_test_v3_arena();
            let mut gauged = master.clone();
            apply_gru_sign_gauge_watermark(&mut gauged, &signs).unwrap();

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
