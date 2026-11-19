//! 3-Layer Buyer Watermarking (`Test 8H`: Layer 1 Coupled Bilinear Gauge, Layer 2 Pole-Angle Phase Coding,
//! Layer 3 Two-Seed Arcsine Tardos) & Prefix-Composable `4 × 56-D = 224-D` Gauge-Invariant Structural Hash.

use crate::schema::DnamError;

pub mod gru_gauge;
pub mod phase;
pub mod tardos;

pub use gru_gauge::{
    GRU_SIGN_LANES_PER_TIER, GRU_TOTAL_LANES, apply_gru_sign_gauge_watermark,
    derive_gru_sign_gauge_24,
};
pub use phase::{
    PHASE_CODING_MAX_RAD, apply_phase_coding_watermark, score_phase_coding_arena,
    score_phase_coding_blackbox,
};
pub use tardos::{
    TARDOS_CASCADE_RANGES, TardosCodebook7680, apply_cascade_head_gauge,
    apply_tardos_watermark_7680, apply_tardos_watermark_7680_envelopes,
    compute_emr_masked_tardos_envelopes, generate_tardos_code_7680, generate_tardos_codebook_7680,
    is_role_a_cascade_index, role_b_head_slot_for_index, score_tardos_watermark_7680,
};

#[inline]
pub(crate) fn read_f32_le(arena: &[u8], byte_offset: usize) -> f32 {
    f32::from_le_bytes([
        arena[byte_offset],
        arena[byte_offset + 1],
        arena[byte_offset + 2],
        arena[byte_offset + 3],
    ])
}

#[inline]
pub(crate) fn write_f32_le(arena: &mut [u8], byte_offset: usize, val: f32) {
    let b = val.to_le_bytes();
    arena[byte_offset..byte_offset + 4].copy_from_slice(&b);
}

#[inline]
pub(crate) const fn max_tiers_for_arena_len(arena_len: usize) -> usize {
    if arena_len >= 28_672 {
        4
    } else if arena_len >= 18_432 {
        3
    } else if arena_len >= 10_240 {
        2
    } else if arena_len >= 4_608 {
        1
    } else {
        0
    }
}

/// Per-tier byte offsets of the 128-byte Stage III Early Reflection FIR block (`16` Left + `16` Right `f32` taps)
/// inside `Arena 0..3`:
/// - `tier == 0` (`Lite`): `2_368..2_496` (`128 B`)
/// - `tier == 1` (`Full`): `5_824..5_952` (`128 B`)
/// - `tier == 2` (`Pro`): `14_848..14_976` (`128 B`)
/// - `tier == 3` (`Ultra`): `20_224..20_352` (`128 B`)
pub const STAGE3_EARLY_FIR_BASES: [usize; 4] = [2_368, 5_824, 14_848, 20_224];

/// Per-tier base byte offsets of the Stage III Orthonormal Kautz 128-byte (`[8, 4]` `f32` SoA) modal blocks
/// inside `Arena 0..3`:
/// - `tier == 0` (`Lite`, `1` block = `4` pole pairs, `128 B`): `2_496..2_624`
/// - `tier == 1` (`Full`, `1` block = `+4` pole pairs, `128 B`): `5_952..6_080`
/// - `tier == 2` (`Pro`, `1` block = `+4` pole pairs, `128 B`): `14_976..15_104`
/// - `tier == 3` (`Ultra`, `1` block = `+4` pole pairs, `128 B`): `20_352..20_480`
pub const STAGE3_KAUTZ_BASES: [usize; 4] = [2_496, 5_952, 14_976, 20_352];

/// Number of 128-byte (`4`-pole-pair) Kautz SoA blocks stored in each tier delta (`Lite=1`, `Full=1`, `Pro=1`, `Ultra=1`).
pub const STAGE3_KAUTZ_BLOCKS_PER_TIER: [usize; 4] = [1, 1, 1, 1];

/// Cumulative Kautz pole-pair channel base index per tier (`[0, 4, 8, 12]`, total `16` pole pairs across all 4 tiers).
pub const STAGE3_KAUTZ_TIER_CHANNEL_BASE: [usize; 4] = [0, 4, 8, 12];

/// Returns the byte offset of the 128-byte modal/Kautz SoA struct (`[8, 4]` floats:
/// `[r_low, dr, theta, b_gamma, c_re_l, c_im_l, c_re_r, c_im_r]`) for `(tier, block_idx)` inside `Arena 0..3`.
#[inline]
pub const fn kautz_soa_byte_offset(tier: usize, block_idx: usize) -> usize {
    let t = if tier < 4 { tier } else { 3 };
    STAGE3_KAUTZ_BASES[t] + block_idx * 128
}

/// Applies **Mechanism A (Exact Isometric Gauge Watermark)** in-place on a prefix-composable
/// `Arena 0..3` buffer (`4,608 B`, `10,240 B`, `18,432 B`, or `28,672 B`):
/// 1. Scales $b_{\gamma,b,c}$ by $\kappa_{b,c} \in [0.985, 1.015]$ and $(c_{\text{real},L/R,b,c}, c_{\text{imag},L/R,b,c})$
///    by $\kappa_{b,c}^{-1}$ across all present Stage III Kautz modal blocks (`[1, 1, 1, 1]` blocks per tier = `16` pole pairs in `Ultra`; `0.0000 dB` audio distortion).
/// 2. Rotates the ControlBus head subspace projections (`CP1` contributors, $\mathbf{U}_{\text{int}}$, and $\mathbf{W}_{\text{ctrl}}^{\text{base}}$ columns `0..4`)
///    by $\mathbf{Q}_{\text{buyer}} \in SO(4)$.
pub fn apply_isometric_gauge_watermark(
    arena: &mut [u8],
    kappa: &[f32],
    q_buyer_4x4: &[[f32; 4]; 4],
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 4_608,
            actual: arena.len(),
        });
    }
    let required_channels = STAGE3_KAUTZ_TIER_CHANNEL_BASE[max_tiers - 1]
        + STAGE3_KAUTZ_BLOCKS_PER_TIER[max_tiers - 1] * 4;
    if kappa.len() < required_channels {
        return Err(DnamError::BufferTooSmall {
            expected: required_channels,
            actual: kappa.len(),
        });
    }

    // 1. Modal/Kautz reciprocal scaling gauge (b_gamma * kappa, c_re_l..c_im_r / kappa)
    for tier in 0..max_tiers {
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        let ch_tier_base = STAGE3_KAUTZ_TIER_CHANNEL_BASE[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let ch_global = ch_tier_base + blk * 4 + lane;
                let k_val = kappa[ch_global];
                let inv_kappa = 1.0 / k_val;

                let b_gamma_off = base + (12 + lane) * 4;
                let c_re_l_off = base + (16 + lane) * 4;
                let c_im_l_off = base + (20 + lane) * 4;
                let c_re_r_off = base + (24 + lane) * 4;
                let c_im_r_off = base + (28 + lane) * 4;

                let b_gamma = read_f32_le(arena, b_gamma_off);
                let c_re_l = read_f32_le(arena, c_re_l_off);
                let c_im_l = read_f32_le(arena, c_im_l_off);
                let c_re_r = read_f32_le(arena, c_re_r_off);
                let c_im_r = read_f32_le(arena, c_im_r_off);

                write_f32_le(arena, b_gamma_off, b_gamma * k_val);
                write_f32_le(arena, c_re_l_off, c_re_l * inv_kappa);
                write_f32_le(arena, c_im_l_off, c_im_l * inv_kappa);
                write_f32_le(arena, c_re_r_off, c_re_r * inv_kappa);
                write_f32_le(arena, c_im_r_off, c_im_r * inv_kappa);
            }
        }
    }

    if crate::stages::weights::has_pipeline_flag(
        arena,
        crate::stages::weights::PIPE_FLAG_V3_MATRYOSHKA,
    ) && (crate::stages::weights::read_u32_le(
        arena,
        crate::stages::weights::ARENA0_V3_META_OFFSET,
    ) & crate::stages::weights::V3_META_STAGE3_PREFIX_OVERRIDES)
        != 0
    {
        let ov_offsets = [
            0usize,
            crate::stages::weights::ARENA1_STAGE3_PREFIX_OVERRIDE_OFFSET,
            crate::stages::weights::ARENA2_STAGE3_PREFIX_OVERRIDE_OFFSET,
            crate::stages::weights::ARENA3_STAGE3_PREFIX_OVERRIDE_OFFSET,
        ];
        for tier in 1..max_tiers {
            let ov = ov_offsets[tier];
            if arena.len() < ov + 192 * tier {
                break;
            }
            let n_fir = tier * 16;
            let n_kautz = tier * 4;
            let k_ov = ov + 2 * n_fir * 4;
            for m in 0..n_kautz {
                let inv_kappa = 1.0 / kappa[m];
                for arr_idx in 0..4 {
                    let off = k_ov + (arr_idx * n_kautz + m) * 4;
                    let c_val = read_f32_le(arena, off);
                    write_f32_le(arena, off, c_val * inv_kappa);
                }
            }
        }
    }

    // Helper closure to post-multiply a 4-float row at `row_byte_off` by `q_buyer_4x4`:
    let rotate_row_4 = |buf: &mut [u8], row_byte_off: usize| {
        let mut old_row = [0.0_f32; 4];
        for m in 0..4 {
            old_row[m] = read_f32_le(buf, row_byte_off + m * 4);
        }
        for k in 0..4 {
            let mut rotated = 0.0_f32;
            for m in 0..4 {
                rotated += old_row[m] * q_buyer_4x4[m][k];
            }
            write_f32_le(buf, row_byte_off + k * 4, rotated);
        }
    };

    // 2. Rotate inline CP1 contributors (`3_200..3_520` / Ultra `21_760..23_872`), `U_int^base` (`3_072..3_200`),
    //    `U_int,Delta` (`21_376..21_760`), and `W_ctrl^base[:, 0..4]` (`3_520..4_544`):
    let dir_off = crate::stages::weights::ARENA0_CP1_DIR_OFFSET;
    let base_off = crate::stages::weights::ARENA0_CP1_BASE_OFFSET;
    let base_end = crate::stages::weights::ARENA0_CP1_BASE_END;
    let uint_base_off = crate::stages::weights::ARENA0_UINT_BASE_OFFSET;
    let w_ctrl_off = crate::stages::weights::ARENA0_W_CTRL_OFFSET;
    let uint_ultra_off = crate::stages::weights::ARENA3_UINT_ULTRA_OFFSET;
    let cp1_ultra_off = crate::stages::weights::ARENA3_CP1_ULTRA_OFFSET;
    let cp1_ultra_end = crate::stages::weights::ARENA3_CP1_ULTRA_END;

    let cp1_count = u16::from_le_bytes([arena[dir_off], arena[dir_off + 1]]) as usize;
    if (1..=26).contains(&cp1_count) {
        let use_3b_dir = cp1_count <= 8
            && arena[dir_off + 2] == 0
            && arena[dir_off + 3] == 0
            && ((1..=3).contains(&(arena[dir_off + 5] & 0x0F))
                || (1..=3).contains(&(arena[dir_off + 6] & 0x0F)));
        let mut b_cursor = base_off;
        let mut u_cursor = cp1_ultra_off;
        for c in 0..cp1_count {
            let packed = if use_3b_dir {
                let b1 = arena[dir_off + 4 + c * 3 + 1];
                let b2 = arena[dir_off + 4 + c * 3 + 2];
                if (1..=3).contains(&(b1 & 0x0F)) {
                    b1
                } else {
                    b2
                }
            } else {
                arena[dir_off + 2 + c]
            };
            let lanes = (packed & 0x0F) as usize;
            let b_bytes = 4 * (9 * lanes + 4);
            if !(1..=3).contains(&lanes) || b_cursor + b_bytes > base_end {
                break;
            }
            for l in 0..lanes {
                rotate_row_4(arena, b_cursor + l * 32);
            }
            b_cursor += b_bytes;
            if max_tiers >= 4 && u_cursor + (32 * lanes + 16) <= cp1_ultra_end {
                for l in 0..lanes {
                    rotate_row_4(arena, u_cursor + l * 32);
                }
                u_cursor += 32 * lanes + 16;
            }
        }
    }

    // Rotate `U_int^base[0..4][0..4]` (`3_072..3_200`) and `W_ctrl^base[row, 0..4]` (`3_520..4_544`):
    for r in 0..4 {
        rotate_row_4(arena, uint_base_off + r * 32);
    }
    for row in 0..16 {
        rotate_row_4(arena, w_ctrl_off + row * 64);
    }
    if max_tiers >= 4 {
        // Rotate `3 x U_int,Delta[0..12][0..4]` at `21_376..21_760` (`W_CTRL_ULTRA_BASE` uses `z_stage_ultra`, not `z_head`)
        for r in 0..12 {
            rotate_row_4(arena, uint_ultra_off + r * 32);
        }
    }

    Ok(())
}

pub use super::fingerprint::{extract_structural_hash_224d, structural_cosine_similarity};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::KAUTZ_TOTAL_CHANNELS;
    use crate::provenance::security::extract_stage3_tier_fingerprint_component;
    use proptest::prelude::*;

    #[test]
    fn test_tardos_4buyer_collusion_role_a_exclusion_and_phase_coding() {
        let mut short = [0u8; 1024];
        assert!(apply_isometric_gauge_watermark(&mut short, &[1.0; 16], &[[0.0; 4]; 4]).is_err());
        assert!(extract_structural_hash_224d(&short).is_err());
        assert!(apply_tardos_watermark_7680(&mut short, &[1.0; 7680], 1e-4).is_err());
        assert!(score_tardos_watermark_7680(&short, &short, &[1.0; 7680]).is_err());
        assert!(
            apply_phase_coding_watermark(&mut short, &[0x11; 32], PHASE_CODING_MAX_RAD).is_err()
        );
        assert!(score_phase_coding_arena(&short, &short, &[0x11; 32]).is_err());
        assert!(score_phase_coding_blackbox(&[0.1; 2], &[0.1; 2], &[0x11; 32]).is_err());

        let mut valid_lite = [0u8; 4_608];
        assert!(
            apply_isometric_gauge_watermark(&mut valid_lite, &[1.0; 2], &[[0.0; 4]; 4]).is_err()
        );
        assert!(apply_tardos_watermark_7680(&mut valid_lite, &[1.0; 100], 1e-4).is_err());
        assert!(score_tardos_watermark_7680(&valid_lite, &valid_lite, &[1.0; 100]).is_err());

        let mut master = alloc::vec![0u8; 28_672];
        for i in 0..7_168 {
            let v = (((i * 37) % 101) as f32 - 50.0) * 0.01;
            master[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        let mp_secret = [0x5Au8; 32];
        let c1 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-1");
        let c2 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-2");
        let c3 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-3");
        let c4 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-4");
        let c_innocent = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-innocent");

        // Verify all 64 Role A indices have chips[i] == 0.0 and accusation_weights[i] == 0.0:
        for &(start, end, _) in &TARDOS_CASCADE_RANGES {
            for idx in start..start + 16 {
                assert_eq!(c1.chips[idx], 0.0);
                assert_eq!(c1.accusation_weights[idx], 0.0);
            }
            for idx in start + 16..end {
                assert!(c1.chips[idx] == 1.0 || c1.chips[idx] == -1.0);
                assert_eq!(c1.chips[idx].signum(), c1.accusation_weights[idx].signum());
            }
        }

        let mut b1 = master.clone();
        let mut b2 = master.clone();
        let mut b3 = master.clone();
        let mut b4 = master.clone();
        apply_tardos_watermark_7680(&mut b1, &c1, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b2, &c2, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b3, &c3, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b4, &c4, 2e-4).unwrap();

        // Verify Role A bytes in b1 are 100% identical to master:
        for &(_, _, byte_base) in &TARDOS_CASCADE_RANGES {
            assert_eq!(
                b1[byte_base..byte_base + 64],
                master[byte_base..byte_base + 64]
            );
        }

        // 1. 4-buyer linear averaging collusion:
        let mut pirated_avg = master.clone();
        // 2. 4-buyer nonlinear majority-vote / interleaving collusion:
        let mut pirated_maj = master.clone();
        for i in 0..7_168 {
            let off = i * 4;
            let (v1, v2, v3, v4) = (
                read_f32_le(&b1, off),
                read_f32_le(&b2, off),
                read_f32_le(&b3, off),
                read_f32_le(&b4, off),
            );
            let avg = 0.25 * (v1 + v2 + v3 + v4);
            write_f32_le(&mut pirated_avg, off, avg);

            let vm = read_f32_le(&master, off);
            let votes = [v1 - vm, v2 - vm, v3 - vm, v4 - vm]
                .iter()
                .fold(0_i32, |acc, &d| {
                    acc + if d > 1e-7 {
                        1
                    } else if d < -1e-7 {
                        -1
                    } else {
                        0
                    }
                });
            let maj = if votes > 0 {
                v1.max(v2).max(v3).max(v4)
            } else if votes < 0 {
                v1.min(v2).min(v3).min(v4)
            } else {
                [v1, v2, v3, v4][i & 3]
            };
            write_f32_le(&mut pirated_maj, off, maj);
        }

        for code in [&c1, &c2, &c3, &c4] {
            let z_avg = score_tardos_watermark_7680(&master, &pirated_avg, code).unwrap();
            assert!(
                z_avg > 8.5,
                "Linear-avg colluder Z-score too low: {z_avg:.2}"
            );
            let z_maj = score_tardos_watermark_7680(&master, &pirated_maj, code).unwrap();
            assert!(
                z_maj > 7.5,
                "Majority-vote colluder Z-score too low: {z_maj:.2}"
            );
        }
        let z_innocent = score_tardos_watermark_7680(&master, &pirated_avg, &c_innocent).unwrap();
        assert!(z_innocent.abs() < 4.5);

        // 3. Verify Layer 2 Phase Coding & 1e-2 rad fingerprint quantization invariance:
        for tier in 0..4 {
            for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                let base = kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    write_f32_le(
                        &mut master,
                        base + (8 + lane) * 4,
                        0.15 + 0.05 * (lane as f32),
                    );
                }
            }
        }

        let hash_pre_phase = extract_structural_hash_224d(&master).unwrap();
        let fp_pre_phase = extract_stage3_tier_fingerprint_component(
            &master[2_368..2_496],
            &master[2_496..2_624],
            4,
        )
        .unwrap();

        let mut phase_arena = master.clone();
        let buyer_phase_seed = [0x3C_u8; 32];
        apply_phase_coding_watermark(&mut phase_arena, &buyer_phase_seed, PHASE_CODING_MAX_RAD)
            .unwrap();

        let hash_post_phase = extract_structural_hash_224d(&phase_arena).unwrap();
        let fp_post_phase = extract_stage3_tier_fingerprint_component(
            &phase_arena[2_368..2_496],
            &phase_arena[2_496..2_624],
            4,
        )
        .unwrap();
        assert_eq!(fp_pre_phase, fp_post_phase);
        assert!((1.0 - structural_cosine_similarity(&hash_pre_phase, &hash_post_phase)) < 1e-12);

        let z_phase = score_phase_coding_arena(&master, &phase_arena, &buyer_phase_seed).unwrap();
        assert!(z_phase > 3.9);
        let z_phase_wrong = score_phase_coding_arena(&master, &phase_arena, &[0x99; 32]).unwrap();
        assert!(z_phase_wrong.abs() < 3.5);

        // Extract all 16 Kautz pole angles and verify black-box Z-score:
        let mut master_angles_16 = alloc::vec::Vec::with_capacity(16);
        let mut suspect_angles_16 = alloc::vec::Vec::with_capacity(16);
        for tier in 0..4 {
            for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                let base = kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    master_angles_16.push(read_f32_le(&master, base + (8 + lane) * 4));
                    suspect_angles_16.push(read_f32_le(&phase_arena, base + (8 + lane) * 4));
                }
            }
        }
        let z_bb_16 =
            score_phase_coding_blackbox(&master_angles_16, &suspect_angles_16, &buyer_phase_seed)
                .unwrap();
        assert!((z_bb_16 - z_phase).abs() < 1e-9);

        // Verify inactive/zero Kautz blocks (theta == 0.0) are not clamped into false +1e-4 chips:
        let mut sparse_master = alloc::vec![0u8; 28_672];
        for lane in 0..4 {
            write_f32_le(
                &mut sparse_master,
                2_496 + (8 + lane) * 4,
                0.20 + 0.05 * (lane as f32),
            );
        }
        let mut sparse_suspect = sparse_master.clone();
        apply_phase_coding_watermark(&mut sparse_suspect, &buyer_phase_seed, PHASE_CODING_MAX_RAD)
            .unwrap();
        let z_sparse =
            score_phase_coding_arena(&sparse_master, &sparse_suspect, &buyer_phase_seed).unwrap();
        assert!((z_sparse - 2.0).abs() < 1e-6);

        // 4. Verify EMR-masked Tardos envelopes and apply_cascade_head_gauge composition:
        let gram_diag = [4.0_f64; 256];
        let envs = compute_emr_masked_tardos_envelopes(&gram_diag, [2e-4, 2e-4]);
        assert_eq!(envs[0], 0.0); // Role A
        assert!((envs[16] - 1e-4).abs() < 1e-7); // Role B: 2e-4 / sqrt(4) = 1e-4
        let mut env_arena = master.clone();
        apply_tardos_watermark_7680_envelopes(&mut env_arena, &c1, &envs).unwrap();
        assert!(score_tardos_watermark_7680(&master, &env_arena, &c1).unwrap() > 15.0);

        let mut gauge_arena = alloc::vec![0u8; 28_672];
        let tier0_base = TARDOS_CASCADE_RANGES[0].2;
        for off in (tier0_base + 64..tier0_base + 576).step_by(4) {
            write_f32_le(&mut gauge_arena, off, 0.5);
        }
        let delta_dyadic = [[1.0_f32; 4]; 32]; // m: 1.0 -> 2.0
        apply_cascade_head_gauge(&mut gauge_arena, 0, &delta_dyadic).unwrap();
        assert_eq!(read_f32_le(&gauge_arena, tier0_base + 64), 0.25);
        // Compose a second dyadic gauge (m: 2.0 -> 4.0) on already non-zero Role B deltas:
        let delta_second = [[2.0_f32; 4]; 32];
        apply_cascade_head_gauge(&mut gauge_arena, 0, &delta_second).unwrap();
        assert_eq!(read_f32_le(&gauge_arena, tier0_base + 64), 0.125);
        assert!(apply_cascade_head_gauge(&mut gauge_arena, 0, &[[-4.0_f32; 4]; 32]).is_err());

        // Exercise Ultra CP1 1-byte and 3-byte directory gauge rotation
        let mut ext_arena = master.clone();
        let dir_off = crate::stages::weights::ARENA0_CP1_DIR_OFFSET;
        ext_arena[dir_off..dir_off + 2].copy_from_slice(&2u16.to_le_bytes());
        ext_arena[dir_off + 2] = 2; // non-zero -> 1-byte dir path
        ext_arena[dir_off + 3] = 1;
        apply_isometric_gauge_watermark(&mut ext_arena, &[1.02; 16], &[[1.0, 0.0, 0.0, 0.0]; 4])
            .unwrap();

        let mut dir3_arena = master.clone();
        dir3_arena[dir_off..dir_off + 2].copy_from_slice(&3u16.to_le_bytes());
        dir3_arena[dir_off + 2..dir_off + 4].fill(0);
        dir3_arena[dir_off + 5] = 2; // c=0 uses b1=2
        dir3_arena[dir_off + 6] = 0;
        dir3_arena[dir_off + 8] = 0; // c=1 uses b2=1
        dir3_arena[dir_off + 9] = 1;
        dir3_arena[dir_off + 11] = 9; // c=2 breaks on invalid lanes > 3
        dir3_arena[dir_off + 12] = 9;
        apply_isometric_gauge_watermark(&mut dir3_arena, &[1.01; 16], &[[1.0, 0.0, 0.0, 0.0]; 4])
            .unwrap();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_isometric_gauge_preserves_224d_structural_hash(
            kappa in prop::array::uniform16(0.985f32..1.015f32),
            angle1 in -3.0f32..3.0f32,
            angle2 in -3.0f32..3.0f32,
        ) {
            let mut master = alloc::vec![0u8; 28_672];
            for i in 0..7_168 {
                let v = (((i * 29 + 7) % 97) as f32 - 48.0) * 0.015;
                master[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            master[crate::stages::weights::ARENA0_CP1_DIR_OFFSET
                ..crate::stages::weights::ARENA0_CP1_DIR_OFFSET + 2]
                .fill(0);
            for tier in 0..4 {
                for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                    let base = kautz_soa_byte_offset(tier, blk);
                    for lane in 0..4 {
                        write_f32_le(&mut master, base + lane * 4, 0.85);
                        write_f32_le(&mut master, base + (4 + lane) * 4, 0.10);
                        write_f32_le(&mut master, base + (8 + lane) * 4, 0.25);
                        write_f32_le(&mut master, base + (12 + lane) * 4, 0.30);
                        write_f32_le(&mut master, base + (16 + lane) * 4, 0.45);
                        write_f32_le(&mut master, base + (20 + lane) * 4, -0.20);
                        write_f32_le(&mut master, base + (24 + lane) * 4, 0.35);
                        write_f32_le(&mut master, base + (28 + lane) * 4, -0.15);
                    }
                }
            }

            let mut kappa_16 = [1.0f32; KAUTZ_TOTAL_CHANNELS];
            for i in 0..KAUTZ_TOTAL_CHANNELS {
                kappa_16[i] = kappa[i % 16];
            }
            let (c1, s1) = (angle1.cos(), angle1.sin());
            let (c2, s2) = (angle2.cos(), angle2.sin());
            let q = [
                [c1, -s1, 0.0, 0.0],
                [s1, c1, 0.0, 0.0],
                [0.0, 0.0, c2, -s2],
                [0.0, 0.0, s2, c2],
            ];

            let mut watermarked = master.clone();
            apply_isometric_gauge_watermark(&mut watermarked, &kappa_16, &q).unwrap();

            let f_master = extract_structural_hash_224d(&master).unwrap();
            let f_watermarked = extract_structural_hash_224d(&watermarked).unwrap();
            let cos_sim = structural_cosine_similarity(&f_master, &f_watermarked);
            prop_assert!((cos_sim - 1.0).abs() < 1e-11);
        }
    }
}
