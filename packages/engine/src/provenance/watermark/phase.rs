//! Layer 2 Pole-Angle Phase Coding buyer watermarking & black-box scoring
//! (`ref[sec.watermark.seed-derivation]`, `Test 8H`).

use super::{
    STAGE3_KAUTZ_BLOCKS_PER_TIER, kautz_soa_byte_offset, max_tiers_for_arena_len, read_f32_le,
    write_f32_le,
};
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, cold_sqrt_f64};

/// Maximum Layer 2 pole-angle phase perturbation (`0.05 deg = 8.726646e-4 rad`, `0.00 dB` magnitude change).
pub const PHASE_CODING_MAX_RAD: f32 = 8.726_646e-4;

#[inline]
fn derive_phase_coding_tier_digest(buyer_seed: &[u8; 32], tier: u8) -> [u8; 32] {
    let mut msg = [0u8; 17 + 32 + 1];
    msg[0..17].copy_from_slice(b"DNAM-WM-PHASE-V1:");
    msg[17..49].copy_from_slice(buyer_seed);
    msg[49] = tier;
    sha256_digest(&msg)
}

#[inline]
fn phase_chip_from_digest(digest: &[u8; 32], chip_idx: usize) -> f32 {
    let byte = digest[(chip_idx / 8) & 31];
    let bit = (byte >> (chip_idx & 7)) & 1;
    if bit == 1 { 1.0_f32 } else { -1.0_f32 }
}

/// Applies **Layer 2 (Kautz & Inter-Stage Pole-Angle Phase Coding)** in-place on `Arena 0..3`
/// using domain tag `"DNAM-WM-PHASE-V1"` (`delta_rad <= PHASE_CODING_MAX_RAD = 8.726646e-4 rad` = `0.05 deg`):
/// - Perturbs each tier's Stage III Kautz modal pole angles $\theta_m$ (`STAGE3_KAUTZ_BASES`, `+32..+48` in each `128 B` block)
///   by $\pm\Delta\theta$ (`0.00 dB` pole-radius change).
/// - Perturbs active Stage II lifted inter-stage resonator pole angles (`theta_mid_1` at `4,180` when `r_mid_1 > 0.0`,
///   and `theta_mid_2` at `4,212` when `r_mid_2 > 0.0`) by $\pm\Delta\theta$.
pub fn apply_phase_coding_watermark(
    arena: &mut [u8],
    buyer_seed: &[u8; 32],
    delta_rad: f32,
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: arena.len(),
        });
    }
    let step_rad = delta_rad.clamp(0.0, PHASE_CODING_MAX_RAD);

    for tier in 0..max_tiers {
        let digest = derive_phase_coding_tier_digest(buyer_seed, tier as u8);
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let theta_off = base + (8 + lane) * 4;
                let theta = read_f32_le(arena, theta_off);
                if (1e-3..=core::f32::consts::PI - 1e-3).contains(&theta) {
                    let chip_idx = blk * 4 + lane;
                    let chip = phase_chip_from_digest(&digest, chip_idx);
                    let perturbed =
                        (theta + step_rad * chip).clamp(1e-4, core::f32::consts::PI - 1e-4);
                    write_f32_le(arena, theta_off, perturbed);
                }
            }
        }
        if tier == 0 {
            let r_mid_1 = read_f32_le(arena, 4_176);
            let th1 = read_f32_le(arena, 4_180);
            if r_mid_1 > 0.0 && (1e-3..=core::f32::consts::PI - 1e-3).contains(&th1) {
                let chip = phase_chip_from_digest(&digest, 16);
                write_f32_le(
                    arena,
                    4_180,
                    (th1 + step_rad * chip).clamp(1e-4, core::f32::consts::PI - 1e-4),
                );
            }
            let r_mid_2 = read_f32_le(arena, 4_208);
            let th2 = read_f32_le(arena, 4_212);
            if r_mid_2 > 0.0 && (1e-3..=core::f32::consts::PI - 1e-3).contains(&th2) {
                let chip = phase_chip_from_digest(&digest, 17);
                write_f32_le(
                    arena,
                    4_212,
                    (th2 + step_rad * chip).clamp(1e-4, core::f32::consts::PI - 1e-4),
                );
            }
        }
    }
    Ok(())
}

/// Computes the Layer 2 pole-angle phase-coding Z-score ($\sigma$) of a suspect `Arena 0..3`
/// against `master_arena` and a candidate `buyer_seed` (`"DNAM-WM-PHASE-V1"`).
pub fn score_phase_coding_arena(
    master_arena: &[u8],
    suspect_arena: &[u8],
    buyer_seed: &[u8; 32],
) -> Result<f64, DnamError> {
    let shared_len = master_arena.len().min(suspect_arena.len());
    let max_tiers = max_tiers_for_arena_len(shared_len);
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: 10_752,
            actual: shared_len,
        });
    }
    let mut corr = 0.0_f64;
    let mut active = 0.0_f64;

    for tier in 0..max_tiers {
        let digest = derive_phase_coding_tier_digest(buyer_seed, tier as u8);
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            for lane in 0..4 {
                let theta_off = base + (8 + lane) * 4;
                let theta_m = read_f32_le(master_arena, theta_off);
                if !(1e-3..=core::f32::consts::PI - 1e-3).contains(&theta_m) {
                    continue;
                }
                let chip_idx = blk * 4 + lane;
                let chip = phase_chip_from_digest(&digest, chip_idx) as f64;
                let diff = (read_f32_le(suspect_arena, theta_off) - theta_m) as f64;
                let y = if diff > 1e-6 {
                    1.0
                } else if diff < -1e-6 {
                    -1.0
                } else {
                    0.0
                };
                corr += chip * y;
                active += y * y;
            }
        }
        if tier == 0 {
            for (r_off, th_off, chip_idx) in
                [(4_176_usize, 4_180_usize, 16_usize), (4_208, 4_212, 17)]
            {
                let theta_m = read_f32_le(master_arena, th_off);
                if read_f32_le(master_arena, r_off) > 0.0
                    && (1e-3..=core::f32::consts::PI - 1e-3).contains(&theta_m)
                {
                    let chip = phase_chip_from_digest(&digest, chip_idx) as f64;
                    let diff = (read_f32_le(suspect_arena, th_off) - theta_m) as f64;
                    let y = if diff > 1e-6 {
                        1.0
                    } else if diff < -1e-6 {
                        -1.0
                    } else {
                        0.0
                    };
                    corr += chip * y;
                    active += y * y;
                }
            }
        }
    }
    Ok(corr / cold_sqrt_f64(active.max(1.0)))
}

/// Computes the black-box Stage C pole-angle phase correlation Z-score $Z_{\text{phase},j}$
/// from extracted/identified complex pole angles (`master_pole_angles` vs `suspect_pole_angles`)
/// against a candidate `buyer_seed` (`"DNAM-WM-PHASE-V1"`).
pub fn score_phase_coding_blackbox(
    master_pole_angles: &[f32],
    suspect_pole_angles: &[f32],
    buyer_seed: &[u8; 32],
) -> Result<f64, DnamError> {
    let n = master_pole_angles.len().min(suspect_pole_angles.len());
    if n < 4 {
        return Err(DnamError::BufferTooSmall {
            expected: 4,
            actual: n,
        });
    }
    let tier_counts = [4_usize, 4, 8, 16];
    let mut corr = 0.0_f64;
    let mut active = 0.0_f64;
    let mut global_idx = 0_usize;

    for (tier, &cnt) in tier_counts.iter().enumerate() {
        let digest = derive_phase_coding_tier_digest(buyer_seed, tier as u8);
        for c in 0..cnt {
            if global_idx >= n {
                break;
            }
            let chip = phase_chip_from_digest(&digest, c) as f64;
            let diff = (suspect_pole_angles[global_idx] - master_pole_angles[global_idx]) as f64;
            let y = if diff > 1e-6 {
                1.0
            } else if diff < -1e-6 {
                -1.0
            } else {
                0.0
            };
            corr += chip * y;
            active += y * y;
            global_idx += 1;
        }
    }
    if n > 32 {
        let digest0 = derive_phase_coding_tier_digest(buyer_seed, 0);
        for extra in 0..(n - 32).min(2) {
            let idx = 32 + extra;
            let chip = phase_chip_from_digest(&digest0, 16 + extra) as f64;
            let diff = (suspect_pole_angles[idx] - master_pole_angles[idx]) as f64;
            let y = if diff > 1e-6 {
                1.0
            } else if diff < -1e-6 {
                -1.0
            } else {
                0.0
            };
            corr += chip * y;
            active += y * y;
        }
    }
    Ok(corr / cold_sqrt_f64(active.max(1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_coding_scoring_sign_and_blackbox_branches() {
        let mut short = [0u8; 512];
        assert!(
            apply_phase_coding_watermark(&mut short, &[0x11; 32], PHASE_CODING_MAX_RAD).is_err()
        );
        assert!(score_phase_coding_arena(&short, &short, &[0x11; 32]).is_err());
        assert!(score_phase_coding_blackbox(&[0.1; 2], &[0.1; 2], &[0x11; 32]).is_err());

        let mut master = alloc::vec![0u8; 64_000];
        for tier in 0..4 {
            for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                let base = kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    write_f32_le(
                        &mut master,
                        base + (8 + lane) * 4,
                        0.25 + 0.05 * (lane as f32),
                    );
                }
            }
        }
        write_f32_le(&mut master, 4_176, 0.85);
        write_f32_le(&mut master, 4_180, 0.35);
        write_f32_le(&mut master, 4_208, 0.85);
        write_f32_le(&mut master, 4_212, 0.65);

        let seed = [0x3Cu8; 32];
        let mut wm = master.clone();
        apply_phase_coding_watermark(&mut wm, &seed, PHASE_CODING_MAX_RAD).unwrap();

        // Identical suspect gives 0.0 (covers `else { 0.0 }` branch); inverted delta covers opposite sign
        assert_eq!(
            score_phase_coding_arena(&master, &master, &seed).unwrap(),
            0.0
        );
        let z_pos = score_phase_coding_arena(&master, &wm, &seed).unwrap();
        assert!(z_pos > 5.0);

        let z_bb_short = score_phase_coding_blackbox(&[0.5; 6], &[0.5; 6], &seed).unwrap();
        assert_eq!(z_bb_short, 0.0);
        let m34 = alloc::vec![0.5; 34];
        let mut s34 = m34.clone();
        s34[0] += 1e-4;
        s34[1] -= 1e-4;
        s34[32] += 1e-4;
        s34[33] -= 1e-4;
        assert!(
            score_phase_coding_blackbox(&m34, &s34, &seed)
                .unwrap()
                .is_finite()
        );
    }
}
