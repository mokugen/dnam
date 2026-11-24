//! 3-Layer Buyer Watermarking (`Test 8H`: Layer 1 Coupled Bilinear / `0-ULP` `SONA` Sign-Flip Gauge,
//! Layer 2 Pole-Angle Phase Coding, Layer 3 Two-Seed Arcsine Tardos) & Prefix-Composable
//! `4 × 56-D = 224-D` Gauge-Invariant Structural Hash.

use crate::container::{CP1_MAGIC, decode_cp1_shared_stream_into_slice, encode_cp1_shared_stream};
use crate::schema::DnamError;
use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, Arena3UltraDeltaWire, Cp1InlineDirWire, KautzSoaBlockWire,
    LINEAR_READOUT_BASES, LinearReadoutFullWire, LinearReadoutProWire, LinearReadoutUltraWire,
    PIPE_FLAG_V3_MATRYOSHKA, PIPE_MAGIC_HI, PIPE_MAGIC_MASK, SONA_META_CONVOLVER_PREFIX_OVERRIDES,
};
use dnam_vm::control::Cp1ContributorSlot;
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromBytes, IntoBytes};

const ARENA_LITE_BYTES: usize = ARENA_TIER_BYTES[0];
const ARENA_FULL_BYTES: usize = ARENA_TIER_BYTES[1];
const ARENA_PRO_BYTES: usize = ARENA_TIER_BYTES[2];
const ARENA_ULTRA_BYTES: usize = ARENA_TIER_BYTES[3];

pub(crate) mod phase;
pub(crate) mod sona_gauge;
pub(crate) mod tardos;

pub use dnam_vm::bin::{
    CONVOLVER_EARLY_FIR_BASES as STAGE3_EARLY_FIR_BASES,
    CONVOLVER_KAUTZ_BASES as STAGE3_KAUTZ_BASES,
    CONVOLVER_KAUTZ_BLOCKS_PER_TIER as STAGE3_KAUTZ_BLOCKS_PER_TIER,
};
/// Cumulative Kautz resonator channel start index across `Lite / Full / Pro / Ultra`.
pub const STAGE3_KAUTZ_TIER_CHANNEL_BASE: [usize; 4] = [0, 8, 12, 24];
pub use phase::{
    PHASE_CODING_MAX_RAD, apply_phase_coding_watermark, score_phase_coding_arena,
    score_phase_coding_blackbox,
};
pub use sona_gauge::{
    SONA_SIGN_LANES_PER_TIER, SONA_TOTAL_LANES, apply_sona_sign_gauge_watermark,
    derive_sona_sign_gauge_40,
};
#[cfg(feature = "crypto")]
pub use tardos::reframe_buyer_dau1_container_with_dbin;
pub use tardos::{
    TARDOS_ACTIVE_LATENT_MASK_THRESH, TARDOS_CASCADE_RANGES, TARDOS_DEFAULT_KNOBS_END,
    TARDOS_DEFAULT_KNOBS_START, TARDOS_GLOBAL_LATENT_MODS_END, TARDOS_GLOBAL_LATENT_MODS_START,
    TARDOS_LITE_LATENT_MODS_END, TARDOS_LITE_LATENT_MODS_START, TARDOS_ULTRA_LATENT_MODS_END,
    TARDOS_ULTRA_LATENT_MODS_START, TARDOS_ULTRA_RESERVE_RANGE, TARDOS_W_READOUT_FP16_RANGES,
    TardosCodebook7680, apply_cascade_head_gauge, apply_tardos_w_readout_fp16,
    apply_tardos_watermark_7680, apply_tardos_watermark_7680_envelopes,
    compute_emr_masked_tardos_envelopes, for_each_tardos_f32_offset_up_to,
    for_each_tardos_fp16_offset_up_to, generate_tardos_code_7680, generate_tardos_codebook_7680,
    is_role_a_cascade_index, mask_tardos_stage_latent_active_coords,
    reframe_buyer_container_with_dbin, role_b_head_slot_for_index, score_tardos_w_readout_fp16,
    score_tardos_watermark_7680, score_tardos_watermark_7680_with_buyer_gauge,
};

pub use super::fingerprint::{extract_structural_hash_224d, structural_cosine_similarity};

#[inline]
pub(crate) const fn max_tiers_for_arena_len(arena_len: usize) -> usize {
    if arena_len >= ARENA_ULTRA_BYTES {
        4
    } else if arena_len >= ARENA_PRO_BYTES {
        3
    } else if arena_len >= ARENA_FULL_BYTES {
        2
    } else if arena_len >= ARENA_LITE_BYTES {
        1
    } else {
        0
    }
}

/// Returns the byte offset of the 128-byte modal/Kautz SoA struct (`[8, 4]` floats)
/// for `(tier, block_idx)` inside `Arena 0..3`.
#[inline]
pub const fn kautz_soa_byte_offset(tier: usize, block_idx: usize) -> usize {
    let t = if tier < 4 { tier } else { 3 };
    STAGE3_KAUTZ_BASES[t] + block_idx * 128
}

#[inline]
fn scale_f32_le(slot: &mut F32, factor: f32) {
    *slot = F32::new(slot.get() * factor);
}

#[inline]
fn scale_fp16_le(slot: &mut U16, factor: f32) {
    let bits = slot.get();
    if (bits & 0x7FFF) != 0 {
        *slot = U16::new(f32_to_fp16(fp16_to_f32(bits) * factor));
    }
}

#[inline]
fn rotate_row_4_f32(row: &mut [F32], q_buyer_4x4: &[[f32; 4]; 4]) {
    let old = [row[0].get(), row[1].get(), row[2].get(), row[3].get()];
    for k in 0..4 {
        let mut rotated = 0.0_f32;
        for m in 0..4 {
            rotated += old[m] * q_buyer_4x4[m][k];
        }
        row[k] = F32::new(rotated);
    }
}

#[inline]
fn cp1_inline_v_base_nonzero(cp1_base_words: &[F32], dir: &Cp1InlineDirWire) -> bool {
    let count = dir.cp1_dir_count.get() as usize;
    if !(1..=8).contains(&count) || dir.flags.get() != 0 {
        return false;
    }
    let mut cursor = 0usize;
    for c in 0..count {
        let lanes = (dir.entries[c][1] & 0x0F) as usize;
        let slot_words = 9 * lanes + 4;
        if !(1..=3).contains(&lanes) || cursor + slot_words > cp1_base_words.len() {
            break;
        }
        let v_off = cursor + 9 * lanes;
        if cp1_base_words[v_off..v_off + 4]
            .iter()
            .any(|w| w.get().abs() > 1e-12)
        {
            return true;
        }
        cursor += slot_words;
    }
    false
}

/// Applies the `SO(4)` `Q_buyer` prefix rotation to overflow `CP1\0` contributors inside Region 4
/// `shared_extent` (`main_base[l][0..4]`, leaving `4..8`, `main_ultra`, and `v_base` untouched).
pub fn apply_isometric_gauge_shared_cp1_stream(
    shared_extent: &mut [u8],
    q_buyer_4x4: &[[f32; 4]; 4],
) -> Result<(), DnamError> {
    let Some(cp1_pos) = shared_extent.windows(4).position(|w| w == CP1_MAGIC) else {
        return Ok(());
    };
    let cp1_tail = &shared_extent[cp1_pos..];
    let count = U16::read_from_prefix(&cp1_tail[4..])
        .map(|(v, _)| v.get() as usize)
        .unwrap_or(0);
    if count == 0 {
        return Ok(());
    }
    let mut slots = alloc::vec![Cp1ContributorSlot::zero(); count];
    decode_cp1_shared_stream_into_slice(cp1_tail, &mut slots)?;
    for slot in &mut slots {
        for l in 0..(slot.lanes as usize).min(3) {
            let old = [
                slot.main_base[l][0],
                slot.main_base[l][1],
                slot.main_base[l][2],
                slot.main_base[l][3],
            ];
            for k in 0..4 {
                let mut rot = 0.0_f32;
                for m in 0..4 {
                    rot += old[m] * q_buyer_4x4[m][k];
                }
                slot.main_base[l][k] = rot;
            }
        }
    }
    let reencoded = encode_cp1_shared_stream(&slots)?;
    if reencoded.len() > cp1_tail.len() {
        return Err(DnamError::BufferTooSmall {
            expected: cp1_pos + reencoded.len(),
            actual: shared_extent.len(),
        });
    }
    shared_extent[cp1_pos..cp1_pos + reencoded.len()].copy_from_slice(&reencoded);
    Ok(())
}

fn scale_prefix_override_kautz(
    prefix_override: &mut [u8],
    fir_bytes: usize,
    k_pre: usize,
    kappa: &[f32],
) {
    if let Some(tail) = prefix_override.get_mut(fir_bytes..)
        && let Ok(c_words) = <[F32]>::mut_from_bytes(tail)
        && c_words.len() == 4 * k_pre
    {
        for m in 0..k_pre {
            let inv_kappa = 1.0 / kappa[m];
            scale_f32_le(&mut c_words[m], inv_kappa);
            scale_f32_le(&mut c_words[k_pre + m], inv_kappa);
            scale_f32_le(&mut c_words[2 * k_pre + m], inv_kappa);
            scale_f32_le(&mut c_words[3 * k_pre + m], inv_kappa);
        }
    }
}

fn scale_readout_kautz_extra_slots(
    w_ro_and_mods: &mut [U16],
    extra_start: usize,
    b_start: usize,
    total_prev: usize,
    new_start: usize,
    new_count: usize,
    kappa: &[f32],
) {
    let extra = &mut w_ro_and_mods[extra_start..];
    for m in 0..total_prev {
        scale_fp16_le(&mut extra[b_start + m], kappa[m]);
    }
    let c_start = b_start + total_prev;
    for m in 0..new_count {
        let inv_kappa = 1.0 / kappa[new_start + m];
        let b = c_start + m * 4;
        for slot in &mut extra[b..b + 4] {
            scale_fp16_le(slot, inv_kappa);
        }
    }
}

/// Applies **Mechanism A (Exact Isometric Gauge Watermark)** in-place on `Arena 0..3`:
/// 1. Scales Kautz $b_{\gamma,b,c}$ by $\kappa_{b,c}$ and $(c_{\text{re},L/R}, c_{\text{im},L/R})$ by $\kappa_{b,c}^{-1}$
///    across Stage III Kautz blocks + `LinearReadout{Full,Pro,Ultra}Wire` prefix overrides & `fp16` slots.
/// 2. Post-multiplies `z_head[0..4]` subspaces (`cp1_base.main_base[l][0..4]`, `u_int_base[r][0..4]`,
///    Block `0` `u_int_ultra[0][r][0..4]`, and `w_ctrl[0..16][0..4]`) by `Q_buyer in SO(4)` while
///    leaving `v_base`, `z_stage`, `u_int_base[r][4..8]`, `u_int_ultra[1..3]`, and `cp1_ultra` untouched.
pub fn apply_isometric_gauge_watermark(
    arena: &mut [u8],
    kappa: &[f32],
    q_buyer_4x4: &[[f32; 4]; 4],
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
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

    // 1. Modal/Kautz reciprocal scaling gauge
    for tier in 0..max_tiers {
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier];
        let ch_tier_base = STAGE3_KAUTZ_TIER_CHANNEL_BASE[tier];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(tier, blk);
            let Ok((k_blk, _)) = KautzSoaBlockWire::mut_from_prefix(&mut arena[base..]) else {
                continue;
            };
            for lane in 0..4 {
                let k_val = kappa[ch_tier_base + blk * 4 + lane];
                let inv_kappa = 1.0 / k_val;
                scale_f32_le(&mut k_blk.b_gamma[lane], k_val);
                scale_f32_le(&mut k_blk.c_re_l[lane], inv_kappa);
                scale_f32_le(&mut k_blk.c_im_l[lane], inv_kappa);
                scale_f32_le(&mut k_blk.c_re_r[lane], inv_kappa);
                scale_f32_le(&mut k_blk.c_im_r[lane], inv_kappa);
            }
        }
    }

    let has_prefix_overrides = Arena0LiteWire::ref_from_prefix(arena).is_ok_and(|(a0, _)| {
        let pf = a0.crossbar.pipeline_flags.get();
        let mf = a0.crossbar.sona_meta_flags.get();
        (pf & PIPE_MAGIC_MASK) == PIPE_MAGIC_HI
            && (pf & PIPE_FLAG_V3_MATRYOSHKA) != 0
            && (mf & SONA_META_CONVOLVER_PREFIX_OVERRIDES) != 0
    });
    if has_prefix_overrides {
        if max_tiers >= 2
            && let Ok((w, _)) =
                LinearReadoutFullWire::mut_from_prefix(&mut arena[LINEAR_READOUT_BASES[1]..])
        {
            scale_prefix_override_kautz(&mut w.prefix_override, 128, 4, kappa);
            scale_readout_kautz_extra_slots(&mut w.w_ro_and_mods_fp16, 296, 32, 8, 4, 4, kappa);
        }
        if max_tiers >= 3
            && let Ok((w, _)) =
                LinearReadoutProWire::mut_from_prefix(&mut arena[LINEAR_READOUT_BASES[2]..])
        {
            scale_prefix_override_kautz(&mut w.prefix_override, 256, 8, kappa);
            scale_readout_kautz_extra_slots(&mut w.w_ro_and_mods_fp16, 392, 32, 12, 8, 4, kappa);
        }
        if max_tiers >= 4
            && let Ok((w, _)) =
                LinearReadoutUltraWire::mut_from_prefix(&mut arena[LINEAR_READOUT_BASES[3]..])
        {
            scale_prefix_override_kautz(&mut w.prefix_override, 384, 12, kappa);
            scale_readout_kautz_extra_slots(&mut w.w_ro_and_mods_fp16, 488, 96, 24, 12, 12, kappa);
        }
    }

    // 2. Rotate ControlBus z_head[0..4] projections in Arena 0 and Block 0 u_int_ultra in Arena 3
    if let Ok((a0, _)) = Arena0LiteWire::mut_from_prefix(arena) {
        let cb = &mut a0.control_bus_lite;
        let cp1_dir = cb.cp1_dir;
        let cp1_count = cp1_dir.cp1_dir_count.get() as usize;
        if let Ok(cp1_words) = <[F32]>::mut_from_bytes(&mut cb.cp1_base) {
            if cb.u_int_base.as_bytes().iter().all(|&b| b == 0)
                && cp1_inline_v_base_nonzero(cp1_words, &cp1_dir)
            {
                for r in 0..4 {
                    cb.u_int_base[r][r] = F32::new(1.0);
                    cb.u_int_base[r][4 + r] = F32::new(1.0);
                }
            }
            if (1..=8).contains(&cp1_count) && cp1_dir.flags.get() == 0 {
                let mut cursor = 0usize;
                for c in 0..cp1_count {
                    let lanes = (cp1_dir.entries[c][1] & 0x0F) as usize;
                    let slot_words = 9 * lanes + 4;
                    if !(1..=3).contains(&lanes) || cursor + slot_words > cp1_words.len() {
                        break;
                    }
                    for l in 0..lanes {
                        rotate_row_4_f32(
                            &mut cp1_words[cursor + l * 8..cursor + l * 8 + 4],
                            q_buyer_4x4,
                        );
                    }
                    cursor += slot_words;
                }
            }
        }
        for r in 0..4 {
            rotate_row_4_f32(&mut cb.u_int_base[r][..4], q_buyer_4x4);
        }
        for row in 0..16 {
            rotate_row_4_f32(&mut cb.w_ctrl[row][..4], q_buyer_4x4);
        }
    }

    if max_tiers >= 4
        && let Ok((a3, _)) = Arena3UltraDeltaWire::mut_from_prefix(&mut arena[ARENA_PRO_BYTES..])
    {
        for r in 0..4 {
            rotate_row_4_f32(
                &mut a3.control_bus_ultra_delta.u_int_ultra[0][r][..4],
                q_buyer_4x4,
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::KAUTZ_TOTAL_CHANNELS;
    use core::mem::offset_of;
    use dnam_vm::bin::Arena0ControlBusWire;
    use proptest::prelude::*;
    use zerocopy::byteorder::little_endian::U16 as LeU16;

    const ARENA0_CP1_DIR_OFFSET: usize =
        offset_of!(Arena0LiteWire, control_bus_lite) + offset_of!(Arena0ControlBusWire, cp1_dir);

    #[inline]
    fn read_f32_wire(arena: &[u8], byte_offset: usize) -> f32 {
        arena
            .get(byte_offset..)
            .and_then(|s| F32::read_from_prefix(s).ok())
            .map_or(0.0, |(v, _)| v.get())
    }

    #[inline]
    fn write_f32_le(arena: &mut [u8], byte_offset: usize, val: f32) {
        if let Some(dst) = arena.get_mut(byte_offset..) {
            let _ = F32::new(val).write_to_prefix(dst);
        }
    }

    #[test]
    fn test_tardos_4buyer_collusion_gauge_and_phase_coding() {
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

        let mut valid_lite = [0u8; ARENA_LITE_BYTES];
        assert!(
            apply_isometric_gauge_watermark(&mut valid_lite, &[1.0; 2], &[[0.0; 4]; 4]).is_err()
        );
        assert!(apply_tardos_watermark_7680(&mut valid_lite, &[1.0; 100], 1e-4).is_err());
        assert!(score_tardos_watermark_7680(&valid_lite, &valid_lite, &[1.0; 100]).is_err());

        let mut master = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        for_each_tardos_f32_offset_up_to(master.len(), |idx, off| {
            let v = (((idx * 37) % 101) as f32 - 50.0) * 0.01;
            write_f32_le(&mut master, off, v);
        });
        for_each_tardos_fp16_offset_up_to(master.len(), |idx, off| {
            let v = (((idx * 23) % 89) as f32 - 44.0) * 0.001;
            let bits = f32_to_fp16(v);
            tardos::write_u16_le(&mut master, off, bits);
        });

        let mp_secret = [0x5Au8; 32];
        let c1 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-1");
        let c2 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-2");
        let c3 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-3");
        let c4 = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-4");
        let c_innocent = generate_tardos_code_7680(&mp_secret, "mdl-1", "buyer-innocent");

        for &(start, _, byte_base) in &TARDOS_CASCADE_RANGES {
            for idx in start..start + 16 {
                assert_eq!(c1.chips[idx], 0.0);
                assert_eq!(c1.accusation_weights[idx], 0.0);
            }
            let _ = byte_base;
        }

        let mut b1 = master.clone();
        let mut b2 = master.clone();
        let mut b3 = master.clone();
        let mut b4 = master.clone();
        apply_tardos_watermark_7680(&mut b1, &c1, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b2, &c2, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b3, &c3, 2e-4).unwrap();
        apply_tardos_watermark_7680(&mut b4, &c4, 2e-4).unwrap();

        for &(_, _, byte_base) in &TARDOS_CASCADE_RANGES {
            assert_eq!(
                b1[byte_base..byte_base + 64],
                master[byte_base..byte_base + 64]
            );
        }

        let mut pirated_avg = master.clone();
        let mut pirated_maj = master.clone();
        for_each_tardos_f32_offset_up_to(master.len(), |idx, off| {
            let (v1, v2, v3, v4) = (
                read_f32_wire(&b1, off),
                read_f32_wire(&b2, off),
                read_f32_wire(&b3, off),
                read_f32_wire(&b4, off),
            );
            write_f32_le(&mut pirated_avg, off, 0.25 * (v1 + v2 + v3 + v4));
            let vm = read_f32_wire(&master, off);
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
                [v1, v2, v3, v4][idx & 3]
            };
            write_f32_le(&mut pirated_maj, off, maj);
        });
        for_each_tardos_fp16_offset_up_to(master.len(), |idx, off| {
            let read_h = |buf: &[u8]| fp16_to_f32(tardos::read_u16_le(buf, off));
            let (v1, v2, v3, v4) = (read_h(&b1), read_h(&b2), read_h(&b3), read_h(&b4));
            tardos::write_u16_le(
                &mut pirated_avg,
                off,
                f32_to_fp16(0.25 * (v1 + v2 + v3 + v4)),
            );
            let vm = read_h(&master);
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
                [v1, v2, v3, v4][idx & 3]
            };
            tardos::write_u16_le(&mut pirated_maj, off, f32_to_fp16(maj));
        });

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
            let z_ro_avg = score_tardos_w_readout_fp16(&master, &pirated_avg, code).unwrap();
            assert!(
                z_ro_avg > 5.0,
                "fp16 W_readout colluder Z-score too low: {z_ro_avg:.2}"
            );
        }
        assert!(
            score_tardos_watermark_7680(&master, &pirated_avg, &c_innocent)
                .unwrap()
                .abs()
                < 4.5
        );

        // Verify SO(4) Q_buyer rotation leaves Tardos Z-score detectable via Q_buyer^-1 unrotation
        let ang = 0.7_f32;
        let q = [
            [ang.cos(), -ang.sin(), 0.0, 0.0],
            [ang.sin(), ang.cos(), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut rotated_b1 = b1.clone();
        apply_isometric_gauge_watermark(&mut rotated_b1, &[1.0; 32], &q).unwrap();
        let z_unrot =
            score_tardos_watermark_7680_with_buyer_gauge(&master, &rotated_b1, &c1, Some(&q))
                .unwrap();
        assert!(
            z_unrot > 15.0,
            "Q_buyer^-1 Tardos Z-score too low: {z_unrot:.2}"
        );
        // Verify u_int_ultra Block 1..2 remain unrotated (Block 0 only)
        let (a3_orig, _) = Arena3UltraDeltaWire::ref_from_prefix(&b1[ARENA_PRO_BYTES..]).unwrap();
        let (a3_rot, _) =
            Arena3UltraDeltaWire::ref_from_prefix(&rotated_b1[ARENA_PRO_BYTES..]).unwrap();
        assert_eq!(
            a3_orig.control_bus_ultra_delta.u_int_ultra[1],
            a3_rot.control_bus_ultra_delta.u_int_ultra[1]
        );
        assert_eq!(
            a3_orig.control_bus_ultra_delta.u_int_ultra[2],
            a3_rot.control_bus_ultra_delta.u_int_ultra[2]
        );

        // Phase coding & 224-D structural hash checks
        for tier in 0..4 {
            for blk in 0..STAGE3_KAUTZ_BLOCKS_PER_TIER[tier] {
                let base = kautz_soa_byte_offset(tier, blk);
                for lane in 0..4 {
                    write_f32_le(
                        &mut master,
                        base + (8 + lane) * 4,
                        0.155 + 0.05 * (lane as f32),
                    );
                }
            }
        }
        let hash_pre = extract_structural_hash_224d(&master).unwrap();
        let mut phase_arena = master.clone();
        apply_phase_coding_watermark(&mut phase_arena, &[0x3Cu8; 32], PHASE_CODING_MAX_RAD)
            .unwrap();
        let hash_post = extract_structural_hash_224d(&phase_arena).unwrap();
        assert!((1.0 - structural_cosine_similarity(&hash_pre, &hash_post)) < 1e-12);

        // EMR envelopes & cascade gauge
        let gram_diag = [4.0_f64; 256];
        let envs = compute_emr_masked_tardos_envelopes(&gram_diag, [2e-4, 2e-4]);
        assert_eq!(envs[0], 0.0);
        assert!((envs[16] - 1e-4).abs() < 1e-7);

        let mut gauge_arena = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        let tier0_first = tardos::TARDOS_F32_SPANS[0].byte_base;
        apply_cascade_head_gauge(&mut gauge_arena, 0, &[[1.0_f32; 4]; 32]).unwrap();
        let _ = (
            tier0_first,
            ARENA0_CP1_DIR_OFFSET,
            Cp1InlineDirWire {
                cp1_dir_count: LeU16::new(0),
                flags: LeU16::new(0),
                entries: [[0; 3]; 8],
            },
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_isometric_gauge_preserves_224d_structural_hash(
            kappa in prop::array::uniform16(0.985f32..1.015f32),
            angle1 in -3.0f32..3.0f32,
            angle2 in -3.0f32..3.0f32,
        ) {
            let mut master = alloc::vec![0u8; ARENA_ULTRA_BYTES];
            for_each_tardos_f32_offset_up_to(master.len(), |i, off| {
                let v = (((i * 29 + 7) % 97) as f32 - 48.0) * 0.015;
                write_f32_le(&mut master, off, v);
            });
            for_each_tardos_fp16_offset_up_to(master.len(), |i, off| {
                let v = (((i * 19 + 3) % 73) as f32 - 36.0) * 0.002;
                tardos::write_u16_le(&mut master, off, f32_to_fp16(v));
            });
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
            let mut kappa_32 = [1.0f32; KAUTZ_TOTAL_CHANNELS];
            for i in 0..KAUTZ_TOTAL_CHANNELS {
                kappa_32[i] = kappa[i % 16];
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
            apply_isometric_gauge_watermark(&mut watermarked, &kappa_32, &q).unwrap();

            let f_master = extract_structural_hash_224d(&master).unwrap();
            let f_watermarked = extract_structural_hash_224d(&watermarked).unwrap();
            prop_assert!((structural_cosine_similarity(&f_master, &f_watermarked) - 1.0).abs() < 1e-11);
        }
    }

    #[test]
    fn test_isometric_gauge_prefix_overrides_inline_and_shared_cp1() {
        use zerocopy::byteorder::little_endian::{F32 as LeF32, U32 as LeU32};

        // OOB write_f32_le no-op path
        let mut tiny = [0u8; 4];
        write_f32_le(&mut tiny, 8, 1.0);

        let q_rot = [
            [0.0_f32, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        // Shared CP1 stream: absent, zero count, invalid stream, and valid rotation
        let mut no_cp1 = alloc::vec![0u8; 32];
        apply_isometric_gauge_shared_cp1_stream(&mut no_cp1, &q_rot).unwrap();
        let mut zero_cp1 = b"CP1\0\0\0\0\0".to_vec();
        apply_isometric_gauge_shared_cp1_stream(&mut zero_cp1, &q_rot).unwrap();
        let mut bad_cp1 = b"CP1\0\x01\0\0\0".to_vec();
        assert!(apply_isometric_gauge_shared_cp1_stream(&mut bad_cp1, &q_rot).is_err());

        let mut slot = Cp1ContributorSlot::zero();
        slot.lanes = 2;
        slot.control_idx = 0;
        slot.main_base[0][0] = 2.0;
        slot.main_base[0][4] = 5.0;
        slot.main_base[1][1] = 3.0;
        slot.v_base = [7.0, 0.0, 0.0, 0.0];
        let encoded = encode_cp1_shared_stream(&[slot]).unwrap();
        let mut shared = alloc::vec![0u8; 64];
        shared.extend_from_slice(&encoded);
        apply_isometric_gauge_shared_cp1_stream(&mut shared, &q_rot).unwrap();
        let mut decoded = [Cp1ContributorSlot::zero(); 1];
        assert_eq!(
            decode_cp1_shared_stream_into_slice(&shared[64..], &mut decoded).unwrap(),
            1
        );
        assert!((decoded[0].main_base[0][1] + 2.0).abs() < 1e-6);
        assert!((decoded[0].main_base[0][4] - 5.0).abs() < 1e-6);
        assert!((decoded[0].v_base[0] - 7.0).abs() < 1e-6);

        // Lite arena with prefix_overrides enabled + inline CP1 variants
        let mut lite = alloc::vec![0u8; ARENA_LITE_BYTES];
        {
            let (a0, _) = Arena0LiteWire::mut_from_prefix(&mut lite).unwrap();
            a0.crossbar.pipeline_flags = LeU32::new(PIPE_MAGIC_HI | PIPE_FLAG_V3_MATRYOSHKA);
            a0.crossbar.sona_meta_flags = LeU32::new(SONA_META_CONVOLVER_PREFIX_OVERRIDES);
            a0.control_bus_lite.cp1_dir = Cp1InlineDirWire {
                cp1_dir_count: LeU16::new(2),
                flags: LeU16::new(0),
                entries: [[0; 3]; 8],
            };
            a0.control_bus_lite.cp1_dir.entries[0] = [0, 1, 0xFF];
            a0.control_bus_lite.cp1_dir.entries[1] = [1, 2, 0xFF];
            let words = <[LeF32]>::mut_from_bytes(&mut a0.control_bus_lite.cp1_base).unwrap();
            words[13 + 18] = LeF32::new(1.5);
        }
        apply_isometric_gauge_watermark(&mut lite, &[1.25; KAUTZ_TOTAL_CHANNELS], &q_rot).unwrap();
        let (a0_check, _) = Arena0LiteWire::ref_from_prefix(&lite).unwrap();
        assert!((a0_check.control_bus_lite.u_int_base[0][4].get() - 1.0).abs() < 1e-6);

        // Inline CP1 with all-zero v_base, then invalid lanes (0) hitting break
        {
            let (a0, _) = Arena0LiteWire::mut_from_prefix(&mut lite).unwrap();
            a0.control_bus_lite.cp1_base.fill(0);
            a0.control_bus_lite.cp1_dir.cp1_dir_count = LeU16::new(1);
        }
        apply_isometric_gauge_watermark(&mut lite, &[1.0; KAUTZ_TOTAL_CHANNELS], &q_rot).unwrap();
        {
            let (a0, _) = Arena0LiteWire::mut_from_prefix(&mut lite).unwrap();
            a0.control_bus_lite.cp1_dir.entries[0] = [0, 0, 0xFF];
        }
        apply_isometric_gauge_watermark(&mut lite, &[1.0; KAUTZ_TOTAL_CHANNELS], &q_rot).unwrap();

        // Ultra arena with prefix_overrides enabled and both zero & non-zero fp16 slots
        let mut ultra = alloc::vec![0u8; ARENA_ULTRA_BYTES];
        {
            let (a0, _) = Arena0LiteWire::mut_from_prefix(&mut ultra).unwrap();
            a0.crossbar.pipeline_flags = LeU32::new(PIPE_MAGIC_HI | PIPE_FLAG_V3_MATRYOSHKA);
            a0.crossbar.sona_meta_flags = LeU32::new(SONA_META_CONVOLVER_PREFIX_OVERRIDES);
        }
        let half = U16::new(f32_to_fp16(0.5));
        let (wf, _) =
            LinearReadoutFullWire::mut_from_prefix(&mut ultra[LINEAR_READOUT_BASES[1]..]).unwrap();
        wf.w_ro_and_mods_fp16[296 + 32] = half;
        let (wp, _) =
            LinearReadoutProWire::mut_from_prefix(&mut ultra[LINEAR_READOUT_BASES[2]..]).unwrap();
        wp.w_ro_and_mods_fp16[392 + 32] = half;
        let (wu, _) =
            LinearReadoutUltraWire::mut_from_prefix(&mut ultra[LINEAR_READOUT_BASES[3]..]).unwrap();
        wu.w_ro_and_mods_fp16[488 + 96] = half;
        apply_isometric_gauge_watermark(&mut ultra, &[1.25; KAUTZ_TOTAL_CHANNELS], &q_rot).unwrap();
    }
}
