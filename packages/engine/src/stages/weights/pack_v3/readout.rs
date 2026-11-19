//! `Stage III FIR + Kautz`, `Direct Readout`, `ControlBus Params`, & Legacy Overlay Packers.

use super::gru::{unpack_stage4, write_f32_le};
use crate::stages::stage2_gru::{
    NUM_GRU_BLOCKS, V3ControlBusParams, flat_gru_lane_index, tier_block_groups, tier_fir_segments,
    tier_kautz_banks,
};
use crate::stages::stage3_cab::{V3MatryoshkaModel, V3TierReadout};
use crate::stages::weights::{
    ARENA0_DIRECT_READOUT_BYTES, ARENA0_DIRECT_READOUT_OFFSET, ARENA0_STAGE3_EARLY_FIR_OFFSET,
    ARENA0_STAGE3_KAUTZ_OFFSET, ARENA0_STAGE4_TANK_OFFSET, ARENA1_CTRL_PARAMS_OFFSET,
    ARENA1_DIRECT_READOUT_BYTES, ARENA1_DIRECT_READOUT_OFFSET, ARENA1_STAGE3_EARLY_FIR_OFFSET,
    ARENA1_STAGE3_KAUTZ_OFFSET, ARENA1_STAGE3_PREFIX_OVERRIDE_OFFSET, ARENA1_STAGE4_TANK_OFFSET,
    ARENA2_DIRECT_READOUT_BYTES, ARENA2_DIRECT_READOUT_OFFSET, ARENA2_STAGE2_GRU_OFFSET,
    ARENA2_STAGE3_EARLY_FIR_OFFSET, ARENA2_STAGE3_KAUTZ_OFFSET,
    ARENA2_STAGE3_PREFIX_OVERRIDE_OFFSET, ARENA2_STAGE4_TANK_OFFSET, ARENA3_DIRECT_READOUT_BYTES,
    ARENA3_DIRECT_READOUT_OFFSET, ARENA3_STAGE2_GRU_OFFSET, ARENA3_STAGE3_EARLY_FIR_OFFSET,
    ARENA3_STAGE3_KAUTZ_OFFSET, ARENA3_STAGE3_PREFIX_OVERRIDE_OFFSET, ARENA3_STAGE4_TANK_OFFSET,
    read_f32_le,
};
use crate::types::Tier;

pub(super) const READOUT_OFFSETS: [usize; 4] = [
    ARENA0_DIRECT_READOUT_OFFSET,
    ARENA1_DIRECT_READOUT_OFFSET,
    ARENA2_DIRECT_READOUT_OFFSET,
    ARENA3_DIRECT_READOUT_OFFSET,
];
pub(super) const FIR_OFFSETS: [usize; 4] = [
    ARENA0_STAGE3_EARLY_FIR_OFFSET,
    ARENA1_STAGE3_EARLY_FIR_OFFSET,
    ARENA2_STAGE3_EARLY_FIR_OFFSET,
    ARENA3_STAGE3_EARLY_FIR_OFFSET,
];
pub(super) const KAUTZ_OFFSETS: [usize; 4] = [
    ARENA0_STAGE3_KAUTZ_OFFSET,
    ARENA1_STAGE3_KAUTZ_OFFSET,
    ARENA2_STAGE3_KAUTZ_OFFSET,
    ARENA3_STAGE3_KAUTZ_OFFSET,
];
pub(super) const STAGE4_OFFSETS: [usize; 4] = [
    ARENA0_STAGE4_TANK_OFFSET,
    ARENA1_STAGE4_TANK_OFFSET,
    ARENA2_STAGE4_TANK_OFFSET,
    ARENA3_STAGE4_TANK_OFFSET,
];
pub(super) const PREFIX_OVERRIDE_OFFSETS: [usize; 4] = [
    0,
    ARENA1_STAGE3_PREFIX_OVERRIDE_OFFSET,
    ARENA2_STAGE3_PREFIX_OVERRIDE_OFFSET,
    ARENA3_STAGE3_PREFIX_OVERRIDE_OFFSET,
];

#[inline]
fn any_nonzero_bytes(slice: &[u8]) -> bool {
    slice.iter().any(|&b| b != 0)
}

pub(super) fn pack_tier_readout_slice(buf: &mut [u8], tier: Tier, r: &V3TierReadout) {
    let t = tier as usize;
    let fir_off = FIR_OFFSETS[t];
    let fir_start = t * 16;
    for i in 0..16 {
        write_f32_le(buf, fir_off + i * 4, r.fir_l[fir_start + i]);
        write_f32_le(buf, fir_off + 64 + i * 4, r.fir_r[fir_start + i]);
    }
    let k_off = KAUTZ_OFFSETS[t];
    let k_start = t * 4;
    for m in 0..4 {
        let bg = r.kautz_b_gamma[k_start + m];
        if bg > 0.0 {
            write_f32_le(buf, k_off + 48 + m * 4, bg);
        }
        let (cl_re, cl_im) = r.kautz_c_l[k_start + m];
        let (cr_re, cr_im) = r.kautz_c_r[k_start + m];
        write_f32_le(buf, k_off + 64 + m * 4, cl_re);
        write_f32_le(buf, k_off + 80 + m * 4, cl_im);
        write_f32_le(buf, k_off + 96 + m * 4, cr_re);
        write_f32_le(buf, k_off + 112 + m * 4, cr_im);
    }

    let ro_off = READOUT_OFFSETS[t];
    for i in 0..8 {
        write_f32_le(buf, ro_off + i * 4, r.svf_direct_taps_l[i]);
        write_f32_le(buf, ro_off + 32 + i * 4, r.svf_direct_taps_r[i]);
    }
    write_f32_le(buf, ro_off + 64, r.pre_cab_u_gain[0]);
    write_f32_le(buf, ro_off + 68, r.pre_cab_u_gain[1]);
    write_f32_le(buf, ro_off + 72, r.pre_cab_hp_gain[0]);
    write_f32_le(buf, ro_off + 76, r.pre_cab_hp_gain[1]);
    write_f32_le(buf, ro_off + 80, r.direct_s2_u_gain[0]);
    write_f32_le(buf, ro_off + 84, r.direct_s2_u_gain[1]);
    write_f32_le(buf, ro_off + 88, r.direct_s2_hp_gain[0]);
    write_f32_le(buf, ro_off + 92, r.direct_s2_hp_gain[1]);

    let mut w_off = ro_off + 96;
    for ch in 0..2 {
        for b in 0..NUM_GRU_BLOCKS {
            for g in 0..tier_block_groups(tier, b) {
                for l in 0..4 {
                    write_f32_le(buf, w_off, r.pre_cab_w[ch][flat_gru_lane_index(b, g, l)]);
                    w_off += 4;
                }
            }
        }
    }
    for ch in 0..2 {
        for b in 0..NUM_GRU_BLOCKS {
            for g in 0..tier_block_groups(tier, b) {
                for l in 0..4 {
                    write_f32_le(buf, w_off, r.direct_s2_w[ch][flat_gru_lane_index(b, g, l)]);
                    w_off += 4;
                }
            }
        }
    }
    if t >= 1 && buf.len() >= PREFIX_OVERRIDE_OFFSETS[t] + 192 * t {
        let ov = PREFIX_OVERRIDE_OFFSETS[t];
        let n_fir = t * 16;
        let n_kautz = t * 4;
        for i in 0..n_fir {
            write_f32_le(buf, ov + i * 4, r.fir_l[i]);
            write_f32_le(buf, ov + (n_fir + i) * 4, r.fir_r[i]);
        }
        let k_ov = ov + 2 * n_fir * 4;
        for m in 0..n_kautz {
            write_f32_le(buf, k_ov + m * 4, r.kautz_c_l[m].0);
            write_f32_le(buf, k_ov + (n_kautz + m) * 4, r.kautz_c_l[m].1);
            write_f32_le(buf, k_ov + (2 * n_kautz + m) * 4, r.kautz_c_r[m].0);
            write_f32_le(buf, k_ov + (3 * n_kautz + m) * 4, r.kautz_c_r[m].1);
        }
    }
}

pub(super) fn unpack_tier_readout_slice(
    buf: &[u8],
    tier: Tier,
    has_prefix_overrides: bool,
    r: &mut V3TierReadout,
) {
    let t = tier as usize;
    let fir_cnt = tier_fir_segments(tier);
    for (seg, &fir_off) in FIR_OFFSETS.iter().enumerate().take(fir_cnt) {
        if buf.len() >= fir_off + 128 {
            let fir_start = seg * 16;
            for i in 0..16 {
                r.fir_l[fir_start + i] = read_f32_le(buf, fir_off + i * 4);
                r.fir_r[fir_start + i] = read_f32_le(buf, fir_off + 64 + i * 4);
            }
        }
    }
    let k_cnt = tier_kautz_banks(tier);
    for (bank, &k_off) in KAUTZ_OFFSETS.iter().enumerate().take(k_cnt) {
        if buf.len() >= k_off + 128 {
            let k_start = bank * 4;
            for m in 0..4 {
                r.kautz_b_gamma[k_start + m] = read_f32_le(buf, k_off + 48 + m * 4);
                r.kautz_c_l[k_start + m] = (
                    read_f32_le(buf, k_off + 64 + m * 4),
                    read_f32_le(buf, k_off + 80 + m * 4),
                );
                r.kautz_c_r[k_start + m] = (
                    read_f32_le(buf, k_off + 96 + m * 4),
                    read_f32_le(buf, k_off + 112 + m * 4),
                );
            }
        }
    }

    let ro_off = READOUT_OFFSETS[t];
    for i in 0..8 {
        r.svf_direct_taps_l[i] = read_f32_le(buf, ro_off + i * 4);
        r.svf_direct_taps_r[i] = read_f32_le(buf, ro_off + 32 + i * 4);
    }
    r.pre_cab_u_gain = [read_f32_le(buf, ro_off + 64), read_f32_le(buf, ro_off + 68)];
    r.pre_cab_hp_gain = [read_f32_le(buf, ro_off + 72), read_f32_le(buf, ro_off + 76)];
    r.direct_s2_u_gain = [read_f32_le(buf, ro_off + 80), read_f32_le(buf, ro_off + 84)];
    r.direct_s2_hp_gain = [read_f32_le(buf, ro_off + 88), read_f32_le(buf, ro_off + 92)];

    let mut w_off = ro_off + 96;
    for ch in 0..2 {
        for b in 0..NUM_GRU_BLOCKS {
            for g in 0..tier_block_groups(tier, b) {
                for l in 0..4 {
                    r.pre_cab_w[ch][flat_gru_lane_index(b, g, l)] = read_f32_le(buf, w_off);
                    w_off += 4;
                }
            }
        }
    }
    for ch in 0..2 {
        for b in 0..NUM_GRU_BLOCKS {
            for g in 0..tier_block_groups(tier, b) {
                for l in 0..4 {
                    r.direct_s2_w[ch][flat_gru_lane_index(b, g, l)] = read_f32_le(buf, w_off);
                    w_off += 4;
                }
            }
        }
    }
    if has_prefix_overrides && t >= 1 && buf.len() >= PREFIX_OVERRIDE_OFFSETS[t] + 192 * t {
        let ov = PREFIX_OVERRIDE_OFFSETS[t];
        let n_fir = t * 16;
        let n_kautz = t * 4;
        for i in 0..n_fir {
            r.fir_l[i] = read_f32_le(buf, ov + i * 4);
            r.fir_r[i] = read_f32_le(buf, ov + (n_fir + i) * 4);
        }
        let k_ov = ov + 2 * n_fir * 4;
        for m in 0..n_kautz {
            r.kautz_c_l[m] = (
                read_f32_le(buf, k_ov + m * 4),
                read_f32_le(buf, k_ov + (n_kautz + m) * 4),
            );
            r.kautz_c_r[m] = (
                read_f32_le(buf, k_ov + (2 * n_kautz + m) * 4),
                read_f32_le(buf, k_ov + (3 * n_kautz + m) * 4),
            );
        }
    }
}

pub(super) fn pack_ctrl_params(buf: &mut [u8], ctrl: &V3ControlBusParams) {
    let mut off = ARENA1_CTRL_PARAMS_OFFSET;
    for arr in [
        &ctrl.w_env_fc_oct,
        &ctrl.w_env_k,
        &ctrl.w_tone_fc_oct,
        &ctrl.w_tone_k,
        &ctrl.w_tank_psi,
    ] {
        for &v in arr {
            write_f32_le(buf, off, v);
            off += 4;
        }
    }
    for eq in &ctrl.w_tone_eq {
        for &v in eq {
            write_f32_le(buf, off, v);
            off += 4;
        }
    }
    for blk in &ctrl.w_film_gain {
        for lane in blk {
            for &v in lane {
                write_f32_le(buf, off, v);
                off += 4;
            }
        }
    }
    for blk in &ctrl.w_film_bn {
        for (block_idx, lane) in blk.iter().enumerate() {
            for (c, &v) in lane.iter().enumerate() {
                // Keep Pro (`b=1, c=4..8`) and Ultra (`b=0, c=8..12`) weights out of Arena 1
                // so lower-tier arena prefixes remain bit-identical under higher-tier keying.
                let is_upper_tier = (block_idx == 1 && (4..8).contains(&c))
                    || (block_idx == 0 && (8..12).contains(&c));
                write_f32_le(buf, off, if is_upper_tier { 0.0 } else { v });
                off += 4;
            }
        }
    }
    if buf.len() >= ARENA2_STAGE2_GRU_OFFSET + 3_840 {
        for (k, blk) in ctrl.w_film_bn.iter().enumerate() {
            for l in 0..4 {
                write_f32_le(
                    buf,
                    ARENA2_STAGE2_GRU_OFFSET + 3_712 + ((k << 2) | l) * 4,
                    blk[1][4 + l],
                );
            }
        }
    }
    if buf.len() >= ARENA3_STAGE2_GRU_OFFSET + 1_024 {
        for (k, blk) in ctrl.w_film_bn.iter().enumerate() {
            for l in 0..4 {
                write_f32_le(
                    buf,
                    ARENA3_STAGE2_GRU_OFFSET + 896 + ((k << 2) | l) * 4,
                    blk[0][8 + l],
                );
            }
        }
    }
}

pub(super) fn unpack_ctrl_params(buf: &[u8]) -> V3ControlBusParams {
    if buf.len() < ARENA1_CTRL_PARAMS_OFFSET + 2_560 {
        return V3ControlBusParams::identity();
    }
    let mut p = V3ControlBusParams::identity();
    let mut off = ARENA1_CTRL_PARAMS_OFFSET;
    for arr in [
        &mut p.w_env_fc_oct,
        &mut p.w_env_k,
        &mut p.w_tone_fc_oct,
        &mut p.w_tone_k,
        &mut p.w_tank_psi,
    ] {
        for slot in arr.iter_mut() {
            *slot = read_f32_le(buf, off);
            off += 4;
        }
    }
    for eq in &mut p.w_tone_eq {
        for slot in eq.iter_mut() {
            *slot = read_f32_le(buf, off);
            off += 4;
        }
    }
    for blk in &mut p.w_film_gain {
        for lane in blk.iter_mut() {
            for slot in lane.iter_mut() {
                *slot = read_f32_le(buf, off);
                off += 4;
            }
        }
    }
    for blk in &mut p.w_film_bn {
        for lane in blk.iter_mut() {
            for slot in lane.iter_mut() {
                *slot = read_f32_le(buf, off);
                off += 4;
            }
        }
    }
    for blk in &mut p.w_film_bn {
        blk[1][4..8].fill(0.0);
        blk[0][8..12].fill(0.0);
    }
    if buf.len() >= ARENA2_STAGE2_GRU_OFFSET + 3_840 {
        for (k, blk) in p.w_film_bn.iter_mut().enumerate() {
            for l in 0..4 {
                blk[1][4 + l] =
                    read_f32_le(buf, ARENA2_STAGE2_GRU_OFFSET + 3_712 + ((k << 2) | l) * 4);
            }
        }
    }
    if buf.len() >= ARENA3_STAGE2_GRU_OFFSET + 1_024 {
        for (k, blk) in p.w_film_bn.iter_mut().enumerate() {
            for l in 0..4 {
                blk[0][8 + l] =
                    read_f32_le(buf, ARENA3_STAGE2_GRU_OFFSET + 896 + ((k << 2) | l) * 4);
            }
        }
    }
    p.clamp_in_place();
    p
}

pub(super) fn overlay_unflagged_tier_slices(arena: &[u8], model: &mut V3MatryoshkaModel) {
    for (t_idx, &tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .iter()
        .enumerate()
    {
        if arena.len() < tier.expected_arena_bytes() {
            break;
        }
        let fir_off = FIR_OFFSETS[t_idx];
        if any_nonzero_bytes(&arena[fir_off..fir_off + 128]) {
            let fir_start = t_idx * 16;
            for i in 0..16 {
                model.readouts[t_idx].fir_l[fir_start + i] = read_f32_le(arena, fir_off + i * 4);
                model.readouts[t_idx].fir_r[fir_start + i] =
                    read_f32_le(arena, fir_off + 64 + i * 4);
            }
        }
        let k_off = KAUTZ_OFFSETS[t_idx];
        if any_nonzero_bytes(&arena[k_off..k_off + 128]) {
            let k_start = t_idx * 4;
            for m in 0..4 {
                let r_low = read_f32_le(arena, k_off + m * 4);
                let dr = read_f32_le(arena, k_off + 16 + m * 4);
                let th = read_f32_le(arena, k_off + 32 + m * 4);
                let r = r_low + 0.5 * dr;
                if r > 0.0 {
                    model.kautz_poles[k_start + m] = (r, th);
                }
                model.readouts[t_idx].kautz_b_gamma[k_start + m] =
                    read_f32_le(arena, k_off + 48 + m * 4);
                model.readouts[t_idx].kautz_c_l[k_start + m] = (
                    read_f32_le(arena, k_off + 64 + m * 4),
                    read_f32_le(arena, k_off + 80 + m * 4),
                );
                model.readouts[t_idx].kautz_c_r[k_start + m] = (
                    read_f32_le(arena, k_off + 96 + m * 4),
                    read_f32_le(arena, k_off + 112 + m * 4),
                );
            }
        }
        let s4_off = STAGE4_OFFSETS[t_idx];
        if any_nonzero_bytes(&arena[s4_off..s4_off + 64]) {
            model.stage4[t_idx] = unpack_stage4(arena, s4_off);
        }
        let ro_off = READOUT_OFFSETS[t_idx];
        let ro_bytes = [
            ARENA0_DIRECT_READOUT_BYTES,
            ARENA1_DIRECT_READOUT_BYTES,
            ARENA2_DIRECT_READOUT_BYTES,
            ARENA3_DIRECT_READOUT_BYTES,
        ][t_idx];
        if any_nonzero_bytes(&arena[ro_off..ro_off + ro_bytes]) {
            unpack_tier_readout_slice(arena, tier, false, &mut model.readouts[t_idx]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::weights::{ARENA_FULL_BYTES, ARENA_ULTRA_BYTES};
    use alloc::vec;

    #[test]
    fn test_readout_and_ctrl_params_roundtrip() {
        let mut buf = vec![0u8; ARENA_ULTRA_BYTES];
        let mut ro = V3TierReadout::default();
        ro.pre_cab_w[0][0] = 0.75;
        ro.fir_l[20] = 0.35;
        ro.kautz_c_r[4] = (0.15, -0.05);
        pack_tier_readout_slice(&mut buf, Tier::Full, &ro);
        let mut decoded = V3TierReadout::zeros();
        unpack_tier_readout_slice(&buf, Tier::Full, true, &mut decoded);
        assert_eq!(decoded.pre_cab_w[0][0], 0.75);
        assert_eq!(decoded.fir_l[0], 1.0);

        let mut ctrl = V3ControlBusParams::identity();
        ctrl.w_film_gain[1][2][2] = 0.45;
        ctrl.w_film_bn[0][0][1] = 0.25;
        ctrl.w_film_bn[0][1][5] = 0.35;
        ctrl.w_film_bn[0][0][9] = -0.55;
        pack_ctrl_params(&mut buf, &ctrl);
        let unpacked_ultra = unpack_ctrl_params(&buf);
        assert_eq!(unpacked_ultra.w_film_gain[1][2][2], 0.45);
        assert_eq!(unpacked_ultra.w_film_bn[0][0][1], 0.25);
        assert_eq!(unpacked_ultra.w_film_bn[0][1][5], 0.35);
        assert_eq!(unpacked_ultra.w_film_bn[0][0][9], -0.55);
        let unpacked_full = unpack_ctrl_params(&buf[..ARENA_FULL_BYTES]);
        assert_eq!(unpacked_full.w_film_bn[0][0][1], 0.25);
        assert_eq!(unpacked_full.w_film_bn[0][1][5], 0.0);
        assert_eq!(unpacked_full.w_film_bn[0][0][9], 0.0);
    }
}
