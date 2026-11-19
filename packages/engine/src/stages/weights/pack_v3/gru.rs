//! `Stage I`, `Stage II GRU`, and `Stage IV FDN` Binary Slice Packers (`v3` Arena `0..3`).

use crate::stages::stage1_svf::V3Stage1SvfParams;
use crate::stages::stage2_gru::{
    MAX_BLOCK_LANES, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0, V3GruBlockCold, V3InputFeaturesParams,
};
use crate::stages::stage4_tank::v3_fdn::{V3_TANK_LINES, V3Stage4TankParams};
use crate::stages::weights::{
    ARENA0_STAGE2_GRU_OFFSET, ARENA1_STAGE2_GRU_OFFSET, ARENA2_STAGE2_GRU_OFFSET,
    ARENA3_STAGE2_GRU_OFFSET, read_f32_le, read_u32_le,
};
use crate::types::Tier;

/// Byte footprint of one `4`-lane GRU group's feed-forward weights & biases (`320 B`).
pub const GRU_FEED_BYTES: usize = 320;
/// Byte footprint of one `4×4` tile triple (`w_*_z/r/n`, `192 B`).
pub const GRU_TILE_TRIPLE_BYTES: usize = 192;

/// Writes a little-endian `f32` into `buf[offset..offset + 4]` when in bounds.
#[inline(always)]
pub fn write_f32_le(buf: &mut [u8], offset: usize, val: f32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }
}

/// Writes a little-endian `u32` into `buf[offset..offset + 4]` when in bounds.
#[inline(always)]
pub fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }
}

fn write_gru_group_feed(buf: &mut [u8], base: usize, blk: &V3GruBlockCold, g: usize) {
    let lane0 = g << 2;
    let mut off = base;
    for mat in [&blk.w_in_z, &blk.w_in_r, &blk.w_in_n] {
        for f in 0..NUM_IN_FEATS_B0 {
            for l in 0..4 {
                write_f32_le(buf, off, mat[f][lane0 + l]);
                off += 4;
            }
        }
    }
    for vec in [&blk.b_z, &blk.b_r, &blk.b_n, &blk.k_sag, &blk.k_quad] {
        for l in 0..4 {
            write_f32_le(buf, off, vec[lane0 + l]);
            off += 4;
        }
    }
}

fn read_gru_group_feed(buf: &[u8], base: usize, blk: &mut V3GruBlockCold, g: usize) {
    let lane0 = g << 2;
    let mut off = base;
    for mat in [&mut blk.w_in_z, &mut blk.w_in_r, &mut blk.w_in_n] {
        for f in 0..NUM_IN_FEATS_B0 {
            for l in 0..4 {
                mat[f][lane0 + l] = read_f32_le(buf, off);
                off += 4;
            }
        }
    }
    for vec in [
        &mut blk.b_z,
        &mut blk.b_r,
        &mut blk.b_n,
        &mut blk.k_sag,
        &mut blk.k_quad,
    ] {
        for l in 0..4 {
            vec[lane0 + l] = read_f32_le(buf, off);
            off += 4;
        }
    }
}

fn write_gru_tile_triple(
    buf: &mut [u8],
    base: usize,
    z: &[[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    r: &[[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    n: &[[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    dst_g: usize,
    src_g: usize,
) {
    let mut off = base;
    for mat in [z, r, n] {
        for row in 0..4 {
            for col in 0..4 {
                write_f32_le(buf, off, mat[(dst_g << 2) + row][(src_g << 2) + col]);
                off += 4;
            }
        }
    }
}

fn read_gru_tile_triple(
    buf: &[u8],
    base: usize,
    z: &mut [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    r: &mut [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    n: &mut [[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    dst_g: usize,
    src_g: usize,
) {
    let mut off = base;
    for mat in [z, r, n] {
        for row in 0..4 {
            for col in 0..4 {
                mat[(dst_g << 2) + row][(src_g << 2) + col] = read_f32_le(buf, off);
                off += 4;
            }
        }
    }
}

pub(super) fn pack_1x_lite_full_gru(buf: &mut [u8], blocks: &[V3GruBlockCold; NUM_GRU_BLOCKS]) {
    let mut off = ARENA0_STAGE2_GRU_OFFSET;
    write_gru_group_feed(buf, off, &blocks[0], 0);
    off += GRU_FEED_BYTES;
    write_gru_tile_triple(
        buf,
        off,
        &blocks[0].w_rec_z,
        &blocks[0].w_rec_r,
        &blocks[0].w_rec_n,
        0,
        0,
    );
    off += GRU_TILE_TRIPLE_BYTES;
    for blk in blocks.iter().skip(1) {
        write_gru_group_feed(buf, off, blk, 0);
        off += GRU_FEED_BYTES;
        write_gru_tile_triple(buf, off, &blk.w_prev_z, &blk.w_prev_r, &blk.w_prev_n, 0, 0);
        off += GRU_TILE_TRIPLE_BYTES;
        write_gru_tile_triple(buf, off, &blk.w_rec_z, &blk.w_rec_r, &blk.w_rec_n, 0, 0);
        off += GRU_TILE_TRIPLE_BYTES;
    }

    let mut f_off = ARENA1_STAGE2_GRU_OFFSET;
    write_gru_group_feed(buf, f_off, &blocks[0], 1);
    f_off += GRU_FEED_BYTES;
    for src_g in 0..2 {
        write_gru_tile_triple(
            buf,
            f_off,
            &blocks[0].w_rec_z,
            &blocks[0].w_rec_r,
            &blocks[0].w_rec_n,
            1,
            src_g,
        );
        f_off += GRU_TILE_TRIPLE_BYTES;
    }
}

pub(super) fn unpack_1x_lite_full_gru(buf: &[u8], tier: Tier) -> [V3GruBlockCold; NUM_GRU_BLOCKS] {
    let mut blocks = [V3GruBlockCold::default(); NUM_GRU_BLOCKS];
    let mut off = ARENA0_STAGE2_GRU_OFFSET;
    read_gru_group_feed(buf, off, &mut blocks[0], 0);
    off += GRU_FEED_BYTES;
    read_gru_tile_triple(
        buf,
        off,
        &mut blocks[0].w_rec_z,
        &mut blocks[0].w_rec_r,
        &mut blocks[0].w_rec_n,
        0,
        0,
    );
    off += GRU_TILE_TRIPLE_BYTES;
    for blk in blocks.iter_mut().skip(1) {
        read_gru_group_feed(buf, off, blk, 0);
        off += GRU_FEED_BYTES;
        read_gru_tile_triple(
            buf,
            off,
            &mut blk.w_prev_z,
            &mut blk.w_prev_r,
            &mut blk.w_prev_n,
            0,
            0,
        );
        off += GRU_TILE_TRIPLE_BYTES;
        read_gru_tile_triple(
            buf,
            off,
            &mut blk.w_rec_z,
            &mut blk.w_rec_r,
            &mut blk.w_rec_n,
            0,
            0,
        );
        off += GRU_TILE_TRIPLE_BYTES;
    }
    if (tier as usize) >= 1
        && buf.len() >= ARENA1_STAGE2_GRU_OFFSET + GRU_FEED_BYTES + 2 * GRU_TILE_TRIPLE_BYTES
    {
        let mut f_off = ARENA1_STAGE2_GRU_OFFSET;
        read_gru_group_feed(buf, f_off, &mut blocks[0], 1);
        f_off += GRU_FEED_BYTES;
        for src_g in 0..2 {
            read_gru_tile_triple(
                buf,
                f_off,
                &mut blocks[0].w_rec_z,
                &mut blocks[0].w_rec_r,
                &mut blocks[0].w_rec_n,
                1,
                src_g,
            );
            f_off += GRU_TILE_TRIPLE_BYTES;
        }
    }
    blocks
}

pub(super) fn pack_2x_pro_ultra_gru(buf: &mut [u8], blocks: &[V3GruBlockCold; NUM_GRU_BLOCKS]) {
    let mut off = ARENA2_STAGE2_GRU_OFFSET;
    let pro_bg = [2usize, 2, 1];
    for (b, &g_cnt) in pro_bg.iter().enumerate() {
        for g in 0..g_cnt {
            write_gru_group_feed(buf, off, &blocks[b], g);
            off += GRU_FEED_BYTES;
            if b > 0 {
                for src_g in 0..=g {
                    write_gru_tile_triple(
                        buf,
                        off,
                        &blocks[b].w_prev_z,
                        &blocks[b].w_prev_r,
                        &blocks[b].w_prev_n,
                        g,
                        src_g,
                    );
                    off += GRU_TILE_TRIPLE_BYTES;
                }
            }
            for src_g in 0..=g {
                write_gru_tile_triple(
                    buf,
                    off,
                    &blocks[b].w_rec_z,
                    &blocks[b].w_rec_r,
                    &blocks[b].w_rec_n,
                    g,
                    src_g,
                );
                off += GRU_TILE_TRIPLE_BYTES;
            }
        }
    }

    let mut u_off = ARENA3_STAGE2_GRU_OFFSET;
    write_gru_group_feed(buf, u_off, &blocks[0], 2);
    u_off += GRU_FEED_BYTES;
    for src_g in 0..3 {
        write_gru_tile_triple(
            buf,
            u_off,
            &blocks[0].w_rec_z,
            &blocks[0].w_rec_r,
            &blocks[0].w_rec_n,
            2,
            src_g,
        );
        u_off += GRU_TILE_TRIPLE_BYTES;
    }
}

pub(super) fn unpack_2x_pro_ultra_gru(buf: &[u8], tier: Tier) -> [V3GruBlockCold; NUM_GRU_BLOCKS] {
    let mut blocks = [V3GruBlockCold::default(); NUM_GRU_BLOCKS];
    let mut off = ARENA2_STAGE2_GRU_OFFSET;
    let pro_bg = [2usize, 2, 1];
    for (b, &g_cnt) in pro_bg.iter().enumerate() {
        for g in 0..g_cnt {
            read_gru_group_feed(buf, off, &mut blocks[b], g);
            off += GRU_FEED_BYTES;
            if b > 0 {
                for src_g in 0..=g {
                    read_gru_tile_triple(
                        buf,
                        off,
                        &mut blocks[b].w_prev_z,
                        &mut blocks[b].w_prev_r,
                        &mut blocks[b].w_prev_n,
                        g,
                        src_g,
                    );
                    off += GRU_TILE_TRIPLE_BYTES;
                }
            }
            for src_g in 0..=g {
                read_gru_tile_triple(
                    buf,
                    off,
                    &mut blocks[b].w_rec_z,
                    &mut blocks[b].w_rec_r,
                    &mut blocks[b].w_rec_n,
                    g,
                    src_g,
                );
                off += GRU_TILE_TRIPLE_BYTES;
            }
        }
    }
    if tier == Tier::Ultra
        && buf.len() >= ARENA3_STAGE2_GRU_OFFSET + GRU_FEED_BYTES + 3 * GRU_TILE_TRIPLE_BYTES
    {
        let mut u_off = ARENA3_STAGE2_GRU_OFFSET;
        read_gru_group_feed(buf, u_off, &mut blocks[0], 2);
        u_off += GRU_FEED_BYTES;
        for src_g in 0..3 {
            read_gru_tile_triple(
                buf,
                u_off,
                &mut blocks[0].w_rec_z,
                &mut blocks[0].w_rec_r,
                &mut blocks[0].w_rec_n,
                2,
                src_g,
            );
            u_off += GRU_TILE_TRIPLE_BYTES;
        }
    }
    blocks
}

pub(super) fn pack_stage1_and_feat(
    buf: &mut [u8],
    offset: usize,
    svf: &V3Stage1SvfParams,
    feat: &V3InputFeaturesParams,
) {
    let vals = [
        svf.base_g,
        svf.sec2_g_ratio,
        svf.env_oct_sens,
        svf.env_comp,
        svf.det_hp_mix,
        svf.damp_k1,
        svf.damp_k2,
        svf.sec2_bp_feed,
    ];
    for (i, &v) in vals.iter().enumerate() {
        write_f32_le(buf, offset + i * 4, v);
    }
    for (i, &v) in svf.stage2_feed_taps.iter().enumerate() {
        write_f32_le(buf, offset + 32 + i * 4, v);
    }
    let f_vals = [
        feat.in_hp_beta,
        feat.fast_atk_beta,
        feat.fast_rel_beta,
        feat.slow_atk_beta,
        feat.slow_rel_beta,
        feat.slow_det_comp,
        feat.pinch_k,
    ];
    for (i, &v) in f_vals.iter().enumerate() {
        write_f32_le(buf, offset + 64 + i * 4, v);
    }
    let svf_ext = [
        svf.atk_alpha,
        svf.rel_alpha,
        svf.det_lp_pole,
        svf.sec2_hp_feed,
        svf.stereo_link,
        svf.g_min,
        svf.g_max,
    ];
    for (i, &v) in svf_ext.iter().enumerate() {
        write_f32_le(buf, offset + 92 + i * 4, v);
    }
    write_f32_le(buf, offset + 120, feat.key_link);
    write_f32_le(buf, offset + 124, svf.feat_lp2_mix);
}

pub(super) fn unpack_stage1_and_feat(
    buf: &[u8],
    offset: usize,
) -> (V3Stage1SvfParams, V3InputFeaturesParams) {
    let mut svf = V3Stage1SvfParams::identity_bypass();
    svf.base_g = read_f32_le(buf, offset);
    svf.sec2_g_ratio = read_f32_le(buf, offset + 4);
    svf.env_oct_sens = read_f32_le(buf, offset + 8);
    svf.env_comp = read_f32_le(buf, offset + 12);
    svf.det_hp_mix = read_f32_le(buf, offset + 16);
    svf.damp_k1 = read_f32_le(buf, offset + 20);
    svf.damp_k2 = read_f32_le(buf, offset + 24);
    svf.sec2_bp_feed = read_f32_le(buf, offset + 28);
    for i in 0..8 {
        svf.stage2_feed_taps[i] = read_f32_le(buf, offset + 32 + i * 4);
    }
    let g_max_stored = read_f32_le(buf, offset + 116);
    if g_max_stored > 0.0 {
        svf.atk_alpha = read_f32_le(buf, offset + 92);
        svf.rel_alpha = read_f32_le(buf, offset + 96);
        svf.det_lp_pole = read_f32_le(buf, offset + 100);
        svf.sec2_hp_feed = read_f32_le(buf, offset + 104);
        svf.stereo_link = read_f32_le(buf, offset + 108);
        svf.g_min = read_f32_le(buf, offset + 112);
        svf.g_max = g_max_stored;
    }
    svf.feat_lp2_mix = {
        let v = read_f32_le(buf, offset + 124);
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let feat = V3InputFeaturesParams {
        in_hp_beta: read_f32_le(buf, offset + 64),
        fast_atk_beta: read_f32_le(buf, offset + 68),
        fast_rel_beta: read_f32_le(buf, offset + 72),
        slow_atk_beta: read_f32_le(buf, offset + 76),
        slow_rel_beta: read_f32_le(buf, offset + 80),
        slow_det_comp: read_f32_le(buf, offset + 84),
        pinch_k: read_f32_le(buf, offset + 88),
        key_link: {
            let kl = read_f32_le(buf, offset + 120);
            if kl.is_finite() {
                kl.clamp(0.0, 1.0)
            } else {
                0.0
            }
        },
        ..V3InputFeaturesParams::default()
    };
    (svf, feat)
}

pub(super) fn pack_stage4(buf: &mut [u8], offset: usize, p: &V3Stage4TankParams) {
    for l in 0..V3_TANK_LINES {
        write_u32_le(buf, offset + l * 4, p.delay_lengths[l] as u32);
    }
    let floats = [
        p.g_loop,
        p.sigma_knee,
        p.base_psi,
        p.inject_gain,
        p.dry_mix,
        p.wet_mix,
        p.r_dc,
        p.vca_thresh,
        p.vca_kappa,
        p.vca_min_gain,
        p.vca_att,
        p.vca_rel,
    ];
    for (i, &v) in floats.iter().enumerate() {
        write_f32_le(buf, offset + 16 + i * 4, v);
    }
}

pub(super) fn unpack_stage4(buf: &[u8], offset: usize) -> V3Stage4TankParams {
    let mut d = [67usize; V3_TANK_LINES];
    for l in 0..V3_TANK_LINES {
        d[l] = read_u32_le(buf, offset + l * 4) as usize;
    }
    let mut p = V3Stage4TankParams {
        delay_lengths: d,
        g_loop: read_f32_le(buf, offset + 16),
        sigma_knee: read_f32_le(buf, offset + 20),
        base_psi: read_f32_le(buf, offset + 24),
        inject_gain: read_f32_le(buf, offset + 28),
        dry_mix: read_f32_le(buf, offset + 32),
        wet_mix: read_f32_le(buf, offset + 36),
        r_dc: read_f32_le(buf, offset + 40),
        vca_thresh: read_f32_le(buf, offset + 44),
        vca_kappa: read_f32_le(buf, offset + 48),
        vca_min_gain: read_f32_le(buf, offset + 52),
        vca_att: read_f32_le(buf, offset + 56),
        vca_rel: read_f32_le(buf, offset + 60),
        ..V3Stage4TankParams::identity()
    };
    p.clamp_in_place();
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::weights::{
        ARENA_ULTRA_BYTES, ARENA0_STAGE1_PRE_OFFSET, ARENA0_STAGE4_TANK_OFFSET,
    };
    use alloc::vec;

    #[test]
    fn test_pack_unpack_gru_stage1_stage4_roundtrip() {
        let mut buf = vec![0u8; ARENA_ULTRA_BYTES];
        let mut blks = [V3GruBlockCold::default(); NUM_GRU_BLOCKS];
        blks[0].w_in_n[0][0] = 0.5;
        blks[0].w_in_n[0][4] = -0.35;
        blks[0].w_in_n[0][8] = 0.42;
        blks[1].w_prev_n[0][0] = 0.25;
        pack_1x_lite_full_gru(&mut buf, &blks);
        pack_2x_pro_ultra_gru(&mut buf, &blks);
        let u1 = unpack_1x_lite_full_gru(&buf, Tier::Full);
        let u2 = unpack_2x_pro_ultra_gru(&buf, Tier::Ultra);
        assert_eq!(u1[0].w_in_n[0][4], -0.35);
        assert_eq!(u2[0].w_in_n[0][8], 0.42);
        assert_eq!(u2[1].w_prev_n[0][0], 0.25);

        let mut svf = V3Stage1SvfParams::identity_bypass();
        svf.base_g = 0.12;
        svf.feat_lp2_mix = 0.875;
        let feat = V3InputFeaturesParams {
            pinch_k: 140.0,
            key_link: 0.625,
            ..V3InputFeaturesParams::default()
        };
        pack_stage1_and_feat(&mut buf, ARENA0_STAGE1_PRE_OFFSET, &svf, &feat);
        let (us, uf) = unpack_stage1_and_feat(&buf, ARENA0_STAGE1_PRE_OFFSET);
        assert_eq!(us.base_g, 0.12);
        assert_eq!(us.feat_lp2_mix, 0.875);
        assert_eq!(uf.pinch_k, 140.0);
        assert_eq!(uf.key_link, 0.625);

        let s4 = V3Stage4TankParams::identity();
        pack_stage4(&mut buf, ARENA0_STAGE4_TANK_OFFSET, &s4);
        assert_eq!(unpack_stage4(&buf, ARENA0_STAGE4_TANK_OFFSET), s4);
    }
}
