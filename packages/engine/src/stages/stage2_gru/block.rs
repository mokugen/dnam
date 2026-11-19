//! Stage II Block-Lower-Triangular Zero-DC Matryoshka `GRU` Blocks & Stereo Execution (`stage2_gru/block.rs`).

use super::features::V3InputFeaturesState;
use super::math_pade::{
    sigmoid_pade5_pair_vec4, sigmoid_pade5_vec4, tanh_pade5_pair_vec4, tanh_pade5_vec4,
};
use super::zoh::V3GruBlockCold;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_vec4};
use crate::types::Tier;

/// Maximum number of 4-lane SIMD channel groups per GRU block (`12` lanes/block at `Ultra` Block 0).
pub const MAX_GROUPS: usize = 3;
/// Number of sequential Matryoshka GRU blocks in Stage II.
pub const NUM_GRU_BLOCKS: usize = 3;
/// Tapered maximum active 4-lane group count per block (`[3, 2, 1]` at `Ultra`).
pub const MAX_GROUPS_PER_BLOCK: [usize; NUM_GRU_BLOCKS] = [3, 2, 1];
/// Alias matching historical constant name (`3` Stage II blocks).
pub const NUM_CASCADE_BLOCKS: usize = NUM_GRU_BLOCKS;
/// Maximum lanes per Stage II GRU block (`4 * MAX_GROUPS = 12`).
pub const MAX_BLOCK_LANES: usize = MAX_GROUPS * 4;
/// Maximum total Stage II recurrent lanes capacity across all 3 blocks (`36` padded).
pub const MAX_TOTAL_GRU_LANES: usize = NUM_GRU_BLOCKS * MAX_BLOCK_LANES;
/// Legacy constant alias (`36` total Stage II lane capacity).
pub const MAX_CASCADE_LANES: usize = MAX_TOTAL_GRU_LANES;
/// Number of lower-triangular `4x4` matrix tiles (`1 + 2 + 3 = 6`).
pub const LOWER_TRI_TILES: usize = (MAX_GROUPS * (MAX_GROUPS + 1)) >> 1;
/// Number of multi-timescale input features consumed by Block 0 (`[u, u_hp, u_pinch, env_fast, env_slow]`).
pub const NUM_IN_FEATS_B0: usize = 5;
/// Number of multi-timescale input features consumed by subsequent blocks (`[u, u_hp, u_pinch, env_fast, env_slow]`).
pub const NUM_IN_FEATS_BN: usize = 5;
/// Maximum number of 16-tap Stage III Early FIR segments (`64` taps at `Ultra`).
pub const MAX_FIR_SEGMENTS: usize = 4;
/// Maximum number of 4-pole Kautz modal banks (`16` complex conjugate pairs = `32` real poles at `Ultra`).
pub const MAX_KAUTZ_BANKS: usize = 4;
/// Maximum Stage III conjugate pole pairs (`16`).
pub const MAX_KAUTZ_POLES: usize = MAX_KAUTZ_BANKS * 4;
/// Power-of-two Stage III FIR ring capacity (`128` samples, mirrored to `256`).
pub const FIR_RING_CAP: usize = 128;
/// Power-of-two mask (`FIR_RING_CAP - 1`).
pub const FIR_RING_MASK: usize = FIR_RING_CAP - 1;

/// Tapered per-block 4-lane group counts `[B0, B1, B2]` (`[1,1,1]`, `[2,1,1]`, `[2,2,1]`, `[3,2,1]`).
#[inline(always)]
#[must_use]
pub const fn tier_block_groups_array(tier: Tier) -> [usize; NUM_GRU_BLOCKS] {
    match tier {
        Tier::Lite => [1, 1, 1],
        Tier::Full => [2, 1, 1],
        Tier::Pro => [2, 2, 1],
        Tier::Ultra => [3, 2, 1],
    }
}

/// Active 4-lane group count for `tier` and Stage II block `block_idx`.
#[inline(always)]
#[must_use]
pub const fn tier_block_groups(tier: Tier, block_idx: usize) -> usize {
    tier_block_groups_array(tier)[block_idx]
}

/// Tapered per-block active recurrent lane counts `[B0, B1, B2]`.
#[inline(always)]
#[must_use]
pub const fn tier_block_lanes_array(tier: Tier) -> [usize; NUM_GRU_BLOCKS] {
    let g = tier_block_groups_array(tier);
    [g[0] << 2, g[1] << 2, g[2] << 2]
}

/// Active lane count for `tier` and Stage II block `block_idx`.
#[inline(always)]
#[must_use]
pub const fn tier_block_lanes_at(tier: Tier, block_idx: usize) -> usize {
    tier_block_groups(tier, block_idx) << 2
}

/// Active Matryoshka pyramid unlock group count for `tier` (`1, 2, 3, 3`).
#[inline(always)]
#[must_use]
pub const fn tier_groups(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 1,
        Tier::Full => 2,
        Tier::Pro | Tier::Ultra => 3,
    }
}

/// Active Stage II GRU block count for `tier` (`3` across all tiers).
#[inline(always)]
#[must_use]
pub const fn tier_gru_blocks(_tier: Tier) -> usize {
    NUM_GRU_BLOCKS
}

/// Active lane count in Block 0 for `tier` (`4, 8, 8, 12`).
#[inline(always)]
#[must_use]
pub const fn tier_block_lanes(tier: Tier) -> usize {
    tier_block_groups(tier, 0) << 2
}

/// Active total GRU lanes across all 3 blocks for `tier` (`12, 16, 20, 24`).
#[inline(always)]
#[must_use]
pub const fn tier_total_gru_lanes(tier: Tier) -> usize {
    let g = tier_block_groups_array(tier);
    (g[0] + g[1] + g[2]) << 2
}

/// Legacy helper returning active lanes in Block 0 (`4, 8, 8, 12`).
#[inline(always)]
#[must_use]
pub const fn tier_cascade_lanes(tier: Tier) -> usize {
    tier_block_lanes(tier)
}

/// Active Stage III 16-tap FIR segment count (`1, 2, 3, 4` -> `16, 32, 48, 64` taps).
#[inline(always)]
#[must_use]
pub const fn tier_fir_segments(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 1,
        Tier::Full => 2,
        Tier::Pro => 3,
        Tier::Ultra => 4,
    }
}

/// Active Stage III 4-pole Kautz bank count (`1, 2, 3, 4` -> `4, 8, 12, 16` pole pairs = `8..32` real poles).
#[inline(always)]
#[must_use]
pub const fn tier_kautz_banks(tier: Tier) -> usize {
    match tier {
        Tier::Lite => 1,
        Tier::Full => 2,
        Tier::Pro => 3,
        Tier::Ultra => 4,
    }
}

/// Packed lower-triangular `4x4` tile index for `(dst_g, src_g)` (`src_g <= dst_g`, `0..6`).
#[inline(always)]
#[must_use]
pub const fn tri_tile_index(dst_g: usize, src_g: usize) -> usize {
    ((dst_g * (dst_g + 1)) >> 1) + src_g
}

/// Flattened readout lane index for `(block_idx, grp_idx, lane_idx)` in `0..MAX_TOTAL_GRU_LANES`.
#[inline(always)]
#[must_use]
pub const fn flat_gru_lane_index(block_idx: usize, grp_idx: usize, lane_idx: usize) -> usize {
    block_idx * MAX_BLOCK_LANES + (grp_idx << 2) + lane_idx
}

/// Column-major `4x4` matrix tile (`64 B`, aligned for 4-lane SIMD column broadcasts).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat4Tile {
    pub cols: [[f32; 4]; 4],
}

/// Per-group (`4` lanes) packed GRU input & bias parameters (`320 B`, 5 cache lines).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3GruGroupHot {
    pub w_in_z: [[f32; 4]; NUM_IN_FEATS_B0],
    pub w_in_r: [[f32; 4]; NUM_IN_FEATS_B0],
    pub w_in_n: [[f32; 4]; NUM_IN_FEATS_B0],
    pub b_z: [f32; 4],
    pub b_r: [f32; 4],
    pub b_n: [f32; 4],
    /// Backend-precomputed `tanh_pade5_vec4::<B>(b_n)` guaranteeing bit-exact `+0.0` at rest.
    pub tanh_b_n: [f32; 4],
    /// Multiplicative zero-DC power-supply headroom sag (`[0.0, 8.0]`).
    pub k_sag: [f32; 4],
}

/// Hot L1D-resident Matryoshka GRU block (`64 B` aligned; lower-triangular prefix packed).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3GruBlockHot {
    pub groups: [V3GruGroupHot; MAX_GROUPS],
    pub k_quad: [[f32; 4]; MAX_GROUPS],
    pub w_prev_z: [Mat4Tile; LOWER_TRI_TILES],
    pub w_prev_r: [Mat4Tile; LOWER_TRI_TILES],
    pub w_prev_n: [Mat4Tile; LOWER_TRI_TILES],
    pub w_rec_z: [Mat4Tile; LOWER_TRI_TILES],
    pub w_rec_r: [Mat4Tile; LOWER_TRI_TILES],
    pub w_rec_n: [Mat4Tile; LOWER_TRI_TILES],
}

fn pack_lower_tri_tiles(
    src_mat: &[[f32; MAX_BLOCK_LANES]; MAX_BLOCK_LANES],
    g_cnt: usize,
) -> [Mat4Tile; LOWER_TRI_TILES] {
    let mut out = [Mat4Tile::default(); LOWER_TRI_TILES];
    for dst_g in 0..g_cnt {
        for src_g in 0..=dst_g {
            let mut cols = [[0.0_f32; 4]; 4];
            for col in 0..4 {
                for row in 0..4 {
                    cols[col][row] = src_mat[(dst_g << 2) + row][(src_g << 2) + col];
                }
            }
            out[tri_tile_index(dst_g, src_g)].cols = cols;
        }
    }
    out
}

impl V3GruBlockHot {
    /// Packs `V3GruBlockCold` into contiguous lower-triangular SIMD tiles with precomputed `tanh(b_n)`.
    pub fn from_cold<B: MathBackend>(cold: &V3GruBlockCold, num_groups: usize) -> Self {
        let mut c = *cold;
        c.clamp_in_place();
        let g_cnt = num_groups.clamp(1, MAX_GROUPS);
        let mut out = Self::default();
        for g in 0..g_cnt {
            let base = g << 2;
            let mut grp = V3GruGroupHot::default();
            for f in 0..NUM_IN_FEATS_B0 {
                for l in 0..4 {
                    grp.w_in_z[f][l] = c.w_in_z[f][base + l];
                    grp.w_in_r[f][l] = c.w_in_r[f][base + l];
                    grp.w_in_n[f][l] = c.w_in_n[f][base + l];
                }
            }
            for l in 0..4 {
                grp.b_z[l] = c.b_z[base + l];
                grp.b_r[l] = c.b_r[base + l];
                grp.b_n[l] = c.b_n[base + l];
                grp.k_sag[l] = c.k_sag[base + l];
                out.k_quad[g][l] = c.k_quad[base + l];
            }
            grp.tanh_b_n = tanh_pade5_vec4::<B>(B::Vec4::load(&grp.b_n)).to_array();
            out.groups[g] = grp;
        }
        out.w_prev_z = pack_lower_tri_tiles(&c.w_prev_z, g_cnt);
        out.w_prev_r = pack_lower_tri_tiles(&c.w_prev_r, g_cnt);
        out.w_prev_n = pack_lower_tri_tiles(&c.w_prev_n, g_cnt);
        out.w_rec_z = pack_lower_tri_tiles(&c.w_rec_z, g_cnt);
        out.w_rec_r = pack_lower_tri_tiles(&c.w_rec_r, g_cnt);
        out.w_rec_n = pack_lower_tri_tiles(&c.w_rec_n, g_cnt);
        out
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Inline GRU block step passes block tiles and accumulation state without spilling"
    )]
    #[inline(always)]
    pub fn step_block<B: MathBackend>(
        &self,
        block_idx: usize,
        num_groups: usize,
        feats: &[f32; NUM_IN_FEATS_B0],
        prev_block_h: &[[f32; 4]; MAX_GROUPS],
        block_h: &mut [[f32; 4]; MAX_GROUPS],
        cab_acc: &mut B::Vec4,
        dir_acc: &mut B::Vec4,
        pre_cab_w_blk: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w_blk: &[[f32; 4]; MAX_GROUPS],
    ) {
        let h_old = *block_h;
        let xf = feats.map(B::Vec4::splat);
        let one = B::Vec4::splat(1.0);
        let lo12 = B::Vec4::splat(-12.0);
        let hi12 = B::Vec4::splat(12.0);
        let lo25 = B::Vec4::splat(-2.5);
        let hi25 = B::Vec4::splat(2.5);
        for dst_g in 0..num_groups {
            let grp = &self.groups[dst_g];
            let mut az = B::Vec4::load(&grp.b_z);
            let mut ar = B::Vec4::load(&grp.b_r);
            let mut an_in = B::Vec4::zero();
            for (f, &xf_v) in xf.iter().enumerate() {
                az = az.fma(xf_v, B::Vec4::load(&grp.w_in_z[f]));
                ar = ar.fma(xf_v, B::Vec4::load(&grp.w_in_r[f]));
                an_in = an_in.fma(xf_v, B::Vec4::load(&grp.w_in_n[f]));
            }

            let mut rec_n = B::Vec4::zero();
            for src_g in 0..=dst_g {
                let tile_idx = tri_tile_index(dst_g, src_g);
                if block_idx > 0 {
                    let x_prev = &prev_block_h[src_g];
                    az = B::fma_mat4x4_vec4(az, &self.w_prev_z[tile_idx].cols, x_prev);
                    ar = B::fma_mat4x4_vec4(ar, &self.w_prev_r[tile_idx].cols, x_prev);
                    an_in = B::fma_mat4x4_vec4(an_in, &self.w_prev_n[tile_idx].cols, x_prev);
                }
                let x_old = &h_old[src_g];
                az = B::fma_mat4x4_vec4(az, &self.w_rec_z[tile_idx].cols, x_old);
                ar = B::fma_mat4x4_vec4(ar, &self.w_rec_r[tile_idx].cols, x_old);
                rec_n = B::fma_mat4x4_vec4(rec_n, &self.w_rec_n[tile_idx].cols, x_old);
            }

            let z = sigmoid_pade5_vec4::<B>(az);
            let r = sigmoid_pade5_vec4::<B>(ar);
            let a_sig = an_in.add(rec_n).clamp(lo12, hi12);
            let a_eff = a_sig.fma(a_sig.mul(a_sig), B::Vec4::load(&self.k_quad[dst_g]));
            let g_sag = one.fma(B::Vec4::load(&grp.k_sag), xf[4]);
            let h_sag = g_sag.rcp_ge1_nr2();
            let an = B::Vec4::load(&grp.b_n).fma(g_sag, a_eff);
            let n = h_sag.mul(tanh_pade5_vec4::<B>(an).sub(B::Vec4::load(&grp.tanh_b_n)));
            let h_old_v = B::Vec4::load(&h_old[dst_g]);
            let h_ret = one.sub(z).mul(h_old_v);
            let h_new = flush_recursive_state_vec4::<B::Vec4>(h_ret.fma(r, n)).clamp(lo25, hi25);

            h_new.store(&mut block_h[dst_g]);
            *cab_acc = cab_acc.fma(h_new, B::Vec4::load(&pre_cab_w_blk[dst_g]));
            *dir_acc = dir_acc.fma(h_new, B::Vec4::load(&direct_s2_w_blk[dst_g]));
        }
    }

    #[inline(always)]
    fn fma_in_feats_stereo<B: MathBackend>(
        acc_l: B::Vec4,
        acc_r: B::Vec4,
        w: &[[f32; 4]; NUM_IN_FEATS_B0],
        f03_l: B::Vec4,
        f03_r: B::Vec4,
        f4_l: B::Vec4,
        f4_r: B::Vec4,
    ) -> (B::Vec4, B::Vec4) {
        let c0 = B::Vec4::load(&w[0]);
        let c1 = B::Vec4::load(&w[1]);
        let c2 = B::Vec4::load(&w[2]);
        let c3 = B::Vec4::load(&w[3]);
        let c4 = B::Vec4::load(&w[4]);
        (
            acc_l
                .fma_lane::<0>(c0, f03_l)
                .fma_lane::<1>(c1, f03_l)
                .fma_lane::<2>(c2, f03_l)
                .fma_lane::<3>(c3, f03_l)
                .fma(f4_l, c4),
            acc_r
                .fma_lane::<0>(c0, f03_r)
                .fma_lane::<1>(c1, f03_r)
                .fma_lane::<2>(c2, f03_r)
                .fma_lane::<3>(c3, f03_r)
                .fma(f4_r, c4),
        )
    }

    /// Dual-issue stereo Matryoshka GRU block step (`L + R` simultaneously via `fma_mat4x4_stereo`).
    #[expect(
        clippy::too_many_arguments,
        reason = "Dual-issue stereo GRU step keeps L/R accumulators and tile loads in registers"
    )]
    #[inline(always)]
    pub fn step_block_stereo<B: MathBackend>(
        &self,
        block_idx: usize,
        num_groups: usize,
        feats_l: &[f32; NUM_IN_FEATS_B0],
        feats_r: &[f32; NUM_IN_FEATS_B0],
        prev_block_h_l: &[[f32; 4]; MAX_GROUPS],
        prev_block_h_r: &[[f32; 4]; MAX_GROUPS],
        block_h_l: &mut [[f32; 4]; MAX_GROUPS],
        block_h_r: &mut [[f32; 4]; MAX_GROUPS],
        film_gain_blk: &[[f32; 4]; MAX_GROUPS],
        bn_eff_blk: &[[f32; 4]; MAX_GROUPS],
        tanh_bn_eff_blk: &[[f32; 4]; MAX_GROUPS],
        cab_acc_l: &mut B::Vec4,
        cab_acc_r: &mut B::Vec4,
        dir_acc_l: &mut B::Vec4,
        dir_acc_r: &mut B::Vec4,
        pre_cab_w_l_blk: &[[f32; 4]; MAX_GROUPS],
        pre_cab_w_r_blk: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w_l_blk: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w_r_blk: &[[f32; 4]; MAX_GROUPS],
    ) {
        let h_old_l = *block_h_l;
        let h_old_r = *block_h_r;
        let xf_l = feats_l.map(B::Vec4::splat);
        let xf_r = feats_r.map(B::Vec4::splat);
        let one = B::Vec4::splat(1.0);
        let lo12 = B::Vec4::splat(-12.0);
        let hi12 = B::Vec4::splat(12.0);
        let lo25 = B::Vec4::splat(-2.5);
        let hi25 = B::Vec4::splat(2.5);
        for dst_g in 0..num_groups {
            let grp = &self.groups[dst_g];
            let mut az_l = B::Vec4::load(&grp.b_z);
            let mut az_r = az_l;
            let mut ar_l = B::Vec4::load(&grp.b_r);
            let mut ar_r = ar_l;
            let mut an_in_l = B::Vec4::zero();
            let mut an_in_r = B::Vec4::zero();
            for f in 0..NUM_IN_FEATS_B0 {
                let xfl = xf_l[f];
                let xfr = xf_r[f];
                let wz = B::Vec4::load(&grp.w_in_z[f]);
                let wr = B::Vec4::load(&grp.w_in_r[f]);
                let wn = B::Vec4::load(&grp.w_in_n[f]);
                az_l = az_l.fma(xfl, wz);
                az_r = az_r.fma(xfr, wz);
                ar_l = ar_l.fma(xfl, wr);
                ar_r = ar_r.fma(xfr, wr);
                an_in_l = an_in_l.fma(xfl, wn);
                an_in_r = an_in_r.fma(xfr, wn);
            }

            let mut rec_n_l = B::Vec4::zero();
            let mut rec_n_r = B::Vec4::zero();
            for src_g in 0..=dst_g {
                let tile_idx = tri_tile_index(dst_g, src_g);
                if block_idx > 0 {
                    let pl = B::Vec4::load(&prev_block_h_l[src_g]);
                    let pr = B::Vec4::load(&prev_block_h_r[src_g]);
                    (az_l, az_r) =
                        B::fma_mat4x4_stereo(az_l, az_r, &self.w_prev_z[tile_idx].cols, pl, pr);
                    (ar_l, ar_r) =
                        B::fma_mat4x4_stereo(ar_l, ar_r, &self.w_prev_r[tile_idx].cols, pl, pr);
                    (an_in_l, an_in_r) = B::fma_mat4x4_stereo(
                        an_in_l,
                        an_in_r,
                        &self.w_prev_n[tile_idx].cols,
                        pl,
                        pr,
                    );
                }
                let ol = B::Vec4::load(&h_old_l[src_g]);
                let or = B::Vec4::load(&h_old_r[src_g]);
                (az_l, az_r) =
                    B::fma_mat4x4_stereo(az_l, az_r, &self.w_rec_z[tile_idx].cols, ol, or);
                (ar_l, ar_r) =
                    B::fma_mat4x4_stereo(ar_l, ar_r, &self.w_rec_r[tile_idx].cols, ol, or);
                (rec_n_l, rec_n_r) =
                    B::fma_mat4x4_stereo(rec_n_l, rec_n_r, &self.w_rec_n[tile_idx].cols, ol, or);
            }

            let (z_l, z_r) = sigmoid_pade5_pair_vec4::<B>(az_l, az_r);
            let (r_l, r_r) = sigmoid_pade5_pair_vec4::<B>(ar_l, ar_r);
            let a_sig_l = an_in_l.add(rec_n_l).clamp(lo12, hi12);
            let a_sig_r = an_in_r.add(rec_n_r).clamp(lo12, hi12);
            let kq = B::Vec4::load(&self.k_quad[dst_g]);
            let film = B::Vec4::load(&film_gain_blk[dst_g]);
            let a_eff_l = a_sig_l.fma(a_sig_l.mul(a_sig_l), kq).mul(film);
            let a_eff_r = a_sig_r.fma(a_sig_r.mul(a_sig_r), kq).mul(film);

            let ks = B::Vec4::load(&grp.k_sag);
            let g_sag_l = one.fma(ks, xf_l[4]);
            let g_sag_r = one.fma(ks, xf_r[4]);
            let (h_sag_l, h_sag_r) = g_sag_l.rcp_ge1_pair_nr2(g_sag_r);

            let bn_v = B::Vec4::load(&bn_eff_blk[dst_g]);
            let tanh_bn_v = B::Vec4::load(&tanh_bn_eff_blk[dst_g]);
            let (th_l, th_r) =
                tanh_pade5_pair_vec4::<B>(bn_v.fma(g_sag_l, a_eff_l), bn_v.fma(g_sag_r, a_eff_r));
            let n_l = h_sag_l.mul(th_l.sub(tanh_bn_v));
            let n_r = h_sag_r.mul(th_r.sub(tanh_bn_v));

            let h_ret_l = one.sub(z_l).mul(B::Vec4::load(&h_old_l[dst_g]));
            let h_ret_r = one.sub(z_r).mul(B::Vec4::load(&h_old_r[dst_g]));
            let h_new_l =
                flush_recursive_state_vec4::<B::Vec4>(h_ret_l.fma(r_l, n_l)).clamp(lo25, hi25);
            let h_new_r =
                flush_recursive_state_vec4::<B::Vec4>(h_ret_r.fma(r_r, n_r)).clamp(lo25, hi25);

            h_new_l.store(&mut block_h_l[dst_g]);
            h_new_r.store(&mut block_h_r[dst_g]);
            *cab_acc_l = cab_acc_l.fma(h_new_l, B::Vec4::load(&pre_cab_w_l_blk[dst_g]));
            *cab_acc_r = cab_acc_r.fma(h_new_r, B::Vec4::load(&pre_cab_w_r_blk[dst_g]));
            *dir_acc_l = dir_acc_l.fma(h_new_l, B::Vec4::load(&direct_s2_w_l_blk[dst_g]));
            *dir_acc_r = dir_acc_r.fma(h_new_r, B::Vec4::load(&direct_s2_w_r_blk[dst_g]));
        }
    }

    /// Register-resident stereo GRU block step across `[B::Vec4; G]` state registers (`0` stack spills).
    #[expect(
        clippy::too_many_arguments,
        reason = "Register-resident stereo GRU step eliminates stack spills across lanes"
    )]
    #[inline(always)]
    pub fn step_block_stereo_reg<B: MathBackend, const G: usize, const HAS_PREV: bool>(
        &self,
        f03_l: B::Vec4,
        f03_r: B::Vec4,
        f4_l: B::Vec4,
        f4_r: B::Vec4,
        prev_block_h_l: &[B::Vec4; G],
        prev_block_h_r: &[B::Vec4; G],
        block_h_l: &mut [B::Vec4; G],
        block_h_r: &mut [B::Vec4; G],
        film_gain_blk: &[[f32; 4]; MAX_GROUPS],
        bn_eff_blk: &[[f32; 4]; MAX_GROUPS],
        tanh_bn_eff_blk: &[[f32; 4]; MAX_GROUPS],
        cab_acc_l: &mut B::Vec4,
        cab_acc_r: &mut B::Vec4,
        dir_acc_l: &mut B::Vec4,
        dir_acc_r: &mut B::Vec4,
        pre_cab_w_l_blk: &[[f32; 4]; MAX_GROUPS],
        pre_cab_w_r_blk: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w_l_blk: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w_r_blk: &[[f32; 4]; MAX_GROUPS],
    ) {
        const { assert!(G <= MAX_GROUPS) };
        let one = B::Vec4::splat(1.0);
        let lo12 = B::Vec4::splat(-12.0);
        let hi12 = B::Vec4::splat(12.0);
        let lo25 = B::Vec4::splat(-2.5);
        let hi25 = B::Vec4::splat(2.5);

        for dst_g in (0..G).rev() {
            let grp = &self.groups[dst_g];
            let az0 = B::Vec4::load(&grp.b_z);
            let ar0 = B::Vec4::load(&grp.b_r);
            let (mut az_l, mut az_r) =
                Self::fma_in_feats_stereo::<B>(az0, az0, &grp.w_in_z, f03_l, f03_r, f4_l, f4_r);
            let (mut ar_l, mut ar_r) =
                Self::fma_in_feats_stereo::<B>(ar0, ar0, &grp.w_in_r, f03_l, f03_r, f4_l, f4_r);
            let (mut an_in_l, mut an_in_r) = Self::fma_in_feats_stereo::<B>(
                B::Vec4::zero(),
                B::Vec4::zero(),
                &grp.w_in_n,
                f03_l,
                f03_r,
                f4_l,
                f4_r,
            );

            let mut rec_n_l = B::Vec4::zero();
            let mut rec_n_r = B::Vec4::zero();
            for src_g in 0..=dst_g {
                let tile_idx = tri_tile_index(dst_g, src_g);
                if HAS_PREV {
                    let pl = prev_block_h_l[src_g];
                    let pr = prev_block_h_r[src_g];
                    (az_l, az_r) =
                        B::fma_mat4x4_stereo(az_l, az_r, &self.w_prev_z[tile_idx].cols, pl, pr);
                    (ar_l, ar_r) =
                        B::fma_mat4x4_stereo(ar_l, ar_r, &self.w_prev_r[tile_idx].cols, pl, pr);
                    (an_in_l, an_in_r) = B::fma_mat4x4_stereo(
                        an_in_l,
                        an_in_r,
                        &self.w_prev_n[tile_idx].cols,
                        pl,
                        pr,
                    );
                }
                let ol = block_h_l[src_g];
                let or = block_h_r[src_g];
                (az_l, az_r) =
                    B::fma_mat4x4_stereo(az_l, az_r, &self.w_rec_z[tile_idx].cols, ol, or);
                (ar_l, ar_r) =
                    B::fma_mat4x4_stereo(ar_l, ar_r, &self.w_rec_r[tile_idx].cols, ol, or);
                (rec_n_l, rec_n_r) =
                    B::fma_mat4x4_stereo(rec_n_l, rec_n_r, &self.w_rec_n[tile_idx].cols, ol, or);
            }

            let (z_l, z_r) = sigmoid_pade5_pair_vec4::<B>(az_l, az_r);
            let (r_l, r_r) = sigmoid_pade5_pair_vec4::<B>(ar_l, ar_r);
            let a_sig_l = an_in_l.add(rec_n_l).clamp(lo12, hi12);
            let a_sig_r = an_in_r.add(rec_n_r).clamp(lo12, hi12);
            let kq = B::Vec4::load(&self.k_quad[dst_g]);
            let film = B::Vec4::load(&film_gain_blk[dst_g]);
            let a_eff_l = a_sig_l.fma(a_sig_l.mul(a_sig_l), kq).mul(film);
            let a_eff_r = a_sig_r.fma(a_sig_r.mul(a_sig_r), kq).mul(film);

            let ks = B::Vec4::load(&grp.k_sag);
            let g_sag_l = one.fma(ks, f4_l);
            let g_sag_r = one.fma(ks, f4_r);
            let (h_sag_l, h_sag_r) = g_sag_l.rcp_ge1_pair_nr2(g_sag_r);

            let bn_v = B::Vec4::load(&bn_eff_blk[dst_g]);
            let tanh_bn_v = B::Vec4::load(&tanh_bn_eff_blk[dst_g]);
            let (th_l, th_r) =
                tanh_pade5_pair_vec4::<B>(bn_v.fma(g_sag_l, a_eff_l), bn_v.fma(g_sag_r, a_eff_r));
            let n_l = h_sag_l.mul(th_l.sub(tanh_bn_v));
            let n_r = h_sag_r.mul(th_r.sub(tanh_bn_v));

            let h_ret_l = one.sub(z_l).mul(block_h_l[dst_g]);
            let h_ret_r = one.sub(z_r).mul(block_h_r[dst_g]);
            let h_new_l =
                flush_recursive_state_vec4::<B::Vec4>(h_ret_l.fma(r_l, n_l)).clamp(lo25, hi25);
            let h_new_r =
                flush_recursive_state_vec4::<B::Vec4>(h_ret_r.fma(r_r, n_r)).clamp(lo25, hi25);

            block_h_l[dst_g] = h_new_l;
            block_h_r[dst_g] = h_new_r;
            *cab_acc_l = cab_acc_l.fma(h_new_l, B::Vec4::load(&pre_cab_w_l_blk[dst_g]));
            *cab_acc_r = cab_acc_r.fma(h_new_r, B::Vec4::load(&pre_cab_w_r_blk[dst_g]));
            *dir_acc_l = dir_acc_l.fma(h_new_l, B::Vec4::load(&direct_s2_w_l_blk[dst_g]));
            *dir_acc_r = dir_acc_r.fma(h_new_r, B::Vec4::load(&direct_s2_w_r_blk[dst_g]));
        }
    }
}

/// Per-channel Stage II Matryoshka GRU recursive state (`MAX_GROUPS = 3` × `3` blocks).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3GruChannelState {
    pub feat: V3InputFeaturesState,
    pub h: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
}

impl V3GruChannelState {
    /// Resets all input feature and GRU hidden states to exact resting `+0.0`.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    /// verify[dsp.op-n.polyadaa.drive-bank]
    #[test]
    fn test_v3_gru_step_block_stereo_matches_mono_and_zero_dc() {
        let mut cold = V3GruBlockCold::default();
        cold.w_in_n[0][0] = 0.85;
        cold.w_in_n[1][2] = -0.5;
        cold.b_z[0] = -1.2;
        cold.b_n[0] = 0.4;
        cold.k_sag[0] = 1.8;
        cold.k_quad[0] = 0.25;
        cold.w_rec_n[0][0] = 0.15;
        let hot = V3GruBlockHot::from_cold::<DefaultMathBackend>(&cold, 2);

        let film_one = [[1.0_f32; 4]; MAX_GROUPS];
        let mut bn_eff = [[0.0_f32; 4]; MAX_GROUPS];
        let mut tanh_bn = [[0.0_f32; 4]; MAX_GROUPS];
        for g in 0..MAX_GROUPS {
            bn_eff[g] = hot.groups[g].b_n;
            tanh_bn[g] = hot.groups[g].tanh_b_n;
        }
        let w = [[0.25_f32; 4]; MAX_GROUPS];
        let zero_feats = [0.0_f32; NUM_IN_FEATS_B0];
        let mut hl = [[[0.0_f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS];
        let mut hr = [[[0.0_f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS];
        let (mut cl, mut cr, mut dl, mut dr) = (
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
        );
        let prev = [[[0.0_f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS];
        hot.step_block_stereo::<DefaultMathBackend>(
            0,
            2,
            &zero_feats,
            &zero_feats,
            &prev[0],
            &prev[0],
            &mut hl[0],
            &mut hr[0],
            &film_one,
            &bn_eff,
            &tanh_bn,
            &mut cl,
            &mut cr,
            &mut dl,
            &mut dr,
            &w,
            &w,
            &w,
            &w,
        );
        assert_eq!(cl.horizontal_sum().to_bits(), 0);
        assert_eq!(cr.horizontal_sum().to_bits(), 0);

        let feats = [0.6_f32, -0.2, 0.1, 0.4, 0.35];
        let mut hm = [[0.0_f32; 4]; MAX_GROUPS];
        let mut cm = <DefaultMathBackend as MathBackend>::Vec4::zero();
        let mut dm = <DefaultMathBackend as MathBackend>::Vec4::zero();
        hot.step_block::<DefaultMathBackend>(
            0, 2, &feats, &prev[0], &mut hm, &mut cm, &mut dm, &w, &w,
        );
        hot.step_block_stereo::<DefaultMathBackend>(
            0, 2, &feats, &feats, &prev[0], &prev[0], &mut hl[0], &mut hr[0], &film_one, &bn_eff,
            &tanh_bn, &mut cl, &mut cr, &mut dl, &mut dr, &w, &w, &w, &w,
        );
        assert!((cm.horizontal_sum() - cl.horizontal_sum()).abs() < 1.0e-5);
        assert!((dm.horizontal_sum() - dr.horizontal_sum()).abs() < 1.0e-5);

        let f03 = <DefaultMathBackend as MathBackend>::Vec4::from_array([
            feats[0], feats[1], feats[2], feats[3],
        ]);
        let f4 = <DefaultMathBackend as MathBackend>::Vec4::splat(feats[4]);
        let prev_reg = [<DefaultMathBackend as MathBackend>::Vec4::from_array(prev[0][0]); 2];
        let mut h_reg_l = [<DefaultMathBackend as MathBackend>::Vec4::zero(); 2];
        let mut h_reg_r = [<DefaultMathBackend as MathBackend>::Vec4::zero(); 2];
        let (mut cr_l, mut cr_r, mut dr_l, mut dr_r) = (
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
            <DefaultMathBackend as MathBackend>::Vec4::zero(),
        );
        hot.step_block_stereo_reg::<DefaultMathBackend, 2, true>(
            f03,
            f03,
            f4,
            f4,
            &prev_reg,
            &prev_reg,
            &mut h_reg_l,
            &mut h_reg_r,
            &film_one,
            &bn_eff,
            &tanh_bn,
            &mut cr_l,
            &mut cr_r,
            &mut dr_l,
            &mut dr_r,
            &w,
            &w,
            &w,
            &w,
        );
        assert!((cm.horizontal_sum() - cr_l.horizontal_sum()).abs() < 1.0e-5);
        assert!((dm.horizontal_sum() - dr_r.horizontal_sum()).abs() < 1.0e-5);

        let mut cold_scaled = cold;
        cold_scaled.b_z[0] = -2.0;
        cold_scaled.b_r[0] = -2.0;
        cold_scaled.rescale_2x_to_1x();
        assert!((cold_scaled.b_z[0] - cold_scaled.b_r[0]).abs() < 1.0e-5);
    }
}
