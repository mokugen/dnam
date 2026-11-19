//! Stage II sub-block feed-forward projection & lower-triangular tile pre-pass helpers.

use crate::math::{MathBackend, SimdVec4};
use crate::stages::stage2_gru::block::{LOWER_TRI_TILES, Mat4Tile, V3GruBlockHot, V3GruGroupHot};

#[inline(always)]
pub(super) fn load_tile_cols<V: SimdVec4>(tile: &Mat4Tile) -> [V; 4] {
    [
        V::load(&tile.cols[0]),
        V::load(&tile.cols[1]),
        V::load(&tile.cols[2]),
        V::load(&tile.cols[3]),
    ]
}

#[inline(always)]
pub(super) fn fma4_stereo<V: SimdVec4>(acc_l: V, acc_r: V, cols: [V; 4], x_l: V, x_r: V) -> (V, V) {
    let [c0, c1, c2, c3] = cols;
    let l = acc_l
        .fma_lane::<0>(c0, x_l)
        .fma_lane::<1>(c1, x_l)
        .fma_lane::<2>(c2, x_l)
        .fma_lane::<3>(c3, x_l);
    let r = acc_r
        .fma_lane::<0>(c0, x_r)
        .fma_lane::<1>(c1, x_r)
        .fma_lane::<2>(c2, x_r)
        .fma_lane::<3>(c3, x_r);
    (l, r)
}

#[inline(always)]
pub(super) fn fma4_stereo_tree<V: SimdVec4>(
    acc_l: V,
    acc_r: V,
    cols: [V; 4],
    x_l: V,
    x_r: V,
) -> (V, V) {
    let [c0, c1, c2, c3] = cols;
    let l = acc_l
        .fma_lane::<0>(c0, x_l)
        .fma_lane::<1>(c1, x_l)
        .add(c2.mul_lane::<2>(x_l).fma_lane::<3>(c3, x_l));
    let r = acc_r
        .fma_lane::<0>(c0, x_r)
        .fma_lane::<1>(c1, x_r)
        .add(c2.mul_lane::<2>(x_r).fma_lane::<3>(c3, x_r));
    (l, r)
}

#[inline(always)]
fn fma_in5_stereo<V: SimdVec4>(
    b: V,
    w: [V; 4],
    w4: V,
    f03_l: V,
    f03_r: V,
    f4_l: V,
    f4_r: V,
) -> (V, V) {
    let (l, r) = fma4_stereo(b, b, w, f03_l, f03_r);
    (l.fma(w4, f4_l), r.fma(w4, f4_r))
}

#[inline(always)]
fn mul_in5_stereo<V: SimdVec4>(w: [V; 4], w4: V, f03_l: V, f03_r: V, f4_l: V, f4_r: V) -> (V, V) {
    let [w0, w1, w2, w3] = w;
    let l = w0
        .mul_lane::<0>(f03_l)
        .fma_lane::<1>(w1, f03_l)
        .fma_lane::<2>(w2, f03_l)
        .fma_lane::<3>(w3, f03_l)
        .fma(w4, f4_l);
    let r = w0
        .mul_lane::<0>(f03_r)
        .fma_lane::<1>(w1, f03_r)
        .fma_lane::<2>(w2, f03_r)
        .fma_lane::<3>(w3, f03_r)
        .fma(w4, f4_r);
    (l, r)
}

#[inline(always)]
fn load_w_in_group<V: SimdVec4>(grp: &V3GruGroupHot) -> ([V; 4], V, [V; 4], V, [V; 4], V) {
    let wz = [
        V::load(&grp.w_in_z[0]),
        V::load(&grp.w_in_z[1]),
        V::load(&grp.w_in_z[2]),
        V::load(&grp.w_in_z[3]),
    ];
    let wr = [
        V::load(&grp.w_in_r[0]),
        V::load(&grp.w_in_r[1]),
        V::load(&grp.w_in_r[2]),
        V::load(&grp.w_in_r[3]),
    ];
    let wn = [
        V::load(&grp.w_in_n[0]),
        V::load(&grp.w_in_n[1]),
        V::load(&grp.w_in_n[2]),
        V::load(&grp.w_in_n[3]),
    ];
    (
        wz,
        V::load(&grp.w_in_z[4]),
        wr,
        V::load(&grp.w_in_r[4]),
        wn,
        V::load(&grp.w_in_n[4]),
    )
}

#[inline(always)]
fn load_f03_for_pre<B: MathBackend, const ADD_TONE: bool>(
    f03_i: &[[f32; 4]; 2],
    tone_i: [f32; 2],
    e0: B::Vec4,
    inj_v: B::Vec4,
) -> (B::Vec4, B::Vec4) {
    let mut fl = B::Vec4::load(&f03_i[0]);
    let mut fr = B::Vec4::load(&f03_i[1]);
    if ADD_TONE {
        fl = fl.fma(e0, inj_v.mul(B::Vec4::splat(tone_i[0])));
        fr = fr.fma(e0, inj_v.mul(B::Vec4::splat(tone_i[1])));
    }
    (fl, fr)
}

pub(super) struct PrePassIn<'a, const CAP: usize> {
    pub blk: &'a V3GruBlockHot,
    pub b1_inj: f32,
    pub f03: &'a [[[f32; 4]; 2]; CAP],
    pub f4: &'a [[[f32; 4]; 2]; CAP],
    pub tone_out: &'a [[f32; 2]; CAP],
    pub h_src: &'a [[[f32; 4]; 2]],
    pub n: usize,
}

#[inline(always)]
pub(super) fn fill_pre_with_opt_tile<
    B: MathBackend,
    const DST_G: usize,
    const ADD_TONE: bool,
    const HAS_TILE: bool,
    const CAP: usize,
>(
    inp: &PrePassIn<'_, CAP>,
    tiles_z: &[Mat4Tile; LOWER_TRI_TILES],
    tiles_r: &[Mat4Tile; LOWER_TRI_TILES],
    tiles_n: &[Mat4Tile; LOWER_TRI_TILES],
    tile_idx: usize,
    pre_zrn: &mut [[[[f32; 4]; 2]; 3]; CAP],
) {
    let len = inp.n.min(CAP);
    let grp = &inp.blk.groups[DST_G];
    let bz = B::Vec4::load(&grp.b_z);
    let br = B::Vec4::load(&grp.b_r);
    let (wz, wz4, wr, wr4, wn, wn4) = load_w_in_group(grp);
    let tz = load_tile_cols::<B::Vec4>(&tiles_z[tile_idx]);
    let tr = load_tile_cols::<B::Vec4>(&tiles_r[tile_idx]);
    let tn = load_tile_cols::<B::Vec4>(&tiles_n[tile_idx]);
    let e0 = B::Vec4::from_array([1.0, 0.0, 0.0, 0.0]);
    let inj_v = B::Vec4::splat(inp.b1_inj);

    for i in 0..len {
        let (fl, fr) = load_f03_for_pre::<B, ADD_TONE>(&inp.f03[i], inp.tone_out[i], e0, inj_v);
        let es_l = B::Vec4::load(&inp.f4[i][0]);
        let es_r = B::Vec4::load(&inp.f4[i][1]);
        let (mut zl, mut zr) = fma_in5_stereo(bz, wz, wz4, fl, fr, es_l, es_r);
        let (mut rl, mut rr) = fma_in5_stereo(br, wr, wr4, fl, fr, es_l, es_r);
        let (mut nl, mut nr) = mul_in5_stereo(wn, wn4, fl, fr, es_l, es_r);
        if HAS_TILE {
            let hl = B::Vec4::load(&inp.h_src[i][0]);
            let hr = B::Vec4::load(&inp.h_src[i][1]);
            (zl, zr) = fma4_stereo(zl, zr, tz, hl, hr);
            (rl, rr) = fma4_stereo(rl, rr, tr, hl, hr);
            (nl, nr) = fma4_stereo(nl, nr, tn, hl, hr);
        }
        zl.store(&mut pre_zrn[i][0][0]);
        zr.store(&mut pre_zrn[i][0][1]);
        rl.store(&mut pre_zrn[i][1][0]);
        rr.store(&mut pre_zrn[i][1][1]);
        nl.store(&mut pre_zrn[i][2][0]);
        nr.store(&mut pre_zrn[i][2][1]);
    }
}

#[inline(always)]
pub(super) fn accum_tile_group_to_pre<B: MathBackend, const CAP: usize>(
    tiles_z: &[Mat4Tile; LOWER_TRI_TILES],
    tiles_r: &[Mat4Tile; LOWER_TRI_TILES],
    tiles_n: &[Mat4Tile; LOWER_TRI_TILES],
    tile_idx: usize,
    h_src: &[[[f32; 4]; 2]],
    pre_zrn: &mut [[[[f32; 4]; 2]; 3]; CAP],
    n: usize,
) {
    let len = n.min(CAP);
    let tz = load_tile_cols::<B::Vec4>(&tiles_z[tile_idx]);
    let tr = load_tile_cols::<B::Vec4>(&tiles_r[tile_idx]);
    let tn = load_tile_cols::<B::Vec4>(&tiles_n[tile_idx]);
    for i in 0..len {
        let hl = B::Vec4::load(&h_src[i][0]);
        let hr = B::Vec4::load(&h_src[i][1]);
        let (zl, zr) = fma4_stereo(
            B::Vec4::load(&pre_zrn[i][0][0]),
            B::Vec4::load(&pre_zrn[i][0][1]),
            tz,
            hl,
            hr,
        );
        let (rl, rr) = fma4_stereo(
            B::Vec4::load(&pre_zrn[i][1][0]),
            B::Vec4::load(&pre_zrn[i][1][1]),
            tr,
            hl,
            hr,
        );
        let (nl, nr) = fma4_stereo(
            B::Vec4::load(&pre_zrn[i][2][0]),
            B::Vec4::load(&pre_zrn[i][2][1]),
            tn,
            hl,
            hr,
        );
        zl.store(&mut pre_zrn[i][0][0]);
        zr.store(&mut pre_zrn[i][0][1]);
        rl.store(&mut pre_zrn[i][1][0]);
        rr.store(&mut pre_zrn[i][1][1]);
        nl.store(&mut pre_zrn[i][2][0]);
        nr.store(&mut pre_zrn[i][2][1]);
    }
}

#[inline(always)]
pub(super) fn accum_prev_and_rec_pair_to_pre<B: MathBackend, const CAP: usize>(
    blk: &V3GruBlockHot,
    prev_tile: usize,
    rec_tile: usize,
    h_prev: &[[[f32; 4]; 2]],
    h_rec: &[[[f32; 4]; 2]],
    pre_zrn: &mut [[[[f32; 4]; 2]; 3]; CAP],
    n: usize,
) {
    let len = n.min(CAP);
    let pz = load_tile_cols::<B::Vec4>(&blk.w_prev_z[prev_tile]);
    let rz = load_tile_cols::<B::Vec4>(&blk.w_rec_z[rec_tile]);
    let pr = load_tile_cols::<B::Vec4>(&blk.w_prev_r[prev_tile]);
    let rr = load_tile_cols::<B::Vec4>(&blk.w_rec_r[rec_tile]);
    let pn = load_tile_cols::<B::Vec4>(&blk.w_prev_n[prev_tile]);
    let rn = load_tile_cols::<B::Vec4>(&blk.w_rec_n[rec_tile]);

    for i in 0..len {
        let p_l = B::Vec4::load(&h_prev[i][0]);
        let p_r = B::Vec4::load(&h_prev[i][1]);
        let r_l = B::Vec4::load(&h_rec[i][0]);
        let r_r = B::Vec4::load(&h_rec[i][1]);

        let (mut zl, mut zr) = fma4_stereo(
            B::Vec4::load(&pre_zrn[i][0][0]),
            B::Vec4::load(&pre_zrn[i][0][1]),
            pz,
            p_l,
            p_r,
        );
        (zl, zr) = fma4_stereo(zl, zr, rz, r_l, r_r);
        zl.store(&mut pre_zrn[i][0][0]);
        zr.store(&mut pre_zrn[i][0][1]);

        let (mut rl, mut rr_) = fma4_stereo(
            B::Vec4::load(&pre_zrn[i][1][0]),
            B::Vec4::load(&pre_zrn[i][1][1]),
            pr,
            p_l,
            p_r,
        );
        (rl, rr_) = fma4_stereo(rl, rr_, rr, r_l, r_r);
        rl.store(&mut pre_zrn[i][1][0]);
        rr_.store(&mut pre_zrn[i][1][1]);

        let (mut nl, mut nr) = fma4_stereo(
            B::Vec4::load(&pre_zrn[i][2][0]),
            B::Vec4::load(&pre_zrn[i][2][1]),
            pn,
            p_l,
            p_r,
        );
        (nl, nr) = fma4_stereo(nl, nr, rn, r_l, r_r);
        nl.store(&mut pre_zrn[i][2][0]);
        nr.store(&mut pre_zrn[i][2][1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::stages::stage2_gru::block::tri_tile_index;

    #[test]
    fn test_tri_tile_index_and_tile_prepass_smoke() {
        assert_eq!(tri_tile_index(0, 0), 0);
        assert_eq!(tri_tile_index(1, 0), 1);
        assert_eq!(tri_tile_index(2, 1), 4);

        let mut blk = V3GruBlockHot::default();
        blk.groups[0].b_z = [0.25; 4];
        blk.groups[0].w_in_n[0] = [1.0, -1.0, 0.5, -0.5];
        blk.w_rec_n[0].cols[0] = [0.5, 0.0, 0.0, 0.0];
        let mut f03 = [[[0.0_f32; 4]; 2]; 16];
        let f4 = [[[0.0_f32; 4]; 2]; 16];
        let tone_out = [[0.0_f32; 2]; 16];
        f03[0][0] = [2.0, 0.0, 0.0, 0.0];
        let h_src = [[[0.0_f32; 4]; 2]; 16];
        let mut pre_zrn = [[[[0.0_f32; 4]; 2]; 3]; 16];
        let inp = PrePassIn {
            blk: &blk,
            b1_inj: 0.0,
            f03: &f03,
            f4: &f4,
            tone_out: &tone_out,
            h_src: &h_src,
            n: 1,
        };
        fill_pre_with_opt_tile::<DefaultMathBackend, 0, false, true, 16>(
            &inp,
            &blk.w_rec_z,
            &blk.w_rec_r,
            &blk.w_rec_n,
            0,
            &mut pre_zrn,
        );
        assert_eq!(pre_zrn[0][0][0], [0.25; 4]);
        assert_eq!(pre_zrn[0][2][0], [2.0, -2.0, 1.0, -1.0]);
    }
}
