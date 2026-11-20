//! Stage 0 Compact Execution, Top-of-Frame Crossbar, Register-Tile SONA Group Core,
//! Stage III Sub-Block FIR/Kautz & Stage IV FDN Integration (`tick16/crossbar_step.rs`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    SonaBlockOutputStereo8, SonaGroupParamsStereo4, StereoVec8, compute_op_p_and_phasor_pair,
    step_kautz_quad_stereo8, step_sona_group_stereo8,
};

use crate::graph::GraphBackend;
use crate::graph::cabinet::CabinetSubblockScratch;
use crate::graph::sona_core::model::{
    MAX_FIR_SEGMENTS, MAX_KAUTZ_BANKS, MAX_SONA_GROUPS, MAX_SONA_TILES, SONA_BLOCK_GROUP_SLOTS,
    SONA_INTERBLOCK_TILES, tier_tank_quads,
};
use crate::graph::sona_core::slab::ctrl_apply::kautz_bank_mut;
use crate::graph::sona_core::slab::{
    CrossbarFrame, FLAG_COLD_FIR_EXT, FLAG_FREEZE, FLAG_XFADE_ACTIVE, SonaGroupReadoutRegs,
    SonaHotSlab, SonaInterBlockTileHot, TIER_SWITCH_XFADE_SAMPLES,
};
use crate::math::fp16::fp16_to_f32;
use crate::math::{MathBackend, SimdVec4, flush_recursive_state_f32};

const TIER_XFADE_INV: f32 = 1.0 / (TIER_SWITCH_XFADE_SAMPLES as f32);

const fn build_tile_lut() -> [[u8; MAX_SONA_GROUPS]; MAX_SONA_GROUPS] {
    let mut lut = [[u8::MAX; MAX_SONA_GROUPS]; MAX_SONA_GROUPS];
    let mut i = 0usize;
    while i < MAX_SONA_TILES {
        let (sg, dg) = SONA_INTERBLOCK_TILES[i];
        lut[sg][dg] = i as u8;
        i += 1;
    }
    lut
}

pub(crate) const TILE_LUT: [[u8; MAX_SONA_GROUPS]; MAX_SONA_GROUPS] = build_tile_lut();

#[derive(Clone, Copy)]
pub(crate) struct Stage0HostCoeffs {
    pub(crate) g0: f32,
    pub(crate) g1: f32,
    pub(crate) m00: f32,
    pub(crate) m01: f32,
    pub(crate) m10: f32,
    pub(crate) m11: f32,
    pub(crate) xo_low: f32,
    pub(crate) xo_high: f32,
    pub(crate) clean_sub: f32,
    pub(crate) gate_thresh: f32,
    pub(crate) gate_hyst: f32,
    pub(crate) gate_inv_knee: f32,
    pub(crate) gate_floor: f32,
    pub(crate) gate_atk_beta: f32,
    pub(crate) gate_rel_beta: f32,
}

impl Stage0HostCoeffs {
    #[inline(always)]
    pub(crate) fn from_slab<B: GraphBackend>(slab: &SonaHotSlab<B>) -> Self {
        let [g0, g1, m00, m01, m10, m11] = slab.lite.stage0_gain_mix_fp16.map(fp16_to_f32);
        let [xo_low, xo_high, clean_sub] = slab.lite.stage0_xo_fp16.map(fp16_to_f32);
        let [
            gate_thresh,
            gate_hyst,
            gate_inv_knee,
            gate_floor,
            gate_atk_beta,
            gate_rel_beta,
        ] = slab.lite.stage0_gate_fp16.map(fp16_to_f32);
        Self {
            g0,
            g1,
            m00,
            m01,
            m10,
            m11,
            xo_low,
            xo_high,
            clean_sub,
            gate_thresh,
            gate_hyst,
            gate_inv_knee,
            gate_floor,
            gate_atk_beta,
            gate_rel_beta,
        }
    }
}

/// Evaluates unconditional Stage 0 on [`SonaLiteHotPrefix`] (`0 FDIV`, `0` bypass flags).
#[inline(always)]
pub(crate) fn step_stage0_hot<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    c: &Stage0HostCoeffs,
    in_l: f32,
    in_r: f32,
) -> [f32; 4] {
    let xl = c.g0 * in_l;
    let xr = c.g1 * in_r;
    let ml = c.m00 * xl + c.m01 * xr;
    let mr = c.m10 * xl + c.m11 * xr;

    let peak = 0.5 * (xl.abs() + xr.abs());
    let beta = if peak > slab.lite.stage0_gate_env {
        c.gate_atk_beta
    } else {
        c.gate_rel_beta
    };
    let next_env = flush_recursive_state_f32(
        slab.lite.stage0_gate_env + beta * (peak - slab.lite.stage0_gate_env),
    );
    slab.lite.stage0_gate_env = next_env;

    let thr = c.gate_thresh - c.gate_hyst * slab.lite.stage0_gate_gain;
    let target_gate = ((next_env - thr) * c.gate_inv_knee + c.gate_floor).clamp(0.0, 1.0);
    slab.lite.stage0_gate_gain = flush_recursive_state_f32(
        slab.lite.stage0_gate_gain + beta * (target_gate - slab.lite.stage0_gate_gain),
    );

    let gated_l = slab.lite.stage0_gate_gain * ml;
    let gated_r = slab.lite.stage0_gate_gain * mr;

    let xo_p = slab.lite.stage0_xo_pole;
    let xo_g = 0.5 * (1.0 - xo_p);
    let mut x_prev_l = slab.lite.stage0_xo_x0[0];
    let mut x_prev_r = slab.lite.stage0_xo_x0[1];
    slab.lite.stage0_xo_x0 = [
        flush_recursive_state_f32(gated_l),
        flush_recursive_state_f32(gated_r),
    ];
    let mut ul = gated_l;
    let mut ur = gated_r;
    for i in 0..4 {
        let yl = xo_p * slab.lite.stage0_xo_y_l[i] + xo_g * (ul + x_prev_l);
        let yr = xo_p * slab.lite.stage0_xo_y_r[i] + xo_g * (ur + x_prev_r);
        x_prev_l = slab.lite.stage0_xo_y_l[i];
        x_prev_r = slab.lite.stage0_xo_y_r[i];
        slab.lite.stage0_xo_y_l[i] = flush_recursive_state_f32(yl);
        slab.lite.stage0_xo_y_r[i] = flush_recursive_state_f32(yr);
        ul = yl;
        ur = yr;
    }
    [
        c.xo_low * ul + c.xo_high * (gated_l - ul),
        c.xo_low * ur + c.xo_high * (gated_r - ur),
        c.clean_sub * ul,
        c.clean_sub * ur,
    ]
}

/// Evaluates `5 Hz` DC-blocked `Op-P`, quadrature phasors + `phasor_stereo_phase_rad` SSB rotation,
/// and causal integer `Op-T` feedback ONCE per host frame (skipped completely when inactive).
#[inline]
pub(crate) fn prepare_crossbar_frame<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    s0_l: f32,
    s0_r: f32,
) -> (f32, f32, CrossbarFrame<B::Vec4>) {
    if !slab.crossbar_active() {
        return (
            s0_l,
            s0_r,
            CrossbarFrame {
                active: false,
                op_p_vec: B::Vec4::zero(),
                ssb_l: B::Vec4::zero(),
                ssb_r: B::Vec4::zero(),
            },
        );
    }
    prepare_crossbar_frame_active(slab, s0_l, s0_r)
}

#[inline(never)]
fn prepare_crossbar_frame_active<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    mut s0_l: f32,
    mut s0_r: f32,
) -> (f32, f32, CrossbarFrame<B::Vec4>) {
    let mono_in = 0.5 * (s0_l + s0_r);
    let op_p = compute_op_p_and_phasor_pair(
        &mut slab.lite.t1_dc,
        &mut slab.lite.hilbert,
        &mut slab.lite.t1,
        &mut slab.lite.phasors,
        mono_in,
    );

    let wet_scale = (1.0 + slab.cold.ctrl_state.crossbar_wet_mod).clamp(0.0, 2.0);
    let op_p_gain = fp16_to_f32(slab.lite.op_gains_fp16[0]);
    let op_t_gain = fp16_to_f32(slab.lite.op_gains_fp16[1]);

    let op_p_vec = B::Vec4::load(&op_p.features).mul(B::Vec4::splat(op_p_gain));
    let p_cos = B::Vec4::load(&op_p.phasor_cos);
    let p_sin = B::Vec4::load(&op_p.phasor_sin);
    let c_st = B::fp16x4_to_vec4(&slab.lite.phasor_st_cos_fp16);
    let s_st = B::fp16x4_to_vec4(&slab.lite.phasor_st_sin_fp16);
    let p_cos_r = p_cos.mul(c_st).sub(p_sin.mul(s_st));
    let p_sin_r = p_sin.mul(c_st).add(p_cos.mul(s_st));
    let hi = B::Vec4::splat(op_p.hilbert_i);
    let hq = B::Vec4::splat(op_p.hilbert_q);
    let ws = B::Vec4::splat(wet_scale);
    let ssb_l = hi.mul(p_cos).sub(hq.mul(p_sin)).mul(ws);
    let ssb_r = hi.mul(p_cos_r).sub(hq.mul(p_sin_r)).mul(ws);

    if op_t_gain.abs() > 1.0e-7 {
        let aq = tier_tank_quads(slab.lite.tier);
        for q in 0..aq {
            let inline = if q == 0 {
                Some(&slab.lite.quad0_frames)
            } else {
                None
            };
            let ext_opt = if slab.cold.stage4_ext_active[q] {
                slab.cold.stage4_ext[q].as_ref()
            } else {
                None
            };
            let taps = slab.quad_hot(q).peek_integer_taps::<B>(inline, ext_opt);
            let base = q << 2;
            for l in 0..4 {
                let lr = slab.opt_to_lr_fp16(base + l);
                s0_l += op_t_gain * fp16_to_f32(lr[0]) * taps[l];
                s0_r += op_t_gain * fp16_to_f32(lr[1]) * taps[l];
            }
        }
    }

    (
        s0_l,
        s0_r,
        CrossbarFrame {
            active: true,
            op_p_vec,
            ssb_l,
            ssb_r,
        },
    )
}

/// Decodes one `4x4` `fp16` inter-block tile directly into SIMD registers (`0` stack spill array).
#[inline(always)]
fn fma_fp16_tile<B: GraphBackend>(
    acc: B::Stereo8,
    cols_fp16: &[[u16; 4]; 4],
    x: B::Stereo8,
) -> B::Stereo8 {
    let c0 = B::fp16x4_to_vec4(&cols_fp16[0]);
    let c1 = B::fp16x4_to_vec4(&cols_fp16[1]);
    let c2 = B::fp16x4_to_vec4(&cols_fp16[2]);
    let c3 = B::fp16x4_to_vec4(&cols_fp16[3]);
    let (acc_l, acc_r) = acc.into_lr();
    let (x_l, x_r) = x.into_lr();
    let p01_l = c0.mul_lane::<0>(x_l).fma_lane::<1>(c1, x_l);
    let p23_l = c2.mul_lane::<2>(x_l).fma_lane::<3>(c3, x_l);
    let p01_r = c0.mul_lane::<0>(x_r).fma_lane::<1>(c1, x_r);
    let p23_r = c2.mul_lane::<2>(x_r).fma_lane::<3>(c3, x_r);
    B::Stereo8::from_lr(acc_l.add(p01_l.add(p23_l)), acc_r.add(p01_r.add(p23_r)))
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn run_group_and_readout<B: GraphBackend>(
    g: usize,
    u_acc: B::Stereo8,
    u_l: f32,
    u_r: f32,
    cb: &CrossbarFrame<B::Vec4>,
    slab: &mut SonaHotSlab<B>,
    g_param: &SonaGroupParamsStereo4<B::Vec4>,
    ro: &SonaGroupReadoutRegs<B::Vec4, B::Stereo8>,
    pre_cab_acc: &mut B::Stereo8,
    dir_acc: &mut B::Stereo8,
) -> SonaBlockOutputStereo8<B::Stereo8> {
    let mut u_g = u_acc.add(B::Stereo8::from_lr(
        ro.in_x_weight.mul(B::Vec4::splat(u_l)),
        ro.in_x_weight.mul(B::Vec4::splat(u_r)),
    ));
    if cb.active {
        let base_p = ro.op_p_w.mul(cb.op_p_vec);
        u_g = u_g.add(B::Stereo8::from_lr(
            base_p.fma(ro.ph_w, cb.ssb_l),
            base_p.fma(ro.ph_w, cb.ssb_r),
        ));
    }
    let out = step_sona_group_stereo8::<B::Stereo8>(u_g, slab.state_mut(g), g_param);
    *pre_cab_acc = pre_cab_acc
        .fma(ro.pre_cab_odd, out.adaa_odd)
        .fma(ro.pre_cab_even, out.adaa_even)
        .fma(ro.pre_cab_lru, out.lru_state);
    *dir_acc = dir_acc
        .fma(ro.direct_odd, out.adaa_odd)
        .fma(ro.direct_even, out.adaa_even)
        .fma(ro.direct_lru, out.lru_state);
    out
}

/// Evaluates `G_CAP` groups across `4` Matryoshka blocks using register-hoisted readouts & direct `fp16` tile FMA.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_sona_core_sample<
    B: GraphBackend,
    const G_CAP: usize,
    const T_CAP: usize,
    const B0: usize,
    const B1: usize,
    const B2: usize,
    const B3: usize,
>(
    slab: &mut SonaHotSlab<B>,
    g_params: &[SonaGroupParamsStereo4<B::Vec4>; G_CAP],
    ro_regs: &[SonaGroupReadoutRegs<B::Vec4, B::Stereo8>; G_CAP],
    u_l: f32,
    u_r: f32,
    cb: &CrossbarFrame<B::Vec4>,
) -> [f32; 4] {
    let one_m_r = 1.0 - slab.lite.stage0_hp_lp_pole;
    slab.lite.stage0_hp_lp_state[0] += one_m_r * (u_l - slab.lite.stage0_hp_lp_state[0]);
    slab.lite.stage0_hp_lp_state[1] += one_m_r * (u_r - slab.lite.stage0_hp_lp_state[1]);
    let hp_l = u_l - slab.lite.stage0_hp_lp_state[0];
    let hp_r = u_r - slab.lite.stage0_hp_lp_state[1];

    let mut pre_cab_acc = B::Stereo8::zero();
    let mut dir_acc = B::Stereo8::zero();
    let zero_out = SonaBlockOutputStereo8 {
        lru_state: B::Stereo8::zero(),
        adaa_odd: B::Stereo8::zero(),
        adaa_even: B::Stereo8::zero(),
    };
    let mut blk_out: [SonaBlockOutputStereo8<B::Stereo8>; G_CAP] = [zero_out; G_CAP];

    for d in 0..B0 {
        let g = SONA_BLOCK_GROUP_SLOTS[0][d];
        blk_out[g] = run_group_and_readout(
            g,
            B::Stereo8::zero(),
            u_l,
            u_r,
            cb,
            slab,
            &g_params[g],
            &ro_regs[g],
            &mut pre_cab_acc,
            &mut dir_acc,
        );
    }

    let block_cnts = [B0, B1, B2, B3];
    for (blk, &dst_cnt) in block_cnts.iter().enumerate().skip(1) {
        for d in 0..dst_cnt {
            let dst_g = SONA_BLOCK_GROUP_SLOTS[blk][d];
            let mut acc = B::Stereo8::zero();
            for (prev_blk, &src_cnt) in block_cnts.iter().enumerate().take(blk) {
                for s in 0..src_cnt {
                    let src_g = SONA_BLOCK_GROUP_SLOTS[prev_blk][s];
                    let ti = TILE_LUT[src_g][dst_g] as usize;
                    if ti < T_CAP {
                        let tile: &SonaInterBlockTileHot = slab.tile_hot(ti);
                        let src_o = &blk_out[src_g];
                        acc = B::Stereo8::fma_mat4x4_shared(acc, &tile.w_odd_4x4, src_o.adaa_odd);
                        acc = fma_fp16_tile::<B>(acc, &tile.w_even_4x4_fp16, src_o.adaa_even);
                        acc = fma_fp16_tile::<B>(acc, &tile.w_lru_4x4_fp16, src_o.lru_state);
                    }
                }
            }
            blk_out[dst_g] = run_group_and_readout(
                dst_g,
                acc,
                u_l,
                u_r,
                cb,
                slab,
                &g_params[dst_g],
                &ro_regs[dst_g],
                &mut pre_cab_acc,
                &mut dir_acc,
            );
        }
    }

    let (mut pre_l, mut pre_r) = pre_cab_acc.horizontal_sum_lr();
    let (mut dir_l, mut dir_r) = dir_acc.horizontal_sum_lr();
    let rg = &slab.lite.ro_gains;
    pre_l += rg[0] * u_l + rg[2] * hp_l;
    pre_r += rg[1] * u_r + rg[3] * hp_r;
    dir_l += rg[4] * u_l + rg[6] * hp_l;
    dir_r += rg[5] * u_r + rg[7] * hp_r;
    [pre_l, pre_r, dir_l, dir_r]
}

#[inline(always)]
fn fir_taps_slice<B: GraphBackend>(
    slab: &SonaHotSlab<B>,
    seg: usize,
) -> (&[[f32; 4]; 4], &[[f32; 4]; 4]) {
    match seg {
        0..=1 => (&slab.lite.fir_taps_l[seg], &slab.lite.fir_taps_r[seg]),
        2 => (
            &slab.full_delta.fir_taps_l_full[0],
            &slab.full_delta.fir_taps_r_full[0],
        ),
        3..=5 => (
            &slab.pro_delta.fir_taps_l_pro[seg - 3],
            &slab.pro_delta.fir_taps_r_pro[seg - 3],
        ),
        _ => (
            &slab.ultra_delta.fir_taps_l_ultra[seg - 6],
            &slab.ultra_delta.fir_taps_r_ultra[seg - 6],
        ),
    }
}

#[inline(always)]
fn dot16_ring_seg<B: MathBackend>(
    ring: &[f32],
    s: usize,
    cap: usize,
    taps: &[[f32; 4]; 4],
    acc: B::Vec4,
) -> B::Vec4 {
    if s + 16 <= ring.len() && (cap != 64 || s <= 48) {
        let (h, _) = ring[s..s + 16].as_chunks::<4>();
        let f0 = B::Vec4::load(&h[0]).mul(B::Vec4::load(&taps[0]));
        let f1 = B::Vec4::load(&h[1]).mul(B::Vec4::load(&taps[1]));
        acc.add(f0.fma(B::Vec4::load(&h[2]), B::Vec4::load(&taps[2])))
            .add(f1.fma(B::Vec4::load(&h[3]), B::Vec4::load(&taps[3])))
    } else {
        let mut tmp = [0.0_f32; 16];
        let first = cap - s;
        tmp[..first].copy_from_slice(&ring[s..cap]);
        tmp[first..].copy_from_slice(&ring[..16 - first]);
        let (h, _) = tmp.as_chunks::<4>();
        let f0 = B::Vec4::load(&h[0]).mul(B::Vec4::load(&taps[0]));
        let f1 = B::Vec4::load(&h[1]).mul(B::Vec4::load(&taps[1]));
        acc.add(f0.fma(B::Vec4::load(&h[2]), B::Vec4::load(&taps[2])))
            .add(f1.fma(B::Vec4::load(&h[3]), B::Vec4::load(&taps[3])))
    }
}

/// Evaluates Stage III FIR + bank-outer register-hoisted Kautz across `0..len` (`<= 16`),
/// unconditionally advancing both L/R FIR histories and modal states (`ref[dsp.router.topology-dispatch]`).
#[inline(never)]
pub(crate) fn stage3_step_subblock<
    B: GraphBackend,
    const FIR_SEGS: usize,
    const KAUTZ_BANKS: usize,
    const MONO: bool,
    const CAP: usize,
>(
    slab: &mut SonaHotSlab<B>,
    pre_cab: &[[f32; 2]; CAP],
    sc: &mut CabinetSubblockScratch<CAP>,
    len: usize,
) {
    const { assert!(FIR_SEGS <= MAX_FIR_SEGMENTS && KAUTZ_BANKS <= MAX_KAUTZ_BANKS) };
    let n = len.min(CAP);
    if FIR_SEGS == 0 && KAUTZ_BANKS == 0 {
        sc.cab_out[..n].fill([0.0, 0.0]);
        return;
    }

    if (slab.lite.flags & FLAG_COLD_FIR_EXT) != 0 {
        let ext = &mut slab.cold.stage3_fir_ext;
        for i in 0..n {
            let (fir, dl, dr) = ext.step_frame::<B, MONO>(pre_cab[i][0], pre_cab[i][1]);
            sc.cab_out[i] = fir;
            sc.x_del_l[i] = dl;
            sc.x_del_r[i] = dr;
        }
    } else if FIR_SEGS <= 2 {
        let onset = (slab.lite.fir_onset_samples as usize).clamp(1, 31);
        for i in 0..n {
            let pos = (slab.lite.fir_write_pos as usize).wrapping_sub(1) & 31;
            slab.lite.fir_write_pos = pos as u16;
            let [l, r] = pre_cab[i];
            slab.lite.fir_hist_l[pos] = l;
            slab.lite.fir_hist_l[pos + 32] = l;
            slab.lite.fir_hist_r[pos] = r;
            slab.lite.fir_hist_r[pos + 32] = r;
            sc.x_del_l[i] = slab.lite.fir_hist_l[pos + onset];
            sc.x_del_r[i] = slab.lite.fir_hist_r[pos + onset];
            let (mut acc_l, mut acc_r) = (B::Vec4::zero(), B::Vec4::zero());
            for seg in 0..FIR_SEGS {
                let s = pos + (seg << 4);
                let (tl, tr) = fir_taps_slice(slab, seg);
                acc_l = dot16_ring_seg::<B>(&slab.lite.fir_hist_l, s, 64, tl, acc_l);
                acc_r = dot16_ring_seg::<B>(&slab.lite.fir_hist_r, s, 64, tr, acc_r);
            }
            let (fl, fr) = acc_l.horizontal_sum_pair(acc_r);
            sc.cab_out[i] = [fl, fr];
        }
    } else if FIR_SEGS == 3 {
        let onset = (slab.lite.fir_onset_samples as usize).clamp(1, 63);
        for i in 0..n {
            let pos = (slab.lite.fir_write_pos as usize).wrapping_sub(1) & 63;
            slab.lite.fir_write_pos = pos as u16;
            slab.lite.fir_hist_l[pos] = pre_cab[i][0];
            slab.lite.fir_hist_r[pos] = pre_cab[i][1];
            sc.x_del_l[i] = slab.lite.fir_hist_l[(pos + onset) & 63];
            sc.x_del_r[i] = slab.lite.fir_hist_r[(pos + onset) & 63];
            let (mut acc_l, mut acc_r) = (B::Vec4::zero(), B::Vec4::zero());
            for seg in 0..3 {
                let s = (pos + (seg << 4)) & 63;
                let (tl, tr) = fir_taps_slice(slab, seg);
                acc_l = dot16_ring_seg::<B>(&slab.lite.fir_hist_l, s, 64, tl, acc_l);
                acc_r = dot16_ring_seg::<B>(&slab.lite.fir_hist_r, s, 64, tr, acc_r);
            }
            let (fl, fr) = acc_l.horizontal_sum_pair(acc_r);
            sc.cab_out[i] = [fl, fr];
        }
    } else {
        let onset = (slab.lite.fir_onset_samples as usize).clamp(1, 127);
        for i in 0..n {
            let pos = (slab.lite.fir_write_pos as usize).wrapping_sub(1) & 127;
            slab.lite.fir_write_pos = pos as u16;
            let [l, r] = pre_cab[i];
            slab.pro_delta.fir_hist_l_pro[pos] = l;
            slab.pro_delta.fir_hist_l_pro[pos + 128] = l;
            slab.pro_delta.fir_hist_r_pro[pos] = r;
            slab.pro_delta.fir_hist_r_pro[pos + 128] = r;
            sc.x_del_l[i] = slab.pro_delta.fir_hist_l_pro[pos + onset];
            sc.x_del_r[i] = slab.pro_delta.fir_hist_r_pro[pos + onset];
            let (mut acc_l, mut acc_r) = (B::Vec4::zero(), B::Vec4::zero());
            for seg in 0..FIR_SEGS {
                let s = pos + (seg << 4);
                let (tl, tr) = fir_taps_slice(slab, seg);
                acc_l = dot16_ring_seg::<B>(&slab.pro_delta.fir_hist_l_pro, s, 256, tl, acc_l);
                acc_r = dot16_ring_seg::<B>(&slab.pro_delta.fir_hist_r_pro, s, 256, tr, acc_r);
            }
            let (fl, fr) = acc_l.horizontal_sum_pair(acc_r);
            sc.cab_out[i] = [fl, fr];
        }
    }

    for b in 0..KAUTZ_BANKS {
        let mut st = kautz_bank_mut(slab, b).to_kernel_state::<B::Vec4>();
        for i in 0..n {
            let (kl, kr) =
                step_kautz_quad_stereo8::<B::Stereo8>(&mut st, sc.x_del_l[i], sc.x_del_r[i]);
            sc.cab_out[i][0] += kl;
            sc.cab_out[i][1] += kr;
        }
        kautz_bank_mut(slab, b).store_kernel_state(&st);
    }
}

/// Steps Stage IV active quads `0..QUADS` across sub-block `[0..len]`, plus `48`-sample tier xfade.
#[inline(never)]
pub(crate) fn stage4_step_subblock<B: GraphBackend, const QUADS: usize, const MONO: bool>(
    slab: &mut SonaHotSlab<B>,
    in_l: &[f32; 16],
    in_r: &[f32; 16],
    out_l: &mut [f32],
    out_r: &mut [f32],
    len: usize,
) {
    let freeze = (slab.lite.flags & FLAG_FREEZE) != 0;
    let d_mod = slab.cold.ctrl_state.tank_delay_mod;
    let step_q =
        |q: usize, slab: &mut SonaHotSlab<B>, accumulate: bool, ol: &mut [f32], or: &mut [f32]| {
            let mut q_hdr = *slab.quad_hot(q);
            let inline = if q == 0 {
                Some(&mut slab.lite.quad0_frames)
            } else {
                None
            };
            let ext = if slab.cold.stage4_ext_active[q] {
                slab.cold.stage4_ext[q].as_mut()
            } else {
                None
            };
            let ap = slab.cold.stage4_schroeder[q].as_mut();
            if freeze {
                q_hdr.step_subblock::<B, MONO, true>(
                    inline, ext, ap, d_mod, in_l, in_r, ol, or, len, accumulate,
                );
            } else {
                q_hdr.step_subblock::<B, MONO, false>(
                    inline, ext, ap, d_mod, in_l, in_r, ol, or, len, accumulate,
                );
            }
            *slab.quad_hot_mut(q) = q_hdr;
        };

    step_q(0, slab, false, out_l, out_r);
    for q in 1..QUADS {
        step_q(q, slab, true, out_l, out_r);
    }

    if (slab.lite.flags & FLAG_XFADE_ACTIVE) != 0 {
        let mut rem = slab.cold.xfade_rem;
        let [prev_l, prev_r] = slab.cold.xfade_prev_out;
        let prev_mid = 0.5 * (prev_l + prev_r);
        for i in 0..len {
            if rem == 0 {
                break;
            }
            let a = (rem as f32) * TIER_XFADE_INV;
            if MONO {
                out_l[i] = prev_mid * a + out_l[i] * (1.0 - a);
                out_r[i] = out_l[i];
            } else {
                out_l[i] = prev_l * a + out_l[i] * (1.0 - a);
                out_r[i] = prev_r * a + out_r[i] * (1.0 - a);
            }
            rem -= 1;
        }
        slab.cold.xfade_rem = rem;
        if rem == 0 {
            slab.lite.flags &= !FLAG_XFADE_ACTIVE;
        }
    }
    if len > 0 {
        slab.cold.xfade_prev_out = [out_l[len - 1], out_r[len - 1]];
    }
}

/// Steps Stage IV active quads for 1 stereo frame (single-sample fallback & `4`-port frame).
#[inline(never)]
pub(crate) fn stage4_step_stereo<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    in_l: f32,
    in_r: f32,
) -> [f32; 2] {
    let freeze = (slab.lite.flags & FLAG_FREEZE) != 0;
    let mono = slab.mono_out();
    let aq = tier_tank_quads(slab.lite.tier);
    let d_mod = slab.cold.ctrl_state.tank_delay_mod;
    let mut sum_l = 0.0_f32;
    let mut sum_r = 0.0_f32;
    for q in 0..aq {
        let mut q_hdr = *slab.quad_hot(q);
        let inline = if q == 0 {
            Some(&mut slab.lite.quad0_frames)
        } else {
            None
        };
        let ext = if slab.cold.stage4_ext_active[q] {
            slab.cold.stage4_ext[q].as_mut()
        } else {
            None
        };
        let ap = slab.cold.stage4_schroeder[q].as_mut();
        let [ql, qr] = q_hdr.step_stereo::<B>(inline, ext, ap, freeze, d_mod, in_l, in_r);
        *slab.quad_hot_mut(q) = q_hdr;
        sum_l += ql;
        sum_r += qr;
    }
    if mono {
        let mid = 0.5 * (sum_l + sum_r);
        sum_l = mid;
        sum_r = mid;
    }
    if (slab.lite.flags & FLAG_XFADE_ACTIVE) != 0 && slab.cold.xfade_rem > 0 {
        let a = (slab.cold.xfade_rem as f32) * TIER_XFADE_INV;
        let [pl, pr] = slab.cold.xfade_prev_out;
        if mono {
            let pm = 0.5 * (pl + pr);
            sum_l = pm * a + sum_l * (1.0 - a);
            sum_r = sum_l;
        } else {
            sum_l = pl * a + sum_l * (1.0 - a);
            sum_r = pr * a + sum_r * (1.0 - a);
        }
        slab.cold.xfade_rem -= 1;
        if slab.cold.xfade_rem == 0 {
            slab.lite.flags &= !FLAG_XFADE_ACTIVE;
        }
    }
    slab.cold.xfade_prev_out = [sum_l, sum_r];
    [sum_l, sum_r]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::sona_core::model::SonaColdTierModel;
    use crate::math::DefaultMathBackend;
    use crate::types::{FtzDazGuard, Tier};

    #[test]
    fn test_crossbar_tile_lut_zero_stage3_and_stage4_xfade_mono_termination() {
        let _g = FtzDazGuard::new();
        let lut = core::hint::black_box(build_tile_lut());
        assert_eq!(lut, TILE_LUT);

        let cold = SonaColdTierModel::identity_for_tier(Tier::Lite, false, true);
        let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
        let mut sc = CabinetSubblockScratch::<16>::zeroed();
        sc.cab_out[..4].fill([1.0, -1.0]);
        stage3_step_subblock::<DefaultMathBackend, 0, 0, false, 16>(
            &mut slab,
            &[[0.5, -0.5]; 16],
            &mut sc,
            4,
        );
        assert_eq!(sc.cab_out[..4], [[0.0, 0.0]; 4]);

        slab.lite.flags |= FLAG_XFADE_ACTIVE;
        slab.cold.xfade_rem = 3;
        slab.cold.xfade_prev_out = [0.2, -0.1];
        let in_l = [0.4_f32; 16];
        let in_r = [-0.2_f32; 16];
        let mut out_l = [0.0_f32; 8];
        let mut out_r = [0.0_f32; 8];
        stage4_step_subblock::<DefaultMathBackend, 1, true>(
            &mut slab, &in_l, &in_r, &mut out_l, &mut out_r, 8,
        );
        assert_eq!(slab.cold.xfade_rem, 0);
        assert_eq!(slab.lite.flags & FLAG_XFADE_ACTIVE, 0);
        assert_eq!(out_l[0].to_bits(), out_r[0].to_bits());

        let prev_saved = slab.cold.xfade_prev_out;
        stage4_step_subblock::<DefaultMathBackend, 1, false>(
            &mut slab, &in_l, &in_r, &mut out_l, &mut out_r, 0,
        );
        assert_eq!(
            slab.cold.xfade_prev_out.map(f32::to_bits),
            prev_saved.map(f32::to_bits)
        );
    }

    #[test]
    fn test_stage4_step_stereo_mono_and_stereo_xfade_completion() {
        let _g = FtzDazGuard::new();
        let mut cold_mono = SonaColdTierModel::identity_for_tier(Tier::Full, false, true);
        cold_mono.stage4.wet_mix = 0.5;
        cold_mono.stage4.g_loop = 0.7;
        cold_mono.stage4.allpass_delays = [7, 11, 13, 17];
        cold_mono.stage4.allpass_g = [0.5; 4];
        let mut slab_mono = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold_mono);
        slab_mono.lite.flags |= FLAG_XFADE_ACTIVE;
        slab_mono.cold.xfade_rem = 2;
        slab_mono.cold.xfade_prev_out = [0.3, -0.1];
        let y0 = stage4_step_stereo(&mut slab_mono, 0.4, -0.2);
        assert_eq!(y0[0].to_bits(), y0[1].to_bits());
        assert_eq!(slab_mono.cold.xfade_rem, 1);
        let y1 = stage4_step_stereo(&mut slab_mono, 0.2, 0.1);
        assert_eq!(y1[0].to_bits(), y1[1].to_bits());
        assert_eq!(slab_mono.cold.xfade_rem, 0);
        assert_eq!(slab_mono.lite.flags & FLAG_XFADE_ACTIVE, 0);

        let mut cold_st = SonaColdTierModel::identity_for_tier(Tier::Pro, false, false);
        cold_st.stage4.wet_mix = 0.5;
        cold_st.stage4.g_loop = 0.7;
        let mut slab_st = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold_st);
        slab_st.ensure_cold_tank_cap(&cold_st.stage4, 256);
        slab_st.lite.flags |= FLAG_XFADE_ACTIVE;
        slab_st.cold.xfade_rem = 2;
        slab_st.cold.xfade_prev_out = [0.25, -0.25];
        let s0 = stage4_step_stereo(&mut slab_st, 0.5, -0.3);
        let s1 = stage4_step_stereo(&mut slab_st, 0.1, 0.4);
        assert!(s0.iter().all(|v| v.is_finite()) && s1.iter().all(|v| v.is_finite()));
        assert_eq!(slab_st.cold.xfade_rem, 0);
        assert_eq!(slab_st.lite.flags & FLAG_XFADE_ACTIVE, 0);
    }
}
