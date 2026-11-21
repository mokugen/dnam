//! Register-hoisted causal Matryoshka group core evaluation, Top-of-Frame `Op-P` / `Q0.32`
//! carrier projection, and `hp_lp` core substep (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    CarrierBasisWeightsVec4, SonaBlockStateStereo8, SonaGroupParamsStereo4, SonaParaLruCoeffs4,
    SonaToneWeights4, StereoAnalyticQuadrature, StereoCarrierRotorVec4, StereoVec8, TptSvfCoeffs4,
    eval_carrier_3basis_stereo_vec4, os2x_midpoint_f32x4, step_sona_group_stereo8,
};

use crate::graph::GraphBackend;
use crate::math::{SimdVec4, flush_recursive_state_f32, fp16_to_f32};
use crate::types::Tier;

use super::super::layout::{
    CrossbarFrame, FLAG_CROSSBAR_ACTIVE, MAX_SONA_GROUPS, MAX_SONA_TILES, SONA_INTERBLOCK_TILES,
    SONA_ULTRA_GROUP_TRAVERSAL_ORDER, SonaGroupReadoutRegs, SonaHotSlab, SonaInterBlockTileHot,
};

const MAX_OUTGOING_PER_GROUP: usize = 6;
const TILE_EMPTY: u8 = 0xFF;

/// Compile-time outgoing tile index table per source group `0..10` covering all `35` tiles.
const SONA_SRC_OUTGOING_TILES: [[u8; MAX_OUTGOING_PER_GROUP]; MAX_SONA_GROUPS] =
    build_outgoing_tile_table();

const fn build_outgoing_tile_table() -> [[u8; MAX_OUTGOING_PER_GROUP]; MAX_SONA_GROUPS] {
    let mut table = [[TILE_EMPTY; MAX_OUTGOING_PER_GROUP]; MAX_SONA_GROUPS];
    let mut counts = [0_usize; MAX_SONA_GROUPS];
    let mut t = 0_usize;
    while t < MAX_SONA_TILES {
        let sg = SONA_INTERBLOCK_TILES[t].0;
        let slot = counts[sg];
        table[sg][slot] = t as u8;
        counts[sg] = slot + 1;
        t += 1;
    }
    table
}

const fn verify_outgoing_tile_table() -> bool {
    let mut total = 0_usize;
    let mut g = 0_usize;
    while g < MAX_SONA_GROUPS {
        let mut s = 0_usize;
        while s < MAX_OUTGOING_PER_GROUP {
            if SONA_SRC_OUTGOING_TILES[g][s] != TILE_EMPTY {
                total += 1;
            }
            s += 1;
        }
        g += 1;
    }
    total == MAX_SONA_TILES
}

const _: () = assert!(verify_outgoing_tile_table());

/// `1x` host-rate `Op-P` analytic quadrature & `T1` 4-feature output prior to optional `OS2X`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OpP1xFrame {
    pub iq_lr: [f32; 4],
    pub feat: [f32; 4],
}

/// Register-resident hoisted Matryoshka core parameters + crossbars held constant across a
/// `1`- or `4`-frame micro-chunk (`0-ULP` parity).
#[derive(Clone, Copy, Debug)]
pub struct HoistedCoreChunk<B: GraphBackend> {
    pub num_groups: usize,
    pub num_tiles: usize,
    pub params: [SonaGroupParamsStereo4<B::Vec4>; MAX_SONA_GROUPS],
    pub readouts: [SonaGroupReadoutRegs<B::Vec4, B::Stereo8>; MAX_SONA_GROUPS],
    pub carrier_w: CarrierBasisWeightsVec4<B::Vec4>,
    pub cos_st: B::Vec4,
    pub sin_st: B::Vec4,
    pub ro_gains: [f32; 8],
    pub op_p_gain: f32,
    pub op_t_gain: f32,
    pub wet_scale: f32,
    pub hp_lp_pole: f32,
}

#[inline(always)]
fn zero_group_params<B: GraphBackend>() -> SonaGroupParamsStereo4<B::Vec4> {
    let z = B::Vec4::zero();
    SonaGroupParamsStereo4 {
        svf: TptSvfCoeffs4 {
            a1: z,
            a2: z,
            a3: z,
            k: z,
            two_k: z,
        },
        tone: SonaToneWeights4 {
            lp: z,
            bp: z,
            hp: z,
            ap: z,
            dir: z,
        },
        lru: SonaParaLruCoeffs4 {
            alpha_rel: z,
            alpha_atk: z,
            kappa: z,
        },
        rect_sign: B::Vec4::splat(1.0),
        sag_depth: z,
        adaa_drive: z,
        lru_bias_drive: z,
        wavefolder_mix: z,
    }
}

#[inline(always)]
fn zero_group_readouts<B: GraphBackend>() -> SonaGroupReadoutRegs<B::Vec4, B::Stereo8> {
    let z = B::Vec4::zero();
    let zs = B::Stereo8::zero();
    SonaGroupReadoutRegs {
        in_x_weight: z,
        op_p_w: z,
        ph_w: z,
        pre_cab_odd: zs,
        pre_cab_even: zs,
        pre_cab_lru: zs,
        direct_odd: zs,
        direct_even: zs,
        direct_lru: zs,
    }
}

#[inline(always)]
fn hoist_group_params_into<B: GraphBackend>(
    slab: &SonaHotSlab<'_, B>,
    num_groups: usize,
    params: &mut [SonaGroupParamsStereo4<B::Vec4>; MAX_SONA_GROUPS],
) {
    let cold = slab.cold();
    for g in 0..num_groups {
        let gh = slab.group_hot(g);
        let l0 = g * 4;
        params[g] = gh.to_kernel_params_modulated::<B>(
            B::Vec4::load(&[
                cold.lane_g_scale[l0],
                cold.lane_g_scale[l0 + 1],
                cold.lane_g_scale[l0 + 2],
                cold.lane_g_scale[l0 + 3],
            ]),
            B::Vec4::load(&[
                cold.lane_k_delta[l0],
                cold.lane_k_delta[l0 + 1],
                cold.lane_k_delta[l0 + 2],
                cold.lane_k_delta[l0 + 3],
            ]),
            cold.group_adaa_scale[g],
            cold.group_lru_bias[g],
            B::Vec4::load(&[
                cold.lane_wf_delta[l0],
                cold.lane_wf_delta[l0 + 1],
                cold.lane_wf_delta[l0 + 2],
                cold.lane_wf_delta[l0 + 3],
            ]),
            slab.rect_mask4(g),
        );
    }
}

impl<B: GraphBackend> HoistedCoreChunk<B> {
    /// Returns a zero-initialized core snapshot (`!alloc`, `0` NR calls) for unhydrated slabs.
    #[must_use]
    pub fn empty() -> Self {
        let z = B::Vec4::zero();
        Self {
            num_groups: 0,
            num_tiles: 0,
            params: [zero_group_params::<B>(); MAX_SONA_GROUPS],
            readouts: [zero_group_readouts::<B>(); MAX_SONA_GROUPS],
            carrier_w: CarrierBasisWeightsVec4 {
                w0: z,
                w1: z,
                w2: z,
            },
            cos_st: B::Vec4::splat(1.0),
            sin_st: z,
            ro_gains: [0.0; 8],
            op_p_gain: 0.0,
            op_t_gain: 0.0,
            wet_scale: 1.0,
            hp_lp_pole: 0.0,
        }
    }

    /// Materializes active group DSP parameters, readout matrices, and crossbar weights into registers.
    #[inline]
    pub fn from_slab(slab: &SonaHotSlab<'_, B>) -> Self {
        let tier = slab.lite.tier;
        let num_groups = tier.simd_groups();
        let num_tiles = match tier {
            Tier::Lite => 3,
            Tier::Full => 13,
            Tier::Pro => 23,
            Tier::Ultra => 35,
        };
        let cold = slab.cold();
        let wet_scale =
            crate::control::bus::laws::eval_crossbar_wet_scale(cold.latched_stage_params[15]);
        let mut params = [zero_group_params::<B>(); MAX_SONA_GROUPS];
        let mut readouts = [zero_group_readouts::<B>(); MAX_SONA_GROUPS];
        hoist_group_params_into(slab, num_groups, &mut params);
        for g in 0..num_groups {
            let gh = slab.group_hot(g);
            readouts[g] = slab.readout_hot(g).decode_regs::<B>(
                gh.in_x_weight_vec4::<B>(),
                B::fp16x4_to_vec4(slab.op_p_to_group_fp16(g)),
                B::fp16x4_to_vec4(slab.phasor_to_group_fp16(g)),
            );
        }

        let mut ro_gains = [0.0_f32; 8];
        for (i, &bits) in slab.lite.ro_gains_fp16.iter().enumerate() {
            ro_gains[i] = fp16_to_f32(bits);
        }
        let carrier_w = CarrierBasisWeightsVec4 {
            w0: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[0]),
            w1: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[1]),
            w2: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[2]),
        };
        Self {
            num_groups,
            num_tiles,
            params,
            readouts,
            carrier_w,
            cos_st: B::fp16x4_to_vec4(&slab.lite.phasor_st_cos_fp16),
            sin_st: B::fp16x4_to_vec4(&slab.lite.phasor_st_sin_fp16),
            ro_gains,
            op_p_gain: cold.base_op_p_gain * wet_scale,
            op_t_gain: cold.base_op_t_gain * wet_scale,
            wet_scale,
            hp_lp_pole: slab.lite.hp_lp_pole,
        }
    }

    /// Updates only the `3 kHz`/`12 kHz` modulated group parameters and crossbar gains while
    /// preserving static readout/carrier/phasor matrices decoded at tier/rate hydration.
    #[inline]
    pub fn refresh_modulated(&mut self, slab: &SonaHotSlab<'_, B>) {
        let cold = slab.cold();
        let wet_scale =
            crate::control::bus::laws::eval_crossbar_wet_scale(cold.latched_stage_params[15]);
        hoist_group_params_into(slab, self.num_groups, &mut self.params);
        self.op_p_gain = cold.base_op_p_gain * wet_scale;
        self.op_t_gain = cold.base_op_t_gain * wet_scale;
        self.wet_scale = wet_scale;
    }
}

#[inline(always)]
fn accum_tile_col<B: GraphBackend, const COL: usize>(
    acc_l: &mut B::Vec4,
    acc_r: &mut B::Vec4,
    tile: &SonaInterBlockTileHot,
    odd: (B::Vec4, B::Vec4),
    ev: (B::Vec4, B::Vec4),
    lr: (B::Vec4, B::Vec4),
) {
    let wo = B::Vec4::load(&tile.w_odd_4x4[COL]);
    let we = B::fp16x4_to_vec4(&tile.w_even_4x4_fp16[COL]);
    let wl = B::fp16x4_to_vec4(&tile.w_lru_4x4_fp16[COL]);
    *acc_l = acc_l
        .fma_lane::<COL>(wo, odd.0)
        .fma_lane::<COL>(we, ev.0)
        .fma_lane::<COL>(wl, lr.0);
    *acc_r = acc_r
        .fma_lane::<COL>(wo, odd.1)
        .fma_lane::<COL>(we, ev.1)
        .fma_lane::<COL>(wl, lr.1);
}

#[inline(always)]
fn accum_tile_4x4<B: GraphBackend>(
    dst: &mut B::Stereo8,
    tile: &SonaInterBlockTileHot,
    out_odd: B::Stereo8,
    out_even: B::Stereo8,
    out_lru: B::Stereo8,
) {
    let odd = out_odd.into_lr();
    let ev = out_even.into_lr();
    let lr = out_lru.into_lr();
    let (mut acc_l, mut acc_r) = (*dst).into_lr();
    accum_tile_col::<B, 0>(&mut acc_l, &mut acc_r, tile, odd, ev, lr);
    accum_tile_col::<B, 1>(&mut acc_l, &mut acc_r, tile, odd, ev, lr);
    accum_tile_col::<B, 2>(&mut acc_l, &mut acc_r, tile, odd, ev, lr);
    accum_tile_col::<B, 3>(&mut acc_l, &mut acc_r, tile, odd, ev, lr);
    *dst = B::Stereo8::from_lr(acc_l, acc_r);
}

struct HotStatesMut<'a, B: GraphBackend> {
    lite: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 4],
    full: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 2],
    pro: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 2],
    ultra: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 2],
}

impl<B: GraphBackend> HotStatesMut<'_, B> {
    #[inline(always)]
    fn state_mut(&mut self, g: usize) -> &mut SonaBlockStateStereo8<B::Stereo8> {
        match g {
            0..=3 => &mut self.lite[g],
            4..=5 => &mut self.full[g - 4],
            6..=7 => &mut self.pro[g - 6],
            _ => &mut self.ultra[g - 8],
        }
    }
}

struct HotTilesRef<'a> {
    lite: &'a [SonaInterBlockTileHot; 3],
    full: &'a [SonaInterBlockTileHot; 10],
    pro: &'a [SonaInterBlockTileHot; 10],
    ultra: &'a [SonaInterBlockTileHot; 12],
}

impl HotTilesRef<'_> {
    #[inline(always)]
    fn tile_hot(&self, t: usize) -> &SonaInterBlockTileHot {
        match t {
            0..=2 => &self.lite[t],
            3..=12 => &self.full[t - 3],
            13..=22 => &self.pro[t - 13],
            _ => &self.ultra[t - 23],
        }
    }
}

#[inline(always)]
fn evaluate_matryoshka_groups_disjoint<B: GraphBackend>(
    states: &mut HotStatesMut<'_, B>,
    tiles: &HotTilesRef<'_>,
    hoisted: &HoistedCoreChunk<B>,
    crossbar: &CrossbarFrame<B::Vec4>,
    u_lr: (f32, f32),
) -> ((f32, f32), (f32, f32)) {
    let in_s8 = B::Stereo8::from_lr(B::Vec4::splat(u_lr.0), B::Vec4::splat(u_lr.1));
    let mut group_in = [B::Stereo8::zero(); MAX_SONA_GROUPS];
    let mut pre_cab_acc = B::Stereo8::zero();
    let mut direct_acc = B::Stereo8::zero();

    for &g in &SONA_ULTRA_GROUP_TRAVERSAL_ORDER {
        if g >= hoisted.num_groups {
            continue;
        }
        let reg = &hoisted.readouts[g];
        let mut u_g = in_s8
            .mul(B::Stereo8::from_lr(reg.in_x_weight, reg.in_x_weight))
            .add(group_in[g]);
        if crossbar.active {
            let op_p_s8 = B::Stereo8::from_lr(
                crossbar.op_p_vec.mul(reg.op_p_w),
                crossbar.op_p_vec.mul(reg.op_p_w),
            );
            let car_s8 =
                B::Stereo8::from_lr(crossbar.car_l.mul(reg.ph_w), crossbar.car_r.mul(reg.ph_w));
            u_g = u_g.add(op_p_s8).add(car_s8);
        }

        let out =
            step_sona_group_stereo8::<B::Stereo8>(u_g, states.state_mut(g), &hoisted.params[g]);
        pre_cab_acc = pre_cab_acc
            .fma(out.adaa_odd, reg.pre_cab_odd)
            .fma(out.adaa_even, reg.pre_cab_even)
            .fma(out.lru_state, reg.pre_cab_lru);
        direct_acc = direct_acc
            .fma(out.adaa_odd, reg.direct_odd)
            .fma(out.adaa_even, reg.direct_even)
            .fma(out.lru_state, reg.direct_lru);

        for &tile_raw in &SONA_SRC_OUTGOING_TILES[g] {
            if tile_raw == TILE_EMPTY {
                break;
            }
            let t_idx = usize::from(tile_raw);
            if t_idx >= hoisted.num_tiles {
                continue;
            }
            let dg = SONA_INTERBLOCK_TILES[t_idx].1;
            accum_tile_4x4::<B>(
                &mut group_in[dg],
                tiles.tile_hot(t_idx),
                out.adaa_odd,
                out.adaa_even,
                out.lru_state,
            );
        }
    }

    let (pl, pr) = pre_cab_acc.into_lr();
    let (dl, dr) = direct_acc.into_lr();
    (pl.horizontal_sum_pair(pr), dl.horizontal_sum_pair(dr))
}

/// Evaluates all active Matryoshka core groups in causal block-major order (`0..num_groups`),
/// accumulating inter-block `4x4` tiles and `VarPro` readouts into `((pre_cab_l, r), (dir_l, r))`.
#[inline(always)]
pub fn evaluate_matryoshka_groups<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    hoisted: &HoistedCoreChunk<B>,
    crossbar: &CrossbarFrame<B::Vec4>,
    u_lr: (f32, f32),
) -> ((f32, f32), (f32, f32)) {
    let mut states = HotStatesMut {
        lite: &mut slab.lite.core_states,
        full: &mut slab.full_delta.core_states_full,
        pro: &mut slab.pro_delta.core_states_pro,
        ultra: &mut slab.ultra_delta.core_states_ultra,
    };
    let tiles = HotTilesRef {
        lite: &slab.lite.tiles,
        full: &slab.full_delta.tiles_full,
        pro: &slab.pro_delta.tiles_pro,
        ultra: &slab.ultra_delta.tiles_ultra,
    };
    evaluate_matryoshka_groups_disjoint(&mut states, &tiles, hoisted, crossbar, u_lr)
}

#[inline(always)]
pub(crate) fn step_op_p_1x<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
) -> OpP1xFrame {
    let (dc_l, dc_r) = slab.lite.t1_dc.step_stereo(in_l, in_r);
    let ((i_l, q_l), (i_r, q_r)) = slab.lite.hilbert.step_stereo_recording(
        dc_l,
        dc_r,
        &mut slab.cold.as_mut_slice()[0].rem_hist,
    );
    let feat = slab.lite.t1.step(0.5 * (i_l + i_r), 0.5 * (q_l + q_r));
    OpP1xFrame {
        iq_lr: [i_l, q_l, i_r, q_r],
        feat,
    }
}

#[inline(always)]
pub(crate) fn step_core_subsample<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    op_p_1x: &OpP1xFrame,
    os2x: bool,
    sub_idx: usize,
    u_lr: (f32, f32),
) -> [f32; 4] {
    let hoisted = &slab.cold.as_slice()[0].hoisted_core;
    let crossbar = if (slab.lite.flags & FLAG_CROSSBAR_ACTIVE) != 0 {
        let (iq_sub, feat_sub) = if !os2x {
            (op_p_1x.iq_lr, op_p_1x.feat)
        } else if sub_idx == 0 {
            (
                os2x_midpoint_f32x4(slab.os2x_hot.op_p_prev_iq_lr, op_p_1x.iq_lr),
                os2x_midpoint_f32x4(slab.os2x_hot.op_p_prev_feat, op_p_1x.feat),
            )
        } else {
            slab.os2x_hot.op_p_prev_iq_lr = op_p_1x.iq_lr;
            slab.os2x_hot.op_p_prev_feat = op_p_1x.feat;
            (op_p_1x.iq_lr, op_p_1x.feat)
        };
        let (cos_l, sin_l) = slab.lite.phasors.step_quadrature_vec4::<B::Vec4>();
        let rotor = StereoCarrierRotorVec4::from_left_and_offset(
            cos_l,
            sin_l,
            hoisted.cos_st,
            hoisted.sin_st,
        );
        let quad = StereoAnalyticQuadrature {
            i_l: iq_sub[0],
            q_l: iq_sub[1],
            i_r: iq_sub[2],
            q_r: iq_sub[3],
        };
        let (car_l, car_r) =
            eval_carrier_3basis_stereo_vec4(&quad, &rotor, &hoisted.carrier_w, hoisted.wet_scale);
        CrossbarFrame {
            active: true,
            op_p_vec: B::Vec4::load(&feat_sub).mul(B::Vec4::splat(hoisted.op_p_gain)),
            car_l,
            car_r,
        }
    } else {
        if os2x && sub_idx == 1 {
            slab.os2x_hot.op_p_prev_iq_lr = op_p_1x.iq_lr;
            slab.os2x_hot.op_p_prev_feat = op_p_1x.feat;
        }
        slab.lite.phasors.advance();
        CrossbarFrame {
            active: false,
            op_p_vec: B::Vec4::zero(),
            car_l: B::Vec4::zero(),
            car_r: B::Vec4::zero(),
        }
    };

    let mut states = HotStatesMut {
        lite: &mut slab.lite.core_states,
        full: &mut slab.full_delta.core_states_full,
        pro: &mut slab.pro_delta.core_states_pro,
        ultra: &mut slab.ultra_delta.core_states_ultra,
    };
    let tiles = HotTilesRef {
        lite: &slab.lite.tiles,
        full: &slab.full_delta.tiles_full,
        pro: &slab.pro_delta.tiles_pro,
        ultra: &slab.ultra_delta.tiles_ultra,
    };
    let ((pre_l, pre_r), (dir_l, dir_r)) =
        evaluate_matryoshka_groups_disjoint(&mut states, &tiles, hoisted, &crossbar, u_lr);

    let pole = hoisted.hp_lp_pole;
    let one_m_p = 1.0 - pole;
    let lp_l = flush_recursive_state_f32(pole * slab.lite.hp_lp_state[0] + one_m_p * u_lr.0);
    let lp_r = flush_recursive_state_f32(pole * slab.lite.hp_lp_state[1] + one_m_p * u_lr.1);
    slab.lite.hp_lp_state = [lp_l, lp_r];
    let (hp_l, hp_r) = (u_lr.0 - lp_l, u_lr.1 - lp_r);
    let rg = &hoisted.ro_gains;
    [
        pre_l + rg[0] * u_lr.0 + rg[2] * hp_l,
        pre_r + rg[1] * u_lr.1 + rg[3] * hp_r,
        dir_l + rg[4] * u_lr.0 + rg[6] * hp_l,
        dir_r + rg[5] * u_lr.1 + rg[7] * hp_r,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outgoing_tile_table_partition_exact() {
        let mut seen = [false; MAX_SONA_TILES];
        for row in &SONA_SRC_OUTGOING_TILES {
            for &t in row {
                if t != TILE_EMPTY {
                    assert!(!seen[usize::from(t)]);
                    seen[usize::from(t)] = true;
                }
            }
        }
        assert!(seen.iter().all(|&v| v));
    }
}
