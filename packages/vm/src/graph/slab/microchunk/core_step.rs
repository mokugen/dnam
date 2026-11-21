//! Register-hoisted causal Matryoshka `CoreQuad` evaluation, Top-of-Frame `Op-P` / `Q0.32`
//! carrier projection, and `hp_lp` core substep (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    CarrierBasisWeightsVec4, CoreQuadParamsStereo4, SonaBlockStateStereo8, SonaParaLruDuhemCoeffs4,
    SonaToneWeights4, StereoAnalyticQuadrature, StereoCarrierRotorVec4, StereoVec8, TptSvfCoeffs4,
    couple_ms_input_stereo8, eval_carrier_3basis_stereo_vec4, os2x_midpoint_f32x4,
    step_core_quad_stereo8,
};

use crate::graph::GraphBackend;
use crate::math::{SimdVec4, flush_recursive_state_f32, fp16_to_f32};
use crate::types::Tier;

use super::super::layout::{
    CoreQuadReadoutRegs, CrossbarFrame, FLAG_CROSSBAR_ACTIVE, MAX_CORE_QUADS, MAX_SONA_TILES,
    SONA_INTERBLOCK_TILES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER, SonaHotSlab, SonaInterBlockTileHot,
};

const MAX_OUTGOING_PER_QUAD: usize = 6;
const TILE_EMPTY: u8 = 0xFF;

/// Compile-time outgoing tile index table per source quad `0..10` covering all `35` tiles.
const SONA_SRC_OUTGOING_TILES: [[u8; MAX_OUTGOING_PER_QUAD]; MAX_CORE_QUADS] =
    build_outgoing_tile_table();

const fn build_outgoing_tile_table() -> [[u8; MAX_OUTGOING_PER_QUAD]; MAX_CORE_QUADS] {
    let mut table = [[TILE_EMPTY; MAX_OUTGOING_PER_QUAD]; MAX_CORE_QUADS];
    let mut counts = [0_usize; MAX_CORE_QUADS];
    let mut t = 0_usize;
    while t < MAX_SONA_TILES {
        let sq = SONA_INTERBLOCK_TILES[t].0;
        let slot = counts[sq];
        table[sq][slot] = t as u8;
        counts[sq] = slot + 1;
        t += 1;
    }
    table
}

const fn verify_outgoing_tile_table() -> bool {
    let mut total = 0_usize;
    let mut q = 0_usize;
    while q < MAX_CORE_QUADS {
        let mut s = 0_usize;
        while s < MAX_OUTGOING_PER_QUAD {
            if SONA_SRC_OUTGOING_TILES[q][s] != TILE_EMPTY {
                total += 1;
            }
            s += 1;
        }
        q += 1;
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
    pub num_quads: usize,
    pub num_tiles: usize,
    pub params: [CoreQuadParamsStereo4<B::Vec4>; MAX_CORE_QUADS],
    pub readouts: [CoreQuadReadoutRegs<B::Vec4, B::Stereo8>; MAX_CORE_QUADS],
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
fn zero_quad_params<B: GraphBackend>() -> CoreQuadParamsStereo4<B::Vec4> {
    let z = B::Vec4::zero();
    CoreQuadParamsStereo4 {
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
        lru: SonaParaLruDuhemCoeffs4::from_clamped(z, z, z, z, z, z),
        two_svf_g: z,
        rect_sign: B::Vec4::splat(1.0),
        sag_depth: z,
        adaa_drive: z,
        lru_bias_drive: z,
        wavefolder_mix: z,
        glu_mix: z,
    }
}

#[inline(always)]
fn zero_quad_readouts<B: GraphBackend>() -> CoreQuadReadoutRegs<B::Vec4, B::Stereo8> {
    let z = B::Vec4::zero();
    let zs = B::Stereo8::zero();
    CoreQuadReadoutRegs {
        in_mid_weight: z,
        in_side_weight: z,
        op_p_w: z,
        ph_w: z,
        pre_conv_odd: zs,
        pre_conv_even: zs,
        pre_conv_lru: zs,
        direct_odd: zs,
        direct_even: zs,
        direct_lru: zs,
    }
}

#[inline(always)]
fn hoist_quad_params_into<B: GraphBackend>(
    slab: &SonaHotSlab<'_, B>,
    num_quads: usize,
    params: &mut [CoreQuadParamsStereo4<B::Vec4>; MAX_CORE_QUADS],
) {
    let cold = slab.cold();
    for q in 0..num_quads {
        let qh = slab.core_quad_hot(q);
        let ext = slab.core_quad_ext(q);
        let l0 = q * 4;
        params[q] = qh.to_kernel_params_modulated::<B>(
            ext,
            crate::graph::slab::CoreQuadModulation {
                g_scale: B::Vec4::load(&[
                    cold.lane_g_scale[l0],
                    cold.lane_g_scale[l0 + 1],
                    cold.lane_g_scale[l0 + 2],
                    cold.lane_g_scale[l0 + 3],
                ]),
                k_delta: B::Vec4::load(&[
                    cold.lane_k_delta[l0],
                    cold.lane_k_delta[l0 + 1],
                    cold.lane_k_delta[l0 + 2],
                    cold.lane_k_delta[l0 + 3],
                ]),
                adaa_scale: cold.quad_adaa_scale[q],
                bias_delta: cold.quad_lru_bias[q],
                wf_delta: B::Vec4::load(&[
                    cold.lane_wf_delta[l0],
                    cold.lane_wf_delta[l0 + 1],
                    cold.lane_wf_delta[l0 + 2],
                    cold.lane_wf_delta[l0 + 3],
                ]),
                rect_mask4: slab.rect_mask4(q),
            },
        );
    }
}

impl<B: GraphBackend> HoistedCoreChunk<B> {
    /// Returns a zero-initialized core snapshot (`!alloc`, `0` NR calls) for unhydrated slabs.
    #[must_use]
    pub fn empty() -> Self {
        let z = B::Vec4::zero();
        Self {
            num_quads: 0,
            num_tiles: 0,
            params: [zero_quad_params::<B>(); MAX_CORE_QUADS],
            readouts: [zero_quad_readouts::<B>(); MAX_CORE_QUADS],
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

    /// Materializes active quad DSP parameters, readout matrices, and crossbar weights into registers.
    #[inline]
    pub fn from_slab(slab: &SonaHotSlab<'_, B>) -> Self {
        let tier = slab.lite.tier;
        let num_quads = tier.simd_groups();
        let num_tiles = match tier {
            Tier::Lite => 3,
            Tier::Full => 13,
            Tier::Pro => 23,
            Tier::Ultra => 35,
        };
        let cold = slab.cold();
        let wet_scale =
            crate::control::bus::laws::eval_crossbar_wet_scale(cold.latched_stage_params[15]);
        let mut params = [zero_quad_params::<B>(); MAX_CORE_QUADS];
        let mut readouts = [zero_quad_readouts::<B>(); MAX_CORE_QUADS];
        hoist_quad_params_into(slab, num_quads, &mut params);
        for q in 0..num_quads {
            let qh = slab.core_quad_hot(q);
            let ext = slab.core_quad_ext(q);
            readouts[q] = slab.readout_hot(q).decode_regs::<B>(
                qh.in_mid_weight_vec4::<B>(),
                ext.in_side_weight_vec4::<B>(),
                B::fp16x4_to_vec4(slab.op_p_to_core_fp16(q)),
                B::fp16x4_to_vec4(slab.phasor_to_core_fp16(q)),
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
            num_quads,
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

    /// Updates only the `3 kHz`/`12 kHz` modulated quad parameters and crossbar gains while
    /// preserving static readout/carrier/phasor matrices decoded at tier/rate hydration.
    #[inline]
    pub fn refresh_modulated(&mut self, slab: &SonaHotSlab<'_, B>) {
        let cold = slab.cold();
        let wet_scale =
            crate::control::bus::laws::eval_crossbar_wet_scale(cold.latched_stage_params[15]);
        hoist_quad_params_into(slab, self.num_quads, &mut self.params);
        self.op_p_gain = cold.base_op_p_gain * wet_scale;
        self.op_t_gain = cold.base_op_t_gain * wet_scale;
        self.wet_scale = wet_scale;
    }
}

#[inline(always)]
fn fma_fp16_tile<B: GraphBackend>(
    acc: B::Stereo8,
    cols_fp16: &[[u16; 4]; 4],
    x: B::Stereo8,
) -> B::Stereo8 {
    let cols = [
        B::fp16x4_to_vec4(&cols_fp16[0]),
        B::fp16x4_to_vec4(&cols_fp16[1]),
        B::fp16x4_to_vec4(&cols_fp16[2]),
        B::fp16x4_to_vec4(&cols_fp16[3]),
    ];
    let (acc_l, acc_r) = acc.into_lr();
    let (x_l, x_r) = x.into_lr();
    let (nl, nr) = B::fma_cols4_stereo(acc_l, acc_r, cols, x_l, x_r);
    B::Stereo8::from_lr(nl, nr)
}

struct HotStatesMut<'a, B: GraphBackend> {
    lite: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 4],
    full: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 2],
    pro: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 2],
    ultra: &'a mut [SonaBlockStateStereo8<B::Stereo8>; 2],
}

impl<B: GraphBackend> HotStatesMut<'_, B> {
    #[inline(always)]
    fn state_mut(&mut self, q: usize) -> &mut SonaBlockStateStereo8<B::Stereo8> {
        match q {
            0..=3 => &mut self.lite[q],
            4..=5 => &mut self.full[q - 4],
            6..=7 => &mut self.pro[q - 6],
            _ => &mut self.ultra[q - 8],
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
fn evaluate_matryoshka_core_quads_disjoint<B: GraphBackend>(
    states: &mut HotStatesMut<'_, B>,
    tiles: &HotTilesRef<'_>,
    hoisted: &HoistedCoreChunk<B>,
    crossbar: &CrossbarFrame<B::Vec4>,
    u_lr: (f32, f32),
) -> ((f32, f32), (f32, f32)) {
    let mut quad_in = [B::Stereo8::zero(); MAX_CORE_QUADS];
    let mut pre_conv_acc = B::Stereo8::zero();
    let mut direct_acc = B::Stereo8::zero();

    for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER {
        if q >= hoisted.num_quads {
            continue;
        }
        let reg = &hoisted.readouts[q];
        let mut u_q = couple_ms_input_stereo8(
            quad_in[q],
            u_lr.0,
            u_lr.1,
            reg.in_mid_weight,
            reg.in_side_weight,
        );
        if crossbar.active {
            let op_p_vec_w = crossbar.op_p_vec.mul(reg.op_p_w);
            let op_p_s8 = B::Stereo8::from_lr(op_p_vec_w, op_p_vec_w);
            let car_s8 =
                B::Stereo8::from_lr(crossbar.car_l.mul(reg.ph_w), crossbar.car_r.mul(reg.ph_w));
            u_q = u_q.add(op_p_s8.add(car_s8));
        }

        let out =
            step_core_quad_stereo8::<B::Stereo8>(u_q, states.state_mut(q), &hoisted.params[q]);
        pre_conv_acc = pre_conv_acc
            .fma(out.adaa_odd, reg.pre_conv_odd)
            .fma(out.adaa_even, reg.pre_conv_even)
            .fma(out.lru_state, reg.pre_conv_lru);
        direct_acc = direct_acc
            .fma(out.adaa_odd, reg.direct_odd)
            .fma(out.adaa_even, reg.direct_even)
            .fma(out.lru_state, reg.direct_lru);

        for &tile_raw in &SONA_SRC_OUTGOING_TILES[q] {
            if tile_raw == TILE_EMPTY {
                break;
            }
            let t_idx = usize::from(tile_raw);
            if t_idx >= hoisted.num_tiles {
                continue;
            }
            let dq = SONA_INTERBLOCK_TILES[t_idx].1;
            let tile = tiles.tile_hot(t_idx);
            let acc = B::Stereo8::fma_mat4x4_shared(quad_in[dq], &tile.w_odd_4x4, out.adaa_odd);
            let acc = fma_fp16_tile::<B>(acc, &tile.w_even_4x4_fp16, out.adaa_even);
            quad_in[dq] = fma_fp16_tile::<B>(acc, &tile.w_lru_4x4_fp16, out.lru_state);
        }
    }

    let (pl, pr) = pre_conv_acc.into_lr();
    let (dl, dr) = direct_acc.into_lr();
    (pl.horizontal_sum_pair(pr), dl.horizontal_sum_pair(dr))
}

/// Evaluates all active Matryoshka `CoreQuad`s in causal block-major order (`0..num_quads`),
/// accumulating inter-block `4x4` tiles (`odd -> even -> lru`) and `VarPro` readouts into
/// `((pre_conv_l, r), (dir_l, r))`.
#[inline(always)]
pub fn evaluate_matryoshka_core_quads<B: GraphBackend>(
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
    evaluate_matryoshka_core_quads_disjoint(&mut states, &tiles, hoisted, crossbar, u_lr)
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
        evaluate_matryoshka_core_quads_disjoint(&mut states, &tiles, hoisted, &crossbar, u_lr);

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
    use crate::graph::slab::layout::SONA_CORE_QUAD_BLOCK_AND_LOCAL;

    /// verify[dsp.head.lower-triangular]
    /// verify[dsp.op-n.core.interblock-tiles]
    #[test]
    fn test_outgoing_tile_table_partition_exact() {
        let mut seen = [false; MAX_SONA_TILES];
        for (src_q, row) in SONA_SRC_OUTGOING_TILES.iter().enumerate() {
            for &t in row {
                if t != TILE_EMPTY {
                    let ti = usize::from(t);
                    assert!(!seen[ti]);
                    seen[ti] = true;
                    let (sq, dq) = SONA_INTERBLOCK_TILES[ti];
                    assert_eq!(sq, src_q);
                    assert!(
                        SONA_CORE_QUAD_BLOCK_AND_LOCAL[sq].0 < SONA_CORE_QUAD_BLOCK_AND_LOCAL[dq].0
                    );
                }
            }
        }
        assert!(seen.iter().all(|&v| v));
    }
}
