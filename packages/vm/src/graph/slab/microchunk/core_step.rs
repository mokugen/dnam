//! Register-hoisted causal Matryoshka `CoreQuad` evaluation, Top-of-Frame `Op-P` / `Q0.32`
//! carrier projection, and `hp_lp` core substep (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    CarrierBasisWeightsVec4, CoreQuadParamsStereo4, StereoAnalyticQuadrature,
    StereoCarrierRotorVec4, StereoVec8, couple_ms_input_stereo8, eval_carrier_3basis_stereo_vec4,
    os2x_midpoint_f32x4, step_core_quad_phs_stereo8, step_core_quad_stereo8,
};

use crate::graph::GraphBackend;
use crate::graph::oversample::RenderMode;
use crate::math::{SimdVec4, flush_recursive_state_f32, fp16_to_f32};
use crate::types::Tier;

use super::super::layout::{
    CoreQuadReadoutRegs, CrossbarFrame, FLAG_CROSSBAR_ACTIVE, MAX_CORE_QUADS, MAX_SONA_TILES,
    SONA_INTERBLOCK_TILES, SONA_ULTRA_QUAD_TRAVERSAL_ORDER, SonaHotSlab,
};

const MAX_OUTGOING_PER_QUAD: usize = dnam_kernels::MAX_OUTGOING_PER_QUAD_V4;
pub(super) const TILE_EMPTY: u8 = 0xFF;

/// Compile-time outgoing tile index table per source quad `0..10` covering all `37` tiles.
pub(super) const SONA_SRC_OUTGOING_TILES: [[u8; MAX_OUTGOING_PER_QUAD]; MAX_CORE_QUADS] =
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

/// Returns `(active_quads, active_tiles)` for `tier` (`v4.0`: `[5, 13, 24, 37]`).
#[inline(always)]
pub(super) const fn active_quad_and_tile_counts(tier: Tier) -> (usize, usize) {
    let num_quads = tier.simd_groups();
    let num_tiles = match tier {
        Tier::Lite => 5,
        Tier::Full => 13,
        Tier::Pro => 24,
        Tier::Ultra => 37,
    };
    (num_quads, num_tiles)
}

/// Advances `MasterCtrlDelayRingCold` (`delta_D_up = 16`) at each `4`-frame microchunk start.
#[inline(always)]
pub(crate) fn advance_master_ctrl_delay_if_chunk_start<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
) {
    let cold = slab.cold_mut();
    let snap = cold.master_ctrl_snapshot();
    cold.master_os.active_master_core_ctrl = cold.master_os.ctrl_delay_ring.push_and_pop(snap);
}

/// Decodes modulated `CoreQuadParamsStereo4` directly from `CoreQuadHot` + `CoreQuadExtCold` +
/// `cold` modulations (or delayed `Master` `delta_D_up = 16` snapshot) into `f32` SIMD registers.
#[inline(always)]
pub(super) fn decode_core_quad_params<B: GraphBackend>(
    slab: &SonaHotSlab<'_, B>,
    q: usize,
) -> CoreQuadParamsStereo4<B::Vec4> {
    let cold = slab.cold();
    let qh = slab.core_quad_hot(q);
    let ext = slab.core_quad_ext(q);
    let l0 = q * 4;
    let use_delayed = matches!(cold.render_mode, RenderMode::Master) && cold.zoh.m_core == 4;
    let (lane_g, lane_k, lane_wf, adaa_scale, bias_delta) = if use_delayed {
        let act = &cold.master_os.active_master_core_ctrl;
        (
            &act.lane_g_scale,
            &act.lane_k_delta,
            &act.lane_wf_delta,
            act.quad_adaa_scale[q],
            act.quad_lru_bias[q],
        )
    } else {
        (
            &cold.lane_g_scale,
            &cold.lane_k_delta,
            &cold.lane_wf_delta,
            cold.quad_adaa_scale[q],
            cold.quad_lru_bias[q],
        )
    };
    qh.to_kernel_params_modulated::<B>(
        ext,
        crate::graph::slab::CoreQuadModulation {
            g_scale: B::Vec4::load(&[lane_g[l0], lane_g[l0 + 1], lane_g[l0 + 2], lane_g[l0 + 3]]),
            k_delta: B::Vec4::load(&[lane_k[l0], lane_k[l0 + 1], lane_k[l0 + 2], lane_k[l0 + 3]]),
            adaa_scale,
            bias_delta,
            wf_delta: B::Vec4::load(&[
                lane_wf[l0],
                lane_wf[l0 + 1],
                lane_wf[l0 + 2],
                lane_wf[l0 + 3],
            ]),
            rect_mask4: slab.rect_mask4(q),
        },
    )
}

/// Decodes `CoreQuadReadoutRegs` directly from `CoreQuadHot`, `CoreQuadExtCold`,
/// `CoreQuadReadoutHot`, and hot crossbar matrices into `f32` SIMD registers (`0-ULP`).
#[inline(always)]
pub(super) fn decode_core_quad_readout<B: GraphBackend>(
    slab: &SonaHotSlab<'_, B>,
    q: usize,
) -> CoreQuadReadoutRegs<B::Vec4, B::Stereo8> {
    let qh = slab.core_quad_hot(q);
    let ext = slab.core_quad_ext(q);
    slab.readout_hot(q).decode_regs::<B>(
        qh.in_mid_weight_vec4::<B>(),
        ext.in_side_weight_vec4::<B>(),
        B::fp16x4_to_vec4(slab.op_p_to_core_fp16(q)),
        B::fp16x4_to_vec4(slab.phasor_to_core_fp16(q)),
    )
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

struct CoreAccumulators<S> {
    quad_in: [S; MAX_CORE_QUADS],
    pre_conv: S,
    direct: S,
}

#[inline(always)]
fn eval_core_quad_step<B: GraphBackend, const BLOCK0_PHS: bool>(
    q: usize,
    num_tiles: usize,
    slab: &mut SonaHotSlab<'_, B>,
    crossbar: &CrossbarFrame<B::Vec4>,
    u_lr: (f32, f32),
    acc: &mut CoreAccumulators<B::Stereo8>,
) {
    let params_q = decode_core_quad_params(slab, q);
    let reg = decode_core_quad_readout(slab, q);
    let mut u_q = couple_ms_input_stereo8(
        if BLOCK0_PHS {
            B::Stereo8::zero()
        } else {
            acc.quad_in[q]
        },
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

    let mut state_q = *slab.state_mut(q);
    let out = if BLOCK0_PHS {
        step_core_quad_phs_stereo8::<B::Stereo8>(u_q, &mut state_q, &params_q)
    } else {
        step_core_quad_stereo8::<B::Stereo8>(u_q, &mut state_q, &params_q)
    };
    *slab.state_mut(q) = state_q;

    acc.pre_conv = acc
        .pre_conv
        .fma(out.adaa_odd, reg.pre_conv_odd)
        .fma(out.adaa_even, reg.pre_conv_even)
        .fma(out.lru_state, reg.pre_conv_lru);
    acc.direct = acc
        .direct
        .fma(out.adaa_odd, reg.direct_odd)
        .fma(out.adaa_even, reg.direct_even)
        .fma(out.lru_state, reg.direct_lru);

    for &tile_raw in &SONA_SRC_OUTGOING_TILES[q] {
        if tile_raw == TILE_EMPTY {
            break;
        }
        let t_idx = usize::from(tile_raw);
        if t_idx >= num_tiles {
            continue;
        }
        let dq = SONA_INTERBLOCK_TILES[t_idx].1;
        let tile = slab.tile_hot(t_idx);
        let tile_acc =
            B::Stereo8::fma_mat4x4_shared(acc.quad_in[dq], &tile.w_odd_4x4, out.adaa_odd);
        let tile_acc = fma_fp16_tile::<B>(tile_acc, &tile.w_even_4x4_fp16, out.adaa_even);
        acc.quad_in[dq] = fma_fp16_tile::<B>(tile_acc, &tile.w_lru_4x4_fp16, out.lru_state);
    }
}

#[inline(always)]
fn eval_core_quad_step_master<B: GraphBackend, const BLOCK0_PHS: bool>(
    q: usize,
    num_tiles: usize,
    slab: &mut SonaHotSlab<'_, B>,
    crossbar: &CrossbarFrame<B::Vec4>,
    u_lr: (f32, f32),
    acc: &mut CoreAccumulators<B::Stereo8>,
) {
    let params_q = decode_core_quad_params(slab, q);
    let reg = decode_core_quad_readout(slab, q);
    let mut u_q = couple_ms_input_stereo8(
        if BLOCK0_PHS {
            B::Stereo8::zero()
        } else {
            acc.quad_in[q]
        },
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

    let mut state_q = *slab.state_mut(q);
    let out = if BLOCK0_PHS {
        step_core_quad_phs_stereo8::<B::Stereo8>(u_q, &mut state_q, &params_q)
    } else {
        step_core_quad_stereo8::<B::Stereo8>(u_q, &mut state_q, &params_q)
    };
    *slab.state_mut(q) = state_q;

    let lin_v4 = B::fp16x4_to_vec4(&slab.core_quad_ext(q).lin_mix_fp16);
    let lin_mix = B::Stereo8::from_lr(lin_v4, lin_v4);
    let (box_odd, box_even) =
        slab.cold_mut()
            .master_os
            .apply_box2_quad::<B>(q, out.adaa_odd, out.adaa_even, lin_mix);

    acc.pre_conv = acc
        .pre_conv
        .fma(box_odd, reg.pre_conv_odd)
        .fma(box_even, reg.pre_conv_even)
        .fma(out.lru_state, reg.pre_conv_lru);
    acc.direct = acc
        .direct
        .fma(box_odd, reg.direct_odd)
        .fma(box_even, reg.direct_even)
        .fma(out.lru_state, reg.direct_lru);

    for &tile_raw in &SONA_SRC_OUTGOING_TILES[q] {
        if tile_raw == TILE_EMPTY {
            break;
        }
        let t_idx = usize::from(tile_raw);
        if t_idx >= num_tiles {
            continue;
        }
        let dq = SONA_INTERBLOCK_TILES[t_idx].1;
        let tile = slab.tile_hot(t_idx);
        let tile_acc = B::Stereo8::fma_mat4x4_shared(acc.quad_in[dq], &tile.w_odd_4x4, box_odd);
        let tile_acc = fma_fp16_tile::<B>(tile_acc, &tile.w_even_4x4_fp16, box_even);
        acc.quad_in[dq] = fma_fp16_tile::<B>(tile_acc, &tile.w_lru_4x4_fp16, out.lru_state);
    }
}

/// Evaluates all active Matryoshka `CoreQuad`s in causal block-major order (`0..num_quads`),
/// accumulating inter-block `4x4` tiles (`odd -> even -> lru`) and `VarPro` readouts into
/// `((pre_conv_l, r), (dir_l, r))`.
#[inline(always)]
pub fn evaluate_matryoshka_core_quads<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    crossbar: &CrossbarFrame<B::Vec4>,
    u_lr: (f32, f32),
) -> ((f32, f32), (f32, f32)) {
    let (num_quads, num_tiles) = active_quad_and_tile_counts(slab.lite.tier);
    let mut acc = CoreAccumulators {
        quad_in: [B::Stereo8::zero(); MAX_CORE_QUADS],
        pre_conv: B::Stereo8::zero(),
        direct: B::Stereo8::zero(),
    };

    for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER[..3] {
        if q < num_quads {
            eval_core_quad_step::<B, true>(q, num_tiles, slab, crossbar, u_lr, &mut acc);
        }
    }
    for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER[3..] {
        if q < num_quads {
            eval_core_quad_step::<B, false>(q, num_tiles, slab, crossbar, u_lr, &mut acc);
        }
    }

    let (pl, pr) = acc.pre_conv.into_lr();
    let (dl, dr) = acc.direct.into_lr();
    (pl.horizontal_sum_pair(pr), dl.horizontal_sum_pair(dr))
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
    m_core: usize,
    sub_idx: usize,
    u_lr: (f32, f32),
) -> [f32; 4] {
    let crossbar = if (slab.lite.flags & FLAG_CROSSBAR_ACTIVE) != 0 {
        let (iq_sub, feat_sub) = match m_core {
            4 => {
                let prev_iq = slab.lite.op_p_prev_iq_lr;
                let prev_feat = slab.lite.op_p_prev_feat;
                match sub_idx {
                    0 => {
                        let iq =
                            core::array::from_fn(|k| 0.75 * prev_iq[k] + 0.25 * op_p_1x.iq_lr[k]);
                        let ft =
                            core::array::from_fn(|k| 0.75 * prev_feat[k] + 0.25 * op_p_1x.feat[k]);
                        (iq, ft)
                    }
                    1 => (
                        os2x_midpoint_f32x4(prev_iq, op_p_1x.iq_lr),
                        os2x_midpoint_f32x4(prev_feat, op_p_1x.feat),
                    ),
                    2 => {
                        let iq =
                            core::array::from_fn(|k| 0.25 * prev_iq[k] + 0.75 * op_p_1x.iq_lr[k]);
                        let ft =
                            core::array::from_fn(|k| 0.25 * prev_feat[k] + 0.75 * op_p_1x.feat[k]);
                        (iq, ft)
                    }
                    _ => {
                        slab.lite.op_p_prev_iq_lr = op_p_1x.iq_lr;
                        slab.lite.op_p_prev_feat = op_p_1x.feat;
                        (op_p_1x.iq_lr, op_p_1x.feat)
                    }
                }
            }
            2 => {
                if sub_idx == 0 {
                    (
                        os2x_midpoint_f32x4(slab.lite.op_p_prev_iq_lr, op_p_1x.iq_lr),
                        os2x_midpoint_f32x4(slab.lite.op_p_prev_feat, op_p_1x.feat),
                    )
                } else {
                    slab.lite.op_p_prev_iq_lr = op_p_1x.iq_lr;
                    slab.lite.op_p_prev_feat = op_p_1x.feat;
                    (op_p_1x.iq_lr, op_p_1x.feat)
                }
            }
            _ => (op_p_1x.iq_lr, op_p_1x.feat),
        };
        let cold = slab.cold();
        let use_delayed = matches!(cold.render_mode, RenderMode::Master) && cold.zoh.m_core == 4;
        let stage_15 = if use_delayed {
            cold.master_os.active_master_core_ctrl.latched_stage_params[15]
        } else {
            cold.latched_stage_params[15]
        };
        let wet_scale = crate::control::bus::laws::eval_crossbar_wet_scale(stage_15);
        let op_p_gain = slab.cold().base_op_p_gain * wet_scale;
        let cos_st = B::fp16x4_to_vec4(&slab.lite.phasor_st_cos_fp16);
        let sin_st = B::fp16x4_to_vec4(&slab.lite.phasor_st_sin_fp16);
        let carrier_w = CarrierBasisWeightsVec4 {
            w0: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[0]),
            w1: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[1]),
            w2: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[2]),
        };
        let (cos_l, sin_l) = slab.lite.phasors.step_quadrature_vec4::<B::Vec4>();
        let rotor = StereoCarrierRotorVec4::from_left_and_offset(cos_l, sin_l, cos_st, sin_st);
        let quad = StereoAnalyticQuadrature {
            i_l: iq_sub[0],
            q_l: iq_sub[1],
            i_r: iq_sub[2],
            q_r: iq_sub[3],
        };
        let (car_l, car_r) = eval_carrier_3basis_stereo_vec4(&quad, &rotor, &carrier_w, wet_scale);
        CrossbarFrame {
            active: true,
            op_p_vec: B::Vec4::load(&feat_sub).mul(B::Vec4::splat(op_p_gain)),
            car_l,
            car_r,
        }
    } else {
        if (m_core == 2 && sub_idx == 1) || (m_core == 4 && sub_idx == 3) {
            slab.lite.op_p_prev_iq_lr = op_p_1x.iq_lr;
            slab.lite.op_p_prev_feat = op_p_1x.feat;
        }
        slab.lite.phasors.advance();
        CrossbarFrame {
            active: false,
            op_p_vec: B::Vec4::zero(),
            car_l: B::Vec4::zero(),
            car_r: B::Vec4::zero(),
        }
    };

    let master_4x = matches!(slab.cold().render_mode, RenderMode::Master) && m_core == 4;
    let (pre_l, pre_r, dir_l, dir_r) = if master_4x {
        let (num_quads, num_tiles) = active_quad_and_tile_counts(slab.lite.tier);
        let mut acc = CoreAccumulators {
            quad_in: [B::Stereo8::zero(); MAX_CORE_QUADS],
            pre_conv: B::Stereo8::zero(),
            direct: B::Stereo8::zero(),
        };
        for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER[..3] {
            if q < num_quads {
                eval_core_quad_step_master::<B, true>(
                    q, num_tiles, slab, &crossbar, u_lr, &mut acc,
                );
            }
        }
        for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER[3..] {
            if q < num_quads {
                eval_core_quad_step_master::<B, false>(
                    q, num_tiles, slab, &crossbar, u_lr, &mut acc,
                );
            }
        }
        let (pl, pr) = acc.pre_conv.into_lr();
        let (dl, dr) = acc.direct.into_lr();
        let (p_l, p_r) = pl.horizontal_sum_pair(pr);
        let (d_l, d_r) = dl.horizontal_sum_pair(dr);
        (p_l, p_r, d_l, d_r)
    } else {
        let ((p_l, p_r), (d_l, d_r)) = evaluate_matryoshka_core_quads(slab, &crossbar, u_lr);
        (p_l, p_r, d_l, d_r)
    };

    let pole = slab.lite.hp_lp_pole;
    let one_m_p = 1.0 - pole;
    let lp_l = flush_recursive_state_f32(pole * slab.lite.hp_lp_state[0] + one_m_p * u_lr.0);
    let lp_r = flush_recursive_state_f32(pole * slab.lite.hp_lp_state[1] + one_m_p * u_lr.1);
    slab.lite.hp_lp_state = [lp_l, lp_r];
    let (hp_l, hp_r) = (u_lr.0 - lp_l, u_lr.1 - lp_r);
    let rg = slab.lite.ro_gains_fp16.map(fp16_to_f32);
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
