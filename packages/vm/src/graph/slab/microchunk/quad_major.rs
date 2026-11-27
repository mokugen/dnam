//! Quad-major `4`-frame / `8`-subsample register-resident microchunk execution
//! (`#![no_std]`, `#![forbid(unsafe_code)]`, `0-ULP` parity vs 4× `step_frame_sona`).

#![forbid(unsafe_code)]

use dnam_kernels::{
    CarrierBasisWeightsVec4, StereoAnalyticQuadrature, StereoCarrierRotorVec4, StereoVec8,
    couple_ms_input_stereo8, eval_carrier_3basis_stereo_vec4, os2x_midpoint_f32x4,
    step_core_quad_phs_stereo8, step_core_quad_stereo8,
};

use crate::control::bus::laws::eval_crossbar_wet_scale;
use crate::graph::GraphBackend;
use crate::graph::transport::{QuadTopRead, TRANSPORT_MAX_QUADS};
use crate::math::{SimdVec4, flush_recursive_state_f32, fp16_to_f32};
use crate::types::VmError;

use super::super::layout::{
    CrossbarFrame, FLAG_CROSSBAR_ACTIVE, MAX_CORE_QUADS, SONA_INTERBLOCK_TILES,
    SONA_ULTRA_QUAD_TRAVERSAL_ORDER, SonaHotSlab,
};
use super::core_step::{
    SONA_SRC_OUTGOING_TILES, TILE_EMPTY, active_quad_and_tile_counts, decode_core_quad_params,
    decode_core_quad_readout, step_op_p_1x,
};
use super::telemetry::{BlockTelemetry, finalize_frame_output_with_telem};
use super::{eval_op_t_crossbar_taps, step_transport_end_via_slab_with_telem};

#[derive(Clone, Copy)]
struct CrossbarSubParams<V: dnam_kernels::KernelVec4> {
    active: bool,
    cos_st: V,
    sin_st: V,
    carrier_w: CarrierBasisWeightsVec4<V>,
    wet_scale: f32,
    op_p_gain: f32,
}

pub(super) struct PreparedMicrochunkInputs<V: SimdVec4> {
    pub num_sub: usize,
    pub u_sub: [(f32, f32); 8],
    pub crossbar_sub: [CrossbarFrame<V>; 8],
}

struct CoreChunkBuffers<S> {
    quad_in: [[S; MAX_CORE_QUADS]; 8],
    pre_conv_acc: [S; 8],
    direct_acc: [S; 8],
}

#[inline(always)]
fn build_crossbar_sub<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    cb_params: &CrossbarSubParams<B::Vec4>,
    iq_sub: [f32; 4],
    feat_sub: [f32; 4],
) -> CrossbarFrame<B::Vec4> {
    if cb_params.active {
        let (cos_l, sin_l) = slab.lite.phasors.step_quadrature_vec4::<B::Vec4>();
        let rotor = StereoCarrierRotorVec4::from_left_and_offset(
            cos_l,
            sin_l,
            cb_params.cos_st,
            cb_params.sin_st,
        );
        let quad = StereoAnalyticQuadrature {
            i_l: iq_sub[0],
            q_l: iq_sub[1],
            i_r: iq_sub[2],
            q_r: iq_sub[3],
        };
        let (car_l, car_r) = eval_carrier_3basis_stereo_vec4(
            &quad,
            &rotor,
            &cb_params.carrier_w,
            cb_params.wet_scale,
        );
        CrossbarFrame {
            active: true,
            op_p_vec: B::Vec4::load(&feat_sub).mul(B::Vec4::splat(cb_params.op_p_gain)),
            car_l,
            car_r,
        }
    } else {
        slab.lite.phasors.advance();
        CrossbarFrame {
            active: false,
            op_p_vec: B::Vec4::zero(),
            car_l: B::Vec4::zero(),
            car_r: B::Vec4::zero(),
        }
    }
}

/// Executes 4 frames of `Op-P 1x`, `Op-T` crossbar taps, `Op-N[Pre]`, optional `OS2X` upsample,
/// and `Q0.32` carrier projection (`S = 4` or `8`), updating `tel.last_op_p_*` and `tel.last_in`.
#[inline(always)]
pub(super) fn prepare_microchunk_crossbar_and_u_sub<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    reads_4: &[[QuadTopRead; TRANSPORT_MAX_QUADS]; 4],
    u_sense_lr: &[[f32; 2]; 4],
    tel: &mut BlockTelemetry,
) -> PreparedMicrochunkInputs<B::Vec4> {
    let os2x = slab.os2x();
    let crossbar_active = (slab.lite.flags & FLAG_CROSSBAR_ACTIVE) != 0;
    let wet_scale = eval_crossbar_wet_scale(slab.cold().latched_stage_params[15]);
    let op_p_gain = slab.cold().base_op_p_gain * wet_scale;
    let op_t_gain = slab.cold().base_op_t_gain * wet_scale;
    let (d_scale, delta_step) = (
        slab.cold().op_n_pre_hoisted.d_scale,
        slab.cold().op_n_pre_hoisted.delta_step,
    );

    let (cos_st, sin_st, carrier_w) = if crossbar_active {
        (
            B::fp16x4_to_vec4(&slab.lite.phasor_st_cos_fp16),
            B::fp16x4_to_vec4(&slab.lite.phasor_st_sin_fp16),
            CarrierBasisWeightsVec4 {
                w0: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[0]),
                w1: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[1]),
                w2: B::fp16x4_to_vec4(&slab.lite.carrier_basis_w_fp16[2]),
            },
        )
    } else {
        let z = B::Vec4::zero();
        (
            z,
            z,
            CarrierBasisWeightsVec4 {
                w0: z,
                w1: z,
                w2: z,
            },
        )
    };
    let cb_params = CrossbarSubParams {
        active: crossbar_active,
        cos_st,
        sin_st,
        carrier_w,
        wet_scale,
        op_p_gain,
    };

    let os_params = slab.os2x_hot.os_params;
    let mut prev_iq_lr = slab.os2x_hot.op_p_prev_iq_lr;
    let mut prev_feat = slab.os2x_hot.op_p_prev_feat;

    let mut u_sub = [(0.0_f32, 0.0_f32); 8];
    let empty_cb = CrossbarFrame {
        active: false,
        op_p_vec: B::Vec4::zero(),
        car_l: B::Vec4::zero(),
        car_r: B::Vec4::zero(),
    };
    let mut crossbar_sub = [empty_cb; 8];

    for f in 0..4 {
        let op_p_1x = step_op_p_1x(slab, u_sense_lr[f][0], u_sense_lr[f][1]);
        let (t_cb_l, t_cb_r) = eval_op_t_crossbar_taps(slab, &reads_4[f], op_t_gain);
        let (s0_l, s0_r) = (u_sense_lr[f][0] + t_cb_l, u_sense_lr[f][1] + t_cb_r);
        let (ux_l, ux_r) = slab
            .lite
            .op_n_pre
            .step_stereo(s0_l, s0_r, d_scale, delta_step);
        if f == 3 {
            tel.last_op_p_iq_lr = op_p_1x.iq_lr;
            tel.last_op_p_feat = op_p_1x.feat;
            tel.last_in = [ux_l, ux_r];
        }
        if os2x {
            let (u0, u1) = slab
                .os2x_hot
                .os_stereo
                .upsample_stereo::<B>(ux_l, ux_r, &os_params);
            let s0 = 2 * f;
            u_sub[s0] = (u0[0], u0[1]);
            u_sub[s0 + 1] = (u1[0], u1[1]);
            let iq0 = os2x_midpoint_f32x4(prev_iq_lr, op_p_1x.iq_lr);
            let feat0 = os2x_midpoint_f32x4(prev_feat, op_p_1x.feat);
            crossbar_sub[s0] = build_crossbar_sub::<B>(slab, &cb_params, iq0, feat0);
            prev_iq_lr = op_p_1x.iq_lr;
            prev_feat = op_p_1x.feat;
            crossbar_sub[s0 + 1] =
                build_crossbar_sub::<B>(slab, &cb_params, op_p_1x.iq_lr, op_p_1x.feat);
        } else {
            u_sub[f] = (ux_l, ux_r);
            crossbar_sub[f] =
                build_crossbar_sub::<B>(slab, &cb_params, op_p_1x.iq_lr, op_p_1x.feat);
        }
    }

    if os2x {
        slab.os2x_hot.op_p_prev_iq_lr = prev_iq_lr;
        slab.os2x_hot.op_p_prev_feat = prev_feat;
    }

    PreparedMicrochunkInputs {
        num_sub: if os2x { 8 } else { 4 },
        u_sub,
        crossbar_sub,
    }
}

#[inline(always)]
fn eval_core_quad_chunk_pass<B: GraphBackend, const BLOCK0_PHS: bool>(
    q: usize,
    num_tiles: usize,
    slab: &mut SonaHotSlab<'_, B>,
    inputs: &PreparedMicrochunkInputs<B::Vec4>,
    bufs: &mut CoreChunkBuffers<B::Stereo8>,
) {
    let params_q = decode_core_quad_params(slab, q);
    let reg_q = decode_core_quad_readout(slab, q);
    let mut state_q = *slab.state_mut(q);
    let num_sub = inputs.num_sub;

    let mut q_out_odd = [B::Stereo8::zero(); 8];
    let mut q_out_even = [B::Stereo8::zero(); 8];
    let mut q_out_lru = [B::Stereo8::zero(); 8];

    for s in 0..num_sub {
        let mut u_q = couple_ms_input_stereo8(
            if BLOCK0_PHS {
                B::Stereo8::zero()
            } else {
                bufs.quad_in[s][q]
            },
            inputs.u_sub[s].0,
            inputs.u_sub[s].1,
            reg_q.in_mid_weight,
            reg_q.in_side_weight,
        );
        let cb = &inputs.crossbar_sub[s];
        if cb.active {
            let op_p_vec_w = cb.op_p_vec.mul(reg_q.op_p_w);
            let op_p_s8 = B::Stereo8::from_lr(op_p_vec_w, op_p_vec_w);
            let car_s8 = B::Stereo8::from_lr(cb.car_l.mul(reg_q.ph_w), cb.car_r.mul(reg_q.ph_w));
            u_q = u_q.add(op_p_s8.add(car_s8));
        }

        let out = if BLOCK0_PHS {
            step_core_quad_phs_stereo8::<B::Stereo8>(u_q, &mut state_q, &params_q)
        } else {
            step_core_quad_stereo8::<B::Stereo8>(u_q, &mut state_q, &params_q)
        };
        bufs.pre_conv_acc[s] = bufs.pre_conv_acc[s]
            .fma(out.adaa_odd, reg_q.pre_conv_odd)
            .fma(out.adaa_even, reg_q.pre_conv_even)
            .fma(out.lru_state, reg_q.pre_conv_lru);
        bufs.direct_acc[s] = bufs.direct_acc[s]
            .fma(out.adaa_odd, reg_q.direct_odd)
            .fma(out.adaa_even, reg_q.direct_even)
            .fma(out.lru_state, reg_q.direct_lru);
        q_out_odd[s] = out.adaa_odd;
        q_out_even[s] = out.adaa_even;
        q_out_lru[s] = out.lru_state;
    }

    *slab.state_mut(q) = state_q;

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
        let w_even_cols = [
            B::fp16x4_to_vec4(&tile.w_even_4x4_fp16[0]),
            B::fp16x4_to_vec4(&tile.w_even_4x4_fp16[1]),
            B::fp16x4_to_vec4(&tile.w_even_4x4_fp16[2]),
            B::fp16x4_to_vec4(&tile.w_even_4x4_fp16[3]),
        ];
        let w_lru_cols = [
            B::fp16x4_to_vec4(&tile.w_lru_4x4_fp16[0]),
            B::fp16x4_to_vec4(&tile.w_lru_4x4_fp16[1]),
            B::fp16x4_to_vec4(&tile.w_lru_4x4_fp16[2]),
            B::fp16x4_to_vec4(&tile.w_lru_4x4_fp16[3]),
        ];
        for s in 0..num_sub {
            let tile_acc =
                B::Stereo8::fma_mat4x4_shared(bufs.quad_in[s][dq], &tile.w_odd_4x4, q_out_odd[s]);
            let (t_l, t_r) = tile_acc.into_lr();
            let (ev_l, ev_r) = q_out_even[s].into_lr();
            let (t_l, t_r) = B::fma_cols4_stereo(t_l, t_r, w_even_cols, ev_l, ev_r);
            let (lr_l, lr_r) = q_out_lru[s].into_lr();
            let (t_l, t_r) = B::fma_cols4_stereo(t_l, t_r, w_lru_cols, lr_l, lr_r);
            bufs.quad_in[s][dq] = B::Stereo8::from_lr(t_l, t_r);
        }
    }
}

/// Evaluates all active `CoreQuad`s in quad-major order (`q` outer in topological Matryoshka order,
/// `s in 0..num_sub` inner in registers), applies `hp_lp` + `ro_gains` and optional `OS2X`
/// downsampling, and updates `tel.last_s2_dir`.
#[inline(always)]
pub(super) fn eval_core_quads_quad_major<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    inputs: &PreparedMicrochunkInputs<B::Vec4>,
    tel: &mut BlockTelemetry,
) -> ([[f32; 2]; 4], [[f32; 2]; 4]) {
    let (num_quads, num_tiles) = active_quad_and_tile_counts(slab.lite.tier);
    let mut bufs = CoreChunkBuffers {
        quad_in: [[B::Stereo8::zero(); MAX_CORE_QUADS]; 8],
        pre_conv_acc: [B::Stereo8::zero(); 8],
        direct_acc: [B::Stereo8::zero(); 8],
    };

    for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER[..4] {
        if q < num_quads {
            eval_core_quad_chunk_pass::<B, true>(q, num_tiles, slab, inputs, &mut bufs);
        }
    }
    for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER[4..] {
        if q < num_quads {
            eval_core_quad_chunk_pass::<B, false>(q, num_tiles, slab, inputs, &mut bufs);
        }
    }

    let num_sub = inputs.num_sub;
    let pole = slab.lite.hp_lp_pole;
    let one_m_p = 1.0 - pole;
    let [mut lp_l, mut lp_r] = slab.lite.hp_lp_state;
    let rg = slab.lite.ro_gains_fp16.map(fp16_to_f32);
    let mut sub_out = [[0.0_f32; 4]; 8];
    for s in 0..num_sub {
        let (pl, pr) = bufs.pre_conv_acc[s].into_lr();
        let (dl, dr) = bufs.direct_acc[s].into_lr();
        let (pre_l, pre_r) = pl.horizontal_sum_pair(pr);
        let (dir_l, dir_r) = dl.horizontal_sum_pair(dr);
        lp_l = flush_recursive_state_f32(pole * lp_l + one_m_p * inputs.u_sub[s].0);
        lp_r = flush_recursive_state_f32(pole * lp_r + one_m_p * inputs.u_sub[s].1);
        let (hp_l, hp_r) = (inputs.u_sub[s].0 - lp_l, inputs.u_sub[s].1 - lp_r);
        sub_out[s] = [
            pre_l + rg[0] * inputs.u_sub[s].0 + rg[2] * hp_l,
            pre_r + rg[1] * inputs.u_sub[s].1 + rg[3] * hp_r,
            dir_l + rg[4] * inputs.u_sub[s].0 + rg[6] * hp_l,
            dir_r + rg[5] * inputs.u_sub[s].1 + rg[7] * hp_r,
        ];
    }
    slab.lite.hp_lp_state = [lp_l, lp_r];

    let mut pre_lr = [[0.0_f32; 2]; 4];
    let mut dir_lr = [[0.0_f32; 2]; 4];
    if num_sub == 8 {
        let os_params = slab.os2x_hot.os_params;
        for f in 0..4 {
            let d = slab.os2x_hot.os_stereo.downsample_4ch::<B>(
                sub_out[2 * f],
                sub_out[2 * f + 1],
                &os_params,
            );
            pre_lr[f] = [d[0], d[1]];
            dir_lr[f] = [d[2], d[3]];
        }
    } else {
        for f in 0..4 {
            pre_lr[f] = [sub_out[f][0], sub_out[f][1]];
            dir_lr[f] = [sub_out[f][2], sub_out[f][3]];
        }
    }
    tel.last_s2_dir = [pre_lr[3][0], pre_lr[3][1], dir_lr[3][0], dir_lr[3][1]];
    (pre_lr, dir_lr)
}

/// Runs 4 frames of `Op-T` end-of-frame writes and `finalize_frame_output_with_telem`, advancing
/// `sample_tick` once by `4`.
#[inline(always)]
pub(super) fn finish_transport_and_output_chunk4<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    reads_4: &[[QuadTopRead; TRANSPORT_MAX_QUADS]; 4],
    dir_lr: &[[f32; 2]; 4],
    conv_lr: &[[f32; 2]; 4],
    out_l: &mut [f32],
    out_r: &mut [f32],
    tel: &mut BlockTelemetry,
) -> Result<(), VmError> {
    for f in 0..4 {
        let x_tank = (dir_lr[f][0] + conv_lr[f][0], dir_lr[f][1] + conv_lr[f][1]);
        let end = step_transport_end_via_slab_with_telem(slab, &reads_4[f], x_tank, tel)?;
        let (ol, or) = finalize_frame_output_with_telem(
            end.tank_lr.0,
            end.tank_lr.1,
            end.core_lr,
            end.wet_lr,
            tel,
        );
        out_l[f] = ol;
        out_r[f] = or;
    }
    slab.lite.sample_tick = slab.lite.sample_tick.wrapping_add(4);
    Ok(())
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use crate::bin::{
        Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire, ControlClampDescriptorWire,
        DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT, DNBN_FLAG_WET_POLARITY_INV,
        DnamBinBuildInput, DnamBinView, SONA_META_CORE_MODS_PRESENT, SONA_META_OS_EXT_PRESENT,
        SONA_META_SENSORY_QUAD, encode_dnam_bin_into,
    };
    use crate::graph::GraphBackend;
    use crate::graph::slab::hydrate::hydrate_slab_owned;
    use crate::graph::slab::hydrate::lifecycle::switch_tier;
    use crate::math::{DefaultMathBackend, ScalarBackend, f32_to_fp16};
    use crate::types::Tier;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16, U32};

    fn build_rich_parity_bin(
        os_all_tiers: bool,
        single_input: bool,
        mono_out: bool,
    ) -> ([u8; 32_768], usize) {
        let mut a0 = DnamBinBuildInput::default_arena0_lite();
        let meta_flags =
            SONA_META_SENSORY_QUAD | SONA_META_OS_EXT_PRESENT | SONA_META_CORE_MODS_PRESENT;
        a0.crossbar.sona_meta_flags = U32::new(meta_flags);
        a0.crossbar.cell_desc.cell_flags = U32::new(meta_flags);
        a0.crossbar.cell_desc.tier_flags = U16::new(0x000F);
        a0.crossbar.cell_desc.oversample_factor = U16::new(if os_all_tiers { 2 } else { 1 });
        a0.crossbar.crossbar_op_gains_fp16 =
            [U16::new(f32_to_fp16(0.35)), U16::new(f32_to_fp16(-0.25))];
        a0.crossbar.op_n_pre_base.gamma_bc = F32::new(0.4);
        a0.crossbar.op_n_pre_base.zoh_hold_depth = F32::new(0.3);
        a0.crossbar.op_n_pre_base.beta_q = F32::new(0.8);
        a0.crossbar.phasor_freq_hz = [110.0, 220.0, 330.0, 440.0].map(F32::new);
        a0.crossbar.carrier_basis_w_fp16 = [
            [0.2, -0.1, 0.15, 0.05].map(|v| U16::new(f32_to_fp16(v))),
            [-0.1, 0.25, -0.05, 0.1].map(|v| U16::new(f32_to_fp16(v))),
            [0.05, 0.1, 0.2, -0.15].map(|v| U16::new(f32_to_fp16(v))),
        ];
        a0.crossbar.sense_svf_g = [0.05, 0.2, 0.8, 2.5].map(F32::new);
        a0.crossbar.sense_svf_k_fp16 = [0.6, 0.7, 0.8, 1.1].map(|v| U16::new(f32_to_fp16(v)));
        a0.crossbar.sense_in_l_fp16 = [1.0, 0.5, 0.5, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        a0.crossbar.sense_in_r_fp16 = [0.0, 0.5, -0.5, 1.0].map(|v| U16::new(f32_to_fp16(v)));
        a0.crossbar.sense_tone_lp_fp16 = [U16::new(f32_to_fp16(0.8)); 4];
        a0.crossbar.sense_tone_bp_fp16 = [U16::new(f32_to_fp16(0.3)); 4];
        a0.crossbar.sense_tone_hp_fp16 = [U16::new(f32_to_fp16(-0.2)); 4];
        a0.crossbar.sense_lru_alpha_rel = [0.98, 0.95, 0.91, 0.85].map(F32::new);
        a0.crossbar.sense_lru_alpha_atk = [0.2, 0.35, 0.5, 0.7].map(F32::new);
        a0.crossbar.sense_lru_kappa_fp16 = [U16::new(f32_to_fp16(0.25)); 4];
        a0.crossbar.sense_vca_w_fp16 = [0.4, -0.3, 0.2, 0.1].map(|v| U16::new(f32_to_fp16(v)));

        for q in 0..4 {
            a0.core_quads[q].in_mid_weight_fp16 =
                [0.4, 0.2, -0.1, 0.3].map(|v| U16::new(f32_to_fp16(v)));
            a0.core_quads[q].svf_g = [0.15, 0.3, 0.6, 1.2].map(F32::new);
            a0.core_quads[q].svf_k_fp16 = [U16::new(f32_to_fp16(0.7)); 4];
            a0.core_quads[q].adaa_drive_fp16 = [U16::new(f32_to_fp16(0.5)); 4];
            a0.core_quads[q].lru_alpha_rel = [F32::new(0.95); 4];
            a0.core_quads[q].lru_alpha_atk = [F32::new(0.25); 4];
            a0.core_quads[q].lru_kappa_fp16 = [U16::new(f32_to_fp16(0.2)); 4];
            a0.core_quads[q].tone_lp_fp16 = [U16::new(f32_to_fp16(0.5)); 4];
            a0.core_quads[q].tone_dir_fp16 = [U16::new(f32_to_fp16(0.7)); 4];
            a0.core_quads[q].wavefolder_mix_fp16 = [U16::new(f32_to_fp16(0.15)); 4];
        }
        for t in 0..3 {
            a0.tiles[t].w_odd_4x4 = [[F32::new(0.08); 4]; 4];
            a0.tiles[t].w_even_4x4_fp16 = [[U16::new(f32_to_fp16(0.04)); 4]; 4];
            a0.tiles[t].w_lru_4x4_fp16 = [[U16::new(f32_to_fp16(0.03)); 4]; 4];
        }
        a0.linear_readout_lite.direct_s2_u_gain = [F32::new(0.6), F32::new(0.4)];
        a0.linear_readout_lite.w_ro_fp16 = [U16::new(f32_to_fp16(0.18)); 208];
        for i in 0..32 {
            a0.conv_fir_lite.fir_l[i] = F32::new(((i as f32) + 1.0) * 0.0025);
            a0.conv_fir_lite.fir_r[i] = F32::new(-((i as f32) + 1.0) * 0.002);
        }
        a0.conv_kautz_lite[0].r_low = [F32::new(0.75); 4];
        a0.conv_kautz_lite[0].dr = [F32::new(0.05); 4];
        a0.conv_kautz_lite[0].theta = [F32::new(0.25); 4];
        a0.conv_kautz_lite[0].b_gamma = [F32::new(0.2); 4];
        a0.conv_kautz_lite[0].c_re_l = [F32::new(0.3); 4];
        a0.conv_kautz_lite[0].c_re_r = [F32::new(-0.25); 4];
        a0.transport_lite.delay_samples = [12.0, 17.0, 23.0, 31.0].map(F32::new);
        a0.transport_lite.g_loop = F32::new(0.35);
        a0.transport_lite.input_gains_fp16 = [U16::new(f32_to_fp16(0.25)); 4];
        a0.transport_lite.wet_tap_l_fp16 = [U16::new(f32_to_fp16(0.2)); 4];
        a0.transport_lite.wet_tap_r_fp16 = [U16::new(f32_to_fp16(0.2)); 4];
        a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(0.85));
        a0.control_bus_lite.lfo_dphi_train = [F32::new(0.004), F32::new(0.009)];
        a0.control_bus_lite.w_ctrl[0][0] = F32::new(0.12);
        a0.control_bus_lite.w_ctrl[15][8] = F32::new(0.05);

        let mut a1 = Arena1FullDeltaWire::new_zeroed();
        a1.core_quads[0] = a0.core_quads[0];
        a1.core_quads[1] = a0.core_quads[1];
        a1.transport_full = a0.transport_lite;
        a1.transport_full.delay_samples = [19.0, 29.0, 37.0, 41.0].map(F32::new);

        let mut a2 = Arena2ProDeltaWire::new_zeroed();
        a2.core_quads[0] = a0.core_quads[0];
        a2.core_quads[1] = a0.core_quads[1];
        a2.transport_pro[0] = a0.transport_lite;
        a2.transport_pro[0].delay_samples = [21.0, 33.0, 43.0, 53.0].map(F32::new);

        let mut a3 = Arena3UltraDeltaWire::new_zeroed();
        a3.core_quads[0] = a0.core_quads[0];
        a3.core_quads[1] = a0.core_quads[1];
        a3.transport_ultra[0] = a0.transport_lite;
        a3.transport_ultra[0].delay_samples = [25.0, 35.0, 47.0, 59.0].map(F32::new);

        let ctrl_desc = [ControlClampDescriptorWire::new_zeroed()];
        let mut flags = if mono_out { 0 } else { DNBN_FLAG_STEREO_OUT };
        if single_input {
            flags |= DNBN_FLAG_SINGLE_INPUT;
        }
        flags |= DNBN_FLAG_WET_POLARITY_INV;

        let input = DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: flags,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 128,
            dc_blocker_pole_5hz: crate::math::TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 100.0,
            ro_pole_48k: 0.99,
            switch_xfade_samples: 48,
            content_hash_prefix: [0; 8],
            controls: &ctrl_desc,
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0,
            arena1: Some(&a1),
            arena2: Some(&a2),
            arena3: Some(&a3),
        };
        let mut buf = [0_u8; 32_768];
        let n = encode_dnam_bin_into(&input, &mut buf).unwrap();
        (buf, n)
    }

    fn verify_block_vs_step_frame_zero_ulp<B: GraphBackend>(
        tier: Tier,
        os_all_tiers: bool,
        single_input: bool,
        mono_out: bool,
        trigger_xfade: bool,
    ) {
        let (bin, bin_len) = build_rich_parity_bin(os_all_tiers, single_input, mono_out);
        let view = DnamBinView::parse(&bin[..bin_len]).unwrap();

        let init_tier = if trigger_xfade { Tier::Ultra } else { tier };
        let mut slab_a = hydrate_slab_owned::<B>(&view, init_tier, 48_000.0, false).unwrap();
        let mut slab_b = hydrate_slab_owned::<B>(&view, init_tier, 48_000.0, false).unwrap();

        if trigger_xfade && tier != Tier::Ultra {
            switch_tier(&mut slab_a, tier).unwrap();
            switch_tier(&mut slab_b, tier).unwrap();
            assert!(slab_a.xfade_active());
        }
        assert_eq!(slab_a.os2x(), tier != Tier::Lite && os_all_tiers);

        // 53 frames: exercises multiple 16-frame 3 kHz ticks, 4-frame microchunks, and peel.
        let mut in_l = [0.0_f32; 53];
        let mut in_r = [0.0_f32; 53];
        for i in 0..53 {
            in_l[i] = ((i as f32) * 0.13 - 2.5) * 0.2;
            in_r[i] = ((i as f32) * -0.09 + 1.4) * 0.2;
        }
        let (mut out_a_l, mut out_a_r) = ([0.0_f32; 53], [0.0_f32; 53]);
        let (mut out_b_l, mut out_b_r) = ([0.0_f32; 53], [0.0_f32; 53]);

        slab_a
            .process_block(&in_l, &in_r, &mut out_a_l, &mut out_a_r)
            .unwrap();
        for i in 0..53 {
            let (bl, br) = super::super::step_frame_sona(&mut slab_b, in_l[i], in_r[i]).unwrap();
            out_b_l[i] = bl;
            out_b_r[i] = br;
        }

        for i in 0..53 {
            assert_eq!(out_a_l[i].to_bits(), out_b_l[i].to_bits());
            assert_eq!(out_a_r[i].to_bits(), out_b_r[i].to_bits());
        }
        assert_eq!(
            slab_a.cold().last_in.map(f32::to_bits),
            slab_b.cold().last_in.map(f32::to_bits)
        );
        assert_eq!(
            slab_a.cold().last_s2_dir.map(f32::to_bits),
            slab_b.cold().last_s2_dir.map(f32::to_bits)
        );
        assert_eq!(
            slab_a.cold().last_core_lr.map(f32::to_bits),
            slab_b.cold().last_core_lr.map(f32::to_bits)
        );
        assert_eq!(
            slab_a.cold().last_transport_wet_lr.map(f32::to_bits),
            slab_b.cold().last_transport_wet_lr.map(f32::to_bits)
        );
        assert_eq!(slab_a.cold().xfade_rem, slab_b.cold().xfade_rem);
        assert_eq!(slab_a.xfade_active(), slab_b.xfade_active());
    }

    #[test]
    fn test_quad_major_microchunk_zero_ulp_parity_all_tiers_and_modes() {
        for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            verify_block_vs_step_frame_zero_ulp::<ScalarBackend>(tier, false, false, false, false);
            verify_block_vs_step_frame_zero_ulp::<DefaultMathBackend>(
                tier, false, false, false, false,
            );
            verify_block_vs_step_frame_zero_ulp::<DefaultMathBackend>(
                tier, true, false, false, false,
            );
        }
        // Exercise xfade_active, mono_out, and single_input.
        verify_block_vs_step_frame_zero_ulp::<DefaultMathBackend>(
            Tier::Lite,
            false,
            true,
            true,
            true,
        );
    }
}
