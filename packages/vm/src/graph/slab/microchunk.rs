//! Causal `4`-frame micro-chunking, `1`-frame peel, `2x2` pure stereo / `4x` bounce frame
//! stepping, and `0-ULP` chunk-split invariant SONA block execution (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

pub(super) mod convolver_chunk;
pub mod core_step;
mod inline_hot;
pub(super) mod quad_major;
pub(super) mod telemetry;

use dnam_kernels::scrub_and_flush_f32_bits;

use crate::control::bus::laws::eval_crossbar_wet_scale;
use crate::graph::GraphBackend;
use crate::graph::transport::{QuadTopRead, TRANSPORT_MAX_QUADS, resolve_quad_ring};
use crate::math::{fp16_to_f32, saturation_flags};
use crate::types::VmError;

use super::layout::SonaHotSlab;
use convolver_chunk::step_convolver_frame;
use core_step::{step_core_subsample, step_op_p_1x};
use telemetry::{BlockTelemetry, finalize_frame_output_with_telem};

pub use core_step::evaluate_matryoshka_core_quads;
#[cfg(target_arch = "x86_64")]
pub(crate) use inline_hot::{
    process_block_sona_inline, process_tick16_const_inline, step_frame_sona_inline,
};

#[inline(always)]
fn eval_op_t_crossbar_taps<B: GraphBackend>(
    slab: &SonaHotSlab<'_, B>,
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    op_t_gain: f32,
) -> (f32, f32) {
    if op_t_gain == 0.0 {
        return (0.0, 0.0);
    }
    let active_lines = slab.cold().active_transport_quads.min(TRANSPORT_MAX_QUADS) * 4;
    let mut acc_l = 0.0_f32;
    let mut acc_r = 0.0_f32;
    for idx in 0..active_lines {
        let v = reads[idx >> 2].v_read[idx & 3];
        let [wl, wr] = slab.opt_to_lr_fp16(idx);
        acc_l += v * fp16_to_f32(wl);
        acc_r += v * fp16_to_f32(wr);
    }
    (op_t_gain * acc_l, op_t_gain * acc_r)
}

#[inline(always)]
fn step_transport_top_via_slab<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
) -> Result<[QuadTopRead; TRANSPORT_MAX_QUADS], VmError> {
    let cold = &mut slab.cold.as_mut_slice()[0];
    let active_q = cold.active_transport_quads.min(TRANSPORT_MAX_QUADS);
    let quad_refs = [
        &mut slab.lite.transport_quad0,
        &mut slab.full_delta.transport_quad1,
        &mut slab.pro_delta.transport_quad2,
        &mut slab.ultra_delta.transport_quad3,
    ];
    let mut reads = [QuadTopRead::default(); TRANSPORT_MAX_QUADS];
    for q in 0..active_q {
        let ring = resolve_quad_ring(
            q,
            quad_refs[q].flags,
            &mut slab.lite.transport0_ring,
            &mut cold.transport_ext,
        )?;
        reads[q] = quad_refs[q].read_top_of_frame_v_read::<B>(
            ring,
            cold.transport_schroeder.get_mut(q),
            &cold.transport_quad_eff[q],
        );
    }
    Ok(reads)
}

#[inline(always)]
fn step_transport_top_chunk4_via_slab<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
) -> Result<[[QuadTopRead; TRANSPORT_MAX_QUADS]; 4], VmError> {
    let cold = &mut slab.cold.as_mut_slice()[0];
    let active_q = cold.active_transport_quads.min(TRANSPORT_MAX_QUADS);
    let quad_refs = [
        &mut slab.lite.transport_quad0,
        &mut slab.full_delta.transport_quad1,
        &mut slab.pro_delta.transport_quad2,
        &mut slab.ultra_delta.transport_quad3,
    ];
    let mut reads_4 = [[QuadTopRead::default(); TRANSPORT_MAX_QUADS]; 4];
    for q in 0..active_q {
        let ring = resolve_quad_ring(
            q,
            quad_refs[q].flags,
            &mut slab.lite.transport0_ring,
            &mut cold.transport_ext,
        )?;
        let q_reads = quad_refs[q].read_top_of_frame_chunk4::<B>(
            ring,
            cold.transport_schroeder.get_mut(q),
            &cold.transport_quad_eff[q],
        );
        for f in 0..4 {
            reads_4[f][q] = q_reads[f];
        }
    }
    Ok(reads_4)
}

pub(super) struct TransportEndFrameOut {
    pub tank_lr: (f32, f32),
    pub core_lr: [f32; 2],
    pub wet_lr: [f32; 2],
}

#[inline(always)]
fn step_transport_end_via_slab_with_telem<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    x_tank_lr: (f32, f32),
    tel: &mut BlockTelemetry,
) -> Result<TransportEndFrameOut, VmError> {
    let cold = &mut slab.cold.as_mut_slice()[0];
    let active_q = cold.active_transport_quads.min(TRANSPORT_MAX_QUADS);
    let quad_refs = [
        &mut slab.lite.transport_quad0,
        &mut slab.full_delta.transport_quad1,
        &mut slab.pro_delta.transport_quad2,
        &mut slab.ultra_delta.transport_quad3,
    ];
    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    let mut sat = false;
    for q in 0..active_q {
        let eff = &cold.transport_quad_eff[q];
        let (yl, yr) = quad_refs[q].finish_post_convolver_output::<B>(eff, reads[q].v_read);
        sum_l += yl;
        sum_r += yr;
        let ring = resolve_quad_ring(
            q,
            quad_refs[q].flags,
            &mut slab.lite.transport0_ring,
            &mut cold.transport_ext,
        )?;
        sat |= quad_refs[q].step_end_of_frame_write::<B>(ring, eff, reads[q], x_tank_lr);
    }
    let wet_scale = if active_q > 0 {
        cold.transport_quad_eff[0].wet_scale
    } else {
        1.0
    };
    if sat {
        tel.sat_flags |= saturation_flags::FP16_SATURATED;
    }
    let stage_dry_mix_eff = fp16_to_f32(slab.lite.tank_dry_mix_eff_fp16);
    let (clean_xl, clean_xr) = (
        scrub_and_flush_f32_bits(x_tank_lr.0),
        scrub_and_flush_f32_bits(x_tank_lr.1),
    );
    let core_l = stage_dry_mix_eff * clean_xl;
    let core_r = stage_dry_mix_eff * clean_xr;
    let wet_l = wet_scale * sum_l;
    let wet_r = wet_scale * sum_r;
    Ok(TransportEndFrameOut {
        tank_lr: (core_l + wet_l, core_r + wet_r),
        core_lr: [core_l, core_r],
        wet_lr: [wet_l, wet_r],
    })
}

#[inline(always)]
fn step_post_sensory_with_reads<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    u_sense_l: f32,
    u_sense_r: f32,
    tel: &mut BlockTelemetry,
) -> Result<(f32, f32), VmError> {
    let op_p_1x = step_op_p_1x(slab, u_sense_l, u_sense_r);
    tel.last_op_p_iq_lr = op_p_1x.iq_lr;
    tel.last_op_p_feat = op_p_1x.feat;
    let wet_scale = eval_crossbar_wet_scale(slab.cold().latched_stage_params[15]);
    let op_t_gain = slab.cold().base_op_t_gain * wet_scale;
    let (t_cb_l, t_cb_r) = eval_op_t_crossbar_taps(slab, reads, op_t_gain);
    let (s0_l, s0_r) = (u_sense_l + t_cb_l, u_sense_r + t_cb_r);

    let (d_scale, delta_step) = (
        slab.cold().op_n_pre_hoisted.d_scale,
        slab.cold().op_n_pre_hoisted.delta_step,
    );
    let (ux_l, ux_r) = slab
        .lite
        .op_n_pre
        .step_stereo(s0_l, s0_r, d_scale, delta_step);
    tel.last_in = [ux_l, ux_r];

    let (m_core, is_master) = {
        let cold = slab.cold();
        (
            cold.zoh.m_core,
            matches!(
                cold.render_mode,
                crate::graph::oversample::RenderMode::Master
            ),
        )
    };
    if is_master && m_core == 4 && (slab.lite.sample_tick & 3) == 0 {
        core_step::advance_master_ctrl_delay_if_chunk_start(slab);
    }
    let [pre_l, pre_r, dir_l, dir_r] = match (is_master, m_core) {
        (false, 2) => {
            let up = slab.lite.live_os.upsample_stereo(ux_l, ux_r);
            let s0 = step_core_subsample(slab, &op_p_1x, 2, 0, (up[0][0], up[0][1]));
            let s1 = step_core_subsample(slab, &op_p_1x, 2, 1, (up[1][0], up[1][1]));
            slab.lite.live_os.downsample_4ch::<B>(s0, s1)
        }
        (true, 4) => {
            let up = slab.cold_mut().master_os.upsample_1x_to_4x(ux_l, ux_r);
            let mut sub = [[0.0_f32; 4]; 4];
            for s in 0..4 {
                sub[s] = step_core_subsample(slab, &op_p_1x, 4, s, (up[s][0], up[s][1]));
            }
            slab.cold_mut().master_os.downsample_4x_to_1x::<B>(sub)
        }
        (true, 2) => {
            let up = slab.cold_mut().master_os.upsample_1x_to_2x(ux_l, ux_r);
            let s0 = step_core_subsample(slab, &op_p_1x, 2, 0, (up[0][0], up[0][1]));
            let s1 = step_core_subsample(slab, &op_p_1x, 2, 1, (up[1][0], up[1][1]));
            slab.cold_mut().master_os.downsample_2x_to_1x::<B>([s0, s1])
        }
        _ => step_core_subsample(slab, &op_p_1x, 1, 0, (ux_l, ux_r)),
    };
    tel.last_s2_dir = [pre_l, pre_r, dir_l, dir_r];

    let (conv_l, conv_r) = step_convolver_frame(slab, pre_l, pre_r);
    let x_tank = (dir_l + conv_l, dir_r + conv_r);
    let end = step_transport_end_via_slab_with_telem(slab, reads, x_tank, tel)?;
    let out = finalize_frame_output_with_telem(
        end.tank_lr.0,
        end.tank_lr.1,
        end.core_lr,
        end.wet_lr,
        tel,
    );
    slab.lite.sample_tick = slab.lite.sample_tick.wrapping_add(1);
    Ok(out)
}

#[inline(always)]
fn step_post_sensory<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    u_sense_l: f32,
    u_sense_r: f32,
    tel: &mut BlockTelemetry,
) -> Result<(f32, f32), VmError> {
    let reads = step_transport_top_via_slab(slab)?;
    step_post_sensory_with_reads(slab, &reads, u_sense_l, u_sense_r, tel)
}

/// Executes one complete `2x2` stereo host VM frame (`Steps 1..10`) with deterministic `0-ULP`
/// invariance against [`process_microchunk_4`] and [`process_block_sona`].
#[inline(never)]
pub fn step_frame_sona<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    inline_hot::step_frame_sona_inner::<B, false>(slab, in_l, in_r)
}

#[inline]
fn process_microchunk_4<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    tel: &mut BlockTelemetry,
) -> Result<(), VmError> {
    inline_hot::process_microchunk_4_inner::<B, false>(slab, in_l, in_r, out_l, out_r, tel)
}

/// Processes a stereo audio block with strict causal `4`-frame micro-chunk gating
/// `(sample_tick & 3) == 0 && rem_frames >= 4 && rem_tick >= 4` and `0-ULP` chunk invariance.
pub fn process_block_sona<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), VmError> {
    inline_hot::process_block_sona_inner::<B, false>(slab, in_l, in_r, out_l, out_r)
}

/// Processes a fixed `16`-frame (`48 kHz` `3 kHz` tick) slice via causal [`process_block_sona`].
#[inline(never)]
pub fn process_tick16_const<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32; 16],
    in_r: &[f32; 16],
    out_l: &mut [f32; 16],
    out_r: &mut [f32; 16],
) -> Result<(), VmError> {
    process_block_sona(slab, in_l, in_r, out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_max_quads_and_telemetry_smoke() {
        assert_eq!(TRANSPORT_MAX_QUADS, 4);
    }
}
