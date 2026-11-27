//! Shared `const INLINE_HOT: bool` microchunk/block/frame execution helpers for standard and
//! `#[target_feature]` dispatch (`#![no_std]`, `#![forbid(unsafe_code)]`, `0-ULP` parity).

#![forbid(unsafe_code)]

use crate::graph::GraphBackend;
use crate::graph::sensory::SensoryChunkRegs;
use crate::graph::slab::ctrl_latch::{
    hoist_core_quads_and_readouts, hoist_core_quads_and_readouts_inline, step_control_tick_3khz,
    step_control_tick_3khz_inline,
};
use crate::graph::slab::layout::SonaHotSlab;
use crate::math::{
    CERTIFIED_INPUT_ENVELOPE_MIN_F32, sanitize_input_sample, sanitize_output_sample,
};
use crate::types::VmError;

use super::convolver_chunk::step_convolver_chunk4;
use super::quad_major::{
    eval_core_quads_quad_major, finish_transport_and_output_chunk4,
    prepare_microchunk_crossbar_and_u_sub,
};
use super::telemetry::BlockTelemetry;
use super::{process_microchunk_4, step_post_sensory, step_transport_top_chunk4_via_slab};

#[inline(always)]
fn control_step_frame_inner<B: GraphBackend, const INLINE_HOT: bool>(
    slab: &mut SonaHotSlab<'_, B>,
) {
    let tick_mod4 = (slab.lite.sample_tick & 3) == 0;
    let det_env = slab.lite.sensory_quad.envelopes();
    let cold = &mut slab.cold.as_mut_slice()[0];
    let mut need_3khz = false;
    let mut need_hoist = false;

    if let Some(bus) = cold.ctrl_bus.as_mut() {
        let was_zero = bus.ctrl_sub_counter == 0;
        let active = bus.should_tick();
        if active && was_zero {
            need_3khz = true;
        }
        if INLINE_HOT {
            bus.step_frame_inline(Some(det_env));
        } else {
            bus.step_frame(Some(det_env));
        }
        if tick_mod4 || need_3khz {
            let new_z = bus.eff_z_stage();
            let stage_changed = (0..32).any(|i| {
                bus.current_stage_params[i].to_bits() != cold.latched_stage_params[i].to_bits()
            }) || (0..8)
                .any(|i| new_z[i].to_bits() != cold.latched_z_stage[i].to_bits());
            cold.latched_stage_params = bus.current_stage_params;
            cold.latched_z_stage = new_z;
            if stage_changed {
                need_hoist = true;
            }
        }
    }

    if need_3khz {
        if INLINE_HOT {
            step_control_tick_3khz_inline(slab);
        } else {
            step_control_tick_3khz(slab);
        }
    } else if need_hoist {
        if INLINE_HOT {
            hoist_core_quads_and_readouts_inline(slab);
        } else {
            hoist_core_quads_and_readouts(slab);
        }
    }
}

#[inline(always)]
fn step_frame_sona_with_telem<B: GraphBackend, const INLINE_HOT: bool>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
    tel: &mut BlockTelemetry,
) -> Result<(f32, f32), VmError> {
    let in_r = if slab.single_input() { in_l } else { in_r };
    let clean_l = sanitize_input_sample(in_l, CERTIFIED_INPUT_ENVELOPE_MIN_F32, &mut tel.sat_flags);
    let clean_r = sanitize_input_sample(in_r, CERTIFIED_INPUT_ENVELOPE_MIN_F32, &mut tel.sat_flags);
    control_step_frame_inner::<B, INLINE_HOT>(slab);
    let slot30 = slab.cold().latched_stage_params[30];
    let (u_l, u_r) =
        slab.sensory_params
            .step_frame::<B>(&mut slab.lite.sensory_quad, clean_l, clean_r, slot30);
    step_post_sensory(slab, u_l, u_r, tel)
}

#[inline(always)]
pub(super) fn step_frame_sona_inner<B: GraphBackend, const INLINE_HOT: bool>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    let mut tel = BlockTelemetry::begin_block(slab);
    let res = step_frame_sona_with_telem::<B, INLINE_HOT>(slab, in_l, in_r, &mut tel);
    tel.flush_into(slab);
    res
}

#[inline(always)]
fn step_frame_4x_sona_with_telem<B: GraphBackend, const INLINE_HOT: bool>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
    tel: &mut BlockTelemetry,
) -> Result<(f32, f32), VmError> {
    let in_r = if slab.single_input() { in_l } else { in_r };
    let clean_l = sanitize_input_sample(in_l, CERTIFIED_INPUT_ENVELOPE_MIN_F32, &mut tel.sat_flags);
    let clean_r = sanitize_input_sample(in_r, CERTIFIED_INPUT_ENVELOPE_MIN_F32, &mut tel.sat_flags);
    let up4 = slab.cold.as_mut_slice()[0]
        .bounce4x
        .upsample_4x(clean_l, clean_r);
    let mut sub_out = [[0.0_f32; 2]; 4];
    let mut sub_core = [[0.0_f32; 2]; 4];
    let mut sub_wet = [[0.0_f32; 2]; 4];
    for (s, &[sl, sr]) in up4.iter().enumerate() {
        let (ol, or) = step_frame_sona_with_telem::<B, INLINE_HOT>(slab, sl, sr, tel)?;
        sub_out[s] = [ol, or];
        sub_core[s] = tel.last_core_lr;
        sub_wet[s] = tel.last_transport_wet_lr;
    }
    let (dl, dr, d_core_l, d_core_r, d_wet_l, d_wet_r) = {
        let cold = slab.cold_mut();
        let [dl, dr] = cold.bounce4x.downsample_4x(sub_out);
        let [d_core_l, d_core_r] = cold.bounce4x_core_decim.downsample_4x(sub_core);
        let [d_wet_l, d_wet_r] = cold.bounce4x_wet_decim.downsample_4x(sub_wet);
        (dl, dr, d_core_l, d_core_r, d_wet_l, d_wet_r)
    };
    let (out_l, out_r) = (
        sanitize_output_sample(dl, &mut tel.sat_flags),
        sanitize_output_sample(dr, &mut tel.sat_flags),
    );
    let mut part_flags = 0_u32;
    tel.last_core_lr = [
        sanitize_output_sample(d_core_l, &mut part_flags),
        sanitize_output_sample(d_core_r, &mut part_flags),
    ];
    tel.last_transport_wet_lr = [
        sanitize_output_sample(d_wet_l, &mut part_flags),
        sanitize_output_sample(d_wet_r, &mut part_flags),
    ];
    Ok((out_l, out_r))
}

#[inline(always)]
pub(super) fn step_frame_4x_sona_inner<B: GraphBackend, const INLINE_HOT: bool>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    let mut tel = BlockTelemetry::begin_block(slab);
    let res = step_frame_4x_sona_with_telem::<B, INLINE_HOT>(slab, in_l, in_r, &mut tel);
    tel.flush_into(slab);
    res
}

#[inline(always)]
pub(super) fn process_microchunk_4_inner<B: GraphBackend, const INLINE_HOT: bool>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    tel: &mut BlockTelemetry,
) -> Result<(), VmError> {
    let single_in = slab.single_input();
    let mut u_sense_lr = [[0.0_f32; 2]; 4];
    let cl0 = sanitize_input_sample(
        in_l[0],
        CERTIFIED_INPUT_ENVELOPE_MIN_F32,
        &mut tel.sat_flags,
    );
    let cr0 = sanitize_input_sample(
        if single_in { in_l[0] } else { in_r[0] },
        CERTIFIED_INPUT_ENVELOPE_MIN_F32,
        &mut tel.sat_flags,
    );
    control_step_frame_inner::<B, INLINE_HOT>(slab);

    let slot30 = slab.cold().latched_stage_params[30];
    let mut sense =
        SensoryChunkRegs::<B::Vec4>::load(&slab.sensory_params, &slab.lite.sensory_quad, slot30);
    let (u0_l, u0_r) = sense.step_frame(cl0, cr0);
    u_sense_lr[0] = [u0_l, u0_r];

    for f in 1..4 {
        let clf = sanitize_input_sample(
            in_l[f],
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut tel.sat_flags,
        );
        let crf = sanitize_input_sample(
            if single_in { in_l[f] } else { in_r[f] },
            CERTIFIED_INPUT_ENVELOPE_MIN_F32,
            &mut tel.sat_flags,
        );
        let det_env = sense.envelopes_array();
        if let Some(bus) = slab.cold.as_mut_slice()[0].ctrl_bus.as_mut() {
            if INLINE_HOT {
                bus.step_frame_inline(Some(det_env));
            } else {
                bus.step_frame(Some(det_env));
            }
        }
        let (uf_l, uf_r) = sense.step_frame(clf, crf);
        u_sense_lr[f] = [uf_l, uf_r];
    }
    sense.store(&mut slab.lite.sensory_quad);

    let reads_4 = step_transport_top_chunk4_via_slab(slab)?;
    let inputs = prepare_microchunk_crossbar_and_u_sub(slab, &reads_4, &u_sense_lr, tel);
    let (pre_lr, dir_lr) = eval_core_quads_quad_major(slab, &inputs, tel);
    let conv_lr = step_convolver_chunk4(slab, pre_lr);
    finish_transport_and_output_chunk4(slab, &reads_4, &dir_lr, &conv_lr, out_l, out_r, tel)
}

#[inline(always)]
pub(super) fn process_block_sona_inner<B: GraphBackend, const INLINE_HOT: bool>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), VmError> {
    let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
    if n == 0 {
        return Ok(());
    }
    let mut tel = BlockTelemetry::begin_block(slab);
    if slab.cold().offline_bounce {
        for i in 0..n {
            match step_frame_4x_sona_with_telem::<B, INLINE_HOT>(slab, in_l[i], in_r[i], &mut tel) {
                Ok((yl, yr)) => {
                    out_l[i] = yl;
                    out_r[i] = yr;
                }
                Err(e) => {
                    tel.flush_into(slab);
                    return Err(e);
                }
            }
        }
        tel.flush_into(slab);
        return Ok(());
    }

    let mut idx = 0_usize;
    while idx < n {
        let rem_frames = n - idx;
        let rem_tick = match slab.cold().ctrl_bus.as_ref() {
            Some(bus) if bus.should_tick() => {
                (bus.ctrl_stride.saturating_sub(bus.ctrl_sub_counter) as usize).max(1)
            }
            _ => usize::MAX,
        };
        if (slab.lite.sample_tick & 3) == 0 && rem_frames >= 4 && rem_tick >= 4 {
            let res = if INLINE_HOT {
                process_microchunk_4_inner::<B, true>(
                    slab,
                    &in_l[idx..idx + 4],
                    &in_r[idx..idx + 4],
                    &mut out_l[idx..idx + 4],
                    &mut out_r[idx..idx + 4],
                    &mut tel,
                )
            } else {
                process_microchunk_4(
                    slab,
                    &in_l[idx..idx + 4],
                    &in_r[idx..idx + 4],
                    &mut out_l[idx..idx + 4],
                    &mut out_r[idx..idx + 4],
                    &mut tel,
                )
            };
            if let Err(e) = res {
                tel.flush_into(slab);
                return Err(e);
            }
            idx += 4;
        } else {
            match step_frame_sona_with_telem::<B, INLINE_HOT>(slab, in_l[idx], in_r[idx], &mut tel)
            {
                Ok((yl, yr)) => {
                    out_l[idx] = yl;
                    out_r[idx] = yr;
                    idx += 1;
                }
                Err(e) => {
                    tel.flush_into(slab);
                    return Err(e);
                }
            }
        }
    }
    tel.flush_into(slab);
    Ok(())
}

/// Always-inlined single-frame execution for `#[target_feature]` dispatchers (`0-ULP`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn step_frame_sona_inline<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    step_frame_sona_inner::<B, true>(slab, in_l, in_r)
}

/// Always-inlined `4x` bounce single-frame execution for `#[target_feature]` dispatchers.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn step_frame_4x_sona_inline<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    step_frame_4x_sona_inner::<B, true>(slab, in_l, in_r)
}

/// Always-inlined block execution for `#[target_feature]` dispatchers (`0-ULP`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn process_block_sona_inline<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), VmError> {
    process_block_sona_inner::<B, true>(slab, in_l, in_r, out_l, out_r)
}

/// Always-inlined `16`-frame execution for `#[target_feature]` dispatchers (`0-ULP`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn process_tick16_const_inline<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32; 16],
    in_r: &[f32; 16],
    out_l: &mut [f32; 16],
    out_r: &mut [f32; 16],
) -> Result<(), VmError> {
    process_block_sona_inner::<B, true>(slab, in_l, in_r, out_l, out_r)
}
