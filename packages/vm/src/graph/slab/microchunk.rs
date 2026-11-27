//! Causal `4`-frame micro-chunking, `1`-frame peel, `2x2` pure stereo / `4x` bounce frame
//! stepping, and `0-ULP` chunk-split invariant SONA block execution (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

pub mod core_step;
mod inline_hot;

use dnam_kernels::scrub_and_flush_f32_bits;

use crate::graph::GraphBackend;
use crate::graph::convolver::kautz::KautzStereoBank4;
use crate::graph::transport::{QuadTopRead, TRANSPORT_MAX_QUADS, resolve_quad_ring};
use crate::math::{
    flush_recursive_state_f32, fp16_to_f32, sanitize_output_sample, saturation_flags,
};
use crate::types::{Tier, VmError};

use super::layout::{FLAG_XFADE_ACTIVE, SonaHotSlab, TIER_SWITCH_XFADE_SAMPLES};
use core_step::{step_core_subsample, step_op_p_1x};

pub use core_step::evaluate_matryoshka_core_quads;
#[cfg(target_arch = "x86_64")]
pub(crate) use inline_hot::{
    process_block_sona_inline, process_tick16_const_inline, step_frame_4x_sona_inline,
    step_frame_sona_inline,
};

const INV_XFADE_LEN_F32: f32 = 1.0 / (TIER_SWITCH_XFADE_SAMPLES as f32);

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
pub(super) fn step_transport_top_chunk4_via_slab<B: GraphBackend>(
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

#[inline(always)]
fn step_transport_end_via_slab<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    x_tank_lr: (f32, f32),
) -> Result<(f32, f32), VmError> {
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
        cold.saturation_flags |= saturation_flags::FP16_SATURATED;
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
    cold.last_core_lr = [core_l, core_r];
    cold.last_transport_wet_lr = [wet_l, wet_r];
    Ok((core_l + wet_l, core_r + wet_r))
}

#[inline(always)]
fn block16_to_4x4(b: &[f32; 16]) -> [[f32; 4]; 4] {
    let (c, _) = b.as_chunks::<4>();
    [c[0], c[1], c[2], c[3]]
}

#[inline(always)]
fn shift_hist_and_carry<const N: usize>(hist: &mut [f32; N], carry_in: f32) -> f32 {
    let carry_out = hist[N - 1];
    hist.copy_within(0..N - 1, 1);
    hist[0] = carry_in;
    carry_out
}

#[inline(always)]
fn accum_fir_segs<B: GraphBackend>(
    acc_l: &mut f32,
    acc_r: &mut f32,
    wl_segs: &[[[f32; 4]; 4]],
    wr_segs: &[[[f32; 4]; 4]],
    hl_blocks: &[[f32; 16]],
    hr_blocks: &[[f32; 16]],
) {
    for (((wl, wr), hl), hr) in wl_segs.iter().zip(wr_segs).zip(hl_blocks).zip(hr_blocks) {
        let xl = block16_to_4x4(hl);
        let xr = block16_to_4x4(hr);
        let (sl, sr) = B::dot16_stereo(wl, wr, &xl, &xr);
        *acc_l += sl;
        *acc_r += sr;
    }
}

#[inline(always)]
fn step_convolver_frame<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    pre_l: f32,
    pre_r: f32,
) -> (f32, f32) {
    let tier = slab.lite.tier;
    let cold_ext = slab.cold_fir_ext();
    let (early, onset) = if cold_ext {
        slab.cold.as_mut_slice()[0]
            .convolver_fir_ext
            .step_frame(pre_l, pre_r)
    } else {
        let c_lite_l =
            shift_hist_and_carry(&mut slab.lite.fir_hist_l, flush_recursive_state_f32(pre_l));
        let c_lite_r =
            shift_hist_and_carry(&mut slab.lite.fir_hist_r, flush_recursive_state_f32(pre_r));
        let onset = (slab.lite.fir_hist_l[16], slab.lite.fir_hist_r[16]);

        let mut early_l = 0.0_f32;
        let mut early_r = 0.0_f32;
        accum_fir_segs::<B>(
            &mut early_l,
            &mut early_r,
            &slab.lite.fir_weights_l,
            &slab.lite.fir_weights_r,
            slab.lite.fir_hist_l.as_chunks::<16>().0,
            slab.lite.fir_hist_r.as_chunks::<16>().0,
        );
        if tier >= Tier::Full {
            let c_full_l = shift_hist_and_carry(&mut slab.full_delta.fir_hist_l_full, c_lite_l);
            let c_full_r = shift_hist_and_carry(&mut slab.full_delta.fir_hist_r_full, c_lite_r);
            accum_fir_segs::<B>(
                &mut early_l,
                &mut early_r,
                &slab.full_delta.fir_weights_l_full,
                &slab.full_delta.fir_weights_r_full,
                core::slice::from_ref(&slab.full_delta.fir_hist_l_full),
                core::slice::from_ref(&slab.full_delta.fir_hist_r_full),
            );
            if tier >= Tier::Pro {
                let c_pro_l = shift_hist_and_carry(&mut slab.pro_delta.fir_hist_l_pro, c_full_l);
                let c_pro_r = shift_hist_and_carry(&mut slab.pro_delta.fir_hist_r_pro, c_full_r);
                accum_fir_segs::<B>(
                    &mut early_l,
                    &mut early_r,
                    &slab.pro_delta.fir_weights_l_pro,
                    &slab.pro_delta.fir_weights_r_pro,
                    slab.pro_delta.fir_hist_l_pro.as_chunks::<16>().0,
                    slab.pro_delta.fir_hist_r_pro.as_chunks::<16>().0,
                );
                if tier == Tier::Ultra {
                    shift_hist_and_carry(&mut slab.ultra_delta.fir_hist_l_ultra, c_pro_l);
                    shift_hist_and_carry(&mut slab.ultra_delta.fir_hist_r_ultra, c_pro_r);
                    accum_fir_segs::<B>(
                        &mut early_l,
                        &mut early_r,
                        &slab.ultra_delta.fir_weights_l_ultra,
                        &slab.ultra_delta.fir_weights_r_ultra,
                        slab.ultra_delta.fir_hist_l_ultra.as_chunks::<16>().0,
                        slab.ultra_delta.fir_hist_r_ultra.as_chunks::<16>().0,
                    );
                }
            }
        }
        ((early_l, early_r), onset)
    };

    let active_banks = slab.cold().active_kautz_blocks;
    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for b in 0..active_banks {
        let (yl, yr) = KautzStereoBank4::step_banks_stereo_inline::<B>(
            core::slice::from_mut(slab.kautz_bank_mut(b)),
            onset.0,
            onset.1,
        );
        sum_l += yl;
        sum_r += yr;
    }
    (early.0 + sum_l, early.1 + sum_r)
}

#[inline(always)]
fn fold_mono_lr(mono: bool, l: f32, r: f32) -> (f32, f32) {
    if mono {
        let m = 0.5 * (l + r);
        (m, m)
    } else {
        (l, r)
    }
}

#[inline(always)]
fn finalize_frame_output<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    tank_l: f32,
    tank_r: f32,
) -> (f32, f32) {
    let mono = slab.mono_out();
    let (mut out_l, mut out_r) = fold_mono_lr(mono, tank_l, tank_r);
    let [raw_core_l, raw_core_r] = slab.cold().last_core_lr;
    let [raw_wet_l, raw_wet_r] = slab.cold().last_transport_wet_lr;
    let (mut core_l, mut core_r) = fold_mono_lr(mono, raw_core_l, raw_core_r);
    let (mut wet_l, mut wet_r) = fold_mono_lr(mono, raw_wet_l, raw_wet_r);

    let rem = if slab.xfade_active() {
        slab.cold().xfade_rem
    } else {
        0
    };
    if rem > 0 {
        let a = f32::from(rem) * INV_XFADE_LEN_F32;
        let one_m_a = 1.0 - a;
        let prev = slab.cold().xfade_prev_out;
        let (prev_l, prev_r) = fold_mono_lr(mono, prev[0], prev[1]);
        out_l = prev_l * a + out_l * one_m_a;
        out_r = prev_r * a + out_r * one_m_a;

        let prev_c = slab.cold().xfade_prev_core;
        let (prev_cl, prev_cr) = fold_mono_lr(mono, prev_c[0], prev_c[1]);
        core_l = prev_cl * a + core_l * one_m_a;
        core_r = prev_cr * a + core_r * one_m_a;

        let prev_w = slab.cold().xfade_prev_wet;
        let (prev_wl, prev_wr) = fold_mono_lr(mono, prev_w[0], prev_w[1]);
        wet_l = prev_wl * a + wet_l * one_m_a;
        wet_r = prev_wr * a + wet_r * one_m_a;

        let next_rem = rem - 1;
        slab.cold_mut().xfade_rem = next_rem;
        if next_rem == 0 {
            slab.set_flag(FLAG_XFADE_ACTIVE, false);
            let cold = slab.cold_mut();
            cold.xfade_prev_out = [out_l, out_r];
            cold.xfade_prev_core = [core_l, core_r];
            cold.xfade_prev_wet = [wet_l, wet_r];
        }
    } else {
        let cold = slab.cold_mut();
        cold.xfade_prev_out = [out_l, out_r];
        cold.xfade_prev_core = [core_l, core_r];
        cold.xfade_prev_wet = [wet_l, wet_r];
    }

    let tier_idx = slab.lite.tier.index();
    let sign = if slab.wet_polarity_inv() { -1.0 } else { 1.0 };
    let eff_trim = sign * slab.cold().tier_gain_trim[tier_idx];
    out_l *= eff_trim;
    out_r *= eff_trim;
    core_l *= eff_trim;
    core_r *= eff_trim;
    wet_l *= eff_trim;
    wet_r *= eff_trim;

    slab.lite.sample_tick = slab.lite.sample_tick.wrapping_add(1);
    let flags = &mut slab.cold.as_mut_slice()[0].saturation_flags;
    let clean_l = sanitize_output_sample(out_l, flags);
    let clean_r = sanitize_output_sample(out_r, flags);

    let mut part_flags = 0_u32;
    let clean_core_l = sanitize_output_sample(core_l, &mut part_flags);
    let clean_core_r = sanitize_output_sample(core_r, &mut part_flags);
    let clean_wet_l = sanitize_output_sample(wet_l, &mut part_flags);
    let clean_wet_r = sanitize_output_sample(wet_r, &mut part_flags);
    let cold = slab.cold_mut();
    cold.last_core_lr = [clean_core_l, clean_core_r];
    cold.last_transport_wet_lr = [clean_wet_l, clean_wet_r];
    (clean_l, clean_r)
}

#[inline(always)]
pub(super) fn step_post_sensory_with_reads<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    u_sense_l: f32,
    u_sense_r: f32,
) -> Result<(f32, f32), VmError> {
    let op_p_1x = step_op_p_1x(slab, u_sense_l, u_sense_r);
    {
        let cold = slab.cold_mut();
        cold.last_op_p_iq_lr = op_p_1x.iq_lr;
        cold.last_op_p_feat = op_p_1x.feat;
    }
    let op_t_gain = slab.cold().hoisted_core.op_t_gain;
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
    slab.cold_mut().last_in = [ux_l, ux_r];

    let os2x = slab.os2x();
    let [pre_l, pre_r, dir_l, dir_r] = if os2x {
        let os_params = slab.os2x_hot.os_params;
        let (u0, u1) = slab
            .os2x_hot
            .os_stereo
            .upsample_stereo::<B>(ux_l, ux_r, &os_params);
        let s0 = step_core_subsample(slab, &op_p_1x, true, 0, (u0[0], u0[1]));
        let s1 = step_core_subsample(slab, &op_p_1x, true, 1, (u1[0], u1[1]));
        slab.os2x_hot
            .os_stereo
            .downsample_4ch::<B>(s0, s1, &os_params)
    } else {
        step_core_subsample(slab, &op_p_1x, false, 0, (ux_l, ux_r))
    };
    slab.cold_mut().last_s2_dir = [pre_l, pre_r, dir_l, dir_r];

    let (conv_l, conv_r) = step_convolver_frame(slab, pre_l, pre_r);
    let x_tank = (dir_l + conv_l, dir_r + conv_r);
    let (tank_l, tank_r) = step_transport_end_via_slab(slab, reads, x_tank)?;
    Ok(finalize_frame_output(slab, tank_l, tank_r))
}

#[inline(always)]
pub(super) fn step_post_sensory<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    u_sense_l: f32,
    u_sense_r: f32,
) -> Result<(f32, f32), VmError> {
    let reads = step_transport_top_via_slab(slab)?;
    step_post_sensory_with_reads(slab, &reads, u_sense_l, u_sense_r)
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

/// Executes one host sample under deterministic `4x` offline bounce oversampling (`Bounce4xOversampler`).
#[inline(never)]
pub fn step_frame_4x_sona<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: f32,
    in_r: f32,
) -> Result<(f32, f32), VmError> {
    inline_hot::step_frame_4x_sona_inner::<B, false>(slab, in_l, in_r)
}

#[inline]
fn process_microchunk_4<B: GraphBackend>(
    slab: &mut SonaHotSlab<'_, B>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), VmError> {
    inline_hot::process_microchunk_4_inner::<B, false>(slab, in_l, in_r, out_l, out_r)
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
    fn test_xfade_reciprocal_constant_exact() {
        assert!((INV_XFADE_LEN_F32 * 48.0 - 1.0).abs() < 1.0e-7);
        assert_eq!(TRANSPORT_MAX_QUADS, 4);
    }
}
