//! Multi-quad Spatial Tank orchestration (`Lite` 1..`Ultra` 4 orthogonal 4-lane `Op-T` quads),
//! `3 kHz` `ControlBus` dispatch (`slots 12..=25`), causal Top-of-Frame + End-of-Frame stereo
//! stepping, and submodules ([`quad`], [`ring`]) (`ref[dsp.op-t.ring.declaration]`, `ref[dsp.tank.feedback-loop]`).

pub mod quad;
pub mod ring;

use core::f32::consts::PI;

use dnam_kernels::{
    QUAD_FLAG_HAS_EXT, algebraic_unit_rotor_f32, compute_half_alpha_cubic_f32,
    compute_op_t_pitch_down_g_loop_eff,
};

use crate::bin::arena_wire::{TANK_FLAG_SCHROEDER_ENABLE, TankQuadWire};
use crate::control::bus::laws::{
    eval_crossbar_wet_scale, eval_op_t_we_eff, eval_tank_allpass_gains, eval_tank_damp_pole,
    eval_tank_delay_mod_48k, eval_tank_delay_scale, eval_tank_dry_mix_for_quad, eval_tank_g_loop,
    eval_tank_input_gains, eval_tank_lfo_depths, eval_tank_lfo_inc_q32,
    eval_tank_ramp_rate_and_inc_q32, eval_tank_target_psi, eval_tank_window_depth,
};
use crate::math::{MathBackend, f32_to_fp16, fp16_to_f32};
use crate::types::VmError;

pub use quad::{
    QuadColdMeta, QuadEffectiveTargets, QuadHydrateConfig, QuadRingMut, QuadStepResult,
    TankQuadHotCompact,
};
pub use ring::{
    PartitionedSchroederBank, TANK_ALLPASS_G_MAX, TANK_INLINE_QUAD0_CAP, TANK_MAX_EXT_CAP,
    TANK_MIN_EXT_CAP, TANK_SCHROEDER_MAX_DELAY, TankExtRing, TankExtStorage,
    compute_quad_ext_requirements, compute_schroeder_requirements, f32_to_fp16_rtz_with_sat,
    required_ext_elements,
};

/// Maximum number of orthogonal 4-lane quads across tiers (`Ultra`: `4` quads = `16` lines).
pub const TANK_MAX_QUADS: usize = 4;

const PSI_SLEW_MAX_RAD: f32 = 0.25;

/// Extracted `ControlBus` `stage_targets[12..=25]` slots governing Spatial Tank modulation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuadModSlots {
    /// Slot 12: `tank_delay_mod` in `48 kHz` reference samples.
    pub tank_delay_mod: f32,
    /// Slot 13: `tank_gain_mod`.
    pub tank_gain_mod: f32,
    /// Slot 14 (plus any `w_tank_psi` sensitivity dot): `tank_psi_delta` in `[-pi, pi]` rad.
    pub tank_psi_delta: f32,
    /// Slot 15: `crossbar_wet_mod` in `[-1.0, 1.0]`.
    pub crossbar_wet_mod: f32,
    /// Slot 16: `op_t_we_drive` snapped immediately (`we_eff = clamp(1 + slot 16, 0, 1)`).
    pub op_t_we_drive: f32,
    /// Slot 17: `op_t_delay_scale_oct` in `[-3.0, 3.0]`.
    pub op_t_delay_scale_oct: f32,
    /// Slot 18: `op_t_ramp_rate_hz_delta`.
    pub op_t_ramp_rate_hz_delta: f32,
    /// Slot 19: `op_t_lfo_rate_hz_delta`.
    pub op_t_lfo_rate_hz_delta: f32,
    /// Slot 20: `op_t_lfo_depth_scale`.
    pub op_t_lfo_depth_scale: f32,
    /// Slot 21: `op_t_damp_pole_delta`.
    pub op_t_damp_pole_delta: f32,
    /// Slot 22: `op_t_dry_mix_mod` (applied strictly to Quad 0; `Quads 1..=3` remain constant).
    pub op_t_dry_mix_mod: f32,
    /// Slot 23: `op_t_input_gain_mod`.
    pub op_t_input_gain_mod: f32,
    /// Slot 24: `op_t_window_depth_delta`.
    pub op_t_window_depth_delta: f32,
    /// Slot 25: `op_t_allpass_g_delta`.
    pub op_t_allpass_g_delta: f32,
}

impl QuadModSlots {
    /// Maps a canonical 32-element `ControlBus` `stage_targets` slice (`[12..=25]`) into [`QuadModSlots`].
    #[inline]
    pub const fn from_targets_32(t: &[f32; 32]) -> Self {
        Self {
            tank_delay_mod: t[12],
            tank_gain_mod: t[13],
            tank_psi_delta: t[14],
            crossbar_wet_mod: t[15],
            op_t_we_drive: t[16],
            op_t_delay_scale_oct: t[17],
            op_t_ramp_rate_hz_delta: t[18],
            op_t_lfo_rate_hz_delta: t[19],
            op_t_lfo_depth_scale: t[20],
            op_t_damp_pole_delta: t[21],
            op_t_dry_mix_mod: t[22],
            op_t_input_gain_mod: t[23],
            op_t_window_depth_delta: t[24],
            op_t_allpass_g_delta: t[25],
        }
    }
}

/// Cumulative cold external storage element requirements across all active tank quads (`!alloc`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TankColdStorageSummary {
    /// Total `u16` elements required across `fp16` external quad delay rings.
    pub ext_fp16_u16: usize,
    /// Total `f32` elements required across `f32` external quad delay rings.
    pub ext_f32: usize,
    /// Total `f32` elements required across enabled partitioned Schroeder diffuser banks.
    pub schroeder_f32: usize,
}

/// Cumulative Top-of-Frame output across all active spatial tank quads (`<= 4`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TankTopOfFrameResult {
    /// Summed stereo wet+dry spatial tank output `(y_{tank,L}, y_{tank,R})`.
    pub y_lr: (f32, f32),
    /// Per-quad `v_read` vectors (`0..p_q`) forwarded into [`step_tank_end_of_frame`].
    pub v_reads: [[f32; 4]; TANK_MAX_QUADS],
}

/// Computes deterministic cold external delay and Schroeder element requirements for `wires[..min(4)]`.
pub fn compute_tank_storage_requirements(
    wires: &[TankQuadWire],
    rho_vm: f64,
    declared_max_delay_48k: u32,
    ext_ring_pow2_frames: u8,
) -> TankColdStorageSummary {
    let mut sum = TankColdStorageSummary::default();
    for (q, wire) in wires.iter().take(TANK_MAX_QUADS).enumerate() {
        let (u16_len, f32_len) = required_ext_elements(
            wire,
            q,
            rho_vm,
            declared_max_delay_48k,
            ext_ring_pow2_frames,
        );
        sum.ext_fp16_u16 += u16_len;
        sum.ext_f32 += f32_len;
        if (wire.flags & TANK_FLAG_SCHROEDER_ENABLE) != 0 {
            let (_, _, _, sch_len) = compute_schroeder_requirements(wire, rho_vm);
            sum.schroeder_f32 += sch_len;
        }
    }
    sum
}

/// Latches all non-slew `Op-T` modulation targets (`slots 12, 13, 15, 17..=25`) at `4`-frame
/// `(sample_tick & 3) == 0` or `3 kHz` cadence without stepping `WE` Schmitt or `psi` slew
/// (`ref[dsp.ctrl.dispatch.micro-ramp]`).
pub fn latch_quad_nonslew_targets(
    quad: &mut TankQuadHotCompact,
    meta: &QuadColdMeta,
    slots: &QuadModSlots,
    schroeder: Option<&mut PartitionedSchroederBank<'_>>,
    eff: &mut QuadEffectiveTargets,
) {
    quad.g_loop = eval_tank_g_loop(meta.base_g_loop, slots.tank_gain_mod);
    let (ramp_rate, ramp_inc_q32) = eval_tank_ramp_rate_and_inc_q32(
        meta.base_ramp_rate_hz,
        slots.op_t_ramp_rate_hz_delta,
        meta.inv_fs_vm,
    );
    quad.ramp_inc_q32 = ramp_inc_q32;
    let g_loop_eff =
        compute_op_t_pitch_down_g_loop_eff(quad.g_loop, quad.ramp_spans_samples, quad.ramp_inc_q32);
    quad.half_alpha_cubic_fp16 = f32_to_fp16(compute_half_alpha_cubic_f32(g_loop_eff));

    let (lfo_fp16, lfo_eff, max_lfo) =
        eval_tank_lfo_depths(meta.base_lfo_depths, slots.op_t_lfo_depth_scale, meta.cap);
    quad.lfo_depths_fp16 = lfo_fp16;
    eff.lfo_depths_eff = lfo_eff;

    let (inj_fp16, inj_eff) =
        eval_tank_input_gains(meta.base_input_gains, slots.op_t_input_gain_mod);
    quad.input_gains_fp16 = inj_fp16;
    eff.input_gains_eff = inj_eff;

    eff.allpass_g_eff = eval_tank_allpass_gains(meta.base_allpass_g, slots.op_t_allpass_g_delta);
    if let Some(ap) = schroeder {
        ap.set_allpass_g(eff.allpass_g_eff);
    }

    quad.lfo_inc_q32 = eval_tank_lfo_inc_q32(
        meta.base_lfo_rate_hz,
        slots.op_t_lfo_rate_hz_delta,
        max_lfo,
        meta.inv_fs_vm,
    );
    quad.damp_pole = eval_tank_damp_pole(meta.base_damp_pole_zoh, slots.op_t_damp_pole_delta);
    let dry_eff = eval_tank_dry_mix_for_quad(
        usize::from(meta.quad_idx),
        meta.base_dry_mix,
        slots.op_t_dry_mix_mod,
    );
    quad.dry_mix_fp16 = f32_to_fp16(dry_eff);
    let win_eff = eval_tank_window_depth(meta.base_window_depth, slots.op_t_window_depth_delta);
    quad.window_depth_fp16 = f32_to_fp16(win_eff);

    let (oct, delay_scale_pow2) = eval_tank_delay_scale(slots.op_t_delay_scale_oct);
    eff.cap = meta.cap;
    eff.scale_inv_rho_vm = meta.scale_inv_rho_vm;
    eff.tank_delay_mod_48k = eval_tank_delay_mod_48k(slots.tank_delay_mod);
    eff.delay_scale_oct = oct;
    eff.delay_scale_pow2 = delay_scale_pow2;
    eff.g_loop_eff = g_loop_eff;
    eff.inv_sigma = fp16_to_f32(quad.inv_sigma_fp16);
    eff.half_alpha_cubic = fp16_to_f32(quad.half_alpha_cubic_fp16);
    (eff.cos_psi, eff.sin_psi) = (quad.cos_psi, quad.sin_psi);
    eff.damp_pole_eff = quad.damp_pole;
    eff.dry_mix_eff = fp16_to_f32(quad.dry_mix_fp16);
    eff.wet_scale = eval_crossbar_wet_scale(slots.crossbar_wet_mod);
    eff.window_depth_eff = fp16_to_f32(quad.window_depth_fp16);
    eff.ramp_rate_hz_eff = ramp_rate;
    eff.vca_pole = quad.vca_pole;
    eff.vca_sens = fp16_to_f32(quad.vca_sens_fp16);
    eff.attack_cap_vca = meta.attack_cap_vca;
}

/// Applies `3 kHz` `ControlBus` modulation (`slots 12..=25`), `<= 0.25` rad/tick Givens slewing,
/// immediate `WE` Schmitt trigger update, and Schroeder gain updates on one quad.
pub fn tick_quad_3khz(
    quad: &mut TankQuadHotCompact,
    meta: &QuadColdMeta,
    slots: &QuadModSlots,
    ext_ring: Option<&mut TankExtRing<'_>>,
    schroeder: Option<&mut PartitionedSchroederBank<'_>>,
    eff: &mut QuadEffectiveTargets,
) {
    let we_eff = eval_op_t_we_eff(slots.op_t_we_drive);
    quad.update_we_schmitt(we_eff, ext_ring);

    let target_psi = eval_tank_target_psi(meta.base_psi, slots.tank_psi_delta);
    let step_psi = (target_psi - quad.current_psi).clamp(-PSI_SLEW_MAX_RAD, PSI_SLEW_MAX_RAD);
    quad.current_psi = (quad.current_psi + step_psi).clamp(-PI, PI);
    (quad.cos_psi, quad.sin_psi) = algebraic_unit_rotor_f32(quad.current_psi);

    latch_quad_nonslew_targets(quad, meta, slots, schroeder, eff);
}

/// Applies `3 kHz` `ControlBus` modulation uniformly across all `p_q` active tank quads, while
/// gating `slot 22` (`op_t_dry_mix_mod`) strictly onto `Quad 0`.
pub fn tick_tank_3khz(
    quads: &mut [TankQuadHotCompact],
    metas: &[QuadColdMeta],
    p_q: usize,
    slots: &QuadModSlots,
    ext_rings: &mut [TankExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &mut [QuadEffectiveTargets],
) {
    let active_q = p_q
        .min(quads.len())
        .min(metas.len())
        .min(effective.len())
        .min(TANK_MAX_QUADS);
    for q in 0..active_q {
        tick_quad_3khz(
            &mut quads[q],
            &metas[q],
            slots,
            ext_rings.get_mut(q),
            schroeders.get_mut(q),
            &mut effective[q],
        );
    }
}

/// Resets all active tank quads, inline `Quad 0` ring, external rings, and Schroeder banks.
pub fn reset_tank(
    quads: &mut [TankQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TANK_INLINE_QUAD0_CAP],
    ext_rings: &mut [TankExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
) {
    let active_q = p_q.min(quads.len()).min(TANK_MAX_QUADS);
    inline_q0_ring.fill([0; 4]);
    for q in 0..active_q {
        quads[q].reset(ext_rings.get_mut(q), None, schroeders.get_mut(q));
    }
}

#[inline(always)]
pub(crate) fn resolve_quad_ring<'a, 'b>(
    q: usize,
    flags: u8,
    inline_q0_ring: &'a mut [[u16; 4]; TANK_INLINE_QUAD0_CAP],
    ext_rings: &'a mut [TankExtRing<'b>],
) -> Result<QuadRingMut<'a, 'b>, VmError> {
    if (flags & QUAD_FLAG_HAS_EXT) != 0 {
        let ext = ext_rings
            .get_mut(q)
            .ok_or(VmError::InsufficientBackingCapacity)?;
        if !ext.is_external() {
            return Err(VmError::InsufficientBackingCapacity);
        }
        Ok(QuadRingMut::Ext(ext))
    } else if q == 0 {
        Ok(QuadRingMut::Inline(inline_q0_ring))
    } else {
        Err(VmError::InsufficientBackingCapacity)
    }
}

/// Executes Top-of-Frame causal readout across `0..p_q` active spatial tank quads, summing their
/// stereo wet+dry outputs and returning per-quad `v_reads` for End-of-Frame writeback. Supports
/// `Lite` inline `Quad 0` operation with empty `ext_rings` and fails closed on missing `HAS_EXT`.
#[inline]
pub fn step_tank_top_of_frame<B: MathBackend>(
    quads: &mut [TankQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TANK_INLINE_QUAD0_CAP],
    ext_rings: &mut [TankExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &[QuadEffectiveTargets],
    x_tank_lr: (f32, f32),
) -> Result<TankTopOfFrameResult, VmError> {
    let active_q = p_q
        .min(quads.len())
        .min(effective.len())
        .min(TANK_MAX_QUADS);
    let mut out = TankTopOfFrameResult::default();
    for q in 0..active_q {
        let ring = resolve_quad_ring(q, quads[q].flags, inline_q0_ring, ext_rings)?;
        let res = quads[q].step_top_of_frame_read::<B>(
            ring,
            schroeders.get_mut(q),
            &effective[q],
            x_tank_lr,
        );
        out.y_lr.0 += res.y_lr.0;
        out.y_lr.1 += res.y_lr.1;
        out.v_reads[q] = res.v_read;
    }
    Ok(out)
}

/// Executes End-of-Frame loop feedback + delay writeback across `0..p_q` active spatial tank quads,
/// returning `true` if any `fp16` delay ring saturated at `±65,504.0`.
#[inline]
pub fn step_tank_end_of_frame<B: MathBackend>(
    quads: &mut [TankQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TANK_INLINE_QUAD0_CAP],
    ext_rings: &mut [TankExtRing<'_>],
    effective: &[QuadEffectiveTargets],
    v_reads: &[[f32; 4]; TANK_MAX_QUADS],
    x_tank_lr: (f32, f32),
) -> Result<bool, VmError> {
    let active_q = p_q
        .min(quads.len())
        .min(effective.len())
        .min(TANK_MAX_QUADS);
    let mut sat = false;
    for q in 0..active_q {
        let ring = resolve_quad_ring(q, quads[q].flags, inline_q0_ring, ext_rings)?;
        sat |= quads[q].step_end_of_frame_write::<B>(ring, &effective[q], v_reads[q], x_tank_lr);
    }
    Ok(sat)
}

/// Executes Step 3 Top-of-Frame delay reads (`v_reads[0..p_q]`) ahead of `Op-T` crossbar taps
/// without mutating wet DC or VCA states before Stage III completes.
#[inline]
pub fn step_tank_top_of_frame_v_reads<B: MathBackend>(
    quads: &mut [TankQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TANK_INLINE_QUAD0_CAP],
    ext_rings: &mut [TankExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &[QuadEffectiveTargets],
) -> Result<[[f32; 4]; TANK_MAX_QUADS], VmError> {
    let active_q = p_q
        .min(quads.len())
        .min(effective.len())
        .min(TANK_MAX_QUADS);
    let mut v_reads = [[0.0_f32; 4]; TANK_MAX_QUADS];
    for q in 0..active_q {
        let ring = resolve_quad_ring(q, quads[q].flags, inline_q0_ring, ext_rings)?;
        v_reads[q] =
            quads[q].read_top_of_frame_v_read::<B>(ring, schroeders.get_mut(q), &effective[q]);
    }
    Ok(v_reads)
}

/// Executes Step 10 post-Cabinet tank output mixing (`wet_tap -> wet_dc -> ducking VCA -> dry/wet`)
/// and End-of-Frame feedback writeback across `0..p_q` active quads, returning `((y_l, y_r), sat)`.
#[inline]
pub fn step_tank_end_of_frame_with_output<B: MathBackend>(
    quads: &mut [TankQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TANK_INLINE_QUAD0_CAP],
    ext_rings: &mut [TankExtRing<'_>],
    effective: &[QuadEffectiveTargets],
    v_reads: &[[f32; 4]; TANK_MAX_QUADS],
    x_tank_lr: (f32, f32),
) -> Result<((f32, f32), bool), VmError> {
    let active_q = p_q
        .min(quads.len())
        .min(effective.len())
        .min(TANK_MAX_QUADS);
    let mut out_l = 0.0_f32;
    let mut out_r = 0.0_f32;
    let mut sat = false;
    for q in 0..active_q {
        let (ql, qr) =
            quads[q].finish_post_cabinet_output::<B>(&effective[q], v_reads[q], x_tank_lr);
        out_l += ql;
        out_r += qr;
        let ring = resolve_quad_ring(q, quads[q].flags, inline_q0_ring, ext_rings)?;
        sat |= quads[q].step_end_of_frame_write::<B>(ring, &effective[q], v_reads[q], x_tank_lr);
    }
    Ok(((out_l, out_r), sat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::arena_wire::TANK_FLAG_EXT_F32;
    use crate::math::{DC_BLOCKER_5HZ_48K_F32, DefaultMathBackend, const_abs_f32};
    use crate::types::ColdBuf;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn make_quad_wire(delays: [f32; 4], dry_mix: f32, flags: u8) -> TankQuadWire {
        let mut w = TankQuadWire::new_zeroed();
        w.delay_samples = delays.map(F32::new);
        w.g_loop = F32::new(0.8);
        w.r_dc = F32::new(0.999);
        w.damp_pole = F32::new(0.25);
        w.vca_pole = F32::new(0.999);
        w.input_gains_fp16 = [0.5, 0.5, 0.5, 0.5].map(|v| U16::new(f32_to_fp16(v)));
        w.wet_tap_l_fp16 = [0.25, 0.25, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        w.wet_tap_r_fp16 = [0.0, 0.0, 0.25, 0.25].map(|v| U16::new(f32_to_fp16(v)));
        w.sigma_knee_fp16 = U16::new(f32_to_fp16(1.0));
        w.dry_mix_fp16 = U16::new(f32_to_fp16(dry_mix));
        w.flags = flags;
        w
    }

    #[test]
    fn test_lite_inline_with_empty_ext_and_fail_closed_when_missing() {
        let w0 = make_quad_wire([53.0, 67.0, 79.0, 89.0], 0.5, 0);
        let cfg0 = QuadHydrateConfig {
            quad_idx: 0,
            fs_vm: 48_000.0,
            rho_vm: 1.0,
            declared_max_delay_48k: 89,
            ext_ring_pow2_frames: 7,
            dc_blocker_pole_5hz: DC_BLOCKER_5HZ_48K_F32,
        };
        let (q0, m0) = TankQuadHotCompact::from_wire(&w0, &cfg0).unwrap();
        let eff0 = [QuadEffectiveTargets::from_quad_and_meta(&q0, &m0)];
        let mut quads = [q0];
        let mut inline_ring = [[0_u16; 4]; TANK_INLINE_QUAD0_CAP];

        let top = step_tank_top_of_frame::<DefaultMathBackend>(
            &mut quads,
            1,
            &mut inline_ring,
            &mut [],
            &mut [],
            &eff0,
            (1.0, -0.5),
        )
        .unwrap();
        assert_eq!(top.y_lr, (0.5, -0.25));
        assert!(
            !step_tank_end_of_frame::<DefaultMathBackend>(
                &mut quads,
                1,
                &mut inline_ring,
                &mut [],
                &eff0,
                &top.v_reads,
                (1.0, -0.5),
            )
            .unwrap()
        );

        // Setting HAS_EXT without ext_rings fails closed with InsufficientBackingCapacity.
        quads[0].flags |= QUAD_FLAG_HAS_EXT;
        assert_eq!(
            step_tank_top_of_frame::<DefaultMathBackend>(
                &mut quads,
                1,
                &mut inline_ring,
                &mut [],
                &mut [],
                &eff0,
                (1.0, 0.0),
            )
            .unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
    }

    #[test]
    fn test_multiquad_slot22_q0_only_psi_slew_schroeder_and_reconfigure() {
        let mut w0 = make_quad_wire(
            [16.0, 20.0, 24.0, 28.0],
            0.75,
            TANK_FLAG_EXT_F32 | TANK_FLAG_SCHROEDER_ENABLE,
        );
        (w0.allpass_delays, w0.vca_sens_fp16) = ([13, 0, 17, 0], U16::new(f32_to_fp16(-4.0)));
        let w1 = make_quad_wire([32.0, 40.0, 48.0, 56.0], 0.25, TANK_FLAG_EXT_F32);
        let wires = [w0, w1];
        let summary = compute_tank_storage_requirements(&wires, 0.5, 56, 7);
        assert_eq!(
            summary,
            TankColdStorageSummary {
                ext_fp16_u16: 0,
                ext_f32: 128 * 4 * 2,
                schroeder_f32: 64 * 4
            }
        );

        let cfg0 = QuadHydrateConfig {
            quad_idx: 0,
            fs_vm: 96_000.0,
            rho_vm: 0.5,
            declared_max_delay_48k: 56,
            ext_ring_pow2_frames: 7,
            dc_blocker_pole_5hz: DC_BLOCKER_5HZ_48K_F32,
        };
        let cfg1 = QuadHydrateConfig {
            quad_idx: 1,
            ..cfg0
        };
        let (mut q0, mut m0) = TankQuadHotCompact::from_wire(&w0, &cfg0).unwrap();
        let (q1, m1) = TankQuadHotCompact::from_wire(&w1, &cfg1).unwrap();
        let mut quads = [q0, q1];
        let metas = [m0, m1];
        let mut effs = [
            QuadEffectiveTargets::from_quad_and_meta(&quads[0], &metas[0]),
            QuadEffectiveTargets::from_quad_and_meta(&quads[1], &metas[1]),
        ];

        let (mut f0, mut f1, mut ap_buf) =
            ([0.0_f32; 128 * 4], [0.0_f32; 128 * 4], [0.0_f32; 64 * 4]);
        let mut ext_rings = [
            TankExtRing::from_f32_buf(ColdBuf::borrowed(&mut f0), 128).unwrap(),
            TankExtRing::from_f32_buf(ColdBuf::borrowed(&mut f1), 128).unwrap(),
        ];
        let mut schroeders = [PartitionedSchroederBank::from_slice(&mut ap_buf, &w0, 0.5).unwrap()];

        let slots = QuadModSlots {
            tank_psi_delta: 2.0,
            op_t_dry_mix_mod: -0.5,
            op_t_allpass_g_delta: 0.35,
            ..Default::default()
        };
        tick_tank_3khz(
            &mut quads,
            &metas,
            2,
            &slots,
            &mut ext_rings,
            &mut schroeders,
            &mut effs,
        );
        // Givens angle slews at <= 0.25 rad/tick; slot 22 applies strictly to Quad 0 (Quad 1 untouched).
        assert!(const_abs_f32(quads[0].current_psi - 0.25) < 1.0e-6);
        assert!(const_abs_f32(effs[0].dry_mix_eff - 0.25) < 1.0e-3);
        assert!(const_abs_f32(effs[1].dry_mix_eff - 0.25) < 1.0e-3);
        assert!(const_abs_f32(schroeders[0].allpass_g()[0] - 0.35) < 1.0e-5);

        let mut inline_ring = [[0_u16; 4]; TANK_INLINE_QUAD0_CAP];
        let top = step_tank_top_of_frame::<DefaultMathBackend>(
            &mut quads,
            2,
            &mut inline_ring,
            &mut ext_rings,
            &mut schroeders,
            &effs,
            (0.5, 0.5),
        )
        .unwrap();
        assert!(
            !step_tank_end_of_frame::<DefaultMathBackend>(
                &mut quads,
                2,
                &mut inline_ring,
                &mut ext_rings,
                &effs,
                &top.v_reads,
                (0.5, 0.5),
            )
            .unwrap()
        );

        reset_tank(
            &mut quads,
            2,
            &mut inline_ring,
            &mut ext_rings,
            &mut schroeders,
        );
        q0.set_host_write_lock(true, None);
        m0 = q0
            .reconfigure_sample_rate(
                &w0,
                &QuadHydrateConfig {
                    fs_vm: 48_000.0,
                    rho_vm: 1.0,
                    ..cfg0
                },
                Some(&mut ext_rings[0]),
                None,
                Some(&mut schroeders[0]),
            )
            .unwrap();
        assert_eq!(
            (q0.flags & dnam_kernels::QUAD_FLAG_HOST_WRITE_LOCK, m0.cap),
            (dnam_kernels::QUAD_FLAG_HOST_WRITE_LOCK, 128)
        );
    }
}
