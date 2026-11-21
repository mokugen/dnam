//! Multi-quad `Op-T` Transport orchestration (`Lite` 1..`Ultra` 4 orthogonal 4-lane quads),
//! `3 kHz` `ControlBus` dispatch (`slots 12..=25`), causal Top-of-Frame + End-of-Frame stereo
//! stepping, and submodules ([`quad`], [`ring`]) (`ref[dsp.op-t.ring.declaration]`,
//! `ref[dsp.op-t.window.partition-of-unity]`).

mod dispatch;
pub mod quad;
pub mod ring;

use dnam_kernels::{QUAD_FLAG_HAS_EXT, scrub_and_flush_f32_bits};

use crate::bin::arena_wire::{TRANSPORT_FLAG_SCHROEDER_ENABLE, TransportQuadWire};
use crate::math::MathBackend;
use crate::types::VmError;

pub use dispatch::{
    TransportModSlots, apply_transport_mod_slots, latch_quad_nonslew_targets, tick_quad_3khz,
    tick_transport_3khz,
};
pub use quad::{
    QuadColdMeta, QuadEffectiveTargets, QuadHydrateConfig, QuadRingMut, QuadStepResult,
    QuadTopRead, TransportQuadHotCompact,
};
#[cfg(feature = "alloc")]
pub use ring::ExtRingAllocSpec;
pub use ring::{
    PartitionedSchroederBank, TRANSPORT_ALLPASS_G_MAX, TRANSPORT_INLINE_QUAD0_CAP,
    TRANSPORT_MAX_EXT_CAP, TRANSPORT_MIN_EXT_CAP, TRANSPORT_SCHROEDER_MAX_DELAY, TransportExtRing,
    TransportExtStorage, compute_quad_ext_requirements, compute_schroeder_requirements,
    f32_to_fp16_rtz_with_sat, required_ext_elements,
};

/// Maximum number of orthogonal 4-lane transport quads across tiers (`Ultra`: `4` quads = `16` lines).
pub const TRANSPORT_MAX_QUADS: usize = 4;

/// Cumulative cold external storage element requirements across all active transport quads (`!alloc`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransportColdStorageSummary {
    /// Total `u16` elements required across `fp16` external quad delay rings.
    pub ext_fp16_u16: usize,
    /// Total `f32` elements required across `f32` external quad delay rings.
    pub ext_f32: usize,
    /// Total `f32` elements required across enabled partitioned Schroeder diffuser banks.
    pub schroeder_f32: usize,
}

/// Cumulative Top-of-Frame output across all active transport quads (`<= 4`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransportTopOfFrameResult {
    /// Combined stereo wet+dry transport output `(y_{tank,L}, y_{tank,R})`.
    pub y_lr: (f32, f32),
    /// Per-quad `(v_read, r_win)` read vectors (`0..p_q`) forwarded into [`step_transport_end_of_frame`].
    pub reads: [QuadTopRead; TRANSPORT_MAX_QUADS],
}

/// Mutable multi-quad transport backing shared between Top-of-Frame and End-of-Frame stages.
#[derive(Debug)]
pub struct TransportStageState<'a, 'b> {
    /// Active compact transport quads (`0..p_q`).
    pub quads: &'a mut [TransportQuadHotCompact],
    /// Number of active transport quads (`1..=4`).
    pub p_q: usize,
    /// Inline `fp16` delay ring backing `Quad 0` when `QUAD_FLAG_HAS_EXT` is cleared.
    pub inline_q0_ring: &'a mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP],
    /// Cold external `fp16`/`f32` delay rings (`0..p_q`).
    pub ext_rings: &'a mut [TransportExtRing<'b>],
}

/// Computes deterministic cold external delay and Schroeder element requirements for `wires[..min(4)]`.
pub fn compute_transport_storage_requirements(
    wires: &[TransportQuadWire],
    rho_vm: f64,
    declared_max_delay_48k: u32,
    ext_ring_pow2_frames: u8,
) -> TransportColdStorageSummary {
    let mut sum = TransportColdStorageSummary::default();
    for (q, wire) in wires.iter().take(TRANSPORT_MAX_QUADS).enumerate() {
        let (u16_len, f32_len) = required_ext_elements(
            wire,
            q,
            rho_vm,
            declared_max_delay_48k,
            ext_ring_pow2_frames,
        );
        sum.ext_fp16_u16 += u16_len;
        sum.ext_f32 += f32_len;
        if (wire.flags & TRANSPORT_FLAG_SCHROEDER_ENABLE) != 0 {
            let (_, _, _, sch_len) = compute_schroeder_requirements(wire, rho_vm);
            sum.schroeder_f32 += sch_len;
        }
    }
    sum
}

/// Resets all active transport quads, inline `Quad 0` ring, external rings, and Schroeder banks.
pub fn reset_transport(
    quads: &mut [TransportQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP],
    ext_rings: &mut [TransportExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
) {
    let active_q = p_q.min(quads.len()).min(TRANSPORT_MAX_QUADS);
    inline_q0_ring.fill([0; 4]);
    for q in 0..active_q {
        quads[q].reset(ext_rings.get_mut(q), None, schroeders.get_mut(q));
    }
}

#[inline(always)]
pub(crate) fn resolve_quad_ring<'a, 'b>(
    q: usize,
    flags: u8,
    inline_q0_ring: &'a mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP],
    ext_rings: &'a mut [TransportExtRing<'b>],
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

/// Executes Top-of-Frame causal readout across `0..p_q` active transport quads, summing their
/// pure-wet outputs, scaling by `wet_scale`, adding stage-level `stage_dry_mix_eff * x_tank_lr`
/// per `op_t_transport.md` Eq. 171, and returning per-quad `(v_read, r_win)` for End-of-Frame.
#[inline]
pub fn step_transport_top_of_frame<B: MathBackend>(
    state: &mut TransportStageState<'_, '_>,
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &[QuadEffectiveTargets],
    x_tank_lr: (f32, f32),
    stage_dry_mix_eff: f32,
) -> Result<TransportTopOfFrameResult, VmError> {
    let active_q = state
        .p_q
        .min(state.quads.len())
        .min(effective.len())
        .min(TRANSPORT_MAX_QUADS);
    let mut out = TransportTopOfFrameResult::default();
    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for q in 0..active_q {
        let ring = resolve_quad_ring(
            q,
            state.quads[q].flags,
            state.inline_q0_ring,
            state.ext_rings,
        )?;
        let res =
            state.quads[q].step_top_of_frame_read::<B>(ring, schroeders.get_mut(q), &effective[q]);
        sum_l += res.y_lr.0;
        sum_r += res.y_lr.1;
        out.reads[q] = QuadTopRead {
            v_read: res.v_read,
            r_win: res.r_win,
        };
    }
    let wet_scale = if active_q > 0 {
        effective[0].wet_scale
    } else {
        1.0
    };
    let (clean_xl, clean_xr) = (
        scrub_and_flush_f32_bits(x_tank_lr.0),
        scrub_and_flush_f32_bits(x_tank_lr.1),
    );
    out.y_lr = (
        stage_dry_mix_eff * clean_xl + wet_scale * sum_l,
        stage_dry_mix_eff * clean_xr + wet_scale * sum_r,
    );
    Ok(out)
}

/// Executes End-of-Frame loop feedback + delay writeback across `0..p_q` active transport quads,
/// returning `true` if any `fp16` delay ring saturated at `±65,504.0`.
#[inline]
pub fn step_transport_end_of_frame<B: MathBackend>(
    quads: &mut [TransportQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP],
    ext_rings: &mut [TransportExtRing<'_>],
    effective: &[QuadEffectiveTargets],
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    x_tank_lr: (f32, f32),
) -> Result<bool, VmError> {
    let active_q = p_q
        .min(quads.len())
        .min(effective.len())
        .min(TRANSPORT_MAX_QUADS);
    let mut sat = false;
    for q in 0..active_q {
        let ring = resolve_quad_ring(q, quads[q].flags, inline_q0_ring, ext_rings)?;
        sat |= quads[q].step_end_of_frame_write::<B>(ring, &effective[q], reads[q], x_tank_lr);
    }
    Ok(sat)
}

/// Executes Step 3 Top-of-Frame delay reads (`reads[0..p_q]`) ahead of `Op-T` crossbar taps
/// without mutating wet DC states before Stage III completes.
#[inline]
pub fn step_transport_top_of_frame_reads<B: MathBackend>(
    quads: &mut [TransportQuadHotCompact],
    p_q: usize,
    inline_q0_ring: &mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP],
    ext_rings: &mut [TransportExtRing<'_>],
    schroeders: &mut [PartitionedSchroederBank<'_>],
    effective: &[QuadEffectiveTargets],
) -> Result<[QuadTopRead; TRANSPORT_MAX_QUADS], VmError> {
    let active_q = p_q
        .min(quads.len())
        .min(effective.len())
        .min(TRANSPORT_MAX_QUADS);
    let mut reads = [QuadTopRead::default(); TRANSPORT_MAX_QUADS];
    for q in 0..active_q {
        let ring = resolve_quad_ring(q, quads[q].flags, inline_q0_ring, ext_rings)?;
        reads[q] =
            quads[q].read_top_of_frame_v_read::<B>(ring, schroeders.get_mut(q), &effective[q]);
    }
    Ok(reads)
}

/// Executes Step 10 post-Convolver transport output mixing (`wet_tap -> wet_dc -> stage dry/wet`)
/// and End-of-Frame feedback writeback across `0..p_q` active quads, returning `((y_l, y_r), sat)`
/// per `op_t_transport.md` Eq. 171.
#[inline]
pub fn step_transport_end_of_frame_with_output<B: MathBackend>(
    state: &mut TransportStageState<'_, '_>,
    effective: &[QuadEffectiveTargets],
    reads: &[QuadTopRead; TRANSPORT_MAX_QUADS],
    x_tank_lr: (f32, f32),
    stage_dry_mix_eff: f32,
) -> Result<((f32, f32), bool), VmError> {
    let active_q = state
        .p_q
        .min(state.quads.len())
        .min(effective.len())
        .min(TRANSPORT_MAX_QUADS);
    let mut sum_l = 0.0_f32;
    let mut sum_r = 0.0_f32;
    let mut sat = false;
    for q in 0..active_q {
        let (ql, qr) =
            state.quads[q].finish_post_convolver_output::<B>(&effective[q], reads[q].v_read);
        sum_l += ql;
        sum_r += qr;
        let ring = resolve_quad_ring(
            q,
            state.quads[q].flags,
            state.inline_q0_ring,
            state.ext_rings,
        )?;
        sat |=
            state.quads[q].step_end_of_frame_write::<B>(ring, &effective[q], reads[q], x_tank_lr);
    }
    let wet_scale = if active_q > 0 {
        effective[0].wet_scale
    } else {
        1.0
    };
    let (clean_xl, clean_xr) = (
        scrub_and_flush_f32_bits(x_tank_lr.0),
        scrub_and_flush_f32_bits(x_tank_lr.1),
    );
    Ok((
        (
            stage_dry_mix_eff * clean_xl + wet_scale * sum_l,
            stage_dry_mix_eff * clean_xr + wet_scale * sum_r,
        ),
        sat,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::arena_wire::TRANSPORT_FLAG_EXT_F32;
    use crate::math::{DC_BLOCKER_5HZ_48K_F32, DefaultMathBackend, const_abs_f32, f32_to_fp16};
    use crate::types::ColdBuf;
    use dnam_kernels::{QUAD_FLAG_HOST_WRITE_LOCK, TANK_MAX_SIGMA_SAT_FP16};
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn make_quad_wire(delays: [f32; 4], freeze_mix: f32, flags: u8) -> TransportQuadWire {
        let mut w = TransportQuadWire::new_zeroed();
        w.delay_samples = delays.map(F32::new);
        w.g_loop = F32::new(0.8);
        w.r_dc = F32::new(0.999);
        w.damp_pole = F32::new(0.25);
        w.input_gains_fp16 = [0.5, 0.5, 0.5, 0.5].map(|v| U16::new(f32_to_fp16(v)));
        w.wet_tap_l_fp16 = [0.25, 0.25, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        w.wet_tap_r_fp16 = [0.0, 0.0, 0.25, 0.25].map(|v| U16::new(f32_to_fp16(v)));
        w.sigma_knee_fp16 = U16::new(TANK_MAX_SIGMA_SAT_FP16);
        w.freeze_mix_fp16 = U16::new(f32_to_fp16(freeze_mix));
        w.flags = flags;
        w
    }

    /// verify[dsp.op-t.ring.admission]
    /// verify[dsp.op-t.ring.capacity]
    #[test]
    fn test_lite_inline_with_empty_ext_and_fail_closed_when_missing() {
        let w0 = make_quad_wire([53.0, 67.0, 79.0, 89.0], 0.0, 0);
        let cfg0 = QuadHydrateConfig {
            quad_idx: 0,
            fs_vm: 48_000.0,
            rho_vm: 1.0,
            declared_max_delay_48k: 89,
            ext_ring_pow2_frames: 7,
            dc_blocker_pole_5hz: DC_BLOCKER_5HZ_48K_F32,
        };
        let (q0, m0) = TransportQuadHotCompact::from_wire(&w0, &cfg0).unwrap();
        let eff0 = [QuadEffectiveTargets::from_quad_and_meta(&q0, &m0)];
        let mut quads = [q0];
        let mut inline_ring = [[0_u16; 4]; TRANSPORT_INLINE_QUAD0_CAP];

        let top = {
            let mut state = TransportStageState {
                quads: &mut quads,
                p_q: 1,
                inline_q0_ring: &mut inline_ring,
                ext_rings: &mut [],
            };
            step_transport_top_of_frame::<DefaultMathBackend>(
                &mut state,
                &mut [],
                &eff0,
                (1.0, -0.5),
                0.5,
            )
            .unwrap()
        };
        assert_eq!(top.y_lr, (0.5, -0.25));
        assert!(
            !step_transport_end_of_frame::<DefaultMathBackend>(
                &mut quads,
                1,
                &mut inline_ring,
                &mut [],
                &eff0,
                &top.reads,
                (1.0, -0.5),
            )
            .unwrap()
        );

        quads[0].flags |= QUAD_FLAG_HAS_EXT;
        let mut state = TransportStageState {
            quads: &mut quads,
            p_q: 1,
            inline_q0_ring: &mut inline_ring,
            ext_rings: &mut [],
        };
        assert_eq!(
            step_transport_top_of_frame::<DefaultMathBackend>(
                &mut state,
                &mut [],
                &eff0,
                (1.0, 0.0),
                0.5,
            )
            .unwrap_err(),
            VmError::InsufficientBackingCapacity
        );
    }

    /// verify[dsp.op-t.ring.branchless-wrap]
    /// verify[dsp.tank.captured-freeze]
    /// verify[dsp.tank.feedback-loop]
    #[test]
    fn test_multiquad_stage_dry_mix_psi_slew_freeze_release_and_reconfigure() {
        let mut w0 = make_quad_wire(
            [16.0, 20.0, 24.0, 28.0],
            1.0,
            TRANSPORT_FLAG_EXT_F32 | TRANSPORT_FLAG_SCHROEDER_ENABLE,
        );
        w0.allpass_delays = [13, 0, 17, 0];
        w0.lfo_rate_hz = F32::new(2.0);
        w0.lfo_depths_fp16 = [4.0, 4.0, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        let w1 = make_quad_wire([32.0, 40.0, 48.0, 56.0], 1.0, TRANSPORT_FLAG_EXT_F32);
        let summary = compute_transport_storage_requirements(&[w0, w1], 0.5, 56, 7);
        assert_eq!(
            summary,
            TransportColdStorageSummary {
                ext_fp16_u16: 0,
                ext_f32: 128 * 4 * 2,
                schroeder_f32: 64 * 4,
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
        let (mut q0, mut m0) = TransportQuadHotCompact::from_wire(&w0, &cfg0).unwrap();
        let (q1, m1) = TransportQuadHotCompact::from_wire(&w1, &cfg1).unwrap();
        let (mut quads, metas) = ([q0, q1], [m0, m1]);
        let mut effs = [
            QuadEffectiveTargets::from_quad_and_meta(&quads[0], &metas[0]),
            QuadEffectiveTargets::from_quad_and_meta(&quads[1], &metas[1]),
        ];

        let (mut f0, mut f1, mut ap_buf) =
            ([0.0_f32; 128 * 4], [0.0_f32; 128 * 4], [0.0_f32; 64 * 4]);
        let mut ext_rings = [
            TransportExtRing::from_f32_buf(ColdBuf::borrowed(&mut f0), 128).unwrap(),
            TransportExtRing::from_f32_buf(ColdBuf::borrowed(&mut f1), 128).unwrap(),
        ];
        let mut schroeders = [PartitionedSchroederBank::from_slice(&mut ap_buf, &w0, 0.5).unwrap()];
        let mut inline_ring = [[0_u16; 4]; TRANSPORT_INLINE_QUAD0_CAP];

        let held_reads = step_transport_top_of_frame_reads::<DefaultMathBackend>(
            &mut quads,
            2,
            &mut inline_ring,
            &mut ext_rings,
            &mut schroeders,
            &effs,
        )
        .unwrap();
        assert!(ext_rings[0].ext_we_hold_off[0] > 0);
        let ((out_l, out_r), sat) = {
            let mut state = TransportStageState {
                quads: &mut quads,
                p_q: 2,
                inline_q0_ring: &mut inline_ring,
                ext_rings: &mut ext_rings,
            };
            step_transport_end_of_frame_with_output::<DefaultMathBackend>(
                &mut state,
                &effs,
                &held_reads,
                (0.5, -0.25),
                0.75,
            )
            .unwrap()
        };
        assert!(
            !sat && const_abs_f32(out_l - 0.375) < 1.0e-6 && const_abs_f32(out_r + 0.1875) < 1.0e-6
        );

        let mod_slots = TransportModSlots {
            crossbar_wet_mod: 0.5,
            op_t_freeze_mix_delta: -0.5,
            op_t_allpass_g_delta: 0.4,
            ..Default::default()
        };
        apply_transport_mod_slots(
            &mut quads,
            &metas,
            2,
            &mod_slots,
            &mut ext_rings,
            &mut schroeders,
            &mut effs,
        );
        assert_eq!(ext_rings[0].ext_we_hold_off, [0; 4]);
        assert!(const_abs_f32(effs[0].wet_scale - 1.5) < 1.0e-6);
        assert!(const_abs_f32(schroeders[0].allpass_g()[0] - 0.2) < 1.0e-5);

        let slew_slots = TransportModSlots {
            tank_psi_delta: 2.0,
            ..mod_slots
        };
        tick_transport_3khz(
            &mut quads,
            &metas,
            2,
            &slew_slots,
            &mut ext_rings,
            &mut schroeders,
            &mut effs,
        );
        assert!(const_abs_f32(quads[0].current_psi - 0.25) < 1.0e-6);

        reset_transport(
            &mut quads,
            2,
            &mut inline_ring,
            &mut ext_rings,
            &mut schroeders,
        );
        q0.set_host_write_lock(true, None, effs[0].mu_freeze);
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
            (q0.flags & QUAD_FLAG_HOST_WRITE_LOCK, m0.cap),
            (QUAD_FLAG_HOST_WRITE_LOCK, 128)
        );
    }
}
