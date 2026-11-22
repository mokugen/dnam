//! `216 B` 8-byte-aligned `Op-T` transport quad active-hot state ([`TransportQuadHotCompact`]),
//! cold wire hydration/ZOH rescaling, unified `sigma_sat` & continuous `freeze_mix` holding, and
//! causal 4-lane Top-of-Frame read + End-of-Frame write execution (`ref[dsp.op-t.ring.declaration]`,
//! `ref[dsp.op-t.reader.catmull-rom]`).

mod hydrate;
mod step;

use dnam_kernels::{
    QUAD_FLAG_HOST_WRITE_LOCK, TANK_MAX_SIGMA_SAT_F32, TANK_MAX_SIGMA_SAT_FP16,
    algebraic_unit_rotor_f32, compute_half_alpha_cubic_f32, compute_op_t_pitch_down_g_loop_eff,
};

use crate::math::{DC_BLOCKER_5HZ_48K_F32, clamp_finite_f32, f32_to_fp16, fp16_to_f32};

use super::ring::{PartitionedSchroederBank, TRANSPORT_INLINE_QUAD0_CAP, TransportExtRing};

pub(crate) const DAMP_EFF_MAX: f32 = 0.999_95;
pub(crate) const MIN_SIGMA_SAT_F32: f32 = 1.0e-3;
const FULL_HOLD_MU_THRESHOLD: f32 = 1.0 - 1.0e-6;

/// Compile-time `fp16` quantization of default `2.0 / (3.0 * TANK_MAX_SIGMA_SAT_F32)`.
const DEFAULT_INV_SIGMA_23_FP16: u16 =
    f32_to_fp16((2.0_f64 / (3.0_f64 * (TANK_MAX_SIGMA_SAT_F32 as f64))) as f32);

/// Active-hot `216 B` 8-byte-aligned working-set block for one 4-lane `Op-T` transport quad
/// (`op_t_transport.md §1.1.1`).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportQuadHotCompact {
    // [0..64): 2-state Direct-Form I loop DC blockers, HF damp state, stereo wet DC blockers.
    pub dc_x_prev: [f32; 4],
    pub dc_y_prev: [f32; 4],
    pub damp_state: [f32; 4],
    pub wet_dc_x: [f32; 2],
    pub wet_dc_y: [f32; 2],
    // [64..96): effective-rate base delays and ramp spans.
    pub delay_samples: [f32; 4],
    pub ramp_spans_samples: [f32; 4],
    // [96..144): backward cursor, LFO/ramp accumulators, poles, Givens rotor, wet DC pole.
    pub write_ptr: u32,
    pub lfo_phase_q32: u32,
    pub lfo_inc_q32: u32,
    pub ramp_phase_q32: u32,
    pub ramp_inc_q32: i32,
    pub g_loop: f32,
    pub r_dc: f32,
    pub damp_pole: f32,
    pub current_psi: f32,
    pub cos_psi: f32,
    pub sin_psi: f32,
    pub wet_dc_pole: f32,
    // [144..200): inline hold offsets, Q16 phase offsets, fp16 gains.
    pub we_hold_off: [u16; 4],
    pub lfo_phase_off_q16: [u16; 4],
    pub ramp_phase_off_q16: [u16; 4],
    pub lfo_depths_fp16: [u16; 4],
    pub input_gains_fp16: [u16; 4],
    pub wet_tap_l_fp16: [u16; 4],
    pub wet_tap_r_fp16: [u16; 4],
    // [200..216): fp16 scalars, flags, explicit tail pad.
    pub window_depth_fp16: u16,
    pub freeze_mix_fp16: u16,
    pub sigma_sat_fp16: u16,
    pub inv_sigma_23_fp16: u16,
    pub half_alpha_cubic_fp16: u16,
    pub base_psi_fp16: u16,
    pub flags: u8,
    pub _pad8: u8,
    pub _pad16: u16,
}

const _: () = assert!(core::mem::size_of::<TransportQuadHotCompact>() == 216);
const _: () = assert!(core::mem::align_of::<TransportQuadHotCompact>() == 8);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, delay_samples) == 64);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, write_ptr) == 96);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, we_hold_off) == 144);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, window_depth_fp16) == 200);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, freeze_mix_fp16) == 202);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, sigma_sat_fp16) == 204);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, inv_sigma_23_fp16) == 206);
const _: () = assert!(core::mem::offset_of!(TransportQuadHotCompact, flags) == 212);

impl Default for TransportQuadHotCompact {
    fn default() -> Self {
        Self {
            dc_x_prev: [0.0; 4],
            dc_y_prev: [0.0; 4],
            damp_state: [0.0; 4],
            wet_dc_x: [0.0; 2],
            wet_dc_y: [0.0; 2],
            delay_samples: [5.0; 4],
            ramp_spans_samples: [0.0; 4],
            write_ptr: 0,
            lfo_phase_q32: 0,
            lfo_inc_q32: 0,
            ramp_phase_q32: 0,
            ramp_inc_q32: 0,
            g_loop: 0.0,
            r_dc: DC_BLOCKER_5HZ_48K_F32,
            damp_pole: 0.0,
            current_psi: 0.0,
            cos_psi: 1.0,
            sin_psi: 0.0,
            wet_dc_pole: DC_BLOCKER_5HZ_48K_F32,
            we_hold_off: [0; 4],
            lfo_phase_off_q16: [0; 4],
            ramp_phase_off_q16: [0; 4],
            lfo_depths_fp16: [0; 4],
            input_gains_fp16: [0; 4],
            wet_tap_l_fp16: [0; 4],
            wet_tap_r_fp16: [0; 4],
            window_depth_fp16: 0,
            freeze_mix_fp16: 0,
            sigma_sat_fp16: TANK_MAX_SIGMA_SAT_FP16,
            inv_sigma_23_fp16: DEFAULT_INV_SIGMA_23_FP16,
            half_alpha_cubic_fp16: f32_to_fp16(0.5),
            base_psi_fp16: 0,
            flags: 0,
            _pad8: 0,
            _pad16: 0,
        }
    }
}

/// Cold hydration parameters shared across VM initialization and sample-rate reconfiguration.
#[derive(Clone, Copy, Debug)]
pub struct QuadHydrateConfig {
    pub quad_idx: usize,
    pub fs_vm: f64,
    pub rho_vm: f64,
    pub declared_max_delay_48k: u32,
    pub ext_ring_pow2_frames: u8,
    pub dc_blocker_pole_5hz: f32,
}

/// Cold immutable ZOH-converted base metadata retained alongside [`TransportQuadHotCompact`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadColdMeta {
    pub quad_idx: u8,
    pub cap: usize,
    pub base_g_loop: f32,
    pub base_damp_pole_zoh: f32,
    pub base_lfo_rate_hz: f32,
    pub base_ramp_rate_hz: f32,
    pub base_freeze_mix: f32,
    pub base_psi: f32,
    pub base_scat_psi: f32,
    pub base_window_depth: f32,
    pub inv_fs_vm: f64,
    pub scale_inv_rho_vm: f32,
    pub base_lfo_depths: [f32; 4],
    pub base_input_gains: [f32; 4],
    pub base_allpass_g: [f32; 4],
}

impl Default for QuadColdMeta {
    #[inline]
    fn default() -> Self {
        Self {
            quad_idx: 0,
            cap: TRANSPORT_INLINE_QUAD0_CAP,
            base_g_loop: 0.0,
            base_damp_pole_zoh: 0.0,
            base_lfo_rate_hz: 0.0,
            base_ramp_rate_hz: 0.0,
            base_freeze_mix: 0.0,
            base_psi: 0.0,
            base_scat_psi: 0.0,
            base_window_depth: 0.0,
            inv_fs_vm: 1.0 / 48_000.0,
            scale_inv_rho_vm: 1.0,
            base_lfo_depths: [0.0; 4],
            base_input_gains: [0.0; 4],
            base_allpass_g: [0.0; 4],
        }
    }
}

/// Cohesive effective per-sample parameters derived from [`TransportQuadHotCompact`] and
/// `ControlBus` updates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadEffectiveTargets {
    pub cap: usize,
    pub scale_inv_rho_vm: f32,
    pub tank_delay_mod_48k: f32,
    pub delay_scale_oct: f32,
    pub delay_scale_pow2: f32,
    pub g_loop_eff: f32,
    pub sigma_sat: f32,
    pub inv_sigma_23: f32,
    pub mu_freeze: f32,
    pub half_alpha_cubic: f32,
    pub cos_psi: f32,
    pub sin_psi: f32,
    pub cos_scat: f32,
    pub sin_scat: f32,
    pub damp_pole_eff: f32,
    pub wet_scale: f32,
    pub window_depth_eff: f32,
    pub ramp_rate_hz_eff: f32,
    pub lfo_depths_eff: [f32; 4],
    pub input_gains_eff: [f32; 4],
    pub allpass_g_eff: [f32; 4],
}

impl Default for QuadEffectiveTargets {
    #[inline]
    fn default() -> Self {
        Self::from_quad_and_meta(
            &TransportQuadHotCompact::default(),
            &QuadColdMeta::default(),
        )
    }
}

impl QuadEffectiveTargets {
    /// Initializes unmodulated effective parameters directly from `(quad, meta)`.
    #[inline]
    pub fn from_quad_and_meta(quad: &TransportQuadHotCompact, meta: &QuadColdMeta) -> Self {
        let mu_freeze = if (quad.flags & QUAD_FLAG_HOST_WRITE_LOCK) != 0 {
            1.0
        } else {
            clamp_finite_f32(fp16_to_f32(quad.freeze_mix_fp16), 0.0, 1.0, 0.0)
        };
        let taper = (1.0 - mu_freeze).clamp(0.0, 1.0);
        let g_loop_eff = compute_op_t_pitch_down_g_loop_eff(
            quad.g_loop,
            quad.ramp_spans_samples,
            quad.ramp_inc_q32,
        );
        let sigma_sat = clamp_finite_f32(
            fp16_to_f32(quad.sigma_sat_fp16),
            MIN_SIGMA_SAT_F32,
            TANK_MAX_SIGMA_SAT_F32,
            TANK_MAX_SIGMA_SAT_F32,
        );
        let mut lfo_depths_eff = quad.lfo_depths_fp16.map(fp16_to_f32);
        let mut allpass_g_eff = meta.base_allpass_g;
        for k in 0..4 {
            lfo_depths_eff[k] *= taper;
            allpass_g_eff[k] *= taper;
        }
        let mu_freeze_unlocked = clamp_finite_f32(meta.base_freeze_mix, 0.0, 1.0, 0.0);
        let (cos_scat, sin_scat) =
            algebraic_unit_rotor_f32((1.0 - mu_freeze_unlocked) * meta.base_scat_psi);
        Self {
            cap: meta.cap,
            scale_inv_rho_vm: meta.scale_inv_rho_vm,
            tank_delay_mod_48k: 0.0,
            delay_scale_oct: 0.0,
            delay_scale_pow2: 1.0,
            g_loop_eff,
            sigma_sat,
            inv_sigma_23: fp16_to_f32(quad.inv_sigma_23_fp16),
            mu_freeze,
            half_alpha_cubic: compute_half_alpha_cubic_f32(g_loop_eff, mu_freeze),
            cos_psi: quad.cos_psi,
            sin_psi: quad.sin_psi,
            cos_scat,
            sin_scat,
            damp_pole_eff: quad.damp_pole,
            wet_scale: 1.0,
            window_depth_eff: fp16_to_f32(quad.window_depth_fp16),
            ramp_rate_hz_eff: meta.base_ramp_rate_hz,
            lfo_depths_eff,
            input_gains_eff: quad.input_gains_fp16.map(fp16_to_f32),
            allpass_g_eff,
        }
    }
}

/// Top-of-Frame 4-lane read outputs separating post-Schroeder `v_read` and pre-Schroeder `r_win`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuadTopRead {
    /// Post-window, post-Schroeder 4-lane vector fed to wet taps, crossbar, and loop saturator.
    pub v_read: [f32; 4],
    /// Post-window, pre-Schroeder 4-lane vector blended into passive `freeze_mix` writeback.
    pub r_win: [f32; 4],
}

/// Result emitted by [`TransportQuadHotCompact::step_top_of_frame_read`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuadStepResult {
    /// Post-window, post-Schroeder 4-lane vector fed to [`TransportQuadHotCompact::step_end_of_frame_write`].
    pub v_read: [f32; 4],
    /// Post-window, pre-Schroeder 4-lane vector fed to [`TransportQuadHotCompact::step_end_of_frame_write`].
    pub r_win: [f32; 4],
    /// Pure-wet stereo quad output `(y_{wet,q,L}, y_{wet,q,R})` after Givens rotation and wet DC.
    pub y_lr: (f32, f32),
}

/// Borrowed mutable delay ring backing passed into [`TransportQuadHotCompact`] step calls.
#[derive(Debug)]
pub enum QuadRingMut<'a, 'b> {
    /// Inline `Quad 0` `128 × fp16[4]` buffer (`1,024 B`).
    Inline(&'a mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP]),
    /// External cold [`TransportExtRing`] (`128..=524_288` frames in `fp16` or `f32`).
    Ext(&'a mut TransportExtRing<'b>),
}

impl TransportQuadHotCompact {
    /// Restores deterministic resting equilibrium while preserving `HOST_WRITE_LOCK` (bit 2).
    pub fn reset(
        &mut self,
        ext_ring: Option<&mut TransportExtRing<'_>>,
        inline_q0_ring: Option<&mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP]>,
        schroeder: Option<&mut PartitionedSchroederBank<'_>>,
    ) {
        (self.dc_x_prev, self.dc_y_prev, self.damp_state) = ([0.0; 4], [0.0; 4], [0.0; 4]);
        (self.wet_dc_x, self.wet_dc_y) = ([0.0; 2], [0.0; 2]);
        (self.write_ptr, self.lfo_phase_q32, self.ramp_phase_q32) = (0, 0, 0);
        self.we_hold_off = [0; 4];
        self.current_psi = fp16_to_f32(self.base_psi_fp16);
        (self.cos_psi, self.sin_psi) = algebraic_unit_rotor_f32(self.current_psi);

        if let Some(ext) = ext_ring {
            ext.reset();
        }
        if let Some(inline) = inline_q0_ring {
            inline.fill([0; 4]);
        }
        if let Some(ap) = schroeder {
            ap.reset();
        }
    }

    /// Returns `true` when full freeze hold is engaged via `HOST_WRITE_LOCK` (bit 2) or
    /// `mu_freeze >= 1.0 - 1.0e-6`.
    #[inline(always)]
    #[must_use]
    pub fn is_full_hold(&self, mu_freeze: f32) -> bool {
        (self.flags & QUAD_FLAG_HOST_WRITE_LOCK) != 0 || mu_freeze >= FULL_HOLD_MU_THRESHOLD
    }

    /// Sets or clears bit 2 (`HOST_WRITE_LOCK`), clearing hold offsets when leaving full hold.
    #[inline]
    pub fn set_host_write_lock(
        &mut self,
        locked: bool,
        ext_ring: Option<&mut TransportExtRing<'_>>,
        mu_freeze_eff: f32,
    ) {
        let was_locked = (self.flags & QUAD_FLAG_HOST_WRITE_LOCK) != 0;
        self.flags = if locked {
            self.flags | QUAD_FLAG_HOST_WRITE_LOCK
        } else {
            self.flags & !QUAD_FLAG_HOST_WRITE_LOCK
        };
        if was_locked && !self.is_full_hold(mu_freeze_eff) {
            self.clear_hold_offsets(ext_ring);
        }
    }

    /// Clears inline and optional external freeze-hold lag offset counters.
    #[inline(always)]
    pub(crate) fn clear_hold_offsets(&mut self, ext_ring: Option<&mut TransportExtRing<'_>>) {
        self.we_hold_off = [0; 4];
        if let Some(ext) = ext_ring {
            ext.ext_we_hold_off = [0; 4];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::types::ColdBuf;

    #[test]
    fn test_effective_targets_taper_and_host_write_lock_hold_semantics() {
        let mut q0 = TransportQuadHotCompact {
            delay_samples: [53.0, 67.0, 79.0, 89.0],
            g_loop: 0.85,
            lfo_depths_fp16: [4.0, 2.0, 0.0, 0.0].map(f32_to_fp16),
            input_gains_fp16: [1.0, 0.5, -0.5, 0.25].map(f32_to_fp16),
            freeze_mix_fp16: f32_to_fp16(0.5),
            sigma_sat_fp16: f32_to_fp16(0.25),
            ..Default::default()
        };
        let meta = QuadColdMeta {
            base_allpass_g: [0.4, -0.4, 0.0, 0.0],
            base_freeze_mix: 0.5,
            ..Default::default()
        };
        let eff_mid = QuadEffectiveTargets::from_quad_and_meta(&q0, &meta);
        assert_eq!((eff_mid.sigma_sat, eff_mid.mu_freeze), (0.25, 0.5));
        assert_eq!(eff_mid.lfo_depths_eff, [2.0, 1.0, 0.0, 0.0]);
        assert_eq!(eff_mid.allpass_g_eff, [0.2, -0.2, 0.0, 0.0]);
        assert!(!q0.is_full_hold(eff_mid.mu_freeze));
        assert!(q0.is_full_hold(1.0 - 5.0e-7));

        let mut inline_ring = [[0_u16; 4]; TRANSPORT_INLINE_QUAD0_CAP];
        for _ in 0..8 {
            let r = q0.step_top_of_frame_read::<DefaultMathBackend>(
                QuadRingMut::Inline(&mut inline_ring),
                None,
                &eff_mid,
            );
            q0.step_end_of_frame_write::<DefaultMathBackend>(
                QuadRingMut::Inline(&mut inline_ring),
                &eff_mid,
                QuadTopRead {
                    v_read: r.v_read,
                    r_win: r.r_win,
                },
                (0.25, -0.25),
            );
        }

        let wr_frozen = q0.write_ptr;
        q0.set_host_write_lock(true, None, eff_mid.mu_freeze);
        assert!(q0.is_full_hold(0.0));
        let ring_snap = inline_ring;
        let held = q0.step_top_of_frame_read::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            None,
            &eff_mid,
        );
        assert!(!q0.step_end_of_frame_write::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            &eff_mid,
            QuadTopRead {
                v_read: held.v_read,
                r_win: held.r_win,
            },
            (0.1, 0.1),
        ));
        assert_eq!((q0.write_ptr, inline_ring), (wr_frozen, ring_snap));
        assert!(q0.we_hold_off[0] > 0);

        // Unlocking while mu_freeze_eff >= 1.0 - 1e-6 keeps hold offsets; releasing clears them.
        q0.set_host_write_lock(false, None, 1.0);
        assert!(q0.we_hold_off[0] > 0);
        q0.set_host_write_lock(true, None, 0.0);
        q0.set_host_write_lock(false, None, 0.5);
        assert_eq!(q0.we_hold_off, [0; 4]);

        q0.set_host_write_lock(true, None, 0.0);
        q0.reset(None, Some(&mut inline_ring), None);
        assert_eq!(
            (q0.flags & QUAD_FLAG_HOST_WRITE_LOCK, q0.we_hold_off),
            (QUAD_FLAG_HOST_WRITE_LOCK, [0; 4])
        );
        q0.set_host_write_lock(false, None, 0.0);

        let mut fp16_buf = [0_u16; 128 * 4];
        let mut ext16 =
            TransportExtRing::from_fp16_buf(ColdBuf::borrowed(&mut fp16_buf), 128).unwrap();
        let eff_unheld = QuadEffectiveTargets {
            mu_freeze: 0.0,
            ..eff_mid
        };
        assert!(q0.step_end_of_frame_write::<DefaultMathBackend>(
            QuadRingMut::Ext(&mut ext16),
            &eff_unheld,
            QuadTopRead::default(),
            (100_000.0, 0.0),
        ));
    }
}
