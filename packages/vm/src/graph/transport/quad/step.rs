//! Causal Top-of-Frame read, End-of-Frame feedback/write, and continuous `freeze_mix` /
//! host write-lock hold stepping for [`super::TransportQuadHotCompact`]
//! (`ref[dsp.op-t.reader.catmull-rom]`).

use dnam_kernels::{
    OpTFeedbackParams, OpTFeedbackState, Q32_INV_2POW32_F32, QUAD_FLAG_SCHROEDER_ENABLE,
    blend_catmull_rom_modulated_f32, compute_op_t_mu_win_f32, eval_op_t_diagonal_window_vec4,
    q32_smooth_sin_f32, scrub_and_flush_f32_bits, step_op_t_feedback_vec4,
    step_op_t_wet_dc_stereo_f32, step_we_hold_lag, step_we_hold_lag_u16,
};

use crate::math::{MathBackend, SimdVec4, clamp_finite_f32, const_abs_f32, floor_f32, fp16_to_f32};

use super::super::ring::{
    PartitionedSchroederBank, TRANSPORT_INLINE_QUAD0_CAP, TRANSPORT_MAX_EXT_CAP,
    TRANSPORT_MIN_EXT_CAP, f32_to_fp16_rtz_with_sat,
};
use super::{
    QuadEffectiveTargets, QuadRingMut, QuadStepResult, QuadTopRead, TransportQuadHotCompact,
};

impl QuadRingMut<'_, '_> {
    #[inline(always)]
    fn cap(&self) -> usize {
        match self {
            Self::Inline(_) => TRANSPORT_INLINE_QUAD0_CAP,
            Self::Ext(ext) => ext.cap.clamp(TRANSPORT_MIN_EXT_CAP, TRANSPORT_MAX_EXT_CAP),
        }
    }

    #[inline(always)]
    fn read_tap(&self, idx: usize, lane: usize) -> f32 {
        match self {
            Self::Inline(ring) => {
                fp16_to_f32(ring[idx & (TRANSPORT_INLINE_QUAD0_CAP - 1)][lane & 3])
            }
            Self::Ext(ext) => ext.read_lane_tap(idx, lane),
        }
    }

    #[inline(always)]
    fn read_catmull_rom_4<B: MathBackend>(&self, wr: u32, d_int: usize, lane: usize) -> [f32; 4] {
        match self {
            Self::Inline(ring) => {
                let mask = TRANSPORT_INLINE_QUAD0_CAP - 1;
                let (base, l) = ((wr as usize).wrapping_add(d_int), lane & 3);
                B::fp16x4_to_f32x4(&[
                    ring[base.wrapping_sub(1) & mask][l],
                    ring[base & mask][l],
                    ring[base.wrapping_add(1) & mask][l],
                    ring[base.wrapping_add(2) & mask][l],
                ])
            }
            Self::Ext(ext) => ext.read_catmull_rom_4_simd::<B>(wr, d_int, lane),
        }
    }

    #[inline(always)]
    fn write_frame(&mut self, wr: u32, raw_vals: [f32; 4], clamped_vals: [f32; 4]) -> bool {
        match self {
            Self::Inline(ring) => {
                let idx = (wr as usize) & (TRANSPORT_INLINE_QUAD0_CAP - 1);
                let mut sat = false;
                for k in 0..4 {
                    let (h, did_sat) =
                        f32_to_fp16_rtz_with_sat(scrub_and_flush_f32_bits(raw_vals[k]));
                    ring[idx][k] = h;
                    sat |= did_sat;
                }
                sat
            }
            Self::Ext(ext) => {
                if ext.is_f32_storage() {
                    ext.write_frame(wr, clamped_vals)
                } else {
                    ext.write_frame(wr, raw_vals)
                }
            }
        }
    }
}

impl TransportQuadHotCompact {
    #[inline(always)]
    fn read_held_v_read<B: MathBackend>(&mut self, ring: &mut QuadRingMut<'_, '_>) -> B::Vec4 {
        let cap = ring.cap();
        let mut r_frozen = [0.0_f32; 4];
        for k in 0..4 {
            let d_round = ((self.delay_samples[k] + 0.5).clamp(5.0, (cap - 2) as f32)) as u32;
            let lag = match ring {
                QuadRingMut::Inline(_) => step_we_hold_lag_u16(&mut self.we_hold_off[k], d_round),
                QuadRingMut::Ext(ext) => step_we_hold_lag(&mut ext.ext_we_hold_off[k], d_round),
            };
            r_frozen[k] = ring.read_tap((self.write_ptr as usize).wrapping_add(lag as usize), k);
        }
        B::Vec4::load(&r_frozen)
    }

    #[inline(always)]
    fn read_active_v_read<B: MathBackend>(
        &mut self,
        ring: &QuadRingMut<'_, '_>,
        schroeder: Option<&mut PartitionedSchroederBank<'_>>,
        eff: &QuadEffectiveTargets,
    ) -> (B::Vec4, B::Vec4) {
        let cap = ring.cap();
        let (lfo_base, ramp_base) = (self.lfo_phase_q32, self.ramp_phase_q32);
        self.lfo_phase_q32 = lfo_base.wrapping_add(self.lfo_inc_q32);
        self.ramp_phase_q32 = ramp_base.wrapping_add_signed(self.ramp_inc_q32);

        let taper = clamp_finite_f32(1.0 - eff.mu_freeze, 0.0, 1.0, 1.0);
        let tapered_spans = self.ramp_spans_samples.map(|s| s * taper);
        let modulated = eff.lfo_depths_eff.iter().any(|&d| d > 0.0)
            || tapered_spans.iter().any(|&s| s > 0.0)
            || const_abs_f32(eff.delay_scale_oct) > 1.0e-6
            || const_abs_f32(eff.tank_delay_mod_48k) > 1.0e-6;
        let max_tau = (cap - if modulated { 3 } else { 2 }) as f32;

        let (mut r_raw, mut ramp_phs, mut mu_win) = ([0.0_f32; 4], [0_u32; 4], [0.0_f32; 4]);
        for k in 0..4 {
            let lfo_ph = lfo_base.wrapping_add(u32::from(self.lfo_phase_off_q16[k]) << 16);
            let ramp_ph = ramp_base.wrapping_add(u32::from(self.ramp_phase_off_q16[k]) << 16);
            ramp_phs[k] = ramp_ph;
            let phi_ramp = (ramp_ph as f32) * Q32_INV_2POW32_F32;
            let tau_raw = self.delay_samples[k] * eff.delay_scale_pow2
                + eff.lfo_depths_eff[k] * q32_smooth_sin_f32(lfo_ph)
                + tapered_spans[k] * (phi_ramp - 0.5)
                + eff.tank_delay_mod_48k * eff.scale_inv_rho_vm;
            let tau = clamp_finite_f32(tau_raw, 5.0, max_tau, 5.0);
            let d_floor = floor_f32(tau);
            let d_int = (d_floor as usize).clamp(5, cap - 2);
            let taps = ring.read_catmull_rom_4::<B>(self.write_ptr, d_int, k);
            r_raw[k] = blend_catmull_rom_modulated_f32(
                taps[0],
                taps[1],
                taps[2],
                taps[3],
                (tau - d_floor).clamp(0.0, 1.0),
                eff.half_alpha_cubic,
            );
            mu_win[k] = compute_op_t_mu_win_f32(
                eff.window_depth_eff,
                tapered_spans[k],
                eff.ramp_rate_hz_eff,
            );
        }
        let win = eval_op_t_diagonal_window_vec4::<B::Vec4>(ramp_phs, B::Vec4::load(&mu_win));
        let r_win = B::Vec4::load(&r_raw).mul(win);
        let v_read = if (self.flags & QUAD_FLAG_SCHROEDER_ENABLE) != 0
            && !self.is_full_hold(eff.mu_freeze)
        {
            if let Some(ap) = schroeder {
                ap.step_vec4::<B>(r_win)
            } else {
                r_win
            }
        } else {
            r_win
        };
        (v_read, r_win)
    }

    /// Top-of-Frame causal delay read + diagonal window + Schroeder (`(v_read, r_win)`).
    #[inline(always)]
    pub(crate) fn read_top_of_frame_v_read<B: MathBackend>(
        &mut self,
        mut ring: QuadRingMut<'_, '_>,
        schroeder: Option<&mut PartitionedSchroederBank<'_>>,
        eff: &QuadEffectiveTargets,
    ) -> QuadTopRead {
        if self.is_full_hold(eff.mu_freeze) {
            let held_arr = self.read_held_v_read::<B>(&mut ring).to_array();
            QuadTopRead {
                v_read: held_arr,
                r_win: held_arr,
            }
        } else {
            let (v_read, r_win) = self.read_active_v_read::<B>(&ring, schroeder, eff);
            QuadTopRead {
                v_read: v_read.to_array(),
                r_win: r_win.to_array(),
            }
        }
    }

    /// Post-Convolver pure-wet tap projection + Givens rotation + 5 Hz wet DC blocker (Step 10).
    #[inline(always)]
    pub(crate) fn finish_post_convolver_output<B: MathBackend>(
        &mut self,
        eff: &QuadEffectiveTargets,
        v_read: [f32; 4],
    ) -> (f32, f32) {
        let v_read_vec = B::Vec4::load(&v_read);
        let (y_raw_l, y_raw_r) = v_read_vec
            .mul(B::fp16x4_to_vec4(&self.wet_tap_l_fp16))
            .horizontal_sum_pair(v_read_vec.mul(B::fp16x4_to_vec4(&self.wet_tap_r_fp16)));
        step_op_t_wet_dc_stereo_f32(
            (y_raw_l, y_raw_r),
            eff.cos_psi,
            eff.sin_psi,
            self.wet_dc_pole,
            &mut self.wet_dc_x,
            &mut self.wet_dc_y,
        )
    }

    /// Top-of-Frame causal read + diagonal window + Schroeder + pure-wet stereo Givens & 5 Hz DC.
    #[inline(always)]
    pub fn step_top_of_frame_read<B: MathBackend>(
        &mut self,
        ring: QuadRingMut<'_, '_>,
        schroeder: Option<&mut PartitionedSchroederBank<'_>>,
        eff: &QuadEffectiveTargets,
    ) -> QuadStepResult {
        let read = self.read_top_of_frame_v_read::<B>(ring, schroeder, eff);
        let y_lr = self.finish_post_convolver_output::<B>(eff, read.v_read);
        QuadStepResult {
            v_read: read.v_read,
            r_win: read.r_win,
            y_lr,
        }
    }

    /// End-of-Frame loop feedback, unified `sigma_sat` cubic saturator, Givens + Householder
    /// mixing, HF damping, loop DC blocker, continuous `mu_freeze` writeback blend, and
    /// write-before-decrement. Returns `true` on `fp16` saturation.
    #[inline(always)]
    pub fn step_end_of_frame_write<B: MathBackend>(
        &mut self,
        mut ring: QuadRingMut<'_, '_>,
        eff: &QuadEffectiveTargets,
        read: QuadTopRead,
        x_tank_lr: (f32, f32),
    ) -> bool {
        if self.is_full_hold(eff.mu_freeze) {
            return false;
        }
        let mu = clamp_finite_f32(eff.mu_freeze, 0.0, 1.0, 0.0);
        let params = OpTFeedbackParams {
            g_inj: B::Vec4::load(&eff.input_gains_eff),
            g_loop_eff: eff.g_loop_eff,
            sigma_sat: eff.sigma_sat,
            inv_sigma_23: eff.inv_sigma_23,
            mu_freeze: mu,
            cos_psi: eff.cos_psi,
            sin_psi: eff.sin_psi,
            damp_pole_eff: eff.damp_pole_eff,
            r_dc: self.r_dc,
        };
        let mut fb_state = OpTFeedbackState {
            damp_state: B::Vec4::load(&self.damp_state),
            dc_x_prev: B::Vec4::load(&self.dc_x_prev),
            dc_y_prev: B::Vec4::load(&self.dc_y_prev),
        };
        let v_read_vec = B::Vec4::load(&read.v_read);
        let r_win_vec = B::Vec4::load(&read.r_win);
        let y_clamped =
            step_op_t_feedback_vec4(v_read_vec, r_win_vec, x_tank_lr, &params, &mut fb_state);
        fb_state.damp_state.store(&mut self.damp_state);
        fb_state.dc_x_prev.store(&mut self.dc_x_prev);
        fb_state.dc_y_prev.store(&mut self.dc_y_prev);

        let raw_w = if mu <= 0.0 {
            self.dc_y_prev
        } else {
            B::Vec4::splat(1.0 - mu)
                .mul(B::Vec4::load(&self.dc_y_prev))
                .fma(B::Vec4::splat(mu), r_win_vec)
                .to_array()
        };
        let sat = ring.write_frame(self.write_ptr, raw_w, y_clamped.to_array());
        self.write_ptr = self.write_ptr.wrapping_sub(1) & ((ring.cap() - 1) as u32);
        sat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bin::arena_wire::{TRANSPORT_FLAG_SCHROEDER_ENABLE, TransportQuadWire};
    use crate::math::{DefaultMathBackend, const_abs_f32, f32_to_fp16};
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    #[test]
    fn test_tapered_ramp_spans_mu_freeze_blend_and_full_hold_freeze() {
        let mut inline_ring = [[0_u16; 4]; TRANSPORT_INLINE_QUAD0_CAP];
        inline_ring[120] = [f32_to_fp16(0.5); 4];
        inline_ring[110] = [f32_to_fp16(1.0); 4];

        let mut ap_wire = TransportQuadWire::new_zeroed();
        ap_wire.allpass_delays = [7, 0, 0, 0];
        ap_wire.allpass_g_fp16 = [0.5, 0.0, 0.0, 0.0].map(|v| U16::new(f32_to_fp16(v)));
        ap_wire.flags = TRANSPORT_FLAG_SCHROEDER_ENABLE;
        let mut ap_buf = [0.0_f32; 32];
        let mut ap = PartitionedSchroederBank::from_slice(&mut ap_buf, &ap_wire, 1.0).unwrap();

        let mut quad = TransportQuadHotCompact {
            delay_samples: [10.0; 4],
            ramp_spans_samples: [16.0; 4],
            ramp_inc_q32: 0x0100_0000,
            lfo_inc_q32: 0x0200_0000,
            flags: QUAD_FLAG_SCHROEDER_ENABLE,
            wet_tap_l_fp16: [f32_to_fp16(0.25); 4],
            wet_tap_r_fp16: [f32_to_fp16(-0.25); 4],
            ..Default::default()
        };
        let mut eff = QuadEffectiveTargets {
            mu_freeze: 0.5,
            half_alpha_cubic: 0.25,
            allpass_g_eff: [0.25, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        ap.set_allpass_g(eff.allpass_g_eff);

        let top = quad.step_top_of_frame_read::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            Some(&mut ap),
            &eff,
        );
        assert!(top.v_read[0].is_finite());
        assert_eq!(quad.ramp_spans_samples, [16.0; 4]);
        // Pure-wet output has zero dry leak when x_tank_lr is not added on quad.
        assert!(const_abs_f32(top.y_lr.0 + top.y_lr.1) < 1.0e-5);

        // Near-unity mu_freeze engages full hold: LFO/ramp accumulators and ring freeze at 0-ULP.
        eff.mu_freeze = 1.0 - 1.0e-7;
        let (lfo_snap, ramp_snap, wr_snap) =
            (quad.lfo_phase_q32, quad.ramp_phase_q32, quad.write_ptr);
        let ring_snap = inline_ring;
        let held = quad.step_top_of_frame_read::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            Some(&mut ap),
            &eff,
        );
        assert_eq!(held.v_read, held.r_win);
        assert!(!quad.step_end_of_frame_write::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            &eff,
            QuadTopRead {
                v_read: held.v_read,
                r_win: held.r_win,
            },
            (1.0, -1.0),
        ));
        assert_eq!(
            (quad.lfo_phase_q32, quad.ramp_phase_q32, quad.write_ptr),
            (lfo_snap, ramp_snap, wr_snap)
        );
        assert_eq!(inline_ring, ring_snap);

        // Blended mu_freeze still reports fp16 saturation when unclamped write exceeds 65,504.
        eff.mu_freeze = 0.25;
        eff.input_gains_eff = [1.0; 4];
        let sat = quad.step_end_of_frame_write::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            &eff,
            QuadTopRead {
                v_read: [0.0; 4],
                r_win: [0.0; 4],
            },
            (100_000.0, 0.0),
        );
        assert!(sat);
        let _ = F32::new(0.0);
    }
}
