//! Causal Top-of-Frame read, End-of-Frame feedback/write, and dual-latch hold stepping for
//! [`super::TankQuadHotCompact`] (`ref[dsp.op-t.reader.catmull-rom]`).

use dnam_kernels::{
    OpTFeedbackParams, OpTFeedbackState, Q32_INV_2POW32_F32, QUAD_FLAG_SCHROEDER_ENABLE,
    QUAD_WE_LOCK_MASK, blend_catmull_rom_modulated_f32, compute_op_t_mu_win_f32,
    eval_op_t_diagonal_window_vec4, q32_smooth_sin_f32, scrub_and_flush_f32_bits,
    step_op_t_ducking_vca_f32, step_op_t_feedback_vec4, step_op_t_wet_dc_stereo_f32,
    step_we_hold_lag, step_we_hold_lag_u16,
};

use crate::math::{MathBackend, SimdVec4, clamp_finite_f32, const_abs_f32, floor_f32, fp16_to_f32};

use super::super::ring::{
    PartitionedSchroederBank, TANK_INLINE_QUAD0_CAP, TANK_MAX_EXT_CAP, TANK_MIN_EXT_CAP,
    f32_to_fp16_rtz_with_sat,
};
use super::{QuadEffectiveTargets, QuadRingMut, QuadStepResult, TankQuadHotCompact};

impl QuadRingMut<'_, '_> {
    #[inline(always)]
    fn cap(&self) -> usize {
        match self {
            Self::Inline(_) => TANK_INLINE_QUAD0_CAP,
            Self::Ext(ext) => ext.cap.clamp(TANK_MIN_EXT_CAP, TANK_MAX_EXT_CAP),
        }
    }

    #[inline(always)]
    fn read_tap(&self, idx: usize, lane: usize) -> f32 {
        match self {
            Self::Inline(ring) => fp16_to_f32(ring[idx & (TANK_INLINE_QUAD0_CAP - 1)][lane & 3]),
            Self::Ext(ext) => ext.read_lane_tap(idx, lane),
        }
    }

    #[inline(always)]
    fn read_catmull_rom_4<B: MathBackend>(&self, wr: u32, d_int: usize, lane: usize) -> [f32; 4] {
        match self {
            Self::Inline(ring) => {
                let mask = TANK_INLINE_QUAD0_CAP - 1;
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
                let idx = (wr as usize) & (TANK_INLINE_QUAD0_CAP - 1);
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

impl TankQuadHotCompact {
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
    ) -> B::Vec4 {
        let cap = ring.cap();
        let (lfo_base, ramp_base) = (self.lfo_phase_q32, self.ramp_phase_q32);
        self.lfo_phase_q32 = lfo_base.wrapping_add(self.lfo_inc_q32);
        self.ramp_phase_q32 = ramp_base.wrapping_add_signed(self.ramp_inc_q32);

        let modulated = eff.lfo_depths_eff.iter().any(|&d| d > 0.0)
            || self.ramp_spans_samples.iter().any(|&s| s > 0.0)
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
                + self.ramp_spans_samples[k] * (phi_ramp - 0.5)
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
                self.ramp_spans_samples[k],
                eff.ramp_rate_hz_eff,
            );
        }
        let win = eval_op_t_diagonal_window_vec4::<B::Vec4>(ramp_phs, B::Vec4::load(&mu_win));
        let r_win = B::Vec4::load(&r_raw).mul(win);
        if (self.flags & QUAD_FLAG_SCHROEDER_ENABLE) != 0 {
            if let Some(ap) = schroeder {
                ap.step_vec4::<B>(r_win)
            } else {
                r_win
            }
        } else {
            r_win
        }
    }

    /// Top-of-Frame causal delay read + diagonal window + Schroeder (`v_read` for Step 3/4).
    #[inline(always)]
    pub(crate) fn read_top_of_frame_v_read<B: MathBackend>(
        &mut self,
        mut ring: QuadRingMut<'_, '_>,
        schroeder: Option<&mut PartitionedSchroederBank<'_>>,
        eff: &QuadEffectiveTargets,
    ) -> [f32; 4] {
        let v_read_vec = if (self.flags & QUAD_WE_LOCK_MASK) != 0 {
            self.read_held_v_read::<B>(&mut ring)
        } else {
            self.read_active_v_read::<B>(&ring, schroeder, eff)
        };
        v_read_vec.to_array()
    }

    /// Post-Cabinet wet tap projection + wet DC + Givens + ducking VCA + dry/wet blend (Step 10).
    #[inline(always)]
    pub(crate) fn finish_post_cabinet_output<B: MathBackend>(
        &mut self,
        eff: &QuadEffectiveTargets,
        v_read: [f32; 4],
        x_tank_lr: (f32, f32),
    ) -> (f32, f32) {
        let v_read_vec = B::Vec4::load(&v_read);
        let (y_raw_l, y_raw_r) = v_read_vec
            .mul(B::fp16x4_to_vec4(&self.wet_tap_l_fp16))
            .horizontal_sum_pair(v_read_vec.mul(B::fp16x4_to_vec4(&self.wet_tap_r_fp16)));
        let y_wdc = step_op_t_wet_dc_stereo_f32(
            (y_raw_l, y_raw_r),
            eff.cos_psi,
            eff.sin_psi,
            self.wet_dc_pole,
            &mut self.wet_dc_x,
            &mut self.wet_dc_y,
        );
        let (y_duck_l, y_duck_r) = step_op_t_ducking_vca_f32(
            y_wdc,
            x_tank_lr,
            eff.vca_pole,
            eff.vca_sens,
            eff.attack_cap_vca,
            &mut self.vca_env,
        );
        let (xl, xr) = (
            scrub_and_flush_f32_bits(x_tank_lr.0),
            scrub_and_flush_f32_bits(x_tank_lr.1),
        );
        (
            eff.dry_mix_eff * xl + eff.wet_scale * y_duck_l,
            eff.dry_mix_eff * xr + eff.wet_scale * y_duck_r,
        )
    }

    /// Top-of-Frame causal read + diagonal window + Schroeder + stereo wet Givens + 5 Hz wet DC +
    /// Family 4 ducking VCA + wet/dry blend.
    #[inline(always)]
    pub fn step_top_of_frame_read<B: MathBackend>(
        &mut self,
        ring: QuadRingMut<'_, '_>,
        schroeder: Option<&mut PartitionedSchroederBank<'_>>,
        eff: &QuadEffectiveTargets,
        x_tank_lr: (f32, f32),
    ) -> QuadStepResult {
        let v_read = self.read_top_of_frame_v_read::<B>(ring, schroeder, eff);
        let y_lr = self.finish_post_cabinet_output::<B>(eff, v_read, x_tank_lr);
        QuadStepResult { v_read, y_lr }
    }

    /// End-of-Frame loop feedback, soft-knee, cubic saturator, Givens + Householder mixing, HF
    /// damping, loop DC blocker, and write-before-decrement. Returns `true` on `fp16` saturation.
    #[inline(always)]
    pub fn step_end_of_frame_write<B: MathBackend>(
        &mut self,
        mut ring: QuadRingMut<'_, '_>,
        eff: &QuadEffectiveTargets,
        v_read: [f32; 4],
        x_tank_lr: (f32, f32),
    ) -> bool {
        if (self.flags & QUAD_WE_LOCK_MASK) != 0 {
            return false;
        }
        let params = OpTFeedbackParams {
            g_inj: B::Vec4::load(&eff.input_gains_eff),
            g_loop_eff: eff.g_loop_eff,
            inv_sigma: eff.inv_sigma,
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
        let y_clamped =
            step_op_t_feedback_vec4(B::Vec4::load(&v_read), x_tank_lr, &params, &mut fb_state);
        fb_state.damp_state.store(&mut self.damp_state);
        fb_state.dc_x_prev.store(&mut self.dc_x_prev);
        fb_state.dc_y_prev.store(&mut self.dc_y_prev);

        let sat = ring.write_frame(self.write_ptr, self.dc_y_prev, y_clamped.to_array());
        self.write_ptr = self.write_ptr.wrapping_sub(1) & ((ring.cap() - 1) as u32);
        sat
    }
}
