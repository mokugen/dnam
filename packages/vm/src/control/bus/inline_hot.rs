//! Always-inlined `ControlBus` frame and `3 kHz` boundary helpers shared by standard and
//! `#[target_feature]` block dispatch (`#![no_std]`, `#![forbid(unsafe_code)]`, `0-ULP` parity).

#![forbid(unsafe_code)]

use crate::math::MathBackend;

use super::ControlBus;
use super::dispatch::{
    DispatchInputs, DispatchMatrices, STAGE_TARGET_SLOTS, advance_micro_ramp_frame,
    evaluate_w_ctrl_32_inline, prepare_micro_ramp_deltas, step_ou_drift,
};
use super::tick::{step_lfos_and_skewed_q2, update_env_states};
use crate::types::Tier;

impl<B: MathBackend> ControlBus<'_, B> {
    /// Always-inlined `3 kHz` control-boundary update for `#[target_feature]` block dispatch.
    #[inline(always)]
    pub(crate) fn tick_control_boundary_inline(&mut self, sensory_env: Option<[f32; 4]>) {
        if !self.should_tick() {
            return;
        }
        self.clamp_and_eval_static_if_dirty();
        let q2 = step_lfos_and_skewed_q2(
            &mut self.lfo_phases_q192,
            &self.lfo_dphi_host_bits,
            &mut self.lfo_phasors,
            &self.lfo_skew,
        );
        update_env_states(&mut self.env_states, sensory_env);
        evaluate_w_ctrl_32_inline::<B>(
            DispatchInputs {
                q16_head: &self.z_head,
                q16_basis: &self.ctrl_basis_4,
                q16_lfo: &q2,
                q16_env: &self.env_states,
                qu: &self.z_stage_ultra,
                is_ultra: self.tier == Tier::Ultra,
                cext_present: self.cext_present,
            },
            DispatchMatrices {
                w_ctrl: &self.w_ctrl,
                w_ctrl_ultra: &self.w_ctrl_ultra,
                w_ctrl_ext: &self.w_ctrl_ext,
                w_ctrl_ultra_ext: &self.w_ctrl_ultra_ext,
            },
            &mut self.stage_targets,
        );
        step_ou_drift(
            self.live_analog_drift_enabled,
            self.offline_bounce,
            self.sigma_drift,
            self.rho_drift_host,
            &mut self.drift_prng_state,
            &mut self.ou_drift_state,
        );
        prepare_micro_ramp_deltas::<B>(
            &self.stage_targets,
            &mut self.prev_stage_targets,
            &mut self.current_stage_params,
            &mut self.stage_ramp_deltas,
            self.alpha_ramp,
        );
    }

    /// Always-inlined frame-step helper for `#[target_feature]` block dispatch.
    #[inline(always)]
    pub(crate) fn step_frame_inline(
        &mut self,
        sensory_env: Option<[f32; 4]>,
    ) -> Option<&[f32; STAGE_TARGET_SLOTS]> {
        if !self.should_tick() {
            return None;
        }
        if self.ctrl_sub_counter == 0 {
            self.tick_control_boundary_inline(sensory_env);
        }
        advance_micro_ramp_frame::<B>(
            &self.stage_targets,
            &self.prev_stage_targets,
            &mut self.current_stage_params,
            &self.stage_ramp_deltas,
            self.ctrl_sub_counter,
            self.ctrl_stride,
        );
        self.ctrl_sub_counter += 1;
        if self.ctrl_sub_counter >= self.ctrl_stride {
            self.ctrl_sub_counter = 0;
        }
        Some(&self.current_stage_params)
    }
}
