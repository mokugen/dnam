//! Isolated offline-bounce toggling and PDC-stripped rendering for [`DnamEngine`].

use super::super::{DnamEngine, EngineBackend, OFFLINE_BOUNCE_PDC_SAMPLES};
use crate::container::ZohRescaledParams;
use crate::types::Tier;

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Reconfigures ZOH parameters for a live/offline-bounce transition without wiping
    /// recursive DSP state or dynamic `ControlBus` state (`R20-E4`).
    ///
    /// Returns `true` on success (`0` `DnamError` heap allocation or drop glue on the RT path).
    #[inline]
    pub(crate) fn reconfigure_for_bounce_toggle(&mut self) -> bool {
        self.control_bus.offline_bounce = self.offline_bounce;
        self.control_bus.ultra_tier_active = self.effective_tier() == Tier::Ultra;
        let sample_rate_hz = self.zoh.sample_rate_host_hz;
        let (r_ac_tr, r_dc_tr, tau_ref_tr, switch_xfade_tr) = self.training_zoh_poles();
        let oversample_factor = if self.offline_bounce { 4 } else { 1 };
        let Some(new_zoh) = ZohRescaledParams::compute_oversampled_opt(
            self.preamble.sample_rate_train_hz,
            sample_rate_hz,
            oversample_factor,
            r_ac_tr,
            r_dc_tr,
            tau_ref_tr,
            switch_xfade_tr,
        ) else {
            return false;
        };
        self.zoh = new_zoh;
        let rho_ctrl = self.zoh.rho_ctrl;
        {
            let eff_len = self
                .effective_tier()
                .expected_arena_bytes()
                .min(self.arena_storage.as_bytes().len());
            let arena = &self.arena_storage.as_bytes()[..eff_len];
            self.control_bus.update_sample_rate_params(arena, rho_ctrl);
        }
        self.load_v3_hot_core_from_arena(false);
        self.sync_immediate_control_state_inner();
        self.oversampler.reset();
        self.oversampler_aux.reset();
        self.ctrl_sub_counter = 0;
        self.ctrl_energy_accum = 0.0;
        true
    }

    /// Renders a complete stereo buffer in isolated Offline Bounce mode, stripping the 12-sample
    /// PDC lead-in while preserving all live recursive DSP, `ControlBus`, and router state on `self`
    /// (`0` runtime heap allocations, zero `reset_to_dc_equilibrium` on live `self`).
    pub fn render_offline_bounce(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let p_cnt = self
            .control_bus
            .num_controls
            .min(self.control_bus.current_u_norm.len())
            .min(self.bounce_scratch.u_norm.len());
        self.bounce_scratch.u_norm[..p_cnt]
            .copy_from_slice(&self.control_bus.current_u_norm[..p_cnt]);
        self.bounce_scratch.prev_u_norm[..p_cnt]
            .copy_from_slice(&self.control_bus.prev_eval_u_norm[..p_cnt]);
        self.bounce_scratch.prev_psi[..p_cnt]
            .copy_from_slice(&self.control_bus.prev_eval_psi[..p_cnt]);

        let saved_zoh = self.zoh;
        let saved_bounce = self.offline_bounce;
        let saved_os = self.oversampler.clone();
        let saved_os_aux = self.oversampler_aux.clone();
        let saved_tempo = self.tempo_state.clone();
        let saved_sub_counter = self.ctrl_sub_counter;
        let saved_energy_accum = self.ctrl_energy_accum;
        let saved_stage_params = self.current_stage_params;
        let saved_ramp_deltas = self.stage_ramp_deltas;
        let saved_sat_flags = self.saturation_flags;

        let saved_cb_lat = self.control_bus.latent_acc;
        let saved_cb_svx_u = self.control_bus.s_vx_ultra;
        let saved_cb_sv2_u = self.control_bus.s_v2x2_ultra;
        self.bounce_scratch.exact_fm_acc = self.control_bus.exact_fm_acc;
        let saved_cb_dirty = self.control_bus.is_dirty;
        let saved_cb_zh = self.control_bus.z_head;
        let saved_cb_zhu = self.control_bus.z_head_ultra;
        let saved_cb_zsu = self.control_bus.z_stage_ultra;
        let saved_cb_basis = self.control_bus.ctrl_basis_4;
        let saved_cb_zs = self.control_bus.z_stage;
        let saved_cb_env = self.control_bus.env_states;
        let saved_cb_env_p = self.control_bus.env_poles;
        let saved_cb_lfo_ph = self.control_bus.lfo_phasors;
        let saved_cb_lfo_rot = self.control_bus.lfo_rotors;
        let saved_cb_q192 = self.control_bus.lfo_phases_q192;
        let saved_cb_ab = self.control_bus.lfo_angle_bits;
        let saved_cb_cab = self.control_bus.lfo_cached_angle_bits;
        let saved_cb_cq = self.control_bus.lfo_cached_q192_inc;
        let saved_cb_lfo_h = self.control_bus.lfo_harmonics;
        let saved_cb_targets = self.control_bus.stage_targets;
        let saved_cb_tdel = self.control_bus.tank_delay_mod_samples;
        let saved_cb_tgain = self.control_bus.tank_gain_targets;
        let saved_cb_twarp = self.control_bus.tank_warp_mod;
        let saved_cb_tglen = self.control_bus.tank_grain_len_scale;
        let saved_cb_tstate = self.control_bus.tank_state_mod;
        let saved_cb_tdirty = self.control_bus.target_dispatch_dirty;
        let saved_cb_rho = self.control_bus.rho_drift_host;
        let saved_cb_off = self.control_bus.offline_bounce;
        let saved_cb_ultra_active = self.control_bus.ultra_tier_active;
        let saved_cb_prng = self.control_bus.drift_prng_state;
        let saved_cb_ou = self.control_bus.ou_drift_state;

        core::mem::swap(&mut self.hot_core, &mut self.bounce_scratch.hot_core);
        core::mem::swap(&mut self.io_router, &mut self.bounce_scratch.io_router);

        self.offline_bounce = true;
        self.control_bus.offline_bounce = true;
        let _ = self.reconfigure_for_bounce_toggle();
        self.reset_to_dc_equilibrium();

        let n = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        let pdc = OFFLINE_BOUNCE_PDC_SAMPLES;
        let mut sink_l = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
        let mut sink_r = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];

        if n <= pdc {
            let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
            self.process_block(
                &left_in[..n],
                &right_in[..n],
                &mut sink_l[..n],
                &mut sink_r[..n],
            );
            let rem = pdc - n;
            if rem > 0 {
                self.process_block(
                    &zeros[..rem],
                    &zeros[..rem],
                    &mut sink_l[..rem],
                    &mut sink_r[..rem],
                );
            }
            self.process_block(
                &zeros[..n],
                &zeros[..n],
                &mut left_out[..n],
                &mut right_out[..n],
            );
        } else {
            self.process_block(&left_in[..pdc], &right_in[..pdc], &mut sink_l, &mut sink_r);
            let body_len = n - pdc;
            self.process_block(
                &left_in[pdc..n],
                &right_in[pdc..n],
                &mut left_out[..body_len],
                &mut right_out[..body_len],
            );
            let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
            self.process_block(
                &zeros,
                &zeros,
                &mut left_out[body_len..n],
                &mut right_out[body_len..n],
            );
        }

        core::mem::swap(&mut self.hot_core, &mut self.bounce_scratch.hot_core);
        core::mem::swap(&mut self.io_router, &mut self.bounce_scratch.io_router);

        self.zoh = saved_zoh;
        self.offline_bounce = saved_bounce;
        self.oversampler = saved_os;
        self.oversampler_aux = saved_os_aux;
        self.tempo_state = saved_tempo;
        self.ctrl_sub_counter = saved_sub_counter;
        self.ctrl_energy_accum = saved_energy_accum;
        self.current_stage_params = saved_stage_params;
        self.stage_ramp_deltas = saved_ramp_deltas;
        self.saturation_flags = saved_sat_flags;

        self.control_bus.current_u_norm[..p_cnt]
            .copy_from_slice(&self.bounce_scratch.u_norm[..p_cnt]);
        self.control_bus.prev_eval_u_norm[..p_cnt]
            .copy_from_slice(&self.bounce_scratch.prev_u_norm[..p_cnt]);
        self.control_bus.prev_eval_psi[..p_cnt]
            .copy_from_slice(&self.bounce_scratch.prev_psi[..p_cnt]);
        self.control_bus.latent_acc = saved_cb_lat;
        self.control_bus.s_vx_ultra = saved_cb_svx_u;
        self.control_bus.s_v2x2_ultra = saved_cb_sv2_u;
        self.control_bus.exact_fm_acc = self.bounce_scratch.exact_fm_acc;
        self.control_bus.is_dirty = saved_cb_dirty;
        self.control_bus.z_head = saved_cb_zh;
        self.control_bus.z_head_ultra = saved_cb_zhu;
        self.control_bus.z_stage_ultra = saved_cb_zsu;
        self.control_bus.ctrl_basis_4 = saved_cb_basis;
        self.control_bus.z_stage = saved_cb_zs;
        self.control_bus.env_states = saved_cb_env;
        self.control_bus.env_poles = saved_cb_env_p;
        self.control_bus.lfo_phasors = saved_cb_lfo_ph;
        self.control_bus.lfo_rotors = saved_cb_lfo_rot;
        self.control_bus.lfo_phases_q192 = saved_cb_q192;
        self.control_bus.lfo_angle_bits = saved_cb_ab;
        self.control_bus.lfo_cached_angle_bits = saved_cb_cab;
        self.control_bus.lfo_cached_q192_inc = saved_cb_cq;
        self.control_bus.lfo_harmonics = saved_cb_lfo_h;
        self.control_bus.stage_targets = saved_cb_targets;
        self.control_bus.tank_delay_mod_samples = saved_cb_tdel;
        self.control_bus.tank_gain_targets = saved_cb_tgain;
        self.control_bus.tank_warp_mod = saved_cb_twarp;
        self.control_bus.tank_grain_len_scale = saved_cb_tglen;
        self.control_bus.tank_state_mod = saved_cb_tstate;
        self.control_bus.target_dispatch_dirty = saved_cb_tdirty;
        self.control_bus.rho_drift_host = saved_cb_rho;
        self.control_bus.offline_bounce = saved_cb_off;
        self.control_bus.ultra_tier_active = saved_cb_ultra_active;
        self.control_bus.drift_prng_state = saved_cb_prng;
        self.control_bus.ou_drift_state = saved_cb_ou;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
    use crate::math::ScalarBackend;
    use crate::schema::DnamJsonHeader;
    use alloc::vec;

    #[test]
    fn test_render_offline_bounce_preserves_live_state_and_toggles_bounce_cleanly() {
        let mut header = DnamJsonHeader::default_reference();
        let arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let mut ref_engine =
            DnamEngine::<ScalarBackend>::from_bytes(&container, Tier::Lite, 48_000).unwrap();

        let in_l = [0.2_f32; 32];
        let in_r = [-0.15_f32; 32];
        let (mut out_l1, mut out_r1) = ([0.0_f32; 32], [0.0_f32; 32]);
        let (mut ref_l1, mut ref_r1) = ([0.0_f32; 32], [0.0_f32; 32]);
        engine.process_block(&in_l, &in_r, &mut out_l1, &mut out_r1);
        ref_engine.process_block(&in_l, &in_r, &mut ref_l1, &mut ref_r1);

        let tick_before = engine.hot_core.sample_tick();
        let (mut b_l, mut b_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        engine.render_offline_bounce(&in_l, &in_r, &mut b_l, &mut b_r);
        assert_eq!(engine.hot_core.sample_tick(), tick_before);
        assert!(!engine.offline_bounce);
        assert!(!engine.hot_core.cold.is_bounce_4x);

        let (mut out_l2, mut out_r2) = ([0.0_f32; 32], [0.0_f32; 32]);
        let (mut ref_l2, mut ref_r2) = ([0.0_f32; 32], [0.0_f32; 32]);
        engine.process_block(&in_l, &in_r, &mut out_l2, &mut out_r2);
        ref_engine.process_block(&in_l, &in_r, &mut ref_l2, &mut ref_r2);
        for i in 0..32 {
            assert_eq!(out_l2[i].to_bits(), ref_l2[i].to_bits());
            assert_eq!(out_r2[i].to_bits(), ref_r2[i].to_bits());
        }

        engine.set_offline_bounce(true);
        assert!(engine.hot_core.cold.is_bounce_4x);
        engine.set_offline_bounce(false);
        assert!(!engine.hot_core.cold.is_bounce_4x);
    }
}
