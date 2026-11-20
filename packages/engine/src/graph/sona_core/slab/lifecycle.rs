//! Runtime Lifecycle, Tier Switching, Cold Preallocation & ZOH Reconfiguration for [`SonaHotSlab`]
//! (`graph/sona_core/slab/lifecycle.rs`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::StereoVec8;

use crate::container::ZohRescaledParams;
use crate::graph::GraphBackend;
use crate::graph::sona_core::model::{
    MAX_KAUTZ_BANKS, MAX_SONA_GROUPS, MAX_TANK_LINES, MAX_TANK_QUADS, MAX_TOTAL_SONA_LANES,
    SonaColdTierModel, tier_kautz_banks,
};
use crate::graph::tank::fdn::{SpatialTankParams, V3_TANK_CAP};
use crate::types::Tier;

use super::SonaHotSlab;
use super::ctrl_apply::{
    self, apply_control_bus_impl, ensure_stage4_quad_cold, sync_stage3_with_optional_zoh,
    sync_stage4_params,
};
use super::layout::{
    FLAG_FREEZE, FLAG_XFADE_ACTIVE, SonaControlBusState, TIER_SWITCH_XFADE_SAMPLES,
};

impl<B: GraphBackend> SonaHotSlab<B> {
    fn sync_fir_hist_on_tier_switch(&mut self, from_tier: Tier, to_tier: Tier) {
        if from_tier == to_tier {
            return;
        }
        let was_hi = matches!(from_tier, Tier::Pro | Tier::Ultra);
        let now_hi = matches!(to_tier, Tier::Pro | Tier::Ultra);
        if was_hi && now_hi {
            return;
        }
        let wr = self.lite.fir_write_pos as usize;
        let mut tmp_l = [0.0_f32; 128];
        let mut tmp_r = [0.0_f32; 128];
        if was_hi {
            for k in 0..128 {
                let idx = (wr + k) & 127;
                tmp_l[k] = self.pro_delta.fir_hist_l_pro[idx];
                tmp_r[k] = self.pro_delta.fir_hist_r_pro[idx];
            }
        } else {
            let cap_from = if from_tier == Tier::Lite { 32 } else { 64 };
            let mask_from = cap_from - 1;
            for k in 0..cap_from {
                let idx = (wr + k) & mask_from;
                tmp_l[k] = self.lite.fir_hist_l[idx];
                tmp_r[k] = self.lite.fir_hist_r[idx];
            }
        }
        self.lite.fir_write_pos = 0;
        if now_hi {
            self.pro_delta.fir_hist_l_pro[..128].copy_from_slice(&tmp_l);
            self.pro_delta.fir_hist_l_pro[128..].copy_from_slice(&tmp_l);
            self.pro_delta.fir_hist_r_pro[..128].copy_from_slice(&tmp_r);
            self.pro_delta.fir_hist_r_pro[128..].copy_from_slice(&tmp_r);
        } else if to_tier == Tier::Lite {
            self.lite.fir_hist_l[..32].copy_from_slice(&tmp_l[..32]);
            self.lite.fir_hist_l[32..].copy_from_slice(&tmp_l[..32]);
            self.lite.fir_hist_r[..32].copy_from_slice(&tmp_r[..32]);
            self.lite.fir_hist_r[32..].copy_from_slice(&tmp_r[..32]);
        } else {
            self.lite.fir_hist_l.copy_from_slice(&tmp_l[..64]);
            self.lite.fir_hist_r.copy_from_slice(&tmp_r[..64]);
        }
    }

    /// Switches active tier & readout weights in-place (`0` RT heap allocs, `0` reset),
    /// preserving `phasors.phase`, `zoh`, `warm_seed` on `1x -> 2x`, and `48`-sample crossfade.
    pub fn switch_tier(&mut self, model: &SonaColdTierModel) {
        let was_tier = self.lite.tier;
        let was_os2x = self.os2x();
        let was_frozen = self.tank_freeze();

        self.sync_fir_hist_on_tier_switch(was_tier, model.tier);
        if !was_os2x && model.os2x {
            self.os2x_hot
                .os_stereo
                .warm_seed(self.cold.last_in, self.cold.last_s2_dir);
        }

        let stored_zoh = self.cold.zoh;
        self.apply_weights_common(model, stored_zoh.as_ref());
        // Stage II downsamples back to 1x host rate before Stage III; only 4x offline bounce changes Stage III rate.
        let bounce_4x = self.cold.is_bounce_4x;
        let stage3_os = if bounce_4x { 4 } else { 1 };
        sync_stage3_with_optional_zoh(
            self,
            &model.readout,
            &model.kautz_poles,
            stage3_os,
            bounce_4x,
            stored_zoh.as_ref(),
        );
        sync_stage4_params(self, &model.stage4, stored_zoh.as_ref(), false);
        self.set_tank_freeze(was_frozen);

        let last_t = self.cold.last_stage_targets;
        let last_z = self.cold.last_z_stage;
        if last_t != [0.0; 16] || last_z != [0.0; 8] {
            apply_control_bus_impl(self, &last_t, &last_z, true);
        }

        if was_tier != model.tier {
            self.cold.xfade_rem = TIER_SWITCH_XFADE_SAMPLES;
            self.cold.xfade_total = TIER_SWITCH_XFADE_SAMPLES;
            self.lite.flags |= FLAG_XFADE_ACTIVE;
        }
    }

    /// Reloads a complete model in place and resets all DSP state (`0` heap allocs, `0` drops).
    pub fn reload_in_place(&mut self, model: &SonaColdTierModel) {
        self.cold.zoh = None;
        self.cold.os_factor = if model.os2x { 2 } else { 1 };
        self.cold.is_bounce_4x = false;
        self.apply_weights_common(model, None);
        self.lite.phasors.phase = model.crossbar.phasor_init_phase_q32;
        sync_stage3_with_optional_zoh(self, &model.readout, &model.kautz_poles, 1, false, None);
        sync_stage4_params(self, &model.stage4, None, true);
        self.reset_to_dc_equilibrium();
    }

    /// Pre-allocates cold extended tank delay & Schroeder buffers cleanly on non-RT paths.
    pub fn ensure_cold_tank_cap(&mut self, params: &SpatialTankParams, min_frames_cap: usize) {
        let mut p = *params;
        p.clamp_in_place();
        let aq = p.active_quads.clamp(1, MAX_TANK_QUADS);
        for q in 0..MAX_TANK_QUADS {
            if q == 0 || q < aq || min_frames_cap > V3_TANK_CAP {
                let qp = p.effective_quad_params(q);
                ensure_stage4_quad_cold(
                    &mut self.cold.stage4_ext[q],
                    &mut self.cold.stage4_schroeder[q],
                    &qp,
                    min_frames_cap,
                    q == 0,
                );
            }
        }
    }

    /// Reconfigures continuous-time ZOH poles, delays, and Stage III/IV models with explicit OS mode.
    pub fn reconfigure_with_zoh_mode(
        &mut self,
        base: &SonaColdTierModel,
        host_zoh: &ZohRescaledParams,
        os_factor: usize,
        is_bounce_4x: bool,
    ) {
        self.cold.zoh = Some(*host_zoh);
        self.cold.os_factor = os_factor.max(1);
        self.cold.is_bounce_4x = is_bounce_4x;
        let saved_psi = self.cold.ctrl_state.tank_psi_delta;
        let saved_k_oct = self.cold.ctrl_state.kautz_oct_delta;
        let saved_k_damp = self.cold.ctrl_state.kautz_damp_delta;
        self.apply_weights_common(base, Some(host_zoh));
        // Stage II downsamples back to 1x host rate before Stage III; only 4x offline bounce changes Stage III rate.
        let stage3_os = if is_bounce_4x { 4 } else { 1 };
        sync_stage3_with_optional_zoh(
            self,
            &base.readout,
            &base.kautz_poles,
            stage3_os,
            is_bounce_4x,
            Some(host_zoh),
        );
        sync_stage4_params(self, &base.stage4, Some(host_zoh), false);
        for q in 0..MAX_TANK_QUADS {
            self.quad_hot_mut(q).snap_psi_immediate(saved_psi);
        }
        if saved_k_oct.abs() > 1.0e-7 || saved_k_damp.abs() > 1.0e-7 {
            ctrl_apply::retune_slab_kautz::<B>(
                self,
                tier_kautz_banks(self.lite.tier),
                saved_k_oct,
                saved_k_damp,
                true,
            );
        }
        let _ = MAX_TOTAL_SONA_LANES;
    }

    /// Reconfigures all continuous-time ZOH poles, delays, and Stage III/IV models (`0` RT drops).
    pub fn reconfigure_with_zoh(&mut self, base: &SonaColdTierModel, host_zoh: &ZohRescaledParams) {
        let is_bounce_4x = host_zoh.rho_sr <= 0.28;
        let os_factor = if base.os2x && !is_bounce_4x { 2 } else { 1 };
        self.reconfigure_with_zoh_mode(base, host_zoh, os_factor, is_bounce_4x);
    }

    /// Snaps `stage_targets[0..16]` and `z_stage[0..8]` across Stages I–IV immediately (`0` slew lag).
    #[inline]
    pub fn snap_control_bus_with_z_stage(&mut self, stage_targets: &[f32; 16], z_stage: &[f32; 8]) {
        apply_control_bus_impl(self, stage_targets, z_stage, true);
    }

    /// Applies unified `3 kHz` `stage_targets[0..16]` + `z_stage[0..8]` across Stages I–IV (`0` alloc).
    #[inline]
    pub fn apply_control_bus_with_z_stage(
        &mut self,
        stage_targets: &[f32; 16],
        z_stage: &[f32; 8],
    ) {
        apply_control_bus_impl(self, stage_targets, z_stage, false);
    }

    /// Snaps `stage_targets[0..16]` immediately with zero latent delta (`z_stage = 0`).
    #[inline]
    pub fn snap_control_bus_stage_targets(&mut self, stage_targets: &[f32; 16]) {
        apply_control_bus_impl(self, stage_targets, &[0.0; 8], true);
    }

    /// Applies unified `3 kHz` `stage_targets[0..16]` with zero latent delta (`z_stage = 0`).
    #[inline]
    pub fn apply_control_bus_stage_targets(&mut self, stage_targets: &[f32; 16]) {
        apply_control_bus_impl(self, stage_targets, &[0.0; 8], false);
    }

    /// Resets all recursive filters, `Op-P` lattice, `SONA` states, FIR, and tank states to `+0.0`.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.lite.stage0_xo_x0 = [0.0; 2];
        self.lite.stage0_xo_y_l = [0.0; 4];
        self.lite.stage0_xo_y_r = [0.0; 4];
        self.lite.stage0_gate_env = 0.0;
        self.lite.stage0_gate_gain = 1.0;
        self.lite.stage0_hp_lp_state = [0.0; 2];
        self.lite.t1_dc.reset();
        self.lite.hilbert.reset();
        self.lite.t1.reset();
        self.lite.phasors.phase = self.cold.crossbar_init_phase_q32;
        for g in 0..MAX_SONA_GROUPS {
            let st = self.state_mut(g);
            st.svf_s1 = B::Stereo8::zero();
            st.svf_s2 = B::Stereo8::zero();
            st.lru_v = B::Stereo8::zero();
            st.adaa_prev = B::Stereo8::zero();
        }
        self.os2x_hot.os_stereo.reset();
        self.lite.fir_hist_l.fill(0.0);
        self.lite.fir_hist_r.fill(0.0);
        self.pro_delta.fir_hist_l_pro.fill(0.0);
        self.pro_delta.fir_hist_r_pro.fill(0.0);
        self.lite.fir_write_pos = 0;
        self.cold.stage3_fir_ext.reset();
        for b in 0..MAX_KAUTZ_BANKS {
            ctrl_apply::kautz_bank_mut(self, b).reset();
        }
        self.cold.kautz_slew.reset_slew();
        ctrl_apply::retune_slab_kautz::<B>(self, tier_kautz_banks(self.lite.tier), 0.0, 0.0, true);
        self.lite.quad0_frames.fill([0; MAX_TANK_LINES / 4]);
        for q in 0..MAX_TANK_QUADS {
            self.quad_hot_mut(q).reset();
            if let Some(ext) = self.cold.stage4_ext[q].as_mut() {
                ext.fill_zero();
            }
            if let Some(ap) = self.cold.stage4_schroeder[q].as_mut() {
                ap.reset();
            }
        }
        self.cold.ctrl_state = SonaControlBusState::identity();
        self.lite.sample_tick = 0;
        self.lite.flags &= !(FLAG_FREEZE | FLAG_XFADE_ACTIVE);
        self.cold.last_in = [0.0; 2];
        self.cold.max_op_n_drive = 0.0;
        self.cold.last_s2_dir = [0.0; 4];
        self.cold.last_stage_targets = [0.0; 16];
        self.cold.last_z_stage = [0.0; 8];
        self.cold.xfade_rem = 0;
        self.cold.xfade_prev_out = [0.0; 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;

    #[test]
    fn test_sync_fir_hist_on_tier_switch_preserves_continuity_across_all_tiers() {
        let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
        for &from_tier in &tiers {
            for &to_tier in &tiers {
                let model = SonaColdTierModel::identity_for_tier(from_tier, false, false);
                let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&model);
                let wr = 45usize;
                slab.lite.fir_write_pos = match from_tier {
                    Tier::Lite => (wr & 31) as u16,
                    Tier::Full => (wr & 63) as u16,
                    Tier::Pro | Tier::Ultra => (wr & 127) as u16,
                };
                let cap_from = match from_tier {
                    Tier::Lite => 32,
                    Tier::Full => 64,
                    Tier::Pro | Tier::Ultra => 128,
                };
                for k in 0..cap_from {
                    let s_l = (100 + k) as f32;
                    let s_r = -((200 + k) as f32);
                    if matches!(from_tier, Tier::Pro | Tier::Ultra) {
                        let idx = (wr + k) & 127;
                        slab.pro_delta.fir_hist_l_pro[idx] = s_l;
                        slab.pro_delta.fir_hist_l_pro[idx + 128] = s_l;
                        slab.pro_delta.fir_hist_r_pro[idx] = s_r;
                        slab.pro_delta.fir_hist_r_pro[idx + 128] = s_r;
                    } else if from_tier == Tier::Lite {
                        let idx = (wr + k) & 31;
                        slab.lite.fir_hist_l[idx] = s_l;
                        slab.lite.fir_hist_l[idx + 32] = s_l;
                        slab.lite.fir_hist_r[idx] = s_r;
                        slab.lite.fir_hist_r[idx + 32] = s_r;
                    } else {
                        let idx = (wr + k) & 63;
                        slab.lite.fir_hist_l[idx] = s_l;
                        slab.lite.fir_hist_r[idx] = s_r;
                    }
                }

                slab.sync_fir_hist_on_tier_switch(from_tier, to_tier);

                let cap_to = match to_tier {
                    Tier::Lite => 32,
                    Tier::Full => 64,
                    Tier::Pro | Tier::Ultra => 128,
                };
                let new_wr = slab.lite.fir_write_pos as usize;
                let shared = cap_from.min(cap_to);
                for k in 0..cap_to {
                    let expected_l = if k < shared { (100 + k) as f32 } else { 0.0 };
                    let expected_r = if k < shared { -((200 + k) as f32) } else { 0.0 };
                    if matches!(to_tier, Tier::Pro | Tier::Ultra) {
                        let idx = (new_wr + k) & 127;
                        assert_eq!(slab.pro_delta.fir_hist_l_pro[idx], expected_l);
                        assert_eq!(slab.pro_delta.fir_hist_l_pro[idx + 128], expected_l);
                        assert_eq!(slab.pro_delta.fir_hist_r_pro[idx], expected_r);
                        assert_eq!(slab.pro_delta.fir_hist_r_pro[idx + 128], expected_r);
                    } else if to_tier == Tier::Lite {
                        let idx = (new_wr + k) & 31;
                        assert_eq!(slab.lite.fir_hist_l[idx], expected_l);
                        assert_eq!(slab.lite.fir_hist_l[idx + 32], expected_l);
                        assert_eq!(slab.lite.fir_hist_r[idx], expected_r);
                        assert_eq!(slab.lite.fir_hist_r[idx + 32], expected_r);
                    } else {
                        let idx = (new_wr + k) & 63;
                        assert_eq!(slab.lite.fir_hist_l[idx], expected_l);
                        assert_eq!(slab.lite.fir_hist_r[idx], expected_r);
                    }
                }
            }
        }
    }

    #[test]
    fn test_stage3_stays_1x_live_when_stage2_os2x_is_enabled() {
        let mut model_2x = SonaColdTierModel::identity_for_tier(Tier::Full, true, false);
        model_2x.readout.fir_l[0] = 1.0;
        model_2x.readout.fir_r[0] = 1.0;
        let zoh_44 = ZohRescaledParams::new(48_000, 44_100);
        let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&model_2x);
        slab.reconfigure_with_zoh(&model_2x, &zoh_44);
        assert!(slab.os2x());
        assert!(!slab.cold.is_bounce_4x);
        // Live1xDiff evaluates tap0 directly at sample 0 (1.0), whereas erroneous 2x (88.2 kHz)
        // DenseResampled would scale by rho_sr = 48000 / 88200 (~0.5442).
        let (out_zoh, _, _) = slab
            .cold
            .stage3_fir_ext
            .step_frame::<DefaultMathBackend, false>(1.0, 1.0);
        assert!((out_zoh[0] - 1.0).abs() < 1e-6);

        slab.switch_tier(&model_2x);
        let (out_sw, _, _) = slab
            .cold
            .stage3_fir_ext
            .step_frame::<DefaultMathBackend, false>(1.0, 1.0);
        assert!((out_sw[0] - 1.0).abs() < 1e-6);
    }
}
