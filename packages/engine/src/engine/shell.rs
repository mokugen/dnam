//! Layer 1 host shell accessors, patchbay routing wrappers, VM lock forwarding, and telemetry
//! publication helpers for [`DnamEngine`].

use dnam_vm::bin::{DNBN_FLAG_AUTO_UNITY, DNBN_FLAG_SINGLE_INPUT, DNBN_FLAG_STEREO_OUT};

use super::{
    DnamEngine, EngineBackend, HostShellPatchbay, IoRoutingMode, ParallelAlignmentMode,
    WdwDryTapMode,
};
use crate::provenance::PreviewGuardState;
use crate::types::{DnamTelemetrySnapshot, TELEMETRY_CONTROL_SLOTS, verification_status_code};

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Returns whether storefront preview containment (`PreviewGuardState` + `Lite` ceiling) is active.
    #[inline]
    pub const fn is_preview_mode(&self) -> bool {
        self.preview_mode
    }

    /// Returns an immutable view of the Layer 5/6 [`PreviewGuardState`].
    #[inline]
    pub const fn preview_guard(&self) -> &PreviewGuardState {
        &self.preview_guard
    }

    /// Returns an immutable view of the Layer 1 [`HostShellPatchbay`].
    #[inline]
    pub const fn patchbay(&self) -> &HostShellPatchbay {
        &self.patchbay
    }

    /// Returns a mutable view of the Layer 1 [`HostShellPatchbay`].
    #[inline]
    pub fn patchbay_mut(&mut self) -> &mut HostShellPatchbay {
        &mut self.patchbay
    }

    /// Sets the active multi-port [`IoRoutingMode`] across live and bounce patchbays.
    #[inline]
    pub fn set_routing_mode(&mut self, mode: IoRoutingMode) {
        self.patchbay.set_routing_mode(mode);
        self.bounce_patchbay.set_routing_mode(mode);
    }

    /// Configures WDW dry center tap selection (`PreSpatialCore` vs `DirectDry`).
    #[inline]
    pub fn set_wdw_mode(&mut self, mode: WdwDryTapMode) {
        self.patchbay.set_wdw_mode(mode);
        self.bounce_patchbay.set_wdw_mode(mode);
    }

    /// Enables or disables WDW output routing across live and bounce patchbays.
    #[inline]
    pub fn set_wdw_enabled(&mut self, enabled: bool) {
        self.patchbay.set_wdw_enabled(enabled);
        self.bounce_patchbay.set_wdw_enabled(enabled);
    }

    /// Configures external FX return fractional latency delays in active VM samples.
    #[inline]
    pub fn configure_return_alignment(&mut self, delay_l_samples: f32, delay_r_samples: f32) {
        self.patchbay
            .configure_return_alignment(delay_l_samples, delay_r_samples);
        self.bounce_patchbay
            .configure_return_alignment(delay_l_samples, delay_r_samples);
    }

    /// Configures parallel-path Kaiser/Catmull-Rom alignment mode, engagement, and `48 kHz` skews.
    #[inline]
    pub fn configure_parallel_alignment(
        &mut self,
        mode: ParallelAlignmentMode,
        enabled: bool,
        delta_l_48k: f32,
        delta_r_48k: f32,
    ) {
        self.patchbay
            .configure_parallel_alignment(mode, enabled, delta_l_48k, delta_r_48k);
        self.bounce_patchbay
            .configure_parallel_alignment(mode, enabled, delta_l_48k, delta_r_48k);
    }

    /// Sets or clears bit 2 (`QUAD_FLAG_HOST_WRITE_LOCK`) across all active `TransportQuad`s.
    #[inline]
    pub fn set_quad_write_enable(&mut self, locked: bool) {
        self.vm.set_quad_write_enable(locked);
        self.bounce_vm.set_quad_write_enable(locked);
    }

    /// Sets or clears host-direct `Op-T` write lock on `SonaVm`.
    #[inline]
    pub fn set_op_t_write_enable(&mut self, locked: bool) {
        self.vm.set_op_t_write_enable(locked);
        self.bounce_vm.set_op_t_write_enable(locked);
    }

    /// Enables or disables optional live Ornstein–Uhlenbeck analog drift (`false` by default).
    #[inline]
    pub fn set_live_analog_drift(&mut self, enabled: bool) {
        self.vm.set_live_analog_drift_enabled(enabled);
    }

    /// Returns whether live Ornstein–Uhlenbeck analog drift is enabled.
    #[inline]
    pub fn live_analog_drift_enabled(&self) -> bool {
        self.vm
            .slab()
            .cold()
            .ctrl_bus
            .as_ref()
            .is_some_and(|b| b.live_analog_drift_enabled)
    }

    #[inline]
    pub(crate) const fn tempo_total_frames(&self) -> u64 {
        self.tempo_state.total_frames
    }

    #[inline]
    pub(crate) fn tier_os2x(&self) -> [bool; 4] {
        self.vm.slab().cold().tier_os2x
    }

    #[inline]
    pub(crate) fn ref_sample_rate_hz(&self) -> f32 {
        self.vm.slab().cold().zoh.reference_rate_hz
    }

    #[cfg(any(test, feature = "std"))]
    #[inline]
    pub(crate) fn compute_zoh_for_mode(
        &self,
        sample_rate_hz: u32,
        offline_bounce: bool,
    ) -> dnam_vm::ZohRateContext {
        let tier = if self.preview_mode {
            crate::types::Tier::Lite
        } else if offline_bounce && self.maximize_tier_on_bounce {
            self.max_available_tier.min(self.max_supported_tier)
        } else {
            self.tier
                .min(self.max_available_tier)
                .min(self.max_supported_tier)
        };
        let cold = self.vm.slab().cold();
        let os2x = cold.tier_os2x[tier.index()];
        dnam_vm::ZohRateContext::compute(
            cold.zoh.reference_rate_hz,
            sample_rate_hz as f32,
            tier,
            os2x,
            offline_bounce,
        )
        .unwrap_or(cold.zoh)
    }

    /// Builds a wait-free-compatible [`DnamTelemetrySnapshot`] from `self.vm`, `patchbay`, and
    /// host shell metadata (`0` heap allocations).
    pub fn telemetry_snapshot(&self) -> DnamTelemetrySnapshot {
        let slab = self.vm.slab();
        let cold = slab.cold();
        let vm_tel = self.vm.telemetry();

        let mut active_clamp_mask = self.extra_clamp_mask;
        let mut z_head = [0.0_f32; 4];
        let mut env_states = [0.0_f32; 4];
        let mut lfo_harmonics = [0.0_f32; 8];
        let mut clamped_controls = [0.0_f32; TELEMETRY_CONTROL_SLOTS];

        if let Some(bus) = cold.ctrl_bus.as_ref() {
            let n = bus.num_controls;
            let sw_n = n.min(bus.backing.active_switch_options.len());
            let slot_n = n
                .min(TELEMETRY_CONTROL_SLOTS)
                .min(bus.backing.descriptors.len())
                .min(bus.backing.raw_values.len())
                .min(bus.backing.clamped_values.len());
            for i in 0..slot_n {
                clamped_controls[i] = bus.backing.clamped_values[i];
                let (_, _, was) = bus.backing.descriptors[i].clamp_and_normalize(
                    bus.backing.raw_values[i],
                    &bus.backing.active_switch_options[..sw_n],
                );
                if was {
                    active_clamp_mask |= 1_u16 << i;
                }
            }
            z_head = bus.z_head;
            env_states = bus.env_states;
            for j in 0..2 {
                let x = bus.lfo_phasors[j][0];
                let t2 = 2.0 * x * x - 1.0;
                let t3 = 2.0 * x * t2 - x;
                let t4 = 2.0 * t2 * t2 - 1.0;
                lfo_harmonics[j * 4] = x;
                lfo_harmonics[j * 4 + 1] = t2;
                lfo_harmonics[j * 4 + 2] = t3;
                lfo_harmonics[j * 4 + 3] = t4;
            }
        }

        let active_kautz_pairs = self
            .effective_tier()
            .kautz_pairs()
            .min(cold.kautz_slew.host_poles.len());
        let mut max_pole_radius = 0.0_f32;
        for &(r, _) in &cold.kautz_slew.host_poles[..active_kautz_pairs] {
            if r > max_pole_radius {
                max_pole_radius = r;
            }
        }

        let active_quads = cold
            .active_transport_quads
            .min(cold.transport_quad_eff.len());
        let mut max_effective_loop_gain = 0.0_f32;
        for q in &cold.transport_quad_eff[..active_quads] {
            if q.g_loop_eff > max_effective_loop_gain {
                max_effective_loop_gain = q.g_loop_eff;
            }
        }

        let bin_header_flags = self.vm_dnbn_flags();
        let stereo_out =
            if (bin_header_flags & DNBN_FLAG_STEREO_OUT) != 0 || self.preamble.is_stereo_out() {
                1
            } else {
                0
            };

        let effective_wet_pol_inv =
            self.vm.slab().wet_polarity_inv() || self.json_wet_polarity_inverted();
        let telemetry_wet_gain = if effective_wet_pol_inv {
            -self.mix_wet_gain
        } else {
            self.mix_wet_gain
        };

        DnamTelemetrySnapshot {
            active_tier: vm_tel.active_tier as u8,
            max_available_tier: self.max_available_tier as u8,
            max_supported_tier: self.max_supported_tier as u8,
            stereo_out,
            bypassed: u8::from(self.bypassed),
            offline_bounce: u8::from(self.offline_bounce),
            host_write_lock: u8::from(vm_tel.host_write_lock),
            verification_status: verification_status_code(self.verification_status()),
            active_clamp_mask,
            control_decimation_host: cold.zoh.k_ctrl.min(u16::MAX as usize) as u16,
            latency_samples: self.latency_samples().min(u32::MAX as usize) as u32,
            sample_rate_hz: self.tempo_state.sample_rate_hz,
            saturation_flags: self.saturation_flags | vm_tel.saturation_flags,
            mix_dry_gain: self.mix_dry_gain,
            mix_wet_gain: telemetry_wet_gain,
            stereo_width: self.patchbay.ms_width,
            tempo_bpm: self.tempo_state.current_bpm,
            max_op_n_drive: self.last_block_max_drive,
            max_pole_radius,
            max_effective_loop_gain,
            tail_time_ms: self.tail_time_ms,
            z_head,
            env_states,
            lfo_harmonics,
            clamped_controls,
        }
    }

    #[inline]
    pub(crate) fn vm_dnbn_flags(&self) -> u16 {
        let mut flags = 0_u16;
        if self.single_input {
            flags |= DNBN_FLAG_SINGLE_INPUT;
        }
        if self.preamble.is_stereo_out() {
            flags |= DNBN_FLAG_STEREO_OUT;
        }
        if self
            .reloaded_header_flags
            .is_some_and(|f| f.auto_unity_enabled)
            || self
                .header()
                .is_some_and(|h| h.io_config.leveling.auto_unity_enabled)
        {
            flags |= DNBN_FLAG_AUTO_UNITY;
        }
        flags
    }
}
