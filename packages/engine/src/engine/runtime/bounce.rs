//! Isolated offline-bounce toggling and PDC-stripped rendering over `bounce_vm` + `bounce_patchbay`.

use dnam_vm::graph::slab::ctrl_latch::latch_immediate_at_init_or_reset;

use super::super::process::{StereoShellParams, process_stereo_vm_block};
use super::super::{DnamEngine, EngineBackend, OFFLINE_BOUNCE_PDC_SAMPLES};
use crate::types::Tier;

impl<'a, B: EngineBackend> DnamEngine<'a, B> {
    /// Reconfigures ZOH parameters for a live/offline-bounce transition without wiping
    /// recursive DSP state or control values (`0` heap allocations).
    #[inline]
    pub(crate) fn reconfigure_for_bounce_toggle(&mut self) -> bool {
        if self.vm.set_offline_bounce(self.offline_bounce).is_err() {
            return false;
        }
        let target_tier = self.effective_tier();
        if self.vm.slab().lite.tier != target_tier && self.vm.switch_tier(target_tier).is_err() {
            return false;
        }
        let zoh = self.vm.slab().cold().zoh;
        self.patchbay
            .reconfigure_sample_rate(zoh.f_s_vm, zoh.is_bounce_4x);
        self.sync_immediate_control_state();
        true
    }

    /// Renders a complete stereo buffer in isolated Offline Bounce mode, stripping the 12-sample
    /// PDC lead-in while preserving all live [`SonaVm`], [`HostShellPatchbay`], tempo, and preview
    /// state on `self` (`0` runtime heap allocations).
    pub fn render_offline_bounce(
        &mut self,
        left_in: &[f32],
        right_in: &[f32],
        left_out: &mut [f32],
        right_out: &mut [f32],
    ) {
        let bounce_tier = if self.preview_mode {
            Tier::Lite
        } else if self.maximize_tier_on_bounce {
            self.max_available_tier.min(self.max_supported_tier)
        } else {
            self.tier
                .min(self.max_available_tier)
                .min(self.max_supported_tier)
        };

        self.bounce_vm.reset();
        let _ = self.bounce_vm.set_offline_bounce(true);
        let _ = self.bounce_vm.switch_tier(bounce_tier);

        // Copy live physical control positions into isolated bounce VM (`0` mutation on live VM).
        if let (Some(live_bus), Some(bounce_bus)) = (
            self.vm.slab().cold().ctrl_bus.as_ref(),
            self.bounce_vm.slab_mut().cold_mut().ctrl_bus.as_mut(),
        ) {
            let n = live_bus
                .num_controls
                .min(bounce_bus.num_controls)
                .min(live_bus.backing.raw_values.len())
                .min(bounce_bus.backing.raw_values.len());
            bounce_bus.backing.raw_values[..n].copy_from_slice(&live_bus.backing.raw_values[..n]);
            bounce_bus.is_dirty = true;
        }
        latch_immediate_at_init_or_reset(self.bounce_vm.slab_mut());

        let bzoh = self.bounce_vm.slab().cold().zoh;
        self.bounce_patchbay.ms_width = self.patchbay.ms_width;
        self.bounce_patchbay.routing_mode = self.patchbay.routing_mode;
        self.bounce_patchbay.wdw_mode = self.patchbay.wdw_mode;
        self.bounce_patchbay.wdw_enabled = self.patchbay.wdw_enabled;
        self.bounce_patchbay
            .reconfigure_sample_rate(bzoh.f_s_vm, bzoh.is_bounce_4x);
        self.bounce_patchbay.reset();

        let n = left_in
            .len()
            .min(right_in.len())
            .min(left_out.len())
            .min(right_out.len());
        let pdc = OFFLINE_BOUNCE_PDC_SAMPLES;
        let mut sink_l = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
        let mut sink_r = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];
        let zeros = [0.0_f32; OFFLINE_BOUNCE_PDC_SAMPLES];

        let mut bounce_preview_guard = self.preview_guard.clone();
        if self.preview_mode {
            bounce_preview_guard.reset();
        }
        let mut dummy_sat = 0_u32;
        let mut dummy_drive = 0.0_f32;
        let mut params = StereoShellParams {
            bypassed: self.bypassed,
            trails_enabled: self.trails_enabled,
            single_input: self.single_input,
            stereo_width: self.bounce_patchbay.ms_width,
            mix_dry_gain: self.mix_dry_gain,
            mix_wet_gain: self.mix_wet_gain,
            wet_polarity_inverted: self.json_wet_polarity_inverted(),
            preview_mode: self.preview_mode,
            preview_guard: if self.preview_mode {
                Some(&mut bounce_preview_guard)
            } else {
                None
            },
            saturation_flags: &mut dummy_sat,
            last_block_max_drive: &mut dummy_drive,
        };

        if n <= pdc {
            process_stereo_vm_block(
                &mut self.bounce_vm,
                &left_in[..n],
                &right_in[..n],
                &mut sink_l[..n],
                &mut sink_r[..n],
                &mut params,
            );
            let rem = pdc - n;
            if rem > 0 {
                process_stereo_vm_block(
                    &mut self.bounce_vm,
                    &zeros[..rem],
                    &zeros[..rem],
                    &mut sink_l[..rem],
                    &mut sink_r[..rem],
                    &mut params,
                );
            }
            process_stereo_vm_block(
                &mut self.bounce_vm,
                &zeros[..n],
                &zeros[..n],
                &mut left_out[..n],
                &mut right_out[..n],
                &mut params,
            );
        } else {
            process_stereo_vm_block(
                &mut self.bounce_vm,
                &left_in[..pdc],
                &right_in[..pdc],
                &mut sink_l,
                &mut sink_r,
                &mut params,
            );
            let body_len = n - pdc;
            process_stereo_vm_block(
                &mut self.bounce_vm,
                &left_in[pdc..n],
                &right_in[pdc..n],
                &mut left_out[..body_len],
                &mut right_out[..body_len],
                &mut params,
            );
            process_stereo_vm_block(
                &mut self.bounce_vm,
                &zeros,
                &zeros,
                &mut left_out[body_len..n],
                &mut right_out[body_len..n],
                &mut params,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::DnamJsonHeader;
    use crate::test_support::build_test_dnam_container;
    use dnam_vm::math::ScalarBackend;

    #[test]
    fn test_render_offline_bounce_preserves_live_state_and_toggles_bounce_cleanly() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Ultra, true);

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

        let tick_before = engine.vm.slab().lite.sample_tick;
        let (mut b_l, mut b_r) = ([0.0_f32; 32], [0.0_f32; 32]);
        engine.render_offline_bounce(&in_l, &in_r, &mut b_l, &mut b_r);
        assert_eq!(engine.vm.slab().lite.sample_tick, tick_before);
        assert!(!engine.offline_bounce);
        assert!(!engine.zoh().is_bounce_4x);
        assert!(b_l.iter().all(|v| v.is_finite()) && b_r.iter().all(|v| v.is_finite()));

        let (mut out_l2, mut out_r2) = ([0.0_f32; 32], [0.0_f32; 32]);
        let (mut ref_l2, mut ref_r2) = ([0.0_f32; 32], [0.0_f32; 32]);
        engine.process_block(&in_l, &in_r, &mut out_l2, &mut out_r2);
        ref_engine.process_block(&in_l, &in_r, &mut ref_l2, &mut ref_r2);
        for i in 0..32 {
            assert_eq!(out_l2[i].to_bits(), ref_l2[i].to_bits());
            assert_eq!(out_r2[i].to_bits(), ref_r2[i].to_bits());
        }

        engine.set_offline_bounce(true);
        assert!(engine.zoh().is_bounce_4x);
        assert_eq!(engine.latency_samples(), OFFLINE_BOUNCE_PDC_SAMPLES);
        engine.set_offline_bounce(false);
        assert!(!engine.zoh().is_bounce_4x);
        assert_eq!(engine.latency_samples(), 0);
    }

    #[test]
    fn test_render_offline_bounce_four_polarity_truth_table() {
        use crate::test_support::build_spatial_test_dnam_container_with_dnbn_polarity;

        let render_case = |dnbn_inv: bool, json_inv: bool| {
            let mut hdr = DnamJsonHeader::default_reference();
            hdr.io_config.capabilities.wet_polarity_inverted = json_inv;
            let bytes = build_spatial_test_dnam_container_with_dnbn_polarity(
                &mut hdr,
                Tier::Lite,
                dnbn_inv,
            );
            let mut eng =
                DnamEngine::<ScalarBackend>::from_bytes(&bytes, Tier::Lite, 48_000).unwrap();
            let in_l = [0.35_f32; 32];
            let in_r = [-0.2_f32; 32];
            let (mut out_l, mut out_r) = ([0.0_f32; 32], [0.0_f32; 32]);
            eng.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
            (out_l, out_r)
        };

        let (l_ff, r_ff) = render_case(false, false);
        let (l_tf, r_tf) = render_case(true, false);
        let (l_ft, r_ft) = render_case(false, true);
        let (l_tt, r_tt) = render_case(true, true);
        assert!(l_ff.iter().any(|&v| v.abs() > 1.0e-4));
        for i in 0..32 {
            assert!((l_tf[i] + l_ff[i]).abs() < 1.0e-5);
            assert!((l_ft[i] + l_ff[i]).abs() < 1.0e-5);
            assert!((l_tt[i] + l_ff[i]).abs() < 1.0e-5);
            assert!((r_tf[i] + r_ff[i]).abs() < 1.0e-5);
            assert!((r_ft[i] + r_ff[i]).abs() < 1.0e-5);
            assert!((r_tt[i] + r_ff[i]).abs() < 1.0e-5);
        }
    }
}
