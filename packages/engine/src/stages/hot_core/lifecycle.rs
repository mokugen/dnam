//! Allocation-free construction, live tier switching, ZOH rescaling, ControlBus dispatch,
//! and DC-equilibrium reset for [`V3HotCore`].

use core::marker::PhantomData;

use crate::container::ZohRescaledParams;
use crate::math::MathBackend;
use crate::stages::hot_core::{V3HotCore, V3PostStageConditioner};
use crate::stages::router::V3Stage0Hot;
use crate::stages::stage1_svf::{V3Stage1SvfState, exp2_signed_oct_f32};
use crate::stages::stage2_gru::{
    MAX_GROUPS, MAX_GROUPS_PER_BLOCK, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS, V3ControlBusState,
    V3GruBlockHot, V3GruChannelState, V3InterBlockToneSvfState, V3StereoOversampler2x,
    flat_gru_lane_index, tier_block_groups_array, tier_fir_segments, tier_kautz_banks,
};
use crate::stages::stage3_cab::{V3ColdTierModel, V3Stage3Hot, active_hot_bytes_for_tier};
use crate::stages::stage4_tank::v3_fdn::{V3Stage4TankHot, V3Stage4TankParams};
use crate::types::Tier;

fn pack_readout_weights(
    src: &[f32; MAX_TOTAL_GRU_LANES],
    block_groups: &[usize; NUM_GRU_BLOCKS],
) -> [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS] {
    let mut dst = [[[0.0_f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS];
    for (b, dst_blk) in dst.iter_mut().enumerate() {
        for (g, dst_grp) in dst_blk
            .iter_mut()
            .enumerate()
            .take(block_groups[b].min(MAX_GROUPS))
        {
            for (l, dst_lane) in dst_grp.iter_mut().enumerate() {
                *dst_lane = src[flat_gru_lane_index(b, g, l)];
            }
        }
    }
    dst
}

impl<B: MathBackend> V3HotCore<B> {
    /// Returns the active hot working-set bytes read/written per host sample for `tier`.
    #[inline]
    #[must_use]
    pub const fn active_hot_bytes(tier: Tier, os2x: bool) -> usize {
        active_hot_bytes_for_tier(tier, os2x)
    }

    /// Compiles a cold [`V3ColdTierModel`] into an L1D-packed [`V3HotCore<B>`].
    #[inline(never)]
    #[must_use]
    pub fn from_cold_model(model: &V3ColdTierModel) -> Self {
        Self::from_cold_model_with_stage4_cap(model, 0)
    }

    /// Compiles a cold [`V3ColdTierModel`] into [`V3HotCore<B>`], pre-sizing Stage IV's extended
    /// delay history buffer when `stage4_min_cap > V3_TANK_CAP` (`0` RT/switch/reload allocations).
    #[inline(never)]
    #[must_use]
    pub fn from_cold_model_with_stage4_cap(model: &V3ColdTierModel, stage4_min_cap: usize) -> Self {
        let block_groups = tier_block_groups_array(model.tier);
        let num_groups = block_groups[0];
        let num_blocks = model.num_blocks.clamp(1, NUM_GRU_BLOCKS);
        let mut os_params = model.os_params;
        os_params.clamp_in_place();
        let mut svf_params = model.svf;
        svf_params.direct_taps_l = model.readout.svf_direct_taps_l;
        svf_params.direct_taps_r = model.readout.svf_direct_taps_r;
        svf_params.clamp_in_place();
        let mut feat_params = model.feat_params;
        feat_params.clamp_in_place();

        let blocks = core::array::from_fn(|b| {
            V3GruBlockHot::from_cold::<B>(&model.blocks[b], MAX_GROUPS_PER_BLOCK[b])
        });
        let ctrl_bus =
            V3ControlBusState::from_cold::<B>(&model.ctrl, &blocks, num_blocks, &block_groups);
        let mut tone_svf = V3InterBlockToneSvfState::from_cold(&model.tone_svf);
        tone_svf.refresh_3khz(
            ctrl_bus.tone_fc_oct_delta,
            ctrl_bus.tone_k_delta,
            ctrl_bus.tone_eq_delta,
        );

        let mut stage4 = V3Stage4TankHot::from_cold_with_min_cap(&model.stage4, stage4_min_cap);
        stage4.refresh_3khz(ctrl_bus.tank_psi_delta);

        Self {
            num_groups,
            block_groups,
            num_blocks,
            num_fir_segments: tier_fir_segments(model.tier),
            num_kautz_banks: tier_kautz_banks(model.tier),
            tier: model.tier,
            os2x: model.os2x,
            mono_out: model.mono_out,
            post_cond_active: false,
            sample_tick: 0,
            last_in: [0.0; 2],
            last_s2_dir: [0.0; 4],
            last_stage_targets: [0.0; 16],
            stage0: V3Stage0Hot::from_cold(&model.stage0),
            os_params,
            svf_params,
            svf_state: V3Stage1SvfState::default(),
            os_stereo: V3StereoOversampler2x::default(),
            feat_params,
            blocks,
            state_l: V3GruChannelState::default(),
            state_r: V3GruChannelState::default(),
            tone_svf,
            pre_cab_u_gain: model.readout.pre_cab_u_gain,
            pre_cab_hp_gain: model.readout.pre_cab_hp_gain,
            direct_s2_u_gain: model.readout.direct_s2_u_gain,
            direct_s2_hp_gain: model.readout.direct_s2_hp_gain,
            pre_cab_w_l: pack_readout_weights(&model.readout.pre_cab_w[0], &block_groups),
            pre_cab_w_r: pack_readout_weights(&model.readout.pre_cab_w[1], &block_groups),
            direct_s2_w_l: pack_readout_weights(&model.readout.direct_s2_w[0], &block_groups),
            direct_s2_w_r: pack_readout_weights(&model.readout.direct_s2_w[1], &block_groups),
            stage3: V3Stage3Hot::from_cold(model),
            stage4,
            ctrl_bus,
            post_cond: V3PostStageConditioner::new_48k(),
            _backend: PhantomData,
        }
    }

    /// Switches active Matryoshka tier, sampling mode, and readout weights in-place (`0` alloc).
    pub fn switch_tier(&mut self, model: &V3ColdTierModel) {
        self.switch_tier_inner(model, false);
    }

    /// Reloads a complete cold model into this hot core in-place (`0` allocs, `0` drops).
    pub fn reload_in_place(&mut self, model: &V3ColdTierModel) {
        self.switch_tier_inner(model, true);
        self.stage3.onset_delay_samples = 16;
        self.stage4.reload_in_place(&model.stage4);
        self.post_cond.reload_48k();
        self.post_cond_active = false;
        self.reset_to_dc_equilibrium();
        self.tone_svf.refresh_3khz(
            self.ctrl_bus.tone_fc_oct_delta,
            self.ctrl_bus.tone_k_delta,
            self.ctrl_bus.tone_eq_delta,
        );
        self.stage4.refresh_3khz(self.ctrl_bus.tank_psi_delta);
    }

    /// Pre-allocates Stage IV cold extended delay storage outside audited RT/reconfigure call graphs.
    pub fn ensure_cold_stage4_cap(&mut self, params: &V3Stage4TankParams, min_frames_cap: usize) {
        self.stage4.ensure_cold_ext_cap(params, min_frames_cap);
    }

    fn switch_tier_inner(&mut self, model: &V3ColdTierModel, force_rebuild_blocks: bool) {
        let old_bg = self.block_groups;
        let old_blocks = self.num_blocks;
        let old_os2x = self.os2x;
        let new_bg = tier_block_groups_array(model.tier);
        let new_groups = new_bg[0];
        let new_blocks = model.num_blocks.clamp(1, NUM_GRU_BLOCKS);

        for b in 0..old_blocks.min(new_blocks) {
            if new_bg[b] > old_bg[b] {
                for g in old_bg[b]..new_bg[b] {
                    self.state_l.h[b][g] = [0.0; 4];
                    self.state_r.h[b][g] = [0.0; 4];
                }
            }
        }
        if new_blocks > old_blocks {
            for b in old_blocks..new_blocks {
                self.state_l.h[b] = [[0.0; 4]; MAX_GROUPS];
                self.state_r.h[b] = [[0.0; 4]; MAX_GROUPS];
            }
        }
        if !old_os2x && model.os2x {
            self.os_stereo.warm_seed(self.last_in, self.last_s2_dir);
        }

        self.tier = model.tier;
        self.block_groups = new_bg;
        self.num_groups = new_groups;
        self.num_blocks = new_blocks;
        self.num_fir_segments = tier_fir_segments(model.tier);
        self.num_kautz_banks = tier_kautz_banks(model.tier);
        self.os2x = model.os2x;
        self.mono_out = model.mono_out;

        self.stage0.update_params(&model.stage0);
        let mut os_params = model.os_params;
        os_params.clamp_in_place();
        self.os_params = os_params;

        let mut svf_params = model.svf;
        svf_params.direct_taps_l = model.readout.svf_direct_taps_l;
        svf_params.direct_taps_r = model.readout.svf_direct_taps_r;
        svf_params.clamp_in_place();
        self.svf_params = svf_params;

        let mut feat_params = model.feat_params;
        feat_params.clamp_in_place();
        self.feat_params = feat_params;

        if force_rebuild_blocks || old_os2x != model.os2x {
            self.blocks = core::array::from_fn(|b| {
                V3GruBlockHot::from_cold::<B>(&model.blocks[b], MAX_GROUPS_PER_BLOCK[b])
            });
        }

        self.tone_svf.update_params(&model.tone_svf);
        let was_frozen = self.stage4.freeze;
        self.stage4.update_params(&model.stage4);
        self.stage4.freeze = was_frozen;
        self.ctrl_bus
            .update_params::<B>(&model.ctrl, &self.blocks, new_blocks, &new_bg);
        self.tone_svf.refresh_3khz(
            self.ctrl_bus.tone_fc_oct_delta,
            self.ctrl_bus.tone_k_delta,
            self.ctrl_bus.tone_eq_delta,
        );
        self.stage4.refresh_3khz(self.ctrl_bus.tank_psi_delta);

        self.pre_cab_u_gain = model.readout.pre_cab_u_gain;
        self.pre_cab_hp_gain = model.readout.pre_cab_hp_gain;
        self.direct_s2_u_gain = model.readout.direct_s2_u_gain;
        self.direct_s2_hp_gain = model.readout.direct_s2_hp_gain;
        self.pre_cab_w_l = pack_readout_weights(&model.readout.pre_cab_w[0], &new_bg);
        self.pre_cab_w_r = pack_readout_weights(&model.readout.pre_cab_w[1], &new_bg);
        self.direct_s2_w_l = pack_readout_weights(&model.readout.direct_s2_w[0], &new_bg);
        self.direct_s2_w_r = pack_readout_weights(&model.readout.direct_s2_w[1], &new_bg);
        self.stage3
            .update_readout(&model.readout, &model.kautz_poles);
        self.last_stage_targets = [0.0; 16];
    }

    /// Reconfigures Stages 0–IV physical time constants, poles, FIR taps, and delays under continuous-time ZOH.
    pub fn reconfigure_with_zoh(&mut self, base: &V3ColdTierModel, host_zoh: &ZohRescaledParams) {
        let s2_zoh = ZohRescaledParams::new(48_000, host_zoh.sample_rate_host_hz);
        self.stage0.reconfigure_with_zoh(&base.stage0, host_zoh);

        let mut svf = base.svf;
        svf.direct_taps_l = base.readout.svf_direct_taps_l;
        svf.direct_taps_r = base.readout.svf_direct_taps_r;
        svf.rescale_with_zoh(&s2_zoh);
        self.svf_params = svf;

        let mut feat = base.feat_params;
        feat.rescale_with_zoh(&s2_zoh);
        self.feat_params = feat;

        self.blocks = core::array::from_fn(|b| {
            let mut blk_cold = base.blocks[b];
            blk_cold.rescale_with_zoh(&s2_zoh);
            V3GruBlockHot::from_cold::<B>(&blk_cold, MAX_GROUPS_PER_BLOCK[b])
        });

        let eff_sr_hz = (host_zoh.sample_rate_host_hz as f32) * if self.os2x { 2.0 } else { 1.0 };
        self.tone_svf.update_params(&base.tone_svf);
        self.tone_svf.reconfigure_sample_rate(eff_sr_hz);
        self.post_cond
            .reconfigure_sample_rate(host_zoh.sample_rate_host_hz as f32);

        self.stage3.reconfigure_with_zoh(
            &base.readout,
            &base.kautz_poles,
            self.num_fir_segments,
            host_zoh,
        );
        self.stage4.reconfigure_with_zoh(&base.stage4, host_zoh);

        self.ctrl_bus.update_params::<B>(
            &base.ctrl,
            &self.blocks,
            self.num_blocks,
            &self.block_groups,
        );
        self.tone_svf.refresh_3khz(
            self.ctrl_bus.tone_fc_oct_delta,
            self.ctrl_bus.tone_k_delta,
            self.ctrl_bus.tone_eq_delta,
        );
        self.stage4.refresh_3khz(self.ctrl_bus.tank_psi_delta);
        self.last_stage_targets = [0.0; 16];
    }

    fn apply_stage_targets_impl(&mut self, stage_targets: &[f32; 16], immediate: bool) {
        if !immediate {
            let mut max_diff = 0.0_f32;
            for i in 0..16 {
                max_diff = max_diff.max((stage_targets[i] - self.last_stage_targets[i]).abs());
            }
            let target_psi = (self.stage4.params.base_psi + self.ctrl_bus.tank_psi_delta)
                .clamp(-core::f32::consts::PI, core::f32::consts::PI);
            if max_diff <= 1.0e-7
                && !self.stage3.kautz_slew_active
                && (target_psi - self.stage4.current_psi).abs() <= 1.0e-7
            {
                return;
            }
        }
        self.last_stage_targets = *stage_targets;

        self.ctrl_bus.env_fc_scale = exp2_signed_oct_f32(stage_targets[0].clamp(-3.0, 3.0));
        self.ctrl_bus.env_k_delta = stage_targets[1].clamp(-0.8, 1.5);
        self.ctrl_bus.apply_direct_block_film::<B>(
            self.num_blocks,
            &self.block_groups,
            &self.blocks,
            [stage_targets[2], stage_targets[3], stage_targets[4]],
            [stage_targets[5], stage_targets[6], stage_targets[7]],
        );

        let kautz_oct = stage_targets[8].clamp(-1.5, 1.5);
        let kautz_damp = stage_targets[15].clamp(-1.5, 1.5);
        if immediate {
            self.stage3
                .snap_kautz_immediate::<B>(self.num_kautz_banks, kautz_oct, kautz_damp);
        } else {
            self.stage3
                .retune_kautz_3khz::<B>(self.num_kautz_banks, kautz_oct, kautz_damp);
        }

        let tone_svf_active = self.tone_svf.params.cab_inject_w.abs() > 1.0e-7
            || self.tone_svf.params.b1_inject_w.abs() > 1.0e-7;
        if tone_svf_active {
            self.ctrl_bus.tone_fc_oct_delta = stage_targets[9].clamp(-3.0, 3.0);
            self.ctrl_bus.tone_k_delta = stage_targets[10].clamp(-1.0, 2.0);
            self.ctrl_bus.tone_eq_delta = [
                stage_targets[12].clamp(-1.5, 1.5),
                stage_targets[13].clamp(-1.5, 1.5),
                stage_targets[14].clamp(-1.5, 1.5),
            ];
            self.tone_svf.refresh_3khz(
                self.ctrl_bus.tone_fc_oct_delta,
                self.ctrl_bus.tone_k_delta,
                self.ctrl_bus.tone_eq_delta,
            );
            self.post_cond_active = self.post_cond.refresh_3khz(0.0, 0.0, 0.0, 0.0, 0.0, false);
        } else {
            self.post_cond_active = self.post_cond.refresh_3khz(
                stage_targets[9],
                stage_targets[10],
                stage_targets[12],
                stage_targets[13],
                stage_targets[14],
                true,
            );
        }

        self.ctrl_bus.tank_psi_delta =
            stage_targets[11].clamp(-core::f32::consts::PI, core::f32::consts::PI);
        if immediate {
            self.stage4.snap_psi_immediate(self.ctrl_bus.tank_psi_delta);
        } else {
            self.stage4.refresh_3khz(self.ctrl_bus.tank_psi_delta);
        }
    }

    /// Snaps `stage_targets[0..16]` across Stages I–IV immediately without slew lag or pole contraction.
    #[inline]
    pub fn snap_control_bus_stage_targets(&mut self, stage_targets: &[f32; 16]) {
        self.apply_stage_targets_impl(stage_targets, true);
    }

    /// Applies unified `3 kHz` `ControlBus::stage_targets[0..16]` across Stages I–IV with
    /// zero-cost early exit when modulation targets are stationary (`0` heap alloc, `0 FDIV`).
    #[inline]
    pub fn apply_control_bus_stage_targets(&mut self, stage_targets: &[f32; 16]) {
        self.apply_stage_targets_impl(stage_targets, false);
    }

    /// Resets all recursive filter, envelope, oversampler, GRU, FIR, and tank states to exact `+0.0`.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.stage0.reset();
        self.tone_svf.reset();
        self.svf_state.reset();
        self.os_stereo.reset();
        self.state_l.reset();
        self.state_r.reset();
        self.stage3.reset();
        self.stage3
            .snap_kautz_immediate::<B>(self.num_kautz_banks, 0.0, 0.0);
        self.stage4.reset();
        self.post_cond.reset();
        self.post_cond_active = false;
        self.sample_tick = 0;
        self.last_in = [0.0; 2];
        self.last_s2_dir = [0.0; 4];
        self.last_stage_targets = [0.0; 16];
    }
}
