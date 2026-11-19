//! Cache-Coherent SIMD Hot Execution Core (`Zero-Lag 2x/1x` -> `Stage I Env-SVF` ->
//! `Stage II 3-Block Tapered Zero-DC Matryoshka GRU` -> `Stage III FIR + Kautz` -> `Stage IV FDN`).

#![forbid(unsafe_code)]

pub mod instance;
pub mod lifecycle;
pub mod post_cond;

pub use instance::{V3CoreInstance, v3_process_block_default, v3_process_block_infallible_default};
pub use post_cond::V3PostStageConditioner;

use core::marker::PhantomData;

use crate::math::{DefaultMathBackend, MathBackend};
use crate::stages::router::V3Stage0Hot;
use crate::stages::stage1_svf::{V3Stage1SvfParams, V3Stage1SvfState};
use crate::stages::stage2_gru::{
    MAX_GROUPS, NUM_GRU_BLOCKS, Stage2StepContext, V3ControlBusState, V3GruBlockHot,
    V3GruChannelState, V3InputFeaturesParams, V3InterBlockToneSvfState, V3OversamplerParams,
    V3StereoOversampler2x,
    kernels::{self, V3HotScratch},
    step_stage2_stereo_with_lp2_vec4,
};
use crate::stages::stage3_cab::V3Stage3Hot;
use crate::stages::stage4_tank::v3_fdn::V3Stage4TankHot;
use crate::types::Tier;

/// Cache-line aligned hot execution core for `v3` Matryoshka models (`Lite..Ultra`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct V3HotCore<B: MathBackend = DefaultMathBackend> {
    pub num_groups: usize,
    pub block_groups: [usize; NUM_GRU_BLOCKS],
    pub num_blocks: usize,
    pub num_fir_segments: usize,
    pub num_kautz_banks: usize,
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub post_cond_active: bool,
    pub sample_tick: u32,
    pub last_in: [f32; 2],
    pub last_s2_dir: [f32; 4],
    pub last_stage_targets: [f32; 16],
    pub stage0: V3Stage0Hot,
    pub os_params: V3OversamplerParams,
    pub svf_params: V3Stage1SvfParams,
    pub svf_state: V3Stage1SvfState,
    pub os_stereo: V3StereoOversampler2x,
    pub feat_params: V3InputFeaturesParams,
    pub blocks: [V3GruBlockHot; NUM_GRU_BLOCKS],
    pub state_l: V3GruChannelState,
    pub state_r: V3GruChannelState,
    pub tone_svf: V3InterBlockToneSvfState,
    pub pre_cab_u_gain: [f32; 2],
    pub pre_cab_hp_gain: [f32; 2],
    pub direct_s2_u_gain: [f32; 2],
    pub direct_s2_hp_gain: [f32; 2],
    pub pre_cab_w_l: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub pre_cab_w_r: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub direct_s2_w_l: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub direct_s2_w_r: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub stage3: V3Stage3Hot,
    pub stage4: V3Stage4TankHot,
    /// Trailing `3 kHz` control bus (`params` live at tail to minimize hot L1D footprint).
    pub ctrl_bus: V3ControlBusState,
    /// Trailing post-Stage-3 broadband gain & 4-band TPT conditioner (`post_cond_active`-gated).
    pub post_cond: V3PostStageConditioner,
    _backend: PhantomData<B>,
}

impl<B: MathBackend> V3HotCore<B> {
    #[inline(never)]
    pub(crate) fn refresh_3khz_tick(&mut self) {
        if !self.ctrl_bus.dirty {
            return;
        }
        self.ctrl_bus
            .step_3khz::<B>(self.num_blocks, &self.block_groups, &self.blocks);
        self.tone_svf.refresh_3khz(
            self.ctrl_bus.tone_fc_oct_delta,
            self.ctrl_bus.tone_k_delta,
            self.ctrl_bus.tone_eq_delta,
        );
        self.stage4.refresh_3khz(self.ctrl_bus.tank_psi_delta);
        let target_psi = (self.stage4.params.base_psi + self.ctrl_bus.tank_psi_delta)
            .clamp(-core::f32::consts::PI, core::f32::consts::PI);
        if (target_psi - self.stage4.current_psi).abs() > 1.0e-7 {
            self.ctrl_bus.dirty = true;
        }
    }

    #[inline(always)]
    fn step_stage2_stereo(
        &mut self,
        u_s2_l: f32,
        u_s2_r: f32,
        lp2: [f32; 2],
    ) -> (f32, f32, f32, f32) {
        let ctx = Stage2StepContext {
            blocks: &self.blocks,
            num_blocks: self.num_blocks,
            block_groups: &self.block_groups,
            feat_params: &self.feat_params,
            ctrl_bus: &self.ctrl_bus,
            pre_cab_u_gain: &self.pre_cab_u_gain,
            pre_cab_hp_gain: &self.pre_cab_hp_gain,
            direct_s2_u_gain: &self.direct_s2_u_gain,
            direct_s2_hp_gain: &self.direct_s2_hp_gain,
            pre_cab_w_l: &self.pre_cab_w_l,
            pre_cab_w_r: &self.pre_cab_w_r,
            direct_s2_w_l: &self.direct_s2_w_l,
            direct_s2_w_r: &self.direct_s2_w_r,
        };
        step_stage2_stereo_with_lp2_vec4::<B>(
            &ctx,
            &mut self.state_l,
            &mut self.state_r,
            &mut self.tone_svf,
            u_s2_l,
            u_s2_r,
            lp2,
            self.svf_params.feat_lp2_mix,
        )
    }

    #[inline(always)]
    fn step_stage0_and_tick(&mut self, in_l: f32, in_r: f32) -> [f32; 4] {
        let xl = if in_l.is_finite() {
            in_l.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        let keep_in_r = !self.mono_out || self.feat_params.key_link > 0.0;
        let xr = if !keep_in_r {
            xl
        } else if in_r.is_finite() {
            in_r.clamp(-1.0e9, 1.0e9)
        } else {
            0.0
        };
        self.last_in = [xl, xr];
        let s0 = self.stage0.step_stereo(xl, xr);
        let tick = self.sample_tick;
        self.sample_tick = tick.wrapping_add(1);
        if (tick & 15) == 0 && self.ctrl_bus.dirty {
            self.refresh_3khz_tick();
        }
        s0
    }

    #[inline(always)]
    fn step_stage1_and_stage2(&mut self, x_l: f32, x_r: f32) -> [f32; 4] {
        let ([u_s2_l, u_s2_r], [svf_l, svf_r], taps) =
            self.svf_state.step_stereo_fullrate_with_taps::<B>(
                &self.svf_params,
                x_l,
                x_r,
                self.ctrl_bus.env_fc_scale,
                self.ctrl_bus.env_k_delta,
            );
        let (s2_l, s2_r, d2_l, d2_r) =
            self.step_stage2_stereo(u_s2_l, u_s2_r, [taps[0][4], taps[1][4]]);
        [s2_l, s2_r, d2_l + svf_l, d2_r + svf_r]
    }

    #[inline(always)]
    fn step_post_stage2(
        &mut self,
        s2_l: f32,
        s2_r: f32,
        dir_l: f32,
        dir_r: f32,
        clean_sub_l: f32,
        clean_sub_r: f32,
    ) -> [f32; 2] {
        self.last_s2_dir = [s2_l, s2_r, dir_l, dir_r];
        let [cab_l, cab_r] = self.stage3.step_stereo::<B>(
            s2_l,
            s2_r,
            self.num_fir_segments,
            self.num_kautz_banks,
            self.mono_out,
        );
        let pre_tank_l = dir_l + cab_l + clean_sub_l;
        let pre_tank_r = if self.mono_out {
            pre_tank_l
        } else {
            dir_r + cab_r + clean_sub_r
        };
        let [out_l, out_r] = self.stage4.step_stereo::<B>(pre_tank_l, pre_tank_r);
        let [out_l, out_r] = if self.post_cond_active {
            self.post_cond.step_stereo(out_l, out_r, self.mono_out)
        } else {
            [out_l, out_r]
        };
        if self.mono_out {
            [out_l, out_l]
        } else {
            [out_l, out_r]
        }
    }

    /// Steps one 4-port (`4CM` / `WDW` / `Sidechain`) stereo frame preserving strict `S_{ret->send} == 0`.
    #[inline]
    pub fn step_4port_frame(
        &mut self,
        router: &mut crate::stages::Stage0Router,
        in_l: f32,
        in_r: f32,
        ret_l: f32,
        ret_r: f32,
        has_return: bool,
    ) -> ([f32; 2], [f32; 2], [f32; 2]) {
        let [s0_l, s0_r, clean_sub_l, clean_sub_r] = self.step_stage0_and_tick(in_l, in_r);
        let _ = router.route_frame_no_return(in_l, in_r);
        let send_l = router.s_matrix[0][0] * s0_l + router.s_matrix[0][1] * s0_r;
        let send_r = router.s_matrix[1][0] * s0_l + router.s_matrix[1][1] * s0_r;
        let (x_l, x_r) = if has_return {
            let ret_l_a = router.return_aligner_l.step(ret_l);
            let ret_r_a = router.return_aligner_r.step(ret_r);
            let s = &router.s_matrix;
            (
                s0_l + s[2][2] * ret_l_a + s[2][3] * ret_r_a,
                s0_r + s[3][2] * ret_l_a + s[3][3] * ret_r_a,
            )
        } else {
            (s0_l, s0_r)
        };
        let [s2_l, s2_r, dir_l, dir_r] = if self.os2x {
            let (up0, up1) = self
                .os_stereo
                .upsample_stereo::<B>(x_l, x_r, &self.os_params);
            let sub0 = self.step_stage1_and_stage2(up0[0], up0[1]);
            let sub1 = self.step_stage1_and_stage2(up1[0], up1[1]);
            self.os_stereo
                .downsample_4ch::<B>(sub0, sub1, &self.os_params)
        } else {
            self.step_stage1_and_stage2(x_l, x_r)
        };
        self.last_s2_dir = [s2_l, s2_r, dir_l, dir_r];
        let [cab_l, cab_r] = self.stage3.step_stereo::<B>(
            s2_l,
            s2_r,
            self.num_fir_segments,
            self.num_kautz_banks,
            self.mono_out,
        );
        let core_l = dir_l + cab_l + clean_sub_l;
        let core_r = if self.mono_out {
            core_l
        } else {
            dir_r + cab_r + clean_sub_r
        };
        let [wet_l, wet_r] = self.stage4.step_stereo::<B>(core_l, core_r);
        let [wet_l, wet_r] = if self.post_cond_active {
            self.post_cond.step_stereo(wet_l, wet_r, self.mono_out)
        } else {
            [wet_l, wet_r]
        };
        let out = if self.mono_out {
            [wet_l, wet_l]
        } else {
            [wet_l, wet_r]
        };
        (out, [core_l, core_r], [send_l, send_r])
    }

    #[inline(always)]
    fn step_1x_frame(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let [s0_l, s0_r, clean_sub_l, clean_sub_r] = self.step_stage0_and_tick(in_l, in_r);
        let [s2_l, s2_r, dir_l, dir_r] = self.step_stage1_and_stage2(s0_l, s0_r);
        self.step_post_stage2(s2_l, s2_r, dir_l, dir_r, clean_sub_l, clean_sub_r)
    }

    #[inline(always)]
    fn step_2x_frame(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let [s0_l, s0_r, clean_sub_l, clean_sub_r] = self.step_stage0_and_tick(in_l, in_r);
        let (up0, up1) = self
            .os_stereo
            .upsample_stereo::<B>(s0_l, s0_r, &self.os_params);
        let sub0 = self.step_stage1_and_stage2(up0[0], up0[1]);
        let sub1 = self.step_stage1_and_stage2(up1[0], up1[1]);
        let [s2_l, s2_r, dir_l, dir_r] =
            self.os_stereo
                .downsample_4ch::<B>(sub0, sub1, &self.os_params);
        self.step_post_stage2(s2_l, s2_r, dir_l, dir_r, clean_sub_l, clean_sub_r)
    }

    #[inline(always)]
    pub(crate) fn process_block_1x(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        for i in 0..n {
            let [yl, yr] = self.step_1x_frame(in_l[i], in_r[i]);
            out_l[i] = yl;
            out_r[i] = yr;
        }
    }

    #[inline(always)]
    pub(crate) fn process_block_2x(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        for i in 0..n {
            let [yl, yr] = self.step_2x_frame(in_l[i], in_r[i]);
            out_l[i] = yl;
            out_r[i] = yr;
        }
    }

    /// Processes a single stereo sample pair at host rate (`48 kHz`) across all 5 unconditional stages.
    /// Direct callers must hold [`crate::FtzDazGuard`] for the enclosing audio block.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        if self.os2x {
            self.step_2x_frame(in_l, in_r)
        } else {
            self.step_1x_frame(in_l, in_r)
        }
    }

    /// Processes a stereo sub-block reusing caller-owned `V3HotScratch` (caller holds [`crate::FtzDazGuard`]).
    #[inline(always)]
    pub(crate) fn process_subblock_no_ftz(
        &mut self,
        sc: &mut V3HotScratch,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        sc.dispatch(self, in_l, in_r, out_l, out_r);
    }

    /// Processes a contiguous stereo block under [`crate::FtzDazGuard`] (`0` alloc, `0 FDIV`).
    #[inline(never)]
    pub fn process_block_stereo(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let _ftz = crate::FtzDazGuard::new();
        match (
            self.num_blocks,
            self.block_groups,
            self.num_fir_segments,
            self.num_kautz_banks,
            self.os2x,
        ) {
            (3, [2, 2, 1], 3, 3, false) => {
                kernels::process_block_const::<B, 2, 2, 3, 3, false, 16, 17>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            (3, [2, 2, 1], 3, 3, true) => {
                kernels::process_block_const::<B, 2, 2, 3, 3, true, 32, 33>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            (3, [1, 1, 1], 1, 1, false) => {
                kernels::process_block_const::<B, 1, 1, 1, 1, false, 16, 17>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            (3, [1, 1, 1], 1, 1, true) => {
                kernels::process_block_const::<B, 1, 1, 1, 1, true, 32, 33>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            (3, [2, 1, 1], 2, 2, false) => {
                kernels::process_block_const::<B, 2, 1, 2, 2, false, 16, 17>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            (3, [2, 1, 1], 2, 2, true) => {
                kernels::process_block_const::<B, 2, 1, 2, 2, true, 32, 33>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            (3, [3, 2, 1], 4, 4, false) => {
                kernels::process_block_const::<B, 3, 2, 4, 4, false, 16, 17>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            (3, [3, 2, 1], 4, 4, true) => {
                kernels::process_block_const::<B, 3, 2, 4, 4, true, 32, 33>(
                    self, in_l, in_r, out_l, out_r,
                )
            }
            _ => {
                if self.os2x {
                    self.process_block_2x(in_l, in_r, out_l, out_r);
                } else {
                    self.process_block_1x(in_l, in_r, out_l, out_r);
                }
            }
        }
    }

    /// Infallible block processing API using Rust `1.100.0` stabilized never type `!`.
    #[inline(never)]
    pub fn process_block_infallible(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), !> {
        self.process_block_stereo(in_l, in_r, out_l, out_r);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ZohRescaledParams;
    use crate::math::ScalarBackend;
    use crate::stages::stage2_gru::{
        MAX_BLOCK_LANES, NUM_IN_FEATS_B0, NUM_IN_FEATS_BN, flat_gru_lane_index, tier_block_groups,
        tier_fir_segments, tier_kautz_banks,
    };
    use crate::stages::stage3_cab::V3MatryoshkaModel;
    use crate::types::Tier;
    use alloc::alloc::Global;
    use proptest::prelude::*;

    fn sample_nontrivial_matryoshka(os2x: bool, mono_out: bool) -> V3MatryoshkaModel {
        let mut m = V3MatryoshkaModel::identity(os2x, mono_out);
        m.os_params.up_pre_emph = 0.14;
        m.os_params.dec_aa_mix = 0.85;
        m.os_params.up_aa_mix = 0.85;
        m.svf.base_g = 0.065;
        m.svf.sec2_g_ratio = 1.45;
        m.svf.env_oct_sens = 2.1;
        m.svf.env_comp = 2.5;
        m.svf.det_hp_mix = 0.35;
        m.svf.damp_k1 = 0.65;
        m.svf.damp_k2 = 0.90;
        m.svf.sec2_bp_feed = 0.25;
        m.svf.stage2_feed_taps[..7].copy_from_slice(&[0.6, 0.25, 0.35, -0.15, 0.20, 0.15, 0.05]);
        m.feat_params.pinch_k = 180.0;
        m.feat_params.slow_det_comp = 6.0;

        for b in 0..NUM_GRU_BLOCKS {
            let n_feats = if b == 0 {
                NUM_IN_FEATS_B0
            } else {
                NUM_IN_FEATS_BN
            };
            for l in 0..MAX_BLOCK_LANES {
                let lf = l as f32;
                for f in 0..n_feats {
                    m.blocks[b].w_in_z[f][l] = 0.12 - 0.02 * (f as f32) + 0.01 * lf;
                    m.blocks[b].w_in_r[f][l] = -0.08 + 0.03 * (f as f32);
                    m.blocks[b].w_in_n[f][l] =
                        (0.65 + 0.08 * lf) * if (f + l) % 2 == 0 { 1.0 } else { -0.7 };
                }
                m.blocks[b].b_z[l] = -1.2 + 0.25 * (l % 4) as f32;
                m.blocks[b].b_r[l] = 1.4 - 0.15 * (l % 4) as f32;
                m.blocks[b].b_n[l] = -0.35 + 0.12 * lf;
                m.blocks[b].k_sag[l] = 0.25 * ((l % 4) as f32);
                m.blocks[b].k_quad[l] = 0.08 * if l % 2 == 0 { 1.0 } else { -0.75 };
                let dst_g = l >> 2;
                for s in 0..MAX_BLOCK_LANES {
                    if (s >> 2) <= dst_g {
                        let scale = if l == s { 0.28 } else { -0.04 };
                        m.blocks[b].w_rec_z[l][s] = 0.3 * scale;
                        m.blocks[b].w_rec_r[l][s] = -0.25 * scale;
                        m.blocks[b].w_rec_n[l][s] = 0.55 * scale;
                        if b > 0 {
                            m.blocks[b].w_prev_z[l][s] = 0.15 * scale;
                            m.blocks[b].w_prev_r[l][s] = 0.20 * scale;
                            m.blocks[b].w_prev_n[l][s] = if l == s { 0.45 } else { 0.06 * scale };
                        }
                    }
                }
            }
        }
        for k in 0..crate::stages::stage2_gru::MAX_KAUTZ_POLES {
            m.kautz_poles[k] = (0.85 + 0.003 * (k as f32), 0.12 + 0.06 * (k as f32));
        }
        for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
            .into_iter()
            .enumerate()
        {
            let r = &mut m.readouts[t_idx];
            r.svf_direct_taps_l[..7].copy_from_slice(&[0.05, 0.02, -0.01, 0.01, 0.03, 0.01, -0.02]);
            r.svf_direct_taps_r[..7].copy_from_slice(&[0.04, 0.03, -0.02, 0.02, 0.02, -0.01, 0.01]);
            r.pre_cab_u_gain = [0.22, 0.19];
            r.pre_cab_hp_gain = [0.06, -0.04];
            r.direct_s2_u_gain = [0.02, -0.02];
            r.direct_s2_hp_gain = [0.03, 0.02];
            for b in 0..NUM_GRU_BLOCKS {
                for g in 0..tier_block_groups(tier, b) {
                    for l in 0..4 {
                        let idx = flat_gru_lane_index(b, g, l);
                        r.pre_cab_w[0][idx] = 0.08 * if l % 2 == 0 { 1.0 } else { -0.8 };
                        r.pre_cab_w[1][idx] = 0.07 * if l % 2 == 0 { 1.0 } else { -0.8 };
                        r.direct_s2_w[0][idx] = 0.015;
                        r.direct_s2_w[1][idx] = -0.012;
                    }
                }
            }
            for i in 0..tier_fir_segments(tier) * 16 {
                r.fir_l[i] = 0.45 * (0.88_f32).powi(i as i32);
                r.fir_r[i] = 0.42 * (0.86_f32).powi(i as i32);
            }
            for k in 0..tier_kautz_banks(tier) * 4 {
                r.kautz_c_l[k] = (0.04 * if k % 2 == 0 { 1.0 } else { -1.0 }, 0.02);
                r.kautz_c_r[k] = (0.035 * if k % 2 == 0 { 1.0 } else { -1.0 }, -0.015);
            }
        }
        m.stage0.op_p_sub_mix = 0.15;
        m.stage0.clean_sub_mix = 0.08;
        m.stage0.gate_thresh = 0.005;
        m.stage0.gate_hyst = 0.002;
        m.stage0.gate_inv_knee = 120.0;
        m.stage0.gate_floor = 0.1;
        m.ctrl.w_tone_fc_oct[0] = 0.8;
        m.ctrl.w_tone_eq[0] = [0.25, -0.15, 0.2];
        m.ctrl.w_tank_psi[0] = 0.35;
        m.ctrl.w_film_gain[0][0][0] = 0.4;
        m.ctrl.w_film_bn[0][0][0] = 0.3;
        m.tone_svf.h0_proj_w = [0.25, -0.15, 0.12, 0.08];
        m.tone_svf.s1_proj_w = 0.35;
        m.tone_svf.lp_mix = 0.4;
        m.tone_svf.bp_mix = 0.2;
        m.tone_svf.cab_inject_w = 0.25;
        m.tone_svf.b1_inject_w = 0.18;
        for tank in &mut m.stage4 {
            tank.wet_mix = 0.15;
            tank.g_loop = 0.82;
        }
        m
    }

    #[test]
    fn test_v3_zero_dc_equilibrium_all_tiers_and_os2x() {
        for os2x in [false, true] {
            for mono_out in [false, true] {
                let mat = sample_nontrivial_matryoshka(os2x, mono_out);
                for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
                    let model = mat.extract_tier(tier);
                    let mut inst = V3CoreInstance::<DefaultMathBackend, _>::from_cold_model_in(
                        &model, &Global,
                    );
                    assert_eq!((&raw const *inst.hot_core() as usize) % 64, 0);
                    let in_z = [0.0_f32; 256];
                    let mut out_l = [1.0_f32; 256];
                    let mut out_r = [1.0_f32; 256];
                    let Ok(()) =
                        inst.process_block_infallible(&in_z, &in_z, &mut out_l, &mut out_r);
                    for i in 0..256 {
                        assert_eq!(out_l[i].to_bits(), 0, "nonzero L on {tier:?} os2x={os2x}");
                        assert_eq!(out_r[i].to_bits(), 0, "nonzero R on {tier:?} os2x={os2x}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_v3_matryoshka_prefix_equivalence_zoh_and_ctrl_targets() {
        let mat = sample_nontrivial_matryoshka(true, false);
        let lite = mat.extract_tier(Tier::Lite);
        let mut lite_inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&lite);
        let mut scalar_inst = V3CoreInstance::<ScalarBackend>::from_cold_model(&lite);

        let mut in_l = [0.0_f32; 512];
        let mut in_r = [0.0_f32; 512];
        for i in 0..512 {
            let t = i as f32 * 0.031;
            in_l[i] = 0.7 * t.sin() + 0.25 * (3.1 * t).cos();
            in_r[i] = 0.65 * (1.1 * t).cos() - 0.2 * (2.7 * t).sin();
        }
        let (mut l_simd, mut r_simd) = ([0.0_f32; 512], [0.0_f32; 512]);
        let (mut l_scal, mut r_scal) = ([0.0_f32; 512], [0.0_f32; 512]);
        lite_inst.process_block_stereo(&in_l, &in_r, &mut l_simd, &mut r_simd);
        scalar_inst.process_block_stereo(&in_l, &in_r, &mut l_scal, &mut r_scal);
        for i in 0..512 {
            assert!((l_simd[i] - l_scal[i]).abs() < 1.0e-4);
            assert!((r_simd[i] - r_scal[i]).abs() < 1.0e-4);
        }

        // Exercise ZOH sample-rate reconfiguration and ControlBus stage_targets application.
        let zoh96 = ZohRescaledParams::new(48_000, 96_000);
        lite_inst.hot_core_mut().reconfigure_with_zoh(&lite, &zoh96);
        let mut targets = [0.0_f32; 16];
        targets[0] = 0.25;
        targets[2] = 0.15;
        targets[8] = 0.08;
        lite_inst
            .hot_core_mut()
            .apply_control_bus_stage_targets(&targets);
        let (mut zoh_l, mut zoh_r) = ([0.0_f32; 64], [0.0_f32; 64]);
        lite_inst.process_block_stereo(&in_l[..64], &in_r[..64], &mut zoh_l, &mut zoh_r);
        assert!(zoh_l.iter().chain(zoh_r.iter()).all(|v| v.is_finite()));

        assert!(V3HotCore::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false) <= 4_608);
        assert!(V3HotCore::<DefaultMathBackend>::active_hot_bytes(Tier::Ultra, true) <= 28_672);

        lite_inst
            .hot_core_mut()
            .ensure_cold_stage4_cap(&lite.stage4, 2_048);
        lite_inst.hot_core_mut().reload_in_place(&lite);
        assert!(lite_inst.hot_core().stage4.effective_cap() >= 2_048);
    }

    #[test]
    fn test_v3_live_tier_switching_zero_transient() {
        let mut mat = sample_nontrivial_matryoshka(true, false);
        mat.tier_os2x = [false, false, true, true];
        let tiers = mat.to_four_tiers();
        let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&tiers[0]);
        let mut out_l = [0.0_f32; 256];
        let mut out_r = [0.0_f32; 256];
        let mut in_l = [0.0_f32; 256];
        let mut in_r = [0.0_f32; 256];
        for i in 0..256 {
            let t = i as f32 * 0.04;
            in_l[i] = 0.4 * t.sin();
            in_r[i] = 0.35 * t.cos();
        }
        inst.hot_core_mut().ctrl_bus.set_knob(0, 0.9);
        for (block_idx, &tier_idx) in [0usize, 1, 2, 3, 2, 0].iter().enumerate() {
            if block_idx > 0 {
                inst.switch_tier(&tiers[tier_idx]);
            }
            inst.process_block_stereo(&in_l, &in_r, &mut out_l, &mut out_r);
            for &y in out_l.iter().chain(out_r.iter()) {
                assert!(y.is_finite() && y.abs() < 50.0);
            }
        }
    }

    #[test]
    fn test_v3_sag_zero_thump_and_process_block_matches_step() {
        let mut mat = V3MatryoshkaModel::identity(false, true);
        mat.feat_params.slow_det_comp = 8.0;
        mat.feat_params.slow_atk_beta = 0.25;
        mat.feat_params.slow_rel_beta = 1.0e-5;
        for b in 0..NUM_GRU_BLOCKS {
            mat.blocks[b].b_z[0] = 8.0;
            mat.blocks[b].b_n[0] = 0.45;
            mat.blocks[b].k_sag[0] = 2.5;
            mat.blocks[b].w_in_n[0][0] = 1.8;
        }
        mat.readouts[0].pre_cab_u_gain = [0.0, 0.0];
        mat.readouts[0].pre_cab_w[0][0] = 1.0;

        let lite = mat.extract_tier(Tier::Lite);
        let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&lite);
        let burst = [0.5_f32; 128];
        let (mut out_l, mut out_r) = ([0.0_f32; 128], [0.0_f32; 128]);
        inst.process_block_stereo(&burst, &burst, &mut out_l, &mut out_r);
        assert!(out_l[127].abs() > 0.01);

        let silence = [0.0_f32; 32];
        let (mut sil_l, mut sil_r) = ([1.0_f32; 32], [1.0_f32; 32]);
        inst.process_block_stereo(&silence, &silence, &mut sil_l, &mut sil_r);
        assert!(sil_l[31].abs() < 1.0e-5);

        for &(tier, os2x) in &[
            (Tier::Lite, false),
            (Tier::Full, false),
            (Tier::Pro, true),
            (Tier::Ultra, true),
        ] {
            let mut m = sample_nontrivial_matryoshka(os2x, false);
            for b in 0..NUM_GRU_BLOCKS {
                m.blocks[b].k_sag.fill(0.45);
                m.blocks[b].k_quad.fill(0.12);
            }
            let model = m.extract_tier(tier);
            let mut inst_step = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&model);
            let mut inst_blk = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&model);
            inst_step.hot_core_mut().ctrl_bus.set_knob(0, 0.85);
            inst_blk.hot_core_mut().ctrl_bus.set_knob(0, 0.85);

            const N: usize = 45;
            let in_l: [f32; N] = core::array::from_fn(|i| 0.35 * ((i as f32) * 0.17).sin());
            let in_r: [f32; N] = core::array::from_fn(|i| -0.3 * ((i as f32) * 0.23).cos());
            let mut step_out_l = [0.0_f32; N];
            let mut step_out_r = [0.0_f32; N];
            {
                let _ftz = crate::FtzDazGuard::new();
                for i in 0..N {
                    let [yl, yr] = inst_step.step_stereo(in_l[i], in_r[i]);
                    step_out_l[i] = yl;
                    step_out_r[i] = yr;
                }
            }
            let mut blk_out_l = [0.0_f32; N];
            let mut blk_out_r = [0.0_f32; N];
            inst_blk.process_block_stereo(
                &in_l[..11],
                &in_r[..11],
                &mut blk_out_l[..11],
                &mut blk_out_r[..11],
            );
            inst_blk.process_block_stereo(
                &in_l[11..],
                &in_r[11..],
                &mut blk_out_l[11..],
                &mut blk_out_r[11..],
            );

            let max_diff = step_out_l
                .iter()
                .zip(step_out_r.iter())
                .zip(blk_out_l.iter().zip(blk_out_r.iter()))
                .map(|((&sl, &sr), (&bl, &br))| (sl - bl).abs().max((sr - br).abs()))
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff < 1.0e-5,
                "tier={tier:?} os2x={os2x} diff={max_diff}"
            );
        }
    }

    proptest! {
        #[test]
        fn prop_v3_bibo_bounded_and_finite(
            x0 in -20.0_f32..20.0_f32,
            x1 in -20.0_f32..20.0_f32,
            bias in -3.5_f32..3.5_f32,
            os2x in proptest::bool::ANY,
        ) {
            let mut mat = sample_nontrivial_matryoshka(os2x, false);
            for b in 0..NUM_GRU_BLOCKS {
                mat.blocks[b].b_n[0] = bias;
                mat.blocks[b].b_n[1] = -bias;
            }
            let model = mat.extract_tier(Tier::Full);
            let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&model);
            let _ftz = crate::FtzDazGuard::new();
            for s in 0..256 {
                let u0 = if s % 2 == 0 { x0 } else { x1 };
                let u1 = if s % 3 == 0 { -x1 } else { x0 };
                let [yl, yr] = inst.step_stereo(u0, u1);
                prop_assert!(yl.is_finite() && yr.is_finite());
                prop_assert!(yl.abs() < 2_000.0 && yr.abs() < 2_000.0);
            }
        }
    }
}
