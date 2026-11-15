//! Cache-Coherent SIMD `v3` Core Engine (`Zero-Lag 2x/1x` -> `Stage I Env-SVF` ->
//! `Stage II 3-Block Block-Lower-Triangular Zero-DC Matryoshka GRU` -> `Stage III FIR + Kautz`).
//!
//! Designed around three hardware constraints:
//! 1. Tight sequential `#[repr(C, align(64))]` hot working set (`~3.0 KB` Lite to `~17.8 KB` Ultra)
//!    so all active parameters and recursive states remain resident in L1D cache.
//! 2. Pure safe Rust (`#![forbid(unsafe_code)]`) executing via [`MathBackend::Vec4`] (`128-bit` NEON/
//!    SSE/WASM) with zero `FDIV`, zero `FSQRT`, zero `libm`, zero integer division, and zero allocations.
//! 3. Causally nested Matryoshka tiers (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`) enabling single-run shared
//!    training (`REQ-5.1`) and allocation-free, click-free live runtime tier switching (`switch_tier`).

#![forbid(unsafe_code)]

pub mod cascade;
pub mod instance;
pub mod math_pade;
pub mod stage3;
pub mod svf;

use core::marker::PhantomData;

use crate::math::{DefaultMathBackend, MathBackend, SimdVec4};
use crate::stages::v3_core::cascade::{
    MAX_GROUPS, NUM_GRU_BLOCKS, V3GruBlockHot, V3GruChannelState, V3InputFeaturesParams,
    V3OversamplerParams, ZeroLagOversampler2x, flat_gru_lane_index, tier_fir_segments, tier_groups,
    tier_kautz_banks,
};
use crate::stages::v3_core::stage3::{V3Stage3Hot, active_hot_bytes_for_tier};
use crate::stages::v3_core::svf::{V3Stage1SvfParams, V3Stage1SvfState};
use crate::types::Tier;

pub use cascade::{V3CascadeBlockCold, V3GruBlockCold};
pub use instance::{V3CoreInstance, v3_process_block_default, v3_process_block_infallible_default};
pub use stage3::{V3ColdTierModel, V3MatryoshkaModel, V3TierReadout};
pub use svf::V3Stage1SvfParams as V3SvfParams;

/// Cache-line aligned hot execution core for `v3` Matryoshka models (`Lite..Ultra`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct V3HotCore<B: MathBackend = DefaultMathBackend> {
    pub num_groups: usize,
    pub num_fir_segments: usize,
    pub num_kautz_banks: usize,
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub os_params: V3OversamplerParams,
    pub svf_params: V3Stage1SvfParams,
    pub svf_state: V3Stage1SvfState,
    pub os_l: ZeroLagOversampler2x,
    pub os_r: ZeroLagOversampler2x,
    pub feat_params: V3InputFeaturesParams,
    pub blocks: [V3GruBlockHot; NUM_GRU_BLOCKS],
    pub state_l: V3GruChannelState,
    pub state_r: V3GruChannelState,
    pub pre_cab_u_gain: [f32; 2],
    pub pre_cab_hp_gain: [f32; 2],
    pub direct_s2_u_gain: [f32; 2],
    pub direct_s2_hp_gain: [f32; 2],
    pub pre_cab_w_l: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub pre_cab_w_r: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub direct_s2_w_l: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub direct_s2_w_r: [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub stage3: V3Stage3Hot,
    _backend: PhantomData<B>,
}

fn pack_readout_weights(
    src: &[f32; cascade::MAX_TOTAL_GRU_LANES],
    num_groups: usize,
) -> [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS] {
    let mut dst = [[[0.0_f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS];
    for (b, dst_blk) in dst.iter_mut().enumerate() {
        for (g, dst_grp) in dst_blk.iter_mut().enumerate().take(num_groups) {
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
    ///
    /// Packs all [`MAX_GROUPS`] lower-triangular GRU prefix tiles so later [`Self::switch_tier`]
    /// calls across nested Matryoshka tiers require zero memory allocation or matrix repacking.
    #[must_use]
    pub fn from_cold_model(model: &V3ColdTierModel) -> Self {
        let num_groups = tier_groups(model.tier);
        let mut os_params = model.os_params;
        os_params.clamp_in_place();
        let mut svf_params = model.svf;
        svf_params.direct_taps_l = model.readout.svf_direct_taps_l;
        svf_params.direct_taps_r = model.readout.svf_direct_taps_r;
        svf_params.clamp_in_place();
        let mut feat_params = model.feat_params;
        feat_params.clamp_in_place();

        let blocks = [
            V3GruBlockHot::from_cold::<B>(&model.blocks[0], MAX_GROUPS),
            V3GruBlockHot::from_cold::<B>(&model.blocks[1], MAX_GROUPS),
            V3GruBlockHot::from_cold::<B>(&model.blocks[2], MAX_GROUPS),
        ];

        Self {
            num_groups,
            num_fir_segments: tier_fir_segments(model.tier),
            num_kautz_banks: tier_kautz_banks(model.tier),
            tier: model.tier,
            os2x: model.os2x,
            mono_out: model.mono_out,
            os_params,
            svf_params,
            svf_state: V3Stage1SvfState::default(),
            os_l: ZeroLagOversampler2x::default(),
            os_r: ZeroLagOversampler2x::default(),
            feat_params,
            blocks,
            state_l: V3GruChannelState::default(),
            state_r: V3GruChannelState::default(),
            pre_cab_u_gain: model.readout.pre_cab_u_gain,
            pre_cab_hp_gain: model.readout.pre_cab_hp_gain,
            direct_s2_u_gain: model.readout.direct_s2_u_gain,
            direct_s2_hp_gain: model.readout.direct_s2_hp_gain,
            pre_cab_w_l: pack_readout_weights(&model.readout.pre_cab_w[0], num_groups),
            pre_cab_w_r: pack_readout_weights(&model.readout.pre_cab_w[1], num_groups),
            direct_s2_w_l: pack_readout_weights(&model.readout.direct_s2_w[0], num_groups),
            direct_s2_w_r: pack_readout_weights(&model.readout.direct_s2_w[1], num_groups),
            stage3: V3Stage3Hot::from_cold(model),
            _backend: PhantomData,
        }
    }

    /// Switches active Matryoshka tier and readout weights in-place (`0` alloc, `0` state reset).
    pub fn switch_tier(&mut self, model: &V3ColdTierModel) {
        let old_groups = self.num_groups;
        let new_groups = tier_groups(model.tier);
        if new_groups > old_groups {
            for b in 0..NUM_GRU_BLOCKS {
                for g in old_groups..new_groups {
                    self.state_l.h[b][g] = [0.0; 4];
                    self.state_r.h[b][g] = [0.0; 4];
                }
            }
        }
        self.tier = model.tier;
        self.num_groups = new_groups;
        self.num_fir_segments = tier_fir_segments(model.tier);
        self.num_kautz_banks = tier_kautz_banks(model.tier);
        self.svf_params.direct_taps_l = model.readout.svf_direct_taps_l;
        self.svf_params.direct_taps_r = model.readout.svf_direct_taps_r;
        self.svf_params.direct_taps_l[7] = 0.0;
        self.svf_params.direct_taps_r[7] = 0.0;
        self.pre_cab_u_gain = model.readout.pre_cab_u_gain;
        self.pre_cab_hp_gain = model.readout.pre_cab_hp_gain;
        self.direct_s2_u_gain = model.readout.direct_s2_u_gain;
        self.direct_s2_hp_gain = model.readout.direct_s2_hp_gain;
        self.pre_cab_w_l = pack_readout_weights(&model.readout.pre_cab_w[0], new_groups);
        self.pre_cab_w_r = pack_readout_weights(&model.readout.pre_cab_w[1], new_groups);
        self.direct_s2_w_l = pack_readout_weights(&model.readout.direct_s2_w[0], new_groups);
        self.direct_s2_w_r = pack_readout_weights(&model.readout.direct_s2_w[1], new_groups);
        self.stage3
            .update_readout(&model.readout, &model.kautz_poles);
    }

    /// Resets all recursive filter, envelope, oversampler, GRU, FIR, and Kautz states to exact `+0.0`.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.svf_state.reset();
        self.os_l.reset();
        self.os_r.reset();
        self.state_l.reset();
        self.state_r.reset();
        self.stage3.reset();
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn step_stage2_channel(
        blocks: &[V3GruBlockHot; NUM_GRU_BLOCKS],
        num_groups: usize,
        feat_params: &V3InputFeaturesParams,
        pre_cab_u_gain: f32,
        pre_cab_hp_gain: f32,
        direct_s2_u_gain: f32,
        direct_s2_hp_gain: f32,
        pre_cab_w: &[[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
        direct_s2_w: &[[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
        u_s2: f32,
        state: &mut V3GruChannelState,
    ) -> (f32, f32) {
        let feats = state.feat.step(feat_params, u_s2);
        let mut cab_acc = B::Vec4::zero();
        let mut dir_acc = B::Vec4::zero();
        let mut prev_h = [[0.0_f32; 4]; MAX_GROUPS];
        for (b, blk) in blocks.iter().enumerate() {
            blk.step_block::<B>(
                b,
                num_groups,
                &feats,
                &prev_h,
                &mut state.h[b],
                &mut cab_acc,
                &mut dir_acc,
                &pre_cab_w[b],
                &direct_s2_w[b],
            );
            prev_h = state.h[b];
        }
        let (cab_dot, dir_dot) = cab_acc.horizontal_sum_pair(dir_acc);
        let s2_out = pre_cab_u_gain * feats[0] + pre_cab_hp_gain * feats[1] + cab_dot;
        let dir_out = direct_s2_u_gain * feats[0] + direct_s2_hp_gain * feats[1] + dir_dot;
        (s2_out, dir_out)
    }

    /// Processes a single stereo sample pair at host rate (`48 kHz`).
    /// Direct callers must hold [`crate::FtzDazGuard`] for the enclosing audio block.
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let xl = if in_l.is_finite() { in_l } else { 0.0 };
        let xr = if self.mono_out {
            xl
        } else if in_r.is_finite() {
            in_r
        } else {
            0.0
        };

        let (s2_out_l, s2_out_r, dir_s2_l, dir_s2_r, dir_svf_l, dir_svf_r) = if !self.os2x {
            let ([u_s2_l, u_s2_r], [svf_d_l, svf_d_r]) =
                self.svf_state.step_stereo(&self.svf_params, xl, xr);
            let (s2_l, d2_l) = Self::step_stage2_channel(
                &self.blocks,
                self.num_groups,
                &self.feat_params,
                self.pre_cab_u_gain[0],
                self.pre_cab_hp_gain[0],
                self.direct_s2_u_gain[0],
                self.direct_s2_hp_gain[0],
                &self.pre_cab_w_l,
                &self.direct_s2_w_l,
                u_s2_l,
                &mut self.state_l,
            );
            let (s2_r, d2_r) = if self.mono_out {
                (s2_l, d2_l)
            } else {
                Self::step_stage2_channel(
                    &self.blocks,
                    self.num_groups,
                    &self.feat_params,
                    self.pre_cab_u_gain[1],
                    self.pre_cab_hp_gain[1],
                    self.direct_s2_u_gain[1],
                    self.direct_s2_hp_gain[1],
                    &self.pre_cab_w_r,
                    &self.direct_s2_w_r,
                    u_s2_r,
                    &mut self.state_r,
                )
            };
            (s2_l, s2_r, d2_l, d2_r, svf_d_l, svf_d_r)
        } else {
            let [xl0, xl1] = self.os_l.upsample(xl, &self.os_params);
            let [xr0, xr1] = if self.mono_out {
                [xl0, xl1]
            } else {
                self.os_r.upsample(xr, &self.os_params)
            };

            let ([u0_l, u0_r], [svf0_l, svf0_r]) =
                self.svf_state.step_stereo(&self.svf_params, xl0, xr0);
            let (s2_0_l, d2_0_l) = Self::step_stage2_channel(
                &self.blocks,
                self.num_groups,
                &self.feat_params,
                self.pre_cab_u_gain[0],
                self.pre_cab_hp_gain[0],
                self.direct_s2_u_gain[0],
                self.direct_s2_hp_gain[0],
                &self.pre_cab_w_l,
                &self.direct_s2_w_l,
                u0_l,
                &mut self.state_l,
            );
            let (s2_0_r, d2_0_r) = if self.mono_out {
                (s2_0_l, d2_0_l)
            } else {
                Self::step_stage2_channel(
                    &self.blocks,
                    self.num_groups,
                    &self.feat_params,
                    self.pre_cab_u_gain[1],
                    self.pre_cab_hp_gain[1],
                    self.direct_s2_u_gain[1],
                    self.direct_s2_hp_gain[1],
                    &self.pre_cab_w_r,
                    &self.direct_s2_w_r,
                    u0_r,
                    &mut self.state_r,
                )
            };

            let ([u1_l, u1_r], [svf1_l, svf1_r]) =
                self.svf_state.step_stereo(&self.svf_params, xl1, xr1);
            let (s2_1_l, d2_1_l) = Self::step_stage2_channel(
                &self.blocks,
                self.num_groups,
                &self.feat_params,
                self.pre_cab_u_gain[0],
                self.pre_cab_hp_gain[0],
                self.direct_s2_u_gain[0],
                self.direct_s2_hp_gain[0],
                &self.pre_cab_w_l,
                &self.direct_s2_w_l,
                u1_l,
                &mut self.state_l,
            );
            let (s2_1_r, d2_1_r) = if self.mono_out {
                (s2_1_l, d2_1_l)
            } else {
                Self::step_stage2_channel(
                    &self.blocks,
                    self.num_groups,
                    &self.feat_params,
                    self.pre_cab_u_gain[1],
                    self.pre_cab_hp_gain[1],
                    self.direct_s2_u_gain[1],
                    self.direct_s2_hp_gain[1],
                    &self.pre_cab_w_r,
                    &self.direct_s2_w_r,
                    u1_r,
                    &mut self.state_r,
                )
            };

            let s2_l = ZeroLagOversampler2x::downsample(
                s2_0_l,
                s2_1_l,
                &mut self.os_l.dec_s2_prev1,
                &mut self.os_l.dec_s2_aa,
                &self.os_params,
            );
            let d2_l = ZeroLagOversampler2x::downsample(
                d2_0_l,
                d2_1_l,
                &mut self.os_l.dec_dir_s2_prev1,
                &mut self.os_l.dec_dir_s2_aa,
                &self.os_params,
            );
            let svf_l = ZeroLagOversampler2x::downsample(
                svf0_l,
                svf1_l,
                &mut self.os_l.dec_dir_svf_prev1,
                &mut self.os_l.dec_dir_svf_aa,
                &self.os_params,
            );
            if self.mono_out {
                (s2_l, s2_l, d2_l, d2_l, svf_l, svf_l)
            } else {
                let s2_r = ZeroLagOversampler2x::downsample(
                    s2_0_r,
                    s2_1_r,
                    &mut self.os_r.dec_s2_prev1,
                    &mut self.os_r.dec_s2_aa,
                    &self.os_params,
                );
                let d2_r = ZeroLagOversampler2x::downsample(
                    d2_0_r,
                    d2_1_r,
                    &mut self.os_r.dec_dir_s2_prev1,
                    &mut self.os_r.dec_dir_s2_aa,
                    &self.os_params,
                );
                let svf_r = ZeroLagOversampler2x::downsample(
                    svf0_r,
                    svf1_r,
                    &mut self.os_r.dec_dir_svf_prev1,
                    &mut self.os_r.dec_dir_svf_aa,
                    &self.os_params,
                );
                (s2_l, s2_r, d2_l, d2_r, svf_l, svf_r)
            }
        };

        let [cab_l, cab_r] = self.stage3.step_stereo::<B>(
            s2_out_l,
            s2_out_r,
            self.num_fir_segments,
            self.num_kautz_banks,
            self.mono_out,
        );
        if self.mono_out {
            let y_l = dir_svf_l + dir_s2_l + cab_l;
            [y_l, y_l]
        } else {
            [dir_svf_l + dir_s2_l + cab_l, dir_svf_r + dir_s2_r + cab_r]
        }
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
        let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
        for i in 0..n {
            let [yl, yr] = self.step_stereo(in_l[i], in_r[i]);
            out_l[i] = yl;
            out_r[i] = yr;
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
    use crate::math::ScalarBackend;
    use crate::stages::v3_core::cascade::{MAX_BLOCK_LANES, NUM_IN_FEATS_B0, NUM_IN_FEATS_BN};
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
        for k in 0..cascade::MAX_KAUTZ_POLES {
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
            let g_cnt = tier_groups(tier);
            for b in 0..NUM_GRU_BLOCKS {
                for g in 0..g_cnt {
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
    fn test_v3_matryoshka_prefix_equivalence_and_backend_parity() {
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

        // Embed Lite readout into Full/Pro/Ultra on the full Matryoshka core -> bit-identical output.
        for higher_tier in [Tier::Full, Tier::Pro, Tier::Ultra] {
            let mut higher = lite.clone();
            higher.tier = higher_tier;
            let mut higher_inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&higher);
            let (mut hl, mut hr) = ([0.0_f32; 512], [0.0_f32; 512]);
            higher_inst.process_block_stereo(&in_l, &in_r, &mut hl, &mut hr);
            for i in 0..512 {
                assert!(
                    (hl[i] - l_simd[i]).abs() < 1.0e-6,
                    "{higher_tier:?} prefix diff L at {i}"
                );
                assert!(
                    (hr[i] - r_simd[i]).abs() < 1.0e-6,
                    "{higher_tier:?} prefix diff R at {i}"
                );
            }
        }

        assert!(V3HotCore::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false) <= 3072);
        assert!(V3HotCore::<DefaultMathBackend>::active_hot_bytes(Tier::Ultra, true) <= 20_480);
    }

    #[test]
    fn test_v3_live_tier_switching_zero_transient() {
        let mat = sample_nontrivial_matryoshka(true, false);
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
    fn test_v3_sag_zero_thump_when_signal_silent() {
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

        // Immediately drop signal to exact 0.0 while env_slow remains primed -> zero envelope thump.
        let silence = [0.0_f32; 32];
        let (mut sil_l, mut sil_r) = ([1.0_f32; 32], [1.0_f32; 32]);
        inst.process_block_stereo(&silence, &silence, &mut sil_l, &mut sil_r);
        assert!(
            sil_l[31].abs() < 1.0e-5,
            "envelope thump leaked: {}",
            sil_l[31]
        );
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
