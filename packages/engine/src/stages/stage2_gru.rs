//! Stage II Zero-DC Tapered Matryoshka `GRU`, Inter-Block `TPT-SVF`, & `Stage2Backend` Facade.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` & `docs/adr/0021-unified-matryoshka-gru-and-direct-register-stages.md`.

pub mod block;
pub mod features;
pub mod film;
pub mod frontend;
pub(crate) mod kernels;
pub mod math_pade;
pub mod tone_svf;
pub mod zoh;

pub use block::{
    FIR_RING_CAP, FIR_RING_MASK, LOWER_TRI_TILES, MAX_BLOCK_LANES, MAX_CASCADE_LANES,
    MAX_FIR_SEGMENTS, MAX_GROUPS, MAX_GROUPS_PER_BLOCK, MAX_KAUTZ_BANKS, MAX_KAUTZ_POLES,
    MAX_TOTAL_GRU_LANES, Mat4Tile, NUM_CASCADE_BLOCKS, NUM_GRU_BLOCKS, NUM_IN_FEATS_B0,
    NUM_IN_FEATS_BN, V3GruBlockHot, V3GruChannelState, V3GruGroupHot, flat_gru_lane_index,
    tier_block_groups, tier_block_groups_array, tier_block_lanes, tier_block_lanes_array,
    tier_block_lanes_at, tier_cascade_lanes, tier_fir_segments, tier_groups, tier_gru_blocks,
    tier_kautz_banks, tier_total_gru_lanes, tri_tile_index,
};
pub use features::{StereoPairFeatures, V3InputFeaturesParams, V3InputFeaturesState};
pub use film::{
    MAX_CTRL_BLOCK_LANES, NUM_CTRL_BLOCKS, NUM_V3_MACRO_KNOBS, V3ControlBusParams,
    V3ControlBusState,
};
pub use frontend::{
    HB_A0, HB_A1, V3OversamplerParams, V3StereoOversampler2x, ZeroLagOversampler2x,
    step_allpass_2_f32, step_hb_decim_f32,
};
pub use math_pade::{
    recip_pos_ge1_f32, sigmoid_pade5_f32, sigmoid_pade5_pair_vec4, sigmoid_pade5_vec4,
    tanh_pade5_6_with_rcp2_vec4, tanh_pade5_batch_vec4, tanh_pade5_f32, tanh_pade5_pair_vec4,
    tanh_pade5_vec4,
};
pub use tone_svf::{V3InterBlockToneSvfParams, V3InterBlockToneSvfState};
pub use zoh::{V3CascadeBlockCold, V3GruBlockCold};

use crate::math::{MathBackend, SimdVec4};

/// Immutable Stage II block, control, and readout parameters passed into [`Stage2Backend::step_stage2_stereo`].
#[derive(Clone, Copy, Debug)]
pub struct Stage2StepContext<'a> {
    pub blocks: &'a [V3GruBlockHot; NUM_GRU_BLOCKS],
    pub num_blocks: usize,
    pub block_groups: &'a [usize; NUM_GRU_BLOCKS],
    pub feat_params: &'a V3InputFeaturesParams,
    pub ctrl_bus: &'a V3ControlBusState,
    pub pre_cab_u_gain: &'a [f32; 2],
    pub pre_cab_hp_gain: &'a [f32; 2],
    pub direct_s2_u_gain: &'a [f32; 2],
    pub direct_s2_hp_gain: &'a [f32; 2],
    pub pre_cab_w_l: &'a [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub pre_cab_w_r: &'a [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub direct_s2_w_l: &'a [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
    pub direct_s2_w_r: &'a [[[f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS],
}

/// Domain-specific SIMD backend trait for tapered `3`-block stereo Stage II `GRU` execution.
///
/// Steps `Left` (`SimdVec4`) and `Right` (`SimdVec4`) hidden blocks side by side inside
/// `step_stage2_stereo` (`r[dsp.op-n.stage2.conv6-simd]`), reusing `4×4` tiles across both pipes.
pub trait Stage2Backend<V: SimdVec4> {
    /// Steps `5`-channel physical input features, `3`-block tapered `GRU`, inter-block `TPT-SVF`,
    /// and linear `pre_cab` + `direct_s2` stereo readouts (`(s2_l, s2_r, dir_l, dir_r)`).
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32);
}

/// Safe `SimdVec4` implementation of [`Stage2Backend::step_stage2_stereo`].
#[inline(always)]
pub fn step_stage2_stereo_vec4<B: MathBackend>(
    ctx: &Stage2StepContext<'_>,
    state_l: &mut V3GruChannelState,
    state_r: &mut V3GruChannelState,
    tone_svf: &mut V3InterBlockToneSvfState,
    u_s2_l: f32,
    u_s2_r: f32,
) -> (f32, f32, f32, f32) {
    step_stage2_stereo_with_lp2_vec4::<B>(
        ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r, [0.0; 2], 0.0,
    )
}

/// Safe `SimdVec4` Stage II step with optional `Stage I lp2` blending into `u_hp`.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "Matches step_stage2_stereo_vec4 without enlarging public Stage2StepContext"
)]
pub fn step_stage2_stereo_with_lp2_vec4<B: MathBackend>(
    ctx: &Stage2StepContext<'_>,
    state_l: &mut V3GruChannelState,
    state_r: &mut V3GruChannelState,
    tone_svf: &mut V3InterBlockToneSvfState,
    u_s2_l: f32,
    u_s2_r: f32,
    lp2: [f32; 2],
    lp2_mix: f32,
) -> (f32, f32, f32, f32) {
    let (feats_l, feats_r) = V3InputFeaturesState::step_stereo_with_lp2::<B>(
        &mut state_l.feat,
        &mut state_r.feat,
        ctx.feat_params,
        u_s2_l,
        u_s2_r,
        lp2,
        lp2_mix,
    );
    let mut cab_acc_l = B::Vec4::zero();
    let mut cab_acc_r = B::Vec4::zero();
    let mut dir_acc_l = B::Vec4::zero();
    let mut dir_acc_r = B::Vec4::zero();
    let mut prev_l = [[0.0_f32; 4]; MAX_GROUPS];
    let mut prev_r = [[0.0_f32; 4]; MAX_GROUPS];
    let mut blk_feats_l = feats_l;
    let mut blk_feats_r = feats_r;
    let mut tone_out = [0.0_f32; 2];
    let nb = ctx.num_blocks.min(NUM_GRU_BLOCKS);

    for b in 0..nb {
        ctx.blocks[b].step_block_stereo::<B>(
            b,
            ctx.block_groups[b],
            &blk_feats_l,
            &blk_feats_r,
            &prev_l,
            &prev_r,
            &mut state_l.h[b],
            &mut state_r.h[b],
            &ctx.ctrl_bus.block_film_gain[b],
            &ctx.ctrl_bus.block_bn_eff[b],
            &ctx.ctrl_bus.block_tanh_bn_eff[b],
            &mut cab_acc_l,
            &mut cab_acc_r,
            &mut dir_acc_l,
            &mut dir_acc_r,
            &ctx.pre_cab_w_l[b],
            &ctx.pre_cab_w_r[b],
            &ctx.direct_s2_w_l[b],
            &ctx.direct_s2_w_r[b],
        );
        prev_l = state_l.h[b];
        prev_r = state_r.h[b];
        if b == 0 {
            let h0w = &tone_svf.params.h0_proj_w;
            let h0l = &prev_l[0];
            let h0r = &prev_r[0];
            let tone_in_l = tone_svf.params.s1_proj_w * u_s2_l
                + h0w[0] * h0l[0]
                + h0w[1] * h0l[1]
                + h0w[2] * h0l[2]
                + h0w[3] * h0l[3];
            let tone_in_r = tone_svf.params.s1_proj_w * u_s2_r
                + h0w[0] * h0r[0]
                + h0w[1] * h0r[1]
                + h0w[2] * h0r[2]
                + h0w[3] * h0r[3];
            tone_out = tone_svf.step_stereo(tone_in_l, tone_in_r);
            blk_feats_l[0] = feats_l[0] + tone_svf.params.b1_inject_w * tone_out[0];
            blk_feats_r[0] = feats_r[0] + tone_svf.params.b1_inject_w * tone_out[1];
        } else {
            blk_feats_l[0] = feats_l[0];
            blk_feats_r[0] = feats_r[0];
        }
    }

    let (cab_dot_l, cab_dot_r) = cab_acc_l.horizontal_sum_pair(cab_acc_r);
    let (dir_dot_l, dir_dot_r) = dir_acc_l.horizontal_sum_pair(dir_acc_r);
    let cab_inj = tone_svf.params.cab_inject_w;
    let s2_out_l = ctx.pre_cab_u_gain[0] * feats_l[0]
        + ctx.pre_cab_hp_gain[0] * feats_l[1]
        + cab_dot_l
        + cab_inj * tone_out[0];
    let s2_out_r = ctx.pre_cab_u_gain[1] * feats_r[0]
        + ctx.pre_cab_hp_gain[1] * feats_r[1]
        + cab_dot_r
        + cab_inj * tone_out[1];
    let dir_s2_l =
        ctx.direct_s2_u_gain[0] * feats_l[0] + ctx.direct_s2_hp_gain[0] * feats_l[1] + dir_dot_l;
    let dir_s2_r =
        ctx.direct_s2_u_gain[1] * feats_r[0] + ctx.direct_s2_hp_gain[1] * feats_r[1] + dir_dot_r;
    (s2_out_l, s2_out_r, dir_s2_l, dir_s2_r)
}

impl Stage2Backend<crate::math::ScalarVec4> for crate::math::ScalarBackend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage2Backend<crate::math::NeonVec4> for crate::math::NeonBackend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage2Backend<crate::math::Sve2Vec4> for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(target_arch = "aarch64")]
impl Stage2Backend<crate::math::NeonVec4> for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage2Backend<crate::math::Sse2Vec4> for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage2Backend<crate::math::Sse42Vec4> for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(target_arch = "x86_64")]
impl Stage2Backend<crate::math::Avx2FmaVec4> for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(target_arch = "wasm32")]
impl Stage2Backend<crate::math::WasmVec4> for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn step_stage2_stereo(
        ctx: &Stage2StepContext<'_>,
        state_l: &mut V3GruChannelState,
        state_r: &mut V3GruChannelState,
        tone_svf: &mut V3InterBlockToneSvfState,
        u_s2_l: f32,
        u_s2_r: f32,
    ) -> (f32, f32, f32, f32) {
        step_stage2_stereo_vec4::<Self>(ctx, state_l, state_r, tone_svf, u_s2_l, u_s2_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};
    use crate::types::Tier;

    #[test]
    fn test_stage2_backend_stereo_dispatch_parity() {
        let mut cold_blk = V3GruBlockCold::default();
        cold_blk.w_in_n[0][0] = 0.65;
        cold_blk.b_n[0] = 0.25;
        cold_blk.w_rec_n[0][0] = 0.15;

        let blocks = core::array::from_fn(|_| {
            V3GruBlockHot::from_cold::<ScalarBackend>(&cold_blk, MAX_GROUPS)
        });
        let block_groups = tier_block_groups_array(Tier::Full);
        let feat_params = V3InputFeaturesParams::default();
        let ctrl_params = V3ControlBusParams::identity();
        let ctrl_bus = V3ControlBusState::from_cold::<ScalarBackend>(
            &ctrl_params,
            &blocks,
            NUM_GRU_BLOCKS,
            &block_groups,
        );
        let mut w_pre = [[[0.0_f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS];
        w_pre[0][0] = [0.4, -0.2, 0.1, -0.05];
        let mut w_dir = [[[0.0_f32; 4]; MAX_GROUPS]; NUM_GRU_BLOCKS];
        w_dir[1][0] = [0.15, -0.1, 0.05, 0.0];
        let ctx = Stage2StepContext {
            blocks: &blocks,
            num_blocks: NUM_GRU_BLOCKS,
            block_groups: &block_groups,
            feat_params: &feat_params,
            ctrl_bus: &ctrl_bus,
            pre_cab_u_gain: &[0.2, 0.2],
            pre_cab_hp_gain: &[0.05, -0.05],
            direct_s2_u_gain: &[0.02, 0.02],
            direct_s2_hp_gain: &[0.01, -0.01],
            pre_cab_w_l: &w_pre,
            pre_cab_w_r: &w_pre,
            direct_s2_w_l: &w_dir,
            direct_s2_w_r: &w_dir,
        };

        let mut sl_s = V3GruChannelState::default();
        let mut sr_s = V3GruChannelState::default();
        let mut ts_s = V3InterBlockToneSvfState::from_cold(&V3InterBlockToneSvfParams::identity());
        let mut sl_d = sl_s;
        let mut sr_d = sr_s;
        let mut ts_d = ts_s;

        for i in 0..32 {
            let x_l = 0.4 * ((i as f32) * 0.19).sin();
            let x_r = -0.35 * ((i as f32) * 0.23).cos();
            let ys = <ScalarBackend as Stage2Backend<_>>::step_stage2_stereo(
                &ctx, &mut sl_s, &mut sr_s, &mut ts_s, x_l, x_r,
            );
            let yd = <DefaultMathBackend as Stage2Backend<_>>::step_stage2_stereo(
                &ctx, &mut sl_d, &mut sr_d, &mut ts_d, x_l, x_r,
            );
            assert!(
                (ys.0 - yd.0).abs() < 1.0e-5
                    && (ys.1 - yd.1).abs() < 1.0e-5
                    && (ys.2 - yd.2).abs() < 1.0e-5
                    && (ys.3 - yd.3).abs() < 1.0e-5,
                "parity diff at {i}: scalar={ys:?} default={yd:?}"
            );
        }
    }
}
