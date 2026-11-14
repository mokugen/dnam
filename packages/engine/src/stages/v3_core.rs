//! Cache-Coherent SIMD `v3` Core Engine (`Stage I Env-SVF -> Stage II 2x/1x 3-Block Cascade -> Stage III FIR + Kautz`).
//!
//! Designed around three hardware constraints:
//! 1. Tight sequential `#[repr(C, align(64))]` hot working set (`~1.5 KB` Lite to `~8.2 KB` Ultra) so
//!    all active parameters and recursive states remain resident in L1D cache.
//! 2. Pure safe Rust (`#![forbid(unsafe_code)]`) executing via [`MathBackend::Vec4`] (`128-bit` NEON/
//!    SSE/WASM) with zero `FDIV`, zero `FSQRT`, zero `libm`, zero integer division, and zero allocations.
//! 3. Idiomatic Rust `1.100.0` `core::alloc::Allocator` injection (`Box::new_in`) and infallible
//!    block processing via `Result<(), !>`.

#![forbid(unsafe_code)]

pub mod cascade;
pub mod svf;

use alloc::alloc::Global;
use alloc::boxed::Box;
use core::alloc::Allocator;
use core::marker::PhantomData;

use crate::math::{DefaultMathBackend, MathBackend, SimdVec4};
use crate::stages::v3_core::cascade::{
    ChannelOversampler2x, MAX_CASCADE_LANES, MAX_GROUPS, NUM_CASCADE_BLOCKS, V3CascadeBlockHot,
    V3CascadeChannelState, V3Stage3Hot, active_hot_bytes_for_tier, tier_fir_segments, tier_groups,
    tier_kautz_banks,
};
use crate::stages::v3_core::svf::{V3Stage1SvfParams, V3Stage1SvfState};
use crate::types::Tier;

pub use cascade::{V3CascadeBlockCold, V3ColdTierModel};
pub use svf::V3Stage1SvfParams as V3SvfParams;

/// Cache-line aligned hot execution core for `v3` models (`Lite..Ultra`).
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct V3HotCore<B: MathBackend = DefaultMathBackend> {
    pub num_groups: usize,
    pub num_fir_segments: usize,
    pub num_kautz_banks: usize,
    pub tier: Tier,
    pub os2x: bool,
    pub mono_out: bool,
    pub pre_cab_u_gain: [f32; 2],
    pub svf_params: V3Stage1SvfParams,
    pub svf_state: V3Stage1SvfState,
    pub os_l: ChannelOversampler2x,
    pub os_r: ChannelOversampler2x,
    pub blocks: [V3CascadeBlockHot; NUM_CASCADE_BLOCKS],
    pub state_l: V3CascadeChannelState,
    pub state_r: V3CascadeChannelState,
    pub pre_cab_w_l: [[f32; 4]; MAX_GROUPS],
    pub pre_cab_w_r: [[f32; 4]; MAX_GROUPS],
    pub direct_s2_w_l: [[f32; 4]; MAX_GROUPS],
    pub direct_s2_w_r: [[f32; 4]; MAX_GROUPS],
    pub stage3: V3Stage3Hot,
    _backend: PhantomData<B>,
}

impl<B: MathBackend> V3HotCore<B> {
    /// Returns the active hot working-set bytes read/written per sample for `tier`.
    #[inline]
    #[must_use]
    pub const fn active_hot_bytes(tier: Tier, os2x: bool) -> usize {
        active_hot_bytes_for_tier(tier, os2x)
    }

    /// Compiles a cold [`V3ColdTierModel`] into an L1D-packed [`V3HotCore<B>`].
    #[must_use]
    pub fn from_cold_model(model: &V3ColdTierModel) -> Self {
        let num_groups = tier_groups(model.tier);
        let mut svf_params = model.svf;
        svf_params.clamp_in_place();

        let blocks = [
            V3CascadeBlockHot::from_cold::<B>(&model.blocks[0], num_groups),
            V3CascadeBlockHot::from_cold::<B>(&model.blocks[1], num_groups),
            V3CascadeBlockHot::from_cold::<B>(&model.blocks[2], num_groups),
        ];

        let mut pre_cab_w_l = [[0.0_f32; 4]; MAX_GROUPS];
        let mut pre_cab_w_r = [[0.0_f32; 4]; MAX_GROUPS];
        let mut direct_s2_w_l = [[0.0_f32; 4]; MAX_GROUPS];
        let mut direct_s2_w_r = [[0.0_f32; 4]; MAX_GROUPS];
        for g in 0..num_groups {
            for l in 0..4 {
                let idx = g * 4 + l;
                if idx < MAX_CASCADE_LANES {
                    pre_cab_w_l[g][l] = model.pre_cab_w[0][idx];
                    pre_cab_w_r[g][l] = model.pre_cab_w[1][idx];
                    direct_s2_w_l[g][l] = model.direct_s2_w[0][idx];
                    direct_s2_w_r[g][l] = model.direct_s2_w[1][idx];
                }
            }
        }

        Self {
            num_groups,
            num_fir_segments: tier_fir_segments(model.tier),
            num_kautz_banks: tier_kautz_banks(model.tier),
            tier: model.tier,
            os2x: model.os2x,
            mono_out: model.mono_out,
            pre_cab_u_gain: model.pre_cab_u_gain,
            svf_params,
            svf_state: V3Stage1SvfState::default(),
            os_l: ChannelOversampler2x::default(),
            os_r: ChannelOversampler2x::default(),
            blocks,
            state_l: V3CascadeChannelState::default(),
            state_r: V3CascadeChannelState::default(),
            pre_cab_w_l,
            pre_cab_w_r,
            direct_s2_w_l,
            direct_s2_w_r,
            stage3: V3Stage3Hot::from_cold(model),
            _backend: PhantomData,
        }
    }

    /// Resets all recursive filter, envelope, oversampler, FIR, and Kautz states to exact `+0.0`.
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
    fn step_stage2_substep(
        blocks: &[V3CascadeBlockHot; NUM_CASCADE_BLOCKS],
        num_groups: usize,
        pre_cab_u_gain: f32,
        pre_cab_w: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w: &[[f32; 4]; MAX_GROUPS],
        u_s2: f32,
        state: &mut V3CascadeChannelState,
    ) -> (f32, f32) {
        let mut h_buf = [[0.0_f32; 4]; MAX_GROUPS];
        for b in 0..NUM_CASCADE_BLOCKS {
            blocks[b].step_channel::<B>(
                b,
                num_groups,
                u_s2,
                &mut h_buf,
                &mut state.lp_state[b],
                &mut state.env_q[b],
            );
        }
        let mut cab_acc = B::Vec4::zero();
        let mut dir_acc = B::Vec4::zero();
        for g in 0..num_groups {
            let h = B::Vec4::load(&h_buf[g]);
            cab_acc = cab_acc.fma(h, B::Vec4::load(&pre_cab_w[g]));
            dir_acc = dir_acc.fma(h, B::Vec4::load(&direct_s2_w[g]));
        }
        let (cab_dot, dir_dot) = cab_acc.horizontal_sum_pair(dir_acc);
        (pre_cab_u_gain * u_s2 + cab_dot, dir_dot)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn step_stage2_channel(
        os2x: bool,
        blocks: &[V3CascadeBlockHot; NUM_CASCADE_BLOCKS],
        num_groups: usize,
        pre_cab_u_gain: f32,
        pre_cab_w: &[[f32; 4]; MAX_GROUPS],
        direct_s2_w: &[[f32; 4]; MAX_GROUPS],
        u_s2: f32,
        os: &mut ChannelOversampler2x,
        state: &mut V3CascadeChannelState,
    ) -> (f32, f32) {
        if !os2x {
            Self::step_stage2_substep(
                blocks,
                num_groups,
                pre_cab_u_gain,
                pre_cab_w,
                direct_s2_w,
                u_s2,
                state,
            )
        } else {
            let [u0, u1] = os.upsample(u_s2);
            let (s2_0, d_0) = Self::step_stage2_substep(
                blocks,
                num_groups,
                pre_cab_u_gain,
                pre_cab_w,
                direct_s2_w,
                u0,
                state,
            );
            let (s2_1, d_1) = Self::step_stage2_substep(
                blocks,
                num_groups,
                pre_cab_u_gain,
                pre_cab_w,
                direct_s2_w,
                u1,
                state,
            );
            (os.downsample_s2(s2_0, s2_1), os.downsample_dir(d_0, d_1))
        }
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

        let ([u_s2_l, u_s2_r], [dir_svf_l, dir_svf_r]) =
            self.svf_state.step_stereo(&self.svf_params, xl, xr);

        let (s2_out_l, dir_s2_l) = Self::step_stage2_channel(
            self.os2x,
            &self.blocks,
            self.num_groups,
            self.pre_cab_u_gain[0],
            &self.pre_cab_w_l,
            &self.direct_s2_w_l,
            u_s2_l,
            &mut self.os_l,
            &mut self.state_l,
        );

        let (s2_out_r, dir_s2_r) = if self.mono_out {
            (s2_out_l, dir_s2_l)
        } else {
            Self::step_stage2_channel(
                self.os2x,
                &self.blocks,
                self.num_groups,
                self.pre_cab_u_gain[1],
                &self.pre_cab_w_r,
                &self.direct_s2_w_r,
                u_s2_r,
                &mut self.os_r,
                &mut self.state_r,
            )
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

/// Owned heap/arena-allocated [`V3HotCore`] parameterized by [`core::alloc::Allocator`].
#[derive(Clone, Debug)]
pub struct V3CoreInstance<B: MathBackend = DefaultMathBackend, A: Allocator = Global> {
    hot: Box<V3HotCore<B>, A>,
}

impl<B: MathBackend> V3CoreInstance<B, Global> {
    /// Constructs a `V3CoreInstance` in the global allocator from `model`.
    #[must_use]
    pub fn from_cold_model(model: &V3ColdTierModel) -> Self {
        Self::from_cold_model_in(model, Global)
    }
}

impl<B: MathBackend, A: Allocator> V3CoreInstance<B, A> {
    /// Constructs a `V3CoreInstance` inside a custom [`core::alloc::Allocator`] (`Box::new_in`).
    #[must_use]
    pub fn from_cold_model_in(model: &V3ColdTierModel, alloc: A) -> Self {
        Self {
            hot: Box::new_in(V3HotCore::from_cold_model(model), alloc),
        }
    }

    /// Returns an immutable reference to the underlying cache-coherent [`V3HotCore`].
    #[inline(always)]
    #[must_use]
    pub fn hot_core(&self) -> &V3HotCore<B> {
        &self.hot
    }

    /// Returns a mutable reference to the underlying cache-coherent [`V3HotCore`].
    #[inline(always)]
    pub fn hot_core_mut(&mut self) -> &mut V3HotCore<B> {
        &mut self.hot
    }

    /// Resets all recursive DSP state to exact resting `+0.0`.
    #[inline]
    pub fn reset_to_dc_equilibrium(&mut self) {
        self.hot.reset_to_dc_equilibrium();
    }

    /// Processes one sample pair (caller must hold [`crate::FtzDazGuard`] for batch loops).
    #[inline(always)]
    pub fn step_stereo(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        self.hot.step_stereo(in_l, in_r)
    }

    /// Processes a contiguous stereo audio block with [`crate::FtzDazGuard`] enabled.
    #[inline(never)]
    pub fn process_block_stereo(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        self.hot.process_block_stereo(in_l, in_r, out_l, out_r);
    }

    /// Infallible stereo block execution (`Result<(), !>`).
    #[inline(never)]
    pub fn process_block_infallible(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> Result<(), !> {
        self.hot.process_block_infallible(in_l, in_r, out_l, out_r)
    }
}

/// Concrete `DefaultMathBackend` entrypoints emitted directly into `dnam_engine` `.rlib` so
/// hardware/disassembly audits verify concrete NEON/SSE codegen (`0 FDIV`, `0 FSQRT`, `0 alloc`).
#[inline(never)]
pub fn v3_process_block_default(
    core: &mut V3HotCore<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    core.process_block_stereo(in_l, in_r, out_l, out_r);
}

/// Concrete infallible `DefaultMathBackend` entrypoint returning `Result<(), !>`.
#[inline(never)]
pub fn v3_process_block_infallible_default(
    inst: &mut V3CoreInstance<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> Result<(), !> {
    inst.process_block_infallible(in_l, in_r, out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::ScalarBackend;
    use proptest::prelude::*;

    fn sample_nontrivial_cold_model(tier: Tier, os2x: bool, mono_out: bool) -> V3ColdTierModel {
        let mut m = V3ColdTierModel::identity_for_tier(tier, os2x, mono_out);
        m.svf.base_g = 0.065;
        m.svf.sec2_g_ratio = 1.45;
        m.svf.env_oct_sens = 2.1;
        m.svf.env_comp = 2.5;
        m.svf.det_hp_mix = 0.35;
        m.svf.damp_k1 = 0.65;
        m.svf.damp_k2 = 0.90;
        m.svf.sec2_bp_feed = 0.25;
        m.svf.stage2_feed_taps[..7].copy_from_slice(&[0.6, 0.25, 0.35, -0.15, 0.20, 0.15, 0.05]);
        m.svf.direct_taps_l[..7].copy_from_slice(&[0.05, 0.02, -0.01, 0.01, 0.03, 0.01, -0.02]);
        m.svf.direct_taps_r[..7].copy_from_slice(&[0.04, 0.03, -0.02, 0.02, 0.02, -0.01, 0.01]);

        let active_lanes = cascade::tier_cascade_lanes(tier);
        for b in 0..NUM_CASCADE_BLOCKS {
            m.blocks[b].gate_mix = if b == 0 { 0.25 } else { 0.0 };
            m.blocks[b].gate_sharpness = 12.0;
            for l in 0..active_lanes {
                let lf = l as f32;
                m.blocks[b].in_u_gain[l] = if b == 0 {
                    0.8 + 0.15 * lf
                } else {
                    0.1 * (lf + 1.0)
                };
                m.blocks[b].lp_pole[l] = (0.35 + 0.04 * lf).min(0.95);
                m.blocks[b].lp_mix[l] = 0.85;
                m.blocks[b].hp_mix[l] = 0.40;
                m.blocks[b].static_bias[l] = -0.35 + 0.12 * lf;
                m.blocks[b].env_bias_sens[l] = 0.18 - 0.03 * lf;
                m.blocks[b].skip_u_gain[l] = 0.22;
                m.blocks[b].prev_skip_gain[l] = 0.18;
                if b > 0 {
                    for s in 0..active_lanes {
                        m.blocks[b].inter_mat[l][s] = if l == s { 0.45 } else { -0.04 };
                    }
                }
            }
        }
        for l in 0..active_lanes {
            let w = 0.18 / (active_lanes as f32);
            m.pre_cab_w[0][l] = w * ((l as f32) + 1.0);
            m.pre_cab_w[1][l] = w * ((l as f32) + 0.8);
            m.direct_s2_w[0][l] = 0.08 * w;
            m.direct_s2_w[1][l] = -0.05 * w;
        }
        m.pre_cab_u_gain = [0.20, 0.18];
        for i in 0..tier_fir_segments(tier) * 16 {
            m.fir_l[i] = 0.45 * (0.88_f32).powi(i as i32);
            m.fir_r[i] = 0.42 * (0.86_f32).powi(i as i32);
        }
        for k in 0..tier_kautz_banks(tier) * 4 {
            m.kautz_poles[k] = (0.85 + 0.003 * (k as f32), 0.12 + 0.06 * (k as f32));
            m.kautz_c_l[k] = (0.04 * if k % 2 == 0 { 1.0 } else { -1.0 }, 0.02);
            m.kautz_c_r[k] = (0.035 * if k % 2 == 0 { 1.0 } else { -1.0 }, -0.015);
        }
        m
    }

    #[test]
    fn test_v3_zero_dc_equilibrium_all_tiers_and_os2x() {
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            for os2x in [false, true] {
                for mono_out in [false, true] {
                    let model = sample_nontrivial_cold_model(tier, os2x, mono_out);
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
    fn test_v3_monotone_tier_embedding_and_backend_parity() {
        let lite = sample_nontrivial_cold_model(Tier::Lite, true, false);
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

        // Embed Lite exactly into Full/Pro/Ultra (extra lanes/taps/poles stay 0).
        for higher_tier in [Tier::Full, Tier::Pro, Tier::Ultra] {
            let mut higher = lite.clone();
            higher.tier = higher_tier;
            let mut higher_inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&higher);
            let (mut hl, mut hr) = ([0.0_f32; 512], [0.0_f32; 512]);
            higher_inst.process_block_stereo(&in_l, &in_r, &mut hl, &mut hr);
            for i in 0..512 {
                assert!(
                    (hl[i] - l_simd[i]).abs() < 1.0e-5,
                    "{higher_tier:?} diff L at {i}"
                );
                assert!(
                    (hr[i] - r_simd[i]).abs() < 1.0e-5,
                    "{higher_tier:?} diff R at {i}"
                );
            }
        }

        assert!(V3HotCore::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false) <= 2048);
        assert!(V3HotCore::<DefaultMathBackend>::active_hot_bytes(Tier::Ultra, true) <= 10_240);
    }

    proptest! {
        #[test]
        fn prop_v3_bibo_bounded_and_finite(
            x0 in -20.0_f32..20.0_f32,
            x1 in -20.0_f32..20.0_f32,
            bias in -2.5_f32..2.5_f32,
            os2x in proptest::bool::ANY,
        ) {
            let mut model = sample_nontrivial_cold_model(Tier::Full, os2x, false);
            for b in 0..NUM_CASCADE_BLOCKS {
                model.blocks[b].static_bias[0] = bias;
                model.blocks[b].static_bias[1] = -bias;
            }
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
