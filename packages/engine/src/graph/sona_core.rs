//! `SONA` `v3.0` Top-of-Frame Crossbar, 4-Block Matryoshka Core, Early FIR + Kautz Cabinet & Spatial
//! `Op-T` Tank Engine Core (`graph/sona_core.rs`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

pub mod frontend;
pub mod model;
pub mod slab;
pub mod tick16;

use alloc::alloc::Global;
use alloc::boxed::Box;
use core::alloc::Allocator;

pub use frontend::{SonaOversamplerParams, SonaStereoOversampler2x, ZeroLagOversampler2x};
pub use model::{
    FIR_RING_CAP, FIR_RING_MASK, MAX_FIR_SEGMENTS, MAX_GROUPS_PER_BLOCK, MAX_KAUTZ_BANKS,
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_SONA_TILES, MAX_STAGE_LATENTS, MAX_TANK_LINES,
    MAX_TANK_QUADS, MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS, NUM_SONA_MACRO_KNOBS,
    SONA_BLOCK_GROUP_SLOTS, SONA_FIR_MAX_TAPS, SONA_GROUP_BLOCK_AND_LOCAL, SONA_INTERBLOCK_TILES,
    SonaColdTierModel, SonaControlBusParams, SonaCrossbarConfig, SonaGroupCold,
    SonaMatryoshkaModel, SonaTierReadout, SonaTileCold, flat_sona_lane_index,
    tier_block_groups_array, tier_fir_segments, tier_kautz_banks, tier_sona_groups, tier_spec,
    tier_tank_quads, tier_tile_count,
};
pub use slab::{
    SonaControlBusState, SonaGroupHot, SonaGroupReadoutHot, SonaHotSlab, SonaInterBlockTileHot,
};

use crate::graph::GraphBackend;
use crate::math::DefaultMathBackend;

/// Owned heap/arena-allocated [`SonaHotSlab`] parameterized by [`core::alloc::Allocator`].
#[derive(Clone, Debug)]
pub struct SonaInstance<B: GraphBackend = DefaultMathBackend, A: Allocator = Global> {
    hot: Box<SonaHotSlab<B>, A>,
}

impl<B: GraphBackend> SonaInstance<B, Global> {
    /// Constructs a [`SonaInstance`] in the global allocator from `model`.
    #[must_use]
    pub fn from_cold_model(model: &SonaColdTierModel) -> Self {
        Self::from_cold_model_in(model, Global)
    }
}

impl<B: GraphBackend, A: Allocator> SonaInstance<B, A> {
    /// Constructs a [`SonaInstance`] inside a custom [`core::alloc::Allocator`] (`Box::new_in`).
    #[must_use]
    pub fn from_cold_model_in(model: &SonaColdTierModel, alloc: A) -> Self {
        Self {
            hot: Box::new_in(SonaHotSlab::from_cold_model(model), alloc),
        }
    }

    /// Returns an immutable reference to the underlying cache-coherent [`SonaHotSlab`].
    #[inline(always)]
    #[must_use]
    pub fn hot_core(&self) -> &SonaHotSlab<B> {
        &self.hot
    }

    /// Returns a mutable reference to the underlying cache-coherent [`SonaHotSlab`].
    #[inline(always)]
    #[must_use]
    pub fn hot_core_mut(&mut self) -> &mut SonaHotSlab<B> {
        &mut self.hot
    }

    /// Switches active tier & readout weights in-place (`0` alloc, `0` reset).
    #[inline]
    pub fn switch_tier(&mut self, model: &SonaColdTierModel) {
        self.hot.switch_tier(model);
    }

    /// Reloads a new model in-place and resets runtime state (`0` heap alloc, `0` drop).
    #[inline]
    pub fn reload_in_place(&mut self, model: &SonaColdTierModel) {
        self.hot.reload_in_place(model);
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

/// Concrete `DefaultMathBackend` entrypoint emitted into `dnam_engine` `.rlib` for disassembly
/// audits (`0 FDIV`, `0 FSQRT`, `0 IDIV/UDIV`, `0 alloc`).
#[inline(never)]
pub fn sona_process_block_default(
    core: &mut SonaHotSlab<DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    core.process_block_stereo(in_l, in_r, out_l, out_r);
}

/// Concrete infallible `DefaultMathBackend` entrypoint returning `Result<(), !>`.
#[inline(never)]
pub fn sona_process_block_infallible_default(
    inst: &mut SonaInstance<DefaultMathBackend>,
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
    use crate::container::ZohRescaledParams;
    use crate::math::ScalarBackend;
    use crate::types::{FtzDazGuard, Tier};

    #[test]
    fn test_sona_hot_slab_all_tiers_parity_subblock_and_zoh() {
        let _guard = FtzDazGuard::new();
        for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            for os2x in [false, true] {
                let mut cold = SonaColdTierModel::identity_for_tier(tier, os2x, false);
                cold.groups[0].adaa_drive = [1.2, 0.8, 0.4, 0.2];
                cold.groups[0].tone_lp = [0.6; 4];
                cold.groups[0].tone_dir = [0.4; 4];
                cold.readout.pre_cab_w_odd[0][0] = 0.35;
                cold.readout.pre_cab_w_odd[1][0] = -0.35;
                cold.tiles[0].w_odd_4x4[0][0] = 0.25;
                cold.crossbar.op_p_crossbar_gain = 0.3;
                cold.crossbar.op_p_to_group[0] = [0.2, 0.0, 0.0, 0.0];
                cold.crossbar.phasor_freq_hz = [2.5, 5.0, 7.5, 10.0];
                cold.crossbar.phasor_to_group[0] = [0.1, 0.0, 0.0, 0.0];

                let mut simd_inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&cold);
                let mut scal_inst = SonaInstance::<ScalarBackend>::from_cold_model(&cold);
                let mut blk_inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&cold);

                assert_eq!(simd_inst.step_stereo(0.0, 0.0).map(f32::to_bits), [0, 0]);
                simd_inst.reset_to_dc_equilibrium();

                let mut in_l = [0.0_f32; 32];
                let mut in_r = [0.0_f32; 32];
                for n in 0..32 {
                    in_l[n] = 0.5 * ((n as f32) * 0.19).sin();
                    in_r[n] = 0.4 * ((n as f32) * 0.13).cos();
                }
                let mut blk_l = [0.0_f32; 32];
                let mut blk_r = [0.0_f32; 32];
                blk_inst
                    .process_block_infallible(&in_l, &in_r, &mut blk_l, &mut blk_r)
                    .unwrap();

                for n in 0..32 {
                    let s_simd = simd_inst.step_stereo(in_l[n], in_r[n]);
                    let s_scal = scal_inst.step_stereo(in_l[n], in_r[n]);
                    assert!(s_simd[0].is_finite() && s_simd[1].is_finite());
                    assert!(
                        (s_simd[0] - s_scal[0]).abs() < 1.0e-5,
                        "Scalar/SIMD parity L diff tier={tier:?} os2x={os2x} n={n}"
                    );
                    assert!(
                        (s_simd[0] - blk_l[n]).abs() < 1.0e-6,
                        "step_stereo vs tick16 L diff tier={tier:?} os2x={os2x} n={n}"
                    );
                    assert!(
                        (s_simd[1] - blk_r[n]).abs() < 1.0e-6,
                        "step_stereo vs tick16 R diff tier={tier:?} os2x={os2x} n={n}"
                    );
                }

                // ZOH reconfigure across 44.1..192 kHz + clean equilibrium reset
                for sr in [44_100_u32, 96_000, 192_000] {
                    let zoh = ZohRescaledParams::new(48_000, sr);
                    simd_inst.hot_core_mut().reconfigure_with_zoh(&cold, &zoh);
                    let z_out = simd_inst.step_stereo(0.1, -0.1);
                    assert!(z_out[0].is_finite() && z_out[1].is_finite());
                }
                simd_inst.reset_to_dc_equilibrium();
                assert_eq!(simd_inst.step_stereo(0.0, 0.0).map(f32::to_bits), [0, 0]);
            }
        }
    }

    /// verify[dsp.head.mid-side-factorization]
    #[test]
    fn test_head_mid_side_factorization_mono_preserves_mid_slice() {
        let mut mat = SonaMatryoshkaModel::identity(false, false);
        mat.groups[0].in_x_weight[0] = 0.5;
        mat.groups[0].adaa_drive[0] = 1.0;
        for ro in &mut mat.readouts {
            ro.direct_w_odd[0][0] = 0.8;
            ro.direct_w_odd[1][0] = -0.2;
        }
        let stereo_model = mat.extract_tier(Tier::Lite);
        let mut mono_model = stereo_model.clone();
        mono_model.mono_out = true;

        let mut s_eng = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&stereo_model);
        let mut m_eng = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&mono_model);
        for i in 0..32 {
            let x = 0.35 * ((i as f32) * 0.19).sin();
            let [yl, yr] = s_eng.step_stereo(x, x);
            let [ml, mr] = m_eng.step_stereo(x, x);
            let expected_mid = 0.5 * (yl + yr);
            assert_eq!(ml.to_bits(), mr.to_bits());
            assert!(
                (ml - expected_mid).abs() < 1.0e-5,
                "i={i} ml={ml} expected_mid={expected_mid}"
            );
        }
    }

    fn sample_nontrivial_matryoshka(os2x: bool, mono_out: bool) -> SonaMatryoshkaModel {
        let mut m = SonaMatryoshkaModel::identity(os2x, mono_out);
        for g in 0..m.groups.len() {
            m.groups[g].svf_g = [0.18 + 0.02 * (g as f32); 4];
            m.groups[g].svf_k = [0.65; 4];
            m.groups[g].lru_alpha_rel = [0.92; 4];
            m.groups[g].lru_alpha_atk = [0.15; 4];
            m.groups[g].lru_kappa = [0.25; 4];
            m.groups[g].adaa_drive = [1.2, 0.8, 0.4, 0.2];
            m.groups[g].in_x_weight = if g == 0 {
                [0.45, -0.35, 0.25, -0.15]
            } else {
                [0.12, -0.08, 0.05, 0.03]
            };
            m.groups[g].tone_lp = [0.5; 4];
            m.groups[g].tone_dir = [0.5; 4];
        }
        for t in 0..m.tiles.len() {
            m.tiles[t].w_odd_4x4[0][0] = 0.22;
            m.tiles[t].w_even_4x4[1][1] = 0.14;
            m.tiles[t].w_lru_4x4[2][2] = 0.10;
        }
        for (t_idx, &tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
            .iter()
            .enumerate()
        {
            let r = &mut m.readouts[t_idx];
            r.pre_cab_u_gain = [0.22, 0.19];
            r.pre_cab_hp_gain = [0.06, -0.04];
            r.direct_s2_u_gain = [0.02, -0.02];
            r.direct_s2_hp_gain = [0.03, 0.02];
            for idx in 0..(tier_sona_groups(tier) << 2) {
                r.pre_cab_w_odd[0][idx] = 0.08 * if idx % 2 == 0 { 1.0 } else { -0.8 };
                r.pre_cab_w_odd[1][idx] = 0.07 * if idx % 2 == 0 { 1.0 } else { -0.8 };
                r.direct_w_odd[0][idx] = 0.015;
                r.direct_w_odd[1][idx] = -0.012;
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
        m.stage0.clean_sub_mix = 0.08;
        m.stage0.gate_thresh = 0.005;
        m.stage0.gate_hyst = 0.002;
        m.stage0.gate_inv_knee = 120.0;
        m.stage0.gate_floor = 0.1;
        m.crossbar.op_p_crossbar_gain = 0.25;
        m.crossbar.op_p_to_group[0] = [0.15; 4];
        m.crossbar.phasor_freq_hz = [1.5, 3.0, 4.5, 6.0];
        m.crossbar.phasor_to_group[0] = [0.1; 4];
        m.ctrl.w_svf_g_oct[0] = 0.8;
        m.ctrl.w_tank_psi[0] = 0.35;
        for tank in &mut m.stage4 {
            tank.wet_mix = 0.15;
            tank.g_loop = 0.82;
        }
        m
    }

    fn exercise_sona_backend<B: GraphBackend>(os2x: bool, mono_out: bool) {
        let mat = SonaMatryoshkaModel::identity(os2x, mono_out);
        let four = mat.to_four_tiers();
        assert_eq!(mat.extract_tier(Tier::Full).tier, Tier::Full);
        assert_eq!(four[3].tier, Tier::Ultra);

        let mut inst = SonaInstance::<B>::from_cold_model(&four[3]);
        assert_eq!(inst.hot_core().tier(), Tier::Ultra);
        let bytes_ultra = SonaHotSlab::<B>::active_hot_bytes(Tier::Ultra, os2x);
        inst.switch_tier(&four[0]);
        assert!(SonaHotSlab::<B>::active_hot_bytes(Tier::Lite, os2x) < bytes_ultra);
        inst.switch_tier(&four[1]);
        inst.switch_tier(&four[2]);
        inst.switch_tier(&four[3]);
        inst.reset_to_dc_equilibrium();

        let n = 48usize;
        let clean_l: alloc::vec::Vec<f32> =
            (0..n).map(|i| 0.4 * ((i as f32) * 0.21).sin()).collect();
        let clean_r: alloc::vec::Vec<f32> =
            (0..n).map(|i| 0.4 * ((i as f32) * 0.17).cos()).collect();
        let mut dirty_l = clean_l.clone();
        let mut dirty_r = clean_r.clone();
        dirty_l[4] = f32::NAN;
        dirty_r[4] = f32::INFINITY;
        dirty_l[5] = f32::NEG_INFINITY;
        dirty_r[5] = -100.0;
        let (mut out_l, mut out_r) = (alloc::vec![0.0_f32; n], alloc::vec![0.0_f32; n]);
        inst.process_block_stereo(&clean_l, &clean_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().chain(out_r.iter()).all(|v| v.is_finite()));
        assert!(inst.step_stereo(0.25, -0.25).iter().all(|v| v.is_finite()));

        let (mut out_il, mut out_ir) = (alloc::vec![0.0_f32; n], alloc::vec![0.0_f32; n]);
        let Ok(()) = inst.process_block_infallible(&dirty_l, &dirty_r, &mut out_il, &mut out_ir);
        assert!(out_il.iter().chain(out_ir.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_sona_core_instance_backends_tiers_and_default_entrypoints() {
        let _ftz = FtzDazGuard::new();
        for &(os2x, mono_out) in &[(false, false), (true, false), (false, true), (true, true)] {
            exercise_sona_backend::<ScalarBackend>(os2x, mono_out);
            exercise_sona_backend::<DefaultMathBackend>(os2x, mono_out);
            #[cfg(target_arch = "aarch64")]
            {
                exercise_sona_backend::<crate::math::ArmSve2Backend>(os2x, mono_out);
                exercise_sona_backend::<crate::math::ArmSmeBackend>(os2x, mono_out);
            }
        }
        let ultra_model = SonaColdTierModel::identity_for_tier(Tier::Ultra, false, false);
        let mut core_direct = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&ultra_model);
        let mut inst_def = SonaInstance::<DefaultMathBackend>::from_cold_model(&ultra_model);
        let in_l = [0.2_f32; 24];
        let in_r = [-0.2_f32; 24];
        let (mut out_l, mut out_r) = ([0.0_f32; 24], [0.0_f32; 24]);
        sona_process_block_default(&mut core_direct, &in_l, &in_r, &mut out_l, &mut out_r);
        let Ok(()) = sona_process_block_infallible_default(
            &mut inst_def,
            &in_l,
            &in_r,
            &mut out_l,
            &mut out_r,
        );
        assert!(out_l.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sona_zero_dc_equilibrium_all_tiers_and_os2x() {
        for os2x in [false, true] {
            for mono_out in [false, true] {
                let mat = sample_nontrivial_matryoshka(os2x, mono_out);
                for tier in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
                    let model = mat.extract_tier(tier);
                    let mut inst =
                        SonaInstance::<DefaultMathBackend, _>::from_cold_model_in(&model, &Global);
                    assert_eq!((&raw const *inst.hot_core() as usize) % 64, 0);
                    let in_z = [0.0_f32; 256];
                    let (mut out_l, mut out_r) = ([1.0_f32; 256], [1.0_f32; 256]);
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
    fn test_sona_matryoshka_parity_zoh_and_ctrl_targets() {
        let mat = sample_nontrivial_matryoshka(true, false);
        let lite = mat.extract_tier(Tier::Lite);
        let mut lite_inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&lite);
        let mut scalar_inst = SonaInstance::<ScalarBackend>::from_cold_model(&lite);
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
        assert!(SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false) <= 4_608);
        assert!(SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Ultra, true) <= 28_672);
        lite_inst
            .hot_core_mut()
            .ensure_cold_tank_cap(&lite.stage4, 2_048);
        lite_inst.hot_core_mut().reload_in_place(&lite);
        assert!(lite_inst.hot_core().tank_effective_cap() >= 2_048);
    }

    #[test]
    fn test_sona_live_tier_switching_and_block_vs_step_parity() {
        let mut mat = sample_nontrivial_matryoshka(true, false);
        mat.tier_os2x = [false, false, true, true];
        let tiers = mat.to_four_tiers();
        let mut inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&tiers[0]);
        let (mut out_l, mut out_r) = ([0.0_f32; 256], [0.0_f32; 256]);
        let in_l: [f32; 256] = core::array::from_fn(|i| 0.4 * ((i as f32) * 0.04).sin());
        let in_r: [f32; 256] = core::array::from_fn(|i| 0.35 * ((i as f32) * 0.04).cos());
        for (block_idx, &tier_idx) in [0usize, 1, 2, 3, 2, 0].iter().enumerate() {
            if block_idx > 0 {
                inst.switch_tier(&tiers[tier_idx]);
            }
            inst.process_block_stereo(&in_l, &in_r, &mut out_l, &mut out_r);
            for &y in out_l.iter().chain(out_r.iter()) {
                assert!(y.is_finite() && y.abs() < 50.0);
            }
        }
        for &(tier, os2x) in &[
            (Tier::Lite, false),
            (Tier::Full, false),
            (Tier::Pro, true),
            (Tier::Ultra, true),
        ] {
            let mut m2 = mat.extract_tier(tier);
            m2.os2x = os2x;
            let mut inst_step = SonaInstance::<DefaultMathBackend>::from_cold_model(&m2);
            let mut inst_blk = SonaInstance::<DefaultMathBackend>::from_cold_model(&m2);
            const N: usize = 45;
            let il: [f32; N] = core::array::from_fn(|i| 0.35 * ((i as f32) * 0.17).sin());
            let ir: [f32; N] = core::array::from_fn(|i| -0.3 * ((i as f32) * 0.23).cos());
            let (mut sl, mut sr) = ([0.0_f32; N], [0.0_f32; N]);
            {
                let _ftz = FtzDazGuard::new();
                for i in 0..N {
                    let [yl, yr] = inst_step.step_stereo(il[i], ir[i]);
                    sl[i] = yl;
                    sr[i] = yr;
                }
            }
            let (mut bl, mut br) = ([0.0_f32; N], [0.0_f32; N]);
            inst_blk.process_block_stereo(&il[..11], &ir[..11], &mut bl[..11], &mut br[..11]);
            inst_blk.process_block_stereo(&il[11..], &ir[11..], &mut bl[11..], &mut br[11..]);
            let max_diff = sl
                .iter()
                .zip(sr.iter())
                .zip(bl.iter().zip(br.iter()))
                .map(|((&a, &b), (&c, &d))| (a - c).abs().max((b - d).abs()))
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff < 1.0e-5,
                "tier={tier:?} os2x={os2x} diff={max_diff}"
            );
        }
    }

    proptest::proptest! {
        #[test]
        fn prop_sona_bibo_bounded_and_finite(
            x0 in -20.0_f32..20.0_f32,
            x1 in -20.0_f32..20.0_f32,
            bias in -3.5_f32..3.5_f32,
            os2x in proptest::bool::ANY,
        ) {
            let mut mat = sample_nontrivial_matryoshka(os2x, false);
            for g in &mut mat.groups {
                g.lru_bias_drive = [bias, -bias, 0.5 * bias, -0.5 * bias];
            }
            let model = mat.extract_tier(Tier::Full);
            let mut inst = SonaInstance::<DefaultMathBackend>::from_cold_model(&model);
            let _ftz = FtzDazGuard::new();
            for s in 0..256 {
                let u0 = if s % 2 == 0 { x0 } else { x1 };
                let u1 = if s % 3 == 0 { -x1 } else { x0 };
                let [yl, yr] = inst.step_stereo(u0, u1);
                proptest::prop_assert!(yl.is_finite() && yr.is_finite());
                proptest::prop_assert!(yl.abs() < 2_000.0 && yr.abs() < 2_000.0);
            }
        }
    }
}
