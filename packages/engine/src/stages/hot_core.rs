//! Hot Core Compatibility Facade delegating directly to [`crate::stages::sona_core`]
//! (`stages/hot_core.rs`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

pub mod instance;

pub use crate::stages::sona_core::{
    SonaHotSlab, SonaInstance, sona_process_block_default, sona_process_block_infallible_default,
};
pub use instance::{V3CoreInstance, v3_process_block_default, v3_process_block_infallible_default};

use crate::math::DefaultMathBackend;

/// Compatibility alias for [`SonaHotSlab`].
pub type V3HotCore<B = DefaultMathBackend> = SonaHotSlab<B>;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;
    use proptest::prelude::*;

    use crate::container::ZohRescaledParams;
    use crate::math::ScalarBackend;
    use crate::stages::sona_core::{
        SonaMatryoshkaModel, tier_fir_segments, tier_kautz_banks, tier_sona_groups,
    };
    use crate::types::Tier;

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

    #[test]
    fn test_sona_zero_dc_equilibrium_all_tiers_and_os2x() {
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
        assert!(lite_inst.hot_core().stage4_effective_cap() >= 2_048);
    }

    #[test]
    fn test_sona_live_tier_switching_and_block_vs_step_parity() {
        let mut mat = sample_nontrivial_matryoshka(true, false);
        mat.tier_os2x = [false, false, true, true];
        let tiers = mat.to_four_tiers();
        let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&tiers[0]);
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
            let model = mat.extract_tier(tier);
            let mut m2 = model.clone();
            m2.os2x = os2x;
            let mut inst_step = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&m2);
            let mut inst_blk = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&m2);
            const N: usize = 45;
            let il: [f32; N] = core::array::from_fn(|i| 0.35 * ((i as f32) * 0.17).sin());
            let ir: [f32; N] = core::array::from_fn(|i| -0.3 * ((i as f32) * 0.23).cos());
            let (mut sl, mut sr) = ([0.0_f32; N], [0.0_f32; N]);
            {
                let _ftz = crate::FtzDazGuard::new();
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

    proptest! {
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
