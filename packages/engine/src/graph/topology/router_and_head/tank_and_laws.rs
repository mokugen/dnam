//! Spatial `Op-T` Tank Loop Saturator, Blended Quiet-Feedback Knee, Feature Bus Slots & Stage
//! Modulation Laws (`graph/topology/router_and_head/tank_and_laws.rs`).

use crate::math::{newton6_reciprocal_f64, widen_f32_bits};
use crate::types::Tier;

/// Largest `f32` bit pattern $\le 0.985$ (`0x3F7C28F5`, since `0x3F7C28F6 > 0.985`,
/// `ref[dsp.tank.loop-saturator]`).
pub const TANK_MAX_V_CEIL_BITS: u32 = crate::math::TANK_MAX_V_CEIL_BITS_DERIVED;

/// Largest `f32` bit pattern $\le 0.9990$ (`0x3F7FBE76`, since `0x3F7FBE77 > 0.9990`,
/// `ref[dsp.tank.quiet-feedback-knee]`).
pub const TANK_MAX_G_LIN_BITS: u32 = crate::math::TANK_MAX_G_LIN_BITS_DERIVED;

/// Evaluates the Spatial Tank unity-slope cubic loop saturator with explicit outer clamp to $\pm V_{\text{ceil}}$
/// (`0 < V_ceil <= f32::from_bits(0x3F7C28F5)`, `ref[dsp.tank.loop-saturator]`).
#[inline]
pub fn eval_tank_loop_saturator(u: f32, v_ceil: f32) -> Option<f32> {
    let max_ceil = f32::from_bits(TANK_MAX_V_CEIL_BITS);
    if !u.is_finite() || !v_ceil.is_finite() || !(v_ceil > 0.0 && v_ceil <= max_ceil) {
        return None;
    }
    let two_over_3v = if v_ceil.to_bits() == TANK_MAX_V_CEIL_BITS {
        (2.0_f64 / (3.0_f64 * (f32::from_bits(TANK_MAX_V_CEIL_BITS) as f64))) as f32
    } else {
        (2.0_f64 * newton6_reciprocal_f64(3.0_f64 * (v_ceil as f64))) as f32
    };
    let v = (u * two_over_3v).clamp(-1.0, 1.0);
    let prod = v_ceil * v;
    let half_sq = (0.5 * v) * v;
    let raw = prod * (1.5 - half_sq);
    Some(raw.clamp(-v_ceil, v_ceil))
}

/// Fast hot-path Spatial Tank loop saturator at the canonical `TANK_MAX_V_CEIL_BITS` ceiling (`0 FDIV`).
#[inline(always)]
pub fn eval_tank_loop_saturator_fast(u: f32) -> f32 {
    const V_CEIL: f32 = f32::from_bits(TANK_MAX_V_CEIL_BITS);
    const TWO_OVER_3V: f32 = (2.0_f64 / (3.0_f64 * (V_CEIL as f64))) as f32;
    let v = (u * TWO_OVER_3V).clamp(-1.0, 1.0);
    let prod = V_CEIL * v;
    let half_sq = (0.5 * v) * v;
    (prod * (1.5 - half_sq)).clamp(-V_CEIL, V_CEIL)
}

/// Fast hot-path Spatial Tank blended quiet-feedback knee with cold-prepared `inv_sigma = 1.0 / sigma` (`0 FDIV`).
#[inline(always)]
pub fn eval_tank_blended_feedback_knee_fast(u: f32, g_lin: f32, inv_sigma: f64, alpha: f64) -> f32 {
    let u64_val = u as f64;
    let g64 = g_lin as f64;
    let ratio = u64_val.abs() * inv_sigma;
    let denom = 1.0 + ratio * ratio;
    let inv_denom = newton6_reciprocal_f64(denom);
    let factor = (1.0 - alpha) + alpha * inv_denom;
    ((g64 * u64_val) * factor) as f32
}

/// Evaluates the Spatial Tank blended quiet-feedback knee
/// $K(u) = g_{\text{lin}} u ((1 - \alpha) + \alpha / (1 + (|u| / \sigma)^2))$ (`ref[dsp.tank.quiet-feedback-knee]`).
#[inline]
pub fn eval_tank_blended_feedback_knee(u: f32, g_lin: f32, sigma: f32, alpha: f32) -> Option<f32> {
    let max_gain = f32::from_bits(TANK_MAX_G_LIN_BITS);
    if !u.is_finite()
        || !g_lin.is_finite()
        || !sigma.is_finite()
        || !alpha.is_finite()
        || !(0.0..=max_gain).contains(&g_lin)
        || !(0.0..=1.0).contains(&alpha)
        || sigma <= 0.0
    {
        return None;
    }
    let u64_val = widen_f32_bits(u.to_bits())?;
    let g64 = widen_f32_bits(g_lin.to_bits())?;
    let s64 = widen_f32_bits(sigma.to_bits())?;
    let a64 = widen_f32_bits(alpha.to_bits())?;

    let inv_sigma = newton6_reciprocal_f64(s64);
    let ratio = u64_val.abs() * inv_sigma;
    let denom = 1.0 + ratio * ratio;
    let inv_denom = newton6_reciprocal_f64(denom);
    let factor = (1.0 - a64) + a64 * inv_denom;
    let inter = (g64 * u64_val) * factor;
    Some(inter as f32)
}

/// Returns `(modal_slot_range, other_stage_slot_range, fir_lr_slots)` for `tier` inside the
/// 256-D feature bus $\boldsymbol{\Phi}[n]$ (`ref[dsp.head.direct-register-readout]`).
pub const fn cabinet_feature_bus_slot_ranges(
    tier: Tier,
) -> (core::ops::Range<usize>, core::ops::Range<usize>, [usize; 2]) {
    match tier {
        Tier::Lite => (0..4, 4..62, [62, 63]),
        Tier::Full => (64..68, 68..126, [126, 127]),
        Tier::Pro => (128..136, 136..190, [190, 191]),
        Tier::Ultra => (192..208, 208..254, [254, 255]),
    }
}

/// Pair-groups 4 complex modal outputs `[y0, y1, y2, y3]` for L and R into the 4 Cabinet feature
/// bus slots `[y0_L + y2_L, y1_L + y3_L, y0_R + y2_R, y1_R + y3_R]` (`ref[dsp.head.direct-register-readout]`).
#[inline]
pub fn pack_cabinet_modal_block_features(y_l: &[f32; 4], y_r: &[f32; 4]) -> [f32; 4] {
    [
        y_l[0] + y_l[2],
        y_l[1] + y_l[3],
        y_r[0] + y_r[2],
        y_r[1] + y_r[3],
    ]
}

/// Evaluates the closed-form per-sample stage modulation laws (`ref[dsp.ctrl.dispatch.modulation-laws]`)
/// mapping micro-ramped `theta[0..16]` and base parameters into clamped stage parameters (`0 FDIV`, `0 FSQRT`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn eval_stage_modulation_laws(
    theta: &[f32; 16],
    base_send_gains: [f32; 2],
    base_tau_pre: f32,
    max_tau_pre: f32,
    base_tau_tank: f32,
    max_tau_tank: f32,
    base_g_lin: f32,
    base_a_diff: f32,
    base_width: f32,
) -> [f32; 16] {
    let max_g_lin = f32::from_bits(TANK_MAX_G_LIN_BITS);
    [
        (base_send_gains[0] + theta[0]).clamp(0.0, 2.0),
        (base_send_gains[1] + theta[1]).clamp(0.0, 2.0),
        (base_tau_pre + theta[2]).clamp(4.0, max_tau_pre.max(4.0)),
        (24.0 * theta[3]).clamp(-24.0, 24.0),
        (1.0 + theta[4]).clamp(0.0, 16.0),
        (1.0 + 0.25 * theta[5]).clamp(0.25, 4.0),
        (0.5 + theta[6]).clamp(0.0, 1.0),
        (1.0 + 0.25 * theta[7]).clamp(0.25, 4.0),
        (0.2 + 0.1 * theta[8]).clamp(0.0, 0.8),
        theta[9].clamp(-core::f32::consts::FRAC_PI_4, core::f32::consts::FRAC_PI_4),
        (0.5 + theta[10]).clamp(0.0, 1.0),
        (1.0 + theta[11]).clamp(0.0, 2.0),
        (base_tau_tank + theta[12]).clamp(4.0, max_tau_tank.max(4.0)),
        (base_g_lin + theta[13]).clamp(0.0, max_g_lin),
        (base_a_diff + theta[14]).clamp(-0.75, 0.75),
        (base_width + theta[15]).clamp(0.0, 2.0),
    ]
}

/// Validates a 64-byte-aligned `tensor_directory` slice entry (`byte_offset`, `byte_length`) against
/// its owning tier prefix boundary `tier_arena_bytes` (`ref[container.dnam.tensor-directory]`).
#[inline]
pub const fn validate_tensor_directory_slice(
    byte_offset: usize,
    byte_length: usize,
    tier_arena_bytes: usize,
) -> bool {
    if !byte_offset.is_multiple_of(64) || !byte_length.is_multiple_of(64) {
        return false;
    }
    match byte_offset.checked_add(byte_length) {
        Some(end) => end <= tier_arena_bytes,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container, strip_container_to_tier};
    use crate::graph::sona_core::SonaMatryoshkaModel;
    use crate::graph::tank::{SpatialTankHot, SpatialTankParams};
    use crate::graph::weights::pack_sona_matryoshka_to_arena;
    use crate::graph::{SonaColdTierModel, SonaHotSlab};
    use crate::math::DefaultMathBackend;
    use crate::schema::DnamJsonHeader;
    use crate::{DnamEngine, FtzDazGuard};
    use alloc::vec;
    use dnam_kernels::T1DcBlockerKernel;

    /// verify[dsp.core.matryoshka-cascade]
    /// verify[dsp.cabinet.fir-kautz-bank]
    /// verify[dsp.tank.feedback-loop]
    /// verify[dsp.tank.captured-freeze]
    /// verify[dsp.tank.dc-blocker]
    /// verify[dsp.tank.loop-saturator]
    /// verify[dsp.tank.quiet-feedback-knee]
    #[test]
    fn test_sona_stages_loop_saturator_blended_knee_dc_blocker_and_freeze() {
        let _guard = FtzDazGuard::new();

        let cold = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
        let [s2_l, s2_r] = slab.step_stereo(0.25, -0.25);
        assert!(s2_l.is_finite() && s2_r.is_finite());

        let mut s4 = SpatialTankHot::from_cold(&SpatialTankParams::identity());
        s4.freeze = true;
        let [w_l, w_r] = s4.step_stereo::<DefaultMathBackend>(0.5, -0.5);
        assert!(w_l.is_finite() && w_r.is_finite());

        let v_ceil = f32::from_bits(TANK_MAX_V_CEIL_BITS);
        assert!((v_ceil as f64) <= 0.985);
        assert!((f32::from_bits(TANK_MAX_V_CEIL_BITS + 1) as f64) > 0.985);

        for u in [0.985_f32, 1.5, 10.0, f32::from_bits(0x3f7f_f4d1)] {
            let sat_pos = eval_tank_loop_saturator(u, v_ceil).unwrap();
            let sat_neg = eval_tank_loop_saturator(-u, v_ceil).unwrap();
            assert!(sat_pos.to_bits() <= TANK_MAX_V_CEIL_BITS);
            assert_eq!(sat_pos, -sat_neg);
        }
        assert!(eval_tank_loop_saturator(1.0, f32::from_bits(TANK_MAX_V_CEIL_BITS + 1)).is_none());

        let g_max = f32::from_bits(TANK_MAX_G_LIN_BITS);
        assert!((g_max as f64) <= 0.999);
        assert!((f32::from_bits(TANK_MAX_G_LIN_BITS + 1) as f64) > 0.999);

        let k_lin = eval_tank_blended_feedback_knee(2.0, 0.8, 0.5, 0.0).unwrap();
        assert!((k_lin - 1.6).abs() < 1e-6);
        let k_rat = eval_tank_blended_feedback_knee(2.0, 0.8, 0.5, 1.0).unwrap();
        assert!((k_rat - (1.6 / 17.0)).abs() < 1e-6);
        assert!(eval_tank_blended_feedback_knee(1.0e9, 0.5, 0.05, 1.0).unwrap() > 0.0);
        assert!(
            eval_tank_blended_feedback_knee(1.0, f32::from_bits(TANK_MAX_G_LIN_BITS + 1), 0.5, 0.5)
                .is_none()
        );

        let mut dc = T1DcBlockerKernel::new();
        let mut dc_out = 0.0f32;
        for _ in 0..24_000 {
            dc_out = dc.step(1.0);
        }
        assert!(dc_out.abs() < 1e-5);
    }

    /// verify[dsp.head.direct-register-readout]
    /// verify[dsp.head.tier-equivalence]
    /// verify[dsp.head.lower-triangular]
    /// verify[dsp.router.topology-dispatch]
    #[test]
    fn test_head_feature_bus_slots_topologies_and_effective_tier_equivalence() {
        let _guard = FtzDazGuard::new();

        assert_eq!(
            cabinet_feature_bus_slot_ranges(Tier::Lite),
            (0..4, 4..62, [62, 63])
        );
        assert_eq!(
            cabinet_feature_bus_slot_ranges(Tier::Full),
            (64..68, 68..126, [126, 127])
        );
        assert_eq!(
            cabinet_feature_bus_slot_ranges(Tier::Pro),
            (128..136, 136..190, [190, 191])
        );
        assert_eq!(
            cabinet_feature_bus_slot_ranges(Tier::Ultra),
            (192..208, 208..254, [254, 255])
        );
        assert_eq!(
            pack_cabinet_modal_block_features(&[1.0, 2.0, 3.0, 4.0], &[-1.0, -2.0, -3.0, -4.0]),
            [4.0, 6.0, -4.0, -6.0]
        );

        let mut header = DnamJsonHeader::default_reference();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let mut mat = SonaMatryoshkaModel::dual_rate_default(false);
        for g in 0..mat.groups.len() {
            mat.groups[g].adaa_drive = [0.5; 4];
            mat.groups[g].svf_g = [0.22; 4];
        }
        for ro in &mut mat.readouts {
            ro.pre_cab_w_odd[0][0] = 0.25;
            ro.pre_cab_w_odd[1][0] = -0.25;
            ro.fir_l[0] = 0.4;
            ro.fir_r[0] = 0.4;
        }
        pack_sona_matryoshka_to_arena(&mat, &mut arena);
        let full_container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let in_l: [f32; 32] = core::array::from_fn(|n| 0.25 * ((n as f32) * 0.2).sin());
        let in_r: [f32; 32] = core::array::from_fn(|n| -0.20 * ((n as f32) * 0.3).cos());

        for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            let stripped = strip_container_to_tier(&full_container, tier).unwrap();
            let mut eng_unstripped =
                DnamEngine::<DefaultMathBackend>::from_bytes(&full_container, tier, 48_000)
                    .unwrap();
            let mut eng_stripped =
                DnamEngine::<DefaultMathBackend>::from_bytes(&stripped, tier, 48_000).unwrap();

            let mut out_ul = [0.0f32; 32];
            let mut out_ur = [0.0f32; 32];
            let mut out_sl = [0.0f32; 32];
            let mut out_sr = [0.0f32; 32];
            eng_unstripped.process_block(&in_l, &in_r, &mut out_ul, &mut out_ur);
            eng_stripped.process_block(&in_l, &in_r, &mut out_sl, &mut out_sr);
            for n in 0..32 {
                assert_eq!(out_ul[n].to_bits(), out_sl[n].to_bits());
                assert_eq!(out_ur[n].to_bits(), out_sr[n].to_bits());
            }
        }
    }

    /// verify[dsp.ctrl.dispatch.modulation-laws]
    /// verify[container.dnam.tensor-directory]
    #[test]
    fn test_stage_modulation_laws_and_tensor_directory_slices() {
        let theta_zero = [0.0f32; 16];
        let base = eval_stage_modulation_laws(
            &theta_zero,
            [1.0, 1.0],
            64.0,
            2048.0,
            149.0,
            4096.0,
            0.85,
            0.5,
            1.0,
        );
        assert_eq!(base[0], 1.0);
        assert_eq!(base[2], 64.0);
        assert_eq!(base[10], 0.5);
        assert_eq!(base[13], 0.85);
        assert_eq!(base[14], 0.5);
        assert_eq!(base[15], 1.0);

        let theta_extreme = [5000.0f32; 16];
        let clamped = eval_stage_modulation_laws(
            &theta_extreme,
            [1.0, 1.0],
            64.0,
            2048.0,
            149.0,
            4096.0,
            0.85,
            0.5,
            1.0,
        );
        assert_eq!(clamped[0], 2.0);
        assert_eq!(clamped[2], 2048.0);
        assert_eq!(clamped[3], 24.0);
        assert_eq!(clamped[4], 16.0);
        assert_eq!(clamped[5], 4.0);
        assert_eq!(clamped[10], 1.0);
        assert_eq!(clamped[13].to_bits(), TANK_MAX_G_LIN_BITS);
        assert_eq!(clamped[14], 0.75);
        assert_eq!(clamped[15], 2.0);

        assert!(validate_tensor_directory_slice(0, 64, 4_608));
        assert!(validate_tensor_directory_slice(128, 256, 4_608));
        assert!(!validate_tensor_directory_slice(32, 64, 4_608));
        assert!(!validate_tensor_directory_slice(64, 100, 4_608));
        assert!(!validate_tensor_directory_slice(4_608, 64, 4_608));
    }
}
