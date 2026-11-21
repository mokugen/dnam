//! Gauge-invariant `eff_z_stage` modulation sensitivity laws, continuous passive `freeze_mix`
//! and stage `tank_dry_mix` rules, `Op-T` `Q32` phase increments, `Op-N[Pre]` clamps, `SensoryQuad`
//! VCA bias, `Op-P` phasors, and per-block / per-`CoreQuad` / per-lane `ADAA` / `ParaLRU` / `TPT-SVF`
//! scaling (`#![no_std]`, `#![forbid(unsafe_code)]`, `ref[dsp.ctrl.dispatch.modulation-laws]`).

#![forbid(unsafe_code)]

mod transport_and_sensory;

pub use transport_and_sensory::*;

use core::f32::consts::PI;

use dnam_kernels::{OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, fp16_to_f32};

use crate::graph::slab::layout::{
    MAX_CORE_QUADS, MAX_TOTAL_SONA_LANES, NUM_SONA_BLOCKS, SONA_CORE_QUAD_BLOCK_AND_LOCAL,
};
use crate::math::{clamp_finite_f32, exp2_signed_oct_f32, sanitize_finite_f32};

/// Sequential unfused `binary32` dot product over `z_stage[0..active_dim]` starting from `+0.0`.
#[inline]
pub fn stage_latent_dot_fp16(weights_fp16: &[u16], z_stage: &[f32; 8], active_dim: usize) -> f32 {
    let n = active_dim.min(weights_fp16.len()).min(8);
    let mut sum = 0.0_f32;
    for k in 0..n {
        sum += fp16_to_f32(weights_fp16[k]) * z_stage[k];
    }
    sanitize_finite_f32(sum)
}

/// Evaluates modulated `svf_g` octave shift (`[-3.0, +3.0]`) from `stage_targets[0]` and `w_svf_g`.
#[inline]
pub fn eval_svf_g_oct(
    stage_target_0: f32,
    w_svf_g_oct_fp16: &[u16],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> f32 {
    clamp_finite_f32(
        stage_target_0 + stage_latent_dot_fp16(w_svf_g_oct_fp16, z_stage, active_dim),
        -3.0,
        3.0,
        0.0,
    )
}

/// Evaluates modulated `svf_k` damping delta (`[-0.8, 1.5]`) from `stage_targets[1]`.
#[inline]
pub fn eval_svf_k_delta(
    stage_target_1: f32,
    w_svf_k_delta_fp16: &[u16],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> f32 {
    clamp_finite_f32(
        stage_target_1 + stage_latent_dot_fp16(w_svf_k_delta_fp16, z_stage, active_dim),
        -0.8,
        1.5,
        0.0,
    )
}

/// Evaluates modulated Kautz octave delta (`[-1.5, 1.5]`) and damping delta (`[-1.5, 1.5]`).
#[inline]
pub fn eval_kautz_deltas(
    stage_targets: &[f32; 32],
    w_kautz_oct_fp16: &[u16],
    w_kautz_damp_fp16: &[u16],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> (f32, f32) {
    let oct = clamp_finite_f32(
        stage_targets[10] + stage_latent_dot_fp16(w_kautz_oct_fp16, z_stage, active_dim),
        -1.5,
        1.5,
        0.0,
    );
    let damp = clamp_finite_f32(
        stage_targets[11] + stage_latent_dot_fp16(w_kautz_damp_fp16, z_stage, active_dim),
        -1.5,
        1.5,
        0.0,
    );
    (oct, damp)
}

/// Evaluates `Op-T` Givens angle delta (`[-pi, +pi]`) from `stage_targets[14]` and `w_tank_psi`.
#[inline]
pub fn eval_tank_psi_delta(
    stage_target_14: f32,
    w_tank_psi_fp16: &[u16],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> f32 {
    clamp_finite_f32(
        stage_target_14 + stage_latent_dot_fp16(w_tank_psi_fp16, z_stage, active_dim),
        -PI,
        PI,
        0.0,
    )
}

/// Evaluates all 5 shared stage modulation deltas `[g_oct, k_delta, k_oct, k_damp, psi_delta]`.
#[inline]
pub fn eval_global_stage_mods(
    mods_fp16: &[u16; 96],
    stage_targets: &[f32; 32],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> [f32; 5] {
    let g_oct = eval_svf_g_oct(stage_targets[0], &mods_fp16[8..16], z_stage, active_dim);
    let k_delta = eval_svf_k_delta(stage_targets[1], &mods_fp16[16..24], z_stage, active_dim);
    let (k_oct, k_damp) = eval_kautz_deltas(
        stage_targets,
        &mods_fp16[24..32],
        &mods_fp16[32..40],
        z_stage,
        active_dim,
    );
    let psi_delta = eval_tank_psi_delta(stage_targets[14], &mods_fp16[40..48], z_stage, active_dim);
    [g_oct, k_delta, k_oct, k_damp, psi_delta]
}

/// Evaluates per-block (`Lite`–`Pro`) or per-`CoreQuad` (`Ultra`) `PolyADAA1` linear scale (`exp2`)
/// and `ParaLRU` signed bias delta (`theta_lru`) from `stage_targets` + `eff_z_stage`.
pub fn eval_block_and_group_adaa_lru(
    targets_and_z: (&[f32; 32], &[f32; 8]),
    lite_mods_fp16: &[u16; 96],
    ultra_ctrl_fp16: &[u16; 256],
    is_ultra: bool,
    out_core_quads: (&mut [f32; MAX_CORE_QUADS], &mut [f32; MAX_CORE_QUADS]),
) {
    let (t, z) = targets_and_z;
    let (quad_adaa_scale, quad_lru_bias) = out_core_quads;
    if !is_ultra {
        let mut block_adaa_scale = [1.0_f32; NUM_SONA_BLOCKS];
        let mut block_lru_bias = [0.0_f32; NUM_SONA_BLOCKS];
        for b in 0..NUM_SONA_BLOCKS {
            let mut sum_a = 0.0_f32;
            let mut sum_l = 0.0_f32;
            for k in 0..4 {
                sum_a += fp16_to_f32(lite_mods_fp16[48 + k * 4 + b]) * z[k];
                sum_l += fp16_to_f32(lite_mods_fp16[64 + k * 4 + b]) * z[k];
            }
            let th_a = clamp_finite_f32(t[2 + b] + sum_a, -1.5, 1.5, 0.0);
            let th_l = clamp_finite_f32(t[6 + b] + sum_l, -2.0, 2.0, 0.0);
            block_adaa_scale[b] = exp2_signed_oct_f32(th_a);
            block_lru_bias[b] = th_l;
        }
        for (g, &(blk, _)) in SONA_CORE_QUAD_BLOCK_AND_LOCAL.iter().enumerate() {
            quad_adaa_scale[g] = block_adaa_scale[blk];
            quad_lru_bias[g] = block_lru_bias[blk];
        }
    } else {
        for (g, &(blk, _)) in SONA_CORE_QUAD_BLOCK_AND_LOCAL.iter().enumerate() {
            let mut sum_a = 0.0_f32;
            let mut sum_l = 0.0_f32;
            for k in 0..8 {
                sum_a += fp16_to_f32(ultra_ctrl_fp16[k * 10 + g]) * z[k];
                sum_l += fp16_to_f32(ultra_ctrl_fp16[80 + k * 10 + g]) * z[k];
            }
            let th_a = clamp_finite_f32(t[2 + blk] + sum_a, -1.5, 1.5, 0.0);
            let th_l = clamp_finite_f32(t[6 + blk] + sum_l, -2.0, 2.0, 0.0);
            quad_adaa_scale[g] = exp2_signed_oct_f32(th_a);
            quad_lru_bias[g] = th_l;
        }
    }
}

/// Evaluates per-lane `(g_scale, k_delta, wf_delta)` arrays across `0..active_lanes`, applying
/// `Ultra` `SONA_META_CORE_MODS_PRESENT` (`[8][128]` row-major) sensitivity when active.
pub fn eval_lane_core_mods(
    global_gk: (f32, f32),
    z_stage: &[f32; 8],
    core_mods_fp16: &[u16; 1_024],
    use_core_mods: bool,
    active_lanes: usize,
    out_lanes: (
        &mut [f32; MAX_TOTAL_SONA_LANES],
        &mut [f32; MAX_TOTAL_SONA_LANES],
        &mut [f32; MAX_TOTAL_SONA_LANES],
    ),
) {
    let g_oct_global = clamp_finite_f32(global_gk.0, -3.0, 3.0, 0.0);
    let k_delta_global = clamp_finite_f32(global_gk.1, -0.8, 1.5, 0.0);
    let g_scale_global = exp2_signed_oct_f32(g_oct_global);
    let (lane_g_scale, lane_k_delta, lane_wf_delta) = out_lanes;
    let z_nonzero = z_stage.iter().any(|&v| v != 0.0);
    if !use_core_mods || !z_nonzero {
        lane_g_scale.fill(g_scale_global);
        lane_k_delta.fill(k_delta_global);
        lane_wf_delta.fill(0.0);
        return;
    }
    let active = active_lanes.min(MAX_TOTAL_SONA_LANES);
    for l in 0..active {
        let mut dg = 0.0_f32;
        let mut dk = 0.0_f32;
        let mut dw = 0.0_f32;
        for k in 0..8 {
            let zk = z_stage[k];
            if zk != 0.0 {
                let row = k * 128;
                dg += fp16_to_f32(core_mods_fp16[row + l]) * zk;
                dk += fp16_to_f32(core_mods_fp16[row + 40 + l]) * zk;
                dw += fp16_to_f32(core_mods_fp16[row + 80 + l]) * zk;
            }
        }
        let g_oct_l = clamp_finite_f32(g_oct_global + sanitize_finite_f32(dg), -3.0, 3.0, 0.0);
        lane_g_scale[l] = exp2_signed_oct_f32(g_oct_l);
        lane_k_delta[l] = k_delta_global + sanitize_finite_f32(dk);
        lane_wf_delta[l] = sanitize_finite_f32(dw);
    }
    lane_g_scale[active..].fill(g_scale_global);
    lane_k_delta[active..].fill(k_delta_global);
    lane_wf_delta[active..].fill(0.0);
}

/// Evaluates modulated `Op-N[Pre]` parameters `(gamma_bc, zoh_hold_depth, beta_q)` from `26..=28`.
#[inline]
pub fn eval_op_n_pre_modulated(
    base_gamma_bc: f32,
    base_zoh_hold_depth: f32,
    base_beta_q: f32,
    stage_params: &[f32; 32],
) -> (f32, f32, f32) {
    let gamma_bc = clamp_finite_f32(base_gamma_bc + stage_params[26], 0.0, 1.0, 0.0);
    let zoh_hold_depth = clamp_finite_f32(
        base_zoh_hold_depth + stage_params[27],
        0.0,
        OP_N_PRE_MAX_ZOH_HOLD_DEPTH,
        0.0,
    );
    let beta_q = clamp_finite_f32(
        base_beta_q + stage_params[28],
        0.0,
        OP_N_PRE_MAX_BETA_Q,
        0.0,
    );
    (gamma_bc, zoh_hold_depth, beta_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::f32_to_fp16;

    /// verify[dsp.ctrl.dispatch.modulation-laws]
    /// verify[dsp.op-n.core.control-sensitivity]
    #[test]
    fn test_modulation_laws_finite_guards_stage_mods_and_a2_parity() {
        assert_eq!(eval_op_t_freeze_mix_eff(0.4, 0.2, false), 0.6);
        assert_eq!(eval_op_t_freeze_mix_eff(0.0, -0.5, true), 1.0);
        assert_eq!(eval_transport_dry_mix_stage(0.5, 0.25), 0.75);
        assert_eq!(eval_sensory_vca_bias_slot30(-1.5), -1.0);

        let mut lite_mods = [0_u16; 96];
        lite_mods[64] = f32_to_fp16(0.5);
        let ultra_ctrl = [0_u16; 256];
        let mut st = [0.0_f32; 32];
        st[6] = 0.25;
        let zs = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (mut grp_a, mut grp_l) = ([1.0; MAX_CORE_QUADS], [0.0; MAX_CORE_QUADS]);
        eval_block_and_group_adaa_lru(
            (&st, &zs),
            &lite_mods,
            &ultra_ctrl,
            false,
            (&mut grp_a, &mut grp_l),
        );
        assert_eq!(grp_l[0], 0.75);
        // Mechanism A2 sign-flip parity: flipping both `b_lru_base` and `rect_sign` flips `b_lru_eff` at 0 ULP.
        let (sign_pos, sign_neg) = (1.0_f32, -1.0_f32);
        let b_pos = (0.2_f32 + sign_pos * grp_l[0]).clamp(-16.0, 16.0);
        let b_neg = (-0.2_f32 + sign_neg * grp_l[0]).clamp(-16.0, 16.0);
        assert_eq!(b_neg.to_bits(), (-b_pos).to_bits());

        let mut mods = [0_u16; 96];
        let mut t = [0.0_f32; 32];
        for slot in &mut mods[8..48] {
            *slot = f32_to_fp16(0.5);
        }
        t[0] = 0.5;
        t[1] = 0.5;
        t[10] = 0.25;
        t[11] = -0.25;
        t[14] = 0.2;
        let gmods = eval_global_stage_mods(&mods, &t, &zs, 8);
        assert_eq!(gmods[0], 1.0);
        assert_eq!(gmods[1], 1.0);
        assert!((gmods[2] - 0.75).abs() < 1.0e-6);
        assert!((gmods[3] - 0.25).abs() < 1.0e-6);
        assert!((gmods[4] - 0.7).abs() < 1.0e-6);
        assert_eq!(stage_latent_dot_fp16(&[f32_to_fp16(f32::NAN)], &zs, 1), 0.0);

        st[26] = 0.25;
        st[27] = 2.0;
        st[28] = 1.5;
        let (gbc, hold_d, bq) = eval_op_n_pre_modulated(0.5, 0.5, 1.0, &st);
        assert_eq!((gbc, hold_d, bq), (0.75, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, 2.5));

        let mut ultra_ctrl = [0_u16; 256];
        ultra_ctrl[0] = f32_to_fp16(0.5);
        ultra_ctrl[80] = f32_to_fp16(-0.5);
        eval_block_and_group_adaa_lru(
            (&st, &zs),
            &lite_mods,
            &ultra_ctrl,
            true,
            (&mut grp_a, &mut grp_l),
        );
        assert!(grp_a[0] > 1.0 && grp_l[0] < 0.0);

        let mut core_mods = [0_u16; 1_024];
        core_mods[0] = f32_to_fp16(0.5);
        core_mods[40] = f32_to_fp16(0.25);
        core_mods[80] = f32_to_fp16(0.3);
        let (mut lg, mut lk, mut lw) = (
            [1.0; MAX_TOTAL_SONA_LANES],
            [0.0; MAX_TOTAL_SONA_LANES],
            [0.0; MAX_TOTAL_SONA_LANES],
        );
        eval_lane_core_mods(
            (0.5, 0.25),
            &zs,
            &core_mods,
            true,
            40,
            (&mut lg, &mut lk, &mut lw),
        );
        assert!((lg[0] - 2.0).abs() < 1.0e-3);
        assert!((lk[0] - 0.5).abs() < 1.0e-6);
        assert!((lw[0] - 0.3).abs() < 1.0e-3);

        eval_lane_core_mods(
            (f32::NAN, 99.0),
            &zs,
            &core_mods,
            true,
            40,
            (&mut lg, &mut lk, &mut lw),
        );
        assert!(lg[0].is_finite() && (lg[0] - core::f32::consts::SQRT_2).abs() < 1.0e-3);
        assert!((lk[0] - 1.75).abs() < 1.0e-6);

        eval_lane_core_mods(
            (f32::NAN, f32::INFINITY),
            &[0.0; 8],
            &core_mods,
            true,
            40,
            (&mut lg, &mut lk, &mut lw),
        );
        assert_eq!(lg[0], 1.0);
        assert_eq!(lk[0], 0.0);
        assert_eq!(lw[0], 0.0);
    }
}
