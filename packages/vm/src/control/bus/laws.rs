//! Gauge-invariant `z_stage` modulation sensitivity laws, Schmitt `(0.45, 0.55)` write-enable
//! latch, Quad-0-only `dry_mix` rule, `Op-T` `Q32` phase increments, and Mechanism A2-equivariant
//! per-block/per-group `ADAA` / `ParaLRU` scaling (`ref[dsp.ctrl.dispatch.modulation-laws]`).

use dnam_kernels::fp16_to_f32;
use zerocopy::byteorder::little_endian::U16;

use crate::math::{ScalarVec4, SimdVec4, TANK_MAX_G_LIN_BITS_DERIVED, round_half_away_f64};

#[inline(always)]
fn exp2_clamped5_f32(x: f32) -> f32 {
    ScalarVec4::splat(x).exp2_clamped5().to_array()[0]
}

/// Block index mapping for all 10 SONA groups `{0,4,6,8}->0, {1,5,9}->1, {2,7}->2, {3}->3`.
pub const SONA_GROUP_BLOCK_MAP: [usize; 10] = [0, 1, 2, 3, 0, 1, 0, 2, 0, 1];

/// Sequential unfused `binary32` dot product over `z_stage[0..active_dim]` starting from `+0.0`.
#[inline]
pub fn stage_latent_dot_fp16(weights: &[U16], z_stage: &[f32; 8], active_dim: usize) -> f32 {
    let n = active_dim.min(weights.len()).min(8);
    let mut sum = 0.0_f32;
    for k in 0..n {
        sum += fp16_to_f32(weights[k].get()) * z_stage[k];
    }
    sum
}

/// Evaluates modulated `svf_g` octave shift (`[-3.0, +3.0]`) and clamped `[1e-5, 64.0]` linear `g`.
#[inline]
pub fn eval_svf_g_modulated(
    base_svf_g: f32,
    stage_target_0: f32,
    w_svf_g_oct_fp16: &[U16; 8],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> (f32, f32) {
    let oct = (stage_target_0 + stage_latent_dot_fp16(w_svf_g_oct_fp16, z_stage, active_dim))
        .clamp(-3.0, 3.0);
    let g_eff = (base_svf_g * exp2_clamped5_f32(oct)).clamp(1.0e-5, 64.0);
    (oct, g_eff)
}

/// Evaluates modulated `svf_k` damping (`[0.0, 8.0]`) from `stage_targets[1]`.
#[inline]
pub fn eval_svf_k_modulated(
    base_svf_k: f32,
    stage_target_1: f32,
    w_svf_k_delta_fp16: &[U16; 8],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> (f32, f32) {
    let delta = (stage_target_1 + stage_latent_dot_fp16(w_svf_k_delta_fp16, z_stage, active_dim))
        .clamp(-0.8, 1.5);
    let k_eff = (base_svf_k + delta).clamp(0.0, 8.0);
    (delta, k_eff)
}

/// Evaluates modulated Kautz octave delta (`[-1.5, 1.5]`) and damping delta (`[-1.5, 1.5]`).
#[inline]
pub fn eval_kautz_deltas(
    stage_targets: &[f32; 32],
    w_kautz_oct_fp16: &[U16; 8],
    w_kautz_damp_fp16: &[U16; 8],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> (f32, f32) {
    let oct = (stage_targets[10] + stage_latent_dot_fp16(w_kautz_oct_fp16, z_stage, active_dim))
        .clamp(-1.5, 1.5);
    let damp = (stage_targets[11] + stage_latent_dot_fp16(w_kautz_damp_fp16, z_stage, active_dim))
        .clamp(-1.5, 1.5);
    (oct, damp)
}

/// Evaluates `Op-T` target Givens angle `target_psi` (`[-pi, +pi]`) from `stage_targets[14]` and
/// `w_tank_psi_fp16`.
#[inline]
pub fn eval_tank_target_psi(
    base_psi: f32,
    stage_target_14: f32,
    w_tank_psi_fp16: &[U16; 8],
    z_stage: &[f32; 8],
    active_dim: usize,
) -> f32 {
    let pi = core::f32::consts::PI;
    let delta = (stage_target_14 + stage_latent_dot_fp16(w_tank_psi_fp16, z_stage, active_dim))
        .clamp(-pi, pi);
    (base_psi + delta).clamp(-pi, pi)
}

/// Evaluates Crossbar & Tank `wet_scale` (`clamp(1.0 + clamp(stage_targets[15], -1.0, 1.0), 0.0, 2.0)`).
#[inline]
pub fn eval_crossbar_wet_scale(stage_target_15: f32) -> f32 {
    (1.0 + stage_target_15.clamp(-1.0, 1.0)).clamp(0.0, 2.0)
}

/// Evaluates slot `16` (`op_t_we_drive`) Schmitt hysteresis (`0.45` set / `0.55` release),
/// returning updated `WE_HOLD_ACTIVE` flag.
#[inline]
pub fn update_op_t_we_hold_schmitt(stage_target_16: f32, prev_hold_active: bool) -> bool {
    let we_eff = (1.0 + stage_target_16).clamp(0.0, 1.0);
    (dnam_kernels::update_we_schmitt_flag(u8::from(prev_hold_active), we_eff)
        & dnam_kernels::QUAD_FLAG_WE_HOLD_ACTIVE)
        != 0
}

/// Evaluates Quad-0-only `dry_mix` modulation from slot `22` (`op_t_dry_mix_mod`), leaving
/// `quad_idx >= 1` unmodified (`ref[dsp.ctrl.dispatch.modulation-laws]`).
#[inline]
pub fn eval_tank_dry_mix_for_quad(quad_idx: usize, base_dry_mix: f32, stage_target_22: f32) -> f32 {
    if quad_idx == 0 {
        (base_dry_mix + stage_target_22).clamp(-2.0, 2.0)
    } else {
        base_dry_mix
    }
}

/// Evaluates `Op-T` loop feedback gain from slot `13`, clamped to `[0.0, 0x3F7F_BE76]`.
#[inline]
pub fn eval_tank_g_loop(base_g_loop: f32, stage_target_13: f32) -> f32 {
    (base_g_loop + stage_target_13).clamp(0.0, f32::from_bits(TANK_MAX_G_LIN_BITS_DERIVED))
}

/// Converts effective `ramp_rate_hz` (`base + t[18]` clamped `[-16, +16]`) to signed `i32`
/// `Q32` phase increment using `inv_fs_vm` and round-half-away-from-zero.
#[inline]
pub fn eval_tank_ramp_inc_q32(
    base_ramp_rate_hz: f32,
    stage_target_18: f32,
    inv_fs_vm_f64: f64,
) -> i32 {
    let rate_eff = (base_ramp_rate_hz + stage_target_18).clamp(-16.0, 16.0);
    round_half_away_f64(f64::from(rate_eff) * 4_294_967_296.0 * inv_fs_vm_f64) as i64 as i32
}

/// Converts effective `lfo_rate_hz` (`base + t[19]` clamped `[0, 16]`) to unsigned `u32` `Q32`
/// phase increment, returning `0` when rate or max depth `<= 0.0`.
#[inline]
pub fn eval_tank_lfo_inc_q32(
    base_lfo_rate_hz: f32,
    stage_target_19: f32,
    max_lfo_depth_eff: f32,
    inv_fs_vm_f64: f64,
) -> u32 {
    let rate_eff = (base_lfo_rate_hz + stage_target_19).clamp(0.0, 16.0);
    if rate_eff <= 0.0 || max_lfo_depth_eff <= 0.0 {
        return 0;
    }
    (f64::from(rate_eff) * 4_294_967_296.0 * inv_fs_vm_f64 + 0.5) as u64 as u32
}

/// Evaluates modulated `Op-N[Pre]` parameters from slots `26..=28`.
#[inline]
pub fn eval_op_n_pre_modulated(
    base_gamma_bc: f32,
    base_zoh_hold_depth: f32,
    base_beta_q: f32,
    stage_params: &[f32; 32],
) -> (f32, f32, f32) {
    let gamma_bc = (base_gamma_bc + stage_params[26]).clamp(0.0, 1.0);
    let zoh_hold_depth = (base_zoh_hold_depth + stage_params[27]).clamp(0.0, 0.9999);
    let beta_q = (base_beta_q + stage_params[28]).clamp(0.0, 16.0);
    (gamma_bc, zoh_hold_depth, beta_q)
}

/// Group baseline & Mechanism A2 sign configuration for `PolyADAA1` and `ParaLRU` modulation.
#[derive(Clone, Copy, Debug)]
pub struct GroupModBase {
    /// Wire baseline `adaa_drive` before `exp2_clamped5(theta_adaa)`.
    pub base_adaa_drive: f32,
    /// Wire baseline `lru_bias_drive` before signed `rect_sign * theta_lru` shift.
    pub base_lru_bias_drive: f32,
    /// Mechanism A2 sign parity (`-1.0` or `+1.0`).
    pub rect_sign: f32,
}

/// Evaluates `PolyADAA1` drive and Mechanism A2-equivariant `ParaLRU` bias for group `g in 0..10`.
#[inline]
pub fn eval_group_adaa_and_lru_effective(
    group_idx: usize,
    is_ultra: bool,
    stage_params: &[f32; 32],
    z_stage: &[f32; 8],
    lite_mods_fp16: &[U16; 96],
    ultra_ctrl_fp16: &[U16; 256],
    base: GroupModBase,
) -> (f32, f32) {
    let g = group_idx.min(9);
    let b = SONA_GROUP_BLOCK_MAP[g];
    let (adaa_dot, lru_dot) = if is_ultra {
        let mut a = 0.0_f32;
        let mut l = 0.0_f32;
        for k in 0..8 {
            a += fp16_to_f32(ultra_ctrl_fp16[k * 10 + g].get()) * z_stage[k];
            l += fp16_to_f32(ultra_ctrl_fp16[80 + k * 10 + g].get()) * z_stage[k];
        }
        (a, l)
    } else {
        let mut a = 0.0_f32;
        let mut l = 0.0_f32;
        for k in 0..4 {
            a += fp16_to_f32(lite_mods_fp16[48 + k * 4 + b].get()) * z_stage[k];
            l += fp16_to_f32(lite_mods_fp16[64 + k * 4 + b].get()) * z_stage[k];
        }
        (a, l)
    };
    let theta_adaa = (stage_params[2 + b] + adaa_dot).clamp(-1.5, 1.5);
    let theta_lru = (stage_params[6 + b] + lru_dot).clamp(-2.0, 2.0);
    let d_adaa_eff = base.base_adaa_drive.clamp(-32.0, 32.0) * exp2_clamped5_f32(theta_adaa);
    let sign = if base.rect_sign < 0.0 { -1.0 } else { 1.0 };
    let b_lru_eff = (base.base_lru_bias_drive + sign * theta_lru).clamp(-16.0, 16.0);
    (d_adaa_eff, b_lru_eff)
}

/// Base core lane parameters modulated by `stage_latent_core_mods_fp16`.
#[derive(Clone, Copy, Debug)]
pub struct UltraCoreLaneBase {
    /// Wire `svf_g` (`48 kHz` / ZOH baseline).
    pub svf_g_base: f32,
    /// Shared `svf_g_oct` offset from `eval_svf_g_modulated`.
    pub svf_g_oct: f32,
    /// Wire `svf_k` damping baseline.
    pub svf_k_base: f32,
    /// Shared `svf_k_delta` offset from `eval_svf_k_modulated`.
    pub svf_k_delta: f32,
    /// Wire `wavefolder_mix` baseline.
    pub wavefolder_mix_base: f32,
}

/// Applies Ultra `SONA_META_CORE_MODS_PRESENT` (`[8][128]` row-major) per-lane modulation on
/// global lane `lane in 0..40`, returning `(svf_g_eff, svf_k_eff, wavefolder_mix_eff)`.
#[inline]
pub fn apply_ultra_core_mods_lane(
    lane: usize,
    core_mods_fp16: &[U16; 1_024],
    z_stage: &[f32; 8],
    base: UltraCoreLaneBase,
) -> (f32, f32, f32) {
    let l = lane.min(39);
    let mut g_dot = 0.0_f32;
    let mut k_dot = 0.0_f32;
    let mut wf_dot = 0.0_f32;
    for k in 0..8 {
        let row = k * 128;
        let z = z_stage[k];
        g_dot += fp16_to_f32(core_mods_fp16[row + l].get()) * z;
        k_dot += fp16_to_f32(core_mods_fp16[row + 40 + l].get()) * z;
        wf_dot += fp16_to_f32(core_mods_fp16[row + 80 + l].get()) * z;
    }
    let g_oct = (base.svf_g_oct + g_dot).clamp(-3.0, 3.0);
    let g_eff = (base.svf_g_base * exp2_clamped5_f32(g_oct)).clamp(1.0e-5, 64.0);
    let k_eff = (base.svf_k_base + base.svf_k_delta + k_dot).clamp(0.0, 8.0);
    let wf_eff = (base.wavefolder_mix_base + wf_dot).clamp(0.0, 1.0);
    (g_eff, k_eff, wf_eff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_kernels::f32_to_fp16;

    #[test]
    fn test_schmitt_quad0_dry_mix_and_a2_parity() {
        assert!(!update_op_t_we_hold_schmitt(0.0, false));
        assert!(update_op_t_we_hold_schmitt(-0.6, false));
        assert!(update_op_t_we_hold_schmitt(-0.5, true));
        assert!(update_op_t_we_hold_schmitt(-0.45, true));
        assert!(!update_op_t_we_hold_schmitt(-0.44, true));
        assert!(!update_op_t_we_hold_schmitt(-0.4, true));

        assert_eq!(eval_tank_dry_mix_for_quad(0, 0.5, 0.25), 0.75);
        assert_eq!(eval_tank_dry_mix_for_quad(1, 0.5, 0.25), 0.5);

        let mut lite_mods = [U16::new(0); 96];
        lite_mods[64] = U16::new(f32_to_fp16(0.5));
        let ultra_ctrl = [U16::new(0); 256];
        let mut st = [0.0; 32];
        st[6] = 0.25;
        let zs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_, b_pos) = eval_group_adaa_and_lru_effective(
            0,
            false,
            &st,
            &zs,
            &lite_mods,
            &ultra_ctrl,
            GroupModBase {
                base_adaa_drive: 1.0,
                base_lru_bias_drive: 0.2,
                rect_sign: 1.0,
            },
        );
        let (_, b_neg) = eval_group_adaa_and_lru_effective(
            0,
            false,
            &st,
            &zs,
            &lite_mods,
            &ultra_ctrl,
            GroupModBase {
                base_adaa_drive: 1.0,
                base_lru_bias_drive: -0.2,
                rect_sign: -1.0,
            },
        );
        assert_eq!(b_neg.to_bits(), (-b_pos).to_bits());
        let (d_oob, b_oob) = eval_group_adaa_and_lru_effective(
            10,
            true,
            &st,
            &zs,
            &lite_mods,
            &ultra_ctrl,
            GroupModBase {
                base_adaa_drive: 1.0,
                base_lru_bias_drive: 0.2,
                rect_sign: 1.0,
            },
        );
        assert!(d_oob.is_finite() && b_oob.is_finite());

        let w = [U16::new(f32_to_fp16(0.5)); 8];
        assert!((stage_latent_dot_fp16(&w, &zs, 8) - 0.5).abs() < 1.0e-6);
        let (oct, g_eff) = eval_svf_g_modulated(0.5, 0.5, &w, &zs, 8);
        assert_eq!(oct, 1.0);
        assert!((g_eff - 1.0).abs() < 1.0e-4);
        let (k_delta, k_eff) = eval_svf_k_modulated(1.0, 0.5, &w, &zs, 8);
        assert_eq!(k_delta, 1.0);
        assert_eq!(k_eff, 2.0);
        st[10] = 0.25;
        st[11] = -0.25;
        let (k_oct, k_damp) = eval_kautz_deltas(&st, &w, &w, &zs, 8);
        assert!((k_oct - 0.75).abs() < 1.0e-6);
        assert!((k_damp - 0.25).abs() < 1.0e-6);
        let psi = eval_tank_target_psi(0.1, 0.2, &w, &zs, 8);
        assert!((psi - 0.8).abs() < 1.0e-6);
        assert_eq!(eval_crossbar_wet_scale(0.5), 1.5);
        assert_eq!(eval_crossbar_wet_scale(-2.0), 0.0);
        assert!(
            (eval_tank_g_loop(0.95, 0.5) - f32::from_bits(TANK_MAX_G_LIN_BITS_DERIVED)).abs()
                < 1.0e-7
        );

        let inv_48k = 1.0 / 48_000.0;
        let r_q32 = eval_tank_ramp_inc_q32(2.0, -4.0, inv_48k);
        assert!(r_q32 < 0);
        assert_eq!(eval_tank_lfo_inc_q32(0.0, 0.0, 1.0, inv_48k), 0);
        assert_eq!(eval_tank_lfo_inc_q32(1.0, 0.0, 0.0, inv_48k), 0);
        assert!(eval_tank_lfo_inc_q32(1.0, 0.5, 0.5, inv_48k) > 0);
        st[26] = 0.25;
        st[27] = 2.0;
        st[28] = 1.5;
        let (gbc, hold_d, bq) = eval_op_n_pre_modulated(0.5, 0.5, 1.0, &st);
        assert_eq!((gbc, hold_d, bq), (0.75, 0.9999, 2.5));

        let mut ultra_ctrl = [U16::new(0); 256];
        ultra_ctrl[0] = U16::new(f32_to_fp16(0.5));
        ultra_ctrl[80] = U16::new(f32_to_fp16(-0.5));
        let (d_u, b_u) = eval_group_adaa_and_lru_effective(
            0,
            true,
            &st,
            &zs,
            &lite_mods,
            &ultra_ctrl,
            GroupModBase {
                base_adaa_drive: 1.0,
                base_lru_bias_drive: 0.5,
                rect_sign: 1.0,
            },
        );
        assert!(d_u > 1.0 && b_u < 0.5);

        let mut core_mods = [U16::new(0); 1_024];
        core_mods[0] = U16::new(f32_to_fp16(0.5));
        core_mods[40] = U16::new(f32_to_fp16(0.25));
        core_mods[80] = U16::new(f32_to_fp16(0.3));
        let (cg, ck, cwf) = apply_ultra_core_mods_lane(
            0,
            &core_mods,
            &zs,
            UltraCoreLaneBase {
                svf_g_base: 0.5,
                svf_g_oct: 0.5,
                svf_k_base: 1.0,
                svf_k_delta: 0.25,
                wavefolder_mix_base: 0.2,
            },
        );
        assert!(
            (cg - 1.0).abs() < 1.0e-4 && (ck - 1.5).abs() < 1.0e-6 && (cwf - 0.5).abs() < 1.0e-3
        );
    }
}
