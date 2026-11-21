//! 32-slot `W_ctrl` + `"CEXT"` dispatch evaluation, `xorshift64*` OU analog drift,
//! control stride/ratio helpers, and clamped linear micro-ramping (`ref[dsp.ctrl.dispatch.matrix]`,
//! `ref[dsp.ctrl.dispatch.micro-ramp]`, `ref[dsp.ctrl.dispatch.sparse-and-active-ramps]`).

use crate::math::{
    MathBackend, SimdVec4, const_f32_pow2, flush_subnormal_f32_bits, rcp_nr6_f64, round_f64,
};

/// Total `3 kHz` control dispatch output slots (`0..16` base + `16..32` `"CEXT"`).
pub const STAGE_TARGET_SLOTS: usize = 32;
/// Default `48 kHz` micro-ramp step pole (`1.0 / 16.0`).
pub const DEFAULT_ANTI_ZIPPER_POLE: f32 = 0.0625;
/// Threshold below which wire `anti_zipper_ramp_scale` falls back to [`DEFAULT_ANTI_ZIPPER_POLE`].
pub const ANTI_ZIPPER_SNAP_EPS: f32 = 1.0e-7;
/// Canonical `xorshift64*` PRNG initial seed for live OU drift (`ref[dsp.ctrl.dispatch.sparse-and-active-ramps]`).
pub const DEFAULT_DRIFT_PRNG_SEED: u64 = 0x853C_49E6_748F_EA9B;

/// Computes the control-bus decimation stride $K_{\text{ctrl}}(f_s) = \max(8, \mathrm{round}(f_s / 3000))$
/// using precomputed `binary64` reciprocal multiplication (`0 FDIV`, `ref[dsp.ctrl.op-m.ballistic-lfo]`).
#[inline]
pub fn control_stride_for_rate(sample_rate_hz: f32) -> u32 {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return 16;
    }
    let scaled = f64::from(sample_rate_hz) * rcp_nr6_f64(3_000.0);
    let rounded = round_f64(scaled) as i64;
    if rounded < 8 { 8 } else { rounded as u32 }
}

/// Computes the dimensionless control-tick duration ratio
/// $\rho_{\text{ctrl}} = (K_{\text{ctrl}}(f_s) / f_s) / (K_{\text{ctrl}}(f_{\text{ref}}) / f_{\text{ref}})$ (`0 FDIV`).
#[inline]
pub fn control_tick_ratio(sample_rate_hz: f32, ref_sample_rate_hz: f32) -> f64 {
    let fs = if sample_rate_hz.is_finite() && sample_rate_hz > 1.0 {
        f64::from(sample_rate_hz)
    } else {
        48_000.0
    };
    let fs_ref = if ref_sample_rate_hz.is_finite() && ref_sample_rate_hz > 1.0 {
        f64::from(ref_sample_rate_hz)
    } else {
        48_000.0
    };
    let k_host = f64::from(control_stride_for_rate(fs as f32));
    let k_ref = f64::from(control_stride_for_rate(fs_ref as f32));
    (k_host * fs_ref) * rcp_nr6_f64(fs * k_ref)
}

#[inline(always)]
fn any_nonzero_f32_4(v: &[f32; 4]) -> bool {
    v[0] != 0.0 || v[1] != 0.0 || v[2] != 0.0 || v[3] != 0.0
}

/// Dynamic input vectors (`q0..q3`, `qu`) and flags passed to 32-slot dispatch evaluation.
pub(crate) struct DispatchInputs<'a> {
    pub q16_head: &'a [f32; 4],
    pub q16_basis: &'a [f32; 4],
    pub q16_lfo: &'a [f32; 4],
    pub q16_env: &'a [f32; 4],
    pub qu: &'a [f32; 4],
    pub is_ultra: bool,
    pub cext_present: bool,
}

/// Matrix bank (`W_ctrl`, `W_ctrl_ultra`, `W_ctrl_ext`, `W_ctrl_ultra_ext`) evaluated into `stage_targets`.
pub(crate) struct DispatchMatrices<'a> {
    pub w_ctrl: &'a [[f32; 16]; 16],
    pub w_ctrl_ultra: &'a [[f32; 4]; 16],
    pub w_ctrl_ext: &'a [[f32; 16]; 16],
    pub w_ctrl_ultra_ext: &'a [[f32; 4]; 16],
}

/// Evaluates all 32 `stage_targets` across base `W_ctrl` (`0..16`) and optional `"CEXT"`
/// (`16..31`, reserving `[31] == 0.0`) using 4-lane `B::Vec4` FMA and balanced pairwise tree sum.
pub(crate) fn evaluate_w_ctrl_32<B: MathBackend>(
    inputs: DispatchInputs<'_>,
    mats: DispatchMatrices<'_>,
    stage_targets: &mut [f32; STAGE_TARGET_SLOTS],
) {
    let q0 = B::Vec4::load(inputs.q16_head);
    let q1 = B::Vec4::load(inputs.q16_basis);
    let q2 = B::Vec4::load(inputs.q16_lfo);
    let q3 = B::Vec4::load(inputs.q16_env);
    let v_qu = B::Vec4::load(inputs.qu);
    let q01_active = any_nonzero_f32_4(inputs.q16_head) || any_nonzero_f32_4(inputs.q16_basis);
    let qu_active = inputs.is_ultra && any_nonzero_f32_4(inputs.qu);

    for r in 0..16 {
        let row = &mats.w_ctrl[r];
        let mut acc = B::Vec4::load(&[row[8], row[9], row[10], row[11]])
            .mul(q2)
            .fma(B::Vec4::load(&[row[12], row[13], row[14], row[15]]), q3);
        if q01_active {
            acc = acc
                .fma(B::Vec4::load(&[row[0], row[1], row[2], row[3]]), q0)
                .fma(B::Vec4::load(&[row[4], row[5], row[6], row[7]]), q1);
        }
        if qu_active {
            acc = acc.fma(B::Vec4::load(&mats.w_ctrl_ultra[r]), v_qu);
        }
        stage_targets[r] = acc.horizontal_sum();
    }

    if inputs.cext_present {
        for r in 0..15 {
            let row = &mats.w_ctrl_ext[r];
            let mut acc = B::Vec4::load(&[row[8], row[9], row[10], row[11]])
                .mul(q2)
                .fma(B::Vec4::load(&[row[12], row[13], row[14], row[15]]), q3);
            if q01_active {
                acc = acc
                    .fma(B::Vec4::load(&[row[0], row[1], row[2], row[3]]), q0)
                    .fma(B::Vec4::load(&[row[4], row[5], row[6], row[7]]), q1);
            }
            if qu_active {
                acc = acc.fma(B::Vec4::load(&mats.w_ctrl_ultra_ext[r]), v_qu);
            }
            stage_targets[16 + r] = acc.horizontal_sum();
        }
    } else {
        stage_targets[16..32].fill(0.0);
    }
    stage_targets[31] = 0.0;
}

/// Steps optional `xorshift64*` Ornstein–Uhlenbeck analog drift state (`ou_drift_state[0..4]`)
/// for projection onto `eff_z_stage`, or resets drift state to `+0.0` when disabled.
#[inline]
pub(crate) fn step_ou_drift(
    live_analog_drift_enabled: bool,
    offline_bounce: bool,
    sigma_drift: f32,
    rho_drift_host: f32,
    drift_prng_state: &mut u64,
    ou_drift_state: &mut [f32; 4],
) {
    if !(live_analog_drift_enabled && !offline_bounce && sigma_drift > 0.0) {
        *ou_drift_state = [0.0; 4];
        return;
    }
    let scale = sigma_drift * (1.0 - rho_drift_host);
    for d in 0..4 {
        let mut x = if *drift_prng_state == 0 {
            DEFAULT_DRIFT_PRNG_SEED
        } else {
            *drift_prng_state
        };
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *drift_prng_state = x;
        let rand_u32 = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32;
        let xi = ((rand_u32 >> 8) as f32) * const_f32_pow2(-23) - 1.0;
        ou_drift_state[d] =
            flush_subnormal_f32_bits(rho_drift_host * ou_drift_state[d] + scale * xi);
    }
}

/// Recomputes per-sample linear micro-ramp deltas uniformly across all 32 slots (`0..32`,
/// reserving `[31] == 0.0`, `ref[dsp.ctrl.dispatch.micro-ramp]`).
#[inline]
pub(crate) fn prepare_micro_ramp_deltas<B: MathBackend>(
    stage_targets: &[f32; STAGE_TARGET_SLOTS],
    prev_stage_targets: &mut [f32; STAGE_TARGET_SLOTS],
    current_stage_params: &mut [f32; STAGE_TARGET_SLOTS],
    stage_ramp_deltas: &mut [f32; STAGE_TARGET_SLOTS],
    alpha_ramp: f32,
) {
    *prev_stage_targets = *current_stage_params;
    let v_alpha = B::Vec4::splat(alpha_ramp);
    let (target_quads, _) = stage_targets.as_chunks::<4>();
    let (prev_quads, _) = prev_stage_targets.as_chunks::<4>();
    let (delta_quads, _) = stage_ramp_deltas.as_chunks_mut::<4>();
    for ((delta_q, target_q), prev_q) in delta_quads
        .iter_mut()
        .zip(target_quads.iter())
        .zip(prev_quads.iter())
    {
        let target_v = B::Vec4::load(target_q);
        let prev_v = B::Vec4::load(prev_q);
        target_v.sub(prev_v).mul(v_alpha).store(delta_q);
    }
    stage_ramp_deltas[31] = 0.0;
    current_stage_params[31] = 0.0;
    prev_stage_targets[31] = 0.0;
}

/// Advances all 32 micro-ramps uniformly by one VM frame with bounded interval clamping
/// (no overshoot) and final end-of-stride snap (`[31] == 0.0`).
#[inline]
pub(crate) fn advance_micro_ramp_frame<B: MathBackend>(
    stage_targets: &[f32; STAGE_TARGET_SLOTS],
    prev_stage_targets: &[f32; STAGE_TARGET_SLOTS],
    current_stage_params: &mut [f32; STAGE_TARGET_SLOTS],
    stage_ramp_deltas: &[f32; STAGE_TARGET_SLOTS],
    ctrl_sub_counter: u32,
    ctrl_stride: u32,
) {
    if ctrl_sub_counter + 1 >= ctrl_stride {
        *current_stage_params = *stage_targets;
        current_stage_params[31] = 0.0;
        return;
    }
    let (target_quads, _) = stage_targets.as_chunks::<4>();
    let (prev_quads, _) = prev_stage_targets.as_chunks::<4>();
    let (delta_quads, _) = stage_ramp_deltas.as_chunks::<4>();
    let (cur_quads, _) = current_stage_params.as_chunks_mut::<4>();
    for (((cur_q, target_q), prev_q), delta_q) in cur_quads
        .iter_mut()
        .zip(target_quads.iter())
        .zip(prev_quads.iter())
        .zip(delta_quads.iter())
    {
        let target_v = B::Vec4::load(target_q);
        let prev_v = B::Vec4::load(prev_q);
        let cur_v = B::Vec4::load(cur_q);
        let delta_v = B::Vec4::load(delta_q);
        let lo_v = prev_v.min(target_v);
        let hi_v = prev_v.max(target_v);
        cur_v.add(delta_v).clamp(lo_v, hi_v).store(cur_q);
    }
    current_stage_params[31] = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_stride_ratio_and_uniform_32_slot_ramping() {
        assert_eq!(control_stride_for_rate(44_100.0), 15);
        assert_eq!(control_stride_for_rate(48_000.0), 16);
        assert_eq!(control_stride_for_rate(96_000.0), 32);
        assert!((control_tick_ratio(48_000.0, 48_000.0) - 1.0).abs() < 1e-12);
        assert!((control_tick_ratio(96_000.0, 48_000.0) - 1.0).abs() < 1e-12);

        let mut targets = [0.0_f32; 32];
        targets[0] = 1.0;
        targets[16] = -0.6;
        targets[30] = 0.8;
        targets[31] = 99.0;
        let mut prev = [0.0_f32; 32];
        let mut cur = [0.0_f32; 32];
        let mut deltas = [0.0_f32; 32];
        prepare_micro_ramp_deltas::<crate::math::ScalarBackend>(
            &targets,
            &mut prev,
            &mut cur,
            &mut deltas,
            0.0625,
        );
        assert_eq!(cur[16], 0.0);
        assert!((deltas[16] - (-0.6 * 0.0625)).abs() < 1e-7);
        assert!((deltas[30] - (0.8 * 0.0625)).abs() < 1e-7);
        assert_eq!(deltas[31], 0.0);
        assert_eq!(cur[31], 0.0);

        advance_micro_ramp_frame::<crate::math::ScalarBackend>(
            &targets, &prev, &mut cur, &deltas, 0, 16,
        );
        assert!((cur[0] - 0.0625).abs() < 1e-6);
        assert!((cur[16] - (-0.0375)).abs() < 1e-6);
        assert!((cur[30] - 0.05).abs() < 1e-6);
        assert_eq!(cur[31], 0.0);

        // Verify interval clamping prevents overshoot before end-of-stride snap.
        let mut cur_clamp = [0.75_f32; 32];
        let prev_clamp = [0.0_f32; 32];
        let target_clamp = [0.8_f32; 32];
        let deltas_clamp = [0.5_f32; 32];
        advance_micro_ramp_frame::<crate::math::ScalarBackend>(
            &target_clamp,
            &prev_clamp,
            &mut cur_clamp,
            &deltas_clamp,
            2,
            16,
        );
        assert_eq!(cur_clamp[16], 0.8);
        assert_eq!(cur_clamp[31], 0.0);

        advance_micro_ramp_frame::<crate::math::ScalarBackend>(
            &targets, &prev, &mut cur, &deltas, 15, 16,
        );
        assert_eq!(cur[0], 1.0);
        assert_eq!(cur[16], -0.6);
        assert_eq!(cur[30], 0.8);
        assert_eq!(cur[31], 0.0);

        let w = [[0.25_f32; 16]; 16];
        let wu = [[0.5_f32; 4]; 16];
        let mut out_t = [0.0_f32; 32];
        evaluate_w_ctrl_32::<crate::math::ScalarBackend>(
            DispatchInputs {
                q16_head: &[1.0, 0.0, 0.0, 0.0],
                q16_basis: &[0.0, 1.0, 0.0, 0.0],
                q16_lfo: &[0.0, 0.0, 1.0, 0.0],
                q16_env: &[0.0, 0.0, 0.0, 1.0],
                qu: &[0.5, 0.0, 0.0, 0.0],
                is_ultra: true,
                cext_present: true,
            },
            DispatchMatrices {
                w_ctrl: &w,
                w_ctrl_ultra: &wu,
                w_ctrl_ext: &w,
                w_ctrl_ultra_ext: &wu,
            },
            &mut out_t,
        );
        assert_eq!(out_t[0], 1.25);
        assert_eq!(out_t[16], 1.25);
        assert_eq!(out_t[31], 0.0);

        evaluate_w_ctrl_32::<crate::math::ScalarBackend>(
            DispatchInputs {
                q16_head: &[1.0, 0.0, 0.0, 0.0],
                q16_basis: &[0.0, 1.0, 0.0, 0.0],
                q16_lfo: &[0.0, 0.0, 1.0, 0.0],
                q16_env: &[0.0, 0.0, 0.0, 1.0],
                qu: &[0.5, 0.0, 0.0, 0.0],
                is_ultra: true,
                cext_present: false,
            },
            DispatchMatrices {
                w_ctrl: &w,
                w_ctrl_ultra: &wu,
                w_ctrl_ext: &w,
                w_ctrl_ultra_ext: &wu,
            },
            &mut out_t,
        );
        assert_eq!(out_t[0], 1.25);
        assert_eq!(&out_t[16..32], &[0.0; 16]);

        let mut prng = 0_u64;
        let mut drift = [0.0_f32; 4];
        let before_targets = out_t;
        step_ou_drift(true, false, 0.1, 0.95, &mut prng, &mut drift);
        assert_ne!(prng, 0);
        assert_ne!(drift, [0.0; 4]);
        assert_eq!(out_t, before_targets);
        step_ou_drift(true, true, 0.1, 0.95, &mut prng, &mut drift);
        assert_eq!(drift, [0.0; 4]);
    }
}
