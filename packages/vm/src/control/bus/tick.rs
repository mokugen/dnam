//! `Op-N[Ctrl]` exact-integer static evaluation, four independent 4-D trust-region clips,
//! and `Op-M[Ctrl]` skewed `Q0.192` LFO + `SensoryQuad` envelope latching (`ref[dsp.ctrl.op-n.engine]`,
//! `ref[dsp.ctrl.op-n.trust-clip]`, `ref[dsp.ctrl.modulators.dynamic]`).

use crate::control::cp1::{eval_uint_base_4x8, eval_uint_ultra_delta_3x4x8};
use crate::control::exact_int::{ExactIntFactorAccumulator, ExactIntMainAccumulator};
use crate::control::{ControlClampDescriptor, Cp1ContributorSlot};
use crate::math::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_SIGN_MASK, MAX_ROTOR_ANGLE_F32_BITS, Q0_192Phase,
    const_abs_f32, flush_subnormal_f32_bits, sanitize_finite_f32, step_lfo_phasor_zero_hold,
    trust_clip_outward_4d,
};

#[inline]
fn contributor_basis_and_scalar(
    slot: &Cp1ContributorSlot,
    descriptors: &[ControlClampDescriptor],
    norm_vals: &[f32],
    active_switches: &[u8],
) -> ([f32; 3], f32) {
    let p = usize::from(slot.control_idx);
    match slot.ctrl_kind {
        0 => {
            let u_p = norm_vals[p];
            (descriptors[p].eval_lobatto_basis(u_p), u_p)
        }
        1 => {
            let def_opt = descriptors[p].default_switch_option();
            let active_opt = active_switches[p];
            let active = if slot.aux == 0xFF {
                active_opt != def_opt
            } else {
                active_opt != def_opt && active_opt == slot.aux
            };
            let s = if active { 1.0_f32 } else { 0.0_f32 };
            ([s, 0.0, 0.0], s)
        }
        2 => {
            let q = usize::from(slot.aux);
            let lob_p = descriptors[p].eval_lobatto_basis(norm_vals[p]);
            let lob_q = descriptors[q].eval_lobatto_basis(norm_vals[q]);
            let c0 = lob_p[0] * lob_q[0];
            ([c0, lob_p[1] * lob_q[0], lob_p[0] * lob_q[1]], c0)
        }
        _ => ([0.0; 3], 0.0),
    }
}

/// Read-only static `CP1` + `U_int` inputs for boundary evaluation.
pub(crate) struct StaticEvalInputs<'a> {
    pub descriptors: &'a [ControlClampDescriptor],
    pub norm_vals: &'a [f32],
    pub active_switches: &'a [u8],
    pub contributors: &'a [Cp1ContributorSlot],
    pub distinct_controls: usize,
    pub is_ultra: bool,
    pub trust_radius_mh: f32,
    pub u_int_base: &'a [[f32; 8]; 4],
    pub u_int_ultra: &'a [[[f32; 8]; 4]; 3],
}

/// Mutable exact-integer accumulators and clipped 4-D static latent outputs.
pub(crate) struct StaticEvalOutputs<'a> {
    pub exact_fm_base: &'a mut [ExactIntFactorAccumulator; 4],
    pub exact_fm_ultra: &'a mut [ExactIntFactorAccumulator; 4],
    pub z_head: &'a mut [f32; 4],
    pub ctrl_basis_4: &'a mut [f32; 4],
    pub z_head_ultra: &'a mut [f32; 4],
    pub z_stage_ultra: &'a mut [f32; 4],
}

/// Evaluates static `CP1` contributors through `9-word` linear sums + `9/18-word` FM accumulators,
/// adds `U_int` when `distinct_contributor_controls >= 2`, and applies 4 independent 4-D trust clips.
pub(crate) fn evaluate_static_cp1_latents(inp: StaticEvalInputs<'_>, out: StaticEvalOutputs<'_>) {
    let mut main_base = [ExactIntMainAccumulator::zero(); 8];
    let mut main_ultra = [ExactIntMainAccumulator::zero(); 8];

    for (c_idx, slot) in inp.contributors.iter().enumerate() {
        let (lob, scalar) =
            contributor_basis_and_scalar(slot, inp.descriptors, inp.norm_vals, inp.active_switches);
        let lanes = usize::from(slot.lanes).min(3);
        let has_emb = slot.emb.iter().any(|&w| const_abs_f32(w) > 1.0e-12);
        let z_c = if has_emb {
            ((slot.emb[0] * lob[0]) + (slot.emb[1] * lob[1])) + (slot.emb[2] * lob[2])
        } else {
            scalar
        };

        for l in 0..lanes {
            let lb = lob[l];
            for d in 0..8 {
                let prod_b = slot.main_base[l][d] * lb;
                main_base[d].add_f32_bits_rt(prod_b.to_bits());
                if inp.is_ultra && slot.has_ultra {
                    let prod_u = slot.main_ultra[l][d] * lb;
                    main_ultra[d].add_f32_bits_rt(prod_u.to_bits());
                }
            }
        }
        for f in 0..4 {
            let v_b = slot.v_base[f] * z_c;
            out.exact_fm_base[f].update_contributor_rt(c_idx, v_b.to_bits());
            let v_u = if inp.is_ultra && slot.has_ultra {
                slot.v_ultra[f] * z_c
            } else {
                0.0
            };
            out.exact_fm_ultra[f].update_contributor_rt(c_idx, v_u.to_bits());
        }
    }

    let mut z_main_base = [0.0_f32; 8];
    let mut z_main_ultra = [0.0_f32; 8];
    for d in 0..8 {
        z_main_base[d] = main_base[d].to_f32();
        z_main_ultra[d] = main_ultra[d].to_f32();
    }

    let use_interactions = inp.distinct_controls >= 2;
    let mut z_fm_base = [0.0_f32; 4];
    let mut z_fm_ultra = [0.0_f32; 4];
    if use_interactions {
        for f in 0..4 {
            if let Some(bits) = out.exact_fm_base[f].interaction_f32_bits() {
                z_fm_base[f] = f32::from_bits(bits);
            }
            if inp.is_ultra
                && let Some(bits) = out.exact_fm_ultra[f].interaction_f32_bits()
            {
                z_fm_ultra[f] = f32::from_bits(bits);
            }
        }
    }

    let uint_b = if use_interactions {
        eval_uint_base_4x8(inp.u_int_base, &z_fm_base)
    } else {
        [0.0; 8]
    };
    let mut z_base_8 = [0.0_f32; 8];
    for d in 0..8 {
        z_base_8[d] = z_main_base[d] + uint_b[d];
    }

    if inp.is_ultra {
        let (delta_base, uint_u) = if use_interactions {
            eval_uint_ultra_delta_3x4x8(inp.u_int_ultra, &z_fm_base, &z_fm_ultra)
        } else {
            ([0.0; 8], [0.0; 8])
        };
        for d in 0..8 {
            z_base_8[d] += delta_base[d];
        }
        let mut z_u_8 = [0.0_f32; 8];
        for d in 0..8 {
            z_u_8[d] = z_main_ultra[d] + uint_u[d];
        }
        let mut zh_u = [z_u_8[0], z_u_8[1], z_u_8[2], z_u_8[3]];
        let mut zs_u = [z_u_8[4], z_u_8[5], z_u_8[6], z_u_8[7]];
        trust_clip_outward_4d(&mut zh_u, inp.trust_radius_mh);
        trust_clip_outward_4d(&mut zs_u, inp.trust_radius_mh);
        *out.z_head_ultra = zh_u;
        *out.z_stage_ultra = zs_u;
    } else {
        *out.z_head_ultra = [0.0; 4];
        *out.z_stage_ultra = [0.0; 4];
    }

    let mut zh = [z_base_8[0], z_base_8[1], z_base_8[2], z_base_8[3]];
    let mut zs = [z_base_8[4], z_base_8[5], z_base_8[6], z_base_8[7]];
    trust_clip_outward_4d(&mut zh, inp.trust_radius_mh);
    trust_clip_outward_4d(&mut zs, inp.trust_radius_mh);
    *out.z_head = zh;
    *out.ctrl_basis_4 = zs;
}

/// Evaluates Chebyshev-2 skewed LFO control vector
/// `q2 = [c0 + beta0 * (2*c0^2 - 1), s0, c1 + beta1 * (2*c1^2 - 1), s1]`
/// (`ref[dsp.ctrl.modulators.dynamic]`).
#[inline]
pub(crate) fn eval_skewed_lfo_q2(lfo_phasors: &[[f32; 2]; 2], lfo_skew: &[f32; 2]) -> [f32; 4] {
    let c0 = lfo_phasors[0][0];
    let s0 = lfo_phasors[0][1];
    let c1 = lfo_phasors[1][0];
    let s1 = lfo_phasors[1][1];
    [
        c0 + lfo_skew[0] * (2.0 * c0 * c0 - 1.0),
        s0,
        c1 + lfo_skew[1] * (2.0 * c1 * c1 - 1.0),
        s1,
    ]
}

/// Advances both `Q0.192` LFOs and evaluates skewed `q2` (`ref[dsp.ctrl.modulators.dynamic]`).
#[inline]
pub(crate) fn step_lfos_and_skewed_q2(
    lfo_phases_q192: &mut [Q0_192Phase; 2],
    lfo_dphi_host_bits: &[u32; 2],
    lfo_phasors: &mut [[f32; 2]; 2],
    lfo_skew: &[f32; 2],
) -> [f32; 4] {
    for j in 0..2 {
        let bits = lfo_dphi_host_bits[j] & IEEE754_F32_ABS_MASK;
        if bits != 0 {
            let raw_bits = lfo_dphi_host_bits[j];
            let clamped_bits =
                (raw_bits & IEEE754_F32_SIGN_MASK) | bits.min(MAX_ROTOR_ANGLE_F32_BITS);
            if lfo_phases_q192[j].step_angle_bits(clamped_bits).is_some() {
                let (c, s) = lfo_phases_q192[j].to_phasor();
                lfo_phasors[j] = [c, s];
            } else {
                let theta = f32::from_bits(clamped_bits);
                if let Some((re, im)) =
                    step_lfo_phasor_zero_hold(lfo_phasors[j][0], lfo_phasors[j][1], theta)
                {
                    lfo_phasors[j] = [re, im];
                }
            }
        }
    }
    eval_skewed_lfo_q2(lfo_phasors, lfo_skew)
}

/// Latches 4-band `SensoryQuad` envelopes `env_states[0..4]` into `q3`, flushing non-finite,
/// subnormal, and `-0.0` entries cleanly to `+0.0` (zero fallback envelope IIR).
#[inline]
pub(crate) fn update_env_states(env_states: &mut [f32; 4], sensory_env: Option<[f32; 4]>) {
    let raw = sensory_env.unwrap_or([0.0; 4]);
    for i in 0..4 {
        let v = flush_subnormal_f32_bits(sanitize_finite_f32(raw[i]));
        env_states[i] = if v == 0.0 { 0.0 } else { v };
    }
}

/// Returns `true` when either LFO angular increment is non-zero or any dynamic dispatch column
/// (`W_ctrl` / `W_ctrl_ext` columns `8..16` for `q2` and `q3`) has a non-zero weight.
#[inline]
pub(crate) fn compute_has_internal_modulators(
    lfo_angle_bits: &[u32; 2],
    w_ctrl: &[[f32; 16]; 16],
    w_ctrl_ext: &[[f32; 16]; 16],
    cext_present: bool,
) -> bool {
    let lfo_nonzero = ((lfo_angle_bits[0] & IEEE754_F32_ABS_MASK) != 0)
        || ((lfo_angle_bits[1] & IEEE754_F32_ABS_MASK) != 0);
    let row_has_dyn = |row: &[f32; 16]| {
        row[8..16]
            .iter()
            .any(|v| (v.to_bits() & IEEE754_F32_ABS_MASK) != 0)
    };
    let w_dyn_nonzero = w_ctrl.iter().any(row_has_dyn)
        || (cext_present && w_ctrl_ext[..15].iter().any(row_has_dyn));
    lfo_nonzero || w_dyn_nonzero
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skewed_lfo_sensory_latch_and_activity_flag() {
        let mut phases = [Q0_192Phase::ZERO; 2];
        let mut phasors = [[1.0, 0.0]; 2];
        let q2_zero = step_lfos_and_skewed_q2(&mut phases, &[0, 0], &mut phasors, &[1.0, -1.0]);
        assert_eq!(q2_zero, [2.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            eval_skewed_lfo_q2(&[[0.0, 1.0], [0.0, -1.0]], &[1.0, -1.0]),
            [-1.0, 1.0, 1.0, -1.0]
        );

        let mut env = [0.4_f32; 4];
        update_env_states(&mut env, None);
        assert_eq!(env.map(f32::to_bits), [0; 4]);

        update_env_states(&mut env, Some([0.25, f32::NAN, -0.0, f32::from_bits(1)]));
        assert_eq!(env[0], 0.25);
        assert_eq!(env[1].to_bits(), 0);
        assert_eq!(env[2].to_bits(), 0);
        assert_eq!(env[3].to_bits(), 0);

        let neg_dphi = (-0.25_f32).to_bits();
        let q2_neg =
            step_lfos_and_skewed_q2(&mut phases, &[neg_dphi, 0], &mut phasors, &[0.5, 0.0]);
        assert!(q2_neg[1] < 0.0);
        let expected_c0 = phasors[0][0];
        assert!(
            (q2_neg[0] - (expected_c0 + 0.5 * (2.0 * expected_c0 * expected_c0 - 1.0))).abs()
                < 1.0e-6
        );

        let active_bits = [0.1_f32.to_bits(), (-0.05_f32).to_bits()];
        let q2_act =
            step_lfos_and_skewed_q2(&mut phases, &active_bits, &mut phasors, &[-0.75, 0.75]);
        assert!(phasors[0][0].abs() <= 1.0 && q2_act[1].abs() <= 1.0);
        assert!(q2_act[0].abs() <= 1.75 && q2_act[2].abs() <= 1.75);

        let mut w = [[0.0_f32; 16]; 16];
        let mut w_ext = [[0.0_f32; 16]; 16];
        w[0][0] = 1.0;
        assert!(!compute_has_internal_modulators(
            &[(-0.0_f32).to_bits(), 0],
            &w,
            &w_ext,
            true,
        ));
        assert!(compute_has_internal_modulators(
            &[(-0.05_f32).to_bits(), 0],
            &w,
            &w_ext,
            false,
        ));
        w[3][12] = 0.25;
        assert!(compute_has_internal_modulators(&[0, 0], &w, &w_ext, false));
        w[3][12] = 0.0;
        w_ext[2][14] = -0.1;
        assert!(!compute_has_internal_modulators(&[0, 0], &w, &w_ext, false));
        assert!(compute_has_internal_modulators(&[0, 0], &w, &w_ext, true));

        let descs = [
            ControlClampDescriptor::range(0.0, 1.0, 0.5, 3, [0.0, -0.5, 0.5]),
            ControlClampDescriptor::switch(2, 0b11, 0),
        ];
        let mut slot0 = Cp1ContributorSlot::zero();
        slot0.control_idx = 1;
        slot0.ctrl_kind = 1;
        slot0.aux = 1;
        slot0.lanes = 1;
        slot0.main_base[0][0] = 0.5;
        slot0.v_base[0] = 0.5;
        let mut slot1 = Cp1ContributorSlot::zero();
        slot1.control_idx = 0;
        slot1.ctrl_kind = 2;
        slot1.aux = 0;
        slot1.lanes = 1;
        slot1.emb = [1.0, 0.0, 0.0];
        slot1.v_base[0] = 0.5;
        let mut fm_b = [ExactIntFactorAccumulator::new(2); 4];
        let mut fm_u = [ExactIntFactorAccumulator::new(2); 4];
        let (mut zh, mut cb, mut zhu, mut zsu) = ([0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
        let u_int_b = [[0.25; 8]; 4];
        let u_int_u = [[[0.125; 8]; 4]; 3];
        evaluate_static_cp1_latents(
            StaticEvalInputs {
                descriptors: &descs,
                norm_vals: &[0.5, 1.0],
                active_switches: &[0, 1],
                contributors: &[slot0, slot1],
                distinct_controls: 2,
                is_ultra: true,
                trust_radius_mh: 4.0,
                u_int_base: &u_int_b,
                u_int_ultra: &u_int_u,
            },
            StaticEvalOutputs {
                exact_fm_base: &mut fm_b,
                exact_fm_ultra: &mut fm_u,
                z_head: &mut zh,
                ctrl_basis_4: &mut cb,
                z_head_ultra: &mut zhu,
                z_stage_ultra: &mut zsu,
            },
        );
        assert!(zh[0].is_finite() && zhu[0].is_finite());
    }
}
