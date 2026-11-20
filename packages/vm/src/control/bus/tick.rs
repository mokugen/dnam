//! `Op-N[Ctrl]` exact-integer static evaluation, four independent 4-D trust-region clips,
//! and `Op-M[Ctrl]` `Q0.192` + envelope advancement (`ref[dsp.ctrl.op-n.*]`).

use crate::control::cp1::{eval_uint_base_4x8, eval_uint_ultra_delta_3x4x8};
use crate::control::exact_int::{ExactIntFactorAccumulator, ExactIntMainAccumulator};
use crate::control::{ControlClampDescriptor, Cp1ContributorSlot};
use crate::math::{
    MAX_ROTOR_ANGLE_F32_BITS, Q0_192Phase, flush_subnormal_f32_bits, step_lfo_phasor_zero_hold,
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
            let def_opt = if descriptors[p].default_val.is_finite() {
                let pos = if descriptors[p].default_val > 0.0 {
                    descriptors[p].default_val
                } else {
                    0.0
                };
                ((pos + 0.5) as u8).min(15)
            } else {
                0
            };
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
        let has_emb = slot.emb.iter().any(|w| {
            let abs_w = if *w >= 0.0 { *w } else { -*w };
            abs_w > 1.0e-12
        });
        let z_c = if has_emb {
            ((slot.emb[0] * lob[0]) + (slot.emb[1] * lob[1])) + (slot.emb[2] * lob[2])
        } else {
            scalar
        };

        for l in 0..lanes {
            let lb = lob[l];
            for d in 0..8 {
                let prod_b = slot.main_base[l][d] * lb;
                let _ = main_base[d].add_f32_bits_rt(prod_b.to_bits());
                if inp.is_ultra && slot.has_ultra {
                    let prod_u = slot.main_ultra[l][d] * lb;
                    let _ = main_ultra[d].add_f32_bits_rt(prod_u.to_bits());
                }
            }
        }
        for f in 0..4 {
            let v_b = slot.v_base[f] * z_c;
            let _ = out.exact_fm_base[f].update_contributor_rt(c_idx, v_b.to_bits());
            let v_u = if inp.is_ultra && slot.has_ultra {
                slot.v_ultra[f] * z_c
            } else {
                0.0
            };
            let _ = out.exact_fm_ultra[f].update_contributor_rt(c_idx, v_u.to_bits());
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

/// Advances both `Q0.192` LFOs and refreshes Chebyshev `[T_1, T_2, T_3, T_4]` harmonics
/// + `q2 = [lfo0_mix, lfo1_mix, lfo0_cos, lfo0_sin]` (`ref[dsp.ctrl.modulators.dynamic]`).
#[inline]
pub(crate) fn step_lfos_and_harmonics(
    lfo_phases_q192: &mut [Q0_192Phase; 2],
    lfo_dphi_host_bits: &[u32; 2],
    lfo_phasors: &mut [[f32; 2]; 2],
    lfo_harmonics: &mut [f32; 8],
) -> [f32; 4] {
    for j in 0..2 {
        let bits = lfo_dphi_host_bits[j] & 0x7FFF_FFFF;
        if bits != 0 {
            let raw_bits = lfo_dphi_host_bits[j];
            let clamped_bits = (raw_bits & 0x8000_0000) | bits.min(MAX_ROTOR_ANGLE_F32_BITS);
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
        let c = lfo_phasors[j][0];
        let t1 = c;
        let t2 = 2.0 * c * c - 1.0;
        let t3 = 2.0 * c * t2 - t1;
        let t4 = 2.0 * c * t3 - t2;
        lfo_harmonics[j * 4] = t1;
        lfo_harmonics[j * 4 + 1] = t2;
        lfo_harmonics[j * 4 + 2] = t3;
        lfo_harmonics[j * 4 + 3] = t4;
    }
    [
        lfo_harmonics[0] + 0.35 * lfo_harmonics[1],
        lfo_harmonics[4] + 0.35 * lfo_harmonics[5],
        lfo_phasors[0][0],
        lfo_phasors[0][1],
    ]
}

/// Updates 4-band `env_states[0..4]` from router detectors or standalone exponential ballistics.
#[inline]
pub(crate) fn update_env_states(
    env_states: &mut [f32; 4],
    env_poles_host: &[f32; 3],
    router_envelopes: Option<[f32; 4]>,
    audio_abs_energy: f32,
) {
    if let Some(ext) = router_envelopes {
        for i in 0..4 {
            let v = if ext[i].is_finite() { ext[i] } else { 0.0 };
            env_states[i] = flush_subnormal_f32_bits(v);
        }
        return;
    }
    let a = if audio_abs_energy.is_finite() && audio_abs_energy >= 0.0 {
        audio_abs_energy
    } else {
        0.0
    };
    for b in 0..3 {
        let p = env_poles_host[b];
        env_states[b] = flush_subnormal_f32_bits(p * env_states[b] + (1.0 - p) * a);
    }
    let diff = env_states[2] - env_states[0];
    env_states[3] = flush_subnormal_f32_bits(if diff > 0.0 { diff } else { 0.0 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfo_zero_hold_and_standalone_envelopes() {
        let mut phases = [Q0_192Phase::ZERO; 2];
        let mut phasors = [[1.0, 0.0]; 2];
        let mut harms = [0.0; 8];
        let q2 = step_lfos_and_harmonics(&mut phases, &[0, 0], &mut phasors, &mut harms);
        assert_eq!(q2[2], 1.0);
        assert_eq!(q2[3], 0.0);

        let mut env = [0.0_f32; 4];
        update_env_states(&mut env, &[0.5, 0.75, 0.9], None, 1.0);
        assert_eq!(env[0], 0.5);
        assert_eq!(env[1], 0.25);
        assert!((env[2] - 0.1).abs() < 1.0e-6);
        assert_eq!(env[3], 0.0);

        let neg_dphi = (-0.25_f32).to_bits();
        let q2_neg = step_lfos_and_harmonics(&mut phases, &[neg_dphi, 0], &mut phasors, &mut harms);
        assert!(q2_neg[3] < 0.0);

        update_env_states(
            &mut env,
            &[0.5, 0.75, 0.9],
            Some([0.25, f32::NAN, 0.5, -0.5]),
            0.0,
        );
        assert_eq!(env, [0.25, 0.0, 0.5, -0.5]);

        let active_bits = [0.1_f32.to_bits(), (-0.05_f32).to_bits()];
        let q2_act = step_lfos_and_harmonics(&mut phases, &active_bits, &mut phasors, &mut harms);
        assert!(q2_act[2].abs() <= 1.0 && q2_act[3].abs() <= 1.0);

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
