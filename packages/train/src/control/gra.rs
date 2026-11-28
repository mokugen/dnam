//! Differentiable `ControlBus v4 (GRA)` (`ref[dsp.ctrl.gra.rational-and-entmax]`,
//! `ref[dsp.ctrl.gra.bibo-convex-hull]`, `ref[dsp.ctrl.dispatch.micro-ramp]`):
//! - Unrotated `CP1` centered basis (`SO(4)`-isolated `z_head[0..4]`)
//! - Zero-anchored `SwiGLU` + smooth clamped `Padé-[1/1]` (`1 + d_r >= 0.25`)
//! - Anchored `1.5-Entmax` prototypes (`m = 0` `StarCenter` anchor, `k* == 1` `0-ULP` fast-exit)
//! - Bilinear dynamic/inertia gate (`q2` Knob x LFO asymmetric slew, `q3` Padé env, `slot 31 L/R`)
//! - `O(P + M)` global convex-hull BIBO verifier + `4`-frame-aligned micro-ramp trajectory.

pub mod adjoint;
pub mod bibo;
pub mod bridge;
pub mod entmax;
pub mod forward;
pub mod model;
pub mod ramp;

pub use adjoint::{GraOutputsGrad, GraParamGrads, pullback_gra};
pub use bibo::{GraBiboReport, audit_gra_bibo_hull};
pub use bridge::GraCoreOpTBridge;
pub use entmax::{
    Entmax15Output, INV_K_F32, INV_K_F64, SORT_NET_8, SORT_NET_16, entmax15_forward_f32_mirror,
    entmax15_forward_f64, entmax15_vjp_scores,
};
pub use forward::{
    GraCache, GraOutputs, GraOutputsF32, eval_gra_f32_mirror, eval_gra_forward, sigma_poly_f32,
    sigma_poly_f64, trust_clip_4d_f64,
};
pub use model::{
    CONTROL_BUS_MAX_TOTAL_BYTES, GRA_MAX_D, GRA_MAX_H, GRA_MAX_M, GRA_MAX_R_PADE, GRA_MAX_XI,
    GRA_NUM_EDGES, GRA_NUM_OP_T_MODS, GRA_OUT_DELTA_DIM, GRA_RECLAIMED_TOTAL_BYTES, GRA_Z_HEAD_DIM,
    GRA_Z_HEAD_UNROTATED_DIM, GRA_Z_STAGE_DIM, GraDynGateWeights, GraParams, GraTierProfile,
    GraUnrotatedBasis, K_CTRL_44K1_48K, MICRO_RAMP_FRAMES, PADE_DEN_MAX, PADE_DEN_MIN,
    TRUST_RAD_DEFAULT, TRUST_RAD_MAX, TRUST_RAD_MIN, canonicalize_zero_f32, canonicalize_zero_f64,
    gra_wire_bytes,
};
pub use ramp::{
    GRA_RAMP_VEC_LEN, GraRampGrads, GraRampTickInput, GraRampTrajectory,
    alpha_chunk_for_sample_rate, eval_micro_ramp_trajectory, is_unramped_kautz_or_psi_slot,
    k_ctrl_aligned_4, k_ctrl_for_sample_rate, pullback_micro_ramp_trajectory,
};

/// Evaluates static (`q2 = q3 = 0`) `GRA` forward and optionally computes exact reverse-mode
/// parameter gradients wrt `d_out_opt`.
#[must_use]
pub fn eval_gra_static_and_grad(
    params: &GraParams,
    basis: &GraUnrotatedBasis,
    d_out_opt: Option<&GraOutputsGrad>,
) -> (GraOutputs, Option<GraParamGrads>) {
    let xi = &basis.xi[..basis.len.min(params.xi_dim)];
    let (out, cache) = eval_gra_forward(params, xi, [0.0; 4], [0.0; 4], 0.0);
    let grads = d_out_opt.map(|d_out| pullback_gra(params, &cache, d_out));
    (out, grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::stability::StageTargetBounds;
    use dnam_vm::control::ControlClampDescriptor;

    #[test]
    fn test_gra_end_to_end_static_and_grad_with_bibo_and_ramp() {
        let desc = ControlClampDescriptor::range(0.0, 10.0, 5.0, 2, [0.0, -0.5, 0.5]);
        let basis_star = GraUnrotatedBasis::from_descriptors(&[desc], &[0.0], &[0]);
        let mut params = GraParams::zeros(GraTierProfile::Pro, basis_star.len);
        params.w_in[0][0] = 0.9;
        params.w_in[1][1] = -0.6;
        params.w_g[0][0] = 0.5;
        params.w_v[0][0] = 0.4;
        params.w_o[0][0] = 0.3;
        params.w_den[0][0] = 0.4;
        params.w_num[1][0] = 1.6;
        params.w_num[2][1] = 1.1;
        params.v_proto[1][13] = -0.75;
        params.v_proto[1][31] = 0.35;
        params.v_proto[1][32] = -0.35;
        params.v_proto[2][18] = 24.0;
        params.enforce_structural_invariants();

        // 1. StarCenter yields exact 0-ULP output and zero gradients on v_proto[0].
        let (star_out, _) = eval_gra_static_and_grad(&params, &basis_star, None);
        assert!(star_out.to_flat().iter().all(|&v| v == 0.0));
        let star32 = eval_gra_f32_mirror(&params, &[0.0; 2], [0.0; 4], [0.0; 4], 0.0);
        assert!(star32.to_bits_all().iter().all(|&b| b == 0));

        // 2. O(P+M) BIBO audit passes canonical v4 bounds.
        let bounds = StageTargetBounds::canonical_v4(true, 5.0);
        let bibo = audit_gra_bibo_hull(&params, &bounds, [1.0; 4], [1.0; 4]);
        assert!(bibo.is_bibo_safe);

        // 3. Off-center static evaluation + gradient + 4-frame micro-ramp integration.
        let basis_off = GraUnrotatedBasis::from_descriptors(&[desc], &[0.85], &[0]);
        let mut d_out = GraOutputsGrad::default();
        d_out.d_stage_targets[13] = 1.0;
        d_out.d_stage_targets[18] = 0.25;
        d_out.d_slot31_lr = [0.5, -0.5];
        let (off_out, grads) = eval_gra_static_and_grad(&params, &basis_off, Some(&d_out));
        let g = grads.expect("grads returned");
        assert!(g.d_v_proto[0].iter().all(|&v| v == 0.0));
        assert!(off_out.stage_targets[13].abs() > 0.0 || off_out.stage_targets[18].abs() > 0.0);

        let mut target = [0.0_f64; GRA_RAMP_VEC_LEN];
        target[0..32].copy_from_slice(&off_out.stage_targets);
        target[32] = off_out.slot31_lr[1];
        let traj = eval_micro_ramp_trajectory(
            &[GraRampTickInput {
                target,
                lfo_rate_target: 0.0,
                is_dirty: true,
                has_internal_modulators: false,
                is_analog_drift_active: false,
            }],
            &[0.0; GRA_RAMP_VEC_LEN],
            0.0,
            44_100.0,
            0.5,
            0.25,
        );
        assert_eq!(traj.chunk_states.len(), 4);
        assert!((traj.chunk_states[3][13] - off_out.stage_targets[13]).abs() < 1.0e-12);
    }
}
