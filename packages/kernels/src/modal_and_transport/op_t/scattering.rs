//! Fused `SO(4)` paraunitary Givens butterfly scattering (`R_odd(theta) * diag(A_k) * R_even(theta)`)
//! for `Op-T` partitioned Schroeder diffusers (`ref[dsp.op-t.diffuser.recurrence]`).
//!
//! Product `Q(theta) = R_odd(theta) * R_even(theta)` is a lossless 4×4 orthogonal circulant matrix
//! with first row `[c^2, -c*s, s^2, c*s]` commuting with `H_4(1)`, achieving orthonormal Hadamard
//! scattering at `theta = pi/4` and bit-exact `0-ULP` identity at `(c, s) = (1.0, 0.0)`.

use crate::simd::KernelVec4;

/// Fast bitwise identity guard for `SO(4)` scattering rotors (`(c, s) == (1.0, 0.0)`).
#[inline(always)]
#[must_use]
pub fn is_so4_identity(cos_scat: f32, sin_scat: f32) -> bool {
    cos_scat == 1.0 && sin_scat == 0.0
}

/// Applies the pre-allpass `R_even(theta)` orthogonal Givens block:
/// `x1 = cos_scat * x + sin_scat * [-x1, x0, -x3, x2]`.
#[inline(always)]
#[must_use]
pub fn apply_so4_even_vec4<V: KernelVec4>(x: V, cos_scat: f32, sin_scat: f32) -> V {
    if is_so4_identity(cos_scat, sin_scat) {
        return x;
    }
    x.mul(V::splat(cos_scat))
        .fma(V::splat(sin_scat), x.pair_swap_neg())
}

/// Applies the post-allpass `R_odd(theta)` orthogonal Givens block:
/// `v_read = cos_scat * y + sin_scat * [y3, -y2, y1, -y0]`.
#[inline(always)]
#[must_use]
pub fn apply_so4_odd_vec4<V: KernelVec4>(y: V, cos_scat: f32, sin_scat: f32) -> V {
    if is_so4_identity(cos_scat, sin_scat) {
        return y;
    }
    y.mul(V::splat(cos_scat))
        .fma(V::splat(sin_scat), y.odd_pair_swap_neg())
}

/// Applies `R_even(theta)` across 4 consecutive frames in registers.
#[inline(always)]
#[must_use]
pub fn apply_so4_even_chunk4<V: KernelVec4>(x: [V; 4], cos_scat: f32, sin_scat: f32) -> [V; 4] {
    if is_so4_identity(cos_scat, sin_scat) {
        return x;
    }
    let c = V::splat(cos_scat);
    let s = V::splat(sin_scat);
    [
        x[0].mul(c).fma(s, x[0].pair_swap_neg()),
        x[1].mul(c).fma(s, x[1].pair_swap_neg()),
        x[2].mul(c).fma(s, x[2].pair_swap_neg()),
        x[3].mul(c).fma(s, x[3].pair_swap_neg()),
    ]
}

/// Applies `R_odd(theta)` across 4 consecutive frames in registers.
#[inline(always)]
#[must_use]
pub fn apply_so4_odd_chunk4<V: KernelVec4>(y: [V; 4], cos_scat: f32, sin_scat: f32) -> [V; 4] {
    if is_so4_identity(cos_scat, sin_scat) {
        return y;
    }
    let c = V::splat(cos_scat);
    let s = V::splat(sin_scat);
    [
        y[0].mul(c).fma(s, y[0].odd_pair_swap_neg()),
        y[1].mul(c).fma(s, y[1].odd_pair_swap_neg()),
        y[2].mul(c).fma(s, y[2].odd_pair_swap_neg()),
        y[3].mul(c).fma(s, y[3].odd_pair_swap_neg()),
    ]
}

/// Reverse-time transpose `R_even(-theta)` mapping adjoint `g_x1` back to `g_r_win`.
#[inline(always)]
#[must_use]
pub fn apply_so4_even_transpose_vec4<V: KernelVec4>(g_x1: V, cos_scat: f32, sin_scat: f32) -> V {
    apply_so4_even_vec4(g_x1, cos_scat, -sin_scat)
}

/// Reverse-time transpose `R_odd(-theta)` mapping adjoint `g_out` back to `g_y1`.
#[inline(always)]
#[must_use]
pub fn apply_so4_odd_transpose_vec4<V: KernelVec4>(g_out: V, cos_scat: f32, sin_scat: f32) -> V {
    apply_so4_odd_vec4(g_out, cos_scat, -sin_scat)
}

/// Evaluates reverse-time angular sensitivity `dL/dtheta_eff = g_out^T J_odd(v_read) + g_x1^T J_even(x1)`.
#[inline(always)]
#[must_use]
pub fn so4_theta_grad_vec4<V: KernelVec4>(g_out: V, v_read: V, g_x1: V, x1: V) -> f32 {
    g_out.dot(v_read.odd_pair_swap_neg()) + g_x1.dot(x1.pair_swap_neg())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebraic_unit_rotor_f32;
    use crate::simd::GpuVec4;
    use core::f32::consts::{FRAC_PI_4, PI};
    use proptest::prelude::*;

    fn apply_q_circulant_oracle(x: [f64; 4], theta: f64) -> [f64; 4] {
        let c = theta.cos();
        let s = theta.sin();
        let row = [c * c, -c * s, s * s, c * s];
        core::array::from_fn(|r| (0..4).map(|k| row[(k + 4 - r) & 3] * x[k]).sum())
    }

    fn householder4(x: [f64; 4]) -> [f64; 4] {
        let half_sum = 0.5 * x.iter().sum::<f64>();
        x.map(|v| v - half_sum)
    }

    /// verify[dsp.op-t.diffuser.recurrence]
    #[test]
    fn test_so4_identity_hadamard_and_period_pi() {
        let x_neg_zero = GpuVec4::from_array([-0.0, 1.5, -2.25, -0.0]);
        let y_id = apply_so4_odd_vec4(apply_so4_even_vec4(x_neg_zero, 1.0, 0.0), 1.0, 0.0);
        assert_eq!(
            y_id.to_array().map(f32::to_bits),
            x_neg_zero.to_array().map(f32::to_bits)
        );
        let chunk_id =
            apply_so4_odd_chunk4(apply_so4_even_chunk4([x_neg_zero; 4], 1.0, 0.0), 1.0, 0.0);
        assert_eq!(
            chunk_id[3].to_array().map(f32::to_bits),
            x_neg_zero.to_array().map(f32::to_bits)
        );

        let (c4, s4) = algebraic_unit_rotor_f32(FRAC_PI_4);
        let x_basis = [
            GpuVec4::from_array([1.0, 0.0, 0.0, 0.0]),
            GpuVec4::from_array([0.0, 1.0, 0.0, 0.0]),
            GpuVec4::from_array([0.0, 0.0, 1.0, 0.0]),
            GpuVec4::from_array([0.0, 0.0, 0.0, 1.0]),
        ];
        let cols: [[f32; 4]; 4] =
            x_basis.map(|e| apply_so4_odd_vec4(apply_so4_even_vec4(e, c4, s4), c4, s4).to_array());
        for col in cols {
            for entry in col {
                assert!((entry.abs() - 0.5).abs() < 1.0e-6);
            }
        }

        let (cp, sp) = algebraic_unit_rotor_f32(0.37 + PI);
        let (c0, s0) = algebraic_unit_rotor_f32(0.37);
        let probe = GpuVec4::from_array([0.3, -1.1, 0.7, 2.2]);
        let q0 = apply_so4_odd_vec4(apply_so4_even_vec4(probe, c0, s0), c0, s0).to_array();
        let qp = apply_so4_odd_vec4(apply_so4_even_vec4(probe, cp, sp), cp, sp).to_array();
        for i in 0..4 {
            assert!((q0[i] - qp[i]).abs() < 1.0e-6);
        }
    }

    proptest! {
        #[test]
        fn prop_so4_unitary_circulant_householder_commuting_and_adjoint(
            theta in -PI..PI,
            x in prop::array::uniform4(-4.0_f32..4.0_f32),
            g_out in prop::array::uniform4(-4.0_f32..4.0_f32),
            d in prop::array::uniform4(0.1_f32..2.0_f32),
        ) {
            let (c, s) = algebraic_unit_rotor_f32(theta);
            let xv = GpuVec4::from_array(x);
            let x1 = apply_so4_even_vec4(xv, c, s);
            let y1 = x1.mul(GpuVec4::from_array(d));
            let v_read = apply_so4_odd_vec4(y1, c, s);

            // Unit norm under unitary diag(|a| = 1)
            let qv = apply_so4_odd_vec4(x1, c, s);
            let in_norm_sq: f64 = x.iter().map(|v| f64::from(*v).powi(2)).sum();
            let out_norm_sq: f64 = qv.to_array().iter().map(|v| f64::from(*v).powi(2)).sum();
            prop_assert!((out_norm_sq - in_norm_sq).abs() < 1.0e-4 * in_norm_sq.max(1.0));

            // Exact circulant matrix match & Householder commutation: Q(H_4 x) == H_4(Q x)
            let oracle = apply_q_circulant_oracle(x.map(f64::from), f64::from(theta));
            for i in 0..4 {
                prop_assert!((f64::from(qv.to_array()[i]) - oracle[i]).abs() < 1.0e-5);
            }
            let h_x = householder4(x.map(f64::from));
            let q_h_x = apply_q_circulant_oracle(h_x, f64::from(theta));
            let h_q_x = householder4(oracle);
            for i in 0..4 {
                prop_assert!((q_h_x[i] - h_q_x[i]).abs() < 1.0e-12);
            }

            // 4-frame chunk helper matches 1-frame 0-ULP
            let ch = apply_so4_odd_chunk4(apply_so4_even_chunk4([xv; 4], c, s), c, s);
            for f in 0..4 {
                prop_assert_eq!(ch[f].to_array().map(f32::to_bits), qv.to_array().map(f32::to_bits));
            }

            // Reverse-time adjoint check against f64 central finite differences
            let g_out_v = GpuVec4::from_array(g_out);
            let g_y1 = apply_so4_odd_transpose_vec4(g_out_v, c, s);
            let g_x1 = g_y1.mul(GpuVec4::from_array(d));
            let g_r = apply_so4_even_transpose_vec4(g_x1, c, s);
            let g_theta = so4_theta_grad_vec4(g_out_v, v_read, g_x1, x1);

            let eval_loss_f64 = |th: f64, rx: [f64; 4]| -> f64 {
                let (ct, st) = (th.cos(), th.sin());
                let x1_0 = ct * rx[0] - st * rx[1];
                let x1_1 = ct * rx[1] + st * rx[0];
                let x1_2 = ct * rx[2] - st * rx[3];
                let x1_3 = ct * rx[3] + st * rx[2];
                let y1 = [
                    x1_0 * f64::from(d[0]),
                    x1_1 * f64::from(d[1]),
                    x1_2 * f64::from(d[2]),
                    x1_3 * f64::from(d[3]),
                ];
                let vr = [
                    ct * y1[0] + st * y1[3],
                    ct * y1[1] - st * y1[2],
                    ct * y1[2] + st * y1[1],
                    ct * y1[3] - st * y1[0],
                ];
                (0..4).map(|i| f64::from(g_out[i]) * vr[i]).sum()
            };
            let eps = 1.0e-5_f64;
            let fd_theta = (eval_loss_f64(f64::from(theta) + eps, x.map(f64::from))
                - eval_loss_f64(f64::from(theta) - eps, x.map(f64::from)))
                / (2.0 * eps);
            prop_assert!((f64::from(g_theta) - fd_theta).abs() < 1.0e-4 * fd_theta.abs().max(1.0));
            for i in 0..4 {
                let mut xp = x.map(f64::from);
                let mut xm = xp;
                xp[i] += eps;
                xm[i] -= eps;
                let fd_xi = (eval_loss_f64(f64::from(theta), xp) - eval_loss_f64(f64::from(theta), xm)) / (2.0 * eps);
                prop_assert!((f64::from(g_r.to_array()[i]) - fd_xi).abs() < 1.0e-4 * fd_xi.abs().max(1.0));
            }
        }
    }
}
