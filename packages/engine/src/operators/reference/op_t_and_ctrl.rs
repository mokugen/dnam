//! Reference `f64` `Op-T` Transport, Tank Mixing & Control-Plane Mathematical Kernels
//! (`ref[dsp.op-t.reader.catmull-rom]`, `ref[dsp.tank.feedback-loop]`,
//! `ref[dsp.ctrl.basis.lobatto]`, `ref[dsp.ctrl.op-n.trust-clip]`).

/// Reference `f64` 4-tap Catmull-Rom cubic delay interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
#[inline]
pub fn catmull_rom_hermite_f64(s_m1: f64, s_0: f64, s_1: f64, s_2: f64, mu: f64) -> f64 {
    let c3 = -0.5 * s_m1 + 1.5 * s_0 - 1.5 * s_1 + 0.5 * s_2;
    let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
    let c1 = -0.5 * s_m1 + 0.5 * s_1;
    let c0 = s_0;
    ((c3 * mu + c2) * mu + c1) * mu + c0
}

/// Reference `f64` 4-tap Cubic Lagrange fractional delay weights (`ref[dsp.op-t.reader.catmull-rom]`):
#[inline]
pub fn cubic_lagrange_weights_f64(mu: f64) -> [f64; 4] {
    let mu_p1 = mu + 1.0;
    let mu_m1 = mu - 1.0;
    let mu_m2 = mu - 2.0;
    let prod_m1_m2 = mu_m1 * mu_m2;
    let prod_mu_p1 = mu * mu_p1;
    [
        (-1.0 / 6.0) * mu * prod_m1_m2,
        0.5 * mu_p1 * prod_m1_m2,
        -0.5 * prod_mu_p1 * mu_m2,
        (1.0 / 6.0) * prod_mu_p1 * mu_m1,
    ]
}

/// Reference `f64` 8-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.tank.feedback-loop]`).
pub fn givens_householder_mix_8ch_f64(x: &[f64; 8], cos_psi: f64, sin_psi: f64) -> [f64; 8] {
    let mut g_rot = [0.0_f64; 8];
    for p in 0..4 {
        let i0 = p * 2;
        let i1 = i0 + 1;
        g_rot[i0] = cos_psi * x[i0] - sin_psi * x[i1];
        g_rot[i1] = sin_psi * x[i0] + cos_psi * x[i1];
    }
    let mut sum = 0.0_f64;
    for &v in &g_rot {
        sum += v;
    }
    let sub = 0.25 * sum;
    let mut out = [0.0_f64; 8];
    for i in 0..8 {
        out[i] = g_rot[i] - sub;
    }
    out
}

/// Reference `f64` 16-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.tank.feedback-loop]`).
pub fn givens_householder_mix_16ch_f64(
    x: &[f64; 16],
    cos_psi: f64,
    sin_psi: f64,
    cross_cos: &[f64; 8],
    cross_sin: &[f64; 8],
) -> [f64; 16] {
    let mut g_rot = [0.0_f64; 16];
    for p in 0..8 {
        let i0 = p * 2;
        let i1 = i0 + 1;
        g_rot[i0] = cos_psi * x[i0] - sin_psi * x[i1];
        g_rot[i1] = sin_psi * x[i0] + cos_psi * x[i1];
    }
    for m in 0..8 {
        let a = g_rot[m];
        let b = g_rot[m + 8];
        let c = cross_cos[m];
        let s = cross_sin[m];
        g_rot[m] = c * a - s * b;
        g_rot[m + 8] = s * a + c * b;
    }
    let sum: f64 = g_rot.iter().sum();
    let sub = 0.125 * sum;
    let mut out = [0.0_f64; 16];
    for i in 0..16 {
        out[i] = g_rot[i] - sub;
    }
    out
}

/// Reference `f64` single-sample Op-T Coercive-Saturating Lyapunov loop step with exact gain-normalized
/// $5\text{ Hz}$ DC blocker (`ref[dsp.tank.dc-blocker]`, `ref[dsp.tank.loop-saturator]`, `ref[dsp.tank.quiet-feedback-knee]`).
#[inline]
pub fn op_t_coercive_step_f64(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: &mut f64,
    dc_y_prev: &mut f64,
) -> f64 {
    let alpha_dc = 0.5 * (1.0 + r_dc);
    let y_dc = alpha_dc * (u_in - *dc_x_prev) + r_dc * (*dc_y_prev);
    *dc_x_prev = u_in;
    *dc_y_prev = y_dc;

    let ratio = y_dc.abs() / sigma_knee;
    let g_eff = g_lin / (1.0 + ratio * ratio);
    let s_pre = g_eff * y_dc;
    let ratio_ceil = (s_pre.abs() / v_ceil).min(1.0);
    s_pre * (1.0 - (1.0 / 3.0) * ratio_ceil * ratio_ceil)
}

/// Processes a contiguous `f64` audio slice through a single Op-T coercive tank loop channel.
pub fn op_t_tank_slice_f64(
    input: &[f64],
    output: &mut [f64],
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) {
    let mut dc_x_prev = 0.0_f64;
    let mut dc_y_prev = 0.0_f64;
    for (x, y) in input.iter().zip(output.iter_mut()) {
        *y = op_t_coercive_step_f64(
            *x,
            g_lin,
            sigma_knee,
            v_ceil,
            r_dc,
            &mut dc_x_prev,
            &mut dc_y_prev,
        );
    }
}

/// Evaluates the 1D Node-Anchored Newton/Lobatto basis polynomials $\psi_k(u)$ (`ref[dsp.ctrl.basis.lobatto]`).
#[inline]
pub fn eval_newton_lobatto_basis_f64(u: f64, nodes: &[f64; 4], degree: usize) -> [f64; 4] {
    let mut psi = [0.0; 4];
    if degree >= 1 {
        psi[0] = u - nodes[0];
    }
    if degree >= 2 {
        psi[1] = psi[0] * (u - nodes[1]);
    }
    if degree >= 3 {
        psi[2] = psi[1] * (u - nodes[2]);
    }
    psi
}

/// Evaluates the Rank-4 Factorization Machine (FM) interaction term.
pub fn eval_rank4_fm_interaction_f64(v_matrix: &[[f64; 4]], u_norm: &[f64]) -> [f64; 4] {
    let p = u_norm.len().min(v_matrix.len());
    let mut sum_v = [0.0f64; 4];
    let mut sum_v2 = [0.0f64; 4];

    for i in 0..p {
        let ui = u_norm[i];
        let ui2 = ui * ui;
        for f in 0..4 {
            let vif = v_matrix[i][f];
            sum_v[f] += vif * ui;
            sum_v2[f] += vif * vif * ui2;
        }
    }

    let mut out = [0.0f64; 4];
    for f in 0..4 {
        out[f] = 0.5 * (sum_v[f] * sum_v[f] - sum_v2[f]);
    }
    out
}

/// Evaluates the 4-harmonic Chebyshev polynomial LFO waveshaper.
#[inline]
pub fn eval_chebyshev_lfo_4harm_f64(cos_phi: f64) -> [f64; 4] {
    let x = cos_phi.clamp(-1.0, 1.0);
    let x2 = x * x;
    let t1 = x;
    let t2 = 2.0 * x2 - 1.0;
    let t3 = 4.0 * x2 * x - 3.0 * x;
    let t4 = 8.0 * x2 * x2 - 8.0 * x2 + 1.0;
    [t1, t2, t3, t4]
}

/// Evaluates the Ellipsoidal Trust-Region Norm Soft-Clipping (`ref[dsp.ctrl.op-n.trust-clip]`, `0 FDIV`, `0 FSQRT`).
#[inline]
pub fn eval_trust_region_softclip_f64(z: &mut [f64; 4], trust_radius_mh: f64) {
    let norm_sq = z[0] * z[0] + z[1] * z[1] + z[2] * z[2] + z[3] * z[3];
    let mh_sq = trust_radius_mh * trust_radius_mh;
    if norm_sq > mh_sq && norm_sq > 0.0 {
        let inv_norm = crate::math::newton6_reciprocal_f64(crate::math::sqrt_f64(norm_sq));
        let scale = trust_radius_mh * inv_norm;
        z[0] *= scale;
        z[1] *= scale;
        z[2] *= scale;
        z[3] *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use core::f64::consts::TAU;

    use crate::FtzDazGuard;
    use crate::math::{MathBackend, ScalarBackend, SimdVec4};
    use crate::operators::op_m::SymplecticModal4Lane;
    use crate::operators::op_n::{OpNBackend, c2_wavefolder};
    use crate::operators::op_t::{OP_T_LITE_RING_CAPACITY, OpTDelayLine};
    use proptest::prelude::*;

    #[test]
    fn test_interpolations_and_weights_reference() {
        let s_m1 = 1.0;
        let s_0 = 2.0;
        let s_1 = 3.0;
        let s_2 = 4.0;
        assert!((catmull_rom_hermite_f64(s_m1, s_0, s_1, s_2, 0.0) - s_0).abs() < 1e-12);
        assert!((catmull_rom_hermite_f64(s_m1, s_0, s_1, s_2, 1.0) - s_1).abs() < 1e-12);

        let w = cubic_lagrange_weights_f64(0.4);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_givens_householder_orthogonal_reference() {
        let psi: f64 = 0.3;
        let c = psi.cos();
        let s = psi.sin();
        let x8 = [1.0, 2.0, -1.0, 0.5, 0.2, -0.4, 0.8, -0.6];
        let y8 = givens_householder_mix_8ch_f64(&x8, c, s);
        let norm_in: f64 = x8.iter().map(|v| v * v).sum();
        let norm_out: f64 = y8.iter().map(|v| v * v).sum();
        assert!((norm_in - norm_out).abs() < 1e-10);

        let x16 = [
            1.0, -0.5, 0.2, 0.8, -1.2, 0.4, -0.1, 0.9, 0.3, -0.7, 0.6, -0.2, 0.5, -0.4, 0.1, -0.8,
        ];
        let cross_c = [0.8; 8];
        let cross_s = [0.6; 8];
        let y16 = givens_householder_mix_16ch_f64(&x16, c, s, &cross_c, &cross_s);
        let norm_in16: f64 = x16.iter().map(|v| v * v).sum();
        let norm_out16: f64 = y16.iter().map(|v| v * v).sum();
        assert!((norm_in16 - norm_out16).abs() < 1e-10);
    }

    #[test]
    fn test_op_t_coercive_reference() {
        let mut dc_x = 0.0;
        let mut dc_y = 0.0;
        let y = op_t_coercive_step_f64(0.5, 0.9, 1.0, 2.0, 0.999, &mut dc_x, &mut dc_y);
        assert!(y.is_finite());
        assert_eq!(dc_x, 0.5);

        let input = vec![0.2, -0.4, 0.8, -0.1];
        let mut output = vec![0.0; 4];
        op_t_tank_slice_f64(&input, &mut output, 0.95, 1.2, 1.5, 0.998);
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_control_math_reference() {
        let nodes = [0.0, 0.25, 0.75, 1.0];
        let psi0 = eval_newton_lobatto_basis_f64(0.5, &nodes, 0);
        assert_eq!(psi0, [0.0; 4]);
        let psi1 = eval_newton_lobatto_basis_f64(0.5, &nodes, 1);
        assert_eq!(psi1[0], 0.5);
        let psi2 = eval_newton_lobatto_basis_f64(0.5, &nodes, 2);
        assert_eq!(psi2[1], 0.5 * 0.25);
        let psi3 = eval_newton_lobatto_basis_f64(0.5, &nodes, 3);
        assert_eq!(psi3[2], 0.5 * 0.25 * (-0.25));

        let v_mat = [[1.0, 0.5, 0.2, 0.1], [0.3, 0.4, 0.5, 0.6]];
        let u_norm = [0.8, 0.6];
        let fm = eval_rank4_fm_interaction_f64(&v_mat, &u_norm);
        for &v in &fm {
            assert!(v.is_finite());
        }

        let cheb = eval_chebyshev_lfo_4harm_f64(0.5);
        assert!((cheb[0] - 0.5).abs() < 1e-12);
        assert!((cheb[1] - (2.0 * 0.25 - 1.0)).abs() < 1e-12);

        let mut z = [1.0, 1.0, 1.0, 1.0];
        eval_trust_region_softclip_f64(&mut z, 1.0);
        let norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);

        let mut z_in = [0.1, 0.1, 0.1, 0.1];
        eval_trust_region_softclip_f64(&mut z_in, 1.0);
        assert_eq!(z_in, [0.1, 0.1, 0.1, 0.1]);
    }

    /// Test 3 (`verify[dsp.op-t.reader.catmull-rom]`): Catmull-Rom vs. recursive Thiran boundary jerk.
    #[test]
    fn test_3_stateless_catmull_rom_vs_recursive_thiran_boundary_jerk() {
        const FS: f64 = 48_000.0;
        const F_SIG: f64 = 5_000.0;
        const N: usize = 4096;
        const BUF_LEN: usize = 512;

        let mut ring = vec![0.0_f64; BUF_LEN];
        let mut wr = 0_usize;
        let mut err_catmull = Vec::with_capacity(N);
        let mut err_thiran = Vec::with_capacity(N);
        let mut thiran_state = 0.0_f64;

        for n in 0..N {
            let t = (n as f64) / FS;
            let x = (TAU * F_SIG * t).sin();
            ring[wr] = x;

            let tau = 120.0 + 60.0 * (TAU * 3.0 * t).sin();
            let d_int = tau.floor() as usize;
            let mu = (tau - (d_int as f64)).clamp(0.05, 0.95);
            let mu_exact = tau - (d_int as f64);

            let p0 = (wr + BUF_LEN - d_int) & (BUF_LEN - 1);
            let pm1 = (p0 + 1) & (BUF_LEN - 1);
            let p1 = (p0 + BUF_LEN - 1) & (BUF_LEN - 1);
            let p2 = (p0 + BUF_LEN - 2) & (BUF_LEN - 1);

            let y_cat = catmull_rom_hermite_f64(ring[pm1], ring[p0], ring[p1], ring[p2], mu_exact);
            let a_th = (1.0 - mu) / (1.0 + mu);
            let y_th = a_th * (ring[p0] - thiran_state) + ring[p1];
            thiran_state = y_th;

            if n >= 256 {
                let exact = (TAU * F_SIG * (t - tau / FS)).sin();
                err_catmull.push(y_cat - exact);
                err_thiran.push(y_th - exact);
            }
            wr = (wr + 1) & (BUF_LEN - 1);
        }

        let max_jerk = |e: &[f64]| -> f64 {
            e.windows(3)
                .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
                .fold(0.0_f64, f64::max)
        };

        assert!(max_jerk(&err_thiran) > 15.0 * max_jerk(&err_catmull));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_givens_householder_8ch_isometry_and_coercive_bound(
            psi in 0.0f64..core::f64::consts::FRAC_PI_2,
            x8 in prop::array::uniform8(-5.0f64..5.0f64),
            u_in in -20.0f64..20.0f64,
            g_lin in 0.5f64..1.25f64,
            v_ceil in 0.5f64..2.0f64,
        ) {
            let (c, s) = (psi.cos(), psi.sin());
            let y8 = givens_householder_mix_8ch_f64(&x8, c, s);
            let norm_x: f64 = x8.iter().map(|v| v * v).sum();
            let norm_y: f64 = y8.iter().map(|v| v * v).sum();
            prop_assert!((norm_x - norm_y).abs() < 1e-10);

            let mut dc_x = 0.0;
            let mut dc_y = 0.0;
            let y_coercive = op_t_coercive_step_f64(u_in, g_lin, 0.5 * v_ceil, v_ceil, 0.995, &mut dc_x, &mut dc_y);
            prop_assert!(y_coercive.abs() <= v_ceil + 1e-9);
        }

        #[test]
        fn prop_operators_bounded_and_finite(
            x in -10.0_f32..10.0_f32,
            g_base in 0.1_f32..5.0_f32,
            bias in -1.0_f32..1.0_f32,
            delay in 4.0_f32..500.0_f32,
        ) {
            let _audio = FtzDazGuard::new();
            let mut modal = SymplecticModal4Lane::default();
            modal.set_pole(0, 0.96, 0.3);
            let (re, im) = modal.step_lane(0, x);
            prop_assert!(re.is_finite() && im.is_finite());

            let drive_in = g_base * re + bias;
            let t4 = ScalarBackend::tanh_pade5_4ch(<ScalarBackend as MathBackend>::Vec4::from_array([
                drive_in,
                -drive_in,
                0.25 * drive_in,
                0.0,
            ]))
            .to_array();
            prop_assert!(t4.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
            let (wf_odd, wf_even) = c2_wavefolder(re);
            prop_assert!(wf_odd.is_finite() && wf_even.is_finite());

            let mut dl = OpTDelayLine::<OP_T_LITE_RING_CAPACITY>::default();
            dl.set_target_delay(delay, false);
            let y_dl = dl.step(t4[0]);
            prop_assert!(y_dl.is_finite());
        }
    }
}
