//! Reference `f64` `PolyADAA` & `Op-M` Modal DSP Kernels and Empirical Verification Suite
//! (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.bspline-ac]`,
//! `ref[dsp.op-m.recurrence.symplectic]`, `ref[dsp.op-m.rotor.contractive]`).

/// Reference `f64` division-free algebraic `PolyADAA` kernel (`ref[dsp.op-n.polyadaa.midpoint]`):
#[inline]
pub fn poly_adaa_algebraic_f64(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    let m = 0.5 * (u_curr + u_prev);
    let d = 0.5 * (u_curr - u_prev);
    let m2 = m * m;
    let d2 = d * d;
    let odd = beta * m * (1.5 - 0.5 * (m2 + d2));
    let even = delta * ((m2 + d2 / 3.0) - (m2 * m2 + 2.0 * m2 * d2 + 0.2 * d2 * d2));
    odd + even
}

/// Reference `f64` single-sample `PolyADAA` step with $1\times$ passive B-spline weights $(0.875, 0.125)$
/// and $5\text{ Hz}$ AC coupling ($R_{\text{ac}}$, `ref[dsp.op-n.polyadaa.bspline-ac]`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_step_f64(
    x: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: &mut f64,
    a_prev: &mut f64,
    m_dc: &mut f64,
) -> f64 {
    let u_curr = (x * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let a_curr = poly_adaa_algebraic_f64(*u_prev, u_curr, beta, delta);
    let y_nl = 0.875 * a_curr + 0.125 * (*a_prev);
    let m_dc_curr = r_ac * (*m_dc) + (1.0 - r_ac) * y_nl;
    *u_prev = u_curr;
    *a_prev = a_curr;
    *m_dc = m_dc_curr;
    alpha * x + (y_nl - m_dc_curr)
}

/// Processes a contiguous `f64` audio slice through the reference `PolyADAA` kernel.
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_slice_f64(
    input: &[f64],
    output: &mut [f64],
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
) {
    let u_rest = (b_conv * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let u2 = u_rest * u_rest;
    let g_rest = beta * u_rest * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2);
    let mut u_prev = u_rest;
    let mut a_prev = g_rest;
    let mut m_dc = g_rest;

    for (x, y) in input.iter().zip(output.iter_mut()) {
        let u_curr = ((*x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
        let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
        let y_nl = 0.875 * a_curr + 0.125 * a_prev;
        let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
        u_prev = u_curr;
        a_prev = a_curr;
        m_dc = m_dc_curr;
        *y = alpha * (*x) + (y_nl - m_dc_curr);
    }
}

/// Reference `f64` Radial-Polar `Op-M` modal single-sample step (`ref[dsp.op-m.recurrence.symplectic]`, `ref[dsp.op-m.kautz.radial-gate]`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn op_m_modal_step_f64(
    x: f64,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    z_re: &mut f64,
    z_im: &mut f64,
) -> f64 {
    let q = (w_sel * x.abs()).min(1.0);
    let g_sel = q * q * (3.0 - 2.0 * q);
    let r = (r_low + dr_span * (1.0 - g_sel)).clamp(0.25, 0.9995);
    let a_re = r * u_rot_cos;
    let a_im = r * u_rot_sin;
    let b_in = b_gamma_norm * x;
    let next_re = a_re * (*z_re) - a_im * (*z_im) + b_in;
    let next_im = a_im * (*z_re) + a_re * (*z_im);
    *z_re = next_re;
    *z_im = next_im;
    c_real * next_re - c_imag * next_im
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use core::f64::consts::{PI, TAU};

    use crate::FtzDazGuard;
    use crate::math::ScalarBackend;
    use crate::operators::op_m::{KautzModalBank, algebraic_unit_rotor_op_m};
    use crate::operators::reference::eval_saturated_polyadaa_rail_crossing_f64;
    use proptest::prelude::*;

    #[test]
    fn test_poly_adaa_reference() {
        let u_prev = 0.2;
        let u_curr = 0.5;
        let val = poly_adaa_algebraic_f64(u_prev, u_curr, 1.0, 0.5);
        assert!(val.is_finite());

        let mut up = 0.0;
        let mut ap = 0.0;
        let mut dc = 0.0;
        let out = poly_adaa_step_f64(0.3, 0.1, 1.0, 0.2, 0.999, &mut up, &mut ap, &mut dc);
        assert!(out.is_finite());
        assert_ne!(up, 0.0);

        let input = vec![0.1, -0.2, 0.5, -0.8, 0.4];
        let mut output = vec![0.0; 5];
        poly_adaa_slice_f64(&input, &mut output, 0.05, 0.2, 0.8, 0.3, 0.99);
        for &y in &output {
            assert!(y.is_finite());
        }
    }

    #[test]
    fn test_op_m_modal_reference() {
        let mut z_re = 0.0;
        let mut z_im = 0.0;
        let rot_cos = (0.1_f64).cos();
        let rot_sin = (0.1_f64).sin();
        let y = op_m_modal_step_f64(
            0.5, 0.8, 0.15, 0.9, 0.5, 0.7, 0.3, rot_cos, rot_sin, &mut z_re, &mut z_im,
        );
        assert!(y.is_finite());
        assert!(z_re.is_finite());
        assert!(z_im.is_finite());
    }

    /// Test 1A (`verify[dsp.op-n.polyadaa.linear-residual]`, `verify[dsp.op-n.polyadaa.bspline-ac]`):
    /// Verifies linear passband preservation vs. 3-stage naive midpoint droop.
    #[test]
    fn test_1a_linear_passband_preservation_vs_midpoint_droop() {
        const N: usize = 2048;
        const FS: f64 = 48_000.0;
        const F_TEST: f64 = 22_000.0;

        let input: Vec<f64> = (0..N)
            .map(|n| (TAU * F_TEST * (n as f64) / FS).sin())
            .collect();

        let mut naive = input.clone();
        for _stage in 0..3 {
            let mut next = vec![0.0_f64; N];
            let mut prev = 0.0_f64;
            for i in 0..N {
                next[i] = 0.5 * (naive[i] + prev);
                prev = naive[i];
            }
            naive = next;
        }

        let mut dnam = input.clone();
        for _stage in 0..3 {
            let mut next = vec![0.0_f64; N];
            poly_adaa_slice_f64(&dnam, &mut next, 0.0, 1.0, 0.0, 0.0, 0.9993);
            dnam = next;
        }

        let rms = |sig: &[f64]| -> f64 {
            let s: f64 = sig[256..].iter().map(|&v| v * v).sum();
            (s / ((sig.len() - 256) as f64)).sqrt()
        };

        let rms_in = rms(&input);
        let droop_naive_db = 20.0 * (rms(&naive) / rms_in).log10();
        let droop_dnam_db = 20.0 * (rms(&dnam) / rms_in).log10();

        assert!(
            droop_naive_db < -35.0,
            "Expected naive midpoint filter droop < -35 dB, got {droop_naive_db:.2} dB"
        );
        assert!(
            droop_dnam_db.abs() < 1e-12,
            "Expected DNAM linear passband droop == 0.00 dB, got {droop_dnam_db:.6e} dB"
        );
    }

    /// Test 1B (`verify[dsp.op-n.polyadaa.midpoint]`): Verifies exact divided-difference equivalence across 10k pairs.
    #[test]
    fn test_1b_division_free_midpoint_difference_identity_10k_pairs() {
        let beta = 0.85_f64;
        let delta = -0.25_f64;
        let antideriv = |u: f64| -> f64 {
            let u2 = u * u;
            let u3 = u2 * u;
            let u4 = u2 * u2;
            let u5 = u4 * u;
            beta * (0.75 * u2 - 0.125 * u4) + delta * ((1.0 / 3.0) * u3 - 0.2 * u5)
        };

        let mut max_alg_err = 0.0_f64;
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let mut next_u = || -> f64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let unit = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            -0.99 + 1.98 * unit
        };

        for _ in 0..10_000 {
            let u0 = next_u();
            let u1 = next_u();
            let alg = poly_adaa_algebraic_f64(u0, u1, beta, delta);
            if (u1 - u0).abs() > 1e-5 {
                let exact = (antideriv(u1) - antideriv(u0)) / (u1 - u0);
                max_alg_err = max_alg_err.max((alg - exact).abs());
            }
        }
        assert!(
            max_alg_err < 1e-12,
            "Test 1B max algebraic error exceeded 1e-12: {max_alg_err:e}"
        );

        for step in -99..=99 {
            let u = (step as f64) * 0.01;
            let inst = beta * u * (1.5 - 0.5 * u * u) + delta * (u * u - u * u * u * u);
            let alg64 = poly_adaa_algebraic_f64(u, u, beta, delta);
            assert!((alg64 - inst).abs() < 1e-15);

            let (rc_odd, rc_even, crossed) = eval_saturated_polyadaa_rail_crossing_f64(u, u);
            assert!(!crossed);
            assert!((rc_even - (u * u - u * u * u * u)).abs() < 1e-12);
            assert!((rc_odd - u * (1.5 - 0.5 * u * u)).abs() < 1e-12);
        }
    }

    /// Test 1C (`verify[dsp.op-n.polyadaa.midpoint]`, `verify[dsp.op-n.polyadaa.bspline-ac]`):
    /// Verifies aliasing suppression in a 3-stage nonlinear cascade.
    #[test]
    fn test_1c_commuted_cascade_aliasing_suppression_at_3450hz() {
        const N: usize = 1024;
        const K0: usize = 73;
        const TOTAL: usize = 1280;

        let input_full: Vec<f64> = (0..TOTAL)
            .map(|n| 1.6 * (TAU * (K0 as f64) * (n as f64) / (N as f64)).sin())
            .collect();

        let mut base_full = input_full.clone();
        for _stage in 0..3 {
            for x in &mut base_full {
                let u = 1.6 * (*x) + 0.12;
                *x = if u > 1.0 {
                    1.0 + 0.01 * (u - 1.0)
                } else if u < -1.0 {
                    -1.0 + 0.01 * (u + 1.0)
                } else {
                    u
                };
            }
        }

        let mut dnam_full = input_full;
        for _stage in 0..3 {
            let mut next = vec![0.0_f64; TOTAL];
            poly_adaa_slice_f64(&dnam_full, &mut next, 0.05, 0.0, 1.0, 0.12, 0.9993);
            dnam_full = next;
        }

        let compute_alias_dbc = |sig: &[f64]| -> f64 {
            let win_slice = &sig[TOTAL - N..];
            let mut harmonic_bins = [false; N / 2];
            for m in 1..=7 {
                let center = m * K0;
                for b in center.saturating_sub(2)..=(center + 2).min(N / 2 - 1) {
                    harmonic_bins[b] = true;
                }
            }
            let mut fund_e = 0.0_f64;
            let mut alias_e = 0.0_f64;
            for bin in 4..(N / 2) {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (n, &s) in win_slice.iter().enumerate() {
                    let w = 0.5 - 0.5 * (TAU * (n as f64) / (N as f64)).cos();
                    let ang = TAU * (bin as f64) * (n as f64) / (N as f64);
                    re += s * w * ang.cos();
                    im -= s * w * ang.sin();
                }
                let e = re * re + im * im;
                if (bin as isize - K0 as isize).unsigned_abs() <= 2 {
                    fund_e += e;
                } else if !harmonic_bins[bin] {
                    alias_e += e;
                }
            }
            10.0 * (alias_e / fund_e).log10()
        };

        let alias_base_db = compute_alias_dbc(&base_full);
        let alias_dnam_db = compute_alias_dbc(&dnam_full);
        let suppression_db = alias_base_db - alias_dnam_db;
        assert!(
            suppression_db > 15.0,
            "Expected PolyADAA + B-Spline aliasing suppression > 15 dB, got {suppression_db:.2} dB"
        );
    }

    /// Test 2A (`verify[dsp.op-m.recurrence.symplectic]`, `verify[dsp.op-m.kautz.radial-gate]`):
    /// Verifies radial-polar Q preservation vs. Cartesian midpoint collapse.
    #[test]
    fn test_2a_radial_polar_q_preservation_vs_cartesian_collapse() {
        let r0 = 0.992_f64;
        let q_orig = 1.0 / (1.0 - r0);
        let theta_low = 0.05 * PI;
        let theta_high = 0.40 * PI;

        let re_mid = 0.5 * r0 * (theta_low.cos() + theta_high.cos());
        let im_mid = 0.5 * r0 * (theta_low.sin() + theta_high.sin());
        let r_cart = (re_mid * re_mid + im_mid * im_mid).sqrt();
        let q_cart = 1.0 / (1.0 - r_cart);

        let theta_mid = 0.5 * (theta_low + theta_high);
        let (c_rot, s_rot) = algebraic_unit_rotor_op_m(theta_mid as f32);
        let r_polar =
            r0 * ((c_rot as f64) * (c_rot as f64) + (s_rot as f64) * (s_rot as f64)).sqrt();
        let q_polar = 1.0 / (1.0 - r_polar);

        assert!(q_cart < 0.10 * q_orig);
        assert!(q_polar > 0.99 * q_orig);
    }

    /// Test 2B (`verify[dsp.op-m.rotor.contractive]`): Unit-circle rotor accuracy over 20k angles.
    #[test]
    fn test_2b_branchless_newton_unit_circle_rotor_20k_angles() {
        const STEPS: usize = 20_000;
        let mut max_err_f64 = 0.0_f64;
        let mut max_unit_f64 = 0.0_f64;

        for i in 0..=STEPS {
            let theta = PI * (i as f64) / (STEPS as f64);
            let u = (2.0 / PI) * theta - 1.0;
            let alpha = (PI * 0.25) * (1.0 - u * u);
            let a2 = alpha * alpha;
            let c_hat = 1.0 - a2 * (0.5 - a2 * (0.041_666_664_183_139_8 - a2 * 0.001_388_839_7));
            let s_hat = alpha * (1.0 - a2 * (0.166_666_656_732_559_2 - a2 * 0.008_333_025_5));
            let r2_hat = c_hat * c_hat + s_hat * s_hat;
            let y1 = 1.5 - 0.5 * r2_hat;
            let c = c_hat * y1;
            let s = s_hat * y1;
            let mag = (c * c + s * s).sqrt();
            max_err_f64 = max_err_f64.max((1.0 - mag).abs());
            max_unit_f64 = max_unit_f64.max(mag);

            if i > 0 && i < STEPS {
                let (c32, s32) = algebraic_unit_rotor_op_m(theta as f32);
                let mag32 = ((c32 as f64) * (c32 as f64) + (s32 as f64) * (s32 as f64)).sqrt();
                assert!(mag32 <= 1.000_001);
            }
        }

        assert!(max_err_f64 < 2e-8);
        assert!(max_unit_f64 <= 1.0 + 1e-15);
    }

    /// Test 2C (`verify[arch.operators.differentiability]`): Sequential vs. Blelloch scan over L = 1,024.
    #[test]
    fn test_2c_blelloch_parallel_associative_scan_equivalence_1024() {
        const L: usize = 1024;
        type Cplx = (f64, f64);

        let c_mul =
            |(ar, ai): Cplx, (br, bi): Cplx| -> Cplx { (ar * br - ai * bi, ar * bi + ai * br) };
        let c_add = |(ar, ai): Cplx, (br, bi): Cplx| -> Cplx { (ar + br, ai + bi) };

        let mut a = vec![(0.0, 0.0); L];
        let mut b = vec![(0.0, 0.0); L];
        for n in 0..L {
            let r = 0.985 + 0.010 * ((n as f64) * 0.017).sin();
            let theta = 0.12 + 0.04 * ((n as f64) * 0.031).cos();
            a[n] = (r * theta.cos(), r * theta.sin());
            b[n] = (
                0.25 * ((n as f64) * 0.11).sin(),
                0.15 * ((n as f64) * 0.07).cos(),
            );
        }

        let mut seq = vec![(0.0, 0.0); L];
        let mut state = (0.0, 0.0);
        for n in 0..L {
            state = c_add(c_mul(a[n], state), b[n]);
            seq[n] = state;
        }

        let mut scan_a = a;
        let mut scan_b = b;
        let mut stride = 1;
        while stride < L {
            let prev_a = scan_a.clone();
            let prev_b = scan_b.clone();
            for i in stride..L {
                scan_a[i] = c_mul(prev_a[i], prev_a[i - stride]);
                scan_b[i] = c_add(c_mul(prev_a[i], prev_b[i - stride]), prev_b[i]);
            }
            stride <<= 1;
        }

        let mut max_scan_err = 0.0_f64;
        for n in 0..L {
            let dr = seq[n].0 - scan_b[n].0;
            let di = seq[n].1 - scan_b[n].1;
            max_scan_err = max_scan_err.max((dr * dr + di * di).sqrt());
        }
        assert!(max_scan_err < 1e-13);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_kautz_4lane_simd_matches_scalar_step(
            r in 0.1_f32..0.998_f32,
            theta in 0.05_f32..3.05_f32,
            c_re in -1.5_f32..1.5_f32,
            c_im in -1.5_f32..1.5_f32,
            x in -4.0_f32..4.0_f32,
        ) {
            let _audio = FtzDazGuard::new();
            let mut b_scalar = KautzModalBank::<4>::default();
            let mut b_simd = KautzModalBank::<4>::default();
            for m in 0..4 {
                let mf = (m + 1) as f32 * 0.15;
                b_scalar.set_pole(m, (r * (1.0 - 0.02 * mf)).min(0.998), theta * (1.0 - 0.05 * mf), 0.1, c_re, c_im);
                b_simd.set_pole(m, (r * (1.0 - 0.02 * mf)).min(0.998), theta * (1.0 - 0.05 * mf), 0.1, c_re, c_im);
            }
            for k in 0..16 {
                let sample = x * ((k as f32) * 0.3).cos();
                let y_sc = b_scalar.step(sample);
                let y_sm: f32 = b_simd.step_4lane_simd::<ScalarBackend>(sample).iter().sum();
                prop_assert!((y_sc - y_sm).abs() < 1e-6);
            }
        }
    }
}
