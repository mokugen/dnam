#![forbid(unsafe_code)]

//! Cross-module `SONA` kernel scan, `0-ULP` stereo/watermark parity, Zero-DC equilibrium,
//! and exact `i32` Ozaki reconstruction integration suites.
//!
//! verify[repo.code.shared-nostd-kernels]
//! verify[train.pipeline.webgpu-scans-and-ozaki]

use dnam_kernels::{
    ComplexModalScanElem, GpuVec4, HilbertH1Kernel, InterleavedStereo8, KautzQuadState, KernelVec4,
    ParaLruParams, SonaBlockParamsStereo4, SonaBlockStateStereo8, SonaParaLruCoeffs4,
    SonaToneWeights4, StereoVec8, T1SubHarmonicKernel, TptScanElem, TptSvfCoeffs4,
    accum_ozaki_chunk_i32, apply_tpt_init, compose_complex_modal_scan, compose_tpt_scan,
    eval_poly_adaa1_odd_even_vec4, eval_poly_adaa1_stereo8, pow2_ceil_fp32,
    prewarp_tan_pi_f_over_fs_f32, reconstruct_ozaki_dot_f64, run_paralru_newton_scan_tile,
    step_kautz_quad_stereo8, step_kautz_quad_vec4, step_paralru_vec4, step_sona_block_stereo8,
    step_tpt_svf_stereo8, step_tpt_svf_vec4,
};
use proptest::prelude::*;

fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    let ia = a.to_bits() as i32;
    let ib = b.to_bits() as i32;
    let oa = if ia < 0 { i32::MIN - ia } else { ia };
    let ob = if ib < 0 { i32::MIN - ib } else { ib };
    oa.abs_diff(ob)
}

#[test]
fn test_sona_kernels_zero_dc_and_0ulp_stereo_and_watermark_invariance() {
    let g = GpuVec4::from_array([
        prewarp_tan_pi_f_over_fs_f32(0.015),
        prewarp_tan_pi_f_over_fs_f32(0.045),
        prewarp_tan_pi_f_over_fs_f32(0.12),
        prewarp_tan_pi_f_over_fs_f32(0.30),
    ]);
    let k = GpuVec4::from_array([0.4, core::f32::consts::FRAC_1_SQRT_2, 1.0, 1.8]);
    let svf = TptSvfCoeffs4::from_gk(g, k);
    let tone = SonaToneWeights4 {
        lp: GpuVec4::from_array([0.5, 0.25, 0.0, 0.2]),
        bp: GpuVec4::from_array([0.2, 0.5, 0.2, 0.3]),
        hp: GpuVec4::from_array([0.1, 0.1, 0.6, 0.2]),
        ap: GpuVec4::from_array([0.1, 0.05, 0.1, 0.2]),
        dir: GpuVec4::from_array([0.1, 0.1, 0.1, 0.1]),
    };
    let lru = SonaParaLruCoeffs4::from_clamped(
        GpuVec4::from_array([0.99, 0.96, 0.92, 0.85]),
        GpuVec4::from_array([0.2, 0.4, 0.6, 0.8]),
        GpuVec4::from_array([0.1, 0.3, 0.7, 0.95]),
    );
    let params = SonaBlockParamsStereo4 {
        svf,
        tone,
        lru,
        adaa_drive: GpuVec4::from_array([0.5, 1.25, 2.5, 5.0]),
    };

    let mut stereo_state = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();
    for _ in 0..256 {
        let out = step_sona_block_stereo8(InterleavedStereo8::zero(), &mut stereo_state, &params);
        let (ll, lr) = out.lru_state.into_lr();
        let (ol, or) = out.adaa_odd.into_lr();
        let (el, er) = out.adaa_even.into_lr();
        for lane in 0..4 {
            assert_eq!(ll.to_array()[lane].to_bits(), 0);
            assert_eq!(lr.to_array()[lane].to_bits(), 0);
            assert_eq!(ol.to_array()[lane].to_bits(), 0);
            assert_eq!(or.to_array()[lane].to_bits(), 0);
            assert_eq!(el.to_array()[lane].to_bits(), 0);
            assert_eq!(er.to_array()[lane].to_bits(), 0);
        }
    }

    // Separate Vec4 left/right paths vs StereoVec8 across non-trivial excitation must match 0 ULP.
    let mut s1_l = GpuVec4::zero();
    let mut s2_l = GpuVec4::zero();
    let mut s1_r = GpuVec4::zero();
    let mut s2_r = GpuVec4::zero();
    let mut lru_l = GpuVec4::zero();
    let mut lru_r = GpuVec4::zero();
    let mut adaa_prev_l = GpuVec4::zero();
    let mut adaa_prev_r = GpuVec4::zero();
    let mut st8 = SonaBlockStateStereo8::<InterleavedStereo8<GpuVec4>>::zero();

    for t in 0..128 {
        let xl = ((t as f32) * 0.13 - 8.0) * 0.4;
        let xr = -((t as f32) * 0.09 - 5.0) * 0.6;
        let ul = GpuVec4::splat_pair(xl, xl * 0.75);
        let ur = GpuVec4::splat_pair(xr, xr * 1.25);
        let mut ts1 = st8.svf_s1;
        let mut ts2 = st8.svf_s2;

        let svf_l = step_tpt_svf_vec4(ul, &mut s1_l, &mut s2_l, &params.svf);
        let svf_r = step_tpt_svf_vec4(ur, &mut s1_r, &mut s2_r, &params.svf);
        let svf8 = step_tpt_svf_stereo8(
            InterleavedStereo8::from_lr(ul, ur),
            &mut ts1,
            &mut ts2,
            &params.svf,
        );
        assert_eq!(svf8.lp.into_lr().0.to_array(), svf_l.lp.to_array());
        assert_eq!(svf8.lp.into_lr().1.to_array(), svf_r.lp.to_array());
        assert_eq!(svf8.ap.into_lr().0.to_array(), svf_l.ap.to_array());
        let tone_l = svf_l
            .lp
            .mul(params.tone.lp)
            .fma(svf_l.bp, params.tone.bp)
            .fma(svf_l.hp, params.tone.hp)
            .fma(svf_l.ap, params.tone.ap)
            .fma(ul, params.tone.dir);
        let tone_r = svf_r
            .lp
            .mul(params.tone.lp)
            .fma(svf_r.bp, params.tone.bp)
            .fma(svf_r.hp, params.tone.hp)
            .fma(svf_r.ap, params.tone.ap)
            .fma(ur, params.tone.dir);
        let lru_out_l = step_paralru_vec4(
            tone_l,
            &mut lru_l,
            params.lru.alpha_rel,
            params.lru.alpha_atk,
            params.lru.kappa,
        );
        let lru_out_r = step_paralru_vec4(
            tone_r,
            &mut lru_r,
            params.lru.alpha_rel,
            params.lru.alpha_atk,
            params.lru.kappa,
        );
        let rl = lru_out_l.mul(params.adaa_drive);
        let rr = lru_out_r.mul(params.adaa_drive);
        let (odd_l, even_l, prev_l_next) = eval_poly_adaa1_odd_even_vec4(adaa_prev_l, rl);
        let (odd_r, even_r, prev_r_next) = eval_poly_adaa1_odd_even_vec4(adaa_prev_r, rr);
        adaa_prev_l = prev_l_next;
        adaa_prev_r = prev_r_next;

        let out8 = step_sona_block_stereo8(InterleavedStereo8::from_lr(ul, ur), &mut st8, &params);
        let (o8_l, o8_r) = out8.adaa_odd.into_lr();
        let (e8_l, e8_r) = out8.adaa_even.into_lr();
        assert_eq!(o8_l.to_array(), odd_l.to_array());
        assert_eq!(o8_r.to_array(), odd_r.to_array());
        assert_eq!(e8_l.to_array(), even_l.to_array());
        assert_eq!(e8_r.to_array(), even_r.to_array());
    }

    // Kautz Vec4 vs StereoVec8 0-ULP parity + Op-P zero-DC equilibrium.
    let mut kz_v4 = KautzQuadState::new(
        GpuVec4::from_array([0.92, 0.88, 0.80, 0.72]),
        GpuVec4::from_array([0.18, 0.25, 0.35, 0.45]),
        GpuVec4::from_array([0.3, 0.4, 0.5, 0.6]),
        GpuVec4::from_array([1.0, -0.5, 0.25, 0.8]),
        GpuVec4::from_array([0.1, 0.2, -0.3, 0.4]),
        GpuVec4::from_array([-0.5, 0.9, 0.1, -0.2]),
        GpuVec4::from_array([0.2, -0.1, 0.3, -0.4]),
    );
    let mut kz_s8 = kz_v4;
    for t in 0..64 {
        let xl = (t as f32) * 0.02 - 0.5;
        let xr = 0.5 - (t as f32) * 0.03;
        let (yv4_l, yv4_r) = step_kautz_quad_vec4(&mut kz_v4, xl, xr);
        let (ys8_l, ys8_r) =
            step_kautz_quad_stereo8::<InterleavedStereo8<GpuVec4>>(&mut kz_s8, xl, xr);
        assert_eq!(yv4_l.to_bits(), ys8_l.to_bits());
        assert_eq!(yv4_r.to_bits(), ys8_r.to_bits());
    }

    let mut hilbert = HilbertH1Kernel::new();
    let mut tracker = T1SubHarmonicKernel::new();
    for _ in 0..64 {
        let (hi, hq) = hilbert.step(0.0);
        assert_eq!(hi.to_bits(), 0);
        assert_eq!(hq.to_bits(), 0);
        assert_eq!(tracker.step(hi, hq).to_bits(), 0);
    }

    // Check Kautz associative complex modal scan parity on lane 0.
    let mut kz_seq = KautzQuadState::new(
        GpuVec4::splat(0.9),
        GpuVec4::splat(0.2),
        GpuVec4::splat(0.4),
        GpuVec4::splat(1.0),
        GpuVec4::splat(0.5),
        GpuVec4::splat(1.0),
        GpuVec4::splat(0.5),
    );
    let mut scan_acc = ComplexModalScanElem::identity();
    for t in 0..32 {
        let x = ((t as f32) * 0.21).wrap_burst();
        let _ = step_kautz_quad_vec4(&mut kz_seq, x, 0.0);
        scan_acc =
            compose_complex_modal_scan(scan_acc, ComplexModalScanElem::from_step(0.9, 0.2, 0.4, x));
    }
    assert!(ulp_distance_f32(kz_seq.z_re_l.to_array()[0], scan_acc.b_re) <= 2);
    assert!(ulp_distance_f32(kz_seq.z_im_l.to_array()[0], scan_acc.b_im) <= 2);
}

trait BurstHelper {
    fn wrap_burst(self) -> f32;
}

impl BurstHelper for f32 {
    fn wrap_burst(self) -> f32 {
        let k = (self * 0.1) as i32;
        self - (k as f32) * 10.0 - 0.4
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_sona_tpt_svf_and_paralru_scans_match_sequential_and_ozaki_matches_f64(
        u0 in -2.5_f32..2.5_f32,
        u1 in -2.5_f32..2.5_f32,
        freq_ratio in 0.01_f32..0.38_f32,
        q_val in 0.5_f32..8.0_f32,
        alpha_rel in 0.70_f32..0.995_f32,
        alpha_atk in 0.05_f32..0.85_f32,
        kappa in 0.0_f32..1.0_f32,
        sign_flip in prop::bool::ANY,
    ) {
        // 1. TPT-SVF scan vs sequential step (< 2 ULP).
        let g = prewarp_tan_pi_f_over_fs_f32(freq_ratio);
        let k = GpuVec4::splat(q_val).rcp_pos_nr3().to_array()[0];
        let coeffs = TptSvfCoeffs4::from_gk(GpuVec4::splat(g), GpuVec4::splat(k));
        let a1 = coeffs.a1.to_array()[0];
        let a2 = coeffs.a2.to_array()[0];
        let a3 = coeffs.a3.to_array()[0];

        let mut s1 = GpuVec4::zero();
        let mut s2 = GpuVec4::zero();
        let mut tpt_acc = TptScanElem::identity();
        let inputs = [u0, u1, -0.5 * u0, 0.75 * u1, u0 + 0.25 * u1, -u1, 0.3 * u0, 0.0];
        for &u in &inputs {
            let _ = step_tpt_svf_vec4(GpuVec4::splat(u), &mut s1, &mut s2, &coeffs);
            tpt_acc = compose_tpt_scan(tpt_acc, TptScanElem::from_step(a1, a2, a3, u));
        }
        let scan_s = apply_tpt_init(tpt_acc, [0.0, 0.0]);
        prop_assert!((s1.to_array()[0] - scan_s[0]).abs() < 1.0e-5);
        prop_assert!((s2.to_array()[0] - scan_s[1]).abs() < 1.0e-5);

        // 2. K=3 ParaLRU Newton scan converges cleanly to sequential recurrence.
        let mut seq_tile = [0.0_f32; 16];
        let mut lru_state = GpuVec4::zero();
        let mut tile_in = [0.0_f32; 16];
        for (i, slot) in tile_in.iter_mut().enumerate() {
            *slot = inputs[i % inputs.len()];
            let v = step_paralru_vec4(
                GpuVec4::splat(*slot),
                &mut lru_state,
                GpuVec4::splat(alpha_rel),
                GpuVec4::splat(alpha_atk),
                GpuVec4::splat(kappa),
            );
            seq_tile[i] = v.to_array()[0];
        }
        let lru_p = ParaLruParams::clamped(alpha_rel, alpha_atk, kappa);
        let scan_tile = run_paralru_newton_scan_tile::<16, 3>(
            &tile_in,
            0.0,
            lru_p.alpha_rel,
            lru_p.alpha_atk,
            lru_p.kappa,
        );
        for i in 0..16 {
            prop_assert!((seq_tile[i] - scan_tile[i]).abs() < 1.0e-3);
        }

        // 3. Odd-lane sign-flip watermark gauge (s_i in {-1, +1}) preserves 0-ULP invariance in ADAA1.
        let s = if sign_flip { -1.0_f32 } else { 1.0_f32 };
        let up = GpuVec4::splat_pair(s * u0, u0);
        let uc = GpuVec4::splat_pair(s * u1, u1);
        let (odd8, even8) = eval_poly_adaa1_stereo8(
            InterleavedStereo8::from_lr(up, up),
            InterleavedStereo8::from_lr(uc, uc),
        );
        let odd_arr = odd8.into_lr().0.to_array();
        let even_arr = even8.into_lr().0.to_array();
        prop_assert_eq!((s * odd_arr[0] + 0.0).to_bits(), (odd_arr[1] + 0.0).to_bits());
        prop_assert_eq!(even_arr[0].to_bits(), even_arr[1].to_bits());

        // 4. Ozaki i32 3x8-bit chunk dot-product matches f64 oracle cleanly without overflow.
        let mut vx = [0.0_f32; 64];
        let mut vy = [0.0_f32; 64];
        let mut max_x = 0.0_f32;
        let mut max_y = 0.0_f32;
        let mut exact_f64 = 0.0_f64;
        for i in 0..64 {
            vx[i] = inputs[i % inputs.len()] * (1.0 + 0.1 * (i as f32));
            vy[i] = inputs[(i + 3) % inputs.len()] * (0.8 - 0.05 * (i as f32));
            max_x = max_x.max(vx[i].abs());
            max_y = max_y.max(vy[i].abs());
            exact_f64 += f64::from(vx[i]) * f64::from(vy[i]);
        }
        let (sx, inv_sx) = pow2_ceil_fp32(max_x);
        let (sy, inv_sy) = pow2_ceil_fp32(max_y);
        let sums = accum_ozaki_chunk_i32(&vx, &vy, inv_sx, inv_sy);
        let approx_f64 = reconstruct_ozaki_dot_f64(&sums, sx, sy);
        let ozaki_bound = 64.0 * 2.1 * f64::from(sx) * f64::from(sy) * 2.0_f64.powi(-24) + 1.0e-12;
        prop_assert!((approx_f64 - exact_f64).abs() <= ozaki_bound);
    }
}
