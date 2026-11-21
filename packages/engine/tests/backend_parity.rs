//! Cross-Backend Parity & 4-Topology Verification Suite (`tests/backend_parity.rs`).
//!
//! Verifies that `ScalarBackend`, `DefaultMathBackend`, architecture-specific SIMD backends
//! (`NeonBackend`, `ArmSve2Backend`, `ArmSmeBackend`, `X86Sse2Backend`, `X86Sse42Backend`,
//! `X86Avx2FmaBackend`), and `DynamicDnamEngine` produce matching audio across `Lite..Ultra`
//! tiers and all 4 I/O topologies through Layer 1 `DnamEngine` wrapping `dnam-vm`.

#![forbid(unsafe_code)]

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use arena_builder::build_sona_extended_container;
use dnam_engine::engine::EngineBackend;
use dnam_engine::{DnamEngine, DynamicDnamEngine, FtzDazGuard, Tier};
use dnam_kernels::{StereoVec8, f32_to_fp16, fp16_to_f32, step_sona_block_stereo8};
use dnam_vm::math::{BackendKind, DefaultMathBackend, MathBackend, ScalarBackend, SimdVec4};

/// verify[arch.simd.two-level-traits]
/// verify[arch.simd.single-library-dispatch]
/// verify[dsp.op-m.recurrence.stereo-simd]
/// verify[dsp.op-m.tracker.stereo-batching]
/// verify[dsp.op-n.polyadaa.stereo-and-rail-guard]
/// verify[dsp.op-n.core.stereo-simd]
/// verify[dsp.op-t.reader.contiguous-fp16x4]
/// verify[dsp.cabinet.mirrored-fir-simd]
/// verify[dsp.tank.simd-loop]
/// verify[dsp.head.precollapse-and-zerocopy]
/// verify[dsp.ctrl.op-n.static-and-interior-fastpath]
/// verify[dsp.ctrl.dispatch.sparse-and-active-ramps]
#[test]
fn test_cross_backend_parity_and_4_io_topologies() {
    let _ftz = FtzDazGuard::new();
    let container = build_sona_extended_container(false, false);
    let in_l: Vec<f32> = (0..512).map(|n| 0.4 * ((n as f32) * 0.09).sin()).collect();
    let in_r: Vec<f32> = (0..512).map(|n| 0.4 * ((n as f32) * 0.13).cos()).collect();

    for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let mut eng_scalar =
            DnamEngine::<ScalarBackend>::from_bytes(&container, tier, 48_000).unwrap();
        let mut eng_default =
            DnamEngine::<DefaultMathBackend>::from_bytes(&container, tier, 48_000).unwrap();

        eng_scalar.set_control_value(0, 7.5);
        eng_default.set_control_value(0, 7.5);
        eng_scalar.set_control_immediate(1, 0.0);
        eng_default.set_control_immediate(1, 0.0);

        let (mut out_sl, mut out_sr) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
        let (mut out_dl, mut out_dr) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);

        eng_scalar.process_block(&in_l, &in_r, &mut out_sl, &mut out_sr);
        eng_default.process_block(&in_l, &in_r, &mut out_dl, &mut out_dr);

        let max_err_l = out_sl
            .iter()
            .zip(out_dl.iter())
            .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        let max_err_r = out_sr
            .iter()
            .zip(out_dr.iter())
            .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
        assert!(
            max_err_l < 1e-4 && max_err_r < 1e-4 && out_sl.iter().any(|&v| v.abs() > 1e-5),
            "Backend parity mismatch on {tier:?}: max_err_l={max_err_l:e}, max_err_r={max_err_r:e}"
        );

        #[cfg(target_arch = "aarch64")]
        {
            for (mut out_arch_l, mut out_arch_r, label) in [
                {
                    let mut eng = DnamEngine::<dnam_vm::math::aarch64::NeonBackend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    eng.set_control_value(0, 7.5);
                    eng.set_control_immediate(1, 0.0);
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r, "NEON")
                },
                {
                    let mut eng = DnamEngine::<dnam_vm::math::aarch64::ArmSve2Backend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    eng.set_control_value(0, 7.5);
                    eng.set_control_immediate(1, 0.0);
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r, "SVE2")
                },
                {
                    let mut eng = DnamEngine::<dnam_vm::math::aarch64::ArmSmeBackend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    eng.set_control_value(0, 7.5);
                    eng.set_control_immediate(1, 0.0);
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r, "SME")
                },
            ] {
                let err_l = out_sl
                    .iter()
                    .zip(out_arch_l.drain(..))
                    .fold(0.0_f32, |acc, (&a, b)| acc.max((a - b).abs()));
                let err_r = out_sr
                    .iter()
                    .zip(out_arch_r.drain(..))
                    .fold(0.0_f32, |acc, (&a, b)| acc.max((a - b).abs()));
                assert!(
                    err_l < 1e-4 && err_r < 1e-4,
                    "ARM {label} parity mismatch on {tier:?}: err_l={err_l:e}, err_r={err_r:e}"
                );
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            for (mut eng_x86_l, mut eng_x86_r) in [
                {
                    let mut eng = DnamEngine::<dnam_vm::math::x86::X86Sse2Backend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    eng.set_control_value(0, 7.5);
                    eng.set_control_immediate(1, 0.0);
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
                {
                    let mut eng = DnamEngine::<dnam_vm::math::x86::X86Sse42Backend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    eng.set_control_value(0, 7.5);
                    eng.set_control_immediate(1, 0.0);
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
                {
                    let mut eng = DnamEngine::<dnam_vm::math::x86::X86Avx2FmaBackend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    eng.set_control_value(0, 7.5);
                    eng.set_control_immediate(1, 0.0);
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
            ] {
                let err_l = out_sl
                    .iter()
                    .zip(eng_x86_l.drain(..))
                    .fold(0.0_f32, |acc, (&a, b)| acc.max((a - b).abs()));
                let err_r = out_sr
                    .iter()
                    .zip(eng_x86_r.drain(..))
                    .fold(0.0_f32, |acc, (&a, b)| acc.max((a - b).abs()));
                assert!(err_l < 1e-4 && err_r < 1e-4);
            }
        }

        // Verify Topology 2 (in-place) matches Topology 1 (separate buffers)
        eng_default.reset_to_dc_equilibrium();
        let mut ip_l = in_l.clone();
        let mut ip_r = in_r.clone();
        eng_default.process_block_in_place(&mut ip_l, &mut ip_r);
        assert_eq!(ip_l, out_dl);
        assert_eq!(ip_r, out_dr);

        // Verify Topology 3 (mono_to_stereo), Topology 4 (mono), and Stereo-to-Mono mid reconstruction
        eng_default.reset_to_dc_equilibrium();
        let (mut m2s_l, mut m2s_r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
        eng_default.process_mono_to_stereo(&in_l, &mut m2s_l, &mut m2s_r);

        eng_default.reset_to_dc_equilibrium();
        let mut m_out = vec![0.0_f32; 512];
        eng_default.process_mono(&in_l, &mut m_out);
        for (i, (&m, (&ml, &mr))) in m_out.iter().zip(m2s_l.iter().zip(m2s_r.iter())).enumerate() {
            let expected_mid = 0.5 * (ml + mr);
            assert!(
                (m - expected_mid).abs() < 1e-6,
                "Mono mid mismatch on {tier:?} at {i}: m={m}, expected={expected_mid}"
            );
        }

        eng_default.reset_to_dc_equilibrium();
        let mut s2m_out = vec![0.0_f32; 512];
        eng_default.process_stereo_to_mono(&in_l, &in_r, &mut s2m_out);
        for (i, (&s, (&dl, &dr))) in s2m_out
            .iter()
            .zip(out_dl.iter().zip(out_dr.iter()))
            .enumerate()
        {
            let expected_mid = 0.5 * (dl + dr);
            assert!(
                (s - expected_mid).abs() < 1e-6,
                "Stereo-to-mono mid mismatch on {tier:?} at {i}: s={s}, expected={expected_mid}"
            );
        }
    }

    // Verify DynamicDnamEngine runtime dispatch parity against scalar backend
    let mut dyn_scalar = DynamicDnamEngine::from_bytes_with_backend(
        &container,
        Tier::Full,
        48_000,
        BackendKind::Scalar,
    )
    .unwrap();
    let mut dyn_default = DynamicDnamEngine::from_bytes(&container, Tier::Full, 48_000).unwrap();
    let (mut d_sl, mut d_sr) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    let (mut d_dl, mut d_dr) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    dyn_scalar.process_block(&in_l[..128], &in_r[..128], &mut d_sl, &mut d_sr);
    dyn_default.process_block(&in_l[..128], &in_r[..128], &mut d_dl, &mut d_dr);

    let dyn_err = d_sl
        .iter()
        .zip(d_dl.iter())
        .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
    assert!(dyn_err < 1e-4);
}

fn exercise_math_backend<B: EngineBackend>() {
    let _ = B::NAME;
    let _ = B::KIND.name();

    let v0 = B::Vec4::zero();
    assert_eq!(v0.to_array(), [0.0; 4]);
    let va = B::Vec4::load(&[1.0, -2.0, 3.0, -4.0]);
    let mut stored = [0.0_f32; 4];
    va.store(&mut stored);
    assert_eq!(stored, [1.0, -2.0, 3.0, -4.0]);
    let vb = B::Vec4::splat(2.0);
    let vp = B::Vec4::splat_pair(1.5, -1.5);
    assert_eq!(vp.to_array(), [1.5, -1.5, 1.5, -1.5]);
    assert_eq!(va.pair_swap_neg().to_array(), [2.0, 1.0, 4.0, 3.0]);
    assert_eq!(
        B::Vec4::from_array([f32::from_bits(1), -f32::from_bits(1), 1.0, -1.0])
            .flush_subnormal()
            .to_array(),
        [0.0, -0.0, 1.0, -1.0]
    );
    let _ = va.add(vb);
    let _ = va.sub(vb);
    let _ = va.mul(vb);
    let _ = va.fma(vb, vb);
    let _ = va
        .fma_lane::<0>(vb, vp)
        .fma_lane::<1>(vb, vp)
        .fma_lane::<2>(vb, vp)
        .fma_lane::<3>(vb, vp);
    let _ = va.fms(vb, vb);
    let _ = va.max(vb);
    let _ = va.min(vb);
    let _ = va.clamp(B::Vec4::splat(-1.0), B::Vec4::splat(1.0));
    let _ = va.abs();
    let hs_a = va.horizontal_sum();
    let hs_b = vb.horizontal_sum();
    let (hsp_a, hsp_b) = va.horizontal_sum_pair(vb);
    assert!((hs_a - hsp_a).abs() < 1e-6 && (hs_b - hsp_b).abs() < 1e-6);
    let _ = va.dot(vb);

    let mat = [[0.1_f32; 4]; 4];
    let v_mat = B::fma_mat4x4_vec4(B::Vec4::zero(), &mat, &[1.0, 2.0, 3.0, 4.0]);
    assert!(v_mat.horizontal_sum().is_finite());

    let xl_arr = [0.25_f32, -0.5, 0.75, 1.0];
    let xr_arr = [-0.4_f32, 0.8, -0.2, 0.6];
    let (st_l, st_r) =
        B::fma_mat4x4_stereo(va, vb, &mat, B::Vec4::load(&xl_arr), B::Vec4::load(&xr_arr));
    let ref_l = B::fma_mat4x4_vec4(va, &mat, &xl_arr).to_array();
    let ref_r = B::fma_mat4x4_vec4(vb, &mat, &xr_arr).to_array();
    for i in 0..4 {
        assert!((st_l.to_array()[i] - ref_l[i]).abs() < 1.0e-6);
        assert!((st_r.to_array()[i] - ref_r[i]).abs() < 1.0e-6);
    }

    let seed_inputs = [0.125_f32, 1.0, 3.75, 12.5];
    let sv_seed_in = <ScalarBackend as MathBackend>::Vec4::load(&seed_inputs);
    let bv_seed_in = B::Vec4::load(&seed_inputs);
    assert_eq!(
        bv_seed_in.rcp_pos_seed_bits().to_array().map(f32::to_bits),
        sv_seed_in.rcp_pos_seed_bits().to_array().map(f32::to_bits)
    );
    assert_eq!(
        bv_seed_in
            .rsqrt_pos_seed_bits()
            .to_array()
            .map(f32::to_bits),
        sv_seed_in
            .rsqrt_pos_seed_bits()
            .to_array()
            .map(f32::to_bits)
    );
    let r2 = bv_seed_in.rcp_pos_nr2().to_array();
    let r3 = bv_seed_in.rcp_pos_nr3().to_array();
    let s2 = bv_seed_in.rsqrt_pos_nr2().to_array();
    let s3 = bv_seed_in.rsqrt_pos_nr3().to_array();
    for i in 0..4 {
        let d = seed_inputs[i];
        assert!((r2[i] * d - 1.0).abs() < 1.0e-5);
        assert!((r3[i] * d - 1.0).abs() < 1.0e-6);
        let ref_rsq = 1.0 / d.sqrt();
        assert!(((s2[i] - ref_rsq) / ref_rsq).abs() < 2.5e-3);
        assert!(((s3[i] - ref_rsq) / ref_rsq).abs() < 2.0e-5);
    }

    let (d16_l, d16_r) = B::dot16_stereo(&mat, &mat, &mat, &mat);
    assert!((d16_l - 0.16).abs() < 1e-6 && (d16_r - 0.16).abs() < 1e-6);

    let fp16_4 = B::f32x4_to_fp16x4(&[0.5, -0.5, 1.0, 0.0]);
    let f32_4 = B::fp16x4_to_f32x4(&fp16_4);
    assert!((f32_4[0] - 0.5).abs() < 1e-3);
    assert_eq!(f32_to_fp16(fp16_to_f32(fp16_4[0])), fp16_4[0]);

    let e_pos = B::Vec4::splat(0.5).exp2_clamped5().to_array()[0];
    let e_neg = B::Vec4::splat(-0.5).exp2_clamped5().to_array()[0];
    assert!((e_pos * e_neg - 1.0).abs() < 1e-6);
    assert_eq!(B::crc32c(b"123456789"), 0xE306_9283);
    let (cp, sp) = B::fast_unit_phasor(0.7);
    assert!((cp * cp + sp * sp - 1.0).abs() < 1e-4);

    let kv = B::Stereo8::zero();
    let (kl, kr) = kv.into_lr();
    let (sl, sr) = <ScalarBackend as MathBackend>::Stereo8::zero().into_lr();
    assert_eq!(kl.to_array(), sl.to_array());
    assert_eq!(kr.to_array(), sr.to_array());
    let kv_sum = B::Stereo8::splat_lr(0.25, -0.25).horizontal_sum_lr();
    assert_eq!(kv_sum, (1.0, -1.0));
    let _ = step_sona_block_stereo8::<B::Stereo8> as fn(_, _, _) -> _;
}

#[test]
fn test_all_math_backends_and_backend_kind_names() {
    for (kind, expected) in [
        (BackendKind::Scalar, "Scalar"),
        (BackendKind::ArmNeon, "ARM64 NEON"),
        (BackendKind::Neon, "ARM64 NEON"),
        (BackendKind::ArmSve2, "ARMv9 SVE2"),
        (BackendKind::ArmSme, "ARM SME"),
        (BackendKind::X86Sse2, "x86 SSE2"),
        (BackendKind::X86Sse42, "x86 SSE4.2"),
        (BackendKind::X86Avx2Fma, "x86 AVX2"),
        (BackendKind::X86Simd, "x86 AVX2"),
        (BackendKind::WasmSimd128, "WASM SIMD"),
    ] {
        assert_eq!(kind.name(), expected);
    }
    let _ = dnam_vm::math::detect_best_backend();

    exercise_math_backend::<ScalarBackend>();
    exercise_math_backend::<DefaultMathBackend>();
    #[cfg(target_arch = "aarch64")]
    {
        exercise_math_backend::<dnam_vm::math::aarch64::NeonBackend>();
        exercise_math_backend::<dnam_vm::math::aarch64::ArmSve2Backend>();
        exercise_math_backend::<dnam_vm::math::aarch64::ArmSmeBackend>();
    }
    #[cfg(target_arch = "x86_64")]
    {
        exercise_math_backend::<dnam_vm::math::x86::X86Sse2Backend>();
        exercise_math_backend::<dnam_vm::math::x86::X86Sse42Backend>();
        exercise_math_backend::<dnam_vm::math::x86::X86Avx2FmaBackend>();
    }
}
