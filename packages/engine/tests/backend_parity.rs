//! Cross-Backend Parity & 4-Topology Verification Suite (`tests/backend_parity.rs`).
//!
//! - `verify[arch.simd.two-level-traits]`
//! - `verify[arch.simd.single-library-dispatch]`
//! - `verify[dsp.op-m.recurrence.stereo-simd]`
//! - `verify[dsp.op-m.tracker.stereo-batching]`
//! - `verify[dsp.op-n.polyadaa.stereo-and-rail-guard]`
//! - `verify[dsp.op-n.core.stereo-simd]`
//! - `verify[dsp.op-t.reader.contiguous-fp16x4]`
//! - `verify[dsp.cabinet.mirrored-fir-simd]`
//! - `verify[dsp.tank.simd-loop]`
//! - `verify[dsp.head.precollapse-and-zerocopy]`
//! - `verify[dsp.ctrl.op-n.static-and-interior-fastpath]`
//! - `verify[dsp.ctrl.dispatch.sparse-and-active-ramps]`
//!
//! Verifies that `ScalarBackend`, `NeonBackend`, `ArmSmeBackend`, and `DynamicDnamEngine`
//! produce matching audio across `Lite`, `Full`, and `Ultra` tiers and all 4 I/O topologies.

use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container};
use dnam_engine::engine::EngineBackend;
use dnam_engine::graph::sona_core::{
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_SONA_TILES, MAX_TOTAL_SONA_LANES, SONA_FIR_MAX_TAPS,
    SonaMatryoshkaModel, SonaTierReadout,
};
use dnam_engine::graph::weights::pack_sona_matryoshka_to_arena;
use dnam_engine::math::{
    BackendKind, DefaultMathBackend, MathBackend, ScalarBackend, SimdF64Vec2, SimdVec4,
    TrainerMathBackend,
};
use dnam_engine::schema::DnamJsonHeader;
use dnam_engine::{DnamEngine, DynamicDnamEngine, Tier};

fn build_parity_container() -> Vec<u8> {
    let mut header = DnamJsonHeader::default_reference();
    header.metadata.name = "Backend Parity Test Rig".to_string();

    let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
    let mut model = SonaMatryoshkaModel::dual_rate_default(false);
    model.stage0.xo_low_gain = 1.05;
    model.stage0.xo_high_gain = 0.95;
    model.stage0.clean_sub_mix = 0.25;
    model.crossbar.op_p_crossbar_gain = 0.15;
    model.crossbar.op_t_crossbar_gain = 0.10;
    for g in 0..MAX_SONA_GROUPS {
        model.crossbar.op_p_to_group[g][0] = 0.08;
        let grp = &mut model.groups[g];
        for l in 0..4 {
            grp.svf_g[l] = 0.20 + 0.04 * (l as f32);
            grp.svf_k[l] = core::f32::consts::FRAC_1_SQRT_2;
            grp.lru_alpha_rel[l] = 0.90;
            grp.in_x_weight[l] = if l == 0 {
                0.75
            } else {
                0.10 * if ((g + l) & 1) == 0 { 1.0 } else { -1.0 }
            };
            grp.adaa_drive[l] = 0.65;
            grp.wavefolder_mix[l] = 0.08;
            grp.sag_depth[l] = 0.20;
            grp.lru_kappa[l] = 0.30;
            grp.lru_alpha_atk[l] = 0.05;
            grp.tone_lp[l] = 0.20;
            grp.tone_dir[l] = 0.70;
            grp.rect_sign[l] = 1.0;
        }
    }
    for tile in &mut model.tiles[..MAX_SONA_TILES] {
        for col in 0..4 {
            for row in 0..4 {
                tile.w_odd_4x4[col][row] = if col == row { 0.12 } else { 0.02 };
                tile.w_lru_4x4[col][row] = if col == row { 0.04 } else { 0.0 };
            }
        }
    }
    for m in 0..MAX_KAUTZ_POLES {
        model.kautz_poles[m] = (0.90, 0.15 + 0.08 * ((m & 3) as f32));
    }
    for tier in 0..4 {
        let mut ro = SonaTierReadout::default();
        for ch in 0..2 {
            for l in 0..MAX_TOTAL_SONA_LANES {
                ro.pre_cab_w_odd[ch][l] = 0.04 * (((l + ch + tier) % 5) as f32 - 2.0);
                ro.direct_w_odd[ch][l] = 0.025;
            }
        }
        for tap in 0..SONA_FIR_MAX_TAPS {
            ro.fir_l[tap] = if tap % 16 == 0 { 0.06 } else { 0.004 };
            ro.fir_r[tap] = if tap % 16 == 0 { 0.05 } else { 0.004 };
        }
        for m in 0..MAX_KAUTZ_POLES {
            ro.kautz_b_gamma[m] = 0.35;
            ro.kautz_c_l[m] = (0.15, -0.08);
            ro.kautz_c_r[m] = (0.12, -0.06);
        }
        model.readouts[tier] = ro;
        model.stage4[tier].g_loop = 0.72;
        model.stage4[tier].wet_mix = 0.2;
    }
    pack_sona_matryoshka_to_arena(&model, &mut arena);

    build_dnam_container(&mut header, &arena, &arena, false).unwrap()
}

#[test]
fn test_cross_backend_parity_and_4_io_topologies() {
    let container = build_parity_container();
    let in_l: Vec<f32> = (0..512).map(|n| 0.4 * ((n as f32) * 0.09).sin()).collect();
    let in_r: Vec<f32> = (0..512).map(|n| 0.4 * ((n as f32) * 0.13).cos()).collect();

    for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let mut eng_scalar =
            DnamEngine::<ScalarBackend>::from_bytes(&container, tier, 48_000).unwrap();
        let mut eng_default =
            DnamEngine::<DefaultMathBackend>::from_bytes(&container, tier, 48_000).unwrap();

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
            max_err_l < 1e-5 && max_err_r < 1e-5,
            "Backend parity mismatch on {tier:?}: max_err_l={max_err_l:e}, max_err_r={max_err_r:e}"
        );

        #[cfg(target_arch = "aarch64")]
        {
            let mut eng_sve2 = DnamEngine::<dnam_engine::math::ArmSve2Backend>::from_bytes(
                &container, tier, 48_000,
            )
            .unwrap();
            let (mut out_sve2_l, mut out_sve2_r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
            eng_sve2.process_block(&in_l, &in_r, &mut out_sve2_l, &mut out_sve2_r);
            let sve2_err_l = out_sl
                .iter()
                .zip(out_sve2_l.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            let sve2_err_r = out_sr
                .iter()
                .zip(out_sve2_r.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            assert!(
                sve2_err_l < 1e-5 && sve2_err_r < 1e-5,
                "ARM SVE2 parity mismatch on {tier:?}: sve2_err_l={sve2_err_l:e}, sve2_err_r={sve2_err_r:e}"
            );

            let mut eng_sme = DnamEngine::<dnam_engine::math::ArmSmeBackend>::from_bytes(
                &container, tier, 48_000,
            )
            .unwrap();
            let (mut out_sme_l, mut out_sme_r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
            eng_sme.process_block(&in_l, &in_r, &mut out_sme_l, &mut out_sme_r);
            let sme_err_l = out_sl
                .iter()
                .zip(out_sme_l.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            let sme_err_r = out_sr
                .iter()
                .zip(out_sme_r.iter())
                .fold(0.0_f32, |acc, (&a, &b)| acc.max((a - b).abs()));
            assert!(
                sme_err_l < 1e-5 && sme_err_r < 1e-5,
                "ARM SME parity mismatch on {tier:?}: sme_err_l={sme_err_l:e}, sme_err_r={sme_err_r:e}"
            );
        }

        #[cfg(target_arch = "x86_64")]
        {
            for (mut eng_x86_l, mut eng_x86_r) in [
                {
                    let mut eng = DnamEngine::<dnam_engine::math::X86Sse2Backend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
                {
                    let mut eng = DnamEngine::<dnam_engine::math::X86Sse42Backend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
                    let (mut l, mut r) = (vec![0.0_f32; 512], vec![0.0_f32; 512]);
                    eng.process_block(&in_l, &in_r, &mut l, &mut r);
                    (l, r)
                },
                {
                    let mut eng = DnamEngine::<dnam_engine::math::X86Avx2FmaBackend>::from_bytes(
                        &container, tier, 48_000,
                    )
                    .unwrap();
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
                assert!(err_l < 1e-5 && err_r < 1e-5);
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

    // Verify DynamicDnamEngine across available backends
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
    assert!(dyn_err < 1e-5);
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
    let (d16_vl, d16_vr) = dnam_engine::graph::dot16_stereo_vec4::<B::Vec4>(&mat, &mat, &mat, &mat);
    assert!((d16_vl - 0.16).abs() < 1e-6 && (d16_vr - 0.16).abs() < 1e-6);

    let _ = <B as dnam_engine::operators::op_n::OpNBackend<B::Vec4>>::tanh_pade5_4ch(va);
    let _ = va.rcp_ge1_nr2();
    let _ = B::Vec4::rsqrt_pos_pair_nr2(B::Vec4::splat(1.5), B::Vec4::splat(2.5));
    let _ = B::cubic_lagrange_weights_4ch(B::Vec4::splat(0.25));

    let fp16_4 = B::f32x4_to_fp16x4(&[0.5, -0.5, 1.0, 0.0]);
    let f32_4 = B::fp16x4_to_f32x4(&fp16_4);
    assert!((f32_4[0] - 0.5).abs() < 1e-3);

    let r0 = B::rsqrt_nr2_f32(4.0);
    let (rp0, rp1) = B::rsqrt_nr2_pair_f32(4.0, 0.25);
    assert!((r0 - 0.5).abs() < 1e-3 && (rp0 - 0.5).abs() < 1e-3 && (rp1 - 2.0).abs() < 1e-2);
    let (sd_l, su_l, sd_r, su_r) = B::tracker_exp_scales_f32(0.5, 2.0);
    assert_eq!((sd_l * su_l, sd_r * su_r), (1.0, 1.0));

    assert_eq!(B::crc32c(b"123456789"), 0xE306_9283);
    assert_eq!(
        dnam_engine::math::crc32c::crc32c_software_slice(0, b"123456789"),
        0xE306_9283
    );
    let (cp, sp) = B::fast_unit_phasor(0.7);
    assert!((cp * cp + sp * sp - 1.0).abs() < 1e-4);
}

fn exercise_trainer_math_backend<B: TrainerMathBackend>() {
    assert_eq!(B::trainer_backend_kind(), B::KIND);
    let va = B::F64Vec2::from_f32_pair(1.5, -2.5);
    let vb = B::F64Vec2::splat_pair(0.5, 4.0);
    assert_eq!(
        va.mul_add_ieee(vb, B::F64Vec2::splat(2.0)).to_array(),
        [2.5, 5.5]
    );
    assert!((va.dot(vb) - (-9.25)).abs() <= 1e-12);

    let a: Vec<f64> = (0..19)
        .map(|i| 0.3 * ((i as f64) * 0.17).sin() - 0.1)
        .collect();
    let b: Vec<f64> = (0..19)
        .map(|i| -0.25 * ((i as f64) * 0.23).cos() + 0.08)
        .collect();
    let c: Vec<f64> = (0..19)
        .map(|i| 0.4 * ((i as f64) * 0.11).sin() + 0.04)
        .collect();
    assert!((ScalarBackend::dot_f64(&a, &b) - B::dot_f64(&a, &b)).abs() <= 1e-11);
    let (sp0, sp1) = ScalarBackend::dot_pair_f64(&a, &b, &c);
    let (bp0, bp1) = B::dot_pair_f64(&a, &b, &c);
    assert!((sp0 - bp0).abs() <= 1e-11 && (sp1 - bp1).abs() <= 1e-11);
    assert!((ScalarBackend::sum_sq_f64(&a) - B::sum_sq_f64(&a)).abs() <= 1e-11);

    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let (mut ds, mut db) = (vec![0.0; 19], vec![0.0; 19]);
    ScalarBackend::sub_f32_to_f64(&mut ds, &a32, &b32);
    B::sub_f32_to_f64(&mut db, &a32, &b32);
    assert!(ds.iter().zip(&db).all(|(x, y)| x.to_bits() == y.to_bits()));

    let fi0: Vec<[f64; 4]> = (0..11).map(|k| [k as f64 * 0.1, -0.3, 0.7, 0.2]).collect();
    let fi1: Vec<[f64; 4]> = (0..11)
        .map(|k| [-0.4, k as f64 * 0.05, 0.11, -0.9])
        .collect();
    let wfj: Vec<[f64; 4]> = (0..11)
        .map(|k| [0.5, -0.25, 0.125 + k as f64 * 0.02, -0.75])
        .collect();
    let (t4s, t4b) = (
        ScalarBackend::contract_tile_4x4_ieee_f64(&fi0, &wfj),
        B::contract_tile_4x4_ieee_f64(&fi0, &wfj),
    );
    let ((t8as, t8bs), (t8ab, t8bb)) = (
        ScalarBackend::contract_tile_8x4_ieee_f64(&fi0, &fi1, &wfj),
        B::contract_tile_8x4_ieee_f64(&fi0, &fi1, &wfj),
    );
    for lj in 0..4 {
        for li in 0..4 {
            assert_eq!(t4s[lj][li].to_bits(), t4b[lj][li].to_bits());
            assert_eq!(t8as[lj][li].to_bits(), t8ab[lj][li].to_bits());
            assert_eq!(t8bs[lj][li].to_bits(), t8bb[lj][li].to_bits());
        }
    }

    let mut fb = [0.0_f32; 256];
    for (idx, v) in fb.iter_mut().enumerate() {
        *v = ((idx as f32) * 0.19).sin() * 0.35;
    }
    let (mut u1s, mut u1b) = ([[0.0_f64; 256]; 7], [[0.0_f64; 256]; 7]);
    let (mut u2s, mut u2b) = ([[0.0_f64; 256]; 7], [[0.0_f64; 256]; 7]);
    ScalarBackend::advance_bezout7_lp_dim_f64(&fb, 65, 0.96, 0.04, &mut u1s);
    B::advance_bezout7_lp_dim_f64(&fb, 65, 0.96, 0.04, &mut u1b);
    ScalarBackend::advance_bezout7_lp_dim_f64(&fb, 65, 0.72, 0.28, &mut u2s);
    B::advance_bezout7_lp_dim_f64(&fb, 65, 0.72, 0.28, &mut u2b);
    for st in 0..7 {
        assert!(
            u1s[st][..65]
                .iter()
                .zip(&u1b[st][..65])
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        assert!(
            u2s[st][..65]
                .iter()
                .zip(&u2b[st][..65])
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }
    let (mut lfs, mut mfs, mut hfs) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
    let (mut lfb, mut mfb, mut hfb) = ([0.0_f32; 256], [0.0_f32; 256], [0.0_f32; 256]);
    ScalarBackend::readout_bezout7_bands_dim_f64(&fb, 65, &u1s, &u2s, &mut lfs, &mut mfs, &mut hfs);
    B::readout_bezout7_bands_dim_f64(&fb, 65, &u1b, &u2b, &mut lfb, &mut mfb, &mut hfb);
    for d in 0..65 {
        assert_eq!(lfs[d].to_bits(), lfb[d].to_bits());
        assert_eq!(mfs[d].to_bits(), mfb[d].to_bits());
        assert_eq!(hfs[d].to_bits(), hfb[d].to_bits());
    }
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
    let _ = dnam_engine::math::detect_best_backend();

    exercise_math_backend::<ScalarBackend>();
    exercise_math_backend::<DefaultMathBackend>();
    exercise_trainer_math_backend::<ScalarBackend>();
    exercise_trainer_math_backend::<DefaultMathBackend>();
    #[cfg(target_arch = "aarch64")]
    {
        exercise_math_backend::<dnam_engine::math::ArmSve2Backend>();
        exercise_math_backend::<dnam_engine::math::ArmSmeBackend>();
        exercise_trainer_math_backend::<dnam_engine::math::NeonBackend>();
        exercise_trainer_math_backend::<dnam_engine::math::ArmSve2Backend>();
        exercise_trainer_math_backend::<dnam_engine::math::ArmSmeBackend>();
    }
    #[cfg(target_arch = "x86_64")]
    {
        exercise_math_backend::<dnam_engine::math::X86Sse2Backend>();
        exercise_math_backend::<dnam_engine::math::X86Sse42Backend>();
        exercise_math_backend::<dnam_engine::math::X86Avx2FmaBackend>();
        exercise_trainer_math_backend::<dnam_engine::math::X86Sse2Backend>();
        exercise_trainer_math_backend::<dnam_engine::math::X86Sse42Backend>();
        exercise_trainer_math_backend::<dnam_engine::math::X86Avx2FmaBackend>();
    }
}
