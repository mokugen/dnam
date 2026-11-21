//! End-to-end M10b `dnam-vm` integration tests: 0-ULP chunk-split invariance, Matryoshka
//! prefix stripping & crossfade, cross-ISA parity, `SO(4)` `Q_buyer` gauge invariance, ZOH
//! multi-rate Cases (a)-(d) + `4x` PDC, saturation telemetry, and `!alloc` execution.

#![forbid(unsafe_code)]

mod sona_vm_pipeline_and_parity {
    pub(crate) mod helpers;
    pub(crate) mod m10b_contracts;
    pub(crate) mod m9_contracts;
}

use dnam_vm::graph::ConvolverHighSrMode;
use dnam_vm::math::{detect_best_backend, saturation_flags};
use dnam_vm::{
    BackendKind, DefaultMathBackend, DynamicSonaVm, OFFLINE_BOUNCE_PDC_SAMPLES, ScalarBackend,
    SonaVm, TIER_SWITCH_XFADE_SAMPLES, Tier, control_stride_for_rate, detect_cpu_simd_features,
    strip_dnam_bin_to_tier,
};
use sona_vm_pipeline_and_parity::helpers::{
    assert_finite_non_silent, build_full_pipeline_ultra_fixture, build_so4_qbuyer_pair,
    run_no_alloc_lite, stereo_test_signal,
};

fn run_chunk_schedule(
    bin: &[u8],
    rate_hz: f32,
    in_l: &[f32],
    in_r: &[f32],
    schedule_a: &[usize],
    mid_update: Option<(usize, f32, usize, f32)>,
    schedule_b: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(bin, Tier::Ultra, rate_hz)
        .expect("open Ultra VM");
    let total = in_l.len();
    let mut out_l = vec![0.0_f32; total];
    let mut out_r = vec![0.0_f32; total];
    let mut cursor = 0usize;
    let mut idx = 0usize;
    let first_end = if mid_update.is_some() { 100 } else { total };

    while cursor < first_end {
        let step = schedule_a[idx % schedule_a.len()].min(first_end - cursor);
        vm.process_block(
            &in_l[cursor..cursor + step],
            &in_r[cursor..cursor + step],
            &mut out_l[cursor..cursor + step],
            &mut out_r[cursor..cursor + step],
        )
        .expect("chunk Phase A");
        cursor += step;
        idx += 1;
    }

    if let Some((c0, v0, c1, v1)) = mid_update {
        vm.set_control_physical(c0, v0).expect("mid ctrl 0");
        vm.set_control_physical(c1, v1).expect("mid ctrl 2");
        idx = 0;
        while cursor < total {
            let step = schedule_b[idx % schedule_b.len()].min(total - cursor);
            vm.process_block(
                &in_l[cursor..cursor + step],
                &in_r[cursor..cursor + step],
                &mut out_l[cursor..cursor + step],
                &mut out_r[cursor..cursor + step],
            )
            .expect("chunk Phase B");
            cursor += step;
            idx += 1;
        }
    }
    (out_l, out_r)
}

#[test]
fn test_zero_ulp_chunk_split_invariance_48k_and_44k1_ultra() {
    assert_eq!(control_stride_for_rate(48_000.0), 16);
    assert_eq!(control_stride_for_rate(44_100.0), 15);
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(512, 1);

    let primes = [1_usize, 2, 5, 11, 13, 17, 29, 37];
    let schedules: &[&[usize]] = &[&[1], &[3], &[4], &[7], &[16], &[31], &[64], &[128], &primes];

    for &rate in &[48_000.0_f32, 44_100.0_f32] {
        // Continuous 512-frame chunk splits without mid-stream breakpoints.
        let (ref_l, ref_r) = run_chunk_schedule(&ultra, rate, &in_l, &in_r, &[1], None, &[]);
        assert!(ref_l.iter().all(|v| v.is_finite()) && ref_l.iter().any(|v| v.abs() > 1.0e-4));
        for sched in schedules {
            let (gl, gr) = run_chunk_schedule(&ultra, rate, &in_l, &in_r, sched, None, &[]);
            for i in 0..512 {
                assert_eq!(gl[i].to_bits(), ref_l[i].to_bits(), "rate={rate} L@{i}");
                assert_eq!(gr[i].to_bits(), ref_r[i].to_bits(), "rate={rate} R@{i}");
            }
        }

        // step_frame (512x1) & process_tick16_const (32x16) 0-ULP parity.
        let mut vm_step =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, rate).unwrap();
        let mut vm_t16 =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, rate).unwrap();
        assert!(vm_step.telemetry().os2x_active);
        for i in 0..512 {
            let (sl, sr) = vm_step.step_frame(in_l[i], in_r[i]).unwrap();
            assert_eq!(sl.to_bits(), ref_l[i].to_bits());
            assert_eq!(sr.to_bits(), ref_r[i].to_bits());
        }
        for blk in 0..32 {
            let off = blk * 16;
            let il: [f32; 16] = in_l[off..off + 16].try_into().unwrap();
            let ir: [f32; 16] = in_r[off..off + 16].try_into().unwrap();
            let (mut ol, mut or) = ([0.0_f32; 16], [0.0_f32; 16]);
            vm_t16
                .process_tick16_const(&il, &ir, &mut ol, &mut or)
                .unwrap();
            for j in 0..16 {
                assert_eq!(ol[j].to_bits(), ref_l[off + j].to_bits());
                assert_eq!(or[j].to_bits(), ref_r[off + j].to_bits());
            }
        }

        // Mid-stream set_control_physical at frame 100 across chunk schedules.
        let mid = Some((0_usize, 1_100.0_f32, 2_usize, 0.8_f32));
        let (mid_ref_l, mid_ref_r) =
            run_chunk_schedule(&ultra, rate, &in_l, &in_r, &[1], mid, &[1]);
        for sched in schedules {
            let (gl, gr) = run_chunk_schedule(&ultra, rate, &in_l, &in_r, sched, mid, sched);
            for i in 0..512 {
                assert_eq!(gl[i].to_bits(), mid_ref_l[i].to_bits());
                assert_eq!(gr[i].to_bits(), mid_ref_r[i].to_bits());
            }
        }
    }
}

#[test]
fn test_matryoshka_tier_strip_prefix_equivalence_and_xfade() {
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(256, 2);

    for &tier in &[Tier::Lite, Tier::Full, Tier::Pro] {
        let stripped = strip_dnam_bin_to_tier(&ultra, tier).expect("strip tier");
        let mut vm_strip =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(stripped, tier, 48_000.0).unwrap();
        let mut vm_full =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, tier, 48_000.0).unwrap();

        assert_eq!(vm_strip.apply_tempo_bpm(135.0), 1);
        assert_eq!(vm_full.apply_tempo_bpm(135.0), 1);
        vm_strip.set_control_physical(0, 900.0).unwrap();
        vm_full.set_control_physical(0, 900.0).unwrap();

        let (mut sl, mut sr, mut fl, mut fr) = (
            vec![0.0_f32; 256],
            vec![0.0_f32; 256],
            vec![0.0_f32; 256],
            vec![0.0_f32; 256],
        );
        vm_strip
            .process_block(&in_l, &in_r, &mut sl, &mut sr)
            .unwrap();
        vm_full
            .process_block(&in_l, &in_r, &mut fl, &mut fr)
            .unwrap();
        assert!(sl.iter().any(|v| v.abs() > 1.0e-4));
        for i in 0..256 {
            assert_eq!(sl[i].to_bits(), fl[i].to_bits(), "tier={tier:?} L@{i}");
            assert_eq!(sr[i].to_bits(), fr[i].to_bits(), "tier={tier:?} R@{i}");
        }
    }

    // Dynamic switch_tier sequence Ultra -> Lite -> Pro -> Ultra past 48-frame crossfade.
    assert_eq!(TIER_SWITCH_XFADE_SAMPLES, 48);
    let mut vm = DynamicSonaVm::open_dnam_bin(&ultra, Tier::Ultra, 48_000.0).unwrap();
    let (mut out_l, mut out_r) = ([0.0_f32; 64], [0.0_f32; 64]);
    vm.process_block(&in_l[..64], &in_r[..64], &mut out_l, &mut out_r)
        .unwrap();
    for &target in &[Tier::Lite, Tier::Pro, Tier::Full, Tier::Ultra] {
        vm.switch_tier(target).unwrap();
        assert_eq!(vm.telemetry().active_tier, target);
        vm.process_block(&in_l[64..128], &in_r[64..128], &mut out_l, &mut out_r)
            .unwrap();
        assert!(out_l.iter().all(|v| v.is_finite()) && out_r.iter().all(|v| v.is_finite()));
        assert!(out_l[63].abs() > 1.0e-4);
    }
}

fn push_unique_backend(kinds: &mut Vec<BackendKind>, kind: BackendKind) {
    if !kinds.contains(&kind) {
        kinds.push(kind);
    }
}

fn host_backend_kinds() -> Vec<BackendKind> {
    let features = detect_cpu_simd_features();
    assert_eq!(features.backend, detect_best_backend());
    let mut kinds = Vec::with_capacity(5);
    push_unique_backend(&mut kinds, BackendKind::Scalar);
    push_unique_backend(&mut kinds, features.backend);
    #[cfg(target_arch = "aarch64")]
    {
        push_unique_backend(&mut kinds, BackendKind::ArmNeon);
        if features.sve2 {
            push_unique_backend(&mut kinds, BackendKind::ArmSve2);
        }
        if features.sme {
            push_unique_backend(&mut kinds, BackendKind::ArmSme);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        push_unique_backend(&mut kinds, BackendKind::X86Sse2);
        push_unique_backend(&mut kinds, BackendKind::X86Sse42);
        if features.avx2_fma {
            push_unique_backend(&mut kinds, BackendKind::X86Avx2Fma);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        push_unique_backend(&mut kinds, BackendKind::WasmSimd128);
    }
    kinds
}

#[test]
fn test_cross_isa_scalar_vs_default_and_dynamic_parity() {
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(1_024, 4);
    let host_kinds = host_backend_kinds();
    assert!(
        (1..host_kinds.len()).all(|i| !host_kinds[..i].contains(&host_kinds[i])),
        "host_backend_kinds must contain unique entries: {host_kinds:?}"
    );

    for tier in Tier::ALL {
        let mut vm_scalar =
            SonaVm::<ScalarBackend>::open_dnam_bin_owned(&ultra, tier, 48_000.0).unwrap();
        let mut vm_default =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, tier, 48_000.0).unwrap();
        let (mut sl, mut sr, mut dl, mut dr) = (
            vec![0.0_f32; 1_024],
            vec![0.0_f32; 1_024],
            vec![0.0_f32; 1_024],
            vec![0.0_f32; 1_024],
        );
        vm_scalar
            .process_block(&in_l, &in_r, &mut sl, &mut sr)
            .unwrap();
        vm_default
            .process_block(&in_l, &in_r, &mut dl, &mut dr)
            .unwrap();

        let mut max_diff = 0.0_f32;
        for i in 0..1_024 {
            assert!(dl[i].is_finite() && dr[i].is_finite());
            max_diff = max_diff
                .max((dl[i] - sl[i]).abs())
                .max((dr[i] - sr[i]).abs());
        }
        assert_finite_non_silent(&sl, &sr, &format!("scalar parity {tier:?}"));
        assert_finite_non_silent(&dl, &dr, &format!("default parity {tier:?}"));
        assert!(
            max_diff <= 1.0e-5,
            "Scalar vs Default diff {max_diff} at {tier:?}"
        );

        for &kind in &host_kinds {
            let mut dyn_vm =
                DynamicSonaVm::open_with_backend(&ultra, tier, 48_000.0, kind).unwrap();
            let (mut yl, mut yr) = (vec![0.0_f32; 1_024], vec![0.0_f32; 1_024]);
            dyn_vm
                .process_block(&in_l, &in_r, &mut yl, &mut yr)
                .unwrap();
            assert_finite_non_silent(&yl, &yr, &format!("dyn parity {kind:?} {tier:?}"));
            let mut d_diff = 0.0_f32;
            for i in 0..1_024 {
                assert!(yl[i].is_finite() && yr[i].is_finite());
                d_diff = d_diff.max((yl[i] - sl[i]).abs()).max((yr[i] - sr[i]).abs());
            }
            assert!(d_diff <= 1.0e-5, "{kind:?} diff {d_diff} at {tier:?}");
        }
    }
}

#[test]
fn test_dynamic_and_sona_vm_lifecycle_and_batch_controls() {
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(64, 9);
    let kinds = host_backend_kinds();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let wasm_fallback = DynamicSonaVm::open_with_backend(
            &ultra,
            Tier::Lite,
            48_000.0,
            BackendKind::WasmSimd128,
        )
        .unwrap();
        assert_eq!(wasm_fallback.backend_kind(), BackendKind::Scalar);
    }

    for &kind in &kinds {
        let mut dyn_vm =
            DynamicSonaVm::open_with_backend(&ultra, Tier::Ultra, 48_000.0, kind).unwrap();
        assert_eq!(dyn_vm.backend_kind(), kind);
        assert_eq!(dyn_vm.apply_tempo_bpm(135.0), 1);
        dyn_vm.set_control_physical(0, 1_050.0).unwrap();
        dyn_vm
            .set_controls_batch(&[(0, 950.0), (1, 1.0), (2, 0.5)])
            .unwrap();
        dyn_vm.set_quad_write_enable(true);
        assert!(dyn_vm.telemetry().host_write_lock);
        dyn_vm.set_quad_write_enable(false);
        assert!(!dyn_vm.telemetry().host_write_lock);
        dyn_vm.set_offline_bounce(true).unwrap();
        dyn_vm.set_offline_bounce(false).unwrap();
        dyn_vm.switch_tier(Tier::Pro).unwrap();
        assert_eq!(dyn_vm.telemetry().active_tier, Tier::Pro);
        dyn_vm.retime_sample_rate(96_000.0).unwrap();

        let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);
        dyn_vm
            .process_block(&in_l, &in_r, &mut ol, &mut or)
            .unwrap();
        assert_finite_non_silent(&ol, &or, &format!("dyn lifecycle {kind:?}"));

        let il16: [f32; 16] = in_l[..16].try_into().unwrap();
        let ir16: [f32; 16] = in_r[..16].try_into().unwrap();
        let (mut ol16, mut or16) = ([0.0_f32; 16], [0.0_f32; 16]);
        dyn_vm
            .process_tick16_const(&il16, &ir16, &mut ol16, &mut or16)
            .unwrap();
        assert!(ol16.iter().all(|v| v.is_finite()));

        let (sf_l, sf_r) = dyn_vm.step_frame(in_l[0], in_r[0]).unwrap();
        assert!(sf_l.is_finite() && sf_r.is_finite());

        dyn_vm.reset();
        assert!(!dyn_vm.telemetry().host_write_lock);
    }

    // Exercise SonaVm batch, 3 kHz tick, lock, drift, and slab_mut APIs.
    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 48_000.0).unwrap();
    vm.set_controls_batch(&[(0, 920.0), (1, 1.0), (2, 0.25)])
        .unwrap();
    vm.step_control_tick_3khz();
    vm.set_op_t_write_enable(true);
    assert!(vm.telemetry().host_write_lock);
    vm.set_op_t_write_enable(false);
    vm.set_live_analog_drift_enabled(false);
    vm.set_live_analog_drift_enabled(true);
    let _ = vm.slab_mut().lite.transport_quad0.flags;
    let (sl, sr) = vm.step_frame(in_l[0], in_r[0]).unwrap();
    assert!(sl.is_finite() && sr.is_finite());
    vm.clear_saturation_flags();
    assert_eq!(vm.saturation_flags(), 0);
}

#[test]
fn test_intra_block_so4_q_buyer_gauge_invariance() {
    let (bytes_a, bytes_b) = build_so4_qbuyer_pair(0.73);
    let mut vm_a =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bytes_a, Tier::Lite, 48_000.0).unwrap();
    let mut vm_b =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&bytes_b, Tier::Lite, 48_000.0).unwrap();

    let (in_l, in_r) = stereo_test_signal(512, 5);
    let (mut al, mut ar, mut bl, mut br) = (
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
    );
    vm_a.process_block(&in_l, &in_r, &mut al, &mut ar).unwrap();
    vm_b.process_block(&in_l, &in_r, &mut bl, &mut br).unwrap();

    let max_sig = al.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(max_sig > 1.0e-4, "expected nontrivial gauge test output");
    let mut max_diff = 0.0_f32;
    for i in 0..512 {
        max_diff = max_diff
            .max((al[i] - bl[i]).abs())
            .max((ar[i] - br[i]).abs());
    }
    assert!(
        max_diff <= 1.0e-5,
        "SO(4) Q_buyer max diff {max_diff} exceeded 1e-5"
    );
}

#[test]
fn test_multi_rate_zoh_cases_abcd_os2x_bypass_retime_and_4x_pdc() {
    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(64, 6);
    let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);

    let rate_cases = [
        (44_100.0_f32, ConvolverHighSrMode::CaseB, true),
        (48_000.0_f32, ConvolverHighSrMode::Native48k, true),
        (88_200.0_f32, ConvolverHighSrMode::CaseD, true),
        (
            96_000.0_f32,
            ConvolverHighSrMode::CaseA { stride_q: 2 },
            true,
        ),
        (176_400.0_f32, ConvolverHighSrMode::CaseD, true),
        (
            192_000.0_f32,
            ConvolverHighSrMode::CaseA { stride_q: 4 },
            true,
        ),
        (220_500.0_f32, ConvolverHighSrMode::CaseD, false),
        (
            384_000.0_f32,
            ConvolverHighSrMode::CaseA { stride_q: 8 },
            false,
        ),
    ];
    for (rate, exp_mode, exp_os2x) in rate_cases {
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, rate).unwrap();
        assert_eq!(vm.slab().cold().convolver_fir_ext.mode(), exp_mode);
        assert_eq!(vm.telemetry().os2x_active, exp_os2x);
        vm.process_block(&in_l, &in_r, &mut ol, &mut or).unwrap();
        assert_finite_non_silent(&ol, &or, &format!("open rate={rate}"));
    }

    // 4x offline bounce Case (c) at 44.1 kHz (176.4 kHz effective VM rate) + OS2X auto-bypass.
    let mut vm_b44 =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 44_100.0).unwrap();
    vm_b44.set_offline_bounce(true).unwrap();
    assert_eq!(
        vm_b44.slab().cold().convolver_fir_ext.mode(),
        ConvolverHighSrMode::CaseC
    );
    assert!(!vm_b44.telemetry().os2x_active);
    assert_eq!(vm_b44.get_latency_samples(), OFFLINE_BOUNCE_PDC_SAMPLES);
    vm_b44
        .process_block(&in_l, &in_r, &mut ol, &mut or)
        .unwrap();
    assert_finite_non_silent(&ol, &or, "4x offline bounce 44.1k");

    // Live retime_sample_rate across 44.1..=384 kHz.
    let mut vm_live =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 48_000.0).unwrap();
    for &(rate, exp_mode, exp_os2x) in &rate_cases {
        vm_live.retime_sample_rate(rate).unwrap();
        assert_eq!(vm_live.slab().cold().convolver_fir_ext.mode(), exp_mode);
        assert_eq!(vm_live.telemetry().os2x_active, exp_os2x);
        vm_live
            .process_block(&in_l, &in_r, &mut ol, &mut or)
            .unwrap();
        assert_finite_non_silent(&ol, &or, &format!("live retime={rate}"));
    }

    // Exact 12-sample PDC group-delay alignment on a low-frequency windowed sine pulse.
    let lite_bytes = strip_dnam_bin_to_tier(&ultra, Tier::Lite).unwrap();
    let mut pulse = vec![0.0_f32; 220];
    for (i, sample) in pulse[20..100].iter_mut().enumerate() {
        let env = 0.5 * (1.0 - (2.0 * core::f32::consts::PI * (i as f32) / 80.0).cos());
        let carrier = (2.0 * core::f32::consts::PI * 200.0 * (i as f32) / 48_000.0).sin();
        *sample = 0.4 * env * carrier;
    }
    let mut vm_l1 =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(lite_bytes, Tier::Lite, 48_000.0)
            .unwrap();
    let mut vm_b1 =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(lite_bytes, Tier::Lite, 48_000.0)
            .unwrap();
    vm_b1.set_offline_bounce(true).unwrap();
    assert_eq!(vm_l1.get_latency_samples(), 0);
    assert_eq!(vm_b1.get_latency_samples(), 12);

    let (mut live_l, mut live_r, mut bnc_l, mut bnc_r) = (
        vec![0.0_f32; 220],
        vec![0.0_f32; 220],
        vec![0.0_f32; 220],
        vec![0.0_f32; 220],
    );
    vm_l1
        .process_block(&pulse, &pulse, &mut live_l, &mut live_r)
        .unwrap();
    vm_b1
        .process_block(&pulse, &pulse, &mut bnc_l, &mut bnc_r)
        .unwrap();

    let mut best_lag = 0_usize;
    let mut best_corr = f64::NEG_INFINITY;
    for lag in 0..=24_usize {
        let mut corr = 0.0_f64;
        for n in 20..(220 - 24) {
            corr += f64::from(live_l[n]) * f64::from(bnc_l[n + lag]);
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }
    assert_eq!(best_lag, OFFLINE_BOUNCE_PDC_SAMPLES);
}

#[test]
fn test_saturation_telemetry_and_no_alloc_lite_parity() {
    use dnam_vm::bin::{DNBN_FLAG_STEREO_OUT, DnamBinBuildInput, SONA_META_TRANSPORT_TAP_GAINS};
    use dnam_vm::math::f32_to_fp16;
    use sona_vm_pipeline_and_parity::helpers::{
        encode_lite_fixture, make_core_quad, make_transport_quad,
    };
    use zerocopy::byteorder::little_endian::{F32, U16, U32};

    let ultra = build_full_pipeline_ultra_fixture();
    let (in_l, in_r) = stereo_test_signal(64, 7);

    let mut vm =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&ultra, Tier::Ultra, 48_000.0).unwrap();
    vm.clear_saturation_flags();
    assert_eq!(vm.saturation_flags(), 0);
    let _ = vm.step_frame(f32::NAN, 0.1).unwrap();
    assert_ne!(
        vm.saturation_flags() & saturation_flags::INVALID_NONFINITE,
        0
    );
    vm.clear_saturation_flags();

    let _ = vm.step_frame(200.0, -200.0).unwrap();
    assert_ne!(
        vm.saturation_flags() & saturation_flags::OUTSIDE_CERTIFIED_INPUT,
        0
    );
    vm.clear_saturation_flags();

    // Linear direct path (VCA weights 0) drives transport fp16 saturation and output ceiling clamp.
    let mut a0_sat = DnamBinBuildInput::default_arena0_lite();
    a0_sat.crossbar.sona_meta_flags = U32::new(SONA_META_TRANSPORT_TAP_GAINS);
    a0_sat.crossbar.cell_desc.cell_flags = U32::new(SONA_META_TRANSPORT_TAP_GAINS);
    a0_sat.core_quads[0] = make_core_quad(0);
    a0_sat.conv_fir_lite.fir_l[0] = F32::new(1.0);
    a0_sat.conv_fir_lite.fir_r[0] = F32::new(1.0);
    a0_sat.linear_readout_lite.direct_s2_u_gain = [F32::new(2.0), F32::new(2.0)];
    a0_sat.transport_lite = make_transport_quad(0);
    a0_sat.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(1.0));
    let sat_bin = encode_lite_fixture(&a0_sat, DNBN_FLAG_STEREO_OUT, 0.99, &[], None);
    let mut vm_sat =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&sat_bin, Tier::Lite, 48_000.0).unwrap();

    let _ = vm_sat.step_frame(2_000_000.0, 2_000_000.0).unwrap();
    assert_ne!(
        vm_sat.saturation_flags() & saturation_flags::FP16_SATURATED,
        0
    );
    assert_eq!(
        vm_sat.saturation_flags() & saturation_flags::INPUT_SATURATED,
        0
    );
    vm_sat.clear_saturation_flags();

    for _ in 0..4 {
        let (clamped_l, clamped_r) = vm_sat.step_frame(2.0e9, 2.0e9).unwrap();
        assert!(clamped_l.abs() <= 1.0e9 && clamped_r.abs() <= 1.0e9);
    }
    let flags = vm_sat.telemetry().saturation_flags;
    assert_ne!(flags & saturation_flags::INPUT_SATURATED, 0);
    assert_ne!(flags & saturation_flags::OUTPUT_SATURATED, 0);
    assert_ne!(flags & saturation_flags::OUTSIDE_CERTIFIED_INPUT, 0);
    vm_sat.clear_saturation_flags();
    assert_eq!(vm_sat.saturation_flags(), 0);

    // !alloc stack-only Lite VM parity vs owned Lite VM.
    let lite_bytes = strip_dnam_bin_to_tier(&ultra, Tier::Lite).unwrap();
    let in_l_64: [f32; 64] = in_l[..64].try_into().unwrap();
    let in_r_64: [f32; 64] = in_r[..64].try_into().unwrap();
    let (nl0, nr0, nl1, nr1) = run_no_alloc_lite(lite_bytes, &in_l_64, &in_r_64);

    let mut vm_owned =
        SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(lite_bytes, Tier::Lite, 48_000.0)
            .unwrap();
    let (mut ol, mut or) = ([0.0_f32; 64], [0.0_f32; 64]);
    vm_owned
        .process_block(&in_l_64, &in_r_64, &mut ol, &mut or)
        .unwrap();
    for i in 0..64 {
        assert_eq!(nl0[i].to_bits(), ol[i].to_bits());
        assert_eq!(nr0[i].to_bits(), or[i].to_bits());
        assert_eq!(nl1[i].to_bits(), ol[i].to_bits());
        assert_eq!(nr1[i].to_bits(), or[i].to_bits());
    }
}
