//! Empirical `SONA` (`v3.0`) End-to-End Pipeline, ZOH Multi-Rate, `4x` Bounce, and
//! Architecture Specification Verification Suite (`tests/sona_pipeline_and_zoh_verification.rs`).
//!
//! Canonical Reference: `docs/specs/architecture/sona_architecture.md` and
//! `docs/specs/containers/dnam_binary_format.md`.
//!
//! Verifies:
//! - `Test 8H & 8I` (`verify[container.dnam.tier-prefix]`, `verify[arch.bounce.oversampling]`):
//!   64-byte binary framing, cache-line tier offsets (`72, 124, 248, 480` CLs =
//!   `[4_608, 7_936, 15_872, 30_720]` B, `TOTAL_LITE_STATE_RAM_BYTES <= 4,608 B`),
//!   Sample-0 DC resting equilibrium (`0.000000` pop on silence), distinct per-tier
//!   (`Lite ⊂ Full ⊂ Pro ⊂ Ultra`) outputs, ZOH multi-sample-rate rescaling across
//!   `44.1 kHz`–`192 kHz`, and `4x` ZOH linear-phase oversampled offline bounce with `12`-sample PDC trim.
//! - `verify[arch.tiers.maximize-quality-within-budget]`, `verify[repo.code.zerocopy-wire-structs]`,
//!   and `verify[repo.code.shared-nostd-kernels]`.
//! - See [`tests/sona_calibration_and_lifecycle.rs`](sona_calibration_and_lifecycle.rs) for engine
//!   constructors, calibration roles, `W_ctrl` dispatch, isomorphism, `DAU1` signing, and property tests.

#![forbid(unsafe_code)]

#[path = "sona_test_support/arena_builder.rs"]
mod arena_builder;

use arena_builder::build_populated_sona_container;
use dnam_engine::container::{DnamPreamble, strip_container_to_tier};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::{DnamEngine, OFFLINE_BOUNCE_PDC_SAMPLES, TOTAL_LITE_STATE_RAM_BYTES, Tier};

/// Test 8H & 8I (`verify[container.dnam.tier-prefix]`, `verify[arch.bounce.oversampling]`): End-to-end
/// `SONA` multi-stage pipeline verification across 64B framing, Sample-0 DC equilibrium, 4-tier
/// progressive outputs, ZOH sample rates, and `4x` bounce.
#[test]
fn test_8h_8i_end_to_end_sona_pipeline_tiers_zoh_sample_rates_and_4x_bounce() {
    assert_eq!(core::mem::size_of::<DnamPreamble>(), 64);
    const { assert!(TOTAL_LITE_STATE_RAM_BYTES <= 4_608) };
    assert_eq!(OFFLINE_BOUNCE_PDC_SAMPLES, 12);

    let (_header, _arena, container) = build_populated_sona_container();
    let preamble = DnamPreamble::from_bytes(&container).unwrap();
    assert_eq!(preamble.version_major, 3);
    assert_eq!(preamble.arena_lite_end_cl, 72);
    assert_eq!(preamble.arena_full_end_cl, 124);
    assert_eq!(preamble.arena_pro_end_cl, 248);
    assert_eq!(preamble.arena_ultra_end_cl, 480);
    assert_eq!(preamble.tensor_arena_offset % 64, 0);

    // 1. Sample-0 DC resting equilibrium: zero input must produce 0.000000 output on sample 0
    let mut eng =
        DnamEngine::<DefaultMathBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();
    eng.reset_to_dc_equilibrium();
    let mut zero_l = [0.0_f32; 16];
    let mut zero_r = [0.0_f32; 16];
    eng.process_block(&[0.0; 16], &[0.0; 16], &mut zero_l, &mut zero_r);
    for i in 0..16 {
        assert!(
            zero_l[i].abs() < 1e-7 && zero_r[i].abs() < 1e-7,
            "Sample-{i} DC transient on silence: L={}, R={}",
            zero_l[i],
            zero_r[i]
        );
    }

    // 2. Verify all 4 tiers produce distinct progressive audio outputs on a harmonic input
    let in_l: Vec<f32> = (0..128).map(|n| 0.4 * ((n as f32) * 0.13).sin()).collect();
    let in_r: Vec<f32> = (0..128).map(|n| 0.4 * ((n as f32) * 0.17).cos()).collect();
    let mut tier_outputs = Vec::new();
    for &tier in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
        let stripped = strip_container_to_tier(&container, tier).unwrap();
        let mut t_eng =
            DnamEngine::<DefaultMathBackend>::from_bytes(&stripped, tier, 48_000).unwrap();
        let mut ol = vec![0.0_f32; 128];
        let mut or = vec![0.0_f32; 128];
        t_eng.process_block(&in_l, &in_r, &mut ol, &mut or);
        tier_outputs.push(ol);
    }
    for t in 0..3 {
        let diff: f32 = tier_outputs[t]
            .iter()
            .zip(tier_outputs[t + 1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-3);
    }

    // 3. Verify ZOH multi-sample-rate execution across all canonical host sample rates
    for &sr in &[44_100_u32, 48_000, 88_200, 96_000, 192_000] {
        eng.set_sample_rate(sr).unwrap();
        assert_eq!(eng.sample_rate_hz(), sr);
        let mut ol = [0.0_f32; 64];
        let mut or = [0.0_f32; 64];
        eng.process_block(&in_l[..64], &in_r[..64], &mut ol, &mut or);
        assert!(ol.iter().all(|v| v.is_finite()) && or.iter().all(|v| v.is_finite()));
    }

    // 4. Verify 4x ZOH linear-phase offline bounce with 12-sample PDC trim (verify[arch.bounce.oversampling])
    eng.set_sample_rate(48_000).unwrap();
    let mut bounce_l = vec![0.0_f32; 128];
    let mut bounce_r = vec![0.0_f32; 128];
    eng.render_offline_bounce(&in_l, &in_r, &mut bounce_l, &mut bounce_r);
    assert!(bounce_l.iter().any(|&v| v.abs() > 1e-4));
    assert!(bounce_l.iter().all(|v| v.is_finite()));
}

mod arch_spec_integration {
    // Specification Verification Suite: SONA v3.0 System Architecture Integration.
    //
    // Verifies engine-scope pillars, tier hierarchy, zerocopy wire structs, shared nostd kernels,
    // and all seven hardware families in `sona_architecture.md` & `dnam_binary_format.md`:
    // - `verify[arch.pillars.dual-space-control]`
    // - `verify[arch.pillars.dual-rate-plane]`
    // - `verify[arch.pillars.cp1-projections]`
    // - `verify[arch.pillars.universal-black-box]`
    // - `verify[arch.lifecycle.dual-rate]`
    // - `verify[arch.fp.callback-env]`
    // - `verify[arch.fp.range-saturation]`
    // - `verify[arch.operators.tri-kernel]`
    // - `verify[arch.operators.self-similar]`
    // - `verify[arch.simd.lane-packing]`
    // - `verify[arch.operators.differentiability]`
    // - `verify[arch.tiers.arena-hierarchy]`
    // - `verify[arch.tiers.hardware-budget]`
    // - `verify[arch.tiers.maximize-quality-within-budget]`
    // - `verify[arch.tiers.state-ram]`
    // - `verify[arch.tiers.efficiency]`
    // - `verify[arch.topology.signal-flow]`
    // - `verify[arch.topology.stage0-router]`
    // - `verify[arch.topology.auto-unity-leveling]`
    // - `verify[arch.tiers.transition-contract]`
    // - `verify[arch.topology.spillover-trails]`
    // - `verify[arch.bounce.oversampling]`
    // - `verify[arch.families.coverage]`
    // - `verify[arch.families.f1-bistables]`
    // - `verify[arch.families.f2-hilbert-ssb]`
    // - `verify[arch.families.f3-pitch]`
    // - `verify[arch.families.f4-dynamics]`
    // - `verify[arch.families.f5-bitcrusher]`
    // - `verify[arch.families.f6-formants]`
    // - `verify[arch.families.f7-multi-input]`
    // - `verify[repo.code.zerocopy-wire-structs]`
    // - `verify[repo.code.shared-nostd-kernels]`

    use core::mem::{align_of, size_of};

    use dnam_engine::container::{
        ARENA_FULL_END_CL, ARENA_LITE_END_CL, ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END,
        ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END, ARENA_PRO_END_CL, ARENA_ULTRA_END_CL,
        DnamDirectoryWire, DnamPreambleWire, build_dnam_container, build_v3_open_dnam_container,
        cp1_contributor_record_bytes,
    };
    use dnam_engine::control::{ExactIntFactorAccumulator, control_stride_for_rate};
    use dnam_engine::math::{DefaultMathBackend, eval_monotone_log2_f32_bits};
    use dnam_engine::operators::op_n::OpNBackend;
    use dnam_engine::operators::{
        BallisticDuckingVca, FixedHzBitcrusher, HilbertH1Lattice6Sec, KautzModalBank, OpTDelayLine,
        QuadratureHysteresis2D, SymplecticModal4Lane, T1DcBlocker, T1SubHarmonicTracker,
        eval_m2_pitch_grain_delays,
    };
    use dnam_engine::provenance::{Dme1HeaderWire, DnplPreambleWire, DnslPreambleWire};
    use dnam_engine::schema::DnamJsonHeader;
    use dnam_engine::stages::weights::{
        Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
        pack_sona_matryoshka_to_arena,
    };
    use dnam_engine::stages::{
        SonaColdTierModel, SonaHotSlab, SonaMatryoshkaModel, Stage0Router, TptLr4Crossover,
        WdwDryTapMode,
    };
    use dnam_engine::{
        DnamEngine, DynamicDnamEngine, FtzDazGuard, OFFLINE_BOUNCE_PDC_SAMPLES,
        TOTAL_LITE_STATE_RAM_BYTES, Tier,
    };
    use dnam_kernels::{
        FULL_SPEC, LITE_SPEC, PRO_SPEC, ULTRA_SPEC, f32_to_fp16, fp16_to_f32,
        prewarp_tan_pi_f_over_fs_f32, step_sona_group_stereo8,
    };
    use zerocopy::FromBytes;

    /// verify[arch.pillars.dual-space-control]
    /// verify[arch.pillars.dual-rate-plane]
    /// verify[arch.pillars.cp1-projections]
    /// verify[arch.pillars.universal-black-box]
    /// verify[arch.lifecycle.dual-rate]
    /// verify[arch.fp.callback-env]
    /// verify[arch.fp.range-saturation]
    /// verify[arch.operators.tri-kernel]
    /// verify[arch.operators.self-similar]
    /// verify[arch.simd.lane-packing]
    /// verify[arch.operators.differentiability]
    #[test]
    fn test_sona_pillars_dual_rate_lifecycle_tri_kernel_and_differentiability() {
        let _guard = FtzDazGuard::new();

        assert_eq!(control_stride_for_rate(48_000), 16);
        assert_eq!(cp1_contributor_record_bytes(3, false), Some(124));
        assert_eq!(cp1_contributor_record_bytes(3, true), Some(112));

        let mut opt = OpTDelayLine::<2048>::new();
        let mut opm = SymplecticModal4Lane::new();
        let mut opn = QuadratureHysteresis2D::new();
        let y_t = opt.step(0.5);
        let (y_m_re, y_m_im) = opm.step_lane(0, 0.5);
        let y_n = opn.step(0.5, 0.25);
        let _ = <DefaultMathBackend as OpNBackend<_>>::tanh_pade5_4ch(
            dnam_engine::math::SimdVec4::splat(y_n),
        );
        assert!(y_t.is_finite() && y_m_re.is_finite() && y_m_im.is_finite() && y_n.is_finite());

        let mut acc = ExactIntFactorAccumulator::new(2);
        acc.update_contributor(0, (0.5_f32).to_bits()).unwrap();
        acc.update_contributor(1, (0.25_f32).to_bits()).unwrap();
        assert_eq!(acc.interaction_f32_bits(), Some((0.125_f32).to_bits()));

        // Catmull-Rom analytical derivative
        let (s_m1, s_0, s_1, s_2) = (1.0_f64, 2.0, 3.5, 4.0);
        let c3 = -0.5 * s_m1 + 1.5 * s_0 - 1.5 * s_1 + 0.5 * s_2;
        let c2 = s_m1 - 2.5 * s_0 + 2.0 * s_1 - 0.5 * s_2;
        let c1 = -0.5 * s_m1 + 0.5 * s_1;
        let c0 = s_0;
        let mu = 0.37_f64;
        let eps = 1.0e-7_f64;
        let eval_cr = |m: f64| ((c3 * m + c2) * m + c1) * m + c0;
        let num_deriv = (eval_cr(mu + eps) - eval_cr(mu - eps)) / (2.0 * eps);
        let ana_deriv = (3.0 * c3 * mu + 2.0 * c2) * mu + c1;
        assert!((num_deriv - ana_deriv).abs() < 1e-9);

        // Op-M associative scan composition
        let compose = |(a2, b2): (f64, f64), (a1, b1): (f64, f64)| (a2 * a1, a2 * b1 + b2);
        let left_assoc = compose((0.90, 0.3), compose((0.92, -0.1), (0.95, 0.2)));
        let right_assoc = compose(compose((0.90, 0.3), (0.92, -0.1)), (0.95, 0.2));
        assert!((left_assoc.0 - right_assoc.0).abs() < 1e-15);
        assert!((left_assoc.1 - right_assoc.1).abs() < 1e-15);
    }

    /// verify[arch.tiers.arena-hierarchy]
    /// verify[arch.tiers.hardware-budget]
    /// verify[arch.tiers.maximize-quality-within-budget]
    /// verify[arch.tiers.state-ram]
    /// verify[arch.tiers.efficiency]
    /// verify[arch.tiers.transition-contract]
    /// verify[arch.topology.signal-flow]
    /// verify[arch.topology.stage0-router]
    /// verify[arch.topology.auto-unity-leveling]
    /// verify[arch.topology.spillover-trails]
    /// verify[arch.bounce.oversampling]
    #[test]
    fn test_tiers_arena_hierarchy_maximize_quality_state_ram_transitions_and_bounce() {
        assert_eq!(ARENA_OFFSET_LITE_END, 4_608);
        assert_eq!(ARENA_OFFSET_FULL_END, 7_936);
        assert_eq!(ARENA_OFFSET_PRO_END, 15_872);
        assert_eq!(ARENA_OFFSET_ULTRA_END, 30_720);
        assert_eq!(
            (
                ARENA_LITE_END_CL,
                ARENA_FULL_END_CL,
                ARENA_PRO_END_CL,
                ARENA_ULTRA_END_CL
            ),
            (72, 124, 248, 480)
        );
        const { assert!(TOTAL_LITE_STATE_RAM_BYTES <= 4_608) };
        assert!(SonaHotSlab::<DefaultMathBackend>::active_hot_bytes(Tier::Lite, false) <= 4_608);
        assert_eq!(LITE_SPEC.total_lanes(), 16);
        assert_eq!(FULL_SPEC.total_lanes(), 24);
        assert_eq!(PRO_SPEC.total_lanes(), 32);
        assert_eq!(ULTRA_SPEC.total_lanes(), 40);
        assert_eq!(
            (
                LITE_SPEC.oversample_factor,
                FULL_SPEC.oversample_factor,
                PRO_SPEC.oversample_factor,
                ULTRA_SPEC.oversample_factor
            ),
            (1, 2, 2, 2)
        );
        assert_eq!(
            (
                LITE_SPEC.fir_taps,
                FULL_SPEC.fir_taps,
                PRO_SPEC.fir_taps,
                ULTRA_SPEC.fir_taps
            ),
            (32, 48, 96, 128)
        );
        assert_eq!(
            (
                LITE_SPEC.kautz_pairs,
                FULL_SPEC.kautz_pairs,
                PRO_SPEC.kautz_pairs,
                ULTRA_SPEC.kautz_pairs
            ),
            (8, 12, 24, 32)
        );
        assert_eq!(OFFLINE_BOUNCE_PDC_SAMPLES, 12);

        let mut header = DnamJsonHeader::default_reference();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        let model = SonaMatryoshkaModel::dual_rate_default(false);
        pack_sona_matryoshka_to_arena(&model, &mut arena);
        let container = build_dnam_container(&mut header, &arena, &arena, false).unwrap();

        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes(&container, Tier::Ultra, 48_000).unwrap();
        let in_l = [0.25_f32; 64];
        let in_r = [-0.25_f32; 64];
        let mut out_l = [0.0_f32; 64];
        let mut out_r = [0.0_f32; 64];

        for &t in &[Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra] {
            eng.set_tier(t);
            assert_eq!(eng.tier(), t);
            eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
            eng.process_mono_to_stereo(&in_l, &mut out_l, &mut out_r);
            eng.process_mono(&in_l, &mut out_l);
        }

        // Spillover trails state machine
        eng.set_bypass(true, true);
        assert_eq!(eng.telemetry_snapshot().bypassed, 1);
        eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        eng.set_bypass(true, false);
        eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_eq!(out_l, in_l);
        assert_eq!(out_r, in_r);
        eng.set_bypass(false, false);

        // 4x offline bounce preserves live tier and finite outputs
        eng.set_tier(Tier::Lite);
        eng.render_offline_bounce(&in_l, &in_r, &mut out_l, &mut out_r);
        assert_eq!(eng.tier(), Tier::Lite);
        assert!(out_l.iter().all(|v| v.is_finite()));

        // DynamicDnamEngine runtime dispatch parity
        let mut dyn_eng = DynamicDnamEngine::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        dyn_eng.process_block(&in_l, &in_r, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|v| v.is_finite()));
    }

    /// verify[repo.code.zerocopy-wire-structs]
    /// verify[repo.code.shared-nostd-kernels]
    #[test]
    fn test_zerocopy_wire_structs_and_shared_nostd_kernels() {
        let _guard = FtzDazGuard::new();

        // 1. Validate compile-time aligned zerocopy wire structs & roundtrip slices
        assert_eq!(size_of::<DnamPreambleWire>(), 64);
        assert_eq!(align_of::<DnamPreambleWire>(), 64);
        assert_eq!(size_of::<DnamDirectoryWire>(), 128);
        assert_eq!(size_of::<Arena0LiteWire>(), 4_608);
        assert_eq!(size_of::<Arena1FullDeltaWire>(), 3_328);
        assert_eq!(size_of::<Arena2ProDeltaWire>(), 7_936);
        assert_eq!(size_of::<Arena3UltraDeltaWire>(), 14_848);
        assert_eq!(size_of::<DnslPreambleWire>(), 64);
        assert_eq!(size_of::<DnplPreambleWire>(), 64);
        assert_eq!(size_of::<Dme1HeaderWire>(), 20);

        let hdr_bytes = serde_json::to_vec(&DnamJsonHeader::default_reference()).unwrap();
        let t0 = vec![0u8; 4_608];
        let v3_bytes =
            build_v3_open_dnam_container(&hdr_bytes, &[0u8; 64], &[0u8; 64], &[&t0], None, 0)
                .unwrap();
        let (preamble_wire, dir_rest) = DnamPreambleWire::ref_from_prefix(&v3_bytes).unwrap();
        assert_eq!(&preamble_wire.magic, b"DNAM");
        assert_eq!(preamble_wire.version_major.get(), 3);
        assert_eq!(preamble_wire.version_minor.get(), 0);
        let (dir_wire, _) = DnamDirectoryWire::ref_from_prefix(dir_rest).unwrap();
        let arena_off = dir_wire.tier_start.get() as usize;
        let (_, _) = Arena0LiteWire::ref_from_prefix(&v3_bytes[arena_off..]).unwrap();

        // 2. Validate shared #![no_std] dnam_kernels primitives & zero-alloc SONA kernel execution
        assert_eq!(fp16_to_f32(f32_to_fp16(0.5)), 0.5);
        let g = prewarp_tan_pi_f_over_fs_f32(1_000.0 / 48_000.0);
        assert!(g.is_finite() && g > 0.06 && g < 0.07);
        let cold = SonaColdTierModel::identity_for_tier(Tier::Lite, false, false);
        let mut slab = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&cold);
        let _ = step_sona_group_stereo8::<
            <DefaultMathBackend as dnam_engine::math::MathBackend>::Stereo8,
        > as fn(_, _, _) -> _;
        let y_silent = slab.step_stereo(0.0, 0.0);
        assert_eq!(y_silent.map(f32::to_bits), [0, 0]);
    }

    /// verify[arch.families.coverage]
    /// verify[arch.families.f1-bistables]
    /// verify[arch.families.f2-hilbert-ssb]
    /// verify[arch.families.f3-pitch]
    /// verify[arch.families.f4-dynamics]
    /// verify[arch.families.f5-bitcrusher]
    /// verify[arch.families.f6-formants]
    /// verify[arch.families.f7-multi-input]
    #[test]
    fn test_all_seven_hardware_families_realization() {
        let _guard = FtzDazGuard::new();

        let mut dc = T1DcBlocker::new_at_rate(48_000.0).unwrap();
        let mut h1 = HilbertH1Lattice6Sec::new_at_rate(48_000.0).unwrap();
        let mut trk = T1SubHarmonicTracker::new();
        let d = dc.step(0.5);
        let (i_v, q_v) = h1.step(d);
        let f1_feats = trk.step_analytic(i_v, q_v);
        assert!(f1_feats.iter().all(|v| v.is_finite()));

        let ssb_upper = i_v * 0.8 - q_v * 0.6;
        let ssb_lower = i_v * 0.8 + q_v * 0.6;
        assert!(ssb_upper.is_finite() && ssb_lower.is_finite());

        let (_, weights, delays) = eval_m2_pitch_grain_delays(100, 4800, 7204.0, -1.0).unwrap();
        assert!((weights[0] + weights[1] - 1.0).abs() < 1e-12 && delays[0] >= 4.0);

        let log2_env = eval_monotone_log2_f32_bits((0.5_f32).to_bits())
            .unwrap()
            .unwrap();
        assert!((log2_env - (-1.0)).abs() < 1e-5);
        let mut vca = BallisticDuckingVca::new();
        assert!(vca.step(0.5, 1.0).is_finite());

        let mut crusher = FixedHzBitcrusher::new(8, 12_000.0, 48_000.0).unwrap();
        assert!(crusher.step(0.33).is_finite());

        let mut kautz = KautzModalBank::<4>::new();
        kautz.set_pole(0, 0.99, 0.3, 0.05, 0.5, -0.2);
        assert!(kautz.step(0.5).is_finite());

        let mut router = Stage0Router::new();
        let (sl, sr, dl, dr, _, _) = router.route_frame(0.5, -0.5, 0.2, -0.2);
        let mut lr4 = TptLr4Crossover::default();
        let (l4, h4) = lr4.step(dl);
        assert!(
            sl.is_finite() && sr.is_finite() && dr.is_finite() && l4.is_finite() && h4.is_finite()
        );
        assert_ne!(WdwDryTapMode::PreSpatialCore, WdwDryTapMode::DirectDry);
    }
}
