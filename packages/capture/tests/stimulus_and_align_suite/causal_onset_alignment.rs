#![forbid(unsafe_code)]
//! Fast integration checks verifying strictly causal onset lag preservation across physical
//! black-box fixtures (`Chain01SymmetricOd`, `Chain05PureLtiCabAndRoom`) and delayed captures.

// verify[train.capture.alignment.multi-regime]
// verify[train.capture.invariants]

use std::collections::BTreeSet;

use dnam_capture::align::{AlignmentRegime, estimate_hybrid_multiregime_alignment};
use dnam_capture::solver::{
    build_v3_causal_probe_excitation, estimate_v3_causal_deadtime_and_polarity,
};
use dnam_capture::stimulus::TopologyProbeMode;
use dnam_capture::stimulus::pipeline::schoukens::build_schoukens_pintelon_coreset_48k;
use dnam_capture::stimulus::waveforms::deterministic_seed_256;
use dnam_fixtures::{BlackBoxModel, Chain01SymmetricOd, Chain05PureLtiCabAndRoom};

fn render_mono_fixture<M: BlackBoxModel>(model: &mut M, dry: &[f32], controls: &[f32]) -> Vec<f32> {
    model.reset();
    let n_out = model.num_outputs().clamp(1, 2);
    let mut out = Vec::with_capacity(dry.len());
    let mut buf = [0.0_f32; 2];
    for &x in dry {
        model.process_frame(&[x], &mut buf[..n_out], controls, &[]);
        out.push(buf[0]);
    }
    out
}

#[test]
fn test_causal_onset_alignment_blocks_acausal_advance_on_chain01_and_chain05() {
    let probe = build_v3_causal_probe_excitation();
    let probe_l: Vec<f32> = probe.iter().map(|lr| lr[0]).collect();

    let seed = deterministic_seed_256(0x2501_A001);
    let exc = build_schoukens_pintelon_coreset_48k(
        &seed,
        TopologyProbeMode::Mono1To1,
        48_000,
        &BTreeSet::new(),
    );
    let dry = &exc.dry_l;

    // 1. Chain01SymmetricOd at nominal 0-delay controls=[0, 0, 0]: tau_coarse == 0, lag == 0.
    let mut c1 = Chain01SymmetricOd::new();
    let ctrls_c1 = [0.0_f32, 0.0, 0.0];
    let probe_wet_c1 = render_mono_fixture(&mut c1, &probe_l, &ctrls_c1);
    let (tau_c1, inv_c1_probe) = estimate_v3_causal_deadtime_and_polarity(&probe_l, &probe_wet_c1);
    assert_eq!(tau_c1, 0.0);
    assert!(!inv_c1_probe);

    let wet_c1 = render_mono_fixture(&mut c1, dry, &ctrls_c1);
    let est_c1 =
        estimate_hybrid_multiregime_alignment(dry, &wet_c1, 48_000, 128, Some(tau_c1), false)
            .expect("Chain01 alignment");
    assert_eq!(est_c1.total_lag_samples, 0.0);
    assert!(!est_c1.polarity_inverted);
    assert_eq!(est_c1.clock_drift_ppm, 0.0);

    // 2. Chain01SymmetricOd with exact +24-sample pure converter deadtime: stays <= 24.0 (never 25.5).
    let mut probe_del24 = vec![0.0_f32; probe_l.len()];
    probe_del24[24..].copy_from_slice(&probe_wet_c1[..(probe_l.len() - 24)]);
    let (tau_del24, _) = estimate_v3_causal_deadtime_and_polarity(&probe_l, &probe_del24);
    assert_eq!(tau_del24, 24.0);

    let mut wet_del24 = vec![0.0_f32; dry.len()];
    wet_del24[24..].copy_from_slice(&wet_c1[..(dry.len() - 24)]);
    let est_del24 =
        estimate_hybrid_multiregime_alignment(dry, &wet_del24, 48_000, 128, Some(tau_del24), false)
            .expect("Chain01 +24-sample delayed alignment");
    assert!(
        (est_del24.total_lag_samples - 24.0).abs() < 1.0e-6,
        "expected causal 24.0 lag, got {}",
        est_del24.total_lag_samples
    );
    assert_ne!(est_del24.regime, AlignmentRegime::DispersiveAllPass);

    // 3. Chain05PureLtiCabAndRoom (LTI cabinet + room reflections): tau_coarse == 0, lag == 0.
    let mut c5 = Chain05PureLtiCabAndRoom::new();
    let probe_wet_c5 = render_mono_fixture(&mut c5, &probe_l, &[]);
    let (tau_c5, inv_c5_probe) = estimate_v3_causal_deadtime_and_polarity(&probe_l, &probe_wet_c5);
    assert_eq!(tau_c5, 0.0);
    assert!(!inv_c5_probe);

    let wet_c5 = render_mono_fixture(&mut c5, dry, &[]);
    let est_c5 =
        estimate_hybrid_multiregime_alignment(dry, &wet_c5, 48_000, 128, Some(tau_c5), false)
            .expect("Chain05 alignment");
    assert_eq!(est_c5.total_lag_samples, 0.0);
    assert!(!est_c5.polarity_inverted);
}
