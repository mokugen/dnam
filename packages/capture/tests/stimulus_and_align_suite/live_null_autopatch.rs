#![forbid(unsafe_code)]
//! `1.0s` Live Null-Test stereo `out_2` sensitivity, `1.0 * STAT` preservation, `96 kHz` rig
//! sample-rate replay, and non-stationary rollback guards (`step2_2_adversarial_edges` split).
// verify[train.capture.invariants]
// verify[train.capture.alignment.multi-regime]
// verify[train.cache.proj-vault]

use std::collections::BTreeMap;

use dnam_capture::proj::{DnamProjManifest, DnamProjView, ProjChunkKind};
use dnam_capture::stimulus::{
    AdaptiveTrainingOptions, BatchStimulusOptions, TopologyProbeMode, generate_stimulus_batch,
    run_live_null_test_and_autopatch, solve_from_responses, train_adaptive_target,
};
use dnam_capture::{DistillationTarget, StemResponse, StemStimulus};
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::Chain01SymmetricOd;
use dnam_vm::math::DefaultMathBackend;

fn mono_resp(stem_id: impl Into<String>, sample_rate_hz: u32, out_1: Vec<f32>) -> StemResponse {
    StemResponse {
        stem_id: stem_id.into(),
        sample_rate_hz,
        out_1,
        out_2: None,
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    }
}

fn stereo_resp(
    stem_id: impl Into<String>,
    sample_rate_hz: u32,
    out_1: Vec<f32>,
    out_2: Vec<f32>,
) -> StemResponse {
    StemResponse {
        stem_id: stem_id.into(),
        sample_rate_hz,
        out_1,
        out_2: Some(out_2),
        out_3: None,
        out_4: None,
        internal_nodes: BTreeMap::new(),
    }
}

struct StereoRightMismatchTarget {
    ref_dnam: Vec<u8>,
    seen_sample_rates: Vec<u32>,
    nonstationary_flip: bool,
    call_count: usize,
}

impl DistillationTarget for StereoRightMismatchTarget {
    fn reset_state(&mut self) {}

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        self.seen_sample_rates.push(stim.sample_rate_hz);
        self.call_count += 1;
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&self.ref_dnam, Tier::Ultra, 48_000)
                .unwrap();
        let n = stim.in_1.len();
        let (mut out_1, mut out_2) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            // Map 96 kHz stimulus index back to 48 kHz step rate cleanly when testing rig rate
            let il = stim.in_1[i];
            let ir = stim.in_2.as_ref().map_or(il, |r| r[i]);
            let [ml, mr] = eng.step_frame_sona(il, ir, false);
            if self.nonstationary_flip {
                // Orthogonal 12 kHz carrier unrelated to input feature bus so Cholesky patch degrades ESR!
                let ortho = 0.45 * if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
                out_1.push(ml + ortho);
                out_2.push(mr - ortho);
            } else {
                // Left channel matches model 100% (`dl == 0`), ONLY Right channel (`out_2`) has 2.2x gain drift!
                out_1.push(ml);
                out_2.push(2.2 * mr);
            }
        }
        Ok(stereo_resp(
            &stim.stem_id,
            stim.sample_rate_hz,
            out_1,
            out_2,
        ))
    }
}

#[test]
fn test_live_null_test_stereo_out2_sensitivity_stat_preservation_and_nonstationary_rollback() {
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.inputs = 1;
    manifest.base.io_config.output_mode = "stereo".to_string();
    manifest.base.controls_schema.clear();

    let batch = generate_stimulus_batch(
        &manifest,
        &BatchStimulusOptions {
            topology_mode: TopologyProbeMode::MonoToStereo1To2,
            ..BatchStimulusOptions::default()
        },
    )
    .unwrap();
    let resps: Vec<StemResponse> = batch
        .stems
        .iter()
        .map(|s| {
            let half: Vec<f32> = s.in_1.iter().map(|&x| 0.5 * x).collect();
            stereo_resp(&s.stem_id, 48_000, half.clone(), half)
        })
        .collect();
    let solved = solve_from_responses(manifest, &[(&batch, &resps)], None).unwrap();

    let crypto = DefaultCryptoProvider::default();
    let view_before = DnamProjView::from_bytes(&solved.proj_bytes, &crypto).unwrap();
    let trace_stat = |buf: &[u8]| -> f64 {
        (0..256)
            .map(|d| {
                let off = ((d * (d + 1)) / 2 + d) * 8;
                f64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
            })
            .sum()
    };
    let tr_before = trace_stat(view_before.chunk_slice(ProjChunkKind::Stat, 0).unwrap());

    // 1. Right-channel-only error MUST be detected (`pre_esr_db > -15 dB`), patched, and preserve >= 1.0 * STAT!
    let mut right_target = StereoRightMismatchTarget {
        ref_dnam: solved.runtime_dnam_bytes.clone(),
        seen_sample_rates: Vec::new(),
        nonstationary_flip: false,
        call_count: 0,
    };
    let (patched_proj, _, rep) =
        run_live_null_test_and_autopatch(&solved.proj_bytes, &mut right_target, -30.0)
            .expect("stereo out_2 null test");
    assert!(
        rep.pre_patch_null_esr_db > -15.0,
        "stereo out_2 mismatch must be reflected in pre_patch_null_esr_db (got {} dB)",
        rep.pre_patch_null_esr_db
    );
    assert!(
        rep.autopatch_applied,
        "expected autopatch applied, got {rep:?}"
    );
    assert!(rep.post_patch_null_esr_db < rep.pre_patch_null_esr_db - 1.5);

    let view_after = DnamProjView::from_bytes(&patched_proj, &crypto).unwrap();
    let tr_after = trace_stat(view_after.chunk_slice(ProjChunkKind::Stat, 0).unwrap());
    assert!(
        tr_before > 1.0 && tr_after > tr_before,
        "STAT must preserve 1.0 * tr(G_0) + gamma * tr(dG) without 0.25x downscaling ({tr_before} -> {tr_after})"
    );

    // 2. Non-stationary / uncorrelated interference guard: if Cholesky patch degrades ESR, rollback!
    let mut nonstat_target = StereoRightMismatchTarget {
        ref_dnam: solved.runtime_dnam_bytes.clone(),
        seen_sample_rates: Vec::new(),
        nonstationary_flip: true,
        call_count: 0,
    };
    let (rb_proj, _, rb_rep) =
        run_live_null_test_and_autopatch(&solved.proj_bytes, &mut nonstat_target, -60.0)
            .expect("nonstationary null test");
    if !rb_rep.autopatch_applied {
        assert_eq!(rb_proj, solved.proj_bytes);
    } else {
        assert!(rb_rep.post_patch_null_esr_db <= rb_rep.pre_patch_null_esr_db + 0.05);
    }

    // 3. Rig device sample rate (`96_000 Hz`) honored by run_live_null_test_and_autopatch
    let mut m96 = DnamProjManifest::default_reference();
    m96.rig_routing.sample_rate_device_hz = 96_000;
    m96.base.controls_schema.clear();
    let b96 = generate_stimulus_batch(
        &m96,
        &BatchStimulusOptions {
            sample_rate_hz: 96_000,
            ..BatchStimulusOptions::default()
        },
    )
    .unwrap();
    let r96 = vec![mono_resp(
        &b96.stems[0].stem_id,
        96_000,
        b96.stems[0].in_1.iter().map(|&x| 0.4 * x).collect(),
    )];
    let s96 = solve_from_responses(m96, &[(&b96, &r96)], None).unwrap();
    let mut t96 = StereoRightMismatchTarget {
        ref_dnam: s96.runtime_dnam_bytes,
        seen_sample_rates: Vec::new(),
        nonstationary_flip: false,
        call_count: 0,
    };
    let _ = run_live_null_test_and_autopatch(&s96.proj_bytes, &mut t96, 10.0).unwrap();
    assert_eq!(t96.seen_sample_rates, vec![96_000]);
}

#[test]
fn test_mode1_streaming_prefer_mode_b_stationary_stops() {
    let mut manifest_b = DnamProjManifest::default_reference();
    manifest_b.base.controls_schema = vec![super::dual_mode_and_align::make_range_control(
        "gain", 0.0, 1.0, 0.5,
    )];
    let mut target =
        super::dual_mode_and_align::SimpleFixtureTarget::new(Chain01SymmetricOd::new());
    let rep_b = train_adaptive_target(
        manifest_b,
        &mut target,
        &AdaptiveTrainingOptions {
            prefer_mode_b: true,
            max_frames_per_stem: 2,
            run_live_null_test: false,
            null_test_autopatch_threshold_db: -60.0,
            calb: Some(super::dual_mode_and_align::sample_calb()),
            ..AdaptiveTrainingOptions::default()
        },
    )
    .expect("Mode 1 prefer_mode_b");
    assert_eq!(rep_b.stem_count, 3); // StarCenter + 2 stationary stops
}
