#![forbid(unsafe_code)]
//! `Case #19` parallelized routing topology certification (`1->1`, `2->1`, `1->2`, `2->2`, `4CM`, `WDW`).

use dnam_capture::DistillationTarget;
use dnam_capture::proj::DnamProjManifest;
use dnam_capture::stimulus::{AdaptiveTrainingOptions, TopologyProbeMode, train_adaptive_target};
use dnam_engine::{DnamAudioIo, DnamEngine, Tier};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain05PureLtiCabAndRoom, Chain07StereoChorusAndRotary,
    Chain09La2aOpticalCompressor, FixtureBitDepth, FixtureDeviceMetadata,
};
use dnam_vm::math::DefaultMathBackend;

use super::adversaries_and_topologies::BlackBoxDistillationAdapter;

#[derive(Clone, Debug)]
struct FourCableAndWdwFixture {
    n_in: usize,
    n_out: usize,
}

const ROUTER_META: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Router-4x4-Rig",
    name: "4CM & WDW Multiport Rig",
    category: "routing",
    sample_rate_hz: 48_000,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 15.0,
    supports_tempo_sync: false,
    controls: &[],
    switches: &[],
};

impl BlackBoxModel for FourCableAndWdwFixture {
    fn metadata(&self) -> FixtureDeviceMetadata {
        ROUTER_META
    }
    fn num_inputs(&self) -> usize {
        self.n_in
    }
    fn num_outputs(&self) -> usize {
        self.n_out
    }
    fn reset(&mut self) {}
    fn process_frame(&mut self, inputs: &[f32], outputs: &mut [f32], _: &[f32], _: &[u8]) {
        let i0 = inputs.first().copied().unwrap_or(0.0);
        let i1 = inputs.get(1).copied().unwrap_or(i0);
        let r0 = inputs.get(2).copied().unwrap_or(0.0);
        let r1 = inputs.get(3).copied().unwrap_or(r0);
        let send_l = (1.6 * i0).tanh();
        let send_r = (1.5 * i1).tanh();
        if let Some(o0) = outputs.get_mut(0) {
            *o0 = 0.7 * (send_l + 0.6 * r0);
        }
        if let Some(o1) = outputs.get_mut(1) {
            *o1 = 0.7 * (send_r + 0.6 * r1);
        }
        if let Some(o2) = outputs.get_mut(2) {
            *o2 = send_l;
        }
        if let Some(o3) = outputs.get_mut(3) {
            *o3 = send_r;
        }
    }
}

fn distill_topology_bytes(
    topo: TopologyProbeMode,
    model_box: &mut dyn DistillationTarget,
    n_in: u8,
    n_out: u8,
) -> Vec<u8> {
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.io_config.inputs = n_in;
    manifest.base.io_config.output_mode = if n_out >= 2 {
        "stereo".to_string()
    } else {
        "mono".to_string()
    };
    train_adaptive_target(
        manifest,
        model_box,
        &AdaptiveTrainingOptions {
            topology_mode: topo,
            fixed_duration: true,
            prefer_mode_b: true,
            max_frames_per_stem: 1,
            run_live_null_test: false,
            ..AdaptiveTrainingOptions::default()
        },
    )
    .unwrap()
    .runtime_dnam_bytes
}

fn topology_test_signals_256() -> (Vec<f32>, Vec<f32>) {
    let sig_a: Vec<f32> = (0..256)
        .map(|n| 0.3 * (std::f32::consts::TAU * 220.0 * (n as f32) / 48_000.0).sin())
        .collect();
    let sig_b: Vec<f32> = (0..256)
        .map(|n| 0.25 * (std::f32::consts::TAU * 550.0 * (n as f32) / 48_000.0).cos())
        .collect();
    (sig_a, sig_b)
}

#[test]
fn test_case_19_routing_topology_1to1() {
    let (sig_a, _) = topology_test_signals_256();
    let mut m_1to1 = BlackBoxDistillationAdapter::from_model(Chain01SymmetricOd::new());
    let dnam_1to1 = distill_topology_bytes(TopologyProbeMode::Mono1To1, &mut m_1to1, 1, 1);
    let mut eng_1to1 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_1to1, Tier::Ultra, 48_000)
            .unwrap();
    let mut o1 = vec![0.0_f32; 256];
    eng_1to1.process_block_io(DnamAudioIo {
        in_1: &sig_a,
        in_2: None,
        in_3: None,
        in_4: None,
        out_1: &mut o1,
        out_2: None,
        out_3: None,
        out_4: None,
    });
    assert!(o1.iter().all(|v| v.is_finite()) && o1.iter().any(|v| v.abs() > 1.0e-4));
}

#[test]
fn test_case_19_routing_topology_2to1() {
    let (sig_a, sig_b) = topology_test_signals_256();
    let mut m_2to1 = BlackBoxDistillationAdapter::from_model(Chain09La2aOpticalCompressor::new());
    let dnam_2to1 = distill_topology_bytes(TopologyProbeMode::DualInToMono2To1, &mut m_2to1, 2, 1);
    let mut eng_2to1 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_2to1, Tier::Ultra, 48_000)
            .unwrap();
    let mut o2_l = vec![0.0_f32; 256];
    let mut o2_r = vec![0.0_f32; 256];
    eng_2to1.process_block_sidechain(&sig_a, &sig_a, &sig_b, &sig_b, &mut o2_l, &mut o2_r);
    assert!(o2_l.iter().all(|v| v.is_finite()));
}

#[test]
fn test_case_19_routing_topology_1to2() {
    let (sig_a, _) = topology_test_signals_256();
    let mut m_1to2 = BlackBoxDistillationAdapter::from_model(Chain05PureLtiCabAndRoom::new());
    let dnam_1to2 = distill_topology_bytes(TopologyProbeMode::MonoToStereo1To2, &mut m_1to2, 1, 2);
    let mut eng_1to2 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_1to2, Tier::Ultra, 48_000)
            .unwrap();
    let mut o2_l = vec![0.0_f32; 256];
    let mut o2_r = vec![0.0_f32; 256];
    eng_1to2.process_block(&sig_a, &sig_a, &mut o2_l, &mut o2_r);
    assert!(o2_l.iter().chain(&o2_r).all(|v| v.is_finite()));
}

#[test]
fn test_case_19_routing_topology_2to2() {
    let (sig_a, sig_b) = topology_test_signals_256();
    let mut m_2to2 = BlackBoxDistillationAdapter::from_model(Chain07StereoChorusAndRotary::new());
    let dnam_2to2 = distill_topology_bytes(TopologyProbeMode::TrueStereo2To2, &mut m_2to2, 2, 2);
    let mut eng_2to2 =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_2to2, Tier::Ultra, 48_000)
            .unwrap();
    let mut o2_l = vec![0.0_f32; 256];
    let mut o2_r = vec![0.0_f32; 256];
    eng_2to2.process_block(&sig_a, &sig_b, &mut o2_l, &mut o2_r);
    assert!(o2_l.iter().chain(&o2_r).all(|v| v.is_finite()));
}

#[test]
fn test_case_19_routing_topology_4cm() {
    let (sig_a, sig_b) = topology_test_signals_256();
    // 4CM (4-Cable Method): verify strict S_{ret -> send} == 0 (0.000000 ULP causal DAG isolation!)
    let mut m_4cm =
        BlackBoxDistillationAdapter::from_model(FourCableAndWdwFixture { n_in: 4, n_out: 4 });
    let dnam_4cm = distill_topology_bytes(TopologyProbeMode::FourCableMethod4Cm, &mut m_4cm, 4, 4);
    let zeros = vec![0.0_f32; 256];
    let mut eng_4cm_a =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_4cm, Tier::Ultra, 48_000).unwrap();
    let mut eng_4cm_b =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_4cm, Tier::Ultra, 48_000).unwrap();
    let (mut ml_a, mut mr_a, mut sl_a, mut sr_a) = (
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
    );
    let (mut ml_b, mut mr_b, mut sl_b, mut sr_b) = (
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
        vec![0.0_f32; 256],
    );
    eng_4cm_a.process_block_4cm(
        &sig_a, &sig_a, &zeros, &zeros, &mut ml_a, &mut mr_a, &mut sl_a, &mut sr_a,
    );
    eng_4cm_b.process_block_4cm(
        &sig_a, &sig_a, &sig_b, &sig_b, &mut ml_b, &mut mr_b, &mut sl_b, &mut sr_b,
    );
    for i in 0..256 {
        assert_eq!(
            sl_a[i].to_bits(),
            sl_b[i].to_bits(),
            "4CM S_{{ret->send}} L feedback leak at frame {i}"
        );
        assert_eq!(
            sr_a[i].to_bits(),
            sr_b[i].to_bits(),
            "4CM S_{{ret->send}} R feedback leak at frame {i}"
        );
    }
}

#[test]
fn test_case_19_routing_topology_wdw() {
    let (sig_a, _) = topology_test_signals_256();
    let mut m_wdw =
        BlackBoxDistillationAdapter::from_model(FourCableAndWdwFixture { n_in: 2, n_out: 3 });
    let dnam_wdw = distill_topology_bytes(TopologyProbeMode::WetDryWet1To3, &mut m_wdw, 2, 3);
    let mut eng_wdw =
        DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_wdw, Tier::Ultra, 48_000).unwrap();
    let (mut wl, mut wr, mut dc) = (vec![0.0_f32; 256], vec![0.0_f32; 256], vec![0.0_f32; 256]);
    eng_wdw.process_block_wdw(&sig_a, &sig_a, &mut wl, &mut wr, &mut dc);
    assert!(wl.iter().chain(&wr).chain(&dc).all(|v| v.is_finite()));
    assert!(
        dc.iter().any(|v| v.abs() > 1.0e-4),
        "WDW dry_center must carry signal"
    );
}
