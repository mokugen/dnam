#![forbid(unsafe_code)]
//! End-to-end phased integration suite for [`dnam_train::solve_sona_model`] covering:
//! - Phase A (isolated operators): True-stereo Linear/FIR+Modal and Nonlinear+`Op-T` comb
//!   components across Level 1 (fixed-setting trained wet reproduction) and Level 2 (fixed-setting
//!   untrained wet).
//! - Phase B (coupled multi-component + control plane): `CP1` overflow (`> 8` slots), `"CEXT"`,
//!   `"TSYN"`, Phase-1..3 clamping & `+6 dB` idempotence, exact 0-ULP `StarCenter` invariance,
//!   Level-3 analytical ground truth at omitted intermediate control points, audio-driven Pass-4
//!   Wiener + 32-slot Jacobian extraction, and CPU vs `wgpu` parity.

use dnam_train::control::clamp::{
    ControlKindSpec, ControlSpec, Phase3AuditInput, StageStabilityConstraint,
    audit_phase3_dynamic_and_head_gain, evaluate_readout_spectral_magnitudes,
};
use dnam_train::control::cp1::{Cp1ControlSample, build_tsyn_entry_wire};
use dnam_train::{
    AtomicDnprProgress, ControlAudioStem, MAX_INTER_TIER_LU_DRIFT, PROGRESS_MAGIC_DNPR,
    SonaSolverConfig, SonaTrainDataset, TrainRunner, TrainingPhase, TrainingStatus,
    solve_sona_model,
};
use dnam_vm::bin::{CP1_STREAM_MAGIC, DnamBinView, strip_dnam_bin_to_tier};
use dnam_vm::control::{ControlBus, STAGE_TARGET_SLOTS};
use dnam_vm::{DefaultMathBackend, SonaVm, Tier};

fn gen_broadband(len: usize, mut seed: u64, amp: f64, tone_freq: f64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = 2.0 * (((seed >> 33) as f64) / ((1_u64 << 31) as f64)) - 1.0;
            let tone = 0.25 * ((i as f64) * tone_freq).sin();
            amp * (0.75 * u + tone)
        })
        .collect()
}

fn eval_linear_modal_channel(
    x: &[f64],
    cross: &[f64],
    r: f64,
    theta: f64,
    taps: [(usize, f64); 3],
    cross_mix: f64,
) -> Vec<f64> {
    let mut y = vec![0.0_f64; x.len()];
    let a1 = -2.0 * r * theta.cos();
    let a2 = r * r;
    let (mut m1, mut m2) = (0.0_f64, 0.0_f64);
    for t in 0..x.len() {
        let mode = 0.14 * x[t] - a1 * m1 - a2 * m2;
        m2 = m1;
        m1 = mode;
        let fir: f64 = taps
            .iter()
            .map(|&(d, g)| if t >= d { g * x[t - d] } else { 0.0 })
            .sum();
        y[t] = fir + mode + cross_mix * cross[t];
    }
    y
}

fn eval_linear_modal_operator_stereo(in_l: &[f64], in_r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let y_l = eval_linear_modal_channel(
        in_l,
        in_r,
        0.88,
        0.45,
        [(0, 0.72), (6, 0.25), (18, 0.18)],
        0.04,
    );
    let y_r = eval_linear_modal_channel(
        in_r,
        in_l,
        0.83,
        0.62,
        [(0, 0.66), (9, 0.21), (22, -0.15)],
        0.06,
    );
    (y_l, y_r)
}

fn eval_nonlinear_comb_channel(
    x: &[f64],
    drive: f64,
    delay: usize,
    fb_gain: f64,
    dry_mix: f64,
    wet_mix: f64,
) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0_f64; n];
    let mut comb = vec![0.0_f64; n];
    for t in 0..n {
        let sat = (drive * x[t]).tanh();
        let tap = if t >= delay { comb[t - delay] } else { 0.0 };
        comb[t] = sat + fb_gain * tap;
        y[t] = dry_mix * sat + wet_mix * tap;
    }
    y
}

fn eval_nonlinear_comb_operator_stereo(in_l: &[f64], in_r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Delays >= 160 samples sit cleanly outside the 128-tap early FIR so Op-T owns them.
    let y_l = eval_nonlinear_comb_channel(in_l, 1.6, 160, 0.50, 0.68, 0.35);
    let y_r = eval_nonlinear_comb_channel(in_r, 1.35, 176, 0.44, 0.62, 0.32);
    (y_l, y_r)
}

fn eval_coupled_stereo(in_l: &[f64], in_r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (lin_l, lin_r) = eval_linear_modal_operator_stereo(in_l, in_r);
    let (nl_l, nl_r) = eval_nonlinear_comb_operator_stereo(in_l, in_r);
    let y_l = lin_l
        .iter()
        .zip(&nl_l)
        .map(|(&a, &b)| 0.65 * a + 0.35 * b)
        .collect();
    let y_r = lin_r
        .iter()
        .zip(&nl_r)
        .map(|(&a, &b)| 0.65 * a + 0.35 * b)
        .collect();
    (y_l, y_r)
}

fn normalized_mse(pred: &[f64], target: &[f64], warmup: usize) -> f64 {
    let n = pred.len().min(target.len());
    let mut err = 0.0_f64;
    let mut ref_e = 0.0_f64;
    for t in warmup..n {
        let d = pred[t] - target[t];
        err += d * d;
        ref_e += target[t] * target[t];
    }
    err / ref_e.max(1.0e-12)
}

fn run_vm_block(
    vm: &mut SonaVm<'_, DefaultMathBackend>,
    in_l: &[f64],
    in_r: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let in_l32: Vec<f32> = in_l.iter().map(|&v| v as f32).collect();
    let in_r32: Vec<f32> = in_r.iter().map(|&v| v as f32).collect();
    let mut out_l = vec![0.0_f32; in_l32.len()];
    let mut out_r = vec![0.0_f32; in_r32.len()];
    vm.process_block(&in_l32, &in_r32, &mut out_l, &mut out_r)
        .expect("process_block");
    (
        out_l.into_iter().map(f64::from).collect(),
        out_r.into_iter().map(f64::from).collect(),
    )
}

fn render_vm_tier(bin: &[u8], tier: Tier, in_l: &[f64], in_r: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let sliced = strip_dnam_bin_to_tier(bin, tier).expect("strip tier");
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(sliced, tier, 48_000.0)
        .expect("open SonaVm");
    run_vm_block(&mut vm, in_l, in_r)
}

fn render_vm_with_controls(
    bin: &[u8],
    tier: Tier,
    controls: &[(usize, f32)],
    in_l: &[f64],
    in_r: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let sliced = strip_dnam_bin_to_tier(bin, tier).expect("strip tier");
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(sliced, tier, 48_000.0)
        .expect("open SonaVm");
    vm.set_controls_batch(controls).expect("set_controls_batch");
    if let Some(bus) = vm.slab_mut().cold_mut().ctrl_bus.as_mut() {
        bus.eval_static_and_snap_if_dirty();
        let params = *bus.current_stage_params();
        vm.slab_mut().cold_mut().latched_stage_params = params;
    }
    vm.step_control_tick_3khz();
    run_vm_block(&mut vm, in_l, in_r)
}

fn render_vm_pinned_stage_targets(
    bin: &[u8],
    tier: Tier,
    st32: &[f64; STAGE_TARGET_SLOTS],
    in_l: &[f64],
    in_r: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let sliced = strip_dnam_bin_to_tier(bin, tier).expect("strip tier");
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(sliced, tier, 48_000.0)
        .expect("open SonaVm");
    let mut params = [0.0_f32; STAGE_TARGET_SLOTS];
    for (dst, &src) in params.iter_mut().zip(st32) {
        *dst = src as f32;
    }
    {
        let cold = vm.slab_mut().cold_mut();
        cold.ctrl_bus = None;
        cold.latched_stage_params = params;
    }
    vm.step_control_tick_3khz();
    run_vm_block(&mut vm, in_l, in_r)
}

fn eval_test3_analytical_targets(
    c0: f32,
    c2: f32,
) -> ([f64; STAGE_TARGET_SLOTS], [f64; STAGE_TARGET_SLOTS]) {
    let norm0 = f64::from((c0 - 5.0) / 3.0);
    let norm2 = f64::from((c2 - 5.0) / 5.0);
    let mut st = [0.0_f64; STAGE_TARGET_SLOTS];
    let mut sud = [0.0_f64; STAGE_TARGET_SLOTS];
    st[0] = 0.25 * norm2 + 0.15 * norm0;
    st[2] = 0.30 * norm0;
    st[15] = 0.35 * norm0;
    st[26] = 0.45 * norm0;
    st[30] = 0.20 * norm2;
    sud[0] = 0.08 * norm0;
    (st, sud)
}

#[test]
fn test_phase_a_isolated_linear_modal_level1_and_level2_cpu_and_wgpu() {
    let sr = 48_000.0_f64;
    let warmup = 64_usize;
    let in_l = gen_broadband(1_024, 0xA11C_E501_9876_5432, 0.35, 0.037);
    let in_r = gen_broadband(1_024, 0xA11C_E502_1357_2468, 0.31, 0.053);
    let (tgt_l, tgt_r) = eval_linear_modal_operator_stereo(&in_l, &in_r);

    let dataset = SonaTrainDataset {
        sample_rate_hz: sr,
        warmup_frames: warmup,
        input_l: in_l.clone(),
        input_r: in_r.clone(),
        target_l: tgt_l.clone(),
        target_r: tgt_r.clone(),
        controls: Vec::new(),
        stability_constraint: StageStabilityConstraint::default(),
        control_samples: Vec::new(),
        control_audio_stems: Vec::new(),
        tsyn_entries: Vec::new(),
    };
    let mut cfg = SonaSolverConfig::default();
    cfg.pass2.max_epochs = 4;
    cfg.pass2.diminishing_patience = 2;

    let prog = AtomicDnprProgress::new();
    let res_cpu = solve_sona_model(&TrainRunner::cpu(), &dataset, &cfg, Some(&prog)).unwrap();

    // Level 1: fixed-setting true-stereo trained reproduction, 4-tier LUFS <= 0.05 LU & DNPR check.
    let snap = prog.snapshot();
    assert_eq!(snap.magic.get(), PROGRESS_MAGIC_DNPR);
    assert_eq!(snap.status(), TrainingStatus::Completed);
    assert_eq!(snap.phase(), TrainingPhase::Completed);
    assert_eq!(snap.progress_permille(), 1_000);
    assert!(DnamBinView::parse(&res_cpu.pass5.dnam_bin).is_ok());
    assert!(
        res_cpu.pass5.adjacent_tier_monotone,
        "tier_reports={:#?}\nos2x={} pfx={:?} trim={:?}",
        res_cpu.pass5.tier_reports,
        res_cpu.model.oversample_upper_tiers_2x,
        res_cpu.model.prefix_override_needed,
        res_cpu.pass5.tier_gain_trim,
    );
    let ultra_lufs = res_cpu.pass5.tier_lufs[3];
    for &tl in &res_cpu.pass5.tier_lufs {
        assert!(
            (tl - ultra_lufs).abs() <= MAX_INTER_TIER_LU_DRIFT,
            "tier LUFS drift {tl} vs {ultra_lufs}"
        );
    }
    let l1_u_l = normalized_mse(&res_cpu.pass5.vm_renders[3].0, &tgt_l, warmup);
    let l1_u_r = normalized_mse(&res_cpu.pass5.vm_renders[3].1, &tgt_r, warmup);
    assert!(
        l1_u_l < 0.05 && l1_u_r < 0.05,
        "Level-1 L={l1_u_l} R={l1_u_r}"
    );

    // Level 2: fixed-setting unseen stereo dry excitation without retraining.
    let held_l = gen_broadband(1_024, 0xB22D_F612_7654_3210, 0.42, 0.041);
    let held_r = gen_broadband(1_024, 0xB22D_F613_2468_1357, 0.37, 0.067);
    let (held_tgt_l, held_tgt_r) = eval_linear_modal_operator_stereo(&held_l, &held_r);
    let (gen_ul, gen_ur) = render_vm_tier(&res_cpu.pass5.dnam_bin, Tier::Ultra, &held_l, &held_r);
    let (gen_ll, gen_lr) = render_vm_tier(&res_cpu.pass5.dnam_bin, Tier::Lite, &held_l, &held_r);
    let l2_ul = normalized_mse(&gen_ul, &held_tgt_l, warmup);
    let l2_ur = normalized_mse(&gen_ur, &held_tgt_r, warmup);
    let l2_ll = normalized_mse(&gen_ll, &held_tgt_l, warmup);
    let l2_lr = normalized_mse(&gen_lr, &held_tgt_r, warmup);
    assert!(
        l2_ul < 0.05 && l2_ur < 0.05,
        "Level-2 Ultra L={l2_ul} R={l2_ur}"
    );
    assert!(
        l2_ll < 0.12 && l2_lr < 0.12,
        "Level-2 Lite L={l2_ll} R={l2_lr}"
    );

    // Verify auto_detect() (wgpu Metal on macOS or CPU fallback) matches CPU on both channels.
    let res_auto = solve_sona_model(&TrainRunner::auto_detect(), &dataset, &cfg, None).unwrap();
    let (auto_ul, auto_ur) =
        render_vm_tier(&res_auto.pass5.dnam_bin, Tier::Ultra, &held_l, &held_r);
    let l2_al = normalized_mse(&auto_ul, &held_tgt_l, warmup);
    let l2_ar = normalized_mse(&auto_ur, &held_tgt_r, warmup);
    assert!(
        (l2_al - l2_ul).abs() < 1.0e-2 && (l2_ar - l2_ur).abs() < 1.0e-2,
        "CPU vs auto_detect divergence: L=({l2_ul}, {l2_al}) R=({l2_ur}, {l2_ar})"
    );
}

#[test]
fn test_phase_a_isolated_nonlinear_plus_op_t_comb_level1_and_level2() {
    let sr = 48_000.0_f64;
    let warmup = 64_usize;
    let in_l = gen_broadband(1_024, 0xC33E_0723_5432_1098, 0.45, 0.033);
    let in_r = gen_broadband(1_024, 0xC33E_0724_9876_1234, 0.41, 0.049);
    let (tgt_l, tgt_r) = eval_nonlinear_comb_operator_stereo(&in_l, &in_r);

    let dataset = SonaTrainDataset {
        sample_rate_hz: sr,
        warmup_frames: warmup,
        input_l: in_l.clone(),
        input_r: in_r.clone(),
        target_l: tgt_l.clone(),
        target_r: tgt_r.clone(),
        controls: Vec::new(),
        stability_constraint: StageStabilityConstraint::default(),
        control_samples: Vec::new(),
        control_audio_stems: Vec::new(),
        tsyn_entries: Vec::new(),
    };
    let mut cfg = SonaSolverConfig::default();
    cfg.pass2.max_epochs = 6;
    cfg.pass2.diminishing_patience = 2;

    let res = solve_sona_model(&TrainRunner::cpu(), &dataset, &cfg, None).unwrap();
    assert!(res.pass1.dominant_transport_delay_samples >= 5.0);
    assert!(res.pass2.best_loss.is_finite());
    assert!(
        res.pass5.adjacent_tier_monotone,
        "tier_reports={:#?}\nos2x={} pfx={:?} trim={:?}",
        res.pass5.tier_reports,
        res.model.oversample_upper_tiers_2x,
        res.model.prefix_override_needed,
        res.pass5.tier_gain_trim,
    );
    assert!(DnamBinView::parse(&res.pass5.dnam_bin).is_ok());

    let l1_l = normalized_mse(&res.pass5.vm_renders[3].0, &tgt_l, warmup);
    let l1_r = normalized_mse(&res.pass5.vm_renders[3].1, &tgt_r, warmup);
    assert!(
        l1_l < 0.20 && l1_r < 0.20,
        "Level-1 comb L={l1_l} R={l1_r} head_scale={} trim={:?}",
        res.pass5.head_gain_scale,
        res.pass5.tier_gain_trim,
    );

    // Level 2: unseen stereo dry waveform without re-training.
    let held_l = gen_broadband(1_024, 0xD44F_1834_3210_9876, 0.38, 0.039);
    let held_r = gen_broadband(1_024, 0xD44F_1835_8765_4321, 0.35, 0.058);
    let (held_tgt_l, held_tgt_r) = eval_nonlinear_comb_operator_stereo(&held_l, &held_r);
    let (gen_l, gen_r) = render_vm_tier(&res.pass5.dnam_bin, Tier::Ultra, &held_l, &held_r);
    assert!(
        gen_l
            .iter()
            .chain(&gen_r)
            .all(|v| v.is_finite() && v.abs() < 10.0)
    );
    let l2_l = normalized_mse(&gen_l, &held_tgt_l, warmup);
    let l2_r = normalized_mse(&gen_r, &held_tgt_r, warmup);
    assert!(l2_l < 0.25 && l2_r < 0.25, "Level-2 comb L={l2_l} R={l2_r}");
}

#[test]
fn test_phase_b_coupled_controls_cp1_cext_tsyn_level3_intermediate_and_wgpu_parity() {
    let sr = 48_000.0_f64;
    let warmup = 64_usize;
    let in_l = gen_broadband(1_024, 0xE550_2945_1098_7654, 0.36, 0.037);
    let in_r = gen_broadband(1_024, 0xE550_2946_6543_2109, 0.33, 0.051);
    let (tgt_l, tgt_r) = eval_coupled_stereo(&in_l, &in_r);

    // 10 controls -> triggers `"CP1\0"` overflow (> 8 slots), `"CEXT"` (`slot 26/30`), and `"TSYN"`.
    // Ctrl 0 observes `[2.0..8.0]` inside declared `[0.0..10.0]` (`[CLAMP-RANGE]`).
    // Ctrl 1 observes bits `0b0111` inside declared `0b1111` (`[CLAMP-SWITCH]`).
    let mut controls = vec![
        ControlSpec {
            kind: ControlKindSpec::Range {
                declared_min: 0.0,
                declared_max: 10.0,
                default_val: 5.0,
                obs_min: 2.0,
                obs_max: 8.0,
                unique_values: 3,
                lobatto_anchors: [2.0, 5.0, 8.0],
            },
        },
        ControlSpec {
            kind: ControlKindSpec::Switch {
                num_options: 4,
                default_opt: 0,
                declared_mask: 0b1111,
                observed_mask: 0b0111,
            },
        },
    ];
    for _ in 2..10 {
        controls.push(ControlSpec {
            kind: ControlKindSpec::Range {
                declared_min: 0.0,
                declared_max: 10.0,
                default_val: 5.0,
                obs_min: 0.0,
                obs_max: 10.0,
                unique_values: 3,
                lobatto_anchors: [0.0, 5.0, 10.0],
            },
        });
    }

    let constraint = StageStabilityConstraint {
        delay_ctrl_idx: Some(2),
        delay_slopes: vec![0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        base_tau: 16.0,
        base_g_lin: 0.5,
        g_lin_slopes: vec![0.0; 10],
        switch_hazards: vec![(1, 2, 0.50)],
        base_loop_gain: 0.25,
        loop_gain_slopes: vec![0.05; 10],
        feedback_ctrl_idx: Some(0),
        crossbar_coupled: true,
    };

    // Anchor training points ONLY at c0 in {2.0, 5.0, 8.0} — omitting intermediate {3.25, 6.75}.
    let mut samples = Vec::new();
    for &c0 in &[2.0_f32, 5.0, 8.0] {
        for &c2 in &[0.0_f32, 5.0, 10.0] {
            for &sw in &[0_u8, 1, 2] {
                let mut phys = vec![5.0_f32; 10];
                phys[0] = c0;
                phys[1] = f32::from(sw);
                phys[2] = c2;
                let mut sw_opts = vec![0_u8; 10];
                sw_opts[1] = sw;
                let (st, sud) = eval_test3_analytical_targets(c0, c2);
                samples.push(Cp1ControlSample {
                    phys_controls: phys,
                    switch_options: sw_opts,
                    stage_targets_base: st,
                    stage_targets_ultra_delta: Some(sud),
                });
            }
        }
    }

    let tsyn = vec![
        build_tsyn_entry_wire(0, false, 1.0, 120.0, [2.0, 5.0, 8.0], (2.0, 8.0), true)
            .expect("valid tempo-sync entry"),
    ];

    let dataset = SonaTrainDataset {
        sample_rate_hz: sr,
        warmup_frames: warmup,
        input_l: in_l.clone(),
        input_r: in_r.clone(),
        target_l: tgt_l,
        target_r: tgt_r,
        controls,
        stability_constraint: constraint,
        control_samples: samples,
        control_audio_stems: Vec::new(),
        tsyn_entries: tsyn,
    };
    let mut cfg = SonaSolverConfig::default();
    cfg.pass2.max_epochs = 4;
    cfg.pass2.diminishing_patience = 2;

    let res_cpu = solve_sona_model(&TrainRunner::cpu(), &dataset, &cfg, None).unwrap();
    let res_auto = solve_sona_model(&TrainRunner::auto_detect(), &dataset, &cfg, None).unwrap();

    let view = DnamBinView::parse(&res_cpu.pass5.dnam_bin).unwrap();
    assert_eq!(view.control_descriptors.len(), 10);
    assert!(view.cp1_overflow.starts_with(&CP1_STREAM_MAGIC));
    assert!(view.cext.is_some());
    assert_eq!(view.tsyn_entries.len(), 1);

    // Exact 0-ULP StarCenter invariance at default control point (5.0, sw=0, 5.0).
    let mut bus =
        ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view, Tier::Ultra, 48_000.0)
            .expect("ControlBus");
    bus.set_controls_batch(&[(0, 5.0), (1, 0.0), (2, 5.0)])
        .unwrap();
    bus.eval_static_and_snap_if_dirty();
    for s in 0..STAGE_TARGET_SLOTS {
        assert_eq!(
            bus.stage_targets()[s].to_bits(),
            0_u32,
            "slot {s} must be exact +0.0_f32 at StarCenter"
        );
    }

    // Level 3: unseen stereo excitation at StarCenter & omitted intermediate points (3.25, 6.75).
    let held_l = gen_broadband(1_024, 0xF661_3A56_9876_5432, 0.34, 0.037);
    let held_r = gen_broadband(1_024, 0xF661_3A57_5432_9876, 0.30, 0.053);
    let (star_ref_l, star_ref_r) =
        render_vm_tier(&res_cpu.pass5.dnam_bin, Tier::Ultra, &held_l, &held_r);
    let (star_ctl_l, star_ctl_r) = render_vm_with_controls(
        &res_cpu.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 5.0), (1, 0.0), (2, 5.0)],
        &held_l,
        &held_r,
    );
    for (al, bl) in star_ref_l.iter().zip(&star_ctl_l) {
        assert_eq!(
            (al - bl).abs(),
            0.0,
            "StarCenter Left must be 0-ULP invariant"
        );
    }
    for (ar, br) in star_ref_r.iter().zip(&star_ctl_r) {
        assert_eq!(
            (ar - br).abs(),
            0.0,
            "StarCenter Right must be 0-ULP invariant"
        );
    }

    // Unobserved range clamping: 0.0 clamps bit-identically to obs_min = 2.0 on both channels.
    let (clamped_l, clamped_r) = render_vm_with_controls(
        &res_cpu.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 0.0)],
        &held_l,
        &held_r,
    );
    let (anchor_l, anchor_r) = render_vm_with_controls(
        &res_cpu.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 2.0)],
        &held_l,
        &held_r,
    );
    for ((a_l, b_l), (a_r, b_r)) in clamped_l
        .iter()
        .zip(&anchor_l)
        .zip(clamped_r.iter().zip(&anchor_r))
    {
        assert_eq!((a_l - b_l).abs(), 0.0);
        assert_eq!((a_r - b_r).abs(), 0.0);
    }

    // Analytical intermediate target ground truth at omitted points (3.25, 3.5) and (6.75, 7.5).
    for &(c0, c2) in &[(3.25_f32, 3.5_f32), (6.75_f32, 7.5_f32)] {
        bus.set_controls_batch(&[(0, c0), (1, 0.0), (2, c2)])
            .unwrap();
        bus.eval_static_and_snap_if_dirty();
        let (st_e, sud_e) = eval_test3_analytical_targets(c0, c2);
        let got = bus.stage_targets();
        for &(slot, exp) in &[
            (0, st_e[0] + sud_e[0]),
            (2, st_e[2]),
            (15, st_e[15]),
            (26, st_e[26]),
            (30, st_e[30]),
        ] {
            assert!(
                (f64::from(got[slot]) - exp).abs() < 2.5e-2,
                "slot {slot} at ({c0},{c2}): got={} exp={exp}",
                got[slot]
            );
        }
    }

    let (st_hi, sud_hi) = eval_test3_analytical_targets(6.75, 7.5);
    let mut st_ultra_exp = [0.0_f64; STAGE_TARGET_SLOTS];
    for s in 0..STAGE_TARGET_SLOTS {
        st_ultra_exp[s] = st_hi[s] + sud_hi[s];
    }
    let (gt_l, gt_r) = render_vm_pinned_stage_targets(
        &res_cpu.pass5.dnam_bin,
        Tier::Ultra,
        &st_ultra_exp,
        &held_l,
        &held_r,
    );
    let (mid_hi_l, mid_hi_r) = render_vm_with_controls(
        &res_cpu.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 6.75), (2, 7.5)],
        &held_l,
        &held_r,
    );
    let (mid_auto_l, mid_auto_r) = render_vm_with_controls(
        &res_auto.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 6.75), (2, 7.5)],
        &held_l,
        &held_r,
    );
    let gt_mse_l = normalized_mse(&mid_hi_l, &gt_l, warmup);
    let gt_mse_r = normalized_mse(&mid_hi_r, &gt_r, warmup);
    assert!(
        gt_mse_l < 2.0e-2 && gt_mse_r < 2.0e-2,
        "Level-3 analytical ground-truth mismatch: L={gt_mse_l} R={gt_mse_r}"
    );
    let diff_inter_l = normalized_mse(&mid_hi_l, &anchor_l, warmup);
    assert!(
        diff_inter_l > 1.0e-5,
        "intermediate control must modulate output cleanly: {diff_inter_l}"
    );
    let parity_l = normalized_mse(&mid_auto_l, &mid_hi_l, warmup);
    let parity_r = normalized_mse(&mid_auto_r, &mid_hi_r, warmup);
    assert!(
        parity_l < 1.0e-3 && parity_r < 1.0e-3,
        "CPU vs auto_detect Level-3 stereo parity mismatch: L={parity_l} R={parity_r}"
    );

    // Explicit +6 dB Phase-3 head-gain idempotence verification.
    let raw_gains = vec![2.5_f64, 0.25_f64];
    let mags = evaluate_readout_spectral_magnitudes(&raw_gains, 0.2, sr, 64);
    let mut p3 = Phase3AuditInput {
        min_delay_samples: 8.0,
        max_delay_excursion: 1.0,
        max_paralru_r_excursion: 0.99,
        max_tank_g_excursion: 0.95,
        crossbar_coupled: true,
        ro_pole_48k: 0.2,
        readout_gains_l: raw_gains,
        readout_gains_r: vec![1.0, 0.0],
        freq_mag_l: mags,
        freq_mag_r: vec![1.0; 64],
        burst_peak_abs: 2.0,
    };
    let r_first = audit_phase3_dynamic_and_head_gain(&mut p3).unwrap();
    assert!(r_first.head_gain_scale < 1.0);
    let r_second = audit_phase3_dynamic_and_head_gain(&mut p3).unwrap();
    assert_eq!(r_second.head_gain_scale, 1.0);
}

#[test]
fn test_phase_b_pass4_audio_driven_wiener_control_extraction() {
    let sr = 48_000.0_f64;
    let warmup = 64_usize;
    let in_l = gen_broadband(1_024, 0x9988_7766_5544_3322, 0.38, 0.037);
    let in_r = gen_broadband(1_024, 0x1122_3344_5566_7788, 0.34, 0.053);
    let (base_l, base_r) = eval_coupled_stereo(&in_l, &in_r);

    let controls = vec![ControlSpec {
        kind: ControlKindSpec::Range {
            declared_min: 0.0,
            declared_max: 10.0,
            default_val: 5.0,
            obs_min: 2.0,
            obs_max: 8.0,
            unique_values: 3,
            lobatto_anchors: [2.0, 5.0, 8.0],
        },
    }];

    let stem_in_l = gen_broadband(768, 0x5566_7788_1122_3344, 0.36, 0.043);
    let stem_in_r = gen_broadband(768, 0x4433_2211_8877_6655, 0.32, 0.059);
    let (mid_l, mid_r) = eval_coupled_stereo(&stem_in_l, &stem_in_r);
    let low_l: Vec<f64> = mid_l.iter().map(|&v| 0.55 * v).collect();
    let low_r: Vec<f64> = mid_r.iter().map(|&v| 0.55 * v).collect();
    let hi_l: Vec<f64> = mid_l
        .iter()
        .map(|&v| (1.75 * v).tanh() + 0.45 * v)
        .collect();
    let hi_r: Vec<f64> = mid_r
        .iter()
        .map(|&v| (1.75 * v).tanh() + 0.45 * v)
        .collect();

    let stems = vec![
        ControlAudioStem {
            phys_controls: vec![2.0],
            switch_options: vec![0],
            input_l: stem_in_l.clone(),
            input_r: stem_in_r.clone(),
            target_l: low_l,
            target_r: low_r,
        },
        ControlAudioStem {
            phys_controls: vec![5.0],
            switch_options: vec![0],
            input_l: stem_in_l.clone(),
            input_r: stem_in_r.clone(),
            target_l: mid_l,
            target_r: mid_r,
        },
        ControlAudioStem {
            phys_controls: vec![8.0],
            switch_options: vec![0],
            input_l: stem_in_l,
            input_r: stem_in_r,
            target_l: hi_l,
            target_r: hi_r,
        },
    ];

    let dataset = SonaTrainDataset {
        sample_rate_hz: sr,
        warmup_frames: warmup,
        input_l: in_l,
        input_r: in_r,
        target_l: base_l,
        target_r: base_r,
        controls,
        stability_constraint: StageStabilityConstraint::default(),
        control_samples: Vec::new(),
        control_audio_stems: stems,
        tsyn_entries: Vec::new(),
    };
    let mut cfg = SonaSolverConfig::default();
    cfg.pass2.max_epochs = 4;
    cfg.pass2.diminishing_patience = 2;

    let res = solve_sona_model(&TrainRunner::cpu(), &dataset, &cfg, None).unwrap();
    assert!(res.control_bus.is_some());
    let view = DnamBinView::parse(&res.pass5.dnam_bin).unwrap();
    let mut bus =
        ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(&view, Tier::Ultra, 48_000.0)
            .expect("ControlBus");
    bus.set_control_physical(0, 5.0).unwrap();
    bus.eval_static_and_snap_if_dirty();
    for s in 0..STAGE_TARGET_SLOTS {
        assert_eq!(bus.stage_targets()[s].to_bits(), 0_u32);
    }

    let held_l = gen_broadband(768, 0x1234_5678_9ABC_DEF0, 0.35, 0.041);
    let held_r = gen_broadband(768, 0x0FED_CBA9_8765_4321, 0.32, 0.061);
    let (clamp_l, clamp_r) = render_vm_with_controls(
        &res.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 0.0)],
        &held_l,
        &held_r,
    );
    let (low_out_l, low_out_r) = render_vm_with_controls(
        &res.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 2.0)],
        &held_l,
        &held_r,
    );
    let (hi_out_l, hi_out_r) = render_vm_with_controls(
        &res.pass5.dnam_bin,
        Tier::Ultra,
        &[(0, 6.5)],
        &held_l,
        &held_r,
    );
    for (a, b) in clamp_l
        .iter()
        .zip(&low_out_l)
        .chain(clamp_r.iter().zip(&low_out_r))
    {
        assert_eq!((a - b).abs(), 0.0);
    }
    let mod_l = normalized_mse(&hi_out_l, &low_out_l, warmup);
    let mod_r = normalized_mse(&hi_out_r, &low_out_r, warmup);
    assert!(
        mod_l > 1.0e-5 && mod_r > 1.0e-5,
        "Wiener audio stems must drive modulation: L={mod_l} R={mod_r}"
    );
}
