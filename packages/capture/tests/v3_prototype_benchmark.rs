#![forbid(unsafe_code)]
//! Head-to-head `v2.0` (`DnamEngine`) vs `v3` (`V3CoreInstance` + `VarPro`/Adjoint L-BFGS) benchmark
//! across `[Lite, Full, Pro, Ultra]` on flagship hard nonlinear & dynamic black-box chains.

use std::collections::BTreeMap;
use std::time::Instant;

use dnam_capture::metrics::{
    ArchetypeFamily, PerceptualCertificationSummary, canonical_ltas_33_centers_hz,
    certify_4metric_signal_chain, evaluate_besr_from_band_energies, split_bezout_3band_energies,
    verify_monotone_4tier_summaries,
};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, ProjChunkKind, STMC_FLAG_WET_POLARITY_INVERTED,
    export_runtime_dnam, verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use dnam_capture::solver::{
    V3TrainDataset, V3TrainStem, canonical_besr_warmup, fit_v3_core_four_tiers,
    star_center_control_values,
};
use dnam_capture::stimulus::{AdaptiveTrainingOptions, TopologyProbeMode, train_adaptive_target};
use dnam_capture::{
    DistillationTarget, StemResponse, StemStimulus, StimulusFrame, realign_fractional_kaiser64,
    resample_arbitrary_rate, resample_control_trajectory, resampled_frame_len, scale_freeze_window,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_engine::stages::v3_core::V3CoreInstance;
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain04TubeAmpFmv4x12,
    Chain06EnvelopeResonantSvf,
};

#[derive(Clone, Debug)]
struct MatrixAdapter<M: BlackBoxModel> {
    inner: M,
    switches: Vec<u8>,
}

impl<M: BlackBoxModel> MatrixAdapter<M> {
    fn from_model(inner: M) -> Self {
        let switches = inner.metadata().default_switches();
        Self { inner, switches }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_internal(
        &mut self,
        stem_id: String,
        stim_sr: u32,
        in_1: &[f32],
        in_2: Option<&[f32]>,
        in_3: Option<&[f32]>,
        in_4: Option<&[f32]>,
        control_values: &[f32],
        control_trajectories: Option<&Vec<Vec<f32>>>,
        freeze_window: Option<[usize; 2]>,
    ) -> Result<StemResponse, DnamError> {
        let dev_sr = self.inner.sample_rate_hz();
        let orig_n = in_1.len();
        if orig_n == 0 {
            let n_out = self.inner.num_outputs().clamp(1, 4);
            return Ok(StemResponse {
                stem_id,
                sample_rate_hz: stim_sr,
                out_1: Vec::new(),
                out_2: (n_out >= 2).then(Vec::new),
                out_3: (n_out >= 3).then(Vec::new),
                out_4: (n_out >= 4).then(Vec::new),
                internal_nodes: BTreeMap::new(),
            });
        }
        let needs_resample = stim_sr != dev_sr;
        let dev_n = if needs_resample {
            resampled_frame_len(orig_n, stim_sr, dev_sr)
        } else {
            orig_n
        };
        let (rs1, rs2, rs3, rs4, rstraj);
        let (dev_in1, dev_in2, dev_in3, dev_in4, dev_trajs, dev_freeze) = if needs_resample {
            rs1 = resample_arbitrary_rate(in_1, stim_sr, dev_sr, Some(dev_n));
            rs2 = in_2.map(|c| resample_arbitrary_rate(c, stim_sr, dev_sr, Some(dev_n)));
            rs3 = in_3.map(|c| resample_arbitrary_rate(c, stim_sr, dev_sr, Some(dev_n)));
            rs4 = in_4.map(|c| resample_arbitrary_rate(c, stim_sr, dev_sr, Some(dev_n)));
            rstraj = control_trajectories.map(|ts| {
                ts.iter()
                    .map(|tr| resample_control_trajectory(tr, dev_n))
                    .collect::<Vec<Vec<f32>>>()
            });
            (
                rs1.as_slice(),
                rs2.as_deref(),
                rs3.as_deref(),
                rs4.as_deref(),
                rstraj.as_ref(),
                freeze_window.map(|w| scale_freeze_window(w, orig_n, dev_n)),
            )
        } else {
            (in_1, in_2, in_3, in_4, control_trajectories, freeze_window)
        };

        let meta = self.inner.metadata();
        let default_controls = meta.default_controls();
        let freeze_sw_idx = meta
            .switches
            .iter()
            .position(|sw| sw.id == "freeze")
            .unwrap_or(0);
        let n_sw = self.switches.len().max(meta.switches.len());
        let mut sw_scratch: Vec<u8> = (0..n_sw)
            .map(|k| {
                self.switches
                    .get(k)
                    .copied()
                    .or_else(|| meta.switches.get(k).map(|s| s.default_index))
                    .unwrap_or(0)
            })
            .collect();
        let base_freeze_sw = sw_scratch.get(freeze_sw_idx).copied().unwrap_or(0);
        let n_in = self.inner.num_inputs().clamp(1, 4);
        let n_out = self.inner.num_outputs().clamp(1, 4);
        let n_range = meta.controls.len();
        let mut ctrl_scratch = vec![0.0_f32; n_range];
        let mut dev_out1 = vec![0.0_f32; dev_n];
        let mut dev_out2 = (n_out >= 2).then(|| vec![0.0_f32; dev_n]);
        let mut dev_out3 = (n_out >= 3).then(|| vec![0.0_f32; dev_n]);
        let mut dev_out4 = (n_out >= 4).then(|| vec![0.0_f32; dev_n]);
        let sidechain_uses_in3 = dev_in2.is_none() && dev_in3.is_some();

        for i in 0..dev_n {
            let x1 = dev_in1[i];
            let x2 = if sidechain_uses_in3 {
                dev_in3.and_then(|v| v.get(i).copied()).unwrap_or(x1)
            } else {
                dev_in2.and_then(|v| v.get(i).copied()).unwrap_or(x1)
            };
            let x3 = dev_in3.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            let x4 = dev_in4.and_then(|v| v.get(i).copied()).unwrap_or(0.0);
            for p in 0..n_range {
                let fb = control_values
                    .get(p)
                    .copied()
                    .or_else(|| default_controls.get(p).copied())
                    .unwrap_or(0.0);
                ctrl_scratch[p] = dev_trajs
                    .and_then(|t| t.get(p))
                    .and_then(|t| t.get(i).copied())
                    .unwrap_or(fb);
            }
            for k in 0..n_sw {
                let fb_sw = self
                    .switches
                    .get(k)
                    .copied()
                    .or_else(|| meta.switches.get(k).map(|s| s.default_index))
                    .unwrap_or(0);
                let max_opt = meta
                    .switches
                    .get(k)
                    .map_or(255, |s| s.options.len().saturating_sub(1));
                sw_scratch[k] = dev_trajs
                    .and_then(|t| t.get(n_range + k))
                    .and_then(|t| t.get(i).copied())
                    .or_else(|| control_values.get(n_range + k).copied())
                    .map_or(fb_sw, |v| (v.round().max(0.0) as usize).min(max_opt) as u8);
            }
            if let Some([fs, fe]) = dev_freeze
                && let Some(slot) = sw_scratch.get_mut(freeze_sw_idx)
            {
                *slot = if i >= fs && i < fe { 1 } else { base_freeze_sw };
            }
            let in_f = [x1, x2, x3, x4];
            let mut y = [0.0_f32; 4];
            self.inner
                .process_frame(&in_f[..n_in], &mut y[..n_out], &ctrl_scratch, &sw_scratch);
            dev_out1[i] = y[0];
            if let Some(ref mut c2) = dev_out2 {
                c2[i] = y[1];
            }
            if let Some(ref mut c3) = dev_out3 {
                c3[i] = y[2];
            }
            if let Some(ref mut c4) = dev_out4 {
                c4[i] = y[3];
            }
        }
        let (out_1, out_2, out_3, out_4) = if needs_resample {
            (
                resample_arbitrary_rate(&dev_out1, dev_sr, stim_sr, Some(orig_n)),
                dev_out2.map(|c| resample_arbitrary_rate(&c, dev_sr, stim_sr, Some(orig_n))),
                dev_out3.map(|c| resample_arbitrary_rate(&c, dev_sr, stim_sr, Some(orig_n))),
                dev_out4.map(|c| resample_arbitrary_rate(&c, dev_sr, stim_sr, Some(orig_n))),
            )
        } else {
            (dev_out1, dev_out2, dev_out3, dev_out4)
        };
        Ok(StemResponse {
            stem_id,
            sample_rate_hz: stim_sr,
            out_1,
            out_2,
            out_3,
            out_4,
            internal_nodes: BTreeMap::new(),
        })
    }
}

impl<M: BlackBoxModel> DistillationTarget for MatrixAdapter<M> {
    fn reset_state(&mut self) {
        self.inner.reset();
    }
    fn render_stem(&mut self, s: &StemStimulus) -> Result<StemResponse, DnamError> {
        self.reset_state();
        self.render_internal(
            s.stem_id.clone(),
            s.sample_rate_hz,
            &s.in_1,
            s.in_2.as_deref(),
            s.in_3.as_deref(),
            s.in_4.as_deref(),
            &s.control_values,
            s.control_trajectories.as_ref(),
            s.freeze_transition_window,
        )
    }
    fn render_frame(&mut self, f: &StimulusFrame) -> Result<StemResponse, DnamError> {
        self.render_internal(
            format!("frame_{}", f.frame_index),
            f.sample_rate_hz,
            &f.in_1,
            f.in_2.as_deref(),
            f.in_3.as_deref(),
            f.in_4.as_deref(),
            &f.control_values,
            f.control_trajectories.as_ref(),
            None,
        )
    }
}

fn manifest_for_fixture<M: BlackBoxModel>(model: &M, label_suffix: &str) -> DnamProjManifest {
    let meta = model.metadata();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.metadata.name = format!("{} {label_suffix}", meta.name);
    manifest.base.io_config.inputs = model.num_inputs().clamp(1, 2) as u8;
    manifest.base.io_config.output_mode = if model.num_outputs() >= 2 {
        "stereo".to_string()
    } else {
        "mono".to_string()
    };
    let mut schema = Vec::new();
    for c in meta.controls {
        let def: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": c.id,
            "name": c.name,
            "type": "range",
            "lobatto_degree": 3,
            "declared_range": [-1.0, 1.0],
            "safe_clamped_range": [-1.0, 1.0],
            "default": c.default_norm
        }))
        .unwrap();
        schema.push(def);
    }
    for sw in meta.switches {
        let def: ControlSchemaDef = serde_json::from_value(serde_json::json!({
            "id": sw.id,
            "name": sw.name,
            "type": "switch",
            "declared_options": sw.options,
            "active_options": sw.options,
            "default": sw.options[usize::from(sw.default_index)]
        }))
        .unwrap();
        schema.push(def);
    }
    manifest.base.controls_schema = schema;
    manifest
}

fn combine_stereo_summary(
    l: PerceptualCertificationSummary,
    r: PerceptualCertificationSummary,
) -> PerceptualCertificationSummary {
    PerceptualCertificationSummary {
        family: l.family,
        tier: l.tier,
        raw_besr_db: l.raw_besr_db.max(r.raw_besr_db),
        besr_db: l.besr_db.max(r.besr_db),
        band_esr_db: std::array::from_fn(|b| l.band_esr_db[b].max(r.band_esr_db[b])),
        qualified_band_mask: std::array::from_fn(|b| {
            l.qualified_band_mask[b] || r.qualified_band_mask[b]
        }),
        mr_iir: l.mr_iir.max(r.mr_iir),
        ltas_max_db: l.ltas_max_db.max(r.ltas_max_db),
        ltas_unadjusted_max_db: l.ltas_unadjusted_max_db.max(r.ltas_unadjusted_max_db),
        ltas_excluded_bands: std::array::from_fn(|b| {
            l.ltas_excluded_bands[b] && r.ltas_excluded_bands[b]
        }),
        ltas_residual_band_power: std::array::from_fn(|b| {
            l.ltas_residual_band_power[b].max(r.ltas_residual_band_power[b])
        }),
        seg_esr_db: l.seg_esr_db.max(r.seg_esr_db),
        passes_target: l.passes_target && r.passes_target,
        passes_hard_ceiling: l.passes_hard_ceiling && r.passes_hard_ceiling,
    }
}

fn build_unseen_holdout_excitation(n: usize, sr_hz: f64) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64) / sr_hz;
        let env = (0.35 + 0.65 * (-((i % 2_000) as f64) / 650.0).exp())
            * (0.7 + 0.3 * (std::f64::consts::TAU * 1.8 * t).sin());
        let chord = 0.28 * (std::f64::consts::TAU * 110.0 * t).sin()
            + 0.22 * (std::f64::consts::TAU * 164.81 * t + 0.3).sin()
            + 0.16 * (std::f64::consts::TAU * 293.66 * t + 0.7).sin()
            + 0.11 * (std::f64::consts::TAU * 587.33 * t + 1.1).sin()
            + 0.07 * (std::f64::consts::TAU * 1_480.0 * t).cos()
            + 0.04 * (std::f64::consts::TAU * 3_720.0 * t).sin();
        let xl = (env * chord) as f32;
        let xr = (env * (chord * 0.96 + 0.02 * (std::f64::consts::TAU * 220.0 * t).cos())) as f32;
        out.push([xl, xr]);
    }
    out
}

fn build_identification_sweep_excitation(n: usize, sr_hz: f64) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0_f64;
    let f0 = 20.0_f64;
    let ratio = (20_000.0_f64 / f0).ln();
    for i in 0..n {
        let frac = (i as f64) / (n.max(1) as f64);
        let t = (i as f64) / sr_hz;
        let freq = f0 * (ratio * frac).exp();
        phase += std::f64::consts::TAU * freq / sr_hz;
        let pick_decay = (-((i % 1_600) as f64) / 480.0).exp();
        let swell = (0.5 - 0.5 * (std::f64::consts::TAU * 1.6 * t).cos()).powf(1.4);
        let amp = 0.03 + 0.39 * swell * (0.25 + 0.75 * pick_decay);
        let bass_mod = 0.28
            * (std::f64::consts::TAU * 98.0 * t + 0.3 * (std::f64::consts::TAU * 2.1 * t).sin())
                .sin();
        let xl = (amp * (0.72 * phase.sin() + bass_mod)) as f32;
        let xr = (amp * (0.70 * (phase + 0.08).sin() + 0.95 * bass_mod)) as f32;
        out.push([xl, xr]);
    }
    out
}

fn build_fullband_eval_excitation(n: usize, sr_hz: f64) -> Vec<[f32; 2]> {
    let centers = canonical_ltas_33_centers_hz();
    let weights: [f64; 33] =
        std::array::from_fn(|k| 1.0 / (1.0 + (centers[k] / 2_400.0).powi(2)).sqrt());
    let norm = weights
        .iter()
        .map(|w| w * w)
        .sum::<f64>()
        .sqrt()
        .max(1.0e-12);
    let scale = 0.52 / norm;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64) / sr_hz;
        let env = (0.45 + 0.55 * (-((i % 2_400) as f64) / 800.0).exp())
            * (0.72 + 0.28 * (std::f64::consts::TAU * 2.3 * t).sin());
        let mut sl = 0.0_f64;
        let mut sr = 0.0_f64;
        for (k, &fc) in centers.iter().enumerate() {
            let phi_l =
                ((k as f64 + 1.0) * std::f64::consts::GOLDEN_RATIO).fract() * std::f64::consts::TAU;
            let phi_r = phi_l + 0.29 * ((k & 3) as f64 + 0.5);
            let arg = std::f64::consts::TAU * fc * t;
            sl += weights[k] * (arg + phi_l).cos();
            sr += weights[k] * (arg + phi_r).cos();
        }
        out.push([(env * scale * sl) as f32, (env * scale * sr) as f32]);
    }
    out
}

struct CompareRow {
    label: String,
    tier: Tier,
    v2_sum: PerceptualCertificationSummary,
    v3_sum: PerceptualCertificationSummary,
    v2_holdout_db: f64,
    v3_holdout_db: f64,
    v3_os2x: bool,
    v2_ms: u128,
    v3_ms: u128,
    v3_monotone: bool,
}

fn benchmark_chain<M: BlackBoxModel>(
    label: &str,
    model: M,
    family: ArchetypeFamily,
    rows: &mut Vec<CompareRow>,
) {
    let crypto = DefaultCryptoProvider::default();
    let n_in = model.num_inputs();
    let n_out = model.num_outputs();
    let topo = match (n_in >= 2, n_out >= 2) {
        (true, true) => TopologyProbeMode::TrueStereo2To2,
        (true, false) => TopologyProbeMode::DualInToMono2To1,
        (false, true) => TopologyProbeMode::MonoToStereo1To2,
        (false, false) => TopologyProbeMode::Mono1To1,
    };
    let manifest = manifest_for_fixture(&model, label);
    let mut adapter = MatrixAdapter::from_model(model);
    let opts = AdaptiveTrainingOptions {
        sample_rate_hz: 48_000,
        topology_mode: topo,
        fixed_duration: true,
        prefer_mode_b: true,
        max_frames_per_stem: 3,
        run_live_null_test: false,
        null_test_autopatch_threshold_db: -42.0,
        calb: None,
    };
    let t_v2 = Instant::now();
    let report = train_adaptive_target(manifest, &mut adapter, &opts).unwrap();
    let v2_ms = t_v2.elapsed().as_millis();

    let proj_view = DnamProjView::from_bytes(&report.proj_bytes, &crypto).unwrap();
    let _inpc_view =
        verify_proj_inpc_capsule(proj_view.chunk_slice(ProjChunkKind::Inpc, 0).unwrap()).unwrap();
    let star_entry = proj_view
        .manifest
        .stems_registry
        .iter()
        .find(|s| s.enabled && s.is_star_center)
        .unwrap();
    let stmc_view = verify_proj_stmc_capsule(
        proj_view
            .chunk_slice(ProjChunkKind::Stmc, star_entry.chunk_id)
            .unwrap(),
    )
    .unwrap();
    let mono_only = proj_view.manifest.base.io_config.output_mode == "mono";

    let star_ctrls = star_center_control_values(&proj_view.manifest);
    let hold_inv_pol = (stmc_view.flags & STMC_FLAG_WET_POLARITY_INVERTED) != 0;
    let hold_lag = f64::from(stmc_view.fractional_lag_samples_f32);

    let render_aligned =
        |adapter: &mut MatrixAdapter<M>, name: &str, dry: &[[f32; 2]]| -> (Vec<f32>, Vec<f32>) {
            let h1: Vec<f32> = dry.iter().map(|v| v[0]).collect();
            let h2: Vec<f32> = dry.iter().map(|v| v[1]).collect();
            adapter.reset_state();
            let resp = adapter
                .render_internal(
                    name.to_string(),
                    48_000,
                    &h1,
                    (n_in >= 2).then_some(h2.as_slice()),
                    None,
                    None,
                    &star_ctrls,
                    None,
                    None,
                )
                .unwrap();
            let al = realign_fractional_kaiser64(&resp.out_1, hold_lag, 0.0, hold_inv_pol);
            let ar = resp.out_2.as_ref().map_or_else(
                || al.clone(),
                |v| realign_fractional_kaiser64(v, hold_lag, 0.0, hold_inv_pol),
            );
            (al, ar)
        };

    // Unseen holdout target
    let holdout_dry = build_unseen_holdout_excitation(6_000, 48_000.0);
    let (hold_al_l, hold_al_r) = render_aligned(&mut adapter, "holdout", &holdout_dry);
    let hold_tgt_l: Vec<f64> = hold_al_l.iter().map(|&v| f64::from(v)).collect();
    let hold_tgt_r: Vec<f64> = hold_al_r.iter().map(|&v| f64::from(v)).collect();
    let hold_warmup = canonical_besr_warmup(hold_tgt_l.len());
    let hold_in_l: Vec<f32> = holdout_dry.iter().map(|v| v[0]).collect();
    let hold_in_r: Vec<f32> = holdout_dry.iter().map(|v| v[1]).collect();

    // Full-bandwidth 20 Hz-20 kHz f32 identification & certification signal (12,000 samples across all 33 LTAS bands)
    let eval_dry = build_fullband_eval_excitation(12_000, 48_000.0);
    let (eval_al_l, eval_al_r) = render_aligned(&mut adapter, "fullband_eval", &eval_dry);
    let n_eval = eval_dry.len().min(eval_al_l.len()).min(eval_al_r.len());
    let eval_tgt_l: Vec<f64> = eval_al_l[..n_eval].iter().map(|&v| f64::from(v)).collect();
    let eval_tgt_r: Vec<f64> = eval_al_r[..n_eval].iter().map(|&v| f64::from(v)).collect();
    let eval_warmup = canonical_besr_warmup(n_eval);
    let eval_in_l: Vec<f32> = eval_dry[..n_eval].iter().map(|v| v[0]).collect();
    let eval_in_r: Vec<f32> = eval_dry[..n_eval].iter().map(|v| v[1]).collect();

    // v3 Identification Dataset: [fullband StarCenter excitation, disjoint 20 Hz-20 kHz sweep]
    let sweep_dry = build_identification_sweep_excitation(8_000, 48_000.0);
    let (sw_al_l, sw_al_r) = render_aligned(&mut adapter, "v3_id_sweep", &sweep_dry);
    let fullband_stem = V3TrainStem {
        in_l: eval_in_l.clone(),
        in_r: eval_in_r.clone(),
        tgt_l: eval_al_l[..n_eval].to_vec(),
        tgt_r: eval_al_r[..n_eval].to_vec(),
        warmup: eval_warmup,
    };
    let sw_len = sweep_dry.len().min(sw_al_l.len()).min(sw_al_r.len());
    let sweep_stem = V3TrainStem {
        in_l: sweep_dry[..sw_len].iter().map(|v| v[0]).collect(),
        in_r: sweep_dry[..sw_len].iter().map(|v| v[1]).collect(),
        tgt_l: sw_al_l[..sw_len].to_vec(),
        tgt_r: sw_al_r[..sw_len].to_vec(),
        warmup: canonical_besr_warmup(sw_len),
    };
    let ds =
        V3TrainDataset::new_weighted(vec![(fullband_stem, 0.80), (sweep_stem, 0.20)], 48_000.0);

    let t_v3 = Instant::now();
    let v3_models = fit_v3_core_four_tiers(&ds, mono_only);
    let v3_ms = t_v3.elapsed().as_millis();

    let eval_hold = |mdl_l: &[f64], mdl_r: &[f64]| -> f64 {
        let (s, e, ns) =
            split_bezout_3band_energies(&hold_tgt_l, mdl_l, None, 48_000.0, hold_warmup);
        let rl = evaluate_besr_from_band_energies(s, e, ns).besr_db;
        if mono_only {
            return rl;
        }
        let (sr, er, nr) =
            split_bezout_3band_energies(&hold_tgt_r, mdl_r, None, 48_000.0, hold_warmup);
        rl.max(evaluate_besr_from_band_energies(sr, er, nr).besr_db)
    };

    let cert_fullband =
        |tier: Tier, mdl_l: &[f64], mdl_r: &[f64]| -> PerceptualCertificationSummary {
            let sl = certify_4metric_signal_chain(
                family,
                tier,
                &eval_tgt_l,
                mdl_l,
                None,
                48_000.0,
                eval_warmup,
            );
            if mono_only {
                sl
            } else {
                let sr = certify_4metric_signal_chain(
                    family,
                    tier,
                    &eval_tgt_r,
                    mdl_r,
                    None,
                    48_000.0,
                    eval_warmup,
                );
                combine_stereo_summary(sl, sr)
            }
        };

    let mut v2_sums = Vec::with_capacity(4);
    let mut v3_sums = Vec::with_capacity(4);
    let mut v2_holds = [0.0_f64; 4];
    let mut v3_holds = [0.0_f64; 4];

    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let dnam = export_runtime_dnam(&report.proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam, tier, 48_000).unwrap();
        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let v2_c: Vec<[f32; 2]> = eval_dry[..n_eval]
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let v2_cl: Vec<f64> = v2_c.iter().map(|v| f64::from(v[0])).collect();
        let v2_cr: Vec<f64> = v2_c.iter().map(|v| f64::from(v[1])).collect();
        v2_sums.push(cert_fullband(tier, &v2_cl, &v2_cr));

        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let v2_h: Vec<[f32; 2]> = holdout_dry
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let v2_hl: Vec<f64> = v2_h.iter().map(|v| f64::from(v[0])).collect();
        let v2_hr: Vec<f64> = v2_h.iter().map(|v| f64::from(v[1])).collect();
        v2_holds[t_idx] = eval_hold(&v2_hl, &v2_hr);

        let mut inst = V3CoreInstance::<DefaultMathBackend>::from_cold_model(&v3_models[t_idx]);
        let (mut o3_l, mut o3_r) = (vec![0.0_f32; n_eval], vec![0.0_f32; n_eval]);
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(
            &eval_in_l,
            if mono_only { &eval_in_l } else { &eval_in_r },
            &mut o3_l,
            &mut o3_r,
        );
        let v3_cl: Vec<f64> = o3_l.iter().map(|&v| f64::from(v)).collect();
        let v3_cr: Vec<f64> = o3_r.iter().map(|&v| f64::from(v)).collect();
        v3_sums.push(cert_fullband(tier, &v3_cl, &v3_cr));

        let (mut h3_l, mut h3_r) = (
            vec![0.0_f32; hold_in_l.len()],
            vec![0.0_f32; hold_in_r.len()],
        );
        inst.reset_to_dc_equilibrium();
        inst.process_block_stereo(
            &hold_in_l,
            if mono_only { &hold_in_l } else { &hold_in_r },
            &mut h3_l,
            &mut h3_r,
        );
        let v3_hl: Vec<f64> = h3_l.iter().map(|&v| f64::from(v)).collect();
        let v3_hr: Vec<f64> = h3_r.iter().map(|&v| f64::from(v)).collect();
        v3_holds[t_idx] = eval_hold(&v3_hl, &v3_hr);
    }

    let v3_arr: [PerceptualCertificationSummary; 4] = v3_sums.clone().try_into().unwrap();
    let v3_monotone = verify_monotone_4tier_summaries(&v3_arr);
    for (t_idx, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        rows.push(CompareRow {
            label: label.to_string(),
            tier,
            v2_sum: v2_sums[t_idx].clone(),
            v3_sum: v3_sums[t_idx].clone(),
            v2_holdout_db: v2_holds[t_idx],
            v3_holdout_db: v3_holds[t_idx],
            v3_os2x: v3_models[t_idx].os2x,
            v2_ms,
            v3_ms,
            v3_monotone,
        });
    }
}

#[test]
fn test_v3_prototype_head_to_head_vs_v2() {
    let mut rows = Vec::with_capacity(16);
    benchmark_chain(
        "Chain-04 TubeAmp4x12",
        Chain04TubeAmpFmv4x12::new(),
        ArchetypeFamily::CrunchHighGainAmp,
        &mut rows,
    );
    benchmark_chain(
        "Chain-02 GatedFuzz",
        Chain02GatedAsymmetricFuzz::new(),
        ArchetypeFamily::FuzzOctaveChaos,
        &mut rows,
    );
    benchmark_chain(
        "Chain-06 EnvSVF",
        Chain06EnvelopeResonantSvf::new(),
        ArchetypeFamily::CrunchHighGainAmp,
        &mut rows,
    );
    benchmark_chain(
        "Chain-01 SymmetricOD",
        Chain01SymmetricOd::new(),
        ArchetypeFamily::CleanPedalPreamp,
        &mut rows,
    );

    println!("\n=== v2.0 vs v3 Prototype Head-to-Head Benchmark ===");
    println!(
        "| Chain | Tier | v2 raw bESR | v3 raw bESR | Delta (v3-v2) | v3 LF / MF / HF (dB) | v2 LTAS | v3 LTAS | v2 Hold | v3 Hold | OS2x | v2 ms | v3 ms | Mono |"
    );
    println!(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: |"
    );
    for r in &rows {
        let delta = r.v3_sum.raw_besr_db - r.v2_sum.raw_besr_db;
        println!(
            "| {} | {:?} | {:+.2} dB | {:+.2} dB | {:+.2} dB | {:+.1} / {:+.1} / {:+.1} | {:.2} | {:.2} | {:+.2} | {:+.2} | {} | {} | {} | {} |",
            r.label,
            r.tier,
            r.v2_sum.raw_besr_db,
            r.v3_sum.raw_besr_db,
            delta,
            r.v3_sum.band_esr_db[0],
            r.v3_sum.band_esr_db[1],
            r.v3_sum.band_esr_db[2],
            r.v2_sum.ltas_max_db,
            r.v3_sum.ltas_max_db,
            r.v2_holdout_db,
            r.v3_holdout_db,
            if r.v3_os2x { "2x" } else { "1x" },
            r.v2_ms,
            r.v3_ms,
            r.v3_monotone,
        );
    }

    for r in &rows {
        assert!(
            r.v3_sum.ltas_excluded_bands.iter().all(|&ex| !ex),
            "{} {:?} v3 unexpectedly excluded LTAS bands",
            r.label,
            r.tier
        );
        assert!(
            r.v2_sum.ltas_excluded_bands.iter().all(|&ex| !ex),
            "{} {:?} v2 unexpectedly excluded LTAS bands",
            r.label,
            r.tier
        );
        assert!(
            r.v3_monotone,
            "{} v3 failed strict 7-metric 4-tier monotonicity",
            r.label
        );
        if r.tier != Tier::Lite {
            let delta = r.v3_sum.raw_besr_db - r.v2_sum.raw_besr_db;
            if r.label == "Chain-01 SymmetricOD" {
                assert!(
                    delta <= 0.5,
                    "{} {:?} regressed raw bESR: v2={:.2}, v3={:.2}",
                    r.label,
                    r.tier,
                    r.v2_sum.raw_besr_db,
                    r.v3_sum.raw_besr_db
                );
            } else {
                assert!(
                    delta <= -4.0,
                    "{} {:?} did not beat v2 by >= 4.0 dB: v2={:.2}, v3={:.2} (delta={delta:+.2})",
                    r.label,
                    r.tier,
                    r.v2_sum.raw_besr_db,
                    r.v3_sum.raw_besr_db
                );
            }
            assert!(
                r.v3_holdout_db <= r.v2_holdout_db + 0.5,
                "{} {:?} regressed unseen holdout bESR: v2={:.2}, v3={:.2}",
                r.label,
                r.tier,
                r.v2_holdout_db,
                r.v3_holdout_db
            );
        }
    }
}
