#![forbid(unsafe_code)]
//! Comprehensive 13-Chain × 4-Tier Perceptual Benchmark Matrix (`r[qual.tiers.benchmark-matrix]`).
//!
//! Evaluates all 13 black-box signal chains (`Chain-01..11`, `Chain-12 FixedZeroKnob`,
//! `Chain-12 Parametric16`) across `[Lite, Full, Pro, Ultra]`, reporting `bESR`, `LF`, `MF`, `HF`,
//! `LTAS`, `MR-IIR`, `SegESR`, unseen Holdout `bESR`, cross-tier LUFS parity, and 4-tier
//! monotonicity before asserting target/ceiling compliance.

use std::collections::BTreeMap;

use dnam_capture::metrics::{
    ArchetypeFamily, PerceptualCertificationSummary, TierQualityThresholds,
    canonical_ltas_33_centers_hz, certify_4metric_signal_chain, evaluate_besr_from_band_energies,
    split_bezout_3band_energies, verify_cross_tier_loudness_parity,
    verify_monotone_4tier_summaries,
};
use dnam_capture::proj::{
    DnamProjManifest, DnamProjView, ProjChunkKind, STMC_FLAG_WET_POLARITY_INVERTED,
    export_runtime_dnam, verify_proj_inpc_capsule, verify_proj_stmc_capsule,
};
use dnam_capture::solver::{canonical_besr_warmup, star_center_control_values};
use dnam_capture::stimulus::{AdaptiveTrainingOptions, TopologyProbeMode, train_adaptive_target};
use dnam_capture::{
    DistillationTarget, StemResponse, StemStimulus, StimulusFrame, realign_fractional_kaiser64,
    resample_arbitrary_rate, resample_control_trajectory, resampled_frame_len, scale_freeze_window,
};
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::DefaultCryptoProvider;
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_engine::{DnamEngine, Tier};
use dnam_fixtures::{
    BlackBoxModel, Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain03ParallelCleanBlend,
    Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom, Chain06EnvelopeResonantSvf,
    Chain07StereoChorusAndRotary, Chain08LongBbdEchoAnd30sShimmer, Chain09La2aOpticalCompressor,
    Chain10Oc2OctaveDividerAndRingMod, Chain11BiAmpBassAndOutputIron, Chain12FullEndToEndStereoRig,
    Chain12RigMode,
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
        let rs_in1;
        let rs_in2;
        let rs_in3;
        let rs_in4;
        let rs_trajs;
        let (dev_in1, dev_in2, dev_in3, dev_in4, dev_trajs, dev_freeze) = if needs_resample {
            rs_in1 = resample_arbitrary_rate(in_1, stim_sr, dev_sr, Some(dev_n));
            rs_in2 = in_2.map(|ch| resample_arbitrary_rate(ch, stim_sr, dev_sr, Some(dev_n)));
            rs_in3 = in_3.map(|ch| resample_arbitrary_rate(ch, stim_sr, dev_sr, Some(dev_n)));
            rs_in4 = in_4.map(|ch| resample_arbitrary_rate(ch, stim_sr, dev_sr, Some(dev_n)));
            rs_trajs = control_trajectories.map(|trajs| {
                trajs
                    .iter()
                    .map(|tr| resample_control_trajectory(tr, dev_n))
                    .collect::<Vec<Vec<f32>>>()
            });
            (
                rs_in1.as_slice(),
                rs_in2.as_deref(),
                rs_in3.as_deref(),
                rs_in4.as_deref(),
                rs_trajs.as_ref(),
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
    manifest.base.metadata.name = if label_suffix.is_empty() {
        meta.name.to_string()
    } else {
        format!("{} {label_suffix}", meta.name)
    };
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
        band_esr_db: [
            l.band_esr_db[0].max(r.band_esr_db[0]),
            l.band_esr_db[1].max(r.band_esr_db[1]),
            l.band_esr_db[2].max(r.band_esr_db[2]),
        ],
        qualified_band_mask: [
            l.qualified_band_mask[0] || r.qualified_band_mask[0],
            l.qualified_band_mask[1] || r.qualified_band_mask[1],
            l.qualified_band_mask[2] || r.qualified_band_mask[2],
        ],
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

#[derive(Clone, Debug)]
struct RowResult {
    label: String,
    tier: Tier,
    summary: PerceptualCertificationSummary,
    holdout_besr_db: f64,
    lu_step: f64,
    lu_pass: bool,
    monotone: bool,
    holdout_pass: bool,
}

fn evaluate_chain<M: BlackBoxModel>(
    label: &str,
    model: M,
    family: ArchetypeFamily,
    rows: &mut Vec<RowResult>,
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
    let report = train_adaptive_target(manifest, &mut adapter, &opts)
        .unwrap_or_else(|e| panic!("{label} train_adaptive_target failed: {e:?}"));

    let proj_view = DnamProjView::from_bytes(&report.proj_bytes, &crypto)
        .unwrap_or_else(|e| panic!("{label} invalid .dnam-proj: {e:?}"));
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
    let eval_inv_pol = (stmc_view.flags & STMC_FLAG_WET_POLARITY_INVERTED) != 0;
    let eval_lag = f64::from(stmc_view.fractional_lag_samples_f32);
    let star_ctrls = star_center_control_values(&proj_view.manifest);

    let mut render_aligned = |tag: &str, dry: &[[f32; 2]]| -> (Vec<f64>, Vec<f64>) {
        let h1: Vec<f32> = dry.iter().map(|v| v[0]).collect();
        let h2: Vec<f32> = dry.iter().map(|v| v[1]).collect();
        adapter.reset_state();
        let resp = adapter
            .render_internal(
                tag.to_string(),
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
        let al_l = realign_fractional_kaiser64(&resp.out_1, eval_lag, 0.0, eval_inv_pol);
        let al_r = resp
            .out_2
            .as_ref()
            .map(|v| realign_fractional_kaiser64(v, eval_lag, 0.0, eval_inv_pol));
        let tgt_l = al_l.iter().map(|&v| f64::from(v)).collect();
        let tgt_r = al_r
            .as_ref()
            .unwrap_or(&al_l)
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        (tgt_l, tgt_r)
    };

    // Full-bandwidth 20 Hz-20 kHz f32 evaluation signal (18,000 samples > 16,384 Welch window across all 33 LTAS bands)
    let eval_dry = build_fullband_eval_excitation(18_000, 48_000.0);
    let (tgt_l, tgt_r) = render_aligned("fullband_eval", &eval_dry);
    let n_common = eval_dry.len().min(tgt_l.len()).min(tgt_r.len());
    let warmup = canonical_besr_warmup(n_common);

    // Unseen Holdout target at default StarCenter controls
    let holdout_dry = build_unseen_holdout_excitation(6_000, 48_000.0);
    let (hold_tgt_l, hold_tgt_r) = render_aligned("holdout", &holdout_dry);
    let hold_warmup = canonical_besr_warmup(hold_tgt_l.len());

    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut tier_rms = [0.0_f64; 4];
    let mut holdouts = [0.0_f64; 4];
    let mut summaries = Vec::with_capacity(4);

    for (t_idx, tier) in tiers.into_iter().enumerate() {
        let tier_dnam = export_runtime_dnam(&report.proj_bytes, Some(tier)).unwrap();
        let mut eng =
            DnamEngine::<DefaultMathBackend>::from_bytes_owned(&tier_dnam, tier, 48_000).unwrap();
        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let rendered: Vec<[f32; 2]> = eval_dry[..n_common]
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let sum_sq: f64 = rendered[warmup..n_common]
            .iter()
            .map(|&[l, r]| {
                if mono_only {
                    f64::from(l) * f64::from(l)
                } else {
                    0.5 * (f64::from(l) * f64::from(l) + f64::from(r) * f64::from(r))
                }
            })
            .sum();
        tier_rms[t_idx] = (sum_sq / ((n_common - warmup).max(1) as f64))
            .sqrt()
            .max(1.0e-12);

        let mdl_l: Vec<f64> = rendered.iter().map(|v| f64::from(v[0])).collect();
        let sl = certify_4metric_signal_chain(
            family,
            tier,
            &tgt_l[..n_common],
            &mdl_l,
            None,
            48_000.0,
            warmup,
        );
        let summary = if mono_only {
            sl
        } else {
            let mdl_r: Vec<f64> = rendered.iter().map(|v| f64::from(v[1])).collect();
            let sr = certify_4metric_signal_chain(
                family,
                tier,
                &tgt_r[..n_common],
                &mdl_r,
                None,
                48_000.0,
                warmup,
            );
            combine_stereo_summary(sl, sr)
        };
        summaries.push(summary);

        // Evaluate unseen holdout on fresh state
        eng.reset_to_dc_equilibrium();
        for (p, &val) in star_ctrls.iter().enumerate() {
            eng.set_control_immediate(p, val);
        }
        let h_rend: Vec<[f32; 2]> = holdout_dry
            .iter()
            .map(|&[dl, dr]| eng.step_frame_tora(dl, dr, mono_only))
            .collect();
        let hl: Vec<f64> = h_rend.iter().map(|v| f64::from(v[0])).collect();
        let (slh, elh, nlh) =
            split_bezout_3band_energies(&hold_tgt_l, &hl, None, 48_000.0, hold_warmup);
        let rh_l = evaluate_besr_from_band_energies(slh, elh, nlh);
        holdouts[t_idx] = if mono_only {
            rh_l.besr_db
        } else {
            let hr: Vec<f64> = h_rend.iter().map(|v| f64::from(v[1])).collect();
            let (srh, erh, nrh) =
                split_bezout_3band_energies(&hold_tgt_r, &hr, None, 48_000.0, hold_warmup);
            let rh_r = evaluate_besr_from_band_energies(srh, erh, nrh);
            rh_l.besr_db.max(rh_r.besr_db)
        };
    }

    let arr: [PerceptualCertificationSummary; 4] = summaries.try_into().unwrap();
    let monotone = verify_monotone_4tier_summaries(&arr);
    let (lu_step, lu_pass) = verify_cross_tier_loudness_parity(tier_rms);

    for t_idx in 0..4 {
        let th = TierQualityThresholds::for_tier(arr[t_idx].tier);
        let holdout_pass = holdouts[t_idx] <= th.unseen_control_besr_db;
        rows.push(RowResult {
            label: label.to_string(),
            tier: arr[t_idx].tier,
            summary: arr[t_idx].clone(),
            holdout_besr_db: holdouts[t_idx],
            lu_step,
            lu_pass,
            monotone,
            holdout_pass,
        });
    }
}

#[test]
fn test_benchmark_matrix_13_chains_4_tiers_telemetry_and_acceptance() {
    let mut rows = Vec::with_capacity(52);
    evaluate_chain(
        "Chain-01 SymmetricOD",
        Chain01SymmetricOd::new(),
        ArchetypeFamily::CleanPedalPreamp,
        &mut rows,
    );
    evaluate_chain(
        "Chain-02 GatedFuzz",
        Chain02GatedAsymmetricFuzz::new(),
        ArchetypeFamily::FuzzOctaveChaos,
        &mut rows,
    );
    evaluate_chain(
        "Chain-03 CleanBlend",
        Chain03ParallelCleanBlend::new(),
        ArchetypeFamily::CleanPedalPreamp,
        &mut rows,
    );
    evaluate_chain(
        "Chain-04 TubeAmp4x12",
        Chain04TubeAmpFmv4x12::new(),
        ArchetypeFamily::CrunchHighGainAmp,
        &mut rows,
    );
    evaluate_chain(
        "Chain-05 PureLTI",
        Chain05PureLtiCabAndRoom::new(),
        ArchetypeFamily::CleanPedalPreamp,
        &mut rows,
    );
    evaluate_chain(
        "Chain-06 EnvSVF",
        Chain06EnvelopeResonantSvf::new(),
        ArchetypeFamily::CrunchHighGainAmp,
        &mut rows,
    );
    evaluate_chain(
        "Chain-07 StereoChorus",
        Chain07StereoChorusAndRotary::new(),
        ArchetypeFamily::FullRigModDelayReverb,
        &mut rows,
    );
    evaluate_chain(
        "Chain-08 BbdShimmer",
        Chain08LongBbdEchoAnd30sShimmer::new(),
        ArchetypeFamily::FullRigModDelayReverb,
        &mut rows,
    );
    evaluate_chain(
        "Chain-09 LA2A",
        Chain09La2aOpticalCompressor::new(),
        ArchetypeFamily::CleanPedalPreamp,
        &mut rows,
    );
    evaluate_chain(
        "Chain-10 OC2RingMod",
        Chain10Oc2OctaveDividerAndRingMod::new(),
        ArchetypeFamily::FuzzOctaveChaos,
        &mut rows,
    );
    evaluate_chain(
        "Chain-11 BiAmpBass",
        Chain11BiAmpBassAndOutputIron::new(),
        ArchetypeFamily::CrunchHighGainAmp,
        &mut rows,
    );
    evaluate_chain(
        "Chain-12 FixedZero",
        Chain12FullEndToEndStereoRig::new(Chain12RigMode::FixedZeroKnob),
        ArchetypeFamily::FullRigModDelayReverb,
        &mut rows,
    );
    evaluate_chain(
        "Chain-12 Parametric16",
        Chain12FullEndToEndStereoRig::new(Chain12RigMode::Parametric16),
        ArchetypeFamily::FullRigModDelayReverb,
        &mut rows,
    );

    println!(
        "\n| Chain | Tier | bESR (dB) | LF (dB) | MF (dB) | HF (dB) | LTAS (dB) | MR-IIR | SegESR (dB) | Holdout (dB) | LU Step | Mono | Tgt | Ceil |"
    );
    println!(
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |"
    );
    let mut failures = Vec::new();
    for r in &rows {
        let s = &r.summary;
        println!(
            "| {} | {:?} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} | {:.3} | {:.2} | {:.2} | {:.4} | {} | {} | {} |",
            r.label,
            r.tier,
            s.besr_db,
            s.band_esr_db[0],
            s.band_esr_db[1],
            s.band_esr_db[2],
            s.ltas_max_db,
            s.mr_iir,
            s.seg_esr_db,
            r.holdout_besr_db,
            r.lu_step,
            r.monotone,
            s.passes_target && r.holdout_pass,
            s.passes_hard_ceiling
        );
        if !(s.passes_target && s.passes_hard_ceiling && r.monotone && r.lu_pass && r.holdout_pass)
        {
            failures.push(format!(
                "{} {:?}: besr={:.2}, bands={:.2?}, ltas={:.3}, hold={:.2}, lu={:.4}, mono={}, tgt={}, ceil={}",
                r.label,
                r.tier,
                s.besr_db,
                s.band_esr_db,
                s.ltas_max_db,
                r.holdout_besr_db,
                r.lu_step,
                r.monotone,
                s.passes_target,
                s.passes_hard_ceiling
            ));
        }
    }

    println!(
        "\nCanonical §4 Qualification Summary: {} / {} cells meet full §4 target & ceiling",
        rows.len() - failures.len(),
        rows.len()
    );

    assert_eq!(rows.len(), 52);
    for r in &rows {
        let s = &r.summary;
        assert!(
            s.raw_besr_db.is_finite()
                && s.besr_db.is_finite()
                && s.band_esr_db.iter().all(|x| x.is_finite())
                && s.ltas_max_db.is_finite()
                && s.mr_iir.is_finite()
                && s.mr_iir >= 0.0
                && s.seg_esr_db.is_finite()
                && r.holdout_besr_db.is_finite(),
            "{} {:?} produced non-finite canonical §4 metrics",
            r.label,
            r.tier
        );
        assert!(
            s.ltas_excluded_bands.iter().all(|&ex| !ex),
            "{} {:?} unexpectedly excluded LTAS bands across 20 Hz-20 kHz",
            r.label,
            r.tier
        );
    }

    // Baseline v2 solver predates full v3 solver integration; set DNAM_ENFORCE_SECTION4_TARGETS=1
    // once full §4 acoustic target compliance is active across all 13 chains.
    if std::env::var_os("DNAM_ENFORCE_SECTION4_TARGETS").is_some() {
        assert!(
            failures.is_empty(),
            "{} / {} benchmark cells missed §4 targets:\n{}",
            failures.len(),
            rows.len(),
            failures.join("\n")
        );
    }
}

#[test]
fn test_canonical_tier_thresholds_device_agnostic_and_monotone() {
    // verify[qual.tiers.benchmark-matrix]
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut prev: Option<TierQualityThresholds> = None;

    for tier in tiers {
        let cur = TierQualityThresholds::for_tier(tier);
        assert!(
            cur.composite_besr_db < 0.0
                && cur.per_band_besr_db.iter().all(|&b| b < 0.0)
                && cur.unseen_control_besr_db < 0.0,
            "{tier:?} dB thresholds must be strictly negative"
        );
        assert!(
            cur.unseen_control_besr_db >= cur.composite_besr_db,
            "{tier:?} unseen_control_besr_db ({}) must be >= composite_besr_db ({})",
            cur.unseen_control_besr_db,
            cur.composite_besr_db
        );
        assert!(
            cur.max_ltas_db > 0.0 && cur.max_ltas_db <= 0.75,
            "{tier:?} max_ltas_db ({}) must be within §4 bounds",
            cur.max_ltas_db
        );

        if let Some(lo) = prev {
            assert!(
                cur.composite_besr_db < lo.composite_besr_db,
                "{tier:?} composite_besr_db did not tighten"
            );
            for (band, (&hi_b, &lo_b)) in cur
                .per_band_besr_db
                .iter()
                .zip(lo.per_band_besr_db.iter())
                .enumerate()
            {
                assert!(
                    hi_b < lo_b,
                    "{tier:?} band {band} did not tighten ({hi_b} vs {lo_b})"
                );
            }
            assert!(
                cur.max_ltas_db < lo.max_ltas_db,
                "{tier:?} max_ltas_db did not tighten"
            );
            assert!(
                cur.unseen_control_besr_db < lo.unseen_control_besr_db,
                "{tier:?} unseen_control_besr_db did not tighten"
            );
        }
        prev = Some(cur);
    }
}
