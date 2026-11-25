#![forbid(unsafe_code)]
//! Generic [`DistillationTarget`] adapter, manifest builders, and empirical repeatability
//! measurement over arbitrary [`BlackBoxModel`] implementations (`0` fixture heuristics).

use std::collections::BTreeMap;

use dnam_capture::metrics::repeatability::{
    TargetRepeatabilityProfile, build_repeatability_probe_stream,
};
use dnam_capture::proj::DnamProjManifest;
use dnam_capture::realign_fractional_kaiser64;
use dnam_capture::stimulus::TopologyProbeMode;
use dnam_capture::{
    DistillationTarget, StemResponse, StemStimulus, StimulusFrame, resample_arbitrary_rate,
    resample_control_trajectory, resampled_frame_len, scale_freeze_window,
};
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_fixtures::BlackBoxModel;

/// Derives canonical [`TopologyProbeMode`] purely from input/output channel counts.
#[must_use]
pub(crate) fn infer_topology_mode<M: BlackBoxModel>(model: &M) -> TopologyProbeMode {
    match (model.num_inputs() >= 2, model.num_outputs() >= 2) {
        (true, true) => TopologyProbeMode::TrueStereo2To2,
        (true, false) => TopologyProbeMode::DualInToMono2To1,
        (false, true) => TopologyProbeMode::MonoToStereo1To2,
        (false, false) => TopologyProbeMode::Mono1To1,
    }
}

struct AdapterRenderReq<'a> {
    stem_id: String,
    stim_sr: u32,
    in_1: &'a [f32],
    in_2: Option<&'a [f32]>,
    in_3: Option<&'a [f32]>,
    in_4: Option<&'a [f32]>,
    control_values: &'a [f32],
    control_trajectories: Option<&'a Vec<Vec<f32>>>,
    freeze_window: Option<[usize; 2]>,
}

/// Generic sample-rate-adapting wrapper turning any [`BlackBoxModel`] into a [`DistillationTarget`].
#[derive(Clone, Debug)]
pub(crate) struct FixtureDistillationAdapter<M: BlackBoxModel> {
    inner: M,
    switches: Vec<u8>,
}

impl<M: BlackBoxModel> FixtureDistillationAdapter<M> {
    /// Wraps `inner` initialized to its default switch positions.
    #[must_use]
    pub(crate) fn from_model(inner: M) -> Self {
        let switches = inner.metadata().default_switches();
        Self { inner, switches }
    }

    fn render_internal(&mut self, req: AdapterRenderReq<'_>) -> Result<StemResponse, DnamError> {
        let dev_sr = self.inner.sample_rate_hz();
        let orig_n = req.in_1.len();
        if orig_n == 0 {
            let n_out = self.inner.num_outputs().clamp(1, 4);
            return Ok(StemResponse {
                stem_id: req.stem_id,
                sample_rate_hz: req.stim_sr,
                out_1: Vec::new(),
                out_2: (n_out >= 2).then(Vec::new),
                out_3: (n_out >= 3).then(Vec::new),
                out_4: (n_out >= 4).then(Vec::new),
                internal_nodes: BTreeMap::new(),
            });
        }
        let needs_resample = req.stim_sr != dev_sr;
        let dev_n = if needs_resample {
            resampled_frame_len(orig_n, req.stim_sr, dev_sr)
        } else {
            orig_n
        };
        let rs_in1;
        let rs_in2;
        let rs_in3;
        let rs_in4;
        let rs_trajs;
        let (dev_in1, dev_in2, dev_in3, dev_in4, dev_trajs, dev_freeze) = if needs_resample {
            rs_in1 = resample_arbitrary_rate(req.in_1, req.stim_sr, dev_sr, Some(dev_n));
            rs_in2 = req
                .in_2
                .map(|ch| resample_arbitrary_rate(ch, req.stim_sr, dev_sr, Some(dev_n)));
            rs_in3 = req
                .in_3
                .map(|ch| resample_arbitrary_rate(ch, req.stim_sr, dev_sr, Some(dev_n)));
            rs_in4 = req
                .in_4
                .map(|ch| resample_arbitrary_rate(ch, req.stim_sr, dev_sr, Some(dev_n)));
            rs_trajs = req.control_trajectories.map(|trajs| {
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
                req.freeze_window
                    .map(|w| scale_freeze_window(w, orig_n, dev_n)),
            )
        } else {
            (
                req.in_1,
                req.in_2,
                req.in_3,
                req.in_4,
                req.control_trajectories,
                req.freeze_window,
            )
        };

        let meta = self.inner.metadata();
        let default_controls = meta.default_controls();
        let freeze_sw_idx = meta
            .switches
            .iter()
            .position(|sw| sw.id == "freeze")
            .unwrap_or(0);
        let n_sw = self
            .switches
            .len()
            .max(meta.switches.len())
            .max(if dev_freeze.is_some() {
                freeze_sw_idx + 1
            } else {
                0
            });
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
                let fb = req
                    .control_values
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
                    .or_else(|| req.control_values.get(n_range + k).copied())
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
                resample_arbitrary_rate(&dev_out1, dev_sr, req.stim_sr, Some(orig_n)),
                dev_out2.map(|c| resample_arbitrary_rate(&c, dev_sr, req.stim_sr, Some(orig_n))),
                dev_out3.map(|c| resample_arbitrary_rate(&c, dev_sr, req.stim_sr, Some(orig_n))),
                dev_out4.map(|c| resample_arbitrary_rate(&c, dev_sr, req.stim_sr, Some(orig_n))),
            )
        } else {
            (dev_out1, dev_out2, dev_out3, dev_out4)
        };
        Ok(StemResponse {
            stem_id: req.stem_id,
            sample_rate_hz: req.stim_sr,
            out_1,
            out_2,
            out_3,
            out_4,
            internal_nodes: BTreeMap::new(),
        })
    }
}

impl<M: BlackBoxModel> DistillationTarget for FixtureDistillationAdapter<M> {
    fn reset_state(&mut self) {
        self.inner.reset();
    }

    fn render_stem(&mut self, s: &StemStimulus) -> Result<StemResponse, DnamError> {
        self.reset_state();
        self.render_internal(AdapterRenderReq {
            stem_id: s.stem_id.clone(),
            stim_sr: s.sample_rate_hz,
            in_1: &s.in_1,
            in_2: s.in_2.as_deref(),
            in_3: s.in_3.as_deref(),
            in_4: s.in_4.as_deref(),
            control_values: &s.control_values,
            control_trajectories: s.control_trajectories.as_ref(),
            freeze_window: s.freeze_transition_window,
        })
    }

    fn render_frame(&mut self, f: &StimulusFrame) -> Result<StemResponse, DnamError> {
        self.render_internal(AdapterRenderReq {
            stem_id: format!("frame_{}", f.frame_index),
            stim_sr: f.sample_rate_hz,
            in_1: &f.in_1,
            in_2: f.in_2.as_deref(),
            in_3: f.in_3.as_deref(),
            in_4: f.in_4.as_deref(),
            control_values: &f.control_values,
            control_trajectories: f.control_trajectories.as_ref(),
            freeze_window: None,
        })
    }
}

fn base_manifest_for_model<M: BlackBoxModel>(model: &M, tag: &str) -> DnamProjManifest {
    let meta = model.metadata();
    let mut manifest = DnamProjManifest::default_reference();
    manifest.base.metadata.name = format!("{} ({tag})", meta.name);
    manifest.base.io_config.inputs = model.num_inputs().clamp(1, 2) as u8;
    manifest.base.io_config.output_mode = if model.num_outputs() >= 2 {
        "stereo".to_string()
    } else {
        "mono".to_string()
    };
    manifest
}

/// Builds a fixed-setting (`controls_schema = []`) [`DnamProjManifest`] for `L1`/`L2` qualification.
#[must_use]
pub(crate) fn build_star_manifest_for_model<M: BlackBoxModel>(model: &M) -> DnamProjManifest {
    let mut manifest = base_manifest_for_model(model, "StarCenter");
    manifest.base.controls_schema = Vec::new();
    manifest
}

/// Builds a parametric (`range` + `switch`) [`DnamProjManifest`] for `L3` qualification.
#[must_use]
pub(crate) fn build_parametric_manifest_for_model<M: BlackBoxModel>(model: &M) -> DnamProjManifest {
    let meta = model.metadata();
    let mut manifest = base_manifest_for_model(model, "Parametric");
    let mut schema = Vec::with_capacity(meta.controls.len() + meta.switches.len());
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

/// Measures empirical `[L, R]` repeatability profiles strictly from `(x, y)` probe renders at `ctrls`.
pub(crate) fn measure_empirical_repeatability_at_setting<M: BlackBoxModel>(
    adapter: &mut FixtureDistillationAdapter<M>,
    topology: TopologyProbeMode,
    ctrls: &[f32],
    align: (f64, f64, bool, bool),
) -> [TargetRepeatabilityProfile; 2] {
    let (probe_dry, probe_windows, late_tail_window) = build_repeatability_probe_stream(48_000.0);
    let in_1: Vec<f32> = probe_dry.iter().map(|lr| lr[0]).collect();
    let need_stereo_in = matches!(
        topology,
        TopologyProbeMode::TrueStereo2To2 | TopologyProbeMode::DualInToMono2To1
    );
    let in_2 = need_stereo_in.then(|| probe_dry.iter().map(|lr| lr[1]).collect());
    let resp = adapter
        .render_stem(&StemStimulus {
            stem_id: "qualify_repeatability_probe".to_string(),
            sample_rate_hz: 48_000,
            in_1,
            in_2,
            in_3: None,
            in_4: None,
            control_values: ctrls.to_vec(),
            control_trajectories: None,
            freeze_transition_window: None,
        })
        .unwrap();
    let (lag, ppm, inv_l, inv_r) = align;
    let al_l = realign_fractional_kaiser64(&resp.out_1, lag, ppm, inv_l);
    let wl: Vec<f64> = al_l.into_iter().map(f64::from).collect();
    let prof_l = TargetRepeatabilityProfile::from_probe_response(
        &wl,
        &probe_windows,
        late_tail_window,
        48_000.0,
    );
    let prof_r = resp.out_2.as_deref().map_or(prof_l, |o2| {
        let al_r = realign_fractional_kaiser64(o2, lag, ppm, inv_r);
        let wr: Vec<f64> = al_r.into_iter().map(f64::from).collect();
        TargetRepeatabilityProfile::from_probe_response(
            &wr,
            &probe_windows,
            late_tail_window,
            48_000.0,
        )
    });
    [prof_l, prof_r]
}
