#![forbid(unsafe_code)]
//! Shared benchmark harness, `BlackBoxModel` adapter, and excitation builders for `v3_prototype_benchmark`.

use std::collections::BTreeMap;

use dnam_capture::metrics::{PerceptualCertificationSummary, canonical_ltas_33_centers_hz};
use dnam_capture::proj::DnamProjManifest;
use dnam_capture::{
    DistillationTarget, StemResponse, StemStimulus, StimulusFrame, resample_arbitrary_rate,
    resample_control_trajectory, resampled_frame_len, scale_freeze_window,
};
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_fixtures::BlackBoxModel;

#[derive(Clone, Debug)]
pub struct MatrixAdapter<M: BlackBoxModel> {
    pub inner: M,
    pub switches: Vec<u8>,
}

impl<M: BlackBoxModel> MatrixAdapter<M> {
    pub fn from_model(inner: M) -> Self {
        let switches = inner.metadata().default_switches();
        Self { inner, switches }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_internal(
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

pub fn manifest_for_fixture<M: BlackBoxModel>(model: &M, label_suffix: &str) -> DnamProjManifest {
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

pub fn combine_stereo_summary(
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

pub fn build_unseen_holdout_excitation(n: usize, sr_hz: f64) -> Vec<[f32; 2]> {
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

pub fn build_identification_sweep_excitation(n: usize, sr_hz: f64) -> Vec<[f32; 2]> {
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

pub fn build_fullband_eval_excitation(n: usize, sr_hz: f64) -> Vec<[f32; 2]> {
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
