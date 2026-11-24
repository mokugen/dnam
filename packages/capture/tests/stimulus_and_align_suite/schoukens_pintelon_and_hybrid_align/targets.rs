#![forbid(unsafe_code)]
//! Lightweight synthetic `DistillationTarget` fixtures and manifest builders for M24
//! Two-Tier `Schoukens–Pintelon` and Hybrid Sub-Sample/PPM alignment verification.

use std::collections::BTreeMap;

use dnam_capture::align::realign_fractional_kaiser64;
use dnam_capture::proj::DnamProjManifest;
use dnam_capture::{DistillationTarget, StemResponse, StemStimulus};
use dnam_engine::schema::{ControlSchemaDef, DnamError};
use dnam_fixtures::{BlackBoxModel, Chain01SymmetricOd, FixtureBitDepth, FixtureDeviceMetadata};

#[derive(Default)]
pub struct IdentityWire;

impl BlackBoxModel for IdentityWire {
    fn reset(&mut self) {}
    fn metadata(&self) -> FixtureDeviceMetadata {
        Chain01SymmetricOd::new().metadata()
    }
    fn set_sample_rate_hz(&mut self, _: u32) {}
    fn set_bit_depth(&mut self, _: FixtureBitDepth) {}
    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn process_frame(&mut self, inputs: &[f32], outputs: &mut [f32], _: &[f32], _: &[u8]) {
        if !outputs.is_empty() && !inputs.is_empty() {
            outputs[0] = inputs[0];
        }
    }
}

pub fn make_range_control(id: &str, min: f32, max: f32, default: f32) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "range",
        "lobatto_degree": 2,
        "declared_range": [min, max],
        "safe_clamped_range": [min, max],
        "default": default
    }))
    .unwrap()
}

pub fn make_switch_control(id: &str, opts: &[&str], def_opt: &str) -> ControlSchemaDef {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "name": id,
        "type": "switch",
        "declared_options": opts,
        "active_options": opts,
        "default": def_opt
    }))
    .unwrap()
}

/// Converter-delay + PPM clock-skew + polarity + optional DC/noise/LTV modulation target.
pub struct ConverterRigTarget {
    pub lag_samples: f64,
    pub clock_drift_ppm: f64,
    pub invert_l: bool,
    pub invert_r: bool,
    pub stereo_out: bool,
    pub dc_offset: f32,
    pub ltv_mod_depth: f64,
    pub tail_t60_s: Option<f64>,
    pub reverb_tail: Vec<f32>,
}

impl Default for ConverterRigTarget {
    fn default() -> Self {
        Self {
            lag_samples: 0.0,
            clock_drift_ppm: 0.0,
            invert_l: false,
            invert_r: false,
            stereo_out: false,
            dc_offset: 0.0,
            ltv_mod_depth: 0.0,
            tail_t60_s: None,
            reverb_tail: Vec::new(),
        }
    }
}

impl DistillationTarget for ConverterRigTarget {
    fn reset_state(&mut self) {
        self.reverb_tail.clear();
    }

    fn render_stem(&mut self, stim: &StemStimulus) -> Result<StemResponse, DnamError> {
        let n = stim.in_1.len();
        let sr = f64::from(stim.sample_rate_hz.max(1));
        let ctrl_gain = stim
            .control_values
            .first()
            .map_or(1.0_f64, |&c| 0.5 + 0.5 * f64::from(c));
        let mut proc_l = Vec::with_capacity(n);
        let mut rv_state = 0.0_f64;
        let rv_pole = self
            .tail_t60_s
            .map_or(0.0, |t60| (-6.907_755_278_982_137 / (t60 * sr)).exp());

        for (i, &x) in stim.in_1.iter().enumerate() {
            let g = stim
                .control_trajectories
                .as_ref()
                .and_then(|tr| tr.first())
                .and_then(|v| v.get(i))
                .map_or(ctrl_gain, |&c| 0.5 + 0.5 * f64::from(c));
            let ltv = if self.ltv_mod_depth > 0.0 {
                1.0 + self.ltv_mod_depth * (std::f64::consts::TAU * 2.1 * (i as f64) / sr).sin()
            } else {
                1.0
            };
            let base = 0.72 * g * ltv * f64::from(x) + 0.03 * f64::from(x).powi(3);
            let y = if rv_pole > 0.0 {
                rv_state = rv_pole * rv_state + 0.03 * base;
                0.70 * base + 0.30 * rv_state
            } else {
                base
            };
            proc_l.push(y as f32);
        }

        let out_1_shifted = realign_fractional_kaiser64(
            &proc_l,
            -self.lag_samples,
            -self.clock_drift_ppm,
            self.invert_l,
        );
        let out_1: Vec<f32> = out_1_shifted
            .into_iter()
            .map(|v| v + self.dc_offset)
            .collect();

        let out_2 = if self.stereo_out {
            let src_r = stim.in_2.as_deref().unwrap_or(&stim.in_1);
            let proc_r: Vec<f32> = src_r
                .iter()
                .enumerate()
                .map(|(i, &xr)| {
                    let xl = stim.in_1[i];
                    let mix = 0.85 * f64::from(xr) + 0.12 * f64::from(xl);
                    (0.72 * ctrl_gain * mix) as f32
                })
                .collect();
            let shifted_r = realign_fractional_kaiser64(
                &proc_r,
                -self.lag_samples,
                -self.clock_drift_ppm,
                self.invert_r,
            );
            Some(shifted_r.into_iter().map(|v| v + self.dc_offset).collect())
        } else {
            None
        };

        Ok(StemResponse {
            stem_id: stim.stem_id.clone(),
            sample_rate_hz: stim.sample_rate_hz,
            out_1,
            out_2,
            out_3: None,
            out_4: None,
            internal_nodes: BTreeMap::new(),
        })
    }
}

pub fn base_manifest(mono_only: bool) -> DnamProjManifest {
    let mut m = DnamProjManifest::default_reference();
    m.base.io_config.inputs = 1;
    m.base.io_config.output_mode = if mono_only {
        "mono".to_string()
    } else {
        "stereo".to_string()
    };
    m
}
