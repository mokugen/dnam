//! `Chain-13`: Studio Tri-Band VCA Compressor (`1 -> 1`, `4x`-oversampled `f64`).

#![forbid(unsafe_code)]

use crate::chains::{
    Oversampler4xF64, TptLr4Crossover2, const_phys_to_norm, flush_subnormal_f64,
    scale_alpha_from_192k, substep_rate_4x,
};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz,
};

const CHAIN13_LOW_XOVER_MIN_HZ: f32 = 50.0;
const CHAIN13_LOW_XOVER_MAX_HZ: f32 = 800.0;
const CHAIN13_LOW_XOVER_DEF_HZ: f32 = 250.0;
const CHAIN13_HIGH_XOVER_MIN_HZ: f32 = 800.0;
const CHAIN13_HIGH_XOVER_MAX_HZ: f32 = 6_000.0;
const CHAIN13_HIGH_XOVER_DEF_HZ: f32 = 1_800.0;
const CHAIN13_MIN_XOVER_SPACING: f64 = 1.25;
const CHAIN13_SOFT_CLIP_DRIVE: f64 = 0.85;
const CHAIN13_RATIO_OPTIONS: [&str; 4] = ["2:1", "4:1", "8:1", "12:1"];
const CHAIN13_RATIOS: [f64; 4] = [2.0, 4.0, 8.0, 12.0];
const CHAIN13_SLOPES: [f64; 4] = [
    1.0 - 1.0 / CHAIN13_RATIOS[0],
    1.0 - 1.0 / CHAIN13_RATIOS[1],
    1.0 - 1.0 / CHAIN13_RATIOS[2],
    1.0 - 1.0 / CHAIN13_RATIOS[3],
];

const CHAIN13_CONTROLS: [FixtureControlSpec; 8] = [
    FixtureControlSpec {
        id: "low_crossover",
        name: "Low / Mid Crossover",
        unit: "Hz",
        default_norm: const_phys_to_norm(
            CHAIN13_LOW_XOVER_DEF_HZ,
            CHAIN13_LOW_XOVER_MIN_HZ,
            CHAIN13_LOW_XOVER_MAX_HZ,
        ),
        min_physical: CHAIN13_LOW_XOVER_MIN_HZ,
        max_physical: CHAIN13_LOW_XOVER_MAX_HZ,
    },
    FixtureControlSpec {
        id: "high_crossover",
        name: "Mid / High Crossover",
        unit: "Hz",
        default_norm: const_phys_to_norm(
            CHAIN13_HIGH_XOVER_DEF_HZ,
            CHAIN13_HIGH_XOVER_MIN_HZ,
            CHAIN13_HIGH_XOVER_MAX_HZ,
        ),
        min_physical: CHAIN13_HIGH_XOVER_MIN_HZ,
        max_physical: CHAIN13_HIGH_XOVER_MAX_HZ,
    },
    FixtureControlSpec {
        id: "low_threshold",
        name: "Low Threshold",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -48.0,
        max_physical: 0.0,
    },
    FixtureControlSpec {
        id: "low_level",
        name: "Low Level",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -12.0,
        max_physical: 12.0,
    },
    FixtureControlSpec {
        id: "mid_threshold",
        name: "Mid Threshold",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -48.0,
        max_physical: 0.0,
    },
    FixtureControlSpec {
        id: "mid_level",
        name: "Mid Level",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -12.0,
        max_physical: 12.0,
    },
    FixtureControlSpec {
        id: "high_threshold",
        name: "High Threshold",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -48.0,
        max_physical: 0.0,
    },
    FixtureControlSpec {
        id: "high_level",
        name: "High Level",
        unit: "dB",
        default_norm: 0.0,
        min_physical: -12.0,
        max_physical: 12.0,
    },
];

const CHAIN13_SWITCHES: [FixtureSwitchSpec; 3] = [
    FixtureSwitchSpec {
        id: "low_ratio",
        name: "Low Ratio",
        options: &CHAIN13_RATIO_OPTIONS,
        default_index: 1,
    },
    FixtureSwitchSpec {
        id: "mid_ratio",
        name: "Mid Ratio",
        options: &CHAIN13_RATIO_OPTIONS,
        default_index: 1,
    },
    FixtureSwitchSpec {
        id: "high_ratio",
        name: "High Ratio",
        options: &CHAIN13_RATIO_OPTIONS,
        default_index: 1,
    },
];

const CHAIN13_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-13-TriBand-VCA-Compressor",
    name: "Studio Tri-Band VCA Compressor",
    category: "compressor",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 400.0,
    supports_tempo_sync: false,
    controls: &CHAIN13_CONTROLS,
    switches: &CHAIN13_SWITCHES,
};

const _: () = assert!(CHAIN13_CONTROLS.len() == 8);
const _: () = assert!(CHAIN13_SWITCHES.len() == 3);
const _: () =
    assert!(CHAIN13_RATIOS[0] < CHAIN13_RATIOS[1] && CHAIN13_RATIOS[2] < CHAIN13_RATIOS[3]);
const _: () =
    assert!(CHAIN13_SLOPES[0] < CHAIN13_SLOPES[1] && CHAIN13_SLOPES[2] < CHAIN13_SLOPES[3]);
const _: () =
    assert!(CHAIN13_CONTROLS[0].default_norm >= -1.0 && CHAIN13_CONTROLS[0].default_norm <= 1.0);
const _: () =
    assert!(CHAIN13_CONTROLS[1].default_norm >= -1.0 && CHAIN13_CONTROLS[1].default_norm <= 1.0);

#[inline]
fn vca_band_gain(env: f64, thresh_lin: f64, slope: f64) -> f64 {
    if env <= 1.0e-18 || env <= thresh_lin {
        return 1.0;
    }
    // Exact linear-domain RMS VCA law g = (thresh / env)^slope with smooth soft-knee margin.
    let knee_env = (env * env + (0.15 * thresh_lin) * (0.15 * thresh_lin)).sqrt();
    if knee_env <= thresh_lin {
        1.0
    } else {
        (thresh_lin / knee_env).powf(slope).clamp(0.05, 1.0)
    }
}

/// `Chain-13-TriBand-VCA-Compressor`: `1 -> 1` `4x`-oversampled phase-coherent `LR4` 3-band
/// VCA compressor (`low_crossover` / `high_crossover`, per-band threshold/ratio/level, soft-iron output).
#[derive(Clone, Debug)]
pub struct Chain13TriBandVcaCompressor {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    xover_hi: TptLr4Crossover2,
    xover_lo: TptLr4Crossover2,
    hi_ap: TptLr4Crossover2,
    env_sq: [f64; 3],
}

impl Default for Chain13TriBandVcaCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain13TriBandVcaCompressor {
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        Self {
            sample_rate_hz: clamp_sample_rate_hz(sample_rate_hz),
            bit_depth,
            os: Oversampler4xF64::default(),
            xover_hi: TptLr4Crossover2::default(),
            xover_lo: TptLr4Crossover2::default(),
            hi_ap: TptLr4Crossover2::default(),
            env_sq: [0.0; 3],
        }
    }

    #[inline]
    fn band_control_phys(idx: usize, controls: &[f32]) -> f64 {
        let spec = &CHAIN13_CONTROLS[idx];
        let norm = controls.get(idx).copied().unwrap_or(spec.default_norm);
        f64::from(spec.norm_to_physical(norm))
    }

    #[inline]
    pub fn step_sample(&mut self, x: f32, controls: &[f32], switches: &[u8]) -> f32 {
        let x_clean = self.bit_depth.quantize_sample(x);
        let fc_lo = Self::band_control_phys(0, controls);
        let fc_hi = Self::band_control_phys(1, controls).max(fc_lo * CHAIN13_MIN_XOVER_SPACING);

        let thresh_lin = [
            10.0_f64.powf(Self::band_control_phys(2, controls) / 20.0),
            10.0_f64.powf(Self::band_control_phys(4, controls) / 20.0),
            10.0_f64.powf(Self::band_control_phys(6, controls) / 20.0),
        ];
        let trim = [
            10.0_f64.powf(Self::band_control_phys(3, controls) / 20.0),
            10.0_f64.powf(Self::band_control_phys(5, controls) / 20.0),
            10.0_f64.powf(Self::band_control_phys(7, controls) / 20.0),
        ];
        let slopes = [0, 1, 2].map(|b| {
            let def_idx = CHAIN13_SWITCHES[b].default_index;
            let idx = usize::from(switches.get(b).copied().unwrap_or(def_idx).min(3));
            CHAIN13_SLOPES[idx]
        });

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let a_atk = [
            scale_alpha_from_192k(0.000_8, fs_4x),
            scale_alpha_from_192k(0.001_6, fs_4x),
            scale_alpha_from_192k(0.003_2, fs_4x),
        ];
        let a_rel = [
            scale_alpha_from_192k(0.000_045, fs_4x),
            scale_alpha_from_192k(0.000_080, fs_4x),
            scale_alpha_from_192k(0.000_140, fs_4x),
        ];

        let [y, _] = self
            .os
            .process_frame_4x(f64::from(x_clean), 0.0, |_, u, _| {
                let (lo_mid, hi_raw) = self.xover_hi.step(u, fc_hi, fs_4x);
                let (lo, mid) = self.xover_lo.step(lo_mid, fc_lo, fs_4x);
                let (hi_ap_l, hi_ap_h) = self.hi_ap.step(hi_raw, fc_lo, fs_4x);
                let hi = hi_ap_l + hi_ap_h;
                let bands = [lo, mid, hi];
                let mut sum = 0.0_f64;
                for b in 0..3 {
                    let sq = bands[b] * bands[b];
                    let alpha = if sq > self.env_sq[b] {
                        a_atk[b]
                    } else {
                        a_rel[b]
                    };
                    self.env_sq[b] =
                        flush_subnormal_f64(self.env_sq[b] + alpha * (sq - self.env_sq[b]));
                    let env = self.env_sq[b].sqrt();
                    let g = vca_band_gain(env, thresh_lin[b], slopes[b]);
                    sum += g * trim[b] * bands[b];
                }
                let out = (sum * CHAIN13_SOFT_CLIP_DRIVE).tanh() / CHAIN13_SOFT_CLIP_DRIVE;
                [flush_subnormal_f64(out), 0.0]
            });
        self.bit_depth.quantize_f64_to_f32(y)
    }
}

impl BlackBoxModel for Chain13TriBandVcaCompressor {
    fn reset(&mut self) {
        self.os.reset();
        self.xover_hi.reset();
        self.xover_lo.reset();
        self.hi_ap.reset();
        self.env_sq = [0.0; 3];
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN13_METADATA
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.sample_rate_hz = clamp_sample_rate_hz(sample_rate_hz);
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.bit_depth = bit_depth;
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let y = self.step_sample(inputs.first().copied().unwrap_or(0.0), controls, switches);
        if let Some(o) = outputs.first_mut() {
            *o = y;
        }
        if outputs.len() > 1 {
            outputs[1..].fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{run_mono_block, sine_wave};

    #[test]
    fn test_chain13_per_band_threshold_ratio_level_xover_and_zero_dc() {
        let meta = Chain13TriBandVcaCompressor::new().metadata();
        let base_c = meta.default_controls();
        let base_s = meta.default_switches();

        // Mid band (520 Hz): lower threshold compresses harder; higher ratio compresses harder.
        let mut c_thr_lo = base_c.clone();
        let mut c_thr_hi = base_c.clone();
        c_thr_lo[4] = -0.85;
        c_thr_hi[4] = 0.85;
        let x_mid = sine_wave(2_048, 520.0, 0.8);
        let y_thr_lo = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_mid,
            &c_thr_lo,
            &base_s,
        );
        let y_thr_hi = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_mid,
            &c_thr_hi,
            &base_s,
        );
        let rms = |y: &[f32]| -> f64 { y[512..].iter().map(|&v| f64::from(v * v)).sum() };
        assert!(
            rms(&y_thr_hi) > rms(&y_thr_lo) * 1.4,
            "lower mid threshold should compress harder"
        );

        let y_r2 = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_mid,
            &base_c,
            &[1, 0, 1],
        );
        let y_r12 = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_mid,
            &base_c,
            &[1, 3, 1],
        );
        assert!(
            rms(&y_r2) > rms(&y_r12) * 1.15,
            "12:1 mid ratio should compress harder than 2:1"
        );

        // Low crossover shifts 120 Hz between low and mid bands when levels differ.
        let mut c_xover_lo = base_c.clone();
        c_xover_lo[3] = 0.8;
        c_xover_lo[5] = -0.8;
        let mut c_xover_hi = c_xover_lo.clone();
        c_xover_lo[0] = -0.9;
        c_xover_hi[0] = 0.9;
        let x_lo = sine_wave(2_048, 120.0, 0.4);
        let y_x_lo = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_lo,
            &c_xover_lo,
            &base_s,
        );
        let y_x_hi = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_lo,
            &c_xover_hi,
            &base_s,
        );
        assert!(
            rms(&y_x_hi) > rms(&y_x_lo) * 2.0,
            "raising low_crossover routes 120 Hz from cut mid band into boosted low band"
        );

        // High crossover shifts 3 kHz between high and mid bands.
        let mut c_hi_lo = base_c.clone();
        c_hi_lo[5] = -0.8;
        c_hi_lo[7] = 0.8;
        let mut c_hi_hi = c_hi_lo.clone();
        c_hi_lo[1] = -0.8;
        c_hi_hi[1] = 0.8;
        let x_hi = sine_wave(2_048, 3_000.0, 0.4);
        let y_h_lo = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_hi,
            &c_hi_lo,
            &base_s,
        );
        let y_h_hi = run_mono_block(
            &mut Chain13TriBandVcaCompressor::new(),
            &x_hi,
            &c_hi_hi,
            &base_s,
        );
        assert!(
            rms(&y_h_lo) > rms(&y_h_hi) * 2.0,
            "lowering high_crossover routes 3 kHz into boosted high band"
        );

        let mut comp_z = Chain13TriBandVcaCompressor::new();
        let y_z = run_mono_block(&mut comp_z, &[0.0_f32; 128], &base_c, &[2, 3, 0]);
        assert!(y_z.iter().all(|&v| v == 0.0));
    }
}
