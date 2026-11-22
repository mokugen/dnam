//! `Chain-17`: Endless 3-Phase Barberpole Phaser & Comb Sweep (`2 -> 2`, $B_2$ Quadratic B-Spline).

#![forbid(unsafe_code)]

use crate::chains::{
    Oversampler4xF64, catmull_rom_ring_read, const_phys_to_norm, flush_subnormal_f64,
    substep_rate_4x,
};
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};

const RATE_DEFAULT_NORM: f32 = const_phys_to_norm(1.20, 0.05, 4.0);
const SPAN_DEFAULT_NORM: f32 = const_phys_to_norm(4.0, 0.5, 10.0);
const FB_DEFAULT_NORM: f32 = const_phys_to_norm(45.0, -75.0, 75.0);
const MIX_DEFAULT_NORM: f32 = const_phys_to_norm(65.0, 0.0, 100.0);

const _: () = assert!(FB_DEFAULT_NORM > 0.59 && FB_DEFAULT_NORM < 0.61);

const CHAIN17_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-17-Barberpole-Sweep",
    name: "Endless 3-Phase Barberpole Phaser & Comb Sweep",
    category: "barberpole",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 80.0,
    supports_tempo_sync: true,
    controls: &[
        FixtureControlSpec {
            id: "rate",
            name: "Sweep Rate",
            unit: "Hz",
            default_norm: RATE_DEFAULT_NORM,
            min_physical: 0.05,
            max_physical: 4.0,
        },
        FixtureControlSpec {
            id: "span",
            name: "Comb Span",
            unit: "ms",
            default_norm: SPAN_DEFAULT_NORM,
            min_physical: 0.5,
            max_physical: 10.0,
        },
        FixtureControlSpec {
            id: "feedback",
            name: "Regeneration Feedback",
            unit: "%",
            default_norm: FB_DEFAULT_NORM,
            min_physical: -75.0,
            max_physical: 75.0,
        },
        FixtureControlSpec {
            id: "mix",
            name: "Mix",
            unit: "%",
            default_norm: MIX_DEFAULT_NORM,
            min_physical: 0.0,
            max_physical: 100.0,
        },
    ],
    switches: &[FixtureSwitchSpec {
        id: "direction",
        name: "Sweep Direction",
        options: &["Rising", "Falling"],
        default_index: 0,
    }],
};

#[inline]
fn ring_len_for_rate(sample_rate_hz: u32) -> usize {
    let fs_4x = substep_rate_4x(sample_rate_hz);
    let max_needed = ((0.020 * fs_4x).ceil() as usize) + 32;
    max_needed.next_power_of_two().max(4_096)
}

/// Cardinal quadratic B-spline window $B_2(3 \cdot \phi)$ with $\sum_{m=0}^2 B_2(\phi + m/3) \equiv 1.0$.
#[inline]
pub(crate) fn quadratic_bspline_3phase_weight(phi: f64) -> f64 {
    let u = 3.0 * phi.rem_euclid(1.0);
    if u < 1.0 {
        0.5 * u * u
    } else if u < 2.0 {
        -u * u + 3.0 * u - 1.5
    } else {
        let d = 3.0 - u;
        0.5 * d * d
    }
}

/// `Chain-17-Barberpole-Sweep`: `2 -> 2` `4x`-oversampled endless 3-phase cardinal quadratic B-spline
/// barberpole comb/phaser sweep with $C^1$ zero-slope phase wraps (`ref[dsp.op-t.window.m3-barberpole]`).
#[derive(Clone, Debug)]
pub struct Chain17BarberpoleSweep {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    ring_l: Vec<f64>,
    ring_r: Vec<f64>,
    wr_idx: usize,
    phase_l: f64,
    phase_r: f64,
    fb_state_l: f64,
    fb_state_r: f64,
}

impl Default for Chain17BarberpoleSweep {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain17BarberpoleSweep {
    /// Creates a `Chain-17` barberpole sweep fixture at `48 kHz` / `Float32`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    /// Creates a `Chain-17` barberpole sweep fixture configured for `(sample_rate_hz, bit_depth)`.
    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let rlen = ring_len_for_rate(sr);
        Self {
            sample_rate_hz: sr,
            bit_depth,
            os: Oversampler4xF64::default(),
            ring_l: vec![0.0; rlen],
            ring_r: vec![0.0; rlen],
            wr_idx: 0,
            phase_l: 0.0,
            phase_r: 0.25,
            fb_state_l: 0.0,
            fb_state_r: 0.0,
        }
    }

    /// Processes a single stereo sample pair at the configured device sample rate and bit depth.
    #[inline]
    pub fn step_stereo(
        &mut self,
        in_l: f32,
        in_r: f32,
        controls: &[f32],
        switches: &[u8],
    ) -> [f32; 2] {
        let xl = self.bit_depth.quantize_sample(in_l);
        let xr = self.bit_depth.quantize_sample(in_r);

        let rate_u = f64::from(sanitize_norm(
            controls.first().copied().unwrap_or(RATE_DEFAULT_NORM),
            RATE_DEFAULT_NORM,
        ));
        let span_u = f64::from(sanitize_norm(
            controls.get(1).copied().unwrap_or(SPAN_DEFAULT_NORM),
            SPAN_DEFAULT_NORM,
        ));
        let fb_u = f64::from(sanitize_norm(
            controls.get(2).copied().unwrap_or(FB_DEFAULT_NORM),
            FB_DEFAULT_NORM,
        ));
        let mix_u = f64::from(sanitize_norm(
            controls.get(3).copied().unwrap_or(MIX_DEFAULT_NORM),
            MIX_DEFAULT_NORM,
        ));
        let falling = switches.first().copied().unwrap_or(0) != 0;

        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let rate_hz = 2.025 + 1.975 * rate_u;
        let span_ms = 5.25 + 4.75 * span_u;
        let base_samp = (0.001_5 * fs_4x).max(4.0);
        let span_samp = span_ms * 1.0e-3 * fs_4x;
        let max_d = (self.ring_l.len().saturating_sub(6)) as f64;
        let fb_gain = 0.75 * fb_u;
        let wet_mix = 0.5 * (mix_u + 1.0);
        let ring_mask = self.ring_l.len() - 1;
        let phase_step = rate_hz / fs_4x;

        let read_3phase = |ring: &[f64], wr: usize, base_phi: f64| -> f64 {
            let mut acc = 0.0_f64;
            for m in 0..3 {
                let pm = (base_phi + (m as f64) / 3.0).rem_euclid(1.0);
                let w = quadratic_bspline_3phase_weight(pm);
                let ramp = if falling { pm } else { 1.0 - pm };
                let dly = (base_samp + span_samp * ramp).clamp(2.0, max_d);
                acc += w * catmull_rom_ring_read(ring, wr, dly);
            }
            acc
        };

        let [yl, yr] = self
            .os
            .process_frame_4x(f64::from(xl), f64::from(xr), |_, ul, ur| {
                self.ring_l[self.wr_idx] = flush_subnormal_f64(ul + self.fb_state_l);
                self.ring_r[self.wr_idx] = flush_subnormal_f64(ur + self.fb_state_r);

                let comb_l = read_3phase(&self.ring_l, self.wr_idx, self.phase_l);
                let comb_r = read_3phase(&self.ring_r, self.wr_idx, self.phase_r);

                self.wr_idx = (self.wr_idx + 1) & ring_mask;
                self.phase_l = (self.phase_l + phase_step).rem_euclid(1.0);
                self.phase_r = (self.phase_r + phase_step).rem_euclid(1.0);

                self.fb_state_l = flush_subnormal_f64((fb_gain * comb_l).tanh());
                self.fb_state_r = flush_subnormal_f64((fb_gain * comb_r).tanh());

                [
                    flush_subnormal_f64((1.0 - wet_mix) * ul + wet_mix * comb_l),
                    flush_subnormal_f64((1.0 - wet_mix) * ur + wet_mix * comb_r),
                ]
            });

        [
            self.bit_depth.quantize_f64_to_f32(yl),
            self.bit_depth.quantize_f64_to_f32(yr),
        ]
    }
}

impl BlackBoxModel for Chain17BarberpoleSweep {
    fn reset(&mut self) {
        self.os.reset();
        self.ring_l.fill(0.0);
        self.ring_r.fill(0.0);
        self.wr_idx = 0;
        self.phase_l = 0.0;
        self.phase_r = 0.25;
        self.fb_state_l = 0.0;
        self.fb_state_r = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN17_METADATA
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.sample_rate_hz = clamp_sample_rate_hz(sample_rate_hz);
        let rlen = ring_len_for_rate(self.sample_rate_hz);
        if self.ring_l.len() != rlen {
            self.ring_l = vec![0.0; rlen];
            self.ring_r = vec![0.0; rlen];
        }
        self.reset();
    }

    fn set_bit_depth(&mut self, bit_depth: FixtureBitDepth) {
        self.bit_depth = bit_depth;
        self.reset();
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        2
    }

    fn process_frame(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        controls: &[f32],
        switches: &[u8],
    ) {
        let xl = inputs.first().copied().unwrap_or(0.0);
        let xr = inputs.get(1).copied().unwrap_or(xl);
        let [yl, yr] = self.step_stereo(xl, xr, controls, switches);
        crate::chains::write_stereo_output(outputs, yl, yr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{exercise_blackbox_metadata_and_format, run_stereo_block, sine_wave};

    #[test]
    fn test_chain17_barberpole_bspline_unity_direction_and_format() {
        let mut c17 = Chain17BarberpoleSweep::default();
        assert_eq!(c17.metadata().id, "Chain-17-Barberpole-Sweep");
        exercise_blackbox_metadata_and_format(&mut c17);

        for step in 0..90 {
            let p = (step as f64) / 90.0;
            let sum = quadratic_bspline_3phase_weight(p)
                + quadratic_bspline_3phase_weight(p + 1.0 / 3.0)
                + quadratic_bspline_3phase_weight(p + 2.0 / 3.0);
            assert!((sum - 1.0).abs() < 1.0e-12);
        }

        let dry_l = sine_wave(4_800, 330.0, 0.45);
        let dry_r = sine_wave(4_800, 330.0, 0.45);
        let c = c17.metadata().default_controls();
        let (y_up_l, y_up_r) = run_stereo_block(&mut c17, &dry_l, &dry_r, &c, &[0]);
        c17.reset();
        let (y_dn_l, _) = run_stereo_block(&mut c17, &dry_l, &dry_r, &c, &[1]);
        assert_ne!(y_up_l, y_up_r);
        assert_ne!(y_up_l, y_dn_l);
    }
}
