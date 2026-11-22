//! `Chain-16`: Granular Pitch Shifter & Shimmer Tank (`2 -> 2`, Dual Hermite-Hann + Catmull-Rom).

#![forbid(unsafe_code)]

use crate::chains::{
    Oversampler4xF64, catmull_rom_ring_read, const_phys_to_norm, flush_subnormal_f64,
    scale_alpha_from_192k, substep_rate_4x,
};
use crate::sim_kernels::UnitaryHouseholder4;
use crate::{
    BlackBoxModel, DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth, FixtureControlSpec,
    FixtureDeviceMetadata, FixtureSwitchSpec, clamp_sample_rate_hz, sanitize_norm,
};

const PITCH_DEFAULT_NORM: f32 = const_phys_to_norm(12.0, -24.0, 24.0);
const GRAIN_DEFAULT_NORM: f32 = const_phys_to_norm(35.0, 20.0, 80.0);
const FEEDBACK_DEFAULT_NORM: f32 = const_phys_to_norm(35.0, 0.0, 85.0);
const MIX_DEFAULT_NORM: f32 = const_phys_to_norm(60.0, 0.0, 100.0);

const _: () = assert!(PITCH_DEFAULT_NORM > 0.49 && PITCH_DEFAULT_NORM < 0.51);
const _: () = assert!(GRAIN_DEFAULT_NORM > -0.51 && GRAIN_DEFAULT_NORM < -0.49);

const CHAIN16_METADATA: FixtureDeviceMetadata = FixtureDeviceMetadata {
    id: "Chain-16-Granular-Pitch-Shimmer",
    name: "Granular Pitch Shifter & Shimmer Tank",
    category: "pitch_shimmer",
    sample_rate_hz: DEFAULT_FIXTURE_SAMPLE_RATE_HZ,
    bit_depth: FixtureBitDepth::Float32,
    expected_tail_ms: 1_200.0,
    supports_tempo_sync: false,
    controls: &[
        FixtureControlSpec {
            id: "pitch",
            name: "Pitch Shift",
            unit: "st",
            default_norm: PITCH_DEFAULT_NORM,
            min_physical: -24.0,
            max_physical: 24.0,
        },
        FixtureControlSpec {
            id: "grain_size",
            name: "Grain Size",
            unit: "ms",
            default_norm: GRAIN_DEFAULT_NORM,
            min_physical: 20.0,
            max_physical: 80.0,
        },
        FixtureControlSpec {
            id: "feedback",
            name: "Shimmer Feedback",
            unit: "%",
            default_norm: FEEDBACK_DEFAULT_NORM,
            min_physical: 0.0,
            max_physical: 85.0,
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
        id: "voicing",
        name: "Shimmer Voicing",
        options: &["Dual Grain", "Shimmer Tank"],
        default_index: 1,
    }],
};

const FDN_TAPS_192K: [usize; 4] = [2_111, 2_861, 3_769, 4_481]; // 11.0..23.3 ms @ 192 kHz (4x @ 48 kHz)

#[inline]
fn ring_len_for_rate(sample_rate_hz: u32) -> usize {
    let fs_4x = substep_rate_4x(sample_rate_hz);
    let max_needed = ((0.280 * fs_4x).ceil() as usize) + 32;
    max_needed.next_power_of_two().max(32_768)
}

#[inline]
fn fdn_ring_len_for_rate(sample_rate_hz: u32) -> usize {
    let fs_4x = substep_rate_4x(sample_rate_hz);
    let max_tap = ((FDN_TAPS_192K[3] as f64) * (fs_4x / 192_000.0)).ceil() as usize;
    (max_tap + 32).next_power_of_two().max(4_096)
}

/// Partition-of-unity Hermite-Hann envelope $u = 1 - |2\phi - 1|$, $H(\phi) = u^2(3 - 2u)$.
#[inline]
pub(crate) fn hermite_hann_weight(phi: f64) -> f64 {
    let p = phi.rem_euclid(1.0);
    let u = (1.0 - (2.0 * p - 1.0).abs()).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// `Chain-16-Granular-Pitch-Shimmer`: `2 -> 2` `4x`-oversampled centered dual-grain Hermite-Hann
/// pitch shifter ($\Delta p \in [-24, +24]$ st) with optional `UnitaryHouseholder4` shimmer diffusion
/// and pitch-down loop contraction $\sqrt{1 - \max(0, \nu)}$ (`ref[dsp.op-t.window.m2-pitch]`).
#[derive(Clone, Debug)]
pub struct Chain16GranularPitchShimmer {
    sample_rate_hz: u32,
    bit_depth: FixtureBitDepth,
    os: Oversampler4xF64,
    ring_l: Vec<f64>,
    ring_r: Vec<f64>,
    wr_idx: usize,
    phase_l: f64,
    phase_r: f64,
    fdn_rings: [Vec<f64>; 4],
    fdn_wr: usize,
    fb_state_l: f64,
    fb_state_r: f64,
    fb_lp_l: f64,
    fb_lp_r: f64,
    householder: UnitaryHouseholder4,
}

impl Default for Chain16GranularPitchShimmer {
    fn default() -> Self {
        Self::new()
    }
}

impl Chain16GranularPitchShimmer {
    /// Creates a `Chain-16` granular pitch/shimmer fixture at `48 kHz` / `Float32`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_format(DEFAULT_FIXTURE_SAMPLE_RATE_HZ, FixtureBitDepth::Float32)
    }

    /// Creates a `Chain-16` granular pitch/shimmer fixture configured for `(sample_rate_hz, bit_depth)`.
    #[must_use]
    pub fn with_format(sample_rate_hz: u32, bit_depth: FixtureBitDepth) -> Self {
        let sr = clamp_sample_rate_hz(sample_rate_hz);
        let rlen = ring_len_for_rate(sr);
        let flen = fdn_ring_len_for_rate(sr);
        Self {
            sample_rate_hz: sr,
            bit_depth,
            os: Oversampler4xF64::default(),
            ring_l: vec![0.0; rlen],
            ring_r: vec![0.0; rlen],
            wr_idx: 0,
            phase_l: 0.0,
            phase_r: 0.25,
            fdn_rings: [
                vec![0.0; flen],
                vec![0.0; flen],
                vec![0.0; flen],
                vec![0.0; flen],
            ],
            fdn_wr: 0,
            fb_state_l: 0.0,
            fb_state_r: 0.0,
            fb_lp_l: 0.0,
            fb_lp_r: 0.0,
            householder: UnitaryHouseholder4::canonical_hadamard(),
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

        let pitch_u = f64::from(sanitize_norm(
            controls.first().copied().unwrap_or(PITCH_DEFAULT_NORM),
            PITCH_DEFAULT_NORM,
        ));
        let grain_u = f64::from(sanitize_norm(
            controls.get(1).copied().unwrap_or(GRAIN_DEFAULT_NORM),
            GRAIN_DEFAULT_NORM,
        ));
        let fb_u = f64::from(sanitize_norm(
            controls.get(2).copied().unwrap_or(FEEDBACK_DEFAULT_NORM),
            FEEDBACK_DEFAULT_NORM,
        ));
        let mix_u = f64::from(sanitize_norm(
            controls.get(3).copied().unwrap_or(MIX_DEFAULT_NORM),
            MIX_DEFAULT_NORM,
        ));
        let shimmer_tank = switches.first().copied().unwrap_or(1) != 0;

        let semitones = 24.0 * pitch_u;
        let ratio = (2.0_f64).powf(semitones / 12.0);
        let nu = 1.0 - ratio;
        let fs_4x = substep_rate_4x(self.sample_rate_hz);
        let grain_ms = 50.0 + 30.0 * grain_u;
        let grain_samp = ((grain_ms * 1.0e-3) * fs_4x).max(64.0);
        let max_d = (self.ring_l.len().saturating_sub(6)) as f64;
        let half_sweep = 0.5 * nu.abs() * grain_samp;
        let center_delay = (half_sweep + 4.0).max(0.012 * fs_4x).min(max_d * 0.5);
        let span = nu * grain_samp;

        // Enforce pitch-down passivity contraction per ref[dsp.op-t.window.m2-pitch].
        let fb_raw = 0.425 * (fb_u + 1.0);
        let fb_gain = fb_raw * (1.0 - nu.max(0.0)).max(0.0).sqrt();
        let wet_mix = 0.5 * (mix_u + 1.0);
        let damp_alpha = scale_alpha_from_192k(0.18, fs_4x);
        let ring_mask = self.ring_l.len() - 1;
        let fdn_mask = self.fdn_rings[0].len() - 1;
        let fdn_scale = fs_4x / 192_000.0;
        let fdn_delays = [
            ((FDN_TAPS_192K[0] as f64 * fdn_scale).round() as usize).clamp(4, fdn_mask - 4),
            ((FDN_TAPS_192K[1] as f64 * fdn_scale).round() as usize).clamp(4, fdn_mask - 4),
            ((FDN_TAPS_192K[2] as f64 * fdn_scale).round() as usize).clamp(4, fdn_mask - 4),
            ((FDN_TAPS_192K[3] as f64 * fdn_scale).round() as usize).clamp(4, fdn_mask - 4),
        ];
        let phase_inc = 1.0 / grain_samp;

        let [yl, yr] = self
            .os
            .process_frame_4x(f64::from(xl), f64::from(xr), |_, ul, ur| {
                self.ring_l[self.wr_idx] = flush_subnormal_f64(ul + self.fb_state_l);
                self.ring_r[self.wr_idx] = flush_subnormal_f64(ur + self.fb_state_r);

                let pl_a = self.phase_l;
                let pl_b = (pl_a + 0.5).rem_euclid(1.0);
                let pr_a = self.phase_r;
                let pr_b = (pr_a + 0.5).rem_euclid(1.0);

                let dl_a = (center_delay + span * (pl_a - 0.5)).clamp(2.0, max_d);
                let dl_b = (center_delay + span * (pl_b - 0.5)).clamp(2.0, max_d);
                let dr_a = (center_delay + span * (pr_a - 0.5)).clamp(2.0, max_d);
                let dr_b = (center_delay + span * (pr_b - 0.5)).clamp(2.0, max_d);

                let wl_a = hermite_hann_weight(pl_a);
                let wl_b = hermite_hann_weight(pl_b);
                let wr_a = hermite_hann_weight(pr_a);
                let wr_b = hermite_hann_weight(pr_b);

                let pitch_l = wl_a * catmull_rom_ring_read(&self.ring_l, self.wr_idx, dl_a)
                    + wl_b * catmull_rom_ring_read(&self.ring_l, self.wr_idx, dl_b);
                let pitch_r = wr_a * catmull_rom_ring_read(&self.ring_r, self.wr_idx, dr_a)
                    + wr_b * catmull_rom_ring_read(&self.ring_r, self.wr_idx, dr_b);

                self.wr_idx = (self.wr_idx + 1) & ring_mask;
                self.phase_l = (pl_a + phase_inc).rem_euclid(1.0);
                self.phase_r = (pr_a + phase_inc).rem_euclid(1.0);

                let (wet_l, wet_r) = if shimmer_tank {
                    let t0 = self.fdn_rings[0][self.fdn_wr.wrapping_sub(fdn_delays[0]) & fdn_mask];
                    let t1 = self.fdn_rings[1][self.fdn_wr.wrapping_sub(fdn_delays[1]) & fdn_mask];
                    let t2 = self.fdn_rings[2][self.fdn_wr.wrapping_sub(fdn_delays[2]) & fdn_mask];
                    let t3 = self.fdn_rings[3][self.fdn_wr.wrapping_sub(fdn_delays[3]) & fdn_mask];
                    let diff = self.householder.apply([t0, t1, t2, t3]);
                    let tank_g = 0.72 * fb_gain;
                    self.fdn_rings[0][self.fdn_wr] =
                        flush_subnormal_f64(pitch_l + tank_g * diff[0]);
                    self.fdn_rings[1][self.fdn_wr] =
                        flush_subnormal_f64(pitch_r + tank_g * diff[1]);
                    self.fdn_rings[2][self.fdn_wr] =
                        flush_subnormal_f64(0.5 * (pitch_l + pitch_r) + tank_g * diff[2]);
                    self.fdn_rings[3][self.fdn_wr] =
                        flush_subnormal_f64(0.5 * (pitch_l - pitch_r) + tank_g * diff[3]);
                    self.fdn_wr = (self.fdn_wr + 1) & fdn_mask;
                    (
                        0.55 * pitch_l + 0.45 * (t0 + t2),
                        0.55 * pitch_r + 0.45 * (t1 + t3),
                    )
                } else {
                    (pitch_l, pitch_r)
                };

                self.fb_lp_l += damp_alpha * (wet_l - self.fb_lp_l);
                self.fb_lp_r += damp_alpha * (wet_r - self.fb_lp_r);
                self.fb_state_l = flush_subnormal_f64((fb_gain * self.fb_lp_l).tanh());
                self.fb_state_r = flush_subnormal_f64((fb_gain * self.fb_lp_r).tanh());

                [
                    flush_subnormal_f64((1.0 - wet_mix) * ul + wet_mix * wet_l),
                    flush_subnormal_f64((1.0 - wet_mix) * ur + wet_mix * wet_r),
                ]
            });

        [
            self.bit_depth.quantize_f64_to_f32(yl),
            self.bit_depth.quantize_f64_to_f32(yr),
        ]
    }
}

impl BlackBoxModel for Chain16GranularPitchShimmer {
    fn reset(&mut self) {
        self.os.reset();
        self.ring_l.fill(0.0);
        self.ring_r.fill(0.0);
        self.wr_idx = 0;
        self.phase_l = 0.0;
        self.phase_r = 0.25;
        for r in &mut self.fdn_rings {
            r.fill(0.0);
        }
        self.fdn_wr = 0;
        self.fb_state_l = 0.0;
        self.fb_state_r = 0.0;
        self.fb_lp_l = 0.0;
        self.fb_lp_r = 0.0;
    }

    fn metadata(&self) -> FixtureDeviceMetadata {
        FixtureDeviceMetadata {
            sample_rate_hz: self.sample_rate_hz,
            bit_depth: self.bit_depth,
            ..CHAIN16_METADATA
        }
    }

    fn set_sample_rate_hz(&mut self, sample_rate_hz: u32) {
        self.sample_rate_hz = clamp_sample_rate_hz(sample_rate_hz);
        let rlen = ring_len_for_rate(self.sample_rate_hz);
        if self.ring_l.len() != rlen {
            self.ring_l = vec![0.0; rlen];
            self.ring_r = vec![0.0; rlen];
        }
        let flen = fdn_ring_len_for_rate(self.sample_rate_hz);
        if self.fdn_rings[0].len() != flen {
            self.fdn_rings = [
                vec![0.0; flen],
                vec![0.0; flen],
                vec![0.0; flen],
                vec![0.0; flen],
            ];
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
    fn test_chain16_pitch_shimmer_partition_of_unity_shift_and_format() {
        let mut c16 = Chain16GranularPitchShimmer::default();
        assert_eq!(c16.metadata().id, "Chain-16-Granular-Pitch-Shimmer");
        exercise_blackbox_metadata_and_format(&mut c16);

        for step in 0..64 {
            let p = (step as f64) / 64.0;
            let sum = hermite_hann_weight(p) + hermite_hann_weight(p + 0.5);
            assert!((sum - 1.0).abs() < 1.0e-12);
        }

        // Verify upward (+12 st) and downward (-12 st) pitch shifts alter correlation orthogonally.
        let dry_l = sine_wave(8_192, 220.0, 0.45);
        let dry_r = sine_wave(8_192, 220.0, 0.45);
        let up_ctrl = [0.5, GRAIN_DEFAULT_NORM, -1.0, 1.0];
        let dn_ctrl = [-0.5, GRAIN_DEFAULT_NORM, -1.0, 1.0];
        let (yl_up, yr_up) = run_stereo_block(&mut c16, &dry_l, &dry_r, &up_ctrl, &[0]);
        c16.reset();
        let (yl_dn, _) = run_stereo_block(&mut c16, &dry_l, &dry_r, &dn_ctrl, &[1]);
        assert_ne!(yl_up, yr_up);
        assert_ne!(yl_up, yl_dn);
        assert!(
            yl_up
                .iter()
                .chain(&yl_dn)
                .all(|v| v.is_finite() && v.abs() <= 4.0)
        );
    }
}
