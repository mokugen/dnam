//! `15` `4x`-Oversampled Black-Box Physical Musical Signal-Chain Models (`Chain-01..15`),
//! supporting configurable device sample rates (`8..192 kHz`) and bit depths
//! (`Int12`, `Int16`, `Int24`, `Int32`, `Float32`), organized into category and per-chain submodules:
//! - [`drives`]: [`Chain01SymmetricOd`], [`Chain02GatedAsymmetricFuzz`], [`Chain03ParallelCleanBlend`], [`Chain14TriBandCompDistortion`]
//! - [`amps_and_cabs`]: [`Chain04TubeAmpFmv4x12`], [`Chain05PureLtiCabAndRoom`], [`Chain15TubeAmpHeadDi`]
//! - [`mod_and_spatial`]: [`Chain06EnvelopeResonantSvf`], [`Chain07StereoChorusAndRotary`], [`Chain08LongBbdEchoAnd30sShimmer`]
//! - [`dynamics_and_pitch`]: [`Chain09La2aOpticalCompressor`], [`Chain10Oc2OctaveDividerAndRingMod`], [`Chain13TriBandVcaCompressor`]
//! - [`rigs`]: [`Chain11BiAmpBassAndOutputIron`], [`Chain12FullEndToEndStereoRig`]

#![forbid(unsafe_code)]

pub mod amps_and_cabs;
pub mod drives;
pub mod dynamics_and_pitch;
pub mod mod_and_spatial;
pub mod rigs;

pub use amps_and_cabs::{Chain04TubeAmpFmv4x12, Chain05PureLtiCabAndRoom, Chain15TubeAmpHeadDi};
pub use drives::{
    Chain01SymmetricOd, Chain02GatedAsymmetricFuzz, Chain03ParallelCleanBlend,
    Chain14TriBandCompDistortion,
};
pub use dynamics_and_pitch::{
    Chain09La2aOpticalCompressor, Chain10Oc2OctaveDividerAndRingMod, Chain13TriBandVcaCompressor,
};
pub use mod_and_spatial::{
    CANONICAL_SUBDIVISION_RATIOS, Chain06EnvelopeResonantSvf, Chain07StereoChorusAndRotary,
    Chain08LongBbdEchoAnd30sShimmer,
};
pub use rigs::{Chain11BiAmpBassAndOutputIron, Chain12FullEndToEndStereoRig, Chain12RigMode};

use crate::clamp_sample_rate_hz;
use core::f64::consts::PI;

/// Returns the `4x` internal oversampled substep rate in Hz for `sample_rate_hz`.
#[inline]
#[must_use]
pub(crate) fn substep_rate_4x(sample_rate_hz: u32) -> f64 {
    4.0 * f64::from(clamp_sample_rate_hz(sample_rate_hz))
}

/// Maps a `192 kHz` (`4x` @ `48 kHz`) one-pole step coefficient `alpha_192k` to an arbitrary `fs_4x`
/// while preserving the exact continuous-time pole $s = \ln(1 - \alpha_{192\text{k}}) \cdot 192{,}000$.
#[inline]
#[must_use]
pub(crate) fn scale_alpha_from_192k(alpha_192k: f64, fs_4x: f64) -> f64 {
    if (fs_4x - 192_000.0).abs() <= 1.0e-9 {
        return alpha_192k;
    }
    (1.0 - (1.0 - alpha_192k.clamp(1.0e-8, 0.999)).powf(192_000.0 / fs_4x)).clamp(1.0e-8, 0.999)
}

/// Maps a `192 kHz` complex modal pole `(radius_192k, omega_rad_192k)` to an arbitrary `fs_4x`.
#[inline]
#[must_use]
pub(crate) fn scale_modal_pole_from_192k(
    radius_192k: f64,
    omega_rad_192k: f64,
    fs_4x: f64,
) -> (f64, f64) {
    if (fs_4x - 192_000.0).abs() <= 1.0e-9 {
        return (radius_192k, omega_rad_192k);
    }
    let ratio = 192_000.0 / fs_4x;
    let r = radius_192k.powf(ratio).clamp(0.5, 0.999_99);
    let omega = (omega_rad_192k * ratio).min(PI * 0.95);
    (r, omega)
}

/// Writes a single mono output channel and zeroes any extra caller output channels (`outputs[1..]`).
#[inline]
pub(crate) fn write_mono_output(outputs: &mut [f32], y: f32) {
    if let Some((first, rest)) = outputs.split_first_mut() {
        *first = y;
        rest.fill(0.0);
    }
}

/// Writes two stereo output channels and zeroes any extra caller output channels (`outputs[2..]`).
#[inline]
pub(crate) fn write_stereo_output(outputs: &mut [f32], yl: f32, yr: f32) {
    if let Some(out_l) = outputs.first_mut() {
        *out_l = yl;
    }
    if let Some(out_r) = outputs.get_mut(1) {
        *out_r = yr;
    }
    if outputs.len() > 2 {
        outputs[2..].fill(0.0);
    }
}

const fn const_sin_cos_f64(x: f64) -> (f64, f64) {
    let k_real = x * core::f64::consts::FRAC_2_PI;
    let k = if k_real >= 0.0 {
        (k_real + 0.5) as i64
    } else {
        (k_real - 0.5) as i64
    };
    let r = x - (k as f64) * core::f64::consts::FRAC_PI_2;
    let r2 = r * r;
    let mut sin_term = r;
    let mut sin_sum = r;
    let mut n = 1_u32;
    while n <= 10 {
        sin_term = -sin_term * r2 / ((2 * n) * (2 * n + 1)) as f64;
        sin_sum += sin_term;
        n += 1;
    }
    let mut cos_term = 1.0_f64;
    let mut cos_sum = 1.0_f64;
    let mut m = 1_u32;
    while m <= 10 {
        cos_term = -cos_term * r2 / ((2 * m - 1) * (2 * m)) as f64;
        cos_sum += cos_term;
        m += 1;
    }
    match k & 3 {
        0 => (sin_sum, cos_sum),
        1 => (cos_sum, -sin_sum),
        2 => (-sin_sum, -cos_sum),
        _ => (-cos_sum, sin_sum),
    }
}

const fn const_tan_f64(x: f64) -> f64 {
    let (s, c) = const_sin_cos_f64(x);
    s / c
}

const fn const_butter_two_r(order: usize, k: usize) -> f64 {
    let angle = ((2 * k + 1) as f64) * PI / ((2 * order) as f64);
    2.0 * const_sin_cos_f64(angle).0
}

/// 4th-order Butterworth damping factors `2 * sin((2k + 1) * pi / 8)` derived at compile time.
const BUTTER4_TWO_R: [f64; 2] = [const_butter_two_r(4, 0), const_butter_two_r(4, 1)];

/// 8th-order Butterworth damping factors `2 * sin((2k + 1) * pi / 16)` derived at compile time.
const BUTTER8_TWO_R: [f64; 4] = [
    const_butter_two_r(8, 0),
    const_butter_two_r(8, 1),
    const_butter_two_r(8, 2),
    const_butter_two_r(8, 3),
];

/// Prewarped relative cutoff `g = tan(pi * 0.475 / 4)` derived at compile time for `< 0.35 dB`
/// passband flatness up to `0.375 * fs` and exact zero DC imaging ripple.
const AA_SVF_G: f64 = const_tan_f64(PI * 0.475 / 4.0);

/// Converts a default physical control value in `[min_phys, max_phys]` into normalized `[-1.0, 1.0]`
/// at compile time (`ref[repo.code.constant-eval-and-derivations]`).
pub(crate) const fn const_phys_to_norm(phys: f32, min_phys: f32, max_phys: f32) -> f32 {
    let span = max_phys - min_phys;
    let t = (phys - min_phys) / span;
    let t_clamped = if t < 0.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        t
    };
    2.0 * t_clamped - 1.0
}

const _: () = assert!(BUTTER4_TWO_R[0] < BUTTER4_TWO_R[1] && BUTTER4_TWO_R[1] < 2.0);
const _: () = assert!(AA_SVF_G > 0.39 && AA_SVF_G < 0.40);
const _: () = assert!(const_phys_to_norm(250.0, 50.0, 800.0) > -0.5);
const _: () = assert!(const_phys_to_norm(250.0, 50.0, 800.0) < -0.4);

#[inline]
pub(crate) fn flush_subnormal_f64(x: f64) -> f64 {
    if x.abs() < f64::MIN_POSITIVE { 0.0 } else { x }
}

/// Phase-coherent 4th-order Linkwitz–Riley (`LR4`) TPT-SVF crossover splitting a signal into
/// `(lp4, hp4)` whose direct sum `(lp4 + hp4)` maintains unity all-pass magnitude (`0.00 dB` ripple).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TptLr4Crossover2 {
    lp_st: [[f64; 2]; 2],
    hp_st: [[f64; 2]; 2],
}

impl TptLr4Crossover2 {
    pub(crate) fn reset(&mut self) {
        self.lp_st = [[0.0; 2]; 2];
        self.hp_st = [[0.0; 2]; 2];
    }

    #[inline]
    pub(crate) fn step(&mut self, x: f64, fc_hz: f64, fs_hz: f64) -> (f64, f64) {
        let g = (PI * fc_hz.clamp(20.0, 0.42 * fs_hz) / fs_hz).tan();
        let k = core::f64::consts::SQRT_2;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let mut lp = x;
        for st in &mut self.lp_st {
            let v3 = lp - st[1];
            let v1 = a1 * st[0] + a2 * v3;
            let v2 = st[1] + a2 * st[0] + a3 * v3;
            st[0] = flush_subnormal_f64(2.0 * v1 - st[0]);
            st[1] = flush_subnormal_f64(2.0 * v2 - st[1]);
            lp = flush_subnormal_f64(v2);
        }

        let mut hp = x;
        for st in &mut self.hp_st {
            let v3 = hp - st[1];
            let v1 = a1 * st[0] + a2 * v3;
            let v2 = st[1] + a2 * st[0] + a3 * v3;
            let hp_out = hp - k * v1 - v2;
            st[0] = flush_subnormal_f64(2.0 * v1 - st[0]);
            st[1] = flush_subnormal_f64(2.0 * v2 - st[1]);
            hp = flush_subnormal_f64(hp_out);
        }

        (lp, hp)
    }
}

#[inline]
fn step_butter4_svf_lp(x: f64, state: &mut [[f64; 2]; 2]) -> f64 {
    let mut sig = x;
    for (stage, &two_r) in state.iter_mut().zip(&BUTTER4_TWO_R) {
        let a1 = 1.0 / (1.0 + AA_SVF_G * (AA_SVF_G + two_r));
        let a2 = AA_SVF_G * a1;
        let a3 = AA_SVF_G * a2;
        let v3 = sig - stage[1];
        let v1 = a1 * stage[0] + a2 * v3;
        let v2 = stage[1] + a2 * stage[0] + a3 * v3;
        stage[0] = flush_subnormal_f64(2.0 * v1 - stage[0]);
        stage[1] = flush_subnormal_f64(2.0 * v2 - stage[1]);
        sig = flush_subnormal_f64(v2);
    }
    sig
}

#[inline]
fn step_butter8_svf_lp(x: f64, state: &mut [[f64; 2]; 4]) -> f64 {
    let mut sig = x;
    for (stage, &two_r) in state.iter_mut().zip(&BUTTER8_TWO_R) {
        let a1 = 1.0 / (1.0 + AA_SVF_G * (AA_SVF_G + two_r));
        let a2 = AA_SVF_G * a1;
        let a3 = AA_SVF_G * a2;
        let v3 = sig - stage[1];
        let v1 = a1 * stage[0] + a2 * v3;
        let v2 = stage[1] + a2 * stage[0] + a3 * v3;
        stage[0] = flush_subnormal_f64(2.0 * v1 - stage[0]);
        stage[1] = flush_subnormal_f64(2.0 * v2 - stage[1]);
        sig = flush_subnormal_f64(v2);
    }
    sig
}

/// Internal `4x`-oversampled Butterworth TPT SVF anti-imaging / anti-aliasing
/// upsampler and decimator state used by all physical signal chains (`Chain-01..12`).
#[derive(Clone, Debug, Default)]
pub struct Oversampler4xF64 {
    prev_in_l: f64,
    prev_in_r: f64,
    up_l: [[f64; 2]; 2],
    up_r: [[f64; 2]; 2],
    down_l: [[f64; 2]; 4],
    down_r: [[f64; 2]; 4],
}

impl Oversampler4xF64 {
    /// Resets all `4x` anti-imaging and anti-aliasing TPT SVF integrator states to zero DC equilibrium.
    pub fn reset(&mut self) {
        self.prev_in_l = 0.0;
        self.prev_in_r = 0.0;
        self.up_l = [[0.0; 2]; 2];
        self.up_r = [[0.0; 2]; 2];
        self.down_l = [[0.0; 2]; 4];
        self.down_r = [[0.0; 2]; 4];
    }

    /// Upsamples one stereo frame `(in_l, in_r)` by `4x` via sinc-droop-compensated ZOH + 4th-order
    /// Butterworth TPT SVF anti-imaging (placing exact spectral nulls at `fs, 2fs, 3fs` for zero DC ripple),
    /// executes `f_substep(sub_idx, u_l, u_r)`, and decimates `4 -> 1` via an 8th-order Butterworth TPT SVF.
    #[inline]
    pub fn process_frame_4x<F>(&mut self, in_l: f64, in_r: f64, mut f_substep: F) -> [f64; 2]
    where
        F: FnMut(usize, f64, f64) -> [f64; 2],
    {
        let x_l = 1.185 * in_l - 0.185 * self.prev_in_l;
        let x_r = 1.185 * in_r - 0.185 * self.prev_in_r;
        self.prev_in_l = in_l;
        self.prev_in_r = in_r;

        let mut out_l = 0.0_f64;
        let mut out_r = 0.0_f64;

        for sub in 0..4 {
            let u_l = step_butter4_svf_lp(x_l, &mut self.up_l);
            let u_r = step_butter4_svf_lp(x_r, &mut self.up_r);

            let [y_l, y_r] = f_substep(sub, u_l, u_r);

            let d_l = step_butter8_svf_lp(y_l, &mut self.down_l);
            let d_r = step_butter8_svf_lp(y_r, &mut self.down_r);
            if sub == 3 {
                out_l = d_l;
                out_r = d_r;
            }
        }
        [out_l, out_r]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    #[test]
    fn test_oversampler_4x_f64_dc_convergence_hf_flatness_and_pole_scaling() {
        assert_eq!(substep_rate_4x(48_000), 192_000.0);
        assert_eq!(substep_rate_4x(96_000), 384_000.0);
        assert_eq!(scale_alpha_from_192k(0.05, 192_000.0), 0.05);
        let a_384 = scale_alpha_from_192k(0.05, 384_000.0);
        assert!(((1.0 - a_384).powi(2) - 0.95).abs() < 1.0e-12);
        let (r_384, w_384) = scale_modal_pole_from_192k(0.99, 0.02, 384_000.0);
        assert!((r_384 * r_384 - 0.99).abs() < 1.0e-12);
        assert!((w_384 - 0.01).abs() < 1.0e-12);

        let mut os = Oversampler4xF64::default();
        let mut last = [0.0_f64; 2];
        for _ in 0..128 {
            last = os.process_frame_4x(1.0, -0.5, |_, ul, ur| [ul, ur]);
        }
        assert!((last[0] - 1.0).abs() < 1.0e-9);
        assert!((last[1] - (-0.5)).abs() < 1.0e-9);

        // Verify < 0.35 dB passband flatness at 0.375 * fs across 44.1 kHz, 48 kHz, and 96 kHz
        for fs in [44_100.0_f64, 48_000.0, 96_000.0] {
            os.reset();
            let mut sum_sq_l = 0.0_f64;
            let mut sum_sq_r = 0.0_f64;
            let n_total = 1024_usize;
            let warmup = 128_usize;
            let f_probe = 0.375 * fs;
            for n in 0..n_total {
                let sig = (TAU * f_probe * (n as f64) / fs).sin();
                let [yl, yr] = os.process_frame_4x(sig, 0.0, |_, ul, ur| [ul, ur]);
                if n >= warmup {
                    sum_sq_l += yl * yl;
                    sum_sq_r += yr * yr;
                }
            }
            let rms_l = (sum_sq_l / ((n_total - warmup) as f64)).sqrt();
            let gain_db = 20.0 * (rms_l / 0.5_f64.sqrt()).log10();
            assert!(
                gain_db.abs() < 0.35,
                "Passband droop at {f_probe} Hz / {fs} Hz exceeded 0.35 dB: {gain_db:.4} dB"
            );
            assert_eq!(sum_sq_r, 0.0);
        }

        os.reset();
        let after_reset = os.process_frame_4x(0.0, 0.0, |_, ul, ur| [ul, ur]);
        assert_eq!(after_reset, [0.0, 0.0]);

        let mut xover = TptLr4Crossover2::default();
        assert_eq!(xover.step(0.0, 500.0, 48_000.0), (0.0, 0.0));
        xover.reset();
    }
}
