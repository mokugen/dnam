//! Unified 2-mode (`RenderMode::{Live, Master}`) Core-Only Multirate oversampler (`CoreOversampler`)
//! separating the `1x` host-rate perimeter from the `<= 384 kHz` `Core` (`ADR-0026`,
//! `ref[arch.bounce.oversampling]`, `ref[dsp.router.signal-flow]`).
//!
//! - `Live`: Hot `CoreOversamplerLiveHot` (`HbFir15UpsamplerStereo` + `HbFir15Downsampler4Ch`,
//!   `232 B`, `PDC = 7` @ `44.1..192 kHz`, `0` @ `352.8/384 kHz`) inside `SonaLiteHotPrefix`.
//! - `Master`: Cold `CoreOversamplerMasterCold` (`HB63` + `z^-1@2x` + `HB23` + `Box2` +
//!   `MasterCtrlDelayRingCold`, `PDC = 37` @ `44.1..96 kHz`, `31` @ `176.4/192 kHz`,
//!   `0` @ `352.8/384 kHz`) inside `SonaColdTail`.
//!
//! Strictly `#![no_std]`, `#![forbid(unsafe_code)]`, zero `FDIV`/`FSQRT`/`IDIV`/`libm`.

#![forbid(unsafe_code)]

pub mod master;

pub use master::{
    CoreOversamplerMasterCold, MASTER_CTRL_DELAY_MICROCHUNKS, MasterCtrlDelayRingCold,
    MasterCtrlSnapshotCold,
};

use core::mem::{align_of, size_of};
use dnam_kernels::{
    CORE_OS_LIVE_LATENCY_1X, CORE_OS_MASTER_2X_LATENCY_1X, CORE_OS_MASTER_4X_LATENCY_1X,
    HbFir15Downsampler4Ch, HbFir15UpsamplerStereo,
};

use crate::graph::GraphBackend;
use crate::math::SimdVec4;

/// Core oversampling execution mode (`ADR-0026`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum RenderMode {
    /// Low-latency real-time mode (`2x` `15`-tap `HB-FIR` capped at `384 kHz`, `PDC = 7` or `0`).
    #[default]
    Live = 0,
    /// Offline bounce / mastering mode (`4x` `63/23`-tap `HB-FIR` + `Box2`, `PDC = 37`, `31`, or `0`).
    Master = 1,
}

/// Exact `232 B` (`align(4)`) `Live`-mode `15`-tap `2x` `HB-FIR` hot state stored inside
/// `SonaLiteHotPrefix` (`56 B` upsampler + `176 B` 4-channel downsampler).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CoreOversamplerLiveHot {
    /// `1x -> 2x` stereo `15`-tap `HB-FIR` upsampler (`56 B`).
    pub up15: HbFir15UpsamplerStereo,
    /// `2x -> 1x` 4-channel `[pre_L, pre_R, dir_L, dir_R]` `15`-tap `HB-FIR` downsampler (`176 B`).
    pub dn15: HbFir15Downsampler4Ch,
}

const _: () = assert!(size_of::<CoreOversamplerLiveHot>() == 232);
const _: () = assert!(align_of::<CoreOversamplerLiveHot>() == 4);

impl CoreOversamplerLiveHot {
    /// Creates a zero-initialized [`CoreOversamplerLiveHot`] at `+0.0` equilibrium.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            up15: HbFir15UpsamplerStereo::new(),
            dn15: HbFir15Downsampler4Ch::new(),
        }
    }

    /// Resets all upsampler and downsampler delay registers to `+0.0`.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.up15.reset();
        self.dn15.reset();
    }

    /// Upsamples one `1x` stereo frame `[in_l, in_r]` into two `2x` stereo frames `[sub0, sub1]`.
    #[inline(always)]
    pub fn upsample_stereo(&mut self, in_l: f32, in_r: f32) -> [[f32; 2]; 2] {
        self.up15.step([in_l, in_r])
    }

    /// Downsamples two `2x` 4-channel frames `(sub0, sub1)` back to one `1x` 4-channel frame.
    #[inline(always)]
    pub fn downsample_4ch<B: GraphBackend>(&mut self, sub0: [f32; 4], sub1: [f32; 4]) -> [f32; 4] {
        self.dn15
            .step_vec4(B::Vec4::load(&sub0), B::Vec4::load(&sub1))
            .to_array()
    }
}

/// Unified 2-mode rate & PDC descriptor for the Core-Only Multirate oversampler (`ADR-0026`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CoreOversampler {
    /// Active render mode (`Live` vs `Master`).
    pub mode: RenderMode,
    /// Effective Core integer oversampling multiplier `m_core in {1, 2, 4}` (`m_core * f_host <= 384 kHz`).
    pub m_core: usize,
    /// Exact integer Plugin Delay Compensation (`PDC`) in `1x` host samples (`7`, `37`, `31`, or `0`).
    pub pdc_samples: usize,
}

impl Default for CoreOversampler {
    fn default() -> Self {
        Self::for_rate(RenderMode::Live, 48_000.0)
    }
}

impl CoreOversampler {
    /// Computes `(m_core, pdc_samples)` for `(mode, host_rate_hz)` capped at `f_s_core <= 384 kHz`
    /// (`m_core in {1, 2, 4}`, strictly comparison-based -> `0 FDIV`).
    #[must_use]
    pub const fn resolve_m_core_and_pdc(mode: RenderMode, host_rate_hz: f64) -> (usize, usize) {
        let m_core = if host_rate_hz <= 0.0 || !host_rate_hz.is_finite() {
            1
        } else {
            match mode {
                RenderMode::Live => {
                    if host_rate_hz <= 192_000.0 {
                        2
                    } else {
                        1
                    }
                }
                RenderMode::Master => {
                    if host_rate_hz <= 96_000.0 {
                        4
                    } else if host_rate_hz <= 192_000.0 {
                        2
                    } else {
                        1
                    }
                }
            }
        };
        let pdc = match (mode, m_core) {
            (RenderMode::Live, 2) => CORE_OS_LIVE_LATENCY_1X,
            (RenderMode::Master, 4) => CORE_OS_MASTER_4X_LATENCY_1X,
            (RenderMode::Master, 2) => CORE_OS_MASTER_2X_LATENCY_1X,
            _ => 0,
        };
        (m_core, pdc)
    }

    /// Returns the exact integer PDC latency `D_os` in host samples (`7 / 37 / 31 / 0`).
    #[must_use]
    pub const fn pdc_latency_samples(mode: RenderMode, host_rate_hz: f64) -> usize {
        Self::resolve_m_core_and_pdc(mode, host_rate_hz).1
    }

    /// Constructs a [`CoreOversampler`] descriptor for `(mode, host_rate_hz)`.
    #[must_use]
    pub const fn for_rate(mode: RenderMode, host_rate_hz: f64) -> Self {
        let (m_core, pdc_samples) = Self::resolve_m_core_and_pdc(mode, host_rate_hz);
        Self {
            mode,
            m_core,
            pdc_samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{MathBackend, ScalarBackend};
    use dnam_kernels::{HB15_DELAY_SAMPLES_1X, StereoVec8};
    use proptest::prelude::*;

    #[test]
    fn core_oversampler_pdc_and_impulse_response_exact() {
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Live, 44_100.0),
            7
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Live, 48_000.0),
            7
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Live, 192_000.0),
            7
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Live, 352_800.0),
            0
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Live, 384_000.0),
            0
        );

        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Master, 44_100.0),
            37
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Master, 48_000.0),
            37
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Master, 96_000.0),
            37
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Master, 176_400.0),
            31
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Master, 192_000.0),
            31
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Master, 352_800.0),
            0
        );
        assert_eq!(
            CoreOversampler::pdc_latency_samples(RenderMode::Master, 384_000.0),
            0
        );

        let mut live = CoreOversamplerLiveHot::new();
        let mut ir_live = [0.0_f32; 24];
        for (n, dst) in ir_live.iter_mut().enumerate() {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let up = live.upsample_stereo(x, x);
            let dn = live.downsample_4ch::<ScalarBackend>(
                [up[0][0], up[0][1], up[0][0], up[0][1]],
                [up[1][0], up[1][1], up[1][0], up[1][1]],
            );
            *dst = dn[0];
        }
        let peak_live = ir_live
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .expect("non-empty")
            .0;
        assert_eq!(peak_live, HB15_DELAY_SAMPLES_1X);

        let mut master = CoreOversamplerMasterCold::new();
        let mut ir_4x = [0.0_f32; 96];
        for (n, dst) in ir_4x.iter_mut().enumerate() {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let up = master.upsample_1x_to_4x(x, x);
            let sub4 = up.map(|lr| [lr[0], lr[1], lr[0], lr[1]]);
            let dn = master.downsample_4x_to_1x::<ScalarBackend>(sub4);
            *dst = dn[0];
        }
        let peak_4x = ir_4x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .expect("non-empty")
            .0;
        assert_eq!(peak_4x, 37);

        master.reset();
        let mut ir_2x = [0.0_f32; 80];
        for (n, dst) in ir_2x.iter_mut().enumerate() {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let up = master.upsample_1x_to_2x(x, x);
            let dn = master.downsample_2x_to_1x::<ScalarBackend>([
                [up[0][0], up[0][1], up[0][0], up[0][1]],
                [up[1][0], up[1][1], up[1][0], up[1][1]],
            ]);
            *dst = dn[0];
        }
        let peak_2x = ir_2x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .expect("non-empty")
            .0;
        assert_eq!(peak_2x, 31);
    }

    #[test]
    fn master_box2_cubic_alias_floor_and_ctrl_delay_ring_alignment() {
        // Drive a pure cubic nonlinearity y = x - x^3/3 at 6.23 kHz @ 48 kHz (12x harmonic aliases
        // fold across 96 kHz in Live 2x, suppressed by HB63+HB23+Box2 in Master 4x).
        let fs_1x = 48_000.0_f64;
        let f_tone = 6_230.0_f64;
        let n_fft = 1_024_usize;
        let warmup = 256_usize;

        let mut live = CoreOversamplerLiveHot::new();
        let mut master = CoreOversamplerMasterCold::new();
        let mut y_live = [0.0_f32; 1_024];
        let mut y_master = [0.0_f32; 1_024];

        let soft_clip = |x: f32| (x * 2.5).tanh();
        let mut prev_nl_master = 0.0_f32;

        for n in 0..(warmup + n_fft) {
            let x = (core::f64::consts::TAU * f_tone * (n as f64) / fs_1x).sin() as f32 * 0.9;

            let up_l = live.upsample_stereo(x, x);
            let nl_l0 = soft_clip(up_l[0][0]);
            let nl_l1 = soft_clip(up_l[1][0]);
            let out_l = live.downsample_4ch::<ScalarBackend>(
                [nl_l0, nl_l0, up_l[0][0], up_l[0][1]],
                [nl_l1, nl_l1, up_l[1][0], up_l[1][1]],
            );

            let up_m = master.upsample_1x_to_4x(x, x);
            let mut sub_m = [[0.0_f32; 4]; 4];
            for s in 0..4 {
                let raw = soft_clip(up_m[s][0]);
                let b2 = 0.5 * (prev_nl_master + raw);
                prev_nl_master = raw;
                sub_m[s] = [b2, b2, up_m[s][0], up_m[s][1]];
            }
            let out_m = master.downsample_4x_to_1x::<ScalarBackend>(sub_m);

            if n >= warmup {
                y_live[n - warmup] = out_l[0];
                y_master[n - warmup] = out_m[0];
            }
        }

        // Measure in-band spurious alias energy above fundamental + harmonics not near expected integer bins
        // and verify Master has cleanly lower aliasing than Live!
        let dft_mag_sq = |sig: &[f32; 1_024], bin: usize| -> f64 {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (n, &v) in sig.iter().enumerate() {
                let w = 0.5 * (1.0 - (core::f64::consts::TAU * n as f64 / 1024.0).cos());
                let ang = core::f64::consts::TAU * (bin as f64) * (n as f64) / 1024.0;
                re += f64::from(v) * w * ang.cos();
                im -= f64::from(v) * w * ang.sin();
            }
            re * re + im * im
        };
        // Fundamental 6.23 kHz -> bin ~ 133; 3rd harmonic 18.69 kHz -> bin ~ 399;
        // 5th harmonic 31.15 kHz aliases to 16.85 kHz -> bin ~ 359;
        // 7th harmonic 43.61 kHz aliases to 4.39 kHz -> bin ~ 94!
        let f0_bin = (f_tone * 1024.0 / fs_1x).round() as usize;
        let alias7_bin = ((48_000.0 - 7.0 * f_tone) * 1024.0 / fs_1x).round() as usize;
        let fund_live_db = 10.0 * dft_mag_sq(&y_live, f0_bin).log10();
        let a7_live_db = 10.0 * dft_mag_sq(&y_live, alias7_bin).max(1e-24).log10() - fund_live_db;
        let fund_master_db = 10.0 * dft_mag_sq(&y_master, f0_bin).log10();
        let a7_master_db =
            10.0 * dft_mag_sq(&y_master, alias7_bin).max(1e-24).log10() - fund_master_db;
        assert!(
            a7_master_db < a7_live_db - 15.0 && a7_master_db < -60.0,
            "Master alias7 {a7_master_db:.2} dB vs Live {a7_live_db:.2} dB"
        );

        // Verify 4-slot cold control ring delta_D_up = 16 samples alignment.
        let mut ring = MasterCtrlDelayRingCold::new();
        let mut s0 = MasterCtrlSnapshotCold::default();
        s0.latched_stage_params[0] = 10.0;
        ring.seed(s0);
        for step in 1..=8_usize {
            let mut nxt = MasterCtrlSnapshotCold::default();
            nxt.latched_stage_params[0] = 10.0 + step as f32;
            let popped = ring.push_and_pop(nxt);
            let expected = if step <= 4 {
                10.0
            } else {
                10.0 + (step - 4) as f32
            };
            assert_eq!(popped.latched_stage_params[0], expected);
        }

        // Verify apply_box2_quad matches lin_mix = 1.0 bypass and 0.5 average.
        type S8 = <ScalarBackend as MathBackend>::Stereo8;
        type V4 = <ScalarBackend as MathBackend>::Vec4;
        let lin = S8::from_lr(
            V4::from_array([0.0, 1.0, 0.25, 1.0]),
            V4::from_array([1.0, 0.0, 0.5, 1.0]),
        );
        let a1 = S8::from_lr(
            V4::from_array([2.0, 4.0, 6.0, 8.0]),
            V4::from_array([1.0, 3.0, 5.0, 7.0]),
        );
        let b1 = S8::from_lr(
            V4::from_array([4.0, 6.0, 8.0, 10.0]),
            V4::from_array([3.0, 5.0, 7.0, 9.0]),
        );
        let (bo, be) = master.apply_box2_quad::<ScalarBackend>(0, a1, b1, lin);
        let (bo_l, _) = bo.into_lr();
        let (be_l, _) = be.into_lr();
        assert_eq!(bo_l.to_array(), [1.0, 4.0, 3.0, 8.0]);
        assert_eq!(be_l.to_array(), [2.0, 6.0, 4.0, 10.0]);
    }

    proptest! {
        #[test]
        fn prop_core_oversampler_m_core_never_exceeds_384k(
            rate_hz in 44_100_u32..=384_000_u32,
            is_master in any::<bool>(),
        ) {
            let mode = if is_master { RenderMode::Master } else { RenderMode::Live };
            let os = CoreOversampler::for_rate(mode, f64::from(rate_hz));
            prop_assert!(os.m_core * (rate_hz as usize) <= 384_000);
            prop_assert!(matches!(os.m_core, 1 | 2 | 4));
            prop_assert!(matches!(os.pdc_samples, 0 | 7 | 31 | 37));
        }
    }
}
