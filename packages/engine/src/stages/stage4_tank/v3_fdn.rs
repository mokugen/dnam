//! Multi-quad `SONA` Stage IV spatial & LFO-modulated FDN tank (`stage4_tank/v3_fdn.rs`).
//!
//! Supports `1` quad (`4` lines) on `Lite`, `2` quads (`8` lines) on `Full`, `3` quads (`12` lines)
//! on `Pro`, and `4` quads (`16` lines) on `Ultra`, plus partitioned Schroeder all-pass diffusion,
//! frame-major `fp16` inline L1D storage, cold power-of-two `ext_lines` (`"fp16"` | `"f32"`),
//! `0-ULP` write-locked integer-lag freeze, and stereo-linked `BallisticDuckingVca`.

pub mod lfo_interp;
pub mod params;
pub mod quad_state;

use dnam_kernels::DC_BLOCKER_5HZ_48K_F32;

use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, newton6_reciprocal_f64};
use crate::types::Tier;
pub(crate) use lfo_interp::PartitionedSchroederBank;
#[cfg(test)]
use lfo_interp::SubblockBuffers;
pub(crate) use params::clamp_quad_params_to_cap;
pub use params::{
    SONA_TANK_MAX_LINES, SONA_TANK_QUADS, STAGE4_SCHROEDER_MAX_DELAY,
    STAGE4_SCHROEDER_REF_DELAYS_48K, Stage4ExtFormat, Stage4QuadParams, V3_TANK_CAP,
    V3_TANK_INLINE_MAX_DELAY, V3_TANK_LINES, V3_TANK_MAX_DECLARED_DELAY, V3_TANK_MAX_EXT_CAP,
    V3_TANK_MAX_LFO_DEPTH, V3_TANK_RING_STRIDE, V3Stage4TankParams,
};
pub(crate) use quad_state::{Stage4ExtRing, Stage4ExtStorage};
pub use quad_state::{Stage4QuadHotCompact, Stage4QuadHotState};

const V3_TANK_MASK: usize = V3_TANK_CAP - 1;

/// Hot `1..=4`-quad Stage IV `Op-T` FDN tank with frame-major inline L1D storage for `Quad 0`
/// and cold power-of-two extended / expansion quad storage.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct V3Stage4TankHot {
    pub quad0: Stage4QuadHotState,
    pub params: V3Stage4TankParams,
    pub active_quads: usize,
    /// Unitary captured hardware freeze latch (`ref[dsp.stage4.captured-freeze]`).
    pub freeze: bool,
    pub frames: [[u16; V3_TANK_LINES]; V3_TANK_CAP],
    pub extra_quads: [Stage4QuadHotState; 3],
}

impl V3Stage4TankHot {
    /// Returns active Stage IV L1D working-set bytes touched per sample for `tier`.
    #[must_use]
    pub const fn active_hot_bytes(tier: Tier) -> usize {
        let quads = tier.tank_groups();
        let quad_hdr =
            core::mem::size_of::<Stage4QuadHotState>() - core::mem::size_of::<usize>() * 2;
        quads * quad_hdr + V3Stage4TankParams::active_ring_bytes(tier) + 16
    }

    /// Returns the active ring capacity of `Quad 0` (`V3_TANK_CAP` inline or cold `ext_cap`).
    #[inline(always)]
    #[must_use]
    pub fn effective_cap(&self) -> usize {
        self.quad0.effective_cap(true)
    }

    /// Initializes Stage IV FDN tank from cold parameters, pre-allocating cold rings cleanly.
    #[must_use]
    pub fn from_cold_with_min_cap(params: &V3Stage4TankParams, min_frames_cap: usize) -> Self {
        let mut clamped = *params;
        clamped.clamp_in_place();
        let aq = clamped.active_quads.clamp(1, SONA_TANK_QUADS);
        let q0_p = clamped.effective_quad_params(0);
        let quad0 = Stage4QuadHotState::from_params(&q0_p, min_frames_cap, true);
        let extra_quads = core::array::from_fn(|idx| {
            let q = idx + 1;
            let qp = clamped.effective_quad_params(q);
            if q < aq {
                Stage4QuadHotState::from_params(&qp, min_frames_cap, false)
            } else {
                let mut idle = Stage4QuadHotState::from_params(&qp, 0, true);
                idle.ext_ring = None;
                idle
            }
        });
        clamped.quads[0] = quad0.params;
        clamped.sync_top_from_quad0();
        for idx in 0..3 {
            clamped.quads[idx + 1] = extra_quads[idx].params;
        }
        Self {
            quad0,
            params: clamped,
            active_quads: aq,
            freeze: false,
            frames: [[0; V3_TANK_LINES]; V3_TANK_CAP],
            extra_quads,
        }
    }

    /// Compiles Stage IV FDN tank hot state from cold parameters.
    #[must_use]
    pub fn from_cold(params: &V3Stage4TankParams) -> Self {
        Self::from_cold_with_min_cap(params, 0)
    }

    /// Ensures cold extended delay buffers and Schroeder segments accommodate `params` (`0` RT allocs).
    pub fn ensure_cold_ext_cap(&mut self, params: &V3Stage4TankParams, min_frames_cap: usize) {
        let mut p = *params;
        p.clamp_in_place();
        let aq = p.active_quads.clamp(1, SONA_TANK_QUADS);
        self.active_quads = aq;
        let q0_p = p.effective_quad_params(0);
        self.quad0.ensure_cold_ext_cap(&q0_p, min_frames_cap, true);
        for idx in 0..3 {
            let q = idx + 1;
            if q < aq {
                let qp = p.effective_quad_params(q);
                self.extra_quads[idx].ensure_cold_ext_cap(&qp, min_frames_cap, false);
            }
        }
        self.params = p;
        self.params.quads[0] = self.quad0.params;
        self.params.sync_top_from_quad0();
        for idx in 0..3 {
            self.params.quads[idx + 1] = self.extra_quads[idx].params;
        }
    }

    /// Updates Stage IV parameters in place without clearing delay memory (`0` heap allocs).
    pub fn update_params(&mut self, params: &V3Stage4TankParams) {
        let mut p = *params;
        p.clamp_in_place();
        self.active_quads = p.active_quads.clamp(1, SONA_TANK_QUADS);
        let mut q0 = p.effective_quad_params(0);
        let cap0 = self.quad0.effective_cap(true);
        clamp_quad_params_to_cap(&mut q0, cap0, V3_TANK_MAX_LFO_DEPTH);
        self.quad0.apply_runtime_params(q0);
        for idx in 0..3 {
            let mut qp = p.effective_quad_params(idx + 1);
            let cap_q = self.extra_quads[idx].effective_cap(false);
            if cap_q >= 8 {
                clamp_quad_params_to_cap(&mut qp, cap_q, V3_TANK_MAX_LFO_DEPTH);
            }
            self.extra_quads[idx].apply_runtime_params(qp);
            p.quads[idx + 1] = qp;
        }
        self.params = p;
        self.params.quads[0] = q0;
        self.params.sync_top_from_quad0();
    }

    /// Reloads Stage IV tank parameters in place and resets runtime state (`0` allocs, `0` drops).
    pub fn reload_in_place(&mut self, params: &V3Stage4TankParams) {
        self.quad0.host_sample_rate_hz = 48_000.0;
        self.quad0.wet_dc_pole = DC_BLOCKER_5HZ_48K_F32;
        self.quad0.wet_dc_alpha = 0.5 * (1.0 + DC_BLOCKER_5HZ_48K_F32);
        for q in &mut self.extra_quads {
            q.host_sample_rate_hz = 48_000.0;
            q.wet_dc_pole = DC_BLOCKER_5HZ_48K_F32;
            q.wet_dc_alpha = 0.5 * (1.0 + DC_BLOCKER_5HZ_48K_F32);
        }
        self.update_params(params);
        self.freeze = false;
        self.quad0.snap_psi_immediate(0.0);
        for q in &mut self.extra_quads {
            q.snap_psi_immediate(0.0);
        }
        self.reset();
    }

    /// Reconfigures delays, LFO rate/depth, DC pole, VCA, and Schroeder diffusers under ZOH (`cold`).
    pub fn reconfigure_with_zoh(&mut self, base: &V3Stage4TankParams, zoh: &ZohRescaledParams) {
        let mut clamped = *base;
        clamped.clamp_in_place();
        let inv_rho = if (zoh.rho_sr - 1.0).abs() >= 1.0e-12 {
            newton6_reciprocal_f64(zoh.rho_sr.max(1.0e-6))
        } else {
            1.0
        };
        let host_sr = (48_000.0 * inv_rho) as f32;
        self.quad0.host_sample_rate_hz = host_sr;
        self.quad0.reconfigure_wet_dc(zoh);
        let q0_base = clamped.effective_quad_params(0);
        let q0_zoh_probe =
            q0_base.with_zoh_rescaled(zoh, V3_TANK_MAX_EXT_CAP, V3_TANK_MAX_LFO_DEPTH * 4.0);
        let q0_max_d = q0_zoh_probe
            .delay_lengths
            .iter()
            .copied()
            .max()
            .unwrap_or(67);
        let q0_needed = q0_max_d.saturating_add((q0_zoh_probe.lfo_depth_samples as usize) + 6);
        let need_ext0 = q0_base.ext_format == Stage4ExtFormat::F32
            || q0_max_d > V3_TANK_INLINE_MAX_DELAY
            || q0_needed > V3_TANK_CAP;
        self.quad0.ext_active = need_ext0 && self.quad0.ext_ring.is_some();
        let cap0 = self.quad0.effective_cap(true);
        let max_depth0 = ((cap0 / 4) as f32).min(V3_TANK_MAX_LFO_DEPTH * 4.0);
        let q0_zoh = q0_base.with_zoh_rescaled(zoh, cap0, max_depth0);
        self.quad0.apply_runtime_params(q0_zoh);
        self.params.quads[0] = q0_zoh;
        self.params.sync_top_from_quad0();
        for idx in 0..3 {
            self.extra_quads[idx].host_sample_rate_hz = host_sr;
            self.extra_quads[idx].reconfigure_wet_dc(zoh);
            let cap_q = self.extra_quads[idx].effective_cap(false);
            if cap_q >= 8 {
                let md = ((cap_q / 4) as f32).min(V3_TANK_MAX_LFO_DEPTH * 4.0);
                let qz = clamped
                    .effective_quad_params(idx + 1)
                    .with_zoh_rescaled(zoh, cap_q, md);
                self.extra_quads[idx].apply_runtime_params(qz);
                self.params.quads[idx + 1] = qz;
            }
        }
    }

    /// Snaps the Givens rotor across active quads directly to `base_psi + psi_delta`.
    #[inline]
    pub fn snap_psi_immediate(&mut self, psi_delta: f32) {
        self.quad0.snap_psi_immediate(psi_delta);
        for q in &mut self.extra_quads {
            q.snap_psi_immediate(psi_delta);
        }
    }

    /// Updates Givens rotation rotor at `3 kHz` (`tick & 15 == 0`) with bounded slew.
    #[inline(always)]
    pub fn refresh_3khz(&mut self, psi_delta: f32) {
        self.quad0.refresh_3khz(psi_delta);
        for idx in 0..self.active_quads.saturating_sub(1).min(3) {
            self.extra_quads[idx].refresh_3khz(psi_delta);
        }
    }

    /// Current `Quad 0` instantaneous Givens angle.
    #[inline(always)]
    #[must_use]
    pub const fn current_psi(&self) -> f32 {
        self.quad0.current_psi
    }

    /// Current `Quad 0` `Q0.32` LFO phase increment.
    #[inline(always)]
    #[must_use]
    pub const fn lfo_inc_q32(&self) -> u32 {
        self.quad0.lfo_inc_q32
    }

    /// Resets delay frames, LFO phase, DC blockers, Schroeder buffers, and VCA to silence.
    pub fn reset(&mut self) {
        self.frames.fill([0; V3_TANK_LINES]);
        self.quad0.reset();
        for q in &mut self.extra_quads {
            q.reset();
        }
    }

    /// Reads the static integer-delay tap values of `Quad 0` prior to advancing the current sample.
    #[must_use]
    pub fn peek_integer_taps<B: MathBackend>(&self) -> [f32; V3_TANK_LINES] {
        let wr = self.quad0.write_ptr as usize;
        let d = self.quad0.params.delay_lengths;
        if self.quad0.ext_active
            && let Some(ext) = self.quad0.ext_ring.as_deref()
        {
            let m = ext.mask;
            match &ext.storage {
                Stage4ExtStorage::Fp16(buf) => {
                    let raw = [
                        buf[((wr + d[0]) & m) * 4],
                        buf[((wr + d[1]) & m) * 4 + 1],
                        buf[((wr + d[2]) & m) * 4 + 2],
                        buf[((wr + d[3]) & m) * 4 + 3],
                    ];
                    B::fp16x4_to_f32x4(&raw)
                }
                Stage4ExtStorage::F32(buf) => [
                    buf[((wr + d[0]) & m) * 4],
                    buf[((wr + d[1]) & m) * 4 + 1],
                    buf[((wr + d[2]) & m) * 4 + 2],
                    buf[((wr + d[3]) & m) * 4 + 3],
                ],
            }
        } else {
            let raw = [
                self.frames[(wr + d[0]) & V3_TANK_MASK][0],
                self.frames[(wr + d[1]) & V3_TANK_MASK][1],
                self.frames[(wr + d[2]) & V3_TANK_MASK][2],
                self.frames[(wr + d[3]) & V3_TANK_MASK][3],
            ];
            B::fp16x4_to_f32x4(&raw)
        }
    }

    /// Steps all active Stage IV FDN quads (`1..=4` quads) unconditionally (`0 FDIV`, `0 FSQRT`).
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let [mut sum_l, mut sum_r] = self.quad0.step_stereo_with_opt_inline::<B>(
            Some(&mut self.frames),
            self.freeze,
            in_l,
            in_r,
        );
        for idx in 0..self.active_quads.saturating_sub(1).min(3) {
            let [ql, qr] = self.extra_quads[idx].step_stereo_with_opt_inline::<B>(
                None,
                self.freeze,
                in_l,
                in_r,
            );
            sum_l += ql;
            sum_r += qr;
        }
        [sum_l, sum_r]
    }

    /// Sub-block const-generic quad stepper (`QUADS in 1..=4`, `0` dynamic quad loop overhead).
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn step_subblock_in<B: MathBackend, const QUADS: usize, const MONO: bool>(
        &mut self,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32],
        out_r: &mut [f32],
        len: usize,
    ) {
        if self.freeze {
            self.step_subblock_in_mode::<B, QUADS, MONO, true>(in_l, in_r, out_l, out_r, len);
        } else {
            self.step_subblock_in_mode::<B, QUADS, MONO, false>(in_l, in_r, out_l, out_r, len);
        }
    }

    #[cfg(test)]
    #[inline(always)]
    fn step_subblock_in_mode<
        B: MathBackend,
        const QUADS: usize,
        const MONO: bool,
        const FREEZE: bool,
    >(
        &mut self,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32],
        out_r: &mut [f32],
        len: usize,
    ) {
        const { assert!(QUADS >= 1 && QUADS <= SONA_TANK_QUADS) };
        self.quad0.step_subblock_with_opt_inline::<B, MONO, FREEZE>(
            Some(&mut self.frames),
            SubblockBuffers {
                in_l,
                in_r,
                out_l,
                out_r,
                len,
                accumulate_output: false,
            },
        );
        for idx in 0..(QUADS - 1) {
            self.extra_quads[idx].step_subblock_with_opt_inline::<B, MONO, FREEZE>(
                None,
                SubblockBuffers {
                    in_l,
                    in_r,
                    out_l,
                    out_r,
                    len,
                    accumulate_output: true,
                },
            );
        }
    }

    /// Test-only sub-block stepper combining `[s2_dir, cab_out, clean_sub]` across active quads.
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn step_subblock<B: MathBackend, const MONO: bool>(
        &mut self,
        s2_dir: &[[f32; 4]; 16],
        cab_out: &[[f32; 2]; 16],
        clean_sub: &[[f32; 2]; 16],
        out_l: &mut [f32],
        out_r: &mut [f32],
        len: usize,
    ) {
        let mut in_l = [0.0_f32; 16];
        let mut in_r = [0.0_f32; 16];
        let n = len.min(16);
        for i in 0..n {
            in_l[i] = s2_dir[i][2] + cab_out[i][0] + clean_sub[i][0];
            in_r[i] = s2_dir[i][3] + cab_out[i][1] + clean_sub[i][1];
        }
        match self.active_quads {
            1 => self.step_subblock_in::<B, 1, MONO>(&in_l, &in_r, out_l, out_r, len),
            2 => self.step_subblock_in::<B, 2, MONO>(&in_l, &in_r, out_l, out_r, len),
            3 => self.step_subblock_in::<B, 3, MONO>(&in_l, &in_r, out_l, out_r, len),
            _ => self.step_subblock_in::<B, 4, MONO>(&in_l, &in_r, out_l, out_r, len),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};

    #[test]
    fn test_stage4_tank_parity_bibo_zoh_freeze_and_multi_quad() {
        let mut tp = V3Stage4TankParams::identity();
        tp.wet_mix = 0.35;
        tp.g_loop = 0.88;
        let mut t_simd = V3Stage4TankHot::from_cold(&tp);
        let mut t_scal = V3Stage4TankHot::from_cold(&tp);

        assert_eq!(
            t_simd
                .step_stereo::<DefaultMathBackend>(0.0, 0.0)
                .map(f32::to_bits),
            [0; 2]
        );
        t_simd.reset();
        t_scal.reset();

        for n in 0..600 {
            let x = if n < 400 {
                0.8 * (n as f32 * 0.07).sin()
            } else {
                0.0
            };
            let yl = t_simd.step_stereo::<DefaultMathBackend>(x, 0.6 * x);
            let yr = t_scal.step_stereo::<ScalarBackend>(x, 0.6 * x);
            assert!(yl[0].is_finite() && yl[1].is_finite());
            assert!(yl[0].abs() < 20.0 && yl[1].abs() < 20.0);
            assert!((yl[0] - yr[0]).abs() < 5.0e-4, "L parity diff at {n}");
            assert!((yl[1] - yr[1]).abs() < 5.0e-4, "R parity diff at {n}");
        }

        // Verify f32 external ring yields tighter (< 1e-5) NEON/Scalar parity under high loop gain.
        let mut tp_f32 = tp;
        tp_f32.ext_format = Stage4ExtFormat::F32;
        tp_f32.delay_lengths = [131, 151, 173, 197];
        let mut t_f32_simd = V3Stage4TankHot::from_cold(&tp_f32);
        let mut t_f32_scal = V3Stage4TankHot::from_cold(&tp_f32);
        for n in 0..400 {
            let x = 0.8 * (n as f32 * 0.07).sin();
            let yl = t_f32_simd.step_stereo::<DefaultMathBackend>(x, 0.6 * x);
            let yr = t_f32_scal.step_stereo::<ScalarBackend>(x, 0.6 * x);
            assert!((yl[0] - yr[0]).abs() < 1.0e-5, "F32 L parity diff at {n}");
            assert!((yl[1] - yr[1]).abs() < 1.0e-5, "F32 R parity diff at {n}");
        }

        let zoh96 = ZohRescaledParams::new(48_000, 96_000);
        let mut t_zoh = V3Stage4TankHot::from_cold(&tp);
        t_zoh.reconfigure_with_zoh(&tp, &zoh96);
        assert_eq!(t_zoh.params.delay_lengths[0], tp.delay_lengths[0] * 2);

        // Verify 0-ULP freeze preserves frame buffer bit-exactness across sustained playback.
        let frozen_before = t_simd.frames;
        let frozen_wr = t_simd.quad0.write_ptr;
        t_simd.freeze = true;
        let mut freeze_peak = 0.0_f32;
        for _ in 0..1_024 {
            let [fl, fr] = t_simd.step_stereo::<DefaultMathBackend>(0.0, 0.0);
            freeze_peak = freeze_peak.max(fl.abs()).max(fr.abs());
        }
        assert!(freeze_peak > 1.0e-3);
        assert_eq!(t_simd.frames, frozen_before);
        assert_eq!(t_simd.quad0.write_ptr, frozen_wr);

        t_simd.freeze = false;
        let extreme = t_simd.step_stereo::<DefaultMathBackend>(1.0e9, -1.0e9);
        assert!(extreme[0].is_finite() && extreme[1].is_finite());

        // Verify Ultra 4-quad (16 lines) execution + Schroeder diffuser > 255 ZOH without truncation.
        let mut tp_ultra = V3Stage4TankParams::identity_for_tier(Tier::Ultra);
        tp_ultra.allpass_delays = [19, 31, 47, 223];
        tp_ultra.allpass_g = [0.6, -0.5, 0.45, 0.55];
        tp_ultra.quads[3].wet_mix = 0.5;
        tp_ultra.quads[3].ext_format = Stage4ExtFormat::F32;
        let mut t_ultra = V3Stage4TankHot::from_cold(&tp_ultra);
        assert_eq!(t_ultra.active_quads, 4);
        let zoh384 = ZohRescaledParams::new(48_000, 384_000);
        t_ultra.ensure_cold_ext_cap(&tp_ultra, 4_096);
        t_ultra.reconfigure_with_zoh(&tp_ultra, &zoh384);
        assert_eq!(t_ultra.params.allpass_delays[3], 223 * 8);
        let y_u = t_ultra.step_stereo::<DefaultMathBackend>(0.3, -0.2);
        assert!(y_u[0].is_finite() && y_u[1].is_finite());
    }

    #[test]
    fn test_stage4_long_delay_cold_ext_lines_f32_4x_zoh_and_lfo() {
        let mut tp_lfo = V3Stage4TankParams {
            delay_lengths: [96, 104, 112, 120],
            g_loop: 0.995,
            inject_gain: 0.5,
            dry_mix: 0.2,
            wet_mix: 0.8,
            lfo_rate_hz: 3.2,
            lfo_depth_samples: 24.0,
            lfo_stereo_phase: core::f32::consts::FRAC_PI_2,
            ..V3Stage4TankParams::identity()
        };
        tp_lfo.clamp_in_place();
        let mut t_lfo = V3Stage4TankHot::from_cold(&tp_lfo);
        let mut t_lfo_sub = V3Stage4TankHot::from_cold(&tp_lfo);
        let mut t_lfo_scal = V3Stage4TankHot::from_cold(&tp_lfo);
        let s2_dir = [[0.0, 0.0, 0.2, -0.15]; 16];
        let cab_zero = [[0.0, 0.0]; 16];
        let mut sub_l = [0.0_f32; 16];
        let mut sub_r = [0.0_f32; 16];
        t_lfo_sub.step_subblock::<DefaultMathBackend, false>(
            &s2_dir, &cab_zero, &cab_zero, &mut sub_l, &mut sub_r, 16,
        );
        for k in 0..16 {
            let [sl, sr] = t_lfo.step_stereo::<DefaultMathBackend>(0.2, -0.15);
            let [scl, scr] = t_lfo_scal.step_stereo::<ScalarBackend>(0.2, -0.15);
            assert_eq!(sl.to_bits(), sub_l[k].to_bits());
            assert_eq!(sr.to_bits(), sub_r[k].to_bits());
            assert!((sl - scl).abs() < 1e-5 && (sr - scr).abs() < 1e-5);
        }

        let mut tp = V3Stage4TankParams {
            delay_lengths: [20_160, 20_167, 20_173, 20_179],
            g_loop: 0.55,
            inject_gain: 1.0,
            dry_mix: 0.0,
            wet_mix: 1.0,
            tap_gains: [1.0, 1.0, 0.0, 0.0],
            lfo_rate_hz: 1.5,
            lfo_depth_samples: 18.0,
            lfo_stereo_phase: core::f32::consts::FRAC_PI_2,
            ext_format: Stage4ExtFormat::F32,
            ..V3Stage4TankParams::identity()
        };
        tp.clamp_in_place();
        let mut t_ext = V3Stage4TankHot::from_cold(&tp);
        assert!(t_ext.effective_cap() >= 80_640);
        let mut pre_echo_peak = 0.0_f32;
        let mut echo_peak = 0.0_f32;
        for n in 0..20_300 {
            let x = if n == 0 { 0.8 } else { 0.0 };
            let [yl, yr] = t_ext.step_stereo::<DefaultMathBackend>(x, x);
            assert!(yl.is_finite() && yr.is_finite());
            if n < 20_100 {
                pre_echo_peak = pre_echo_peak.max(yl.abs());
            } else {
                echo_peak = echo_peak.max(yl.abs());
            }
        }
        assert!(pre_echo_peak < 1e-6);
        assert!(echo_peak > 0.05);

        let zoh_4x = ZohRescaledParams::new(48_000, 192_000);
        t_ext.reconfigure_with_zoh(&tp, &zoh_4x);
        assert_eq!(t_ext.params.delay_lengths[0], 80_640);
        t_ext.reset();
        assert_eq!(
            t_ext.peek_integer_taps::<DefaultMathBackend>(),
            [0.0; V3_TANK_LINES]
        );
    }

    #[test]
    fn test_stage4_120_delay_spill_loop_dc_unity_and_givens_feedback() {
        assert_eq!(V3_TANK_CAP, 128);
        assert_eq!(
            core::mem::size_of::<[[u16; V3_TANK_LINES]; V3_TANK_CAP]>(),
            1_024
        );

        let mut tp_inline = V3Stage4TankParams {
            delay_lengths: [53, 67, 79, 120],
            lfo_depth_samples: 0.0,
            ..V3Stage4TankParams::identity()
        };
        tp_inline.clamp_in_place();
        let t_inline = V3Stage4TankHot::from_cold(&tp_inline);
        assert_eq!(t_inline.effective_cap(), 128);

        let mut tp_spill = V3Stage4TankParams {
            delay_lengths: [53, 67, 79, 121],
            lfo_depth_samples: 0.0,
            ..V3Stage4TankParams::identity()
        };
        tp_spill.clamp_in_place();
        let t_spill = V3Stage4TankHot::from_cold(&tp_spill);
        assert!(t_spill.effective_cap() > 128);

        // Wet readout 5 Hz DC blocker removes steady DC offset on asymmetric output taps.
        let mut tp_dc = V3Stage4TankParams {
            delay_lengths: [8, 12, 16, 20],
            g_loop: 0.9,
            inject_gain: 0.5,
            dry_mix: 0.0,
            wet_mix: 1.0,
            tap_gains: [1.0, 0.0, 0.0, 0.0],
            base_psi: 0.0,
            ..V3Stage4TankParams::identity()
        };
        tp_dc.clamp_in_place();
        let mut t_dc = V3Stage4TankHot::from_cold(&tp_dc);
        let mut last = [0.0_f32; 2];
        for _ in 0..24_000 {
            last = t_dc.step_stereo::<DefaultMathBackend>(1.0, 1.0);
        }
        assert!(last[0].abs() < 1e-4 && last[1].abs() < 1e-4);

        // Circulating U(psi, v) = H(v) R(psi): non-zero psi mixes stereo pairs inside feedback.
        let mut tp_psi0 = V3Stage4TankParams {
            delay_lengths: [4, 4, 4, 4],
            g_loop: 0.85,
            inject_gain: 1.0,
            dry_mix: 0.0,
            wet_mix: 1.0,
            tap_gains: [1.0, 1.0, 1.0, 1.0],
            base_psi: 0.0,
            ..V3Stage4TankParams::identity()
        };
        tp_psi0.clamp_in_place();
        let mut tp_psi_rot = tp_psi0;
        tp_psi_rot.base_psi = core::f32::consts::FRAC_PI_4;
        let mut t0 = V3Stage4TankHot::from_cold(&tp_psi0);
        let mut t1 = V3Stage4TankHot::from_cold(&tp_psi_rot);
        let mut diff = 0.0_f32;
        for n in 0..32 {
            let x = if n == 0 { 0.8 } else { 0.0 };
            let y0 = t0.step_stereo::<DefaultMathBackend>(x, -x);
            let y1 = t1.step_stereo::<DefaultMathBackend>(x, -x);
            diff = diff.max((y0[0] - y1[0]).abs()).max((y0[1] - y1[1]).abs());
        }
        assert!(diff > 1e-2);
    }
}
