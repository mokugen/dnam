//! Stage IV `Op-T` per-quad runtime state (`Stage4QuadHotState`), compact `SonaHotSlab` quad
//! header (`Stage4QuadHot`), and cold power-of-two `ext_ring` storage (`v3_fdn/quad_state.rs`).

use alloc::{boxed::Box, vec};
use dnam_kernels::{DC_BLOCKER_5HZ_48K_F32, f32_to_fp16};

use super::lfo_interp::{
    ExtFrameRingF32, ExtFrameRingFp16, InlineFrameRing, PartitionedSchroederBank, SubblockBuffers,
    TankMutableRefs, TankStepCtx, cold_lfo_inc_and_half_alpha, step_stereo_ring,
    step_subblock_ring,
};
use super::params::{
    Stage4ExtFormat, Stage4QuadParams, V3_TANK_CAP, V3_TANK_INLINE_MAX_DELAY, V3_TANK_LINES,
    V3_TANK_MAX_EXT_CAP, V3_TANK_MAX_LFO_DEPTH, clamp_quad_params_to_cap,
};
use crate::container::ZohRescaledParams;
use crate::math::{MathBackend, algebraic_unit_rotor, fp16::fp16_to_f32, newton6_reciprocal_f64};

#[inline(always)]
fn rescale_wet_dc_5hz(zoh: &ZohRescaledParams) -> (f32, f32) {
    let pole = zoh.rescale_real_pole(DC_BLOCKER_5HZ_48K_F32);
    (pole, 0.5 * (1.0 + pole))
}

#[derive(Clone, Debug)]
pub(crate) enum Stage4ExtStorage {
    Fp16(Box<[u16]>),
    F32(Box<[f32]>),
}

#[derive(Clone, Debug)]
pub(crate) struct Stage4ExtRing {
    pub(crate) storage: Stage4ExtStorage,
    pub(crate) cap: usize,
    pub(crate) mask: usize,
    /// Full `usize` delays when `delay_lengths > u16::MAX` (e.g. `4x` bounce `48k * 4 = 192k`).
    pub(crate) ext_delays: [usize; V3_TANK_LINES],
    pub(crate) ext_freeze_off: [u32; V3_TANK_LINES],
}

impl Stage4ExtRing {
    pub(crate) fn new(
        cap: usize,
        fmt: Stage4ExtFormat,
        ext_delays: [usize; V3_TANK_LINES],
    ) -> Self {
        let len = cap * V3_TANK_LINES;
        let storage = match fmt {
            Stage4ExtFormat::Fp16 => Stage4ExtStorage::Fp16(vec![0u16; len].into_boxed_slice()),
            Stage4ExtFormat::F32 => Stage4ExtStorage::F32(vec![0.0_f32; len].into_boxed_slice()),
        };
        Self {
            storage,
            cap,
            mask: cap - 1,
            ext_delays,
            ext_freeze_off: [0; V3_TANK_LINES],
        }
    }

    pub(crate) fn matches_format(&self, fmt: Stage4ExtFormat) -> bool {
        matches!(
            (&self.storage, fmt),
            (Stage4ExtStorage::Fp16(_), Stage4ExtFormat::Fp16)
                | (Stage4ExtStorage::F32(_), Stage4ExtFormat::F32)
        )
    }

    pub(crate) fn fill_zero(&mut self) {
        match &mut self.storage {
            Stage4ExtStorage::Fp16(b) => b.fill(0),
            Stage4ExtStorage::F32(b) => b.fill(0.0),
        }
        self.ext_freeze_off = [0; V3_TANK_LINES];
    }
}

/// Compact L1D-friendly `144 B` Stage IV `Quad` runtime header embedded inside `SonaHotSlab`.
///
/// Recursive poles, feedback DC accumulators, wet readout `5 Hz` DC blockers, and phases stay in
/// `f32`; non-recursive static gains use `fp16`; inline delays `<= 120` stay in `[u16; 4]`, while
/// declared delays `> 120` or `> V3_TANK_CAP` spill cleanly to cold `ext_ring`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct Stage4QuadHotCompact {
    pub dc_x_prev: [f32; V3_TANK_LINES],
    pub dc_y_prev: [f32; V3_TANK_LINES],
    pub wet_dc_x: [f32; 2],
    pub wet_dc_y: [f32; 2],
    pub write_ptr: u32,
    pub lfo_phase_q32: u32,
    pub lfo_inc_q32: u32,
    pub current_psi: f32,
    pub cos_psi: f32,
    pub sin_psi: f32,
    pub vca_env: f32,
    pub g_loop: f32,
    pub r_dc: f32,
    pub vca_att: f32,
    pub vca_rel: f32,
    pub wet_dc_pole: f32,
    pub delay_lengths_u16: [u16; V3_TANK_LINES],
    pub freeze_off_u16: [u16; V3_TANK_LINES],
    pub tap_gains_fp16: [u16; V3_TANK_LINES],
    /// `[inject_gain, dry_mix, wet_mix, base_psi, vca_thresh, vca_kappa, vca_min_gain, lfo_depth, inv_sigma, half_alpha_cubic, lfo_cos_stereo, lfo_sin_stereo]`.
    pub nonrec_fp16: [u16; 12],
}

const _: () = assert!(core::mem::size_of::<Stage4QuadHotCompact>() == 144);

impl Stage4QuadHotCompact {
    /// Initializes compact L1D quad header from clamped `Stage4QuadParams`.
    #[must_use]
    pub fn from_params(p: &Stage4QuadParams, host_sample_rate_hz: f32) -> Self {
        let mut out = Self {
            dc_x_prev: [0.0; V3_TANK_LINES],
            dc_y_prev: [0.0; V3_TANK_LINES],
            wet_dc_x: [0.0; 2],
            wet_dc_y: [0.0; 2],
            write_ptr: 0,
            lfo_phase_q32: 0,
            lfo_inc_q32: 0,
            current_psi: p.base_psi,
            cos_psi: 1.0,
            sin_psi: 0.0,
            vca_env: 0.0,
            g_loop: p.g_loop,
            r_dc: p.r_dc,
            vca_att: p.vca_att,
            vca_rel: p.vca_rel,
            wet_dc_pole: DC_BLOCKER_5HZ_48K_F32,
            delay_lengths_u16: p.delay_lengths.map(|d| d.min(u16::MAX as usize) as u16),
            freeze_off_u16: [0; V3_TANK_LINES],
            tap_gains_fp16: p.tap_gains.map(f32_to_fp16),
            nonrec_fp16: [0; 12],
        };
        out.apply_params(p, host_sample_rate_hz);
        out.snap_psi_immediate(0.0);
        out
    }

    /// Updates runtime coefficients in place without zeroing feedback delay memory.
    pub fn apply_params(&mut self, p: &Stage4QuadParams, host_sample_rate_hz: f32) {
        let inv_sigma = newton6_reciprocal_f64(f64::from(p.sigma_knee)) as f32;
        let (cs, ss) = algebraic_unit_rotor(p.lfo_stereo_phase);
        let (inc, ha) = cold_lfo_inc_and_half_alpha(
            p.lfo_rate_hz,
            p.lfo_depth_samples,
            p.g_loop,
            f64::from(host_sample_rate_hz.max(8_000.0)),
        );
        self.lfo_inc_q32 = inc;
        self.g_loop = p.g_loop;
        self.r_dc = p.r_dc;
        self.vca_att = p.vca_att;
        self.vca_rel = p.vca_rel;
        self.delay_lengths_u16 = p.delay_lengths.map(|d| d.min(u16::MAX as usize) as u16);
        self.tap_gains_fp16 = p.tap_gains.map(f32_to_fp16);
        self.nonrec_fp16 = [
            f32_to_fp16(p.inject_gain),
            f32_to_fp16(p.dry_mix),
            f32_to_fp16(p.wet_mix),
            f32_to_fp16(p.base_psi),
            f32_to_fp16(p.vca_thresh),
            f32_to_fp16(p.vca_kappa),
            f32_to_fp16(p.vca_min_gain),
            f32_to_fp16(p.lfo_depth_samples),
            f32_to_fp16(inv_sigma),
            f32_to_fp16(ha),
            f32_to_fp16(cs),
            f32_to_fp16(ss),
        ];
    }

    /// Reconfigures ZOH-rescaled wet `5 Hz` DC blocker pole without touching audio buffers.
    pub fn reconfigure_wet_dc(&mut self, zoh: &ZohRescaledParams) {
        self.wet_dc_pole = rescale_wet_dc_5hz(zoh).0;
    }

    /// Reads causal integer delay taps for `Op-T` crossbar pre-core coupling.
    #[inline(always)]
    #[must_use]
    pub(crate) fn peek_integer_taps<B: MathBackend>(
        &self,
        inline: Option<&[[u16; V3_TANK_LINES]; V3_TANK_CAP]>,
        ext_ring: Option<&Stage4ExtRing>,
    ) -> [f32; V3_TANK_LINES] {
        let wr = self.write_ptr as usize;
        if let Some(ext) = ext_ring {
            let m = ext.mask;
            let d = ext.ext_delays;
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
        } else if let Some(frames) = inline {
            let d = self.delay_lengths_u16.map(usize::from);
            let mask = V3_TANK_CAP - 1;
            let raw = [
                frames[(wr + d[0]) & mask][0],
                frames[(wr + d[1]) & mask][1],
                frames[(wr + d[2]) & mask][2],
                frames[(wr + d[3]) & mask][3],
            ];
            B::fp16x4_to_f32x4(&raw)
        } else {
            [0.0; V3_TANK_LINES]
        }
    }

    /// Resets recursive loop/readout state to DC equilibrium (`0.0`).
    pub fn reset(&mut self) {
        self.write_ptr = 0;
        self.freeze_off_u16 = [0; V3_TANK_LINES];
        self.dc_x_prev = [0.0; V3_TANK_LINES];
        self.dc_y_prev = [0.0; V3_TANK_LINES];
        self.wet_dc_x = [0.0; 2];
        self.wet_dc_y = [0.0; 2];
        self.vca_env = 0.0;
        self.lfo_phase_q32 = 0;
        self.snap_psi_immediate(0.0);
    }

    /// Snaps the Givens rotor directly to `base_psi + psi_delta`.
    #[inline]
    pub fn snap_psi_immediate(&mut self, psi_delta: f32) {
        let base_psi = fp16_to_f32(self.nonrec_fp16[3]);
        self.current_psi =
            (base_psi + psi_delta).clamp(-core::f32::consts::PI, core::f32::consts::PI);
        let (c, s) = algebraic_unit_rotor(self.current_psi);
        self.cos_psi = c;
        self.sin_psi = s;
    }

    /// Slews the Givens rotor at `3 kHz` (`|Delta psi| <= 0.25 rad/tick`).
    #[inline(always)]
    pub fn refresh_3khz(&mut self, psi_delta: f32) {
        let base_psi = fp16_to_f32(self.nonrec_fp16[3]);
        let target = (base_psi + psi_delta).clamp(-core::f32::consts::PI, core::f32::consts::PI);
        let diff = target - self.current_psi;
        if diff.abs() > 1.0e-7 {
            self.current_psi += diff.clamp(-0.25, 0.25);
            let (c, s) = algebraic_unit_rotor(self.current_psi);
            self.cos_psi = c;
            self.sin_psi = s;
        }
    }

    #[inline(always)]
    fn build_step_ctx(
        &self,
        delays: [usize; V3_TANK_LINES],
        cap: usize,
        freeze: bool,
        delay_mod_samples: f32,
    ) -> TankStepCtx {
        let cap_s = cap.max(8);
        let lfo_depth = fp16_to_f32(self.nonrec_fp16[7]);
        let max_tau = if lfo_depth > 0.0 || delay_mod_samples.abs() > 1.0e-6 {
            (cap_s - 3) as f32
        } else {
            (cap_s - 2) as f32
        };
        TankStepCtx {
            params: Stage4QuadParams {
                delay_lengths: delays,
                tap_gains: self.tap_gains_fp16.map(fp16_to_f32),
                allpass_delays: [0; V3_TANK_LINES],
                allpass_g: [0.0; V3_TANK_LINES],
                g_loop: self.g_loop,
                sigma_knee: 1.0,
                base_psi: fp16_to_f32(self.nonrec_fp16[3]),
                inject_gain: fp16_to_f32(self.nonrec_fp16[0]),
                dry_mix: fp16_to_f32(self.nonrec_fp16[1]),
                wet_mix: fp16_to_f32(self.nonrec_fp16[2]),
                r_dc: self.r_dc,
                vca_thresh: fp16_to_f32(self.nonrec_fp16[4]),
                vca_kappa: fp16_to_f32(self.nonrec_fp16[5]),
                vca_min_gain: fp16_to_f32(self.nonrec_fp16[6]),
                vca_att: self.vca_att,
                vca_rel: self.vca_rel,
                lfo_rate_hz: 0.0,
                lfo_depth_samples: lfo_depth,
                lfo_stereo_phase: 0.0,
                ext_format: Stage4ExtFormat::Fp16,
            },
            inv_sigma: fp16_to_f32(self.nonrec_fp16[8]),
            cos_psi: self.cos_psi,
            sin_psi: self.sin_psi,
            cos_st: fp16_to_f32(self.nonrec_fp16[10]),
            sin_st: fp16_to_f32(self.nonrec_fp16[11]),
            half_alpha_cubic: fp16_to_f32(self.nonrec_fp16[9]),
            delay_mod_samples,
            lfo_inc: self.lfo_inc_q32,
            max_tau,
            freeze,
            wet_dc_pole: self.wet_dc_pole,
            wet_dc_alpha: 0.5 * (1.0 + self.wet_dc_pole),
        }
    }

    /// Steps compact `Quad` for one stereo sample (`0` allocations).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_stereo<B: MathBackend>(
        &mut self,
        inline: Option<&mut [[u16; V3_TANK_LINES]; V3_TANK_CAP]>,
        ext_ring: Option<&mut Stage4ExtRing>,
        schroeder: Option<&mut PartitionedSchroederBank>,
        freeze: bool,
        delay_mod_samples: f32,
        in_l: f32,
        in_r: f32,
    ) -> [f32; 2] {
        if let Some(ext) = ext_ring {
            let ctx = self.build_step_ctx(ext.ext_delays, ext.cap, freeze, delay_mod_samples);
            let mut st = TankMutableRefs {
                write_ptr: &mut self.write_ptr,
                freeze_off: &mut ext.ext_freeze_off,
                dc_x_prev: &mut self.dc_x_prev,
                dc_y_prev: &mut self.dc_y_prev,
                wet_dc_x: &mut self.wet_dc_x,
                wet_dc_y: &mut self.wet_dc_y,
                lfo_phase_q32: &mut self.lfo_phase_q32,
                vca_env: &mut self.vca_env,
                schroeder,
            };
            let m = ext.mask;
            match &mut ext.storage {
                Stage4ExtStorage::Fp16(buf) => {
                    let mut ring = ExtFrameRingFp16 { buf, mask: m };
                    step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r)
                }
                Stage4ExtStorage::F32(buf) => {
                    let mut ring = ExtFrameRingF32 { buf, mask: m };
                    step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r)
                }
            }
        } else if let Some(frames) = inline {
            let delays = self.delay_lengths_u16.map(usize::from);
            let ctx = self.build_step_ctx(delays, V3_TANK_CAP, freeze, delay_mod_samples);
            let mut freeze_u32 = self.freeze_off_u16.map(u32::from);
            let mut st = TankMutableRefs {
                write_ptr: &mut self.write_ptr,
                freeze_off: &mut freeze_u32,
                dc_x_prev: &mut self.dc_x_prev,
                dc_y_prev: &mut self.dc_y_prev,
                wet_dc_x: &mut self.wet_dc_x,
                wet_dc_y: &mut self.wet_dc_y,
                lfo_phase_q32: &mut self.lfo_phase_q32,
                vca_env: &mut self.vca_env,
                schroeder,
            };
            let mut ring = InlineFrameRing(frames);
            let out = step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r);
            self.freeze_off_u16 = freeze_u32.map(|v| v as u16);
            out
        } else {
            [0.0, 0.0]
        }
    }

    /// Steps compact `Quad` across a `<= 16`-sample sub-block using inline L1D or cold `ext_ring`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_subblock<B: MathBackend, const MONO: bool, const FREEZE: bool>(
        &mut self,
        inline: Option<&mut [[u16; V3_TANK_LINES]; V3_TANK_CAP]>,
        ext_ring: Option<&mut Stage4ExtRing>,
        schroeder: Option<&mut PartitionedSchroederBank>,
        delay_mod_samples: f32,
        in_l: &[f32; 16],
        in_r: &[f32; 16],
        out_l: &mut [f32],
        out_r: &mut [f32],
        len: usize,
        accumulate_output: bool,
    ) {
        if let Some(ext) = ext_ring {
            let ctx = self.build_step_ctx(ext.ext_delays, ext.cap, FREEZE, delay_mod_samples);
            let mut st = TankMutableRefs {
                write_ptr: &mut self.write_ptr,
                freeze_off: &mut ext.ext_freeze_off,
                dc_x_prev: &mut self.dc_x_prev,
                dc_y_prev: &mut self.dc_y_prev,
                wet_dc_x: &mut self.wet_dc_x,
                wet_dc_y: &mut self.wet_dc_y,
                lfo_phase_q32: &mut self.lfo_phase_q32,
                vca_env: &mut self.vca_env,
                schroeder,
            };
            let m = ext.mask;
            let io = SubblockBuffers {
                in_l,
                in_r,
                out_l,
                out_r,
                len,
                accumulate_output,
            };
            match &mut ext.storage {
                Stage4ExtStorage::Fp16(buf) => {
                    let mut ring = ExtFrameRingFp16 { buf, mask: m };
                    step_subblock_ring::<B, MONO, FREEZE, _>(&mut ring, ctx, &mut st, io);
                }
                Stage4ExtStorage::F32(buf) => {
                    let mut ring = ExtFrameRingF32 { buf, mask: m };
                    step_subblock_ring::<B, MONO, FREEZE, _>(&mut ring, ctx, &mut st, io);
                }
            }
        } else if let Some(frames) = inline {
            let delays = self.delay_lengths_u16.map(usize::from);
            let ctx = self.build_step_ctx(delays, V3_TANK_CAP, FREEZE, delay_mod_samples);
            let mut freeze_u32 = self.freeze_off_u16.map(u32::from);
            let mut st = TankMutableRefs {
                write_ptr: &mut self.write_ptr,
                freeze_off: &mut freeze_u32,
                dc_x_prev: &mut self.dc_x_prev,
                dc_y_prev: &mut self.dc_y_prev,
                wet_dc_x: &mut self.wet_dc_x,
                wet_dc_y: &mut self.wet_dc_y,
                lfo_phase_q32: &mut self.lfo_phase_q32,
                vca_env: &mut self.vca_env,
                schroeder,
            };
            let mut ring = InlineFrameRing(frames);
            step_subblock_ring::<B, MONO, FREEZE, _>(
                &mut ring,
                ctx,
                &mut st,
                SubblockBuffers {
                    in_l,
                    in_r,
                    out_l,
                    out_r,
                    len,
                    accumulate_output,
                },
            );
            self.freeze_off_u16 = freeze_u32.map(|v| v as u16);
        }
    }
}

/// Cold + inline runtime state for one `Op-T` tank quad (`Stage4QuadParams` + cold overflow rings).
#[repr(C, align(16))]
#[derive(Clone, Debug)]
pub struct Stage4QuadHotState {
    pub params: Stage4QuadParams,
    pub write_ptr: u32,
    pub freeze_off: [u32; V3_TANK_LINES],
    pub inv_sigma: f32,
    pub current_psi: f32,
    pub cos_psi: f32,
    pub sin_psi: f32,
    pub vca_env: f32,
    pub dc_x_prev: [f32; V3_TANK_LINES],
    pub dc_y_prev: [f32; V3_TANK_LINES],
    pub wet_dc_x: [f32; 2],
    pub wet_dc_y: [f32; 2],
    pub wet_dc_pole: f32,
    pub wet_dc_alpha: f32,
    pub lfo_phase_q32: u32,
    pub lfo_inc_q32: u32,
    pub lfo_cos_stereo: f32,
    pub lfo_sin_stereo: f32,
    pub half_alpha_cubic: f32,
    pub host_sample_rate_hz: f32,
    pub(crate) ext_active: bool,
    pub(crate) ext_ring: Option<Box<Stage4ExtRing>>,
    pub(crate) schroeder: Option<Box<PartitionedSchroederBank>>,
}

impl Stage4QuadHotState {
    pub(crate) fn from_params(
        params: &Stage4QuadParams,
        min_frames_cap: usize,
        allow_inline_128: bool,
    ) -> Self {
        let mut p = *params;
        p.clamp_in_place();
        let max_d = p.delay_lengths.iter().copied().max().unwrap_or(67);
        let lfo_margin = (p.lfo_depth_samples as usize) + 6;
        let needed_1x = max_d.saturating_add(lfo_margin);
        let need_ext = !allow_inline_128
            || p.ext_format == Stage4ExtFormat::F32
            || max_d > V3_TANK_INLINE_MAX_DELAY
            || needed_1x > V3_TANK_CAP
            || min_frames_cap > V3_TANK_CAP;
        let prealloc_cap = needed_1x
            .saturating_mul(8)
            .max(min_frames_cap)
            .max(V3_TANK_CAP * 8)
            .next_power_of_two()
            .min(V3_TANK_MAX_EXT_CAP);
        let eff_target_cap = if !need_ext { V3_TANK_CAP } else { prealloc_cap };
        clamp_quad_params_to_cap(&mut p, eff_target_cap, V3_TANK_MAX_LFO_DEPTH);
        let ext_ring = if !need_ext && !allow_inline_128 {
            None
        } else {
            Some(Box::new(Stage4ExtRing::new(
                prealloc_cap,
                p.ext_format,
                p.delay_lengths,
            )))
        };

        let inv_sigma = newton6_reciprocal_f64(f64::from(p.sigma_knee)) as f32;
        let (cos_psi, sin_psi) = algebraic_unit_rotor(p.base_psi);
        let (lfo_cos_stereo, lfo_sin_stereo) = algebraic_unit_rotor(p.lfo_stereo_phase);
        let (lfo_inc_q32, half_alpha_cubic) =
            cold_lfo_inc_and_half_alpha(p.lfo_rate_hz, p.lfo_depth_samples, p.g_loop, 48_000.0);
        let schroeder =
            PartitionedSchroederBank::from_params(p.allpass_delays, p.allpass_g).map(Box::new);
        Self {
            params: p,
            write_ptr: 0,
            freeze_off: [0; V3_TANK_LINES],
            inv_sigma,
            current_psi: p.base_psi,
            cos_psi,
            sin_psi,
            vca_env: 0.0,
            dc_x_prev: [0.0; V3_TANK_LINES],
            dc_y_prev: [0.0; V3_TANK_LINES],
            wet_dc_x: [0.0; 2],
            wet_dc_y: [0.0; 2],
            wet_dc_pole: DC_BLOCKER_5HZ_48K_F32,
            wet_dc_alpha: 0.5 * (1.0 + DC_BLOCKER_5HZ_48K_F32),
            lfo_phase_q32: 0,
            lfo_inc_q32,
            lfo_cos_stereo,
            lfo_sin_stereo,
            half_alpha_cubic,
            host_sample_rate_hz: 48_000.0,
            ext_active: need_ext,
            ext_ring,
            schroeder,
        }
    }

    pub(crate) fn effective_cap(&self, allow_inline_128: bool) -> usize {
        if !self.ext_active && allow_inline_128 {
            return V3_TANK_CAP;
        }
        self.ext_ring
            .as_deref()
            .map_or(if allow_inline_128 { V3_TANK_CAP } else { 0 }, |ext| {
                ext.cap
            })
    }

    pub(crate) fn apply_runtime_params(&mut self, p: Stage4QuadParams) {
        self.inv_sigma = newton6_reciprocal_f64(f64::from(p.sigma_knee)) as f32;
        let (cs, ss) = algebraic_unit_rotor(p.lfo_stereo_phase);
        self.lfo_cos_stereo = cs;
        self.lfo_sin_stereo = ss;
        let (inc, ha) = cold_lfo_inc_and_half_alpha(
            p.lfo_rate_hz,
            p.lfo_depth_samples,
            p.g_loop,
            f64::from(self.host_sample_rate_hz.max(8_000.0)),
        );
        self.lfo_inc_q32 = inc;
        self.half_alpha_cubic = ha;
        if let Some(ap) = self.schroeder.as_deref_mut() {
            ap.apply_params(p.allpass_delays, p.allpass_g);
        }
        if self.ext_active
            && let Some(ext) = self.ext_ring.as_deref_mut()
        {
            ext.ext_delays = p.delay_lengths;
        }
        self.params = p;
    }

    pub(crate) fn ensure_cold_ext_cap(
        &mut self,
        params: &Stage4QuadParams,
        min_frames_cap: usize,
        allow_inline_128: bool,
    ) {
        let mut p = *params;
        p.clamp_in_place();
        let max_d = p.delay_lengths.iter().copied().max().unwrap_or(67);
        let lfo_margin = (p.lfo_depth_samples as usize) + 6;
        let needed_1x = max_d.saturating_add(lfo_margin);
        let max_ap = p.allpass_delays.iter().copied().max().unwrap_or(0);
        if max_ap > 0 {
            if let Some(ap) = self.schroeder.as_deref_mut() {
                ap.ensure_cap(p.allpass_delays, p.allpass_g);
            } else {
                self.schroeder =
                    PartitionedSchroederBank::from_params(p.allpass_delays, p.allpass_g)
                        .map(Box::new);
            }
        }
        let need_ext = !allow_inline_128
            || p.ext_format == Stage4ExtFormat::F32
            || max_d > V3_TANK_INLINE_MAX_DELAY
            || needed_1x > V3_TANK_CAP
            || min_frames_cap > V3_TANK_CAP;
        self.ext_active = need_ext;
        if !need_ext {
            return;
        }
        let req = needed_1x
            .saturating_mul(8)
            .max(min_frames_cap)
            .max(V3_TANK_CAP);
        let target_cap = req.next_power_of_two().min(V3_TANK_MAX_EXT_CAP);
        if let Some(ext) = self.ext_ring.as_deref_mut()
            && ext.cap >= target_cap
            && ext.matches_format(p.ext_format)
        {
            clamp_quad_params_to_cap(&mut p, ext.cap, V3_TANK_MAX_LFO_DEPTH);
            self.apply_runtime_params(p);
            return;
        }
        clamp_quad_params_to_cap(&mut p, target_cap, V3_TANK_MAX_LFO_DEPTH);
        self.ext_ring = Some(Box::new(Stage4ExtRing::new(
            target_cap,
            p.ext_format,
            p.delay_lengths,
        )));
        self.apply_runtime_params(p);
    }

    pub(crate) fn reconfigure_wet_dc(&mut self, zoh: &ZohRescaledParams) {
        let (pole, alpha) = rescale_wet_dc_5hz(zoh);
        self.wet_dc_pole = pole;
        self.wet_dc_alpha = alpha;
    }

    pub(crate) fn reset(&mut self) {
        if let Some(ext) = self.ext_ring.as_deref_mut() {
            ext.fill_zero();
        }
        if let Some(ap) = self.schroeder.as_deref_mut() {
            ap.reset();
        }
        self.write_ptr = 0;
        self.freeze_off = [0; V3_TANK_LINES];
        self.dc_x_prev = [0.0; V3_TANK_LINES];
        self.dc_y_prev = [0.0; V3_TANK_LINES];
        self.wet_dc_x = [0.0; 2];
        self.wet_dc_y = [0.0; 2];
        self.vca_env = 0.0;
        self.lfo_phase_q32 = 0;
        self.snap_psi_immediate(0.0);
    }

    pub(crate) fn snap_psi_immediate(&mut self, psi_delta: f32) {
        self.current_psi =
            (self.params.base_psi + psi_delta).clamp(-core::f32::consts::PI, core::f32::consts::PI);
        let (c, s) = algebraic_unit_rotor(self.current_psi);
        self.cos_psi = c;
        self.sin_psi = s;
    }

    pub(crate) fn refresh_3khz(&mut self, psi_delta: f32) {
        let target =
            (self.params.base_psi + psi_delta).clamp(-core::f32::consts::PI, core::f32::consts::PI);
        let diff = target - self.current_psi;
        if diff.abs() > 1.0e-7 {
            self.current_psi += diff.clamp(-0.25, 0.25);
            let (c, s) = algebraic_unit_rotor(self.current_psi);
            self.cos_psi = c;
            self.sin_psi = s;
        }
    }

    fn build_ctx(&self, cap: usize, freeze: bool) -> TankStepCtx {
        let cap_s = cap.max(8);
        let max_tau = if self.params.lfo_depth_samples > 0.0 {
            (cap_s - 3) as f32
        } else {
            (cap_s - 2) as f32
        };
        TankStepCtx {
            params: self.params,
            inv_sigma: self.inv_sigma,
            cos_psi: self.cos_psi,
            sin_psi: self.sin_psi,
            cos_st: self.lfo_cos_stereo,
            sin_st: self.lfo_sin_stereo,
            half_alpha_cubic: self.half_alpha_cubic,
            delay_mod_samples: 0.0,
            lfo_inc: self.lfo_inc_q32,
            max_tau,
            freeze,
            wet_dc_pole: self.wet_dc_pole,
            wet_dc_alpha: self.wet_dc_alpha,
        }
    }

    #[inline(always)]
    pub(crate) fn step_stereo_with_opt_inline<B: MathBackend>(
        &mut self,
        inline: Option<&mut [[u16; V3_TANK_LINES]; V3_TANK_CAP]>,
        freeze: bool,
        in_l: f32,
        in_r: f32,
    ) -> [f32; 2] {
        let allow_inline = inline.is_some();
        let ctx = self.build_ctx(self.effective_cap(allow_inline), freeze);
        let mut st = TankMutableRefs {
            write_ptr: &mut self.write_ptr,
            freeze_off: &mut self.freeze_off,
            dc_x_prev: &mut self.dc_x_prev,
            dc_y_prev: &mut self.dc_y_prev,
            wet_dc_x: &mut self.wet_dc_x,
            wet_dc_y: &mut self.wet_dc_y,
            lfo_phase_q32: &mut self.lfo_phase_q32,
            vca_env: &mut self.vca_env,
            schroeder: self.schroeder.as_deref_mut(),
        };
        if self.ext_active
            && let Some(ext) = self.ext_ring.as_deref_mut()
        {
            let m = ext.mask;
            match &mut ext.storage {
                Stage4ExtStorage::Fp16(buf) => {
                    let mut ring = ExtFrameRingFp16 { buf, mask: m };
                    step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r)
                }
                Stage4ExtStorage::F32(buf) => {
                    let mut ring = ExtFrameRingF32 { buf, mask: m };
                    step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r)
                }
            }
        } else if let Some(frames) = inline {
            let mut ring = InlineFrameRing(frames);
            step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r)
        } else {
            [0.0, 0.0]
        }
    }

    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn step_subblock_with_opt_inline<
        B: MathBackend,
        const MONO: bool,
        const FREEZE: bool,
    >(
        &mut self,
        inline: Option<&mut [[u16; V3_TANK_LINES]; V3_TANK_CAP]>,
        io: SubblockBuffers<'_>,
    ) {
        let allow_inline = inline.is_some();
        let ctx = self.build_ctx(self.effective_cap(allow_inline), FREEZE);
        let mut st = TankMutableRefs {
            write_ptr: &mut self.write_ptr,
            freeze_off: &mut self.freeze_off,
            dc_x_prev: &mut self.dc_x_prev,
            dc_y_prev: &mut self.dc_y_prev,
            wet_dc_x: &mut self.wet_dc_x,
            wet_dc_y: &mut self.wet_dc_y,
            lfo_phase_q32: &mut self.lfo_phase_q32,
            vca_env: &mut self.vca_env,
            schroeder: self.schroeder.as_deref_mut(),
        };
        if self.ext_active
            && let Some(ext) = self.ext_ring.as_deref_mut()
        {
            let m = ext.mask;
            match &mut ext.storage {
                Stage4ExtStorage::Fp16(buf) => {
                    let mut ring = ExtFrameRingFp16 { buf, mask: m };
                    step_subblock_ring::<B, MONO, FREEZE, _>(&mut ring, ctx, &mut st, io);
                }
                Stage4ExtStorage::F32(buf) => {
                    let mut ring = ExtFrameRingF32 { buf, mask: m };
                    step_subblock_ring::<B, MONO, FREEZE, _>(&mut ring, ctx, &mut st, io);
                }
            }
        } else if let Some(frames) = inline {
            let mut ring = InlineFrameRing(frames);
            step_subblock_ring::<B, MONO, FREEZE, _>(&mut ring, ctx, &mut st, io);
        }
    }
}
