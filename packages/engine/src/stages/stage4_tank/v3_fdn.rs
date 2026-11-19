//! Unconditional 4-line `fp16` Stage IV spatial & LFO-modulated FDN tank (`stage4_tank/v3_fdn.rs`).
//!
//! Executes every frame with zero branches inside the inner sample loop, zero `FDIV`, zero `FSQRT`,
//! zero `libm` calls, autonomous `Q0.32` quadrature LFO + convex-blended 4-tap Catmull-Rom
//! fractional reads (`0-ULP` resting endpoint identity), in-register `SimdVec4::rcp_pos_nr2`
//! coercive loop saturation, Householder + Givens feedback mix, `5 Hz` SIMD DC blocking,
//! `±65,504.0` `fp16` storage clamping (`REQ-7.1`), unitary hardware freeze latch
//! (`ref[dsp.stage4.captured-freeze]`), and stereo-linked `BallisticDuckingVca`.

mod lfo_interp;

use alloc::{boxed::Box, vec};

use crate::container::ZohRescaledParams;
use crate::math::{
    MathBackend, STAGE4_DC_BLOCKER_48K_F32, algebraic_unit_rotor, newton6_reciprocal_f64,
};
use crate::stages::topology::STAGE4_MAX_G_LIN_BITS;
use lfo_interp::{
    ExtTankRing, InlineTankRing, SubblockBuffers, TankMutableRefs, TankStepCtx, step_stereo_ring,
    step_subblock_ring,
};

/// Number of interleaved `fp16` FDN delay lines in Stage IV (`4` lines).
pub const V3_TANK_LINES: usize = 4;

/// Circular ring buffer capacity per Stage IV inline L1D line (`1024` quad frames = `8 KiB` `fp16`).
pub const V3_TANK_CAP: usize = 1024;
/// Line-major stride including 4 mirrored tail samples (`1028 * 4 * 2 = 8,224 B`) for wrap-free 64-bit quad loads.
pub const V3_TANK_RING_STRIDE: usize = V3_TANK_CAP + 4;
const V3_TANK_MASK: usize = V3_TANK_CAP - 1;

/// Maximum model-declared Stage IV delay length in `48 kHz` reference samples (`1.0 s`).
pub const V3_TANK_MAX_DECLARED_DELAY: usize = 48_000;
/// Maximum Stage IV LFO excursion depth in host samples.
pub const V3_TANK_MAX_LFO_DEPTH: f32 = 512.0;
/// Maximum cold-allocated power-of-two ring capacity (accommodates `4x` bounce on `48,000`-sample delays).
pub const V3_TANK_MAX_EXT_CAP: usize = 262_144;
const _: () = assert!(V3_TANK_CAP.is_power_of_two() && V3_TANK_MAX_EXT_CAP.is_power_of_two());
const _: () = assert!(V3_TANK_MAX_EXT_CAP >= 4 * V3_TANK_MAX_DECLARED_DELAY + 2_048);

/// Cold parameters for the unconditional 4-line `fp16` Stage IV spatial FDN tank.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3Stage4TankParams {
    pub delay_lengths: [usize; V3_TANK_LINES],
    pub g_loop: f32,
    pub sigma_knee: f32,
    pub base_psi: f32,
    pub inject_gain: f32,
    pub dry_mix: f32,
    pub wet_mix: f32,
    pub r_dc: f32,
    pub vca_thresh: f32,
    pub vca_kappa: f32,
    pub vca_min_gain: f32,
    pub vca_att: f32,
    pub vca_rel: f32,
    pub tap_gains: [f32; V3_TANK_LINES],
    pub lfo_rate_hz: f32,
    pub lfo_depth_samples: f32,
    pub lfo_stereo_phase: f32,
}

impl Default for V3Stage4TankParams {
    fn default() -> Self {
        Self::identity()
    }
}

impl V3Stage4TankParams {
    /// Resting identity Stage IV tank (`dry_mix = 1.0`, `wet_mix = 0.0`), executes FDN every sample.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            delay_lengths: [67, 83, 97, 113],
            g_loop: 0.72,
            sigma_knee: 1.0,
            base_psi: core::f32::consts::FRAC_PI_4,
            inject_gain: 0.5,
            dry_mix: 1.0,
            wet_mix: 0.0,
            r_dc: STAGE4_DC_BLOCKER_48K_F32,
            vca_thresh: 0.1,
            vca_kappa: 1.5,
            vca_min_gain: 0.15,
            vca_att: 0.99,
            vca_rel: 0.9995,
            tap_gains: [0.5; V3_TANK_LINES],
            lfo_rate_hz: 0.0,
            lfo_depth_samples: 0.0,
            lfo_stereo_phase: 0.0,
        }
    }

    /// Returns default prime FDN delay lengths for `tier`.
    #[must_use]
    pub const fn default_delays_for_tier(tier: crate::types::Tier) -> [usize; V3_TANK_LINES] {
        match tier {
            crate::types::Tier::Lite => [53, 67, 79, 89],
            crate::types::Tier::Full => [149, 179, 211, 239],
            crate::types::Tier::Pro => [263, 293, 331, 367],
            crate::types::Tier::Ultra => [389, 431, 479, 521],
        }
    }

    /// Active delay window bytes (`delay_max * 8`) touched per cycle in L1D.
    #[must_use]
    pub const fn active_ring_bytes(tier: crate::types::Tier) -> usize {
        let d = Self::default_delays_for_tier(tier);
        let mut max_d = d[0];
        if d[1] > max_d {
            max_d = d[1];
        }
        if d[2] > max_d {
            max_d = d[2];
        }
        if d[3] > max_d {
            max_d = d[3];
        }
        (max_d + 1) * V3_TANK_LINES * 2
    }

    fn clamp_common_fields(&mut self) {
        let max_g = f32::from_bits(STAGE4_MAX_G_LIN_BITS);
        self.g_loop = self.g_loop.clamp(0.0, max_g);
        self.sigma_knee = self.sigma_knee.clamp(1.0e-3, 8.0);
        self.base_psi = self
            .base_psi
            .clamp(-core::f32::consts::PI, core::f32::consts::PI);
        self.inject_gain = self.inject_gain.clamp(-2.0, 2.0);
        self.dry_mix = self.dry_mix.clamp(-2.0, 2.0);
        self.wet_mix = self.wet_mix.clamp(-2.0, 2.0);
        for g in &mut self.tap_gains {
            *g = if g.is_finite() {
                g.clamp(-2.0, 2.0)
            } else {
                0.5
            };
        }
        self.r_dc = self.r_dc.clamp(0.95, 0.9999);
        self.vca_thresh = self.vca_thresh.clamp(0.0, 2.0);
        self.vca_kappa = self.vca_kappa.clamp(0.0, 8.0);
        self.vca_min_gain = self.vca_min_gain.clamp(0.05, 1.0);
        self.vca_att = self.vca_att.clamp(0.5, 0.9999);
        self.vca_rel = self.vca_rel.clamp(0.5, 0.99999);
        self.lfo_rate_hz = if self.lfo_rate_hz.is_finite() {
            self.lfo_rate_hz.clamp(0.0, 16.0)
        } else {
            0.0
        };
        self.lfo_stereo_phase = if self.lfo_stereo_phase.is_finite() {
            self.lfo_stereo_phase
                .clamp(-core::f32::consts::PI, core::f32::consts::PI)
        } else {
            0.0
        };
    }

    /// Clamps tank loop gain `< 0.9990`, model-declared delay bounds (`4..=48,000`), and VCA coefficients.
    pub fn clamp_in_place(&mut self) {
        self.clamp_common_fields();
        for d in &mut self.delay_lengths {
            *d = (*d).clamp(4, V3_TANK_MAX_DECLARED_DELAY);
        }
        self.lfo_depth_samples = if self.lfo_depth_samples.is_finite() {
            self.lfo_depth_samples.clamp(0.0, V3_TANK_MAX_LFO_DEPTH)
        } else {
            0.0
        };
    }
}

#[derive(Clone, Debug)]
struct V3Stage4ExtRing {
    data: Box<[u16]>,
    cap: usize,
    stride: usize,
    mask: usize,
}

/// Hot line-major mirrored `fp16` 4-line FDN tank with unconditional branch-free `SimdVec4` saturation.
#[repr(C, align(16))]
#[derive(Clone, Debug)]
pub struct V3Stage4TankHot {
    pub lines: [[u16; V3_TANK_RING_STRIDE]; V3_TANK_LINES],
    pub params: V3Stage4TankParams,
    pub write_ptr: u32,
    pub inv_sigma: f32,
    pub current_psi: f32,
    pub cos_psi: f32,
    pub sin_psi: f32,
    pub vca_env: f32,
    pub dc_x_prev: [f32; V3_TANK_LINES],
    pub dc_y_prev: [f32; V3_TANK_LINES],
    pub lfo_phase_q32: u32,
    pub lfo_inc_q32: u32,
    pub lfo_cos_stereo: f32,
    pub lfo_sin_stereo: f32,
    pub half_alpha_cubic: f32,
    pub host_sample_rate_hz: f32,
    ext_ring: Option<Box<V3Stage4ExtRing>>,
    /// Unitary captured hardware freeze latch (`ref[dsp.stage4.captured-freeze]`).
    pub freeze: bool,
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<V3Stage4TankHot>() == 8_432);

fn clamp_params_to_cap(p: &mut V3Stage4TankParams, cap: usize, max_lfo_depth: f32) {
    let max_delay = cap.saturating_sub(1).max(4);
    for d in &mut p.delay_lengths {
        *d = (*d).clamp(4, max_delay);
    }
    let cap_depth = ((cap / 4) as f32).min(max_lfo_depth);
    p.lfo_depth_samples = if p.lfo_depth_samples.is_finite() {
        p.lfo_depth_samples.clamp(0.0, cap_depth)
    } else {
        0.0
    };
}

impl V3Stage4TankHot {
    /// Returns the active ring capacity (`V3_TANK_CAP` for inline L1D or cold-allocated `ext_cap`).
    #[inline(always)]
    #[must_use]
    pub fn effective_cap(&self) -> usize {
        self.ext_ring.as_deref().map_or(V3_TANK_CAP, |ext| ext.cap)
    }

    /// Initializes Stage IV FDN tank hot state from cold parameters, pre-allocating a
    /// power-of-two extended ring (`ext_lines`) strictly on cold init when required capacity
    /// exceeds the inline `1,024`-frame L1D buffer (`0` allocations on RT / switch / ZOH paths).
    #[must_use]
    pub fn from_cold_with_min_cap(params: &V3Stage4TankParams, min_frames_cap: usize) -> Self {
        let mut p = *params;
        p.clamp_in_place();
        let max_d = p.delay_lengths.iter().copied().max().unwrap_or(67);
        let lfo_margin = (p.lfo_depth_samples as usize) + 6;
        let needed_1x = max_d.saturating_add(lfo_margin);
        let ext_ring = if needed_1x <= V3_TANK_CAP && min_frames_cap <= V3_TANK_CAP {
            None
        } else {
            let req = needed_1x
                .saturating_mul(4)
                .max(min_frames_cap)
                .max(V3_TANK_CAP * 2);
            let cap = req.next_power_of_two().min(V3_TANK_MAX_EXT_CAP);
            let stride = cap + 4;
            let mask = cap - 1;
            Some(Box::new(V3Stage4ExtRing {
                data: vec![0u16; stride * V3_TANK_LINES].into_boxed_slice(),
                cap,
                stride,
                mask,
            }))
        };
        let eff_cap = ext_ring.as_deref().map_or(V3_TANK_CAP, |ext| ext.cap);
        clamp_params_to_cap(&mut p, eff_cap, V3_TANK_MAX_LFO_DEPTH);

        let inv_sigma = newton6_reciprocal_f64(f64::from(p.sigma_knee)) as f32;
        let (cos_psi, sin_psi) = algebraic_unit_rotor(p.base_psi);
        let (lfo_cos_stereo, lfo_sin_stereo) = algebraic_unit_rotor(p.lfo_stereo_phase);
        let (lfo_inc_q32, half_alpha_cubic) = lfo_interp::cold_lfo_inc_and_half_alpha(
            p.lfo_rate_hz,
            p.lfo_depth_samples,
            p.g_loop,
            48_000.0,
        );
        Self {
            lines: [[0; V3_TANK_RING_STRIDE]; V3_TANK_LINES],
            params: p,
            write_ptr: 0,
            inv_sigma,
            current_psi: p.base_psi,
            cos_psi,
            sin_psi,
            vca_env: 0.0,
            dc_x_prev: [0.0; V3_TANK_LINES],
            dc_y_prev: [0.0; V3_TANK_LINES],
            lfo_phase_q32: 0,
            lfo_inc_q32,
            lfo_cos_stereo,
            lfo_sin_stereo,
            half_alpha_cubic,
            host_sample_rate_hz: 48_000.0,
            ext_ring,
            freeze: false,
        }
    }

    /// Compiles Stage IV FDN tank hot state from cold parameters.
    #[must_use]
    pub fn from_cold(params: &V3Stage4TankParams) -> Self {
        Self::from_cold_with_min_cap(params, 0)
    }

    fn apply_runtime_params(&mut self, p: V3Stage4TankParams) {
        self.inv_sigma = newton6_reciprocal_f64(f64::from(p.sigma_knee)) as f32;
        let (cs, ss) = algebraic_unit_rotor(p.lfo_stereo_phase);
        self.lfo_cos_stereo = cs;
        self.lfo_sin_stereo = ss;
        let (inc, ha) = lfo_interp::cold_lfo_inc_and_half_alpha(
            p.lfo_rate_hz,
            p.lfo_depth_samples,
            p.g_loop,
            f64::from(self.host_sample_rate_hz.max(8_000.0)),
        );
        self.lfo_inc_q32 = inc;
        self.half_alpha_cubic = ha;
        self.params = p;
    }

    /// Ensures Stage IV's cold extended delay ring accommodates `params` and `min_frames_cap`
    /// without heap allocations on real-time, tier-switch, ZOH, or in-place reload paths.
    pub fn ensure_cold_ext_cap(&mut self, params: &V3Stage4TankParams, min_frames_cap: usize) {
        let mut p = *params;
        p.clamp_in_place();
        let max_d = p.delay_lengths.iter().copied().max().unwrap_or(67);
        let lfo_margin = (p.lfo_depth_samples as usize) + 6;
        let needed_1x = max_d.saturating_add(lfo_margin);
        if needed_1x <= V3_TANK_CAP && min_frames_cap <= V3_TANK_CAP {
            return;
        }
        let req = needed_1x
            .saturating_mul(4)
            .max(min_frames_cap)
            .max(V3_TANK_CAP * 2);
        let target_cap = req.next_power_of_two().min(V3_TANK_MAX_EXT_CAP);
        if self.effective_cap() >= target_cap {
            clamp_params_to_cap(&mut p, self.effective_cap(), V3_TANK_MAX_LFO_DEPTH);
            self.apply_runtime_params(p);
            return;
        }
        let stride = target_cap + 4;
        let mask = target_cap - 1;
        let new_ring = Box::new(V3Stage4ExtRing {
            data: vec![0u16; stride * V3_TANK_LINES].into_boxed_slice(),
            cap: target_cap,
            stride,
            mask,
        });
        if let Some(ext) = self.ext_ring.as_mut() {
            *ext = new_ring;
        } else {
            self.ext_ring = Some(new_ring);
        }
        clamp_params_to_cap(&mut p, target_cap, V3_TANK_MAX_LFO_DEPTH);
        self.apply_runtime_params(p);
    }

    /// Updates Stage IV tank parameters in place without clearing delay memory (`0` heap allocs).
    pub fn update_params(&mut self, params: &V3Stage4TankParams) {
        let mut p = *params;
        p.clamp_in_place();
        clamp_params_to_cap(&mut p, self.effective_cap(), V3_TANK_MAX_LFO_DEPTH);
        self.apply_runtime_params(p);
    }

    /// Reloads Stage IV tank parameters in place and resets all runtime state (`0` allocs, `0` drops).
    pub fn reload_in_place(&mut self, params: &V3Stage4TankParams) {
        self.host_sample_rate_hz = 48_000.0;
        self.update_params(params);
        self.freeze = false;
        self.current_psi = self.params.base_psi;
        let (c, s) = algebraic_unit_rotor(self.current_psi);
        self.cos_psi = c;
        self.sin_psi = s;
        self.reset();
    }

    /// Reconfigures delay lengths, LFO rate/depth, DC blocker pole, and VCA poles under continuous-time ZOH (`0` allocs).
    pub fn reconfigure_with_zoh(&mut self, base: &V3Stage4TankParams, zoh: &ZohRescaledParams) {
        let mut p = *base;
        p.clamp_in_place();
        let cap = self.effective_cap();
        let max_depth = ((cap / 4) as f32).min(V3_TANK_MAX_LFO_DEPTH * 4.0);
        if (zoh.rho_sr - 1.0).abs() >= 1.0e-12 {
            let inv_rho = newton6_reciprocal_f64(zoh.rho_sr.max(1.0e-6));
            for d in &mut p.delay_lengths {
                *d = ((f64::from(*d as u32) * inv_rho + 0.5) as usize).clamp(4, cap - 1);
            }
            p.lfo_depth_samples =
                ((f64::from(p.lfo_depth_samples) * inv_rho) as f32).clamp(0.0, max_depth);
            p.r_dc = zoh.rescale_real_pole(p.r_dc);
            p.vca_att = zoh.rescale_real_pole(p.vca_att);
            p.vca_rel = zoh.rescale_real_pole(p.vca_rel);
            self.host_sample_rate_hz = (48_000.0 * inv_rho) as f32;
        } else {
            self.host_sample_rate_hz = 48_000.0;
        }
        p.clamp_common_fields();
        clamp_params_to_cap(&mut p, cap, max_depth);
        self.apply_runtime_params(p);
    }

    /// Snaps the Givens rotor directly to `base_psi + psi_delta` without slew limiting.
    #[inline]
    pub fn snap_psi_immediate(&mut self, psi_delta: f32) {
        self.current_psi =
            (self.params.base_psi + psi_delta).clamp(-core::f32::consts::PI, core::f32::consts::PI);
        let (c, s) = algebraic_unit_rotor(self.current_psi);
        self.cos_psi = c;
        self.sin_psi = s;
    }

    /// Updates Givens rotation rotor at `3 kHz` (`tick & 15 == 0`) with bounded slew.
    #[inline(always)]
    pub fn refresh_3khz(&mut self, psi_delta: f32) {
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

    /// Resets delay lines, LFO phase, DC blockers, rotor, and VCA envelope to resting equilibrium.
    pub fn reset(&mut self) {
        self.lines.fill([0; V3_TANK_RING_STRIDE]);
        if let Some(ext) = self.ext_ring.as_deref_mut() {
            ext.data.fill(0);
        }
        self.write_ptr = 0;
        self.dc_x_prev = [0.0; V3_TANK_LINES];
        self.dc_y_prev = [0.0; V3_TANK_LINES];
        self.vca_env = 0.0;
        self.lfo_phase_q32 = 0;
        self.snap_psi_immediate(0.0);
    }

    /// Reads the static integer-delay `fp16` tap values prior to advancing the current sample.
    #[must_use]
    pub fn peek_integer_taps<B: MathBackend>(&self) -> [f32; V3_TANK_LINES] {
        let wr = self.write_ptr as usize;
        let d = self.params.delay_lengths;
        let raw_fp16 = if let Some(ext) = self.ext_ring.as_deref() {
            [
                ext.data[(wr + d[0]) & ext.mask],
                ext.data[ext.stride + ((wr + d[1]) & ext.mask)],
                ext.data[2 * ext.stride + ((wr + d[2]) & ext.mask)],
                ext.data[3 * ext.stride + ((wr + d[3]) & ext.mask)],
            ]
        } else {
            [
                self.lines[0][(wr + d[0]) & V3_TANK_MASK],
                self.lines[1][(wr + d[1]) & V3_TANK_MASK],
                self.lines[2][(wr + d[2]) & V3_TANK_MASK],
                self.lines[3][(wr + d[3]) & V3_TANK_MASK],
            ]
        };
        B::fp16x4_to_f32x4(&raw_fp16)
    }

    #[inline(always)]
    fn build_step_ctx(&self) -> TankStepCtx {
        let cap = self.effective_cap();
        let max_tau = if self.params.lfo_depth_samples > 0.0 {
            (cap - 3) as f32
        } else {
            (cap - 1) as f32
        };
        TankStepCtx {
            params: self.params,
            inv_sigma: self.inv_sigma,
            cos_psi: self.cos_psi,
            sin_psi: self.sin_psi,
            cos_st: self.lfo_cos_stereo,
            sin_st: self.lfo_sin_stereo,
            half_alpha_cubic: self.half_alpha_cubic,
            lfo_inc: self.lfo_inc_q32,
            max_tau,
            freeze: self.freeze,
        }
    }

    /// Steps the 4-line `fp16` LFO-modulated spatial FDN unconditionally (`0 FDIV`, `0 FSQRT`).
    #[inline(always)]
    pub fn step_stereo<B: MathBackend>(&mut self, in_l: f32, in_r: f32) -> [f32; 2] {
        let ctx = self.build_step_ctx();
        let mut st = TankMutableRefs {
            write_ptr: &mut self.write_ptr,
            dc_x_prev: &mut self.dc_x_prev,
            dc_y_prev: &mut self.dc_y_prev,
            lfo_phase_q32: &mut self.lfo_phase_q32,
            vca_env: &mut self.vca_env,
        };
        if let Some(ext) = self.ext_ring.as_deref_mut() {
            let mut ring = ExtTankRing {
                buf: &mut ext.data,
                cap: ext.cap,
                stride: ext.stride,
                mask: ext.mask,
            };
            step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r)
        } else {
            let mut ring = InlineTankRing(&mut self.lines);
            step_stereo_ring::<B, _>(&mut ring, ctx, &mut st, in_l, in_r)
        }
    }

    #[inline(always)]
    fn step_subblock_mode<B: MathBackend, const MONO: bool, const FREEZE: bool>(
        &mut self,
        s2_dir: &[[f32; 4]; 16],
        cab_out: &[[f32; 2]; 16],
        clean_sub: &[[f32; 2]; 16],
        out_l: &mut [f32],
        out_r: &mut [f32],
        len: usize,
    ) {
        let ctx = self.build_step_ctx();
        let mut st = TankMutableRefs {
            write_ptr: &mut self.write_ptr,
            dc_x_prev: &mut self.dc_x_prev,
            dc_y_prev: &mut self.dc_y_prev,
            lfo_phase_q32: &mut self.lfo_phase_q32,
            vca_env: &mut self.vca_env,
        };
        let io = SubblockBuffers {
            s2_dir,
            cab_out,
            clean_sub,
            out_l,
            out_r,
            len,
        };
        if let Some(ext) = self.ext_ring.as_deref_mut() {
            let mut ring = ExtTankRing {
                buf: &mut ext.data,
                cap: ext.cap,
                stride: ext.stride,
                mask: ext.mask,
            };
            step_subblock_ring::<B, MONO, FREEZE, _>(&mut ring, ctx, &mut st, io);
        } else {
            let mut ring = InlineTankRing(&mut self.lines);
            step_subblock_ring::<B, MONO, FREEZE, _>(&mut ring, ctx, &mut st, io);
        }
    }

    /// Sub-block Stage IV FDN tank + ballistic VCA step hoisting SIMD constants and state across `0..len`.
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
        if self.freeze {
            self.step_subblock_mode::<B, MONO, true>(s2_dir, cab_out, clean_sub, out_l, out_r, len);
        } else {
            self.step_subblock_mode::<B, MONO, false>(
                s2_dir, cab_out, clean_sub, out_l, out_r, len,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DefaultMathBackend, ScalarBackend};

    #[test]
    fn test_stage4_tank_parity_bibo_zoh_freeze_and_zero_dc() {
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
            assert!((yl[0] - yr[0]).abs() < 1.0e-5, "L parity diff at {n}");
            assert!((yl[1] - yr[1]).abs() < 1.0e-5, "R parity diff at {n}");
        }

        // Verify ZOH rescaling doubles delay lengths at 96 kHz & unitary freeze sustains energy.
        let zoh96 = ZohRescaledParams::new(48_000, 96_000);
        let mut t_zoh = V3Stage4TankHot::from_cold(&tp);
        t_zoh.reconfigure_with_zoh(&tp, &zoh96);
        assert_eq!(t_zoh.params.delay_lengths[0], tp.delay_lengths[0] * 2);

        t_simd.freeze = true;
        let mut freeze_peak = 0.0_f32;
        for _ in 0..128 {
            let [fl, fr] = t_simd.step_stereo::<DefaultMathBackend>(0.0, 0.0);
            freeze_peak = freeze_peak.max(fl.abs()).max(fr.abs());
        }
        assert!(freeze_peak > 1.0e-3);

        // Verify ±1e9 input never produces non-finite fp16 storage or output.
        t_simd.freeze = false;
        let extreme = t_simd.step_stereo::<DefaultMathBackend>(1.0e9, -1.0e9);
        assert!(extreme[0].is_finite() && extreme[1].is_finite());

        // Verify per-line tap_gains clamp and scale independent stereo delay taps.
        let mut tp_tap = V3Stage4TankParams {
            delay_lengths: [8, 12, 16, 20],
            g_loop: 0.0,
            inject_gain: 1.0,
            dry_mix: 0.0,
            wet_mix: 1.0,
            r_dc: 0.0,
            tap_gains: [1.5, -0.75, -5.0, 5.0],
            ..V3Stage4TankParams::identity()
        };
        tp_tap.clamp_in_place();
        assert_eq!(tp_tap.tap_gains, [1.5, -0.75, -2.0, 2.0]);
        let mut t_tap = V3Stage4TankHot::from_cold(&tp_tap);
        assert_eq!(t_tap.peek_integer_taps::<DefaultMathBackend>(), [0.0; 4]);
        t_tap.step_stereo::<DefaultMathBackend>(1.0, 0.0);
        let mut peak_l = 0.0_f32;
        for _ in 0..24 {
            let _taps = t_tap.peek_integer_taps::<DefaultMathBackend>();
            let [yl, _yr] = t_tap.step_stereo::<DefaultMathBackend>(0.0, 0.0);
            peak_l = peak_l.max(yl.abs());
        }
        assert!(
            peak_l > 0.25,
            "custom tap_gains should modulate delay tap output"
        );

        // Verify active LFO modulation: 0-ULP silence, step_stereo==step_subblock & Scalar parity,
        // 96 kHz + 4x bounce ZOH invariance, Doppler sidebands, and g_loop=0.995 BIBO stability.
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
        assert_eq!(
            t_lfo
                .step_stereo::<DefaultMathBackend>(0.0, 0.0)
                .map(f32::to_bits),
            [0; 2]
        );
        t_lfo.reset();

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

        // BIBO decay over 48,000 zero-input samples under high g_loop=0.995 + depth=24.0
        let mut max_abs = 0.0_f32;
        for _ in 0..48_000 {
            let [yl, yr] = t_lfo.step_stereo::<DefaultMathBackend>(0.0, 0.0);
            assert!(yl.is_finite() && yr.is_finite());
            max_abs = max_abs.max(yl.abs()).max(yr.abs());
        }
        assert!(max_abs < 8.0, "BIBO ceiling violated: {max_abs}");

        // ZOH 96 kHz + 4x bounce preserves LFO Hz increment & doubles/quadruples depth in samples
        let mut t_lfo_zoh = V3Stage4TankHot::from_cold(&tp_lfo);
        t_lfo_zoh.reconfigure_with_zoh(&tp_lfo, &zoh96);
        assert!((t_lfo_zoh.params.lfo_depth_samples - 48.0).abs() < 1e-3);
        let base_inc = V3Stage4TankHot::from_cold(&tp_lfo).lfo_inc_q32;
        assert!((t_lfo_zoh.lfo_inc_q32 as i64 - (base_inc / 2) as i64).abs() <= 2);

        // Spectral FM Doppler sidebands vs static delay on pure carrier tone
        let mut tp_static = tp_lfo;
        tp_static.g_loop = 0.0;
        tp_static.dry_mix = 0.0;
        tp_static.wet_mix = 1.0;
        tp_static.lfo_depth_samples = 0.0;
        let mut tp_mod = tp_static;
        tp_mod.lfo_rate_hz = 12.0;
        tp_mod.lfo_depth_samples = 20.0;
        let (mut m_static, mut m_mod) = (
            V3Stage4TankHot::from_cold(&tp_static),
            V3Stage4TankHot::from_cold(&tp_mod),
        );
        let mut diff_energy = 0.0_f64;
        for n in 0..4_000 {
            let tone = 0.5 * (core::f32::consts::TAU * 600.0 * (n as f32) / 48_000.0).sin();
            let [ys, _] = m_static.step_stereo::<DefaultMathBackend>(tone, tone);
            let [ym, _] = m_mod.step_stereo::<DefaultMathBackend>(tone, tone);
            if n > 512 {
                let d = f64::from(ym - ys);
                diff_energy += d * d;
            }
        }
        assert!(diff_energy > 1.0, "expected Doppler sideband difference");
    }

    #[test]
    fn test_stage4_long_delay_cold_ext_lines_bbd_4x_zoh_and_freeze() {
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
            ..V3Stage4TankParams::identity()
        };
        tp.clamp_in_place();
        assert_eq!(tp.delay_lengths[0], 20_160);

        let mut t_ext = V3Stage4TankHot::from_cold(&tp);
        let mut t_sub = V3Stage4TankHot::from_cold(&tp);
        let mut t_scal = V3Stage4TankHot::from_cold(&tp);
        assert!(t_ext.effective_cap() >= 80_640);

        // Impulse appears around 20,160 samples (~420 ms @ 48 kHz), with LFO modulation & freeze
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
            if n < 32 {
                let [sl, sr] = t_scal.step_stereo::<ScalarBackend>(x, x);
                assert!((yl - sl).abs() < 1e-6 && (yr - sr).abs() < 1e-6);
            }
        }
        assert!(
            pre_echo_peak < 1e-6,
            "unexpected early echo {pre_echo_peak}"
        );
        assert!(
            echo_peak > 0.05,
            "expected 420 ms echo peak, got {echo_peak}"
        );

        // step_subblock parity on extended ring
        let mut dir_blk = [[0.0_f32; 4]; 16];
        dir_blk[0] = [0.0, 0.0, 0.5, -0.5];
        let cab_zero = [[0.0_f32; 2]; 16];
        let mut sub_l = [0.0_f32; 16];
        let mut sub_r = [0.0_f32; 16];
        t_sub.step_subblock::<DefaultMathBackend, false>(
            &dir_blk, &cab_zero, &cab_zero, &mut sub_l, &mut sub_r, 16,
        );
        let _ = t_sub.peek_integer_taps::<DefaultMathBackend>();

        // 4x ZOH rescale preserves long echo in physical seconds without clamping to 1023
        let zoh_4x = ZohRescaledParams::new(48_000, 192_000);
        t_ext.reconfigure_with_zoh(&tp, &zoh_4x);
        assert_eq!(t_ext.params.delay_lengths[0], 80_640);
        t_ext.freeze = true;
        let _ = t_ext.step_stereo::<DefaultMathBackend>(0.0, 0.0);
        t_ext.reset();
        assert_eq!(
            t_ext.peek_integer_taps::<DefaultMathBackend>(),
            [0.0; V3_TANK_LINES]
        );

        // Cold ring growth + reuse + in-place reload (`0` RT allocs/drops)
        let mut t_grow = V3Stage4TankHot::from_cold(&V3Stage4TankParams::identity());
        t_grow.ensure_cold_ext_cap(&V3Stage4TankParams::identity(), 0);
        assert_eq!(t_grow.effective_cap(), V3_TANK_CAP);
        t_grow.ensure_cold_ext_cap(&tp, 0);
        assert!(t_grow.effective_cap() >= 80_640);
        let cap_before = t_grow.effective_cap();
        t_grow.ensure_cold_ext_cap(&tp, 4_096);
        assert_eq!(t_grow.effective_cap(), cap_before);
        let mut tp_longer = tp;
        tp_longer.delay_lengths = [36_000, 36_007, 36_013, 36_019];
        t_grow.ensure_cold_ext_cap(&tp_longer, 144_500);
        assert!(t_grow.effective_cap() >= 144_500);
        t_grow.reload_in_place(&tp);
        assert_eq!(t_grow.params.delay_lengths[0], 20_160);
        assert!(!t_grow.freeze);
    }
}
