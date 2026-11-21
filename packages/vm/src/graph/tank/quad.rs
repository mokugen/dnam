//! `224 B` 16-byte-aligned spatial tank quad active-hot state (`TankQuadHotCompact`), cold wire
//! hydration/ZOH rescaling, dual-latch `WE` hold control, and causal 4-lane Top-of-Frame read +
//! End-of-Frame write execution (`ref[dsp.op-t.ring.declaration]`, `ref[dsp.op-t.reader.catmull-rom]`).

mod step;

use core::f32::consts::PI;

use dnam_kernels::{
    QUAD_FLAG_HAS_EXT, QUAD_FLAG_HOST_WRITE_LOCK, QUAD_FLAG_SCHROEDER_ENABLE,
    QUAD_FLAG_WE_HOLD_ACTIVE, QUAD_WE_LOCK_MASK, TANK_MAX_G_LIN_F32, algebraic_unit_rotor_f32,
    compute_half_alpha_cubic_f32, compute_op_t_pitch_down_g_loop_eff, update_we_schmitt_flag,
};

use crate::bin::arena_wire::{TANK_FLAG_SCHROEDER_ENABLE, TankQuadWire};
use crate::control::bus::laws::{quantize_tank_lfo_inc_q32, quantize_tank_ramp_inc_q32};
use crate::math::{
    DC_BLOCKER_5HZ_48K_F32, clamp_finite_f32, const_abs_f64, f32_to_fp16, floor_f32, fp16_to_f32,
    powf_f64, rcp_nr6_f64,
};
use crate::types::VmError;

use super::ring::{
    PartitionedSchroederBank, TANK_INLINE_QUAD0_CAP, TankExtRing, compute_quad_ext_requirements,
    compute_schroeder_requirements,
};

pub(crate) const DAMP_EFF_MAX: f32 = 0.999_95;
const R_DC_WIRE_MIN: f32 = 0.95;
const R_DC_WIRE_MAX: f32 = 0.9999;
const LOOP_DC_ZOH_MIN: f32 = 1.0e-4;
const LOOP_DC_ZOH_MAX: f32 = 0.999_995;
const DAMP_WIRE_MAX: f32 = 0.9995;
const VCA_POLE_MAX: f32 = 0.999_99;

#[inline]
fn zoh_scale_pole(
    raw: f32,
    rho_vm: f64,
    lo: f32,
    hi: f32,
    zero_floor: f32,
    exact_48k: bool,
) -> f32 {
    if raw <= zero_floor {
        return 0.0;
    }
    if exact_48k {
        raw.clamp(lo, hi)
    } else {
        (powf_f64(f64::from(raw), rho_vm) as f32).clamp(lo, hi)
    }
}

/// Active-hot `224 B` 16-byte-aligned working-set block for one 4-lane spatial tank quad (`op_t_transport.md §1.1.1`).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TankQuadHotCompact {
    // [0..64): 2-state Direct-Form I loop DC blockers, HF damp state, stereo wet DC blockers.
    pub dc_x_prev: [f32; 4],
    pub dc_y_prev: [f32; 4],
    pub damp_state: [f32; 4],
    pub wet_dc_x: [f32; 2],
    pub wet_dc_y: [f32; 2],
    // [64..96): effective-rate base delays and ramp spans.
    pub delay_samples: [f32; 4],
    pub ramp_spans_samples: [f32; 4],
    // [96..152): backward cursor, LFO/ramp accumulators, poles, Givens rotor, VCA state.
    pub write_ptr: u32,
    pub lfo_phase_q32: u32,
    pub lfo_inc_q32: u32,
    pub ramp_phase_q32: u32,
    pub ramp_inc_q32: i32,
    pub g_loop: f32,
    pub r_dc: f32,
    pub damp_pole: f32,
    pub current_psi: f32,
    pub cos_psi: f32,
    pub sin_psi: f32,
    pub wet_dc_pole: f32,
    pub vca_env: f32,
    pub vca_pole: f32,
    // [152..224): inline hold offsets, Q16 phase offsets, fp16 gains/scalars, flags, explicit tail pad.
    pub we_hold_off: [u16; 4],
    pub lfo_phase_off_q16: [u16; 4],
    pub ramp_phase_off_q16: [u16; 4],
    pub lfo_depths_fp16: [u16; 4],
    pub input_gains_fp16: [u16; 4],
    pub wet_tap_l_fp16: [u16; 4],
    pub wet_tap_r_fp16: [u16; 4],
    pub window_depth_fp16: u16,
    pub dry_mix_fp16: u16,
    pub inv_sigma_fp16: u16,
    pub half_alpha_cubic_fp16: u16,
    pub base_psi_fp16: u16,
    pub vca_sens_fp16: u16,
    pub flags: u8,
    pub _pad8: u8,
    pub _pad16: u16,
}

const _: () = assert!(core::mem::size_of::<TankQuadHotCompact>() == 224);
const _: () = assert!(core::mem::align_of::<TankQuadHotCompact>() == 16);
const _: () = assert!(core::mem::offset_of!(TankQuadHotCompact, delay_samples) == 64);
const _: () = assert!(core::mem::offset_of!(TankQuadHotCompact, write_ptr) == 96);
const _: () = assert!(core::mem::offset_of!(TankQuadHotCompact, we_hold_off) == 152);
const _: () = assert!(core::mem::offset_of!(TankQuadHotCompact, window_depth_fp16) == 208);
const _: () = assert!(core::mem::offset_of!(TankQuadHotCompact, flags) == 220);

impl Default for TankQuadHotCompact {
    fn default() -> Self {
        Self {
            dc_x_prev: [0.0; 4],
            dc_y_prev: [0.0; 4],
            damp_state: [0.0; 4],
            wet_dc_x: [0.0; 2],
            wet_dc_y: [0.0; 2],
            delay_samples: [5.0; 4],
            ramp_spans_samples: [0.0; 4],
            write_ptr: 0,
            lfo_phase_q32: 0,
            lfo_inc_q32: 0,
            ramp_phase_q32: 0,
            ramp_inc_q32: 0,
            g_loop: 0.0,
            r_dc: DC_BLOCKER_5HZ_48K_F32,
            damp_pole: 0.0,
            current_psi: 0.0,
            cos_psi: 1.0,
            sin_psi: 0.0,
            wet_dc_pole: DC_BLOCKER_5HZ_48K_F32,
            vca_env: 0.0,
            vca_pole: 0.0,
            we_hold_off: [0; 4],
            lfo_phase_off_q16: [0; 4],
            ramp_phase_off_q16: [0; 4],
            lfo_depths_fp16: [0; 4],
            input_gains_fp16: [0; 4],
            wet_tap_l_fp16: [0; 4],
            wet_tap_r_fp16: [0; 4],
            window_depth_fp16: 0,
            dry_mix_fp16: f32_to_fp16(1.0),
            inv_sigma_fp16: f32_to_fp16(1.0),
            half_alpha_cubic_fp16: f32_to_fp16(0.5),
            base_psi_fp16: 0,
            vca_sens_fp16: 0,
            flags: 0,
            _pad8: 0,
            _pad16: 0,
        }
    }
}

/// Cold hydration parameters shared across VM initialization and sample-rate reconfiguration.
#[derive(Clone, Copy, Debug)]
pub struct QuadHydrateConfig {
    pub quad_idx: usize,
    pub fs_vm: f64,
    pub rho_vm: f64,
    pub declared_max_delay_48k: u32,
    pub ext_ring_pow2_frames: u8,
    pub dc_blocker_pole_5hz: f32,
}

/// Cold immutable ZOH-converted base metadata retained alongside [`TankQuadHotCompact`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadColdMeta {
    pub quad_idx: u8,
    pub cap: usize,
    pub base_g_loop: f32,
    pub base_damp_pole_zoh: f32,
    pub base_lfo_rate_hz: f32,
    pub base_ramp_rate_hz: f32,
    pub base_dry_mix: f32,
    pub base_psi: f32,
    pub base_window_depth: f32,
    pub attack_cap_vca: f32,
    pub inv_fs_vm: f64,
    pub scale_inv_rho_vm: f32,
    pub base_lfo_depths: [f32; 4],
    pub base_input_gains: [f32; 4],
    pub base_allpass_g: [f32; 4],
}

impl Default for QuadColdMeta {
    #[inline]
    fn default() -> Self {
        Self {
            quad_idx: 0,
            cap: TANK_INLINE_QUAD0_CAP,
            base_g_loop: 0.0,
            base_damp_pole_zoh: 0.0,
            base_lfo_rate_hz: 0.0,
            base_ramp_rate_hz: 0.0,
            base_dry_mix: 1.0,
            base_psi: 0.0,
            base_window_depth: 0.0,
            attack_cap_vca: 0.99,
            inv_fs_vm: 1.0 / 48_000.0,
            scale_inv_rho_vm: 1.0,
            base_lfo_depths: [0.0; 4],
            base_input_gains: [0.0; 4],
            base_allpass_g: [0.0; 4],
        }
    }
}

/// Cohesive effective per-sample parameters derived from [`TankQuadHotCompact`] and `3 kHz` ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadEffectiveTargets {
    pub cap: usize,
    pub scale_inv_rho_vm: f32,
    pub tank_delay_mod_48k: f32,
    pub delay_scale_oct: f32,
    pub delay_scale_pow2: f32,
    pub g_loop_eff: f32,
    pub inv_sigma: f32,
    pub half_alpha_cubic: f32,
    pub cos_psi: f32,
    pub sin_psi: f32,
    pub damp_pole_eff: f32,
    pub dry_mix_eff: f32,
    pub wet_scale: f32,
    pub window_depth_eff: f32,
    pub ramp_rate_hz_eff: f32,
    pub vca_pole: f32,
    pub vca_sens: f32,
    pub attack_cap_vca: f32,
    pub lfo_depths_eff: [f32; 4],
    pub input_gains_eff: [f32; 4],
    pub allpass_g_eff: [f32; 4],
}

impl Default for QuadEffectiveTargets {
    #[inline]
    fn default() -> Self {
        Self::from_quad_and_meta(&TankQuadHotCompact::default(), &QuadColdMeta::default())
    }
}

impl QuadEffectiveTargets {
    /// Initializes unmodulated effective parameters directly from `(quad, meta)`.
    #[inline]
    pub fn from_quad_and_meta(quad: &TankQuadHotCompact, meta: &QuadColdMeta) -> Self {
        let g_loop_eff = compute_op_t_pitch_down_g_loop_eff(
            quad.g_loop,
            quad.ramp_spans_samples,
            quad.ramp_inc_q32,
        );
        Self {
            cap: meta.cap,
            scale_inv_rho_vm: meta.scale_inv_rho_vm,
            tank_delay_mod_48k: 0.0,
            delay_scale_oct: 0.0,
            delay_scale_pow2: 1.0,
            g_loop_eff,
            inv_sigma: fp16_to_f32(quad.inv_sigma_fp16),
            half_alpha_cubic: fp16_to_f32(quad.half_alpha_cubic_fp16),
            cos_psi: quad.cos_psi,
            sin_psi: quad.sin_psi,
            damp_pole_eff: quad.damp_pole,
            dry_mix_eff: fp16_to_f32(quad.dry_mix_fp16),
            wet_scale: 1.0,
            window_depth_eff: fp16_to_f32(quad.window_depth_fp16),
            ramp_rate_hz_eff: meta.base_ramp_rate_hz,
            vca_pole: quad.vca_pole,
            vca_sens: fp16_to_f32(quad.vca_sens_fp16),
            attack_cap_vca: meta.attack_cap_vca,
            lfo_depths_eff: quad.lfo_depths_fp16.map(fp16_to_f32),
            input_gains_eff: quad.input_gains_fp16.map(fp16_to_f32),
            allpass_g_eff: meta.base_allpass_g,
        }
    }
}

/// Result emitted by [`TankQuadHotCompact::step_top_of_frame_read`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QuadStepResult {
    /// Post-window, post-Schroeder 4-lane vector fed to [`TankQuadHotCompact::step_end_of_frame_write`].
    pub v_read: [f32; 4],
    /// Stereo quad wet+dry output `(y_{q,L}, y_{q,R})`.
    pub y_lr: (f32, f32),
}

/// Borrowed mutable delay ring backing passed into [`TankQuadHotCompact`] step calls.
#[derive(Debug)]
pub enum QuadRingMut<'a, 'b> {
    /// Inline `Quad 0` `128 × fp16[4]` buffer (`1,024 B`).
    Inline(&'a mut [[u16; 4]; TANK_INLINE_QUAD0_CAP]),
    /// External cold [`TankExtRing`] (`128..=524_288` frames in `fp16` or `f32`).
    Ext(&'a mut TankExtRing<'b>),
}

impl TankQuadHotCompact {
    /// Cold hydration from [`TankQuadWire`] into [`TankQuadHotCompact`] and [`QuadColdMeta`].
    pub fn from_wire(
        wire: &TankQuadWire,
        cfg: &QuadHydrateConfig,
    ) -> Result<(Self, QuadColdMeta), VmError> {
        let mut hot = Self::default();
        let meta = hot.apply_wire(wire, cfg)?;
        Ok((hot, meta))
    }

    /// Re-hydrates all parameters from `wire` according to [`QuadHydrateConfig`].
    pub fn apply_wire(
        &mut self,
        wire: &TankQuadWire,
        cfg: &QuadHydrateConfig,
    ) -> Result<QuadColdMeta, VmError> {
        if !cfg.fs_vm.is_finite()
            || cfg.fs_vm <= 0.0
            || !cfg.rho_vm.is_finite()
            || cfg.rho_vm <= 0.0
        {
            return Err(VmError::InvalidSampleRate);
        }
        let (inv_fs_vm, scale) = (rcp_nr6_f64(cfg.fs_vm), rcp_nr6_f64(cfg.rho_vm));
        let exact_48k = const_abs_f64(cfg.rho_vm - 1.0) < 1.0e-12;
        let (has_ext, cap) = compute_quad_ext_requirements(
            wire,
            cfg.quad_idx,
            cfg.rho_vm,
            cfg.declared_max_delay_48k,
            cfg.ext_ring_pow2_frames,
        );

        let wet_dc_base = if cfg.dc_blocker_pole_5hz > 0.8
            && cfg.dc_blocker_pole_5hz < 1.0
            && cfg.dc_blocker_pole_5hz.is_finite()
        {
            cfg.dc_blocker_pole_5hz
        } else {
            DC_BLOCKER_5HZ_48K_F32
        };
        let raw_r_dc = clamp_finite_f32(
            wire.r_dc.get(),
            R_DC_WIRE_MIN,
            R_DC_WIRE_MAX,
            DC_BLOCKER_5HZ_48K_F32,
        );
        let raw_damp = clamp_finite_f32(wire.damp_pole.get(), 0.0, DAMP_WIRE_MAX, 0.0);
        let raw_vca = clamp_finite_f32(wire.vca_pole.get(), 0.0, VCA_POLE_MAX, 0.0);

        self.r_dc = zoh_scale_pole(
            raw_r_dc,
            cfg.rho_vm,
            LOOP_DC_ZOH_MIN,
            LOOP_DC_ZOH_MAX,
            -1.0,
            exact_48k,
        );
        self.wet_dc_pole = zoh_scale_pole(
            wet_dc_base,
            cfg.rho_vm,
            LOOP_DC_ZOH_MIN,
            LOOP_DC_ZOH_MAX,
            -1.0,
            exact_48k,
        );
        self.damp_pole = zoh_scale_pole(raw_damp, cfg.rho_vm, 0.0, DAMP_EFF_MAX, 1.0e-7, exact_48k);
        self.vca_pole = zoh_scale_pole(raw_vca, cfg.rho_vm, 0.0, VCA_POLE_MAX, 0.0, exact_48k);
        let attack_cap_vca = zoh_scale_pole(0.99, cfg.rho_vm, 0.0, VCA_POLE_MAX, 0.0, exact_48k);

        let (mut base_lfo_depths, mut base_input_gains) = ([0.0_f32; 4], [0.0_f32; 4]);
        let lfo_cap_max = (512.0_f32).min(floor_f32((cap as f32) * 0.25));
        let mut is_cold_mod = false;
        for k in 0..4 {
            let lfo_w =
                clamp_finite_f32(fp16_to_f32(wire.lfo_depths_fp16[k].get()), 0.0, 512.0, 0.0);
            let span_w = clamp_finite_f32(wire.ramp_spans_samples[k].get(), 0.0, 48_000.0, 0.0);
            is_cold_mod |= lfo_w > 0.0 || span_w > 0.0;
            base_lfo_depths[k] =
                clamp_finite_f32((f64::from(lfo_w) * scale) as f32, 0.0, lfo_cap_max, 0.0);
            self.lfo_depths_fp16[k] = f32_to_fp16(base_lfo_depths[k]);
            self.lfo_phase_off_q16[k] = wire.lfo_phase_off_q16[k].get();
            self.ramp_phase_off_q16[k] = wire.ramp_phase_off_q16[k].get();
            base_input_gains[k] =
                clamp_finite_f32(fp16_to_f32(wire.input_gains_fp16[k].get()), -2.0, 2.0, 0.0);
            self.input_gains_fp16[k] = f32_to_fp16(base_input_gains[k]);
            self.wet_tap_l_fp16[k] = f32_to_fp16(clamp_finite_f32(
                fp16_to_f32(wire.wet_tap_l_fp16[k].get()),
                -2.0,
                2.0,
                0.0,
            ));
            self.wet_tap_r_fp16[k] = f32_to_fp16(clamp_finite_f32(
                fp16_to_f32(wire.wet_tap_r_fp16[k].get()),
                -2.0,
                2.0,
                0.0,
            ));
        }

        let max_tau = (cap - if is_cold_mod { 3 } else { 2 }) as f32;
        let max_span = (max_tau - 5.0).max(0.0);
        for k in 0..4 {
            let d_w = clamp_finite_f32(wire.delay_samples[k].get(), 5.0, 48_000.0, 5.0);
            let s_w = clamp_finite_f32(wire.ramp_spans_samples[k].get(), 0.0, 48_000.0, 0.0);
            self.delay_samples[k] = ((f64::from(d_w) * scale) as f32).clamp(5.0, max_tau);
            self.ramp_spans_samples[k] = ((f64::from(s_w) * scale) as f32).clamp(0.0, max_span);
        }

        let base_g_loop = clamp_finite_f32(wire.g_loop.get(), 0.0, TANK_MAX_G_LIN_F32, 0.0);
        self.g_loop = base_g_loop;
        let sigma = clamp_finite_f32(fp16_to_f32(wire.sigma_knee_fp16.get()), 1.0e-3, 8.0, 1.0);
        self.inv_sigma_fp16 = f32_to_fp16(rcp_nr6_f64(f64::from(sigma)) as f32);

        let raw_dry_bits = wire.dry_mix_fp16.get();
        let raw_dry = fp16_to_f32(raw_dry_bits);
        let base_dry_mix = if raw_dry_bits == 0 {
            0.0
        } else {
            clamp_finite_f32(
                raw_dry,
                -2.0,
                2.0,
                if cfg.quad_idx == 0 { 1.0 } else { 0.0 },
            )
        };
        self.dry_mix_fp16 = if raw_dry_bits == 0 {
            0
        } else {
            f32_to_fp16(base_dry_mix)
        };

        let base_psi = clamp_finite_f32(fp16_to_f32(wire.fb_psi_fp16.get()), -PI, PI, 0.0);
        self.base_psi_fp16 = f32_to_fp16(base_psi);
        self.current_psi = base_psi;
        (self.cos_psi, self.sin_psi) = algebraic_unit_rotor_f32(base_psi);

        let base_window_depth =
            clamp_finite_f32(fp16_to_f32(wire.window_depth_fp16.get()), 0.0, 1.0, 0.0);
        self.window_depth_fp16 = f32_to_fp16(base_window_depth);
        let base_vca_sens = clamp_finite_f32(fp16_to_f32(wire.vca_sens_fp16.get()), -8.0, 8.0, 0.0);
        self.vca_sens_fp16 = f32_to_fp16(base_vca_sens);

        let base_lfo_rate_hz = clamp_finite_f32(wire.lfo_rate_hz.get(), 0.0, 16.0, 0.0);
        let base_ramp_rate_hz = clamp_finite_f32(wire.ramp_rate_hz.get(), -16.0, 16.0, 0.0);
        let max_lfo = base_lfo_depths.iter().copied().fold(0.0_f32, f32::max);
        self.lfo_inc_q32 = quantize_tank_lfo_inc_q32(base_lfo_rate_hz, max_lfo, inv_fs_vm);
        self.ramp_inc_q32 = quantize_tank_ramp_inc_q32(base_ramp_rate_hz, inv_fs_vm);
        let g_eff_cold = compute_op_t_pitch_down_g_loop_eff(
            base_g_loop,
            self.ramp_spans_samples,
            self.ramp_inc_q32,
        );
        self.half_alpha_cubic_fp16 = f32_to_fp16(compute_half_alpha_cubic_f32(g_eff_cold));

        self.flags = (if has_ext { QUAD_FLAG_HAS_EXT } else { 0 })
            | (if (wire.flags & TANK_FLAG_SCHROEDER_ENABLE) != 0 {
                QUAD_FLAG_SCHROEDER_ENABLE
            } else {
                0
            });
        (self._pad8, self._pad16) = (0, 0);
        let (_, base_allpass_g, _, _) = compute_schroeder_requirements(wire, cfg.rho_vm);

        Ok(QuadColdMeta {
            quad_idx: cfg.quad_idx as u8,
            cap,
            base_g_loop,
            base_damp_pole_zoh: self.damp_pole,
            base_lfo_rate_hz,
            base_ramp_rate_hz,
            base_dry_mix,
            base_psi,
            base_window_depth,
            attack_cap_vca,
            inv_fs_vm,
            scale_inv_rho_vm: scale as f32,
            base_lfo_depths,
            base_input_gains,
            base_allpass_g,
        })
    }

    /// Re-hydrates a quad for a new sample rate while zero-resetting dynamic runtime state
    /// and preserving `HOST_WRITE_LOCK` (bit 2).
    pub fn reconfigure_sample_rate(
        &mut self,
        wire: &TankQuadWire,
        cfg: &QuadHydrateConfig,
        ext_ring: Option<&mut TankExtRing<'_>>,
        inline_q0_ring: Option<&mut [[u16; 4]; TANK_INLINE_QUAD0_CAP]>,
        mut schroeder: Option<&mut PartitionedSchroederBank<'_>>,
    ) -> Result<QuadColdMeta, VmError> {
        if let Some(ap) = schroeder.as_deref_mut() {
            ap.reconfigure_sample_rate(wire, cfg.rho_vm)?;
        }
        let host_lock = self.flags & QUAD_FLAG_HOST_WRITE_LOCK;
        let meta = self.apply_wire(wire, cfg)?;
        self.flags |= host_lock;
        self.reset(ext_ring, inline_q0_ring, schroeder);
        Ok(meta)
    }

    /// Restores deterministic resting equilibrium while preserving `HOST_WRITE_LOCK` (bit 2).
    pub fn reset(
        &mut self,
        ext_ring: Option<&mut TankExtRing<'_>>,
        inline_q0_ring: Option<&mut [[u16; 4]; TANK_INLINE_QUAD0_CAP]>,
        schroeder: Option<&mut PartitionedSchroederBank<'_>>,
    ) {
        (self.dc_x_prev, self.dc_y_prev, self.damp_state) = ([0.0; 4], [0.0; 4], [0.0; 4]);
        (self.wet_dc_x, self.wet_dc_y) = ([0.0; 2], [0.0; 2]);
        (
            self.write_ptr,
            self.lfo_phase_q32,
            self.ramp_phase_q32,
            self.vca_env,
        ) = (0, 0, 0, 0.0);
        self.we_hold_off = [0; 4];
        self.current_psi = fp16_to_f32(self.base_psi_fp16);
        (self.cos_psi, self.sin_psi) = algebraic_unit_rotor_f32(self.current_psi);
        self.flags &= !QUAD_FLAG_WE_HOLD_ACTIVE;

        if let Some(ext) = ext_ring {
            ext.reset();
        }
        if let Some(inline) = inline_q0_ring {
            inline.fill([0; 4]);
        }
        if let Some(ap) = schroeder {
            ap.reset();
        }
    }

    /// Sets or clears bit 2 (`HOST_WRITE_LOCK`), clearing hold offsets on hold release.
    #[inline]
    pub fn set_host_write_lock(&mut self, locked: bool, ext_ring: Option<&mut TankExtRing<'_>>) {
        let was_held = (self.flags & QUAD_WE_LOCK_MASK) != 0;
        self.flags = if locked {
            self.flags | QUAD_FLAG_HOST_WRITE_LOCK
        } else {
            self.flags & !QUAD_FLAG_HOST_WRITE_LOCK
        };
        if was_held && (self.flags & QUAD_WE_LOCK_MASK) == 0 {
            self.clear_hold_offsets(ext_ring);
        }
    }

    /// Updates bit 0 (`WE_HOLD_ACTIVE`) via Schmitt `(0.45, 0.55)`, clearing hold offsets on release.
    #[inline]
    pub fn update_we_schmitt(&mut self, we_eff: f32, ext_ring: Option<&mut TankExtRing<'_>>) {
        let was_held = (self.flags & QUAD_WE_LOCK_MASK) != 0;
        self.flags = update_we_schmitt_flag(self.flags, we_eff);
        if was_held && (self.flags & QUAD_WE_LOCK_MASK) == 0 {
            self.clear_hold_offsets(ext_ring);
        }
    }

    #[inline(always)]
    fn clear_hold_offsets(&mut self, ext_ring: Option<&mut TankExtRing<'_>>) {
        self.we_hold_off = [0; 4];
        if let Some(ext) = ext_ring {
            ext.ext_we_hold_off = [0; 4];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DefaultMathBackend;
    use crate::types::ColdBuf;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    #[test]
    fn test_wire_hydration_dry_mix_rdc_and_we_dual_latch_hold() {
        let mut w0 = TankQuadWire::new_zeroed();
        w0.delay_samples = [53.0, 67.0, 79.0, 89.0].map(F32::new);
        (w0.g_loop, w0.r_dc, w0.damp_pole, w0.vca_pole) = (
            F32::new(0.85),
            F32::new(0.5),
            F32::new(0.2),
            F32::new(0.9995),
        );
        w0.input_gains_fp16 = [1.0, 0.5, -0.5, 0.25].map(|v| U16::new(f32_to_fp16(v)));
        w0.wet_tap_l_fp16 = [0.5, 0.25, -0.25, 0.1].map(|v| U16::new(f32_to_fp16(v)));
        w0.wet_tap_r_fp16 = [-0.25, 0.5, 0.1, -0.3].map(|v| U16::new(f32_to_fp16(v)));
        w0.sigma_knee_fp16 = U16::new(f32_to_fp16(1.0));
        let cfg0 = QuadHydrateConfig {
            quad_idx: 0,
            fs_vm: 48_000.0,
            rho_vm: 1.0,
            declared_max_delay_48k: 89,
            ext_ring_pow2_frames: 7,
            dc_blocker_pole_5hz: DC_BLOCKER_5HZ_48K_F32,
        };
        let (mut q0, meta0) = TankQuadHotCompact::from_wire(&w0, &cfg0).unwrap();
        assert_eq!(
            (q0.dry_mix_fp16, q0.r_dc, q0.flags & QUAD_FLAG_HAS_EXT),
            (0x0000, 0.95, 0)
        );

        w0.dry_mix_fp16 = U16::new(0x7E00); // NaN fallback: 1.0 on Q0, 0.0 on Q1
        assert_eq!(
            fp16_to_f32(
                TankQuadHotCompact::from_wire(&w0, &cfg0)
                    .unwrap()
                    .0
                    .dry_mix_fp16
            ),
            1.0
        );
        assert_eq!(
            TankQuadHotCompact::from_wire(
                &w0,
                &QuadHydrateConfig {
                    quad_idx: 1,
                    ..cfg0
                }
            )
            .unwrap()
            .0
            .dry_mix_fp16,
            0x0000
        );

        let mut inline_ring = [[0_u16; 4]; TANK_INLINE_QUAD0_CAP];
        let eff0 = QuadEffectiveTargets::from_quad_and_meta(&q0, &meta0);
        for _ in 0..8 {
            let r = q0.step_top_of_frame_read::<DefaultMathBackend>(
                QuadRingMut::Inline(&mut inline_ring),
                None,
                &eff0,
                (0.25, -0.25),
            );
            q0.step_end_of_frame_write::<DefaultMathBackend>(
                QuadRingMut::Inline(&mut inline_ring),
                &eff0,
                r.v_read,
                (0.25, -0.25),
            );
        }
        let wr_frozen = q0.write_ptr;
        q0.update_we_schmitt(0.2, None);
        q0.set_host_write_lock(true, None);
        assert_eq!(q0.flags & QUAD_WE_LOCK_MASK, QUAD_WE_LOCK_MASK);

        let ring_snap = inline_ring;
        let held = q0.step_top_of_frame_read::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            None,
            &eff0,
            (0.1, 0.1),
        );
        assert!(!q0.step_end_of_frame_write::<DefaultMathBackend>(
            QuadRingMut::Inline(&mut inline_ring),
            &eff0,
            held.v_read,
            (0.1, 0.1)
        ));
        assert_eq!((q0.write_ptr, inline_ring), (wr_frozen, ring_snap));
        assert!(q0.we_hold_off[0] > 0);

        // Clearing bit 0 keeps bit 2 and hold counters; reset clears WE only; overload saturates fp16.
        q0.update_we_schmitt(0.9, None);
        assert_eq!(q0.flags & QUAD_WE_LOCK_MASK, QUAD_FLAG_HOST_WRITE_LOCK);
        assert!(q0.we_hold_off[0] > 0);
        q0.reset(None, Some(&mut inline_ring), None);
        assert_eq!(
            (q0.flags & QUAD_WE_LOCK_MASK, q0.we_hold_off),
            (QUAD_FLAG_HOST_WRITE_LOCK, [0; 4])
        );
        q0.set_host_write_lock(false, None);
        assert_eq!(q0.flags & QUAD_WE_LOCK_MASK, 0);

        let mut fp16_buf = [0_u16; 128 * 4];
        let mut ext16 = TankExtRing::from_fp16_buf(ColdBuf::borrowed(&mut fp16_buf), 128).unwrap();
        assert!(q0.step_end_of_frame_write::<DefaultMathBackend>(
            QuadRingMut::Ext(&mut ext16),
            &eff0,
            [0.0; 4],
            (100_000.0, 0.0)
        ));
    }
}
