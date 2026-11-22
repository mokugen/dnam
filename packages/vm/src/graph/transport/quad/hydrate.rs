//! Cold [`TransportQuadWire`] hydration, ZOH pole rescaling, and sample-rate reconfiguration
//! for [`super::TransportQuadHotCompact`] (`ref[dsp.op-t.ring.declaration]`).

use core::f32::consts::PI;

use dnam_kernels::{
    QUAD_FLAG_HAS_EXT, QUAD_FLAG_HOST_WRITE_LOCK, TANK_MAX_G_LIN_F32, TANK_MAX_SIGMA_SAT_F32,
    algebraic_unit_rotor_f32, compute_half_alpha_cubic_f32, compute_op_t_pitch_down_g_loop_eff,
};

use crate::bin::arena_wire::TransportQuadWire;
use crate::control::bus::laws::{quantize_transport_lfo_inc_q32, quantize_transport_ramp_inc_q32};
use crate::math::{
    DC_BLOCKER_5HZ_48K_F32, clamp_finite_f32, const_abs_f64, f32_to_fp16, floor_f32, fp16_to_f32,
    powf_f64, rcp_nr6_f64,
};
use crate::types::VmError;

use super::super::ring::{
    PartitionedSchroederBank, TRANSPORT_INLINE_QUAD0_CAP, TransportExtRing,
    compute_quad_ext_requirements, compute_schroeder_requirements,
};
use super::{
    DAMP_EFF_MAX, MIN_SIGMA_SAT_F32, QuadColdMeta, QuadHydrateConfig, TransportQuadHotCompact,
};

const R_DC_WIRE_MIN: f32 = 0.95;
const R_DC_WIRE_MAX: f32 = 0.9999;
const LOOP_DC_ZOH_MIN: f32 = 1.0e-4;
const LOOP_DC_ZOH_MAX: f32 = 0.999_995;
const DAMP_WIRE_MAX: f32 = 0.9995;

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

impl TransportQuadHotCompact {
    /// Cold hydration from [`TransportQuadWire`] into [`TransportQuadHotCompact`] and [`QuadColdMeta`].
    pub fn from_wire(
        wire: &TransportQuadWire,
        cfg: &QuadHydrateConfig,
    ) -> Result<(Self, QuadColdMeta), VmError> {
        let mut hot = Self::default();
        let meta = hot.apply_wire(wire, cfg)?;
        Ok((hot, meta))
    }

    /// Re-hydrates all parameters from `wire` according to [`QuadHydrateConfig`].
    pub fn apply_wire(
        &mut self,
        wire: &TransportQuadWire,
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

        let sigma_sat = clamp_finite_f32(
            fp16_to_f32(wire.sigma_knee_fp16.get()),
            MIN_SIGMA_SAT_F32,
            TANK_MAX_SIGMA_SAT_F32,
            TANK_MAX_SIGMA_SAT_F32,
        );
        self.sigma_sat_fp16 = f32_to_fp16(sigma_sat);
        let sigma_q = fp16_to_f32(self.sigma_sat_fp16);
        let inv_sigma_23 = (2.0_f64 * rcp_nr6_f64(3.0 * f64::from(sigma_q))) as f32;
        self.inv_sigma_23_fp16 = f32_to_fp16(inv_sigma_23);

        let base_freeze_mix =
            clamp_finite_f32(fp16_to_f32(wire.freeze_mix_fp16.get()), 0.0, 1.0, 0.0);
        self.freeze_mix_fp16 = f32_to_fp16(base_freeze_mix);
        let freeze_q = fp16_to_f32(self.freeze_mix_fp16);

        let base_psi = clamp_finite_f32(fp16_to_f32(wire.fb_psi_fp16.get()), -PI, PI, 0.0);
        let base_scat_psi = clamp_finite_f32(fp16_to_f32(wire.scat_psi_fp16.get()), -PI, PI, 0.0);
        self.base_psi_fp16 = f32_to_fp16(base_psi);
        self.current_psi = base_psi;
        (self.cos_psi, self.sin_psi) = algebraic_unit_rotor_f32(base_psi);

        let base_window_depth =
            clamp_finite_f32(fp16_to_f32(wire.window_depth_fp16.get()), 0.0, 1.0, 0.0);
        self.window_depth_fp16 = f32_to_fp16(base_window_depth);

        let base_lfo_rate_hz = clamp_finite_f32(wire.lfo_rate_hz.get(), 0.0, 16.0, 0.0);
        let base_ramp_rate_hz = clamp_finite_f32(wire.ramp_rate_hz.get(), -16.0, 16.0, 0.0);
        let max_lfo = base_lfo_depths.iter().copied().fold(0.0_f32, f32::max);
        self.lfo_inc_q32 = quantize_transport_lfo_inc_q32(base_lfo_rate_hz, max_lfo, inv_fs_vm);
        self.ramp_inc_q32 = quantize_transport_ramp_inc_q32(base_ramp_rate_hz, inv_fs_vm);
        let g_eff_cold = compute_op_t_pitch_down_g_loop_eff(
            base_g_loop,
            self.ramp_spans_samples,
            self.ramp_inc_q32,
        );
        self.half_alpha_cubic_fp16 =
            f32_to_fp16(compute_half_alpha_cubic_f32(g_eff_cold, freeze_q));

        self.flags = if has_ext { QUAD_FLAG_HAS_EXT } else { 0 };
        (self._pad8, self._pad16) = (0, 0);
        let (_, base_allpass_g, _, _) = compute_schroeder_requirements(wire, cfg.rho_vm);

        Ok(QuadColdMeta {
            quad_idx: cfg.quad_idx as u8,
            cap,
            base_g_loop,
            base_damp_pole_zoh: self.damp_pole,
            base_lfo_rate_hz,
            base_ramp_rate_hz,
            base_freeze_mix: freeze_q,
            base_psi,
            base_scat_psi,
            base_window_depth,
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
        wire: &TransportQuadWire,
        cfg: &QuadHydrateConfig,
        ext_ring: Option<&mut TransportExtRing<'_>>,
        inline_q0_ring: Option<&mut [[u16; 4]; TRANSPORT_INLINE_QUAD0_CAP]>,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::const_abs_f32;
    use dnam_kernels::TANK_MAX_SIGMA_SAT_FP16;
    use zerocopy::FromZeros;
    use zerocopy::byteorder::little_endian::{F32, U16};

    /// verify[dsp.zoh.op-t-delays]
    #[test]
    fn test_wire_hydration_sigma_sat_freeze_mix_and_zoh_bounds() {
        let mut w0 = TransportQuadWire::new_zeroed();
        w0.delay_samples = [53.0, 67.0, 79.0, 89.0].map(F32::new);
        (w0.g_loop, w0.r_dc, w0.damp_pole) = (F32::new(0.85), F32::new(0.5), F32::new(0.2));
        w0.sigma_knee_fp16 = U16::new(0x7E00); // NaN fallback -> TANK_MAX_SIGMA_SAT_FP16 (0x3BE0).
        w0.freeze_mix_fp16 = U16::new(0x7E00); // NaN fallback -> 0.0.

        let cfg0 = QuadHydrateConfig {
            quad_idx: 0,
            fs_vm: 48_000.0,
            rho_vm: 1.0,
            declared_max_delay_48k: 89,
            ext_ring_pow2_frames: 7,
            dc_blocker_pole_5hz: DC_BLOCKER_5HZ_48K_F32,
        };
        let (q0, meta0) = TransportQuadHotCompact::from_wire(&w0, &cfg0).unwrap();
        assert_eq!(
            (
                q0.sigma_sat_fp16,
                q0.freeze_mix_fp16,
                meta0.base_freeze_mix,
                q0.r_dc,
                q0.flags & QUAD_FLAG_HAS_EXT,
            ),
            (TANK_MAX_SIGMA_SAT_FP16, 0x0000, 0.0, 0.95, 0)
        );
        let inv_23_err =
            const_abs_f32(fp16_to_f32(q0.inv_sigma_23_fp16) - 2.0 / (3.0 * TANK_MAX_SIGMA_SAT_F32));
        assert!(inv_23_err < 1.0e-3);

        w0.sigma_knee_fp16 = U16::new(f32_to_fp16(1.0e-5)); // Clamps up to 1e-3.
        w0.freeze_mix_fp16 = U16::new(f32_to_fp16(0.5));
        let (mut q_clamped, meta_clamped) = TransportQuadHotCompact::from_wire(&w0, &cfg0).unwrap();
        assert!(const_abs_f32(fp16_to_f32(q_clamped.sigma_sat_fp16) - 1.0e-3) < 1.0e-5);
        assert!(const_abs_f32(fp16_to_f32(q_clamped.inv_sigma_23_fp16) - 2.0 / 3.0e-3) < 1.0);
        assert_eq!(meta_clamped.base_freeze_mix, 0.5);

        q_clamped.set_host_write_lock(true, None, 0.5);
        let meta_reconf = q_clamped
            .reconfigure_sample_rate(
                &w0,
                &QuadHydrateConfig {
                    fs_vm: 96_000.0,
                    rho_vm: 0.5,
                    ..cfg0
                },
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(meta_reconf.cap, 256);
        assert_ne!(q_clamped.flags & QUAD_FLAG_HOST_WRITE_LOCK, 0);
        assert_eq!(
            TransportQuadHotCompact::from_wire(&w0, &QuadHydrateConfig { fs_vm: 0.0, ..cfg0 },)
                .unwrap_err(),
            VmError::InvalidSampleRate
        );
    }
}
