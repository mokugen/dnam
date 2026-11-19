//! Cold & ZOH-rescaled Stage IV multi-quad `Op-T` tank parameters (`stage4_tank/v3_fdn/params.rs`).

use crate::container::ZohRescaledParams;
use crate::math::{STAGE4_DC_BLOCKER_48K_F32, newton6_reciprocal_f64};
use crate::stages::topology::STAGE4_MAX_G_LIN_BITS;
use crate::types::Tier;

/// Number of delay lines per Stage IV `Op-T` quad (`4` lines).
pub const V3_TANK_LINES: usize = 4;
/// Maximum number of Stage IV `Op-T` quads across `Lite..Ultra` (`4` quads = `16` lines).
pub const SONA_TANK_QUADS: usize = 4;
/// Maximum number of Stage IV FDN delay lines across all `4` quads (`16`).
pub const SONA_TANK_MAX_LINES: usize = V3_TANK_LINES * SONA_TANK_QUADS;

/// Circular ring buffer capacity per inline `Quad 0` frame-major L1D ring (`128` frames = `1,024 B` = `16 CL`).
pub const V3_TANK_CAP: usize = 128;
/// Maximum declared `Quad 0` inline delay length (`120` frames) reserving LFO/Catmull-Rom headroom inside `128` frames.
/// Declared delays `> 120` spill cleanly to cold `ext_ring`.
pub const V3_TANK_INLINE_MAX_DELAY: usize = 120;
/// Legacy stride alias (`132`) exported for wire/compatibility metadata.
pub const V3_TANK_RING_STRIDE: usize = V3_TANK_CAP + 4;

/// Maximum model-declared Stage IV delay length in `48 kHz` reference samples (`1.0 s`).
pub const V3_TANK_MAX_DECLARED_DELAY: usize = 48_000;
/// Maximum Stage IV LFO excursion depth in host samples.
pub const V3_TANK_MAX_LFO_DEPTH: f32 = 512.0;
/// Maximum cold-allocated power-of-two ring capacity (`4x` bounce on `48,000`-sample delays).
pub const V3_TANK_MAX_EXT_CAP: usize = 262_144;
/// Maximum cold Schroeder diffuser delay per lane in host samples (`16,384`; never capped at `255`).
pub const STAGE4_SCHROEDER_MAX_DELAY: usize = 16_384;
/// Canonical `48 kHz` coprime Schroeder diffuser reference delays (`op_t_transport.md` §3.2).
pub const STAGE4_SCHROEDER_REF_DELAYS_48K: [usize; 8] = [19, 31, 47, 73, 103, 139, 179, 223];

const _: () = assert!(V3_TANK_CAP.is_power_of_two() && V3_TANK_MAX_EXT_CAP.is_power_of_two());
const _: () = assert!(V3_TANK_MAX_EXT_CAP >= 4 * V3_TANK_MAX_DECLARED_DELAY + 2_048);

/// Cold sample representation format for cold-allocated `ext_lines` Stage IV delay buffers.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stage4ExtFormat {
    /// IEEE-754 binary16 storage (`a = 2 B/sample`).
    #[default]
    Fp16 = 0,
    /// Native IEEE-754 binary32 storage (`a = 4 B/sample`).
    F32 = 1,
}

impl Stage4ExtFormat {
    /// Converts wire integer tag (`0 => Fp16`, `1 => F32`, default `Fp16`).
    #[inline(always)]
    #[must_use]
    pub const fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::F32,
            _ => Self::Fp16,
        }
    }

    /// Returns wire integer tag (`0` or `1`).
    #[inline(always)]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Cold parameters for one `4`-line Stage IV `Op-T` quad (`Quad 0..3`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage4QuadParams {
    pub delay_lengths: [usize; V3_TANK_LINES],
    pub tap_gains: [f32; V3_TANK_LINES],
    pub allpass_delays: [usize; V3_TANK_LINES],
    pub allpass_g: [f32; V3_TANK_LINES],
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
    pub lfo_rate_hz: f32,
    pub lfo_depth_samples: f32,
    pub lfo_stereo_phase: f32,
    pub ext_format: Stage4ExtFormat,
}

impl Default for Stage4QuadParams {
    fn default() -> Self {
        Self::identity()
    }
}

impl Stage4QuadParams {
    /// Resting identity parameters for `Quad 0` (`dry_mix = 1.0`, `wet_mix = 0.0`).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            delay_lengths: [67, 83, 97, 113],
            tap_gains: [0.5; V3_TANK_LINES],
            allpass_delays: [0; V3_TANK_LINES],
            allpass_g: [0.0; V3_TANK_LINES],
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
            lfo_rate_hz: 0.0,
            lfo_depth_samples: 0.0,
            lfo_stereo_phase: 0.0,
            ext_format: Stage4ExtFormat::Fp16,
        }
    }

    /// Silent zero-dry resting parameters for expansion `Quads 1..=3` (`dry_mix = 0.0, wet_mix = 0.0`).
    #[must_use]
    pub const fn silent_identity() -> Self {
        Self {
            dry_mix: 0.0,
            wet_mix: 0.0,
            ..Self::identity()
        }
    }

    /// Returns canonical prime delay lengths for quad index `q in 0..4`.
    #[must_use]
    pub const fn default_delays_for_quad(q: usize) -> [usize; V3_TANK_LINES] {
        match q {
            0 => [53, 67, 79, 89],
            1 => [149, 179, 211, 239],
            2 => [263, 293, 331, 367],
            _ => [389, 431, 479, 521],
        }
    }

    /// Returns resting quad defaults for `(tier, q)` matching `SONA` Matryoshka tiers.
    #[must_use]
    pub const fn default_for_quad(_tier: Tier, q: usize) -> Self {
        let base = if q == 0 {
            Self::identity()
        } else {
            Self::silent_identity()
        };
        Self {
            delay_lengths: Self::default_delays_for_quad(q),
            ..base
        }
    }

    fn clamp_common_fields(&mut self) {
        let max_g = f32::from_bits(STAGE4_MAX_G_LIN_BITS);
        self.g_loop = if self.g_loop.is_finite() {
            self.g_loop.clamp(0.0, max_g)
        } else {
            0.0
        };
        self.sigma_knee = if self.sigma_knee.is_finite() {
            self.sigma_knee.clamp(1.0e-3, 8.0)
        } else {
            1.0
        };
        self.base_psi = if self.base_psi.is_finite() {
            self.base_psi
                .clamp(-core::f32::consts::PI, core::f32::consts::PI)
        } else {
            core::f32::consts::FRAC_PI_4
        };
        self.inject_gain = if self.inject_gain.is_finite() {
            self.inject_gain.clamp(-2.0, 2.0)
        } else {
            0.0
        };
        self.dry_mix = if self.dry_mix.is_finite() {
            self.dry_mix.clamp(-2.0, 2.0)
        } else {
            1.0
        };
        self.wet_mix = if self.wet_mix.is_finite() {
            self.wet_mix.clamp(-2.0, 2.0)
        } else {
            0.0
        };
        for l in 0..V3_TANK_LINES {
            let g = self.tap_gains[l];
            self.tap_gains[l] = if g.is_finite() {
                g.clamp(-2.0, 2.0)
            } else {
                0.5
            };
            let ag = self.allpass_g[l];
            self.allpass_g[l] = if ag.is_finite() {
                ag.clamp(-0.75, 0.75)
            } else {
                0.0
            };
        }
        self.r_dc = if self.r_dc.is_finite() {
            self.r_dc.clamp(0.95, 0.9999)
        } else {
            STAGE4_DC_BLOCKER_48K_F32
        };
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

    /// Clamps loop gain `< 0.9990`, delay bounds (`4..=48,000`), Schroeder gains, and VCA params.
    pub fn clamp_in_place(&mut self) {
        self.clamp_common_fields();
        for l in 0..V3_TANK_LINES {
            self.delay_lengths[l] = self.delay_lengths[l].clamp(4, V3_TANK_MAX_DECLARED_DELAY);
            if self.allpass_delays[l] > 0 {
                self.allpass_delays[l] =
                    self.allpass_delays[l].clamp(1, STAGE4_SCHROEDER_MAX_DELAY);
            }
        }
        self.lfo_depth_samples = if self.lfo_depth_samples.is_finite() {
            self.lfo_depth_samples.clamp(0.0, V3_TANK_MAX_LFO_DEPTH)
        } else {
            0.0
        };
    }

    /// Rescales delays, LFO, DC pole, VCA, and Schroeder diffuser delays under continuous-time ZOH.
    #[must_use]
    pub fn with_zoh_rescaled(&self, zoh: &ZohRescaledParams, cap: usize, max_depth: f32) -> Self {
        let mut p = *self;
        p.clamp_in_place();
        if (zoh.rho_sr - 1.0).abs() >= 1.0e-12 {
            let inv_rho = newton6_reciprocal_f64(zoh.rho_sr.max(1.0e-6));
            for l in 0..V3_TANK_LINES {
                p.delay_lengths[l] = ((f64::from(p.delay_lengths[l] as u32) * inv_rho + 0.5)
                    as usize)
                    .clamp(4, cap - 1);
                if p.allpass_delays[l] > 0 {
                    p.allpass_delays[l] = ((f64::from(p.allpass_delays[l] as u32) * inv_rho + 0.5)
                        as usize)
                        .clamp(1, STAGE4_SCHROEDER_MAX_DELAY);
                }
            }
            p.lfo_depth_samples =
                ((f64::from(p.lfo_depth_samples) * inv_rho) as f32).clamp(0.0, max_depth);
            p.r_dc = zoh.rescale_real_pole(p.r_dc);
            p.vca_att = zoh.rescale_real_pole(p.vca_att);
            p.vca_rel = zoh.rescale_real_pole(p.vca_rel);
        }
        p.clamp_common_fields();
        clamp_quad_params_to_cap(&mut p, cap, max_depth);
        p
    }
}

/// Clamps per-quad runtime delay lengths and LFO excursion against an allocated ring capacity.
pub(crate) fn clamp_quad_params_to_cap(p: &mut Stage4QuadParams, cap: usize, max_lfo_depth: f32) {
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

/// Cold parameters for the Matryoshka `1..4`-quad Stage IV `Op-T` FDN tank (`Quad 0..3`).
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
    pub allpass_delays: [usize; V3_TANK_LINES],
    pub allpass_g: [f32; V3_TANK_LINES],
    pub ext_format: Stage4ExtFormat,
    pub active_quads: usize,
    pub quads: [Stage4QuadParams; SONA_TANK_QUADS],
}

impl Default for V3Stage4TankParams {
    fn default() -> Self {
        Self::identity()
    }
}

impl V3Stage4TankParams {
    /// Resting identity Stage IV tank (`1` quad, `dry_mix = 1.0`, `wet_mix = 0.0`).
    #[must_use]
    pub const fn identity() -> Self {
        let q0 = Stage4QuadParams::identity();
        let qs = Stage4QuadParams::silent_identity();
        Self {
            delay_lengths: q0.delay_lengths,
            g_loop: q0.g_loop,
            sigma_knee: q0.sigma_knee,
            base_psi: q0.base_psi,
            inject_gain: q0.inject_gain,
            dry_mix: q0.dry_mix,
            wet_mix: q0.wet_mix,
            r_dc: q0.r_dc,
            vca_thresh: q0.vca_thresh,
            vca_kappa: q0.vca_kappa,
            vca_min_gain: q0.vca_min_gain,
            vca_att: q0.vca_att,
            vca_rel: q0.vca_rel,
            tap_gains: q0.tap_gains,
            lfo_rate_hz: q0.lfo_rate_hz,
            lfo_depth_samples: q0.lfo_depth_samples,
            lfo_stereo_phase: q0.lfo_stereo_phase,
            allpass_delays: q0.allpass_delays,
            allpass_g: q0.allpass_g,
            ext_format: q0.ext_format,
            active_quads: 1,
            quads: [q0, qs, qs, qs],
        }
    }

    /// Builds a resting identity multi-quad Stage IV tank configured for `tier`.
    #[must_use]
    pub const fn identity_for_tier(tier: Tier) -> Self {
        let q0 = Stage4QuadParams::default_for_quad(tier, 0);
        let q1 = Stage4QuadParams::default_for_quad(tier, 1);
        let q2 = Stage4QuadParams::default_for_quad(tier, 2);
        let q3 = Stage4QuadParams::default_for_quad(tier, 3);
        let mut base = Self::identity();
        base.delay_lengths = q0.delay_lengths;
        base.active_quads = tier.tank_groups();
        base.quads = [q0, q1, q2, q3];
        base
    }

    /// Returns default prime FDN delay lengths for `tier`'s outermost quad.
    #[must_use]
    pub const fn default_delays_for_tier(tier: Tier) -> [usize; V3_TANK_LINES] {
        Stage4QuadParams::default_delays_for_quad(tier as usize)
    }

    /// Active delay window bytes touched per cycle in L1D across `tier`'s active quads.
    #[must_use]
    pub const fn active_ring_bytes(tier: Tier) -> usize {
        let quads = tier.tank_groups();
        let mut total = 0usize;
        let mut q = 0usize;
        while q < quads {
            let d = Stage4QuadParams::default_delays_for_quad(q);
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
            total += (max_d + 1) * V3_TANK_LINES * 2;
            q += 1;
        }
        total
    }

    /// Returns top-level fields projected as a [`Stage4QuadParams`] (`Quad 0`).
    #[must_use]
    pub const fn top_level_quad(&self) -> Stage4QuadParams {
        Stage4QuadParams {
            delay_lengths: self.delay_lengths,
            tap_gains: self.tap_gains,
            allpass_delays: self.allpass_delays,
            allpass_g: self.allpass_g,
            g_loop: self.g_loop,
            sigma_knee: self.sigma_knee,
            base_psi: self.base_psi,
            inject_gain: self.inject_gain,
            dry_mix: self.dry_mix,
            wet_mix: self.wet_mix,
            r_dc: self.r_dc,
            vca_thresh: self.vca_thresh,
            vca_kappa: self.vca_kappa,
            vca_min_gain: self.vca_min_gain,
            vca_att: self.vca_att,
            vca_rel: self.vca_rel,
            lfo_rate_hz: self.lfo_rate_hz,
            lfo_depth_samples: self.lfo_depth_samples,
            lfo_stereo_phase: self.lfo_stereo_phase,
            ext_format: self.ext_format,
        }
    }

    /// Synchronizes top-level scalar fields from `self.quads[0]`.
    pub fn sync_top_from_quad0(&mut self) {
        let q0 = self.quads[0];
        self.delay_lengths = q0.delay_lengths;
        self.tap_gains = q0.tap_gains;
        self.allpass_delays = q0.allpass_delays;
        self.allpass_g = q0.allpass_g;
        self.g_loop = q0.g_loop;
        self.sigma_knee = q0.sigma_knee;
        self.base_psi = q0.base_psi;
        self.inject_gain = q0.inject_gain;
        self.dry_mix = q0.dry_mix;
        self.wet_mix = q0.wet_mix;
        self.r_dc = q0.r_dc;
        self.vca_thresh = q0.vca_thresh;
        self.vca_kappa = q0.vca_kappa;
        self.vca_min_gain = q0.vca_min_gain;
        self.vca_att = q0.vca_att;
        self.vca_rel = q0.vca_rel;
        self.lfo_rate_hz = q0.lfo_rate_hz;
        self.lfo_depth_samples = q0.lfo_depth_samples;
        self.lfo_stereo_phase = q0.lfo_stereo_phase;
        self.ext_format = q0.ext_format;
    }

    /// Synchronizes `self.quads[0]` from top-level scalar fields.
    pub fn sync_quad0_from_top(&mut self) {
        self.quads[0] = self.top_level_quad();
    }

    /// Returns effective [`Stage4QuadParams`] for quad `q` (`Quad 0` honors direct top-level edits).
    #[must_use]
    pub fn effective_quad_params(&self, q: usize) -> Stage4QuadParams {
        if q == 0 {
            let top = self.top_level_quad();
            let id = Stage4QuadParams::identity();
            if self.quads[0] == id || top != id {
                top
            } else {
                self.quads[0]
            }
        } else {
            self.quads[q.min(SONA_TANK_QUADS - 1)]
        }
    }

    /// Clamps all quad parameters (`0..4`) into passive BIBO-safe bounds.
    pub fn clamp_in_place(&mut self) {
        self.active_quads = self.active_quads.clamp(1, SONA_TANK_QUADS);
        let mut q0 = self.effective_quad_params(0);
        q0.clamp_in_place();
        self.quads[0] = q0;
        self.sync_top_from_quad0();
        for q in 1..SONA_TANK_QUADS {
            self.quads[q].clamp_in_place();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage4_quad_params_zoh_clamping_and_active_ring_bytes() {
        assert_eq!(Stage4ExtFormat::from_u32(0), Stage4ExtFormat::Fp16);
        assert_eq!(Stage4ExtFormat::from_u32(1), Stage4ExtFormat::F32);
        assert_eq!(Stage4ExtFormat::F32.as_u32(), 1);

        let mut p = V3Stage4TankParams::identity_for_tier(Tier::Ultra);
        assert_eq!(p.active_quads, 4);
        p.allpass_delays = [19, 223, 0, 16_384];
        p.allpass_g = [0.99, -0.99, 0.4, f32::NAN];
        p.clamp_in_place();
        assert_eq!(p.allpass_g, [0.75, -0.75, 0.4, 0.0]);
        assert!(V3Stage4TankParams::active_ring_bytes(Tier::Lite) == 720);
        assert!(
            V3Stage4TankParams::active_ring_bytes(Tier::Ultra)
                > V3Stage4TankParams::active_ring_bytes(Tier::Lite)
        );
    }
}
