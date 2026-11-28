//! Compile-time layout invariants, unrotated `CP1` centered basis (`GraUnrotatedBasis`),
//! and bilinear dynamic/inertia gate weights (`GraDynGateWeights`) for `ControlBus v4 (GRA)`
//! (`ref[dsp.ctrl.gra.rational-and-entmax]`, `ref[dsp.ctrl.ledger.memory-compute]`,
//! `ref[repo.code.constant-eval-and-derivations]`).

use dnam_vm::control::{ControlClampDescriptor, STAGE_TARGET_SLOTS};

/// Maximum unrotated `GRA` hidden dimension `D` (`8` Lite–Pro, `16` Ultra).
pub const GRA_MAX_D: usize = 16;
/// Maximum `SwiGLU` hidden width `H` (`8` Lite–Pro, `16` Ultra).
pub const GRA_MAX_H: usize = 16;
/// Maximum anchored `1.5-Entmax` prototype count `M` (`8` Lite–Pro, `16` Ultra).
pub const GRA_MAX_M: usize = 16;
/// Maximum `Padé-[1/1]` denominator group count `R` (`4` Lite–Pro, `8` Ultra).
pub const GRA_MAX_R_PADE: usize = 8;
/// Maximum unrotated `CP1` centered basis dimension `P_xi`.
pub const GRA_MAX_XI: usize = 32;
/// Number of `A.4` block-pair scalar routing gates `g_edge[0..6]`.
pub const GRA_NUM_EDGES: usize = 6;
/// Number of `A.4` per-quad `Op-T` modulation slots `op_t_q[0..16]`.
pub const GRA_NUM_OP_T_MODS: usize = 16;
/// Number of `z_stage[0..8]` latent coordinates.
pub const GRA_Z_STAGE_DIM: usize = 8;
/// Number of unrotated `z_head[4..8]` latent coordinates (`z_head[0..4]` stays `SO(4)`-isolated).
pub const GRA_Z_HEAD_UNROTATED_DIM: usize = 4;
/// Total `z_head[0..8]` latent coordinates across `SO(4)` and `GRA`.
pub const GRA_Z_HEAD_DIM: usize = 8;
/// Total `GRA`-emitted delta vector length:
/// `stage_targets(32) + slot31_r(1) + z_stage(8) + z_head[4..8](4) + g_edge(6) + op_t_q(16) = 67`.
pub const GRA_OUT_DELTA_DIM: usize = STAGE_TARGET_SLOTS
    + 1
    + GRA_Z_STAGE_DIM
    + GRA_Z_HEAD_UNROTATED_DIM
    + GRA_NUM_EDGES
    + GRA_NUM_OP_T_MODS;

/// Lower clamp on `Padé-[1/1]` denominator offset `d_r`, guaranteeing `1 + d_r >= 0.25`.
pub const PADE_DEN_MIN: f64 = -0.75;
/// Upper clamp on `Padé-[1/1]` denominator offset `d_r`.
pub const PADE_DEN_MAX: f64 = 7.0;
/// Minimum 4-D outward trust-region radius (`ref[dsp.ctrl.op-n.trust-clip]`).
pub const TRUST_RAD_MIN: f64 = 0.01;
/// Maximum 4-D outward trust-region radius.
pub const TRUST_RAD_MAX: f64 = 16.0;
/// Default 4-D outward trust-region radius.
pub const TRUST_RAD_DEFAULT: f64 = 2.5;
/// Canonical 4-frame-aligned `44.1 / 48 kHz` control stride (`K_ctrl = 16`).
pub const K_CTRL_44K1_48K: u32 = 16;
/// Canonical control micro-ramp frame stride (`4` frames).
pub const MICRO_RAMP_FRAMES: u32 = 4;

/// Reclaimed legacy `cp1_base` bytes (`576 B`).
pub const RECLAIMED_CP1_BASE_BYTES: usize = 576;
/// Reclaimed legacy `u_int_base[:,4..8]` bytes (`64 B`).
pub const RECLAIMED_U_INT_BASE_TAIL_BYTES: usize = 64;
/// Reclaimed legacy `"CEXT"` dispatch bytes (`1,280 B`).
pub const RECLAIMED_CEXT_BYTES: usize = 1_280;
/// Reclaimed legacy `u_int_ultra[1..3]` bytes (`256 B`).
pub const RECLAIMED_U_INT_ULTRA_TAIL_BYTES: usize = 256;
/// Reclaimed legacy `cp1_ultra` bytes (`2,112 B`).
pub const RECLAIMED_CP1_ULTRA_BYTES: usize = 2_112;
/// Reclaimed legacy `stage_latent_core_mods_fp16` bytes (`2,048 B`).
pub const RECLAIMED_STAGE_LATENT_CORE_MODS_BYTES: usize = 2_048;
/// Total non-Tardos, non-`SO(4)` reclaimed control-plane budget (`6,336 B`).
pub const GRA_RECLAIMED_TOTAL_BYTES: usize = RECLAIMED_CP1_BASE_BYTES
    + RECLAIMED_U_INT_BASE_TAIL_BYTES
    + RECLAIMED_CEXT_BYTES
    + RECLAIMED_U_INT_ULTRA_TAIL_BYTES
    + RECLAIMED_CP1_ULTRA_BYTES
    + RECLAIMED_STAGE_LATENT_CORE_MODS_BYTES;
/// Maximum total `ControlBus` byte budget (`9,280 B`).
pub const CONTROL_BUS_MAX_TOTAL_BYTES: usize = 9_280;

/// Compile-time Euclidean GCD helper for alignment assertions.
pub const fn const_gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Tier profile selecting `GRA` hidden width `D = H`, prototype count `M`, and Padé groups `R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraTierProfile {
    /// Lite (`D = 8, H = 8, M = 8, R = 4`, Arena 0 self-contained).
    Lite,
    /// Full (`D = 8, H = 8, M = 8, R = 4`, Arena 0 self-contained).
    Full,
    /// Pro (`D = 8, H = 8, M = 8, R = 4`, Arena 0 self-contained).
    Pro,
    /// Ultra (`D = 16, H = 16, M = 16, R = 8`, Arena 0 + Arena 3 delta).
    Ultra,
}

impl GraTierProfile {
    /// Returns `(d, h, m, r_pade)` dimensions for this profile.
    #[inline]
    #[must_use]
    pub const fn dims(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Lite | Self::Full | Self::Pro => (8, 8, 8, 4),
            Self::Ultra => (16, 16, 16, 8),
        }
    }

    /// Maps prototype index `m` to its Padé denominator group `r(m) in 0..R`.
    #[inline]
    #[must_use]
    pub const fn pade_group(self, m: usize) -> usize {
        let (_, _, _, r_pade) = self.dims();
        m % r_pade
    }
}

/// Wire-active dynamic-gate target count (`stage_targets[0..32] + slot31_r + z_stage[0..8] = 41`).
pub const GRA_DYN_WIRE_SLOTS: usize = STAGE_TARGET_SLOTS + 1 + GRA_Z_STAGE_DIM;

/// Computes `fp16` tensor wire footprint in bytes for a `GraTierProfile` and input basis width.
#[inline]
#[must_use]
pub const fn gra_wire_bytes(profile: GraTierProfile, xi_dim: usize) -> usize {
    let (d, h, m, r_pade) = profile.dims();
    let w_in_elems = d * xi_dim;
    let swiglu_elems = 2 * h * d + d * h;
    let pade_elems = r_pade * d + m * d;
    let proto_elems = (m - 1) * GRA_OUT_DELTA_DIM;
    let dyn_elems = 8 * (4 + d) + GRA_DYN_WIRE_SLOTS * 8;
    (w_in_elems + swiglu_elems + pade_elems + proto_elems + dyn_elems) * 2
}

const _: () = assert!(GRA_OUT_DELTA_DIM == 67);
const _: () = assert!(GRA_RECLAIMED_TOTAL_BYTES == 6_336);
const _: () = assert!(GRA_RECLAIMED_TOTAL_BYTES <= CONTROL_BUS_MAX_TOTAL_BYTES);
const _: () = assert!(const_gcd_u32(K_CTRL_44K1_48K, MICRO_RAMP_FRAMES) == 4);
const _: () = assert!(PADE_DEN_MIN == -0.75 && 1.0 + PADE_DEN_MIN >= 0.25);
const _: () = assert!(PADE_DEN_MAX == 7.0);
const _: () = assert!(TRUST_RAD_MIN == 0.01 && TRUST_RAD_MAX == 16.0);
const _: () = assert!(gra_wire_bytes(GraTierProfile::Lite, 16) <= 3_072);
const _: () = assert!(gra_wire_bytes(GraTierProfile::Ultra, 24) <= GRA_RECLAIMED_TOTAL_BYTES);

/// Canonicalizes signed `-0.0` to `+0.0` in `binary64`.
#[inline(always)]
#[must_use]
pub fn canonicalize_zero_f64(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

/// Canonicalizes signed `-0.0` to `+0.0` in `binary32`.
#[inline(always)]
#[must_use]
pub fn canonicalize_zero_f32(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

/// Unrotated `CP1` centered basis vector `xi` (`xi === +0.0` at `StarCenter`).
#[derive(Clone, Debug, PartialEq)]
pub struct GraUnrotatedBasis {
    /// Centered basis elements `[0..len]` (`0.0` above `len`).
    pub xi: [f64; GRA_MAX_XI],
    /// Active basis dimensionality (`<= GRA_MAX_XI`).
    pub len: usize,
}

impl Default for GraUnrotatedBasis {
    fn default() -> Self {
        Self {
            xi: [0.0; GRA_MAX_XI],
            len: 0,
        }
    }
}

impl GraUnrotatedBasis {
    /// Builds `xi` from `ControlClampDescriptor`s (`eval_lobatto_basis` degree <= 2 for continuous
    /// controls + centered switch indicators `1_{opt != def && opt == aux}`), yielding exact
    /// `+0.0` across every lane at `StarCenter`.
    #[must_use]
    pub fn from_descriptors(
        descriptors: &[ControlClampDescriptor],
        u_norm: &[f32],
        active_switch_opts: &[u8],
    ) -> Self {
        let mut basis = Self::default();
        for (idx, desc) in descriptors.iter().enumerate() {
            if basis.len >= GRA_MAX_XI {
                break;
            }
            if desc.ctrl_kind == 0 {
                let u = u_norm
                    .get(idx)
                    .copied()
                    .unwrap_or(desc.anchor_nodes_norm[0]);
                let lob = desc.eval_lobatto_basis(u);
                let deg = usize::from(desc.lobatto_degree).min(2);
                for &val in lob.iter().take(deg) {
                    if basis.len >= GRA_MAX_XI {
                        break;
                    }
                    basis.xi[basis.len] = canonicalize_zero_f64(f64::from(val));
                    basis.len += 1;
                }
            } else if desc.ctrl_kind == 1 {
                let def_opt = desc.default_switch_option();
                let cur_opt = active_switch_opts.get(idx).copied().unwrap_or(def_opt);
                let max_opt = usize::from(desc.lobatto_degree).min(15);
                for opt in 0..=max_opt {
                    if opt as u8 == def_opt || (desc.active_options_mask & (1_u16 << opt)) == 0 {
                        continue;
                    }
                    if basis.len >= GRA_MAX_XI {
                        break;
                    }
                    let indicator = if cur_opt != def_opt && usize::from(cur_opt) == opt {
                        1.0
                    } else {
                        0.0
                    };
                    basis.xi[basis.len] = indicator;
                    basis.len += 1;
                }
            }
        }
        basis
    }

    /// Builds `xi` directly from pre-centered `f64` values, canonicalizing `-0.0 -> +0.0`.
    #[must_use]
    pub fn from_centered_slice(values: &[f64]) -> Self {
        let mut basis = Self::default();
        let len = values.len().min(GRA_MAX_XI);
        for (i, &v) in values.iter().take(len).enumerate() {
            basis.xi[i] = canonicalize_zero_f64(if v.is_finite() { v } else { 0.0 });
        }
        basis.len = len;
        basis
    }
}

/// Bilinear dynamic/inertia gate parameters coupling `Knob x q2` (with asymmetric `alpha_up` /
/// `alpha_dn` LFO rate slew) and `Knob x q3` (`Padé` envelope gate).
#[derive(Clone, Debug, PartialEq)]
pub struct GraDynGateWeights {
    /// Static-agnostic LFO baseline weights (`0.0` preserves `StarCenter` with live LFO).
    pub w_lfo_base: [f64; 4],
    /// Knob-to-LFO bilinear coupling weights `[4][D]` multiplying `h_1`.
    pub w_knob_lfo: [[f64; GRA_MAX_D]; 4],
    /// Knob-to-LFO rate modulation weights `[D]` multiplying `h_1`.
    pub lfo_rate_knob: [f64; GRA_MAX_D],
    /// Upward LFO rate inertia slew coefficient (`(0, 1]`).
    pub alpha_up: f64,
    /// Downward LFO rate inertia slew coefficient (`(0, 1]`).
    pub alpha_dn: f64,
    /// Envelope baseline weights (`0.0` preserves `StarCenter` with nonzero `q3`).
    pub w_env_base: [f64; 4],
    /// Knob-to-envelope bilinear coupling weights `[4][D]` multiplying `h_1`.
    pub w_knob_env: [[f64; GRA_MAX_D]; 4],
    /// Non-negative Padé denominator damping `[4]` (`1 + d_env * |q3| >= 1`).
    pub env_pade_den: [f64; 4],
    /// Projection from `[q2_eff(4), q3_eff(4)]` onto all `67` `GRA` output slots (`slot 31 L/R` active).
    pub dyn_to_out: [[f64; 8]; GRA_OUT_DELTA_DIM],
}

impl Default for GraDynGateWeights {
    fn default() -> Self {
        Self {
            w_lfo_base: [0.0; 4],
            w_knob_lfo: [[0.0; GRA_MAX_D]; 4],
            lfo_rate_knob: [0.0; GRA_MAX_D],
            alpha_up: 0.5,
            alpha_dn: 0.25,
            w_env_base: [0.0; 4],
            w_knob_env: [[0.0; GRA_MAX_D]; 4],
            env_pade_den: [0.0; 4],
            dyn_to_out: [[0.0; 8]; GRA_OUT_DELTA_DIM],
        }
    }
}

/// Complete trainable `ControlBus v4 (GRA)` parameter set (`0-ULP StarCenter` anchored).
#[derive(Clone, Debug, PartialEq)]
pub struct GraParams {
    /// Tier profile (`Lite`, `Full`, `Pro`, `Ultra`).
    pub profile: GraTierProfile,
    /// Active unrotated basis dimension `P_xi <= GRA_MAX_XI`.
    pub xi_dim: usize,
    /// Outward trust-region radius `R_trust in [0.01, 16.0]`.
    pub trust_radius: f64,
    /// `StarCenter` anchor score margin (`>= 1.0`; `s_tilde[0] = anchor_margin + 0.5 * y_rat[0]`).
    pub anchor_margin: f64,
    /// Stage 1 unrotated projection `W_in [D][P_xi]`.
    pub w_in: [[f64; GRA_MAX_XI]; GRA_MAX_D],
    /// Stage 2 `SwiGLU` gate weights `W_g [H][D]`.
    pub w_g: [[f64; GRA_MAX_D]; GRA_MAX_H],
    /// Stage 2 `SwiGLU` value weights `W_v [H][D]`.
    pub w_v: [[f64; GRA_MAX_D]; GRA_MAX_H],
    /// Stage 2 `SwiGLU` residual output weights `W_o [D][H]`.
    pub w_o: [[f64; GRA_MAX_H]; GRA_MAX_D],
    /// Stage 2 `Padé-[1/1]` denominator weights `W_den [R][D]`.
    pub w_den: [[f64; GRA_MAX_D]; GRA_MAX_R_PADE],
    /// Stage 2 `Padé-[1/1]` numerator weights `W_num [M][D]`.
    pub w_num: [[f64; GRA_MAX_D]; GRA_MAX_M],
    /// Stage 3 anchored `1.5-Entmax` prototype delta table `V_proto [M][67]` (`V_proto[0] == 0`).
    pub v_proto: [[f64; GRA_OUT_DELTA_DIM]; GRA_MAX_M],
    /// Optional direct rational delta readout `W_rat_direct [67][M]` (also `0` at `StarCenter`).
    pub w_rat_direct: [[f64; GRA_MAX_M]; GRA_OUT_DELTA_DIM],
    /// Stage 4 bilinear dynamic/inertia modulator gate weights.
    pub dyn_gate: GraDynGateWeights,
}

impl GraParams {
    /// Creates a zero-initialized `GraParams` for `profile` and input basis width `xi_dim`.
    #[must_use]
    pub fn zeros(profile: GraTierProfile, xi_dim: usize) -> Self {
        Self {
            profile,
            xi_dim: xi_dim.min(GRA_MAX_XI),
            trust_radius: TRUST_RAD_DEFAULT,
            anchor_margin: 1.0,
            w_in: [[0.0; GRA_MAX_XI]; GRA_MAX_D],
            w_g: [[0.0; GRA_MAX_D]; GRA_MAX_H],
            w_v: [[0.0; GRA_MAX_D]; GRA_MAX_H],
            w_o: [[0.0; GRA_MAX_H]; GRA_MAX_D],
            w_den: [[0.0; GRA_MAX_D]; GRA_MAX_R_PADE],
            w_num: [[0.0; GRA_MAX_D]; GRA_MAX_M],
            v_proto: [[0.0; GRA_OUT_DELTA_DIM]; GRA_MAX_M],
            w_rat_direct: [[0.0; GRA_MAX_M]; GRA_OUT_DELTA_DIM],
            dyn_gate: GraDynGateWeights::default(),
        }
    }

    /// Returns sanitized `(trust_radius, anchor_margin)` clamped to canonical spec intervals.
    #[inline]
    #[must_use]
    pub fn sanitized_scalars(&self) -> (f64, f64) {
        let r = if self.trust_radius.is_finite() && self.trust_radius > 0.0 {
            self.trust_radius.clamp(TRUST_RAD_MIN, TRUST_RAD_MAX)
        } else {
            1.0
        };
        let margin = if self.anchor_margin.is_finite() {
            self.anchor_margin.max(1.0)
        } else {
            1.0
        };
        (r, margin)
    }

    /// Enforces `V_proto[0] == 0.0`, non-negative `env_pade_den`, and valid inertia slews.
    pub fn enforce_structural_invariants(&mut self) {
        self.xi_dim = self.xi_dim.min(GRA_MAX_XI);
        let (r, m) = self.sanitized_scalars();
        self.trust_radius = r;
        self.anchor_margin = m;
        self.v_proto[0] = [0.0; GRA_OUT_DELTA_DIM];
        self.dyn_gate.alpha_up = if self.dyn_gate.alpha_up.is_finite() {
            self.dyn_gate.alpha_up.clamp(1.0e-5, 1.0)
        } else {
            0.5
        };
        self.dyn_gate.alpha_dn = if self.dyn_gate.alpha_dn.is_finite() {
            self.dyn_gate.alpha_dn.clamp(1.0e-5, 1.0)
        } else {
            0.25
        };
        for d in &mut self.dyn_gate.env_pade_den {
            *d = if d.is_finite() { d.max(0.0) } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_vm::control::ControlClampDescriptor;

    #[test]
    fn test_gra_unrotated_basis_zero_at_star_center_and_wire_budgets() {
        let range_desc = ControlClampDescriptor::range(0.0, 10.0, 5.0, 2, [0.0, -0.5, 0.5]);
        let switch_desc = ControlClampDescriptor::switch(3, 0b111, 0);

        let basis_star =
            GraUnrotatedBasis::from_descriptors(&[range_desc, switch_desc], &[0.0], &[0, 0]);
        assert_eq!(basis_star.len, 4);
        for i in 0..basis_star.len {
            assert_eq!(basis_star.xi[i].to_bits(), 0);
        }

        let basis_moved =
            GraUnrotatedBasis::from_descriptors(&[range_desc, switch_desc], &[0.5], &[0, 2]);
        assert!((basis_moved.xi[0] - 0.5).abs() < 1.0e-6);
        assert!((basis_moved.xi[1] - 0.5).abs() < 1.0e-6);
        assert_eq!(basis_moved.xi[2], 0.0);
        assert_eq!(basis_moved.xi[3], 1.0);

        assert!(gra_wire_bytes(GraTierProfile::Lite, 16) <= 3_072);
        assert!(gra_wire_bytes(GraTierProfile::Ultra, 24) <= GRA_RECLAIMED_TOTAL_BYTES);
    }
}
