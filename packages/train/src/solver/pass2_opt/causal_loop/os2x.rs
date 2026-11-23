//! Per-frame `2x` oversampling and reverse-time adjoint carry helpers for `CausalLoop`
//! matching [`crate::solver::model::cascade`] and `dnam-vm` (`Full..Ultra` at `2x`,
//! `Op-N[Pre]` and `FIR + Kautz + Op-T` at `1x`).

use super::super::super::model::cascade::sanitize_unit_f64;

/// Stateful `1x -> 2x` quadratic midpoint upsampler (`SonaOversamplerParams` poly3 mode).
#[derive(Clone, Copy, Debug)]
pub(super) struct Os2xUpsampler {
    up_coeff: f64,
    prev1: [f64; 2],
    prev2: [f64; 2],
}

impl Os2xUpsampler {
    #[inline]
    pub(super) fn new(up_quad_coeff: f64) -> Self {
        Self {
            up_coeff: sanitize_unit_f64(up_quad_coeff, -0.20, 0.05, -0.125),
            prev1: [0.0; 2],
            prev2: [0.0; 2],
        }
    }

    #[inline]
    pub(super) fn step(&mut self, x_l: f64, x_r: f64) -> ([f64; 2], [f64; 2]) {
        let dl = x_l - 2.0 * self.prev1[0] + self.prev2[0];
        let dr = x_r - 2.0 * self.prev1[1] + self.prev2[1];
        let sub0_l = 0.5 * x_l + 0.5 * self.prev1[0] + self.up_coeff * dl;
        let sub0_r = 0.5 * x_r + 0.5 * self.prev1[1] + self.up_coeff * dr;
        self.prev2 = self.prev1;
        self.prev1 = [x_l, x_r];
        ([sub0_l, sub0_r], [x_l, x_r])
    }
}

/// Stateful `2x -> 1x` 4-channel polyphase decimator (`[pre_l, pre_r, tank_l, tank_r]`).
#[derive(Clone, Copy, Debug)]
pub(super) struct Os2xDecimator {
    c_p1: f64,
    c0: f64,
    c1: f64,
    prev1: [f64; 4],
}

impl Os2xDecimator {
    #[inline]
    pub(super) fn new(dec_c_prev1: f64, dec_c0: f64) -> Self {
        let c_p1 = sanitize_unit_f64(dec_c_prev1, 0.0, 0.28, 0.15);
        let c0 = sanitize_unit_f64(dec_c0, 0.0, 0.55, 0.50);
        Self {
            c_p1,
            c0,
            c1: 1.0 - c_p1 - c0,
            prev1: [0.0; 4],
        }
    }

    #[inline]
    pub(super) fn step(&mut self, sub0: [f64; 4], sub1: [f64; 4]) -> [f64; 4] {
        let out = core::array::from_fn(|k| {
            self.c_p1 * self.prev1[k] + self.c0 * sub0[k] + self.c1 * sub1[k]
        });
        self.prev1 = sub1;
        out
    }
}

/// Exact reverse-time carry state pulling back `1x` readout/decimation sensitivities through
/// `Os2xDecimator` -> `sub{1,0}` and `Os2xUpsampler` -> `1x` `Op-N[Pre]`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Os2xReverseState {
    up_coeff: f64,
    c_p1: f64,
    c0: f64,
    c1: f64,
    dec_carry: [f64; 4],
    up_carry_p1: [f64; 2],
    up_carry_p2: [f64; 2],
}

impl Os2xReverseState {
    #[inline]
    pub(super) fn new(up_quad_coeff: f64, dec_c_prev1: f64, dec_c0: f64) -> Self {
        let c_p1 = sanitize_unit_f64(dec_c_prev1, 0.0, 0.28, 0.15);
        let c0 = sanitize_unit_f64(dec_c0, 0.0, 0.55, 0.50);
        Self {
            up_coeff: sanitize_unit_f64(up_quad_coeff, -0.20, 0.05, -0.125),
            c_p1,
            c0,
            c1: 1.0 - c_p1 - c0,
            dec_carry: [0.0; 4],
            up_carry_p1: [0.0; 2],
            up_carry_p2: [0.0; 2],
        }
    }

    /// Maps `1x` `[g_pre_l, g_pre_r, g_tank_l, g_tank_r]` to `(g_sub1, g_sub0)` at `2x`.
    #[inline]
    pub(super) fn split_decimator(&mut self, g_1x: [f64; 4]) -> ([f64; 4], [f64; 4]) {
        let g_sub1 = core::array::from_fn(|k| self.c1 * g_1x[k] + self.dec_carry[k]);
        let g_sub0 = core::array::from_fn(|k| self.c0 * g_1x[k]);
        self.dec_carry = core::array::from_fn(|k| self.c_p1 * g_1x[k]);
        (g_sub1, g_sub0)
    }

    /// Maps `2x` core-input sensitivities `(g_sub0, g_sub1)` back into `1x` `[g_ux_l, g_ux_r]`.
    #[inline]
    pub(super) fn pullback_upsampler(&mut self, g_sub0: [f64; 2], g_sub1: [f64; 2]) -> [f64; 2] {
        let w0 = 0.5 + self.up_coeff;
        let w1 = 0.5 - 2.0 * self.up_coeff;
        let w2 = self.up_coeff;
        let mut out = [0.0; 2];
        for c in 0..2 {
            out[c] = g_sub1[c] + w0 * g_sub0[c] + self.up_carry_p1[c];
            self.up_carry_p1[c] = w1 * g_sub0[c] + self.up_carry_p2[c];
            self.up_carry_p2[c] = w2 * g_sub0[c];
        }
        out
    }
}
