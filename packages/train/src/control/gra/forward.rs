//! Forward `ControlBus v4 (GRA)` evaluator (`f64` tape + `f32` runtime-mirror using
//! `dnam_vm::math::{trust_clip_outward_4d, recip_nr3_pos_f32, rsqrt_nr3_f32}`) guaranteeing
//! `0-ULP StarCenter` parity across `Lite..Ultra` (`ref[dsp.ctrl.gra.rational-and-entmax]`).

use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::{recip_nr3_pos_f32, sanitize_finite_f32, trust_clip_outward_4d};

use super::entmax::{Entmax15Output, entmax15_forward_f32_mirror, entmax15_forward_f64};
use super::model::{
    GRA_MAX_D, GRA_MAX_H, GRA_MAX_M, GRA_MAX_R_PADE, GRA_MAX_XI, GRA_NUM_EDGES, GRA_NUM_OP_T_MODS,
    GRA_OUT_DELTA_DIM, GRA_Z_HEAD_UNROTATED_DIM, GRA_Z_STAGE_DIM, GraParams, PADE_DEN_MAX,
    PADE_DEN_MIN, canonicalize_zero_f32, canonicalize_zero_f64,
};

const TRUST_CLIP_EPS_F64: f64 = f32::MIN_POSITIVE as f64;

/// Cubic `C^1` smoothstep gate `sigma_poly(g)` (`sigma_poly(0) = 0.5`,
/// `sigma_poly(g) + sigma_poly(-g) == 1.0`, `0 FDIV`), returning `(sigma, dsigma/dg)`.
#[inline]
#[must_use]
pub fn sigma_poly_f64(g: f64) -> (f64, f64) {
    let t_raw = 0.5 + 0.125 * g;
    let t = t_raw.clamp(0.0, 1.0);
    let sigma = t * t * (3.0 - 2.0 * t);
    let dsigma = if t_raw > 0.0 && t_raw < 1.0 {
        0.75 * t * (1.0 - t)
    } else {
        0.0
    };
    (sigma, dsigma)
}

/// `binary32` runtime-mirror of [`sigma_poly_f64`].
#[inline(always)]
#[must_use]
pub fn sigma_poly_f32(g: f32) -> f32 {
    let t = (0.5_f32 + 0.125_f32 * g).clamp(0.0_f32, 1.0_f32);
    t * t * (3.0_f32 - 2.0_f32 * t)
}

/// Smooth 4-D trust-region projection in `f64`, returning `(clipped, was_clipped, scale)`.
#[inline]
#[must_use]
pub fn trust_clip_4d_f64(u: [f64; 4], radius: f64) -> ([f64; 4], bool, f64) {
    if !radius.is_finite() || radius <= 0.0 || !u.iter().all(|v| v.is_finite()) {
        return ([0.0; 4], true, 0.0);
    }
    let norm2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
    let norm_eps = (norm2 + TRUST_CLIP_EPS_F64).sqrt();
    if norm_eps <= radius {
        (u.map(canonicalize_zero_f64), false, 1.0)
    } else {
        let scale = radius / norm_eps;
        (
            [
                canonicalize_zero_f64(u[0] * scale),
                canonicalize_zero_f64(u[1] * scale),
                canonicalize_zero_f64(u[2] * scale),
                canonicalize_zero_f64(u[3] * scale),
            ],
            true,
            scale,
        )
    }
}

/// Differentiable `binary64` `ControlBus v4 (GRA)` output bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct GraOutputs {
    /// `stage_targets[0..32]` (`slot 31` holds `slot31_lr[0]`).
    pub stage_targets: [f64; STAGE_TARGET_SLOTS],
    /// Stereo differential quadrature AM output gains `[L, R]`.
    pub slot31_lr: [f64; 2],
    /// Unrotated `z_stage[0..8]` latent modulation vector.
    pub z_stage: [f64; GRA_Z_STAGE_DIM],
    /// Unrotated `z_head[4..8]` (`z_head[0..4]` remains `SO(4)`-isolated).
    pub z_head_4_8: [f64; GRA_Z_HEAD_UNROTATED_DIM],
    /// `A.4` 6-edge scalar block-pair routing gates `g_edge[0..6]`.
    pub g_edge: [f64; GRA_NUM_EDGES],
    /// `A.4` per-quad `Op-T` modulation slots `op_t_q[0..16]`.
    pub op_t_q: [f64; GRA_NUM_OP_T_MODS],
    /// Asymmetric-inertia-slewed LFO rate output.
    pub lfo_rate_next: f64,
}

impl Default for GraOutputs {
    fn default() -> Self {
        Self {
            stage_targets: [0.0; STAGE_TARGET_SLOTS],
            slot31_lr: [0.0; 2],
            z_stage: [0.0; GRA_Z_STAGE_DIM],
            z_head_4_8: [0.0; GRA_Z_HEAD_UNROTATED_DIM],
            g_edge: [0.0; GRA_NUM_EDGES],
            op_t_q: [0.0; GRA_NUM_OP_T_MODS],
            lfo_rate_next: 0.0,
        }
    }
}

impl GraOutputs {
    /// Packs the `67`-element `GRA` delta layout `[stage_targets(32), slot31_r(1), z_stage(8),
    /// z_head[4..8](4), g_edge(6), op_t_q(16)]` from flat array `raw`.
    #[must_use]
    pub fn from_flat(raw: &[f64; GRA_OUT_DELTA_DIM], lfo_rate_next: f64) -> Self {
        let mut out = Self::default();
        out.stage_targets.copy_from_slice(&raw[0..32]);
        out.slot31_lr = [raw[31], raw[32]];
        out.z_stage.copy_from_slice(&raw[33..41]);
        out.z_head_4_8.copy_from_slice(&raw[41..45]);
        out.g_edge.copy_from_slice(&raw[45..51]);
        out.op_t_q.copy_from_slice(&raw[51..67]);
        out.lfo_rate_next = canonicalize_zero_f64(lfo_rate_next);
        out
    }

    /// Flattens `self` back into the `67`-element canonical delta array.
    #[must_use]
    pub fn to_flat(&self) -> [f64; GRA_OUT_DELTA_DIM] {
        let mut raw = [0.0_f64; GRA_OUT_DELTA_DIM];
        raw[0..32].copy_from_slice(&self.stage_targets);
        raw[31] = self.slot31_lr[0];
        raw[32] = self.slot31_lr[1];
        raw[33..41].copy_from_slice(&self.z_stage);
        raw[41..45].copy_from_slice(&self.z_head_4_8);
        raw[45..51].copy_from_slice(&self.g_edge);
        raw[51..67].copy_from_slice(&self.op_t_q);
        raw
    }
}

/// `binary32` runtime-mirror `GRA` output bundle for `0-ULP StarCenter` bit-identity checks.
#[derive(Clone, Debug, PartialEq)]
pub struct GraOutputsF32 {
    /// `stage_targets[0..32]` (`slot 31` holds `slot31_lr[0]`).
    pub stage_targets: [f32; STAGE_TARGET_SLOTS],
    /// Stereo differential quadrature AM output gains `[L, R]`.
    pub slot31_lr: [f32; 2],
    /// Unrotated `z_stage[0..8]`.
    pub z_stage: [f32; GRA_Z_STAGE_DIM],
    /// Unrotated `z_head[4..8]`.
    pub z_head_4_8: [f32; GRA_Z_HEAD_UNROTATED_DIM],
    /// `A.4` 6-edge scalar routing gates `g_edge[0..6]`.
    pub g_edge: [f32; GRA_NUM_EDGES],
    /// `A.4` per-quad `Op-T` modulation slots `op_t_q[0..16]`.
    pub op_t_q: [f32; GRA_NUM_OP_T_MODS],
    /// Asymmetric-inertia-slewed LFO rate output.
    pub lfo_rate_next: f32,
}

impl GraOutputsF32 {
    /// Returns raw IEEE-754 `u32` bit patterns across all `67` delta lanes plus `lfo_rate_next`.
    #[must_use]
    pub fn to_bits_all(&self) -> [u32; GRA_OUT_DELTA_DIM + 1] {
        let mut bits = [0_u32; GRA_OUT_DELTA_DIM + 1];
        for (i, &v) in self.stage_targets.iter().enumerate() {
            bits[i] = v.to_bits();
        }
        bits[31] = self.slot31_lr[0].to_bits();
        bits[32] = self.slot31_lr[1].to_bits();
        for (i, &v) in self.z_stage.iter().enumerate() {
            bits[33 + i] = v.to_bits();
        }
        for (i, &v) in self.z_head_4_8.iter().enumerate() {
            bits[41 + i] = v.to_bits();
        }
        for (i, &v) in self.g_edge.iter().enumerate() {
            bits[45 + i] = v.to_bits();
        }
        for (i, &v) in self.op_t_q.iter().enumerate() {
            bits[51 + i] = v.to_bits();
        }
        bits[67] = self.lfo_rate_next.to_bits();
        bits
    }
}

/// Forward tape cached by [`eval_gra_forward`] for exact reverse-mode [`super::adjoint::pullback_gra`].
#[derive(Clone, Debug)]
pub struct GraCache {
    pub(crate) xi: [f64; GRA_MAX_XI],
    pub(crate) q2: [f64; 4],
    pub(crate) q3: [f64; 4],
    pub(crate) lfo_rate_prev: f64,
    pub(crate) h0: [f64; GRA_MAX_D],
    pub(crate) g: [f64; GRA_MAX_H],
    pub(crate) v: [f64; GRA_MAX_H],
    pub(crate) sigma_g: [f64; GRA_MAX_H],
    pub(crate) dsigma_g: [f64; GRA_MAX_H],
    pub(crate) a: [f64; GRA_MAX_H],
    pub(crate) u1: [f64; GRA_MAX_D],
    pub(crate) h1: [f64; GRA_MAX_D],
    pub(crate) block_clipped: [bool; GRA_MAX_D / 4],
    pub(crate) block_scale: [f64; GRA_MAX_D / 4],
    pub(crate) d_raw: [f64; GRA_MAX_R_PADE],
    pub(crate) d_clamped: [f64; GRA_MAX_R_PADE],
    pub(crate) rho: [f64; GRA_MAX_R_PADE],
    pub(crate) num_dot: [f64; GRA_MAX_M],
    pub(crate) y_rat: [f64; GRA_MAX_M],
    pub(crate) entmax: Entmax15Output,
    pub(crate) lfo_rate_target: f64,
    pub(crate) lfo_rate_alpha: f64,
    pub(crate) q2_gain: [f64; 4],
    pub(crate) q3_gain: [f64; 4],
    pub(crate) q3_inv_den: [f64; 4],
    pub(crate) q3_pade: [f64; 4],
    pub(crate) dyn_vec: [f64; 8],
}

impl Default for GraCache {
    fn default() -> Self {
        Self {
            xi: [0.0; GRA_MAX_XI],
            q2: [0.0; 4],
            q3: [0.0; 4],
            lfo_rate_prev: 0.0,
            h0: [0.0; GRA_MAX_D],
            g: [0.0; GRA_MAX_H],
            v: [0.0; GRA_MAX_H],
            sigma_g: [0.0; GRA_MAX_H],
            dsigma_g: [0.0; GRA_MAX_H],
            a: [0.0; GRA_MAX_H],
            u1: [0.0; GRA_MAX_D],
            h1: [0.0; GRA_MAX_D],
            block_clipped: [false; GRA_MAX_D / 4],
            block_scale: [1.0; GRA_MAX_D / 4],
            d_raw: [0.0; GRA_MAX_R_PADE],
            d_clamped: [0.0; GRA_MAX_R_PADE],
            rho: [1.0; GRA_MAX_R_PADE],
            num_dot: [0.0; GRA_MAX_M],
            y_rat: [0.0; GRA_MAX_M],
            entmax: Entmax15Output::default(),
            lfo_rate_target: 0.0,
            lfo_rate_alpha: 0.5,
            q2_gain: [0.0; 4],
            q3_gain: [0.0; 4],
            q3_inv_den: [1.0; 4],
            q3_pade: [0.0; 4],
            dyn_vec: [0.0; 8],
        }
    }
}

/// Evaluates differentiable `binary64` `ControlBus v4 (GRA)` forward pass and records [`GraCache`].
#[must_use]
pub fn eval_gra_forward(
    params: &GraParams,
    xi_slice: &[f64],
    q2: [f64; 4],
    q3: [f64; 4],
    lfo_rate_prev: f64,
) -> (GraOutputs, GraCache) {
    let (d, h, m, r_pade) = params.profile.dims();
    let xi_dim = params.xi_dim.min(GRA_MAX_XI);
    let (r_trust, anchor_margin) = params.sanitized_scalars();
    let mut cache = GraCache::default();

    for i in 0..xi_dim {
        let v = xi_slice.get(i).copied().unwrap_or(0.0);
        cache.xi[i] = canonicalize_zero_f64(if v.is_finite() { v } else { 0.0 });
    }
    for i in 0..4 {
        cache.q2[i] = canonicalize_zero_f64(if q2[i].is_finite() { q2[i] } else { 0.0 });
        cache.q3[i] = canonicalize_zero_f64(if q3[i].is_finite() { q3[i] } else { 0.0 });
    }
    cache.lfo_rate_prev = canonicalize_zero_f64(if lfo_rate_prev.is_finite() {
        lfo_rate_prev
    } else {
        0.0
    });

    // Stage 1: h_0 = W_in * xi (isolated from SO(4) z_head[0..4]).
    for row in 0..d {
        let mut acc = 0.0_f64;
        for col in 0..xi_dim {
            acc += params.w_in[row][col] * cache.xi[col];
        }
        cache.h0[row] = canonicalize_zero_f64(acc);
    }

    // Stage 2a: Zero-anchored SwiGLU (sigma_poly cubic smoothstep) + residual u_1 = h_0 + W_o * a.
    for row in 0..h {
        let mut g_acc = 0.0_f64;
        let mut v_acc = 0.0_f64;
        for col in 0..d {
            let h0_c = cache.h0[col];
            g_acc += params.w_g[row][col] * h0_c;
            v_acc += params.w_v[row][col] * h0_c;
        }
        let (sg, dsg) = sigma_poly_f64(g_acc);
        cache.g[row] = g_acc;
        cache.v[row] = v_acc;
        cache.sigma_g[row] = sg;
        cache.dsigma_g[row] = dsg;
        cache.a[row] = canonicalize_zero_f64(v_acc * g_acc * sg);
    }
    for row in 0..d {
        let mut acc = cache.h0[row];
        for col in 0..h {
            acc += params.w_o[row][col] * cache.a[col];
        }
        cache.u1[row] = canonicalize_zero_f64(acc);
    }

    // Independent 4-D trust-region projection blocks on h_1.
    for b in 0..(d / 4) {
        let base = b * 4;
        let blk = [
            cache.u1[base],
            cache.u1[base + 1],
            cache.u1[base + 2],
            cache.u1[base + 3],
        ];
        let (clipped, was_clipped, scale) = trust_clip_4d_f64(blk, r_trust);
        cache.h1[base..base + 4].copy_from_slice(&clipped);
        cache.block_clipped[b] = was_clipped;
        cache.block_scale[b] = scale;
    }

    // Stage 2b: Smooth clamped Padé-[1/1] (`1 + d_r >= 0.25`, `rho_r <= 4.0`).
    for r in 0..r_pade {
        let mut d_acc = 0.0_f64;
        for col in 0..d {
            d_acc += params.w_den[r][col] * cache.h1[col];
        }
        let dc = d_acc.clamp(PADE_DEN_MIN, PADE_DEN_MAX);
        cache.d_raw[r] = d_acc;
        cache.d_clamped[r] = dc;
        cache.rho[r] = 1.0 / (1.0 + dc);
    }
    for k in 0..m {
        let mut n_acc = 0.0_f64;
        for col in 0..d {
            n_acc += params.w_num[k][col] * cache.h1[col];
        }
        let grp = params.profile.pade_group(k);
        cache.num_dot[k] = canonicalize_zero_f64(n_acc);
        cache.y_rat[k] = canonicalize_zero_f64(n_acc * cache.rho[grp]);
    }

    // Stage 3: Anchored `1.5-Entmax` prototypes (`m = 0` StarCenter anchor, `V_0 == 0`).
    let mut scores = [0.0_f64; GRA_MAX_M];
    scores[0] = 2.0 * anchor_margin + cache.y_rat[0];
    scores[1..m].copy_from_slice(&cache.y_rat[1..m]);
    cache.entmax = entmax15_forward_f64(&scores[..m]);

    // Stage 4: Bilinear dynamic/inertia gate (`q2` Knob x LFO w/ asymmetric slew, `q3` Padé env).
    let mut rate_target = 0.0_f64;
    for col in 0..d {
        rate_target += params.dyn_gate.lfo_rate_knob[col] * cache.h1[col];
    }
    cache.lfo_rate_target = canonicalize_zero_f64(rate_target);
    let alpha_up = if params.dyn_gate.alpha_up.is_finite() {
        params.dyn_gate.alpha_up.clamp(1.0e-5, 1.0)
    } else {
        0.5
    };
    let alpha_dn = if params.dyn_gate.alpha_dn.is_finite() {
        params.dyn_gate.alpha_dn.clamp(1.0e-5, 1.0)
    } else {
        0.25
    };
    let alpha_slew = if rate_target >= cache.lfo_rate_prev {
        alpha_up
    } else {
        alpha_dn
    };
    cache.lfo_rate_alpha = alpha_slew;
    let lfo_rate_next = cache.lfo_rate_prev + alpha_slew * (rate_target - cache.lfo_rate_prev);

    for i in 0..4 {
        let mut g2 = params.dyn_gate.w_lfo_base[i];
        let mut g3 = params.dyn_gate.w_env_base[i];
        for col in 0..d {
            g2 += params.dyn_gate.w_knob_lfo[i][col] * cache.h1[col];
            g3 += params.dyn_gate.w_knob_env[i][col] * cache.h1[col];
        }
        cache.q2_gain[i] = canonicalize_zero_f64(g2);
        cache.q3_gain[i] = canonicalize_zero_f64(g3);
        let d_env = params.dyn_gate.env_pade_den[i].max(0.0);
        let inv_den = 1.0 / (1.0 + d_env * cache.q3[i].abs());
        let pade_q3 = cache.q3[i] * inv_den;
        cache.q3_inv_den[i] = inv_den;
        cache.q3_pade[i] = canonicalize_zero_f64(pade_q3);
        cache.dyn_vec[i] = canonicalize_zero_f64(g2 * cache.q2[i]);
        cache.dyn_vec[4 + i] = canonicalize_zero_f64(g3 * pade_q3);
    }

    // Assemble all 67 output deltas (`V_proto[0] == 0.0` ensures `0-ULP StarCenter` when `xi == 0`).
    let mut flat = [0.0_f64; GRA_OUT_DELTA_DIM];
    for (out_idx, dst) in flat.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for k in 1..m {
            acc += cache.entmax.probs[k] * params.v_proto[k][out_idx];
        }
        for k in 0..m {
            acc += params.w_rat_direct[out_idx][k] * cache.y_rat[k];
        }
        for j in 0..8 {
            acc += params.dyn_gate.dyn_to_out[out_idx][j] * cache.dyn_vec[j];
        }
        *dst = canonicalize_zero_f64(acc);
    }
    (GraOutputs::from_flat(&flat, lfo_rate_next), cache)
}

/// Runtime `binary32` mirror of [`eval_gra_forward`] (`0 FDIV`, `0 FSQRT`, `0-ULP StarCenter`).
#[must_use]
pub fn eval_gra_f32_mirror(
    params: &GraParams,
    xi_slice: &[f32],
    q2: [f32; 4],
    q3: [f32; 4],
    lfo_rate_prev: f32,
) -> GraOutputsF32 {
    let (d, h, m, r_pade) = params.profile.dims();
    let xi_dim = params.xi_dim.min(GRA_MAX_XI);
    let (r_trust_64, anchor_margin_64) = params.sanitized_scalars();
    let r_trust = r_trust_64 as f32;
    let anchor_margin = anchor_margin_64 as f32;

    let mut xi = [0.0_f32; GRA_MAX_XI];
    for i in 0..xi_dim {
        xi[i] = canonicalize_zero_f32(sanitize_finite_f32(xi_slice.get(i).copied().unwrap_or(0.0)));
    }
    let q2s = q2.map(|v| canonicalize_zero_f32(sanitize_finite_f32(v)));
    let q3s = q3.map(|v| canonicalize_zero_f32(sanitize_finite_f32(v)));
    let rate_prev = canonicalize_zero_f32(sanitize_finite_f32(lfo_rate_prev));

    let mut h0 = [0.0_f32; GRA_MAX_D];
    for row in 0..d {
        let mut acc = 0.0_f32;
        for col in 0..xi_dim {
            acc += (params.w_in[row][col] as f32) * xi[col];
        }
        h0[row] = canonicalize_zero_f32(acc);
    }

    let mut a = [0.0_f32; GRA_MAX_H];
    for row in 0..h {
        let mut g_acc = 0.0_f32;
        let mut v_acc = 0.0_f32;
        for col in 0..d {
            let h0_c = h0[col];
            g_acc += (params.w_g[row][col] as f32) * h0_c;
            v_acc += (params.w_v[row][col] as f32) * h0_c;
        }
        a[row] = canonicalize_zero_f32(v_acc * g_acc * sigma_poly_f32(g_acc));
    }

    let mut h1 = [0.0_f32; GRA_MAX_D];
    for row in 0..d {
        let mut acc = h0[row];
        for col in 0..h {
            acc += (params.w_o[row][col] as f32) * a[col];
        }
        h1[row] = canonicalize_zero_f32(acc);
    }
    for b in 0..(d / 4) {
        let base = b * 4;
        let mut blk = [h1[base], h1[base + 1], h1[base + 2], h1[base + 3]];
        trust_clip_outward_4d(&mut blk, r_trust);
        for i in 0..4 {
            h1[base + i] = canonicalize_zero_f32(blk[i]);
        }
    }

    let mut rho = [1.0_f32; GRA_MAX_R_PADE];
    for r in 0..r_pade {
        let mut d_acc = 0.0_f32;
        for col in 0..d {
            d_acc += (params.w_den[r][col] as f32) * h1[col];
        }
        let dc = sanitize_finite_f32(d_acc).clamp(PADE_DEN_MIN as f32, PADE_DEN_MAX as f32);
        rho[r] = recip_nr3_pos_f32(1.0_f32 + dc);
    }

    let mut y_rat = [0.0_f32; GRA_MAX_M];
    for k in 0..m {
        let mut n_acc = 0.0_f32;
        for col in 0..d {
            n_acc += (params.w_num[k][col] as f32) * h1[col];
        }
        let grp = params.profile.pade_group(k);
        y_rat[k] = canonicalize_zero_f32(n_acc * rho[grp]);
    }

    let mut scores = [0.0_f32; GRA_MAX_M];
    scores[0] = 2.0_f32 * anchor_margin + y_rat[0];
    scores[1..m].copy_from_slice(&y_rat[1..m]);
    let (probs, _) = entmax15_forward_f32_mirror(&scores[..m]);

    let mut rate_target = 0.0_f32;
    for col in 0..d {
        rate_target += (params.dyn_gate.lfo_rate_knob[col] as f32) * h1[col];
    }
    let alpha_up = if params.dyn_gate.alpha_up.is_finite() {
        (params.dyn_gate.alpha_up as f32).clamp(1.0e-5, 1.0)
    } else {
        0.5_f32
    };
    let alpha_dn = if params.dyn_gate.alpha_dn.is_finite() {
        (params.dyn_gate.alpha_dn as f32).clamp(1.0e-5, 1.0)
    } else {
        0.25_f32
    };
    let alpha_slew = if rate_target >= rate_prev {
        alpha_up
    } else {
        alpha_dn
    };
    let lfo_rate_next = canonicalize_zero_f32(rate_prev + alpha_slew * (rate_target - rate_prev));

    let mut dyn_vec = [0.0_f32; 8];
    for i in 0..4 {
        let mut g2 = params.dyn_gate.w_lfo_base[i] as f32;
        let mut g3 = params.dyn_gate.w_env_base[i] as f32;
        for col in 0..d {
            g2 += (params.dyn_gate.w_knob_lfo[i][col] as f32) * h1[col];
            g3 += (params.dyn_gate.w_knob_env[i][col] as f32) * h1[col];
        }
        let d_env = (params.dyn_gate.env_pade_den[i] as f32).max(0.0_f32);
        let inv_den = recip_nr3_pos_f32(1.0_f32 + d_env * q3s[i].abs());
        dyn_vec[i] = canonicalize_zero_f32(g2 * q2s[i]);
        dyn_vec[4 + i] = canonicalize_zero_f32(g3 * q3s[i] * inv_den);
    }

    let mut flat = [0.0_f32; GRA_OUT_DELTA_DIM];
    for (out_idx, dst) in flat.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for k in 1..m {
            acc += probs[k] * (params.v_proto[k][out_idx] as f32);
        }
        for k in 0..m {
            acc += (params.w_rat_direct[out_idx][k] as f32) * y_rat[k];
        }
        for j in 0..8 {
            acc += (params.dyn_gate.dyn_to_out[out_idx][j] as f32) * dyn_vec[j];
        }
        *dst = canonicalize_zero_f32(acc);
    }

    let mut out = GraOutputsF32 {
        stage_targets: [0.0; STAGE_TARGET_SLOTS],
        slot31_lr: [flat[31], flat[32]],
        z_stage: [0.0; GRA_Z_STAGE_DIM],
        z_head_4_8: [0.0; GRA_Z_HEAD_UNROTATED_DIM],
        g_edge: [0.0; GRA_NUM_EDGES],
        op_t_q: [0.0; GRA_NUM_OP_T_MODS],
        lfo_rate_next,
    };
    out.stage_targets.copy_from_slice(&flat[0..32]);
    out.z_stage.copy_from_slice(&flat[33..41]);
    out.z_head_4_8.copy_from_slice(&flat[41..45]);
    out.g_edge.copy_from_slice(&flat[45..51]);
    out.op_t_q.copy_from_slice(&flat[51..67]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::gra::model::GraTierProfile;

    fn sample_params(profile: GraTierProfile, xi_dim: usize) -> GraParams {
        let mut p = GraParams::zeros(profile, xi_dim);
        let (d, h, m, r_pade) = profile.dims();
        p.trust_radius = 1.8;
        p.anchor_margin = 1.05;
        for row in 0..d {
            for col in 0..xi_dim {
                p.w_in[row][col] = 0.35 * (((row * 7 + col * 3 + 1) as f64) * 0.41).sin();
            }
        }
        for row in 0..h {
            for col in 0..d {
                p.w_g[row][col] = 0.45 * (((row * 5 + col * 11 + 2) as f64) * 0.37).cos();
                p.w_v[row][col] = 0.40 * (((row * 3 + col * 13 + 4) as f64) * 0.29).sin();
                p.w_o[col][row] = 0.30 * (((row * 9 + col * 2 + 5) as f64) * 0.31).sin();
            }
        }
        for r in 0..r_pade {
            for col in 0..d {
                p.w_den[r][col] = 0.55 * (((r * 7 + col * 5 + 1) as f64) * 0.53).sin();
            }
        }
        for k in 0..m {
            for col in 0..d {
                p.w_num[k][col] = 0.75 * (((k * 11 + col * 3 + 2) as f64) * 0.47).cos();
            }
            if k > 0 {
                for out_idx in 0..GRA_OUT_DELTA_DIM {
                    p.v_proto[k][out_idx] =
                        0.25 * (((k * 17 + out_idx * 13 + 3) as f64) * 0.19).sin();
                }
            }
        }
        for out_idx in 0..GRA_OUT_DELTA_DIM {
            for j in 0..8 {
                p.dyn_gate.dyn_to_out[out_idx][j] =
                    0.15 * (((out_idx * 5 + j * 11 + 1) as f64) * 0.23).cos();
            }
        }
        for i in 0..4 {
            p.dyn_gate.env_pade_den[i] = 0.4 + 0.1 * (i as f64);
            for col in 0..d {
                p.dyn_gate.w_knob_lfo[i][col] = 0.3 * (((i * 7 + col + 1) as f64) * 0.61).sin();
                p.dyn_gate.w_knob_env[i][col] = 0.25 * (((i * 5 + col + 2) as f64) * 0.43).cos();
            }
        }
        for col in 0..d {
            p.dyn_gate.lfo_rate_knob[col] = 0.35 * (((col + 1) as f64) * 0.57).sin();
        }
        p.enforce_structural_invariants();
        p
    }

    #[test]
    fn test_zero_ulp_star_center_and_f32_f64_parity_all_tiers() {
        for profile in [
            GraTierProfile::Lite,
            GraTierProfile::Full,
            GraTierProfile::Pro,
            GraTierProfile::Ultra,
        ] {
            let p = sample_params(profile, 6);
            // 1. At StarCenter (xi == +0.0, static or with nonzero q2/q3), output bits are all 0x0000_0000!
            let out_star_32 =
                eval_gra_f32_mirror(&p, &[0.0_f32; 6], [0.6, -0.4, 0.2, 0.1], [0.5; 4], 0.0);
            assert!(
                out_star_32.to_bits_all().iter().all(|&b| b == 0),
                "{profile:?} failed 0-ULP StarCenter"
            );

            // 2. Away from StarCenter, f32 mirror matches f64 forward within NR3 tolerance.
            let xi = [0.75_f64, -0.45, 0.35, -0.60, 1.0, 0.0];
            let xi32 = xi.map(|v| v as f32);
            let q2 = [0.45_f64, -0.30, 0.20, 0.10];
            let q3 = [0.55_f64, 0.25, 0.80, 0.15];
            let (out64, _) = eval_gra_forward(&p, &xi, q2, q3, 0.1);
            let out32 =
                eval_gra_f32_mirror(&p, &xi32, q2.map(|v| v as f32), q3.map(|v| v as f32), 0.1);
            let flat64 = out64.to_flat();
            let flat32_bits = out32.to_bits_all();
            for i in 0..GRA_OUT_DELTA_DIM {
                let v32 = f32::from_bits(flat32_bits[i]);
                assert!(
                    (f64::from(v32) - flat64[i]).abs() < 2.0e-5,
                    "{profile:?} idx={i} f32={v32} f64={}",
                    flat64[i]
                );
            }
            assert!((f64::from(out32.lfo_rate_next) - out64.lfo_rate_next).abs() < 1.0e-6);
        }

        // 3. SwiGLU cubic smoothstep symmetry: g * sigma(g) - (-g) * sigma(-g) == g.
        for &g in &[0.0, 0.25, 1.5, 3.9, 6.0] {
            let (sp, _) = sigma_poly_f64(g);
            let (sm, _) = sigma_poly_f64(-g);
            assert!((sp + sm - 1.0).abs() < 1.0e-15);
            assert!(((g * sp - (-g) * sm) - g).abs() < 1.0e-14);
        }
    }
}
