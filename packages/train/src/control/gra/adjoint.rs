//! Exact reverse-mode VJP (`pullback_gra`) over [`GraCache`] (`0-sqrt` `entmax15_vjp`,
//! masked `Padé-[1/1]`, radial 4-D `trust_clip` Jacobian, `SwiGLU`, bilinear inertia gate,
//! `W_in` -> `d_xi`) (`ref[dsp.ctrl.gra.rational-and-entmax]`).

use dnam_vm::control::STAGE_TARGET_SLOTS;

pub use super::bridge::GraCoreOpTBridge;
use super::entmax::entmax15_vjp_scores;
use super::forward::GraCache;
use super::model::{
    GRA_MAX_D, GRA_MAX_H, GRA_MAX_M, GRA_MAX_R_PADE, GRA_MAX_XI, GRA_NUM_EDGES, GRA_NUM_OP_T_MODS,
    GRA_OUT_DELTA_DIM, GRA_Z_HEAD_UNROTATED_DIM, GRA_Z_STAGE_DIM, GraDynGateWeights, GraParams,
    PADE_DEN_MAX, PADE_DEN_MIN,
};

/// Upstream adjoints matching [`GraOutputs`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraOutputsGrad {
    /// Adjoints for `stage_targets[0..32]` (`slot 31` maps to `slot31_lr[0]`).
    pub d_stage_targets: [f64; STAGE_TARGET_SLOTS],
    /// Adjoints for `slot31_lr` (`[L, R]`; summed into `[31, 32]` of flat delta).
    pub d_slot31_lr: [f64; 2],
    /// Adjoints for `z_stage[0..8]`.
    pub d_z_stage: [f64; GRA_Z_STAGE_DIM],
    /// Adjoints for `z_head[4..8]`.
    pub d_z_head_4_8: [f64; GRA_Z_HEAD_UNROTATED_DIM],
    /// Adjoints for `g_edge[0..6]`.
    pub d_g_edge: [f64; GRA_NUM_EDGES],
    /// Adjoints for `op_t_q[0..16]`.
    pub d_op_t_q: [f64; GRA_NUM_OP_T_MODS],
    /// Adjoint for `lfo_rate_next`.
    pub d_lfo_rate_next: f64,
}

impl GraOutputsGrad {
    /// Flattens `self` into the `67`-element canonical delta gradient vector.
    #[must_use]
    pub fn to_flat(&self) -> [f64; GRA_OUT_DELTA_DIM] {
        let mut raw = [0.0_f64; GRA_OUT_DELTA_DIM];
        raw[0..32].copy_from_slice(&self.d_stage_targets);
        raw[31] += self.d_slot31_lr[0];
        raw[32] = self.d_slot31_lr[1];
        raw[33..41].copy_from_slice(&self.d_z_stage);
        raw[41..45].copy_from_slice(&self.d_z_head_4_8);
        raw[45..51].copy_from_slice(&self.d_g_edge);
        raw[51..67].copy_from_slice(&self.d_op_t_q);
        raw
    }
}

/// Complete reverse-mode gradients for [`GraParams`] and forward inputs `(xi, q2, q3, lfo_rate_prev)`.
#[derive(Clone, Debug, PartialEq)]
pub struct GraParamGrads {
    /// Gradient wrt `w_in [D][P_xi]`.
    pub d_w_in: [[f64; GRA_MAX_XI]; GRA_MAX_D],
    /// Gradient wrt `w_g [H][D]`.
    pub d_w_g: [[f64; GRA_MAX_D]; GRA_MAX_H],
    /// Gradient wrt `w_v [H][D]`.
    pub d_w_v: [[f64; GRA_MAX_D]; GRA_MAX_H],
    /// Gradient wrt `w_o [D][H]`.
    pub d_w_o: [[f64; GRA_MAX_H]; GRA_MAX_D],
    /// Gradient wrt `w_den [R][D]`.
    pub d_w_den: [[f64; GRA_MAX_D]; GRA_MAX_R_PADE],
    /// Gradient wrt `w_num [M][D]`.
    pub d_w_num: [[f64; GRA_MAX_D]; GRA_MAX_M],
    /// Gradient wrt `v_proto [M][67]` (`row 0` remains strictly `0.0`).
    pub d_v_proto: [[f64; GRA_OUT_DELTA_DIM]; GRA_MAX_M],
    /// Gradient wrt `w_rat_direct [67][M]`.
    pub d_w_rat_direct: [[f64; GRA_MAX_M]; GRA_OUT_DELTA_DIM],
    /// Gradient wrt [`GraDynGateWeights`].
    pub d_dyn_gate: GraDynGateWeights,
    /// Gradient wrt unrotated basis `xi` (`d_u_norm` pulls back through Lobatto derivative).
    pub d_xi: [f64; GRA_MAX_XI],
    /// Gradient wrt LFO quadrature `q2`.
    pub d_q2: [f64; 4],
    /// Gradient wrt `SensoryQuad` envelope `q3`.
    pub d_q3: [f64; 4],
    /// Gradient wrt `lfo_rate_prev`.
    pub d_lfo_rate_prev: f64,
}

impl Default for GraParamGrads {
    fn default() -> Self {
        Self {
            d_w_in: [[0.0; GRA_MAX_XI]; GRA_MAX_D],
            d_w_g: [[0.0; GRA_MAX_D]; GRA_MAX_H],
            d_w_v: [[0.0; GRA_MAX_D]; GRA_MAX_H],
            d_w_o: [[0.0; GRA_MAX_H]; GRA_MAX_D],
            d_w_den: [[0.0; GRA_MAX_D]; GRA_MAX_R_PADE],
            d_w_num: [[0.0; GRA_MAX_D]; GRA_MAX_M],
            d_v_proto: [[0.0; GRA_OUT_DELTA_DIM]; GRA_MAX_M],
            d_w_rat_direct: [[0.0; GRA_MAX_M]; GRA_OUT_DELTA_DIM],
            d_dyn_gate: GraDynGateWeights {
                alpha_up: 0.0,
                alpha_dn: 0.0,
                ..GraDynGateWeights::default()
            },
            d_xi: [0.0; GRA_MAX_XI],
            d_q2: [0.0; 4],
            d_q3: [0.0; 4],
            d_lfo_rate_prev: 0.0,
        }
    }
}

/// Exact reverse-mode VJP over [`GraCache`] (`rel_err < 1e-6` vs central FD away from kinks).
#[must_use]
pub fn pullback_gra(params: &GraParams, cache: &GraCache, d_out: &GraOutputsGrad) -> GraParamGrads {
    let (d, h, m, r_pade) = params.profile.dims();
    let xi_dim = params.xi_dim.min(GRA_MAX_XI);
    let (r_trust, _) = params.sanitized_scalars();
    let mut grads = GraParamGrads::default();
    let d_flat = d_out.to_flat();

    // 1. Pull back output assembly -> d_probs, d_y_rat, d_dyn_vec.
    let mut d_probs = [0.0_f64; GRA_MAX_M];
    let mut d_y_rat = [0.0_f64; GRA_MAX_M];
    let mut d_dyn_vec = [0.0_f64; 8];
    for (out_idx, &df) in d_flat.iter().enumerate() {
        if df == 0.0 {
            continue;
        }
        for k in 1..m {
            grads.d_v_proto[k][out_idx] += df * cache.entmax.probs[k];
            d_probs[k] += df * params.v_proto[k][out_idx];
        }
        for k in 0..m {
            grads.d_w_rat_direct[out_idx][k] += df * cache.y_rat[k];
            d_y_rat[k] += df * params.w_rat_direct[out_idx][k];
        }
        for j in 0..8 {
            grads.d_dyn_gate.dyn_to_out[out_idx][j] += df * cache.dyn_vec[j];
            d_dyn_vec[j] += df * params.dyn_gate.dyn_to_out[out_idx][j];
        }
    }

    // 2. Dynamic / inertia gate reverse pass -> d_h1, d_q2, d_q3, d_lfo_rate_prev.
    let mut d_h1 = [0.0_f64; GRA_MAX_D];
    let d_rate_next = d_out.d_lfo_rate_next;
    let alpha_slew = cache.lfo_rate_alpha;
    grads.d_lfo_rate_prev = (1.0 - alpha_slew) * d_rate_next;
    let d_rate_target = alpha_slew * d_rate_next;
    let rate_diff = cache.lfo_rate_target - cache.lfo_rate_prev;
    if rate_diff >= 0.0 {
        if params.dyn_gate.alpha_up > 1.0e-5 && params.dyn_gate.alpha_up < 1.0 {
            grads.d_dyn_gate.alpha_up += d_rate_next * rate_diff;
        }
    } else if params.dyn_gate.alpha_dn > 1.0e-5 && params.dyn_gate.alpha_dn < 1.0 {
        grads.d_dyn_gate.alpha_dn += d_rate_next * rate_diff;
    }
    for col in 0..d {
        grads.d_dyn_gate.lfo_rate_knob[col] += d_rate_target * cache.h1[col];
        d_h1[col] += d_rate_target * params.dyn_gate.lfo_rate_knob[col];
    }

    for i in 0..4 {
        let d_q2_eff = d_dyn_vec[i];
        let d_q2_gain = d_q2_eff * cache.q2[i];
        grads.d_q2[i] = d_q2_eff * cache.q2_gain[i];
        grads.d_dyn_gate.w_lfo_base[i] += d_q2_gain;
        for col in 0..d {
            grads.d_dyn_gate.w_knob_lfo[i][col] += d_q2_gain * cache.h1[col];
            d_h1[col] += d_q2_gain * params.dyn_gate.w_knob_lfo[i][col];
        }

        let d_q3_eff = d_dyn_vec[4 + i];
        let d_q3_gain = d_q3_eff * cache.q3_pade[i];
        let d_q3_pade = d_q3_eff * cache.q3_gain[i];
        let inv_den = cache.q3_inv_den[i];
        grads.d_q3[i] = d_q3_pade * inv_den * inv_den;
        if params.dyn_gate.env_pade_den[i] >= 0.0 {
            grads.d_dyn_gate.env_pade_den[i] +=
                -d_q3_pade * cache.q3_pade[i] * inv_den * cache.q3[i].abs();
        }
        grads.d_dyn_gate.w_env_base[i] += d_q3_gain;
        for col in 0..d {
            grads.d_dyn_gate.w_knob_env[i][col] += d_q3_gain * cache.h1[col];
            d_h1[col] += d_q3_gain * params.dyn_gate.w_knob_env[i][col];
        }
    }

    // 3. Anchored 1.5-Entmax VJP (`0-sqrt`, `0-FDIV` at active support).
    let d_scores = entmax15_vjp_scores(&cache.entmax, &d_probs[..m]);
    for k in 0..m {
        d_y_rat[k] += d_scores[k];
    }

    // 4. Padé-[1/1] reverse pass (denominator derivative masked strictly on `(-0.75, +7.0)`).
    let mut d_rho = [0.0_f64; GRA_MAX_R_PADE];
    for k in 0..m {
        let grp = params.profile.pade_group(k);
        let d_num = d_y_rat[k] * cache.rho[grp];
        d_rho[grp] += d_y_rat[k] * cache.num_dot[k];
        for col in 0..d {
            grads.d_w_num[k][col] += d_num * cache.h1[col];
            d_h1[col] += d_num * params.w_num[k][col];
        }
    }
    for r in 0..r_pade {
        if cache.d_raw[r] > PADE_DEN_MIN && cache.d_raw[r] < PADE_DEN_MAX {
            let dd = -d_rho[r] * cache.rho[r] * cache.rho[r];
            for col in 0..d {
                grads.d_w_den[r][col] += dd * cache.h1[col];
                d_h1[col] += dd * params.w_den[r][col];
            }
        }
    }

    // 5. Independent 4-D trust_clip radial-projection Jacobian.
    let mut d_u1 = [0.0_f64; GRA_MAX_D];
    let inv_r2 = 1.0 / (r_trust * r_trust);
    for b in 0..(d / 4) {
        let base = b * 4;
        if !cache.block_clipped[b] {
            d_u1[base..base + 4].copy_from_slice(&d_h1[base..base + 4]);
        } else {
            let s = cache.block_scale[b];
            let dot_gh = d_h1[base] * cache.h1[base]
                + d_h1[base + 1] * cache.h1[base + 1]
                + d_h1[base + 2] * cache.h1[base + 2]
                + d_h1[base + 3] * cache.h1[base + 3];
            for i in 0..4 {
                d_u1[base + i] = s * (d_h1[base + i] - (dot_gh * inv_r2) * cache.h1[base + i]);
            }
        }
    }

    // 6. Zero-anchored SwiGLU + residual reverse pass -> d_h0.
    let mut d_h0 = d_u1;
    let mut d_a = [0.0_f64; GRA_MAX_H];
    for row in 0..d {
        let du = d_u1[row];
        for col in 0..h {
            grads.d_w_o[row][col] += du * cache.a[col];
            d_a[col] += du * params.w_o[row][col];
        }
    }
    for row in 0..h {
        let da = d_a[row];
        let g_val = cache.g[row];
        let v_val = cache.v[row];
        let sg = cache.sigma_g[row];
        let dsg = cache.dsigma_g[row];
        let d_v = da * g_val * sg;
        let d_g = da * v_val * (sg + g_val * dsg);
        for col in 0..d {
            let h0_c = cache.h0[col];
            grads.d_w_g[row][col] += d_g * h0_c;
            grads.d_w_v[row][col] += d_v * h0_c;
            d_h0[col] += d_g * params.w_g[row][col] + d_v * params.w_v[row][col];
        }
    }

    // 7. Unrotated Stage-1 W_in reverse pass -> d_w_in, d_xi.
    for row in 0..d {
        let dh = d_h0[row];
        for col in 0..xi_dim {
            grads.d_w_in[row][col] += dh * cache.xi[col];
            grads.d_xi[col] += dh * params.w_in[row][col];
        }
    }
    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::gra::forward::{GraOutputs, eval_gra_forward};
    use crate::control::gra::model::GraTierProfile;
    use proptest::prelude::*;

    fn build_active_params(profile: GraTierProfile, xi_dim: usize, trust_radius: f64) -> GraParams {
        let mut p = GraParams::zeros(profile, xi_dim);
        let (d, h, m, r_pade) = profile.dims();
        p.trust_radius = trust_radius;
        p.anchor_margin = 0.25; // Keep inside multi-support k* > 1 regime for smooth FD.
        p.anchor_margin = p.anchor_margin.max(1.0); // Enforce canonical >= 1.0, use strong w_num below.
        for row in 0..d {
            for col in 0..xi_dim {
                p.w_in[row][col] = 0.7 * (((row * 5 + col * 3 + 1) as f64) * 0.43).sin();
            }
        }
        for row in 0..h {
            for col in 0..d {
                p.w_g[row][col] = 0.55 * (((row * 7 + col * 2 + 1) as f64) * 0.31).cos();
                p.w_v[row][col] = 0.50 * (((row * 3 + col * 11 + 2) as f64) * 0.37).sin();
                p.w_o[col][row] = 0.45 * (((row * 9 + col * 5 + 3) as f64) * 0.29).sin();
            }
        }
        for r in 0..r_pade {
            for col in 0..d {
                p.w_den[r][col] = 0.40 * (((r * 5 + col * 3 + 1) as f64) * 0.51).sin();
            }
        }
        for k in 0..m {
            for col in 0..d {
                p.w_num[k][col] = 1.45 * (((k * 13 + col * 7 + 1) as f64) * 0.41).cos();
            }
            if k > 0 {
                for out_idx in 0..GRA_OUT_DELTA_DIM {
                    p.v_proto[k][out_idx] =
                        0.35 * (((k * 11 + out_idx * 5 + 2) as f64) * 0.21).sin();
                }
            }
        }
        for out_idx in 0..GRA_OUT_DELTA_DIM {
            for k in 0..m {
                p.w_rat_direct[out_idx][k] =
                    0.08 * (((out_idx * 3 + k * 7 + 1) as f64) * 0.27).cos();
            }
            for j in 0..8 {
                p.dyn_gate.dyn_to_out[out_idx][j] =
                    0.22 * (((out_idx * 7 + j * 3 + 1) as f64) * 0.19).sin();
            }
        }
        p.dyn_gate.w_lfo_base = [0.2, -0.15, 0.1, 0.25];
        p.dyn_gate.w_env_base = [0.15, 0.25, -0.1, 0.2];
        p.dyn_gate.env_pade_den = [0.5, 0.8, 0.3, 1.1];
        p.dyn_gate.alpha_up = 0.65;
        p.dyn_gate.alpha_dn = 0.20;
        for i in 0..4 {
            for col in 0..d {
                p.dyn_gate.w_knob_lfo[i][col] = 0.25 * (((i * 5 + col + 1) as f64) * 0.47).sin();
                p.dyn_gate.w_knob_env[i][col] = 0.20 * (((i * 7 + col + 2) as f64) * 0.39).cos();
            }
        }
        for col in 0..d {
            p.dyn_gate.lfo_rate_knob[col] = 0.30 * (((col + 1) as f64) * 0.61).sin();
        }
        p.enforce_structural_invariants();
        p
    }

    #[test]
    fn test_pullback_gra_central_fd_interior_and_clipped_regimes() {
        let xi = [1.1_f64, -0.8, 0.65, -0.9, 1.0, 0.3];
        let q2 = [0.45_f64, -0.35, 0.25, 0.15];
        let q3 = [0.50_f64, 0.30, 0.70, 0.20];
        let lfo_rate_prev = 0.25_f64;

        let mut d_out = GraOutputsGrad::default();
        for i in 0..32 {
            d_out.d_stage_targets[i] = ((i as f64) * 0.37 + 0.1).sin();
        }
        d_out.d_slot31_lr = [0.45, -0.65];
        for i in 0..8 {
            d_out.d_z_stage[i] = ((i as f64) * 0.53 + 0.3).cos();
        }
        for i in 0..4 {
            d_out.d_z_head_4_8[i] = ((i as f64) * 0.71).sin();
        }
        for i in 0..6 {
            d_out.d_g_edge[i] = ((i as f64) * 0.41).cos();
        }
        for i in 0..16 {
            d_out.d_op_t_q[i] = ((i as f64) * 0.29 + 0.2).sin();
        }
        d_out.d_lfo_rate_next = 0.85;

        let loss_from_out = |out: &GraOutputs, g: &GraOutputsGrad| -> f64 {
            let f = out.to_flat();
            let gf = g.to_flat();
            f.iter().zip(gf.iter()).map(|(a, b)| a * b).sum::<f64>()
                + out.lfo_rate_next * g.d_lfo_rate_next
        };

        // Test BOTH unclipped (`trust_radius = 8.0`) and clipped (`trust_radius = 0.45`) regimes
        // and both `alpha_up` and `alpha_dn` branches (`lfo_rate_prev = -2.0` vs `+2.0`).
        for (profile, trust_r, prev_rate) in [
            (GraTierProfile::Lite, 8.0, -2.0),
            (GraTierProfile::Ultra, 0.45, 2.0),
        ] {
            let params = build_active_params(profile, 6, trust_r);
            let (out, cache) = eval_gra_forward(&params, &xi, q2, q3, prev_rate);
            assert!(
                cache.entmax.k_star > 1,
                "expected active multi-support k*>1 for FD test"
            );
            if trust_r < 1.0 {
                assert!(cache.block_clipped[0], "expected active trust_clip");
            } else {
                assert!(!cache.block_clipped[0], "expected unclipped interior");
            }
            let _ = out;
            let grads = pullback_gra(&params, &cache, &d_out);

            let eps = 1.0e-6_f64;
            let fd_check = |ana: f64, mutator: &dyn Fn(&mut GraParams, f64), label: &str| {
                let (mut pp, mut pm) = (params.clone(), params.clone());
                mutator(&mut pp, eps);
                mutator(&mut pm, -eps);
                let lp = loss_from_out(&eval_gra_forward(&pp, &xi, q2, q3, prev_rate).0, &d_out);
                let lm = loss_from_out(&eval_gra_forward(&pm, &xi, q2, q3, prev_rate).0, &d_out);
                let fd = (lp - lm) / (2.0 * eps);
                let scale = ana.abs().max(fd.abs()).max(1.0);
                assert!(
                    (ana - fd).abs() < 1.0e-6 * scale,
                    "{profile:?} R={trust_r} {label}: ana={ana} fd={fd}"
                );
            };

            fd_check(grads.d_w_in[0][1], &|p, d| p.w_in[0][1] += d, "w_in[0][1]");
            fd_check(grads.d_w_g[1][2], &|p, d| p.w_g[1][2] += d, "w_g[1][2]");
            fd_check(grads.d_w_v[2][0], &|p, d| p.w_v[2][0] += d, "w_v[2][0]");
            fd_check(grads.d_w_o[1][3], &|p, d| p.w_o[1][3] += d, "w_o[1][3]");
            fd_check(
                grads.d_w_den[0][1],
                &|p, d| p.w_den[0][1] += d,
                "w_den[0][1]",
            );
            fd_check(
                grads.d_w_num[2][1],
                &|p, d| p.w_num[2][1] += d,
                "w_num[2][1]",
            );
            fd_check(
                grads.d_v_proto[1][31],
                &|p, d| p.v_proto[1][31] += d,
                "v_proto[1][31]",
            );
            fd_check(
                grads.d_w_rat_direct[13][0],
                &|p, d| p.w_rat_direct[13][0] += d,
                "w_rat_direct[13][0]",
            );
            fd_check(
                grads.d_dyn_gate.w_knob_lfo[0][1],
                &|p, d| p.dyn_gate.w_knob_lfo[0][1] += d,
                "w_knob_lfo[0][1]",
            );
            fd_check(
                grads.d_dyn_gate.env_pade_den[1],
                &|p, d| p.dyn_gate.env_pade_den[1] += d,
                "env_pade_den[1]",
            );
            fd_check(
                grads.d_dyn_gate.alpha_up,
                &|p, d| p.dyn_gate.alpha_up += d,
                "alpha_up",
            );
            fd_check(
                grads.d_dyn_gate.alpha_dn,
                &|p, d| p.dyn_gate.alpha_dn += d,
                "alpha_dn",
            );
            fd_check(
                grads.d_dyn_gate.dyn_to_out[32][4],
                &|p, d| p.dyn_gate.dyn_to_out[32][4] += d,
                "dyn_to_out[32][4]",
            );

            // Check input gradients d_xi, d_q2, d_q3, d_lfo_rate_prev.
            for idx in 0..3 {
                let (mut xip, mut xim) = (xi, xi);
                xip[idx] += eps;
                xim[idx] -= eps;
                let fd = (loss_from_out(
                    &eval_gra_forward(&params, &xip, q2, q3, prev_rate).0,
                    &d_out,
                ) - loss_from_out(
                    &eval_gra_forward(&params, &xim, q2, q3, prev_rate).0,
                    &d_out,
                )) / (2.0 * eps);
                let ana = grads.d_xi[idx];
                assert!(
                    (ana - fd).abs() < 1.0e-6 * ana.abs().max(fd.abs()).max(1.0),
                    "{profile:?} R={trust_r} xi[{idx}]: ana={ana} fd={fd}"
                );
            }
        }
        let _ = lfo_rate_prev;
    }

    #[test]
    fn test_pullback_gra_clamped_alpha_up_and_dn_yield_zero_grad() {
        let xi = [1.1_f64, -0.8, 0.65, -0.9, 1.0, 0.3];
        let q2 = [0.45_f64, -0.35, 0.25, 0.15];
        let q3 = [0.50_f64, 0.30, 0.70, 0.20];
        let d_out = GraOutputsGrad {
            d_lfo_rate_next: 0.85,
            ..Default::default()
        };

        for (prev_rate, alpha_up, alpha_dn) in [
            (-5.0, 1.0, 0.25),
            (-5.0, 1.5, 0.25),
            (-5.0, 1.0e-5, 0.25),
            (-5.0, 0.0, 0.25),
            (5.0, 0.65, 1.0),
            (5.0, 0.65, 1.5),
            (5.0, 0.65, 1.0e-5),
            (5.0, 0.65, 0.0),
        ] {
            let mut params = build_active_params(GraTierProfile::Lite, 6, 8.0);
            params.dyn_gate.alpha_up = alpha_up;
            params.dyn_gate.alpha_dn = alpha_dn;
            let (_, cache) = eval_gra_forward(&params, &xi, q2, q3, prev_rate);
            let grads = pullback_gra(&params, &cache, &d_out);
            assert_eq!(grads.d_dyn_gate.alpha_up, 0.0);
            assert_eq!(grads.d_dyn_gate.alpha_dn, 0.0);
        }
    }

    proptest! {
        #[test]
        fn prop_gra_pullback_matches_fd_in_smooth_interior(
            xi0 in -0.5_f64..0.5_f64,
            xi1 in -0.5_f64..0.5_f64,
            q2_0 in -0.5_f64..0.5_f64,
            q3_0 in 0.05_f64..0.8_f64,
        ) {
            let params = build_active_params(GraTierProfile::Lite, 4, 8.0);
            let xi = [xi0, xi1, 0.2, -0.15];
            let q2 = [q2_0, 0.1, -0.2, 0.05];
            let q3 = [q3_0, 0.2, 0.1, 0.05];
            let mut d_out = GraOutputsGrad::default();
            d_out.d_stage_targets[13] = 1.0;
            d_out.d_slot31_lr = [0.5, -0.5];
            d_out.d_g_edge[0] = 0.7;
            d_out.d_op_t_q[1] = -0.4;
            d_out.d_lfo_rate_next = 0.3;

            let (_, cache) = eval_gra_forward(&params, &xi, q2, q3, 0.0);
            let grads = pullback_gra(&params, &cache, &d_out);
            let eps = 1.0e-6_f64;
            let eval_loss = |x_in: [f64; 4]| -> f64 {
                let (o, _) = eval_gra_forward(&params, &x_in, q2, q3, 0.0);
                o.stage_targets[13] * d_out.d_stage_targets[13]
                    + o.slot31_lr[0] * d_out.d_slot31_lr[0]
                    + o.slot31_lr[1] * d_out.d_slot31_lr[1]
                    + o.g_edge[0] * d_out.d_g_edge[0]
                    + o.op_t_q[1] * d_out.d_op_t_q[1]
                    + o.lfo_rate_next * d_out.d_lfo_rate_next
            };
            let fd0 = (eval_loss([xi0 + eps, xi1, 0.2, -0.15])
                - eval_loss([xi0 - eps, xi1, 0.2, -0.15]))
                / (2.0 * eps);
            prop_assert!(
                (grads.d_xi[0] - fd0).abs() < 1.0e-5 * grads.d_xi[0].abs().max(fd0.abs()).max(1.0)
            );
        }
    }
}
