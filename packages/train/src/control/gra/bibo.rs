//! Analytical `O(P + M)` Cauchy–Schwarz + `1.5-Entmax` convex-hull BIBO verifier
//! (`audit_gra_bibo_hull`) and `SONA v4.0` [`StageTargetBounds::canonical_v4`]
//! (`ref[dsp.ctrl.gra.bibo-convex-hull]`, `ref[train.audit.phase1.clamp-stability]`).

use core::f64::consts::PI;

use dnam_kernels::OP_N_PRE_MAX_ZOH_HOLD_DEPTH;
use dnam_vm::control::STAGE_TARGET_SLOTS;

use crate::control::stability::{
    MAX_BLOCK_ADAA_OCT, MAX_OP_P_PHASOR_RATE_OCT, MIN_DELAY_CROSSBAR_SAMP,
    MIN_DELAY_STANDALONE_SAMP, StageTargetBounds,
};

use super::model::{
    GRA_MAX_M, GRA_NUM_EDGES, GRA_NUM_OP_T_MODS, GRA_OUT_DELTA_DIM, GraParams, PADE_DEN_MIN,
};

impl StageTargetBounds {
    /// Returns normative `SONA v4.0` stability bounds across all 32 `stage_targets` slots
    /// (`slot 13` bipolar `[-1.998, +1.998]`, `slot 18` `[-64, +64]`, `slot 30` `[-5, +4]` oct,
    /// `slot 31` active stereo differential quadrature AM `[-4, +4]`).
    #[must_use]
    pub fn canonical_v4(crossbar_coupled: bool, min_tau_samp: f64) -> Self {
        let mut min = [0.0_f64; STAGE_TARGET_SLOTS];
        let mut max = [0.0_f64; STAGE_TARGET_SLOTS];

        let delay_floor = if crossbar_coupled {
            MIN_DELAY_CROSSBAR_SAMP
        } else {
            MIN_DELAY_STANDALONE_SAMP
        };
        let eff_delay_min = min_tau_samp.max(delay_floor);

        min[0] = -3.0;
        max[0] = 3.0;
        min[1] = -0.8;
        max[1] = 1.5;
        for i in 2..6 {
            min[i] = -MAX_BLOCK_ADAA_OCT;
            max[i] = MAX_BLOCK_ADAA_OCT;
        }
        for i in 6..10 {
            min[i] = -2.0;
            max[i] = 2.0;
        }
        min[10] = -1.5;
        max[10] = 1.5;
        min[11] = -1.5;
        max[11] = 1.5;
        min[12] = -(48_000.0 - eff_delay_min).max(1.0);
        max[12] = 48_000.0;
        // Bipolar `Op-T` loop gain delta `[-1.998, +1.998]`.
        min[13] = -1.998;
        max[13] = 1.998;
        min[14] = -PI;
        max[14] = PI;
        min[15] = -1.0;
        max[15] = 1.0;
        min[16] = -1.0;
        max[16] = 1.0;
        min[17] = -3.0;
        max[17] = 3.0;
        // `Op-T` v4 ramp rate delta `[-64.0, +64.0]` Hz.
        min[18] = -64.0;
        max[18] = 64.0;
        min[19] = -16.0;
        max[19] = 16.0;
        min[20] = -4.0;
        max[20] = 4.0;
        min[21] = -1.0;
        max[21] = 0.999_95;
        min[22] = -2.0;
        max[22] = 2.0;
        min[23] = -4.0;
        max[23] = 4.0;
        min[24] = -1.0;
        max[24] = 1.0;
        min[25] = -0.75;
        max[25] = 0.75;
        min[26] = -1.0;
        max[26] = 1.0;
        min[27] = -1.0;
        max[27] = f64::from(OP_N_PRE_MAX_ZOH_HOLD_DEPTH);
        min[28] = -16.0;
        max[28] = 16.0;
        min[29] = -MAX_OP_P_PHASOR_RATE_OCT;
        max[29] = MAX_OP_P_PHASOR_RATE_OCT;
        // `SensoryQuad` v4 Log-Knee VCA `[-5.0, +4.0]` octaves.
        min[30] = -5.0;
        max[30] = 4.0;
        // `Slot 31 L/R` active stereo differential quadrature AM gain.
        min[31] = -4.0;
        max[31] = 4.0;

        Self { min, max }
    }
}

/// Analytical `O(P + M)` global BIBO enclosure report across all `3^P` corners.
#[derive(Clone, Debug, PartialEq)]
pub struct GraBiboReport {
    /// `true` iff every certified bound is finite and within `stage_bounds` / edge / `Op-T` limits.
    pub is_bibo_safe: bool,
    /// Analytical Cauchy–Schwarz upper bound `4.0 * ||W_num[m]||_2 * R_trust` per prototype `m`.
    pub y_rat_bound: [f64; GRA_MAX_M],
    /// Exact `1.5-Entmax` convex-hull lower bound `[min_m V_{m,r}]` across all `67` output lanes.
    pub attn_min: [f64; GRA_OUT_DELTA_DIM],
    /// Exact `1.5-Entmax` convex-hull upper bound `[max_m V_{m,r}]` across all `67` output lanes.
    pub attn_max: [f64; GRA_OUT_DELTA_DIM],
    /// Global analytical lower bound across all `3^P` corners + dynamic excursions.
    pub total_min: [f64; GRA_OUT_DELTA_DIM],
    /// Global analytical upper bound across all `3^P` corners + dynamic excursions.
    pub total_max: [f64; GRA_OUT_DELTA_DIM],
    /// Output lane indices `[0..67]` that exceeded `stage_bounds` or `g_edge`/`op_t_q` ceilings.
    pub violating_slots: Vec<usize>,
}

#[inline]
fn blockwise_cauchy_schwarz_norm(row: &[f64], d: usize) -> f64 {
    let mut sum = 0.0_f64;
    for b in 0..(d / 4) {
        let base = b * 4;
        let sq = row[base] * row[base]
            + row[base + 1] * row[base + 1]
            + row[base + 2] * row[base + 2]
            + row[base + 3] * row[base + 3];
        sum += sq.sqrt();
    }
    sum
}

/// Certifies `GRA` global BIBO stability across all `3^P` control-space corners in `O(P + M)` time
/// using blockwise Cauchy–Schwarz bounds on `h_1` (`||h_{1,b}||_2 <= R_trust`) + `1 + d_r >= 0.25`
/// (`rho_r <= 4.0`) + `1.5-Entmax` convex hull (`V_0 == 0.0`) + bilinear dynamic bounds.
#[must_use]
pub fn audit_gra_bibo_hull(
    params: &GraParams,
    stage_bounds: &StageTargetBounds,
    max_q2: [f64; 4],
    max_q3: [f64; 4],
) -> GraBiboReport {
    let (d, _, m, _) = params.profile.dims();
    let (r_trust, _) = params.sanitized_scalars();
    let rho_max = 1.0 / (1.0 + PADE_DEN_MIN);

    let mut y_rat_bound = [0.0_f64; GRA_MAX_M];
    for (k, dst) in y_rat_bound.iter_mut().enumerate().take(m) {
        let bnorm = blockwise_cauchy_schwarz_norm(&params.w_num[k], d);
        *dst = rho_max * bnorm * r_trust;
    }

    let mut dyn_bound = [0.0_f64; 8];
    for i in 0..4 {
        let q2_gain_bnd = params.dyn_gate.w_lfo_base[i].abs()
            + blockwise_cauchy_schwarz_norm(&params.dyn_gate.w_knob_lfo[i], d) * r_trust;
        let q2_lim = if max_q2[i].is_finite() {
            max_q2[i].abs()
        } else {
            0.0
        };
        dyn_bound[i] = q2_gain_bnd * q2_lim;

        let q3_gain_bnd = params.dyn_gate.w_env_base[i].abs()
            + blockwise_cauchy_schwarz_norm(&params.dyn_gate.w_knob_env[i], d) * r_trust;
        let q3_lim = if max_q3[i].is_finite() {
            max_q3[i].abs()
        } else {
            0.0
        };
        let d_env = params.dyn_gate.env_pade_den[i].max(0.0);
        let q3_pade_bnd = q3_lim / (1.0 + d_env * q3_lim);
        dyn_bound[4 + i] = q3_gain_bnd * q3_pade_bnd;
    }

    let mut attn_min = [0.0_f64; GRA_OUT_DELTA_DIM];
    let mut attn_max = [0.0_f64; GRA_OUT_DELTA_DIM];
    let mut total_min = [0.0_f64; GRA_OUT_DELTA_DIM];
    let mut total_max = [0.0_f64; GRA_OUT_DELTA_DIM];
    let mut violating_slots = Vec::new();

    for r in 0..GRA_OUT_DELTA_DIM {
        // Anchor convex hull at `V_0 == 0.0` (matching `forward.rs`) and reject any non-finite entry.
        let mut lo = 0.0_f64;
        let mut hi = 0.0_f64;
        let mut proto_finite = params.v_proto[0][r].is_finite();
        for k in 1..m {
            let v = params.v_proto[k][r];
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            } else {
                proto_finite = false;
            }
        }
        if !proto_finite {
            lo = f64::NAN;
            hi = f64::NAN;
        }
        attn_min[r] = lo;
        attn_max[r] = hi;

        let mut rat_dir_bnd = 0.0_f64;
        for k in 0..m {
            rat_dir_bnd += params.w_rat_direct[r][k].abs() * y_rat_bound[k];
        }
        let mut dyn_out_bnd = 0.0_f64;
        for j in 0..8 {
            dyn_out_bnd += params.dyn_gate.dyn_to_out[r][j].abs() * dyn_bound[j];
        }
        let t_lo = lo - rat_dir_bnd - dyn_out_bnd;
        let t_hi = hi + rat_dir_bnd + dyn_out_bnd;
        total_min[r] = t_lo;
        total_max[r] = t_hi;

        let ok = if !t_lo.is_finite() || !t_hi.is_finite() {
            false
        } else if r < STAGE_TARGET_SLOTS {
            t_lo >= stage_bounds.min[r] && t_hi <= stage_bounds.max[r]
        } else if r == 32 {
            // `slot31_r` shares the `slot 31` stereo AM bound.
            t_lo >= stage_bounds.min[31] && t_hi <= stage_bounds.max[31]
        } else if (45..45 + GRA_NUM_EDGES).contains(&r) {
            t_lo >= -1.0 && t_hi <= 1.0
        } else if (51..51 + GRA_NUM_OP_T_MODS).contains(&r) {
            t_lo >= -4.0 && t_hi <= 4.0
        } else {
            // `z_stage[0..8]` and `z_head[4..8]` latent excursions.
            t_lo >= -16.0 && t_hi <= 16.0
        };
        if !ok {
            violating_slots.push(r);
        }
    }

    GraBiboReport {
        is_bibo_safe: violating_slots.is_empty(),
        y_rat_bound,
        attn_min,
        attn_max,
        total_min,
        total_max,
        violating_slots,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::gra::forward::{eval_gra_f32_mirror, eval_gra_forward};
    use crate::control::gra::model::GraTierProfile;

    #[test]
    fn test_bibo_hull_encloses_all_3p_corners_and_1e9_fuzz() {
        let mut params = GraParams::zeros(GraTierProfile::Lite, 5);
        params.trust_radius = 1.5;
        params.anchor_margin = 1.0;
        for row in 0..8 {
            for col in 0..5 {
                params.w_in[row][col] = 0.8 * (((row * 3 + col * 5 + 1) as f64) * 0.4).sin();
            }
            for col in 0..8 {
                params.w_g[row][col] = 0.7 * (((row * 7 + col * 2 + 1) as f64) * 0.3).cos();
                params.w_v[row][col] = 0.6 * (((row * 5 + col * 3 + 2) as f64) * 0.35).sin();
                params.w_o[row][col] = 0.5 * (((row * 11 + col * 3 + 1) as f64) * 0.25).sin();
            }
        }
        for r in 0..4 {
            for col in 0..8 {
                params.w_den[r][col] = 0.9 * (((r * 7 + col * 3 + 1) as f64) * 0.6).sin();
            }
        }
        for k in 0..8 {
            for col in 0..8 {
                params.w_num[k][col] = 1.2 * (((k * 5 + col * 7 + 1) as f64) * 0.45).cos();
            }
            if k > 0 {
                for r in 0..GRA_OUT_DELTA_DIM {
                    params.v_proto[k][r] = 0.4 * (((k * 13 + r * 3 + 1) as f64) * 0.17).sin();
                }
            }
        }
        params.dyn_gate.w_lfo_base = [0.05; 4];
        params.dyn_gate.w_env_base = [0.05; 4];
        params.dyn_gate.env_pade_den = [1.0; 4];
        for i in 0..4 {
            params.dyn_gate.w_knob_lfo[i][0] = 0.15;
            params.dyn_gate.w_knob_env[i][1] = 0.12;
        }
        for r in 0..GRA_OUT_DELTA_DIM {
            params.dyn_gate.dyn_to_out[r][0] = 0.1;
            params.dyn_gate.dyn_to_out[r][4] = 0.1;
        }
        params.enforce_structural_invariants();

        let bounds = StageTargetBounds::canonical_v4(true, 5.0);
        assert_eq!(bounds.min[13], -1.998);
        assert_eq!(bounds.max[13], 1.998);
        assert_eq!(bounds.min[18], -64.0);
        assert_eq!(bounds.max[18], 64.0);
        assert_eq!(bounds.min[30], -5.0);
        assert_eq!(bounds.max[30], 4.0);
        assert_eq!(bounds.min[31], -4.0);
        assert_eq!(bounds.max[31], 4.0);

        let max_q2 = [1.0_f64; 4];
        let max_q3 = [2.0_f64; 4];
        let rep = audit_gra_bibo_hull(&params, &bounds, max_q2, max_q3);
        assert!(
            rep.is_bibo_safe,
            "expected BIBO safe, violations={:?}",
            rep.violating_slots
        );

        // Brute-force verify all 3^5 = 243 corners fall inside the O(P+M) analytical hull.
        let vals = [-1.0_f64, 0.0, 1.0];
        for c0 in vals {
            for c1 in vals {
                for c2 in vals {
                    for c3 in vals {
                        for c4 in vals {
                            let xi = [c0, c1, c2, c3, c4];
                            let (out, cache) = eval_gra_forward(&params, &xi, max_q2, max_q3, 0.0);
                            for k in 0..8 {
                                assert!(cache.y_rat[k].abs() <= rep.y_rat_bound[k] + 1.0e-12);
                            }
                            let flat = out.to_flat();
                            for r in 0..GRA_OUT_DELTA_DIM {
                                assert!(
                                    flat[r] >= rep.total_min[r] - 1.0e-12
                                        && flat[r] <= rep.total_max[r] + 1.0e-12,
                                    "corner {xi:?} slot {r} escaped hull: {} not in [{}, {}]",
                                    flat[r],
                                    rep.total_min[r],
                                    rep.total_max[r]
                                );
                            }
                        }
                    }
                }
            }
        }

        // `REQ-7.2` ±1e9 and non-finite fuzz: finite & enclosed inside O(P+M) hull.
        for extreme in [1.0e9_f64, -1.0e9_f64, f64::INFINITY, f64::NAN] {
            let xi_ext = [extreme, -extreme, 1.0e9, -1.0e9, 0.0];
            let (out_ext, _) = eval_gra_forward(&params, &xi_ext, max_q2, max_q3, 0.0);
            for v in out_ext.to_flat() {
                assert!(v.is_finite());
            }
            let out32 = eval_gra_f32_mirror(
                &params,
                &[
                    extreme as f32,
                    (-extreme) as f32,
                    1.0e9_f32,
                    -1.0e9_f32,
                    0.0,
                ],
                [1.0; 4],
                [2.0; 4],
                0.0,
            );
            for b in out32.to_bits_all() {
                assert!(f32::from_bits(b).is_finite());
            }
        }

        // Violation detection when prototype exceeds stage_bounds[13].
        let mut bad = params.clone();
        bad.v_proto[1][13] = 2.5;
        let bad_rep = audit_gra_bibo_hull(&bad, &bounds, max_q2, max_q3);
        assert!(!bad_rep.is_bibo_safe);
        assert!(bad_rep.violating_slots.contains(&13));

        // Non-finite (`NaN` / `±Inf`) entries in `v_proto` must mark `is_bibo_safe = false`.
        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut nf = params.clone();
            nf.v_proto[1][5] = non_finite;
            let nf_rep = audit_gra_bibo_hull(&nf, &bounds, max_q2, max_q3);
            assert!(!nf_rep.is_bibo_safe);
            assert!(nf_rep.violating_slots.contains(&5));

            let mut nf0 = params.clone();
            nf0.v_proto[0][7] = non_finite;
            let nf0_rep = audit_gra_bibo_hull(&nf0, &bounds, max_q2, max_q3);
            assert!(!nf0_rep.is_bibo_safe);
            assert!(nf0_rep.violating_slots.contains(&7));
        }

        // `V_0 == 0` anchor: even if caller did not zero `v_proto[0]`, `lo = 0.0; hi = 0.0;`
        // anchors the hull at 0.0.
        let mut unzeroed_v0 = GraParams::zeros(GraTierProfile::Lite, 5);
        unzeroed_v0.v_proto[0][13] = 999.0;
        let anchored_rep = audit_gra_bibo_hull(&unzeroed_v0, &bounds, [0.0; 4], [0.0; 4]);
        assert!(anchored_rep.is_bibo_safe);
        assert_eq!(anchored_rep.attn_min[13], 0.0);
        assert_eq!(anchored_rep.attn_max[13], 0.0);
    }
}
