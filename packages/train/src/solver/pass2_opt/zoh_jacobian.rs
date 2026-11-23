//! Exact reverse-time ZOH rate-rescaling Jacobians matching `dnam_vm::math::rescale_tpt_svf_g_zoh`
//! and `dnam_vm::rescale_paralru_lane` (`ref[dsp.zoh.op-m-poles]`, `ref[dsp.zoh.core-paralru]`).

use core::f64::consts::{FRAC_1_PI, PI};

use dnam_kernels::{MAX_PARALRU_R_F32, MIN_TPT_G};

use super::super::model::CoreQuadF64;

pub(super) struct LaneZohParams {
    pub svf_g: f64,
    pub lru_alpha_rel: f64,
    pub lru_alpha_atk: f64,
    pub lru_kappa: f64,
    pub rho_core: f64,
}

pub(super) struct LaneZohGrads<'a> {
    pub g_svf_g: &'a mut f64,
    pub g_rel: &'a mut f64,
    pub g_atk: &'a mut f64,
    pub g_kap: &'a mut f64,
}

/// Propagates adjoints with respect to rate-rescaled `(svf_g_eff, lru_alpha_rel_eff,
/// lru_alpha_atk_eff, lru_kappa_eff)` back into `48 kHz` reference parameters for one
/// `CoreQuad` (or `SensoryQuad` lanes) under execution ratio `rho_core = f_ref / f_core`.
pub(super) fn apply_lane_zoh_jacobian(params: LaneZohParams, grads: LaneZohGrads<'_>) {
    let LaneZohParams {
        svf_g,
        lru_alpha_rel,
        lru_alpha_atk,
        lru_kappa,
        rho_core,
    } = params;
    let LaneZohGrads {
        g_svf_g,
        g_rel,
        g_atk,
        g_kap,
    } = grads;
    let rho = if rho_core.is_finite() && rho_core > 0.0 {
        rho_core.clamp(0.125, 8.0)
    } else {
        1.0
    };
    if (rho - 1.0).abs() <= 1.0e-12 {
        return;
    }

    // 1. Bilinear prewarped TPT-SVF g ZOH sensitivity.
    let g_min = f64::from(MIN_TPT_G);
    if svf_g.is_finite() && svf_g > g_min && svf_g < 64.0 {
        let r_eff = svf_g.atan() * FRAC_1_PI * rho;
        if r_eff > 1.0e-5 && r_eff < 0.495 {
            let g_eff = (PI * r_eff).tan();
            let dg_eff_dg = rho * (1.0 + g_eff * g_eff) / (1.0 + svf_g * svf_g);
            *g_svf_g *= dg_eff_dg;
        } else {
            *g_svf_g = 0.0;
        }
    } else {
        *g_svf_g = 0.0;
    }

    // 2. Selective-Duhem ParaLRU exact 3x3 Jacobian wrt reference (alpha_rel, alpha_atk, kappa).
    let r_max = f64::from(MAX_PARALRU_R_F32);
    let r = if lru_alpha_rel.is_finite() {
        lru_alpha_rel.clamp(0.0, r_max)
    } else {
        0.0
    };
    let atk = if lru_alpha_atk.is_finite() {
        lru_alpha_atk.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let kap = if lru_kappa.is_finite() {
        lru_kappa.clamp(0.0, 1.0)
    } else {
        0.0
    };

    let beta = 1.0 - atk;
    let kb = kap * beta;
    let a_atk = r * (1.0 - kb);
    let one_m_a = 1.0 - a_atk;

    let r_out = r.powf(rho);
    let dr_out_dr = if r > 0.0 && lru_alpha_rel > 0.0 && lru_alpha_rel < r_max && r_out < r_max {
        rho * r.powf(rho - 1.0)
    } else {
        0.0
    };

    let (ratio, dratio_da) = if one_m_a.abs() <= 1.0e-12 {
        (rho, 0.5 * rho * (rho - 1.0))
    } else if a_atk > 0.0 {
        let ratio = (1.0 - a_atk.powf(rho)) / one_m_a;
        let dratio_da = (ratio - rho * a_atk.powf(rho - 1.0)) / one_m_a;
        (ratio, dratio_da)
    } else {
        (1.0 / one_m_a, 1.0 / (one_m_a * one_m_a))
    };

    let beta_prime_raw = beta * ratio;
    let beta_prime = beta_prime_raw.clamp(0.0, 1.0);
    let bp_unclamped = beta_prime_raw > 0.0 && beta_prime_raw < 1.0;

    let dbeta_datk = -1.0;
    let dkb_datk = -kap;
    let dkb_dkap = beta;
    let da_dr = 1.0 - kb;
    let da_datk = -r * dkb_datk;
    let da_dkap = -r * dkb_dkap;

    let dbp_dr = if bp_unclamped && lru_alpha_rel > 0.0 && lru_alpha_rel < r_max {
        beta * dratio_da * da_dr
    } else {
        0.0
    };
    let dbp_datk = if bp_unclamped && lru_alpha_atk > 0.0 && lru_alpha_atk < 1.0 {
        dbeta_datk * ratio + beta * dratio_da * da_datk
    } else {
        0.0
    };
    let dbp_dkap = if bp_unclamped && lru_kappa > 0.0 && lru_kappa < 1.0 {
        beta * dratio_da * da_dkap
    } else {
        0.0
    };

    let (dkap_dr, dkap_datk, dkap_dkap) = if beta_prime > 1.0e-12 {
        let one_m_kb = 1.0 - kb;
        let num = 1.0 - one_m_kb.powf(rho);
        let kap_out_raw = num / beta_prime;
        if kap_out_raw > 0.0 && kap_out_raw < 1.0 {
            let inv_bp = 1.0 / beta_prime;
            let dnum_dkb = if one_m_kb > 0.0 {
                rho * one_m_kb.powf(rho - 1.0)
            } else {
                0.0
            };
            let dnum_datk = if lru_alpha_atk > 0.0 && lru_alpha_atk < 1.0 {
                dnum_dkb * dkb_datk
            } else {
                0.0
            };
            let dnum_dkap = if lru_kappa > 0.0 && lru_kappa < 1.0 {
                dnum_dkb * dkb_dkap
            } else {
                0.0
            };
            (
                -kap_out_raw * dbp_dr * inv_bp,
                (dnum_datk - kap_out_raw * dbp_datk) * inv_bp,
                (dnum_dkap - kap_out_raw * dbp_dkap) * inv_bp,
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    } else if lru_kappa > 0.0 && lru_kappa < 1.0 {
        (0.0, 0.0, 1.0)
    } else {
        (0.0, 0.0, 0.0)
    };

    let (gr_out, gatk_out, gkap_out) = (*g_rel, *g_atk, *g_kap);
    *g_rel = gr_out * dr_out_dr + gatk_out * (-dbp_dr) + gkap_out * dkap_dr;
    *g_atk = gatk_out * (-dbp_datk) + gkap_out * dkap_datk;
    *g_kap = gatk_out * (-dbp_dkap) + gkap_out * dkap_dkap;
}

/// Applies [`apply_lane_zoh_jacobian`] across all 4 stereo lanes of a [`CoreQuadF64`].
pub(super) fn apply_core_quad_zoh_jacobian(
    quad: &CoreQuadF64,
    rho_core: f64,
    g_svf_g: &mut [f64; 4],
    g_rel: &mut [f64; 4],
    g_atk: &mut [f64; 4],
    g_kap: &mut [f64; 4],
) {
    for l in 0..4 {
        apply_lane_zoh_jacobian(
            LaneZohParams {
                svf_g: quad.svf_g[l],
                lru_alpha_rel: quad.lru_alpha_rel[l],
                lru_alpha_atk: quad.lru_alpha_atk[l],
                lru_kappa: quad.lru_kappa[l],
                rho_core,
            },
            LaneZohGrads {
                g_svf_g: &mut g_svf_g[l],
                g_rel: &mut g_rel[l],
                g_atk: &mut g_atk[l],
                g_kap: &mut g_kap[l],
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_vm::math::{rescale_real_pole_zoh, rescale_tpt_svf_g_zoh};
    use dnam_vm::rescale_paralru_lane;

    #[test]
    fn test_zoh_jacobian_fd_parity_for_core_and_sensory() {
        let q = CoreQuadF64 {
            svf_g: [0.08, 0.25, 0.72, 1.6],
            lru_alpha_rel: [0.992, 0.975, 0.93, 0.85],
            lru_alpha_atk: [0.18, 0.35, 0.62, 0.82],
            lru_kappa: [0.22, 0.45, 0.71, 0.90],
            ..CoreQuadF64::default()
        };

        let w_g = [1.3, -0.7, 0.4, 2.1];
        let w_r = [-0.5, 1.1, 0.8, -1.4];
        let w_a = [0.9, -1.2, 0.3, 0.6];
        let w_k = [-0.4, 0.6, -1.1, 0.9];

        let eval_fn = |m: &CoreQuadF64, rho: f64| -> f64 {
            let mr = m.rescaled_for_rho(rho);
            (0..4)
                .map(|l| {
                    w_g[l] * mr.svf_g[l]
                        + w_r[l] * mr.lru_alpha_rel[l]
                        + w_a[l] * mr.lru_alpha_atk[l]
                        + w_k[l] * mr.lru_kappa[l]
                })
                .sum()
        };

        for &rho in &[0.5_f64, 2.0_f64] {
            let mut g_g = w_g;
            let mut g_r = w_r;
            let mut g_a = w_a;
            let mut g_k = w_k;
            apply_core_quad_zoh_jacobian(&q, rho, &mut g_g, &mut g_r, &mut g_a, &mut g_k);

            let eps = 1.0e-3_f64;
            for l in 0..4 {
                let fd = |setter: fn(&mut CoreQuadF64, usize, f64), base: f64| -> f64 {
                    let mut qp = q;
                    let mut qm = q;
                    setter(&mut qp, l, base + eps);
                    setter(&mut qm, l, base - eps);
                    (eval_fn(&qp, rho) - eval_fn(&qm, rho)) / (2.0 * eps)
                };
                let fg = fd(|m, idx, v| m.svf_g[idx] = v, q.svf_g[l]);
                let fr = fd(|m, idx, v| m.lru_alpha_rel[idx] = v, q.lru_alpha_rel[l]);
                let fa = fd(|m, idx, v| m.lru_alpha_atk[idx] = v, q.lru_alpha_atk[l]);
                let fk = fd(|m, idx, v| m.lru_kappa[idx] = v, q.lru_kappa[l]);

                // CoreQuad::rescaled_for_rho internally casts via f32; eps = 1e-3 balances O(eps^2) & f32 rounding.
                for (name, got, want) in [
                    ("svf_g", g_g[l], fg),
                    ("rel", g_r[l], fr),
                    ("atk", g_a[l], fa),
                    ("kap", g_k[l], fk),
                ] {
                    let tol = 5.0e-4 * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tol,
                        "ZOH jacobian mismatch rho={rho} lane={l} {name}: got={got} want={want}"
                    );
                }
            }
        }
        let _ = (
            rescale_tpt_svf_g_zoh,
            rescale_real_pole_zoh,
            rescale_paralru_lane,
        );
    }
}
