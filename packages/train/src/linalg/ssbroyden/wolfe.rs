//! Safeguarded Nocedal & Wright (Algorithm 3.5 / 3.6) Strong Wolfe zoom line search
//! (`c1 = 1e-4`, `c2 = 0.9`, max `8` evals) for `SsBroyden` / `L-SSBroyden`.

use crate::error::TrainError;

use super::super::coords::{ScaledCoordSpace, dot_simd};

pub(super) struct WolfeTrialOut {
    pub x: Vec<f64>,
    pub jac: Vec<f64>,
    pub grad_u: Vec<f64>,
    pub loss: f64,
    pub g_dot_d: f64,
}

pub(super) struct WolfeSpec {
    pub f0: f64,
    pub g_dot_d0: f64,
    pub c1: f64,
    pub c2: f64,
    pub alpha_init: f64,
    pub alpha_max: f64,
    pub max_evals: usize,
}

pub(super) struct WolfeOutcome {
    pub accepted: Option<(f64, WolfeTrialOut)>,
    pub evals_used: usize,
}

fn eval_step<F>(
    space: &ScaledCoordSpace<'_>,
    u0: &[f64],
    dir: &[f64],
    alpha: f64,
    gx_scratch: &mut [f64],
    eval: &mut F,
) -> Result<Option<WolfeTrialOut>, TrainError>
where
    F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
{
    let dim = space.dim();
    let cand_u: Vec<f64> = u0
        .iter()
        .zip(dir)
        .map(|(&ui, &di)| ui + alpha * di)
        .collect();
    let mut x = vec![0.0_f64; dim];
    let mut jac = vec![0.0_f64; dim];
    space.decode(&cand_u, &mut x, &mut jac);
    let loss = eval(&x, Some(gx_scratch))?;
    if !loss.is_finite() || !gx_scratch.iter().all(|v| v.is_finite()) {
        return Ok(None);
    }
    let mut grad_u = vec![0.0_f64; dim];
    space.pullback_grad(gx_scratch, &jac, &mut grad_u);
    let g_dot_d = dot_simd(&grad_u, dir);
    if !g_dot_d.is_finite() {
        return Ok(None);
    }
    Ok(Some(WolfeTrialOut {
        x,
        jac,
        grad_u,
        loss,
        g_dot_d,
    }))
}

#[inline]
fn safeguarded_interp(a: f64, fa: f64, ga: f64, b: f64, fb: f64) -> f64 {
    let lo = a.min(b);
    let hi = a.max(b);
    let span = hi - lo;
    let d1 = b - a;
    let denom = 2.0 * (fb - fa - ga * d1);
    if fa.is_finite() && fb.is_finite() && ga.is_finite() && d1.abs() > 1.0e-15 && denom > 1.0e-15 {
        let cand = a - (ga * d1 * d1) / denom;
        if cand.is_finite() {
            let margin = 0.02 * span;
            return cand.clamp(lo + margin, hi - margin);
        }
    }
    0.5 * (a + b)
}

fn update_best(best: &mut Option<(f64, WolfeTrialOut)>, alpha: f64, trial: WolfeTrialOut) {
    let take = match best {
        None => true,
        Some((_, prev)) => trial.loss < prev.loss,
    };
    if take {
        *best = Some((alpha, trial));
    }
}

pub(super) fn strong_wolfe_line_search<F>(
    space: &ScaledCoordSpace<'_>,
    u0: &[f64],
    dir: &[f64],
    spec: &WolfeSpec,
    gx_scratch: &mut [f64],
    eval: &mut F,
) -> Result<WolfeOutcome, TrainError>
where
    F: FnMut(&[f64], Option<&mut [f64]>) -> Result<f64, TrainError>,
{
    let budget = spec.max_evals.max(1);
    let mut used = 0_usize;
    let mut best_strict: Option<(f64, WolfeTrialOut)> = None;
    let curvature_lim = spec.c2 * (-spec.g_dot_d0);

    let mut a_prev = 0.0_f64;
    let mut f_prev = spec.f0;
    let mut g_prev = spec.g_dot_d0;
    let mut a_cur = spec.alpha_init.clamp(1.0e-12, spec.alpha_max);

    let mut zoom_bracket: Option<(f64, f64, f64, f64, f64, f64)> = None;
    for iter in 0..budget {
        used += 1;
        let Some(trial) = eval_step(space, u0, dir, a_cur, gx_scratch, eval)? else {
            if a_cur > a_prev {
                a_cur = 0.5 * (a_prev + a_cur);
                continue;
            }
            break;
        };
        let f_cur = trial.loss;
        let g_cur = trial.g_dot_d;
        if f_cur < spec.f0 {
            update_best(
                &mut best_strict,
                a_cur,
                WolfeTrialOut {
                    x: trial.x.clone(),
                    jac: trial.jac.clone(),
                    grad_u: trial.grad_u.clone(),
                    loss: trial.loss,
                    g_dot_d: trial.g_dot_d,
                },
            );
        }
        if f_cur > spec.f0 + spec.c1 * a_cur * spec.g_dot_d0 || (iter > 0 && f_cur >= f_prev) {
            zoom_bracket = Some((a_prev, f_prev, g_prev, a_cur, f_cur, g_cur));
            break;
        }
        if g_cur.abs() <= curvature_lim {
            return Ok(WolfeOutcome {
                accepted: Some((a_cur, trial)),
                evals_used: used,
            });
        }
        if g_cur >= 0.0 {
            zoom_bracket = Some((a_cur, f_cur, g_cur, a_prev, f_prev, g_prev));
            break;
        }
        a_prev = a_cur;
        f_prev = f_cur;
        g_prev = g_cur;
        if (a_cur - spec.alpha_max).abs() <= 1.0e-12 {
            break;
        }
        a_cur = (2.0 * a_cur).min(spec.alpha_max);
    }

    if let Some((mut alo, mut flo, mut glo, mut ahi, mut fhi, _ghi)) = zoom_bracket {
        while used < budget {
            if (ahi - alo).abs() <= 1.0e-12 * ahi.abs().max(1.0) {
                break;
            }
            let aj = safeguarded_interp(alo, flo, glo, ahi, fhi);
            used += 1;
            let Some(trial) = eval_step(space, u0, dir, aj, gx_scratch, eval)? else {
                ahi = aj;
                fhi = f64::INFINITY;
                continue;
            };
            let fj = trial.loss;
            let gj = trial.g_dot_d;
            if fj < spec.f0 {
                update_best(
                    &mut best_strict,
                    aj,
                    WolfeTrialOut {
                        x: trial.x.clone(),
                        jac: trial.jac.clone(),
                        grad_u: trial.grad_u.clone(),
                        loss: trial.loss,
                        g_dot_d: trial.g_dot_d,
                    },
                );
            }
            if fj > spec.f0 + spec.c1 * aj * spec.g_dot_d0 || fj >= flo {
                ahi = aj;
                fhi = fj;
            } else {
                if gj.abs() <= curvature_lim {
                    return Ok(WolfeOutcome {
                        accepted: Some((aj, trial)),
                        evals_used: used,
                    });
                }
                if gj * (ahi - alo) >= 0.0 {
                    ahi = alo;
                    fhi = flo;
                }
                alo = aj;
                flo = fj;
                glo = gj;
            }
        }
    }

    Ok(WolfeOutcome {
        accepted: best_strict,
        evals_used: used,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safeguarded_interp_and_nonfinite_zoom_fallback() {
        // Finite quadratic interpolation stays safely within (lo, hi).
        let q = safeguarded_interp(0.0, 1.0, -2.0, 1.0, 0.5);
        assert!(q > 0.02 && q < 0.98);
        // Non-finite hi (zoom fallback after eval_step returns None) cleanly bisects.
        assert_eq!(safeguarded_interp(0.2, 1.0, -0.5, 0.8, f64::INFINITY), 0.5);
    }
}
