//! Deterministic CPU reference implementation of `(1, num_candidates, 1)` `inner_opt_step`
//! matching WGSL `inner_opt_step` (`AdamW` + `grad_clip_norm`, `phi_k = 0` L-SSBroyden with
//! `m <= 32` & `tau_k` self-scaling, and `ste_fp16` across transforms `0..=5`)
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use dnam_kernels::{
    IEEE754_F32_ABS_MASK, IEEE754_F32_EXP_BIAS, IEEE754_F32_MANTISSA_BITS, IEEE754_F32_ONE_BITS,
    canonicalize_zero_f32_bits, f32_to_fp16_u32, flush_subnormal_f32_bits, fp16_u32_to_f32,
};

use crate::error::TrainError;
use crate::gpu::abi::{
    COORD_TRANSFORM_ASINH_BOX_NONNEG, COORD_TRANSFORM_ASINH_BOX_SYM, COORD_TRANSFORM_BOX,
    COORD_TRANSFORM_COMPLEMENT_LOG, COORD_TRANSFORM_FREE, COORD_TRANSFORM_LOG_BOX,
    INNER_OPT_MODE_DIAG_PRECOND, INNER_OPT_MODE_LINE_SEARCH_TRIAL, INNER_OPT_MODE_LSSBROYDEN,
    INNER_OPT_MODE_RESET_STATE, INNER_OPT_MODE_STE_FP16, LSSBROYDEN_MAX_HISTORY_M,
    STE_FP16_META_BIT,
};
use crate::runner::types::{InnerOptInput, InnerOptOutput};

const U_SATURATION_LIMIT_F32: f32 = 24.0;
const LSSBROYDEN_CURVATURE_EPS: f32 = 1.0e-10;
const TAU_K_MIN: f32 = 1.0e-3;
const TAU_K_MAX: f32 = 1.0e3;
const MIN_H0_DENOM: f32 = 1.0e-12;
const H0_DIAG_MIN: f32 = 1.0e-12;
const H0_DIAG_MAX: f32 = 1.0e12;
const ADAMW_BETA1: f32 = 0.9;

#[inline(always)]
fn load_h0_diag(diag_precond: Option<&[f32]>, p: usize, use_diag_precond: bool) -> f32 {
    if !use_diag_precond {
        return 1.0;
    }
    let Some(dp) = diag_precond else {
        return 1.0;
    };
    let raw = flush_subnormal_f32_bits(dp[p.min(dp.len().saturating_sub(1))]);
    if raw.is_finite() && raw > 0.0 {
        clamp_f32_safe(raw, H0_DIAG_MIN, H0_DIAG_MAX)
    } else {
        1.0
    }
}
const ADAMW_BETA2: f32 = 0.999;
const ADAMW_EPS: f32 = 1.0e-8;
const EXP2_C2: f32 = 0.240_221_09;
const EXP2_C3: f32 = 0.055_504_11;
const EXP2_C4: f32 = 0.009_675_892;
const EXP2_C5: f32 = 0.001_333_355_8;

#[inline(always)]
fn flush_zero(x: f32) -> f32 {
    if !x.is_finite() {
        0.0
    } else {
        canonicalize_zero_f32_bits(x)
    }
}

#[inline(always)]
fn clamp_f32_safe(x: f32, lo: f32, hi: f32) -> f32 {
    if !x.is_finite() {
        0.5 * (lo + hi)
    } else {
        flush_zero(x.clamp(lo, hi))
    }
}

#[inline(always)]
fn rcp_pos_f32(d: f32) -> f32 {
    if !d.is_finite() || d <= 0.0 {
        0.0
    } else {
        flush_zero(1.0 / d)
    }
}

#[inline]
fn exp_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return 1.0;
    }
    let y = clamp_f32_safe(x * core::f32::consts::LOG2_E, -100.0, 100.0);
    let k = y.round() as i32;
    let f = y - (k as f32);
    let p = EXP2_C4 + EXP2_C5 * f;
    let p = EXP2_C3 + p * f;
    let p = EXP2_C2 + p * f;
    let p = core::f32::consts::LN_2 + p * f;
    let mant = 1.0 + p * f;
    let exp_bits =
        ((k + (IEEE754_F32_EXP_BIAS as i32)).clamp(1, 254) as u32) << IEEE754_F32_MANTISSA_BITS;
    flush_zero(f32::from_bits(exp_bits) * mant)
}

#[inline]
fn ln_f32(x: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 {
        return -80.0;
    }
    let bits = x.to_bits() & IEEE754_F32_ABS_MASK;
    let mut k = ((bits >> IEEE754_F32_MANTISSA_BITS) as i32) - (IEEE754_F32_EXP_BIAS as i32);
    let mut m = f32::from_bits((bits & 0x007F_FFFF) | IEEE754_F32_ONE_BITS);
    if m >= core::f32::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    let u = (m - 1.0) * rcp_pos_f32(m + 1.0);
    let u2 = u * u;
    let poly = 1.0
        + u2 * (0.333_333_34
            + u2 * (0.2 + u2 * (0.142_857_15 + u2 * (0.111_111_11 + u2 * 0.090_909_09))));
    flush_zero((k as f32) * core::f32::consts::LN_2 + 2.0 * u * poly)
}

#[inline]
fn sigmoid_f32(u: f32) -> f32 {
    if u >= 0.0 {
        rcp_pos_f32(1.0 + exp_f32(-u))
    } else {
        let e = exp_f32(u);
        e * rcp_pos_f32(1.0 + e)
    }
}

#[inline]
fn sinh_f32(z: f32) -> f32 {
    0.5 * (exp_f32(z) - exp_f32(-z))
}

#[inline]
fn asinh_f32(x: f32) -> f32 {
    let ax = x.abs();
    let val = if ax > 1.0e4 {
        ln_f32(2.0 * ax)
    } else if ax < 1.0e-4 {
        ax
    } else {
        ln_f32(ax + (ax * ax + 1.0).sqrt())
    };
    if x < 0.0 { -val } else { val }
}

pub(crate) fn clamp_u_for_meta(u: f32, bound_meta: [f32; 4], min_bound_eps: f32) -> f32 {
    let tid = bound_meta[3].to_bits() & 0xF;
    if tid == COORD_TRANSFORM_FREE {
        if bound_meta[0].is_finite() && bound_meta[1].is_finite() && bound_meta[0] < bound_meta[1] {
            let eps = if min_bound_eps.is_finite() && min_bound_eps > 0.0 {
                min_bound_eps
            } else {
                0.0
            };
            let lo = bound_meta[0] + eps;
            let hi = bound_meta[1] - eps;
            if lo < hi {
                return clamp_f32_safe(u, lo, hi);
            }
            return clamp_f32_safe(u, bound_meta[0], bound_meta[1]);
        }
        clamp_f32_safe(u, -1.0e6, 1.0e6)
    } else {
        clamp_f32_safe(u, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
    }
}

pub(crate) fn apply_ste_fp16_u(u_raw: f32, bound_meta: [f32; 4]) -> f32 {
    let meta = bound_meta[3].to_bits();
    let tid = meta & 0xF;
    if (meta & STE_FP16_META_BIT) == 0 {
        return if tid == COORD_TRANSFORM_FREE {
            u_raw
        } else {
            clamp_f32_safe(u_raw, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
        };
    }
    let uc = if tid == COORD_TRANSFORM_FREE {
        clamp_f32_safe(u_raw, -1.0e6, 1.0e6)
    } else {
        clamp_f32_safe(u_raw, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
    };
    let lo = bound_meta[0];
    let hi = bound_meta[1];
    let scale = bound_meta[2].max(1.0e-12);

    let phys = match tid {
        COORD_TRANSFORM_FREE => uc,
        COORD_TRANSFORM_BOX => lo + (hi - lo) * sigmoid_f32(uc),
        COORD_TRANSFORM_LOG_BOX => {
            let la = ln_f32(lo.max(1.0e-15));
            let lb = ln_f32(hi.max(lo * 1.000_001));
            exp_f32(la + (lb - la) * sigmoid_f32(uc))
        }
        COORD_TRANSFORM_COMPLEMENT_LOG => {
            let lmin = ln_f32(lo.max(1.0e-15));
            let lmax = ln_f32(hi.max(lo * 1.000_001));
            1.0 - exp_f32(lmax + (lmin - lmax) * sigmoid_f32(uc))
        }
        COORD_TRANSFORM_ASINH_BOX_SYM => {
            let v_lim = hi.abs().max(lo.abs()).max(1.0e-12);
            let c = asinh_f32(v_lim * rcp_pos_f32(scale));
            scale * sinh_f32(c * (2.0 * sigmoid_f32(uc) - 1.0))
        }
        COORD_TRANSFORM_ASINH_BOX_NONNEG => {
            let v_lim = hi.max(1.0e-12);
            let c = asinh_f32(v_lim * rcp_pos_f32(scale));
            scale * sinh_f32(c * sigmoid_f32(uc))
        }
        _ => uc,
    };

    let phys_q = fp16_u32_to_f32(f32_to_fp16_u32(phys));
    let u_q = match tid {
        COORD_TRANSFORM_FREE => phys_q,
        COORD_TRANSFORM_BOX => {
            let span = (hi - lo).max(1.0e-12);
            let vc = clamp_f32_safe(phys_q, lo + 1.0e-6 * span, hi - 1.0e-6 * span);
            let p = clamp_f32_safe((vc - lo) * rcp_pos_f32(span), 1.0e-6, 1.0 - 1.0e-6);
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_LOG_BOX => {
            let la = ln_f32(lo.max(1.0e-15));
            let lb = ln_f32(hi.max(lo * 1.000_001));
            let vc = clamp_f32_safe(phys_q, lo * (1.0 + 1.0e-6), hi * (1.0 - 1.0e-6));
            let p = clamp_f32_safe(
                (ln_f32(vc) - la) * rcp_pos_f32((lb - la).max(1.0e-12)),
                1.0e-6,
                1.0 - 1.0e-6,
            );
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_COMPLEMENT_LOG => {
            let c_min = lo.max(1.0e-15);
            let c_max = hi.max(c_min * 1.000_001);
            let lmin = ln_f32(c_min);
            let lmax = ln_f32(c_max);
            let c_val =
                clamp_f32_safe(1.0 - phys_q, c_min * (1.0 + 1.0e-6), c_max * (1.0 - 1.0e-6));
            let p = clamp_f32_safe(
                (lmax - ln_f32(c_val)) * rcp_pos_f32((lmax - lmin).max(1.0e-12)),
                1.0e-6,
                1.0 - 1.0e-6,
            );
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_ASINH_BOX_SYM => {
            let v_lim = hi.abs().max(lo.abs()).max(1.0e-12);
            let inv_s = rcp_pos_f32(scale);
            let c = asinh_f32(v_lim * inv_s).max(1.0e-12);
            let vc = clamp_f32_safe(phys_q, -v_lim, v_lim);
            let t = clamp_f32_safe(
                asinh_f32(vc * inv_s) * rcp_pos_f32(c),
                -(1.0 - 2.0e-6),
                1.0 - 2.0e-6,
            );
            let p = clamp_f32_safe(0.5 * (t + 1.0), 1.0e-6, 1.0 - 1.0e-6);
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        COORD_TRANSFORM_ASINH_BOX_NONNEG => {
            let v_lim = hi.max(1.0e-12);
            let inv_s = rcp_pos_f32(scale);
            let c = asinh_f32(v_lim * inv_s).max(1.0e-12);
            let vc = clamp_f32_safe(phys_q, 0.0, v_lim);
            let p = clamp_f32_safe(asinh_f32(vc * inv_s) * rcp_pos_f32(c), 1.0e-6, 1.0 - 1.0e-6);
            ln_f32(p * rcp_pos_f32(1.0 - p))
        }
        _ => uc,
    };
    if tid == COORD_TRANSFORM_FREE {
        if lo.is_finite() && hi.is_finite() && lo < hi {
            clamp_f32_safe(u_q, lo, hi)
        } else {
            clamp_f32_safe(u_q, -1.0e6, 1.0e6)
        }
    } else {
        clamp_f32_safe(u_q, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
    }
}

fn step_adamw_cand(
    cand: usize,
    input: &InnerOptInput<'_>,
    theta: &mut [f32],
    state: &mut [[f32; 4]],
    dir: &mut [f32],
) {
    let u = &input.uniforms;
    let n = u.num_params as usize;
    let u_base = cand * n;
    let lr = if u.step_size.is_finite() && u.step_size >= 0.0 {
        u.step_size
    } else {
        0.0
    };
    let use_ste = (u.mode_flags & INNER_OPT_MODE_STE_FP16) != 0;
    if (u.mode_flags & INNER_OPT_MODE_RESET_STATE) != 0 {
        for p in 0..n {
            let cur_u = clamp_u_for_meta(theta[u_base + p], input.param_bounds[p], u.min_bound_eps);
            state[u_base + p] = [0.0, 0.0, cur_u, 0.0];
            dir[u_base + p] = 0.0;
        }
        return;
    }

    let g_sq: f32 = (0..n)
        .map(|p| {
            let g = flush_zero(input.grad[u_base + p]);
            g * g
        })
        .sum();
    let clip = if u.grad_clip_norm.is_finite() && u.grad_clip_norm > 0.0 {
        u.grad_clip_norm
    } else {
        0.0
    };
    let clip_scale = if clip > 0.0 && g_sq.is_finite() && g_sq > clip * clip {
        flush_zero(clip / g_sq.sqrt())
    } else {
        1.0
    };

    let t_pow = u.step_idx.saturating_add(1) as i32;
    let inv_1mb1 = rcp_pos_f32((1.0 - ADAMW_BETA1.powi(t_pow)).max(1.0e-7));
    let inv_1mb2 = rcp_pos_f32((1.0 - ADAMW_BETA2.powi(t_pow)).max(1.0e-7));

    for p in 0..n {
        let idx = u_base + p;
        let g = flush_zero(input.grad[idx] * clip_scale);
        let st = state[idx];
        let m_new = flush_zero(ADAMW_BETA1 * st[0] + (1.0 - ADAMW_BETA1) * g);
        let v_new = flush_zero(ADAMW_BETA2 * st[1] + (1.0 - ADAMW_BETA2) * g * g).max(0.0);
        let denom = (v_new * inv_1mb2).sqrt() + ADAMW_EPS;
        let delta = flush_zero(-lr * (m_new * inv_1mb1) * rcp_pos_f32(denom));

        let meta = input.param_bounds[p];
        let u_shadow_base = if use_ste && u.step_idx > 0 {
            st[2]
        } else {
            theta[idx]
        };
        let u_shadow_next = clamp_u_for_meta(u_shadow_base + delta, meta, u.min_bound_eps);
        state[idx] = [m_new, v_new, u_shadow_next, g];
        dir[idx] = delta;
        theta[idx] = if use_ste {
            apply_ste_fp16_u(u_shadow_next, meta)
        } else {
            u_shadow_next
        };
    }
}

fn step_lssbroyden_cand(
    cand: usize,
    input: &InnerOptInput<'_>,
    theta: &mut [f32],
    state: &mut [[f32; 4]],
    dir: &mut [f32],
) {
    let u = &input.uniforms;
    let n = u.num_params as usize;
    let m = (u.history_m as usize).min(LSSBROYDEN_MAX_HISTORY_M);
    let u_base = cand * n;
    let s_base = cand * (m + 1) * n;
    let damping = if u.damping_lambda.is_finite() && u.damping_lambda > 0.0 {
        u.damping_lambda
    } else {
        0.0
    };
    let step_sz = if u.step_size.is_finite() {
        u.step_size
    } else {
        0.0
    };
    let use_ste = (u.mode_flags & INNER_OPT_MODE_STE_FP16) != 0;
    let use_diag_precond = (u.mode_flags & INNER_OPT_MODE_DIAG_PRECOND) != 0;

    if (u.mode_flags & INNER_OPT_MODE_RESET_STATE) != 0 {
        for slot in 0..m {
            for p in 0..n {
                state[s_base + slot * n + p] = [0.0; 4];
            }
        }
        for p in 0..n {
            let cur_u = clamp_u_for_meta(theta[u_base + p], input.param_bounds[p], u.min_bound_eps);
            let cur_g = flush_zero(input.grad[u_base + p]);
            let h0_p = load_h0_diag(input.diag_precond, p, use_diag_precond);
            state[s_base + m * n + p] = [cur_u, cur_g, 0.0, 0.0];
            dir[u_base + p] = flush_zero(-cur_g * h0_p);
        }
        return;
    }

    if (u.mode_flags & INNER_OPT_MODE_LINE_SEARCH_TRIAL) != 0 {
        for p in 0..n {
            let meta = input.param_bounds[p];
            let theta_ref = flush_zero(state[s_base + m * n + p][0]);
            let d = flush_zero(dir[u_base + p]);
            let u_next = clamp_u_for_meta(theta_ref + step_sz * d, meta, u.min_bound_eps);
            theta[u_base + p] = if use_ste {
                apply_ste_fp16_u(u_next, meta)
            } else {
                u_next
            };
        }
        return;
    }

    if u.step_idx > 0 && m > 0 {
        let (mut ys, mut ss, mut yy) = (0.0_f32, 0.0_f32, 0.0_f32);
        for p in 0..n {
            let cur_t = flush_zero(theta[u_base + p]);
            let cur_g = flush_zero(input.grad[u_base + p]);
            let r = state[s_base + m * n + p];
            let s = flush_zero(cur_t - r[0]);
            let y = flush_zero(cur_g - r[1]);
            ys += s * y;
            ss += s * s;
            yy += y * y;
        }
        let write_slot = ((u.step_idx - 1) as usize) % m;
        let valid = ys.is_finite()
            && ss.is_finite()
            && yy.is_finite()
            && ys > LSSBROYDEN_CURVATURE_EPS * (ss.sqrt() * yy.sqrt())
            && ys > 0.0;
        for p in 0..n {
            state[s_base + write_slot * n + p] = if valid {
                let cur_t = flush_zero(theta[u_base + p]);
                let cur_g = flush_zero(input.grad[u_base + p]);
                let r = state[s_base + m * n + p];
                [flush_zero(cur_t - r[0]), flush_zero(cur_g - r[1]), 0.0, 0.0]
            } else {
                [0.0; 4]
            };
        }
    }

    for p in 0..n {
        dir[u_base + p] = flush_zero(input.grad[u_base + p]);
    }
    let used = if u.step_idx == 0 || m == 0 {
        0
    } else {
        (u.step_idx as usize).min(m)
    };
    let mut alphas = vec![0.0_f32; used];
    let mut rhos = vec![0.0_f32; used];
    let mut found_h0 = false;
    let mut h0_scale = rcp_pos_f32(1.0 + damping);

    for age in 0..used {
        let slot = ((u.step_idx as usize) - 1 - age) % m;
        let (mut sy, mut sq, mut yy) = (0.0_f32, 0.0_f32, 0.0_f32);
        for p in 0..n {
            let pair = state[s_base + slot * n + p];
            sy += pair[0] * pair[1];
            sq += pair[0] * dir[u_base + p];
            if !found_h0 {
                let h0_p = load_h0_diag(input.diag_precond, p, use_diag_precond);
                yy += h0_p * pair[1] * pair[1];
            }
        }
        if sy.is_finite() && sy > 0.0 {
            let r = rcp_pos_f32(sy);
            let a = flush_zero(r * sq);
            if !found_h0 {
                let raw_tau = flush_zero(sy * rcp_pos_f32(yy + damping.max(MIN_H0_DENOM)));
                h0_scale = clamp_f32_safe(raw_tau, TAU_K_MIN, TAU_K_MAX);
                found_h0 = true;
            }
            for p in 0..n {
                dir[u_base + p] = flush_zero(dir[u_base + p] - a * state[s_base + slot * n + p][1]);
            }
            alphas[age] = a;
            rhos[age] = r;
        }
    }
    for p in 0..n {
        let h0_p = load_h0_diag(input.diag_precond, p, use_diag_precond);
        dir[u_base + p] = flush_zero(dir[u_base + p] * (h0_scale * h0_p));
    }
    for rev in 0..used {
        let age = used - 1 - rev;
        let slot = ((u.step_idx as usize) - 1 - age) % m;
        let mut yq = 0.0_f32;
        for p in 0..n {
            yq += state[s_base + slot * n + p][1] * dir[u_base + p];
        }
        let coeff = alphas[age] - flush_zero(rhos[age] * yq);
        for p in 0..n {
            dir[u_base + p] = flush_zero(dir[u_base + p] + state[s_base + slot * n + p][0] * coeff);
        }
    }

    let mut gd = 0.0_f32;
    let mut has_nonfinite = false;
    for p in 0..n {
        let q = dir[u_base + p];
        if !q.is_finite() {
            has_nonfinite = true;
        }
        gd += flush_zero(input.grad[u_base + p]) * flush_zero(-q);
    }
    let fallback = !gd.is_finite() || gd >= 0.0 || has_nonfinite;
    let inv_damping = rcp_pos_f32(1.0 + damping);
    let hist_flag = if used > 0 { 1.0 } else { 0.0 };
    for p in 0..n {
        let meta = input.param_bounds[p];
        let cur_t = clamp_u_for_meta(theta[u_base + p], meta, u.min_bound_eps);
        let cur_g = flush_zero(input.grad[u_base + p]);
        let h0_p = load_h0_diag(input.diag_precond, p, use_diag_precond);
        let dir_p = if fallback {
            flush_zero(-cur_g * h0_p * inv_damping)
        } else {
            flush_zero(-dir[u_base + p])
        };
        state[s_base + m * n + p] = [cur_t, cur_g, hist_flag, 0.0];
        dir[u_base + p] = dir_p;
        let u_next = clamp_u_for_meta(cur_t + step_sz * dir_p, meta, u.min_bound_eps);
        theta[u_base + p] = if use_ste {
            apply_ste_fp16_u(u_next, meta)
        } else {
            u_next
        };
    }
}

/// Deterministic CPU reference matching `(1, num_candidates, 1)` GPU `inner_opt_step`.
pub fn run_inner_opt_step_cpu(input: &InnerOptInput<'_>) -> Result<InnerOptOutput, TrainError> {
    let u = &input.uniforms;
    let n = u.num_params as usize;
    let m = u.history_m as usize;
    let num_cands = input.num_candidates;
    if n == 0 || num_cands == 0 || m > LSSBROYDEN_MAX_HISTORY_M {
        return Err(TrainError::InvalidDimensions {
            context: "inner_opt_step",
            expected: LSSBROYDEN_MAX_HISTORY_M,
            actual: m,
        });
    }
    let exp_vec = num_cands * n;
    let is_lssb = (u.mode_flags & INNER_OPT_MODE_LSSBROYDEN) != 0;
    let min_state = if is_lssb {
        num_cands * (m + 1) * n
    } else {
        exp_vec
    };
    if input.theta.len() < exp_vec
        || input.grad.len() < exp_vec
        || input.param_bounds.len() < n
        || input.lbfgs_s_y.len() < min_state
        || input.search_dir.len() < exp_vec
    {
        return Err(TrainError::InvalidDimensions {
            context: "inner_opt_step(buffers)",
            expected: min_state,
            actual: input.lbfgs_s_y.len(),
        });
    }
    if let Some(dp) = input.diag_precond
        && dp.len() < n
    {
        return Err(TrainError::InvalidDimensions {
            context: "inner_opt_step(diag_precond)",
            expected: n,
            actual: dp.len(),
        });
    }

    let mut theta = input.theta[..exp_vec].to_vec();
    let mut state = input.lbfgs_s_y.to_vec();
    let mut dir = input.search_dir[..exp_vec].to_vec();
    for cand in 0..num_cands {
        if is_lssb {
            step_lssbroyden_cand(cand, input, &mut theta, &mut state, &mut dir);
        } else {
            step_adamw_cand(cand, input, &mut theta, &mut state, &mut dir);
        }
    }
    Ok(InnerOptOutput {
        theta,
        lbfgs_s_y: state,
        search_dir: dir,
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::gpu::abi::InnerOptUniforms;
    use crate::linalg::ssbroyden::{SsBroydenConfig, SsBroydenState};

    #[test]
    fn test_diag_precond_rejects_short_slice_and_handles_invalid_entries() {
        let bounds = [[-10.0_f32, 10.0, 1.0, f32::from_bits(COORD_TRANSFORM_FREE)]; 4];
        let err = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: InnerOptUniforms {
                num_params: 4,
                history_m: 2,
                step_idx: 0,
                mode_flags: INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND,
                step_size: 0.5,
                damping_lambda: 1.0e-6,
                min_bound_eps: 0.0,
                grad_clip_norm: 0.0,
            },
            num_candidates: 1,
            theta: &[1.0; 4],
            grad: &[0.5; 4],
            param_bounds: &bounds,
            lbfgs_s_y: &[[0.0; 4]; 12],
            search_dir: &[0.0; 4],
            diag_precond: Some(&[1.0, 2.0]),
        })
        .unwrap_err();
        assert!(matches!(err, TrainError::InvalidDimensions { .. }));

        let sanitized = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: InnerOptUniforms {
                num_params: 4,
                history_m: 2,
                step_idx: 0,
                mode_flags: INNER_OPT_MODE_LSSBROYDEN
                    | INNER_OPT_MODE_DIAG_PRECOND
                    | INNER_OPT_MODE_RESET_STATE,
                step_size: 0.5,
                damping_lambda: 0.0,
                min_bound_eps: 0.0,
                grad_clip_norm: 0.0,
            },
            num_candidates: 1,
            theta: &[1.0; 4],
            grad: &[1.0, 1.0, 1.0, 1.0],
            param_bounds: &bounds,
            lbfgs_s_y: &[[0.0; 4]; 12],
            search_dir: &[0.0; 4],
            diag_precond: Some(&[f32::NAN, -2.0, 1.0e-18, 1.0e18]),
        })
        .unwrap();
        assert_eq!(sanitized.search_dir[0], -1.0);
        assert_eq!(sanitized.search_dir[1], -1.0);
        assert_eq!(sanitized.search_dir[2], -H0_DIAG_MIN);
        assert_eq!(sanitized.search_dir[3], -H0_DIAG_MAX);

        let subnormal = run_inner_opt_step_cpu(&InnerOptInput {
            uniforms: InnerOptUniforms {
                num_params: 4,
                history_m: 2,
                step_idx: 0,
                mode_flags: INNER_OPT_MODE_LSSBROYDEN
                    | INNER_OPT_MODE_DIAG_PRECOND
                    | INNER_OPT_MODE_RESET_STATE,
                step_size: 0.5,
                damping_lambda: 0.0,
                min_bound_eps: 0.0,
                grad_clip_norm: 0.0,
            },
            num_candidates: 1,
            theta: &[1.0; 4],
            grad: &[0.25, 0.25, 0.25, 0.25],
            param_bounds: &bounds,
            lbfgs_s_y: &[[0.0; 4]; 12],
            search_dir: &[0.0; 4],
            diag_precond: Some(&[1.0e-40_f32, -1.0e-40_f32, f32::MIN_POSITIVE, 1.0]),
        })
        .unwrap();
        assert_eq!(subnormal.search_dir[0], -0.25);
        assert_eq!(subnormal.search_dir[1], -0.25);
        assert_eq!(subnormal.search_dir[2], -0.25 * H0_DIAG_MIN);
        assert_eq!(subnormal.search_dir[3], -0.25);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_cpu_inner_opt_diag_precond_matches_ssbroyden_state(
            n in 4_usize..=20,
            exps in proptest::collection::vec(-4.0_f64..2.0_f64, 20),
            w_vals in proptest::collection::vec(0.2_f64..4.0_f64, 20),
            x_vals in proptest::collection::vec(-1.5_f64..1.5_f64, 20),
        ) {
            let h0_f64: Vec<f64> = exps[..n].iter().map(|&e| 10.0_f64.powf(e)).collect();
            let h0_f32: Vec<f32> = h0_f64.iter().map(|&v| v as f32).collect();
            let hess: Vec<f64> = w_vals[..n]
                .iter()
                .zip(&h0_f64)
                .map(|(&w, &h0)| w / h0.max(1.0))
                .collect();
            let bounds = vec![[-1.0e6_f32, 1.0e6, 1.0, f32::from_bits(COORD_TRANSFORM_FREE)]; n];

            let m = 8_usize;
            let cfg = SsBroydenConfig {
                history_m: m,
                dense_dim_threshold: 0,
                tau_min: f64::from(TAU_K_MIN),
                tau_max: f64::from(TAU_K_MAX),
                damping_lambda: 1.0e-12,
                curvature_tol: f64::from(LSSBROYDEN_CURVATURE_EPS),
                ..SsBroydenConfig::default()
            };
            let mut ssb = SsBroydenState::new(n, &cfg, Some(&h0_f64));
            let mut x_ssb: Vec<f64> = x_vals[..n].to_vec();
            let mut x_cpu: Vec<f32> = x_ssb.iter().map(|&v| v as f32).collect();
            let mut sy_cpu = vec![[0.0_f32; 4]; (m + 1) * n];
            let mut dir_cpu = vec![0.0_f32; n];
            let mut prev_x_ssb = x_ssb.clone();
            let mut prev_g_ssb = vec![0.0_f64; n];

            for step in 0..4_u32 {
                let g_ssb: Vec<f64> = x_ssb.iter().zip(&hess).map(|(&x, &w)| w * x).collect();
                let g_cpu: Vec<f32> = x_cpu.iter().zip(&hess).map(|(&x, &w)| (w as f32) * x).collect();
                if step > 0 {
                    let s: Vec<f64> = x_ssb.iter().zip(&prev_x_ssb).map(|(a, b)| a - b).collect();
                    let y: Vec<f64> = g_ssb.iter().zip(&prev_g_ssb).map(|(a, b)| a - b).collect();
                    let _ = ssb.update(&s, &y, &cfg);
                }
                let mut dir_ssb = vec![0.0_f64; n];
                ssb.compute_direction(&g_ssb, &h0_f64, &cfg, &mut dir_ssb);

                let out = run_inner_opt_step_cpu(&InnerOptInput {
                    uniforms: InnerOptUniforms {
                        num_params: n as u32,
                        history_m: m as u32,
                        step_idx: step,
                        mode_flags: INNER_OPT_MODE_LSSBROYDEN | INNER_OPT_MODE_DIAG_PRECOND,
                        step_size: 0.1,
                        damping_lambda: 1.0e-12,
                        min_bound_eps: 0.0,
                        grad_clip_norm: 0.0,
                    },
                    num_candidates: 1,
                    theta: &x_cpu,
                    grad: &g_cpu,
                    param_bounds: &bounds,
                    lbfgs_s_y: &sy_cpu,
                    search_dir: &dir_cpu,
                    diag_precond: Some(&h0_f32),
                })
                .unwrap();

                for p in 0..n {
                    let tol = 2.0e-4 * dir_ssb[p].abs().max(1.0);
                    prop_assert!(
                        (f64::from(out.search_dir[p]) - dir_ssb[p]).abs() <= tol,
                        "step={step} p={p}: cpu={} ssb={}",
                        out.search_dir[p],
                        dir_ssb[p]
                    );
                }

                prev_x_ssb.clone_from(&x_ssb);
                prev_g_ssb.clone_from(&g_ssb);
                for p in 0..n {
                    x_ssb[p] += 0.1 * dir_ssb[p];
                }
                x_cpu = out.theta;
                sy_cpu = out.lbfgs_s_y;
                dir_cpu = out.search_dir;
            }
        }
    }
}
