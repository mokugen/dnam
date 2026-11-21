#![forbid(unsafe_code)]

//! Single-workgroup `(1, 1, 1)` strided global projected `m <= 8` L-BFGS optimizer step with
//! Armijo backtracking trial support (`inner_opt_step`) over arbitrary `num_params`, operating
//! strictly within `2,176 B` workgroup SRAM (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::abi::{InnerOptUniforms, LBFGS_MAX_HISTORY_M, MAX_WORKGROUP_SRAM_BYTES};
use crate::reduce::{
    WG_THREADS, clamp_f32_safe, clamp_idx, flush_zero, is_finite_bits, rcp_pos_f32,
    reduce_sum_f32_256, reduce_sum_f32_pair_256, wg_barrier,
};
use core::mem::size_of;
use spirv_std::glam::{UVec3, Vec2, Vec4};

/// Workgroup `scratch_q` capacity (`256 × f32 = 1,024 B`).
pub const INNER_OPT_Q_FLOATS: usize = WG_THREADS;

/// Workgroup `dot_scratch` capacity (`256 × f32 = 1,024 B`).
pub const INNER_OPT_DOT_FLOATS: usize = WG_THREADS;

/// Workgroup `lbfgs_scalars` capacity (`32 × f32 = 128 B`: `alpha[0..8]`, `rho[0..8]`, `sy[0..8]`, status).
pub const INNER_OPT_SCALAR_FLOATS: usize = 32;

/// Total `inner_opt_step` workgroup SRAM footprint (`2,176 B <= 15,872 B`).
pub const INNER_OPT_SRAM_BYTES: usize =
    (INNER_OPT_Q_FLOATS + INNER_OPT_DOT_FLOATS + INNER_OPT_SCALAR_FLOATS) * size_of::<f32>();

const _: () = assert!(LBFGS_MAX_HISTORY_M == 8);
const _: () = assert!(INNER_OPT_SRAM_BYTES == 2_176);
const _: () = assert!(INNER_OPT_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);

/// Curvature positivity threshold `s·y > 1e-12 * (||s||^2 + ||y||^2)`.
const LBFGS_CURVATURE_EPS: f32 = 1.0e-12;

/// Minimum denominator floor in `H0` scaling (`yy + damping`).
const MIN_H0_DENOM: f32 = 1.0e-12;

/// Division-free `a % m` via bounded shift-and-subtract (`m >= 1`).
#[inline(always)]
fn mod_u32_divfree(a: u32, m: u32) -> usize {
    if m <= 1 {
        return 0;
    }
    let mut rem = a;
    let mut sub = m;
    let mut count = 0_u32;
    while sub <= (rem >> 1) {
        sub <<= 1;
        count += 1;
    }
    let mut step = 0_u32;
    while step <= count {
        if rem >= sub {
            rem -= sub;
        }
        sub >>= 1;
        step += 1;
    }
    rem as usize
}

#[inline(always)]
fn clamp_theta_with_bounds(candidate: f32, bounds: Vec2, min_eps: f32) -> f32 {
    let lo_raw = if is_finite_bits(bounds.x) {
        bounds.x
    } else {
        -1.0e6
    };
    let hi_raw = if is_finite_bits(bounds.y) {
        bounds.y
    } else {
        1.0e6
    };
    let lo_b = if lo_raw <= hi_raw { lo_raw } else { hi_raw };
    let hi_b = if lo_raw <= hi_raw { hi_raw } else { lo_raw };
    let lo_eff = lo_b + min_eps;
    let hi_eff = hi_b - min_eps;
    if lo_eff <= hi_eff {
        clamp_f32_safe(candidate, lo_eff, hi_eff)
    } else {
        flush_zero(0.5 * (lo_b + hi_b))
    }
}

/// Records the latest strided secant pair `(s_k, y_k)` in `lbfgs_s_y` when curvature holds.
#[inline]
#[allow(clippy::too_many_arguments)]
fn update_lbfgs_history_slot(
    lid: usize,
    sg_lid: u32,
    step_idx: u32,
    num_params: usize,
    m: usize,
    theta: &[f32],
    grad: &[f32],
    lbfgs_s_y: &mut [Vec4],
    scratch_q: &mut [f32; INNER_OPT_Q_FLOATS],
    dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
) {
    let mut ys_local = 0.0_f32;
    let mut ss_local = 0.0_f32;
    let mut yy_local = 0.0_f32;
    let mut p = lid;
    while p < num_params {
        let cur_t = flush_zero(theta[clamp_idx(p, theta.len())]);
        let cur_g = flush_zero(grad[clamp_idx(p, grad.len())]);
        let ref_entry = lbfgs_s_y[clamp_idx(m * num_params + p, lbfgs_s_y.len())];
        let s = flush_zero(cur_t - ref_entry.x);
        let y = flush_zero(cur_g - ref_entry.y);
        ys_local += s * y;
        ss_local += s * s;
        yy_local += y * y;
        p += WG_THREADS;
    }

    let ys = reduce_sum_f32_256(dot_scratch, lid, ys_local, sg_lid);
    let (ss, yy) = reduce_sum_f32_pair_256(scratch_q, dot_scratch, lid, ss_local, yy_local, sg_lid);

    let write_slot = mod_u32_divfree(step_idx - 1, m as u32);
    let valid_curv = is_finite_bits(ys)
        && is_finite_bits(ss)
        && is_finite_bits(yy)
        && ys > LBFGS_CURVATURE_EPS * (ss + yy)
        && ys > 0.0;

    let mut p = lid;
    while p < num_params {
        let dst = write_slot * num_params + p;
        let entry = if valid_curv {
            let cur_t = flush_zero(theta[clamp_idx(p, theta.len())]);
            let cur_g = flush_zero(grad[clamp_idx(p, grad.len())]);
            let ref_entry = lbfgs_s_y[clamp_idx(m * num_params + p, lbfgs_s_y.len())];
            Vec4::new(
                flush_zero(cur_t - ref_entry.x),
                flush_zero(cur_g - ref_entry.y),
                0.0,
                0.0,
            )
        } else {
            Vec4::ZERO
        };
        lbfgs_s_y[clamp_idx(dst, lbfgs_s_y.len())] = entry;
        p += WG_THREADS;
    }
    wg_barrier();
}

/// Computes global strided projected `m <= 8` L-BFGS direction in `search_dir` and updates
/// `lbfgs_s_y` and `theta` on `line_search_iter == 0`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_lbfgs_direction_and_step(
    lid: usize,
    sg_lid: u32,
    uniforms: &InnerOptUniforms,
    theta: &mut [f32],
    grad: &[f32],
    param_bounds: &[Vec2],
    lbfgs_s_y: &mut [Vec4],
    search_dir: &mut [f32],
    scratch_q: &mut [f32; INNER_OPT_Q_FLOATS],
    dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
    lbfgs_scalars: &mut [f32; INNER_OPT_SCALAR_FLOATS],
) {
    let num_params = uniforms.num_params as usize;
    let m = (uniforms.history_m as usize).min(LBFGS_MAX_HISTORY_M);
    let m_u32 = m as u32;
    let damping = if is_finite_bits(uniforms.damping_lambda) && uniforms.damping_lambda > 0.0 {
        uniforms.damping_lambda
    } else {
        0.0
    };
    let min_eps = if is_finite_bits(uniforms.min_bound_eps) && uniforms.min_bound_eps >= 0.0 {
        uniforms.min_bound_eps
    } else {
        0.0
    };
    let step_sz = if is_finite_bits(uniforms.step_size) {
        uniforms.step_size
    } else {
        0.0
    };

    if uniforms.step_idx > 0 && m > 0 {
        update_lbfgs_history_slot(
            lid,
            sg_lid,
            uniforms.step_idx,
            num_params,
            m,
            theta,
            grad,
            lbfgs_s_y,
            scratch_q,
            dot_scratch,
        );
    }

    let mut p = lid;
    while p < num_params {
        search_dir[clamp_idx(p, search_dir.len())] = flush_zero(grad[clamp_idx(p, grad.len())]);
        p += WG_THREADS;
    }
    if lid < INNER_OPT_SCALAR_FLOATS {
        lbfgs_scalars[lid] = 0.0;
    }
    wg_barrier();

    let used = if uniforms.step_idx == 0 || m == 0 {
        0
    } else {
        (uniforms.step_idx as usize).min(m)
    };

    let mut found_h0 = false;
    let mut h0_scale = rcp_pos_f32(1.0 + damping);

    for age in 0..used {
        let hist_slot = mod_u32_divfree(uniforms.step_idx - 1 - (age as u32), m_u32);
        let mut sy_local = 0.0_f32;
        let mut sq_local = 0.0_f32;
        let mut yy_local = 0.0_f32;
        let mut p = lid;
        while p < num_params {
            let pair = lbfgs_s_y[clamp_idx(hist_slot * num_params + p, lbfgs_s_y.len())];
            let q_p = search_dir[clamp_idx(p, search_dir.len())];
            sy_local += pair.x * pair.y;
            sq_local += pair.x * q_p;
            if !found_h0 {
                yy_local += pair.y * pair.y;
            }
            p += WG_THREADS;
        }

        let (sy_k, sq_k) =
            reduce_sum_f32_pair_256(scratch_q, dot_scratch, lid, sy_local, sq_local, sg_lid);
        let yy_k = if !found_h0 {
            reduce_sum_f32_256(dot_scratch, lid, yy_local, sg_lid)
        } else {
            0.0
        };

        let (rho_k, alpha_k) = if is_finite_bits(sy_k) && sy_k > 0.0 {
            let r = rcp_pos_f32(sy_k);
            let a = flush_zero(r * sq_k);
            if !found_h0 {
                let denom = yy_k
                    + if damping > MIN_H0_DENOM {
                        damping
                    } else {
                        MIN_H0_DENOM
                    };
                h0_scale = flush_zero(sy_k * rcp_pos_f32(denom));
                found_h0 = true;
            }
            let mut p = lid;
            while p < num_params {
                let y_p = lbfgs_s_y[clamp_idx(hist_slot * num_params + p, lbfgs_s_y.len())].y;
                let idx = clamp_idx(p, search_dir.len());
                search_dir[idx] = flush_zero(search_dir[idx] - a * y_p);
                p += WG_THREADS;
            }
            (r, a)
        } else {
            (0.0, 0.0)
        };

        if lid == 0 {
            lbfgs_scalars[age] = alpha_k;
            lbfgs_scalars[8 + age] = rho_k;
            lbfgs_scalars[16 + age] = sy_k;
        }
        wg_barrier();
    }

    let mut p = lid;
    while p < num_params {
        let idx = clamp_idx(p, search_dir.len());
        search_dir[idx] = flush_zero(search_dir[idx] * h0_scale);
        p += WG_THREADS;
    }
    if lid == 0 {
        lbfgs_scalars[24] = h0_scale;
        lbfgs_scalars[26] = used as f32;
    }
    wg_barrier();

    for rev in 0..used {
        let age = used - 1 - rev;
        let hist_slot = mod_u32_divfree(uniforms.step_idx - 1 - (age as u32), m_u32);
        let mut yq_local = 0.0_f32;
        let mut p = lid;
        while p < num_params {
            let y_p = lbfgs_s_y[clamp_idx(hist_slot * num_params + p, lbfgs_s_y.len())].y;
            let q_p = search_dir[clamp_idx(p, search_dir.len())];
            yq_local += y_p * q_p;
            p += WG_THREADS;
        }
        let yq_k = reduce_sum_f32_256(dot_scratch, lid, yq_local, sg_lid);
        let alpha_k = lbfgs_scalars[age];
        let rho_k = lbfgs_scalars[8 + age];
        let beta_k = flush_zero(rho_k * yq_k);
        let coeff = alpha_k - beta_k;
        let mut p = lid;
        while p < num_params {
            let s_p = lbfgs_s_y[clamp_idx(hist_slot * num_params + p, lbfgs_s_y.len())].x;
            let idx = clamp_idx(p, search_dir.len());
            search_dir[idx] = flush_zero(search_dir[idx] + s_p * coeff);
            p += WG_THREADS;
        }
        wg_barrier();
    }

    let mut gd_local = 0.0_f32;
    let mut bad_dir_local = 0.0_f32;
    let mut p = lid;
    while p < num_params {
        let cur_g = flush_zero(grad[clamp_idx(p, grad.len())]);
        let raw_q = search_dir[clamp_idx(p, search_dir.len())];
        if !is_finite_bits(raw_q) {
            bad_dir_local += 1.0;
        }
        let dir_p = flush_zero(-raw_q);
        gd_local += cur_g * dir_p;
        p += WG_THREADS;
    }
    let (gd, bad_dir) =
        reduce_sum_f32_pair_256(scratch_q, dot_scratch, lid, gd_local, bad_dir_local, sg_lid);
    if lid == 0 {
        lbfgs_scalars[25] = gd;
    }
    wg_barrier();

    let fallback = !is_finite_bits(gd) || gd >= 0.0 || bad_dir > 0.0;
    let inv_damping = rcp_pos_f32(1.0 + damping);
    let hist_flag = if used > 0 { 1.0 } else { 0.0 };
    let mut p = lid;
    while p < num_params {
        let cur_t = flush_zero(theta[clamp_idx(p, theta.len())]);
        let cur_g = flush_zero(grad[clamp_idx(p, grad.len())]);
        let bounds = param_bounds[clamp_idx(p, param_bounds.len())];
        let raw_q = search_dir[clamp_idx(p, search_dir.len())];
        let dir_p = if fallback {
            flush_zero(-cur_g * inv_damping)
        } else {
            flush_zero(-raw_q)
        };
        let ref_dst = m * num_params + p;
        lbfgs_s_y[clamp_idx(ref_dst, lbfgs_s_y.len())] = Vec4::new(cur_t, cur_g, hist_flag, 0.0);
        search_dir[clamp_idx(p, search_dir.len())] = dir_p;
        theta[clamp_idx(p, theta.len())] =
            clamp_theta_with_bounds(cur_t + step_sz * dir_p, bounds, min_eps);
        p += WG_THREADS;
    }
}

/// Executes single-workgroup `(1, 1, 1)` strided `inner_opt_step` across `256` threads
/// (`2,176 B` SRAM, zero `FDIV`/`FSQRT`/`IDIV`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn inner_opt_step_impl(
    wg_id: UVec3,
    lid_u32: u32,
    sg_lid: u32,
    uniforms: &InnerOptUniforms,
    theta: &mut [f32],
    grad: &[f32],
    param_bounds: &[Vec2],
    lbfgs_s_y: &mut [Vec4],
    search_dir: &mut [f32],
    scratch_q: &mut [f32; INNER_OPT_Q_FLOATS],
    dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
    lbfgs_scalars: &mut [f32; INNER_OPT_SCALAR_FLOATS],
) {
    if wg_id.x != 0 {
        return;
    }
    let lid = (lid_u32 as usize) & 255;
    let num_params = uniforms.num_params as usize;

    if uniforms.line_search_iter == 0 {
        run_lbfgs_direction_and_step(
            lid,
            sg_lid,
            uniforms,
            theta,
            grad,
            param_bounds,
            lbfgs_s_y,
            search_dir,
            scratch_q,
            dot_scratch,
            lbfgs_scalars,
        );
    } else {
        let m = (uniforms.history_m as usize).min(LBFGS_MAX_HISTORY_M);
        let min_eps = if is_finite_bits(uniforms.min_bound_eps) && uniforms.min_bound_eps >= 0.0 {
            uniforms.min_bound_eps
        } else {
            0.0
        };
        let step_sz = if is_finite_bits(uniforms.step_size) {
            uniforms.step_size
        } else {
            0.0
        };
        let mut p = lid;
        while p < num_params {
            let theta_ref = flush_zero(lbfgs_s_y[clamp_idx(m * num_params + p, lbfgs_s_y.len())].x);
            let dir = flush_zero(search_dir[clamp_idx(p, search_dir.len())]);
            let bounds = param_bounds[clamp_idx(p, param_bounds.len())];
            theta[clamp_idx(p, theta.len())] =
                clamp_theta_with_bounds(theta_ref + step_sz * dir, bounds, min_eps);
            p += WG_THREADS;
        }
    }
}
