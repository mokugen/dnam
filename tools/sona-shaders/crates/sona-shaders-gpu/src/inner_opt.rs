#![forbid(unsafe_code)]

//! Parallel multi-candidate `(1, num_candidates, 1)` strided unconstrained `u`-space `AdamW`,
//! `phi_k = 0` `L-SSBroyden` (`m <= 32`), and `STE binary16` quantization optimizer step
//! (`inner_opt_step`) operating strictly within `2,560 B` workgroup SRAM
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`).

pub mod ste;

use crate::abi::{
    COORD_TRANSFORM_FREE, INNER_OPT_MODE_LINE_SEARCH_TRIAL, INNER_OPT_MODE_LSSBROYDEN,
    INNER_OPT_MODE_RESET_STATE, INNER_OPT_MODE_STE_FP16, InnerOptUniforms,
    LSSBROYDEN_MAX_HISTORY_M, MAX_WORKGROUP_SRAM_BYTES,
};
use crate::reduce::{
    WG_THREADS, clamp_f32_safe, clamp_idx, flush_zero, is_finite_bits, rcp_pos_f32,
    reduce_sum_f32_256, reduce_sum_f32_pair_256, wg_barrier,
};
use core::mem::size_of;
use spirv_std::glam::{UVec3, Vec4};
use ste::{
    U_SATURATION_LIMIT_F32, apply_ste_fp16_u, pow_f32_u32, rsqrt_exact_pos_f32, sqrt_pos_f32,
};

/// Workgroup `scratch_q` capacity (`256 × f32 = 1,024 B`).
pub const INNER_OPT_Q_FLOATS: usize = WG_THREADS;

/// Workgroup `dot_scratch` capacity (`256 × f32 = 1,024 B`).
pub const INNER_OPT_DOT_FLOATS: usize = WG_THREADS;

/// Workgroup `lbfgs_scalars` capacity (`128 × f32 = 512 B`: `alpha[0..32]`, `rho[32..64]`, `sy[64..96]`, status).
pub const INNER_OPT_SCALAR_FLOATS: usize = 128;

/// Total `inner_opt_step` workgroup SRAM footprint (`2,560 B <= 15,872 B`).
pub const INNER_OPT_SRAM_BYTES: usize =
    (INNER_OPT_Q_FLOATS + INNER_OPT_DOT_FLOATS + INNER_OPT_SCALAR_FLOATS) * size_of::<f32>();

const _: () = assert!(LSSBROYDEN_MAX_HISTORY_M == 32);
const _: () = assert!(INNER_OPT_SRAM_BYTES == 2_560);
const _: () = assert!(INNER_OPT_SRAM_BYTES <= MAX_WORKGROUP_SRAM_BYTES);

const LSSBROYDEN_CURVATURE_EPS: f32 = 1.0e-10;
const TAU_K_MIN: f32 = 1.0e-3;
const TAU_K_MAX: f32 = 1.0e3;
const MIN_H0_DENOM: f32 = 1.0e-12;
const ADAMW_BETA1: f32 = 0.9;
const ADAMW_BETA2: f32 = 0.999;
const ADAMW_EPS: f32 = 1.0e-8;

#[inline(always)]
fn clamp_u_for_meta(u: f32, bound_meta: Vec4, min_bound_eps: f32) -> f32 {
    let transform_id = bound_meta.w.to_bits() & 0xF;
    if transform_id == COORD_TRANSFORM_FREE {
        if is_finite_bits(bound_meta.x)
            && is_finite_bits(bound_meta.y)
            && bound_meta.x < bound_meta.y
        {
            let eps = if is_finite_bits(min_bound_eps) && min_bound_eps > 0.0 {
                min_bound_eps
            } else {
                0.0
            };
            let lo = bound_meta.x + eps;
            let hi = bound_meta.y - eps;
            if lo < hi {
                return clamp_f32_safe(u, lo, hi);
            }
            return clamp_f32_safe(u, bound_meta.x, bound_meta.y);
        }
        clamp_f32_safe(u, -1.0e6, 1.0e6)
    } else {
        clamp_f32_safe(u, -U_SATURATION_LIMIT_F32, U_SATURATION_LIMIT_F32)
    }
}

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

#[inline]
#[allow(clippy::too_many_arguments)]
fn run_adamw_step(
    cand: usize,
    lid: usize,
    sg_lid: u32,
    uniforms: &InnerOptUniforms,
    theta: &mut [f32],
    grad: &[f32],
    param_bounds: &[Vec4],
    opt_state: &mut [Vec4],
    search_dir: &mut [f32],
    dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
) {
    let num_params = uniforms.num_params as usize;
    let u_base = cand * num_params;
    let lr = if is_finite_bits(uniforms.step_size) && uniforms.step_size >= 0.0 {
        uniforms.step_size
    } else {
        0.0
    };
    let use_ste = (uniforms.mode_flags & INNER_OPT_MODE_STE_FP16) != 0;
    let reset_state = (uniforms.mode_flags & INNER_OPT_MODE_RESET_STATE) != 0;

    if reset_state {
        let mut p = lid;
        while p < num_params {
            let cur_u = clamp_u_for_meta(
                theta[clamp_idx(u_base + p, theta.len())],
                param_bounds[clamp_idx(p, param_bounds.len())],
                uniforms.min_bound_eps,
            );
            opt_state[clamp_idx(u_base + p, opt_state.len())] = Vec4::new(0.0, 0.0, cur_u, 0.0);
            search_dir[clamp_idx(u_base + p, search_dir.len())] = 0.0;
            p += WG_THREADS;
        }
        return;
    }

    let mut g_sq_local = 0.0_f32;
    let mut p = lid;
    while p < num_params {
        let g = flush_zero(grad[clamp_idx(u_base + p, grad.len())]);
        g_sq_local += g * g;
        p += WG_THREADS;
    }
    let g_sq = reduce_sum_f32_256(dot_scratch, lid, g_sq_local, sg_lid);
    let clip = if is_finite_bits(uniforms.grad_clip_norm) && uniforms.grad_clip_norm > 0.0 {
        uniforms.grad_clip_norm
    } else {
        0.0
    };
    let clip_scale = if clip > 0.0 && is_finite_bits(g_sq) && g_sq > clip * clip {
        flush_zero(clip * rsqrt_exact_pos_f32(g_sq))
    } else {
        1.0
    };

    let t_pow = if uniforms.step_idx < u32::MAX {
        uniforms.step_idx + 1
    } else {
        u32::MAX
    };
    let inv_1mb1 = rcp_pos_f32((1.0 - pow_f32_u32(ADAMW_BETA1, t_pow)).max(1.0e-7));
    let inv_1mb2 = rcp_pos_f32((1.0 - pow_f32_u32(ADAMW_BETA2, t_pow)).max(1.0e-7));

    let mut p = lid;
    while p < num_params {
        let idx = u_base + p;
        let g = flush_zero(grad[clamp_idx(idx, grad.len())] * clip_scale);
        let st = opt_state[clamp_idx(idx, opt_state.len())];
        let m_new = flush_zero(ADAMW_BETA1 * st.x + (1.0 - ADAMW_BETA1) * g);
        let v_new = flush_zero(ADAMW_BETA2 * st.y + (1.0 - ADAMW_BETA2) * g * g).max(0.0);
        let m_hat = m_new * inv_1mb1;
        let v_hat = v_new * inv_1mb2;
        let denom = sqrt_pos_f32(v_hat) + ADAMW_EPS;
        let delta = flush_zero(-lr * m_hat * rcp_pos_f32(denom));

        let meta = param_bounds[clamp_idx(p, param_bounds.len())];
        let u_shadow_base = if use_ste {
            if uniforms.step_idx == 0 {
                theta[clamp_idx(idx, theta.len())]
            } else {
                st.z
            }
        } else {
            theta[clamp_idx(idx, theta.len())]
        };
        let u_shadow_next = clamp_u_for_meta(u_shadow_base + delta, meta, uniforms.min_bound_eps);
        opt_state[clamp_idx(idx, opt_state.len())] = Vec4::new(m_new, v_new, u_shadow_next, g);
        search_dir[clamp_idx(idx, search_dir.len())] = delta;

        let u_eff = if use_ste {
            apply_ste_fp16_u(u_shadow_next, meta)
        } else {
            u_shadow_next
        };
        theta[clamp_idx(idx, theta.len())] = u_eff;
        p += WG_THREADS;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn update_lssbroyden_history_slot(
    cand: usize,
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
    let u_base = cand * num_params;
    let state_base = cand * (m + 1) * num_params;
    let mut ys_local = 0.0_f32;
    let mut ss_local = 0.0_f32;
    let mut yy_local = 0.0_f32;
    let mut p = lid;
    while p < num_params {
        let cur_t = flush_zero(theta[clamp_idx(u_base + p, theta.len())]);
        let cur_g = flush_zero(grad[clamp_idx(u_base + p, grad.len())]);
        let ref_entry = lbfgs_s_y[clamp_idx(state_base + m * num_params + p, lbfgs_s_y.len())];
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
    let norm_prod = sqrt_pos_f32(ss) * sqrt_pos_f32(yy);
    let valid_curv = is_finite_bits(ys)
        && is_finite_bits(ss)
        && is_finite_bits(yy)
        && ys > LSSBROYDEN_CURVATURE_EPS * norm_prod
        && ys > 0.0;

    let mut p = lid;
    while p < num_params {
        let dst = state_base + write_slot * num_params + p;
        let entry = if valid_curv {
            let cur_t = flush_zero(theta[clamp_idx(u_base + p, theta.len())]);
            let cur_g = flush_zero(grad[clamp_idx(u_base + p, grad.len())]);
            let ref_entry = lbfgs_s_y[clamp_idx(state_base + m * num_params + p, lbfgs_s_y.len())];
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn run_lssbroyden_step(
    cand: usize,
    lid: usize,
    sg_lid: u32,
    uniforms: &InnerOptUniforms,
    theta: &mut [f32],
    grad: &[f32],
    param_bounds: &[Vec4],
    lbfgs_s_y: &mut [Vec4],
    search_dir: &mut [f32],
    scratch_q: &mut [f32; INNER_OPT_Q_FLOATS],
    dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
    lbfgs_scalars: &mut [f32; INNER_OPT_SCALAR_FLOATS],
) {
    let num_params = uniforms.num_params as usize;
    let m = (uniforms.history_m as usize).min(LSSBROYDEN_MAX_HISTORY_M);
    let m_u32 = m as u32;
    let u_base = cand * num_params;
    let state_base = cand * (m + 1) * num_params;

    let damping = if is_finite_bits(uniforms.damping_lambda) && uniforms.damping_lambda > 0.0 {
        uniforms.damping_lambda
    } else {
        0.0
    };
    let step_sz = if is_finite_bits(uniforms.step_size) {
        uniforms.step_size
    } else {
        0.0
    };
    let use_ste = (uniforms.mode_flags & INNER_OPT_MODE_STE_FP16) != 0;
    let reset_state = (uniforms.mode_flags & INNER_OPT_MODE_RESET_STATE) != 0;

    if reset_state {
        for slot in 0..m {
            let mut p = lid;
            while p < num_params {
                lbfgs_s_y[clamp_idx(state_base + slot * num_params + p, lbfgs_s_y.len())] =
                    Vec4::ZERO;
                p += WG_THREADS;
            }
        }
        let mut p = lid;
        while p < num_params {
            let cur_u = clamp_u_for_meta(
                theta[clamp_idx(u_base + p, theta.len())],
                param_bounds[clamp_idx(p, param_bounds.len())],
                uniforms.min_bound_eps,
            );
            let cur_g = flush_zero(grad[clamp_idx(u_base + p, grad.len())]);
            lbfgs_s_y[clamp_idx(state_base + m * num_params + p, lbfgs_s_y.len())] =
                Vec4::new(cur_u, cur_g, 0.0, 0.0);
            search_dir[clamp_idx(u_base + p, search_dir.len())] = flush_zero(-cur_g);
            p += WG_THREADS;
        }
        return;
    }

    if uniforms.step_idx > 0 && m > 0 {
        update_lssbroyden_history_slot(
            cand,
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
        search_dir[clamp_idx(u_base + p, search_dir.len())] =
            flush_zero(grad[clamp_idx(u_base + p, grad.len())]);
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
        let slot_off = state_base + hist_slot * num_params;
        let mut sy_local = 0.0_f32;
        let mut sq_local = 0.0_f32;
        let mut yy_local = 0.0_f32;
        let mut p = lid;
        while p < num_params {
            let pair = lbfgs_s_y[clamp_idx(slot_off + p, lbfgs_s_y.len())];
            let q_p = search_dir[clamp_idx(u_base + p, search_dir.len())];
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
                let denom = yy_k + damping.max(MIN_H0_DENOM);
                let raw_tau = flush_zero(sy_k * rcp_pos_f32(denom));
                h0_scale = clamp_f32_safe(raw_tau, TAU_K_MIN, TAU_K_MAX);
                found_h0 = true;
            }
            let mut p = lid;
            while p < num_params {
                let y_p = lbfgs_s_y[clamp_idx(slot_off + p, lbfgs_s_y.len())].y;
                let idx = clamp_idx(u_base + p, search_dir.len());
                search_dir[idx] = flush_zero(search_dir[idx] - a * y_p);
                p += WG_THREADS;
            }
            (r, a)
        } else {
            (0.0, 0.0)
        };

        if lid == 0 {
            lbfgs_scalars[age] = alpha_k;
            lbfgs_scalars[32 + age] = rho_k;
            lbfgs_scalars[64 + age] = sy_k;
        }
        wg_barrier();
    }

    let mut p = lid;
    while p < num_params {
        let idx = clamp_idx(u_base + p, search_dir.len());
        search_dir[idx] = flush_zero(search_dir[idx] * h0_scale);
        p += WG_THREADS;
    }
    if lid == 0 {
        lbfgs_scalars[96] = h0_scale;
        lbfgs_scalars[98] = used as f32;
    }
    wg_barrier();

    for rev in 0..used {
        let age = used - 1 - rev;
        let hist_slot = mod_u32_divfree(uniforms.step_idx - 1 - (age as u32), m_u32);
        let slot_off = state_base + hist_slot * num_params;
        let mut yq_local = 0.0_f32;
        let mut p = lid;
        while p < num_params {
            let y_p = lbfgs_s_y[clamp_idx(slot_off + p, lbfgs_s_y.len())].y;
            let q_p = search_dir[clamp_idx(u_base + p, search_dir.len())];
            yq_local += y_p * q_p;
            p += WG_THREADS;
        }
        let yq_k = reduce_sum_f32_256(dot_scratch, lid, yq_local, sg_lid);
        let alpha_k = lbfgs_scalars[age];
        let rho_k = lbfgs_scalars[32 + age];
        let beta_k = flush_zero(rho_k * yq_k);
        let coeff = alpha_k - beta_k;
        let mut p = lid;
        while p < num_params {
            let s_p = lbfgs_s_y[clamp_idx(slot_off + p, lbfgs_s_y.len())].x;
            let idx = clamp_idx(u_base + p, search_dir.len());
            search_dir[idx] = flush_zero(search_dir[idx] + s_p * coeff);
            p += WG_THREADS;
        }
        wg_barrier();
    }

    let mut gd_local = 0.0_f32;
    let mut bad_dir_local = 0.0_f32;
    let mut p = lid;
    while p < num_params {
        let cur_g = flush_zero(grad[clamp_idx(u_base + p, grad.len())]);
        let raw_q = search_dir[clamp_idx(u_base + p, search_dir.len())];
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
        lbfgs_scalars[97] = gd;
    }
    wg_barrier();

    let fallback = !is_finite_bits(gd) || gd >= 0.0 || bad_dir > 0.0;
    let inv_damping = rcp_pos_f32(1.0 + damping);
    let hist_flag = if used > 0 { 1.0 } else { 0.0 };
    let mut p = lid;
    while p < num_params {
        let meta = param_bounds[clamp_idx(p, param_bounds.len())];
        let cur_t = clamp_u_for_meta(
            theta[clamp_idx(u_base + p, theta.len())],
            meta,
            uniforms.min_bound_eps,
        );
        let cur_g = flush_zero(grad[clamp_idx(u_base + p, grad.len())]);
        let raw_q = search_dir[clamp_idx(u_base + p, search_dir.len())];
        let dir_p = if fallback {
            flush_zero(-cur_g * inv_damping)
        } else {
            flush_zero(-raw_q)
        };
        let ref_dst = state_base + m * num_params + p;
        lbfgs_s_y[clamp_idx(ref_dst, lbfgs_s_y.len())] = Vec4::new(cur_t, cur_g, hist_flag, 0.0);
        search_dir[clamp_idx(u_base + p, search_dir.len())] = dir_p;

        let u_next = clamp_u_for_meta(cur_t + step_sz * dir_p, meta, uniforms.min_bound_eps);
        theta[clamp_idx(u_base + p, theta.len())] = if use_ste {
            apply_ste_fp16_u(u_next, meta)
        } else {
            u_next
        };
        p += WG_THREADS;
    }
}

/// Executes `(1, num_candidates, 1)` strided `inner_opt_step` across `256` threads
/// (`2,560 B` SRAM, zero `FDIV`/`FSQRT`/`IDIV`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn inner_opt_step_impl(
    wg_id: UVec3,
    lid_u32: u32,
    sg_lid: u32,
    uniforms: &InnerOptUniforms,
    theta: &mut [f32],
    grad: &[f32],
    param_bounds: &[Vec4],
    lbfgs_s_y: &mut [Vec4],
    search_dir: &mut [f32],
    scratch_q: &mut [f32; INNER_OPT_Q_FLOATS],
    dot_scratch: &mut [f32; INNER_OPT_DOT_FLOATS],
    lbfgs_scalars: &mut [f32; INNER_OPT_SCALAR_FLOATS],
) {
    if wg_id.x != 0 {
        return;
    }
    let cand = wg_id.y as usize;
    let lid = (lid_u32 as usize) & 255;
    let num_params = uniforms.num_params as usize;
    if num_params == 0 {
        return;
    }
    let is_lssbroyden = (uniforms.mode_flags & INNER_OPT_MODE_LSSBROYDEN) != 0;
    let is_trial = (uniforms.mode_flags & INNER_OPT_MODE_LINE_SEARCH_TRIAL) != 0;

    if !is_lssbroyden {
        run_adamw_step(
            cand,
            lid,
            sg_lid,
            uniforms,
            theta,
            grad,
            param_bounds,
            lbfgs_s_y,
            search_dir,
            dot_scratch,
        );
    } else if !is_trial {
        run_lssbroyden_step(
            cand,
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
        let m = (uniforms.history_m as usize).min(LSSBROYDEN_MAX_HISTORY_M);
        let u_base = cand * num_params;
        let state_base = cand * (m + 1) * num_params;
        let use_ste = (uniforms.mode_flags & INNER_OPT_MODE_STE_FP16) != 0;
        let step_sz = if is_finite_bits(uniforms.step_size) {
            uniforms.step_size
        } else {
            0.0
        };
        let mut p = lid;
        while p < num_params {
            let meta = param_bounds[clamp_idx(p, param_bounds.len())];
            let theta_ref = flush_zero(
                lbfgs_s_y[clamp_idx(state_base + m * num_params + p, lbfgs_s_y.len())].x,
            );
            let dir = flush_zero(search_dir[clamp_idx(u_base + p, search_dir.len())]);
            let u_next = clamp_u_for_meta(theta_ref + step_sz * dir, meta, uniforms.min_bound_eps);
            theta[clamp_idx(u_base + p, theta.len())] = if use_ste {
                apply_ste_fp16_u(u_next, meta)
            } else {
                u_next
            };
            p += WG_THREADS;
        }
    }
}
