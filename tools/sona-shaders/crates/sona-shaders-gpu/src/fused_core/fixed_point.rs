#![forbid(unsafe_code)]

//! Optional bounded fixed-point forward scan (`<= 6` iters, `<= 1e-6` error with exact
//! sequential fallback) executing strictly inside `scan_scratch` (`6,144 B`) & `aux_scratch`
//! (`ref[train.pipeline.webgpu-scans-and-ozaki]`):
//! - Selective-Duhem `ParaLRU`: Backtracking-damped `1D` `DEER` (`alpha in {1.0, 0.5, 0.25}`).

use crate::abi::{
    FORWARD_AUX_SCRATCH_FLOATS, FORWARD_LANE_BUF_FLOATS, FORWARD_SCAN_SCRATCH_FLOATS,
    MAX_WORKGROUP_SRAM_BYTES,
};
use crate::core_math::{
    QuadLaneParams, eval_duhem_step_terms, flush_zero, linear_forward_exclusive_scan_128_pair,
    step_paralru_duhem_scalar,
};
use crate::reduce::{clamp_f32_safe, is_finite_bits, rcp_pos_f32, wg_barrier};
use dnam_kernels::{
    GpuVec4, KernelVec4, LinearScanElem, MAX_PARALRU_R_F32, step_paralru_duhem_v4_vec4,
};

const SUBTILE_FRAMES: usize = 128;
const FIXED_POINT_MAX_ITERS: usize = 6;
const FIXED_POINT_TOL: f32 = 5.0e-7;
const NONFINITE_PENALTY: f32 = 1.0e9;
const REDUCE_AUX_BASE: usize = 48;

const _: () = assert!(REDUCE_AUX_BASE + SUBTILE_FRAMES <= FORWARD_AUX_SCRATCH_FLOATS);
const _: () = assert!(FORWARD_SCAN_SCRATCH_FLOATS == 1536);
const _: () = assert!(FORWARD_LANE_BUF_FLOATS == 2048);
const _: () = assert!(
    (FORWARD_SCAN_SCRATCH_FLOATS + FORWARD_LANE_BUF_FLOATS + FORWARD_AUX_SCRATCH_FLOATS) * 4
        <= MAX_WORKGROUP_SRAM_BYTES
);

#[inline(always)]
pub(super) const fn lane_slot(c: usize, f: usize, t: usize) -> usize {
    (c * 4 + f) * SUBTILE_FRAMES + (t & 127)
}

#[inline(always)]
fn rel_err_finite(a: f32, b: f32) -> f32 {
    if is_finite_bits(a) && is_finite_bits(b) {
        let abs_a = a.abs();
        let abs_b = b.abs();
        let max_ab = if abs_a > abs_b { abs_a } else { abs_b };
        let denom = if max_ab > 1.0 { max_ab } else { 1.0 };
        (a - b).abs() * rcp_pos_f32(denom)
    } else {
        NONFINITE_PENALTY
    }
}

/// Reduces a `256`-thread nonnegative max error (paired across `L` `0..128` and `R` `128..256`)
/// inside `aux_scratch[48..176]` (`0` touch on `aux_scratch[0..48]`), broadcasting to all threads.
#[inline]
fn reduce_max_256_in_aux(
    aux_scratch: &mut [f32; FORWARD_AUX_SCRATCH_FLOATS],
    lid: usize,
    val: f32,
) -> f32 {
    let i = lid & 127;
    if lid < SUBTILE_FRAMES {
        aux_scratch[REDUCE_AUX_BASE + i] = val;
    }
    wg_barrier();
    if lid >= SUBTILE_FRAMES {
        let cur = aux_scratch[REDUCE_AUX_BASE + i];
        aux_scratch[REDUCE_AUX_BASE + i] = if val > cur { val } else { cur };
    }
    wg_barrier();

    let mut stride = SUBTILE_FRAMES >> 1;
    while stride > 0 {
        if i < stride && lid < SUBTILE_FRAMES {
            let a = aux_scratch[REDUCE_AUX_BASE + i];
            let b = aux_scratch[REDUCE_AUX_BASE + i + stride];
            aux_scratch[REDUCE_AUX_BASE + i] = if b > a { b } else { a };
        }
        wg_barrier();
        stride >>= 1;
    }
    let out = aux_scratch[REDUCE_AUX_BASE];
    wg_barrier();
    out
}

#[inline(always)]
fn eval_duhem_step_raw_and_jac(tone: f32, bp: f32, v_prev: f32, lp: &QuadLaneParams) -> (f32, f32) {
    let lru_c = lp.to_lru_coeffs();
    let mut v_reg = GpuVec4::splat(v_prev);
    let v_raw = step_paralru_duhem_v4_vec4(
        GpuVec4::splat(lp.rect_sign * tone),
        GpuVec4::splat(lp.two_svf_g * bp),
        &mut v_reg,
        &lru_c,
    )
    .to_array()[0];
    let terms = eval_duhem_step_terms(
        lp.rect_sign * tone,
        bp,
        v_prev,
        lp.alpha_rel,
        lp.alpha_atk,
        lp.kappa,
        lp.gamma_duhem,
        lp.signed_mix,
        lp.two_svf_g,
    );
    let jac = clamp_f32_safe(terms.jac, -MAX_PARALRU_R_F32, MAX_PARALRU_R_F32);
    (v_raw, jac)
}

/// Optional backtracking-damped `1D` `DEER` Selective-Duhem `ParaLRU` solver (`alpha in {1.0, 0.5, 0.25}`,
/// `<= 6` iters, `<= 1e-6` tol) storing `(v_eff_L, v_eff_R)` in `scan_scratch[0..1024]`, with
/// exact sequential `lid in 0..2` fallback on exceedance.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn run_duhem_damped_deer_subtile_lane(
    lid: usize,
    sub_valid_len: usize,
    c: usize,
    lp: &QuadLaneParams,
    scan_scratch: &mut [f32; FORWARD_SCAN_SCRATCH_FLOATS],
    lane_buf: &mut [f32; FORWARD_LANE_BUF_FLOATS],
    aux_scratch: &mut [f32; FORWARD_AUX_SCRATCH_FLOATS],
) {
    let is_r = (lid >> 7) != 0;
    let t = lid & 127;
    let active = t < sub_valid_len;
    let tone_my = if active {
        lane_buf[lane_slot(c, if is_r { 2 } else { 0 }, t)]
    } else {
        0.0
    };
    let bp_my = if active {
        lane_buf[lane_slot(c, if is_r { 3 } else { 1 }, t)]
    } else {
        0.0
    };
    let v_init = aux_scratch[16 + (if is_r { 4 } else { 0 }) + c];

    // Slots `512..768` store `v_cur` (`L: 512..640`, `R: 640..768`);
    // `768..1024` store `v_trial` (`L: 768..896`, `R: 896..1024`).
    let cur_base = if is_r { 640 } else { 512 };
    let trial_base = if is_r { 896 } else { 768 };
    scan_scratch[cur_base + t] = v_init;
    wg_barrier();

    let mut v_cur = v_init;
    let mut converged = false;

    for _iter in 0..FIXED_POINT_MAX_ITERS {
        if !converged {
            let v_prev = if t == 0 {
                v_init
            } else {
                scan_scratch[cur_base + t - 1]
            };
            let (v_f, jac) = if active {
                eval_duhem_step_raw_and_jac(tone_my, bp_my, v_prev, lp)
            } else {
                (0.0, 1.0)
            };
            let err_cur = if active {
                rel_err_finite(v_f, v_cur)
            } else {
                0.0
            };
            let max_err = reduce_max_256_in_aux(aux_scratch, lid, err_cur);
            if max_err <= FIXED_POINT_TOL {
                converged = true;
            } else {
                let elem = if active {
                    LinearScanElem {
                        a: jac,
                        b: v_f - jac * v_prev,
                    }
                } else {
                    LinearScanElem::identity()
                };
                let (v_newton, _v_end) = linear_forward_exclusive_scan_128_pair(
                    scan_scratch,
                    lid,
                    sub_valid_len,
                    elem,
                    v_init,
                );
                let delta = v_newton - v_cur;
                let mut accepted = false;
                let mut alpha = 1.0_f32;
                for bt in 0..3 {
                    if !accepted {
                        let cand = if active {
                            clamp_f32_safe(v_cur + alpha * delta, -1.0e6, 1.0e6)
                        } else {
                            v_init
                        };
                        scan_scratch[trial_base + t] = cand;
                        wg_barrier();
                        let v_prev_tr = if t == 0 {
                            v_init
                        } else {
                            scan_scratch[trial_base + t - 1]
                        };
                        let err_tr = if active {
                            let (vf_tr, _) =
                                eval_duhem_step_raw_and_jac(tone_my, bp_my, v_prev_tr, lp);
                            rel_err_finite(vf_tr, cand)
                        } else {
                            0.0
                        };
                        let max_tr = reduce_max_256_in_aux(aux_scratch, lid, err_tr);
                        if max_tr <= max_err || bt == 2 {
                            v_cur = cand;
                            scan_scratch[cur_base + t] = cand;
                            wg_barrier();
                            if max_tr <= FIXED_POINT_TOL {
                                converged = true;
                            }
                            accepted = true;
                        }
                        alpha *= 0.5;
                    }
                }
            }
        }
    }

    // Final exact fixed-point residual verification across `0..sub_valid_len`.
    let v_prev_chk = if t == 0 {
        v_init
    } else {
        scan_scratch[cur_base + t - 1]
    };
    let (v_f_chk, _) = if active {
        eval_duhem_step_raw_and_jac(tone_my, bp_my, v_prev_chk, lp)
    } else {
        (v_cur, 1.0)
    };
    let final_err = if active {
        rel_err_finite(v_f_chk, v_cur)
    } else {
        0.0
    };
    let max_final = reduce_max_256_in_aux(aux_scratch, lid, final_err);

    if max_final > FIXED_POINT_TOL {
        // Guaranteed `0-ULP` fallback split across `lid < 2` (`L` and `R` independently),
        // storing `v_raw` into dead `bp` slots `(f = 1, 3)` cleanly without disturbing `tone`.
        if lid < 2 {
            let is_r_fb = lid == 1;
            let lru_c = lp.to_lru_coeffs();
            let tone_f = if is_r_fb { 2 } else { 0 };
            let bp_f = tone_f + 1;
            let v_idx = if is_r_fb { 20 + c } else { 16 + c };
            let mut v = aux_scratch[v_idx];
            for k in 0..sub_valid_len {
                let ti = k & 127;
                let tone = lane_buf[lane_slot(c, tone_f, ti)];
                let bp = lane_buf[lane_slot(c, bp_f, ti)];
                let v_raw = step_paralru_duhem_scalar(tone, bp, &mut v, lp, &lru_c);
                lane_buf[lane_slot(c, bp_f, ti)] = v_raw;
            }
            aux_scratch[v_idx] = flush_zero(v);
        }
        wg_barrier();
    } else {
        // Store `(v_raw_L, v_raw_R)` into dead `bp` slots `(f = 1, 3)` before Stage 3 stereo link.
        wg_barrier();
        if lid < SUBTILE_FRAMES && active {
            lane_buf[lane_slot(c, 1, t)] = scan_scratch[512 + t];
            lane_buf[lane_slot(c, 3, t)] = scan_scratch[640 + t];
        }
        if lid == 0 && sub_valid_len > 0 {
            let last = (sub_valid_len - 1) & 127;
            aux_scratch[16 + c] = flush_zero(scan_scratch[512 + last]);
            aux_scratch[20 + c] = flush_zero(scan_scratch[640 + last]);
        }
        wg_barrier();
    }
}
