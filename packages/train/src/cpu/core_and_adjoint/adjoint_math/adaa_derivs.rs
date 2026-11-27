//! Exact analytical derivatives for the odd-only $C^2$ wavefolder on clamped
//! `[-4096.0, 4096.0]` endpoints `(r_prev_safe, r_curr_safe)` matching `tools/sona-shaders`'s
//! `core_math/adaa_deriv.rs` (`ref[dsp.op-n.polyadaa.rail-crossing]`,
//! `ref[dsp.op-n.wavefolder.cubic-c2]`).

use dnam_kernels::{
    GpuVec4, KernelVec4, TINY_CROSSING_EPS, canonicalize_zero_f32_bits, const_f32_pow2,
};

use super::is_finite_f32;

const TWO_POW_64: f32 = const_f32_pow2(64);
const TINY_CROSSING_DENOM_FLOOR: f32 = 1.0e-24;

const _: () = assert!(TINY_CROSSING_EPS > 0.0);
const _: () = assert!(TINY_CROSSING_DENOM_FLOOR > 0.0);

/// Forward value and exact endpoint partial derivatives for the odd-only $C^2$ wavefolder.
#[derive(Copy, Clone)]
pub(super) struct WfOddDerivResult {
    pub wf_odd: f32,
    pub d_wf_da: f32,
    pub d_wf_db: f32,
}

#[inline(always)]
fn eval_wf_odd_interior_and_derivs(lo: f32, hi: f32, max_abs: f32) -> (f32, f32, f32) {
    let m = 0.5 * (hi + lo);
    let d = 0.5 * (hi - lo);
    let m2 = m * m;
    let d2 = d * d;
    let s = m2 + d2;
    let om = 1.0 - s;

    let wf_odd = if max_abs <= 0.0 {
        0.0
    } else {
        canonicalize_zero_f32_bits(m * om * (om * om + 4.0 * m2 * d2))
    };

    let p = m * om;
    let q = om * om + 4.0 * m2 * d2;
    let dp_dm = om - 2.0 * m2;
    let dq_dm = 4.0 * m * (2.0 * d2 - om);
    let dp_dd = -2.0 * m * d;
    let dq_dd = 4.0 * d * (2.0 * m2 - om);
    let d_wf_dm = dp_dm * q + p * dq_dm;
    let d_wf_dd = dp_dd * q + p * dq_dd;

    (wf_odd, 0.5 * (d_wf_dm - d_wf_dd), 0.5 * (d_wf_dm + d_wf_dd))
}

#[inline(always)]
fn sat_wf_odd_point_value(x: f32) -> f32 {
    if x <= -1.0 || x >= 1.0 {
        0.0
    } else {
        let x2 = x * x;
        let om = 1.0 - x2;
        x * om * om * om
    }
}

#[inline(always)]
fn rcp_crossing_dist(dist: f32) -> f32 {
    if dist < TINY_CROSSING_EPS {
        let d_safe = dist.max(TINY_CROSSING_DENOM_FLOOR);
        GpuVec4::splat(d_safe * TWO_POW_64).rcp_pos_nr2().to_array()[0] * TWO_POW_64
    } else {
        GpuVec4::splat(dist).rcp_pos_nr2().to_array()[0]
    }
}

/// Evaluates odd-only $C^2$ wavefolder `wf_odd` and exact analytical endpoint derivatives
/// `(∂wf_odd/∂r_prev_safe, ∂wf_odd/∂r_curr_safe)` on clamped `[-4096.0, 4096.0]` endpoints.
#[inline]
pub(super) fn eval_c2_wf_odd_with_derivs(
    r_prev_safe: f32,
    r_curr_safe: f32,
    tiny_eps: f32,
) -> WfOddDerivResult {
    let a = r_prev_safe;
    let b = r_curr_safe;
    let lo_is_a = a <= b;
    let (lo, hi) = if lo_is_a { (a, b) } else { (b, a) };
    let dist = hi - lo;
    let max_abs = a.abs().max(b.abs());
    let tiny = if is_finite_f32(tiny_eps) && tiny_eps > 0.0 {
        tiny_eps
    } else {
        TINY_CROSSING_EPS
    };

    let (wf_odd, d_wf_dlo, d_wf_dhi) = if lo >= 1.0 || hi <= -1.0 {
        (0.0, 0.0, 0.0)
    } else if lo >= -1.0 && hi <= 1.0 {
        eval_wf_odd_interior_and_derivs(lo, hi, max_abs)
    } else {
        let cl_lo = lo.max(-1.0);
        let cl_hi = hi.min(1.0);
        let (wf_int, d_lo_int, d_hi_int) = eval_wf_odd_interior_and_derivs(cl_lo, cl_hi, max_abs);
        if dist <= tiny {
            let lo_mask = if lo >= -1.0 { 1.0 } else { 0.0 };
            let hi_mask = if hi <= 1.0 { 1.0 } else { 0.0 };
            (wf_int, d_lo_int * lo_mask, d_hi_int * hi_mask)
        } else {
            let inv = rcp_crossing_dist(dist);
            let inside = (cl_hi - cl_lo).max(0.0);
            let wf_val = canonicalize_zero_f32_bits((inside * wf_int) * inv);
            let sat_wf_lo = sat_wf_odd_point_value(lo);
            let sat_wf_hi = sat_wf_odd_point_value(hi);
            (
                wf_val,
                (wf_val - sat_wf_lo) * inv,
                (sat_wf_hi - wf_val) * inv,
            )
        }
    };

    let (d_wf_da, d_wf_db) = if lo_is_a {
        (d_wf_dlo, d_wf_dhi)
    } else {
        (d_wf_dhi, d_wf_dlo)
    };

    WfOddDerivResult {
        wf_odd,
        d_wf_da,
        d_wf_db,
    }
}

#[cfg(test)]
mod tests {
    use dnam_kernels::eval_c2_wavefolder_adaa_vec4;

    use super::*;

    #[test]
    fn test_c2_wavefolder_analytical_endpoint_derivs_match_fd() {
        let eps = 5.0e-4_f32;
        for &(a, b) in &[
            (-0.6, 0.4),
            (-1.4, 0.5),
            (-0.2, 1.3),
            (-1.8, 1.6),
            (0.999_999, 1.000_002),
            (0.5, 0.500_000_1),
        ] {
            let res = eval_c2_wf_odd_with_derivs(a, b, TINY_CROSSING_EPS);
            let k_wf = eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(a), GpuVec4::splat(b))
                .0
                .to_array()[0];
            assert!((res.wf_odd - k_wf).abs() <= 1.0e-6, "a={a} b={b}");
            if (a - b).abs() > 1.0e-3 && (a.abs() - 1.0).abs() > 2.0 * eps {
                let da_fd =
                    (eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(a + eps), GpuVec4::splat(b))
                        .0
                        .to_array()[0]
                        - eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(a - eps), GpuVec4::splat(b))
                            .0
                            .to_array()[0])
                        / (2.0 * eps);
                assert!(
                    (res.d_wf_da - da_fd).abs() <= 5.0e-3,
                    "a={a} b={b} got={} fd={da_fd}",
                    res.d_wf_da
                );
            }
        }
    }
}
