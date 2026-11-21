//! Analytical `PolyADAA1` (`odd`, `even`) and $C^2$ wavefolder (`wf_odd`) endpoint derivatives
//! matching `tools/sona-shaders`'s `adaa_deriv.rs` across plateaus, rail crossings, and tiny
//! crossing segments (`ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.wavefolder.cubic-c2]`).

use dnam_kernels::{
    GpuVec4, KernelVec4, MAX_ADAA_ENDPOINT, ONE_THIRD_F32, TINY_CROSSING_EPS,
    canonicalize_zero_f32_bits, const_f32_pow2,
};

use super::is_finite_f32;

const TWO_THIRDS: f32 = 2.0 / 3.0;
const ONE_FIFTH: f32 = 1.0 / 5.0;
const FOUR_FIFTHS: f32 = 4.0 / 5.0;
const TWO_POW_64: f32 = const_f32_pow2(64);
const TINY_CROSSING_DENOM_FLOOR: f32 = 1.0e-24;

const _: () = assert!(TWO_THIRDS > 0.666 && TWO_THIRDS < 0.667);
const _: () = assert!(FOUR_FIFTHS == 0.8);
const _: () = assert!(TINY_CROSSING_DENOM_FLOOR > 0.0);

#[inline(always)]
pub(super) fn scrub_endpoint(x: f32) -> f32 {
    canonicalize_zero_f32_bits(x).clamp(-MAX_ADAA_ENDPOINT, MAX_ADAA_ENDPOINT)
}

#[derive(Copy, Clone)]
pub(super) struct AdaaWfDeriv {
    pub odd_poly: f32,
    pub wf_odd: f32,
    pub d_odd_da: f32,
    pub d_odd_db: f32,
    pub d_even_da: f32,
    pub d_even_db: f32,
    pub d_wf_da: f32,
    pub d_wf_db: f32,
}

#[inline(always)]
fn eval_interior(lo: f32, hi: f32, max_abs: f32) -> [f32; 9] {
    let m = 0.5 * (hi + lo);
    let d = 0.5 * (hi - lo);
    let (m2, d2) = (m * m, d * d);
    let (m4, d4) = (m2 * m2, d2 * d2);
    let s = m2 + d2;
    let om = 1.0 - s;
    let (odd, wf_odd) = if max_abs <= 0.0 {
        (0.0, 0.0)
    } else {
        (
            canonicalize_zero_f32_bits(m * (1.5 - 0.5 * s)),
            canonicalize_zero_f32_bits(m * om * (om * om + 4.0 * m2 * d2)),
        )
    };
    let even = if max_abs <= 0.0 {
        0.0
    } else {
        canonicalize_zero_f32_bits(
            (m2 + ONE_THIRD_F32 * d2) - ((m4 + 2.0 * m2 * d2) + ONE_FIFTH * d4),
        )
    };
    let d_odd_dm = 1.5 - 1.5 * m2 - 0.5 * d2;
    let d_odd_dd = -m * d;
    let d_even_dm = 2.0 * m * (1.0 - 2.0 * s);
    let d_even_dd = d * (TWO_THIRDS - 4.0 * m2 - FOUR_FIFTHS * d2);
    let p = m * om;
    let q = om * om + 4.0 * m2 * d2;
    let d_wf_dm = (om - 2.0 * m2) * q + p * (4.0 * m * (2.0 * d2 - om));
    let d_wf_dd = (-2.0 * m * d) * q + p * (4.0 * d * (2.0 * m2 - om));
    [
        odd,
        even,
        wf_odd,
        0.5 * (d_odd_dm - d_odd_dd),
        0.5 * (d_odd_dm + d_odd_dd),
        0.5 * (d_even_dm - d_even_dd),
        0.5 * (d_even_dm + d_even_dd),
        0.5 * (d_wf_dm - d_wf_dd),
        0.5 * (d_wf_dm + d_wf_dd),
    ]
}

#[inline(always)]
fn sat_point(x: f32) -> (f32, f32, f32) {
    if x <= -1.0 {
        (-1.0, 0.0, 0.0)
    } else if x >= 1.0 {
        (1.0, 0.0, 0.0)
    } else {
        let x2 = x * x;
        let om = 1.0 - x2;
        (x * (1.5 - 0.5 * x2), x2 - x2 * x2, x * om * om * om)
    }
}

#[inline(always)]
pub(super) fn eval_adaa_wf_derivs(a_raw: f32, b_raw: f32, tiny_eps: f32) -> AdaaWfDeriv {
    let a = scrub_endpoint(a_raw);
    let b = scrub_endpoint(b_raw);
    let a_ok = is_finite_f32(a_raw) && a_raw.abs() < MAX_ADAA_ENDPOINT;
    let b_ok = is_finite_f32(b_raw) && b_raw.abs() < MAX_ADAA_ENDPOINT;
    let lo_is_a = a <= b;
    let (lo, hi) = if lo_is_a { (a, b) } else { (b, a) };
    let dist = hi - lo;
    let max_abs = a.abs().max(b.abs());
    let tiny = if is_finite_f32(tiny_eps) && tiny_eps > 0.0 {
        tiny_eps
    } else {
        TINY_CROSSING_EPS
    };

    let vals = if lo >= 1.0 {
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    } else if hi <= -1.0 {
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    } else if lo >= -1.0 && hi <= 1.0 {
        eval_interior(lo, hi, max_abs)
    } else {
        let cl_lo = lo.max(-1.0);
        let cl_hi = hi.min(1.0);
        let ev = eval_interior(cl_lo, cl_hi, max_abs);
        if dist <= tiny {
            let lm = if lo >= -1.0 { 1.0 } else { 0.0 };
            let hm = if hi <= 1.0 { 1.0 } else { 0.0 };
            [
                ev[0],
                ev[1],
                ev[2],
                ev[3] * lm,
                ev[4] * hm,
                ev[5] * lm,
                ev[6] * hm,
                ev[7] * lm,
                ev[8] * hm,
            ]
        } else {
            let inv = if dist < TINY_CROSSING_EPS {
                let d_safe = dist.max(TINY_CROSSING_DENOM_FLOOR);
                GpuVec4::splat(d_safe * TWO_POW_64).rcp_pos_nr2().to_array()[0] * TWO_POW_64
            } else {
                GpuVec4::splat(dist).rcp_pos_nr2().to_array()[0]
            };
            let inside = (cl_hi - cl_lo).max(0.0);
            let above = (hi - lo.max(1.0)).max(0.0);
            let below = (hi.min(-1.0) - lo).max(0.0);
            let odd_v = canonicalize_zero_f32_bits(((above - below) + inside * ev[0]) * inv);
            let even_v = canonicalize_zero_f32_bits((inside * ev[1]) * inv);
            let wf_v = canonicalize_zero_f32_bits((inside * ev[2]) * inv);
            let (so_lo, se_lo, sw_lo) = sat_point(lo);
            let (so_hi, se_hi, sw_hi) = sat_point(hi);
            [
                odd_v,
                even_v,
                wf_v,
                (odd_v - so_lo) * inv,
                (so_hi - odd_v) * inv,
                (even_v - se_lo) * inv,
                (se_hi - even_v) * inv,
                (wf_v - sw_lo) * inv,
                (sw_hi - wf_v) * inv,
            ]
        }
    };
    let (mut d_odd_da, mut d_odd_db) = if lo_is_a {
        (vals[3], vals[4])
    } else {
        (vals[4], vals[3])
    };
    let (mut d_even_da, mut d_even_db) = if lo_is_a {
        (vals[5], vals[6])
    } else {
        (vals[6], vals[5])
    };
    let (mut d_wf_da, mut d_wf_db) = if lo_is_a {
        (vals[7], vals[8])
    } else {
        (vals[8], vals[7])
    };
    if !a_ok {
        d_odd_da = 0.0;
        d_even_da = 0.0;
        d_wf_da = 0.0;
    }
    if !b_ok {
        d_odd_db = 0.0;
        d_even_db = 0.0;
        d_wf_db = 0.0;
    }
    AdaaWfDeriv {
        odd_poly: vals[0],
        wf_odd: vals[2],
        d_odd_da,
        d_odd_db,
        d_even_da,
        d_even_db,
        d_wf_da,
        d_wf_db,
    }
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{eval_c2_wavefolder_adaa_vec4, eval_poly_adaa1_odd_even_vec4};

    use super::*;

    fn eval_adaa_wf_f64(a: f64, b: f64) -> (f64, f64, f64) {
        let anti = |x: f64| -> (f64, f64, f64) {
            if x <= -1.0 {
                (-x - 0.375, -2.0 / 15.0, 0.0)
            } else if x >= 1.0 {
                (x - 0.375, 2.0 / 15.0, 0.0)
            } else {
                let x2 = x * x;
                let om = 1.0 - x2;
                (
                    0.75 * x2 - 0.125 * x2 * x2,
                    x * x2 / 3.0 - x2 * x2 * x / 5.0,
                    -0.125 * om * om * om * om,
                )
            }
        };
        if (a - b).abs() < 1.0e-12 {
            let x = a.clamp(-1.0, 1.0);
            let x2 = x * x;
            let om = 1.0 - x2;
            (x * (1.5 - 0.5 * x2), x2 - x2 * x2, x * om * om * om)
        } else {
            let (foa, fea, fwa) = anti(a);
            let (fob, feb, fwb) = anti(b);
            let inv = 1.0 / (b - a);
            ((fob - foa) * inv, (feb - fea) * inv, (fwb - fwa) * inv)
        }
    }

    #[test]
    fn test_adaa_wf_derivs_plateaus_crossings_tiny_and_extreme() {
        // verify[dsp.op-n.polyadaa.rail-crossing]
        // verify[dsp.op-n.wavefolder.cubic-c2]
        let z = eval_adaa_wf_derivs(-0.0, 0.0, TINY_CROSSING_EPS);
        assert_eq!(z.odd_poly.to_bits(), 0);
        assert_eq!(z.wf_odd.to_bits(), 0);

        let hi_p = eval_adaa_wf_derivs(1.5, 2.5, TINY_CROSSING_EPS);
        assert_eq!(
            (hi_p.odd_poly, hi_p.d_odd_da, hi_p.d_wf_db),
            (1.0, 0.0, 0.0)
        );
        let lo_p = eval_adaa_wf_derivs(-3.0, -1.5, TINY_CROSSING_EPS);
        assert_eq!(
            (lo_p.odd_poly, lo_p.d_even_da, lo_p.d_wf_da),
            (-1.0, 0.0, 0.0)
        );

        // Tiny crossing (`dist < tiny_eps` and `TINY_CROSSING_EPS > dist` fallback)
        let tiny_hit = eval_adaa_wf_derivs(0.999_999_9, 1.000_000_1, TINY_CROSSING_EPS);
        assert!(tiny_hit.d_odd_da.is_finite() && tiny_hit.d_odd_db == 0.0);
        let tiny_div = eval_adaa_wf_derivs(0.999_999_8, 1.000_000_2, 1.0e-9);
        assert!(tiny_div.d_odd_da.is_finite() && tiny_div.d_odd_db.is_finite());

        // Extreme + non-finite guard zeroes boundary derivatives
        let ext = eval_adaa_wf_derivs(2.0 * MAX_ADAA_ENDPOINT, f32::NAN, -1.0);
        assert_eq!(
            (ext.d_odd_da, ext.d_even_db, ext.d_wf_da, ext.d_wf_db),
            (0.0, 0.0, 0.0, 0.0)
        );

        // Central FD parity across interior, single-rail, and dual-rail crossing pairs
        let eps = 1.0e-6_f64;
        for &(a, b) in &[
            (-0.4_f32, 0.65_f32),
            (0.45, 1.55),
            (-1.6, -0.35),
            (-1.4, 1.3),
            (1.3, -1.4),
        ] {
            let d = eval_adaa_wf_derivs(a, b, TINY_CROSSING_EPS);
            let (odd_k, _, _) = eval_poly_adaa1_odd_even_vec4(GpuVec4::splat(a), GpuVec4::splat(b));
            let (wf_k, _) = eval_c2_wavefolder_adaa_vec4(GpuVec4::splat(a), GpuVec4::splat(b));
            assert!((d.odd_poly - odd_k.to_array()[0]).abs() <= 1.0e-5);
            assert!((d.wf_odd - wf_k.to_array()[0]).abs() <= 1.0e-5);

            let (fa_p, fe_ap, fw_ap) = eval_adaa_wf_f64(f64::from(a) + eps, f64::from(b));
            let (fa_m, fe_am, fw_am) = eval_adaa_wf_f64(f64::from(a) - eps, f64::from(b));
            let (fb_p, fe_bp, fw_bp) = eval_adaa_wf_f64(f64::from(a), f64::from(b) + eps);
            let (fb_m, fe_bm, fw_bm) = eval_adaa_wf_f64(f64::from(a), f64::from(b) - eps);
            let inv2e = 0.5 / eps;
            assert!((f64::from(d.d_odd_da) - (fa_p - fa_m) * inv2e).abs() <= 1.0e-4);
            assert!((f64::from(d.d_odd_db) - (fb_p - fb_m) * inv2e).abs() <= 1.0e-4);
            assert!((f64::from(d.d_even_da) - (fe_ap - fe_am) * inv2e).abs() <= 1.0e-4);
            assert!((f64::from(d.d_even_db) - (fe_bp - fe_bm) * inv2e).abs() <= 1.0e-4);
            assert!((f64::from(d.d_wf_da) - (fw_ap - fw_am) * inv2e).abs() <= 1.0e-4);
            assert!((f64::from(d.d_wf_db) - (fw_bp - fw_bm) * inv2e).abs() <= 1.0e-4);
        }
    }
}
