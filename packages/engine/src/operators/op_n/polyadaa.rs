//! Op-N Multi-Drive 1x B-Spline PolyADAA Bank & Saturated Rail-Crossing Evaluation.
//!
//! Canonical Reference: `docs/specs/dsp/op_n_commuted_cascade.md` (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.linear-residual]`, `ref[dsp.op-n.polyadaa.bspline-ac]`, `ref[dsp.op-n.polyadaa.drive-bank]`).

use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4,
};

/// Drive scales for the 4-lane Multi-Drive PolyADAA Bank.
pub const MULTI_DRIVE_SCALES: [f32; 4] = [0.25, 1.0, 4.0, 16.0];

/// Domain-specific SIMD backend trait for `Op-N` (Multi-Drive PolyADAA Commuted Cascade).
pub trait OpNBackend<V: SimdVec4> {
    /// Evaluates the 4-channel division-free algebraic `PolyADAA` midpoint polynomial
    /// across 4 parallel SIMD lanes (`ref[dsp.op-n.polyadaa.midpoint]`).
    fn poly_adaa_algebraic_4ch(u_prev: V, u_curr: V, beta: V, delta: V) -> V;
}

/// Safe `SimdVec4` implementation of [`OpNBackend::poly_adaa_algebraic_4ch`].
#[inline(always)]
pub fn poly_adaa_algebraic_4ch_vec4<V: SimdVec4>(u_prev: V, u_curr: V, beta: V, delta: V) -> V {
    poly_adaa_eval_vec4_pair(u_prev, u_curr, beta, delta).0
}

impl OpNBackend<crate::math::ScalarVec4> for crate::math::ScalarBackend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::ScalarVec4,
        u_curr: crate::math::ScalarVec4,
        beta: crate::math::ScalarVec4,
        delta: crate::math::ScalarVec4,
    ) -> crate::math::ScalarVec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpNBackend<crate::math::NeonVec4> for crate::math::NeonBackend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::NeonVec4,
        u_curr: crate::math::NeonVec4,
        beta: crate::math::NeonVec4,
        delta: crate::math::NeonVec4,
    ) -> crate::math::NeonVec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpNBackend<crate::math::Sve2Vec4> for crate::math::ArmSve2Backend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::Sve2Vec4,
        u_curr: crate::math::Sve2Vec4,
        beta: crate::math::Sve2Vec4,
        delta: crate::math::Sve2Vec4,
    ) -> crate::math::Sve2Vec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

#[cfg(target_arch = "aarch64")]
impl OpNBackend<crate::math::NeonVec4> for crate::math::ArmSmeBackend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::NeonVec4,
        u_curr: crate::math::NeonVec4,
        beta: crate::math::NeonVec4,
        delta: crate::math::NeonVec4,
    ) -> crate::math::NeonVec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpNBackend<crate::math::Sse2Vec4> for crate::math::X86Sse2Backend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::Sse2Vec4,
        u_curr: crate::math::Sse2Vec4,
        beta: crate::math::Sse2Vec4,
        delta: crate::math::Sse2Vec4,
    ) -> crate::math::Sse2Vec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpNBackend<crate::math::Sse42Vec4> for crate::math::X86Sse42Backend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::Sse42Vec4,
        u_curr: crate::math::Sse42Vec4,
        beta: crate::math::Sse42Vec4,
        delta: crate::math::Sse42Vec4,
    ) -> crate::math::Sse42Vec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

#[cfg(target_arch = "x86_64")]
impl OpNBackend<crate::math::Avx2FmaVec4> for crate::math::X86Avx2FmaBackend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::Avx2FmaVec4,
        u_curr: crate::math::Avx2FmaVec4,
        beta: crate::math::Avx2FmaVec4,
        delta: crate::math::Avx2FmaVec4,
    ) -> crate::math::Avx2FmaVec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

#[cfg(target_arch = "wasm32")]
impl OpNBackend<crate::math::WasmVec4> for crate::math::WasmSimd128Backend {
    #[inline(always)]
    fn poly_adaa_algebraic_4ch(
        u_prev: crate::math::WasmVec4,
        u_curr: crate::math::WasmVec4,
        beta: crate::math::WasmVec4,
        delta: crate::math::WasmVec4,
    ) -> crate::math::WasmVec4 {
        poly_adaa_algebraic_4ch_vec4(u_prev, u_curr, beta, delta)
    }
}

/// Exact division-free algebraic PolyADAA segment evaluation (`ref[dsp.op-n.polyadaa.midpoint]`).
///
/// Evaluates:
/// `m = 0.5 * (u_curr + u_prev)`
/// `d = 0.5 * (u_curr - u_prev)`
/// `A_odd = m * (1.5 - 0.5 * (m^2 + d^2))`
/// `A_even = (m^2 + 1/3 * d^2) - (m^4 + 2 * m^2 * d^2 + 1/5 * d^4)`
/// `A = beta * A_odd + delta * A_even`
#[inline(always)]
pub fn poly_adaa_algebraic_eval(u_prev: f32, u_curr: f32, beta: f32, delta: f32) -> f32 {
    poly_adaa_algebraic_eval_pair(u_prev, u_curr, beta, delta).0
}

#[inline(always)]
pub fn poly_adaa_algebraic_eval_pair(
    u_prev: f32,
    u_curr: f32,
    beta: f32,
    delta: f32,
) -> (f32, f32) {
    const ONE_THIRD: f32 = 1.0 / 3.0;

    let m = 0.5 * (u_curr + u_prev);
    let d = 0.5 * (u_curr - u_prev);

    let m2 = m * m;
    let d2 = d * d;
    let s = m2 + d2;

    let odd_factor = 1.5 - 0.5 * s;
    let a_odd = m * odd_factor;

    let quad = m2 + ONE_THIRD * d2;
    let d4 = d2 * d2;
    let quart = s * s - 0.8 * d4;
    let a_even = quad - quart;

    (beta * a_odd + delta * a_even, a_even)
}

#[inline(always)]
pub(super) fn poly_adaa_eval_vec4_pair<V: SimdVec4>(
    u_prev: V,
    u_curr: V,
    beta: V,
    delta: V,
) -> (V, V) {
    let half = V::splat(0.5);
    let one_pt_five = V::splat(1.5);
    let one_third = V::splat(1.0 / 3.0);

    let m = half.mul(u_curr.add(u_prev));
    let d = half.mul(u_curr.sub(u_prev));

    let m2 = m.mul(m);
    let d2 = d.mul(d);
    let s = m2.add(d2);

    let odd_factor = one_pt_five.fms(half, s);
    let a_odd = m.mul(odd_factor);

    let quad = m2.fma(one_third, d2);
    let d4 = d2.mul(d2);
    let quart = (s.mul(s)).fms(V::splat(0.8), d4);
    let a_even = quad.sub(quart);

    ((beta.mul(a_odd)).fma(delta, a_even), a_even)
}

pub mod lanes;
pub mod rail;

pub use rail::eval_saturated_polyadaa_rail_crossing_f64;

/// Multi-Drive 1x B-Spline PolyADAA Bank Channel State (`ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.bspline-ac]`).
#[derive(Clone, Copy, Debug)]
pub struct MultiDrivePolyAdaaChannel {
    pub u_prev: [f32; 4],
    pub a_prev: [f32; 4],
    pub m_dc: [f32; 4],
    pub last_a_even: [f32; 4],
}

impl Default for MultiDrivePolyAdaaChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDrivePolyAdaaChannel {
    pub const fn new() -> Self {
        Self {
            u_prev: [0.0; 4],
            a_prev: [0.0; 4],
            m_dc: [0.0; 4],
            last_a_even: [0.0; 4],
        }
    }

    pub fn reset(&mut self) {
        self.u_prev.fill(0.0);
        self.a_prev.fill(0.0);
        self.m_dc.fill(0.0);
        self.last_a_even.fill(0.0);
    }

    /// Initializes resting DC equilibrium fixed point for grid bias `b_conv`.
    pub fn init_dc_equilibrium(&mut self, b_conv: f32, beta: f32, delta: f32) {
        for (i, &drive) in MULTI_DRIVE_SCALES.iter().enumerate() {
            let u_raw = (2.0 / 3.0) * drive * b_conv;
            let u_rest = u_raw.clamp(-1.0, 1.0);
            let (a_rest, a_even_rest) = poly_adaa_algebraic_eval_pair(u_rest, u_rest, beta, delta);
            self.u_prev[i] = u_raw;
            self.a_prev[i] = a_rest;
            self.m_dc[i] = flush_recursive_state_f32(a_rest);
            self.last_a_even[i] = flush_recursive_state_f32(a_even_rest);
        }
    }

    /// Steps the multi-drive bank for input `x`, returning 4-lane nonlinearly shaped basis vector.
    /// Evaluates the unclamped 6-step `f64` Newton rail-crossing `PolyADAA` law when a segment crosses $\pm 1$
    /// (`ref[dsp.op-n.polyadaa.clamp]`, `ref[dsp.op-n.polyadaa.midpoint]`, `ref[dsp.op-n.polyadaa.rail-crossing]`, `ref[dsp.op-n.polyadaa.bspline-ac]`).
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step(&mut self, x: f32, beta: f32, delta: f32, alpha: f32, r_ac: f32) -> [f32; 4] {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let one_minus_rac = 1.0 - r_ac;
        let x_scaled = (2.0 / 3.0) * x;

        let u_raw_v = V::load(&MULTI_DRIVE_SCALES).mul(V::splat(x_scaled));
        let u_prev_v = V::load(&self.u_prev);
        let neg_one = V::splat(-1.0);
        let pos_one = V::splat(1.0);
        let u_p_v = u_prev_v.max(neg_one).min(pos_one);
        let u_c_v = u_raw_v.max(neg_one).min(pos_one);

        let (mut a_v, mut a_even_v) =
            poly_adaa_eval_vec4_pair(u_p_v, u_c_v, V::splat(beta), V::splat(delta));

        let u3_raw = 16.0 * x_scaled;
        let u3_prev = self.u_prev[3];
        if (u3_prev != u3_raw) && (u3_prev.abs() > 1.0 || u3_raw.abs() > 1.0) {
            let crossing_v = (u_p_v.sub(u_c_v))
                .abs()
                .mul((u_p_v.sub(u_prev_v)).abs().add((u_c_v.sub(u_raw_v)).abs()));
            if crossing_v.horizontal_sum() > 0.0 {
                let mut c_arr = [0.0f32; 4];
                crossing_v.store(&mut c_arr);
                let u_raw_arr = [0.25 * x_scaled, x_scaled, 4.0 * x_scaled, u3_raw];
                let mut a_arr = [0.0f32; 4];
                let mut a_even_arr = [0.0f32; 4];
                a_v.store(&mut a_arr);
                a_even_v.store(&mut a_even_arr);
                for i in 0..4 {
                    if c_arr[i] > 0.0 {
                        let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(
                            self.u_prev[i] as f64,
                            u_raw_arr[i] as f64,
                        );
                        a_arr[i] = ((beta as f64) * odd + (delta as f64) * even) as f32;
                        a_even_arr[i] = even as f32;
                    }
                }
                a_v = V::load(&a_arr);
                a_even_v = V::load(&a_even_arr);
            }
        }

        let a_prev_v = V::load(&self.a_prev);
        let m_prev_v = V::load(&self.m_dc);
        let y_nl_v = (V::splat(0.125).mul(a_prev_v)).fma(V::splat(0.875), a_v);
        let next_m_dc_v = (V::splat(r_ac).mul(m_prev_v)).fma(V::splat(one_minus_rac), y_nl_v);
        let out_v = V::splat(alpha * x).add(y_nl_v.sub(next_m_dc_v));

        flush_recursive_state_vec4(u_raw_v).store(&mut self.u_prev);
        flush_recursive_state_vec4(a_v).store(&mut self.a_prev);
        flush_recursive_state_vec4(next_m_dc_v).store(&mut self.m_dc);
        flush_recursive_state_vec4(a_even_v).store(&mut self.last_a_even);

        out_v.to_array()
    }

    /// Steps all 4 drive lanes (`0.25x, 1.0x, 4.0x, 16.0x`) for a stereo `(left, right)` pair simultaneously,
    /// interleaving independent Left and Right SIMD operations and sharing constant registers.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn step_stereo_pair(
        left: &mut Self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
        beta_l: f32,
        delta_l: f32,
        alpha_l: f32,
        beta_r: f32,
        delta_r: f32,
        alpha_r: f32,
        r_ac: f32,
    ) -> ([f32; 4], [f32; 4]) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let one_minus_rac = 1.0 - r_ac;
        let xl_s = (2.0 / 3.0) * x_l;
        let xr_s = (2.0 / 3.0) * x_r;

        let scales = V::load(&MULTI_DRIVE_SCALES);
        let neg_one = V::splat(-1.0);
        let pos_one = V::splat(1.0);
        let half = V::splat(0.5);
        let one_pt_five = V::splat(1.5);
        let one_third = V::splat(1.0 / 3.0);
        let pt_eight = V::splat(0.8);

        let u_raw_l = scales.mul(V::splat(xl_s));
        let u_raw_r = scales.mul(V::splat(xr_s));
        let u_prev_l = V::load(&left.u_prev);
        let u_prev_r = V::load(&right.u_prev);
        let u_p_l = u_prev_l.max(neg_one).min(pos_one);
        let u_p_r = u_prev_r.max(neg_one).min(pos_one);
        let u_c_l = u_raw_l.max(neg_one).min(pos_one);
        let u_c_r = u_raw_r.max(neg_one).min(pos_one);

        let m_l = half.mul(u_c_l.add(u_p_l));
        let m_r = half.mul(u_c_r.add(u_p_r));
        let d_l = half.mul(u_c_l.sub(u_p_l));
        let d_r = half.mul(u_c_r.sub(u_p_r));

        let m2_l = m_l.mul(m_l);
        let m2_r = m_r.mul(m_r);
        let d2_l = d_l.mul(d_l);
        let d2_r = d_r.mul(d_r);
        let s_l = m2_l.add(d2_l);
        let s_r = m2_r.add(d2_r);

        let a_odd_l = m_l.mul(one_pt_five.fms(half, s_l));
        let a_odd_r = m_r.mul(one_pt_five.fms(half, s_r));
        let mut a_even_l = m2_l
            .fma(one_third, d2_l)
            .sub((s_l.mul(s_l)).fms(pt_eight, d2_l.mul(d2_l)));
        let mut a_even_r = m2_r
            .fma(one_third, d2_r)
            .sub((s_r.mul(s_r)).fms(pt_eight, d2_r.mul(d2_r)));

        let mut a_l = (V::splat(beta_l).mul(a_odd_l)).fma(V::splat(delta_l), a_even_l);
        let mut a_r = (V::splat(beta_r).mul(a_odd_r)).fma(V::splat(delta_r), a_even_r);

        let u3_raw_l = 16.0 * xl_s;
        if (left.u_prev[3] != u3_raw_l) && (left.u_prev[3].abs() > 1.0 || u3_raw_l.abs() > 1.0) {
            let crossing_l = (u_p_l.sub(u_c_l))
                .abs()
                .mul((u_p_l.sub(u_prev_l)).abs().add((u_c_l.sub(u_raw_l)).abs()));
            if crossing_l.horizontal_sum() > 0.0 {
                let mut c_arr = [0.0f32; 4];
                crossing_l.store(&mut c_arr);
                let u_raw_arr = [0.25 * xl_s, xl_s, 4.0 * xl_s, u3_raw_l];
                let mut a_arr = [0.0f32; 4];
                let mut ae_arr = [0.0f32; 4];
                a_l.store(&mut a_arr);
                a_even_l.store(&mut ae_arr);
                for i in 0..4 {
                    if c_arr[i] > 0.0 {
                        let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(
                            left.u_prev[i] as f64,
                            u_raw_arr[i] as f64,
                        );
                        a_arr[i] = ((beta_l as f64) * odd + (delta_l as f64) * even) as f32;
                        ae_arr[i] = even as f32;
                    }
                }
                a_l = V::load(&a_arr);
                a_even_l = V::load(&ae_arr);
            }
        }

        let u3_raw_r = 16.0 * xr_s;
        if (right.u_prev[3] != u3_raw_r) && (right.u_prev[3].abs() > 1.0 || u3_raw_r.abs() > 1.0) {
            let crossing_r = (u_p_r.sub(u_c_r))
                .abs()
                .mul((u_p_r.sub(u_prev_r)).abs().add((u_c_r.sub(u_raw_r)).abs()));
            if crossing_r.horizontal_sum() > 0.0 {
                let mut c_arr = [0.0f32; 4];
                crossing_r.store(&mut c_arr);
                let u_raw_arr = [0.25 * xr_s, xr_s, 4.0 * xr_s, u3_raw_r];
                let mut a_arr = [0.0f32; 4];
                let mut ae_arr = [0.0f32; 4];
                a_r.store(&mut a_arr);
                a_even_r.store(&mut ae_arr);
                for i in 0..4 {
                    if c_arr[i] > 0.0 {
                        let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(
                            right.u_prev[i] as f64,
                            u_raw_arr[i] as f64,
                        );
                        a_arr[i] = ((beta_r as f64) * odd + (delta_r as f64) * even) as f32;
                        ae_arr[i] = even as f32;
                    }
                }
                a_r = V::load(&a_arr);
                a_even_r = V::load(&ae_arr);
            }
        }

        let w_prev = V::splat(0.125);
        let w_curr = V::splat(0.875);
        let r_ac_v = V::splat(r_ac);
        let omr_v = V::splat(one_minus_rac);

        let y_nl_l = (w_prev.mul(V::load(&left.a_prev))).fma(w_curr, a_l);
        let y_nl_r = (w_prev.mul(V::load(&right.a_prev))).fma(w_curr, a_r);
        let next_m_l = (r_ac_v.mul(V::load(&left.m_dc))).fma(omr_v, y_nl_l);
        let next_m_r = (r_ac_v.mul(V::load(&right.m_dc))).fma(omr_v, y_nl_r);
        let out_v_l = V::splat(alpha_l * x_l).add(y_nl_l.sub(next_m_l));
        let out_v_r = V::splat(alpha_r * x_r).add(y_nl_r.sub(next_m_r));

        flush_recursive_state_vec4(u_raw_l).store(&mut left.u_prev);
        flush_recursive_state_vec4(a_l).store(&mut left.a_prev);
        flush_recursive_state_vec4(next_m_l).store(&mut left.m_dc);
        flush_recursive_state_vec4(a_even_l).store(&mut left.last_a_even);

        flush_recursive_state_vec4(u_raw_r).store(&mut right.u_prev);
        flush_recursive_state_vec4(a_r).store(&mut right.a_prev);
        flush_recursive_state_vec4(next_m_r).store(&mut right.m_dc);
        flush_recursive_state_vec4(a_even_r).store(&mut right.last_a_even);

        let mut out_l = [0.0f32; 4];
        let mut out_r = [0.0f32; 4];
        out_v_l.store(&mut out_l);
        out_v_r.store(&mut out_r);
        (out_l, out_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;
    use proptest::prelude::*;

    #[test]
    fn test_poly_adaa_algebraic() {
        for &m in &[-0.8_f32, -0.4, 0.0, 0.3, 0.9] {
            let beta = 1.2_f32;
            let delta = -0.35_f32;
            let actual = poly_adaa_algebraic_eval(m, m, beta, delta);
            let expected = beta * m * (1.5 - 0.5 * m * m) + delta * (m * m - m * m * m * m);
            assert!((actual - expected).abs() < 1e-6);
        }

        let d = 0.6_f32;
        let a_sym = poly_adaa_algebraic_eval(-d, d, 1.0, 1.0);
        let expected_sym = (d * d) / 3.0 - 0.2 * d * d * d * d;
        assert!((a_sym - expected_sym).abs() < 1e-6);
    }

    #[test]
    fn test_multi_drive_poly_adaa_channel() {
        let _audio = FtzDazGuard::new();
        let mut chan = MultiDrivePolyAdaaChannel::default();
        chan.reset();
        let out_lin = chan.step(0.4, 0.0, 0.0, 0.75, 0.9993);
        for &v in &out_lin {
            assert!((v - 0.3).abs() < 1e-6);
        }

        chan.reset();
        let out_nl = chan.step(0.2, 1.0, 0.0, 0.0, 0.9993);
        assert!(out_nl[0] < out_nl[1]);
        assert!(out_nl[1] < out_nl[2]);
        assert!(out_nl[2] < out_nl[3]);

        chan.reset();
        assert_eq!(chan.u_prev, [0.0; 4]);
    }

    #[test]
    fn test_poly_adaa_dc_equilibrium_startup() {
        let _audio = FtzDazGuard::new();
        let mut chan = MultiDrivePolyAdaaChannel::default();
        let b_conv = 0.2_f32;
        chan.init_dc_equilibrium(b_conv, 1.0, 0.25);

        let out = chan.step(b_conv, 1.0, 0.25, 0.0, 0.9993);
        for &val in &out {
            assert!(
                val.abs() < 1e-4,
                "DC pop detected at startup from quiescent state: {}",
                val
            );
        }
    }

    /// verify[dsp.op-n.polyadaa.clamp]
    /// verify[dsp.op-n.polyadaa.midpoint]
    /// verify[dsp.op-n.polyadaa.rail-crossing]
    /// verify[dsp.op-n.polyadaa.linear-residual]
    /// verify[dsp.op-n.polyadaa.bspline-ac]
    /// verify[dsp.op-n.polyadaa.drive-bank]
    #[test]
    fn test_polyadaa_clamp_midpoint_rail_crossing_linear_residual_and_drive_bank() {
        let _guard = FtzDazGuard::new();
        assert_eq!(MULTI_DRIVE_SCALES, [0.25, 1.0, 4.0, 16.0]);

        let tol = f64::from_bits((1023 - 44) << 52);
        for &(prev, curr, exp_odd, exp_even, exp_crossed) in &[
            (0.0_f64, 1.0_f64, 0.625_f64, 0.13333333333333333_f64, false),
            (0.0, 2.0, 0.8125, 0.06666666666666667, true),
            (-2.0, 0.0, -0.8125, 0.06666666666666667, true),
            (-2.0, 2.0, 0.0, 0.06666666666666667, true),
            (2.0, 4.0, 1.0, 0.0, false),
            (-4.0, -2.0, -1.0, 0.0, false),
            (0.25, 0.25, 0.3671875, 0.05859375, false),
        ] {
            for (a, b) in [(prev, curr), (curr, prev)] {
                let (odd, even, crossed) = eval_saturated_polyadaa_rail_crossing_f64(a, b);
                assert_eq!(crossed, exp_crossed);
                assert!((odd - exp_odd).abs() < tol);
                assert!((even - exp_even).abs() < tol);
            }
        }

        let a0 = poly_adaa_algebraic_eval(0.5, 0.5, 1.0, 1.0);
        let exp_a0 = 0.5 * (1.5 - 0.5 * 0.25) + (0.25 - 0.0625);
        assert!((a0 - exp_a0).abs() < 1e-6);

        let mut ch = MultiDrivePolyAdaaChannel::new();
        let lin_only = ch.step(0.5, 0.0, 0.0, 2.0, 0.9993);
        assert_eq!(lin_only, [1.0; 4]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_poly_adaa_algebraic_symmetry_and_quotient(
            u0 in -0.95f32..0.95f32,
            u1 in -0.95f32..0.95f32,
            beta in 0.1f32..2.0f32,
            delta in -0.5f32..0.5f32,
        ) {
            let a01 = poly_adaa_algebraic_eval(u0, u1, beta, delta);
            let a10 = poly_adaa_algebraic_eval(u1, u0, beta, delta);
            prop_assert!(a01.is_finite());
            prop_assert!((a01 - a10).abs() < 1e-6);

            let diff = (u1 as f64) - (u0 as f64);
            if diff.abs() > 1e-3 {
                let antideriv = |u: f64| -> f64 {
                    let u2 = u * u;
                    let u3 = u2 * u;
                    let u4 = u2 * u2;
                    let u5 = u4 * u;
                    (beta as f64) * (0.75 * u2 - 0.125 * u4)
                        + (delta as f64) * ((1.0 / 3.0) * u3 - 0.2 * u5)
                };
                let exact = (antideriv(u1 as f64) - antideriv(u0 as f64)) / diff;
                prop_assert!((a01 as f64 - exact).abs() < 1e-5);
            }
        }
    }
}
