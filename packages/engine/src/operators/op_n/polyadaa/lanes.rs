//! Two-lane Stage II Sub-stage 2.2 (`V1B` Driver) PolyADAA kernels for [`super::MultiDrivePolyAdaaChannel`].

use super::{
    MultiDrivePolyAdaaChannel, eval_saturated_polyadaa_rail_crossing_f64,
    poly_adaa_algebraic_eval_pair, poly_adaa_eval_vec4_pair,
};
use crate::math::{
    DefaultMathBackend, MathBackend, SimdVec4, flush_recursive_state_f32,
    flush_recursive_state_vec4,
};

impl MultiDrivePolyAdaaChannel {
    #[inline(always)]
    pub(super) fn crosses_rail(u_prev: f32, u_curr: f32) -> bool {
        ((u_prev - 1.0) * (u_curr - 1.0) < 0.0) || ((u_prev + 1.0) * (u_curr + 1.0) < 0.0)
    }

    /// Steps only drive lanes `1` (`1.0x`) and `2` (`4.0x`) for Stage II Sub-stage 2.2 (`V1B` Driver),
    /// returning `(out_lane1, out_lane2)`.
    /// Direct callers must hold a [`crate::FtzDazGuard`] for the enclosing batch.
    #[inline(always)]
    pub fn step_lanes_1_2(
        &mut self,
        x: f32,
        beta: f32,
        delta: f32,
        alpha: f32,
        r_ac: f32,
    ) -> (f32, f32) {
        let one_minus_rac = 1.0 - r_ac;
        let x_scaled = (2.0 / 3.0) * x;
        let u1_raw = x_scaled;
        let u2_raw = 4.0 * x_scaled;

        let p1 = self.u_prev[1];
        let p2 = self.u_prev[2];
        let u1_p = p1.clamp(-1.0, 1.0);
        let u1_c = u1_raw.clamp(-1.0, 1.0);
        let u2_p = p2.clamp(-1.0, 1.0);
        let u2_c = u2_raw.clamp(-1.0, 1.0);

        let (mut a1, mut ae1) = poly_adaa_algebraic_eval_pair(u1_p, u1_c, beta, delta);
        let (mut a2, mut ae2) = poly_adaa_algebraic_eval_pair(u2_p, u2_c, beta, delta);
        if (p2.abs() > 1.0 || u2_raw.abs() > 1.0)
            && (Self::crosses_rail(p2, u2_raw) || Self::crosses_rail(p1, u1_raw))
        {
            if Self::crosses_rail(p1, u1_raw) {
                let (odd, even, _) =
                    eval_saturated_polyadaa_rail_crossing_f64(p1 as f64, u1_raw as f64);
                a1 = ((beta as f64) * odd + (delta as f64) * even) as f32;
                ae1 = even as f32;
            }
            if Self::crosses_rail(p2, u2_raw) {
                let (odd, even, _) =
                    eval_saturated_polyadaa_rail_crossing_f64(p2 as f64, u2_raw as f64);
                a2 = ((beta as f64) * odd + (delta as f64) * even) as f32;
                ae2 = even as f32;
            }
        }
        let y1_nl = 0.875 * a1 + 0.125 * self.a_prev[1];
        let next_m1_dc = r_ac * self.m_dc[1] + one_minus_rac * y1_nl;
        let out1 = alpha * x + (y1_nl - next_m1_dc);
        self.u_prev[1] = flush_recursive_state_f32(u1_raw);
        self.a_prev[1] = flush_recursive_state_f32(a1);
        self.m_dc[1] = flush_recursive_state_f32(next_m1_dc);
        self.last_a_even[1] = flush_recursive_state_f32(ae1);

        let y2_nl = 0.875 * a2 + 0.125 * self.a_prev[2];
        let next_m2_dc = r_ac * self.m_dc[2] + one_minus_rac * y2_nl;
        let out2 = alpha * x + (y2_nl - next_m2_dc);
        self.u_prev[2] = flush_recursive_state_f32(u2_raw);
        self.a_prev[2] = flush_recursive_state_f32(a2);
        self.m_dc[2] = flush_recursive_state_f32(next_m2_dc);
        self.last_a_even[2] = flush_recursive_state_f32(ae2);

        (out1, out2)
    }

    /// Steps drive lanes `1` (`1.0x`) and `2` (`4.0x`) for a stereo `(left, right)` pair simultaneously
    /// across a single 4-lane SIMD loop, returning `(out_l_1, out_l_2, out_r_1, out_r_2)`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn step_stereo_lanes_1_2(
        left: &mut Self,
        right: &mut Self,
        x_l: f32,
        x_r: f32,
        beta: f32,
        delta: f32,
        alpha: f32,
        r_ac: f32,
    ) -> (f32, f32, f32, f32) {
        Self::step_stereo_lanes_1_2_lr(
            left, right, x_l, x_r, beta, delta, alpha, beta, delta, alpha, r_ac,
        )
    }

    /// Steps drive lanes `1` (`1.0x`) and `2` (`4.0x`) for a stereo `(left, right)` pair simultaneously
    /// with independent Left/Right `(beta, delta, alpha)` parameters across a single 4-lane SIMD loop.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn step_stereo_lanes_1_2_lr(
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
    ) -> (f32, f32, f32, f32) {
        type V = <DefaultMathBackend as MathBackend>::Vec4;
        let one_minus_rac = 1.0 - r_ac;
        let xl_s = (2.0 / 3.0) * x_l;
        let xr_s = (2.0 / 3.0) * x_r;
        let ax_l = alpha_l * x_l;
        let ax_r = alpha_r * x_r;

        let u_prev_4 = [
            left.u_prev[1],
            left.u_prev[2],
            right.u_prev[1],
            right.u_prev[2],
        ];
        let a_prev_4 = [
            left.a_prev[1],
            left.a_prev[2],
            right.a_prev[1],
            right.a_prev[2],
        ];
        let m_dc_4 = [left.m_dc[1], left.m_dc[2], right.m_dc[1], right.m_dc[2]];
        let u_raw_4 = [xl_s, 4.0 * xl_s, xr_s, 4.0 * xr_s];
        let beta_4 = [beta_l, beta_l, beta_r, beta_r];
        let delta_4 = [delta_l, delta_l, delta_r, delta_r];
        let alpha_x_4 = [ax_l, ax_l, ax_r, ax_r];

        let neg_one = V::splat(-1.0);
        let pos_one = V::splat(1.0);
        let u_prev_raw = V::load(&u_prev_4);
        let u_curr_raw = V::load(&u_raw_4);
        let u_prev = u_prev_raw.clamp(neg_one, pos_one);
        let u_curr = u_curr_raw.clamp(neg_one, pos_one);
        let beta_v = V::load(&beta_4);
        let delta_v = V::load(&delta_4);

        let (mut a_v, mut a_even_v) = poly_adaa_eval_vec4_pair(u_prev, u_curr, beta_v, delta_v);

        if u_prev_4[1].abs() > 1.0
            || u_raw_4[1].abs() > 1.0
            || u_prev_4[3].abs() > 1.0
            || u_raw_4[3].abs() > 1.0
        {
            let crossing_v = (u_prev.sub(u_curr)).abs().mul(
                (u_prev.sub(u_prev_raw))
                    .abs()
                    .add((u_curr.sub(u_curr_raw)).abs()),
            );
            if crossing_v.horizontal_sum() > 0.0 {
                let mut c_arr = [0.0f32; 4];
                crossing_v.store(&mut c_arr);
                let mut a_arr = [0.0f32; 4];
                let mut a_even_arr = [0.0f32; 4];
                a_v.store(&mut a_arr);
                a_even_v.store(&mut a_even_arr);
                for i in 0..4 {
                    if c_arr[i] > 0.0 {
                        let (odd, even, _) = eval_saturated_polyadaa_rail_crossing_f64(
                            u_prev_4[i] as f64,
                            u_raw_4[i] as f64,
                        );
                        a_arr[i] = ((beta_4[i] as f64) * odd + (delta_4[i] as f64) * even) as f32;
                        a_even_arr[i] = even as f32;
                    }
                }
                a_v = V::load(&a_arr);
                a_even_v = V::load(&a_even_arr);
            }
        }

        let a_prev_v = V::load(&a_prev_4);
        let m_dc_v = V::load(&m_dc_4);
        let y_nl = a_v.mul(V::splat(0.875)).fma(a_prev_v, V::splat(0.125));
        let next_m_dc = m_dc_v
            .mul(V::splat(r_ac))
            .fma(y_nl, V::splat(one_minus_rac));
        let out_v = V::load(&alpha_x_4).add(y_nl.sub(next_m_dc));

        let mut u_next = [0.0f32; 4];
        let mut a_next = [0.0f32; 4];
        let mut m_next = [0.0f32; 4];
        let mut ae_next = [0.0f32; 4];
        flush_recursive_state_vec4(u_curr_raw).store(&mut u_next);
        flush_recursive_state_vec4(a_v).store(&mut a_next);
        flush_recursive_state_vec4(next_m_dc).store(&mut m_next);
        flush_recursive_state_vec4(a_even_v).store(&mut ae_next);

        left.u_prev[1] = u_next[0];
        left.u_prev[2] = u_next[1];
        right.u_prev[1] = u_next[2];
        right.u_prev[2] = u_next[3];

        left.a_prev[1] = a_next[0];
        left.a_prev[2] = a_next[1];
        right.a_prev[1] = a_next[2];
        right.a_prev[2] = a_next[3];

        left.m_dc[1] = m_next[0];
        left.m_dc[2] = m_next[1];
        right.m_dc[1] = m_next[2];
        right.m_dc[2] = m_next[3];

        left.last_a_even[1] = ae_next[0];
        left.last_a_even[2] = ae_next[1];
        right.last_a_even[1] = ae_next[2];
        right.last_a_even[2] = ae_next[3];

        let mut out_4 = [0.0f32; 4];
        out_v.store(&mut out_4);
        (out_4[0], out_4[1], out_4[2], out_4[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FtzDazGuard;

    #[test]
    fn test_step_lanes_1_2_and_stereo_lanes_1_2_match_full_bank() {
        let _audio = FtzDazGuard::new();
        let mut full_l = MultiDrivePolyAdaaChannel::new();
        let mut full_r = MultiDrivePolyAdaaChannel::new();
        let mut l2_solo = MultiDrivePolyAdaaChannel::new();
        let mut l2_stereo_l = MultiDrivePolyAdaaChannel::new();
        let mut l2_stereo_r = MultiDrivePolyAdaaChannel::new();

        // Linear interior + rail crossings on both lane 1 (1x) and lane 2 (4x)
        let samples = [
            (0.25_f32, -0.2_f32),
            (2.0, -2.0),
            (-2.0, 2.0),
            (2.2, 0.1),
            (0.0, 0.0),
        ];
        for (xl, xr) in samples {
            let (fl, fr) = MultiDrivePolyAdaaChannel::step_stereo_pair(
                &mut full_l,
                &mut full_r,
                xl,
                xr,
                0.9,
                -0.35,
                0.6,
                0.9,
                -0.35,
                0.6,
                0.9993,
            );
            let (s1, s2) = l2_solo.step_lanes_1_2(xl, 0.9, -0.35, 0.6, 0.9993);
            let (sl1, sl2, sr1, sr2) = MultiDrivePolyAdaaChannel::step_stereo_lanes_1_2(
                &mut l2_stereo_l,
                &mut l2_stereo_r,
                xl,
                xr,
                0.9,
                -0.35,
                0.6,
                0.9993,
            );
            assert!((s1 - fl[1]).abs() < 1e-5, "solo lane1 {} != {}", s1, fl[1]);
            assert!((s2 - fl[2]).abs() < 1e-5, "solo lane2 {} != {}", s2, fl[2]);
            assert!((sl1 - fl[1]).abs() < 1e-5, "stereo L1 {} != {}", sl1, fl[1]);
            assert!((sl2 - fl[2]).abs() < 1e-5, "stereo L2 {} != {}", sl2, fl[2]);
            assert!((sr1 - fr[1]).abs() < 1e-5, "stereo R1 {} != {}", sr1, fr[1]);
            assert!((sr2 - fr[2]).abs() < 1e-5, "stereo R2 {} != {}", sr2, fr[2]);
        }
    }
}
