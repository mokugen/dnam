//! Double-precision complex-conjugate Kautz cascade & OMP operator (`Op-Kautz-Train`).

use dnam_engine::math::trainer::{SimdF64Vec2, TrainerMathBackend};

/// Hardware-abstracted complex-conjugate Kautz cascade & OMP operator for `dnam-capture`.
pub trait KautzCascadeBackend: TrainerMathBackend {
    /// Generates interleaved `[re, -im]` basis columns across `[k_base..k_base + 2*P]`
    /// from delayed `s2` using packed `2`-pole IEEE-754 SIMD lanes (`0-ULP` parity).
    fn filter_pole_bank_columns(
        s2: &[f64],
        delay: usize,
        pre_lr_li_bg: &[(f64, f64, f64)],
        cols: &mut [Vec<f64>],
        k_base: usize,
    );

    /// Advances all active Kautz poles for one sample frame and returns `kv` readout sum.
    fn step_kautz_forward_frame(
        x_del: f64,
        poles: &[(f64, f64, f64, f64, f64, f64)],
        kc: &[(f64, f64)],
        s_re: &mut [f64],
        s_im: &mut [f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
    ) -> f64;

    /// Evaluates one reverse-mode adjoint frame across all active Kautz poles and returns `dx_del`.
    #[allow(clippy::too_many_arguments)]
    fn step_kautz_backward_frame(
        dy: f64,
        x_del: f64,
        poles: &[(f64, f64, f64, f64, f64, f64)],
        kc: &[(f64, f64)],
        cur_re: &[f64],
        cur_im: &[f64],
        prev_re: Option<&[f64]>,
        prev_im: Option<&[f64]>,
        ds_re: &mut [f64],
        ds_im: &mut [f64],
        grad_kc: &mut [(f64, f64)],
        grad_poles: &mut [(f64, f64)],
    ) -> f64;

    /// Gram-Schmidt orthonormalizes a complex-conjugate dictionary atom pair `(col_re, col_im)`.
    fn orthonormalize_pole_pair(col_re: &mut [f64], col_im: &mut [f64]);

    /// Computes two-channel matched energy `c_re^2 + c_im^2` against `res_w` in a single pass.
    fn score_pole_pair(res_w: &[f64], col_re: &[f64], col_im: &[f64]) -> f64;
}

impl<M: TrainerMathBackend> KautzCascadeBackend for M {
    fn filter_pole_bank_columns(
        s2: &[f64],
        delay: usize,
        pre_lr_li_bg: &[(f64, f64, f64)],
        cols: &mut [Vec<f64>],
        k_base: usize,
    ) {
        let n = s2.len();
        let k_cnt = pre_lr_li_bg.len();
        let pairs = k_cnt / 2;
        let has_tail = (k_cnt & 1) != 0;

        let mut s_re = [0.0_f64; 32];
        let mut s_im = [0.0_f64; 32];
        let (mut sr_v, mut si_v);
        if k_cnt <= 32 {
            for t in 0..n {
                let x_del = if t >= delay { s2[t - delay] } else { 0.0 };
                let x_v = Self::F64Vec2::splat(x_del);
                for p in 0..pairs {
                    let m0 = p * 2;
                    let m1 = m0 + 1;
                    let (lr0, li0, bg0) = pre_lr_li_bg[m0];
                    let (lr1, li1, bg1) = pre_lr_li_bg[m1];
                    let lr = Self::F64Vec2::from_array([lr0, lr1]);
                    let li = Self::F64Vec2::from_array([li0, li1]);
                    let bg = Self::F64Vec2::from_array([bg0, bg1]);
                    sr_v = Self::F64Vec2::from_array([s_re[m0], s_re[m1]]);
                    si_v = Self::F64Vec2::from_array([s_im[m0], s_im[m1]]);
                    let nr = lr.mul(sr_v).sub(li.mul(si_v)).add(bg.mul(x_v));
                    let ni = li.mul(sr_v).add(lr.mul(si_v));
                    let nr_arr = nr.to_array();
                    let ni_arr = ni.to_array();
                    s_re[m0] = nr_arr[0];
                    s_re[m1] = nr_arr[1];
                    s_im[m0] = ni_arr[0];
                    s_im[m1] = ni_arr[1];
                    cols[k_base + 2 * m0][t] = nr_arr[0];
                    cols[k_base + 2 * m0 + 1][t] = -ni_arr[0];
                    cols[k_base + 2 * m1][t] = nr_arr[1];
                    cols[k_base + 2 * m1 + 1][t] = -ni_arr[1];
                }
                if has_tail {
                    let m = k_cnt - 1;
                    let (lr, li, bg) = pre_lr_li_bg[m];
                    let nr = lr * s_re[m] - li * s_im[m] + bg * x_del;
                    let ni = li * s_re[m] + lr * s_im[m];
                    s_re[m] = nr;
                    s_im[m] = ni;
                    cols[k_base + 2 * m][t] = nr;
                    cols[k_base + 2 * m + 1][t] = -ni;
                }
            }
            return;
        }

        let mut dyn_re = vec![0.0_f64; k_cnt];
        let mut dyn_im = vec![0.0_f64; k_cnt];
        for t in 0..n {
            let x_del = if t >= delay { s2[t - delay] } else { 0.0 };
            for (m, &(lr, li, bg)) in pre_lr_li_bg.iter().enumerate() {
                let nr = lr * dyn_re[m] - li * dyn_im[m] + bg * x_del;
                let ni = li * dyn_re[m] + lr * dyn_im[m];
                dyn_re[m] = nr;
                dyn_im[m] = ni;
                cols[k_base + 2 * m][t] = nr;
                cols[k_base + 2 * m + 1][t] = -ni;
            }
        }
    }

    #[inline]
    fn step_kautz_forward_frame(
        x_del: f64,
        poles: &[(f64, f64, f64, f64, f64, f64)],
        kc: &[(f64, f64)],
        s_re: &mut [f64],
        s_im: &mut [f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
    ) -> f64 {
        let k_cnt = poles.len().min(kc.len()).min(s_re.len()).min(s_im.len());
        let pairs = k_cnt / 2;
        let x_v = Self::F64Vec2::splat(x_del);
        let mut kv = 0.0_f64;

        for p in 0..pairs {
            let m0 = p * 2;
            let m1 = m0 + 1;
            let (r0, bg0, ct0, st0, _, _) = poles[m0];
            let (r1, bg1, ct1, st1, _, _) = poles[m1];
            let r_v = Self::F64Vec2::from_array([r0, r1]);
            let ct_v = Self::F64Vec2::from_array([ct0, ct1]);
            let st_v = Self::F64Vec2::from_array([st0, st1]);
            let lr_v = r_v.mul(ct_v);
            let li_v = r_v.mul(st_v);
            let bg_v = Self::F64Vec2::from_array([bg0, bg1]);
            let pr_v = Self::F64Vec2::from_array([s_re[m0], s_re[m1]]);
            let pi_v = Self::F64Vec2::from_array([s_im[m0], s_im[m1]]);

            let nr_v = lr_v.mul(pr_v).sub(li_v.mul(pi_v)).add(bg_v.mul(x_v));
            let ni_v = li_v.mul(pr_v).add(lr_v.mul(pi_v));
            let nr = nr_v.to_array();
            let ni = ni_v.to_array();
            s_re[m0] = nr[0];
            s_re[m1] = nr[1];
            s_im[m0] = ni[0];
            s_im[m1] = ni[1];
            out_re[m0] = nr[0];
            out_re[m1] = nr[1];
            out_im[m0] = ni[0];
            out_im[m1] = ni[1];

            let kcr = Self::F64Vec2::from_array([kc[m0].0, kc[m1].0]);
            let kci = Self::F64Vec2::from_array([kc[m0].1, kc[m1].1]);
            let term = kcr.mul(nr_v).sub(kci.mul(ni_v)).to_array();
            kv += term[0];
            kv += term[1];
        }

        if (k_cnt & 1) != 0 {
            let m = k_cnt - 1;
            let (r, bg, ct, st, _, _) = poles[m];
            let (lr, li, pr, pi) = (r * ct, r * st, s_re[m], s_im[m]);
            let (nr, ni) = (lr * pr - li * pi + bg * x_del, li * pr + lr * pi);
            s_re[m] = nr;
            s_im[m] = ni;
            out_re[m] = nr;
            out_im[m] = ni;
            kv += kc[m].0 * nr - kc[m].1 * ni;
        }
        kv
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn step_kautz_backward_frame(
        dy: f64,
        x_del: f64,
        poles: &[(f64, f64, f64, f64, f64, f64)],
        kc: &[(f64, f64)],
        cur_re: &[f64],
        cur_im: &[f64],
        prev_re: Option<&[f64]>,
        prev_im: Option<&[f64]>,
        ds_re: &mut [f64],
        ds_im: &mut [f64],
        grad_kc: &mut [(f64, f64)],
        grad_poles: &mut [(f64, f64)],
    ) -> f64 {
        let k_cnt = poles.len().min(kc.len());
        let pairs = k_cnt / 2;
        let dy_v = Self::F64Vec2::splat(dy);
        let neg_dy_v = Self::F64Vec2::splat(-dy);
        let x_del_v = Self::F64Vec2::splat(x_del);
        let mut dx_del = 0.0_f64;

        for p in 0..pairs {
            let m0 = p * 2;
            let m1 = m0 + 1;
            let nr_v = Self::F64Vec2::from_array([cur_re[m0], cur_re[m1]]);
            let ni_v = Self::F64Vec2::from_array([cur_im[m0], cur_im[m1]]);
            let pr_v = match prev_re {
                Some(pr) => Self::F64Vec2::from_array([pr[m0], pr[m1]]),
                None => Self::F64Vec2::splat(0.0),
            };
            let pi_v = match prev_im {
                Some(pi) => Self::F64Vec2::from_array([pi[m0], pi[m1]]),
                None => Self::F64Vec2::splat(0.0),
            };
            let dk_r = nr_v.mul(dy_v).to_array();
            let dk_i = ni_v.mul(neg_dy_v).to_array();
            grad_kc[m0].0 += dk_r[0];
            grad_kc[m0].1 += dk_i[0];
            grad_kc[m1].0 += dk_r[1];
            grad_kc[m1].1 += dk_i[1];

            let kcr = Self::F64Vec2::from_array([kc[m0].0, kc[m1].0]);
            let kci = Self::F64Vec2::from_array([kc[m0].1, kc[m1].1]);
            let dsr_v = Self::F64Vec2::from_array([ds_re[m0], ds_re[m1]]);
            let dsi_v = Self::F64Vec2::from_array([ds_im[m0], ds_im[m1]]);
            let dnr_v = kcr.mul(dy_v).add(dsr_v);
            let dni_v = kci.mul(neg_dy_v).add(dsi_v);

            let (r0, bg0, ct0, st0, r_m0, t_m0) = poles[m0];
            let (r1, bg1, ct1, st1, r_m1, t_m1) = poles[m1];
            let r_v = Self::F64Vec2::from_array([r0, r1]);
            let bg_v = Self::F64Vec2::from_array([bg0, bg1]);
            let ct_v = Self::F64Vec2::from_array([ct0, ct1]);
            let st_v = Self::F64Vec2::from_array([st0, st1]);
            let lr_v = r_v.mul(ct_v);
            let li_v = r_v.mul(st_v);

            let bg_dnr = bg_v.mul(dnr_v).to_array();
            dx_del += bg_dnr[0];
            dx_del += bg_dnr[1];

            let d_lr = pr_v.mul(dnr_v).add(pi_v.mul(dni_v));
            let d_li = pr_v.mul(dni_v).sub(pi_v.mul(dnr_v));
            let d_bg = x_del_v.mul(dnr_v);
            let r_over_bg = Self::F64Vec2::from_array([r0 / bg0, r1 / bg1]);
            let rm_v = Self::F64Vec2::from_array([r_m0, r_m1]);
            let tm_r_v = Self::F64Vec2::from_array([t_m0 * r0, t_m1 * r1]);

            let gr_v = rm_v.mul(ct_v.mul(d_lr).add(st_v.mul(d_li)).sub(r_over_bg.mul(d_bg)));
            let gt_v = tm_r_v.mul(ct_v.mul(d_li).sub(st_v.mul(d_lr)));
            let gr = gr_v.to_array();
            let gt = gt_v.to_array();
            grad_poles[m0].0 += gr[0];
            grad_poles[m0].1 += gt[0];
            grad_poles[m1].0 += gr[1];
            grad_poles[m1].1 += gt[1];

            let next_dsr = lr_v.mul(dnr_v).add(li_v.mul(dni_v)).to_array();
            let next_dsi = lr_v.mul(dni_v).sub(li_v.mul(dnr_v)).to_array();
            ds_re[m0] = next_dsr[0];
            ds_re[m1] = next_dsr[1];
            ds_im[m0] = next_dsi[0];
            ds_im[m1] = next_dsi[1];
        }

        if (k_cnt & 1) != 0 {
            let m = k_cnt - 1;
            let nr = cur_re[m];
            let ni = cur_im[m];
            let pr = prev_re.map_or(0.0, |p| p[m]);
            let pi = prev_im.map_or(0.0, |p| p[m]);
            grad_kc[m].0 += nr * dy;
            grad_kc[m].1 += -ni * dy;
            let dnr = kc[m].0 * dy + ds_re[m];
            let dni = -kc[m].1 * dy + ds_im[m];
            let (r, bg, ct, st, r_m, t_m) = poles[m];
            let (lr, li) = (r * ct, r * st);
            dx_del += bg * dnr;
            let (d_lr, d_li, d_bg) = (pr * dnr + pi * dni, -pi * dnr + pr * dni, x_del * dnr);
            grad_poles[m].0 += r_m * (ct * d_lr + st * d_li - (r / bg) * d_bg);
            grad_poles[m].1 += t_m * r * (-st * d_lr + ct * d_li);
            ds_re[m] = lr * dnr + li * dni;
            ds_im[m] = -li * dnr + lr * dni;
        }
        dx_del
    }

    #[inline]
    fn orthonormalize_pole_pair(col_re: &mut [f64], col_im: &mut [f64]) {
        let inv1 = Self::sum_sq_f64(col_re).sqrt().max(1.0e-15).recip();
        Self::scale_f64(col_re, inv1);
        let proj = Self::dot_f64(col_re, col_im);
        Self::axpy_f64(col_im, -proj, col_re);
        let inv2 = Self::sum_sq_f64(col_im).sqrt().max(1.0e-15).recip();
        Self::scale_f64(col_im, inv2);
    }

    #[inline]
    fn score_pole_pair(res_w: &[f64], col_re: &[f64], col_im: &[f64]) -> f64 {
        let (c1, c2) = Self::dot_pair_f64(res_w, col_re, col_im);
        c1 * c1 + c2 * c2
    }
}
