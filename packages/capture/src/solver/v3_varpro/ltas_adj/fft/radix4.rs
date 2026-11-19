#![forbid(unsafe_code)]
//! Fused radix-4 / radix-2 in-place complex FFT and pruned radix-4 subplan.

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub(super) struct Radix4PrunedSubplan {
    pub(super) m: usize,
    pub(super) sub_plan: Box<Radix2FftPlan>,
    pub(super) tw_cos: [Vec<f64>; 3],
    pub(super) tw_sin: [Vec<f64>; 3],
}

impl Radix4PrunedSubplan {
    fn new(n_fft: usize) -> Self {
        let m = n_fft / 4;
        let sub_plan = Box::new(Radix2FftPlan::new_leaf(m));
        let mut tw_cos = [
            Vec::with_capacity(m),
            Vec::with_capacity(m),
            Vec::with_capacity(m),
        ];
        let mut tw_sin = [
            Vec::with_capacity(m),
            Vec::with_capacity(m),
            Vec::with_capacity(m),
        ];
        for (r_idx, r) in (1..=3).enumerate() {
            for n in 0..m {
                let ang = -2.0 * PI * (n as f64) * (r as f64) / (n_fft as f64);
                let (s, c) = ang.sin_cos();
                tw_cos[r_idx].push(c);
                tw_sin[r_idx].push(s);
            }
        }
        Self {
            m,
            sub_plan,
            tw_cos,
            tw_sin,
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::solver::v3_varpro::ltas_adj) struct Radix2FftPlan {
    pub(in crate::solver::v3_varpro::ltas_adj) n_fft: usize,
    swap_pairs: Vec<(u32, u32)>,
    tw_cos: Vec<f64>,
    tw_sin: Vec<f64>,
    pub(super) pruned_r4: Option<Radix4PrunedSubplan>,
}

impl Radix2FftPlan {
    pub(super) fn new_leaf(n_fft: usize) -> Self {
        let mut bit_rev = vec![0usize; n_fft];
        let mut j = 0usize;
        for slot in bit_rev.iter_mut().skip(1) {
            let mut bit = n_fft >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            *slot = j;
        }
        let swap_pairs: Vec<(u32, u32)> = bit_rev
            .into_iter()
            .enumerate()
            .filter_map(|(i, rev)| (i < rev).then_some((i as u32, rev as u32)))
            .collect();
        let mut tw_cos = Vec::with_capacity(n_fft);
        let mut tw_sin = Vec::with_capacity(n_fft);
        let mut len = 2usize;
        while len <= n_fft {
            let half = len >> 1;
            for k in 0..half {
                let ang = -2.0 * PI * (k as f64) / (len as f64);
                let (s, c) = ang.sin_cos();
                tw_cos.push(c);
                tw_sin.push(s);
            }
            len <<= 1;
        }
        Self {
            n_fft,
            swap_pairs,
            tw_cos,
            tw_sin,
            pruned_r4: None,
        }
    }

    pub(super) fn new(n_fft: usize) -> Self {
        let mut plan = Self::new_leaf(n_fft);
        if n_fft >= 65_536 && (n_fft & 3) == 0 {
            plan.pruned_r4 = Some(Radix4PrunedSubplan::new(n_fft));
        }
        plan
    }

    pub(super) fn fft_in_place(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n_fft;
        if n <= 1 || re.len() < n || im.len() < n {
            return;
        }
        let re = &mut re[..n];
        let im = &mut im[..n];
        for &(i, j) in &self.swap_pairs {
            re.swap(i as usize, j as usize);
            im.swap(i as usize, j as usize);
        }
        if n == 2 {
            let (r0, i0) = (re[0], im[0]);
            let (r1, i1) = (re[1], im[1]);
            re[0] = r0 + r1;
            im[0] = i0 + i1;
            re[1] = r0 - r1;
            im[1] = i0 - i1;
            return;
        }
        // Fused zero-multiply radix-4 base pass (`len = 2` and `len = 4`).
        let mut s = 0usize;
        while s + 3 < n {
            let (r0, r1, r2, r3) = (re[s], re[s + 1], re[s + 2], re[s + 3]);
            let (i0, i1, i2, i3) = (im[s], im[s + 1], im[s + 2], im[s + 3]);
            let (p0_r, p0_i) = (r0 + r1, i0 + i1);
            let (p1_r, p1_i) = (r0 - r1, i0 - i1);
            let (p2_r, p2_i) = (r2 + r3, i2 + i3);
            let (p3_r, p3_i) = (r2 - r3, i2 - i3);
            re[s] = p0_r + p2_r;
            im[s] = p0_i + p2_i;
            re[s + 2] = p0_r - p2_r;
            im[s + 2] = p0_i - p2_i;
            re[s + 1] = p1_r + p3_i;
            im[s + 1] = p1_i - p3_r;
            re[s + 3] = p1_r - p3_i;
            im[s + 3] = p1_i + p3_r;
            s += 4;
        }
        // Fused pairs of radix-2 passes (`len1 = 2*q`, `len2 = 4*q`) with exact W^k and (-i * W^k) twiddles.
        let mut tw_base = 3usize;
        let mut q = 4usize;
        while 4 * q <= n {
            let tw_c1 = &self.tw_cos[tw_base..tw_base + q];
            let tw_s1 = &self.tw_sin[tw_base..tw_base + q];
            let tw_c2 = &self.tw_cos[tw_base + q..tw_base + 2 * q];
            let tw_s2 = &self.tw_sin[tw_base + q..tw_base + 2 * q];
            let step = 4 * q;
            let mut start = 0usize;
            while start + step <= n {
                for k in 0..q {
                    let i0 = start + k;
                    let i1 = i0 + q;
                    let i2 = i1 + q;
                    let i3 = i2 + q;
                    let (wr1, wi1) = (tw_c1[k], tw_s1[k]);
                    let (x1_r, x1_i) = (re[i1], im[i1]);
                    let (v1_r, v1_i) = (x1_r * wr1 - x1_i * wi1, x1_r * wi1 + x1_i * wr1);
                    let (r0, i0_v) = (re[i0], im[i0]);
                    let (t0_r, t0_i) = (r0 + v1_r, i0_v + v1_i);
                    let (t1_r, t1_i) = (r0 - v1_r, i0_v - v1_i);

                    let (x3_r, x3_i) = (re[i3], im[i3]);
                    let (v3_r, v3_i) = (x3_r * wr1 - x3_i * wi1, x3_r * wi1 + x3_i * wr1);
                    let (r2, i2_v) = (re[i2], im[i2]);
                    let (t2_r, t2_i) = (r2 + v3_r, i2_v + v3_i);
                    let (t3_r, t3_i) = (r2 - v3_r, i2_v - v3_i);

                    let (wr2, wi2) = (tw_c2[k], tw_s2[k]);
                    let (u2_r, u2_i) = (t2_r * wr2 - t2_i * wi2, t2_r * wi2 + t2_i * wr2);
                    let (u3_r, u3_i) = (t3_r * wi2 + t3_i * wr2, t3_i * wi2 - t3_r * wr2);
                    re[i0] = t0_r + u2_r;
                    im[i0] = t0_i + u2_i;
                    re[i2] = t0_r - u2_r;
                    im[i2] = t0_i - u2_i;
                    re[i1] = t1_r + u3_r;
                    im[i1] = t1_i + u3_i;
                    re[i3] = t1_r - u3_r;
                    im[i3] = t1_i - u3_i;
                }
                start += step;
            }
            tw_base += 3 * q;
            q <<= 2;
        }
        if 2 * q <= n {
            let tw_c = &self.tw_cos[tw_base..tw_base + q];
            let tw_s = &self.tw_sin[tw_base..tw_base + q];
            let step = 2 * q;
            let mut start = 0usize;
            while start + step <= n {
                for k in 0..q {
                    let i0 = start + k;
                    let i1 = i0 + q;
                    let (wr, wi) = (tw_c[k], tw_s[k]);
                    let (ur, ui) = (re[i0], im[i0]);
                    let (xr, xi) = (re[i1], im[i1]);
                    let (vr, vi) = (xr * wr - xi * wi, xr * wi + xi * wr);
                    re[i0] = ur + vr;
                    im[i0] = ui + vi;
                    re[i1] = ur - vr;
                    im[i1] = ui - vi;
                }
                start += step;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fft_radix2_ref(re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2usize;
        while len <= n {
            let half = len >> 1;
            let step = -2.0 * PI / (len as f64);
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let (wi, wr) = ((k as f64) * step).sin_cos();
                    let (i0, i1) = (start + k, start + k + half);
                    let (vr, vi) = (re[i1] * wr - im[i1] * wi, re[i1] * wi + im[i1] * wr);
                    (re[i0], re[i1]) = (re[i0] + vr, re[i0] - vr);
                    (im[i0], im[i1]) = (im[i0] + vi, im[i0] - vi);
                }
            }
            len <<= 1;
        }
    }

    #[test]
    fn test_fused_radix4_leaf_fft_matches_radix2_ref_all_sizes() {
        for &n in &[2_usize, 4, 8, 16, 32, 1_024, 16_384] {
            let plan = Radix2FftPlan::new_leaf(n);
            let mut re_fast: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.137).sin()).collect();
            let mut im_fast: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.271).cos()).collect();
            let (mut re_ref, mut im_ref) = (re_fast.clone(), im_fast.clone());
            plan.fft_in_place(&mut re_fast, &mut im_fast);
            fft_radix2_ref(&mut re_ref, &mut im_ref);
            let max_err = re_fast
                .iter()
                .zip(&re_ref)
                .chain(im_fast.iter().zip(&im_ref))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let tol = 1.0e-13 * (n as f64).max(10.0);
            assert!(max_err < tol, "n={n}: max_err={max_err} tol={tol}");
        }
    }
}
