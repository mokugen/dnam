//! Pure-Rust IEEE-754 `f64` radix-4 / radix-2 FFT, inverse FFT, paired real-signal FFT
//! packing, symmetric Hann windows, and Welch segmentation (`ref[qual.companion.ltas]`).

use core::f64::consts::PI;

use crate::error::TrainError;

/// Precomputed in-place power-of-2 radix-4 / radix-2 `f64` FFT plan.
#[derive(Clone, Debug)]
pub struct Radix2FftPlan {
    n_fft: usize,
    swap_pairs: Vec<(u32, u32)>,
    tw_cos: Vec<f64>,
    tw_sin: Vec<f64>,
}

impl Radix2FftPlan {
    /// Constructs a radix-2/radix-4 FFT plan for power-of-two `n_fft >= 2`.
    pub fn new(n_fft: usize) -> Result<Self, TrainError> {
        if n_fft < 2 || !n_fft.is_power_of_two() {
            return Err(TrainError::InvalidDimensions {
                context: "Radix2FftPlan::new(n_fft must be power of two >= 2)",
                expected: n_fft.max(2).next_power_of_two(),
                actual: n_fft,
            });
        }
        let mut bit_rev = vec![0_usize; n_fft];
        let mut j = 0_usize;
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
        let mut len = 2_usize;
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
        Ok(Self {
            n_fft,
            swap_pairs,
            tw_cos,
            tw_sin,
        })
    }

    /// Configured FFT transform length.
    #[inline]
    #[must_use]
    pub const fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Forward discrete Fourier transform in place (`re`, `im` of length `>= n_fft`).
    pub fn fft_in_place(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n_fft;
        if re.len() < n || im.len() < n {
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
        let mut s = 0_usize;
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

        let mut tw_base = 3_usize;
        let mut q = 4_usize;
        while 4 * q <= n {
            let tw_c1 = &self.tw_cos[tw_base..tw_base + q];
            let tw_s1 = &self.tw_sin[tw_base..tw_base + q];
            let tw_c2 = &self.tw_cos[tw_base + q..tw_base + 2 * q];
            let tw_s2 = &self.tw_sin[tw_base + q..tw_base + 2 * q];
            let step = 4 * q;
            let mut start = 0_usize;
            while start + step <= n {
                for k in 0..q {
                    let i0 = start + k;
                    let i1 = i0 + q;
                    let i2 = i1 + q;
                    let i3 = i2 + q;

                    let (wr1, wi1) = (tw_c1[k], tw_s1[k]);
                    let (v1_r, v1_i) = (re[i1] * wr1 - im[i1] * wi1, re[i1] * wi1 + im[i1] * wr1);
                    let (t0_r, t0_i) = (re[i0] + v1_r, im[i0] + v1_i);
                    let (t1_r, t1_i) = (re[i0] - v1_r, im[i0] - v1_i);

                    let (v3_r, v3_i) = (re[i3] * wr1 - im[i3] * wi1, re[i3] * wi1 + im[i3] * wr1);
                    let (t2_r, t2_i) = (re[i2] + v3_r, im[i2] + v3_i);
                    let (t3_r, t3_i) = (re[i2] - v3_r, im[i2] - v3_i);

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
            let mut start = 0_usize;
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

    /// Exact normalized inverse DFT `(1 / n_fft) * sum_k X_k e^{+i 2 pi k n / n_fft}` in place.
    pub fn ifft_in_place(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n_fft;
        if re.len() < n || im.len() < n {
            return;
        }
        for v in &mut im[..n] {
            *v = -*v;
        }
        self.fft_in_place(re, im);
        let inv_n = 1.0 / (n as f64);
        for (r, i) in re[..n].iter_mut().zip(&mut im[..n]) {
            *r *= inv_n;
            *i = -(*i) * inv_n;
        }
    }
}

/// Computes symmetric Hann window weights `w[i] = 0.5 * (1 - cos(2*pi*i/(win_len-1)))` and
/// returns `(window, 1.0 / sum(w^2))` (`ref[qual.companion.ltas]`).
#[must_use]
pub fn build_hann_window(seg_len: usize) -> (Vec<f64>, f64) {
    let mut win = vec![1.0_f64; seg_len];
    let mut norm = seg_len as f64;
    if seg_len >= 8 {
        norm = 0.0;
        let denom = (seg_len - 1) as f64;
        for (i, w) in win.iter_mut().enumerate() {
            let v = 0.5 * (1.0 - (2.0 * PI * (i as f64) / denom).cos());
            *w = v;
            norm += v * v;
        }
    }
    (win, 1.0 / norm.max(1.0e-12))
}

/// Computes 50%-overlapping Welch window start offsets over `len_post`, including a trailing
/// aligned `[len_post - seg_len, len_post)` segment when the hop stride does not terminate at the tail.
#[must_use]
pub fn compute_welch_seg_starts(len_post: usize, win_cap: usize) -> (usize, Vec<usize>) {
    if len_post == 0 || win_cap == 0 {
        return (0, Vec::new());
    }
    let seg_len = len_post.min(win_cap);
    let hop = (seg_len / 2).max(1);
    let mut starts = Vec::new();
    let mut start = 0_usize;
    while start + seg_len <= len_post {
        starts.push(start);
        start += hop;
    }
    let last_start = len_post - seg_len;
    if last_start + hop != start {
        starts.push(last_start);
    }
    (seg_len, starts)
}

pub(crate) struct PairedFftWorkspace {
    pub(crate) fft_re: Vec<f64>,
    pub(crate) fft_im: Vec<f64>,
}

impl PairedFftWorkspace {
    pub(crate) fn new(n_fft: usize) -> Self {
        Self {
            fft_re: vec![0.0_f64; n_fft],
            fft_im: vec![0.0_f64; n_fft],
        }
    }
}

/// Evaluates forward DFT of up to two real windowed signals packed as `s0 + i * s1`, returning
/// one-sided spectra (`len = n_fft / 2 + 1`) cleanly separated into `(re0, im0)` and `(re1, im1)`.
pub(crate) fn paired_real_forward_spectra(
    plan: &Radix2FftPlan,
    s0: &[f64],
    s1_opt: Option<&[f64]>,
    win: &[f64],
    ws: &mut PairedFftWorkspace,
    out0: (&mut [f64], &mut [f64]),
    mut out1: Option<(&mut [f64], &mut [f64])>,
) {
    let n_fft = plan.n_fft;
    ws.fft_re[..n_fft].fill(0.0);
    ws.fft_im[..n_fft].fill(0.0);
    let seg_len = s0.len().min(win.len()).min(n_fft);
    for i in 0..seg_len {
        ws.fft_re[i] = s0[i] * win[i];
    }
    if let Some(s1) = s1_opt {
        for i in 0..seg_len.min(s1.len()) {
            ws.fft_im[i] = s1[i] * win[i];
        }
    }
    plan.fft_in_place(&mut ws.fft_re, &mut ws.fft_im);

    let half = n_fft / 2;
    for k in 0..=half {
        let (r0, i0, r1, i1) = if k == 0 || k == half {
            (ws.fft_re[k], 0.0, ws.fft_im[k], 0.0)
        } else {
            let nk = n_fft - k;
            (
                0.5 * (ws.fft_re[k] + ws.fft_re[nk]),
                0.5 * (ws.fft_im[k] - ws.fft_im[nk]),
                0.5 * (ws.fft_im[k] + ws.fft_im[nk]),
                0.5 * (ws.fft_re[nk] - ws.fft_re[k]),
            )
        };
        out0.0[k] = r0;
        out0.1[k] = i0;
        if let Some((ref mut r1_out, ref mut i1_out)) = out1 {
            r1_out[k] = r1;
            i1_out[k] = i1;
        }
    }
}

/// Evaluates reverse-mode real time-domain adjoint from one-sided frequency-domain gradients
/// `(g_re, g_im)` (`len = n_fft / 2 + 1`) for real windowed inputs `s0` (+ optional `s1`).
pub(crate) fn paired_real_backward_adjoint(
    plan: &Radix2FftPlan,
    win: &[f64],
    g0: (&[f64], &[f64]),
    g1_opt: Option<(&[f64], &[f64])>,
    ws: &mut PairedFftWorkspace,
    dy0: &mut [f64],
    mut dy1_opt: Option<&mut [f64]>,
) {
    let n_fft = plan.n_fft;
    let half = n_fft / 2;
    ws.fft_re[..n_fft].fill(0.0);
    ws.fft_im[..n_fft].fill(0.0);

    ws.fft_re[0] = g0.0[0];
    ws.fft_re[half] = g0.0[half];
    if let Some((g1_re, _)) = g1_opt {
        ws.fft_im[0] = g1_re[0];
        ws.fft_im[half] = g1_re[half];
    }
    for k in 1..half {
        let z0_re = 0.5 * g0.0[k];
        let z0_im = -0.5 * g0.1[k];
        let (z1_re, z1_im) = if let Some((g1_re, g1_im)) = g1_opt {
            (0.5 * g1_re[k], -0.5 * g1_im[k])
        } else {
            (0.0, 0.0)
        };
        let nk = n_fft - k;
        ws.fft_re[k] = z0_re - z1_im;
        ws.fft_im[k] = z0_im + z1_re;
        ws.fft_re[nk] = z0_re + z1_im;
        ws.fft_im[nk] = -z0_im + z1_re;
    }

    plan.fft_in_place(&mut ws.fft_re, &mut ws.fft_im);

    let seg_len = win.len().min(dy0.len()).min(n_fft);
    for i in 0..seg_len {
        dy0[i] += win[i] * ws.fft_re[i];
    }
    if let Some(ref mut dy1) = dy1_opt {
        let len1 = win.len().min(dy1.len()).min(n_fft);
        for i in 0..len1 {
            dy1[i] += win[i] * ws.fft_im[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_fft_roundtrip_parseval_and_paired_real_adjoint_identity() {
        assert!(matches!(
            Radix2FftPlan::new(6),
            Err(TrainError::InvalidDimensions { .. })
        ));
        let plan = Radix2FftPlan::new(32).unwrap();
        assert_eq!(plan.n_fft(), 32);
        let mut re = [0.0_f64; 32];
        let mut im = [0.0_f64; 32];
        for i in 0..32 {
            re[i] = ((i as f64) * 0.23).sin();
            im[i] = ((i as f64) * 0.41).cos() * 0.5;
        }
        let orig_re = re;
        let orig_im = im;
        let time_e: f64 = orig_re
            .iter()
            .zip(&orig_im)
            .map(|(a, b)| a * a + b * b)
            .sum();

        plan.fft_in_place(&mut re, &mut im);
        let freq_e: f64 = re.iter().zip(&im).map(|(a, b)| a * a + b * b).sum::<f64>() / 32.0;
        assert!((time_e - freq_e).abs() <= 1.0e-12);

        plan.ifft_in_place(&mut re, &mut im);
        for i in 0..32 {
            assert!((re[i] - orig_re[i]).abs() <= 1.0e-12);
            assert!((im[i] - orig_im[i]).abs() <= 1.0e-12);
        }

        // Exact adjoint identity: <g, Forward(s)> == <Backward(g), s> for two paired real signals.
        let (win, _) = build_hann_window(20);
        let s0: Vec<f64> = (0..20).map(|i| ((i as f64) * 0.37).sin()).collect();
        let s1: Vec<f64> = (0..20).map(|i| ((i as f64) * 0.19).cos()).collect();
        let mut ws = PairedFftWorkspace::new(32);
        let (mut r0, mut i0, mut r1, mut i1) =
            (vec![0.0; 17], vec![0.0; 17], vec![0.0; 17], vec![0.0; 17]);
        paired_real_forward_spectra(
            &plan,
            &s0,
            Some(&s1),
            &win,
            &mut ws,
            (&mut r0, &mut i0),
            Some((&mut r1, &mut i1)),
        );
        let g0_re: Vec<f64> = (0..17).map(|k| ((k as f64) * 0.3).cos()).collect();
        let g0_im: Vec<f64> = (0..17).map(|k| ((k as f64) * 0.5).sin()).collect();
        let g1_re: Vec<f64> = (0..17).map(|k| ((k as f64) * 0.2).sin()).collect();
        let g1_im: Vec<f64> = (0..17).map(|k| ((k as f64) * 0.7).cos()).collect();
        let lhs: f64 = (0..17)
            .map(|k| g0_re[k] * r0[k] + g0_im[k] * i0[k] + g1_re[k] * r1[k] + g1_im[k] * i1[k])
            .sum();

        let mut dy0 = vec![0.0; 20];
        let mut dy1 = vec![0.0; 20];
        paired_real_backward_adjoint(
            &plan,
            &win,
            (&g0_re, &g0_im),
            Some((&g1_re, &g1_im)),
            &mut ws,
            &mut dy0,
            Some(&mut dy1),
        );
        let rhs: f64 = s0
            .iter()
            .zip(&dy0)
            .chain(s1.iter().zip(&dy1))
            .map(|(a, b)| a * b)
            .sum();
        assert!((lhs - rhs).abs() <= 1.0e-12, "lhs={lhs} rhs={rhs}");

        let (seg_len, starts) = compute_welch_seg_starts(35, 16);
        assert_eq!(seg_len, 16);
        assert_eq!(starts.last().copied(), Some(19));
    }

    proptest! {
        #[test]
        fn prop_fft_roundtrip_arbitrary_signal(
            xs in prop::collection::vec(-4.0_f64..4.0_f64, 64),
        ) {
            let plan = Radix2FftPlan::new(64).unwrap();
            let mut re = xs.clone();
            let mut im = vec![0.0_f64; 64];
            plan.fft_in_place(&mut re, &mut im);
            plan.ifft_in_place(&mut re, &mut im);
            for (a, b) in re.iter().zip(&xs) {
                prop_assert!((a - b).abs() <= 1.0e-12);
            }
            for v in im {
                prop_assert!(v.abs() <= 1.0e-12);
            }
        }
    }
}
