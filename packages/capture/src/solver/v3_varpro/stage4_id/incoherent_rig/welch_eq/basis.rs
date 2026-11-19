#![forbid(unsafe_code)]
//! Precomputed Stage III (`FIR` + `Kautz` + optional `Stage II` modal) basis streams and Cholesky
//! Gram builders for `incoherent_rig/welch_eq`.

use dnam_engine::operators::op_m::algebraic_unit_rotor_op_m;
use dnam_engine::stages::stage2_gru::MAX_KAUTZ_POLES;

use crate::solver::backends::TrainerBackend;
use crate::solver::v3_varpro::primitives::par_map_indexed;

pub(super) const GRAM_CHUNK: usize = 2_048;

#[inline(always)]
fn flush_f32(x: f32) -> f32 {
    if x.abs() < f32::MIN_POSITIVE { 0.0 } else { x }
}

#[inline]
fn dot_f64_f32(a: &[f64], b: &[f32]) -> f64 {
    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let mut a4 = 0.0_f64;
    let mut a5 = 0.0_f64;
    let mut a6 = 0.0_f64;
    let mut a7 = 0.0_f64;
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let ra = ca.remainder();
    let rb = cb.remainder();
    for (xa, xb) in ca.zip(cb) {
        a0 += xa[0] * f64::from(xb[0]);
        a1 += xa[1] * f64::from(xb[1]);
        a2 += xa[2] * f64::from(xb[2]);
        a3 += xa[3] * f64::from(xb[3]);
        a4 += xa[4] * f64::from(xb[4]);
        a5 += xa[5] * f64::from(xb[5]);
        a6 += xa[6] * f64::from(xb[6]);
        a7 += xa[7] * f64::from(xb[7]);
    }
    let mut tail = 0.0_f64;
    for (&x, &y) in ra.iter().zip(rb) {
        tail += x * f64::from(y);
    }
    ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7)) + tail
}

fn accum_f32_batch(out: &mut [f64], batch: &[(&[f32], f64)]) {
    let n = out.len();
    match batch {
        [(c0, w0), (c1, w1), (c2, w2), (c3, w3)] => {
            for i in 0..n {
                out[i] += w0 * f64::from(c0[i])
                    + w1 * f64::from(c1[i])
                    + w2 * f64::from(c2[i])
                    + w3 * f64::from(c3[i]);
            }
        }
        [(c0, w0), (c1, w1), (c2, w2)] => {
            for i in 0..n {
                out[i] += w0 * f64::from(c0[i]) + w1 * f64::from(c1[i]) + w2 * f64::from(c2[i]);
            }
        }
        [(c0, w0), (c1, w1)] => {
            for i in 0..n {
                out[i] += w0 * f64::from(c0[i]) + w1 * f64::from(c1[i]);
            }
        }
        [(c0, w0)] => {
            for i in 0..n {
                out[i] += w0 * f64::from(c0[i]);
            }
        }
        _ => {}
    }
}

fn accum_f32_cols(out: &mut [f64], cols: &[Vec<f32>], weights: &[f64]) {
    let n = out.len();
    let mut idx = 0_usize;
    while idx < cols.len() {
        let mut batch: [(&[f32], f64); 4] = [(&[], 0.0); 4];
        let mut b_len = 0_usize;
        while idx < cols.len() && b_len < 4 {
            let w = weights[idx];
            if w != 0.0 {
                batch[b_len] = (&cols[idx][..n], w);
                b_len += 1;
            }
            idx += 1;
        }
        accum_f32_batch(out, &batch[..b_len]);
    }
}

fn accum_fir_cols(out: &mut [f64], v: &[f32], fir_w: &[f64]) {
    let n = out.len().min(v.len());
    let mut d = 0_usize;
    while d < fir_w.len() && d < n {
        let mut taps: [(usize, f64); 4] = [(0, 0.0); 4];
        let mut b_len = 0_usize;
        while d < fir_w.len() && d < n && b_len < 4 {
            let w = fir_w[d];
            if w != 0.0 {
                taps[b_len] = (d, w);
                b_len += 1;
            }
            d += 1;
        }
        if b_len == 0 {
            continue;
        }
        let d_max = taps[b_len - 1].0;
        for (i, o) in out[..d_max].iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for &(tap_d, td) in &taps[..b_len] {
                if i >= tap_d {
                    acc += td * f64::from(v[i - tap_d]);
                }
            }
            *o += acc;
        }
        let mut batch: [(&[f32], f64); 4] = [(&[], 0.0); 4];
        for (slot, &(tap_d, td)) in batch[..b_len].iter_mut().zip(&taps[..b_len]) {
            *slot = (&v[d_max - tap_d..n - tap_d], td);
        }
        accum_f32_batch(&mut out[d_max..n], &batch[..b_len]);
    }
}

pub(super) fn accum_upper_4x4<B: TrainerBackend>(
    p_dim: usize,
    len: usize,
    loc0: usize,
    cols: &[f64],
    inv_count: f64,
    gram: &mut [f64],
) {
    let slices: Vec<&[f64]> = (0..p_dim)
        .map(|r| &cols[r * len + loc0..(r + 1) * len])
        .collect();
    let blocks = p_dim / 4;
    let rem_start = blocks * 4;
    for bj in 0..blocks {
        let j0 = bj * 4;
        let cols_j = [slices[j0], slices[j0 + 1], slices[j0 + 2], slices[j0 + 3]];
        for bi in 0..bj {
            let i0 = bi * 4;
            let cols_i = [slices[i0], slices[i0 + 1], slices[i0 + 2], slices[i0 + 3]];
            let tile = B::gram_tile_4x4_f64(cols_i, cols_j);
            for (li, row_vals) in tile.iter().enumerate() {
                let row_base = (i0 + li) * p_dim + j0;
                for (lj, &val) in row_vals.iter().enumerate() {
                    gram[row_base + lj] += val * inv_count;
                }
            }
        }
        let sym = B::gram_sym_block4_f64(cols_j);
        let mut k = 0_usize;
        for lj in 0..4 {
            let cj = j0 + lj;
            for li in 0..=lj {
                gram[(j0 + li) * p_dim + cj] += sym[k] * inv_count;
                k += 1;
            }
        }
    }
    for cj in rem_start..p_dim {
        for ri in 0..=cj {
            gram[ri * p_dim + cj] += B::dot_f64(slices[ri], slices[cj]) * inv_count;
        }
    }
}

/// Precomputed complex Kautz pole step coefficients (`r * e^{j theta}` and orthonormal input gain).
#[derive(Clone, Copy)]
pub(super) struct KautzPolePrep {
    pub(super) l_re: f32,
    pub(super) l_im: f32,
    pub(super) bg: f32,
}

/// Prepares active Kautz poles up to `k_cnt` with BIBO clamping and orthonormal input scaling.
pub(super) fn prepare_poles(
    kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    k_cnt: usize,
) -> Vec<KautzPolePrep> {
    kautz_poles[..k_cnt]
        .iter()
        .map(|&(r_raw, th_raw)| {
            let r = r_raw.clamp(0.25, 0.9995);
            let th = th_raw.clamp(0.0, core::f32::consts::PI - 1.0e-4);
            let (cos_t, sin_t) = algebraic_unit_rotor_op_m(th);
            let bg = ((1.0 - f64::from(r) * f64::from(r)).max(1.0e-7)).sqrt() as f32;
            KautzPolePrep {
                l_re: r * cos_t,
                l_im: r * sin_t,
                bg,
            }
        })
        .collect()
}

/// Precomputed `f32` Kautz state basis (`re`, `-im`) paired with zero-copy `v` and `extras` slices.
pub(super) struct PrecomputedStage3Basis<'a> {
    pub(super) v: &'a [f32],
    pub(super) fir_len: usize,
    pub(super) k_cols: Vec<Vec<f32>>,
    pub(super) extras: &'a [Vec<f32>],
}

fn compute_kautz_pole_pair(v: &[f32], p: &KautzPolePrep) -> [Vec<f32>; 2] {
    let n = v.len();
    let mut col_re = vec![0.0_f32; n];
    let mut col_im = vec![0.0_f32; n];
    let (mut sr, mut si) = (0.0_f32, 0.0_f32);
    for i in 0..n {
        let x_del = if i >= 16 { v[i - 16] } else { 0.0 };
        let nr = flush_f32(p.l_re * sr - p.l_im * si + p.bg * x_del);
        let ni = flush_f32(p.l_im * sr + p.l_re * si);
        sr = nr;
        si = ni;
        col_re[i] = nr;
        col_im[i] = -ni;
    }
    [col_re, col_im]
}

impl<'a> PrecomputedStage3Basis<'a> {
    /// Propagates the `k_cnt` Kautz IIR columns once per stream into contiguous `f32` buffers.
    pub(super) fn new(
        v: &'a [f32],
        poles: &[KautzPolePrep],
        fir_len: usize,
        extras: &'a [Vec<f32>],
    ) -> Self {
        let n = v.len();
        let mut k_cols = Vec::with_capacity(2 * poles.len());
        if n >= 16_384 && poles.len() >= 4 {
            for [col_re, col_im] in
                par_map_indexed(poles.len(), |idx| compute_kautz_pole_pair(v, &poles[idx]))
            {
                k_cols.push(col_re);
                k_cols.push(col_im);
            }
        } else {
            for p in poles {
                let [col_re, col_im] = compute_kautz_pole_pair(v, p);
                k_cols.push(col_re);
                k_cols.push(col_im);
            }
        }
        Self {
            v,
            fir_len,
            k_cols,
            extras,
        }
    }

    #[inline]
    pub(super) fn s3_dim(&self) -> usize {
        self.fir_len + self.k_cols.len()
    }

    #[inline]
    pub(super) fn dim(&self) -> usize {
        self.s3_dim() + self.extras.len()
    }

    /// Evaluates linear forward output `out = (add_y0 ? y0 : 0) + H(theta) * v + extras * w`.
    pub(super) fn apply(&self, theta: &[f64], y0: &[f64], add_y0: bool, out: &mut [f64]) {
        let n = self.v.len().min(out.len());
        if add_y0 {
            out[..n].copy_from_slice(&y0[..n]);
        } else {
            out[..n].fill(0.0);
        }
        accum_fir_cols(&mut out[..n], self.v, &theta[..self.fir_len]);
        let s3 = self.s3_dim();
        accum_f32_cols(&mut out[..n], &self.k_cols, &theta[self.fir_len..s3]);
        accum_f32_cols(&mut out[..n], self.extras, &theta[s3..]);
    }

    /// Accumulates reverse-mode gradient `grad += dL/d(theta)` using precomputed unit-stride columns.
    pub(super) fn accum_adjoint(&self, dy: &[f64], grad: &mut [f64]) {
        let n = self.v.len().min(dy.len());
        for (d, g) in grad[..self.fir_len].iter_mut().enumerate() {
            if d < n {
                *g += dot_f64_f32(&dy[d..n], &self.v[..n - d]);
            }
        }
        let s3 = self.s3_dim();
        for (col, g) in self.k_cols.iter().zip(&mut grad[self.fir_len..s3]) {
            *g += dot_f64_f32(&dy[..n], &col[..n]);
        }
        for (ec, g) in self.extras.iter().zip(&mut grad[s3..]) {
            *g += dot_f64_f32(&dy[..n], &ec[..n]);
        }
    }

    /// Copies a contiguous sample chunk `[c0..c0 + len)` into `f64` row-major design matrix storage.
    pub(super) fn fill_chunk(&self, c0: usize, len: usize, cols: &mut [f64]) {
        for d in 0..self.fir_len {
            let dst = &mut cols[d * len..(d + 1) * len];
            for local in 0..len {
                let i = c0 + local;
                dst[local] = if i >= d {
                    f64::from(self.v[i - d])
                } else {
                    0.0
                };
            }
        }
        for (k, col) in self.k_cols.iter().enumerate() {
            let dst = &mut cols[(self.fir_len + k) * len..(self.fir_len + k + 1) * len];
            for (d, &x) in dst.iter_mut().zip(&col[c0..c0 + len]) {
                *d = f64::from(x);
            }
        }
        let s3 = self.s3_dim();
        for (e, ec) in self.extras.iter().enumerate() {
            let dst = &mut cols[(s3 + e) * len..(s3 + e + 1) * len];
            for (d, &x) in dst.iter_mut().zip(&ec[c0..c0 + len]) {
                *d = f64::from(x);
            }
        }
    }

    /// Assembles normalized `p_dim x p_dim` single-channel Gram matrix on `[warmup..n)`.
    pub(super) fn build_gram<B: TrainerBackend>(&self, warmup: usize) -> Vec<f64> {
        let p_dim = self.dim();
        let n = self.v.len();
        let inv_count = 1.0 / (n.saturating_sub(warmup).max(1) as f64);
        let chunks: Vec<(usize, usize, usize)> = (0..n)
            .step_by(GRAM_CHUNK)
            .filter_map(|c0| {
                let len = (n - c0).min(GRAM_CHUNK);
                (c0 + len > warmup).then(|| (c0, len, warmup.saturating_sub(c0)))
            })
            .collect();
        let mut gram = vec![0.0_f64; p_dim * p_dim];
        if n.saturating_sub(warmup) >= 8_192 && chunks.len() > 1 {
            let chunk_grams = par_map_indexed(chunks.len(), |idx| {
                let (c0, len, loc0) = chunks[idx];
                let mut cols = vec![0.0_f64; p_dim * len];
                self.fill_chunk(c0, len, &mut cols);
                let mut local_g = vec![0.0_f64; p_dim * p_dim];
                accum_upper_4x4::<B>(p_dim, len, loc0, &cols, inv_count, &mut local_g);
                local_g
            });
            for cg in &chunk_grams {
                for (dst, &v) in gram.iter_mut().zip(cg) {
                    *dst += v;
                }
            }
        } else {
            let mut cols = vec![0.0_f64; p_dim * GRAM_CHUNK];
            for &(c0, len, loc0) in &chunks {
                self.fill_chunk(c0, len, &mut cols[..p_dim * len]);
                accum_upper_4x4::<B>(p_dim, len, loc0, &cols[..p_dim * len], inv_count, &mut gram);
            }
        }
        for r in 0..p_dim {
            for c in 0..r {
                gram[r * p_dim + c] = gram[c * p_dim + r];
            }
        }
        gram
    }

    /// Assembles normalized `2p x 2p` coupled stereo Gram matrix on `[warmup..n)`.
    pub(super) fn build_coupled_gram<B: TrainerBackend>(
        b_ll: &Self,
        b_lr: &Self,
        b_rl: &Self,
        b_rr: &Self,
        warmup: usize,
    ) -> Vec<f64> {
        let p = b_ll.dim();
        let dim = 2 * p;
        let n = b_ll.v.len();
        let inv_count = 1.0 / (n.saturating_sub(warmup).max(1) as f64);
        let chunks: Vec<(usize, usize, usize)> = (0..n)
            .step_by(GRAM_CHUNK)
            .filter_map(|c0| {
                let len = (n - c0).min(GRAM_CHUNK);
                (c0 + len > warmup).then(|| (c0, len, warmup.saturating_sub(c0)))
            })
            .collect();
        let mut gram = vec![0.0_f64; dim * dim];
        let eval_chunk = |(c0, len, loc0): (usize, usize, usize),
                          buf_l: &mut [f64],
                          buf_r: &mut [f64],
                          out_gram: &mut [f64]| {
            let half = p * len;
            b_ll.fill_chunk(c0, len, &mut buf_l[..half]);
            b_lr.fill_chunk(c0, len, &mut buf_l[half..dim * len]);
            b_rl.fill_chunk(c0, len, &mut buf_r[..half]);
            b_rr.fill_chunk(c0, len, &mut buf_r[half..dim * len]);
            accum_upper_4x4::<B>(dim, len, loc0, &buf_l[..dim * len], inv_count, out_gram);
            accum_upper_4x4::<B>(dim, len, loc0, &buf_r[..dim * len], inv_count, out_gram);
        };
        if n.saturating_sub(warmup) >= 8_192 && chunks.len() > 1 {
            let chunk_grams = par_map_indexed(chunks.len(), |idx| {
                let len = chunks[idx].1;
                let mut local_l = vec![0.0_f64; dim * len];
                let mut local_r = vec![0.0_f64; dim * len];
                let mut local_g = vec![0.0_f64; dim * dim];
                eval_chunk(chunks[idx], &mut local_l, &mut local_r, &mut local_g);
                local_g
            });
            for cg in &chunk_grams {
                for (dst, &v) in gram.iter_mut().zip(cg) {
                    *dst += v;
                }
            }
        } else {
            let mut buf_l = vec![0.0_f64; dim * GRAM_CHUNK];
            let mut buf_r = vec![0.0_f64; dim * GRAM_CHUNK];
            for &ch in &chunks {
                eval_chunk(ch, &mut buf_l, &mut buf_r, &mut gram);
            }
        }
        for r in 0..dim {
            for c in 0..r {
                gram[r * dim + c] = gram[c * dim + r];
            }
        }
        gram
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn ref_apply(
        v: &[f32],
        poles: &[KautzPolePrep],
        fir_len: usize,
        extras: &[Vec<f32>],
        theta: &[f64],
        y0: &[f64],
        out: &mut [f64],
    ) {
        let n = v.len();
        for i in 0..n {
            let mut acc = y0[i];
            for d in 0..fir_len.min(i + 1) {
                acc += theta[d] * f64::from(v[i - d]);
            }
            out[i] = acc;
        }
        for (m, p) in poles.iter().enumerate() {
            let (cr, ci) = (theta[fir_len + 2 * m], theta[fir_len + 2 * m + 1]);
            let (mut sr, mut si) = (0.0_f32, 0.0_f32);
            for i in 0..n {
                let x_del = if i >= 16 { v[i - 16] } else { 0.0 };
                let nr = flush_f32(p.l_re * sr - p.l_im * si + p.bg * x_del);
                let ni = flush_f32(p.l_im * sr + p.l_re * si);
                sr = nr;
                si = ni;
                out[i] += cr * f64::from(nr) - ci * f64::from(ni);
            }
        }
        let s3 = fir_len + 2 * poles.len();
        for (ec, &w) in extras.iter().zip(&theta[s3..]) {
            for (o, &x) in out.iter_mut().zip(ec) {
                *o += w * f64::from(x);
            }
        }
    }

    fn ref_adjoint(
        v: &[f32],
        poles: &[KautzPolePrep],
        fir_len: usize,
        extras: &[Vec<f32>],
        dy: &[f64],
        grad: &mut [f64],
    ) {
        let n = v.len().min(dy.len());
        for d in 0..fir_len {
            let mut acc = 0.0_f64;
            for i in d..n {
                acc += dy[i] * f64::from(v[i - d]);
            }
            grad[d] += acc;
        }
        for (m, p) in poles.iter().enumerate() {
            let (mut sr, mut si) = (0.0_f32, 0.0_f32);
            let (mut gr, mut gi) = (0.0_f64, 0.0_f64);
            for i in 0..n {
                let x_del = if i >= 16 { v[i - 16] } else { 0.0 };
                let nr = flush_f32(p.l_re * sr - p.l_im * si + p.bg * x_del);
                let ni = flush_f32(p.l_im * sr + p.l_re * si);
                sr = nr;
                si = ni;
                gr += dy[i] * f64::from(nr);
                gi -= dy[i] * f64::from(ni);
            }
            grad[fir_len + 2 * m] += gr;
            grad[fir_len + 2 * m + 1] += gi;
        }
        let s3 = fir_len + 2 * poles.len();
        for (ec, g) in extras.iter().zip(&mut grad[s3..]) {
            let mut acc = 0.0_f64;
            for i in 0..n.min(ec.len()) {
                acc += dy[i] * f64::from(ec[i]);
            }
            *g += acc;
        }
    }

    #[test]
    fn test_precomputed_stage3_basis_parity_and_gram() {
        let n = 4_096;
        let fir_len = 16;
        let v: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                (0.4 * (TAU * 220.0 * t).sin() + 0.25 * (TAU * 1_850.0 * t).cos()) as f32
            })
            .collect();
        let extras = vec![vec![0.15_f32; n], vec![-0.2_f32; n]];
        let mut raw_poles = [(0.5_f32, 0.5_f32); MAX_KAUTZ_POLES];
        raw_poles[0] = (0.85, 0.12);
        raw_poles[1] = (0.93, 0.45);
        raw_poles[2] = (0.78, 1.25);
        raw_poles[3] = (0.91, 2.10);
        let poles = prepare_poles(&raw_poles, 4);
        let basis = PrecomputedStage3Basis::new(&v, &poles, fir_len, &extras);
        let p_dim = basis.dim();
        let theta: Vec<f64> = (0..p_dim).map(|j| 0.05 * ((j as f64) - 8.0)).collect();
        let y0: Vec<f64> = (0..n).map(|i| 0.01 * ((i & 7) as f64)).collect();
        let dy: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.013).sin()).collect();

        let (mut got_y, mut exp_y) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        basis.apply(&theta, &y0, true, &mut got_y);
        ref_apply(&v, &poles, fir_len, &extras, &theta, &y0, &mut exp_y);
        let max_y_diff = got_y
            .iter()
            .zip(&exp_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_y_diff < 1.0e-10, "max_y_diff={max_y_diff}");

        let (mut got_g, mut exp_g) = (vec![0.0_f64; p_dim], vec![0.0_f64; p_dim]);
        basis.accum_adjoint(&dy, &mut got_g);
        ref_adjoint(&v, &poles, fir_len, &extras, &dy, &mut exp_g);
        let max_g_diff = got_g
            .iter()
            .zip(&exp_g)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_g_diff < 1.0e-10, "max_g_diff={max_g_diff}");

        let gram = basis.build_gram::<crate::solver::backends::ScalarTrainerBackend>(256);
        assert_eq!(gram.len(), p_dim * p_dim);
        for r in 0..p_dim {
            assert!(gram[r * p_dim + r] > 0.0);
        }

        // Verify parallel build_gram (>= 8192 samples) matches naive inner products.
        let n_par = 9_216;
        let v_par: Vec<f32> = (0..n_par)
            .map(|i| ((i as f64) * 0.031).sin() as f32 * 0.35)
            .collect();
        let ex_par = vec![vec![0.12_f32; n_par]];
        let basis_par = PrecomputedStage3Basis::new(&v_par, &poles, 16, &ex_par);
        let gram_par = basis_par.build_gram::<crate::solver::backends::ScalarTrainerBackend>(512);
        let pd = basis_par.dim();
        let mut full_cols = vec![0.0_f64; pd * n_par];
        basis_par.fill_chunk(0, n_par, &mut full_cols);
        let cnt = (n_par - 512) as f64;
        for r in 0..pd {
            for c in 0..pd {
                let ref_g: f64 = full_cols[r * n_par + 512..(r + 1) * n_par]
                    .iter()
                    .zip(&full_cols[c * n_par + 512..(c + 1) * n_par])
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / cnt;
                assert!((gram_par[r * pd + c] - ref_g).abs() < 1.0e-11);
            }
        }
    }
}
