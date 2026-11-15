#![forbid(unsafe_code)]
//! Pre-Split 3-Band Bézout Stem Cache & `O(P^2)` Per-Band Gram/RHS Assembly.

use dnam_engine::Tier;
use dnam_engine::stages::v3_core::cascade::{
    MAX_BLOCK_LANES, NUM_GRU_BLOCKS, tier_block_lanes, tier_fir_segments, tier_kautz_banks,
};

use crate::solver::backends::TrainerBackend;
use crate::solver::{uplo_u_index, uplo_u_packed_len};

use super::super::model_f64::{Bezout7FilterBank, V3TrainDataset};
use super::super::primitives::par_map_indexed;
use super::{KAUTZ_DELAY_SAMPLES, StemChanTraj, kautz_pre_tuples, linear_basis_dim_with_k};

fn split_slice_3band(
    bank: &Bezout7FilterBank,
    sig: &[f64],
    w: usize,
    n: usize,
    split_buf: &mut [Vec<f64>; 3],
    out_b: &mut [Vec<f64>; 3],
) {
    bank.split_3band(sig, split_buf);
    for b in 0..3 {
        out_b[b].clear();
        out_b[b].extend_from_slice(&split_buf[b][w..n]);
    }
}

/// Unweighted `[LF, MF, HF]` degree-7 Bézout segments `[w..n]` cached once per channel/stem.
pub(in crate::solver::v3_varpro) struct PreSplitStemChan {
    pub(in crate::solver::v3_varpro) n: usize,
    pub(in crate::solver::v3_varpro) w: usize,
    pub(in crate::solver::v3_varpro) seg: usize,
    pub(in crate::solver::v3_varpro) bw: [f64; 3],
    pub(in crate::solver::v3_varpro) c_max: usize,
    pub(in crate::solver::v3_varpro) tap_b: [[Vec<f64>; 7]; 3],
    pub(in crate::solver::v3_varpro) u_b: [Vec<f64>; 3],
    pub(in crate::solver::v3_varpro) hp_b: [Vec<f64>; 3],
    pub(in crate::solver::v3_varpro) h_b: [Vec<Vec<f64>>; 3],
    pub(in crate::solver::v3_varpro) tgt_b: [Vec<f64>; 3],
    pub(in crate::solver::v3_varpro) e_sig: [f64; 3],
}

impl PreSplitStemChan {
    /// Splits shared Stage I/II trajectories (`tap, u, hp, h`) and target into `[LF, MF, HF]` once per stem.
    pub(in crate::solver::v3_varpro) fn build_vec<B: TrainerBackend>(
        dataset: &V3TrainDataset,
        trajs: &[StemChanTraj],
        ch: usize,
        c_max: usize,
    ) -> Vec<Self> {
        let mut out = Vec::with_capacity(dataset.stems.len());

        for (si, st) in dataset.stems.iter().enumerate() {
            let n = st.usable_len();
            let w = st.warmup.min(n);
            let seg = n.saturating_sub(w);
            let bw = dataset.stem_band_weights[si][ch];
            let mut tap_b: [[Vec<f64>; 7]; 3] = Default::default();
            let mut u_b: [Vec<f64>; 3] = Default::default();
            let mut hp_b: [Vec<f64>; 3] = Default::default();
            let mut h_b = [
                vec![Vec::new(); NUM_GRU_BLOCKS * c_max],
                vec![Vec::new(); NUM_GRU_BLOCKS * c_max],
                vec![Vec::new(); NUM_GRU_BLOCKS * c_max],
            ];
            let mut tgt_b: [Vec<f64>; 3] = Default::default();
            let mut e_sig = [0.0_f64; 3];

            if seg > 0 {
                let tr = &trajs[si];
                let tgt = if ch == 0 {
                    &st.tgt_l[..n]
                } else {
                    &st.tgt_r[..n]
                };
                let total_jobs = 9 + NUM_GRU_BLOCKS * c_max + 1;
                let build_job = |job: usize| -> [Vec<f64>; 3] {
                    let mut tmp_sig = vec![0.0_f64; n];
                    if job < 7 {
                        for t in 0..n {
                            tmp_sig[t] = tr.tap[t][job];
                        }
                    } else if job == 7 {
                        tmp_sig.copy_from_slice(&tr.u[..n]);
                    } else if job == 8 {
                        tmp_sig.copy_from_slice(&tr.hp[..n]);
                    } else if job < 9 + NUM_GRU_BLOCKS * c_max {
                        let h_idx = job - 9;
                        let blk = h_idx / c_max;
                        let l = h_idx % c_max;
                        let col = blk * MAX_BLOCK_LANES + l;
                        for t in 0..n {
                            tmp_sig[t] = tr.h[t][col];
                        }
                    } else {
                        for (t, &v) in tgt.iter().enumerate() {
                            tmp_sig[t] = f64::from(v);
                        }
                    }
                    let mut split_buf = [Vec::new(), Vec::new(), Vec::new()];
                    let mut out_3 = [Vec::new(), Vec::new(), Vec::new()];
                    split_slice_3band(&dataset.bank, &tmp_sig, w, n, &mut split_buf, &mut out_3);
                    out_3
                };
                let jobs: Vec<[Vec<f64>; 3]> = if n >= 8_192 {
                    par_map_indexed(total_jobs, build_job)
                } else {
                    (0..total_jobs).map(build_job).collect()
                };
                let mut it = jobs.into_iter();
                for i in 0..7 {
                    let [b0, b1, b2] = it.next().unwrap_or_default();
                    tap_b[0][i] = b0;
                    tap_b[1][i] = b1;
                    tap_b[2][i] = b2;
                }
                u_b = it.next().unwrap_or_default();
                hp_b = it.next().unwrap_or_default();
                for blk in 0..NUM_GRU_BLOCKS {
                    let cache_base = blk * c_max;
                    for l in 0..c_max {
                        let [b0, b1, b2] = it.next().unwrap_or_default();
                        h_b[0][cache_base + l] = b0;
                        h_b[1][cache_base + l] = b1;
                        h_b[2][cache_base + l] = b2;
                    }
                }
                tgt_b = it.next().unwrap_or_default();
                for b in 0..3 {
                    e_sig[b] = B::dot_f64(&tgt_b[b], &tgt_b[b]);
                }
            }

            out.push(Self {
                n,
                w,
                seg,
                bw,
                c_max,
                tap_b,
                u_b,
                hp_b,
                h_b,
                tgt_b,
                e_sig,
            });
        }
        out
    }
}

/// Pre-assembled weighted normal equations across all stems plus stem-0 unweighted per-band matrices.
pub(in crate::solver::v3_varpro) struct SubspaceGramData {
    pub(in crate::solver::v3_varpro) p_dim: usize,
    pub(in crate::solver::v3_varpro) gram_tot: Vec<f64>,
    pub(in crate::solver::v3_varpro) rhs_tot: Vec<[f64; 2]>,
    pub(in crate::solver::v3_varpro) c_tot: f64,
    pub(in crate::solver::v3_varpro) seg0: usize,
    pub(in crate::solver::v3_varpro) gram_b0: [Vec<f64>; 3],
    pub(in crate::solver::v3_varpro) rhs_b0: [Vec<[f64; 2]>; 3],
    pub(in crate::solver::v3_varpro) e_sig0: [f64; 3],
}

#[inline]
pub(in crate::solver::v3_varpro) fn sym_uplo_matvec(
    p: usize,
    gram_u: &[f64],
    x: &[f64],
    out: &mut [f64],
) {
    out.fill(0.0);
    for j in 0..p {
        let xj = x[j];
        for i in 0..j {
            let g = gram_u[uplo_u_index(i, j)];
            out[i] += g * xj;
            out[j] += g * x[i];
        }
        out[j] += gram_u[uplo_u_index(j, j)] * xj;
    }
}

/// Evaluates `O(P^2)` Bézout quadratic loss `theta^T G theta - 2 r^T theta + c_tot`.
#[must_use]
pub(in crate::solver::v3_varpro) fn eval_bezout_loss_from_gram(
    p_dim: usize,
    gram_tot: &[f64],
    rhs_tot: &[[f64; 2]],
    c_tot: f64,
    theta: &[f64],
) -> f64 {
    let mut gx = vec![0.0_f64; p_dim];
    sym_uplo_matvec(p_dim, gram_tot, theta, &mut gx);
    let q_term: f64 = theta.iter().zip(&gx).map(|(&x, &g)| x * g).sum();
    let l_term: f64 = theta.iter().zip(rhs_tot).map(|(&x, r)| x * r[0]).sum();
    (q_term - 2.0 * l_term + c_tot).max(0.0)
}

fn assemble_uplo_gram_and_rhs_slices<B: TrainerBackend>(
    phi_w: &[&[f64]],
    y_w: &[f64],
    gram: &mut [f64],
    rhs: &mut [[f64; 2]],
) {
    let p_dim = phi_w.len();
    let blocks = p_dim / 4;
    let rem_start = blocks * 4;
    for bj in 0..blocks {
        let j0 = bj * 4;
        let cols_j = [phi_w[j0], phi_w[j0 + 1], phi_w[j0 + 2], phi_w[j0 + 3]];
        for bi in 0..bj {
            let i0 = bi * 4;
            let cols_i = [phi_w[i0], phi_w[i0 + 1], phi_w[i0 + 2], phi_w[i0 + 3]];
            let tile = B::gram_tile_4x4_f64(cols_i, cols_j);
            for (lj, cj) in (j0..j0 + 4).enumerate() {
                let base = uplo_u_index(i0, cj);
                for (li, &row_vals) in tile.iter().enumerate() {
                    gram[base + li] = row_vals[lj];
                }
            }
        }
        let sym = B::gram_sym_block4_f64(cols_j);
        let mut k = 0_usize;
        for lj in 0..4 {
            let cj = j0 + lj;
            let base = uplo_u_index(j0, cj);
            for li in 0..=lj {
                gram[base + li] = sym[k];
                k += 1;
            }
        }
    }
    for j in rem_start..p_dim {
        let base = uplo_u_index(0, j);
        for i in 0..=j {
            gram[base + i] = B::dot_f64(phi_w[i], phi_w[j]);
        }
    }
    for (j, col) in phi_w.iter().enumerate() {
        rhs[j] = [B::dot_f64(col, y_w), 0.0];
    }
}

/// Accumulates one single-band zero-copy slice set `(col_b, tgt_b)` into weighted `(gram, rhs)`.
pub(super) fn accum_band_gram_and_rhs_slices<B: TrainerBackend>(
    col_b: &[&[f64]],
    tgt_b: &[f64],
    bw: f64,
    g_scratch: &mut [f64],
    r_scratch: &mut [[f64; 2]],
    gram: &mut [f64],
    rhs: &mut [[f64; 2]],
) {
    assemble_uplo_gram_and_rhs_slices::<B>(col_b, tgt_b, g_scratch, r_scratch);
    for (d, &v) in gram.iter_mut().zip(g_scratch.iter()) {
        *d += bw * v;
    }
    for (d, v) in rhs.iter_mut().zip(r_scratch.iter()) {
        d[0] += bw * v[0];
    }
}

/// Assembles `SubspaceGramData` for `tier` by splitting only `s2_per_stem` (`O(3)` splits per stem)
/// and exploiting causal zero-state LTI commutativity between Bézout, FIR shifts, and Kautz.
pub(in crate::solver::v3_varpro) fn assemble_tier_band_grams<B: TrainerBackend>(
    tier: Tier,
    poles: &[(f64, f64)],
    k_keep: usize,
    bank: &Bezout7FilterBank,
    stems_pre: &[PreSplitStemChan],
    s2_per_stem: &[Vec<f64>],
) -> SubspaceGramData {
    let c_t = tier_block_lanes(tier);
    let f_t = tier_fir_segments(tier) * 16;
    let k_t = (tier_kautz_banks(tier) * 4).min(k_keep);
    let p_dim = linear_basis_dim_with_k(tier, k_t);
    let pk_len = uplo_u_packed_len(p_dim);

    let mut gram_tot = vec![0.0_f64; pk_len];
    let mut rhs_tot = vec![[0.0_f64; 2]; p_dim];
    let mut c_tot = 0.0_f64;
    let mut seg0 = 0_usize;
    let mut gram_b0 = [Vec::new(), Vec::new(), Vec::new()];
    let mut rhs_b0 = [Vec::new(), Vec::new(), Vec::new()];
    let mut e_sig0 = [0.0_f64; 3];

    let pre_poles = if k_t > 0 {
        kautz_pre_tuples(&poles[..k_t])
    } else {
        Vec::new()
    };
    let mut s2_split = [Vec::new(), Vec::new(), Vec::new()];

    for (si, st) in stems_pre.iter().enumerate() {
        if st.seg == 0 {
            continue;
        }
        let (n, w, seg) = (st.n, st.w, st.seg);
        bank.split_3band(&s2_per_stem[si][..n], &mut s2_split);
        let compute_band = |b: usize| -> (Vec<f64>, Vec<[f64; 2]>) {
            let s2_band = &s2_split[b][..n];
            let mut kautz_full = if k_t > 0 {
                let mut kf = vec![vec![0.0_f64; n]; 2 * k_t];
                B::filter_pole_bank_columns(s2_band, KAUTZ_DELAY_SAMPLES, &pre_poles, &mut kf, 0);
                kf
            } else {
                Vec::new()
            };
            let fir_pad: Vec<Option<Vec<f64>>> = (0..f_t)
                .map(|k| {
                    (k > w).then(|| {
                        let mut v = vec![0.0_f64; seg];
                        let z_len = (k - w).min(seg);
                        v[z_len..].copy_from_slice(&s2_band[..seg - z_len]);
                        v
                    })
                })
                .collect();
            let mut col_refs: Vec<&[f64]> = Vec::with_capacity(p_dim);
            for i in 0..7 {
                col_refs.push(&st.tap_b[b][i]);
            }
            col_refs.push(&st.u_b[b]);
            col_refs.push(&st.hp_b[b]);
            for blk in 0..NUM_GRU_BLOCKS {
                let cache_base = blk * st.c_max;
                for l in 0..c_t {
                    col_refs.push(&st.h_b[b][cache_base + l]);
                }
            }
            for (k, pad) in fir_pad.iter().enumerate() {
                match pad {
                    None => col_refs.push(&s2_band[w - k..n - k]),
                    Some(v) => col_refs.push(v.as_slice()),
                }
            }
            for kf in &mut kautz_full {
                col_refs.push(&kf[w..n]);
            }
            let mut g_b = vec![0.0_f64; pk_len];
            let mut r_b = vec![[0.0_f64; 2]; p_dim];
            assemble_uplo_gram_and_rhs_slices::<B>(&col_refs, &st.tgt_b[b], &mut g_b, &mut r_b);
            (g_b, r_b)
        };
        let band_res = if seg >= 8_192 {
            par_map_indexed(3, compute_band)
        } else {
            (0..3).map(compute_band).collect()
        };
        for (b, (g_b, r_b)) in band_res.into_iter().enumerate() {
            let bw = st.bw[b];
            for (d, &v) in gram_tot.iter_mut().zip(g_b.iter()) {
                *d += bw * v;
            }
            for (d, v) in rhs_tot.iter_mut().zip(r_b.iter()) {
                d[0] += bw * v[0];
            }
            c_tot += bw * st.e_sig[b];
            if si == 0 {
                seg0 = seg;
                gram_b0[b] = g_b;
                rhs_b0[b] = r_b;
                e_sig0[b] = st.e_sig[b];
            }
        }
    }

    SubspaceGramData {
        p_dim,
        gram_tot,
        rhs_tot,
        c_tot,
        seg0,
        gram_b0,
        rhs_b0,
        e_sig0,
    }
}
