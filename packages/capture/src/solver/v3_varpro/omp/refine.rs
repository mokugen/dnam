#![forbid(unsafe_code)]
//! Local `(theta, r)` coordinate micro-refinement for OMP-selected complex Kautz poles.

use std::f64::consts::PI;

use crate::solver::backends::TrainerBackend;

use super::super::basis::{KAUTZ_DELAY_SAMPLES, kautz_pre_tuples};
use super::{ChannelOmpState, kautz_pole_overlap};

const TH_MULTS: [f64; 5] = [0.970, 0.985, 1.000, 1.015, 1.030];
const ONE_MINUS_R_MULTS: [f64; 3] = [0.78, 1.00, 1.28];
const REAL_RADIUS_MULTS: [f64; 7] = [0.55, 0.72, 0.86, 1.00, 1.15, 1.32, 1.50];

pub(super) struct StemMeta {
    pub(super) n: usize,
    pub(super) w: usize,
    pub(super) seg: usize,
    pub(super) m_off: usize,
}

pub(super) struct RefineCtx<'a> {
    pub(super) active_ch: usize,
    pub(super) m_w: usize,
    pub(super) pool_meta: &'a [StemMeta],
    pub(super) sq_bw: &'a [Vec<[f64; 3]>],
    pub(super) channels: &'a [ChannelOmpState],
}

/// Output of local `(theta, r)` coordinate refinement for one selected pole.
pub(super) struct RefinedPole {
    pub(super) pole: (f64, f64),
    pub(super) basis_re_ch: Vec<Vec<f64>>,
    pub(super) basis_im_ch: Vec<Vec<f64>>,
}

fn generate_local_candidates(base_pole: (f64, f64), existing: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let (r0, th0) = base_pole;
    let mut cands = vec![base_pole];
    if th0 <= 1.0e-4 {
        for &mr in &REAL_RADIUS_MULTS {
            let rc = (1.0 - (1.0 - r0) * mr).clamp(0.30, 0.9982);
            let cand = (rc, 0.0);
            if !cands.contains(&cand)
                && existing
                    .iter()
                    .all(|&p| kautz_pole_overlap(cand, p) <= 0.85)
            {
                cands.push(cand);
            }
        }
        return cands;
    }

    for &mt in &TH_MULTS {
        let th_c = (th0 * mt).clamp(1.5e-3, PI - 1.0e-3);
        for &mr in &ONE_MINUS_R_MULTS {
            let rc = (1.0 - (1.0 - r0) * mr).clamp(0.30, 0.9982);
            let cand = (rc, th_c);
            if !cands.contains(&cand)
                && existing
                    .iter()
                    .all(|&p| kautz_pole_overlap(cand, p) <= 0.85)
            {
                cands.push(cand);
            }
        }
    }
    cands
}

fn fill_channel_candidates<B: TrainerBackend>(
    ctx: &RefineCtx<'_>,
    ch: usize,
    tuples: &[(f64, f64, f64)],
    cand_re: &mut [Vec<f64>],
    cand_im: &mut [Vec<f64>],
) {
    let cnt = tuples.len();
    for v in cand_re.iter_mut().chain(cand_im.iter_mut()) {
        v.fill(0.0);
    }
    for (si, meta) in ctx.pool_meta.iter().enumerate() {
        if meta.seg == 0 {
            continue;
        }
        let mut cols = vec![vec![0.0_f64; meta.n]; 2 * cnt];
        for (b, band) in ctx.channels[ch].s2_bands[si].iter().enumerate() {
            B::filter_pole_bank_columns(band, KAUTZ_DELAY_SAMPLES, tuples, &mut cols, 0);
            let scale = ctx.sq_bw[ch][si][b];
            let dst_off = meta.m_off + b * meta.seg;
            for ci in 0..cnt {
                let src_r = &cols[2 * ci][meta.w..meta.n];
                let src_i = &cols[2 * ci + 1][meta.w..meta.n];
                let dst_r = &mut cand_re[ci][dst_off..dst_off + meta.seg];
                let dst_i = &mut cand_im[ci][dst_off..dst_off + meta.seg];
                for k in 0..meta.seg {
                    dst_r[k] = scale * src_r[k];
                    dst_i[k] = scale * src_i[k];
                }
            }
        }
    }
    for ci in 0..cnt {
        for (prev_re, prev_im) in &ctx.channels[ch].ortho_basis {
            let pr_re = B::dot_f64(&cand_re[ci], prev_re);
            B::axpy_f64(&mut cand_re[ci], -pr_re, prev_re);
            let pi_re = B::dot_f64(&cand_re[ci], prev_im);
            B::axpy_f64(&mut cand_re[ci], -pi_re, prev_im);
            let pr_im = B::dot_f64(&cand_im[ci], prev_re);
            B::axpy_f64(&mut cand_im[ci], -pr_im, prev_re);
            let pi_im = B::dot_f64(&cand_im[ci], prev_im);
            B::axpy_f64(&mut cand_im[ci], -pi_im, prev_im);
        }
        B::orthonormalize_pole_pair(&mut cand_re[ci], &mut cand_im[ci]);
    }
}

/// Refines a dictionary-selected OMP pole over a local `5 x 3` `(theta, r)` grid,
/// returning the refined `(r, theta)` pole and its orthonormalized per-channel basis columns.
pub(super) fn refine_selected_pole_coord<B: TrainerBackend>(
    base_pole: (f64, f64),
    existing_poles: &[(f64, f64)],
    ctx: &RefineCtx<'_>,
) -> RefinedPole {
    let cands = generate_local_candidates(base_pole, existing_poles);
    let tuples = kautz_pre_tuples(&cands);
    let cnt = cands.len();
    let mut cand_re_ch: Vec<Vec<Vec<f64>>> = (0..ctx.active_ch)
        .map(|_| vec![vec![0.0_f64; ctx.m_w]; cnt])
        .collect();
    let mut cand_im_ch: Vec<Vec<Vec<f64>>> = (0..ctx.active_ch)
        .map(|_| vec![vec![0.0_f64; ctx.m_w]; cnt])
        .collect();
    for ch in 0..ctx.active_ch {
        fill_channel_candidates::<B>(ctx, ch, &tuples, &mut cand_re_ch[ch], &mut cand_im_ch[ch]);
    }

    let mut best_ci = 0_usize;
    let mut best_score = -1.0_f64;
    for ci in 0..cnt {
        let mut sc = 0.0_f64;
        for ch in 0..ctx.active_ch {
            sc += B::score_pole_pair(
                &ctx.channels[ch].res_w,
                &cand_re_ch[ch][ci],
                &cand_im_ch[ch][ci],
            );
        }
        if sc.is_finite() && sc > best_score {
            best_score = sc;
            best_ci = ci;
        }
    }

    let basis_re_ch = (0..ctx.active_ch)
        .map(|ch| std::mem::take(&mut cand_re_ch[ch][best_ci]))
        .collect();
    let basis_im_ch = (0..ctx.active_ch)
        .map(|ch| std::mem::take(&mut cand_im_ch[ch][best_ci]))
        .collect();
    RefinedPole {
        pole: cands[best_ci],
        basis_re_ch,
        basis_im_ch,
    }
}
