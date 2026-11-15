#![forbid(unsafe_code)]
//! `O(P^2)` BESR Hinge + Single-Stem Welch-33 LTAS Continuation Subspace L-BFGS Polish.

use std::f64::consts::LN_10;

use dnam_engine::Tier;

use crate::metrics::TierQualityThresholds;
use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::backends::TrainerBackend;
use crate::solver::{factorize_uplo_u_cholesky, uplo_u_index, uplo_u_packed_len};

use super::super::basis::{SubspaceGramData, eval_bezout_loss_from_gram, sym_uplo_matvec};
use super::super::ltas_adj::{
    Welch33Scratch, c2_bounded_hinge, eval_welch33_minimax_loss_and_adjoint,
};
use super::super::model_f64::{V3TrainDataset, V3TrainStem};

#[inline]
fn dot_rhs(rhs: &[[f64; 2]], x: &[f64]) -> f64 {
    rhs.iter().zip(x).map(|(r, &xi)| r[0] * xi).sum()
}

struct SubspaceObjectiveCfg<'a> {
    p_dim: usize,
    grams: &'a SubspaceGramData,
    r_u: &'a [f64],
    theta_ref: &'a [f64],
    cols_stem0: &'a [Vec<f64>],
    n0: usize,
    ch: usize,
    dataset: &'a V3TrainDataset,
    lim_b: [f64; 3],
    lim_comp: f64,
    ltas_goal: f64,
    bez_ref: f64,
    bez_w: f64,
    bar_w_b: f64,
    bar_w_c: f64,
    welch_scale: f64,
    reg_w: f64,
}

struct SubspaceEvalScratch {
    gx_tot: Vec<f64>,
    gx_b: [Vec<f64>; 3],
    y_hat: Vec<f64>,
    y_prev: Vec<f64>,
    y_dir: Vec<f64>,
    dy: Vec<f64>,
    wel_sc: Welch33Scratch,
}

impl SubspaceEvalScratch {
    fn new(p_dim: usize, n0: usize) -> Self {
        Self {
            gx_tot: vec![0.0_f64; p_dim],
            gx_b: [
                vec![0.0_f64; p_dim],
                vec![0.0_f64; p_dim],
                vec![0.0_f64; p_dim],
            ],
            y_hat: vec![0.0_f64; n0],
            y_prev: vec![0.0_f64; n0],
            y_dir: vec![0.0_f64; n0],
            dy: vec![0.0_f64; n0],
            wel_sc: Welch33Scratch::default(),
        }
    }
}

struct LbfgsHistory {
    s: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    rho: Vec<f64>,
}

impl LbfgsHistory {
    fn with_capacity(cap: usize) -> Self {
        Self {
            s: Vec::with_capacity(cap),
            y: Vec::with_capacity(cap),
            rho: Vec::with_capacity(cap),
        }
    }

    fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    fn clear(&mut self) {
        self.s.clear();
        self.y.clear();
        self.rho.clear();
    }

    fn compute_direction<B: TrainerBackend>(&mut self, grad_z: &[f64], q_z: &mut [f64]) {
        for (qd, &gz) in q_z.iter_mut().zip(grad_z) {
            *qd = -gz;
        }
        if self.is_empty() {
            return;
        }
        let m = self.s.len();
        let mut alpha = vec![0.0; m];
        for i in (0..m).rev() {
            alpha[i] = self.rho[i] * B::dot_f64(&self.s[i], q_z);
            B::axpy_f64(q_z, -alpha[i], &self.y[i]);
        }
        if let (Some(sl), Some(yl)) = (self.s.last(), self.y.last()) {
            let yy = B::dot_f64(yl, yl).max(1.0e-18);
            let scale = (B::dot_f64(sl, yl) / yy).clamp(1.0e-10, 1.0e8);
            B::scale_f64(q_z, scale);
        }
        for i in 0..m {
            let beta = self.rho[i] * B::dot_f64(&self.y[i], q_z);
            B::axpy_f64(q_z, alpha[i] - beta, &self.s[i]);
        }
        let dir_dot = B::dot_f64(grad_z, q_z);
        if !dir_dot.is_finite() || dir_dot >= 0.0 {
            self.clear();
            for (qd, &gz) in q_z.iter_mut().zip(grad_z) {
                *qd = -gz;
            }
        }
    }

    fn push_pair<B: TrainerBackend>(
        &mut self,
        z: &[f64],
        prev_z: &[f64],
        grad_z: &[f64],
        prev_gz: &[f64],
    ) {
        let s: Vec<f64> = z.iter().zip(prev_z).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = grad_z.iter().zip(prev_gz).map(|(a, b)| a - b).collect();
        let sy = B::dot_f64(&s, &y);
        if sy.is_finite() && sy > 1.0e-18 {
            if self.s.len() == 12 {
                self.s.remove(0);
                self.y.remove(0);
                self.rho.remove(0);
            }
            self.s.push(s);
            self.y.push(y);
            self.rho.push(1.0 / sy);
        }
    }
}

struct SubspaceSolverState {
    theta: Vec<f64>,
    z: Vec<f64>,
    g_theta: Vec<f64>,
    grad_z: Vec<f64>,
    d_theta: Vec<f64>,
    dummy_grad: Vec<f64>,
    prev_z: Vec<f64>,
    prev_theta: Vec<f64>,
    prev_gz: Vec<f64>,
    cand_theta: Vec<f64>,
    cand_z: Vec<f64>,
    q_z: Vec<f64>,
}

impl SubspaceSolverState {
    fn new(init_theta: &[f64], init_z: &[f64]) -> Self {
        let p_dim = init_theta.len();
        Self {
            theta: init_theta.to_vec(),
            z: init_z.to_vec(),
            g_theta: vec![0.0_f64; p_dim],
            grad_z: vec![0.0_f64; p_dim],
            d_theta: vec![0.0_f64; p_dim],
            dummy_grad: vec![0.0_f64; p_dim],
            prev_z: vec![0.0; p_dim],
            prev_theta: vec![0.0; p_dim],
            prev_gz: vec![0.0; p_dim],
            cand_theta: vec![0.0; p_dim],
            cand_z: vec![0.0; p_dim],
            q_z: vec![0.0_f64; p_dim],
        }
    }
}

fn build_linear_combo<B: TrainerBackend>(
    cols: &[Vec<f64>],
    weights: &[f64],
    n: usize,
    out: &mut Vec<f64>,
) {
    out.resize(n, 0.0);
    out.fill(0.0);
    for (&w, col) in weights.iter().zip(cols) {
        if w != 0.0 {
            B::axpy_f64(&mut out[..n], w, &col[..n]);
        }
    }
}

fn eval_bezout_and_reg<B: TrainerBackend>(
    cfg: &SubspaceObjectiveCfg<'_>,
    sc: &mut SubspaceEvalScratch,
    x: &[f64],
    need_g: bool,
    g_out: &mut [f64],
) -> (f64, f64) {
    let p_dim = cfg.p_dim;
    let grams = cfg.grams;
    sym_uplo_matvec(p_dim, &grams.gram_tot, x, &mut sc.gx_tot);
    let q_term = B::dot_f64(x, &sc.gx_tot);
    let l_bez = (q_term - 2.0 * dot_rhs(&grams.rhs_tot, x) + grams.c_tot).max(0.0);

    let mut l_reg = 0.0_f64;
    for j in 0..p_dim {
        let d = x[j] - cfg.theta_ref[j];
        l_reg += cfg.reg_w * d * d;
        if need_g {
            g_out[j] +=
                2.0 * cfg.bez_w * (sc.gx_tot[j] - grams.rhs_tot[j][0]) + 2.0 * cfg.reg_w * d;
        }
    }
    (l_bez, l_reg)
}

fn eval_besr_barriers<B: TrainerBackend>(
    cfg: &SubspaceObjectiveCfg<'_>,
    sc: &mut SubspaceEvalScratch,
    x: &[f64],
    need_g: bool,
    g_out: &mut [f64],
) -> f64 {
    let p_dim = cfg.p_dim;
    let grams = cfg.grams;
    if grams.seg0 == 0 {
        return 0.0;
    }

    let ln10_inv_10 = 10.0 / LN_10;
    let mut l_bar = 0.0_f64;
    let mut esr = [0.0_f64; 3];
    let mut active_any = false;
    for b in 0..3 {
        let e_s = grams.e_sig0[b];
        if e_s > 1.0e-18 {
            active_any = true;
            sym_uplo_matvec(p_dim, &grams.gram_b0[b], x, &mut sc.gx_b[b]);
            let e_err =
                (B::dot_f64(x, &sc.gx_b[b]) - 2.0 * dot_rhs(&grams.rhs_b0[b], x) + e_s).max(0.0);
            esr[b] = e_err / e_s;
            if esr[b] > cfg.lim_b[b] {
                let ex = ln10_inv_10 * (esr[b] / cfg.lim_b[b]).ln();
                let (bv, bd) = c2_bounded_hinge(ex, 4.0, 0.35);
                l_bar += cfg.bar_w_b * bv;
                if need_g {
                    let w_b = cfg.bar_w_b * bd * ln10_inv_10 / (esr[b] * e_s);
                    for j in 0..p_dim {
                        g_out[j] += 2.0 * w_b * (sc.gx_b[b][j] - grams.rhs_b0[b][j][0]);
                    }
                }
            }
        }
    }
    let besr_lin = (esr[0] + esr[1] + esr[2]) / 3.0;
    if active_any && besr_lin > cfg.lim_comp {
        let ex_c = ln10_inv_10 * (besr_lin / cfg.lim_comp).ln();
        let (bv, bd) = c2_bounded_hinge(ex_c, 4.0, 0.35);
        l_bar += cfg.bar_w_c * bv;
        if need_g {
            let fac = cfg.bar_w_c * bd * ln10_inv_10 / (3.0 * besr_lin);
            for b in 0..3 {
                let e_s = grams.e_sig0[b];
                if e_s > 1.0e-18 {
                    let w_b = fac / e_s;
                    for j in 0..p_dim {
                        g_out[j] += 2.0 * w_b * (sc.gx_b[b][j] - grams.rhs_b0[b][j][0]);
                    }
                }
            }
        }
    }
    l_bar
}

fn eval_welch_ltas_term<B: TrainerBackend>(
    cfg: &SubspaceObjectiveCfg<'_>,
    sc: &mut SubspaceEvalScratch,
    x: &[f64],
    y_precomputed: bool,
    need_g: bool,
    g_out: &mut [f64],
) -> f64 {
    if cfg.n0 == 0 || cfg.cols_stem0.is_empty() {
        return 0.0;
    }
    let n = cfg.n0;
    if !y_precomputed {
        build_linear_combo::<B>(cfg.cols_stem0, x, n, &mut sc.y_hat);
    }
    sc.dy.resize(n, 0.0);
    sc.dy.fill(0.0);
    let sw = cfg.dataset.stem_weights[0];
    let l_spec = eval_welch33_minimax_loss_and_adjoint(
        &cfg.dataset.welch33_summaries[0][cfg.ch],
        &sc.y_hat[..n],
        need_g.then(|| &mut sc.dy[..n]),
        cfg.welch_scale * sw,
        cfg.ltas_goal,
        &mut sc.wel_sc,
    );
    if need_g {
        for (j, col) in cfg.cols_stem0.iter().enumerate() {
            g_out[j] += B::dot_f64(&col[..n], &sc.dy[..n]);
        }
    }
    l_spec
}

fn eval_subspace_obj<B: TrainerBackend>(
    cfg: &SubspaceObjectiveCfg<'_>,
    sc: &mut SubspaceEvalScratch,
    x: &[f64],
    y_precomputed: bool,
    need_g: bool,
    g_out: &mut [f64],
) -> (f64, f64) {
    g_out.fill(0.0);
    let (l_bez, l_reg) = eval_bezout_and_reg::<B>(cfg, sc, x, need_g, g_out);
    let l_bar = eval_besr_barriers::<B>(cfg, sc, x, need_g, g_out);
    let l_spec = eval_welch_ltas_term::<B>(cfg, sc, x, y_precomputed, need_g, g_out);
    (cfg.bez_w * l_bez + l_bar + l_spec + l_reg, l_bar + l_spec)
}

fn build_whitening_cholesky(p_dim: usize, gram_tot: &[f64]) -> Vec<f64> {
    let max_diag = (0..p_dim)
        .map(|j| {
            let v = gram_tot[uplo_u_index(j, j)];
            if v.is_finite() { v.abs() } else { 0.0 }
        })
        .fold(0.0_f64, f64::max);
    let mut lam = (1.0e-9 * max_diag).max(1.0e-12);
    for _ in 0..5 {
        if let Ok(r_u) = factorize_uplo_u_cholesky(p_dim, gram_tot, lam) {
            let min_diag = (0..p_dim)
                .map(|j| r_u[uplo_u_index(j, j)])
                .fold(f64::INFINITY, f64::min);
            if min_diag.is_finite() && min_diag > 1.0e-9 {
                return r_u;
            }
        }
        lam *= 100.0;
    }
    let mut r_diag = vec![0.0_f64; uplo_u_packed_len(p_dim)];
    let floor = 1.0e-6 * max_diag.max(1.0);
    for j in 0..p_dim {
        let d = gram_tot[uplo_u_index(j, j)];
        let safe = if d.is_finite() && d > 0.0 {
            d + floor
        } else {
            floor
        };
        r_diag[uplo_u_index(j, j)] = safe.sqrt();
    }
    r_diag
}

#[inline]
fn solve_rt_lower(p_dim: usize, r_u: &[f64], g_theta: &[f64], g_z: &mut [f64]) {
    for i in 0..p_dim {
        let col_i = uplo_u_index(0, i);
        let mut s = g_theta[i];
        for (&r_ki, &gz_k) in r_u[col_i..col_i + i].iter().zip(&g_z[..i]) {
            s -= r_ki * gz_k;
        }
        g_z[i] = s / r_u[col_i + i];
    }
}

#[inline]
fn solve_r_upper(p_dim: usize, r_u: &[f64], z: &[f64], d_theta: &mut [f64]) {
    d_theta[..p_dim].copy_from_slice(&z[..p_dim]);
    for k in (0..p_dim).rev() {
        let col_k = uplo_u_index(0, k);
        let dk = d_theta[k] / r_u[col_k + k];
        d_theta[k] = dk;
        for (di, &r_ik) in d_theta[..k].iter_mut().zip(&r_u[col_k..col_k + k]) {
            *di -= r_ik * dk;
        }
    }
}

fn armijo_line_search<B: TrainerBackend>(
    cfg: &SubspaceObjectiveCfg<'_>,
    sc: &mut SubspaceEvalScratch,
    st: &mut SubspaceSolverState,
    hist: &mut LbfgsHistory,
    has_cols0: bool,
    cur_obj: f64,
) -> Option<(f64, f64)> {
    let p_dim = cfg.p_dim;
    let n0 = cfg.n0;
    st.prev_z.copy_from_slice(&st.z);
    st.prev_theta.copy_from_slice(&st.theta);
    st.prev_gz.copy_from_slice(&st.grad_z);

    for attempt in 0..2 {
        if attempt == 1 {
            if hist.is_empty() {
                break;
            }
            hist.clear();
            for j in 0..p_dim {
                st.q_z[j] = -st.prev_gz[j];
            }
        }
        solve_r_upper(p_dim, cfg.r_u, &st.q_z, &mut st.d_theta);
        if has_cols0 {
            build_linear_combo::<B>(cfg.cols_stem0, &st.d_theta, n0, &mut sc.y_dir);
        }
        let init_step = if attempt == 1 || hist.is_empty() {
            let qz_norm = B::dot_f64(&st.q_z, &st.q_z).sqrt().max(1.0e-12);
            (0.15 * cfg.bez_ref.sqrt().max(1.0e-4) / qz_norm).clamp(1.0e-5, 1.0)
        } else {
            1.0
        };
        let slope = B::dot_f64(&st.prev_gz, &st.q_z);
        if !slope.is_finite() || slope >= 0.0 {
            continue;
        }
        let mut step = init_step;
        for _ls_try in 0..18 {
            for j in 0..p_dim {
                st.cand_theta[j] = st.prev_theta[j] + step * st.d_theta[j];
                st.cand_z[j] = st.prev_z[j] + step * st.q_z[j];
            }
            if has_cols0 {
                sc.y_hat[..n0].copy_from_slice(&sc.y_prev[..n0]);
                B::axpy_f64(&mut sc.y_hat[..n0], step, &sc.y_dir[..n0]);
            }
            let (co, cp) = eval_subspace_obj::<B>(
                cfg,
                sc,
                &st.cand_theta,
                has_cols0,
                false,
                &mut st.dummy_grad,
            );
            if co.is_finite() && co <= cur_obj + 1.0e-4 * step * slope {
                st.theta.copy_from_slice(&st.cand_theta);
                st.z.copy_from_slice(&st.cand_z);
                if has_cols0 {
                    sc.y_prev[..n0].copy_from_slice(&sc.y_hat[..n0]);
                }
                let (no, _) =
                    eval_subspace_obj::<B>(cfg, sc, &st.theta, has_cols0, true, &mut st.g_theta);
                solve_rt_lower(p_dim, cfg.r_u, &st.g_theta, &mut st.grad_z);
                return Some((no, cp));
            }
            step = if co.is_finite() {
                let denom = 2.0 * (co - cur_obj - slope * step);
                if denom > 1.0e-18 {
                    ((-slope * step * step) / denom).clamp(0.15 * step, 0.50 * step)
                } else {
                    0.50 * step
                }
            } else {
                0.25 * step
            };
        }
    }
    None
}

fn run_subspace_stage<B: TrainerBackend>(
    cfg: &SubspaceObjectiveCfg<'_>,
    sc: &mut SubspaceEvalScratch,
    init_theta: &[f64],
    init_z: &[f64],
    max_iters: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let p_dim = cfg.p_dim;
    let n0 = cfg.n0;
    let has_cols0 = n0 > 0 && !cfg.cols_stem0.is_empty();
    let mut st = SubspaceSolverState::new(init_theta, init_z);
    if has_cols0 {
        build_linear_combo::<B>(cfg.cols_stem0, &st.theta, n0, &mut sc.y_prev);
        sc.y_hat[..n0].copy_from_slice(&sc.y_prev[..n0]);
    }
    let (mut cur_obj, pen0) =
        eval_subspace_obj::<B>(cfg, sc, &st.theta, has_cols0, true, &mut st.g_theta);
    solve_rt_lower(p_dim, cfg.r_u, &st.g_theta, &mut st.grad_z);
    if !cur_obj.is_finite() || pen0 <= 1.0e-8 * cfg.bar_w_c.max(1.0e-12) {
        return (st.theta, st.z, pen0);
    }
    let mut best_obj = cur_obj;
    let mut best_theta = st.theta.clone();
    let mut best_z = st.z.clone();
    let mut best_pen = pen0;
    let mut hist = LbfgsHistory::with_capacity(12);

    for _iter in 0..max_iters {
        let g_norm = B::dot_f64(&st.grad_z, &st.grad_z).sqrt();
        if !g_norm.is_finite() || g_norm < 1.0e-8 {
            break;
        }
        hist.compute_direction::<B>(&st.grad_z, &mut st.q_z);
        let Some((no, cp)) =
            armijo_line_search::<B>(cfg, sc, &mut st, &mut hist, has_cols0, cur_obj)
        else {
            break;
        };
        cur_obj = no;
        if cur_obj < best_obj {
            best_obj = cur_obj;
            best_theta.copy_from_slice(&st.theta);
            best_z.copy_from_slice(&st.z);
            best_pen = cp;
        }
        hist.push_pair::<B>(&st.z, &st.prev_z, &st.grad_z, &st.prev_gz);
    }
    (best_theta, best_z, best_pen)
}

fn eval_initial_besr0<B: TrainerBackend>(
    p_dim: usize,
    grams: &SubspaceGramData,
    theta0: &[f64],
) -> f64 {
    if grams.seg0 == 0 {
        return 0.0;
    }
    let mut gx_tmp = vec![0.0_f64; p_dim];
    let mut cur_besr0 = 0.0_f64;
    let mut cnt = 0_usize;
    for b in 0..3 {
        if grams.e_sig0[b] > 1.0e-18 {
            sym_uplo_matvec(p_dim, &grams.gram_b0[b], theta0, &mut gx_tmp);
            let e_err = (B::dot_f64(theta0, &gx_tmp) - 2.0 * dot_rhs(&grams.rhs_b0[b], theta0)
                + grams.e_sig0[b])
                .max(0.0);
            cur_besr0 += e_err / grams.e_sig0[b];
            cnt += 1;
        }
    }
    if cnt > 0 { cur_besr0 / 3.0 } else { 0.0 }
}

pub(super) fn subspace_lbfgs_polish_chan<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    tier: Tier,
    prev_sum: Option<&PerceptualCertificationSummary>,
    grams: &SubspaceGramData,
    cols_stem0: &[Vec<f64>],
    ch: usize,
    theta0: &[f64],
) -> Vec<f64> {
    let p_dim = theta0.len();
    let n0 = dataset.stems.first().map_or(0, V3TrainStem::usable_len);
    if p_dim == 0 || n0 == 0 {
        return theta0.to_vec();
    }

    let thr = TierQualityThresholds::for_tier(tier);
    let mut b_db = thr.per_band_besr_db.map(|v| v - 0.30);
    let mut c_db = thr.composite_besr_db - 0.30;
    let mut ltas_goal = thr.max_ltas_db * 0.72;
    let mut has_prev_ltas = false;
    if let Some(ps) = prev_sum {
        for (dst, &prev_b) in b_db.iter_mut().zip(&ps.band_esr_db) {
            if prev_b.is_finite() {
                *dst = dst.min(prev_b - 0.02);
            }
        }
        if ps.raw_besr_db.is_finite() {
            c_db = c_db.min(ps.raw_besr_db - 0.02);
        }
        if ps.ltas_max_db.is_finite() {
            ltas_goal = ltas_goal.min((ps.ltas_max_db * 0.94).max(0.004));
            has_prev_ltas = true;
        }
    }
    if !has_prev_ltas {
        ltas_goal = ltas_goal.max(0.018);
    }
    let lim_b = b_db.map(|db| 10.0_f64.powf(db / 10.0));
    let lim_comp = 10.0_f64.powf(c_db / 10.0);

    let bez0 =
        eval_bezout_loss_from_gram(p_dim, &grams.gram_tot, &grams.rhs_tot, grams.c_tot, theta0)
            .max(1.0e-12);
    let bez_ref = bez0.max(1.0e-4);
    let cur_besr0 = eval_initial_besr0::<B>(p_dim, grams, theta0);
    let bez_w = if cur_besr0 > 0.0 && cur_besr0 < 0.5 * lim_comp {
        0.12
    } else {
        0.45
    };
    let ltas_denom = ltas_goal.max(0.04);
    let reg_w = 5.0e-5 * bez_ref / B::dot_f64(theta0, theta0).max(1.0);
    let r_u = build_whitening_cholesky(p_dim, &grams.gram_tot);
    let z0 = vec![0.0_f64; p_dim];
    let mut sc = SubspaceEvalScratch::new(p_dim, n0);

    let cfg0 = SubspaceObjectiveCfg {
        p_dim,
        grams,
        r_u: &r_u,
        theta_ref: theta0,
        cols_stem0,
        n0,
        ch,
        dataset,
        lim_b,
        lim_comp,
        ltas_goal,
        bez_ref,
        bez_w,
        bar_w_b: 4.0 * bez_ref,
        bar_w_c: 6.0 * bez_ref,
        welch_scale: 1.8 * bez_ref / (ltas_denom * ltas_denom),
        reg_w,
    };
    let (theta1, z1, pen1) = run_subspace_stage::<B>(&cfg0, &mut sc, theta0, &z0, 36);
    if pen1 <= 1.0e-9 * bez_ref {
        return theta1;
    }

    let cfg1 = SubspaceObjectiveCfg {
        bar_w_b: 14.0 * bez_ref,
        bar_w_c: 18.0 * bez_ref,
        welch_scale: 8.0 * bez_ref / (ltas_denom * ltas_denom),
        ..cfg0
    };
    run_subspace_stage::<B>(&cfg1, &mut sc, &theta1, &z1, 36).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitening_cholesky_adjoint_parity() {
        let p = 3;
        let mut gram = vec![0.0_f64; uplo_u_packed_len(p)];
        let cols = [[1.0_f64, 0.5, -0.25], [0.5, 2.0, 0.4], [-0.25, 0.4, 1.5]];
        for j in 0..p {
            for i in 0..=j {
                gram[uplo_u_index(i, j)] = cols[i][j];
            }
        }
        let r_u = build_whitening_cholesky(p, &gram);
        let g_theta = [0.7, -1.2, 0.35];
        let q_z = [-0.4, 0.9, -0.6];
        let mut g_z = [0.0_f64; 3];
        let mut d_theta = [0.0_f64; 3];
        solve_rt_lower(p, &r_u, &g_theta, &mut g_z);
        solve_r_upper(p, &r_u, &q_z, &mut d_theta);
        let lhs: f64 = g_z.iter().zip(&q_z).map(|(a, b)| a * b).sum();
        let rhs: f64 = g_theta.iter().zip(&d_theta).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1.0e-12);
    }
}
