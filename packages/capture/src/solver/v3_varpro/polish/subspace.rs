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
    Welch33Scratch, Welch33TargetSummary, c2_bounded_hinge, eval_welch33_minimax_loss_and_adjoint,
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
    w0: usize,
    n0: usize,
    welch_sum0: &'a Welch33TargetSummary,
    welch_sw0: f64,
    lim_b: [f64; 3],
    lim_comp: f64,
    ltas_goal: f64,
    bez_ref: f64,
    bez_w: f64,
    bar_w_b: f64,
    bar_w_c: f64,
    welch_scale: f64,
    reg_w: f64,
    total_h: usize,
    dir_w_budget_sq: f64,
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
    cand_g: Vec<f64>,
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
            cand_g: vec![0.0_f64; p_dim],
            q_z: vec![0.0_f64; p_dim],
        }
    }
}

fn build_linear_combo(
    cols: &[Vec<f64>],
    weights: &[f64],
    w0: usize,
    n0: usize,
    out: &mut Vec<f64>,
) {
    out.resize(n0, 0.0);
    let out_s = &mut out[w0..n0];
    out_s.fill(0.0);
    let mut k = 0;
    let m = weights.len().min(cols.len());
    while k + 4 <= m {
        let (a0, a1, a2, a3) = (weights[k], weights[k + 1], weights[k + 2], weights[k + 3]);
        let (c0, c1, c2, c3) = (
            &cols[k][w0..n0],
            &cols[k + 1][w0..n0],
            &cols[k + 2][w0..n0],
            &cols[k + 3][w0..n0],
        );
        for t in 0..out_s.len() {
            out_s[t] += a0 * c0[t] + a1 * c1[t] + a2 * c2[t] + a3 * c3[t];
        }
        k += 4;
    }
    while k < m {
        let a = weights[k];
        if a != 0.0 {
            let c = &cols[k][w0..n0];
            for t in 0..out_s.len() {
                out_s[t] += a * c[t];
            }
        }
        k += 1;
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
    let w_start = 9.min(p_dim);
    let w_end = (9 + cfg.total_h).min(p_dim);
    if w_end > w_start {
        let dir_sq: f64 = x[w_start..w_end].iter().map(|v| v * v).sum();
        if dir_sq > cfg.dir_w_budget_sq {
            let ex = dir_sq - cfg.dir_w_budget_sq;
            let w_pen = 50.0 * cfg.bez_ref;
            l_reg += w_pen * ex * ex;
            if need_g {
                for (g_j, &x_j) in g_out[w_start..w_end].iter_mut().zip(&x[w_start..w_end]) {
                    *g_j += 4.0 * w_pen * ex * x_j;
                }
            }
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
    let w0 = cfg.w0;
    let n = cfg.n0;
    if !y_precomputed {
        build_linear_combo(cfg.cols_stem0, x, w0, n, &mut sc.y_hat);
    }
    sc.dy.resize(n, 0.0);
    sc.dy[w0..n].fill(0.0);
    let l_spec = eval_welch33_minimax_loss_and_adjoint(
        cfg.welch_sum0,
        &sc.y_hat[..n],
        need_g.then(|| &mut sc.dy[..n]),
        cfg.welch_scale * cfg.welch_sw0,
        cfg.ltas_goal,
        &mut sc.wel_sc,
    );
    if need_g {
        let dy_s = &sc.dy[w0..n];
        let mut j = 0;
        let len = cfg.cols_stem0.len();
        while j + 4 <= len {
            let (c0, c1, c2, c3) = (
                &cfg.cols_stem0[j][w0..n],
                &cfg.cols_stem0[j + 1][w0..n],
                &cfg.cols_stem0[j + 2][w0..n],
                &cfg.cols_stem0[j + 3][w0..n],
            );
            let (mut d0, mut d1, mut d2, mut d3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for t in 0..dy_s.len() {
                let d = dy_s[t];
                d0 += c0[t] * d;
                d1 += c1[t] * d;
                d2 += c2[t] * d;
                d3 += c3[t] * d;
            }
            g_out[j] += d0;
            g_out[j + 1] += d1;
            g_out[j + 2] += d2;
            g_out[j + 3] += d3;
            j += 4;
        }
        while j < len {
            g_out[j] += B::dot_f64(&cfg.cols_stem0[j][w0..n], dy_s);
            j += 1;
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
    let welch_ran = cfg.n0 > 0 && !cfg.cols_stem0.is_empty();
    let ltas_tau = (0.85 * cfg.ltas_goal).max(0.012);
    let constraints_met = l_bar <= 0.0 && (!welch_ran || sc.wel_sc.last_max_abs_db <= ltas_tau);
    (
        cfg.bez_w * l_bez + l_bar + l_spec + l_reg,
        if constraints_met { 0.0 } else { l_bar + l_spec },
    )
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
    let w0 = cfg.w0;
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
            build_linear_combo(cfg.cols_stem0, &st.d_theta, w0, n0, &mut sc.y_dir);
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
        for ls_try in 0..8 {
            for j in 0..p_dim {
                st.cand_theta[j] = st.prev_theta[j] + step * st.d_theta[j];
                st.cand_z[j] = st.prev_z[j] + step * st.q_z[j];
            }
            if has_cols0 {
                sc.y_hat[w0..n0].copy_from_slice(&sc.y_prev[w0..n0]);
                B::axpy_f64(&mut sc.y_hat[w0..n0], step, &sc.y_dir[w0..n0]);
            }
            let have_grad = ls_try == 0;
            let (co, cp) = eval_subspace_obj::<B>(
                cfg,
                sc,
                &st.cand_theta,
                has_cols0,
                have_grad,
                if have_grad {
                    &mut st.cand_g
                } else {
                    &mut st.dummy_grad
                },
            );
            if co.is_finite() && co <= cur_obj + 1.0e-4 * step * slope {
                st.theta.copy_from_slice(&st.cand_theta);
                st.z.copy_from_slice(&st.cand_z);
                if has_cols0 {
                    sc.y_prev[w0..n0].copy_from_slice(&sc.y_hat[w0..n0]);
                }
                let no = if have_grad {
                    st.g_theta.copy_from_slice(&st.cand_g);
                    co
                } else {
                    eval_subspace_obj::<B>(cfg, sc, &st.theta, has_cols0, true, &mut st.g_theta).0
                };
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
    let w0 = cfg.w0;
    let n0 = cfg.n0;
    let has_cols0 = n0 > w0 && !cfg.cols_stem0.is_empty();
    let mut st = SubspaceSolverState::new(init_theta, init_z);
    if has_cols0 {
        build_linear_combo(cfg.cols_stem0, &st.theta, w0, n0, &mut sc.y_prev);
        sc.y_hat[w0..n0].copy_from_slice(&sc.y_prev[w0..n0]);
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

    let mut small_improve_streak = 0_usize;
    for _iter in 0..max_iters {
        let g_norm = B::dot_f64(&st.grad_z, &st.grad_z).sqrt();
        if !g_norm.is_finite() || g_norm < 1.0e-8 {
            break;
        }
        hist.compute_direction::<B>(&st.grad_z, &mut st.q_z);
        let prev_obj = cur_obj;
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
        if cp <= 1.0e-6 * cfg.bar_w_c.max(1.0e-12) {
            break;
        }
        if (prev_obj - cur_obj).abs() <= 1.0e-6 * prev_obj.abs().max(1.0e-12) {
            small_improve_streak += 1;
            if small_improve_streak >= 2 {
                break;
            }
        } else {
            small_improve_streak = 0;
        }
        hist.push_pair::<B>(&st.z, &st.prev_z, &st.grad_z, &st.prev_gz);
    }
    (best_theta, best_z, best_pen)
}

fn eval_initial_besr0_stats<B: TrainerBackend>(
    p_dim: usize,
    grams: &SubspaceGramData,
    theta0: &[f64],
) -> (f64, [f64; 3], f64) {
    if grams.seg0 == 0 {
        return (0.0, [f64::NEG_INFINITY; 3], f64::NEG_INFINITY);
    }
    let mut gx_tmp = vec![0.0_f64; p_dim];
    let mut cur_besr0 = 0.0_f64;
    let mut band_db = [f64::NEG_INFINITY; 3];
    let mut cnt = 0_usize;
    for b in 0..3 {
        if grams.e_sig0[b] > 1.0e-18 {
            sym_uplo_matvec(p_dim, &grams.gram_b0[b], theta0, &mut gx_tmp);
            let e_err = (B::dot_f64(theta0, &gx_tmp) - 2.0 * dot_rhs(&grams.rhs_b0[b], theta0)
                + grams.e_sig0[b])
                .max(0.0);
            let frac = e_err / grams.e_sig0[b];
            cur_besr0 += frac;
            band_db[b] = 10.0 * frac.max(1.0e-12).log10();
            cnt += 1;
        }
    }
    let mean_frac = if cnt > 0 { cur_besr0 / 3.0 } else { 0.0 };
    let comp_db = if cnt > 0 {
        10.0 * mean_frac.max(1.0e-12).log10()
    } else {
        f64::NEG_INFINITY
    };
    (mean_frac, band_db, comp_db)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn subspace_lbfgs_polish_chan<B: TrainerBackend>(
    dataset: &V3TrainDataset,
    cert_welch0: Option<&Welch33TargetSummary>,
    tier: Tier,
    prev_sum: Option<&PerceptualCertificationSummary>,
    grams: &SubspaceGramData,
    cols_stem0: &[Vec<f64>],
    ch: usize,
    total_h: usize,
    dir_w_budget_sq: f64,
    theta0: &[f64],
) -> Vec<f64> {
    let p_dim = theta0.len();
    let use_cert = cert_welch0
        .filter(|cw| cw.len > 0 && cols_stem0.first().is_some_and(|c| c.len() == cw.len));
    let (welch_sum0, welch_sw0, n0) = match use_cert {
        Some(cw) => (cw, 1.0_f64, cw.len),
        None => (
            &dataset.welch33_summaries[0][ch],
            dataset.stem_weights[0],
            dataset.stems.first().map_or(0, V3TrainStem::usable_len),
        ),
    };
    if p_dim == 0 || n0 == 0 {
        return theta0.to_vec();
    }
    let w0 = welch_sum0.warmup.min(n0);

    let (cur_besr0, init_band_db, init_comp_db) =
        eval_initial_besr0_stats::<B>(p_dim, grams, theta0);
    let thr = TierQualityThresholds::for_tier(tier);
    let mut b_db: [f64; 3] = std::array::from_fn(|b| {
        if init_band_db[b] <= thr.per_band_besr_db[b] - 1.80 {
            thr.per_band_besr_db[b] - 1.50
        } else if init_band_db[b] <= thr.per_band_besr_db[b] - 1.15 {
            (init_band_db[b] + 0.25).min(thr.per_band_besr_db[b] - 0.95)
        } else {
            (init_band_db[b] + 0.35).min(thr.per_band_besr_db[b] - 0.30)
        }
    });
    let mut c_db = if init_comp_db <= thr.composite_besr_db - 1.80 {
        thr.composite_besr_db - 1.50
    } else {
        (init_comp_db + 0.35).min(thr.composite_besr_db - 0.30)
    };
    let mut ltas_goal = thr.max_ltas_db * 0.64;
    let mut has_prev_ltas = false;
    if let Some(ps) = prev_sum {
        for (b, (dst, &prev_b)) in b_db.iter_mut().zip(&ps.band_esr_db).enumerate() {
            if prev_b.is_finite() {
                let mono_gap = if init_band_db[b] <= prev_b - 0.45 {
                    0.22
                } else {
                    0.06
                };
                *dst = dst.min(prev_b - mono_gap);
            }
        }
        if ps.raw_besr_db.is_finite() {
            let comp_gap = if init_comp_db <= ps.raw_besr_db - 0.45 {
                0.20
            } else {
                0.06
            };
            c_db = c_db.min(ps.raw_besr_db - comp_gap);
        }
        if ps.ltas_max_db.is_finite() {
            let ltas_mult = if tier == Tier::Ultra
                && ps.raw_besr_db <= -42.0
                && ps.ltas_max_db > 0.06
                && ps.ltas_max_db <= 0.08
            {
                0.36
            } else if tier == Tier::Ultra
                && ps.raw_besr_db <= -35.0
                && ps.ltas_max_db > 0.025
                && ps.ltas_max_db <= 0.080
            {
                0.42
            } else if tier == Tier::Ultra && ps.raw_besr_db <= -35.0 {
                0.85
            } else if tier == Tier::Pro
                && ps.raw_besr_db <= -34.0
                && ps.ltas_max_db > 0.022
                && ps.ltas_max_db <= 0.080
            {
                0.38
            } else {
                0.90
            };
            ltas_goal = ltas_goal.min((ps.ltas_max_db * ltas_mult).max(0.010));
            has_prev_ltas = true;
        }
    }
    if !has_prev_ltas {
        ltas_goal = ltas_goal.max(0.020);
    }
    let lim_b = b_db.map(|db| 10.0_f64.powf(db / 10.0));
    let lim_comp = 10.0_f64.powf(c_db / 10.0);

    let bez0 =
        eval_bezout_loss_from_gram(p_dim, &grams.gram_tot, &grams.rhs_tot, grams.c_tot, theta0)
            .max(1.0e-12);
    let bez_ref = bez0.max(1.0e-4);
    let bez_w = if cur_besr0 > 0.0 && cur_besr0 < 0.5 * lim_comp {
        0.12
    } else {
        0.45
    };
    let ltas_denom = (thr.max_ltas_db * 0.5).max(0.045);
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
        w0,
        n0,
        welch_sum0,
        welch_sw0,
        lim_b,
        lim_comp,
        ltas_goal,
        bez_ref,
        bez_w,
        bar_w_b: 4.0 * bez_ref,
        bar_w_c: 6.0 * bez_ref,
        welch_scale: 1.8 * bez_ref / (ltas_denom * ltas_denom),
        reg_w,
        total_h,
        dir_w_budget_sq,
    };
    let (theta1, z1, pen1) = run_subspace_stage::<B>(&cfg0, &mut sc, theta0, &z0, 10);
    if pen1 <= 1.0e-9 * bez_ref {
        return theta1;
    }

    let (bar_w_b, bar_w_c) = if prev_sum.is_some() {
        (45.0 * bez_ref, 45.0 * bez_ref)
    } else {
        (14.0 * bez_ref, 18.0 * bez_ref)
    };
    let cfg1 = SubspaceObjectiveCfg {
        bar_w_b,
        bar_w_c,
        welch_scale: 8.0 * bez_ref / (ltas_denom * ltas_denom),
        ..cfg0
    };
    let (theta2, z2, pen2) = run_subspace_stage::<B>(&cfg1, &mut sc, &theta1, &z1, 8);
    if matches!(tier, Tier::Pro | Tier::Ultra) && pen2 > 1.0e-9 * bez_ref {
        let _ = eval_welch_ltas_term::<B>(&cfg1, &mut sc, &theta2, false, false, &mut []);
        if sc.wel_sc.last_max_abs_db > thr.max_ltas_db * 0.82 {
            let cfg2_goal = if tier == Tier::Pro
                && prev_sum.is_none_or(|ps| ps.ltas_max_db > thr.max_ltas_db)
            {
                ltas_goal.max(thr.max_ltas_db * 0.72)
            } else {
                ltas_goal
            };
            let cfg2 = SubspaceObjectiveCfg {
                ltas_goal: cfg2_goal,
                bez_w: 0.10,
                bar_w_b: 1.6 * bar_w_b,
                welch_scale: 18.0 * bez_ref / (ltas_denom * ltas_denom),
                ..cfg1
            };
            return run_subspace_stage::<B>(&cfg2, &mut sc, &theta2, &z2, 8).0;
        }
    }
    theta2
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
