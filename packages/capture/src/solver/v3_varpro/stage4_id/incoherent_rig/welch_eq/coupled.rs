#![forbid(unsafe_code)]
//! Coupled stereo MIMO `Stage III + Stage II-Group0` L-BFGS solver (`welch_eq/coupled.rs`).

use dnam_engine::stages::stage4_tank::v3_fdn::{V3Stage4TankHot, V3Stage4TankParams};

use crate::solver::backends::TrainerBackend;

use super::super::super::super::ltas_adj::{
    Welch33Scratch, Welch33TargetSummary, eval_welch33_lse_minimax_with_nyq_and_adjoint,
};
use super::super::mr_iir_adj::{MrIirFloorSpec, MrIirScratch, eval_mr_iir_and_adjoint};
use super::super::s2_modal::CoupledS2Cols;
use super::super::wave_hinge::{WaveFloorSpec, WaveHingeScratch, eval_wave_hinge_and_adjoint};
use super::super::whitened_lbfgs::{WhitenedLbfgs, dot, solve_rt_g};
use super::basis::{KautzPolePrep, PrecomputedStage3Basis};
use super::{PreStage3Signals, accum_s2_l2_barrier};

pub(super) struct CoupledPolishCtx<'a> {
    pub(super) sig: &'a PreStage3Signals,
    pub(super) ltv_true_lr: Option<(&'a [f32], &'a [f32])>,
    pub(super) params: &'a V3Stage4TankParams,
    pub(super) poles: &'a [KautzPolePrep],
    pub(super) fir_len: usize,
    pub(super) k_cnt: usize,
    pub(super) extra: Option<&'a CoupledS2Cols>,
    pub(super) sum_l: &'a Welch33TargetSummary,
    pub(super) sum_r: &'a Welch33TargetSummary,
    pub(super) wave_l: Option<&'a WaveFloorSpec<'a>>,
    pub(super) wave_r: Option<&'a WaveFloorSpec<'a>>,
    pub(super) mr_l: Option<&'a MrIirFloorSpec<'a>>,
    pub(super) mr_r: Option<&'a MrIirFloorSpec<'a>>,
    pub(super) target_rms_sq: Option<f64>,
    pub(super) s2_soft_l2_sq_l: f64,
    pub(super) s2_soft_l2_sq_r: f64,
    pub(super) goal_db: f64,
    pub(super) reg_scale: f64,
}

struct CoupledChanScratch {
    y: Vec<f64>,
    tmp: Vec<f64>,
    dy: Vec<f64>,
    sc: Welch33Scratch,
    w_sc: WaveHingeScratch,
    mr_sc: MrIirScratch,
}

struct CoupledChanArgs<'a> {
    b_first: &'a PrecomputedStage3Basis<'a>,
    th_first: &'a [f64],
    b_second: &'a PrecomputedStage3Basis<'a>,
    th_second: &'a [f64],
    y0: &'a [f64],
    sum: &'a Welch33TargetSummary,
    wave: Option<&'a WaveFloorSpec<'a>>,
    mr: Option<&'a MrIirFloorSpec<'a>>,
    goal_db: f64,
    warmup: usize,
    n: usize,
    count: f64,
    want_ms: bool,
    need_grad: bool,
}

impl CoupledChanScratch {
    fn new(n: usize) -> Self {
        Self {
            y: vec![0.0_f64; n],
            tmp: vec![0.0_f64; n],
            dy: vec![0.0_f64; n],
            sc: Welch33Scratch::default(),
            w_sc: WaveHingeScratch::default(),
            mr_sc: MrIirScratch::default(),
        }
    }

    fn eval_channel(&mut self, args: &CoupledChanArgs<'_>) -> (f64, f64) {
        args.b_first
            .apply(args.th_first, args.y0, true, &mut self.y);
        args.b_second
            .apply(args.th_second, args.y0, false, &mut self.tmp);
        for (d, &s) in self.y.iter_mut().zip(&self.tmp) {
            *d += s;
        }
        if args.need_grad {
            self.dy.fill(0.0);
        }
        let l_welch = eval_welch33_lse_minimax_with_nyq_and_adjoint(
            args.sum,
            &self.y,
            args.need_grad.then_some(self.dy.as_mut_slice()),
            16.0,
            28.0,
            args.goal_db,
            &mut self.sc,
        );
        let l_wave = args.wave.map_or(0.0, |w| {
            eval_wave_hinge_and_adjoint(
                w,
                &self.y,
                args.warmup,
                args.need_grad.then_some(self.dy.as_mut_slice()),
                &mut self.w_sc,
            )
        });
        let l_mr = args.mr.map_or(0.0, |m| {
            eval_mr_iir_and_adjoint(
                m,
                &self.y,
                args.need_grad.then_some(self.dy.as_mut_slice()),
                &mut self.mr_sc,
            )
        });
        let ms = if args.want_ms {
            self.y[args.warmup..args.n]
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                / args.count
        } else {
            0.0
        };
        (l_welch + l_wave + l_mr, ms)
    }
}

struct CoupledEvalState<'a> {
    ctx: &'a CoupledPolishCtx<'a>,
    b_ll: &'a PrecomputedStage3Basis<'a>,
    b_lr: &'a PrecomputedStage3Basis<'a>,
    b_rl: &'a PrecomputedStage3Basis<'a>,
    b_rr: &'a PrecomputedStage3Basis<'a>,
    y0_l: Vec<f64>,
    y0_r: Vec<f64>,
    r_mat: Vec<f64>,
    z0: Vec<f64>,
    eff_reg: f64,
    s3_dim: usize,
    p: usize,
    n: usize,
    warmup: usize,
    count: f64,
    chan_l: CoupledChanScratch,
    chan_r: CoupledChanScratch,
    tmp: Vec<f64>,
    g_th: Vec<f64>,
}

impl CoupledEvalState<'_> {
    fn eval_step(&mut self, theta: &[f64], z: &[f64], grad_z: Option<&mut [f64]>) -> f64 {
        let need_grad = grad_z.is_some();
        let want_ms = self.ctx.target_rms_sq.is_some();
        let (th_l, th_r) = theta.split_at(self.p);

        #[cfg(not(target_arch = "wasm32"))]
        let ((loss_l, ms_l), (loss_r, ms_r)) = if self.n >= 8_192 {
            let chan_l = &mut self.chan_l;
            let chan_r = &mut self.chan_r;
            let ctx = self.ctx;
            let (b_ll, b_lr, b_rl, b_rr) = (self.b_ll, self.b_lr, self.b_rl, self.b_rr);
            let (y0_l, y0_r) = (&self.y0_l, &self.y0_r);
            let (warmup, n, count) = (self.warmup, self.n, self.count);
            std::thread::scope(|s| {
                let h_l = s.spawn(move || {
                    chan_l.eval_channel(&CoupledChanArgs {
                        b_first: b_ll,
                        th_first: th_l,
                        b_second: b_lr,
                        th_second: th_r,
                        y0: y0_l,
                        sum: ctx.sum_l,
                        wave: ctx.wave_l,
                        mr: ctx.mr_l,
                        goal_db: ctx.goal_db,
                        warmup,
                        n,
                        count,
                        want_ms,
                        need_grad,
                    })
                });
                let res_r = chan_r.eval_channel(&CoupledChanArgs {
                    b_first: b_rl,
                    th_first: th_l,
                    b_second: b_rr,
                    th_second: th_r,
                    y0: y0_r,
                    sum: ctx.sum_r,
                    wave: ctx.wave_r,
                    mr: ctx.mr_r,
                    goal_db: ctx.goal_db,
                    warmup,
                    n,
                    count,
                    want_ms,
                    need_grad,
                });
                let res_l = h_l.join().unwrap_or((0.0, 0.0));
                (res_l, res_r)
            })
        } else {
            (
                self.chan_l.eval_channel(&CoupledChanArgs {
                    b_first: self.b_ll,
                    th_first: th_l,
                    b_second: self.b_lr,
                    th_second: th_r,
                    y0: &self.y0_l,
                    sum: self.ctx.sum_l,
                    wave: self.ctx.wave_l,
                    mr: self.ctx.mr_l,
                    goal_db: self.ctx.goal_db,
                    warmup: self.warmup,
                    n: self.n,
                    count: self.count,
                    want_ms,
                    need_grad,
                }),
                self.chan_r.eval_channel(&CoupledChanArgs {
                    b_first: self.b_rl,
                    th_first: th_l,
                    b_second: self.b_rr,
                    th_second: th_r,
                    y0: &self.y0_r,
                    sum: self.ctx.sum_r,
                    wave: self.ctx.wave_r,
                    mr: self.ctx.mr_r,
                    goal_db: self.ctx.goal_db,
                    warmup: self.warmup,
                    n: self.n,
                    count: self.count,
                    want_ms,
                    need_grad,
                }),
            )
        };
        #[cfg(target_arch = "wasm32")]
        let ((loss_l, ms_l), (loss_r, ms_r)) = (
            self.chan_l.eval_channel(&CoupledChanArgs {
                b_first: self.b_ll,
                th_first: th_l,
                b_second: self.b_lr,
                th_second: th_r,
                y0: &self.y0_l,
                sum: self.ctx.sum_l,
                wave: self.ctx.wave_l,
                mr: self.ctx.mr_l,
                goal_db: self.ctx.goal_db,
                warmup: self.warmup,
                n: self.n,
                count: self.count,
                want_ms,
                need_grad,
            }),
            self.chan_r.eval_channel(&CoupledChanArgs {
                b_first: self.b_rl,
                th_first: th_l,
                b_second: self.b_rr,
                th_second: th_r,
                y0: &self.y0_r,
                sum: self.ctx.sum_r,
                wave: self.ctx.wave_r,
                mr: self.ctx.mr_r,
                goal_db: self.ctx.goal_db,
                warmup: self.warmup,
                n: self.n,
                count: self.count,
                want_ms,
                need_grad,
            }),
        );

        let mut l_rms = 0.0_f64;
        if let Some(t_sq) = self.ctx.target_rms_sq {
            let inv_t = 1.0 / t_sq.max(1.0e-12);
            let rms_sq = 0.5 * (ms_l + ms_r);
            let ratio = (rms_sq - t_sq) * inv_t;
            if need_grad {
                let gf = 30.0 * ratio * inv_t / self.count;
                for i in self.warmup..self.n {
                    self.chan_l.dy[i] += gf * self.chan_l.y[i];
                    self.chan_r.dy[i] += gf * self.chan_r.y[i];
                }
            }
            l_rms = 15.0 * ratio * ratio;
        }
        let reg_scale = self.eff_reg;
        let reg_grad = 2.0 * reg_scale;
        let mut reg = 0.0_f64;
        for j in 0..(2 * self.p) {
            let dz = z[j] - self.z0[j];
            reg += reg_scale * dz * dz;
        }
        let l_s2 = if let Some(gz) = grad_z {
            self.g_th.fill(0.0);
            let (gl, gr) = self.g_th.split_at_mut(self.p);
            #[cfg(not(target_arch = "wasm32"))]
            if self.n >= 8_192 {
                let (dyl, dyr) = (&self.chan_l.dy, &self.chan_r.dy);
                let (b_ll, b_lr, b_rl, b_rr) = (self.b_ll, self.b_lr, self.b_rl, self.b_rr);
                let gl_ref: &mut [f64] = &mut *gl;
                let gr_ref: &mut [f64] = &mut *gr;
                std::thread::scope(|s| {
                    let h = s.spawn(move || {
                        b_ll.accum_adjoint(dyl, gl_ref);
                        b_rl.accum_adjoint(dyr, gl_ref);
                    });
                    b_lr.accum_adjoint(dyl, gr_ref);
                    b_rr.accum_adjoint(dyr, gr_ref);
                    let _ = h.join();
                });
            } else {
                self.b_ll.accum_adjoint(&self.chan_l.dy, gl);
                self.b_rl.accum_adjoint(&self.chan_r.dy, gl);
                self.b_lr.accum_adjoint(&self.chan_l.dy, gr);
                self.b_rr.accum_adjoint(&self.chan_r.dy, gr);
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.b_ll.accum_adjoint(&self.chan_l.dy, gl);
                self.b_rl.accum_adjoint(&self.chan_r.dy, gl);
                self.b_lr.accum_adjoint(&self.chan_l.dy, gr);
                self.b_rr.accum_adjoint(&self.chan_r.dy, gr);
            }
            let s2 = if self.p > self.s3_dim {
                accum_s2_l2_barrier(
                    &th_l[self.s3_dim..],
                    self.ctx.s2_soft_l2_sq_l,
                    Some(&mut gl[self.s3_dim..]),
                ) + accum_s2_l2_barrier(
                    &th_r[self.s3_dim..],
                    self.ctx.s2_soft_l2_sq_r,
                    Some(&mut gr[self.s3_dim..]),
                )
            } else {
                0.0
            };
            solve_rt_g(2 * self.p, &self.r_mat, &self.g_th, gz);
            for (g_j, (&zj, &z0j)) in gz.iter_mut().zip(z.iter().zip(&self.z0)) {
                *g_j += reg_grad * (zj - z0j);
            }
            s2
        } else if self.p > self.s3_dim {
            accum_s2_l2_barrier(&th_l[self.s3_dim..], self.ctx.s2_soft_l2_sq_l, None)
                + accum_s2_l2_barrier(&th_r[self.s3_dim..], self.ctx.s2_soft_l2_sq_r, None)
        } else {
            0.0
        };
        loss_l + loss_r + l_rms + l_s2 + reg
    }
}

fn compute_coupled_y0_defect(
    sig: &PreStage3Signals,
    b_ll: &PrecomputedStage3Basis<'_>,
    b_lr: &PrecomputedStage3Basis<'_>,
    b_rl: &PrecomputedStage3Basis<'_>,
    b_rr: &PrecomputedStage3Basis<'_>,
    theta_lr: &[f64],
    true_l: &[f32],
    true_r: &[f32],
    yc_l: &mut [f64],
    yc_r: &mut [f64],
    tmp: &mut [f64],
) {
    let n = sig.y0_l.len();
    let (th_l, th_r) = theta_lr.split_at(b_ll.dim());
    b_ll.apply(th_l, &sig.y0_l, true, yc_l);
    b_lr.apply(th_r, &sig.y0_l, false, tmp);
    for i in 0..n {
        yc_l[i] = sig.y0_l[i] + (f64::from(true_l[i]) - (yc_l[i] + tmp[i]));
    }
    b_rl.apply(th_l, &sig.y0_r, true, yc_r);
    b_rr.apply(th_r, &sig.y0_r, false, tmp);
    for i in 0..n {
        yc_r[i] = sig.y0_r[i] + (f64::from(true_r[i]) - (yc_r[i] + tmp[i]));
    }
}

pub(super) fn polish_coupled_stereo<B: TrainerBackend, F>(
    ctx: &CoupledPolishCtx<'_>,
    theta0_lr: &[f64],
    passes: &[usize],
    mut render_mid: F,
) -> Vec<f64>
where
    F: FnMut(&mut [f64]) -> (Vec<f32>, Vec<f32>),
{
    let (sig, params, poles, fir_len, k_cnt) =
        (ctx.sig, ctx.params, ctx.poles, ctx.fir_len, ctx.k_cnt);
    let n = sig.u_l.len();
    let s3_dim = fir_len + 2 * k_cnt;
    let ex_ll: &[Vec<f32>] = ctx.extra.map_or(&[], |c| &c.ll);
    let ex_lr: &[Vec<f32>] = ctx.extra.map_or(&[], |c| &c.lr);
    let ex_rl: &[Vec<f32>] = ctx.extra.map_or(&[], |c| &c.rl);
    let ex_rr: &[Vec<f32>] = ctx.extra.map_or(&[], |c| &c.rr);
    let p = s3_dim + ex_ll.len();
    let dim = 2 * p;
    let warmup = ctx.sum_l.warmup;
    let count = n.saturating_sub(warmup).max(1) as f64;

    let (mut v_ll, mut v_rl, mut v_lr, mut v_rr) = (
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
    );
    let mut tank_l = V3Stage4TankHot::from_cold(params);
    let mut tank_r = V3Stage4TankHot::from_cold(params);
    for i in 0..n {
        let [a, b] = tank_l.step_stereo::<B>(sig.u_l[i], 0.0);
        let [c, d] = tank_r.step_stereo::<B>(0.0, sig.u_r[i]);
        (v_ll[i], v_rl[i], v_lr[i], v_rr[i]) = (a, b, c, d);
    }

    let b_ll = PrecomputedStage3Basis::new(&v_ll, poles, fir_len, ex_ll);
    let b_lr = PrecomputedStage3Basis::new(&v_lr, poles, fir_len, ex_lr);
    let b_rl = PrecomputedStage3Basis::new(&v_rl, poles, fir_len, ex_rl);
    let b_rr = PrecomputedStage3Basis::new(&v_rr, poles, fir_len, ex_rr);

    let gram = PrecomputedStage3Basis::build_coupled_gram::<B>(&b_ll, &b_lr, &b_rl, &b_rr, warmup);
    let Some((mut opt, mut init_z)) = WhitenedLbfgs::new(dim, &gram, theta0_lr) else {
        return theta0_lr.to_vec();
    };
    let mut y0_l_adj = sig.y0_l.clone();
    let mut y0_r_adj = sig.y0_r.clone();
    let mut tmp = vec![0.0_f64; n];
    if let Some((true_l, true_r)) = ctx.ltv_true_lr {
        compute_coupled_y0_defect(
            sig,
            &b_ll,
            &b_lr,
            &b_rl,
            &b_rr,
            theta0_lr,
            true_l,
            true_r,
            &mut y0_l_adj,
            &mut y0_r_adj,
            &mut tmp,
        );
    }

    let mut cur_th = theta0_lr.to_vec();
    let eff_reg = ctx.reg_scale / dot(&opt.z0, &opt.z0).max(1.0e-6);
    let mut state = CoupledEvalState {
        ctx,
        b_ll: &b_ll,
        b_lr: &b_lr,
        b_rl: &b_rl,
        b_rr: &b_rr,
        y0_l: y0_l_adj,
        y0_r: y0_r_adj,
        r_mat: opt.r_mat.clone(),
        z0: opt.z0.clone(),
        eff_reg,
        s3_dim,
        p,
        n,
        warmup,
        count,
        chan_l: CoupledChanScratch::new(n),
        chan_r: CoupledChanScratch::new(n),
        tmp,
        g_th: vec![0.0_f64; dim],
    };

    for (pass_idx, &it) in passes.iter().enumerate() {
        let z_start = std::mem::take(&mut init_z);
        cur_th = opt.minimize(z_start, it, |theta, z, grad_z| {
            state.eval_step(theta, z, grad_z)
        });
        if pass_idx + 1 < passes.len() {
            let (true_l, true_r) = render_mid(&mut cur_th);
            compute_coupled_y0_defect(
                sig,
                &b_ll,
                &b_lr,
                &b_rl,
                &b_rr,
                &cur_th,
                &true_l,
                &true_r,
                &mut state.y0_l,
                &mut state.y0_r,
                &mut state.tmp,
            );
            init_z = opt.warm_reset(&cur_th);
            state.z0.clone_from(&opt.z0);
            state.eff_reg = ctx.reg_scale / dot(&state.z0, &state.z0).max(1.0e-6);
        }
    }
    cur_th
}
