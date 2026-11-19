#![forbid(unsafe_code)]
//! Post-Stage-IV MIMO LTI Welch-33 minimax Stage III (`FIR` + `Kautz`) equalizer
//! in Cholesky-whitened coordinates (`incoherent_rig/welch_eq.rs`).

mod basis;
mod coupled;

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, tier_kautz_banks};
use dnam_engine::stages::stage4_tank::v3_fdn::{V3Stage4TankHot, V3Stage4TankParams};
use dnam_engine::stages::{V3MatryoshkaModel, V3TierReadout};

use crate::solver::backends::TrainerBackend;

use super::super::super::ltas_adj::{
    Welch33Scratch, Welch33TargetSummary, eval_welch33_lse_minimax_with_nyq_and_adjoint,
};
use super::super::super::model_f64::V3TrainStem;
use super::super::render_tier_stereo;
use super::mr_iir_adj::{MrIirFloorSpec, MrIirScratch, eval_mr_iir_and_adjoint};
use super::s2_modal::{
    S2_MODAL_COLS, extract_coupled_s2_cols, extract_decoupled_s2_cols, has_1x_modal_basis,
    max_direct_s2_l2_sq, read_direct_s2_weights, write_direct_s2_weights,
};
use super::wave_hinge::{WaveFloorSpec, WaveHingeScratch, eval_wave_hinge_and_adjoint};
use super::whitened_lbfgs::{WhitenedLbfgs, dot, solve_rt_g};
use basis::{PrecomputedStage3Basis, prepare_poles};
use coupled::{CoupledPolishCtx, polish_coupled_stereo};

pub(super) struct PreStage3Signals {
    pub(super) u_l: Vec<f32>,
    pub(super) u_r: Vec<f32>,
    pub(super) y0_l: Vec<f64>,
    pub(super) y0_r: Vec<f64>,
}

pub(super) fn extract_pre_stage3_signals<B: TrainerBackend>(
    model: &V3MatryoshkaModel,
    tier: Tier,
    stem: &V3TrainStem,
) -> PreStage3Signals {
    let n = stem.usable_len();
    let t = tier as usize;
    let mut m_zero = model.clone();
    m_zero.readouts[t].fir_l = [0.0; 64];
    m_zero.readouts[t].fir_r = [0.0; 64];
    m_zero.readouts[t].kautz_c_l = [(0.0, 0.0); MAX_KAUTZ_POLES];
    m_zero.readouts[t].kautz_c_r = [(0.0, 0.0); MAX_KAUTZ_POLES];
    if has_1x_modal_basis(model, tier) {
        m_zero.readouts[t].direct_s2_w = [[0.0; MAX_TOTAL_GRU_LANES]; 2];
    }
    let clean_sub_zero = model.stage0.clean_sub_mix == 0.0 && !model.stage0.op_p_route_clean;
    let ro_z = &m_zero.readouts[t];
    let direct_s2_zero = ro_z.direct_s2_u_gain == [0.0, 0.0]
        && ro_z.direct_s2_hp_gain == [0.0, 0.0]
        && ro_z.svf_direct_taps_l == [0.0; 8]
        && ro_z.svf_direct_taps_r == [0.0; 8]
        && ro_z.direct_s2_w.iter().flatten().all(|&w| w == 0.0);
    if clean_sub_zero && direct_s2_zero {
        m_zero.stage4[t] = V3Stage4TankParams {
            wet_mix: 0.0,
            dry_mix: 1.0,
            g_loop: 0.0,
            base_psi: 0.0,
            vca_kappa: 0.0,
            vca_min_gain: 1.0,
            ..V3Stage4TankParams::identity()
        };
        m_zero.readouts[t].fir_l[0] = 1.0;
        m_zero.readouts[t].fir_r[0] = 1.0;
        let (d1_l, d1_r) =
            render_tier_stereo::<B>(&m_zero.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);
        return PreStage3Signals {
            u_l: d1_l,
            u_r: d1_r,
            y0_l: vec![0.0; n],
            y0_r: vec![0.0; n],
        };
    }
    let (y0_l32, y0_r32) =
        render_tier_stereo::<B>(&m_zero.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);

    m_zero.stage4[t] = V3Stage4TankParams {
        wet_mix: 0.0,
        dry_mix: 1.0,
        g_loop: 0.0,
        base_psi: 0.0,
        vca_kappa: 0.0,
        vca_min_gain: 1.0,
        ..V3Stage4TankParams::identity()
    };
    let (d0_l, d0_r) =
        render_tier_stereo::<B>(&m_zero.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);
    m_zero.readouts[t].fir_l[0] = 1.0;
    m_zero.readouts[t].fir_r[0] = 1.0;
    let (d1_l, d1_r) =
        render_tier_stereo::<B>(&m_zero.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);

    let u_l: Vec<f32> = d1_l.iter().zip(&d0_l).map(|(&a, &b)| a - b).collect();
    let u_r: Vec<f32> = d1_r.iter().zip(&d0_r).map(|(&a, &b)| a - b).collect();
    let y0_l: Vec<f64> = y0_l32.into_iter().map(f64::from).collect();
    let y0_r: Vec<f64> = y0_r32.into_iter().map(f64::from).collect();
    PreStage3Signals {
        u_l,
        u_r,
        y0_l,
        y0_r,
    }
}

fn read_theta(
    ro: &V3TierReadout,
    fir_len: usize,
    k_cnt: usize,
    use_s2: bool,
    ch: usize,
) -> Vec<f64> {
    let s2_cnt = if use_s2 { S2_MODAL_COLS } else { 0 };
    let p_dim = fir_len + 2 * k_cnt + s2_cnt;
    let mut out = vec![0.0_f64; p_dim];
    let fir = if ch == 0 { &ro.fir_l } else { &ro.fir_r };
    let kc = if ch == 0 {
        &ro.kautz_c_l
    } else {
        &ro.kautz_c_r
    };
    for d in 0..fir_len {
        out[d] = f64::from(fir[d]);
    }
    for m in 0..k_cnt {
        out[fir_len + 2 * m] = f64::from(kc[m].0);
        out[fir_len + 2 * m + 1] = f64::from(kc[m].1);
    }
    if use_s2 {
        read_direct_s2_weights(ro, ch, &mut out[fir_len + 2 * k_cnt..]);
    }
    out
}

fn write_theta(
    theta: &[f64],
    fir_len: usize,
    k_cnt: usize,
    use_s2: bool,
    ch: usize,
    ro: &mut V3TierReadout,
) {
    let fir = if ch == 0 {
        &mut ro.fir_l
    } else {
        &mut ro.fir_r
    };
    let kc = if ch == 0 {
        &mut ro.kautz_c_l
    } else {
        &mut ro.kautz_c_r
    };
    for d in 0..fir_len {
        fir[d] = (theta[d] as f32).clamp(-64.0, 64.0);
    }
    for m in 0..k_cnt {
        kc[m] = (
            (theta[fir_len + 2 * m] as f32).clamp(-16.0, 16.0),
            (theta[fir_len + 2 * m + 1] as f32).clamp(-16.0, 16.0),
        );
    }
    if use_s2 {
        write_direct_s2_weights(&theta[fir_len + 2 * k_cnt..], ch, ro);
    }
}

pub(super) fn accum_s2_l2_barrier(
    w_s2: &[f64],
    soft_l2_sq: f64,
    grad_s2: Option<&mut [f64]>,
) -> f64 {
    let sq: f64 = w_s2.iter().map(|&w| w * w).sum();
    if sq <= soft_l2_sq {
        return 0.0;
    }
    let ex = sq - soft_l2_sq;
    if let Some(grad) = grad_s2 {
        for (g, &w) in grad.iter_mut().zip(w_s2) {
            *g += 8.0 * ex * w;
        }
    }
    2.0 * ex * ex
}

struct DecoupledChanCfg<'a> {
    basis: &'a PrecomputedStage3Basis<'a>,
    y0: &'a [f64],
    summary: &'a Welch33TargetSummary,
    wave: Option<&'a WaveFloorSpec<'a>>,
    mr: Option<&'a MrIirFloorSpec<'a>>,
    target_rms_sq: Option<f64>,
    s2_soft_l2_sq: f64,
    goal_db: f64,
    reg_scale: f64,
}

struct DecoupledEvalState<'a> {
    cfg: &'a DecoupledChanCfg<'a>,
    r_mat: &'a [f64],
    z0: &'a [f64],
    eff_reg: f64,
    p_dim: usize,
    s3_dim: usize,
    n: usize,
    warmup: usize,
    count: f64,
    y_hat: Vec<f64>,
    dy: Vec<f64>,
    g_th: Vec<f64>,
    sc: Welch33Scratch,
    w_sc: WaveHingeScratch,
    mr_sc: MrIirScratch,
}

impl DecoupledEvalState<'_> {
    fn eval_step(&mut self, theta: &[f64], z: &[f64], grad_z: Option<&mut [f64]>) -> f64 {
        let need_grad = grad_z.is_some();
        self.cfg
            .basis
            .apply(theta, self.cfg.y0, true, &mut self.y_hat);
        if need_grad {
            self.dy.fill(0.0);
        }
        let l_welch = eval_welch33_lse_minimax_with_nyq_and_adjoint(
            self.cfg.summary,
            &self.y_hat,
            need_grad.then_some(self.dy.as_mut_slice()),
            16.0,
            28.0,
            self.cfg.goal_db,
            &mut self.sc,
        );
        let l_wave = self.cfg.wave.map_or(0.0, |w| {
            eval_wave_hinge_and_adjoint(
                w,
                &self.y_hat,
                self.warmup,
                need_grad.then_some(self.dy.as_mut_slice()),
                &mut self.w_sc,
            )
        });
        let l_mr = self.cfg.mr.map_or(0.0, |m| {
            eval_mr_iir_and_adjoint(
                m,
                &self.y_hat,
                need_grad.then_some(self.dy.as_mut_slice()),
                &mut self.mr_sc,
            )
        });
        let mut l_rms = 0.0_f64;
        if let Some(t_sq) = self.cfg.target_rms_sq {
            let inv_t = 1.0 / t_sq.max(1.0e-12);
            let ms_y: f64 = self.y_hat[self.warmup..self.n]
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                / self.count;
            let ratio = (ms_y - t_sq) * inv_t;
            if need_grad {
                let g_fac = 60.0 * ratio * inv_t / self.count;
                for i in self.warmup..self.n {
                    self.dy[i] += g_fac * self.y_hat[i];
                }
            }
            l_rms = 15.0 * ratio * ratio;
        }
        let reg_scale = self.eff_reg;
        let reg_grad = 2.0 * reg_scale;
        let mut reg = 0.0_f64;
        for j in 0..self.p_dim {
            let dz = z[j] - self.z0[j];
            reg += reg_scale * dz * dz;
        }
        let l_s2 = if let Some(gz) = grad_z {
            self.g_th.fill(0.0);
            self.cfg.basis.accum_adjoint(&self.dy, &mut self.g_th);
            let s2 = if self.p_dim > self.s3_dim {
                accum_s2_l2_barrier(
                    &theta[self.s3_dim..],
                    self.cfg.s2_soft_l2_sq,
                    Some(&mut self.g_th[self.s3_dim..]),
                )
            } else {
                0.0
            };
            solve_rt_g(self.p_dim, self.r_mat, &self.g_th, gz);
            for (g_j, (&zj, &z0j)) in gz.iter_mut().zip(z.iter().zip(self.z0)) {
                *g_j += reg_grad * (zj - z0j);
            }
            s2
        } else if self.p_dim > self.s3_dim {
            accum_s2_l2_barrier(&theta[self.s3_dim..], self.cfg.s2_soft_l2_sq, None)
        } else {
            0.0
        };
        l_welch + l_wave + l_mr + l_rms + l_s2 + reg
    }
}

struct DecoupledChanBuild<'a> {
    cfg: DecoupledChanCfg<'a>,
    basis: &'a PrecomputedStage3Basis<'a>,
    opt: WhitenedLbfgs,
    init_z: Vec<f64>,
    y0_adj: Vec<f64>,
}

fn setup_decoupled_chan<'a, B: TrainerBackend>(
    basis: &'a PrecomputedStage3Basis<'a>,
    y0_base: &'a [f64],
    true_init: &[f32],
    summary: &'a Welch33TargetSummary,
    wave: Option<&'a WaveFloorSpec<'a>>,
    mr: Option<&'a MrIirFloorSpec<'a>>,
    target_rms_sq: Option<f64>,
    s2_soft_l2_sq: f64,
    goal_db: f64,
    reg_scale: f64,
    theta0: &[f64],
) -> Option<DecoupledChanBuild<'a>> {
    let p_dim = theta0.len();
    let gram = basis.build_gram::<B>(summary.warmup);
    let (opt, init_z) = WhitenedLbfgs::new(p_dim, &gram, theta0)?;
    let mut y0_adj = y0_base.to_vec();
    apply_decoupled_ltv_defect(basis, y0_base, theta0, true_init, &mut y0_adj);
    let cfg = DecoupledChanCfg {
        basis,
        y0: y0_base,
        summary,
        wave,
        mr,
        target_rms_sq,
        s2_soft_l2_sq,
        goal_db,
        reg_scale,
    };
    Some(DecoupledChanBuild {
        cfg,
        basis,
        opt,
        init_z,
        y0_adj,
    })
}

fn run_decoupled_chan_pass(ch_state: &mut DecoupledChanBuild<'_>, iters: usize) -> Vec<f64> {
    let p_dim = ch_state.init_z.len();
    let s3_dim = ch_state.basis.s3_dim();
    let n = ch_state.basis.v.len();
    let warmup = ch_state.cfg.summary.warmup;
    let count = n.saturating_sub(warmup).max(1) as f64;
    let z0 = ch_state.opt.z0.clone();
    let r_mat = ch_state.opt.r_mat.clone();
    let eff_reg = ch_state.cfg.reg_scale / dot(&z0, &z0).max(1.0e-6);
    let cfg = DecoupledChanCfg {
        basis: ch_state.cfg.basis,
        y0: &ch_state.y0_adj,
        summary: ch_state.cfg.summary,
        wave: ch_state.cfg.wave,
        mr: ch_state.cfg.mr,
        target_rms_sq: ch_state.cfg.target_rms_sq,
        s2_soft_l2_sq: ch_state.cfg.s2_soft_l2_sq,
        goal_db: ch_state.cfg.goal_db,
        reg_scale: ch_state.cfg.reg_scale,
    };
    let z_start = std::mem::take(&mut ch_state.init_z);
    let mut eval_state = DecoupledEvalState {
        cfg: &cfg,
        r_mat: &r_mat,
        z0: &z0,
        eff_reg,
        p_dim,
        s3_dim,
        n,
        warmup,
        count,
        y_hat: vec![0.0_f64; n],
        dy: vec![0.0_f64; n],
        g_th: vec![0.0_f64; p_dim],
        sc: Welch33Scratch::default(),
        w_sc: WaveHingeScratch::default(),
        mr_sc: MrIirScratch::default(),
    };
    ch_state.opt.minimize(z_start, iters, |theta, z, grad_z| {
        eval_state.eval_step(theta, z, grad_z)
    })
}

pub(super) fn scale_summary_band_tilt(
    base: &Welch33TargetSummary,
    tilt_db: &[f64; 33],
) -> Welch33TargetSummary {
    let mut out = base.clone();
    for b in 0..33 {
        out.tgt_band_p[b] *= 10.0_f64.powf(-tilt_db[b] / 10.0);
    }
    if let Some(b_hi) = (0..33).rev().find(|&b| out.active[b]) {
        out.tgt_nyq_p *= 10.0_f64.powf(-tilt_db[b_hi] / 10.0);
    }
    out
}

pub(super) struct WelchPolishSpec<'a> {
    pub(super) tier: Tier,
    pub(super) stem: &'a V3TrainStem,
    pub(super) sum_l: &'a Welch33TargetSummary,
    pub(super) sum_r: Option<&'a Welch33TargetSummary>,
    pub(super) wave_l: Option<&'a WaveFloorSpec<'a>>,
    pub(super) wave_r: Option<&'a WaveFloorSpec<'a>>,
    pub(super) mr_l: Option<&'a MrIirFloorSpec<'a>>,
    pub(super) mr_r: Option<&'a MrIirFloorSpec<'a>>,
    pub(super) lock_rms_lr: Option<(f64, f64, f64)>,
    pub(super) goal_db: f64,
    pub(super) iters: usize,
}

fn eval_rendered_polish_loss(
    ol: &[f32],
    or: &[f32],
    spec: &WelchPolishSpec<'_>,
    coupled: bool,
    mono: bool,
) -> f64 {
    let n = ol.len().min(or.len());
    let warmup = spec.sum_l.warmup;
    let mut sc = Welch33Scratch::default();
    let mut w_sc = WaveHingeScratch::default();
    let mut mr_sc = MrIirScratch::default();
    let y_l: Vec<f64> = ol.iter().map(|&v| f64::from(v)).collect();
    let mut loss = eval_welch33_lse_minimax_with_nyq_and_adjoint(
        spec.sum_l,
        &y_l,
        None,
        16.0,
        28.0,
        spec.goal_db,
        &mut sc,
    );
    if let Some(wl) = spec.wave_l {
        loss += eval_wave_hinge_and_adjoint(wl, &y_l, warmup, None, &mut w_sc);
    }
    if let Some(ml) = spec.mr_l {
        loss += eval_mr_iir_and_adjoint(ml, &y_l, None, &mut mr_sc);
    }
    let y_r_opt = (!mono).then(|| or.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>());
    if let Some(yr) = y_r_opt.as_ref() {
        if let Some(sr) = spec.sum_r {
            loss += eval_welch33_lse_minimax_with_nyq_and_adjoint(
                sr,
                yr,
                None,
                16.0,
                28.0,
                spec.goal_db,
                &mut sc,
            );
        }
        if let Some(wr) = spec.wave_r {
            loss += eval_wave_hinge_and_adjoint(wr, yr, warmup, None, &mut w_sc);
        }
        if let Some(mr) = spec.mr_r {
            loss += eval_mr_iir_and_adjoint(mr, yr, None, &mut mr_sc);
        }
    }
    if let Some((r_tot, l_sq, r_sq)) = spec.lock_rms_lr {
        let count = n.saturating_sub(warmup).max(1) as f64;
        let ms_l = if n > warmup {
            y_l[warmup..n].iter().map(|&x| x * x).sum::<f64>() / count
        } else {
            0.0
        };
        let ms_r = match (y_r_opt.as_ref(), n > warmup) {
            (Some(yr), true) => yr[warmup..n].iter().map(|&x| x * x).sum::<f64>() / count,
            _ => ms_l,
        };
        if mono || coupled {
            let rms_sq = if mono { ms_l } else { 0.5 * (ms_l + ms_r) };
            let t_sq = (r_tot * r_tot).max(1.0e-12);
            let ratio = (rms_sq - t_sq) / t_sq;
            loss += 15.0 * ratio * ratio;
        } else {
            let rl = (ms_l - l_sq) / l_sq.max(1.0e-12);
            loss += 15.0 * rl * rl;
            if spec.sum_r.is_some() {
                let rr = (ms_r - r_sq) / r_sq.max(1.0e-12);
                loss += 15.0 * rr * rr;
            }
        }
    }
    loss
}

fn apply_decoupled_ltv_defect(
    basis: &PrecomputedStage3Basis<'_>,
    y0_base: &[f64],
    th0: &[f64],
    y_true: &[f32],
    y0_out: &mut [f64],
) {
    let n = basis.v.len();
    let mut y_comm = vec![0.0_f64; n];
    basis.apply(th0, y0_base, true, &mut y_comm);
    for i in 0..n {
        y0_out[i] = y0_base[i] + (f64::from(y_true[i]) - y_comm[i]);
    }
}

fn blend_readout_half(a: &V3TierReadout, b: &V3TierReadout) -> V3TierReadout {
    let mut out = b.clone();
    for i in 0..64 {
        out.fir_l[i] = 0.5 * (a.fir_l[i] + b.fir_l[i]);
        out.fir_r[i] = 0.5 * (a.fir_r[i] + b.fir_r[i]);
    }
    for m in 0..MAX_KAUTZ_POLES {
        out.kautz_c_l[m] = (
            0.5 * (a.kautz_c_l[m].0 + b.kautz_c_l[m].0),
            0.5 * (a.kautz_c_l[m].1 + b.kautz_c_l[m].1),
        );
        out.kautz_c_r[m] = (
            0.5 * (a.kautz_c_r[m].0 + b.kautz_c_r[m].0),
            0.5 * (a.kautz_c_r[m].1 + b.kautz_c_r[m].1),
        );
    }
    for ch in 0..2 {
        for l in 0..MAX_TOTAL_GRU_LANES {
            out.direct_s2_w[ch][l] = 0.5 * (a.direct_s2_w[ch][l] + b.direct_s2_w[ch][l]);
        }
    }
    out
}

pub(super) fn polish_tier_welch33<B: TrainerBackend>(
    model: &mut V3MatryoshkaModel,
    spec: &WelchPolishSpec<'_>,
) {
    let lfo_depth = model.stage4[spec.tier as usize].lfo_depth_samples;
    let deep_lfo = lfo_depth >= 14.0;
    let reg_scale = if deep_lfo { 5.0e-5 } else { 1.0e-5 };
    let passes_buf: [usize; 2];
    let active_passes: &[usize] = if lfo_depth >= 1.5 && spec.iters >= 24 {
        let it0 = (spec.iters * 5) / 8;
        passes_buf = [it0, spec.iters - it0];
        &passes_buf
    } else {
        passes_buf = [spec.iters, 0];
        &passes_buf[..1]
    };

    let (tier, stem, sum_l, sum_r) = (spec.tier, spec.stem, spec.sum_l, spec.sum_r);
    let goal_db = spec.goal_db;
    let t = tier as usize;
    let fir_len = (t + 1) * 16;
    let k_cnt = tier_kautz_banks(tier) * 4;
    let use_s2 = has_1x_modal_basis(model, tier);
    let p_dim = fir_len + 2 * k_cnt + if use_s2 { S2_MODAL_COLS } else { 0 };
    let poles = prepare_poles(&model.kautz_poles, k_cnt);
    let sig = extract_pre_stage3_signals::<B>(model, tier, stem);
    let n = sig.u_l.len();
    let mono = model.mono_out;
    let ro_before = model.readouts[t].clone();
    let (before_l, before_r) =
        render_tier_stereo::<B>(&model.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);

    let max_l = max_direct_s2_l2_sq(&model.readouts[t], 0);
    let max_r = max_direct_s2_l2_sq(&model.readouts[t], 1);
    let s2_soft_l = if use_s2 {
        (0.20 * max_l).clamp(1.0, 48.0).min(0.85 * max_l)
    } else {
        0.0
    };
    let s2_soft_r = if use_s2 {
        (0.20 * max_r).clamp(1.0, 48.0).min(0.85 * max_r)
    } else {
        0.0
    };

    let coupled = !mono && (model.stage4[t].g_loop != 0.0 || model.stage4[t].base_psi != 0.0);
    if coupled && let Some(sr) = sum_r {
        let s4_params = model.stage4[t];
        let s2_coupled = use_s2.then(|| extract_coupled_s2_cols::<B>(model, tier, stem));
        let mut th0 = read_theta(&model.readouts[t], fir_len, k_cnt, use_s2, 0);
        th0.extend(read_theta(&model.readouts[t], fir_len, k_cnt, use_s2, 1));
        let sol = polish_coupled_stereo::<B, _>(
            &CoupledPolishCtx {
                sig: &sig,
                ltv_true_lr: Some((before_l.as_slice(), before_r.as_slice())),
                params: &s4_params,
                poles: &poles,
                fir_len,
                k_cnt,
                extra: s2_coupled.as_ref(),
                sum_l,
                sum_r: sr,
                wave_l: spec.wave_l,
                wave_r: spec.wave_r,
                mr_l: spec.mr_l,
                mr_r: spec.mr_r,
                target_rms_sq: spec.lock_rms_lr.map(|(r, _, _)| r * r),
                s2_soft_l2_sq_l: s2_soft_l,
                s2_soft_l2_sq_r: s2_soft_r,
                goal_db,
                reg_scale,
            },
            &th0,
            active_passes,
            |mid_th| {
                write_theta(
                    &mid_th[..p_dim],
                    fir_len,
                    k_cnt,
                    use_s2,
                    0,
                    &mut model.readouts[t],
                );
                write_theta(
                    &mid_th[p_dim..],
                    fir_len,
                    k_cnt,
                    use_s2,
                    1,
                    &mut model.readouts[t],
                );
                model.readouts[t].mask_to_tier(tier);
                mid_th[..p_dim].copy_from_slice(&read_theta(
                    &model.readouts[t],
                    fir_len,
                    k_cnt,
                    use_s2,
                    0,
                ));
                mid_th[p_dim..].copy_from_slice(&read_theta(
                    &model.readouts[t],
                    fir_len,
                    k_cnt,
                    use_s2,
                    1,
                ));
                render_tier_stereo::<B>(&model.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n])
            },
        );
        write_theta(
            &sol[..p_dim],
            fir_len,
            k_cnt,
            use_s2,
            0,
            &mut model.readouts[t],
        );
        write_theta(
            &sol[p_dim..],
            fir_len,
            k_cnt,
            use_s2,
            1,
            &mut model.readouts[t],
        );
    } else {
        let s2_dec = use_s2.then(|| extract_decoupled_s2_cols::<B>(model, tier, stem));
        let mut tank = V3Stage4TankHot::from_cold(&model.stage4[t]);
        let (mut vl, mut vr) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        for i in 0..n {
            let [a, b] =
                tank.step_stereo::<B>(sig.u_l[i], if mono { sig.u_l[i] } else { sig.u_r[i] });
            vl[i] = a;
            vr[i] = b;
        }
        let th0_l = read_theta(&model.readouts[t], fir_len, k_cnt, use_s2, 0);
        let th0_r = (!mono && sum_r.is_some())
            .then(|| read_theta(&model.readouts[t], fir_len, k_cnt, use_s2, 1));
        let ex_l: &[Vec<f32>] = s2_dec.as_ref().map_or(&[][..], |c| &c.l[..]);
        let ex_r: &[Vec<f32>] = s2_dec.as_ref().map_or(&[][..], |c| &c.r[..]);
        let basis_l = PrecomputedStage3Basis::new(&vl, &poles, fir_len, ex_l);
        let mut build_l = setup_decoupled_chan::<B>(
            &basis_l,
            &sig.y0_l,
            &before_l,
            sum_l,
            spec.wave_l,
            spec.mr_l,
            spec.lock_rms_lr.map(|(_, l_sq, _)| l_sq),
            s2_soft_l,
            goal_db,
            reg_scale,
            &th0_l,
        );
        let basis_r_opt = (!mono && sum_r.is_some())
            .then(|| PrecomputedStage3Basis::new(&vr, &poles, fir_len, ex_r));
        let mut build_r = match (basis_r_opt.as_ref(), sum_r, th0_r.as_ref()) {
            (Some(br), Some(sr), Some(tr0)) => setup_decoupled_chan::<B>(
                br,
                &sig.y0_r,
                &before_r,
                sr,
                spec.wave_r,
                spec.mr_r,
                spec.lock_rms_lr.map(|(_, _, r_sq)| r_sq),
                s2_soft_r,
                goal_db,
                reg_scale,
                tr0,
            ),
            _ => None,
        };

        for (pass_idx, &it) in active_passes.iter().enumerate() {
            #[cfg(not(target_arch = "wasm32"))]
            let (sol_l_opt, sol_r_opt) = if n >= 8_192 && build_l.is_some() && build_r.is_some() {
                std::thread::scope(|s| {
                    let h_l =
                        s.spawn(|| build_l.as_mut().map(|bl| run_decoupled_chan_pass(bl, it)));
                    let sol_r = build_r.as_mut().map(|br| run_decoupled_chan_pass(br, it));
                    let sol_l = h_l.join().ok().flatten();
                    (sol_l, sol_r)
                })
            } else {
                (
                    build_l.as_mut().map(|bl| run_decoupled_chan_pass(bl, it)),
                    build_r.as_mut().map(|br| run_decoupled_chan_pass(br, it)),
                )
            };
            #[cfg(target_arch = "wasm32")]
            let (sol_l_opt, sol_r_opt) = (
                build_l.as_mut().map(|bl| run_decoupled_chan_pass(bl, it)),
                build_r.as_mut().map(|br| run_decoupled_chan_pass(br, it)),
            );
            if let Some(sol_l) = sol_l_opt {
                write_theta(&sol_l, fir_len, k_cnt, use_s2, 0, &mut model.readouts[t]);
            }
            if let Some(sol_r) = sol_r_opt {
                write_theta(&sol_r, fir_len, k_cnt, use_s2, 1, &mut model.readouts[t]);
            } else {
                model.readouts[t].fir_r = model.readouts[t].fir_l;
                model.readouts[t].kautz_c_r = model.readouts[t].kautz_c_l;
                model.readouts[t].direct_s2_w[1] = model.readouts[t].direct_s2_w[0];
            }
            if pass_idx + 1 < active_passes.len() {
                model.readouts[t].mask_to_tier(tier);
                let (mid_l, mid_r) = render_tier_stereo::<B>(
                    &model.extract_tier(tier),
                    &stem.in_l[..n],
                    &stem.in_r[..n],
                );
                if let Some(ref mut bl) = build_l {
                    let cur_l = read_theta(&model.readouts[t], fir_len, k_cnt, use_s2, 0);
                    apply_decoupled_ltv_defect(bl.basis, &sig.y0_l, &cur_l, &mid_l, &mut bl.y0_adj);
                    bl.init_z = bl.opt.warm_reset(&cur_l);
                }
                if let Some(ref mut br) = build_r {
                    let cur_r = read_theta(&model.readouts[t], fir_len, k_cnt, use_s2, 1);
                    apply_decoupled_ltv_defect(br.basis, &sig.y0_r, &cur_r, &mid_r, &mut br.y0_adj);
                    br.init_z = br.opt.warm_reset(&cur_r);
                }
            }
        }
    }
    model.readouts[t].mask_to_tier(tier);

    let loss_before = eval_rendered_polish_loss(&before_l, &before_r, spec, coupled, mono);
    let (after_l, after_r) =
        render_tier_stereo::<B>(&model.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);
    let loss_after = eval_rendered_polish_loss(&after_l, &after_r, spec, coupled, mono);
    if !loss_after.is_finite() || loss_after > loss_before + 1.0e-6 {
        let ro_after = model.readouts[t].clone();
        model.readouts[t] = blend_readout_half(&ro_before, &ro_after);
        model.readouts[t].mask_to_tier(tier);
        let (half_l, half_r) =
            render_tier_stereo::<B>(&model.extract_tier(tier), &stem.in_l[..n], &stem.in_r[..n]);
        let loss_half = eval_rendered_polish_loss(&half_l, &half_r, spec, coupled, mono);
        let after_ok = loss_after.is_finite();
        let half_ok = loss_half.is_finite();
        let before_ok = loss_before.is_finite();
        let keep_half = half_ok
            && (!before_ok || loss_half <= loss_before)
            && (!after_ok || loss_half <= loss_after);
        if !keep_half {
            let restore_after = after_ok && (!before_ok || loss_after < loss_before);
            model.readouts[t] = if restore_after { ro_after } else { ro_before };
            model.readouts[t].mask_to_tier(tier);
        }
    }
}
