#![forbid(unsafe_code)]
//! Parallel Per-Tier Readout Forward/Backward Passes and Single-Pass Fused `d_sub_*` Reduction.

use dnam_engine::Tier;
use dnam_engine::stages::stage2_gru::{
    MAX_BLOCK_LANES, MAX_KAUTZ_POLES, MAX_TOTAL_GRU_LANES, NUM_GRU_BLOCKS, tier_block_lanes_array,
    tier_fir_segments,
};

use crate::solver::backends::TrainerBackend;

use super::super::gabor::eval_gabor_loss_and_adjoint_with_backend;
use super::super::gru_core::{AdjointWorkspace, TierReadoutWorkspace};
use super::super::ltas_adj::eval_welch33_ltas_loss_and_adjoint;
use super::super::model_f64::{V3MatryoshkaOptF64, V3TierReadoutF64, V3TrainDataset};
use super::super::primitives::{
    ClampedCoreParams, downsample_2x_fwd, downsample_2x_rev, par_map_indexed,
};

pub(super) const ALL_TIERS: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

pub(super) struct StemEvalCtx<'a> {
    pub(super) model: &'a V3MatryoshkaOptF64,
    pub(super) cp: &'a ClampedCoreParams,
    pub(super) dataset: &'a V3TrainDataset,
    pub(super) st_idx: usize,
    pub(super) n: usize,
    pub(super) active_ch: usize,
    pub(super) ch_scale: f64,
}

struct TierReadoutGrad {
    gr: Box<V3TierReadoutF64>,
    gk: [(f64, f64); MAX_KAUTZ_POLES],
    d_dec: [f64; 3],
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn project_substep(
    r: &V3TierReadoutF64,
    dir_t: &[f64; 7],
    ch: usize,
    c_t: [usize; NUM_GRU_BLOCKS],
    taps: &[f64; 7],
    u: f64,
    hp: f64,
    h: &[f64; MAX_TOTAL_GRU_LANES],
) -> (f64, f64) {
    let mut s2v = r.pre_cab_u_gain[ch] * u + r.pre_cab_hp_gain[ch] * hp;
    let mut dirv = r.direct_s2_u_gain[ch] * u + r.direct_s2_hp_gain[ch] * hp;
    for i in 0..7 {
        dirv += dir_t[i] * taps[i];
    }
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        for idx in base..base + c_t[b] {
            s2v += r.pre_cab_w[ch][idx] * h[idx];
            dirv += r.direct_s2_w[ch][idx] * h[idx];
        }
    }
    (s2v, dirv)
}

#[allow(clippy::too_many_arguments)]
fn backward_substep_readout_weights(
    gr: &mut V3TierReadoutF64,
    ws: &AdjointWorkspace,
    tws: &mut TierReadoutWorkspace,
    ch: usize,
    sub_t: usize,
    ddir: f64,
    ds2_s: f64,
    c_t: [usize; NUM_GRU_BLOCKS],
) {
    tws.ddir_sub[ch][sub_t] = ddir;
    tws.ds2_sub[ch][sub_t] = ds2_s;
    let d_dir_t = if ch == 0 {
        &mut gr.svf_direct_taps_l
    } else {
        &mut gr.svf_direct_taps_r
    };
    let st_taps = &ws.sub_taps[ch][sub_t];
    for i in 0..7 {
        d_dir_t[i] += st_taps[i] * ddir;
    }
    let su = ws.sub_u[ch][sub_t];
    let shp = ws.sub_hp[ch][sub_t];
    let sh = &ws.sub_h[ch][sub_t];
    gr.direct_s2_u_gain[ch] += su * ddir;
    gr.direct_s2_hp_gain[ch] += shp * ddir;
    gr.pre_cab_u_gain[ch] += su * ds2_s;
    gr.pre_cab_hp_gain[ch] += shp * ds2_s;
    for b in 0..NUM_GRU_BLOCKS {
        let base = b * MAX_BLOCK_LANES;
        for idx in base..base + c_t[b] {
            gr.direct_s2_w[ch][idx] += sh[idx] * ddir;
            gr.pre_cab_w[ch][idx] += sh[idx] * ds2_s;
        }
    }
}

struct TierPassDims {
    tau: usize,
    c_t: [usize; NUM_GRU_BLOCKS],
    f_t: usize,
    k_t: usize,
}

fn eval_tier_forward_pass<B: TrainerBackend>(
    ctx: &StemEvalCtx<'_>,
    dims: &TierPassDims,
    need_grad: bool,
    ws: &AdjointWorkspace,
    tws: &mut TierReadoutWorkspace,
) -> f64 {
    let (tau, c_t, f_t, k_t) = (dims.tau, dims.c_t, dims.f_t, dims.k_t);
    let (n, active_ch) = (ctx.n, ctx.active_ch);
    let (m, cp, dataset, st_idx) = (ctx.model, ctx.cp, ctx.dataset, ctx.st_idx);
    let stem = &dataset.stems[st_idx];
    let (w, sw, wt) = (
        stem.warmup.min(n),
        dataset.stem_weights[st_idx],
        m.active_tier_weights[tau],
    );
    let r = &m.readouts[tau];
    let mut tier_loss = 0.0_f64;

    for ch in 0..active_ch {
        let (mut s_re, mut s_im) = ([0.0_f64; MAX_KAUTZ_POLES], [0.0_f64; MAX_KAUTZ_POLES]);
        let (mut dec_s2_p, mut dec_dir_p) = (0.0_f64, 0.0_f64);
        let (mut dec_s2_aa, mut dec_dir_aa) = ([0.0_f64; 4], [0.0_f64; 4]);
        let dir_t = if ch == 0 {
            &r.svf_direct_taps_l
        } else {
            &r.svf_direct_taps_r
        };
        let fir = if ch == 0 { &r.fir_l } else { &r.fir_r };
        let kc = if ch == 0 { &r.kautz_c_l } else { &r.kautz_c_r };
        let mut sc_re = [0.0_f64; MAX_KAUTZ_POLES];
        let mut sc_im = [0.0_f64; MAX_KAUTZ_POLES];
        for t in 0..n {
            let dirv = if !m.os2x {
                let (s2v, dv) = project_substep(
                    r,
                    dir_t,
                    ch,
                    c_t,
                    &ws.sub_taps[ch][t],
                    ws.sub_u[ch][t],
                    ws.sub_hp[ch][t],
                    &ws.sub_h[ch][t],
                );
                tws.s2_tier[ch][t] = s2v;
                dv
            } else {
                let (s0, s1) = (t * 2, t * 2 + 1);
                let (s2_0, d_0) = project_substep(
                    r,
                    dir_t,
                    ch,
                    c_t,
                    &ws.sub_taps[ch][s0],
                    ws.sub_u[ch][s0],
                    ws.sub_hp[ch][s0],
                    &ws.sub_h[ch][s0],
                );
                let (s2_1, d_1) = project_substep(
                    r,
                    dir_t,
                    ch,
                    c_t,
                    &ws.sub_taps[ch][s1],
                    ws.sub_u[ch][s1],
                    ws.sub_hp[ch][s1],
                    &ws.sub_h[ch][s1],
                );
                if need_grad {
                    tws.s2_sub_tier[ch][s0] = s2_0;
                    tws.s2_sub_tier[ch][s1] = s2_1;
                    tws.dir_sub_tier[ch][s0] = d_0;
                    tws.dir_sub_tier[ch][s1] = d_1;
                }
                let (s2v, _, s2_diff) = downsample_2x_fwd(
                    s2_0,
                    s2_1,
                    &mut dec_s2_p,
                    &mut dec_s2_aa,
                    cp.dec_c_prev1.0,
                    cp.dec_c0.0,
                    cp.dec_aa_mix.0,
                    need_grad,
                );
                let (dv, _, d_diff) = downsample_2x_fwd(
                    d_0,
                    d_1,
                    &mut dec_dir_p,
                    &mut dec_dir_aa,
                    cp.dec_c_prev1.0,
                    cp.dec_c0.0,
                    cp.dec_aa_mix.0,
                    need_grad,
                );
                tws.s2_tier[ch][t] = s2v;
                if need_grad {
                    tws.s2_aa_diff[ch][t] = s2_diff;
                    tws.dir_aa_diff[ch][t] = d_diff;
                }
                dv
            };
            let x_del = if t >= 16 {
                tws.s2_tier[ch][t - 16]
            } else {
                0.0
            };
            let (ore, oim) = if need_grad {
                let off = t * k_t;
                (
                    &mut tws.kautz_re[ch][off..off + k_t],
                    &mut tws.kautz_im[ch][off..off + k_t],
                )
            } else {
                (&mut sc_re[..k_t], &mut sc_im[..k_t])
            };
            let kv = B::step_kautz_forward_frame(
                x_del,
                &cp.kautz_pre[..k_t],
                &kc[..k_t],
                &mut s_re[..k_t],
                &mut s_im[..k_t],
                ore,
                oim,
            );
            tws.y_tier[ch][t] = dirv + kv;
        }
        for k in 0..f_t.min(n) {
            B::axpy_f64(&mut tws.y_tier[ch][k..n], fir[k], &tws.s2_tier[ch][..n - k]);
        }

        let tgt = if ch == 0 {
            &stem.tgt_l[..n]
        } else {
            &stem.tgt_r[..n]
        };
        for t in 0..n {
            tws.err_buf[t] = tws.y_tier[ch][t] - f64::from(tgt[t]);
        }
        if need_grad {
            tws.dy_tier[ch][..n].fill(0.0);
        }
        let bw = dataset.stem_band_weights[st_idx][ch].map(|v| wt * ctx.ch_scale * v);
        tier_loss += dataset.bank.eval_loss_and_adjoint(
            &tws.err_buf[..n],
            w,
            bw,
            need_grad.then(|| &mut tws.dy_tier[ch][..n]),
            &mut tws.bezout_scratch,
        );
        tier_loss += eval_gabor_loss_and_adjoint_with_backend::<B>(
            &dataset.gabor_bank,
            &dataset.gabor_summaries[st_idx][ch],
            &tws.y_tier[ch][..n],
            need_grad.then(|| &mut tws.dy_tier[ch][..n]),
            wt * sw,
            ctx.ch_scale,
            dataset.gabor_env_weight,
            dataset.gabor_ltas_weight,
            &mut tws.gabor_scratch,
        );
        tier_loss += eval_welch33_ltas_loss_and_adjoint(
            &dataset.welch33_summaries[st_idx][ch],
            &tws.y_tier[ch][..n],
            need_grad.then(|| &mut tws.dy_tier[ch][..n]),
            wt * sw,
            ctx.ch_scale,
            dataset.welch_ltas_weight,
            &mut tws.welch_scratch,
        );
    }

    tier_loss
}

fn eval_tier_backward_pass<B: TrainerBackend>(
    ctx: &StemEvalCtx<'_>,
    dims: &TierPassDims,
    ws: &AdjointWorkspace,
    tws: &mut TierReadoutWorkspace,
) -> TierReadoutGrad {
    let (tau, c_t, f_t, k_t) = (dims.tau, dims.c_t, dims.f_t, dims.k_t);
    let (m, cp, n, active_ch) = (ctx.model, ctx.cp, ctx.n, ctx.active_ch);
    let r = &m.readouts[tau];
    let mut tg = TierReadoutGrad {
        gr: Box::new(V3TierReadoutF64::zeros()),
        gk: [(0.0, 0.0); MAX_KAUTZ_POLES],
        d_dec: [0.0; 3],
    };
    let (gr, gk) = (&mut *tg.gr, &mut tg.gk);

    for ch in 0..active_ch {
        tws.ds2_tier[ch][..n].fill(0.0);
        let (mut ds_re, mut ds_im) = ([0.0_f64; MAX_KAUTZ_POLES], [0.0_f64; MAX_KAUTZ_POLES]);
        let (mut r_dec_s2, mut r_dec_dir) = (0.0_f64, 0.0_f64);
        let (mut r_dec_s2_aa, mut r_dec_dir_aa) = ([0.0_f64; 4], [0.0_f64; 4]);
        let fir = if ch == 0 { &r.fir_l } else { &r.fir_r };
        let d_fir = if ch == 0 {
            &mut gr.fir_l
        } else {
            &mut gr.fir_r
        };
        for k in 0..f_t.min(n) {
            d_fir[k] += B::dot_f64(&tws.s2_tier[ch][..n - k], &tws.dy_tier[ch][k..n]);
            B::axpy_f64(
                &mut tws.ds2_tier[ch][..n - k],
                fir[k],
                &tws.dy_tier[ch][k..n],
            );
        }
        let kc = if ch == 0 { &r.kautz_c_l } else { &r.kautz_c_r };
        for t in (0..n).rev() {
            let dy = tws.dy_tier[ch][t];
            let x_del = if t >= 16 {
                tws.s2_tier[ch][t - 16]
            } else {
                0.0
            };
            let off = t * k_t;
            let (prev_re, prev_im) = if t > 0 {
                let po = (t - 1) * k_t;
                (
                    Some(&tws.kautz_re[ch][po..po + k_t]),
                    Some(&tws.kautz_im[ch][po..po + k_t]),
                )
            } else {
                (None, None)
            };
            let d_kc = if ch == 0 {
                &mut gr.kautz_c_l[..k_t]
            } else {
                &mut gr.kautz_c_r[..k_t]
            };
            let dx_del = B::step_kautz_backward_frame(
                dy,
                x_del,
                &cp.kautz_pre[..k_t],
                &kc[..k_t],
                &tws.kautz_re[ch][off..off + k_t],
                &tws.kautz_im[ch][off..off + k_t],
                prev_re,
                prev_im,
                &mut ds_re[..k_t],
                &mut ds_im[..k_t],
                d_kc,
                &mut gk[..k_t],
            );
            if t >= 16 {
                tws.ds2_tier[ch][t - 16] += dx_del;
            }
            let ds2 = tws.ds2_tier[ch][t];

            let (sub_pair, n_sub) = if !m.os2x {
                ([(t, dy, ds2), (0, 0.0, 0.0)], 1)
            } else {
                let (s0, s1) = (t * 2, t * 2 + 1);
                let sp_prev = t.checked_sub(1).map(|p| p * 2 + 1);
                let old_dir = sp_prev.map_or(0.0, |p| tws.dir_sub_tier[ch][p]);
                let old_s2 = sp_prev.map_or(0.0, |p| tws.s2_sub_tier[ch][p]);
                let (dd0, dd1, dcp_d, dc0_d, daa_d) = downsample_2x_rev(
                    dy,
                    old_dir,
                    tws.dir_sub_tier[ch][s0],
                    tws.dir_sub_tier[ch][s1],
                    tws.dir_aa_diff[ch][t],
                    cp.dec_c_prev1.0,
                    cp.dec_c0.0,
                    cp.dec_aa_mix.0,
                    &mut r_dec_dir,
                    &mut r_dec_dir_aa,
                );
                let (ds0, ds1, dcp_s, dc0_s, daa_s) = downsample_2x_rev(
                    ds2,
                    old_s2,
                    tws.s2_sub_tier[ch][s0],
                    tws.s2_sub_tier[ch][s1],
                    tws.s2_aa_diff[ch][t],
                    cp.dec_c_prev1.0,
                    cp.dec_c0.0,
                    cp.dec_aa_mix.0,
                    &mut r_dec_s2,
                    &mut r_dec_s2_aa,
                );
                tg.d_dec[0] += dcp_d + dcp_s;
                tg.d_dec[1] += dc0_d + dc0_s;
                tg.d_dec[2] += daa_d + daa_s;
                ([(s0, dd0, ds0), (s1, dd1, ds1)], 2)
            };

            for &(sub_t, ddir, ds2_s) in &sub_pair[..n_sub] {
                backward_substep_readout_weights(gr, ws, tws, ch, sub_t, ddir, ds2_s, c_t);
            }
        }
    }

    tg
}

fn eval_tier_forward_and_backward<B: TrainerBackend>(
    ctx: &StemEvalCtx<'_>,
    tau: usize,
    k_t: usize,
    need_grad: bool,
    ws: &AdjointWorkspace,
) -> (usize, f64, Option<TierReadoutGrad>) {
    let tier = ALL_TIERS[tau];
    let dims = TierPassDims {
        tau,
        c_t: tier_block_lanes_array(tier),
        f_t: tier_fir_segments(tier) * 16,
        k_t,
    };
    let mut tws = ws.tier_ws[tau]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tier_loss = eval_tier_forward_pass::<B>(ctx, &dims, need_grad, ws, &mut tws);
    let tier_grad = need_grad.then(|| eval_tier_backward_pass::<B>(ctx, &dims, ws, &mut tws));
    (tau, tier_loss, tier_grad)
}

fn fuse_d_sub_across_tiers(
    model: &V3MatryoshkaOptF64,
    active_taus: &[usize],
    active_ch: usize,
    total_sub: usize,
    ws: &mut AdjointWorkspace,
) {
    let guards: Vec<(
        usize,
        [usize; NUM_GRU_BLOCKS],
        std::sync::MutexGuard<'_, TierReadoutWorkspace>,
    )> = active_taus
        .iter()
        .map(|&tau| {
            (
                tau,
                tier_block_lanes_array(ALL_TIERS[tau]),
                ws.tier_ws[tau]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            )
        })
        .collect();

    for ch in 0..active_ch {
        let d_taps = &mut ws.d_sub_taps[ch];
        let d_u = &mut ws.d_sub_u[ch];
        let d_hp = &mut ws.d_sub_hp[ch];
        let d_h = &mut ws.d_sub_h[ch];
        for sub_t in 0..total_sub {
            let mut dt = [0.0_f64; 7];
            let mut du = 0.0_f64;
            let mut dhp = 0.0_f64;
            let mut dh = [0.0_f64; MAX_TOTAL_GRU_LANES];
            for &(tau, c_t, ref tws) in &guards {
                let r = &model.readouts[tau];
                let ddir = tws.ddir_sub[ch][sub_t];
                let ds2_s = tws.ds2_sub[ch][sub_t];
                let dir_t = if ch == 0 {
                    &r.svf_direct_taps_l
                } else {
                    &r.svf_direct_taps_r
                };
                for i in 0..7 {
                    dt[i] += dir_t[i] * ddir;
                }
                du += r.direct_s2_u_gain[ch] * ddir + r.pre_cab_u_gain[ch] * ds2_s;
                dhp += r.direct_s2_hp_gain[ch] * ddir + r.pre_cab_hp_gain[ch] * ds2_s;
                for b in 0..NUM_GRU_BLOCKS {
                    let base = b * MAX_BLOCK_LANES;
                    for idx in base..base + c_t[b] {
                        dh[idx] += r.direct_s2_w[ch][idx] * ddir + r.pre_cab_w[ch][idx] * ds2_s;
                    }
                }
            }
            d_taps[sub_t] = dt;
            d_u[sub_t] = du;
            d_hp[sub_t] = dhp;
            d_h[sub_t] = dh;
        }
    }
}

pub(super) fn eval_active_tiers_and_fuse<B: TrainerBackend>(
    ctx: &StemEvalCtx<'_>,
    tiers_needed: &[usize; 4],
    total_sub: usize,
    ws: &mut AdjointWorkspace,
    grad: Option<&mut V3MatryoshkaOptF64>,
) -> f64 {
    let mut active_taus = Vec::with_capacity(4);
    for tau in 0..4 {
        if tiers_needed[tau] > 0 && ctx.model.active_tier_weights[tau] > 0.0 {
            active_taus.push(tau);
        }
    }
    if active_taus.is_empty() {
        if grad.is_some() {
            for ch in 0..ctx.active_ch {
                ws.d_sub_taps[ch][..total_sub].fill([0.0; 7]);
                ws.d_sub_u[ch][..total_sub].fill(0.0);
                ws.d_sub_hp[ch][..total_sub].fill(0.0);
                ws.d_sub_h[ch][..total_sub].fill([0.0; MAX_TOTAL_GRU_LANES]);
            }
        }
        return 0.0;
    }

    let need_grad = grad.is_some();
    let results: Vec<(usize, f64, Option<TierReadoutGrad>)> =
        if active_taus.len() >= 2 && ctx.n >= 2_048 {
            let ws_ref: &AdjointWorkspace = ws;
            par_map_indexed(active_taus.len(), |idx| {
                let tau = active_taus[idx];
                eval_tier_forward_and_backward::<B>(ctx, tau, tiers_needed[tau], need_grad, ws_ref)
            })
        } else {
            active_taus
                .iter()
                .map(|&tau| {
                    eval_tier_forward_and_backward::<B>(ctx, tau, tiers_needed[tau], need_grad, ws)
                })
                .collect()
        };

    let mut stem_loss = 0.0_f64;
    for &(_, l, _) in &results {
        stem_loss += l;
    }
    if let Some(g_mut) = grad {
        for (tau, _, tg_opt) in results {
            if let Some(tg) = tg_opt {
                g_mut.readouts[tau].add_assign(&tg.gr);
                let k_t = tiers_needed[tau];
                for m_idx in 0..k_t {
                    g_mut.kautz_poles[m_idx].0 += tg.gk[m_idx].0;
                    g_mut.kautz_poles[m_idx].1 += tg.gk[m_idx].1;
                }
                g_mut.os_dec_c_prev1 += ctx.cp.dec_c_prev1.1 * tg.d_dec[0];
                g_mut.os_dec_c0 += ctx.cp.dec_c0.1 * tg.d_dec[1];
                g_mut.os_dec_aa_mix += ctx.cp.dec_aa_mix.1 * tg.d_dec[2];
            }
        }
        fuse_d_sub_across_tiers(ctx.model, &active_taus, ctx.active_ch, total_sub, ws);
    }
    stem_loss
}
