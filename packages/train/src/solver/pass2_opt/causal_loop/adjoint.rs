//! Reverse-time frame-interleaved BPTT over [`CausalLoopBuffers`] into [`Pass2FullGrads`].

use dnam_kernels::{KernelVec4, StereoVec8, TINY_CROSSING_EPS};
use dnam_vm::{SONA_CORE_QUAD_BLOCK_AND_LOCAL, SONA_ULTRA_QUAD_TRAVERSAL_ORDER};

use crate::gpu::abi::{FEATURE_VEC4_STRIDE, TILE_FRAMES};

use super::super::super::model::SonaModelF64;
use super::super::super::model::cascade::sanitize_unit_f64;
use super::super::super::model::v3_core_bridge::{
    FwdFrameState, LaneParams, ReverseCarry, ReverseStepContext, TileGradAccum,
    V3_QUAD_PARAMS_VEC4_STRIDE, flush_zero, pack_v3_core_quad_params, read_v3_checkpoint,
    step_lane_forward, step_lane_reverse, v3_quad_param_slots, write_v3_lane_param_grads,
};
use super::super::adjoints::KautzChannelReverseState;
use super::super::crossbar::assemble_crossbar_adjoint;
use super::super::eval::core_adj::{
    CoreInputBackpropCtx, CoreInputBackpropOut, Pass2FullGrads, active_tiles_for_quads,
    add_tile_param_grads_to_quad, backprop_core_input_frame, finalize_core_quad_zoh,
};
use super::super::op_n_pre::OpNPreFrameSample;
use super::super::op_t_quad::adjoint::OpTQuadReverseState;
use super::super::sensory::eval_sensory_quad_and_adjoint;
use super::CausalLoopBuffers;
use super::os2x::Os2xReverseState;

struct CoreSubframePullback<'a> {
    model: &'a SonaModelF64,
    buf: &'a CausalLoopBuffers,
    cux_l: &'a [f64],
    cux_r: &'a [f64],
    n_core: usize,
    tile_base: usize,
    active_tiles: usize,
    kp_list: &'a [(bool, [LaneParams; 4])],
    tile_fwd: &'a [Vec<[FwdFrameState; TILE_FRAMES]>],
    carries: &'a mut [[ReverseCarry; 4]; 10],
    tile_acc: &'a mut [[TileGradAccum; 4]; 10],
    d_feats: &'a mut [[f32; 4]],
    d_cb_l: &'a mut [[f64; 4]],
    d_cb_r: &'a mut [[f64; 4]],
    ro_adj_next: &'a mut [f64; 2],
    ro_d_p: &'a mut f64,
    grads: &'a mut Pass2FullGrads,
}

fn pullback_core_subframe(
    ctx: &mut CoreSubframePullback<'_>,
    sub_t: usize,
    g_out: [f64; 4],
) -> [f64; 2] {
    let [g_pre_l, g_pre_r, g_tank_l, g_tank_r] = g_out;
    let model = ctx.model;
    let buf = ctx.buf;
    let q_count = buf.q_count;
    let n_core = ctx.n_core;
    let local_sub = sub_t - ctx.tile_base;
    let cuxl_t = ctx.cux_l[sub_t];
    let cuxr_t = ctx.cux_r[sub_t];
    let (lpl_t, lpr_t) = (buf.ro_lp_l[sub_t], buf.ro_lp_r[sub_t]);
    let (hpl_t, hpr_t) = (cuxl_t - lpl_t, cuxr_t - lpr_t);

    ctx.grads.d_direct_s2_u[0] += g_tank_l * cuxl_t;
    ctx.grads.d_direct_s2_u[1] += g_tank_r * cuxr_t;
    ctx.grads.d_direct_s2_hp[0] += g_tank_l * hpl_t;
    ctx.grads.d_direct_s2_hp[1] += g_tank_r * hpr_t;
    ctx.grads.d_pre_conv_u[0] += g_pre_l * cuxl_t;
    ctx.grads.d_pre_conv_u[1] += g_pre_r * cuxr_t;
    ctx.grads.d_pre_conv_hp[0] += g_pre_l * hpl_t;
    ctx.grads.d_pre_conv_hp[1] += g_pre_r * hpr_t;

    let d_hp_l = model.direct_s2_hp_gain[0] * g_tank_l + model.pre_conv_hp_gain[0] * g_pre_l;
    let d_hp_r = model.direct_s2_hp_gain[1] * g_tank_r + model.pre_conv_hp_gain[1] * g_pre_r;
    let mut g_cux_l = model.direct_s2_u_gain[0] * g_tank_l + model.pre_conv_u_gain[0] * g_pre_l;
    let mut g_cux_r = model.direct_s2_u_gain[1] * g_tank_r + model.pre_conv_u_gain[1] * g_pre_r;

    for q in 0..q_count {
        let fbase = (q * n_core + sub_t) * FEATURE_VEC4_STRIDE;
        let vl = buf.feature_outputs[fbase];
        let vr = buf.feature_outputs[fbase + 1];
        let ol = buf.feature_outputs[fbase + 2];
        let or_ = buf.feature_outputs[fbase + 3];
        let el = buf.feature_outputs[fbase + 4];
        let er = buf.feature_outputs[fbase + 5];
        for k in 0..4 {
            let row_i = q * 4 + k;
            let r = &model.w_ro[row_i];
            let dw = &mut ctx.grads.d_w_ro[row_i];
            dw[0] += g_pre_l * f64::from(ol[k]);
            dw[1] += g_pre_r * f64::from(or_[k]);
            dw[2] += g_pre_l * f64::from(el[k]);
            dw[3] += g_pre_r * f64::from(er[k]);
            dw[4] += g_pre_l * f64::from(vl[k]);
            dw[5] += g_pre_r * f64::from(vr[k]);
            dw[6] += g_tank_l * f64::from(ol[k]);
            dw[7] += g_tank_r * f64::from(or_[k]);
            dw[8] += g_tank_l * f64::from(el[k]);
            dw[9] += g_tank_r * f64::from(er[k]);
            dw[10] += g_tank_l * f64::from(vl[k]);
            dw[11] += g_tank_r * f64::from(vr[k]);
            ctx.d_feats[fbase][k] += (g_pre_l * r[4] + g_tank_l * r[10]) as f32;
            ctx.d_feats[fbase + 1][k] += (g_pre_r * r[5] + g_tank_r * r[11]) as f32;
            ctx.d_feats[fbase + 2][k] += (g_pre_l * r[0] + g_tank_l * r[6]) as f32;
            ctx.d_feats[fbase + 3][k] += (g_pre_r * r[1] + g_tank_r * r[7]) as f32;
            ctx.d_feats[fbase + 4][k] += (g_pre_l * r[2] + g_tank_l * r[8]) as f32;
            ctx.d_feats[fbase + 5][k] += (g_pre_r * r[3] + g_tank_r * r[9]) as f32;
        }
    }

    for &q in SONA_ULTRA_QUAD_TRAVERSAL_ORDER.iter().rev() {
        if q >= q_count {
            continue;
        }
        let is_phs = ctx.kp_list[q].0;
        let mut gin_l = [0.0_f32; 4];
        let mut gin_r = [0.0_f32; 4];
        for c in 0..4 {
            let step_ctx = ReverseStepContext {
                d_loss_d_features: ctx.d_feats,
                feature_outputs: &buf.feature_outputs,
                q_global: q,
                num_frames: n_core,
                c,
                tiny_eps: TINY_CROSSING_EPS,
            };
            let (dul, dur) = step_lane_reverse(
                &step_ctx,
                &ctx.tile_fwd[q][c][local_sub],
                &ctx.kp_list[q].1[c],
                sub_t,
                &mut ctx.carries[q][c],
                &mut ctx.tile_acc[q][c],
                is_phs,
            );
            gin_l[c] = dul;
            gin_r[c] = dur;
        }
        let (add_l, add_r) = backprop_core_input_frame(
            &CoreInputBackpropCtx {
                model,
                q,
                t: sub_t,
                n: n_core,
                q_count,
                active_tiles: ctx.active_tiles,
                ux_l_t: cuxl_t,
                ux_r_t: cuxr_t,
                feature_outputs: &buf.feature_outputs,
            },
            gin_l,
            gin_r,
            CoreInputBackpropOut {
                d_feats: ctx.d_feats,
                d_quad: &mut ctx.grads.core_quads[q],
                d_tiles: &mut ctx.grads.tiles,
                d_cb_opt: Some((ctx.d_cb_l, ctx.d_cb_r)),
            },
        );
        g_cux_l += add_l;
        g_cux_r += add_r;
    }

    let lp_p_l = if sub_t > 0 {
        buf.ro_lp_l[sub_t - 1]
    } else {
        0.0
    };
    let lp_p_r = if sub_t > 0 {
        buf.ro_lp_r[sub_t - 1]
    } else {
        0.0
    };
    g_cux_l += buf.ro_resolved.step_reverse_frame(
        d_hp_l,
        cuxl_t,
        lp_p_l,
        &mut ctx.ro_adj_next[0],
        ctx.ro_d_p,
    );
    g_cux_r += buf.ro_resolved.step_reverse_frame(
        d_hp_r,
        cuxr_t,
        lp_p_r,
        &mut ctx.ro_adj_next[1],
        ctx.ro_d_p,
    );
    [g_cux_l, g_cux_r]
}

/// Runs exact frame-interleaved reverse-time BPTT over [`CausalLoopBuffers`] into [`Pass2FullGrads`].
pub(crate) fn assemble_causal_loop_adjoint(
    model: &SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    buf: &CausalLoopBuffers,
    gl: &[f64],
    gr: &[f64],
) -> Pass2FullGrads {
    let n = buf.pred_l.len();
    let q_count = buf.q_count;
    let active_tiles = active_tiles_for_quads(q_count);
    let mut grads = Pass2FullGrads::with_dims(q_count, buf.active_k);
    if n == 0 {
        return grads;
    }
    let os2x = buf.os2x;
    let n_core = if os2x { 2 * n } else { n };
    let cux_l: &[f64] = if os2x { &buf.core_ux_l } else { &buf.ux_l };
    let cux_r: &[f64] = if os2x { &buf.core_ux_r } else { &buf.ux_r };

    let kp_list: Vec<_> = (0..q_count)
        .map(|q| {
            let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
            let kp = model.core_quads[q]
                .rescaled_for_rho(buf.rho_core)
                .to_kernel_params_for_block(block_idx);
            let qp = pack_v3_core_quad_params(&kp);
            let lps: [LaneParams; 4] = core::array::from_fn(|c| {
                LaneParams::extract(&kp, qp[v3_quad_param_slots::SVF_G], c)
            });
            (block_idx == 0, lps)
        })
        .collect();

    let mut gx_tank_l = vec![0.0_f64; n];
    let mut gx_tank_r = vec![0.0_f64; n];
    let mut d_onset_l = vec![0.0_f64; n];
    let mut d_onset_r = vec![0.0_f64; n];
    let mut d_feats = vec![[0.0_f32; 4]; q_count * n_core * FEATURE_VEC4_STRIDE];
    let mut d_cb_l = vec![[0.0_f64; 4]; q_count * n_core];
    let mut d_cb_r = vec![[0.0_f64; 4]; q_count * n_core];
    let mut d_s0_l = vec![0.0_f64; n];
    let mut d_s0_r = vec![0.0_f64; n];

    let mut opt_rev: Vec<Option<OpTQuadReverseState>> = (0..buf.active_opt_quads)
        .map(|q| buf.opt_active[q].then(|| OpTQuadReverseState::new(n)))
        .collect();
    let mut kautz_rev_l = KautzChannelReverseState::new(buf.active_k);
    let mut kautz_rev_r = KautzChannelReverseState::new(buf.active_k);
    let mut carries = [[ReverseCarry::default(); 4]; 10];
    let mut op_n_adj_next = [0.0_f64; 2];
    let mut ro_adj_next = [0.0_f64; 2];
    let mut ro_d_p = 0.0_f64;
    let mut os2x_rev = Os2xReverseState::new(
        model.os_up_quad_coeff,
        model.os_dec_c_prev1,
        model.os_dec_c0,
    );
    let num_tiles = n_core.div_ceil(TILE_FRAMES);
    let mut tile_fwd = vec![vec![[FwdFrameState::default(); TILE_FRAMES]; 4]; q_count];

    for tile in (0..num_tiles).rev() {
        let tile_base = tile * TILE_FRAMES;
        let valid_len = n_core.saturating_sub(tile_base).min(TILE_FRAMES);
        if valid_len == 0 {
            continue;
        }
        for q in 0..q_count {
            let ckpt = read_v3_checkpoint(&buf.tile_checkpoints, q, num_tiles, tile);
            let (s1_l, s1_r) = ckpt.svf_s1.into_lr();
            let (s2_l, s2_r) = ckpt.svf_s2.into_lr();
            let (v_l, v_r) = ckpt.lru_v.into_lr();
            let (ap_l, ap_r) = ckpt.adaa_prev.into_lr();
            let (s1la, s1ra) = (s1_l.to_array(), s1_r.to_array());
            let (s2la, s2ra) = (s2_l.to_array(), s2_r.to_array());
            let (vla, vra) = (v_l.to_array(), v_r.to_array());
            let (apla, apra) = (ap_l.to_array(), ap_r.to_array());
            let is_phs = kp_list[q].0;
            for c in 0..4 {
                let lp = &kp_list[q].1[c];
                let (svf_c, lru_c) = (lp.svf_coeffs(), lp.lru_coeffs());
                let mut lane_state = [
                    s1la[c], s2la[c], s1ra[c], s2ra[c], vla[c], vra[c], apla[c], apra[c],
                ];
                for local_t in 0..valid_len {
                    let sub_t = tile_base + local_t;
                    let in_base = (q * n_core + sub_t) * 2;
                    let u_l = flush_zero(buf.core_inputs[in_base][c]);
                    let u_r = flush_zero(buf.core_inputs[in_base + 1][c]);
                    tile_fwd[q][c][local_t] =
                        step_lane_forward(lp, &svf_c, &lru_c, u_l, u_r, &mut lane_state, is_phs);
                }
            }
        }

        let mut tile_acc = [[TileGradAccum::default(); 4]; 10];
        let (t_lo, t_hi) = if os2x {
            (tile_base / 2, (tile_base + valid_len) / 2)
        } else {
            (tile_base, tile_base + valid_len)
        };
        let mut core_ctx = CoreSubframePullback {
            model,
            buf,
            cux_l,
            cux_r,
            n_core,
            tile_base,
            active_tiles,
            kp_list: &kp_list,
            tile_fwd: &tile_fwd,
            carries: &mut carries,
            tile_acc: &mut tile_acc,
            d_feats: &mut d_feats,
            d_cb_l: &mut d_cb_l,
            d_cb_r: &mut d_cb_r,
            ro_adj_next: &mut ro_adj_next,
            ro_d_p: &mut ro_d_p,
            grads: &mut grads,
        };

        for t in (t_lo..t_hi).rev() {
            if buf.dry_mix_open {
                core_ctx.grads.d_tank_dry_mix += gl[t] * buf.x_tank_l[t] + gr[t] * buf.x_tank_r[t];
            }
            if buf.wet_scale_open {
                core_ctx.grads.d_op_t_wet_scale +=
                    gl[t] * buf.op_t_wet_l[t] + gr[t] * buf.op_t_wet_r[t];
            }
            let mut g_tank_l = buf.dry_mix * gl[t];
            let mut g_tank_r = buf.dry_mix * gr[t];
            let gl_wet_t = buf.wet_scale * gl[t];
            let gr_wet_t = buf.wet_scale * gr[t];

            let mut d_vr_end = [[0.0_f64; 4]; 4];
            let mut d_rwin_freeze = [[0.0_f64; 4]; 4];
            for q_opt in 0..buf.active_opt_quads {
                if let (Some(rev), Some(fwd)) =
                    (opt_rev[q_opt].as_mut(), buf.opt_fwd[q_opt].as_ref())
                {
                    let (vr_e, rwf, dxl, dxr) = rev.reverse_end_of_frame_write(
                        fwd,
                        t,
                        buf.x_tank_l[t],
                        buf.x_tank_r[t],
                        gl_wet_t,
                        gr_wet_t,
                    );
                    d_vr_end[q_opt] = vr_e;
                    d_rwin_freeze[q_opt] = rwf;
                    g_tank_l += dxl;
                    g_tank_r += dxr;
                }
            }
            gx_tank_l[t] = g_tank_l;
            gx_tank_r[t] = g_tank_r;

            d_onset_l[t] =
                kautz_rev_l.step_reverse_frame(&buf.kautz_l, t, buf.onset_pre_l[t], g_tank_l);
            d_onset_r[t] =
                kautz_rev_r.step_reverse_frame(&buf.kautz_r, t, buf.onset_pre_r[t], g_tank_r);

            let mut g_pre_l = if t + 16 < n { d_onset_l[t + 16] } else { 0.0 };
            let mut g_pre_r = if t + 16 < n { d_onset_r[t + 16] } else { 0.0 };
            for k in 0..buf.active_fir.min(n - t) {
                g_pre_l += model.fir_l[k] * gx_tank_l[t + k];
                g_pre_r += model.fir_r[k] * gx_tank_r[t + k];
            }

            let g_1x = [g_pre_l, g_pre_r, g_tank_l, g_tank_r];
            let [g_ux_l, g_ux_r] = if os2x {
                let (g_sub1, g_sub0) = os2x_rev.split_decimator(g_1x);
                let gc1 = pullback_core_subframe(&mut core_ctx, 2 * t + 1, g_sub1);
                let gc0 = pullback_core_subframe(&mut core_ctx, 2 * t, g_sub0);
                os2x_rev.pullback_upsampler(gc0, gc1)
            } else {
                pullback_core_subframe(&mut core_ctx, t, g_1x)
            };

            let hp_p_l = if t > 0 { buf.op_n_held_l[t - 1] } else { 0.0 };
            let hp_p_r = if t > 0 { buf.op_n_held_r[t - 1] } else { 0.0 };
            let g_s0l = buf.op_n_resolved.step_reverse_frame(
                OpNPreFrameSample {
                    dy: g_ux_l,
                    xt: buf.s0_l[t],
                    ht: buf.op_n_held_l[t],
                    h_prev: hp_p_l,
                    g_clk: buf.op_n_gclk_l[t],
                    dg_dzh: buf.op_n_dg_l[t],
                },
                &mut op_n_adj_next[0],
                &mut core_ctx.grads.op_n_pre,
            );
            let g_s0r = buf.op_n_resolved.step_reverse_frame(
                OpNPreFrameSample {
                    dy: g_ux_r,
                    xt: buf.s0_r[t],
                    ht: buf.op_n_held_r[t],
                    h_prev: hp_p_r,
                    g_clk: buf.op_n_gclk_r[t],
                    dg_dzh: buf.op_n_dg_r[t],
                },
                &mut op_n_adj_next[1],
                &mut core_ctx.grads.op_n_pre,
            );
            d_s0_l[t] = g_s0l;
            d_s0_r[t] = g_s0r;

            let v_reads = &buf.v_reads_hist[t];
            for q_opt in 0..buf.active_opt_quads {
                if !buf.opt_active[q_opt] {
                    continue;
                }
                let mut d_vr_cb = [0.0_f64; 4];
                for l in 0..4 {
                    let idx = q_opt * 4 + l;
                    let vr = v_reads[q_opt][l];
                    let [wl_raw, wr_raw] = model.opt_to_lr[idx];
                    let wl = sanitize_unit_f64(wl_raw, -4.0, 4.0, 0.0);
                    let wr = sanitize_unit_f64(wr_raw, -4.0, 4.0, 0.0);
                    if buf.op_t_gain_open {
                        core_ctx.grads.crossbar.d_crossbar_op_gains[1] +=
                            g_s0l * wl * vr + g_s0r * wr * vr;
                    }
                    if wl_raw.is_finite() && wl_raw.abs() < 4.0 {
                        core_ctx.grads.crossbar.d_opt_to_lr[idx][0] += g_s0l * buf.op_t_gain * vr;
                    }
                    if wr_raw.is_finite() && wr_raw.abs() < 4.0 {
                        core_ctx.grads.crossbar.d_opt_to_lr[idx][1] += g_s0r * buf.op_t_gain * vr;
                    }
                    d_vr_cb[l] = buf.op_t_gain * (g_s0l * wl + g_s0r * wr);
                }
                if let (Some(rev), Some(fwd)) =
                    (opt_rev[q_opt].as_mut(), buf.opt_fwd[q_opt].as_ref())
                {
                    let d_vr_tot = core::array::from_fn(|k| d_vr_end[q_opt][k] + d_vr_cb[k]);
                    rev.reverse_top_of_frame_read(fwd, t, d_vr_tot, d_rwin_freeze[q_opt]);
                }
            }
        }

        let mut pg_buf = [[0.0_f32; 4]; V3_QUAD_PARAMS_VEC4_STRIDE];
        for q in 0..q_count {
            let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
            let is_phs = block_idx == 0;
            for c in 0..4 {
                write_v3_lane_param_grads(
                    &mut pg_buf,
                    0,
                    c,
                    &kp_list[q].1[c],
                    &tile_acc[q][c],
                    is_phs,
                );
            }
            add_tile_param_grads_to_quad(
                &model.core_quads[q],
                block_idx,
                &pg_buf,
                0,
                &mut grads.core_quads[q],
            );
        }
    }

    for q in 0..q_count {
        finalize_core_quad_zoh(&model.core_quads[q], buf.rho_core, &mut grads.core_quads[q]);
    }
    grads.d_ro_pole_48k = ro_d_p * buf.ro_resolved.dp_dro;

    let kgl = kautz_rev_l.finalize(&buf.kautz_l);
    let kgr = kautz_rev_r.finalize(&buf.kautz_r);
    for k in 0..buf.active_k {
        grads.d_kautz[k] = [
            kgl[k][0] + kgr[k][0],
            kgl[k][1] + kgr[k][1],
            kgl[k][2] + kgr[k][2],
            kgl[k][3],
            kgl[k][4],
            kgr[k][3],
            kgr[k][4],
        ];
    }

    for q_opt in 0..buf.active_opt_quads {
        if let (Some(rev), Some(fwd)) = (opt_rev[q_opt].take(), buf.opt_fwd[q_opt].as_ref()) {
            grads.op_t[q_opt] = rev.finalize(fwd);
        }
    }

    let mut d_u_sense_l = vec![0.0_f64; n];
    let mut d_u_sense_r = vec![0.0_f64; n];
    let mut cb_grads = assemble_crossbar_adjoint(
        model,
        &buf.cb_trace,
        &d_cb_l,
        &d_cb_r,
        Some(&mut d_u_sense_l),
        Some(&mut d_u_sense_r),
    );
    cb_grads.d_crossbar_op_gains[1] += grads.crossbar.d_crossbar_op_gains[1];
    cb_grads.d_opt_to_lr = grads.crossbar.d_opt_to_lr;
    grads.crossbar = cb_grads;

    for t in 0..n {
        d_u_sense_l[t] += d_s0_l[t];
        d_u_sense_r[t] += d_s0_r[t];
    }
    let (_, _, sense_grads) = eval_sensory_quad_and_adjoint(
        model,
        &in_l[..n],
        &in_r[..n],
        Some((&d_u_sense_l, &d_u_sense_r)),
        None,
    );
    grads.sensory = sense_grads;
    grads
}
