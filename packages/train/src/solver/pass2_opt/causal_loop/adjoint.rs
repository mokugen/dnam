//! Reverse-time frame-interleaved BPTT over [`CausalLoopBuffers`] into [`Pass2FullGrads`].

use dnam_kernels::{KernelVec4, StereoVec8, TINY_CROSSING_EPS};
use dnam_vm::{SONA_CORE_QUAD_BLOCK_AND_LOCAL, SONA_ULTRA_QUAD_TRAVERSAL_ORDER};

use crate::cpu::core_and_adjoint::{
    FwdFrameState, LaneParams, ReverseCarry, ReverseStepContext, TileGradAccum, flush_zero,
    read_checkpoint, step_lane_forward, step_lane_reverse, write_lane_param_grads,
};
use crate::gpu::abi::{
    FEATURE_VEC4_STRIDE, QUAD_PARAMS_VEC4_STRIDE, TILE_FRAMES, quad_param_slots,
};
use crate::runner::pack_core_quad_params;

use super::super::super::model::SonaModelF64;
use super::super::super::model::cascade::sanitize_unit_f64;
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

    let kp_list: Vec<_> = (0..q_count)
        .map(|q| {
            let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
            let kp = model.core_quads[q]
                .rescaled_for_rho(buf.rho_core)
                .to_kernel_params_for_block(block_idx);
            let qp = pack_core_quad_params(&kp);
            let lps: [LaneParams; 4] =
                core::array::from_fn(|c| LaneParams::extract(&kp, qp[quad_param_slots::SVF_G], c));
            (block_idx == 0, lps)
        })
        .collect();

    let mut gx_tank_l = vec![0.0_f64; n];
    let mut gx_tank_r = vec![0.0_f64; n];
    let mut d_onset_l = vec![0.0_f64; n];
    let mut d_onset_r = vec![0.0_f64; n];
    let mut d_feats = vec![[0.0_f32; 4]; q_count * n * FEATURE_VEC4_STRIDE];
    let mut d_cb_l = vec![[0.0_f64; 4]; q_count * n];
    let mut d_cb_r = vec![[0.0_f64; 4]; q_count * n];
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
    let num_tiles = n.div_ceil(TILE_FRAMES);

    let mut tile_fwd = vec![vec![[FwdFrameState::default(); TILE_FRAMES]; 4]; q_count];

    for tile in (0..num_tiles).rev() {
        let tile_base = tile * TILE_FRAMES;
        let valid_len = n.saturating_sub(tile_base).min(TILE_FRAMES);
        if valid_len == 0 {
            continue;
        }
        for q in 0..q_count {
            let ckpt = read_checkpoint(&buf.tile_checkpoints, q, num_tiles, tile);
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
                    let t = tile_base + local_t;
                    let in_base = (q * n + t) * 2;
                    let u_l = flush_zero(buf.core_inputs[in_base][c]);
                    let u_r = flush_zero(buf.core_inputs[in_base + 1][c]);
                    tile_fwd[q][c][local_t] =
                        step_lane_forward(lp, &svf_c, &lru_c, u_l, u_r, &mut lane_state, is_phs);
                }
            }
        }

        let mut tile_acc = [[TileGradAccum::default(); 4]; 10];
        for local_t in (0..valid_len).rev() {
            let t = tile_base + local_t;
            if buf.dry_mix_open {
                grads.d_tank_dry_mix += gl[t] * buf.x_tank_l[t] + gr[t] * buf.x_tank_r[t];
            }
            let mut g_tank_l = buf.dry_mix * gl[t];
            let mut g_tank_r = buf.dry_mix * gr[t];

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
                        gl[t],
                        gr[t],
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

            let (lpl_t, lpr_t) = (buf.ro_lp_l[t], buf.ro_lp_r[t]);
            let (hpl_t, hpr_t) = (buf.ux_l[t] - lpl_t, buf.ux_r[t] - lpr_t);
            grads.d_direct_s2_u[0] += g_tank_l * buf.ux_l[t];
            grads.d_direct_s2_u[1] += g_tank_r * buf.ux_r[t];
            grads.d_direct_s2_hp[0] += g_tank_l * hpl_t;
            grads.d_direct_s2_hp[1] += g_tank_r * hpr_t;
            grads.d_pre_conv_u[0] += g_pre_l * buf.ux_l[t];
            grads.d_pre_conv_u[1] += g_pre_r * buf.ux_r[t];
            grads.d_pre_conv_hp[0] += g_pre_l * hpl_t;
            grads.d_pre_conv_hp[1] += g_pre_r * hpr_t;

            let d_hp_l =
                model.direct_s2_hp_gain[0] * g_tank_l + model.pre_conv_hp_gain[0] * g_pre_l;
            let d_hp_r =
                model.direct_s2_hp_gain[1] * g_tank_r + model.pre_conv_hp_gain[1] * g_pre_r;
            let mut g_ux_l =
                model.direct_s2_u_gain[0] * g_tank_l + model.pre_conv_u_gain[0] * g_pre_l;
            let mut g_ux_r =
                model.direct_s2_u_gain[1] * g_tank_r + model.pre_conv_u_gain[1] * g_pre_r;

            for q in 0..q_count {
                let fbase = (q * n + t) * FEATURE_VEC4_STRIDE;
                let vl = buf.feature_outputs[fbase];
                let vr = buf.feature_outputs[fbase + 1];
                let ol = buf.feature_outputs[fbase + 2];
                let or_ = buf.feature_outputs[fbase + 3];
                let el = buf.feature_outputs[fbase + 4];
                let er = buf.feature_outputs[fbase + 5];
                for k in 0..4 {
                    let row_i = q * 4 + k;
                    let r = &model.w_ro[row_i];
                    let dw = &mut grads.d_w_ro[row_i];
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
                    d_feats[fbase][k] += (g_pre_l * r[4] + g_tank_l * r[10]) as f32;
                    d_feats[fbase + 1][k] += (g_pre_r * r[5] + g_tank_r * r[11]) as f32;
                    d_feats[fbase + 2][k] += (g_pre_l * r[0] + g_tank_l * r[6]) as f32;
                    d_feats[fbase + 3][k] += (g_pre_r * r[1] + g_tank_r * r[7]) as f32;
                    d_feats[fbase + 4][k] += (g_pre_l * r[2] + g_tank_l * r[8]) as f32;
                    d_feats[fbase + 5][k] += (g_pre_r * r[3] + g_tank_r * r[9]) as f32;
                }
            }

            for &q in SONA_ULTRA_QUAD_TRAVERSAL_ORDER.iter().rev() {
                if q >= q_count {
                    continue;
                }
                let is_phs = kp_list[q].0;
                let mut gin_l = [0.0_f32; 4];
                let mut gin_r = [0.0_f32; 4];
                for c in 0..4 {
                    let ctx = ReverseStepContext {
                        d_loss_d_features: &d_feats,
                        feature_outputs: &buf.feature_outputs,
                        q_global: q,
                        num_frames: n,
                        c,
                        tiny_eps: TINY_CROSSING_EPS,
                    };
                    let (dul, dur) = step_lane_reverse(
                        &ctx,
                        &tile_fwd[q][c][local_t],
                        &kp_list[q].1[c],
                        t,
                        &mut carries[q][c],
                        &mut tile_acc[q][c],
                        is_phs,
                    );
                    gin_l[c] = dul;
                    gin_r[c] = dur;
                }
                let (add_l, add_r) = backprop_core_input_frame(
                    &CoreInputBackpropCtx {
                        model,
                        q,
                        t,
                        n,
                        q_count,
                        active_tiles,
                        ux_l_t: buf.ux_l[t],
                        ux_r_t: buf.ux_r[t],
                        feature_outputs: &buf.feature_outputs,
                    },
                    gin_l,
                    gin_r,
                    CoreInputBackpropOut {
                        d_feats: &mut d_feats,
                        d_quad: &mut grads.core_quads[q],
                        d_tiles: &mut grads.tiles,
                        d_cb_opt: Some((&mut d_cb_l, &mut d_cb_r)),
                    },
                );
                g_ux_l += add_l;
                g_ux_r += add_r;
            }

            let lp_p_l = if t > 0 { buf.ro_lp_l[t - 1] } else { 0.0 };
            let lp_p_r = if t > 0 { buf.ro_lp_r[t - 1] } else { 0.0 };
            g_ux_l += buf.ro_resolved.step_reverse_frame(
                d_hp_l,
                buf.ux_l[t],
                lp_p_l,
                &mut ro_adj_next[0],
                &mut ro_d_p,
            );
            g_ux_r += buf.ro_resolved.step_reverse_frame(
                d_hp_r,
                buf.ux_r[t],
                lp_p_r,
                &mut ro_adj_next[1],
                &mut ro_d_p,
            );

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
                &mut grads.op_n_pre,
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
                &mut grads.op_n_pre,
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
                        grads.crossbar.d_crossbar_op_gains[1] += g_s0l * wl * vr + g_s0r * wr * vr;
                    }
                    if wl_raw.is_finite() && wl_raw.abs() < 4.0 {
                        grads.crossbar.d_opt_to_lr[idx][0] += g_s0l * buf.op_t_gain * vr;
                    }
                    if wr_raw.is_finite() && wr_raw.abs() < 4.0 {
                        grads.crossbar.d_opt_to_lr[idx][1] += g_s0r * buf.op_t_gain * vr;
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

        let mut pg_buf = [[0.0_f32; 4]; QUAD_PARAMS_VEC4_STRIDE];
        for q in 0..q_count {
            let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
            let is_phs = block_idx == 0;
            for c in 0..4 {
                write_lane_param_grads(
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
