//! Frame-interleaved exact causal forward evaluator when `Op-T` Top-of-Frame crossbar
//! feedback (`opt_to_lr * crossbar_op_gains[1]`) closes a live recurrent loop
//! (`ref[train.pipeline.backbone-trajectory-opt]`, `ref[dsp.op-t.ring.declaration]`).

use dnam_kernels::{
    CoreQuadParamsStereo4, GpuVec4, InterleavedStereo8, KernelVec4, SonaBlockStateStereo8,
    StereoVec8, step_core_quad_phs_stereo8, step_core_quad_stereo8,
};
use dnam_vm::{SONA_CORE_QUAD_BLOCK_AND_LOCAL, SONA_ULTRA_QUAD_TRAVERSAL_ORDER};

use crate::cpu::core_and_adjoint::{canon_vec4, flush_zero, write_checkpoint};
use crate::gpu::abi::{FEATURE_VEC4_STRIDE, TILE_CKPT_VEC4_STRIDE, TILE_FRAMES};

use super::super::super::model::cascade::sanitize_unit_f64;
use super::super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64};
use super::super::adjoints::{KautzChannelStepper, RoPoleResolved};
use super::super::crossbar::{
    eval_crossbar_forward, eval_crossbar_forward_os2x, step_opt_crossbar_taps_and_adjoint,
};
use super::super::eval::core_adj::{
    active_opt_quads_for_core_quads, active_tiles_for_quads, collect_incoming_tiles,
};
use super::super::op_n_pre::OpNPreResolved;
use super::super::op_t_quad::forward::{OpTQuadStepState, OpTResolvedParams};
use super::super::op_t_quad::is_op_t_quad_active;
use super::super::sensory::eval_sensory_quad_and_adjoint;
use super::CausalLoopBuffers;
use super::os2x::{Os2xDecimator, Os2xUpsampler};

/// Evaluates the full frame-interleaved forward graph matching `dnam-vm`.
pub(crate) fn eval_causal_loop_forward(
    model: &SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    active_quads: usize,
    record_trace: bool,
) -> CausalLoopBuffers {
    let n = in_l.len().min(in_r.len());
    let q_count = active_quads.clamp(1, 10);
    let os2x = model.oversample_upper_tiers_2x && q_count > 4;
    let n_core = if os2x { 2 * n } else { n };
    let active_opt_quads = active_opt_quads_for_core_quads(q_count);
    let active_tiles = active_tiles_for_quads(q_count);
    let include_stereo_phase = q_count > 4;
    let sr = if model.sample_rate_hz.is_finite() && model.sample_rate_hz > 0.0 {
        model.sample_rate_hz
    } else {
        48_000.0
    };
    let rho_1x = 48_000.0 / sr.max(8_000.0);
    let rho_core = if os2x { 0.5 * rho_1x } else { rho_1x };

    let (u_sense_l, u_sense_r, _) =
        eval_sensory_quad_and_adjoint(model, &in_l[..n], &in_r[..n], None, None);
    let cb_trace = if os2x {
        eval_crossbar_forward_os2x(model, &u_sense_l, &u_sense_r, q_count, include_stereo_phase)
    } else {
        eval_crossbar_forward(model, &u_sense_l, &u_sense_r, q_count, include_stereo_phase)
    };

    let op_t_gain_raw = model.crossbar_op_gains[1];
    let op_t_gain = if op_t_gain_raw.is_finite() {
        op_t_gain_raw.clamp(-4.0, 4.0)
    } else {
        0.0
    };
    let op_t_gain_open = op_t_gain_raw.is_finite() && op_t_gain_raw.abs() < 4.0;
    let opt_to_lr_san: [[f64; 2]; 16] =
        core::array::from_fn(|i| model.opt_to_lr[i].map(|v| sanitize_unit_f64(v, -4.0, 4.0, 0.0)));
    let dry_mix = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let dry_mix_open = model.tank_dry_mix.is_finite() && model.tank_dry_mix.abs() < 2.0;
    let wet_scale = sanitize_unit_f64(model.op_t_wet_scale, 0.0, 2.0, 1.0);
    let wet_scale_open = model.op_t_wet_scale.is_finite()
        && model.op_t_wet_scale >= 0.0
        && model.op_t_wet_scale < 2.0;

    let op_n_resolved = OpNPreResolved::new(
        model.op_n_pre_gamma_bc,
        model.op_n_pre_zoh_hold_depth,
        model.op_n_pre_beta_q,
        sr,
    );
    let ro_resolved = RoPoleResolved::new(model.ro_pole_48k, rho_core);
    let active_fir = model.active_fir_taps.min(NUM_FIR_TAPS_ULTRA);
    let active_k = model.active_kautz_pairs.min(model.kautz.len());

    let mut opt_active = [false; 4];
    let mut opt_fwd: Vec<Option<OpTQuadStepState>> = (0..active_opt_quads)
        .map(|q| {
            let active = is_op_t_quad_active(&model.transport_quads[q]);
            opt_active[q] = active;
            active.then(|| {
                let p = OpTResolvedParams::from_quad_with_wet_dc(
                    &model.transport_quads[q],
                    sr,
                    n,
                    model.dc_blocker_pole_5hz,
                );
                OpTQuadStepState::new(p, n, record_trace)
            })
        })
        .collect();

    let kparams: Vec<CoreQuadParamsStereo4<GpuVec4>> = (0..q_count)
        .map(|q| {
            let block_idx = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 as u32;
            model.core_quads[q]
                .rescaled_for_rho(rho_core)
                .to_kernel_params_for_block(block_idx)
        })
        .collect();
    let in_mids: Vec<[f64; 4]> = (0..q_count)
        .map(|q| {
            core::array::from_fn(|d| {
                sanitize_unit_f64(model.core_quads[q].in_mid_weight[d], -16.0, 16.0, 0.0)
            })
        })
        .collect();
    let in_sides: Vec<[f64; 4]> = (0..q_count)
        .map(|q| {
            core::array::from_fn(|d| {
                sanitize_unit_f64(model.core_quads[q].in_side_weight[d], -16.0, 16.0, 0.0)
            })
        })
        .collect();
    let mut quad_done = [false; 10];
    let mut incoming_per_q: Vec<_> = (0..q_count).map(|_| Vec::new()).collect();
    for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER {
        if q < q_count {
            incoming_per_q[q] = collect_incoming_tiles(model, q, q_count, active_tiles, &quad_done);
            quad_done[q] = true;
        }
    }
    let mut quad_states: Vec<SonaBlockStateStereo8<InterleavedStereo8<GpuVec4>>> =
        vec![SonaBlockStateStereo8::zero(); q_count];

    let num_tiles = n_core.div_ceil(TILE_FRAMES);
    let ckpt_per_quad = if record_trace {
        num_tiles * TILE_CKPT_VEC4_STRIDE
    } else {
        0
    };
    let mut core_inputs = if record_trace {
        vec![[0.0_f32; 4]; q_count * n_core * 2]
    } else {
        Vec::new()
    };
    let mut tile_checkpoints = vec![[0.0_f32; 4]; q_count * ckpt_per_quad];
    let mut feature_outputs = vec![[0.0_f32; 4]; q_count * n_core * FEATURE_VEC4_STRIDE];

    let mut kautz_l = KautzChannelStepper::new(
        &model.kautz[..active_k],
        false,
        if record_trace { n } else { 0 },
    );
    let mut kautz_r = KautzChannelStepper::new(
        &model.kautz[..active_k],
        true,
        if record_trace { n } else { 0 },
    );

    let mut pred_l = vec![0.0_f64; n];
    let mut pred_r = vec![0.0_f64; n];
    let mut op_t_wet_l = vec![0.0_f64; if record_trace { n } else { 0 }];
    let mut op_t_wet_r = vec![0.0_f64; if record_trace { n } else { 0 }];
    let mut s0_l = Vec::with_capacity(if record_trace { n } else { 0 });
    let mut s0_r = Vec::with_capacity(if record_trace { n } else { 0 });
    let mut ux_l = Vec::with_capacity(n);
    let mut ux_r = Vec::with_capacity(n);
    let cap_core = if record_trace && os2x { n_core } else { 0 };
    let mut core_ux_l = Vec::with_capacity(cap_core);
    let mut core_ux_r = Vec::with_capacity(cap_core);
    let mut pre_conv_l = vec![0.0_f64; n];
    let mut pre_conv_r = vec![0.0_f64; n];
    let mut onset_pre_l = vec![0.0_f64; n];
    let mut onset_pre_r = vec![0.0_f64; n];
    let mut x_tank_l = vec![0.0_f64; n];
    let mut x_tank_r = vec![0.0_f64; n];

    let cap_t = if record_trace { n } else { 0 };
    let cap_ro = if record_trace { n_core } else { 0 };
    let mut op_n_held_l = Vec::with_capacity(cap_t);
    let mut op_n_held_r = Vec::with_capacity(cap_t);
    let mut op_n_gclk_l = Vec::with_capacity(cap_t);
    let mut op_n_dg_l = Vec::with_capacity(cap_t);
    let mut op_n_gclk_r = Vec::with_capacity(cap_t);
    let mut op_n_dg_r = Vec::with_capacity(cap_t);
    let mut ro_lp_l = Vec::with_capacity(cap_ro);
    let mut ro_lp_r = Vec::with_capacity(cap_ro);
    let mut v_reads_hist = Vec::with_capacity(cap_t);

    let (mut op_n_prev_l, mut op_n_prev_r) = (0.0_f64, 0.0_f64);
    let (mut ro_prev_l, mut ro_prev_r) = (0.0_f64, 0.0_f64);
    let mut os_up = Os2xUpsampler::new(model.os_up_quad_coeff);
    let mut os_dec = Os2xDecimator::new(model.os_dec_c_prev1, model.os_dec_c0);

    let (pc_l, pc_r) = (model.pre_conv_u_gain[0], model.pre_conv_u_gain[1]);
    let (pc_hp_l, pc_hp_r) = (model.pre_conv_hp_gain[0], model.pre_conv_hp_gain[1]);
    let (dir_l, dir_r) = (model.direct_s2_u_gain[0], model.direct_s2_u_gain[1]);
    let (dir_hp_l, dir_hp_r) = (model.direct_s2_hp_gain[0], model.direct_s2_hp_gain[1]);

    for t in 0..n {
        let mut tops = [None, None, None, None];
        let mut v_reads = [[0.0_f64; 4]; 4];
        for q_opt in 0..active_opt_quads {
            if let Some(ref st) = opt_fwd[q_opt] {
                let top = st.step_top_of_frame_read();
                v_reads[q_opt] = top.v_read;
                tops[q_opt] = Some(top);
            }
        }
        let (t_cb_l, t_cb_r) = step_opt_crossbar_taps_and_adjoint(
            &v_reads,
            &opt_to_lr_san,
            op_t_gain,
            active_opt_quads,
        );
        let s0l_t = u_sense_l[t] + t_cb_l;
        let s0r_t = u_sense_r[t] + t_cb_r;

        let (uxl_t, h_l, gclk_l, dg_l) =
            op_n_resolved.step_forward_frame(t, s0l_t, &mut op_n_prev_l);
        let (uxr_t, h_r, gclk_r, dg_r) =
            op_n_resolved.step_forward_frame(t, s0r_t, &mut op_n_prev_r);
        ux_l.push(uxl_t);
        ux_r.push(uxr_t);
        if record_trace {
            s0_l.push(s0l_t);
            s0_r.push(s0r_t);
            op_n_held_l.push(h_l);
            op_n_held_r.push(h_r);
            op_n_gclk_l.push(gclk_l);
            op_n_dg_l.push(dg_l);
            op_n_gclk_r.push(gclk_r);
            op_n_dg_r.push(dg_r);
            v_reads_hist.push(v_reads);
        }

        let mut step_core_subframe = |sub_t: usize, cuxl: f64, cuxr: f64| -> [f64; 4] {
            if record_trace && os2x {
                core_ux_l.push(cuxl);
                core_ux_r.push(cuxr);
            }
            let (hpl_t, lpl_t) = ro_resolved.step_forward_frame(cuxl, &mut ro_prev_l);
            let (hpr_t, lpr_t) = ro_resolved.step_forward_frame(cuxr, &mut ro_prev_r);
            if record_trace {
                ro_lp_l.push(lpl_t);
                ro_lp_r.push(lpr_t);
                if sub_t.is_multiple_of(TILE_FRAMES) {
                    let tile = sub_t / TILE_FRAMES;
                    for q in 0..q_count {
                        write_checkpoint(
                            &mut tile_checkpoints,
                            q,
                            num_tiles,
                            tile,
                            &quad_states[q],
                        );
                    }
                }
            }

            let ul_s = sanitize_unit_f64(cuxl, -1.0e6, 1.0e6, 0.0);
            let ur_s = sanitize_unit_f64(cuxr, -1.0e6, 1.0e6, 0.0);
            let mid = 0.5 * (ul_s + ur_s);
            let side = 0.5 * (ul_s - ur_s);

            let mut sub_bus_l = pc_l * cuxl + pc_hp_l * hpl_t;
            let mut sub_bus_r = pc_r * cuxr + pc_hp_r * hpr_t;
            let mut sub_tank_l = dir_l * cuxl + dir_hp_l * hpl_t;
            let mut sub_tank_r = dir_r * cuxr + dir_hp_r * hpr_t;

            for &q in &SONA_ULTRA_QUAD_TRAVERSAL_ORDER {
                if q >= q_count {
                    continue;
                }
                let cb_idx = q * n_core + sub_t;
                let mut acc_l = [0.0_f64; 4];
                let mut acc_r = [0.0_f64; 4];
                for d in 0..4 {
                    acc_l[d] =
                        in_mids[q][d] * mid + in_sides[q][d] * side + cb_trace.cb_add_l[cb_idx][d];
                    acc_r[d] =
                        in_mids[q][d] * mid - in_sides[q][d] * side + cb_trace.cb_add_r[cb_idx][d];
                }
                for inc in &incoming_per_q[q] {
                    let fbase = (inc.src_q * n_core + sub_t) * FEATURE_VEC4_STRIDE;
                    let vl = feature_outputs[fbase];
                    let vr = feature_outputs[fbase + 1];
                    let ol = feature_outputs[fbase + 2];
                    let or_ = feature_outputs[fbase + 3];
                    let el = feature_outputs[fbase + 4];
                    let er = feature_outputs[fbase + 5];
                    for s in 0..4 {
                        let (ol_s, or_s) = (f64::from(ol[s]), f64::from(or_[s]));
                        let (el_s, er_s) = (f64::from(el[s]), f64::from(er[s]));
                        let (vl_s, vr_s) = (f64::from(vl[s]), f64::from(vr[s]));
                        for d in 0..4 {
                            acc_l[d] +=
                                inc.wo[s][d] * ol_s + inc.we[s][d] * el_s + inc.wl[s][d] * vl_s;
                            acc_r[d] +=
                                inc.wo[s][d] * or_s + inc.we[s][d] * er_s + inc.wl[s][d] * vr_s;
                        }
                    }
                }
                let in_l_f32 = acc_l.map(|v| v as f32);
                let in_r_f32 = acc_r.map(|v| v as f32);
                if record_trace {
                    let in_idx = (q * n_core + sub_t) * 2;
                    core_inputs[in_idx] = in_l_f32;
                    core_inputs[in_idx + 1] = in_r_f32;
                }
                let u_vec = InterleavedStereo8::from_lr(
                    GpuVec4::from_array(in_l_f32.map(flush_zero)),
                    GpuVec4::from_array(in_r_f32.map(flush_zero)),
                );
                let is_phs = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 == 0;
                let out = if is_phs {
                    step_core_quad_phs_stereo8(u_vec, &mut quad_states[q], &kparams[q])
                } else {
                    step_core_quad_stereo8(u_vec, &mut quad_states[q], &kparams[q])
                };
                let (vl, vr) = out.lru_state.into_lr();
                let (ol, or_) = out.adaa_odd.into_lr();
                let (el, er) = out.adaa_even.into_lr();
                let (vl_a, vr_a) = (canon_vec4(vl), canon_vec4(vr));
                let (ol_a, or_a) = (canon_vec4(ol), canon_vec4(or_));
                let (el_a, er_a) = (canon_vec4(el), canon_vec4(er));
                let fbase = (q * n_core + sub_t) * FEATURE_VEC4_STRIDE;
                feature_outputs[fbase] = vl_a;
                feature_outputs[fbase + 1] = vr_a;
                feature_outputs[fbase + 2] = ol_a;
                feature_outputs[fbase + 3] = or_a;
                feature_outputs[fbase + 4] = el_a;
                feature_outputs[fbase + 5] = er_a;
                for k in 0..4 {
                    let r = &model.w_ro[q * 4 + k];
                    sub_bus_l += r[0] * f64::from(ol_a[k])
                        + r[2] * f64::from(el_a[k])
                        + r[4] * f64::from(vl_a[k]);
                    sub_bus_r += r[1] * f64::from(or_a[k])
                        + r[3] * f64::from(er_a[k])
                        + r[5] * f64::from(vr_a[k]);
                    sub_tank_l += r[6] * f64::from(ol_a[k])
                        + r[8] * f64::from(el_a[k])
                        + r[10] * f64::from(vl_a[k]);
                    sub_tank_r += r[7] * f64::from(or_a[k])
                        + r[9] * f64::from(er_a[k])
                        + r[11] * f64::from(vr_a[k]);
                }
            }
            [sub_bus_l, sub_bus_r, sub_tank_l, sub_tank_r]
        };

        let [bus_l, bus_r, mut tank_l, mut tank_r] = if os2x {
            let (u0, u1) = os_up.step(uxl_t, uxr_t);
            let y0 = step_core_subframe(2 * t, u0[0], u0[1]);
            let y1 = step_core_subframe(2 * t + 1, u1[0], u1[1]);
            os_dec.step(y0, y1)
        } else {
            step_core_subframe(t, uxl_t, uxr_t)
        };

        pre_conv_l[t] = bus_l;
        pre_conv_r[t] = bus_r;
        for k in 0..active_fir.min(t + 1) {
            tank_l += model.fir_l[k] * pre_conv_l[t - k];
            tank_r += model.fir_r[k] * pre_conv_r[t - k];
        }
        let on_l = if t >= 16 { pre_conv_l[t - 16] } else { 0.0 };
        let on_r = if t >= 16 { pre_conv_r[t - 16] } else { 0.0 };
        onset_pre_l[t] = on_l;
        onset_pre_r[t] = on_r;
        tank_l += kautz_l.step_forward_frame(on_l, record_trace);
        tank_r += kautz_r.step_forward_frame(on_r, record_trace);

        x_tank_l[t] = tank_l;
        x_tank_r[t] = tank_r;
        let (mut wet_l, mut wet_r) = (0.0_f64, 0.0_f64);
        for q_opt in 0..active_opt_quads {
            if let (Some(st), Some(top)) = (opt_fwd[q_opt].as_mut(), tops[q_opt].as_ref()) {
                let (ql, qr) = st.step_end_of_frame_write(top, tank_l, tank_r);
                wet_l += ql;
                wet_r += qr;
            }
        }
        if record_trace {
            op_t_wet_l[t] = wet_l;
            op_t_wet_r[t] = wet_r;
        }
        pred_l[t] = dry_mix * tank_l + wet_scale * wet_l;
        pred_r[t] = dry_mix * tank_r + wet_scale * wet_r;
    }

    CausalLoopBuffers {
        q_count,
        os2x,
        active_opt_quads,
        active_k,
        active_fir,
        op_t_gain,
        op_t_gain_open,
        dry_mix,
        dry_mix_open,
        wet_scale,
        wet_scale_open,
        rho_core,
        pred_l,
        pred_r,
        op_t_wet_l,
        op_t_wet_r,
        s0_l,
        s0_r,
        ux_l,
        ux_r,
        core_ux_l,
        core_ux_r,
        onset_pre_l,
        onset_pre_r,
        x_tank_l,
        x_tank_r,
        cb_trace,
        core_inputs,
        tile_checkpoints,
        feature_outputs,
        op_n_resolved,
        op_n_held_l,
        op_n_held_r,
        op_n_gclk_l,
        op_n_dg_l,
        op_n_gclk_r,
        op_n_dg_r,
        ro_resolved,
        ro_lp_l,
        ro_lp_r,
        kautz_l,
        kautz_r,
        opt_active,
        opt_fwd,
        v_reads_hist,
    }
}
