//! Periodic exact host `binary64` `uplo='U'` VarPro synchronization of strictly linear-in-output
//! readout weights during Pass-2 trajectory optimization, enforcing §2.2 Op-T separability and
//! strict monotone composite-loss non-regression (`ref[train.pipeline.backbone-trajectory-opt]`).

mod basis;

use crate::error::TrainError;
use crate::gpu::abi::FEATURE_VEC4_STRIDE;
use crate::losses::{CompositeLossEvaluator, CompositeLossWeights};
use crate::runner::TrainRunner;

use super::super::model::cascade::{downsample_lane_2x_f64, sanitize_unit_f64};
use super::super::model::{NUM_FIR_TAPS_ULTRA, SonaModelF64, StereoTrainingPairs};
use super::causal_loop::is_op_t_crossbar_coupled;
use super::eval::core_adj::active_opt_quads_for_core_quads;
use super::eval::{Pass2EvalBuffers, eval_model_pass2_audio};
use super::op_t_quad::has_wet_op_t_output;
use super::theta::{TIER_FIR_TAPS, TIER_KAUTZ, TIER_QUADS};
use basis::{
    ChannelReadoutSol, append_fir_cols, append_kautz_cols, apply_readout_solution,
    extract_kautz_basis_modes, solve_ls_column_system, unpack_core_and_fir_sol,
};

/// Shared immutable context for periodic Pass-2 VarPro readout refreshes.
pub(super) struct VarProSyncCtx<'a> {
    pub runner: &'a TrainRunner,
    pub pairs: StereoTrainingPairs<'a>,
    pub loss_eval: &'a CompositeLossEvaluator,
    pub weights: CompositeLossWeights,
    pub warmup: usize,
}

/// Builds a tier-scoped view of `model`, capping active FIR/Kautz counts to `tier_idx`,
/// zeroing `phasor_stereo_phase` at `Lite`, and overlaying any Pass-2/3 linear readout overrides.
#[must_use]
pub(super) fn materialize_tier_view(model: &SonaModelF64, tier_idx: usize) -> SonaModelF64 {
    let ti = tier_idx.min(3);
    let q_count = TIER_QUADS[ti];
    let mut out = model.clone();
    out.active_fir_taps = model.active_fir_taps.min(TIER_FIR_TAPS[ti]);
    out.active_kautz_pairs = model.active_kautz_pairs.min(TIER_KAUTZ[ti]);
    if ti == 0 {
        out.phasor_stereo_phase = [0.0; 4];
    }
    if let Some(ref ovr) = model.tier_overrides[ti] {
        out.direct_s2_u_gain = ovr.direct_s2_u_gain;
        out.direct_s2_hp_gain = ovr.direct_s2_hp_gain;
        for row in 0..(q_count * 4) {
            out.w_ro[row][6..12].copy_from_slice(&ovr.w_ro[row][6..12]);
        }
        let f_cap = out.active_fir_taps.min(NUM_FIR_TAPS_ULTRA);
        out.fir_l[..f_cap].copy_from_slice(&ovr.fir_l[..f_cap]);
        out.fir_r[..f_cap].copy_from_slice(&ovr.fir_r[..f_cap]);
        let k_cap = out.active_kautz_pairs.min(out.kautz.len());
        for k in 0..k_cap {
            if out.kautz[k].r_low > 0.0 && out.kautz[k].b_gamma > 0.0 {
                out.kautz[k].c_re_l = ovr.kautz_c_re_l[k];
                out.kautz[k].c_im_l = ovr.kautz_c_im_l[k];
                out.kautz[k].c_re_r = ovr.kautz_c_re_r[k];
                out.kautz[k].c_im_r = ovr.kautz_c_im_r[k];
            } else {
                out.kautz[k].c_re_l = 0.0;
                out.kautz[k].c_im_l = 0.0;
                out.kautz[k].c_re_r = 0.0;
                out.kautz[k].c_im_r = 0.0;
            }
        }
    }
    out
}

/// Onset delay in 48 kHz samples preceding `ModalConvolver` Kautz branches
/// (`CONVOLVER_KAUTZ_ONSET_DELAY_48K`), used to decouple Kautz modal residues from `FIR[16..]`.
const KAUTZ_ONSET_FIR_TAPS: usize = 16;
const _: () = assert!(KAUTZ_ONSET_FIR_TAPS == 16);

fn reconstruct_pre_conv_stereo(
    view: &SonaModelF64,
    buf: &Pass2EvalBuffers,
    q_count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (c_ux_l, c_ux_r): (&[f64], &[f64]) = if buf.os2x {
        (&buf.core_ux_l, &buf.core_ux_r)
    } else {
        (&buf.ux_l, &buf.ux_r)
    };
    let n_core = c_ux_l.len();
    let (pc_l, pc_r) = (view.pre_conv_u_gain[0], view.pre_conv_u_gain[1]);
    let (pc_hp_l, pc_hp_r) = (view.pre_conv_hp_gain[0], view.pre_conv_hp_gain[1]);
    let mut core_pre_l = vec![0.0_f64; n_core];
    let mut core_pre_r = vec![0.0_f64; n_core];

    for t in 0..n_core {
        let mut bl = pc_l * c_ux_l[t] + pc_hp_l * buf.core_ro_hp_l[t];
        let mut br = pc_r * c_ux_r[t] + pc_hp_r * buf.core_ro_hp_r[t];
        for q in 0..q_count {
            let fbase = (q * n_core + t) * FEATURE_VEC4_STRIDE;
            let v_eff_l = buf.cascade.feature_outputs[fbase];
            let v_eff_r = buf.cascade.feature_outputs[fbase + 1];
            let odd_l = buf.cascade.feature_outputs[fbase + 2];
            let odd_r = buf.cascade.feature_outputs[fbase + 3];
            let even_l = buf.cascade.feature_outputs[fbase + 4];
            let even_r = buf.cascade.feature_outputs[fbase + 5];
            for k in 0..4 {
                let r = &view.w_ro[q * 4 + k];
                bl += r[0] * f64::from(odd_l[k])
                    + r[2] * f64::from(even_l[k])
                    + r[4] * f64::from(v_eff_l[k]);
                br += r[1] * f64::from(odd_r[k])
                    + r[3] * f64::from(even_r[k])
                    + r[5] * f64::from(v_eff_r[k]);
            }
        }
        core_pre_l[t] = bl;
        core_pre_r[t] = br;
    }

    if buf.os2x {
        (
            downsample_lane_2x_f64(&core_pre_l, view.os_dec_c_prev1, view.os_dec_c0),
            downsample_lane_2x_f64(&core_pre_r, view.os_dec_c_prev1, view.os_dec_c0),
        )
    } else {
        (core_pre_l, core_pre_r)
    }
}

fn solve_channel_readouts(
    ctx: &VarProSyncCtx<'_>,
    view: &SonaModelF64,
    buf: &Pass2EvalBuffers,
    pre_conv_ch: &[f64],
    is_right: bool,
    tier_idx: usize,
) -> Result<ChannelReadoutSol, TrainError> {
    let ti = tier_idx.min(3);
    let q_count = TIER_QUADS[ti];
    let n = ctx.pairs.aligned_len();
    let w = ctx.warmup.min(n.saturating_sub(1));
    let m = n - w;
    let ch = usize::from(is_right);
    let dry_mix = sanitize_unit_f64(view.tank_dry_mix, -2.0, 2.0, 1.0);
    let n_core = if buf.os2x {
        buf.core_ux_l.len()
    } else {
        buf.ux_l.len()
    };

    let active_fir = view
        .active_fir_taps
        .min(TIER_FIR_TAPS[ti])
        .min(NUM_FIR_TAPS_ULTRA);
    let active_k_cap = view
        .active_kautz_pairs
        .min(TIER_KAUTZ[ti])
        .min(view.kautz.len());
    let active_k_indices: Vec<usize> = (0..active_k_cap)
        .filter(|&k| view.kautz[k].r_low > 0.0 && view.kautz[k].b_gamma > 0.0)
        .collect();

    let n_rows = q_count * 4;
    let base_dim = 2 + 3 * n_rows;
    let max_extra =
        active_fir.max(KAUTZ_ONSET_FIR_TAPS.min(active_fir) + 2 * active_k_indices.len());
    let mut base_feats = Vec::with_capacity((base_dim + max_extra) * m);
    let mut base_anchors = Vec::with_capacity(base_dim + max_extra);

    let to_1x = |core_sig: &[f64]| -> Vec<f64> {
        if buf.os2x {
            downsample_lane_2x_f64(core_sig, view.os_dec_c_prev1, view.os_dec_c0)
        } else {
            core_sig.to_vec()
        }
    };
    let mut push_base_col = |sig_1x: &[f64], scale: f64, old_val: f64| {
        let col_idx = base_anchors.len();
        for &v in &sig_1x[w..n] {
            base_feats.push((scale * v) as f32);
        }
        base_anchors.push((col_idx, old_val));
    };

    let c_ux: &[f64] = match (is_right, buf.os2x) {
        (false, true) => &buf.core_ux_l,
        (false, false) => &buf.ux_l,
        (true, true) => &buf.core_ux_r,
        (true, false) => &buf.ux_r,
    };
    let c_hp: &[f64] = if is_right {
        &buf.core_ro_hp_r
    } else {
        &buf.core_ro_hp_l
    };

    let ux_1x = to_1x(c_ux);
    let hp_1x = to_1x(c_hp);
    push_base_col(&ux_1x, dry_mix, view.direct_s2_u_gain[ch]);
    push_base_col(&hp_1x, dry_mix, view.direct_s2_hp_gain[ch]);

    let mut lane_core = vec![0.0_f64; n_core];
    for row in 0..n_rows {
        let q = row / 4;
        let lane = row % 4;
        for (slot_pair, old_col) in [(2_usize, 6_usize), (4, 8), (0, 10)] {
            let slot = slot_pair + ch;
            for t in 0..n_core {
                let fbase = (q * n_core + t) * FEATURE_VEC4_STRIDE;
                lane_core[t] = f64::from(buf.cascade.feature_outputs[fbase + slot][lane]);
            }
            let lane_1x = to_1x(&lane_core);
            push_base_col(&lane_1x, dry_mix, view.w_ro[row][old_col + ch]);
        }
    }

    let existing_fir = if is_right { &view.fir_r } else { &view.fir_l };
    let tgt_slice = if is_right {
        &ctx.pairs.target_r[w..n]
    } else {
        &ctx.pairs.target_l[w..n]
    };
    let tgts_f32: Vec<f32> = tgt_slice.iter().map(|&v| v as f32).collect();

    if active_k_indices.is_empty() || active_fir <= KAUTZ_ONSET_FIR_TAPS {
        let feat_dim = base_dim + active_fir + 2 * active_k_indices.len();
        append_fir_cols(
            &mut base_feats,
            &mut base_anchors,
            pre_conv_ch,
            existing_fir,
            active_fir,
            (w, n, dry_mix),
        );
        if !active_k_indices.is_empty() {
            let onset = if is_right {
                &buf.onset_pre_r[..n]
            } else {
                &buf.onset_pre_l[..n]
            };
            let modes =
                extract_kautz_basis_modes(view, onset, active_k_cap, &active_k_indices, is_right);
            append_kautz_cols(
                &mut base_feats,
                &mut base_anchors,
                view,
                &modes,
                &active_k_indices,
                (w, n, dry_mix, is_right),
            );
        }
        let sol = solve_ls_column_system(
            ctx.runner,
            &base_feats,
            &tgts_f32,
            &base_anchors,
            m,
            feat_dim,
        )?;
        let (direct_s2_u, direct_s2_hp, w_ro_6_to_11, fir) =
            unpack_core_and_fir_sol(&sol, n_rows, active_fir);
        let k_off = base_dim + active_fir;
        let mut kautz_c_re = Vec::with_capacity(active_k_indices.len());
        let mut kautz_c_im = Vec::with_capacity(active_k_indices.len());
        for pos in 0..active_k_indices.len() {
            kautz_c_re.push(sanitize_unit_f64(sol[k_off + 2 * pos], -8.0, 8.0, 0.0));
            kautz_c_im.push(sanitize_unit_f64(sol[k_off + 2 * pos + 1], -8.0, 8.0, 0.0));
        }
        return Ok(ChannelReadoutSol {
            direct_s2_u,
            direct_s2_hp,
            w_ro_6_to_11,
            fir,
            kautz_indices: active_k_indices,
            kautz_c_re,
            kautz_c_im,
        });
    }

    // Two-stage onset-decoupled solve: Stage 1 solves onset FIR[0..16] + delayed Kautz so
    // FIR[16..] cannot absorb modal pole energy; Stage 2 solves FIR[0..active_fir] on the residual.
    let onset = if is_right {
        &buf.onset_pre_r[..n]
    } else {
        &buf.onset_pre_l[..n]
    };
    let modes = extract_kautz_basis_modes(view, onset, active_k_cap, &active_k_indices, is_right);

    let mut stage1_feats = base_feats.clone();
    let mut stage1_anchors = base_anchors.clone();
    let s1_fir = KAUTZ_ONSET_FIR_TAPS;
    append_fir_cols(
        &mut stage1_feats,
        &mut stage1_anchors,
        pre_conv_ch,
        existing_fir,
        s1_fir,
        (w, n, dry_mix),
    );
    append_kautz_cols(
        &mut stage1_feats,
        &mut stage1_anchors,
        view,
        &modes,
        &active_k_indices,
        (w, n, dry_mix, is_right),
    );
    let s1_dim = base_dim + s1_fir + 2 * active_k_indices.len();
    let sol1 = solve_ls_column_system(
        ctx.runner,
        &stage1_feats,
        &tgts_f32,
        &stage1_anchors,
        m,
        s1_dim,
    )?;
    let k_off = base_dim + s1_fir;
    let mut kautz_c_re = Vec::with_capacity(active_k_indices.len());
    let mut kautz_c_im = Vec::with_capacity(active_k_indices.len());
    for pos in 0..active_k_indices.len() {
        kautz_c_re.push(sanitize_unit_f64(sol1[k_off + 2 * pos], -8.0, 8.0, 0.0));
        kautz_c_im.push(sanitize_unit_f64(sol1[k_off + 2 * pos + 1], -8.0, 8.0, 0.0));
    }

    let res_tgts_f32: Vec<f32> = (0..m)
        .map(|i| {
            let t = w + i;
            let mut k_sum = 0.0_f64;
            for pos in 0..active_k_indices.len() {
                k_sum +=
                    2.0 * (kautz_c_re[pos] * modes[pos].0[t] - kautz_c_im[pos] * modes[pos].1[t]);
            }
            (tgt_slice[i] - dry_mix * k_sum) as f32
        })
        .collect();

    append_fir_cols(
        &mut base_feats,
        &mut base_anchors,
        pre_conv_ch,
        existing_fir,
        active_fir,
        (w, n, dry_mix),
    );
    let s2_dim = base_dim + active_fir;
    let sol2 = solve_ls_column_system(
        ctx.runner,
        &base_feats,
        &res_tgts_f32,
        &base_anchors,
        m,
        s2_dim,
    )?;
    let (direct_s2_u, direct_s2_hp, w_ro_6_to_11, fir) =
        unpack_core_and_fir_sol(&sol2, n_rows, active_fir);

    Ok(ChannelReadoutSol {
        direct_s2_u,
        direct_s2_hp,
        w_ro_6_to_11,
        fir,
        kautz_indices: active_k_indices,
        kautz_c_re,
        kautz_c_im,
    })
}

/// Refreshes strictly linear-in-output readouts (`direct_s2`, `w_ro[..][6..12]`, active early
/// `fir`, and active Kautz residues) for tier `tier_idx` using exact host `uplo='U'` Cholesky,
/// enforcing §2.2 Op-T separability and monotone composite-loss non-regression.
pub(super) fn refresh_tier_readouts_varpro(
    ctx: &VarProSyncCtx<'_>,
    model: &mut SonaModelF64,
    tier_idx: usize,
    sync_top_level: bool,
) -> Result<bool, TrainError> {
    let ti = tier_idx.min(3);
    let q_count = TIER_QUADS[ti];
    let active_opt = active_opt_quads_for_core_quads(q_count);
    let n = ctx.pairs.aligned_len();
    let dry_mix = sanitize_unit_f64(model.tank_dry_mix, -2.0, 2.0, 1.0);
    let op_t_wet_or_loop = is_op_t_crossbar_coupled(model, q_count)
        || model.transport_quads[..active_opt]
            .iter()
            .any(|tq| has_wet_op_t_output(tq) || tq.g_loop.abs() > 1.0e-9);
    if op_t_wet_or_loop || n <= ctx.warmup + 4 || dry_mix.abs() < 1.0e-6 {
        return Ok(false);
    }

    let base_view = materialize_tier_view(model, ti);
    let base_buf = eval_model_pass2_audio(
        ctx.runner,
        &base_view,
        &ctx.pairs.in_l[..n],
        &ctx.pairs.in_r[..n],
        q_count,
    )?;
    if base_buf.causal_loop.is_some() {
        return Ok(false);
    }
    let prev_loss = ctx
        .loss_eval
        .eval_and_adjoint_with_weights(&base_buf.pred_l, &base_buf.pred_r, ctx.weights, None, None)?
        .total;
    if !prev_loss.is_finite() {
        return Ok(false);
    }

    let (pre_conv_l, pre_conv_r) = reconstruct_pre_conv_stereo(&base_view, &base_buf, q_count);
    let sol_l = solve_channel_readouts(ctx, &base_view, &base_buf, &pre_conv_l, false, ti)?;
    let sol_r = solve_channel_readouts(ctx, &base_view, &base_buf, &pre_conv_r, true, ti)?;

    for alpha in [1.0_f64, 0.5, 0.25, 0.125] {
        let mut cand_model = model.clone();
        apply_readout_solution(
            &mut cand_model,
            &base_view,
            ti,
            (&sol_l, &sol_r),
            sync_top_level,
            alpha,
        );
        let cand_view = materialize_tier_view(&cand_model, ti);
        let cand_buf = eval_model_pass2_audio(
            ctx.runner,
            &cand_view,
            &ctx.pairs.in_l[..n],
            &ctx.pairs.in_r[..n],
            q_count,
        )?;
        let cand_loss = ctx
            .loss_eval
            .eval_and_adjoint_with_weights(
                &cand_buf.pred_l,
                &cand_buf.pred_r,
                ctx.weights,
                None,
                None,
            )?
            .total;
        if cand_loss.is_finite() && cand_loss <= prev_loss {
            *model = cand_model;
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::CompositeLossConfig;

    #[test]
    fn test_varpro_sync_recovers_1x_2x_linear_fir_kautz_and_gates_op_t() {
        use crate::losses::canonical_ltas_33_centers_hz;

        let runner = TrainRunner::cpu();
        let n = 1_024_usize;
        let warmup = 32_usize;
        let centers = canonical_ltas_33_centers_hz();
        let make_sig = |phase: f64| -> Vec<f64> {
            (0..n)
                .map(|t| {
                    let sec = (t as f64) / 48_000.0;
                    let mut acc = 0.0_f64;
                    for (b, &fc) in centers.iter().enumerate() {
                        acc += 0.03
                            * (core::f64::consts::TAU * fc * sec + phase + 0.41 * (b as f64)).sin();
                    }
                    acc
                })
                .collect()
        };
        let in_l = make_sig(0.0);
        let in_r = make_sig(0.8);

        let mut oracle = SonaModelF64::new(48_000.0);
        oracle.core_quads[0].svf_g = [0.12, 0.28, 0.65, 1.40];
        oracle.core_quads[0].tone_bp = [1.0, 0.6, 0.4, 0.8];
        oracle.core_quads[0].adaa_drive = [1.2, 1.8, 0.8, 2.2];
        oracle.core_quads[4].tone_dir = [0.8, -0.5, 0.4, -0.2];
        oracle.direct_s2_u_gain = [0.45, -0.35];
        oracle.w_ro[0][6] = 0.28;
        oracle.w_ro[0][7] = -0.22;
        oracle.pre_conv_u_gain = [0.6, 0.5];
        oracle.active_fir_taps = 4;
        oracle.fir_l[..4].copy_from_slice(&[0.35, -0.18, 0.11, -0.04]);
        oracle.fir_r[..4].copy_from_slice(&[-0.25, 0.20, -0.09, 0.05]);
        oracle.active_kautz_pairs = 1;
        oracle.kautz[0].r_low = 0.85;
        oracle.kautz[0].theta = 0.55;
        oracle.kautz[0].b_gamma = 0.8;
        oracle.kautz[0].c_re_l = 0.25;
        oracle.kautz[0].c_im_l = -0.15;
        oracle.kautz[0].c_re_r = -0.18;
        oracle.kautz[0].c_im_r = 0.22;

        for (tier_idx, q_count) in [(0_usize, 4_usize), (1_usize, 6_usize)] {
            let tgt_buf = eval_model_pass2_audio(&runner, &oracle, &in_l, &in_r, q_count).unwrap();
            let loss_eval = CompositeLossEvaluator::new_for_homotopy(
                &tgt_buf.pred_l,
                &tgt_buf.pred_r,
                None,
                None,
                CompositeLossConfig {
                    sample_rate_hz: 48_000.0,
                    warmup_frames: warmup,
                    ..CompositeLossConfig::default()
                },
            )
            .unwrap();
            let ctx = VarProSyncCtx {
                runner: &runner,
                pairs: StereoTrainingPairs {
                    in_l: &in_l,
                    in_r: &in_r,
                    target_l: &tgt_buf.pred_l,
                    target_r: &tgt_buf.pred_r,
                },
                loss_eval: &loss_eval,
                weights: CompositeLossWeights::homotopy_at(1.0),
                warmup,
            };

            let mut fitted = oracle.clone();
            fitted.direct_s2_u_gain = [0.0; 2];
            fitted.w_ro = [[0.0; 12]; 40];
            fitted.fir_l = [0.0; NUM_FIR_TAPS_ULTRA];
            fitted.fir_r = [0.0; NUM_FIR_TAPS_ULTRA];
            fitted.kautz[0].c_re_l = 0.0;
            fitted.kautz[0].c_im_l = 0.0;
            fitted.kautz[0].c_re_r = 0.0;
            fitted.kautz[0].c_im_r = 0.0;

            let accepted = refresh_tier_readouts_varpro(&ctx, &mut fitted, tier_idx, true).unwrap();
            assert!(accepted, "tier {tier_idx} VarPro refresh should accept");
            let post_buf = eval_model_pass2_audio(&runner, &fitted, &in_l, &in_r, q_count).unwrap();
            let post_bd = loss_eval
                .eval_and_adjoint_with_weights(
                    &post_buf.pred_l,
                    &post_buf.pred_r,
                    ctx.weights,
                    None,
                    None,
                )
                .unwrap();
            let l2_num: f64 = post_buf.pred_l[warmup..]
                .iter()
                .zip(&tgt_buf.pred_l[warmup..])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let l2_den: f64 = tgt_buf.pred_l[warmup..]
                .iter()
                .map(|b| b * b)
                .sum::<f64>()
                .max(1.0e-12);
            assert!(
                post_bd.total < 1.0e-3 && (l2_num / l2_den) < 5.0e-6,
                "tier {tier_idx} VarPro failed: total={} besr={} (esr_l={:.3e}) mr_iir={} spec_ot={} gabor={} ltas={} if_gd={} seg_esr={} nyq={} | c_re_l={} (ora={}) c_im_l={} (ora={})",
                post_bd.total,
                post_bd.besr,
                l2_num / l2_den,
                post_bd.mr_iir,
                post_bd.spec_ot,
                post_bd.gabor,
                post_bd.ltas,
                post_bd.if_gd,
                post_bd.seg_esr,
                post_bd.nyquist,
                fitted.kautz[0].c_re_l,
                oracle.kautz[0].c_re_l,
                fitted.kautz[0].c_im_l,
                oracle.kautz[0].c_im_l,
            );
            assert!(
                (fitted.kautz[0].c_im_l - oracle.kautz[0].c_im_l).abs() < 1.0e-2
                    && (fitted.kautz[0].c_re_r - oracle.kautz[0].c_re_r).abs() < 1.0e-2
            );

            // Running again at exact optimum preserves monotone non-regression.
            let second = refresh_tier_readouts_varpro(&ctx, &mut fitted, tier_idx, true).unwrap();
            let second_buf =
                eval_model_pass2_audio(&runner, &fitted, &in_l, &in_r, q_count).unwrap();
            let second_loss = loss_eval
                .eval_and_adjoint_with_weights(
                    &second_buf.pred_l,
                    &second_buf.pred_r,
                    ctx.weights,
                    None,
                    None,
                )
                .unwrap()
                .total;
            assert!(second_loss <= post_bd.total + 1.0e-12 || !second);
        }

        // §2.2 Op-T separability gate must return Ok(false) cleanly without mutating model.
        let mut coupled = oracle.clone();
        coupled.transport_quads[0].g_loop = 0.4;
        coupled.transport_quads[0].wet_tap_l = [0.5, 0.0, 0.0, 0.0];
        let tgt_c = eval_model_pass2_audio(&runner, &coupled, &in_l, &in_r, 4).unwrap();
        let loss_c = CompositeLossEvaluator::new_for_homotopy(
            &tgt_c.pred_l,
            &tgt_c.pred_r,
            None,
            None,
            CompositeLossConfig {
                sample_rate_hz: 48_000.0,
                warmup_frames: warmup,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let ctx_c = VarProSyncCtx {
            runner: &runner,
            pairs: StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_c.pred_l,
                target_r: &tgt_c.pred_r,
            },
            loss_eval: &loss_c,
            weights: CompositeLossWeights::homotopy_at(1.0),
            warmup,
        };
        assert!(!refresh_tier_readouts_varpro(&ctx_c, &mut coupled, 0, true).unwrap());
    }

    #[test]
    fn test_varpro_sync_two_stage_preserves_8_kautz_modes_with_64_fir_taps() {
        use crate::losses::canonical_ltas_33_centers_hz;

        let runner = TrainRunner::cpu();
        let n = 1_536_usize;
        let warmup = 64_usize;
        let centers = canonical_ltas_33_centers_hz();
        let make_sig = |phase: f64| -> Vec<f64> {
            let n_f = n as f64;
            (0..n)
                .map(|t| {
                    let tf = t as f64;
                    let sec = tf / 48_000.0;
                    let tones: f64 = centers
                        .iter()
                        .enumerate()
                        .map(|(b, &fc)| {
                            0.018
                                * (core::f64::consts::TAU * fc * sec + phase + 0.31 * (b as f64))
                                    .sin()
                        })
                        .sum();
                    let prn = ((tf * 12.989_8 + phase * 78.233).sin() * 43_758.545_3).fract() - 0.5;
                    let chirp = 0.08
                        * (core::f64::consts::TAU * (40.0 * sec + 9_500.0 * sec * tf / n_f)
                            + phase)
                            .sin();
                    tones + 0.14 * prn + chirp
                })
                .collect()
        };
        let in_l = make_sig(0.0);
        let in_r = make_sig(0.9);

        const MODES: [(f64, f64, f64); 8] = [
            (0.945, 0.08, 0.18),
            (0.938, 0.19, -0.16),
            (0.930, 0.34, 0.15),
            (0.922, 0.55, -0.14),
            (0.912, 0.84, 0.13),
            (0.902, 1.22, -0.12),
            (0.892, 1.72, 0.11),
            (0.882, 2.32, -0.10),
        ];
        let mut oracle = SonaModelF64::new(48_000.0);
        oracle.pre_conv_u_gain = [1.0, 1.0];
        oracle.direct_s2_u_gain = [0.35, 0.35];
        oracle.active_fir_taps = 64;
        oracle.fir_l[0] = 0.25;
        oracle.fir_l[4] = -0.08;
        oracle.fir_r[0] = 0.25;
        oracle.fir_r[4] = -0.08;
        oracle.active_kautz_pairs = 8;
        for (k, &(r, th, c_re)) in MODES.iter().enumerate() {
            oracle.kautz[k].r_low = r;
            oracle.kautz[k].theta = th;
            oracle.kautz[k].b_gamma = (1.0 - r * r).sqrt();
            oracle.kautz[k].c_re_l = c_re;
            oracle.kautz[k].c_re_r = c_re;
        }
        let tgt_buf = eval_model_pass2_audio(&runner, &oracle, &in_l, &in_r, 6).unwrap();
        let loss_eval = CompositeLossEvaluator::new_for_homotopy(
            &tgt_buf.pred_l,
            &tgt_buf.pred_r,
            None,
            None,
            CompositeLossConfig {
                sample_rate_hz: 48_000.0,
                warmup_frames: warmup,
                ..CompositeLossConfig::default()
            },
        )
        .unwrap();
        let ctx = VarProSyncCtx {
            runner: &runner,
            pairs: StereoTrainingPairs {
                in_l: &in_l,
                in_r: &in_r,
                target_l: &tgt_buf.pred_l,
                target_r: &tgt_buf.pred_r,
            },
            loss_eval: &loss_eval,
            weights: CompositeLossWeights::homotopy_at(1.0),
            warmup,
        };

        let mut fitted = oracle.clone();
        fitted.direct_s2_u_gain = [0.0; 2];
        fitted.fir_l = [0.0; NUM_FIR_TAPS_ULTRA];
        fitted.fir_r = [0.0; NUM_FIR_TAPS_ULTRA];
        for kp in &mut fitted.kautz[..8] {
            kp.c_re_l = 0.0;
            kp.c_im_l = 0.0;
            kp.c_re_r = 0.0;
            kp.c_im_r = 0.0;
        }
        assert!(refresh_tier_readouts_varpro(&ctx, &mut fitted, 1, true).unwrap());
        for k in 0..8 {
            assert!(
                fitted.kautz[k].c_re_l.abs() > 0.05,
                "mode {k} residue collapsed under 64-tap FIR: {}",
                fitted.kautz[k].c_re_l
            );
        }
        let post_buf = eval_model_pass2_audio(&runner, &fitted, &in_l, &in_r, 6).unwrap();
        let num: f64 = post_buf.pred_l[warmup..]
            .iter()
            .zip(&tgt_buf.pred_l[warmup..])
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f64 = tgt_buf.pred_l[warmup..]
            .iter()
            .map(|b| b * b)
            .sum::<f64>()
            .max(1.0e-12);
        assert!(num / den < 5.0e-5, "residual esr too high: {}", num / den);
    }
}
