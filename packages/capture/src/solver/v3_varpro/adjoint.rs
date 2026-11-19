#![forbid(unsafe_code)]
//! Single-Forward / Single-Reverse 4-Tier Matryoshka Loss & Exact BPTT Adjoint (`v3` Core).
//!
//! Projects shared core trajectories to active nested readouts (`Lite..Ultra`), accumulates
//! Bézout-3 + Gabor-33 + Welch-33 LTAS gradients, and runs one exact reverse sweep.

use dnam_engine::stages::stage2_gru::{MAX_GROUPS, NUM_GRU_BLOCKS, tier_groups, tier_kautz_banks};

use crate::solver::backends::{DynamicTrainer, TrainerBackend, TrainerTask};

mod backward;
mod rescale_1x;
mod tier_pass;
use super::basis::active_block_lanes;
pub use super::gru_core::AdjointWorkspace;
use super::gru_core::{forward_shared_core, max_lanes_for_mask};
use super::model_f64::{V3MatryoshkaOptF64, V3TrainDataset};
use super::primitives::{ClampedCoreParams, par_map_indexed};
use backward::backward_shared_core;
use rescale_1x::pullback_grad_1x_to_2x;
pub(in crate::solver::v3_varpro) use rescale_1x::{
    calibrate_1x_tiers_with_backend, rescale_model_2x_to_1x_f64,
};
use tier_pass::{ALL_TIERS, StemEvalCtx, eval_active_tiers_and_fuse};

#[allow(clippy::too_many_arguments)]
fn eval_single_stem_loss_and_grad<B: TrainerBackend>(
    model: &V3MatryoshkaOptF64,
    cp: &ClampedCoreParams,
    dataset: &V3TrainDataset,
    st_idx: usize,
    tiers_needed: &[usize; 4],
    c_blk: [usize; NUM_GRU_BLOCKS],
    active_ch: usize,
    ch_scale: f64,
    sub_steps: usize,
    ws: &mut AdjointWorkspace,
    mut grad: Option<&mut V3MatryoshkaOptF64>,
) -> f64 {
    let stem = &dataset.stems[st_idx];
    let n = stem.usable_len();
    if n == 0 {
        return 0.0;
    }
    let need_grad = grad.is_some();
    let total_sub = n * sub_steps;
    ws.ensure_capacity(n, sub_steps, active_ch, tiers_needed, need_grad, false);
    forward_shared_core(
        model,
        cp,
        &stem.in_l[..n],
        &stem.in_r[..n],
        n,
        c_blk,
        active_ch,
        need_grad,
        false,
        ws,
    );

    let ctx = StemEvalCtx {
        model,
        cp,
        dataset,
        st_idx,
        n,
        active_ch,
        ch_scale,
    };
    let stem_loss =
        eval_active_tiers_and_fuse::<B>(&ctx, tiers_needed, total_sub, ws, grad.as_deref_mut());
    if let Some(g_mut) = grad {
        backward_shared_core(model, cp, n, c_blk, active_ch, ws, g_mut);
    }
    stem_loss
}

#[derive(Clone, Copy)]
struct RatePassSpec {
    tiers_needed: [usize; 4],
    c_blk: [usize; NUM_GRU_BLOCKS],
    active_ch: usize,
    ch_scale: f64,
}

fn eval_rate_pass<B: TrainerBackend>(
    model: &V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    spec: RatePassSpec,
    ws: &mut AdjointWorkspace,
    mut grad: Option<&mut V3MatryoshkaOptF64>,
) -> f64 {
    let cp = ClampedCoreParams::from_model(model);
    let sub_steps = if model.os2x { 2 } else { 1 };
    let need_grad = grad.is_some();
    if dataset.stems.len() >= 2 && dataset.stems.first().map_or(0, |s| s.usable_len()) >= 4_096 {
        ws.ensure_stem_pool(dataset.stems.len());
        let pool_ref = &ws.stem_pool[..dataset.stems.len()];
        let per_stem = par_map_indexed(dataset.stems.len(), |st_idx| {
            let mut stem_ws = pool_ref[st_idx]
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut stem_grad = need_grad.then(|| Box::new(model.zero_like()));
            let loss = eval_single_stem_loss_and_grad::<B>(
                model,
                &cp,
                dataset,
                st_idx,
                &spec.tiers_needed,
                spec.c_blk,
                spec.active_ch,
                spec.ch_scale,
                sub_steps,
                &mut stem_ws,
                stem_grad.as_deref_mut(),
            );
            (loss, stem_grad)
        });
        let mut sum_loss = 0.0_f64;
        for (stem_loss, stem_grad) in per_stem {
            sum_loss += stem_loss;
            if let (Some(g_tot), Some(sg)) = (grad.as_deref_mut(), stem_grad) {
                g_tot.add_assign_grad(&sg);
            }
        }
        sum_loss
    } else {
        let mut sum_loss = 0.0_f64;
        for st_idx in 0..dataset.stems.len() {
            sum_loss += eval_single_stem_loss_and_grad::<B>(
                model,
                &cp,
                dataset,
                st_idx,
                &spec.tiers_needed,
                spec.c_blk,
                spec.active_ch,
                spec.ch_scale,
                sub_steps,
                ws,
                grad.as_deref_mut(),
            );
        }
        sum_loss
    }
}

/// Evaluates weighted 4-tier Matryoshka loss and optional exact reverse-mode BPTT gradient via `B`.
pub fn evaluate_matryoshka_loss_and_grad_with_backend<B: TrainerBackend>(
    model: &V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
    mut grad: Option<&mut V3MatryoshkaOptF64>,
) -> f64 {
    if let Some(g) = grad.as_deref_mut() {
        *g = model.zero_like();
    }
    let g_cnt = model.max_active_groups.clamp(1, MAX_GROUPS);
    let c_cap = active_block_lanes(g_cnt);
    let active_ch = if model.mono_out { 1 } else { 2 };
    let ch_scale = if model.mono_out { 1.0 } else { 0.5 };

    let mut tiers_1x = [0usize; 4];
    let mut tiers_2x = [0usize; 4];
    for (tau, &tier) in ALL_TIERS.iter().enumerate() {
        if tier_groups(tier) <= g_cnt && model.active_tier_weights[tau] > 0.0 {
            let k_t = tier_kautz_banks(tier) * 4;
            if model.os2x && model.tier_os2x[tau] {
                tiers_2x[tau] = k_t;
            } else {
                tiers_1x[tau] = k_t;
            }
        }
    }

    let mut total_loss = 0.0_f64;
    if tiers_1x.iter().any(|&k| k > 0) {
        let spec_1x = RatePassSpec {
            tiers_needed: tiers_1x,
            c_blk: max_lanes_for_mask(&tiers_1x, c_cap),
            active_ch,
            ch_scale,
        };
        if model.os2x {
            let m_1x = Box::new(rescale_model_2x_to_1x_f64(model));
            let mut g_1x = grad.as_deref().map(|g| Box::new(g.zero_like()));
            total_loss += eval_rate_pass::<B>(&m_1x, dataset, spec_1x, ws, g_1x.as_deref_mut());
            if let (Some(g_out), Some(ref g1)) = (grad.as_deref_mut(), g_1x) {
                pullback_grad_1x_to_2x(model, g1, g_out);
            }
        } else {
            total_loss += eval_rate_pass::<B>(model, dataset, spec_1x, ws, grad.as_deref_mut());
        }
    }

    if tiers_2x.iter().any(|&k| k > 0) {
        let spec_2x = RatePassSpec {
            tiers_needed: tiers_2x,
            c_blk: max_lanes_for_mask(&tiers_2x, c_cap),
            active_ch,
            ch_scale,
        };
        total_loss += eval_rate_pass::<B>(model, dataset, spec_2x, ws, grad.as_deref_mut());
    }

    if let Some(g_mut) = grad {
        for blk in &mut g_mut.blocks {
            blk.enforce_lower_triangular();
        }
        g_mut.mirror_mono();
    }
    total_loss
}

/// Evaluates scalar 4-tier Matryoshka loss via explicit backend `B`.
#[must_use]
pub fn evaluate_matryoshka_loss_f64_with_backend<B: TrainerBackend>(
    model: &V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
    ws: &mut AdjointWorkspace,
) -> f64 {
    evaluate_matryoshka_loss_and_grad_with_backend::<B>(model, dataset, ws, None)
}

/// Evaluates scalar 4-tier Matryoshka loss dispatching the fastest detected CPU backend.
#[must_use]
pub fn evaluate_matryoshka_loss_f64(model: &V3MatryoshkaOptF64, dataset: &V3TrainDataset) -> f64 {
    struct EvalTask<'a>(&'a V3MatryoshkaOptF64, &'a V3TrainDataset);
    impl TrainerTask<f64> for EvalTask<'_> {
        fn run<B: TrainerBackend>(self) -> f64 {
            let mut ws = AdjointWorkspace::default();
            evaluate_matryoshka_loss_f64_with_backend::<B>(self.0, self.1, &mut ws)
        }
    }
    DynamicTrainer::detect().dispatch(EvalTask(model, dataset))
}

/// Evaluates 4-tier Matryoshka loss and exact gradient via detected CPU backend.
#[must_use]
pub fn evaluate_matryoshka_loss_and_grad(
    model: &V3MatryoshkaOptF64,
    dataset: &V3TrainDataset,
) -> (f64, V3MatryoshkaOptF64) {
    struct GradTask<'a>(&'a V3MatryoshkaOptF64, &'a V3TrainDataset);
    impl TrainerTask<(f64, V3MatryoshkaOptF64)> for GradTask<'_> {
        fn run<B: TrainerBackend>(self) -> (f64, V3MatryoshkaOptF64) {
            let mut ws = AdjointWorkspace::default();
            let mut g = self.0.zero_like();
            let l = evaluate_matryoshka_loss_and_grad_with_backend::<B>(
                self.0,
                self.1,
                &mut ws,
                Some(&mut g),
            );
            (l, g)
        }
    }
    DynamicTrainer::detect().dispatch(GradTask(model, dataset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    use dnam_engine::stages::V3MatryoshkaModel;
    use dnam_engine::stages::stage2_gru::{
        MAX_BLOCK_LANES, MAX_GROUPS_PER_BLOCK, NUM_GRU_BLOCKS, tier_block_lanes_array,
    };

    use super::super::basis::ALL_TIERS;
    use super::super::model_f64::V3TrainStem;

    fn build_fd_test_model(os2x: bool, mono_out: bool) -> V3MatryoshkaOptF64 {
        let mut m =
            V3MatryoshkaOptF64::from_matryoshka_cold(&V3MatryoshkaModel::identity(os2x, mono_out));
        m.max_active_groups = 3;
        m.active_tier_weights = [0.25, 0.35, 0.0, 0.40];
        m.os_up_quad = -0.06;
        m.os_dec_c_prev1 = 0.14;
        m.os_dec_c0 = 0.36;
        m.os_up_pre_emph = 0.12;
        m.os_dec_aa_mix = 0.65;
        m.os_up_aa_mix = 0.68;
        m.os_up_zoh_mix = 0.35;
        m.svf_base_g = 0.14;
        m.svf_sec2_g_ratio = 1.6;
        m.svf_env_oct_sens = 1.25;
        m.svf_env_comp = 0.8;
        m.svf_atk_alpha = 0.72;
        m.svf_rel_alpha = 0.88;
        m.svf_det_lp_pole = 0.62;
        m.svf_det_hp_mix = 0.35;
        m.svf_damp_k1 = 0.75;
        m.svf_damp_k2 = 0.95;
        m.svf_sec2_bp_feed = 0.25;
        m.svf_sec2_hp_feed = -0.18;
        m.svf_stereo_link = 0.40;
        m.svf_g_min = 0.02;
        m.svf_g_max = 0.78;
        m.svf_feat_lp2_mix = 0.42;
        m.svf_stage2_feed_taps = [0.40, 0.35, 0.22, -0.11, 0.28, -0.16, 0.09];
        m.feat_in_hp_beta = 0.08;
        m.feat_fast_atk_beta = 0.18;
        m.feat_fast_rel_beta = 0.02;
        m.feat_pinch_k = 6.5;
        m.feat_slow_atk_beta = 0.04;
        m.feat_slow_rel_beta = 0.005;
        m.feat_slow_det_comp = 4.5;
        m.feat_key_link = 0.35;
        m.feat_det_thresh = 0.08;
        for b in 0..NUM_GRU_BLOCKS {
            for dst in 0..(MAX_GROUPS_PER_BLOCK[b] * 4) {
                m.blocks[b].b_z[dst] = 0.15 * ((b + dst) as f64 * 0.2).sin();
                m.blocks[b].b_r[dst] = 0.4 + 0.1 * (dst as f64);
                m.blocks[b].b_n[dst] = -0.12 + 0.05 * (b as f64);
                m.blocks[b].k_sag[dst] = 0.35 + 0.1 * (b as f64);
                m.blocks[b].k_quad[dst] = if dst % 2 == 0 {
                    0.12 - 0.02 * (b as f64)
                } else {
                    -0.09 + 0.015 * (b as f64)
                };
                for f in 0..5 {
                    m.blocks[b].w_in_z[f][dst] = 0.08 * ((f + dst + 1) as f64 * 0.31).sin();
                    m.blocks[b].w_in_r[f][dst] = 0.07 * ((f + dst + 2) as f64 * 0.27).cos();
                    m.blocks[b].w_in_n[f][dst] = 0.11 * ((f + b + dst) as f64 * 0.41).sin();
                }
                for s in 0..(((dst >> 2) + 1) << 2) {
                    m.blocks[b].w_prev_z[dst][s] = 0.06 * ((dst + s + 1) as f64).cos();
                    m.blocks[b].w_prev_r[dst][s] = 0.05 * ((dst + s + 2) as f64).sin();
                    m.blocks[b].w_prev_n[dst][s] = 0.09 * ((dst + s + 3) as f64).sin();
                    m.blocks[b].w_rec_z[dst][s] = 0.04 * ((dst + s + 4) as f64).cos();
                    m.blocks[b].w_rec_r[dst][s] = 0.05 * ((dst + s + 5) as f64).sin();
                    m.blocks[b].w_rec_n[dst][s] = 0.08 * ((dst + s + 6) as f64).cos();
                }
            }
            m.blocks[b].enforce_lower_triangular();
        }
        for k in 0..8 {
            m.kautz_poles[k] = (0.80 + 0.015 * (k as f64), 0.18 + 0.22 * (k as f64));
        }
        for tau in [0, 1, 3] {
            let c_t = tier_block_lanes_array(ALL_TIERS[tau]);
            let r = &mut m.readouts[tau];
            r.pre_cab_u_gain = [0.65, 0.58];
            r.pre_cab_hp_gain = [0.18, -0.14];
            r.direct_s2_u_gain = [0.12, 0.09];
            r.direct_s2_hp_gain = [-0.07, 0.11];
            for i in 0..7 {
                r.svf_direct_taps_l[i] = 0.04 * ((i + tau + 1) as f64).sin();
                r.svf_direct_taps_r[i] = 0.03 * ((i + tau + 2) as f64).cos();
            }
            for b in 0..NUM_GRU_BLOCKS {
                for i in 0..c_t[b] {
                    let idx = b * MAX_BLOCK_LANES + i;
                    r.pre_cab_w[0][idx] = 0.05 * ((idx + 1) as f64).sin();
                    r.pre_cab_w[1][idx] = 0.04 * ((idx + 2) as f64).cos();
                    r.direct_s2_w[0][idx] = -0.03 * ((idx + 3) as f64).sin();
                    r.direct_s2_w[1][idx] = 0.03 * ((idx + 4) as f64).cos();
                }
            }
            for k in 0..8 {
                r.fir_l[k] = if k == 0 {
                    0.72
                } else {
                    0.04 * (k as f64).sin()
                };
                r.fir_r[k] = if k == 0 {
                    0.68
                } else {
                    0.03 * (k as f64).cos()
                };
            }
            for m_idx in 0..8 {
                r.kautz_c_l[m_idx] = (
                    0.04 * (m_idx as f64 + 1.0).sin(),
                    0.03 * (m_idx as f64).cos(),
                );
                r.kautz_c_r[m_idx] = (
                    0.03 * (m_idx as f64 + 2.0).cos(),
                    -0.02 * (m_idx as f64).sin(),
                );
            }
        }
        m.mirror_mono();
        m
    }

    fn build_fd_dataset(n: usize) -> V3TrainDataset {
        let centers = crate::metrics::canonical_ltas_33_centers_hz();
        let scale = 0.42 / (centers.len() as f64).sqrt();
        let (mut in_l, mut in_r, mut tgt_l, mut tgt_r) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let t = (i as f64) / 48_000.0;
            let env = (1.0 - (-t * 18.0).exp()) * (0.25 + 0.75 * (2.0 * PI * 3.1 * t).sin().abs());
            let (mut sl, mut sr) = (0.0_f64, 0.0_f64);
            for (k, &fc) in centers.iter().enumerate() {
                sl += (2.0 * PI * fc * t + (k as f64) * 0.51).sin();
                sr += (2.0 * PI * fc * t + (k as f64) * 0.89 + 0.3).cos();
            }
            let xl = env * scale * sl;
            let xr = env * scale * sr;
            in_l.push(xl as f32);
            in_r.push(xr as f32);
            tgt_l.push((0.75 * xl + 0.18 * (xl * 2.1).tanh()) as f32);
            tgt_r.push((0.70 * xr + 0.22 * (xr * 1.9).tanh()) as f32);
        }
        V3TrainDataset::new(
            vec![V3TrainStem {
                in_l,
                in_r,
                tgt_l,
                tgt_r,
                warmup: 48,
            }],
            48_000.0,
        )
        .with_spectral_weights(0.18, 0.14, 0.08)
    }

    #[test]
    fn test_v3_matryoshka_adjoint_matches_central_finite_diff() {
        let ds = build_fd_dataset(384);
        let ds_core = ds.with_spectral_weights(0.18, 0.14, 0.0);
        for (os2x, mono) in [(false, true), (true, false)] {
            let mut m = build_fd_test_model(os2x, mono);
            let (_, grad) = evaluate_matryoshka_loss_and_grad(&m, &ds_core);
            let mut check =
                |name: &str,
                 eps: f64,
                 ana: f64,
                 get_set: fn(&mut V3MatryoshkaOptF64) -> &mut f64| {
                    let orig = *get_set(&mut m);
                    *get_set(&mut m) = orig + eps;
                    let lp = evaluate_matryoshka_loss_f64(&m, &ds_core);
                    *get_set(&mut m) = orig - eps;
                    let lm = evaluate_matryoshka_loss_f64(&m, &ds_core);
                    *get_set(&mut m) = orig;
                    let fd = (lp - lm) / (2.0 * eps);
                    let rel = (ana - fd).abs() / ana.abs().max(fd.abs()).max(1.0e-4);
                    assert!(
                        rel < 1.0e-5,
                        "{name} (os2x={os2x},mono={mono}): ana={ana} fd={fd} rel={rel}"
                    );
                };
            check("svf_base_g", 1e-6, grad.svf_base_g, |x| &mut x.svf_base_g);
            check("svf_sec2_g_ratio", 1e-6, grad.svf_sec2_g_ratio, |x| {
                &mut x.svf_sec2_g_ratio
            });
            check("svf_env_oct_sens", 1e-6, grad.svf_env_oct_sens, |x| {
                &mut x.svf_env_oct_sens
            });
            check("svf_env_comp", 5e-6, grad.svf_env_comp, |x| {
                &mut x.svf_env_comp
            });
            check("svf_atk_alpha", 1e-6, grad.svf_atk_alpha, |x| {
                &mut x.svf_atk_alpha
            });
            check("svf_rel_alpha", 1e-6, grad.svf_rel_alpha, |x| {
                &mut x.svf_rel_alpha
            });
            check("svf_det_lp_pole", 1e-6, grad.svf_det_lp_pole, |x| {
                &mut x.svf_det_lp_pole
            });
            check("svf_det_hp_mix", 1e-6, grad.svf_det_hp_mix, |x| {
                &mut x.svf_det_hp_mix
            });
            check("svf_damp_k1", 1e-6, grad.svf_damp_k1, |x| {
                &mut x.svf_damp_k1
            });
            check("svf_damp_k2", 1e-6, grad.svf_damp_k2, |x| {
                &mut x.svf_damp_k2
            });
            check("svf_sec2_bp_feed", 1e-6, grad.svf_sec2_bp_feed, |x| {
                &mut x.svf_sec2_bp_feed
            });
            check("svf_sec2_hp_feed", 1e-6, grad.svf_sec2_hp_feed, |x| {
                &mut x.svf_sec2_hp_feed
            });
            check(
                "svf_stage2_feed[2]",
                1e-6,
                grad.svf_stage2_feed_taps[2],
                |x| &mut x.svf_stage2_feed_taps[2],
            );
            check("feat_in_hp_beta", 1e-6, grad.feat_in_hp_beta, |x| {
                &mut x.feat_in_hp_beta
            });
            check("feat_fast_atk_beta", 1e-6, grad.feat_fast_atk_beta, |x| {
                &mut x.feat_fast_atk_beta
            });
            check("feat_fast_rel_beta", 1e-6, grad.feat_fast_rel_beta, |x| {
                &mut x.feat_fast_rel_beta
            });
            check("feat_pinch_k", 1e-5, grad.feat_pinch_k, |x| {
                &mut x.feat_pinch_k
            });
            check("feat_slow_atk_beta", 1e-6, grad.feat_slow_atk_beta, |x| {
                &mut x.feat_slow_atk_beta
            });
            check("feat_slow_rel_beta", 1e-6, grad.feat_slow_rel_beta, |x| {
                &mut x.feat_slow_rel_beta
            });
            check("feat_slow_det_comp", 1e-5, grad.feat_slow_det_comp, |x| {
                &mut x.feat_slow_det_comp
            });
            check("svf_feat_lp2_mix", 1e-6, grad.svf_feat_lp2_mix, |x| {
                &mut x.svf_feat_lp2_mix
            });
            check("feat_key_link", 1e-6, grad.feat_key_link, |x| {
                &mut x.feat_key_link
            });
            check("feat_det_thresh", 1e-6, grad.feat_det_thresh, |x| {
                &mut x.feat_det_thresh
            });
            check("b0.w_in_n[2][3]", 1e-6, grad.blocks[0].w_in_n[2][3], |x| {
                &mut x.blocks[0].w_in_n[2][3]
            });
            check("b0.k_sag[3]", 1e-6, grad.blocks[0].k_sag[3], |x| {
                &mut x.blocks[0].k_sag[3]
            });
            check("b0.k_quad[3]", 1e-6, grad.blocks[0].k_quad[3], |x| {
                &mut x.blocks[0].k_quad[3]
            });
            check("b0.b_r[2]", 1e-6, grad.blocks[0].b_r[2], |x| {
                &mut x.blocks[0].b_r[2]
            });
            check(
                "b0.w_rec_n[3][1]",
                1e-6,
                grad.blocks[0].w_rec_n[3][1],
                |x| &mut x.blocks[0].w_rec_n[3][1],
            );
            check(
                "b0.w_rec_n[9][2]",
                1e-6,
                grad.blocks[0].w_rec_n[9][2],
                |x| &mut x.blocks[0].w_rec_n[9][2],
            );
            check("b1.w_in_n[2][3]", 1e-6, grad.blocks[1].w_in_n[2][3], |x| {
                &mut x.blocks[1].w_in_n[2][3]
            });
            check("b1.k_sag[4]", 1e-6, grad.blocks[1].k_sag[4], |x| {
                &mut x.blocks[1].k_sag[4]
            });
            check("b1.k_quad[4]", 1e-6, grad.blocks[1].k_quad[4], |x| {
                &mut x.blocks[1].k_quad[4]
            });
            check(
                "b1.w_prev_z[5][2]",
                1e-6,
                grad.blocks[1].w_prev_z[5][2],
                |x| &mut x.blocks[1].w_prev_z[5][2],
            );
            check(
                "b1.w_rec_n[4][1]",
                1e-6,
                grad.blocks[1].w_rec_n[4][1],
                |x| &mut x.blocks[1].w_rec_n[4][1],
            );
            check("b1.b_n[3]", 1e-6, grad.blocks[1].b_n[3], |x| {
                &mut x.blocks[1].b_n[3]
            });
            check("b2.w_in_n[2][3]", 1e-6, grad.blocks[2].w_in_n[2][3], |x| {
                &mut x.blocks[2].w_in_n[2][3]
            });
            check("b2.k_sag[2]", 1e-6, grad.blocks[2].k_sag[2], |x| {
                &mut x.blocks[2].k_sag[2]
            });
            check("b2.k_quad[2]", 1e-6, grad.blocks[2].k_quad[2], |x| {
                &mut x.blocks[2].k_quad[2]
            });
            check(
                "b2.w_rec_n[3][1]",
                1e-6,
                grad.blocks[2].w_rec_n[3][1],
                |x| &mut x.blocks[2].w_rec_n[3][1],
            );
            check("kautz_poles[2].0", 1e-6, grad.kautz_poles[2].0, |x| {
                &mut x.kautz_poles[2].0
            });
            check("kautz_poles[2].1", 1e-6, grad.kautz_poles[2].1, |x| {
                &mut x.kautz_poles[2].1
            });
            check("r1.fir_l[3]", 1e-6, grad.readouts[1].fir_l[3], |x| {
                &mut x.readouts[1].fir_l[3]
            });
            check(
                "r1.kautz_c_l[1].0",
                1e-6,
                grad.readouts[1].kautz_c_l[1].0,
                |x| &mut x.readouts[1].kautz_c_l[1].0,
            );
            if os2x {
                check("os_up_quad", 1e-6, grad.os_up_quad, |x| &mut x.os_up_quad);
                check("os_dec_c_prev1", 1e-6, grad.os_dec_c_prev1, |x| {
                    &mut x.os_dec_c_prev1
                });
                check("os_dec_c0", 1e-6, grad.os_dec_c0, |x| &mut x.os_dec_c0);
                check("os_up_pre_emph", 1e-6, grad.os_up_pre_emph, |x| {
                    &mut x.os_up_pre_emph
                });
                check("os_dec_aa_mix", 1e-6, grad.os_dec_aa_mix, |x| {
                    &mut x.os_dec_aa_mix
                });
                check("os_up_aa_mix", 1e-6, grad.os_up_aa_mix, |x| {
                    &mut x.os_up_aa_mix
                });
                check("os_up_zoh_mix", 1e-6, grad.os_up_zoh_mix, |x| {
                    &mut x.os_up_zoh_mix
                });
                check("svf_stereo_link", 1e-6, grad.svf_stereo_link, |x| {
                    &mut x.svf_stereo_link
                });
                check(
                    "r1.kautz_c_r[1].1",
                    1e-6,
                    grad.readouts[1].kautz_c_r[1].1,
                    |x| &mut x.readouts[1].kautz_c_r[1].1,
                );
            }

            let (_, grad_full) = evaluate_matryoshka_loss_and_grad(&m, &ds);
            let dirs = [1.0_f64, -0.7, 0.5, -0.9];
            let eps_dir = 1.0e-7_f64;
            let eval_dir = |dataset: &V3TrainDataset,
                            g: &V3MatryoshkaOptF64,
                            mod_m: &mut V3MatryoshkaOptF64,
                            h: f64|
             -> (f64, f64) {
                let d_ana = g.svf_base_g * dirs[0]
                    + g.svf_sec2_g_ratio * dirs[1]
                    + g.blocks[0].w_in_n[2][3] * dirs[2]
                    + g.readouts[1].fir_l[3] * dirs[3];
                let (og, or_, ow, of) = (
                    mod_m.svf_base_g,
                    mod_m.svf_sec2_g_ratio,
                    mod_m.blocks[0].w_in_n[2][3],
                    mod_m.readouts[1].fir_l[3],
                );
                mod_m.svf_base_g = og + h * dirs[0];
                mod_m.svf_sec2_g_ratio = or_ + h * dirs[1];
                mod_m.blocks[0].w_in_n[2][3] = ow + h * dirs[2];
                mod_m.readouts[1].fir_l[3] = of + h * dirs[3];
                let lp = evaluate_matryoshka_loss_f64(mod_m, dataset);
                mod_m.svf_base_g = og - h * dirs[0];
                mod_m.svf_sec2_g_ratio = or_ - h * dirs[1];
                mod_m.blocks[0].w_in_n[2][3] = ow - h * dirs[2];
                mod_m.readouts[1].fir_l[3] = of - h * dirs[3];
                let lm = evaluate_matryoshka_loss_f64(mod_m, dataset);
                mod_m.svf_base_g = og;
                mod_m.svf_sec2_g_ratio = or_;
                mod_m.blocks[0].w_in_n[2][3] = ow;
                mod_m.readouts[1].fir_l[3] = of;
                (d_ana, (lp - lm) / (2.0 * h))
            };
            let (c_ana, c_fd) = eval_dir(&ds_core, &grad, &mut m, eps_dir);
            let core_rel = (c_ana - c_fd).abs() / c_ana.abs().max(c_fd.abs()).max(1.0e-8);
            assert!(
                core_rel < 1.0e-6,
                "core directional (os2x={os2x},mono={mono}): ana={c_ana} fd={c_fd} rel={core_rel}"
            );
            let (f_ana, f_fd) = eval_dir(&ds, &grad_full, &mut m, eps_dir);
            let dir_rel = (f_ana - f_fd).abs() / f_ana.abs().max(f_fd.abs()).max(1.0e-8);
            assert!(
                dir_rel < 1.0e-6,
                "full directional (os2x={os2x},mono={mono}): ana={f_ana} fd={f_fd} rel={dir_rel}"
            );
        }
    }
}
