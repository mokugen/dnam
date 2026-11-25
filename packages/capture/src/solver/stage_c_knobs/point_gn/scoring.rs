#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Window scoring, bandwidth classification, persistent-`SonaVm` rendering, and Lite guard
//! helpers for [`super`] (`solver::stage_c_knobs::point_gn::scoring`).

use dnam_engine::schema::DnamError;
use dnam_train::control::{
    ControlKindSpec, ControlSpec, StageTargetSolveConfig, StageTargetSolveInput,
    solve_point_stage_targets_adjoint,
};
use dnam_train::{CompositeLossConfig, CompositeLossWeights, SsBroydenConfig};
use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use super::super::map_stage_err;
use super::{PASS4_ALL_ACTIVE_SLOTS, PointSolveInputs};
use crate::metrics::{evaluate_besr_from_band_energies, try_split_bezout_3band_energies};

const BROADBAND_MIN_BAND_FRAC: f64 = 1.0e-3;
const _: () = assert!(BROADBAND_MIN_BAND_FRAC > 0.0 && BROADBAND_MIN_BAND_FRAC < 0.1);

pub(super) fn render_vm_with_stage_targets(
    vm: &mut SonaVm<'_, DefaultMathBackend>,
    in_l: &[f32],
    in_r: &[f32],
    st32: &[f64; STAGE_TARGET_SLOTS],
) -> Result<(Vec<f64>, Vec<f64>), DnamError> {
    let mut st32_f32 = [0.0_f32; STAGE_TARGET_SLOTS];
    for (dst, &src) in st32_f32[..STAGE_TARGET_SLOTS - 1].iter_mut().zip(st32) {
        *dst = if src.is_finite() { src as f32 } else { 0.0 };
    }
    vm.reset_with_latched_stage_targets(&st32_f32, &[0.0_f32; 8]);
    let n = in_l.len().min(in_r.len());
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    vm.process_block(&in_l[..n], &in_r[..n], &mut out_l, &mut out_r)
        .map_err(map_stage_err)?;
    Ok((
        out_l.into_iter().map(f64::from).collect(),
        out_r.into_iter().map(f64::from).collect(),
    ))
}

#[inline]
pub(super) fn pack_post_warmup(l: &[f64], r: &[f64], warmup: usize, mono_out: bool) -> Vec<f64> {
    if mono_out {
        l[warmup..].to_vec()
    } else {
        let mut v = Vec::with_capacity((l.len() - warmup) * 2);
        v.extend_from_slice(&l[warmup..]);
        v.extend_from_slice(&r[warmup..]);
        v
    }
}

#[inline]
pub(super) fn vec_mse(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let mut sq = 0.0_f64;
    for (&p, &t) in pred[..n].iter().zip(&target[..n]) {
        let d = p - t;
        sq += d * d;
    }
    sq / (n as f64)
}

#[inline]
pub(super) fn vec_mean_square(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let sq: f64 = v.iter().map(|&x| x * x).sum();
    sq / (v.len() as f64)
}

#[inline]
pub(super) fn window_mean_square_stereo(
    in_l: &[f64],
    in_r: &[f64],
    start: usize,
    len: usize,
) -> f64 {
    if len == 0 {
        return 0.0;
    }
    let end = start + len;
    let sq: f64 = in_l[start..end]
        .iter()
        .zip(&in_r[start..end])
        .map(|(&l, &r)| l * l + r * r)
        .sum();
    sq / (2.0 * (len as f64))
}

#[inline]
pub(super) fn select_adjoint_window(
    in_l: &[f64],
    in_r: &[f64],
    n: usize,
    max_frames: usize,
) -> (usize, usize) {
    let cap = max_frames.max(64);
    if n <= cap {
        return (0, n);
    }
    let head_energy = window_mean_square_stereo(in_l, in_r, 0, cap);
    let mid_off = (n - cap) / 2;
    let mid_energy = window_mean_square_stereo(in_l, in_r, mid_off, cap);
    let tail_off = n - cap;
    let tail_energy = window_mean_square_stereo(in_l, in_r, tail_off, cap);
    if mid_energy >= 0.5 * head_energy.max(tail_energy) && mid_energy > 1.0e-12 {
        (mid_off, cap)
    } else if tail_energy > head_energy {
        (tail_off, cap)
    } else {
        (0, cap)
    }
}

#[inline]
pub(super) fn extract_switch_options(specs: &[ControlSpec], phys_controls: &[f32]) -> Vec<u8> {
    specs
        .iter()
        .enumerate()
        .map(|(i, sp)| match sp.kind {
            ControlKindSpec::Switch { num_options, .. } => {
                let raw = phys_controls.get(i).copied().unwrap_or(0.0);
                (raw.round().max(0.0) as u8).min(num_options.saturating_sub(1))
            }
            ControlKindSpec::Range { .. } => 0,
        })
        .collect()
}

#[inline]
pub(super) fn mid_signal(l: &[f64], r: &[f64]) -> Vec<f64> {
    l.iter().zip(r).map(|(&lv, &rv)| 0.5 * (lv + rv)).collect()
}

#[inline]
pub(super) fn inv_tier_trim(model: &dnam_train::SonaModelF64, tier: Tier) -> f64 {
    let trim = model.tier_gain_trim[tier.index()];
    if trim.is_finite() && trim.abs() > 1.0e-6 {
        1.0 / trim
    } else {
        1.0
    }
}

pub(super) struct LiteGuardCtx<'a> {
    pub(super) inp: &'a PointSolveInputs<'a>,
    pub(super) in_l32: &'a [f32],
    pub(super) in_r32: &'a [f32],
    pub(super) in_l64: &'a [f64],
    pub(super) in_r64: &'a [f64],
    pub(super) wet_l64: &'a [f64],
    pub(super) wet_r64: &'a [f64],
    pub(super) tgt_gn: &'a [f64],
    pub(super) warmup: usize,
    pub(super) adj_off: usize,
    pub(super) n_adj: usize,
    pub(super) warmup_adj: usize,
    pub(super) mse_lite_zero: f64,
    pub(super) broadband_adj: bool,
}

pub(super) struct WindowScoreCtx<'a> {
    pub(super) wet_l: &'a [f64],
    pub(super) wet_r: &'a [f64],
    pub(super) tgt_gn: &'a [f64],
    pub(super) warmup: usize,
    pub(super) mono_out: bool,
    pub(super) target_msq: f64,
    pub(super) is_broadband: bool,
}

pub(super) fn is_broadband_dry_window(in_l: &[f64], warmup: usize) -> bool {
    let Some((sig_e, _, _)) = try_split_bezout_3band_energies(in_l, in_l, None, 48_000.0, warmup)
    else {
        return false;
    };
    let tot = sig_e[0] + sig_e[1] + sig_e[2];
    tot > 1.0e-12 && sig_e[0].min(sig_e[1]).min(sig_e[2]) >= BROADBAND_MIN_BAND_FRAC * tot
}

pub(super) fn eval_vm_window_score(
    ctx: &WindowScoreCtx<'_>,
    mdl_l: &[f64],
    mdl_r: &[f64],
) -> (f64, f64) {
    let mse = vec_mse(
        &pack_post_warmup(mdl_l, mdl_r, ctx.warmup, ctx.mono_out),
        ctx.tgt_gn,
    );
    let nesr = mse / ctx.target_msq.max(1.0e-18);
    if !ctx.is_broadband {
        return (mse, nesr);
    }
    let Some((sl, el, nl)) =
        try_split_bezout_3band_energies(ctx.wet_l, mdl_l, None, 48_000.0, ctx.warmup)
    else {
        return (mse, nesr);
    };
    let rep_l = evaluate_besr_from_band_energies(sl, el, nl);
    if !rep_l.valid || rep_l.besr_db.is_nan() {
        return (mse, nesr);
    }
    let mut besr_db = rep_l.besr_db;
    if !ctx.mono_out {
        let Some((sr, er, nr)) =
            try_split_bezout_3band_energies(ctx.wet_r, mdl_r, None, 48_000.0, ctx.warmup)
        else {
            return (mse, nesr);
        };
        let rep_r = evaluate_besr_from_band_energies(sr, er, nr);
        if !rep_r.valid || rep_r.besr_db.is_nan() {
            return (mse, nesr);
        }
        besr_db = besr_db.max(rep_r.besr_db);
    }
    if besr_db == f64::INFINITY {
        return (mse, f64::INFINITY);
    }
    let besr_lin = if besr_db <= -120.0 {
        0.0
    } else {
        10.0_f64.powf(besr_db / 10.0)
    };
    (mse, 0.5 * nesr + 0.5 * besr_lin)
}

#[inline]
pub(super) fn pass1_loss_and_reg(broadband_adj: bool) -> (CompositeLossWeights, f64) {
    if broadband_adj {
        (
            CompositeLossWeights {
                w_mse: 0.45,
                w_besr: 0.55,
                ..CompositeLossWeights::zero()
            },
            1.0e-6,
        )
    } else {
        (
            CompositeLossWeights {
                w_mse: 1.0,
                ..CompositeLossWeights::zero()
            },
            1.0e-5,
        )
    }
}

pub(super) fn guard_lite_stage_base(
    vm_lite: &mut SonaVm<'_, DefaultMathBackend>,
    ctx: &LiteGuardCtx<'_>,
    st_final: &[f64; STAGE_TARGET_SLOTS],
) -> Result<([f64; STAGE_TARGET_SLOTS], Option<[f64; STAGE_TARGET_SLOTS]>), DnamError> {
    if ctx.inp.max_tier != Tier::Ultra {
        return Ok((*st_final, None));
    }
    let (lf, rf) = render_vm_with_stage_targets(vm_lite, ctx.in_l32, ctx.in_r32, st_final)?;
    let mse_lite_final = vec_mse(
        &pack_post_warmup(&lf, &rf, ctx.warmup, ctx.inp.mono_out),
        ctx.tgt_gn,
    );
    if mse_lite_final <= ctx.mse_lite_zero * 1.01 {
        return Ok((*st_final, None));
    }

    let mut best_st = [0.0_f64; STAGE_TARGET_SLOTS];
    let mut best_mse = ctx.mse_lite_zero;
    for alpha in [0.75_f64, 0.5, 0.25, 0.125, 0.0] {
        let mut cand = [0.0_f64; STAGE_TARGET_SLOTS];
        if alpha > 0.0 {
            for s in PASS4_ALL_ACTIVE_SLOTS {
                cand[s] = (alpha * st_final[s]).clamp(ctx.inp.bounds.min[s], ctx.inp.bounds.max[s]);
            }
        }
        let (cl, cr) = render_vm_with_stage_targets(vm_lite, ctx.in_l32, ctx.in_r32, &cand)?;
        let mse = vec_mse(
            &pack_post_warmup(&cl, &cr, ctx.warmup, ctx.inp.mono_out),
            ctx.tgt_gn,
        );
        if mse < best_mse {
            best_mse = mse;
            best_st = cand;
        }
    }

    if best_mse >= ctx.mse_lite_zero * 0.995 {
        let inv_l = inv_tier_trim(ctx.inp.base_lite, Tier::Lite);
        let adj_end = ctx.adj_off + ctx.n_adj;
        let lite_tl: Vec<f64> = ctx.wet_l64[ctx.adj_off..adj_end]
            .iter()
            .map(|&v| v * inv_l)
            .collect();
        let lite_tr: Vec<f64> = ctx.wet_r64[ctx.adj_off..adj_end]
            .iter()
            .map(|&v| v * inv_l)
            .collect();
        let (lite_weights, lite_reg) = pass1_loss_and_reg(ctx.broadband_adj);
        let cfg = StageTargetSolveConfig {
            active_slots: PASS4_ALL_ACTIVE_SLOTS.to_vec(),
            tier: Tier::Lite,
            warmup_frames: ctx.warmup_adj,
            loss_weights: lite_weights,
            barrier_weight: 0.0,
            l2_reg: lite_reg,
            opt: SsBroydenConfig {
                max_iters: 8,
                max_line_search: 3,
                rel_tol: 1.0e-4,
                patience: 2,
                ..SsBroydenConfig::default()
            },
            vm_micro_polish_iters: 0,
            ..StageTargetSolveConfig::default()
        };
        if let Ok(rep) = solve_point_stage_targets_adjoint(
            &StageTargetSolveInput {
                runner: ctx.inp.runner,
                base: ctx.inp.base_lite,
                in_l: &ctx.in_l64[ctx.adj_off..adj_end],
                in_r: &ctx.in_r64[ctx.adj_off..adj_end],
                target_l: &lite_tl,
                target_r: &lite_tr,
                init_targets: &best_st,
                z_stage: &[0.0; 8],
                bounds: ctx.inp.bounds,
            },
            &cfg,
            None,
        ) && let Ok((cl, cr)) =
            render_vm_with_stage_targets(vm_lite, ctx.in_l32, ctx.in_r32, &rep.stage_targets)
        {
            let mse = vec_mse(
                &pack_post_warmup(&cl, &cr, ctx.warmup, ctx.inp.mono_out),
                ctx.tgt_gn,
            );
            if mse < best_mse && mse < ctx.mse_lite_zero {
                best_mse = mse;
                best_st = rep.stage_targets;
            }
        }
    }

    if best_mse >= ctx.mse_lite_zero {
        best_st = [0.0_f64; STAGE_TARGET_SLOTS];
    }
    let mut delta = [0.0_f64; STAGE_TARGET_SLOTS];
    for s in PASS4_ALL_ACTIVE_SLOTS {
        delta[s] = st_final[s] - best_st[s];
    }
    Ok((best_st, Some(delta)))
}

pub(super) trait CompositeLossWeightsZero {
    fn zero() -> Self;
}

impl CompositeLossWeightsZero for CompositeLossWeights {
    fn zero() -> Self {
        CompositeLossWeights::from_config(&CompositeLossConfig {
            w_besr: 0.0,
            w_mr_iir: 0.0,
            w_spec_ot: 0.0,
            w_gabor: 0.0,
            w_if_gd: 0.0,
            w_ltas: 0.0,
            w_seg_esr: 0.0,
            w_nyquist: 0.0,
            w_mse: 0.0,
            ..CompositeLossConfig::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_train::losses::canonical_ltas_33_centers_hz;
    use std::f64::consts::TAU;

    #[test]
    fn test_select_adjoint_window_prefers_energetic_interior_or_tail() {
        let n = 4_096;
        let mut head_silent_l = vec![0.0_f64; n];
        let mut head_silent_r = vec![0.0_f64; n];
        for i in 1_024..3_072 {
            head_silent_l[i] = 0.25;
            head_silent_r[i] = 0.25;
        }
        let (off_mid, len_mid) = select_adjoint_window(&head_silent_l, &head_silent_r, n, 2_048);
        assert_eq!((off_mid, len_mid), (1_024, 2_048));

        let mut tail_heavy_l = vec![0.0_f64; n];
        let mut tail_heavy_r = vec![0.0_f64; n];
        for i in 2_048..n {
            tail_heavy_l[i] = 0.4;
            tail_heavy_r[i] = 0.4;
        }
        let (off_tail, _) = select_adjoint_window(&tail_heavy_l, &tail_heavy_r, n, 2_048);
        assert_eq!(off_tail, 2_048);
        assert_eq!(
            select_adjoint_window(&tail_heavy_l[..512], &tail_heavy_r[..512], 512, 2_048),
            (0, 512)
        );
    }

    #[test]
    fn test_is_broadband_dry_window_and_eval_vm_window_score() {
        let n = 2_048_usize;
        let sr = 48_000.0_f64;
        let warmup = 256_usize;
        let centers = canonical_ltas_33_centers_hz();
        let broad: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64) / sr;
                let mut s = 0.0_f64;
                for (k, &fc) in centers.iter().enumerate() {
                    s += (TAU * fc * t + 0.19 * (k as f64)).sin();
                }
                0.03 * s
            })
            .collect();
        let narrow: Vec<f64> = (0..n)
            .map(|i| 0.3 * (TAU * 440.0 * (i as f64) / sr).sin())
            .collect();

        assert!(is_broadband_dry_window(&broad, warmup));
        assert!(!is_broadband_dry_window(&narrow, warmup));

        let wet_half: Vec<f64> = broad.iter().map(|&v| 0.5 * v).collect();
        let ctx_broad = WindowScoreCtx {
            wet_l: &broad,
            wet_r: &broad,
            tgt_gn: &broad[warmup..],
            warmup,
            mono_out: true,
            target_msq: vec_mean_square(&broad[warmup..]),
            is_broadband: true,
        };
        let (mse_perf, sc_perf) = eval_vm_window_score(&ctx_broad, &broad, &broad);
        assert!(mse_perf < 1.0e-16 && sc_perf < 1.0e-11);
        let (mse_half, sc_half) = eval_vm_window_score(&ctx_broad, &wet_half, &wet_half);
        assert!(mse_half > 0.0 && sc_half > sc_perf);

        let ctx_narrow = WindowScoreCtx {
            wet_l: &narrow,
            wet_r: &narrow,
            tgt_gn: &narrow[warmup..],
            warmup,
            mono_out: true,
            target_msq: vec_mean_square(&narrow[warmup..]),
            is_broadband: false,
        };
        let (nmse, nsc) = eval_vm_window_score(
            &ctx_narrow,
            &narrow.iter().map(|&v| 0.5 * v).collect::<Vec<_>>(),
            &narrow,
        );
        assert!((nsc - nmse / ctx_narrow.target_msq).abs() < 1.0e-12);
    }
}
