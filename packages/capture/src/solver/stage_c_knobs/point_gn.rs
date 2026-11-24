#![forbid(unsafe_code)]
//! Differentiable `31`-slot reverse-mode adjoint + persistent-`SonaVm` stage-target solver for
//! one Stage C control point (`ref[train.pipeline.control-projection-fit]`).

use dnam_engine::schema::DnamError;
use dnam_train::control::{
    ControlKindSpec, ControlSpec, Cp1ControlSample, FourBandTptBank, StageTargetBounds,
    StageTargetSolveConfig, StageTargetSolveInput, solve_point_stage_targets_adjoint,
    solve_two_pass_four_band_wiener,
};
use dnam_train::{
    CompositeLossConfig, CompositeLossWeights, SonaModelF64, SsBroydenConfig, TrainRunner,
};
use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use super::map_stage_err;

/// All `31` active dispatch slots (`0..30`, slot `31` reserved `+0.0`).
pub(super) const PASS4_ALL_ACTIVE_SLOTS: [usize; 31] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30,
];

const MAX_ADJOINT_CORE_FRAMES: usize = 2_048;
const _: () = assert!(MAX_ADJOINT_CORE_FRAMES == 2_048);

/// Immutable inputs and pre-decoded models passed to [`solve_point_cp1_sample`].
pub(super) struct PointSolveInputs<'a> {
    pub runner: &'a TrainRunner,
    pub base_ultra: &'a SonaModelF64,
    pub base_lite: &'a SonaModelF64,
    pub bounds: &'a StageTargetBounds,
    pub max_tier: Tier,
    pub mono_out: bool,
    pub dual_in: bool,
    pub dry_win: &'a [[f32; 2]],
    pub wet_win: &'a [[f32; 2]],
    pub phys_controls: Vec<f32>,
    pub specs: &'a [ControlSpec],
    pub bank: &'a FourBandTptBank,
}

/// Persistent reusable `SonaVm` instances (`0` re-parses across control points and passes).
pub(super) struct PointSolveVms<'a, 'b> {
    pub vm_ultra: &'a mut SonaVm<'b, DefaultMathBackend>,
    pub vm_lite: &'a mut SonaVm<'b, DefaultMathBackend>,
}

fn render_vm_with_stage_targets(
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
fn pack_post_warmup(l: &[f64], r: &[f64], warmup: usize, mono_out: bool) -> Vec<f64> {
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
fn vec_mse(pred: &[f64], target: &[f64]) -> f64 {
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
fn vec_mean_square(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let sq: f64 = v.iter().map(|&x| x * x).sum();
    sq / (v.len() as f64)
}

#[inline]
fn window_mean_square_stereo(in_l: &[f64], in_r: &[f64], start: usize, len: usize) -> f64 {
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
fn select_adjoint_window(
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
    let tail_off = n - cap;
    let tail_energy = window_mean_square_stereo(in_l, in_r, tail_off, cap);
    if tail_energy > 2.0 * head_energy {
        (tail_off, cap)
    } else {
        (0, cap)
    }
}

#[inline]
fn extract_switch_options(specs: &[ControlSpec], phys_controls: &[f32]) -> Vec<u8> {
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
fn mid_signal(l: &[f64], r: &[f64]) -> Vec<f64> {
    l.iter().zip(r).map(|(&lv, &rv)| 0.5 * (lv + rv)).collect()
}

#[inline]
fn inv_tier_trim(model: &SonaModelF64, tier: Tier) -> f64 {
    let trim = model.tier_gain_trim[tier.index()];
    if trim.is_finite() && trim.abs() > 1.0e-6 {
        1.0 / trim
    } else {
        1.0
    }
}

struct LiteGuardCtx<'a> {
    inp: &'a PointSolveInputs<'a>,
    in_l32: &'a [f32],
    in_r32: &'a [f32],
    in_l64: &'a [f64],
    in_r64: &'a [f64],
    wet_l64: &'a [f64],
    wet_r64: &'a [f64],
    tgt_gn: &'a [f64],
    warmup: usize,
    adj_off: usize,
    n_adj: usize,
    warmup_adj: usize,
    mse_lite_zero: f64,
}

fn guard_lite_stage_base(
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
        let cfg = StageTargetSolveConfig {
            active_slots: PASS4_ALL_ACTIVE_SLOTS.to_vec(),
            tier: Tier::Lite,
            warmup_frames: ctx.warmup_adj,
            loss_weights: CompositeLossWeights {
                w_besr: 1.0,
                ..CompositeLossWeights::zero()
            },
            barrier_weight: 0.0,
            l2_reg: 1.0e-6,
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

/// Solves bounded `31`-slot `stage_targets` (`slot 31 == +0.0`) for one Stage C point using
/// reverse-mode `Pass2` adjoints (`SsBroydenState`) + persistent `SonaVm` parity & two-pass
/// 4-band Wiener recovery (`ref[train.pipeline.control-projection-fit]`).
pub(super) fn solve_point_cp1_sample(
    inp: PointSolveInputs<'_>,
    vms: &mut PointSolveVms<'_, '_>,
) -> Result<Cp1ControlSample, DnamError> {
    let n = inp.dry_win.len().min(inp.wet_win.len());
    let keep_in_r = inp.dual_in || !inp.mono_out;
    let in_l32: Vec<f32> = inp.dry_win[..n].iter().map(|lr| lr[0]).collect();
    let in_r32: Vec<f32> = inp.dry_win[..n]
        .iter()
        .map(|lr| if keep_in_r { lr[1] } else { lr[0] })
        .collect();
    let in_l64: Vec<f64> = in_l32.iter().map(|&v| f64::from(v)).collect();
    let in_r64: Vec<f64> = in_r32.iter().map(|&v| f64::from(v)).collect();

    let wet_l: Vec<f64> = inp.wet_win[..n].iter().map(|lr| f64::from(lr[0])).collect();
    let wet_r: Vec<f64> = inp.wet_win[..n]
        .iter()
        .map(|lr| f64::from(if inp.mono_out { lr[0] } else { lr[1] }))
        .collect();

    let inv_u = inv_tier_trim(inp.base_ultra, inp.max_tier);
    let adj_tgt_l: Vec<f64> = wet_l.iter().map(|&v| v * inv_u).collect();
    let adj_tgt_r: Vec<f64> = wet_r.iter().map(|&v| v * inv_u).collect();

    let warmup = (n / 8).min(256).min(n.saturating_sub(8));
    let tgt_gn = pack_post_warmup(&wet_l, &wet_r, warmup, inp.mono_out);
    let zero_st = [0.0_f64; STAGE_TARGET_SLOTS];

    let (su0, sr_u0) = render_vm_with_stage_targets(vms.vm_ultra, &in_l32, &in_r32, &zero_st)?;
    let y_star_u = mid_signal(&su0, &sr_u0);
    let mse_ultra_zero = vec_mse(
        &pack_post_warmup(&su0, &sr_u0, warmup, inp.mono_out),
        &tgt_gn,
    );

    let (sl0, sr0) = render_vm_with_stage_targets(vms.vm_lite, &in_l32, &in_r32, &zero_st)?;
    let y_star_l = mid_signal(&sl0, &sr0);
    let mse_lite_zero = vec_mse(&pack_post_warmup(&sl0, &sr0, warmup, inp.mono_out), &tgt_gn);

    let target_msq = vec_mean_square(&tgt_gn);
    if mse_ultra_zero <= 1.0e-10 * target_msq.max(1.0e-18) {
        let switch_options = extract_switch_options(inp.specs, &inp.phys_controls);
        return Ok(Cp1ControlSample {
            phys_controls: inp.phys_controls,
            switch_options,
            stage_targets_base: zero_st,
            stage_targets_ultra_delta: None,
        });
    }

    let os2x_active = inp.base_ultra.oversample_upper_tiers_2x && inp.max_tier.simd_groups() > 4;
    let max_adj_frames = if os2x_active {
        MAX_ADJOINT_CORE_FRAMES / 2
    } else {
        MAX_ADJOINT_CORE_FRAMES
    };
    let (adj_off, n_adj) = select_adjoint_window(&in_l64, &in_r64, n, max_adj_frames);
    let adj_end = adj_off + n_adj;
    let warmup_adj = (n_adj / 8).min(256).min(n_adj.saturating_sub(8));

    let pass1_cfg = StageTargetSolveConfig {
        active_slots: PASS4_ALL_ACTIVE_SLOTS.to_vec(),
        tier: inp.max_tier,
        warmup_frames: warmup_adj,
        loss_weights: CompositeLossWeights {
            w_besr: 1.0,
            ..CompositeLossWeights::zero()
        },
        barrier_weight: 0.0,
        l2_reg: 1.0e-6,
        opt: SsBroydenConfig {
            max_iters: 18,
            max_line_search: 5,
            rel_tol: 1.0e-5,
            patience: 2,
            ..SsBroydenConfig::default()
        },
        vm_micro_polish_iters: 0,
        ..StageTargetSolveConfig::default()
    };

    let rep1 = solve_point_stage_targets_adjoint(
        &StageTargetSolveInput {
            runner: inp.runner,
            base: inp.base_ultra,
            in_l: &in_l64[adj_off..adj_end],
            in_r: &in_r64[adj_off..adj_end],
            target_l: &adj_tgt_l[adj_off..adj_end],
            target_r: &adj_tgt_r[adj_off..adj_end],
            init_targets: &zero_st,
            z_stage: &[0.0; 8],
            bounds: inp.bounds,
        },
        &pass1_cfg,
        None,
    )
    .map_err(map_stage_err)?;

    let mut st_pass1 = rep1.stage_targets;
    st_pass1[STAGE_TARGET_SLOTS - 1] = 0.0;

    let (mut cur_l, mut cur_r) =
        render_vm_with_stage_targets(vms.vm_ultra, &in_l32, &in_r32, &st_pass1)?;
    let mut cur_mse = vec_mse(
        &pack_post_warmup(&cur_l, &cur_r, warmup, inp.mono_out),
        &tgt_gn,
    );
    if cur_mse >= mse_ultra_zero {
        let mut polish_cfg = pass1_cfg;
        polish_cfg.opt.max_iters = 2;
        polish_cfg.opt.max_line_search = 2;
        polish_cfg.opt.rel_tol = 1.0e-4;
        polish_cfg.opt.patience = 1;
        polish_cfg.vm_micro_polish_iters = 1;
        if let Ok(pol) = solve_point_stage_targets_adjoint(
            &StageTargetSolveInput {
                runner: inp.runner,
                base: inp.base_ultra,
                in_l: &in_l64[adj_off..adj_end],
                in_r: &in_r64[adj_off..adj_end],
                target_l: &adj_tgt_l[adj_off..adj_end],
                target_r: &adj_tgt_r[adj_off..adj_end],
                init_targets: &st_pass1,
                z_stage: &[0.0; 8],
                bounds: inp.bounds,
            },
            &polish_cfg,
            Some(vms.vm_ultra),
        ) && let Ok((pl, pr)) =
            render_vm_with_stage_targets(vms.vm_ultra, &in_l32, &in_r32, &pol.stage_targets)
        {
            let mse_pol = vec_mse(&pack_post_warmup(&pl, &pr, warmup, inp.mono_out), &tgt_gn);
            if mse_pol < cur_mse {
                st_pass1 = pol.stage_targets;
                cur_l = pl;
                cur_r = pr;
                cur_mse = mse_pol;
            }
        }
    }
    if cur_mse > mse_ultra_zero * 1.0001 {
        st_pass1 = zero_st;
        cur_l = su0;
        cur_r = sr_u0;
        cur_mse = mse_ultra_zero;
    }

    let tgt_m = mid_signal(&wet_l, &wet_r);
    let y_mdl_u = mid_signal(&cur_l, &cur_r);
    let (ml, mr) = render_vm_with_stage_targets(vms.vm_lite, &in_l32, &in_r32, &st_pass1)?;
    let y_mdl_l = mid_signal(&ml, &mr);

    let wr = solve_two_pass_four_band_wiener(
        inp.bank, &tgt_m, &y_star_u, &y_star_l, &y_mdl_u, &y_mdl_l, warmup,
    )
    .map_err(map_stage_err)?;

    let mut st_final = st_pass1;
    if wr.pass2_triggered {
        let pass2_cfg = StageTargetSolveConfig {
            active_slots: PASS4_ALL_ACTIVE_SLOTS.to_vec(),
            tier: inp.max_tier,
            warmup_frames: warmup_adj,
            loss_weights: CompositeLossWeights {
                w_besr: 0.35,
                w_mr_iir: 0.35,
                w_seg_esr: 0.30,
                ..CompositeLossWeights::zero()
            },
            barrier_weight: 0.0,
            l2_reg: 1.0e-6,
            opt: SsBroydenConfig {
                max_iters: 8,
                max_line_search: 4,
                rel_tol: 1.0e-4,
                patience: 2,
                ..SsBroydenConfig::default()
            },
            vm_micro_polish_iters: 0,
            ..StageTargetSolveConfig::default()
        };
        if let Ok(rep2) = solve_point_stage_targets_adjoint(
            &StageTargetSolveInput {
                runner: inp.runner,
                base: inp.base_ultra,
                in_l: &in_l64[adj_off..adj_end],
                in_r: &in_r64[adj_off..adj_end],
                target_l: &adj_tgt_l[adj_off..adj_end],
                target_r: &adj_tgt_r[adj_off..adj_end],
                init_targets: &st_pass1,
                z_stage: &[0.0; 8],
                bounds: inp.bounds,
            },
            &pass2_cfg,
            None,
        ) && let Ok((wl2, wr2)) =
            render_vm_with_stage_targets(vms.vm_ultra, &in_l32, &in_r32, &rep2.stage_targets)
        {
            let mse_w = vec_mse(&pack_post_warmup(&wl2, &wr2, warmup, inp.mono_out), &tgt_gn);
            if mse_w < cur_mse {
                st_final = rep2.stage_targets;
            }
        }
    }

    let (st_base, st_ultra_delta) = {
        let lite_ctx = LiteGuardCtx {
            inp: &inp,
            in_l32: &in_l32,
            in_r32: &in_r32,
            in_l64: &in_l64,
            in_r64: &in_r64,
            wet_l64: &wet_l,
            wet_r64: &wet_r,
            tgt_gn: &tgt_gn,
            warmup,
            adj_off,
            n_adj,
            warmup_adj,
            mse_lite_zero,
        };
        guard_lite_stage_base(vms.vm_lite, &lite_ctx, &st_final)?
    };

    let switch_options = extract_switch_options(inp.specs, &inp.phys_controls);

    Ok(Cp1ControlSample {
        phys_controls: inp.phys_controls,
        switch_options,
        stage_targets_base: st_base,
        stage_targets_ultra_delta: st_ultra_delta,
    })
}

trait CompositeLossWeightsZero {
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
            ..CompositeLossConfig::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_train::control::hoist_stage_targets_onto_model;
    use dnam_train::{Pass5EmitConfig, decode_dnam_bin_to_model, encode_model_to_dnam_bin};
    use dnam_vm::bin::{DnamBinView, strip_dnam_bin_to_tier};
    use std::f64::consts::TAU;

    #[test]
    fn test_solve_point_cp1_sample_recovers_extended_cext_slot_with_persistent_vm() {
        let mut model = SonaModelF64::new(48_000.0);
        model.oversample_upper_tiers_2x = false;
        model.active_fir_taps = 4;
        model.fir_l[0] = 0.5;
        model.fir_r[0] = 0.5;
        model.tank_dry_mix = 0.5;
        model.pre_conv_u_gain = [0.6, 0.6];
        for k in 0..4 {
            model.transport_quads[k].delay_samples = [16.0; 4];
            model.transport_quads[k].g_loop = 0.25;
            model.transport_quads[k].wet_tap_l = [0.25, 0.1, 0.0, 0.0];
            model.transport_quads[k].wet_tap_r = [0.1, 0.25, 0.0, 0.0];
        }

        let mut teacher_targets = [0.0_f64; STAGE_TARGET_SLOTS];
        teacher_targets[0] = 0.4;
        teacher_targets[22] = 0.6; // `"CEXT"` slot: op_t_dry_mix_mod!
        let hoisted =
            hoist_stage_targets_onto_model(&model, &teacher_targets, &[0.0; 8], Tier::Ultra);

        let dbin_base = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).unwrap();
        let dbin_teacher =
            encode_model_to_dnam_bin(&hoisted.model, Pass5EmitConfig::default()).unwrap();

        let n = 512usize;
        let dry_lr: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let t = (i as f64) / 48_000.0;
                let s = (0.3 * (TAU * 220.0 * t).sin() + 0.15 * (TAU * 1100.0 * t).cos()) as f32;
                [s, 0.8 * s]
            })
            .collect();
        let in_l: Vec<f32> = dry_lr.iter().map(|lr| lr[0]).collect();
        let in_r: Vec<f32> = dry_lr.iter().map(|lr| lr[1]).collect();

        let mut vm_teach =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&dbin_teacher, Tier::Ultra, 48_000.0)
                .unwrap();
        let (mut wl, mut wr) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        vm_teach
            .process_block(&in_l, &in_r, &mut wl, &mut wr)
            .unwrap();
        let wet_lr: Vec<[f32; 2]> = wl.into_iter().zip(wr).map(|(l, r)| [l, r]).collect();

        let ultra_slice = strip_dnam_bin_to_tier(&dbin_base, Tier::Ultra).unwrap();
        let lite_slice = strip_dnam_bin_to_tier(&dbin_base, Tier::Lite).unwrap();
        let base_ultra = decode_dnam_bin_to_model(ultra_slice).unwrap();
        let base_lite = decode_dnam_bin_to_model(lite_slice).unwrap();
        let runner = TrainRunner::cpu();
        let bank = FourBandTptBank::new(48_000.0);
        let bounds = StageTargetBounds::canonical(true, 16.0);
        let specs = vec![ControlSpec {
            kind: ControlKindSpec::Range {
                declared_min: 0.0,
                declared_max: 1.0,
                default_val: 0.0,
                obs_min: 0.0,
                obs_max: 1.0,
                unique_values: 2,
                lobatto_anchors: [0.0, 0.5, 1.0],
            },
        }];

        let uv = DnamBinView::parse(ultra_slice).unwrap();
        let lv = DnamBinView::parse(lite_slice).unwrap();
        let mut vm_ultra =
            SonaVm::<DefaultMathBackend>::from_view_owned(&uv, Tier::Ultra, 48_000.0, false)
                .unwrap();
        let mut vm_lite =
            SonaVm::<DefaultMathBackend>::from_view_owned(&lv, Tier::Lite, 48_000.0, false)
                .unwrap();

        let mut vms = PointSolveVms {
            vm_ultra: &mut vm_ultra,
            vm_lite: &mut vm_lite,
        };
        let samp = solve_point_cp1_sample(
            PointSolveInputs {
                runner: &runner,
                base_ultra: &base_ultra,
                base_lite: &base_lite,
                bounds: &bounds,
                max_tier: Tier::Ultra,
                mono_out: false,
                dual_in: true,
                dry_win: &dry_lr,
                wet_win: &wet_lr,
                phys_controls: vec![1.0],
                specs: &specs,
                bank: &bank,
            },
            &mut vms,
        )
        .expect("31-slot adjoint sample solve should succeed");

        assert_eq!(samp.stage_targets_base[31], 0.0);
        let eff_22 =
            samp.stage_targets_base[22] + samp.stage_targets_ultra_delta.map_or(0.0, |d| d[22]);
        assert!(
            eff_22 > 0.25,
            "expected positive slot 22 CEXT recovery, got {eff_22}"
        );
    }
}
