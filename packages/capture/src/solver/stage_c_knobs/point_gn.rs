#![forbid(unsafe_code)]
//! Differentiable `31`-slot reverse-mode adjoint + persistent-`SonaVm` stage-target solver for
//! one Stage C control point (`ref[train.pipeline.control-projection-fit]`).

mod scoring;

use dnam_engine::schema::DnamError;
use dnam_train::control::{
    ControlSpec, Cp1ControlSample, FourBandTptBank, StageTargetBounds, StageTargetSolveConfig,
    StageTargetSolveInput, solve_point_stage_targets_adjoint, solve_two_pass_four_band_wiener,
};
use dnam_train::{CompositeLossWeights, SonaModelF64, SsBroydenConfig, TrainRunner};
use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use super::map_stage_err;
use scoring::{
    CompositeLossWeightsZero, LiteGuardCtx, WindowScoreCtx, eval_vm_window_score,
    extract_switch_options, guard_lite_stage_base, inv_tier_trim, is_broadband_dry_window,
    mid_signal, pack_post_warmup, pass1_loss_and_reg, render_vm_with_stage_targets,
    select_adjoint_window, vec_mean_square, vec_mse, window_mean_square_stereo,
};

/// All `31` active dispatch slots (`0..30`, slot `31` reserved `+0.0`).
pub(super) const PASS4_ALL_ACTIVE_SLOTS: [usize; 31] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30,
];

const BROADBAND_GAIN_DRIVE_SLOTS: [usize; 2] = [22, 30];
const MAX_ADJOINT_CORE_FRAMES: usize = 2_048;
const PASS0_EARLY_EXIT_NESR: f64 = 2.5e-4;
const PASS0_PARSIMONY_NESR: f64 = 2.5e-3;
const PASS0_NARROWBAND_KEEP_NESR: f64 = 1.0e-3;
const PASS0_EARLY_EXIT_REL_RESID: f64 = 0.05;
const PASS0_PARSIMONY_REL_RESID: f64 = 0.25;
const PASS1_PARSIMONY_REL_IMPROVE: f64 = 0.70;
const _: () = assert!(MAX_ADJOINT_CORE_FRAMES == 2_048);
const _: () = assert!(
    BROADBAND_GAIN_DRIVE_SLOTS[0] == 22
        && BROADBAND_GAIN_DRIVE_SLOTS[1] == 30
        && PASS0_EARLY_EXIT_NESR > 0.0
        && PASS0_EARLY_EXIT_NESR < PASS0_NARROWBAND_KEEP_NESR
        && PASS0_NARROWBAND_KEEP_NESR < PASS0_PARSIMONY_NESR
        && PASS0_EARLY_EXIT_REL_RESID > 0.0
        && PASS0_EARLY_EXIT_REL_RESID < PASS0_PARSIMONY_REL_RESID
        && PASS0_PARSIMONY_REL_RESID < 1.0
        && PASS1_PARSIMONY_REL_IMPROVE > 0.0
        && PASS1_PARSIMONY_REL_IMPROVE < 1.0
);

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

struct AdjointStageState {
    st: [f64; STAGE_TARGET_SLOTS],
    l: Vec<f64>,
    r: Vec<f64>,
    mse: f64,
    score: f64,
}

struct AdjWindowSlices<'a> {
    in_l32: &'a [f32],
    in_r32: &'a [f32],
    in_l: &'a [f64],
    in_r: &'a [f64],
    tgt_l: &'a [f64],
    tgt_r: &'a [f64],
    wet_m: &'a [f64],
    y_star_u: &'a [f64],
    y_star_l: &'a [f64],
    warmup_adj: usize,
    mse_ultra_zero: f64,
    broadband_adj: bool,
}

fn run_pass0_broadband_scale(
    inp: &PointSolveInputs<'_>,
    vm_ultra: &mut SonaVm<'_, DefaultMathBackend>,
    score_ctx: &WindowScoreCtx<'_>,
    adj: &AdjWindowSlices<'_>,
    zero_state: AdjointStageState,
) -> Result<AdjointStageState, DnamError> {
    let pass0_cfg = StageTargetSolveConfig {
        active_slots: BROADBAND_GAIN_DRIVE_SLOTS.to_vec(),
        tier: inp.max_tier,
        warmup_frames: adj.warmup_adj,
        loss_weights: CompositeLossWeights {
            w_mse: 1.0,
            ..CompositeLossWeights::zero()
        },
        barrier_weight: 0.0,
        l2_reg: 1.0e-7,
        opt: SsBroydenConfig {
            max_iters: 8,
            max_line_search: 4,
            rel_tol: 1.0e-6,
            patience: 2,
            ..SsBroydenConfig::default()
        },
        vm_micro_polish_iters: 0,
        ..StageTargetSolveConfig::default()
    };
    let Ok(rep0) = solve_point_stage_targets_adjoint(
        &StageTargetSolveInput {
            runner: inp.runner,
            base: inp.base_ultra,
            in_l: adj.in_l,
            in_r: adj.in_r,
            target_l: adj.tgt_l,
            target_r: adj.tgt_r,
            init_targets: &zero_state.st,
            z_stage: &[0.0; 8],
            bounds: inp.bounds,
        },
        &pass0_cfg,
        None,
    ) else {
        return Ok(zero_state);
    };
    let mut st0 = rep0.stage_targets;
    st0[STAGE_TARGET_SLOTS - 1] = 0.0;
    let (l0, r0) = render_vm_with_stage_targets(vm_ultra, adj.in_l32, adj.in_r32, &st0)?;
    let (mse0, score0) = eval_vm_window_score(score_ctx, &l0, &r0);
    if mse0 < zero_state.mse {
        Ok(AdjointStageState {
            st: st0,
            l: l0,
            r: r0,
            mse: mse0,
            score: score0,
        })
    } else {
        Ok(zero_state)
    }
}

fn refine_full_slots_pass1_and_pass2(
    inp: &PointSolveInputs<'_>,
    vms: &mut PointSolveVms<'_, '_>,
    score_ctx: &WindowScoreCtx<'_>,
    adj: &AdjWindowSlices<'_>,
    mut cur: AdjointStageState,
) -> Result<[f64; STAGE_TARGET_SLOTS], DnamError> {
    let pass0_explained_bulk = cur.mse <= PASS0_PARSIMONY_REL_RESID * adj.mse_ultra_zero;
    let pass0_already_good =
        pass0_explained_bulk && cur.mse <= PASS0_PARSIMONY_NESR * score_ctx.target_msq.max(1.0e-18);
    let clean_broadband_seed = adj.broadband_adj && !pass0_already_good;
    let (p1_max_iters, p1_max_ls, p1_patience, p1_init_st) = if clean_broadband_seed {
        (32, 6, 4, [0.0_f64; STAGE_TARGET_SLOTS])
    } else {
        (
            if adj.broadband_adj { 20 } else { 14 },
            if adj.broadband_adj { 5 } else { 4 },
            if adj.broadband_adj { 3 } else { 2 },
            cur.st,
        )
    };
    let (p1_weights, p1_reg) = pass1_loss_and_reg(adj.broadband_adj);
    let pass1_cfg = StageTargetSolveConfig {
        active_slots: PASS4_ALL_ACTIVE_SLOTS.to_vec(),
        tier: inp.max_tier,
        warmup_frames: adj.warmup_adj,
        loss_weights: p1_weights,
        barrier_weight: 0.0,
        l2_reg: p1_reg,
        opt: SsBroydenConfig {
            max_iters: p1_max_iters,
            max_line_search: p1_max_ls,
            rel_tol: 1.0e-5,
            patience: p1_patience,
            ..SsBroydenConfig::default()
        },
        vm_micro_polish_iters: 0,
        ..StageTargetSolveConfig::default()
    };

    let rep1 = solve_point_stage_targets_adjoint(
        &StageTargetSolveInput {
            runner: inp.runner,
            base: inp.base_ultra,
            in_l: adj.in_l,
            in_r: adj.in_r,
            target_l: adj.tgt_l,
            target_r: adj.tgt_r,
            init_targets: &p1_init_st,
            z_stage: &[0.0; 8],
            bounds: inp.bounds,
        },
        &pass1_cfg,
        None,
    )
    .map_err(map_stage_err)?;

    let req_factor = if pass0_already_good {
        PASS1_PARSIMONY_REL_IMPROVE
    } else {
        1.0
    };
    let mut st_pass1 = rep1.stage_targets;
    st_pass1[STAGE_TARGET_SLOTS - 1] = 0.0;
    let (p1_l, p1_r) =
        render_vm_with_stage_targets(vms.vm_ultra, adj.in_l32, adj.in_r32, &st_pass1)?;
    let (mse1, score1) = eval_vm_window_score(score_ctx, &p1_l, &p1_r);
    let accept_p1 = if adj.broadband_adj {
        score1 < req_factor * cur.score && mse1 <= cur.mse * 1.15 + 1.0e-12
    } else {
        mse1 < req_factor * cur.mse
    };
    if accept_p1 {
        cur = AdjointStageState {
            st: st_pass1,
            l: p1_l,
            r: p1_r,
            mse: mse1,
            score: score1,
        };
    } else if !pass0_already_good && cur.mse >= adj.mse_ultra_zero {
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
                in_l: adj.in_l,
                in_r: adj.in_r,
                target_l: adj.tgt_l,
                target_r: adj.tgt_r,
                init_targets: &st_pass1,
                z_stage: &[0.0; 8],
                bounds: inp.bounds,
            },
            &polish_cfg,
            Some(vms.vm_ultra),
        ) {
            let mut pol_st = pol.stage_targets;
            pol_st[STAGE_TARGET_SLOTS - 1] = 0.0;
            if let Ok((pl, pr)) =
                render_vm_with_stage_targets(vms.vm_ultra, adj.in_l32, adj.in_r32, &pol_st)
            {
                let (mse_pol, score_pol) = eval_vm_window_score(score_ctx, &pl, &pr);
                if score_pol < cur.score && mse_pol < adj.mse_ultra_zero {
                    cur = AdjointStageState {
                        st: pol_st,
                        l: pl,
                        r: pr,
                        mse: mse_pol,
                        score: score_pol,
                    };
                }
            }
        }
    }

    let y_mdl_u = mid_signal(&cur.l, &cur.r);
    let (ml, mr) = render_vm_with_stage_targets(vms.vm_lite, adj.in_l32, adj.in_r32, &cur.st)?;
    let y_mdl_l = mid_signal(&ml, &mr);
    let wr = solve_two_pass_four_band_wiener(
        inp.bank,
        adj.wet_m,
        adj.y_star_u,
        adj.y_star_l,
        &y_mdl_u,
        &y_mdl_l,
        score_ctx.warmup,
    )
    .map_err(map_stage_err)?;

    let mut st_final = cur.st;
    if wr.pass2_triggered && (accept_p1 || !pass0_already_good) {
        let p2_weights = if adj.broadband_adj {
            CompositeLossWeights {
                w_mse: 0.40,
                w_besr: 0.40,
                w_mr_iir: 0.20,
                ..CompositeLossWeights::zero()
            }
        } else {
            CompositeLossWeights {
                w_mse: 0.50,
                w_mr_iir: 0.30,
                w_seg_esr: 0.20,
                ..CompositeLossWeights::zero()
            }
        };
        let pass2_cfg = StageTargetSolveConfig {
            active_slots: PASS4_ALL_ACTIVE_SLOTS.to_vec(),
            tier: inp.max_tier,
            warmup_frames: adj.warmup_adj,
            loss_weights: p2_weights,
            barrier_weight: 0.0,
            l2_reg: p1_reg,
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
                in_l: adj.in_l,
                in_r: adj.in_r,
                target_l: adj.tgt_l,
                target_r: adj.tgt_r,
                init_targets: &cur.st,
                z_stage: &[0.0; 8],
                bounds: inp.bounds,
            },
            &pass2_cfg,
            None,
        ) {
            let mut rep2_st = rep2.stage_targets;
            rep2_st[STAGE_TARGET_SLOTS - 1] = 0.0;
            if let Ok((wl2, wr2)) =
                render_vm_with_stage_targets(vms.vm_ultra, adj.in_l32, adj.in_r32, &rep2_st)
            {
                let (mse_w, score_w) = eval_vm_window_score(score_ctx, &wl2, &wr2);
                if score_w < cur.score && mse_w <= cur.mse * 1.05 {
                    st_final = rep2_st;
                }
            }
        }
    }
    Ok(st_final)
}

/// Solves bounded `31`-slot `stage_targets` (`slot 31 == +0.0`) for one Stage C point using
/// parsimonious Pass-0 (`[22, 30]`) + bandwidth-aware reverse-mode adjoints + persistent `SonaVm`
/// parity & two-pass 4-band Wiener recovery (`ref[train.pipeline.control-projection-fit]`).
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
    let zero_st = [0.0_f64; STAGE_TARGET_SLOTS];
    if n <= warmup + 8 || window_mean_square_stereo(&in_l64, &in_r64, warmup, n - warmup) <= 1.0e-12
    {
        let switch_options = extract_switch_options(inp.specs, &inp.phys_controls);
        return Ok(Cp1ControlSample {
            phys_controls: inp.phys_controls,
            switch_options,
            stage_targets_base: zero_st,
            stage_targets_ultra_delta: None,
        });
    }
    let tgt_gn = pack_post_warmup(&wet_l, &wet_r, warmup, inp.mono_out);
    let target_msq = vec_mean_square(&tgt_gn);
    let broadband_full = is_broadband_dry_window(&in_l64, warmup);
    let score_ctx = WindowScoreCtx {
        wet_l: &wet_l,
        wet_r: &wet_r,
        tgt_gn: &tgt_gn,
        warmup,
        mono_out: inp.mono_out,
        target_msq,
        is_broadband: broadband_full,
    };

    let (su0, sr_u0) = render_vm_with_stage_targets(vms.vm_ultra, &in_l32, &in_r32, &zero_st)?;
    let y_star_u = mid_signal(&su0, &sr_u0);
    let (mse_ultra_zero, score_ultra_zero) = eval_vm_window_score(&score_ctx, &su0, &sr_u0);

    let (sl0, sr0) = render_vm_with_stage_targets(vms.vm_lite, &in_l32, &in_r32, &zero_st)?;
    let y_star_l = mid_signal(&sl0, &sr0);
    let mse_lite_zero = vec_mse(&pack_post_warmup(&sl0, &sr0, warmup, inp.mono_out), &tgt_gn);

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
    let broadband_adj = is_broadband_dry_window(&in_l64[adj_off..adj_end], warmup_adj);
    let wet_m = mid_signal(&wet_l, &wet_r);
    let adj_slices = AdjWindowSlices {
        in_l32: &in_l32,
        in_r32: &in_r32,
        in_l: &in_l64[adj_off..adj_end],
        in_r: &in_r64[adj_off..adj_end],
        tgt_l: &adj_tgt_l[adj_off..adj_end],
        tgt_r: &adj_tgt_r[adj_off..adj_end],
        wet_m: &wet_m,
        y_star_u: &y_star_u,
        y_star_l: &y_star_l,
        warmup_adj,
        mse_ultra_zero,
        broadband_adj,
    };

    let cur = run_pass0_broadband_scale(
        &inp,
        vms.vm_ultra,
        &score_ctx,
        &adj_slices,
        AdjointStageState {
            st: zero_st,
            l: su0,
            r: sr_u0,
            mse: mse_ultra_zero,
            score: score_ultra_zero,
        },
    )?;

    let tgt_scale = target_msq.max(1.0e-18);
    let pass0_explained_nearly_all = cur.mse <= PASS0_EARLY_EXIT_REL_RESID * mse_ultra_zero;
    let keep_pass0 = pass0_explained_nearly_all
        && (cur.mse <= PASS0_EARLY_EXIT_NESR * tgt_scale
            || (!broadband_adj && cur.mse <= PASS0_NARROWBAND_KEEP_NESR * tgt_scale));
    let st_final = if keep_pass0 {
        cur.st
    } else {
        refine_full_slots_pass1_and_pass2(&inp, vms, &score_ctx, &adj_slices, cur)?
    };

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
            broadband_adj,
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

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_train::control::{ControlKindSpec, hoist_stage_targets_onto_model};
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
