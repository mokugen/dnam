#![forbid(unsafe_code)]
//! Stereo Gauss–Newton `32`-slot stage-target solver for one Stage C control point
//! (`ref[train.pipeline.control-projection-fit]`).

use dnam_engine::schema::DnamError;
use dnam_train::control::{
    ControlKindSpec, ControlSpec, Cp1ControlSample, FourBandTptBank, render_vm_mono_probe,
    solve_control_jacobian_32slot, solve_two_pass_four_band_wiener,
};
use dnam_vm::control::STAGE_TARGET_SLOTS;
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use super::{PASS4_ACTIVE_SLOTS, map_stage_err};

pub(super) struct PointSolveInputs<'a> {
    pub dnam_bin: &'a [u8],
    pub ultra_slice: &'a [u8],
    pub lite_slice: &'a [u8],
    pub max_tier: Tier,
    pub mono_out: bool,
    pub dry_win: &'a [[f32; 2]],
    pub wet_win: &'a [[f32; 2]],
    pub phys_controls: Vec<f32>,
    pub specs: &'a [ControlSpec],
    pub bank: &'a FourBandTptBank,
}

fn render_stereo_with_stage_targets(
    sliced_bin: &[u8],
    tier: Tier,
    in_l: &[f32],
    in_r: &[f32],
    st32: &[f64; STAGE_TARGET_SLOTS],
) -> Result<(Vec<f64>, Vec<f64>), dnam_train::TrainError> {
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(sliced_bin, tier, 48_000.0)?;
    let mut st32_f32 = [0.0_f32; STAGE_TARGET_SLOTS];
    for (dst, &src) in st32_f32.iter_mut().zip(st32) {
        *dst = if src.is_finite() { src as f32 } else { 0.0 };
    }
    {
        let cold = vm.slab_mut().cold_mut();
        cold.ctrl_bus = None;
        cold.latched_stage_params = st32_f32;
    }
    dnam_vm::graph::slab::ctrl_latch::latch_immediate_at_init_or_reset(vm.slab_mut());
    let n = in_l.len().min(in_r.len());
    let (mut out_l, mut out_r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    vm.process_block(&in_l[..n], &in_r[..n], &mut out_l, &mut out_r)?;
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
fn mid_signal(l: &[f64], r: &[f64]) -> Vec<f64> {
    l.iter().zip(r).map(|(&lv, &rv)| 0.5 * (lv + rv)).collect()
}

#[inline]
fn clamp_stage_slot(slot: usize, val: f64) -> f64 {
    match slot {
        1 => val.clamp(-0.8, 1.5),
        2..=5 | 10..=11 => val.clamp(-1.5, 1.5),
        22 => val.clamp(-1.5, 2.0),
        _ => val.clamp(-2.0, 2.0),
    }
}

fn guard_lite_stage_base(
    inp: &PointSolveInputs<'_>,
    in_l: &[f32],
    in_r: &[f32],
    tgt_gn: &[f64],
    warmup: usize,
    mse_lite_zero: f64,
    st_final: &[f64; STAGE_TARGET_SLOTS],
) -> Result<([f64; STAGE_TARGET_SLOTS], Option<[f64; STAGE_TARGET_SLOTS]>), DnamError> {
    if inp.max_tier != Tier::Ultra {
        return Ok((*st_final, None));
    }
    let (lf, rf) =
        render_stereo_with_stage_targets(inp.lite_slice, Tier::Lite, in_l, in_r, st_final)
            .map_err(map_stage_err)?;
    let mse_lite_final = vec_mse(&pack_post_warmup(&lf, &rf, warmup, inp.mono_out), tgt_gn);
    if mse_lite_final <= mse_lite_zero * 1.01 {
        return Ok((*st_final, None));
    }

    let mut best_st = [0.0_f64; STAGE_TARGET_SLOTS];
    let mut best_mse = mse_lite_zero;
    for alpha in [0.75_f64, 0.5, 0.25, 0.125, 0.0] {
        let mut cand = [0.0_f64; STAGE_TARGET_SLOTS];
        if alpha > 0.0 {
            for s in PASS4_ACTIVE_SLOTS {
                cand[s] = clamp_stage_slot(s, alpha * st_final[s]);
            }
        }
        let (cl, cr) =
            render_stereo_with_stage_targets(inp.lite_slice, Tier::Lite, in_l, in_r, &cand)
                .map_err(map_stage_err)?;
        let mse = vec_mse(&pack_post_warmup(&cl, &cr, warmup, inp.mono_out), tgt_gn);
        if mse < best_mse {
            best_mse = mse;
            best_st = cand;
        }
    }

    if best_mse >= mse_lite_zero * 0.995 {
        let mut render_lite =
            |st: &[f64; STAGE_TARGET_SLOTS]| -> Result<Vec<f64>, dnam_train::TrainError> {
                let (ol, or) =
                    render_stereo_with_stage_targets(inp.lite_slice, Tier::Lite, in_l, in_r, st)?;
                Ok(pack_post_warmup(&ol, &or, warmup, inp.mono_out))
            };
        for _ in 0..2 {
            let Ok(step) = solve_control_jacobian_32slot(
                tgt_gn,
                &best_st,
                &PASS4_ACTIVE_SLOTS,
                1.0e-2,
                1.0e-3,
                &mut render_lite,
            ) else {
                break;
            };
            let mut improved = false;
            for alpha in [1.0_f64, 0.5, 0.25, 0.125] {
                let mut cand = best_st;
                for s in PASS4_ACTIVE_SLOTS {
                    cand[s] = clamp_stage_slot(s, best_st[s] + alpha * step[s]);
                }
                if let Ok((cl, cr)) =
                    render_stereo_with_stage_targets(inp.lite_slice, Tier::Lite, in_l, in_r, &cand)
                {
                    let mse = vec_mse(&pack_post_warmup(&cl, &cr, warmup, inp.mono_out), tgt_gn);
                    if mse < best_mse && mse < mse_lite_zero {
                        best_mse = mse;
                        best_st = cand;
                        improved = true;
                        break;
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }

    if best_mse >= mse_lite_zero {
        best_st = [0.0_f64; STAGE_TARGET_SLOTS];
    }
    let mut delta = [0.0_f64; STAGE_TARGET_SLOTS];
    for s in PASS4_ACTIVE_SLOTS {
        delta[s] = st_final[s] - best_st[s];
    }
    Ok((best_st, Some(delta)))
}

/// Runs stereo damped Gauss–Newton (`5` max iterations, adaptive Levenberg–Marquardt damping
/// floored at `2e-4`) and guarded 4-band Wiener fallback for one Stage C control point
/// (`ref[train.pipeline.control-projection-fit]`).
pub(super) fn solve_point_cp1_sample(
    inp: PointSolveInputs<'_>,
) -> Result<Cp1ControlSample, DnamError> {
    let n = inp.dry_win.len().min(inp.wet_win.len());
    let in_l: Vec<f32> = inp.dry_win[..n].iter().map(|lr| lr[0]).collect();
    let in_r: Vec<f32> = inp.dry_win[..n].iter().map(|lr| lr[1]).collect();
    let wet_l: Vec<f64> = inp.wet_win[..n].iter().map(|lr| f64::from(lr[0])).collect();
    let wet_r: Vec<f64> = inp.wet_win[..n]
        .iter()
        .map(|lr| f64::from(if inp.mono_out { lr[0] } else { lr[1] }))
        .collect();
    let warmup = (n / 8).min(256).min(n.saturating_sub(8));
    let tgt_gn = pack_post_warmup(&wet_l, &wet_r, warmup, inp.mono_out);
    let zero_st = [0.0_f64; STAGE_TARGET_SLOTS];

    let mut render_gn =
        |st: &[f64; STAGE_TARGET_SLOTS]| -> Result<Vec<f64>, dnam_train::TrainError> {
            let (ol, or) =
                render_stereo_with_stage_targets(inp.ultra_slice, inp.max_tier, &in_l, &in_r, st)?;
            Ok(pack_post_warmup(&ol, &or, warmup, inp.mono_out))
        };

    let y_star_u = render_vm_mono_probe(inp.dnam_bin, inp.max_tier, 48_000.0, &in_l)
        .or_else(|_| -> Result<Vec<f64>, dnam_train::TrainError> {
            let (ol, or) = render_stereo_with_stage_targets(
                inp.ultra_slice,
                inp.max_tier,
                &in_l,
                &in_r,
                &zero_st,
            )?;
            Ok(mid_signal(&ol, &or))
        })
        .map_err(map_stage_err)?;
    let (sl0, sr0) =
        render_stereo_with_stage_targets(inp.lite_slice, Tier::Lite, &in_l, &in_r, &zero_st)
            .map_err(map_stage_err)?;
    let y_star_l = mid_signal(&sl0, &sr0);
    let mse_lite_zero = vec_mse(&pack_post_warmup(&sl0, &sr0, warmup, inp.mono_out), &tgt_gn);

    let mut st_cur = zero_st;
    let (mut cur_l, mut cur_r) =
        render_stereo_with_stage_targets(inp.ultra_slice, inp.max_tier, &in_l, &in_r, &st_cur)
            .map_err(map_stage_err)?;
    let y_cur_gn = pack_post_warmup(&cur_l, &cur_r, warmup, inp.mono_out);
    let mut cur_mse = vec_mse(&y_cur_gn, &tgt_gn);

    let mut lm_reg = 2.0e-4_f64;
    for _iter in 0..5 {
        let mut accepted = false;
        let mut keep_going = false;
        for _lm_try in 0..3 {
            let step = solve_control_jacobian_32slot(
                &tgt_gn,
                &st_cur,
                &PASS4_ACTIVE_SLOTS,
                1.0e-2,
                lm_reg,
                &mut render_gn,
            )
            .map_err(map_stage_err)?;
            let step_norm_sq: f64 = PASS4_ACTIVE_SLOTS.iter().map(|&s| step[s] * step[s]).sum();
            if step_norm_sq < 1.0e-10 {
                break;
            }
            for alpha in [1.0_f64, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
                let mut cand = st_cur;
                for s in PASS4_ACTIVE_SLOTS {
                    cand[s] = clamp_stage_slot(s, st_cur[s] + alpha * step[s]);
                }
                let (cl, cr) = render_stereo_with_stage_targets(
                    inp.ultra_slice,
                    inp.max_tier,
                    &in_l,
                    &in_r,
                    &cand,
                )
                .map_err(map_stage_err)?;
                let y_cand_gn = pack_post_warmup(&cl, &cr, warmup, inp.mono_out);
                let mse_cand = vec_mse(&y_cand_gn, &tgt_gn);
                if mse_cand < cur_mse {
                    let rel_gain = (cur_mse - mse_cand) / cur_mse.max(1.0e-18);
                    st_cur = cand;
                    cur_l = cl;
                    cur_r = cr;
                    cur_mse = mse_cand;
                    lm_reg = (lm_reg * 0.5).max(2.0e-4);
                    accepted = true;
                    keep_going = rel_gain >= 1.0e-3;
                    break;
                }
            }
            if accepted {
                break;
            }
            lm_reg = (lm_reg * 20.0).min(1.0);
        }
        if !accepted || !keep_going {
            break;
        }
    }

    let st_pass1 = st_cur;
    let tgt_m = mid_signal(&wet_l, &wet_r);
    let y_mdl_u = mid_signal(&cur_l, &cur_r);
    let (ml, mr) =
        render_stereo_with_stage_targets(inp.lite_slice, Tier::Lite, &in_l, &in_r, &st_pass1)
            .map_err(map_stage_err)?;
    let y_mdl_l = mid_signal(&ml, &mr);

    let wr = solve_two_pass_four_band_wiener(
        inp.bank, &tgt_m, &y_star_u, &y_star_l, &y_mdl_u, &y_mdl_l, warmup,
    )
    .map_err(map_stage_err)?;
    let mut st_final = st_pass1;
    if wr.pass2_triggered {
        let bands = inp.bank.split_4bands(&y_mdl_u);
        let mut tgt_w = vec![0.0_f64; n];
        for t in 0..n {
            let boosted: f64 = (0..4).map(|b| wr.wiener_band_gains[b] * bands[b][t]).sum();
            tgt_w[t] = 0.5 * tgt_m[t] + 0.5 * boosted;
        }
        let delta2 = solve_control_jacobian_32slot(
            &tgt_w,
            &st_pass1,
            &PASS4_ACTIVE_SLOTS,
            1.0e-2,
            2.0e-3,
            |st| {
                let (ol, or) = render_stereo_with_stage_targets(
                    inp.ultra_slice,
                    inp.max_tier,
                    &in_l,
                    &in_r,
                    st,
                )?;
                Ok(mid_signal(&ol, &or))
            },
        )
        .map_err(map_stage_err)?;
        let mut st_w = st_pass1;
        for s in PASS4_ACTIVE_SLOTS {
            st_w[s] = clamp_stage_slot(s, st_pass1[s] + delta2[s]);
        }
        if let Ok((wl2, wr2)) =
            render_stereo_with_stage_targets(inp.ultra_slice, inp.max_tier, &in_l, &in_r, &st_w)
        {
            let mse_w = vec_mse(&pack_post_warmup(&wl2, &wr2, warmup, inp.mono_out), &tgt_gn);
            if mse_w < cur_mse {
                st_final = st_w;
            }
        }
    }

    let (st_base, st_ultra_delta) = guard_lite_stage_base(
        &inp,
        &in_l,
        &in_r,
        &tgt_gn,
        warmup,
        mse_lite_zero,
        &st_final,
    )?;

    let switch_options = inp
        .specs
        .iter()
        .enumerate()
        .map(|(i, sp)| match sp.kind {
            ControlKindSpec::Switch { num_options, .. } => {
                let raw = inp.phys_controls.get(i).copied().unwrap_or(0.0);
                (raw.round().max(0.0) as u8).min(num_options.saturating_sub(1))
            }
            ControlKindSpec::Range { .. } => 0,
        })
        .collect();

    Ok(Cp1ControlSample {
        phys_controls: inp.phys_controls,
        switch_options,
        stage_targets_base: st_base,
        stage_targets_ultra_delta: st_ultra_delta,
    })
}
