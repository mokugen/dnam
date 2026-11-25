//! Post-`Op-T` onset-decoupled `[Kautz + FIR]` joint fitting and alternating series feed-forward
//! reflection deconvolution refinement (`ref[train.pipeline.physics-probe]`).

use super::super::model::{KautzPairF64, NUM_FIR_TAPS_ULTRA, TransportQuadF64};
use super::super::pass2_opt::op_t_quad::eval_op_t_quad_stereo_with_wet_dc_and_adjoint;
use super::fir_toeplitz::{fit_early_fir_taps, residual_after_fir};
use super::kautz_fit::{
    KautzFitInputs, append_log_octave_h2_candidates, audible_kautz_theta_bounds,
    fit_joint_kautz_bank, poles_too_close, synth_kautz_unit_mode,
};
use super::kautz_varpro::{
    VarProPoleConfig, refine_kautz_poles_kaufman_varpro, select_multiregime_blocks,
};
use super::op_t_seed::OpTSeedOutcome;
use super::op_t_seed::reflections::{
    FeedforwardQuadLsInputs, PASSIVE_FF_DC_POLE_F64, causal_invert_feedforward_op_t,
    seed_series_feedforward_reflections, solve_feedforward_quad_ls,
};
use super::steiglitz_mcbride::steiglitz_mcbride_poles as extract_sm_poles;

const MAX_ALT_DECONV_ITERS: usize = 5;
const ALT_CONVERGE_TOL: f64 = 2.0e-3;
const _: () = assert!(MAX_ALT_DECONV_ITERS >= 3 && ALT_CONVERGE_TOL > 0.0);

#[derive(Clone, Copy, Debug)]
pub(super) struct KautzFirFitConfig {
    pub sample_rate_hz: f64,
    pub fir_n: usize,
    pub k_cap: usize,
    pub start: usize,
    pub tgt_tot_e: f64,
    pub r_max_phys: f64,
    pub use_varpro: bool,
}

pub(super) struct KautzFirFitResult {
    pub kautz: Vec<KautzPairF64>,
    pub fir_l: [f64; NUM_FIR_TAPS_ULTRA],
    pub fir_r: [f64; NUM_FIR_TAPS_ULTRA],
    pub res_l: Vec<f64>,
    pub res_r: Vec<f64>,
    pub res_tot_e: f64,
}

type KautzBankTuple = (Vec<KautzPairF64>, Vec<f64>, Vec<f64>);

struct PriorKautzFit {
    kautz: Vec<KautzPairF64>,
    res_l: Vec<f64>,
    res_r: Vec<f64>,
    err: f64,
}

struct Stage12FitResult {
    ref_fit: Option<PriorKautzFit>,
    full_fit: Option<PriorKautzFit>,
    locked_poles: Vec<(f64, f64)>,
    lock_lite: usize,
    lock_full: usize,
}

struct OnsetTargets<'a> {
    input_l: &'a [f64],
    input_r: &'a [f64],
    tgt_l: &'a [f64],
    tgt_r: &'a [f64],
    res_16_l: &'a [f64],
    res_16_r: &'a [f64],
    res_16_tot_e: f64,
    fir_onset_n: usize,
    start16: usize,
    n: usize,
}

fn eval_pair_energy(l: &[f64], r: &[f64], start: usize, end: usize) -> f64 {
    l[start..end]
        .iter()
        .zip(&r[start..end])
        .map(|(a, b)| a * a + b * b)
        .sum()
}

fn fit_stage1_and_stage2_varpro(t: &OnsetTargets<'_>, cfg: KautzFirFitConfig) -> Stage12FitResult {
    let k_lite = cfg.k_cap.min(8);
    let lite_poles =
        if cfg.use_varpro && t.res_16_tot_e > 1.0e-5 * cfg.tgt_tot_e && k_lite >= 1 && t.n >= 128 {
            refine_kautz_poles_kaufman_varpro(
                &t.input_l[..t.n],
                &t.input_r[..t.n],
                &t.tgt_l[..t.n],
                &t.tgt_r[..t.n],
                VarProPoleConfig {
                    sample_rate_hz: cfg.sample_rate_hz,
                    fir_onset_n: t.fir_onset_n,
                    start: t.start16,
                    k_ref: k_lite,
                    r_max: cfg.r_max_phys,
                    max_lm_iters: 24,
                },
                &[],
            )
        } else {
            Vec::new()
        };
    let ref_fit = if !lite_poles.is_empty() {
        let (acc, rl, rr) = fit_joint_kautz_bank(&KautzFitInputs {
            candidates: &lite_poles,
            input_l: &t.input_l[..t.n],
            input_r: &t.input_r[..t.n],
            res_in_l: t.res_16_l,
            res_in_r: t.res_16_r,
            start: t.start16,
            k_limit: cfg.k_cap.min(lite_poles.len()),
            sample_rate_hz: cfg.sample_rate_hz,
            r_max: cfg.r_max_phys,
            lock_prefix_lens: [0, 0],
        });
        let err = eval_pair_energy(&rl, &rr, t.start16, t.n);
        (!acc.is_empty()).then_some(PriorKautzFit {
            kautz: acc,
            res_l: rl,
            res_r: rr,
            err,
        })
    } else {
        None
    };

    let mut locked_poles = lite_poles.clone();
    let lock_lite = locked_poles.len().min(8);
    let k_ext = cfg.k_cap.min(12).saturating_sub(lock_lite).min(4);
    let mut full_fit = None;
    if cfg.use_varpro && k_ext >= 1 && t.n >= 128 {
        let (s1_rl, s1_rr) = ref_fit
            .as_ref()
            .map_or((t.res_16_l, t.res_16_r), |rf| (&rf.res_l, &rf.res_r));
        let s1_e = eval_pair_energy(s1_rl, s1_rr, t.start16, t.n);
        if s1_e > 1.0e-5 * cfg.tgt_tot_e {
            let ext_poles = refine_kautz_poles_kaufman_varpro(
                &t.input_l[..t.n],
                &t.input_r[..t.n],
                s1_rl,
                s1_rr,
                VarProPoleConfig {
                    sample_rate_hz: cfg.sample_rate_hz,
                    fir_onset_n: t.fir_onset_n,
                    start: t.start16,
                    k_ref: k_ext,
                    r_max: cfg.r_max_phys,
                    max_lm_iters: 16,
                },
                &lite_poles,
            );
            for (r, th) in ext_poles {
                if locked_poles.len() < cfg.k_cap.min(12)
                    && !locked_poles
                        .iter()
                        .any(|&(pr, pt)| poles_too_close(pr, pt, r, th))
                {
                    locked_poles.push((r, th));
                }
            }
        }
    }
    let lock_full = locked_poles.len().min(12);
    if lock_full > lock_lite {
        let (acc2, rl2, rr2) = fit_joint_kautz_bank(&KautzFitInputs {
            candidates: &locked_poles,
            input_l: &t.input_l[..t.n],
            input_r: &t.input_r[..t.n],
            res_in_l: t.res_16_l,
            res_in_r: t.res_16_r,
            start: t.start16,
            k_limit: lock_full,
            sample_rate_hz: cfg.sample_rate_hz,
            r_max: cfg.r_max_phys,
            lock_prefix_lens: [lock_lite, lock_full],
        });
        let err2 = eval_pair_energy(&rl2, &rr2, t.start16, t.n);
        if !acc2.is_empty() {
            full_fit = Some(PriorKautzFit {
                kautz: acc2,
                res_l: rl2,
                res_r: rr2,
                err: err2,
            });
        }
    }
    Stage12FitResult {
        ref_fit,
        full_fit,
        locked_poles,
        lock_lite,
        lock_full,
    }
}

fn fit_stage3_sm_and_log_kautz(
    t: &OnsetTargets<'_>,
    cfg: KautzFirFitConfig,
    best_prior: Option<&PriorKautzFit>,
    locked_poles: Vec<(f64, f64)>,
    lock_lite: usize,
    lock_full: usize,
) -> KautzBankTuple {
    let (sm_res_l, sm_res_r) = best_prior.map_or((t.res_16_l, t.res_16_r), |p| {
        (p.res_l.as_slice(), p.res_r.as_slice())
    });
    let blocks = select_multiregime_blocks(sm_res_l, sm_res_r, t.start16, 16_384);
    let (sm_s, sm_e) = blocks
        .iter()
        .copied()
        .max_by(|&(s1, e1), &(s2, e2)| {
            eval_pair_energy(sm_res_l, sm_res_r, s1, e1)
                .total_cmp(&eval_pair_energy(sm_res_l, sm_res_r, s2, e2))
        })
        .unwrap_or((t.start16, t.n.min(t.start16 + 16_384)));
    let sm_len = sm_e.min(sm_s + 16_384);
    let sm_in_s = sm_s.saturating_sub(t.fir_onset_n);
    let sm_in_e = sm_in_s + sm_len.saturating_sub(sm_s);
    let sm_res_l_e: f64 = sm_res_l[sm_s..sm_len].iter().map(|v| v * v).sum();
    let sm_res_r_e: f64 = sm_res_r[sm_s..sm_len].iter().map(|v| v * v).sum();

    let (th_min, th_max) = audible_kautz_theta_bounds(cfg.sample_rate_hz);
    let mut poles = locked_poles;
    let mut sm_poles = Vec::new();
    if sm_res_l_e > 1.0e-6 * cfg.tgt_tot_e && sm_len > sm_s + 24 {
        sm_poles.extend(extract_sm_poles(
            &t.input_l[sm_in_s..sm_in_e],
            &sm_res_l[sm_s..sm_len],
            16,
            4,
        ));
    }
    if sm_res_r_e > 1.0e-6 * cfg.tgt_tot_e && sm_len > sm_s + 24 {
        sm_poles.extend(extract_sm_poles(
            &t.input_r[sm_in_s..sm_in_e],
            &sm_res_r[sm_s..sm_len],
            16,
            4,
        ));
    }
    sm_poles.sort_by(|x, y| y.0.total_cmp(&x.0));
    for (r, th) in sm_poles {
        if r <= cfg.r_max_phys
            && th > th_min
            && th < th_max
            && !poles.iter().any(|&(pr, pt)| poles_too_close(pr, pt, r, th))
        {
            poles.push((r, th));
        }
    }
    if poles.is_empty() || t.res_16_tot_e > 1.0e-5 * cfg.tgt_tot_e {
        append_log_octave_h2_candidates(&mut poles, cfg.sample_rate_hz, cfg.k_cap);
    }

    let (bank_k, bank_rl, bank_rr) = fit_joint_kautz_bank(&KautzFitInputs {
        candidates: &poles,
        input_l: &t.input_l[..t.n],
        input_r: &t.input_r[..t.n],
        res_in_l: t.res_16_l,
        res_in_r: t.res_16_r,
        start: t.start16,
        k_limit: cfg.k_cap,
        sample_rate_hz: cfg.sample_rate_hz,
        r_max: cfg.r_max_phys,
        lock_prefix_lens: [lock_lite, lock_full],
    });
    let bank_e = eval_pair_energy(&bank_rl, &bank_rr, t.start16, t.n);
    if let Some(best) = best_prior
        && (best.err <= bank_e * 1.02 || bank_k.is_empty())
    {
        (best.kautz.clone(), best.res_l.clone(), best.res_r.clone())
    } else {
        (bank_k, bank_rl, bank_rr)
    }
}

fn try_joint_onset_fir_kautz_refine(
    t: &OnsetTargets<'_>,
    cfg: KautzFirFitConfig,
    fir_l: &[f64; NUM_FIR_TAPS_ULTRA],
    fir_r: &[f64; NUM_FIR_TAPS_ULTRA],
    final_kautz: &[KautzPairF64],
    old_e: f64,
) -> Option<KautzFirFitResult> {
    let n = t.n;
    let mut onset_x_l = vec![0.0_f64; n];
    let mut onset_x_r = vec![0.0_f64; n];
    if n > 16 {
        onset_x_l[16..n].copy_from_slice(&t.input_l[..(n - 16)]);
        onset_x_r[16..n].copy_from_slice(&t.input_r[..(n - 16)]);
    }
    let mut res_no_onset_l = t.tgt_l[..n].to_vec();
    let mut res_no_onset_r = t.tgt_r[..n].to_vec();
    for i in 0..n {
        for k in 0..t.fir_onset_n.min(i + 1) {
            res_no_onset_l[i] -= fir_l[k] * t.input_l[i - k];
            res_no_onset_r[i] -= fir_r[k] * t.input_r[i - k];
        }
    }
    let poles_cur: Vec<(f64, f64)> = final_kautz.iter().map(|p| (p.r_low, p.theta)).collect();
    let (re_k, re_rl, re_rr) = fit_joint_kautz_bank(&KautzFitInputs {
        candidates: &poles_cur,
        input_l: &t.input_l[..n],
        input_r: &t.input_r[..n],
        res_in_l: &res_no_onset_l,
        res_in_r: &res_no_onset_r,
        start: t.start16,
        k_limit: final_kautz.len(),
        sample_rate_hz: cfg.sample_rate_hz,
        r_max: cfg.r_max_phys,
        lock_prefix_lens: [final_kautz.len().min(8), final_kautz.len().min(12)],
    });
    let _ = (re_rl, re_rr);
    if re_k.is_empty() {
        return None;
    }
    let mut k_sum_l = vec![0.0_f64; n];
    let mut k_sum_r = vec![0.0_f64; n];
    for p in &re_k {
        let (mr_l, mi_l) = synth_kautz_unit_mode(p.r_low, p.theta, &onset_x_l);
        let (mr_r, mi_r) = synth_kautz_unit_mode(p.r_low, p.theta, &onset_x_r);
        let sc = 2.0 * p.b_gamma;
        for i in 0..n {
            k_sum_l[i] += sc * (p.c_re_l * mr_l[i] - p.c_im_l * mi_l[i]);
            k_sum_r[i] += sc * (p.c_re_r * mr_r[i] - p.c_im_r * mi_r[i]);
        }
    }
    let sub_l: Vec<f64> = (0..n).map(|i| t.tgt_l[i] - k_sum_l[i]).collect();
    let sub_r: Vec<f64> = (0..n).map(|i| t.tgt_r[i] - k_sum_r[i]).collect();
    let new_fir_l = fit_early_fir_taps(&t.input_l[..n], &sub_l, cfg.fir_n);
    let new_fir_r = fit_early_fir_taps(&t.input_r[..n], &sub_r, cfg.fir_n);
    let new_res_l = residual_after_fir(&t.input_l[..n], &sub_l, &new_fir_l, cfg.fir_n);
    let new_res_r = residual_after_fir(&t.input_r[..n], &sub_r, &new_fir_r, cfg.fir_n);
    let new_e = eval_pair_energy(&new_res_l, &new_res_r, cfg.start, n);
    (new_e.is_finite() && new_e < old_e).then_some(KautzFirFitResult {
        kautz: re_k,
        fir_l: new_fir_l,
        fir_r: new_fir_r,
        res_l: new_res_l,
        res_r: new_res_r,
        res_tot_e: new_e,
    })
}

pub(super) fn fit_kautz_and_fir_on_targets(
    input_l: &[f64],
    input_r: &[f64],
    tgt_l: &[f64],
    tgt_r: &[f64],
    cfg: KautzFirFitConfig,
) -> KautzFirFitResult {
    let n = input_l
        .len()
        .min(input_r.len())
        .min(tgt_l.len())
        .min(tgt_r.len());
    let fir_onset_n = cfg.fir_n.min(16);
    let onset_fir_l = fit_early_fir_taps(&input_l[..n], &tgt_l[..n], fir_onset_n);
    let onset_fir_r = fit_early_fir_taps(&input_r[..n], &tgt_r[..n], fir_onset_n);
    let res_16_l = residual_after_fir(&input_l[..n], &tgt_l[..n], &onset_fir_l, fir_onset_n);
    let res_16_r = residual_after_fir(&input_r[..n], &tgt_r[..n], &onset_fir_r, fir_onset_n);
    let start16 = fir_onset_n.min(n / 4);
    let res_16_tot_e = eval_pair_energy(&res_16_l, &res_16_r, start16, n);
    let onset_tgt = OnsetTargets {
        input_l,
        input_r,
        tgt_l,
        tgt_r,
        res_16_l: &res_16_l,
        res_16_r: &res_16_r,
        res_16_tot_e,
        fir_onset_n,
        start16,
        n,
    };

    let s12 = fit_stage1_and_stage2_varpro(&onset_tgt, cfg);
    let early_exit_frac = if cfg.k_cap <= 8 { 0.015 } else { 1.0e-4 };
    let (accepted_kautz, res_after_k_l, res_after_k_r) = if let Some(rf) = &s12.ref_fit
        && rf.err < early_exit_frac * res_16_tot_e
    {
        (rf.kautz.clone(), rf.res_l.clone(), rf.res_r.clone())
    } else if let Some(ff) = &s12.full_fit
        && ff.err < early_exit_frac * res_16_tot_e
    {
        (ff.kautz.clone(), ff.res_l.clone(), ff.res_r.clone())
    } else {
        let best_prior = match (&s12.ref_fit, &s12.full_fit) {
            (Some(r1), Some(r2)) => Some(if r2.err <= r1.err { r2 } else { r1 }),
            (Some(r1), None) => Some(r1),
            (None, Some(r2)) => Some(r2),
            (None, None) => None,
        };
        fit_stage3_sm_and_log_kautz(
            &onset_tgt,
            cfg,
            best_prior,
            s12.locked_poles,
            s12.lock_lite,
            s12.lock_full,
        )
    };

    let mut tgt_for_fir_l = vec![0.0_f64; n];
    let mut tgt_for_fir_r = vec![0.0_f64; n];
    for t in 0..n {
        let mut fl_16 = 0.0_f64;
        let mut fr_16 = 0.0_f64;
        for k in 0..fir_onset_n.min(t + 1) {
            fl_16 += onset_fir_l[k] * input_l[t - k];
            fr_16 += onset_fir_r[k] * input_r[t - k];
        }
        tgt_for_fir_l[t] = res_after_k_l[t] + fl_16;
        tgt_for_fir_r[t] = res_after_k_r[t] + fr_16;
    }

    let fir_l = fit_early_fir_taps(&input_l[..n], &tgt_for_fir_l, cfg.fir_n);
    let fir_r = fit_early_fir_taps(&input_r[..n], &tgt_for_fir_r, cfg.fir_n);
    let res_l = residual_after_fir(&input_l[..n], &tgt_for_fir_l, &fir_l, cfg.fir_n);
    let res_r = residual_after_fir(&input_r[..n], &tgt_for_fir_r, &fir_r, cfg.fir_n);
    let res_tot_e = eval_pair_energy(&res_l, &res_r, cfg.start, n);

    if cfg.use_varpro
        && !accepted_kautz.is_empty()
        && cfg.fir_n > 16
        && let Some(refined) = try_joint_onset_fir_kautz_refine(
            &onset_tgt,
            cfg,
            &fir_l,
            &fir_r,
            &accepted_kautz,
            res_tot_e,
        )
    {
        return refined;
    }

    KautzFirFitResult {
        kautz: accepted_kautz,
        fir_l,
        fir_r,
        res_l,
        res_r,
        res_tot_e,
    }
}

fn eval_series_ff_quad_residual_energy(
    b_l: &[f64],
    b_r: &[f64],
    tgt_l: &[f64],
    tgt_r: &[f64],
    tq: &TransportQuadF64,
    sample_rate_hz: f64,
    start: usize,
) -> f64 {
    let n = b_l.len().min(b_r.len()).min(tgt_l.len()).min(tgt_r.len());
    let (wl, wr, _) = eval_op_t_quad_stereo_with_wet_dc_and_adjoint(
        &b_l[..n],
        &b_r[..n],
        tq,
        sample_rate_hz,
        None,
        PASSIVE_FF_DC_POLE_F64,
    );
    (start..n)
        .map(|t| {
            let dl = tgt_l[t] - (b_l[t] + wl[t]);
            let dr = tgt_r[t] - (b_r[t] + wr[t]);
            dl * dl + dr * dr
        })
        .sum()
}

pub(super) struct AcceptedSeriesReflections {
    pub kautz: Vec<KautzPairF64>,
    pub fir_l: [f64; NUM_FIR_TAPS_ULTRA],
    pub fir_r: [f64; NUM_FIR_TAPS_ULTRA],
    pub res_tot_e: f64,
    pub ff_quad: TransportQuadF64,
    pub delay_samp: f64,
}

struct AltDeconvState<'a> {
    input_l: &'a [f64],
    input_r: &'a [f64],
    target_l: &'a [f64],
    target_r: &'a [f64],
    ff_quad: &'a mut TransportQuadF64,
    deconv_l: &'a mut Vec<f64>,
    deconv_r: &'a mut Vec<f64>,
    cand_lite: &'a mut KautzFirFitResult,
}

fn run_alternating_deconv_loop(
    deconv_lite_cfg: KautzFirFitConfig,
    start: usize,
    sample_rate_hz: f64,
    st: &mut AltDeconvState<'_>,
) {
    let n = st.input_l.len();
    let lags_l: Vec<f64> = [0_usize, 2]
        .into_iter()
        .filter(|&lane| st.ff_quad.input_gains[lane] > 0.0)
        .map(|lane| st.ff_quad.delay_samples[lane])
        .collect();
    let lags_r: Vec<f64> = [1_usize, 3]
        .into_iter()
        .filter(|&lane| st.ff_quad.input_gains[lane] > 0.0)
        .map(|lane| st.ff_quad.delay_samples[lane])
        .collect();

    for _alt in 0..MAX_ALT_DECONV_ITERS {
        let pol_b_l: Vec<f64> = (0..n)
            .map(|t| st.deconv_l[t] - st.cand_lite.res_l[t])
            .collect();
        let pol_b_r: Vec<f64> = (0..n)
            .map(|t| st.deconv_r[t] - st.cand_lite.res_r[t])
            .collect();
        let Some((polished_tq, _)) = solve_feedforward_quad_ls(&FeedforwardQuadLsInputs {
            b_l: &pol_b_l,
            b_r: &pol_b_r,
            target_l: &st.target_l[..n],
            target_r: &st.target_r[..n],
            start,
            sample_rate_hz,
            lags_l: &lags_l,
            lags_r: &lags_r,
        }) else {
            break;
        };
        let max_delta = (0..4)
            .map(|lane| {
                (polished_tq.wet_tap_l[lane] - st.ff_quad.wet_tap_l[lane])
                    .abs()
                    .max((polished_tq.wet_tap_r[lane] - st.ff_quad.wet_tap_r[lane]).abs())
            })
            .fold(0.0_f64, f64::max);
        let Some((pol_deconv_l, pol_deconv_r)) = causal_invert_feedforward_op_t(
            &st.target_l[..n],
            &st.target_r[..n],
            &polished_tq,
            sample_rate_hz,
        ) else {
            break;
        };
        *st.ff_quad = polished_tq;
        *st.deconv_l = pol_deconv_l;
        *st.deconv_r = pol_deconv_r;
        *st.cand_lite = fit_kautz_and_fir_on_targets(
            &st.input_l[..n],
            &st.input_r[..n],
            st.deconv_l,
            st.deconv_r,
            deconv_lite_cfg,
        );
        if max_delta <= ALT_CONVERGE_TOL {
            break;
        }
    }
}

pub(super) fn try_accept_series_ff_reflections(
    input_l: &[f64],
    input_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    op_t_out: &OpTSeedOutcome,
    base_ultra: &KautzFirFitResult,
    cfg: KautzFirFitConfig,
) -> Option<AcceptedSeriesReflections> {
    if op_t_out.mod_detected {
        return None;
    }
    let n = input_l
        .len()
        .min(input_r.len())
        .min(target_l.len())
        .min(target_r.len());
    let lite_fir_n = cfg.fir_n.min(32);
    let lite_k_cap = cfg.k_cap.min(8);
    let same_as_ultra = lite_fir_n == cfg.fir_n && lite_k_cap == cfg.k_cap;
    let lite_cfg = KautzFirFitConfig {
        fir_n: lite_fir_n,
        k_cap: lite_k_cap,
        use_varpro: false,
        ..cfg
    };
    let base_lite_owned = (!same_as_ultra).then(|| {
        fit_kautz_and_fir_on_targets(
            &input_l[..n],
            &input_r[..n],
            &op_t_out.tgt_no_opt_l,
            &op_t_out.tgt_no_opt_r,
            lite_cfg,
        )
    });
    let base_lite = base_lite_owned.as_ref().unwrap_or(base_ultra);
    let (cur_lite_e, cur_ultra_e) = (base_lite.res_tot_e, base_ultra.res_tot_e);
    let raw_lite_owned = (op_t_out.tgt_subtracted || same_as_ultra).then(|| {
        fit_kautz_and_fir_on_targets(
            &input_l[..n],
            &input_r[..n],
            &target_l[..n],
            &target_r[..n],
            lite_cfg,
        )
    });
    let raw_lite = raw_lite_owned.as_ref().unwrap_or(base_lite);
    let b32_l: Vec<f64> = (0..n).map(|t| target_l[t] - raw_lite.res_l[t]).collect();
    let b32_r: Vec<f64> = (0..n).map(|t| target_r[t] - raw_lite.res_r[t]).collect();

    let (mut ff_quad, mut deconv_l, mut deconv_r) = seed_series_feedforward_reflections(
        [
            &input_l[..n],
            &input_r[..n],
            &b32_l,
            &b32_r,
            &raw_lite.res_l,
            &raw_lite.res_r,
            &target_l[..n],
            &target_r[..n],
        ],
        cfg.start,
        cfg.sample_rate_hz,
    )?;

    let deconv_lite_cfg = KautzFirFitConfig {
        use_varpro: same_as_ultra,
        ..lite_cfg
    };
    let mut cand_lite = fit_kautz_and_fir_on_targets(
        &input_l[..n],
        &input_r[..n],
        &deconv_l,
        &deconv_r,
        deconv_lite_cfg,
    );
    let mut alt_st = AltDeconvState {
        input_l: &input_l[..n],
        input_r: &input_r[..n],
        target_l,
        target_r,
        ff_quad: &mut ff_quad,
        deconv_l: &mut deconv_l,
        deconv_r: &mut deconv_r,
        cand_lite: &mut cand_lite,
    };
    run_alternating_deconv_loop(deconv_lite_cfg, cfg.start, cfg.sample_rate_hz, &mut alt_st);

    let b_lite_new_l: Vec<f64> = (0..n).map(|t| deconv_l[t] - cand_lite.res_l[t]).collect();
    let b_lite_new_r: Vec<f64> = (0..n).map(|t| deconv_r[t] - cand_lite.res_r[t]).collect();
    let lite_new_e = eval_series_ff_quad_residual_energy(
        &b_lite_new_l,
        &b_lite_new_r,
        &target_l[..n],
        &target_r[..n],
        &ff_quad,
        cfg.sample_rate_hz,
        cfg.start,
    );
    let (lite_thresh, ultra_thresh) = if op_t_out.tgt_subtracted {
        (0.50 * cur_lite_e, 0.50 * cur_ultra_e)
    } else {
        (0.80 * cur_lite_e, 0.90 * cur_ultra_e)
    };
    if lite_new_e >= lite_thresh {
        return None;
    }

    let cand_ultra = if same_as_ultra {
        cand_lite
    } else {
        fit_kautz_and_fir_on_targets(
            &input_l[..n],
            &input_r[..n],
            &deconv_l,
            &deconv_r,
            KautzFirFitConfig {
                use_varpro: true,
                ..cfg
            },
        )
    };
    let b_ultra_l: Vec<f64> = (0..n).map(|t| deconv_l[t] - cand_ultra.res_l[t]).collect();
    let b_ultra_r: Vec<f64> = (0..n).map(|t| deconv_r[t] - cand_ultra.res_r[t]).collect();
    let ultra_new_e = if same_as_ultra {
        lite_new_e
    } else {
        eval_series_ff_quad_residual_energy(
            &b_ultra_l,
            &b_ultra_r,
            &target_l[..n],
            &target_r[..n],
            &ff_quad,
            cfg.sample_rate_hz,
            cfg.start,
        )
    };
    if ultra_new_e >= ultra_thresh {
        return None;
    }

    let max_delay = (0..4)
        .filter(|&lane| ff_quad.input_gains[lane] > 0.0)
        .map(|lane| ff_quad.delay_samples[lane])
        .fold(0.0_f64, f64::max);
    Some(AcceptedSeriesReflections {
        kautz: cand_ultra.kautz,
        fir_l: cand_ultra.fir_l,
        fir_r: cand_ultra.fir_r,
        res_tot_e: ultra_new_e,
        ff_quad,
        delay_samp: max_delay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_kautz_fir_and_series_reflection_deconv() {
        let sr = 48_000.0_f64;
        let n = 4_096_usize;
        let mut x = vec![0.0_f64; n];
        let mut s = 0x1357_9BDF_u64;
        for v in &mut x {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v = ((s >> 40) as f64) / ((1_u64 << 24) as f64) * 2.0 - 1.0;
        }
        let mut b_y = vec![0.0_f64; n];
        let (mut z1, mut z2) = (0.0_f64, 0.0_f64);
        let r = 0.92_f64;
        let th = core::f64::consts::TAU * 600.0 / sr;
        let a1 = 2.0 * r * th.cos();
        let a2 = r * r;
        for t in 0..n {
            let fir = 0.7 * x[t] - if t >= 2 { 0.3 * x[t - 2] } else { 0.0 };
            let m = x[t] + a1 * z1 - a2 * z2;
            z2 = z1;
            z1 = m;
            b_y[t] = fir + 0.15 * m;
        }
        let tgt_e = eval_pair_energy(&b_y, &b_y, 64, n);
        let cfg = KautzFirFitConfig {
            sample_rate_hz: sr,
            fir_n: 32,
            k_cap: 8,
            start: 64,
            tgt_tot_e: tgt_e,
            r_max_phys: 0.98,
            use_varpro: true,
        };
        let fit = fit_kautz_and_fir_on_targets(&x, &x, &b_y, &b_y, cfg);
        assert!(fit.res_tot_e < 0.05 * tgt_e, "res_e={}", fit.res_tot_e);
        assert!(!fit.kautz.is_empty());

        let mut mod_out = OpTSeedOutcome {
            dominant_delay_samples: 0.0,
            estimated_loop_gain: 0.0,
            incoherent_echo_only_soft_seed: false,
            mod_detected: true,
            tgt_subtracted: false,
            tgt_no_opt_l: b_y.clone(),
            tgt_no_opt_r: b_y.clone(),
        };
        assert!(
            try_accept_series_ff_reflections(&x, &x, &b_y, &b_y, &mod_out, &fit, cfg).is_none()
        );

        let lag = 72_usize;
        let mut y_ref = b_y.clone();
        for t in lag..n {
            y_ref[t] += 0.45 * b_y[t - lag];
        }
        mod_out.mod_detected = false;
        mod_out.tgt_no_opt_l = y_ref.clone();
        mod_out.tgt_no_opt_r = y_ref.clone();
        let base_e = eval_pair_energy(&y_ref, &y_ref, 64, n);
        let ref_cfg = KautzFirFitConfig {
            tgt_tot_e: base_e,
            ..cfg
        };
        let base_fit = fit_kautz_and_fir_on_targets(&x, &x, &y_ref, &y_ref, ref_cfg);
        let acc =
            try_accept_series_ff_reflections(&x, &x, &y_ref, &y_ref, &mod_out, &base_fit, ref_cfg)
                .expect("lag 72 reflection must accept");
        assert!((acc.delay_samp - (lag as f64)).abs() < 1.0);
        assert!(acc.res_tot_e < 0.5 * base_fit.res_tot_e);
    }
}
