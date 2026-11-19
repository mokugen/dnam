#![forbid(unsafe_code)]
//! Candidate generation and ranking for top-down FDN cascade (`incoherent_rig::cascade::cands`).

use dnam_engine::stages::V3MatryoshkaModel;
use dnam_engine::stages::stage4_tank::v3_fdn::V3_TANK_MAX_DECLARED_DELAY;

use crate::metrics::certification::PerceptualCertificationSummary;
use crate::solver::SolverQualityMode;

use super::super::comb_detect::{FdnTankSpec, clamp_lfo_depth_for_delays};
use super::tier_passes_all;

fn pos_db_deficit(hi_v: f64, lo_v: f64, margin: f64) -> f64 {
    if hi_v.is_finite() && lo_v.is_finite() {
        ((hi_v + margin) - lo_v).max(0.0)
    } else {
        0.0
    }
}

fn chan_margin_deficit(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
) -> f64 {
    let mut def = 0.0_f64;
    def += pos_db_deficit(hi.raw_besr_db, lo.raw_besr_db, 0.0025);
    for b in 0..3 {
        def += pos_db_deficit(hi.band_esr_db[b], lo.band_esr_db[b], 0.0025);
    }
    def += 10.0 * pos_db_deficit(hi.mr_iir, lo.mr_iir, 0.002);
    def += 5.0 * pos_db_deficit(hi.ltas_max_db, lo.ltas_max_db, 0.0025);
    def += pos_db_deficit(hi.seg_esr_db, lo.seg_esr_db, 0.0025);
    def
}

fn pair_margin_deficit(
    sl: &PerceptualCertificationSummary,
    sr_opt: Option<&PerceptualCertificationSummary>,
    hi_l: &PerceptualCertificationSummary,
    hi_r: Option<&PerceptualCertificationSummary>,
) -> f64 {
    let mut def = chan_margin_deficit(sl, hi_l);
    if let (Some(sr), Some(hr)) = (sr_opt, hi_r) {
        def = def.max(chan_margin_deficit(sr, hr));
    }
    def
}

pub(super) type FallbackEntry = (
    V3MatryoshkaModel,
    FdnTankSpec,
    PerceptualCertificationSummary,
    Option<PerceptualCertificationSummary>,
    f64,
);

pub(super) struct CandRank {
    pub(super) rank: f64,
    pub(super) near_ok: bool,
    pub(super) near_score: f64,
    pub(super) def: f64,
}

#[derive(Default)]
pub(super) struct CandFallbacks {
    pub(super) full: Option<FallbackEntry>,
    pub(super) near: Option<FallbackEntry>,
}

impl CandFallbacks {
    pub(super) fn record(
        &mut self,
        trial: V3MatryoshkaModel,
        tank_spec: FdnTankSpec,
        sl: PerceptualCertificationSummary,
        sr_opt: Option<PerceptualCertificationSummary>,
        sc: CandRank,
        mode: SolverQualityMode,
    ) {
        if tier_passes_all(&sl, sr_opt.as_ref(), mode) {
            if self.full.as_ref().is_none_or(|b| sc.def < b.4) {
                self.full = Some((trial, tank_spec, sl, sr_opt, sc.def));
            }
        } else if sc.near_ok && self.near.as_ref().is_none_or(|b| sc.near_score < b.4) {
            self.near = Some((trial, tank_spec, sl, sr_opt, sc.near_score));
        }
    }
}

pub(super) fn eval_cand_rank(
    sl: &PerceptualCertificationSummary,
    sr_opt: Option<&PerceptualCertificationSummary>,
    hi_l: &PerceptualCertificationSummary,
    hi_r: Option<&PerceptualCertificationSummary>,
    tier_mr_ceil: f64,
    tier_max_ltas: f64,
) -> CandRank {
    let mut mr_over = (sl.mr_iir - tier_mr_ceil).max(0.0);
    let mut seg_over = (sl.seg_esr_db - 6.0).max(0.0);
    let mut mr_def = ((hi_l.mr_iir + 0.002) - sl.mr_iir).max(0.0);
    let mut mr_step = (sl.mr_iir - hi_l.mr_iir).max(0.0);
    let mut ltas_over = (sl.ltas_max_db - tier_max_ltas).max(0.0);
    if let (Some(sr), Some(hr)) = (sr_opt, hi_r) {
        mr_over = mr_over.max((sr.mr_iir - tier_mr_ceil).max(0.0));
        seg_over = seg_over.max((sr.seg_esr_db - 6.0).max(0.0));
        mr_def = mr_def.max(((hr.mr_iir + 0.002) - sr.mr_iir).max(0.0));
        mr_step = mr_step.max((sr.mr_iir - hr.mr_iir).max(0.0));
        ltas_over = ltas_over.max((sr.ltas_max_db - tier_max_ltas).max(0.0));
    }
    let def = pair_margin_deficit(sl, sr_opt, hi_l, hi_r);
    let rank =
        200.0 * mr_over + 50.0 * seg_over + 80.0 * mr_def + 6.0 * ltas_over + def + 0.5 * mr_step;
    CandRank {
        rank,
        near_ok: mr_over <= 0.06 && seg_over <= 1.0,
        near_score: 6.0 * ltas_over + 20.0 * mr_def + 40.0 * mr_over + def,
        def,
    }
}

pub(super) const fn max_tier_delay_for_tau(tau: usize) -> usize {
    match tau {
        2 => 4_608,
        1 => 2_048,
        _ => 1_024,
    }
}

pub(super) fn build_tier_fdn_cands(cur_tank: FdnTankSpec, tau: usize) -> Vec<(FdnTankSpec, usize)> {
    let mut cands: Vec<(FdnTankSpec, usize)> = Vec::with_capacity(10);
    let is_twin_chorus = cur_tank.g_loop == 0.0
        && cur_tank.tap_gains[0] * cur_tank.tap_gains[2] >= 0.0
        && cur_tank.delays[0].abs_diff(cur_tank.delays[2]) <= 48
        && cur_tank.delays[1].abs_diff(cur_tank.delays[3]) <= 48;
    if tau <= 1 && is_twin_chorus {
        let dl_mid =
            ((cur_tank.delays[0] + cur_tank.delays[2]) / 2).clamp(8, V3_TANK_MAX_DECLARED_DELAY);
        let dr_mid =
            ((cur_tank.delays[1] + cur_tank.delays[3]) / 2).clamp(8, V3_TANK_MAX_DECLARED_DELAY);
        for &(off, wr_scale, depth_scale, tap_scale) in &[
            (0_usize, 0.9985_f64, 0.997_f32, 0.9995_f32),
            (0, 0.9965, 0.995, 0.999),
            (0, 0.994, 0.992, 0.998),
            (0, 0.980, 0.980, 0.996),
            (0, 0.960, 0.960, 0.992),
            (1, 0.990, 0.988, 0.997),
            (1, 0.970, 0.970, 0.994),
            (2, 0.985, 0.98, 0.995),
            (4, 0.95, 0.95, 0.99),
            (8, 0.90, 0.90, 0.985),
            (14, 0.78, 0.82, 0.97),
            (0, 0.85, 0.85, 0.98),
        ] {
            let delays = [
                dl_mid
                    .saturating_sub(off)
                    .clamp(8, V3_TANK_MAX_DECLARED_DELAY),
                dr_mid
                    .saturating_sub(off)
                    .clamp(8, V3_TANK_MAX_DECLARED_DELAY),
                (dl_mid + off).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
                (dr_mid + off).clamp(8, V3_TANK_MAX_DECLARED_DELAY),
            ];
            let (rate, depth, phase) = if cur_tank.lfo_depth_samples >= 1.0 {
                let d = clamp_lfo_depth_for_delays(
                    &delays,
                    (cur_tank.lfo_depth_samples * depth_scale).max(6.0),
                );
                if d > 0.0 {
                    (cur_tank.lfo_rate_hz.max(0.5), d, cur_tank.lfo_stereo_phase)
                } else {
                    (0.0, 0.0, 0.0)
                }
            } else {
                (0.0, 0.0, 0.0)
            };
            cands.push((
                FdnTankSpec {
                    delays,
                    g_loop: 0.0,
                    base_psi: 0.0,
                    wet_ratio: (cur_tank.wet_ratio * wr_scale).clamp(0.05, 0.88),
                    tap_gains: cur_tank.tap_gains.map(|g| g * tap_scale),
                    lfo_rate_hz: rate,
                    lfo_depth_samples: depth,
                    lfo_stereo_phase: phase,
                },
                3,
            ));
        }
    } else if !is_twin_chorus {
        let max_cur_d = *cur_tank.delays.iter().max().unwrap_or(&64);
        let max_tier_d = max_tier_delay_for_tau(tau);
        if max_cur_d > max_tier_d {
            let ratio = (max_tier_d as f64) / (max_cur_d as f64);
            let delays = cur_tank
                .delays
                .map(|d| (((d as f64) * ratio).round() as usize).clamp(29, max_tier_d));
            let g_loop = if cur_tank.g_loop > 1.0e-5 {
                cur_tank.g_loop.powf(ratio as f32).clamp(0.0, 0.92)
            } else {
                0.0
            };
            let min_d = if cur_tank.lfo_depth_samples >= 1.5 {
                6.0_f32
            } else {
                0.0
            };
            let depth =
                clamp_lfo_depth_for_delays(&delays, (cur_tank.lfo_depth_samples * 0.85).max(min_d));
            let (rate, eff_depth, phase) = if depth > 0.0 {
                (
                    cur_tank.lfo_rate_hz.max(0.5),
                    depth,
                    cur_tank.lfo_stereo_phase,
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            cands.push((
                FdnTankSpec {
                    delays,
                    g_loop,
                    base_psi: cur_tank.base_psi,
                    wet_ratio: (cur_tank.wet_ratio * 0.84).clamp(0.03, 0.88),
                    tap_gains: cur_tank.tap_gains.map(|g| g * 0.98),
                    lfo_rate_hz: rate,
                    lfo_depth_samples: eff_depth,
                    lfo_stereo_phase: phase,
                },
                3,
            ));
            cands.push((
                FdnTankSpec {
                    delays,
                    g_loop,
                    base_psi: cur_tank.base_psi,
                    wet_ratio: (cur_tank.wet_ratio + 0.028).clamp(0.05, 0.92),
                    tap_gains: cur_tank.tap_gains.map(|g| g * 0.99),
                    lfo_rate_hz: rate,
                    lfo_depth_samples: eff_depth,
                    lfo_stereo_phase: phase,
                },
                3,
            ));

            let compact_d = if tau == 2 {
                [97, 131, 173, 223]
            } else {
                [53, 67, 79, 89]
            };
            let comp_max = *compact_d.iter().max().unwrap_or(&89);
            let ratio_c = (comp_max as f32) / (max_cur_d as f32);
            let g_compact = if cur_tank.g_loop > 0.2 {
                cur_tank.g_loop.powf(ratio_c).clamp(0.55, 0.90)
            } else {
                0.0
            };
            let smear_target = if cur_tank.lfo_depth_samples >= 1.5 {
                8.0_f32
            } else {
                6.0
            };
            let smear_d = clamp_lfo_depth_for_delays(&compact_d, smear_target);
            for (g_try, wr_fac) in [
                (g_compact, 0.68_f64),
                ((cur_tank.g_loop * 0.82).clamp(0.45, 0.82), 0.72_f64),
            ] {
                cands.push((
                    FdnTankSpec {
                        delays: compact_d,
                        g_loop: g_try,
                        base_psi: core::f32::consts::FRAC_PI_4,
                        wet_ratio: (cur_tank.wet_ratio * wr_fac).clamp(0.03, 0.88),
                        tap_gains: [0.65, 0.65, -0.55, 0.55],
                        lfo_rate_hz: if smear_d > 0.0 { 1.35 } else { 0.0 },
                        lfo_depth_samples: smear_d,
                        lfo_stereo_phase: core::f32::consts::FRAC_PI_4,
                    },
                    3,
                ));
            }
        }
    }
    let mut push_cand = |wr_try: f64,
                         delay_add: usize,
                         decay_mult: f32,
                         psi_add: f32,
                         depth_mult: f32,
                         tap_mult: f32,
                         eq_p: usize,
                         zero_loop: bool| {
        let delays = cur_tank
            .delays
            .map(|d| (d + delay_add).clamp(8, V3_TANK_MAX_DECLARED_DELAY));
        let eff_psi_add =
            if zero_loop || (cur_tank.g_loop == 0.0 && cur_tank.base_psi.abs() < 1.0e-5) {
                0.0
            } else {
                psi_add
            };
        let (rate, depth, phase) = if cur_tank.lfo_depth_samples >= 1.0 {
            let raw_d = (cur_tank.lfo_depth_samples * depth_mult).max(6.0);
            let d = clamp_lfo_depth_for_delays(&delays, raw_d);
            if d > 0.0 {
                (cur_tank.lfo_rate_hz.max(0.5), d, cur_tank.lfo_stereo_phase)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        };
        let g_loop = if zero_loop || cur_tank.g_loop <= 1.0e-5 {
            0.0
        } else {
            (1.0 - (1.0 - cur_tank.g_loop) * decay_mult).clamp(0.0, 0.985)
        };
        cands.push((
            FdnTankSpec {
                delays,
                g_loop,
                base_psi: if zero_loop {
                    0.0
                } else {
                    (cur_tank.base_psi + eff_psi_add)
                        .clamp(-core::f32::consts::PI, core::f32::consts::PI)
                },
                wet_ratio: wr_try.clamp(0.01, 0.88),
                tap_gains: cur_tank.tap_gains.map(|g| g * tap_mult),
                lfo_rate_hz: rate,
                lfo_depth_samples: depth,
                lfo_stereo_phase: phase,
            },
            eq_p,
        ));
    };
    push_cand(
        (cur_tank.wet_ratio * 0.985).clamp(0.03, 0.88),
        0,
        1.00,
        0.0,
        0.98,
        0.995,
        3,
        false,
    );
    push_cand(
        (cur_tank.wet_ratio * 0.95).clamp(0.03, 0.88),
        0,
        1.00,
        0.0,
        0.94,
        0.985,
        2,
        false,
    );
    push_cand(
        (cur_tank.wet_ratio * 0.92).clamp(0.03, 0.88),
        1,
        1.02,
        0.0,
        0.88,
        0.98,
        3,
        false,
    );
    push_cand(
        (cur_tank.wet_ratio * 0.88).clamp(0.03, 0.88),
        2,
        1.04,
        0.0,
        0.84,
        0.98,
        3,
        false,
    );
    push_cand(
        (cur_tank.wet_ratio + 0.025).clamp(0.05, 0.88),
        1,
        0.92,
        0.04,
        0.90,
        0.96,
        3,
        false,
    );
    if !is_twin_chorus || cur_tank.g_loop > 0.25 {
        push_cand(
            (cur_tank.wet_ratio + 0.015).clamp(0.05, 0.90),
            2,
            1.05,
            0.05,
            0.95,
            1.00,
            3,
            false,
        );
        push_cand(
            (cur_tank.wet_ratio + 0.032).clamp(0.05, 0.92),
            5,
            1.10,
            0.09,
            0.92,
            1.00,
            3,
            false,
        );
        push_cand(
            (cur_tank.wet_ratio + 0.055).clamp(0.05, 0.94),
            9,
            1.18,
            0.14,
            0.88,
            1.00,
            3,
            false,
        );
    }
    push_cand(
        (cur_tank.wet_ratio * 0.35).clamp(0.01, 0.88),
        2,
        1.00,
        0.0,
        0.75,
        0.90,
        3,
        true,
    );
    cands
}
