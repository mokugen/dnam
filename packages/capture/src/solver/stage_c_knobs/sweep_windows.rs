#![forbid(unsafe_code)]
//! Multi-window trajectory extraction for 1D `Mode A` sweeps and multi-knob `Hadamard` diagonals
//! (`solver::stage_c_knobs::sweep_windows`).

use std::collections::BTreeMap;

use dnam_train::extract_sweep_window_midpoints;

use crate::proj::directory::StmcTrajectoryWindow;
use crate::stimulus::sealing::CORESET_A5_FRAMES_48K;

const FULLBAND_BODY_SAMPLES: usize = 72_000;
const FULLBAND_SETTLE_GAP_SAMPLES: usize = 1_024;
const FULLBAND_ACTIVE_BODY_END: usize = FULLBAND_BODY_SAMPLES - FULLBAND_SETTLE_GAP_SAMPLES;
const FULLBAND_GAP_PROBE_START: usize = FULLBAND_ACTIVE_BODY_END + 124;
const FULLBAND_GAP_PROBE_END: usize = FULLBAND_BODY_SAMPLES - 100;
const FULLBAND_CHIRP_MIN_BOUND: usize = FULLBAND_BODY_SAMPLES + 4_096;
const _: () = assert!(
    FULLBAND_ACTIVE_BODY_END < FULLBAND_GAP_PROBE_START
        && FULLBAND_GAP_PROBE_START < FULLBAND_GAP_PROBE_END
        && FULLBAND_GAP_PROBE_END < FULLBAND_CHIRP_MIN_BOUND
);

/// Extracted sweep interior window `[start_frame .. start_frame + len]` paired with its midpoint
/// physical control vector.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ExtractedSweepWindow {
    pub(super) start_frame: usize,
    pub(super) phys_controls: Vec<f32>,
}

/// Per-stem sweep segmentation inputs passed to [`extract_sweep_windows_for_stem`].
pub(super) struct SweepStemInputs<'a> {
    pub(super) dry_lr: &'a [[f32; 2]],
    pub(super) wet_lr: &'a [[f32; 2]],
    pub(super) n_fit: usize,
    pub(super) n_active: usize,
    pub(super) windows: &'a [StmcTrajectoryWindow],
    pub(super) base_meta_ctrls: &'a [f32],
}

/// Extracts `2..=4` evenly spaced interior windows from a `Mode A` 1D or `Hadamard` sweep stem,
/// delegating RMS activity segmentation to `dnam_train::extract_sweep_window_midpoints` and
/// labeling each window with its midpoint physical control vector.
pub(super) fn extract_sweep_windows_for_stem(
    inputs: &SweepStemInputs<'_>,
    has_multiknob_pairs: &mut Vec<(usize, usize)>,
    out_windows: &mut Vec<ExtractedSweepWindow>,
) -> bool {
    let SweepStemInputs {
        dry_lr,
        wet_lr,
        n_fit,
        n_active,
        windows,
        base_meta_ctrls,
    } = *inputs;
    if windows.len() < 2 || wet_lr.len() < n_fit || dry_lr.len() < n_fit {
        return false;
    }

    let mut first_val: BTreeMap<usize, f32> = BTreeMap::new();
    let mut last_val: BTreeMap<usize, f32> = BTreeMap::new();
    for w in windows {
        if w.ctrl_kind == 0 {
            let p = usize::from(w.control_id);
            if p < base_meta_ctrls.len() {
                first_val.entry(p).or_insert(w.u_phys);
                last_val.insert(p, w.u_phys);
            }
        }
    }

    let swept_ids: Vec<usize> = first_val
        .iter()
        .filter_map(|(&p, &u0)| {
            let u1 = last_val.get(&p).copied().unwrap_or(u0);
            ((u1 - u0).abs() > 1.0e-4).then_some(p)
        })
        .collect();
    if swept_ids.is_empty() {
        return false;
    }

    if swept_ids.len() == 2 {
        has_multiknob_pairs.push((swept_ids[0], swept_ids[1]));
    }

    let mut c_start = base_meta_ctrls.to_vec();
    let mut c_end = base_meta_ctrls.to_vec();
    for &p in &swept_ids {
        c_start[p] = first_val[&p];
        c_end[p] = last_val[&p];
    }

    let wet_bound = n_active.min(wet_lr.len()).min(dry_lr.len());
    let has_chirp_gap = wet_bound >= FULLBAND_CHIRP_MIN_BOUND
        && dry_lr[FULLBAND_GAP_PROBE_START..FULLBAND_GAP_PROBE_END]
            .iter()
            .all(|&[l, r]| l.abs() <= 1.0e-4 && r.abs() <= 1.0e-4);
    if !has_chirp_gap && wet_bound <= CORESET_A5_FRAMES_48K && swept_ids.len() > 2 {
        // Non-stationary 4-phrase CORESET_A5 cannot separate 3+ simultaneously slewing controls
        // across bass-only Q1/Q2 vs Q3/Q4; consume the stem without corrupting 1D Lobatto stops.
        return true;
    }

    let body_cap = if has_chirp_gap {
        FULLBAND_ACTIVE_BODY_END.min(wet_bound)
    } else {
        wet_bound
    };
    let dry_mono: Vec<f64> = dry_lr[..body_cap]
        .iter()
        .map(|&[l, r]| 0.5 * (f64::from(l) + f64::from(r)))
        .collect();
    let hop = (n_fit / 4).max(64).min(body_cap.max(64));
    let rms_segs = extract_sweep_window_midpoints(&dry_mono, hop, 1.0e-5);

    let active_wet_end = wet_lr[..body_cap]
        .iter()
        .rposition(|&[l, r]| l.abs() > 1.0e-7 || r.abs() > 1.0e-7)
        .map_or(body_cap, |idx| idx + 1)
        .max(n_fit)
        .min(body_cap);
    let traj_len = if !has_chirp_gap && wet_bound <= CORESET_A5_FRAMES_48K {
        wet_bound
    } else {
        active_wet_end
    };

    if rms_segs.len() >= 2
        && rms_segs
            .iter()
            .all(|s| (s.end_frame - s.start_frame) >= n_fit)
    {
        for seg in rms_segs.iter().take(4) {
            let off_w = seg
                .midpoint_frame
                .saturating_sub(n_fit / 2)
                .clamp(seg.start_frame, seg.end_frame - n_fit);
            let frac = ((seg.midpoint_frame as f32) / (traj_len.max(1) as f32)).clamp(0.0, 1.0);
            let mut cw = base_meta_ctrls.to_vec();
            for &p in &swept_ids {
                cw[p] = c_start[p] + frac * (c_end[p] - c_start[p]);
            }
            out_windows.push(ExtractedSweepWindow {
                start_frame: off_w,
                phys_controls: cw,
            });
        }
        return true;
    }

    let active_dry_end = rms_segs
        .last()
        .map_or(body_cap, |s| s.end_frame)
        .max(n_fit)
        .min(body_cap);
    let win_bound = active_dry_end.min(active_wet_end).max(n_fit);
    let span = win_bound.saturating_sub(n_fit);
    let n_wins = (win_bound / n_fit).clamp(2, 4);

    for w_idx in 0..n_wins {
        let off_w = if n_wins <= 1 {
            0
        } else {
            (w_idx * span) / (n_wins - 1)
        };
        let mid_sample = (off_w as f32) + 0.5 * (n_fit as f32);
        let frac = (mid_sample / (traj_len.max(1) as f32)).clamp(0.0, 1.0);
        let mut cw = base_meta_ctrls.to_vec();
        for &p in &swept_ids {
            cw[p] = c_start[p] + frac * (c_end[p] - c_start[p]);
        }
        out_windows.push(ExtractedSweepWindow {
            start_frame: off_w,
            phys_controls: cw,
        });
    }

    true
}

/// Decoded uniform-knot continuous control schedule for an enabled `MODE_A_SWEEP` stem
/// (`ref[train.pipeline.dispatch-solve]`, `ref[container.proj.stmc-capsule]`).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ModeASweepSchedule {
    /// Maximum positive-weight record count `M_max = max_c M_c >= 1` across controls.
    pub(super) m_max: usize,
    /// Ordered positive-weight physical/option knots per schema control (`len == P`).
    pub(super) per_ctrl_knots: Vec<Vec<f32>>,
    /// True iff positive-weight records on control `p` represent discrete switch indices.
    pub(super) per_ctrl_is_switch: Vec<bool>,
}

/// Decodes positive-weight `STMC` trajectory records per control `c in 0..P`, holding controls
/// with `<= 1` positive-weight records static at `base_meta_ctrls[c]`.
#[must_use]
pub(super) fn build_mode_a_sweep_schedule(
    windows: &[StmcTrajectoryWindow],
    base_meta_ctrls: &[f32],
) -> ModeASweepSchedule {
    let p_count = base_meta_ctrls.len();
    let mut per_ctrl_knots: Vec<Vec<f32>> = vec![Vec::new(); p_count];
    let mut per_ctrl_is_switch = vec![false; p_count];

    for w in windows {
        let p = usize::from(w.control_id);
        if p >= p_count || !w.window_weight.is_finite() || w.window_weight <= 0.0 {
            continue;
        }
        if w.ctrl_kind == 1 {
            per_ctrl_is_switch[p] = true;
            per_ctrl_knots[p].push(f32::from(w.active_opt));
        } else if w.ctrl_kind == 0 && w.u_phys.is_finite() {
            per_ctrl_knots[p].push(w.u_phys);
        }
    }

    for (p, knots) in per_ctrl_knots.iter_mut().enumerate() {
        if knots.is_empty() {
            let v = base_meta_ctrls[p];
            knots.push(if v.is_finite() { v } else { 0.0 });
        }
    }
    let m_max = per_ctrl_knots
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
        .max(1);
    ModeASweepSchedule {
        m_max,
        per_ctrl_knots,
        per_ctrl_is_switch,
    }
}

/// Evaluates `u(frame_idx)` linearly interpolated across uniform knot positions over `[0..n_active)`.
#[must_use]
pub(super) fn eval_sweep_schedule_at_frame(
    sched: &ModeASweepSchedule,
    frame_idx: usize,
    n_active: usize,
) -> Vec<f32> {
    let u_frac = if n_active <= 1 {
        0.0_f64
    } else {
        (frame_idx.min(n_active - 1) as f64) / ((n_active - 1) as f64)
    };
    sched
        .per_ctrl_knots
        .iter()
        .enumerate()
        .map(|(p, knots)| {
            if knots.len() <= 1 {
                knots.first().copied().unwrap_or(0.0)
            } else {
                let pos = u_frac * ((knots.len() - 1) as f64);
                if sched.per_ctrl_is_switch[p] {
                    knots[(pos.round() as usize).min(knots.len() - 1)]
                } else {
                    let k0 = (pos.floor() as usize).min(knots.len() - 1);
                    let k1 = (k0 + 1).min(knots.len() - 1);
                    let alpha = (pos - (k0 as f64)).clamp(0.0, 1.0);
                    (f64::from(knots[k0]) + alpha * (f64::from(knots[k1]) - f64::from(knots[k0])))
                        as f32
                }
            }
        })
        .collect()
}

/// Samples canonical `M_max` synchronous Pass-4 rows (`window_idx = 0..M_max`) at
/// `t_k = 0` when `M_max == 1` else `floor(window_idx * (N_active - 1) / (M_max - 1))`.
#[must_use]
pub(super) fn sample_synchronous_m_max_controls(
    sched: &ModeASweepSchedule,
    n_active: usize,
) -> Vec<(usize, Vec<f32>)> {
    let m_max = sched.m_max.max(1);
    let mut rows = Vec::with_capacity(m_max);
    for window_idx in 0..m_max {
        let t_k = if m_max == 1 || n_active <= 1 {
            0
        } else {
            (window_idx * (n_active - 1)) / (m_max - 1)
        };
        rows.push((
            window_idx,
            eval_sweep_schedule_at_frame(sched, t_k, n_active),
        ));
    }
    rows
}

/// Decodes a continuous `3 kHz` (`16`-sample tick) physical control trajectory `u(t)` across
/// `[0..n_frames)`.
#[must_use]
pub(super) fn build_3khz_control_trajectory(
    sched: &ModeASweepSchedule,
    n_frames: usize,
) -> Vec<Vec<f32>> {
    let n_ticks = n_frames.div_ceil(16).max(1);
    let last_frame = n_frames.saturating_sub(1);
    let mut traj = Vec::with_capacity(n_ticks);
    for tick in 0..n_ticks {
        let frame_idx = (tick * 16).min(last_frame);
        traj.push(eval_sweep_schedule_at_frame(sched, frame_idx, n_frames));
    }
    traj
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn mk_win(control_id: u16, u_phys: f32) -> StmcTrajectoryWindow {
        StmcTrajectoryWindow {
            control_id,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys,
            window_weight: 1.0,
            emr_velocity: 0.1,
        }
    }

    #[test]
    fn test_sweep_windows_extracts_interior_midpoints_for_1d_and_hadamard_sweeps() {
        let n = 4_096_usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 0.25 * (TAU * 220.0 * (i as f64) / 48_000.0).sin() as f32;
                [s, s]
            })
            .collect();
        let wet_sweep: Vec<[f32; 2]> = dry
            .iter()
            .enumerate()
            .map(|(i, &[l, r])| {
                let g = 0.6 + 0.8 * ((i as f32) / ((n - 1) as f32));
                [g * l, g * r]
            })
            .collect();

        // Multi-knob Hadamard windows (`control_id` 0 and 1 swept in opposite directions)
        let had_windows = vec![
            mk_win(0, -1.0),
            mk_win(0, 1.0),
            mk_win(1, 1.0),
            mk_win(1, -1.0),
        ];
        let mut pairs = Vec::new();
        let mut extracted = Vec::new();
        assert!(extract_sweep_windows_for_stem(
            &SweepStemInputs {
                dry_lr: &dry,
                wet_lr: &wet_sweep,
                n_fit: 1_024,
                n_active: n,
                windows: &had_windows,
                base_meta_ctrls: &[0.0, 0.0],
            },
            &mut pairs,
            &mut extracted,
        ));
        assert_eq!(pairs, vec![(0, 1)]);
        assert_eq!(extracted.len(), 4);
        assert!(
            (extracted[0].phys_controls[0] - (-0.75)).abs() < 1.0e-4
                && (extracted[0].phys_controls[1] - 0.75).abs() < 1.0e-4
        );
        assert!(
            (extracted[3].phys_controls[0] - 0.75).abs() < 1.0e-4
                && (extracted[3].phys_controls[1] - (-0.75)).abs() < 1.0e-4
        );

        // 1D sweep extracts 4 windows without registering pairs
        let mut pairs_1d = Vec::new();
        let mut extracted_1d = Vec::new();
        assert!(extract_sweep_windows_for_stem(
            &SweepStemInputs {
                dry_lr: &dry,
                wet_lr: &wet_sweep,
                n_fit: 1_024,
                n_active: n,
                windows: &had_windows[..2],
                base_meta_ctrls: &[0.0, 0.0],
            },
            &mut pairs_1d,
            &mut extracted_1d,
        ));
        assert!(pairs_1d.is_empty());
        assert_eq!(extracted_1d.len(), 4);

        // 3+ knob Hadamard sweep on short coreset (<= CORESET_A5_FRAMES_48K): consumed cleanly
        let mut three_knob_windows = had_windows;
        three_knob_windows.push(mk_win(2, -1.0));
        three_knob_windows.push(mk_win(2, 1.0));
        let mut pairs_3d = Vec::new();
        let mut extracted_3d = Vec::new();
        assert!(extract_sweep_windows_for_stem(
            &SweepStemInputs {
                dry_lr: &dry,
                wet_lr: &wet_sweep,
                n_fit: 1_024,
                n_active: n,
                windows: &three_knob_windows,
                base_meta_ctrls: &[0.0, 0.0, 0.0],
            },
            &mut pairs_3d,
            &mut extracted_3d,
        ));
        assert!(extracted_3d.is_empty());

        // 3+ knob sweep on long capture (> CORESET_A5_FRAMES_48K) extracts 4 interior windows
        let n_long = CORESET_A5_FRAMES_48K + 1_024;
        let dry_long: Vec<[f32; 2]> = (0..n_long).map(|i| dry[i % dry.len()]).collect();
        let wet_long: Vec<[f32; 2]> = dry_long
            .iter()
            .enumerate()
            .map(|(i, &[l, r])| {
                let g = 0.6 + 0.8 * ((i as f32) / ((n_long - 1) as f32));
                [g * l, g * r]
            })
            .collect();
        assert!(extract_sweep_windows_for_stem(
            &SweepStemInputs {
                dry_lr: &dry_long,
                wet_lr: &wet_long,
                n_fit: 1_024,
                n_active: n_long,
                windows: &three_knob_windows,
                base_meta_ctrls: &[0.0, 0.0, 0.0],
            },
            &mut pairs_3d,
            &mut extracted_3d,
        ));
        assert_eq!(extracted_3d.len(), 4);
    }

    #[test]
    fn test_mode_a_sweep_schedule_synchronous_m_max_and_3khz_interpolation() {
        let mut w_zero = mk_win(0, 99.0);
        w_zero.window_weight = 0.0;
        let wins = vec![
            w_zero,
            mk_win(0, -1.0),
            mk_win(0, 0.0),
            mk_win(0, 1.0),
            mk_win(1, 0.75),
        ];
        let sched = build_mode_a_sweep_schedule(&wins, &[0.0, 0.25, -0.5]);
        assert_eq!(sched.m_max, 3);
        assert_eq!(sched.per_ctrl_knots[0], vec![-1.0, 0.0, 1.0]);
        assert_eq!(sched.per_ctrl_knots[1], vec![0.75]);
        assert_eq!(sched.per_ctrl_knots[2], vec![-0.5]);

        let sync = sample_synchronous_m_max_controls(&sched, 97);
        assert_eq!(sync.len(), 3);
        assert_eq!(sync[0], (0, vec![-1.0, 0.75, -0.5]));
        assert_eq!(sync[1], (1, vec![0.0, 0.75, -0.5]));
        assert_eq!(sync[2], (2, vec![1.0, 0.75, -0.5]));

        let traj = build_3khz_control_trajectory(&sched, 48);
        assert_eq!(traj.len(), 3);
        assert_eq!(traj[0][0], -1.0);
        assert!(traj[1][0] > -1.0 && traj[1][0] < 0.0);
        assert!(traj[2][0] > 0.0 && traj[2][0] < 1.0);
    }
}
