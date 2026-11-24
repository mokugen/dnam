#![forbid(unsafe_code)]
//! Multi-window trajectory extraction for 1D `Mode A` sweeps and multi-knob `Hadamard` diagonals
//! (`solver::stage_c_knobs::sweep_windows`).

use std::collections::BTreeMap;

use dnam_train::extract_sweep_window_midpoints;

use crate::proj::directory::StmcTrajectoryWindow;
use crate::stimulus::sealing::CORESET_A5_FRAMES_48K;

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
/// evaluating each window's physical controls via [`eval_sweep_schedule_at_frame`].
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

    let mut bounds: BTreeMap<usize, (f32, f32, bool)> = BTreeMap::new();
    for w in windows {
        if !w.window_weight.is_finite() || w.window_weight <= 0.0 {
            continue;
        }
        let p = usize::from(w.control_id);
        if p >= base_meta_ctrls.len() {
            continue;
        }
        let (v, is_cont) = match w.ctrl_kind {
            1 => (f32::from(w.active_opt), false),
            0 if w.u_phys.is_finite() => (w.u_phys, true),
            _ => continue,
        };
        bounds
            .entry(p)
            .and_modify(|(lo, hi, has_cont)| {
                *lo = lo.min(v);
                *hi = hi.max(v);
                *has_cont |= is_cont;
            })
            .or_insert((v, v, is_cont));
    }

    let any_swept = bounds.values().any(|&(lo, hi, _)| (hi - lo) > 1.0e-4);
    if !any_swept {
        return false;
    }

    let continuous_swept_ids: Vec<usize> = bounds
        .iter()
        .filter_map(|(&p, &(lo, hi, is_cont))| (is_cont && (hi - lo) > 1.0e-4).then_some(p))
        .collect();
    if continuous_swept_ids.len() == 2 {
        has_multiknob_pairs.push((continuous_swept_ids[0], continuous_swept_ids[1]));
    }

    let wet_bound = n_active.min(wet_lr.len()).min(dry_lr.len());
    if wet_bound <= CORESET_A5_FRAMES_48K && continuous_swept_ids.len() > 2 {
        // Non-stationary 4-phrase CORESET_A5 cannot separate 3+ simultaneously slewing continuous
        // controls across bass-only Q1/Q2 vs Q3/Q4; consume the stem cleanly.
        return true;
    }

    let sched = build_mode_a_sweep_schedule(windows, base_meta_ctrls);
    let body_cap = wet_bound;
    let dry_mono: Vec<f64> = dry_lr[..body_cap]
        .iter()
        .map(|&[l, r]| 0.5 * (f64::from(l) + f64::from(r)))
        .collect();
    let hop = (n_fit / 4).max(64).min(body_cap.max(64));
    let rms_segs = extract_sweep_window_midpoints(&dry_mono, hop, 1.0e-5);
    let traj_len = wet_bound;

    let valid_segs: Vec<_> = rms_segs
        .iter()
        .filter(|s| s.end_frame.saturating_sub(s.start_frame) >= n_fit)
        .collect();
    if rms_segs.len() >= 2 && valid_segs.len() == rms_segs.len() {
        let k_pick = valid_segs.len().min(4);
        for idx in 0..k_pick {
            let seg_idx = (idx * (valid_segs.len() - 1)) / (k_pick - 1);
            let seg = valid_segs[seg_idx];
            let off_w = seg
                .midpoint_frame
                .saturating_sub(n_fit / 2)
                .clamp(seg.start_frame, seg.end_frame - n_fit);
            out_windows.push(ExtractedSweepWindow {
                start_frame: off_w,
                phys_controls: eval_sweep_schedule_at_frame(&sched, seg.midpoint_frame, traj_len),
            });
        }
        return true;
    }

    let active_dry_end = rms_segs
        .last()
        .map_or(body_cap, |s| s.end_frame)
        .max(n_fit)
        .min(body_cap);
    let win_bound = active_dry_end.max(n_fit);
    let span = win_bound.saturating_sub(n_fit);
    let n_wins = (win_bound / n_fit).clamp(2, 4);

    for w_idx in 0..n_wins {
        let off_w = if n_wins <= 1 {
            0
        } else {
            (w_idx * span) / (n_wins - 1)
        };
        let mid_frame = (off_w + n_fit / 2).min(traj_len.saturating_sub(1));
        out_windows.push(ExtractedSweepWindow {
            start_frame: off_w,
            phys_controls: eval_sweep_schedule_at_frame(&sched, mid_frame, traj_len),
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

/// Evaluates `u(frame_idx)` across exact integer knot positions
/// `t_j = floor(j * (n_active - 1) / (M_c - 1))`, interpolating continuous `ctrl_kind == 0`
/// controls linearly over `[t_j, t_{j+1}]` and flipping discrete `ctrl_kind == 1` switches
/// piecewise-constantly at integer `midpoint = floor((t_j + t_{j+1}) / 2)` (`t < midpoint => j`,
/// `t >= midpoint => j + 1`) (`ref[container.proj.stmc-capsule]`).
#[must_use]
pub(super) fn eval_sweep_schedule_at_frame(
    sched: &ModeASweepSchedule,
    frame_idx: usize,
    n_active: usize,
) -> Vec<f32> {
    let last = n_active.saturating_sub(1);
    let t = frame_idx.min(last);
    sched
        .per_ctrl_knots
        .iter()
        .enumerate()
        .map(|(p, knots)| {
            let m = knots.len();
            if m == 0 {
                0.0
            } else if m == 1 || last == 0 {
                knots[0]
            } else {
                let knot_t = |j: usize| -> usize { (j * last) / (m - 1) };
                let mut j = 0_usize;
                while j + 1 < m - 1 && t >= knot_t(j + 1) {
                    j += 1;
                }
                let t_j = knot_t(j);
                let t_next = knot_t(j + 1);
                if sched.per_ctrl_is_switch[p] {
                    let midpoint = (t_j + t_next) / 2;
                    if t < midpoint { knots[j] } else { knots[j + 1] }
                } else if t_next <= t_j {
                    knots[j + 1]
                } else {
                    let alpha = ((t - t_j) as f64 / (t_next - t_j) as f64).clamp(0.0, 1.0);
                    (f64::from(knots[j]) + alpha * (f64::from(knots[j + 1]) - f64::from(knots[j])))
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
        let expected_w0 = (-1.0 + 2.0 * (512.0 / 4_095.0)) as f32;
        let expected_w3 = (-1.0 + 2.0 * (3_584.0 / 4_095.0)) as f32;
        assert!(
            (extracted[0].phys_controls[0] - expected_w0).abs() < 1.0e-5
                && (extracted[0].phys_controls[1] + expected_w0).abs() < 1.0e-5
        );
        assert!(
            (extracted[3].phys_controls[0] - expected_w3).abs() < 1.0e-5
                && (extracted[3].phys_controls[1] + expected_w3).abs() < 1.0e-5
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
    fn test_mode_a_sweep_schedule_exact_integer_knots_switch_midpoint_and_even_burst_selection() {
        let mut w_zero = mk_win(0, 99.0);
        w_zero.window_weight = 0.0;
        let sw0 = StmcTrajectoryWindow {
            control_id: 2,
            ctrl_kind: 1,
            active_opt: 0,
            u_phys: 99.0,
            window_weight: 1.0,
            emr_velocity: 0.0,
        };
        let sw1 = StmcTrajectoryWindow {
            active_opt: 1,
            ..sw0
        };
        let sw2 = StmcTrajectoryWindow {
            active_opt: 2,
            ..sw0
        };
        let wins = vec![
            w_zero,
            mk_win(0, -1.0),
            mk_win(0, 0.0),
            mk_win(0, 1.0),
            mk_win(1, 0.75),
            sw0,
            sw1,
            sw2,
        ];
        let sched = build_mode_a_sweep_schedule(&wins, &[0.0, 0.25, 0.0]);
        assert_eq!(sched.m_max, 3);
        assert_eq!(sched.per_ctrl_knots[0], vec![-1.0, 0.0, 1.0]);
        assert_eq!(sched.per_ctrl_knots[1], vec![0.75]);
        assert_eq!(sched.per_ctrl_knots[2], vec![0.0, 1.0, 2.0]);

        let sync = sample_synchronous_m_max_controls(&sched, 97);
        assert_eq!(sync.len(), 3);
        assert_eq!(sync[0], (0, vec![-1.0, 0.75, 0.0]));
        assert_eq!(sync[1], (1, vec![0.0, 0.75, 1.0]));
        assert_eq!(sync[2], (2, vec![1.0, 0.75, 2.0]));

        // Switch boundaries for N_active = 97, M_c = 3 => t_0 = 0, t_1 = 48, t_2 = 96
        // Interval [0, 48] midpoint = 24; interval [48, 96] midpoint = 72
        assert_eq!(eval_sweep_schedule_at_frame(&sched, 23, 97)[2], 0.0);
        assert_eq!(eval_sweep_schedule_at_frame(&sched, 24, 97)[2], 1.0);
        assert_eq!(eval_sweep_schedule_at_frame(&sched, 48, 97)[2], 1.0);
        assert_eq!(eval_sweep_schedule_at_frame(&sched, 71, 97)[2], 1.0);
        assert_eq!(eval_sweep_schedule_at_frame(&sched, 72, 97)[2], 2.0);
        assert_eq!(eval_sweep_schedule_at_frame(&sched, 96, 97)[2], 2.0);

        let traj = build_3khz_control_trajectory(&sched, 48);
        assert_eq!(traj.len(), 3);
        assert_eq!(traj[0][0], -1.0);
        assert!(traj[1][0] > -1.0 && traj[1][0] < 0.0);
        assert!(traj[2][0] > 0.0 && traj[2][0] < 1.0);

        // Evenly spaced burst selection across 6 separated active bursts
        let n_seg = 1_600_usize;
        let stride = 2_400_usize;
        let total = 6 * stride;
        let mut dry_gapped = vec![[0.0_f32; 2]; total];
        for b in 0..6 {
            for k in 0..n_seg {
                let s = 0.25 * (TAU * 220.0 * (k as f64) / 48_000.0).sin() as f32;
                dry_gapped[b * stride + k] = [s, s];
            }
        }
        let mut g_pairs = Vec::new();
        let mut g_out = Vec::new();
        assert!(extract_sweep_windows_for_stem(
            &SweepStemInputs {
                dry_lr: &dry_gapped,
                wet_lr: &dry_gapped,
                n_fit: 512,
                n_active: total,
                windows: &[mk_win(0, 0.0), mk_win(0, 1.0)],
                base_meta_ctrls: &[0.0],
            },
            &mut g_pairs,
            &mut g_out,
        ));
        assert_eq!(g_out.len(), 4);
        assert!(g_out[0].start_frame < stride);
        assert!(g_out[3].start_frame >= 5 * stride);

        // Switch-only Mode A round-trip sweep (0 -> 1 -> 0) and 2-range + 1-switch coreset stem
        let sw_only = [0_u8, 1, 0].map(|opt| StmcTrajectoryWindow {
            control_id: 2,
            ctrl_kind: 1,
            active_opt: opt,
            u_phys: f32::from(opt),
            window_weight: 1.0,
            emr_velocity: 0.0,
        });
        let (mut sw_pairs, mut sw_out) = (Vec::new(), Vec::new());
        assert!(extract_sweep_windows_for_stem(
            &SweepStemInputs {
                dry_lr: &dry_gapped,
                wet_lr: &dry_gapped,
                n_fit: 512,
                n_active: total,
                windows: &sw_only,
                base_meta_ctrls: &[0.0, 0.0, 0.0],
            },
            &mut sw_pairs,
            &mut sw_out,
        ));
        assert!(sw_pairs.is_empty() && !sw_out.is_empty());

        let mixed = [
            mk_win(0, 0.0),
            mk_win(0, 1.0),
            mk_win(1, -1.0),
            mk_win(1, 1.0),
            sw_only[0],
            sw_only[1],
            StmcTrajectoryWindow {
                active_opt: 2,
                u_phys: 2.0,
                ..sw_only[2]
            },
        ];
        let (mut mx_pairs, mut mx_out) = (Vec::new(), Vec::new());
        assert!(extract_sweep_windows_for_stem(
            &SweepStemInputs {
                dry_lr: &dry_gapped,
                wet_lr: &dry_gapped,
                n_fit: 512,
                n_active: total,
                windows: &mixed,
                base_meta_ctrls: &[0.0, 0.0, 0.0],
            },
            &mut mx_pairs,
            &mut mx_out,
        ));
        assert_eq!(mx_pairs, vec![(0, 1)]);
        assert!(!mx_out.is_empty());
    }
}
