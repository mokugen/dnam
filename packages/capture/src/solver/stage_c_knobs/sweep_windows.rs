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
}
