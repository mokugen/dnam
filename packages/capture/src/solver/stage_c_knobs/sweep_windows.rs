#![forbid(unsafe_code)]
//! Multi-window trajectory extraction for 1D `Mode A` sweeps and multi-knob `Hadamard` diagonals
//! (`solver::stage_c_knobs::sweep_windows`).

use std::collections::BTreeMap;

use crate::proj::directory::StmcTrajectoryWindow;
use crate::stimulus::sealing::CORESET_A5_FRAMES_48K;

use super::jacobian::{StageTargetJacobian, solve_point_stage_targets};

/// Total `fullband` identification excitation body length before the chirp gap (`[0..72_000)`).
const FULLBAND_BODY_SAMPLES: usize = 72_000;
/// Settle gap reserved at the tail of `fullband` `[0..72_000)` before the `72_000..76_096` silence/chirp region.
const FULLBAND_SETTLE_GAP_SAMPLES: usize = 1_024;
/// Active pink-weighted body end (`70_976`) inside standard `fullband` captures.
const FULLBAND_ACTIVE_BODY_END: usize = FULLBAND_BODY_SAMPLES - FULLBAND_SETTLE_GAP_SAMPLES;
/// Start of the `fullband` silent settle-gap probe window (`71_100`).
const FULLBAND_GAP_PROBE_START: usize = FULLBAND_ACTIVE_BODY_END + 124;
/// End of the `fullband` silent settle-gap probe window (`71_900`).
const FULLBAND_GAP_PROBE_END: usize = FULLBAND_BODY_SAMPLES - 100;
/// Minimum active length (`76_096`) when a `fullband` stem appends a chirp gap tail.
const FULLBAND_CHIRP_MIN_BOUND: usize = FULLBAND_BODY_SAMPLES + 4_096;
const _: () = assert!(
    FULLBAND_ACTIVE_BODY_END < FULLBAND_GAP_PROBE_START
        && FULLBAND_GAP_PROBE_START < FULLBAND_GAP_PROBE_END
        && FULLBAND_GAP_PROBE_END < FULLBAND_CHIRP_MIN_BOUND
);

pub(super) struct SweepExtractCtx<'a> {
    pub(super) arena: &'a [u8],
    pub(super) dry_lr: &'a [[f32; 2]],
    pub(super) wet_lr: &'a [[f32; 2]],
    pub(super) jac: &'a StageTargetJacobian,
    pub(super) n_fit: usize,
    pub(super) n_active: usize,
}

/// Extracts `2..=4` evenly spaced interior windows from a `Mode A` 1D or `Hadamard` sweep stem,
/// labeling each window with its midpoint control vector and registering swept multi-knob pairs.
pub(super) fn try_collect_sweep_windows(
    ctx: &SweepExtractCtx<'_>,
    windows: &[StmcTrajectoryWindow],
    base_meta_ctrls: &[f32],
    has_multiknob_pairs: &mut Vec<(usize, usize)>,
    point_targets: &mut Vec<[f32; 16]>,
    point_ctrls: &mut Vec<Vec<f32>>,
) -> bool {
    if windows.len() < 2 || ctx.wet_lr.len() < ctx.n_fit || ctx.dry_lr.len() < ctx.n_fit {
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

    let wet_bound = ctx.n_active.min(ctx.wet_lr.len()).min(ctx.dry_lr.len());
    let has_chirp_gap = wet_bound >= FULLBAND_CHIRP_MIN_BOUND
        && ctx.dry_lr[FULLBAND_GAP_PROBE_START..FULLBAND_GAP_PROBE_END]
            .iter()
            .all(|&[l, r]| l.abs() <= 1.0e-4 && r.abs() <= 1.0e-4);
    if !has_chirp_gap && wet_bound <= CORESET_A5_FRAMES_48K && swept_ids.len() > 2 {
        // Non-stationary 4-phrase CORESET_A5 cannot separate 3+ simultaneously slewing controls
        // across bass-only Q1/Q2 vs Q3/Q4; consume the stem without corrupting 1D Lobatto stops.
        return true;
    }
    let active_dry_end = ctx.dry_lr[..wet_bound]
        .iter()
        .rposition(|&[l, r]| l.abs() > 1.0e-5 || r.abs() > 1.0e-5)
        .map_or(wet_bound, |idx| idx + 1)
        .max(ctx.n_fit)
        .min(wet_bound);
    let active_wet_end = ctx.wet_lr[..wet_bound]
        .iter()
        .rposition(|&[l, r]| l.abs() > 1.0e-7 || r.abs() > 1.0e-7)
        .map_or(wet_bound, |idx| idx + 1)
        .max(ctx.n_fit)
        .min(wet_bound);
    let traj_len = if !has_chirp_gap && wet_bound <= CORESET_A5_FRAMES_48K {
        wet_bound
    } else {
        active_wet_end
    };
    let win_bound = if has_chirp_gap {
        FULLBAND_ACTIVE_BODY_END
            .min(active_wet_end)
            .min(active_dry_end)
    } else {
        active_dry_end.min(active_wet_end)
    }
    .max(ctx.n_fit);
    let span = win_bound.saturating_sub(ctx.n_fit);
    let n_wins = (win_bound / ctx.n_fit).clamp(2, 4);

    for w_idx in 0..n_wins {
        let off_w = if n_wins <= 1 {
            0
        } else {
            (w_idx * span) / (n_wins - 1)
        };
        let mid_sample = (off_w as f32) + 0.5 * (ctx.n_fit as f32);
        let frac = (mid_sample / (traj_len.max(1) as f32)).clamp(0.0, 1.0);
        let mut cw = base_meta_ctrls.to_vec();
        for &p in &swept_ids {
            cw[p] = c_start[p] + frac * (c_end[p] - c_start[p]);
        }
        point_targets.push(solve_point_stage_targets(
            ctx.arena,
            ctx.dry_lr,
            &ctx.wet_lr[off_w..off_w + ctx.n_fit],
            off_w,
            ctx.jac,
        ));
        point_ctrls.push(cw);
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::schema::DnamProjManifest;
    use crate::solver::stage_b_backbone::solve_stage_b_backbone;
    use crate::solver::stage_c_knobs::jacobian::build_stage_target_jacobian;
    use dnam_engine::stages::weights::ARENA_ULTRA_BYTES;
    use std::f64::consts::TAU;

    #[test]
    fn test_sweep_windows_extracts_interior_midpoints_for_1d_and_hadamard_sweeps() {
        let manifest = DnamProjManifest::default_reference();
        let n = 4_096_usize;
        let dry: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let s = 0.25 * (TAU * 220.0 * (i as f64) / 48_000.0).sin() as f32;
                [s, s]
            })
            .collect();
        let wet_star: Vec<[f32; 2]> = dry.iter().map(|&[l, r]| [0.8 * l, 0.8 * r]).collect();
        let wet_sweep: Vec<[f32; 2]> = dry
            .iter()
            .enumerate()
            .map(|(i, &[l, r])| {
                let g = 0.6 + 0.8 * ((i as f32) / ((n - 1) as f32));
                [g * l, g * r]
            })
            .collect();

        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let _ =
            solve_stage_b_backbone(&manifest, &mut arena, &dry, &wet_star, n, 2, false).unwrap();
        let jac = build_stage_target_jacobian(&arena, false, &dry, 1_024);
        let ctx = SweepExtractCtx {
            arena: &arena,
            dry_lr: &dry,
            wet_lr: &wet_sweep,
            jac: &jac,
            n_fit: 1_024,
            n_active: n,
        };

        // Multi-knob Hadamard windows (`control_id` 0 and 1 swept together in opposite directions)
        let had_windows = vec![
            StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: -1.0,
                window_weight: 1.0,
                emr_velocity: 0.1,
            },
            StmcTrajectoryWindow {
                control_id: 0,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: 1.0,
                window_weight: 1.0,
                emr_velocity: 0.1,
            },
            StmcTrajectoryWindow {
                control_id: 1,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: 1.0,
                window_weight: 1.0,
                emr_velocity: 0.1,
            },
            StmcTrajectoryWindow {
                control_id: 1,
                ctrl_kind: 0,
                active_opt: 0,
                u_phys: -1.0,
                window_weight: 1.0,
                emr_velocity: 0.1,
            },
        ];
        let mut pairs = Vec::new();
        let mut targets = Vec::new();
        let mut ctrls = Vec::new();
        assert!(try_collect_sweep_windows(
            &ctx,
            &had_windows,
            &[0.0, 0.0],
            &mut pairs,
            &mut targets,
            &mut ctrls,
        ));
        assert_eq!(pairs, vec![(0, 1)]);
        assert_eq!(ctrls.len(), 4);
        assert!((ctrls[0][0] - (-0.75)).abs() < 1e-4 && (ctrls[0][1] - 0.75).abs() < 1e-4);
        assert!((ctrls[3][0] - 0.75).abs() < 1e-4 && (ctrls[3][1] - (-0.75)).abs() < 1e-4);

        // 1D sweep (swept_ids.len() == 1) extracts 4 windows without registering pairs
        let one_d_windows = had_windows[..2].to_vec();
        let mut pairs_1d = Vec::new();
        let mut targets_1d = Vec::new();
        let mut ctrls_1d = Vec::new();
        assert!(try_collect_sweep_windows(
            &ctx,
            &one_d_windows,
            &[0.0, 0.0],
            &mut pairs_1d,
            &mut targets_1d,
            &mut ctrls_1d,
        ));
        assert!(pairs_1d.is_empty());
        assert_eq!(ctrls_1d.len(), 4);

        // 3+ knob Hadamard sweep on short coreset (<= CORESET_A5_FRAMES_48K): consumed without pushing rows
        let mut three_knob_windows = had_windows;
        three_knob_windows.push(StmcTrajectoryWindow {
            control_id: 2,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: -1.0,
            window_weight: 1.0,
            emr_velocity: 0.1,
        });
        three_knob_windows.push(StmcTrajectoryWindow {
            control_id: 2,
            ctrl_kind: 0,
            active_opt: 0,
            u_phys: 1.0,
            window_weight: 1.0,
            emr_velocity: 0.1,
        });
        let mut pairs_3d = Vec::new();
        let mut targets_3d = Vec::new();
        let mut ctrls_3d = Vec::new();
        assert!(try_collect_sweep_windows(
            &ctx,
            &three_knob_windows,
            &[0.0, 0.0, 0.0],
            &mut pairs_3d,
            &mut targets_3d,
            &mut ctrls_3d,
        ));
        assert!(ctrls_3d.is_empty());
        assert!(targets_3d.is_empty());

        // 3+ knob sweep on long capture (> CORESET_A5_FRAMES_48K) extracts 4 interior windows normally
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
        let ctx_long = SweepExtractCtx {
            arena: &arena,
            dry_lr: &dry_long,
            wet_lr: &wet_long,
            jac: &jac,
            n_fit: 1_024,
            n_active: n_long,
        };
        assert!(try_collect_sweep_windows(
            &ctx_long,
            &three_knob_windows,
            &[0.0, 0.0, 0.0],
            &mut pairs_3d,
            &mut targets_3d,
            &mut ctrls_3d,
        ));
        assert_eq!(ctrls_3d.len(), 4);
        assert_eq!(targets_3d.len(), 4);
    }
}
