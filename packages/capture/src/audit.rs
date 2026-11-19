#![forbid(unsafe_code)]
//! Non-Stop 3-Phase Control-Space Audit Helpers (`dnam_capture::audit`:
//! `ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase1.clamp-switch]`,
//! `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase2.corner-barrier]`,
//! `ref[train.audit.phase2.clamp-rail]`, `ref[train.audit.phase3.clamp-dynamic]`,
//! `ref[train.audit.phase3.clamp-head-gain]`, `ref[train.audit.phase3.numeric-envelope]`,
//! `ref[train.audit.report.prescription]`).

use std::collections::BTreeSet;

use dnam_engine::math::ScalarBackend;
use dnam_engine::stages::hot_core::V3CoreInstance;
use dnam_engine::stages::weights::{
    ARENA_LITE_BYTES, PIPE_FLAG_V3_MATRYOSHKA, has_pipeline_flag, unpack_v3_tier_from_arena,
};
use dnam_engine::{FtzDazGuard, Tier};

pub mod arena;
pub mod manifest;

pub(crate) use arena::{
    ensure_gru_block_zero_input_contraction, ensure_model_overdrive_silence_recovery,
};
pub use arena::{run_full_3phase_arena_audit, run_head_only_3phase_arena_audit};
pub use manifest::{
    recompute_vault_phase1_audit_and_clamps, recompute_vault_phase1_audit_and_clamps_with_windows,
};

/// Phase-1 `[CLAMP-RANGE]`: intersects `declared_range` with `observed_range` and locks `K_p = 0`
/// if `distinct_observed_count <= 1` (`ref[train.audit.phase1.clamp-range]`).
#[inline]
#[must_use]
pub fn audit_phase1_clamp_range(
    declared: (f32, f32),
    observed: (f32, f32),
    distinct_observed_count: usize,
) -> ((f32, f32), u8, bool) {
    let d_lo = if declared.0.is_finite() {
        declared.0
    } else {
        0.0
    };
    let d_hi = if declared.1.is_finite() {
        declared.1.max(d_lo)
    } else {
        d_lo
    };
    let o_lo = if observed.0.is_finite() {
        observed.0
    } else {
        d_lo
    };
    let o_hi = if observed.1.is_finite() {
        observed.1.max(o_lo)
    } else {
        o_lo
    };
    let safe_min = d_lo.clamp(d_lo, d_hi).max(o_lo).min(d_hi);
    let safe_max = d_hi.min(o_hi).max(safe_min);
    let degree = if distinct_observed_count <= 1 || safe_max <= safe_min {
        0u8
    } else {
        (distinct_observed_count - 1).min(3) as u8
    };
    let was_clamped = !declared.0.is_finite()
        || !declared.1.is_finite()
        || !observed.0.is_finite()
        || !observed.1.is_finite()
        || declared.1 < declared.0
        || observed.1 < observed.0
        || safe_min != declared.0
        || safe_max != declared.1;
    ((safe_min, safe_max), degree, was_clamped)
}

/// Phase-1 `[CLAMP-SWITCH]`: restricts `declared_options` to options present in `observed_options`
/// (`ref[train.audit.phase1.clamp-switch]`).
#[must_use]
pub fn audit_phase1_clamp_switch(
    declared_options: &[&str],
    observed_options: &BTreeSet<&str>,
) -> (u16, Vec<String>) {
    let mut mask = 0u16;
    let mut disabled = Vec::new();
    for (idx, &opt) in declared_options.iter().enumerate().take(16) {
        if observed_options.contains(opt) {
            mask |= 1u16 << idx;
        } else {
            disabled.push(opt.to_string());
        }
    }
    (mask, disabled)
}

/// Phase-1 `[CLAMP-STABILITY]` & Phase-3 `[CLAMP-DYNAMIC]` worst-case excursion verifier
/// (`r_worst <= 0.9995`, `g_lin <= 0.9990`, `|a_diff| <= 0.75`, `tau_min >= 4.0`,
/// `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase3.clamp-dynamic]`,
/// `ref[train.audit.phase3.numeric-envelope]`).
#[inline]
#[must_use]
pub fn verify_dynamic_stability_excursion(
    r_high: f32,
    r_mod_l1: f32,
    tau_0: f32,
    tau_mod_l1: f32,
    g_lin: f32,
    a_diff: f32,
) -> (f32, f32, f32, f32, bool) {
    let finite = r_high.is_finite()
        && r_mod_l1.is_finite()
        && tau_0.is_finite()
        && tau_mod_l1.is_finite()
        && g_lin.is_finite()
        && a_diff.is_finite();
    let r_worst = r_high + r_mod_l1.abs();
    let tau_min = tau_0 - tau_mod_l1.abs();
    let ok = finite
        && r_high >= 0.0
        && (0.0..=0.9995).contains(&r_worst)
        && tau_min.is_finite()
        && tau_min >= 4.0
        && (0.0..=0.9990).contains(&g_lin)
        && a_diff.abs() <= 0.75;
    let r_c = if r_worst.is_finite() {
        r_worst.clamp(0.0, 0.9995)
    } else {
        0.0
    };
    let tau_c = if tau_min.is_finite() {
        tau_min.max(4.0)
    } else {
        4.0
    };
    let g_c = if g_lin.is_finite() {
        g_lin.clamp(0.0, 0.9990)
    } else {
        0.0
    };
    let a_c = if a_diff.is_finite() {
        a_diff.clamp(-0.75, 0.75)
    } else {
        0.0
    };
    (r_c, tau_c, g_c, a_c, ok)
}

/// Phase-3 `[CLAMP-HEAD-GAIN]`: clamps rendered head spectral gain to `<= +6.0 dB` (`<= 1.9952623`
/// linear amplitude ratio, `ref[train.audit.phase3.clamp-head-gain]`).
#[inline]
#[must_use]
pub fn clamp_head_spectral_gain_linear(peak_gain_linear: f32) -> (f32, bool) {
    const MAX_GAIN_6DB_LIN: f32 = 1.995_262_3; // 10^(6/20)
    // Stay ~33 ULPs inside +6.0 dB so f32 readout scaling is 0-ULP idempotent under re-audit.
    const MAX_GAIN_IDEMPOTENT: f32 = 1.995_258_3;
    if !peak_gain_linear.is_finite() || peak_gain_linear < 0.0 {
        (0.0, true)
    } else if peak_gain_linear > MAX_GAIN_6DB_LIN {
        (MAX_GAIN_IDEMPOTENT / peak_gain_linear, true)
    } else {
        (1.0, false)
    }
}

/// Formats a consolidated post-training audit & stem prescription banner (`ref[train.audit.report.prescription]`).
#[must_use]
pub fn format_audit_prescription_report(model_name: &str, clamp_lines: &[&str]) -> String {
    let mut out = String::new();
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&format!(
        "DNAM TRAINING COMPLETE — EXPORTED: {model_name}.dnam | {model_name}.dnam-proj\n"
    ));
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&format!(
        "Active Control Clamps Applied ({}):\n",
        clamp_lines.len()
    ));
    for (idx, line) in clamp_lines.iter().enumerate() {
        out.push_str(&format!("  {}. {line}\n", idx + 1));
    }
    out.push_str(
        "================================================================================\n",
    );
    out
}

#[inline]
pub(super) fn read_arena_f32(arena: &[u8], off: usize) -> f32 {
    if off + 4 <= arena.len() {
        f32::from_le_bytes([arena[off], arena[off + 1], arena[off + 2], arena[off + 3]])
    } else {
        0.0
    }
}

#[inline]
pub(super) fn write_arena_f32(arena: &mut [u8], off: usize, val: f32) {
    if off + 4 <= arena.len() {
        arena[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
}

/// Phase-2 Stop-Gradient Corner Barrier Loss $\mathcal{L}_{\mathrm{barrier}}$ evaluated across
/// physical stability bounds ($r_c \le 0.9995$, $g_{\mathrm{lin}} \le 0.9990$,
/// $|a_{\mathrm{diff}}| \le 0.75$, $\tau_m \ge 4.0$; `ref[train.audit.phase2.corner-barrier]`).
#[must_use]
pub fn compute_phase2_corner_barrier_loss(
    r_worst: f64,
    g_lin: f64,
    a_diff: f64,
    tau_min: f64,
) -> f64 {
    if !r_worst.is_finite() || !g_lin.is_finite() || !a_diff.is_finite() || !tau_min.is_finite() {
        return 1.0e6;
    }
    if !(0.0..=0.9995).contains(&r_worst)
        || !(0.0..=0.9990).contains(&g_lin)
        || a_diff.abs() > 0.75
        || tau_min < 4.0
    {
        let over_r = (r_worst - 0.9995).max(0.0) + (-r_worst).max(0.0);
        let over_g = (g_lin - 0.9990).max(0.0) + (-g_lin).max(0.0);
        let over_a = (a_diff.abs() - 0.75).max(0.0);
        let under_tau = (4.0 - tau_min).max(0.0);
        return 1.0e4
            + 1.0e5
                * (over_r * over_r + over_g * over_g + over_a * over_a + under_tau * under_tau);
    }
    let margin_r = (0.9995 - r_worst).max(1.0e-7);
    let margin_g = (0.9990 - g_lin).max(1.0e-7);
    let margin_a = (0.75 - a_diff.abs()).max(1.0e-7);
    let margin_tau = (tau_min - 4.0).max(1.0e-7);
    let barrier = -1.0e-4
        * ((margin_r / 0.01).min(1.0).ln()
            + (margin_g / 0.01).min(1.0).ln()
            + (margin_a / 0.05).min(1.0).ln()
            + (margin_tau / 1.0).min(1.0).ln());
    barrier.max(0.0)
}

/// Phase-2 `[CLAMP-RAIL]` `2,048`-Sample GRU Bounded-State Saturation Audit (`ref[train.audit.phase2.clamp-rail]`).
///
/// Steps `2,048` samples of a multi-tone probe through [`V3CoreInstance`] at `corner_drive`,
/// checking output envelope and GRU hidden-state saturation (`[-2.5, 2.5]`). Returns
/// `(saturation_occupancy_ratio, safe_drive_scale, saturated)` where `saturated == true` when
/// normalized saturation occupancy exceeds `0.95` (`95%`) or any sample is non-finite / `> 100.0`.
#[must_use]
pub fn audit_phase2_gru_state_saturation(
    arena_bytes: &[u8],
    corner_drive: f32,
) -> (f32, f32, bool) {
    const BURST_SAMPLES: usize = 2_048;
    const SAT_OCCUPANCY_LIMIT: f32 = 0.95;

    if !corner_drive.is_finite() || corner_drive <= 0.0 || arena_bytes.len() < ARENA_LITE_BYTES {
        return (1.0, 0.25, true);
    }
    let _ftz = FtzDazGuard::new();
    let cold = if has_pipeline_flag(arena_bytes, PIPE_FLAG_V3_MATRYOSHKA) {
        unpack_v3_tier_from_arena(arena_bytes, Tier::Lite, false)
    } else {
        dnam_engine::stages::stage3_cab::V3ColdTierModel::identity_for_tier(
            Tier::Lite,
            false,
            false,
        )
    };
    let mut inst = V3CoreInstance::<ScalarBackend>::from_cold_model(&cold);
    inst.reset_to_dc_equilibrium();

    let eff_drive = corner_drive.clamp(0.1, 64.0);
    let mut sat_hits = 0usize;
    let mut max_out = 0.0_f32;
    for n in 0..BURST_SAMPLES {
        let t = n as f32;
        let x = 0.85 * eff_drive * ((0.065_449_85 * t).sin() + 0.35 * (0.219_911_49 * t).cos());
        let [yl, yr] = inst.step_stereo(x, -x);
        if !yl.is_finite() || !yr.is_finite() {
            return (1.0, 0.25, true);
        }
        max_out = max_out.max(yl.abs()).max(yr.abs());
        let hot = inst.hot_core();
        let gru_sat = hot
            .state_l
            .h
            .iter()
            .chain(hot.state_r.h.iter())
            .flat_map(|blk| blk.iter().flat_map(|g| g.iter()))
            .any(|&h| h.abs() >= 2.35);
        if gru_sat || yl.abs() >= 0.65 || yr.abs() >= 0.65 || yl.abs() > 25.0 || yr.abs() > 25.0 {
            sat_hits += 1;
        }
    }

    let occupancy = (sat_hits as f32) / (BURST_SAMPLES as f32);
    if occupancy > SAT_OCCUPANCY_LIMIT || max_out > 100.0 {
        let safe_scale = (2.0 / eff_drive.max(2.0)).clamp(0.05, 1.0);
        (occupancy.max(0.96), safe_scale, true)
    } else {
        (occupancy, 1.0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::schema::DnamProjManifest;

    /// verify[train.audit.phase1.clamp-range]
    /// verify[train.audit.phase1.clamp-switch]
    /// verify[train.audit.phase1.clamp-stability]
    /// verify[train.audit.phase2.corner-barrier]
    /// verify[train.audit.phase2.clamp-rail]
    /// verify[train.audit.phase3.clamp-dynamic]
    /// verify[train.audit.phase3.clamp-head-gain]
    /// verify[train.audit.phase3.numeric-envelope]
    /// verify[train.audit.report.prescription]
    #[test]
    fn test_three_phase_control_space_audit_and_prescription_report() {
        let ((s_min, s_max), deg0, clamped) = audit_phase1_clamp_range((0.0, 10.0), (2.0, 8.0), 1);
        assert_eq!((s_min, s_max), (2.0, 8.0));
        assert_eq!(deg0, 0);
        assert!(clamped);

        let (_, deg3, _) = audit_phase1_clamp_range((0.0, 10.0), (0.0, 10.0), 5);
        assert_eq!(deg3, 3);

        let obs_sw = BTreeSet::from(["off", "on"]);
        let (mask, disabled) = audit_phase1_clamp_switch(&["off", "on", "ultra"], &obs_sw);
        assert_eq!(mask, 0b011);
        assert_eq!(disabled, vec!["ultra".to_string()]);

        let (r_c, tau_c, g_c, a_c, ok) =
            verify_dynamic_stability_excursion(0.9990, 0.0020, 3.5, 1.0, 1.02, 0.85);
        assert!(!ok);
        assert_eq!(r_c, 0.9995);
        assert_eq!(tau_c, 4.0);
        assert_eq!(g_c, 0.9990);
        assert_eq!(a_c, 0.75);

        let (scale_ok, clamped_ok) = clamp_head_spectral_gain_linear(1.5);
        assert_eq!((scale_ok, clamped_ok), (1.0, false));
        let (scale_hi, clamped_hi) = clamp_head_spectral_gain_linear(4.0);
        assert!(clamped_hi && scale_hi < 0.5);
        assert_eq!(clamp_head_spectral_gain_linear(f32::NAN), (0.0, true));
        assert_eq!(clamp_head_spectral_gain_linear(-1.0), (0.0, true));
        let (r_n, tau_n, g_n, a_n, ok_n) =
            verify_dynamic_stability_excursion(-10.0, 0.0, f32::INFINITY, 0.0, -1.0, f32::NAN);
        assert!(!ok_n && r_n == 0.0 && tau_n == 4.0 && g_n == 0.0 && a_n == 0.0);
        let (r_nan, _, g_nan, _, ok_nan) =
            verify_dynamic_stability_excursion(f32::NAN, 0.0, 10.0, 0.0, f32::NAN, 0.0);
        assert!(!ok_nan && r_nan == 0.0 && g_nan == 0.0);
        let ((c_lo, c_hi), _, c_nan) = audit_phase1_clamp_range((0.0, 10.0), (f32::NAN, 8.0), 2);
        assert!(c_nan && c_lo.is_finite() && c_hi.is_finite());
        let ((d_lo, d_hi), _, d_nan) =
            audit_phase1_clamp_range((f32::NAN, f32::NAN), (0.0, f32::NAN), 2);
        assert!(d_nan && d_lo == 0.0 && d_hi == 0.0);

        let ((z_lo, z_hi), z_deg, z_clamped) =
            audit_phase1_clamp_range((0.0, 10.0), (-5.0, -2.0), 3);
        assert_eq!((z_lo, z_hi, z_deg, z_clamped), (0.0, 0.0, 0, true));
        let ((inv_lo, inv_hi), inv_deg, inv_clamped) =
            audit_phase1_clamp_range((10.0, 0.0), (10.0, 10.0), 1);
        assert_eq!(
            (inv_lo, inv_hi, inv_deg, inv_clamped),
            (10.0, 10.0, 0, true)
        );
        let (_, narrow_deg, narrow_clamped) =
            audit_phase1_clamp_range((0.0, 1.0e-6), (0.2e-6, 0.8e-6), 2);
        assert_eq!((narrow_deg, narrow_clamped), (1, true));

        let report = format_audit_prescription_report(
            "model",
            &["[CLAMP-RANGE] 'gain': Clamped declared [0.0, 10.0] -> safe [2.0, 8.0]"],
        );
        assert!(report.contains("DNAM TRAINING COMPLETE"));
        assert!(report.contains("Active Control Clamps Applied (1):"));

        let b_safe = compute_phase2_corner_barrier_loss(0.95, 0.85, 0.4, 32.0);
        let b_edge = compute_phase2_corner_barrier_loss(0.9994, 0.9989, 0.749, 4.01);
        let b_viol = compute_phase2_corner_barrier_loss(1.01, 1.02, 0.9, 2.0);
        let b_nan = compute_phase2_corner_barrier_loss(f64::NAN, 0.85, 0.4, 32.0);
        assert!((0.0..1.0).contains(&b_safe));
        assert!(b_edge > b_safe && b_edge < 100.0);
        assert!(b_viol > 1.0e4);
        assert!(b_nan >= 1.0e6);

        use dnam_engine::stages::stage3_cab::V3MatryoshkaModel;
        use dnam_engine::stages::weights::{
            ARENA_ULTRA_BYTES, ARENA0_DIRECT_READOUT_OFFSET, ARENA0_STAGE3_KAUTZ_OFFSET,
            ARENA0_STAGE4_TANK_OFFSET, pack_v3_matryoshka_to_arena,
        };

        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        let v3_id = V3MatryoshkaModel::identity(true, false);
        pack_v3_matryoshka_to_arena(&v3_id, &mut arena);

        let (occ_lo, scale_lo, locked_lo) = audit_phase2_gru_state_saturation(&arena, 0.5);
        assert!(!locked_lo && scale_lo == 1.0 && occ_lo < 0.95);
        let (occ_hi, scale_hi, locked_hi) = audit_phase2_gru_state_saturation(&arena, 12.0);
        assert!(locked_hi && occ_hi > 0.95 && scale_hi < 1.0);
        let (_, _, locked_nan) = audit_phase2_gru_state_saturation(&arena, f32::NAN);
        assert!(locked_nan);

        // Trigger full 3-phase v3 arena audit clamping on out-of-bounds Kautz poles, tank, and readout
        write_arena_f32(&mut arena, ARENA0_STAGE3_KAUTZ_OFFSET, 1.02);
        arena[ARENA0_STAGE4_TANK_OFFSET..ARENA0_STAGE4_TANK_OFFSET + 4]
            .copy_from_slice(&2_u32.to_le_bytes());
        write_arena_f32(&mut arena, ARENA0_STAGE4_TANK_OFFSET + 16, 1.05);
        write_arena_f32(&mut arena, ARENA0_DIRECT_READOUT_OFFSET + 96, 25.0);
        let mut manifest = DnamProjManifest::default_reference();
        run_full_3phase_arena_audit(&mut manifest, &mut arena);
        assert!(read_arena_f32(&arena, ARENA0_STAGE3_KAUTZ_OFFSET) <= 0.9995);
        let clamped_delay = u32::from_le_bytes(
            arena[ARENA0_STAGE4_TANK_OFFSET..ARENA0_STAGE4_TANK_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        assert!(clamped_delay >= 4);
        assert!(read_arena_f32(&arena, ARENA0_STAGE4_TANK_OFFSET + 16) <= 0.9990);
        assert!(read_arena_f32(&arena, ARENA0_DIRECT_READOUT_OFFSET + 96) < 25.0);
        let ar = manifest.audit_report.expect("audit report");
        assert!(ar.phase1_clamps.iter().any(|t| t == "CLAMP-STABILITY"));
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-HEAD-GAIN"));
    }
}
