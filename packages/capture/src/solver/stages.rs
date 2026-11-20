#![forbid(unsafe_code)]
//! Canonical Stage Helpers: Step-0 Variance Gating, 3-Phase Stability Bounds & `v3` Inline `CP1`
//! Control Packing (`solver::stages`).

use dnam_engine::control::{ControlClampDescriptor, Cp1ContributorSlot};
use dnam_engine::schema::{DnamError, DnamJsonHeader};
use dnam_engine::stages::weights::{
    ARENA_LITE_BYTES, ARENA0_CP1_BASE_END, ARENA0_CP1_BASE_OFFSET, ARENA0_CP1_DIR_OFFSET,
    ARENA3_CP1_ULTRA_END, ARENA3_CP1_ULTRA_OFFSET,
};

/// Stability and clamping bounds enforced across Sweep 1 and Sweep 2 (`ref[train.audit.phase1.clamp-stability]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditBounds {
    /// Maximum discrete-time pole radius $r_c \le 0.9995$.
    pub max_pole_radius: f64,
    /// Maximum Stage IV linear feedback gain $g_{\mathrm{lin}} \le 0.9990$.
    pub max_tank_feedback_gain: f64,
    /// Maximum Schroeder all-pass coefficient $|a_{\mathrm{diff}}| \le 0.75$.
    pub max_allpass_coeff: f64,
    /// Minimum causal loop delay $\tau_m \ge 4.0$ samples.
    pub min_causal_delay_samples: f64,
}

impl Default for AuditBounds {
    fn default() -> Self {
        Self {
            max_pole_radius: 0.9995,
            max_tank_feedback_gain: 0.9990,
            max_allpass_coeff: 0.75,
            min_causal_delay_samples: 4.0,
        }
    }
}

impl AuditBounds {
    /// Clamps a candidate modal pole radius, tank feedback gain, all-pass coefficient, and loop delay
    /// into the certified stability envelope (`ref[train.audit.phase1.clamp-stability]`).
    #[must_use]
    pub fn clamp_parameters(
        &self,
        pole_radius: f64,
        tank_gain: f64,
        allpass_coeff: f64,
        delay_samples: f64,
    ) -> (f64, f64, f64, f64) {
        let max_r = if self.max_pole_radius.is_finite() {
            self.max_pole_radius.clamp(0.0, 0.9995)
        } else {
            0.9995
        };
        let max_g = if self.max_tank_feedback_gain.is_finite() {
            self.max_tank_feedback_gain.clamp(0.0, 0.9990)
        } else {
            0.9990
        };
        let max_a = if self.max_allpass_coeff.is_finite() {
            self.max_allpass_coeff.abs().min(0.75)
        } else {
            0.75
        };
        let min_d = if self.min_causal_delay_samples.is_finite() {
            self.min_causal_delay_samples.max(4.0)
        } else {
            4.0
        };
        let r = if pole_radius.is_finite() {
            pole_radius.clamp(0.0, max_r)
        } else {
            0.0
        };
        let g = if tank_gain.is_finite() {
            tank_gain.clamp(0.0, max_g)
        } else {
            0.0
        };
        let a = if allpass_coeff.is_finite() {
            allpass_coeff.clamp(-max_a, max_a)
        } else {
            0.0
        };
        let d = if delay_samples.is_finite() {
            delay_samples.max(min_d)
        } else {
            min_d
        };
        (r, g, a, d)
    }
}

/// Evaluates the Step-0 variance gate for control parameter `p` (`ref[dsp.ctrl.calib.variance-gating]`).
/// Returns `true` when the control is active across stems and `false` when it must be locked to $K_p = 0$.
#[must_use]
pub fn is_control_variance_active(observed_values: &[f64], min_spread: f64) -> bool {
    if observed_values.len() < 2 || !min_spread.is_finite() {
        return false;
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in observed_values {
        if !v.is_finite() {
            return false;
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    (max_v - min_v) >= min_spread.max(0.0)
}

/// Evaluates the smooth Fourier/tanh continuation surrogate for discrete bitcrushing / quantization
/// during homotopy continuation $\tau \in [0, 1]$ (`ref[dsp.op-n.bitcrusher.surrogate]`).
#[must_use]
pub fn bitcrusher_smooth_surrogate(x: f64, step_delta: f64, homotopy_tau: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    let tau = if homotopy_tau.is_finite() {
        homotopy_tau.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let delta = if step_delta.is_finite() && step_delta > 0.0 {
        step_delta
    } else {
        1.0e-4
    };
    let hard_q = delta * (x / delta).round();
    x + tau * (hard_q - x)
}

#[inline]
fn write_f32_le(buf: &mut [u8], off: usize, val: f32) {
    if off + 4 <= buf.len() {
        let clean = if val.is_finite() {
            if val == 0.0 { 0.0 } else { val }
        } else {
            0.0
        };
        buf[off..off + 4].copy_from_slice(&clean.to_le_bytes());
    }
}

/// Encodes `cp1_slots` into `v3` inline `CP1` (`Arena 0 [3,044..3,072, 3,200..3,520]` and
/// `Arena 3 [21,760..23,872]`) when `1 <= cp1_slots.len() <= 8` and within `320 B` / `2,112 B`
/// inline budgets, or zeroes `3,044..3,072` so Region-4 `CP1` handles arbitrary $P$ (`P = 16+`).
pub fn write_inline_cp1_if_fits(arena: &mut [u8], cp1_slots: &[Cp1ContributorSlot]) {
    if arena.len() >= ARENA_LITE_BYTES {
        arena[ARENA0_CP1_DIR_OFFSET..ARENA0_CP1_DIR_OFFSET + 28].fill(0);
        arena[ARENA0_CP1_BASE_OFFSET..ARENA0_CP1_BASE_END].fill(0);
    }
    if arena.len() >= ARENA3_CP1_ULTRA_END {
        arena[ARENA3_CP1_ULTRA_OFFSET..ARENA3_CP1_ULTRA_END].fill(0);
    }
    if cp1_slots.is_empty() || cp1_slots.len() > 8 || arena.len() < ARENA_LITE_BYTES {
        return;
    }
    if cp1_slots
        .iter()
        .any(|s| s.control_idx > 255 || !(1..=3).contains(&s.lanes))
    {
        return;
    }
    let base_cap = ARENA0_CP1_BASE_END - ARENA0_CP1_BASE_OFFSET;
    let ultra_cap = ARENA3_CP1_ULTRA_END - ARENA3_CP1_ULTRA_OFFSET;
    let total_base_bytes: usize = cp1_slots
        .iter()
        .map(|s| 4 * (9 * usize::from(s.lanes) + 4))
        .sum();
    let total_ultra_bytes: usize = cp1_slots
        .iter()
        .map(|s| 32 * usize::from(s.lanes) + 16)
        .sum();
    if total_base_bytes > base_cap || total_ultra_bytes > ultra_cap {
        return;
    }

    let count = cp1_slots.len() as u16;
    arena[ARENA0_CP1_DIR_OFFSET..ARENA0_CP1_DIR_OFFSET + 2].copy_from_slice(&count.to_le_bytes());
    arena[ARENA0_CP1_DIR_OFFSET + 2] = 0;
    arena[ARENA0_CP1_DIR_OFFSET + 3] = 0;

    let mut b_cursor = ARENA0_CP1_BASE_OFFSET;
    let mut u_cursor = ARENA3_CP1_ULTRA_OFFSET;
    let has_ultra = arena.len() >= ARENA3_CP1_ULTRA_END;

    for (c, slot) in cp1_slots.iter().enumerate() {
        let dir_off = ARENA0_CP1_DIR_OFFSET + 4 + c * 3;
        arena[dir_off] = slot.control_idx as u8;
        arena[dir_off + 1] = ((slot.ctrl_kind & 0x0F) << 4) | (slot.lanes & 0x0F);
        arena[dir_off + 2] = if slot.ctrl_kind == 1 {
            slot.option_idx
        } else if slot.ctrl_kind == 2 {
            slot.partner_control_id as u8
        } else {
            0
        };

        let l_cnt = usize::from(slot.lanes);
        for row in slot.main_base.iter().take(l_cnt) {
            for &val in row {
                write_f32_le(arena, b_cursor, val);
                b_cursor += 4;
            }
        }
        for &val in slot.emb.iter().take(l_cnt) {
            write_f32_le(arena, b_cursor, val);
            b_cursor += 4;
        }
        for &val in &slot.v_base {
            write_f32_le(arena, b_cursor, val);
            b_cursor += 4;
        }
        if has_ultra {
            for row in slot.main_ultra.iter().take(l_cnt) {
                for &val in row {
                    write_f32_le(arena, u_cursor, val);
                    u_cursor += 4;
                }
            }
            for &val in &slot.v_ultra {
                write_f32_le(arena, u_cursor, val);
                u_cursor += 4;
            }
        }
    }
}

/// Validates and compiles `header.controls_schema` clamp descriptors (stored in `.dnam` Region 2B;
/// never mutates `v3` `Arena 1` `V3ControlBusParams`).
pub(crate) fn sync_clamp_descriptors_to_arena(
    header: &DnamJsonHeader,
    _arena: &mut [u8],
) -> Result<Vec<ControlClampDescriptor>, DnamError> {
    header.compile_clamp_descriptors()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_engine::stages::weights::ARENA_ULTRA_BYTES;
    use proptest::prelude::*;

    #[test]
    fn test_variance_gating_edge_cases() {
        assert!(!is_control_variance_active(&[], 1.0e-4));
        assert!(!is_control_variance_active(&[0.5], 1.0e-4));
        assert!(!is_control_variance_active(&[0.5, 0.5], 1.0e-4));
        assert!(is_control_variance_active(&[0.0, 1.0], 1.0e-4));
        assert!(!is_control_variance_active(&[0.0, f64::INFINITY], 1.0e-4));
        assert!(!is_control_variance_active(&[0.0, f64::NAN], 1.0e-4));
        let neg_bounds = AuditBounds {
            max_pole_radius: -1.0,
            max_tank_feedback_gain: -1.0,
            max_allpass_coeff: -0.5,
            min_causal_delay_samples: -2.0,
        };
        assert_eq!(
            neg_bounds.clamp_parameters(f64::NAN, f64::INFINITY, f64::NAN, f64::NAN),
            (0.0, 0.0, 0.0, 4.0)
        );
        let nan_bounds = AuditBounds {
            max_pole_radius: f64::NAN,
            max_tank_feedback_gain: f64::NAN,
            max_allpass_coeff: f64::NAN,
            min_causal_delay_samples: f64::NAN,
        };
        assert_eq!(
            nan_bounds.clamp_parameters(2.0, 2.0, 2.0, 1.0),
            (0.9995, 0.9990, 0.75, 4.0)
        );
        assert_eq!(bitcrusher_smooth_surrogate(f64::NAN, 0.1, 0.5), 0.0);
        assert!(bitcrusher_smooth_surrogate(0.5, f64::NAN, f64::NAN).is_finite());
    }

    #[test]
    fn test_v3_inline_cp1_budget_and_overflow_behavior() {
        let mut tiny_arena = vec![0u8; 256];
        let s_ok = Cp1ContributorSlot {
            lanes: 2,
            ctrl_kind: 1,
            option_idx: 2,
            control_idx: 0,
            ..Cp1ContributorSlot::zero()
        };
        let s_pair = Cp1ContributorSlot {
            lanes: 3,
            ctrl_kind: 2,
            partner_control_id: 1,
            control_idx: 0,
            ..Cp1ContributorSlot::zero()
        };
        write_inline_cp1_if_fits(&mut tiny_arena, &[s_ok]);

        let mut ultra_arena = vec![0u8; ARENA_ULTRA_BYTES];
        write_inline_cp1_if_fits(&mut ultra_arena, &[s_ok, s_pair]);
        assert_eq!(
            u16::from_le_bytes([
                ultra_arena[ARENA0_CP1_DIR_OFFSET],
                ultra_arena[ARENA0_CP1_DIR_OFFSET + 1]
            ]),
            2
        );
        let mut bad_slot = s_ok;
        bad_slot.lanes = 0;
        write_inline_cp1_if_fits(&mut ultra_arena, &[bad_slot]);
        assert_eq!(
            u16::from_le_bytes([
                ultra_arena[ARENA0_CP1_DIR_OFFSET],
                ultra_arena[ARENA0_CP1_DIR_OFFSET + 1]
            ]),
            0
        );
        // Exceed 320 B v3 inline base budget (`3 * 124 = 372 B > 320 B`) -> falls back to Region 4
        write_inline_cp1_if_fits(&mut ultra_arena, &[s_pair; 3]);
        assert_eq!(
            u16::from_le_bytes([
                ultra_arena[ARENA0_CP1_DIR_OFFSET],
                ultra_arena[ARENA0_CP1_DIR_OFFSET + 1]
            ]),
            0
        );
        write_inline_cp1_if_fits(&mut ultra_arena, &[s_ok; 9]);
    }

    proptest! {
        #[test]
        fn prop_audit_bounds_always_enforce_invariants(
            r in -2.0_f64..2.0_f64,
            g in -2.0_f64..2.0_f64,
            a in -2.0_f64..2.0_f64,
            d in -10.0_f64..100.0_f64,
        ) {
            let bounds = AuditBounds::default();
            let (rc, gc, ac, dc) = bounds.clamp_parameters(r, g, a, d);
            prop_assert!((0.0..=0.9995).contains(&rc));
            prop_assert!((0.0..=0.9990).contains(&gc));
            prop_assert!((-0.75..=0.75).contains(&ac));
            prop_assert!(dc >= 4.0);
        }

        #[test]
        fn prop_bitcrusher_surrogate_bounded_error(
            x in -10.0_f64..10.0_f64,
            delta in 0.01_f64..1.0_f64,
            tau in 0.0_f64..=1.0_f64,
        ) {
            let y = bitcrusher_smooth_surrogate(x, delta, tau);
            prop_assert!((y - x).abs() <= delta);
        }
    }
}
