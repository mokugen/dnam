#![forbid(unsafe_code)]
//! Canonical Stage Helpers: Step-0 Variance Gating, 3-Phase Stability Bounds & `v3` Inline `CP1`
//! Control Packing (`solver::stages`).

use dnam_train::pack_cp1_slots_into_wire;
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, Arena3ControlBusWire, Arena3UltraDeltaWire, Cp1InlineDirWire,
};
use dnam_vm::control::Cp1ContributorSlot;
use zerocopy::{FromBytes, FromZeros};

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

/// Encodes `cp1_slots` into `v3` inline `CP1` (`Arena0ControlBusWire` `576 B` base /
/// `Arena3ControlBusWire` `2,112 B` ultra) via `dnam_train::pack_cp1_slots_into_wire`,
/// or zeroes inline `CP1` fields if invalid or overflowing inline budgets.
pub fn write_inline_cp1_if_fits(arena: &mut [u8], cp1_slots: &[Cp1ContributorSlot]) {
    let arena_len = arena.len();
    let has_ultra = arena_len >= ARENA_TIER_BYTES[3];
    let (lite_buf, ultra_buf) = if has_ultra {
        let (l, r) = arena.split_at_mut(ARENA_TIER_BYTES[2]);
        (l, Some(r))
    } else {
        (arena, None)
    };
    let Ok((a0, _)) = Arena0LiteWire::mut_from_prefix(lite_buf) else {
        return;
    };
    let a0_cb = &mut a0.control_bus_lite;
    a0_cb.cp1_dir = Cp1InlineDirWire::new_zeroed();
    a0_cb.cp1_base.fill(0);

    let mut scratch_a3 = Arena3ControlBusWire::new_zeroed();
    let a3_cb = ultra_buf
        .and_then(|u| Arena3UltraDeltaWire::mut_from_prefix(u).ok())
        .map(|(a3, _)| &mut a3.control_bus_ultra_delta)
        .unwrap_or(&mut scratch_a3);
    a3_cb.cp1_ultra.fill(0);

    if cp1_slots.is_empty() {
        return;
    }

    let max_idx = cp1_slots
        .iter()
        .map(|s| {
            let base = usize::from(s.control_idx);
            if s.ctrl_kind == 2 {
                base.max(usize::from(s.aux))
            } else {
                base
            }
        })
        .max()
        .unwrap_or(0);
    let num_controls = max_idx + 1;

    let packed = pack_cp1_slots_into_wire(cp1_slots, num_controls, has_ultra, a0_cb, a3_cb);
    if !matches!(packed, Ok(ref overflow) if overflow.is_empty()) {
        a0_cb.cp1_dir = Cp1InlineDirWire::new_zeroed();
        a0_cb.cp1_base.fill(0);
        a3_cb.cp1_ultra.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let s_switch = Cp1ContributorSlot {
            lanes: 1,
            ctrl_kind: 1,
            aux: 2,
            control_idx: 0,
            ..Cp1ContributorSlot::zero()
        };
        let s_pair = Cp1ContributorSlot {
            lanes: 3,
            ctrl_kind: 2,
            aux: 1,
            control_idx: 0,
            ..Cp1ContributorSlot::zero()
        };
        let s_range = Cp1ContributorSlot {
            lanes: 3,
            ctrl_kind: 0,
            aux: 0xFF,
            control_idx: 0,
            ..Cp1ContributorSlot::zero()
        };
        write_inline_cp1_if_fits(&mut tiny_arena, &[s_switch]);

        let mut ultra_arena = vec![0u8; ARENA_TIER_BYTES[3]];
        write_inline_cp1_if_fits(&mut ultra_arena, &[s_switch, s_pair]);
        let (a0, _) = Arena0LiteWire::ref_from_prefix(&ultra_arena).unwrap();
        assert_eq!(a0.control_bus_lite.cp1_dir.cp1_dir_count.get(), 2);

        let mut bad_slot = s_switch;
        bad_slot.lanes = 0;
        write_inline_cp1_if_fits(&mut ultra_arena, &[bad_slot]);
        let (a0_bad, _) = Arena0LiteWire::ref_from_prefix(&ultra_arena).unwrap();
        assert_eq!(a0_bad.control_bus_lite.cp1_dir.cp1_dir_count.get(), 0);

        // Exceed 576 B inline base budget (`5 * 124 = 620 B > 576 B`) -> clears inline CP1 for Region-4 overflow
        write_inline_cp1_if_fits(&mut ultra_arena, &[s_range; 5]);
        let (a0_ovf, _) = Arena0LiteWire::ref_from_prefix(&ultra_arena).unwrap();
        assert_eq!(a0_ovf.control_bus_lite.cp1_dir.cp1_dir_count.get(), 0);

        // Exceed 8-slot inline directory limit -> clears inline CP1 for Region-4 overflow
        write_inline_cp1_if_fits(&mut ultra_arena, &[s_switch; 9]);
        let (a0_dir_ovf, _) = Arena0LiteWire::ref_from_prefix(&ultra_arena).unwrap();
        assert_eq!(a0_dir_ovf.control_bus_lite.cp1_dir.cp1_dir_count.get(), 0);
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
