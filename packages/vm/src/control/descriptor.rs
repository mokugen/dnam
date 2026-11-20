//! Switch-conditioned `64 B` control normalization & safety-clamp descriptor
//! ([`ControlClampDescriptor`] and [`ConditionalClampRule`], `ref[dsp.ctrl.range.normalize-clamp]`).

use crate::bin::{ConditionalClampRuleWire, ControlClampDescriptorWire};
use crate::math::rcp_nr6_f64;

/// Native representation of a switch-coupled conditional clamp rule (`8 B`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConditionalClampRule {
    /// Conditioning `switch` control index (`0..num_controls`).
    pub switch_ctrl_idx: u16,
    /// Conditioning switch option index (`0..=15`).
    pub switch_opt_idx: u8,
    /// Rule opcode (`0=none, 1=max_ceil, 2=min_floor, 3=and_prefix, 5=and_max_keep, 6=and_min_end`).
    pub rule_op: u8,
    /// Conditional physical bound value.
    pub bound_val: f32,
}

impl ConditionalClampRule {
    /// Converts from a `zerocopy` little-endian wire struct.
    #[inline]
    pub const fn from_wire(w: ConditionalClampRuleWire) -> Self {
        Self {
            switch_ctrl_idx: w.switch_ctrl_idx.get(),
            switch_opt_idx: w.switch_opt_idx,
            rule_op: w.rule_op,
            bound_val: w.bound_val.get(),
        }
    }

    /// Converts into a `zerocopy` little-endian wire struct.
    #[inline]
    pub fn to_wire(self) -> ConditionalClampRuleWire {
        ConditionalClampRuleWire {
            switch_ctrl_idx: zerocopy::byteorder::little_endian::U16::new(self.switch_ctrl_idx),
            switch_opt_idx: self.switch_opt_idx,
            rule_op: self.rule_op,
            bound_val: zerocopy::byteorder::little_endian::F32::new(self.bound_val),
        }
    }
}

/// Computes `binary64` pre-inverted normalization `(scale_norm, offset_norm)` for `[safe_min, safe_max]`
/// using `rcp_nr6_f64` (`0 FDIV`, `ref[dsp.ctrl.range.normalize-clamp]`).
#[inline]
pub fn compute_scale_offset_norm(safe_min: f32, safe_max: f32) -> (f32, f32) {
    if !safe_min.is_finite() || !safe_max.is_finite() || safe_max <= safe_min {
        return (0.0, 0.0);
    }
    let min64 = f64::from(safe_min);
    let max64 = f64::from(safe_max);
    let diff = max64 - min64;
    let inv_diff = rcp_nr6_f64(diff);
    let scale = (2.0 * inv_diff) as f32;
    let offset = (-(max64 + min64) * inv_diff) as f32;
    (scale, offset)
}

/// 64-byte aligned runtime control normalization, clamping, and Lobatto descriptor.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlClampDescriptor {
    /// Control kind (`0 = range`, `1 = switch`, `0xFF = unused`).
    pub ctrl_kind: u8,
    /// Lobatto degree (`0..=3` for `range`) or `num_options - 1` (`0..=15` for `switch`).
    pub lobatto_degree: u8,
    /// Bitmask of valid options (`switch`).
    pub active_options_mask: u16,
    /// Default physical value (`range`) or option index (`switch`).
    pub default_val: f32,
    /// Base lower physical limit.
    pub safe_min_base: f32,
    /// Base upper physical limit.
    pub safe_max_base: f32,
    /// Pre-inverted normalization scale (`2 / (max - min)`).
    pub scale_norm: f32,
    /// Precomputed normalization offset (`-(max + min) / (max - min)`).
    pub offset_norm: f32,
    /// Normalized Lobatto anchors `[a1, a2, a3]`.
    pub anchor_nodes_norm: [f32; 3],
    /// Reserved zero word.
    pub _reserved: u32,
    /// Three switch-coupled conditional clamp rules.
    pub cond_rules: [ConditionalClampRule; 3],
}

const _: () = assert!(core::mem::size_of::<ControlClampDescriptor>() == 64);
const _: () = assert!(core::mem::align_of::<ControlClampDescriptor>() == 64);

impl Default for ControlClampDescriptor {
    fn default() -> Self {
        Self::empty()
    }
}

impl ControlClampDescriptor {
    /// Returns an inactive/unused descriptor (`ctrl_kind = 0xFF`).
    #[inline]
    pub const fn empty() -> Self {
        Self {
            ctrl_kind: 0xFF,
            lobatto_degree: 0,
            active_options_mask: 0,
            default_val: 0.0,
            safe_min_base: 0.0,
            safe_max_base: 0.0,
            scale_norm: 0.0,
            offset_norm: 0.0,
            anchor_nodes_norm: [0.0; 3],
            _reserved: 0,
            cond_rules: [
                ConditionalClampRule {
                    switch_ctrl_idx: 0,
                    switch_opt_idx: 0,
                    rule_op: 0,
                    bound_val: 0.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 0,
                    switch_opt_idx: 0,
                    rule_op: 0,
                    bound_val: 0.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 0,
                    switch_opt_idx: 0,
                    rule_op: 0,
                    bound_val: 0.0,
                },
            ],
        }
    }

    /// Constructs a continuous range descriptor on `[safe_min, safe_max]`.
    pub fn range(
        safe_min: f32,
        safe_max: f32,
        default_val: f32,
        lobatto_degree: u8,
        anchor_nodes_norm: [f32; 3],
    ) -> Self {
        let (scale_norm, offset_norm) = compute_scale_offset_norm(safe_min, safe_max);
        Self {
            ctrl_kind: 0,
            lobatto_degree: lobatto_degree.min(3),
            active_options_mask: 0,
            default_val,
            safe_min_base: safe_min,
            safe_max_base: safe_max,
            scale_norm,
            offset_norm,
            anchor_nodes_norm,
            _reserved: 0,
            cond_rules: Self::empty().cond_rules,
        }
    }

    /// Constructs a discrete switch descriptor with `num_options <= 16` and valid option bitmask.
    pub const fn switch(num_options: u8, active_options_mask: u16, default_opt: u8) -> Self {
        let clamped_opts = if num_options == 0 {
            1
        } else if num_options > 16 {
            16
        } else {
            num_options
        };
        Self {
            ctrl_kind: 1,
            lobatto_degree: clamped_opts - 1,
            active_options_mask,
            default_val: (default_opt & 15) as f32,
            safe_min_base: 0.0,
            safe_max_base: (clamped_opts - 1) as f32,
            scale_norm: 0.0,
            offset_norm: 0.0,
            anchor_nodes_norm: [0.0; 3],
            _reserved: 0,
            cond_rules: Self::empty().cond_rules,
        }
    }

    /// Converts from a little-endian wire descriptor, recomputing cancellation-safe `(scale, offset)`
    /// if wire `scale_norm` is `0.0` on a non-degenerate range.
    pub fn from_wire(w: &ControlClampDescriptorWire) -> Self {
        let mut out = Self {
            ctrl_kind: w.ctrl_kind,
            lobatto_degree: w.lobatto_degree,
            active_options_mask: w.active_options_mask.get(),
            default_val: w.default_val.get(),
            safe_min_base: w.safe_min_base.get(),
            safe_max_base: w.safe_max_base.get(),
            scale_norm: w.scale_norm.get(),
            offset_norm: w.offset_norm.get(),
            anchor_nodes_norm: [
                w.anchor_nodes_norm[0].get(),
                w.anchor_nodes_norm[1].get(),
                w.anchor_nodes_norm[2].get(),
            ],
            _reserved: w._reserved.get(),
            cond_rules: [
                ConditionalClampRule::from_wire(w.cond_rules[0]),
                ConditionalClampRule::from_wire(w.cond_rules[1]),
                ConditionalClampRule::from_wire(w.cond_rules[2]),
            ],
        };
        if out.ctrl_kind == 0
            && out.safe_max_base > out.safe_min_base
            && (out.scale_norm == 0.0
                || !out.scale_norm.is_finite()
                || !out.offset_norm.is_finite())
        {
            let (s, o) = compute_scale_offset_norm(out.safe_min_base, out.safe_max_base);
            out.scale_norm = s;
            out.offset_norm = o;
        }
        out
    }

    /// Converts into a little-endian wire descriptor.
    pub fn to_wire(&self) -> ControlClampDescriptorWire {
        ControlClampDescriptorWire {
            ctrl_kind: self.ctrl_kind,
            lobatto_degree: self.lobatto_degree,
            active_options_mask: zerocopy::byteorder::little_endian::U16::new(
                self.active_options_mask,
            ),
            default_val: zerocopy::byteorder::little_endian::F32::new(self.default_val),
            safe_min_base: zerocopy::byteorder::little_endian::F32::new(self.safe_min_base),
            safe_max_base: zerocopy::byteorder::little_endian::F32::new(self.safe_max_base),
            scale_norm: zerocopy::byteorder::little_endian::F32::new(self.scale_norm),
            offset_norm: zerocopy::byteorder::little_endian::F32::new(self.offset_norm),
            anchor_nodes_norm: [
                zerocopy::byteorder::little_endian::F32::new(self.anchor_nodes_norm[0]),
                zerocopy::byteorder::little_endian::F32::new(self.anchor_nodes_norm[1]),
                zerocopy::byteorder::little_endian::F32::new(self.anchor_nodes_norm[2]),
            ],
            _reserved: zerocopy::byteorder::little_endian::U32::new(self._reserved),
            cond_rules: [
                self.cond_rules[0].to_wire(),
                self.cond_rules[1].to_wire(),
                self.cond_rules[2].to_wire(),
            ],
        }
    }

    /// Computes effective `[eff_min, eff_max]` after intersecting active conditional clamp rules.
    pub fn effective_bounds(&self, active_switch_options: &[u8]) -> (f32, f32) {
        let mut eff_min = self.safe_min_base;
        let mut eff_max = self.safe_max_base;
        let mut conj_active = true;
        for rule in &self.cond_rules {
            if rule.rule_op == 0 {
                conj_active = true;
                continue;
            }
            let sw_matches = active_switch_options
                .get(usize::from(rule.switch_ctrl_idx))
                .is_some_and(|&opt| opt == rule.switch_opt_idx);
            match rule.rule_op {
                1 | 6 => {
                    if conj_active && sw_matches {
                        if rule.rule_op == 1 {
                            eff_max = eff_max.min(rule.bound_val);
                        } else {
                            eff_min = eff_min.max(rule.bound_val);
                        }
                    }
                    conj_active = true;
                }
                2 => {
                    if conj_active && sw_matches {
                        eff_min = eff_min.max(rule.bound_val);
                    }
                    conj_active = true;
                }
                3 => {
                    conj_active = conj_active && sw_matches;
                }
                5 => {
                    let hit = conj_active && sw_matches;
                    if hit {
                        eff_max = eff_max.min(rule.bound_val);
                    }
                    conj_active = hit;
                }
                _ => {
                    conj_active = true;
                }
            }
        }
        if eff_max < eff_min {
            eff_max = eff_min;
        }
        (eff_min, eff_max)
    }

    /// Clamps and normalizes `raw_physical_val` according to `ctrl_kind` and active switch options,
    /// returning `(clamped_physical, u_norm, was_clamped)`.
    pub fn clamp_and_normalize(
        &self,
        raw_physical_val: f32,
        active_switch_options: &[u8],
    ) -> (f32, f32, bool) {
        if self.ctrl_kind == 0xFF {
            return (0.0, 0.0, false);
        }
        if self.ctrl_kind == 1 {
            if raw_physical_val.is_finite() {
                let pos = if raw_physical_val > 0.0 {
                    raw_physical_val
                } else {
                    0.0
                };
                let rounded = crate::math::floor_f32(pos + 0.5);
                if rounded < 16.0 {
                    let opt = rounded as u8;
                    let bit = 1_u16.wrapping_shl(u32::from(opt));
                    if (self.active_options_mask & bit) != 0 {
                        let f = f32::from(opt);
                        return (f, f, false);
                    }
                }
            }
            let def_opt = if self.default_val.is_finite() {
                let pos = if self.default_val > 0.0 {
                    self.default_val
                } else {
                    0.0
                };
                (crate::math::floor_f32(pos + 0.5) as u8).min(15)
            } else {
                0
            };
            let def_bit = 1_u16.checked_shl(u32::from(def_opt)).unwrap_or(0);
            let safe_opt = if (self.active_options_mask & def_bit) != 0 {
                def_opt
            } else if self.active_options_mask != 0 {
                (self.active_options_mask.trailing_zeros() as u8).min(15)
            } else {
                0
            };
            let f = f32::from(safe_opt);
            return (f, f, true);
        }

        let (eff_min, eff_max) = self.effective_bounds(active_switch_options);
        let (clamped, was_clamped) = if !raw_physical_val.is_finite() {
            let safe_def = if self.default_val.is_finite() {
                self.default_val
            } else {
                eff_min
            };
            (safe_def.clamp(eff_min, eff_max), true)
        } else {
            let c = raw_physical_val.clamp(eff_min, eff_max);
            let diff = if c >= raw_physical_val {
                c - raw_physical_val
            } else {
                raw_physical_val - c
            };
            (c, diff > 1.0e-6)
        };
        let u_norm = (clamped * self.scale_norm + self.offset_norm).clamp(-1.0, 1.0);
        (clamped, u_norm, was_clamped)
    }

    /// Evaluates the Node-Anchored Newton/Lobatto polynomial basis (`ref[dsp.ctrl.basis.lobatto]`).
    #[inline]
    pub fn eval_lobatto_basis(&self, u_p: f32) -> [f32; 3] {
        if self.ctrl_kind != 0 || self.lobatto_degree == 0 {
            return [0.0; 3];
        }
        let psi1 = u_p - self.anchor_nodes_norm[0];
        let psi2 = if self.lobatto_degree >= 2 {
            psi1 * (u_p - self.anchor_nodes_norm[1])
        } else {
            0.0
        };
        let psi3 = if self.lobatto_degree >= 3 {
            psi2 * (u_p - self.anchor_nodes_norm[2])
        } else {
            0.0
        };
        [psi1, psi2, psi3]
    }

    /// Evaluates two-pass (`switch` first, then `range`) conditional clamping across `num_controls`.
    pub fn clamp_and_normalize_all(
        descriptors: &[Self],
        num_controls: usize,
        raw_values: &[f32],
        clamped_values: &mut [f32],
        normalized_values: &mut [f32],
        active_switch_options: &mut [u8],
    ) -> u16 {
        let mut clamp_mask: u16 = 0;
        let n = num_controls
            .min(descriptors.len())
            .min(raw_values.len())
            .min(clamped_values.len())
            .min(normalized_values.len())
            .min(active_switch_options.len());
        for p in 0..n {
            if descriptors[p].ctrl_kind == 1 {
                let (c, u, was) = descriptors[p].clamp_and_normalize(raw_values[p], &[]);
                clamped_values[p] = c;
                normalized_values[p] = u;
                active_switch_options[p] = (c as u8).min(15);
                if was && p < 16 {
                    clamp_mask |= 1_u16 << p;
                }
            }
        }
        for p in 0..n {
            if descriptors[p].ctrl_kind != 1 {
                let (c, u, was) =
                    descriptors[p].clamp_and_normalize(raw_values[p], &active_switch_options[..n]);
                clamped_values[p] = c;
                normalized_values[p] = u;
                if was && p < 16 {
                    clamp_mask |= 1_u16 << p;
                }
            }
        }
        clamp_mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switch_before_range_conditional_clamping_and_lobatto() {
        let sw = ControlClampDescriptor::switch(3, 0b101, 0);
        let mut rng = ControlClampDescriptor::range(0.0, 10.0, 5.0, 3, [0.0, -1.0, 1.0]);
        rng.cond_rules[0] = ConditionalClampRule {
            switch_ctrl_idx: 0,
            switch_opt_idx: 2,
            rule_op: 3, // and_prefix
            bound_val: 0.0,
        };
        rng.cond_rules[1] = ConditionalClampRule {
            switch_ctrl_idx: 0,
            switch_opt_idx: 2,
            rule_op: 1, // max_ceil
            bound_val: 4.0,
        };
        let descs = [sw, rng];
        let raw = [2.0, 8.0];
        let mut clamped = [0.0; 2];
        let mut norm = [0.0; 2];
        let mut opts = [0; 2];
        let mask = ControlClampDescriptor::clamp_and_normalize_all(
            &descs,
            2,
            &raw,
            &mut clamped,
            &mut norm,
            &mut opts,
        );
        assert_eq!(opts[0], 2);
        assert_eq!(clamped[1], 4.0);
        assert_eq!(mask, 0b10);
        assert_eq!(descs[1].eval_lobatto_basis(0.0), [0.0, 0.0, 0.0]);
        let round = ControlClampDescriptor::from_wire(&rng.to_wire());
        assert_eq!(round.safe_min_base, 0.0);

        // Zero scale_norm recompute, conditional rule ops 2, 5, 6, eff_max < eff_min collapse,
        // non-finite handling, switch fallbacks, and unused kind 0xFF.
        let mut wire0 = rng.to_wire();
        wire0.scale_norm = zerocopy::byteorder::little_endian::F32::new(0.0);
        let recomputed = ControlClampDescriptor::from_wire(&wire0);
        assert!((recomputed.scale_norm - 0.2).abs() < 1.0e-6);

        let mut rules_desc = ControlClampDescriptor::range(0.0, 10.0, 5.0, 3, [0.0, -0.5, 0.5]);
        rules_desc.cond_rules = [
            ConditionalClampRule {
                switch_ctrl_idx: 0,
                switch_opt_idx: 2,
                rule_op: 2, // min_floor -> >= 7.0
                bound_val: 7.0,
            },
            ConditionalClampRule {
                switch_ctrl_idx: 0,
                switch_opt_idx: 2,
                rule_op: 5, // and_max_ceil -> <= 4.0 (collapses eff_max to 7.0)
                bound_val: 4.0,
            },
            ConditionalClampRule {
                switch_ctrl_idx: 0,
                switch_opt_idx: 2,
                rule_op: 6, // override_min -> >= 8.0
                bound_val: 8.0,
            },
        ];
        assert_eq!(rules_desc.effective_bounds(&[2]), (8.0, 8.0));
        assert_eq!(
            rules_desc.clamp_and_normalize(f32::NAN, &[2]),
            (8.0, 0.6, true)
        );
        let mut bad_def = rules_desc;
        bad_def.default_val = f32::NAN;
        assert_eq!(bad_def.clamp_and_normalize(f32::INFINITY, &[2]).0, 8.0);
        assert_eq!(bad_def.eval_lobatto_basis(0.5)[2], 0.0);

        let sw_bad = ControlClampDescriptor::switch(2, 0b10, 1);
        assert_eq!(sw_bad.clamp_and_normalize(99.0, &[]), (1.0, 1.0, true));
        assert_eq!(sw_bad.clamp_and_normalize(f32::NAN, &[]), (1.0, 1.0, true));
        let mut sw_no_def = ControlClampDescriptor::switch(2, 0b100, 0);
        sw_no_def.default_val = -1.0;
        assert_eq!(sw_no_def.clamp_and_normalize(-5.0, &[]), (2.0, 2.0, true));

        let unused = ControlClampDescriptor::empty();
        assert_eq!(unused.clamp_and_normalize(3.0, &[]), (0.0, 0.0, false));
    }
}
