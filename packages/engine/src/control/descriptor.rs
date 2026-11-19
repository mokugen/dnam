//! 64-byte binary control normalization & clamp descriptor (`ControlClampDescriptor`) and
//! switch-conditioned clamp rules (`ConditionalClampRule`).

use core::mem::{align_of, size_of};

use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Declarative little-endian wire layout for [`ConditionalClampRule`] (`8 B`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ConditionalClampRuleWire {
    /// Index of the conditioning `switch` control ($0 \dots P-1$).
    pub switch_ctrl_idx: U16,
    /// Active option index of the `switch` control ($0 \dots 15$).
    pub switch_opt_idx: u8,
    /// Rule operation: `0 = none`, `1 = max_ceil`, `2 = min_floor`, `3 = and_prefix`,
    /// `5 = and_max_keep`, `6 = and_min_end`.
    pub rule_op: u8,
    /// Conditional physical bound value.
    pub bound_val: F32,
}

const _: () = assert!(size_of::<ConditionalClampRuleWire>() == 8);
const _: () = assert!(align_of::<ConditionalClampRuleWire>() == 1);

impl ConditionalClampRuleWire {
    /// Converts from native-endian [`ConditionalClampRule`].
    #[inline]
    pub fn from_rule(rule: &ConditionalClampRule) -> Self {
        Self {
            switch_ctrl_idx: U16::new(rule.switch_ctrl_idx),
            switch_opt_idx: rule.switch_opt_idx,
            rule_op: rule.rule_op,
            bound_val: F32::new(rule.bound_val),
        }
    }

    /// Converts into native-endian [`ConditionalClampRule`].
    #[inline]
    pub fn to_rule(self) -> ConditionalClampRule {
        ConditionalClampRule {
            switch_ctrl_idx: self.switch_ctrl_idx.get(),
            switch_opt_idx: self.switch_opt_idx,
            rule_op: self.rule_op,
            bound_val: self.bound_val.get(),
        }
    }
}

/// Declarative little-endian wire layout for [`ControlClampDescriptor`] (`64 B`, `<2BH8fI3(H2Bf)`).
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ControlClampDescriptorWire {
    /// Control kind: `0 = range`, `1 = switch`, `0xFF = unused slot`.
    pub ctrl_kind: u8,
    /// Lobatto polynomial degree (`range`) or `num_declared_options - 1` (`switch`).
    pub lobatto_degree: u8,
    /// Bitmask of observed/active switch options.
    pub active_options_mask: U16,
    /// Default physical value (`range`) or default option index (`switch`).
    pub default_val: F32,
    /// Base safe minimum physical value.
    pub safe_min_base: F32,
    /// Base safe maximum physical value.
    pub safe_max_base: F32,
    /// Pre-inverted normalization scale.
    pub scale_norm: F32,
    /// Normalization offset.
    pub offset_norm: F32,
    /// Normalized Newton/Lobatto anchor nodes.
    pub anchor_nodes_norm: [F32; 3],
    /// Reserved 32-bit alignment gap (`0`).
    pub _reserved: U32,
    /// Up to 3 switch-coupled conditional clamp rules.
    pub cond_rules: [ConditionalClampRuleWire; 3],
}

const _: () = assert!(size_of::<ControlClampDescriptorWire>() == 64);
const _: () = assert!(align_of::<ControlClampDescriptorWire>() == 1);

impl ControlClampDescriptorWire {
    /// Encodes a native [`ControlClampDescriptor`] into little-endian wire layout.
    pub fn from_descriptor(desc: &ControlClampDescriptor) -> Self {
        Self {
            ctrl_kind: desc.ctrl_kind,
            lobatto_degree: desc.lobatto_degree,
            active_options_mask: U16::new(desc.active_options_mask),
            default_val: F32::new(desc.default_val),
            safe_min_base: F32::new(desc.safe_min_base),
            safe_max_base: F32::new(desc.safe_max_base),
            scale_norm: F32::new(desc.scale_norm),
            offset_norm: F32::new(desc.offset_norm),
            anchor_nodes_norm: [
                F32::new(desc.anchor_nodes_norm[0]),
                F32::new(desc.anchor_nodes_norm[1]),
                F32::new(desc.anchor_nodes_norm[2]),
            ],
            _reserved: U32::new(desc._reserved),
            cond_rules: [
                ConditionalClampRuleWire::from_rule(&desc.cond_rules[0]),
                ConditionalClampRuleWire::from_rule(&desc.cond_rules[1]),
                ConditionalClampRuleWire::from_rule(&desc.cond_rules[2]),
            ],
        }
    }

    /// Decodes into a cache-aligned native [`ControlClampDescriptor`].
    pub fn to_descriptor(&self) -> ControlClampDescriptor {
        ControlClampDescriptor {
            ctrl_kind: self.ctrl_kind,
            lobatto_degree: self.lobatto_degree,
            active_options_mask: self.active_options_mask.get(),
            default_val: self.default_val.get(),
            safe_min_base: self.safe_min_base.get(),
            safe_max_base: self.safe_max_base.get(),
            scale_norm: self.scale_norm.get(),
            offset_norm: self.offset_norm.get(),
            anchor_nodes_norm: [
                self.anchor_nodes_norm[0].get(),
                self.anchor_nodes_norm[1].get(),
                self.anchor_nodes_norm[2].get(),
            ],
            _reserved: self._reserved.get(),
            cond_rules: [
                self.cond_rules[0].to_rule(),
                self.cond_rules[1].to_rule(),
                self.cond_rules[2].to_rule(),
            ],
        }
    }
}

/// Binary switch-conditioned clamp rule (`8 B`, `#[repr(C)]`) inside [`ControlClampDescriptor`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalClampRule {
    /// Index of the conditioning `switch` control ($0 \dots P-1$).
    pub switch_ctrl_idx: u16,
    /// Active option index of the `switch` control ($0 \dots 15$) that triggers this rule.
    pub switch_opt_idx: u8,
    /// Rule operation: `0 = none`, `1 = max_ceil` ($x \le \text{bound}$), `2 = min_floor` ($x \ge \text{bound}$),
    /// `3 = and_prefix`, `5 = and_max_keep`, `6 = and_min_end`.
    pub rule_op: u8,
    /// Conditional physical bound value (`[CLAMP-STABILITY]` or `[CLAMP-DYNAMIC]`).
    pub bound_val: f32,
}

const _: () = assert!(size_of::<ConditionalClampRule>() == 8);

impl ConditionalClampRule {
    /// Returns an inactive (`rule_op = 0`) conditional clamp rule.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            switch_ctrl_idx: 0,
            switch_opt_idx: 0,
            rule_op: 0,
            bound_val: 0.0,
        }
    }
}

/// Fixed 64-byte binary control normalization and clamp descriptor (`#[repr(C, align(64))]`, `<2BH8fI3(H2Bf)`).
///
/// Stored as Cache Line 0 (`00..64 B`) of each 128-byte `ControlChannelRecord[p]`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlClampDescriptor {
    /// Control kind: `0 = range`, `1 = switch`, `0xFF = unused slot`.
    pub ctrl_kind: u8,
    /// For `range`: Lobatto polynomial degree $K_p \in \{0, 1, 2, 3\}$.
    /// For `switch`: `num_declared_options - 1`.
    pub lobatto_degree: u8,
    /// Bitmask of observed/active switch options (`[CLAMP-SWITCH]`; bit $i = 1$ if option $i$ is active).
    pub active_options_mask: u16,
    /// Default physical value (`range`) or default option index (`switch`).
    pub default_val: f32,
    /// Base safe minimum physical value (`[CLAMP-RANGE]`).
    pub safe_min_base: f32,
    /// Base safe maximum physical value (`[CLAMP-RANGE]`).
    pub safe_max_base: f32,
    /// Pre-inverted normalization scale $\frac{2}{u_{\text{decl\_max}} - u_{\text{decl\_min}}}$ (`0 FDIV` at $3\text{ kHz}$).
    pub scale_norm: f32,
    /// Normalization offset $-1 - u_{\text{decl\_min}} \cdot \text{scale\_norm}$.
    pub offset_norm: f32,
    /// Normalized Newton/Lobatto anchor nodes $[u_{p,1}, u_{p,2}, u_{p,3}] \in [-1, +1]^3$.
    pub anchor_nodes_norm: [f32; 3],
    /// Reserved 32-bit alignment gap (`0x24..0x28`, must be `0`).
    pub _reserved: u32,
    /// Up to 3 switch-coupled conditional clamp rules (`[CLAMP-STABILITY]`, `[CLAMP-DYNAMIC]`).
    pub cond_rules: [ConditionalClampRule; 3],
}

const _: () = assert!(size_of::<ControlClampDescriptor>() == 64);
const _: () = assert!(align_of::<ControlClampDescriptor>() == 64);

impl ControlClampDescriptor {
    /// Returns an unused 64-byte [`ControlClampDescriptor`] slot (`ctrl_kind = 0xFF`).
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
            cond_rules: [ConditionalClampRule::empty(); 3],
        }
    }

    /// Serializes this 64-byte descriptor into little-endian bytes (`<2BH8fI3(H2Bf)`).
    pub fn to_bytes(&self) -> [u8; 64] {
        let wire = ControlClampDescriptorWire::from_descriptor(self);
        let mut out = [0u8; 64];
        out.copy_from_slice(wire.as_bytes());
        out
    }

    /// Parses a 64-byte [`ControlClampDescriptor`] from little-endian bytes.
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        ControlClampDescriptorWire::ref_from_bytes(bytes.as_slice())
            .expect("64-byte slice matches ControlClampDescriptorWire")
            .to_descriptor()
    }

    /// Evaluates division-free clamping (`[CLAMP-RANGE]`, `[CLAMP-SWITCH]`, `[CLAMP-STABILITY]`,
    /// `[CLAMP-DYNAMIC]`) and normalization (`ref[dsp.ctrl.range.normalize-clamp]`) in `< 5 ns` with `0 FDIV`.
    ///
    /// Returns `(clamped_physical_val, normalized_u_in_minus1_to_1, was_clamped)`.
    #[inline(always)]
    pub fn clamp_and_normalize(
        &self,
        raw_physical_val: f32,
        active_switch_options: &[u8],
    ) -> (f32, f32, bool) {
        if self.ctrl_kind == 1 {
            if raw_physical_val.is_finite() {
                let req_opt = (raw_physical_val.max(0.0) + 0.5) as u8;
                let bit = 1_u16.checked_shl(req_opt as u32).unwrap_or(0);
                if (self.active_options_mask & bit) != 0 {
                    return (req_opt as f32, req_opt as f32, false);
                }
            }
            let def_opt = if self.default_val.is_finite() {
                (self.default_val.max(0.0) + 0.5) as u8
            } else {
                0
            };
            let def_bit = 1_u16.checked_shl(def_opt as u32).unwrap_or(0);
            let safe_opt = if (self.active_options_mask & def_bit) != 0 {
                def_opt
            } else {
                self.active_options_mask.trailing_zeros().min(15) as u8
            };
            return (safe_opt as f32, safe_opt as f32, true);
        }

        let mut eff_min = self.safe_min_base;
        let mut eff_max = self.safe_max_base;
        let mut conj_active = true;

        for rule in &self.cond_rules {
            if rule.rule_op == 0 {
                conj_active = true;
                continue;
            }
            let sw_matches = active_switch_options
                .get(rule.switch_ctrl_idx as usize)
                .is_some_and(|&active_opt| active_opt == rule.switch_opt_idx);
            match rule.rule_op {
                1 => {
                    if conj_active && sw_matches {
                        eff_max = eff_max.min(rule.bound_val);
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
                6 => {
                    if conj_active && sw_matches {
                        eff_min = eff_min.max(rule.bound_val);
                    }
                    conj_active = true;
                }
                _ => {
                    conj_active = true;
                }
            }
        }
        if eff_max < eff_min {
            eff_max = eff_min;
        }

        let (clamped, was_clamped) = if !raw_physical_val.is_finite() {
            let safe_def = if self.default_val.is_finite() {
                self.default_val
            } else {
                eff_min
            };
            (safe_def.clamp(eff_min, eff_max), true)
        } else {
            let c = raw_physical_val.clamp(eff_min, eff_max);
            let w = (c - raw_physical_val).abs() > 1e-6;
            (c, w)
        };
        let u_norm = (clamped * self.scale_norm + self.offset_norm).clamp(-1.0, 1.0);
        (clamped, u_norm, was_clamped)
    }

    /// Evaluates the Node-Anchored Newton/Lobatto polynomial basis (`ref[dsp.ctrl.basis.lobatto]`) at $u_p \in [-1, +1]$.
    #[inline(always)]
    pub fn eval_lobatto_basis(&self, u_p: f32) -> [f32; 3] {
        if self.ctrl_kind != 0 || self.lobatto_degree == 0 {
            return [0.0; 3];
        }
        let d1 = u_p - self.anchor_nodes_norm[0];
        let psi1 = d1;
        let psi2 = if self.lobatto_degree >= 2 {
            d1 * (u_p - self.anchor_nodes_norm[1])
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

    /// Evaluates two-pass (`switch` then `range`) conditional clamping and normalization across
    /// `num_controls` controls (`ref[dsp.ctrl.range.normalize-clamp]`), updating `clamped_values`,
    /// `normalized_values`, and `active_switch_options` in place and returning the 16-bit
    /// `active_clamp_mask`.
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
                let (clamped, u_norm, was_clamped) =
                    descriptors[p].clamp_and_normalize(raw_values[p], active_switch_options);
                clamped_values[p] = clamped;
                normalized_values[p] = u_norm;
                active_switch_options[p] = (clamped + 0.5) as u8;
                if was_clamped {
                    clamp_mask |= 1_u16 << p.min(15);
                }
            }
        }
        for p in 0..n {
            if descriptors[p].ctrl_kind == 0 {
                let (clamped, u_norm, was_clamped) =
                    descriptors[p].clamp_and_normalize(raw_values[p], active_switch_options);
                clamped_values[p] = clamped;
                normalized_values[p] = u_norm;
                if was_clamped {
                    clamp_mask |= 1_u16 << p.min(15);
                }
            }
        }
        clamp_mask
    }

    /// Evaluates two-pass (`switch` then `range`) conditional clamping across `num_controls`
    /// controls (`ref[dsp.ctrl.range.normalize-clamp]`), updating `clamped_values` and
    /// `active_switch_options` in place and returning the 16-bit `active_clamp_mask`.
    pub fn clamp_all_controls(
        descriptors: &[Self],
        num_controls: usize,
        raw_values: &[f32],
        clamped_values: &mut [f32],
        active_switch_options: &mut [u8],
    ) -> u16 {
        let mut norm_scratch = [0.0_f32; 256];
        let n = num_controls.min(256);
        Self::clamp_and_normalize_all(
            descriptors,
            n,
            raw_values,
            clamped_values,
            &mut norm_scratch[..n],
            active_switch_options,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_descriptor_serialization_and_conditional_rules() {
        let mut d_cond = ControlClampDescriptor {
            ctrl_kind: 0,
            lobatto_degree: 2,
            active_options_mask: 1,
            default_val: 5.0,
            safe_min_base: 0.0,
            safe_max_base: 10.0,
            scale_norm: 0.2,
            offset_norm: -1.0,
            anchor_nodes_norm: [0.0, -1.0, 1.0],
            _reserved: 0,
            cond_rules: [
                ConditionalClampRule {
                    switch_ctrl_idx: 1,
                    switch_opt_idx: 1,
                    rule_op: 1,
                    bound_val: 2.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 1,
                    switch_opt_idx: 1,
                    rule_op: 2,
                    bound_val: 5.0,
                },
                ConditionalClampRule::empty(),
            ],
        };
        let bytes = d_cond.to_bytes();
        let parsed = ControlClampDescriptor::from_bytes(&bytes);
        assert_eq!(parsed, d_cond);

        let mut sw_opts = [0u8; 16];
        sw_opts[1] = 1;
        let (clamped_v, _, was_cl) = d_cond.clamp_and_normalize(8.0, &sw_opts);
        assert_eq!(clamped_v, 5.0);
        assert!(was_cl);

        d_cond.cond_rules[0].bound_val = 7.0;
        let (clamped_v2, _, _) = d_cond.clamp_and_normalize(1.0, &sw_opts);
        assert_eq!(clamped_v2, 5.0);

        let d_sw = ControlClampDescriptor {
            ctrl_kind: 1,
            lobatto_degree: 2,
            active_options_mask: 0b0011,
            default_val: 0.0,
            safe_min_base: 0.0,
            safe_max_base: 2.0,
            scale_norm: 1.0,
            offset_norm: 0.0,
            anchor_nodes_norm: [0.0; 3],
            _reserved: 0,
            cond_rules: [ConditionalClampRule::empty(); 3],
        };
        let descs = [d_cond, d_sw];
        let raw = [8.0_f32, 2.0_f32];
        let mut clamped = [0.0_f32; 2];
        let mut norm = [0.0_f32; 2];
        let mut sw = [0u8; 2];
        let mask = ControlClampDescriptor::clamp_and_normalize_all(
            &descs,
            2,
            &raw,
            &mut clamped,
            &mut norm,
            &mut sw,
        );
        assert_eq!(mask, 0b0010);
        assert_eq!(sw[1], 0);
        assert_eq!(clamped[0], 8.0);
        let mask2 = ControlClampDescriptor::clamp_all_controls(
            &descs,
            2,
            &[8.0_f32, 1.0_f32],
            &mut clamped,
            &mut sw,
        );
        assert_eq!(mask2, 0b0001);
        assert_eq!(clamped[0], 7.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_control_clamping_and_default_lobatto_zero(
            raw_val in -100.0_f32..100.0_f32,
        ) {
            let desc = ControlClampDescriptor {
                ctrl_kind: 0,
                lobatto_degree: 3,
                active_options_mask: 1,
                default_val: 5.0,
                safe_min_base: 2.0,
                safe_max_base: 8.0,
                scale_norm: 0.2,
                offset_norm: -1.0,
                anchor_nodes_norm: [0.0, -1.0, 1.0],
                _reserved: 0,
                cond_rules: [ConditionalClampRule::empty(); 3],
            };
            let sw = [0u8; 16];
            let (clamped, u_norm, _) = desc.clamp_and_normalize(raw_val, &sw);
            prop_assert!((2.0..=8.0).contains(&clamped));
            prop_assert!((-1.0..=1.0).contains(&u_norm));

            let (_, u_def, _) = desc.clamp_and_normalize(5.0, &sw);
            let basis_def = desc.eval_lobatto_basis(u_def);
            prop_assert_eq!(basis_def, [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_nonfinite_guard_and_control_index_ge16_overflow_bit() {
        let mut descs = [ControlClampDescriptor::empty(); 18];
        for desc in &mut descs {
            *desc = ControlClampDescriptor {
                ctrl_kind: 0,
                default_val: 5.0,
                safe_min_base: 2.0,
                safe_max_base: 8.0,
                scale_norm: 0.2,
                offset_norm: -1.0,
                ..ControlClampDescriptor::empty()
            };
        }
        descs[17].ctrl_kind = 1;
        descs[17].active_options_mask = 0b0010;
        descs[17].default_val = 1.0;

        let sw = [0u8; 18];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (clamped, u_norm, was_clamped) = descs[0].clamp_and_normalize(bad, &sw);
            assert!(was_clamped);
            assert_eq!(clamped, 5.0);
            assert!(u_norm.is_finite());

            let (sw_val, sw_norm, sw_clamped) = descs[17].clamp_and_normalize(bad, &sw);
            assert!(sw_clamped);
            assert_eq!(sw_val, 1.0);
            assert_eq!(sw_norm, 1.0);
        }

        let mut raw = [5.0_f32; 18];
        raw[16] = 100.0; // outside [2.0, 8.0], must set overflow bit 1 << 15 (0x8000)
        let mut clamped_all = [0.0_f32; 18];
        let mut norm_all = [0.0_f32; 18];
        let mut sw_all = [0u8; 18];
        let mask = ControlClampDescriptor::clamp_and_normalize_all(
            &descs,
            18,
            &raw,
            &mut clamped_all,
            &mut norm_all,
            &mut sw_all,
        );
        assert_eq!(mask, 1_u16 << 15);
        assert_eq!(clamped_all[16], 8.0);
        assert_eq!(clamped_all[17], 1.0);
    }

    #[test]
    fn test_multi_switch_conjunction_rule_ops_3_5_6_and_3way_max() {
        // 2-switch conjunction (sw1 == 1 && sw2 == 2) with both max (7.0) and min (3.0)
        let desc_2way = ControlClampDescriptor {
            ctrl_kind: 0,
            lobatto_degree: 1,
            active_options_mask: 1,
            default_val: 5.0,
            safe_min_base: 0.0,
            safe_max_base: 10.0,
            scale_norm: 0.2,
            offset_norm: -1.0,
            anchor_nodes_norm: [0.0, -1.0, 1.0],
            _reserved: 0,
            cond_rules: [
                ConditionalClampRule {
                    switch_ctrl_idx: 1,
                    switch_opt_idx: 1,
                    rule_op: 3,
                    bound_val: 0.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 2,
                    switch_opt_idx: 2,
                    rule_op: 5,
                    bound_val: 7.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 2,
                    switch_opt_idx: 2,
                    rule_op: 6,
                    bound_val: 3.0,
                },
            ],
        };
        assert_eq!(desc_2way.clamp_and_normalize(9.0, &[0, 0, 2]).0, 9.0);
        assert_eq!(desc_2way.clamp_and_normalize(1.0, &[0, 0, 2]).0, 1.0);
        assert_eq!(desc_2way.clamp_and_normalize(9.0, &[0, 1, 0]).0, 9.0);
        assert_eq!(desc_2way.clamp_and_normalize(1.0, &[0, 1, 0]).0, 1.0);
        assert_eq!(desc_2way.clamp_and_normalize(9.0, &[0, 1, 2]).0, 7.0);
        assert_eq!(desc_2way.clamp_and_normalize(1.0, &[0, 1, 2]).0, 3.0);

        // 3-switch conjunction (sw1 == 1 && sw2 == 1 && sw3 == 2) with max (4.0)
        let desc_3way = ControlClampDescriptor {
            ctrl_kind: 0,
            lobatto_degree: 1,
            active_options_mask: 1,
            default_val: 5.0,
            safe_min_base: 0.0,
            safe_max_base: 10.0,
            scale_norm: 0.2,
            offset_norm: -1.0,
            anchor_nodes_norm: [0.0, -1.0, 1.0],
            _reserved: 0,
            cond_rules: [
                ConditionalClampRule {
                    switch_ctrl_idx: 1,
                    switch_opt_idx: 1,
                    rule_op: 3,
                    bound_val: 0.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 2,
                    switch_opt_idx: 1,
                    rule_op: 3,
                    bound_val: 0.0,
                },
                ConditionalClampRule {
                    switch_ctrl_idx: 3,
                    switch_opt_idx: 2,
                    rule_op: 1,
                    bound_val: 4.0,
                },
            ],
        };
        assert_eq!(desc_3way.clamp_and_normalize(8.0, &[0, 1, 0, 2]).0, 8.0);
        assert_eq!(desc_3way.clamp_and_normalize(8.0, &[0, 1, 1, 0]).0, 8.0);
        assert_eq!(desc_3way.clamp_and_normalize(8.0, &[0, 1, 1, 2]).0, 4.0);
    }
}
