#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Extreme valid corner generation honoring [`ControlClampDescriptor`] bounds and active
//! `cond_rules` (`dnam_capture::audit::dynamic_and_panama::corners`).

use std::collections::BTreeSet;

use dnam_vm::bin::ControlClampDescriptorWire;
use dnam_vm::control::ControlClampDescriptor;

const MAX_EXTREME_CORNERS: usize = 64;

pub(super) fn switch_active_options(desc: &ControlClampDescriptor) -> (Vec<u8>, u8) {
    let n_opts = (usize::from(desc.lobatto_degree) + 1).clamp(1, 16);
    let mut active = Vec::with_capacity(n_opts);
    for opt in 0..n_opts {
        if (desc.active_options_mask & (1_u16 << opt)) != 0 {
            active.push(opt as u8);
        }
    }
    if active.is_empty() {
        active.push(0);
    }
    let raw_def = (desc.default_val.round().max(0.0) as u8).min((n_opts.saturating_sub(1)) as u8);
    let default_opt = if active.contains(&raw_def) {
        raw_def
    } else {
        active[0]
    };
    (active, default_opt)
}

fn build_switch_assignments(
    descs: &[ControlClampDescriptor],
    switch_indices: &[usize],
    max_configs: usize,
) -> Vec<Vec<u8>> {
    let n_ctrl = descs.len();
    let mut default_vec = vec![0_u8; n_ctrl];
    let mut per_switch_active: Vec<(usize, Vec<u8>)> = Vec::with_capacity(switch_indices.len());
    for &idx in switch_indices {
        let (opts, def_opt) = switch_active_options(&descs[idx]);
        default_vec[idx] = def_opt;
        per_switch_active.push((idx, opts));
    }

    let mut configs = vec![default_vec.clone()];
    for &(idx, ref opts) in &per_switch_active {
        for &opt in opts {
            if opt != default_vec[idx] {
                let mut v = default_vec.clone();
                v[idx] = opt;
                configs.push(v);
            }
        }
    }
    if switch_indices.len() >= 2 {
        let mut all_min = default_vec.clone();
        let mut all_max = default_vec.clone();
        for &(idx, ref opts) in &per_switch_active {
            all_min[idx] = opts[0];
            all_max[idx] = *opts.last().unwrap_or(&opts[0]);
        }
        configs.push(all_min);
        configs.push(all_max);
    }

    let mut seen: BTreeSet<Vec<u8>> = BTreeSet::new();
    configs.retain(|c| seen.insert(c.clone()));
    configs.truncate(max_configs.max(1));
    configs
}

fn range_corner_value(desc: &ControlClampDescriptor, sw_opts: &[u8]) -> (f32, f32, f32) {
    let (mut eff_min, mut eff_max) = desc.effective_bounds(sw_opts);
    let def_fallback = if desc.default_val.is_finite() {
        desc.default_val
    } else {
        0.0
    };
    if !eff_min.is_finite() {
        eff_min = def_fallback;
    }
    if !eff_max.is_finite() || eff_max < eff_min {
        eff_max = eff_min;
    }
    let eff_def = def_fallback.clamp(eff_min, eff_max);
    (eff_min, eff_max, eff_def)
}

fn push_unique_corner(
    corners: &mut Vec<Vec<(usize, f32)>>,
    seen: &mut BTreeSet<Vec<u32>>,
    corner: Vec<(usize, f32)>,
) {
    if corners.len() >= MAX_EXTREME_CORNERS {
        return;
    }
    let key: Vec<u32> = corner.iter().map(|&(_, v)| v.to_bits()).collect();
    if seen.insert(key) {
        corners.push(corner);
    }
}

/// Builds `<= 64` valid extreme control corners obeying [`ControlClampDescriptor`] bounds and
/// active `cond_rules` (`ref[train.audit.phase3.clamp-dynamic]`, `ref[train.audit.phase3.numeric-envelope]`).
#[must_use]
pub fn build_extreme_valid_corners(
    descriptors: &[ControlClampDescriptorWire],
    num_controls: usize,
) -> Vec<Vec<(usize, f32)>> {
    let n = num_controls.min(descriptors.len());
    if n == 0 {
        return vec![Vec::new()];
    }

    let descs: Vec<ControlClampDescriptor> = descriptors[..n]
        .iter()
        .map(ControlClampDescriptor::from_wire)
        .collect();
    let switch_indices: Vec<usize> = (0..n).filter(|&i| descs[i].ctrl_kind == 1).collect();
    let range_indices: Vec<usize> = (0..n).filter(|&i| descs[i].ctrl_kind == 0).collect();

    let sw_cap = if range_indices.is_empty() { 64 } else { 12 };
    let sw_configs = build_switch_assignments(&descs, &switch_indices, sw_cap);

    let mut corners = Vec::new();
    let mut seen = BTreeSet::new();

    for (sw_cfg_idx, sw_opts) in sw_configs.iter().enumerate() {
        let bounds: Vec<(f32, f32, f32)> = (0..n)
            .map(|i| match descs[i].ctrl_kind {
                0 => range_corner_value(&descs[i], sw_opts),
                1 => {
                    let v = f32::from(sw_opts[i]);
                    (v, v, v)
                }
                _ => (0.0, 0.0, 0.0),
            })
            .collect();

        let mk_default = || -> Vec<(usize, f32)> { (0..n).map(|i| (i, bounds[i].2)).collect() };
        let mk_pattern = |mode: u8| -> Vec<(usize, f32)> {
            let mut r_pos = 0_usize;
            (0..n)
                .map(|i| {
                    if descs[i].ctrl_kind != 0 {
                        return (i, bounds[i].2);
                    }
                    let pick_hi = match mode {
                        0 => false,
                        1 => true,
                        2 => r_pos.is_multiple_of(2),
                        _ => !r_pos.is_multiple_of(2),
                    };
                    r_pos += 1;
                    (i, if pick_hi { bounds[i].1 } else { bounds[i].0 })
                })
                .collect()
        };

        push_unique_corner(&mut corners, &mut seen, mk_default());
        for mode in 0..4_u8 {
            push_unique_corner(&mut corners, &mut seen, mk_pattern(mode));
        }

        if sw_cfg_idx == 0 && !range_indices.is_empty() && range_indices.len() <= 4 {
            let combos = 1_usize << range_indices.len();
            for mask in 0..combos {
                let mut corner = mk_default();
                for (bit, &r_idx) in range_indices.iter().enumerate() {
                    let hi = (mask & (1_usize << bit)) != 0;
                    corner[r_idx].1 = if hi { bounds[r_idx].1 } else { bounds[r_idx].0 };
                }
                push_unique_corner(&mut corners, &mut seen, corner);
            }
        }
        if corners.len() >= MAX_EXTREME_CORNERS {
            break;
        }
    }

    if corners.is_empty() {
        corners.push((0..n).map(|i| (i, 0.0)).collect());
    }
    corners
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_vm::bin::ConditionalClampRuleWire;
    use zerocopy::byteorder::little_endian::{F32, U16};

    fn mk_range_wire(min: f32, max: f32, def: f32) -> ControlClampDescriptorWire {
        let mut d = ControlClampDescriptor::empty();
        d.ctrl_kind = 0;
        d.lobatto_degree = 3;
        d.default_val = def;
        d.safe_min_base = min;
        d.safe_max_base = max;
        d.to_wire()
    }

    fn mk_switch_wire(n_opts: u8, active_mask: u16, def: f32) -> ControlClampDescriptorWire {
        let mut d = ControlClampDescriptor::empty();
        d.ctrl_kind = 1;
        d.lobatto_degree = n_opts.saturating_sub(1);
        d.active_options_mask = active_mask;
        d.default_val = def;
        d.safe_max_base = f32::from(n_opts.saturating_sub(1));
        d.to_wire()
    }

    #[test]
    fn test_build_extreme_valid_corners_obeys_conditional_rules_and_caps_at_64() {
        let mut w_range = mk_range_wire(0.0, 10.0, 2.0);
        w_range.cond_rules[0] = ConditionalClampRuleWire {
            switch_ctrl_idx: U16::new(0),
            switch_opt_idx: 1,
            rule_op: 1,
            bound_val: F32::new(6.5),
        };
        let mut descs = vec![mk_switch_wire(2, 0b11, 0.0), w_range];
        for i in 0..7 {
            descs.push(mk_range_wire(
                -1.0,
                1.0,
                if i % 2 == 0 { 0.0 } else { 0.25 },
            ));
        }
        let corners = build_extreme_valid_corners(&descs, descs.len());
        assert!(!corners.is_empty());
        assert!(corners.len() <= MAX_EXTREME_CORNERS);

        let mut saw_sw1 = false;
        for c in &corners {
            assert_eq!(c.len(), descs.len());
            let sw = c[0].1;
            assert!(sw == 0.0 || sw == 1.0);
            if (sw - 1.0).abs() < 1e-6 {
                saw_sw1 = true;
                assert!(c[1].1 <= 6.5 + 1e-6);
            } else {
                assert!(c[1].1 <= 10.0 + 1e-6);
            }
        }
        assert!(saw_sw1);
    }
}
