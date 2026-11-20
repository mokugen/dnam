//! Pure-binary `"TSYN"` tempo synchronization, quadratic Lobatto inversion, `Q32` musical
//! subdivision phase math, and division-free `Q64` host timebase increment (`ref[dsp.tempo.*]`).

use crate::bin::TempoSyncEntryWire;
use crate::control::ControlClampDescriptor;
use crate::math::{floor_f64, rcp_nr6_f64, sqrt_f64};

/// Canonical period ratios `(a_s, b_s)` (`r_sub = a_s / b_s`) across all 12 subdivisions
/// (`1/2., 1/2, 1/2T, 1/4., 1/4, 1/4T, 1/8., 1/8, 1/8T, 1/16., 1/16, 1/16T`).
pub const CANONICAL_SUBDIVISION_RATIOS: [(u32, u32); 12] = [
    (3, 1),
    (2, 1),
    (4, 3),
    (3, 2),
    (1, 1),
    (2, 3),
    (3, 4),
    (1, 2),
    (1, 3),
    (3, 8),
    (1, 4),
    (1, 6),
];

/// Remainder mod 3 lookup table for `0..24` (`beat_mod12` and `product >> 32` up to `23`).
const REM3_TABLE_24: [u8; 24] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
];

/// CRT table mapping `(r3 << 2) | r4` -> `(9 * r4 + 4 * r3) mod 12`.
const CRT_MOD12: [u8; 12] = [0, 9, 6, 3, 4, 1, 10, 7, 8, 5, 2, 11];

/// `"TSYN"` physical target unit interpretation.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TempoSyncMode {
    /// Period in milliseconds (`60_000 / (bpm * bpm_factor) * div_ratio`).
    PeriodMs = 0,
    /// Frequency in Hertz (`(bpm * bpm_factor) / (60 * div_ratio)`).
    FrequencyHz = 1,
}

impl TempoSyncMode {
    /// Parses a wire mode byte (`0 = PeriodMs`, `1 = FrequencyHz`).
    #[inline]
    pub const fn from_u8(mode: u8) -> Option<Self> {
        match mode {
            0 => Some(Self::PeriodMs),
            1 => Some(Self::FrequencyHz),
            _ => None,
        }
    }
}

/// Output of evaluating one active `"TSYN"` entry into normalized & clamped control coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoSyncEvalResult {
    /// Bounded target period (ms) or frequency (Hz) after optional octave folding.
    pub target_physical: f32,
    /// Inverted normalized coordinate `u_norm` in `[-1.0, +1.0]`.
    pub u_norm: f32,
    /// Unclamped physical value inferred on `[declared_min, declared_max]`.
    pub x_physical: f32,
    /// Clamped physical control value after applying conditional descriptor bounds.
    pub clamped_physical: f32,
    /// Normalized coordinate after descriptor clamping.
    pub clamped_u_norm: f32,
    /// `true` if descriptor clamping modified the inferred value.
    pub was_clamped: bool,
}

/// Evaluates the bounded physical target (ms or Hz) for a validated `"TSYN"` row at `bpm`
/// (`clamp(bpm, 20.0, 400.0)`), using zero `FDIV` and zero `libm` (`ref[dsp.tempo.subdivisions]`).
#[inline]
pub fn eval_tsyn_target_value(entry: &TempoSyncEntryWire, bpm: f32) -> f32 {
    let bpm_safe = if bpm.is_finite() {
        bpm.clamp(20.0, 400.0)
    } else {
        120.0
    };
    let d_min = entry.declared_min.get();
    let d_max = entry.declared_max.get();
    let div_ratio = entry.div_ratio.get();
    let bpm_factor = entry.bpm_factor.get();
    if !d_min.is_finite()
        || !d_max.is_finite()
        || d_min > d_max
        || !div_ratio.is_finite()
        || div_ratio <= 0.0
        || !bpm_factor.is_finite()
        || bpm_factor <= 0.0
    {
        return 0.0;
    }

    let t_raw = match entry.mode {
        0 => {
            ((60_000.0 * rcp_nr6_f64(f64::from(bpm_safe) * f64::from(bpm_factor)))
                * f64::from(div_ratio)) as f32
        }
        _ => {
            ((f64::from(bpm_safe) * f64::from(bpm_factor))
                * rcp_nr6_f64(60.0 * f64::from(div_ratio))) as f32
        }
    };

    if !t_raw.is_finite() {
        return d_min;
    }
    let clamp_only = (entry.flags.get() & 1) == 1;
    if clamp_only || !(t_raw > 1.0e-6 && d_min > 1.0e-6 && d_max > d_min) {
        return t_raw.clamp(d_min, d_max);
    }

    let min64 = f64::from(d_min);
    let max64 = f64::from(d_max);
    let mut t_fold = f64::from(t_raw);
    for _ in 0..16 {
        if t_fold > max64 {
            t_fold *= 0.5;
        } else if t_fold < min64 {
            t_fold *= 2.0;
        } else {
            break;
        }
    }
    if !(min64 <= t_fold && t_fold <= max64) {
        t_fold = f64::from(t_raw);
    }
    (t_fold as f32).clamp(d_min, d_max)
}

/// Evaluates the quadratic 3-node Lobatto calibration curve at `u in [-1.0, +1.0]` in `f64`.
#[inline]
pub fn eval_3node_calibration_f64(anchors: [f32; 3], u: f64) -> f64 {
    let v_min = f64::from(anchors[0]);
    let v_mid = f64::from(anchors[1]);
    let v_max = f64::from(anchors[2]);
    let c1 = 0.5 * (v_max - v_min);
    let c2 = 0.5 * (v_max + v_min) - v_mid;
    v_mid + u * (c1 + u * c2)
}

/// Inverts 3-node Lobatto anchors `[v_min, v_mid, v_max]` for `target` without `FDIV` or `libm`
/// (`ref[dsp.tempo.physical-inversion]`).
#[inline]
pub fn invert_tsyn_lobatto_anchors(anchors: [f32; 3], target: f32) -> f32 {
    let v_min = f64::from(anchors[0]);
    let v_mid = f64::from(anchors[1]);
    let v_max = f64::from(anchors[2]);
    let t = f64::from(target);
    let lo = if v_min <= v_max { v_min } else { v_max };
    let hi = if v_min >= v_max { v_min } else { v_max };
    let ascending = v_max >= v_min;
    if t <= lo {
        return if ascending { -1.0 } else { 1.0 };
    }
    if t >= hi {
        return if ascending { 1.0 } else { -1.0 };
    }
    let c1 = 0.5 * (v_max - v_min);
    let c2 = 0.5 * (v_max + v_min) - v_mid;
    let abs_c1 = if c1 >= 0.0 { c1 } else { -c1 };
    let abs_c2 = if c2 >= 0.0 { c2 } else { -c2 };
    let scale_ref = if abs_c1 >= 1.0 { abs_c1 } else { 1.0 };
    if abs_c2 <= 1.0e-7 * scale_ref {
        if abs_c1 <= 1.0e-12 {
            return 0.0;
        }
        let inv_c1 = (if c1 >= 0.0 { 1.0 } else { -1.0 }) * rcp_nr6_f64(abs_c1);
        return (((t - v_mid) * inv_c1) as f32).clamp(-1.0, 1.0);
    }
    let disc_raw = c1 * c1 - 4.0 * c2 * (v_mid - t);
    let disc = if disc_raw > 0.0 { disc_raw } else { 0.0 };
    let root = sqrt_f64(disc);
    let inv_2c2 = (if c2 >= 0.0 { 1.0 } else { -1.0 }) * rcp_nr6_f64(2.0 * abs_c2);
    let r1 = (-c1 + root) * inv_2c2;
    let r2 = (-c1 - root) * inv_2c2;
    let in1 = (-1.0001..=1.0001).contains(&r1);
    let in2 = (-1.0001..=1.0001).contains(&r2);
    let chosen = if in1 && !in2 {
        r1
    } else if in2 && !in1 {
        r2
    } else if c1 >= 0.0 {
        r1
    } else {
        r2
    };
    (chosen as f32).clamp(-1.0, 1.0)
}

/// Evaluates a single matching `"TSYN"` row into physical control value + clamped normalized state.
#[inline]
pub fn eval_tsyn_row_physical(
    entry: &TempoSyncEntryWire,
    descriptor: &ControlClampDescriptor,
    bpm: f32,
    active_switch_options: &[u8],
) -> TempoSyncEvalResult {
    let target_physical = eval_tsyn_target_value(entry, bpm);
    let anchors = [
        entry.lobatto_anchors[0].get(),
        entry.lobatto_anchors[1].get(),
        entry.lobatto_anchors[2].get(),
    ];
    let u_norm = invert_tsyn_lobatto_anchors(anchors, target_physical);
    let d_min = entry.declared_min.get();
    let d_max = entry.declared_max.get();
    let x_physical = d_min + 0.5 * (u_norm + 1.0) * (d_max - d_min);
    let (clamped_physical, clamped_u_norm, was_clamped) =
        descriptor.clamp_and_normalize(x_physical, active_switch_options);
    TempoSyncEvalResult {
        target_physical,
        u_norm,
        x_physical,
        clamped_physical,
        clamped_u_norm,
        was_clamped,
    }
}

/// Applies all matching rows of a `"TSYN"` table at `bpm`, returning the number of controls updated.
pub fn apply_tsyn_table(
    entries: &[TempoSyncEntryWire],
    descriptors: &[ControlClampDescriptor],
    active_switch_options: &[u8],
    bpm: f32,
    raw_values_out: &mut [f32],
    clamped_values_out: &mut [f32],
    normalized_values_out: &mut [f32],
) -> usize {
    let mut updated = 0usize;
    for entry in entries {
        if entry.has_switch_cond == 1 {
            let sw = usize::from(entry.switch_ctrl_idx.get());
            if active_switch_options.get(sw).copied() != Some(entry.switch_opt_idx) {
                continue;
            }
        }
        let ctrl_idx = usize::from(entry.ctrl_idx.get());
        if ctrl_idx >= descriptors.len()
            || ctrl_idx >= raw_values_out.len()
            || ctrl_idx >= clamped_values_out.len()
            || ctrl_idx >= normalized_values_out.len()
        {
            continue;
        }
        let eval =
            eval_tsyn_row_physical(entry, &descriptors[ctrl_idx], bpm, active_switch_options);
        raw_values_out[ctrl_idx] = eval.x_physical;
        clamped_values_out[ctrl_idx] = eval.clamped_physical;
        normalized_values_out[ctrl_idx] = eval.clamped_u_norm;
        updated += 1;
    }
    updated
}

/// Computes exact division-free `Q32` musical subdivision phase (`ref[dsp.tempo.q64-q32-phase]`).
#[inline]
pub fn subdivision_phase_q32(
    beat_mod12: u8,
    within_beat_q32: u32,
    subdiv_idx: u8,
    phase_offset_q32: u32,
) -> u32 {
    let (a_s, b_s) = CANONICAL_SUBDIVISION_RATIOS[usize::from(subdiv_idx).min(11)];
    let beat_rem = match a_s {
        1 => 0_u64,
        2 => u64::from(beat_mod12 & 1),
        3 => {
            if usize::from(beat_mod12) < REM3_TABLE_24.len() {
                u64::from(REM3_TABLE_24[usize::from(beat_mod12)])
            } else {
                let b = u32::from(beat_mod12);
                let q = (b * 0xAAAB) >> 17;
                u64::from(b - q * 3)
            }
        }
        4 => u64::from(beat_mod12 & 3),
        _ => 0,
    };
    let num = u64::from(b_s) * ((beat_rem << 32) | u64::from(within_beat_q32));
    let l = num & 0xFFFF_FFFF;
    let h = (num >> 32) as usize;
    let q = match a_s {
        1 => num,
        2 => (((h as u64 & 1) << 32) + l) >> 1,
        4 => (((h as u64 & 3) << 32) + l) >> 2,
        3 => {
            let r = u64::from(REM3_TABLE_24[h]);
            r * 0x5555_5555_u64 + (((l + r) * 0xAAAA_AAAB_u64) >> 33)
        }
        _ => 0,
    };
    (q as u32).wrapping_add(phase_offset_q32)
}

/// Computes `round_even(2^64 / sample_rate_hz)` via 64-step binary shift-subtract (zero `u128` division
/// and zero `__udivti3` compiler intrinsic emissions).
#[inline]
pub fn host_frame_q64_increment(sample_rate_hz: u32) -> u64 {
    let d = u64::from(sample_rate_hz.max(1));
    if d == 1 {
        return u64::MAX;
    }
    let mut rem = 1_u64;
    let mut quot = 0_u64;
    for _ in 0..64 {
        rem <<= 1;
        quot <<= 1;
        if rem >= d {
            rem -= d;
            quot |= 1;
        }
    }
    let twice_rem = rem << 1;
    if twice_rem > d || (twice_rem == d && (quot & 1) == 1) {
        quot = quot.wrapping_add(1);
    }
    quot
}

/// Helper converting floating beat offset into `(beat_mod12, within_beat_q32)`.
#[inline]
pub fn beat_pos_to_mod12_q32(beats: f64) -> (u8, u32) {
    if !beats.is_finite() {
        return (0, 0);
    }
    let floored = floor_f64(beats);
    let frac = (beats - floored).clamp(0.0, 0.999_999_999);
    let signed = floored as i64;
    let mut u = signed.unsigned_abs();
    u = (u & 0xFFFF_FFFF) + (u >> 32);
    u = (u & 0xFFFF_FFFF) + (u >> 32);
    u = (u & 0xFFFF) + (u >> 16);
    u = (u & 0xFFFF) + (u >> 16);
    u = (u & 0xFF) + (u >> 8);
    u = (u & 0xFF) + (u >> 8);
    u = (u & 0xF) + (u >> 4);
    u = (u & 0xF) + (u >> 4);
    u = (u & 3) + (u >> 2);
    u = (u & 3) + (u >> 2);
    let r3_abs = if u == 3 { 0 } else { u as usize };
    let r3 = if signed < 0 && r3_abs != 0 {
        3 - r3_abs
    } else {
        r3_abs
    };
    let r4 = ((signed as u64) & 3) as usize;
    let whole = CRT_MOD12[(r3 << 2) | r4];
    let q32 = (frac * 4_294_967_296.0) as u64 as u32;
    (whole, q32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::byteorder::little_endian::{F32, U16, U32};

    #[test]
    fn test_tsyn_evaluation_inversion_and_q64_increment() {
        let entry = TempoSyncEntryWire {
            ctrl_idx: U16::new(0),
            switch_ctrl_idx: U16::new(0),
            switch_opt_idx: 0,
            has_switch_cond: 0,
            mode: 0,           // PeriodMs
            default_subdiv: 4, // 1/4 note
            flags: U32::new(0),
            div_ratio: F32::new(1.0),
            bpm_factor: F32::new(1.0),
            lobatto_anchors: [F32::new(100.0), F32::new(400.0), F32::new(1_000.0)],
            declared_min: F32::new(100.0),
            declared_max: F32::new(1_000.0),
        };
        let target_120 = eval_tsyn_target_value(&entry, 120.0);
        assert!((target_120 - 500.0).abs() < 1.0e-3);
        let u = invert_tsyn_lobatto_anchors([100.0, 400.0, 1_000.0], target_120);
        let recovered = eval_3node_calibration_f64([100.0, 400.0, 1_000.0], f64::from(u));
        assert!((recovered - f64::from(target_120)).abs() < 1.0e-3);

        let inc48 = host_frame_q64_increment(48_000);
        assert_eq!(inc48, (((1_u128 << 64) + 24_000) / 48_000) as u64);

        let (_, wb) = beat_pos_to_mod12_q32(12.25);
        assert_eq!(subdivision_phase_q32(0, wb, 4, 0), 0x4000_0000);
        assert_eq!(subdivision_phase_q32(1, 0, 2, 0), 0xC000_0000);

        assert_eq!(TempoSyncMode::from_u8(0), Some(TempoSyncMode::PeriodMs));
        assert_eq!(TempoSyncMode::from_u8(1), Some(TempoSyncMode::FrequencyHz));
        assert_eq!(TempoSyncMode::from_u8(9), None);

        // FrequencyHz + octave folding (high & low) + clamp_only + NaN bpm fallback.
        let mut hz_entry = entry;
        hz_entry.mode = 1;
        hz_entry.lobatto_anchors = [F32::new(0.5), F32::new(4.0), F32::new(8.0)];
        hz_entry.declared_min = F32::new(0.5);
        hz_entry.declared_max = F32::new(8.0);
        let f_hi = eval_tsyn_target_value(&hz_entry, 400.0);
        assert!((0.5..=8.0).contains(&f_hi));
        let f_nan = eval_tsyn_target_value(&hz_entry, f32::NAN);
        assert!((f_nan - 2.0).abs() < 1.0e-4);

        let mut clamp_only = hz_entry;
        clamp_only.flags = U32::new(1);
        clamp_only.declared_min = F32::new(10.0);
        clamp_only.declared_max = F32::new(20.0);
        assert_eq!(eval_tsyn_target_value(&clamp_only, 60.0), 10.0);

        // Linear & degenerate anchor inversions + negative beat_pos_to_mod12_q32 + subdivisions.
        assert!((invert_tsyn_lobatto_anchors([0.0, 5.0, 10.0], 2.5) + 0.5).abs() < 1.0e-5);
        assert_eq!(invert_tsyn_lobatto_anchors([5.0, 5.0, 5.0], 5.0), -1.0);
        assert_eq!(invert_tsyn_lobatto_anchors([10.0, 5.0, 0.0], 11.0), -1.0);
        assert_eq!(invert_tsyn_lobatto_anchors([0.0, 5.0, 10.0], 11.0), 1.0);
        assert_eq!(beat_pos_to_mod12_q32(f64::NAN), (0, 0));
        assert_eq!(beat_pos_to_mod12_q32(-1.0).0, 11);
        assert_eq!(host_frame_q64_increment(1), u64::MAX);
        // Exhaustive check of subdivision_phase_q32 against u128 reference math across all
        // 12 ratios, 12 beats, and overflow-stress within-beat phases, plus beat_pos_to_mod12_q32.
        let sample_phases = [0_u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0xC000_0000, u32::MAX];
        for s in 0..12_u8 {
            let (a_s, b_s) = CANONICAL_SUBDIVISION_RATIOS[usize::from(s)];
            for beat in 0..12_u8 {
                for &wb in &sample_phases {
                    let exact = (u128::from(b_s)
                        * ((u128::from(beat % (a_s as u8)) << 32) | u128::from(wb))
                        / u128::from(a_s)) as u32;
                    assert_eq!(subdivision_phase_q32(beat, wb, s, 0), exact);
                }
            }
        }
        assert_eq!(
            subdivision_phase_q32(25, 0, 0, 0),
            subdivision_phase_q32(1, 0, 0, 0)
        );
        for b in -25_i64..=25 {
            assert_eq!(
                beat_pos_to_mod12_q32(b as f64).0,
                (((b % 12) + 12) % 12) as u8
            );
        }
    }
}
