//! Phase-3 pre-clamp `[CLAMP-DYNAMIC]` excursion/wire verification and 256-frame extreme-corner
//! `[STAB-ZERO-DC]` runtime verification (`ref[train.audit.phase3.clamp-dynamic]`,
//! `ref[train.audit.phase3.numeric-envelope]`, `docs/specs/training/control_space_audit.md`).

mod checks;

use dnam_kernels::{
    OP_N_PRE_MAX_ZOH_HOLD_DEPTH, TANK_MAX_G_LIN_F32, TANK_MAX_SIGMA_SAT_FP16, fp16_to_f32,
};
use dnam_vm::bin::DnamBinView;
use dnam_vm::control::bus::laws::stage_latent_dot_fp16;
use dnam_vm::control::{ControlBus, STAGE_TARGET_SLOTS};
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use crate::control::stability::{
    MIN_DELAY_CROSSBAR_SAMP, MIN_DELAY_STANDALONE_SAMP, StageTargetBounds,
};
use crate::error::TrainError;

use checks::{
    check_kautz_blocks, check_stage_latent_wire_tables, check_transport_quads,
    collect_active_transport_quads, collect_kautz_blocks,
};

/// Number of zero-input frames rendered per extreme knob corner (`[STAB-ZERO-DC]`).
pub const ZERO_DC_CORNER_FRAMES: usize = 256;
/// Maximum permitted absolute DC offset across all `256` zero-input corner frames.
pub const ZERO_DC_CORNER_MAX_ABS: f32 = 1.0e-5;
/// Maximum allowed dynamic Kautz modal pole radius (`[CLAMP-DYNAMIC]`).
pub const DYNAMIC_KAUTZ_R_MAX: f32 = 0.9995;
/// Maximum allowed effective `Op-T` damping pole (`[CLAMP-DYNAMIC]`).
pub const DYNAMIC_DAMP_POLE_MAX: f32 = 0.999_95;
/// Maximum allowed dynamic `Op-T` loop feedback gain (`[CLAMP-DYNAMIC]`).
pub const DYNAMIC_G_LOOP_MAX: f32 = 0.9990;
/// Maximum allowed dynamic `Op-T` Schroeder allpass magnitude (`[CLAMP-DYNAMIC]`).
pub const DYNAMIC_ALLPASS_G_MAX: f32 = 0.75;

const _: () = assert!(ZERO_DC_CORNER_FRAMES == 256);
const _: () = assert!(TANK_MAX_G_LIN_F32.to_bits() == 0x3F7F_BE76);
const _: () = assert!(TANK_MAX_SIGMA_SAT_FP16 == 0x3BE0);
const _: () = assert!(OP_N_PRE_MAX_ZOH_HOLD_DEPTH == 0.9999_f32);
const _: () = assert!(DYNAMIC_KAUTZ_R_MAX > 0.0 && DYNAMIC_KAUTZ_R_MAX < 1.0);
const _: () = assert!(DYNAMIC_DAMP_POLE_MAX > 0.0 && DYNAMIC_DAMP_POLE_MAX < 1.0);
const _: () = assert!(DYNAMIC_G_LOOP_MAX >= TANK_MAX_G_LIN_F32 && DYNAMIC_G_LOOP_MAX < 1.0);
const _: () = assert!(DYNAMIC_ALLPASS_G_MAX > 0.0 && DYNAMIC_ALLPASS_G_MAX < 1.0);
const _: () = assert!(STAGE_TARGET_SLOTS == 32);

/// Structured Phase-3 `[CLAMP-DYNAMIC]` corner stability violation (`ref[train.audit.phase3.clamp-dynamic]`).
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicCornerViolation {
    /// Static target on active slot `0..=30` (or reserved slot `31`) fell outside `StageTargetBounds`.
    RawStageTargetBound {
        /// `stage_targets` slot index (`0..32`).
        slot: usize,
        /// Evaluated static target value.
        value: f64,
        /// Permitted lower interval bound.
        lo: f64,
        /// Permitted upper interval bound.
        hi: f64,
    },
    /// Worst-case modulated Kautz pole radius exceeded `DYNAMIC_KAUTZ_R_MAX` (`0.9995`).
    KautzPoleRadius {
        /// Conjugate mode index across the compiled tier's Kautz banks.
        mode_index: usize,
        /// Effective worst-case pole radius `r_eff`.
        r_eff: f64,
        /// Canonical upper ceiling (`0.9995`).
        max_allowed: f64,
    },
    /// Worst-case modulated `Op-T` damping pole exceeded `DYNAMIC_DAMP_POLE_MAX` (`0.99995`).
    TransportDampPole {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Worst-case `damp_base + t[21] + delta[21]`.
        damp_eff: f64,
        /// Canonical upper ceiling (`0.99995`).
        max_allowed: f64,
    },
    /// Minimum dynamic causal delay violated `tau_floor` (`5.0` crossbar / `4.0` standalone).
    TransportDelayFloor {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Delay lane index (`0..4`).
        lane: usize,
        /// Worst-case minimum delay `tau_min` in 48 kHz samples.
        tau_min: f64,
        /// Canonical causal delay floor (`5.0` or `4.0`).
        tau_floor: f64,
    },
    /// Dynamic `Op-T` loop feedback interval exited `[0.0, DYNAMIC_G_LOOP_MAX]`.
    TransportLoopGain {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Minimum dynamic loop gain `g_base + t[13] - delta[13]`.
        g_lo: f64,
        /// Maximum dynamic loop gain `g_base + t[13] + delta[13]`.
        g_hi: f64,
        /// Canonical upper ceiling (`0.9990`).
        max_allowed: f64,
    },
    /// Dynamic `Op-T` Schroeder allpass gain exceeded `DYNAMIC_ALLPASS_G_MAX` (`0.75`).
    TransportAllpassGain {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Allpass lane index (`0..4`).
        lane: usize,
        /// Worst-case `|a_base + t[25]| + delta[25]`.
        abs_max: f64,
        /// Canonical upper ceiling (`0.75`).
        max_allowed: f64,
    },
    /// Wire `sigma_knee_fp16` exceeded `TANK_MAX_SIGMA_SAT_FP16` (`0x3BE0`) or was invalid.
    TransportSigmaKnee {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Raw little-endian `fp16` bits.
        bits: u16,
        /// Canonical maximum bit pattern (`0x3BE0`).
        max_bits: u16,
    },
    /// Base wire `g_loop` exceeded `TANK_MAX_G_LIN_F32` or was non-finite/negative.
    TransportBaseLoopGain {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Base wire `g_loop` value.
        g_base: f32,
        /// Canonical wire ceiling (`TANK_MAX_G_LIN_F32`).
        max_allowed: f32,
    },
    /// Base wire `freeze_mix_fp16` was non-finite or outside `[0.0, 1.0]`.
    TransportFreezeMix {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Decoded `fp16` base `freeze_mix`.
        base_freeze: f32,
    },
    /// Base wire `window_depth_fp16` (or active-ramp modulation) was non-finite or out of `[0.0, 1.0]`.
    TransportWindowDepth {
        /// Active `TransportQuad` index (`0..4`).
        quad_index: usize,
        /// Decoded `fp16` base `window_depth`.
        base_window_depth: f32,
    },
    /// Dynamic `Op-N[Pre]` `zoh_hold_depth` interval exited `[0.0, 0.9999]`.
    OpNPreZohHoldDepth {
        /// Minimum dynamic `zoh_hold_depth`.
        zoh_lo: f64,
        /// Maximum dynamic `zoh_hold_depth`.
        zoh_hi: f64,
        /// Canonical upper ceiling (`0.9999`).
        max_allowed: f64,
    },
    /// Dynamic `op_p_phasor_rate_oct` (`slot 29`) exceeded `4.0` octaves.
    PhasorRateOctave {
        /// Worst-case `|t[29]| + delta[29]`.
        abs_max: f64,
        /// Canonical ceiling (`4.0`).
        max_allowed: f64,
    },
    /// Dynamic `sense_vca_gain_delta` (`slot 30`) exceeded magnitude `1.0`.
    SensoryVcaGainDelta {
        /// Worst-case `|t[30]| + delta[30]`.
        abs_max: f64,
        /// Canonical ceiling (`1.0`).
        max_allowed: f64,
    },
    /// Wire `stage_latent_*` table entry was non-finite, exceeded bounds, or violated reserved zeros.
    StageLatentTableBound {
        /// Wire table name (`stage_latent_mods_fp16`, `stage_latent_ctrl_ultra_fp16`, or `stage_latent_core_mods_fp16`).
        table: &'static str,
        /// Flat element index inside the table.
        index: usize,
        /// Decoded `fp16` value.
        value: f32,
    },
}

impl DynamicCornerViolation {
    /// Returns `true` iff this violation belongs to `PANAMA` candidate exclusion condition (c)
    /// (`ref[train.audit.report.prescription]`).
    #[must_use]
    pub const fn is_panama_exclusion(&self) -> bool {
        matches!(
            self,
            Self::RawStageTargetBound { .. }
                | Self::KautzPoleRadius { .. }
                | Self::TransportDampPole { .. }
                | Self::TransportDelayFloor { .. }
                | Self::TransportLoopGain { .. }
                | Self::TransportAllpassGain { .. }
        )
    }
}

/// Consolidated Phase-3 `[CLAMP-DYNAMIC]` audit report for a single control corner.
#[derive(Clone, Debug, PartialEq)]
pub struct DynamicCornerAuditReport {
    /// True iff no violations occurred.
    pub passed: bool,
    /// Worst dynamic Kautz pole radius inspected (`0.0` when no active Kautz modes exist).
    pub max_kautz_r: f64,
    /// Minimum dynamic causal delay inspected (`tau_floor` when no active `TransportQuad`s exist).
    pub min_tau_samples: f64,
    /// Maximum dynamic `Op-T` loop gain inspected (`0.0` when no active `TransportQuad`s exist).
    pub max_g_loop: f64,
    /// Minimum dynamic `Op-T` loop gain inspected (`0.0` when no active `TransportQuad`s exist).
    pub min_g_loop: f64,
    /// Maximum dynamic `Op-T` allpass magnitude inspected (`0.0` when no active `TransportQuad`s exist).
    pub max_abs_allpass_g: f64,
    /// All stability/wire violations detected at this corner.
    pub violations: Vec<DynamicCornerViolation>,
}

/// Consolidated 256-frame zero-input DC equilibrium report across extreme valid corners (`[STAB-ZERO-DC]`).
#[derive(Clone, Debug, PartialEq)]
pub struct ExtremeCornerDcReport {
    /// Total extreme corners evaluated.
    pub corners_checked: usize,
    /// Maximum absolute L/R sample amplitude across all checked corners and frames.
    pub max_abs_dc: f32,
    /// Pairs `(corner_index, peak_abs)` where `!peak_abs.is_finite() || peak_abs > ZERO_DC_CORNER_MAX_ABS`.
    pub failing_corners: Vec<(usize, f32)>,
    /// True iff `failing_corners.is_empty()`.
    pub passed: bool,
}

/// Computes the per-slot worst-case dynamic L1 excursion `delta_r = sum_{j=0..8} |w_dyn[r, j]| * m_max[j]`
/// over columns `8..16` of `w_ctrl` and optional `"CEXT"` `w_ctrl_ext` (`ref[train.audit.phase3.clamp-dynamic]`).
#[must_use]
pub fn compute_dynamic_excursion_bounds_32(view: &DnamBinView<'_>) -> [f64; STAGE_TARGET_SLOTS] {
    let decode_skew_abs = |raw_bits: u16| -> f64 {
        let v = fp16_to_f32(raw_bits);
        if v.is_finite() {
            f64::from(v.clamp(-1.0, 1.0)).abs()
        } else {
            0.0
        }
    };
    let beta0 = decode_skew_abs(view.arena0.control_bus_lite.lfo_skew_fp16[0].get());
    let beta1 = decode_skew_abs(view.arena0.control_bus_lite.lfo_skew_fp16[1].get());
    let m_max = [1.0 + beta0, 1.0, 1.0 + beta1, 1.0, 1.0, 1.0, 1.0, 1.0];

    let row_l1 = |row: &[zerocopy::byteorder::little_endian::F32; 16]| -> f64 {
        let mut sum = 0.0_f64;
        for j in 0..8 {
            let w = f64::from(row[8 + j].get());
            if !w.is_finite() {
                return f64::INFINITY;
            }
            sum += w.abs() * m_max[j];
        }
        sum
    };

    let mut delta = [0.0_f64; STAGE_TARGET_SLOTS];
    for (r, d) in delta.iter_mut().enumerate().take(16) {
        *d = row_l1(&view.arena0.control_bus_lite.w_ctrl[r]);
    }
    if let Some(cext) = view.cext {
        for r in 0..15 {
            delta[16 + r] = row_l1(&cext.w_ctrl_ext[r]);
        }
    }
    delta[31] = 0.0;
    delta
}

#[inline]
fn eval_row_static(
    w_row: &[f32; 16],
    w_ultra_row: Option<&[f32; 4]>,
    q0: &[f32; 4],
    q1: &[f32; 4],
    qu: Option<&[f32; 4]>,
) -> f64 {
    let mut v = [0.0_f32; 4];
    for lane in 0..4 {
        let s0 = w_row[lane] * q0[lane];
        let s1 = w_row[4 + lane] * q1[lane];
        let su = match (w_ultra_row, qu) {
            (Some(wu), Some(u)) => wu[lane] * u[lane],
            _ => 0.0_f32,
        };
        v[lane] = (s0 + s1) + su;
    }
    f64::from((v[0] + v[1]) + (v[2] + v[3]))
}

/// Evaluates purely static `ControlBus(u)` dispatch targets (`q2 = q3 = ou = 0`, zero resting LFO
/// leakage) and static 8-D stage sensitivity vector (`static_z_stage`) on `bus` (`ref[train.audit.phase3.clamp-dynamic]`).
pub fn eval_static_control_bus_targets(
    bus: &mut ControlBus<'_, DefaultMathBackend>,
    updates: &[(usize, f32)],
) -> Result<([f64; STAGE_TARGET_SLOTS], [f32; 8]), TrainError> {
    bus.set_controls_batch(updates)?;
    bus.is_dirty = true;
    bus.clamp_and_eval_static_if_dirty();

    let q0 = bus.z_head;
    let q1 = bus.ctrl_basis_4;
    let is_ultra = matches!(bus.tier, Tier::Ultra);
    let qu = if is_ultra {
        bus.z_stage_ultra
    } else {
        [0.0_f32; 4]
    };
    let static_z_stage = [q1[0], q1[1], q1[2], q1[3], qu[0], qu[1], qu[2], qu[3]];

    let mut targets = [0.0_f64; STAGE_TARGET_SLOTS];
    for (r, slot_out) in targets.iter_mut().enumerate().take(16) {
        let wu = is_ultra.then_some(&bus.w_ctrl_ultra[r]);
        *slot_out = eval_row_static(&bus.w_ctrl[r], wu, &q0, &q1, is_ultra.then_some(&qu));
    }
    if bus.cext_present {
        for r in 0..15 {
            let wu = is_ultra.then_some(&bus.w_ctrl_ultra_ext[r]);
            targets[16 + r] =
                eval_row_static(&bus.w_ctrl_ext[r], wu, &q0, &q1, is_ultra.then_some(&qu));
        }
    }
    targets[31] = 0.0;
    Ok((targets, static_z_stage))
}

/// Checks Phase-3 `[CLAMP-DYNAMIC]` stability and wire invariants for one corner's static
/// `stage_targets` and `static_z_stage` (`ref[train.audit.phase3.clamp-dynamic]`).
#[must_use]
pub fn check_corner_dynamic_stability(
    view: &DnamBinView<'_>,
    tier: Tier,
    static_targets: &[f64; STAGE_TARGET_SLOTS],
    static_z_stage: &[f32; 8],
    delta_32: &[f64; STAGE_TARGET_SLOTS],
    crossbar_coupled: bool,
) -> DynamicCornerAuditReport {
    let mut violations = Vec::new();
    let tau_floor = if crossbar_coupled {
        MIN_DELAY_CROSSBAR_SAMP
    } else {
        MIN_DELAY_STANDALONE_SAMP
    };
    let active_quads = collect_active_transport_quads(view, tier);
    let min_base_delay = active_quads
        .iter()
        .flat_map(|&(_, q)| q.delay_samples.iter().map(|d| f64::from(d.get())))
        .filter(|d| d.is_finite())
        .fold(f64::INFINITY, f64::min);
    let eff_base_delay = if min_base_delay.is_finite() {
        min_base_delay.max(tau_floor)
    } else {
        tau_floor
    };
    let bounds = StageTargetBounds::canonical(crossbar_coupled, eff_base_delay);

    for (slot, &val) in static_targets.iter().enumerate().take(31) {
        if !val.is_finite() || val < bounds.min[slot] || val > bounds.max[slot] {
            violations.push(DynamicCornerViolation::RawStageTargetBound {
                slot,
                value: val,
                lo: bounds.min[slot],
                hi: bounds.max[slot],
            });
        }
    }
    if !static_targets[31].is_finite() || static_targets[31].abs() > 0.0 {
        violations.push(DynamicCornerViolation::RawStageTargetBound {
            slot: 31,
            value: static_targets[31],
            lo: 0.0,
            hi: 0.0,
        });
    }

    let mods_u16 = view
        .arena0
        .linear_readout_lite
        .stage_latent_mods_fp16
        .map(|w| w.get());
    let w_kautz_damp_dot =
        stage_latent_dot_fp16(&mods_u16[32..40], static_z_stage, tier.stage_latent_dim());
    let damp_min =
        (static_targets[11] + f64::from(w_kautz_damp_dot) - delta_32[11]).clamp(-1.5, 1.5);
    let kautz_blocks = collect_kautz_blocks(view, tier);
    let max_kautz_r = check_kautz_blocks(&kautz_blocks, damp_min, &mut violations);

    let (min_tau_samples, max_g_loop, min_g_loop, max_abs_allpass_g) = check_transport_quads(
        &active_quads,
        static_targets,
        delta_32,
        tau_floor,
        &mut violations,
    );

    let zoh_base = f64::from(view.arena0.crossbar.op_n_pre_base.zoh_hold_depth.get());
    let zoh_lo = zoh_base + static_targets[27] - delta_32[27];
    let zoh_hi = zoh_base + static_targets[27] + delta_32[27];
    let zoh_max = f64::from(OP_N_PRE_MAX_ZOH_HOLD_DEPTH);
    if !zoh_lo.is_finite() || !zoh_hi.is_finite() || zoh_lo < 0.0 || zoh_hi > zoh_max {
        violations.push(DynamicCornerViolation::OpNPreZohHoldDepth {
            zoh_lo,
            zoh_hi,
            max_allowed: zoh_max,
        });
    }

    let phasor_abs = static_targets[29].abs() + delta_32[29];
    if !phasor_abs.is_finite() || phasor_abs > 4.0 {
        violations.push(DynamicCornerViolation::PhasorRateOctave {
            abs_max: phasor_abs,
            max_allowed: 4.0,
        });
    }

    let vca_abs = static_targets[30].abs() + delta_32[30];
    if !vca_abs.is_finite() || vca_abs > 1.0 {
        violations.push(DynamicCornerViolation::SensoryVcaGainDelta {
            abs_max: vca_abs,
            max_allowed: 1.0,
        });
    }

    check_stage_latent_wire_tables(view, tier, &mut violations);

    DynamicCornerAuditReport {
        passed: violations.is_empty(),
        max_kautz_r,
        min_tau_samples,
        max_g_loop,
        min_g_loop,
        max_abs_allpass_g,
        violations,
    }
}

/// Evaluates `ZERO_DC_CORNER_FRAMES` (`256`) zero-input frames per extreme valid knob corner on a
/// single persistent `SonaVm` against `ZERO_DC_CORNER_MAX_ABS` (`1e-5`) (`[STAB-ZERO-DC]`).
pub fn verify_extreme_corners_zero_dc_256(
    view: &DnamBinView<'_>,
    tier: Tier,
    corners: &[Vec<(usize, f32)>],
) -> Result<ExtremeCornerDcReport, TrainError> {
    let mut vm = SonaVm::<DefaultMathBackend>::from_view_owned(view, tier, 48_000.0, true)?;
    let in_l = [0.0_f32; ZERO_DC_CORNER_FRAMES];
    let in_r = [0.0_f32; ZERO_DC_CORNER_FRAMES];
    let mut out_l = [0.0_f32; ZERO_DC_CORNER_FRAMES];
    let mut out_r = [0.0_f32; ZERO_DC_CORNER_FRAMES];

    let mut max_abs_dc = 0.0_f32;
    let mut failing_corners = Vec::new();
    for (idx, corner) in corners.iter().enumerate() {
        vm.reset_and_set_controls_immediate(corner)?;
        out_l.fill(0.0);
        out_r.fill(0.0);
        vm.process_block(&in_l, &in_r, &mut out_l, &mut out_r)?;

        let mut peak = 0.0_f32;
        for (&l, &r) in out_l.iter().zip(out_r.iter()) {
            if !l.is_finite() || !r.is_finite() {
                peak = f32::INFINITY;
                break;
            }
            peak = peak.max(l.abs()).max(r.abs());
        }
        if peak.is_finite() {
            max_abs_dc = max_abs_dc.max(peak);
        } else {
            max_abs_dc = f32::INFINITY;
        }
        if !peak.is_finite() || peak > ZERO_DC_CORNER_MAX_ABS {
            failing_corners.push((idx, peak));
        }
    }

    Ok(ExtremeCornerDcReport {
        corners_checked: corners.len(),
        max_abs_dc,
        passed: failing_corners.is_empty(),
        failing_corners,
    })
}

#[cfg(test)]
mod tests {
    use dnam_kernels::f32_to_fp16;
    use dnam_vm::Tier;
    use dnam_vm::bin::DnamBinView;
    use dnam_vm::control::ControlBus;
    use dnam_vm::math::DefaultMathBackend;
    use zerocopy::byteorder::little_endian::{F32, U16};

    use crate::solver::model::SonaModelF64;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};

    use super::*;

    fn build_clean_ultra_dnam_bin(dc_carrier: bool) -> Vec<u8> {
        let mut model = SonaModelF64::new(48_000.0);
        model.ro_pole_48k = 0.99;
        model.stage_latent_mods.core_mods_ultra = Some(Box::new([[0.05; 120]; 8]));
        for tq in &mut model.transport_quads {
            tq.delay_samples = [32.0; 4];
            tq.g_loop = 0.5;
            tq.damp_pole = 0.25;
            tq.allpass_g = [0.25; 4];
            tq.window_depth = 0.5;
            tq.freeze_mix = 0.0;
        }
        if dc_carrier {
            model.carrier_basis_w[2][0] = 1.0;
            model.phasor_freq_hz[0] = 0.0;
            model.phasor_init_phase_q32[0] = 0x4000_0000;
            model.crossbar_op_gains[0] = 1.0;
            model.phasor_to_core[0][0] = 1.0;
            model.core_quads[0].tone_dir[0] = 1.0;
            model.fir_l[0] = 1.0;
            model.w_ro[0] = [1.0; 12];
        }
        encode_model_to_dnam_bin(
            &model,
            Pass5EmitConfig {
                calibrate_lufs_trims: false,
                ..Pass5EmitConfig::default()
            },
        )
        .expect("clean Ultra model encodes")
    }

    #[test]
    fn test_dynamic_excursion_bounds_and_static_eval_no_lfo_leakage() {
        let bytes = build_clean_ultra_dnam_bin(false);
        let mut view_owned = DnamBinView::parse(&bytes).unwrap();
        let mut a0 = *view_owned.arena0;
        a0.control_bus_lite.lfo_skew_fp16 =
            [U16::new(f32_to_fp16(0.5)), U16::new(f32_to_fp16(-0.25))];
        a0.control_bus_lite.w_ctrl[13][8] = F32::new(0.1);
        a0.control_bus_lite.w_ctrl[13][12] = F32::new(-0.2);
        let cext = view_owned.cext.copied().map(|mut c| {
            c.w_ctrl_ext[5][10] = F32::new(-0.08);
            c
        });
        view_owned.arena0 = &a0;
        if let Some(ref c) = cext {
            view_owned.cext = Some(c);
        }

        let delta = compute_dynamic_excursion_bounds_32(&view_owned);
        assert!((delta[13] - (0.1 * 1.5 + 0.2 * 1.0)).abs() < 1.0e-6);
        assert_eq!(delta[31], 0.0);
        if cext.is_some() {
            assert!((delta[21] - 0.08 * 1.25).abs() < 1.0e-6);
        }

        let mut bus = ControlBus::<DefaultMathBackend>::from_dnam_bin_owned(
            &view_owned,
            Tier::Lite,
            48_000.0,
        )
        .unwrap();
        let (t_static, z_static) = eval_static_control_bus_targets(&mut bus, &[]).unwrap();
        assert_eq!(t_static[13], 0.0);
        assert_eq!(t_static[31], 0.0);
        assert_eq!(z_static, [0.0; 8]);
        assert!((bus.stage_targets()[13] - 0.1 * 1.5).abs() < 1.0e-5);
    }

    #[test]
    fn test_check_corner_dynamic_stability_and_zero_dc_256_pass_and_fail() {
        let bytes = build_clean_ultra_dnam_bin(false);
        let view = DnamBinView::parse(&bytes).unwrap();
        let t_zero = [0.0_f64; STAGE_TARGET_SLOTS];
        let z_zero = [0.0_f32; 8];
        let d_zero = [0.0_f64; STAGE_TARGET_SLOTS];

        let ok_rep =
            check_corner_dynamic_stability(&view, Tier::Ultra, &t_zero, &z_zero, &d_zero, true);
        assert!(ok_rep.passed, "expected pass, got {:?}", ok_rep.violations);
        assert!(ok_rep.min_tau_samples >= MIN_DELAY_CROSSBAR_SAMP);

        let mut t_bad = t_zero;
        t_bad[14] = 4.0;
        t_bad[27] = 1.0;
        t_bad[29] = 4.5;
        t_bad[30] = 1.5;
        let bad_rep =
            check_corner_dynamic_stability(&view, Tier::Ultra, &t_bad, &z_zero, &d_zero, true);
        assert!(bad_rep.violations.iter().any(|v| {
            matches!(
                v,
                DynamicCornerViolation::RawStageTargetBound { slot: 14, .. }
            ) && v.is_panama_exclusion()
        }));
        assert!(
            bad_rep
                .violations
                .iter()
                .any(|v| matches!(v, DynamicCornerViolation::OpNPreZohHoldDepth { .. }))
        );
        assert!(
            bad_rep
                .violations
                .iter()
                .any(|v| matches!(v, DynamicCornerViolation::PhasorRateOctave { .. }))
        );
        assert!(
            bad_rep
                .violations
                .iter()
                .any(|v| matches!(v, DynamicCornerViolation::SensoryVcaGainDelta { .. }))
        );

        let dc_ok = verify_extreme_corners_zero_dc_256(&view, Tier::Ultra, &[vec![], vec![]])
            .expect("zero-DC pass evaluation succeeds");
        assert!(dc_ok.passed, "expected zero DC, got {dc_ok:?}");
        assert!(dc_ok.max_abs_dc <= ZERO_DC_CORNER_MAX_ABS);

        let dc_bytes = build_clean_ultra_dnam_bin(true);
        let dc_view = DnamBinView::parse(&dc_bytes).unwrap();
        let dc_fail = verify_extreme_corners_zero_dc_256(&dc_view, Tier::Ultra, &[vec![]])
            .expect("zero-DC failure evaluation succeeds");
        assert!(!dc_fail.passed);
        assert_eq!(dc_fail.failing_corners.len(), 1);
        assert!(dc_fail.max_abs_dc > ZERO_DC_CORNER_MAX_ABS);
    }
}
