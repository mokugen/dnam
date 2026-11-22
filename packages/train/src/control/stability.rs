//! Phase-2 intrinsic `SONA` parameter clamps, smooth log-barrier corner evaluator,
//! and 2,048-sample differential stereo `[+x, -x]` `SonaVm` rail-lockup burst
//! (`ref[train.audit.phase2.corner-barrier]`, `ref[train.audit.phase2.clamp-rail]`,
//! `docs/specs/training/control_space_audit.md`).

use core::f64::consts::PI;

mod rail_and_arena;

pub use rail_and_arena::{
    RailBurstReport, check_preact_occupancy, enforce_arena_block0_k_nl_and_clamp,
    enforce_arena0_stability_invariants, run_phase2_rail_burst,
};

use dnam_kernels::{
    MAX_LOOP_GAIN_F32, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32, MIN_SAG_GAIN, MIN_TPT_G,
    OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, OP_N_PRE_MIN_RHO_EFF, TANK_MAX_G_LIN_F32,
    TANK_MAX_SIGMA_SAT_F32, TANK_MAX_V_CEIL_F32,
};
use dnam_vm::control::STAGE_TARGET_SLOTS;

/// Canonical maximum `48 kHz` `LinearReadout` leak pole (`0x3F7F_FFAC`, `ref[train.audit.phase3.clamp-head-gain]`).
pub const MAX_RO_POLE_48K_F32: f32 = MAX_PARALRU_R_F32;
const _: () = assert!(MAX_RO_POLE_48K_F32.to_bits() == 0x3F7F_FFAC);
/// `binary64` representation of [`MAX_RO_POLE_48K_F32`].
pub const MAX_RO_POLE_48K_F64: f64 = MAX_RO_POLE_48K_F32 as f64;
/// Minimum causal delay in samples inside top-of-frame crossbar blocks (`ref[dsp.op-t.ring.declaration]`).
pub const MIN_DELAY_CROSSBAR_SAMP: f64 = 5.0;
/// Minimum causal delay in samples for standalone `Op-T` blocks.
pub const MIN_DELAY_STANDALONE_SAMP: f64 = 4.0;
/// Maximum `PolyADAA1` octave shift in `[-1.5, +1.5]`.
pub const MAX_BLOCK_ADAA_OCT: f64 = 1.5;
/// Maximum symmetric `glu_mix` scalar in `[-4.0, +4.0]`.
pub const MAX_GLU_MIX: f64 = 4.0;
/// Maximum `Op-P` phasor rate excursion in `[-4.0, +4.0]` octaves.
pub const MAX_OP_P_PHASOR_RATE_OCT: f64 = 4.0;
/// Maximum `SensoryQuad` VCA modulation weight magnitude.
pub const MAX_SENSE_VCA_W: f64 = 512.0;
/// Maximum `SensoryQuad` tone modulation weight magnitude.
pub const MAX_SENSE_TONE_W: f64 = 16.0;
/// Maximum `SensoryQuad` input envelope gain.
pub const MAX_SENSE_G_IN: f64 = 2.0;
/// Pre-activation hit threshold (`|u_nl| >= 8.0`).
pub const RAIL_PREACT_HIT_THRESHOLD: f64 = 8.0;
/// Output rail-hit threshold (`|y_c[n]| >= 25.0`).
pub const RAIL_OUTPUT_HIT_THRESHOLD: f32 = 25.0;
/// Hard numeric unclipped output ceiling (`100.0`).
pub const RAIL_OUTPUT_CEIL: f32 = 100.0;
/// Maximum fraction of burst samples allowed at saturation occupancy (`0.95`).
pub const SAT_OCCUPANCY_LIMIT: f64 = 0.95;
/// Differential stereo `[CLAMP-RAIL]` burst frame count.
pub const RAIL_BURST_FRAMES: usize = 2_048;
/// Smooth Phase-2 log-barrier positivity floor (`1e-7`).
pub const CORNER_BARRIER_FLOOR: f64 = 1.0e-7;

pub(super) const RAIL_STIM_OMEGA_A: f64 = 0.065_449_85;
pub(super) const RAIL_STIM_OMEGA_B: f64 = 0.219_911_49;

const _: () = assert!(MAX_RO_POLE_48K_F32 < 1.0);
const _: () = assert!(MIN_DELAY_CROSSBAR_SAMP >= MIN_DELAY_STANDALONE_SAMP);
const _: () = assert!(RAIL_BURST_FRAMES == 2_048);

/// Closed-interval `[lo, hi]` stability bounds across all 32 `stage_targets` slots.
#[derive(Clone, Debug, PartialEq)]
pub struct StageTargetBounds {
    /// Lower bound per `stage_targets[0..32]` slot.
    pub min: [f64; STAGE_TARGET_SLOTS],
    /// Upper bound per `stage_targets[0..32]` slot.
    pub max: [f64; STAGE_TARGET_SLOTS],
}

impl StageTargetBounds {
    /// Returns normative Phase-1/Phase-2 `SONA v3.0` stability bounds for all 32 stage targets
    /// (`ref[dsp.ctrl.dispatch.matrix]`, `ref[train.audit.phase1.clamp-stability]`).
    #[must_use]
    pub fn canonical(crossbar_coupled: bool, min_tau_samp: f64) -> Self {
        let mut min = [0.0_f64; STAGE_TARGET_SLOTS];
        let mut max = [0.0_f64; STAGE_TARGET_SLOTS];

        let delay_floor = if crossbar_coupled {
            MIN_DELAY_CROSSBAR_SAMP
        } else {
            MIN_DELAY_STANDALONE_SAMP
        };
        let eff_delay_min = min_tau_samp.max(delay_floor);

        // Slots 0..2: Core TPT-SVF cutoff/resonance (`svf_g_oct` [-3, 3], `svf_k_delta` [-0.8, 1.5]).
        min[0] = -3.0;
        max[0] = 3.0;
        min[1] = -0.8;
        max[1] = 1.5;

        // Slots 2..6: Core PolyADAA1 `block_adaa_oct[0..4]` (`[-1.5, +1.5]`).
        for i in 2..6 {
            min[i] = -MAX_BLOCK_ADAA_OCT;
            max[i] = MAX_BLOCK_ADAA_OCT;
        }

        // Slots 6..10: Core ParaLRU `block_lru_bias[0..4]` (`[-2.0, +2.0]`).
        for i in 6..10 {
            min[i] = -2.0;
            max[i] = 2.0;
        }

        // Slots 10..12: ModalConvolver Kautz (`kautz_oct_delta`, `kautz_damp_delta` in `[-1.5, +1.5]`).
        min[10] = -1.5;
        max[10] = 1.5;
        min[11] = -1.5;
        max[11] = 1.5;

        // Slots 12..16: Op-T transport (`tank_delay_mod`, `tank_gain_mod`, `tank_psi_delta`, `crossbar_wet_mod`).
        min[12] = -(48_000.0 - eff_delay_min).max(1.0);
        max[12] = 48_000.0;
        min[13] = -1.0;
        max[13] = f64::from(TANK_MAX_G_LIN_F32);
        min[14] = -PI;
        max[14] = PI;
        min[15] = -1.0;
        max[15] = 1.0;

        // Slots 16..31 ("CEXT" dispatch).
        min[16] = -1.0;
        max[16] = 1.0; // `op_t_freeze_mix_delta`
        min[17] = -3.0;
        max[17] = 3.0; // `op_t_delay_scale_oct`
        min[18] = -16.0;
        max[18] = 16.0; // `op_t_ramp_rate_hz_delta`
        min[19] = -16.0;
        max[19] = 16.0; // `op_t_lfo_rate_hz_delta`
        min[20] = -4.0;
        max[20] = 4.0; // `op_t_lfo_depth_scale`
        min[21] = -1.0;
        max[21] = 0.999_95; // `op_t_damp_pole_delta`
        min[22] = -2.0;
        max[22] = 2.0; // `op_t_dry_mix_mod`
        min[23] = -4.0;
        max[23] = 4.0; // `op_t_input_gain_mod`
        min[24] = -1.0;
        max[24] = 1.0; // `op_t_window_depth_delta`
        min[25] = -0.75;
        max[25] = 0.75; // `op_t_allpass_g_delta`
        min[26] = -1.0;
        max[26] = 1.0; // `op_n_pre_gamma_bc_delta`
        min[27] = -1.0;
        max[27] = f64::from(OP_N_PRE_MAX_ZOH_HOLD_DEPTH); // `op_n_pre_zoh_hold_delta`
        min[28] = -16.0;
        max[28] = 16.0; // `op_n_pre_beta_q_delta`
        min[29] = -MAX_OP_P_PHASOR_RATE_OCT;
        max[29] = MAX_OP_P_PHASOR_RATE_OCT; // `op_p_phasor_rate_oct`
        min[30] = -1.0;
        max[30] = 1.0; // `sense_vca_gain_delta`

        // Slot 31 is reserved strictly at `+0.0`.
        min[31] = 0.0;
        max[31] = 0.0;

        Self { min, max }
    }
}

/// Scalar core parameter bundle for authoritative `SONA` bounds projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SonaCoreStabilityParams {
    /// `ParaLRU` pole radius `r`.
    pub paralru_r: f32,
    /// `ParaLRU` selective-Duhem stiffness `gamma`.
    pub paralru_gamma: f32,
    /// `PolyADAA1` octave excursion relative to base slope.
    pub block_adaa_oct: f32,
    /// Symmetric `GLU` residual blend weight.
    pub glu_mix: f32,
    /// Sag supply gain floor clamp.
    pub sag_gain: f32,
    /// `TPT-SVF` prewarped tan cutoff `g`.
    pub tpt_g: f32,
    /// `Op-T` causal transport delay in samples.
    pub delay_samples: f64,
    /// `Op-T` coercive loop gain `||G_loop||`.
    pub loop_gain: f32,
    /// Tank feedback linear scalar `g_lin`.
    pub tank_g_lin: f32,
    /// Tank knee ceiling `sigma_sat`.
    pub tank_sigma_sat: f32,
    /// Tank headroom ceiling `v_ceil`.
    pub tank_v_ceil: f32,
    /// `Op-N[Pre]` zero-order-hold depth in `[0.0, 0.9999]`.
    pub zoh_hold_depth: f32,
    /// `Op-N[Pre]` Fourier smoothness sharpness `beta_q`.
    pub beta_q: f32,
    /// `Op-N[Pre]` minimum active density `rho_eff`.
    pub rho_eff: f32,
    /// `Op-P` octave rate excursion `[-4.0, +4.0]`.
    pub phasor_rate_oct: f32,
    /// `SensoryQuad` input envelope gain `g_in`.
    pub sense_g_in: f32,
    /// `SensoryQuad` VCA bias weight `sense_vca_w`.
    pub sense_vca_w: f32,
    /// `SensoryQuad` tone modulation weight `sense_tone_w`.
    pub sense_tone_w: f32,
    /// Learned `48 kHz` `LinearReadout` leak pole.
    pub ro_pole_48k: f32,
    /// Head DC offset `[STAB-ZERO-DC]` (clamped strictly to `0.0`).
    pub b_head: f64,
}

impl SonaCoreStabilityParams {
    /// Clamps every core parameter to its canonical `dnam-kernels` / `dnam-vm` stability domain.
    pub fn clamp_in_place(&mut self, crossbar_coupled: bool) -> bool {
        let mut modified = false;
        let clamp_f32 = |v: &mut f32, lo: f32, hi: f32, mod_flag: &mut bool| {
            let next = if v.is_finite() { v.clamp(lo, hi) } else { lo };
            if next != *v {
                *v = next;
                *mod_flag = true;
            }
        };

        clamp_f32(&mut self.paralru_r, 0.0, MAX_PARALRU_R_F32, &mut modified);
        clamp_f32(
            &mut self.paralru_gamma,
            0.0,
            MAX_PARALRU_DUHEM_GAMMA_F32,
            &mut modified,
        );
        clamp_f32(
            &mut self.block_adaa_oct,
            -MAX_BLOCK_ADAA_OCT as f32,
            MAX_BLOCK_ADAA_OCT as f32,
            &mut modified,
        );
        clamp_f32(
            &mut self.glu_mix,
            -MAX_GLU_MIX as f32,
            MAX_GLU_MIX as f32,
            &mut modified,
        );
        clamp_f32(&mut self.sag_gain, MIN_SAG_GAIN, 1.0, &mut modified);
        clamp_f32(&mut self.tpt_g, MIN_TPT_G, 64.0, &mut modified);

        let delay_min = if crossbar_coupled {
            MIN_DELAY_CROSSBAR_SAMP
        } else {
            MIN_DELAY_STANDALONE_SAMP
        };
        if !self.delay_samples.is_finite() || self.delay_samples < delay_min {
            self.delay_samples = delay_min;
            modified = true;
        } else if self.delay_samples > 48_000.0 {
            self.delay_samples = 48_000.0;
            modified = true;
        }

        clamp_f32(
            &mut self.loop_gain,
            -MAX_LOOP_GAIN_F32,
            MAX_LOOP_GAIN_F32,
            &mut modified,
        );
        clamp_f32(&mut self.tank_g_lin, 0.0, TANK_MAX_G_LIN_F32, &mut modified);
        clamp_f32(
            &mut self.tank_sigma_sat,
            1.0e-3,
            TANK_MAX_SIGMA_SAT_F32,
            &mut modified,
        );
        clamp_f32(
            &mut self.tank_v_ceil,
            MIN_TPT_G,
            TANK_MAX_V_CEIL_F32,
            &mut modified,
        );
        clamp_f32(
            &mut self.zoh_hold_depth,
            0.0,
            OP_N_PRE_MAX_ZOH_HOLD_DEPTH,
            &mut modified,
        );
        clamp_f32(&mut self.beta_q, 0.0, OP_N_PRE_MAX_BETA_Q, &mut modified);
        clamp_f32(&mut self.rho_eff, OP_N_PRE_MIN_RHO_EFF, 1.0, &mut modified);
        clamp_f32(
            &mut self.phasor_rate_oct,
            -MAX_OP_P_PHASOR_RATE_OCT as f32,
            MAX_OP_P_PHASOR_RATE_OCT as f32,
            &mut modified,
        );
        clamp_f32(
            &mut self.sense_g_in,
            0.0,
            MAX_SENSE_G_IN as f32,
            &mut modified,
        );
        clamp_f32(
            &mut self.sense_vca_w,
            -MAX_SENSE_VCA_W as f32,
            MAX_SENSE_VCA_W as f32,
            &mut modified,
        );
        clamp_f32(
            &mut self.sense_tone_w,
            -MAX_SENSE_TONE_W as f32,
            MAX_SENSE_TONE_W as f32,
            &mut modified,
        );
        clamp_f32(
            &mut self.ro_pole_48k,
            0.0,
            MAX_RO_POLE_48K_F32,
            &mut modified,
        );

        // `[STAB-ZERO-DC]`: zero head DC bias strictly preserved across tiers.
        if self.b_head != 0.0 {
            self.b_head = 0.0;
            modified = true;
        }
        modified
    }
}

/// Evaluates `[0..32]` `stage_targets` against `bounds` using non-negative bounded log-barrier
/// margins (`-ln(min(1, max(margin, 1e-7) / scale))`) plus quadratic infeasible penalties
/// (`1e4 + 1e5 * viol^2`) and `1e6` non-finite penalty (`ref[train.audit.phase2.corner-barrier]`).
#[must_use]
pub fn eval_corner_barrier_loss_and_grad(
    stage_targets: &[f64; STAGE_TARGET_SLOTS],
    bounds: &StageTargetBounds,
    weight: f64,
    mut grad_opt: Option<&mut [f64; STAGE_TARGET_SLOTS]>,
) -> (f64, f64) {
    if let Some(ref mut g) = grad_opt {
        g.fill(0.0);
    }
    if !weight.is_finite() || stage_targets.iter().any(|v| !v.is_finite()) {
        return (1.0e6, f64::INFINITY);
    }
    if weight <= 0.0 {
        return (0.0, 0.0);
    }
    let mut loss = 0.0_f64;
    let mut max_viol = 0.0_f64;
    for i in 0..STAGE_TARGET_SLOTS {
        let lo = bounds.min[i];
        let hi = bounds.max[i];
        let v = stage_targets[i];
        if !lo.is_finite() || !hi.is_finite() || hi < lo {
            return (1.0e6, f64::INFINITY);
        }
        let mut gi = 0.0_f64;
        if (hi - lo).abs() <= 1.0e-12 {
            // Reserved slot (`31` = +0.0) exempt from open-interior log repulsion.
            let viol = v - lo;
            if viol != 0.0 {
                let av = viol.abs();
                max_viol = max_viol.max(av);
                loss += weight * (1.0e4 + 1.0e5 * viol * viol);
                gi += weight * 2.0e5 * viol;
            }
            if let Some(ref mut g) = grad_opt {
                g[i] += gi;
            }
            continue;
        }

        let scale = (0.5 * (hi - lo)).clamp(1.0e-2, 1.0);
        let floor_pen = -(CORNER_BARRIER_FLOOR / scale).min(1.0).ln();

        let margin_lo = v - lo;
        if margin_lo < 0.0 {
            let def = -margin_lo;
            max_viol = max_viol.max(def);
            loss += weight * (floor_pen + 1.0e4 + 1.0e5 * def * def);
            gi -= weight * 2.0e5 * def;
        } else if margin_lo <= CORNER_BARRIER_FLOOR {
            let def = CORNER_BARRIER_FLOOR - margin_lo;
            max_viol = max_viol.max(def);
            loss += weight * (floor_pen + def / CORNER_BARRIER_FLOOR + 1.0e5 * def * def);
            gi -= weight * (1.0 / CORNER_BARRIER_FLOOR + 2.0e5 * def);
        } else if margin_lo < scale {
            loss -= weight * (margin_lo / scale).ln();
            gi -= weight / margin_lo;
        }

        let margin_hi = hi - v;
        if margin_hi < 0.0 {
            let def = -margin_hi;
            max_viol = max_viol.max(def);
            loss += weight * (floor_pen + 1.0e4 + 1.0e5 * def * def);
            gi += weight * 2.0e5 * def;
        } else if margin_hi <= CORNER_BARRIER_FLOOR {
            let def = CORNER_BARRIER_FLOOR - margin_hi;
            max_viol = max_viol.max(def);
            loss += weight * (floor_pen + def / CORNER_BARRIER_FLOOR + 1.0e5 * def * def);
            gi += weight * (1.0 / CORNER_BARRIER_FLOOR + 2.0e5 * def);
        } else if margin_hi < scale {
            loss -= weight * (margin_hi / scale).ln();
            gi += weight / margin_hi;
        }

        if let Some(ref mut g) = grad_opt {
            g[i] += gi;
        }
    }
    (loss, max_viol)
}

/// Evaluates stop-gradient FM corner residual penalty `L_barrier` alongside smooth physical
/// interior log barriers `L_phys` (`r_w <= 0.9995`, `g_lin <= 0.9990`, `|a_diff| <= 0.75`,
/// `tau_min >= tau_floor`) (`ref[train.audit.phase2.corner-barrier]`).
#[must_use]
pub fn eval_stop_grad_corner_and_phys_barrier(
    delta_theta_fm_corners: &[&[f64; STAGE_TARGET_SLOTS]],
    sg_theta_main_corners: &[&[f64; STAGE_TARGET_SLOTS]],
    bounds: &StageTargetBounds,
    lambda_bar: f64,
    phys_state: &[f64; 4],
    crossbar_coupled: bool,
) -> f64 {
    let [r_worst, g_lin, a_diff_abs, tau_min] = *phys_state;
    if ![lambda_bar, r_worst, g_lin, a_diff_abs, tau_min]
        .iter()
        .all(|v| v.is_finite())
    {
        return 1.0e6;
    }
    let tau_floor = if crossbar_coupled {
        MIN_DELAY_CROSSBAR_SAMP
    } else {
        MIN_DELAY_STANDALONE_SAMP
    };
    let mut total = 0.0_f64;
    if lambda_bar > 0.0 {
        for (sg_m, dfm) in sg_theta_main_corners.iter().zip(delta_theta_fm_corners) {
            for i in 0..STAGE_TARGET_SLOTS {
                if !sg_m[i].is_finite() || !dfm[i].is_finite() {
                    return 1.0e6;
                }
                let combined = sg_m[i] + dfm[i];
                let viol_hi = (combined - bounds.max[i]).max(0.0);
                let viol_lo = (bounds.min[i] - combined).max(0.0);
                let v = viol_hi + viol_lo;
                total += lambda_bar * v * v;
            }
        }
    }
    let phys_terms = [
        (0.9995 - r_worst, 0.01),
        (r_worst, 0.01),
        (f64::from(TANK_MAX_G_LIN_F32) - g_lin, 0.01),
        (g_lin, 0.01),
        (0.75 - a_diff_abs.abs(), 0.05),
        (tau_min - tau_floor, 1.0),
    ];
    for (margin, scale) in phys_terms {
        if margin < 0.0 {
            let viol = -margin;
            total += 1.0e4 + 1.0e5 * viol * viol;
        }
        let ratio = (margin.max(CORNER_BARRIER_FLOOR) / scale).min(1.0);
        total -= 1.0e-4 * ratio.ln();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_bounds_clamp_and_corner_barrier_fd() {
        let mut p = SonaCoreStabilityParams {
            paralru_r: 1.5,
            paralru_gamma: 90.0,
            block_adaa_oct: -3.2,
            glu_mix: 9.0,
            sag_gain: 0.001,
            tpt_g: 0.0,
            delay_samples: 2.5,
            loop_gain: 1.05,
            tank_g_lin: 1.2,
            tank_sigma_sat: 2.0,
            tank_v_ceil: 5.0,
            zoh_hold_depth: 1.0,
            beta_q: 32.0,
            rho_eff: 0.0,
            phasor_rate_oct: -8.0,
            sense_g_in: 5.0,
            sense_vca_w: -900.0,
            sense_tone_w: 32.0,
            ro_pole_48k: 1.0,
            b_head: 0.25,
        };
        assert!(p.clamp_in_place(true));
        assert!(p.paralru_r <= MAX_PARALRU_R_F32);
        assert_eq!(p.delay_samples, 5.0);
        assert_eq!(p.b_head, 0.0);
        assert_eq!(p.ro_pole_48k, MAX_RO_POLE_48K_F32);

        let bounds = StageTargetBounds::canonical(true, 5.0);
        let mut st: [f64; STAGE_TARGET_SLOTS] =
            core::array::from_fn(|i| 0.5 * (bounds.min[i] + bounds.max[i]));
        st[2] = 0.7;
        st[10] = 0.4;
        st[16] = 0.55;
        st[31] = 0.0;
        let mut grad = [0.0_f64; STAGE_TARGET_SLOTS];
        let (l0, viol) = eval_corner_barrier_loss_and_grad(&st, &bounds, 0.25, Some(&mut grad));
        assert!(l0 >= 0.0 && l0.is_finite() && viol == 0.0);
        st[31] = 0.45;
        let (l1, viol1) = eval_corner_barrier_loss_and_grad(&st, &bounds, 0.25, Some(&mut grad));
        assert!(l1 > 1.0e3 && viol1 == 0.45);
        st[31] = 0.0;
        let eps = 1.0e-6_f64;
        for idx in [0, 2, 10, 16, 31] {
            let mut st_fd = st;
            if idx == 31 {
                st_fd[31] = 0.2;
            }
            let _ = eval_corner_barrier_loss_and_grad(&st_fd, &bounds, 0.25, Some(&mut grad));
            let (mut sp, mut sm) = (st_fd, st_fd);
            sp[idx] += eps;
            sm[idx] -= eps;
            let fd = (eval_corner_barrier_loss_and_grad(&sp, &bounds, 0.25, None).0
                - eval_corner_barrier_loss_and_grad(&sm, &bounds, 0.25, None).0)
                / (2.0 * eps);
            assert!((grad[idx] - fd).abs() <= 1.0e-5 * grad[idx].abs().max(1.0));
        }
        assert_eq!(
            eval_corner_barrier_loss_and_grad(
                &[f64::NAN; STAGE_TARGET_SLOTS],
                &bounds,
                0.25,
                None,
            )
            .0,
            1.0e6
        );

        let sg_corner = st;
        let fm_zero = [0.0_f64; STAGE_TARGET_SLOTS];
        let b_ok = eval_stop_grad_corner_and_phys_barrier(
            &[&fm_zero],
            &[&sg_corner],
            &bounds,
            10.0,
            &[0.5, 0.5, 0.1, 12.0],
            true,
        );
        let b_neg = eval_stop_grad_corner_and_phys_barrier(
            &[&fm_zero],
            &[&sg_corner],
            &bounds,
            10.0,
            &[-0.1, 0.5, 0.1, 12.0],
            false,
        );
        let b_over = eval_stop_grad_corner_and_phys_barrier(
            &[&fm_zero],
            &[&sg_corner],
            &bounds,
            10.0,
            &[0.5, 1.1, 0.1, 12.0],
            true,
        );
        let b_neg_glin = eval_stop_grad_corner_and_phys_barrier(
            &[&fm_zero],
            &[&sg_corner],
            &bounds,
            10.0,
            &[0.5, -0.1, 0.1, 12.0],
            true,
        );
        let b_nan = eval_stop_grad_corner_and_phys_barrier(
            &[&fm_zero],
            &[&sg_corner],
            &bounds,
            f64::NAN,
            &[0.5, 0.5, 0.1, 12.0],
            true,
        );
        assert!(b_ok.is_finite() && b_ok < 1.0);
        assert!(b_neg > 1.0e4 && b_neg_glin > 1.0e4 && b_over > 1.0e4);
        assert_eq!(b_nan, 1.0e6);
    }
}
