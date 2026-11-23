//! Full-subspace Pass-2 optimization vector layout ([`ThetaLayout`]) across all `8` canonical
//! `SONA v3.0` Matryoshka domains with conditioned coordinate transforms, exact `fp16`/`f32` wire
//! masks, and discrete polarity/phase preservation (`ref[train.pipeline.backbone-trajectory-opt]`).

mod domains;

use crate::linalg::lbfgs::ParamBoundSpec;

use super::super::model::SonaModelF64;
use super::eval::Pass2ParamGrads;
use super::eval::core_adj::{active_opt_quads_for_core_quads, active_tiles_for_quads};
use domains::{
    NonlinearSubspaceSpec, ThetaEntry, append_joint_readout_entries, append_nonlinear_entries,
    append_qat_readout_entries, append_shared_pre_conv_entries,
};

/// Canonical tier core-quad counts `[Lite, Full, Pro, Ultra]`, consumed by M21 per-tier
/// progressive/Matryoshka scheduling and unit tests.
pub(super) const TIER_QUADS: [usize; 4] = [4, 6, 8, 10];
/// Canonical tier Kautz modal pair counts `[Lite, Full, Pro, Ultra]`, consumed by M21 per-tier
/// progressive/Matryoshka scheduling and unit tests.
pub(super) const TIER_KAUTZ: [usize; 4] = [8, 12, 24, 32];
/// Canonical tier Early FIR tap counts `[Lite, Full, Pro, Ultra]`.
pub(super) const TIER_FIR_TAPS: [usize; 4] = [32, 48, 96, 128];
const _: () = assert!(
    TIER_QUADS[0] == 4
        && TIER_QUADS[3] == 10
        && TIER_KAUTZ[0] == 8
        && TIER_KAUTZ[3] == 32
        && TIER_FIR_TAPS[0] == 32
        && TIER_FIR_TAPS[3] == 128
);

/// Continuous optimization layout bridging [`SonaModelF64`] and Pass-2 L-BFGS / `AdamW` /
/// `L-SSBroyden` / STE QAT solvers.
#[derive(Clone, Debug)]
pub(super) struct ThetaLayout {
    pub q_opt: usize,
    pub k_opt: usize,
    pub is_fp16_wire: Vec<bool>,
    pub arena_of_param: Vec<u8>,
    pub zoh_hold_depth_idx: Option<usize>,
    pub kautz_active_mask: [bool; 32],
    entries: Vec<ThetaEntry>,
}

/// Returns `true` when any `Op-T` transport quad or crossbar coupling in active `q_count` is nonzero.
#[must_use]
pub(super) fn has_active_op_t(model: &SonaModelF64, q_count: usize) -> bool {
    let max_opt_quads = active_opt_quads_for_core_quads(q_count);
    model.transport_quads[..max_opt_quads].iter().any(|tq| {
        tq.g_loop.abs() > 1.0e-9
            || tq.input_gains.iter().any(|&v| v.abs() > 1.0e-9)
            || tq.wet_tap_l.iter().any(|&v| v.abs() > 1.0e-9)
            || tq.wet_tap_r.iter().any(|&v| v.abs() > 1.0e-9)
    }) || model.opt_to_lr[..(max_opt_quads * 4)]
        .iter()
        .any(|row| row[0].abs() > 1.0e-9 || row[1].abs() > 1.0e-9)
}

#[derive(Clone, Copy)]
enum ReadoutSubspaceMode {
    None,
    SharedPreConv,
    FullJoint,
    QatTierSafe,
}

impl ThetaLayout {
    fn finalize(
        q_opt: usize,
        k_opt: usize,
        zoh_hold_depth_idx: Option<usize>,
        kautz_active_mask: [bool; 32],
        entries: Vec<ThetaEntry>,
    ) -> Self {
        let is_fp16_wire = entries.iter().map(|e| e.is_fp16_wire).collect();
        let arena_of_param = entries.iter().map(|e| e.kind.arena_id()).collect();
        Self {
            q_opt,
            k_opt,
            is_fp16_wire,
            arena_of_param,
            zoh_hold_depth_idx,
            kautz_active_mask,
            entries,
        }
    }

    /// Builds the canonical full-subspace nonlinear [`ThetaLayout`] for `tier_idx in 0..4`
    /// (`612 / 1,314 / 2,028 / 2,830` base or `637 / 1,359 / 2,101 / 2,927` with extras).
    #[cfg(test)]
    #[must_use]
    pub(super) fn for_tier(tier_idx: usize, include_extras: bool, max_tau: f64) -> Self {
        let ti = tier_idx.min(3);
        let q_count = TIER_QUADS[ti];
        let k_count = TIER_KAUTZ[ti];
        let active_opt_quads = active_opt_quads_for_core_quads(q_count);
        let active_tiles = active_tiles_for_quads(q_count);
        let mut entries = Vec::with_capacity(if include_extras { 2950 } else { 2850 });
        let zoh_hold_depth_idx = append_nonlinear_entries(
            &mut entries,
            &NonlinearSubspaceSpec {
                q_count,
                active_tiles,
                include_stereo_phase: ti > 0,
                k_count,
                active_opt_quads,
                include_extras,
                sr_hz: 48_000.0,
                max_tau,
            },
        );
        Self::finalize(q_count, k_count, zoh_hold_depth_idx, [true; 32], entries)
    }

    /// Builds a model-specific [`ThetaLayout`] for Matryoshka tier `tier_idx in 0..4`, capping
    /// Kautz pairs to `TIER_KAUTZ[ti]` and optionally appending linear readout coordinates.
    #[must_use]
    pub(super) fn for_tier_from_model(
        model: &SonaModelF64,
        tier_idx: usize,
        include_extras: bool,
        include_readouts: bool,
        max_tau: f64,
    ) -> Self {
        let ro_mode = if include_readouts {
            ReadoutSubspaceMode::FullJoint
        } else {
            ReadoutSubspaceMode::None
        };
        Self::for_tier_from_model_inner(model, tier_idx, include_extras, ro_mode, max_tau)
    }

    /// Pass-2 variant unlocking tier-invariant `Lite`-shared pre-convolver coupling
    /// (`w_ro[0..16][0..6]`, `pre_conv_u_gain`) whenever `FIR` or `Kautz` modal stages are active,
    /// without `w_ro[16..40][0..6]` or `pre_conv_hp_gain`.
    #[must_use]
    pub(super) fn for_tier_from_model_pass2(
        model: &SonaModelF64,
        tier_idx: usize,
        include_extras: bool,
        include_direct_readouts: bool,
        max_tau: f64,
    ) -> Self {
        let include_pre_conv = model.active_fir_taps > 0 || model.active_kautz_pairs > 0;
        let ro_mode = if include_direct_readouts {
            ReadoutSubspaceMode::QatTierSafe
        } else if include_pre_conv {
            ReadoutSubspaceMode::SharedPreConv
        } else {
            ReadoutSubspaceMode::None
        };
        Self::for_tier_from_model_inner(model, tier_idx, include_extras, ro_mode, max_tau)
    }

    /// Phase-2D STE `QAT` variant unlocking full direct readouts together with strictly shared
    /// `Lite`-prefix `pre_conv` coupling (`w_ro[0..16][0..6]`, `pre_conv_u_gain`) without
    /// upper-tier `w_ro[16..40][0..6]` or `pre_conv_hp_gain`.
    #[must_use]
    pub(super) fn for_tier_from_model_qat(
        model: &SonaModelF64,
        tier_idx: usize,
        include_extras: bool,
        max_tau: f64,
    ) -> Self {
        Self::for_tier_from_model_inner(
            model,
            tier_idx,
            include_extras,
            ReadoutSubspaceMode::QatTierSafe,
            max_tau,
        )
    }

    fn for_tier_from_model_inner(
        model: &SonaModelF64,
        tier_idx: usize,
        include_extras: bool,
        ro_mode: ReadoutSubspaceMode,
        max_tau: f64,
    ) -> Self {
        let ti = tier_idx.min(3);
        let q_count = TIER_QUADS[ti];
        let k_count = model.active_kautz_pairs.min(TIER_KAUTZ[ti]);
        let active_tiles = active_tiles_for_quads(q_count);
        let max_opt_quads = active_opt_quads_for_core_quads(q_count);
        let active_opt_quads = if has_active_op_t(model, q_count) {
            max_opt_quads
        } else {
            0
        };
        let mut entries = Vec::with_capacity(3200);
        let zoh_hold_depth_idx = append_nonlinear_entries(
            &mut entries,
            &NonlinearSubspaceSpec {
                q_count,
                active_tiles,
                include_stereo_phase: ti > 0,
                k_count,
                active_opt_quads,
                include_extras,
                sr_hz: model.sample_rate_hz,
                max_tau,
            },
        );
        match ro_mode {
            ReadoutSubspaceMode::None => {}
            ReadoutSubspaceMode::SharedPreConv => {
                append_shared_pre_conv_entries(&mut entries, q_count);
            }
            ReadoutSubspaceMode::FullJoint => {
                append_joint_readout_entries(&mut entries, q_count, k_count);
            }
            ReadoutSubspaceMode::QatTierSafe => {
                append_qat_readout_entries(&mut entries, q_count, k_count);
            }
        }
        let mut kautz_active_mask = [false; 32];
        for (k, kp) in model.kautz.iter().enumerate().take(k_count) {
            kautz_active_mask[k] = kp.r_low > 0.0 && kp.b_gamma > 0.0;
        }
        Self::finalize(
            q_count,
            k_count,
            zoh_hold_depth_idx,
            kautz_active_mask,
            entries,
        )
    }

    /// Builds the joint continuous nonlinear + active linear readout layout used by Pass-2
    /// tests verifying joint readout gradients.
    #[cfg(test)]
    #[must_use]
    pub(super) fn for_pass2_joint(model: &SonaModelF64, active_quads: usize, max_tau: f64) -> Self {
        let q_count = active_quads.clamp(1, 10);
        let k_count = model.active_kautz_pairs.min(32);
        let active_tiles = active_tiles_for_quads(q_count);
        let max_opt_quads = active_opt_quads_for_core_quads(q_count);
        let op_t_active = has_active_op_t(model, q_count);
        let active_opt_quads = if op_t_active { max_opt_quads } else { 0 };
        let mut entries = Vec::with_capacity(3200);
        let zoh_hold_depth_idx = append_nonlinear_entries(
            &mut entries,
            &NonlinearSubspaceSpec {
                q_count,
                active_tiles,
                include_stereo_phase: q_count > 4,
                k_count,
                active_opt_quads,
                include_extras: op_t_active,
                sr_hz: model.sample_rate_hz,
                max_tau,
            },
        );
        append_joint_readout_entries(&mut entries, q_count, k_count);
        let mut kautz_active_mask = [false; 32];
        for (k, kp) in model.kautz.iter().enumerate().take(k_count) {
            kautz_active_mask[k] = kp.r_low > 0.0 && kp.b_gamma > 0.0;
        }
        Self::finalize(
            q_count,
            k_count,
            zoh_hold_depth_idx,
            kautz_active_mask,
            entries,
        )
    }

    /// Number of scalar parameters in this layout.
    #[inline]
    #[must_use]
    pub(super) fn dim(&self) -> usize {
        self.entries.len()
    }

    /// Returns a boolean mask marking coordinates directly coupled to linear VarPro readouts
    /// (`ro_pole_48k`, `kautz.*`, and joint readout coordinates when present).
    #[must_use]
    pub(super) fn readout_coupled_mask(&self) -> Vec<bool> {
        self.entries
            .iter()
            .map(|e| e.kind.is_readout_coupled())
            .collect()
    }

    /// Returns per-coordinate `ScaledCoordSpace` scales (`num_frames` for `zoh_hold_depth` to cancel
    /// secular `2 * pi * n` derivative growth, `1.0` elsewhere).
    #[must_use]
    pub(super) fn coord_scales(&self, num_frames: usize) -> Vec<f64> {
        let mut scales = vec![1.0_f64; self.entries.len()];
        if let Some(idx) = self.zoh_hold_depth_idx
            && idx < scales.len()
        {
            scales[idx] = num_frames.max(1) as f64;
        }
        scales
    }

    /// Packs the active Pass-2 subspace from `m` alongside matching coordinate bijections.
    pub(super) fn pack(&self, m: &SonaModelF64) -> (Vec<f64>, Vec<ParamBoundSpec>) {
        let mut x = Vec::with_capacity(self.entries.len());
        let mut b = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            x.push(entry.read(m));
            b.push(entry.bound);
        }
        (x, b)
    }

    /// Clamps and unpacks `cand` into `m`, snapping `lo == 0.0` `ComplementLog` values `< 1e-7` to
    /// `0.0`, preserving inactive Kautz `0.0` sentinels and discrete polarity/phase fields, and
    /// enforcing Table-4 stability bounds on exit.
    pub(super) fn unpack(&self, cand: &[f64], m: &mut SonaModelF64) {
        for (i, entry) in self.entries.iter().enumerate() {
            entry.write(m, cand[i], &self.kautz_active_mask);
        }
        m.clamp_all_stability_bounds(true);
    }

    /// Writes `grads` into `out` in the exact coordinate order produced by [`Self::pack`].
    pub(super) fn write_grad(&self, grads: &Pass2ParamGrads, out: &mut [f64]) {
        for (i, entry) in self.entries.iter().enumerate() {
            out[i] = entry.read_grad(grads, &self.kautz_active_mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::{CompositeLossConfig, CompositeLossEvaluator};
    use crate::runner::TrainRunner;
    use crate::solver::model::TransportQuadF64;
    use crate::solver::pass2_opt::eval::{
        Pass2AdjInput, assemble_pass2_gradients, eval_model_pass2_audio,
    };
    use domains::ParamKind;

    #[test]
    fn test_for_tier_exact_dimensions_masks_sentinels_and_discrete_invariance() {
        let expected_base = [612_usize, 1_314, 2_028, 2_830];
        let expected_ext = [637_usize, 1_359, 2_101, 2_927];
        // Base f32 counts per tier:
        // Block0: {1,2,3,4}*4*3 = 12/24/36/48; Blocks1..3: {3,4,5,6}*4*3 = 36/48/60/72 -> core f32: 48/72/96/120
        // Tiles: {3,13,23,35}*16 = 48/208/368/560; Sensory: 12; Crossbar: 4; OpNPre: 3;
        // Kautz poles + ro_pole: 17/25/49/65; OpT f32: {1,2,3,4}*13 = 13/26/39/52.
        let expected_base_f32 = [
            48 + 48 + 12 + 4 + 3 + 17 + 13,
            72 + 208 + 12 + 4 + 3 + 25 + 26,
            96 + 368 + 12 + 4 + 3 + 49 + 39,
            120 + 560 + 12 + 4 + 3 + 65 + 52,
        ];
        // Extras f32 count = only cautz.b_gamma (8/12/24/32); opt_to_lr/wet_tap/tank_dry_mix are fp16!
        let expected_ext_f32_extra = [8_usize, 12, 24, 32];

        for ti in 0..4 {
            let l_base = ThetaLayout::for_tier(ti, false, 128.0);
            let l_ext = ThetaLayout::for_tier(ti, true, 128.0);
            assert_eq!(l_base.dim(), expected_base[ti]);
            assert_eq!(l_ext.dim(), expected_ext[ti]);
            let f32_base = l_base.is_fp16_wire.iter().filter(|&&f| !f).count();
            let f32_ext = l_ext.is_fp16_wire.iter().filter(|&&f| !f).count();
            assert_eq!(
                f32_base, expected_base_f32[ti],
                "tier {ti} base f32 mismatch"
            );
            assert_eq!(
                f32_ext,
                expected_base_f32[ti] + expected_ext_f32_extra[ti],
                "tier {ti} ext f32 mismatch"
            );
            assert!(l_base.zoh_hold_depth_idx.is_some());
        }

        // Inactive Kautz r_low == 0.0 must stay 0.0, active r_low / r_dc never snap to 0.0,
        // and discrete polarity/phase fields remain invariant.
        let mut m = SonaModelF64::new(48_000.0);
        m.active_kautz_pairs = 4;
        m.kautz[0].r_low = 0.251;
        m.kautz[0].theta = 0.5;
        m.kautz[0].b_gamma = 0.8;
        m.kautz[1].r_low = 0.0;
        m.kautz[1].b_gamma = 0.0;
        m.transport_quads[0].r_dc = 0.95001;
        m.core_quads[0].rect_sign = [-1.0, 1.0, -1.0, 1.0];
        m.phasor_init_phase_q32 = [0x1111_2222, 0x3333_4444, 0x5555_6666, 0x7777_8888];
        m.transport_quads[0].lfo_phase_off_q16 = [100, 200, 300, 400];
        m.transport_quads[0].ramp_phase_off_q16 = [500, 600, 700, 800];
        m.transport_quads[0].allpass_delays = [3, 5, 7, 11];
        m.ro_pole_48k = 5.0e-8; // Below 1e-7 ComplementLog floor -> snaps to 0.0.

        let layout_dormant = ThetaLayout::for_pass2_joint(&m, 4, 128.0);
        assert!(
            !layout_dormant
                .entries
                .iter()
                .any(|e| matches!(e.kind, ParamKind::OpTDelay(_, _) | ParamKind::TankDryMix))
        );
        assert!(!layout_dormant.kautz_active_mask[1]);
        assert!(layout_dormant.kautz_active_mask[0]);

        m.transport_quads[0].input_gains[0] = 0.5;
        let layout = ThetaLayout::for_pass2_joint(&m, 4, 128.0);
        assert!(
            layout
                .entries
                .iter()
                .any(|e| matches!(e.kind, ParamKind::OpTDelay(0, 0)))
        );
        assert!(
            layout
                .entries
                .iter()
                .any(|e| matches!(e.kind, ParamKind::TankDryMix))
        );
        assert!(!layout.kautz_active_mask[1]);
        assert!(layout.kautz_active_mask[0]);
        let (x, _) = layout.pack(&m);
        let mut out = m.clone();
        layout.unpack(&x, &mut out);
        assert!((out.kautz[0].r_low - 0.251).abs() < 1.0e-12);
        assert_eq!(out.kautz[1].r_low, 0.0);
        assert!((out.transport_quads[0].r_dc - 0.95001).abs() < 1.0e-12);
        assert_eq!(out.ro_pole_48k, 0.0);
        assert_eq!(out.core_quads[0].rect_sign, m.core_quads[0].rect_sign);
        assert_eq!(out.phasor_init_phase_q32, m.phasor_init_phase_q32);
        assert_eq!(
            out.transport_quads[0].lfo_phase_off_q16,
            m.transport_quads[0].lfo_phase_off_q16
        );
        assert_eq!(
            out.transport_quads[0].ramp_phase_off_q16,
            m.transport_quads[0].ramp_phase_off_q16
        );
        assert_eq!(
            out.transport_quads[0].allpass_delays,
            m.transport_quads[0].allpass_delays
        );
    }

    #[test]
    fn test_full_subspace_theta_all_families_fd_parity_48k_and_96k() {
        let runner = TrainRunner::cpu();
        let n = 128_usize;
        let in_l: Vec<f64> = (0..n)
            .map(|t| {
                let tf = t as f64;
                0.35 * (tf * 0.025).sin() + 0.40 * (tf * 0.13).sin() + 0.25 * (tf * 0.75).cos()
            })
            .collect();
        let in_r: Vec<f64> = (0..n)
            .map(|t| {
                let tf = t as f64;
                0.30 * (tf * 0.03 + 0.2).cos()
                    + 0.35 * (tf * 0.09 + 0.3).cos()
                    + 0.25 * (tf * 0.68).sin()
            })
            .collect();
        let tgt_l: Vec<f64> = in_l.iter().map(|&v| 0.4 * v).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&v| 0.35 * v).collect();

        for &sr_hz in &[48_000.0_f64, 96_000.0_f64] {
            let mut model = SonaModelF64::new(sr_hz);
            model.sense_lru_rel = [0.85; 4];
            model.sense_vca_w = [1.8, -1.1, 0.7, 0.5];
            model.crossbar_op_gains = [0.6, 0.35];
            model.carrier_basis_w[0] = [0.4, -0.25, 0.2, 0.15];
            model.phasor_to_core[0] = [0.25, 0.2, -0.15, 0.12];
            model.opt_to_lr[0] = [0.55, -0.35];
            model.op_n_pre_gamma_bc = 0.45;
            model.op_n_pre_zoh_hold_depth = 0.28;
            model.op_n_pre_beta_q = 2.0;
            model.ro_pole_48k = 0.88;
            model.pre_conv_u_gain = [0.35, 0.28];
            model.pre_conv_hp_gain = [0.22, -0.18];
            model.direct_s2_u_gain = [0.25, 0.22];
            model.direct_s2_hp_gain = [-0.15, 0.2];
            model.core_quads[0].svf_g = [0.18, 0.28, 0.42, 0.65];
            model.core_quads[0].svf_k_nl = [0.55, 0.35, 0.25, 0.15];
            model.core_quads[0].lru_alpha_rel = [0.85; 4];
            model.core_quads[0].adaa_drive = [1.8, 1.3, 1.1, 0.9];
            model.core_quads[0].tone_dir = [0.55, 0.3, 0.2, -0.15];
            model.core_quads[1].svf_g = [0.22, 0.4, 0.7, 1.1];
            model.core_quads[1].lru_alpha_rel = [0.85; 4];
            model.core_quads[1].tone_bp = [0.3; 4];
            model.tiles[0].w_odd_4x4[0][0] = 0.42;
            model.w_ro[0] = [
                0.35, -0.25, 0.2, 0.15, 0.12, -0.1, 0.45, -0.3, 0.22, 0.15, 0.18, -0.12,
            ];
            model.w_ro[4] = [
                0.2, 0.22, -0.1, 0.12, 0.08, 0.06, 0.28, -0.2, 0.15, 0.11, 0.09, 0.1,
            ];
            model.active_fir_taps = 4;
            model.fir_l[..4].copy_from_slice(&[0.45, 0.22, -0.1, 0.06]);
            model.fir_r[..4].copy_from_slice(&[0.40, -0.18, 0.12, -0.05]);
            model.active_kautz_pairs = 1;
            model.kautz[0].r_low = 0.84;
            model.kautz[0].theta = 0.65;
            model.kautz[0].b_gamma = 0.9;
            model.kautz[0].c_re_l = 0.35;
            model.kautz[0].c_im_l = -0.2;
            model.kautz[0].c_re_r = 0.3;
            model.kautz[0].c_im_r = 0.25;
            model.tank_dry_mix = 0.75;
            model.transport_quads[0] = TransportQuadF64 {
                delay_samples: [7.25, 9.6, 12.4, 15.1],
                g_loop: 0.65,
                damp_pole: 0.3,
                scat_psi: 0.48,
                allpass_g: [0.3, -0.25, 0.35, -0.2],
                allpass_delays: [3, 5, 7, 11],
                input_gains: [0.5, 0.4, -0.3, 0.25],
                wet_tap_l: [0.6, 0.15, -0.4, 0.1],
                wet_tap_r: [0.1, -0.55, 0.2, 0.35],
                ..Default::default()
            };
            model.clamp_all_stability_bounds(true);

            let loss_eval = CompositeLossEvaluator::new(
                &tgt_l,
                &tgt_r,
                CompositeLossConfig {
                    sample_rate_hz: sr_hz,
                    warmup_frames: 16,
                    ..CompositeLossConfig::default()
                },
            )
            .unwrap();
            let layout = ThetaLayout::for_pass2_joint(&model, 2, 128.0);
            let buf = eval_model_pass2_audio(&runner, &model, &in_l, &in_r, 2).unwrap();
            let mut gl = vec![0.0; n];
            let mut gr = vec![0.0; n];
            let _ = loss_eval
                .eval_and_adjoint(&buf.pred_l, &buf.pred_r, Some(&mut gl), Some(&mut gr))
                .unwrap();
            let grads = assemble_pass2_gradients(
                &runner,
                &Pass2AdjInput {
                    model: &model,
                    in_l: &in_l,
                    in_r: &in_r,
                    buf: &buf,
                    gl: &gl,
                    gr: &gr,
                    q_opt: layout.q_opt,
                    k_opt: layout.k_opt,
                },
            )
            .unwrap();
            assert_eq!(grads.core_quads[1].svf_k_nl, [0.0; 4]);

            let (theta0, _) = layout.pack(&model);
            let mut grad_vec = vec![0.0_f64; theta0.len()];
            layout.write_grad(&grads, &mut grad_vec);
            let eval_at = |idx: usize, delta: f64| -> f64 {
                let mut t = theta0.clone();
                t[idx] += delta;
                let mut cand = model.clone();
                layout.unpack(&t, &mut cand);
                let b = eval_model_pass2_audio(&runner, &cand, &in_l, &in_r, 2).unwrap();
                loss_eval
                    .eval_and_adjoint(&b.pred_l, &b.pred_r, None, None)
                    .unwrap()
                    .total
            };

            let find_idx = |pred: &dyn Fn(&ParamKind) -> bool| -> usize {
                layout
                    .entries
                    .iter()
                    .position(|e| pred(&e.kind))
                    .expect("missing param in layout")
            };

            // Verify delta=0.0 unpack round-trip invariance before FD perturbations.
            let mut cand0 = model.clone();
            layout.unpack(&theta0, &mut cand0);
            let buf0 = eval_model_pass2_audio(&runner, &cand0, &in_l, &in_r, 2).unwrap();
            for t in 0..n {
                assert!(
                    (buf0.pred_l[t] - buf.pred_l[t]).abs() <= 1.0e-12
                        && (buf0.pred_r[t] - buf.pred_r[t]).abs() <= 1.0e-12,
                    "sr={sr_hz} delta=0 unpack round-trip mismatch at frame {t}"
                );
            }

            // Pure-f64 sub-check (w_ro=0 eliminates f32 CoreQuad rounding inside Op-T feedback).
            {
                let mut f64_model = model.clone();
                f64_model.w_ro = [[0.0; 12]; 40];
                let f64_buf = eval_model_pass2_audio(&runner, &f64_model, &in_l, &in_r, 2).unwrap();
                let (mut f64_gl, mut f64_gr) = (vec![0.0; n], vec![0.0; n]);
                let _ = loss_eval
                    .eval_and_adjoint(
                        &f64_buf.pred_l,
                        &f64_buf.pred_r,
                        Some(&mut f64_gl),
                        Some(&mut f64_gr),
                    )
                    .unwrap();
                let f64_grads = assemble_pass2_gradients(
                    &runner,
                    &Pass2AdjInput {
                        model: &f64_model,
                        in_l: &in_l,
                        in_r: &in_r,
                        buf: &f64_buf,
                        gl: &f64_gl,
                        gr: &f64_gr,
                        q_opt: layout.q_opt,
                        k_opt: layout.k_opt,
                    },
                )
                .unwrap();
                let (f64_theta0, _) = layout.pack(&f64_model);
                let mut f64_gvec = vec![0.0_f64; f64_theta0.len()];
                layout.write_grad(&f64_grads, &mut f64_gvec);
                let eval_f64 = |idx: usize, delta: f64| -> f64 {
                    let mut t = f64_theta0.clone();
                    t[idx] += delta;
                    let mut cand = f64_model.clone();
                    layout.unpack(&t, &mut cand);
                    let b = eval_model_pass2_audio(&runner, &cand, &in_l, &in_r, 2).unwrap();
                    loss_eval
                        .eval_and_adjoint(&b.pred_l, &b.pred_r, None, None)
                        .unwrap()
                        .total
                };
                let eps64 = 1.0e-6_f64;
                let f64_samples = [
                    (
                        "f64 RoPole48k",
                        find_idx(&|k| matches!(k, ParamKind::RoPole48k)),
                    ),
                    (
                        "f64 OpNPre gamma_bc",
                        find_idx(&|k| matches!(k, ParamKind::OpNPreGammaBc)),
                    ),
                    (
                        "f64 Kautz b_gamma[0]",
                        find_idx(&|k| matches!(k, ParamKind::KautzBGamma(0))),
                    ),
                    (
                        "f64 OpT scat_psi[0]",
                        find_idx(&|k| matches!(k, ParamKind::OpTScatPsi(0))),
                    ),
                    (
                        "f64 OptToLr[0][0]",
                        find_idx(&|k| matches!(k, ParamKind::OptToLr(0, 0))),
                    ),
                    (
                        "f64 TankDryMix",
                        find_idx(&|k| matches!(k, ParamKind::TankDryMix)),
                    ),
                    (
                        "f64 Sense vca_w[0]",
                        find_idx(&|k| matches!(k, ParamKind::SenseVcaW(0))),
                    ),
                ];
                for (name, idx) in f64_samples {
                    let fd = (eval_f64(idx, eps64) - eval_f64(idx, -eps64)) / (2.0 * eps64);
                    let ana = f64_gvec[idx];
                    let scale = ana.abs().max(fd.abs()).max(1.0e-3);
                    assert!(
                        (ana - fd).abs() <= 1.0e-4 * scale,
                        "sr={sr_hz} {name} (idx={idx}) pure-f64 FD mismatch: ana={ana} fd={fd}"
                    );
                }
            }

            let samples = [
                (
                    "Core0 svf_g[0]",
                    find_idx(&|k| matches!(k, ParamKind::CoreSvfG(0, 0))),
                    5.0e-4,
                ),
                (
                    "Core0 svf_k_nl[0]",
                    find_idx(&|k| matches!(k, ParamKind::CoreSvfKNl(0, 0))),
                    1.0e-3,
                ),
                (
                    "Core0 adaa_drive[0]",
                    find_idx(&|k| matches!(k, ParamKind::CoreAdaaDrive(0, 0))),
                    2.0e-3,
                ),
                (
                    "Core1 svf_g[0]",
                    find_idx(&|k| matches!(k, ParamKind::CoreSvfG(1, 0))),
                    1.0e-3,
                ),
                (
                    "Tile0 w_odd[0][0]",
                    find_idx(&|k| matches!(k, ParamKind::TileOdd(0, 0, 0))),
                    1.0e-3,
                ),
                (
                    "Sense vca_w[0]",
                    find_idx(&|k| matches!(k, ParamKind::SenseVcaW(0))),
                    4.0e-3,
                ),
                (
                    "PhasorToCore[0][0]",
                    find_idx(&|k| matches!(k, ParamKind::PhasorToCore(0, 0))),
                    1.0e-3,
                ),
                (
                    "OptToLr[0][0]",
                    find_idx(&|k| matches!(k, ParamKind::OptToLr(0, 0))),
                    2.0e-3,
                ),
                (
                    "OpNPre gamma_bc",
                    find_idx(&|k| matches!(k, ParamKind::OpNPreGammaBc)),
                    2.0e-3,
                ),
                (
                    "RoPole48k",
                    find_idx(&|k| matches!(k, ParamKind::RoPole48k)),
                    1.0e-3,
                ),
                (
                    "Kautz b_gamma[0]",
                    find_idx(&|k| matches!(k, ParamKind::KautzBGamma(0))),
                    1.0e-4,
                ),
                (
                    "OpT scat_psi[0]",
                    find_idx(&|k| matches!(k, ParamKind::OpTScatPsi(0))),
                    1.0e-4,
                ),
                (
                    "TankDryMix",
                    find_idx(&|k| matches!(k, ParamKind::TankDryMix)),
                    1.0e-4,
                ),
                (
                    "WReadout[0][6]",
                    find_idx(&|k| matches!(k, ParamKind::WReadout(0, 6))),
                    1.0e-4,
                ),
            ];
            for (name, idx, eps) in samples {
                let fd = (eval_at(idx, eps) - eval_at(idx, -eps)) / (2.0 * eps);
                let ana = grad_vec[idx];
                let scale = ana.abs().max(fd.abs()).max(0.1);
                assert!(
                    (ana - fd).abs() <= 3.0e-2 * scale,
                    "sr={sr_hz} {name} (idx={idx}) FD mismatch: ana={ana} fd={fd}"
                );
            }
        }
    }

    #[test]
    fn test_theta_arena_partition_masks_scales_and_safe_lower_tier_grad_write() {
        let base_counts = [
            [612_usize, 0, 0, 0, 0],
            [612, 702, 0, 0, 0],
            [612, 702, 714, 0, 0],
            [612, 702, 714, 802, 0],
        ];
        let ext_counts = [
            [637_usize, 0, 0, 0, 0],
            [637, 722, 0, 0, 0],
            [637, 722, 742, 0, 0],
            [637, 722, 742, 826, 0],
        ];
        for ti in 0..4 {
            for (include_ext, expected) in [(false, base_counts[ti]), (true, ext_counts[ti])] {
                let layout = ThetaLayout::for_tier(ti, include_ext, 128.0);
                let mut counts = [0_usize; 5];
                for &arena in &layout.arena_of_param {
                    counts[usize::from(arena)] += 1;
                }
                assert_eq!(
                    counts, expected,
                    "tier={ti} include_ext={include_ext} arena counts mismatch"
                );
            }
        }

        let mut model = SonaModelF64::new(48_000.0);
        model.active_kautz_pairs = 16;
        for kp in &mut model.kautz[..16] {
            kp.r_low = 0.85;
            kp.b_gamma = 1.0;
        }
        let m_lite = ThetaLayout::for_tier_from_model(&model, 0, true, false, 128.0);
        assert_eq!(m_lite.q_opt, 4);
        assert_eq!(m_lite.k_opt, 8);
        assert_eq!(m_lite.arena_of_param.iter().filter(|&&a| a == 4).count(), 0);

        let p2_lite = ThetaLayout::for_tier_from_model_pass2(&model, 0, true, false, 128.0);
        assert_eq!(
            p2_lite.arena_of_param.iter().filter(|&&a| a == 4).count(),
            16 * 6 + 2
        );
        assert!(!p2_lite.entries.iter().any(|e| matches!(
            e.kind,
            ParamKind::DirectS2U(_)
                | ParamKind::DirectS2Hp(_)
                | ParamKind::PreConvHp(_)
                | ParamKind::WReadout(16..=39, 0..=5)
                | ParamKind::KautzCReL(_)
                | ParamKind::KautzCImL(_)
                | ParamKind::KautzCReR(_)
                | ParamKind::KautzCImR(_)
        )));

        let p2_ultra = ThetaLayout::for_tier_from_model_pass2(&model, 3, true, false, 128.0);
        assert_eq!(
            p2_ultra.arena_of_param.iter().filter(|&&a| a == 4).count(),
            16 * 6 + 2
        );
        assert!(!p2_ultra.entries.iter().any(|e| matches!(
            e.kind,
            ParamKind::PreConvHp(_) | ParamKind::WReadout(16..=39, 0..=5)
        )));

        let qat_ultra = ThetaLayout::for_tier_from_model_qat(&model, 3, true, 128.0);
        assert!(!qat_ultra.entries.iter().any(|e| matches!(
            e.kind,
            ParamKind::PreConvHp(_) | ParamKind::WReadout(16..=39, 0..=5)
        )));
        assert!(qat_ultra.entries.iter().any(|e| matches!(
            e.kind,
            ParamKind::DirectS2U(_)
                | ParamKind::DirectS2Hp(_)
                | ParamKind::PreConvU(_)
                | ParamKind::WReadout(32, 6)
                | ParamKind::KautzCReL(_)
        )));

        let m_ultra_ro = ThetaLayout::for_tier_from_model(&model, 3, true, true, 128.0);
        assert_eq!(m_ultra_ro.q_opt, 10);
        assert_eq!(m_ultra_ro.k_opt, 16);
        // Readouts: 40*12 (w_ro) + 8 (scalars) + 16*4 (kautz c_*) = 552
        assert_eq!(
            m_ultra_ro
                .arena_of_param
                .iter()
                .filter(|&&a| a == 4)
                .count(),
            40 * 12 + 8 + 16 * 4
        );
        // Readout-coupled = 1 (ro_pole_48k) + 16*3 (kautz r_low,theta,b_gamma) + 552 readouts = 601
        assert_eq!(
            m_ultra_ro
                .readout_coupled_mask()
                .iter()
                .filter(|&&v| v)
                .count(),
            1 + 16 * 3 + 552
        );
        let scales = m_ultra_ro.coord_scales(4096);
        let zoh_idx = m_ultra_ro
            .zoh_hold_depth_idx
            .expect("zoh_hold_depth present");
        assert_eq!(scales[zoh_idx], 4096.0);
        assert_eq!(scales.iter().filter(|&&s| s > 1.0).count(), 1);
        assert_eq!(TIER_FIR_TAPS, [32, 48, 96, 128]);

        // Writing a smaller Lite Pass2ParamGrads into an Ultra ThetaLayout must safely zero-fill missing slots.
        let lite_grads = crate::solver::pass2_opt::eval::core_adj::Pass2FullGrads::with_dims(4, 8);
        let mut out_grad = vec![99.0_f64; m_ultra_ro.dim()];
        m_ultra_ro.write_grad(&lite_grads, &mut out_grad);
        assert!(out_grad.iter().all(|&v| v == 0.0));
    }
}
