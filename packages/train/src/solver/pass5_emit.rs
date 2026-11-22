//! Pass-5 `Arena0..3` `zerocopy` packing, `.dnam-bin` validation & `REQ-5.6` tier stripping,
//! deterministic `SonaVm` CPU parity rendering, ITU-R BS.1770 `<= 0.05 LU` loudness trim
//! calibration, Phase-2/3 final safety certification, stereo 4-metric certification, and `STAT`
//! `Wide192Accum` digest emission (`ref[train.pipeline.host-parity-and-export]`).

pub mod certify;
pub mod pack;

pub use certify::{MAX_INTER_TIER_LU_DRIFT, compute_itu_bs1770_lufs};
pub use pack::encode_model_to_dnam_bin;

use dnam_vm::Tier;
use dnam_vm::bin::DnamBinView;

use crate::control::stability::run_phase2_rail_burst;
use crate::error::TrainError;
use crate::losses::{TierCertificationReport, check_adjacent_tier_monotone};
use crate::wide_gram::hash_gram_input_v3_f32;

use super::model::SonaModelF64;
use certify::{
    apply_full_quad_phase3_audit, calibrate_lufs_trims_from_renders,
    certify_all_tier_renders_stereo, render_all_tiers, zero_tier_delta_weights,
};

/// Configuration for Pass-5 `.dnam-bin` serialization, LUFS calibration, and final certification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass5EmitConfig {
    /// Emit `DNBN_FLAG_STEREO_OUT` (`true` canonical).
    pub stereo_out: bool,
    /// Emit `DNBN_FLAG_SINGLE_INPUT` (`false` true stereo input).
    pub single_input: bool,
    /// Declared tail time in milliseconds.
    pub tail_time_ms: f32,
    /// Switch crossfade samples.
    pub switch_xfade_samples: u16,
    /// Calibrate inter-tier LUFS within `<= 0.05 LU`.
    pub calibrate_lufs_trims: bool,
}

impl Default for Pass5EmitConfig {
    fn default() -> Self {
        Self {
            stereo_out: true,
            single_input: false,
            tail_time_ms: 80.0,
            switch_xfade_samples: 64,
            calibrate_lufs_trims: true,
        }
    }
}

/// Complete certified output bundle emitted by Pass 5 (`ref[train.pipeline.host-parity-and-export]`).
#[derive(Clone, Debug)]
pub struct Pass5EmitResult {
    /// Complete validated 4-tier `.dnam-bin` bytes (`Lite..Ultra` present).
    pub dnam_bin: Vec<u8>,
    /// Tier-stripped `.dnam-bin` slices `[Lite, Full, Pro, Ultra]`.
    pub stripped_tiers: [Vec<u8>; 4],
    /// Deterministic CPU `SonaVm` stereo renders `[[L, R]; 4]` for `[Lite, Full, Pro, Ultra]`.
    pub vm_renders: [(Vec<f64>, Vec<f64>); 4],
    /// ITU-R BS.1770 K-weighted integrated loudness per tier in LUFS.
    pub tier_lufs: [f64; 4],
    /// Worst-case stereo 4-metric certification reports across `[Lite, Full, Pro, Ultra]`.
    pub tier_reports: [TierCertificationReport; 4],
    /// Adjacent-tier monotone non-regression result across both Left and Right channels.
    pub adjacent_tier_monotone: bool,
    /// Calibrated `tier_gain_trim[0..4]`.
    pub tier_gain_trim: [f64; 4],
    /// Applied Phase-3 head-gain scale (`1.0` when already `<= +6.0 dB`).
    pub head_gain_scale: f64,
    /// 32-byte `STAT` capsule `Wide192Accum` digest (`hash_gram_input_v3_f32`).
    pub stat_gram_digest: [u8; 32],
}

/// Executes Pass-5 with an explicit Pass-3 `feature_dim`-wide `STAT` digest.
pub fn run_pass5_emit_and_certify_with_digest(
    model: &mut SonaModelF64,
    pairs: &super::model::StereoTrainingPairs<'_>,
    warmup: usize,
    config: Pass5EmitConfig,
    stat_digest_opt: Option<[u8; 32]>,
) -> Result<Pass5EmitResult, TrainError> {
    let (in_l, in_r, target_l, target_r) = (pairs.in_l, pairs.in_r, pairs.target_l, pairs.target_r);
    let n = in_l
        .len()
        .min(in_r.len())
        .min(target_l.len())
        .min(target_r.len());
    if n <= warmup {
        return Err(TrainError::InvalidDimensions {
            context: "run_pass5_emit_and_certify",
            expected: warmup + 1,
            actual: n,
        });
    }

    let head_gain_scale = apply_full_quad_phase3_audit(model)?;
    let sr_f32 = model.sample_rate_hz as f32;

    if config.calibrate_lufs_trims {
        let raw_bin = encode_model_to_dnam_bin(model, config)?;
        let _ = DnamBinView::parse(&raw_bin)?;
        let raw_bundle = render_all_tiers(&raw_bin, &in_l[..n], &in_r[..n], sr_f32)?;
        calibrate_lufs_trims_from_renders(model, &raw_bundle.renders, warmup);
    }

    let mut final_bin = encode_model_to_dnam_bin(model, config)?;
    let _ = DnamBinView::parse(&final_bin)?;
    let mut final_bundle = render_all_tiers(&final_bin, &in_l[..n], &in_r[..n], sr_f32)?;
    let mut cert = certify_all_tier_renders_stereo(
        &final_bundle.renders,
        &target_l[..n],
        &target_r[..n],
        model.sample_rate_hz,
        warmup,
    )?;

    for _pass in 0..4 {
        let mut changed = false;
        for t in 1..=3 {
            if !cert.pair_monotone[t] {
                zero_tier_delta_weights(model, t);
                changed = true;
            }
        }
        if !changed {
            break;
        }
        model.tier_gain_trim = [1.0; 4];
        if config.calibrate_lufs_trims {
            let mid_bin = encode_model_to_dnam_bin(model, config)?;
            let mid_bundle = render_all_tiers(&mid_bin, &in_l[..n], &in_r[..n], sr_f32)?;
            calibrate_lufs_trims_from_renders(model, &mid_bundle.renders, warmup);
        }
        final_bin = encode_model_to_dnam_bin(model, config)?;
        let _ = DnamBinView::parse(&final_bin)?;
        final_bundle = render_all_tiers(&final_bin, &in_l[..n], &in_r[..n], sr_f32)?;
        cert = certify_all_tier_renders_stereo(
            &final_bundle.renders,
            &target_l[..n],
            &target_r[..n],
            model.sample_rate_hz,
            warmup,
        )?;
    }

    let (stripped, vm_renders) = (final_bundle.stripped, final_bundle.renders);
    let rail_rep = run_phase2_rail_burst(&final_bin, Tier::Ultra, sr_f32, 1.0)?;
    if rail_rep.flagged {
        return Err(TrainError::AuditViolation {
            context: "Pass-5 Phase-2 [CLAMP-RAIL]",
            detail: "rail-lockup or non-finite output detected across emitted Ultra VM",
        });
    }
    let adjacent_tier_monotone = cert.pair_monotone[1..].iter().all(|&m| m)
        && check_adjacent_tier_monotone(&cert.tier_reports);

    let stat_gram_digest = if let Some(d) = stat_digest_opt {
        d
    } else {
        let gram_samples: Vec<f32> = (warmup..n)
            .flat_map(|t| [target_l[t] as f32, target_r[t] as f32])
            .collect();
        let gram_weights = vec![1.0_f64; n - warmup];
        hash_gram_input_v3_f32(2, &gram_samples, &gram_weights, 1)
    };

    Ok(Pass5EmitResult {
        dnam_bin: final_bin,
        stripped_tiers: stripped,
        vm_renders,
        tier_lufs: cert.tier_lufs,
        tier_reports: cert.tier_reports,
        adjacent_tier_monotone,
        tier_gain_trim: model.tier_gain_trim,
        head_gain_scale,
        stat_gram_digest,
    })
}

/// Executes Pass-5 `.dnam-bin` encoding, `SonaVm` parity rendering, ITU-R BS.1770 LUFS trim
/// calibration, Phase-2/3 final audits, true-stereo 4-metric certification, and `STAT` emission.
pub fn run_pass5_emit_and_certify(
    model: &mut SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    warmup: usize,
    config: Pass5EmitConfig,
) -> Result<Pass5EmitResult, TrainError> {
    let pairs = super::model::StereoTrainingPairs {
        in_l,
        in_r,
        target_l,
        target_r,
    };
    run_pass5_emit_and_certify_with_digest(model, &pairs, warmup, config, None)
}

#[cfg(test)]
mod tests {
    use dnam_vm::bin::{SONA_META_CONVOLVER_PREFIX_OVERRIDES, strip_dnam_bin_to_tier};
    use dnam_vm::math::fp16_to_f32;
    use sha2::{Digest, Sha256};
    use zerocopy::IntoBytes;

    use super::super::model::TierReadoutPrefixF64;
    use super::*;

    #[test]
    fn test_pass5_emit_4tier_dnam_bin_os2x_prefix_override_and_stereo_lufs() {
        let n = 1_024_usize;
        let sr = 48_000.0_f64;
        let in_l: Vec<f64> = (0..n)
            .map(|i| 0.3 * ((i as f64) * 0.08).sin() + 0.15 * ((i as f64) * 0.35).cos())
            .collect();
        let in_r: Vec<f64> = (0..n)
            .map(|i| -0.25 * ((i as f64) * 0.11).cos() + 0.18 * ((i as f64) * 0.27).sin())
            .collect();
        let tgt_l: Vec<f64> = in_l.iter().map(|&x| 0.85 * x).collect();
        let tgt_r: Vec<f64> = in_r.iter().map(|&x| 0.72 * x).collect();

        let mut model = SonaModelF64::new(sr);
        model.oversample_upper_tiers_2x = true;
        model.prefix_override_needed = [false, true, true, true];
        model.direct_s2_u_gain = [0.85, 0.72];
        model.tier_gain_trim = [0.7, 0.85, 1.0, 1.0];
        let pfx_full = TierReadoutPrefixF64 {
            direct_s2_u_gain: [0.85, 0.72],
            ..Default::default()
        };
        model.tier_overrides = [
            Some(Box::new(pfx_full.clone())),
            Some(Box::new(pfx_full.clone())),
            Some(Box::new(pfx_full.clone())),
            Some(Box::new(pfx_full)),
        ];

        let res = run_pass5_emit_and_certify(
            &mut model,
            &in_l,
            &in_r,
            &tgt_l,
            &tgt_r,
            64,
            Pass5EmitConfig::default(),
        )
        .unwrap();

        let view = DnamBinView::parse(&res.dnam_bin).unwrap();
        let mf = view.arena0.crossbar.sona_meta_flags.get();
        assert_ne!(mf & SONA_META_CONVOLVER_PREFIX_OVERRIDES, 0);
        assert_eq!(res.stripped_tiers[0].len(), 128 + 4_608);
        assert_eq!(res.stripped_tiers[3].len(), res.dnam_bin.len());
        for t in 0..3 {
            assert!(
                (res.tier_lufs[t] - res.tier_lufs[3]).abs() <= MAX_INTER_TIER_LU_DRIFT,
                "tier {t} lufs={} vs ultra={}",
                res.tier_lufs[t],
                res.tier_lufs[3]
            );
        }
        assert!(res.adjacent_tier_monotone);

        // Verify post-zeroing fallback re-calibrates LUFS when Right channel regresses on Pro.
        let mut reg_model = SonaModelF64::new(sr);
        reg_model.oversample_upper_tiers_2x = false;
        reg_model.direct_s2_u_gain = [0.85, 0.72];
        reg_model.fir_r[50] = 2.0; // hurts Pro Right channel compared to Full
        let res_reg = run_pass5_emit_and_certify(
            &mut reg_model,
            &in_l,
            &in_r,
            &tgt_l,
            &tgt_r,
            64,
            Pass5EmitConfig::default(),
        )
        .unwrap();
        for t in 0..3 {
            assert!(
                (res_reg.tier_lufs[t] - res_reg.tier_lufs[3]).abs() <= MAX_INTER_TIER_LU_DRIFT,
                "post-zero tier {t} lufs={} vs ultra={}",
                res_reg.tier_lufs[t],
                res_reg.tier_lufs[3]
            );
        }

        // Verify content_hash_prefix depends strictly on Arena0 (preserving REQ-5.6 truncation),
        // `carrier_basis_w` packs into Arena0, and upper-tier wire clamps enforce VM invariants.
        let mut wire_model = SonaModelF64::new(sr);
        wire_model.carrier_basis_w[1][2] = 1.75;
        wire_model.transport_quads[2].g_loop = 1.0;
        wire_model.transport_quads[2].lfo_depths = [-5.0, 600.0, 2.0, 0.0];
        wire_model.opt_to_lr[8] = [-6.0, 6.0];
        wire_model.fir_l[50] = 30.0;
        wire_model.kautz[14].theta = core::f64::consts::PI;
        wire_model.kautz[14].b_gamma = -2.0;
        wire_model.prefix_override_needed = [false, true, true, true];
        let bin = pack::encode_model_to_dnam_bin(&wire_model, Pass5EmitConfig::default()).unwrap();
        let view_full = DnamBinView::parse(&bin).unwrap();
        let lite_bytes = strip_dnam_bin_to_tier(&bin, Tier::Lite).unwrap();
        let view_lite = DnamBinView::parse(lite_bytes).unwrap();
        let mut h = Sha256::new();
        h.update(b"DNAM-BIN-CONTENT-V3\0");
        h.update(view_lite.arena0.as_bytes());
        let expected_hash: [u8; 32] = h.finalize().into();
        assert_eq!(
            view_full.runtime_meta.content_hash_prefix,
            expected_hash[..8]
        );
        assert_eq!(
            view_lite.runtime_meta.content_hash_prefix,
            expected_hash[..8]
        );
        assert!(
            (fp16_to_f32(view_lite.arena0.crossbar.carrier_basis_w_fp16[1][2].get()) - 1.75).abs()
                <= 1e-3
        );
        let a2 = view_full.arena2.unwrap();
        assert!(a2.transport_pro[0].g_loop.get() <= dnam_kernels::TANK_MAX_G_LIN_F32);
        assert_eq!(
            fp16_to_f32(a2.transport_pro[0].lfo_depths_fp16[0].get()),
            0.0
        );
        assert!(fp16_to_f32(a2.transport_pro[0].lfo_depths_fp16[1].get()) <= 512.0);
        assert_eq!(
            fp16_to_f32(a2.crossbar_pro.opt_to_lr_fp16[8][0].get()),
            -4.0
        );
        assert_eq!(a2.conv_fir_pro.fir_l[2].get(), 16.0);
        assert!(a2.conv_kautz_pro[0].theta[2].get() <= core::f32::consts::PI - 1.0e-4);
        assert_eq!(a2.conv_kautz_pro[0].b_gamma[2].get(), 0.0);
    }
}
