//! Pass-5 `SonaVm` stereo rendering, ITU-R BS.1770 LUFS trim calibration, full-quad Phase-3
//! dynamic/head-gain audit, independent Left & Right 4-metric certification, and monotone
//! non-regression enforcement (`ref[train.pipeline.host-parity-and-export]`).

use core::f64::consts::PI;

use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use dnam_vm::bin::strip_dnam_bin_to_tier;
use dnam_vm::math::DefaultMathBackend;
use dnam_vm::{SonaVm, Tier};

use crate::control::clamp::{
    Phase3AuditInput, audit_phase3_dynamic_and_head_gain, evaluate_readout_spectral_magnitudes,
};
use crate::control::stability::{MAX_RO_POLE_48K_F32, MAX_RO_POLE_48K_F64};
use crate::error::TrainError;
use crate::losses::{
    DrySilentGapMask, TierCertificationReport, certify_tier_audio_with_gaps, is_tier_pair_monotone,
};

use super::super::model::{InterBlockTileF64, SonaModelF64, TierReadoutPrefixF64};

/// Maximum inter-tier integrated loudness drift (`0.05 LU`, `REQ-5.4`).
pub const MAX_INTER_TIER_LU_DRIFT: f64 = 0.05;

pub(super) type StereoRender = (Vec<f64>, Vec<f64>);

pub(super) struct TierRenderBundle {
    pub stripped: [Vec<u8>; 4],
    pub renders: [StereoRender; 4],
}

#[derive(Clone)]
pub(super) struct StereoCertBundle {
    pub tier_lufs: [f64; 4],
    pub tier_reports: [TierCertificationReport; 4],
    pub rep_l: [TierCertificationReport; 4],
    pub rep_r: [TierCertificationReport; 4],
    pub pair_monotone: [bool; 4],
}

/// Evaluates integrated ITU-R BS.1770 K-weighted stereo loudness (`LUFS`) over `[warmup..]`.
#[must_use]
pub fn compute_itu_bs1770_lufs(sig_l: &[f64], sig_r: &[f64], sr_hz: f64, warmup: usize) -> f64 {
    let n = sig_l.len().min(sig_r.len());
    if n <= warmup {
        return -120.0;
    }
    let k1 = (PI * 1_681.974_45 / sr_hz.max(8_000.0)).tan();
    let vh = 10.0_f64.powf(4.0 / 20.0);
    let vb = vh.sqrt();
    let d1 = 1.0 + k1 / 0.707_175_2 + k1 * k1;
    let b1 = [
        (vh + vb * k1 / 0.707_175_2 + k1 * k1) / d1,
        2.0 * (k1 * k1 - vh) / d1,
        (vh - vb * k1 / 0.707_175_2 + k1 * k1) / d1,
    ];
    let a1 = [
        2.0 * (k1 * k1 - 1.0) / d1,
        (1.0 - k1 / 0.707_175_2 + k1 * k1) / d1,
    ];

    let k2 = (PI * 38.135_47 / sr_hz.max(8_000.0)).tan();
    let d2 = 1.0 + k2 / 0.500_327_0 + k2 * k2;
    let b2 = [1.0 / d2, -2.0 / d2, 1.0 / d2];
    let a2 = [
        2.0 * (k2 * k2 - 1.0) / d2,
        (1.0 - k2 / 0.500_327_0 + k2 * k2) / d2,
    ];

    let k_filter = |sig: &[f64]| -> f64 {
        let (mut z1_1, mut z1_2) = (0.0_f64, 0.0_f64);
        let (mut z2_1, mut z2_2) = (0.0_f64, 0.0_f64);
        let mut sum_sq = 0.0_f64;
        for (idx, &x) in sig.iter().enumerate() {
            let y1 = b1[0] * x + z1_1;
            z1_1 = b1[1] * x - a1[0] * y1 + z1_2;
            z1_2 = b1[2] * x - a1[1] * y1;
            let y2 = b2[0] * y1 + z2_1;
            z2_1 = b2[1] * y1 - a2[0] * y2 + z2_2;
            z2_2 = b2[2] * y1 - a2[1] * y2;
            if idx >= warmup {
                sum_sq += y2 * y2;
            }
        }
        sum_sq / ((n - warmup) as f64)
    };

    let mean_pow = (k_filter(&sig_l[..n]) + k_filter(&sig_r[..n])).max(1.0e-18);
    (-0.691 + 10.0 * mean_pow.log10()).max(-120.0)
}

/// Renders one stripped tier (`Lite..Ultra`) through `SonaVm` and returns `(stripped_bin, (out_l, out_r))`.
pub(super) fn render_single_tier(
    dnam_bin: &[u8],
    tier: Tier,
    in_l32: &[f32],
    in_r32: &[f32],
    sr_hz: f32,
) -> Result<(Vec<u8>, StereoRender), TrainError> {
    let n = in_l32.len().min(in_r32.len());
    let slice = strip_dnam_bin_to_tier(dnam_bin, tier)?;
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(slice, tier, sr_hz)?;
    let (mut ol, mut or) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    vm.process_block(&in_l32[..n], &in_r32[..n], &mut ol, &mut or)?;
    Ok((
        slice.to_vec(),
        (
            ol.into_iter().map(f64::from).collect(),
            or.into_iter().map(f64::from).collect(),
        ),
    ))
}

pub(super) fn render_all_tiers(
    dnam_bin: &[u8],
    in_l: &[f64],
    in_r: &[f64],
    sr_hz: f32,
) -> Result<TierRenderBundle, TrainError> {
    let n = in_l.len().min(in_r.len());
    let in_l32: Vec<f32> = in_l[..n].iter().map(|&v| v as f32).collect();
    let in_r32: Vec<f32> = in_r[..n].iter().map(|&v| v as f32).collect();
    let tiers = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];
    let mut stripped = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut renders: [(Vec<f64>, Vec<f64>); 4] = Default::default();

    for (i, &tier) in tiers.iter().enumerate() {
        let (slice, render) = render_single_tier(dnam_bin, tier, &in_l32, &in_r32, sr_hz)?;
        stripped[i] = slice;
        renders[i] = render;
    }
    Ok(TierRenderBundle { stripped, renders })
}

#[inline]
fn snap_scalar_fp16(val: &mut f64, lo: f32, hi: f32) {
    let clamped = if val.is_finite() {
        (*val as f32).clamp(lo, hi)
    } else {
        0.0
    };
    *val = f64::from(fp16_to_f32(f32_to_fp16(clamped)));
}

fn snap_slice_fp16(vals: &mut [f64], lo: f32, hi: f32) {
    for v in vals {
        snap_scalar_fp16(v, lo, hi);
    }
}

fn sync_fp16_fallback_band(
    model: &mut SonaModelF64,
    cur_opt: Option<&mut TierReadoutPrefixF64>,
    prev_tier: usize,
    fir_band: core::ops::Range<usize>,
    bg_band: core::ops::Range<usize>,
    c_band: core::ops::Range<usize>,
) {
    if let Some(cur) = cur_opt {
        snap_slice_fp16(&mut cur.fir_l[fir_band.clone()], -16.0, 16.0);
        snap_slice_fp16(&mut cur.fir_r[fir_band.clone()], -16.0, 16.0);
        snap_slice_fp16(&mut cur.kautz_b_gamma[bg_band.clone()], 0.0, 8.0);
        snap_slice_fp16(&mut cur.kautz_c_re_l[c_band.clone()], -8.0, 8.0);
        snap_slice_fp16(&mut cur.kautz_c_im_l[c_band.clone()], -8.0, 8.0);
        snap_slice_fp16(&mut cur.kautz_c_re_r[c_band.clone()], -8.0, 8.0);
        snap_slice_fp16(&mut cur.kautz_c_im_r[c_band.clone()], -8.0, 8.0);
        if let Some(prev_ov) = model.tier_overrides[prev_tier].as_deref_mut() {
            prev_ov.fir_l[fir_band.clone()].copy_from_slice(&cur.fir_l[fir_band.clone()]);
            prev_ov.fir_r[fir_band.clone()].copy_from_slice(&cur.fir_r[fir_band.clone()]);
            prev_ov.kautz_b_gamma[bg_band.clone()]
                .copy_from_slice(&cur.kautz_b_gamma[bg_band.clone()]);
            prev_ov.kautz_c_re_l[c_band.clone()].copy_from_slice(&cur.kautz_c_re_l[c_band.clone()]);
            prev_ov.kautz_c_im_l[c_band.clone()].copy_from_slice(&cur.kautz_c_im_l[c_band.clone()]);
            prev_ov.kautz_c_re_r[c_band.clone()].copy_from_slice(&cur.kautz_c_re_r[c_band.clone()]);
            prev_ov.kautz_c_im_r[c_band.clone()].copy_from_slice(&cur.kautz_c_im_r[c_band.clone()]);
        }
        model.fir_l[fir_band.clone()].copy_from_slice(&cur.fir_l[fir_band.clone()]);
        model.fir_r[fir_band.clone()].copy_from_slice(&cur.fir_r[fir_band]);
        for k in bg_band {
            model.kautz[k].b_gamma = cur.kautz_b_gamma[k];
        }
        for k in c_band {
            model.kautz[k].c_re_l = cur.kautz_c_re_l[k];
            model.kautz[k].c_im_l = cur.kautz_c_im_l[k];
            model.kautz[k].c_re_r = cur.kautz_c_re_r[k];
            model.kautz[k].c_im_r = cur.kautz_c_im_r[k];
        }
        return;
    }

    snap_slice_fp16(&mut model.fir_l[fir_band.clone()], -16.0, 16.0);
    snap_slice_fp16(&mut model.fir_r[fir_band.clone()], -16.0, 16.0);
    for k in bg_band.clone() {
        snap_scalar_fp16(&mut model.kautz[k].b_gamma, 0.0, 8.0);
    }
    for k in c_band.clone() {
        snap_scalar_fp16(&mut model.kautz[k].c_re_l, -8.0, 8.0);
        snap_scalar_fp16(&mut model.kautz[k].c_im_l, -8.0, 8.0);
        snap_scalar_fp16(&mut model.kautz[k].c_re_r, -8.0, 8.0);
        snap_scalar_fp16(&mut model.kautz[k].c_im_r, -8.0, 8.0);
    }
    if let Some(prev_ov) = model.tier_overrides[prev_tier].as_deref_mut() {
        prev_ov.fir_l[fir_band.clone()].copy_from_slice(&model.fir_l[fir_band.clone()]);
        prev_ov.fir_r[fir_band.clone()].copy_from_slice(&model.fir_r[fir_band]);
        for k in bg_band {
            prev_ov.kautz_b_gamma[k] = model.kautz[k].b_gamma;
        }
        for k in c_band {
            prev_ov.kautz_c_re_l[k] = model.kautz[k].c_re_l;
            prev_ov.kautz_c_im_l[k] = model.kautz[k].c_im_l;
            prev_ov.kautz_c_re_r[k] = model.kautz[k].c_re_r;
            prev_ov.kautz_c_im_r[k] = model.kautz[k].c_im_r;
        }
    }
}

pub(super) fn zero_tier_delta_weights(model: &mut SonaModelF64, tier_idx: usize) {
    if tier_idx == 1 && model.oversample_upper_tiers_2x {
        model.oversample_upper_tiers_2x = false;
    }
    model.prefix_override_needed[tier_idx] = model.prefix_override_needed[tier_idx - 1];
    let (lane_range, fir_range, kautz_range, tile_range) = match tier_idx {
        1 => (16..24, 32..48, 8..12, 3..13),
        2 => (24..32, 48..96, 12..24, 13..23),
        3 => (32..40, 96..128, 24..32, 23..35),
        _ => return,
    };
    for lane in &mut model.w_ro[lane_range.clone()] {
        *lane = [0.0; 12];
    }
    for k in fir_range.clone() {
        model.fir_l[k] = 0.0;
        model.fir_r[k] = 0.0;
    }
    for kz in &mut model.kautz[kautz_range.clone()] {
        kz.c_re_l = 0.0;
        kz.c_im_l = 0.0;
        kz.c_re_r = 0.0;
        kz.c_im_r = 0.0;
    }
    for tile in &mut model.tiles[tile_range] {
        *tile = InterBlockTileF64::default();
    }
    model.transport_quads[tier_idx].input_gains = [0.0; 4];
    model.transport_quads[tier_idx].wet_tap_l = [0.0; 4];
    model.transport_quads[tier_idx].wet_tap_r = [0.0; 4];
    model.tier_gain_trim[tier_idx] = model.tier_gain_trim[tier_idx - 1];

    let mut cur_opt = model.tier_overrides[tier_idx - 1].as_deref().cloned();
    if let Some(ref mut cur) = cur_opt {
        for lane in &mut cur.w_ro[lane_range] {
            *lane = [0.0; 12];
        }
        for k in fir_range {
            cur.fir_l[k] = 0.0;
            cur.fir_r[k] = 0.0;
        }
        for k in kautz_range {
            cur.kautz_c_re_l[k] = 0.0;
            cur.kautz_c_im_l[k] = 0.0;
            cur.kautz_c_re_r[k] = 0.0;
            cur.kautz_c_im_r[k] = 0.0;
        }
    } else {
        model.prefix_override_needed[tier_idx] = false;
    }

    let emit_prefix_overrides = model.prefix_override_needed[1..].iter().any(|&v| v);
    if emit_prefix_overrides {
        match tier_idx {
            1 => sync_fp16_fallback_band(model, cur_opt.as_mut(), 0, 16..32, 0..8, 4..8),
            2 => {
                sync_fp16_fallback_band(model, cur_opt.as_mut(), 1, 16..32, 0..8, 4..8);
                sync_fp16_fallback_band(model, cur_opt.as_mut(), 1, 32..48, 8..12, 8..12);
            }
            3 => {
                sync_fp16_fallback_band(model, cur_opt.as_mut(), 2, 32..48, 0..12, 8..12);
                sync_fp16_fallback_band(model, cur_opt.as_mut(), 2, 48..96, 12..24, 12..24);
            }
            _ => {}
        }
    }
    model.tier_overrides[tier_idx] = cur_opt.map(Box::new);
}

pub(super) fn calibrate_lufs_trims_from_renders(
    model: &mut SonaModelF64,
    renders: &[(Vec<f64>, Vec<f64>); 4],
    warmup: usize,
) {
    let lufs_ultra =
        compute_itu_bs1770_lufs(&renders[3].0, &renders[3].1, model.sample_rate_hz, warmup);
    if lufs_ultra <= -100.0 {
        return;
    }
    for (i, (rl, rr)) in renders.iter().enumerate().take(3) {
        let lufs_t = compute_itu_bs1770_lufs(rl, rr, model.sample_rate_hz, warmup);
        if lufs_t > -100.0 {
            let delta_lu = (lufs_ultra - lufs_t).clamp(-12.0, 12.0);
            model.tier_gain_trim[i] =
                (model.tier_gain_trim[i] * 10.0_f64.powf(delta_lu / 20.0)).clamp(0.0625, 16.0);
        }
    }
}

#[inline]
fn worst_stereo_report(
    l: TierCertificationReport,
    r: TierCertificationReport,
) -> TierCertificationReport {
    TierCertificationReport {
        tier: l.tier,
        besr_db: l.besr_db.max(r.besr_db),
        band_besr_db: [
            l.band_besr_db[0].max(r.band_besr_db[0]),
            l.band_besr_db[1].max(r.band_besr_db[1]),
            l.band_besr_db[2].max(r.band_besr_db[2]),
        ],
        mr_iir: l.mr_iir.max(r.mr_iir),
        ltas_max_db: l.ltas_max_db.max(r.ltas_max_db),
        seg_esr_db: l.seg_esr_db.max(r.seg_esr_db),
        excluded_bands: core::array::from_fn(|b| l.excluded_bands[b] && r.excluded_bands[b]),
        excluded_band_residuals: core::array::from_fn(|b| {
            l.excluded_band_residuals[b].max(r.excluded_band_residuals[b])
        }),
        meets_floor: l.meets_floor && r.meets_floor,
    }
}

pub(super) fn certify_single_tier_stereo(
    tier: Tier,
    render: &(Vec<f64>, Vec<f64>),
    target_l: &[f64],
    target_r: &[f64],
    sr_hz: f64,
    warmup: usize,
    gap_masks: Option<&[DrySilentGapMask; 2]>,
) -> Result<
    (
        f64,
        TierCertificationReport,
        TierCertificationReport,
        TierCertificationReport,
    ),
    TrainError,
> {
    let n = target_l.len().min(target_r.len());
    let lufs = compute_itu_bs1770_lufs(&render.0, &render.1, sr_hz, warmup);
    let mask_l = gap_masks.map(|m| &m[0]);
    let mask_r = gap_masks.map(|m| &m[1]);
    let rep_l =
        certify_tier_audio_with_gaps(tier, &render.0, &target_l[..n], None, sr_hz, warmup, mask_l)?;
    let rep_r = if render.0 == render.1 && target_l[..n] == target_r[..n] && mask_l == mask_r {
        rep_l
    } else {
        certify_tier_audio_with_gaps(tier, &render.1, &target_r[..n], None, sr_hz, warmup, mask_r)?
    };
    Ok((lufs, worst_stereo_report(rep_l, rep_r), rep_l, rep_r))
}

#[must_use]
pub(super) fn recompute_pair_monotone(
    rep_l: &[TierCertificationReport; 4],
    rep_r: &[TierCertificationReport; 4],
) -> [bool; 4] {
    let mut pair_monotone = [true; 4];
    for i in 1..4 {
        pair_monotone[i] = is_tier_pair_monotone(&rep_l[i - 1], &rep_l[i])
            && is_tier_pair_monotone(&rep_r[i - 1], &rep_r[i]);
    }
    pair_monotone
}

pub(super) fn certify_all_tier_renders_stereo(
    renders: &[(Vec<f64>, Vec<f64>); 4],
    target_l: &[f64],
    target_r: &[f64],
    sr_hz: f64,
    warmup: usize,
    gap_masks: Option<&[DrySilentGapMask; 2]>,
) -> Result<StereoCertBundle, TrainError> {
    let mut tier_lufs = [0.0_f64; 4];
    let mut combined = Vec::with_capacity(4);
    let mut rep_l_vec = Vec::with_capacity(4);
    let mut rep_r_vec = Vec::with_capacity(4);
    for (i, tier) in [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra]
        .into_iter()
        .enumerate()
    {
        let (lufs, worst, rep_l, rep_r) = certify_single_tier_stereo(
            tier,
            &renders[i],
            target_l,
            target_r,
            sr_hz,
            warmup,
            gap_masks,
        )?;
        tier_lufs[i] = lufs;
        combined.push(worst);
        rep_l_vec.push(rep_l);
        rep_r_vec.push(rep_r);
    }

    let tier_reports: [TierCertificationReport; 4] =
        combined
            .try_into()
            .map_err(|_| TrainError::InvalidDimensions {
                context: "certify_all_tier_renders_stereo",
                expected: 4,
                actual: 0,
            })?;
    let rep_l: [TierCertificationReport; 4] =
        rep_l_vec
            .try_into()
            .map_err(|_| TrainError::InvalidDimensions {
                context: "certify_all_tier_renders_stereo_l",
                expected: 4,
                actual: 0,
            })?;
    let rep_r: [TierCertificationReport; 4] =
        rep_r_vec
            .try_into()
            .map_err(|_| TrainError::InvalidDimensions {
                context: "certify_all_tier_renders_stereo_r",
                expected: 4,
                actual: 0,
            })?;
    let pair_monotone = recompute_pair_monotone(&rep_l, &rep_r);
    Ok(StereoCertBundle {
        tier_lufs,
        tier_reports,
        rep_l,
        rep_r,
        pair_monotone,
    })
}

pub(super) fn apply_full_quad_phase3_audit(model: &mut SonaModelF64) -> Result<f64, TrainError> {
    // Evaluate `[CLAMP-HEAD-GAIN]` strictly on `LinearReadout`'s direct `{u, HP_p(u)}` bypass
    // per `ref[dsp.head.mid-side-factorization]` across Ultra and all tier overrides without
    // truncating resonant Kautz/FIR peaks.
    let mut readout_gains_l = vec![model.direct_s2_u_gain[0], model.direct_s2_hp_gain[0]];
    let mut readout_gains_r = vec![model.direct_s2_u_gain[1], model.direct_s2_hp_gain[1]];
    let mut mags_l = evaluate_readout_spectral_magnitudes(
        &readout_gains_l,
        model.ro_pole_48k,
        model.sample_rate_hz,
        32,
    );
    let mut mags_r = evaluate_readout_spectral_magnitudes(
        &readout_gains_r,
        model.ro_pole_48k,
        model.sample_rate_hz,
        32,
    );
    for pfx in model.tier_overrides.iter().flatten() {
        let gl = [pfx.direct_s2_u_gain[0], pfx.direct_s2_hp_gain[0]];
        let gr = [pfx.direct_s2_u_gain[1], pfx.direct_s2_hp_gain[1]];
        readout_gains_l.extend_from_slice(&gl);
        readout_gains_r.extend_from_slice(&gr);
        let ov_l =
            evaluate_readout_spectral_magnitudes(&gl, model.ro_pole_48k, model.sample_rate_hz, 32);
        let ov_r =
            evaluate_readout_spectral_magnitudes(&gr, model.ro_pole_48k, model.sample_rate_hz, 32);
        for (dst, src) in mags_l.iter_mut().zip(ov_l) {
            *dst = if dst.is_finite() && src.is_finite() {
                dst.max(src)
            } else {
                f64::NAN
            };
        }
        for (dst, src) in mags_r.iter_mut().zip(ov_r) {
            *dst = if dst.is_finite() && src.is_finite() {
                dst.max(src)
            } else {
                f64::NAN
            };
        }
    }

    let min_delay_samples = model
        .transport_quads
        .iter()
        .flat_map(|t| t.delay_samples)
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min)
        .clamp(5.0, 48_000.0);
    let max_delay_excursion = model
        .transport_quads
        .iter()
        .flat_map(|t| t.lfo_depths)
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    let max_paralru_r_excursion = model
        .core_quads
        .iter()
        .flat_map(|q| q.lru_alpha_rel)
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);
    let max_tank_g_excursion = model
        .transport_quads
        .iter()
        .map(|t| t.g_loop)
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);

    let mut p3_in = Phase3AuditInput {
        min_delay_samples,
        max_delay_excursion,
        max_paralru_r_excursion,
        max_tank_g_excursion,
        crossbar_coupled: true,
        ro_pole_48k: (model.ro_pole_48k as f32).clamp(0.0, MAX_RO_POLE_48K_F32),
        readout_gains_l,
        readout_gains_r,
        freq_mag_l: mags_l,
        freq_mag_r: mags_r,
        burst_peak_abs: 1.0,
    };
    let p3_rep = audit_phase3_dynamic_and_head_gain(&mut p3_in)?;

    for tq in &mut model.transport_quads {
        tq.g_loop = tq.g_loop.min(p3_in.max_tank_g_excursion);
        for d in &mut tq.delay_samples {
            *d = d.max(p3_rep.tightened_min_delay);
        }
        for lfo in &mut tq.lfo_depths {
            *lfo = lfo.clamp(0.0, p3_rep.clamped_delay_excursion.min(512.0));
        }
    }
    if p3_rep.head_gain_scale < 1.0 {
        let s = p3_rep.head_gain_scale;
        scale_readout_gains_by(model, s);
    }
    model.ro_pole_48k = f64::from(p3_rep.ro_pole_48k).min(MAX_RO_POLE_48K_F64);
    Ok(p3_rep.head_gain_scale)
}

fn scale_readout_gains_by(model: &mut SonaModelF64, s: f64) {
    // Scale strictly the post-convolver summation side (`direct_s2` + `w_ro[..][6..12]` +
    // convolver outputs `fir` and `kautz.c_*`) so `direct_s2 + Convolver(pre_conv)` scales
    // linearly by `s` rather than quadratically (`s^2`).
    for slot in &mut model.direct_s2_u_gain {
        *slot *= s;
    }
    for slot in &mut model.direct_s2_hp_gain {
        *slot *= s;
    }
    for row in &mut model.w_ro {
        for col in &mut row[6..12] {
            *col *= s;
        }
    }
    for tap in &mut model.fir_l {
        *tap *= s;
    }
    for tap in &mut model.fir_r {
        *tap *= s;
    }
    for k in &mut model.kautz {
        k.c_re_l *= s;
        k.c_im_l *= s;
        k.c_re_r *= s;
        k.c_im_r *= s;
    }
    for ov in &mut model.tier_overrides {
        if let Some(pfx) = ov.as_deref_mut() {
            for c in 0..2 {
                pfx.direct_s2_u_gain[c] *= s;
                pfx.direct_s2_hp_gain[c] *= s;
            }
            for row in &mut pfx.w_ro {
                for cell in &mut row[6..12] {
                    *cell *= s;
                }
            }
            for tap in &mut pfx.fir_l {
                *tap *= s;
            }
            for tap in &mut pfx.fir_r {
                *tap *= s;
            }
            for k in 0..pfx.kautz_c_re_l.len() {
                pfx.kautz_c_re_l[k] *= s;
                pfx.kautz_c_im_l[k] *= s;
                pfx.kautz_c_re_r[k] *= s;
                pfx.kautz_c_im_r[k] *= s;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::model::{SonaModelF64, TierReadoutPrefixF64};
    use crate::solver::pass5_emit::{Pass5EmitConfig, run_pass5_emit_and_certify};

    #[test]
    fn test_head_gain_scale_across_overrides_and_empty_lufs_guard() {
        assert_eq!(compute_itu_bs1770_lufs(&[], &[], 48_000.0, 10), -120.0);
        assert_eq!(
            compute_itu_bs1770_lufs(&[0.0; 256], &[0.0; 256], 48_000.0, 32),
            -120.0
        );
        let mut model = SonaModelF64::new(48_000.0);
        model.pre_conv_u_gain = [0.6, 0.4];
        model.pre_conv_hp_gain = [0.25, -0.15];
        model.w_ro[0][2] = 0.5;
        model.w_ro[0][8] = 0.8;
        model.direct_s2_u_gain = [1.0, 1.0];
        model.direct_s2_hp_gain = [0.0, 0.0];
        model.tier_overrides = [
            Some(Box::new(TierReadoutPrefixF64 {
                pre_conv_u_gain: [0.6, 0.4],
                pre_conv_hp_gain: [0.25, -0.15],
                direct_s2_u_gain: [3.0, 2.5],
                direct_s2_hp_gain: [1.5, 1.25],
                ..Default::default()
            })),
            None,
            Some(Box::new(TierReadoutPrefixF64::default())),
            None,
        ];
        model.tier_overrides[0].as_deref_mut().unwrap().w_ro[0][2] = 0.5;
        model.tier_overrides[0].as_deref_mut().unwrap().w_ro[0][8] = 0.8;
        let n = 512_usize;
        let in_l: Vec<f64> = (0..n).map(|i| 0.25 * ((i as f64) * 0.08).sin()).collect();
        let res = run_pass5_emit_and_certify(
            &mut model,
            &in_l,
            &in_l,
            &in_l,
            &in_l,
            64,
            Pass5EmitConfig::default(),
        )
        .unwrap();
        assert!(res.head_gain_scale > 0.0 && res.head_gain_scale < 1.0);
        assert!((model.direct_s2_u_gain[0] - res.head_gain_scale).abs() <= 1.0e-12);
        assert_eq!(model.pre_conv_u_gain, [0.6, 0.4]);
        assert_eq!(model.pre_conv_hp_gain, [0.25, -0.15]);
        assert_eq!(model.w_ro[0][2], 0.5);
        assert!((model.w_ro[0][8] - 0.8 * res.head_gain_scale).abs() <= 1.0e-12);
        let ov0 = model.tier_overrides[0].as_deref().unwrap();
        assert_eq!(ov0.pre_conv_hp_gain, [0.25, -0.15]);
        assert_eq!(ov0.w_ro[0][2], 0.5);
        assert!((ov0.direct_s2_u_gain[0] - 3.0 * res.head_gain_scale).abs() <= 1.0e-12);
    }

    #[test]
    fn test_zero_tier_delta_weights_syncs_tier1_and_tier2_fp16_bands() {
        let mut model = SonaModelF64::new(48_000.0);
        model.prefix_override_needed = [false, true, true, true];
        model.fir_l[20] = 1.234_567_89;
        model.kautz[5].c_re_l = 0.456_789_12;
        model.kautz[5].b_gamma = 0.789_123_45;
        let mut base = TierReadoutPrefixF64::default();
        base.fir_l[20] = 1.234_567_89;
        base.kautz_c_re_l[5] = 0.456_789_12;
        base.kautz_b_gamma[5] = 0.789_123_45;
        model.tier_overrides = [
            Some(Box::new(base.clone())),
            Some(Box::new(base.clone())),
            Some(Box::new(base.clone())),
            Some(Box::new(base)),
        ];
        zero_tier_delta_weights(&mut model, 1);
        let snapped_fir = f64::from(fp16_to_f32(f32_to_fp16(1.234_567_89_f64 as f32)));
        assert_eq!(
            model.tier_overrides[0].as_deref().unwrap().fir_l[20],
            snapped_fir
        );
        assert_eq!(
            model.tier_overrides[1].as_deref().unwrap().fir_l[20],
            snapped_fir
        );

        let mut model2 = SonaModelF64::new(48_000.0);
        model2.prefix_override_needed = [false, false, false, true];
        let mut ov = TierReadoutPrefixF64::default();
        ov.fir_r[35] = -0.321_654_98;
        ov.kautz_b_gamma[9] = 1.111_222_33;
        model2.tier_overrides = [
            Some(Box::new(ov.clone())),
            Some(Box::new(ov.clone())),
            Some(Box::new(ov.clone())),
            Some(Box::new(ov)),
        ];
        zero_tier_delta_weights(&mut model2, 2);
        let snapped_bg = f64::from(fp16_to_f32(f32_to_fp16(1.111_222_33_f64 as f32)));
        assert_eq!(
            model2.tier_overrides[1].as_deref().unwrap().kautz_b_gamma[9],
            snapped_bg
        );
        assert_eq!(
            model2.tier_overrides[2].as_deref().unwrap().kautz_b_gamma[9],
            snapped_bg
        );
    }
}
