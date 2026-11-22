//! Pure helpers for non-destructive Pass-5 adjacent-tier monotone enforcement
//! (`ref[train.pipeline.host-parity-and-export]`, `ref[qual.tiers.benchmark-matrix]`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use dnam_vm::bin::DnamBinView;

use crate::error::TrainError;
use crate::losses::TierCertificationReport;

use super::super::super::model::{
    NUM_FIR_TAPS_ULTRA, NUM_KAUTZ_PAIRS_ULTRA, SonaModelF64, TierReadoutPrefixF64,
};
use super::super::Pass5EmitConfig;
use super::super::certify::{
    MAX_INTER_TIER_LU_DRIFT, StereoCertBundle, TierRenderBundle, calibrate_lufs_trims_from_renders,
    certify_all_tier_renders_stereo, render_all_tiers,
};
use super::super::pack::encode_model_to_dnam_bin;

pub(super) const TIER_LANES: [usize; 4] = [16, 24, 32, 40];
pub(super) const TIER_FIRS: [usize; 4] = [32, 48, 96, 128];
pub(super) const TIER_KAUTZ: [usize; 4] = [8, 12, 24, 32];

pub(in crate::solver::pass5_emit) struct MonotoneEvalCtx<'a> {
    pub in_l: &'a [f64],
    pub in_r: &'a [f64],
    pub target_l: &'a [f64],
    pub target_r: &'a [f64],
    pub warmup: usize,
    pub config: Pass5EmitConfig,
}

pub(in crate::solver::pass5_emit) struct MonotoneState {
    pub final_bin: Vec<u8>,
    pub final_bundle: TierRenderBundle,
    pub cert: StereoCertBundle,
}

pub(super) struct RepairBaseline<'a> {
    pub besr: &'a [f64; 4],
    pub trims: &'a [f64; 4],
    pub overrides: &'a [TierReadoutPrefixF64; 4],
}

pub(super) fn build_masked_override_from_model(
    model: &SonaModelF64,
    tier_idx: usize,
) -> TierReadoutPrefixF64 {
    let mut ov = TierReadoutPrefixF64 {
        pre_conv_u_gain: model.pre_conv_u_gain,
        pre_conv_hp_gain: model.pre_conv_hp_gain,
        direct_s2_u_gain: model.direct_s2_u_gain,
        direct_s2_hp_gain: model.direct_s2_hp_gain,
        w_ro: model.w_ro,
        fir_l: model.fir_l,
        fir_r: model.fir_r,
        kautz_b_gamma: core::array::from_fn(|k| model.kautz[k].b_gamma),
        kautz_c_re_l: core::array::from_fn(|k| model.kautz[k].c_re_l),
        kautz_c_im_l: core::array::from_fn(|k| model.kautz[k].c_im_l),
        kautz_c_re_r: core::array::from_fn(|k| model.kautz[k].c_re_r),
        kautz_c_im_r: core::array::from_fn(|k| model.kautz[k].c_im_r),
    };
    for lane in &mut ov.w_ro[TIER_LANES[tier_idx]..40] {
        *lane = [0.0; 12];
    }
    ov.fir_l[TIER_FIRS[tier_idx]..NUM_FIR_TAPS_ULTRA].fill(0.0);
    ov.fir_r[TIER_FIRS[tier_idx]..NUM_FIR_TAPS_ULTRA].fill(0.0);
    for k in TIER_KAUTZ[tier_idx]..NUM_KAUTZ_PAIRS_ULTRA {
        ov.kautz_c_re_l[k] = 0.0;
        ov.kautz_c_im_l[k] = 0.0;
        ov.kautz_c_re_r[k] = 0.0;
        ov.kautz_c_im_r[k] = 0.0;
    }
    ov
}

pub(super) fn materialize_tier_overrides(model: &mut SonaModelF64) {
    materialize_w_ro_only_overrides(model);
    model.prefix_override_needed = [false, true, true, true];
}

pub(super) fn materialize_w_ro_only_overrides(model: &mut SonaModelF64) {
    for ti in 0..4 {
        if model.tier_overrides[ti].is_none() {
            let ov = build_masked_override_from_model(model, ti);
            model.tier_overrides[ti] = Some(Box::new(ov));
        }
    }
}

pub(super) fn blend_override_toward(
    dst: &mut TierReadoutPrefixF64,
    src_hi: &TierReadoutPrefixF64,
    src_lo: &TierReadoutPrefixF64,
    alpha: f64,
) {
    let w_hi = alpha.clamp(0.0, 1.0);
    let w_lo = 1.0 - w_hi;
    for c in 0..2 {
        dst.pre_conv_u_gain[c] =
            w_hi * src_hi.pre_conv_u_gain[c] + w_lo * src_lo.pre_conv_u_gain[c];
        dst.pre_conv_hp_gain[c] =
            w_hi * src_hi.pre_conv_hp_gain[c] + w_lo * src_lo.pre_conv_hp_gain[c];
        dst.direct_s2_u_gain[c] =
            w_hi * src_hi.direct_s2_u_gain[c] + w_lo * src_lo.direct_s2_u_gain[c];
        dst.direct_s2_hp_gain[c] =
            w_hi * src_hi.direct_s2_hp_gain[c] + w_lo * src_lo.direct_s2_hp_gain[c];
    }
    for l in 0..40 {
        for c in 0..12 {
            dst.w_ro[l][c] = w_hi * src_hi.w_ro[l][c] + w_lo * src_lo.w_ro[l][c];
        }
    }
    for k in 0..NUM_FIR_TAPS_ULTRA {
        dst.fir_l[k] = w_hi * src_hi.fir_l[k] + w_lo * src_lo.fir_l[k];
        dst.fir_r[k] = w_hi * src_hi.fir_r[k] + w_lo * src_lo.fir_r[k];
    }
    dst.kautz_b_gamma = src_hi.kautz_b_gamma;
    for k in 0..NUM_KAUTZ_PAIRS_ULTRA {
        dst.kautz_c_re_l[k] = w_hi * src_hi.kautz_c_re_l[k] + w_lo * src_lo.kautz_c_re_l[k];
        dst.kautz_c_im_l[k] = w_hi * src_hi.kautz_c_im_l[k] + w_lo * src_lo.kautz_c_im_l[k];
        dst.kautz_c_re_r[k] = w_hi * src_hi.kautz_c_re_r[k] + w_lo * src_lo.kautz_c_re_r[k];
        dst.kautz_c_im_r[k] = w_hi * src_hi.kautz_c_im_r[k] + w_lo * src_lo.kautz_c_im_r[k];
    }
}

pub(super) fn blend_w_ro_and_direct_only(
    dst: &mut TierReadoutPrefixF64,
    src_hi: &TierReadoutPrefixF64,
    src_lo: &TierReadoutPrefixF64,
    alpha: f64,
) {
    let w_hi = alpha.clamp(0.0, 1.0);
    let w_lo = 1.0 - w_hi;
    for c in 0..2 {
        dst.pre_conv_u_gain[c] =
            w_hi * src_hi.pre_conv_u_gain[c] + w_lo * src_lo.pre_conv_u_gain[c];
        dst.pre_conv_hp_gain[c] =
            w_hi * src_hi.pre_conv_hp_gain[c] + w_lo * src_lo.pre_conv_hp_gain[c];
        dst.direct_s2_u_gain[c] =
            w_hi * src_hi.direct_s2_u_gain[c] + w_lo * src_lo.direct_s2_u_gain[c];
        dst.direct_s2_hp_gain[c] =
            w_hi * src_hi.direct_s2_hp_gain[c] + w_lo * src_lo.direct_s2_hp_gain[c];
    }
    for l in 0..40 {
        for c in 0..12 {
            dst.w_ro[l][c] = w_hi * src_hi.w_ro[l][c] + w_lo * src_lo.w_ro[l][c];
        }
    }
}

pub(super) fn damp_w_ro_and_direct_only(
    ov: &mut TierReadoutPrefixF64,
    tier_idx: usize,
    beta: f64,
    detail_only: bool,
) {
    let lanes = TIER_LANES[tier_idx];
    let start_lane = if detail_only { lanes / 2 } else { 0 };
    for lane in &mut ov.w_ro[start_lane..lanes] {
        for w in &mut lane[6..12] {
            *w *= beta;
        }
    }
    if !detail_only {
        for c in 0..2 {
            ov.direct_s2_hp_gain[c] *= beta;
            ov.pre_conv_u_gain[c] *= beta;
            ov.pre_conv_hp_gain[c] *= beta;
        }
        for lane in &mut ov.w_ro[0..lanes] {
            for w in &mut lane[0..6] {
                *w *= beta;
            }
        }
    }
}

pub(super) fn damp_lower_tier_detail(ov: &mut TierReadoutPrefixF64, tier_idx: usize, beta: f64) {
    let lanes = TIER_LANES[tier_idx];
    let fir_cap = TIER_FIRS[tier_idx];
    let kautz_cap = TIER_KAUTZ[tier_idx];
    for lane in &mut ov.w_ro[(lanes / 2)..lanes] {
        for w in &mut lane[6..12] {
            *w *= beta;
        }
    }
    for k in (fir_cap / 4).max(2)..fir_cap {
        ov.fir_l[k] *= beta;
        ov.fir_r[k] *= beta;
    }
    for k in (kautz_cap / 2)..kautz_cap {
        ov.kautz_c_re_l[k] *= beta;
        ov.kautz_c_im_l[k] *= beta;
        ov.kautz_c_re_r[k] *= beta;
        ov.kautz_c_im_r[k] *= beta;
    }
}

pub(super) fn damp_lower_tier_override(ov: &mut TierReadoutPrefixF64, tier_idx: usize, beta: f64) {
    let lanes = TIER_LANES[tier_idx];
    let fir_cap = TIER_FIRS[tier_idx];
    let kautz_cap = TIER_KAUTZ[tier_idx];
    for lane in &mut ov.w_ro[..lanes] {
        for w in &mut lane[6..12] {
            *w *= beta;
        }
    }
    for k in 1..fir_cap {
        ov.fir_l[k] *= beta;
        ov.fir_r[k] *= beta;
    }
    for k in 0..kautz_cap {
        ov.kautz_c_re_l[k] *= beta;
        ov.kautz_c_im_l[k] *= beta;
        ov.kautz_c_re_r[k] *= beta;
        ov.kautz_c_im_r[k] *= beta;
    }
}

pub(super) fn blur_lower_tier_override(ov: &mut TierReadoutPrefixF64, tier_idx: usize, gamma: f64) {
    let g = gamma.clamp(0.0, 0.95);
    let lanes = TIER_LANES[tier_idx];
    let fir_cap = TIER_FIRS[tier_idx];
    let kautz_cap = TIER_KAUTZ[tier_idx];

    // 1. Scale existing non-zero-lag FIR taps, then spread fir[0] causally across LF/MF/HF lags
    //    within Lite's 32-tap window while preserving DC gain (0.45 + 0.25 + 0.20 + 0.10 = 1.0).
    for k in 1..fir_cap {
        ov.fir_l[k] *= 1.0 - 0.60 * g;
        ov.fir_r[k] *= 1.0 - 0.60 * g;
    }
    let f0_l = ov.fir_l[0];
    let f0_r = ov.fir_r[0];
    ov.fir_l[0] = (1.0 - g) * f0_l;
    ov.fir_r[0] = (1.0 - g) * f0_r;
    for &(tap, w) in &[(1_usize, 0.45_f64), (4, 0.25), (12, 0.20), (24, 0.10)] {
        if tap < fir_cap {
            ov.fir_l[tap] += w * g * f0_l;
            ov.fir_r[tap] += w * g * f0_r;
        }
    }

    // 2. Tilt direct bypass from flat into HPF (shifts LF phase/mag without cancelling under LUFS trim).
    for c in 0..2 {
        let d0 = ov.direct_s2_u_gain[c];
        ov.direct_s2_u_gain[c] = (1.0 - g) * d0;
        ov.direct_s2_hp_gain[c] = ov.direct_s2_hp_gain[c] * (1.0 - 0.50 * g) + g * d0;
    }

    // 3. Non-uniform lane/feature damping on core readouts so scalar LUFS trim cannot undo it.
    for (l, lane) in ov.w_ro[..lanes].iter_mut().enumerate() {
        for (c_off, w) in lane[6..12].iter_mut().enumerate() {
            let frac = (((l * 3 + c_off) % 11) as f64) / 10.0;
            *w *= 1.0 - g * (0.10 + 0.80 * frac);
        }
    }

    // 4. Damp lower-tier Kautz residues (never touches Ultra).
    for k in 0..kautz_cap {
        let s = 1.0 - 0.85 * g;
        ov.kautz_c_re_l[k] *= s;
        ov.kautz_c_im_l[k] *= s;
        ov.kautz_c_re_r[k] *= s;
        ov.kautz_c_im_r[k] *= s;
    }
}

#[inline]
fn snap_fp16_bounds(val: &mut f64, lo: f32, hi: f32) {
    let clamped = if val.is_finite() {
        (*val as f32).clamp(lo, hi)
    } else {
        0.0
    };
    *val = f64::from(fp16_to_f32(f32_to_fp16(clamped)));
}

pub(super) fn snap_override_fp16_slice(
    ov: &mut TierReadoutPrefixF64,
    fir_range: core::ops::Range<usize>,
    bg_range: core::ops::Range<usize>,
    c_range: core::ops::Range<usize>,
) {
    for k in fir_range {
        snap_fp16_bounds(&mut ov.fir_l[k], -16.0, 16.0);
        snap_fp16_bounds(&mut ov.fir_r[k], -16.0, 16.0);
    }
    for k in bg_range {
        snap_fp16_bounds(&mut ov.kautz_b_gamma[k], 0.0, 8.0);
    }
    for k in c_range {
        snap_fp16_bounds(&mut ov.kautz_c_re_l[k], -8.0, 8.0);
        snap_fp16_bounds(&mut ov.kautz_c_im_l[k], -8.0, 8.0);
        snap_fp16_bounds(&mut ov.kautz_c_re_r[k], -8.0, 8.0);
        snap_fp16_bounds(&mut ov.kautz_c_im_r[k], -8.0, 8.0);
    }
}

pub(super) fn zero_tier_delta_readouts_only(model: &mut SonaModelF64, tier_idx: usize) {
    let (lane_range, fir_range, kautz_range) = match tier_idx {
        1 => (16..24, 32..48, 8..12),
        2 => (24..32, 48..96, 12..24),
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
    model.transport_quads[tier_idx].input_gains = [0.0; 4];
    model.transport_quads[tier_idx].wet_tap_l = [0.0; 4];
    model.transport_quads[tier_idx].wet_tap_r = [0.0; 4];

    if let Some(prev) = model.tier_overrides[tier_idx - 1].as_deref().cloned() {
        let mut next = prev.clone();
        for lane in &mut next.w_ro[lane_range] {
            *lane = [0.0; 12];
        }
        for k in fir_range.clone() {
            next.fir_l[k] = 0.0;
            next.fir_r[k] = 0.0;
        }
        for k in kautz_range.clone() {
            next.kautz_c_re_l[k] = 0.0;
            next.kautz_c_im_l[k] = 0.0;
            next.kautz_c_re_r[k] = 0.0;
            next.kautz_c_im_r[k] = 0.0;
        }
        if tier_idx == 1 {
            snap_override_fp16_slice(&mut next, 16..32, 0..8, 4..8);
        } else {
            snap_override_fp16_slice(&mut next, 16..48, 0..12, 4..12);
        }
        if let Some(prev_mut) = model.tier_overrides[tier_idx - 1].as_deref_mut() {
            if tier_idx == 1 {
                snap_override_fp16_slice(prev_mut, 16..32, 0..8, 4..8);
            } else {
                snap_override_fp16_slice(prev_mut, 16..48, 0..12, 4..12);
            }
        }
        model.tier_overrides[tier_idx] = Some(Box::new(next));
    }
}

#[inline]
pub(super) fn tier_lufs_and_trim_ok(
    model: &SonaModelF64,
    orig_trims: &[f64; 4],
    cert: &StereoCertBundle,
    config: Pass5EmitConfig,
    tiers: core::ops::RangeInclusive<usize>,
    strict_trim_envelope: bool,
) -> bool {
    if !config.calibrate_lufs_trims || cert.tier_lufs[3] <= -100.0 {
        return true;
    }
    for k in tiers {
        if k >= 3 {
            continue;
        }
        if (cert.tier_lufs[k] - cert.tier_lufs[3]).abs() > MAX_INTER_TIER_LU_DRIFT {
            return false;
        }
        let g = model.tier_gain_trim[k];
        if !g.is_finite() || !(0.0625..=16.0).contains(&g) {
            return false;
        }
        if strict_trim_envelope
            && (0.25..=4.0).contains(&orig_trims[k])
            && !(0.25..=4.0).contains(&g)
        {
            return false;
        }
    }
    true
}

pub(super) fn re_eval(
    model: &mut SonaModelF64,
    in_l: &[f64],
    in_r: &[f64],
    target_l: &[f64],
    target_r: &[f64],
    warmup: usize,
    config: Pass5EmitConfig,
) -> Result<(Vec<u8>, TierRenderBundle, StereoCertBundle), TrainError> {
    let sr_f32 = model.sample_rate_hz as f32;
    model.tier_gain_trim = [1.0; 4];
    if config.calibrate_lufs_trims {
        let mid_bin = encode_model_to_dnam_bin(model, config)?;
        let mid_bundle = render_all_tiers(&mid_bin, in_l, in_r, sr_f32)?;
        calibrate_lufs_trims_from_renders(model, &mid_bundle.renders, warmup);
    }
    let final_bin = encode_model_to_dnam_bin(model, config)?;
    let _ = DnamBinView::parse(&final_bin)?;
    let final_bundle = render_all_tiers(&final_bin, in_l, in_r, sr_f32)?;
    let cert = certify_all_tier_renders_stereo(
        &final_bundle.renders,
        target_l,
        target_r,
        model.sample_rate_hz,
        warmup,
    )?;
    Ok((final_bin, final_bundle, cert))
}

#[inline]
pub(super) fn pass_count(cert: &StereoCertBundle) -> usize {
    cert.pair_monotone[1..].iter().filter(|&&m| m).count()
}

#[inline]
pub(super) fn others_intact(
    prev: &StereoCertBundle,
    cand: &StereoCertBundle,
    target_t: usize,
) -> bool {
    (1..=3).all(|u| u == target_t || !prev.pair_monotone[u] || cand.pair_monotone[u])
}

#[inline]
pub(super) fn strict_besr_preserved(
    orig_besr: &[f64; 4],
    cand: &StereoCertBundle,
    t: usize,
) -> bool {
    orig_besr[t] >= orig_besr[t - 1] - 0.1
        || cand.tier_reports[t].besr_db < cand.tier_reports[t - 1].besr_db
}

#[inline]
fn scalar_excess(lo: f64, hi: f64) -> f64 {
    if lo.is_nan() || hi.is_nan() {
        return 1.0e6;
    }
    if hi > lo {
        let diff = hi - lo;
        if diff.is_finite() { diff } else { 1.0e6 }
    } else {
        0.0
    }
}

#[inline]
fn channel_pair_deficit(lo: &TierCertificationReport, hi: &TierCertificationReport) -> f64 {
    let mut def = scalar_excess(lo.besr_db, hi.besr_db)
        + scalar_excess(lo.ltas_max_db, hi.ltas_max_db)
        + scalar_excess(lo.mr_iir, hi.mr_iir)
        + scalar_excess(lo.seg_esr_db, hi.seg_esr_db);
    for b in 0..3 {
        def += scalar_excess(lo.band_besr_db[b], hi.band_besr_db[b]);
    }
    def
}

pub(super) fn pair_deficit_score(cert: &StereoCertBundle, t: usize) -> f64 {
    if t == 0 || t > 3 {
        return 0.0;
    }
    channel_pair_deficit(&cert.rep_l[t - 1], &cert.rep_l[t])
        + channel_pair_deficit(&cert.rep_r[t - 1], &cert.rep_r[t])
}
