#![forbid(unsafe_code)]
//! Stage E 4-Tier 6-Metric Monotonicity Polish (`solver::stage_e_polish`).
//!
//! Ensures `[Lite, Full, Pro, Ultra]` satisfy `verify_monotone_4tier_summaries` on normalized
//! LUFS output before final Stage E trim packing by convexly scaling incremental higher-tier
//! `W_head` (`Modes 0..4` + Ultra `Modes 5..8`) while preserving Stage III Kautz `c` residues.

use dnam_engine::container::{build_dnam_container_with_cp1, strip_container_to_tier};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{STAGE3_KAUTZ_BLOCKS_PER_TIER, kautz_soa_byte_offset};
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};

use crate::metrics::certification::{
    PerceptualCertificationSummary, certify_blackbox_coreset_chain,
};
use crate::metrics::{
    ArchetypeFamily, evaluate_besr_from_band_energies, try_split_bezout_3band_energies,
};
use crate::proj::DnamProjManifest;

use super::stage_a::STAGE3_EARLY_FIR_BASES;
use super::stage_d::star_center_control_values;
use super::stage_e::compute_integrated_lufs_f64;

const ARENA0_TIER_GAIN_TRIM_OFFSET: usize = 10_624;
const HEAD_BASES: [usize; 4] = [4_928, 18_048, 31_936, 57_600];
const HEAD_MODES_0_TO_4_BYTES: usize = 2_560;
const ULTRA_MODES_5_TO_8_BASE: usize = 49_408;
const ULTRA_MODES_5_TO_8_BYTES: usize = 8_192;
const ULTRA_B_HEAD_BASE: usize = 63_424;
const ULTRA_B_HEAD_BYTES: usize = 32;
const EARLY_FIR_TIER_BYTES: usize = 128;
const KAUTZ_C_BLOCK_OFFSET: usize = 64;
const KAUTZ_C_BLOCK_BYTES: usize = 64;
const KAUTZ_C_FLOOR: f32 = 4.0e-4;
const TIERS_ALL: [Tier; 4] = [Tier::Lite, Tier::Full, Tier::Pro, Tier::Ultra];

struct TierDeltaSnapshot {
    head_modes0_4: Vec<u8>,
    ultra_modes5_8: Vec<u8>,
    ultra_b_head: Vec<u8>,
}

impl TierDeltaSnapshot {
    fn capture(arena_64k: &[u8], tier_idx: usize) -> Self {
        let hb = HEAD_BASES[tier_idx];
        let head_modes0_4 = arena_64k
            .get(hb..hb + HEAD_MODES_0_TO_4_BYTES)
            .map(<[u8]>::to_vec)
            .unwrap_or_default();
        let (ultra_modes5_8, ultra_b_head) = if tier_idx == 3 {
            (
                arena_64k
                    .get(
                        ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + ULTRA_MODES_5_TO_8_BYTES,
                    )
                    .map(<[u8]>::to_vec)
                    .unwrap_or_default(),
                arena_64k
                    .get(ULTRA_B_HEAD_BASE..ULTRA_B_HEAD_BASE + ULTRA_B_HEAD_BYTES)
                    .map(<[u8]>::to_vec)
                    .unwrap_or_default(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            head_modes0_4,
            ultra_modes5_8,
            ultra_b_head,
        }
    }

    fn apply_scaled(&self, arena_64k: &mut [u8], tier_idx: usize, alpha: f32) {
        let hb = HEAD_BASES[tier_idx];
        scale_f32_slice(
            &self.head_modes0_4,
            arena_64k.get_mut(hb..hb + HEAD_MODES_0_TO_4_BYTES),
            alpha,
        );
        if tier_idx == 3 {
            scale_f32_slice(
                &self.ultra_modes5_8,
                arena_64k.get_mut(
                    ULTRA_MODES_5_TO_8_BASE..ULTRA_MODES_5_TO_8_BASE + ULTRA_MODES_5_TO_8_BYTES,
                ),
                alpha,
            );
            scale_f32_slice(
                &self.ultra_b_head,
                arena_64k.get_mut(ULTRA_B_HEAD_BASE..ULTRA_B_HEAD_BASE + ULTRA_B_HEAD_BYTES),
                alpha,
            );
        }
    }
}

struct FullTierDeltaSnapshot {
    head: TierDeltaSnapshot,
    early_fir: Vec<u8>,
    kautz_c: Vec<Vec<u8>>,
}

impl FullTierDeltaSnapshot {
    fn capture(arena_64k: &[u8], tier_idx: usize) -> Self {
        let fb = STAGE3_EARLY_FIR_BASES[tier_idx];
        let early_fir = arena_64k
            .get(fb..fb + EARLY_FIR_TIER_BYTES)
            .map(<[u8]>::to_vec)
            .unwrap_or_default();
        let n_blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[tier_idx];
        let mut kautz_c = Vec::with_capacity(n_blocks);
        for blk in 0..n_blocks {
            let off = kautz_soa_byte_offset(tier_idx, blk) + KAUTZ_C_BLOCK_OFFSET;
            kautz_c.push(
                arena_64k
                    .get(off..off + KAUTZ_C_BLOCK_BYTES)
                    .map(<[u8]>::to_vec)
                    .unwrap_or_default(),
            );
        }
        Self {
            head: TierDeltaSnapshot::capture(arena_64k, tier_idx),
            early_fir,
            kautz_c,
        }
    }

    fn apply_scaled(&self, arena_64k: &mut [u8], tier_idx: usize, alpha: f32) {
        self.head.apply_scaled(arena_64k, tier_idx, alpha);
        let fb = STAGE3_EARLY_FIR_BASES[tier_idx];
        scale_f32_slice(
            &self.early_fir,
            arena_64k.get_mut(fb..fb + EARLY_FIR_TIER_BYTES),
            alpha,
        );
        for (blk, src) in self.kautz_c.iter().enumerate() {
            let off = kautz_soa_byte_offset(tier_idx, blk) + KAUTZ_C_BLOCK_OFFSET;
            scale_kautz_c_block(
                src,
                arena_64k.get_mut(off..off + KAUTZ_C_BLOCK_BYTES),
                alpha,
            );
        }
    }
}

fn scale_kautz_c_block(src: &[u8], dst: Option<&mut [u8]>, alpha: f32) {
    let Some(out) = dst else {
        return;
    };
    if src.len() != KAUTZ_C_BLOCK_BYTES || out.len() != KAUTZ_C_BLOCK_BYTES {
        return;
    }
    if alpha == 0.0 {
        out.fill(0);
        return;
    }
    let read_f32 = |buf: &[u8], off: usize| {
        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };
    for base in [0usize, 32usize] {
        for lane in 0..4 {
            let re_off = base + lane * 4;
            let im_off = base + 16 + lane * 4;
            let cre = read_f32(src, re_off);
            let cim = read_f32(src, im_off);
            let orig_mag = cre.hypot(cim);
            let eff = if orig_mag > 1.0e-7 {
                let min_mag = orig_mag.min(KAUTZ_C_FLOOR);
                if alpha * orig_mag < min_mag {
                    min_mag / orig_mag
                } else {
                    alpha
                }
            } else {
                alpha
            };
            out[re_off..re_off + 4].copy_from_slice(&(cre * eff).to_le_bytes());
            out[im_off..im_off + 4].copy_from_slice(&(cim * eff).to_le_bytes());
        }
    }
}

fn scale_f32_slice(src: &[u8], dst: Option<&mut [u8]>, alpha: f32) {
    let Some(out) = dst else {
        return;
    };
    if src.len() != out.len() {
        return;
    }
    for (s_chunk, d_chunk) in src
        .as_chunks::<4>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<4>().0.iter_mut())
    {
        let v = f32::from_le_bytes([s_chunk[0], s_chunk[1], s_chunk[2], s_chunk[3]]) * alpha;
        d_chunk.copy_from_slice(&v.to_le_bytes());
    }
}

fn combine_stereo_summary(
    l: PerceptualCertificationSummary,
    r: PerceptualCertificationSummary,
) -> PerceptualCertificationSummary {
    PerceptualCertificationSummary {
        family: l.family,
        tier: l.tier,
        raw_besr_db: l.raw_besr_db.max(r.raw_besr_db),
        besr_db: l.besr_db.max(r.besr_db),
        band_esr_db: [
            l.band_esr_db[0].max(r.band_esr_db[0]),
            l.band_esr_db[1].max(r.band_esr_db[1]),
            l.band_esr_db[2].max(r.band_esr_db[2]),
        ],
        qualified_band_mask: [
            l.qualified_band_mask[0] || r.qualified_band_mask[0],
            l.qualified_band_mask[1] || r.qualified_band_mask[1],
            l.qualified_band_mask[2] || r.qualified_band_mask[2],
        ],
        mr_iir: l.mr_iir.max(r.mr_iir),
        ltas_max_db: l.ltas_max_db.max(r.ltas_max_db),
        ltas_unadjusted_max_db: l.ltas_unadjusted_max_db.max(r.ltas_unadjusted_max_db),
        ltas_excluded_bands: std::array::from_fn(|b| {
            l.ltas_excluded_bands[b] && r.ltas_excluded_bands[b]
        }),
        ltas_residual_band_power: std::array::from_fn(|b| {
            l.ltas_residual_band_power[b].max(r.ltas_residual_band_power[b])
        }),
        seg_esr_db: l.seg_esr_db.max(r.seg_esr_db),
        passes_target: l.passes_target && r.passes_target,
        passes_hard_ceiling: l.passes_hard_ceiling && r.passes_hard_ceiling,
    }
}

fn render_tier_wave(
    dnam_bytes: &[u8],
    tier: Tier,
    star_ctrls: &[f32],
    dry_lr: &[[f32; 2]],
    n_common: usize,
    w_start: usize,
    mono_only: bool,
) -> Result<(Vec<[f32; 2]>, f64), DnamError> {
    let tier_bytes = if tier == Tier::Ultra {
        std::borrow::Cow::Borrowed(dnam_bytes)
    } else {
        std::borrow::Cow::Owned(strip_container_to_tier(dnam_bytes, tier)?)
    };
    let mut engine = DnamEngine::<DefaultMathBackend>::from_bytes(&tier_bytes, tier, 48_000)?;
    engine.reset_to_dc_equilibrium();
    for (p, &val) in star_ctrls.iter().enumerate() {
        engine.set_control_immediate(p, val);
    }
    let rendered: Vec<[f32; 2]> = dry_lr[..n_common]
        .iter()
        .map(|&[dl, dr]| engine.step_frame_tora(dl, dr, mono_only))
        .collect();
    let lufs = compute_integrated_lufs_f64(&rendered[w_start..n_common]);
    Ok((rendered, lufs))
}

fn fast_tier_besr(
    wave: &[[f32; 2]],
    lufs_t: f64,
    ultra_lufs: f64,
    tgt_l: &[f64],
    tgt_r: &[f64],
    w_start: usize,
    mono_only: bool,
) -> Option<(f64, [f64; 3])> {
    let trim_db = (ultra_lufs - lufs_t).clamp(-12.0, 12.0);
    let lin_trim = 10.0_f64.powf(trim_db / 20.0) as f32;
    let mdl_l: Vec<f64> = wave
        .iter()
        .map(|v| {
            f64::from(dnam_engine::math::sanitize_output_sample(
                lin_trim * v[0],
                &mut 0,
            ))
        })
        .collect();
    let (s_l, e_l, n_l) = try_split_bezout_3band_energies(tgt_l, &mdl_l, None, 48_000.0, w_start)?;
    let rl = evaluate_besr_from_band_energies(s_l, e_l, n_l);
    if !rl.valid {
        return None;
    }
    if mono_only {
        return Some((rl.besr_db, rl.band_esr_db));
    }
    let mdl_r: Vec<f64> = wave
        .iter()
        .map(|v| {
            f64::from(dnam_engine::math::sanitize_output_sample(
                lin_trim * v[1],
                &mut 0,
            ))
        })
        .collect();
    let (s_r, e_r, n_r) = try_split_bezout_3band_energies(tgt_r, &mdl_r, None, 48_000.0, w_start)?;
    let rr = evaluate_besr_from_band_energies(s_r, e_r, n_r);
    if !rr.valid {
        return None;
    }
    Some((
        rl.besr_db.max(rr.besr_db),
        [
            rl.band_esr_db[0].max(rr.band_esr_db[0]),
            rl.band_esr_db[1].max(rr.band_esr_db[1]),
            rl.band_esr_db[2].max(rr.band_esr_db[2]),
        ],
    ))
}

fn build_summaries_from_waves(
    waves: &[Vec<[f32; 2]>; 4],
    lufs: &[f64; 4],
    tgt_l: &[f64],
    tgt_r: &[f64],
    w_start: usize,
    mono_only: bool,
) -> [PerceptualCertificationSummary; 4] {
    let ultra_lufs = lufs[3];
    std::array::from_fn(|t| {
        let trim_db = (ultra_lufs - lufs[t]).clamp(-12.0, 12.0);
        let lin_trim = 10.0_f64.powf(trim_db / 20.0) as f32;
        let mdl_l: Vec<f64> = waves[t]
            .iter()
            .map(|v| {
                f64::from(dnam_engine::math::sanitize_output_sample(
                    lin_trim * v[0],
                    &mut 0,
                ))
            })
            .collect();
        let sl = certify_blackbox_coreset_chain(
            ArchetypeFamily::FullRigModDelayReverb,
            TIERS_ALL[t],
            tgt_l,
            &mdl_l,
            48_000.0,
            w_start,
        );
        if mono_only {
            sl
        } else {
            let mdl_r: Vec<f64> = waves[t]
                .iter()
                .map(|v| {
                    f64::from(dnam_engine::math::sanitize_output_sample(
                        lin_trim * v[1],
                        &mut 0,
                    ))
                })
                .collect();
            let sr = certify_blackbox_coreset_chain(
                ArchetypeFamily::FullRigModDelayReverb,
                TIERS_ALL[t],
                tgt_r,
                &mdl_r,
                48_000.0,
                w_start,
            );
            combine_stereo_summary(sl, sr)
        }
    })
}

fn summary_finite(s: &PerceptualCertificationSummary) -> bool {
    let valid_db = |x: f64| !x.is_nan() && x < f64::INFINITY;
    let valid_nonneg = |x: f64| x.is_finite() && x >= 0.0;
    valid_db(s.raw_besr_db)
        && valid_db(s.besr_db)
        && valid_nonneg(s.mr_iir)
        && valid_nonneg(s.ltas_max_db)
        && valid_nonneg(s.ltas_unadjusted_max_db)
        && valid_db(s.seg_esr_db)
        && s.band_esr_db.iter().copied().all(valid_db)
        && s.ltas_residual_band_power
            .iter()
            .all(|&p| p.is_finite() && p >= 0.0)
}

fn besr_valid(s: &PerceptualCertificationSummary) -> bool {
    let valid_db = |x: f64| !x.is_nan() && x < f64::INFINITY;
    valid_db(s.raw_besr_db) && valid_db(s.besr_db) && s.band_esr_db.iter().copied().all(valid_db)
}

fn pair_is_monotone(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
) -> bool {
    if !summary_finite(lo) {
        if !besr_valid(lo) || !besr_valid(hi) {
            return false;
        }
        if hi.raw_besr_db > lo.raw_besr_db || (0..3).any(|b| hi.band_esr_db[b] > lo.band_esr_db[b])
        {
            return false;
        }
        if lo.mr_iir.is_finite() && (!hi.mr_iir.is_finite() || hi.mr_iir > lo.mr_iir) {
            return false;
        }
        if lo.ltas_max_db.is_finite()
            && (!hi.ltas_max_db.is_finite() || hi.ltas_max_db > lo.ltas_max_db)
        {
            return false;
        }
        if !lo.seg_esr_db.is_nan()
            && lo.seg_esr_db < f64::INFINITY
            && (hi.seg_esr_db.is_nan() || hi.seg_esr_db > lo.seg_esr_db)
        {
            return false;
        }
        return true;
    }
    if !summary_finite(hi) || hi.ltas_excluded_bands != lo.ltas_excluded_bands {
        return false;
    }
    if hi.raw_besr_db > lo.raw_besr_db
        || hi.mr_iir > lo.mr_iir
        || hi.ltas_max_db > lo.ltas_max_db
        || hi.seg_esr_db > lo.seg_esr_db
    {
        return false;
    }
    if (0..3).any(|b| hi.band_esr_db[b] > lo.band_esr_db[b]) {
        return false;
    }
    (0..33).all(|b| {
        !hi.ltas_excluded_bands[b]
            || hi.ltas_residual_band_power[b] <= lo.ltas_residual_band_power[b]
    })
}

fn pairwise_violation_score(
    lo: &PerceptualCertificationSummary,
    hi: &PerceptualCertificationSummary,
) -> (usize, f64) {
    let mut count = 0_usize;
    let mut excess = 0.0_f64;
    let pairs = [
        (lo.raw_besr_db, hi.raw_besr_db),
        (lo.mr_iir, hi.mr_iir),
        (lo.ltas_max_db, hi.ltas_max_db),
        (lo.seg_esr_db, hi.seg_esr_db),
        (lo.band_esr_db[0], hi.band_esr_db[0]),
        (lo.band_esr_db[1], hi.band_esr_db[1]),
        (lo.band_esr_db[2], hi.band_esr_db[2]),
    ];
    for (lv, hv) in pairs {
        if lv.is_nan() || hv.is_nan() {
            count += 1;
            excess += 1.0e6;
        } else if hv > lv {
            count += 1;
            let d = hv - lv;
            excess += if d.is_finite() { d } else { 1.0e6 };
        }
    }
    if hi.ltas_excluded_bands != lo.ltas_excluded_bands {
        count += 1;
        excess += 10.0;
    }
    for b in 0..33 {
        if hi.ltas_excluded_bands[b]
            && hi.ltas_residual_band_power[b] > lo.ltas_residual_band_power[b]
        {
            count += 1;
            excess += 1.0;
        }
    }
    (count, excess)
}

/// Polishes incremental higher-tier weights (`Full..Ultra`) so all 4 tiers satisfy adjacent
/// 6-metric monotonicity at equalized `LUFS` prior to final Stage E trim computation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn polish_arena_for_6metric_monotonicity(
    manifest: &DnamProjManifest,
    arena_64k: &mut [u8],
    cp1_slots: &[Cp1ContributorSlot],
    dry_lr: &[[f32; 2]],
    wet_lr: &[[f32; 2]],
    n_common: usize,
    w_start: usize,
    mono_only: bool,
) -> Result<(), DnamError> {
    if n_common < w_start + 4_096 || arena_64k.len() < 64_000 {
        return Ok(());
    }
    for slot in 0..16 {
        if ARENA0_TIER_GAIN_TRIM_OFFSET + (slot + 1) * 4 <= arena_64k.len() {
            arena_64k[ARENA0_TIER_GAIN_TRIM_OFFSET + slot * 4
                ..ARENA0_TIER_GAIN_TRIM_OFFSET + (slot + 1) * 4]
                .copy_from_slice(&1.0_f32.to_le_bytes());
        }
    }

    let star_ctrls = star_center_control_values(manifest);
    let tgt_l: Vec<f64> = wet_lr[..n_common].iter().map(|v| f64::from(v[0])).collect();
    let tgt_r: Vec<f64> = wet_lr[..n_common].iter().map(|v| f64::from(v[1])).collect();
    let tgt_silent = tgt_l[w_start..n_common].iter().all(|v| v.abs() <= 1.0e-12)
        && (mono_only || tgt_r[w_start..n_common].iter().all(|v| v.abs() <= 1.0e-12));
    if tgt_silent {
        return Ok(());
    }

    let mut temp_hdr = manifest.base.clone();
    let dnam_bytes =
        build_dnam_container_with_cp1(&mut temp_hdr, arena_64k, arena_64k, false, cp1_slots)?;
    let mut waves: [Vec<[f32; 2]>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut lufs = [-70.0_f64; 4];
    for t in 0..4 {
        let (w, l) = render_tier_wave(
            &dnam_bytes,
            TIERS_ALL[t],
            &star_ctrls,
            dry_lr,
            n_common,
            w_start,
            mono_only,
        )?;
        waves[t] = w;
        lufs[t] = l;
    }

    let mut summaries =
        build_summaries_from_waves(&waves, &lufs, &tgt_l, &tgt_r, w_start, mono_only);
    if (1..4).all(|t| pair_is_monotone(&summaries[t - 1], &summaries[t])) {
        if mono_only {
            apply_mono_right_kautz_observability(arena_64k);
        }
        return Ok(());
    }

    let rerender_from = |arena: &[u8],
                         t_start: usize,
                         waves: &mut [Vec<[f32; 2]>; 4],
                         lufs: &mut [f64; 4]|
     -> Result<[PerceptualCertificationSummary; 4], DnamError> {
        let mut hdr = manifest.base.clone();
        let cand_bytes = build_dnam_container_with_cp1(&mut hdr, arena, arena, false, cp1_slots)?;
        for u in t_start..4 {
            let (w, l) = render_tier_wave(
                &cand_bytes,
                TIERS_ALL[u],
                &star_ctrls,
                dry_lr,
                n_common,
                w_start,
                mono_only,
            )?;
            waves[u] = w;
            lufs[u] = l;
        }
        Ok(build_summaries_from_waves(
            waves, lufs, &tgt_l, &tgt_r, w_start, mono_only,
        ))
    };

    let render_single_tier = |arena: &[u8], u: usize| -> Result<(Vec<[f32; 2]>, f64), DnamError> {
        let mut hdr = manifest.base.clone();
        let cand_bytes = build_dnam_container_with_cp1(&mut hdr, arena, arena, false, cp1_slots)?;
        render_tier_wave(
            &cand_bytes,
            TIERS_ALL[u],
            &star_ctrls,
            dry_lr,
            n_common,
            w_start,
            mono_only,
        )
    };

    const ALPHAS: [f32; 7] = [0.92, 0.80, 0.66, 0.50, 0.32, 0.16, 0.06];
    for t in 1..4 {
        if pair_is_monotone(&summaries[t - 1], &summaries[t]) {
            continue;
        }
        let init_besr_db = summaries[t].raw_besr_db;
        let init_score = pairwise_violation_score(&summaries[t - 1], &summaries[t]);
        let snap = TierDeltaSnapshot::capture(arena_64k, t);
        let mut accepted = false;
        let mut best_cand: Option<(f32, (usize, f64))> = None;

        for &alpha in &ALPHAS {
            snap.apply_scaled(arena_64k, t, alpha);
            let (w_t, l_t) = render_single_tier(arena_64k, t)?;
            let ref_ultra_lufs = if t == 3 { l_t } else { lufs[3] };
            if let Some((fb, _)) = fast_tier_besr(
                &w_t,
                l_t,
                ref_ultra_lufs,
                &tgt_l,
                &tgt_r,
                w_start,
                mono_only,
            ) && fb > init_besr_db + 0.35
            {
                continue;
            }
            waves[t] = w_t;
            lufs[t] = l_t;
            summaries = rerender_from(arena_64k, t + 1, &mut waves, &mut lufs)?;
            if summaries[t].raw_besr_db > init_besr_db + 0.35
                || (t + 1 < 4 && summaries[t + 1].raw_besr_db > summaries[t].raw_besr_db)
            {
                continue;
            }
            if pair_is_monotone(&summaries[t - 1], &summaries[t]) {
                accepted = true;
                break;
            }
            if summaries[t].raw_besr_db <= summaries[t - 1].raw_besr_db {
                let score = pairwise_violation_score(&summaries[t - 1], &summaries[t]);
                if best_cand
                    .is_none_or(|(_, bs)| score.0 < bs.0 || (score.0 == bs.0 && score.1 < bs.1))
                {
                    best_cand = Some((alpha, score));
                }
            }
        }

        if !accepted {
            let chosen = best_cand
                .and_then(|(a, sc)| (sc.0 < init_score.0).then_some(a))
                .unwrap_or(1.0);
            snap.apply_scaled(arena_64k, t, chosen);
            summaries = rerender_from(arena_64k, t, &mut waves, &mut lufs)?;
        }
    }

    const PASS2_ALPHAS: [f32; 13] = [
        0.95, 0.86, 0.74, 0.60, 0.45, 0.30, 0.16, 0.07, 0.02, 0.008, 0.002, 0.0005, 0.0,
    ];
    for _sweep in 0..2 {
        if (1..4).all(|t| pair_is_monotone(&summaries[t - 1], &summaries[t])) {
            break;
        }
        for t in 1..4 {
            if pair_is_monotone(&summaries[t - 1], &summaries[t]) {
                continue;
            }
            let snap = FullTierDeltaSnapshot::capture(arena_64k, t);
            let mut accepted = false;
            let mut best_cand: Option<(f32, (usize, f64))> = None;
            for &alpha in &PASS2_ALPHAS {
                snap.apply_scaled(arena_64k, t, alpha);
                let (w_t, l_t) = render_single_tier(arena_64k, t)?;
                let ref_ultra_lufs = if t == 3 { l_t } else { lufs[3] };
                if alpha > 0.0
                    && let Some((fb, fb_bands)) = fast_tier_besr(
                        &w_t,
                        l_t,
                        ref_ultra_lufs,
                        &tgt_l,
                        &tgt_r,
                        w_start,
                        mono_only,
                    )
                    && (fb > summaries[t - 1].raw_besr_db + 0.05
                        || (0..3).any(|b| fb_bands[b] > summaries[t - 1].band_esr_db[b] + 0.05))
                {
                    continue;
                }
                waves[t] = w_t;
                lufs[t] = l_t;
                summaries = rerender_from(arena_64k, t + 1, &mut waves, &mut lufs)?;
                if t + 1 < 4
                    && alpha > 0.0
                    && summaries[t + 1].raw_besr_db > summaries[t].raw_besr_db
                {
                    continue;
                }
                if pair_is_monotone(&summaries[t - 1], &summaries[t]) {
                    accepted = true;
                    break;
                }
                if besr_valid(&summaries[t]) {
                    let score = pairwise_violation_score(&summaries[t - 1], &summaries[t]);
                    if best_cand
                        .is_none_or(|(_, bs)| score.0 < bs.0 || (score.0 == bs.0 && score.1 < bs.1))
                    {
                        best_cand = Some((alpha, score));
                    }
                }
            }
            if !accepted {
                let chosen = best_cand.map(|(a, _)| a).unwrap_or(0.0);
                snap.apply_scaled(arena_64k, t, chosen);
                summaries = rerender_from(arena_64k, t, &mut waves, &mut lufs)?;
            }
        }
    }

    if mono_only {
        apply_mono_right_kautz_observability(arena_64k);
    }

    Ok(())
}

/// Preserves right-channel Kautz pole observability during mono-only polish so Tier 1..3
/// phase-coded watermark extraction via `render_blackbox_probe_response` remains well-conditioned
/// without perturbing Stage IV tank feedback or `mono_only` `out_l` audio (`0 ULP`).
fn apply_mono_right_kautz_observability(arena_64k: &mut [u8]) {
    if !dnam_engine::stages::weights::has_pipeline_flag(
        arena_64k,
        dnam_engine::stages::weights::PIPE_FLAG_STAGE3_EXT,
    ) {
        return;
    }
    let read_f32 = |buf: &[u8], off: usize| {
        f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };
    for (t, &n_blocks) in STAGE3_KAUTZ_BLOCKS_PER_TIER.iter().enumerate().skip(1) {
        let hb = HEAD_BASES[t];
        for local_f in 0..(n_blocks * 4) {
            let off_r = hb + 256 + local_f * 4;
            if off_r + 4 <= arena_64k.len() && read_f32(arena_64k, off_r).abs() < 1.0e-4 {
                arena_64k[off_r..off_r + 4].copy_from_slice(&1.0e-3_f32.to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::stage_e::*;
    use crate::proj::{
        DnamProjManifest, ProjChunkInput, ProjChunkKind, build_dnam_proj_container,
        build_proj_inpc_capsule,
    };
    use dnam_engine::Tier;
    use dnam_engine::container::{ARENA_OFFSET_ULTRA_END, build_dnam_container_with_cp1};
    use dnam_engine::schema::DnamJsonHeader;

    #[test]
    fn test_stage_e_watermark_edge_branches_and_tier_probes() {
        assert_eq!(compute_integrated_lufs(&[]), -70.0);
        assert_eq!(compute_integrated_lufs(&[[0.0, 0.0]; 256]), -70.0);

        let zero_cfg = BuyerWatermarkConfig {
            marketplace_secret: [0u8; 32],
            model_id: "m".to_string(),
            license_id: "l".to_string(),
            buyer_id: [0u8; 16],
            phase_coding_delta_rad: 2.5e-5,
            epsilon_mask_lr: [1.0e-5, 1.0e-5],
            compensate_role_b_in_head: true,
        };
        assert_eq!(derive_buyer_tardos_codebook(&zero_cfg).chips.len(), 7_680);

        let manifest = DnamProjManifest::default_reference();
        let inpc =
            build_proj_inpc_capsule(1, &[0u8; 256], &[0u8; 320], &vec![0u8; 24_000]).unwrap();
        let chunk = |kind, payload| ProjChunkInput {
            kind,
            chunk_id: 0,
            payload,
        };
        let un_solved =
            build_dnam_proj_container(&manifest, &[chunk(ProjChunkKind::Inpc, inpc.clone())])
                .unwrap();
        assert!(export_runtime_dnam_watermarked(&un_solved, &zero_cfg, None).is_err());

        let mut hdr = DnamJsonHeader::default_reference();
        let mut arena = vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        for (o, v) in [(4_176, 0.9), (4_180, 0.25), (4_208, 0.9), (4_212, 0.35)] {
            write_f32_le(&mut arena, o, v);
        }
        assert_eq!(extract_arena_pole_angles_34(&arena).len(), 34);
        let dnam_u = build_dnam_container_with_cp1(&mut hdr, &arena, &arena, false, &[]).unwrap();
        let solved = build_dnam_proj_container(
            &manifest,
            &[
                chunk(ProjChunkKind::Inpc, inpc),
                chunk(ProjChunkKind::Wght, dnam_u.clone()),
            ],
        )
        .unwrap();
        for tier in [Tier::Lite, Tier::Full, Tier::Pro] {
            let wm_t = export_runtime_dnam_watermarked(&solved, &zero_cfg, Some(tier)).unwrap();
            let probe = render_blackbox_probe_response(&wm_t, 64).unwrap();
            assert_eq!(probe.len(), 64);
        }
        assert!(identify_pole_angles_blackbox_from_audio(&dnam_u, &[[0.0, 0.0]; 16]).is_err());
        assert_eq!(select_eval_window(&[], 64), (0, 0));
        assert_eq!(select_eval_window(&[[0.1, 0.1]; 80], 64), (64, 80));
        let mut win_sig = [[0.01_f32, 0.01]; 256];
        for s in &mut win_sig[128..192] {
            *s = [0.9, -0.9];
        }
        assert_eq!(select_eval_window(&win_sig, 64), (128, 192));
    }
}
