#![forbid(unsafe_code)]
//! 3-Layer Buyer Watermarking (`GRU` Sign-Flip Gauge, `SO(4)` + Kautz Isometric Gauge,
//! Layer-2 Kautz Pole-Angle Phase Coding & Black-Box Phase ID, Layer-3 Tardos 7680).
//!
//! Canonical Reference: `docs/specs/security/watermarking_and_marketplace.md` (`ref[sec.watermark.reciprocal-gauge]`).

use std::f64::consts::TAU;

use dnam_engine::container::{
    DnamContainerView, build_dnam_container_with_cp1, strip_container_to_tier,
};
use dnam_engine::control::Cp1ContributorSlot;
use dnam_engine::math::DefaultMathBackend;
use dnam_engine::provenance::{
    DefaultCryptoProvider, PHASE_CODING_MAX_RAD, STAGE3_KAUTZ_BASES, STAGE3_KAUTZ_BLOCKS_PER_TIER,
    TARDOS_CASCADE_RANGES, TARDOS_CASCADE_WEIGHTS, WatermarkSeedScope,
    apply_gru_sign_gauge_watermark, apply_isometric_gauge_watermark, apply_phase_coding_watermark,
    apply_tardos_watermark_7680_envelopes, derive_gru_sign_gauge_24,
    derive_watermark_seed_hmac_sha256, generate_tardos_codebook_7680, sha256_digest,
};
use dnam_engine::schema::DnamError;
use dnam_engine::stages::weights::{
    ARENA_FULL_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES, ARENA1_CTRL_PARAMS_BYTES,
    ARENA1_DIRECT_READOUT_BYTES, ARENA2_DIRECT_READOUT_BYTES, ARENA3_RESERVE_OFFSET,
};
use dnam_engine::{DnamEngine, Tier};

use crate::proj::DnamProjView;
use crate::solver::{solve_uplo_u_cholesky_2ch, uplo_u_index, uplo_u_packed_len};

/// Buyer watermarking configuration for derived runtime `.dnam` marketplace builds
/// (`ref[sec.watermark.reciprocal-gauge]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuyerWatermarkConfig {
    /// Marketplace HKDF/HMAC-SHA256 root secret (`32 B`).
    pub marketplace_secret: [u8; 32],
    /// Creator model identifier string.
    pub model_id: String,
    /// Per-order license identifier string.
    pub license_id: String,
    /// Unique `16 B` buyer identifier (`non-zero` bytes normalized internally).
    pub buyer_id: [u8; 16],
}

impl BuyerWatermarkConfig {
    /// Constructs a [`BuyerWatermarkConfig`] binding `(marketplace_secret, model_id, license_id, buyer_id)`.
    #[must_use]
    pub fn new(
        marketplace_secret: [u8; 32],
        model_id: impl Into<String>,
        license_id: impl Into<String>,
        buyer_id: [u8; 16],
    ) -> Self {
        Self {
            marketplace_secret,
            model_id: model_id.into(),
            license_id: license_id.into(),
            buyer_id,
        }
    }
}

fn non_zero_16(digest: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    if out.iter().all(|&b| b == 0) {
        out[0] = 1;
    }
    out
}

fn build_watermark_scope(
    cfg: &BuyerWatermarkConfig,
) -> (WatermarkSeedScope, [u8; 32], [u8; 16], [u8; 16]) {
    let mut mkt_pk = cfg.marketplace_secret;
    if mkt_pk.iter().all(|&b| b == 0) {
        mkt_pk[0] = 1;
    }
    let model_hash = sha256_digest(cfg.model_id.as_bytes());
    let lic_hash = sha256_digest(cfg.license_id.as_bytes());
    let model_id_16 = non_zero_16(&model_hash);
    let lic_id_16 = non_zero_16(&lic_hash);
    let mut buyer_id_16 = cfg.buyer_id;
    if buyer_id_16.iter().all(|&b| b == 0) {
        buyer_id_16 = non_zero_16(&lic_hash[16..32]);
    }
    let mut manifest_sha = model_hash;
    manifest_sha[0] |= 1;
    let scope = WatermarkSeedScope {
        marketplace_pubkey: mkt_pk,
        watermark_key_id: model_id_16,
        creator_signed_manifest_sha256: manifest_sha,
        model_id: model_id_16,
        watermark_profile_sha256: [0x5A; 32],
        owning_tier: 0,
    };
    (scope, mkt_pk, lic_id_16, buyer_id_16)
}

/// Derives buyer-specific Layer-1 isometric Kautz reciprocal scales (`16` poles, `[0.985, 1.015]`)
/// and a deterministic `4 × 4` orthogonal `SO(4)` `CP1`/`W_ctrl` gauge matrix.
#[must_use]
pub fn derive_buyer_gauge_params(cfg: &BuyerWatermarkConfig) -> (Vec<f32>, [[f32; 4]; 4]) {
    let seed = derive_buyer_phase_seed(cfg);
    let digest = sha256_digest(&seed);
    let mut kappa = Vec::with_capacity(16);
    for idx in 0..16usize {
        let u = f32::from(digest[idx % 32]) / 255.0;
        kappa.push(0.985 + 0.030 * u);
    }

    let angle = |b: u8| -> f32 { (f32::from(b) / 255.0 - 0.5) * 0.8 };
    let mut q = [
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let pairs = [(0usize, 1usize), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
    for (step, &(r0, r1)) in pairs.iter().enumerate() {
        let (s, c) = angle(digest[16 + step]).sin_cos();
        for col in 0..4 {
            let a = q[r0][col];
            let b = q[r1][col];
            q[r0][col] = c * a - s * b;
            q[r1][col] = s * a + c * b;
        }
    }
    (kappa, q)
}

/// Derives the deterministic `32 B` Layer-1/Layer-2 buyer phase & `GRU` sign-flip seed.
#[must_use]
pub fn derive_buyer_phase_seed(cfg: &BuyerWatermarkConfig) -> [u8; 32] {
    let (scope, mkt_key, lic_16, buyer_16) = build_watermark_scope(cfg);
    derive_watermark_seed_hmac_sha256(&mkt_key, &scope, Some((&lic_16, &buyer_16)))
        .unwrap_or_else(|_| sha256_digest(&cfg.buyer_id))
}

/// Generates the Layer-3 Two-Seed Arcsine `TardosCodebook7680` (`c_max = 5`) for `cfg`.
#[must_use]
pub fn derive_buyer_tardos_codebook(
    cfg: &BuyerWatermarkConfig,
) -> dnam_engine::provenance::TardosCodebook7680 {
    let (scope, mkt_key, lic_16, buyer_16) = build_watermark_scope(cfg);
    generate_tardos_codebook_7680(&mkt_key, &scope, &lic_16, &buyer_16, 5)
        .expect("non-zero watermark scope IDs are always valid")
}

/// Extracts all present Stage III Kautz modal pole angles (`theta`, up to `16`) from `arena`.
#[must_use]
pub fn extract_arena_pole_angles_16(arena: &[u8]) -> Vec<f32> {
    let mut angles = Vec::with_capacity(16);
    let tier_caps = [4_608, ARENA_FULL_BYTES, ARENA_PRO_BYTES, ARENA_ULTRA_BYTES];
    for t in 0..4 {
        if arena.len() < tier_caps[t] {
            break;
        }
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[t];
        for blk in 0..blocks {
            let base = STAGE3_KAUTZ_BASES[t] + blk * 128;
            for lane in 0..4 {
                let off = base + (8 + lane) * 4;
                if off + 4 <= arena.len() {
                    let th = f32::from_le_bytes(arena[off..off + 4].try_into().unwrap());
                    angles.push(th);
                }
            }
        }
    }
    angles
}

/// Alias kept for callers upgrading from legacy 34-pole naming (`extract_arena_pole_angles_16`).
#[must_use]
pub fn extract_arena_pole_angles_34(arena: &[u8]) -> Vec<f32> {
    extract_arena_pole_angles_16(arena)
}

fn build_layer3_nonoverlapping_envelopes(arena_len: usize) -> [f32; TARDOS_CASCADE_WEIGHTS] {
    let mut env = [0.0_f32; TARDOS_CASCADE_WEIGHTS];
    // Tier 0 W_ctrl cols 8..16 (LFO/env modulation slots) sit at idx 80..336 where (idx - 80) % 16 >= 8.
    for (idx, val) in env.iter_mut().enumerate().take(336).skip(80) {
        if (idx - 80) % 16 >= 8 {
            *val = 5.0e-5;
        }
    }
    // Tier 1 pure reserve (ARENA1_RESERVE: 9_088..10_240 B) after DIRECT_READOUT + CTRL_PARAMS
    let t1_reserve_start =
        TARDOS_CASCADE_RANGES[1].0 + (ARENA1_DIRECT_READOUT_BYTES + ARENA1_CTRL_PARAMS_BYTES) / 4;
    for slot in env
        .iter_mut()
        .take(TARDOS_CASCADE_RANGES[1].1)
        .skip(t1_reserve_start)
    {
        *slot = 1.5e-4;
    }
    // Tier 2 pure reserve (ARENA2_RESERVE: 15_616..18_432 B) after ARENA2_DIRECT_READOUT
    let t2_reserve_start = TARDOS_CASCADE_RANGES[2].0 + ARENA2_DIRECT_READOUT_BYTES / 4;
    for slot in env
        .iter_mut()
        .take(TARDOS_CASCADE_RANGES[2].1)
        .skip(t2_reserve_start)
    {
        *slot = 1.5e-4;
    }
    // Tier 3 pure reserve (ARENA3_RESERVE: 23_872..28_672 B; skip Role A 4_608..4_624)
    if arena_len >= ARENA_ULTRA_BYTES && ARENA3_RESERVE_OFFSET == TARDOS_CASCADE_RANGES[3].2 {
        for slot in env
            .iter_mut()
            .take(TARDOS_CASCADE_RANGES[3].1)
            .skip(TARDOS_CASCADE_RANGES[3].0 + 16)
        {
            *slot = 1.5e-4;
        }
    }
    env
}

fn rotate_cp1_slot_by_q4(slot: &mut Cp1ContributorSlot, q: &[[f32; 4]; 4]) {
    for lane in 0..usize::from(slot.lanes).min(3) {
        let orig = slot.main_base[lane];
        for r in 0..4 {
            let mut sum = 0.0_f32;
            for c in 0..4 {
                sum += q[r][c] * orig[c];
            }
            slot.main_base[lane][r] = sum;
        }
    }
}

/// Exports a 3-layer buyer-watermarked runtime `.dnam` container (`Layer 1 GRU + Isometric SO(4)`,
/// `Layer 2 Kautz Phase Coding`, `Layer 3 Non-Overlapping Tardos 7680`,
/// `ref[sec.watermark.reciprocal-gauge]`).
pub fn export_runtime_dnam_watermarked(
    proj_bytes: &[u8],
    cfg: &BuyerWatermarkConfig,
    target_tier: Option<Tier>,
) -> Result<Vec<u8>, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let (_, wght_pair_opt) = DnamProjView::from_bytes_with_wght_view(proj_bytes, &crypto, true)?;
    let Some((_, rt_view)) = wght_pair_opt else {
        return Err(DnamError::SchemaValidationError(
            "cannot export watermarked .dnam: project has no WGHT chunk".to_string(),
        ));
    };

    let mut arena = rt_view.arena_storage.as_bytes().to_vec();
    let mut hdr = rt_view.header.clone();
    hdr.metadata.signatures = None;
    let mut cp1_slots = rt_view.cp1_contributors.to_vec();

    let phase_seed = derive_buyer_phase_seed(cfg);
    let gru_signs = derive_gru_sign_gauge_24(&phase_seed);
    apply_gru_sign_gauge_watermark(&mut arena, &gru_signs)?;

    let (kappa, q_buyer_4x4) = derive_buyer_gauge_params(cfg);
    apply_isometric_gauge_watermark(&mut arena, &kappa, &q_buyer_4x4)?;
    for slot in &mut cp1_slots {
        rotate_cp1_slot_by_q4(slot, &q_buyer_4x4);
    }

    apply_phase_coding_watermark(&mut arena, &phase_seed, PHASE_CODING_MAX_RAD)?;

    let codebook = derive_buyer_tardos_codebook(cfg);
    let envelopes = build_layer3_nonoverlapping_envelopes(arena.len());
    apply_tardos_watermark_7680_envelopes(&mut arena, &codebook.chips, &envelopes)?;

    let mut dnam_bytes =
        build_dnam_container_with_cp1(&mut hdr, &arena, &arena, false, &cp1_slots)?;
    if let Some(max_t) = target_tier
        && (max_t as u8) < (rt_view.preamble.highest_tier() as u8)
    {
        dnam_bytes = strip_container_to_tier(&dnam_bytes, max_t)?;
    }
    Ok(dnam_bytes)
}

fn tier_for_arena_len(arena_len: usize) -> Tier {
    if arena_len >= ARENA_ULTRA_BYTES {
        Tier::Ultra
    } else if arena_len >= ARENA_PRO_BYTES {
        Tier::Pro
    } else if arena_len >= ARENA_FULL_BYTES {
        Tier::Full
    } else {
        Tier::Lite
    }
}

fn build_blackbox_probe_excitation(num_samples: usize) -> Vec<f32> {
    let mut x = vec![0.0_f32; num_samples];
    if num_samples > 48 {
        x[32] = 0.30;
    }
    let denom = num_samples.saturating_sub(16).max(1) as f64;
    for (t, dst) in x.iter_mut().enumerate().skip(16) {
        let tf = (t as f64) / 48_000.0;
        let mut s = 0.0_f64;
        for k in 0..64usize {
            let frac = (k as f64) / 63.0;
            let hz = 35.0 * (19_000.0_f64 / 35.0).powf(frac);
            s += (TAU * hz * tf + 0.37 * (k as f64)).sin();
        }
        let u = ((t - 16) as f64) / denom;
        let phase_chirp = TAU * (30.0 * tf + 0.5 * (18_000.0 - 30.0) * tf * u);
        let val = 0.032 * s + 0.12 * phase_chirp.sin();
        *dst = (*dst + (val as f32)).clamp(-0.85, 0.85);
    }
    x
}

fn render_probe_with_arena(
    view: &DnamContainerView<'_>,
    arena: &[u8],
    probe_x: &[f32],
) -> Result<Vec<f32>, DnamError> {
    let mut hdr = view.header.clone();
    hdr.metadata.signatures = None;
    let dnam_tmp =
        build_dnam_container_with_cp1(&mut hdr, arena, arena, false, &view.cp1_contributors)?;
    let tier = tier_for_arena_len(arena.len());
    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes_owned(&dnam_tmp, tier, 48_000)?;
    let mut out = Vec::with_capacity(probe_x.len());
    for &s in probe_x {
        out.push(eng.step_frame_tora(s, s, false)[0]);
    }
    Ok(out)
}

/// Renders a deterministic broadband small-signal probe through a compiled runtime `.dnam`
/// container for black-box Layer-2 pole-angle identification (`r[sec.watermark.reciprocal-gauge]`).
pub fn render_blackbox_probe_response(
    dnam_bytes: &[u8],
    num_samples: usize,
) -> Result<Vec<f32>, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let view = DnamContainerView::from_bytes(dnam_bytes, &crypto)?;
    let tier = tier_for_arena_len(view.arena_storage.as_bytes().len());
    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes_owned(dnam_bytes, tier, 48_000)?;
    let probe_x = build_blackbox_probe_excitation(num_samples);
    let mut out = Vec::with_capacity(probe_x.len());
    for s in probe_x {
        out.push(eng.step_frame_tora(s, s, false)[0]);
    }
    Ok(out)
}

/// Estimates the suspect's `16` Stage III Kautz modal pole angles solely from `suspect_probe_audio`
/// against a linearized Jacobian of `master_dnam` (`r[sec.watermark.reciprocal-gauge]`).
pub fn identify_pole_angles_blackbox_from_audio(
    master_dnam: &[u8],
    suspect_probe_audio: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let master_view = DnamContainerView::from_bytes(master_dnam, &crypto)?;
    let master_arena = master_view.arena_storage.as_bytes();
    let master_angles = extract_arena_pole_angles_16(master_arena);
    let m_cnt = master_angles.len();
    if m_cnt == 0 || suspect_probe_audio.is_empty() {
        return Ok((master_angles.clone(), master_angles));
    }

    let n = suspect_probe_audio.len();
    let probe_x = build_blackbox_probe_excitation(n);
    let y_master = render_probe_with_arena(&master_view, master_arena, &probe_x)?;

    const EPS_RAD: f32 = 1.5e-3;
    const THETA_MIN: f32 = 1.0e-4;
    const THETA_MAX: f32 = core::f32::consts::PI - 1.0e-4;
    let mut cols = Vec::with_capacity(m_cnt);
    for m in 0..m_cnt {
        let th_m = master_angles[m];
        let th_p = (th_m + EPS_RAD).clamp(THETA_MIN, THETA_MAX);
        let th_n = (th_m - EPS_RAD).clamp(THETA_MIN, THETA_MAX);
        let denom = f64::from(th_p - th_n).max(1.0e-6);
        let off = STAGE3_KAUTZ_BASES[m / 4] + (8 + (m % 4)) * 4;
        let mut pert_p = master_arena.to_vec();
        pert_p[off..off + 4].copy_from_slice(&th_p.to_le_bytes());
        let y_p = render_probe_with_arena(&master_view, &pert_p, &probe_x)?;
        let mut pert_n = master_arena.to_vec();
        pert_n[off..off + 4].copy_from_slice(&th_n.to_le_bytes());
        let y_n = render_probe_with_arena(&master_view, &pert_n, &probe_x)?;
        let col: Vec<f64> = y_p
            .iter()
            .zip(&y_n)
            .map(|(&yp, &yn)| (f64::from(yp) - f64::from(yn)) / denom)
            .collect();
        cols.push(col);
    }

    let col_norms: Vec<f64> = cols
        .iter()
        .map(|c| c.iter().map(|&v| v * v).sum::<f64>().sqrt())
        .collect();
    let max_norm = col_norms.iter().copied().fold(0.0_f64, f64::max);
    let active: Vec<bool> = col_norms
        .iter()
        .map(|&cn| cn > 1.0e-8 && cn > 1.0e-5 * max_norm)
        .collect();

    let mut gram = vec![0.0_f64; uplo_u_packed_len(m_cnt)];
    let mut rhs = vec![[0.0_f64; 2]; m_cnt];
    let res: Vec<f64> = suspect_probe_audio
        .iter()
        .zip(&y_master)
        .map(|(&ys, &ym)| f64::from(ys) - f64::from(ym))
        .collect();
    for j in 0..m_cnt {
        if !active[j] {
            gram[uplo_u_index(j, j)] = 1.0;
            continue;
        }
        rhs[j][0] = cols[j].iter().zip(&res).map(|(&s, &r)| s * r).sum::<f64>() / col_norms[j];
        for i in 0..=j {
            if !active[i] {
                continue;
            }
            let dot: f64 = cols[i].iter().zip(&cols[j]).map(|(&a, &b)| a * b).sum();
            gram[uplo_u_index(i, j)] = dot / (col_norms[i] * col_norms[j]);
        }
    }
    let mut est_angles = master_angles.clone();
    if let Ok(sol) = solve_uplo_u_cholesky_2ch(&gram, &rhs, 1.0e-3) {
        for (m, (dst, &[z, _])) in est_angles.iter_mut().zip(&sol).enumerate() {
            if active[m] {
                let delta = z / col_norms[m];
                if delta.abs() >= 1.5e-4 {
                    *dst += (delta as f32).clamp(-0.01, 0.01);
                }
            }
        }
    }
    Ok((master_angles, est_angles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::bundle::build_dnam_proj_container_with_cp1_and_compacted;
    use crate::proj::schema::DnamProjManifest;
    use dnam_engine::provenance::{score_phase_coding_blackbox, score_tardos_watermark_7680};
    use dnam_engine::stages::V3MatryoshkaModel;
    use dnam_engine::stages::weights::pack_v3_matryoshka_to_arena;

    #[test]
    fn test_watermarked_export_tardos_and_blackbox_phase_recovery() {
        let mut mat = V3MatryoshkaModel::identity(true, false);
        for (m, pole) in mat.kautz_poles.iter_mut().enumerate() {
            pole.0 = 0.90;
            pole.1 = 0.12 + 0.14 * (m as f32);
        }
        for t in 0..4 {
            for lane in 0..(t + 1) * 4 {
                mat.readouts[t].kautz_c_l[lane] = (0.18, 0.06);
                mat.readouts[t].kautz_c_r[lane] = (0.18, 0.06);
            }
        }
        let mut arena = vec![0u8; ARENA_ULTRA_BYTES];
        pack_v3_matryoshka_to_arena(&mat, &mut arena);
        let mut manifest = DnamProjManifest::default_reference();
        let proj = build_dnam_proj_container_with_cp1_and_compacted(
            &mut manifest,
            &[],
            &arena,
            &[],
            false,
        )
        .unwrap();

        let cfg = BuyerWatermarkConfig::new([0x42; 32], "test_v3", "lic_01", [0x11; 16]);
        let inn = BuyerWatermarkConfig::new([0x42; 32], "test_v3", "lic_99", [0x99; 16]);
        let wm_dnam = export_runtime_dnam_watermarked(&proj, &cfg, Some(Tier::Ultra)).unwrap();
        let crypto = DefaultCryptoProvider::default();
        let wm_view = DnamContainerView::from_bytes(&wm_dnam, &crypto).unwrap();
        let master_view = DnamProjView::from_bytes_with_wght_view(&proj, &crypto, true)
            .unwrap()
            .1
            .unwrap()
            .1;
        let master_dnam = crate::proj::export_runtime_dnam(&proj, Some(Tier::Ultra)).unwrap();

        let code = derive_buyer_tardos_codebook(&cfg);
        let z_tardos = score_tardos_watermark_7680(
            master_view.arena_storage.as_bytes(),
            wm_view.arena_storage.as_bytes(),
            &code.accusation_weights,
        )
        .unwrap();
        assert!(z_tardos > 15.0, "expected Tardos Z > 15, got {z_tardos}");

        let probe_audio = render_blackbox_probe_response(&wm_dnam, 2_048).unwrap();
        let (m_ang, s_ang) =
            identify_pole_angles_blackbox_from_audio(&master_dnam, &probe_audio).unwrap();
        assert_eq!(m_ang.len(), 16);
        let z_bb =
            score_phase_coding_blackbox(&m_ang, &s_ang, &derive_buyer_phase_seed(&cfg)).unwrap();
        let z_inn =
            score_phase_coding_blackbox(&m_ang, &s_ang, &derive_buyer_phase_seed(&inn)).unwrap();
        assert!(
            z_bb >= 3.9,
            "expected Z >= 3.9 (16-pole max 4.0), got {z_bb}"
        );
        assert!(z_inn.abs() < 3.0, "expected innocent |Z| < 3, got {z_inn}");
    }
}
