#![forbid(unsafe_code)]
//! 3-Layer Buyer Watermarking (`SONA` `0-ULP` Sign-Flip Gauge, `SO(4)` + 32-Channel Kautz
//! Isometric Gauge, Layer-2 Kautz Pole-Angle Phase Coding & Black-Box Phase ID,
//! Layer-3 EMR-Masked Tardos 7680, `solver::stage_d_audit::watermark`).
//!
//! Canonical Reference: `docs/specs/security/watermarking_and_marketplace.md` (`ref[sec.watermark.reciprocal-gauge]`).

use std::f64::consts::TAU;

use dnam_engine::container::{
    DnamContainerView, build_dnam_container_from_dnam_bin, extract_dnam_bin,
    strip_container_to_tier,
};
use dnam_engine::provenance::{
    DefaultCryptoProvider, KAUTZ_TOTAL_CHANNELS, PHASE_CODING_MAX_RAD,
    STAGE3_KAUTZ_BLOCKS_PER_TIER, TARDOS_CASCADE_WEIGHTS, WatermarkSeedScope,
    apply_isometric_gauge_shared_cp1_stream, apply_isometric_gauge_watermark,
    apply_phase_coding_watermark, apply_sona_sign_gauge_watermark,
    apply_tardos_watermark_7680_envelopes, compute_emr_masked_tardos_envelopes,
    derive_sona_sign_gauge_40, derive_watermark_seed_hmac_sha256, generate_tardos_codebook_7680,
    kautz_soa_byte_offset, mask_tardos_stage_latent_active_coords, sha256_digest,
};
use dnam_engine::schema::DnamError;
use dnam_engine::{DnamEngine, Tier};
use dnam_vm::SonaVm;
use dnam_vm::bin::{
    ARENA_TIER_BYTES, Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire,
    Arena3UltraDeltaWire, DNBN_FLAG_WET_POLARITY_INV, DnamBinBuildInput, DnamBinView,
    KautzSoaBlockWire, encode_dnam_bin,
};
use dnam_vm::math::DefaultMathBackend;
use zerocopy::byteorder::little_endian::{F32, F64};
use zerocopy::{FromBytes, IntoBytes};

use crate::proj::{DnamProjView, ProjChunkKind, export_runtime_dnam};
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

#[inline]
fn map_err(err: impl core::fmt::Display) -> DnamError {
    DnamError::SchemaValidationError(format!("{err}"))
}

/// Derives buyer-specific Layer-1 isometric Kautz reciprocal scales (`32` channels, `[0.985, 1.015]`)
/// and a deterministic `4 × 4` orthogonal `SO(4)` `CP1`/`W_ctrl` gauge matrix.
#[must_use]
pub fn derive_buyer_gauge_params(cfg: &BuyerWatermarkConfig) -> (Vec<f32>, [[f32; 4]; 4]) {
    let seed = derive_buyer_phase_seed(cfg);
    let mut kappa_msg = b"DNAM-WM-KAPPA-V1:".to_vec();
    kappa_msg.extend_from_slice(&seed);
    let kappa_dig = sha256_digest(&kappa_msg);

    let mut kappa = Vec::with_capacity(KAUTZ_TOTAL_CHANNELS);
    for idx in 0..KAUTZ_TOTAL_CHANNELS {
        let u = f32::from(kappa_dig[idx % 32]) / 255.0;
        kappa.push(0.985 + 0.030 * u);
    }

    let q_dig = sha256_digest(&seed);
    let angle = |b: u8| -> f32 { (f32::from(b) / 255.0 - 0.5) * 0.8 };
    let mut q = [
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let pairs = [(0usize, 1usize), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
    for (step, &(r0, r1)) in pairs.iter().enumerate() {
        let (s, c) = angle(q_dig[16 + step]).sin_cos();
        for col in 0..4 {
            let a = q[r0][col];
            let b = q[r1][col];
            q[r0][col] = c * a - s * b;
            q[r1][col] = s * a + c * b;
        }
    }
    (kappa, q)
}

/// Derives the deterministic `32 B` Layer-1/Layer-2 buyer phase & `SONA` sign-flip seed.
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

fn assemble_arena_bytes_from_view(bin_view: &DnamBinView<'_>) -> Vec<u8> {
    let mut arena = Vec::with_capacity(bin_view.retained_tier.arena_bytes());
    arena.extend_from_slice(bin_view.arena0.as_bytes());
    if let Some(a1) = bin_view.arena1 {
        arena.extend_from_slice(a1.as_bytes());
    }
    if let Some(a2) = bin_view.arena2 {
        arena.extend_from_slice(a2.as_bytes());
    }
    if let Some(a3) = bin_view.arena3 {
        arena.extend_from_slice(a3.as_bytes());
    }
    arena
}

fn reencode_dbin_with_mutated_arena_and_cp1(
    bin_view: &DnamBinView<'_>,
    arena: &[u8],
    cp1_overflow: &[u8],
) -> Result<Vec<u8>, DnamError> {
    let a0 = Arena0LiteWire::ref_from_prefix(arena)
        .map_err(|_| DnamError::SchemaValidationError("invalid Arena0LiteWire".to_string()))?
        .0;
    let a1 = (arena.len() >= ARENA_TIER_BYTES[1])
        .then(|| {
            Arena1FullDeltaWire::ref_from_prefix(&arena[ARENA_TIER_BYTES[0]..]).map(|(w, _)| w)
        })
        .transpose()
        .map_err(|_| DnamError::SchemaValidationError("invalid Arena1FullDeltaWire".to_string()))?;
    let a2 = (arena.len() >= ARENA_TIER_BYTES[2])
        .then(|| Arena2ProDeltaWire::ref_from_prefix(&arena[ARENA_TIER_BYTES[1]..]).map(|(w, _)| w))
        .transpose()
        .map_err(|_| DnamError::SchemaValidationError("invalid Arena2ProDeltaWire".to_string()))?;
    let a3 = (arena.len() >= ARENA_TIER_BYTES[3])
        .then(|| {
            Arena3UltraDeltaWire::ref_from_prefix(&arena[ARENA_TIER_BYTES[2]..]).map(|(w, _)| w)
        })
        .transpose()
        .map_err(|_| {
            DnamError::SchemaValidationError("invalid Arena3UltraDeltaWire".to_string())
        })?;

    encode_dnam_bin(&DnamBinBuildInput {
        reference_sample_rate_hz: bin_view.header.reference_sample_rate_hz.get(),
        header_flags: bin_view.header.flags.get(),
        ext_ring_pow2_frames: bin_view.runtime_meta.ext_ring_pow2_frames,
        ext_ring_lines: bin_view.runtime_meta.ext_ring_lines,
        ext_ring_format: bin_view.runtime_meta.ext_ring_format,
        declared_max_delay_48k: bin_view.runtime_meta.declared_max_delay_48k.get(),
        dc_blocker_pole_5hz: bin_view.runtime_meta.dc_blocker_pole_5hz.get(),
        tail_time_ms: bin_view.runtime_meta.tail_time_ms.get(),
        ro_pole_48k: bin_view.runtime_meta.ro_pole_48k.get(),
        switch_xfade_samples: bin_view.runtime_meta.switch_xfade_samples.get(),
        content_hash_prefix: bin_view.runtime_meta.content_hash_prefix,
        controls: bin_view.control_descriptors,
        cp1_overflow,
        tsyn: bin_view.tsyn_entries,
        cext: bin_view.cext,
        arena0: a0,
        arena1: a1,
        arena2: a2,
        arena3: a3,
    })
    .map_err(map_err)
}

/// Extracts all present Stage III Kautz modal pole angles (`theta`, up to `32` across
/// `Lite..Ultra`) from raw `Arena 0..3` bytes or a `.dnam-bin` (`"DNBN"`) buffer using `zerocopy`.
#[must_use]
pub fn extract_arena_pole_angles_32(arena_or_dbin: &[u8]) -> Vec<f32> {
    let assembled;
    let arena = if arena_or_dbin.starts_with(b"DNBN") {
        let Ok(view) = DnamBinView::parse(arena_or_dbin) else {
            return Vec::new();
        };
        assembled = assemble_arena_bytes_from_view(&view);
        &assembled[..]
    } else {
        arena_or_dbin
    };

    let mut angles = Vec::with_capacity(KAUTZ_TOTAL_CHANNELS);
    for (t, &cap) in ARENA_TIER_BYTES.iter().enumerate() {
        if arena.len() < cap {
            break;
        }
        let blocks = STAGE3_KAUTZ_BLOCKS_PER_TIER[t];
        for blk in 0..blocks {
            let base = kautz_soa_byte_offset(t, blk);
            if let Ok((kw, _)) = KautzSoaBlockWire::ref_from_prefix(&arena[base..]) {
                for th in &kw.theta {
                    angles.push(th.get());
                }
            }
        }
    }
    angles
}

/// Compat alias delegating to [`extract_arena_pole_angles_32`].
#[must_use]
pub fn extract_arena_pole_angles_16(arena: &[u8]) -> Vec<f32> {
    extract_arena_pole_angles_32(arena)
}

/// Compat alias delegating to [`extract_arena_pole_angles_32`].
#[must_use]
pub fn extract_arena_pole_angles_34(arena: &[u8]) -> Vec<f32> {
    extract_arena_pole_angles_32(arena)
}

fn extract_gram_diag_256(stat_bytes: Option<&[u8]>) -> [f64; 256] {
    let mut diag = [1.0_f64; 256];
    let Some(raw) = stat_bytes else {
        return diag;
    };
    let n_pack = uplo_u_packed_len(256);
    if raw.len() < n_pack * 8 {
        return diag;
    }
    let Ok(words) = <[F64]>::ref_from_bytes(&raw[..n_pack * 8]) else {
        return diag;
    };
    for (i, dst) in diag.iter_mut().enumerate() {
        let v = words[uplo_u_index(i, i)].get();
        if v.is_finite() && v > 0.0 {
            *dst = v;
        }
    }
    diag
}

fn mask_nonoverlapping_layer3_envelopes(envelopes: &mut [f32; TARDOS_CASCADE_WEIGHTS]) {
    const STAR_SILENT_SPANS: [(usize, usize); 9] = [
        (360, 383),     // Tier 0 F32 rho_head_and_ctrl_tail[1..24]
        (391, 463),     // Tier 0 fp16 stage_latent_mods_fp16[8..80]
        (1_095, 1_159), // Tier 1 fp16 zero-filled tail w_ro_and_mods_fp16[352..416]
        (1_175, 1_879), // Tier 2 F32 reserve_core + transport_pro[1]
        (2_543, 3_051), // Tier 2 fp16 zero-filled tail w_ro_and_mods_fp16[452..960]
        (3_067, 3_707), // Tier 3 F32 reserve_core + transport_ultra[1]
        (3_947, 4_139), // Tier 3 F32 w_ctrl_ultra + reserve
        (4_771, 5_619), // Tier 3 fp16 zero-filled tail w_ro_and_mods_fp16[656..1_504]
        (5_619, 5_871), // Tier 3 fp16 stage_latent_ctrl_ultra_fp16
    ];
    for (idx, val) in envelopes.iter_mut().enumerate() {
        let keep = STAR_SILENT_SPANS
            .iter()
            .any(|&(lo, hi)| idx >= lo && idx < hi);
        if !keep {
            *val = 0.0;
        }
    }
}

/// Exports a 3-layer buyer-watermarked runtime `.dnam` container (`Layer 1 SONA Sign-Flip + Isometric SO(4)`,
/// `Layer 2 Kautz Phase Coding`, `Layer 3 EMR-Masked Tardos 7680`,
/// `ref[sec.watermark.reciprocal-gauge]`).
pub fn export_runtime_dnam_watermarked(
    proj_bytes: &[u8],
    cfg: &BuyerWatermarkConfig,
    target_tier: Option<Tier>,
) -> Result<Vec<u8>, DnamError> {
    let crypto = DefaultCryptoProvider::default();
    let proj_view = DnamProjView::from_bytes(proj_bytes, &crypto)?;
    let master_dnam = export_runtime_dnam(proj_bytes, None)?;
    let rt_view = DnamContainerView::from_bytes(&master_dnam, &crypto)?;
    let dbin = extract_dnam_bin(&master_dnam)?;
    let bin_view = DnamBinView::parse(dbin).map_err(map_err)?;

    let mut arena = assemble_arena_bytes_from_view(&bin_view);
    let mut cp1_overflow = bin_view.cp1_overflow.to_vec();

    let phase_seed = derive_buyer_phase_seed(cfg);
    let signs_40 = derive_sona_sign_gauge_40(&phase_seed);
    apply_sona_sign_gauge_watermark(&mut arena, &signs_40)?;

    let (kappa, q_buyer_4x4) = derive_buyer_gauge_params(cfg);
    apply_isometric_gauge_watermark(&mut arena, &kappa, &q_buyer_4x4)?;
    if !cp1_overflow.is_empty() {
        apply_isometric_gauge_shared_cp1_stream(&mut cp1_overflow, &q_buyer_4x4)?;
    }

    apply_phase_coding_watermark(&mut arena, &phase_seed, PHASE_CODING_MAX_RAD)?;

    let mut codebook = derive_buyer_tardos_codebook(cfg);
    let gram_diag_256 = extract_gram_diag_256(proj_view.chunk_slice(ProjChunkKind::Stat, 0));
    let mut envelopes: [f32; TARDOS_CASCADE_WEIGHTS] =
        compute_emr_masked_tardos_envelopes(&gram_diag_256, [1.0e-5, 1.0e-5]);
    mask_nonoverlapping_layer3_envelopes(&mut envelopes);
    let master_arena = assemble_arena_bytes_from_view(&bin_view);
    mask_tardos_stage_latent_active_coords(&master_arena, &mut codebook, Some(&mut envelopes));
    apply_tardos_watermark_7680_envelopes(&mut arena, &codebook.chips, &envelopes)?;

    let wm_dbin = reencode_dbin_with_mutated_arena_and_cp1(&bin_view, &arena, &cp1_overflow)?;
    let mut hdr = rt_view.header.clone();
    hdr.metadata.signatures = None;
    let mut out_dnam = build_dnam_container_from_dnam_bin(&mut hdr, &wm_dbin)?;
    if let Some(max_t) = target_tier
        && max_t.index() < bin_view.retained_tier.index()
    {
        out_dnam = strip_container_to_tier(&out_dnam, max_t)?;
    }
    Ok(out_dnam)
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

fn render_probe_with_dbin(
    base_view: &DnamBinView<'_>,
    arena: &[u8],
    probe_x: &[f32],
    host_pol_sign: f32,
) -> Result<Vec<f32>, DnamError> {
    let dbin = reencode_dbin_with_mutated_arena_and_cp1(base_view, arena, base_view.cp1_overflow)?;
    let tier = base_view.retained_tier;
    let mut vm = SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(&dbin, tier, 48_000.0)
        .map_err(map_err)?;
    let mut out = Vec::with_capacity(probe_x.len());
    for &s in probe_x {
        out.push(host_pol_sign * vm.step_frame(s, s).map_err(map_err)?.0);
    }
    Ok(out)
}

/// Renders a deterministic broadband small-signal probe through a compiled runtime `.dnam`
/// container for black-box Layer-2 pole-angle identification (`ref[sec.watermark.reciprocal-gauge]`).
pub fn render_blackbox_probe_response(
    dnam_bytes: &[u8],
    num_samples: usize,
) -> Result<Vec<f32>, DnamError> {
    let dbin = extract_dnam_bin(dnam_bytes)?;
    let bin_view = DnamBinView::parse(dbin).map_err(map_err)?;
    let tier = bin_view.retained_tier;
    let mut eng = DnamEngine::<DefaultMathBackend>::from_bytes_owned(dnam_bytes, tier, 48_000)?;
    let probe_x = build_blackbox_probe_excitation(num_samples);
    let mut out = Vec::with_capacity(probe_x.len());
    for s in probe_x {
        out.push(eng.step_frame_sona(s, s, false)[0]);
    }
    Ok(out)
}

fn collect_kautz_slot_coordinates(tier: Tier) -> Vec<(usize, usize, usize)> {
    let mut coords = Vec::with_capacity(KAUTZ_TOTAL_CHANNELS);
    for (t, &blocks) in STAGE3_KAUTZ_BLOCKS_PER_TIER
        .iter()
        .enumerate()
        .take(tier.index() + 1)
    {
        for blk in 0..blocks {
            for lane in 0..4 {
                coords.push((t, blk, lane));
            }
        }
    }
    coords
}

#[inline]
const fn floor_f64_const(x: f64) -> f64 {
    let t = x as i64;
    if (t as f64) > x {
        (t - 1) as f64
    } else {
        t as f64
    }
}

#[inline]
fn snap_pole_angle_guard_band(theta: f32) -> f32 {
    const BIN_WIDTH: f64 = 0.01;
    let cell = f64::from(theta) / BIN_WIDTH;
    let base = floor_f64_const(cell);
    let frac = cell - base;
    let snapped = if frac < 0.2 {
        ((base + 0.2) * BIN_WIDTH) as f32
    } else if frac > 0.8 {
        ((base + 0.8) * BIN_WIDTH) as f32
    } else {
        theta
    };
    snapped.clamp(1.0e-3, core::f32::consts::PI - 1.0e-3)
}

/// Estimates all present Stage III Kautz modal pole angles (`<= 32`) solely from
/// `suspect_probe_audio` against a guard-band carrier linearized Jacobian of `master_dnam`
/// (`ref[sec.watermark.reciprocal-gauge]`).
pub fn identify_pole_angles_blackbox_from_audio(
    master_dnam: &[u8],
    suspect_probe_audio: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), DnamError> {
    let master_view =
        DnamContainerView::from_bytes(master_dnam, &DefaultCryptoProvider::default())?;
    let master_dbin = extract_dnam_bin(master_dnam)?;
    let bin_view = DnamBinView::parse(master_dbin).map_err(map_err)?;
    let vm_wet_pol_inv = (bin_view.header.flags.get() & DNBN_FLAG_WET_POLARITY_INV) != 0;
    let host_pol_sign = if !vm_wet_pol_inv
        && master_view
            .header
            .io_config
            .capabilities
            .wet_polarity_inverted
    {
        -1.0_f32
    } else {
        1.0_f32
    };
    let master_arena = assemble_arena_bytes_from_view(&bin_view);
    let master_angles = extract_arena_pole_angles_32(&master_arena);
    let m_cnt = master_angles.len();
    if m_cnt == 0 || suspect_probe_audio.is_empty() {
        return Ok((master_angles.clone(), master_angles));
    }

    const THETA_MIN: f32 = 1.0e-3;
    const THETA_MAX: f32 = core::f32::consts::PI - 1.0e-3;
    let n = suspect_probe_audio.len();
    let probe_x = build_blackbox_probe_excitation(n);
    let coords = collect_kautz_slot_coordinates(bin_view.retained_tier);

    let mut carrier_angles = master_angles.clone();
    let mut carrier_arena = master_arena;
    for (m, &(t, blk, lane)) in coords.iter().enumerate().take(m_cnt) {
        let th = master_angles[m];
        if (THETA_MIN..=THETA_MAX).contains(&th) {
            let snapped = snap_pole_angle_guard_band(th);
            carrier_angles[m] = snapped;
            let base = kautz_soa_byte_offset(t, blk);
            if let Ok((kw, _)) = KautzSoaBlockWire::mut_from_prefix(&mut carrier_arena[base..]) {
                kw.theta[lane] = F32::new(snapped);
            }
        }
    }
    let y_carrier = render_probe_with_dbin(&bin_view, &carrier_arena, &probe_x, host_pol_sign)?;

    const EPS_RAD: f32 = PHASE_CODING_MAX_RAD;
    let mut cols = Vec::with_capacity(m_cnt);
    for (m, &(t, blk, lane)) in coords.iter().enumerate().take(m_cnt) {
        if !(THETA_MIN..=THETA_MAX).contains(&master_angles[m]) {
            cols.push(vec![0.0_f64; n]);
            continue;
        }
        let th_c = carrier_angles[m];
        let th_p = (th_c + EPS_RAD).clamp(THETA_MIN, THETA_MAX);
        let th_n = (th_c - EPS_RAD).clamp(THETA_MIN, THETA_MAX);
        let denom = f64::from(th_p - th_n).max(1.0e-6);
        let base = kautz_soa_byte_offset(t, blk);

        let mut pert_p = carrier_arena.clone();
        if let Ok((kw, _)) = KautzSoaBlockWire::mut_from_prefix(&mut pert_p[base..]) {
            kw.theta[lane] = F32::new(th_p);
        }
        let y_p = render_probe_with_dbin(&bin_view, &pert_p, &probe_x, host_pol_sign)?;

        let mut pert_n = carrier_arena.clone();
        if let Ok((kw, _)) = KautzSoaBlockWire::mut_from_prefix(&mut pert_n[base..]) {
            kw.theta[lane] = F32::new(th_n);
        }
        let y_n = render_probe_with_dbin(&bin_view, &pert_n, &probe_x, host_pol_sign)?;

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
    let res: Vec<f64> = suspect_probe_audio
        .iter()
        .zip(&y_carrier)
        .map(|(&ys, &yc)| f64::from(ys) - f64::from(yc))
        .collect();
    let mut active: Vec<bool> = col_norms
        .iter()
        .map(|&cn| cn > 1.0e-8 && cn > 1.0e-4 * max_norm)
        .collect();
    if active.iter().filter(|&&a| a).count() < 4 {
        active = col_norms.iter().map(|&cn| cn > 1.0e-8).collect();
    }

    let mut gram = vec![0.0_f64; uplo_u_packed_len(m_cnt)];
    let mut rhs = vec![[0.0_f64; 2]; m_cnt];
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
    const DELTA_MIN_RAD: f64 = 0.15 * (PHASE_CODING_MAX_RAD as f64);
    const DELTA_MAX_RAD: f64 = 4.0 * (PHASE_CODING_MAX_RAD as f64);
    let mut est_angles = master_angles.clone();
    if let Ok(sol) = solve_uplo_u_cholesky_2ch(&gram, &rhs, 5.0e-3) {
        for (m, (dst, &[z, _])) in est_angles.iter_mut().zip(&sol).enumerate() {
            if active[m] {
                let delta = z / col_norms[m];
                if delta.abs() >= DELTA_MIN_RAD && delta.abs() <= DELTA_MAX_RAD {
                    let mut est = (carrier_angles[m] + (delta as f32)).clamp(THETA_MIN, THETA_MAX);
                    if f64::from((est - master_angles[m]).abs()) <= 1.5e-6 {
                        let nudge = 2.0e-6_f32 * (delta.signum() as f32);
                        est = (est + nudge).clamp(THETA_MIN, THETA_MAX);
                    }
                    *dst = est;
                }
            }
        }
    }
    Ok((master_angles, est_angles))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::bundle::build_dnam_proj_container_with_dnam_bin;
    use crate::proj::schema::DnamProjManifest;
    use dnam_engine::provenance::{score_phase_coding_blackbox, score_tardos_watermark_7680};
    use dnam_train::{Pass5EmitConfig, SonaModelF64, encode_model_to_dnam_bin};

    #[test]
    fn test_watermarked_export_tardos_and_blackbox_phase_recovery() {
        let mut model = SonaModelF64::new(48_000.0);
        model.pre_conv_u_gain = [1.0, 1.0];
        model.active_kautz_pairs = 32;
        for m in 0..32 {
            model.kautz[m].r_low = 0.90;
            model.kautz[m].theta = 0.085 + 0.090 * (m as f64);
            model.kautz[m].b_gamma = 1.0;
            model.kautz[m].c_re_l = 0.18;
            model.kautz[m].c_im_l = 0.06;
            model.kautz[m].c_re_r = 0.18;
            model.kautz[m].c_im_r = 0.06;
        }
        let dnam_bin =
            encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).expect("encode");
        let mut manifest = DnamProjManifest::default_reference();
        let proj = build_dnam_proj_container_with_dnam_bin(&mut manifest, &[], &dnam_bin).unwrap();

        let cfg = BuyerWatermarkConfig::new([0x42; 32], "test_v3", "lic_01", [0x11; 16]);
        let inn = BuyerWatermarkConfig::new([0x42; 32], "test_v3", "lic_99", [0x99; 16]);
        let wm_dnam = export_runtime_dnam_watermarked(&proj, &cfg, Some(Tier::Ultra)).unwrap();
        let master_dnam = export_runtime_dnam(&proj, Some(Tier::Ultra)).unwrap();

        let wm_dbin = extract_dnam_bin(&wm_dnam).unwrap();
        let master_dbin = extract_dnam_bin(&master_dnam).unwrap();
        let wm_arena = assemble_arena_bytes_from_view(&DnamBinView::parse(wm_dbin).unwrap());
        let master_arena =
            assemble_arena_bytes_from_view(&DnamBinView::parse(master_dbin).unwrap());

        let code = derive_buyer_tardos_codebook(&cfg);
        let z_tardos =
            score_tardos_watermark_7680(&master_arena, &wm_arena, &code.accusation_weights)
                .unwrap();
        assert!(z_tardos > 15.0, "expected Tardos Z > 15, got {z_tardos}");

        let probe_audio = render_blackbox_probe_response(&wm_dnam, 2_048).unwrap();
        let (m_ang, s_ang) =
            identify_pole_angles_blackbox_from_audio(&master_dnam, &probe_audio).unwrap();
        assert_eq!(m_ang.len(), KAUTZ_TOTAL_CHANNELS);
        let z_bb =
            score_phase_coding_blackbox(&m_ang, &s_ang, &derive_buyer_phase_seed(&cfg)).unwrap();
        let z_inn =
            score_phase_coding_blackbox(&m_ang, &s_ang, &derive_buyer_phase_seed(&inn)).unwrap();
        assert!(
            z_bb >= 4.5,
            "expected Z >= 4.5 (32-pole max ~5.66), got {z_bb}"
        );
        assert!(z_inn.abs() < 3.0, "expected innocent |Z| < 3, got {z_inn}");

        let mut noisy_probe = probe_audio;
        for (i, y) in noisy_probe.iter_mut().enumerate() {
            *y += 0.0025_f32 * ((i as f32 * 0.713).sin() + 0.5 * (i as f32 * 1.917).cos());
        }
        let (m_noisy, s_noisy) =
            identify_pole_angles_blackbox_from_audio(&master_dnam, &noisy_probe).unwrap();
        let z_bb_noisy =
            score_phase_coding_blackbox(&m_noisy, &s_noisy, &derive_buyer_phase_seed(&cfg))
                .unwrap();
        let z_inn_noisy =
            score_phase_coding_blackbox(&m_noisy, &s_noisy, &derive_buyer_phase_seed(&inn))
                .unwrap();
        assert!(
            z_bb_noisy >= 3.85,
            "expected noisy -45 dBFS Z >= 3.85, got {z_bb_noisy}"
        );
        assert!(
            z_inn_noisy.abs() < 3.0,
            "expected noisy innocent |Z| < 3, got {z_inn_noisy}"
        );
    }

    #[test]
    fn test_watermarked_export_masks_active_stage_latent_knobs_and_preserves_cext() {
        use dnam_kernels::f32_to_fp16;
        use dnam_vm::bin::CextDispatchWire;
        use zerocopy::byteorder::little_endian::U16;

        let mut model = SonaModelF64::new(48_000.0);
        model.pre_conv_u_gain = [1.0, 1.0];
        model.active_kautz_pairs = 32;
        for m in 0..32 {
            model.kautz[m].r_low = 0.90;
            model.kautz[m].theta = 0.085 + 0.090 * (m as f64);
            model.kautz[m].b_gamma = 1.0;
            model.kautz[m].c_re_l = 0.18;
            model.kautz[m].c_re_r = 0.18;
        }
        let base = encode_model_to_dnam_bin(&model, Pass5EmitConfig::default()).expect("encode");
        let bv = DnamBinView::parse(&base).unwrap();
        let mut a0 = *bv.arena0;
        let mut a3 = *bv.arena3.unwrap();
        let val = U16::new(f32_to_fp16(0.125));
        a0.linear_readout_lite.stage_latent_mods_fp16[0..8].fill(val);
        a0.linear_readout_lite.stage_latent_mods_fp16[48..80].fill(val);
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[0..160].fill(val);
        let mut cext = CextDispatchWire {
            w_ctrl_ext: [[F32::new(0.0); 16]; 16],
            w_ctrl_ultra_ext: [[F32::new(0.0); 4]; 16],
        };
        cext.w_ctrl_ext[0][4] = F32::new(0.25);
        cext.w_ctrl_ultra_ext[1][2] = F32::new(-0.125);
        let dnam_bin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: bv.header.reference_sample_rate_hz.get(),
            header_flags: bv.header.flags.get(),
            ext_ring_pow2_frames: bv.runtime_meta.ext_ring_pow2_frames,
            ext_ring_lines: bv.runtime_meta.ext_ring_lines,
            ext_ring_format: bv.runtime_meta.ext_ring_format,
            declared_max_delay_48k: bv.runtime_meta.declared_max_delay_48k.get(),
            dc_blocker_pole_5hz: bv.runtime_meta.dc_blocker_pole_5hz.get(),
            tail_time_ms: bv.runtime_meta.tail_time_ms.get(),
            ro_pole_48k: bv.runtime_meta.ro_pole_48k.get(),
            switch_xfade_samples: bv.runtime_meta.switch_xfade_samples.get(),
            content_hash_prefix: bv.runtime_meta.content_hash_prefix,
            controls: bv.control_descriptors,
            cp1_overflow: bv.cp1_overflow,
            tsyn: bv.tsyn_entries,
            cext: Some(&cext),
            arena0: &a0,
            arena1: bv.arena1,
            arena2: bv.arena2,
            arena3: Some(&a3),
        })
        .unwrap();

        let mut manifest = DnamProjManifest::default_reference();
        let proj = build_dnam_proj_container_with_dnam_bin(&mut manifest, &[], &dnam_bin).unwrap();
        let cfg = BuyerWatermarkConfig::new([0x42; 32], "test_v3", "lic_02", [0x22; 16]);
        let wm_dnam = export_runtime_dnam_watermarked(&proj, &cfg, Some(Tier::Ultra)).unwrap();
        let master_dnam = export_runtime_dnam(&proj, Some(Tier::Ultra)).unwrap();
        let mv = DnamBinView::parse(extract_dnam_bin(&master_dnam).unwrap()).unwrap();
        let wv = DnamBinView::parse(extract_dnam_bin(&wm_dnam).unwrap()).unwrap();
        assert_eq!(
            wv.arena0.linear_readout_lite.stage_latent_mods_fp16[0..8],
            mv.arena0.linear_readout_lite.stage_latent_mods_fp16[0..8]
        );
        assert_eq!(
            wv.arena0.linear_readout_lite.stage_latent_mods_fp16[48..80],
            mv.arena0.linear_readout_lite.stage_latent_mods_fp16[48..80]
        );
        assert_eq!(
            wv.arena3
                .unwrap()
                .control_bus_ultra_delta
                .stage_latent_ctrl_ultra_fp16[0..160],
            mv.arena3
                .unwrap()
                .control_bus_ultra_delta
                .stage_latent_ctrl_ultra_fp16[0..160]
        );
        assert_eq!(wv.cext.map(|c| c.as_bytes()), Some(cext.as_bytes()));
        let code = derive_buyer_tardos_codebook(&cfg);
        let z = score_tardos_watermark_7680(
            &assemble_arena_bytes_from_view(&mv),
            &assemble_arena_bytes_from_view(&wv),
            &code.accusation_weights,
        )
        .unwrap();
        assert!(z > 15.0, "expected Tardos Z > 15, got {z}");
    }

    #[test]
    fn test_watermark_helpers_zero_ids_short_inputs_and_tier_strip() {
        assert_eq!(non_zero_16(&[0; 16])[0], 1);
        let zero_cfg = BuyerWatermarkConfig::new([0; 32], "m_z", "l_z", [0; 16]);
        let (kappa, q) = derive_buyer_gauge_params(&zero_cfg);
        assert_eq!(kappa.len(), 32);
        assert!(q[0].iter().map(|v| v * v).sum::<f32>() > 0.99);
        assert!(
            derive_buyer_tardos_codebook(&zero_cfg)
                .accusation_weights
                .len()
                == 7680
        );

        assert!(extract_arena_pole_angles_16(&[0; 8]).is_empty());
        assert!(extract_arena_pole_angles_34(b"DNBNbad").is_empty());
        assert_eq!(extract_gram_diag_256(None)[0], 1.0);
        assert_eq!(extract_gram_diag_256(Some(&[0; 8]))[0], 1.0);
        let packed = vec![0_u8; uplo_u_packed_len(256) * 8];
        assert_eq!(extract_gram_diag_256(Some(&packed))[0], 1.0);

        assert!((snap_pole_angle_guard_band(0.021) - 0.022).abs() < 1e-4);
        assert!((snap_pole_angle_guard_band(0.029) - 0.028).abs() < 1e-4);
        assert!((snap_pole_angle_guard_band(0.025) - 0.025).abs() < 1e-6);

        let dnam_bin =
            encode_model_to_dnam_bin(&SonaModelF64::new(48_000.0), Pass5EmitConfig::default())
                .unwrap();
        let mut manifest = DnamProjManifest::default_reference();
        let proj = build_dnam_proj_container_with_dnam_bin(&mut manifest, &[], &dnam_bin).unwrap();
        let lite_wm = export_runtime_dnam_watermarked(&proj, &zero_cfg, Some(Tier::Lite)).unwrap();
        let view =
            DnamContainerView::from_bytes(&lite_wm, &DefaultCryptoProvider::default()).unwrap();
        assert_eq!(view.preamble.highest_tier(), Tier::Lite);
        let (orig, est) = identify_pole_angles_blackbox_from_audio(&lite_wm, &[]).unwrap();
        assert_eq!(orig, est);
    }
}
