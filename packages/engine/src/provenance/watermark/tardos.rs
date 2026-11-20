//! Layer 3 Two-Seed Arcsine Tardos Traitor-Tracing Codebook (`TardosCodebook7680`),
//! Coordinate-Space `EMR`-Masked Envelopes, `fp16` `W_readout` Watermarking with
//! `2 * fp16_ulp(w)` survival floor, and `SO(4)` `Q_buyer^-1` Unrotation (`watermark/tardos.rs`).
//!
//! Canonical Reference: `docs/specs/security/watermarking_and_marketplace.md` (`ref[sec.watermark.tardos-code]`).

pub(crate) mod layout_7680;

use alloc::vec::Vec;
use core::ops::Deref;

use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use zerocopy::byteorder::little_endian::{F32, U16, U32};
use zerocopy::{FromBytes, IntoBytes};

use crate::graph::weights::{ARENA_LITE_BYTES, Arena0ControlBusWire};
use crate::provenance::TARDOS_CASCADE_WEIGHTS;
use crate::provenance::chacha20::chacha20_block_64b;
use crate::provenance::security::{WatermarkSeedScope, derive_watermark_seed_hmac_sha256};
use crate::provenance::sha256::sha256_digest;
use crate::schema::{DnamError, cold_sqrt_f64};

use super::max_tiers_for_arena_len;
pub use layout_7680::{
    TARDOS_CASCADE_RANGES, TARDOS_ULTRA_RESERVE_RANGE, TARDOS_W_READOUT_FP16_RANGES,
};
pub(crate) use layout_7680::{TARDOS_F32_SPANS, TARDOS_FP16_SPANS, TARDOS_TIER_INDEX_BOUNDS};

const FP16_MAX_FINITE: f32 = 65_504.0;

#[inline]
fn read_f32_wire(arena: &[u8], byte_offset: usize) -> f32 {
    arena
        .get(byte_offset..)
        .and_then(|tail| F32::read_from_prefix(tail).ok())
        .map_or(0.0, |(v, _)| v.get())
}

#[inline]
fn write_f32_wire(arena: &mut [u8], byte_offset: usize, val: f32) {
    if let Some(dst) = arena.get_mut(byte_offset..) {
        let _ = F32::new(val).write_to_prefix(dst);
    }
}

#[inline]
pub(crate) fn read_u16_le(arena: &[u8], byte_offset: usize) -> u16 {
    arena
        .get(byte_offset..)
        .and_then(|tail| U16::read_from_prefix(tail).ok())
        .map_or(0, |(v, _)| v.get())
}

#[inline]
pub(crate) fn write_u16_le(arena: &mut [u8], byte_offset: usize, val: u16) {
    if let Some(dst) = arena.get_mut(byte_offset..) {
        let _ = U16::new(val).write_to_prefix(dst);
    }
}

/// Computes `binary16` (`fp16`) unit-in-last-place at `w`: `2^(max(-24, floor(log2|w|) - 10))`.
#[inline]
pub(crate) fn fp16_ulp(w: f32) -> f32 {
    let aw = w.abs();
    if !aw.is_finite() || aw <= 6.103_515_6e-5 {
        5.960_464_5e-8 // 2^-24 subnormal fp16 ULP
    } else {
        let exp = (((aw.to_bits() >> 23) & 0xFF) as i32) - 127;
        let ulp_exp = (exp - 10).max(-24);
        f32::from_bits(((ulp_exp + 127) as u32) << 23)
    }
}

/// Returns `true` if `global_idx` (`0..7,680`) belongs to the `64` Role A coordinates
/// (`0..16`, `655..671`, `1,159..1,175`, `3,051..3,067`).
#[inline]
pub const fn is_role_a_cascade_index(global_idx: usize) -> bool {
    global_idx < 16
        || (global_idx >= 655 && global_idx < 671)
        || (global_idx >= 1_159 && global_idx < 1_175)
        || (global_idx >= 3_051 && global_idx < 3_067)
}

/// Visits `(global_idx, byte_offset)` of all `2,447` non-backbone `F32` Role B Tardos weights in `arena_len`.
pub fn for_each_tardos_f32_offset_up_to<F: FnMut(usize, usize)>(
    arena_len: usize,
    mut f: F,
) -> usize {
    let max_tiers = max_tiers_for_arena_len(arena_len);
    let mut count = 0_usize;
    for span in TARDOS_F32_SPANS {
        if (span.tier as usize) >= max_tiers {
            continue;
        }
        for w in 0..span.len() {
            f(span.start_idx + w, span.byte_base + w * 4);
            count += 1;
        }
    }
    count
}

/// Visits `(global_idx, byte_offset)` of all `3,364` linear `fp16` (`U16`) Role B carriers in `arena_len`.
pub fn for_each_tardos_fp16_offset_up_to<F: FnMut(usize, usize)>(
    arena_len: usize,
    mut f: F,
) -> usize {
    let max_tiers = max_tiers_for_arena_len(arena_len);
    let mut count = 0_usize;
    for span in TARDOS_FP16_SPANS {
        if (span.tier as usize) >= max_tiers {
            continue;
        }
        for w in 0..span.len() {
            f(span.start_idx + w, span.byte_base + w * 2);
            count += 1;
        }
    }
    count
}

/// Maps a Role B global coordinate `global_idx` (`0..5,875`) to `(tier, slot_32, lane_4)`
/// (`None` for the `64` Role A coordinates and `5,875..7,680` unassigned tail).
#[inline]
pub const fn role_b_head_slot_for_index(global_idx: usize) -> Option<(usize, usize, usize)> {
    if global_idx >= 5_875 || is_role_a_cascade_index(global_idx) {
        return None;
    }
    let mut i = 0;
    while i < 4 {
        let (start, end) = TARDOS_TIER_INDEX_BOUNDS[i];
        if global_idx >= start && global_idx < end {
            let role_b_word = (global_idx - start) - 16;
            let vec4_idx = role_b_word / 4;
            let lane = role_b_word % 4;
            let slot_32 = vec4_idx % 32;
            return Some((i, slot_32, lane));
        }
        i += 1;
    }
    None
}

/// Two-Seed Arcsine Tardos Traitor-Tracing Codebook (`7,680` coordinate view across `Arena 0..3`:
/// `64` Role A zeroed + `5,811` Role B `F32`/`fp16` carriers + `1,805` unassigned `0.0` tail,
/// `ref[sec.watermark.tardos-code]`).
#[derive(Clone, Debug, PartialEq)]
pub struct TardosCodebook7680 {
    pub chips: [f32; TARDOS_CASCADE_WEIGHTS],
    pub accusation_weights: [f32; TARDOS_CASCADE_WEIGHTS],
}

impl Deref for TardosCodebook7680 {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.accusation_weights
    }
}

impl AsRef<[f32]> for TardosCodebook7680 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        &self.accusation_weights
    }
}

#[inline]
fn cold_sin_half_pi_f64(phi: f64) -> f64 {
    let x = phi.clamp(0.0, core::f64::consts::FRAC_PI_2);
    let x2 = x * x;
    x * (1.0
        - x2 * (1.0 / 6.0
            - x2 * (1.0 / 120.0
                - x2 * (1.0 / 5_040.0 - x2 * (1.0 / 362_880.0 - x2 / 39_916_800.0)))))
}

#[inline]
fn cold_asin_small_f64(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0 + x2 * (1.0 / 6.0 + x2 * (3.0 / 40.0 + x2 * (5.0 / 112.0))))
}

#[inline]
fn draw_uniform_01_from_chacha_stream(
    seed: &[u8; 32],
    stream_buf: &mut [u8; 64],
    word_cursor: &mut usize,
    blk_counter: &mut u32,
) -> f64 {
    if *word_cursor >= 16 {
        *stream_buf = chacha20_block_64b(seed, &[0u8; 12], *blk_counter);
        *blk_counter = blk_counter.wrapping_add(1);
        *word_cursor = 0;
    }
    let off = *word_cursor * 4;
    *word_cursor += 1;
    let u = U32::read_from_prefix(&stream_buf[off..]).map_or(0, |(v, _)| v.get());
    ((u as f64) + 0.5) * (1.0 / 4_294_967_296.0)
}

#[allow(clippy::too_many_arguments)]
fn populate_tardos_tier(
    chips: &mut [f32; TARDOS_CASCADE_WEIGHTS],
    accusation_weights: &mut [f32; TARDOS_CASCADE_WEIGHTS],
    start: usize,
    end: usize,
    shared_seed: &[u8; 32],
    buyer_seed: &[u8; 32],
    tau: f64,
    tau_prime: f64,
    span: f64,
) {
    let mut shared_buf = [0u8; 64];
    let mut buyer_buf = [0u8; 64];
    let mut shared_cursor = 16_usize;
    let mut buyer_cursor = 16_usize;
    let mut shared_blk = 0_u32;
    let mut buyer_blk = 0_u32;

    for idx in start..end {
        if (idx - start) < 16 {
            chips[idx] = 0.0;
            accusation_weights[idx] = 0.0;
            continue;
        }
        let u_i = draw_uniform_01_from_chacha_stream(
            shared_seed,
            &mut shared_buf,
            &mut shared_cursor,
            &mut shared_blk,
        );
        let v_ji = draw_uniform_01_from_chacha_stream(
            buyer_seed,
            &mut buyer_buf,
            &mut buyer_cursor,
            &mut buyer_blk,
        );
        let phi = tau_prime + span * u_i;
        let s = cold_sin_half_pi_f64(phi);
        let p_i = (s * s).clamp(tau, 1.0 - tau);

        if v_ji < p_i {
            chips[idx] = 1.0;
            accusation_weights[idx] = cold_sqrt_f64((1.0 - p_i) / p_i) as f32;
        } else {
            chips[idx] = -1.0;
            accusation_weights[idx] = -(cold_sqrt_f64(p_i / (1.0 - p_i)) as f32);
        }
    }
}

/// Generates a Two-Seed Arcsine Tardos Codebook (`TardosCodebook7680`) across `Arena 0..3`
/// using per-tier `(shared_seed[t], buyer_seed[t])` (`owning_tier = t as u8`) derived via
/// [`derive_watermark_seed_hmac_sha256`] (`ref[sec.watermark.seed-derivation]`, `ref[sec.watermark.tardos-code]`).
pub fn generate_tardos_codebook_7680(
    wm_key: &[u8; 32],
    scope_base: &WatermarkSeedScope,
    license_id: &[u8; 16],
    buyer_id: &[u8; 16],
    c_max: usize,
) -> Result<TardosCodebook7680, DnamError> {
    let c_eff = c_max.max(1) as f64;
    let tau = (1.0 / (300.0 * c_eff * c_eff)).clamp(1e-6, 0.25);
    let tau_prime = cold_asin_small_f64(cold_sqrt_f64(tau));
    let span = core::f64::consts::FRAC_PI_2 - 2.0 * tau_prime;

    let mut chips = [0.0_f32; TARDOS_CASCADE_WEIGHTS];
    let mut accusation_weights = [0.0_f32; TARDOS_CASCADE_WEIGHTS];

    for (tier, &(start, end)) in TARDOS_TIER_INDEX_BOUNDS.iter().enumerate() {
        let tier_scope = WatermarkSeedScope {
            owning_tier: tier as u8,
            ..*scope_base
        };
        let shared_seed = derive_watermark_seed_hmac_sha256(wm_key, &tier_scope, None)?;
        let buyer_seed =
            derive_watermark_seed_hmac_sha256(wm_key, &tier_scope, Some((license_id, buyer_id)))?;
        populate_tardos_tier(
            &mut chips,
            &mut accusation_weights,
            start,
            end,
            &shared_seed,
            &buyer_seed,
            tau,
            tau_prime,
            span,
        );
    }

    Ok(TardosCodebook7680 {
        chips,
        accusation_weights,
    })
}

/// Deterministically generates a [`TardosCodebook7680`] (`7,680`-coordinate view) from
/// `(marketplace_secret, model_id, license_id)` with default collusion resistance $c_{\max} = 4$.
pub fn generate_tardos_code_7680(
    marketplace_secret: &[u8; 32],
    model_id: &str,
    license_id: &str,
) -> TardosCodebook7680 {
    let model_digest = sha256_digest(model_id.as_bytes());
    let lic_digest = sha256_digest(license_id.as_bytes());
    let mut model_id_16 = [0u8; 16];
    let mut lic_id_16 = [0u8; 16];
    let mut buyer_id_16 = [0u8; 16];
    model_id_16.copy_from_slice(&model_digest[0..16]);
    lic_id_16.copy_from_slice(&lic_digest[0..16]);
    buyer_id_16.copy_from_slice(&lic_digest[16..32]);
    model_id_16[0] |= 1;
    lic_id_16[0] |= 1;
    buyer_id_16[0] |= 1;

    let mut manifest_sha = model_digest;
    manifest_sha[0] |= 1;
    let mut mkt_pk = *marketplace_secret;
    if mkt_pk.iter().all(|&b| b == 0) {
        mkt_pk[0] = 1;
    }

    let scope_base = WatermarkSeedScope {
        marketplace_pubkey: mkt_pk,
        watermark_key_id: model_id_16,
        creator_signed_manifest_sha256: manifest_sha,
        model_id: model_id_16,
        watermark_profile_sha256: [0x55; 32],
        owning_tier: 0,
    };

    generate_tardos_codebook_7680(marketplace_secret, &scope_base, &lic_id_16, &buyer_id_16, 4)
        .expect("non-zero scope IDs are always valid")
}

/// Computes strictly positive coordinate-space `EMR`-masked perturbation envelopes
/// $\epsilon_i \in [5 \times 10^{-5},\, 2 \times 10^{-4}]$ across all `5,811` Role B coordinates
/// (`0.0` on Role A and unassigned tail coordinates, `ref[sec.watermark.tardos-code]`).
pub fn compute_emr_masked_tardos_envelopes(
    gram_diag_256: &[f64; 256],
    epsilon_mask_lr: [f64; 2],
) -> [f32; TARDOS_CASCADE_WEIGHTS] {
    let mut envelopes = [0.0_f32; TARDOS_CASCADE_WEIGHTS];
    let min_mask = epsilon_mask_lr[0].min(epsilon_mask_lr[1]).max(0.0);
    for idx in 0..TARDOS_CASCADE_WEIGHTS {
        if let Some((tier, slot_32, lane)) = role_b_head_slot_for_index(idx) {
            let group_16 = slot_32 % 16;
            let feat_idx = tier * 64 + group_16 * 4 + lane;
            let g_diag = gram_diag_256[feat_idx].max(1e-12);
            let raw_eps = min_mask / cold_sqrt_f64(g_diag);
            envelopes[idx] = raw_eps.clamp(5e-5, 2e-4) as f32;
        }
    }
    envelopes
}

#[inline]
fn validate_tardos_args(arena_len: usize, code_len: usize) -> Result<usize, DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena_len);
    if max_tiers == 0 {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
            actual: arena_len,
        });
    }
    let needed_chips = TARDOS_TIER_INDEX_BOUNDS[max_tiers - 1].1;
    if code_len < needed_chips {
        return Err(DnamError::BufferTooSmall {
            expected: needed_chips,
            actual: code_len,
        });
    }
    Ok(max_tiers)
}

#[inline]
fn chip_sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline]
fn write_fp16_tardos_sample(arena: &mut [u8], byte_off: usize, chip: f32, eps: f32) {
    if chip == 0.0 || eps <= 0.0 {
        return;
    }
    let w = fp16_to_f32(read_u16_le(arena, byte_off));
    if !w.is_finite() {
        return;
    }
    let delta = eps.max(2.0 * fp16_ulp(w));
    if w.abs() <= FP16_MAX_FINITE - delta {
        write_u16_le(arena, byte_off, f32_to_fp16(w + delta * chip));
    }
}

/// Embeds **Layer 3 (`fp16` `W_readout` Tardos Perturbation)** enforcing the `2 * fp16_ulp(w)` survival
/// floor across all retained linear `fp16` Role B coordinates in `Arena 0..3`.
pub fn apply_tardos_w_readout_fp16(
    arena: &mut [u8],
    code: &[f32],
    epsilon: f32,
) -> Result<(), DnamError> {
    let _ = validate_tardos_args(arena.len(), code.len())?;
    for_each_tardos_fp16_offset_up_to(arena.len(), |idx, byte_off| {
        write_fp16_tardos_sample(arena, byte_off, chip_sign(code[idx]), epsilon);
    });
    Ok(())
}

/// Scores **Layer 3 (`fp16` Role B Tardos Perturbation)** on `suspect_arena` vs `master_arena`.
pub fn score_tardos_w_readout_fp16(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32],
) -> Result<f64, DnamError> {
    let shared_len = master_arena.len().min(suspect_arena.len());
    let _ = validate_tardos_args(shared_len, code.len())?;
    let mut corr = 0.0_f64;
    let mut active_y2 = 0.0_f64;
    for_each_tardos_fp16_offset_up_to(shared_len, |idx, byte_off| {
        let w_m = fp16_to_f32(read_u16_le(master_arena, byte_off));
        let w_s = fp16_to_f32(read_u16_le(suspect_arena, byte_off));
        if !w_m.is_finite() || !w_s.is_finite() {
            return;
        }
        let diff = (w_s - w_m) as f64;
        let y_i = if diff > 1e-7 {
            1.0_f64
        } else if diff < -1e-7 {
            -1.0_f64
        } else {
            0.0_f64
        };
        corr += (code[idx] as f64) * y_i;
        active_y2 += y_i * y_i;
    });
    Ok(corr / cold_sqrt_f64(active_y2.max(1.0)))
}

/// Embeds **Layer 3 (Prefix-Composable Spread-Spectrum Tardos Watermark)** across all retained
/// `5,811` Role B `F32` and `fp16` carriers (`Role A` untouched).
pub fn apply_tardos_watermark_7680(
    arena: &mut [u8],
    code: &[f32],
    epsilon: f32,
) -> Result<(), DnamError> {
    let _ = validate_tardos_args(arena.len(), code.len())?;
    for_each_tardos_f32_offset_up_to(arena.len(), |idx, byte_off| {
        let chip = chip_sign(code[idx]);
        let w = read_f32_wire(arena, byte_off);
        if w.is_finite() {
            write_f32_wire(arena, byte_off, w + epsilon * chip);
        }
    });
    apply_tardos_w_readout_fp16(arena, code, epsilon)
}

/// Embeds **Layer 3 (`EMR`-Masked Spread-Spectrum Tardos Watermark)** using per-coordinate
/// envelopes $\epsilon_i$ on Role B (`2 * fp16_ulp(w)` floor on `fp16`; `0.0` on Role A).
pub fn apply_tardos_watermark_7680_envelopes(
    arena: &mut [u8],
    code: &[f32],
    envelopes: &[f32; TARDOS_CASCADE_WEIGHTS],
) -> Result<(), DnamError> {
    let _ = validate_tardos_args(arena.len(), code.len())?;
    for_each_tardos_f32_offset_up_to(arena.len(), |idx, byte_off| {
        let chip = chip_sign(code[idx]);
        let eps = envelopes[idx].max(0.0);
        let w = read_f32_wire(arena, byte_off);
        if w.is_finite() {
            write_f32_wire(arena, byte_off, w + eps * chip);
        }
    });
    for_each_tardos_fp16_offset_up_to(arena.len(), |idx, byte_off| {
        write_fp16_tardos_sample(
            arena,
            byte_off,
            chip_sign(code[idx]),
            envelopes[idx].max(0.0),
        );
    });
    Ok(())
}

/// Computes the symmetric Škorić–Tardos Z-score ($\sigma$) of `suspect_arena` against `master_arena`
/// and `code` (`Q_buyer == None` for unrotated master comparison).
pub fn score_tardos_watermark_7680(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32],
) -> Result<f64, DnamError> {
    score_tardos_watermark_7680_with_buyer_gauge(master_arena, suspect_arena, code, None)
}

/// Computes the symmetric Škorić–Tardos Z-score ($\sigma$) of `suspect_arena` against `master_arena`,
/// inverse-rotating `ctrl_base.w_ctrl[1..16][0..4]` by `Q_buyer^-1` when `q_buyer_4x4` is `Some`.
pub fn score_tardos_watermark_7680_with_buyer_gauge(
    master_arena: &[u8],
    suspect_arena: &[u8],
    code: &[f32],
    q_buyer_4x4: Option<&[[f32; 4]; 4]>,
) -> Result<f64, DnamError> {
    let shared_len = master_arena.len().min(suspect_arena.len());
    let _ = validate_tardos_args(shared_len, code.len())?;
    let mut corr = 0.0_f64;
    let mut active_y2 = 0.0_f64;

    let suspect_w_ctrl_opt = q_buyer_4x4.and_then(|_| {
        Arena0ControlBusWire::ref_from_prefix(
            suspect_arena.get(crate::graph::weights::ARENA0_CONTROL_BUS_AND_CLAMPS_OFFSET..)?,
        )
        .ok()
        .map(|(cb, _)| cb.w_ctrl)
    });

    for_each_tardos_f32_offset_up_to(shared_len, |idx, byte_off| {
        let w_m = read_f32_wire(master_arena, byte_off);
        let w_s = if (120..360).contains(&idx)
            && let (Some(q), Some(w_ctrl)) = (q_buyer_4x4, suspect_w_ctrl_opt.as_ref())
        {
            let rel = idx - 120;
            let row = 1 + (rel / 16);
            let col = rel % 16;
            if col < 4 {
                let mut acc = 0.0_f32;
                for k in 0..4 {
                    acc += w_ctrl[row][k].get() * q[col][k];
                }
                acc
            } else {
                read_f32_wire(suspect_arena, byte_off)
            }
        } else {
            read_f32_wire(suspect_arena, byte_off)
        };
        if !w_m.is_finite() || !w_s.is_finite() {
            return;
        }
        let diff = (w_s - w_m) as f64;
        let y_i = if diff > 1e-7 {
            1.0_f64
        } else if diff < -1e-7 {
            -1.0_f64
        } else {
            0.0_f64
        };
        corr += (code[idx] as f64) * y_i;
        active_y2 += y_i * y_i;
    });

    for_each_tardos_fp16_offset_up_to(shared_len, |idx, byte_off| {
        let w_m = fp16_to_f32(read_u16_le(master_arena, byte_off));
        let w_s = fp16_to_f32(read_u16_le(suspect_arena, byte_off));
        if !w_m.is_finite() || !w_s.is_finite() {
            return;
        }
        let diff = (w_s - w_m) as f64;
        let y_i = if diff > 1e-7 {
            1.0_f64
        } else if diff < -1e-7 {
            -1.0_f64
        } else {
            0.0_f64
        };
        corr += (code[idx] as f64) * y_i;
        active_y2 += y_i * y_i;
    });
    Ok(corr / cold_sqrt_f64(active_y2.max(1.0)))
}

/// Standalone Role B reciprocal bilinear gauge helper strictly inside Role B `F32` carriers for `tier`.
pub fn apply_cascade_head_gauge(
    arena: &mut [u8],
    tier: usize,
    role_b_delta_32: &[[f32; 4]; 32],
) -> Result<(), DnamError> {
    let max_tiers = max_tiers_for_arena_len(arena.len());
    if max_tiers == 0 || tier >= max_tiers {
        return Err(DnamError::BufferTooSmall {
            expected: ARENA_LITE_BYTES,
            actual: arena.len(),
        });
    }
    let mut offsets = Vec::new();
    for span in TARDOS_F32_SPANS {
        if (span.tier as usize) == tier {
            for w in 0..span.len() {
                offsets.push(span.byte_base + w * 4);
            }
        }
    }
    if offsets.len() < 256 {
        return Err(DnamError::BufferTooSmall {
            expected: 256 * 4,
            actual: offsets.len() * 4,
        });
    }
    let vec4_count = offsets.len() / 4;
    for (s, delta_vec) in role_b_delta_32.iter().enumerate() {
        for lane in 0..4 {
            let mut sum_before = 1.0_f32;
            let mut v_idx = 32 + s;
            while v_idx < vec4_count {
                let off = offsets[v_idx * 4 + lane];
                sum_before += read_f32_wire(arena, off);
                v_idx += 32;
            }
            let delta = delta_vec[lane];
            let sum_after = sum_before + delta;
            if sum_after.abs() < 1e-6 {
                return Err(DnamError::SchemaValidationError(
                    "Role B gauge multiplier sum_after must be non-zero".into(),
                ));
            }
            let first_off = offsets[128 + s * 4 + lane];
            let cur_first = read_f32_wire(arena, first_off);
            write_f32_wire(arena, first_off, cur_first + delta);

            let scale_ratio = sum_before / sum_after;
            let paired_off = offsets[s * 4 + lane];
            let paired_val = read_f32_wire(arena, paired_off);
            write_f32_wire(arena, paired_off, paired_val * scale_ratio);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        ARENA_OFFSET_FULL_END, ARENA_OFFSET_LITE_END, ARENA_OFFSET_PRO_END, ARENA_OFFSET_ULTRA_END,
    };

    #[test]
    fn test_tardos_codebook_7680_partition_and_ulp_floor() {
        assert_eq!(role_b_head_slot_for_index(TARDOS_CASCADE_WEIGHTS), None);
        assert_eq!(role_b_head_slot_for_index(0), None);
        assert_eq!(role_b_head_slot_for_index(655), None);
        assert_eq!(role_b_head_slot_for_index(1_159), None);
        assert_eq!(role_b_head_slot_for_index(3_051), None);
        assert_eq!(role_b_head_slot_for_index(5_875), None);
        assert!(role_b_head_slot_for_index(16).is_some());
        assert!(role_b_head_slot_for_index(383).is_some());
        assert!(role_b_head_slot_for_index(5_619).is_some());

        let codebook = generate_tardos_code_7680(&[0u8; 32], "model-x", "lic-y");
        assert_eq!(codebook.as_ref().len(), TARDOS_CASCADE_WEIGHTS);
        // Verify 5,875..7,680 tail is strictly 0.0
        for idx in 5_875..TARDOS_CASCADE_WEIGHTS {
            assert_eq!(codebook.chips[idx], 0.0);
            assert_eq!(codebook.accusation_weights[idx], 0.0);
        }

        // Verify tiny perturbation still survives on large |w| via 2 * fp16_ulp(w) floor
        let mut big_w_arena = alloc::vec![0u8; ARENA_LITE_BYTES];
        let first_fp16_off = TARDOS_FP16_SPANS[0].byte_base;
        let orig_bits = f32_to_fp16(4.0);
        write_u16_le(&mut big_w_arena, first_fp16_off, orig_bits);
        apply_tardos_w_readout_fp16(&mut big_w_arena, &codebook.chips, 1e-5).unwrap();
        assert_ne!(read_u16_le(&big_w_arena, first_fp16_off), orig_bits);

        let mut tiny = [0u8; 64];
        assert!(apply_tardos_watermark_7680(&mut tiny, &codebook.chips, 1e-4).is_err());
        assert!(apply_tardos_w_readout_fp16(&mut tiny, &codebook.chips, 1e-3).is_err());
        assert!(score_tardos_w_readout_fp16(&tiny, &tiny, &codebook).is_err());
        assert!(
            apply_tardos_watermark_7680_envelopes(
                &mut tiny,
                &codebook.chips,
                &[1e-4; TARDOS_CASCADE_WEIGHTS]
            )
            .is_err()
        );
        assert!(score_tardos_watermark_7680(&tiny, &tiny, &codebook).is_err());
        assert!(apply_cascade_head_gauge(&mut tiny, 0, &[[0.0; 4]; 32]).is_err());

        for size in [
            ARENA_OFFSET_LITE_END as usize,
            ARENA_OFFSET_FULL_END as usize,
            ARENA_OFFSET_PRO_END as usize,
            ARENA_OFFSET_ULTRA_END as usize,
        ] {
            let mut arena = alloc::vec![0u8; size];
            assert!(apply_tardos_watermark_7680(&mut arena, &[1.0; 10], 1e-4).is_err());
            apply_tardos_watermark_7680(&mut arena, &codebook.chips, 1e-4).unwrap();
            let ro_z =
                score_tardos_w_readout_fp16(&alloc::vec![0u8; size], &arena, &codebook).unwrap();
            assert!(
                ro_z > 8.0,
                "fp16 W_readout Z-score on size {size} too low: {ro_z}"
            );
            let zero_score = score_tardos_watermark_7680(&arena, &arena, &codebook).unwrap();
            assert_eq!(zero_score, 0.0);
        }

        // Ultra gauge test & degenerate sum_after ~ 0 rejection + tier 1 small-range rejection
        let mut ultra = alloc::vec![0u8; ARENA_OFFSET_ULTRA_END as usize];
        apply_cascade_head_gauge(&mut ultra, 3, &[[0.5; 4]; 32]).unwrap();
        assert!(apply_cascade_head_gauge(&mut ultra, 1, &[[0.5; 4]; 32]).is_err());
        let mut bad_delta = [[0.0_f32; 4]; 32];
        bad_delta[0][0] = -1.0;
        assert!(apply_cascade_head_gauge(&mut ultra, 0, &bad_delta).is_err());
    }

    #[test]
    fn test_tardos_non_finite_boundary_and_envelope_paths() {
        let mut tiny = [0u8; 2];
        write_f32_wire(&mut tiny, 8, 1.0);
        write_u16_le(&mut tiny, 8, 1);

        let mut code = alloc::vec![0.0_f32; TARDOS_CASCADE_WEIGHTS];
        code[16] = 1.0;
        code[17] = -1.0;
        code[383] = 1.0;
        code[384] = 1.0;
        code[385] = 0.0;
        let envs = [1e-4_f32; TARDOS_CASCADE_WEIGHTS];

        let mut a1 = alloc::vec![0u8; ARENA_LITE_BYTES];
        write_f32_wire(&mut a1, TARDOS_F32_SPANS[0].byte_base, f32::NAN);
        write_u16_le(&mut a1, TARDOS_FP16_SPANS[0].byte_base, 0x7E00);
        write_u16_le(
            &mut a1,
            TARDOS_FP16_SPANS[0].byte_base + 2,
            f32_to_fp16(FP16_MAX_FINITE),
        );
        let mut a2 = a1.clone();
        apply_tardos_watermark_7680(&mut a1, &code, 1e-4).unwrap();
        apply_tardos_w_readout_fp16(&mut a1, &code, 0.0).unwrap();
        apply_tardos_watermark_7680_envelopes(&mut a2, &code, &envs).unwrap();
        let mut clean = alloc::vec![0u8; ARENA_LITE_BYTES];
        apply_tardos_watermark_7680_envelopes(&mut clean, &code, &envs).unwrap();
        let z_ro = score_tardos_w_readout_fp16(&a1, &clean, &code).unwrap();
        let z_all = score_tardos_watermark_7680(&a1, &a2, &code).unwrap();
        assert!(z_ro.is_finite() && z_all.is_finite());
    }
}
