//! Zero-copy wire decoders, `§2.3` non-finite sanitization, and `LinearReadout` / `ModalConvolver`
//! prefix override builders (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use core::mem::{align_of, size_of};

use dnam_kernels::{MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32, algebraic_unit_rotor_f32};
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::bin::{
    CONVOLVER_KAUTZ_BLOCKS_PER_TIER, CoreQuadWire, DnamBinView, LinearReadoutFullWire,
    LinearReadoutLiteWire, LinearReadoutProWire, LinearReadoutUltraWire,
    SONA_META_CONVOLVER_PREFIX_OVERRIDES, SONA_META_OS2X, SONA_META_TIER_OS_EXPLICIT,
    SonaCrossbarBlockWire, SonaInterBlockTilePairWire,
};
use crate::graph::GraphBackend;
use crate::math::{
    FP16_FINITE_MAX_F32, clamp_finite_f32, f32_to_fp16, fp16_to_f32, sanitize_finite_f32,
};
use crate::types::{Tier, VmError};

use super::super::layout::{
    CoreQuadExtCold, CoreQuadHot, CoreQuadReadoutHot, MAX_CORE_QUADS, SonaColdTail, SonaHotSlab,
    SonaInterBlockTileHot, TierCrossbarSnapshot,
};

const FP16_ABS_MAX: f32 = FP16_FINITE_MAX_F32;
const READOUT_WEIGHTS_PER_LANE: usize = 12;
const RO_GAINS_SCALAR_COUNT: usize = 8;
const CORE_QUAD_LANES: usize = 4;
const LITE_CORE_QUADS: usize = Tier::Lite.simd_groups();
const LITE_CORE_LANES: usize = Tier::Lite.channels();
const FULL_CORE_LANES: usize = Tier::Full.channels();
const PRO_CORE_LANES: usize = Tier::Pro.channels();
const ULTRA_CORE_LANES: usize = Tier::Ultra.channels();
const LITE_W_RO_ELEMS: usize = LITE_CORE_LANES * READOUT_WEIGHTS_PER_LANE;
const PHASOR_STEREO_COUNT: usize = 4;
const CROSSBAR_OPT_TO_LR_LINES: usize = Tier::Ultra.tank_lines();
const LITE_OPT_TO_LR_LINES: usize = Tier::Lite.tank_lines();
const LITE_OPT_TO_LR_OFFSET: usize = 24;
const STAGE_LATENT_DEFAULTS_LEN: usize = 8;
const STAGE_LATENT_GLOBAL_MODS_LEN: usize = 5 * 8;
const STAGE_LATENT_BLOCK_MOD_LEN: usize = 4 * 4;
const STAGE_LATENT_OP_P_TO_CORE_OFFSET: usize =
    STAGE_LATENT_DEFAULTS_LEN + STAGE_LATENT_GLOBAL_MODS_LEN + 2 * STAGE_LATENT_BLOCK_MOD_LEN;

const fn w_ro_and_mods_override_offset(tier_lanes: usize) -> usize {
    tier_lanes * READOUT_WEIGHTS_PER_LANE + RO_GAINS_SCALAR_COUNT
}

const FULL_OVERRIDE_OFFSET: usize = w_ro_and_mods_override_offset(FULL_CORE_LANES);
const PRO_OVERRIDE_OFFSET: usize = w_ro_and_mods_override_offset(PRO_CORE_LANES);
const ULTRA_OVERRIDE_OFFSET: usize = w_ro_and_mods_override_offset(ULTRA_CORE_LANES);

const LITE_FIR_TAPS: usize = Tier::Lite.early_fir_taps();
const FULL_FIR_TAPS: usize = Tier::Full.early_fir_taps();
const PRO_FIR_TAPS: usize = Tier::Pro.early_fir_taps();
const ULTRA_FIR_TAPS: usize = Tier::Ultra.early_fir_taps();
const FULL_DELTA_FIR_TAPS: usize = FULL_FIR_TAPS - LITE_FIR_TAPS;
const PRO_DELTA_FIR_TAPS: usize = PRO_FIR_TAPS - FULL_FIR_TAPS;
const ULTRA_DELTA_FIR_TAPS: usize = ULTRA_FIR_TAPS - PRO_FIR_TAPS;

const _: () = assert!(LITE_CORE_QUADS == 4);
const _: () = assert!(LITE_CORE_LANES == 16);
const _: () = assert!(FULL_CORE_LANES == 24);
const _: () = assert!(PRO_CORE_LANES == 32);
const _: () = assert!(ULTRA_CORE_LANES == 40);
const _: () = assert!(LITE_W_RO_ELEMS == 192);
const _: () = assert!(STAGE_LATENT_OP_P_TO_CORE_OFFSET == 80);
const _: () = assert!(FULL_OVERRIDE_OFFSET == 296);
const _: () = assert!(PRO_OVERRIDE_OFFSET == 392);
const _: () = assert!(ULTRA_OVERRIDE_OFFSET == 488);
const _: () = assert!(LITE_W_RO_ELEMS + LITE_CORE_QUADS * CORE_QUAD_LANES == 208);
const _: () = assert!(LITE_FIR_TAPS == 32 && FULL_DELTA_FIR_TAPS == 16);
const _: () = assert!(PRO_DELTA_FIR_TAPS == 48 && ULTRA_DELTA_FIR_TAPS == 32);

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
struct PrefixOverrideFullWire {
    fir_l: [F32; 16],
    fir_r: [F32; 16],
    c_re_l: [F32; 4],
    c_im_l: [F32; 4],
    c_re_r: [F32; 4],
    c_im_r: [F32; 4],
}

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
struct PrefixOverrideProWire {
    fir_l: [F32; 32],
    fir_r: [F32; 32],
    c_re_l: [F32; 8],
    c_im_l: [F32; 8],
    c_re_r: [F32; 8],
    c_im_r: [F32; 8],
}

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
struct PrefixOverrideUltraWire {
    fir_l: [F32; 48],
    fir_r: [F32; 48],
    c_re_l: [F32; 12],
    c_im_l: [F32; 12],
    c_re_r: [F32; 12],
    c_im_r: [F32; 12],
}

const _: () = assert!(
    size_of::<PrefixOverrideFullWire>() == 192 && align_of::<PrefixOverrideFullWire>() == 1
);
const _: () =
    assert!(size_of::<PrefixOverrideProWire>() == 384 && align_of::<PrefixOverrideProWire>() == 1);
const _: () = assert!(
    size_of::<PrefixOverrideUltraWire>() == 576 && align_of::<PrefixOverrideUltraWire>() == 1
);

#[inline(always)]
pub(super) fn clamp_finite_fp16(bits: u16, lo: f32, hi: f32, fallback: f32) -> u16 {
    f32_to_fp16(clamp_finite_f32(fp16_to_f32(bits), lo, hi, fallback))
}

#[inline(always)]
pub(super) fn fp16_sanitized(bits: u16) -> u16 {
    f32_to_fp16(clamp_finite_f32(
        fp16_to_f32(bits),
        -FP16_ABS_MAX,
        FP16_ABS_MAX,
        0.0,
    ))
}

pub(super) fn compute_tier_os2x_flags(meta_flags: u32, oversample_factor: u16) -> [bool; 4] {
    if (meta_flags & SONA_META_TIER_OS_EXPLICIT) != 0 {
        [
            false,
            (meta_flags & (1 << 1)) != 0,
            (meta_flags & (1 << 2)) != 0,
            (meta_flags & (1 << 3)) != 0,
        ]
    } else {
        let on = (meta_flags & SONA_META_OS2X) != 0 || oversample_factor >= 2;
        [false, on, on, on]
    }
}

/// Decodes one `192 B` [`CoreQuadWire`] into (`144 B` [`CoreQuadHot`], `40 B` [`CoreQuadExtCold`],
/// `4`-bit rectifier sign mask).
pub(super) fn decode_core_quad_wire(wire: &CoreQuadWire) -> (CoreQuadHot, CoreQuadExtCold, u8) {
    let mut quad = CoreQuadHot::default();
    let mut ext = CoreQuadExtCold::default();
    let mut rect_bits = 0_u8;
    for l in 0..4 {
        quad.svf_g[l] = clamp_finite_f32(wire.svf_g[l].get(), 1.0e-5, 64.0, 0.25);
        quad.lru_alpha_rel[l] =
            clamp_finite_f32(wire.lru_alpha_rel[l].get(), 0.0, MAX_PARALRU_R_F32, 0.0);
        quad.lru_alpha_atk[l] = clamp_finite_f32(wire.lru_alpha_atk[l].get(), 0.0, 1.0, 0.0);
        quad.svf_k_fp16[l] = clamp_finite_fp16(
            wire.svf_k_fp16[l].get(),
            0.0,
            8.0,
            core::f32::consts::FRAC_1_SQRT_2,
        );
        quad.lru_kappa_fp16[l] = clamp_finite_fp16(wire.lru_kappa_fp16[l].get(), 0.0, 1.0, 0.0);
        quad.adaa_drive_fp16[l] =
            clamp_finite_fp16(wire.adaa_drive_fp16[l].get(), -32.0, 32.0, 0.0);
        quad.lru_bias_drive_fp16[l] =
            clamp_finite_fp16(wire.lru_bias_drive_fp16[l].get(), -16.0, 16.0, 0.0);
        quad.in_mid_weight_fp16[l] =
            clamp_finite_fp16(wire.in_mid_weight_fp16[l].get(), -16.0, 16.0, 0.0);
        quad.tone_lp_fp16[l] = clamp_finite_fp16(wire.tone_lp_fp16[l].get(), -16.0, 16.0, 0.0);
        quad.tone_bp_fp16[l] = clamp_finite_fp16(wire.tone_bp_fp16[l].get(), -16.0, 16.0, 0.0);
        quad.tone_hp_fp16[l] = clamp_finite_fp16(wire.tone_hp_fp16[l].get(), -16.0, 16.0, 0.0);
        quad.tone_ap_fp16[l] = clamp_finite_fp16(wire.tone_ap_fp16[l].get(), -16.0, 16.0, 0.0);
        quad.tone_dir_fp16[l] = clamp_finite_fp16(wire.tone_dir_fp16[l].get(), -16.0, 16.0, 0.0);
        quad.sag_depth_fp16[l] = clamp_finite_fp16(wire.sag_depth_fp16[l].get(), -8.0, 8.0, 0.0);
        quad.wavefolder_mix_fp16[l] =
            clamp_finite_fp16(wire.wavefolder_mix_fp16[l].get(), 0.0, 1.0, 0.0);
        let rs = fp16_to_f32(wire.rect_sign_fp16[l].get());
        if rs.is_finite() && rs < 0.0 {
            rect_bits |= 1 << l;
        }

        ext.in_side_weight_fp16[l] =
            clamp_finite_fp16(wire.in_side_weight_fp16[l].get(), -16.0, 16.0, 0.0);
        ext.lru_stereo_link_fp16[l] =
            clamp_finite_fp16(wire.lru_stereo_link_fp16[l].get(), 0.0, 1.0, 0.0);
        ext.lru_duhem_gamma_fp16[l] = clamp_finite_fp16(
            wire.lru_duhem_gamma_fp16[l].get(),
            0.0,
            MAX_PARALRU_DUHEM_GAMMA_F32,
            0.0,
        );
        ext.lru_signed_mix_fp16[l] =
            clamp_finite_fp16(wire.lru_signed_mix_fp16[l].get(), 0.0, 1.0, 0.0);
        ext.glu_mix_fp16[l] = clamp_finite_fp16(wire.glu_mix_fp16[l].get(), -4.0, 4.0, 0.0);
    }
    (quad, ext, rect_bits)
}

pub(super) fn decode_tile_wire(wire: &SonaInterBlockTilePairWire) -> SonaInterBlockTileHot {
    let mut t = SonaInterBlockTileHot::default();
    for s in 0..4 {
        for d in 0..4 {
            t.w_odd_4x4[s][d] = clamp_finite_f32(wire.w_odd_4x4[s][d].get(), -8.0, 8.0, 0.0);
            t.w_even_4x4_fp16[s][d] =
                clamp_finite_fp16(wire.w_even_4x4_fp16[s][d].get(), -8.0, 8.0, 0.0);
            t.w_lru_4x4_fp16[s][d] =
                clamp_finite_fp16(wire.w_lru_4x4_fp16[s][d].get(), -8.0, 8.0, 0.0);
        }
    }
    t
}

fn unpack_w_ro_into(
    words: &[U16],
    num_lanes: usize,
    dst: &mut [CoreQuadReadoutHot; MAX_CORE_QUADS],
) {
    for l in 0..num_lanes {
        let q = l >> 2;
        let k = l & 3;
        let b = l * READOUT_WEIGHTS_PER_LANE;
        dst[q].pre_conv_odd_l_fp16[k] = fp16_sanitized(words[b].get());
        dst[q].pre_conv_odd_r_fp16[k] = fp16_sanitized(words[b + 1].get());
        dst[q].pre_conv_even_l_fp16[k] = fp16_sanitized(words[b + 2].get());
        dst[q].pre_conv_even_r_fp16[k] = fp16_sanitized(words[b + 3].get());
        dst[q].pre_conv_lru_l_fp16[k] = fp16_sanitized(words[b + 4].get());
        dst[q].pre_conv_lru_r_fp16[k] = fp16_sanitized(words[b + 5].get());
        dst[q].direct_odd_l_fp16[k] = fp16_sanitized(words[b + 6].get());
        dst[q].direct_odd_r_fp16[k] = fp16_sanitized(words[b + 7].get());
        dst[q].direct_even_l_fp16[k] = fp16_sanitized(words[b + 8].get());
        dst[q].direct_even_r_fp16[k] = fp16_sanitized(words[b + 9].get());
        dst[q].direct_lru_l_fp16[k] = fp16_sanitized(words[b + 10].get());
        dst[q].direct_lru_r_fp16[k] = fp16_sanitized(words[b + 11].get());
    }
}

fn decode_crossbar_block_snapshot(
    blk: &SonaCrossbarBlockWire,
    w_ro_and_mods: &[U16],
    num_lanes: usize,
) -> TierCrossbarSnapshot {
    let mut snap = TierCrossbarSnapshot::default();
    for q in 0..MAX_CORE_QUADS {
        for l in 0..CORE_QUAD_LANES {
            snap.op_p_to_core[q][l] =
                clamp_finite_fp16(blk.op_p_to_core_fp16[q][l].get(), -8.0, 8.0, 0.0);
            snap.phasor_to_core[q][l] =
                clamp_finite_fp16(blk.phasor_to_core_fp16[q][l].get(), -8.0, 8.0, 0.0);
        }
    }
    for idx in 0..CROSSBAR_OPT_TO_LR_LINES {
        snap.opt_to_lr[idx] = [
            clamp_finite_fp16(blk.opt_to_lr_fp16[idx][0].get(), -4.0, 4.0, 0.0),
            clamp_finite_fp16(blk.opt_to_lr_fp16[idx][1].get(), -4.0, 4.0, 0.0),
        ];
    }
    for l in 0..PHASOR_STEREO_COUNT {
        let th = clamp_finite_f32(
            fp16_to_f32(blk.phasor_stereo_phase_fp16[l].get()),
            -core::f32::consts::PI,
            core::f32::consts::PI,
            0.0,
        );
        let (c, s) = algebraic_unit_rotor_f32(th);
        snap.phasor_st_cos[l] = f32_to_fp16(c);
        snap.phasor_st_sin[l] = f32_to_fp16(s);
    }
    let w_count = num_lanes * READOUT_WEIGHTS_PER_LANE;
    unpack_w_ro_into(&w_ro_and_mods[..w_count], num_lanes, &mut snap.readouts);
    for i in 0..RO_GAINS_SCALAR_COUNT {
        snap.ro_gains_fp16[i] = fp16_sanitized(w_ro_and_mods[w_count + i].get());
    }
    snap
}

fn decode_lite_crossbar_snapshot(
    ro: &LinearReadoutLiteWire,
    rho_tail: &[F32; 32],
) -> TierCrossbarSnapshot {
    let mut snap = TierCrossbarSnapshot::default();
    let f_gains = [
        ro.pre_conv_u_gain[0].get(),
        ro.pre_conv_u_gain[1].get(),
        ro.pre_conv_hp_gain[0].get(),
        ro.pre_conv_hp_gain[1].get(),
        ro.direct_s2_u_gain[0].get(),
        ro.direct_s2_u_gain[1].get(),
        ro.direct_s2_hp_gain[0].get(),
        ro.direct_s2_hp_gain[1].get(),
    ];
    for (i, &g) in f_gains.iter().enumerate() {
        snap.ro_gains_fp16[i] = f32_to_fp16(clamp_finite_f32(g, -FP16_ABS_MAX, FP16_ABS_MAX, 0.0));
    }
    unpack_w_ro_into(
        &ro.w_ro_fp16[..LITE_W_RO_ELEMS],
        LITE_CORE_LANES,
        &mut snap.readouts,
    );
    for q in 0..LITE_CORE_QUADS {
        for l in 0..CORE_QUAD_LANES {
            snap.op_p_to_core[q][l] = clamp_finite_fp16(
                ro.stage_latent_mods_fp16
                    [STAGE_LATENT_OP_P_TO_CORE_OFFSET + q * CORE_QUAD_LANES + l]
                    .get(),
                -8.0,
                8.0,
                0.0,
            );
            snap.phasor_to_core[q][l] = clamp_finite_fp16(
                ro.w_ro_fp16[LITE_W_RO_ELEMS + q * CORE_QUAD_LANES + l].get(),
                -8.0,
                8.0,
                0.0,
            );
        }
    }
    for idx in 0..LITE_OPT_TO_LR_LINES {
        snap.opt_to_lr[idx] = [
            f32_to_fp16(clamp_finite_f32(
                rho_tail[LITE_OPT_TO_LR_OFFSET + idx * 2].get(),
                -4.0,
                4.0,
                0.0,
            )),
            f32_to_fp16(clamp_finite_f32(
                rho_tail[LITE_OPT_TO_LR_OFFSET + idx * 2 + 1].get(),
                -4.0,
                4.0,
                0.0,
            )),
        ];
    }
    snap
}

fn copy_base_fir_and_kautz<B: GraphBackend>(view: &DnamBinView<'_>, slab: &mut SonaHotSlab<'_, B>) {
    let mut base_fir_l = [0.0_f32; ULTRA_FIR_TAPS];
    let mut base_fir_r = [0.0_f32; ULTRA_FIR_TAPS];
    let cold = slab.cold_mut();
    for (i, tap) in view.arena0.conv_fir_lite.fir_l.iter().enumerate() {
        base_fir_l[i] = clamp_finite_f32(tap.get(), -16.0, 16.0, 0.0);
        base_fir_r[i] =
            clamp_finite_f32(view.arena0.conv_fir_lite.fir_r[i].get(), -16.0, 16.0, 0.0);
    }
    cold.wire_kautz[0] = view.arena0.conv_kautz_lite[0];
    cold.wire_kautz[1] = view.arena0.conv_kautz_lite[1];
    if let Some(a1) = view.arena1 {
        for i in 0..FULL_DELTA_FIR_TAPS {
            base_fir_l[LITE_FIR_TAPS + i] =
                clamp_finite_f32(a1.conv_fir_full.fir_l[i].get(), -16.0, 16.0, 0.0);
            base_fir_r[LITE_FIR_TAPS + i] =
                clamp_finite_f32(a1.conv_fir_full.fir_r[i].get(), -16.0, 16.0, 0.0);
        }
        cold.wire_kautz[2] = a1.conv_kautz_full;
    }
    if let Some(a2) = view.arena2 {
        for i in 0..PRO_DELTA_FIR_TAPS {
            base_fir_l[FULL_FIR_TAPS + i] =
                clamp_finite_f32(a2.conv_fir_pro.fir_l[i].get(), -16.0, 16.0, 0.0);
            base_fir_r[FULL_FIR_TAPS + i] =
                clamp_finite_f32(a2.conv_fir_pro.fir_r[i].get(), -16.0, 16.0, 0.0);
        }
        for b in 0..CONVOLVER_KAUTZ_BLOCKS_PER_TIER[2] {
            cold.wire_kautz[3 + b] = a2.conv_kautz_pro[b];
        }
    }
    if let Some(a3) = view.arena3 {
        for i in 0..ULTRA_DELTA_FIR_TAPS {
            base_fir_l[PRO_FIR_TAPS + i] =
                clamp_finite_f32(a3.conv_fir_ultra.fir_l[i].get(), -16.0, 16.0, 0.0);
            base_fir_r[PRO_FIR_TAPS + i] =
                clamp_finite_f32(a3.conv_fir_ultra.fir_r[i].get(), -16.0, 16.0, 0.0);
        }
        for b in 0..CONVOLVER_KAUTZ_BLOCKS_PER_TIER[3] {
            cold.wire_kautz[6 + b] = a3.conv_kautz_ultra[b];
        }
    }
    let tap_counts = [LITE_FIR_TAPS, FULL_FIR_TAPS, PRO_FIR_TAPS, ULTRA_FIR_TAPS];
    let mut cum_banks = 0_usize;
    for t in 0..=view.retained_tier.index() {
        cum_banks += CONVOLVER_KAUTZ_BLOCKS_PER_TIER[t];
        let tc = tap_counts[t];
        cold.tier_fir_l[t][..tc].copy_from_slice(&base_fir_l[..tc]);
        cold.tier_fir_r[t][..tc].copy_from_slice(&base_fir_r[..tc]);
        for b in 0..cum_banks {
            let blk = &cold.wire_kautz[b];
            for lane in 0..4 {
                let m = b * 4 + lane;
                cold.tier_kautz_b_gamma[t][m] =
                    sanitize_finite_f32(blk.b_gamma[lane].get()).max(0.0);
                cold.tier_kautz_c_l[t][m] = (
                    sanitize_finite_f32(blk.c_re_l[lane].get()),
                    sanitize_finite_f32(blk.c_im_l[lane].get()),
                );
                cold.tier_kautz_c_r[t][m] = (
                    sanitize_finite_f32(blk.c_re_r[lane].get()),
                    sanitize_finite_f32(blk.c_im_r[lane].get()),
                );
            }
        }
    }
}

fn apply_override_slice<
    const N_PRE: usize,
    const N_EXT: usize,
    const K_PRE: usize,
    const K_EXT: usize,
>(
    cold: &mut SonaColdTail<'_, impl GraphBackend>,
    tier_idx: usize,
    ov_fir: (&[F32], &[F32]),
    ov_c: (&[F32], &[F32], &[F32], &[F32]),
    extra: &[U16],
) {
    for i in 0..N_PRE {
        cold.tier_fir_l[tier_idx][i] = clamp_finite_f32(ov_fir.0[i].get(), -16.0, 16.0, 0.0);
        cold.tier_fir_r[tier_idx][i] = clamp_finite_f32(ov_fir.1[i].get(), -16.0, 16.0, 0.0);
    }
    for i in 0..N_EXT {
        cold.tier_fir_l[tier_idx][N_PRE + i] =
            clamp_finite_f32(fp16_to_f32(extra[i].get()), -16.0, 16.0, 0.0);
        cold.tier_fir_r[tier_idx][N_PRE + i] =
            clamp_finite_f32(fp16_to_f32(extra[N_EXT + i].get()), -16.0, 16.0, 0.0);
    }
    for m in 0..K_PRE {
        cold.tier_kautz_c_l[tier_idx][m] = (
            sanitize_finite_f32(ov_c.0[m].get()),
            sanitize_finite_f32(ov_c.1[m].get()),
        );
        cold.tier_kautz_c_r[tier_idx][m] = (
            sanitize_finite_f32(ov_c.2[m].get()),
            sanitize_finite_f32(ov_c.3[m].get()),
        );
    }
    let total_pairs = K_PRE + K_EXT;
    let bg_base = N_EXT * 2;
    for m in 0..total_pairs {
        cold.tier_kautz_b_gamma[tier_idx][m] =
            sanitize_finite_f32(fp16_to_f32(extra[bg_base + m].get())).max(0.0);
    }
    let c_base = bg_base + total_pairs;
    for k in 0..K_EXT {
        let m = K_PRE + k;
        let off = c_base + 4 * k;
        cold.tier_kautz_c_l[tier_idx][m] = (
            sanitize_finite_f32(fp16_to_f32(extra[off].get())),
            sanitize_finite_f32(fp16_to_f32(extra[off + 1].get())),
        );
        cold.tier_kautz_c_r[tier_idx][m] = (
            sanitize_finite_f32(fp16_to_f32(extra[off + 2].get())),
            sanitize_finite_f32(fp16_to_f32(extra[off + 3].get())),
        );
    }
}

pub(super) fn decode_readout_and_overrides<B: GraphBackend>(
    view: &DnamBinView<'_>,
    slab: &mut SonaHotSlab<'_, B>,
) -> Result<(), VmError> {
    copy_base_fir_and_kautz(view, slab);
    let apply_ov = (slab.cold().meta_flags & SONA_META_CONVOLVER_PREFIX_OVERRIDES) != 0;

    let cold = slab.cold_mut();
    cold.tier_crossbars[0] = decode_lite_crossbar_snapshot(
        &view.arena0.linear_readout_lite,
        &view.arena0.control_bus_lite.rho_head_and_ctrl_tail,
    );
    if let Some(a1) = view.arena1 {
        let ro: &LinearReadoutFullWire = &a1.linear_readout_full;
        cold.tier_crossbars[1] = decode_crossbar_block_snapshot(
            &a1.crossbar_full,
            &ro.w_ro_and_mods_fp16,
            FULL_CORE_LANES,
        );
        if apply_ov {
            let ov = PrefixOverrideFullWire::ref_from_bytes(&ro.prefix_override)
                .map_err(|_| VmError::InvalidArenaHeader)?;
            apply_override_slice::<16, 16, 4, 4>(
                cold,
                1,
                (&ov.fir_l, &ov.fir_r),
                (&ov.c_re_l, &ov.c_im_l, &ov.c_re_r, &ov.c_im_r),
                &ro.w_ro_and_mods_fp16[FULL_OVERRIDE_OFFSET..],
            );
        }
    }
    if let Some(a2) = view.arena2 {
        let ro: &LinearReadoutProWire = &a2.linear_readout_pro;
        cold.tier_crossbars[2] = decode_crossbar_block_snapshot(
            &a2.crossbar_pro,
            &ro.w_ro_and_mods_fp16,
            PRO_CORE_LANES,
        );
        if apply_ov {
            let ov = PrefixOverrideProWire::ref_from_bytes(&ro.prefix_override)
                .map_err(|_| VmError::InvalidArenaHeader)?;
            apply_override_slice::<32, 16, 8, 4>(
                cold,
                2,
                (&ov.fir_l, &ov.fir_r),
                (&ov.c_re_l, &ov.c_im_l, &ov.c_re_r, &ov.c_im_r),
                &ro.w_ro_and_mods_fp16[PRO_OVERRIDE_OFFSET..],
            );
        }
    }
    if let Some(a3) = view.arena3 {
        let ro: &LinearReadoutUltraWire = &a3.linear_readout_ultra;
        cold.tier_crossbars[3] = decode_crossbar_block_snapshot(
            &a3.crossbar_ultra,
            &ro.w_ro_and_mods_fp16,
            ULTRA_CORE_LANES,
        );
        if apply_ov {
            let ov = PrefixOverrideUltraWire::ref_from_bytes(&ro.prefix_override)
                .map_err(|_| VmError::InvalidArenaHeader)?;
            apply_override_slice::<48, 48, 12, 12>(
                cold,
                3,
                (&ov.fir_l, &ov.fir_r),
                (&ov.c_re_l, &ov.c_im_l, &ov.c_re_r, &ov.c_im_r),
                &ro.w_ro_and_mods_fp16[ULTRA_OVERRIDE_OFFSET..],
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_tier_os2x_flags_explicit_vs_global() {
        assert_eq!(
            compute_tier_os2x_flags(SONA_META_TIER_OS_EXPLICIT | 0b1010, 1),
            [false, true, false, true]
        );
        assert_eq!(
            compute_tier_os2x_flags(SONA_META_OS2X, 1),
            [false, true, true, true]
        );
        assert_eq!(compute_tier_os2x_flags(0, 1), [false, false, false, false]);
    }

    #[test]
    fn test_wire_decode_clamps_and_interleaved_override_residues() {
        use crate::math::ScalarBackend;
        use zerocopy::FromZeros;

        let mut cw = CoreQuadWire::new_zeroed();
        cw.svf_g[0] = F32::new(f32::NAN);
        cw.lru_kappa_fp16[0] = U16::new(f32_to_fp16(12.0));
        cw.adaa_drive_fp16[0] = U16::new(f32_to_fp16(-12.5));
        cw.lru_bias_drive_fp16[0] = U16::new(f32_to_fp16(-25.0));
        cw.sag_depth_fp16[0] = U16::new(f32_to_fp16(-6.0));
        cw.in_mid_weight_fp16[0] = U16::new(f32_to_fp16(20.0));
        cw.in_side_weight_fp16[0] = U16::new(f32_to_fp16(-20.0));
        cw.lru_stereo_link_fp16[0] = U16::new(f32_to_fp16(2.5));
        cw.lru_duhem_gamma_fp16[0] = U16::new(f32_to_fp16(100.0));
        cw.lru_signed_mix_fp16[0] = U16::new(f32_to_fp16(0.5));
        cw.glu_mix_fp16[0] = U16::new(f32_to_fp16(-6.0));
        cw.rect_sign_fp16[1] = U16::new(f32_to_fp16(-1.0));
        let (hq, h_ext, rect_bits) = decode_core_quad_wire(&cw);
        assert_eq!(rect_bits, 0b0010);
        assert_eq!(hq.svf_g[0], 0.25);
        assert_eq!(fp16_to_f32(hq.lru_kappa_fp16[0]), 1.0);
        assert_eq!(fp16_to_f32(hq.adaa_drive_fp16[0]), -12.5);
        assert_eq!(fp16_to_f32(hq.lru_bias_drive_fp16[0]), -16.0);
        assert_eq!(fp16_to_f32(hq.sag_depth_fp16[0]), -6.0);
        assert_eq!(fp16_to_f32(hq.in_mid_weight_fp16[0]), 16.0);
        assert_eq!(fp16_to_f32(h_ext.in_side_weight_fp16[0]), -16.0);
        assert_eq!(fp16_to_f32(h_ext.lru_stereo_link_fp16[0]), 1.0);
        assert_eq!(fp16_to_f32(h_ext.lru_duhem_gamma_fp16[0]), 64.0);
        assert_eq!(fp16_to_f32(h_ext.lru_signed_mix_fp16[0]), 0.5);
        assert_eq!(fp16_to_f32(h_ext.glu_mix_fp16[0]), -4.0);

        let mut tw = SonaInterBlockTilePairWire::new_zeroed();
        tw.w_odd_4x4[0][0] = F32::new(-12.0);
        tw.w_even_4x4_fp16[0][0] = U16::new(f32_to_fp16(12.0));
        let ht = decode_tile_wire(&tw);
        assert_eq!(ht.w_odd_4x4[0][0], -8.0);
        assert_eq!(fp16_to_f32(ht.w_even_4x4_fp16[0][0]), 8.0);

        let mut cold = SonaColdTail::<ScalarBackend>::empty();
        let ov_l = [F32::new(24.0); 4];
        let ov_r = [F32::new(-24.0); 4];
        let ov_c = [F32::new(0.1); 2];
        let mut extra = [U16::new(0); 32];
        // N_EXT = 2 -> extra[0..4] FIR, bg_base = 4, K_PRE + K_EXT = 4 -> c_base = 8
        extra[0] = U16::new(f32_to_fp16(20.0));
        extra[2] = U16::new(f32_to_fp16(-20.0));
        // Interleaved pole k = 1 at c_base + 4*1 = 12..16: [re_l, im_l, re_r, im_r]
        extra[12] = U16::new(f32_to_fp16(1.25));
        extra[13] = U16::new(f32_to_fp16(-2.5));
        extra[14] = U16::new(f32_to_fp16(3.5));
        extra[15] = U16::new(f32_to_fp16(-4.25));
        apply_override_slice::<4, 2, 2, 2>(
            &mut cold,
            1,
            (&ov_l, &ov_r),
            (&ov_c, &ov_c, &ov_c, &ov_c),
            &extra,
        );
        assert_eq!(cold.tier_fir_l[1][0], 16.0);
        assert_eq!(cold.tier_fir_r[1][0], -16.0);
        assert_eq!(cold.tier_fir_l[1][4], 16.0);
        assert_eq!(cold.tier_fir_r[1][4], -16.0);
        assert_eq!(cold.tier_kautz_c_l[1][3], (1.25, -2.5));
        assert_eq!(cold.tier_kautz_c_r[1][3], (3.5, -4.25));
    }
}
