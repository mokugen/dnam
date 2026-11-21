//! Zero-copy wire decoders, `§2.3` non-finite sanitization, and `VarPro` / Cabinet prefix
//! override builders (`#![no_std]`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use core::mem::{align_of, size_of};

use crate::math::{
    FP16_FINITE_MAX_F32, clamp_finite_f32, f32_to_fp16, fp16_to_f32, sanitize_finite_f32,
};

use dnam_kernels::{MAX_PARALRU_R_F32, algebraic_unit_rotor_f32};
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::bin::{
    DnamBinView, SonaCrossbarBlockWire, SonaGroupWire, SonaInterBlockTilePairWire,
    VarProReadoutFullWire, VarProReadoutLiteWire, VarProReadoutProWire, VarProReadoutUltraWire,
};
use crate::graph::GraphBackend;
use crate::types::VmError;

use super::super::layout::{
    MAX_SONA_GROUPS, SonaColdTail, SonaGroupHot, SonaGroupReadoutHot, SonaHotSlab,
    SonaInterBlockTileHot, TierCrossbarSnapshot,
};

pub(super) const HEADER_FLAG_SINGLE_INPUT: u16 = 1 << 4;
pub(super) const HEADER_FLAG_WET_POLARITY_INV: u16 = 1 << 7;
pub(super) const HEADER_FLAG_STEREO_OUT: u16 = 1 << 9;

pub(super) const META_OS2X: u32 = 1 << 4;
pub(super) const META_TIER_OS_EXPLICIT: u32 = 1 << 6;
pub(super) const META_CABINET_PREFIX_OVERRIDES: u32 = 1 << 7;
pub(super) const META_OS_EXT_PRESENT: u32 = 1 << 8;
pub(super) const META_CORE_MODS_PRESENT: u32 = 1 << 9;
pub(super) const TANK_FLAG_EXT_F32: u8 = 1 << 1;

const FP16_ABS_MAX: f32 = FP16_FINITE_MAX_F32;

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
    if (meta_flags & META_TIER_OS_EXPLICIT) != 0 {
        [
            false,
            (meta_flags & (1 << 1)) != 0,
            (meta_flags & (1 << 2)) != 0,
            (meta_flags & (1 << 3)) != 0,
        ]
    } else {
        let on = (meta_flags & META_OS2X) != 0 || oversample_factor >= 2;
        [false, on, on, on]
    }
}

pub(super) fn decode_group_wire(wire: &SonaGroupWire) -> (SonaGroupHot, u8) {
    let mut g = SonaGroupHot::default();
    let mut rect_bits = 0_u8;
    for l in 0..4 {
        g.svf_g[l] = clamp_finite_f32(wire.svf_g[l].get(), 1.0e-5, 64.0, 0.25);
        g.lru_alpha_rel[l] =
            clamp_finite_f32(wire.lru_alpha_rel[l].get(), 0.0, MAX_PARALRU_R_F32, 0.0);
        g.lru_alpha_atk[l] = clamp_finite_f32(wire.lru_alpha_atk[l].get(), 0.0, 1.0, 0.0);
        g.svf_k_fp16[l] = f32_to_fp16(clamp_finite_f32(
            wire.svf_k[l].get(),
            0.0,
            8.0,
            core::f32::consts::FRAC_1_SQRT_2,
        ));
        g.lru_kappa_fp16[l] = f32_to_fp16(clamp_finite_f32(wire.lru_kappa[l].get(), 0.0, 1.0, 0.0));
        g.adaa_drive_fp16[l] =
            f32_to_fp16(clamp_finite_f32(wire.adaa_drive[l].get(), -32.0, 32.0, 0.0));
        g.lru_bias_drive_fp16[l] = f32_to_fp16(clamp_finite_f32(
            wire.lru_bias_drive[l].get(),
            -16.0,
            16.0,
            0.0,
        ));
        g.in_x_weight_fp16[l] = f32_to_fp16(clamp_finite_f32(
            wire.in_x_weight[l].get(),
            -16.0,
            16.0,
            0.0,
        ));
        g.tone_lp_fp16[l] = clamp_finite_fp16(wire.tone_lp_fp16[l].get(), -16.0, 16.0, 0.0);
        g.tone_bp_fp16[l] = clamp_finite_fp16(wire.tone_bp_fp16[l].get(), -16.0, 16.0, 0.0);
        g.tone_hp_fp16[l] = clamp_finite_fp16(wire.tone_hp_fp16[l].get(), -16.0, 16.0, 0.0);
        g.tone_ap_fp16[l] = clamp_finite_fp16(wire.tone_ap_fp16[l].get(), -16.0, 16.0, 0.0);
        g.tone_dir_fp16[l] = clamp_finite_fp16(wire.tone_dir_fp16[l].get(), -16.0, 16.0, 0.0);
        g.sag_depth_fp16[l] = clamp_finite_fp16(wire.sag_depth_fp16[l].get(), -8.0, 8.0, 0.0);
        g.wavefolder_mix_fp16[l] =
            clamp_finite_fp16(wire.wavefolder_mix_fp16[l].get(), 0.0, 1.0, 0.0);
        let rs = fp16_to_f32(wire.rect_sign_fp16[l].get());
        if rs.is_finite() && rs < 0.0 {
            rect_bits |= 1 << l;
        }
    }
    (g, rect_bits)
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
    dst: &mut [SonaGroupReadoutHot; MAX_SONA_GROUPS],
) {
    for l in 0..num_lanes {
        let g = l >> 2;
        let k = l & 3;
        let b = l * 12;
        dst[g].pre_cab_odd_l_fp16[k] = fp16_sanitized(words[b].get());
        dst[g].pre_cab_odd_r_fp16[k] = fp16_sanitized(words[b + 1].get());
        dst[g].pre_cab_even_l_fp16[k] = fp16_sanitized(words[b + 2].get());
        dst[g].pre_cab_even_r_fp16[k] = fp16_sanitized(words[b + 3].get());
        dst[g].pre_cab_lru_l_fp16[k] = fp16_sanitized(words[b + 4].get());
        dst[g].pre_cab_lru_r_fp16[k] = fp16_sanitized(words[b + 5].get());
        dst[g].direct_odd_l_fp16[k] = fp16_sanitized(words[b + 6].get());
        dst[g].direct_odd_r_fp16[k] = fp16_sanitized(words[b + 7].get());
        dst[g].direct_even_l_fp16[k] = fp16_sanitized(words[b + 8].get());
        dst[g].direct_even_r_fp16[k] = fp16_sanitized(words[b + 9].get());
        dst[g].direct_lru_l_fp16[k] = fp16_sanitized(words[b + 10].get());
        dst[g].direct_lru_r_fp16[k] = fp16_sanitized(words[b + 11].get());
    }
}

fn decode_crossbar_block_snapshot(
    blk: &SonaCrossbarBlockWire,
    w_ro_and_mods: &[U16],
    num_lanes: usize,
) -> TierCrossbarSnapshot {
    let mut snap = TierCrossbarSnapshot::default();
    for g in 0..MAX_SONA_GROUPS {
        for l in 0..4 {
            snap.op_p_to_group[g][l] =
                clamp_finite_fp16(blk.op_p_to_group_fp16[g][l].get(), -8.0, 8.0, 0.0);
            snap.phasor_to_group[g][l] =
                clamp_finite_fp16(blk.phasor_to_group_fp16[g][l].get(), -8.0, 8.0, 0.0);
        }
    }
    for idx in 0..16 {
        snap.opt_to_lr[idx] = [
            clamp_finite_fp16(blk.opt_to_lr_fp16[idx][0].get(), -4.0, 4.0, 0.0),
            clamp_finite_fp16(blk.opt_to_lr_fp16[idx][1].get(), -4.0, 4.0, 0.0),
        ];
    }
    for l in 0..4 {
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
    let w_count = num_lanes * 12;
    unpack_w_ro_into(&w_ro_and_mods[..w_count], num_lanes, &mut snap.readouts);
    for i in 0..8 {
        snap.ro_gains_fp16[i] = fp16_sanitized(w_ro_and_mods[w_count + i].get());
    }
    snap
}

fn decode_lite_crossbar_snapshot(
    ro: &VarProReadoutLiteWire,
    rho_tail: &[F32; 32],
) -> TierCrossbarSnapshot {
    let mut snap = TierCrossbarSnapshot::default();
    let f_gains = [
        ro.pre_cab_u_gain[0].get(),
        ro.pre_cab_u_gain[1].get(),
        ro.pre_cab_hp_gain[0].get(),
        ro.pre_cab_hp_gain[1].get(),
        ro.direct_s2_u_gain[0].get(),
        ro.direct_s2_u_gain[1].get(),
        ro.direct_s2_hp_gain[0].get(),
        ro.direct_s2_hp_gain[1].get(),
    ];
    for (i, &g) in f_gains.iter().enumerate() {
        snap.ro_gains_fp16[i] = f32_to_fp16(clamp_finite_f32(g, -FP16_ABS_MAX, FP16_ABS_MAX, 0.0));
    }
    unpack_w_ro_into(&ro.w_ro_fp16[..192], 16, &mut snap.readouts);
    for g in 0..4 {
        for l in 0..4 {
            snap.op_p_to_group[g][l] = clamp_finite_fp16(
                ro.stage_latent_mods_fp16[80 + g * 4 + l].get(),
                -8.0,
                8.0,
                0.0,
            );
            snap.phasor_to_group[g][l] =
                clamp_finite_fp16(ro.w_ro_fp16[192 + g * 4 + l].get(), -8.0, 8.0, 0.0);
        }
    }
    for idx in 0..4 {
        snap.opt_to_lr[idx] = [
            f32_to_fp16(clamp_finite_f32(
                rho_tail[24 + idx * 2].get(),
                -4.0,
                4.0,
                0.0,
            )),
            f32_to_fp16(clamp_finite_f32(
                rho_tail[24 + idx * 2 + 1].get(),
                -4.0,
                4.0,
                0.0,
            )),
        ];
    }
    snap
}

fn copy_base_fir_and_kautz<B: GraphBackend>(view: &DnamBinView<'_>, slab: &mut SonaHotSlab<'_, B>) {
    let mut base_fir_l = [0.0_f32; 128];
    let mut base_fir_r = [0.0_f32; 128];
    let cold = slab.cold_mut();
    for (i, tap) in view.arena0.cabinet_fir_lite.fir_l.iter().enumerate() {
        base_fir_l[i] = clamp_finite_f32(tap.get(), -16.0, 16.0, 0.0);
        base_fir_r[i] = clamp_finite_f32(
            view.arena0.cabinet_fir_lite.fir_r[i].get(),
            -16.0,
            16.0,
            0.0,
        );
    }
    cold.wire_kautz[0] = view.arena0.cabinet_kautz_lite[0];
    cold.wire_kautz[1] = view.arena0.cabinet_kautz_lite[1];
    if let Some(a1) = view.arena1 {
        for i in 0..16 {
            base_fir_l[32 + i] =
                clamp_finite_f32(a1.cabinet_fir_full.fir_l[i].get(), -16.0, 16.0, 0.0);
            base_fir_r[32 + i] =
                clamp_finite_f32(a1.cabinet_fir_full.fir_r[i].get(), -16.0, 16.0, 0.0);
        }
        cold.wire_kautz[2] = a1.cabinet_kautz_full;
    }
    if let Some(a2) = view.arena2 {
        for i in 0..48 {
            base_fir_l[48 + i] =
                clamp_finite_f32(a2.cabinet_fir_pro.fir_l[i].get(), -16.0, 16.0, 0.0);
            base_fir_r[48 + i] =
                clamp_finite_f32(a2.cabinet_fir_pro.fir_r[i].get(), -16.0, 16.0, 0.0);
        }
        for b in 0..3 {
            cold.wire_kautz[3 + b] = a2.cabinet_kautz_pro[b];
        }
    }
    if let Some(a3) = view.arena3 {
        for i in 0..32 {
            base_fir_l[96 + i] =
                clamp_finite_f32(a3.cabinet_fir_ultra.fir_l[i].get(), -16.0, 16.0, 0.0);
            base_fir_r[96 + i] =
                clamp_finite_f32(a3.cabinet_fir_ultra.fir_r[i].get(), -16.0, 16.0, 0.0);
        }
        for b in 0..2 {
            cold.wire_kautz[6 + b] = a3.cabinet_kautz_ultra[b];
        }
    }
    let tap_counts = [32_usize, 48, 96, 128];
    let bank_counts = [2_usize, 3, 6, 8];
    for t in 0..=view.retained_tier.index() {
        let tc = tap_counts[t];
        cold.tier_fir_l[t][..tc].copy_from_slice(&base_fir_l[..tc]);
        cold.tier_fir_r[t][..tc].copy_from_slice(&base_fir_r[..tc]);
        for b in 0..bank_counts[t] {
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
    let apply_ov = (slab.cold().meta_flags & META_CABINET_PREFIX_OVERRIDES) != 0;

    let cold = slab.cold_mut();
    cold.tier_crossbars[0] = decode_lite_crossbar_snapshot(
        &view.arena0.varpro_readout_lite,
        &view.arena0.control_bus_lite.rho_head_and_ctrl_tail,
    );
    if let Some(a1) = view.arena1 {
        let ro: &VarProReadoutFullWire = &a1.varpro_readout_full;
        cold.tier_crossbars[1] =
            decode_crossbar_block_snapshot(&a1.crossbar_full, &ro.w_ro_and_mods_fp16, 24);
        if apply_ov {
            let ov = PrefixOverrideFullWire::ref_from_bytes(&ro.prefix_override)
                .map_err(|_| VmError::InvalidArenaHeader)?;
            apply_override_slice::<16, 16, 4, 4>(
                cold,
                1,
                (&ov.fir_l, &ov.fir_r),
                (&ov.c_re_l, &ov.c_im_l, &ov.c_re_r, &ov.c_im_r),
                &ro.w_ro_and_mods_fp16[296..],
            );
        }
    }
    if let Some(a2) = view.arena2 {
        let ro: &VarProReadoutProWire = &a2.varpro_readout_pro;
        cold.tier_crossbars[2] =
            decode_crossbar_block_snapshot(&a2.crossbar_pro, &ro.w_ro_and_mods_fp16, 32);
        if apply_ov {
            let ov = PrefixOverrideProWire::ref_from_bytes(&ro.prefix_override)
                .map_err(|_| VmError::InvalidArenaHeader)?;
            apply_override_slice::<32, 16, 8, 4>(
                cold,
                2,
                (&ov.fir_l, &ov.fir_r),
                (&ov.c_re_l, &ov.c_im_l, &ov.c_re_r, &ov.c_im_r),
                &ro.w_ro_and_mods_fp16[392..],
            );
        }
    }
    if let Some(a3) = view.arena3 {
        let ro: &VarProReadoutUltraWire = &a3.varpro_readout_ultra;
        cold.tier_crossbars[3] =
            decode_crossbar_block_snapshot(&a3.crossbar_ultra, &ro.w_ro_and_mods_fp16, 40);
        if apply_ov {
            let ov = PrefixOverrideUltraWire::ref_from_bytes(&ro.prefix_override)
                .map_err(|_| VmError::InvalidArenaHeader)?;
            apply_override_slice::<48, 48, 12, 12>(
                cold,
                3,
                (&ov.fir_l, &ov.fir_r),
                (&ov.c_re_l, &ov.c_im_l, &ov.c_re_r, &ov.c_im_r),
                &ro.w_ro_and_mods_fp16[488..],
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
            compute_tier_os2x_flags(META_TIER_OS_EXPLICIT | 0b1010, 1),
            [false, true, false, true]
        );
        assert_eq!(
            compute_tier_os2x_flags(META_OS2X, 1),
            [false, true, true, true]
        );
        assert_eq!(compute_tier_os2x_flags(0, 1), [false, false, false, false]);
    }

    #[test]
    fn test_wire_decode_clamps_and_interleaved_override_residues() {
        use crate::math::ScalarBackend;
        use zerocopy::FromZeros;

        let mut gw = SonaGroupWire::new_zeroed();
        gw.svf_g[0] = F32::new(f32::NAN);
        gw.lru_kappa[0] = F32::new(12.0);
        gw.adaa_drive[0] = F32::new(-12.5);
        gw.lru_bias_drive[0] = F32::new(-25.0);
        gw.sag_depth_fp16[0] = U16::new(f32_to_fp16(-6.0));
        let (hg, _) = decode_group_wire(&gw);
        assert_eq!(hg.svf_g[0], 0.25);
        assert_eq!(fp16_to_f32(hg.lru_kappa_fp16[0]), 1.0);
        assert_eq!(fp16_to_f32(hg.adaa_drive_fp16[0]), -12.5);
        assert_eq!(fp16_to_f32(hg.lru_bias_drive_fp16[0]), -16.0);
        assert_eq!(fp16_to_f32(hg.sag_depth_fp16[0]), -6.0);

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
