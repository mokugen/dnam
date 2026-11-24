//! Tier-nested `LinearReadout` + `PrefixOverride` wire serialization for `Arena0..Arena3`
//! (`ref[container.dnam.vm-binary]`, `ref[container.dnam.tier-prefix]`).

use core::mem::{align_of, size_of};

use dnam_kernels::fp16_to_f32;
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    LinearReadoutFullWire, LinearReadoutLiteWire, LinearReadoutProWire, LinearReadoutUltraWire,
};
use zerocopy::byteorder::little_endian::{F32, U16};
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::super::super::model::{SonaModelF64, TierReadoutPrefixF64};
use super::blocks::{fp16_clamped, sanitize_clamp_f32};

pub(super) const LITE_OPT_TO_LR_OFFSET: usize = 24;
const STAGE_LATENT_OP_P_TO_CORE_OFFSET: usize = 8 + 5 * 8 + 2 * (4 * 4);
const _: () = assert!(STAGE_LATENT_OP_P_TO_CORE_OFFSET == 80);

pub(in super::super) struct PrefixOverrideWireDecode;

impl PrefixOverrideWireDecode {
    pub(in super::super) fn full_from_bytes(bytes: &[u8]) -> Option<&PrefixOverrideFullEncodeWire> {
        PrefixOverrideFullEncodeWire::ref_from_prefix(bytes)
            .ok()
            .map(|(v, _)| v)
    }

    pub(in super::super) fn pro_from_bytes(bytes: &[u8]) -> Option<&PrefixOverrideProEncodeWire> {
        PrefixOverrideProEncodeWire::ref_from_prefix(bytes)
            .ok()
            .map(|(v, _)| v)
    }

    pub(in super::super) fn ultra_from_bytes(
        bytes: &[u8],
    ) -> Option<&PrefixOverrideUltraEncodeWire> {
        PrefixOverrideUltraEncodeWire::ref_from_prefix(bytes)
            .ok()
            .map(|(v, _)| v)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub(in super::super) struct PrefixOverrideFullEncodeWire {
    pub(in super::super) fir_l: [F32; 16],
    pub(in super::super) fir_r: [F32; 16],
    pub(in super::super) c_re_l: [F32; 4],
    pub(in super::super) c_im_l: [F32; 4],
    pub(in super::super) c_re_r: [F32; 4],
    pub(in super::super) c_im_r: [F32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub(in super::super) struct PrefixOverrideProEncodeWire {
    pub(in super::super) fir_l: [F32; 32],
    pub(in super::super) fir_r: [F32; 32],
    pub(in super::super) c_re_l: [F32; 8],
    pub(in super::super) c_im_l: [F32; 8],
    pub(in super::super) c_re_r: [F32; 8],
    pub(in super::super) c_im_r: [F32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub(in super::super) struct PrefixOverrideUltraEncodeWire {
    pub(in super::super) fir_l: [F32; 48],
    pub(in super::super) fir_r: [F32; 48],
    pub(in super::super) c_re_l: [F32; 12],
    pub(in super::super) c_im_l: [F32; 12],
    pub(in super::super) c_re_r: [F32; 12],
    pub(in super::super) c_im_r: [F32; 12],
}

const _: () = assert!(size_of::<PrefixOverrideFullEncodeWire>() == 192);
const _: () = assert!(align_of::<PrefixOverrideFullEncodeWire>() == 1);
const _: () = assert!(size_of::<PrefixOverrideProEncodeWire>() == 384);
const _: () = assert!(align_of::<PrefixOverrideProEncodeWire>() == 1);
const _: () = assert!(size_of::<PrefixOverrideUltraEncodeWire>() == 576);
const _: () = assert!(align_of::<PrefixOverrideUltraEncodeWire>() == 1);

fn write_w_ro_fp16(w_ro: &[[f64; 12]], dst: &mut [U16], lanes: usize) {
    for l in 0..lanes {
        let base = l * 12;
        for col in 0..12 {
            dst[base + col] = fp16_clamped(w_ro[l][col], -64.0, 64.0, 0.0);
        }
    }
}

fn write_scalar_ro_gains_fp16(
    pre_u: [f64; 2],
    pre_hp: [f64; 2],
    dir_u: [f64; 2],
    dir_hp: [f64; 2],
    dst: &mut [U16],
) {
    let gains = [
        pre_u[0], pre_u[1], pre_hp[0], pre_hp[1], dir_u[0], dir_u[1], dir_hp[0], dir_hp[1],
    ];
    for (i, &g) in gains.iter().enumerate() {
        dst[i] = fp16_clamped(g, -64.0, 64.0, 0.0);
    }
}

fn pack_kautz_b_and_c(
    src: Option<&TierReadoutPrefixF64>,
    model: &SonaModelF64,
    pair_idx: usize,
) -> (f64, f64, f64, f64, f64) {
    if let Some(p) = src {
        (
            p.kautz_b_gamma[pair_idx],
            p.kautz_c_re_l[pair_idx],
            p.kautz_c_im_l[pair_idx],
            p.kautz_c_re_r[pair_idx],
            p.kautz_c_im_r[pair_idx],
        )
    } else {
        let kp = &model.kautz[pair_idx];
        (kp.b_gamma, kp.c_re_l, kp.c_im_l, kp.c_re_r, kp.c_im_r)
    }
}

/// Serializes Lite/Full/Pro/Ultra readout matrices, residual gains, and optional prefix overrides.
pub(super) fn pack_tier_readouts(
    model: &SonaModelF64,
    emit_prefix_overrides: bool,
    a0: &mut Arena0LiteWire,
    a1: &mut Arena1FullDeltaWire,
    a2: &mut Arena2ProDeltaWire,
    a3: &mut Arena3UltraDeltaWire,
) {
    let snap_fp16 = |v: f64| F32::new(fp16_to_f32(fp16_clamped(v, -64.0, 64.0, 0.0).get()));
    let lite_src = model.tier_overrides[0].as_deref();
    let mut ro_lite = LinearReadoutLiteWire::new_zeroed();
    ro_lite.pre_conv_u_gain = lite_src
        .map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain)
        .map(snap_fp16);
    ro_lite.pre_conv_hp_gain = lite_src
        .map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain)
        .map(snap_fp16);
    ro_lite.direct_s2_u_gain = lite_src
        .map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain)
        .map(snap_fp16);
    ro_lite.direct_s2_hp_gain = lite_src
        .map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain)
        .map(snap_fp16);
    let lite_w_ro = lite_src.map_or(&model.w_ro, |p| &p.w_ro);
    write_w_ro_fp16(lite_w_ro, &mut ro_lite.w_ro_fp16[..192], 16);
    model
        .stage_latent_mods
        .pack_lite_into_wire(&mut ro_lite.stage_latent_mods_fp16);
    for q in 0..4 {
        for l in 0..4 {
            ro_lite.stage_latent_mods_fp16[STAGE_LATENT_OP_P_TO_CORE_OFFSET + q * 4 + l] =
                fp16_clamped(model.op_p_to_core[q][l], -8.0, 8.0, 0.0);
            ro_lite.w_ro_fp16[192 + q * 4 + l] =
                fp16_clamped(model.phasor_to_core[q][l], -8.0, 8.0, 0.0);
        }
    }
    a0.linear_readout_lite = ro_lite;

    // Tier 1: Full (prefix 16/4 + delta 16..32 FIR, b_gamma[0..8], c[4..8]).
    let full_src = model.tier_overrides[1].as_deref();
    let mut ro_full = LinearReadoutFullWire::new_zeroed();
    write_w_ro_fp16(
        full_src.map_or(&model.w_ro, |p| &p.w_ro),
        &mut ro_full.w_ro_and_mods_fp16[..288],
        24,
    );
    write_scalar_ro_gains_fp16(
        full_src.map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain),
        full_src.map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain),
        full_src.map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain),
        full_src.map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain),
        &mut ro_full.w_ro_and_mods_fp16[288..296],
    );
    if emit_prefix_overrides {
        let fir_l = full_src.map_or(&model.fir_l, |p| &p.fir_l);
        let fir_r = full_src.map_or(&model.fir_r, |p| &p.fir_r);
        let mut pfx = PrefixOverrideFullEncodeWire::new_zeroed();
        for i in 0..16 {
            pfx.fir_l[i] = F32::new(sanitize_clamp_f32(fir_l[i], -16.0, 16.0, 0.0));
            pfx.fir_r[i] = F32::new(sanitize_clamp_f32(fir_r[i], -16.0, 16.0, 0.0));
            ro_full.w_ro_and_mods_fp16[296 + i] = fp16_clamped(fir_l[16 + i], -16.0, 16.0, 0.0);
            ro_full.w_ro_and_mods_fp16[312 + i] = fp16_clamped(fir_r[16 + i], -16.0, 16.0, 0.0);
        }
        for m in 0..4 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(full_src, model, m);
            pfx.c_re_l[m] = F32::new(sanitize_clamp_f32(cl, -8.0, 8.0, 0.0));
            pfx.c_im_l[m] = F32::new(sanitize_clamp_f32(cil, -8.0, 8.0, 0.0));
            pfx.c_re_r[m] = F32::new(sanitize_clamp_f32(cr, -8.0, 8.0, 0.0));
            pfx.c_im_r[m] = F32::new(sanitize_clamp_f32(cir, -8.0, 8.0, 0.0));
        }
        for m in 0..8 {
            let (bg, _, _, _, _) = pack_kautz_b_and_c(full_src, model, m);
            ro_full.w_ro_and_mods_fp16[328 + m] = fp16_clamped(bg, 0.0, 8.0, 0.0);
        }
        for idx in 0..4 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(full_src, model, 4 + idx);
            let off = 336 + idx * 4;
            ro_full.w_ro_and_mods_fp16[off] = fp16_clamped(cl, -8.0, 8.0, 0.0);
            ro_full.w_ro_and_mods_fp16[off + 1] = fp16_clamped(cil, -8.0, 8.0, 0.0);
            ro_full.w_ro_and_mods_fp16[off + 2] = fp16_clamped(cr, -8.0, 8.0, 0.0);
            ro_full.w_ro_and_mods_fp16[off + 3] = fp16_clamped(cir, -8.0, 8.0, 0.0);
        }
        let _ = pfx.write_to_prefix(&mut ro_full.prefix_override);
    }
    a1.linear_readout_full = ro_full;

    // Tier 2: Pro (prefix 32/8 + delta 32..48 FIR, b_gamma[0..12], c[8..12]).
    let pro_src = model.tier_overrides[2].as_deref();
    let mut ro_pro = LinearReadoutProWire::new_zeroed();
    write_w_ro_fp16(
        pro_src.map_or(&model.w_ro, |p| &p.w_ro),
        &mut ro_pro.w_ro_and_mods_fp16[..384],
        32,
    );
    write_scalar_ro_gains_fp16(
        pro_src.map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain),
        pro_src.map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain),
        pro_src.map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain),
        pro_src.map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain),
        &mut ro_pro.w_ro_and_mods_fp16[384..392],
    );
    if emit_prefix_overrides {
        let fir_l = pro_src.map_or(&model.fir_l, |p| &p.fir_l);
        let fir_r = pro_src.map_or(&model.fir_r, |p| &p.fir_r);
        let mut pfx = PrefixOverrideProEncodeWire::new_zeroed();
        for i in 0..32 {
            pfx.fir_l[i] = F32::new(sanitize_clamp_f32(fir_l[i], -16.0, 16.0, 0.0));
            pfx.fir_r[i] = F32::new(sanitize_clamp_f32(fir_r[i], -16.0, 16.0, 0.0));
        }
        for i in 0..16 {
            ro_pro.w_ro_and_mods_fp16[392 + i] = fp16_clamped(fir_l[32 + i], -16.0, 16.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[408 + i] = fp16_clamped(fir_r[32 + i], -16.0, 16.0, 0.0);
        }
        for m in 0..8 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(pro_src, model, m);
            pfx.c_re_l[m] = F32::new(sanitize_clamp_f32(cl, -8.0, 8.0, 0.0));
            pfx.c_im_l[m] = F32::new(sanitize_clamp_f32(cil, -8.0, 8.0, 0.0));
            pfx.c_re_r[m] = F32::new(sanitize_clamp_f32(cr, -8.0, 8.0, 0.0));
            pfx.c_im_r[m] = F32::new(sanitize_clamp_f32(cir, -8.0, 8.0, 0.0));
        }
        for m in 0..12 {
            let (bg, _, _, _, _) = pack_kautz_b_and_c(pro_src, model, m);
            ro_pro.w_ro_and_mods_fp16[424 + m] = fp16_clamped(bg, 0.0, 8.0, 0.0);
        }
        for idx in 0..4 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(pro_src, model, 8 + idx);
            let off = 436 + idx * 4;
            ro_pro.w_ro_and_mods_fp16[off] = fp16_clamped(cl, -8.0, 8.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[off + 1] = fp16_clamped(cil, -8.0, 8.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[off + 2] = fp16_clamped(cr, -8.0, 8.0, 0.0);
            ro_pro.w_ro_and_mods_fp16[off + 3] = fp16_clamped(cir, -8.0, 8.0, 0.0);
        }
        let _ = pfx.write_to_prefix(&mut ro_pro.prefix_override);
    }
    a2.linear_readout_pro = ro_pro;

    // Tier 3: Ultra (prefix 48/12 + delta 48..96 FIR, b_gamma[0..24], c[12..24]).
    let ultra_src = model.tier_overrides[3].as_deref();
    let mut ro_ultra = LinearReadoutUltraWire::new_zeroed();
    write_w_ro_fp16(
        ultra_src.map_or(&model.w_ro, |p| &p.w_ro),
        &mut ro_ultra.w_ro_and_mods_fp16[..480],
        40,
    );
    write_scalar_ro_gains_fp16(
        ultra_src.map_or(model.pre_conv_u_gain, |p| p.pre_conv_u_gain),
        ultra_src.map_or(model.pre_conv_hp_gain, |p| p.pre_conv_hp_gain),
        ultra_src.map_or(model.direct_s2_u_gain, |p| p.direct_s2_u_gain),
        ultra_src.map_or(model.direct_s2_hp_gain, |p| p.direct_s2_hp_gain),
        &mut ro_ultra.w_ro_and_mods_fp16[480..488],
    );
    if emit_prefix_overrides {
        let fir_l = ultra_src.map_or(&model.fir_l, |p| &p.fir_l);
        let fir_r = ultra_src.map_or(&model.fir_r, |p| &p.fir_r);
        let mut pfx = PrefixOverrideUltraEncodeWire::new_zeroed();
        for i in 0..48 {
            pfx.fir_l[i] = F32::new(sanitize_clamp_f32(fir_l[i], -16.0, 16.0, 0.0));
            pfx.fir_r[i] = F32::new(sanitize_clamp_f32(fir_r[i], -16.0, 16.0, 0.0));
            ro_ultra.w_ro_and_mods_fp16[488 + i] = fp16_clamped(fir_l[48 + i], -16.0, 16.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[536 + i] = fp16_clamped(fir_r[48 + i], -16.0, 16.0, 0.0);
        }
        for m in 0..12 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(ultra_src, model, m);
            pfx.c_re_l[m] = F32::new(sanitize_clamp_f32(cl, -8.0, 8.0, 0.0));
            pfx.c_im_l[m] = F32::new(sanitize_clamp_f32(cil, -8.0, 8.0, 0.0));
            pfx.c_re_r[m] = F32::new(sanitize_clamp_f32(cr, -8.0, 8.0, 0.0));
            pfx.c_im_r[m] = F32::new(sanitize_clamp_f32(cir, -8.0, 8.0, 0.0));
        }
        for m in 0..24 {
            let (bg, _, _, _, _) = pack_kautz_b_and_c(ultra_src, model, m);
            ro_ultra.w_ro_and_mods_fp16[584 + m] = fp16_clamped(bg, 0.0, 8.0, 0.0);
        }
        for idx in 0..12 {
            let (_, cl, cil, cr, cir) = pack_kautz_b_and_c(ultra_src, model, 12 + idx);
            let off = 608 + idx * 4;
            ro_ultra.w_ro_and_mods_fp16[off] = fp16_clamped(cl, -8.0, 8.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[off + 1] = fp16_clamped(cil, -8.0, 8.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[off + 2] = fp16_clamped(cr, -8.0, 8.0, 0.0);
            ro_ultra.w_ro_and_mods_fp16[off + 3] = fp16_clamped(cir, -8.0, 8.0, 0.0);
        }
        let _ = pfx.write_to_prefix(&mut ro_ultra.prefix_override);
    }
    a3.linear_readout_ultra = ro_ultra;
}
