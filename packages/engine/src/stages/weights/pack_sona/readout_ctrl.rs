//! `SONA` wire codecs for Stage III FIR/Kautz, `VarPro` readouts, `Stage IV` quads, and `ControlBus`
//! (`stages/weights/pack_sona/readout_ctrl.rs`, zero manual endian slicing).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};
use zerocopy::byteorder::little_endian::{F32, U16, U32};

use crate::stages::sona_core::model::{
    MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_TOTAL_SONA_LANES, NUM_SONA_MACRO_KNOBS,
    SonaControlBusParams, SonaMatryoshkaModel, SonaTierReadout,
};
use crate::stages::stage4_tank::v3_fdn::{Stage4ExtFormat, Stage4QuadParams, V3Stage4TankParams};
use crate::stages::weights::{
    Arena0ControlBusWire, Arena3ControlBusWire, Cp1InlineDirWire, KautzSoaBlockWire,
    Stage3PrefixOverrideFullWire, Stage3PrefixOverrideProWire, Stage3PrefixOverrideUltraWire,
    Stage4QuadWire, VarProReadoutFullWire, VarProReadoutLiteWire, VarProReadoutProWire,
    VarProReadoutUltraWire,
};
use crate::types::Tier;

#[inline]
fn encode_ro_lane_block(ro: &SonaTierReadout, lane_count: usize, dst: &mut [U16]) {
    let n = lane_count.min(MAX_TOTAL_SONA_LANES);
    for l in 0..n {
        let base = l * 12;
        if base + 11 >= dst.len() {
            break;
        }
        let vals = [
            ro.pre_cab_w_odd[0][l],
            ro.pre_cab_w_odd[1][l],
            ro.pre_cab_w_even[0][l],
            ro.pre_cab_w_even[1][l],
            ro.pre_cab_w_lru[0][l],
            ro.pre_cab_w_lru[1][l],
            ro.direct_w_odd[0][l],
            ro.direct_w_odd[1][l],
            ro.direct_w_even[0][l],
            ro.direct_w_even[1][l],
            ro.direct_w_lru[0][l],
            ro.direct_w_lru[1][l],
        ];
        for (i, v) in vals.iter().enumerate() {
            dst[base + i] = U16::new(f32_to_fp16(*v));
        }
    }
}

#[inline]
fn decode_ro_lane_block(src: &[U16], lane_count: usize, ro: &mut SonaTierReadout) {
    let n = lane_count.min(MAX_TOTAL_SONA_LANES);
    for l in 0..n {
        let base = l * 12;
        if base + 11 >= src.len() {
            break;
        }
        ro.pre_cab_w_odd[0][l] = fp16_to_f32(src[base].get());
        ro.pre_cab_w_odd[1][l] = fp16_to_f32(src[base + 1].get());
        ro.pre_cab_w_even[0][l] = fp16_to_f32(src[base + 2].get());
        ro.pre_cab_w_even[1][l] = fp16_to_f32(src[base + 3].get());
        ro.pre_cab_w_lru[0][l] = fp16_to_f32(src[base + 4].get());
        ro.pre_cab_w_lru[1][l] = fp16_to_f32(src[base + 5].get());
        ro.direct_w_odd[0][l] = fp16_to_f32(src[base + 6].get());
        ro.direct_w_odd[1][l] = fp16_to_f32(src[base + 7].get());
        ro.direct_w_even[0][l] = fp16_to_f32(src[base + 8].get());
        ro.direct_w_even[1][l] = fp16_to_f32(src[base + 9].get());
        ro.direct_w_lru[0][l] = fp16_to_f32(src[base + 10].get());
        ro.direct_w_lru[1][l] = fp16_to_f32(src[base + 11].get());
    }
}

#[inline]
fn encode_u_hp_gains_fp16(ro: &SonaTierReadout, fp16_dst: &mut [U16]) {
    let vals = [
        ro.pre_cab_u_gain[0],
        ro.pre_cab_u_gain[1],
        ro.pre_cab_hp_gain[0],
        ro.pre_cab_hp_gain[1],
        ro.direct_s2_u_gain[0],
        ro.direct_s2_u_gain[1],
        ro.direct_s2_hp_gain[0],
        ro.direct_s2_hp_gain[1],
    ];
    if fp16_dst.len() >= 8 {
        for (i, v) in vals.iter().enumerate() {
            fp16_dst[i] = U16::new(f32_to_fp16(*v));
        }
    }
}

#[inline]
fn decode_u_hp_gains_fp16(fp16_src: &[U16], ro: &mut SonaTierReadout) {
    if fp16_src.len() < 8 {
        return;
    }
    ro.pre_cab_u_gain = [
        fp16_to_f32(fp16_src[0].get()),
        fp16_to_f32(fp16_src[1].get()),
    ];
    ro.pre_cab_hp_gain = [
        fp16_to_f32(fp16_src[2].get()),
        fp16_to_f32(fp16_src[3].get()),
    ];
    ro.direct_s2_u_gain = [
        fp16_to_f32(fp16_src[4].get()),
        fp16_to_f32(fp16_src[5].get()),
    ];
    ro.direct_s2_hp_gain = [
        fp16_to_f32(fp16_src[6].get()),
        fp16_to_f32(fp16_src[7].get()),
    ];
}

pub(super) fn pack_kautz_blocks(
    poles: &[(f32, f32); MAX_KAUTZ_POLES],
    ro: &SonaTierReadout,
    pair_offset: usize,
    blocks: &mut [KautzSoaBlockWire],
) {
    for (blk_idx, dst) in blocks.iter_mut().enumerate() {
        let base = pair_offset + blk_idx * 4;
        for lane in 0..4 {
            let m = base + lane;
            let (r, theta) = poles[m];
            dst.r_low[lane] = F32::new(r);
            dst.dr[lane] = F32::new(0.0);
            dst.theta[lane] = F32::new(theta);
            // Preserve `kautz_b_gamma[m] >= 0.0` verbatim (never coerce 0.0 -> cold_b_gamma).
            let bg = if ro.kautz_b_gamma[m].is_finite() {
                ro.kautz_b_gamma[m].max(0.0)
            } else {
                0.0
            };
            dst.b_gamma[lane] = F32::new(bg);
            dst.c_re_l[lane] = F32::new(ro.kautz_c_l[m].0);
            dst.c_im_l[lane] = F32::new(ro.kautz_c_l[m].1);
            dst.c_re_r[lane] = F32::new(ro.kautz_c_r[m].0);
            dst.c_im_r[lane] = F32::new(ro.kautz_c_r[m].1);
        }
    }
}

pub(super) fn unpack_kautz_blocks(
    blocks: &[KautzSoaBlockWire],
    pair_offset: usize,
    poles: &mut [(f32, f32); MAX_KAUTZ_POLES],
    ro: &mut SonaTierReadout,
) {
    for (blk_idx, src) in blocks.iter().enumerate() {
        let base = pair_offset + blk_idx * 4;
        for lane in 0..4 {
            let m = base + lane;
            if m >= MAX_KAUTZ_POLES {
                break;
            }
            let r = src.r_low[lane].get() + 0.5 * src.dr[lane].get();
            let th = src.theta[lane].get();
            if r > 0.0 {
                poles[m] = (r, th);
            }
            let bg = src.b_gamma[lane].get();
            ro.kautz_b_gamma[m] = if bg.is_finite() { bg.max(0.0) } else { 0.0 };
            ro.kautz_c_l[m] = (src.c_re_l[lane].get(), src.c_im_l[lane].get());
            ro.kautz_c_r[m] = (src.c_re_r[lane].get(), src.c_im_r[lane].get());
        }
    }
}

pub(super) fn pack_stage4_quad_wire(q: &Stage4QuadParams) -> Stage4QuadWire {
    let mut qp = *q;
    qp.clamp_in_place();
    Stage4QuadWire {
        delay_lengths: qp.delay_lengths.map(|d| U32::new(d as u32)),
        tap_gains: qp.tap_gains.map(F32::new),
        allpass_delays: qp.allpass_delays.map(|d| U32::new(d as u32)),
        allpass_g: qp.allpass_g.map(F32::new),
        g_loop: F32::new(qp.g_loop),
        sigma_knee: F32::new(qp.sigma_knee),
        base_psi: F32::new(qp.base_psi),
        inject_gain: F32::new(qp.inject_gain),
        dry_mix: F32::new(qp.dry_mix),
        wet_mix: F32::new(qp.wet_mix),
        r_dc: F32::new(qp.r_dc),
        vca_thresh: F32::new(qp.vca_thresh),
        vca_kappa: F32::new(qp.vca_kappa),
        vca_min_gain: F32::new(qp.vca_min_gain),
        vca_att: F32::new(qp.vca_att),
        vca_rel: F32::new(qp.vca_rel),
        lfo_rate_hz: F32::new(qp.lfo_rate_hz),
        lfo_depth_samples: F32::new(qp.lfo_depth_samples),
        lfo_stereo_phase: F32::new(qp.lfo_stereo_phase),
        ext_format: U32::new(qp.ext_format.as_u32()),
    }
}

pub(super) fn unpack_stage4_quad_wire(w: &Stage4QuadWire) -> Stage4QuadParams {
    let mut q = Stage4QuadParams {
        delay_lengths: w.delay_lengths.map(|d| d.get() as usize),
        tap_gains: w.tap_gains.map(F32::get),
        allpass_delays: w.allpass_delays.map(|d| d.get() as usize),
        allpass_g: w.allpass_g.map(F32::get),
        g_loop: w.g_loop.get(),
        sigma_knee: w.sigma_knee.get(),
        base_psi: w.base_psi.get(),
        inject_gain: w.inject_gain.get(),
        dry_mix: w.dry_mix.get(),
        wet_mix: w.wet_mix.get(),
        r_dc: w.r_dc.get(),
        vca_thresh: w.vca_thresh.get(),
        vca_kappa: w.vca_kappa.get(),
        vca_min_gain: w.vca_min_gain.get(),
        vca_att: w.vca_att.get(),
        vca_rel: w.vca_rel.get(),
        lfo_rate_hz: w.lfo_rate_hz.get(),
        lfo_depth_samples: w.lfo_depth_samples.get(),
        lfo_stereo_phase: w.lfo_stereo_phase.get(),
        ext_format: Stage4ExtFormat::from_u32(w.ext_format.get()),
    };
    q.clamp_in_place();
    q
}

pub(super) fn rebuild_tier_stage4(
    tier: Tier,
    q0: Stage4QuadParams,
    q1: Option<Stage4QuadParams>,
    q2: Option<Stage4QuadParams>,
    q3: Option<Stage4QuadParams>,
) -> V3Stage4TankParams {
    let mut s4 = V3Stage4TankParams::identity_for_tier(tier);
    s4.quads[0] = q0;
    if let Some(v) = q1 {
        s4.quads[1] = v;
    }
    if let Some(v) = q2 {
        s4.quads[2] = v;
    }
    if let Some(v) = q3 {
        s4.quads[3] = v;
    }
    s4.sync_top_from_quad0();
    s4.clamp_in_place();
    s4
}

pub(super) fn pack_readout_lite(
    ro: &SonaTierReadout,
    ctrl: &SonaControlBusParams,
    dst: &mut VarProReadoutLiteWire,
) {
    dst.pre_cab_u_gain = ro.pre_cab_u_gain.map(F32::new);
    dst.pre_cab_hp_gain = ro.pre_cab_hp_gain.map(F32::new);
    dst.direct_s2_u_gain = ro.direct_s2_u_gain.map(F32::new);
    dst.direct_s2_hp_gain = ro.direct_s2_hp_gain.map(F32::new);
    encode_ro_lane_block(ro, 16, &mut dst.w_ro_fp16[..192]);
    pack_ctrl_lite_slots(ctrl, &mut dst.stage_latent_mods_fp16[..80]);
}

pub(super) fn unpack_readout_lite(
    src: &VarProReadoutLiteWire,
    ro: &mut SonaTierReadout,
    ctrl: &mut SonaControlBusParams,
) {
    ro.pre_cab_u_gain = src.pre_cab_u_gain.map(F32::get);
    ro.pre_cab_hp_gain = src.pre_cab_hp_gain.map(F32::get);
    ro.direct_s2_u_gain = src.direct_s2_u_gain.map(F32::get);
    ro.direct_s2_hp_gain = src.direct_s2_hp_gain.map(F32::get);
    decode_ro_lane_block(&src.w_ro_fp16[..192], 16, ro);
    unpack_ctrl_lite_slots(&src.stage_latent_mods_fp16[..80], ctrl);
}

pub(super) fn pack_readout_full(ro: &SonaTierReadout, dst: &mut VarProReadoutFullWire) {
    dst.prefix_override = Stage3PrefixOverrideFullWire {
        fir_l: core::array::from_fn(|i| F32::new(ro.fir_l[i])),
        fir_r: core::array::from_fn(|i| F32::new(ro.fir_r[i])),
        c_re_l: core::array::from_fn(|m| F32::new(ro.kautz_c_l[m].0)),
        c_im_l: core::array::from_fn(|m| F32::new(ro.kautz_c_l[m].1)),
        c_re_r: core::array::from_fn(|m| F32::new(ro.kautz_c_r[m].0)),
        c_im_r: core::array::from_fn(|m| F32::new(ro.kautz_c_r[m].1)),
    };
    encode_ro_lane_block(ro, 24, &mut dst.w_ro_and_mods_fp16[..288]);
    let (fp16_slice, extra) = dst.w_ro_and_mods_fp16[288..].split_at_mut(8);
    encode_u_hp_gains_fp16(ro, fp16_slice);
    for i in 0..16 {
        extra[i] = U16::new(f32_to_fp16(ro.fir_l[16 + i]));
        extra[16 + i] = U16::new(f32_to_fp16(ro.fir_r[16 + i]));
    }
    for m in 0..8 {
        extra[32 + m] = U16::new(f32_to_fp16(ro.kautz_b_gamma[m].max(0.0)));
    }
    for m in 0..4 {
        let b = 40 + m * 4;
        extra[b] = U16::new(f32_to_fp16(ro.kautz_c_l[4 + m].0));
        extra[b + 1] = U16::new(f32_to_fp16(ro.kautz_c_l[4 + m].1));
        extra[b + 2] = U16::new(f32_to_fp16(ro.kautz_c_r[4 + m].0));
        extra[b + 3] = U16::new(f32_to_fp16(ro.kautz_c_r[4 + m].1));
    }
}

pub(super) fn unpack_readout_full(
    src: &VarProReadoutFullWire,
    ro: &mut SonaTierReadout,
    apply_prefix_overrides: bool,
) {
    decode_ro_lane_block(&src.w_ro_and_mods_fp16[..288], 24, ro);
    decode_u_hp_gains_fp16(&src.w_ro_and_mods_fp16[288..296], ro);
    if !apply_prefix_overrides {
        return;
    }
    for i in 0..16 {
        ro.fir_l[i] = src.prefix_override.fir_l[i].get();
        ro.fir_r[i] = src.prefix_override.fir_r[i].get();
    }
    for m in 0..4 {
        ro.kautz_c_l[m] = (
            src.prefix_override.c_re_l[m].get(),
            src.prefix_override.c_im_l[m].get(),
        );
        ro.kautz_c_r[m] = (
            src.prefix_override.c_re_r[m].get(),
            src.prefix_override.c_im_r[m].get(),
        );
    }
    let extra = &src.w_ro_and_mods_fp16[296..];
    for i in 0..16 {
        ro.fir_l[16 + i] = fp16_to_f32(extra[i].get());
        ro.fir_r[16 + i] = fp16_to_f32(extra[16 + i].get());
    }
    for m in 0..8 {
        ro.kautz_b_gamma[m] = fp16_to_f32(extra[32 + m].get()).max(0.0);
    }
    for m in 0..4 {
        let b = 40 + m * 4;
        ro.kautz_c_l[4 + m] = (fp16_to_f32(extra[b].get()), fp16_to_f32(extra[b + 1].get()));
        ro.kautz_c_r[4 + m] = (
            fp16_to_f32(extra[b + 2].get()),
            fp16_to_f32(extra[b + 3].get()),
        );
    }
}

pub(super) fn pack_readout_pro(ro: &SonaTierReadout, dst: &mut VarProReadoutProWire) {
    dst.prefix_override = Stage3PrefixOverrideProWire {
        fir_l: core::array::from_fn(|i| F32::new(ro.fir_l[i])),
        fir_r: core::array::from_fn(|i| F32::new(ro.fir_r[i])),
        c_re_l: core::array::from_fn(|m| F32::new(ro.kautz_c_l[m].0)),
        c_im_l: core::array::from_fn(|m| F32::new(ro.kautz_c_l[m].1)),
        c_re_r: core::array::from_fn(|m| F32::new(ro.kautz_c_r[m].0)),
        c_im_r: core::array::from_fn(|m| F32::new(ro.kautz_c_r[m].1)),
    };
    encode_ro_lane_block(ro, 32, &mut dst.w_ro_and_mods_fp16[..384]);
    let (fp16_slice, extra) = dst.w_ro_and_mods_fp16[384..].split_at_mut(8);
    encode_u_hp_gains_fp16(ro, fp16_slice);
    for i in 0..16 {
        extra[i] = U16::new(f32_to_fp16(ro.fir_l[32 + i]));
        extra[16 + i] = U16::new(f32_to_fp16(ro.fir_r[32 + i]));
    }
    for m in 0..12 {
        extra[32 + m] = U16::new(f32_to_fp16(ro.kautz_b_gamma[m].max(0.0)));
    }
    for m in 0..4 {
        let b = 44 + m * 4;
        extra[b] = U16::new(f32_to_fp16(ro.kautz_c_l[8 + m].0));
        extra[b + 1] = U16::new(f32_to_fp16(ro.kautz_c_l[8 + m].1));
        extra[b + 2] = U16::new(f32_to_fp16(ro.kautz_c_r[8 + m].0));
        extra[b + 3] = U16::new(f32_to_fp16(ro.kautz_c_r[8 + m].1));
    }
}

pub(super) fn unpack_readout_pro(
    src: &VarProReadoutProWire,
    ro: &mut SonaTierReadout,
    apply_prefix_overrides: bool,
) {
    decode_ro_lane_block(&src.w_ro_and_mods_fp16[..384], 32, ro);
    decode_u_hp_gains_fp16(&src.w_ro_and_mods_fp16[384..392], ro);
    if !apply_prefix_overrides {
        return;
    }
    for i in 0..32 {
        ro.fir_l[i] = src.prefix_override.fir_l[i].get();
        ro.fir_r[i] = src.prefix_override.fir_r[i].get();
    }
    for m in 0..8 {
        ro.kautz_c_l[m] = (
            src.prefix_override.c_re_l[m].get(),
            src.prefix_override.c_im_l[m].get(),
        );
        ro.kautz_c_r[m] = (
            src.prefix_override.c_re_r[m].get(),
            src.prefix_override.c_im_r[m].get(),
        );
    }
    let extra = &src.w_ro_and_mods_fp16[392..];
    for i in 0..16 {
        ro.fir_l[32 + i] = fp16_to_f32(extra[i].get());
        ro.fir_r[32 + i] = fp16_to_f32(extra[16 + i].get());
    }
    for m in 0..12 {
        ro.kautz_b_gamma[m] = fp16_to_f32(extra[32 + m].get()).max(0.0);
    }
    for m in 0..4 {
        let b = 44 + m * 4;
        ro.kautz_c_l[8 + m] = (fp16_to_f32(extra[b].get()), fp16_to_f32(extra[b + 1].get()));
        ro.kautz_c_r[8 + m] = (
            fp16_to_f32(extra[b + 2].get()),
            fp16_to_f32(extra[b + 3].get()),
        );
    }
}

pub(super) fn pack_readout_ultra(ro: &SonaTierReadout, dst: &mut VarProReadoutUltraWire) {
    dst.prefix_override = Stage3PrefixOverrideUltraWire {
        fir_l: core::array::from_fn(|i| F32::new(ro.fir_l[i])),
        fir_r: core::array::from_fn(|i| F32::new(ro.fir_r[i])),
        c_re_l: core::array::from_fn(|m| F32::new(ro.kautz_c_l[m].0)),
        c_im_l: core::array::from_fn(|m| F32::new(ro.kautz_c_l[m].1)),
        c_re_r: core::array::from_fn(|m| F32::new(ro.kautz_c_r[m].0)),
        c_im_r: core::array::from_fn(|m| F32::new(ro.kautz_c_r[m].1)),
    };
    encode_ro_lane_block(ro, 40, &mut dst.w_ro_and_mods_fp16[..480]);
    let (fp16_slice, extra) = dst.w_ro_and_mods_fp16[480..].split_at_mut(8);
    encode_u_hp_gains_fp16(ro, fp16_slice);
    for i in 0..48 {
        extra[i] = U16::new(f32_to_fp16(ro.fir_l[48 + i]));
        extra[48 + i] = U16::new(f32_to_fp16(ro.fir_r[48 + i]));
    }
    for m in 0..24 {
        extra[96 + m] = U16::new(f32_to_fp16(ro.kautz_b_gamma[m].max(0.0)));
    }
    for m in 0..12 {
        let b = 120 + m * 4;
        extra[b] = U16::new(f32_to_fp16(ro.kautz_c_l[12 + m].0));
        extra[b + 1] = U16::new(f32_to_fp16(ro.kautz_c_l[12 + m].1));
        extra[b + 2] = U16::new(f32_to_fp16(ro.kautz_c_r[12 + m].0));
        extra[b + 3] = U16::new(f32_to_fp16(ro.kautz_c_r[12 + m].1));
    }
}

pub(super) fn unpack_readout_ultra(
    src: &VarProReadoutUltraWire,
    ro: &mut SonaTierReadout,
    apply_prefix_overrides: bool,
) {
    decode_ro_lane_block(&src.w_ro_and_mods_fp16[..480], 40, ro);
    decode_u_hp_gains_fp16(&src.w_ro_and_mods_fp16[480..488], ro);
    if !apply_prefix_overrides {
        return;
    }
    for i in 0..48 {
        ro.fir_l[i] = src.prefix_override.fir_l[i].get();
        ro.fir_r[i] = src.prefix_override.fir_r[i].get();
    }
    for m in 0..12 {
        ro.kautz_c_l[m] = (
            src.prefix_override.c_re_l[m].get(),
            src.prefix_override.c_im_l[m].get(),
        );
        ro.kautz_c_r[m] = (
            src.prefix_override.c_re_r[m].get(),
            src.prefix_override.c_im_r[m].get(),
        );
    }
    let extra = &src.w_ro_and_mods_fp16[488..];
    for i in 0..48 {
        ro.fir_l[48 + i] = fp16_to_f32(extra[i].get());
        ro.fir_r[48 + i] = fp16_to_f32(extra[48 + i].get());
    }
    for m in 0..24 {
        ro.kautz_b_gamma[m] = fp16_to_f32(extra[96 + m].get()).max(0.0);
    }
    for m in 0..12 {
        let b = 120 + m * 4;
        ro.kautz_c_l[12 + m] = (fp16_to_f32(extra[b].get()), fp16_to_f32(extra[b + 1].get()));
        ro.kautz_c_r[12 + m] = (
            fp16_to_f32(extra[b + 2].get()),
            fp16_to_f32(extra[b + 3].get()),
        );
    }
}

fn pack_ctrl_lite_slots(ctrl: &SonaControlBusParams, dst: &mut [U16]) {
    let mut off = 0usize;
    for arr in [
        &ctrl.default_knobs,
        &ctrl.w_svf_g_oct,
        &ctrl.w_svf_k_delta,
        &ctrl.w_kautz_oct,
        &ctrl.w_kautz_damp,
        &ctrl.w_tank_psi,
    ] {
        for &v in arr {
            dst[off] = U16::new(f32_to_fp16(v));
            off += 1;
        }
    }
    for k in 0..4 {
        for b in 0..4 {
            let val = if ctrl.lite_block_scales_present {
                ctrl.w_adaa_drive_scale_lite[k][b]
            } else {
                ctrl.w_adaa_drive_scale[k][b]
            };
            dst[off] = U16::new(f32_to_fp16(val));
            off += 1;
        }
    }
    for k in 0..4 {
        for b in 0..4 {
            let val = if ctrl.lite_block_scales_present {
                ctrl.w_lru_bias_shift_lite[k][b]
            } else {
                ctrl.w_lru_bias_shift[k][b]
            };
            dst[off] = U16::new(f32_to_fp16(val));
            off += 1;
        }
    }
}

fn unpack_ctrl_lite_slots(src: &[U16], ctrl: &mut SonaControlBusParams) {
    let mut off = 0usize;
    for arr in [
        &mut ctrl.default_knobs,
        &mut ctrl.w_svf_g_oct,
        &mut ctrl.w_svf_k_delta,
        &mut ctrl.w_kautz_oct,
        &mut ctrl.w_kautz_damp,
        &mut ctrl.w_tank_psi,
    ] {
        for slot in arr.iter_mut() {
            *slot = fp16_to_f32(src[off].get());
            off += 1;
        }
    }
    for k in 0..4 {
        for b in 0..4 {
            let val = fp16_to_f32(src[off].get());
            ctrl.w_adaa_drive_scale_lite[k][b] = val;
            ctrl.w_adaa_drive_scale[k][b] = val;
            off += 1;
        }
    }
    for k in 0..4 {
        for b in 0..4 {
            let val = fp16_to_f32(src[off].get());
            ctrl.w_lru_bias_shift_lite[k][b] = val;
            ctrl.w_lru_bias_shift[k][b] = val;
            off += 1;
        }
    }
    ctrl.lite_block_scales_present = true;
}

pub(super) fn pack_arena0_ctrl_wire(ctrl: &SonaControlBusParams, dst: &mut Arena0ControlBusWire) {
    dst.lfo_dphi_train = ctrl.lfo_dphi_train.map(F32::new);
    dst.sigma_drift = F32::new(ctrl.sigma_drift);
    dst.rho_drift_train = F32::new(ctrl.rho_drift_train);
    dst.env_poles = ctrl.env_poles.map(F32::new);
    dst.dither_snr_scale = F32::new(ctrl.dither_snr_scale);
    dst.trust_radius_mh = F32::new(ctrl.trust_radius_mh);
    dst.cp1_dir = Cp1InlineDirWire {
        cp1_dir_count: U16::new(ctrl.cp1_dir_count),
        flags: U16::new(ctrl.cp1_dir_flags),
        entries: ctrl.cp1_dir_entries,
    };
    for r in 0..4 {
        for c in 0..8 {
            dst.u_int_base[r][c] = F32::new(ctrl.u_int_base[r][c]);
        }
    }
    dst.cp1_base = ctrl.cp1_base;
    for r in 0..16 {
        for c in 0..16 {
            dst.w_ctrl[r][c] = F32::new(ctrl.w_ctrl[r][c]);
        }
    }
    for i in 0..24 {
        dst.rho_head_and_ctrl_tail[i] = F32::new(ctrl.rho_head_and_ctrl_tail[i]);
    }
    dst.rho_head_and_ctrl_tail[0] = F32::new(ctrl.trust_radius_mh);
}

pub(super) fn unpack_arena0_ctrl_wire(src: &Arena0ControlBusWire, ctrl: &mut SonaControlBusParams) {
    ctrl.lfo_dphi_train = src.lfo_dphi_train.map(F32::get);
    ctrl.sigma_drift = src.sigma_drift.get();
    let rho = src.rho_drift_train.get();
    if rho > 0.0 || ctrl.sigma_drift > 0.0 {
        ctrl.rho_drift_train = rho;
    }
    let ep = src.env_poles.map(F32::get);
    if ep.iter().any(|&v| v > 0.0) {
        ctrl.env_poles = ep;
    }
    ctrl.dither_snr_scale = src.dither_snr_scale.get();
    let tr = src.trust_radius_mh.get();
    if tr > 0.0 {
        ctrl.trust_radius_mh = tr;
    }
    ctrl.cp1_dir_count = src.cp1_dir.cp1_dir_count.get();
    ctrl.cp1_dir_flags = src.cp1_dir.flags.get();
    ctrl.cp1_dir_entries = src.cp1_dir.entries;
    for r in 0..4 {
        for c in 0..8 {
            ctrl.u_int_base[r][c] = src.u_int_base[r][c].get();
        }
    }
    ctrl.cp1_base = src.cp1_base;
    for r in 0..16 {
        for c in 0..16 {
            ctrl.w_ctrl[r][c] = src.w_ctrl[r][c].get();
        }
    }
    for i in 0..24 {
        ctrl.rho_head_and_ctrl_tail[i] = src.rho_head_and_ctrl_tail[i].get();
    }
}

pub(super) fn pack_arena3_ctrl_wire(ctrl: &SonaControlBusParams, dst: &mut Arena3ControlBusWire) {
    for r in 0..4 {
        for c in 0..16 {
            dst.w_ctrl_ultra[r][c] = F32::new(ctrl.w_ctrl_ultra[r][c]);
        }
    }
    for b in 0..3 {
        for r in 0..4 {
            for c in 0..8 {
                dst.u_int_ultra[b][r][c] = F32::new(ctrl.u_int_ultra[b][r][c]);
            }
        }
    }
    dst.cp1_ultra = ctrl.cp1_ultra;
    let mut off = 0usize;
    for k in 0..NUM_SONA_MACRO_KNOBS {
        for g in 0..MAX_SONA_GROUPS {
            dst.stage_latent_ctrl_ultra_fp16[off] =
                U16::new(f32_to_fp16(ctrl.w_adaa_drive_scale[k][g]));
            off += 1;
        }
    }
    for k in 0..NUM_SONA_MACRO_KNOBS {
        for g in 0..MAX_SONA_GROUPS {
            dst.stage_latent_ctrl_ultra_fp16[off] =
                U16::new(f32_to_fp16(ctrl.w_lru_bias_shift[k][g]));
            off += 1;
        }
    }
}

pub(super) fn unpack_arena3_ctrl_wire(src: &Arena3ControlBusWire, ctrl: &mut SonaControlBusParams) {
    for r in 0..4 {
        for c in 0..16 {
            ctrl.w_ctrl_ultra[r][c] = src.w_ctrl_ultra[r][c].get();
        }
    }
    for b in 0..3 {
        for r in 0..4 {
            for c in 0..8 {
                ctrl.u_int_ultra[b][r][c] = src.u_int_ultra[b][r][c].get();
            }
        }
    }
    ctrl.cp1_ultra = src.cp1_ultra;
    let mut off = 0usize;
    for k in 0..NUM_SONA_MACRO_KNOBS {
        for g in 0..MAX_SONA_GROUPS {
            ctrl.w_adaa_drive_scale[k][g] =
                fp16_to_f32(src.stage_latent_ctrl_ultra_fp16[off].get());
            off += 1;
        }
    }
    for k in 0..NUM_SONA_MACRO_KNOBS {
        for g in 0..MAX_SONA_GROUPS {
            ctrl.w_lru_bias_shift[k][g] = fp16_to_f32(src.stage_latent_ctrl_ultra_fp16[off].get());
            off += 1;
        }
    }
}

pub(super) fn tier_pack_flags(model: &SonaMatryoshkaModel, arena_len: usize) -> u16 {
    let mut mask = 1u16;
    if arena_len >= Tier::Full.expected_arena_bytes() {
        mask |= 2;
    }
    if arena_len >= Tier::Pro.expected_arena_bytes() {
        mask |= 4;
    }
    if arena_len >= Tier::Ultra.expected_arena_bytes() {
        mask |= 8;
    }
    let _ = model;
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::FromZeros;

    #[test]
    fn test_lite_block_scales_pack_unpack_and_arena3_preserves_lite() {
        let mut ctrl = SonaControlBusParams::identity();
        ctrl.w_adaa_drive_scale[1][2] = 0.5;
        ctrl.w_lru_bias_shift[3][1] = -0.75;

        let mut ro_wire = VarProReadoutLiteWire::new_zeroed();
        pack_readout_lite(&SonaTierReadout::default(), &ctrl, &mut ro_wire);

        let mut unp_ctrl = SonaControlBusParams::identity();
        let mut ro_out = SonaTierReadout::default();
        unpack_readout_lite(&ro_wire, &mut ro_out, &mut unp_ctrl);
        assert!(unp_ctrl.lite_block_scales_present);
        assert!((unp_ctrl.w_adaa_drive_scale_lite[1][2] - 0.5).abs() < 1.0e-3);
        assert!((unp_ctrl.w_adaa_drive_scale[1][2] - 0.5).abs() < 1.0e-3);
        assert!((unp_ctrl.w_lru_bias_shift_lite[3][1] + 0.75).abs() < 1.0e-3);
        assert!((unp_ctrl.w_lru_bias_shift[3][1] + 0.75).abs() < 1.0e-3);

        let mut ultra_ctrl = SonaControlBusParams::identity();
        ultra_ctrl.w_adaa_drive_scale[1][2] = 1.25;
        ultra_ctrl.w_lru_bias_shift[3][1] = 1.5;
        let mut a3_wire = Arena3ControlBusWire::new_zeroed();
        pack_arena3_ctrl_wire(&ultra_ctrl, &mut a3_wire);

        unpack_arena3_ctrl_wire(&a3_wire, &mut unp_ctrl);
        assert!((unp_ctrl.w_adaa_drive_scale[1][2] - 1.25).abs() < 1.0e-3);
        assert!((unp_ctrl.w_lru_bias_shift[3][1] - 1.5).abs() < 1.0e-3);
        assert!((unp_ctrl.w_adaa_drive_scale_lite[1][2] - 0.5).abs() < 1.0e-3);
        assert!((unp_ctrl.w_lru_bias_shift_lite[3][1] + 0.75).abs() < 1.0e-3);
    }
}
