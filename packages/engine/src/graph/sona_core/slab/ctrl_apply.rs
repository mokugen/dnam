//! ControlBus `stage_targets[0..16]` + `z_stage` Folding, Per-Group Hoisting & ZOH Scaling
//! (`graph/sona_core/slab/ctrl_apply.rs`, `#![forbid(unsafe_code)]`).

#![forbid(unsafe_code)]

use dnam_kernels::{SonaGroupParamsStereo4, f32_to_fp16};

use super::SonaHotSlab;
use super::layout::{
    FLAG_COLD_FIR_EXT, FLAG_FREEZE, SonaGroupHot, SonaGroupReadoutHot, SonaGroupReadoutRegs,
};
use crate::container::ZohRescaledParams;
use crate::graph::GraphBackend;
use crate::graph::cabinet::{
    KautzRetuneContext, KautzStereoBank4, rescale_kautz_poles_with_zoh, retune_kautz_slice_3khz,
};
use crate::graph::sona_core::model::{
    MAX_KAUTZ_BANKS, MAX_KAUTZ_POLES, MAX_SONA_GROUPS, MAX_TANK_QUADS, NUM_SONA_BLOCKS,
    SONA_FIR_MAX_TAPS, SONA_GROUP_BLOCK_AND_LOCAL, SonaControlBusParams, SonaTierReadout,
    tier_fir_segments, tier_kautz_banks, tier_sona_groups, tier_tank_quads,
};
use crate::graph::tank::fdn::{
    PartitionedSchroederBank, SpatialTankParams, TankExtRing, TankQuadParams, V3_TANK_CAP,
    V3_TANK_INLINE_MAX_DELAY, V3_TANK_MAX_EXT_CAP, V3_TANK_MAX_LFO_DEPTH, clamp_quad_params_to_cap,
};
use crate::math::{
    MAX_LOOP_GAIN_F32, MathBackend, SimdVec4, exp2_signed_oct_f32, newton6_reciprocal_f64,
};
use crate::types::Tier;

#[inline(always)]
fn dot_slice(w: &[f32], z: &[f32; 8], len: usize) -> f32 {
    let mut acc = 0.0_f32;
    for k in 0..len {
        acc += w[k] * z[k];
    }
    acc
}

#[inline(always)]
fn cold_b_gamma_f32(r: f32) -> f32 {
    crate::math::sqrt_f64((1.0 - f64::from(r) * f64::from(r)).max(1.0e-7)) as f32
}

/// Applies `stage_targets[0..16]` + unrotated `z_stage[0..8]` across SONA I–IV (`0` RT allocs).
pub(crate) fn apply_control_bus_impl<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    stage_targets: &[f32; 16],
    z_stage: &[f32; 8],
    immediate: bool,
) {
    if !immediate
        && *stage_targets == slab.cold.last_stage_targets
        && *z_stage == slab.cold.last_z_stage
    {
        return;
    }
    slab.cold.last_stage_targets = *stage_targets;
    slab.cold.last_z_stage = *z_stage;

    let tier = slab.lite.tier;
    let is_ultra = tier == Tier::Ultra;
    let d_s = if is_ultra { 8 } else { 4 };
    let num_kautz_banks = tier_kautz_banks(tier);
    let active_quads = tier_tank_quads(tier);

    let (svf_g_oct, svf_k_delta, kautz_oct, kautz_damp, psi_delta, core_mods_active) = {
        let ctrl: &SonaControlBusParams = &slab.cold.ctrl_params;
        let g_oct =
            (stage_targets[0] + dot_slice(&ctrl.w_svf_g_oct, z_stage, d_s)).clamp(-3.0, 3.0);
        let k_del =
            (stage_targets[1] + dot_slice(&ctrl.w_svf_k_delta, z_stage, d_s)).clamp(-0.8, 1.5);
        slab.cold.ctrl_state.env_fc_scale = exp2_signed_oct_f32(g_oct);
        slab.cold.ctrl_state.env_k_delta = k_del;

        if is_ultra {
            for g in 0..MAX_SONA_GROUPS {
                let b = SONA_GROUP_BLOCK_AND_LOCAL[g].0;
                let mut a_dot = 0.0_f32;
                let mut l_dot = 0.0_f32;
                for k in 0..8 {
                    a_dot += ctrl.w_adaa_drive_scale[k][g] * z_stage[k];
                    l_dot += ctrl.w_lru_bias_shift[k][g] * z_stage[k];
                }
                let oct_g = (stage_targets[2 + b] + a_dot).clamp(-1.5, 1.5);
                let lru_g = (stage_targets[6 + b] + l_dot).clamp(-2.0, 2.0);
                slab.cold.ctrl_state.group_adaa_scale[g] = exp2_signed_oct_f32(oct_g);
                slab.cold.ctrl_state.group_lru_bias[g] = lru_g;
                if g < NUM_SONA_BLOCKS {
                    slab.cold.ctrl_state.block_adaa_scale[g] =
                        slab.cold.ctrl_state.group_adaa_scale[g];
                    slab.cold.ctrl_state.block_lru_bias[g] = lru_g;
                }
            }
        } else {
            for b in 0..NUM_SONA_BLOCKS {
                let mut a_dot = 0.0_f32;
                let mut l_dot = 0.0_f32;
                for k in 0..4 {
                    let (w_a, w_l) = if ctrl.lite_block_scales_present {
                        (
                            ctrl.w_adaa_drive_scale_lite[k][b],
                            ctrl.w_lru_bias_shift_lite[k][b],
                        )
                    } else {
                        (ctrl.w_adaa_drive_scale[k][b], ctrl.w_lru_bias_shift[k][b])
                    };
                    a_dot += w_a * z_stage[k];
                    l_dot += w_l * z_stage[k];
                }
                let oct_b = (stage_targets[2 + b] + a_dot).clamp(-1.5, 1.5);
                let lru_b = (stage_targets[6 + b] + l_dot).clamp(-2.0, 2.0);
                slab.cold.ctrl_state.block_adaa_scale[b] = exp2_signed_oct_f32(oct_b);
                slab.cold.ctrl_state.block_lru_bias[b] = lru_b;
            }
            for g in 0..MAX_SONA_GROUPS {
                let b = SONA_GROUP_BLOCK_AND_LOCAL[g].0;
                slab.cold.ctrl_state.group_adaa_scale[g] = slab.cold.ctrl_state.block_adaa_scale[b];
                slab.cold.ctrl_state.group_lru_bias[g] = slab.cold.ctrl_state.block_lru_bias[b];
            }
        }

        let z_nonzero = z_stage[..d_s].iter().any(|&v| v.abs() > 1.0e-12);
        let cm_active = is_ultra && ctrl.core_mods_present && z_nonzero;
        if cm_active {
            for l in 0..40 {
                let mut dg = 0.0_f32;
                let mut dk = 0.0_f32;
                let mut dwf = 0.0_f32;
                for k in 0..8 {
                    let zk = z_stage[k];
                    dg += ctrl.core_mod(k, l) * zk;
                    dk += ctrl.core_mod(k, 40 + l) * zk;
                    dwf += ctrl.core_mod(k, 80 + l) * zk;
                }
                slab.cold.ctrl_state.lane_g_scale[l] =
                    exp2_signed_oct_f32((g_oct + dg).clamp(-3.0, 3.0));
                slab.cold.ctrl_state.lane_k_delta[l] = k_del + dk;
                slab.cold.ctrl_state.lane_wf_delta[l] = dwf;
            }
        }

        let k_oct =
            (stage_targets[10] + dot_slice(&ctrl.w_kautz_oct, z_stage, d_s)).clamp(-1.5, 1.5);
        let k_damp =
            (stage_targets[11] + dot_slice(&ctrl.w_kautz_damp, z_stage, d_s)).clamp(-1.5, 1.5);
        let psi = (stage_targets[14] + dot_slice(&ctrl.w_tank_psi, z_stage, d_s))
            .clamp(-core::f32::consts::PI, core::f32::consts::PI);
        (g_oct, k_del, k_oct, k_damp, psi, cm_active)
    };

    let _ = svf_g_oct;
    let _ = svf_k_delta;
    slab.cold.ctrl_state.core_mods_active = core_mods_active;
    slab.cold.ctrl_state.kautz_oct_delta = kautz_oct;
    slab.cold.ctrl_state.kautz_damp_delta = kautz_damp;
    slab.cold.ctrl_state.tank_delay_mod = stage_targets[12];
    slab.cold.ctrl_state.tank_gain_mod = stage_targets[13];
    slab.cold.ctrl_state.tank_psi_delta = psi_delta;
    let wet_mod = stage_targets[15].clamp(-1.0, 1.0);
    slab.cold.ctrl_state.crossbar_wet_mod = wet_mod;

    let wet_scale = (1.0 + wet_mod).clamp(0.0, 2.0);
    slab.lite.op_gains_fp16 = [
        f32_to_fp16(slab.cold.base_op_p_gain * wet_scale),
        f32_to_fp16(slab.cold.base_op_t_gain * wet_scale),
    ];

    retune_slab_kautz::<B>(slab, num_kautz_banks, kautz_oct, kautz_damp, immediate);

    let gain_mod = stage_targets[13];
    for q in 0..active_quads {
        let base_g = slab.cold.base_tank_g_loop[q];
        let base_w = slab.cold.base_tank_wet_mix[q];
        let quad = slab.quad_hot_mut(q);
        quad.g_loop = (base_g + gain_mod).clamp(0.0, MAX_LOOP_GAIN_F32);
        quad.nonrec_fp16[2] = f32_to_fp16(base_w * wet_scale);
        if immediate {
            quad.snap_psi_immediate(psi_delta);
        } else {
            quad.refresh_3khz(psi_delta);
        }
    }
    slab.cold.ctrl_state.dirty = true;
}

/// Applies Stage III Kautz formant/damping/ Contraction across prefix-split `KautzStereoBank4` slices.
pub(crate) fn retune_slab_kautz<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    num_kautz_banks: usize,
    oct_shift: f32,
    damp_delta: f32,
    immediate: bool,
) {
    let slew = &mut slab.cold.kautz_slew;
    let prev_oct = if immediate {
        slew.last_dtheta = oct_shift;
        slew.last_damp = damp_delta;
        slew.prev_oct = oct_shift;
        slew.slew_active = false;
        oct_shift
    } else {
        let prev = slew.prev_oct;
        let oct_changed = (oct_shift - slew.last_dtheta).abs() > 1.0e-7;
        let damp_changed = (damp_delta - slew.last_damp).abs() > 1.0e-7;
        let settling = slew.slew_active && (oct_shift - prev).abs() > 1.0e-7;
        let uncontract = slew.slew_active;
        if !oct_changed && !damp_changed && !settling && !uncontract {
            return;
        }
        slew.slew_active = oct_changed || (oct_shift - prev).abs() > 1.0e-7;
        slew.last_dtheta = oct_shift;
        slew.last_damp = damp_delta;
        slew.prev_oct = oct_shift;
        prev
    };
    let ctx = KautzRetuneContext {
        host_poles: &slab.cold.kautz_slew.host_poles,
        host_b_gamma: &slab.cold.kautz_slew.host_b_gamma,
        host_inv_1mr2: &slab.cold.kautz_slew.host_inv_1mr2,
        num_banks: num_kautz_banks,
        oct_shift,
        damp_delta,
        prev_oct_shift: prev_oct,
    };
    retune_kautz_slice_3khz::<B>(0, &mut slab.lite.kautz, &ctx);
    retune_kautz_slice_3khz::<B>(2, &mut slab.full_delta.kautz_full, &ctx);
    retune_kautz_slice_3khz::<B>(3, &mut slab.pro_delta.kautz_pro, &ctx);
    retune_kautz_slice_3khz::<B>(6, &mut slab.ultra_delta.kautz_ultra, &ctx);
    refresh_max_pole_radius(slab);
}

/// Refreshes Stage IV rotor slew and Stage III Kautz sweep contraction at `3 kHz` (`tick & 15 == 0`).
#[inline]
pub(crate) fn refresh_3khz_tick_impl<B: GraphBackend>(slab: &mut SonaHotSlab<B>) {
    if slab.cold.ctrl_state.dirty || slab.cold.kautz_slew.slew_active {
        refresh_3khz_tick_slow(slab);
    }
}

#[inline(never)]
fn refresh_3khz_tick_slow<B: GraphBackend>(slab: &mut SonaHotSlab<B>) {
    let active_quads = tier_tank_quads(slab.lite.tier);
    let psi_delta = slab.cold.ctrl_state.tank_psi_delta;
    let mut quad_settling = false;
    for q in 0..active_quads {
        let quad = slab.quad_hot_mut(q);
        quad.refresh_3khz(psi_delta);
        let base_psi = crate::math::fp16::fp16_to_f32(quad.nonrec_fp16[3]);
        let target_psi =
            (base_psi + psi_delta).clamp(-core::f32::consts::PI, core::f32::consts::PI);
        if (target_psi - quad.current_psi).abs() > 1.0e-7 {
            quad_settling = true;
        }
    }
    if slab.cold.kautz_slew.slew_active {
        let oct = slab.cold.ctrl_state.kautz_oct_delta;
        let damp = slab.cold.ctrl_state.kautz_damp_delta;
        let banks = tier_kautz_banks(slab.lite.tier);
        retune_slab_kautz::<B>(slab, banks, oct, damp, false);
    }
    slab.cold.ctrl_state.dirty = quad_settling || slab.cold.kautz_slew.slew_active;
}

/// Applies Stage III FIR taps across `lite`..`ultra` segments (`0` wrap truncation).
pub(crate) fn write_fir_taps<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    fir_l: &[f32; SONA_FIR_MAX_TAPS],
    fir_r: &[f32; SONA_FIR_MAX_TAPS],
) {
    let copy_seg = |dst_l: &mut [[f32; 4]; 4], dst_r: &mut [[f32; 4]; 4], seg: usize| {
        for k4 in 0..4 {
            for i in 0..4 {
                let idx = (seg << 4) + (k4 << 2) + i;
                dst_l[k4][i] = fir_l[idx];
                dst_r[k4][i] = fir_r[idx];
            }
        }
    };
    for s in 0..2 {
        copy_seg(
            &mut slab.lite.fir_taps_l[s],
            &mut slab.lite.fir_taps_r[s],
            s,
        );
    }
    copy_seg(
        &mut slab.full_delta.fir_taps_l_full[0],
        &mut slab.full_delta.fir_taps_r_full[0],
        2,
    );
    for s in 0..3 {
        copy_seg(
            &mut slab.pro_delta.fir_taps_l_pro[s],
            &mut slab.pro_delta.fir_taps_r_pro[s],
            3 + s,
        );
    }
    for s in 0..2 {
        copy_seg(
            &mut slab.ultra_delta.fir_taps_l_ultra[s],
            &mut slab.ultra_delta.fir_taps_r_ultra[s],
            6 + s,
        );
    }
}

/// Writes Kautz poles & orthonormal `b_gamma` across prefix slices preserving `b_gamma >= 0.0` verbatim.
pub(crate) fn apply_kautz_poles_scaled<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    b_gamma_stored: &[f32; MAX_KAUTZ_POLES],
    c_l: &[(f32, f32); MAX_KAUTZ_POLES],
    c_r: &[(f32, f32); MAX_KAUTZ_POLES],
    kautz_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    zoh_base_poles: Option<&[(f32, f32); MAX_KAUTZ_POLES]>,
) {
    slab.cold.kautz_slew.host_poles = *kautz_poles;
    slab.cold.kautz_slew.reset_slew();
    for b in 0..MAX_KAUTZ_BANKS {
        let mut poles_arr = [(0.0_f32, 0.0_f32); 4];
        let mut bg_arr = [0.0_f32; 4];
        let mut cl_arr = [(0.0_f32, 0.0_f32); 4];
        let mut cr_arr = [(0.0_f32, 0.0_f32); 4];
        for lane in 0..4 {
            let m = (b << 2) + lane;
            let (r_raw, th_raw) = kautz_poles[m];
            let r = r_raw.clamp(0.25, 0.9995);
            let th = th_raw.clamp(0.0, core::f32::consts::PI - 1.0e-4);
            let bg_raw = if b_gamma_stored[m].is_finite() {
                b_gamma_stored[m].max(0.0)
            } else {
                0.0
            };
            let bg = if bg_raw > 0.0 {
                if let Some(base_p) = zoh_base_poles {
                    let rb = base_p[m].0.clamp(0.25, 0.9995);
                    let ratio = cold_b_gamma_f32(r)
                        * (newton6_reciprocal_f64(f64::from(cold_b_gamma_f32(rb))) as f32);
                    bg_raw * ratio
                } else {
                    bg_raw
                }
            } else {
                0.0
            };
            slab.cold.kautz_slew.host_b_gamma[m] = bg;
            slab.cold.kautz_slew.host_inv_1mr2[m] =
                newton6_reciprocal_f64((1.0 - f64::from(r) * f64::from(r)).max(1.0e-7)) as f32;
            poles_arr[lane] = (r, th);
            bg_arr[lane] = bg;
            cl_arr[lane] = c_l[m];
            cr_arr[lane] = c_r[m];
        }
        let bank = kautz_bank_mut(slab, b);
        for lane in 0..4 {
            bank.set_pole_stereo(
                lane,
                poles_arr[lane].0,
                poles_arr[lane].1,
                bg_arr[lane],
                cl_arr[lane],
                cr_arr[lane],
            );
        }
    }
    refresh_max_pole_radius(slab);
}

#[inline(always)]
pub(crate) fn kautz_bank<B: GraphBackend>(slab: &SonaHotSlab<B>, b: usize) -> &KautzStereoBank4 {
    match b {
        0..=1 => &slab.lite.kautz[b],
        2 => &slab.full_delta.kautz_full[0],
        3..=5 => &slab.pro_delta.kautz_pro[b - 3],
        _ => &slab.ultra_delta.kautz_ultra[b - 6],
    }
}

/// Recomputes peak active recursive pole radius across SONA `ParaLRU` lanes and active Kautz banks
/// (`0 FDIV`, `0 FSQRT`, `0 libm`, bounded in `[0.0, MAX_PARALRU_R_F32]`).
#[inline]
pub(crate) fn refresh_max_pole_radius<B: GraphBackend>(slab: &mut SonaHotSlab<B>) {
    let tier = slab.lite.tier;
    let num_groups = tier_sona_groups(tier);
    let num_banks = tier_kautz_banks(tier);
    let max_pole_v = B::Vec4::splat(dnam_kernels::MAX_PARALRU_R_F32);
    let tiny_r2_v = B::Vec4::splat(1.0e-30);
    let mut peak_v = B::Vec4::zero();

    for g in 0..num_groups {
        let alpha_rel = B::Vec4::load(&slab.group_hot(g).lru_alpha_rel).min(max_pole_v);
        peak_v = peak_v.max(alpha_rel.max(B::Vec4::zero()));
    }
    for b in 0..num_banks {
        let bank = kautz_bank(slab, b);
        let re = B::Vec4::load(&bank.lambda_re);
        let im = B::Vec4::load(&bank.lambda_im);
        let r2 = re.mul(re).fma(im, im);
        let r2_pos = r2.max(tiny_r2_v);
        let r_est = r2_pos.mul(r2_pos.rsqrt_pos_nr3()).min(max_pole_v);
        peak_v = peak_v.max(r_est);
    }
    let arr = peak_v.to_array();
    let max_r = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);
    slab.cold.max_pole_radius = if max_r.is_finite() {
        max_r.clamp(0.0, dnam_kernels::MAX_PARALRU_R_F32)
    } else {
        0.0
    };
}

#[inline(always)]
pub(crate) fn kautz_bank_mut<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    b: usize,
) -> &mut KautzStereoBank4 {
    match b {
        0..=1 => &mut slab.lite.kautz[b],
        2 => &mut slab.full_delta.kautz_full[0],
        3..=5 => &mut slab.pro_delta.kautz_pro[b - 3],
        _ => &mut slab.ultra_delta.kautz_ultra[b - 6],
    }
}

/// Applies Stage III readout + poles in-place (`48 kHz` hot or ZOH cold extended FIR).
pub(crate) fn sync_stage3_with_optional_zoh<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    ro: &SonaTierReadout,
    base_poles: &[(f32, f32); MAX_KAUTZ_POLES],
    os_factor: usize,
    is_bounce_4x: bool,
    zoh_opt: Option<&ZohRescaledParams>,
) {
    slab.cold.base_fir_l = ro.fir_l;
    slab.cold.base_fir_r = ro.fir_r;
    slab.cold.base_kautz_poles = *base_poles;
    slab.cold.base_kautz_b_gamma = ro.kautz_b_gamma;
    write_fir_taps(slab, &ro.fir_l, &ro.fir_r);

    let use_ext = zoh_opt.is_some_and(|z| (z.rho_sr - 1.0).abs() >= 1.0e-12);
    if !use_ext {
        slab.lite.flags &= !FLAG_COLD_FIR_EXT;
        slab.lite.fir_onset_samples = 16;
        apply_kautz_poles_scaled(
            slab,
            &ro.kautz_b_gamma,
            &ro.kautz_c_l,
            &ro.kautz_c_r,
            base_poles,
            None,
        );
        return;
    }
    let zoh = zoh_opt.expect("checked Some");
    let active_taps = (tier_fir_segments(slab.lite.tier) * 16).clamp(16, SONA_FIR_MAX_TAPS);
    slab.cold.stage3_fir_ext.reconfigure(
        &ro.fir_l,
        &ro.fir_r,
        active_taps,
        os_factor,
        is_bounce_4x,
        zoh,
    );
    slab.lite.fir_onset_samples = slab.cold.stage3_fir_ext.onset_delay_samples as u16;
    slab.lite.flags |= FLAG_COLD_FIR_EXT;
    let scaled_poles = rescale_kautz_poles_with_zoh(base_poles, zoh);
    apply_kautz_poles_scaled(
        slab,
        &ro.kautz_b_gamma,
        &ro.kautz_c_l,
        &ro.kautz_c_r,
        &scaled_poles,
        Some(base_poles),
    );
}

/// Ensures cold Stage IV ring & Schroeder buffers for quad `q`, returning effective ring cap.
pub(crate) fn ensure_stage4_quad_cold(
    ext_slot: &mut Option<TankExtRing>,
    ap_slot: &mut Option<PartitionedSchroederBank>,
    params: &TankQuadParams,
    min_frames_cap: usize,
    is_quad0: bool,
) -> usize {
    let mut p = *params;
    p.clamp_in_place();
    let max_d = p.delay_lengths.iter().copied().max().unwrap_or(67);
    let needed = max_d.saturating_add((crate::math::ceil_f32(p.lfo_depth_samples) as usize) + 6);
    let need_ext = !is_quad0
        || max_d > V3_TANK_INLINE_MAX_DELAY
        || needed > V3_TANK_CAP
        || min_frames_cap > V3_TANK_CAP;
    if need_ext {
        let target_cap = needed
            .saturating_mul(4)
            .max(min_frames_cap)
            .max(V3_TANK_CAP)
            .next_power_of_two()
            .min(V3_TANK_MAX_EXT_CAP);
        let needs_realloc = ext_slot
            .as_ref()
            .is_none_or(|e| e.cap < target_cap || !e.matches_format(p.ext_format));
        if needs_realloc {
            *ext_slot = Some(TankExtRing::new(target_cap, p.ext_format, p.delay_lengths));
        } else if let Some(ext) = ext_slot.as_mut() {
            ext.ext_delays = p.delay_lengths;
        }
    }
    if p.allpass_delays.iter().any(|&d| d > 0) {
        if let Some(ap) = ap_slot.as_mut() {
            ap.ensure_cap(p.allpass_delays, p.allpass_g);
        } else {
            *ap_slot = PartitionedSchroederBank::from_params(p.allpass_delays, p.allpass_g);
        }
    }
    ext_slot
        .as_ref()
        .map_or(if is_quad0 { V3_TANK_CAP } else { 0 }, |e| e.cap)
}

/// Reconfigures Stage IV quads (`0` RT allocs on non-growing ZOH switches; updates `base_tank_*`).
pub(crate) fn sync_stage4_params<B: GraphBackend>(
    slab: &mut SonaHotSlab<B>,
    s4: &SpatialTankParams,
    zoh_opt: Option<&ZohRescaledParams>,
    is_reset: bool,
) {
    let mut clamped = *s4;
    clamped.clamp_in_place();
    let aq = clamped.active_quads.clamp(1, MAX_TANK_QUADS);
    slab.cold.base_active_quads = aq;
    let host_sr = zoh_opt.map_or(48_000.0_f32, |z| {
        let inv = if (z.rho_sr - 1.0).abs() >= 1.0e-12 {
            newton6_reciprocal_f64(z.rho_sr.max(1.0e-6))
        } else {
            1.0
        };
        (48_000.0 * inv) as f32
    });

    for q in 0..MAX_TANK_QUADS {
        let base_qp = clamped.effective_quad_params(q);
        let (eff_qp, use_ext) = if let Some(zoh) = zoh_opt {
            let max_probe =
                base_qp.with_zoh_rescaled(zoh, V3_TANK_MAX_EXT_CAP, V3_TANK_MAX_LFO_DEPTH * 4.0);
            let max_d = max_probe.delay_lengths.iter().copied().max().unwrap_or(67);
            let needed = max_d.saturating_add((max_probe.lfo_depth_samples as usize) + 6);
            let need_ext_q = q > 0 || max_d > V3_TANK_INLINE_MAX_DELAY || needed > V3_TANK_CAP;
            let pre_cap = slab.cold.stage4_ext[q].as_ref().map_or(0, |e| e.cap);
            let active_ext = need_ext_q && pre_cap > 0;
            let cap_q = if active_ext {
                pre_cap
            } else if q == 0 {
                V3_TANK_CAP
            } else {
                0
            };
            if cap_q >= 8 {
                let md = ((cap_q / 4) as f32).min(V3_TANK_MAX_LFO_DEPTH * 4.0);
                (base_qp.with_zoh_rescaled(zoh, cap_q, md), active_ext)
            } else {
                (base_qp, active_ext)
            }
        } else {
            let max_d = base_qp.delay_lengths.iter().copied().max().unwrap_or(67);
            let needed = max_d
                .saturating_add((crate::math::ceil_f32(base_qp.lfo_depth_samples) as usize) + 6);
            let need_ext_q = q > 0 || max_d > V3_TANK_INLINE_MAX_DELAY || needed > V3_TANK_CAP;
            let active_ext = need_ext_q && slab.cold.stage4_ext[q].is_some();
            let cap_q = if active_ext {
                slab.cold.stage4_ext[q].as_ref().map_or(0, |e| e.cap)
            } else if q == 0 {
                V3_TANK_CAP
            } else {
                0
            };
            let mut qp = base_qp;
            if cap_q >= 8 {
                clamp_quad_params_to_cap(&mut qp, cap_q, V3_TANK_MAX_LFO_DEPTH);
            }
            (qp, active_ext)
        };
        slab.cold.stage4_ext_active[q] = use_ext;
        slab.cold.base_tank_g_loop[q] = eff_qp.g_loop;
        slab.cold.base_tank_wet_mix[q] = eff_qp.wet_mix;
        if let Some(ext) = slab.cold.stage4_ext[q].as_mut() {
            ext.ext_delays = eff_qp.delay_lengths;
        }
        if let Some(ap) = slab.cold.stage4_schroeder[q].as_mut() {
            ap.apply_params(eff_qp.allpass_delays, eff_qp.allpass_g);
        }
        let q_hot = slab.quad_hot_mut(q);
        q_hot.apply_params(&eff_qp, host_sr);
        if let Some(zoh) = zoh_opt {
            q_hot.reconfigure_wet_dc(zoh);
        } else {
            q_hot.wet_dc_pole = dnam_kernels::DC_BLOCKER_5HZ_48K_F32;
        }
        if is_reset {
            q_hot.reset();
        }
    }
    if is_reset {
        slab.lite.flags &= !FLAG_FREEZE;
    }
}

#[inline(always)]
fn load_lane4<B: MathBackend>(src: &[f32; MAX_SONA_GROUPS * 4], base: usize) -> B::Vec4 {
    let arr = [src[base], src[base + 1], src[base + 2], src[base + 3]];
    B::Vec4::load(&arr)
}

type HoistedGroupsAndReadouts<B, const G_CAP: usize> = (
    [SonaGroupParamsStereo4<<B as MathBackend>::Vec4>; G_CAP],
    [SonaGroupReadoutRegs<<B as MathBackend>::Vec4, <B as MathBackend>::Stereo8>; G_CAP],
);

/// Hoists all `G_CAP` `SonaGroupParamsStereo4` & `SonaGroupReadoutRegs` into stack/registers once per `16`-frame tick.
#[inline(never)]
pub(crate) fn hoist_groups_and_readouts<B: GraphBackend, const G_CAP: usize>(
    slab: &SonaHotSlab<B>,
) -> HoistedGroupsAndReadouts<B, G_CAP> {
    let g_scale_scalar = slab.cold.ctrl_state.env_fc_scale;
    let k_delta_scalar = slab.cold.ctrl_state.env_k_delta;
    let cm_active = slab.cold.ctrl_state.core_mods_active;

    let params = core::array::from_fn(|g| {
        let gh: &SonaGroupHot = slab.group_hot(g);
        let mask4 = slab.rect_mask4(g);
        let (gs, kd, wf) = if cm_active {
            let base = g << 2;
            (
                load_lane4::<B>(&slab.cold.ctrl_state.lane_g_scale, base),
                load_lane4::<B>(&slab.cold.ctrl_state.lane_k_delta, base),
                load_lane4::<B>(&slab.cold.ctrl_state.lane_wf_delta, base),
            )
        } else {
            (
                B::Vec4::splat(g_scale_scalar),
                B::Vec4::splat(k_delta_scalar),
                B::Vec4::zero(),
            )
        };
        gh.to_kernel_params_modulated::<B>(
            gs,
            kd,
            slab.cold.ctrl_state.group_adaa_scale[g],
            slab.cold.ctrl_state.group_lru_bias[g],
            wf,
            mask4,
        )
    });

    let readouts = core::array::from_fn(|g| {
        let in_w = slab.group_hot(g).in_x_weight_vec4::<B>();
        let op_p_w = B::fp16x4_to_vec4(slab.op_p_to_group_fp16(g));
        let ph_w = B::fp16x4_to_vec4(slab.phasor_to_group_fp16(g));
        let ro: &SonaGroupReadoutHot = slab.ro_group_hot(g);
        ro.decode_regs::<B>(in_w, op_p_w, ph_w)
    });

    (params, readouts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::sona_core::model::SonaMatryoshkaModel;
    use crate::math::DefaultMathBackend;

    #[test]
    fn test_ctrl_apply_uses_lite_block_scales_when_present_for_non_ultra() {
        let mut mat = SonaMatryoshkaModel::identity(false, false);
        mat.ctrl.lite_block_scales_present = true;
        mat.ctrl.w_adaa_drive_scale_lite[0][1] = 0.5;
        mat.ctrl.w_lru_bias_shift_lite[0][1] = -0.25;
        mat.ctrl.w_adaa_drive_scale[0][1] = 1.25;
        mat.ctrl.w_lru_bias_shift[0][1] = 0.9;

        let full = mat.extract_tier(Tier::Full);
        let mut slab_full = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&full);
        let z = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        apply_control_bus_impl(&mut slab_full, &[0.0; 16], &z, true);
        let expected_adaa = exp2_signed_oct_f32(0.5);
        assert!((slab_full.cold.ctrl_state.block_adaa_scale[1] - expected_adaa).abs() < 1.0e-5);
        assert!((slab_full.cold.ctrl_state.block_lru_bias[1] + 0.25).abs() < 1.0e-5);

        let ultra = mat.extract_tier(Tier::Ultra);
        let mut slab_ultra = SonaHotSlab::<DefaultMathBackend>::from_cold_model(&ultra);
        apply_control_bus_impl(&mut slab_ultra, &[0.0; 16], &z, true);
        let expected_ultra_adaa = exp2_signed_oct_f32(1.25);
        assert!(
            (slab_ultra.cold.ctrl_state.group_adaa_scale[1] - expected_ultra_adaa).abs() < 1.0e-5
        );
        assert!((slab_ultra.cold.ctrl_state.group_lru_bias[1] - 0.9).abs() < 1.0e-5);
    }
}
