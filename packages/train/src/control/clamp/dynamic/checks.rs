//! Active-tier extraction and per-operator wire/dynamic checks for `control::clamp::dynamic`
//! (`ref[train.audit.phase3.clamp-dynamic]`).

use dnam_kernels::{TANK_MAX_G_LIN_F32, TANK_MAX_SIGMA_SAT_FP16, fp16_to_f32};
use dnam_vm::Tier;
use dnam_vm::bin::{
    DnamBinView, KautzSoaBlockWire, SONA_META_CORE_MODS_PRESENT, TransportQuadWire,
};
use dnam_vm::control::STAGE_TARGET_SLOTS;

use crate::solver::model::stage_latent::{
    STAGE_LATENT_ADAA_SCALE_MAX, STAGE_LATENT_CORE_MODS_COLS, STAGE_LATENT_CORE_MODS_MAX,
    STAGE_LATENT_CORE_MODS_STRIDE, STAGE_LATENT_GLOBAL_MOD_COUNT, STAGE_LATENT_GLOBAL_MODS_MAX,
    STAGE_LATENT_GLOBAL_MODS_START, STAGE_LATENT_LITE_ADAA_START, STAGE_LATENT_LITE_LRU_START,
    STAGE_LATENT_LRU_BIAS_MAX, STAGE_LATENT_OU_ULTRA_MAX, STAGE_LATENT_ULTRA_ADAA_START,
    STAGE_LATENT_ULTRA_LRU_START, STAGE_LATENT_ULTRA_OU_START,
};

use super::{
    DYNAMIC_ALLPASS_G_MAX, DYNAMIC_DAMP_POLE_MAX, DYNAMIC_G_LOOP_MAX, DYNAMIC_KAUTZ_R_MAX,
    DynamicCornerViolation,
};

#[inline]
pub(super) fn is_transport_quad_active(q: &TransportQuadWire) -> bool {
    q.delay_samples.iter().any(|d| d.get() != 0.0)
        || q.g_loop.get() != 0.0
        || q.input_gains_fp16.iter().any(|w| w.get() != 0)
        || q.wet_tap_l_fp16.iter().any(|w| w.get() != 0)
        || q.wet_tap_r_fp16.iter().any(|w| w.get() != 0)
}

pub(super) fn collect_active_transport_quads<'a>(
    view: &DnamBinView<'a>,
    tier: Tier,
) -> Vec<(usize, &'a TransportQuadWire)> {
    let mut all: Vec<(usize, &'a TransportQuadWire)> = Vec::with_capacity(4);
    if is_transport_quad_active(&view.arena0.transport_lite) {
        all.push((0, &view.arena0.transport_lite));
    }
    if tier as u8 >= Tier::Full as u8
        && let Some(a1) = view.arena1
        && is_transport_quad_active(&a1.transport_full)
    {
        all.push((1, &a1.transport_full));
    }
    if tier as u8 >= Tier::Pro as u8
        && let Some(a2) = view.arena2
        && is_transport_quad_active(&a2.transport_pro[0])
    {
        all.push((2, &a2.transport_pro[0]));
    }
    if matches!(tier, Tier::Ultra)
        && let Some(a3) = view.arena3
        && is_transport_quad_active(&a3.transport_ultra[0])
    {
        all.push((3, &a3.transport_ultra[0]));
    }
    all
}

pub(super) fn collect_kautz_blocks<'a>(
    view: &DnamBinView<'a>,
    tier: Tier,
) -> Vec<&'a KautzSoaBlockWire> {
    let mut blocks: Vec<&'a KautzSoaBlockWire> = Vec::with_capacity(8);
    blocks.push(&view.arena0.conv_kautz_lite[0]);
    blocks.push(&view.arena0.conv_kautz_lite[1]);
    if tier as u8 >= Tier::Full as u8
        && let Some(a1) = view.arena1
    {
        blocks.push(&a1.conv_kautz_full);
    }
    if tier as u8 >= Tier::Pro as u8
        && let Some(a2) = view.arena2
    {
        for kb in &a2.conv_kautz_pro {
            blocks.push(kb);
        }
    }
    if matches!(tier, Tier::Ultra)
        && let Some(a3) = view.arena3
    {
        for kb in &a3.conv_kautz_ultra {
            blocks.push(kb);
        }
    }
    blocks
}

pub(super) fn check_kautz_blocks(
    blocks: &[&KautzSoaBlockWire],
    damp_min: f64,
    violations: &mut Vec<DynamicCornerViolation>,
) -> f64 {
    let mut max_r = 0.0_f64;
    let r_lim = f64::from(DYNAMIC_KAUTZ_R_MAX);
    let pow_damp = 2.0_f64.powf(damp_min);
    for (blk_idx, kb) in blocks.iter().enumerate() {
        for l in 0..4 {
            let mode_index = blk_idx * 4 + l;
            let rl = f64::from(kb.r_low[l].get());
            let dr = f64::from(kb.dr[l].get());
            let bg = f64::from(kb.b_gamma[l].get());
            if !rl.is_finite() || !dr.is_finite() || !bg.is_finite() {
                violations.push(DynamicCornerViolation::KautzPoleRadius {
                    mode_index,
                    r_eff: f64::NAN,
                    max_allowed: r_lim,
                });
                continue;
            }
            let dr_pos = dr.max(0.0);
            let r_base = rl + 0.5 * dr_pos;
            let r_high = rl + dr_pos;
            if r_base > 0.0 && bg > 0.0 {
                let r_eff = 1.0 - (1.0 - r_base) * pow_damp;
                if r_eff.is_finite() {
                    max_r = max_r.max(r_eff).max(r_high);
                }
                if !r_eff.is_finite() || !r_high.is_finite() || r_high > r_lim || r_eff > r_lim {
                    let reported = if !r_high.is_finite() || (r_eff.is_finite() && r_high > r_eff) {
                        r_high
                    } else {
                        r_eff
                    };
                    violations.push(DynamicCornerViolation::KautzPoleRadius {
                        mode_index,
                        r_eff: reported,
                        max_allowed: r_lim,
                    });
                }
            }
        }
    }
    max_r
}

pub(super) fn check_transport_quads(
    quads: &[(usize, &TransportQuadWire)],
    static_targets: &[f64; STAGE_TARGET_SLOTS],
    delta_32: &[f64; STAGE_TARGET_SLOTS],
    tau_floor: f64,
    violations: &mut Vec<DynamicCornerViolation>,
) -> (f64, f64, f64, f64) {
    let mut min_tau = tau_floor;
    let mut max_g = 0.0_f64;
    let mut min_g = 0.0_f64;
    let mut max_ap = 0.0_f64;
    let mut seen_any = false;

    let oct_min = (static_targets[17] - delta_32[17]).clamp(-3.0, 3.0);
    let s_lfo_max = (static_targets[20] + delta_32[20]).clamp(-4.0, 4.0);
    let delay_scale = 2.0_f64.powf(oct_min);
    let lfo_scale = 2.0_f64.powf(s_lfo_max);
    let damp_lim = f64::from(DYNAMIC_DAMP_POLE_MAX);
    let g_lim = f64::from(DYNAMIC_G_LOOP_MAX);
    let ap_lim = f64::from(DYNAMIC_ALLPASS_G_MAX);

    for &(quad_index, q) in quads {
        let damp_eff = f64::from(q.damp_pole.get()) + static_targets[21] + delta_32[21];
        if !damp_eff.is_finite() || damp_eff > damp_lim {
            violations.push(DynamicCornerViolation::TransportDampPole {
                quad_index,
                damp_eff,
                max_allowed: damp_lim,
            });
        }

        let g_base = q.g_loop.get();
        if !g_base.is_finite() || g_base < 0.0 || g_base > TANK_MAX_G_LIN_F32 {
            violations.push(DynamicCornerViolation::TransportBaseLoopGain {
                quad_index,
                g_base,
                max_allowed: TANK_MAX_G_LIN_F32,
            });
        }
        let g_lo = f64::from(g_base) + static_targets[13] - delta_32[13];
        let g_hi = f64::from(g_base) + static_targets[13] + delta_32[13];
        if g_lo.is_finite() && g_hi.is_finite() {
            if !seen_any {
                min_g = g_lo;
                max_g = g_hi;
                seen_any = true;
            } else {
                min_g = min_g.min(g_lo);
                max_g = max_g.max(g_hi);
            }
        }
        if !g_lo.is_finite() || !g_hi.is_finite() || g_lo < 0.0 || g_hi > g_lim {
            violations.push(DynamicCornerViolation::TransportLoopGain {
                quad_index,
                g_lo,
                g_hi,
                max_allowed: g_lim,
            });
        }

        let sk_bits = q.sigma_knee_fp16.get();
        let sk_val = fp16_to_f32(sk_bits);
        if (sk_bits & 0x8000) != 0 || sk_bits > TANK_MAX_SIGMA_SAT_FP16 || !sk_val.is_finite() {
            violations.push(DynamicCornerViolation::TransportSigmaKnee {
                quad_index,
                bits: sk_bits,
                max_bits: TANK_MAX_SIGMA_SAT_FP16,
            });
        }

        let base_freeze = fp16_to_f32(q.freeze_mix_fp16.get());
        if !base_freeze.is_finite() || !(0.0..=1.0).contains(&base_freeze) {
            violations.push(DynamicCornerViolation::TransportFreezeMix {
                quad_index,
                base_freeze,
            });
        }

        let base_wd = fp16_to_f32(q.window_depth_fp16.get());
        let active_ramp = (0..4).any(|k| {
            let ramp_rate_eff = f64::from(q.ramp_rate_hz.get()) + static_targets[18];
            f64::from(q.ramp_spans_samples[k].get()).abs() * ramp_rate_eff.abs() != 0.0
        });
        let wd_mod_finite = (base_wd + static_targets[24] as f32).is_finite();
        if !base_wd.is_finite()
            || !(0.0..=1.0).contains(&base_wd)
            || (active_ramp && !wd_mod_finite)
        {
            violations.push(DynamicCornerViolation::TransportWindowDepth {
                quad_index,
                base_window_depth: base_wd,
            });
        }

        for lane in 0..4 {
            let d_nom = f64::from(q.delay_samples[lane].get());
            let d_lfo = f64::from(fp16_to_f32(q.lfo_depths_fp16[lane].get()));
            let s_span = f64::from(q.ramp_spans_samples[lane].get());
            let tau_min = d_nom * delay_scale + (static_targets[12] - delta_32[12])
                - d_lfo.abs() * lfo_scale
                - 0.5 * s_span.abs();
            if tau_min.is_finite() {
                min_tau = min_tau.min(tau_min);
            }
            if !tau_min.is_finite() || tau_min < tau_floor {
                violations.push(DynamicCornerViolation::TransportDelayFloor {
                    quad_index,
                    lane,
                    tau_min,
                    tau_floor,
                });
            }

            let a_base = f64::from(fp16_to_f32(q.allpass_g_fp16[lane].get()));
            let abs_ap = (a_base + static_targets[25]).abs() + delta_32[25];
            if abs_ap.is_finite() {
                max_ap = max_ap.max(abs_ap);
            }
            if !abs_ap.is_finite() || abs_ap > ap_lim {
                violations.push(DynamicCornerViolation::TransportAllpassGain {
                    quad_index,
                    lane,
                    abs_max: abs_ap,
                    max_allowed: ap_lim,
                });
            }
        }
    }
    (min_tau, max_g, min_g, max_ap)
}

pub(super) fn check_stage_latent_wire_tables(
    view: &DnamBinView<'_>,
    tier: Tier,
    violations: &mut Vec<DynamicCornerViolation>,
) {
    let push_bad = |violations: &mut Vec<DynamicCornerViolation>,
                    table: &'static str,
                    index: usize,
                    value: f32| {
        violations.push(DynamicCornerViolation::StageLatentTableBound {
            table,
            index,
            value,
        });
    };

    let mods = &view.arena0.linear_readout_lite.stage_latent_mods_fp16;
    let global_end = STAGE_LATENT_GLOBAL_MODS_START + STAGE_LATENT_GLOBAL_MOD_COUNT * 8;
    for (idx, w) in mods
        .iter()
        .enumerate()
        .take(global_end)
        .skip(STAGE_LATENT_GLOBAL_MODS_START)
    {
        let v = fp16_to_f32(w.get());
        if !v.is_finite() || v.abs() > STAGE_LATENT_GLOBAL_MODS_MAX {
            push_bad(violations, "stage_latent_mods_fp16", idx, v);
        }
    }
    for (idx, w) in mods
        .iter()
        .enumerate()
        .take(STAGE_LATENT_LITE_LRU_START)
        .skip(STAGE_LATENT_LITE_ADAA_START)
    {
        let v = fp16_to_f32(w.get());
        if !v.is_finite() || v.abs() > STAGE_LATENT_ADAA_SCALE_MAX {
            push_bad(violations, "stage_latent_mods_fp16", idx, v);
        }
    }
    for (idx, w) in mods
        .iter()
        .enumerate()
        .take(80)
        .skip(STAGE_LATENT_LITE_LRU_START)
    {
        let v = fp16_to_f32(w.get());
        if !v.is_finite() || v.abs() > STAGE_LATENT_LRU_BIAS_MAX {
            push_bad(violations, "stage_latent_mods_fp16", idx, v);
        }
    }

    if !matches!(tier, Tier::Ultra) {
        return;
    }
    let Some(a3) = view.arena3 else {
        return;
    };
    let u_ctrl = &a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16;
    for (idx, w) in u_ctrl
        .iter()
        .enumerate()
        .take(STAGE_LATENT_ULTRA_LRU_START)
        .skip(STAGE_LATENT_ULTRA_ADAA_START)
    {
        let v = fp16_to_f32(w.get());
        if !v.is_finite() || v.abs() > STAGE_LATENT_ADAA_SCALE_MAX {
            push_bad(violations, "stage_latent_ctrl_ultra_fp16", idx, v);
        }
    }
    for (idx, w) in u_ctrl
        .iter()
        .enumerate()
        .take(STAGE_LATENT_ULTRA_OU_START)
        .skip(STAGE_LATENT_ULTRA_LRU_START)
    {
        let v = fp16_to_f32(w.get());
        if !v.is_finite() || v.abs() > STAGE_LATENT_LRU_BIAS_MAX {
            push_bad(violations, "stage_latent_ctrl_ultra_fp16", idx, v);
        }
    }
    for (idx, w) in u_ctrl
        .iter()
        .enumerate()
        .take(STAGE_LATENT_ULTRA_OU_START + 4)
        .skip(STAGE_LATENT_ULTRA_OU_START)
    {
        let v = fp16_to_f32(w.get());
        if !v.is_finite() || v.abs() > STAGE_LATENT_OU_ULTRA_MAX {
            push_bad(violations, "stage_latent_ctrl_ultra_fp16", idx, v);
        }
    }
    for (idx, w) in u_ctrl
        .iter()
        .enumerate()
        .skip(STAGE_LATENT_ULTRA_OU_START + 4)
    {
        let raw = w.get();
        if raw != 0 {
            push_bad(
                violations,
                "stage_latent_ctrl_ultra_fp16",
                idx,
                fp16_to_f32(raw),
            );
        }
    }

    if (view.arena0.crossbar.sona_meta_flags.get() & SONA_META_CORE_MODS_PRESENT) != 0 {
        for k in 0..8 {
            let row_base = k * STAGE_LATENT_CORE_MODS_STRIDE;
            for c in 0..STAGE_LATENT_CORE_MODS_COLS {
                let idx = row_base + c;
                let v = fp16_to_f32(a3.stage_latent_core_mods_fp16[idx].get());
                if !v.is_finite() || v.abs() > STAGE_LATENT_CORE_MODS_MAX {
                    push_bad(violations, "stage_latent_core_mods_fp16", idx, v);
                }
            }
            for c in STAGE_LATENT_CORE_MODS_COLS..STAGE_LATENT_CORE_MODS_STRIDE {
                let idx = row_base + c;
                let raw = a3.stage_latent_core_mods_fp16[idx].get();
                if raw != 0 {
                    push_bad(
                        violations,
                        "stage_latent_core_mods_fp16",
                        idx,
                        fp16_to_f32(raw),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dnam_kernels::{TANK_MAX_SIGMA_SAT_FP16, f32_to_fp16};
    use dnam_vm::Tier;
    use dnam_vm::bin::{DnamBinView, SONA_META_CORE_MODS_PRESENT};
    use dnam_vm::control::STAGE_TARGET_SLOTS;
    use zerocopy::byteorder::little_endian::{F32, U16, U32};

    use crate::solver::model::SonaModelF64;
    use crate::solver::pass5_emit::{Pass5EmitConfig, encode_model_to_dnam_bin};

    use super::*;

    #[test]
    fn test_wire_check_helpers_and_panama_exclusion_classification() {
        let mut model = SonaModelF64::new(48_000.0);
        model.stage_latent_mods.core_mods_ultra = Some(Box::new([[0.05; 120]; 8]));
        for tq in &mut model.transport_quads {
            tq.delay_samples = [12.0; 4];
            tq.g_loop = 0.5;
        }
        let bytes = encode_model_to_dnam_bin(
            &model,
            Pass5EmitConfig {
                calibrate_lufs_trims: false,
                ..Pass5EmitConfig::default()
            },
        )
        .unwrap();
        let base_view = DnamBinView::parse(&bytes).unwrap();
        let mut a0 = *base_view.arena0;
        a0.conv_kautz_lite[0].r_low[0] = F32::new(0.9999);
        a0.conv_kautz_lite[0].b_gamma[0] = F32::new(1.0);
        a0.transport_lite.g_loop = F32::new(1.05);
        a0.transport_lite.sigma_knee_fp16 = U16::new(TANK_MAX_SIGMA_SAT_FP16 + 1);
        a0.transport_lite.freeze_mix_fp16 = U16::new(f32_to_fp16(1.5));
        a0.transport_lite.window_depth_fp16 = U16::new(f32_to_fp16(-0.2));
        a0.linear_readout_lite.stage_latent_mods_fp16[10] = U16::new(f32_to_fp16(99.0));
        a0.crossbar.sona_meta_flags = U32::new(SONA_META_CORE_MODS_PRESENT);
        let mut a3 = *base_view.arena3.unwrap();
        a3.stage_latent_core_mods_fp16[121] = U16::new(f32_to_fp16(0.5));
        a3.control_bus_ultra_delta.stage_latent_ctrl_ultra_fp16[170] = U16::new(f32_to_fp16(0.5));

        let mut view = base_view;
        view.arena0 = &a0;
        view.arena3 = Some(&a3);

        let mut v = Vec::new();
        let blocks = collect_kautz_blocks(&view, Tier::Ultra);
        assert_eq!(blocks.len(), 8);
        check_kautz_blocks(&blocks, 0.0, &mut v);

        let quads = collect_active_transport_quads(&view, Tier::Ultra);
        assert_eq!(quads.len(), 4);
        let t = [0.0_f64; STAGE_TARGET_SLOTS];
        let d = [0.0_f64; STAGE_TARGET_SLOTS];
        check_transport_quads(&quads, &t, &d, 5.0, &mut v);
        check_stage_latent_wire_tables(&view, Tier::Ultra, &mut v);

        assert!(v.iter().any(|x| {
            matches!(x, DynamicCornerViolation::KautzPoleRadius { .. }) && x.is_panama_exclusion()
        }));
        assert!(v.iter().any(|x| {
            matches!(x, DynamicCornerViolation::TransportBaseLoopGain { .. })
                && !x.is_panama_exclusion()
        }));
        assert!(v.iter().any(|x| {
            matches!(x, DynamicCornerViolation::TransportSigmaKnee { .. })
                && !x.is_panama_exclusion()
        }));
        assert!(v.iter().any(|x| {
            matches!(x, DynamicCornerViolation::TransportFreezeMix { .. })
                && !x.is_panama_exclusion()
        }));
        assert!(v.iter().any(|x| {
            matches!(x, DynamicCornerViolation::TransportWindowDepth { .. })
                && !x.is_panama_exclusion()
        }));
        assert!(v.iter().any(|x| {
            matches!(
                x,
                DynamicCornerViolation::StageLatentTableBound {
                    table: "stage_latent_mods_fp16",
                    index: 10,
                    ..
                }
            ) && !x.is_panama_exclusion()
        }));
        assert!(v.iter().any(|x| {
            matches!(
                x,
                DynamicCornerViolation::StageLatentTableBound {
                    table: "stage_latent_core_mods_fp16",
                    index: 121,
                    ..
                }
            )
        }));
        assert!(v.iter().any(|x| {
            matches!(
                x,
                DynamicCornerViolation::StageLatentTableBound {
                    table: "stage_latent_ctrl_ultra_fp16",
                    index: 170,
                    ..
                }
            )
        }));
    }
}
