#![forbid(unsafe_code)]
//! Non-Stop 3-Phase Control-Space Audit Helpers (`dnam_capture::audit`:
//! `ref[train.audit.phase1.clamp-range]`, `ref[train.audit.phase1.clamp-switch]`,
//! `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase2.corner-barrier]`,
//! `ref[train.audit.phase2.clamp-rail]`, `ref[train.audit.phase3.clamp-dynamic]`,
//! `ref[train.audit.phase3.clamp-head-gain]`, `ref[train.audit.phase3.numeric-envelope]`,
//! `ref[train.audit.report.prescription]`).

pub mod manifest;

use std::collections::BTreeSet;

use dnam_kernels::{MAX_PARALRU_R_F32, f32_to_fp16, fp16_to_f32};
use dnam_train::control::{
    MAX_RO_POLE_48K_F32, Phase3AuditInput, audit_phase3_dynamic_and_head_gain,
    enforce_arena_block0_k_nl_and_clamp, evaluate_readout_spectral_magnitudes,
    run_phase2_rail_burst,
};
use dnam_vm::Tier;
use dnam_vm::bin::{
    Arena0LiteWire, Arena1FullDeltaWire, Arena2ProDeltaWire, Arena3UltraDeltaWire,
    DnamBinBuildInput, DnamBinView, KautzSoaBlockWire, LinearReadoutLiteWire,
    SONA_META_CONVOLVER_PREFIX_OVERRIDES, encode_dnam_bin,
};
use zerocopy::FromBytes;
use zerocopy::byteorder::little_endian::{F32, U16};

pub use manifest::{
    recompute_vault_phase1_audit_and_clamps, recompute_vault_phase1_audit_and_clamps_with_windows,
};

use crate::proj::schema::DnamProjManifest;

/// Phase-1 `[CLAMP-RANGE]`: intersects `declared_range` with `observed_range` and locks `K_p = 0`
/// if `distinct_observed_count <= 1` (`ref[train.audit.phase1.clamp-range]`).
#[inline]
#[must_use]
pub fn audit_phase1_clamp_range(
    declared: (f32, f32),
    observed: (f32, f32),
    distinct_observed_count: usize,
) -> ((f32, f32), u8, bool) {
    let d_lo = if declared.0.is_finite() {
        declared.0
    } else {
        0.0
    };
    let d_hi = if declared.1.is_finite() {
        declared.1.max(d_lo)
    } else {
        d_lo
    };
    let o_lo = if observed.0.is_finite() {
        observed.0
    } else {
        d_lo
    };
    let o_hi = if observed.1.is_finite() {
        observed.1.max(o_lo)
    } else {
        o_lo
    };
    let safe_min = d_lo.clamp(d_lo, d_hi).max(o_lo).min(d_hi);
    let safe_max = d_hi.min(o_hi).max(safe_min);
    let degree = if distinct_observed_count <= 1 || safe_max <= safe_min {
        0u8
    } else {
        (distinct_observed_count - 1).min(3) as u8
    };
    let was_clamped = !declared.0.is_finite()
        || !declared.1.is_finite()
        || !observed.0.is_finite()
        || !observed.1.is_finite()
        || declared.1 < declared.0
        || observed.1 < observed.0
        || safe_min != declared.0
        || safe_max != declared.1;
    ((safe_min, safe_max), degree, was_clamped)
}

/// Phase-1 `[CLAMP-SWITCH]`: restricts `declared_options` to options present in `observed_options`
/// (`ref[train.audit.phase1.clamp-switch]`).
#[must_use]
pub fn audit_phase1_clamp_switch(
    declared_options: &[&str],
    observed_options: &BTreeSet<&str>,
) -> (u16, Vec<String>) {
    let mut mask = 0u16;
    let mut disabled = Vec::new();
    for (idx, &opt) in declared_options.iter().enumerate().take(16) {
        if observed_options.contains(opt) {
            mask |= 1u16 << idx;
        } else {
            disabled.push(opt.to_string());
        }
    }
    (mask, disabled)
}

/// Phase-1 `[CLAMP-STABILITY]` & Phase-3 `[CLAMP-DYNAMIC]` worst-case excursion verifier
/// (`r_worst <= 0.9995`, `g_lin <= 0.9990`, `|a_diff| <= 0.75`, `tau_min >= 4.0`,
/// `ref[train.audit.phase1.clamp-stability]`, `ref[train.audit.phase3.clamp-dynamic]`,
/// `ref[train.audit.phase3.numeric-envelope]`).
#[inline]
#[must_use]
pub fn verify_dynamic_stability_excursion(
    r_high: f32,
    r_mod_l1: f32,
    tau_0: f32,
    tau_mod_l1: f32,
    g_lin: f32,
    a_diff: f32,
) -> (f32, f32, f32, f32, bool) {
    let finite = r_high.is_finite()
        && r_mod_l1.is_finite()
        && tau_0.is_finite()
        && tau_mod_l1.is_finite()
        && g_lin.is_finite()
        && a_diff.is_finite();
    let r_worst = r_high + r_mod_l1.abs();
    let tau_min = tau_0 - tau_mod_l1.abs();
    let ok = finite
        && r_high >= 0.0
        && (0.0..=0.9995).contains(&r_worst)
        && tau_min.is_finite()
        && tau_min >= 4.0
        && (0.0..=0.9990).contains(&g_lin)
        && a_diff.abs() <= 0.75;
    let r_c = if r_worst.is_finite() {
        r_worst.clamp(0.0, 0.9995)
    } else {
        0.0
    };
    let tau_c = if tau_min.is_finite() {
        tau_min.max(4.0)
    } else {
        4.0
    };
    let g_c = if g_lin.is_finite() {
        g_lin.clamp(0.0, 0.9990)
    } else {
        0.0
    };
    let a_c = if a_diff.is_finite() {
        a_diff.clamp(-0.75, 0.75)
    } else {
        0.0
    };
    (r_c, tau_c, g_c, a_c, ok)
}

/// Phase-3 `[CLAMP-HEAD-GAIN]`: clamps rendered head spectral gain to `<= +6.0 dB` (`<= 1.9952623`
/// linear amplitude ratio, `ref[train.audit.phase3.clamp-head-gain]`).
#[inline]
#[must_use]
pub fn clamp_head_spectral_gain_linear(peak_gain_linear: f32) -> (f32, bool) {
    const MAX_GAIN_6DB_LIN: f32 = 1.995_262_3;
    const MAX_GAIN_IDEMPOTENT: f32 = 1.995_258_3;
    if !peak_gain_linear.is_finite() || peak_gain_linear < 0.0 {
        (0.0, true)
    } else if peak_gain_linear > MAX_GAIN_6DB_LIN {
        (MAX_GAIN_IDEMPOTENT / peak_gain_linear, true)
    } else {
        (1.0, false)
    }
}

/// Formats a consolidated post-training audit & stem prescription banner (`ref[train.audit.report.prescription]`).
#[must_use]
pub fn format_audit_prescription_report(model_name: &str, clamp_lines: &[&str]) -> String {
    let mut out = String::new();
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&format!(
        "DNAM TRAINING COMPLETE — EXPORTED: {model_name}.dnam | {model_name}.dnam-proj\n"
    ));
    out.push_str(
        "================================================================================\n",
    );
    out.push_str(&format!(
        "Active Control Clamps Applied ({}):\n",
        clamp_lines.len()
    ));
    for (idx, line) in clamp_lines.iter().enumerate() {
        out.push_str(&format!("  {}. {line}\n", idx + 1));
    }
    out.push_str(
        "================================================================================\n",
    );
    out
}

/// Phase-2 Stop-Gradient Corner Barrier Loss $\mathcal{L}_{\mathrm{barrier}}$ evaluated across
/// physical stability bounds ($r_c \le 0.9995$, $g_{\mathrm{lin}} \le 0.9990$,
/// $|a_{\mathrm{diff}}| \le 0.75$, $\tau_m \ge 4.0$; `ref[train.audit.phase2.corner-barrier]`).
#[must_use]
pub fn compute_phase2_corner_barrier_loss(
    r_worst: f64,
    g_lin: f64,
    a_diff: f64,
    tau_min: f64,
) -> f64 {
    if !r_worst.is_finite() || !g_lin.is_finite() || !a_diff.is_finite() || !tau_min.is_finite() {
        return 1.0e6;
    }
    if !(0.0..=0.9995).contains(&r_worst)
        || !(0.0..=0.9990).contains(&g_lin)
        || a_diff.abs() > 0.75
        || tau_min < 4.0
    {
        let over_r = (r_worst - 0.9995).max(0.0) + (-r_worst).max(0.0);
        let over_g = (g_lin - 0.9990).max(0.0) + (-g_lin).max(0.0);
        let over_a = (a_diff.abs() - 0.75).max(0.0);
        let under_tau = (4.0 - tau_min).max(0.0);
        return 1.0e4
            + 1.0e5
                * (over_r * over_r + over_g * over_g + over_a * over_a + under_tau * under_tau);
    }
    let margin_r = (0.9995 - r_worst).max(1.0e-7);
    let margin_g = (0.9990 - g_lin).max(1.0e-7);
    let margin_a = (0.75 - a_diff.abs()).max(1.0e-7);
    let margin_tau = (tau_min - 4.0).max(1.0e-7);
    let barrier = -1.0e-4
        * ((margin_r / 0.01).min(1.0).ln()
            + (margin_g / 0.01).min(1.0).ln()
            + (margin_a / 0.05).min(1.0).ln()
            + (margin_tau / 1.0).min(1.0).ln());
    barrier.max(0.0)
}

/// Phase-2 `[CLAMP-RAIL]` `2,048`-Sample `SONA` Bounded-State Saturation Audit (`ref[train.audit.phase2.clamp-rail]`).
#[must_use]
pub fn audit_phase2_gru_state_saturation(dnam_bin: &[u8], corner_drive: f32) -> (f32, f32, bool) {
    if !corner_drive.is_finite() || corner_drive <= 0.0 {
        return (1.0, 0.25, true);
    }
    match run_phase2_rail_burst(dnam_bin, Tier::Lite, 48_000.0, f64::from(corner_drive)) {
        Ok(rep) => (rep.occupancy as f32, rep.safe_scale as f32, rep.flagged),
        Err(_) => (1.0, 0.25, true),
    }
}

fn push_unique_tag(tags: &mut Vec<String>, tag: &str) {
    if !tags.iter().any(|existing| existing == tag) {
        tags.push(tag.to_string());
    }
}

fn inspect_pre_clamp_stability(view: &DnamBinView<'_>) -> (bool, bool) {
    let mut stab_viol = false;
    let mut dyn_viol = false;
    for kb in &view.arena0.conv_kautz_lite {
        for i in 0..4 {
            let rl = kb.r_low[i].get();
            let dr = kb.dr[i].get();
            if !rl.is_finite() || !dr.is_finite() || !(0.0..=0.9995).contains(&(rl + 0.5 * dr)) {
                stab_viol = true;
            }
        }
    }
    for cq in &view.arena0.core_quads {
        for i in 0..4 {
            let r = cq.lru_alpha_rel[i].get();
            if !r.is_finite() || r < 0.0 || r > MAX_PARALRU_R_F32 {
                stab_viol = true;
            }
        }
    }
    let quads = [
        Some(&view.arena0.transport_lite),
        view.arena1.map(|a| &a.transport_full),
        view.arena2.map(|a| &a.transport_pro[0]),
        view.arena3.map(|a| &a.transport_ultra[0]),
    ];
    for tq in quads.into_iter().flatten() {
        let gl = tq.g_loop.get();
        if !gl.is_finite() || !(0.0..=0.9990).contains(&gl) {
            stab_viol = true;
        }
        for i in 0..4 {
            let d = tq.delay_samples[i].get();
            let lfo = fp16_to_f32(tq.lfo_depths_fp16[i].get());
            let ap = fp16_to_f32(tq.allpass_g_fp16[i].get());
            if !d.is_finite() || d < 5.0 || !ap.is_finite() || ap.abs() > 0.75 {
                stab_viol = true;
            }
            if !lfo.is_finite() || (d - lfo.abs()) < 5.0 {
                dyn_viol = true;
            }
        }
    }
    (stab_viol, dyn_viol)
}

#[inline]
fn scale_f32_slice(words: &mut [F32], scale: f32) {
    for slot in words {
        *slot = F32::new(slot.get() * scale);
    }
}

#[inline]
fn scale_fp16_slice(words: &mut [U16], scale: f32) {
    for slot in words {
        let scaled = fp16_to_f32(slot.get()) * scale;
        *slot = U16::new(f32_to_fp16(scaled));
    }
}

fn scale_w_ro_direct(words: &mut [U16], lanes: usize, scale: f32) {
    for l in 0..lanes {
        scale_fp16_slice(&mut words[l * 12 + 6..l * 12 + 12], scale);
    }
}

fn scale_kautz_residues(blks: &mut [KautzSoaBlockWire], scale: f32) {
    for blk in blks {
        scale_f32_slice(&mut blk.c_re_l, scale);
        scale_f32_slice(&mut blk.c_im_l, scale);
        scale_f32_slice(&mut blk.c_re_r, scale);
        scale_f32_slice(&mut blk.c_im_r, scale);
    }
}

fn scale_prefix_override_f32s(bytes: &mut [u8], scale: f32) {
    if let Ok(words) = <[F32]>::mut_from_bytes(bytes) {
        scale_f32_slice(words, scale);
    }
}

fn scale_linear_readout_lite(ro: &mut LinearReadoutLiteWire, scale: f32) {
    scale_f32_slice(&mut ro.direct_s2_u_gain, scale);
    scale_f32_slice(&mut ro.direct_s2_hp_gain, scale);
}

fn scale_upper_tier_direct_s2(
    a1: Option<&mut Arena1FullDeltaWire>,
    a2: Option<&mut Arena2ProDeltaWire>,
    a3: Option<&mut Arena3UltraDeltaWire>,
    scale: f32,
) {
    if let Some(w) = a1 {
        scale_fp16_slice(
            &mut w.linear_readout_full.w_ro_and_mods_fp16[292..296],
            scale,
        );
    }
    if let Some(w) = a2 {
        scale_fp16_slice(
            &mut w.linear_readout_pro.w_ro_and_mods_fp16[388..392],
            scale,
        );
    }
    if let Some(w) = a3 {
        scale_fp16_slice(
            &mut w.linear_readout_ultra.w_ro_and_mods_fp16[484..488],
            scale,
        );
    }
}

fn scale_full_post_convolver_wire(
    a0: &mut Arena0LiteWire,
    mut a1: Option<&mut Arena1FullDeltaWire>,
    mut a2: Option<&mut Arena2ProDeltaWire>,
    mut a3: Option<&mut Arena3UltraDeltaWire>,
    scale: f32,
) {
    scale_linear_readout_lite(&mut a0.linear_readout_lite, scale);
    scale_upper_tier_direct_s2(
        a1.as_deref_mut(),
        a2.as_deref_mut(),
        a3.as_deref_mut(),
        scale,
    );

    scale_w_ro_direct(&mut a0.linear_readout_lite.w_ro_fp16[..192], 16, scale);
    scale_f32_slice(&mut a0.conv_fir_lite.fir_l, scale);
    scale_f32_slice(&mut a0.conv_fir_lite.fir_r, scale);
    scale_kautz_residues(&mut a0.conv_kautz_lite, scale);

    let emit_pfx = (a0.crossbar.sona_meta_flags.get() & SONA_META_CONVOLVER_PREFIX_OVERRIDES) != 0;
    if let Some(w) = a1 {
        scale_w_ro_direct(
            &mut w.linear_readout_full.w_ro_and_mods_fp16[..288],
            24,
            scale,
        );
        scale_f32_slice(&mut w.conv_fir_full.fir_l, scale);
        scale_f32_slice(&mut w.conv_fir_full.fir_r, scale);
        scale_kautz_residues(core::slice::from_mut(&mut w.conv_kautz_full), scale);
        if emit_pfx {
            scale_prefix_override_f32s(&mut w.linear_readout_full.prefix_override, scale);
            scale_fp16_slice(
                &mut w.linear_readout_full.w_ro_and_mods_fp16[296..328],
                scale,
            );
            scale_fp16_slice(
                &mut w.linear_readout_full.w_ro_and_mods_fp16[336..352],
                scale,
            );
        }
    }
    if let Some(w) = a2 {
        scale_w_ro_direct(
            &mut w.linear_readout_pro.w_ro_and_mods_fp16[..384],
            32,
            scale,
        );
        scale_f32_slice(&mut w.conv_fir_pro.fir_l, scale);
        scale_f32_slice(&mut w.conv_fir_pro.fir_r, scale);
        scale_kautz_residues(&mut w.conv_kautz_pro, scale);
        if emit_pfx {
            scale_prefix_override_f32s(&mut w.linear_readout_pro.prefix_override, scale);
            scale_fp16_slice(
                &mut w.linear_readout_pro.w_ro_and_mods_fp16[392..424],
                scale,
            );
            scale_fp16_slice(
                &mut w.linear_readout_pro.w_ro_and_mods_fp16[436..452],
                scale,
            );
        }
    }
    if let Some(w) = a3 {
        scale_w_ro_direct(
            &mut w.linear_readout_ultra.w_ro_and_mods_fp16[..480],
            40,
            scale,
        );
        scale_f32_slice(&mut w.conv_fir_ultra.fir_l, scale);
        scale_f32_slice(&mut w.conv_fir_ultra.fir_r, scale);
        scale_kautz_residues(&mut w.conv_kautz_ultra, scale);
        if emit_pfx {
            scale_prefix_override_f32s(&mut w.linear_readout_ultra.prefix_override, scale);
            scale_fp16_slice(
                &mut w.linear_readout_ultra.w_ro_and_mods_fp16[488..584],
                scale,
            );
            scale_fp16_slice(
                &mut w.linear_readout_ultra.w_ro_and_mods_fp16[608..656],
                scale,
            );
        }
    }
}

fn run_3phase_audit_inner(
    manifest: &mut DnamProjManifest,
    dnam_bin: &mut Vec<u8>,
    full_backbone: bool,
) {
    let Ok(view) = DnamBinView::parse(dnam_bin) else {
        return;
    };
    let (stab_viol, dyn_viol) = inspect_pre_clamp_stability(&view);
    let mut a0 = *view.arena0;
    let mut a1 = view.arena1.copied();
    let mut a2 = view.arena2.copied();
    let mut a3 = view.arena3.copied();
    let mut ro_pole = view.runtime_meta.ro_pole_48k.get();

    if full_backbone {
        enforce_arena_block0_k_nl_and_clamp(
            &mut a0,
            a1.as_mut(),
            a2.as_mut(),
            a3.as_mut(),
            &mut ro_pole,
        );
    } else {
        ro_pole = if ro_pole.is_finite() {
            ro_pole.clamp(0.0, MAX_RO_POLE_48K_F32)
        } else {
            0.0
        };
    }

    let sr_hz = f64::from(view.header.reference_sample_rate_hz.get());
    let ro_l = &a0.linear_readout_lite;
    let gains_l = vec![
        f64::from(ro_l.direct_s2_u_gain[0].get()),
        f64::from(ro_l.direct_s2_hp_gain[0].get()),
    ];
    let gains_r = vec![
        f64::from(ro_l.direct_s2_u_gain[1].get()),
        f64::from(ro_l.direct_s2_hp_gain[1].get()),
    ];
    let mut mags_l = evaluate_readout_spectral_magnitudes(&gains_l, f64::from(ro_pole), sr_hz, 32);
    let mut mags_r = evaluate_readout_spectral_magnitudes(&gains_r, f64::from(ro_pole), sr_hz, 32);
    let upper_ro_slots: [Option<[U16; 4]>; 3] = [
        a1.as_ref().and_then(|w| {
            w.linear_readout_full.w_ro_and_mods_fp16[292..296]
                .try_into()
                .ok()
        }),
        a2.as_ref().and_then(|w| {
            w.linear_readout_pro.w_ro_and_mods_fp16[388..392]
                .try_into()
                .ok()
        }),
        a3.as_ref().and_then(|w| {
            w.linear_readout_ultra.w_ro_and_mods_fp16[484..488]
                .try_into()
                .ok()
        }),
    ];
    for slots in upper_ro_slots.into_iter().flatten() {
        let ul = [
            f64::from(fp16_to_f32(slots[0].get())),
            f64::from(fp16_to_f32(slots[2].get())),
        ];
        let ur = [
            f64::from(fp16_to_f32(slots[1].get())),
            f64::from(fp16_to_f32(slots[3].get())),
        ];
        let ml = evaluate_readout_spectral_magnitudes(&ul, f64::from(ro_pole), sr_hz, 32);
        let mr = evaluate_readout_spectral_magnitudes(&ur, f64::from(ro_pole), sr_hz, 32);
        for (dst, src) in mags_l.iter_mut().zip(ml) {
            *dst = if dst.is_finite() && src.is_finite() {
                dst.max(src)
            } else {
                f64::NAN
            };
        }
        for (dst, src) in mags_r.iter_mut().zip(mr) {
            *dst = if dst.is_finite() && src.is_finite() {
                dst.max(src)
            } else {
                f64::NAN
            };
        }
    }

    let min_delay_samples = a0
        .transport_lite
        .delay_samples
        .iter()
        .map(|d| f64::from(d.get()))
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min)
        .clamp(5.0, 48_000.0);
    let max_delay_excursion = a0
        .transport_lite
        .lfo_depths_fp16
        .iter()
        .map(|w| f64::from(fp16_to_f32(w.get()).abs()))
        .fold(0.0_f64, f64::max);
    let max_paralru_r_excursion = a0
        .core_quads
        .iter()
        .flat_map(|q| q.lru_alpha_rel.iter().map(|r| f64::from(r.get())))
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);
    let max_tank_g_excursion = f64::from(a0.transport_lite.g_loop.get()).clamp(0.0, 0.9990);

    let mut p3_in = Phase3AuditInput {
        min_delay_samples,
        max_delay_excursion,
        max_paralru_r_excursion,
        max_tank_g_excursion,
        crossbar_coupled: true,
        ro_pole_48k: ro_pole,
        readout_gains_l: gains_l,
        readout_gains_r: gains_r,
        freq_mag_l: mags_l,
        freq_mag_r: mags_r,
        burst_peak_abs: 1.0,
    };
    let mut head_clamped = false;
    if let Ok(p3_rep) = audit_phase3_dynamic_and_head_gain(&mut p3_in) {
        ro_pole = p3_rep.ro_pole_48k;
        if p3_rep.head_gain_scale < 1.0 {
            head_clamped = true;
            let scale_f32 = p3_rep.head_gain_scale as f32;
            if full_backbone {
                scale_full_post_convolver_wire(
                    &mut a0,
                    a1.as_mut(),
                    a2.as_mut(),
                    a3.as_mut(),
                    scale_f32,
                );
            } else {
                scale_linear_readout_lite(&mut a0.linear_readout_lite, scale_f32);
                scale_upper_tier_direct_s2(a1.as_mut(), a2.as_mut(), a3.as_mut(), scale_f32);
            }
        }
        if full_backbone && max_delay_excursion > p3_rep.clamped_delay_excursion + 1.0e-4 {
            let max_depth = p3_rep.clamped_delay_excursion.min(512.0) as f32;
            for w in &mut a0.transport_lite.lfo_depths_fp16 {
                let d = fp16_to_f32(w.get()).clamp(0.0, max_depth);
                *w = zerocopy::byteorder::little_endian::U16::new(f32_to_fp16(d));
            }
        }
    }

    let ar = manifest.audit_report.get_or_insert_with(Default::default);
    if full_backbone && stab_viol {
        push_unique_tag(&mut ar.phase1_clamps, "CLAMP-STABILITY");
    }
    if full_backbone && dyn_viol {
        push_unique_tag(&mut ar.phase3_clamps, "CLAMP-DYNAMIC");
    }
    if head_clamped {
        push_unique_tag(&mut ar.phase3_clamps, "CLAMP-HEAD-GAIN");
    }

    if (full_backbone || head_clamped)
        && let Ok(reencoded) = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: view.header.reference_sample_rate_hz.get(),
            header_flags: view.header.flags.get(),
            ext_ring_pow2_frames: view.runtime_meta.ext_ring_pow2_frames,
            ext_ring_lines: view.runtime_meta.ext_ring_lines,
            ext_ring_format: view.runtime_meta.ext_ring_format,
            declared_max_delay_48k: view.runtime_meta.declared_max_delay_48k.get(),
            dc_blocker_pole_5hz: view.runtime_meta.dc_blocker_pole_5hz.get(),
            tail_time_ms: view.runtime_meta.tail_time_ms.get(),
            ro_pole_48k: ro_pole,
            switch_xfade_samples: view.runtime_meta.switch_xfade_samples.get(),
            content_hash_prefix: view.runtime_meta.content_hash_prefix,
            controls: view.control_descriptors,
            cp1_overflow: view.cp1_overflow,
            tsyn: view.tsyn_entries,
            cext: view.cext,
            arena0: &a0,
            arena1: a1.as_ref(),
            arena2: a2.as_ref(),
            arena3: a3.as_ref(),
        })
    {
        *dnam_bin = reencoded;
    }

    if let Ok(rail) = run_phase2_rail_burst(dnam_bin, Tier::Lite, 48_000.0, 3.0)
        && rail.flagged
    {
        let msg = format!(
            "[CLAMP-RAIL] 3-phase corner burst saturated; scale {:.3}",
            rail.safe_scale
        );
        if !ar.phase2_rail_warnings.contains(&msg) {
            ar.phase2_rail_warnings.push(msg);
        }
    }
}

/// Runs the full 3-Phase stability, dynamic excursion, rail burst, and `+6 dB` readout head-gain
/// audit across `SONA v3.0` `.dnam-bin` (`Path 2`, mutating out-of-bounds poles/delays/gains).
pub fn run_full_3phase_arena_audit(manifest: &mut DnamProjManifest, dnam_bin: &mut Vec<u8>) {
    run_3phase_audit_inner(manifest, dnam_bin, true);
}

/// Runs the head-gain + dynamic rail 3-Phase audit while keeping `StarCenter` backbone weights
/// untouched (`Path 1`).
pub fn run_head_only_3phase_arena_audit(manifest: &mut DnamProjManifest, dnam_bin: &mut Vec<u8>) {
    run_3phase_audit_inner(manifest, dnam_bin, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::schema::DnamProjManifest;
    use crate::proj::vault::initialize_backbone_dnam_bin_if_needed;

    #[test]
    fn test_three_phase_control_space_audit_and_prescription_report() {
        let ((s_min, s_max), deg0, clamped) = audit_phase1_clamp_range((0.0, 10.0), (2.0, 8.0), 1);
        assert_eq!((s_min, s_max), (2.0, 8.0));
        assert_eq!(deg0, 0);
        assert!(clamped);

        let (_, deg3, _) = audit_phase1_clamp_range((0.0, 10.0), (0.0, 10.0), 5);
        assert_eq!(deg3, 3);

        let obs_sw = BTreeSet::from(["off", "on"]);
        let (mask, disabled) = audit_phase1_clamp_switch(&["off", "on", "ultra"], &obs_sw);
        assert_eq!(mask, 0b011);
        assert_eq!(disabled, vec!["ultra".to_string()]);

        let (r_c, tau_c, g_c, a_c, ok) =
            verify_dynamic_stability_excursion(0.9990, 0.0020, 3.5, 1.0, 1.02, 0.85);
        assert!(!ok);
        assert_eq!(r_c, 0.9995);
        assert_eq!(tau_c, 4.0);
        assert_eq!(g_c, 0.9990);
        assert_eq!(a_c, 0.75);

        let (scale_ok, clamped_ok) = clamp_head_spectral_gain_linear(1.5);
        assert_eq!((scale_ok, clamped_ok), (1.0, false));
        let (scale_hi, clamped_hi) = clamp_head_spectral_gain_linear(4.0);
        assert!(clamped_hi && scale_hi < 0.5);
        assert_eq!(clamp_head_spectral_gain_linear(f32::NAN), (0.0, true));
        assert_eq!(clamp_head_spectral_gain_linear(-1.0), (0.0, true));

        let report = format_audit_prescription_report(
            "model",
            &["[CLAMP-RANGE] 'gain': Clamped declared [0.0, 10.0] -> safe [2.0, 8.0]"],
        );
        assert!(report.contains("DNAM TRAINING COMPLETE"));
        assert!(report.contains("Active Control Clamps Applied (1):"));

        let b_safe = compute_phase2_corner_barrier_loss(0.95, 0.85, 0.4, 32.0);
        let b_edge = compute_phase2_corner_barrier_loss(0.9994, 0.9989, 0.749, 4.01);
        let b_viol = compute_phase2_corner_barrier_loss(1.01, 1.02, 0.9, 2.0);
        let b_nan = compute_phase2_corner_barrier_loss(f64::NAN, 0.85, 0.4, 32.0);
        assert!((0.0..1.0).contains(&b_safe));
        assert!(b_edge > b_safe && b_edge < 100.0);
        assert!(b_viol > 1.0e4);
        assert!(b_nan >= 1.0e6);

        let mut dnam_bin = initialize_backbone_dnam_bin_if_needed(None).expect("neutral dnam-bin");
        let (occ_lo, scale_lo, locked_lo) = audit_phase2_gru_state_saturation(&dnam_bin, 0.5);
        assert!(!locked_lo && scale_lo == 1.0 && occ_lo < 0.95);
        let (_, _, locked_nan) = audit_phase2_gru_state_saturation(&dnam_bin, f32::NAN);
        assert!(locked_nan);

        // Trigger full 3-phase .dnam-bin audit clamping on out-of-bounds Kautz, transport, and readout
        let view = DnamBinView::parse(&dnam_bin).unwrap();
        let mut a0 = *view.arena0;
        a0.conv_kautz_lite[0].r_low[0] = F32::new(1.02);
        a0.transport_lite.delay_samples[0] = F32::new(2.0);
        a0.transport_lite.g_loop = F32::new(1.05);
        a0.linear_readout_lite.direct_s2_u_gain[0] = F32::new(25.0);
        dnam_bin = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: view.header.reference_sample_rate_hz.get(),
            header_flags: view.header.flags.get(),
            ext_ring_pow2_frames: view.runtime_meta.ext_ring_pow2_frames,
            ext_ring_lines: view.runtime_meta.ext_ring_lines,
            ext_ring_format: view.runtime_meta.ext_ring_format,
            declared_max_delay_48k: view.runtime_meta.declared_max_delay_48k.get(),
            dc_blocker_pole_5hz: view.runtime_meta.dc_blocker_pole_5hz.get(),
            tail_time_ms: view.runtime_meta.tail_time_ms.get(),
            ro_pole_48k: view.runtime_meta.ro_pole_48k.get(),
            switch_xfade_samples: view.runtime_meta.switch_xfade_samples.get(),
            content_hash_prefix: view.runtime_meta.content_hash_prefix,
            controls: view.control_descriptors,
            cp1_overflow: view.cp1_overflow,
            tsyn: view.tsyn_entries,
            cext: view.cext,
            arena0: &a0,
            arena1: view.arena1,
            arena2: view.arena2,
            arena3: view.arena3,
        })
        .unwrap();
        let (occ_hi, scale_hi, locked_hi) = audit_phase2_gru_state_saturation(&dnam_bin, 40.0);
        assert!(locked_hi && occ_hi > 0.85 && scale_hi < 1.0);

        let mut manifest = DnamProjManifest::default_reference();
        run_full_3phase_arena_audit(&mut manifest, &mut dnam_bin);
        let audited = DnamBinView::parse(&dnam_bin).unwrap();
        assert!(audited.arena0.conv_kautz_lite[0].r_low[0].get() <= 0.9995);
        assert!(audited.arena0.transport_lite.delay_samples[0].get() >= 5.0);
        assert!(audited.arena0.transport_lite.g_loop.get() <= 0.9990);
        assert!(audited.arena0.linear_readout_lite.direct_s2_u_gain[0].get() < 25.0);
        let ar = manifest.audit_report.expect("audit report");
        assert!(ar.phase1_clamps.iter().any(|t| t == "CLAMP-STABILITY"));
        assert!(ar.phase3_clamps.iter().any(|t| t == "CLAMP-HEAD-GAIN"));
    }
}
