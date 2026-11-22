//! Arena 0 Table-4 invariant clamping and Phase-2 2,048-sample differential stereo
//! `[CLAMP-RAIL]` lockup burst (`ref[train.audit.phase2.clamp-rail]`).

use dnam_kernels::{
    MAX_PARALRU_R_F32, MIN_TPT_G, OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MAX_ZOH_HOLD_DEPTH,
    TANK_MAX_G_LIN_F32, TANK_MAX_SIGMA_SAT_F32, TANK_MAX_SIGMA_SAT_FP16, f32_to_fp16, fp16_to_f32,
};
use dnam_vm::bin::{Arena0LiteWire, DNBN_FLAG_STEREO_OUT, DnamBinView};
use dnam_vm::math::{DefaultMathBackend, TANK_DC_BLOCKER_48K_F32};
use dnam_vm::{SonaVm, Tier};
use zerocopy::byteorder::little_endian::{F32, U16};

use crate::error::TrainError;

use super::{
    MAX_RO_POLE_48K_F32, RAIL_BURST_FRAMES, RAIL_OUTPUT_CEIL, RAIL_OUTPUT_HIT_THRESHOLD,
    RAIL_PREACT_HIT_THRESHOLD, RAIL_STIM_OMEGA_A, RAIL_STIM_OMEGA_B, SAT_OCCUPANCY_LIMIT,
};

fn clamp_f32_wire(w: F32, lo: f32, hi: f32, fallback: f32) -> F32 {
    let v = w.get();
    F32::new(if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    })
}

fn clamp_fp16_wire(w: U16, lo: f32, hi: f32, fallback: f32) -> U16 {
    let v = fp16_to_f32(w.get());
    let clamped = if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    };
    U16::new(f32_to_fp16(clamped))
}

/// Clamps an [`Arena0LiteWire`] control/transport header block to canonical Table-4 stability invariants.
pub fn enforce_arena0_stability_invariants(a0: &mut Arena0LiteWire, ro_pole_48k: &mut f32) {
    *ro_pole_48k = if ro_pole_48k.is_finite() {
        ro_pole_48k.clamp(0.0, MAX_RO_POLE_48K_F32)
    } else {
        0.0
    };

    let t = &mut a0.transport_lite;
    for d in &mut t.delay_samples {
        *d = clamp_f32_wire(*d, 5.0, 48_000.0, 5.0);
    }
    t.g_loop = clamp_f32_wire(t.g_loop, 0.0, TANK_MAX_G_LIN_F32, 0.0);
    t.r_dc = clamp_f32_wire(t.r_dc, 0.95, 0.9999, TANK_DC_BLOCKER_48K_F32);
    t.damp_pole = clamp_f32_wire(t.damp_pole, 0.0, 0.9995, 0.0);
    t.lfo_rate_hz = clamp_f32_wire(t.lfo_rate_hz, 0.0, 16.0, 0.0);
    t.ramp_rate_hz = clamp_f32_wire(t.ramp_rate_hz, -16.0, 16.0, 0.0);
    for ld in &mut t.lfo_depths_fp16 {
        *ld = clamp_fp16_wire(*ld, 0.0, 512.0, 0.0);
    }
    for ag in &mut t.allpass_g_fp16 {
        *ag = clamp_fp16_wire(*ag, -0.75, 0.75, 0.0);
    }
    let sk = fp16_to_f32(t.sigma_knee_fp16.get());
    t.sigma_knee_fp16 = if !sk.is_finite() {
        U16::new(TANK_MAX_SIGMA_SAT_FP16)
    } else {
        U16::new(f32_to_fp16(sk.clamp(1.0e-3, TANK_MAX_SIGMA_SAT_F32)).min(TANK_MAX_SIGMA_SAT_FP16))
    };
    t.freeze_mix_fp16 = clamp_fp16_wire(t.freeze_mix_fp16, 0.0, 1.0, 0.0);
    t.window_depth_fp16 = clamp_fp16_wire(t.window_depth_fp16, 0.0, 1.0, 0.0);

    let pre = &mut a0.crossbar.op_n_pre_base;
    pre.gamma_bc = clamp_f32_wire(pre.gamma_bc, 0.0, 1.0, 0.0);
    pre.zoh_hold_depth = clamp_f32_wire(pre.zoh_hold_depth, 0.0, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, 0.0);
    pre.beta_q = clamp_f32_wire(pre.beta_q, 0.0, OP_N_PRE_MAX_BETA_Q, 0.0);

    for cq in &mut a0.core_quads {
        for l in 0..4 {
            cq.svf_g[l] = clamp_f32_wire(cq.svf_g[l], MIN_TPT_G, 64.0, 0.25);
            cq.lru_alpha_rel[l] = clamp_f32_wire(cq.lru_alpha_rel[l], 0.0, MAX_PARALRU_R_F32, 0.0);
            cq.lru_alpha_atk[l] = clamp_f32_wire(cq.lru_alpha_atk[l], 0.0, 1.0, 0.0);
            cq.glu_mix_fp16[l] = clamp_fp16_wire(cq.glu_mix_fp16[l], -4.0, 4.0, 0.0);
        }
    }

    let theta_hi = core::f32::consts::PI - 1.0e-4;
    for kb in &mut a0.conv_kautz_lite {
        for l in 0..4 {
            let rl = kb.r_low[l].get();
            let rl_c = if rl.is_finite() {
                rl.clamp(0.0, 0.9995)
            } else {
                0.0
            };
            kb.r_low[l] = F32::new(rl_c);
            let dr = kb.dr[l].get();
            let max_dr = (0.9995 - rl_c).max(0.0);
            let dr_c = if dr.is_finite() {
                dr.clamp(0.0, max_dr)
            } else {
                0.0
            };
            kb.dr[l] = F32::new(dr_c);
            kb.theta[l] = clamp_f32_wire(kb.theta[l], 0.0, theta_hi, 0.0);
            let bg = kb.b_gamma[l].get();
            kb.b_gamma[l] = F32::new(if bg.is_finite() { bg.max(0.0) } else { 0.0 });
        }
    }

    let dry_mix = fp16_to_f32(a0.control_bus_lite.tank_dry_mix_fp16.get());
    let clamped_dry = if dry_mix.is_finite() {
        dry_mix.clamp(-2.0, 2.0)
    } else {
        1.0
    };
    a0.control_bus_lite.tank_dry_mix_fp16 = U16::new(f32_to_fp16(clamped_dry));
    let rho_drift = a0.control_bus_lite.rho_drift_train.get();
    let clamped_rho = if rho_drift.is_finite() && rho_drift > 0.0 {
        rho_drift.clamp(f32::MIN_POSITIVE, 0.9999)
    } else {
        0.999
    };
    a0.control_bus_lite.rho_drift_train = F32::new(clamped_rho);
    let trust = a0.control_bus_lite.trust_radius_mh.get();
    let clamped_trust = if trust.is_finite() && trust > 0.0 {
        trust.clamp(0.01, 16.0)
    } else {
        2.5
    };
    a0.control_bus_lite.trust_radius_mh = F32::new(clamped_trust);
    a0.control_bus_lite.rho_head_and_ctrl_tail[0] = F32::new(clamped_trust);
}

/// Outcome of the Phase-2 2,048-sample differential stereo `[CLAMP-RAIL]` burst.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RailBurstReport {
    /// Count of samples exceeding `|y_c[n]| >= 25.0` or non-finite.
    pub sat_hits: usize,
    /// Fraction `sat_hits / 2048`.
    pub occupancy: f64,
    /// Peak output sample magnitude observed across the burst.
    pub max_abs_out: f32,
    /// Whether any output sample was `NaN` or infinite.
    pub non_finite: bool,
    /// Whether rail lockup / saturation occupancy `> 0.95` / ceiling breach triggered.
    pub flagged: bool,
    /// Safe input attenuation factor (`clamp(2.0 / max(eff_drive, 2.0), 0.05, 1.0)` when flagged).
    pub safe_scale: f64,
}

/// Runs the normative Phase-2 `[CLAMP-RAIL]` 2,048-sample differential stereo `[+x[n], -x[n]]`
/// stability burst through `dnam_vm::SonaVm` on `tier` without gradient tracking
/// (`ref[train.audit.phase2.clamp-rail]`).
pub fn run_phase2_rail_burst(
    dnam_bin: &[u8],
    tier: Tier,
    sample_rate_hz: f32,
    d_corner: f64,
) -> Result<RailBurstReport, TrainError> {
    let view = DnamBinView::parse(dnam_bin)?;
    let is_stereo = (view.header.flags.get() & DNBN_FLAG_STEREO_OUT) != 0;
    let eff_drive = if d_corner.is_finite() {
        d_corner.clamp(0.1, 64.0)
    } else {
        1.0
    };
    let mut stim = [0.0_f32; RAIL_BURST_FRAMES];
    for n in 0..RAIL_BURST_FRAMES {
        let nf = n as f64;
        stim[n] = (0.85
            * eff_drive
            * ((RAIL_STIM_OMEGA_A * nf).sin() + 0.35 * (RAIL_STIM_OMEGA_B * nf).cos()))
            as f32;
    }

    let eval_polarity = |sign_r: f32| -> Result<(usize, f32, bool), TrainError> {
        let mut vm =
            SonaVm::<DefaultMathBackend>::open_dnam_bin_owned(dnam_bin, tier, sample_rate_hz)?;
        let mut in_r = [0.0_f32; RAIL_BURST_FRAMES];
        for (dst, &s) in in_r.iter_mut().zip(&stim) {
            *dst = sign_r * s;
        }
        let mut out_l = [0.0_f32; RAIL_BURST_FRAMES];
        let mut out_r = [0.0_f32; RAIL_BURST_FRAMES];
        let mut off = 0_usize;
        while off < RAIL_BURST_FRAMES {
            let chunk = (RAIL_BURST_FRAMES - off).min(64);
            vm.process_block(
                &stim[off..off + chunk],
                &in_r[off..off + chunk],
                &mut out_l[off..off + chunk],
                &mut out_r[off..off + chunk],
            )?;
            off += chunk;
        }
        let mut sat_hits = 0_usize;
        let mut max_abs_out = 0.0_f32;
        let mut non_finite = false;
        for (&yl, &yr) in out_l.iter().zip(&out_r) {
            if !yl.is_finite() || !yr.is_finite() {
                non_finite = true;
                sat_hits += 1;
                continue;
            }
            let mag = yl.abs().max(yr.abs());
            max_abs_out = max_abs_out.max(mag);
            if mag >= RAIL_OUTPUT_HIT_THRESHOLD {
                sat_hits += 1;
            }
        }
        Ok((sat_hits, max_abs_out, non_finite))
    };

    let (mut sat_hits, mut max_abs_out, mut non_finite) = eval_polarity(-1.0)?;
    if !is_stereo {
        let (sh2, max2, nf2) = eval_polarity(1.0)?;
        sat_hits = sat_hits.max(sh2);
        max_abs_out = max_abs_out.max(max2);
        non_finite |= nf2;
    }

    let occupancy = (sat_hits as f64) / (RAIL_BURST_FRAMES as f64);
    let flagged = non_finite || max_abs_out > RAIL_OUTPUT_CEIL || occupancy > SAT_OCCUPANCY_LIMIT;
    let safe_scale = if flagged {
        (2.0 / eff_drive.max(2.0)).clamp(0.05, 1.0)
    } else {
        1.0
    };
    Ok(RailBurstReport {
        sat_hits,
        occupancy,
        max_abs_out,
        non_finite,
        flagged,
        safe_scale,
    })
}

/// Evaluates whether a scalar pre-activation trajectory stayed strictly below the
/// `[CLAMP-RAIL]` occupancy ceiling (`|u_nl| >= 8.0`).
#[must_use]
pub fn check_preact_occupancy(u_nl: &[f64]) -> bool {
    if u_nl.is_empty() {
        return true;
    }
    let hits = u_nl
        .iter()
        .filter(|&&v| !v.is_finite() || v.abs() >= RAIL_PREACT_HIT_THRESHOLD)
        .count();
    (hits as f64) / (u_nl.len() as f64) <= SAT_OCCUPANCY_LIMIT
}

#[cfg(test)]
mod tests {
    use super::*;
    use dnam_vm::DnamBinBuildInput;
    use dnam_vm::bin::encode_dnam_bin;

    #[test]
    fn test_arena0_invariants_and_rail_burst() {
        assert!(check_preact_occupancy(&[1.0, 2.0, 9.0, 0.5]));
        assert!(!check_preact_occupancy(&[9.0; 25]));

        let mut a0_safe = DnamBinBuildInput::default_arena0_lite();
        a0_safe.linear_readout_lite.direct_s2_u_gain = [F32::new(1.0), F32::new(1.0)];
        let mut ro_p = 0.99_f32;
        enforce_arena0_stability_invariants(&mut a0_safe, &mut ro_p);
        let bin_safe = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 100.0,
            ro_pole_48k: ro_p,
            switch_xfade_samples: 64,
            content_hash_prefix: [7; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0_safe,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap();
        let rep_safe = run_phase2_rail_burst(&bin_safe, Tier::Lite, 48_000.0, 1.0).unwrap();
        assert!(!rep_safe.flagged && rep_safe.safe_scale == 1.0);

        let mut a0_rdc = a0_safe;
        a0_rdc.transport_lite.r_dc = F32::new(0.985);
        a0_rdc.transport_lite.sigma_knee_fp16 = U16::new(0xFC00);
        a0_rdc.conv_kautz_lite[0].r_low[0] = F32::new(0.5);
        a0_rdc.conv_kautz_lite[0].dr[0] = F32::new(0.999);
        a0_rdc.conv_kautz_lite[0].r_low[1] = F32::new(0.5);
        a0_rdc.conv_kautz_lite[0].dr[1] = F32::new(-0.2);
        enforce_arena0_stability_invariants(&mut a0_rdc, &mut ro_p);
        assert_eq!(a0_rdc.transport_lite.r_dc.get(), 0.985);
        assert_eq!(
            a0_rdc.transport_lite.sigma_knee_fp16.get(),
            TANK_MAX_SIGMA_SAT_FP16
        );
        let kb0 = &a0_rdc.conv_kautz_lite[0];
        assert!(kb0.dr[0].get() >= 0.0 && kb0.r_low[0].get() + kb0.dr[0].get() <= 0.9995 + 1e-6);
        assert_eq!(kb0.dr[1].get(), 0.0);

        let mut a0_hot = a0_safe;
        a0_hot.linear_readout_lite.direct_s2_u_gain = [F32::new(12.0), F32::new(12.0)];
        let bin_hot = encode_dnam_bin(&DnamBinBuildInput {
            reference_sample_rate_hz: 48_000,
            header_flags: DNBN_FLAG_STEREO_OUT,
            ext_ring_pow2_frames: 0,
            ext_ring_lines: 0,
            ext_ring_format: 0,
            declared_max_delay_48k: 4_096,
            dc_blocker_pole_5hz: TANK_DC_BLOCKER_48K_F32,
            tail_time_ms: 100.0,
            ro_pole_48k: 0.9,
            switch_xfade_samples: 64,
            content_hash_prefix: [8; 8],
            controls: &[],
            cp1_overflow: &[],
            tsyn: &[],
            cext: None,
            arena0: &a0_hot,
            arena1: None,
            arena2: None,
            arena3: None,
        })
        .unwrap();
        let rep_hot = run_phase2_rail_burst(&bin_hot, Tier::Lite, 48_000.0, 40.0).unwrap();
        assert!(
            rep_hot.flagged && (rep_hot.safe_scale - 0.05).abs() < 1.0e-12,
            "unexpected rep_hot: {rep_hot:?}"
        );
    }
}
