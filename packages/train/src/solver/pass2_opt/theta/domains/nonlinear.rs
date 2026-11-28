//! Continuous nonlinear subspace builders decomposing `CoreQuad`, `InterBlockTile`,
//! `SensoryQuad`, `Op-P` Crossbar, `Op-N[Pre]`, `ModalConvolver` poles, and `Op-T` quads
//! onto conditioned `u`-space coordinate bijections (`ref[train.pipeline.backbone-trajectory-opt]`).

use core::f64::consts::{FRAC_1_PI, PI};

use dnam_kernels::{
    MAX_G_PRE_GAIN, MAX_OUT_GAIN, MAX_PARALRU_DUHEM_GAMMA_F32, MAX_PARALRU_R_F32, MAX_SVF_K_NL,
    MIN_TPT_G, OP_N_PRE_MAX_BETA_Q, OP_N_PRE_MAX_ZOH_HOLD_DEPTH, TANK_MAX_G_LIN_F32,
};
use dnam_vm::SONA_CORE_QUAD_BLOCK_AND_LOCAL;

use crate::control::stability::{MAX_RO_POLE_48K_F64, MIN_DELAY_CROSSBAR_SAMP};
use crate::linalg::lbfgs::ParamBoundSpec;

use super::{ParamKind, ThetaEntry};

/// Conditioned `LogBox` in normalized frequency ratio `r = fc / fs = atan(g) / pi`
/// (`op_n_paralru_and_polyadaa.md:370`).
#[inline]
#[must_use]
pub(super) fn svf_g_freq_ratio_log_box() -> ParamBoundSpec {
    ParamBoundSpec::LogBox {
        lo: f64::from(MIN_TPT_G).atan() * FRAC_1_PI,
        hi: 64.0_f64.atan() * FRAC_1_PI,
    }
}

/// Conditioned `ComplementLog` for `lru_alpha_atk` / `sense_lru_atk` (`< 1e-7 -> 0.0` zero-snap).
#[inline]
#[must_use]
pub(super) fn lru_atk_complement_log() -> ParamBoundSpec {
    ParamBoundSpec::ComplementLog {
        lo_c: 1.0e-7,
        hi_c: 1.0,
    }
}

/// Conditioned `LogBox` for `svf_k` / `sense_svf_k` (`fp16`-safe normal floor `1e-3`).
#[inline]
#[must_use]
pub(super) fn svf_k_log_box() -> ParamBoundSpec {
    ParamBoundSpec::LogBox {
        lo: 1.0e-3,
        hi: 8.0,
    }
}

/// Prepared `SONA v4.0` `CoreQuadF64` dynamic pre-shaper / `SE-Bus` / output trim bounds
/// (gated out of default `ThetaLayout::for_tier` until `C3.1`).
#[allow(dead_code)]
#[must_use]
pub(super) fn v4_core_quad_bounds() -> [ParamBoundSpec; 4] {
    [
        ParamBoundSpec::Box {
            lo: 0.0,
            hi: f64::from(MAX_G_PRE_GAIN),
        },
        ParamBoundSpec::AsinhBoxNonNeg {
            scale: 1.0,
            limit: 1.0e6,
        },
        ParamBoundSpec::Box { lo: 0.0, hi: 1.0 },
        ParamBoundSpec::AsinhBoxSym {
            scale: 1.0,
            limit: f64::from(MAX_OUT_GAIN),
        },
    ]
}

/// Appends prepared `SONA v4.0` `CoreQuadF64` entries (`g0`, `g_inf`, `d_pre`, `se_w_sq`,
/// `se_d_bus`, `lin_mix`, `out_gain`) for Phase-3 `C3.1` opt-in callers.
#[allow(dead_code)]
pub(super) fn append_v4_core_quad_extra_entries(entries: &mut Vec<ThetaEntry>, q_count: usize) {
    let [g_pre_box, nonneg_box, unit_box, out_gain_box] = v4_core_quad_bounds();
    for q in 0..q_count {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::CoreG0(q, l), g_pre_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreGInf(q, l), g_pre_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreDPre(q, l), nonneg_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreSeWSq(q, l), nonneg_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreSeDBus(q, l), nonneg_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreLinMix(q, l), unit_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreOutGain(q, l), out_gain_box));
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::solver::pass2_opt) struct NonlinearSubspaceSpec {
    pub q_count: usize,
    pub active_tiles: usize,
    pub include_stereo_phase: bool,
    pub k_count: usize,
    pub active_opt_quads: usize,
    pub opt_feedforward_only_mask: [bool; 4],
    pub include_extras: bool,
    pub sr_hz: f64,
    pub max_tau: f64,
    pub kautz_r_max: f64,
}

fn append_core_quad_entries(entries: &mut Vec<ThetaEntry>, q_count: usize) {
    let r_hi = f64::from(MAX_PARALRU_R_F32);
    let svf_g_box = svf_g_freq_ratio_log_box();
    let rel_clog = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - r_hi,
        hi_c: 1.0,
    };
    let atk_clog = lru_atk_complement_log();
    let unit_box = ParamBoundSpec::Box { lo: 0.0, hi: 1.0 };
    let svf_k_box = svf_k_log_box();
    let k_nl_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 1.0,
        limit: f64::from(MAX_SVF_K_NL),
    };
    let tone_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 16.0,
    };
    let duhem_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 2.0,
        limit: f64::from(MAX_PARALRU_DUHEM_GAMMA_F32),
    };
    let sag_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 8.0,
    };
    let adaa_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 32.0,
    };
    let glu_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 4.0,
    };

    for q in 0..q_count {
        let is_block0 = SONA_CORE_QUAD_BLOCK_AND_LOCAL[q].0 == 0;
        for l in 0..4 {
            entries.push(ThetaEntry::f32(ParamKind::CoreSvfG(q, l), svf_g_box));
            entries.push(ThetaEntry::f32(ParamKind::CoreLruRel(q, l), rel_clog));
            entries.push(ThetaEntry::f32(ParamKind::CoreLruAtk(q, l), atk_clog));
            entries.push(ThetaEntry::fp16(ParamKind::CoreSvfK(q, l), svf_k_box));
            if is_block0 {
                entries.push(ThetaEntry::fp16(ParamKind::CoreSvfKNl(q, l), k_nl_box));
            }
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneLp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneBp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneHp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneAp(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreToneDir(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreLruKap(q, l), unit_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreLruDuhem(q, l), duhem_box));
            entries.push(ThetaEntry::fp16(
                ParamKind::CoreLruSignedMix(q, l),
                unit_box,
            ));
            entries.push(ThetaEntry::fp16(
                ParamKind::CoreLruStereoLink(q, l),
                unit_box,
            ));
            entries.push(ThetaEntry::fp16(ParamKind::CoreSagDepth(q, l), sag_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreAdaaDrive(q, l), adaa_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreLruBias(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreWavefolder(q, l), unit_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreGluMix(q, l), glu_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreInMid(q, l), tone_box));
            entries.push(ThetaEntry::fp16(ParamKind::CoreInSide(q, l), tone_box));
        }
    }
}

fn append_tile_entries(entries: &mut Vec<ThetaEntry>, active_tiles: usize) {
    let w_tile_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 8.0,
    };
    for t in 0..active_tiles {
        for s in 0..4 {
            for d in 0..4 {
                entries.push(ThetaEntry::f32(ParamKind::TileOdd(t, s, d), w_tile_box));
                entries.push(ThetaEntry::fp16(ParamKind::TileEven(t, s, d), w_tile_box));
                entries.push(ThetaEntry::fp16(ParamKind::TileLru(t, s, d), w_tile_box));
            }
        }
    }
}

fn append_sensory_entries(entries: &mut Vec<ThetaEntry>) {
    let r_hi = f64::from(MAX_PARALRU_R_F32);
    let svf_g_box = svf_g_freq_ratio_log_box();
    let rel_clog = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - r_hi,
        hi_c: 1.0,
    };
    let atk_clog = lru_atk_complement_log();
    let unit_box = ParamBoundSpec::Box { lo: 0.0, hi: 1.0 };
    let svf_k_box = svf_k_log_box();
    let tone_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 16.0,
    };
    let vca_box = ParamBoundSpec::AsinhBoxSym {
        scale: 4.0,
        limit: 512.0,
    };
    let sin_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 4.0,
    };
    for l in 0..4 {
        entries.push(ThetaEntry::f32(ParamKind::SenseSvfG(l), svf_g_box));
        entries.push(ThetaEntry::f32(ParamKind::SenseLruRel(l), rel_clog));
        entries.push(ThetaEntry::f32(ParamKind::SenseLruAtk(l), atk_clog));
        entries.push(ThetaEntry::fp16(ParamKind::SenseSvfK(l), svf_k_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseToneLp(l), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseToneBp(l), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseToneHp(l), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseLruKappa(l), unit_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseVcaW(l), vca_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseInL(l), sin_box));
        entries.push(ThetaEntry::fp16(ParamKind::SenseInR(l), sin_box));
    }
}

fn append_crossbar_entries(entries: &mut Vec<ThetaEntry>, spec: &NonlinearSubspaceSpec) {
    let glu_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 4.0,
    };
    let w_tile_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 8.0,
    };
    let phasor_lim = 23_520.0_f64.min(0.49 * spec.sr_hz.max(44_100.0));
    let freq_box = ParamBoundSpec::AsinhBoxSym {
        scale: 10.0,
        limit: phasor_lim,
    };
    let angle_box = ParamBoundSpec::Box { lo: -PI, hi: PI };
    for l in 0..4 {
        entries.push(ThetaEntry::f32(ParamKind::PhasorFreqHz(l), freq_box));
    }
    for c in 0..2 {
        entries.push(ThetaEntry::fp16(ParamKind::CrossbarOpGain(c), glu_box));
    }
    for b in 0..3 {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::CarrierBasisW(b, l), w_tile_box));
        }
    }
    for q in 0..spec.q_count {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::OpPToCore(q, l), w_tile_box));
        }
    }
    for q in 0..spec.q_count {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::PhasorToCore(q, l), w_tile_box));
        }
    }
    if spec.include_stereo_phase {
        for l in 0..4 {
            entries.push(ThetaEntry::fp16(ParamKind::PhasorStereoPhase(l), angle_box));
        }
    }
    if spec.include_extras {
        for q in 0..spec.active_opt_quads {
            if spec.opt_feedforward_only_mask[q] {
                continue;
            }
            for lane in 0..4 {
                let i = q * 4 + lane;
                for c in 0..2 {
                    entries.push(ThetaEntry::fp16(ParamKind::OptToLr(i, c), glu_box));
                }
            }
        }
    }
}

fn append_op_n_pre_entries(entries: &mut Vec<ThetaEntry>) -> usize {
    let unit_box = ParamBoundSpec::Box { lo: 0.0, hi: 1.0 };
    entries.push(ThetaEntry::f32(ParamKind::OpNPreGammaBc, unit_box));
    let zoh_idx = entries.len();
    entries.push(ThetaEntry::f32(
        ParamKind::OpNPreZohHoldDepth,
        ParamBoundSpec::Box {
            lo: 0.0,
            hi: f64::from(OP_N_PRE_MAX_ZOH_HOLD_DEPTH),
        },
    ));
    entries.push(ThetaEntry::f32(
        ParamKind::OpNPreBetaQ,
        ParamBoundSpec::AsinhBoxNonNeg {
            scale: 1.0,
            limit: f64::from(OP_N_PRE_MAX_BETA_Q),
        },
    ));
    zoh_idx
}

fn append_modal_entries(entries: &mut Vec<ThetaEntry>, spec: &NonlinearSubspaceSpec) {
    entries.push(ThetaEntry::f32(
        ParamKind::RoPole48k,
        ParamBoundSpec::ComplementLog {
            lo_c: 1.0 - MAX_RO_POLE_48K_F64,
            hi_c: 1.0,
        },
    ));
    let r_hi = spec.kautz_r_max.clamp(0.5, 0.9995);
    let kautz_r_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - r_hi,
        hi_c: 1.0 - 0.25,
    };
    let (th_min, th_max) =
        crate::solver::pass1_probe::kautz_fit::audible_kautz_theta_bounds(spec.sr_hz);
    let kautz_theta_box = ParamBoundSpec::LogBox {
        lo: th_min,
        hi: th_max,
    };
    for k in 0..spec.k_count {
        entries.push(ThetaEntry::f32(ParamKind::KautzRLow(k), kautz_r_box));
        entries.push(ThetaEntry::f32(ParamKind::KautzTheta(k), kautz_theta_box));
        if spec.include_extras {
            entries.push(ThetaEntry::f32(
                ParamKind::KautzBGamma(k),
                ParamBoundSpec::LogBox {
                    lo: 1.0e-4,
                    hi: 8.0,
                },
            ));
        }
    }
}

fn append_op_t_entries(entries: &mut Vec<ThetaEntry>, spec: &NonlinearSubspaceSpec) {
    let unit_box = ParamBoundSpec::Box { lo: 0.0, hi: 1.0 };
    let angle_box = ParamBoundSpec::Box { lo: -PI, hi: PI };
    let tone_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 16.0,
    };
    let tau_max = spec.max_tau.clamp(MIN_DELAY_CROSSBAR_SAMP + 1.0, 48_000.0);
    let span_max = (tau_max - MIN_DELAY_CROSSBAR_SAMP).max(1.0);
    let delay_box = ParamBoundSpec::LogBox {
        lo: MIN_DELAY_CROSSBAR_SAMP,
        hi: tau_max,
    };
    let span_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 32.0,
        limit: span_max,
    };
    let gloop_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - f64::from(TANK_MAX_G_LIN_F32),
        hi_c: 1.0,
    };
    let rdc_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - f64::from(MAX_PARALRU_R_F32),
        hi_c: 1.0 - 0.95,
    };
    let damp_box = ParamBoundSpec::ComplementLog {
        lo_c: 1.0 - 0.9995,
        hi_c: 1.0,
    };
    let lfo_rate_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 1.0,
        limit: 16.0,
    };
    let lfo_depth_box = ParamBoundSpec::AsinhBoxNonNeg {
        scale: 4.0,
        limit: 512.0,
    };
    let tap_box = ParamBoundSpec::AsinhBoxSym {
        scale: 1.0,
        limit: 2.0,
    };
    let ap_box = ParamBoundSpec::Box {
        lo: -0.75,
        hi: 0.75,
    };
    let sigma_box = ParamBoundSpec::LogBox {
        lo: 1.0e-3,
        hi: 0.984_375,
    };
    let freeze_box = ParamBoundSpec::Box { lo: 0.0, hi: 0.999 };

    for q in 0..spec.active_opt_quads {
        if spec.opt_feedforward_only_mask[q] {
            for l in 0..4 {
                entries.push(ThetaEntry::f32(ParamKind::OpTDelay(q, l), delay_box));
                entries.push(ThetaEntry::fp16(ParamKind::OpTInputGain(q, l), tap_box));
                if spec.include_extras {
                    entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapL(q, l), tap_box));
                    entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapR(q, l), tap_box));
                }
            }
            continue;
        }
        for l in 0..4 {
            entries.push(ThetaEntry::f32(ParamKind::OpTDelay(q, l), delay_box));
            entries.push(ThetaEntry::f32(ParamKind::OpTRampSpan(q, l), span_box));
            entries.push(ThetaEntry::fp16(
                ParamKind::OpTLfoDepth(q, l),
                lfo_depth_box,
            ));
            entries.push(ThetaEntry::fp16(ParamKind::OpTInputGain(q, l), tap_box));
            entries.push(ThetaEntry::fp16(ParamKind::OpTAllpassG(q, l), ap_box));
            if spec.include_extras {
                entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapL(q, l), tap_box));
                entries.push(ThetaEntry::fp16(ParamKind::OpTWetTapR(q, l), tap_box));
            }
        }
        entries.push(ThetaEntry::f32(ParamKind::OpTGLoop(q), gloop_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTRDc(q), rdc_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTDampPole(q), damp_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTLfoRate(q), lfo_rate_box));
        entries.push(ThetaEntry::f32(ParamKind::OpTRampRate(q), tone_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTFbPsi(q), angle_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTSigmaKnee(q), sigma_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTFreezeMix(q), freeze_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTWindowDepth(q), unit_box));
        entries.push(ThetaEntry::fp16(ParamKind::OpTScatPsi(q), angle_box));
    }
    if spec.include_extras && spec.active_opt_quads > 0 {
        entries.push(ThetaEntry::fp16(ParamKind::TankDryMix, tap_box));
    }
}

/// Appends the canonical full-subspace continuous nonlinear descriptors matching
/// `webgpu_and_varpro_solver.md` lines `93–128`, returning `zoh_hold_depth_idx`.
pub(in super::super) fn append_nonlinear_entries(
    entries: &mut Vec<ThetaEntry>,
    spec: &NonlinearSubspaceSpec,
) -> Option<usize> {
    append_core_quad_entries(entries, spec.q_count);
    append_tile_entries(entries, spec.active_tiles);
    append_sensory_entries(entries);
    append_crossbar_entries(entries, spec);
    let zoh_hold_depth_idx = Some(append_op_n_pre_entries(entries));
    append_modal_entries(entries, spec);
    append_op_t_entries(entries, spec);
    zoh_hold_depth_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conditioned_param_bound_specs_and_v3_op_t_bounds() {
        let mut entries = Vec::new();
        let spec = NonlinearSubspaceSpec {
            q_count: 4,
            active_tiles: 3,
            include_stereo_phase: false,
            k_count: 8,
            active_opt_quads: 1,
            opt_feedforward_only_mask: [false; 4],
            include_extras: true,
            sr_hz: 48_000.0,
            max_tau: 128.0,
            kautz_r_max: 0.9992,
        };
        let zoh = append_nonlinear_entries(&mut entries, &spec);
        assert!(zoh.is_some());
        assert_eq!(entries.len(), 637);

        let check = |pred: fn(&ParamKind) -> bool| -> ParamBoundSpec {
            entries
                .iter()
                .find(|e| pred(&e.kind))
                .expect("expected ParamKind")
                .bound
        };
        assert_eq!(
            check(|k| matches!(k, ParamKind::CoreLruAtk(0, 0))),
            ParamBoundSpec::ComplementLog {
                lo_c: 1.0e-7,
                hi_c: 1.0
            }
        );
        assert_eq!(
            check(|k| matches!(k, ParamKind::SenseLruAtk(0))),
            ParamBoundSpec::ComplementLog {
                lo_c: 1.0e-7,
                hi_c: 1.0
            }
        );
        assert_eq!(
            check(|k| matches!(k, ParamKind::CoreSvfK(0, 0))),
            ParamBoundSpec::LogBox {
                lo: 1.0e-3,
                hi: 8.0
            }
        );
        assert_eq!(
            check(|k| matches!(k, ParamKind::SenseSvfK(0))),
            ParamBoundSpec::LogBox {
                lo: 1.0e-3,
                hi: 8.0
            }
        );
        assert_eq!(
            check(|k| matches!(k, ParamKind::CoreSvfG(0, 0))),
            svf_g_freq_ratio_log_box()
        );
        assert_eq!(
            check(|k| matches!(k, ParamKind::SenseSvfG(0))),
            svf_g_freq_ratio_log_box()
        );
        // Zero-snap edge below 1e-7 preserved on ComplementLog atk.
        assert_eq!(
            ParamBoundSpec::ComplementLog {
                lo_c: 1.0e-7,
                hi_c: 1.0
            }
            .snap_zero_edge(5.0e-8),
            0.0
        );
        // v3 Op-T sigma_knee stays <= 0.984375 until Phase 3.
        assert_eq!(
            check(|k| matches!(k, ParamKind::OpTSigmaKnee(0))),
            ParamBoundSpec::LogBox {
                lo: 1.0e-3,
                hi: 0.984_375
            }
        );
    }
}
