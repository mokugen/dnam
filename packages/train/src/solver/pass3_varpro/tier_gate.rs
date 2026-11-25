//! Complexity-penalized upper-tier inheritance & `fp16` `PrefixOverride` synchronization for
//! Pass-3 nested `VarPro` (`ref[train.pipeline.nested-varpro-schur]`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};

use super::super::model::{SonaModelF64, TierReadoutPrefixF64};
use super::features::unpack_weights_into_prefix;
use super::spectral_refine::{compute_channel_hf_mse, compute_channel_mse};

#[derive(Clone, Debug)]
pub(super) struct TierGateCurrent {
    pub wl: Vec<f64>,
    pub wr: Vec<f64>,
    pub mse: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TierGatePrev<'a> {
    pub wl: &'a [f64],
    pub wr: &'a [f64],
    pub mse: f64,
    pub same_basis: bool,
}

#[derive(Clone, Debug)]
pub(super) struct TierInheritanceDecision {
    pub inherited: bool,
    pub wl: Vec<f64>,
    pub wr: Vec<f64>,
    pub mse: f64,
}

const INHERIT_HIGH_ACCURACY_NESR_CEIL: f64 = 1.0e-4;
const INHERIT_DEEP_FLOOR_NESR_CEIL: f64 = 1.0e-5;
const INHERIT_DEEP_MIN_REL_DROP: f64 = 0.50;

const _: () = assert!(
    INHERIT_DEEP_FLOOR_NESR_CEIL > 0.0
        && INHERIT_DEEP_FLOOR_NESR_CEIL < INHERIT_HIGH_ACCURACY_NESR_CEIL
        && INHERIT_DEEP_MIN_REL_DROP > 0.08
        && INHERIT_DEEP_MIN_REL_DROP < 1.0
);

/// Complexity-penalized upper-tier gate: when adjacent tiers share the exact same underlying
/// nonlinear/Kautz/OS basis and the lower tier already achieves high accuracy (`prev_nesr < 1e-4`),
/// requires the extra upper-tier linear parameters to justify their complexity (`>= 50%` / `3 dB`
/// residual drop below `-50 dB`, else `clamp(0.001 * delta_dim, 0.02, 0.08)`); otherwise inherits
/// `prev` zero-padded into `dim_t` so holdout `L2`/`L3` metrics stay monotonically non-regressing.
pub(super) fn maybe_inherit_prev_tier_weights(
    ti: usize,
    dim_t: usize,
    tgt_var: f64,
    cur: TierGateCurrent,
    prev: TierGatePrev<'_>,
) -> TierInheritanceDecision {
    if prev.same_basis
        && ti >= 1
        && dim_t > prev.wl.len()
        && prev.wl.len() == prev.wr.len()
        && prev.mse.is_finite()
        && cur.mse.is_finite()
        && tgt_var.is_finite()
        && tgt_var > 1.0e-18
    {
        let prev_nesr = prev.mse / tgt_var;
        if prev_nesr < INHERIT_HIGH_ACCURACY_NESR_CEIL {
            let min_rel_drop = if prev_nesr < INHERIT_DEEP_FLOOR_NESR_CEIL {
                INHERIT_DEEP_MIN_REL_DROP
            } else {
                (((dim_t - prev.wl.len()) as f64) * 0.001).clamp(0.02, 0.08)
            };
            if cur.mse > prev.mse * (1.0 - min_rel_drop) {
                let mut wl = vec![0.0_f64; dim_t];
                let mut wr = vec![0.0_f64; dim_t];
                wl[..prev.wl.len()].copy_from_slice(prev.wl);
                wr[..prev.wr.len()].copy_from_slice(prev.wr);
                return TierInheritanceDecision {
                    inherited: true,
                    wl,
                    wr,
                    mse: prev.mse,
                };
            }
        }
    }
    TierInheritanceDecision {
        inherited: false,
        wl: cur.wl,
        wr: cur.wr,
        mse: cur.mse,
    }
}

#[inline]
fn snap_fp16_range(v: f64, lo: f64, hi: f64) -> f64 {
    let clamped = if v.is_finite() {
        (v as f32).clamp(lo as f32, hi as f32)
    } else {
        0.0
    };
    let s = f64::from(fp16_to_f32(f32_to_fp16(clamped)));
    if s == 0.0 { 0.0 } else { s }
}

/// Synchronizes `fp16` precision across `PrefixOverride` boundaries between `ti - 1` (encoded as
/// `f32` in its own delta arena) and `ti` (encoded as `fp16` when `emit_prefix_overrides == true`),
/// covering both dual-band overlap intervals so inherited upper tiers match `prev` identically.
pub(super) fn sync_inherited_prefix_fp16(
    ti: usize,
    emit_prefix_overrides: bool,
    prev_pfx: Option<&mut TierReadoutPrefixF64>,
    cur_pfx: &mut TierReadoutPrefixF64,
) {
    if !emit_prefix_overrides {
        return;
    }
    let (fir_range, bg_end, c_range) = match ti {
        1 => (16..32, 8, 4..8),
        2 => (16..48, 12, 4..12),
        3 => (32..96, 24, 8..24),
        _ => return,
    };
    for p in cur_pfx.fir_l[fir_range.clone()]
        .iter_mut()
        .chain(&mut cur_pfx.fir_r[fir_range.clone()])
    {
        *p = snap_fp16_range(*p, -16.0, 16.0);
    }
    for bg in &mut cur_pfx.kautz_b_gamma[..bg_end] {
        *bg = snap_fp16_range(*bg, 0.0, 8.0);
    }
    for m in c_range.clone() {
        cur_pfx.kautz_c_re_l[m] = snap_fp16_range(cur_pfx.kautz_c_re_l[m], -8.0, 8.0);
        cur_pfx.kautz_c_im_l[m] = snap_fp16_range(cur_pfx.kautz_c_im_l[m], -8.0, 8.0);
        cur_pfx.kautz_c_re_r[m] = snap_fp16_range(cur_pfx.kautz_c_re_r[m], -8.0, 8.0);
        cur_pfx.kautz_c_im_r[m] = snap_fp16_range(cur_pfx.kautz_c_im_r[m], -8.0, 8.0);
    }
    if let Some(prev) = prev_pfx {
        prev.fir_l[fir_range.clone()].copy_from_slice(&cur_pfx.fir_l[fir_range.clone()]);
        prev.fir_r[fir_range.clone()].copy_from_slice(&cur_pfx.fir_r[fir_range]);
        prev.kautz_b_gamma[..bg_end].copy_from_slice(&cur_pfx.kautz_b_gamma[..bg_end]);
        prev.kautz_c_re_l[c_range.clone()].copy_from_slice(&cur_pfx.kautz_c_re_l[c_range.clone()]);
        prev.kautz_c_im_l[c_range.clone()].copy_from_slice(&cur_pfx.kautz_c_im_l[c_range.clone()]);
        prev.kautz_c_re_r[c_range.clone()].copy_from_slice(&cur_pfx.kautz_c_re_r[c_range.clone()]);
        prev.kautz_c_im_r[c_range.clone()].copy_from_slice(&cur_pfx.kautz_c_im_r[c_range]);
    }
}

pub(super) type PrevTierSol = (Vec<f64>, Vec<f64>, f64);

#[derive(Debug)]
pub(super) struct TierCommitState<'a> {
    pub prev_sol: &'a mut Option<PrevTierSol>,
    pub tier_mse: &'a mut f64,
    pub tier_hf_mse: &'a mut [f64; 2],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TierCommitCtx<'a> {
    pub ti: usize,
    pub dim_t: usize,
    pub active_fir_taps: usize,
    pub pc_gain: [f64; 2],
    pub emit_prefix_overrides: bool,
    pub same_basis: bool,
    pub fl_ref: &'a [f32],
    pub tl_ref: &'a [f32],
    pub fr_ref: &'a [f32],
    pub tr_ref: &'a [f32],
}

pub(super) fn commit_tier_with_inheritance(
    model: &mut SonaModelF64,
    ctx: TierCommitCtx<'_>,
    wl_refined: Vec<f64>,
    wr_refined: Vec<f64>,
    state: TierCommitState<'_>,
) {
    let TierCommitState {
        prev_sol,
        tier_mse,
        tier_hf_mse,
    } = state;
    let cur_mse = 0.5
        * (compute_channel_mse(&wl_refined, ctx.fl_ref, ctx.tl_ref, ctx.dim_t)
            + compute_channel_mse(&wr_refined, ctx.fr_ref, ctx.tr_ref, ctx.dim_t));
    let tgt_var = 0.5
        * (ctx
            .tl_ref
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            + ctx
                .tr_ref
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>())
        / (ctx.tl_ref.len().max(1) as f64);

    let (wl_final, wr_final, tier_mse_val, inherited) =
        if let Some((ref pwl, ref pwr, pmse)) = *prev_sol {
            let dec = maybe_inherit_prev_tier_weights(
                ctx.ti,
                ctx.dim_t,
                tgt_var,
                TierGateCurrent {
                    wl: wl_refined,
                    wr: wr_refined,
                    mse: cur_mse,
                },
                TierGatePrev {
                    wl: pwl,
                    wr: pwr,
                    mse: pmse,
                    same_basis: ctx.same_basis,
                },
            );
            (dec.wl, dec.wr, dec.mse, dec.inherited)
        } else {
            (wl_refined, wr_refined, cur_mse, false)
        };

    *tier_mse = tier_mse_val;
    if ctx.ti <= 1 {
        tier_hf_mse[ctx.ti] = 0.5
            * (compute_channel_hf_mse(&wl_final, ctx.fl_ref, ctx.tl_ref, ctx.dim_t)
                + compute_channel_hf_mse(&wr_final, ctx.fr_ref, ctx.tr_ref, ctx.dim_t));
    }

    let mut pfx = unpack_weights_into_prefix(
        &wl_final,
        &wr_final,
        ctx.active_fir_taps,
        ctx.pc_gain,
        ctx.ti,
        model,
    );
    if inherited && ctx.ti >= 1 {
        sync_inherited_prefix_fp16(
            ctx.ti,
            ctx.emit_prefix_overrides,
            model.tier_overrides[ctx.ti - 1].as_deref_mut(),
            &mut pfx,
        );
    }
    if ctx.ti == 3 {
        model.direct_s2_u_gain = pfx.direct_s2_u_gain;
        model.direct_s2_hp_gain = pfx.direct_s2_hp_gain;
        model.pre_conv_u_gain = pfx.pre_conv_u_gain;
        model.pre_conv_hp_gain = pfx.pre_conv_hp_gain;
        for l in 0..40 {
            model.w_ro[l][6..12].copy_from_slice(&pfx.w_ro[l][6..12]);
        }
        model.fir_l = pfx.fir_l;
        model.fir_r = pfx.fir_r;
    }
    model.tier_overrides[ctx.ti] = Some(Box::new(pfx));
    *prev_sol = Some((wl_final, wr_final, tier_mse_val));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_gate_inheritance_conditions_and_fp16_sync() {
        let prev_w = vec![1.0_f64; 100];
        let cur_w = vec![0.5_f64; 156];

        // Regime [1e-5, 1e-4): requires min_rel_drop = clamp(0.001 * 56, 0.02, 0.08) = 5.6%.
        let dec = maybe_inherit_prev_tier_weights(
            3,
            156,
            1.0,
            TierGateCurrent {
                wl: cur_w.clone(),
                wr: cur_w.clone(),
                mse: 4.875e-5, // 2.5% drop < 5.6% min_rel_drop
            },
            TierGatePrev {
                wl: &prev_w,
                wr: &prev_w,
                mse: 5.0e-5,
                same_basis: true,
            },
        );
        assert!(dec.inherited);
        assert_eq!(dec.wl.len(), 156);
        assert_eq!(dec.wl[..100], prev_w[..]);
        assert!(dec.wl[100..].iter().all(|&v| v == 0.0));
        assert_eq!(dec.mse, 5.0e-5);

        // Regime < 1e-5 (deep floor < -50 dB): 29% drop inherits, 55% drop adopts new basis.
        let dec_deep_inherit = maybe_inherit_prev_tier_weights(
            2,
            156,
            1.0,
            TierGateCurrent {
                wl: cur_w.clone(),
                wr: cur_w.clone(),
                mse: 7.1e-7, // 29% drop < 50%
            },
            TierGatePrev {
                wl: &prev_w,
                wr: &prev_w,
                mse: 1.0e-6,
                same_basis: true,
            },
        );
        assert!(dec_deep_inherit.inherited);
        assert_eq!(dec_deep_inherit.mse, 1.0e-6);

        let dec_deep_keep = maybe_inherit_prev_tier_weights(
            2,
            156,
            1.0,
            TierGateCurrent {
                wl: cur_w.clone(),
                wr: cur_w.clone(),
                mse: 4.5e-7, // 55% drop >= 50%
            },
            TierGatePrev {
                wl: &prev_w,
                wr: &prev_w,
                mse: 1.0e-6,
                same_basis: true,
            },
        );
        assert!(!dec_deep_keep.inherited);
        assert_eq!(dec_deep_keep.mse, 4.5e-7);

        // Does not trigger when prev_nesr >= 1e-4, improvement >= min_rel_drop in [1e-5, 1e-4), or !same_basis.
        for (cur_mse, prev_mse, same_basis) in [
            (9.75e-4, 1.0e-3, true),
            (4.2e-5, 5.0e-5, true), // 16% drop in [1e-5, 1e-4)
            (4.875e-5, 5.0e-5, false),
        ] {
            let dec_no = maybe_inherit_prev_tier_weights(
                3,
                156,
                1.0,
                TierGateCurrent {
                    wl: cur_w.clone(),
                    wr: cur_w.clone(),
                    mse: cur_mse,
                },
                TierGatePrev {
                    wl: &prev_w,
                    wr: &prev_w,
                    mse: prev_mse,
                    same_basis,
                },
            );
            assert!(!dec_no.inherited);
            assert_eq!(dec_no.mse, cur_mse);
        }

        // Verify fp16 synchronization aligns dual-band boundaries for ti==2 (16..48, 4..12) and ti==3 (32..96, 8..24).
        let mut full = TierReadoutPrefixF64::default();
        full.fir_l[20] = 0.234_567_89;
        full.fir_r[38] = -0.345_678_9;
        full.kautz_c_re_l[6] = 0.444_444_4;
        let mut pro = full.clone();
        sync_inherited_prefix_fp16(2, true, Some(&mut full), &mut pro);
        assert_eq!(full.fir_l[20], pro.fir_l[20]);
        assert_eq!(full.fir_r[38], pro.fir_r[38]);
        assert_eq!(full.kautz_c_re_l[6], pro.kautz_c_re_l[6]);

        pro.fir_l[50] = 0.123_456_789;
        pro.fir_r[75] = -0.987_654_321;
        pro.kautz_c_re_l[14] = 0.333_333_3;
        let mut ultra = pro.clone();
        sync_inherited_prefix_fp16(3, true, Some(&mut pro), &mut ultra);
        assert_eq!(pro.fir_l[50], ultra.fir_l[50]);
        assert_eq!(pro.fir_r[75], ultra.fir_r[75]);
        assert_eq!(pro.kautz_c_re_l[14], ultra.kautz_c_re_l[14]);
        assert!((pro.fir_l[50] - 0.123_456_789).abs() < 1.0e-3);
    }
}
