//! Gram-domain `fp16` Gauss-Seidel coordinate rounding for Pass-3 nested `VarPro` readout
//! weights, compensating `binary16` quantization error across correlated features and `binary32`
//! FIR columns (`ref[train.pipeline.nested-varpro-schur]`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};

use crate::wide_gram::uplo_u_index;

/// Canonical wire-exact `fp16` snap inside `[lo, hi]` (`+0.0` canonicalized).
#[inline]
#[must_use]
pub(super) fn snap_fp16_val(v: f64, lo: f32, hi: f32) -> f64 {
    let clamped = if v.is_finite() {
        (v as f32).clamp(lo, hi)
    } else {
        0.0
    };
    let snapped = fp16_to_f32(f32_to_fp16(if clamped == 0.0 { 0.0 } else { clamped }));
    f64::from(if snapped == 0.0 { 0.0 } else { snapped })
}

/// Canonical `binary32` snap inside `[lo, hi]` (`+0.0` canonicalized).
#[inline]
#[must_use]
fn snap_f32_val(v: f64, lo: f32, hi: f32) -> f64 {
    let clamped = if v.is_finite() {
        (v as f32).clamp(lo, hi)
    } else {
        0.0
    };
    f64::from(if clamped == 0.0 { 0.0 } else { clamped })
}

/// Builds per-column wire precision metadata (`Some((lo, hi))` for `fp16`-packed columns, `None`
/// for `binary32` FIR columns) matching [`super::features::unpack_weights_into_prefix`] and Pass 5
/// `.dnam-bin` tier readout packing for `tier_idx in 0..4`.
#[must_use]
pub(super) fn build_tier_fp16_col_bounds(
    tier_idx: usize,
    active_taps: usize,
    emit_prefix_overrides: bool,
) -> Vec<Option<(f32, f32)>> {
    let capped_taps = active_taps.min(128);
    let lite_fir = capped_taps.min(32);
    let full_fir = capped_taps.saturating_sub(32).min(16);
    let pro_fir = capped_taps.saturating_sub(48).min(48);
    let ultra_fir = capped_taps.saturating_sub(96).min(32);
    let tier_defs = [
        (0..16_usize, 0..lite_fir),
        (16..24_usize, 32..(32 + full_fir)),
        (24..32_usize, 48..(48 + pro_fir)),
        (32..40_usize, 96..(96 + ultra_fir)),
    ];

    let mut bounds = Vec::with_capacity(2 + 40 * 3 + capped_taps);
    bounds.push(Some((-8.0, 8.0))); // col 0: direct_s2_u_gain
    bounds.push(Some((-8.0, 8.0))); // col 1: direct_s2_hp_gain

    for (t, (lane_range, fir_range)) in tier_defs.into_iter().enumerate() {
        if t > tier_idx {
            break;
        }
        for _ in lane_range {
            for _ in 0..3 {
                bounds.push(Some((-8.0, 8.0)));
            }
        }
        for k in fir_range {
            let is_fp16_override_tap = emit_prefix_overrides
                && matches!((tier_idx, k), (1, 16..32) | (2, 32..48) | (3, 48..96));
            bounds.push(is_fp16_override_tap.then_some((-8.0, 8.0)));
        }
    }
    bounds
}

/// Snaps every column of `w` directly to its `fp16` (`Some`) or `f32` (`None`) wire domain.
#[must_use]
pub(super) fn snap_weights_direct(w_cont: &[f64], fp16_bounds: &[Option<(f32, f32)>]) -> Vec<f64> {
    w_cont
        .iter()
        .enumerate()
        .map(|(j, &v)| match fp16_bounds.get(j).copied().flatten() {
            Some((lo, hi)) => snap_fp16_val(v, lo, hi),
            None => snap_f32_val(v, -8.0, 8.0),
        })
        .collect()
}

fn eval_quad_diff(w: &[f64], w_cont: &[f64], gram_uplo_u: &[f64], ridge: f64) -> f64 {
    let dim = w.len();
    let mut err = 0.0_f64;
    for j in 0..dim {
        let ej = w[j] - w_cont[j];
        let mut h_ej = ridge * ej + gram_uplo_u[uplo_u_index(j, j)] * ej;
        for i in 0..dim {
            if i != j {
                h_ej += gram_uplo_u[uplo_u_index(i, j)] * (w[i] - w_cont[i]);
            }
        }
        err += ej * h_ej;
    }
    err
}

/// Minimizes `0.5 * (w - w_cont)^T (G + ridge * I) (w - w_cont)` via 2 cyclic Gauss-Seidel
/// sweeps over quantized `fp16` columns followed by continuous `f32` columns.
#[must_use]
pub(super) fn snap_weights_gram_coordinate(
    w_cont: &[f64],
    gram_uplo_u: &[f64],
    ridge: f64,
    fp16_bounds: &[Option<(f32, f32)>],
) -> Vec<f64> {
    let dim = w_cont.len();
    if dim == 0 || fp16_bounds.len() != dim {
        return snap_weights_direct(w_cont, fp16_bounds);
    }
    let expected_packed = dim * (dim + 1) / 2;
    if gram_uplo_u.len() < expected_packed || !w_cont.iter().all(|v| v.is_finite()) {
        return snap_weights_direct(w_cont, fp16_bounds);
    }

    let r_eff = if ridge.is_finite() && ridge >= 0.0 {
        ridge
    } else {
        0.0
    };
    let mut w = w_cont.to_vec();
    // Maintain residual r = (G + ridge*I)(w - w_cont), initially exact zero at w = w_cont.
    let mut r = vec![0.0_f64; dim];

    let apply_col_update = |j: usize, w_new: f64, w_buf: &mut [f64], r_buf: &mut [f64]| {
        let delta = w_new - w_buf[j];
        if delta == 0.0 || !delta.is_finite() {
            return;
        }
        w_buf[j] = w_new;
        for (i, ri) in r_buf.iter_mut().enumerate() {
            let g_ij = gram_uplo_u[uplo_u_index(i, j)];
            let h_ij = if i == j { g_ij + r_eff } else { g_ij };
            *ri += h_ij * delta;
        }
    };

    for _sweep in 0..2 {
        for j in 0..dim {
            if let Some((lo, hi)) = fp16_bounds[j] {
                let h_jj = gram_uplo_u[uplo_u_index(j, j)] + r_eff;
                let w_target = if h_jj > 1.0e-15 && r[j].is_finite() {
                    w[j] - r[j] / h_jj
                } else {
                    w[j]
                };
                let w_new = snap_fp16_val(w_target, lo, hi);
                apply_col_update(j, w_new, &mut w, &mut r);
            }
        }
        for j in 0..dim {
            if fp16_bounds[j].is_none() {
                let h_jj = gram_uplo_u[uplo_u_index(j, j)] + r_eff;
                let w_target = if h_jj > 1.0e-15 && r[j].is_finite() {
                    w[j] - r[j] / h_jj
                } else {
                    w[j]
                };
                let w_new = snap_f32_val(w_target, -8.0, 8.0);
                apply_col_update(j, w_new, &mut w, &mut r);
            }
        }
    }

    let w_direct = snap_weights_direct(w_cont, fp16_bounds);
    let err_gs = eval_quad_diff(&w, w_cont, gram_uplo_u, r_eff);
    let err_dir = eval_quad_diff(&w_direct, w_cont, gram_uplo_u, r_eff);
    if err_gs <= err_dir { w } else { w_direct }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_weights_gram_coordinate_identity_and_correlated_improvement() {
        // Exact fp16 + f32 inputs remain unchanged.
        let bounds_mixed = vec![Some((-8.0, 8.0)), Some((-8.0, 8.0)), None];
        let exact = vec![0.5_f64, -1.25_f64, 0.375_f64];
        let gram_diag = vec![4.0, 0.0, 5.0, 0.0, 0.0, 3.0];
        let out_exact = snap_weights_gram_coordinate(&exact, &gram_diag, 1.0e-5, &bounds_mixed);
        assert_eq!(out_exact, exact);

        // High weight correlation between col 0 and col 2 (x = [1, 1.01]): col 2 (f32) compensates col 0 (fp16)!
        let mut gram_corr = [0.0_f64; 6];
        for k in 0..32 {
            let a = (k as f64) * 0.1 - 1.5;
            let row = [
                a,
                0.01 * a + ((k % 3) as f64) * 0.1,
                a + 0.01 * ((k % 5) as f64),
            ];
            for j in 0..3 {
                for i in 0..=j {
                    gram_corr[uplo_u_index(i, j)] += row[i] * row[j];
                }
            }
        }
        let w_cont = vec![1.00048828125_f64, 0.500244140625_f64, -0.3333333333_f64];
        let w_gs = snap_weights_gram_coordinate(&w_cont, &gram_corr, 1.0e-5, &bounds_mixed);
        let w_dir = snap_weights_direct(&w_cont, &bounds_mixed);
        let err_gs = eval_quad_diff(&w_gs, &w_cont, &gram_corr, 1.0e-5);
        let err_dir = eval_quad_diff(&w_dir, &w_cont, &gram_corr, 1.0e-5);
        assert!(
            err_gs < err_dir * 0.25,
            "expected Gram coordinate snap to strongly beat direct rounding: gs={err_gs} dir={err_dir}"
        );
        for j in 0..2 {
            assert_eq!(w_gs[j], snap_fp16_val(w_gs[j], -8.0, 8.0));
        }

        // Also test all-fp16 correlated cancellation where col 1 shifts by +/-1 ULP to halve error.
        let bounds_fp16 = vec![Some((-8.0, 8.0)), Some((-8.0, 8.0))];
        let mut g2 = vec![0.0_f64; 3];
        for k in 0..64 {
            let u = (k as f64) * 0.25 - 8.0;
            let row = [u, u * 0.995 + ((k & 1) as f64) * 0.05];
            for j in 0..2 {
                for i in 0..=j {
                    g2[uplo_u_index(i, j)] += row[i] * row[j];
                }
            }
        }
        let wc2 = vec![2.000_976_562_5_f64, -2.000_976_562_5_f64];
        let wg2 = snap_weights_gram_coordinate(&wc2, &g2, 1.0e-5, &bounds_fp16);
        let wd2 = snap_weights_direct(&wc2, &bounds_fp16);
        assert!(eval_quad_diff(&wg2, &wc2, &g2, 1.0e-5) <= eval_quad_diff(&wd2, &wc2, &g2, 1.0e-5));
        assert_eq!(wg2[0], snap_fp16_val(wg2[0], -8.0, 8.0));
        assert_eq!(wg2[1], snap_fp16_val(wg2[1], -8.0, 8.0));
    }

    #[test]
    fn test_build_tier_fp16_col_bounds_all_tiers_and_prefix_overrides() {
        let taps = 128_usize;
        // Tier 0 (Lite): 2 + 16*3 + 32 FIR (all FIR f32 None).
        let b0 = build_tier_fp16_col_bounds(0, taps, true);
        assert_eq!(b0.len(), 2 + 48 + 32);
        assert!(b0[..50].iter().all(Option::is_some));
        assert!(b0[50..].iter().all(Option::is_none));

        // Tier 1 (Full): Lite FIR 0..32 -> 16..32 become fp16 iff emit_prefix_overrides; delta FIR 32..48 stay f32.
        let b1_no = build_tier_fp16_col_bounds(1, taps, false);
        assert_eq!(b1_no.len(), 2 + 48 + 32 + 24 + 16);
        assert!(b1_no[50..82].iter().all(Option::is_none));
        let b1_ov = build_tier_fp16_col_bounds(1, taps, true);
        assert!(b1_ov[50..66].iter().all(Option::is_none)); // FIR 0..16 in PrefixOverrideFullEncodeWire f32
        assert!(b1_ov[66..82].iter().all(Option::is_some)); // FIR 16..32 in w_ro_and_mods_fp16
        assert!(b1_ov[82..106].iter().all(Option::is_some)); // lanes 16..24 w_ro fp16
        assert!(b1_ov[106..122].iter().all(Option::is_none)); // FIR 32..48 in conv_fir_full f32

        // Tier 2 (Pro): prefix FIR 0..32 f32, FIR 32..48 fp16, delta FIR 48..96 f32.
        let b2_ov = build_tier_fp16_col_bounds(2, taps, true);
        assert_eq!(b2_ov.len(), 122 + 24 + 48);
        assert!(b2_ov[50..82].iter().all(Option::is_none)); // FIR 0..32 f32
        assert!(b2_ov[106..122].iter().all(Option::is_some)); // FIR 32..48 fp16
        assert!(b2_ov[122..146].iter().all(Option::is_some)); // lanes 24..32 w_ro fp16
        assert!(b2_ov[146..194].iter().all(Option::is_none)); // FIR 48..96 f32

        // Tier 3 (Ultra): prefix FIR 0..48 f32, FIR 48..96 fp16, delta FIR 96..128 f32.
        let b3_ov = build_tier_fp16_col_bounds(3, taps, true);
        assert_eq!(b3_ov.len(), 194 + 24 + 32);
        assert!(b3_ov[50..82].iter().all(Option::is_none)); // FIR 0..32 f32
        assert!(b3_ov[106..122].iter().all(Option::is_none)); // FIR 32..48 f32
        assert!(b3_ov[146..194].iter().all(Option::is_some)); // FIR 48..96 fp16
        assert!(b3_ov[194..218].iter().all(Option::is_some)); // lanes 32..40 w_ro fp16
        assert!(b3_ov[218..250].iter().all(Option::is_none)); // FIR 96..128 f32
    }
}
