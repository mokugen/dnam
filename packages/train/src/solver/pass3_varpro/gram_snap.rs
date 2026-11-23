//! Gram-domain `fp16` Gauss-Seidel coordinate rounding for Pass-3 nested `VarPro` readout
//! weights, compensating `binary16` quantization error across correlated features and `binary32`
//! FIR columns (`ref[train.pipeline.nested-varpro-schur]`).

use dnam_kernels::{f32_to_fp16, fp16_to_f32};

use crate::wide_gram::uplo_u_index;

use super::features::TIER_KAUTZ_CAPS;

/// Canonical wire precision & physical clamp specification for one linear column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ColWireSpec {
    Fp16 { lo: f32, hi: f32 },
    F32 { lo: f32, hi: f32 },
}

impl ColWireSpec {
    #[inline]
    #[must_use]
    pub(super) fn is_fp16(self) -> bool {
        matches!(self, Self::Fp16 { .. })
    }

    #[inline]
    #[must_use]
    pub(super) fn snap(self, v: f64) -> f64 {
        match self {
            Self::Fp16 { lo, hi } => snap_fp16_val(v, lo, hi),
            Self::F32 { lo, hi } => snap_f32_val(v, lo, hi),
        }
    }
}

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

/// Builds per-column wire precision metadata ([`ColWireSpec::Fp16`] vs [`ColWireSpec::F32`])
/// matching [`super::features::unpack_weights_into_prefix`] and Pass 5 `.dnam-bin` tier readout
/// packing for `tier_idx in 0..4`.
#[must_use]
pub(super) fn build_tier_col_wire_specs(
    tier_idx: usize,
    active_taps: usize,
    emit_prefix_overrides: bool,
) -> Vec<ColWireSpec> {
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

    let fp16_8 = ColWireSpec::Fp16 { lo: -8.0, hi: 8.0 };
    let f32_8 = ColWireSpec::F32 { lo: -8.0, hi: 8.0 };
    let mut specs = Vec::with_capacity(2 + 40 * 3 + capped_taps);
    specs.push(fp16_8); // col 0: direct_s2_u_gain
    specs.push(fp16_8); // col 1: direct_s2_hp_gain

    for (t, (lane_range, fir_range)) in tier_defs.into_iter().enumerate() {
        if t > tier_idx {
            break;
        }
        for _ in lane_range {
            for _ in 0..3 {
                specs.push(fp16_8);
            }
        }
        for k in fir_range {
            let is_fp16_override_tap = emit_prefix_overrides
                && matches!((tier_idx, k), (1, 16..32) | (2, 32..48) | (3, 48..96));
            specs.push(if is_fp16_override_tap { fp16_8 } else { f32_8 });
        }
    }
    specs
}

/// Builds per-channel wire specs for `opt_to_lr` taps at `tier_idx in 0..4`. All tiers
/// (`Lite..Ultra`) hydrate `opt_to_lr` as `binary16` in `[-4.0, 4.0]` (`f32le` on Lite wire,
/// `fp16` on Full..Ultra wire), consumed by M21 VarPro and unit tests.
#[allow(dead_code)]
#[must_use]
pub(super) fn build_opt_to_lr_wire_specs(tier_idx: usize) -> Vec<ColWireSpec> {
    let quads = tier_idx.min(3) + 1;
    vec![ColWireSpec::Fp16 { lo: -4.0, hi: 4.0 }; quads * 4]
}

/// Builds per-channel `[c_re, c_im]` wire specs for Kautz modal residues at `tier_idx in 0..4`,
/// matching `pass5_emit/unpack.rs` intermediate `binary16` prefix-override bands (`[4..8]`,
/// `[8..12]`, `[12..24]`), consumed by M21 VarPro and unit tests.
#[allow(dead_code)]
#[must_use]
pub(super) fn build_prefix_kautz_wire_specs(
    tier_idx: usize,
    active_kautz_pairs: usize,
    emit_prefix_overrides: bool,
) -> Vec<ColWireSpec> {
    let ti = tier_idx.min(3);
    let pairs = active_kautz_pairs.min(TIER_KAUTZ_CAPS[ti]);
    let fp16_8 = ColWireSpec::Fp16 { lo: -8.0, hi: 8.0 };
    let f32_8 = ColWireSpec::F32 { lo: -8.0, hi: 8.0 };
    let mut specs = Vec::with_capacity(pairs * 2);
    for k in 0..pairs {
        let is_fp16_override =
            emit_prefix_overrides && matches!((ti, k), (1, 4..8) | (2, 8..12) | (3, 12..24));
        let spec = if is_fp16_override { fp16_8 } else { f32_8 };
        specs.push(spec);
        specs.push(spec);
    }
    specs
}

/// Snaps every column of `w` directly to its wire domain ([`ColWireSpec`]).
#[must_use]
pub(super) fn snap_weights_direct(w_cont: &[f64], col_specs: &[ColWireSpec]) -> Vec<f64> {
    w_cont
        .iter()
        .enumerate()
        .map(|(j, &v)| {
            col_specs
                .get(j)
                .copied()
                .unwrap_or(ColWireSpec::F32 { lo: -8.0, hi: 8.0 })
                .snap(v)
        })
        .collect()
}

fn eval_quad_diff(w: &[f64], w_cont: &[f64], gram_uplo_u: &[f64], lambda_diag: &[f64]) -> f64 {
    let dim = w.len();
    let mut err = 0.0_f64;
    for j in 0..dim {
        let ej = w[j] - w_cont[j];
        let mut h_ej = lambda_diag[j] * ej + gram_uplo_u[uplo_u_index(j, j)] * ej;
        for i in 0..dim {
            if i != j {
                h_ej += gram_uplo_u[uplo_u_index(i, j)] * (w[i] - w_cont[i]);
            }
        }
        err += ej * h_ej;
    }
    err
}

/// Minimizes `0.5 * (w - w_cont)^T (G + diag(lambda_diag)) (w - w_cont)` via 2 cyclic Gauss-Seidel
/// sweeps over quantized `fp16` columns followed by continuous `f32` columns.
#[must_use]
pub(super) fn snap_weights_gram_coordinate(
    w_cont: &[f64],
    gram_uplo_u: &[f64],
    lambda_diag: &[f64],
    col_specs: &[ColWireSpec],
) -> Vec<f64> {
    let dim = w_cont.len();
    if dim == 0 || col_specs.len() != dim || lambda_diag.len() != dim {
        return snap_weights_direct(w_cont, col_specs);
    }
    let expected_packed = dim * (dim + 1) / 2;
    if gram_uplo_u.len() < expected_packed || !w_cont.iter().all(|v| v.is_finite()) {
        return snap_weights_direct(w_cont, col_specs);
    }

    let lam: Vec<f64> = lambda_diag
        .iter()
        .map(|&v| if v.is_finite() && v >= 0.0 { v } else { 0.0 })
        .collect();
    let mut w = w_cont.to_vec();
    // Maintain residual r = (G + diag(lam))(w - w_cont), initially exact zero at w = w_cont.
    let mut r = vec![0.0_f64; dim];

    let apply_col_update = |j: usize, w_new: f64, w_buf: &mut [f64], r_buf: &mut [f64]| {
        let delta = w_new - w_buf[j];
        if delta == 0.0 || !delta.is_finite() {
            return;
        }
        w_buf[j] = w_new;
        for (i, ri) in r_buf.iter_mut().enumerate() {
            let g_ij = gram_uplo_u[uplo_u_index(i, j)];
            let h_ij = if i == j { g_ij + lam[j] } else { g_ij };
            *ri += h_ij * delta;
        }
    };

    for _sweep in 0..2 {
        for want_fp16 in [true, false] {
            for (j, spec) in col_specs.iter().enumerate() {
                if spec.is_fp16() != want_fp16 {
                    continue;
                }
                let h_jj = gram_uplo_u[uplo_u_index(j, j)] + lam[j];
                let w_target = if h_jj > 1.0e-15 && r[j].is_finite() {
                    w[j] - r[j] / h_jj
                } else {
                    w[j]
                };
                let w_new = spec.snap(w_target);
                apply_col_update(j, w_new, &mut w, &mut r);
            }
        }
    }

    let w_direct = snap_weights_direct(w_cont, col_specs);
    let err_gs = eval_quad_diff(&w, w_cont, gram_uplo_u, &lam);
    let err_dir = eval_quad_diff(&w_direct, w_cont, gram_uplo_u, &lam);
    if err_gs <= err_dir { w } else { w_direct }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_weights_gram_coordinate_identity_and_correlated_improvement() {
        // Exact fp16 + f32 inputs remain unchanged.
        let fp16_8 = ColWireSpec::Fp16 { lo: -8.0, hi: 8.0 };
        let f32_8 = ColWireSpec::F32 { lo: -8.0, hi: 8.0 };
        let specs_mixed = vec![fp16_8, fp16_8, f32_8];
        let exact = vec![0.5_f64, -1.25_f64, 0.375_f64];
        let gram_diag = vec![4.0, 0.0, 5.0, 0.0, 0.0, 3.0];
        let lam3 = vec![1.0e-5, 2.0e-4, 1.0e-5];
        let out_exact = snap_weights_gram_coordinate(&exact, &gram_diag, &lam3, &specs_mixed);
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
        let w_gs = snap_weights_gram_coordinate(&w_cont, &gram_corr, &lam3, &specs_mixed);
        let w_dir = snap_weights_direct(&w_cont, &specs_mixed);
        let err_gs = eval_quad_diff(&w_gs, &w_cont, &gram_corr, &lam3);
        let err_dir = eval_quad_diff(&w_dir, &w_cont, &gram_corr, &lam3);
        assert!(
            err_gs < err_dir * 0.25,
            "expected Gram coordinate snap to strongly beat direct rounding: gs={err_gs} dir={err_dir}"
        );
        for &val in &w_gs[..2] {
            assert_eq!(val, snap_fp16_val(val, -8.0, 8.0));
        }

        // Verify non-uniform diagonal lambda_diag preserves strongly regularized column while
        // letting weakly regularized column compensate (0 vs high diag lam).
        let specs_2 = vec![fp16_8, fp16_8];
        let g_diag_tied = vec![1.0, 1.0, 1.0];
        let wc_lam = vec![1.000_976_562_5_f64, -1.000_976_562_5_f64];
        let lam_asym = vec![100.0_f64, 1.0e-6_f64];
        let wg_lam = snap_weights_gram_coordinate(&wc_lam, &g_diag_tied, &lam_asym, &specs_2);
        assert_eq!(wg_lam[0], fp16_8.snap(wc_lam[0]));
        assert!(
            eval_quad_diff(&wg_lam, &wc_lam, &g_diag_tied, &lam_asym)
                <= eval_quad_diff(
                    &snap_weights_direct(&wc_lam, &specs_2),
                    &wc_lam,
                    &g_diag_tied,
                    &lam_asym
                )
        );

        // Mismatched dimension guards gracefully fall back to direct snap.
        assert_eq!(
            snap_weights_gram_coordinate(&wc_lam, &g_diag_tied, &[1.0e-5], &specs_2),
            snap_weights_direct(&wc_lam, &specs_2)
        );
        assert_eq!(
            snap_weights_gram_coordinate(&wc_lam, &[1.0], &lam_asym, &specs_2),
            snap_weights_direct(&wc_lam, &specs_2)
        );

        // Also test all-fp16 correlated cancellation where col 1 shifts by +/-1 ULP to halve error.
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
        let lam2 = vec![1.0e-5, 1.0e-5];
        let wg2 = snap_weights_gram_coordinate(&wc2, &g2, &lam2, &specs_2);
        let wd2 = snap_weights_direct(&wc2, &specs_2);
        assert!(eval_quad_diff(&wg2, &wc2, &g2, &lam2) <= eval_quad_diff(&wd2, &wc2, &g2, &lam2));
        assert_eq!(wg2[0], snap_fp16_val(wg2[0], -8.0, 8.0));
        assert_eq!(wg2[1], snap_fp16_val(wg2[1], -8.0, 8.0));
    }

    #[test]
    fn test_build_tier_col_wire_specs_opt_to_lr_and_kautz_overrides() {
        let taps = 128_usize;
        // Tier 0 (Lite): 2 + 16*3 + 32 FIR (all FIR f32).
        let b0 = build_tier_col_wire_specs(0, taps, true);
        assert_eq!(b0.len(), 2 + 48 + 32);
        assert!(b0[..50].iter().all(|s| s.is_fp16()));
        assert!(b0[50..].iter().all(|s| !s.is_fp16()));

        // Tier 1 (Full): Lite FIR 0..32 -> 16..32 become fp16 iff emit_prefix_overrides; delta FIR 32..48 stay f32.
        let b1_no = build_tier_col_wire_specs(1, taps, false);
        assert_eq!(b1_no.len(), 2 + 48 + 32 + 24 + 16);
        assert!(b1_no[50..82].iter().all(|s| !s.is_fp16()));
        let b1_ov = build_tier_col_wire_specs(1, taps, true);
        assert!(b1_ov[50..66].iter().all(|s| !s.is_fp16())); // FIR 0..16 f32
        assert!(b1_ov[66..82].iter().all(|s| s.is_fp16())); // FIR 16..32 fp16
        assert!(b1_ov[82..106].iter().all(|s| s.is_fp16())); // lanes 16..24 w_ro fp16
        assert!(b1_ov[106..122].iter().all(|s| !s.is_fp16())); // FIR 32..48 f32

        // Tier 2 (Pro): prefix FIR 0..32 f32, FIR 32..48 fp16, delta FIR 48..96 f32.
        let b2_ov = build_tier_col_wire_specs(2, taps, true);
        assert_eq!(b2_ov.len(), 122 + 24 + 48);
        assert!(b2_ov[50..82].iter().all(|s| !s.is_fp16()));
        assert!(b2_ov[106..122].iter().all(|s| s.is_fp16()));
        assert!(b2_ov[122..146].iter().all(|s| s.is_fp16()));
        assert!(b2_ov[146..194].iter().all(|s| !s.is_fp16()));

        // Tier 3 (Ultra): prefix FIR 0..48 f32, FIR 48..96 fp16, delta FIR 96..128 f32.
        let b3_ov = build_tier_col_wire_specs(3, taps, true);
        assert_eq!(b3_ov.len(), 194 + 24 + 32);
        assert!(b3_ov[50..82].iter().all(|s| !s.is_fp16()));
        assert!(b3_ov[106..122].iter().all(|s| !s.is_fp16()));
        assert!(b3_ov[146..194].iter().all(|s| s.is_fp16()));
        assert!(b3_ov[194..218].iter().all(|s| s.is_fp16()));
        assert!(b3_ov[218..250].iter().all(|s| !s.is_fp16()));

        // opt_to_lr: hydrated Fp16 [-4, 4] across ALL tiers including Lite!
        for ti in 0..4 {
            let opt_specs = build_opt_to_lr_wire_specs(ti);
            assert_eq!(opt_specs.len(), (ti + 1) * 4);
            assert!(
                opt_specs
                    .iter()
                    .all(|&s| s == ColWireSpec::Fp16 { lo: -4.0, hi: 4.0 })
            );
        }

        // Prefix Kautz intermediate override bands: Full [4..8], Pro [8..12], Ultra [12..24].
        assert!(
            build_prefix_kautz_wire_specs(0, 8, true)
                .iter()
                .all(|s| !s.is_fp16())
        );
        let k1 = build_prefix_kautz_wire_specs(1, 12, true);
        assert_eq!(k1.len(), 24);
        assert!(k1[..8].iter().all(|s| !s.is_fp16()));
        assert!(k1[8..16].iter().all(|s| s.is_fp16()));
        assert!(k1[16..].iter().all(|s| !s.is_fp16()));

        let k2 = build_prefix_kautz_wire_specs(2, 24, true);
        assert_eq!(k2.len(), 48);
        assert!(k2[..16].iter().all(|s| !s.is_fp16()));
        assert!(k2[16..24].iter().all(|s| s.is_fp16()));
        assert!(k2[24..].iter().all(|s| !s.is_fp16()));

        let k3 = build_prefix_kautz_wire_specs(3, 32, true);
        assert_eq!(k3.len(), 64);
        assert!(k3[..24].iter().all(|s| !s.is_fp16()));
        assert!(k3[24..48].iter().all(|s| s.is_fp16()));
        assert!(k3[48..].iter().all(|s| !s.is_fp16()));
    }
}
