//! `O(log2 N)` `LinearScanElem` prefix scan primitives and exact `0-ULP` `ParaLRU` forward tile
//! execution (`ref[dsp.op-n.stage2.sub-stages]`, `ref[train.pipeline.webgpu-scans-and-ozaki]`).

use crate::simd::{GpuVec4, KernelVec4, MAX_PARALRU_R_F32, scrub_and_flush_f32_bits};

use super::step_paralru_vec4;

/// Scalar `1D` affine scan element representing `v_out = a * v_in + b`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearScanElem {
    pub a: f32,
    pub b: f32,
}

impl LinearScanElem {
    /// Identity linear scan element (`a = 1.0`, `b = 0.0`).
    #[inline(always)]
    #[must_use]
    pub const fn identity() -> Self {
        Self { a: 1.0, b: 0.0 }
    }
}

/// Associatively composes `first` followed by `second`: `(a2 * a1, a2 * b1 + b2)`.
#[inline(always)]
#[must_use]
pub fn compose_linear_scan(first: LinearScanElem, second: LinearScanElem) -> LinearScanElem {
    LinearScanElem {
        a: second.a * first.a,
        b: second.a * first.b + second.b,
    }
}

/// Applies `elem` to initial scalar state `v0`.
#[inline(always)]
#[must_use]
pub fn apply_linear_scan(elem: LinearScanElem, v0: f32) -> f32 {
    elem.a * v0 + elem.b
}

/// Executes an in-place `O(log2 N)` Hillis-Steele doubling inclusive prefix scan over
/// [`LinearScanElem`] (`compose_linear_scan`).
#[inline]
#[must_use]
pub fn scan_linear_prefix_tile<const N: usize>(elems: &[LinearScanElem; N]) -> [LinearScanElem; N] {
    let mut out = *elems;
    let mut step = 1_usize;
    while step < N {
        let mut i = N;
        while i > step {
            i -= 1;
            out[i] = compose_linear_scan(out[i - step], out[i]);
        }
        step <<= 1;
    }
    out
}

/// Steps `valid_len.min(N)` frames of `u[0..N]` through exact decoupled `1D` scalar `ParaLRU`
/// via [`step_paralru_vec4`] lane `0` (`0-ULP` CPU/SIMD bit-identity), zeroing unused tail entries
/// `i >= valid_len` and returning `(v_traj, v_end_state)` (`ref[dsp.op-n.stage2.sub-stages]`).
#[inline]
#[must_use]
pub fn run_paralru_forward_partial_tile<const N: usize>(
    u: &[f32; N],
    valid_len: usize,
    v_init: f32,
    alpha_rel: f32,
    alpha_atk: f32,
    kappa: f32,
) -> ([f32; N], f32) {
    let mut v_out = [0.0_f32; N];
    let v0 = scrub_and_flush_f32_bits(v_init) + 0.0;
    let active = valid_len.min(N);
    if active == 0 {
        return (v_out, v0);
    }

    let a_rel_c = if alpha_rel.is_nan() {
        0.0
    } else {
        alpha_rel.clamp(0.0, MAX_PARALRU_R_F32)
    };
    let a_atk_c = if alpha_atk.is_nan() {
        0.0
    } else {
        alpha_atk.clamp(0.0, 1.0)
    };
    let kap_c = if kappa.is_nan() {
        0.0
    } else {
        kappa.clamp(0.0, 1.0)
    };

    let mut state = GpuVec4::splat(v0);
    let a_rel_v = GpuVec4::splat(a_rel_c);
    let a_atk_v = GpuVec4::splat(a_atk_c);
    let kap_v = GpuVec4::splat(kap_c);

    for (dst, &ui) in v_out[..active].iter_mut().zip(u[..active].iter()) {
        let u_clean = scrub_and_flush_f32_bits(ui) + 0.0;
        let next = step_paralru_vec4(GpuVec4::splat(u_clean), &mut state, a_rel_v, a_atk_v, kap_v);
        *dst = next.to_array()[0];
    }

    (v_out, state.to_array()[0])
}

/// Steps all `N` frames of `u[0..N]` through exact decoupled `1D` scalar `ParaLRU`, returning
/// `(v_traj, v_end_state)` with exact `0-ULP` parity against [`step_paralru_vec4`] and
/// `step_paralru_stereo8` (`ref[dsp.op-n.stage2.sub-stages]`,
/// `ref[train.pipeline.webgpu-scans-and-ozaki]`).
#[inline]
#[must_use]
pub fn run_paralru_forward_tile<const N: usize>(
    u: &[f32; N],
    v_init: f32,
    alpha_rel: f32,
    alpha_atk: f32,
    kappa: f32,
) -> ([f32; N], f32) {
    run_paralru_forward_partial_tile(u, N, v_init, alpha_rel, alpha_atk, kappa)
}

#[cfg(test)]
mod tests {
    use super::super::{ParaLruParams, step_paralru_stereo8};
    use super::*;
    use crate::simd::{InterleavedStereo8, StereoVec8};
    use proptest::prelude::*;

    /// verify[dsp.op-n.stage2.sub-stages]
    #[test]
    fn test_paralru_forward_tile_and_linear_scan_identities() {
        assert_eq!(
            run_paralru_forward_tile::<0>(&[], -1.5, 0.99, 0.5, 1.0),
            ([], -1.5)
        );

        let (partial0, end0) =
            run_paralru_forward_partial_tile(&[2.0_f32; 8], 0, 1.25, 0.9, 0.2, 0.5);
        assert_eq!(partial0, [0.0; 8]);
        assert_eq!(end0, 1.25);

        let rest_zeros = [0.0_f32; 16];
        let (rest, rest_end) = run_paralru_forward_tile(&rest_zeros, 0.0, 0.999, 0.1, 0.9);
        for x in rest {
            assert_eq!(x.to_bits(), 0);
        }
        assert_eq!(rest_end.to_bits(), 0);

        let mut mixed = [0.0_f32; 40];
        mixed[0] = 6.0;
        mixed[5] = f32::NAN;
        mixed[16] = 3.5;
        mixed[30] = -4.0;
        let (part, part_end) = run_paralru_forward_partial_tile(&mixed, 27, 0.5, 0.92, 0.3, 0.7);
        assert_eq!(part_end.to_bits(), part[26].to_bits());
        for &tail in &part[27..] {
            assert_eq!(tail.to_bits(), 0);
        }

        let elem_a = LinearScanElem { a: 0.75, b: 1.25 };
        let elem_b = LinearScanElem { a: 0.5, b: -0.4 };
        let id = LinearScanElem::identity();
        assert_eq!(compose_linear_scan(id, elem_a), elem_a);
        assert_eq!(compose_linear_scan(elem_a, id), elem_a);
        assert!(
            (apply_linear_scan(compose_linear_scan(elem_a, elem_b), 2.0)
                - apply_linear_scan(elem_b, apply_linear_scan(elem_a, 2.0)))
            .abs()
                < 1.0e-6
        );

        let seq_elems = [elem_a, elem_b, LinearScanElem { a: 0.9, b: 0.3 }, elem_a];
        let scanned = scan_linear_prefix_tile(&seq_elems);
        let mut ref_acc = LinearScanElem::identity();
        for (i, &e) in seq_elems.iter().enumerate() {
            ref_acc = compose_linear_scan(ref_acc, e);
            assert!((scanned[i].a - ref_acc.a).abs() < 1.0e-6);
            assert!((scanned[i].b - ref_acc.b).abs() < 1.0e-6);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_linear_scan_prefix_associativity_and_paralru_forward_zero_ulp(
            u_vec in proptest::collection::vec(-15.0_f32..15.0_f32, 64),
            v_init in -5.0_f32..10.0_f32,
            alpha_rel in 0.05_f32..MAX_PARALRU_R_F32,
            alpha_atk in 0.01_f32..0.98_f32,
            kappa in 0.0_f32..1.0_f32,
            valid_len in 1_usize..=64_usize,
        ) {
            let mut u = [0.0_f32; 64];
            u.copy_from_slice(&u_vec);
            let p = ParaLruParams::clamped(alpha_rel, alpha_atk, kappa);

            let (full_fwd, full_end) =
                run_paralru_forward_tile(&u, v_init, p.alpha_rel, p.alpha_atk, p.kappa);
            let (part_fwd, part_end) = run_paralru_forward_partial_tile(
                &u,
                valid_len,
                v_init,
                p.alpha_rel,
                p.alpha_atk,
                p.kappa,
            );

            let mut v_vec = GpuVec4::splat(v_init);
            let mut v_st = InterleavedStereo8::<GpuVec4>::splat(v_init);
            let a_rel_v = GpuVec4::splat(p.alpha_rel);
            let a_atk_v = GpuVec4::splat(p.alpha_atk);
            let kap_v = GpuVec4::splat(p.kappa);

            for (idx, &ui) in u.iter().enumerate() {
                let y_v = step_paralru_vec4(GpuVec4::splat(ui), &mut v_vec, a_rel_v, a_atk_v, kap_v);
                let y_st = step_paralru_stereo8(
                    InterleavedStereo8::splat(ui),
                    &mut v_st,
                    a_rel_v,
                    a_atk_v,
                    kap_v,
                );
                let (yl, yr) = y_st.into_lr();
                prop_assert_eq!(yl.to_array(), y_v.to_array());
                prop_assert_eq!(yr.to_array(), y_v.to_array());
                prop_assert_eq!(full_fwd[idx].to_bits(), y_v.to_array()[0].to_bits());
                if idx < valid_len {
                    prop_assert_eq!(part_fwd[idx].to_bits(), full_fwd[idx].to_bits());
                } else {
                    prop_assert_eq!(part_fwd[idx].to_bits(), 0);
                }
            }
            prop_assert_eq!(full_end.to_bits(), full_fwd[63].to_bits());
            prop_assert_eq!(part_end.to_bits(), full_fwd[valid_len - 1].to_bits());
        }
    }
}
