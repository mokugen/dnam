//! Compile-time `INV_K`, branchless compare-exchange sorting networks (`M=8` 19 ops,
//! `M=16` 60 ops), Welford `1.5-Entmax` scan with `k* == 1` `0-ULP` fast-exit, `f32` runtime
//! `rsqrt_nr3_f32` parity mirror, and closed-form `0-sqrt` `entmax15_vjp`
//! (`ref[dsp.ctrl.gra.rational-and-entmax]`, `ref[repo.code.constant-eval-and-derivations]`).

use dnam_vm::math::rsqrt_nr3_f32;

use super::model::{GRA_MAX_M, canonicalize_zero_f32, canonicalize_zero_f64};

const fn build_inv_k_f64() -> [f64; GRA_MAX_M + 1] {
    let mut out = [0.0_f64; GRA_MAX_M + 1];
    let mut k = 1_usize;
    while k <= GRA_MAX_M {
        out[k] = 1.0 / (k as f64);
        k += 1;
    }
    out
}

const fn build_inv_k_f32() -> [f32; GRA_MAX_M + 1] {
    let mut out = [0.0_f32; GRA_MAX_M + 1];
    let mut k = 1_usize;
    while k <= GRA_MAX_M {
        out[k] = (1.0_f64 / (k as f64)) as f32;
        k += 1;
    }
    out
}

/// Compile-time reciprocals `1 / k` (`k = 1..=16`) in `binary64`.
pub const INV_K_F64: [f64; GRA_MAX_M + 1] = build_inv_k_f64();
/// Compile-time reciprocals `1 / k` (`k = 1..=16`) in `binary32`.
pub const INV_K_F32: [f32; GRA_MAX_M + 1] = build_inv_k_f32();

/// Optimal 19-comparator Batcher sorting network for `M = 8` lanes.
pub const SORT_NET_8: [(usize, usize); 19] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (1, 2),
    (5, 6),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (2, 4),
    (3, 5),
    (1, 2),
    (3, 4),
    (5, 6),
];

/// Optimal 60-comparator Green sorting network for `M = 16` lanes.
pub const SORT_NET_16: [(usize, usize); 60] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (8, 10),
    (9, 11),
    (12, 14),
    (13, 15),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (8, 12),
    (9, 13),
    (10, 14),
    (11, 15),
    (0, 8),
    (1, 9),
    (2, 10),
    (3, 11),
    (4, 12),
    (5, 13),
    (6, 14),
    (7, 15),
    (5, 10),
    (6, 9),
    (3, 12),
    (13, 14),
    (7, 11),
    (1, 2),
    (4, 8),
    (1, 4),
    (7, 13),
    (2, 8),
    (11, 14),
    (2, 4),
    (5, 6),
    (9, 10),
    (11, 13),
    (3, 8),
    (7, 12),
    (6, 8),
    (3, 5),
    (7, 9),
    (10, 12),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (6, 7),
    (8, 9),
];

const _: () = assert!(SORT_NET_8.len() == 19 && SORT_NET_16.len() == 60);
const _: () = assert!((INV_K_F64[16] - 0.0625).abs() == 0.0);
const _: () = assert!(INV_K_F32[8].to_bits() == 0.125_f32.to_bits());

#[inline(always)]
fn cmp_swap_desc_f64(vals: &mut [f64], i: usize, j: usize) {
    let a = vals[i];
    let b = vals[j];
    let swap = a < b;
    vals[i] = if swap { b } else { a };
    vals[j] = if swap { a } else { b };
}

#[inline(always)]
fn cmp_swap_desc_f32(vals: &mut [f32], i: usize, j: usize) {
    let a = vals[i];
    let b = vals[j];
    let swap = a < b;
    vals[i] = if swap { b } else { a };
    vals[j] = if swap { a } else { b };
}

/// Branchless descending sorting network over `M = 8` `f64` scores.
#[inline]
pub fn sort_desc_8_f64(vals: &mut [f64; 8]) {
    for &(i, j) in &SORT_NET_8 {
        cmp_swap_desc_f64(vals, i, j);
    }
}

/// Branchless descending sorting network over `M = 16` `f64` scores.
#[inline]
pub fn sort_desc_16_f64(vals: &mut [f64; 16]) {
    for &(i, j) in &SORT_NET_16 {
        cmp_swap_desc_f64(vals, i, j);
    }
}

/// Branchless descending sorting network over `M = 8` `f32` scores.
#[inline]
pub fn sort_desc_8_f32(vals: &mut [f32; 8]) {
    for &(i, j) in &SORT_NET_8 {
        cmp_swap_desc_f32(vals, i, j);
    }
}

/// Branchless descending sorting network over `M = 16` `f32` scores.
#[inline]
pub fn sort_desc_16_f32(vals: &mut [f32; 16]) {
    for &(i, j) in &SORT_NET_16 {
        cmp_swap_desc_f32(vals, i, j);
    }
}

/// Forward `1.5-Entmax` output and cached support state for exact `0-sqrt` VJP.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entmax15Output {
    /// Active prototype count (`8` or `16`).
    pub len: usize,
    /// Normalized probability weights `p_m = q_m^2` (`sum p_m == 1.0`).
    pub probs: [f64; GRA_MAX_M],
    /// Unsquared active support offsets `q_m = max(0, 0.5 * s_m - tau)`.
    pub q: [f64; GRA_MAX_M],
    /// Learned threshold `tau`.
    pub tau: f64,
    /// Active support size `k* in 1..=len`.
    pub k_star: usize,
}

impl Default for Entmax15Output {
    fn default() -> Self {
        Self {
            len: 0,
            probs: [0.0; GRA_MAX_M],
            q: [0.0; GRA_MAX_M],
            tau: 0.0,
            k_star: 1,
        }
    }
}

/// Evaluates `1.5-Entmax` in `f64` using branchless sorting (`M=8` or `16`), Welford online
/// scan with `k* == 1` fast-exit, and a single `sqrt` only when `k* > 1`.
#[must_use]
pub fn entmax15_forward_f64(scores: &[f64]) -> Entmax15Output {
    let m = scores.len();
    assert!(
        m == 8 || m == 16,
        "entmax15_forward_f64 expects M in {{8, 16}}, got {m}"
    );
    let mut s_tilde = [0.0_f64; GRA_MAX_M];
    let mut sorted = [f64::NEG_INFINITY; GRA_MAX_M];
    for i in 0..m {
        let val = if scores[i].is_finite() {
            0.5 * scores[i]
        } else {
            0.0
        };
        s_tilde[i] = val;
        sorted[i] = val;
    }

    if m == 8 {
        let mut s8 = [0.0_f64; 8];
        s8.copy_from_slice(&sorted[..8]);
        sort_desc_8_f64(&mut s8);
        sorted[..8].copy_from_slice(&s8);
    } else {
        sort_desc_16_f64(&mut sorted);
    }

    let mut out = Entmax15Output {
        len: m,
        ..Entmax15Output::default()
    };

    // Fast-exit when `k* == 1` (`s_tilde[0] - s_tilde[1] >= 1.0`): zero sqrt/rsqrt calls.
    if sorted[0] - sorted[1] >= 1.0 {
        out.tau = sorted[0] - 1.0;
        out.k_star = 1;
        let mut winner = 0_usize;
        let mut best = s_tilde[0];
        for (i, &val) in s_tilde.iter().enumerate().take(m).skip(1) {
            if val > best {
                best = val;
                winner = i;
            }
        }
        out.q[winner] = 1.0;
        out.probs[winner] = 1.0;
        return out;
    }

    // Shift by `s_max = sorted[0]` before Welford scan so `tau_rel` subtraction is shift-invariant
    // and never catastrophically cancels on `±1e9` (`REQ-7.2`) logits.
    let s_max = sorted[0];
    let mut mu_rel = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut best_k = 1_usize;
    let mut best_mu_rel = 0.0_f64;
    let mut best_rem = 1.0_f64;
    for (j, &x) in sorted.iter().enumerate().take(m).skip(1) {
        let x_rel = x - s_max;
        if !x_rel.is_finite() {
            break;
        }
        let k = j + 1;
        let delta1 = x_rel - mu_rel;
        mu_rel += delta1 * INV_K_F64[k];
        let delta2 = x_rel - mu_rel;
        m2 += delta1 * delta2;
        let rem = 1.0 - m2;
        if rem <= 0.0 || !rem.is_finite() {
            break;
        }
        let diff = mu_rel - x_rel;
        if rem > (k as f64) * diff * diff {
            best_k = k;
            best_mu_rel = mu_rel;
            best_rem = rem;
        }
    }

    let tau_rel = best_mu_rel - (best_rem * INV_K_F64[best_k]).sqrt();
    out.tau = s_max + tau_rel;
    out.k_star = best_k;
    for i in 0..m {
        let s_tilde_rel = s_tilde[i] - s_max;
        let qm = (s_tilde_rel - tau_rel).max(0.0);
        out.q[i] = canonicalize_zero_f64(qm);
        out.probs[i] = canonicalize_zero_f64(qm * qm);
    }
    out
}

/// Evaluates runtime-mirror `1.5-Entmax` in `binary32` using branchless sorting, Welford scan,
/// `k* == 1` `0-ULP` fast-exit, and a single `rsqrt_nr3_f32` call (`0 FDIV`, `0 FSQRT`).
#[must_use]
pub fn entmax15_forward_f32_mirror(scores: &[f32]) -> ([f32; GRA_MAX_M], usize) {
    let m = scores.len();
    assert!(
        m == 8 || m == 16,
        "entmax15_forward_f32_mirror expects M in {{8, 16}}, got {m}"
    );
    let mut s_tilde = [0.0_f32; GRA_MAX_M];
    let mut sorted = [f32::NEG_INFINITY; GRA_MAX_M];
    for i in 0..m {
        let val = if scores[i].is_finite() {
            0.5 * scores[i]
        } else {
            0.0
        };
        s_tilde[i] = val;
        sorted[i] = val;
    }

    if m == 8 {
        let mut s8 = [0.0_f32; 8];
        s8.copy_from_slice(&sorted[..8]);
        sort_desc_8_f32(&mut s8);
        sorted[..8].copy_from_slice(&s8);
    } else {
        sort_desc_16_f32(&mut sorted);
    }

    let mut probs = [0.0_f32; GRA_MAX_M];
    if sorted[0] - sorted[1] >= 1.0 {
        let mut winner = 0_usize;
        let mut best = s_tilde[0];
        for (i, &val) in s_tilde.iter().enumerate().take(m).skip(1) {
            if val > best {
                best = val;
                winner = i;
            }
        }
        probs[winner] = 1.0_f32;
        return (probs, 1);
    }

    // Shift by `s_max = sorted[0]` before Welford scan so `tau_rel` subtraction is shift-invariant
    // and never catastrophically cancels on `±1e9` (`REQ-7.2`) logits.
    let s_max = sorted[0];
    let mut mu_rel = 0.0_f32;
    let mut m2 = 0.0_f32;
    let mut best_k = 1_usize;
    let mut best_mu_rel = 0.0_f32;
    let mut best_rem = 1.0_f32;
    for (j, &x) in sorted.iter().enumerate().take(m).skip(1) {
        let x_rel = x - s_max;
        if !x_rel.is_finite() {
            break;
        }
        let k = j + 1;
        let delta1 = x_rel - mu_rel;
        mu_rel += delta1 * INV_K_F32[k];
        let delta2 = x_rel - mu_rel;
        m2 += delta1 * delta2;
        let rem = 1.0_f32 - m2;
        if rem <= 0.0 || !rem.is_finite() {
            break;
        }
        let diff = mu_rel - x_rel;
        if rem > (k as f32) * diff * diff {
            best_k = k;
            best_mu_rel = mu_rel;
            best_rem = rem;
        }
    }

    let rad = (best_rem * INV_K_F32[best_k]).max(1.0e-12_f32);
    let sqrt_rad = rad * rsqrt_nr3_f32(rad);
    let tau_rel = best_mu_rel - sqrt_rad;
    for i in 0..m {
        let s_tilde_rel = s_tilde[i] - s_max;
        let qm = (s_tilde_rel - tau_rel).max(0.0_f32);
        probs[i] = canonicalize_zero_f32(qm * qm);
    }
    (probs, best_k)
}

/// Exact closed-form `0-sqrt` reverse-mode VJP for `1.5-Entmax` with respect to raw scores
/// `s = 2 * s_tilde`: `dL/ds[m] = q_m * (g_m - g_bar)` where `g_bar = sum(q_j * g_j) / sum(q_j)`.
#[must_use]
pub fn entmax15_vjp_scores(out: &Entmax15Output, grad_probs: &[f64]) -> [f64; GRA_MAX_M] {
    let mut grad_scores = [0.0_f64; GRA_MAX_M];
    let m = out.len.min(grad_probs.len()).min(GRA_MAX_M);
    if m == 0 || out.k_star <= 1 {
        return grad_scores;
    }
    let mut sum_q = 0.0_f64;
    let mut sum_qg = 0.0_f64;
    for i in 0..m {
        let qi = out.q[i];
        if qi > 0.0 {
            sum_q += qi;
            sum_qg += qi * grad_probs[i];
        }
    }
    if sum_q <= 0.0 {
        return grad_scores;
    }
    let g_bar = sum_qg / sum_q;
    for i in 0..m {
        let qi = out.q[i];
        if qi > 0.0 {
            // `dL/ds_tilde[i] = 2 * q_i * (g_i - g_bar)`, and `ds_tilde/ds = 0.5`.
            grad_scores[i] = qi * (grad_probs[i] - g_bar);
        }
    }
    grad_scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_sorting_networks_zero_one_principle_exhaustive() {
        // Knuth Zero-One Principle: verifying all 2^8 and 2^16 binary inputs proves exact sort.
        for mask in 0..(1_u32 << 8) {
            let mut a = core::array::from_fn(|i| ((mask >> i) & 1) as f64);
            sort_desc_8_f64(&mut a);
            for w in a.windows(2) {
                assert!(w[0] >= w[1], "M=8 sort failed for mask={mask}");
            }
        }
        for mask in 0..(1_u32 << 16) {
            let mut a = core::array::from_fn(|i| ((mask >> i) & 1) as f64);
            sort_desc_16_f64(&mut a);
            for w in a.windows(2) {
                assert!(w[0] >= w[1], "M=16 sort failed for mask={mask}");
            }
        }
    }

    #[test]
    fn test_entmax15_fast_exit_uniform_and_vjp_fd() {
        // 1. `k* == 1` fast-exit gives exact [1, 0, ...] in both f64 and f32 mirror.
        let s_fast = [2.0_f64, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out_fast = entmax15_forward_f64(&s_fast);
        assert_eq!(out_fast.k_star, 1);
        assert_eq!(out_fast.probs[0], 1.0);
        assert!(out_fast.probs[1..8].iter().all(|&p| p == 0.0));
        let (p32_fast, k32) = entmax15_forward_f32_mirror(&s_fast.map(|v| v as f32));
        assert_eq!(k32, 1);
        assert_eq!(p32_fast[0].to_bits(), 1.0_f32.to_bits());
        assert!(p32_fast[1..8].iter().all(|p| p.to_bits() == 0));
        let g_fast = entmax15_vjp_scores(&out_fast, &[1.0; 8]);
        assert!(g_fast[..8].iter().all(|&g| g == 0.0));

        // 2. Uniform scores yield exact 1/M.
        let out_unif = entmax15_forward_f64(&[0.0; 16]);
        assert_eq!(out_unif.k_star, 16);
        for p in out_unif.probs {
            assert!((p - 1.0 / 16.0).abs() < 1.0e-12);
        }

        // 3. Interior multi-support central FD for M=8 and M=16.
        for scores in [
            vec![1.4, 0.9, 0.6, 0.2, -0.5, -1.2, 0.4, -0.1],
            vec![
                1.2, 0.85, 0.7, 0.55, 0.3, -0.2, -0.8, 0.1, 0.95, 0.4, -1.0, 0.0, 0.6, -0.4, 0.2,
                -0.6,
            ],
        ] {
            let weights: Vec<f64> = (0..scores.len())
                .map(|i| ((i as f64) * 0.73 + 0.2).sin())
                .collect();
            let out = entmax15_forward_f64(&scores);
            assert!(out.k_star > 1);
            let sum_p: f64 = out.probs[..scores.len()].iter().sum();
            assert!((sum_p - 1.0).abs() < 1.0e-12);

            let (p32, k32_m) =
                entmax15_forward_f32_mirror(&scores.iter().map(|&v| v as f32).collect::<Vec<_>>());
            assert_eq!(k32_m, out.k_star);
            for i in 0..scores.len() {
                assert!(
                    (f64::from(p32[i]) - out.probs[i]).abs() < 2.0e-6,
                    "f32/f64 mismatch at {i}: {} vs {}",
                    p32[i],
                    out.probs[i]
                );
            }

            let ana = entmax15_vjp_scores(&out, &weights);
            let eps = 1.0e-6_f64;
            for i in 0..scores.len() {
                let mut sp = scores.clone();
                let mut sm = scores.clone();
                sp[i] += eps;
                sm[i] -= eps;
                let fp: f64 = entmax15_forward_f64(&sp).probs[..scores.len()]
                    .iter()
                    .zip(&weights)
                    .map(|(p, w)| p * w)
                    .sum();
                let fm: f64 = entmax15_forward_f64(&sm).probs[..scores.len()]
                    .iter()
                    .zip(&weights)
                    .map(|(p, w)| p * w)
                    .sum();
                let fd = (fp - fm) / (2.0 * eps);
                assert!(
                    (ana[i] - fd).abs() < 1.0e-7 * ana[i].abs().max(1.0),
                    "M={} idx={i} ana={} fd={fd}",
                    scores.len(),
                    ana[i]
                );
            }
        }
    }

    #[test]
    fn test_entmax15_shift_invariance_1e9_f32_mirror_and_vjp_fd() {
        use crate::control::gra::forward::{eval_gra_f32_mirror, eval_gra_forward};
        use crate::control::gra::model::{GraParams, GraTierProfile};

        // 1. `f32` mirror under large positive shifts (`1e5`, `1e7`, `1e8`, `1e9`) and `REQ-7.2`.
        for shift in [1.0e5_f32, 1.0e7_f32, 1.0e8_f32, 1.0e9_f32] {
            let scores = [shift, shift, shift - 1.0_f32, -shift, 0.0, 0.0, 0.0, 0.0];
            let (probs, k_star) = entmax15_forward_f32_mirror(&scores);
            let sum: f32 = probs[..8].iter().sum();
            assert!(
                (sum - 1.0_f32).abs() < 1.0e-5_f32,
                "shift={shift}: sum={sum}, k_star={k_star}, probs={:?}",
                &probs[..8]
            );
            assert!(probs[..8].iter().all(|p| p.is_finite() && *p >= 0.0));
        }

        // Exact `1e9` test case from challenger report (M=8).
        let exact_1e9 = [1.0e9_f32, 1.0e9, 1.0e9 - 1.0, -1.0e9, 0.0, 0.0, 0.0, 0.0];
        let (p_1e9, _) = entmax15_forward_f32_mirror(&exact_1e9);
        let sum_1e9: f32 = p_1e9[..8].iter().sum();
        assert!((sum_1e9 - 1.0_f32).abs() < 1.0e-5_f32, "sum_1e9={sum_1e9}");

        // 2. End-to-end `eval_gra_f32_mirror` vs `eval_gra_forward` with huge tied/near-tied `w_num`.
        let mut params = GraParams::zeros(GraTierProfile::Lite, 2);
        params.w_in[0][0] = 1.0;
        params.w_num[0][0] = 1.0e8 - 2.0;
        params.w_num[1][0] = 1.0e8;
        params.v_proto[1][0] = 2.0;
        params.enforce_structural_invariants();

        let (out64, _) = eval_gra_forward(&params, &[1.0, 0.0], [0.0; 4], [0.0; 4], 0.0);
        let out32 = eval_gra_f32_mirror(&params, &[1.0, 0.0], [0.0; 4], [0.0; 4], 0.0);
        assert!(
            (out64.stage_targets[0] - 1.0).abs() < 1.0e-12,
            "f64 stage_targets[0]={}",
            out64.stage_targets[0]
        );
        assert!(
            (f64::from(out32.stage_targets[0]) - 1.0).abs() < 1.0e-5,
            "f32 stage_targets[0]={}",
            out32.stage_targets[0]
        );

        // 3. `f64` VJP FD parity `< 1e-8` at `+500.0` and `+1e8` logit shifts.
        let base = [1.4, 0.9, 0.6, 0.2, -0.5, -1.2, 0.4, -0.1];
        let weights: Vec<f64> = (0..8).map(|i| ((i as f64) * 0.73 + 0.2).sin()).collect();
        let eps = 1.0e-6_f64;

        let shifted_500: Vec<f64> = base.iter().map(|&v| v + 500.0).collect();
        let out_500 = entmax15_forward_f64(&shifted_500);
        assert!((out_500.probs[..8].iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
        let ana_500 = entmax15_vjp_scores(&out_500, &weights);
        for i in 0..8 {
            let mut sp = shifted_500.clone();
            let mut sm = shifted_500.clone();
            sp[i] += eps;
            sm[i] -= eps;
            let fp: f64 = entmax15_forward_f64(&sp).probs[..8]
                .iter()
                .zip(&weights)
                .map(|(p, w)| p * w)
                .sum();
            let fm: f64 = entmax15_forward_f64(&sm).probs[..8]
                .iter()
                .zip(&weights)
                .map(|(p, w)| p * w)
                .sum();
            let fd = (fp - fm) / (2.0 * eps);
            assert!(
                (ana_500[i] - fd).abs() < 1.0e-8 * ana_500[i].abs().max(1.0),
                "+500 shift idx={i}: ana={} fd={fd}",
                ana_500[i]
            );
        }

        // At `+1e8`, compare VJP at `+1e8` against Sterbenz-exact de-shifted FD (`s_1e8 - 1e8`).
        let shifted_1e8: Vec<f64> = base.iter().map(|&v| v + 1.0e8).collect();
        let out_1e8 = entmax15_forward_f64(&shifted_1e8);
        assert!((out_1e8.probs[..8].iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
        let ana_1e8 = entmax15_vjp_scores(&out_1e8, &weights);
        let exact_deshifted: Vec<f64> = shifted_1e8.iter().map(|&v| v - 1.0e8).collect();
        for i in 0..8 {
            let mut sp = exact_deshifted.clone();
            let mut sm = exact_deshifted.clone();
            sp[i] += eps;
            sm[i] -= eps;
            let fp: f64 = entmax15_forward_f64(&sp).probs[..8]
                .iter()
                .zip(&weights)
                .map(|(p, w)| p * w)
                .sum();
            let fm: f64 = entmax15_forward_f64(&sm).probs[..8]
                .iter()
                .zip(&weights)
                .map(|(p, w)| p * w)
                .sum();
            let fd = (fp - fm) / (2.0 * eps);
            assert!(
                (ana_1e8[i] - fd).abs() < 1.0e-8 * ana_1e8[i].abs().max(1.0),
                "+1e8 shift idx={i}: ana={} fd={fd}",
                ana_1e8[i]
            );
        }
    }

    proptest! {
        #[test]
        fn prop_entmax15_simplex_and_vjp_sum_zero(
            scores in prop::collection::vec(-3.0_f64..3.0_f64, 16),
            grad_p in prop::collection::vec(-2.0_f64..2.0_f64, 16),
        ) {
            let out = entmax15_forward_f64(&scores);
            let sum_p: f64 = out.probs.iter().sum();
            prop_assert!((sum_p - 1.0).abs() < 1.0e-10, "sum_p={sum_p}");
            prop_assert!(out.probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
            let g_s = entmax15_vjp_scores(&out, &grad_p);
            let sum_g: f64 = g_s.iter().sum();
            prop_assert!(sum_g.abs() < 1.0e-11, "VJP sum must vanish: {sum_g}");
        }
    }
}
