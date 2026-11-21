#![forbid(unsafe_code)]
//! Integration suite for `dnam-train` `f64` SIMD [`TrainerMathBackend`], [`DynamicTrainer`],
//! [`Wide192Accum`], register-tiled `uplo='U'` contractions, and canonical `DNAM-GRAM-INPUT-V3\0`
//! input hashing (`verify[train.gram.wide192]`, `verify[train.gram.accuracy-contract]`).

use dnam_train::{
    BackendKind, DefaultMathBackend, DynamicTrainer, ScalarBackend, SimdF64Vec2,
    TrainerMathBackend, TrainerTask, Wide192Accum, accumulate_uplo_u_wide192,
    contract_256d_gram_tiled_wide192, contract_dim_cross_only_wide192,
    contract_dim_gram_tiled_wide192, contract_dim_gram_tiled_wide192_with_backend,
    dispatch_trainer, hash_gram_input_v3, hash_gram_input_v3_f32, uplo_u_index, uplo_u_packed_len,
    warm_simd_feature_caches,
};
use proptest::prelude::*;
use sha2::{Digest, Sha256};

#[test]
fn test_math_backend_zero_ulp_non_fma_and_dynamic_dispatch() {
    // verify[train.gram.accuracy-contract]
    warm_simd_feature_caches();

    struct BitParityTask {
        fi: Vec<[f64; 4]>,
        wfj: Vec<[f64; 4]>,
    }
    impl TrainerTask<[[f64; 4]; 4]> for BitParityTask {
        fn run<B: TrainerMathBackend>(self) -> [[f64; 4]; 4] {
            let mut u1 = [[0.0_f64; 256]; 7];
            let mut u2 = [[0.0_f64; 256]; 7];
            let fb: [f32; 256] = core::array::from_fn(|i| ((i as f32) * 0.19).sin() * 0.5);
            B::advance_bezout7_lp_dim_f64(&fb, 73, 0.94, 0.06, &mut u1);
            B::advance_bezout7_lp_dim_f64(&fb, 73, 0.71, 0.29, &mut u2);
            B::contract_tile_4x4_ieee_f64(&self.fi, &self.wfj)
        }
    }

    let fi: Vec<[f64; 4]> = (0..19)
        .map(|k| [0.11 * (k as f64), -0.35, 0.67, -0.19])
        .collect();
    let wfj: Vec<[f64; 4]> = (0..19)
        .map(|k| [0.43, -0.21, 0.15 + 0.02 * (k as f64), 0.77])
        .collect();
    let scalar_res = DynamicTrainer::scalar().dispatch(BitParityTask {
        fi: fi.clone(),
        wfj: wfj.clone(),
    });
    let hw_res = dispatch_trainer(BitParityTask { fi, wfj });
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(scalar_res[r][c].to_bits(), hw_res[r][c].to_bits());
        }
    }

    let dyn_auto = DynamicTrainer::default();
    assert_eq!(dyn_auto.kind(), DefaultMathBackend::trainer_backend_kind());
    assert_eq!(
        DynamicTrainer::from_kind(BackendKind::Scalar).kind(),
        BackendKind::Scalar
    );
    assert_eq!(
        <ScalarBackend as TrainerMathBackend>::F64Vec2::splat_pair(1.5, -2.5)
            .dot(<ScalarBackend as TrainerMathBackend>::F64Vec2::splat(2.0)),
        -2.0
    );
}

#[test]
fn test_wide192_cancellation_prime_stride_and_dnam_gram_input_v3_transcript() {
    // verify[train.gram.wide192]
    // verify[train.gram.accuracy-contract]
    let two_neg60 = 2.0_f64.powi(-60);
    let tiny_f32 = 2.0_f32.powi(-30);
    let mut feats = vec![[0.0_f32; 256]; 520];
    let mut targets = vec![[0.0_f64; 2]; 520];
    let weights = vec![1.0_f64; 520];

    feats[0][0] = 1.0;
    feats[0][9] = 1.0;
    feats[1][0] = tiny_f32;
    feats[1][9] = tiny_f32;
    feats[2][0] = -1.0;
    feats[2][9] = 1.0;

    feats[256][5] = 1.0;
    feats[257][5] = 1.0;
    feats[519][5] = 1.0;
    targets[256] = [1.0, -1.0];
    targets[257] = [two_neg60, two_neg60];
    targets[519] = [-1.0, 1.0];

    let (g, c) = contract_256d_gram_tiled_wide192(&feats, &targets, &weights, 1);
    assert_eq!(g[uplo_u_index(0, 9)].to_bits(), two_neg60.to_bits());
    assert_eq!(c[5][0].to_bits(), two_neg60.to_bits());
    assert_eq!(c[5][1].to_bits(), two_neg60.to_bits());

    // Exact normative byte transcript verification for DNAM-GRAM-INPUT-V3\0 (using zerocopy endianness).
    use zerocopy::IntoBytes;
    use zerocopy::byteorder::little_endian::{F64 as LeF64, U32 as LeU32, U64 as LeU64};

    let row0 = [-0.0_f64, 1.25];
    let row1 = [2.5_f64, -3.75];
    let w = [1.0_f64, 0.5_f64];
    let actual_hash = hash_gram_input_v3(2, &[&row0, &row1], &w);

    let mut manual = Sha256::new();
    manual.update(b"DNAM-GRAM-INPUT-V3\0");
    manual.update(LeU32::new(2).as_bytes());
    manual.update([2_u8, 2_u8]);
    for (idx, (&row, &wt)) in [&row0, &row1].iter().zip(&w).enumerate() {
        manual.update([1_u8]);
        manual.update(LeU64::new(idx as u64).as_bytes());
        manual.update(LeF64::new(if wt == 0.0 { 0.0 } else { wt }).as_bytes());
        for &x in row {
            manual.update(LeF64::new(if x == 0.0 { 0.0 } else { x }).as_bytes());
        }
    }
    manual.update([0_u8]);
    manual.update(LeU64::new(2).as_bytes());
    let expected_hash: [u8; 32] = manual.finalize().into();
    assert_eq!(actual_hash, expected_hash);

    let flat_f32: Vec<f32> = [
        row0[0] as f32,
        row0[1] as f32,
        row1[0] as f32,
        row1[1] as f32,
    ]
    .to_vec();
    assert_eq!(hash_gram_input_v3_f32(2, &flat_f32, &w, 1), expected_hash);

    // Non-divisible prime stride integral conservation check
    let const_feats = vec![1.0_f32; 100 * 6];
    let ones_tgt = vec![[1.0, -0.5]; 100];
    let unit_w = vec![1.0; 100];
    let (g_s7, c_s7) = contract_dim_gram_tiled_wide192(&const_feats, &ones_tgt, &unit_w, 6, 7);
    let c_only = contract_dim_cross_only_wide192(&const_feats, &ones_tgt, &unit_w, 6, 7);
    assert!((g_s7[uplo_u_index(2, 5)] - 100.0).abs() <= 1.0e-11);
    assert!((c_s7[0][0] - 100.0).abs() <= 1.0e-11);
    assert!((c_s7[0][1] + 50.0).abs() <= 1.0e-11);
    assert_eq!(c_s7, c_only);
}

proptest! {
    #[test]
    fn prop_tiled_gram_matches_naive_wide192_and_scalar(
        num_frames in 1usize..40,
        dim in 1usize..12,
        vals in prop::collection::vec(-4.0_f32..4.0_f32, 480),
        w_raw in prop::collection::vec(0.1_f64..2.0_f64, 40),
    ) {
        let feats: Vec<f32> = vals[..num_frames * dim].to_vec();
        let targets: Vec<[f64; 2]> = (0..num_frames)
            .map(|t| [f64::from(feats[t * dim]) * 0.5, -0.25 * (t as f64)])
            .collect();
        let weights = &w_raw[..num_frames];

        let (g_def, c_def) = contract_dim_gram_tiled_wide192(&feats, &targets, weights, dim, 1);
        let (g_scl, c_scl) = contract_dim_gram_tiled_wide192_with_backend::<ScalarBackend>(
            &feats, &targets, weights, dim, 1,
        );
        prop_assert_eq!(g_def.len(), uplo_u_packed_len(dim));
        for (a, b) in g_def.iter().zip(&g_scl) {
            prop_assert!((a - b).abs() <= 1.0e-11);
        }
        prop_assert_eq!(c_def.len(), c_scl.len());

        let mut g_ref = vec![Wide192Accum::zero(); uplo_u_packed_len(dim)];
        let mut c_ref = vec![[Wide192Accum::zero(); 2]; dim];
        for t in 0..num_frames {
            let row64: Vec<f64> = feats[t * dim..(t + 1) * dim].iter().map(|&v| f64::from(v)).collect();
            accumulate_uplo_u_wide192(&mut g_ref, &mut c_ref, &row64, targets[t], weights[t]);
        }
        for (got, exp) in g_def.iter().zip(&g_ref) {
            prop_assert!((*got - exp.to_f64()).abs() <= 1.0e-11);
        }
    }
}
