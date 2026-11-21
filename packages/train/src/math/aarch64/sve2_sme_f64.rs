//! ARMv9 `ArmSve2Backend` and `ArmSmeBackend` double-precision [`TrainerMathBackend`]
//! implementations delegating fixed 128-bit `f64` NEON lanes to [`NeonBackend`] for exact
//! cross-backend IEEE-754 parity on host hardware without requiring optional SVE2/SME traps.

use super::neon_f64::NeonF64Vec2;
use crate::math::trainer::TrainerMathBackend;
use crate::math::{ArmSmeBackend, ArmSve2Backend, NeonBackend};

impl TrainerMathBackend for ArmSve2Backend {
    type F64Vec2 = NeonF64Vec2;

    #[inline(always)]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        NeonBackend::sub_f32_to_f64(dst, a, b);
    }

    #[inline]
    fn gram_tile_4x4_f64(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
        NeonBackend::gram_tile_4x4_f64(cols_i, cols_j)
    }

    #[inline]
    fn gram_sym_block4_f64(cols: [&[f64]; 4]) -> [f64; 10] {
        NeonBackend::gram_sym_block4_f64(cols)
    }

    #[inline]
    fn advance_bezout7_lp_dim_f64(
        fb: &[f32; 256],
        dim: usize,
        r: f64,
        omr: f64,
        u: &mut [[f64; 256]; 7],
    ) {
        NeonBackend::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
    }
}

impl TrainerMathBackend for ArmSmeBackend {
    type F64Vec2 = NeonF64Vec2;

    #[inline(always)]
    fn sub_f32_to_f64(dst: &mut [f64], a: &[f32], b: &[f32]) {
        NeonBackend::sub_f32_to_f64(dst, a, b);
    }

    #[inline]
    fn gram_tile_4x4_f64(cols_i: [&[f64]; 4], cols_j: [&[f64]; 4]) -> [[f64; 4]; 4] {
        NeonBackend::gram_tile_4x4_f64(cols_i, cols_j)
    }

    #[inline]
    fn gram_sym_block4_f64(cols: [&[f64]; 4]) -> [f64; 10] {
        NeonBackend::gram_sym_block4_f64(cols)
    }

    #[inline]
    fn advance_bezout7_lp_dim_f64(
        fb: &[f32; 256],
        dim: usize,
        r: f64,
        omr: f64,
        u: &mut [[f64; 256]; 7],
    ) {
        NeonBackend::advance_bezout7_lp_dim_f64(fb, dim, r, omr, u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::BackendKind;
    use crate::math::trainer::{DynamicTrainer, TrainerTask, verify_trainer_backend_parity};

    #[test]
    fn test_sve2_sme_trainer_math_parity() {
        verify_trainer_backend_parity::<ArmSve2Backend>();
        verify_trainer_backend_parity::<ArmSmeBackend>();
        struct KindTask;
        impl TrainerTask<BackendKind> for KindTask {
            fn run<B: TrainerMathBackend>(self) -> BackendKind {
                B::trainer_backend_kind()
            }
        }
        assert_eq!(
            DynamicTrainer::from_kind(BackendKind::ArmSve2).dispatch(KindTask),
            BackendKind::ArmSve2
        );
        assert_eq!(
            DynamicTrainer::from_kind(BackendKind::ArmSme).dispatch(KindTask),
            BackendKind::ArmSme
        );
    }
}
