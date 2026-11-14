#![forbid(unsafe_code)]
//! Capture/Trainer Operator & Composite Backend Abstraction (`#![forbid(unsafe_code)]`).
//!
//! Mirrors [`dnam_engine`]'s `StageBackend`/`EngineBackend` hierarchy by extending
//! [`TrainerMathBackend`] directly via supertraits ([`GaborOperatorBackend`],
//! [`KautzCascadeBackend`], [`GramCholeskyBackend`]) and a blanket [`TrainerBackend`]
//! implemented for every CPU math backend, dispatched once per job via [`DynamicTrainer::detect`].

pub mod gabor_op;
pub mod gram_op;
pub mod kautz_op;

use dnam_engine::math::trainer::TrainerMathBackend;
use dnam_engine::math::{BackendKind, DefaultMathBackend, ScalarBackend, detect_best_backend};

pub use gabor_op::GaborOperatorBackend;
pub use gram_op::GramCholeskyBackend;
pub use kautz_op::KautzCascadeBackend;

/// Composite CPU trainer backend extending [`TrainerMathBackend`] with all 3 operator domains.
pub trait TrainerBackend:
    TrainerMathBackend + GaborOperatorBackend + KautzCascadeBackend + GramCholeskyBackend
{
    /// Returns the runtime backend kind.
    #[inline(always)]
    fn backend_kind() -> BackendKind {
        Self::trainer_backend_kind()
    }

    /// Returns the human-readable hardware backend name.
    #[inline(always)]
    fn name() -> &'static str {
        Self::trainer_backend_kind().name()
    }
}

impl<M> TrainerBackend for M where
    M: TrainerMathBackend + GaborOperatorBackend + KautzCascadeBackend + GramCholeskyBackend
{
}

/// Portable scalar trainer backend (`#![forbid(unsafe_code)]`).
pub type ScalarTrainerBackend = ScalarBackend;
/// Compile-time default hardware trainer backend for the current target.
pub type DefaultTrainerBackend = DefaultMathBackend;

/// ARM64 NEON trainer backend.
#[cfg(target_arch = "aarch64")]
pub type NeonTrainerBackend = dnam_engine::math::NeonBackend;
/// ARMv9 SVE2 trainer backend.
#[cfg(target_arch = "aarch64")]
pub type ArmSve2TrainerBackend = dnam_engine::math::ArmSve2Backend;
/// ARMv9.2 SME trainer backend.
#[cfg(target_arch = "aarch64")]
pub type ArmSmeTrainerBackend = dnam_engine::math::ArmSmeBackend;

/// x86-64 SSE2 trainer backend.
#[cfg(target_arch = "x86_64")]
pub type X86Sse2TrainerBackend = dnam_engine::math::X86Sse2Backend;
/// x86-64 SSE4.2 trainer backend.
#[cfg(target_arch = "x86_64")]
pub type X86Sse42TrainerBackend = dnam_engine::math::X86Sse42Backend;
/// x86-64 AVX2 + FMA trainer backend (with strict non-FMA AVX `0-ULP` tiles/Bézout).
#[cfg(target_arch = "x86_64")]
pub type X86Avx2FmaTrainerBackend = dnam_engine::math::X86Avx2FmaBackend;

/// WebAssembly SIMD128 trainer backend.
#[cfg(target_arch = "wasm32")]
pub type WasmSimd128TrainerBackend = dnam_engine::math::WasmSimd128Backend;

/// Generic callback executed inside the selected hardware [`TrainerBackend`].
pub trait TrainerTask<R> {
    /// Executes the trainer task monomorphized for `B`.
    fn run<B: TrainerBackend>(self) -> R;
}

/// Runtime-dispatched trainer backend selector (one-time dispatch per job, zero inner branches).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicTrainer {
    kind: BackendKind,
}

impl DynamicTrainer {
    /// Selects the fastest trainer backend supported by the current CPU.
    #[inline]
    #[must_use]
    pub fn detect() -> Self {
        Self {
            kind: detect_best_backend(),
        }
    }

    /// Selects the portable scalar reference backend.
    #[inline]
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            kind: BackendKind::Scalar,
        }
    }

    /// Selects an explicit [`BackendKind`] (unsupported kinds fall back to scalar).
    #[inline]
    #[must_use]
    pub const fn from_kind(kind: BackendKind) -> Self {
        Self { kind }
    }

    /// Returns the configured [`BackendKind`].
    #[inline]
    #[must_use]
    pub const fn kind(self) -> BackendKind {
        self.kind
    }

    /// Returns a concise human-readable hardware backend name.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.kind.name()
    }

    /// Dispatches `task` once to the matching monomorphized CPU [`TrainerBackend`].
    #[inline]
    pub fn dispatch<T: TrainerTask<R>, R>(self, task: T) -> R {
        match self.kind {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => task.run::<ArmSmeTrainerBackend>(),
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSve2 => task.run::<ArmSve2TrainerBackend>(),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon | BackendKind::ArmNeon => task.run::<NeonTrainerBackend>(),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Avx2Fma | BackendKind::X86Simd => {
                task.run::<X86Avx2FmaTrainerBackend>()
            }
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse42 => task.run::<X86Sse42TrainerBackend>(),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse2 => task.run::<X86Sse2TrainerBackend>(),
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => task.run::<WasmSimd128TrainerBackend>(),
            _ => task.run::<ScalarTrainerBackend>(),
        }
    }
}

impl Default for DynamicTrainer {
    #[inline]
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::solver::uplo_u_packed_len;

    fn verify_gabor_kautz_gram_parity<B: TrainerBackend>() {
        let len = 49usize;
        let x: Vec<f64> = (0..len).map(|i| ((i as f64) * 0.19).sin() * 0.6).collect();
        let cos_k: Vec<f64> = (0..21).map(|i| ((i as f64) * 0.31).cos()).collect();
        let sin_k: Vec<f64> = (0..21).map(|i| ((i as f64) * 0.31).sin()).collect();
        let (mut rs, mut is, mut rb, mut ib) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
        ScalarTrainerBackend::project_frames(&x, 3, 21, 16, &cos_k, &sin_k, &mut rs, &mut is);
        B::project_frames(&x, 3, 21, 16, &cos_k, &sin_k, &mut rb, &mut ib);
        for m in 0..2 {
            assert!((rs[m] - rb[m]).abs() <= 1.0e-11);
            assert!((is[m] - ib[m]).abs() <= 1.0e-11);
        }
        let (mut dys, mut dyb) = (vec![0.0_f64; len], vec![0.0_f64; len]);
        ScalarTrainerBackend::overlap_add_adjoint(
            &mut dys,
            3,
            21,
            16,
            &cos_k,
            &sin_k,
            &[0.4, -0.3],
            &[-0.2, 0.5],
        );
        B::overlap_add_adjoint(
            &mut dyb,
            3,
            21,
            16,
            &cos_k,
            &sin_k,
            &[0.4, -0.3],
            &[-0.2, 0.5],
        );
        assert!(dys.iter().zip(&dyb).all(|(a, b)| (a - b).abs() <= 1.0e-12));

        let s2: Vec<f64> = (0..24).map(|i| ((i as f64) * 0.13).sin()).collect();
        let pre = [
            (0.81, 0.25, 0.5),
            (0.75, -0.33, 0.55),
            (0.90, 0.12, 0.41),
            (0.68, -0.45, 0.61),
            (0.84, 0.21, 0.48),
        ];
        let mut cols_s = vec![vec![0.0_f64; 24]; 12];
        let mut cols_b = vec![vec![0.0_f64; 24]; 12];
        ScalarTrainerBackend::filter_pole_bank_columns(&s2, 4, &pre, &mut cols_s, 2);
        B::filter_pole_bank_columns(&s2, 4, &pre, &mut cols_b, 2);
        for c in 0..12 {
            assert!(
                cols_s[c]
                    .iter()
                    .zip(&cols_b[c])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }

        let poles: Vec<_> = pre
            .iter()
            .map(|&(lr, li, bg)| {
                let r = (lr * lr + li * li).sqrt();
                (r, bg, lr / r, li / r, 0.9, 1.1)
            })
            .collect();
        let kc = [
            (0.2, -0.1),
            (-0.3, 0.4),
            (0.15, 0.25),
            (-0.08, -0.12),
            (0.33, 0.07),
        ];
        let (mut srs, mut sis, mut srb, mut sib) = ([0.0; 5], [0.0; 5], [0.0; 5], [0.0; 5]);
        let (mut ors, mut ois, mut orb, mut oib) = ([0.0; 5], [0.0; 5], [0.0; 5], [0.0; 5]);
        let kvs = ScalarTrainerBackend::step_kautz_forward_frame(
            0.35, &poles, &kc, &mut srs, &mut sis, &mut ors, &mut ois,
        );
        let kvb =
            B::step_kautz_forward_frame(0.35, &poles, &kc, &mut srb, &mut sib, &mut orb, &mut oib);
        assert_eq!(kvs.to_bits(), kvb.to_bits());

        let (mut dsrs, mut dsis, mut dsrb, mut dsib) = ([0.1; 5], [-0.2; 5], [0.1; 5], [-0.2; 5]);
        let (mut gkcs, mut gkcb) = ([(0.0, 0.0); 5], [(0.0, 0.0); 5]);
        let (mut gps, mut gpb) = ([(0.0, 0.0); 5], [(0.0, 0.0); 5]);
        let dxs = ScalarTrainerBackend::step_kautz_backward_frame(
            -0.6,
            0.35,
            &poles,
            &kc,
            &ors,
            &ois,
            Some(&[0.05; 5]),
            Some(&[-0.03; 5]),
            &mut dsrs,
            &mut dsis,
            &mut gkcs,
            &mut gps,
        );
        let dxb = B::step_kautz_backward_frame(
            -0.6,
            0.35,
            &poles,
            &kc,
            &orb,
            &oib,
            Some(&[0.05; 5]),
            Some(&[-0.03; 5]),
            &mut dsrb,
            &mut dsib,
            &mut gkcb,
            &mut gpb,
        );
        assert_eq!(dxs.to_bits(), dxb.to_bits());
        for m in 0..5 {
            assert_eq!(gps[m].0.to_bits(), gpb[m].0.to_bits());
            assert_eq!(gps[m].1.to_bits(), gpb[m].1.to_bits());
            assert_eq!(gkcs[m].0.to_bits(), gkcb[m].0.to_bits());
            assert_eq!(gkcs[m].1.to_bits(), gkcb[m].1.to_bits());
        }

        let phi_w: Vec<Vec<f64>> = (0..11)
            .map(|c| {
                (0..23)
                    .map(|t| (((t + c * 3) as f64) * 0.17).sin())
                    .collect()
            })
            .collect();
        let y_w: Vec<f64> = (0..23).map(|t| ((t as f64) * 0.23).cos()).collect();
        let mut gs = vec![0.0_f64; uplo_u_packed_len(11)];
        let mut gb = vec![0.0_f64; uplo_u_packed_len(11)];
        let mut rhss = vec![[0.0_f64; 2]; 11];
        let mut rhsb = vec![[0.0_f64; 2]; 11];
        let ds = ScalarTrainerBackend::assemble_uplo_gram_and_rhs(&phi_w, &y_w, &mut gs, &mut rhss);
        let db = B::assemble_uplo_gram_and_rhs(&phi_w, &y_w, &mut gb, &mut rhsb);
        assert!((ds - db).abs() <= 1.0e-11);
        assert!(gs.iter().zip(&gb).all(|(a, b)| (a - b).abs() <= 1.0e-11));
        assert!(
            rhss.iter()
                .zip(&rhsb)
                .all(|(a, b)| (a[0] - b[0]).abs() <= 1.0e-11)
        );
    }

    #[test]
    fn test_trainer_backend_hierarchy_and_dynamic_dispatch() {
        struct VerifyTask;
        impl TrainerTask<BackendKind> for VerifyTask {
            fn run<B: TrainerBackend>(self) -> BackendKind {
                assert_eq!(B::name(), B::backend_kind().name());
                verify_gabor_kautz_gram_parity::<B>();
                B::backend_kind()
            }
        }
        let det = DynamicTrainer::default();
        assert_eq!(det.name(), det.kind().name());
        assert_eq!(det.dispatch(VerifyTask), det.kind());
        assert_eq!(
            DynamicTrainer::scalar().dispatch(VerifyTask),
            BackendKind::Scalar
        );
        assert_eq!(
            DynamicTrainer::from_kind(BackendKind::Scalar).dispatch(VerifyTask),
            BackendKind::Scalar
        );
        assert_eq!(
            DefaultTrainerBackend::name(),
            DefaultTrainerBackend::backend_kind().name()
        );
        verify_gabor_kautz_gram_parity::<DefaultTrainerBackend>();
        #[cfg(target_arch = "aarch64")]
        {
            verify_gabor_kautz_gram_parity::<NeonTrainerBackend>();
            verify_gabor_kautz_gram_parity::<ArmSve2TrainerBackend>();
            verify_gabor_kautz_gram_parity::<ArmSmeTrainerBackend>();
            assert_eq!(
                DynamicTrainer::from_kind(BackendKind::ArmSve2).dispatch(VerifyTask),
                BackendKind::ArmSve2
            );
            assert_eq!(
                DynamicTrainer::from_kind(BackendKind::ArmSme).dispatch(VerifyTask),
                BackendKind::ArmSme
            );
        }
    }

    proptest! {
        #[test]
        fn prop_capture_backends_gram_tile_and_omp_parity(
            xs in prop::collection::vec(-5.0_f64..5.0_f64, 16..48),
            ys in prop::collection::vec(-5.0_f64..5.0_f64, 16..48),
        ) {
            let n = xs.len().min(ys.len());
            let (mut re_s, mut im_s) = (xs[..n].to_vec(), ys[..n].to_vec());
            let (mut re_d, mut im_d) = (re_s.clone(), im_s.clone());
            if DefaultTrainerBackend::sum_sq_f64(&re_s) > 1.0e-6
                && DefaultTrainerBackend::sum_sq_f64(&im_s) > 1.0e-6
            {
                ScalarTrainerBackend::orthonormalize_pole_pair(&mut re_s, &mut im_s);
                DefaultTrainerBackend::orthonormalize_pole_pair(&mut re_d, &mut im_d);
                let sc_s = ScalarTrainerBackend::score_pole_pair(&xs[..n], &re_s, &im_s);
                let sc_d = DefaultTrainerBackend::score_pole_pair(&xs[..n], &re_d, &im_d);
                prop_assert!((sc_s - sc_d).abs() <= 1.0e-9 * sc_s.abs().max(1.0));
            }

            let frames = n / 4;
            if frames > 0 {
                let fi: Vec<[f64; 4]> = (0..frames).map(|k| [xs[k * 4], xs[k * 4 + 1], xs[k * 4 + 2], xs[k * 4 + 3]]).collect();
                let wj: Vec<[f64; 4]> = (0..frames).map(|k| [ys[k * 4], ys[k * 4 + 1], ys[k * 4 + 2], ys[k * 4 + 3]]).collect();
                let ts = ScalarTrainerBackend::contract_tile_4x4_ieee(&fi, &wj);
                let td = DefaultTrainerBackend::contract_tile_4x4_ieee(&fi, &wj);
                for r in 0..4 {
                    for c in 0..4 {
                        prop_assert_eq!(ts[r][c].to_bits(), td[r][c].to_bits());
                    }
                }
            }
        }
    }
}
