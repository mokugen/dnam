//! Block-level runtime-dispatched DNAM engine (`DynamicDnamEngine<'a>`).

pub mod dispatch;

use dnam_vm::math::{BackendKind, ScalarBackend, detect_best_backend, warm_simd_feature_caches};

use crate::engine::DnamEngine;
use crate::engine::runtime::load_engine_with_options;
use crate::provenance::{CryptoProvider, DefaultCryptoProvider};
use crate::schema::DnamError;
use crate::types::Tier;

/// Runtime-dispatched wrapper around [`DnamEngine`] supporting block-level backend selection
/// ([`BackendKind`]) with **zero per-sample branching**.
#[derive(Debug)]
pub enum DynamicDnamEngine<'a> {
    /// Portable `#![no_std]` IEEE-754 scalar backend.
    Scalar(DnamEngine<'a, ScalarBackend>),
    /// ARM64 NEON `float32x4_t` backend.
    #[cfg(target_arch = "aarch64")]
    Neon(DnamEngine<'a, dnam_vm::math::NeonBackend>),
    /// ARMv9-A Scalable Vector Extension 2 (`ARM SVE2` / `FEAT_SVE2`) backend.
    #[cfg(target_arch = "aarch64")]
    ArmSve2(DnamEngine<'a, dnam_vm::math::ArmSve2Backend>),
    /// ARMv9.2-A Scalable Matrix Extension (`ARM SME` / `SME2`) backend.
    #[cfg(target_arch = "aarch64")]
    ArmSme(DnamEngine<'a, dnam_vm::math::ArmSmeBackend>),
    /// x86-64 SSE2 (`x86-64-v1`) backend.
    #[cfg(target_arch = "x86_64")]
    X86Sse2(DnamEngine<'a, dnam_vm::math::X86Sse2Backend>),
    /// x86-64 SSE4.2 (`x86-64-v2`) backend.
    #[cfg(target_arch = "x86_64")]
    X86Sse42(DnamEngine<'a, dnam_vm::math::X86Sse42Backend>),
    /// x86-64 AVX2 + FMA + F16C (`x86-64-v3`) backend.
    #[cfg(target_arch = "x86_64")]
    X86Avx2Fma(DnamEngine<'a, dnam_vm::math::X86Avx2FmaBackend>),
    /// Alias variant for x86-64 SIMD ([`dnam_vm::math::X86SimdBackend`]).
    #[cfg(target_arch = "x86_64")]
    X86Simd(DnamEngine<'a, dnam_vm::math::X86SimdBackend>),
    /// WebAssembly SIMD128 `v128` backend.
    #[cfg(target_arch = "wasm32")]
    WasmSimd128(DnamEngine<'a, dnam_vm::math::WasmSimd128Backend>),
}

impl<'a> DynamicDnamEngine<'a> {
    fn load_for_backend<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
        crypto: &C,
        preview_seed: Option<[u8; 32]>,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let allow_preview = preview_seed.is_some();
        warm_simd_feature_caches();
        match backend {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => Ok(DynamicDnamEngine::ArmSme(load_engine_with_options(
                bytes,
                tier,
                max_supported_tier,
                sample_rate_hz,
                crypto,
                allow_preview,
                preview_seed,
            )?)),
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSve2 => Ok(DynamicDnamEngine::ArmSve2(load_engine_with_options(
                bytes,
                tier,
                max_supported_tier,
                sample_rate_hz,
                crypto,
                allow_preview,
                preview_seed,
            )?)),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon | BackendKind::ArmNeon => {
                Ok(DynamicDnamEngine::Neon(load_engine_with_options(
                    bytes,
                    tier,
                    max_supported_tier,
                    sample_rate_hz,
                    crypto,
                    allow_preview,
                    preview_seed,
                )?))
            }
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse2 => Ok(DynamicDnamEngine::X86Sse2(load_engine_with_options(
                bytes,
                tier,
                max_supported_tier,
                sample_rate_hz,
                crypto,
                allow_preview,
                preview_seed,
            )?)),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse42 => Ok(DynamicDnamEngine::X86Sse42(load_engine_with_options(
                bytes,
                tier,
                max_supported_tier,
                sample_rate_hz,
                crypto,
                allow_preview,
                preview_seed,
            )?)),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Avx2Fma => Ok(DynamicDnamEngine::X86Avx2Fma(load_engine_with_options(
                bytes,
                tier,
                max_supported_tier,
                sample_rate_hz,
                crypto,
                allow_preview,
                preview_seed,
            )?)),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Simd => Ok(DynamicDnamEngine::X86Simd(load_engine_with_options(
                bytes,
                tier,
                max_supported_tier,
                sample_rate_hz,
                crypto,
                allow_preview,
                preview_seed,
            )?)),
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => {
                Ok(DynamicDnamEngine::WasmSimd128(load_engine_with_options(
                    bytes,
                    tier,
                    max_supported_tier,
                    sample_rate_hz,
                    crypto,
                    allow_preview,
                    preview_seed,
                )?))
            }
            _ => Ok(DynamicDnamEngine::Scalar(load_engine_with_options(
                bytes,
                tier,
                max_supported_tier,
                sample_rate_hz,
                crypto,
                allow_preview,
                preview_seed,
            )?)),
        }
    }

    /// Creates an empty pre-allocated [`DynamicDnamEngine`] with full reload headroom.
    pub fn with_max_tier(max_supported_tier: Tier, backend: BackendKind) -> Self {
        warm_simd_feature_caches();
        match backend {
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSme => Self::ArmSme(DnamEngine::with_max_tier(max_supported_tier)),
            #[cfg(target_arch = "aarch64")]
            BackendKind::ArmSve2 => Self::ArmSve2(DnamEngine::with_max_tier(max_supported_tier)),
            #[cfg(target_arch = "aarch64")]
            BackendKind::Neon | BackendKind::ArmNeon => {
                Self::Neon(DnamEngine::with_max_tier(max_supported_tier))
            }
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse2 => Self::X86Sse2(DnamEngine::with_max_tier(max_supported_tier)),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Sse42 => Self::X86Sse42(DnamEngine::with_max_tier(max_supported_tier)),
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Avx2Fma => {
                Self::X86Avx2Fma(DnamEngine::with_max_tier(max_supported_tier))
            }
            #[cfg(target_arch = "x86_64")]
            BackendKind::X86Simd => Self::X86Simd(DnamEngine::with_max_tier(max_supported_tier)),
            #[cfg(target_arch = "wasm32")]
            BackendKind::WasmSimd128 => {
                Self::WasmSimd128(DnamEngine::with_max_tier(max_supported_tier))
            }
            _ => Self::Scalar(DnamEngine::with_max_tier(max_supported_tier)),
        }
    }

    /// Creates an empty pre-allocated `Ultra` [`DynamicDnamEngine`] using runtime CPU detection.
    pub fn with_max_capacity() -> Self {
        Self::with_max_tier(Tier::Ultra, detect_best_backend())
    }

    /// Creates a [`DynamicDnamEngine`] using runtime CPU feature detection.
    pub fn from_bytes(bytes: &'a [u8], tier: Tier, sample_rate_hz: u32) -> Result<Self, DnamError> {
        Self::from_bytes_with_backend(bytes, tier, sample_rate_hz, detect_best_backend())
    }

    /// Creates a [`DynamicDnamEngine`] capped at `max_supported_tier` using runtime CPU feature detection.
    pub fn from_bytes_with_max_tier(
        bytes: &'a [u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::load_for_backend(
            bytes,
            tier,
            max_supported_tier,
            sample_rate_hz,
            detect_best_backend(),
            &crypto,
            None,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] using runtime CPU feature detection.
    pub fn from_bytes_owned(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_crypto(bytes, tier, sample_rate_hz, &crypto)
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] capped at `max_supported_tier`.
    pub fn from_bytes_owned_with_max_tier(
        bytes: &[u8],
        tier: Tier,
        max_supported_tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::load_for_backend(
            bytes,
            tier,
            max_supported_tier,
            sample_rate_hz,
            detect_best_backend(),
            &crypto,
            None,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] using a custom [`CryptoProvider`].
    pub fn from_bytes_owned_with_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        crypto: &C,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        Self::from_bytes_owned_with_backend_and_crypto(
            bytes,
            tier,
            sample_rate_hz,
            detect_best_backend(),
            crypto,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] with an explicit [`BackendKind`].
    pub fn from_bytes_owned_with_backend(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::from_bytes_owned_with_backend_and_crypto(
            bytes,
            tier,
            sample_rate_hz,
            backend,
            &crypto,
        )
    }

    /// Creates an owned `'static` [`DynamicDnamEngine`] with an explicit [`BackendKind`] and [`CryptoProvider`].
    pub fn from_bytes_owned_with_backend_and_crypto<C: CryptoProvider>(
        bytes: &[u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
        crypto: &C,
    ) -> Result<DynamicDnamEngine<'static>, DnamError> {
        Self::load_for_backend(
            bytes,
            tier,
            Tier::Ultra,
            sample_rate_hz,
            backend,
            crypto,
            None,
        )
    }

    /// Creates a [`DynamicDnamEngine`] with an explicit [`BackendKind`].
    pub fn from_bytes_with_backend(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        backend: BackendKind,
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::load_for_backend(
            bytes,
            tier,
            Tier::Ultra,
            sample_rate_hz,
            backend,
            &crypto,
            None,
        )
    }

    /// Loads a storefront `.dnam-prev` container (`Tier::Lite` + `PreviewGuardState`).
    pub fn from_preview_bytes(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
    ) -> Result<Self, DnamError> {
        let seed = crate::provenance::sha256_digest(bytes);
        Self::from_preview_bytes_with_seed(bytes, tier, sample_rate_hz, &seed)
    }

    /// Loads a storefront `.dnam-prev` container keyed by explicit 32-byte `seed`.
    pub fn from_preview_bytes_with_seed(
        bytes: &'a [u8],
        tier: Tier,
        sample_rate_hz: u32,
        seed: &[u8; 32],
    ) -> Result<Self, DnamError> {
        let crypto = DefaultCryptoProvider::default();
        Self::load_for_backend(
            bytes,
            tier,
            Tier::Lite,
            sample_rate_hz,
            detect_best_backend(),
            &crypto,
            Some(*seed),
        )
    }

    /// Returns the active [`BackendKind`].
    #[inline]
    pub const fn backend_kind(&self) -> BackendKind {
        match self {
            Self::Scalar(_) => BackendKind::Scalar,
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => BackendKind::Neon,
            #[cfg(target_arch = "aarch64")]
            Self::ArmSve2(_) => BackendKind::ArmSve2,
            #[cfg(target_arch = "aarch64")]
            Self::ArmSme(_) => BackendKind::ArmSme,
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse2(_) => BackendKind::X86Sse2,
            #[cfg(target_arch = "x86_64")]
            Self::X86Sse42(_) => BackendKind::X86Sse42,
            #[cfg(target_arch = "x86_64")]
            Self::X86Avx2Fma(_) => BackendKind::X86Avx2Fma,
            #[cfg(target_arch = "x86_64")]
            Self::X86Simd(_) => BackendKind::X86Simd,
            #[cfg(target_arch = "wasm32")]
            Self::WasmSimd128(_) => BackendKind::WasmSimd128,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::DnamJsonHeader;
    use crate::test_support::build_test_dnam_container;

    #[test]
    fn test_dynamic_engine_constructors() {
        let mut header = DnamJsonHeader::default_reference();
        let container = build_test_dnam_container(&mut header, Tier::Ultra, true);

        let e1 = DynamicDnamEngine::from_bytes(&container, Tier::Lite, 48_000).unwrap();
        let expected_backend = match detect_best_backend() {
            BackendKind::ArmNeon => BackendKind::Neon,
            backend => backend,
        };
        assert_eq!(e1.backend_kind(), expected_backend);

        let e2 =
            DynamicDnamEngine::from_bytes_with_max_tier(&container, Tier::Lite, Tier::Full, 48_000)
                .unwrap();
        assert!(e2.effective_tier() <= Tier::Full);

        let e3 = DynamicDnamEngine::from_bytes_owned(&container, Tier::Lite, 48_000).unwrap();
        assert_eq!(e3.effective_tier(), Tier::Lite);

        let e4 = DynamicDnamEngine::from_bytes_owned_with_max_tier(
            &container,
            Tier::Lite,
            Tier::Pro,
            48_000,
        )
        .unwrap();
        assert_eq!(e4.effective_tier(), Tier::Lite);

        let crypto = DefaultCryptoProvider::default();
        let e5 = DynamicDnamEngine::from_bytes_owned_with_crypto(
            &container,
            Tier::Lite,
            48_000,
            &crypto,
        )
        .unwrap();
        assert_eq!(e5.effective_tier(), Tier::Lite);

        let e6 = DynamicDnamEngine::from_bytes_owned_with_backend(
            &container,
            Tier::Lite,
            48_000,
            BackendKind::Scalar,
        )
        .unwrap();
        assert_eq!(e6.backend_kind(), BackendKind::Scalar);

        let e7 = DynamicDnamEngine::from_bytes_with_backend(
            &container,
            Tier::Lite,
            48_000,
            BackendKind::Scalar,
        )
        .unwrap();
        assert_eq!(e7.backend_kind(), BackendKind::Scalar);

        let mut prev_hdr = DnamJsonHeader::default_reference();
        prev_hdr.container_kind = alloc::string::String::from("preview");
        let prev_c = build_test_dnam_container(&mut prev_hdr, Tier::Lite, true);
        let prev_eng = DynamicDnamEngine::from_preview_bytes(&prev_c, Tier::Ultra, 48_000).unwrap();
        assert!(prev_eng.is_preview_mode());
        assert_eq!(prev_eng.effective_tier(), Tier::Lite);
    }
}
