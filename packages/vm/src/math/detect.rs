//! Lock-free cached CPU SIMD capability detection (`BackendKind`, `CpuSimdLevel`,
//! `CpuSimdFeatures`, `warm_simd_feature_caches`, `detect_cpu_simd`, `detect_cpu_simd_features`).

/// Compile-time optimal default `MathBackend` for `aarch64`.
#[cfg(target_arch = "aarch64")]
pub type DefaultMathBackend = super::aarch64::NeonBackend;

/// Compile-time optimal default `MathBackend` for `x86_64`.
#[cfg(target_arch = "x86_64")]
pub type DefaultMathBackend = super::x86::X86SimdBackend;

/// Compile-time optimal default `MathBackend` for `wasm32`.
#[cfg(target_arch = "wasm32")]
pub type DefaultMathBackend = super::wasm::WasmSimd128Backend;

/// Compile-time fallback `MathBackend` for portable architectures.
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
pub type DefaultMathBackend = super::scalar::ScalarBackend;

/// Hardware SIMD ISA capability tier selected during cold setup (`ref[sdk.rust.explicit-simd-and-coverage]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuSimdLevel {
    /// Portable scalar fallback.
    Scalar,
    /// ARM64 baseline NEON (`128-bit`).
    ArmNeon,
    /// ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`).
    ArmSve2,
    /// ARMv9.2-A Scalable Matrix Extension (`FEAT_SME` / `SME2`).
    ArmSme,
    /// x86-64 baseline `SSE2`.
    X86Sse2,
    /// x86-64 `SSE3` + `SSSE3` + `SSE4.1` + `SSE4.2` + hardware `CRC-32C`.
    X86Sse42,
    /// x86-64 `AVX` + `AVX2` + `FMA` + `F16C`.
    X86Avx2Fma,
    /// WebAssembly `128-bit` `simd128`.
    WasmSimd128,
}

/// Snapshot of runtime-detected SIMD and hardware CRC-32C capabilities (`0` hot-path syscalls).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuSimdFeatures {
    /// Selected highest SIMD tier.
    pub level: CpuSimdLevel,
    /// Mapped [`BackendKind`] identifier.
    pub backend: BackendKind,
    /// Hardware Castagnoli `CRC-32C` instruction availability (`FEAT_CRC32` or `SSE4.2`).
    pub hw_crc32c: bool,
    /// Hardware `FEAT_SVE2` availability (`aarch64`).
    pub sve2: bool,
    /// Hardware `FEAT_SME` availability (`aarch64`).
    pub sme: bool,
    /// Hardware `AVX2 + FMA + F16C` availability (`x86_64`).
    pub avx2_fma: bool,
    /// WebAssembly `simd128` compilation flag (`wasm32`).
    pub wasm_simd128: bool,
}

/// Identifier for available hardware `f32` math backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Portable 16-byte aligned scalar reference backend.
    Scalar,
    /// ARM64 128-bit NEON (`float32x4_t`, `vfmaq_f32`, hardware `CRC-32C`).
    ArmNeon,
    /// Alias for [`Self::ArmNeon`].
    Neon,
    /// ARMv9-A Scalable Vector Extension 2 (`FEAT_SVE2`).
    ArmSve2,
    /// ARMv9.2-A Scalable Matrix Extension (`FEAT_SME`).
    ArmSme,
    /// x86-64 SSE2 architectural baseline (`x86-64-v1`).
    X86Sse2,
    /// x86-64 SSE4.2 (`x86-64-v2` + `_mm_crc32_u64`).
    X86Sse42,
    /// x86-64 AVX2 + FMA + F16C (`x86-64-v3`).
    X86Avx2Fma,
    /// Alias for [`Self::X86Avx2Fma`].
    X86Simd,
    /// WebAssembly 128-bit SIMD (`v128` `f32x4`).
    WasmSimd128,
}

impl BackendKind {
    /// Returns a concise human-readable name for benchmark tables and telemetry.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::ArmNeon | Self::Neon => "ARM64 NEON",
            Self::ArmSve2 => "ARMv9 SVE2",
            Self::ArmSme => "ARM SME",
            Self::X86Sse2 => "x86 SSE2",
            Self::X86Sse42 => "x86 SSE4.2",
            Self::X86Avx2Fma | Self::X86Simd => "x86 AVX2",
            Self::WasmSimd128 => "WASM SIMD",
        }
    }

    /// Maps a [`BackendKind`] to its normalized [`CpuSimdLevel`].
    #[inline]
    pub const fn simd_level(self) -> CpuSimdLevel {
        match self {
            Self::Scalar => CpuSimdLevel::Scalar,
            Self::ArmNeon | Self::Neon => CpuSimdLevel::ArmNeon,
            Self::ArmSve2 => CpuSimdLevel::ArmSve2,
            Self::ArmSme => CpuSimdLevel::ArmSme,
            Self::X86Sse2 => CpuSimdLevel::X86Sse2,
            Self::X86Sse42 => CpuSimdLevel::X86Sse42,
            Self::X86Avx2Fma | Self::X86Simd => CpuSimdLevel::X86Avx2Fma,
            Self::WasmSimd128 => CpuSimdLevel::WasmSimd128,
        }
    }
}

/// Pre-warms all architecture-specific runtime CPU SIMD and CRC-32C feature caches outside the
/// real-time audio path so hot vector kernels execute only relaxed atomic flag loads.
#[inline]
pub fn warm_simd_feature_caches() {
    #[cfg(target_arch = "aarch64")]
    super::aarch64::warm_aarch64_feature_caches();
    #[cfg(target_arch = "x86_64")]
    super::x86::warm_x86_feature_caches();
}

/// Detects the highest-performance hardware `f32` math backend supported at runtime.
#[inline]
pub fn detect_best_backend() -> BackendKind {
    warm_simd_feature_caches();
    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::detect_aarch64_backend()
    }
    #[cfg(target_arch = "x86_64")]
    {
        super::x86::detect_x86_backend()
    }
    #[cfg(target_arch = "wasm32")]
    {
        super::wasm::detect_wasm_backend()
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )))]
    {
        BackendKind::Scalar
    }
}

/// Returns the detected [`CpuSimdLevel`] after warming lock-free capability caches.
#[inline]
pub fn detect_cpu_simd() -> CpuSimdLevel {
    detect_best_backend().simd_level()
}

/// Returns a complete [`CpuSimdFeatures`] snapshot for the active host target.
#[inline]
pub fn detect_cpu_simd_features() -> CpuSimdFeatures {
    let backend = detect_best_backend();
    let level = backend.simd_level();
    #[cfg(target_arch = "aarch64")]
    {
        CpuSimdFeatures {
            level,
            backend,
            hw_crc32c: super::aarch64::neon::is_crc32_supported_cached(),
            sve2: super::aarch64::ArmSve2Backend::is_supported_cached(),
            sme: super::aarch64::ArmSmeBackend::is_supported_cached(),
            avx2_fma: false,
            wasm_simd128: false,
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        CpuSimdFeatures {
            level,
            backend,
            hw_crc32c: super::x86::X86Sse42Backend::is_supported_cached(),
            sve2: false,
            sme: false,
            avx2_fma: super::x86::X86Avx2FmaBackend::is_supported_cached(),
            wasm_simd128: false,
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let has_simd = cfg!(target_feature = "simd128");
        CpuSimdFeatures {
            level,
            backend,
            hw_crc32c: false,
            sve2: false,
            sme: false,
            avx2_fma: false,
            wasm_simd128: has_simd,
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        target_arch = "wasm32"
    )))]
    {
        CpuSimdFeatures {
            level,
            backend,
            hw_crc32c: false,
            sve2: false,
            sme: false,
            avx2_fma: false,
            wasm_simd128: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_kind_names_and_feature_detection() {
        let all_kinds = [
            (BackendKind::Scalar, "Scalar", CpuSimdLevel::Scalar),
            (BackendKind::ArmNeon, "ARM64 NEON", CpuSimdLevel::ArmNeon),
            (BackendKind::Neon, "ARM64 NEON", CpuSimdLevel::ArmNeon),
            (BackendKind::ArmSve2, "ARMv9 SVE2", CpuSimdLevel::ArmSve2),
            (BackendKind::ArmSme, "ARM SME", CpuSimdLevel::ArmSme),
            (BackendKind::X86Sse2, "x86 SSE2", CpuSimdLevel::X86Sse2),
            (BackendKind::X86Sse42, "x86 SSE4.2", CpuSimdLevel::X86Sse42),
            (
                BackendKind::X86Avx2Fma,
                "x86 AVX2",
                CpuSimdLevel::X86Avx2Fma,
            ),
            (BackendKind::X86Simd, "x86 AVX2", CpuSimdLevel::X86Avx2Fma),
            (
                BackendKind::WasmSimd128,
                "WASM SIMD",
                CpuSimdLevel::WasmSimd128,
            ),
        ];
        for (kind, expected_name, expected_level) in all_kinds {
            assert_eq!(kind.name(), expected_name);
            assert_eq!(kind.simd_level(), expected_level);
        }
        let detected = detect_best_backend();
        assert!(!detected.name().is_empty());
        assert_eq!(detect_cpu_simd(), detected.simd_level());
        let feats = detect_cpu_simd_features();
        assert_eq!(feats.backend, detected);
        assert_eq!(feats.level, detected.simd_level());
    }
}
