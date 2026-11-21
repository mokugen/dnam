# `src/provenance` — Marketplace Signatures, Hardware Enclaves & Traitor Tracing

Implements cryptographic provenance, hardware enclave DRM, and forensic watermarking (`#[forbid(unsafe_code)]` on all submodules except `#[cfg(target_arch = "aarch64")]` `sha256_hw.rs`).

**Canonical Specs (SSOT):**
[`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) &middot; [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md)

## 1. Submodule Map

| File | Responsibility | Spec Scope |
| :--- | :--- | :--- |
| [`security.rs`](security.rs) ([`security/AGENTS.md`](security/AGENTS.md)) | `DAU1`, prefix commitments, `OemDeviceCertificate`, HPKE, `DME1` & `.dnam-prev` | Provenance §2–§4, Watermark §1–§3 |
| [`provider.rs`](provider.rs) | `CryptoProvider`, `OpenArenaCryptoProvider`, `RustCryptoProvider`, `VerificationStatus` | Provenance §1–§3 |
| [`verify.rs`](verify.rs) ([`verify/AGENTS.md`](verify/AGENTS.md)) | JCS header projection & signature provenance verification | Provenance §1–§2 |
| [`seal.rs`](seal.rs) ([`seal/AGENTS.md`](seal/AGENTS.md)) | `.dnam-seal` (`DNSL`) HPKE + AEAD outer envelope sealing and unsealing | Provenance §3 |
| [`chacha20.rs`](chacha20.rs) | `#![no_std]` RFC 8439 `ChaCha20` 64B block stream cipher + `proptest` | Provenance §3 |
| [`sha256.rs`](sha256.rs) ([`sha256_hw.rs`](sha256_hw.rs)) | `#![no_std]` FIPS 180-4 SHA-256 (optional AArch64 hardware SHA2) & hex codec | Provenance §2–§3 |
| [`preview_guard.rs`](preview_guard.rs) | RT division-free `.dnam-prev` probe guard & micro-echo watermark (`#[forbid(unsafe_code)]`) | Watermark §3.1 |
| [`watermark.rs`](watermark.rs) ([`watermark/AGENTS.md`](watermark/AGENTS.md)) | Layer 1/2/3 gauges, SONA sign-flip, phase, Tardos & buyer `dbin_prefix` re-frame | Watermark §1 |
| [`fingerprint.rs`](fingerprint.rs) | 224-D structural hash (`1e-2 rad` pole grid, `f32le` `TransportQuad`) & similarity | Watermark §2.1 |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for JCS canonicalization and preset-invariance rules.
