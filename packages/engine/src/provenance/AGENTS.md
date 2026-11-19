# `src/provenance` — Marketplace Signatures, Hardware Enclaves & Traitor Tracing

Implements cryptographic provenance, hardware enclave DRM, and forensic watermarking (`#[forbid(unsafe_code)]`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`security.rs`](security.rs) ([`security/AGENTS.md`](security/AGENTS.md)) | `DAU1`, prefix commitments, `OemDeviceCertificate`, HPKE, `DME1`, watermark & `.dnam-prev` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md), [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`provider.rs`](provider.rs) | `CryptoProvider` trait, `OpenArenaCryptoProvider`, `RustCryptoProvider`, `VerificationStatus` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`verify.rs`](verify.rs) ([`jcs.rs`](verify/jcs.rs)) | JCS header projection, creator/marketplace signatures, `verify_container_provenance` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`seal.rs`](seal.rs) | `.dnam-seal` (`DNSL`) outer envelope encryption, unsealing, and OEM device certs | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`chacha20.rs`](chacha20.rs) | `#![no_std]` RFC 8439 `ChaCha20` 64B block stream cipher + `proptest` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`sha256.rs`](sha256.rs) | `#![no_std]` FIPS 180-4 `SHA-256` & constant-time hex codec + `proptest` | [`provenance_and_enclave_drm.md`](../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`watermark.rs`](watermark.rs) ([`watermark/AGENTS.md`](watermark/AGENTS.md)) | Layer 1/2/3 bilinear gauge, GRU sign-flip gauge, phase coding, and Tardos watermarking | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`fingerprint.rs`](fingerprint.rs) | 224-D structural hash (`1e-2 rad` pole grid) & cosine similarity | [`watermarking_and_marketplace.md`](../../../../docs/specs/security/watermarking_and_marketplace.md) |

See [`packages/engine/CONTRIBUTING.md`](../../CONTRIBUTING.md) for JCS canonicalization and preset-invariance rules.
