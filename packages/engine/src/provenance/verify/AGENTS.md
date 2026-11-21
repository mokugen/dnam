# `src/provenance/verify` — JCS Header Canonicalization & Provenance Verification

Implements RFC 8785 JSON Canonicalization Scheme (JCS), creator/marketplace signing transcripts, and `OemDeviceCertificate` verification (`#[forbid(unsafe_code)]`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`jcs.rs`](jcs.rs) | RFC 8785 JCS serializer, deterministic key sorting & ECMAScript number formatting | [`provenance_and_enclave_drm.md`](../../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`../verify.rs`](../verify.rs) | Schema header projection, creator/receipt verification & signature transcripts | [`provenance_and_enclave_drm.md`](../../../../../docs/specs/security/provenance_and_enclave_drm.md) |
