# `src/provenance/security` — `DAU1`, Prefix Commitments, Enclave DRM & Watermarking Specs

Implements v2 security verification, envelope encoding, and contract tests (`#[forbid(unsafe_code)]`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`prefix_and_dau1.rs`](prefix_and_dau1.rs) | `DAU1` 2-certificate offline auth, taxonomy, and policy enforcement | [`provenance_and_enclave_drm.md`](../../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`prefix_delegation_crypto.rs`](prefix_delegation_crypto.rs) | Real `Ed25519` prefix-delegation reference proofs | [`provenance_and_enclave_drm.md`](../../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`oem_and_drm.rs`](oem_and_drm.rs) | `OemDeviceCertificate`, HPKE contexts, `DNSL`, `DNPL`, and `DME1` framing | [`provenance_and_enclave_drm.md`](../../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`watermark_and_preview.rs`](watermark_and_preview.rs) | Watermark seed derivation, Tardos/phase contracts, and `.dnam-prev` preview containment | [`watermarking_and_marketplace.md`](../../../../../docs/specs/security/watermarking_and_marketplace.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for security invariants.
