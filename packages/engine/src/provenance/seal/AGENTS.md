# `src/provenance/seal` — RFC 9180 Base HPKE & Canonical `DNSL` Sealing

Implements RFC 9180 Base HPKE (`DHKEM(X25519, HKDF-SHA256)` + `ChaCha20Poly1305`) and v3.0 `.dnam-seal` (`b"DNSL"` + `DNPL` + `DAI1` + `DSA1` + `DME1`) encoding and unsealing (`#[forbid(unsafe_code)]`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`hpke_base.rs`](hpke_base.rs) | Audited RFC 9180 Base HPKE seal/open and ChaCha20-Poly1305 record AEAD | [`provenance_and_enclave_drm.md`](../../../../../docs/specs/security/provenance_and_enclave_drm.md) |
| [`dnsl.rs`](dnsl.rs) | `Dsa1ManifestHeaderWire`, `UnsealedDnslInner`, facade over [`seal_encode.rs`](dnsl/seal_encode.rs), [`seal_unseal.rs`](dnsl/seal_unseal.rs) & [`unsealed_dnam.rs`](dnsl/unsealed_dnam.rs) | [`provenance_and_enclave_drm.md`](../../../../../docs/specs/security/provenance_and_enclave_drm.md) |

See [`packages/engine/CONTRIBUTING.md`](../../../CONTRIBUTING.md) for enclave DRM and AEAD invariants.
