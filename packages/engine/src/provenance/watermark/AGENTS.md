# `src/provenance/watermark` — Buyer Watermarking Submodules

Implements Layer 2 pole-angle phase coding and Layer 3 `Role A = 0` `EMR` Tardos watermarking (`#[forbid(unsafe_code)]`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`sona_gauge.rs`](sona_gauge.rs) | Layer 1 `0-ULP` 40-lane `SONA` (`TPT-SVF`/`ParaLRU`/`PolyADAA1`) sign-flip gauge | [`watermarking_and_marketplace.md`](../../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`phase.rs`](phase.rs) | Layer 2 pole-angle phase coding and black-box `Z_phase` scoring | [`watermarking_and_marketplace.md`](../../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`tardos.rs`](tardos.rs) ([`layout_7680.rs`](tardos/layout_7680.rs)) | Layer 3 two-seed `EMR` Tardos codebook, envelopes, and colluder scoring | [`watermarking_and_marketplace.md`](../../../../../docs/specs/security/watermarking_and_marketplace.md) |
