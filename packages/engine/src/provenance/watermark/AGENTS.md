# `src/provenance/watermark` — Buyer Watermarking Submodules

Implements Layer 2 pole-angle phase coding and Layer 3 `Role A = 0` `EMR` Tardos watermarking (`#[forbid(unsafe_code)]`).

## 1. Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`phase.rs`](phase.rs) | Layer 2 pole-angle phase coding and black-box `Z_phase` scoring | [`watermarking_and_marketplace.md`](../../../../../docs/specs/security/watermarking_and_marketplace.md) |
| [`tardos.rs`](tardos.rs) | Layer 3 two-seed `EMR` Tardos codebook, envelopes, and colluder scoring | [`watermarking_and_marketplace.md`](../../../../../docs/specs/security/watermarking_and_marketplace.md) |
