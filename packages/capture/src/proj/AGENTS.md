# `packages/capture/src/proj/` — `.dnam-proj` (`DNPJ`) Container & Vault Guide

Implements the single-file `64 B`-aligned `v3.0` `.dnam-proj` (`DNPJ`) project container, `HTTP Range` manifest parser, `INPC`/`STMC`/`STAT` capsules, Two-Path Capture Vault, and `< 0.2 ms` runtime `.dnam` export.

## Submodule Map

| File | Responsibility | Canonical Spec (SSOT) |
| :--- | :--- | :--- |
| [`../proj.rs`](../proj.rs) | `DNPJ` magic/flags, `PROJ_STAT_BYTES_*`, `uplo='U'` pack/unpack, and RNE `fp16` slice helpers | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`schema.rs`](schema.rs) | `DnamProjManifest`, `ProjectMeta`, `RigRoutingConfig`, `StemRegistryEntry` (`enabled`), and JCS digest | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`directory.rs`](directory.rs) | `DnamProjPreamble`, `ProjChunkDirEntry` (`crc32c` + `sha256`), `parse_proj_manifest_range`, and `AUTH` | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`capsules.rs`](capsules.rs) | `CalbChunk64B`, `INPC` (`384,640 B`/`768,640 B`), `STMC` (`388,800 B`/`772,800 B`), and `STAT` (`uplo='U'`) | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`bundle.rs`](bundle.rs), [`dbin.rs`](bundle/dbin.rs), [`export.rs`](bundle/export.rs) | `DnamProjView`, `build_dnam_proj_container`, `[STAB-ZERO-DC]`, `compact_to_capsules`, and `export_runtime_dnam` | [`project_container_format.md`](../../../../docs/specs/containers/project_container_format.md) |
| [`vault.rs`](vault.rs), [`edit.rs`](vault/edit.rs) | Two-Path Capture Vault (`add_capture`, `remove_capture`, `replace_capture`, `set_capture_enabled`) | [`iterative_resume_and_caching.md`](../../../../docs/specs/training/iterative_resume_and_caching.md) |
