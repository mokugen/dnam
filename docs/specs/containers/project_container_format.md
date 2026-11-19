---
id: spec.container.proj
title: "Single-File Cloud-Ready Project & Capture Vault Container (.dnam-proj / DNPJ)"
status: canonical
version: "3.0.0"
domain: containers
rule_prefixes:
  - "r[container.proj."
requirements:
  - REQ-3.1
  - REQ-3.2
  - REQ-3.3
  - REQ-5.3
  - REQ-8.1
  - REQ-8.2
  - REQ-8.8
adrs:
  - ADR-0020
  - ADR-0022
---

# Single-File Cloud-Ready Project & Capture Vault Container (`.dnam-proj` / `DNPJ`)

This specification defines the **`.dnam-proj` (`b"DNPJ"`, `v3.0`-Only, `"architecture": "sona"`) Project & Capture Vault Binary Format** built from declarative `#[repr(C)]` `zerocopy` wire structs (`ref[repo.code.zerocopy-wire-structs]`) with **zero legacy `v2` or `TORA` backwards compatibility**, owned by [`packages/capture/`](../../../packages/capture/README.md) (`dnam_capture::proj`) and pairing with [`dnam_binary_format.md`](dnam_binary_format.md) and `DNWB` white-box pipe headers (`DnwbHeader64BWire`, `version = 3`) to fulfill [`REQ-3.1`–`REQ-3.3`, `REQ-5.3`, `REQ-8.1`, `REQ-8.2`, `REQ-8.8`](../../requirements/system_goals_and_requirements.md) (see [`../../adr/0020-modular-monorepo-and-dnam-proj-container.md`](../../adr/0020-modular-monorepo-and-dnam-proj-container.md) and [`../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)).

---

## 1. Front-Loaded Cloud-Ready Wire Framing & Chunk Directory

```mermaid
stateDiagram-v2
    [*] --> Calibrated: Record CALB (64 B zerocopy)
    Calibrated --> Capturing: Append SPCM raw stems + 2.0-2.5s fp32 INPC/STMC capsules
    Capturing --> TrainedFull: Run WebGPU Scans + VarPro -> STAT + WGHT (+ AUTH)
    TrainedFull --> TrainedCompacted: compact_to_capsules() strips SPCM (< 1 ms)
    TrainedFull --> RuntimeExport: export_runtime_dnam() slices WGHT (< 0.2 ms)
    TrainedCompacted --> TrainedCompacted: Vault add/remove/replace/toggle (Two-Path)
    TrainedCompacted --> RuntimeExport: export_runtime_dnam() slices WGHT (< 0.2 ms)
```

```text
+===================================================================================================+
| .dnam-proj ("DNPJ", v3.0, "sona") FRONT-LOADED 64-BYTE-ALIGNED ZEROCOPY WIRE LAYOUT               |
+===================================================================================================+
| [0 .. 64)                  DnamProjPreambleWire (#[repr(C, align(64))], zerocopy LE)              |
|   [0..4]    magic = b"DNPJ"           | [4..6]   version_major = 3 (U16)| [6..8] version_minor = 0|
|   [8..12]   lifecycle_flags (U32 LE)  | [12..16] chunk_count (N, U32)   | [16..24] chunk_dir_off  |
|   [24..32]  manifest_offset (U64)     | [32..40] manifest_len (U64)     | [40..44] header_crc32c  |
|   [44..64]  reserved = [0u8; 20]      |                                                           |
+---------------------------------------------------------------------------------------------------+
| [64 .. 64 + 64*N)          Chunk Directory: N x ProjChunkDirEntryWire (64 B each, zerocopy)       |
|   [0..4]    kind FourCC ("CALB"|"SPCM"|"INPC"|"STMC"|"STAT"|"WGHT"|"AUTH")                        |
|   [4..8]    chunk_id (U32 LE)         | [8..12]  flags (U32 LE)       | [12..16] crc32c (U32 LE)  |
|   [16..24]  offset (64 B aligned U64) | [24..32] length (exact U64)   | [32..64] sha256 ([u8; 32])|
+---------------------------------------------------------------------------------------------------+
| [manifest_offset .. align64(manifest_offset + manifest_len))  RFC 8785 JCS Manifest + 0x00 pad    |
+---------------------------------------------------------------------------------------------------+
| [Payload Region: 64 B-aligned chunks in contiguous ascending file-offset order]                   |
|   • CALB (64 B CalbChunkWire)          Loopback latency, converter gain, noise-floor profile      |
|   • SPCM (optional raw PCM stems)      Active-session lossless audio (stripped on compaction)     |
|   • INPC (InpcCapsuleHeaderWire+fp32)  Shared 2.0-2.5 s (96k-120k) full-bandwidth fp32 dry capsule|
|   • STMC (StmcCapsuleHeaderWire+fp32)  128 B hdr + R<=36 StmcTrajectoryWindowWire + 4096 B + fp32 |
|   • STAT (cached trajectories + uploU) StarCenter SONA trajectories + f64 uplo='U' const Gram     |
|   • WGHT (canonical v3.0 .dnam bytes)  Complete [STAB-ZERO-DC] "sona" runtime model (< 0.2 ms)    |
|   • AUTH (128 B ProjAuthRecordWire)    Creator signature ("DNPJ-PROJECT-MANIFEST-V3")             |
+===================================================================================================+
```

### 1.1 64B Preamble & Content-Addressed Chunk Directory

`r[container.proj.preamble-and-directory]`

Every `v3.0` `.dnam-proj` container is a declarative, code-execution-free binary file (`REQ-8.8`) parsed via `#[repr(C)]` `zerocopy::byteorder::little_endian::{U16, U32, U64, F32, F64}` structs (`FromBytes, IntoBytes, KnownLayout, Immutable`, `ref[repo.code.zerocopy-wire-structs]`) with zero manual byte slicing, starting with `DnamProjPreambleWire` at byte `0` (`magic = b"DNPJ"`, `version_major = 3`, `version_minor = 0`; any `v1`/`v2` container is rejected fail-fast with `DnamError::UnsupportedVersion`):

1. **Front-Loaded `HTTP Range` Inspectability:** `chunk_dir_offset` must equal `64`, and `manifest_offset` must equal `64 + 64 * chunk_count`, placing the entire Preamble, `ProjChunkDirEntryWire` directory, and JCS Manifest at the front of the file so cloud storage clients (`HTTP Range: bytes=0-65535`) can parse all project metadata and chunk `SHA-256` digests in a single range read (`parse_proj_manifest_range`, or a second exact range read to `align64(manifest_offset + manifest_len)` if the header exceeds `64 KB`) without fetching payload chunks.
2. **Strict `64 B` Alignment & Bounds Validation:** The total file byte length, `manifest_offset`, `align64(manifest_offset + manifest_len)`, and every chunk `offset` must be exact multiples of `64 B`. `header_crc32c` (at bytes `40..44`) covers bytes `0..40` of the preamble concatenated with the `64 * chunk_count` directory bytes and the exact `manifest_len` unpadded JCS manifest bytes, while padding bytes `[manifest_offset + manifest_len .. align64(manifest_offset + manifest_len))` must be `0x00`. Every `ProjChunkDirEntryWire` must specify a known FourCC (`CALB`, `SPCM`, `INPC`, `STMC`, `STAT`, `WGHT`, or `AUTH`), contiguous non-overlapping ascending `64 B`-aligned file bounds, zeroed padding (`0x00`) between `offset + length` and the next `64 B` boundary, and matching `CRC32C` + `SHA-256` digests over its exact `length` bytes. Unknown `lifecycle_flags` bits, non-zero `reserved` bytes (`44..64`), truncated files, or digest mismatches fail fast with `DnamError`.

### 1.2 RFC 8785 JCS Project Manifest

`r[container.proj.jcs-manifest]`

The manifest slice at `[manifest_offset .. manifest_offset + manifest_len)` stores a strict UTF-8 **RFC 8785 JSON Canonicalization Scheme (`JCS`)** serialization of `DnamProjManifest` (zero-padded with `0x00` to the next `64 B` boundary), rejecting UTF-8 BOMs, duplicate JSON keys, unknown schema fields, and non-canonical key ordering:

| Manifest Section | Required Fields | Normative Invariant |
| :--- | :--- | :--- |
| **Envelope & Nested `base` Header** | `version = "3.0.0"`, `container_kind = "dnam-proj"`, `base` (`DnamJsonHeader` with `base.container_kind = "dnam"`, `base.architecture = "sona"`) | Shares the exact validated `v3.0` `"sona"` `DnamJsonHeader` schema with `dnam-engine` (`ref[container.dnam.jcs-header-schema]`). |
| **`project_meta`, `rig_routing` & `calibration_summary`** | `project_meta.{project_id, rig_notes, updated_at}`, `rig_routing.{sample_rate_device_hz, bit_depth, input_channels, output_channels, reamp_send_channel}`, `calibration_summary` | Tracks project identity, native hardware rate (`8..384 kHz`), bit depth (`int12`, `int16`, `int24`, `int32`, `float32`), and `CalbChunkWire` (`CALB` `v3`) telemetry. |
| **`stems_registry`** | Ordered `StemRegistryEntry` list (`stem_id`, `chunk_id`, `enabled`, `is_star_center`, `controls_meta`, `provenance`, `user_metadata`, `extracted_physics`) | Binds each stem's `chunk_id` (`u16` `STMC` `stem_index`), `SHA-256` dry/wet `<f4` PCM provenance, and `DNAM-GRAM-INPUT-V3` fingerprint (`REQ-3.2`). |
| **`audit_report` & `solver_telemetry`** | `audit_report` (`AuditReportSummary`), `solver_telemetry.{active_stem_count, total_solve_ms, tier_esr_db}` | Persists Phase-1 control-space audit clamps and 4-tier ESR certification metrics across incremental vault edits. |

### 1.3 Lifecycle States, Capsule Compaction & Cloud Range Sync

`r[container.proj.lifecycle-states]`

A `.dnam-proj` container transitions monotonically or non-destructively across valid flag-consistent lifecycle configurations (`PROJ_FLAG_HAS_CALIBRATION`, `PROJ_FLAG_HAS_RAW_STEMS`, `PROJ_FLAG_HAS_CAPSULES`, `PROJ_FLAG_COMPACTED_CAPSULES`, `PROJ_FLAG_HAS_STAT`, `PROJ_FLAG_HAS_WGHT`, `PROJ_FLAG_HAS_AUTH`, `PROJ_FLAG_STEREO_OUT`):

1. **Active Capture (`CALB` + `SPCM` + `INPC`/`STMC`):** Retains lossless raw `SPCM` audio stems alongside extracted `INPC`/`STMC` capsules during active hardware recording.
2. **Trained Full Project (`+ STAT + WGHT`):** Embeds `StarCenter` `uplo='U'` sufficient statistics (`STAT`) and the certified 4-tier `v3.0` `.dnam` runtime model (`WGHT`).
3. **Capsule Compaction (`compact_to_capsules`):** Requires `INPC` and `STMC` capsules whenever `stems_registry` is non-empty, strips all raw `SPCM` chunks in place without re-running solver passes, preserves `PROJ_FLAG_COMPACTED_CAPSULES` across capsule-only vault edits, and shrinks multi-stem projects to compact capsules (`~0.8–2.5 MB`) that still support deterministic full re-pooling and fast runtime `.dnam` export (`REQ-3.1`, `REQ-3.3`).

---

## 2. Capture Capsules & Compositional Sufficient Statistics

### 2.1 Shared Input Coreset Capsule (`INPC`)

`r[container.proj.inpc-capsule]`

The `INPC` chunk (`chunk_id = 0`) stores the shared **`2.0–2.5 s` (`96,000–120,000`-frame at `48 kHz`) full-bandwidth (`20 Hz–20 kHz`) `fp32` (`IEEE-754 binary32`)** dry excitation capsule inside a `64 B`-aligned record (`~384–480 KiB` for mono `C_in = 1`, `~768–960 KiB` for stereo `C_in = 2`), prefixed by the `640 B` `zerocopy` `InpcCapsuleHeaderWire` (`magic = b"INPC"`, `version = 3`, `channels`, `coreset_frames`, `header_crc32c` at `24..28`, `256 B` dry spectral envelope, and `320 B` transient/comb leverage index), retaining full `18–20 kHz` HF air and sub-second/multi-second dynamic envelopes inside `< 20 MB` multi-stem `.dnam-proj` vaults (`REQ-5.4`).

### 2.2 Per-Stem Response Coreset Capsule (`STMC`)

`r[container.proj.stmc-capsule]`

Each stem $s$ in `stems_registry` owns a dedicated `STMC` chunk (`chunk_id == stem_index`) of `64 B`-aligned size (`~389–485 KiB` for mono `C_out = 1`, `~773–965 KiB` for stereo `C_out = 2`, keeping compacted multi-stem projects strictly `< 20 MB` per `REQ-5.4`) containing:
1. **`128 B` `StmcCapsuleHeaderWire` (`magic = b"STMC"`, `version = 3`, `zerocopy`):** `stem_index` (`U16`, matching `StemRegistryEntry::chunk_id`), `flags` (`STMC_FLAG_ENABLED`, `STMC_FLAG_STAR_CENTER`, `STMC_FLAG_MODE_A_SWEEP`, `STMC_FLAG_STEREO`, verified in strict bijection with `stems_registry`), `window_count` (`1..=36`), `pad_gain_f32` (`F32`), `fractional_lag_samples_f32` (`F32`), `88 B` `uplo='U'` band-energy PSD summary (`24..112`), `n_comb_bins` (`112..116`), `coreset_frames` (`96,000..=120,000` at `48 kHz`), and `capsule_crc32c` (`120..124`).
2. **`576 B` Trajectory Window Table (`128..704`) & `4,096 B` Spectral Summary (`704..4,800`):** Up to `R <= 36` `16 B` `StmcTrajectoryWindowWire` records (`control_id`, `ctrl_kind`, `active_opt`, `u_phys`, `window_weight`, `emr_velocity`) driving Phase-1 range/switch unlocking and `CP1` Newton-Lobatto assembly, followed by a `4,096 B` harmonic/sideband/ballistic summary block.
3. **Full-Bandwidth `fp32` Aligned Wet Capsule (`4,800..`):** `96,000–120,000`-frame `f32` phase-aligned full-bandwidth wet response paired sample-for-sample with `INPC`, enabling lossless raw-audio-free `Path 2` WebGPU/CPU re-pooling without source WAV files.

### 2.3 Tier-Ordered Upper-Triangular `StarCenter` Statistics (`STAT`)

`r[container.proj.stat-uplo-u]`

The `STAT` chunk stores cached `StarCenter` Stage B `SONA` trajectories paired with fixed compile-time `const` (`LiteSpec..UltraSpec`, `0` capture-time resizing) tier-ordered column-major upper-triangular (`uplo = 'U'`) `IEEE-754 float64` `VarPro` sufficient statistics (`Lite` $\subset$ `Full` $\subset$ `Pro` $\subset$ `Ultra`, `REQ-5.3`):
$$\mathrm{idx}(i, j) = i + \frac{j(j + 1)}{2}, \qquad 0 \le i \le j < D_T$$
At active tier readout dimension $D_T$ (constant per `SonaTierSpec`), `STAT` packs $\frac{D_T(D_T + 1)}{2}$ upper-triangular `f64` Gram entries ($\mathbf{G}_0$), $2 \times D_T$ `f64` two-channel cross-correlation entries ($\mathbf{C}_0$), and the cached `StarCenter` trajectory prefix via `zerocopy` slice views, verified on load for exact byte alignment, finite `f64`/`f32` values, and non-negative diagonal entries $G_{ii} \ge 0$, guaranteeing `< 50 ms` `Path 1` incremental Vault edits with `0-ULP` `StarCenter` invariance.

---

## 3. Embedded Runtime Model & Provenance Envelope

### 3.1 Embedded Runtime Weights & Zero-Solve `.dnam` Export (`WGHT`)

`r[container.proj.wght-and-export]`

Every trained `v3.0` `.dnam-proj` embeds a complete, self-contained `v3.0` (`"architecture": "sona"`) `.dnam` container inside its `WGHT` chunk (`ref[container.dnam.open-layout]`):
1. **`[STAB-ZERO-DC]` Hard Invariant (`b_head == 0.0`):** Both container assembly (`build_dnam_proj_container_with_cp1`) and `DnamProjView::from_bytes` verify that all output-head DC bias slots across `Arena0LiteWire` (`Lite`) through `Arena3UltraDeltaWire` (`Ultra`) are strictly `0.0` (`b_head = 0`), rejecting non-zero DC bias payloads immediately.
2. **Zero-Solve Runtime Export (`export_runtime_dnam`, `< 0.2 ms`):** Exporting a deployable `v3.0` `.dnam` file at any target `max_tier` (`Lite`, `Full`, `Pro`, or `Ultra`) validates `DistributionPolicy`, merges mutable `presets`, `user_metadata`, and `controls_schema[*].user_metadata` into the `DMU1` suffix (automatically growing compact `48 B` `DMU1` frames up to the `4,096 B` `DMU1` + `DMUE` trailer ceiling), and applies lossless keyless prefix truncation (`strip_container_to_tier`, `ref[container.dnam.tier-prefix]`, preserving `Arena 0` control descriptors and `Lite` `CP1`/`W_readout` slices) with zero solver recomputation.

### 3.2 Cryptographic Provenance Envelope (`AUTH`)

`r[container.proj.auth-signature]`

When signed (`PROJ_FLAG_HAS_AUTH`), the `AUTH` chunk stores a `128 B` (`2`-cache-line) `zerocopy` `ProjAuthRecordWire` (`magic = b"AUTH"`, `version = 3`, `signer_pubkey_ed25519: [u8; 32]` at `32..64`, `signature_ed25519: [u8; 64]` at `64..128`) containing an `Ed25519` signature over the domain-separated message (`b"DNPJ-PROJECT-MANIFEST-V3"` + immutable JCS `DnamProjManifest` `SHA-256` + ordered non-`AUTH` `ProjChunkDirEntryWire` `(kind, chunk_id, length, sha256)` tuples). Any tampering with the immutable manifest fields, chunk directory, `STMC` stem capsules, `STAT` Gram entries, or `WGHT` weights fails fast with `DnamError::CryptoVerificationFailed`.
