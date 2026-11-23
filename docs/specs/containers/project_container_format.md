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

This specification defines the `.dnam-proj` (`"DNPJ"`, v3.0, `"architecture": "sona"`) Project & Capture Vault Binary Format using packed little-endian 64 B-aligned binary records.

It pairs with [dnam_binary_format.md](dnam_binary_format.md) and `DNWB` white-box pipe headers (64-byte `DNWB` Header, wire framing version 1) to fulfill [REQ-3.1–REQ-3.3, REQ-5.3, REQ-8.1, REQ-8.2, REQ-8.8](../../requirements/system_goals_and_requirements.md), [ADR-0020](../../adr/0020-modular-monorepo-and-dnam-proj-container.md), and [ADR-0022](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md).

---

## 1. Front-Loaded Cloud-Ready Wire Framing & Chunk Directory

```mermaid
stateDiagram-v2
    [*] --> Calibrated: Record CALB (64 B aligned)
    Calibrated --> Capturing: Append SPCM raw stems + 2.0-2.5s fp32 INPC/STMC capsules
    Capturing --> TrainedFull: Run WebGPU Scans + VarPro -> STAT + WGHT (+ AUTH)
    TrainedFull --> TrainedCompacted: Capsule compaction strips SPCM (< 1 ms)
    TrainedFull --> RuntimeExport: Runtime export slices WGHT (< 0.2 ms)
    TrainedCompacted --> TrainedCompacted: Vault add/remove/replace/toggle (Two-Path)
    TrainedCompacted --> RuntimeExport: Runtime export slices WGHT (< 0.2 ms)
```

```text
+===================================================================================================+
| .dnam-proj ("DNPJ", v3.0, "sona") FRONT-LOADED 64-BYTE-ALIGNED BINARY LAYOUT                      |
+===================================================================================================+
| [0 .. 64)                  Project Container Preamble (64 B, 64-byte aligned, little-endian)        |
|   [0..4]    magic = "DNPJ"            | [4..6]   version_major = 3 (u16le)| [6..8] version_minor=0|
|   [8..12]   lifecycle_flags (u32le)   | [12..16] chunk_count (N, u32le)   | [16..24] chunk_dir_off|
|   [24..32]  manifest_offset (u64le)   | [32..40] manifest_len (u64le)     | [40..44] header_crc32c|
|   [44..64]  reserved = 20 zero bytes  |                                                           |
+---------------------------------------------------------------------------------------------------+
| [64 .. 64 + 64*N)          Chunk Directory: N x 64-byte Directory Entry (little-endian)           |
|   [0..4]    kind FourCC ("CALB"|"SPCM"|"INPC"|"STMC"|"STAT"|"WGHT"|"AUTH")                        |
|   [4..8]    chunk_id (u32le)          | [8..12]  flags (u32le)          | [12..16] crc32c (u32le) |
|   [16..24]  offset (64 B align u64le) | [24..32] length (exact u64le)   | [32..64] sha256 (32 B)  |
+---------------------------------------------------------------------------------------------------+
| [manifest_offset .. align64(manifest_offset + manifest_len))  RFC 8785 JCS Manifest + 0x00 pad    |
+---------------------------------------------------------------------------------------------------+
| [Payload Region: 64 B-aligned chunks in contiguous ascending file-offset order]                   |
|   • CALB (64 B Calibration Record)     Loopback latency, converter gain, noise-floor profile      |
|   • SPCM (optional raw PCM stems)      Active-session lossless audio (stripped on compaction)     |
|   • INPC (640 B INPC Header + fp32)    Shared 2.0-2.5 s (96k-120k) full-bandwidth fp32 dry capsule|
|   • STMC (128 B STMC Header + fp32)    128 B hdr + R<=36 16 B Trajectory Windows + 4096 B + fp32  |
|   • STAT (cached trajectories + uploU) StarCenter SONA trajectories + f64 uplo='U' fixed Gram     |
|   • WGHT (canonical v3.0 .dnam bytes)  "sona" .dnam wrapper around embedded .dnam-bin VM (< 0.2 ms)|
|   • AUTH (128 B AUTH Record)           Creator signature ("DNPJ-PROJECT-MANIFEST-V3")             |
+===================================================================================================+
```

### 1.1 64B Preamble & Content-Addressed Chunk Directory

`r[container.proj.preamble-and-directory]`

Every v3.0 `.dnam-proj` container is a declarative, code-execution-free binary file (REQ-8.8) parsed from fixed little-endian binary records with strict bounds validation.

Parsing begins with the 64-byte Project Container Preamble at byte 0 (`magic = "DNPJ"`, `version_major = 3`, `version_minor = 0`), rejecting any mismatched version fail-fast with `UnsupportedVersion`.

To enable single-request cloud inspection via HTTP Range (`bytes=0-65535`), `chunk_dir_offset` must equal 64 and `manifest_offset` must equal `64 + 64 * chunk_count`. Cloud clients issue a single range read to inspect all project metadata and chunk SHA-256 digests in one round-trip, issuing a second exact range read to `align64(manifest_offset + manifest_len)` only when the manifest exceeds 64 KB.

The total file length, `manifest_offset`, `align64(manifest_offset + manifest_len)`, and every chunk `offset` must be exact multiples of 64 B. Checksum `header_crc32c` covers preamble bytes 0..40 concatenated with the `64 * chunk_count` directory bytes and `manifest_len` unpadded JCS manifest bytes, while padding in `[manifest_offset + manifest_len .. align64(manifest_offset + manifest_len))` must equal `0x00`.

Each 64-byte Chunk Directory Entry must specify a known FourCC (`CALB`, `SPCM`, `INPC`, `STMC`, `STAT`, `WGHT`, or `AUTH`). Entries must declare contiguous ascending 64 B-aligned bounds and zeroed trailing padding to the next 64 B boundary.

Every chunk payload must also satisfy matching CRC-32C and SHA-256 digests over its `byte_len` bytes. Unknown `lifecycle_flags` bits, non-zero reserved bytes in `[44..64)`, truncated payloads, or digest mismatches fail fast with a validation error.

**1. Project Container Preamble (64 B, 64-byte aligned, file offset 0..64):**

| Offset | Bytes | LE Wire Type | Field Name | Clean-Room Validation Invariant |
| :---: | :---: | :--- | :--- | :--- |
| 0..4 | 4 | `u8[4]` | `magic` | Exact `"DNPJ"` |
| 4..6 | 2 | `u16le` | `version_major` | Must equal 3 |
| 6..8 | 2 | `u16le` | `version_minor` | Must equal 0 |
| 8..12 | 4 | `u32le` | `lifecycle_flags` | `PROJ_FLAG_*`: `1<<0` `HAS_CALIBRATION`, `1<<1` `HAS_RAW_STEMS`, `1<<2` `HAS_CAPSULES`, `1<<3` `HAS_STAT`, `1<<4` `HAS_WGHT`, `1<<5` `HAS_AUTH`, `1<<6` `STEREO_OUT`, `1<<7` `COMPACTED_CAPSULES` (`KNOWN_MASK = 0x00FF`) |
| 12..16 | 4 | `u32le` | `chunk_count` | Number of 64 B TOC records `N` (0–4,096 = `MAX_PROJ_CHUNKS`) |
| 16..24 | 8 | `u64le` | `chunk_dir_offset` | Must equal 64 |
| 24..32 | 8 | `u64le` | `manifest_offset` | Must equal `64 + 64 * chunk_count` |
| 32..40 | 8 | `u64le` | `manifest_len` | Unpadded UTF-8 RFC 8785 JCS manifest length (> 0) |
| 40..44 | 4 | `u32le` | `header_crc32c` | Castagnoli CRC-32C over `preamble[0..40] \|\| dir[64 * chunk_count] \|\| manifest[manifest_len]` |
| 44..64 | 20 | `u8[20]` | `reserved` | All 20 bytes must equal `0x00` |

**2. Chunk Directory Entry (64 B TOC entry, repeated `N` times at `64..64+64*N`):**

| Offset | Bytes | LE Wire Type | Field Name | Clean-Room Validation Invariant |
| :---: | :---: | :--- | :--- | :--- |
| 0..4 | 4 | `u8[4]` | `fourcc` | `"CALB"`, `"SPCM"`, `"INPC"`, `"STMC"`, `"STAT"`, `"WGHT"`, or `"AUTH"` |
| 4..8 | 4 | `u32le` | `chunk_id` | 0 for singletons (`CALB`/`INPC`/`STAT`/`WGHT`/`AUTH`); `stem_index` matching `stems_registry` for `SPCM`/`STMC` |
| 8..12 | 4 | `u32le` | `flags` | Must equal 0 |
| 12..16 | 4 | `u32le` | `crc32c` | Castagnoli CRC-32C over `bytes[offset..offset + byte_len]` |
| 16..24 | 8 | `u64le` | `offset` | 64 B-aligned Region 3 offset (`(offset & 0x3F) == 0`, ascending non-overlapping) |
| 24..32 | 8 | `u64le` | `byte_len` | Unpadded chunk payload byte count (`[offset + byte_len .. align64(offset + byte_len))` all `0x00`) |
| 32..64 | 32 | `u8[32]` | `sha256` | SHA-256 digest of `bytes[offset..offset + byte_len]` |

**3. `CALB` (64-byte Calibration Record) & `SPCM` (64 B Header + Aligned `fp32` Audio):**

| Chunk | Offset | Bytes | LE Wire Type | Field & Invariant |
| :--- | :---: | :---: | :--- | :--- |
| `CALB` (64 B) | 0..4 | 4 | `u8[4]` | `magic`: ASCII `"CALB"` |
| `CALB` | 4..6 | 2 | `u16le` | `version`: Must equal 3 |
| `CALB` | 6..7 | 1 | `u8` | `bit_depth_code`: 0=Int12, 1=Int16, 2=Int24, 3=Int32, 4=Float32 (> 4 rejected) |
| `CALB` | 7..8 | 1 | `u8` | `polarity_inverted`: 0 or 1 (> 1 rejected) |
| `CALB` | 8..12 | 4 | `u32le` | `sample_rate_device_hz`: 8,000–384,000 |
| `CALB` | 12..16 | 4 | `u8[4]` | `reserved_pad`: All `0x00` |
| `CALB` | 16..24 | 8 | `f64le` | `loopback_lag_samples`: Finite sub-sample lag at 48 kHz |
| `CALB` | 24..32 | 8 | `f64le` | `signed_gain`: Finite linear loopback gain |
| `CALB` | 32..36 | 4 | `f32le` | `input_reference_dbfs`: Finite calibration reference level |
| `CALB` | 36..40 | 4 | `f32le` | `noise_floor_dbfs`: Finite quiescent hardware noise floor |
| `CALB` | 40..60 | 20 | `u8[20]` | `reserved_tail`: All `0x00` |
| `CALB` | 60..64 | 4 | `u32le` | `crc32c`: Castagnoli CRC-32C over `bytes[0..60]` |
| `SPCM` (>= 64 B) | 0..4 | 4 | `u8[4]` | `magic`: ASCII `"SPCM"` (60 B header pad 4..64, stripped in < 1 ms by capsule compaction) |
| `SPCM` | 64.. | `4 * L` | `f32le[L]` | Aligned raw stem `fp32` PCM samples (`channels * num_frames`) |

**4. `DNWB` White-Box Pipe Wire Protocol (64-byte `DNWB` Header), `DNPR` Telemetry Seqlock (256 B) & WAV Ingestion:**

| Protocol / Container | Offset | Bytes | LE Wire Type | Field & Clean-Room Invariant |
| :--- | :---: | :---: | :--- | :--- |
| `DNWB` (64 B) | 0..4 | 4 | `u8[4]` | `magic`: `"DNWB"` |
| `DNWB` | 4..6 | 2 | `u16le` | `version`: 1 (wire framing version) |
| `DNWB` | 6..8 | 2 | `u16le` | `command`: 0 = Reset, 1 = RenderStem, 2 = RenderFrame |
| `DNWB` | 8..12 | 4 | `u32le` | `sample_rate_hz`: Supported 8,000–1,536,000 |
| `DNWB` | 12..16 | 4 | `u32le` | `num_frames`: Sample frame count |
| `DNWB` | 16..26 | 10 | `5 × u16le` | `num_input_channels`, `num_output_channels`, `num_controls`, `num_switches`, `num_internal_nodes` |
| `DNWB` | 26..28 | 2 | `u16le` | `flags`: `1<<0 STRUCTURED_V1`, `1<<1 HAS_TRAJECTORIES`, `1<<2 HAS_FREEZE_WINDOW` (white-box fixture hold interval lowered to data-driven `ControlBus` `freeze_mix` slot 16 targets during training), bits 8..11 = `ch_mask` (`popcount(ch_mask)` matches active channels) |
| `DNWB` | 28..36 | 8 | `u64le` | `payload_byte_len`: `<= MAX_DNWB_PAYLOAD_BYTES` = 64 MiB (67,108,864 B); followed by request (`u16le stem_id_len \|\| stem_id \|\| f32le[ctrl] \|\| u8[sw] \|\| f32le[ch*frames] \|\| opt. (u16le traj_count \|\| f32le[traj_count*frames]) \|\| opt. u64le[2] hold_window`) or response (`u16le stem_id_len \|\| stem_id \|\| f32le[out_ch*frames] \|\| internal_nodes`) |
| `DNWB` | 36..64 | 28 | `u8[28]` | `reserved`: All `0x00` |
| `DNPR` (256 B) | 0..64 | 64 | `16 × u32le` | Words 0..16: `w[0..7]` (`w[0] seq` odd=writing/even=consistent, `w[1] status\|phase<<8\|tier<<16\|permille<<20`, `w[2] step\|total<<16`, `w[3] stg_step\|stg_total<<16`, `w[4] best_loss f32`, `w[5] mr_iir f32`, `w[6] clamped_ctrls\|vis<<16`), `w[7..11] tier_besr_db[4]`, `w[11..15] tier_ltas_db[4]`, `w[15] = PROGRESS_MAGIC_DNPR = 0x444E_5052` |
| `DNPR` | 64..256 | 192 | `48 × u32le` | Words 16..64: 16..32 `target_wave_i8[64]`, 32..48 `model_wave_i8[64]`, 48..56 `ltas_error_q20_i8[32]` (0.05 dB/step, ±6.35 dB), 56..64 `kautz_poles_q8[16]` (`16 × (r_u8, f_u8)`) |
| WAV `RIFF` | 0..12 | 12 | `4+u32le+4` | `"RIFF" \|\| riff_size u32le (>= 4, <= len-8) \|\| "WAVE"`, followed by even-padded `(id[4], len u32le, body[len], opt pad0)` chunks |
| WAV `fmt ` | 0..16+ | >= 16 | LE fields | `audio_format` (1 = PCM, 3 = IEEE Float, `0xFFFE` = `WAVE_FORMAT_EXTENSIBLE` >= 40 B, `cbSize >= 22`), `channels >= 1`, `sample_rate_hz >= 1`, `avg_bytes_per_sec == sample_rate_hz * block_align`, `block_align == channels * container_bytes`, `bits_per_sample`: Int12 (2 B via `0xFFFE`), Int16 (2 B), Int24 (3 B), Int32 (4 B), Float32 (4 B, `SubFormat` 1/3 + `KSDATAFORMAT_SUBTYPE_TAIL`) |
| WAV `data` | 0..8+ | `4+u32le+N` | PCM/`f32le` | Must follow `fmt ` chunk; finite Float32, exact multiple of `channels * container_bytes` bytes |

Inside `DNPR` telemetry words 48..64, `ltas_error_q20_i8[32]` (bytes 192..224) quantizes `10 * log10((P_model[b] + floor) / (P_tgt[b] + floor)) * 20` clamped to `[-127, 127]` (0.05 dB/step over ±6.35 dB).

Words 56..64 (bytes 224..256) encode `kautz_poles_q8[16]` as `(r_u8, f_u8)` pairs where `r_u8 = round(clamp((r - 0.5) / 0.5, 0, 1) * 255)` and `f_u8 = round(clamp(ln(clamp(theta * sr / (2 * pi), 20, 20000) / 20) / ln(1000), 0, 1) * 255)`.

### 1.2 RFC 8785 JCS Project Manifest

`r[container.proj.jcs-manifest]`

The manifest slice at `[manifest_offset .. manifest_offset + manifest_len)` stores a strict UTF-8 RFC 8785 JSON Canonicalization Scheme (`JCS`) serialization of the project JSON manifest (`manifest`, zero-padded with `0x00` to the next 64 B boundary), rejecting UTF-8 BOMs, duplicate JSON keys, unknown schema fields, and non-canonical key ordering:

| Manifest Section | Required Fields | Normative Invariant |
| :--- | :--- | :--- |
| Envelope & Nested `base` Header | `version = "3.0.0"`, `container_kind = "dnam-proj"`, `base` (v3.0 JSON header with `base.container_kind = "dnam"`, `base.architecture = "sona"`) | Shares the exact validated v3.0 `"sona"` JSON header schema (`ref[container.dnam.jcs-header-schema]`). |
| `project_meta`, `rig_routing` & `calibration_summary` | `project_meta.{project_id, rig_notes, updated_at}`, `rig_routing.{sample_rate_device_hz, bit_depth, input_channels, output_channels, reamp_send_channel}`, `calibration_summary` | Tracks project identity, native hardware rate (8..384 kHz), bit depth (`int12`, `int16`, `int24`, `int32`, `float32`), and 64-byte `CALB` v3 telemetry. |
| `stems_registry` | Ordered `StemRegistryEntry` list (`stem_id`, `chunk_id`, `enabled`, `is_star_center`, `controls_meta`, `provenance`, `user_metadata`, `extracted_physics`) | Binds each stem's `chunk_id` (`u16` `STMC` `stem_index`), SHA-256 dry/wet `<f4` PCM provenance, and `DNAM-GRAM-INPUT-V3` fingerprint (REQ-3.2); optional `extracted_physics` carries the 12 physical descriptor fields defined below under strict unknown-key rejection. |
| `audit_report` & `solver_telemetry` | `audit_report` (`AuditReport`), `solver_telemetry.{active_stem_count, total_solve_ms, tier_esr_db}` | Persists Phase-1 control-space audit clamps, deterministic `[PANAMA]` corner recommendations inside `audit_report.prescriptions` (zero manifest key additions), and 4-tier ESR certification metrics across incremental vault edits. |

Under `stems_registry[*].extracted_physics`, strict JCS schema decoding retains `delays_ms`, `lfo_freqs_hz`, and `decay_gains_lin` (number arrays, empty default), `detected_grain_size_ms` and `detected_pitch_shift_semitones` (numbers, 0.0 default), and adds seven optional defaulting capture fields:

| `extracted_physics` Optional Field | JSON Wire Type & Omission Default | Clean-Room Semantics |
| :--- | :--- | :--- |
| `bla_mag_33_db` | Number array of 33 elements when non-empty (empty array when omitted) | Best Linear Approximation 1/3-octave magnitude profile in dB |
| `even_dist_ratio_db` | Number (0.0 default when omitted) | Broadband even-order distortion ratio relative to fundamental in dB |
| `odd_dist_ratio_db` | Number (0.0 default when omitted) | Broadband odd-order distortion ratio relative to fundamental in dB |
| `noise_var_33_lin` | Number array of 33 non-negative elements when non-empty (empty default) | Law 0.A-gated 1/3-octave repeatability noise variance in linear power |
| `edc_t60_s` | Non-negative number (0.0 default when omitted) | Schroeder Energy Decay Curve 60 dB reverberation time in seconds |
| `law0a_ltv_flag` | Boolean (`false` default when omitted) | True when Law 0.A detects time-varying modulation across Regime 1 periods |
| `coreset_regime_offsets` | 5-element integer array (`[0,0,0,0,0]` default) | Monotonic sample boundaries `[0, r2_start, r3_start, r4_start, total_frames]` |

### 1.3 Lifecycle States, Capsule Compaction & Cloud Range Sync

`r[container.proj.lifecycle-states]`

A `.dnam-proj` container transitions monotonically or non-destructively across flag-consistent lifecycle states governed by `PROJ_FLAG_*` bitflags. These bitflags cover calibration (`HAS_CALIBRATION`), raw stems (`HAS_RAW_STEMS`), capsules (`HAS_CAPSULES`, `COMPACTED_CAPSULES`), and stereo output (`STEREO_OUT`).

Trained and signed containers add `HAS_STAT`, `HAS_WGHT`, and `HAS_AUTH` to track embedded `StarCenter` statistics, runtime weights, and cryptographic signatures without dropping prior state metadata.

| Lifecycle State | Embedded Chunks | Key Behavioral Invariant |
| :--- | :--- | :--- |
| Active Capture | `CALB` + `SPCM` + `INPC`/`STMC` | Retains lossless raw `SPCM` stems (spanning full 8.0–12.0 s StarCenter 4-regime sequences for Pass 1 physics identification) alongside extracted condensed `INPC`/`STMC` capsules |
| Trained Full Project | `+ STAT + WGHT` (+ optional `AUTH`) | Embeds `StarCenter` `uplo='U'` sufficient statistics (`STAT`) and certified 4-tier v3.0 `.dnam` weights (`WGHT`, wrapping the compiled `.dnam-bin` `"DNBN"` register program) |
| Capsule Compaction | Strips `SPCM`, keeps `INPC`/`STMC`/`STAT`/`WGHT` | Requires condensed `INPC`/`STMC` whenever `stems_registry` is non-empty; preserves `PROJ_FLAG_COMPACTED_CAPSULES` across capsule edits and shrinks 16-stem stereo captures to <= 16.4 MB of capsules (~16.7 MB / <= 16.0 MiB full `.dnam-proj` with `STAT` and `WGHT`, under < 20 MB REQ-3.3) |

---

## 2. Capture Capsules & Compositional Sufficient Statistics

### 2.1 Shared Input Coreset Capsule (`INPC`)

`r[container.proj.inpc-capsule]`

The `INPC` chunk (`chunk_id = 0`) stores the shared 2.0–2.5 s (96,000–120,000 frames at 48 kHz, `num_frames % 16 == 0`, canonically a condensed 2.5 s / 120,000-frame 4-regime coreset) full-bandwidth (20 Hz–20 kHz) fp32 dry excitation capsule inside a 64 B-aligned record. Mono capsules occupy 384,640–480,640 B and stereo capsules occupy 768,640–960,640 B, preserving 18–20 kHz air and multi-second envelopes inside multi-stem `.dnam-proj` vaults under 20 MB (REQ-3.3), while full 8.0–12.0 s active `StarCenter` captures live strictly in transient `SPCM` ahead of compaction.

| Offset | Bytes | LE Wire Type | Field Name | Clean-Room Invariant |
| :---: | :---: | :--- | :--- | :--- |
| 0..4 | 4 | `u8[4]` | `magic` | Exact `"INPC"` |
| 4..6 | 2 | `u16le` | `version` | Must equal 3 |
| 6..8 | 2 | `u16le` | `channels` | 1 (mono) or 2 (stereo) |
| 8..12 | 4 | `u32le` | `sample_rate_hz` | Must equal 48,000 |
| 12..16 | 4 | `u32le` | `num_frames` | Coreset frame count (96,000–120,000, divisible by 16) |
| 16..56 | 40 | `u8[40]` | `reserved` | All 40 bytes must equal `0x00` |
| 56..60 | 4 | `u32le` | `rec_len` | Total byte length (`640 + channels * num_frames * 4`) |
| 60..64 | 4 | `u32le` | `crc32c` | Castagnoli CRC-32C over `bytes[64..rec_len]` |
| 64..320 | 256 | `u8[256]` | `seed_256` | Deterministic PRNG seed (256 B) |
| 320..640 | 320 | `u8[320]` | `schedule_320` | Adaptive stimulus schedule (`num_carriers`, `valid_frames`, `active_samples`, `reserved_0`, `carrier_bins[0..76]`) |
| 640..rec_len | `4 * C * N` | `f32le[C * N]` | `dry_coreset_fp32` | Finite interleaved fp32 dry stimulus coreset (384,000 B mono / 768,000 B stereo at 2.0 s) |

Inside `[64..320)`, `seed_256` stores the 256 B deterministic PRNG seed reproducing the shared stimulus sequence, and `schedule_320` immediately follows in `[320..640)`.

Relative bytes 0..16 of `schedule_320` hold four `u32le` header fields (`num_carriers`, `valid_frames`, `active_samples`, and `reserved_0`), followed by `carrier_bins` as 76 little-endian 32-bit unsigned entries across relative bytes 16..320.

### 2.2 Per-Stem Response Coreset Capsule (`STMC`)

`r[container.proj.stmc-capsule]`

Each stem s in `stems_registry` owns a dedicated STMC chunk (`chunk_id == stem_index`) of 64 B-aligned size covering 96,000–120,000 frames (2.0–2.5 s). Mono captures occupy 388,800–484,800 B and stereo captures occupy 772,800–964,800 B, keeping compacted 16-stem stereo projects under 20 MB per REQ-3.3. Loaders determine channel count strictly from the STMC `STEREO` flag (`1<<3`) and INPC `channels` field, never inferring channel count from byte length.

| Offset | Bytes | LE Wire Type | Field Name | Clean-Room Invariant |
| :---: | :---: | :--- | :--- | :--- |
| 0..4 | 4 | `u8[4]` | `magic` | Exact `"STMC"` |
| 4..6 | 2 | `u16le` | `version` | Must equal 3 |
| 6..8 | 2 | `u16le` | `stem_index` | Matches `stems_registry[*].chunk_id` |
| 8..10 | 2 | `u16le` | `flags` | `1<<0 ENABLED`, `1<<1 STAR_CENTER`, `1<<2 MODE_A_SWEEP`, `1<<3 STEREO`, `1<<4 WET_POLARITY_INVERTED` (`KNOWN_MASK = 0x001F`) |
| 10..12 | 2 | `u16le` | `n_windows` | Active trajectory window count `R` (1–36 = `STMC_MAX_TRAJECTORY_WINDOWS`) |
| 12..16 | 4 | `f32le` | `pad_gain_f32` | Finite hardware input pad gain |
| 16..20 | 4 | `f32le` | `fractional_lag_samples_f32` | Finite sub-sample alignment lag |
| 20..108 | 88 | `u8[88]` | `band_psd_summary_88` | Band-energy PSD summary (4×4 `uplo='U'` `10 × f64le` upper triangle at 0..80 + `total_power: f64le` at 80..88) |
| 108..112 | 4 | `u32le` | `num_frames` | Coreset frame count (96,000–120,000, divisible by 16) |
| 112..116 | 4 | `u32le` | `n_comb_bins` | Measured comb bin count |
| 116..120 | 4 | `u32le` | `rec_len` | Total byte length (`4,800 + channels * num_frames * 4`) |
| 120..124 | 4 | `u32le` | `body_crc32c` | Castagnoli CRC-32C over `bytes[128..rec_len]` |
| 124..128 | 4 | `u8[4]` | `reserved` | All 4 bytes must equal `0x00` |
| 128..704 | 576 | `36 × 16 B records` | `windows` | `R` trajectory window records (16 B each: +0..2 `control_id: u16le`, +2 `ctrl_kind: u8`, +3 `active_opt: u8`, +4..8 `u_phys: f32le`, +8..12 `window_weight: f32le >= 0`, +12..16 `emr_velocity: f32le`); unused tail `128+16*R..704` must be `0x00` |
| 704..4,800 | 4,096 | `u8[4,096]` | `spectral_summary_4096` | 1,024 × `f32le` averaged 2:1 DFT magnitude bins (`0.5 * (\|X[2k]\| + \|X[2k+1]\|) * (2 / 4,096)`) |
| 4,800..rec_len | `4 * C * N` | `f32le[C * N]` | `wet_coreset_fp32` | Finite aligned full-bandwidth wet response for Path 2 GPU/CPU re-pooling without raw WAVs |

For continuous `MODE_A_SWEEP` capsules decoded into per-tick trajectories (`N_active = STMC.num_frames`), filter out `window_weight <= 0.0`; controls with `M_c == 0` remaining records hold their `controls_meta` value (or control default if omitted), and `M_c == 1` controls stay constant across all frames.

All controls with `M_c >= 2` place ordered knots uniformly via $t_j = \lfloor j (N_{\mathrm{active}} - 1) / (M_c - 1) \rfloor$ without altering the 16-byte layout: continuous sweeps interpolate between adjacent knots, whereas discrete switches (`ctrl_kind == 1`) read `active_opt` (ignoring `u_phys`) piecewise-constantly using integer `midpoint = floor((t_j + t_{j+1}) / 2)` (frames t < midpoint use knot j, and t >= midpoint use knot j + 1).

Inside `[20..108)`, `band_psd_summary_88` stores 10 `f64le` upper-triangle entries `(i <= j)` equal to `sqrt(band_power[i]) * sqrt(band_power[j])` in relative bytes 0..80, followed by `total_power` (`f64le`) in relative bytes 80..88.

Both summaries evaluate the unwindowed `N_fft = 4,096`-point real-to-complex DFT `X[k]` of `(L[n] + R[n]) * 0.5` over leading samples `n = 0..4,095`, defining single-sided bin power $P[k] = |X[k]|^2 \cdot (2 / 4096)^2$ for $k = 1\dots 2047$.

Normalized frequency `s = k / 2,048` splits `P[k]` via degree-7 Bezout polynomial $B_7(h) = 35h^4 - 84h^5 + 70h^6 - 20h^7$ with $h_1 = (1-s)^2$ and $h_2 = 1-s^2$: $w_{\mathrm{lo}} = B_7(h_1)$, $w_{\mathrm{mid}} = B_7(h_2) - B_7(h_1)$, and $w_{\mathrm{hi}} = B_7(s^2)$, summing `band_power = [sum w_lo*P[k], sum 0.5*w_mid*P[k], sum 0.5*w_mid*P[k], sum w_hi*P[k]]`.

Immediately after the trajectory windows, `spectral_summary_4096` (`[704..4,800)`) packs 1,024 `f32le` 2:1 averaged DFT magnitude bins `0.5 * (|X[2k]| + |X[2k+1]|) * (2 / 4,096)` for `k = 0..1,023`, and sets `total_power = sum(band_power)`.

### 2.3 Tier-Ordered Upper-Triangular `StarCenter` Statistics (`STAT`)

`r[container.proj.stat-uplo-u]`

The `STAT` chunk (`chunk_id = 0`) stores cached StarCenter Pass 2 SONA trajectories paired with fixed nested tier dimensions (Lite through Ultra, zero capture-time resizing) as tier-ordered column-major upper-triangular (`uplo = 'U'`) IEEE-754 float64 VarPro sufficient statistics (Lite ⊂ Full ⊂ Pro ⊂ Ultra, REQ-5.3):
$$\mathrm{idx}(i, j) = i + \frac{j(j + 1)}{2}, \qquad 0 \le i \le j < D_T$$

At active tier readout dimension $D_T$ (fixed per tier), `STAT` packs $D_T(D_T + 1) / 2$ upper-triangular `f64le` Gram entries for $\mathbf{G}_0$ starting at byte 0, followed immediately by $2 \times D_T$ `f64le` two-channel cross-correlation entries for $\mathbf{C}_0$ (`16 * D_T` bytes). Loaders verify exact tier byte size, finite `f64` values, and non-negative diagonal entries $G_{ii} \ge 0$, guaranteeing < 50 ms Path 1 Vault edits with 0-ULP StarCenter invariance.

| Retained Tier (`STAT` Constant) | Dimension `D_T` | Packed `uplo='U'` `G_0` Elements (`j(j+1)/2 + i`) | `G_0` Bytes (0..) | `C_0` Bytes (`16 * D_T`) | Exact Chunk Byte Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `PROJ_STAT_BYTES_LITE` | 64 | 2,080 (64 × 65 / 2) | 16,640 B | 1,024 B | 17,664 B (276 CL) |
| `PROJ_STAT_BYTES_FULL` | 128 | 8,256 (128 × 129 / 2) | 66,048 B | 2,048 B | 68,096 B (1,064 CL) |
| `PROJ_STAT_BYTES_PRO` | 192 | 18,528 (192 × 193 / 2) | 148,224 B | 3,072 B | 151,296 B (2,364 CL) |
| `PROJ_STAT_BYTES_ULTRA` | 256 | 32,896 (256 × 257 / 2) | 263,168 B | 4,096 B | 267,264 B (4,176 CL) |

---

## 3. Embedded Runtime Model & Provenance Envelope

### 3.1 Embedded Runtime Weights & Zero-Solve `.dnam` Export (`WGHT`)

`r[container.proj.wght-and-export]`

Every trained v3.0 `.dnam-proj` embeds a complete, self-contained v3.0 (`"architecture": "sona"`) `.dnam` byte stream inside its `WGHT` chunk (`chunk_id = 0`), wrapping the compiled Layer 0 `.dnam-bin` (`"DNBN"`) VM register image together with the shell JCS header, security tables, and mutable trailer footer; see [dnam_binary_format.md](dnam_binary_format.md).

Container assembly and validation enforce `[STAB-ZERO-DC]` invariants across Arena 0..3 (`b_head == 0.0`, resting DC <= 1e-5), rejecting non-zero DC bias payloads immediately.

Runtime `.dnam` export completes in < 0.2 ms at any target tier (Lite, Full, Pro, or Ultra) with zero solver recomputation, and can also yield the bare Layer 0 `.dnam-bin` register program directly from the embedded wrapper.

Export validates `DistributionPolicy` and merges mutable `presets`, `user_metadata`, `stem_user_metadata`, and `controls_schema[*].user_metadata` into the `DMU1` suffix, growing compact 48 B frames up to the 4,096 B mutable `DMU1` + `DMUE` trailer ceiling.

Export then applies lossless keyless prefix truncation (`ref[container.dnam.tier-prefix]`), slicing both the shell prefix and the embedded `.dnam-bin` payload at its immutable `tier_end_offsets[t]` while retaining self-contained 1×-calibrated Arena 0 controls and Lite `CP1`/`W_readout` slices without re-signing.

### 3.2 Cryptographic Provenance Envelope (`AUTH`)

`r[container.proj.auth-signature]`

When signed (`PROJ_FLAG_HAS_AUTH`), the `AUTH` chunk (`chunk_id = 0`) stores an exact 128 B (2-CL) `AUTH` record (`PROJ_AUTH_RECORD_BYTES = 128`) containing an Ed25519 signature over the domain-separated transcript encoded with `encode_domain_field(k, v) = u32le(len(k)) || k || u64le(len(v)) || v`:

| Offset / Segment | Bytes | LE Wire Type | Field & Signing Transcript |
| :--- | :---: | :--- | :--- |
| `AUTH` 0..4 | 4 | `u8[4]` | `magic`: Exact `"AUTH"` |
| `AUTH` 4..6 | 2 | `u16le` | `version`: Must equal 3 |
| `AUTH` 6..32 | 26 | `u8[26]` | `reserved`: All 26 bytes must equal `0x00` |
| `AUTH` 32..64 | 32 | `u8[32]` | `signer_pubkey_ed25519`: Creator Ed25519 verifying public key |
| `AUTH` 64..128 | 64 | `u8[64]` | `signature_ed25519`: Ed25519 signature over concatenated `encode_domain_field` tuples below |
| Sign Msg Field 1 | 42 | UTF-8 + bytes | `encode_domain_field("domain", "DNPJ-PROJECT-MANIFEST-V3")` |
| Sign Msg Field 2 | 65 | bytes | `encode_domain_field("signer_pubkey_ed25519", signer_pubkey_ed25519[32])` |
| Sign Msg Field 3 | 69 | bytes | `encode_domain_field("manifest_immutable_sha256", immutable_jcs_sha256[32])` |
| Sign Msg Field 4 | `34 + 52 * M` | bytes | `encode_domain_field("chunk_directory_v3", u32le(M) \|\| M × (fourcc[4] \|\| chunk_id u32le \|\| flags u32le \|\| byte_len u64le \|\| sha256[32]))` for all `M` non-`AUTH` chunks |

Any tampering with the immutable manifest fields, chunk directory, `STMC` stem capsules, `STAT` Gram entries, or `WGHT` weights fails fast with `CryptoVerificationFailed`.
