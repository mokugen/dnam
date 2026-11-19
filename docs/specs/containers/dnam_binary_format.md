---
id: spec.container.dnam
title: ".dnam Hybrid Binary Container & < 50 ns Zero-Copy Hardware Loader"
status: canonical
version: "3.0.0"
domain: containers
rule_prefixes:
  - "r[container.dnam."
requirements:
  - REQ-5.1
  - REQ-5.6
  - REQ-8.1
  - REQ-8.8
adrs:
  - ADR-0007
  - ADR-0009
  - ADR-0020
  - ADR-0022
---

# `.dnam` Hybrid Binary Container & `< 50 ns` Zero-Copy Hardware Loader

DNAM defines four companion **64-byte cache-line and SIMD-aligned `v3.0` (`"architecture": "sona"`) Hybrid Binary Container formats** engineered via declarative `zerocopy` wire structs (`ref[repo.code.zerocopy-wire-structs]`) with **zero legacy `v2` or `TORA` backwards compatibility** ([`REQ-5.1`, `REQ-5.6`](../../requirements/system_goals_and_requirements.md#6-goal-5--single-run-slimmable-multi-tier-architecture-lite--full--pro--ultra), [`REQ-8.1`, `REQ-8.8`](../../requirements/system_goals_and_requirements.md#9-goal-8--model-provenance-hardware-drm-forensic-watermarking--marketplace-security), [`ADR-0007`](../../adr/0007-exact-integer-control-bus-and-tempo-inversion.md), [`ADR-0009`](../../adr/0009-jcs-prefix-signing-hpke-and-preview-containment.md), [`ADR-0020`](../../adr/0020-modular-monorepo-and-dnam-proj-container.md), [`ADR-0022`](../../adr/0022-sona-state-space-operator-and-webgpu-architecture.md)):

1. `r[container.dnam.formats]` **`.dnam` (Magic `b"DNAM"`, Version `3.0`, `"architecture": "sona"`):** Runtime deployment model built from `#[repr(C)]` `zerocopy::byteorder::little_endian::{U16, U32, U64, F32, F64}` structs (`FromBytes, IntoBytes, KnownLayout, Immutable`): a `64 B` `DnamPreambleWire`, `128 B` `DnamDirectoryWire`, RFC 8785 JCS immutable header, Region 4 `shared_extent` (`P * 64 B` `ControlClampDescriptorWire` table + overflow `CP1` contributor stream), `DAU1`/`DAI1` authentication envelope, four nested compile-time `const` `SonaTierSpec` L1D-resident payloads (`Arena0LiteWire..Arena3UltraDeltaWire`: `Lite ⊂ Full ⊂ Pro ⊂ Ultra`, `ActiveHot <= 4.5 / 7.8 / 15.5 / 30.0 KiB`, cumulative `64 B`-aligned bounds `<= 4,608 / 7,936 / 15,872 / 30,720 B` locked at `Gate 0b`), and a growable `DMU1`/`DME1` + `DMUE` mutable suffix. Loaders reject `version_major != 3 || version_minor != 0` or non-`"sona"` headers fail-fast with `DnamError::UnsupportedVersion`.
2. **`.dnam-proj` (Magic `b"DNPJ"`, Version `3.0`):** Single-file cloud-ready project and Capture Vault container front-loading a `64 B` preamble, content-addressed `SHA-256` chunk directory, and RFC 8785 JCS manifest for `HTTP Range` inspection alongside `CALB`, `SPCM`, `INPC`/`STMC` `2.0–2.5 s` full-bandwidth (`20 Hz–20 kHz`) `fp32` capsules, `STAT` cached `StarCenter` trajectories & fixed-`const` `uplo='U'` sufficient statistics, `WGHT` embedded `v3.0` `.dnam` weights, and `AUTH` provenance ([`project_container_format.md`](project_container_format.md)).
3. **`.dnam-seal` (Magic `b"DNSL"`, Version `3.0`):** Opaque 64-byte-aligned OEM hardware-enclave wrapper (`DnamProtectionProfile::EnclaveSealed`, `DSA1` manifest; see [`../security/provenance_and_enclave_drm.md` §3](../security/provenance_and_enclave_drm.md#3-pillar-2-hardware-enclave--oem-drm-oemdevicecertificate-attestation--dnam-seal-bdnsl-wrapper)).
4. **`.dnam-prev` (`container_kind: "preview"`, Version `3.0`):** Dedicated `Lite`-only storefront auditioning container that physically omits `Full`, `Pro`, and `Ultra` payloads while retaining self-contained `Arena 0` control descriptors and `Lite` `CP1`/`W_readout` slices ([`../security/watermarking_and_marketplace.md` §3](../security/watermarking_and_marketplace.md#3-commercial-storefront-platform-auditioning-royalty-splits--storefront-metadata)).

---

## 1. Hybrid Binary Container Wire Framing & `< 50 ns` Zero-JSON Hardware Fast-Path

`r[container.dnam.wire-framing]`

```mermaid
flowchart LR
    subgraph Header ["1. Fixed Preamble, Directory & Signed Header (zerocopy v3.0)"]
        direction TB
        P["Region 1 (0..63, 64 B): DnamPreambleWire<br/>Magic DNAM, Version 3.0, Profile, AuthKind, MaxTier, Train SR"]
        D["Region 2 (64..191, 128 B): DnamDirectoryWire<br/>U64 Offsets & Lengths + 4 Original Tier Ends"]
        H["Region 3 (192..): RFC 8785 JCS Immutable Header ('sona')<br/>+ Region 4 shared_extent + DAU1/DAI1 Auth Envelope"]
        P --> D --> H
    end

    subgraph Arenas ["2. Contiguous 64B-Aligned Constant-Size SONA Tier Payloads (tier_start..tier_end)"]
        direction TB
        A0["Arena 0 (Arena0LiteWire, <= 4.5 KiB):<br/>[1,1,1,1] SONA Core + 32 FIR + 8 Kautz + Quad 0 Op-T + Self-Contained Controls & W_readout"]
        A1["Arena 1 (Arena1FullDeltaWire, cum. <= 7.8 KiB):<br/>[2,2,1,1] @ 2x OS Delta + +16 FIR (48) + +4 Kautz (12) + Quad 1 Op-T"]
        A2["Arena 2 (Arena2ProDeltaWire, cum. <= 15.5 KiB):<br/>[3,2,2,1] @ 2x OS Delta + +48 FIR (96) + +12 Kautz (24) + Quads 2-3 Op-T"]
        A3["Arena 3 (Arena3UltraDeltaWire, cum. <= 30.0 KiB):<br/>[4,3,2,1] @ 2x OS Delta + +32 FIR (128) + +8 Kautz (32) + Quad 3 Diffuser + +8-D Ultra Ctrl"]
        A0 --> A1 --> A2 --> A3
    end

    subgraph Suffix ["3. Growable Mutable Suffix"]
        M["DMU1 (Open) / DME1 (Encrypted) Frame<br/>Presets & Open user_metadata + 16B DmueFooterWire"]
    end

    Header --> Arenas --> Suffix
```

```text
+-----------------------------------------------------------------------+ 0x0000 (Byte 0, 64B-aligned)
| 1. DnamPreambleWire (#[repr(C, align(64))], zerocopy LE)              |
|    • 0..4:   magic = "DNAM" (or "DNSL" for enclave-sealed wrapper)    |
|    • 4..8:   version_major = 3 (U16), version_minor = 0 (U16)         |
|    • 8..12:  protection_profile (u8), auth_kind (u8), max_tier (u8)   |
|    • 12..16: sample_rate_train_hz = 48000 (U32) | 16..64: zero (48 B) |
+-----------------------------------------------------------------------+ 0x0040 (Byte 64, 64B-aligned)
| 2. DnamDirectoryWire (#[repr(C)], 128 B, zerocopy U64 LE)             |
|    • 0..16:  immutable_header_offset = 192, immutable_header_length   |
|    • 16..32: shared_binary_offset, shared_binary_length               |
|    • 32..48: authentication_envelope_offset, auth_envelope_length     |
|    • 64..104: tier_start (U64) + original_tier_ends[4] (Lite..Ultra)  |
+-----------------------------------------------------------------------+ 0x00C0 (Byte 192, 64B-aligned)
| 3. Canonical RFC 8785 JCS Immutable JSON Header ("sona") + Zero Pad   |
+-----------------------------------------------------------------------+ shared_binary_offset (64B-aligned)
| 4. Shared Binary Descriptors (shared_extent: P * 64B ControlClamp-    |
|    DescriptorWire Table + Overflow CP1 Stream) + Zero Pad to 64 B     |
+-----------------------------------------------------------------------+ authentication_envelope_offset (64B)
| 5. Authentication Envelope (DAU1 Open / DAI1 Protected) + Zero Pad    |
+=======================================================================+ tier_start (64B-aligned)
| 6A. Arena 0: Arena0LiteWire [tier_start .. lite_end]                  | <- Lite strip boundary
|     (Includes self-contained Lite controls, CP1, U_int, W_ctrl & RO)  |
+-----------------------------------------------------------------------+ original_tier_ends[0] (64B-aligned)
| 6B. Arena 1: Arena1FullDeltaWire [lite_end .. full_end]               | <- Full strip boundary
+-----------------------------------------------------------------------+ original_tier_ends[1] (64B-aligned)
| 6C. Arena 2: Arena2ProDeltaWire [full_end .. pro_end]                 | <- Pro strip boundary
+-----------------------------------------------------------------------+ original_tier_ends[2] (64B-aligned)
| 6D. Arena 3: Arena3UltraDeltaWire [pro_end .. ultra_end]              | <- Ultra strip boundary
+=======================================================================+ retained_tier_end (64B-aligned)
| 7. Growable Mutable Frame (DMU1 Open / DME1 HPKE) + 16-Byte           |
|    DmueFooterWire (magic "DMUE" + U64 frame_len + U32 reserved=0)     |
+-----------------------------------------------------------------------+ EOF
```

### 1.1 Preamble & Directory Invariants

`r[container.dnam.preamble-directory]`

All integer and floating-point fields use `zerocopy::byteorder::little_endian::{U16, U32, U64, F32, F64}` (`#[repr(C)]` deriving `FromBytes, IntoBytes, KnownLayout, Immutable` with compile-time `size_of` / alignment assertions and `#![forbid(unsafe_code)]`, `ref[repo.code.zerocopy-wire-structs]`). Every binary region start (`shared_binary_offset`, `authentication_envelope_offset`, `tier_start`, and each issued `original_tier_ends[t]`) is `64 B`-aligned, strictly ordered, and checked against `u64` overflow before allocation (`validate_v3_dnam_layout_no_alloc`). Region 4 (`shared_extent`) packs the `P * 64 B` `ControlClampDescriptorWire` table followed by any overflow `CP1` contributor stream (`ADR-0007`), while `Arena0LiteWire` also embeds self-contained `Lite` control tables so tier stripping preserves parametric controls. Header `slimmable_tiers.*.arena_end_byte_offset` and `tensor_directory.*.byte_offset` values are relative to `tier_start` so lower-tier buyer issuances retain the creator-signed immutable header unmodified.

### 1.2 Open `.dnam` Structural Layout

`r[container.dnam.open-layout]`

The fixed `64 B` `DnamPreambleWire` (`0..63`) and `128 B` `DnamDirectoryWire` (`64..191`) enforce `version_major = 3`, `version_minor = 0`, `protection_profile = 0` for `Open`, and `authentication_kind = 0` (creator-only) or `1` (creator + buyer). Stripping a container to tier `T` (`strip_container_to_tier`) retains the original preamble, directory, immutable header, shared binary, authentication envelope, and tier payloads through `original_tier_ends[T]` (preserving full `Arena 0` controls even when `T = Lite`), placing the mutable frame and `DmueFooterWire` immediately after `original_tier_ends[T]` without re-signing.

#### Open Mutable Frame

`r[container.dnam.mutable-frame]`

Immediately after the retained tier boundary, every Open container places a growable `DMU1` frame (`Dmu1HeaderWire`) followed by a `16 B` `DmueFooterWire` (`#[repr(C)]`, `zerocopy`):

```text
Offset   Size     Field (Dmu1HeaderWire / DmueFooterWire, zerocopy LE)
  0        4      magic = ASCII "DMU1"
  4        4      Castagnoli CRC-32C (U32) over ("DMU1" || u64le(N) || JSON bytes)
                  (poly 0x82F63B78, init/XOR 0xFFFFFFFF; "{}" = 0xA1CCB48F, "123456789" = 0xE3069283)
  8        8      UTF-8 JSON byte length N (U64, N >= 2)
 16        N      UTF-8 RFC 8259 JSON object (no BOM, no duplicate keys)
16+N      pad     Optional zero-filled padding
EOF-16    16      DmueFooterWire: ASCII "DMUE" (4 B) || frame_len (U64) || reserved=0 (U32)
```

Protected containers replace `DMU1` with the HPKE-wrapped `DME1` frame using the same `16 B` `DmueFooterWire` ([`../security/provenance_and_enclave_drm.md` §3.3](../security/provenance_and_enclave_drm.md#encrypted-mutable-suffix)).

#### Mutable JSON Members

`r[container.dnam.mutable-json]`

| Member | Type & Specification |
| :--- | :--- |
| `version` | Optional mutable envelope version (`1` when present; omitted defaults to `1`). |
| `presets` | Array of preset objects (`id`, `name`, `controls: dict[str, float \| str]`, optional `description`, `user_metadata`). |
| `user_metadata` | Open JSON object for model-level catalog tags and host data. |
| `control_user_metadata` | Object mapping signed `control_id` strings to open metadata objects. |
| `stem_user_metadata` | Object mapping signed `stem_id` strings to open metadata objects. |

On container load, the loader validates `Dmu1HeaderWire`/`Dme1HeaderWire` CRC-32C and `DmueFooterWire` via zero-copy `Ref::from_bytes` and merges mutable `presets`, `user_metadata`, `control_user_metadata`, and `stem_user_metadata` over the signed immutable header view without modifying the authenticated prefix bytes. Loading a partial preset restores signed defaults for all omitted controls at the effective tier before overlaying the preset's assignments and applying whole-batch conditional clamping ([`../dsp/parametric_control_bus.md` §2.4.1](../dsp/parametric_control_bus.md#conditional-batch-safety)).

---

## 2. Tier-Prefix Hardware Loading

### 2.1 Lower-Block-Triangular Tier Payloads

`r[container.dnam.tier-prefix]`

Every tier's runtime topology is locked to a compile-time `const` `SonaTierSpec` (`LiteSpec ⊂ FullSpec ⊂ ProSpec ⊂ UltraSpec`, `0` capture-time tier resizing, `ref[arch.tiers.maximize-quality-within-budget]`), yielding constant-size `zerocopy` arena structs:
- **`Arena0LiteWire` (`Arena 0`):** Stores the `Lite` (`[1,1,1,1]`) base parameters, Stage 0 crossbar & `4`-lane `Q0.32` phasors, `Op-P` config, `32` early FIR taps + `8` Kautz pairs, `Quad 0` `Op-T` tank, **self-contained base control tables (`ControlClampDescriptorWire`, `CP1`, `U_int`, `W_ctrl`)**, and `Lite` `fp16` `W_readout` (`VarPro` subspace), so `strip_container_to_tier(Lite)` retains full parametric control fidelity (`REQ-5.6`).
- **`Arena1FullDeltaWire..Arena3UltraDeltaWire` (`Arena 1..3`):** Store causally nested `Full` (`[2,2,1,1] @ 2×`), `Pro` (`[3,2,2,1] @ 2×`), and `Ultra` (`[4,3,2,1] @ 2×`) state deltas, incremental Stage III FIR/Kautz blocks, `Op-T` quads/diffusers, per-tier `W_readout` projections, and `Arena 3` `+8-D` `Ultra` control delta (`REQ-5.1`, `REQ-5.2`).

### 2.2 Model-Declared Delay Histories (`delay_histories`)

`r[container.dnam.delay-histories]`

The signed immutable JSON header declares all required `Op-T` delay and all-pass histories in `delay_histories`, honoring `bytes_per_sample` `a ∈ {2, 4}` (`"fp16"` inline/cold on `Lite`/`Full`, optional `"fp16"` or `"f32"` cold `ext_lines` on `Pro`/`Ultra` for ultra-low-floor `30 s` shimmer tails):

```json
{
  "id": "tank.quad0.line0",
  "stage": "op_t_tank",
  "kind": "fractional",
  "introduced_tier": "lite",
  "sample_format": "fp16",
  "reference_samples": { "lite": "432000" }
}
```

Each `reference_samples` entry is canonical ASCII decimal text parsed as a nonzero `u64` (avoiding IEEE-754 `binary64` integer limits) and admitted against the host memory ceiling on the cold path ([`../dsp/op_t_transport.md` §1.1](../dsp/op_t_transport.md#11-exact-sized-delay-histories-and-branchless-wrap)).

### 2.3 Pre-Packed SIMD Data, `zerocopy` Descriptors & Load-Time Validation

`r[container.dnam.load-validation]`

Loaders validate all `zerocopy` wire headers, tensor offsets, 64-byte alignments, shapes, finite coefficients, and signature commitments (`SonaCellDescWire`, `#[repr(C)]` with `FromBytes, IntoBytes, KnownLayout, Immutable`) before exposing zero-copy `SonaHotSlab` views, and initialize recursive `TPT-SVF`, `ParaLRU`, and `Op-T` states to exact resting zero equilibrium ([`../dsp/op_n_commuted_cascade.md` §3.4](../dsp/op_n_commuted_cascade.md#34-boundary-conditions-resting-dc-equilibrium--pop-free-startup)).

`r[container.dnam.tensor-directory]` **Canonical 64B-Aligned Constant-Size Tier Arena Slice Directory (`ADR-0022`, `ref[repo.code.zerocopy-wire-structs]`):**
Every slice declared in `tensor_directory` has a 64-byte-aligned `byte_offset` relative to `tier_start` (`byte_offset % 64 == 0`), a `byte_length` within its owning tier's constant-size `Arena*Wire` cumulative prefix boundary `original_tier_ends[t] - tier_start` (`<= 4,608 / 7,936 / 15,872 / 30,720 B` targeting `ActiveHot <= 4.5 / 7.8 / 15.5 / 30.0 KiB` across `Lite / Full / Pro / Ultra`), and uses the canonical per-tier slice identifiers below:

| Slice Identifier | Owning Tier Arena | Element Shape & `zerocopy` Wire Layout | Aligned Size |
| :--- | :---: | :--- | :---: |
| `stage0_crossbar_phasors` | `Arena 0..3` | `S_router[4, 4]` (`S_ret->send == 0`), `SonaCellDescWire`, `Op-P` `HilbertH1`/`T1` config, `4`-lane `Q0.32` phasors (`0.05 Hz–20 kHz`), coercive `G_loop` | `128..384 B` |
| `sona_core_{lite,full,pro,ultra}` | `Arena 0..3` | 4-block Matryoshka `SONA` core (`[1,1,1,1]..[4,3,2,1]`): passive `2x2` `TPT-SVF` modes (`4..12`), `ParaLRU` `(alpha_rel, alpha_atk, kappa)`, `PolyADAA1` + `C^2` wavefolder, `4x4` spatial tiles | Constant per `SonaTierSpec` |
| `stage3_early_fir_{lite,full,pro,ultra}` | `Arena 0..3` | `32 / +16 / +48 / +32` L+R `f32` taps (`32 / 48 / 96 / 128` cum. taps/ch, `ref[container.dnam.stage3-slices]`) | `256 / 128 / 384 / 256 B` |
| `stage3_kautz_modals_{lite,full,pro,ultra}` | `Arena 0..3` | `2 / 1 / 3 / 2` incremental `128 B` 8-Vec SoA blocks (`8 / 12 / 24 / 32` cum. pairs, `ref[container.dnam.stage3-slices]`) | `256 / 128 / 384 / 256 B` |
| `stage4_tank_{lite,full,pro,ultra}` | `Arena 0..3` | `Quad 0..3` delay lines (`4 / 8 / 12–16 / 16` lines), `PartitionedSchroederDiffuserBank`, `a_diff`, `g_lin`, `V_ceil`, `sigma_knee`, Householder `v`, `Q0.32` LFOs, `fp16`/`f32` ext descriptors | `64..256 B` |
| `varpro_readout_{lite,full,pro,ultra}` | `Arena 0..3` | `fp16` `W_readout` (`VarPro` subspace) + direct Mid/Side accumulators + calibrated `tier_gain_trim_lin` | Constant per `SonaTierSpec` |
| `control_projection_lite` / `_ultra_delta` | `Arena 0` / `Arena 3` | Self-contained `Arena 0` `ControlClampDescriptorWire` base + `CP1` stream (`ref[container.dnam.cp1-profile]`, padded to `64 B`) | `64 B * ceil(...)` |
| `control_interaction_lite` / `_ultra_delta` | `Arena 0` / `Arena 3` | `[4, 8]` base (`128 B` in `Arena 0`) / `[3, 4, 8]` Ultra delta (`384 B` in `Arena 3`) FM matrices (`0 B` if `< 2` controls) | `0 / 128 / 384 B` |
| `control_dispatch_lite` / `_ultra_delta` | `Arena 0` / `Arena 3` | `W_ctrl^base[16, 16]` (`1,024 B` in `Arena 0`) / `W_ctrl,Delta^Ultra[4, 16]` (`256 B` in `Arena 3`) | `1,024 / 256 B` |

---

### 2.4 Tier-Local Projection & Stage III Slice Layout

#### Stage III Separate 64B-Aligned Early FIR & Kautz Modal Slices

`r[container.dnam.stage3-slices]`

Each tier stores its Stage III early FIR taps and shared-pole Kautz modal blocks in separate, non-overlapping 64-byte-aligned slices ([`../dsp/op_m_modal_kautz.md` §4.1](../dsp/op_m_modal_kautz.md#41-stage-iii-architecture--tier-allocation)):
- **Early FIR Slice (`stage3_early_fir`, `32 / 48 / 96 / 128` cumulative taps/ch):** Left `f32` taps followed by Right `f32` taps (`+32` taps/ch = `256 B` in `Lite`, `+16` = `128 B` in `Full`, `+48` = `384 B` in `Pro`, `+32` = `256 B` in `Ultra`; `1,024 B` cumulative across `Arena 0..3`).
- **Kautz Modal Slice (`stage3_kautz_modals`, `8 / 12 / 24 / 32` cumulative pole pairs):** Packs `4` shared pole pairs per `128 B` (`2 CL`) block (`2 / 1 / 3 / 2` incremental blocks across `Lite / Full / Pro / Ultra`) as eight consecutive `f32 × 4` SIMD vectors (`[r_low, dr, theta, b_gamma, c_re_l, c_im_l, c_re_r, c_im_r]`):

```text
+-----------------------------------------------------------------------------------+
| STAGE III 64B-ALIGNED SLICE LAYOUT ACROSS ARENA 0..3 (SONA v3.0)                  |
+-----------------------------------------------------------------------------------+
|  Tier    | New Pairs | Early FIR Slice      | Kautz Modal Slice (8-Vec SoA) | Sum |
|  Lite    |     8     | +32 taps/ch = 256 B  |  2 blocks = 256 B (4 CL)      | 512 |
|  Full    |     4     | +16 taps/ch = 128 B  |  1 block  = 128 B (2 CL)      | 256 |
|  Pro     |    12     | +48 taps/ch = 384 B  |  3 blocks = 384 B (6 CL)      | 768 |
|  Ultra   |     8     | +32 taps/ch = 256 B  |  2 blocks = 256 B (4 CL)      | 512 |
|  Total   |    32     | 128 taps/ch = 1,024B |  8 blocks = 1,024 B (16 CL)   |2,048|
+-----------------------------------------------------------------------------------+
```

#### Packed Control Projection Profile

`r[container.dnam.cp1-profile]`

The `CP1` profile places all `Lite..Pro` base control descriptors, `CP1` contributor streams, `U_int`, `W_ctrl`, and `Lite` `W_readout` inside **`Arena 0` (`Arena0LiteWire`)** so `strip_container_to_tier(Lite)` preserves full parametric controls (`REQ-5.6`). Contributors follow `controls_schema` order: each `range` with `lobatto_degree = k > 0` contributes one `k`-lane record (`k = 0` emits no record), and each switch option in `represented_switch_options` contributes one `1`-lane record:

```text
+-----------------------------------------------------------------------------------+
| CP1 CONTRIBUTOR RECORDS & SELF-CONTAINED ARENA 0 CONTROL TENSORS                  |
+-----------------------------------------------------------------------------------+
|  Per-Contributor Record (k active lanes, concatenated in contributor order):      |
|    • control_projection_lite (Arena 0):        main_base[k][8] + emb[k] + v[4]    |
|                                                = (9k + 4) f32 = 4(9k + 4) B       |
|    • control_projection_ultra_delta (Arena 3): main_delta[k][8] + v_delta[4]      |
|                                                = (8k + 4) f32 = 4(8k + 4) B       |
+-----------------------------------------------------------------------------------+
|  Global Control Tensors (64B-aligned, Self-Contained in Arena 0 + Ultra Delta):   |
|    • control_interaction_lite        (Arena 0, [4, 8],      128 B =  2 CL)        |
|    • control_interaction_ultra_delta (Arena 3, [3, 4, 8],   384 B =  6 CL)        |
|    • control_dispatch_lite           (Arena 0, [16, 16],  1,024 B = 16 CL)        |
|    • control_dispatch_ultra_delta    (Arena 3, [4, 16],     256 B =  4 CL)        |
+-----------------------------------------------------------------------------------+
```

When fewer than two distinct physical controls have contributors, both interaction tensors are omitted (`0 B`).

#### Conditional Control Rules

`r[container.dnam.conditional-rules]`

Conditional safety bounds (`conditional_bounds`) and multi-range physical calibration (`calibration.conditional_physical_anchors`) use structured switch conjunctions (`when: [{"control_id": "...", "option_id": "..."}]`) ordered by `controls_schema` rather than delimiter-joined `"switch=option"` strings:

```json
{
  "conditional_bounds": [
    {
      "when": [
        { "control_id": "range", "option_id": "long" },
        { "control_id": "voicing", "option_id": "lead" }
      ],
      "min": 0.2,
      "max": 0.8
    }
  ],
  "calibration": {
    "conditional_physical_anchors": [
      {
        "when": [{ "control_id": "range", "option_id": "long" }],
        "anchors": [10.0, 100.0, 1000.0]
      }
    ]
  }
}
```

All matching `conditional_bounds` entries are intersected (`[L, U - 1]` inward `f32` rank intersection); for calibration, the matching entry with the greatest selector specificity is selected (reporting any specificity tie as ambiguous).

---

## 3. Essential Metadata, Provenance, Presets, `ui_hints` & Open `user_metadata`

`r[container.dnam.metadata-ui-hints]`

1. **Minimal Standardized `metadata`:** `name`, `author`, `description`, `license`, `created_at`, `input_reference_dbfs`, per-tier `slimmable_tiers` (`loudness_lufs`, `output_trim_db`, `validation_esr_db`), and content-addressed `provenance` (`dataset_manifest_sha256`, `config_recipe_sha256`).
2. **Intrinsic Per-Control `ui_hints`:** `section`, `order`, `size` (`"sm" | "md" | "lg"`), `color_hex`, `widget` (`"knob" | "slider" | "trim"` or `"toggle" | "segmented" | "rotary_select" | "footswitch"`), `polarity`, `detents`, `step`, `decimals`, `scale_labels`, `option_labels`, and midpoint-anchored power-skew curve:
   $$\gamma_{\text{skew}} = \frac{\ln(0.5)}{\ln\left(\frac{x_{\text{mid}} - x_{\min}}{x_{\max} - x_{\min}}\right)}, \qquad x(t) = x_{\min} + (x_{\max} - x_{\min}) t^{1 / \gamma_{\text{skew}}}, \qquad t(x) = \left(\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right)^{\gamma_{\text{skew}}}$$
3. **Per-Control Physical Calibration (`controls_schema[*].calibration`):** `role`, `lobatto_physical_anchors`, `conditional_physical_anchors`, and `tempo_sync_default_subdivision`.
4. **Model-Level `io_config.capabilities` & `io_config.leveling`:** `blend_mode`, `default_blend`, `wet_polarity_inverted`, `supports_trails`, `tempo_sync_capable`, `supports_freeze`, and default-control-anchored `rms_dbfs` / `k_weighted_lufs` auto-unity leveling.

---

## 4. Training Sample-Rate Anchoring & Continuous-Time ZOH Cross-Reference

`r[container.dnam.sample-rate-anchor]`

All `v3.0` `.dnam` models serialize discrete-time parameters at the canonical reference rate `sample_rate_train_hz = 48000` (including Stage III's 16-sample modal onset). Source captures at other sample rates are converted to aligned `48 kHz` working copies during training while preserving source PCM provenance. At runtime, hosts executing at $44.1\text{–}96\text{ kHz}$ or $4\times$ offline bounce apply analytical bilinear frequency prewarping (`TPT-SVF`) and continuous-time ZOH transformations per **[`../dsp/router_and_stages_topology.md` §9](../dsp/router_and_stages_topology.md#9-continuous-time-zero-order-hold-zoh-sample-rate-adaptability-44196-khz--4-bounce)**.

---

## 5. Canonical `.dnam` JSON Header Schema

`r[container.dnam.jcs-header-schema]`

The RFC 8785 JCS immutable header locks `"version": "3.0.0"`, `"container_kind": "dnam"` (or `"preview"`), `"architecture": "sona"`, `"sample_rate_train_hz": 48000`, `metadata`, `io_config`, `controls_schema`, `delay_histories`, `slimmable_tiers`, and `tensor_directory`; mutable presets and `*_user_metadata` reside in the `DMU1`/`DME1` suffix (`§1.2`), and cryptographic signatures reside in the authentication envelope (`§6`).

---

## 6. Marketplace Security, Signatures & Hardware DRM Cross-Reference

`r[container.dnam.security-crossref]`

| Container Security Mechanism | Wire / Envelope Realization | Canonical Specification |
| :--- | :--- | :--- |
| **Preset-Safe Canonical Digest** | SHA-256 of the projected RFC 8785 JCS immutable header (`immutable_header_sha256`). | [`../security/provenance_and_enclave_drm.md` §2.1](../security/provenance_and_enclave_drm.md#21-preset-safe-canonical-header-digest-rfc-8785-jcs) |
| **Creator Exclusivity Policy** | `metadata.distribution_policy` (`allowed_protection_profiles`, `allowed_oem_vendors`). | [`../security/provenance_and_enclave_drm.md` §2.2](../security/provenance_and_enclave_drm.md#22-creator-signed-distributionpolicy--protection-profile-exclusivity) |
| **2-Certificate `Ed25519` Delegation & 4-Tier Stripping** | `DAU1` (Open) and `DAI1` (Protected) envelopes binding creator and buyer 4-tier prefix-digest lists + self-contained `Arena 0` controls. | [`../security/provenance_and_enclave_drm.md` §2.3](../security/provenance_and_enclave_drm.md#23-two-certificate-ed25519-delegation-chain--4-tier-cumulative-arena-digests) |
| **OEM Hardware Enclave & `.dnam-seal` (`b"DNSL"`)** | `DSA1` outer manifest + RFC 9180 HPKE recipient slots + `ChaCha20-Poly1305` tier records (`zerocopy` `v3.0`). | [`../security/provenance_and_enclave_drm.md` §3](../security/provenance_and_enclave_drm.md#3-pillar-2-hardware-enclave--oem-drm-oemdevicecertificate-attestation--dnam-seal-bdnsl-wrapper) |
| **Dual-Mechanism Buyer Watermarking (`Arena 0..3`)** | `0-ULP` odd-symmetric `SONA` (`PolyADAA1`/`ParaLRU`) hidden-lane sign-flip gauge + Stage III Kautz gauges + EMR-masked Tardos code | [`../security/watermarking_and_marketplace.md` §1](../security/watermarking_and_marketplace.md#1-pillar-3-forensic-traitor-tracing--02-ms-dual-mechanism-buyer-watermarking) |
| **3 Container Protection Profiles** | `DnamProtectionProfile::{Open, DeviceAllowlisted, EnclaveSealed}`. | [`../security/provenance_and_enclave_drm.md` §4](../security/provenance_and_enclave_drm.md#4-three-container-protection-profiles-dnamprotectionprofile--loader-auto-detection) |
