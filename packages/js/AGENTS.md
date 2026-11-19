# `packages/js` — TypeScript 7, Node.js N-API v10 & WebAudio SDK Routing Guide

**`@dnam-project/dnam`** (`dnam-js` crate) wraps `packages/engine` (`rust/` -> N-API v10 + `wasm32` SIMD128) and the TypeScript 7 / WebAudio SDK (`src/`), per [`typescript_sdk.md`](../../docs/specs/sdks/typescript_sdk.md) and [`ADR-0013`](../../docs/adr/0013-typescript-sdk-architecture-and-dual-target-runtime.md).

## 1. Module Routing Map

- [`rust/`](rust/lib.rs): Target-agnostic core (`rust/core/`) + N-API v10 (`rust/napi/`) and WASM SIMD128 (`rust/wasm/`).
- [`src/index.ts`](src/index.ts), [`src/node.ts`](src/node.ts), [`src/browser.ts`](src/browser.ts), [`src/internal/native/`](src/internal/native/driver.ts): `NapiDriver` and `WasmDriver` routing.
- [`src/model/`](src/model/model.ts): `Model<TParams>`, `GramMatrix` (`uplo='U'`), and `crc32c`.
- [`src/engine/`](src/engine/engine.ts): `Engine<TParams>` (`process`, `processAsync`, `render`, `bounce`, `stream`, `bypass`/`freeze`, `telemetry`).
- [`src/parameters/`](src/parameters/index.ts), [`src/tempo/`](src/tempo/index.ts): `Parameter`, `ParameterMap`, `ControlSpace`, `clampParameters`, `defineModelTypes`, `TempoController`.
- [`src/schema/`](src/schema/index.ts), [`src/errors.ts`](src/errors.ts): Rust-backed `decodeHeader`/`encodeHeader`, `headerToJcs`, `DnamError`.
- [`src/provenance/`](src/provenance/index.ts), [`src/dsp/`](src/dsp/index.ts): `verify`, `signCreatorHeader`, `signMarketplaceReceipt`, `seal`/`unseal`, watermarking, `SONA` DSP kernels (`Phase 6`: `navigator.gpu` WebGPU + `v3.0` `"sona"` `zerocopy`).
- [`src/web-audio/`](src/web-audio/index.ts): `DnamAudioNode<TParams>`, `DnamAudioWorkletProcessor`, `WorkletSharedState` (`SharedArrayBuffer`).

## 2. Guides & Verification

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — Zero-duplication rule, dual-target Rust build (`moon run js:build`), `fast-check` testing, and `moon check js`.
2. [`README.md`](README.md) & [`docs/specs/sdks/typescript_sdk.md`](../../docs/specs/sdks/typescript_sdk.md) — API reference, subpath exports, and benchmarks.

