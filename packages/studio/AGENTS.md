# `packages/studio` — Standalone Tauri v2 Desktop Studio Routing Guide

**`dnam-studio` (`@dnam-project/studio`)** owns the standalone Tauri v2 Desktop Studio GUI, native OS audio device enumeration (`cpal`), wait-free SPSC ring-buffer audio I/O, and the isolated IPC bridge over `dnam-engine` (`.dnam` `v3.0` `"sona"` in Phase 6) and `dnam-capture` (`.dnam-proj` + native `gpu-wgpu` `32-bit WebGPU` / `DNPR` telemetry in Phase 6).

## 1. Key Modules

- [`rust/capture.rs`](rust/capture.rs) & [`rust/capture/backends.rs`](rust/capture/backends.rs): `AtomicF32RingBuffer` and `cpal` native audio device enumeration (`CoreAudio` / `ALSA` / `WASAPI`).
- [`rust/ipc.rs`](rust/ipc.rs), [`rust/ipc/paths.rs`](rust/ipc/paths.rs), [`rust/ipc/deep_link.rs`](rust/ipc/deep_link.rs) & [`rust/main.rs`](rust/main.rs): Tauri v2 `StudioIpcHandler` (`.dnam-proj` inspection, isolated IPC allowlisting, `dnam://` deep-link routing, private exports) and standalone `dnam-studio` desktop binary.
- [`src/ipc.ts`](src/ipc.ts), [`src/host.ts`](src/host.ts), [`src/ui/StudioApp.tsx`](src/ui/StudioApp.tsx) & [`src/ui/studio_store.ts`](src/ui/studio_store.ts): TypeScript IPC client, `TauriNativeAudioHost`, and Tauri v2 React 19 UI consuming `@dnam-project/ui/*` leaf subpath imports.

## 2. Guides & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`README.md`](README.md) for Rust 2024 module layout, Tauri v2 IPC isolation rules, and `moon check studio` / `moon run studio:test-e2e` verification commands.
