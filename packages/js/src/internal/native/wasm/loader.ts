/**
 * WebAssembly memory creation, Section 2 (`Import`) patching helpers, and `WebAssembly.Module`
 * compilation/instantiation helpers (`ref[sdk.ts.api.sync-init]`).
 *
 * Zero `node:*` dependencies so it runs identically in browsers, Node.js, and
 * `AudioWorkletGlobalScope`.
 *
 * @module
 */

import { DnamError } from "../../../errors.js";
import { coerceBytes } from "../../coerce.js";
import type { InitWasmOptions, WasmMemoryOptions } from "../wasm.js";
import { isSharedBuffer } from "./arena.js";
import type { RawWasmExports } from "./exports.js";
import {
  patchWasmMemoryImport,
  WASM_DEFAULT_INITIAL_PAGES,
  WASM_DEFAULT_MAXIMUM_PAGES,
  WASM_MAX_PAGES,
  WASM_MIN_INITIAL_PAGES,
} from "./memory.js";

let cachedWasmBytes: Uint8Array<ArrayBuffer> | undefined;
let cachedWasmModule: WebAssembly.Module | undefined;
let cachedSharedWasmModule: WebAssembly.Module | undefined;

export function isSharedWasmMemorySupportedImpl(): boolean {
  if (typeof SharedArrayBuffer === "undefined" || typeof Atomics === "undefined") {
    return false;
  }
  if (
    typeof globalThis !== "undefined" &&
    "crossOriginIsolated" in globalThis &&
    (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === false
  ) {
    return false;
  }
  try {
    const mem = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true });
    return isSharedBuffer(mem.buffer);
  } catch {
    return false;
  }
}

export function createWasmMemoryImpl(options?: WasmMemoryOptions): WebAssembly.Memory {
  if (options?.memory instanceof WebAssembly.Memory) {
    return options.memory;
  }
  const initial = Math.min(
    WASM_MAX_PAGES,
    Math.max(
      WASM_MIN_INITIAL_PAGES,
      Math.trunc(options?.initialPages ?? WASM_DEFAULT_INITIAL_PAGES),
    ),
  );
  const maximum = Math.min(
    WASM_MAX_PAGES,
    Math.max(initial, Math.trunc(options?.maximumPages ?? WASM_DEFAULT_MAXIMUM_PAGES)),
  );
  if (options?.shared && isSharedWasmMemorySupportedImpl()) {
    try {
      return new WebAssembly.Memory({ initial, maximum, shared: true });
    } catch {
      // Fall back cleanly to non-shared WebAssembly.Memory if disallowed by runtime policy.
    }
  }
  return new WebAssembly.Memory({ initial, maximum });
}

export function setCachedWasmBytesImpl(bytes: Uint8Array): void {
  cachedWasmBytes = patchWasmMemoryImport(bytes, false, WASM_DEFAULT_MAXIMUM_PAGES);
}

function shouldRequestSharedMemory(options?: WasmMemoryOptions): boolean {
  if (options?.memory instanceof WebAssembly.Memory) {
    return isSharedBuffer(options.memory.buffer);
  }
  return Boolean(options?.shared) && isSharedWasmMemorySupportedImpl();
}

function shouldUseUnboundedMemory(options?: WasmMemoryOptions): boolean {
  return (
    options?.memory instanceof WebAssembly.Memory ||
    (options?.maximumPages !== undefined && options.maximumPages > WASM_DEFAULT_MAXIMUM_PAGES)
  );
}

function cacheCompiledModule(mod: WebAssembly.Module, shared: boolean): WebAssembly.Module {
  if (shared) {
    cachedSharedWasmModule = mod;
  } else {
    cachedWasmModule = mod;
  }
  return mod;
}

export function getCompiledWasmModuleImpl(shared = false): WebAssembly.Module | undefined {
  const existing = shared ? cachedSharedWasmModule : cachedWasmModule;
  if (existing || !cachedWasmBytes) {
    return existing;
  }
  try {
    return cacheCompiledModule(
      new WebAssembly.Module(
        patchWasmMemoryImport(cachedWasmBytes, shared, WASM_DEFAULT_MAXIMUM_PAGES),
      ),
      shared,
    );
  } catch {
    // Browser main threads disallow synchronous compilation > 4 KiB; use await compileWasmModule().
    return undefined;
  }
}

export interface InstantiatedWasmResult {
  readonly exports: RawWasmExports;
  readonly memory: WebAssembly.Memory;
}

function tryInstantiateCompatModule(
  memory: WebAssembly.Memory,
  wantShared: boolean,
  options?: WasmMemoryOptions,
): InstantiatedWasmResult | undefined {
  const explicit =
    options?.shared !== undefined ||
    options?.memory !== undefined ||
    (options?.maximumPages !== undefined && options.maximumPages > WASM_DEFAULT_MAXIMUM_PAGES);
  if (!explicit || !cachedWasmBytes) {
    return undefined;
  }
  const useUnbounded = shouldUseUnboundedMemory(options);
  try {
    const compatModule = new WebAssembly.Module(
      patchWasmMemoryImport(
        cachedWasmBytes,
        wantShared,
        useUnbounded ? null : WASM_DEFAULT_MAXIMUM_PAGES,
      ),
    );
    const instance = new WebAssembly.Instance(compatModule, { env: { memory } });
    if (!useUnbounded) {
      cacheCompiledModule(compatModule, wantShared);
    }
    return { exports: instance.exports as RawWasmExports, memory };
  } catch {
    return undefined;
  }
}

function tryInstantiateAltMemory(
  module: WebAssembly.Module,
  wantShared: boolean,
  options?: WasmMemoryOptions,
): InstantiatedWasmResult | undefined {
  if (options?.memory !== undefined) {
    return undefined;
  }
  try {
    const altMemory = createWasmMemoryImpl({
      ...(options?.initialPages !== undefined ? { initialPages: options.initialPages } : {}),
      ...(options?.maximumPages !== undefined ? { maximumPages: options.maximumPages } : {}),
      shared: !wantShared,
    });
    const instance = new WebAssembly.Instance(module, { env: { memory: altMemory } });
    cacheCompiledModule(module, isSharedBuffer(altMemory.buffer));
    return { exports: instance.exports as RawWasmExports, memory: altMemory };
  } catch {
    return undefined;
  }
}

export function instantiateWasmImpl(
  module: WebAssembly.Module,
  options?: WasmMemoryOptions,
): InstantiatedWasmResult {
  const memory = createWasmMemoryImpl(options);
  const wantShared = isSharedBuffer(memory.buffer);
  try {
    const instance = new WebAssembly.Instance(module, { env: { memory } });
    cacheCompiledModule(module, wantShared);
    return { exports: instance.exports as RawWasmExports, memory };
  } catch (err) {
    const compat = tryInstantiateCompatModule(memory, wantShared, options);
    if (compat) {
      return compat;
    }
    const alt = tryInstantiateAltMemory(module, wantShared, options);
    if (alt) {
      return alt;
    }
    throw err;
  }
}

function resolveDefaultWasmUrl(callerMetaUrl: string): URL {
  if (callerMetaUrl.includes("/src/internal/native/")) {
    return new URL(/* @vite-ignore */ "../../../dist/wasm/debug/dnam.wasm", callerMetaUrl);
  }
  return new URL(/* @vite-ignore */ "./wasm/release/dnam.wasm", callerMetaUrl);
}

async function compileAndCacheRawBytes(
  rawBytes: Uint8Array,
  wantShared: boolean,
  useUnbounded: boolean,
): Promise<WebAssembly.Module> {
  const importMaxPages = useUnbounded ? null : WASM_DEFAULT_MAXIMUM_PAGES;
  const mod = await WebAssembly.compile(
    patchWasmMemoryImport(rawBytes, wantShared, importMaxPages),
  );
  setCachedWasmBytesImpl(rawBytes);
  return useUnbounded ? mod : cacheCompiledModule(mod, wantShared);
}

export async function compileWasmModuleImpl(
  callerMetaUrl: string,
  options?: InitWasmOptions,
): Promise<WebAssembly.Module> {
  const wantShared = shouldRequestSharedMemory(options);
  const useUnbounded = shouldUseUnboundedMemory(options);
  const importMaxPages = useUnbounded ? null : WASM_DEFAULT_MAXIMUM_PAGES;

  if (options?.module instanceof WebAssembly.Module) {
    return cacheCompiledModule(options.module, wantShared);
  }
  if (options?.bytes !== undefined) {
    return compileAndCacheRawBytes(
      coerceBytes(options.bytes as Uint8Array | ArrayBuffer),
      wantShared,
      useUnbounded,
    );
  }
  if (options?.url === undefined) {
    const existing = useUnbounded
      ? undefined
      : wantShared
        ? cachedSharedWasmModule
        : cachedWasmModule;
    if (existing) {
      return existing;
    }
    if (cachedWasmBytes) {
      const mod = await WebAssembly.compile(
        patchWasmMemoryImport(cachedWasmBytes, wantShared, importMaxPages),
      );
      return useUnbounded ? mod : cacheCompiledModule(mod, wantShared);
    }
  }
  if (typeof fetch !== "function") {
    throw new DnamError(
      "ERR_DNAM_CONTAINER",
      "Unable to fetch dnam.wasm in this environment; pass { module } or { bytes } to initWasm()",
    );
  }
  const targetUrl = options?.url ?? resolveDefaultWasmUrl(callerMetaUrl);
  const resp = await fetch(targetUrl);
  if ("ok" in resp && resp.ok === false) {
    throw new DnamError(
      "ERR_DNAM_CONTAINER",
      `Failed to fetch dnam.wasm from ${String(targetUrl)} (HTTP ${String(resp.status)})`,
    );
  }
  return compileAndCacheRawBytes(
    new Uint8Array(await resp.arrayBuffer()),
    wantShared,
    useUnbounded,
  );
}
