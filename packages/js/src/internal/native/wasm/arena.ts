/**
 * Linear-memory scratch arena, typed-array view cache, and UTF-8 / `SharedArrayBuffer` codecs for
 * `WasmDriver` (`ref[sdk.ts.runtime.dual-target-drivers]`). Zero `node:*` dependencies.
 *
 * @module
 */

import { fromNativeError } from "../../../errors.js";
import type { RawWasmExports } from "./exports.js";

const sharedEncoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : undefined;
const sharedDecoder = typeof TextDecoder !== "undefined" ? new TextDecoder() : undefined;

export function encodeUtf8(str: string): Uint8Array {
  return sharedEncoder ? sharedEncoder.encode(str) : Uint8Array.from(str, (c) => c.charCodeAt(0));
}

export function isSharedBuffer(buf: unknown): boolean {
  return typeof SharedArrayBuffer !== "undefined" && buf instanceof SharedArrayBuffer;
}

export function decodeUtf8(bytes: Uint8Array): string {
  if (!sharedDecoder) {
    let out = "";
    for (let i = 0; i < bytes.length; i++) {
      out += String.fromCharCode(bytes[i] ?? 0);
    }
    return out;
  }
  const safeBytes = isSharedBuffer(bytes.buffer) ? bytes.slice() : bytes;
  return sharedDecoder.decode(safeBytes);
}

export class WasmMemoryArena {
  readonly exports: RawWasmExports;
  readonly memory: WebAssembly.Memory;
  #cachedBuffer: ArrayBufferLike | undefined;
  #u8View: Uint8Array | undefined;
  #f32View: Float32Array | undefined;
  #f64View: Float64Array | undefined;
  #u32View: Uint32Array | undefined;

  // Persistent small scratch region for telemetry (56 f64 = 448 B) + control values (256 f32 = 1024 B)
  readonly smallF64Ptr: number;
  readonly smallF32Ptr: number;
  readonly smallU32Ptr: number;

  // Persistent dynamically grown audio scratch region (4 * maxFrames * 4 bytes)
  #audioPtr = 0;
  #audioCapacityFrames = 0;

  constructor(exports: RawWasmExports, memory?: WebAssembly.Memory) {
    this.exports = exports;
    this.memory = memory ?? exports.memory;
    this.smallF64Ptr = this.alloc(64 * 8);
    this.smallF32Ptr = this.alloc(256 * 4);
    this.smallU32Ptr = this.alloc(256 * 4);
    this.#ensureAudioScratch(4096);
  }

  alloc(size: number): number {
    const ptr = this.exports.dnam_wasm_alloc(size, 64);
    if (ptr === 0) {
      this.throwLastError();
    }
    return ptr;
  }

  #refreshViews(): void {
    const buf = this.memory.buffer;
    if (this.#cachedBuffer !== buf || this.#u8View?.byteLength !== buf.byteLength) {
      this.#cachedBuffer = buf;
      this.#u8View = new Uint8Array(buf);
      this.#f32View = new Float32Array(buf);
      this.#f64View = new Float64Array(buf);
      this.#u32View = new Uint32Array(buf);
    }
  }

  u8(): Uint8Array {
    this.#refreshViews();
    return this.#u8View as Uint8Array;
  }

  f32(): Float32Array {
    this.#refreshViews();
    return this.#f32View as Float32Array;
  }

  f64(): Float64Array {
    this.#refreshViews();
    return this.#f64View as Float64Array;
  }

  u32(): Uint32Array {
    this.#refreshViews();
    return this.#u32View as Uint32Array;
  }

  ensureAudioScratch(frames: number): {
    readonly ch0InPtr: number;
    readonly ch1InPtr: number;
    readonly ch0OutPtr: number;
    readonly ch1OutPtr: number;
  } {
    this.#ensureAudioScratch(frames);
    const strideBytes = this.#audioCapacityFrames * 4;
    return {
      ch0InPtr: this.#audioPtr,
      ch1InPtr: this.#audioPtr + strideBytes,
      ch0OutPtr: this.#audioPtr + strideBytes * 2,
      ch1OutPtr: this.#audioPtr + strideBytes * 3,
    };
  }

  #ensureAudioScratch(frames: number): void {
    if (frames <= this.#audioCapacityFrames && this.#audioPtr !== 0) {
      return;
    }
    const nextCap = Math.max(4096, 1 << Math.ceil(Math.log2(Math.max(1, frames))));
    const nextPtr = this.alloc(nextCap * 16);
    if (this.#audioPtr !== 0) {
      this.exports.dnam_wasm_dealloc(this.#audioPtr, this.#audioCapacityFrames * 16, 64);
    }
    this.#audioPtr = nextPtr;
    this.#audioCapacityFrames = nextCap;
  }

  throwLastError(): never {
    const ptr = this.exports.dnam_wasm_last_error_ptr();
    const len = this.exports.dnam_wasm_last_error_len();
    const msg =
      len > 0
        ? decodeUtf8(this.u8().subarray(ptr, ptr + len))
        : "[DNAM-VALIDATION] WASM call failed";
    throw fromNativeError(new Error(msg));
  }

  checkStatus(rc: number): number {
    if (rc < 0) {
      this.throwLastError();
    }
    return rc;
  }

  readResultBytes(): Uint8Array {
    const ptr = this.exports.dnam_wasm_result_ptr();
    const len = this.exports.dnam_wasm_result_len();
    return this.u8().slice(ptr, ptr + len);
  }

  readResultString(): string {
    const ptr = this.exports.dnam_wasm_result_ptr();
    const len = this.exports.dnam_wasm_result_len();
    return decodeUtf8(this.u8().subarray(ptr, ptr + len));
  }

  readArenaBytes(): Uint8Array {
    const ptr = this.exports.dnam_wasm_arena_ptr();
    const len = this.exports.dnam_wasm_arena_len();
    return this.u8().slice(ptr, ptr + len);
  }

  withBytes<T>(data: Uint8Array | undefined, fn: (ptr: number, len: number) => T): T {
    if (!data || data.length === 0) {
      return fn(0, 0);
    }
    const ptr = this.alloc(data.length);
    try {
      this.u8().set(data, ptr);
      return fn(ptr, data.length);
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, data.length, 64);
    }
  }

  withString<T>(str: string | undefined, fn: (ptr: number, len: number) => T): T {
    if (!str || str.length === 0) {
      return fn(0, 0);
    }
    return this.withBytes(encodeUtf8(str), fn);
  }

  withF32Slice<T>(data: Float32Array, fn: (ptr: number, len: number) => T): T {
    const byteLen = Math.max(4, data.byteLength);
    const ptr = this.alloc(byteLen);
    try {
      this.f32().set(data, ptr >>> 2);
      return fn(ptr, data.length);
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, byteLen, 64);
    }
  }

  withF64Slice<T>(data: Float64Array, fn: (ptr: number, len: number) => T): T {
    const byteLen = Math.max(8, data.byteLength);
    const ptr = this.alloc(byteLen);
    try {
      this.f64().set(data, ptr >>> 3);
      return fn(ptr, data.length);
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, byteLen, 64);
    }
  }

  withF64Out<T>(
    len: number,
    fn: (ptr: number) => T,
  ): { readonly result: T; readonly out: Float64Array } {
    const byteLen = Math.max(8, len * 8);
    const ptr = this.alloc(byteLen);
    try {
      const result = fn(ptr);
      const start = ptr >>> 3;
      const out = this.f64().slice(start, start + len);
      return { result, out };
    } finally {
      this.exports.dnam_wasm_dealloc(ptr, byteLen, 64);
    }
  }
}
