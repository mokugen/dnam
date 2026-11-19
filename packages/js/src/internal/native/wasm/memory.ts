/**
 * WebAssembly `SIMD128` linear-memory binary patching (`Section 2` `"env"."memory"` limits)
 * with decomposed helpers (`ref[sdk.ts.api.sync-init]`). Zero `node:*` dependencies.
 *
 * @module
 */

/**
 * Minimum initial 64 KiB WebAssembly linear-memory pages enforced by the SDK (`18` pages = `1.125 MiB`).
 */
export const WASM_MIN_INITIAL_PAGES = 18;

/**
 * Default initial 64 KiB WebAssembly linear-memory pages (`18` pages = `1.125 MiB`).
 */
export const WASM_DEFAULT_INITIAL_PAGES = 18;

/**
 * Default maximum 64 KiB WebAssembly linear-memory pages (`4,096` pages = `256 MiB`).
 */
export const WASM_DEFAULT_MAXIMUM_PAGES = 4096;

/**
 * Architectural `wasm32` ceiling in 64 KiB pages (`65,536` pages = `4 GiB`).
 */
export const WASM_MAX_PAGES = 65536;

const ENV_MEMORY_IMPORT_PREFIX = [
  0x01, 0x03, 0x65, 0x6e, 0x76, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02,
] as const;

function encodeLeb128U32(val: number): number[] {
  const out: number[] = [];
  let v = val >>> 0;
  do {
    let b = v & 0x7f;
    v >>>= 7;
    if (v !== 0) {
      b |= 0x80;
    }
    out.push(b);
  } while (v !== 0);
  return out;
}

function readLeb128U32At(bytes: Uint8Array, startPos: number): readonly [number, number] {
  let res = 0;
  let shift = 0;
  let pos = startPos;
  while (pos < bytes.length) {
    const b = bytes[pos++] ?? 0;
    res |= (b & 0x7f) << shift;
    if ((b & 0x80) === 0) {
      return [res >>> 0, pos] as const;
    }
    shift += 7;
  }
  return [res >>> 0, pos] as const;
}

function hasWasmMagic(bytes: Uint8Array): boolean {
  return (
    bytes.length >= 8 &&
    bytes[0] === 0x00 &&
    bytes[1] === 0x61 &&
    bytes[2] === 0x73 &&
    bytes[3] === 0x6d
  );
}

function matchesEnvMemoryPrefix(bytes: Uint8Array, payloadStart: number): boolean {
  for (let i = 0; i < ENV_MEMORY_IMPORT_PREFIX.length; i++) {
    if (bytes[payloadStart + i] !== ENV_MEMORY_IMPORT_PREFIX[i]) {
      return false;
    }
  }
  return true;
}

function copyBytesOwned(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy;
}

function resolveLimitBytes(
  minPages: number,
  shared: boolean,
  maxPages: number | null,
): readonly number[] {
  const resolvedMax =
    maxPages !== null
      ? Math.min(WASM_MAX_PAGES, Math.max(minPages, Math.trunc(maxPages)))
      : shared
        ? WASM_MAX_PAGES
        : null;
  if (resolvedMax === null) {
    return [0x00, ...encodeLeb128U32(minPages)];
  }
  return [shared ? 0x03 : 0x01, ...encodeLeb128U32(minPages), ...encodeLeb128U32(resolvedMax)];
}

function rewriteMemoryImportSection(
  wasmBytes: Uint8Array,
  secStart: number,
  payloadStart: number,
  secEnd: number,
  shared: boolean,
  maxPages: number | null,
): Uint8Array<ArrayBuffer> {
  let limitPos = payloadStart + ENV_MEMORY_IMPORT_PREFIX.length;
  const oldFlags = wasmBytes[limitPos++] ?? 0;
  const [minPages, afterMin] = readLeb128U32At(wasmBytes, limitPos);
  limitPos = afterMin;
  if ((oldFlags & 0x01) !== 0) {
    limitPos = readLeb128U32At(wasmBytes, limitPos)[1];
  }
  const restPayload = wasmBytes.subarray(limitPos, secEnd);
  const newLimits = resolveLimitBytes(minPages, shared, maxPages);
  const newPayloadLen = ENV_MEMORY_IMPORT_PREFIX.length + newLimits.length + restPayload.length;
  const secLenBytes = encodeLeb128U32(newPayloadLen);
  const out = new Uint8Array(
    secStart + 1 + secLenBytes.length + newPayloadLen + (wasmBytes.length - secEnd),
  );
  let w = 0;
  out.set(wasmBytes.subarray(0, secStart), w);
  w += secStart;
  out[w++] = 2;
  out.set(secLenBytes, w);
  w += secLenBytes.length;
  out.set(wasmBytes.subarray(payloadStart, payloadStart + ENV_MEMORY_IMPORT_PREFIX.length), w);
  w += ENV_MEMORY_IMPORT_PREFIX.length;
  out.set(newLimits, w);
  w += newLimits.length;
  out.set(restPayload, w);
  w += restPayload.length;
  out.set(wasmBytes.subarray(secEnd), w);
  return out;
}

/**
 * Returns a `Uint8Array` copy of `wasmBytes` whose Section 2 (`Import`) `"env"."memory"` descriptor
 * matches `shared` and `maxPages`.
 */
export function patchWasmMemoryImport(
  wasmBytes: Uint8Array,
  shared: boolean,
  maxPages: number | null = WASM_DEFAULT_MAXIMUM_PAGES,
): Uint8Array<ArrayBuffer> {
  if (!hasWasmMagic(wasmBytes)) {
    return copyBytesOwned(wasmBytes);
  }
  let pos = 8;
  while (pos < wasmBytes.length) {
    const secStart = pos;
    const secId = wasmBytes[pos++] ?? 0;
    const [secLen, payloadStart] = readLeb128U32At(wasmBytes, pos);
    const secEnd = payloadStart + secLen;
    if (secEnd > wasmBytes.length) {
      break;
    }
    if (
      secId === 2 &&
      secLen >= ENV_MEMORY_IMPORT_PREFIX.length + 2 &&
      matchesEnvMemoryPrefix(wasmBytes, payloadStart)
    ) {
      return rewriteMemoryImportSection(
        wasmBytes,
        secStart,
        payloadStart,
        secEnd,
        shared,
        maxPages,
      );
    }
    pos = secEnd;
  }
  return copyBytesOwned(wasmBytes);
}
