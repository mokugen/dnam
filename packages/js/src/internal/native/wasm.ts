/**
 * High-performance WebAssembly `SIMD128` driver (`WasmDriver`) and initialization helpers
 * (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.api.sync-init]`).
 *
 * Zero `node:*` dependencies so it runs identically in browsers, Node.js, and
 * `AudioWorkletGlobalScope`.
 *
 * @module
 */

import { backendToCode } from "../coerce.js";
import type {
  CreateNativeEngineOptions,
  NativeControlSpaceHandle,
  NativeDriver,
  NativeEngineHandle,
  NativeValidatedContainerInfo,
} from "./driver.js";
import { isSharedBuffer, WasmMemoryArena } from "./wasm/arena.js";
import {
  wasmBuildDmu1Frame,
  wasmBuildDnamContainer,
  wasmBuildReferenceContainer,
  wasmCanonicalizeJcs,
  wasmComputeZoh,
  wasmCrc32c,
  wasmEval3NodeCalibration,
  wasmEvalSubdivisionPhase,
  wasmInvertTempoCalibration,
  wasmPackUploUF64Into,
  wasmParseDmu1Frame,
  wasmParseSchemaVersion,
  wasmStripContainerToTier,
  wasmStripV2OpenDnamContainer,
  wasmUnpackUploUF64Into,
  wasmUpdateHeaderInPlace,
  wasmValidateAndNormalizeHeader,
  wasmValidateContainer,
  wasmValidateV2OpenDnamLayout,
} from "./wasm/container.js";
import type { RawWasmExports } from "./wasm/exports.js";
import { WasmControlSpaceHandle, WasmEngineHandle } from "./wasm/handles.js";
import {
  wasmCatmullRomHermiteSlice,
  wasmCubicLagrangeWeights,
  wasmCubicLagrangeWeightsSlice,
  wasmEvalChebyshevLfo,
  wasmEvalChebyshevLfoSlice,
  wasmEvalLobattoBasis,
  wasmEvalLobattoBasisSlice,
  wasmEvalRank4Fm,
  wasmEvalRank4FmSlice,
  wasmEvalTrustRegionSoftclipSlice,
  wasmGivensHouseholder8ch,
  wasmGivensHouseholder8chSlice,
  wasmGivensHouseholder16ch,
  wasmGivensHouseholder16chSlice,
  wasmOpMModalSlice,
  wasmOpMModalStep,
  wasmOpTCoerciveStep,
  wasmOpTTankSlice,
  wasmPolyAdaaSlice,
  wasmPolyAdaaStep,
  wasmUnitRotor,
  wasmUnitRotorSlice,
} from "./wasm/kernels.js";
import {
  compileWasmModuleImpl,
  createWasmMemoryImpl,
  getCompiledWasmModuleImpl,
  instantiateWasmImpl,
  isSharedWasmMemorySupportedImpl,
  setCachedWasmBytesImpl,
} from "./wasm/loader.js";
import { patchWasmMemoryImport, WASM_DEFAULT_MAXIMUM_PAGES } from "./wasm/memory.js";
import {
  wasmBuildCreatorSigningMessage,
  wasmBuildDme1Header,
  wasmBuildMarketplaceSigningMessage,
  wasmBuildOemCertV2SigningMessage,
  wasmChacha20XorArena,
  wasmComputePreviewWindow,
  wasmDecodeDau1Envelope,
  wasmEd25519PubkeyHex,
  wasmEd25519SignHex,
  wasmEncodeDau1Envelope,
  wasmEncodeHpkeSlots,
  wasmEncodeRecipientKeyIds,
  wasmEncodeRecordDirectory,
  wasmEnforceDistributionPolicy,
  wasmEvaluateAnticlonePolicy,
  wasmHmacSha256Hex,
  wasmImmutableHeaderSha256,
  wasmParseDme1Plaintext,
  wasmParseHpkeSlots,
  wasmParseRecipientKeyIds,
  wasmParseRecordDirectory,
  wasmSealContainer,
  wasmSha256Hex,
  wasmSignCreatorHeader,
  wasmSignMarketplaceReceipt,
  wasmSignOemDeviceCert,
  wasmUnsealContainer,
  wasmValidatePreviewSwitchOptions,
  wasmValidateRoyaltySplits,
  wasmVerifyContainerProvenance,
  wasmVerifyOemDeviceCert,
  wasmWrapDeviceCekHex,
  wasmX25519PubkeyHex,
} from "./wasm/provenance.js";
import {
  wasmApplyIsometricGaugeWatermark,
  wasmApplyTardosWatermark,
  wasmDeriveWatermarkSeedHex,
  wasmExtractStage3Fingerprint,
  wasmExtractStructuralHash,
  wasmGenerateTardosCode,
  wasmScoreTardosWatermark,
  wasmStructuralCosineSimilarity,
} from "./wasm/watermark.js";

/**
 * Unified WebAssembly `SIMD128` native driver delegating linear-memory transfers to domain modules.
 */
export class WasmDriver implements NativeDriver {
  readonly kind = "wasm" as const;
  readonly #mem: WasmMemoryArena;

  constructor(exports: RawWasmExports, memory?: WebAssembly.Memory) {
    this.#mem = new WasmMemoryArena(exports, memory);
  }

  /** Underlying linear `WebAssembly.Memory` imported into and used by this `WasmDriver` instance. */
  get memory(): WebAssembly.Memory {
    return this.#mem.memory;
  }

  /** Returns `true` when `this.memory` is backed by a `SharedArrayBuffer` (`shared: true`). */
  get isSharedMemory(): boolean {
    return isSharedBuffer(this.#mem.memory.buffer);
  }

  createEngine(options: CreateNativeEngineOptions): NativeEngineHandle {
    const backendCode = backendToCode(options.backend);
    const ptr = this.#mem.withBytes(options.containerBytes, (cPtr, cLen) =>
      this.#mem.exports.dnam_wasm_engine_create(
        cPtr,
        cLen,
        options.tier,
        options.sampleRateHz,
        backendCode,
        options.maxSupportedTier ?? -1,
        options.offlineBounce ? 1 : 0,
        options.maximizeTierOnBounce === false ? 0 : 1,
      ),
    );
    if (ptr === 0) {
      this.#mem.throwLastError();
    }
    return new WasmEngineHandle(this.#mem, ptr);
  }

  createControlSpace(headerJson: string): NativeControlSpaceHandle {
    const ptr = this.#mem.withString(headerJson, (hPtr, hLen) =>
      this.#mem.exports.dnam_wasm_control_space_create(hPtr, hLen),
    );
    if (ptr === 0) {
      this.#mem.throwLastError();
    }
    return new WasmControlSpaceHandle(this.#mem, ptr);
  }

  validateContainer(
    ...args: Parameters<NativeDriver["validateContainer"]>
  ): NativeValidatedContainerInfo {
    return wasmValidateContainer(this.#mem, ...args);
  }

  buildReferenceContainer(
    ...args: Parameters<NativeDriver["buildReferenceContainer"]>
  ): Uint8Array {
    return wasmBuildReferenceContainer(this.#mem, ...args);
  }

  buildDnamContainer(...args: Parameters<NativeDriver["buildDnamContainer"]>): Uint8Array {
    return wasmBuildDnamContainer(this.#mem, ...args);
  }

  updateHeaderInPlace(...args: Parameters<NativeDriver["updateHeaderInPlace"]>): Uint8Array {
    return wasmUpdateHeaderInPlace(this.#mem, ...args);
  }

  stripContainerToTier(...args: Parameters<NativeDriver["stripContainerToTier"]>): Uint8Array {
    return wasmStripContainerToTier(this.#mem, ...args);
  }

  parseSchemaVersion(
    ...args: Parameters<NativeDriver["parseSchemaVersion"]>
  ): readonly [number, number, number, boolean] {
    return wasmParseSchemaVersion(this.#mem, ...args);
  }

  crc32c(...args: Parameters<NativeDriver["crc32c"]>): number {
    return wasmCrc32c(this.#mem, ...args);
  }

  validateAndNormalizeHeader(
    ...args: Parameters<NativeDriver["validateAndNormalizeHeader"]>
  ): string {
    return wasmValidateAndNormalizeHeader(this.#mem, ...args);
  }

  canonicalizeJcs(...args: Parameters<NativeDriver["canonicalizeJcs"]>): string {
    return wasmCanonicalizeJcs(this.#mem, ...args);
  }

  buildDmu1Frame(...args: Parameters<NativeDriver["buildDmu1Frame"]>): Uint8Array {
    return wasmBuildDmu1Frame(this.#mem, ...args);
  }

  parseDmu1Frame(...args: Parameters<NativeDriver["parseDmu1Frame"]>): string {
    return wasmParseDmu1Frame(this.#mem, ...args);
  }

  validateV2OpenDnamLayout(...args: Parameters<NativeDriver["validateV2OpenDnamLayout"]>): string {
    return wasmValidateV2OpenDnamLayout(this.#mem, ...args);
  }

  stripV2OpenDnamContainer(
    ...args: Parameters<NativeDriver["stripV2OpenDnamContainer"]>
  ): Uint8Array {
    return wasmStripV2OpenDnamContainer(this.#mem, ...args);
  }

  uploUPackedLen(nDim: number): number {
    return this.#mem.exports.dnam_wasm_uplo_u_packed_len(nDim);
  }

  packUploUF64Into(...args: Parameters<NativeDriver["packUploUF64Into"]>): void {
    wasmPackUploUF64Into(this.#mem, ...args);
  }

  unpackUploUF64Into(...args: Parameters<NativeDriver["unpackUploUF64Into"]>): void {
    wasmUnpackUploUF64Into(this.#mem, ...args);
  }

  computeZoh(...args: Parameters<NativeDriver["computeZoh"]>): Float64Array {
    return wasmComputeZoh(this.#mem, ...args);
  }

  controlStrideForRate(sampleRateHz: number): number {
    return this.#mem.exports.dnam_wasm_control_stride_for_rate(sampleRateHz);
  }

  controlTickRatio(sampleRateHz: number): number {
    return this.#mem.exports.dnam_wasm_control_tick_ratio(sampleRateHz);
  }

  evalSubdivisionPhase(
    ...args: Parameters<NativeDriver["evalSubdivisionPhase"]>
  ): readonly [number, number, number] {
    return wasmEvalSubdivisionPhase(this.#mem, ...args);
  }

  eval3NodeCalibration(...args: Parameters<NativeDriver["eval3NodeCalibration"]>): number {
    return wasmEval3NodeCalibration(this.#mem, ...args);
  }

  invertTempoCalibration(
    ...args: Parameters<NativeDriver["invertTempoCalibration"]>
  ): Float64Array {
    return wasmInvertTempoCalibration(this.#mem, ...args);
  }

  verifyContainerProvenance(
    ...args: Parameters<NativeDriver["verifyContainerProvenance"]>
  ): string {
    return wasmVerifyContainerProvenance(this.#mem, ...args);
  }

  enforceDistributionPolicy(...args: Parameters<NativeDriver["enforceDistributionPolicy"]>): void {
    wasmEnforceDistributionPolicy(this.#mem, ...args);
  }

  evaluateAnticlonePolicy(...args: Parameters<NativeDriver["evaluateAnticlonePolicy"]>): string {
    return wasmEvaluateAnticlonePolicy(this.#mem, ...args);
  }

  immutableHeaderSha256(...args: Parameters<NativeDriver["immutableHeaderSha256"]>): string {
    return wasmImmutableHeaderSha256(this.#mem, ...args);
  }

  sha256Hex(...args: Parameters<NativeDriver["sha256Hex"]>): string {
    return wasmSha256Hex(this.#mem, ...args);
  }

  hmacSha256Hex(...args: Parameters<NativeDriver["hmacSha256Hex"]>): string {
    return wasmHmacSha256Hex(this.#mem, ...args);
  }

  ed25519PubkeyHex(...args: Parameters<NativeDriver["ed25519PubkeyHex"]>): string {
    return wasmEd25519PubkeyHex(this.#mem, ...args);
  }

  ed25519SignHex(...args: Parameters<NativeDriver["ed25519SignHex"]>): string {
    return wasmEd25519SignHex(this.#mem, ...args);
  }

  x25519PubkeyHex(...args: Parameters<NativeDriver["x25519PubkeyHex"]>): string {
    return wasmX25519PubkeyHex(this.#mem, ...args);
  }

  buildCreatorSigningMessage(
    ...args: Parameters<NativeDriver["buildCreatorSigningMessage"]>
  ): Uint8Array {
    return wasmBuildCreatorSigningMessage(this.#mem, ...args);
  }

  buildMarketplaceSigningMessage(
    ...args: Parameters<NativeDriver["buildMarketplaceSigningMessage"]>
  ): Uint8Array {
    return wasmBuildMarketplaceSigningMessage(this.#mem, ...args);
  }

  signCreatorHeader(...args: Parameters<NativeDriver["signCreatorHeader"]>): string {
    return wasmSignCreatorHeader(this.#mem, ...args);
  }

  signMarketplaceReceipt(...args: Parameters<NativeDriver["signMarketplaceReceipt"]>): string {
    return wasmSignMarketplaceReceipt(this.#mem, ...args);
  }

  signOemDeviceCert(...args: Parameters<NativeDriver["signOemDeviceCert"]>): string {
    return wasmSignOemDeviceCert(this.#mem, ...args);
  }

  verifyOemDeviceCert(...args: Parameters<NativeDriver["verifyOemDeviceCert"]>): string {
    return wasmVerifyOemDeviceCert(this.#mem, ...args);
  }

  buildOemCertV2SigningMessage(
    ...args: Parameters<NativeDriver["buildOemCertV2SigningMessage"]>
  ): Uint8Array {
    return wasmBuildOemCertV2SigningMessage(this.#mem, ...args);
  }

  chacha20XorArena(...args: Parameters<NativeDriver["chacha20XorArena"]>): Uint8Array {
    return wasmChacha20XorArena(this.#mem, ...args);
  }

  wrapDeviceCekHex(...args: Parameters<NativeDriver["wrapDeviceCekHex"]>): string {
    return wasmWrapDeviceCekHex(this.#mem, ...args);
  }

  sealContainer(...args: Parameters<NativeDriver["sealContainer"]>): Uint8Array {
    return wasmSealContainer(this.#mem, ...args);
  }

  unsealContainer(...args: Parameters<NativeDriver["unsealContainer"]>): Uint8Array {
    return wasmUnsealContainer(this.#mem, ...args);
  }

  encodeDau1Envelope(...args: Parameters<NativeDriver["encodeDau1Envelope"]>): Uint8Array {
    return wasmEncodeDau1Envelope(this.#mem, ...args);
  }

  decodeDau1Envelope(...args: Parameters<NativeDriver["decodeDau1Envelope"]>): string {
    return wasmDecodeDau1Envelope(this.#mem, ...args);
  }

  buildDme1Header(...args: Parameters<NativeDriver["buildDme1Header"]>): Uint8Array {
    return wasmBuildDme1Header(this.#mem, ...args);
  }

  parseDme1Plaintext(...args: Parameters<NativeDriver["parseDme1Plaintext"]>): string {
    return wasmParseDme1Plaintext(this.#mem, ...args);
  }

  encodeRecipientKeyIds(...args: Parameters<NativeDriver["encodeRecipientKeyIds"]>): Uint8Array {
    return wasmEncodeRecipientKeyIds(this.#mem, ...args);
  }

  parseRecipientKeyIds(
    ...args: Parameters<NativeDriver["parseRecipientKeyIds"]>
  ): readonly string[] {
    return wasmParseRecipientKeyIds(this.#mem, ...args);
  }

  encodeHpkeSlots(...args: Parameters<NativeDriver["encodeHpkeSlots"]>): Uint8Array {
    return wasmEncodeHpkeSlots(this.#mem, ...args);
  }

  parseHpkeSlots(...args: Parameters<NativeDriver["parseHpkeSlots"]>): string {
    return wasmParseHpkeSlots(this.#mem, ...args);
  }

  encodeRecordDirectory(...args: Parameters<NativeDriver["encodeRecordDirectory"]>): Uint8Array {
    return wasmEncodeRecordDirectory(this.#mem, ...args);
  }

  parseRecordDirectory(...args: Parameters<NativeDriver["parseRecordDirectory"]>): string {
    return wasmParseRecordDirectory(this.#mem, ...args);
  }

  computePreviewWindow(
    ...args: Parameters<NativeDriver["computePreviewWindow"]>
  ): readonly [number, number] {
    return wasmComputePreviewWindow(this.#mem, ...args);
  }

  validatePreviewSwitchOptions(
    ...args: Parameters<NativeDriver["validatePreviewSwitchOptions"]>
  ): readonly [string, string] {
    return wasmValidatePreviewSwitchOptions(this.#mem, ...args);
  }

  validateRoyaltySplits(...args: Parameters<NativeDriver["validateRoyaltySplits"]>): void {
    wasmValidateRoyaltySplits(this.#mem, ...args);
  }

  applyIsometricGaugeWatermark(
    ...args: Parameters<NativeDriver["applyIsometricGaugeWatermark"]>
  ): Uint8Array {
    return wasmApplyIsometricGaugeWatermark(this.#mem, ...args);
  }

  generateTardosCode(...args: Parameters<NativeDriver["generateTardosCode"]>): Float32Array {
    return wasmGenerateTardosCode(this.#mem, ...args);
  }

  applyTardosWatermark(...args: Parameters<NativeDriver["applyTardosWatermark"]>): Uint8Array {
    return wasmApplyTardosWatermark(this.#mem, ...args);
  }

  scoreTardosWatermark(...args: Parameters<NativeDriver["scoreTardosWatermark"]>): number {
    return wasmScoreTardosWatermark(this.#mem, ...args);
  }

  extractStructuralHash(...args: Parameters<NativeDriver["extractStructuralHash"]>): Float64Array {
    return wasmExtractStructuralHash(this.#mem, ...args);
  }

  structuralCosineSimilarity(
    ...args: Parameters<NativeDriver["structuralCosineSimilarity"]>
  ): number {
    return wasmStructuralCosineSimilarity(this.#mem, ...args);
  }

  deriveWatermarkSeedHex(...args: Parameters<NativeDriver["deriveWatermarkSeedHex"]>): string {
    return wasmDeriveWatermarkSeedHex(this.#mem, ...args);
  }

  extractStage3Fingerprint(
    ...args: Parameters<NativeDriver["extractStage3Fingerprint"]>
  ): Float64Array {
    return wasmExtractStage3Fingerprint(this.#mem, ...args);
  }

  catmullRomHermite(sM1: number, s0: number, s1: number, s2: number, mu: number): number {
    return this.#mem.exports.dnam_wasm_catmull_rom_hermite(sM1, s0, s1, s2, mu);
  }

  catmullRomHermiteSlice(...args: Parameters<NativeDriver["catmullRomHermiteSlice"]>): void {
    wasmCatmullRomHermiteSlice(this.#mem, ...args);
  }

  opTCoerciveStep(
    ...args: Parameters<NativeDriver["opTCoerciveStep"]>
  ): readonly [number, number, number] {
    return wasmOpTCoerciveStep(this.#mem, ...args);
  }

  opTTankSlice(...args: Parameters<NativeDriver["opTTankSlice"]>): void {
    wasmOpTTankSlice(this.#mem, ...args);
  }

  opMModalStep(
    ...args: Parameters<NativeDriver["opMModalStep"]>
  ): readonly [number, number, number] {
    return wasmOpMModalStep(this.#mem, ...args);
  }

  opMModalSlice(...args: Parameters<NativeDriver["opMModalSlice"]>): readonly [number, number] {
    return wasmOpMModalSlice(this.#mem, ...args);
  }

  unitRotor(...args: Parameters<NativeDriver["unitRotor"]>): readonly [number, number] {
    return wasmUnitRotor(this.#mem, ...args);
  }

  unitRotorSlice(...args: Parameters<NativeDriver["unitRotorSlice"]>): void {
    wasmUnitRotorSlice(this.#mem, ...args);
  }

  polyAdaaAlgebraic(uPrev: number, uCurr: number, beta: number, delta: number): number {
    return this.#mem.exports.dnam_wasm_poly_adaa_algebraic(uPrev, uCurr, beta, delta);
  }

  polyAdaaStep(
    ...args: Parameters<NativeDriver["polyAdaaStep"]>
  ): readonly [number, number, number, number] {
    return wasmPolyAdaaStep(this.#mem, ...args);
  }

  polyAdaaSlice(
    ...args: Parameters<NativeDriver["polyAdaaSlice"]>
  ): readonly [number, number, number] {
    return wasmPolyAdaaSlice(this.#mem, ...args);
  }

  evalLobattoBasis(
    ...args: Parameters<NativeDriver["evalLobattoBasis"]>
  ): readonly [number, number, number] {
    return wasmEvalLobattoBasis(this.#mem, ...args);
  }

  evalLobattoBasisSlice(...args: Parameters<NativeDriver["evalLobattoBasisSlice"]>): void {
    wasmEvalLobattoBasisSlice(this.#mem, ...args);
  }

  evalRank4Fm(
    ...args: Parameters<NativeDriver["evalRank4Fm"]>
  ): readonly [number, number, number, number] {
    return wasmEvalRank4Fm(this.#mem, ...args);
  }

  evalRank4FmSlice(...args: Parameters<NativeDriver["evalRank4FmSlice"]>): void {
    wasmEvalRank4FmSlice(this.#mem, ...args);
  }

  evalChebyshevLfo(
    ...args: Parameters<NativeDriver["evalChebyshevLfo"]>
  ): readonly [number, number, number, number] {
    return wasmEvalChebyshevLfo(this.#mem, ...args);
  }

  evalChebyshevLfoSlice(...args: Parameters<NativeDriver["evalChebyshevLfoSlice"]>): void {
    wasmEvalChebyshevLfoSlice(this.#mem, ...args);
  }

  evalTrustRegionSoftclip(z: number, mh: number): number {
    return this.#mem.exports.dnam_wasm_eval_trust_region_softclip(z, mh);
  }

  evalTrustRegionSoftclipSlice(
    ...args: Parameters<NativeDriver["evalTrustRegionSoftclipSlice"]>
  ): void {
    wasmEvalTrustRegionSoftclipSlice(this.#mem, ...args);
  }

  cubicLagrangeWeights(
    ...args: Parameters<NativeDriver["cubicLagrangeWeights"]>
  ): readonly [number, number, number, number] {
    return wasmCubicLagrangeWeights(this.#mem, ...args);
  }

  cubicLagrangeWeightsSlice(...args: Parameters<NativeDriver["cubicLagrangeWeightsSlice"]>): void {
    wasmCubicLagrangeWeightsSlice(this.#mem, ...args);
  }

  givensHouseholder8ch(...args: Parameters<NativeDriver["givensHouseholder8ch"]>): Float64Array {
    return wasmGivensHouseholder8ch(this.#mem, ...args);
  }

  givensHouseholder8chSlice(...args: Parameters<NativeDriver["givensHouseholder8chSlice"]>): void {
    wasmGivensHouseholder8chSlice(this.#mem, ...args);
  }

  givensHouseholder16ch(...args: Parameters<NativeDriver["givensHouseholder16ch"]>): Float64Array {
    return wasmGivensHouseholder16ch(this.#mem, ...args);
  }

  givensHouseholder16chSlice(
    ...args: Parameters<NativeDriver["givensHouseholder16chSlice"]>
  ): void {
    wasmGivensHouseholder16chSlice(this.#mem, ...args);
  }
}

/**
 * WebAssembly linear memory configuration options (`ref[sdk.ts.api.sync-init]`).
 */
export interface WasmMemoryOptions {
  readonly memory?: WebAssembly.Memory;
  readonly initialPages?: number;
  readonly maximumPages?: number;
  readonly shared?: boolean;
}

/**
 * Options for `initWasm(...)` (`ref[sdk.ts.api.sync-init]`).
 */
export interface InitWasmOptions extends WasmMemoryOptions {
  readonly module?: WebAssembly.Module;
  readonly bytes?: BufferSource;
  readonly url?: string | URL;
}

/**
 * Returns `true` if `new WebAssembly.Memory({ initial, maximum, shared: true })` (`SharedArrayBuffer`)
 * is supported and permitted by the current execution environment (`crossOriginIsolated !== false`).
 */
export function isSharedWasmMemorySupported(): boolean {
  return isSharedWasmMemorySupportedImpl();
}

/** Returns `true` when `memory.buffer` is backed by a `SharedArrayBuffer` (`shared: true`). */
export function isSharedWasmMemory(memory: WebAssembly.Memory): boolean {
  return isSharedBuffer(memory.buffer);
}

/**
 * Creates a bounded `WebAssembly.Memory` suitable for `{ env: { memory } }` instantiation of
 * `dnam.wasm`, supporting optional `shared: true` when permitted by the environment.
 */
export function createWasmMemory(options?: WasmMemoryOptions): WebAssembly.Memory {
  return createWasmMemoryImpl(options);
}

/**
 * Returns a `Uint8Array` copy of `wasmBytes` whose Section 2 (`Import`) `"env"."memory"`
 * descriptor matches `shared` and `maxPages`.
 */
export function configureWasmMemoryImport(
  wasmBytes: Uint8Array,
  shared: boolean,
  maxPages: number | null = WASM_DEFAULT_MAXIMUM_PAGES,
): Uint8Array<ArrayBuffer> {
  return patchWasmMemoryImport(wasmBytes, shared, maxPages);
}

/**
 * Records raw `dnam.wasm` bytes so both unshared and `shared: true` `WebAssembly.Module` variants
 * can be compiled on demand.
 */
export function setCachedWasmBytes(bytes: Uint8Array): void {
  setCachedWasmBytesImpl(bytes);
}

/** Returns the cached compiled `WebAssembly.Module` matching `shared`, if initialized. */
export function getCompiledWasmModule(shared = false): WebAssembly.Module | undefined {
  return getCompiledWasmModuleImpl(shared);
}

/**
 * Instantiates a synchronous `WasmDriver` from a compiled `WebAssembly.Module` and a
 * `WebAssembly.Memory` passed via `{ env: { memory } }`.
 */
export function instantiateWasmDriver(
  module: WebAssembly.Module,
  options?: WasmMemoryOptions,
): WasmDriver {
  const { exports, memory } = instantiateWasmImpl(module, options);
  return new WasmDriver(exports, memory);
}

/**
 * Compiles and caches the WebAssembly `SIMD128` module from a `WebAssembly.Module`,
 * `BufferSource`, or URL (`ref[sdk.ts.api.sync-init]`).
 */
export function compileWasmModule(options?: InitWasmOptions): Promise<WebAssembly.Module> {
  return compileWasmModuleImpl(import.meta.url, options);
}
