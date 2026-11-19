/**
 * Node.js 26 `N-API v10` driver implementation (`NapiDriver`) and test-only parity bindings
 * (`ref[sdk.ts.runtime.dual-target-drivers]`, `ref[sdk.ts.engine.zero-copy-and-async]`).
 *
 * @module
 */

import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { DnamError } from "../../errors.js";
import { backendToCode } from "../coerce.js";
import type {
  CreateNativeEngineOptions,
  NativeControlSpaceHandle,
  NativeDriver,
  NativeEngineHandle,
  NativeValidatedContainerInfo,
} from "./driver.js";
import type { NativeTestingBindings, RawNapiAddon } from "./napi/bindings.js";
import {
  napiBuildDnamContainer,
  napiBuildReferenceContainer,
  napiComputeZoh,
  napiEvalSubdivisionPhase,
  napiInvertTempoCalibration,
  napiParseSchemaVersion,
  napiValidateContainer,
} from "./napi/container.js";
import { callNapi, NapiControlSpaceHandle, NapiEngineHandle } from "./napi/handles.js";
import {
  napiCubicLagrangeWeights,
  napiEvalChebyshevLfo,
  napiEvalLobattoBasis,
  napiEvalRank4Fm,
  napiOpMModalSlice,
  napiOpMModalStep,
  napiOpTCoerciveStep,
  napiPolyAdaaSlice,
  napiPolyAdaaStep,
} from "./napi/kernels.js";
import {
  napiBuildOemCertV2SigningMessage,
  napiComputePreviewWindow,
  napiDeriveWatermarkSeedHex,
  napiSealContainer,
  napiSignCreatorHeader,
  napiSignOemDeviceCert,
  napiUnsealContainer,
  napiValidatePreviewSwitchOptions,
  napiVerifyContainerProvenance,
} from "./napi/provenance.js";
import {
  configureWasmMemoryImport,
  instantiateWasmDriver,
  setCachedWasmBytes,
  type WasmDriver,
  type WasmMemoryOptions,
} from "./wasm.js";

/**
 * Node.js 26 `N-API v10` native driver delegating FFI and reference DSP calls to `dnam.*.node`.
 */
export class NapiDriver implements NativeDriver {
  readonly kind = "napi" as const;
  readonly #addon: RawNapiAddon;

  constructor(addon: RawNapiAddon) {
    this.#addon = addon;
  }

  createEngine(options: CreateNativeEngineOptions): NativeEngineHandle {
    // Preserve exact coercion validation before invoking the native addon constructor.
    backendToCode(options.backend);
    const raw = callNapi(
      () =>
        new this.#addon.NativeEngine(
          options.containerBytes,
          options.tier,
          options.sampleRateHz,
          options.backend,
          options.maxSupportedTier,
          options.offlineBounce,
          options.maximizeTierOnBounce,
        ),
    );
    return new NapiEngineHandle(raw);
  }

  createControlSpace(headerJson: string): NativeControlSpaceHandle {
    const raw = callNapi(() => new this.#addon.NativeControlSpace(headerJson));
    return new NapiControlSpaceHandle(raw);
  }

  validateContainer(
    ...args: Parameters<NativeDriver["validateContainer"]>
  ): NativeValidatedContainerInfo {
    return napiValidateContainer(this.#addon, ...args);
  }

  buildReferenceContainer(
    ...args: Parameters<NativeDriver["buildReferenceContainer"]>
  ): Uint8Array {
    return napiBuildReferenceContainer(this.#addon, ...args);
  }

  buildDnamContainer(...args: Parameters<NativeDriver["buildDnamContainer"]>): Uint8Array {
    return napiBuildDnamContainer(this.#addon, ...args);
  }

  updateHeaderInPlace(...args: Parameters<NativeDriver["updateHeaderInPlace"]>): Uint8Array {
    return callNapi(() => this.#addon.updateHeaderInPlaceNative(...args));
  }

  stripContainerToTier(...args: Parameters<NativeDriver["stripContainerToTier"]>): Uint8Array {
    return callNapi(() => this.#addon.stripContainerToTierNative(...args));
  }

  parseSchemaVersion(
    ...args: Parameters<NativeDriver["parseSchemaVersion"]>
  ): readonly [number, number, number, boolean] {
    return napiParseSchemaVersion(this.#addon, ...args);
  }

  crc32c(...args: Parameters<NativeDriver["crc32c"]>): number {
    return callNapi(() => this.#addon.crc32CComputeNative(...args)) >>> 0;
  }

  validateAndNormalizeHeader(
    ...args: Parameters<NativeDriver["validateAndNormalizeHeader"]>
  ): string {
    return callNapi(() => this.#addon.validateAndNormalizeHeaderNative(...args));
  }

  canonicalizeJcs(jsonStr: string, stripMutable = false, rejectFloats = false): string {
    return callNapi(() => this.#addon.canonicalizeJcsNative(jsonStr, stripMutable, rejectFloats));
  }

  buildDmu1Frame(...args: Parameters<NativeDriver["buildDmu1Frame"]>): Uint8Array {
    return callNapi(() => this.#addon.buildDmu1FrameNative(...args));
  }

  parseDmu1Frame(...args: Parameters<NativeDriver["parseDmu1Frame"]>): string {
    return callNapi(() => this.#addon.parseDmu1FrameNative(...args));
  }

  validateV2OpenDnamLayout(...args: Parameters<NativeDriver["validateV2OpenDnamLayout"]>): string {
    return callNapi(() => this.#addon.validateV2OpenDnamLayoutNative(...args));
  }

  stripV2OpenDnamContainer(
    ...args: Parameters<NativeDriver["stripV2OpenDnamContainer"]>
  ): Uint8Array {
    return callNapi(() => this.#addon.stripV2OpenDnamContainerNative(...args));
  }

  uploUPackedLen(...args: Parameters<NativeDriver["uploUPackedLen"]>): number {
    return callNapi(() => this.#addon.uploUPackedLenNative(...args));
  }

  packUploUF64Into(...args: Parameters<NativeDriver["packUploUF64Into"]>): void {
    callNapi(() => this.#addon.packUploUF64IntoNative(...args));
  }

  unpackUploUF64Into(...args: Parameters<NativeDriver["unpackUploUF64Into"]>): void {
    callNapi(() => this.#addon.unpackUploUF64IntoNative(...args));
  }

  computeZoh(...args: Parameters<NativeDriver["computeZoh"]>): Float64Array {
    return napiComputeZoh(this.#addon, ...args);
  }

  controlStrideForRate(...args: Parameters<NativeDriver["controlStrideForRate"]>): number {
    return callNapi(() => this.#addon.controlStrideForRateNative(...args));
  }

  controlTickRatio(...args: Parameters<NativeDriver["controlTickRatio"]>): number {
    return callNapi(() => this.#addon.controlTickRatioNative(...args));
  }

  evalSubdivisionPhase(
    ...args: Parameters<NativeDriver["evalSubdivisionPhase"]>
  ): readonly [number, number, number] {
    return napiEvalSubdivisionPhase(this.#addon, ...args);
  }

  eval3NodeCalibration(...args: Parameters<NativeDriver["eval3NodeCalibration"]>): number {
    return callNapi(() => this.#addon.eval3NodeCalibrationNative(...args));
  }

  invertTempoCalibration(
    ...args: Parameters<NativeDriver["invertTempoCalibration"]>
  ): Float64Array {
    return napiInvertTempoCalibration(this.#addon, ...args);
  }

  verifyContainerProvenance(
    ...args: Parameters<NativeDriver["verifyContainerProvenance"]>
  ): string {
    return napiVerifyContainerProvenance(this.#addon, ...args);
  }

  enforceDistributionPolicy(...args: Parameters<NativeDriver["enforceDistributionPolicy"]>): void {
    callNapi(() => this.#addon.enforceDistributionPolicyNative(...args));
  }

  evaluateAnticlonePolicy(...args: Parameters<NativeDriver["evaluateAnticlonePolicy"]>): string {
    return callNapi(() => this.#addon.evaluateAnticlonePolicyNative(...args));
  }

  immutableHeaderSha256(...args: Parameters<NativeDriver["immutableHeaderSha256"]>): string {
    return callNapi(() => this.#addon.immutableHeaderSha256Native(...args));
  }

  sha256Hex(...args: Parameters<NativeDriver["sha256Hex"]>): string {
    return callNapi(() => this.#addon.sha256HexNative(...args));
  }

  hmacSha256Hex(...args: Parameters<NativeDriver["hmacSha256Hex"]>): string {
    return callNapi(() => this.#addon.hmacSha256HexNative(...args));
  }

  ed25519PubkeyHex(...args: Parameters<NativeDriver["ed25519PubkeyHex"]>): string {
    return callNapi(() => this.#addon.ed25519PubkeyHexNative(...args));
  }

  ed25519SignHex(...args: Parameters<NativeDriver["ed25519SignHex"]>): string {
    return callNapi(() => this.#addon.ed25519SignHexNative(...args));
  }

  x25519PubkeyHex(...args: Parameters<NativeDriver["x25519PubkeyHex"]>): string {
    return callNapi(() => this.#addon.x25519PubkeyHexNative(...args));
  }

  buildCreatorSigningMessage(
    ...args: Parameters<NativeDriver["buildCreatorSigningMessage"]>
  ): Uint8Array {
    return callNapi(() => this.#addon.buildCreatorSigningMessageNative(...args));
  }

  buildMarketplaceSigningMessage(
    ...args: Parameters<NativeDriver["buildMarketplaceSigningMessage"]>
  ): Uint8Array {
    return callNapi(() => this.#addon.buildMarketplaceSigningMessageNative(...args));
  }

  signCreatorHeader(...args: Parameters<NativeDriver["signCreatorHeader"]>): string {
    return napiSignCreatorHeader(this.#addon, ...args);
  }

  signMarketplaceReceipt(...args: Parameters<NativeDriver["signMarketplaceReceipt"]>): string {
    return callNapi(() => this.#addon.signMarketplaceReceiptNative(...args));
  }

  signOemDeviceCert(...args: Parameters<NativeDriver["signOemDeviceCert"]>): string {
    return napiSignOemDeviceCert(this.#addon, ...args);
  }

  verifyOemDeviceCert(...args: Parameters<NativeDriver["verifyOemDeviceCert"]>): string {
    return callNapi(() => this.#addon.verifyOemDeviceCertNative(...args));
  }

  buildOemCertV2SigningMessage(
    ...args: Parameters<NativeDriver["buildOemCertV2SigningMessage"]>
  ): Uint8Array {
    return napiBuildOemCertV2SigningMessage(this.#addon, ...args);
  }

  chacha20XorArena(...args: Parameters<NativeDriver["chacha20XorArena"]>): Uint8Array {
    return callNapi(() => this.#addon.chacha20XorArenaNative(...args));
  }

  wrapDeviceCekHex(...args: Parameters<NativeDriver["wrapDeviceCekHex"]>): string {
    return callNapi(() => this.#addon.wrapDeviceCekHexNative(...args));
  }

  sealContainer(...args: Parameters<NativeDriver["sealContainer"]>): Uint8Array {
    return napiSealContainer(this.#addon, ...args);
  }

  unsealContainer(...args: Parameters<NativeDriver["unsealContainer"]>): Uint8Array {
    return napiUnsealContainer(this.#addon, ...args);
  }

  encodeDau1Envelope(...args: Parameters<NativeDriver["encodeDau1Envelope"]>): Uint8Array {
    return callNapi(() => this.#addon.encodeDau1EnvelopeNative(...args));
  }

  decodeDau1Envelope(...args: Parameters<NativeDriver["decodeDau1Envelope"]>): string {
    return callNapi(() => this.#addon.decodeDau1EnvelopeNative(...args));
  }

  buildDme1Header(...args: Parameters<NativeDriver["buildDme1Header"]>): Uint8Array {
    return callNapi(() => this.#addon.buildDme1HeaderNative(...args));
  }

  parseDme1Plaintext(plaintext: Uint8Array, recipientKeyDigestsHex: readonly string[]): string {
    return callNapi(() =>
      this.#addon.parseDme1PlaintextNative(plaintext, [...recipientKeyDigestsHex]),
    );
  }

  encodeRecipientKeyIds(keyDigestsHex: readonly string[]): Uint8Array {
    return callNapi(() => this.#addon.encodeRecipientKeyIdsNative([...keyDigestsHex]));
  }

  parseRecipientKeyIds(
    ...args: Parameters<NativeDriver["parseRecipientKeyIds"]>
  ): readonly string[] {
    return callNapi(() => this.#addon.parseRecipientKeyIdsNative(...args));
  }

  encodeHpkeSlots(...args: Parameters<NativeDriver["encodeHpkeSlots"]>): Uint8Array {
    return callNapi(() => this.#addon.encodeHpkeSlotsNative(...args));
  }

  parseHpkeSlots(...args: Parameters<NativeDriver["parseHpkeSlots"]>): string {
    return callNapi(() => this.#addon.parseHpkeSlotsNative(...args));
  }

  encodeRecordDirectory(...args: Parameters<NativeDriver["encodeRecordDirectory"]>): Uint8Array {
    return callNapi(() => this.#addon.encodeRecordDirectoryNative(...args));
  }

  parseRecordDirectory(...args: Parameters<NativeDriver["parseRecordDirectory"]>): string {
    return callNapi(() => this.#addon.parseRecordDirectoryNative(...args));
  }

  computePreviewWindow(
    ...args: Parameters<NativeDriver["computePreviewWindow"]>
  ): readonly [number, number] {
    return napiComputePreviewWindow(this.#addon, ...args);
  }

  validatePreviewSwitchOptions(
    ...args: Parameters<NativeDriver["validatePreviewSwitchOptions"]>
  ): readonly [string, string] {
    return napiValidatePreviewSwitchOptions(this.#addon, ...args);
  }

  validateRoyaltySplits(...args: Parameters<NativeDriver["validateRoyaltySplits"]>): void {
    callNapi(() => this.#addon.validateRoyaltySplitsNative(...args));
  }

  applyIsometricGaugeWatermark(
    ...args: Parameters<NativeDriver["applyIsometricGaugeWatermark"]>
  ): Uint8Array {
    return callNapi(() => this.#addon.applyIsometricGaugeWatermarkNative(...args));
  }

  generateTardosCode(...args: Parameters<NativeDriver["generateTardosCode"]>): Float32Array {
    return callNapi(() => this.#addon.generateTardosCodeNative(...args));
  }

  applyTardosWatermark(...args: Parameters<NativeDriver["applyTardosWatermark"]>): Uint8Array {
    return callNapi(() => this.#addon.applyTardosWatermarkNative(...args));
  }

  scoreTardosWatermark(...args: Parameters<NativeDriver["scoreTardosWatermark"]>): number {
    return callNapi(() => this.#addon.scoreTardosWatermarkNative(...args));
  }

  extractStructuralHash(...args: Parameters<NativeDriver["extractStructuralHash"]>): Float64Array {
    return callNapi(() => this.#addon.extractStructuralHashNative(...args));
  }

  structuralCosineSimilarity(
    ...args: Parameters<NativeDriver["structuralCosineSimilarity"]>
  ): number {
    return callNapi(() => this.#addon.structuralCosineSimilarityNative(...args));
  }

  deriveWatermarkSeedHex(...args: Parameters<NativeDriver["deriveWatermarkSeedHex"]>): string {
    return napiDeriveWatermarkSeedHex(this.#addon, ...args);
  }

  extractStage3Fingerprint(
    ...args: Parameters<NativeDriver["extractStage3Fingerprint"]>
  ): Float64Array {
    return callNapi(() => this.#addon.extractStage3FingerprintNative(...args));
  }

  catmullRomHermite(...args: Parameters<NativeDriver["catmullRomHermite"]>): number {
    return callNapi(() => this.#addon.catmullRomHermiteNative(...args));
  }

  catmullRomHermiteSlice(...args: Parameters<NativeDriver["catmullRomHermiteSlice"]>): void {
    callNapi(() => this.#addon.catmullRomHermiteSliceNative(...args));
  }

  opTCoerciveStep(
    ...args: Parameters<NativeDriver["opTCoerciveStep"]>
  ): readonly [number, number, number] {
    return napiOpTCoerciveStep(this.#addon, ...args);
  }

  opTTankSlice(...args: Parameters<NativeDriver["opTTankSlice"]>): void {
    callNapi(() => this.#addon.opTTankSliceNative(...args));
  }

  opMModalStep(
    ...args: Parameters<NativeDriver["opMModalStep"]>
  ): readonly [number, number, number] {
    return napiOpMModalStep(this.#addon, ...args);
  }

  opMModalSlice(...args: Parameters<NativeDriver["opMModalSlice"]>): readonly [number, number] {
    return napiOpMModalSlice(this.#addon, ...args);
  }

  unitRotor(...args: Parameters<NativeDriver["unitRotor"]>): readonly [number, number] {
    const res = callNapi(() => this.#addon.unitRotorNative(...args));
    return [Number(res[0]), Number(res[1])];
  }

  unitRotorSlice(...args: Parameters<NativeDriver["unitRotorSlice"]>): void {
    callNapi(() => this.#addon.unitRotorSliceNative(...args));
  }

  polyAdaaAlgebraic(...args: Parameters<NativeDriver["polyAdaaAlgebraic"]>): number {
    return callNapi(() => this.#addon.polyAdaaAlgebraicNative(...args));
  }

  polyAdaaStep(
    ...args: Parameters<NativeDriver["polyAdaaStep"]>
  ): readonly [number, number, number, number] {
    return napiPolyAdaaStep(this.#addon, ...args);
  }

  polyAdaaSlice(
    ...args: Parameters<NativeDriver["polyAdaaSlice"]>
  ): readonly [number, number, number] {
    return napiPolyAdaaSlice(this.#addon, ...args);
  }

  evalLobattoBasis(
    ...args: Parameters<NativeDriver["evalLobattoBasis"]>
  ): readonly [number, number, number] {
    return napiEvalLobattoBasis(this.#addon, ...args);
  }

  evalLobattoBasisSlice(...args: Parameters<NativeDriver["evalLobattoBasisSlice"]>): void {
    callNapi(() => this.#addon.evalLobattoBasisSliceNative(...args));
  }

  evalRank4Fm(
    ...args: Parameters<NativeDriver["evalRank4Fm"]>
  ): readonly [number, number, number, number] {
    return napiEvalRank4Fm(this.#addon, ...args);
  }

  evalRank4FmSlice(...args: Parameters<NativeDriver["evalRank4FmSlice"]>): void {
    callNapi(() => this.#addon.evalRank4FmSliceNative(...args));
  }

  evalChebyshevLfo(
    ...args: Parameters<NativeDriver["evalChebyshevLfo"]>
  ): readonly [number, number, number, number] {
    return napiEvalChebyshevLfo(this.#addon, ...args);
  }

  evalChebyshevLfoSlice(...args: Parameters<NativeDriver["evalChebyshevLfoSlice"]>): void {
    callNapi(() => this.#addon.evalChebyshevLfoSliceNative(...args));
  }

  evalTrustRegionSoftclip(...args: Parameters<NativeDriver["evalTrustRegionSoftclip"]>): number {
    return callNapi(() => this.#addon.evalTrustRegionSoftclipNative(...args));
  }

  evalTrustRegionSoftclipSlice(
    ...args: Parameters<NativeDriver["evalTrustRegionSoftclipSlice"]>
  ): void {
    callNapi(() => this.#addon.evalTrustRegionSoftclipSliceNative(...args));
  }

  cubicLagrangeWeights(
    ...args: Parameters<NativeDriver["cubicLagrangeWeights"]>
  ): readonly [number, number, number, number] {
    return napiCubicLagrangeWeights(this.#addon, ...args);
  }

  cubicLagrangeWeightsSlice(...args: Parameters<NativeDriver["cubicLagrangeWeightsSlice"]>): void {
    callNapi(() => this.#addon.cubicLagrangeWeightsSliceNative(...args));
  }

  givensHouseholder8ch(...args: Parameters<NativeDriver["givensHouseholder8ch"]>): Float64Array {
    return callNapi(() => this.#addon.givensHouseholder8ChNative(...args));
  }

  givensHouseholder8chSlice(...args: Parameters<NativeDriver["givensHouseholder8chSlice"]>): void {
    callNapi(() => this.#addon.givensHouseholder8ChSliceNative(...args));
  }

  givensHouseholder16ch(...args: Parameters<NativeDriver["givensHouseholder16ch"]>): Float64Array {
    return callNapi(() => this.#addon.givensHouseholder16ChNative(...args));
  }

  givensHouseholder16chSlice(
    ...args: Parameters<NativeDriver["givensHouseholder16chSlice"]>
  ): void {
    callNapi(() => this.#addon.givensHouseholder16ChSliceNative(...args));
  }
}

type BuildProfile = "release" | "debug" | "profiling";
const BUILD_PROFILES: readonly BuildProfile[] = ["release", "debug", "profiling"];

function hasDistArtifacts(dir: string): boolean {
  const addonFileName = `dnam.${process.platform}-${process.arch}.node`;
  return BUILD_PROFILES.some(
    (profile) =>
      existsSync(join(dir, "wasm", profile, "dnam.wasm")) ||
      existsSync(join(dir, "native", profile, addonFileName)),
  );
}

function resolveRootDistDir(): string {
  const currentDir = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    resolve(currentDir, "../../../dist"),
    resolve(currentDir),
    resolve(currentDir, "../../dist"),
  ];
  for (const candidate of candidates) {
    if (hasDistArtifacts(candidate)) {
      return candidate;
    }
  }
  return candidates[0] ?? resolve(currentDir);
}

function resolveProfileOrder(): readonly BuildProfile[] {
  const requested = process.env["DNAM_BUILD_PROFILE"];
  if (requested === "debug" || requested === "profiling" || requested === "release") {
    return [requested, ...BUILD_PROFILES.filter((p) => p !== requested)];
  }
  return BUILD_PROFILES;
}

let cachedAddon: RawNapiAddon | undefined;
let cachedNapiDriver: NapiDriver | undefined;
let cachedNodeWasmDriver: WasmDriver | undefined;

export function getNapiAddon(): RawNapiAddon {
  if (cachedAddon) {
    return cachedAddon;
  }
  const rootDist = resolveRootDistDir();
  const addonFileName = `dnam.${process.platform}-${process.arch}.node`;
  const profiles = resolveProfileOrder();
  let addonPath = join(rootDist, "native", profiles[0] ?? "release", addonFileName);
  for (const profile of profiles) {
    const candidate = join(rootDist, "native", profile, addonFileName);
    if (existsSync(candidate)) {
      addonPath = candidate;
      break;
    }
  }
  if (!existsSync(addonPath)) {
    throw new DnamError("ERR_DNAM_CONTAINER", `Native N-API addon not found at ${addonPath}`);
  }
  const req = createRequire(import.meta.url);
  cachedAddon = req(addonPath) as RawNapiAddon;
  return cachedAddon;
}

export function getTestingBindings(): NativeTestingBindings {
  return getNapiAddon();
}

export function readDistWasmBytes(): Uint8Array<ArrayBuffer> {
  const rootDist = resolveRootDistDir();
  const profiles = resolveProfileOrder();
  let wasmPath = join(rootDist, "wasm", profiles[0] ?? "release", "dnam.wasm");
  for (const profile of profiles) {
    const candidate = join(rootDist, "wasm", profile, "dnam.wasm");
    if (existsSync(candidate)) {
      wasmPath = candidate;
      break;
    }
  }
  const bytes = new Uint8Array(readFileSync(wasmPath));
  setCachedWasmBytes(bytes);
  return bytes;
}

export function loadNodeWasmDriverSync(options?: WasmMemoryOptions): WasmDriver {
  if (!options && cachedNodeWasmDriver) {
    return cachedNodeWasmDriver;
  }
  const wasmBytes = readDistWasmBytes();
  const wantShared = Boolean(options?.shared);
  const mod = new WebAssembly.Module(configureWasmMemoryImport(wasmBytes, wantShared));
  const driver = instantiateWasmDriver(mod, options);
  if (!options) {
    cachedNodeWasmDriver = driver;
  }
  return driver;
}

export function loadNodeDriverSync(): NativeDriver {
  if (cachedNapiDriver) {
    return cachedNapiDriver;
  }
  try {
    cachedNapiDriver = new NapiDriver(getNapiAddon());
    return cachedNapiDriver;
  } catch {
    return loadNodeWasmDriverSync();
  }
}
