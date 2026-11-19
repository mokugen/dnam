/**
 * `N-API v10` container, `uplo='U'` Gram matrix, and tempo/ZOH delegates
 * (`ref[sdk.ts.runtime.dual-target-drivers]`).
 *
 * @module
 */

import type { NativeValidatedContainerInfo } from "../driver.js";
import type { RawNapiAddon } from "./bindings.js";
import { callNapi } from "./handles.js";

export function napiValidateContainer(
  addon: RawNapiAddon,
  containerBytes: Uint8Array,
  deviceIdHash?: string,
  deviceSecret32?: Uint8Array,
  trustedMarketplacePubkeyHex?: string,
): NativeValidatedContainerInfo {
  return callNapi(() =>
    addon.validateContainerWithKeyNative(
      containerBytes,
      deviceIdHash,
      deviceSecret32,
      trustedMarketplacePubkeyHex,
    ),
  );
}

export function napiBuildReferenceContainer(
  addon: RawNapiAddon,
  supportsFreeze: boolean,
  inputs: number,
  outputMode: string,
): Uint8Array {
  return callNapi(() => addon.buildReferenceContainerNative(supportsFreeze, inputs, outputMode));
}

export function napiBuildDnamContainer(
  addon: RawNapiAddon,
  headerJson: string,
  arenaBytes: Uint8Array,
  encryptedArenaBytes?: Uint8Array,
  hasSignaturesFlag?: boolean,
): Uint8Array {
  return callNapi(() =>
    addon.buildDnamContainerNative(headerJson, arenaBytes, encryptedArenaBytes, hasSignaturesFlag),
  );
}

export function napiParseSchemaVersion(
  addon: RawNapiAddon,
  versionStr: string,
): readonly [number, number, number, boolean] {
  const res = callNapi(() => addon.parseSchemaVersionNative(versionStr));
  return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3]) !== 0];
}

export function napiComputeZoh(
  addon: RawNapiAddon,
  sampleRateTrainHz: number,
  sampleRateHostHz: number,
  rAcTrain: number,
  rDcTrain: number,
  tauRefTrain: number,
  switchXfadeTrain: number,
): Float64Array {
  return callNapi(() =>
    addon.computeZohNative(
      sampleRateTrainHz,
      sampleRateHostHz,
      rAcTrain,
      rDcTrain,
      tauRefTrain,
      switchXfadeTrain,
    ),
  );
}

export function napiEvalSubdivisionPhase(
  addon: RawNapiAddon,
  beatMod12: number,
  withinBeatQ32: number,
  subdivision: string,
  offsetQ32: number,
): readonly [number, number, number] {
  const res = callNapi(() =>
    addon.evalSubdivisionPhaseNative(beatMod12, withinBeatQ32, subdivision, offsetQ32),
  );
  return [Number(res[0]), Number(res[1]), Number(res[2])];
}

export function napiInvertTempoCalibration(
  addon: RawNapiAddon,
  declaredMin: number,
  declaredMax: number,
  safeMin: number,
  safeMax: number,
  anchors3: Float64Array,
  target: number,
  flushSubnormals: boolean,
): Float64Array {
  return callNapi(() =>
    addon.invertTempoCalibrationNative(
      declaredMin,
      declaredMax,
      safeMin,
      safeMax,
      anchors3,
      target,
      flushSubnormals,
    ),
  );
}
