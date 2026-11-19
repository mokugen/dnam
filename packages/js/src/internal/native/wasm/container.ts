/**
 * WebAssembly container, `uplo='U'` Gram matrix, and tempo/ZOH delegates
 * (`ref[sdk.ts.runtime.dual-target-drivers]`). Zero `node:*` dependencies.
 *
 * @module
 */

import { DnamError } from "../../../errors.js";
import type { NativeValidatedContainerInfo } from "../driver.js";
import type { WasmMemoryArena } from "./arena.js";

export function wasmValidateContainer(
  mem: WasmMemoryArena,
  containerBytes: Uint8Array,
  deviceIdHash?: string,
  deviceSecret32?: Uint8Array,
  trustedMarketplacePubkeyHex?: string,
): NativeValidatedContainerInfo {
  return mem.withBytes(containerBytes, (cPtr, cLen) =>
    mem.withString(deviceIdHash, (idPtr, idLen) =>
      mem.withBytes(deviceSecret32, (secPtr, secLen) =>
        mem.withString(trustedMarketplacePubkeyHex, (mktPtr, mktLen) => {
          mem.checkStatus(
            mem.exports.dnam_wasm_validate_container_with_key(
              cPtr,
              cLen,
              idPtr,
              idLen,
              secPtr,
              secLen,
              mktPtr,
              mktLen,
            ),
          );
          const parsed = JSON.parse(mem.readResultString()) as Omit<
            NativeValidatedContainerInfo,
            "arenaBytes"
          >;
          return {
            ...parsed,
            arenaBytes: mem.readArenaBytes(),
          };
        }),
      ),
    ),
  );
}

export function wasmBuildReferenceContainer(
  mem: WasmMemoryArena,
  supportsFreeze: boolean,
  inputs: number,
  outputMode: string,
): Uint8Array {
  return mem.withString(outputMode, (mPtr, mLen) => {
    mem.checkStatus(
      mem.exports.dnam_wasm_build_reference_container(supportsFreeze ? 1 : 0, inputs, mPtr, mLen),
    );
    return mem.readResultBytes();
  });
}

export function wasmBuildDnamContainer(
  mem: WasmMemoryArena,
  headerJson: string,
  arenaBytes: Uint8Array,
  encryptedArenaBytes?: Uint8Array,
  hasSignaturesFlag = false,
): Uint8Array {
  return mem.withString(headerJson, (hPtr, hLen) =>
    mem.withBytes(arenaBytes, (aPtr, aLen) =>
      mem.withBytes(encryptedArenaBytes, (ePtr, eLen) => {
        mem.checkStatus(
          mem.exports.dnam_wasm_build_dnam_container(
            hPtr,
            hLen,
            aPtr,
            aLen,
            ePtr,
            eLen,
            hasSignaturesFlag ? 1 : 0,
          ),
        );
        return mem.readResultBytes();
      }),
    ),
  );
}

export function wasmUpdateHeaderInPlace(
  mem: WasmMemoryArena,
  containerBytes: Uint8Array,
  newHeaderJson: string,
): Uint8Array {
  return mem.withBytes(containerBytes, (cPtr, cLen) =>
    mem.withString(newHeaderJson, (hPtr, hLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_update_header_in_place(cPtr, cLen, hPtr, hLen));
      return mem.readResultBytes();
    }),
  );
}

export function wasmStripContainerToTier(
  mem: WasmMemoryArena,
  containerBytes: Uint8Array,
  maxTier: number,
): Uint8Array {
  return mem.withBytes(containerBytes, (cPtr, cLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_strip_container_to_tier(cPtr, cLen, maxTier));
    return mem.readResultBytes();
  });
}

export function wasmParseSchemaVersion(
  mem: WasmMemoryArena,
  versionStr: string,
): readonly [number, number, number, boolean] {
  return mem.withString(versionStr, (vPtr, vLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_parse_schema_version(vPtr, vLen, mem.smallU32Ptr));
    const start = mem.smallU32Ptr >>> 2;
    const u32 = mem.u32();
    return [
      Number(u32[start]),
      Number(u32[start + 1]),
      Number(u32[start + 2]),
      Number(u32[start + 3]) !== 0,
    ];
  });
}

export function wasmCrc32c(mem: WasmMemoryArena, data: Uint8Array): number {
  return mem.withBytes(data, (dPtr, dLen) => mem.exports.dnam_wasm_crc32c(dPtr, dLen) >>> 0);
}

export function wasmValidateAndNormalizeHeader(mem: WasmMemoryArena, headerJson: string): string {
  return mem.withString(headerJson, (hPtr, hLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_validate_and_normalize_header(hPtr, hLen));
    return mem.readResultString();
  });
}

export function wasmCanonicalizeJcs(
  mem: WasmMemoryArena,
  jsonStr: string,
  stripMutable = false,
  rejectFloats = false,
): string {
  return mem.withString(jsonStr, (jPtr, jLen) => {
    mem.checkStatus(
      mem.exports.dnam_wasm_canonicalize_jcs(
        jPtr,
        jLen,
        stripMutable ? 1 : 0,
        rejectFloats ? 1 : 0,
      ),
    );
    return mem.readResultString();
  });
}

export function wasmBuildDmu1Frame(
  mem: WasmMemoryArena,
  mutableJsonBytes: Uint8Array,
  capacity: number,
): Uint8Array {
  return mem.withBytes(mutableJsonBytes, (jPtr, jLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_build_dmu1_frame(jPtr, jLen, capacity));
    return mem.readResultBytes();
  });
}

export function wasmParseDmu1Frame(mem: WasmMemoryArena, frameBytes: Uint8Array): string {
  return mem.withBytes(frameBytes, (fPtr, fLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_parse_dmu1_frame(fPtr, fLen));
    return mem.readResultString();
  });
}

export function wasmValidateV2OpenDnamLayout(
  mem: WasmMemoryArena,
  containerBytes: Uint8Array,
): string {
  return mem.withBytes(containerBytes, (cPtr, cLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_validate_v2_open_dnam_layout(cPtr, cLen));
    return mem.readResultString();
  });
}

export function wasmStripV2OpenDnamContainer(
  mem: WasmMemoryArena,
  containerBytes: Uint8Array,
  targetTier: number,
): Uint8Array {
  return mem.withBytes(containerBytes, (cPtr, cLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_strip_v2_open_dnam_container(cPtr, cLen, targetTier));
    return mem.readResultBytes();
  });
}

export function wasmPackUploUF64Into(
  mem: WasmMemoryArena,
  denseFlat: Float64Array,
  nDim: number,
  outPacked: Float64Array,
): void {
  mem.withF64Slice(denseFlat, (dPtr, dLen) => {
    const { out } = mem.withF64Out(outPacked.length, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_pack_uplo_u_f64_into(dPtr, dLen, nDim, oPtr, outPacked.length),
      ),
    );
    outPacked.set(out);
  });
}

export function wasmUnpackUploUF64Into(
  mem: WasmMemoryArena,
  packedU: Float64Array,
  nTier: number,
  outDense: Float64Array,
): void {
  mem.withF64Slice(packedU, (pPtr, pLen) => {
    const { out } = mem.withF64Out(outDense.length, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_unpack_uplo_u_f64_into(pPtr, pLen, nTier, oPtr, outDense.length),
      ),
    );
    outDense.set(out);
  });
}

export function wasmComputeZoh(
  mem: WasmMemoryArena,
  sampleRateTrainHz: number,
  sampleRateHostHz: number,
  rAcTrain: number,
  rDcTrain: number,
  tauRefTrain: number,
  switchXfadeTrain: number,
): Float64Array {
  mem.checkStatus(
    mem.exports.dnam_wasm_compute_zoh(
      sampleRateTrainHz,
      sampleRateHostHz,
      rAcTrain,
      rDcTrain,
      tauRefTrain,
      switchXfadeTrain,
      mem.smallF64Ptr,
    ),
  );
  const start = mem.smallF64Ptr >>> 3;
  return mem.f64().slice(start, start + 9);
}

export function wasmEvalSubdivisionPhase(
  mem: WasmMemoryArena,
  beatMod12: number,
  withinBeatQ32: number,
  subdivision: string,
  offsetQ32: number,
): readonly [number, number, number] {
  return mem.withString(subdivision, (sPtr, sLen) => {
    mem.checkStatus(
      mem.exports.dnam_wasm_eval_subdivision_phase(
        beatMod12,
        withinBeatQ32,
        sPtr,
        sLen,
        offsetQ32,
        mem.smallF64Ptr,
      ),
    );
    const start = mem.smallF64Ptr >>> 3;
    const f64 = mem.f64();
    return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])] as const;
  });
}

export function wasmEval3NodeCalibration(
  mem: WasmMemoryArena,
  anchors3: Float64Array,
  u: number,
): number {
  if (anchors3.length !== 3) {
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      `anchors must have 3 elements, got ${String(anchors3.length)}`,
    );
  }
  const start = mem.smallF64Ptr >>> 3;
  mem.f64().set(anchors3, start);
  const outPtr = mem.smallF64Ptr + 24;
  mem.checkStatus(mem.exports.dnam_wasm_eval_3node_calibration(mem.smallF64Ptr, u, outPtr));
  return mem.f64()[outPtr >>> 3] ?? 0;
}

export function wasmInvertTempoCalibration(
  mem: WasmMemoryArena,
  declaredMin: number,
  declaredMax: number,
  safeMin: number,
  safeMax: number,
  anchors3: Float64Array,
  target: number,
  flushSubnormals: boolean,
): Float64Array {
  if (anchors3.length !== 3) {
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      `anchors must have 3 elements, got ${String(anchors3.length)}`,
    );
  }
  const start = mem.smallF64Ptr >>> 3;
  mem.f64().set(anchors3, start);
  const outPtr = mem.smallF64Ptr + 24;
  mem.checkStatus(
    mem.exports.dnam_wasm_invert_tempo_calibration(
      declaredMin,
      declaredMax,
      safeMin,
      safeMax,
      mem.smallF64Ptr,
      target,
      flushSubnormals ? 1 : 0,
      outPtr,
    ),
  );
  const outStart = outPtr >>> 3;
  return mem.f64().slice(outStart, outStart + 5);
}
