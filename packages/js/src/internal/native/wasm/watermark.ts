/**
 * WebAssembly isometric gauge / Tardos buyer watermarking and structural fingerprinting delegates
 * (`ref[sdk.ts.provenance.security-and-drm]`). Zero `node:*` dependencies.
 *
 * @module
 */

import type { WasmMemoryArena } from "./arena.js";

export function wasmApplyIsometricGaugeWatermark(
  mem: WasmMemoryArena,
  arenaBytes: Uint8Array,
  kappa64: Float32Array,
  qBuyerFlat16: Float32Array,
): Uint8Array {
  return mem.withBytes(arenaBytes, (aPtr, aLen) =>
    mem.withF32Slice(kappa64, (kPtr, kLen) =>
      mem.withF32Slice(qBuyerFlat16, (qPtr, qLen) => {
        mem.checkStatus(
          mem.exports.dnam_wasm_apply_isometric_gauge_watermark(aPtr, aLen, kPtr, kLen, qPtr, qLen),
        );
        return mem.readResultBytes();
      }),
    ),
  );
}

export function wasmGenerateTardosCode(
  mem: WasmMemoryArena,
  marketplaceSecret32: Uint8Array,
  modelId: string,
  licenseId: string,
): Float32Array {
  return mem.withBytes(marketplaceSecret32, (sPtr, sLen) =>
    mem.withString(modelId, (mPtr, mLen) =>
      mem.withString(licenseId, (lPtr, lLen) => {
        const outLen = 7680;
        const outPtr = mem.alloc(outLen * 4);
        try {
          mem.checkStatus(
            mem.exports.dnam_wasm_generate_tardos_code(
              sPtr,
              sLen,
              mPtr,
              mLen,
              lPtr,
              lLen,
              outPtr,
              outLen,
            ),
          );
          const start = outPtr >>> 2;
          return mem.f32().slice(start, start + outLen);
        } finally {
          mem.exports.dnam_wasm_dealloc(outPtr, outLen * 4, 64);
        }
      }),
    ),
  );
}

export function wasmApplyTardosWatermark(
  mem: WasmMemoryArena,
  arenaBytes: Uint8Array,
  code7680: Float32Array,
  epsilon: number,
): Uint8Array {
  return mem.withBytes(arenaBytes, (aPtr, aLen) =>
    mem.withF32Slice(code7680, (cPtr, cLen) => {
      mem.checkStatus(
        mem.exports.dnam_wasm_apply_tardos_watermark(aPtr, aLen, cPtr, cLen, epsilon),
      );
      return mem.readResultBytes();
    }),
  );
}

export function wasmScoreTardosWatermark(
  mem: WasmMemoryArena,
  masterArena: Uint8Array,
  suspectArena: Uint8Array,
  code7680: Float32Array,
): number {
  return mem.withBytes(masterArena, (mPtr, mLen) =>
    mem.withBytes(suspectArena, (sPtr, sLen) =>
      mem.withF32Slice(code7680, (cPtr, cLen) => {
        mem.checkStatus(
          mem.exports.dnam_wasm_score_tardos_watermark(
            mPtr,
            mLen,
            sPtr,
            sLen,
            cPtr,
            cLen,
            mem.smallF64Ptr,
          ),
        );
        return mem.f64()[mem.smallF64Ptr >>> 3] ?? 0;
      }),
    ),
  );
}

export function wasmExtractStructuralHash(
  mem: WasmMemoryArena,
  arenaBytes: Uint8Array,
): Float64Array {
  return mem.withBytes(arenaBytes, (aPtr, aLen) => {
    const cap = 224;
    const { result: len, out } = mem.withF64Out(cap, (oPtr) =>
      mem.checkStatus(mem.exports.dnam_wasm_extract_structural_hash(aPtr, aLen, oPtr, cap)),
    );
    return out.slice(0, len);
  });
}

export function wasmStructuralCosineSimilarity(
  mem: WasmMemoryArena,
  a: Float64Array,
  b: Float64Array,
): number {
  return mem.withF64Slice(a, (aPtr, aLen) =>
    mem.withF64Slice(b, (bPtr, bLen) => {
      mem.checkStatus(
        mem.exports.dnam_wasm_structural_cosine_similarity(aPtr, aLen, bPtr, bLen, mem.smallF64Ptr),
      );
      return mem.f64()[mem.smallF64Ptr >>> 3] ?? 0;
    }),
  );
}

export function wasmDeriveWatermarkSeedHex(
  mem: WasmMemoryArena,
  key32: Uint8Array,
  marketplacePubkey32: Uint8Array,
  watermarkKeyId16: Uint8Array,
  creatorSignedManifestSha25632: Uint8Array,
  modelId16: Uint8Array,
  watermarkProfileSha25632: Uint8Array,
  owningTier: number,
  licenseId16?: Uint8Array,
  buyerId16?: Uint8Array,
): string {
  return mem.withBytes(key32, (kPtr, kLen) =>
    mem.withBytes(marketplacePubkey32, (mpPtr, mpLen) =>
      mem.withBytes(watermarkKeyId16, (wkPtr, wkLen) =>
        mem.withBytes(creatorSignedManifestSha25632, (cmPtr, cmLen) =>
          mem.withBytes(modelId16, (miPtr, miLen) =>
            mem.withBytes(watermarkProfileSha25632, (wpPtr, wpLen) =>
              mem.withBytes(licenseId16, (liPtr, liLen) =>
                mem.withBytes(buyerId16, (biPtr, biLen) => {
                  mem.checkStatus(
                    mem.exports.dnam_wasm_derive_watermark_seed_hex(
                      kPtr,
                      kLen,
                      mpPtr,
                      mpLen,
                      wkPtr,
                      wkLen,
                      cmPtr,
                      cmLen,
                      miPtr,
                      miLen,
                      wpPtr,
                      wpLen,
                      owningTier,
                      liPtr,
                      liLen,
                      biPtr,
                      biLen,
                    ),
                  );
                  return mem.readResultString();
                }),
              ),
            ),
          ),
        ),
      ),
    ),
  );
}

export function wasmExtractStage3Fingerprint(
  mem: WasmMemoryArena,
  earlyFir128b: Uint8Array,
  kautzModalBytes: Uint8Array,
  addedModes: number,
): Float64Array {
  return mem.withBytes(earlyFir128b, (fPtr, fLen) =>
    mem.withBytes(kautzModalBytes, (kPtr, kLen) => {
      const cap = 256;
      const { result: len, out } = mem.withF64Out(cap, (oPtr) =>
        mem.checkStatus(
          mem.exports.dnam_wasm_extract_stage3_fingerprint(
            fPtr,
            fLen,
            kPtr,
            kLen,
            addedModes,
            oPtr,
            cap,
          ),
        ),
      );
      return out.slice(0, len);
    }),
  );
}
