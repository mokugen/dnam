/**
 * WebAssembly v2 provenance, Ed25519/X25519 signing, ChaCha20/`DAU1`/`DME1` sealing,
 * and preview/royalty validation delegates (`ref[sdk.ts.provenance.v2-security]`).
 * Zero `node:*` dependencies.
 *
 * @module
 */

import type { WasmMemoryArena } from "./arena.js";

export function wasmVerifyContainerProvenance(
  mem: WasmMemoryArena,
  headerJson: string,
  plaintextArena: Uint8Array,
  hasSignaturesFlag: boolean,
  deviceIdHash?: string,
  deviceSecret32?: Uint8Array,
  trustedMarketplacePubkeyHex?: string,
): string {
  return mem.withString(headerJson, (hPtr, hLen) =>
    mem.withBytes(plaintextArena, (aPtr, aLen) =>
      mem.withString(deviceIdHash, (idPtr, idLen) =>
        mem.withBytes(deviceSecret32, (secPtr, secLen) =>
          mem.withString(trustedMarketplacePubkeyHex, (mktPtr, mktLen) => {
            mem.checkStatus(
              mem.exports.dnam_wasm_verify_container_provenance(
                hPtr,
                hLen,
                aPtr,
                aLen,
                hasSignaturesFlag ? 1 : 0,
                idPtr,
                idLen,
                secPtr,
                secLen,
                mktPtr,
                mktLen,
              ),
            );
            return mem.readResultString();
          }),
        ),
      ),
    ),
  );
}

export function wasmEnforceDistributionPolicy(
  mem: WasmMemoryArena,
  headerJson: string,
  activeProfile: string,
): void {
  mem.withString(headerJson, (hPtr, hLen) =>
    mem.withString(activeProfile, (pPtr, pLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_enforce_distribution_policy(hPtr, hLen, pPtr, pLen));
    }),
  );
}

export function wasmEvaluateAnticlonePolicy(
  mem: WasmMemoryArena,
  fingerprintMatched: boolean,
  buyerWatermarkIdentified: boolean,
  independentDeviceSourceEvidence: boolean,
): string {
  mem.checkStatus(
    mem.exports.dnam_wasm_evaluate_anticlone_policy(
      fingerprintMatched ? 1 : 0,
      buyerWatermarkIdentified ? 1 : 0,
      independentDeviceSourceEvidence ? 1 : 0,
    ),
  );
  return mem.readResultString();
}

export function wasmImmutableHeaderSha256(mem: WasmMemoryArena, headerJson: string): string {
  return mem.withString(headerJson, (hPtr, hLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_immutable_header_sha256(hPtr, hLen));
    return mem.readResultString();
  });
}

export function wasmSha256Hex(mem: WasmMemoryArena, data: Uint8Array): string {
  return mem.withBytes(data, (dPtr, dLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_sha256_hex(dPtr, dLen));
    return mem.readResultString();
  });
}

export function wasmHmacSha256Hex(
  mem: WasmMemoryArena,
  key32: Uint8Array,
  message: Uint8Array,
): string {
  return mem.withBytes(key32, (kPtr, kLen) =>
    mem.withBytes(message, (mPtr, mLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_hmac_sha256_hex(kPtr, kLen, mPtr, mLen));
      return mem.readResultString();
    }),
  );
}

export function wasmEd25519PubkeyHex(mem: WasmMemoryArena, seed32: Uint8Array): string {
  return mem.withBytes(seed32, (sPtr, sLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_ed25519_pubkey_hex(sPtr, sLen));
    return mem.readResultString();
  });
}

export function wasmEd25519SignHex(
  mem: WasmMemoryArena,
  seed32: Uint8Array,
  message: Uint8Array,
): string {
  return mem.withBytes(seed32, (sPtr, sLen) =>
    mem.withBytes(message, (mPtr, mLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_ed25519_sign_hex(sPtr, sLen, mPtr, mLen));
      return mem.readResultString();
    }),
  );
}

export function wasmX25519PubkeyHex(mem: WasmMemoryArena, secret32: Uint8Array): string {
  return mem.withBytes(secret32, (sPtr, sLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_x25519_pubkey_hex(sPtr, sLen));
    return mem.readResultString();
  });
}

export function wasmBuildCreatorSigningMessage(
  mem: WasmMemoryArena,
  immutableHeaderSha256Hex: string,
  masterArenaSha256Hex: string,
): Uint8Array {
  return mem.withString(immutableHeaderSha256Hex, (hPtr, hLen) =>
    mem.withString(masterArenaSha256Hex, (aPtr, aLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_build_creator_signing_message(hPtr, hLen, aPtr, aLen));
      return mem.readResultBytes();
    }),
  );
}

export function wasmBuildMarketplaceSigningMessage(
  mem: WasmMemoryArena,
  immutableHeaderSha256Hex: string,
  masterArenaSha256Hex: string,
  receiptJson: string,
): Uint8Array {
  return mem.withString(immutableHeaderSha256Hex, (hPtr, hLen) =>
    mem.withString(masterArenaSha256Hex, (aPtr, aLen) =>
      mem.withString(receiptJson, (rPtr, rLen) => {
        mem.checkStatus(
          mem.exports.dnam_wasm_build_marketplace_signing_message(
            hPtr,
            hLen,
            aPtr,
            aLen,
            rPtr,
            rLen,
          ),
        );
        return mem.readResultBytes();
      }),
    ),
  );
}

export function wasmSignCreatorHeader(
  mem: WasmMemoryArena,
  headerJson: string,
  masterArena: Uint8Array,
  creatorSeed32: Uint8Array,
  signerId: string,
  includeTierPrefixes: boolean,
): string {
  return mem.withString(headerJson, (hPtr, hLen) =>
    mem.withBytes(masterArena, (aPtr, aLen) =>
      mem.withBytes(creatorSeed32, (sPtr, sLen) =>
        mem.withString(signerId, (idPtr, idLen) => {
          mem.checkStatus(
            mem.exports.dnam_wasm_sign_creator_header(
              hPtr,
              hLen,
              aPtr,
              aLen,
              sPtr,
              sLen,
              idPtr,
              idLen,
              includeTierPrefixes ? 1 : 0,
            ),
          );
          return mem.readResultString();
        }),
      ),
    ),
  );
}

export function wasmSignMarketplaceReceipt(
  mem: WasmMemoryArena,
  headerJson: string,
  buyerArena: Uint8Array,
  marketplaceSeed32: Uint8Array,
  specJson: string,
): string {
  return mem.withString(headerJson, (hPtr, hLen) =>
    mem.withBytes(buyerArena, (aPtr, aLen) =>
      mem.withBytes(marketplaceSeed32, (sPtr, sLen) =>
        mem.withString(specJson, (spPtr, spLen) => {
          mem.checkStatus(
            mem.exports.dnam_wasm_sign_marketplace_receipt(
              hPtr,
              hLen,
              aPtr,
              aLen,
              sPtr,
              sLen,
              spPtr,
              spLen,
            ),
          );
          return mem.readResultString();
        }),
      ),
    ),
  );
}

export function wasmSignOemDeviceCert(
  mem: WasmMemoryArena,
  oemSeed32: Uint8Array,
  oemVendor: string,
  modelCode: string,
  deviceIdHash: string,
  pubkeyX2551932: Uint8Array,
): string {
  return mem.withBytes(oemSeed32, (sPtr, sLen) =>
    mem.withString(oemVendor, (vPtr, vLen) =>
      mem.withString(modelCode, (mPtr, mLen) =>
        mem.withString(deviceIdHash, (dPtr, dLen) =>
          mem.withBytes(pubkeyX2551932, (pPtr, pLen) => {
            mem.checkStatus(
              mem.exports.dnam_wasm_sign_oem_device_cert(
                sPtr,
                sLen,
                vPtr,
                vLen,
                mPtr,
                mLen,
                dPtr,
                dLen,
                pPtr,
                pLen,
              ),
            );
            return mem.readResultString();
          }),
        ),
      ),
    ),
  );
}

export function wasmVerifyOemDeviceCert(
  mem: WasmMemoryArena,
  certJson: string,
  trustedOemPubkeyHex: string,
): string {
  return mem.withString(certJson, (cPtr, cLen) =>
    mem.withString(trustedOemPubkeyHex, (pPtr, pLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_verify_oem_device_cert(cPtr, cLen, pPtr, pLen));
      return mem.readResultString();
    }),
  );
}

export function wasmBuildOemCertV2SigningMessage(
  mem: WasmMemoryArena,
  oemVendor: string,
  modelCode: string,
  deviceIdHash16: Uint8Array,
  pubkeyX2551932: Uint8Array,
  issuerPubkeyEd2551932: Uint8Array,
  assurance: number,
  attestationPolicySha25632?: Uint8Array,
  attestationEvidenceSha25632?: Uint8Array,
): Uint8Array {
  return mem.withString(oemVendor, (vPtr, vLen) =>
    mem.withString(modelCode, (mPtr, mLen) =>
      mem.withBytes(deviceIdHash16, (dPtr, dLen) =>
        mem.withBytes(pubkeyX2551932, (pPtr, pLen) =>
          mem.withBytes(issuerPubkeyEd2551932, (iPtr, iLen) =>
            mem.withBytes(attestationPolicySha25632, (polPtr, polLen) =>
              mem.withBytes(attestationEvidenceSha25632, (evPtr, evLen) => {
                mem.checkStatus(
                  mem.exports.dnam_wasm_build_oem_cert_v2_signing_message(
                    vPtr,
                    vLen,
                    mPtr,
                    mLen,
                    dPtr,
                    dLen,
                    pPtr,
                    pLen,
                    iPtr,
                    iLen,
                    assurance,
                    polPtr,
                    polLen,
                    evPtr,
                    evLen,
                  ),
                );
                return mem.readResultBytes();
              }),
            ),
          ),
        ),
      ),
    ),
  );
}

export function wasmChacha20XorArena(
  mem: WasmMemoryArena,
  arenaBytes: Uint8Array,
  key32: Uint8Array,
  nonce12: Uint8Array,
  startBlockCounter: number,
): Uint8Array {
  return mem.withBytes(arenaBytes, (aPtr, aLen) =>
    mem.withBytes(key32, (kPtr, kLen) =>
      mem.withBytes(nonce12, (nPtr, nLen) => {
        mem.checkStatus(
          mem.exports.dnam_wasm_chacha20_xor_arena(
            aPtr,
            aLen,
            kPtr,
            kLen,
            nPtr,
            nLen,
            startBlockCounter,
          ),
        );
        return mem.readResultBytes();
      }),
    ),
  );
}

export function wasmWrapDeviceCekHex(
  mem: WasmMemoryArena,
  ephemeralSecret32: Uint8Array,
  devicePubkey32: Uint8Array,
  deviceIdHash: string,
  cek32: Uint8Array,
): string {
  return mem.withBytes(ephemeralSecret32, (ePtr, eLen) =>
    mem.withBytes(devicePubkey32, (dPtr, dLen) =>
      mem.withString(deviceIdHash, (idPtr, idLen) =>
        mem.withBytes(cek32, (cPtr, cLen) => {
          mem.checkStatus(
            mem.exports.dnam_wasm_wrap_device_cek_hex(
              ePtr,
              eLen,
              dPtr,
              dLen,
              idPtr,
              idLen,
              cPtr,
              cLen,
            ),
          );
          return mem.readResultString();
        }),
      ),
    ),
  );
}

export function wasmSealContainer(
  mem: WasmMemoryArena,
  innerDnamBytes: Uint8Array,
  recipientCertsJson: string,
  trustedOemPubkeysJson: string,
  marketplaceSeed32: Uint8Array,
  ephemeralSecret32: Uint8Array,
  cek32: Uint8Array,
  nonce12: Uint8Array,
): Uint8Array {
  return mem.withBytes(innerDnamBytes, (iPtr, iLen) =>
    mem.withString(recipientCertsJson, (cPtr, cLen) =>
      mem.withString(trustedOemPubkeysJson, (oPtr, oLen) =>
        mem.withBytes(marketplaceSeed32, (mPtr, mLen) =>
          mem.withBytes(ephemeralSecret32, (ePtr, eLen) =>
            mem.withBytes(cek32, (kPtr, kLen) =>
              mem.withBytes(nonce12, (nPtr, nLen) => {
                mem.checkStatus(
                  mem.exports.dnam_wasm_seal_container(
                    iPtr,
                    iLen,
                    cPtr,
                    cLen,
                    oPtr,
                    oLen,
                    mPtr,
                    mLen,
                    ePtr,
                    eLen,
                    kPtr,
                    kLen,
                    nPtr,
                    nLen,
                  ),
                );
                return mem.readResultBytes();
              }),
            ),
          ),
        ),
      ),
    ),
  );
}

export function wasmUnsealContainer(
  mem: WasmMemoryArena,
  sealedBytes: Uint8Array,
  deviceIdHash?: string,
  deviceSecret32?: Uint8Array,
  trustedMarketplacePubkeyHex?: string,
): Uint8Array {
  return mem.withBytes(sealedBytes, (sPtr, sLen) =>
    mem.withString(deviceIdHash, (idPtr, idLen) =>
      mem.withBytes(deviceSecret32, (secPtr, secLen) =>
        mem.withString(trustedMarketplacePubkeyHex, (mktPtr, mktLen) => {
          mem.checkStatus(
            mem.exports.dnam_wasm_unseal_container(
              sPtr,
              sLen,
              idPtr,
              idLen,
              secPtr,
              secLen,
              mktPtr,
              mktLen,
            ),
          );
          return mem.readResultBytes();
        }),
      ),
    ),
  );
}

export function wasmEncodeDau1Envelope(mem: WasmMemoryArena, specJson: string): Uint8Array {
  return mem.withString(specJson, (sPtr, sLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_encode_dau1_envelope(sPtr, sLen));
    return mem.readResultBytes();
  });
}

export function wasmDecodeDau1Envelope(mem: WasmMemoryArena, wireBytes: Uint8Array): string {
  return mem.withBytes(wireBytes, (wPtr, wLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_decode_dau1_envelope(wPtr, wLen));
    return mem.readResultString();
  });
}

export function wasmBuildDme1Header(
  mem: WasmMemoryArena,
  profile: number,
  recipientCount: number,
  ciphertextLength: number,
): Uint8Array {
  mem.checkStatus(
    mem.exports.dnam_wasm_build_dme1_header(profile, recipientCount, BigInt(ciphertextLength)),
  );
  return mem.readResultBytes();
}

export function wasmParseDme1Plaintext(
  mem: WasmMemoryArena,
  plaintext: Uint8Array,
  recipientKeyDigestsHex: readonly string[],
): string {
  return mem.withBytes(plaintext, (pPtr, pLen) =>
    mem.withString(JSON.stringify(recipientKeyDigestsHex), (dPtr, dLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_parse_dme1_plaintext(pPtr, pLen, dPtr, dLen));
      return mem.readResultString();
    }),
  );
}

export function wasmEncodeRecipientKeyIds(
  mem: WasmMemoryArena,
  keyDigestsHex: readonly string[],
): Uint8Array {
  return mem.withString(JSON.stringify(keyDigestsHex), (jPtr, jLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_encode_recipient_key_ids(jPtr, jLen));
    return mem.readResultBytes();
  });
}

export function wasmParseRecipientKeyIds(
  mem: WasmMemoryArena,
  data: Uint8Array,
  maxRecipients: number,
): readonly string[] {
  return mem.withBytes(data, (dPtr, dLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_parse_recipient_key_ids(dPtr, dLen, maxRecipients));
    return JSON.parse(mem.readResultString()) as string[];
  });
}

export function wasmEncodeHpkeSlots(mem: WasmMemoryArena, slotsJson: string): Uint8Array {
  return mem.withString(slotsJson, (jPtr, jLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_encode_hpke_slots(jPtr, jLen));
    return mem.readResultBytes();
  });
}

export function wasmParseHpkeSlots(
  mem: WasmMemoryArena,
  data: Uint8Array,
  expectedCount: number,
  maxRecipients: number,
): string {
  return mem.withBytes(data, (dPtr, dLen) => {
    mem.checkStatus(
      mem.exports.dnam_wasm_parse_hpke_slots(dPtr, dLen, expectedCount, maxRecipients),
    );
    return mem.readResultString();
  });
}

export function wasmEncodeRecordDirectory(
  mem: WasmMemoryArena,
  maximumTier: number,
  recordsJson: string,
): Uint8Array {
  return mem.withString(recordsJson, (jPtr, jLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_encode_record_directory(maximumTier, jPtr, jLen));
    return mem.readResultBytes();
  });
}

export function wasmParseRecordDirectory(
  mem: WasmMemoryArena,
  data: Uint8Array,
  maximumTier: number,
): string {
  return mem.withBytes(data, (dPtr, dLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_parse_record_directory(dPtr, dLen, maximumTier));
    return mem.readResultString();
  });
}

export function wasmComputePreviewWindow(
  mem: WasmMemoryArena,
  anchor: number,
  delta: number,
  safeMin: number,
  safeMax: number,
): readonly [number, number] {
  mem.checkStatus(
    mem.exports.dnam_wasm_compute_preview_window(anchor, delta, safeMin, safeMax, mem.smallF64Ptr),
  );
  const start = mem.smallF64Ptr >>> 3;
  const f64 = mem.f64();
  return [Number(f64[start]), Number(f64[start + 1])];
}

export function wasmValidatePreviewSwitchOptions(
  mem: WasmMemoryArena,
  preset: string,
  alternative: string,
  observed: readonly string[],
  active: readonly string[],
  safe: readonly string[],
): readonly [string, string] {
  const specJson = JSON.stringify({ preset, alternative, observed, active, safe });
  return mem.withString(specJson, (jPtr, jLen) => {
    mem.checkStatus(mem.exports.dnam_wasm_validate_preview_switch_options(jPtr, jLen));
    const parsed = JSON.parse(mem.readResultString()) as [string, string];
    return [parsed[0], parsed[1]];
  });
}

export function wasmValidateRoyaltySplits(
  mem: WasmMemoryArena,
  licenseTier: string,
  royaltySplitsJson: string,
): void {
  mem.withString(licenseTier, (tPtr, tLen) =>
    mem.withString(royaltySplitsJson, (sPtr, sLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_validate_royalty_splits(tPtr, tLen, sPtr, sLen));
    }),
  );
}
