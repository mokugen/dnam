/**
 * `N-API v10` provenance, sealing, preview/royalty validation, and watermarking delegates
 * (`ref[sdk.ts.provenance.security-and-drm]`).
 *
 * @module
 */

import type { RawNapiAddon } from "./bindings.js";
import { callNapi } from "./handles.js";

export function napiVerifyContainerProvenance(
  addon: RawNapiAddon,
  headerJson: string,
  plaintextArena: Uint8Array,
  hasSignaturesFlag: boolean,
  deviceIdHash?: string,
  deviceSecret32?: Uint8Array,
  trustedMarketplacePubkeyHex?: string,
): string {
  return callNapi(() =>
    addon.verifyContainerProvenanceNative(
      headerJson,
      plaintextArena,
      hasSignaturesFlag,
      deviceIdHash,
      deviceSecret32,
      trustedMarketplacePubkeyHex,
    ),
  );
}

export function napiSignCreatorHeader(
  addon: RawNapiAddon,
  headerJson: string,
  masterArena: Uint8Array,
  creatorSeed32: Uint8Array,
  signerId: string,
  includeTierPrefixes: boolean,
): string {
  return callNapi(() =>
    addon.signCreatorHeaderNative(
      headerJson,
      masterArena,
      creatorSeed32,
      signerId,
      includeTierPrefixes,
    ),
  );
}

export function napiSignOemDeviceCert(
  addon: RawNapiAddon,
  oemSeed32: Uint8Array,
  oemVendor: string,
  modelCode: string,
  deviceIdHash: string,
  pubkeyX2551932: Uint8Array,
): string {
  return callNapi(() =>
    addon.signOemDeviceCertNative(oemSeed32, oemVendor, modelCode, deviceIdHash, pubkeyX2551932),
  );
}

export function napiBuildOemCertV2SigningMessage(
  addon: RawNapiAddon,
  oemVendor: string,
  modelCode: string,
  deviceIdHash16: Uint8Array,
  pubkeyX2551932: Uint8Array,
  issuerPubkeyEd2551932: Uint8Array,
  assurance: number,
  attestationPolicySha25632?: Uint8Array,
  attestationEvidenceSha25632?: Uint8Array,
): Uint8Array {
  return callNapi(() =>
    addon.buildOemCertV2SigningMessageNative(
      oemVendor,
      modelCode,
      deviceIdHash16,
      pubkeyX2551932,
      issuerPubkeyEd2551932,
      assurance,
      attestationPolicySha25632,
      attestationEvidenceSha25632,
    ),
  );
}

export function napiSealContainer(
  addon: RawNapiAddon,
  innerDnamBytes: Uint8Array,
  recipientCertsJson: string,
  trustedOemPubkeysJson: string,
  marketplaceSeed32: Uint8Array,
  ephemeralSecret32: Uint8Array,
  cek32: Uint8Array,
  nonce12: Uint8Array,
): Uint8Array {
  return callNapi(() =>
    addon.sealContainerNative(
      innerDnamBytes,
      recipientCertsJson,
      trustedOemPubkeysJson,
      marketplaceSeed32,
      ephemeralSecret32,
      cek32,
      nonce12,
    ),
  );
}

export function napiUnsealContainer(
  addon: RawNapiAddon,
  sealedBytes: Uint8Array,
  deviceIdHash?: string,
  deviceSecret32?: Uint8Array,
  trustedMarketplacePubkeyHex?: string,
): Uint8Array {
  return callNapi(() =>
    addon.unsealContainerNative(
      sealedBytes,
      deviceIdHash,
      deviceSecret32,
      trustedMarketplacePubkeyHex,
    ),
  );
}

export function napiComputePreviewWindow(
  addon: RawNapiAddon,
  anchor: number,
  delta: number,
  safeMin: number,
  safeMax: number,
): readonly [number, number] {
  const res = callNapi(() => addon.computePreviewWindowNative(anchor, delta, safeMin, safeMax));
  return [Number(res[0]), Number(res[1])];
}

export function napiValidatePreviewSwitchOptions(
  addon: RawNapiAddon,
  preset: string,
  alternative: string,
  observed: readonly string[],
  active: readonly string[],
  safe: readonly string[],
): readonly [string, string] {
  const res = callNapi(() =>
    addon.validatePreviewSwitchOptionsNative(
      preset,
      alternative,
      [...observed],
      [...active],
      [...safe],
    ),
  );
  return [String(res[0]), String(res[1])];
}

export function napiDeriveWatermarkSeedHex(
  addon: RawNapiAddon,
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
  return callNapi(() =>
    addon.deriveWatermarkSeedHexNative(
      key32,
      marketplacePubkey32,
      watermarkKeyId16,
      creatorSignedManifestSha25632,
      modelId16,
      watermarkProfileSha25632,
      owningTier,
      licenseId16,
      buyerId16,
    ),
  );
}
