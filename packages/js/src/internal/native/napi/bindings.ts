/**
 * Raw Node.js 26 `N-API v10` native addon interface declarations (`ref[sdk.ts.runtime.dual-target-drivers]`).
 *
 * @module
 */

import type { NativeValidatedContainerInfo } from "../driver.js";

export interface RawNapiEngine {
  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void;
  readonly backend: string;
  readonly tier: number;
  setTier(tier: number): void;
  readonly maxSupportedTier: number;
  setMaxSupportedTier(tier: number): void;
  readonly maxAvailableTier: number;
  readonly effectiveTier: number;
  readonly sampleRate: number;
  setSampleRate(sampleRateHz: number): void;
  readonly latencySamples: number;
  readonly offlineBounce: boolean;
  setOfflineBounce(enabled: boolean): void;
  readonly maximizeTierOnBounce: boolean;
  setMaximizeTierOnBounce(enabled: boolean): void;
  readonly numControls: number;
  rawControlValues(): Float32Array;
  clampedControlValues(): Float32Array;
  clampedControlValue(ctrlIdx: number): number;
  setControlValue(ctrlIdx: number, value: number): void;
  setControlImmediate(ctrlIdx: number, value: number): void;
  setControlsImmediateBatch(indices: number[], values: number[]): void;
  resetControlsToDefaults(): void;
  reset(): void;
  readonly supportsFreeze: boolean;
  readonly isFrozen: boolean;
  setFreeze(freeze: boolean): void;
  setFreezeAmount(alpha: number): void;
  bypassState(): boolean[];
  setBypass(bypassed: boolean, trails: boolean): void;
  readonly stereoWidth: number;
  setStereoWidth(width: number): void;
  configureHostMix(blendWet: number, outputGainDb: number, applyUnityLeveling: boolean): void;
  readonly currentBpm: number;
  syncTempo(bpm: number, subdivision?: string, foldOctave?: boolean): number;
  registerTapTempo(subdivision?: string, foldOctave?: boolean): number[];
  applyPreset(presetId: string): void;
  processStereoInPlace(leftIo: Float32Array, rightIo: Float32Array): void;
  processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void;
  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void;
  processMono(monoIn: Float32Array, monoOut: Float32Array, blockSize?: number): void;
  renderOfflineBounce(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
  ): void;
  processStereoAsync(
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
  ): Promise<Float32Array[]>;
  telemetryInto(out56: Float64Array): void;
}

export interface RawNapiControlSpace {
  readonly numControls: number;
  clampInto(rawNumeric: Float32Array, clampedOut: Float32Array, normOut: Float32Array): number;
  toPhysical(ctrlIdx: number, normalizedU: number): number;
  toNormalized(ctrlIdx: number, physicalVal: number): number;
}

export interface NativeTestingBindings {
  testingRenderReferenceStereo(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
    reset?: boolean,
  ): Float32Array[];
  testingRenderReferenceMonoToStereo(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    monoIn: Float32Array,
    blockSize?: number,
    reset?: boolean,
  ): Float32Array[];
  testingRenderReferenceMono(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    monoIn: Float32Array,
    blockSize?: number,
    reset?: boolean,
  ): Float32Array;
  testingBenchmarkNativeStereoNs(
    containerBytes: Uint8Array,
    tier: number,
    backend: string,
    sampleRateHz: number,
    frames: number,
    iterations: number,
  ): number;
  testingReferenceHeaderJson(kind?: string): string;
  testingErrorVariantNames(): string[];
  testingRaiseErrorVariant(name: string): void;
}

export interface RawNapiAddon extends NativeTestingBindings {
  NativeEngine: new (
    containerBytes: Uint8Array,
    tier: number,
    sampleRateHz: number,
    backend?: string,
    maxSupportedTier?: number,
    offlineBounce?: boolean,
    maximizeTierOnBounce?: boolean,
  ) => RawNapiEngine;
  NativeControlSpace: new (headerJson: string) => RawNapiControlSpace;
  validateContainerWithKeyNative(
    containerBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): NativeValidatedContainerInfo;
  buildReferenceContainerNative(
    supportsFreeze?: boolean,
    inputs?: number,
    outputMode?: string,
  ): Uint8Array;
  buildDnamContainerNative(
    headerJson: string,
    arenaBytes: Uint8Array,
    encryptedArenaBytes?: Uint8Array,
    hasSignaturesFlag?: boolean,
  ): Uint8Array;
  updateHeaderInPlaceNative(containerBytes: Uint8Array, newHeaderJson: string): Uint8Array;
  stripContainerToTierNative(containerBytes: Uint8Array, maxTier: number): Uint8Array;
  parseSchemaVersionNative(versionStr: string): number[];
  crc32CComputeNative(data: Uint8Array): number;
  validateAndNormalizeHeaderNative(headerJson: string): string;
  canonicalizeJcsNative(jsonStr: string, stripMutable?: boolean, rejectFloats?: boolean): string;
  buildDmu1FrameNative(mutableJsonBytes: Uint8Array, capacity: number): Uint8Array;
  parseDmu1FrameNative(frameBytes: Uint8Array): string;
  validateV2OpenDnamLayoutNative(containerBytes: Uint8Array): string;
  stripV2OpenDnamContainerNative(containerBytes: Uint8Array, targetTier: number): Uint8Array;
  uploUPackedLenNative(nDim: number): number;
  packUploUF64IntoNative(denseFlat: Float64Array, nDim: number, outPacked: Float64Array): void;
  unpackUploUF64IntoNative(packedU: Float64Array, nTier: number, outDense: Float64Array): void;
  computeZohNative(
    sampleRateTrainHz: number,
    sampleRateHostHz: number,
    rAcTrain?: number,
    rDcTrain?: number,
    tauRefTrain?: number,
    switchXfadeTrain?: number,
  ): Float64Array;
  controlStrideForRateNative(sampleRateHz: number): number;
  controlTickRatioNative(sampleRateHz: number): number;
  evalSubdivisionPhaseNative(
    beatMod12: number,
    withinBeatQ32: number,
    subdivision: string,
    offsetQ32?: number,
  ): number[];
  eval3NodeCalibrationNative(anchors: Float64Array, u: number): number;
  invertTempoCalibrationNative(
    declaredMin: number,
    declaredMax: number,
    safeMin: number,
    safeMax: number,
    anchors: Float64Array,
    target: number,
    flushSubnormals?: boolean,
  ): Float64Array;
  verifyContainerProvenanceNative(
    headerJson: string,
    plaintextArena: Uint8Array,
    hasSignaturesFlag?: boolean,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): string;
  enforceDistributionPolicyNative(headerJson: string, activeProfile: string): void;
  evaluateAnticlonePolicyNative(
    fingerprintMatched: boolean,
    buyerWatermarkIdentified: boolean,
    independentDeviceSourceEvidence: boolean,
  ): string;
  immutableHeaderSha256Native(headerJson: string): string;
  sha256HexNative(data: Uint8Array): string;
  hmacSha256HexNative(key32: Uint8Array, message: Uint8Array): string;
  ed25519PubkeyHexNative(seed32: Uint8Array): string;
  ed25519SignHexNative(seed32: Uint8Array, message: Uint8Array): string;
  x25519PubkeyHexNative(secret32: Uint8Array): string;
  buildCreatorSigningMessageNative(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
  ): Uint8Array;
  buildMarketplaceSigningMessageNative(
    immutableHeaderSha256Hex: string,
    masterArenaSha256Hex: string,
    receiptJson: string,
  ): Uint8Array;
  signCreatorHeaderNative(
    headerJson: string,
    masterArena: Uint8Array,
    creatorSeed32: Uint8Array,
    signerId: string,
    includeTierPrefixes?: boolean,
  ): string;
  signMarketplaceReceiptNative(
    headerJson: string,
    buyerArena: Uint8Array,
    marketplaceSeed32: Uint8Array,
    specJson: string,
  ): string;
  signOemDeviceCertNative(
    oemSeed32: Uint8Array,
    oemVendor: string,
    modelCode: string,
    deviceIdHash: string,
    pubkeyX2551932: Uint8Array,
  ): string;
  verifyOemDeviceCertNative(certJson: string, trustedOemPubkeyHex: string): string;
  buildOemCertV2SigningMessageNative(
    oemVendor: string,
    modelCode: string,
    deviceIdHash16: Uint8Array,
    pubkeyX2551932: Uint8Array,
    issuerPubkeyEd2551932: Uint8Array,
    assurance?: number,
    attestationPolicySha25632?: Uint8Array,
    attestationEvidenceSha25632?: Uint8Array,
  ): Uint8Array;
  chacha20XorArenaNative(
    arenaBytes: Uint8Array,
    key32: Uint8Array,
    nonce12: Uint8Array,
    startBlockCounter?: number,
  ): Uint8Array;
  wrapDeviceCekHexNative(
    ephemeralSecret32: Uint8Array,
    devicePubkey32: Uint8Array,
    deviceIdHash: string,
    cek32: Uint8Array,
  ): string;
  sealContainerNative(
    innerDnamBytes: Uint8Array,
    recipientCertsJson: string,
    trustedOemPubkeysJson: string,
    marketplaceSeed32: Uint8Array,
    ephemeralSecret32: Uint8Array,
    cek32: Uint8Array,
    nonce12: Uint8Array,
  ): Uint8Array;
  unsealContainerNative(
    sealedBytes: Uint8Array,
    deviceIdHash?: string,
    deviceSecret32?: Uint8Array,
    trustedMarketplacePubkeyHex?: string,
  ): Uint8Array;
  encodeDau1EnvelopeNative(specJson: string): Uint8Array;
  decodeDau1EnvelopeNative(wireBytes: Uint8Array): string;
  buildDme1HeaderNative(
    profile: number,
    recipientCount: number,
    ciphertextLength: number,
  ): Uint8Array;
  parseDme1PlaintextNative(plaintext: Uint8Array, recipientKeyDigestsHex: string[]): string;
  encodeRecipientKeyIdsNative(keyDigestsHex: string[]): Uint8Array;
  parseRecipientKeyIdsNative(data: Uint8Array, maxRecipients?: number): string[];
  encodeHpkeSlotsNative(slotsJson: string): Uint8Array;
  parseHpkeSlotsNative(data: Uint8Array, expectedCount: number, maxRecipients?: number): string;
  encodeRecordDirectoryNative(maximumTier: number, recordsJson: string): Uint8Array;
  parseRecordDirectoryNative(data: Uint8Array, maximumTier: number): string;
  computePreviewWindowNative(
    anchor: number,
    delta: number,
    safeMin: number,
    safeMax: number,
  ): Float64Array;
  validatePreviewSwitchOptionsNative(
    preset: string,
    alternative: string,
    observed: string[],
    active: string[],
    safe: string[],
  ): string[];
  validateRoyaltySplitsNative(licenseTier: string, royaltySplitsJson: string): void;
  applyIsometricGaugeWatermarkNative(
    arenaBytes: Uint8Array,
    kappa: Float32Array,
    qBuyerFlat16: Float32Array,
  ): Uint8Array;
  generateTardosCodeNative(
    marketplaceSecret32: Uint8Array,
    modelId: string,
    licenseId: string,
  ): Float32Array;
  applyTardosWatermarkNative(
    arenaBytes: Uint8Array,
    code: Float32Array,
    epsilon: number,
  ): Uint8Array;
  scoreTardosWatermarkNative(
    masterArena: Uint8Array,
    suspectArena: Uint8Array,
    code: Float32Array,
  ): number;
  extractStructuralHashNative(arenaBytes: Uint8Array): Float64Array;
  structuralCosineSimilarityNative(a: Float64Array, b: Float64Array): number;
  deriveWatermarkSeedHexNative(
    key32: Uint8Array,
    marketplacePubkey32: Uint8Array,
    watermarkKeyId16: Uint8Array,
    creatorSignedManifestSha25632: Uint8Array,
    modelId16: Uint8Array,
    watermarkProfileSha25632: Uint8Array,
    owningTier: number,
    licenseId16?: Uint8Array,
    buyerId16?: Uint8Array,
  ): string;
  extractStage3FingerprintNative(
    earlyFir128b: Uint8Array,
    kautzModalBytes: Uint8Array,
    addedModes: number,
  ): Float64Array;
  catmullRomHermiteNative(sM1: number, s0: number, s1: number, s2: number, mu: number): number;
  catmullRomHermiteSliceNative(
    sM1: Float64Array,
    s0: Float64Array,
    s1: Float64Array,
    s2: Float64Array,
    mu: Float64Array,
    out: Float64Array,
  ): void;
  opTCoerciveStepNative(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
  ): Float64Array;
  opTTankSliceNative(
    inp: Float64Array,
    out: Float64Array,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): void;
  opMModalStepNative(
    x: number,
    rLow: number,
    drSpan: number,
    wSel: number,
    bGammaNorm: number,
    cReal: number,
    cImag: number,
    uRotCos: number,
    uRotSin: number,
    zRe: number,
    zIm: number,
  ): Float64Array;
  opMModalSliceNative(
    inp: Float64Array,
    out: Float64Array,
    rLow: number,
    drSpan: number,
    wSel: number,
    bGammaNorm: number,
    cReal: number,
    cImag: number,
    uRotCos: number,
    uRotSin: number,
    zRe: number,
    zIm: number,
    reset: boolean,
  ): Float64Array;
  unitRotorNative(phi: number): Float64Array;
  unitRotorSliceNative(phi: Float64Array, out: Float64Array): void;
  polyAdaaAlgebraicNative(uPrev: number, uCurr: number, beta: number, delta: number): number;
  polyAdaaStepNative(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
  ): Float64Array;
  polyAdaaSliceNative(
    inp: Float64Array,
    out: Float64Array,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
    reset: boolean,
  ): Float64Array;
  evalLobattoBasisNative(u: number, anchors: Float64Array, degree: number): Float64Array;
  evalLobattoBasisSliceNative(
    u: Float64Array,
    out: Float64Array,
    anchors: Float64Array,
    degree: number,
  ): void;
  evalRank4FmNative(features: Float64Array, vFactorsFlat: Float64Array): Float64Array;
  evalRank4FmSliceNative(
    featuresFlat: Float64Array,
    nFrames: number,
    dDim: number,
    vFactorsFlat: Float64Array,
    out: Float64Array,
  ): void;
  evalChebyshevLfoNative(cosPhi: number): Float64Array;
  evalChebyshevLfoSliceNative(cosPhi: Float64Array, out: Float64Array): void;
  evalTrustRegionSoftclipNative(z: number, mh: number): number;
  evalTrustRegionSoftclipSliceNative(z: Float64Array, out: Float64Array, mh: number): void;
  cubicLagrangeWeightsNative(mu: number): Float64Array;
  cubicLagrangeWeightsSliceNative(mu: Float64Array, out: Float64Array): void;
  givensHouseholder8ChNative(x: Float64Array, cosPsi: number, sinPsi: number): Float64Array;
  givensHouseholder8ChSliceNative(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
  ): void;
  givensHouseholder16ChNative(
    x: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos: Float64Array,
    crossSin: Float64Array,
  ): Float64Array;
  givensHouseholder16ChSliceNative(
    xFlat: Float64Array,
    out: Float64Array,
    cosPsi: number,
    sinPsi: number,
    crossCos: Float64Array,
    crossSin: Float64Array,
  ): void;
}
