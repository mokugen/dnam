/**
 * Typed raw WebAssembly SIMD128 C-ABI export table (`RawWasmExports`).
 *
 * @module
 */

export interface RawWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  dnam_wasm_alloc(size: number, align: number): number;
  dnam_wasm_dealloc(ptr: number, size: number, align: number): void;
  dnam_wasm_last_error_ptr(): number;
  dnam_wasm_last_error_len(): number;
  dnam_wasm_result_ptr(): number;
  dnam_wasm_result_len(): number;
  dnam_wasm_arena_ptr(): number;
  dnam_wasm_arena_len(): number;

  dnam_wasm_engine_create(
    containerPtr: number,
    containerLen: number,
    tier: number,
    sampleRateHz: number,
    backendCode: number,
    maxSupportedTier: number,
    offlineBounce: number,
    maximizeTierOnBounce: number,
  ): number;
  dnam_wasm_engine_free(ptr: number): void;
  dnam_wasm_engine_reload(
    ptr: number,
    containerPtr: number,
    containerLen: number,
    tier: number,
    sampleRateHz: number,
  ): number;
  dnam_wasm_engine_backend_code(ptr: number): number;
  dnam_wasm_engine_tier(ptr: number): number;
  dnam_wasm_engine_set_tier(ptr: number, tier: number): number;
  dnam_wasm_engine_max_supported_tier(ptr: number): number;
  dnam_wasm_engine_set_max_supported_tier(ptr: number, tier: number): number;
  dnam_wasm_engine_max_available_tier(ptr: number): number;
  dnam_wasm_engine_effective_tier(ptr: number): number;
  dnam_wasm_engine_sample_rate(ptr: number): number;
  dnam_wasm_engine_set_sample_rate(ptr: number, sampleRateHz: number): number;
  dnam_wasm_engine_latency_samples(ptr: number): number;
  dnam_wasm_engine_offline_bounce(ptr: number): number;
  dnam_wasm_engine_set_offline_bounce(ptr: number, enabled: number): void;
  dnam_wasm_engine_maximize_tier_on_bounce(ptr: number): number;
  dnam_wasm_engine_set_maximize_tier_on_bounce(ptr: number, enabled: number): void;
  dnam_wasm_engine_num_controls(ptr: number): number;
  dnam_wasm_engine_raw_controls_into(ptr: number, outPtr: number, len: number): number;
  dnam_wasm_engine_clamped_controls_into(ptr: number, outPtr: number, len: number): number;
  dnam_wasm_engine_clamped_control_value(ptr: number, ctrlIdx: number, outPtr: number): number;
  dnam_wasm_engine_set_control_value(ptr: number, ctrlIdx: number, value: number): number;
  dnam_wasm_engine_set_control_immediate(ptr: number, ctrlIdx: number, value: number): number;
  dnam_wasm_engine_set_controls_immediate_batch(
    ptr: number,
    indicesPtr: number,
    valuesPtr: number,
    len: number,
  ): number;
  dnam_wasm_engine_reset_controls_to_defaults(ptr: number): void;
  dnam_wasm_engine_reset(ptr: number): void;
  dnam_wasm_engine_supports_freeze(ptr: number): number;
  dnam_wasm_engine_is_frozen(ptr: number): number;
  dnam_wasm_engine_set_freeze(ptr: number, freeze: number): void;
  dnam_wasm_engine_set_freeze_amount(ptr: number, alpha: number): void;
  dnam_wasm_engine_bypass_state(ptr: number): number;
  dnam_wasm_engine_set_bypass(ptr: number, bypassed: number, trails: number): void;
  dnam_wasm_engine_stereo_width(ptr: number): number;
  dnam_wasm_engine_set_stereo_width(ptr: number, width: number): void;
  dnam_wasm_engine_configure_host_mix(
    ptr: number,
    blendWet: number,
    outputGainDb: number,
    applyUnityLeveling: number,
  ): void;
  dnam_wasm_engine_current_bpm(ptr: number): number;
  dnam_wasm_engine_sync_tempo(
    ptr: number,
    bpm: number,
    subPtr: number,
    subLen: number,
    foldOctave: number,
  ): number;
  dnam_wasm_engine_register_tap_tempo(
    ptr: number,
    subPtr: number,
    subLen: number,
    foldOctave: number,
    outBpmPtr: number,
  ): number;
  dnam_wasm_engine_apply_preset(ptr: number, presetPtr: number, presetLen: number): number;
  dnam_wasm_engine_process_stereo_in_place(
    ptr: number,
    leftIoPtr: number,
    rightIoPtr: number,
    numSamples: number,
  ): number;
  dnam_wasm_engine_process_stereo(
    ptr: number,
    leftInPtr: number,
    rightInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
    blockSize: number,
  ): number;
  dnam_wasm_engine_process_mono_to_stereo(
    ptr: number,
    monoInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
    blockSize: number,
  ): number;
  dnam_wasm_engine_process_mono(
    ptr: number,
    monoInPtr: number,
    monoOutPtr: number,
    numSamples: number,
    blockSize: number,
  ): number;
  dnam_wasm_engine_render_offline_bounce(
    ptr: number,
    leftInPtr: number,
    rightInPtr: number,
    leftOutPtr: number,
    rightOutPtr: number,
    numSamples: number,
  ): number;
  dnam_wasm_engine_telemetry_into(ptr: number, outPtr: number, outLen: number): number;

  dnam_wasm_control_space_create(hdrPtr: number, hdrLen: number): number;
  dnam_wasm_control_space_free(ptr: number): void;
  dnam_wasm_control_space_num_controls(ptr: number): number;
  dnam_wasm_control_space_clamp_into(
    ptr: number,
    rawPtr: number,
    clampedPtr: number,
    normPtr: number,
    len: number,
  ): number;
  dnam_wasm_control_space_to_physical(
    ptr: number,
    ctrlIdx: number,
    normalizedU: number,
    outPtr: number,
  ): number;
  dnam_wasm_control_space_to_normalized(
    ptr: number,
    ctrlIdx: number,
    physicalVal: number,
    outPtr: number,
  ): number;
  dnam_wasm_compute_zoh(
    srTrain: number,
    srHost: number,
    rAc: number,
    rDc: number,
    tauRef: number,
    switchXfade: number,
    out9Ptr: number,
  ): number;
  dnam_wasm_control_stride_for_rate(sampleRateHz: number): number;
  dnam_wasm_control_tick_ratio(sampleRateHz: number): number;
  dnam_wasm_eval_subdivision_phase(
    beatMod12: number,
    withinBeatQ32: number,
    subPtr: number,
    subLen: number,
    offsetQ32: number,
    out3Ptr: number,
  ): number;
  dnam_wasm_eval_3node_calibration(anchorsPtr: number, u: number, outPtr: number): number;
  dnam_wasm_invert_tempo_calibration(
    declMin: number,
    declMax: number,
    safeMin: number,
    safeMax: number,
    anchorsPtr: number,
    target: number,
    flushSubnormals: number,
    out5Ptr: number,
  ): number;

  dnam_wasm_validate_container_with_key(
    cPtr: number,
    cLen: number,
    idPtr: number,
    idLen: number,
    secPtr: number,
    secLen: number,
    mktPtr: number,
    mktLen: number,
  ): number;
  dnam_wasm_build_reference_container(
    supportsFreeze: number,
    inputs: number,
    modePtr: number,
    modeLen: number,
  ): number;
  dnam_wasm_build_dnam_container(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    encPtr: number,
    encLen: number,
    hasSig: number,
  ): number;
  dnam_wasm_update_header_in_place(
    cPtr: number,
    cLen: number,
    hdrPtr: number,
    hdrLen: number,
  ): number;
  dnam_wasm_strip_container_to_tier(cPtr: number, cLen: number, maxTier: number): number;
  dnam_wasm_parse_schema_version(verPtr: number, verLen: number, out4Ptr: number): number;
  dnam_wasm_crc32c(dataPtr: number, dataLen: number): number;
  dnam_wasm_validate_and_normalize_header(hdrPtr: number, hdrLen: number): number;
  dnam_wasm_canonicalize_jcs(
    jsonPtr: number,
    jsonLen: number,
    stripMutable: number,
    rejectFloats: number,
  ): number;
  dnam_wasm_build_dmu1_frame(jsonPtr: number, jsonLen: number, capacity: number): number;
  dnam_wasm_parse_dmu1_frame(framePtr: number, frameLen: number): number;
  dnam_wasm_validate_v2_open_dnam_layout(cPtr: number, cLen: number): number;
  dnam_wasm_strip_v2_open_dnam_container(cPtr: number, cLen: number, targetTier: number): number;
  dnam_wasm_uplo_u_packed_len(nDim: number): number;
  dnam_wasm_pack_uplo_u_f64_into(
    densePtr: number,
    denseLen: number,
    nDim: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_unpack_uplo_u_f64_into(
    packedPtr: number,
    packedLen: number,
    nTier: number,
    outPtr: number,
    outLen: number,
  ): number;

  dnam_wasm_verify_container_provenance(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    hasSig: number,
    idPtr: number,
    idLen: number,
    secPtr: number,
    secLen: number,
    mktPtr: number,
    mktLen: number,
  ): number;
  dnam_wasm_enforce_distribution_policy(
    hdrPtr: number,
    hdrLen: number,
    profPtr: number,
    profLen: number,
  ): number;
  dnam_wasm_evaluate_anticlone_policy(
    fpMatched: number,
    wmIdentified: number,
    indepEvidence: number,
  ): number;
  dnam_wasm_immutable_header_sha256(hdrPtr: number, hdrLen: number): number;
  dnam_wasm_sha256_hex(dataPtr: number, dataLen: number): number;
  dnam_wasm_hmac_sha256_hex(keyPtr: number, keyLen: number, msgPtr: number, msgLen: number): number;
  dnam_wasm_ed25519_pubkey_hex(seedPtr: number, seedLen: number): number;
  dnam_wasm_ed25519_sign_hex(
    seedPtr: number,
    seedLen: number,
    msgPtr: number,
    msgLen: number,
  ): number;
  dnam_wasm_x25519_pubkey_hex(secPtr: number, secLen: number): number;
  dnam_wasm_build_creator_signing_message(
    hdrShaPtr: number,
    hdrShaLen: number,
    arenaShaPtr: number,
    arenaShaLen: number,
  ): number;
  dnam_wasm_build_marketplace_signing_message(
    hdrShaPtr: number,
    hdrShaLen: number,
    arenaShaPtr: number,
    arenaShaLen: number,
    rcptPtr: number,
    rcptLen: number,
  ): number;
  dnam_wasm_sign_creator_header(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    seedPtr: number,
    seedLen: number,
    signerPtr: number,
    signerLen: number,
    includeTierPrefixes: number,
  ): number;
  dnam_wasm_sign_marketplace_receipt(
    hdrPtr: number,
    hdrLen: number,
    arenaPtr: number,
    arenaLen: number,
    seedPtr: number,
    seedLen: number,
    specPtr: number,
    specLen: number,
  ): number;
  dnam_wasm_sign_oem_device_cert(
    seedPtr: number,
    seedLen: number,
    vendorPtr: number,
    vendorLen: number,
    modelPtr: number,
    modelLen: number,
    devIdPtr: number,
    devIdLen: number,
    pkPtr: number,
    pkLen: number,
  ): number;
  dnam_wasm_verify_oem_device_cert(
    certPtr: number,
    certLen: number,
    pkHexPtr: number,
    pkHexLen: number,
  ): number;
  dnam_wasm_build_oem_cert_v2_signing_message(
    vendorPtr: number,
    vendorLen: number,
    modelPtr: number,
    modelLen: number,
    devIdPtr: number,
    devIdLen: number,
    pkPtr: number,
    pkLen: number,
    issPkPtr: number,
    issPkLen: number,
    assurance: number,
    polPtr: number,
    polLen: number,
    evPtr: number,
    evLen: number,
  ): number;
  dnam_wasm_chacha20_xor_arena(
    arenaPtr: number,
    arenaLen: number,
    keyPtr: number,
    keyLen: number,
    noncePtr: number,
    nonceLen: number,
    startCounter: number,
  ): number;
  dnam_wasm_wrap_device_cek_hex(
    ephPtr: number,
    ephLen: number,
    devPkPtr: number,
    devPkLen: number,
    devIdPtr: number,
    devIdLen: number,
    cekPtr: number,
    cekLen: number,
  ): number;
  dnam_wasm_seal_container(
    innerPtr: number,
    innerLen: number,
    certsPtr: number,
    certsLen: number,
    oemPksPtr: number,
    oemPksLen: number,
    mktSeedPtr: number,
    mktSeedLen: number,
    ephSecPtr: number,
    ephSecLen: number,
    cekPtr: number,
    cekLen: number,
    noncePtr: number,
    nonceLen: number,
  ): number;
  dnam_wasm_unseal_container(
    sealedPtr: number,
    sealedLen: number,
    idPtr: number,
    idLen: number,
    secPtr: number,
    secLen: number,
    mktPtr: number,
    mktLen: number,
  ): number;
  dnam_wasm_encode_dau1_envelope(specPtr: number, specLen: number): number;
  dnam_wasm_decode_dau1_envelope(wirePtr: number, wireLen: number): number;
  dnam_wasm_build_dme1_header(
    profile: number,
    recipientCount: number,
    ciphertextLength: bigint,
  ): number;
  dnam_wasm_parse_dme1_plaintext(
    ptPtr: number,
    ptLen: number,
    digestsPtr: number,
    digestsLen: number,
  ): number;
  dnam_wasm_encode_recipient_key_ids(jsonPtr: number, jsonLen: number): number;
  dnam_wasm_parse_recipient_key_ids(
    dataPtr: number,
    dataLen: number,
    maxRecipients: number,
  ): number;
  dnam_wasm_encode_hpke_slots(jsonPtr: number, jsonLen: number): number;
  dnam_wasm_parse_hpke_slots(
    dataPtr: number,
    dataLen: number,
    expectedCount: number,
    maxRecipients: number,
  ): number;
  dnam_wasm_encode_record_directory(maximumTier: number, jsonPtr: number, jsonLen: number): number;
  dnam_wasm_parse_record_directory(dataPtr: number, dataLen: number, maximumTier: number): number;
  dnam_wasm_compute_preview_window(
    anchor: number,
    delta: number,
    safeMin: number,
    safeMax: number,
    out2Ptr: number,
  ): number;
  dnam_wasm_validate_preview_switch_options(jsonPtr: number, jsonLen: number): number;
  dnam_wasm_validate_royalty_splits(
    tierPtr: number,
    tierLen: number,
    splitsPtr: number,
    splitsLen: number,
  ): number;
  dnam_wasm_apply_isometric_gauge_watermark(
    arenaPtr: number,
    arenaLen: number,
    kappaPtr: number,
    kappaLen: number,
    qPtr: number,
    qLen: number,
  ): number;
  dnam_wasm_generate_tardos_code(
    secPtr: number,
    secLen: number,
    modelPtr: number,
    modelLen: number,
    licPtr: number,
    licLen: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_apply_tardos_watermark(
    arenaPtr: number,
    arenaLen: number,
    codePtr: number,
    codeLen: number,
    epsilon: number,
  ): number;
  dnam_wasm_score_tardos_watermark(
    masterPtr: number,
    masterLen: number,
    suspectPtr: number,
    suspectLen: number,
    codePtr: number,
    codeLen: number,
    outScorePtr: number,
  ): number;
  dnam_wasm_extract_structural_hash(
    arenaPtr: number,
    arenaLen: number,
    outPtr: number,
    outCap: number,
  ): number;
  dnam_wasm_structural_cosine_similarity(
    aPtr: number,
    aLen: number,
    bPtr: number,
    bLen: number,
    outSimPtr: number,
  ): number;
  dnam_wasm_derive_watermark_seed_hex(
    keyPtr: number,
    keyLen: number,
    mktPkPtr: number,
    mktPkLen: number,
    wmKidPtr: number,
    wmKidLen: number,
    manShaPtr: number,
    manShaLen: number,
    modelIdPtr: number,
    modelIdLen: number,
    profShaPtr: number,
    profShaLen: number,
    owningTier: number,
    licPtr: number,
    licLen: number,
    buyerPtr: number,
    buyerLen: number,
  ): number;
  dnam_wasm_extract_stage3_fingerprint(
    firPtr: number,
    firLen: number,
    kautzPtr: number,
    kautzLen: number,
    addedModes: number,
    outPtr: number,
    outCap: number,
  ): number;

  dnam_wasm_catmull_rom_hermite(
    sM1: number,
    s0: number,
    s1: number,
    s2: number,
    mu: number,
  ): number;
  dnam_wasm_catmull_rom_hermite_slice(
    sM1Ptr: number,
    s0Ptr: number,
    s1Ptr: number,
    s2Ptr: number,
    muPtr: number,
    outPtr: number,
    len: number,
  ): number;
  dnam_wasm_op_t_coercive_step(
    uIn: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
    dcXPrev: number,
    dcYPrev: number,
    out3Ptr: number,
  ): void;
  dnam_wasm_op_t_tank_slice(
    inpPtr: number,
    outPtr: number,
    len: number,
    gLin: number,
    sigmaKnee: number,
    vCeil: number,
    rDc: number,
  ): number;
  dnam_wasm_op_m_modal_step(
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
    out3Ptr: number,
  ): void;
  dnam_wasm_op_m_modal_slice(
    inpPtr: number,
    outPtr: number,
    len: number,
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
    reset: number,
    state2Ptr: number,
  ): number;
  dnam_wasm_unit_rotor(phi: number, out2Ptr: number): void;
  dnam_wasm_unit_rotor_slice(phiPtr: number, len: number, outPtr: number, outLen: number): number;
  dnam_wasm_poly_adaa_algebraic(uPrev: number, uCurr: number, beta: number, delta: number): number;
  dnam_wasm_poly_adaa_step(
    x: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
    out4Ptr: number,
  ): void;
  dnam_wasm_poly_adaa_slice(
    inpPtr: number,
    outPtr: number,
    len: number,
    bConv: number,
    alpha: number,
    beta: number,
    delta: number,
    rAc: number,
    uPrev: number,
    aPrev: number,
    mDc: number,
    reset: number,
    state3Ptr: number,
  ): number;
  dnam_wasm_eval_lobatto_basis(
    u: number,
    anchorsPtr: number,
    anchorsLen: number,
    degree: number,
    out3Ptr: number,
  ): number;
  dnam_wasm_eval_lobatto_basis_slice(
    uPtr: number,
    uLen: number,
    outPtr: number,
    outLen: number,
    anchorsPtr: number,
    anchorsLen: number,
    degree: number,
  ): number;
  dnam_wasm_eval_rank4_fm(
    featPtr: number,
    featLen: number,
    vfPtr: number,
    vfLen: number,
    out4Ptr: number,
  ): number;
  dnam_wasm_eval_rank4_fm_slice(
    featPtr: number,
    featLen: number,
    nFrames: number,
    dDim: number,
    vfPtr: number,
    vfLen: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_eval_chebyshev_lfo(cosPhi: number, out4Ptr: number): void;
  dnam_wasm_eval_chebyshev_lfo_slice(
    cosPhiPtr: number,
    len: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_eval_trust_region_softclip(z: number, mh: number): number;
  dnam_wasm_eval_trust_region_softclip_slice(
    zPtr: number,
    outPtr: number,
    len: number,
    mh: number,
  ): number;
  dnam_wasm_cubic_lagrange_weights(mu: number, out4Ptr: number): void;
  dnam_wasm_cubic_lagrange_weights_slice(
    muPtr: number,
    len: number,
    outPtr: number,
    outLen: number,
  ): number;
  dnam_wasm_givens_householder_8ch(
    xPtr: number,
    xLen: number,
    cosPsi: number,
    sinPsi: number,
    out8Ptr: number,
  ): number;
  dnam_wasm_givens_householder_8ch_slice(
    xPtr: number,
    outPtr: number,
    len: number,
    cosPsi: number,
    sinPsi: number,
  ): number;
  dnam_wasm_givens_householder_16ch(
    xPtr: number,
    xLen: number,
    cosPsi: number,
    sinPsi: number,
    ccPtr: number,
    ccLen: number,
    csPtr: number,
    csLen: number,
    out16Ptr: number,
  ): number;
  dnam_wasm_givens_householder_16ch_slice(
    xPtr: number,
    outPtr: number,
    len: number,
    cosPsi: number,
    sinPsi: number,
    ccPtr: number,
    ccLen: number,
    csPtr: number,
    csLen: number,
  ): number;
}
