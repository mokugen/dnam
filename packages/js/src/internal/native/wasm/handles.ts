/**
 * Stateful WebAssembly `SIMD128` engine and control-space handles (`WasmEngineHandle`,
 * `WasmControlSpaceHandle`) (`ref[sdk.ts.engine.zero-copy-and-async]`). Zero `node:*` dependencies.
 *
 * @module
 */

import { DnamError } from "../../../errors.js";
import type { ActiveBackend } from "../../../types.js";
import { backendFromCode } from "../../coerce.js";
import {
  type NativeControlSpaceHandle,
  type NativeEngineHandle,
  TELEMETRY_F64_LEN,
} from "../driver.js";
import type { WasmMemoryArena } from "./arena.js";

export class WasmEngineHandle implements NativeEngineHandle {
  readonly #mem: WasmMemoryArena;
  #ptr: number;
  #busy = false;

  constructor(mem: WasmMemoryArena, ptr: number) {
    this.#mem = mem;
    this.#ptr = ptr;
  }

  #ensurePtr(): number {
    if (this.#ptr === 0) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Engine has been disposed");
    }
    return this.#ptr;
  }

  #ensureAudioIdlePtr(): number {
    const ptr = this.#ensurePtr();
    if (this.#busy) {
      throw new DnamError(
        "ERR_DNAM_BUSY",
        "Engine is busy executing an in-flight processAsync task",
      );
    }
    return ptr;
  }

  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void {
    const ptr = this.#ensurePtr();
    this.#mem.withBytes(containerBytes, (cPtr, cLen) => {
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_reload(ptr, cPtr, cLen, tier ?? -1, sampleRateHz ?? 0),
      );
    });
  }

  backend(): ActiveBackend {
    return backendFromCode(this.#mem.exports.dnam_wasm_engine_backend_code(this.#ensurePtr()));
  }

  tier(): number {
    return this.#mem.exports.dnam_wasm_engine_tier(this.#ensurePtr());
  }

  setTier(tier: number): void {
    this.#mem.checkStatus(this.#mem.exports.dnam_wasm_engine_set_tier(this.#ensurePtr(), tier));
  }

  maxSupportedTier(): number {
    return this.#mem.exports.dnam_wasm_engine_max_supported_tier(this.#ensurePtr());
  }

  setMaxSupportedTier(tier: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_max_supported_tier(this.#ensurePtr(), tier),
    );
  }

  maxAvailableTier(): number {
    return this.#mem.exports.dnam_wasm_engine_max_available_tier(this.#ensurePtr());
  }

  effectiveTier(): number {
    return this.#mem.exports.dnam_wasm_engine_effective_tier(this.#ensurePtr());
  }

  sampleRate(): number {
    return this.#mem.exports.dnam_wasm_engine_sample_rate(this.#ensurePtr());
  }

  setSampleRate(sampleRateHz: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_sample_rate(this.#ensureAudioIdlePtr(), sampleRateHz),
    );
  }

  latencySamples(): number {
    return this.#mem.exports.dnam_wasm_engine_latency_samples(this.#ensurePtr());
  }

  offlineBounce(): boolean {
    return this.#mem.exports.dnam_wasm_engine_offline_bounce(this.#ensurePtr()) !== 0;
  }

  setOfflineBounce(enabled: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_offline_bounce(this.#ensurePtr(), enabled ? 1 : 0);
  }

  maximizeTierOnBounce(): boolean {
    return this.#mem.exports.dnam_wasm_engine_maximize_tier_on_bounce(this.#ensurePtr()) !== 0;
  }

  setMaximizeTierOnBounce(enabled: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_maximize_tier_on_bounce(
      this.#ensurePtr(),
      enabled ? 1 : 0,
    );
  }

  numControls(): number {
    return this.#mem.exports.dnam_wasm_engine_num_controls(this.#ensurePtr());
  }

  rawControlValues(): Float32Array {
    const ptr = this.#ensurePtr();
    const n = this.#mem.exports.dnam_wasm_engine_num_controls(ptr);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_raw_controls_into(ptr, this.#mem.smallF32Ptr, n),
    );
    const start = this.#mem.smallF32Ptr >>> 2;
    return this.#mem.f32().slice(start, start + n);
  }

  clampedControlValue(ctrlIdx: number): number {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_clamped_control_value(ptr, ctrlIdx, this.#mem.smallF32Ptr),
    );
    return this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 0;
  }

  setControlValue(ctrlIdx: number, value: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_control_value(this.#ensurePtr(), ctrlIdx, value),
    );
  }

  setControlImmediate(ctrlIdx: number, value: number): void {
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_control_immediate(this.#ensurePtr(), ctrlIdx, value),
    );
  }

  setControlsImmediateBatch(indices: readonly number[], values: readonly number[]): void {
    const ptr = this.#ensurePtr();
    const len = indices.length;
    if (len === 0) {
      return;
    }
    const u32View = this.#mem.u32();
    const f32View = this.#mem.f32();
    const u32Start = this.#mem.smallU32Ptr >>> 2;
    const f32Start = this.#mem.smallF32Ptr >>> 2;
    for (let i = 0; i < len; i++) {
      u32View[u32Start + i] = indices[i] ?? 0;
      f32View[f32Start + i] = values[i] ?? 0;
    }
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_set_controls_immediate_batch(
        ptr,
        this.#mem.smallU32Ptr,
        this.#mem.smallF32Ptr,
        len,
      ),
    );
  }

  resetControlsToDefaults(): void {
    this.#mem.exports.dnam_wasm_engine_reset_controls_to_defaults(this.#ensurePtr());
  }

  reset(): void {
    this.#mem.exports.dnam_wasm_engine_reset(this.#ensurePtr());
  }

  supportsFreeze(): boolean {
    return this.#mem.exports.dnam_wasm_engine_supports_freeze(this.#ensurePtr()) !== 0;
  }

  isFrozen(): boolean {
    return this.#mem.exports.dnam_wasm_engine_is_frozen(this.#ensurePtr()) !== 0;
  }

  setFreeze(freeze: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_freeze(this.#ensurePtr(), freeze ? 1 : 0);
  }

  setFreezeAmount(alpha: number): void {
    this.#mem.exports.dnam_wasm_engine_set_freeze_amount(this.#ensurePtr(), alpha);
  }

  bypassState(): readonly [boolean, boolean] {
    const bits = this.#mem.exports.dnam_wasm_engine_bypass_state(this.#ensurePtr());
    return [(bits & 1) !== 0, (bits & 2) !== 0];
  }

  setBypass(bypassed: boolean, trails: boolean): void {
    this.#mem.exports.dnam_wasm_engine_set_bypass(
      this.#ensurePtr(),
      bypassed ? 1 : 0,
      trails ? 1 : 0,
    );
  }

  stereoWidth(): number {
    return this.#mem.exports.dnam_wasm_engine_stereo_width(this.#ensurePtr());
  }

  setStereoWidth(width: number): void {
    this.#mem.exports.dnam_wasm_engine_set_stereo_width(this.#ensurePtr(), width);
  }

  configureHostMix(blendWet: number, outputGainDb: number, applyUnityLeveling: boolean): void {
    this.#mem.exports.dnam_wasm_engine_configure_host_mix(
      this.#ensurePtr(),
      blendWet,
      outputGainDb,
      applyUnityLeveling ? 1 : 0,
    );
  }

  currentBpm(): number {
    return this.#mem.exports.dnam_wasm_engine_current_bpm(this.#ensurePtr());
  }

  syncTempo(bpm: number, subdivision: string | undefined, foldOctave: boolean): number {
    const ptr = this.#ensurePtr();
    return this.#mem.withString(subdivision, (sPtr, sLen) =>
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_sync_tempo(ptr, bpm, sPtr, sLen, foldOctave ? 1 : 0),
      ),
    );
  }

  registerTapTempo(
    subdivision: string | undefined,
    foldOctave: boolean,
  ): readonly [number, number] {
    const ptr = this.#ensurePtr();
    return this.#mem.withString(subdivision, (sPtr, sLen) => {
      const count = this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_register_tap_tempo(
          ptr,
          sPtr,
          sLen,
          foldOctave ? 1 : 0,
          this.#mem.smallF32Ptr,
        ),
      );
      const bpm = this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 120;
      return [bpm, count] as const;
    });
  }

  applyPreset(presetId: string): void {
    const ptr = this.#ensurePtr();
    this.#mem.withString(presetId, (pPtr, pLen) => {
      this.#mem.checkStatus(this.#mem.exports.dnam_wasm_engine_apply_preset(ptr, pPtr, pLen));
    });
  }

  processStereoInPlace(leftIo: Float32Array, rightIo: Float32Array): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = leftIo.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch1InPtr } = this.#mem.ensureAudioScratch(Math.max(n, rightIo.length));
    const f32 = this.#mem.f32();
    const lIdx = ch0InPtr >>> 2;
    const rIdx = ch1InPtr >>> 2;
    f32.set(leftIo, lIdx);
    f32.set(rightIo, rIdx);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_stereo_in_place(ptr, ch0InPtr, ch1InPtr, n),
    );
    const f32After = this.#mem.f32();
    leftIo.set(f32After.subarray(lIdx, lIdx + n));
    rightIo.set(f32After.subarray(rIdx, rIdx + n));
  }

  processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = leftIn.length;
    if (n === 0) {
      return;
    }
    if (!blockSize || blockSize >= n) {
      const { ch0InPtr, ch1InPtr } = this.#mem.ensureAudioScratch(n);
      const f32 = this.#mem.f32();
      const lIdx = ch0InPtr >>> 2;
      const rIdx = ch1InPtr >>> 2;
      f32.set(leftIn, lIdx);
      f32.set(rightIn, rIdx);
      this.#mem.checkStatus(
        this.#mem.exports.dnam_wasm_engine_process_stereo_in_place(ptr, ch0InPtr, ch1InPtr, n),
      );
      const f32After = this.#mem.f32();
      leftOut.set(f32After.subarray(lIdx, lIdx + n));
      rightOut.set(f32After.subarray(rIdx, rIdx + n));
      return;
    }
    const { ch0InPtr, ch1InPtr, ch0OutPtr, ch1OutPtr } = this.#mem.ensureAudioScratch(n);
    const f32 = this.#mem.f32();
    f32.set(leftIn, ch0InPtr >>> 2);
    f32.set(rightIn, ch1InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_stereo(
        ptr,
        ch0InPtr,
        ch1InPtr,
        ch0OutPtr,
        ch1OutPtr,
        n,
        blockSize,
      ),
    );
    const f32After = this.#mem.f32();
    const loIdx = ch0OutPtr >>> 2;
    const roIdx = ch1OutPtr >>> 2;
    leftOut.set(f32After.subarray(loIdx, loIdx + n));
    rightOut.set(f32After.subarray(roIdx, roIdx + n));
  }

  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = monoIn.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch0OutPtr, ch1OutPtr } = this.#mem.ensureAudioScratch(n);
    this.#mem.f32().set(monoIn, ch0InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_mono_to_stereo(
        ptr,
        ch0InPtr,
        ch0OutPtr,
        ch1OutPtr,
        n,
        blockSize ?? 0,
      ),
    );
    const f32After = this.#mem.f32();
    const loIdx = ch0OutPtr >>> 2;
    const roIdx = ch1OutPtr >>> 2;
    leftOut.set(f32After.subarray(loIdx, loIdx + n));
    rightOut.set(f32After.subarray(roIdx, roIdx + n));
  }

  processMono(monoIn: Float32Array, monoOut: Float32Array, blockSize?: number): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = monoIn.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch0OutPtr } = this.#mem.ensureAudioScratch(n);
    this.#mem.f32().set(monoIn, ch0InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_process_mono(ptr, ch0InPtr, ch0OutPtr, n, blockSize ?? 0),
    );
    const moIdx = ch0OutPtr >>> 2;
    monoOut.set(this.#mem.f32().subarray(moIdx, moIdx + n));
  }

  renderOfflineBounce(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
  ): void {
    const ptr = this.#ensureAudioIdlePtr();
    const n = leftIn.length;
    if (n === 0) {
      return;
    }
    const { ch0InPtr, ch1InPtr, ch0OutPtr, ch1OutPtr } = this.#mem.ensureAudioScratch(n);
    const f32 = this.#mem.f32();
    f32.set(leftIn, ch0InPtr >>> 2);
    f32.set(rightIn, ch1InPtr >>> 2);
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_render_offline_bounce(
        ptr,
        ch0InPtr,
        ch1InPtr,
        ch0OutPtr,
        ch1OutPtr,
        n,
      ),
    );
    const f32After = this.#mem.f32();
    const loIdx = ch0OutPtr >>> 2;
    const roIdx = ch1OutPtr >>> 2;
    leftOut.set(f32After.subarray(loIdx, loIdx + n));
    rightOut.set(f32After.subarray(roIdx, roIdx + n));
  }

  processStereoAsync(
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
  ): Promise<readonly [Float32Array, Float32Array]> {
    this.#ensureAudioIdlePtr();
    if (rightIn.length !== leftIn.length) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `right_in length (${String(rightIn.length)}) must match left_in (${String(leftIn.length)})`,
      );
    }
    const leftSnapshot = leftIn.slice();
    const rightSnapshot = rightIn.slice();
    this.#busy = true;
    return Promise.resolve().then(() => {
      try {
        const leftOut = new Float32Array(leftSnapshot.length);
        const rightOut = new Float32Array(rightSnapshot.length);
        this.#busy = false;
        this.processStereo(leftSnapshot, rightSnapshot, leftOut, rightOut, blockSize);
        return [leftOut, rightOut] as const;
      } finally {
        this.#busy = false;
      }
    });
  }

  telemetryInto(out56: Float64Array): void {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_engine_telemetry_into(
        ptr,
        this.#mem.smallF64Ptr,
        TELEMETRY_F64_LEN,
      ),
    );
    const start = this.#mem.smallF64Ptr >>> 3;
    out56.set(this.#mem.f64().subarray(start, start + TELEMETRY_F64_LEN));
  }

  dispose(): void {
    if (this.#ptr !== 0) {
      this.#mem.exports.dnam_wasm_engine_free(this.#ptr);
      this.#ptr = 0;
    }
  }
}

export class WasmControlSpaceHandle implements NativeControlSpaceHandle {
  readonly #mem: WasmMemoryArena;
  #ptr: number;

  constructor(mem: WasmMemoryArena, ptr: number) {
    this.#mem = mem;
    this.#ptr = ptr;
  }

  #ensurePtr(): number {
    if (this.#ptr === 0) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return this.#ptr;
  }

  clampInto(rawNumeric: Float32Array, clampedOut: Float32Array, normOut: Float32Array): number {
    const ptr = this.#ensurePtr();
    const n = rawNumeric.length;
    if (n === 0) {
      return 0;
    }
    const baseIdx = this.#mem.smallF32Ptr >>> 2;
    const f32 = this.#mem.f32();
    f32.set(rawNumeric, baseIdx);
    const clampedPtr = this.#mem.smallF32Ptr + n * 4;
    const normPtr = clampedPtr + n * 4;
    const mask =
      this.#mem.exports.dnam_wasm_control_space_clamp_into(
        ptr,
        this.#mem.smallF32Ptr,
        clampedPtr,
        normPtr,
        n,
      ) >>> 0;
    const f32After = this.#mem.f32();
    const cIdx = clampedPtr >>> 2;
    const nIdx = normPtr >>> 2;
    clampedOut.set(f32After.subarray(cIdx, cIdx + n));
    normOut.set(f32After.subarray(nIdx, nIdx + n));
    return mask;
  }

  toPhysical(ctrlIdx: number, normalizedU: number): number {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_control_space_to_physical(
        ptr,
        ctrlIdx,
        normalizedU,
        this.#mem.smallF32Ptr,
      ),
    );
    return this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 0;
  }

  toNormalized(ctrlIdx: number, physicalVal: number): number {
    const ptr = this.#ensurePtr();
    this.#mem.checkStatus(
      this.#mem.exports.dnam_wasm_control_space_to_normalized(
        ptr,
        ctrlIdx,
        physicalVal,
        this.#mem.smallF32Ptr,
      ),
    );
    return this.#mem.f32()[this.#mem.smallF32Ptr >>> 2] ?? 0;
  }

  dispose(): void {
    if (this.#ptr !== 0) {
      this.#mem.exports.dnam_wasm_control_space_free(this.#ptr);
      this.#ptr = 0;
    }
  }
}
