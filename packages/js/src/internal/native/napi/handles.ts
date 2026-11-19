/**
 * Stateful `N-API v10` engine and control-space handles (`NapiEngineHandle`,
 * `NapiControlSpaceHandle`) (`ref[sdk.ts.engine.zero-copy-and-async]`).
 *
 * @module
 */

import { DnamError, fromNativeError } from "../../../errors.js";
import type { ActiveBackend } from "../../../types.js";
import { coerceBackend } from "../../coerce.js";
import type { NativeControlSpaceHandle, NativeEngineHandle } from "../driver.js";
import type { RawNapiControlSpace, RawNapiEngine } from "./bindings.js";

export function callNapi<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    throw fromNativeError(err);
  }
}

export class NapiEngineHandle implements NativeEngineHandle {
  readonly #raw: RawNapiEngine;
  #disposed = false;
  #busy = false;

  constructor(raw: RawNapiEngine) {
    this.#raw = raw;
  }

  #ensureAlive(): RawNapiEngine {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Engine has been disposed");
    }
    return this.#raw;
  }

  #ensureAudioIdle(): RawNapiEngine {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "Engine has been disposed");
    }
    if (this.#busy) {
      throw new DnamError(
        "ERR_DNAM_BUSY",
        "Engine is busy executing an in-flight processAsync task",
      );
    }
    return this.#raw;
  }

  reload(containerBytes: Uint8Array, tier?: number, sampleRateHz?: number): void {
    callNapi(() => this.#ensureAlive().reload(containerBytes, tier, sampleRateHz));
  }

  backend(): ActiveBackend {
    return coerceBackend(callNapi(() => this.#ensureAlive().backend)) as ActiveBackend;
  }

  tier(): number {
    return callNapi(() => this.#ensureAlive().tier);
  }

  setTier(tier: number): void {
    callNapi(() => this.#ensureAlive().setTier(tier));
  }

  maxSupportedTier(): number {
    return callNapi(() => this.#ensureAlive().maxSupportedTier);
  }

  setMaxSupportedTier(tier: number): void {
    callNapi(() => this.#ensureAlive().setMaxSupportedTier(tier));
  }

  maxAvailableTier(): number {
    return callNapi(() => this.#ensureAlive().maxAvailableTier);
  }

  effectiveTier(): number {
    return callNapi(() => this.#ensureAlive().effectiveTier);
  }

  sampleRate(): number {
    return callNapi(() => this.#ensureAlive().sampleRate);
  }

  setSampleRate(sampleRateHz: number): void {
    callNapi(() => this.#ensureAudioIdle().setSampleRate(sampleRateHz));
  }

  latencySamples(): number {
    return callNapi(() => this.#ensureAlive().latencySamples);
  }

  offlineBounce(): boolean {
    return callNapi(() => this.#ensureAlive().offlineBounce);
  }

  setOfflineBounce(enabled: boolean): void {
    callNapi(() => this.#ensureAlive().setOfflineBounce(enabled));
  }

  maximizeTierOnBounce(): boolean {
    return callNapi(() => this.#ensureAlive().maximizeTierOnBounce);
  }

  setMaximizeTierOnBounce(enabled: boolean): void {
    callNapi(() => this.#ensureAlive().setMaximizeTierOnBounce(enabled));
  }

  numControls(): number {
    return callNapi(() => this.#ensureAlive().numControls);
  }

  rawControlValues(): Float32Array {
    return callNapi(() => this.#ensureAlive().rawControlValues());
  }

  clampedControlValue(ctrlIdx: number): number {
    return callNapi(() => this.#ensureAlive().clampedControlValue(ctrlIdx));
  }

  setControlValue(ctrlIdx: number, value: number): void {
    callNapi(() => this.#ensureAlive().setControlValue(ctrlIdx, value));
  }

  setControlImmediate(ctrlIdx: number, value: number): void {
    callNapi(() => this.#ensureAlive().setControlImmediate(ctrlIdx, value));
  }

  setControlsImmediateBatch(indices: readonly number[], values: readonly number[]): void {
    callNapi(() => this.#ensureAlive().setControlsImmediateBatch([...indices], [...values]));
  }

  resetControlsToDefaults(): void {
    callNapi(() => this.#ensureAlive().resetControlsToDefaults());
  }

  reset(): void {
    callNapi(() => this.#ensureAlive().reset());
  }

  supportsFreeze(): boolean {
    return callNapi(() => this.#ensureAlive().supportsFreeze);
  }

  isFrozen(): boolean {
    return callNapi(() => this.#ensureAlive().isFrozen);
  }

  setFreeze(freeze: boolean): void {
    callNapi(() => this.#ensureAlive().setFreeze(freeze));
  }

  setFreezeAmount(alpha: number): void {
    callNapi(() => this.#ensureAlive().setFreezeAmount(alpha));
  }

  bypassState(): readonly [boolean, boolean] {
    const res = callNapi(() => this.#ensureAlive().bypassState());
    return [Boolean(res[0]), Boolean(res[1])];
  }

  setBypass(bypassed: boolean, trails: boolean): void {
    callNapi(() => this.#ensureAlive().setBypass(bypassed, trails));
  }

  stereoWidth(): number {
    return callNapi(() => this.#ensureAlive().stereoWidth);
  }

  setStereoWidth(width: number): void {
    callNapi(() => this.#ensureAlive().setStereoWidth(width));
  }

  configureHostMix(blendWet: number, outputGainDb: number, applyUnityLeveling: boolean): void {
    callNapi(() =>
      this.#ensureAlive().configureHostMix(blendWet, outputGainDb, applyUnityLeveling),
    );
  }

  currentBpm(): number {
    return callNapi(() => this.#ensureAlive().currentBpm);
  }

  syncTempo(bpm: number, subdivision: string | undefined, foldOctave: boolean): number {
    return callNapi(() => this.#ensureAlive().syncTempo(bpm, subdivision, foldOctave));
  }

  registerTapTempo(
    subdivision: string | undefined,
    foldOctave: boolean,
  ): readonly [number, number] {
    const res = callNapi(() => this.#ensureAlive().registerTapTempo(subdivision, foldOctave));
    return [Number(res[0]), Number(res[1])];
  }

  applyPreset(presetId: string): void {
    callNapi(() => this.#ensureAlive().applyPreset(presetId));
  }

  processStereoInPlace(leftIo: Float32Array, rightIo: Float32Array): void {
    callNapi(() => this.#ensureAudioIdle().processStereoInPlace(leftIo, rightIo));
  }

  processStereo(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    callNapi(() =>
      this.#ensureAudioIdle().processStereo(leftIn, rightIn, leftOut, rightOut, blockSize),
    );
  }

  processMonoToStereo(
    monoIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
    blockSize?: number,
  ): void {
    callNapi(() =>
      this.#ensureAudioIdle().processMonoToStereo(monoIn, leftOut, rightOut, blockSize),
    );
  }

  processMono(monoIn: Float32Array, monoOut: Float32Array, blockSize?: number): void {
    callNapi(() => this.#ensureAudioIdle().processMono(monoIn, monoOut, blockSize));
  }

  renderOfflineBounce(
    leftIn: Float32Array,
    rightIn: Float32Array,
    leftOut: Float32Array,
    rightOut: Float32Array,
  ): void {
    callNapi(() => this.#ensureAudioIdle().renderOfflineBounce(leftIn, rightIn, leftOut, rightOut));
  }

  processStereoAsync(
    leftIn: Float32Array,
    rightIn: Float32Array,
    blockSize?: number,
  ): Promise<readonly [Float32Array, Float32Array]> {
    const raw = this.#ensureAudioIdle();
    this.#busy = true;
    let task: Promise<Float32Array[]>;
    try {
      task = raw.processStereoAsync(leftIn, rightIn, blockSize);
    } catch (err) {
      this.#busy = false;
      throw fromNativeError(err);
    }
    return task.then(
      (out) => {
        this.#busy = false;
        const l = out[0];
        const r = out[1];
        if (!l || !r) {
          throw new DnamError("ERR_DNAM_VALIDATION", "Invalid output from processStereoAsync");
        }
        return [l, r] as const;
      },
      (err: unknown) => {
        this.#busy = false;
        throw fromNativeError(err);
      },
    );
  }

  telemetryInto(out56: Float64Array): void {
    callNapi(() => this.#ensureAlive().telemetryInto(out56));
  }

  dispose(): void {
    this.#disposed = true;
  }
}

export class NapiControlSpaceHandle implements NativeControlSpaceHandle {
  readonly #raw: RawNapiControlSpace;
  #disposed = false;

  constructor(raw: RawNapiControlSpace) {
    this.#raw = raw;
  }

  clampInto(rawNumeric: Float32Array, clampedOut: Float32Array, normOut: Float32Array): number {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return callNapi(() => this.#raw.clampInto(rawNumeric, clampedOut, normOut));
  }

  toPhysical(ctrlIdx: number, normalizedU: number): number {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return callNapi(() => this.#raw.toPhysical(ctrlIdx, normalizedU));
  }

  toNormalized(ctrlIdx: number, physicalVal: number): number {
    if (this.#disposed) {
      throw new DnamError("ERR_DNAM_DISPOSED", "ControlSpace has been disposed");
    }
    return callNapi(() => this.#raw.toNormalized(ctrlIdx, physicalVal));
  }

  dispose(): void {
    this.#disposed = true;
  }
}
