/**
 * Buffer validation, scratch allocation, and mono downmix helpers for `Engine`.
 *
 * @module
 */

import { DnamError } from "../errors.js";

/**
 * Validates a caller-supplied output buffer array for channel count and frame length.
 */
export function validateOutputBuffers(
  out: Float32Array[],
  outChannelsCount: number,
  frames: number,
  label: string,
): Float32Array[] {
  if (out.length !== outChannelsCount) {
    throw new DnamError(
      "ERR_DNAM_VALIDATION",
      `${label} must have ${String(outChannelsCount)} channel(s), got ${String(out.length)}`,
    );
  }
  for (let c = 0; c < outChannelsCount; c++) {
    if (out[c]?.length !== frames) {
      throw new DnamError(
        "ERR_DNAM_VALIDATION",
        `${label}[${String(c)}] length (${String(out[c]?.length)}) must match input frames (${String(frames)})`,
      );
    }
  }
  return out;
}

/**
 * Validates `supplied` output buffers or allocates fresh channel arrays.
 */
export function resolveOutputBuffers(
  supplied: Float32Array[] | undefined,
  outChannelsCount: number,
  frames: number,
): Float32Array[] {
  if (supplied !== undefined) {
    return validateOutputBuffers(supplied, outChannelsCount, frames, "options.output");
  }
  return outChannelsCount === 2
    ? [new Float32Array(frames), new Float32Array(frames)]
    : [new Float32Array(frames)];
}

/**
 * Averages stereo channels into a mono buffer (`2 -> 1` topology).
 */
export function averageStereoToMono(
  left: Float32Array,
  right: Float32Array,
  monoOut: Float32Array,
  frames: number,
): void {
  for (let i = 0; i < frames; i++) {
    monoOut[i] = 0.5 * ((left[i] ?? 0) + (right[i] ?? 0));
  }
}

/**
 * Finalizes asynchronous stereo engine output for `1->1`, `2->1`, `1->2`, and `2->2` topologies.
 */
export function finalizeAsyncOutput(
  leftOut: Float32Array,
  rightOut: Float32Array,
  inChannels: number,
  outChannelsCount: number,
  frames: number,
  suppliedOut: Float32Array[] | undefined,
): Float32Array[] {
  if (outChannelsCount === 1) {
    const monoOut = suppliedOut !== undefined ? (suppliedOut[0] as Float32Array) : leftOut;
    if (inChannels === 2) {
      averageStereoToMono(leftOut, rightOut, monoOut, frames);
    } else if (suppliedOut !== undefined) {
      monoOut.set(leftOut);
    }
    return suppliedOut ?? [monoOut];
  }
  if (suppliedOut !== undefined) {
    suppliedOut[0]?.set(leftOut);
    suppliedOut[1]?.set(rightOut);
    return suppliedOut;
  }
  return [leftOut, rightOut];
}
