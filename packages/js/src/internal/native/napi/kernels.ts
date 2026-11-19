/**
 * `N-API v10` reference TORA DSP kernel delegates (`ref[sdk.ts.runtime.dual-target-drivers]`).
 *
 * @module
 */

import type { RawNapiAddon } from "./bindings.js";
import { callNapi } from "./handles.js";

export function napiOpTCoerciveStep(
  addon: RawNapiAddon,
  uIn: number,
  gLin: number,
  sigmaKnee: number,
  vCeil: number,
  rDc: number,
  dcXPrev: number,
  dcYPrev: number,
): readonly [number, number, number] {
  const res = callNapi(() =>
    addon.opTCoerciveStepNative(uIn, gLin, sigmaKnee, vCeil, rDc, dcXPrev, dcYPrev),
  );
  return [Number(res[0]), Number(res[1]), Number(res[2])];
}

export function napiOpMModalStep(
  addon: RawNapiAddon,
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
): readonly [number, number, number] {
  const res = callNapi(() =>
    addon.opMModalStepNative(
      x,
      rLow,
      drSpan,
      wSel,
      bGammaNorm,
      cReal,
      cImag,
      uRotCos,
      uRotSin,
      zRe,
      zIm,
    ),
  );
  return [Number(res[0]), Number(res[1]), Number(res[2])];
}

export function napiOpMModalSlice(
  addon: RawNapiAddon,
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
): readonly [number, number] {
  const res = callNapi(() =>
    addon.opMModalSliceNative(
      inp,
      out,
      rLow,
      drSpan,
      wSel,
      bGammaNorm,
      cReal,
      cImag,
      uRotCos,
      uRotSin,
      zRe,
      zIm,
      reset,
    ),
  );
  return [Number(res[0]), Number(res[1])];
}

export function napiPolyAdaaStep(
  addon: RawNapiAddon,
  x: number,
  bConv: number,
  alpha: number,
  beta: number,
  delta: number,
  rAc: number,
  uPrev: number,
  aPrev: number,
  mDc: number,
): readonly [number, number, number, number] {
  const res = callNapi(() =>
    addon.polyAdaaStepNative(x, bConv, alpha, beta, delta, rAc, uPrev, aPrev, mDc),
  );
  return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
}

export function napiPolyAdaaSlice(
  addon: RawNapiAddon,
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
): readonly [number, number, number] {
  const res = callNapi(() =>
    addon.polyAdaaSliceNative(inp, out, bConv, alpha, beta, delta, rAc, uPrev, aPrev, mDc, reset),
  );
  return [Number(res[0]), Number(res[1]), Number(res[2])];
}

export function napiEvalLobattoBasis(
  addon: RawNapiAddon,
  u: number,
  anchors3: Float64Array,
  degree: number,
): readonly [number, number, number] {
  const res = callNapi(() => addon.evalLobattoBasisNative(u, anchors3, degree));
  return [Number(res[0]), Number(res[1]), Number(res[2])];
}

export function napiEvalRank4Fm(
  addon: RawNapiAddon,
  features: Float64Array,
  vFactorsFlat: Float64Array,
): readonly [number, number, number, number] {
  const res = callNapi(() => addon.evalRank4FmNative(features, vFactorsFlat));
  return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
}

export function napiEvalChebyshevLfo(
  addon: RawNapiAddon,
  cosPhi: number,
): readonly [number, number, number, number] {
  const res = callNapi(() => addon.evalChebyshevLfoNative(cosPhi));
  return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
}

export function napiCubicLagrangeWeights(
  addon: RawNapiAddon,
  mu: number,
): readonly [number, number, number, number] {
  const res = callNapi(() => addon.cubicLagrangeWeightsNative(mu));
  return [Number(res[0]), Number(res[1]), Number(res[2]), Number(res[3])];
}
