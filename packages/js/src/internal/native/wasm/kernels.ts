/**
 * WebAssembly reference TORA DSP kernel delegates (`ref[sdk.ts.runtime.dual-target-drivers]`).
 * Zero `node:*` dependencies.
 *
 * @module
 */

import type { WasmMemoryArena } from "./arena.js";

export function wasmCatmullRomHermiteSlice(
  mem: WasmMemoryArena,
  sM1: Float64Array,
  s0: Float64Array,
  s1: Float64Array,
  s2: Float64Array,
  mu: Float64Array,
  out: Float64Array,
): void {
  const n = s0.length;
  mem.withF64Slice(sM1, (pM1) =>
    mem.withF64Slice(s0, (p0) =>
      mem.withF64Slice(s1, (p1) =>
        mem.withF64Slice(s2, (p2) =>
          mem.withF64Slice(mu, (pMu) => {
            const { out: res } = mem.withF64Out(n, (pOut) =>
              mem.checkStatus(
                mem.exports.dnam_wasm_catmull_rom_hermite_slice(pM1, p0, p1, p2, pMu, pOut, n),
              ),
            );
            out.set(res);
          }),
        ),
      ),
    ),
  );
}

export function wasmOpTCoerciveStep(
  mem: WasmMemoryArena,
  uIn: number,
  gLin: number,
  sigmaKnee: number,
  vCeil: number,
  rDc: number,
  dcXPrev: number,
  dcYPrev: number,
): readonly [number, number, number] {
  mem.exports.dnam_wasm_op_t_coercive_step(
    uIn,
    gLin,
    sigmaKnee,
    vCeil,
    rDc,
    dcXPrev,
    dcYPrev,
    mem.smallF64Ptr,
  );
  const start = mem.smallF64Ptr >>> 3;
  const f64 = mem.f64();
  return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])];
}

export function wasmOpTTankSlice(
  mem: WasmMemoryArena,
  inp: Float64Array,
  out: Float64Array,
  gLin: number,
  sigmaKnee: number,
  vCeil: number,
  rDc: number,
): void {
  mem.withF64Slice(inp, (iPtr, n) => {
    const { out: res } = mem.withF64Out(n, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_op_t_tank_slice(iPtr, oPtr, n, gLin, sigmaKnee, vCeil, rDc),
      ),
    );
    out.set(res);
  });
}

export function wasmOpMModalStep(
  mem: WasmMemoryArena,
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
  mem.exports.dnam_wasm_op_m_modal_step(
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
    mem.smallF64Ptr,
  );
  const start = mem.smallF64Ptr >>> 3;
  const f64 = mem.f64();
  return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])];
}

export function wasmOpMModalSlice(
  mem: WasmMemoryArena,
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
  return mem.withF64Slice(inp, (iPtr, n) => {
    const { out: res } = mem.withF64Out(n, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_op_m_modal_slice(
          iPtr,
          oPtr,
          n,
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
          reset ? 1 : 0,
          mem.smallF64Ptr,
        ),
      ),
    );
    out.set(res);
    const start = mem.smallF64Ptr >>> 3;
    const f64 = mem.f64();
    return [Number(f64[start]), Number(f64[start + 1])] as const;
  });
}

export function wasmUnitRotor(mem: WasmMemoryArena, phi: number): readonly [number, number] {
  mem.exports.dnam_wasm_unit_rotor(phi, mem.smallF64Ptr);
  const start = mem.smallF64Ptr >>> 3;
  const f64 = mem.f64();
  return [Number(f64[start]), Number(f64[start + 1])];
}

export function wasmUnitRotorSlice(
  mem: WasmMemoryArena,
  phi: Float64Array,
  out: Float64Array,
): void {
  mem.withF64Slice(phi, (pPtr, n) => {
    const { out: res } = mem.withF64Out(out.length, (oPtr) =>
      mem.checkStatus(mem.exports.dnam_wasm_unit_rotor_slice(pPtr, n, oPtr, out.length)),
    );
    out.set(res);
  });
}

export function wasmPolyAdaaStep(
  mem: WasmMemoryArena,
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
  mem.exports.dnam_wasm_poly_adaa_step(
    x,
    bConv,
    alpha,
    beta,
    delta,
    rAc,
    uPrev,
    aPrev,
    mDc,
    mem.smallF64Ptr,
  );
  const start = mem.smallF64Ptr >>> 3;
  const f64 = mem.f64();
  return [
    Number(f64[start]),
    Number(f64[start + 1]),
    Number(f64[start + 2]),
    Number(f64[start + 3]),
  ];
}

export function wasmPolyAdaaSlice(
  mem: WasmMemoryArena,
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
  return mem.withF64Slice(inp, (iPtr, n) => {
    const { out: res } = mem.withF64Out(n, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_poly_adaa_slice(
          iPtr,
          oPtr,
          n,
          bConv,
          alpha,
          beta,
          delta,
          rAc,
          uPrev,
          aPrev,
          mDc,
          reset ? 1 : 0,
          mem.smallF64Ptr,
        ),
      ),
    );
    out.set(res);
    const start = mem.smallF64Ptr >>> 3;
    const f64 = mem.f64();
    return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])] as const;
  });
}

export function wasmEvalLobattoBasis(
  mem: WasmMemoryArena,
  u: number,
  anchors3: Float64Array,
  degree: number,
): readonly [number, number, number] {
  return mem.withF64Slice(anchors3, (aPtr, aLen) => {
    mem.checkStatus(
      mem.exports.dnam_wasm_eval_lobatto_basis(u, aPtr, aLen, degree, mem.smallF64Ptr),
    );
    const start = mem.smallF64Ptr >>> 3;
    const f64 = mem.f64();
    return [Number(f64[start]), Number(f64[start + 1]), Number(f64[start + 2])] as const;
  });
}

export function wasmEvalLobattoBasisSlice(
  mem: WasmMemoryArena,
  u: Float64Array,
  out: Float64Array,
  anchors3: Float64Array,
  degree: number,
): void {
  mem.withF64Slice(u, (uPtr, uLen) =>
    mem.withF64Slice(anchors3, (aPtr, aLen) => {
      const { out: res } = mem.withF64Out(out.length, (oPtr) =>
        mem.checkStatus(
          mem.exports.dnam_wasm_eval_lobatto_basis_slice(
            uPtr,
            uLen,
            oPtr,
            out.length,
            aPtr,
            aLen,
            degree,
          ),
        ),
      );
      out.set(res);
    }),
  );
}

export function wasmEvalRank4Fm(
  mem: WasmMemoryArena,
  features: Float64Array,
  vFactorsFlat: Float64Array,
): readonly [number, number, number, number] {
  return mem.withF64Slice(features, (fPtr, fLen) =>
    mem.withF64Slice(vFactorsFlat, (vPtr, vLen) => {
      mem.checkStatus(mem.exports.dnam_wasm_eval_rank4_fm(fPtr, fLen, vPtr, vLen, mem.smallF64Ptr));
      const start = mem.smallF64Ptr >>> 3;
      const f64 = mem.f64();
      return [
        Number(f64[start]),
        Number(f64[start + 1]),
        Number(f64[start + 2]),
        Number(f64[start + 3]),
      ] as const;
    }),
  );
}

export function wasmEvalRank4FmSlice(
  mem: WasmMemoryArena,
  featuresFlat: Float64Array,
  nFrames: number,
  dDim: number,
  vFactorsFlat: Float64Array,
  out: Float64Array,
): void {
  mem.withF64Slice(featuresFlat, (fPtr, fLen) =>
    mem.withF64Slice(vFactorsFlat, (vPtr, vLen) => {
      const { out: res } = mem.withF64Out(out.length, (oPtr) =>
        mem.checkStatus(
          mem.exports.dnam_wasm_eval_rank4_fm_slice(
            fPtr,
            fLen,
            nFrames,
            dDim,
            vPtr,
            vLen,
            oPtr,
            out.length,
          ),
        ),
      );
      out.set(res);
    }),
  );
}

export function wasmEvalChebyshevLfo(
  mem: WasmMemoryArena,
  cosPhi: number,
): readonly [number, number, number, number] {
  mem.exports.dnam_wasm_eval_chebyshev_lfo(cosPhi, mem.smallF64Ptr);
  const start = mem.smallF64Ptr >>> 3;
  const f64 = mem.f64();
  return [
    Number(f64[start]),
    Number(f64[start + 1]),
    Number(f64[start + 2]),
    Number(f64[start + 3]),
  ];
}

export function wasmEvalChebyshevLfoSlice(
  mem: WasmMemoryArena,
  cosPhi: Float64Array,
  out: Float64Array,
): void {
  mem.withF64Slice(cosPhi, (cPtr, cLen) => {
    const { out: res } = mem.withF64Out(out.length, (oPtr) =>
      mem.checkStatus(mem.exports.dnam_wasm_eval_chebyshev_lfo_slice(cPtr, cLen, oPtr, out.length)),
    );
    out.set(res);
  });
}

export function wasmEvalTrustRegionSoftclipSlice(
  mem: WasmMemoryArena,
  z: Float64Array,
  out: Float64Array,
  mh: number,
): void {
  mem.withF64Slice(z, (zPtr, n) => {
    const { out: res } = mem.withF64Out(n, (oPtr) =>
      mem.checkStatus(mem.exports.dnam_wasm_eval_trust_region_softclip_slice(zPtr, oPtr, n, mh)),
    );
    out.set(res);
  });
}

export function wasmCubicLagrangeWeights(
  mem: WasmMemoryArena,
  mu: number,
): readonly [number, number, number, number] {
  mem.exports.dnam_wasm_cubic_lagrange_weights(mu, mem.smallF64Ptr);
  const start = mem.smallF64Ptr >>> 3;
  const f64 = mem.f64();
  return [
    Number(f64[start]),
    Number(f64[start + 1]),
    Number(f64[start + 2]),
    Number(f64[start + 3]),
  ];
}

export function wasmCubicLagrangeWeightsSlice(
  mem: WasmMemoryArena,
  mu: Float64Array,
  out: Float64Array,
): void {
  mem.withF64Slice(mu, (mPtr, n) => {
    const { out: res } = mem.withF64Out(out.length, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_cubic_lagrange_weights_slice(mPtr, n, oPtr, out.length),
      ),
    );
    out.set(res);
  });
}

export function wasmGivensHouseholder8ch(
  mem: WasmMemoryArena,
  x8: Float64Array,
  cosPsi: number,
  sinPsi: number,
): Float64Array {
  return mem.withF64Slice(x8, (xPtr, xLen) => {
    const { out } = mem.withF64Out(8, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_givens_householder_8ch(xPtr, xLen, cosPsi, sinPsi, oPtr),
      ),
    );
    return out;
  });
}

export function wasmGivensHouseholder8chSlice(
  mem: WasmMemoryArena,
  xFlat: Float64Array,
  out: Float64Array,
  cosPsi: number,
  sinPsi: number,
): void {
  mem.withF64Slice(xFlat, (xPtr, n) => {
    const { out: res } = mem.withF64Out(n, (oPtr) =>
      mem.checkStatus(
        mem.exports.dnam_wasm_givens_householder_8ch_slice(xPtr, oPtr, n, cosPsi, sinPsi),
      ),
    );
    out.set(res);
  });
}

export function wasmGivensHouseholder16ch(
  mem: WasmMemoryArena,
  x16: Float64Array,
  cosPsi: number,
  sinPsi: number,
  crossCos8: Float64Array,
  crossSin8: Float64Array,
): Float64Array {
  return mem.withF64Slice(x16, (xPtr, xLen) =>
    mem.withF64Slice(crossCos8, (ccPtr, ccLen) =>
      mem.withF64Slice(crossSin8, (csPtr, csLen) => {
        const { out } = mem.withF64Out(16, (oPtr) =>
          mem.checkStatus(
            mem.exports.dnam_wasm_givens_householder_16ch(
              xPtr,
              xLen,
              cosPsi,
              sinPsi,
              ccPtr,
              ccLen,
              csPtr,
              csLen,
              oPtr,
            ),
          ),
        );
        return out;
      }),
    ),
  );
}

export function wasmGivensHouseholder16chSlice(
  mem: WasmMemoryArena,
  xFlat: Float64Array,
  out: Float64Array,
  cosPsi: number,
  sinPsi: number,
  crossCos8: Float64Array,
  crossSin8: Float64Array,
): void {
  mem.withF64Slice(xFlat, (xPtr, n) =>
    mem.withF64Slice(crossCos8, (ccPtr, ccLen) =>
      mem.withF64Slice(crossSin8, (csPtr, csLen) => {
        const { out: res } = mem.withF64Out(n, (oPtr) =>
          mem.checkStatus(
            mem.exports.dnam_wasm_givens_householder_16ch_slice(
              xPtr,
              oPtr,
              n,
              cosPsi,
              sinPsi,
              ccPtr,
              ccLen,
              csPtr,
              csLen,
            ),
          ),
        );
        out.set(res);
      }),
    ),
  );
}
