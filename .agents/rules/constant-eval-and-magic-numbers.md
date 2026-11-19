---
trigger: always_on
---

# Compile-Time Constant Evaluation, Derived Layouts & Documented Derivations

These architectural rules are mandatory for every code and specification edit in DNAM (`ref[repo.code.constant-eval-and-derivations]`):

1. **Derive Numerical Constants & Structural Layouts via Compile-Time Evaluation:**
   - **Never hand-enter unexplained magic numbers or manually summed byte offsets.** Derive filter poles/gains, rational polynomial coefficients, IEEE-754 bit boundaries, normalized window/table weights, buffer strides, arena offsets, and packet layouts via pure compile-time expressions (`const fn` in `dnam_engine::math::const_eval` or module-local `const fn`, cumulative `const` arithmetic such as `PREV_OFFSET + PREV_BYTES`, and TypeScript symbolic constants imported or derived from canonical primitives).
   - **Enforce Algebraic & Layout Invariants at Compile Time:** Pair derived Rust constants, wire offsets, and table sizes with `const _: () = assert!(...)` assertions (verifying alignment, prefix monotonicity, IEEE-754 bounds, partition-of-unity sums, and total struct/arena footprints at compile time with zero runtime overhead).

2. **Document Complex or Expensive Derivations in Canonical Specifications:**
   - **When `const fn` Is Infeasible or Hurts Compile Time:** When a constant originates from offline Chebyshev/Remez minimax optimization, global multi-binade floating-point analysis, spectral factorization, or empirical acoustic calibration whose solver cannot run deterministically or cheaply inside `const` evaluation, **do not duplicate raw literals anonymously across call sites**:
     1. Define the constant once in its canonical module with a concise doc comment stating the symbolic formula/origin and citing its specification rule (`ref[...]`).
     2. Document the complete mathematical derivation, boundary conditions, rounding direction, and error/precision bounds inside the canonical `docs/specs/` file.
     3. Verify the constant's defining mathematical properties via `const` assertions or co-located `#[cfg(test)]` / Vitest unit tests.
