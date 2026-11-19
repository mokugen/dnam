---
trigger: always_on
---

# Universal Trainable & Data-Programmable Black-Box Engine & Solver Rules

These architectural rules are mandatory for every engine, schema, and solver edit in DNAM (`REQ-1.6`, `REQ-1.9`, `ref[arch.pillars.universal-black-box]`, `ref[repo.code.universal-black-box]`):

1. **Treat `DnamEngine` as a Fixed, Universal, Data-Programmable Black-Box Mathematical Machine (`REQ-1.9`, `ref[arch.pillars.universal-black-box]`):**
   - Model newly encountered audio hardware (including multi-band crossovers, overlapping passbands, dynamics, nonlinear amplifiers, and modulated spatial rigs) strictly through learned container weights and declarative coefficients over the engine's general mathematical basis operators (`Op-T`, `Op-M`, `Op-N`, `TPT-SVF`, Zero-DC `GRU`, `FIR` + `Kautz`, `fp16` `FDN`, `3 kHz` `ControlBus`).
   - **Never** add device-, brand-, or circuit-specific white-box sub-circuits, hardcoded device crossover frequencies, or conditional stage-bypass flags (`REQ-6.1`).

2. **Keep `dnam-capture` 100% Data-Driven (`REQ-1.6`, `REQ-1.9`, `ref[repo.code.universal-black-box]`):**
   - Never reference `dnam-fixtures`, fixture names, hardcoded target frequencies, circuit-template heuristics, or control-name heuristics inside `packages/capture/src/`.
