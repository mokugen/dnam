/**
 * Zero-allocation binary reader & React hook for the `256 B` (`64`-word) `DNPR` seqlock
 * training progress & realtime fitting snapshot published by `dnam-capture`.
 */

import { useEffect, useState } from "react";
import type { DnamModelTier } from "./types.js";

/** Magic header word (`"DNPR"`, big-endian ASCII bytes) stored at word `WORD_MAGIC` (`15`). */
export const PROGRESS_MAGIC_DNPR = 0x444e5052;

/** Cumulative word offsets mirroring `packages/capture/src/solver/progress.rs`. */
export const WORD_SEQ = 0;
export const WORD_FLAGS = WORD_SEQ + 1;
export const WORD_STEPS = WORD_FLAGS + 1;
export const WORD_STAGE_STEPS = WORD_STEPS + 1;
export const WORD_BEST_LOSS = WORD_STAGE_STEPS + 1;
export const WORD_MR_IIR = WORD_BEST_LOSS + 1;
export const WORD_VIS_META = WORD_MR_IIR + 1;
export const WORD_TIER_BESR = WORD_VIS_META + 1;
export const WORD_TIER_LTAS = WORD_TIER_BESR + 4;
export const WORD_MAGIC = WORD_TIER_LTAS + 4;
export const WORD_TARGET_WAVE_WORDS = 64 / 4;
export const WORD_TARGET_WAVE = WORD_MAGIC + 1;
export const WORD_MODEL_WAVE_WORDS = 64 / 4;
export const WORD_MODEL_WAVE = WORD_TARGET_WAVE + WORD_TARGET_WAVE_WORDS;
export const WORD_LTAS_ERR_WORDS = 32 / 4;
export const WORD_LTAS_ERR = WORD_MODEL_WAVE + WORD_MODEL_WAVE_WORDS;
export const WORD_KAUTZ_POLES_WORDS = 16 / 2;
export const WORD_KAUTZ_POLES = WORD_LTAS_ERR + WORD_LTAS_ERR_WORDS;

/** Total `u32` words in the `DNPR` wire frame (`64`). */
export const PROGRESS_WIRE_WORDS = WORD_KAUTZ_POLES + WORD_KAUTZ_POLES_WORDS;

/** Total byte size of a serialized `DNPR` wire frame (`256 B`). */
export const PROGRESS_WIRE_BYTES = PROGRESS_WIRE_WORDS * 4;

export type TrainingStatusName = "idle" | "running" | "completed" | "failed";

export type TrainingPhaseName =
  | "idle"
  | "capture"
  | "stage_a_physics_probe"
  | "stage_b_backbone"
  | "stage_4_spatial_cascade"
  | "stage_c_knobs"
  | "stage_d_audit"
  | "live_null_test"
  | "completed"
  | "failed";

export interface KautzPolePoint {
  readonly rU8: number;
  readonly fU8: number;
  readonly radius: number;
  readonly freqHz: number;
}

export interface TrainerProgressFrame {
  readonly seq: number;
  readonly status: TrainingStatusName;
  readonly phase: TrainingPhaseName;
  readonly activeTier: DnamModelTier | null;
  readonly clampedControls: number;
  readonly hasVisualFrame: boolean;
  readonly progressPermille: number;
  readonly step: number;
  readonly totalSteps: number;
  readonly stageStep: number;
  readonly stageTotalSteps: number;
  readonly bestLoss: number;
  readonly mrIirLoss: number;
  readonly tierBesrDb: readonly [number, number, number, number];
  readonly tierLtasDb: readonly [number, number, number, number];
  /** Normalized `[-1, 1]` peak-window target waveform (`64` samples). */
  readonly targetWave: readonly number[];
  /** Normalized `[-1, 1]` peak-window model waveform (`64` samples). */
  readonly modelWave: readonly number[];
  /** Signed `i8` target waveform samples (`[-127, 127]`) for lossless wire re-encoding. */
  readonly targetWaveI8: readonly number[];
  /** Signed `i8` model waveform samples (`[-127, 127]`) for lossless wire re-encoding. */
  readonly modelWaveI8: readonly number[];
  /** Signed `i8` LTAS error samples in `0.05 dB` steps (`±6.35 dB`, `32` bands). */
  readonly ltasErrorQ20: readonly number[];
  /** `32`-band `1/3`-octave spectral deviation in decibels. */
  readonly ltasErrorDb: readonly number[];
  /** `16` shared Kautz pole positions decoded from normalized quantization bytes. */
  readonly kautzPoles: readonly KautzPolePoint[];
}

export type TrainerProgressSource =
  | TrainerProgressFrame
  | Int32Array
  | Uint32Array
  | (() => TrainerProgressFrame | Uint8Array | Int32Array | Uint32Array | null | undefined)
  | null
  | undefined;

const STATUS_NAMES: readonly TrainingStatusName[] = ["idle", "running", "completed", "failed"];

const PHASE_NAMES: readonly TrainingPhaseName[] = [
  "idle",
  "capture",
  "stage_a_physics_probe",
  "stage_b_backbone",
  "stage_4_spatial_cascade",
  "stage_c_knobs",
  "stage_d_audit",
  "live_null_test",
  "completed",
  "failed",
];

const TIER_FROM_CODE: readonly (DnamModelTier | null)[] = [null, "lite", "full", "pro", "ultra"];
const SCRATCH_U32 = new ArrayBuffer(4);
const SCRATCH_VIEW = new DataView(SCRATCH_U32);

function wordAt(words: ArrayLike<number>, index: number): number {
  const value = words[index];
  return value === undefined ? 0 : value >>> 0;
}

function u32ToF32(bits: number): number {
  SCRATCH_VIEW.setUint32(0, bits >>> 0, true);
  return SCRATCH_VIEW.getFloat32(0, true);
}

function f32ToU32(value: number): number {
  SCRATCH_VIEW.setFloat32(0, value, true);
  return SCRATCH_VIEW.getUint32(0, true);
}

function toSignedI8(byteVal: number): number {
  const b = byteVal & 0xff;
  return b >= 128 ? b - 256 : b;
}

function unpackSignedWave(words: ArrayLike<number>, wordOffset: number, wordCount: number) {
  const i8Values: number[] = [];
  const normValues: number[] = [];
  for (let i = 0; i < wordCount; i += 1) {
    const word = wordAt(words, wordOffset + i);
    for (let byteIdx = 0; byteIdx < 4; byteIdx += 1) {
      const signed = toSignedI8((word >>> (byteIdx * 8)) & 0xff);
      i8Values.push(signed);
      normValues.push(Math.max(-1, Math.min(1, signed / 127)));
    }
  }
  return { i8Values, normValues };
}

function unpackKautzPoles(words: ArrayLike<number>): KautzPolePoint[] {
  const poles: KautzPolePoint[] = [];
  for (let i = 0; i < WORD_KAUTZ_POLES_WORDS; i += 1) {
    const word = wordAt(words, WORD_KAUTZ_POLES + i);
    for (let slot = 0; slot < 2; slot += 1) {
      const shift = slot * 16;
      const rU8 = (word >>> shift) & 0xff;
      const fU8 = (word >>> (shift + 8)) & 0xff;
      poles.push({
        rU8,
        fU8,
        radius: 0.5 + 0.5 * (rU8 / 255),
        freqHz: 20 * 1000 ** (fU8 / 255),
      });
    }
  }
  return poles;
}

function statusToCode(status: TrainingStatusName): number {
  return Math.max(STATUS_NAMES.indexOf(status), 0);
}

function phaseToCode(phase: TrainingPhaseName): number {
  return Math.max(PHASE_NAMES.indexOf(phase), 0);
}

function tierToCode(tier: DnamModelTier | null): number {
  if (tier === null) {
    return 0;
  }
  return Math.max(TIER_FROM_CODE.indexOf(tier), 0);
}

function clampI8(value: number): number {
  const rounded = Math.round(Number.isFinite(value) ? value : 0);
  return Math.max(-127, Math.min(127, rounded)) & 0xff;
}

function packI8ToWords(
  values: readonly number[],
  wordOffset: number,
  wordCount: number,
  outWords: Uint32Array,
): void {
  for (let i = 0; i < wordCount; i += 1) {
    const base = i * 4;
    const b0 = clampI8(values[base] ?? 0);
    const b1 = clampI8(values[base + 1] ?? 0);
    const b2 = clampI8(values[base + 2] ?? 0);
    const b3 = clampI8(values[base + 3] ?? 0);
    outWords[wordOffset + i] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) >>> 0;
  }
}

/** Parses a `64`-word `DNPR` seqlock array, returning `null` if torn (`seq & 1`) or bad magic. */
export function parseTrainerProgressWords(words: ArrayLike<number>): TrainerProgressFrame | null {
  if (words.length < PROGRESS_WIRE_WORDS) {
    return null;
  }
  const seq = wordAt(words, WORD_SEQ);
  if ((seq & 1) !== 0 || wordAt(words, WORD_MAGIC) !== PROGRESS_MAGIC_DNPR) {
    return null;
  }
  const flags = wordAt(words, WORD_FLAGS);
  const status = STATUS_NAMES[flags & 0xff] ?? "idle";
  const phase = PHASE_NAMES[(flags >>> 8) & 0xff] ?? "idle";
  const activeTier = TIER_FROM_CODE[(flags >>> 16) & 0x0f] ?? null;
  const progressPermille = Math.min(1000, (flags >>> 20) & 0x0fff);

  const stepsWord = wordAt(words, WORD_STEPS);
  const stageWord = wordAt(words, WORD_STAGE_STEPS);
  const metaWord = wordAt(words, WORD_VIS_META);

  const tierBesrDb: [number, number, number, number] = [
    u32ToF32(wordAt(words, WORD_TIER_BESR)),
    u32ToF32(wordAt(words, WORD_TIER_BESR + 1)),
    u32ToF32(wordAt(words, WORD_TIER_BESR + 2)),
    u32ToF32(wordAt(words, WORD_TIER_BESR + 3)),
  ];
  const tierLtasDb: [number, number, number, number] = [
    u32ToF32(wordAt(words, WORD_TIER_LTAS)),
    u32ToF32(wordAt(words, WORD_TIER_LTAS + 1)),
    u32ToF32(wordAt(words, WORD_TIER_LTAS + 2)),
    u32ToF32(wordAt(words, WORD_TIER_LTAS + 3)),
  ];

  const tgtWave = unpackSignedWave(words, WORD_TARGET_WAVE, WORD_TARGET_WAVE_WORDS);
  const mdlWave = unpackSignedWave(words, WORD_MODEL_WAVE, WORD_MODEL_WAVE_WORDS);
  const ltasRaw = unpackSignedWave(words, WORD_LTAS_ERR, WORD_LTAS_ERR_WORDS);

  return {
    seq,
    status,
    phase,
    activeTier,
    clampedControls: metaWord & 0xffff,
    hasVisualFrame: ((metaWord >>> 16) & 1) !== 0,
    progressPermille,
    step: stepsWord & 0xffff,
    totalSteps: (stepsWord >>> 16) & 0xffff,
    stageStep: stageWord & 0xffff,
    stageTotalSteps: (stageWord >>> 16) & 0xffff,
    bestLoss: u32ToF32(wordAt(words, WORD_BEST_LOSS)),
    mrIirLoss: u32ToF32(wordAt(words, WORD_MR_IIR)),
    tierBesrDb,
    tierLtasDb,
    targetWave: tgtWave.normValues,
    modelWave: mdlWave.normValues,
    targetWaveI8: tgtWave.i8Values,
    modelWaveI8: mdlWave.i8Values,
    ltasErrorQ20: ltasRaw.i8Values,
    ltasErrorDb: ltasRaw.i8Values.map((q) => q * 0.05),
    kautzPoles: unpackKautzPoles(words),
  };
}

/** Parses a little-endian `256 B` byte slice into a validated [`TrainerProgressFrame`]. */
export function parseTrainerProgressBytes(bytes: Uint8Array): TrainerProgressFrame | null {
  if (bytes.byteLength < PROGRESS_WIRE_BYTES) {
    return null;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, PROGRESS_WIRE_BYTES);
  const words = new Uint32Array(PROGRESS_WIRE_WORDS);
  for (let i = 0; i < PROGRESS_WIRE_WORDS; i += 1) {
    words[i] = view.getUint32(i * 4, true);
  }
  return parseTrainerProgressWords(words);
}

/** Encodes a [`TrainerProgressFrame`] back into its canonical little-endian `256 B` `DNPR` packet. */
export function encodeTrainerProgressBytes(frame: TrainerProgressFrame): Uint8Array {
  const words = new Uint32Array(PROGRESS_WIRE_WORDS);
  const permille = Math.max(0, Math.min(1000, Math.round(frame.progressPermille)));
  words[WORD_SEQ] = frame.seq >>> 0;
  words[WORD_FLAGS] =
    (statusToCode(frame.status) |
      (phaseToCode(frame.phase) << 8) |
      (tierToCode(frame.activeTier) << 16) |
      (permille << 20)) >>>
    0;
  words[WORD_STEPS] = ((frame.step & 0xffff) | ((frame.totalSteps & 0xffff) << 16)) >>> 0;
  words[WORD_STAGE_STEPS] =
    ((frame.stageStep & 0xffff) | ((frame.stageTotalSteps & 0xffff) << 16)) >>> 0;
  words[WORD_BEST_LOSS] = f32ToU32(frame.bestLoss);
  words[WORD_MR_IIR] = f32ToU32(frame.mrIirLoss);
  words[WORD_VIS_META] =
    ((frame.clampedControls & 0xffff) | ((frame.hasVisualFrame ? 1 : 0) << 16)) >>> 0;
  for (let t = 0; t < 4; t += 1) {
    words[WORD_TIER_BESR + t] = f32ToU32(frame.tierBesrDb[t] ?? Number.NaN);
    words[WORD_TIER_LTAS + t] = f32ToU32(frame.tierLtasDb[t] ?? Number.NaN);
  }
  words[WORD_MAGIC] = PROGRESS_MAGIC_DNPR;

  const tgtI8 =
    frame.targetWaveI8.length >= 64
      ? frame.targetWaveI8
      : frame.targetWave.map((v) => Math.round(v * 127));
  const mdlI8 =
    frame.modelWaveI8.length >= 64
      ? frame.modelWaveI8
      : frame.modelWave.map((v) => Math.round(v * 127));
  const ltasI8 =
    frame.ltasErrorQ20.length >= 32
      ? frame.ltasErrorQ20
      : frame.ltasErrorDb.map((db) => Math.round(db * 20));
  packI8ToWords(tgtI8, WORD_TARGET_WAVE, WORD_TARGET_WAVE_WORDS, words);
  packI8ToWords(mdlI8, WORD_MODEL_WAVE, WORD_MODEL_WAVE_WORDS, words);
  packI8ToWords(ltasI8, WORD_LTAS_ERR, WORD_LTAS_ERR_WORDS, words);

  for (let i = 0; i < WORD_KAUTZ_POLES_WORDS; i += 1) {
    const p0 = frame.kautzPoles[2 * i];
    const p1 = frame.kautzPoles[2 * i + 1];
    const r0 = (p0?.rU8 ?? 0) & 0xff;
    const f0 = (p0?.fU8 ?? 0) & 0xff;
    const r1 = (p1?.rU8 ?? 0) & 0xff;
    const f1 = (p1?.fU8 ?? 0) & 0xff;
    words[WORD_KAUTZ_POLES + i] = (r0 | (f0 << 8) | (r1 << 16) | (f1 << 24)) >>> 0;
  }

  const out = new Uint8Array(PROGRESS_WIRE_BYTES);
  const outView = new DataView(out.buffer);
  for (let i = 0; i < PROGRESS_WIRE_WORDS; i += 1) {
    outView.setUint32(i * 4, words[i] ?? 0, true);
  }
  return out;
}

/** Performs a tear-free seqlock read via `Atomics.load`, returning `null` if odd/torn. */
export function readTrainerProgressShared(
  sharedWords: Int32Array | Uint32Array,
): TrainerProgressFrame | null {
  if (sharedWords.length < PROGRESS_WIRE_WORDS) {
    return null;
  }
  const seqBefore = Atomics.load(sharedWords, WORD_SEQ) >>> 0;
  if ((seqBefore & 1) !== 0) {
    return null;
  }
  const snapshot = new Uint32Array(PROGRESS_WIRE_WORDS);
  for (let i = 0; i < PROGRESS_WIRE_WORDS; i += 1) {
    snapshot[i] = Atomics.load(sharedWords, i) >>> 0;
  }
  const seqAfter = Atomics.load(sharedWords, WORD_SEQ) >>> 0;
  if (seqAfter !== seqBefore) {
    return null;
  }
  return parseTrainerProgressWords(snapshot);
}

/** Returns user-friendly `[0..100]` Tone Match % matching `TrainerProgressSnapshot::tone_match_percent()`. */
export function computeToneMatchPercent(frame: TrainerProgressFrame): number | null {
  const tierIdx = frame.activeTier === null ? -1 : TIER_FROM_CODE.indexOf(frame.activeTier) - 1;
  let activeEsr = tierIdx >= 0 ? frame.tierBesrDb[tierIdx] : undefined;
  if (activeEsr === undefined || !Number.isFinite(activeEsr)) {
    activeEsr = [...frame.tierBesrDb].reverse().find((db) => Number.isFinite(db));
  }
  if (activeEsr === undefined || !Number.isFinite(activeEsr)) {
    if (Number.isFinite(frame.bestLoss) && frame.bestLoss >= 0) {
      activeEsr = 10 * Math.log10(Math.max(frame.bestLoss, 1e-12));
    }
  }
  if (activeEsr === undefined || !Number.isFinite(activeEsr)) {
    return null;
  }
  const rawMatch = (1 - 10 ** (activeEsr / 20)) * 100;
  return Math.max(0, Math.min(100, rawMatch));
}

function sampleTrainerProgressSource(source: TrainerProgressSource): TrainerProgressFrame | null {
  if (source === null || source === undefined) {
    return null;
  }
  if (source instanceof Int32Array || source instanceof Uint32Array) {
    return readTrainerProgressShared(source);
  }
  if (typeof source === "function") {
    const raw = source();
    if (raw === null || raw === undefined) {
      return null;
    }
    if (raw instanceof Uint8Array) {
      return parseTrainerProgressBytes(raw);
    }
    if (raw instanceof Int32Array || raw instanceof Uint32Array) {
      return readTrainerProgressShared(raw);
    }
    return raw;
  }
  return source;
}

/** React hook polling a seqlock buffer, byte snapshot getter, or direct frame at `pollIntervalMs`. */
export function useTrainerProgress(
  source: TrainerProgressSource,
  pollIntervalMs = 50,
): TrainerProgressFrame | null {
  const [frame, setFrame] = useState<TrainerProgressFrame | null>(() =>
    sampleTrainerProgressSource(source),
  );

  useEffect(() => {
    const first = sampleTrainerProgressSource(source);
    if (first !== null) {
      setFrame(first);
    }
    if (source === null || source === undefined) {
      setFrame(null);
      return undefined;
    }
    if (
      typeof source !== "function" &&
      !(source instanceof Int32Array) &&
      !(source instanceof Uint32Array)
    ) {
      return undefined;
    }
    const id = globalThis.setInterval(
      () => {
        const next = sampleTrainerProgressSource(source);
        if (next !== null) {
          setFrame(next);
        }
      },
      Math.max(16, pollIntervalMs),
    );
    return () => {
      globalThis.clearInterval(id);
    };
  }, [pollIntervalMs, source]);

  return frame;
}
