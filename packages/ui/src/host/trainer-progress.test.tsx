import { useEffect, useState } from "react";
import { describe, expect, it } from "vitest";
import { page } from "vitest/browser";
import { render } from "vitest-browser-react";
import {
  computeToneMatchPercent,
  encodeTrainerProgressBytes,
  PROGRESS_MAGIC_DNPR,
  PROGRESS_WIRE_BYTES,
  PROGRESS_WIRE_WORDS,
  parseTrainerProgressBytes,
  parseTrainerProgressWords,
  readTrainerProgressShared,
  type TrainerProgressFrame,
  type TrainerProgressSource,
  useTrainerProgress,
  WORD_MAGIC,
  WORD_SEQ,
} from "./trainer-progress.js";

function makeSampleFrame(overrides?: Partial<TrainerProgressFrame>): TrainerProgressFrame {
  const targetWaveI8 = Array.from({ length: 64 }, (_, i) => Math.round(90 * Math.sin(i * 0.25)));
  const modelWaveI8 = Array.from({ length: 64 }, (_, i) => Math.round(86 * Math.sin(i * 0.25)));
  const ltasErrorQ20 = Array.from({ length: 32 }, (_, i) => (i % 7) - 3);
  const kautzPoles = Array.from({ length: 16 }, (_, i) => {
    const rU8 = 120 + i * 7;
    const fU8 = 30 + i * 12;
    return {
      rU8,
      fU8,
      radius: 0.5 + 0.5 * (rU8 / 255),
      freqHz: 20 * 1000 ** (fU8 / 255),
    };
  });
  return {
    seq: 42,
    status: "running",
    phase: "stage_b_backbone",
    activeTier: "pro",
    clampedControls: 3,
    hasVisualFrame: true,
    progressPermille: 640,
    step: 24,
    totalSteps: 40,
    stageStep: 4,
    stageTotalSteps: 6,
    bestLoss: 0.0025,
    mrIirLoss: 0.018,
    tierBesrDb: [-18.5, -22.0, -26.4, Number.NaN],
    tierLtasDb: [0.65, 0.35, 0.14, Number.NaN],
    targetWave: targetWaveI8.map((v) => v / 127),
    modelWave: modelWaveI8.map((v) => v / 127),
    targetWaveI8,
    modelWaveI8,
    ltasErrorQ20,
    ltasErrorDb: ltasErrorQ20.map((q) => q * 0.05),
    kautzPoles,
    ...overrides,
  };
}

function ProgressHookProbe({ source }: { readonly source: TrainerProgressSource }) {
  const frame = useTrainerProgress(source, 20);
  return (
    <div>
      <span data-testid="hook-seq">{frame ? frame.seq : "none"}</span>
      <span data-testid="hook-permille">{frame ? frame.progressPermille : 0}</span>
    </div>
  );
}

function DynamicProgressHarness({
  shared,
  sab,
  nextBytes,
}: {
  readonly shared: Int32Array;
  readonly sab: ArrayBuffer;
  readonly nextBytes: Uint8Array;
}) {
  const [source, setSource] = useState<TrainerProgressSource>(shared);
  useEffect(() => {
    const t = globalThis.setTimeout(() => {
      new Uint8Array(sab).set(nextBytes);
      setSource(() => () => new Uint8Array(sab));
    }, 30);
    return () => {
      globalThis.clearTimeout(t);
    };
  }, [nextBytes, sab]);
  return <ProgressHookProbe source={source} />;
}

describe("256 B DNPR trainer progress reader & seqlock", () => {
  it("verifies compile-time wire invariants and roundtrips frames accurately", () => {
    expect(WORD_MAGIC).toBe(15);
    expect(PROGRESS_WIRE_WORDS).toBe(64);
    expect(PROGRESS_WIRE_BYTES).toBe(256);

    const sample = makeSampleFrame();
    const bytes = encodeTrainerProgressBytes(sample);
    expect(bytes.byteLength).toBe(256);

    const parsed = parseTrainerProgressBytes(bytes);
    expect(parsed).not.toBeNull();
    expect(parsed?.seq).toBe(42);
    expect(parsed?.status).toBe("running");
    expect(parsed?.phase).toBe("stage_b_backbone");
    expect(parsed?.activeTier).toBe("pro");
    expect(parsed?.progressPermille).toBe(640);
    expect(parsed?.step).toBe(24);
    expect(parsed?.totalSteps).toBe(40);
    expect(parsed?.targetWaveI8).toEqual(sample.targetWaveI8);
    expect(parsed?.modelWaveI8).toEqual(sample.modelWaveI8);
    expect(parsed?.ltasErrorQ20).toEqual(sample.ltasErrorQ20);
    expect(parsed?.kautzPoles[0]?.rU8).toBe(sample.kautzPoles[0]?.rU8);
    expect(parsed?.kautzPoles[15]?.fU8).toBe(sample.kautzPoles[15]?.fU8);
    expect(parsed?.tierBesrDb[2]).toBeCloseTo(-26.4, 3);
    expect(Number.isNaN(parsed?.tierBesrDb[3] ?? 0)).toBe(true);
  });

  it("rejects odd seqlock counts, wrong magic words, short buffers, and torn reads", () => {
    const bytes = encodeTrainerProgressBytes(makeSampleFrame({ seq: 8 }));
    const oddBytes = bytes.slice();
    oddBytes[0] = 9;
    expect(parseTrainerProgressBytes(oddBytes)).toBeNull();
    expect(parseTrainerProgressBytes(bytes.slice(0, 128))).toBeNull();

    const badMagic = new Uint32Array(PROGRESS_WIRE_WORDS);
    badMagic[WORD_MAGIC] = PROGRESS_MAGIC_DNPR ^ 1;
    expect(parseTrainerProgressWords(badMagic)).toBeNull();

    const shared = new Int32Array(new ArrayBuffer(PROGRESS_WIRE_BYTES));
    const view = new Uint8Array(shared.buffer);
    view.set(bytes);
    expect(readTrainerProgressShared(shared)?.seq).toBe(8);

    Atomics.store(shared, WORD_SEQ, 9);
    expect(readTrainerProgressShared(shared)).toBeNull();
    expect(readTrainerProgressShared(new Int32Array(8))).toBeNull();
  });

  it("roundtrips randomized seeded frames and computes tone match percentage", () => {
    let seed = 0x1357_9bdf;
    const nextU32 = () => {
      seed = (Math.imul(seed, 1_664_525) + 1_013_904_223) >>> 0;
      return seed;
    };
    for (let trial = 0; trial < 20; trial += 1) {
      const seq = (nextU32() & 0xfffe) >>> 0;
      const permille = nextU32() % 1001;
      const waveA = Array.from({ length: 64 }, () => ((nextU32() % 255) - 127) | 0);
      const waveB = Array.from({ length: 64 }, () => ((nextU32() % 255) - 127) | 0);
      const ltasQ20 = Array.from({ length: 32 }, () => ((nextU32() % 255) - 127) | 0);
      const frame = makeSampleFrame({
        seq,
        progressPermille: permille,
        targetWaveI8: waveA,
        modelWaveI8: waveB,
        targetWave: waveA.map((v) => v / 127),
        modelWave: waveB.map((v) => v / 127),
        ltasErrorQ20: ltasQ20,
        ltasErrorDb: ltasQ20.map((q) => q * 0.05),
      });
      const decoded = parseTrainerProgressBytes(encodeTrainerProgressBytes(frame));
      expect(decoded?.seq).toBe(seq);
      expect(decoded?.progressPermille).toBe(permille);
      expect(decoded?.targetWaveI8).toEqual(waveA);
      expect(decoded?.ltasErrorQ20).toEqual(ltasQ20);
    }

    expect(computeToneMatchPercent(makeSampleFrame())).toBeGreaterThan(94);
    expect(
      computeToneMatchPercent(
        makeSampleFrame({
          activeTier: null,
          tierBesrDb: [Number.NaN, Number.NaN, Number.NaN, Number.NaN],
          bestLoss: 0.01,
        }),
      ),
    ).toBeCloseTo(90, 2);
    expect(
      computeToneMatchPercent(
        makeSampleFrame({
          activeTier: null,
          tierBesrDb: [Number.NaN, Number.NaN, Number.NaN, Number.NaN],
          bestLoss: Number.NaN,
        }),
      ),
    ).toBeNull();
  });

  it("polls shared seqlock updates in useTrainerProgress", async () => {
    const sab = new ArrayBuffer(PROGRESS_WIRE_BYTES);
    const shared = new Int32Array(sab);
    new Uint8Array(sab).set(
      encodeTrainerProgressBytes(makeSampleFrame({ seq: 2, progressPermille: 125 })),
    );
    const nextBytes = encodeTrainerProgressBytes(
      makeSampleFrame({ seq: 4, progressPermille: 880 }),
    );

    await render(<DynamicProgressHarness nextBytes={nextBytes} sab={sab} shared={shared} />);
    await expect.element(page.getByTestId("hook-seq")).toHaveTextContent("4");
    await expect.element(page.getByTestId("hook-permille")).toHaveTextContent("880");
  });
});
