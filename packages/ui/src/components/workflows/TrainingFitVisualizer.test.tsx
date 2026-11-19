import { describe, expect, it } from "vitest";
import { page } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations } from "../../../tests/setup.js";
import type { TrainerProgressFrame } from "../../host/trainer-progress.js";
import { TrainingFitVisualizer } from "./TrainingFitVisualizer.js";

const FORBIDDEN_SOLVER_JARGON_PATTERN =
  /\b(?:ESR|bESR|MR-IIR|LTAS|VarPro|Kautz|RG-LRU|UFDN|Kronecker|r_max)\b|\|\|U\|\|_2/;

function makeLiveFitFrame(): TrainerProgressFrame {
  const targetWaveI8 = Array.from({ length: 64 }, (_, i) => Math.round(80 * Math.sin(i * 0.3)));
  const modelWaveI8 = Array.from({ length: 64 }, (_, i) => Math.round(78 * Math.sin(i * 0.3)));
  const ltasErrorQ20 = Array.from({ length: 32 }, (_, i) => (i % 5) - 2);
  return {
    seq: 16,
    status: "running",
    phase: "stage_4_spatial_cascade",
    activeTier: "pro",
    clampedControls: 2,
    hasVisualFrame: true,
    progressPermille: 760,
    step: 19,
    totalSteps: 25,
    stageStep: 3,
    stageTotalSteps: 4,
    bestLoss: 0.001,
    mrIirLoss: 0.012,
    tierBesrDb: [-19.2, -24.1, -28.0, Number.NaN],
    tierLtasDb: [0.5, 0.25, 0.11, Number.NaN],
    targetWave: targetWaveI8.map((v) => v / 127),
    modelWave: modelWaveI8.map((v) => v / 127),
    targetWaveI8,
    modelWaveI8,
    ltasErrorQ20,
    ltasErrorDb: ltasErrorQ20.map((q) => q * 0.05),
    kautzPoles: [],
  };
}

describe("TrainingFitVisualizer", () => {
  it("renders Tone Lock, waveform match, EQ lock, and tier badges without solver jargon", async () => {
    const screen = await render(
      <div className="flex flex-col gap-4">
        <TrainingFitVisualizer frame={makeLiveFitFrame()} />
        <TrainingFitVisualizer status="completed" progressFraction={1} />
        <TrainingFitVisualizer status="idle" progressFraction={0} />
      </div>,
    );

    await expect.element(page.getByText("Matching Room & Air")).toBeVisible();
    await expect.element(page.getByText("Tone Locked")).toBeVisible();
    await expect.element(page.getByText("Ready to Listen")).toBeVisible();

    const text = screen.container.textContent ?? "";
    expect(text).not.toMatch(FORBIDDEN_SOLVER_JARGON_PATTERN);
    await expectZeroAxeViolations(screen.container);
  });
});
