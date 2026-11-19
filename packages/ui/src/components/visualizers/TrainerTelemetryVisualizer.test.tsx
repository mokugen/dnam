import { describe, expect, it } from "vitest";
import { page, userEvent } from "vitest/browser";
import { render } from "vitest-browser-react";
import { expectZeroAxeViolations } from "../../../tests/setup.js";
import type { TrainerProgressFrame } from "../../host/trainer-progress.js";
import { TrainerTelemetryVisualizer } from "./TrainerTelemetryVisualizer.js";

describe("TrainerTelemetryVisualizer", () => {
  it("renders default interactive-docs telemetry and switches selected tier metrics", async () => {
    const customFrame: TrainerProgressFrame = {
      seq: 18,
      status: "running",
      phase: "stage_b_backbone",
      activeTier: "full",
      clampedControls: 4,
      hasVisualFrame: true,
      progressPermille: 550,
      step: 11,
      totalSteps: 20,
      stageStep: 2,
      stageTotalSteps: 4,
      bestLoss: 0.0031,
      mrIirLoss: 0.0215,
      tierBesrDb: [-18.25, -24.75, Number.NaN, Number.NaN],
      tierLtasDb: [0.55, 0.22, Number.NaN, Number.NaN],
      targetWave: [],
      modelWave: [],
      targetWaveI8: [],
      modelWaveI8: [],
      ltasErrorQ20: Array.from({ length: 32 }, (_, i) => i - 16),
      ltasErrorDb: Array.from({ length: 32 }, (_, i) => (i - 16) * 0.05),
      kautzPoles: [{ rU8: 200, fU8: 90, radius: 0.89, freqHz: 229 }],
    };

    const screen = await render(
      <div className="flex flex-col gap-4">
        <TrainerTelemetryVisualizer />
        <TrainerTelemetryVisualizer frame={customFrame} />
      </div>,
    );

    await expect.element(page.getByText(/seq=18/)).toBeVisible();
    await expect.element(page.getByText("-24.75 dB")).toBeVisible();

    const ultraButtons = page.getByRole("button", { name: "Ultra" });
    await userEvent.click(ultraButtons.nth(1));
    await expect.element(page.getByText("--").first()).toBeVisible();

    await expectZeroAxeViolations(screen.container);
  });
});
