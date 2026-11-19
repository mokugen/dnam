import { Flame, Play } from "lucide-react";
import type { TrainerProgressFrame } from "../../host/trainer-progress.js";
import { Badge } from "../base/badge.js";
import { Button } from "../base/button.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../base/card.js";
import { Progress } from "../base/progress.js";
import type { TrainingTimelineView } from "./models.js";
import { TrainingFitVisualizer } from "./TrainingFitVisualizer.js";

export interface TrainingTimelinePanelProps {
  readonly timeline: TrainingTimelineView;
  readonly onStartTraining?: (() => void) | undefined;
}

function buildPreviewFrame(
  timeline: TrainingTimelineView,
  progressFraction: number,
): TrainerProgressFrame | null {
  if (timeline.progressFrame !== undefined) {
    return timeline.progressFrame;
  }
  const latest = timeline.history[timeline.history.length - 1];
  if (!latest) {
    return null;
  }
  return {
    seq: 2,
    status: timeline.status,
    phase: timeline.status === "completed" ? "completed" : "stage_b_backbone",
    activeTier: "pro",
    clampedControls: timeline.clampedParameterCount,
    hasVisualFrame: false,
    progressPermille: Math.round(progressFraction * 1000),
    step: latest.epoch,
    totalSteps: latest.totalEpochs,
    stageStep: latest.epoch,
    stageTotalSteps: latest.totalEpochs,
    bestLoss: latest.esrLoss,
    mrIirLoss: latest.mrIirLoss,
    tierBesrDb: [Number.NaN, Number.NaN, Number.NaN, Number.NaN],
    tierLtasDb: [Number.NaN, Number.NaN, Number.NaN, Number.NaN],
    targetWave: [],
    modelWave: [],
    targetWaveI8: [],
    modelWaveI8: [],
    ltasErrorQ20: [],
    ltasErrorDb: [],
    kautzPoles: [],
  };
}

export function TrainingTimelinePanel({ timeline, onStartTraining }: TrainingTimelinePanelProps) {
  const latest = timeline.history[timeline.history.length - 1];
  const progressPct =
    timeline.progressFrame !== undefined && timeline.progressFrame !== null
      ? timeline.progressFrame.progressPermille * 0.1
      : latest && latest.totalEpochs > 0
        ? (latest.epoch / latest.totalEpochs) * 100
        : timeline.status === "completed"
          ? 100
          : 0;
  const fitFrame = buildPreviewFrame(timeline, progressPct / 100);

  return (
    <Card data-testid={`training-timeline-panel-${timeline.modelId}`}>
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Flame className="size-4 text-amber-400" />
            <CardTitle>Two-Sweep Analytical Trainer ({timeline.modelId})</CardTitle>
          </div>
          <CardDescription>
            Automatic Safe Knob Range Protection ({timeline.backendDevice})
          </CardDescription>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="control">{timeline.clampedParameterCount} Auto-Clamped Controls</Badge>
          {onStartTraining ? (
            <Button size="sm" onClick={onStartTraining}>
              <Play data-icon="inline-start" />
              Run Capture Fit
            </Button>
          ) : null}
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between font-mono text-xs text-slate-300">
          <span>Status: {timeline.status.toUpperCase()}</span>
          {latest ? (
            <span>
              Stage: {latest.stage} (Epoch {latest.epoch}/{latest.totalEpochs})
            </span>
          ) : (
            <span>Ready</span>
          )}
        </div>

        <Progress value={progressPct} indicatorClassName="bg-violet-400" />

        <TrainingFitVisualizer
          frame={fitFrame}
          status={timeline.status}
          progressFraction={progressPct / 100}
        />
      </CardContent>
    </Card>
  );
}
