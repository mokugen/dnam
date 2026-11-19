import { Activity } from "lucide-react";
import { useId, useState } from "react";
import type { TrainerProgressFrame } from "../../host/trainer-progress.js";
import { DNAM_MODEL_TIER_LABELS, DNAM_MODEL_TIERS, type DnamModelTier } from "../../host/types.js";
import { cn } from "../../lib/utils.js";
import { Badge } from "../base/badge.js";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../base/card.js";

export interface TrainerTelemetryVisualizerProps {
  readonly frame?: TrainerProgressFrame | null | undefined;
}

const DEFAULT_SAMPLE_FRAME: TrainerProgressFrame = (() => {
  const targetWaveI8 = Array.from({ length: 64 }, (_, i) => Math.round(92 * Math.sin(i * 0.24)));
  const modelWaveI8 = Array.from({ length: 64 }, (_, i) => Math.round(90 * Math.sin(i * 0.24)));
  const ltasErrorQ20 = Array.from({ length: 32 }, (_, i) => Math.round(4 * Math.sin(i * 0.4)));
  const kautzPoles = Array.from({ length: 16 }, (_, i) => {
    const rU8 = 185 + Math.round((i % 5) * 14);
    const fU8 = Math.min(255, 18 + i * 14);
    return {
      rU8,
      fU8,
      radius: 0.5 + 0.5 * (rU8 / 255),
      freqHz: 20 * 1000 ** (fU8 / 255),
    };
  });
  return {
    seq: 24,
    status: "running",
    phase: "stage_b_backbone",
    activeTier: "pro",
    clampedControls: 2,
    hasVisualFrame: true,
    progressPermille: 720,
    step: 18,
    totalSteps: 25,
    stageStep: 4,
    stageTotalSteps: 6,
    bestLoss: 0.0018,
    mrIirLoss: 0.0142,
    tierBesrDb: [-21.4, -27.8, -33.2, -36.5],
    tierLtasDb: [0.42, 0.19, 0.08, 0.04],
    targetWave: targetWaveI8.map((v) => v / 127),
    modelWave: modelWaveI8.map((v) => v / 127),
    targetWaveI8,
    modelWaveI8,
    ltasErrorQ20,
    ltasErrorDb: ltasErrorQ20.map((q) => q * 0.05),
    kautzPoles,
  };
})();

interface LtasBarDatum {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly height: number;
  readonly positive: boolean;
}

function buildLtasBars(ltasErrorDb: readonly number[]): readonly LtasBarDatum[] {
  const baselineY = 48;
  const maxBarHalf = 40;
  return Array.from({ length: 32 }, (_, b) => {
    const db = Math.max(-6.35, Math.min(6.35, ltasErrorDb[b] ?? 0));
    const h = Math.max(1.5, (Math.abs(db) / 6.35) * maxBarHalf);
    const y = db >= 0 ? baselineY - h : baselineY;
    return {
      id: `ltas-band-${b}`,
      x: 4 + b * 9.8,
      y,
      height: h,
      positive: db >= 0,
    };
  });
}

function formatDbMetric(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) {
    return "--";
  }
  return `${value.toFixed(2)} dB`;
}

/** Interactive-docs internal diagnostics visualizer for the 256 B DNPR trainer telemetry frame. */
export function TrainerTelemetryVisualizer({ frame }: TrainerTelemetryVisualizerProps) {
  const spectrumTitleId = useId();
  const poleTitleId = useId();
  const effectiveFrame = frame ?? DEFAULT_SAMPLE_FRAME;
  const [selectedTier, setSelectedTier] = useState<DnamModelTier>(
    effectiveFrame.activeTier ?? "pro",
  );
  const selectedIdx = DNAM_MODEL_TIERS.indexOf(selectedTier);
  const ltasBars = buildLtasBars(effectiveFrame.ltasErrorDb);

  const poles = effectiveFrame.kautzPoles.slice(0, 16).map((pole, idx) => ({
    id: `kautz-pole-${idx}`,
    cx: 12 + (pole.fU8 / 255) * 236,
    cy: 86 - ((pole.radius - 0.5) / 0.5) * 74,
  }));

  return (
    <Card data-testid="visualizer-trainer-telemetry">
      <CardHeader className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="size-4 text-cyan-400" />
            <CardTitle>Trainer Internals: 32-Band LTAS &amp; 16-Pole Kautz Diagnostics</CardTitle>
          </div>
          <CardDescription>
            256 B DNPR Seqlock Telemetry (bESR, 1/3-Octave LTAS Error, MR-IIR, Shared Kautz Map)
          </CardDescription>
        </div>
        <Badge variant="control">
          seq={effectiveFrame.seq} ({effectiveFrame.phase})
        </Badge>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <div className="rounded border border-slate-800 bg-slate-950/70 p-2.5">
            <div className="mb-1 flex items-center justify-between font-mono text-xs text-slate-300">
              <span>32-Band 1/3-Octave LTAS Error</span>
              <span className="text-cyan-300">±6.35 dB</span>
            </div>
            <svg aria-labelledby={spectrumTitleId} className="h-24 w-full" viewBox="0 0 320 96">
              <title id={spectrumTitleId}>32-Band 1/3-Octave LTAS Error Spectrum</title>
              <line x1="0" x2="320" y1="48" y2="48" stroke="#334155" strokeWidth="1" />
              {ltasBars.map((bar) => (
                <rect
                  key={bar.id}
                  fill={bar.positive ? "#f59e0b" : "#06b6d4"}
                  height={bar.height}
                  rx="1"
                  width="7"
                  x={bar.x}
                  y={bar.y}
                />
              ))}
            </svg>
          </div>

          <div className="rounded border border-slate-800 bg-slate-950/70 p-2.5">
            <div className="mb-1 flex items-center justify-between font-mono text-xs text-slate-300">
              <span>16-Pole Kautz Resonance Distribution</span>
              <span className="text-amber-300">r ∈ [0.50, 1.00]</span>
            </div>
            <svg aria-labelledby={poleTitleId} className="h-24 w-full" viewBox="0 0 260 96">
              <title id={poleTitleId}>Shared 16-Pole Kautz Log-Frequency Stability Map</title>
              <line
                stroke="#10b981"
                strokeDasharray="3 3"
                strokeWidth="1"
                x1="8"
                x2="252"
                y1="14"
                y2="14"
              />
              {poles.map((pt) => (
                <circle
                  key={pt.id}
                  cx={pt.cx}
                  cy={pt.cy}
                  fill="#f59e0b"
                  r="3.2"
                  stroke="#090a0f"
                  strokeWidth="1"
                />
              ))}
            </svg>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="inline-flex items-center gap-1 rounded border border-slate-800 bg-slate-950 p-1">
            {DNAM_MODEL_TIERS.map((tier) => (
              <button
                key={tier}
                type="button"
                aria-pressed={selectedTier === tier}
                onClick={() => setSelectedTier(tier)}
                className={cn(
                  "rounded px-2 py-0.5 font-mono text-xs transition",
                  selectedTier === tier
                    ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-400/50"
                    : "text-slate-400 hover:text-slate-200",
                )}
              >
                {DNAM_MODEL_TIER_LABELS[tier]}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-2 font-mono text-xs sm:grid-cols-4">
            <div className="rounded border border-slate-800 bg-slate-950/70 px-2.5 py-1.5">
              <span className="text-slate-400">bESR: </span>
              <span className="text-emerald-300">
                {formatDbMetric(effectiveFrame.tierBesrDb[selectedIdx])}
              </span>
            </div>
            <div className="rounded border border-slate-800 bg-slate-950/70 px-2.5 py-1.5">
              <span className="text-slate-400">LTAS: </span>
              <span className="text-cyan-300">
                {formatDbMetric(effectiveFrame.tierLtasDb[selectedIdx])}
              </span>
            </div>
            <div className="rounded border border-slate-800 bg-slate-950/70 px-2.5 py-1.5">
              <span className="text-slate-400">MR-IIR: </span>
              <span className="text-amber-300">
                {Number.isFinite(effectiveFrame.mrIirLoss)
                  ? effectiveFrame.mrIirLoss.toFixed(4)
                  : "--"}
              </span>
            </div>
            <div className="rounded border border-slate-800 bg-slate-950/70 px-2.5 py-1.5">
              <span className="text-slate-400">Clamped: </span>
              <span className="text-violet-300">{effectiveFrame.clampedControls}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
