import { Lock } from "lucide-react";
import { useId } from "react";
import {
  computeToneMatchPercent,
  type TrainerProgressFrame,
  type TrainingPhaseName,
  type TrainingStatusName,
} from "../../host/trainer-progress.js";
import { DNAM_MODEL_TIER_LABELS, DNAM_MODEL_TIERS, type DnamModelTier } from "../../host/types.js";
import { cn } from "../../lib/utils.js";

export interface TrainingFitVisualizerProps {
  readonly frame?: TrainerProgressFrame | null | undefined;
  readonly status?: TrainingStatusName | undefined;
  readonly progressFraction?: number | undefined;
}

interface EqBandGroup {
  readonly id: string;
  readonly label: string;
  readonly startBand: number;
  readonly endBand: number;
}

const EQ_BAND_GROUPS: readonly EqBandGroup[] = [
  { id: "bass", label: "Bass", startBand: 0, endBand: 8 },
  { id: "mids", label: "Mids", startBand: 8, endBand: 18 },
  { id: "presence", label: "Presence", startBand: 18, endBand: 26 },
  { id: "air", label: "Air", startBand: 26, endBand: 32 },
] as const;

const PHASE_SUBTITLES: Readonly<Record<TrainingPhaseName, string>> = {
  idle: "Ready to Listen",
  capture: "Recording Your Gear",
  stage_a_physics_probe: "Listening to Dynamics",
  stage_b_backbone: "Shaping Tube & Pedal Feel",
  stage_4_spatial_cascade: "Matching Room & Air",
  stage_c_knobs: "Mapping Knob Sweeps",
  stage_d_audit: "Verifying Safe Knob Ranges",
  live_null_test: "Final Sound Check",
  completed: "Tone Locked",
  failed: "Check Audio Connection",
} as const;

function resolveFriendlySubtitle(
  status: TrainingStatusName,
  phase: TrainingPhaseName | undefined,
): string {
  if (status === "completed") {
    return PHASE_SUBTITLES.completed;
  }
  if (status === "failed") {
    return PHASE_SUBTITLES.failed;
  }
  if (status === "idle") {
    return PHASE_SUBTITLES.idle;
  }
  if (phase !== undefined && phase !== "idle") {
    return PHASE_SUBTITLES[phase];
  }
  return PHASE_SUBTITLES.stage_b_backbone;
}

function buildFallbackWaveforms(toneLockRatio: number): {
  readonly target: number[];
  readonly modeled: number[];
} {
  const drift = Math.max(0, 1 - toneLockRatio);
  const target: number[] = [];
  const modeled: number[] = [];
  for (let i = 0; i < 64; i += 1) {
    const phase = i * 0.24;
    const env = Math.exp(-i * 0.015);
    const refVal = env * (0.78 * Math.sin(phase) + 0.22 * Math.sin(phase * 2.3));
    const curVal =
      (1 - 0.35 * drift) *
      env *
      (0.78 * Math.sin(phase + 0.45 * drift) + 0.22 * Math.sin(phase * 2.3));
    target.push(refVal);
    modeled.push(curVal);
  }
  return { target, modeled };
}

function samplesToSvgPoints(samples: readonly number[], width: number, height: number): string {
  const count = Math.max(2, samples.length);
  const midY = height * 0.5;
  const amp = height * 0.4;
  return samples
    .map((sample, idx) => {
      const x = (idx / (count - 1)) * width;
      const y = midY - Math.max(-1, Math.min(1, sample)) * amp;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function computeEqBandLocks(
  frame: TrainerProgressFrame | null | undefined,
  fallbackRatio: number,
): readonly { readonly id: string; readonly label: string; readonly lockPct: number }[] {
  const hasSpectrum = Boolean(frame?.hasVisualFrame && frame.ltasErrorDb.length >= 32);
  return EQ_BAND_GROUPS.map((group, idx) => {
    if (!hasSpectrum || !frame) {
      const pct = Math.max(0, Math.min(100, fallbackRatio * (96 + idx)));
      return { id: group.id, label: group.label, lockPct: pct };
    }
    let sumSq = 0;
    const len = group.endBand - group.startBand;
    for (let b = group.startBand; b < group.endBand; b += 1) {
      const err = frame.ltasErrorDb[b] ?? 0;
      sumSq += err * err;
    }
    const rmsDb = Math.sqrt(sumSq / Math.max(1, len));
    const pct = Math.max(0, Math.min(100, 100 - rmsDb * 16));
    return { id: group.id, label: group.label, lockPct: pct };
  });
}

function isTierReady(
  tier: DnamModelTier,
  tierIndex: number,
  frame: TrainerProgressFrame | null | undefined,
  status: TrainingStatusName,
  toneLockPct: number,
): boolean {
  if (status === "completed") {
    return true;
  }
  if (frame !== null && frame !== undefined) {
    const db = frame.tierBesrDb[tierIndex];
    if (db !== undefined && Number.isFinite(db)) {
      return true;
    }
    if (frame.activeTier === tier) {
      return true;
    }
  }
  return toneLockPct >= (tierIndex + 1) * 24;
}

/** Musician-friendly "Tone Lock" sound convergence view free of solver/DSP jargon. */
export function TrainingFitVisualizer({
  frame,
  status,
  progressFraction,
}: TrainingFitVisualizerProps) {
  const waveTitleId = useId();
  const effectiveStatus: TrainingStatusName = frame?.status ?? status ?? "idle";
  const subtitle = resolveFriendlySubtitle(effectiveStatus, frame?.phase);
  const frameMatch = frame ? computeToneMatchPercent(frame) : null;
  const fallbackFraction = Math.max(
    0,
    Math.min(1, progressFraction ?? (effectiveStatus === "completed" ? 1 : 0)),
  );
  const toneLockPct =
    frameMatch !== null
      ? frameMatch
      : frame !== null && frame !== undefined
        ? Math.max(0, Math.min(100, frame.progressPermille * 0.1))
        : fallbackFraction * 100;

  const wavePair =
    frame?.hasVisualFrame && frame.targetWave.length >= 2 && frame.modelWave.length >= 2
      ? { target: frame.targetWave, modeled: frame.modelWave }
      : buildFallbackWaveforms(toneLockPct / 100);

  const targetPoints = samplesToSvgPoints(wavePair.target, 320, 76);
  const modelPoints = samplesToSvgPoints(wavePair.modeled, 320, 76);
  const eqBands = computeEqBandLocks(frame, toneLockPct / 100);

  return (
    <div
      data-testid="training-fit-visualizer"
      className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-950/70 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Lock className="size-3.5 text-emerald-400" />
          <span className="text-xs font-semibold text-slate-100">Tone Lock</span>
          <span className="text-xs text-slate-400">{subtitle}</span>
        </div>
        <span
          data-testid="tone-lock-percent"
          className="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-xs font-semibold text-emerald-300"
        >
          {toneLockPct.toFixed(1)}%
        </span>
      </div>

      <div className="rounded border border-slate-800/80 bg-slate-900/60 p-2">
        <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
          <span>Sound Match Preview</span>
          <span className="font-mono">Target vs. Modeled</span>
        </div>
        <svg
          aria-labelledby={waveTitleId}
          className="h-16 w-full"
          preserveAspectRatio="none"
          viewBox="0 0 320 76"
        >
          <title id={waveTitleId}>Target Gear vs Modeled Tone Waveform Match</title>
          <line x1="0" x2="320" y1="38" y2="38" stroke="#1e293b" strokeWidth="1" />
          <polyline
            fill="none"
            points={targetPoints}
            stroke="#22d3ee"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeOpacity="0.6"
            strokeWidth="2"
          />
          <polyline
            fill="none"
            points={modelPoints}
            stroke="#f59e0b"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
          />
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {eqBands.map((band) => (
          <div
            key={band.id}
            className="flex flex-col gap-1 rounded border border-slate-800/80 bg-slate-900/50 p-2"
          >
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300">{band.label}</span>
              <span className="font-mono text-emerald-300">{Math.round(band.lockPct)}%</span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full bg-emerald-400 transition-all"
                style={{ width: `${band.lockPct.toFixed(1)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="text-slate-400">Quality Profiles Ready</span>
        <div className="flex items-center gap-1.5">
          {DNAM_MODEL_TIERS.map((tier, idx) => {
            const ready = isTierReady(tier, idx, frame, effectiveStatus, toneLockPct);
            return (
              <span
                key={tier}
                className={cn(
                  "rounded border px-2 py-0.5 font-mono text-xs",
                  ready
                    ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-200"
                    : "border-slate-800 bg-slate-900 text-slate-500",
                )}
              >
                {DNAM_MODEL_TIER_LABELS[tier]}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
