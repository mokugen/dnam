# `@dnam-project/ui` — DNAM Unified Design System

Shared React 19, Tailwind CSS v4 (`@tailwindcss/vite`), `shadcn/ui`, and `lucide-react` design system for the **Dynamic Neural Audio Modeler (DNAM)** Tauri v2 Desktop Studio (`packages/studio`), WebAudio SDK (`packages/js`), and Astro/Starlight Documentation (`docs/`).

## Subpath Exports (Zero Barrel Files)

- `@dnam-project/ui/styles.css` — Shared Tailwind CSS v4 `@theme` and Starlight stylesheet.
- `@dnam-project/ui/tokens` — Design token constants and color/typography descriptors.
- `@dnam-project/ui/utils` — Class-name and decibel formatting utilities (`cn()`, `formatDecibels()`).
- `@dnam-project/ui/base/*` — `shadcn/ui` Base UI primitives (`badge`, `button`, `card`, `progress`, `tabs`).
- `@dnam-project/ui/controls/*` — `RotaryKnob`, `DiscreteSwitch`, `StatusBadges`, and SVG `geometry`.
- `@dnam-project/ui/rack/*` — `DnamPluginRack`, `TriPhysicsTelemetry`, and rack model types.
- `@dnam-project/ui/workflows/*` — `CapturePlanPanel`, `TrainingTimelinePanel`, and musician-facing `TrainingFitVisualizer`.
- `@dnam-project/ui/visualizers/*` — Interactive `SONA` DSP and `TrainerTelemetryVisualizer` (`DNPR` seqlock) components.
- `@dnam-project/ui/host/*` — `DnamAudioHost` (`types`), `DnamAudioHostProvider` (`context`), `WebAudioWorkletHost` (`web_worklet_host`), and `256 B` seqlock reader (`trainer-progress`).
