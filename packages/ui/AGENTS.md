# `packages/ui` — Unified Design System (`@dnam-project/ui`) Routing Guide

**`@dnam-project/ui`** provides the shared React 19, Tailwind CSS v4 (`@tailwindcss/vite`), `shadcn/ui` (`@base-ui/react`), and `lucide-react` design system for `packages/studio`, `packages/js` WebAudio hosts, and Astro Starlight (`docs/`). Every module is imported via concrete subpath exports with zero internal barrels (`ref[repo.code.modular-decomposition-and-tests]`).

## 1. Key Modules

- [`src/styles.css`](src/styles.css) & [`src/tokens.ts`](src/tokens.ts): Tailwind CSS v4 `@theme` design tokens and palette constants (`@dnam-project/ui/tokens`, `./styles.css`).
- [`src/lib/utils.ts`](src/lib/utils.ts): `cn()` (`clsx` + `tailwind-merge`) and numerical formatting (`@dnam-project/ui/utils`).
- [`src/host/`](src/host/types.ts): `DnamAudioHost` ([`types.ts`](src/host/types.ts)), `<DnamAudioHostProvider>` ([`context.tsx`](src/host/context.tsx)), `WebAudioWorkletHost` ([`web_worklet_host.ts`](src/host/web_worklet_host.ts)), and `256 B` `DNPR` seqlock reader ([`trainer-progress.ts`](src/host/trainer-progress.ts)).
- [`src/components/base/`](src/components/base/button.tsx): `shadcn/ui` Base UI primitives (`@dnam-project/ui/base/*`).
- [`src/components/controls/`](src/components/controls/RotaryKnob.tsx): `RotaryKnob`, `DiscreteSwitch`, `StatusBadges`, and [`geometry.ts`](src/components/controls/geometry.ts) (`@dnam-project/ui/controls/*`).
- [`src/components/rack/`](src/components/rack/DnamPluginRack.tsx) & [`src/components/workflows/`](src/components/workflows/TrainingTimelinePanel.tsx): `DnamPluginRack`, `TriPhysicsTelemetry`, `CapturePlanPanel`, `TrainingTimelinePanel`, and musician-facing `TrainingFitVisualizer` (`@dnam-project/ui/{rack,workflows}/*`).
- [`src/components/visualizers/`](src/components/visualizers/TrainerTelemetryVisualizer.tsx): Interactive `SONA` DSP and `DNPR` trainer/WebGPU telemetry visualizers (`@dnam-project/ui/visualizers/*`).

## 2. Guides & Verification

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for Tailwind CSS v4 tokens, `shadcn/ui` Base UI conventions, React Cosmos (`cosmos.config.json`), Vitest Browser Mode + `axe-core` tests, and `moon check ui`.
