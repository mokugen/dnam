import { ClampAuditVisualizer } from "./ClampAuditVisualizer.js";
import { ControlBusVisualizer } from "./ControlBusVisualizer.js";
import { KroneckerResumeVisualizer } from "./KroneckerResumeVisualizer.js";
import { ModulatedUfdnVisualizer } from "./ModulatedUfdnVisualizer.js";
import { PolyAdaaVisualizer } from "./PolyAdaaVisualizer.js";
import { RgLruPoleVisualizer } from "./RgLruPoleVisualizer.js";
import { TrainerTelemetryVisualizer } from "./TrainerTelemetryVisualizer.js";
import { ZohRescalerVisualizer } from "./ZohRescalerVisualizer.js";

export default {
  "1. Core 1: 1x B-Spline PolyADAA": <PolyAdaaVisualizer />,
  "2. Core 1: Radial-Polar RG-LRU Pole Trajectory": <RgLruPoleVisualizer />,
  "3. Core 2: Modulated UFDN Dual-Tap Crossfade": <ModulatedUfdnVisualizer />,
  "4. Control Bus: Lobatto HDMR-FM & 3 kHz LFO": <ControlBusVisualizer />,
  "5. Iterative Resume: Surrogate Delay & Kronecker": <KroneckerResumeVisualizer />,
  "6. Stability Registry: Mahalanobis Trust Region": <ClampAuditVisualizer />,
  "7. Container: Continuous-Time ZOH Rescaler": <ZohRescalerVisualizer />,
  "8. Trainer Internals: LTAS & Kautz Telemetry": <TrainerTelemetryVisualizer />,
};
