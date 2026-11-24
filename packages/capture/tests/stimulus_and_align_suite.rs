#![forbid(unsafe_code)]
//! Integration suites for adaptive stimulus generation, alignment, resampling, and spec contracts.

#[path = "stimulus_and_align_suite/align_and_capsules_edges.rs"]
mod align_and_capsules_edges;
#[path = "stimulus_and_align_suite/dual_mode_and_align.rs"]
mod dual_mode_and_align;
#[path = "stimulus_and_align_suite/live_null_autopatch.rs"]
mod live_null_autopatch;
#[path = "stimulus_and_align_suite/multirate_bitdepth_fixtures.rs"]
mod multirate_bitdepth_fixtures;
#[path = "stimulus_and_align_suite/schoukens_pintelon_and_hybrid_align.rs"]
mod schoukens_pintelon_and_hybrid_align;
#[path = "stimulus_and_align_suite/spec_verification.rs"]
mod spec_verification;
#[path = "stimulus_and_align_suite/step2_2_adversarial_edges.rs"]
mod step2_2_adversarial_edges;
#[path = "stimulus_and_align_suite/stimulus_and_safeguards.rs"]
mod stimulus_and_safeguards;
