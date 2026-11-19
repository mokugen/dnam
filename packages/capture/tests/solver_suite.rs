#![forbid(unsafe_code)]
//! Integration suites for `VarPro` solver oracles, chains, watermarks, topologies, and audits.

#[path = "solver_suite/adversaries_and_topologies.rs"]
mod adversaries_and_topologies;
#[path = "solver_suite/chains_and_watermarks.rs"]
mod chains_and_watermarks;
#[path = "solver_suite/multisetting_unseen_validation.rs"]
mod multisetting_unseen_validation;
#[path = "solver_suite/oracles.rs"]
mod oracles;
#[path = "solver_suite/routing_topologies.rs"]
mod routing_topologies;
#[path = "solver_suite/step2_3_correctness_audit.rs"]
mod step2_3_correctness_audit;
#[path = "solver_suite/step2_3_regressions.rs"]
mod step2_3_regressions;
