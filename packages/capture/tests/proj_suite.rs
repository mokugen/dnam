#![forbid(unsafe_code)]
//! Integration suites for `.dnam-proj` containers, `AUTH`/`DAU1` security, and capture vault flows.

#[path = "proj_suite/adversarial_regressions.rs"]
mod adversarial_regressions;
#[path = "proj_suite/bundle_and_vault_edges.rs"]
mod bundle_and_vault_edges;
#[path = "proj_suite/dau1_security.rs"]
mod dau1_security;
#[path = "proj_suite/security.rs"]
mod security;
#[path = "proj_suite/test_support.rs"]
mod test_support;
