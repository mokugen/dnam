#![forbid(unsafe_code)]
//! Shared 3-Level (`L1` / `L2` / `L3`) Universal Black-Box Phased Qualification Suite (`M25`).

#[path = "qualification_suite/adapter.rs"]
pub(crate) mod adapter;
#[path = "qualification_suite/excitations.rs"]
pub(crate) mod excitations;
#[path = "qualification_suite/harness.rs"]
pub(crate) mod harness;
#[path = "qualification_suite/self_tests.rs"]
mod self_tests;
