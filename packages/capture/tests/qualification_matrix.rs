#![forbid(unsafe_code)]
//! Slow acoustic qualification suites (`--ignored`; run via `moon run capture:qualify`).

#[path = "qualification_matrix/harness.rs"]
mod harness;
#[path = "qualification_matrix/solver_benchmark_matrix.rs"]
mod solver_benchmark_matrix;
#[path = "qualification_matrix/v3_prototype_benchmark.rs"]
mod v3_prototype_benchmark;
