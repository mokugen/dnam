#![forbid(unsafe_code)]
//! `M25` Phase A component-isolating fixture qualification tests (`L1 -> L2 -> L3`, CPU).

use dnam_capture::solver::ComputeBackendPreference;
use dnam_fixtures::Chain05PureLtiCabAndRoom;

use super::harness::{QualifyConfig, assert_qualification_passed, qualify_fixture_3level};

#[test]
#[ignore = "M25 Phase A qualification (run via moon run capture:qualify)"]
fn test_m25_a01_lti_cab_room() {
    let report = qualify_fixture_3level(
        Chain05PureLtiCabAndRoom::new,
        QualifyConfig::canonical(ComputeBackendPreference::Cpu),
    );
    assert_eq!(report.fixture_id, "Chain-05-Pure-LTI-Cab-And-Room");
    assert_eq!(report.num_controls, 0);
    assert!(report.level3.is_none(), "P=0 fixture must skip Level 3");
    assert_qualification_passed(&report);
}
