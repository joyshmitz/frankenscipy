use fsci_conformance::{HarnessConfig, run_smoke};

#[test]
fn smoke_report_is_stable() {
    let cfg = HarnessConfig::default_paths();
    let report = run_smoke(&cfg).expect("smoke packet should run");
    assert_eq!(report.suite, "smoke");
    assert!(report.cases_run >= 1);
    assert_eq!(report.failed_cases, 0);
    assert!(report.oracle_present);
}
