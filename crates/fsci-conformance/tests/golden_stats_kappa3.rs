#![forbid(unsafe_code)]
//! Golden-artifact pin for `scipy.stats.kappa3`.
//!
//! Resolves [frankenscipy-89f2v]. Loads the frozen reference
//! table from `fixtures/FSCI-P2C-007_kappa3_golden.json` and
//! pins fsci-stats Kappa3 pdf/cdf/sf/ppf against it. Runs offline
//! — catches drift even on workers without scipy
//! (per [frankenscipy-v10ie]).
//!
//! Per-family tolerances (Kappa3 is fully closed-form, so
//! tighter than kstwobign's series-based pins):
//!  • pdf, cdf, sf : 1e-12 absolute
//!  • ppf          : 1e-9 relative

use std::path::PathBuf;

use fsci_stats::{ContinuousDistribution, Kappa3};
use serde::Deserialize;

const POINT_TOL: f64 = 1.0e-12;
const PPF_REL_TOL: f64 = 1.0e-9;

#[derive(Debug, Deserialize)]
struct PointCase {
    case_id: String,
    a: f64,
    x: f64,
    pdf: f64,
    cdf: f64,
    sf: f64,
}

#[derive(Debug, Deserialize)]
struct PpfCase {
    case_id: String,
    a: f64,
    q: f64,
    ppf: f64,
}

#[derive(Debug, Deserialize)]
struct GoldenTable {
    packet_id: String,
    family: String,
    oracle: String,
    points: Vec<PointCase>,
    ppf: Vec<PpfCase>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/FSCI-P2C-007_kappa3_golden.json")
}

#[test]
fn golden_stats_kappa3() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read kappa3 golden fixture {path:?}: {e}"));
    let table: GoldenTable =
        serde_json::from_str(&raw).expect("parse kappa3 golden fixture");
    assert_eq!(table.packet_id, "FSCI-P2C-007");
    assert_eq!(table.family, "stats.kappa3");
    assert_eq!(table.oracle, "scipy.stats.kappa3");

    let mut max_pdf_diff = 0.0_f64;
    let mut max_cdf_diff = 0.0_f64;
    let mut max_sf_diff = 0.0_f64;
    let mut max_ppf_diff = 0.0_f64;

    for case in &table.points {
        let dist = Kappa3::new(case.a);
        let pdf_diff = (dist.pdf(case.x) - case.pdf).abs();
        let cdf_diff = (dist.cdf(case.x) - case.cdf).abs();
        let sf_diff = (dist.sf(case.x) - case.sf).abs();
        max_pdf_diff = max_pdf_diff.max(pdf_diff);
        max_cdf_diff = max_cdf_diff.max(cdf_diff);
        max_sf_diff = max_sf_diff.max(sf_diff);
        assert!(
            pdf_diff < POINT_TOL,
            "{}: pdf(a={}, x={}) drift = {pdf_diff} > {POINT_TOL}",
            case.case_id,
            case.a,
            case.x
        );
        assert!(
            cdf_diff < POINT_TOL,
            "{}: cdf(a={}, x={}) drift = {cdf_diff} > {POINT_TOL}",
            case.case_id,
            case.a,
            case.x
        );
        assert!(
            sf_diff < POINT_TOL,
            "{}: sf(a={}, x={}) drift = {sf_diff} > {POINT_TOL}",
            case.case_id,
            case.a,
            case.x
        );
    }

    for case in &table.ppf {
        let dist = Kappa3::new(case.a);
        let rust = dist.ppf(case.q);
        let scale = case.ppf.abs().max(1.0);
        let diff = (rust - case.ppf).abs();
        max_ppf_diff = max_ppf_diff.max(diff);
        assert!(
            diff < PPF_REL_TOL * scale,
            "{}: ppf(a={}, q={}) drift = {diff}, want {}",
            case.case_id,
            case.a,
            case.q,
            case.ppf
        );
    }

    eprintln!(
        "kappa3 golden: pdf_max={max_pdf_diff:e} cdf_max={max_cdf_diff:e} \
         sf_max={max_sf_diff:e} ppf_max={max_ppf_diff:e}"
    );
}
