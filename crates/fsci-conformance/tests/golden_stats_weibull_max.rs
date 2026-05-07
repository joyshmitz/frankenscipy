#![forbid(unsafe_code)]
//! Golden-artifact pin for `scipy.stats.weibull_max`.
//!
//! Resolves [frankenscipy-re5fw]. Loads the frozen reference
//! table from `fixtures/FSCI-P2C-007_weibull_max_golden.json`
//! and pins fsci-stats WeibullMax pdf/cdf/sf/ppf against it.
//! Runs offline — catches drift even on workers without scipy
//! (per [frankenscipy-v10ie]).
//!
//! Per-family tolerances (WeibullMax is fully closed-form):
//!   • pdf, cdf, sf : 1e-12 absolute
//!   • ppf          : 1e-9  relative

use std::path::PathBuf;

use fsci_stats::{ContinuousDistribution, WeibullMax};
use serde::Deserialize;

const POINT_TOL: f64 = 1.0e-12;
const PPF_REL_TOL: f64 = 1.0e-9;

#[derive(Debug, Deserialize)]
struct PointCase {
    case_id: String,
    c: f64,
    x: f64,
    pdf: f64,
    cdf: f64,
    sf: f64,
}

#[derive(Debug, Deserialize)]
struct PpfCase {
    case_id: String,
    c: f64,
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
        .join("fixtures/FSCI-P2C-007_weibull_max_golden.json")
}

#[test]
fn golden_stats_weibull_max() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read weibull_max golden fixture {path:?}: {e}"));
    let table: GoldenTable =
        serde_json::from_str(&raw).expect("parse weibull_max golden fixture");
    assert_eq!(table.packet_id, "FSCI-P2C-007");
    assert_eq!(table.family, "stats.weibull_max");
    assert_eq!(table.oracle, "scipy.stats.weibull_max");

    let mut max_pdf = 0.0_f64;
    let mut max_cdf = 0.0_f64;
    let mut max_sf = 0.0_f64;
    let mut max_ppf = 0.0_f64;

    for case in &table.points {
        let dist = WeibullMax::new(case.c);
        let pdf_diff = (dist.pdf(case.x) - case.pdf).abs();
        let cdf_diff = (dist.cdf(case.x) - case.cdf).abs();
        let sf_diff = (dist.sf(case.x) - case.sf).abs();
        max_pdf = max_pdf.max(pdf_diff);
        max_cdf = max_cdf.max(cdf_diff);
        max_sf = max_sf.max(sf_diff);
        assert!(
            pdf_diff < POINT_TOL,
            "{}: pdf(c={}, x={}) drift = {pdf_diff} > {POINT_TOL}",
            case.case_id,
            case.c,
            case.x
        );
        assert!(
            cdf_diff < POINT_TOL,
            "{}: cdf(c={}, x={}) drift = {cdf_diff} > {POINT_TOL}",
            case.case_id,
            case.c,
            case.x
        );
        assert!(
            sf_diff < POINT_TOL,
            "{}: sf(c={}, x={}) drift = {sf_diff} > {POINT_TOL}",
            case.case_id,
            case.c,
            case.x
        );
    }

    for case in &table.ppf {
        let dist = WeibullMax::new(case.c);
        let rust = dist.ppf(case.q);
        let scale = case.ppf.abs().max(1.0);
        let diff = (rust - case.ppf).abs();
        max_ppf = max_ppf.max(diff);
        assert!(
            diff < PPF_REL_TOL * scale,
            "{}: ppf(c={}, q={}) drift = {diff}, want {}",
            case.case_id,
            case.c,
            case.q,
            case.ppf
        );
    }

    eprintln!(
        "weibull_max golden: pdf_max={max_pdf:e} cdf_max={max_cdf:e} \
         sf_max={max_sf:e} ppf_max={max_ppf:e}"
    );
}
