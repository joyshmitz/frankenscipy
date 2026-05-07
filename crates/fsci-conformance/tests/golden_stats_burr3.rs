#![forbid(unsafe_code)]
//! Golden-artifact pin for `scipy.stats.burr` (Burr Type III).
//!
//! Resolves [frankenscipy-m65b9]. Loads the frozen reference
//! table from `fixtures/FSCI-P2C-007_burr3_golden.json` and
//! pins fsci-stats Burr3 pdf/cdf/sf/ppf against it. Runs offline
//! — catches drift even on workers without scipy
//! (per [frankenscipy-v10ie]).
//!
//! Per-family tolerances (Burr3 is fully closed-form, so
//! tighter than kstwobign's series-based pins):
//!   • pdf, cdf, sf : 1e-12 absolute
//!   • ppf          : 1e-9  relative

use std::path::PathBuf;

use fsci_stats::{Burr3, ContinuousDistribution};
use serde::Deserialize;

const POINT_TOL: f64 = 1.0e-12;
const PPF_REL_TOL: f64 = 1.0e-9;

#[derive(Debug, Deserialize)]
struct PointCase {
    case_id: String,
    c: f64,
    d: f64,
    x: f64,
    pdf: f64,
    cdf: f64,
    sf: f64,
}

#[derive(Debug, Deserialize)]
struct PpfCase {
    case_id: String,
    c: f64,
    d: f64,
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
        .join("fixtures/FSCI-P2C-007_burr3_golden.json")
}

#[test]
fn golden_stats_burr3() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read burr3 golden fixture {path:?}: {e}"));
    let table: GoldenTable =
        serde_json::from_str(&raw).expect("parse burr3 golden fixture");
    assert_eq!(table.packet_id, "FSCI-P2C-007");
    assert_eq!(table.family, "stats.burr");
    assert_eq!(table.oracle, "scipy.stats.burr");

    let mut max_pdf = 0.0_f64;
    let mut max_cdf = 0.0_f64;
    let mut max_sf = 0.0_f64;
    let mut max_ppf = 0.0_f64;

    for case in &table.points {
        let dist = Burr3::new(case.c, case.d);
        let pdf_diff = (dist.pdf(case.x) - case.pdf).abs();
        let cdf_diff = (dist.cdf(case.x) - case.cdf).abs();
        let sf_diff = (dist.sf(case.x) - case.sf).abs();
        max_pdf = max_pdf.max(pdf_diff);
        max_cdf = max_cdf.max(cdf_diff);
        max_sf = max_sf.max(sf_diff);
        assert!(
            pdf_diff < POINT_TOL,
            "{}: pdf(c={}, d={}, x={}) drift = {pdf_diff} > {POINT_TOL}",
            case.case_id,
            case.c,
            case.d,
            case.x
        );
        assert!(
            cdf_diff < POINT_TOL,
            "{}: cdf(c={}, d={}, x={}) drift = {cdf_diff} > {POINT_TOL}",
            case.case_id,
            case.c,
            case.d,
            case.x
        );
        assert!(
            sf_diff < POINT_TOL,
            "{}: sf(c={}, d={}, x={}) drift = {sf_diff} > {POINT_TOL}",
            case.case_id,
            case.c,
            case.d,
            case.x
        );
    }

    for case in &table.ppf {
        let dist = Burr3::new(case.c, case.d);
        let rust = dist.ppf(case.q);
        let scale = case.ppf.abs().max(1.0);
        let diff = (rust - case.ppf).abs();
        max_ppf = max_ppf.max(diff);
        assert!(
            diff < PPF_REL_TOL * scale,
            "{}: ppf(c={}, d={}, q={}) drift = {diff}, want {}",
            case.case_id,
            case.c,
            case.d,
            case.q,
            case.ppf
        );
    }

    eprintln!(
        "burr3 golden: pdf_max={max_pdf:e} cdf_max={max_cdf:e} \
         sf_max={max_sf:e} ppf_max={max_ppf:e}"
    );
}
