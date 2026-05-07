#![forbid(unsafe_code)]
//! Golden-artifact pin for `scipy.stats.kstwobign`.
//!
//! Resolves [frankenscipy-0a6ol]. Loads the frozen reference
//! table from `fixtures/FSCI-P2C-007_kstwobign_golden.json` and
//! pins fsci-stats KsTwoBign pdf/cdf/sf/ppf against it.
//!
//! Unlike `diff_stats_kstwobign.rs` (which spawns python every
//! run), this test runs offline against a checked-in fixture.
//! That makes it the canary for *fsci-side* drift in either of
//! KsTwoBign's two series branches — the large-x alternating
//! sum (x ≥ 1.18) or the Jacobi-theta dual (x < 1.18).
//!
//! Per-family tolerances:
//!  • cdf, sf : 1e-13 absolute (both fsci and scipy converge
//!    their truncated series to fp epsilon).
//!  • pdf     : 5e-8 absolute. fsci computes pdf as the
//!    analytical derivative of the Jacobi-theta cdf series;
//!    scipy uses a separate cephes routine (kolmogp) whose
//!    own pdf-vs-cdf-derivative drifts by ≈ 1e-8 around
//!    x = 0.85. We're more consistent than scipy here, but
//!    have to absorb scipy's drift to match its frozen value.
//!  • ppf     : 1e-9 relative.

use std::path::PathBuf;

use fsci_stats::{ContinuousDistribution, KsTwoBign};
use serde::Deserialize;

const CDF_TOL: f64 = 1.0e-13;
const SF_TOL: f64 = 1.0e-13;
const PDF_TOL: f64 = 5.0e-8;
const PPF_REL_TOL: f64 = 1.0e-9;

#[derive(Debug, Deserialize)]
struct PointCase {
    case_id: String,
    x: f64,
    pdf: f64,
    cdf: f64,
    sf: f64,
}

#[derive(Debug, Deserialize)]
struct PpfCase {
    case_id: String,
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
        .join("fixtures/FSCI-P2C-007_kstwobign_golden.json")
}

#[test]
fn golden_stats_kstwobign() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read kstwobign golden fixture {path:?}: {e}"));
    let table: GoldenTable =
        serde_json::from_str(&raw).expect("parse kstwobign golden fixture");
    assert_eq!(table.packet_id, "FSCI-P2C-007");
    assert_eq!(table.family, "stats.kstwobign");
    assert_eq!(table.oracle, "scipy.stats.kstwobign");

    let dist = KsTwoBign;
    let mut max_pdf_diff = 0.0_f64;
    let mut max_cdf_diff = 0.0_f64;
    let mut max_sf_diff = 0.0_f64;
    let mut max_ppf_diff = 0.0_f64;

    for case in &table.points {
        let pdf_diff = (dist.pdf(case.x) - case.pdf).abs();
        let cdf_diff = (dist.cdf(case.x) - case.cdf).abs();
        let sf_diff = (dist.sf(case.x) - case.sf).abs();
        max_pdf_diff = max_pdf_diff.max(pdf_diff);
        max_cdf_diff = max_cdf_diff.max(cdf_diff);
        max_sf_diff = max_sf_diff.max(sf_diff);
        assert!(
            pdf_diff < PDF_TOL,
            "{}: pdf({}) drift = {pdf_diff} > {PDF_TOL}",
            case.case_id,
            case.x
        );
        assert!(
            cdf_diff < CDF_TOL,
            "{}: cdf({}) drift = {cdf_diff} > {CDF_TOL}",
            case.case_id,
            case.x
        );
        assert!(
            sf_diff < SF_TOL,
            "{}: sf({}) drift = {sf_diff} > {SF_TOL}",
            case.case_id,
            case.x
        );
    }

    for case in &table.ppf {
        // ppf is bisection-derived — relative tol on the recovered x.
        let rust = dist.ppf(case.q);
        let scale = case.ppf.abs().max(1.0);
        let diff = (rust - case.ppf).abs();
        max_ppf_diff = max_ppf_diff.max(diff);
        assert!(
            diff < PPF_REL_TOL * scale,
            "{}: ppf({}) drift = {diff}, want {}",
            case.case_id,
            case.q,
            case.ppf
        );
    }

    eprintln!(
        "kstwobign golden: pdf_max={max_pdf_diff:e} cdf_max={max_cdf_diff:e} \
         sf_max={max_sf_diff:e} ppf_max={max_ppf_diff:e}"
    );
}
