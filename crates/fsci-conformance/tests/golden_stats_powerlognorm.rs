#![forbid(unsafe_code)]
//! Golden-artifact pin for `scipy.stats.powerlognorm`.
//!
//! Resolves [frankenscipy-z5tcs]. Loads the frozen reference
//! table from `fixtures/FSCI-P2C-007_powerlognorm_golden.json`
//! and pins fsci-stats PowerLognorm pdf/cdf/sf/ppf against it.
//! Runs offline — catches drift even on workers without scipy
//! (per [frankenscipy-v10ie]).
//!
//! Tolerances match the live diff harness in
//! `diff_stats_powerlognorm.rs` and document the same precision-
//! floor amplification path: PowerLognorm composes
//!   • powf(Φ(−z), c − 1) in pdf
//!   • exp(−s · Φ⁻¹) in ppf
//! Both amplify the underlying erf/erfc series and Beasley-
//! Springer-Moro Φ⁻¹ helper precision floors.

use std::path::PathBuf;

use fsci_stats::{ContinuousDistribution, PowerLognorm};
use serde::Deserialize;

const PDF_TOL: f64 = 1.0e-7;
const CDF_TOL: f64 = 1.0e-7;
const PPF_TOL_ABS: f64 = 5.0e-6;

#[derive(Debug, Deserialize)]
struct PointCase {
    case_id: String,
    c: f64,
    s: f64,
    x: f64,
    pdf: f64,
    cdf: f64,
    sf: f64,
}

#[derive(Debug, Deserialize)]
struct PpfCase {
    case_id: String,
    c: f64,
    s: f64,
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
        .join("fixtures/FSCI-P2C-007_powerlognorm_golden.json")
}

#[test]
fn golden_stats_powerlognorm() {
    let path = fixture_path();
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read powerlognorm golden fixture {path:?}: {e}"));
    let table: GoldenTable =
        serde_json::from_str(&raw).expect("parse powerlognorm golden fixture");
    assert_eq!(table.packet_id, "FSCI-P2C-007");
    assert_eq!(table.family, "stats.powerlognorm");
    assert_eq!(table.oracle, "scipy.stats.powerlognorm");

    let mut max_pdf = 0.0_f64;
    let mut max_cdf = 0.0_f64;
    let mut max_sf = 0.0_f64;
    let mut max_ppf = 0.0_f64;

    for case in &table.points {
        let dist = PowerLognorm::new(case.c, case.s);
        let pdf_diff = (dist.pdf(case.x) - case.pdf).abs();
        let cdf_diff = (dist.cdf(case.x) - case.cdf).abs();
        let sf_diff = (dist.sf(case.x) - case.sf).abs();
        max_pdf = max_pdf.max(pdf_diff);
        max_cdf = max_cdf.max(cdf_diff);
        max_sf = max_sf.max(sf_diff);
        assert!(
            pdf_diff < PDF_TOL,
            "{}: pdf(c={}, s={}, x={}) drift = {pdf_diff} > {PDF_TOL}",
            case.case_id,
            case.c,
            case.s,
            case.x
        );
        assert!(
            cdf_diff < CDF_TOL,
            "{}: cdf(c={}, s={}, x={}) drift = {cdf_diff} > {CDF_TOL}",
            case.case_id,
            case.c,
            case.s,
            case.x
        );
        assert!(
            sf_diff < CDF_TOL,
            "{}: sf(c={}, s={}, x={}) drift = {sf_diff} > {CDF_TOL}",
            case.case_id,
            case.c,
            case.s,
            case.x
        );
    }

    for case in &table.ppf {
        let dist = PowerLognorm::new(case.c, case.s);
        let rust = dist.ppf(case.q);
        let diff = (rust - case.ppf).abs();
        max_ppf = max_ppf.max(diff);
        assert!(
            diff < PPF_TOL_ABS,
            "{}: ppf(c={}, s={}, q={}) drift = {diff}, want {}",
            case.case_id,
            case.c,
            case.s,
            case.q,
            case.ppf
        );
    }

    eprintln!(
        "powerlognorm golden: pdf_max={max_pdf:e} cdf_max={max_cdf:e} \
         sf_max={max_sf:e} ppf_max={max_ppf:e}"
    );
}
