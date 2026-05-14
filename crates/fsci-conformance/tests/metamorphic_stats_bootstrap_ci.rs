#![forbid(unsafe_code)]
//! Metamorphic invariants for `fsci_stats::bootstrap_ci(data,
//! stat_fn, n_bootstrap, confidence, seed)` — percentile-
//! bootstrap confidence interval for an arbitrary statistic.
//!
//! Resolves [frankenscipy-i19ff]. fsci uses a deterministic
//! LCG to draw indices and computes the percentile CI from the
//! sorted bootstrap distribution; scipy.stats.bootstrap uses
//! PCG64 and BCa by default, so direct parity is impossible.
//! This harness verifies invariants any sensible bootstrap CI
//! procedure must satisfy:
//!
//!   1. Ordering: lo ≤ hi.
//!   2. Coverage: lo ≤ original_stat ≤ hi for the mean
//!      statistic (typical data; a percentile bootstrap of the
//!      mean must contain the empirical mean for a centered
//!      bootstrap distribution).
//!   3. Determinism: same seed → identical (lo, hi).
//!   4. Confidence monotonicity: width(99%) ≥ width(95%) ≥
//!      width(80%) for the same data + seed.
//!   5. NaN guard: empty data or n_bootstrap=0 returns
//!      (NaN, NaN).
//!
//! 3 fixtures × variable invariants ≈ 18 cases.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::bootstrap_ci;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseLog {
    case_id: String,
    invariant: String,
    detail: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MetamorphicLog {
    test_id: String,
    case_count: usize,
    pass_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseLog>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/artifacts/{PACKET_ID}/metamorphic"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create bootstrap_ci metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json =
        serde_json::to_string_pretty(log).expect("serialize bootstrap_ci metamorphic log");
    fs::write(path, json).expect("write bootstrap_ci metamorphic log");
}

fn mean(d: &[f64]) -> f64 {
    d.iter().sum::<f64>() / d.len() as f64
}

#[test]
fn metamorphic_stats_bootstrap_ci() {
    let start = Instant::now();
    let mut cases = Vec::new();

    let fixtures: Vec<(&str, Vec<f64>)> = vec![
        // Compact, deterministic
        ("compact_n10", (1..=10).map(|i| i as f64).collect()),
        // Larger sample, mild spread
        (
            "spread_n30",
            (1..=30).map(|i| (i as f64).sqrt() * 4.0).collect(),
        ),
        // Includes negatives
        (
            "with_negatives_n12",
            vec![
                -3.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0, 16.0,
            ],
        ),
    ];

    let n_boot = 5000_usize;
    let conf = 0.95_f64;
    let seed = 42_u64;

    for (name, data) in &fixtures {
        let orig_mean = mean(data);
        let (lo, hi) = bootstrap_ci(data, mean, n_boot, conf, seed);

        // Invariant 1: lo ≤ hi.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "lo_le_hi".into(),
            detail: format!("lo={lo}, hi={hi}"),
            pass: lo.is_finite() && hi.is_finite() && lo <= hi,
        });

        // Invariant 2: coverage of empirical statistic. The percentile
        //              bootstrap of the mean must bracket the original
        //              empirical mean (the bootstrap distribution is
        //              centered on it).
        let bracket = lo <= orig_mean && orig_mean <= hi;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "covers_empirical_mean".into(),
            detail: format!(
                "lo={lo}, mean={orig_mean}, hi={hi}, bracketed={bracket}"
            ),
            pass: bracket,
        });

        // Invariant 3: determinism at fixed seed.
        let (lo2, hi2) = bootstrap_ci(data, mean, n_boot, conf, seed);
        let determ = lo == lo2 && hi == hi2;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "deterministic_same_seed".into(),
            detail: format!("first=({lo}, {hi}), second=({lo2}, {hi2})"),
            pass: determ,
        });

        // Invariant 4: confidence monotonicity. Higher confidence →
        //              wider interval.
        let (lo80, hi80) = bootstrap_ci(data, mean, n_boot, 0.80, seed);
        let (lo99, hi99) = bootstrap_ci(data, mean, n_boot, 0.99, seed);
        let w80 = hi80 - lo80;
        let w95 = hi - lo;
        let w99 = hi99 - lo99;
        let widening = w80 <= w95 && w95 <= w99;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "confidence_widens_interval".into(),
            detail: format!("w80={w80}, w95={w95}, w99={w99}"),
            pass: widening,
        });
    }

    // Invariant 5: degenerate inputs return NaN.
    let empty: Vec<f64> = vec![];
    let (lo_e, hi_e) = bootstrap_ci(&empty, mean, n_boot, conf, seed);
    cases.push(CaseLog {
        case_id: "edge_empty_data".into(),
        invariant: "empty_data_returns_nan".into(),
        detail: format!("lo={lo_e}, hi={hi_e}"),
        pass: lo_e.is_nan() && hi_e.is_nan(),
    });
    let single = vec![1.0, 2.0, 3.0];
    let (lo_z, hi_z) = bootstrap_ci(&single, mean, 0, conf, seed);
    cases.push(CaseLog {
        case_id: "edge_zero_bootstrap".into(),
        invariant: "zero_bootstrap_returns_nan".into(),
        detail: format!("lo={lo_z}, hi={hi_z}"),
        pass: lo_z.is_nan() && hi_z.is_nan(),
    });

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_stats_bootstrap_ci".into(),
        case_count: cases.len(),
        pass_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: cases.clone(),
    };

    emit_log(&log);

    for c in &cases {
        if !c.pass {
            eprintln!(
                "bootstrap_ci metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "bootstrap_ci metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
