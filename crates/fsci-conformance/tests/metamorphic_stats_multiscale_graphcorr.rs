#![forbid(unsafe_code)]
//! Metamorphic invariants for `fsci_stats::multiscale_graphcorr`.
//!
//! Resolves [frankenscipy-y5req]. fsci's MGC implementation
//! permutes via a deterministic LCG (random_state seed) while
//! scipy.stats.multiscale_graphcorr uses PCG64, so the
//! permutation pvalue cannot agree at any seed. Furthermore
//! MGC is sensitive to k-NN ranking ties: small numerical
//! differences propagate non-trivially through the centered
//! distance correlation. This harness verifies invariants any
//! sensible MGC implementation must satisfy:
//!
//!   1. Statistic ∈ [0, 1] (centered distance correlation,
//!      bounded by construction).
//!   2. Pvalue ∈ [0, 1].
//!   3. mgc_map shape n × n.
//!   4. opt_scale entries in [1, n].
//!   5. Identical y = x dependence → high statistic (≥ 0.5).
//!   6. Pvalue is exactly 1.0 when reps = 0 (scipy's sentinel).
//!   7. Determinism: same random_state → identical statistic
//!      AND pvalue (across 2 runs).
//!
//! 4 fixtures × variable arms ≈ 24 cases.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::multiscale_graphcorr;
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
    fs::create_dir_all(output_dir())
        .expect("create multiscale_graphcorr metamorphic output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &MetamorphicLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log)
        .expect("serialize multiscale_graphcorr metamorphic log");
    fs::write(path, json).expect("write multiscale_graphcorr metamorphic log");
}

#[test]
fn metamorphic_stats_multiscale_graphcorr() {
    let start = Instant::now();
    let mut cases = Vec::new();

    // (label, x, y, expect_high_stat)
    let fixtures: Vec<(&str, Vec<Vec<f64>>, Vec<Vec<f64>>, bool)> = vec![
        // 1-D linear: y = 2x − strong positive dependence
        (
            "linear_n20",
            (1..=20).map(|i| vec![i as f64]).collect(),
            (1..=20).map(|i| vec![2.0 * i as f64 - 1.0]).collect(),
            true,
        ),
        // 1-D quadratic: non-monotone — MGC should still detect the
        // dependence (better than Pearson)
        (
            "quadratic_n20",
            (-10..=10).map(|i| vec![i as f64]).collect(),
            (-10..=10).map(|i| vec![(i as f64).powi(2)]).collect(),
            true,
        ),
        // 2-D shifted dependence: y = x with small noise
        (
            "two_d_dep_n15",
            (1..=15)
                .map(|i| vec![i as f64, (i as f64).sqrt()])
                .collect(),
            (1..=15)
                .map(|i| vec![i as f64 + 0.3, (i as f64).sqrt() + 0.1])
                .collect(),
            true,
        ),
        // Permuted indices: x's row order has no relationship to y's
        // → low MGC (detects only spurious noise)
        (
            "permuted_n20",
            (1..=20).map(|i| vec![i as f64]).collect(),
            vec![
                vec![5.0],
                vec![19.0],
                vec![3.0],
                vec![17.0],
                vec![1.0],
                vec![15.0],
                vec![11.0],
                vec![7.0],
                vec![13.0],
                vec![9.0],
                vec![20.0],
                vec![6.0],
                vec![18.0],
                vec![4.0],
                vec![16.0],
                vec![2.0],
                vec![14.0],
                vec![10.0],
                vec![8.0],
                vec![12.0],
            ],
            false,
        ),
    ];

    let reps = 50_usize;
    let seed = 42_u64;

    for (name, x, y, expect_high) in &fixtures {
        let r = multiscale_graphcorr(x, y, reps, Some(seed)).expect("MGC");

        // Invariant 1: statistic ∈ [0, 1].
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "statistic_in_0_1".into(),
            detail: format!("statistic={}", r.statistic),
            pass: r.statistic.is_finite() && (0.0..=1.0).contains(&r.statistic),
        });

        // Invariant 2: pvalue ∈ [0, 1].
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "pvalue_in_0_1".into(),
            detail: format!("pvalue={}", r.pvalue),
            pass: r.pvalue.is_finite() && (0.0..=1.0).contains(&r.pvalue),
        });

        // Invariant 3: mgc_map is n × n.
        let n = x.len();
        let map_shape_ok = r.mgc_map.len() == n
            && r.mgc_map.iter().all(|row| row.len() == n);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "mgc_map_n_by_n".into(),
            detail: format!(
                "rows={}, all_cols_n={}, expected n={}",
                r.mgc_map.len(),
                r.mgc_map.iter().all(|row| row.len() == n),
                n
            ),
            pass: map_shape_ok,
        });

        // Invariant 4: opt_scale entries in [1, n].
        let (k, l) = r.opt_scale;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "opt_scale_in_1_n".into(),
            detail: format!("opt_scale=({k}, {l}), n={n}"),
            pass: k >= 1 && k <= n && l >= 1 && l <= n,
        });

        // Invariant 5: dependence-strength expectation.
        if *expect_high {
            cases.push(CaseLog {
                case_id: name.to_string(),
                invariant: "dependent_data_high_statistic".into(),
                detail: format!("statistic={}, expected ≥ 0.5", r.statistic),
                pass: r.statistic >= 0.5,
            });
        }

        // Invariant 7: determinism.
        let r2 = multiscale_graphcorr(x, y, reps, Some(seed)).expect("MGC");
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "deterministic_same_seed".into(),
            detail: format!(
                "stat=({}, {}), pval=({}, {})",
                r.statistic, r2.statistic, r.pvalue, r2.pvalue
            ),
            pass: r.statistic == r2.statistic && r.pvalue == r2.pvalue,
        });
    }

    // Invariant 6: scipy sentinel — pvalue=1 when reps=0.
    let x_simple: Vec<Vec<f64>> = (1..=10).map(|i| vec![i as f64]).collect();
    let y_simple: Vec<Vec<f64>> = (1..=10).map(|i| vec![2.0 * i as f64]).collect();
    let r_zero = multiscale_graphcorr(&x_simple, &y_simple, 0, Some(seed)).expect("MGC reps=0");
    cases.push(CaseLog {
        case_id: "edge_reps_zero".into(),
        invariant: "reps_zero_pvalue_sentinel".into(),
        detail: format!("pvalue={}", r_zero.pvalue),
        pass: (r_zero.pvalue - 1.0).abs() < 1e-15,
    });

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_stats_multiscale_graphcorr".into(),
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
                "multiscale_graphcorr metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "multiscale_graphcorr metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
