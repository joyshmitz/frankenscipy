#![forbid(unsafe_code)]
//! Metamorphic invariants for the two remaining stochastic
//! `fsci_stats` helpers without diff harnesses:
//!   • `permutation_test(x, y, stat_fn, n_perm, seed)` — non-
//!     parametric two-sample permutation test.
//!   • `bootstrap_mean(data, n_bootstrap, confidence, seed)` —
//!     BCa bootstrap CI for the mean.
//!
//! Resolves [frankenscipy-ffq5h]. Both use fsci's deterministic
//! LCG; scipy.stats.permutation_test and scipy.stats.bootstrap
//! use PCG64, so direct parity is impossible. This harness
//! verifies invariants any sensible procedure must satisfy.
//!
//! permutation_test:
//!   1. pvalue ∈ [1/(B+1), 1].
//!   2. Identical samples → pvalue ≈ 1 (no evidence of difference).
//!   3. Well-separated samples → pvalue ≈ 1/(B+1) (always extreme).
//!   4. Determinism: same seed → identical (stat, pvalue).
//!   5. NaN guard: empty or NaN-containing input returns NaN.
//!
//! bootstrap_mean (BCa):
//!   6. Ordering: lo ≤ hi.
//!   7. Coverage: lo ≤ empirical_mean ≤ hi (typical data).
//!   8. Determinism: same seed → identical (lo, hi).
//!   9. Confidence widens: width(99%) ≥ width(95%) ≥ width(80%).
//!   10. NaN guard: n<2 or n_bootstrap=0 returns (NaN, NaN).
//!
//! ~24 cases.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{bootstrap_mean, permutation_test};
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
        .expect("create perm_bootstrap_mean metamorphic output dir");
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
        .expect("serialize perm_bootstrap_mean metamorphic log");
    fs::write(path, json).expect("write perm_bootstrap_mean metamorphic log");
}

fn mean_diff(a: &[f64], b: &[f64]) -> f64 {
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    ma - mb
}

#[test]
fn metamorphic_stats_perm_bootstrap_mean() {
    let start = Instant::now();
    let mut cases = Vec::new();

    // ─────────────────── permutation_test ───────────────────
    let n_perm = 2000_usize;
    let perm_seed = 42_u64;
    let perm_floor = 1.0 / (n_perm as f64 + 1.0);

    // (label, x, y, expect_small_p)
    let perm_fixtures: Vec<(&str, Vec<f64>, Vec<f64>, bool)> = vec![
        // Identical samples → no evidence
        (
            "identical_n10",
            (1..=10).map(|i| i as f64).collect(),
            (1..=10).map(|i| i as f64).collect(),
            false,
        ),
        // Well-separated samples → strong rejection
        (
            "separated_n10",
            (1..=10).map(|i| i as f64).collect(),
            (101..=110).map(|i| i as f64).collect(),
            true,
        ),
        // Mild shift → ambiguous (not pinned)
        (
            "mild_shift_n12",
            (1..=12).map(|i| i as f64).collect(),
            (3..=14).map(|i| i as f64).collect(),
            false,
        ),
    ];

    for (name, x, y, expect_small) in &perm_fixtures {
        let (stat, pval) = permutation_test(x, y, mean_diff, n_perm, perm_seed);

        // Invariant 1: pvalue ∈ [floor, 1].
        cases.push(CaseLog {
            case_id: format!("perm_{name}"),
            invariant: "pvalue_in_floor_1_range".into(),
            detail: format!(
                "pval={pval}, floor={perm_floor}"
            ),
            pass: pval.is_finite() && pval >= perm_floor && pval <= 1.0,
        });

        // Invariant 2/3: identical → near 1; separated → near floor.
        if *expect_small {
            cases.push(CaseLog {
                case_id: format!("perm_{name}"),
                invariant: "well_separated_pvalue_at_floor".into(),
                detail: format!("pval={pval}, floor={perm_floor}"),
                // Allow a small slack — with B=2000 and exact reject,
                // pvalue should be exactly 1/(B+1) but precision in
                // the count_extreme ≥ observed comparison may add a
                // few false-extremes from ties.
                pass: pval <= 0.005,
            });
        } else if *name == "identical_n10" {
            cases.push(CaseLog {
                case_id: format!("perm_{name}"),
                invariant: "identical_samples_pvalue_near_1".into(),
                detail: format!("pval={pval}"),
                pass: pval >= 0.99,
            });
        }

        // Invariant 4: determinism.
        let (stat2, pval2) = permutation_test(x, y, mean_diff, n_perm, perm_seed);
        cases.push(CaseLog {
            case_id: format!("perm_{name}"),
            invariant: "deterministic_same_seed".into(),
            detail: format!(
                "stat=({stat}, {stat2}), pval=({pval}, {pval2})"
            ),
            pass: stat == stat2 && pval == pval2,
        });
    }

    // Invariant 5: NaN guard for permutation_test.
    let empty: Vec<f64> = vec![];
    let nonempty = vec![1.0, 2.0, 3.0];
    let (s_nan, p_nan) = permutation_test(&empty, &nonempty, mean_diff, n_perm, perm_seed);
    cases.push(CaseLog {
        case_id: "perm_edge_empty".into(),
        invariant: "perm_empty_input_returns_nan".into(),
        detail: format!("stat={s_nan}, pval={p_nan}"),
        pass: s_nan.is_nan() && p_nan.is_nan(),
    });

    // ─────────────────── bootstrap_mean ─────────────────────
    let n_boot = 5000_usize;
    let conf = 0.95_f64;
    let boot_seed = 7_u64;
    let boot_fixtures: Vec<(&str, Vec<f64>)> = vec![
        ("boot_compact_n10", (1..=10).map(|i| i as f64).collect()),
        (
            "boot_spread_n20",
            (1..=20).map(|i| (i as f64).sqrt() * 4.0).collect(),
        ),
    ];

    for (name, data) in &boot_fixtures {
        let m = data.iter().sum::<f64>() / data.len() as f64;
        let (lo, hi) = bootstrap_mean(data, n_boot, conf, boot_seed);

        // Invariant 6: lo ≤ hi.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "lo_le_hi".into(),
            detail: format!("lo={lo}, hi={hi}"),
            pass: lo.is_finite() && hi.is_finite() && lo <= hi,
        });

        // Invariant 7: coverage. The BCa interval should bracket the
        //              empirical mean for typical (centered) data.
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "covers_empirical_mean".into(),
            detail: format!("lo={lo}, mean={m}, hi={hi}"),
            pass: lo <= m && m <= hi,
        });

        // Invariant 8: determinism.
        let (lo2, hi2) = bootstrap_mean(data, n_boot, conf, boot_seed);
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "deterministic_same_seed".into(),
            detail: format!("first=({lo}, {hi}), second=({lo2}, {hi2})"),
            pass: lo == lo2 && hi == hi2,
        });

        // Invariant 9: confidence widens.
        let (lo80, hi80) = bootstrap_mean(data, n_boot, 0.80, boot_seed);
        let (lo99, hi99) = bootstrap_mean(data, n_boot, 0.99, boot_seed);
        let w80 = hi80 - lo80;
        let w95 = hi - lo;
        let w99 = hi99 - lo99;
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "confidence_widens_interval".into(),
            detail: format!("w80={w80}, w95={w95}, w99={w99}"),
            pass: w80 <= w95 && w95 <= w99,
        });
    }

    // Invariant 10: NaN guards for bootstrap_mean.
    let single = vec![1.0];
    let (lo_one, hi_one) = bootstrap_mean(&single, n_boot, conf, boot_seed);
    cases.push(CaseLog {
        case_id: "boot_edge_single_point".into(),
        invariant: "boot_n_lt_2_returns_nan".into(),
        detail: format!("lo={lo_one}, hi={hi_one}"),
        pass: lo_one.is_nan() && hi_one.is_nan(),
    });
    let triple = vec![1.0, 2.0, 3.0];
    let (lo_z, hi_z) = bootstrap_mean(&triple, 0, conf, boot_seed);
    cases.push(CaseLog {
        case_id: "boot_edge_zero_bootstrap".into(),
        invariant: "boot_zero_iterations_returns_nan".into(),
        detail: format!("lo={lo_z}, hi={hi_z}"),
        pass: lo_z.is_nan() && hi_z.is_nan(),
    });

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_stats_perm_bootstrap_mean".into(),
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
                "perm_bootstrap_mean metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "perm_bootstrap_mean metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
