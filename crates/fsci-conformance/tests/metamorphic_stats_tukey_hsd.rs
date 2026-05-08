#![forbid(unsafe_code)]
//! Metamorphic invariants for `fsci_stats::tukey_hsd` —
//! pairwise post-hoc multiple-comparison test of group means.
//!
//! Resolves [frankenscipy-f7p0w]. fsci approximates Tukey's
//! HSD via Bonferroni-corrected t-tests; scipy.stats.tukey_hsd
//! uses the actual studentized range distribution. The two
//! pvalue paths can't agree numerically, so this harness
//! verifies normalisation-invariant METAMORPHIC properties
//! that any sensible pairwise post-hoc must satisfy:
//!
//!   1. Statistic matrix is anti-symmetric: stat[i][j] =
//!      −stat[j][i] (it's the mean difference m_i − m_j).
//!   2. Statistic diagonal is zero: stat[i][i] = 0.
//!   3. Pvalue matrix is symmetric: pvalue[i][j] = pvalue[j][i]
//!      (significance of "i ≠ j" doesn't depend on direction).
//!   4. Pvalue diagonal is 1.0: a group is identical to itself.
//!   5. Pvalues bounded [0, 1].
//!   6. Identical groups (same data) produce pvalue = 1 across
//!      all off-diagonal pairs.
//!   7. Strongly separated groups produce small pvalue (< 0.05)
//!      for the separated pair.
//!
//! 4 fixtures × variable arms = ~28 cases.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::tukey_hsd;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ANTI_SYMM_TOL: f64 = 1.0e-12;
const SYMM_TOL: f64 = 1.0e-12;

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
    fs::create_dir_all(output_dir()).expect("create tukey_hsd metamorphic output dir");
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
        serde_json::to_string_pretty(log).expect("serialize tukey_hsd metamorphic log");
    fs::write(path, json).expect("write tukey_hsd metamorphic log");
}

#[test]
fn metamorphic_stats_tukey_hsd() {
    let start = Instant::now();
    let mut cases = Vec::new();

    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // Three groups, distinct means
        (
            "three_distinct_means",
            vec![
                (1..=10).map(|i| i as f64).collect(),
                (11..=20).map(|i| i as f64).collect(),
                (21..=30).map(|i| i as f64).collect(),
            ],
        ),
        // Four groups, mixed
        (
            "four_mixed",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                vec![20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            ],
        ),
        // Three IDENTICAL groups — same data each
        (
            "three_identical",
            {
                let g: Vec<f64> = (1..=8).map(|i| i as f64).collect();
                vec![g.clone(), g.clone(), g]
            },
        ),
        // Two well-separated groups
        (
            "two_separated",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            ],
        ),
    ];

    for (name, groups) in &fixtures {
        let group_refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
        let result = tukey_hsd(&group_refs);
        let k = groups.len();

        // Invariant 1: stat is anti-symmetric.
        let mut max_anti_symm_diff = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                let antisym_err = (result.statistic[i][j] + result.statistic[j][i]).abs();
                max_anti_symm_diff = max_anti_symm_diff.max(antisym_err);
            }
        }
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "stat_antisymmetric".into(),
            detail: format!("max |s[i,j] + s[j,i]| = {max_anti_symm_diff}"),
            pass: max_anti_symm_diff <= ANTI_SYMM_TOL,
        });

        // Invariant 2: stat diagonal is zero.
        let mut diag_max = 0.0_f64;
        for i in 0..k {
            diag_max = diag_max.max(result.statistic[i][i].abs());
        }
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "stat_diagonal_zero".into(),
            detail: format!("max |s[i,i]| = {diag_max}"),
            pass: diag_max <= ANTI_SYMM_TOL,
        });

        // Invariant 3: pvalue is symmetric.
        let mut max_symm_diff = 0.0_f64;
        for i in 0..k {
            for j in 0..k {
                let diff = (result.pvalue[i][j] - result.pvalue[j][i]).abs();
                max_symm_diff = max_symm_diff.max(diff);
            }
        }
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "pvalue_symmetric".into(),
            detail: format!("max |p[i,j] - p[j,i]| = {max_symm_diff}"),
            pass: max_symm_diff <= SYMM_TOL,
        });

        // Invariant 4: pvalue diagonal is 1.0.
        let mut diag_p_max_diff = 0.0_f64;
        for i in 0..k {
            diag_p_max_diff = diag_p_max_diff.max((result.pvalue[i][i] - 1.0).abs());
        }
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "pvalue_diagonal_one".into(),
            detail: format!("max |p[i,i] - 1| = {diag_p_max_diff}"),
            pass: diag_p_max_diff <= 1.0e-15,
        });

        // Invariant 5: pvalues bounded [0, 1].
        let mut bounded_pass = true;
        let mut bound_violation = String::new();
        for i in 0..k {
            for j in 0..k {
                let p = result.pvalue[i][j];
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    bounded_pass = false;
                    bound_violation = format!("p[{i},{j}] = {p}");
                    break;
                }
            }
            if !bounded_pass {
                break;
            }
        }
        cases.push(CaseLog {
            case_id: name.to_string(),
            invariant: "pvalues_bounded_0_1".into(),
            detail: if bounded_pass {
                "all in [0, 1]".into()
            } else {
                bound_violation
            },
            pass: bounded_pass,
        });

        // Invariant 6 (identical-groups fixture only): all off-diagonal
        // pvalues should be ≈ 1.
        if *name == "three_identical" {
            let mut min_p = 1.0_f64;
            for i in 0..k {
                for j in 0..k {
                    if i != j {
                        min_p = min_p.min(result.pvalue[i][j]);
                    }
                }
            }
            cases.push(CaseLog {
                case_id: name.to_string(),
                invariant: "identical_groups_pvalue_near_1".into(),
                detail: format!("min off-diagonal p = {min_p}"),
                pass: min_p > 0.99,
            });
        }

        // Invariant 7 (separated-groups fixture only): the off-diagonal
        // pvalue should be very small.
        if *name == "two_separated" {
            let p_off = result.pvalue[0][1];
            cases.push(CaseLog {
                case_id: name.to_string(),
                invariant: "separated_groups_pvalue_small".into(),
                detail: format!("p[0][1] = {p_off}"),
                pass: p_off < 0.05,
            });
        }
    }

    let pass_count = cases.iter().filter(|c| c.pass).count();
    let all_pass = pass_count == cases.len();

    let log = MetamorphicLog {
        test_id: "metamorphic_stats_tukey_hsd".into(),
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
                "tukey_hsd metamorphic fail: {} {} — {}",
                c.case_id, c.invariant, c.detail
            );
        }
    }

    assert!(
        all_pass,
        "tukey_hsd metamorphic failed: {} of {} cases pass",
        pass_count,
        cases.len()
    );
}
