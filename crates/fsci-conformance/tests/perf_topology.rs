//! Perf topology regression test (frankenscipy-rvox).
//!
//! Verifies that every perf_*.rs file in crates/fsci-conformance/tests/ is
//! included in CI G6's perf gate or explicitly excluded with rationale.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("should find project root from conformance crate")
}

const G6_PERF_TESTS: &[&str] = &[
    "perf_arrayapi",
    "perf_casp",
    "perf_fft",
    "perf_ivp",
    "perf_linalg",
    "perf_optimize",
    "perf_sparse",
    "perf_special",
];

const EXCLUDED_PERF_TESTS: &[&str] = &[
    "perf_topology",
];

#[test]
fn every_perf_test_in_g6_or_excluded() {
    let tests_dir = project_root().join("crates/fsci-conformance/tests");

    if !tests_dir.exists() {
        eprintln!("tests/ directory not found, skipping topology check");
        return;
    }

    let g6_set: HashSet<&str> = G6_PERF_TESTS.iter().copied().collect();
    let excluded_set: HashSet<&str> = EXCLUDED_PERF_TESTS.iter().copied().collect();

    let mut perf_files: Vec<String> = Vec::new();
    for entry in fs::read_dir(&tests_dir).expect("read tests dir") {
        let entry = entry.expect("read entry");
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("perf_") && name.ends_with(".rs") {
            let test_name = name.trim_end_matches(".rs").to_string();
            perf_files.push(test_name);
        }
    }

    let mut missing: Vec<&str> = Vec::new();
    for test_name in &perf_files {
        if !g6_set.contains(test_name.as_str()) && !excluded_set.contains(test_name.as_str()) {
            missing.push(test_name);
        }
    }

    if !missing.is_empty() {
        missing.sort();
        panic!(
            "Perf tests not in CI G6 gate or EXCLUDED_PERF_TESTS:\n  {}\n\n\
             Per bead frankenscipy-rvox, every perf_*.rs file must either be:\n\
             - Listed in G6_PERF_TESTS (runs in CI), or\n\
             - Listed in EXCLUDED_PERF_TESTS with rationale\n\
             Update this file or .github/workflows/ci.yml G6 section.",
            missing.join("\n  ")
        );
    }
}

#[test]
fn g6_tests_all_exist() {
    let tests_dir = project_root().join("crates/fsci-conformance/tests");

    if !tests_dir.exists() {
        eprintln!("tests/ directory not found, skipping existence check");
        return;
    }

    let mut missing: Vec<&str> = Vec::new();
    for test_name in G6_PERF_TESTS {
        let test_path = tests_dir.join(format!("{}.rs", test_name));
        if !test_path.exists() {
            missing.push(test_name);
        }
    }

    if !missing.is_empty() {
        missing.sort();
        panic!(
            "G6_PERF_TESTS references non-existent test files:\n  {}\n\n\
             Either create the test file or remove it from G6_PERF_TESTS.",
            missing.join("\n  ")
        );
    }
}
