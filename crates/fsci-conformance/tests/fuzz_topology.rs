//! Fuzz topology regression test (frankenscipy-ff7t).
//!
//! Verifies that every fuzz target in fuzz/fuzz_targets/ has a corresponding
//! seed directory in fuzz/seeds/ with at least one seed file. This catches
//! cases where new fuzz targets are added without initial corpus seeds.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("should find project root from conformance crate")
}

#[test]
fn every_fuzz_target_has_seed_directory() {
    let fuzz_targets_dir = project_root().join("fuzz/fuzz_targets");
    let fuzz_seeds_dir = project_root().join("fuzz/seeds");

    if !fuzz_targets_dir.exists() {
        eprintln!("fuzz/fuzz_targets/ not found, skipping topology check");
        return;
    }

    let mut targets: HashSet<String> = HashSet::new();
    for entry in fs::read_dir(&fuzz_targets_dir).expect("read fuzz_targets dir") {
        let entry = entry.expect("read entry");
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".rs") {
            let target_name = name.trim_end_matches(".rs").to_string();
            targets.insert(target_name);
        }
    }

    let mut seeds: HashSet<String> = HashSet::new();
    if fuzz_seeds_dir.exists() {
        for entry in fs::read_dir(&fuzz_seeds_dir).expect("read seeds dir") {
            let entry = entry.expect("read entry");
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().to_string();
                seeds.insert(name);
            }
        }
    }

    let mut missing_seeds: Vec<&str> = Vec::new();
    for target in &targets {
        if !seeds.contains(target) {
            missing_seeds.push(target);
        }
    }

    if !missing_seeds.is_empty() {
        missing_seeds.sort();
        panic!(
            "Fuzz targets missing seed directories:\n  {}\n\n\
             Per /testing-fuzzing rule, every fuzz target needs a seeds directory \
             with at least minimal/boundary/adversarial seeds.\n\
             Add: fuzz/seeds/<target>/ with representative seed files.",
            missing_seeds.join("\n  ")
        );
    }
}

#[test]
fn seed_directories_are_not_empty() {
    let fuzz_seeds_dir = project_root().join("fuzz/seeds");

    if !fuzz_seeds_dir.exists() {
        eprintln!("fuzz/seeds/ not found, skipping emptiness check");
        return;
    }

    let mut empty_dirs: Vec<String> = Vec::new();
    for entry in fs::read_dir(&fuzz_seeds_dir).expect("read seeds dir") {
        let entry = entry.expect("read entry");
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let seed_count = fs::read_dir(entry.path())
                .map(|d| d.filter(|e| e.is_ok()).count())
                .unwrap_or(0);
            if seed_count == 0 {
                empty_dirs.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }

    if !empty_dirs.is_empty() {
        empty_dirs.sort();
        panic!(
            "Seed directories with no seed files:\n  {}\n\n\
             Each seed directory should have at least one seed file.",
            empty_dirs.join("\n  ")
        );
    }
}
