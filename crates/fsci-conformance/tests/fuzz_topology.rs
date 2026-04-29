//! Fuzz topology regression test (frankenscipy-ff7t).
//!
//! Verifies that every executable fuzz target declared in fuzz/Cargo.toml has a
//! corresponding seed directory in fuzz/seeds/ with at least one seed file.
//! This catches cases where new fuzz targets are added without initial corpus
//! seeds, or where a target file is present but not wired into cargo-fuzz.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const FUZZ_HELPER_MODULES: &[(&str, &str)] = &[(
    "p2c007_stats_common",
    "shared arbitrary value helpers imported by the p2c007 stats fuzz targets",
)];

#[derive(Debug)]
struct FuzzBin {
    name: String,
    path: String,
}

fn project_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = manifest_dir.parent().and_then(|p| p.parent()) {
        root.to_path_buf()
    } else {
        manifest_dir
    }
}

fn fuzz_target_files() -> TestResult<HashSet<String>> {
    let fuzz_targets_dir = project_root().join("fuzz/fuzz_targets");
    let mut targets = HashSet::new();

    if !fuzz_targets_dir.exists() {
        eprintln!("fuzz/fuzz_targets/ not found, skipping topology check");
        return Ok(targets);
    }

    for entry in fs::read_dir(&fuzz_targets_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".rs") {
            targets.insert(name.trim_end_matches(".rs").to_string());
        }
    }

    Ok(targets)
}

fn fuzz_seed_directories() -> TestResult<HashSet<String>> {
    let fuzz_seeds_dir = project_root().join("fuzz/seeds");
    let mut seeds = HashSet::new();

    if !fuzz_seeds_dir.exists() {
        return Ok(seeds);
    }

    for entry in fs::read_dir(&fuzz_seeds_dir)? {
        let entry = entry?;
        if entry.file_type().map(|file_type| file_type.is_dir())? {
            seeds.insert(entry.file_name().to_string_lossy().to_string());
        }
    }

    Ok(seeds)
}

fn manifest_fuzz_bins() -> TestResult<Vec<FuzzBin>> {
    let manifest_path = project_root().join("fuzz/Cargo.toml");
    let manifest = fs::read_to_string(&manifest_path)?;
    let manifest: toml::Value = toml::from_str(&manifest)?;
    let Some(bin_entries) = manifest.get("bin").and_then(toml::Value::as_array) else {
        return Err(format!(
            "{} does not define any [[bin]] fuzz targets",
            manifest_path.display()
        )
        .into());
    };

    let mut bins = Vec::new();
    for bin in bin_entries {
        let Some(name) = bin.get("name").and_then(toml::Value::as_str) else {
            return Err(format!(
                "{} contains a [[bin]] without name",
                manifest_path.display()
            )
            .into());
        };
        let Some(path) = bin.get("path").and_then(toml::Value::as_str) else {
            return Err(format!(
                "{} [[bin]] {} does not declare path",
                manifest_path.display(),
                name
            )
            .into());
        };
        if path.starts_with("fuzz_targets/") && path.ends_with(".rs") {
            bins.push(FuzzBin {
                name: name.to_string(),
                path: path.to_string(),
            });
        }
    }

    Ok(bins)
}

#[test]
fn every_declared_fuzz_target_has_seed_directory() -> TestResult {
    let targets: HashSet<String> = manifest_fuzz_bins()?
        .into_iter()
        .map(|bin| bin.name)
        .collect();
    let seeds = fuzz_seed_directories()?;

    let mut missing_seeds: Vec<&str> = Vec::new();
    for target in &targets {
        if !seeds.contains(target) {
            missing_seeds.push(target);
        }
    }

    if !missing_seeds.is_empty() {
        missing_seeds.sort();
        return Err(format!(
            "Fuzz targets missing seed directories:\n  {}\n\n\
             Per /testing-fuzzing rule, every fuzz target needs a seeds directory \
             with at least minimal/boundary/adversarial seeds.\n\
             Add: fuzz/seeds/<target>/ with representative seed files.",
            missing_seeds.join("\n  "),
        )
        .into());
    }

    Ok(())
}

#[test]
fn every_declared_fuzz_target_path_exists() -> TestResult {
    let root = project_root();
    let mut missing: Vec<String> = Vec::new();

    for bin in manifest_fuzz_bins()? {
        let expected_path = format!("fuzz_targets/{}.rs", bin.name);
        if bin.path != expected_path {
            missing.push(format!(
                "{} declares path {}, expected {}",
                bin.name, bin.path, expected_path
            ));
            continue;
        }

        if !root.join("fuzz").join(&bin.path).exists() {
            missing.push(format!("{} declares missing {}", bin.name, bin.path));
        }
    }

    if !missing.is_empty() {
        missing.sort();
        return Err(format!(
            "fuzz/Cargo.toml declares invalid fuzz target paths:\n  {}\n\n\
             Each [[bin]] should map name = <target> to \
             path = \"fuzz_targets/<target>.rs\".",
            missing.join("\n  ")
        )
        .into());
    }

    Ok(())
}

#[test]
fn every_fuzz_target_file_is_declared_or_helper() -> TestResult {
    let declared: HashSet<String> = manifest_fuzz_bins()?
        .into_iter()
        .map(|bin| bin.name)
        .collect();
    let helpers: HashSet<&str> = FUZZ_HELPER_MODULES.iter().map(|(name, _)| *name).collect();
    let mut missing: Vec<&str> = Vec::new();

    let target_files = fuzz_target_files()?;
    for target in &target_files {
        if !declared.contains(target) && !helpers.contains(target.as_str()) {
            missing.push(target);
        }
    }

    if !missing.is_empty() {
        missing.sort();
        return Err(format!(
            "Fuzz target files are not declared in fuzz/Cargo.toml or FUZZ_HELPER_MODULES:\n  {}\n\n\
             Add a [[bin]] entry so cargo-fuzz can build the target, or document \
             the file as a shared helper module in FUZZ_HELPER_MODULES.",
            missing.join("\n  ")
        )
        .into());
    }

    Ok(())
}

#[test]
fn seed_directories_are_not_empty() -> TestResult {
    let fuzz_seeds_dir = project_root().join("fuzz/seeds");

    if !fuzz_seeds_dir.exists() {
        eprintln!("fuzz/seeds/ not found, skipping emptiness check");
        return Ok(());
    }

    let mut empty_dirs: Vec<String> = Vec::new();
    for entry in fs::read_dir(&fuzz_seeds_dir)? {
        let entry = entry?;
        if entry.file_type().map(|file_type| file_type.is_dir())? {
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
        return Err(format!(
            "Seed directories with no seed files:\n  {}\n\n\
             Each seed directory should have at least one seed file.",
            empty_dirs.join("\n  ")
        )
        .into());
    }

    Ok(())
}
