#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const RCH_ENV_FILE: &str = ".rch.env";

fn value_truthy(value: &str) -> bool {
    matches!(
        value.trim(),
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
    )
}

fn env_truthy(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| value_truthy(&value))
}

fn workspace_rch_env_path() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR")?);
    Some(manifest_dir.parent()?.parent()?.join(RCH_ENV_FILE))
}

fn assignment_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let assignment = trimmed.strip_prefix("export ").unwrap_or(trimmed);
    let (key, value) = assignment.split_once('=')?;
    (key.trim() == name).then_some(value.trim().trim_matches(['"', '\'']))
}

fn env_file_truthy(path: &Path, name: &str) -> bool {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return false;
    };
    contents
        .lines()
        .filter_map(|line| assignment_value(line, name))
        .next_back()
        .is_some_and(value_truthy)
}

fn require_scipy_oracle() -> bool {
    env_truthy(REQUIRE_SCIPY_ENV).unwrap_or_else(|| {
        workspace_rch_env_path()
            .as_deref()
            .is_some_and(|path| env_file_truthy(path, REQUIRE_SCIPY_ENV))
    })
}

fn main() {
    println!("cargo:rerun-if-env-changed={REQUIRE_SCIPY_ENV}");
    println!("cargo:rerun-if-env-changed=RCH_ENV_ALLOWLIST");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");
    if let Some(path) = workspace_rch_env_path() {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    if require_scipy_oracle() {
        println!("cargo:rustc-env={REQUIRE_SCIPY_ENV}=1");
    }
}
