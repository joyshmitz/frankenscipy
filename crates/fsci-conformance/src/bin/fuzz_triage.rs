#![forbid(unsafe_code)]

use blake3::hash;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct Config {
    input_dir: PathBuf,
    output_dir: PathBuf,
    regression_dir: PathBuf,
    retention_days: u64,
    file_beads: bool,
}

#[derive(Debug, Clone, Serialize)]
struct FindingReport {
    finding_id: String,
    stack_hash: String,
    classification: String,
    source_files: Vec<String>,
    promoted_fixture: String,
    created_at_epoch_seconds: u64,
    retention_days: u64,
    bead_id: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct TriageIndex {
    #[serde(default)]
    bead_by_stack_hash: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct TriageSummary {
    schema_version: u32,
    generated_at_epoch_seconds: u64,
    input_dir: String,
    output_dir: String,
    regression_dir: String,
    unique_findings: usize,
    source_files_processed: usize,
    reports: Vec<FindingReport>,
}

fn now_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn default_config() -> Config {
    let root = repo_root();
    Config {
        input_dir: root.join("fuzz/artifacts"),
        output_dir: root.join("crates/fsci-conformance/fixtures/adversarial/triage"),
        regression_dir: root.join("crates/fsci-conformance/fixtures/regressions/fuzz"),
        retention_days: 90,
        file_beads: true,
    }
}

fn parse_args() -> Config {
    let mut config = default_config();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input-dir" => {
                if let Some(path) = args.next() {
                    config.input_dir = PathBuf::from(path);
                    continue;
                }
                eprintln!("missing value for --input-dir");
                std::process::exit(2);
            }
            "--output-dir" => {
                if let Some(path) = args.next() {
                    config.output_dir = PathBuf::from(path);
                    continue;
                }
                eprintln!("missing value for --output-dir");
                std::process::exit(2);
            }
            "--regression-dir" => {
                if let Some(path) = args.next() {
                    config.regression_dir = PathBuf::from(path);
                    continue;
                }
                eprintln!("missing value for --regression-dir");
                std::process::exit(2);
            }
            "--retention-days" => {
                if let Some(days) = args.next() {
                    match days.parse::<u64>() {
                        Ok(value) => config.retention_days = value,
                        Err(_) => {
                            eprintln!("invalid value for --retention-days: {days}");
                            std::process::exit(2);
                        }
                    }
                    continue;
                }
                eprintln!("missing value for --retention-days");
                std::process::exit(2);
            }
            "--no-file-beads" => {
                config.file_beads = false;
            }
            "--file-beads" => {
                config.file_beads = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("unrecognized argument: {flag}");
                print_help();
                std::process::exit(2);
            }
        }
    }
    config
}

fn print_help() {
    println!(
        "Usage: cargo run -p fsci-conformance --bin fuzz_triage -- [OPTIONS]

Options:
  --input-dir <PATH>       Source crash artifacts directory (default: fuzz/artifacts)
  --output-dir <PATH>      Triage report directory
  --regression-dir <PATH>  Regression fixture promotion directory
  --retention-days <N>     Retention horizon for raw crashes (default: 90)
  --file-beads             Auto-file beads labeled fuzz-finding (default: on)
  --no-file-beads          Disable automatic bead filing"
    );
}

fn looks_like_crash_file(path: &Path) -> bool {
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
    name.starts_with("crash-") || name.starts_with("oom-") || name.starts_with("timeout-")
}

fn walk_crash_files(root: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_crash_files(&path, files)?;
            continue;
        }
        if looks_like_crash_file(&path) {
            files.push(path);
        }
    }
    Ok(())
}

fn signature_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut picked = Vec::new();
    for line in text.lines() {
        if line.contains("SUMMARY:")
            || line.contains("panicked at")
            || line.contains("thread '")
            || line.contains("stack")
        {
            picked.push(line.trim().to_owned());
        }
        if picked.len() >= 8 {
            break;
        }
    }
    if picked.is_empty() {
        picked.extend(text.lines().take(8).map(|line| line.trim().to_owned()));
    }
    if picked.is_empty() {
        return "empty-crash-artifact".to_owned();
    }
    picked.join("\n")
}

fn classification_for(bytes: &[u8]) -> &'static str {
    let lower = String::from_utf8_lossy(bytes).to_ascii_lowercase();
    if lower.contains("nan") || lower.contains("inf") || lower.contains("non-finite") {
        return "non_finite_input";
    }
    if lower.contains("shape") || lower.contains("dimension") || lower.contains("mismatch") {
        return "shape_mismatch";
    }
    if lower.contains("overflow") || lower.contains("oom") || lower.contains("allocation") {
        return "resource_exhaustion";
    }
    if lower.contains("compat") || lower.contains("version") {
        return "compatibility_drift";
    }
    "unknown"
}

fn load_index(path: &Path) -> TriageIndex {
    let Ok(raw) = fs::read_to_string(path) else {
        return TriageIndex::default();
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

fn save_index(path: &Path, index: &TriageIndex) -> io::Result<()> {
    let payload =
        serde_json::to_string_pretty(index).map_err(|error| io::Error::other(error.to_string()))?;
    fs::write(path, payload)
}

fn file_fuzz_finding_bead(
    stack_hash: &str,
    classification: &str,
    sources: usize,
) -> Option<String> {
    let title = format!("[FUZZ-FINDING] {classification} {stack_hash}");
    let description = format!(
        "Auto-filed by fuzz triage.\n\nstack_hash: {stack_hash}\nclassification: {classification}\nsource_crashes: {sources}\n"
    );
    let output = Command::new("br")
        .args([
            "create",
            "--title",
            &title,
            "--type",
            "bug",
            "--priority",
            "1",
            "--labels",
            "fuzz-finding,security",
            "--description",
            &description,
            "--json",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = serde_json::from_str::<serde_json::Value>(&stdout).ok()?;
    parsed
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args();
    fs::create_dir_all(&config.output_dir)?;
    fs::create_dir_all(&config.regression_dir)?;

    let mut files = Vec::new();
    walk_crash_files(&config.input_dir, &mut files)?;

    let mut grouped: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut bytes_by_hash: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for file in &files {
        let bytes = fs::read(file)?;
        let signature = signature_text(&bytes);
        let stack_hash = hash(signature.as_bytes()).to_hex().to_string();
        grouped
            .entry(stack_hash.clone())
            .or_default()
            .push(file.clone());
        bytes_by_hash.entry(stack_hash).or_insert(bytes);
    }

    let index_path = config.output_dir.join("triage_index.json");
    let mut index = load_index(&index_path);

    let now = now_epoch_seconds();
    let mut reports = Vec::new();
    for (stack_hash, group) in grouped {
        let sample_bytes = bytes_by_hash.get(&stack_hash).cloned().unwrap_or_default();
        let classification = classification_for(&sample_bytes).to_owned();
        let source_files = group
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        let finding_id = format!("finding-{stack_hash}");

        let fixture_path = config.regression_dir.join(format!("{stack_hash}.bin"));
        if let Some(primary) = group.first() {
            let _ = fs::copy(primary, &fixture_path);
        }

        let bead_id = if let Some(existing) = index.bead_by_stack_hash.get(&stack_hash) {
            Some(existing.clone())
        } else if config.file_beads {
            file_fuzz_finding_bead(&stack_hash, &classification, source_files.len())
        } else {
            None
        };
        if let Some(issue_id) = bead_id.clone() {
            index
                .bead_by_stack_hash
                .insert(stack_hash.clone(), issue_id);
        }

        let report = FindingReport {
            finding_id: finding_id.clone(),
            stack_hash: stack_hash.clone(),
            classification,
            source_files,
            promoted_fixture: fixture_path.to_string_lossy().to_string(),
            created_at_epoch_seconds: now,
            retention_days: config.retention_days,
            bead_id,
        };
        let report_path = config.output_dir.join(format!("{finding_id}.json"));
        let report_payload = serde_json::to_string_pretty(&report)?;
        fs::write(report_path, report_payload)?;
        reports.push(report);
    }

    save_index(&index_path, &index)?;

    let summary = TriageSummary {
        schema_version: 1,
        generated_at_epoch_seconds: now,
        input_dir: config.input_dir.to_string_lossy().to_string(),
        output_dir: config.output_dir.to_string_lossy().to_string(),
        regression_dir: config.regression_dir.to_string_lossy().to_string(),
        unique_findings: reports.len(),
        source_files_processed: files.len(),
        reports,
    };
    let summary_path = config.output_dir.join("summary.json");
    fs::write(summary_path, serde_json::to_string_pretty(&summary)?)?;

    println!(
        "triaged {} crash artifact(s) into {} unique finding(s)",
        summary.source_files_processed, summary.unique_findings
    );
    Ok(())
}
