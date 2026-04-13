//! Regenerate fixture expected values from oracle capture.
//!
//! This tool reads an oracle_capture.json file and updates the corresponding
//! fixture file with oracle-captured expected values and provenance metadata.
//!
//! Usage:
//!   cargo run --bin fixture_regen -- --fixture <path> --oracle <path> [--dry-run]
//!
//! Example:
//!   cargo run --bin fixture_regen -- \
//!     --fixture crates/fsci-conformance/fixtures/FSCI-P2C-002_linalg_core.json \
//!     --oracle crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-002/oracle_capture.json

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct OracleCapture {
    packet_id: String,
    #[allow(dead_code)]
    family: String,
    #[allow(dead_code)]
    generated_unix_ms: u128,
    runtime: Option<OracleRuntimeInfo>,
    provenance: Option<OracleCaptureProvenance>,
    case_outputs: Vec<OracleCaseOutput>,
}

#[derive(Debug, Deserialize)]
struct OracleRuntimeInfo {
    #[allow(dead_code)]
    python_version: String,
    numpy_version: String,
    scipy_version: String,
}

#[derive(Debug, Deserialize)]
struct OracleCaptureProvenance {
    #[allow(dead_code)]
    fixture_input_blake3: String,
    oracle_output_blake3: String,
    #[allow(dead_code)]
    capture_blake3: String,
}

#[derive(Debug, Deserialize)]
struct OracleCaseOutput {
    case_id: String,
    status: String,
    result_kind: String,
    result: Value,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct FixtureProvenance {
    oracle_hash: String,
    scipy_version: String,
    numpy_version: String,
    regenerated_at_unix_ms: u128,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut fixture_path: Option<PathBuf> = None;
    let mut oracle_path: Option<PathBuf> = None;
    let mut dry_run = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fixture" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--fixture requires a value");
                    std::process::exit(1);
                }
                fixture_path = Some(PathBuf::from(&args[i]));
            }
            "--oracle" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--oracle requires a value");
                    std::process::exit(1);
                }
                oracle_path = Some(PathBuf::from(&args[i]));
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--help" | "-h" => {
                eprintln!("Usage: fixture_regen --fixture <path> --oracle <path> [--dry-run]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --fixture <path>  Path to fixture JSON file");
                eprintln!("  --oracle <path>   Path to oracle_capture.json file");
                eprintln!("  --dry-run         Print changes without modifying the fixture");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let fixture_path = fixture_path.expect("--fixture is required");
    let oracle_path = oracle_path.expect("--oracle is required");

    // Load oracle capture
    let oracle_raw = fs::read_to_string(&oracle_path).expect("read oracle capture");
    let oracle: OracleCapture = serde_json::from_str(&oracle_raw).expect("parse oracle capture");

    // Load fixture as generic JSON
    let fixture_raw = fs::read_to_string(&fixture_path).expect("read fixture");
    let mut fixture: Value = serde_json::from_str(&fixture_raw).expect("parse fixture");

    // Validate packet id matches to avoid corrupting unrelated fixtures.
    let fixture_packet_id = fixture
        .get("packet_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if fixture_packet_id != oracle.packet_id {
        eprintln!(
            "Packet id mismatch: fixture={} oracle={}",
            fixture_packet_id, oracle.packet_id
        );
        std::process::exit(1);
    }

    // Build case_id -> oracle output mapping
    let oracle_map: HashMap<&str, &OracleCaseOutput> = oracle
        .case_outputs
        .iter()
        .map(|o| (o.case_id.as_str(), o))
        .collect();

    // Extract provenance info
    let provenance = oracle.provenance.as_ref();
    let runtime = oracle.runtime.as_ref();

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_millis();

    let fixture_provenance = FixtureProvenance {
        oracle_hash: provenance
            .map(|p| p.oracle_output_blake3.clone())
            .unwrap_or_default(),
        scipy_version: runtime.map(|r| r.scipy_version.clone()).unwrap_or_default(),
        numpy_version: runtime.map(|r| r.numpy_version.clone()).unwrap_or_default(),
        regenerated_at_unix_ms: now_ms,
    };

    // Update fixture cases
    let cases = fixture
        .get_mut("cases")
        .and_then(Value::as_array_mut)
        .expect("fixture.cases array");

    let mut updated_count = 0;
    let mut skipped_count = 0;
    let mut error_count = 0;

    for case in cases.iter_mut() {
        let case_obj = case.as_object_mut().expect("case object");
        let case_id = case_obj
            .get("case_id")
            .and_then(Value::as_str)
            .expect("case_id")
            .to_owned();

        let Some(oracle_output) = oracle_map.get(case_id.as_str()) else {
            eprintln!("  WARN: no oracle output for case_id={case_id}");
            skipped_count += 1;
            continue;
        };

        let expected = case_obj
            .get_mut("expected")
            .and_then(Value::as_object_mut)
            .expect("expected object");

        // Handle based on oracle status
        let expected_kind = expected
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("unknown");

        if oracle_output.status == "ok" {
            if expected_kind == "error" {
                eprintln!(
                    "  WARN {case_id}: oracle succeeded but fixture expects error; skipping"
                );
                skipped_count += 1;
                continue;
            }
            match oracle_output.result_kind.as_str() {
                "vector" => {
                    if let Some(values) = oracle_output.result.get("values") {
                        expected.insert("values".to_owned(), values.clone());
                        add_provenance(expected, &fixture_provenance);
                        updated_count += 1;
                        if dry_run {
                            eprintln!("  UPDATE {case_id}: values from oracle");
                        }
                    }
                }
                "matrix" => {
                    if let Some(values) = oracle_output.result.get("values") {
                        expected.insert("values".to_owned(), values.clone());
                        add_provenance(expected, &fixture_provenance);
                        updated_count += 1;
                        if dry_run {
                            eprintln!("  UPDATE {case_id}: matrix values from oracle");
                        }
                    }
                }
                "scalar" => {
                    if let Some(value) = oracle_output.result.get("value") {
                        expected.insert("value".to_owned(), value.clone());
                        add_provenance(expected, &fixture_provenance);
                        updated_count += 1;
                        if dry_run {
                            eprintln!("  UPDATE {case_id}: scalar from oracle");
                        }
                    }
                }
                "lstsq" => {
                    if let Some(x) = oracle_output.result.get("x") {
                        expected.insert("x".to_owned(), x.clone());
                    }
                    if let Some(residuals) = oracle_output.result.get("residuals") {
                        expected.insert("residuals".to_owned(), residuals.clone());
                    }
                    if let Some(rank) = oracle_output.result.get("rank") {
                        expected.insert("rank".to_owned(), rank.clone());
                    }
                    if let Some(sv) = oracle_output.result.get("singular_values") {
                        expected.insert("singular_values".to_owned(), sv.clone());
                    }
                    add_provenance(expected, &fixture_provenance);
                    updated_count += 1;
                    if dry_run {
                        eprintln!("  UPDATE {case_id}: lstsq result from oracle");
                    }
                }
                "pinv" => {
                    if let Some(values) = oracle_output.result.get("values") {
                        expected.insert("values".to_owned(), values.clone());
                    }
                    if let Some(rank) = oracle_output.result.get("rank") {
                        expected.insert("rank".to_owned(), rank.clone());
                    }
                    add_provenance(expected, &fixture_provenance);
                    updated_count += 1;
                    if dry_run {
                        eprintln!("  UPDATE {case_id}: pinv result from oracle");
                    }
                }
                other => {
                    eprintln!("  SKIP {case_id}: unsupported result_kind={other}");
                    skipped_count += 1;
                }
            }
        } else if oracle_output.status == "error" {
            // Oracle returned error - keep fixture expected.kind="error"
            if expected_kind == "error" {
                // Update error message from oracle if available
                if let Some(err) = &oracle_output.error {
                    expected.insert("error".to_owned(), Value::String(err.clone()));
                    add_provenance(expected, &fixture_provenance);
                    error_count += 1;
                    if dry_run {
                        eprintln!("  UPDATE {case_id}: error message from oracle");
                    }
                }
            } else {
                eprintln!("  WARN {case_id}: oracle returned error but fixture expected non-error");
                skipped_count += 1;
            }
        } else {
            eprintln!(
                "  SKIP {case_id}: unknown oracle status={}",
                oracle_output.status
            );
            skipped_count += 1;
        }
    }

    // Add top-level provenance
    let fixture_obj = fixture.as_object_mut().expect("fixture object");
    fixture_obj.insert(
        "oracle_provenance".to_owned(),
        serde_json::to_value(&fixture_provenance).expect("serialize provenance"),
    );

    eprintln!();
    eprintln!("Summary:");
    eprintln!("  Updated: {updated_count}");
    eprintln!("  Errors:  {error_count}");
    eprintln!("  Skipped: {skipped_count}");
    eprintln!("  Oracle:  {}", oracle.packet_id);
    eprintln!(
        "  SciPy:   {}",
        runtime
            .map(|r| r.scipy_version.as_str())
            .unwrap_or("unknown")
    );

    if dry_run {
        eprintln!();
        eprintln!("Dry run - no changes written.");
    } else {
        let output = serde_json::to_string_pretty(&fixture).expect("serialize fixture");
        fs::write(&fixture_path, output).expect("write fixture");
        eprintln!();
        eprintln!("Fixture updated: {}", fixture_path.display());
    }
}

fn add_provenance(expected: &mut Map<String, Value>, prov: &FixtureProvenance) {
    expected.insert(
        "oracle_hash".to_owned(),
        Value::String(prov.oracle_hash.clone()),
    );
    expected.insert(
        "scipy_version".to_owned(),
        Value::String(prov.scipy_version.clone()),
    );
}
