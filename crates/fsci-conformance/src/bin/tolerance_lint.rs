//! Lint conformance fixture cases against `artifacts/TOLERANCE_POLICY.md`.
//!
//! For each FSCI-P2C-*.json fixture, walks every case and compares its
//! `(rtol, atol)` against the per-packet baseline tier from §2 of the policy.
//! Cases looser than baseline that lack a `rationale` field are violations.
//!
//! Usage:
//!     tolerance_lint                       # default fixture dir
//!     tolerance_lint --fixtures-dir DIR    # override fixture dir
//!     tolerance_lint --json                # emit JSON report
//!     tolerance_lint --max-violations N    # exit 1 if violations exceed N (default 0 = strict)
//!     tolerance_lint --baseline N          # exit 1 if violations regress past starting point N
//!
//! Exit codes:
//!     0 — within budget
//!     1 — violation count over the configured threshold
//!     2 — IO/parse error

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::Value;

/// Per-packet baseline tolerance tier (rtol). Cases looser than this require
/// an explicit `rationale` field per `artifacts/TOLERANCE_POLICY.md` §2.
/// `None` means the packet has no numeric baseline (structural / Tnone).
fn packet_baseline_rtol(packet: &str) -> Option<f64> {
    match packet {
        "FSCI-P2C-001" => None,
        "FSCI-P2C-002" => Some(1e-12),
        "FSCI-P2C-003" => None,
        "FSCI-P2C-004" => Some(1e-10),
        "FSCI-P2C-005" => Some(1e-10),
        "FSCI-P2C-006" => Some(1e-12),
        "FSCI-P2C-007" => None,
        "FSCI-P2C-008" => None,
        "FSCI-P2C-009" => Some(1e-10),
        "FSCI-P2C-010" => Some(1e-10),
        "FSCI-P2C-011" => Some(1e-10),
        "FSCI-P2C-012" => Some(1e-10),
        "FSCI-P2C-013" => Some(1e-10),
        "FSCI-P2C-014" => Some(1e-12),
        "FSCI-P2C-015" => Some(1e-12),
        "FSCI-P2C-016" => Some(1e-15),
        "FSCI-P2C-017" => Some(1e-12),
        _ => None,
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct Violation {
    packet: String,
    case_id: String,
    rtol: f64,
    atol: f64,
    baseline_rtol: f64,
    multiple: f64,
}

fn extract_case_tolerance(case: &Value) -> Option<(f64, f64, Option<String>)> {
    let case_obj = case.as_object()?;
    let expected = case_obj
        .get("expected")
        .and_then(|v| v.as_object());
    let rtol = expected
        .and_then(|e| e.get("rtol"))
        .or_else(|| case_obj.get("rtol"))
        .and_then(|v| v.as_f64())?;
    let atol = expected
        .and_then(|e| e.get("atol"))
        .or_else(|| case_obj.get("atol"))
        .and_then(|v| v.as_f64())?;
    let rationale = case_obj
        .get("rationale")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .or_else(|| {
            expected
                .and_then(|e| e.get("rationale"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
        });
    Some((rtol, atol, rationale))
}

fn lint_fixture(path: &PathBuf) -> Result<Vec<Violation>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: Value = serde_json::from_str(&content)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    let cases = parsed
        .as_object()
        .and_then(|o| o.get("cases"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let packet = stem.split('_').next().unwrap_or("").to_owned();
    let Some(baseline) = packet_baseline_rtol(&packet) else {
        return Ok(Vec::new());
    };

    let mut violations = Vec::new();
    for case in &cases {
        let Some((rtol, atol, rationale)) = extract_case_tolerance(case) else {
            continue;
        };
        if rationale.is_some() {
            continue;
        }
        if rtol > baseline * 1.001 {
            let case_id = case
                .as_object()
                .and_then(|o| o.get("case_id").or_else(|| o.get("operation")))
                .and_then(|v| v.as_str())
                .unwrap_or("(unnamed)")
                .to_owned();
            let multiple = if baseline > 0.0 { rtol / baseline } else { f64::INFINITY };
            violations.push(Violation {
                packet: packet.clone(),
                case_id,
                rtol,
                atol,
                baseline_rtol: baseline,
                multiple,
            });
        }
    }
    Ok(violations)
}

fn print_text_report(violations: &[Violation]) {
    let mut by_packet: std::collections::BTreeMap<String, Vec<&Violation>> =
        std::collections::BTreeMap::new();
    for v in violations {
        by_packet.entry(v.packet.clone()).or_default().push(v);
    }
    println!("Tolerance Lint Report");
    println!("=====================");
    println!("Reference: artifacts/TOLERANCE_POLICY.md §2 baseline tiers");
    println!();
    for (packet, items) in &by_packet {
        println!("--- {packet} ({} violations) ---", items.len());
        for v in items {
            println!(
                "  {:50} rtol={:.0e} (×{:>6.1} baseline {:.0e})",
                v.case_id, v.rtol, v.multiple, v.baseline_rtol
            );
        }
        println!();
    }
    println!("Total: {} violations across {} packets", violations.len(), by_packet.len());
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: tolerance_lint [OPTIONS]");
        println!();
        println!("Options:");
        println!(
            "  --fixtures-dir DIR     Override fixture directory (default: crates/fsci-conformance/fixtures)"
        );
        println!("  --json                 Emit JSON instead of text report");
        println!("  --max-violations N     Pass if violation count <= N (default 0)");
        println!("  --baseline N           Pass if violation count <= N (alias for --max-violations)");
        println!("  -h, --help             Show this help");
        return ExitCode::SUCCESS;
    }

    let fixtures_dir = args
        .iter()
        .position(|a| a == "--fixtures-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/fsci-conformance/fixtures"));
    let emit_json = args.iter().any(|a| a == "--json");
    let max_violations = args
        .iter()
        .position(|a| a == "--max-violations" || a == "--baseline")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    let entries = match fs::read_dir(&fixtures_dir) {
        Ok(it) => it,
        Err(e) => {
            eprintln!("error: read {}: {e}", fixtures_dir.display());
            return ExitCode::from(2);
        }
    };

    let mut all = Vec::new();
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("FSCI-P2C-") && n.ends_with(".json"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    for path in &paths {
        match lint_fixture(path) {
            Ok(vs) => all.extend(vs),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    }

    if emit_json {
        let payload = serde_json::json!({
            "total_violations": all.len(),
            "max_violations": max_violations,
            "violations": all,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap_or_default());
    } else {
        print_text_report(&all);
    }

    if all.len() > max_violations {
        eprintln!(
            "FAIL: {} violations > {} threshold",
            all.len(),
            max_violations
        );
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
