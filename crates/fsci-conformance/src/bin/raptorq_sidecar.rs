//! Generate RaptorQ sidecars for arbitrary long-lived artifacts.
//!
//! SPEC §6 / §9 require RaptorQ-everywhere durability for every long-lived
//! artifact family, not just conformance fixture parity reports. This bin
//! reuses `generate_raptorq_sidecar` from the conformance crate over any
//! file path the caller hands it, writing `<path>.raptorq.json` and
//! `<path>.decode_proof.json` next to it.
//!
//! Usage:
//!     raptorq_sidecar `<file>` [`<file>` ...]
//!     raptorq_sidecar --verify `<file>` [`<file>` ...]
//!     raptorq_sidecar --verify --scrub-report `<out.json>` `<file>` [`<file>` ...]
//!
//! Exit codes:
//!     0 — all sidecars generated/verified
//!     1 — verification mismatch
//!     2 — IO/argument error

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_conformance::{
    DecodeProofArtifact, RaptorQSidecar, generate_decode_proof_artifact, generate_raptorq_sidecar,
};

#[derive(Debug, Clone)]
struct Cli {
    verify_mode: bool,
    scrub_report: Option<PathBuf>,
    targets: Vec<PathBuf>,
}

#[derive(Debug, serde::Serialize)]
struct ScrubReport {
    schema_version: u8,
    generated_unix_ms: u128,
    checked_artifacts: usize,
    entries: Vec<ScrubEntry>,
}

#[derive(Debug, serde::Serialize)]
struct ScrubEntry {
    artifact_path: String,
    sidecar_path: String,
    decode_proof_path: String,
    source_hash: String,
    source_symbols: usize,
    repair_symbols: usize,
    recovered_blocks: usize,
    status: String,
}

struct GeneratedArtifacts {
    sidecar_path: PathBuf,
    decode_proof_path: PathBuf,
}

fn sidecar_path_for(path: &Path) -> PathBuf {
    let mut out = path.as_os_str().to_owned();
    out.push(".raptorq.json");
    PathBuf::from(out)
}

fn decode_proof_path_for(path: &Path) -> PathBuf {
    let mut out = path.as_os_str().to_owned();
    out.push(".decode_proof.json");
    PathBuf::from(out)
}

fn write_artifacts(target: &Path) -> Result<GeneratedArtifacts, String> {
    let payload = fs::read(target).map_err(|e| format!("read {}: {e}", target.display()))?;
    let sidecar = generate_raptorq_sidecar(&payload).map_err(|e| format!("encode: {e}"))?;
    let decode_proof = generate_decode_proof_artifact(&payload, &sidecar)
        .map_err(|e| format!("decode proof: {e}"))?;

    let serialized = serde_json::to_vec_pretty(&sidecar).map_err(|e| format!("serialize: {e}"))?;
    let sidecar_path = sidecar_path_for(target);
    fs::write(&sidecar_path, &serialized)
        .map_err(|e| format!("write {}: {e}", sidecar_path.display()))?;

    let serialized_decode =
        serde_json::to_vec_pretty(&decode_proof).map_err(|e| format!("serialize: {e}"))?;
    let decode_proof_path = decode_proof_path_for(target);
    fs::write(&decode_proof_path, &serialized_decode)
        .map_err(|e| format!("write {}: {e}", decode_proof_path.display()))?;

    Ok(GeneratedArtifacts {
        sidecar_path,
        decode_proof_path,
    })
}

fn verify_artifacts(target: &Path) -> Result<ScrubEntry, String> {
    let payload = fs::read(target).map_err(|e| format!("read {}: {e}", target.display()))?;
    let sidecar_path = sidecar_path_for(target);
    let sidecar_bytes =
        fs::read(&sidecar_path).map_err(|e| format!("read {}: {e}", sidecar_path.display()))?;
    let stored: RaptorQSidecar = serde_json::from_slice(&sidecar_bytes)
        .map_err(|e| format!("parse {}: {e}", sidecar_path.display()))?;
    let recomputed = generate_raptorq_sidecar(&payload).map_err(|e| format!("encode: {e}"))?;
    if recomputed.source_hash != stored.source_hash {
        return Err(format!(
            "{}: source_hash mismatch — file changed since sidecar was written",
            target.display()
        ));
    }
    if recomputed.repair_symbol_hashes != stored.repair_symbol_hashes {
        return Err(format!(
            "{}: repair_symbol_hashes mismatch — sidecar payload integrity broken",
            target.display()
        ));
    }

    let decode_proof_path = decode_proof_path_for(target);
    let decode_bytes = fs::read(&decode_proof_path)
        .map_err(|e| format!("read {}: {e}", decode_proof_path.display()))?;
    let decode_proof: DecodeProofArtifact = serde_json::from_slice(&decode_bytes)
        .map_err(|e| format!("parse {}: {e}", decode_proof_path.display()))?;
    if decode_proof.proof_hash != recomputed.source_hash {
        return Err(format!(
            "{}: decode proof hash mismatch — expected {}, got {}",
            target.display(),
            recomputed.source_hash,
            decode_proof.proof_hash
        ));
    }
    if decode_proof.recovered_blocks == 0 {
        return Err(format!(
            "{}: decode proof did not simulate any recovered blocks",
            target.display()
        ));
    }

    Ok(ScrubEntry {
        artifact_path: target.display().to_string(),
        sidecar_path: sidecar_path.display().to_string(),
        decode_proof_path: decode_proof_path.display().to_string(),
        source_hash: recomputed.source_hash,
        source_symbols: recomputed.source_symbols,
        repair_symbols: recomputed.repair_symbols,
        recovered_blocks: decode_proof.recovered_blocks,
        status: "ok".to_owned(),
    })
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut verify_mode = false;
    let mut scrub_report = None;
    let mut targets = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--verify" => {
                verify_mode = true;
            }
            "--scrub-report" => {
                let Some(path) = args.next() else {
                    return Err("--scrub-report requires an output path".to_owned());
                };
                scrub_report = Some(PathBuf::from(path));
            }
            unknown if unknown.starts_with("--") => {
                return Err(format!("unknown option: {unknown}"));
            }
            path => {
                targets.push(PathBuf::from(path));
            }
        }
    }

    if targets.is_empty() {
        return Err("no input files".to_owned());
    }

    Ok(Cli {
        verify_mode,
        scrub_report,
        targets,
    })
}

fn write_scrub_report(path: &Path, entries: Vec<ScrubEntry>) -> Result<(), String> {
    let report = ScrubReport {
        schema_version: 1,
        generated_unix_ms: now_unix_ms(),
        checked_artifacts: entries.len(),
        entries,
    };
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?;
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("Usage: raptorq_sidecar [--verify] [--scrub-report OUT] <file> [<file> ...]");
        eprintln!();
        eprintln!("Writes <file>.raptorq.json and <file>.decode_proof.json next to each input.");
        eprintln!("Use --verify to check that existing sidecars and decode proofs still");
        eprintln!("match their source files. Use --scrub-report OUT to persist a JSON");
        eprintln!("verification report for long-lived artifact families.");
        return if args.is_empty() {
            ExitCode::from(2)
        } else {
            ExitCode::SUCCESS
        };
    }

    let cli = match parse_cli(&args) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let mut had_error = false;
    let mut had_mismatch = false;
    for target in &cli.targets {
        if cli.verify_mode {
            match verify_artifacts(target) {
                Ok(_) => println!("ok {}", target.display()),
                Err(e) => {
                    had_mismatch = true;
                    eprintln!("FAIL {}", e);
                }
            }
        } else {
            match write_artifacts(target) {
                Ok(out) => println!(
                    "wrote {} and {}",
                    out.sidecar_path.display(),
                    out.decode_proof_path.display()
                ),
                Err(e) => {
                    had_error = true;
                    eprintln!("error {}: {e}", target.display());
                }
            }
        }
    }

    if let Some(report_path) = &cli.scrub_report {
        if had_error || had_mismatch {
            eprintln!("error: refusing to write scrub report after failed generation/verification");
        } else {
            match cli
                .targets
                .iter()
                .map(|target| verify_artifacts(target))
                .collect::<Result<Vec<_>, _>>()
                .and_then(|entries| write_scrub_report(report_path, entries))
            {
                Ok(()) => println!("wrote {}", report_path.display()),
                Err(e) => {
                    had_error = true;
                    eprintln!("error: {e}");
                }
            }
        }
    }

    if had_mismatch {
        ExitCode::from(1)
    } else if had_error {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}
