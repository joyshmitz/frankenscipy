//! Generate RaptorQ sidecars for arbitrary long-lived artifacts.
//!
//! SPEC §6 / §9 require RaptorQ-everywhere durability for every long-lived
//! artifact family, not just conformance fixture parity reports. This bin
//! reuses `generate_raptorq_sidecar` from the conformance crate over any
//! file path the caller hands it, writing `<path>.raptorq.json` next to it.
//!
//! Usage:
//!     raptorq_sidecar <file> [<file> ...]
//!     raptorq_sidecar --verify <file> [<file> ...]
//!
//! Exit codes:
//!     0 — all sidecars generated/verified
//!     1 — verification mismatch
//!     2 — IO/argument error

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use fsci_conformance::{RaptorQSidecar, generate_raptorq_sidecar};

fn sidecar_path_for(path: &Path) -> PathBuf {
    let mut out = path.as_os_str().to_owned();
    out.push(".raptorq.json");
    PathBuf::from(out)
}

fn write_sidecar(target: &Path) -> Result<PathBuf, String> {
    let payload = fs::read(target).map_err(|e| format!("read {}: {e}", target.display()))?;
    let sidecar = generate_raptorq_sidecar(&payload).map_err(|e| format!("encode: {e}"))?;
    let serialized = serde_json::to_vec_pretty(&sidecar).map_err(|e| format!("serialize: {e}"))?;
    let out_path = sidecar_path_for(target);
    fs::write(&out_path, &serialized).map_err(|e| format!("write {}: {e}", out_path.display()))?;
    Ok(out_path)
}

fn verify_sidecar(target: &Path) -> Result<(), String> {
    let payload = fs::read(target).map_err(|e| format!("read {}: {e}", target.display()))?;
    let sidecar_path = sidecar_path_for(target);
    let sidecar_bytes = fs::read(&sidecar_path)
        .map_err(|e| format!("read {}: {e}", sidecar_path.display()))?;
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
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("Usage: raptorq_sidecar [--verify] <file> [<file> ...]");
        eprintln!();
        eprintln!("Writes <file>.raptorq.json next to each input. Use --verify to check");
        eprintln!("that an existing sidecar still matches its source file.");
        return if args.is_empty() {
            ExitCode::from(2)
        } else {
            ExitCode::SUCCESS
        };
    }

    let verify_mode = args.iter().any(|a| a == "--verify");
    let targets: Vec<PathBuf> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .collect();

    if targets.is_empty() {
        eprintln!("error: no input files");
        return ExitCode::from(2);
    }

    let mut had_error = false;
    let mut had_mismatch = false;
    for target in &targets {
        if verify_mode {
            match verify_sidecar(target) {
                Ok(()) => println!("ok {}", target.display()),
                Err(e) => {
                    had_mismatch = true;
                    eprintln!("FAIL {}", e);
                }
            }
        } else {
            match write_sidecar(target) {
                Ok(out) => println!("wrote {}", out.display()),
                Err(e) => {
                    had_error = true;
                    eprintln!("error {}: {e}", target.display());
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
