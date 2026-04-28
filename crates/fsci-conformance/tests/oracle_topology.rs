//! Oracle topology regression test (frankenscipy-1i07).
//!
//! Verifies oracle capture coverage across P2C packet families.
//! Each packet family either has:
//! - An oracle_capture.json artifact, OR
//! - An explicit documented exemption in ORACLE_EXEMPT

use std::collections::HashSet;
use std::fs;
use std::path::Path;

fn project_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("should find project root from conformance crate")
}

fn conformance_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

const ORACLE_EXEMPT: &[(&str, &str)] = &[
    ("P2C-001", "IVP/solve_ivp uses internal step-by-step validation, not SciPy oracle comparison"),
    ("P2C-004", "Sparse format/spmv uses property-based parity gates, not pointwise oracle"),
    ("P2C-008", "Integrate/quad uses adaptive additivity property, not oracle comparison"),
    ("P2C-009", "Signal/convolve uses algebraic equivalence oracle, not SciPy capture"),
    ("P2C-010", "Interpolate uses passthrough property (interp(xi)=yi), not oracle capture"),
    ("P2C-011", "Spatial uses squareform roundtrip property, not oracle capture"),
    ("P2C-013", "Ndimage uses idempotence property for constant arrays, not oracle capture"),
];

#[test]
fn every_packet_has_oracle_or_exemption() {
    let artifacts_dir = conformance_root().join("fixtures/artifacts");
    let oracle_dir = conformance_root().join("python_oracle");

    if !artifacts_dir.exists() {
        eprintln!("fixtures/artifacts/ not found, skipping oracle topology check");
        return;
    }

    let exempt_set: HashSet<&str> = ORACLE_EXEMPT.iter().map(|(p, _)| *p).collect();

    let mut packets_with_oracle: HashSet<String> = HashSet::new();
    for entry in fs::read_dir(&artifacts_dir).expect("read artifacts dir") {
        let entry = entry.expect("read entry");
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("FSCI-P2C-") {
            continue;
        }
        let packet_dir = entry.path();
        let has_oracle = packet_dir.join("oracle_capture.json").exists()
            || packet_dir.join("differential/oracle_capture.json").exists();
        if has_oracle {
            let short_name = name.strip_prefix("FSCI-").unwrap_or(&name).to_string();
            packets_with_oracle.insert(short_name);
        }
    }

    let oracle_scripts: HashSet<String> = if oracle_dir.exists() {
        fs::read_dir(&oracle_dir)
            .expect("read oracle dir")
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("scipy_") && name.ends_with("_oracle.py") {
                    Some(name)
                } else {
                    None
                }
            })
            .collect()
    } else {
        HashSet::new()
    };

    let mut all_packets: Vec<String> = Vec::new();
    for entry in fs::read_dir(&artifacts_dir).expect("read artifacts dir") {
        let entry = entry.expect("read entry");
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("FSCI-P2C-") {
            let short_name = name.strip_prefix("FSCI-").unwrap_or(&name).to_string();
            all_packets.push(short_name);
        }
    }
    all_packets.sort();

    let mut missing: Vec<&str> = Vec::new();
    for packet in &all_packets {
        let has_oracle = packets_with_oracle.contains(packet);
        let is_exempt = exempt_set.contains(packet.as_str());
        if !has_oracle && !is_exempt {
            missing.push(packet);
        }
    }

    if !missing.is_empty() {
        missing.sort();
        panic!(
            "Packet families missing oracle_capture.json or ORACLE_EXEMPT entry:\n  {}\n\n\
             Per bead frankenscipy-1i07, every P2C packet must either have:\n\
             - oracle_capture.json in fixtures/artifacts/FSCI-<packet>/, or\n\
             - An entry in ORACLE_EXEMPT with documented rationale\n\
             Add the missing capture or update ORACLE_EXEMPT in this file.",
            missing.join("\n  ")
        );
    }

    eprintln!("\n── Oracle Coverage Matrix ──");
    eprintln!("  Packets with oracle capture: {}", packets_with_oracle.len());
    for p in &packets_with_oracle {
        eprintln!("    [oracle] {}", p);
    }
    eprintln!("  Packets with exemption: {}", ORACLE_EXEMPT.len());
    for (p, reason) in ORACLE_EXEMPT {
        eprintln!("    [exempt] {} — {}", p, reason);
    }
    eprintln!("  Oracle scripts present: {}", oracle_scripts.len());
}

#[test]
fn exempt_packets_exist() {
    let artifacts_dir = conformance_root().join("fixtures/artifacts");

    if !artifacts_dir.exists() {
        eprintln!("fixtures/artifacts/ not found, skipping exemption check");
        return;
    }

    let mut missing: Vec<&str> = Vec::new();
    for (packet, _reason) in ORACLE_EXEMPT {
        let packet_dir = artifacts_dir.join(format!("FSCI-{}", packet));
        if !packet_dir.exists() {
            missing.push(packet);
        }
    }

    if !missing.is_empty() {
        missing.sort();
        panic!(
            "ORACLE_EXEMPT references non-existent packet directories:\n  {}\n\n\
             Either create the packet directory or remove the exemption.",
            missing.join("\n  ")
        );
    }
}
