#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_constants::value()`
//! name-based CODATA lookup. Maps fsci's lowercase key to the
//! corresponding scipy.constants.value() canonical name.
//!
//! Resolves [frankenscipy-uquh5]. Rel 1e-7 (fsci ships CODATA-2018).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_constants as fc;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    fsci_key: String,
    scipy_key: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create value_lookup diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize value_lookup diff log");
    fs::write(path, json).expect("write value_lookup diff log");
}

/// (case_id, fsci_key, scipy_key) for paired lookups.
fn key_pairs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("speed_of_light", "speed of light", "speed of light in vacuum"),
        ("planck", "planck", "Planck constant"),
        ("hbar", "hbar", "reduced Planck constant"),
        ("g_grav", "gravitational constant", "Newtonian constant of gravitation"),
        ("elementary_charge", "elementary charge", "elementary charge"),
        ("gas_constant", "gas constant", "molar gas constant"),
        ("avogadro", "avogadro", "Avogadro constant"),
        ("boltzmann", "boltzmann", "Boltzmann constant"),
        ("stefan_boltzmann", "stefan-boltzmann", "Stefan-Boltzmann constant"),
        ("wien", "wien", "Wien wavelength displacement law constant"),
        ("electron_mass", "electron mass", "electron mass"),
        ("proton_mass", "proton mass", "proton mass"),
        ("neutron_mass", "neutron mass", "neutron mass"),
        ("rydberg", "rydberg", "Rydberg constant"),
        ("fine_structure", "fine-structure", "fine-structure constant"),
        ("bohr_radius", "bohr radius", "Bohr radius"),
        ("faraday", "faraday constant", "Faraday constant"),
        ("electron_g_factor", "electron g factor", "electron g factor"),
        ("thomson_cross_section", "thomson cross section", "Thomson cross section"),
        ("characteristic_impedance_of_vacuum", "characteristic impedance of vacuum", "characteristic impedance of vacuum"),
    ]
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: key_pairs()
            .iter()
            .map(|(case_id, fk, sk)| PointCase {
                case_id: (*case_id).into(),
                fsci_key: (*fk).into(),
                scipy_key: (*sk).into(),
            })
            .collect(),
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import constants

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; sk = case["scipy_key"]
    try:
        v = float(constants.value(sk))
        if not math.isfinite(v):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize value_lookup query");
    let mut child = match Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for value_lookup oracle: {e}"
            );
            eprintln!("skipping value_lookup oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open value_lookup oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "value_lookup oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping value_lookup oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for value_lookup oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "value_lookup oracle failed: {stderr}"
        );
        eprintln!("skipping value_lookup oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse value_lookup oracle JSON"))
}

#[test]
fn diff_constants_value_lookup() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Some(fsci_v) = fc::value(&case.fsci_key) else {
            // fsci doesn't know this key — count as failure to surface
            // any naming regressions.
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                rel_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        };
        let r = if scipy_v.abs() > 0.0 {
            (fsci_v - scipy_v).abs() / scipy_v.abs()
        } else {
            (fsci_v - scipy_v).abs()
        };
        max_overall = max_overall.max(r);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            rel_diff: r,
            pass: r <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_constants_value_lookup".into(),
        category: "scipy.constants.value() name-based lookup".into(),
        case_count: diffs.len(),
        max_rel_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("value() mismatch: {} rel_diff={}", d.case_id, d.rel_diff);
        }
    }

    assert!(
        all_pass,
        "value_lookup conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
}
