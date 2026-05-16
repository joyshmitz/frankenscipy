#![forbid(unsafe_code)]
//! Live scipy parity for fsci_special Kelvin functions.
//!
//! Resolves [frankenscipy-3iudm]. Compares ber(x), bei(x), ker(x),
//! kei(x) against scipy.special equivalents across x ∈ [0.1, 6.0].
//! ber/bei use direct alternating series and are accurate up to ~6.
//! ker/kei use a series with harmonic-number correction terms; the
//! oracle compares within rel tol 1e-3 to allow for moderate drift.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{bei, ber, kei, ker};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL_BER_BEI: f64 = 1.0e-9;
const REL_TOL_KER_KEI: f64 = 1.0e-3;
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    func: String,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    actual: f64,
    expected: f64,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create kelvin diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();
    let xs: &[f64] = &[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0];
    for &x in xs {
        for func in ["ber", "bei", "ker", "kei"] {
            pts.push(CasePoint {
                case_id: format!("{func}_x{x}"),
                func: func.into(),
                x,
            });
        }
    }
    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
from scipy import special

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]; fn = c["func"]; x = c["x"]
    try:
        if fn == "ber":   v = float(special.ber(x))
        elif fn == "bei": v = float(special.bei(x))
        elif fn == "ker": v = float(special.ker(x))
        elif fn == "kei": v = float(special.kei(x))
        else: v = None
        if v is None or not math.isfinite(v):
            out.append({"case_id": cid, "value": None})
        else:
            out.append({"case_id": cid, "value": v})
    except Exception:
        out.append({"case_id": cid, "value": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping kelvin oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping kelvin oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping kelvin oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_special_kelvin_ber_bei_ker_kei() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (c, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(c.case_id, o.case_id);
        let Some(expected) = o.value else {
            continue;
        };

        let actual = match c.func.as_str() {
            "ber" => ber(c.x),
            "bei" => bei(c.x),
            "ker" => ker(c.x),
            "kei" => kei(c.x),
            _ => panic!("unknown func {}", c.func),
        };

        let abs_d = (actual - expected).abs();
        let denom = expected.abs().max(1.0e-300);
        let rel_diff = abs_d / denom;
        let tol = if matches!(c.func.as_str(), "ber" | "bei") {
            REL_TOL_BER_BEI
        } else {
            REL_TOL_KER_KEI
        };
        let pass = rel_diff <= tol || abs_d <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: c.case_id.clone(),
            func: c.func.clone(),
            actual,
            expected,
            rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_kelvin_ber_bei_ker_kei".into(),
        category: "scipy.special Kelvin functions ber/bei/ker/kei".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "kelvin mismatch: {} ({}) actual={} expected={} rel={}",
                d.case_id, d.func, d.actual, d.expected, d.rel_diff
            );
        }
    }

    assert!(all_pass, "Kelvin function parity failed: {} cases", diffs.len());
}
