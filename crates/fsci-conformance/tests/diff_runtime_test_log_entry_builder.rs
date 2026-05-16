#![forbid(unsafe_code)]
//! Cover fsci_runtime::TestLogEntry builder + JSON serialization.
//!
//! Resolves [frankenscipy-qtscj]. Walks the public TestLogEntry API:
//!   * new(test_id, module, message) defaults level=Info, timestamp_ms
//!     populated, all Optional fields None
//!   * with_result/with_seed/with_mode/with_fixture set the
//!     corresponding fields and remain composable (chain order
//!     irrelevant)
//!   * to_json_line() emits a single-line JSON object
//!   * Optional fields elided from JSON when None
//!   * Optional fields present in JSON when set
//!   * Round-trip via serde_json::from_str preserves all fields

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{RuntimeMode, TestLogEntry, TestLogLevel, TestResult};
use serde::Serialize;
use serde_json::Value;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
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
    fs::create_dir_all(output_dir()).expect("create test_log_entry diff dir");
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

#[test]
fn diff_runtime_test_log_entry_builder() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === 1. new() defaults ===
    {
        let entry = TestLogEntry::new("t1", "m1", "msg1");
        check(
            "new_test_id",
            entry.test_id == "t1",
            format!("test_id={}", entry.test_id),
        );
        check(
            "new_module",
            entry.module == "m1",
            format!("module={}", entry.module),
        );
        check(
            "new_message",
            entry.message == "msg1",
            format!("message={}", entry.message),
        );
        check(
            "new_level_info",
            matches!(entry.level, TestLogLevel::Info),
            format!("level={:?}", entry.level),
        );
        check(
            "new_timestamp_recent",
            entry.timestamp_ms > 1_700_000_000_000,
            format!("ts={}", entry.timestamp_ms),
        );
        check(
            "new_seed_none",
            entry.seed.is_none(),
            String::new(),
        );
        check(
            "new_fixture_none",
            entry.fixture_id.is_none(),
            String::new(),
        );
        check(
            "new_mode_none",
            entry.mode.is_none(),
            String::new(),
        );
        check(
            "new_result_none",
            entry.result.is_none(),
            String::new(),
        );
    }

    // === 2. Builder chain populates Optional fields ===
    {
        let entry = TestLogEntry::new("t2", "m2", "msg2")
            .with_result(TestResult::Pass)
            .with_seed(42)
            .with_mode(RuntimeMode::Hardened)
            .with_fixture("fixture-abc");
        check(
            "builder_result_set",
            entry.result == Some(TestResult::Pass),
            format!("result={:?}", entry.result),
        );
        check(
            "builder_seed_set",
            entry.seed == Some(42),
            format!("seed={:?}", entry.seed),
        );
        check(
            "builder_mode_set",
            entry.mode == Some(RuntimeMode::Hardened),
            format!("mode={:?}", entry.mode),
        );
        check(
            "builder_fixture_set",
            entry.fixture_id.as_deref() == Some("fixture-abc"),
            format!("fixture_id={:?}", entry.fixture_id),
        );
    }

    // === 3. to_json_line emits single-line JSON ===
    {
        let entry = TestLogEntry::new("t3", "m3", "msg3").with_result(TestResult::Fail);
        let line = entry.to_json_line();
        check(
            "json_single_line",
            !line.contains('\n'),
            format!("line={line}"),
        );
        let v: Value = serde_json::from_str(&line).expect("parse");
        check(
            "json_test_id_field",
            v.get("test_id").and_then(|x| x.as_str()) == Some("t3"),
            format!("v={v:?}"),
        );
        check(
            "json_result_field",
            v.get("result").and_then(|x| x.as_str()) == Some("fail"),
            format!("result={:?}", v.get("result")),
        );
        check(
            "json_level_info",
            v.get("level").and_then(|x| x.as_str()) == Some("info"),
            format!("level={:?}", v.get("level")),
        );
    }

    // === 4. Optional fields elided when None ===
    {
        let entry = TestLogEntry::new("t4", "m4", "msg4");
        let line = entry.to_json_line();
        let v: Value = serde_json::from_str(&line).expect("parse");
        let obj = v.as_object().expect("object");
        check(
            "json_seed_elided",
            !obj.contains_key("seed"),
            format!("keys={:?}", obj.keys().collect::<Vec<_>>()),
        );
        check(
            "json_fixture_elided",
            !obj.contains_key("fixture_id"),
            String::new(),
        );
        check(
            "json_mode_elided",
            !obj.contains_key("mode"),
            String::new(),
        );
        check(
            "json_result_elided",
            !obj.contains_key("result"),
            String::new(),
        );
        check(
            "json_artifact_refs_elided",
            !obj.contains_key("artifact_refs"),
            String::new(),
        );
    }

    // === 5. Optional fields present when set ===
    {
        let entry = TestLogEntry::new("t5", "m5", "msg5")
            .with_seed(7)
            .with_fixture("fix5")
            .with_mode(RuntimeMode::Strict)
            .with_result(TestResult::Skip);
        let line = entry.to_json_line();
        let v: Value = serde_json::from_str(&line).expect("parse");
        let obj = v.as_object().expect("object");
        check(
            "json_seed_present",
            obj.get("seed").and_then(|x| x.as_u64()) == Some(7),
            String::new(),
        );
        check(
            "json_fixture_present",
            obj.get("fixture_id").and_then(|x| x.as_str()) == Some("fix5"),
            String::new(),
        );
        check(
            "json_mode_present",
            obj.contains_key("mode"),
            String::new(),
        );
        check(
            "json_result_skip",
            obj.get("result").and_then(|x| x.as_str()) == Some("skip"),
            String::new(),
        );
    }

    // === 6. JSON round-trip preserves all fields ===
    {
        let original = TestLogEntry::new("t6", "m6", "msg6")
            .with_seed(99)
            .with_mode(RuntimeMode::Hardened)
            .with_result(TestResult::Warn)
            .with_fixture("rt-fix");
        let line = original.to_json_line();
        let parsed: TestLogEntry = serde_json::from_str(&line).expect("round-trip parse");
        check(
            "roundtrip_test_id",
            parsed.test_id == original.test_id,
            String::new(),
        );
        check(
            "roundtrip_module",
            parsed.module == original.module,
            String::new(),
        );
        check(
            "roundtrip_seed",
            parsed.seed == original.seed,
            String::new(),
        );
        check(
            "roundtrip_mode",
            parsed.mode == original.mode,
            String::new(),
        );
        check(
            "roundtrip_result",
            parsed.result == original.result,
            String::new(),
        );
        check(
            "roundtrip_fixture",
            parsed.fixture_id == original.fixture_id,
            String::new(),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_runtime_test_log_entry_builder".into(),
        category: "fsci_runtime::TestLogEntry builder + JSON serialization".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("test_log_entry mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "TestLogEntry builder coverage failed: {} cases",
        diffs.len(),
    );
}
