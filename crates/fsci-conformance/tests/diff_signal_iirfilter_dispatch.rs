#![forbid(unsafe_code)]
//! Cover fsci_signal::iirfilter dispatch across all 5 IIR families.
//!
//! Resolves [frankenscipy-he0h9]. iirfilter is a dispatch wrapper
//! that routes to butter / cheby1 / cheby2 / bessel / ellip based on
//! the IirFamily enum, validating that the required ripple/attenuation
//! parameters are supplied. Verifies:
//!   * Each family produces a BaCoeffs equal to its direct designer
//!   * Cheby1/Elliptic without rp → error
//!   * Cheby2/Elliptic without rs → error

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{
    FilterType, IirFamily, bessel, butter, cheby1, cheby2, ellip, iirfilter,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-14;

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
    fs::create_dir_all(output_dir()).expect("create iirfilter diff dir");
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

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_signal_iirfilter_dispatch() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    let order = 3;
    let wn = vec![0.3_f64];
    let btype = FilterType::Lowpass;
    let rp = 0.5;
    let rs = 40.0;

    // === Butterworth ===
    {
        let direct = butter(order, &wn, btype).expect("butter");
        let dispatched = iirfilter(order, &wn, btype, IirFamily::Butterworth, None, None)
            .expect("iirfilter butter");
        let mab = max_abs(&direct.b, &dispatched.b);
        let maa = max_abs(&direct.a, &dispatched.a);
        check(
            "butterworth_matches_direct",
            mab <= ABS_TOL && maa <= ABS_TOL,
            format!("b_max={mab} a_max={maa}"),
        );
    }

    // === Chebyshev1 ===
    {
        let direct = cheby1(order, rp, &wn, btype).expect("cheby1");
        let dispatched = iirfilter(order, &wn, btype, IirFamily::Chebyshev1, Some(rp), None)
            .expect("iirfilter cheby1");
        let mab = max_abs(&direct.b, &dispatched.b);
        let maa = max_abs(&direct.a, &dispatched.a);
        check(
            "chebyshev1_matches_direct",
            mab <= ABS_TOL && maa <= ABS_TOL,
            format!("b_max={mab} a_max={maa}"),
        );
    }

    // === Chebyshev2 ===
    {
        let direct = cheby2(order, rs, &wn, btype).expect("cheby2");
        let dispatched = iirfilter(order, &wn, btype, IirFamily::Chebyshev2, None, Some(rs))
            .expect("iirfilter cheby2");
        let mab = max_abs(&direct.b, &dispatched.b);
        let maa = max_abs(&direct.a, &dispatched.a);
        check(
            "chebyshev2_matches_direct",
            mab <= ABS_TOL && maa <= ABS_TOL,
            format!("b_max={mab} a_max={maa}"),
        );
    }

    // === Bessel ===
    {
        let direct = bessel(order, &wn, btype).expect("bessel");
        let dispatched = iirfilter(order, &wn, btype, IirFamily::Bessel, None, None)
            .expect("iirfilter bessel");
        let mab = max_abs(&direct.b, &dispatched.b);
        let maa = max_abs(&direct.a, &dispatched.a);
        check(
            "bessel_matches_direct",
            mab <= ABS_TOL && maa <= ABS_TOL,
            format!("b_max={mab} a_max={maa}"),
        );
    }

    // === Elliptic ===
    {
        let direct = ellip(order, rp, rs, &wn, btype).expect("ellip");
        let dispatched = iirfilter(order, &wn, btype, IirFamily::Elliptic, Some(rp), Some(rs))
            .expect("iirfilter ellip");
        let mab = max_abs(&direct.b, &dispatched.b);
        let maa = max_abs(&direct.a, &dispatched.a);
        check(
            "elliptic_matches_direct",
            mab <= ABS_TOL && maa <= ABS_TOL,
            format!("b_max={mab} a_max={maa}"),
        );
    }

    // === Missing-rp errors for cheby1 and elliptic ===
    {
        let r = iirfilter(order, &wn, btype, IirFamily::Chebyshev1, None, None);
        check(
            "chebyshev1_missing_rp_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }
    {
        let r = iirfilter(order, &wn, btype, IirFamily::Elliptic, None, Some(rs));
        check(
            "elliptic_missing_rp_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    // === Missing-rs errors for cheby2 and elliptic ===
    {
        let r = iirfilter(order, &wn, btype, IirFamily::Chebyshev2, None, None);
        check(
            "chebyshev2_missing_rs_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }
    {
        let r = iirfilter(order, &wn, btype, IirFamily::Elliptic, Some(rp), None);
        check(
            "elliptic_missing_rs_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_iirfilter_dispatch".into(),
        category: "fsci_signal::iirfilter dispatch coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("iirfilter mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "iirfilter dispatch coverage failed: {} cases",
        diffs.len()
    );
}
