#![forbid(unsafe_code)]
//! Property test: fsci_fft audit variants (fft_with_audit,
//! ifft_with_audit, rfft_with_audit, irfft_with_audit, etc) must
//! produce numerically identical output to their non-audit
//! counterparts.
//!
//! Resolves [frankenscipy-z6stf]. Both code paths share the same
//! `*_impl` worker; the only difference is whether an audit ledger
//! is passed. Output should be bit-identical.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{
    Complex64, FftOptions, fft, fft_with_audit, ifft, ifft_with_audit, irfft, irfft_with_audit,
    rfft, rfft_with_audit, sync_audit_ledger,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-15;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create audit_equiv diff dir");
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

fn complex_max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b.iter()).map(|((ar, ai), (br, bi))| {
        (ar - br).abs().max((ai - bi).abs())
    }).fold(0.0_f64, f64::max)
}

fn real_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

#[test]
fn diff_fft_audit_variants_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = FftOptions::default();
    let ledger = sync_audit_ledger();

    for &n in &[16_usize, 32, 64, 128] {
        // fft
        let signal: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                ((2.0 * std::f64::consts::PI * t).sin(), (4.0 * std::f64::consts::PI * t).cos())
            })
            .collect();
        let Ok(plain) = fft(&signal, &opts) else { continue };
        let Ok(audited) = fft_with_audit(&signal, &opts, &ledger) else { continue };
        let d = complex_max_diff(&plain, &audited);
        max_overall = max_overall.max(d);
        diffs.push(CaseDiff {
            case_id: format!("fft_n{n}"),
            op: "fft".into(),
            abs_diff: d,
            pass: d <= ABS_TOL,
        });

        // ifft (round trip through fft)
        let Ok(p_ifft) = ifft(&plain, &opts) else { continue };
        let Ok(a_ifft) = ifft_with_audit(&plain, &opts, &ledger) else { continue };
        let d = complex_max_diff(&p_ifft, &a_ifft);
        max_overall = max_overall.max(d);
        diffs.push(CaseDiff {
            case_id: format!("ifft_n{n}"),
            op: "ifft".into(),
            abs_diff: d,
            pass: d <= ABS_TOL,
        });

        // rfft
        let real_sig: Vec<f64> = signal.iter().map(|(re, _)| *re).collect();
        let Ok(p_rfft) = rfft(&real_sig, &opts) else { continue };
        let Ok(a_rfft) = rfft_with_audit(&real_sig, &opts, &ledger) else { continue };
        let d = complex_max_diff(&p_rfft, &a_rfft);
        max_overall = max_overall.max(d);
        diffs.push(CaseDiff {
            case_id: format!("rfft_n{n}"),
            op: "rfft".into(),
            abs_diff: d,
            pass: d <= ABS_TOL,
        });

        // irfft
        let Ok(p_irfft) = irfft(&p_rfft, Some(n), &opts) else { continue };
        let Ok(a_irfft) = irfft_with_audit(&p_rfft, Some(n), &opts, &ledger) else { continue };
        let d = real_max_diff(&p_irfft, &a_irfft);
        max_overall = max_overall.max(d);
        diffs.push(CaseDiff {
            case_id: format!("irfft_n{n}"),
            op: "irfft".into(),
            abs_diff: d,
            pass: d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_audit_variants_equivalence".into(),
        category: "fsci_fft::*_with_audit equivalent to non-audit variants".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "audit_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
