#![forbid(unsafe_code)]
//! Property test: fsci_fft 2D/N-D audit variants produce numerically
//! identical output to their non-audit counterparts.
//!
//! Resolves [frankenscipy-r28td]. Covers fft2/ifft2/fftn/ifftn,
//! rfft2/irfft2/rfftn/irfftn, hfft/ihfft. 1e-15 abs.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{
    Complex64, FftOptions, fft2, fft2_with_audit, fftn, fftn_with_audit, hfft, hfft_with_audit,
    ifft2, ifft2_with_audit, ifftn, ifftn_with_audit, ihfft, ihfft_with_audit, irfft2,
    irfft2_with_audit, irfftn, irfftn_with_audit, rfft2, rfft2_with_audit, rfftn,
    rfftn_with_audit, sync_audit_ledger,
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
    fs::create_dir_all(output_dir()).expect("create audit_nd diff dir");
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
fn diff_fft_audit_variants_nd_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = FftOptions::default();
    let ledger = sync_audit_ledger();

    // 2D probes
    for &shape in &[(8_usize, 8_usize), (16, 16), (8, 16)] {
        let (r, c) = shape;
        let signal: Vec<Complex64> = (0..r * c)
            .map(|i| {
                let t = i as f64 / (r * c) as f64;
                ((2.0 * std::f64::consts::PI * t).sin(), (4.0 * std::f64::consts::PI * t).cos())
            })
            .collect();
        let real_sig: Vec<f64> = signal.iter().map(|(re, _)| *re).collect();

        if let (Ok(p), Ok(a)) = (fft2(&signal, shape, &opts), fft2_with_audit(&signal, shape, &opts, &ledger)) {
            let d = complex_max_diff(&p, &a);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff { case_id: format!("fft2_{r}x{c}"), op: "fft2".into(), abs_diff: d, pass: d <= ABS_TOL });
            // ifft2 on the fft2 result
            if let (Ok(pi), Ok(ai)) = (ifft2(&p, shape, &opts), ifft2_with_audit(&p, shape, &opts, &ledger)) {
                let d = complex_max_diff(&pi, &ai);
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff { case_id: format!("ifft2_{r}x{c}"), op: "ifft2".into(), abs_diff: d, pass: d <= ABS_TOL });
            }
        }

        // rfft2
        if let (Ok(p), Ok(a)) = (rfft2(&real_sig, shape, &opts), rfft2_with_audit(&real_sig, shape, &opts, &ledger)) {
            let d = complex_max_diff(&p, &a);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff { case_id: format!("rfft2_{r}x{c}"), op: "rfft2".into(), abs_diff: d, pass: d <= ABS_TOL });
            if let (Ok(pi), Ok(ai)) = (irfft2(&p, shape, &opts), irfft2_with_audit(&p, shape, &opts, &ledger)) {
                let d = real_max_diff(&pi, &ai);
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff { case_id: format!("irfft2_{r}x{c}"), op: "irfft2".into(), abs_diff: d, pass: d <= ABS_TOL });
            }
        }
    }

    // N-D probes
    for shape in &[vec![4_usize, 4, 4], vec![8, 8, 4]] {
        let n: usize = shape.iter().product();
        let signal: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                ((2.0 * std::f64::consts::PI * t).sin(), (4.0 * std::f64::consts::PI * t).cos())
            })
            .collect();
        let real_sig: Vec<f64> = signal.iter().map(|(re, _)| *re).collect();
        let s = shape.as_slice();
        if let (Ok(p), Ok(a)) = (fftn(&signal, s, &opts), fftn_with_audit(&signal, s, &opts, &ledger)) {
            let d = complex_max_diff(&p, &a);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff { case_id: format!("fftn_{shape:?}"), op: "fftn".into(), abs_diff: d, pass: d <= ABS_TOL });
            if let (Ok(pi), Ok(ai)) = (ifftn(&p, s, &opts), ifftn_with_audit(&p, s, &opts, &ledger)) {
                let d = complex_max_diff(&pi, &ai);
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff { case_id: format!("ifftn_{shape:?}"), op: "ifftn".into(), abs_diff: d, pass: d <= ABS_TOL });
            }
        }
        if let (Ok(p), Ok(a)) = (rfftn(&real_sig, s, &opts), rfftn_with_audit(&real_sig, s, &opts, &ledger)) {
            let d = complex_max_diff(&p, &a);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff { case_id: format!("rfftn_{shape:?}"), op: "rfftn".into(), abs_diff: d, pass: d <= ABS_TOL });
            if let (Ok(pi), Ok(ai)) = (irfftn(&p, s, &opts), irfftn_with_audit(&p, s, &opts, &ledger)) {
                let d = real_max_diff(&pi, &ai);
                max_overall = max_overall.max(d);
                diffs.push(CaseDiff { case_id: format!("irfftn_{shape:?}"), op: "irfftn".into(), abs_diff: d, pass: d <= ABS_TOL });
            }
        }
    }

    // hfft / ihfft (1D)
    for &n in &[16_usize, 32, 64] {
        let cmpx: Vec<Complex64> = (0..n / 2 + 1)
            .map(|i| (i as f64 * 0.3, i as f64 * 0.2))
            .collect();
        if let (Ok(p), Ok(a)) = (hfft(&cmpx, Some(n), &opts), hfft_with_audit(&cmpx, Some(n), &opts, &ledger)) {
            let d = real_max_diff(&p, &a);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff { case_id: format!("hfft_n{n}"), op: "hfft".into(), abs_diff: d, pass: d <= ABS_TOL });
        }
        let real_sig: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        if let (Ok(p), Ok(a)) = (ihfft(&real_sig, Some(n), &opts), ihfft_with_audit(&real_sig, Some(n), &opts, &ledger)) {
            let d = complex_max_diff(&p, &a);
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff { case_id: format!("ihfft_n{n}"), op: "ihfft".into(), abs_diff: d, pass: d <= ABS_TOL });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_audit_variants_nd_equivalence".into(),
        category: "fsci_fft N-D audit variants equivalent to non-audit".into(),
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
        "audit_nd_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
