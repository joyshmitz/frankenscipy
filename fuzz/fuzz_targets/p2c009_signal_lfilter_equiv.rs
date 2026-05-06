#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::lfilter;
use libfuzzer_sys::fuzz_target;

// lfilter equivalence oracle for [frankenscipy-7pnw4].
//
// After the fb1f358 perf opt (pad b/a to nfilt, hoist b0,
// drop per-iter bound checks), verify:
//   1. Output matches a naïve in-fuzz dual-path implementation
//      that mirrors the pre-fix branchy logic.
//   2. Output length equals input length.
//   3. No panic for any sanitized finite input.

const BOUND: f64 = 1.0e3;
const MAX_FILT: usize = 12;
const MAX_X: usize = 256;
const MIN_X: usize = 1;

#[derive(Debug, Arbitrary)]
struct LfilterInput {
    b: Vec<f64>,
    a: Vec<f64>,
    x: Vec<f64>,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-BOUND, BOUND)
    } else {
        0.0
    }
}

fn naive_lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let a0 = a[0];
    let nb = b.len();
    let na = a.len();
    let nfilt = nb.max(na);
    let b_norm: Vec<f64> = b.iter().map(|&v| v / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&v| v / a0).collect();
    let mut y = Vec::with_capacity(x.len());
    let mut d = vec![0.0_f64; nfilt];
    for &xi in x {
        let yi = b_norm.first().copied().unwrap_or(0.0) * xi + d[0];
        y.push(yi);
        for j in 0..nfilt - 1 {
            let bj = if j + 1 < nb { b_norm[j + 1] } else { 0.0 };
            let aj = if j + 1 < na { a_norm[j + 1] } else { 0.0 };
            d[j] = bj * xi - aj * yi + if j + 1 < nfilt - 1 { d[j + 1] } else { 0.0 };
        }
    }
    y
}

fuzz_target!(|input: LfilterInput| {
    // b must be non-empty.
    let mut b: Vec<f64> = input
        .b
        .iter()
        .take(MAX_FILT)
        .copied()
        .map(sanitize)
        .collect();
    if b.is_empty() {
        b.push(1.0);
    }

    // a must be non-empty and a[0] != 0.
    let mut a: Vec<f64> = input
        .a
        .iter()
        .take(MAX_FILT)
        .copied()
        .map(sanitize)
        .collect();
    if a.is_empty() {
        a.push(1.0);
    } else if a[0] == 0.0 {
        a[0] = 1.0;
    }

    let x: Vec<f64> = input
        .x
        .iter()
        .take(MAX_X)
        .copied()
        .map(sanitize)
        .collect();
    if x.len() < MIN_X {
        return;
    }

    let opt = match lfilter(&b, &a, &x, None) {
        Ok(y) => y,
        Err(e) => panic!("lfilter rejected sanitized input: {e}"),
    };
    assert_eq!(opt.len(), x.len(), "output length must equal input length");

    let nav = naive_lfilter(&b, &a, &x);
    assert_eq!(opt.len(), nav.len());
    for (i, (&u, &v)) in opt.iter().zip(nav.iter()).enumerate() {
        let scale = u.abs().max(v.abs()).max(1e-12);
        if (u - v).abs() > 1e-9 * scale {
            panic!(
                "lfilter perf-opt diverges from naive at i={i}: opt={u}, naive={v} (b.len()={}, a.len()={}, x.len()={})",
                b.len(),
                a.len(),
                x.len()
            );
        }
    }
});
