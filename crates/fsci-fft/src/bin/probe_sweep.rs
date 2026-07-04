//! Broad FFT size sweep to spot anomalously slow size classes.
use fsci_fft::{fft, Complex64, FftOptions};
use std::time::Instant;

fn factor_label(mut n: usize) -> String {
    let mut parts = Vec::new();
    for p in [2usize, 3, 5, 7, 11, 13] {
        let mut c = 0;
        while n % p == 0 {
            n /= p;
            c += 1;
        }
        if c > 0 {
            parts.push(format!("{p}^{c}"));
        }
    }
    if n > 1 {
        parts.push(format!("{n}(prime)"));
    }
    parts.join("·")
}

fn time_fft(n: usize) -> f64 {
    let mut state = 0x9E37u64;
    let input: Vec<Complex64> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 12) as f64 / (1u64 << 52) as f64 - 0.5, 0.0)
        })
        .collect();
    let opts = FftOptions::default();
    let _ = fft(&input, &opts).unwrap();
    let reps = (2_000_000 / n).max(20);
    let t = Instant::now();
    let mut acc = 0.0;
    for _ in 0..reps {
        acc += fft(&input, &opts).unwrap()[1].0;
    }
    std::hint::black_box(acc);
    t.elapsed().as_secs_f64() * 1e9 / reps as f64 / n as f64 // ns/point
}

fn main() {
    // sizes chosen to isolate factor classes near ~1400 and ~2200
    let sizes = [
        1024usize, 2048, 4096, // pow2 baseline
        1200, 1440, 2400, // 5-smooth
        1400, 2100, // factor 7
        1408, 2200, 2816, 1331, // factor 11 (1408=2^7·11, 1331=11^3)
        1352, 2197, // factor 13 (2197=13^3)
        1409, 2213, // primes
    ];
    let mut results: Vec<(usize, f64, String)> = sizes
        .iter()
        .map(|&n| (n, time_fft(n), factor_label(n)))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("--- fsci fft ns/point, slowest first ---");
    for (n, nsp, lab) in &results {
        println!("n={n:>5}  {nsp:>7.2} ns/pt   {lab}");
    }
}
