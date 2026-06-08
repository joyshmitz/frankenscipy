//! Byte-identity + timing harness for loadtxt, whose per-line f64 parsing is now
//! chunked across cores (with an exact serial fallback on malformed input). The parsed
//! (rows, cols, data) is bit-identical to the serial loop (same deterministic f64 parse,
//! same row order, same cols). Compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-io --bin perf_loadtxt`.

use std::hint::black_box;
use std::time::Instant;

use fsci_io::loadtxt;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn make(rows: usize, cols: usize, seed: u64) -> String {
    let mut s = seed;
    let mut out = String::with_capacity(rows * cols * 20);
    for r in 0..rows {
        if r % 17 == 0 {
            out.push_str("# comment line\n");
        }
        for c in 0..cols {
            if c > 0 {
                out.push(' ');
            }
            // mix magnitudes / signs so parsing exercises real float formatting
            let v = (lcg(&mut s) - 0.5) * 10f64.powi((r as i32 % 7) - 3);
            out.push_str(&format!("{v:.9e}"));
        }
        out.push('\n');
    }
    out
}

fn golden(content: &str) -> (usize, usize, u64) {
    let (rows, cols, data) = loadtxt(content).expect("loadtxt");
    let mut acc = 0u64;
    for (i, &v) in data.iter().enumerate() {
        acc ^= v.to_bits().rotate_left((i % 64) as u32);
    }
    (rows, cols, acc)
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(r, c) in &[(100usize, 5usize), (5000, 8), (20000, 3)] {
        let content = make(r, c, 1);
        let (rows, cols, acc) = golden(&content);
        println!("req=({r}x{c}) rows={rows} cols={cols} data_xor_bits={acc:016x}");
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(r, c) in &[(300_000usize, 8usize), (800_000, 4)] {
        let content = make(r, c, 7);
        let reps = 5;
        let _ = loadtxt(&content);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += loadtxt(black_box(&content)).expect("loadtxt").2[0];
        }
        println!("rows={r} cols={c}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
