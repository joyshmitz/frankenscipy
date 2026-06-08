//! Byte-identity + timing harness for read_csv, whose per-row f64 parsing is now chunked
//! across cores (with an exact serial fallback on any anomaly). The returned (header, data)
//! is bit-identical to the serial loop (same deterministic f64 parse, same row order).
//! Compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-io --bin perf_read_csv`.

use std::hint::black_box;
use std::time::Instant;

use fsci_io::read_csv;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn make(rows: usize, cols: usize, seed: u64, header: bool) -> String {
    let mut s = seed;
    let mut out = String::with_capacity(rows * cols * 16);
    if header {
        for c in 0..cols {
            if c > 0 {
                out.push(',');
            }
            out.push_str(&format!("col{c}"));
        }
        out.push('\n');
    }
    for r in 0..rows {
        if r % 23 == 0 {
            out.push_str("# comment\n");
        }
        for c in 0..cols {
            if c > 0 {
                out.push(',');
            }
            let v = (lcg(&mut s) - 0.5) * 10f64.powi((r as i32 % 7) - 3);
            out.push_str(&format!("{v:.9e}"));
        }
        out.push('\n');
    }
    out
}

fn golden(content: &str, header: bool) -> (usize, usize, u64) {
    let (h, data) = read_csv(content, ',', header).expect("read_csv");
    let mut acc = 0u64;
    for (i, row) in data.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            acc ^= v.to_bits().rotate_left(((i * 7 + j) % 64) as u32);
        }
    }
    (data.len(), h.as_ref().map_or(0, Vec::len), acc)
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(r, c, hdr) in &[(100usize, 5usize, true), (5000, 8, true), (20000, 3, false)] {
        let content = make(r, c, 1, hdr);
        let (rows, hcols, acc) = golden(&content, hdr);
        println!("req=({r}x{c},hdr={hdr}) rows={rows} hcols={hcols} data_xor_bits={acc:016x}");
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(r, c) in &[(300_000usize, 8usize), (800_000, 4)] {
        let content = make(r, c, 7, true);
        let reps = 5;
        let _ = read_csv(&content, ',', true);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += read_csv(black_box(&content), ',', true).expect("read_csv").1[0][0];
        }
        println!("rows={r} cols={c}  {:>10.3?}/call (acc={acc:.6})", t0.elapsed() / reps);
    }
}
