//! Profiling-only harness for the dense linear-solve hot path.
//!
//! NOT a product binary — it exists so `samply`/`perf`/`hyperfine` have a
//! tight, deterministic scenario to attach to (see
//! `tests/artifacts/perf/`). Build under the `release-perf` profile:
//!
//! ```bash
//! RUSTFLAGS="-C force-frame-pointers=yes" \
//!   cargo build -p fsci-linalg --profile release-perf --bin perf_solve
//! ```
//!
//! Usage: `perf_solve <mode> <n> <repeats> [seed]`
//!   mode    = solve | lu_factor | lu_solve   (which stage to exercise)
//!   n       = matrix dimension
//!   repeats = number of timed iterations
//!
//! Prints one JSON line with per-call timing so it doubles as a wall-clock
//! probe. Matrix is deterministic (diagonally dominant => well-conditioned,
//! so CASP takes the DirectLU fast path and we profile the kernel, not the
//! fallback ladder).

use std::hint::black_box;
use std::time::Instant;

use fsci_linalg::{lu_factor, lu_solve, solve, DecompOptions, SolveOptions};

/// Deterministic diagonally-dominant matrix; no RNG crate so the workload is
/// byte-for-byte reproducible across runs/hosts.
fn make_matrix(n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        // SplitMix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) as f64) / (u64::MAX as f64) - 0.5
    };
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut rowsum = 0.0;
        for j in 0..n {
            let v = next();
            a[i][j] = v;
            if i != j {
                rowsum += v.abs();
            }
        }
        // Force diagonal dominance for a well-conditioned system.
        a[i][i] = rowsum + 1.0;
    }
    a
}

fn make_rhs(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed ^ 0xDEAD_BEEF;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("solve");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let repeats: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(42);

    let a = make_matrix(n, seed);
    let b = make_rhs(n, seed);

    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..repeats {
        match mode {
            "solve" => {
                let r = solve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap();
                checksum += r.x.iter().sum::<f64>();
            }
            "lu_factor" => {
                // LuFactorResult fields are opaque (wraps nalgebra LU); just
                // keep the result live so the factorization isn't elided.
                let r = lu_factor(black_box(&a), DecompOptions::default()).unwrap();
                black_box(&r);
                checksum += 1.0;
            }
            "lu_solve" => {
                let f = lu_factor(&a, DecompOptions::default()).unwrap();
                let r = lu_solve(black_box(&f), black_box(&b)).unwrap();
                checksum += r.x.iter().sum::<f64>();
            }
            other => {
                eprintln!("unknown mode: {other}");
                std::process::exit(2);
            }
        }
    }
    let elapsed = t0.elapsed();
    let per_call_ms = elapsed.as_secs_f64() * 1e3 / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"n\":{n},\"repeats\":{repeats},\"total_ms\":{:.3},\"per_call_ms\":{:.4},\"checksum\":{:.6e}}}",
        elapsed.as_secs_f64() * 1e3,
        per_call_ms,
        checksum
    );
}
