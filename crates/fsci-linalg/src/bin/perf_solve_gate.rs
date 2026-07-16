// A/B: does routing medium-n solve through the fast mixed-precision blocked-LU
// path (override gate low) beat the slow portfolio path (default gate 1000)?
use fsci_linalg::{SOLVE_FLAT_MIN_OVERRIDE, SolveOptions, solve};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut seed = 11u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[32usize, 64, 96, 128, 192, 256] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let b: Vec<f64> = (0..n).map(|_| r()).collect();
        // accuracy: portfolio x vs fast x
        SOLVE_FLAT_MIN_OVERRIDE.store(0, Ordering::Relaxed);
        let xp = solve(&a, &b, SolveOptions::default()).unwrap().x;
        SOLVE_FLAT_MIN_OVERRIDE.store(128, Ordering::Relaxed);
        let xf = solve(&a, &b, SolveOptions::default()).unwrap().x;
        let maxdiff = xp
            .iter()
            .zip(&xf)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let reps = 5;
        let bench = |ov: usize| {
            SOLVE_FLAT_MIN_OVERRIDE.store(ov, Ordering::Relaxed);
            let _ = black_box(solve(&a, &b, SolveOptions::default()).unwrap());
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(
                    solve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap(),
                );
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let portfolio = bench(0).min(bench(0));
        let fast = bench(128).min(bench(128));
        println!(
            "solve n={n:4}: portfolio {portfolio:8.2}ms -> fast {fast:8.2}ms = {:.2}x  maxdiff={maxdiff:.1e}",
            portfolio / fast
        );
    }
    SOLVE_FLAT_MIN_OVERRIDE.store(0, Ordering::Relaxed);
}
