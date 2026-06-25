// Same-process A/B for the centered-discrepancy upper-triangle symmetry lever.
// `old_full_centered` is the pre-change full-n^2 double loop; the library
// `centered_discrepancy` now sums the diagonal once + each off-diagonal pair
// twice (the pair product is bit-symmetric). Measured d>=3 so BOTH paths use the
// general kernel and the only difference is the symmetry (the d==2 fast path
// receives the identical transform). Same process / same worker => no
// cross-worker noise; values must agree to ~1e-9 (sum reassociation only).
use fsci_stats::centered_discrepancy;
use std::time::Instant;

fn old_full_centered(sample: &[f64], dimension: usize) -> f64 {
    let n = sample.len() / dimension;
    let leading = (13.0_f64 / 12.0).powi(dimension as i32);
    let mut single = 0.0_f64;
    for i in 0..n {
        let mut prod = 1.0_f64;
        for k in 0..dimension {
            let centered = sample[i * dimension + k] - 0.5;
            prod *= 1.0 + 0.5 * centered.abs() - 0.5 * centered * centered;
        }
        single += prod;
    }
    let mut double = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut prod = 1.0_f64;
            for k in 0..dimension {
                let xi = sample[i * dimension + k];
                let xj = sample[j * dimension + k];
                prod *=
                    1.0 + 0.5 * (xi - 0.5).abs() + 0.5 * (xj - 0.5).abs() - 0.5 * (xi - xj).abs();
            }
            double += prod;
        }
    }
    let n_f = n as f64;
    leading - 2.0 / n_f * single + double / (n_f * n_f)
}

fn build_sample(n: usize, d: usize) -> Vec<f64> {
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    (0..n * d)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

fn best_of(reps: usize, mut f: impl FnMut() -> f64) -> (std::time::Duration, f64) {
    let mut best = std::time::Duration::MAX;
    let mut val = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        val = std::hint::black_box(f());
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, val)
}

fn main() {
    println!(
        "{:>6} {:>3} {:>12} {:>12} {:>9}  {:>10}",
        "n", "d", "old_us", "new_us", "speedup", "valdiff"
    );
    for &(n, d) in &[(512usize, 3usize), (512, 6), (1024, 4), (2048, 4)] {
        let s = build_sample(n, d);
        let (t_old, v_old) = best_of(5, || old_full_centered(&s, d));
        let (t_new, v_new) = best_of(5, || centered_discrepancy(&s, d).unwrap());
        let old_us = t_old.as_secs_f64() * 1e6;
        let new_us = t_new.as_secs_f64() * 1e6;
        println!(
            "{n:>6} {d:>3} {old_us:>12.2} {new_us:>12.2} {:>8.2}x  {:>10.2e}",
            old_us / new_us,
            (v_old - v_new).abs()
        );
    }
}
