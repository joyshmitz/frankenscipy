// Same-process A/B for the DFT matrix build (scipy.linalg.dft). `dft_old` calls
// cos/sin in every one of the n² cells; `dft_new` precomputes the n distinct
// scaled roots of unity (m = j·k mod n in 0..n) once and indexes them. The angle
// `m` is already reduced mod n in both, so each entry is BYTE-IDENTICAL. Same
// process / same worker => no cross-worker noise; matrices must be exactly equal.
use std::time::Instant;

fn dft_old(n: usize, factor: f64) -> Vec<Vec<(f64, f64)>> {
    let mut result = vec![vec![(0.0, 0.0); n]; n];
    let base = -2.0 * std::f64::consts::PI / n as f64;
    for (j, row) in result.iter_mut().enumerate() {
        for (k, entry) in row.iter_mut().enumerate() {
            let m = (j * k) % n;
            let theta = base * m as f64;
            *entry = (theta.cos() * factor, theta.sin() * factor);
        }
    }
    result
}

fn dft_new(n: usize, factor: f64) -> Vec<Vec<(f64, f64)>> {
    let mut result = vec![vec![(0.0, 0.0); n]; n];
    let base = -2.0 * std::f64::consts::PI / n as f64;
    let roots: Vec<(f64, f64)> = (0..n)
        .map(|m| {
            let theta = base * m as f64;
            (theta.cos() * factor, theta.sin() * factor)
        })
        .collect();
    for (j, row) in result.iter_mut().enumerate() {
        for (k, entry) in row.iter_mut().enumerate() {
            *entry = roots[(j * k) % n];
        }
    }
    result
}

fn best_of(
    reps: usize,
    mut f: impl FnMut() -> Vec<Vec<(f64, f64)>>,
) -> (std::time::Duration, Vec<Vec<(f64, f64)>>) {
    let mut best = std::time::Duration::MAX;
    let mut out = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        out = std::hint::black_box(f());
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, out)
}

fn main() {
    let factor = 1.0;
    println!(
        "{:>5} {:>12} {:>12} {:>8}  {}",
        "n", "old_us", "new_us", "speedup", "exact"
    );
    for &n in &[256usize, 512, 1024, 2048] {
        let (t_old, v_old) = best_of(5, || dft_old(n, factor));
        let (t_new, v_new) = best_of(5, || dft_new(n, factor));
        let exact = v_old == v_new;
        let old_us = t_old.as_secs_f64() * 1e6;
        let new_us = t_new.as_secs_f64() * 1e6;
        println!(
            "{n:>5} {old_us:>12.2} {new_us:>12.2} {:>7.2}x  {}",
            old_us / new_us,
            if exact { "EXACT" } else { "*** DIFFER ***" }
        );
    }
}
