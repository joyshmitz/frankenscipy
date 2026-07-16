use fsci_signal::{LOMBSCARGLE_FUSED_DISABLE, lombscargle};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let mut seed = 12345u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for &(n, m) in &[(500usize, 2000usize), (2000, 4000), (5000, 8000)] {
        let mut t = 0.0;
        let x: Vec<f64> = (0..n)
            .map(|_| {
                t += 0.5 + r();
                t
            })
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (2.3 * xi).sin() + 0.5 * (5.1 * xi).cos() + 0.1 * (r() - 0.5))
            .collect();
        let freqs: Vec<f64> = (0..m).map(|k| 0.01 + 6.0 * k as f64 / m as f64).collect();

        let run = |disable: bool| {
            LOMBSCARGLE_FUSED_DISABLE.store(disable, Ordering::Relaxed);
            lombscargle(&x, &y, &freqs, true).unwrap()
        };
        let fused = run(false);
        let twopass = run(true);
        let maxdiff = fused
            .iter()
            .zip(&twopass)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let maxrel = fused
            .iter()
            .zip(&twopass)
            .map(|(a, b)| {
                if b.abs() > 1e-12 {
                    (a - b).abs() / b.abs()
                } else {
                    0.0
                }
            })
            .fold(0.0f64, f64::max);

        let bench = |disable: bool| {
            LOMBSCARGLE_FUSED_DISABLE.store(disable, Ordering::Relaxed);
            let _ = black_box(lombscargle(&x, &y, &freqs, true).unwrap());
            let reps = 6;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(
                    lombscargle(black_box(&x), black_box(&y), black_box(&freqs), true).unwrap(),
                );
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let (mut fu, mut tp) = (f64::MAX, f64::MAX);
        for _ in 0..4 {
            tp = tp.min(bench(true));
            fu = fu.min(bench(false));
        }
        println!(
            "n={n} m={m}: two-pass {tp:.2}ms  fused {fu:.2}ms  speedup {:.2}x  maxdiff={maxdiff:.1e} maxrel={maxrel:.1e}",
            tp / fu
        );
    }
}
