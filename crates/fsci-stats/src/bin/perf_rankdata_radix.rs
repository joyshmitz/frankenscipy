// Same-binary A/B for the radix argsort in the rankdata rank engine.
use fsci_stats::{RANKDATA_RADIX_DISABLE, rankdata};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn time_it(data: &[f64], method: &str, reps: u32) -> f64 {
    let _ = black_box(rankdata(data, Some(method)).unwrap());
    let t = Instant::now();
    for _ in 0..reps {
        let _ = black_box(rankdata(black_box(data), Some(method)).unwrap());
    }
    t.elapsed().as_secs_f64() / reps as f64 * 1000.0
}

fn main() {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for &n in &[50_000usize, 200_000, 1_000_000, 4_000_000] {
        // Mix: mostly-distinct + some ties (mod), realistic rankdata input.
        let data: Vec<f64> = (0..n)
            .map(|i| {
                if i % 8 == 0 {
                    (i % 1000) as f64
                } else {
                    r() * 1e3 - 500.0
                }
            })
            .collect();
        for method in ["average", "ordinal"] {
            let reps = if n <= 200_000 { 8 } else { 3 };
            RANKDATA_RADIX_DISABLE.store(true, Ordering::Relaxed);
            let base1 = time_it(&data, method, reps);
            RANKDATA_RADIX_DISABLE.store(false, Ordering::Relaxed);
            let rdx1 = time_it(&data, method, reps);
            RANKDATA_RADIX_DISABLE.store(true, Ordering::Relaxed);
            let base2 = time_it(&data, method, reps);
            RANKDATA_RADIX_DISABLE.store(false, Ordering::Relaxed);
            let rdx2 = time_it(&data, method, reps);
            let base = base1.min(base2);
            let rdx = rdx1.min(rdx2);
            println!(
                "n={n:8} {method:8}: sort {base:8.2}ms -> radix {rdx:8.2}ms = {:.2}x",
                base / rdx
            );
        }
    }
}
