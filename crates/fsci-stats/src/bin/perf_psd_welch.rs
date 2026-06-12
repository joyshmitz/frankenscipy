//! Timing + bit-identity harness for `psd_welch`.
//!
//! Each frequency bin's periodogram is an independent O(window) DFT dot summed
//! over the (windowed) segments. The bins fan out across threads in contiguous
//! chunks (one spawn-set); each bin accumulates over the segments in the same
//! order as the serial loop, so values are bit-identical. This dumps an FNV
//! digest of the PSD (compare across the stashed serial build) and times the
//! large-(window, segments) win.
//! Run: `cargo run --profile release-perf -p fsci-stats --bin perf_psd_welch`.

use std::hint::black_box;
use std::time::Instant;

use fsci_stats::psd_welch;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn signal(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s)).collect()
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, &x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, w, ov) in &[
        (4096usize, 6usize, 3usize),
        (8192, 256, 128),
        (16384, 512, 256),
    ] {
        let d = signal(n, 7);
        println!(
            "n={n} w={w} digest={:016x}",
            digest(&psd_welch(&d, w, ov, 100.0))
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(n, w, ov) in &[
        (40000usize, 512usize, 256usize),
        (80000, 1024, 512),
        (120000, 2048, 1024),
    ] {
        let d = signal(n, 7);
        let reps = 5;
        let _ = psd_welch(&d, w, ov, 100.0);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let p = psd_welch(black_box(&d), w, ov, 100.0);
            acc += p[p.len() / 2];
        }
        println!(
            "n={n} w={w}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
