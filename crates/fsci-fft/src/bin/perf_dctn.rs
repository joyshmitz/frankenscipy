//! Timing + golden-digest harness for the N-D DCT/DST (`dctn`/`dstn`), whose
//! per-axis pass (`apply_dct_along_axis`) transforms each fiber along the axis.
//!
//! Each fiber is independent (disjoint flat indices, pure 1-D `dct`), so the
//! fiber loop parallelizes byte-identically — this dumps an FNV digest of the
//! output bits (must be unchanged across the serial/parallel builds) and times
//! the large-array win.
//! Run: `cargo run --release -p fsci-fft --bin perf_dctn`.

use std::hint::black_box;
use std::time::Instant;

use fsci_fft::{FftOptions, dctn, dstn};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn digest(v: &[f64]) -> u64 {
    v.iter().fold(1469598103934665603u64, |h, x| {
        (h ^ x.to_bits()).wrapping_mul(1099511628211)
    })
}

fn bench(label: &str, shape: &[usize], dst: bool) {
    let total: usize = shape.iter().product();
    let mut s = 0x1234_5678_9abc_def0u64;
    let data: Vec<f64> = (0..total).map(|_| lcg(&mut s)).collect();
    let opts = FftOptions::default();

    let run = |d: &[f64]| -> Vec<f64> {
        if dst {
            dstn(d, shape, &opts).unwrap()
        } else {
            dctn(d, shape, &opts).unwrap()
        }
    };

    let out = run(&data);
    let dig = digest(&out);

    let trials = 5;
    let mut t = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t0 = Instant::now();
        black_box(run(&data));
        t.push(t0.elapsed().as_secs_f64());
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{label} shape={shape:?} median={:.2} ms  GOLDEN digest={dig:016x}",
        t[trials / 2] * 1e3
    );
}

fn main() {
    bench("dctn 2D", &[2048, 2048], false);
    bench("dctn 3D", &[160, 160, 160], false);
    bench("dstn 2D", &[2048, 2048], true);
}
