//! Timing harness for the parallel rank/median filter core
//! (`rank_filter_index_usize_axes_with_origins`, now via `fill_pixels_parallel`).
//! Byte-identity is covered by `perf_rank_filter` (golden digests). This times the
//! large-image median (middle-rank) filter; compare across the stashed serial build.
//! Run: `cargo run --profile release-perf -p fsci-ndimage --bin perf_rank_filter_timing`.

use std::hint::black_box;
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, rank_filter_axes};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    // rank_filter_axes (axes subset) routes through the now-parallel
    // rank_filter_index_usize_axes_with_origins core.
    for &(r, c, size) in &[
        (1500usize, 1500usize, 31usize),
        (2000, 2000, 41),
        (3000, 3000, 21),
    ] {
        let mut s = 7u64;
        let a = NdArray::new((0..r * c).map(|_| lcg(&mut s)).collect(), vec![r, c]).unwrap();
        let rank = (size / 2) as isize; // median along axis 0
        let reps = 3;
        let _ = rank_filter_axes(&a, rank, size, &[0], BoundaryMode::Reflect, 0.0);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let out = rank_filter_axes(black_box(&a), rank, size, &[0], BoundaryMode::Reflect, 0.0)
                .unwrap();
            acc += out.data[0];
        }
        println!(
            "r={r} c={c} size={size} axis0-median  {:>10.3?}/call (acc={acc:.4})",
            t0.elapsed() / reps
        );
    }
}
