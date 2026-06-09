//! Timing + byte-identity harness for the parallel direct 2D correlation path.
//! Large image with a small kernel keeps `correlate2d` on the direct loop (FFT is
//! not chosen there); the loop is now parallelized over output rows. Each thread
//! owns disjoint output rows and iterates the same (i,j,ki,kj) order, so every
//! output cell accumulates identically => byte-identical to the serial scatter.
//! Dump an FNV hash of the full output (compare across the stashed serial build)
//! and time it.
//! Run: `cargo run --profile release-perf -p fsci-signal --bin perf_correlate2d_par`.

use std::hint::black_box;
use std::time::Instant;

use fsci_signal::{ConvolveMode, correlate2d};

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn fnv(out: &[f64]) -> u64 {
    out.iter().fold(1469598103934665603u64, |h, &v| {
        (h ^ v.to_bits()).wrapping_mul(1099511628211)
    })
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(ar, ac, vr, vc) in &[(1200usize, 1200usize, 9usize, 9usize), (2000, 2000, 7, 7)] {
        let mut g = Lcg(0x51 ^ ar as u64);
        let a: Vec<f64> = (0..ar * ac).map(|_| g.unit() * 2.0 - 1.0).collect();
        let v: Vec<f64> = (0..vr * vc).map(|_| g.unit() * 2.0 - 1.0).collect();
        let out = correlate2d(&a, (ar, ac), &v, (vr, vc), ConvolveMode::Full).unwrap();
        println!(
            "ar={ar} ac={ac} vr={vr} vc={vc} len={} hash={:016x}",
            out.len(),
            fnv(&out)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(ar, ac, vr, vc) in &[
        (1500usize, 1500usize, 11usize, 11usize),
        (2000, 2000, 15, 15),
        (3000, 3000, 9, 9),
    ] {
        let mut g = Lcg(7);
        let a: Vec<f64> = (0..ar * ac).map(|_| g.unit() * 2.0 - 1.0).collect();
        let v: Vec<f64> = (0..vr * vc).map(|_| g.unit() * 2.0 - 1.0).collect();
        let reps = 3;
        let _ = correlate2d(&a, (ar, ac), &v, (vr, vc), ConvolveMode::Same);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let out = correlate2d(
                black_box(&a),
                (ar, ac),
                black_box(&v),
                (vr, vc),
                ConvolveMode::Same,
            )
            .unwrap();
            acc += out[0];
        }
        println!(
            "ar={ar} ac={ac} kernel={vr}x{vc}  {:>10.3?}/call (acc={acc:.3})",
            t0.elapsed() / reps
        );
    }
}
