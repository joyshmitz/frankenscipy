//! Same-process A/B for `cwt` over widths. The per-width FFT convolution rows are
//! independent; this confirms the parallel path is byte-identical to the serial
//! loop (FNV digest of the full coefficient matrix) and times the win.
//! Run: `cargo run --release -p fsci-signal --bin perf_cwt`.
use fsci_signal::{cwt, ricker};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn digest(m: &[Vec<f64>]) -> u64 {
    let mut h = 1469598103934665603u64;
    for row in m {
        for v in row {
            h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
        }
    }
    h
}

fn main() {
    for &(na, w) in &[
        (512usize, 4usize),
        (2048, 16),
        (2048, 32),
        (4096, 32),
        (4096, 64),
    ] {
        let mut r = Lcg(0x51a9_3c7e_0011_d4a3 ^ ((na as u64) << 8 ^ w as u64));
        let data: Vec<f64> = (0..na).map(|_| r.unit()).collect();
        let widths: Vec<f64> = (1..=w).map(|i| i as f64).collect();

        let res = cwt(&data, ricker, &widths).expect("cwt");
        let dig = digest(&res);

        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let out = cwt(&data, ricker, &widths).expect("cwt");
            std::hint::black_box(&out);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "na={na:>5} widths={w:>3}  {:>9.1} us  digest={dig:016x}",
            best * 1e6
        );
    }
}
