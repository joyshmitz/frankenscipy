//! Same-process A/B for 1D node duplicate-check elimination.
//! BarycentricInterpolator::new fuses the O(n²) dup-scan into its O(n²) weight loop
//! (bit-identical weights). FloaterHormannInterpolator::new replaces the O(n²) dup-scan
//! with an O(n log n) sort+adjacent check (its weight pass is only O(n·d), so the scan
//! dominated). The eval-over-grid digest must match the all-pairs path exactly. Run via
//! stash A/B. Run: `cargo run --release -p fsci-interpolate --bin perf_bary_fh`.
use fsci_interpolate::{BarycentricInterpolator, FloaterHormannInterpolator};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

// Strictly increasing, well-separated nodes on [0, 10] (no duplicates -> accept path).
fn nodes(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Lcg(seed);
    let mut x = 0.0;
    let xi: Vec<f64> = (0..n)
        .map(|_| {
            x += 0.1 + r.unit() * 0.05;
            x
        })
        .collect();
    let yi: Vec<f64> = xi.iter().map(|&v| (v * 0.3).sin() + 0.2 * v).collect();
    (xi, yi)
}

fn digest_vals(v: &[f64]) -> u64 {
    let mut h = 1469598103934665603u64;
    for &x in v {
        h = (h ^ x.to_bits()).wrapping_mul(1099511628211);
    }
    h
}

fn main() {
    let queries: Vec<f64> = (0..2000).map(|i| 0.05 + i as f64 * 0.01).collect();

    println!("== BarycentricInterpolator::new (fused dup-check into O(n²) weight loop) ==");
    for &n in &[200usize, 500, 1000] {
        let (xi, yi) = nodes(n, 0xbac0_0000_0001 ^ n as u64);
        let interp = BarycentricInterpolator::new(&xi, &yi).expect("bary");
        let dig = digest_vals(&interp.eval_many(&queries));
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let out = BarycentricInterpolator::new(&xi, &yi).expect("bary");
            std::hint::black_box(&out);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("n={n:>5}  {:>10.1} us  eval_digest={dig:016x}", best * 1e6);
    }

    println!("== FloaterHormannInterpolator::new (O(n²) dup-scan -> O(n log n) sort) ==");
    for &n in &[200usize, 500, 1000, 2000, 4000] {
        let (xi, yi) = nodes(n, 0xf40a_0000_0001 ^ n as u64);
        let interp = FloaterHormannInterpolator::new(&xi, &yi, 3).expect("fh");
        let dig = digest_vals(&interp.eval_many(&queries));
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let out = FloaterHormannInterpolator::new(&xi, &yi, 3).expect("fh");
            std::hint::black_box(&out);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("n={n:>5}  {:>10.1} us  eval_digest={dig:016x}", best * 1e6);
    }
}
