//! Timing + bit-identity harness for `nnls` (non-negative least squares).
//!
//! The Lawson-Hanson active-set loop rebuilt the passive-set normal equations
//! `AᵀA` (O(p²·m)) and `Aᵀb` (O(p·m)) FROM SCRATCH on every inner solve. The
//! library now precomputes the full Gram matrix `G = AᵀA` and `Aᵀb` once and
//! GATHERS the passive submatrix per iteration. Each gathered entry equals the
//! same `Σ_i` accumulated in the same `for i in 0..m` order, so the solution `x`
//! and residual are bit-identical; only the redundant per-iteration rebuild is
//! removed.
//!
//! This dumps golden (x, residual) bits for small fixed problems (compare across
//! the stashed pre-change build) and times large overdetermined problems.
//! Run: `cargo run --profile release-perf -p fsci-opt --bin perf_nnls`.

use std::hint::black_box;
use std::time::Instant;

use fsci_opt::nnls;

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

// Random well-conditioned overdetermined A (m x n) and target b. A nonnegative
// "ground truth" x makes the active set genuinely explore the passive set.
fn problem(m: usize, n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut g = Lcg(seed);
    let a: Vec<Vec<f64>> = (0..m)
        .map(|_| (0..n).map(|_| g.unit() * 2.0 - 1.0).collect())
        .collect();
    // b = A x_true + small noise, x_true >= 0
    let x_true: Vec<f64> = (0..n).map(|_| g.unit()).collect();
    let b: Vec<f64> = a
        .iter()
        .map(|row| {
            let dot: f64 = row.iter().zip(&x_true).map(|(&aij, &xj)| aij * xj).sum();
            dot + (g.unit() - 0.5) * 0.01
        })
        .collect();
    (a, b)
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(m, n, seed) in &[(20usize, 8usize, 1u64), (60, 20, 2), (200, 40, 3)] {
        let (a, b) = problem(m, n, seed);
        let (x, res) = nnls(&a, &b).expect("nnls");
        let xbits: u64 = x.iter().fold(0u64, |h, &v| h.rotate_left(7) ^ v.to_bits());
        println!(
            "m={m} n={n} seed={seed} xhash={xbits:016x} res={:.17e} resbits={:016x}",
            res,
            res.to_bits()
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &(m, n) in &[(2000usize, 80usize), (4000, 120), (8000, 160)] {
        let (a, b) = problem(m, n, 7);
        let reps = 5;
        let _ = nnls(&a, &b);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += nnls(black_box(&a), black_box(&b)).unwrap().1;
        }
        println!(
            "m={m} n={n}  {:>10.3?}/call (acc={acc:.6})",
            t0.elapsed() / reps
        );
    }
}
