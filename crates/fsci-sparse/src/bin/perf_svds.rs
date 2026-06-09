//! Timing + parity harness for svds: power-iteration-on-AᵀA -> shared Lanczos/
//! Arnoldi Krylov on AᵀA. Rectangular sparse with planted well-separated singular
//! values. Both methods converge to the same top-k singular values
//! (tolerance-parity); Krylov uses far fewer operator applications.
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_svds`.

use std::time::Instant;

use fsci_sparse::{CooMatrix, CsrMatrix, EigsOptions, FormatConvertible, Shape2D, svds};

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

// Rectangular m×n (m>=n) sparse with planted separated singular values: the first
// 12 diagonal entries are large and distinct, the rest small, + small bands.
fn rect_separated(m: usize, n: usize, seed: u64) -> CsrMatrix {
    let mut g = Lcg(seed);
    let offsets = [1usize, 5, 19];
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        let diag = if i < 12 {
            100.0 - 12.0 * i as f64
        } else {
            g.unit()
        };
        rows.push(i);
        cols.push(i);
        data.push(diag);
        for &off in &offsets {
            if i + off < n {
                rows.push(i);
                cols.push(i + off);
                data.push((g.unit() * 2.0 - 1.0) * 0.1);
            }
        }
    }
    // a few extra rows (n..m) with small random entries to make it tall
    for i in n..m {
        let c = (g.next_u64() as usize) % n;
        rows.push(i);
        cols.push(c);
        data.push((g.unit() * 2.0 - 1.0) * 0.1);
    }
    CooMatrix::from_triplets(Shape2D::new(m, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// ||A v - sigma u|| for a singular triplet.
fn sv_residual(a: &CsrMatrix, sigma: f64, u: &[f64], v: &[f64]) -> f64 {
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    let mut s = 0.0;
    for i in 0..a.shape().rows {
        let mut av = 0.0;
        for idx in indptr[i]..indptr[i + 1] {
            av += data[idx] * v[indices[idx]];
        }
        s += (av - sigma * u[i]).powi(2);
    }
    s.sqrt()
}

fn main() {
    for &(m, n, k) in &[
        (2200usize, 2000usize, 6usize),
        (8200, 8000, 6),
        (20200, 20000, 8),
    ] {
        let a = rect_separated(m, n, 0xABCD ^ n as u64);
        let t0 = Instant::now();
        let r = svds(&a, k, EigsOptions::default()).expect("svds");
        let dt = t0.elapsed();
        let mut sv = r.singular_values.clone();
        sv.sort_by(|a, b| b.total_cmp(a));
        let max_resid = r
            .singular_values
            .iter()
            .zip(r.u.iter())
            .zip(r.vt.iter())
            .map(|((&s, u), v)| sv_residual(&a, s, u, v))
            .fold(0.0_f64, f64::max);
        println!(
            "m={m:>6} n={n:>6} k={k} {:>10.3?}  top3=[{:.6},{:.6},{:.6}]  max_resid={max_resid:.2e}",
            dt,
            sv.first().copied().unwrap_or(0.0),
            sv.get(1).copied().unwrap_or(0.0),
            sv.get(2).copied().unwrap_or(0.0),
        );
    }
}
