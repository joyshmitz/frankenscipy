//! Same-process A/B for the ODR LM-step linear solve.
//!
//! `old_solve` reproduces the original route (full Gauss-Jordan inverse, then
//! inverse·rhs); `new_solve` is the library's direct Gauss-Jordan solve of the
//! single RHS (same pivot sequence, ~half the work). We confirm the two agree to
//! within roundoff on random well-conditioned (diagonally dominant, like a damped
//! J^TJ) systems, then time the win across realistic ODR solve dimensions
//! (n = n_params + n_free_delta, one delta per data point).
//! Run: `cargo run --release -p fsci-odr --bin perf_odr_solve`.
#![allow(clippy::needless_range_loop)]

use std::time::Instant;

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

/// Verbatim original solve route: build the full inverse, then multiply.
fn invert_matrix(mut matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut inverse = vec![vec![0.0; n]; n];
    for (idx, row) in inverse.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    for pivot in 0..n {
        let best = (pivot..n).max_by(|&lhs, &rhs| {
            matrix[lhs][pivot]
                .abs()
                .total_cmp(&matrix[rhs][pivot].abs())
        })?;
        if matrix[best][pivot].abs() <= 1.0e-14 {
            return None;
        }
        matrix.swap(pivot, best);
        inverse.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for col in 0..n {
            matrix[pivot][col] /= scale;
            inverse[pivot][col] /= scale;
        }
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..n {
                matrix[row][col] -= factor * matrix[pivot][col];
                inverse[row][col] -= factor * inverse[pivot][col];
            }
        }
    }
    Some(inverse)
}

fn old_solve(normal: Vec<Vec<f64>>, rhs: &[f64]) -> Option<Vec<f64>> {
    let inverse = invert_matrix(normal)?;
    Some(
        inverse
            .iter()
            .map(|row| row.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum())
            .collect(),
    )
}

/// New direct solve — identical to the library's `gaussian_solve`.
fn new_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|&lhs, &r| matrix[lhs][pivot].abs().total_cmp(&matrix[r][pivot].abs()))?;
        if matrix[best][pivot].abs() <= 1.0e-14 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for col in 0..n {
            matrix[pivot][col] /= scale;
        }
        rhs[pivot] /= scale;
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..n {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

fn make_system(r: &mut Lcg, n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Diagonally dominant (well-conditioned), mirroring a damped J^TJ.
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut off = 0.0;
        for j in 0..n {
            if i != j {
                let v = r.unit() * 2.0 - 1.0;
                a[i][j] = v;
                off += v.abs();
            }
        }
        a[i][i] = off + 1.0 + r.unit();
    }
    let rhs: Vec<f64> = (0..n).map(|_| r.unit() * 6.0 - 3.0).collect();
    (a, rhs)
}

fn main() {
    // ---- Agreement: old (invert·rhs) vs new (direct solve) within roundoff ----
    let mut r = Lcg(0x1234_9bdf_0042_aa17);
    let mut worst = 0.0_f64;
    for _ in 0..400 {
        let n = 2 + (r.next_u64() as usize % 60);
        let (a, b) = make_system(&mut r, n);
        let x_old = old_solve(a.clone(), &b).unwrap();
        let x_new = new_solve(a, b).unwrap();
        for (o, ne) in x_old.iter().zip(&x_new) {
            let rel = (o - ne).abs() / o.abs().max(1.0);
            worst = worst.max(rel);
        }
    }
    println!("agreement old-vs-new: worst rel diff = {worst:.3e} (expect <~1e-10)");

    // ---- Timing across realistic ODR solve dimensions ----
    for &n in &[64usize, 128, 256, 512] {
        let mut rr = Lcg(0xfeed_0000_0000_0001 ^ n as u64);
        let systems: Vec<(Vec<Vec<f64>>, Vec<f64>)> =
            (0..20).map(|_| make_system(&mut rr, n)).collect();

        let mut best_old = f64::INFINITY;
        let mut best_new = f64::INFINITY;
        let mut acc = 0.0;
        for _ in 0..7 {
            let t = Instant::now();
            for (a, b) in &systems {
                acc += old_solve(a.clone(), b).unwrap()[0];
            }
            best_old = best_old.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for (a, b) in &systems {
                acc += new_solve(a.clone(), b.clone()).unwrap()[0];
            }
            best_new = best_new.min(t.elapsed().as_secs_f64());
        }
        let ratio = best_old / best_new;
        println!(
            "n={n:>4}  old={:>9.1}us  new={:>9.1}us  ratio={ratio:>5.2}x  (acc={acc:.3})",
            best_old * 1e6 / 20.0,
            best_new * 1e6 / 20.0
        );
    }
}
