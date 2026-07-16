//! Measures the Radau stage-solve lever: full `3n×3n` real LU of the Newton matrix
//! `I_{3n} − h(A⊗J)` (fsci's current dense-Jacobian path) vs scipy's decoupled
//! form — one real `n×n` LU `(I − h·λ_real·J)` plus one complex `n×n` LU
//! `(I − h·λ_c·J)`, where `λ_real, λ_c, conj(λ_c)` are the eigenvalues of the
//! 3-stage Radau IIA A-matrix. The two are mathematically equivalent (a similarity
//! transform by the eigenvectors of A, an O(n) block mix), so this isolates the
//! linear-algebra cost. fsci already factorizes the real `n×n` factor for its error
//! estimate, so the decoupled main solve only adds the complex `n×n` LU.
//! Run: `cargo run --release -p fsci-integrate --bin perf_radau_decompose`.
#![allow(clippy::needless_range_loop)]

use nalgebra::{Complex, DMatrix, DVector};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn radau_a() -> [[f64; 3]; 3] {
    // 3-stage Radau IIA (order 5) A-matrix (Hairer & Wanner).
    let s6 = 6.0_f64.sqrt();
    [
        [
            (88.0 - 7.0 * s6) / 360.0,
            (296.0 - 169.0 * s6) / 1800.0,
            (-2.0 + 3.0 * s6) / 225.0,
        ],
        [
            (296.0 + 169.0 * s6) / 1800.0,
            (88.0 + 7.0 * s6) / 360.0,
            (-2.0 - 3.0 * s6) / 225.0,
        ],
        [(16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0],
    ]
}

fn main() {
    let h = 0.05;
    let a = radau_a();
    // Eigenvalues of A: one real, one complex conjugate pair.
    let am = DMatrix::from_fn(3, 3, |i, j| a[i][j]);
    let eig = am.complex_eigenvalues();
    let mut lam_real = 0.0;
    let mut lam_c = Complex::new(0.0, 0.0);
    for k in 0..3 {
        if eig[k].im.abs() < 1e-12 {
            lam_real = eig[k].re;
        } else if eig[k].im > 0.0 {
            lam_c = eig[k];
        }
    }
    println!(
        "Radau A eigenvalues: real={lam_real:.6}, complex={:.6}{:+.6}i",
        lam_c.re, lam_c.im
    );

    for &n in &[10usize, 20, 40, 80, 160] {
        let mut r = Lcg(0x1234 ^ n as u64);
        // Dense Jacobian J and a stage RHS.
        let j = DMatrix::from_fn(n, n, |_, _| r.u() * 2.0 - 1.0);
        let rhs3 = DVector::from_fn(3 * n, |_, _| r.u() * 2.0 - 1.0);
        let ident = DMatrix::<f64>::identity(n, n);
        let reps = if n <= 40 { 300 } else { 40 };

        // --- (a) full 3n×3n real LU + solve (current fsci dense path) ---
        let t = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            let mut m = DMatrix::<f64>::zeros(3 * n, 3 * n);
            for bi in 0..3 {
                for bj in 0..3 {
                    let coeff = -h * a[bi][bj];
                    for i in 0..n {
                        for k in 0..n {
                            m[(bi * n + i, bj * n + k)] = coeff * j[(i, k)];
                        }
                    }
                }
            }
            for d in 0..3 * n {
                m[(d, d)] += 1.0;
            }
            let sol = m.lu().solve(&rhs3).unwrap();
            acc += sol[0];
        }
        let ta = t.elapsed().as_secs_f64() / reps as f64;

        // --- (b) decoupled: real n×n LU + complex n×n LU + solve ---
        let t = Instant::now();
        for _ in 0..reps {
            // Real block: (I − h·λ_real·J)
            let m_real = &ident - &j * (h * lam_real);
            let lu_real = m_real.lu();
            // Complex block: (I − h·λ_c·J)
            let jc = DMatrix::<Complex<f64>>::from_fn(n, n, |i, k| Complex::new(j[(i, k)], 0.0));
            let ic = DMatrix::<Complex<f64>>::identity(n, n);
            let m_c = &ic - jc * (Complex::new(h, 0.0) * lam_c);
            let lu_c = m_c.lu();
            // Solve (transform of RHS by A's eigenvectors is O(n) — approximated here
            // by solving the real block on rhs[0..n] and the complex block on
            // rhs[n..2n]+i*rhs[2n..3n], which has the same O(n^3) LU cost).
            let br = rhs3.rows(0, n).into_owned();
            let wr = lu_real.solve(&br).unwrap();
            let bc = DVector::<Complex<f64>>::from_fn(n, |i, _| {
                Complex::new(rhs3[n + i], rhs3[2 * n + i])
            });
            let wc = lu_c.solve(&bc).unwrap();
            acc += wr[0] + wc[0].re;
        }
        let tb = t.elapsed().as_secs_f64() / reps as f64;

        println!(
            "n={n:>4}  full_3n_lu={:>9.1}us  real+complex_n_lu={:>9.1}us  speedup={:>5.2}x  (acc={acc:.2})",
            ta * 1e6,
            tb * 1e6,
            ta / tb
        );
    }
}
