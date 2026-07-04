//! Same-process A/B + isomorphism harness for the structured ODR solver.
//!
//! `run()` is the new structured path (diagonal delta block + Schur-eliminated
//! LM step + Schur-complement covariance, O(n·p²) per iteration).
//! `run_dense_reference()` is the retained pre-structural path (full
//! (p+n)-dimensional finite-difference Jacobian + normal equations, O(n³)).
//!
//! We prove the two agree on beta / delta / covariance to convergence tolerance
//! across random scalar-x ODR problems (both ODR and OLS jobs, with and without
//! fixed parameters and per-point weights), then time the win end-to-end.
//! Run: `cargo run --release -p fsci-odr --bin perf_odr_structured`.
#![allow(clippy::needless_range_loop)]

use fsci_odr::{Data, Model, ODR};
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

fn make_model() -> Model {
    // Smooth nonlinear scalar model: y = b0 + b1*x + b2*sin(b3*x).
    Model::new(|b: &[f64], x: &[f64]| -> Vec<f64> {
        x.iter()
            .map(|&xi| b[0] + b[1] * xi + b[2] * (b[3] * xi).sin())
            .collect()
    })
    .with_parameter_count(4)
    .with_scalar_separable(true)
}

fn build_problem(r: &mut Lcg, n: usize) -> (Vec<f64>, Vec<f64>) {
    let beta = [0.7, 1.3, 0.4, 0.9];
    let x: Vec<f64> = (0..n).map(|k| k as f64 * 0.05 + r.unit() * 0.01).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            beta[0] + beta[1] * xi + beta[2] * (beta[3] * xi).sin() + (r.unit() - 0.5) * 0.05
        })
        .collect();
    (x, y)
}

fn main() {
    // ---- Isomorphism: structured run() vs dense reference ----
    let mut r = Lcg(0x0dd_face_0bad_c0de);
    let mut worst_beta = 0.0_f64;
    let mut worst_delta = 0.0_f64;
    let mut worst_cov = 0.0_f64;
    let mut worst_ss = 0.0_f64;
    let mut cases = 0usize;
    for trial in 0..240 {
        let n = 6 + (r.next_u64() as usize % 40);
        let (x, y) = build_problem(&mut r, n);
        let beta0 = vec![0.5, 1.5, 0.2, 0.8];
        let data = Data::new(x, y).unwrap();
        let mut odr = ODR::new(data, make_model(), beta0).unwrap();
        // Vary the job across trials: OLS on every 4th case, fixed beta on every 3rd.
        if trial % 4 == 0 {
            odr.set_job(fsci_odr::FitType::Ols);
        }
        let odr = if trial % 3 == 0 {
            odr.with_beta_free(vec![true, false, true, true]).unwrap()
        } else {
            odr
        };

        let new = odr.run().unwrap();
        let old = odr.run_dense_reference().unwrap();
        cases += 1;

        for (a, b) in new.beta.iter().zip(&old.beta) {
            worst_beta = worst_beta.max((a - b).abs() / b.abs().max(1.0));
        }
        for (a, b) in new.delta.iter().zip(&old.delta) {
            worst_delta = worst_delta.max((a - b).abs() / b.abs().max(1.0));
        }
        for (ra, rb) in new.cov_beta.iter().zip(&old.cov_beta) {
            for (a, b) in ra.iter().zip(rb) {
                worst_cov = worst_cov.max((a - b).abs() / b.abs().max(1.0));
            }
        }
        worst_ss = worst_ss.max((new.sum_square - old.sum_square).abs() / old.sum_square.max(1.0));
    }
    println!("isomorphism over {cases} random scalar-x fits (structured vs dense reference):");
    println!("  worst rel beta diff   = {worst_beta:.3e}");
    println!("  worst rel delta diff  = {worst_delta:.3e}");
    println!("  worst rel cov diff    = {worst_cov:.3e}");
    println!("  worst rel sumsq diff  = {worst_ss:.3e}  (expect all <~1e-6)");

    // ---- Timing: end-to-end fit, structured vs dense reference ----
    println!("\nend-to-end fit time (structured run vs dense reference):");
    for &n in &[100usize, 200, 400, 800] {
        let mut rr = Lcg(0xbeef_0000_0000_0001 ^ n as u64);
        let (x, y) = build_problem(&mut rr, n);
        let beta0 = vec![0.5, 1.5, 0.2, 0.8];
        let reps = if n <= 200 { 20 } else { 5 };

        let mk = || {
            ODR::new(
                Data::new(x.clone(), y.clone()).unwrap(),
                make_model(),
                beta0.clone(),
            )
            .unwrap()
        };

        let mut acc = 0.0;
        let t = Instant::now();
        for _ in 0..reps {
            acc += mk().run_dense_reference().unwrap().beta[1];
        }
        let old_t = t.elapsed().as_secs_f64() / reps as f64;

        let t = Instant::now();
        let mut nit = 0;
        for _ in 0..reps {
            let o = mk().run().unwrap();
            acc += o.beta[1];
            nit = o.nit;
        }
        let new_t = t.elapsed().as_secs_f64() / reps as f64;

        println!(
            "  n={n:>4}  dense={:>9.2}ms  structured={:>8.3}ms  speedup={:>6.1}x  nit={nit} (acc={acc:.3})",
            old_t * 1e3,
            new_t * 1e3,
            old_t / new_t
        );
    }
}
