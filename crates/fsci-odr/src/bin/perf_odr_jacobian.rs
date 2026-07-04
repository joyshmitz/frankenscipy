//! Same-process A/B + isomorphism harness for the ODR finite-difference Jacobian.
//!
//! The ODR least-squares variable vector packs the model parameters `beta`
//! (p entries) AND one x-error `delta` per data point (n entries). The residual
//! vector is `[ sqrt(we)·(y - f(beta, x+delta)) ; sqrt(wd)·delta ]` (length 2n).
//!
//! `old_jac` reproduces the library's dense finite-difference Jacobian: it
//! perturbs each of the (p+n) variables and recomputes the FULL residual vector
//! (a full n-point model evaluation) per column -> O((p+n)·n) point evaluations.
//!
//! `new_jac` exploits ODR's fundamental observation-independence: perturbing
//! delta[j] only changes observation j, so the entire delta block of the
//! Jacobian is one diagonal (response rows) plus an analytic sqrt(wd) penalty
//! (delta rows). We get every diagonal entry from a SINGLE batched model
//! evaluation with the whole delta vector perturbed at once -> O((p+1)·n).
//!
//! We prove the two Jacobians are byte-identical (`.to_bits()`) on random
//! separable models, then time the win across realistic ODR fit sizes.
//! Run: `cargo run --release -p fsci-odr --bin perf_odr_jacobian`.
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

type Problem = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// A separable ODR model applied per observation: y_i = f(beta, x_i).
/// Mirrors what a scipy.odr model closure must satisfy (each output point
/// depends only on the matching input point).
fn model_eval(beta: &[f64], x: &[f64]) -> Vec<f64> {
    // y = b0 + b1*x + b2*sin(b3*x)  (nonlinear, exercises real arithmetic)
    x.iter()
        .map(|&xi| beta[0] + beta[1] * xi + beta[2] * (beta[3] * xi).sin())
        .collect()
}

/// Full weighted residual vector for the packed [beta; delta] variables.
/// (All beta free, all delta free — the common full-ODR case.)
fn residuals(vars: &[f64], p: usize, x: &[f64], y: &[f64], we: &[f64], wd: &[f64]) -> Vec<f64> {
    let n = x.len();
    let beta = &vars[..p];
    let delta = &vars[p..p + n];
    let xplus: Vec<f64> = x.iter().zip(delta).map(|(a, d)| a + d).collect();
    let pred = model_eval(beta, &xplus);
    let mut r = Vec::with_capacity(2 * n);
    for i in 0..n {
        r.push(we[i].sqrt() * (y[i] - pred[i]));
    }
    for i in 0..n {
        r.push(wd[i].sqrt() * delta[i]);
    }
    r
}

/// Verbatim library route: dense finite-difference over every variable.
fn old_jac(
    vars: &[f64],
    p: usize,
    x: &[f64],
    y: &[f64],
    we: &[f64],
    wd: &[f64],
    step: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let r0 = residuals(vars, p, x, y, we, wd);
    let nvar = p + n;
    let mut jac = vec![vec![0.0; nvar]; r0.len()];
    for col in 0..nvar {
        let mut vp = vars.to_vec();
        let h = step * vars[col].abs().max(1.0);
        vp[col] += h;
        let rp = residuals(&vp, p, x, y, we, wd);
        for row in 0..r0.len() {
            jac[row][col] = (rp[row] - r0[row]) / h;
        }
    }
    jac
}

/// Structural route: beta block dense (p full evals), delta block via ONE
/// batched all-delta-perturbed eval (diagonal response rows) + analytic penalty.
fn new_jac(
    vars: &[f64],
    p: usize,
    x: &[f64],
    y: &[f64],
    we: &[f64],
    wd: &[f64],
    step: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let beta = &vars[..p];
    let delta = &vars[p..p + n];
    let xplus: Vec<f64> = x.iter().zip(delta).map(|(a, d)| a + d).collect();
    let r0 = residuals(vars, p, x, y, we, wd);
    let nvar = p + n;
    let mut jac = vec![vec![0.0; nvar]; 2 * n];

    // --- beta columns: dense finite difference (perturb a param, full eval) ---
    for col in 0..p {
        let mut vp = vars.to_vec();
        let h = step * vars[col].abs().max(1.0);
        vp[col] += h;
        let rp = residuals(&vp, p, x, y, we, wd);
        for row in 0..2 * n {
            jac[row][col] = (rp[row] - r0[row]) / h;
        }
    }

    // --- delta columns: single batched eval, all delta perturbed together ---
    // Each h_j = step*max(|delta_j|,1) can differ per column, so perturb every
    // delta by its own h and read the matching output point. Because obs j only
    // depends on x_j+delta_j, output[j] equals the one-at-a-time perturbation.
    let mut hs = vec![0.0; n];
    let mut xplus_pert = xplus.clone();
    for j in 0..n {
        let h = step * delta[j].abs().max(1.0);
        hs[j] = h;
        xplus_pert[j] += h;
    }
    let pred_pert = model_eval(beta, &xplus_pert);
    for j in 0..n {
        let col = p + j;
        // response row j: (rp - r0)/h where rp_j uses the perturbed prediction.
        let rp_resp_j = we[j].sqrt() * (y[j] - pred_pert[j]);
        jac[j][col] = (rp_resp_j - r0[j]) / hs[j];
        // delta-penalty row n+j: d/d delta_j of sqrt(wd_j)*delta_j = sqrt(wd_j),
        // matched to the same one-sided FD the dense path takes (exact here).
        let rp_pen_j = wd[j].sqrt() * (delta[j] + hs[j]);
        jac[n + j][col] = (rp_pen_j - r0[n + j]) / hs[j];
    }
    jac
}

fn jac_eq(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(ra, rb)| {
            ra.len() == rb.len() && ra.iter().zip(rb).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}

fn make_problem(r: &mut Lcg, n: usize) -> Problem {
    let p = 4;
    let beta = vec![0.7, 1.3, 0.4, 0.9];
    let x: Vec<f64> = (0..n).map(|k| k as f64 * 0.05 + r.unit() * 0.01).collect();
    let y: Vec<f64> = model_eval(&beta, &x)
        .iter()
        .map(|&v| v + (r.unit() - 0.5) * 0.02)
        .collect();
    let we: Vec<f64> = (0..n).map(|_| 0.5 + r.unit()).collect();
    let wd: Vec<f64> = (0..n).map(|_| 0.5 + r.unit()).collect();
    // packed vars = [beta; delta], delta small nonzero
    let mut vars = beta.clone();
    vars.extend((0..n).map(|_| (r.unit() - 0.5) * 0.02));
    let _ = p;
    (vars, x, y, we, wd)
}

fn main() {
    let step = 1.4901161193847656e-8; // sqrt(eps), scipy-style default
    let p = 4;

    // ---- Isomorphism: dense vs structural Jacobian byte-identical ----
    let mut r = Lcg(0x0dd_face_1234_5678);
    let mut mism = 0usize;
    let mut total = 0usize;
    for _ in 0..300 {
        let n = 2 + (r.next_u64() as usize % 60);
        let (vars, x, y, we, wd) = make_problem(&mut r, n);
        let a = old_jac(&vars, p, &x, &y, &we, &wd, step);
        let b = new_jac(&vars, p, &x, &y, &we, &wd, step);
        total += 1;
        if !jac_eq(&a, &b) {
            mism += 1;
        }
    }
    println!(
        "isomorphism (dense vs structural Jacobian): {mism} mismatches / {total} (0 == byte-identical)"
    );

    // ---- Timing: full Jacobian build across realistic ODR fit sizes ----
    for &n in &[100usize, 300, 1000, 3000] {
        let mut rr = Lcg(0xabcd_0000_0000_0001 ^ n as u64);
        let (vars, x, y, we, wd) = make_problem(&mut rr, n);
        let reps = if n <= 300 { 200 } else { 40 };

        let mut acc = 0.0;
        let t = Instant::now();
        for _ in 0..reps {
            acc += old_jac(&vars, p, &x, &y, &we, &wd, step)[0][0];
        }
        let old_t = t.elapsed().as_secs_f64() / reps as f64;

        let t = Instant::now();
        for _ in 0..reps {
            acc += new_jac(&vars, p, &x, &y, &we, &wd, step)[0][0];
        }
        let new_t = t.elapsed().as_secs_f64() / reps as f64;

        println!(
            "n={n:>5}  old={:>9.1}us  new={:>9.1}us  ratio={:>6.1}x  (acc={acc:.2})",
            old_t * 1e6,
            new_t * 1e6,
            old_t / new_t
        );
    }
}
