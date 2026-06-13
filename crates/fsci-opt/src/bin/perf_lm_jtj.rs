// Same-process A/B for the least_squares JtJ/Jtr caching lever.
//
// OLD: rebuild jtj = J^T J (O(n²·m)) and jtr = J^T r at the top of EVERY iteration.
// NEW: cache them; recompute only after an accepted step changes (jac, r). On a rejected
//      step (mu ratchets up) jac/r are unchanged, so the rebuild was redundant.
//
// Both variants follow the IDENTICAL trajectory (cached jtj/jtr are bit-identical to the
// recomputed ones), so they take the same steps and make the same accept/reject decisions.
// The timing difference is exactly the JtJ/Jtr work removed on rejected steps.

use std::time::Instant;

// ---- replicated helpers (verbatim from curvefit.rs) ----
fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
fn l2_norm(v: &[f64]) -> f64 {
    dot_vec(v, v).sqrt()
}
fn jtj_matrix(jac: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = jac.first().map_or(0, Vec::len);
    let mut jtj = vec![vec![0.0; n]; n];
    for row in jac {
        for i in 0..n {
            for j in i..n {
                let v = row[i] * row[j];
                jtj[i][j] += v;
                if i != j {
                    jtj[j][i] += v;
                }
            }
        }
    }
    jtj
}
fn jt_vec(jac: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = jac.first().map_or(0, Vec::len);
    let mut result = vec![0.0; n];
    for (row, &vi) in jac.iter().zip(v.iter()) {
        for (j, &jval) in row.iter().enumerate() {
            result[j] += jval * vi;
        }
    }
    result
}
fn finite_diff_jacobian<F>(residuals: &F, x: &[f64], r0: &[f64], eps: f64) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let m = r0.len();
    let mut jac = vec![vec![0.0; n]; m];
    let mut x_perturbed = x.to_vec();
    for j in 0..n {
        let step = eps * (1.0 + x[j].abs());
        let original = x_perturbed[j];
        x_perturbed[j] += step;
        let r_plus = residuals(&x_perturbed);
        x_perturbed[j] = original;
        for i in 0..m {
            jac[i][j] = (r_plus[i] - r0[i]) / step;
        }
    }
    jac
}
fn max_diag_jtj(jac: &[Vec<f64>]) -> f64 {
    let n = jac.first().map_or(0, Vec::len);
    let mut max_val = 0.0_f64;
    for j in 0..n {
        let mut diag = 0.0;
        for row in jac {
            diag += row[j] * row[j];
        }
        max_val = if max_val.is_nan() || diag.is_nan() { f64::NAN } else { max_val.max(diag) };
    }
    max_val
}
fn mat_vec(jac: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    jac.iter().map(|row| dot_vec(row, v)).collect()
}
fn cholesky_decompose(a: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut low = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for (&lik, &ljk) in low[i].iter().zip(low[j].iter()).take(j) {
                sum -= lik * ljk;
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                low[i][j] = sum.sqrt();
            } else {
                low[i][j] = sum / low[j][j];
            }
        }
    }
    Some(low)
}
fn cholesky_solve(a: &[Vec<f64>], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let low = cholesky_decompose(a, n)?;
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= low[i][j] * y[j];
        }
        y[i] = sum / low[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= low[j][i] * x[j];
        }
        x[i] = sum / low[i][i];
    }
    Some(x)
}
fn solve_damped(jtj: &[Vec<f64>], jtr: &[f64], mu: f64, n: usize) -> Vec<f64> {
    let mut a = jtj.to_vec();
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += mu;
    }
    if let Some(step) = cholesky_solve(&a, jtr, n) {
        step.iter().map(|v| -v).collect()
    } else {
        (0..n).map(|i| -jtr[i] / (jtj[i][i] + mu)).collect()
    }
}

// ---- the LM loop, with a `cache` flag toggling OLD vs NEW jtj/jtr handling ----
fn run_lm<F>(residuals: &F, x0: &[f64], max_nfev: usize, cache: bool) -> (Vec<f64>, usize, usize)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let eps = 1e-8;
    let (gtol, xtol, ftol) = (1e-10, 1e-10, 1e-12);
    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let mut cost = 0.5 * dot_vec(&r, &r);
    let mut jac = finite_diff_jacobian(residuals, &x, &r, eps);
    let mut mu = 1.0e-3 * max_diag_jtj(&jac);
    if mu == 0.0 {
        mu = 1.0;
    }
    let mut nu = 2.0;
    let mut nit = 0usize;
    let mut rejects = 0usize;

    let mut jtj = jtj_matrix(&jac);
    let mut jtr = jt_vec(&jac, &r);

    for _ in 0..max_nfev {
        nit += 1;
        if !cache {
            // OLD behaviour: rebuild every iteration regardless.
            jtj = jtj_matrix(&jac);
            jtr = jt_vec(&jac, &r);
        }
        let grad_inf = jtr.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if grad_inf <= gtol {
            break;
        }
        let step = solve_damped(&jtj, &jtr, mu, n);
        let step_norm = l2_norm(&step);
        let x_norm = l2_norm(&x);
        if step_norm <= xtol * (xtol + x_norm) {
            break;
        }
        let x_new: Vec<f64> = x.iter().zip(step.iter()).map(|(a, b)| a + b).collect();
        let r_new = residuals(&x_new);
        let cost_new = 0.5 * dot_vec(&r_new, &r_new);
        let predicted = {
            let jstep = mat_vec(&jac, &step);
            -dot_vec(&jtr, &step) - 0.5 * dot_vec(&jstep, &jstep)
        };
        let actual = cost - cost_new;
        if predicted > 0.0 {
            let rho = actual / predicted;
            if rho > 0.25 {
                x = x_new;
                r = r_new;
                let old_cost = cost;
                cost = cost_new;
                if ((old_cost - cost) / (1.0 + cost)).abs() <= ftol {
                    break;
                }
                jac = finite_diff_jacobian(residuals, &x, &r, eps);
                if cache {
                    // NEW: refresh derived quantities here (loop top no longer does it).
                    jtj = jtj_matrix(&jac);
                    jtr = jt_vec(&jac, &r);
                }
                // OLD path leaves jtj/jtr stale; the loop top rebuilds them next iteration.
                mu *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
            } else {
                mu *= nu;
                nu *= 2.0;
                rejects += 1;
            }
        } else {
            mu *= nu;
            nu *= 2.0;
            rejects += 1;
        }
    }
    (x, nit, rejects)
}

fn main() {
    // Reject-heavy AND jtj-dominated: squared-linear model f_i = (dot(p, B[i]))².
    // Residual eval is CHEAP (one dot + square, O(n·m)) so JtJ (O(n²·m)) dominates; the
    // model is nonlinear and ill-conditioned from a bad (wrong-sign) start → LM rejects
    // many steps (mu ratchets) before settling — exactly the regime the cache helps.
    let n = 30usize;
    let m = 1600usize;
    let mut s: u64 = 0x9e3779b97f4a7c15;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (1u64 << 31) as f64
    };
    let basis: Vec<Vec<f64>> = (0..m)
        .map(|_| (0..n).map(|_| -1.0 + 2.0 * rng()).collect())
        .collect();
    let p_true: Vec<f64> = (0..n).map(|_| 0.2 + 0.6 * rng()).collect();
    let y: Vec<f64> = basis
        .iter()
        .map(|bi| {
            let d = dot_vec(bi, &p_true);
            d * d
        })
        .collect();
    let residuals = move |p: &[f64]| -> Vec<f64> {
        basis
            .iter()
            .zip(y.iter())
            .map(|(bi, &yi)| {
                let d = dot_vec(bi, p);
                d * d - yi
            })
            .collect()
    };
    // Bad start: wrong sign + scale → many rejected steps before convergence.
    let x0: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { -1.5 } else { 0.8 }).collect();
    let max_nfev = 400;

    // Verify byte-identical trajectory/result between the two variants.
    let (xo, nit_o, rej_o) = run_lm(&residuals, &x0, max_nfev, false);
    let (xc, nit_c, rej_c) = run_lm(&residuals, &x0, max_nfev, true);
    let bits_match = xo.iter().zip(xc.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "trajectory: OLD nit={nit_o} rejects={rej_o} | NEW nit={nit_c} rejects={rej_c} | bit-identical result={bits_match}"
    );
    assert!(bits_match, "results diverged — lever is NOT byte-identical");
    assert_eq!((nit_o, rej_o), (nit_c, rej_c), "trajectory diverged");

    let trials = 9;
    let mut old_times = Vec::new();
    let mut new_times = Vec::new();
    for t in 0..trials {
        // alternate order to defeat warmup/thermal bias
        if t % 2 == 0 {
            let s0 = Instant::now();
            let _ = run_lm(&residuals, &x0, max_nfev, false);
            old_times.push(s0.elapsed().as_secs_f64());
            let s1 = Instant::now();
            let _ = run_lm(&residuals, &x0, max_nfev, true);
            new_times.push(s1.elapsed().as_secs_f64());
        } else {
            let s1 = Instant::now();
            let _ = run_lm(&residuals, &x0, max_nfev, true);
            new_times.push(s1.elapsed().as_secs_f64());
            let s0 = Instant::now();
            let _ = run_lm(&residuals, &x0, max_nfev, false);
            old_times.push(s0.elapsed().as_secs_f64());
        }
    }
    old_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    new_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let old_med = old_times[trials / 2];
    let new_med = new_times[trials / 2];
    println!(
        "OLD median {:.3} ms | NEW median {:.3} ms | speedup {:.2}x  (m={m} n={n})",
        old_med * 1e3,
        new_med * 1e3,
        old_med / new_med
    );
}
