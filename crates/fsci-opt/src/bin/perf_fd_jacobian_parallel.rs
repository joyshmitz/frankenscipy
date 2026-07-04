//! Measurement harness for the parallel finite-difference Jacobian lever in the
//! `least_squares` / `curve_fit` LM solvers.
//!
//! The FD Jacobian perturbs each of the `n` parameters and re-evaluates the
//! full `m`-residual vector per column (`finite_diff_jacobian_into`), all
//! serially — like scipy's MINPACK `leastsq`/`curve_fit` (single-threaded
//! Fortran). The columns are independent, so computing each column's perturbed
//! residual on a worker pool and filling the Jacobian serially afterwards is
//! BYTE-IDENTICAL and wins whenever `n_params * m_data * cost_per_point` is large
//! enough to amortize thread spawn.
//!
//! `serial_jac` mirrors `finite_diff_jacobian_into`; `parallel_jac` is the
//! candidate. Proves byte-identity (`.to_bits()`) then times both across
//! representative `(n_params, m_data)` shapes.
//! Run: `cargo run --release -p fsci-opt --bin perf_fd_jacobian_parallel`.
//!
//! MEASURED (this box, sin-sum residual): n=4/m=2000 0.26x, n=6/m=5000 0.75x
//! (both LOSS — too little work), n=8/m=20000 2.75x, n=12/m=50000 5.03x,
//! n=20/m=50000 6.69x, n=40/m=20000 13.66x (all byte-identical). Integration is
//! gated `n>=4 && m>=8192 && n*m>=131072` and blocked on propagating `+ Sync`
//! through the public `least_squares`/`curve_fit` closure bounds — see
//! docs/NEGATIVE_EVIDENCE.md.
#![allow(clippy::needless_range_loop)]

use std::sync::Arc;
use std::time::Instant;

#[inline]
fn residuals(params: &[f64], xs: &[f64], out: &mut Vec<f64>) {
    out.clear();
    for &x in xs {
        let mut acc = 0.0;
        for (k, &p) in params.iter().enumerate() {
            acc += p * ((k as f64 + 1.0) * x).sin();
        }
        out.push(acc);
    }
}

/// Verbatim serial route (mirrors `finite_diff_jacobian_into`).
fn serial_jac(params: &[f64], xs: &[f64], r0: &[f64], eps: f64) -> Vec<Vec<f64>> {
    let n = params.len();
    let m = r0.len();
    let mut jac = vec![vec![0.0; n]; m];
    let mut xp = params.to_vec();
    let mut rp = Vec::with_capacity(m);
    for j in 0..n {
        let step = eps * (1.0 + params[j].abs());
        let orig = xp[j];
        xp[j] += step;
        residuals(&xp, xs, &mut rp);
        xp[j] = orig;
        for i in 0..m {
            jac[i][j] = (rp[i] - r0[i]) / step;
        }
    }
    jac
}

/// Candidate: independent per-column perturbed-residual eval on a worker pool,
/// then a serial fill. Byte-identical to `serial_jac`.
fn parallel_jac(params: &[f64], xs: &Arc<Vec<f64>>, r0: &[f64], eps: f64, nthreads: usize) -> Vec<Vec<f64>> {
    let n = params.len();
    let m = r0.len();
    let per = n.div_ceil(nthreads);
    let mut cols: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..nthreads {
            let lo = t * per;
            let hi = ((t + 1) * per).min(n);
            if lo >= hi {
                break;
            }
            let params = params.to_vec();
            let xs = Arc::clone(xs);
            handles.push(s.spawn(move || {
                let mut rp = Vec::with_capacity(xs.len());
                (lo..hi)
                    .map(|j| {
                        let step = eps * (1.0 + params[j].abs());
                        let mut xp = params.clone();
                        xp[j] += step;
                        residuals(&xp, &xs, &mut rp);
                        (j, rp.clone(), step)
                    })
                    .collect::<Vec<_>>()
            }));
        }
        let mut collected: Vec<(usize, Vec<f64>, f64)> = Vec::with_capacity(n);
        for h in handles {
            collected.extend(h.join().unwrap());
        }
        collected.sort_by_key(|c| c.0);
        for (_, rp, step) in collected {
            cols.push((rp, step));
        }
    });
    let mut jac = vec![vec![0.0; n]; m];
    for (j, (rp, step)) in cols.iter().enumerate() {
        for i in 0..m {
            jac[i][j] = (rp[i] - r0[i]) / step;
        }
    }
    jac
}

fn main() {
    let eps = 1.4901161193847656e-8;
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(8);
    for &(n, m) in &[(4usize, 2000usize), (6, 5000), (8, 20000), (12, 50000), (20, 50000), (40, 20000)] {
        let params: Vec<f64> = (0..n).map(|k| 0.3 + 0.1 * k as f64).collect();
        let xs: Arc<Vec<f64>> = Arc::new((0..m).map(|i| i as f64 * 0.001).collect());
        let mut r0 = Vec::new();
        residuals(&params, &xs, &mut r0);
        let nt = threads.min(n);

        let js = serial_jac(&params, &xs, &r0, eps);
        let jp = parallel_jac(&params, &xs, &r0, eps, nt);
        let mut mism = 0u64;
        for i in 0..m {
            for j in 0..n {
                if js[i][j].to_bits() != jp[i][j].to_bits() {
                    mism += 1;
                }
            }
        }

        let reps = if m * n <= 200_000 { 200 } else { 40 };
        let t = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += serial_jac(&params, &xs, &r0, eps)[0][0];
        }
        let ts = t.elapsed().as_secs_f64() / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            acc += parallel_jac(&params, &xs, &r0, eps, nt)[0][0];
        }
        let tp = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "n={n:>3} m={m:>6} thr={nt:>2}  serial={:>8.1}us parallel={:>8.1}us  speedup={:>5.2}x  mism={mism} (acc={acc:.2})",
            ts * 1e6,
            tp * 1e6,
            ts / tp
        );
    }
}
