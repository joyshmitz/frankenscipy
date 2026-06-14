// Correctness + A/B for factor_analysis. The randomized public function is compared against
// a baseline that runs the identical FA EM but with a FULL SVD inner solver each iteration
// (what fsci would do without the randomized route). Both must converge to the same model;
// the speedup is the wall-clock ratio (the per-iteration top-k SVD dominates).
use fsci_cluster::factor_analysis;
use fsci_linalg::{svd, DecompOptions};
use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Instant;

const SMALL: f64 = 1e-12;

// FA EM with a full-SVD inner solver, mirroring fsci_cluster::factor_analysis.
fn fa_full(x: &[Vec<f64>], k: usize, max_iter: usize, tol: f64) -> (Vec<Vec<f64>>, f64, usize) {
    let n = x.len();
    let d = x[0].len();
    let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
    let xc: Vec<Vec<f64>> = x.iter().map(|r| (0..d).map(|j| r[j] - mean[j]).collect()).collect();
    let var: Vec<f64> = (0..d).map(|j| xc.iter().map(|r| r[j] * r[j]).sum::<f64>() / n as f64).collect();
    let nsqrt = (n as f64).sqrt();
    let llconst = d as f64 * (2.0 * PI).ln() + k as f64;
    let mut psi = vec![1.0f64; d];
    let mut old_ll = f64::NEG_INFINITY;
    let mut w = vec![vec![0.0; d]; k];
    let mut ll = f64::NEG_INFINITY;
    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        let sp: Vec<f64> = psi.iter().map(|&p| p.sqrt() + SMALL).collect();
        let xs: Vec<Vec<f64>> = xc.iter().map(|r| (0..d).map(|j| r[j] / (sp[j] * nsqrt)).collect()).collect();
        let fro2: f64 = xs.iter().flatten().map(|v| v * v).sum();
        let dec = svd(&xs, DecompOptions::default()).expect("svd");
        let s2: Vec<f64> = dec.s.iter().take(k).map(|&s| s * s).collect();
        let cap2: f64 = s2.iter().sum();
        for t in 0..k.min(s2.len()) {
            let scale = (s2[t] - 1.0).max(0.0).sqrt();
            for j in 0..d {
                w[t][j] = scale * dec.vt[t][j] * sp[j];
            }
        }
        let log_s2: f64 = s2.iter().map(|&v| v.max(SMALL).ln()).sum();
        let log_psi: f64 = psi.iter().map(|&p| p.ln()).sum();
        ll = -(d as f64) / 2.0 * (llconst + log_s2 + (fro2 - cap2) + log_psi);
        if ll - old_ll < tol {
            break;
        }
        old_ll = ll;
        for j in 0..d {
            let wj2: f64 = (0..k).map(|t| w[t][j] * w[t][j]).sum();
            psi[j] = (var[j] - wj2).max(SMALL);
        }
    }
    (w, ll, iters)
}

fn main() {
    let (n, d, k) = (2000usize, 400usize, 10usize);
    let max_iter = 20usize;
    let tol = 1e-3;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut gauss = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((st >> 11) as f64) / (1u64 << 53) as f64 + 1e-12;
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((st >> 11) as f64) / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    };
    let r = 10usize;
    let load: Vec<Vec<f64>> = (0..r).map(|_| (0..d).map(|_| gauss()).collect()).collect();
    let x: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let z: Vec<f64> = (0..r).map(|_| gauss()).collect();
            (0..d).map(|j| (0..r).map(|t| z[t] * load[t][j]).sum::<f64>() + 0.4 * gauss()).collect()
        })
        .collect();

    let fa = factor_analysis(&x, k, max_iter, tol, 7).expect("factor_analysis");
    let (_wf, llf, itf) = fa_full(&x, k, max_iter, tol);
    println!(
        "factor_analysis loglike={:.4} ({} it) | full-SVD FA loglike={:.4} ({} it) | dll={:.2e}",
        fa.loglike, fa.n_iter, llf, itf, (fa.loglike - llf).abs()
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(factor_analysis(&x, k, max_iter, tol, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(fa_full(&x, k, max_iter, tol));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!("full-SVD FA {f_ms:.2} ms | randomized factor_analysis {r_ms:.2} ms | speedup {:.2}x  (n={n} d={d} k={k})", f_ms / r_ms);
}
