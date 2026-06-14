// Correctness + A/B for ppca (Probabilistic PCA closed form). The only heavy step is the
// rank-k SVD of the centered data; the randomized public function is compared against a
// baseline computing the same PPCA quantities from a FULL SVD. Both yield the same σ² and
// loadings; the speedup is the wall-clock ratio.
use fsci_cluster::ppca;
use fsci_linalg::{svd, DecompOptions};
use std::hint::black_box;
use std::time::Instant;

// Baseline: full SVD of centered X, closed-form PPCA σ² (mean of discarded eigenvalues).
fn full_ppca_noise(x: &[Vec<f64>], k: usize) -> f64 {
    let n = x.len();
    let d = x[0].len();
    let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
    let xc: Vec<Vec<f64>> = x.iter().map(|r| (0..d).map(|j| r[j] - mean[j]).collect()).collect();
    let total: f64 = (0..d).map(|j| xc.iter().map(|r| r[j] * r[j]).sum::<f64>() / n as f64).sum();
    let dec = svd(&xc, DecompOptions::default()).expect("svd");
    let sum_top: f64 = dec.s.iter().take(k).map(|&s| s * s / n as f64).sum();
    (total - sum_top) / (d - k) as f64
}

fn main() {
    let (n, d, k) = (4000usize, 300usize, 12usize);
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut gauss = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((st >> 11) as f64) / (1u64 << 53) as f64 + 1e-12;
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = ((st >> 11) as f64) / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let r = 12usize;
    let load: Vec<Vec<f64>> = (0..r).map(|_| (0..d).map(|_| gauss()).collect()).collect();
    let x: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let z: Vec<f64> = (0..r).map(|_| gauss()).collect();
            (0..d).map(|j| (0..r).map(|t| z[t] * load[t][j]).sum::<f64>() + 0.5 * gauss()).collect()
        })
        .collect();

    let p = ppca(&x, k, 7).expect("ppca");
    let nf = full_ppca_noise(&x, k);
    println!(
        "ppca noise_variance={:.6} | full-SVD PPCA noise_variance={:.6} | abs_diff={:.2e}",
        p.noise_variance, nf, (p.noise_variance - nf).abs()
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(ppca(&x, k, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_ppca_noise(&x, k));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!("full-SVD PPCA {f_ms:.2} ms | randomized ppca {r_ms:.2} ms | speedup {:.2}x  (n={n} d={d} k={k})", f_ms / r_ms);
}
