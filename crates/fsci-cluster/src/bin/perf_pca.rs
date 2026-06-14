// Correctness + A/B for pca (randomized SVD) vs a full-SVD PCA, on a low-rank-signal
// dataset. The randomized top-k explained variance must match the full PCA's; the speedup
// is the wall-clock ratio.
use fsci_cluster::pca;
use fsci_linalg::{svd, DecompOptions};
use std::hint::black_box;
use std::time::Instant;

fn full_pca_explained_variance(x: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = x.len();
    let d = x[0].len();
    let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
    let xc: Vec<Vec<f64>> = x
        .iter()
        .map(|r| r.iter().zip(&mean).map(|(&v, &m)| v - m).collect())
        .collect();
    let s = svd(&xc, DecompOptions::default()).expect("svd");
    let denom = (n.max(2) - 1) as f64;
    s.s.iter().take(k).map(|&sv| sv * sv / denom).collect()
}

fn main() {
    let n = 4000usize;
    let d = 300usize;
    let r = 20usize; // signal rank
    let k = 15usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    // X = U_signal · V_signalᵀ (rank r) + small noise.
    let u: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let v: Vec<Vec<f64>> = (0..r).map(|_| (0..d).map(|_| rng()).collect()).collect();
    let x: Vec<Vec<f64>> = u
        .iter()
        .map(|ui| (0..d).map(|j| (0..r).map(|t| ui[t] * v[t][j]).sum::<f64>() + 1e-4 * rng()).collect())
        .collect();

    let p = pca(&x, k, 7).expect("pca");
    let full_ev = full_pca_explained_variance(&x, k);
    let mut max_rel = 0.0f64;
    for (a, b) in p.explained_variance.iter().zip(&full_ev) {
        if *b > 1e-9 {
            max_rel = max_rel.max((a - b).abs() / b);
        }
    }
    println!(
        "pca top-{k} explained_variance max_rel_err={max_rel:.3e}  (evr sum {:.4})",
        p.explained_variance_ratio.iter().sum::<f64>()
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(pca(&x, k, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_pca_explained_variance(&x, k));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!("full-SVD PCA {f_ms:.2} ms | randomized pca {r_ms:.2} ms | speedup {:.2}x  (n={n} d={d} k={k})", f_ms / r_ms);
}
