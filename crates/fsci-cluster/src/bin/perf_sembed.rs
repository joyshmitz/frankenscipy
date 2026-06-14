// Correctness + A/B for spectral_embedding (randomized top-k eigh of the normalized affinity)
// vs a full symmetric eigendecomposition of the same matrix. The embedding columns must
// solve the generalized eigenproblem A y = mu D y; the speedup is the wall-clock ratio.
use fsci_cluster::spectral_embedding;
use fsci_linalg::{eigh, DecompOptions};
use std::hint::black_box;
use std::time::Instant;

// Baseline: full eigh of D^{-1/2} A D^{-1/2}, then touch the k largest eigenvalues.
fn full_embed_time(aff: &[Vec<f64>], k: usize) -> f64 {
    let n = aff.len();
    let inv: Vec<f64> = aff
        .iter()
        .map(|r| {
            let d: f64 = r.iter().sum();
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect();
    let norm: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| aff[i][j] * inv[i] * inv[j]).collect())
        .collect();
    let e = eigh(&norm, DecompOptions::default()).expect("eigh");
    e.eigenvalues.iter().rev().take(k).sum()
}

fn main() {
    let n = 1600usize;
    let k = 8usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64
    };
    // Symmetric non-negative Gaussian affinity on random 3-D points.
    let pts: Vec<Vec<f64>> = (0..n).map(|_| (0..3).map(|_| rng() * 4.0).collect()).collect();
    let mut aff = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let d2: f64 = (0..3).map(|t| (pts[i][t] - pts[j][t]).powi(2)).sum();
            let w = (-d2).exp();
            aff[i][j] = w;
            aff[j][i] = w;
        }
    }

    let se = spectral_embedding(&aff, k, 7).expect("spectral_embedding");
    let kk = se.eigenvalues.len();
    let deg: Vec<f64> = aff.iter().map(|r| r.iter().sum()).collect();
    let mut maxres = 0.0f64;
    for c in 0..kk {
        let y: Vec<f64> = se.embedding.iter().map(|r| r[c]).collect();
        let yn: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        for i in 0..n {
            let ay: f64 = (0..n).map(|j| aff[i][j] * y[j]).sum();
            maxres = maxres.max((ay - se.eigenvalues[c] * deg[i] * y[i]).abs() / yn);
        }
    }
    println!("spectral_embedding generalized_eig_residual={maxres:.3e}");

    let trials = 3;
    let mut tr = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(spectral_embedding(&aff, k, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(full_embed_time(&aff, k));
        tf.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!("full eigh embedding {f_ms:.2} ms | randomized spectral_embedding {r_ms:.2} ms | speedup {:.2}x  (n={n} k={k})", f_ms / r_ms);
}
