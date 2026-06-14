// Correctness + A/B for randomized_eigh vs the full symmetric eigh.
// On a rank-r symmetric matrix with the sketch dimension > r, the dominant subspace is
// captured exactly, so the k largest-magnitude eigenvalues match the full eigh's; the
// speedup is the wall-clock ratio (full eigh O(n³) vs randomized O(n²k)).
use fsci_linalg::{DecompOptions, eigh, randomized_eigh};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let n = 1000usize;
    let r = 20usize;
    let k = 10usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    // Symmetric A = Σ_t d_t w_t w_tᵀ, mixed-sign decaying d_t → rank-r symmetric.
    let w: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng()).collect()).collect();
    let d: Vec<f64> = (0..r)
        .map(|t| (if t % 2 == 0 { 1.0 } else { -1.0 }) * (60.0 - 2.0 * t as f64))
        .collect();
    let mut a = vec![vec![0.0; n]; n];
    for t in 0..r {
        let dt = d[t];
        let wt = &w[t];
        for i in 0..n {
            let dwi = dt * wt[i];
            for j in 0..n {
                a[i][j] += dwi * wt[j];
            }
        }
    }

    let full = eigh(&a, DecompOptions::default()).expect("full eigh");
    let reigh = randomized_eigh(&a, k, 15, 2, 7).expect("randomized_eigh");

    // Top-k full eigenvalues by magnitude, sorted ascending.
    let mut by_mag = full.eigenvalues.clone();
    by_mag.sort_by(|x, y| y.abs().total_cmp(&x.abs()));
    let mut top_k: Vec<f64> = by_mag[..k].to_vec();
    top_k.sort_by(|x, y| x.total_cmp(y));

    let mut max_abs = 0.0f64;
    for (g, w) in reigh.eigenvalues.iter().zip(&top_k) {
        max_abs = max_abs.max((g - w).abs());
    }
    println!(
        "k={k} eigenvalues max_abs_err={max_abs:.3e}  (randomized: {:.4} .. {:.4})",
        reigh.eigenvalues[0],
        reigh.eigenvalues[k - 1]
    );

    let trials = 5;
    let mut tf = Vec::new();
    let mut tr = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(eigh(&a, DecompOptions::default()).unwrap());
        tf.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(randomized_eigh(&a, k, 15, 2, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
    }
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f = tf[trials / 2] * 1e3;
    let rr = tr[trials / 2] * 1e3;
    println!(
        "full eigh {f:.2} ms | randomized_eigh {rr:.2} ms | speedup {:.2}x  (n={n} k={k})",
        f / rr
    );
}
