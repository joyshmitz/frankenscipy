// Correctness + A/B for pivoted_cholesky (low-rank, O(n·k²)) vs full cholesky (O(n³)) on a
// numerically rank-r PSD matrix. The pivoted factor must reconstruct A to the noise floor;
// the speedup is the wall-clock ratio.
use fsci_linalg::{DecompOptions, cholesky, pivoted_cholesky};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let n = 1500usize;
    let r = 30usize;
    let k = 40usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    // PSD A = B·Bᵀ (rank r) + 1e-8·I (strictly PD so the full Cholesky is well-behaved).
    let b: Vec<Vec<f64>> = (0..n).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s: f64 = (0..r).map(|t| b[i][t] * b[j][t]).sum();
            if i == j {
                s += 1e-8;
            }
            a[i][j] = s;
        }
    }

    let pc = pivoted_cholesky(&a, k, 1e-12).expect("pivoted_cholesky");
    // Reconstruction residual ‖A − L·Lᵀ‖_F / ‖A‖_F.
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let approx: f64 = (0..pc.rank)
                .map(|t| pc.factor[i][t] * pc.factor[j][t])
                .sum();
            num += (a[i][j] - approx).powi(2);
            den += a[i][j] * a[i][j];
        }
    }
    println!(
        "pivoted_cholesky rank={} rel_reconstruction_err={:.3e}",
        pc.rank,
        (num / den).sqrt()
    );

    let trials = 5;
    let mut tf = Vec::new();
    let mut tp = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(cholesky(&a, true, DecompOptions::default()).unwrap());
        tf.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(pivoted_cholesky(&a, k, 1e-12).unwrap());
        tp.push(t.elapsed().as_secs_f64());
    }
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f = tf[trials / 2] * 1e3;
    let p = tp[trials / 2] * 1e3;
    println!(
        "full cholesky {f:.2} ms | pivoted_cholesky {p:.2} ms | speedup {:.1}x  (n={n} rank≈{r} k={k})",
        f / p
    );
}
