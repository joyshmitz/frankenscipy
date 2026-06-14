// Correctness + A/B for randomized_svd vs the full svd.
// On a rank-r matrix with l = k + oversamples > r, the random sketch captures the entire
// range, so the leading-k singular values match the full SVD to ~machine precision; the
// speedup is the wall-clock ratio (full SVD does O(m·n·min) work, randomized O(m·n·l)).
use fsci_linalg::{DecompOptions, randomized_svd, svd};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let m = 800usize;
    let n = 600usize;
    let r = 15usize; // true rank
    let k = 10usize;
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    // A = B (m×r) · C (r×n)  → exactly rank r.
    let b: Vec<Vec<f64>> = (0..m).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let c: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng()).collect()).collect();
    // Rank-r signal + small full-rank noise → a realistic fast-decaying spectrum with a
    // clear gap after r (so the full SVD is well-conditioned, not pathologically slow on an
    // exactly-degenerate input, and the leading-k are cleanly separated from the floor).
    let a: Vec<Vec<f64>> = b
        .iter()
        .map(|bi| {
            (0..n)
                .map(|j| (0..r).map(|t| bi[t] * c[t][j]).sum::<f64>() + 1e-3 * rng())
                .collect()
        })
        .collect();

    let full = svd(&a, DecompOptions::default()).expect("full svd");
    let rsvd = randomized_svd(&a, k, 10, 2, 42).expect("rsvd");
    let mut max_rel = 0.0f64;
    for i in 0..k {
        let want = full.s[i];
        let got = rsvd.s[i];
        if want.abs() > 1e-9 {
            max_rel = max_rel.max((got - want).abs() / want.abs());
        }
    }
    println!(
        "leading-{k} singular values: max_rel_err={max_rel:.3e}  (full top: {:.4} {:.4} {:.4} ...)",
        full.s[0], full.s[1], full.s[2]
    );

    let trials = 5;
    let mut tf = Vec::new();
    let mut tr = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(svd(&a, DecompOptions::default()).unwrap());
        tf.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(randomized_svd(&a, k, 10, 2, 42).unwrap());
        tr.push(t.elapsed().as_secs_f64());
    }
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f = tf[trials / 2] * 1e3;
    let rr = tr[trials / 2] * 1e3;
    println!(
        "full svd {f:.2} ms | randomized_svd {rr:.2} ms | speedup {:.2}x  (m={m} n={n} k={k})",
        f / rr
    );
}
