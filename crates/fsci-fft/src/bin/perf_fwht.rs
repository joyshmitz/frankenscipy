// Correctness + A/B for fwht (fast Walsh-Hadamard, O(n log n)) vs the naive O(n²) Hadamard
// matrix-vector product H_n·x (the way scipy.linalg.hadamard(n) @ x would compute it).
use fsci_fft::{fwht, FftOptions};
use std::hint::black_box;
use std::time::Instant;

// Naive H_n·x with H[k][j] = (-1)^popcount(k&j), computed on the fly (O(n²)).
fn naive_wht(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            (0..n)
                .map(|j| {
                    if (k & j).count_ones() & 1 == 0 {
                        x[j]
                    } else {
                        -x[j]
                    }
                })
                .sum()
        })
        .collect()
}

fn main() {
    let n = 4096usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let x: Vec<f64> = (0..n).map(|_| rng()).collect();
    let opts = FftOptions::default();

    let fast = fwht(&x, &opts).expect("fwht");
    let slow = naive_wht(&x);
    let mut max_abs = 0.0f64;
    let mut max_mag = 0.0f64;
    for (&f, &s) in fast.iter().zip(&slow) {
        max_abs = max_abs.max((f - s).abs());
        max_mag = max_mag.max(s.abs());
    }
    // Involution up to scale: fwht(fwht(x)) == n·x.
    let twice = fwht(&fast, &opts).expect("fwht");
    let mut inv_err = 0.0f64;
    for (&t, &xi) in twice.iter().zip(&x) {
        inv_err = inv_err.max((t - n as f64 * xi).abs());
    }
    println!(
        "max_abs_err(fwht vs naive)={max_abs:.3e} (rel {:.3e})  involution_err={inv_err:.3e}",
        max_abs / max_mag
    );

    let trials = 5;
    let mut tf = Vec::new();
    let mut ts = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(fwht(&x, &opts).unwrap());
        tf.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(naive_wht(&x));
        ts.push(t.elapsed().as_secs_f64());
    }
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f = tf[trials / 2] * 1e3;
    let s = ts[trials / 2] * 1e3;
    println!("naive O(n²) {s:.3} ms | fwht O(n log n) {f:.3} ms | speedup {:.1}x  (n={n})", s / f);
}
