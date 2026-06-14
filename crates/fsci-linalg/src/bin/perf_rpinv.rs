// Correctness + A/B for randomized_pinv vs the full-SVD pinv, on a numerically rank-r
// matrix. With both truncating at the same relative tolerance (above the noise floor,
// below the signal), the two pseudoinverses agree; the speedup is the wall-clock ratio.
use fsci_linalg::{PinvOptions, pinv, randomized_pinv};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let m = 1200usize;
    let n = 400usize;
    let r = 20usize; // signal rank
    let k = 25usize; // capture all signal + margin
    let rtol = 1e-3;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    // A = B(m×r)·C(r×n) (signal) + tiny noise → numerically rank-r, but full-rank so the
    // full SVD is well-behaved.
    let b: Vec<Vec<f64>> = (0..m).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let c: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng()).collect()).collect();
    let a: Vec<Vec<f64>> = b
        .iter()
        .map(|bi| {
            (0..n)
                .map(|j| (0..r).map(|t| bi[t] * c[t][j]).sum::<f64>() + 1e-6 * rng())
                .collect()
        })
        .collect();

    let opts = PinvOptions {
        rtol: Some(rtol),
        ..Default::default()
    };
    let full = pinv(&a, opts).expect("full pinv");
    let rp = randomized_pinv(&a, k, 10, 2, Some(rtol), 7).expect("randomized_pinv");

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (rf, rr) in full.pseudo_inverse.iter().zip(&rp.pseudo_inverse) {
        for (&x, &y) in rf.iter().zip(rr) {
            num += (x - y) * (x - y);
            den += x * x;
        }
    }
    println!(
        "rel_pinv_err={:.3e}  full.rank={}  randomized.rank={}",
        (num / den).sqrt(),
        full.rank,
        rp.rank
    );

    let trials = 5;
    let mut tf = Vec::new();
    let mut tr = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(pinv(&a, opts).unwrap());
        tf.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(randomized_pinv(&a, k, 10, 2, Some(rtol), 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
    }
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f = tf[trials / 2] * 1e3;
    let rr = tr[trials / 2] * 1e3;
    println!(
        "full pinv {f:.2} ms | randomized_pinv {rr:.2} ms | speedup {:.2}x  (m={m} n={n} k={k})",
        f / rr
    );
}
