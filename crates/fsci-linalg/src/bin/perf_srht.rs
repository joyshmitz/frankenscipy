// A/B for srht_transform (structured O(n·m log m) sketch) vs the equivalent dense Gaussian
// sketch (Ω·A, O(s·m·n)) — the classical pre-SRHT method. Both are subspace embeddings of
// the same quality; the SRHT is much cheaper. Also checks the embedding (E[‖SAx‖²]=‖Ax‖²).
use fsci_linalg::{matmul, srht_transform};
use std::f64::consts::TAU;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let m = 8192usize; // power of two so the SRHT does no padding
    let n = 200usize;
    let s = 600usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut u01 = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut a = vec![vec![0.0; n]; m];
    for row in a.iter_mut() {
        for v in row.iter_mut() {
            *v = u01() - 0.5;
        }
    }
    let inv_sqrt_s = 1.0 / (s as f64).sqrt();
    let build_gaussian = |st_u01: &mut dyn FnMut() -> f64| -> Vec<Vec<f64>> {
        let mut omega = vec![vec![0.0; m]; s];
        for row in omega.iter_mut() {
            for v in row.iter_mut() {
                let u1 = st_u01().max(1e-300);
                let u2 = st_u01();
                *v = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos() * inv_sqrt_s;
            }
        }
        omega
    };

    let sa = srht_transform(&a, s, 7).expect("srht");
    assert_eq!(sa.len(), s);
    assert_eq!(sa[0].len(), n);

    // Embedding quality: mean ‖S·A·x‖² / ‖A·x‖² over random x → ≈ 1.
    let mut ratio = 0.0;
    let trials_emb = 40;
    for _ in 0..trials_emb {
        let x: Vec<f64> = (0..n).map(|_| u01() - 0.5).collect();
        let dot = |row: &[f64]| row.iter().zip(&x).map(|(&p, &q)| p * q).sum::<f64>();
        let nax: f64 = a.iter().map(|r| dot(r).powi(2)).sum();
        let nsax: f64 = sa.iter().map(|r| dot(r).powi(2)).sum();
        if nax > 1e-12 {
            ratio += nsax / nax;
        }
    }
    println!(
        "srht embedding mean ‖SAx‖²/‖Ax‖² = {:.4} (≈1 expected)",
        ratio / trials_emb as f64
    );

    let trials = 5;
    let mut tsr = Vec::new();
    let mut tg = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(srht_transform(&a, s, 7).unwrap());
        tsr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let omega = build_gaussian(&mut u01);
        black_box(matmul(&omega, &a).unwrap());
        tg.push(t.elapsed().as_secs_f64());
    }
    tsr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sr = tsr[trials / 2] * 1e3;
    let g = tg[trials / 2] * 1e3;
    println!(
        "dense Gaussian sketch {g:.2} ms | srht_transform {sr:.2} ms | speedup {:.1}x  (m={m} n={n} s={s})",
        g / sr
    );
}
