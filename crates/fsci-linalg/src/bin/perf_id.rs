// Correctness + A/B for interp_decomp (randomized, SRHT-sketched column pivoting) vs a
// deterministic full-matrix column-pivoted ID. Both reconstruct a low-rank A as
// A[:,skeleton]·proj; the randomized version pivots a tiny sketch instead of all m rows.
use fsci_linalg::{PinvOptions, interp_decomp, matmul, pinv};
use std::hint::black_box;
use std::time::Instant;

// Deterministic ID baseline: column-pivoted Gram-Schmidt on the FULL A (O(m·n·k)).
fn deterministic_id(a: &[Vec<f64>], k: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
    let m = a.len();
    let n = a[0].len();
    let cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    let mut norms: Vec<f64> = cols
        .iter()
        .map(|c| c.iter().map(|&x| x * x).sum())
        .collect();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut skel = Vec::new();
    for _ in 0..k {
        let p = (0..n)
            .max_by(|&a, &b| norms[a].total_cmp(&norms[b]))
            .unwrap();
        if norms[p] <= 0.0 {
            break;
        }
        let mut v = cols[p].clone();
        for q in &basis {
            let d: f64 = q.iter().zip(&v).map(|(&a, &b)| a * b).sum();
            for (vi, &qi) in v.iter_mut().zip(q) {
                *vi -= d * qi;
            }
        }
        let nv = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if nv <= 1e-12 {
            norms[p] = f64::NEG_INFINITY;
            continue;
        }
        for vi in v.iter_mut() {
            *vi /= nv;
        }
        for j in 0..n {
            if norms[j] == f64::NEG_INFINITY {
                continue;
            }
            let d: f64 = v.iter().zip(&cols[j]).map(|(&a, &b)| a * b).sum();
            norms[j] = (norms[j] - d * d).max(0.0);
        }
        basis.push(v);
        skel.push(p);
        norms[p] = f64::NEG_INFINITY;
    }
    let aj: Vec<Vec<f64>> = a
        .iter()
        .map(|r| skel.iter().map(|&j| r[j]).collect())
        .collect();
    let pinv_aj = pinv(&aj, PinvOptions::default()).unwrap().pseudo_inverse;
    let proj = matmul(&pinv_aj, a).unwrap();
    (skel, proj)
}

fn main() {
    let m = 12000usize;
    let n = 400usize;
    let r = 30usize;
    let k = 40usize;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let b: Vec<Vec<f64>> = (0..m).map(|_| (0..r).map(|_| rng()).collect()).collect();
    let c: Vec<Vec<f64>> = (0..r).map(|_| (0..n).map(|_| rng()).collect()).collect();
    let a: Vec<Vec<f64>> = b
        .iter()
        .map(|bi| {
            (0..n)
                .map(|j| (0..r).map(|t| bi[t] * c[t][j]).sum::<f64>() + 1e-7 * rng())
                .collect()
        })
        .collect();

    let id = interp_decomp(&a, k, 10, 7).expect("interp_decomp");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let approx: f64 = (0..id.rank)
                .map(|t| a[i][id.skeleton[t]] * id.proj[t][j])
                .sum();
            num += (a[i][j] - approx).powi(2);
            den += a[i][j] * a[i][j];
        }
    }
    println!(
        "interp_decomp rank={} rel_reconstruction_err={:.3e}",
        id.rank,
        (num / den).sqrt()
    );

    let trials = 3;
    let mut tr = Vec::new();
    let mut td = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(interp_decomp(&a, k, 10, 7).unwrap());
        tr.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(deterministic_id(&a, k));
        td.push(t.elapsed().as_secs_f64());
    }
    tr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    td.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_ms = tr[trials / 2] * 1e3;
    let d_ms = td[trials / 2] * 1e3;
    println!(
        "deterministic full-pivot ID {d_ms:.2} ms | randomized interp_decomp {r_ms:.2} ms | speedup {:.2}x  (m={m} n={n} k={k})",
        d_ms / r_ms
    );
}
