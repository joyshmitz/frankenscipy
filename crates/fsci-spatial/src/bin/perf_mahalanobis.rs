// Correctness + A/B for cdist_mahalanobis (GEMM-expansion) vs the naive per-pair loop.
// Expansion must match the direct per-pair mahalanobis within ~1e-9; the speedup is ~d×.
use fsci_spatial::{cdist_mahalanobis, mahalanobis, pdist_mahalanobis};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let na = 400usize;
    let nb = 400usize;
    let d = 64usize;
    let mut s: u64 = 0x9e3779b97f4a7c15;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / (1u64 << 53) as f64
    };
    let xa: Vec<Vec<f64>> = (0..na).map(|_| (0..d).map(|_| rng() * 2.0 - 1.0).collect()).collect();
    let xb: Vec<Vec<f64>> = (0..nb).map(|_| (0..d).map(|_| rng() * 2.0 - 1.0).collect()).collect();
    // A symmetric positive-definite VI = M·Mᵀ/d + I (well-conditioned inverse covariance).
    let m: Vec<Vec<f64>> = (0..d).map(|_| (0..d).map(|_| rng() - 0.5).collect()).collect();
    let mut vi = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            let dot: f64 = (0..d).map(|k| m[i][k] * m[j][k]).sum();
            vi[i][j] = dot / d as f64 + if i == j { 1.0 } else { 0.0 };
        }
    }

    let got = cdist_mahalanobis(&xa, &xb, &vi).expect("cdist_mahalanobis");
    // Correctness vs per-pair direct mahalanobis.
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for i in 0..na {
        for j in 0..nb {
            let want = mahalanobis(&xa[i], &xb[j], &vi);
            let diff = (got[i][j] - want).abs();
            max_abs = max_abs.max(diff);
            if want.abs() > 1e-12 {
                max_rel = max_rel.max(diff / want.abs());
            }
        }
    }
    println!("max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e}");

    let trials = 5;
    let mut t_exp = Vec::new();
    let mut t_naive = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(cdist_mahalanobis(&xa, &xb, &vi).unwrap());
        t_exp.push(t.elapsed().as_secs_f64());

        let t = Instant::now();
        let mut naive = vec![vec![0.0; nb]; na];
        for i in 0..na {
            for j in 0..nb {
                naive[i][j] = mahalanobis(&xa[i], &xb[j], &vi);
            }
        }
        black_box(&naive);
        t_naive.push(t.elapsed().as_secs_f64());
    }
    t_exp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t_naive.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let exp = t_exp[trials / 2] * 1e3;
    let naive = t_naive[trials / 2] * 1e3;
    println!(
        "cdist: naive(per-pair) {naive:.2} ms | expansion {exp:.2} ms | speedup {:.2}x  (na={na} nb={nb} d={d})",
        naive / exp
    );

    // pdist (self-pairs, condensed) — same lever.
    let np = 560usize;
    let xp: Vec<Vec<f64>> = (0..np).map(|_| (0..d).map(|_| rng() * 2.0 - 1.0).collect()).collect();
    let mut pe = Vec::new();
    let mut pn = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(pdist_mahalanobis(&xp, &vi).unwrap());
        pe.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let mut cond = Vec::with_capacity(np * (np - 1) / 2);
        for i in 0..np {
            for j in (i + 1)..np {
                cond.push(mahalanobis(&xp[i], &xp[j], &vi));
            }
        }
        black_box(&cond);
        pn.push(t.elapsed().as_secs_f64());
    }
    pe.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pem = pe[trials / 2] * 1e3;
    let pnm = pn[trials / 2] * 1e3;
    println!(
        "pdist: naive(per-pair) {pnm:.2} ms | expansion {pem:.2} ms | speedup {:.2}x  (n={np} d={d})",
        pnm / pem
    );
}
