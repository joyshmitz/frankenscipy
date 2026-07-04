//! Times fsci solve_triangular / solve_banded / solveh_banded vs scipy.
use fsci_linalg::{solve_banded, solve_triangular, solveh_banded, SolveOptions, TriangularSolveOptions};
use std::time::Instant;

fn main() {
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 11) as f64 / (1u64<<53) as f64 * 2.0 - 1.0 };
    for &n in &[64usize, 256, 512, 1024, 2048] {
        let b: Vec<f64> = (0..n).map(|_| rng()).collect();
        // lower-triangular, diagonally dominant
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n { for j in 0..=i { a[i][j] = if i==j { n as f64 } else { rng()*0.1 }; } }
        let topt = TriangularSolveOptions { lower: true, ..Default::default() };
        let reps = if n <= 512 { 200 } else { 50 };
        let mut acc = 0.0;
        let t = Instant::now();
        for _ in 0..reps { acc += solve_triangular(&a, &b, topt).unwrap().x[0]; }
        let ttri = t.elapsed().as_secs_f64()/reps as f64;

        // banded pentadiagonal (l=u=2), ab storage rows = l+u+1 = 5
        let (l, u) = (2usize, 2usize);
        let mut ab = vec![vec![0.0; n]; l+u+1];
        for j in 0..n { for i in j.saturating_sub(u)..=(j+l).min(n-1) {
            let r = u + i - j; ab[r][j] = if i==j { (n as f64)*2.0 } else { rng()*0.1 };
        }}
        let t = Instant::now();
        for _ in 0..reps { acc += solve_banded((l,u), &ab, &b, SolveOptions::default()).unwrap().x[0]; }
        let tband = t.elapsed().as_secs_f64()/reps as f64;

        // symmetric banded (lower), bandwidth 2 → rows = 3
        let kd = 2usize;
        let mut abh = vec![vec![0.0; n]; kd+1];
        for j in 0..n { for i in j..=(j+kd).min(n-1) { abh[i-j][j] = if i==j { (n as f64)*2.0 } else { rng()*0.1 }; } }
        let t = Instant::now();
        for _ in 0..reps { acc += solveh_banded(&abh, &b, true).unwrap().x[0]; }
        let tsh = t.elapsed().as_secs_f64()/reps as f64;

        println!("n={n:>5}  solve_triangular={:>8.1}us  solve_banded={:>8.1}us  solveh_banded={:>8.1}us  (acc={acc:.2})",
            ttri*1e6, tband*1e6, tsh*1e6);
    }
}
