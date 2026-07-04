//! Times fsci solve_toeplitz / solve_circulant across n, to compare against scipy.
use fsci_linalg::{solve_circulant, solve_toeplitz};
use std::time::Instant;

fn main() {
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rng = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 11) as f64 / (1u64<<53) as f64 * 2.0 - 1.0 };
    for &n in &[64usize, 256, 512, 1024, 2048] {
        // Diagonally dominant Toeplitz (well-conditioned): c[0] large.
        let mut c: Vec<f64> = (0..n).map(|_| rng()*0.1).collect(); c[0] = n as f64;
        let r: Vec<f64> = { let mut r: Vec<f64> = (0..n).map(|_| rng()*0.1).collect(); r[0]=c[0]; r };
        let b: Vec<f64> = (0..n).map(|_| rng()).collect();
        let reps = if n <= 512 { 200 } else { 40 };
        let mut acc = 0.0;
        let t = Instant::now();
        for _ in 0..reps { acc += solve_toeplitz(&c, Some(&r), &b).unwrap()[0]; }
        let tt = t.elapsed().as_secs_f64()/reps as f64;
        // circulant: positive c[0]-dominant
        let mut cc: Vec<f64> = (0..n).map(|_| rng()*0.1).collect(); cc[0] = n as f64;
        let t = Instant::now();
        for _ in 0..reps { acc += solve_circulant(&cc, &b).unwrap()[0]; }
        let tc = t.elapsed().as_secs_f64()/reps as f64;
        println!("n={n:>5}  solve_toeplitz={:>9.1}us  solve_circulant={:>9.1}us  (acc={acc:.2})", tt*1e6, tc*1e6);
    }
}
