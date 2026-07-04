//! End-to-end Radau timing on a stiff system with a DENSE Jacobian (exercises the
//! eigen-decoupled n×n real+complex factor path that replaced the full 3n×3n LU).
//! Run: cargo run --release -p fsci-integrate --bin perf_radau_e2e
use fsci_integrate::api::{solve_ivp, SolveIvpOptions, SolverKind};
use fsci_integrate::validation::ToleranceValue;
use fsci_runtime::RuntimeMode;
use std::time::Instant;

fn main() {
    for &n in &[20usize, 40, 80] {
        let mut seed = 0x9e37u64 ^ n as u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        // Stiff system with a DENSE Jacobian: strongly negative diagonal + dense
        // off-diagonal coupling → non-diagonal J, hits the decoupled factor path.
        let m: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { -60.0 - 12.0 * i as f64 } else { 0.4 * rng() })
                    .collect()
            })
            .collect();
        let mut fun = move |_t: f64, y: &[f64]| -> Vec<f64> {
            (0..n).map(|i| (0..n).map(|j| m[i][j] * y[j]).sum::<f64>()).collect()
        };
        let y0: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
        let reps = if n <= 40 { 30 } else { 10 };

        let mut acc = 0.0;
        let mut nfev = 0;
        let t = Instant::now();
        for _ in 0..reps {
            let opts = SolveIvpOptions {
                t_span: (0.0, 1.0),
                y0: &y0,
                method: SolverKind::Radau,
                rtol: 1e-6,
                atol: ToleranceValue::Scalar(1e-8),
                mode: RuntimeMode::Strict,
                ..SolveIvpOptions::default()
            };
            let r = solve_ivp(&mut fun, &opts).unwrap();
            acc += r.y.last().map(|row| row[0]).unwrap_or(0.0);
            nfev = r.nfev;
        }
        let total = t.elapsed().as_secs_f64() / reps as f64;
        println!("n={n:>3}  total={:>8.2}ms  nfev={nfev:>5}  (yend0={acc:.4})", total * 1e3);
    }
}
