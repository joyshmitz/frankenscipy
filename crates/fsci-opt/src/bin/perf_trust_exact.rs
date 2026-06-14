// Proof probe for the trust_exact Hessian-cache lever. Runs the REAL public
// minimize(TrustExact) on a reject-heavy high-dimensional Rosenbrock and prints the
// converged point (as bits) plus eval counts. Run once on the cached (NEW) build and once
// on the stashed (OLD) build: x must be bit-identical; nfev/njev are the speedup.
use fsci_opt::{MinimizeOptions, OptimizeMethod, minimize};

fn rosenbrock_nd(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        acc += a * a + 100.0 * b * b;
    }
    acc
}

// Powell singular function (classic trust-region rejecter), n must be a multiple of 4.
fn powell_singular(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() / 4 {
        let (a, b, c, d) = (x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
        let t1 = a + 10.0 * b;
        let t2 = c - d;
        let t3 = b - 2.0 * c;
        let t4 = a - d;
        acc += t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 + 10.0 * t4 * t4 * t4 * t4;
    }
    acc
}

fn run(name: &str, f: fn(&[f64]) -> f64, x0: &[f64], opt: &[f64]) {
    let options = MinimizeOptions {
        method: Some(OptimizeMethod::TrustExact),
        tol: Some(1e-8),
        maxiter: Some(300),
        maxfev: Some(2_000_000),
        ..MinimizeOptions::default()
    };
    let r = minimize(f, x0, options).expect("minimize");
    let dist: f64 =
        r.x.iter()
            .zip(opt.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
    println!(
        "{name}: nfev={} njev={} nhev={} fun={:.10e} dist_to_opt={:.4e}",
        r.nfev,
        r.njev,
        r.nhev,
        r.fun.unwrap_or(f64::NAN),
        dist
    );
}

fn main() {
    let rb10: Vec<f64> = (0..10)
        .map(|i| if i % 2 == 0 { -1.5 } else { 1.7 })
        .collect();
    let rb20: Vec<f64> = (0..20)
        .map(|i| if i % 2 == 0 { -1.5 } else { 1.7 })
        .collect();
    let pw: Vec<f64> = (0..12)
        .map(|i| match i % 4 {
            0 => 3.0,
            1 => -1.0,
            2 => 0.0,
            _ => 1.0,
        })
        .collect();
    let ones10 = vec![1.0; 10];
    let ones20 = vec![1.0; 20];
    let zeros12 = vec![0.0; 12];
    run("rosenbrock_n10", rosenbrock_nd, &rb10, &ones10);
    run("rosenbrock_n20", rosenbrock_nd, &rb20, &ones20);
    run("powell_n12", powell_singular, &pw, &zeros12);
}
