//! Same-process A/B for RbfInterpolator eval. The r²-based kernels (gaussian,
//! multiquadric, inverse-multiquadric) drop the distance sqrt (the kernel depends
//! on r² only); r-based kernels (linear, thin-plate) stay bit-identical. The
//! per-kernel `sum` is stable to ~12 digits across the toggle (proving the change
//! is ULP-level), and the timing shows the win. Run via stash A/B.
//! Run: `cargo run --release -p fsci-interpolate --bin perf_rbf_eval`.
use fsci_interpolate::{RbfInterpolator, RbfKernel};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    let mut r = Lcg(0x9e37_79b9_7f4a_7c15);
    let d = 3usize;
    let nc = 500usize;
    let nq = 5000usize;
    let points: Vec<Vec<f64>> = (0..nc)
        .map(|_| (0..d).map(|_| r.unit() * 4.0 - 2.0).collect())
        .collect();
    let values: Vec<f64> = (0..nc).map(|_| r.unit() * 2.0 - 1.0).collect();
    let queries: Vec<Vec<f64>> = (0..nq)
        .map(|_| (0..d).map(|_| r.unit() * 4.0 - 2.0).collect())
        .collect();

    for (name, kernel) in [
        ("gaussian       ", RbfKernel::Gaussian),
        ("multiquadric   ", RbfKernel::Multiquadric),
        ("inv_multiquad  ", RbfKernel::InverseMultiquadric),
        ("linear(control)", RbfKernel::Linear),
        ("thinplate(ctl) ", RbfKernel::ThinPlateSpline),
    ] {
        let rbf = RbfInterpolator::new(&points, &values, kernel, 1.3).expect("rbf");
        let res = rbf.eval_many(&queries);
        let sum: f64 = res.iter().sum();

        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let out = rbf.eval_many(&queries);
            std::hint::black_box(&out);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{name}  nc={nc} nq={nq}  {:>9.1} us  sum={sum:+.12e}",
            best * 1e6
        );
    }
}
