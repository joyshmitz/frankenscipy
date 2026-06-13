// A/B + byte-identity probe for the MGC permutation-test parallelization.
// The p-value must be bit-identical between the serial and parallel builds (same
// pre-generated permutations + order-independent exceedance count); the speedup is the
// wall-clock ratio. Run on the parallel build, then `git stash push -- lib.rs` for serial.
use fsci_stats::multiscale_graphcorr;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    // Deterministic correlated data so the permutation test does real work.
    let n = 120usize;
    let d = 3usize;
    let reps = 2000usize;
    println!(
        "available_parallelism = {}",
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0)
    );
    let mut s: u64 = 0x243f6a8885a308d3;
    let mut rng = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / (1u64 << 31) as f64 - 1.0 // ~[-1,1)
    };
    let x: Vec<Vec<f64>> = (0..n).map(|_| (0..d).map(|_| rng()).collect()).collect();
    let y: Vec<Vec<f64>> = x
        .iter()
        .map(|xi| xi.iter().map(|&v| v * 0.7 + 0.3 * rng()).collect())
        .collect();

    let r0 = multiscale_graphcorr(&x, &y, reps, Some(42)).expect("mgc");
    println!(
        "pvalue_bits={:016x} pvalue={:.17e} statistic_bits={:016x} opt_scale={:?}",
        r0.pvalue.to_bits(),
        r0.pvalue,
        r0.statistic.to_bits(),
        r0.opt_scale
    );

    let trials = 5;
    let mut times = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        let r = multiscale_graphcorr(&x, &y, reps, Some(42)).expect("mgc");
        black_box(&r);
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "median {:.2} ms over {trials} trials (n={n} reps={reps})",
        times[trials / 2] * 1e3
    );
}
