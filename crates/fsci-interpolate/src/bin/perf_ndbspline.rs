// A/B probe: NdBSpline compact-support contraction vs full ∏ns[d] tensor sweep.
// Toggle via NDBSPLINE_COMPACT_DISABLE; verifies byte-identical output (maxdiff=0.0).
use fsci_interpolate::{NDBSPLINE_COMPACT_DISABLE, NdBSpline};
use std::sync::atomic::Ordering;
use std::time::Instant;

// Clamped degree-k knot vector with `n` coefficients (len = n+k+1).
fn clamped_knots(n: usize, k: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; k + 1];
    for i in 1..=(n - k - 1) {
        t.push(i as f64);
    }
    let end = (n - k) as f64;
    for _ in 0..=k {
        t.push(end);
    }
    t
}

// Deterministic pseudo-random coefficients (avoid Math.random-style nondeterminism).
fn lcg_fill(len: usize) -> Vec<f64> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn ab(label: &str, k: usize, ns: &[usize], npts: usize) {
    let ndim = ns.len();
    let t: Vec<Vec<f64>> = ns.iter().map(|&n| clamped_knots(n, k)).collect();
    let total: usize = ns.iter().product();
    let c = lcg_fill(total);
    let nb = NdBSpline::new(t, c, k).unwrap();

    // Evaluation points spread across each axis' [0, n-k] parameter range.
    let mut ps: u64 = 0x1234_5678_9ABC_DEF0;
    let pts: Vec<Vec<f64>> = (0..npts)
        .map(|_| {
            (0..ndim)
                .map(|d| {
                    ps = ps.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u = (ps >> 11) as f64 / (1u64 << 53) as f64;
                    u * (ns[d] - k) as f64
                })
                .collect()
        })
        .collect();

    let run = |full: bool| -> (f64, Vec<f64>) {
        NDBSPLINE_COMPACT_DISABLE.store(full, Ordering::Relaxed);
        // warm
        let _ = nb.evaluate_many(&pts);
        let mut best = f64::INFINITY;
        let mut out = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            out = nb.evaluate_many(&pts);
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        (best, out)
    };

    // interleave
    let (mut tf, mut tc) = (f64::INFINITY, f64::INFINITY);
    let (mut of, mut oc) = (Vec::new(), Vec::new());
    for _ in 0..3 {
        let (a, oa) = run(true);
        let (b, ob) = run(false);
        tf = tf.min(a);
        tc = tc.min(b);
        of = oa;
        oc = ob;
    }
    let maxdiff = of
        .iter()
        .zip(oc.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "{label} k={k} ns={ns:?} npts={npts} total_coef={total}: full {tf:.2}ms compact {tc:.2}ms {:.2}x maxdiff={maxdiff:.1e}",
        tf / tc
    );
}

fn main() {
    ab("2d", 3, &[60, 60], 20000);
    ab("2d", 3, &[100, 100], 20000);
    ab("3d", 3, &[20, 20, 20], 8000);
    ab("3d", 3, &[30, 30, 30], 8000);
    ab("4d", 3, &[12, 12, 12, 12], 4000);
}
