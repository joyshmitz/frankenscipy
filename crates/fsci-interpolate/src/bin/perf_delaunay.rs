//! Measurement harness for Delaunay2D::new — random vs clustered inputs, to locate the
//! bottleneck (O(t) circumcircle scan vs O(|bad|^2) cavity-boundary extraction) and to
//! A/B the boundary edge-count optimization. Golden = number of simplices + an order-
//! sensitive xor over the simplex index triples (changes if simplex order changes).
//! Run: `cargo run --profile release-perf -p fsci-interpolate --bin perf_delaunay`.

use std::hint::black_box;
use std::time::Instant;

use fsci_interpolate::Delaunay2D;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn random_pts(n: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|_| (lcg(&mut s) * 1000.0, lcg(&mut s) * 1000.0))
        .collect()
}

// Clustered: many tight clusters -> large Bowyer-Watson cavities (|bad| big) so the
// O(|bad|^2) boundary extraction is exercised.
fn clustered_pts(n: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut s = seed;
    let clusters = (n / 50).max(1);
    (0..n)
        .map(|i| {
            let c = i % clusters;
            let cx = (c as f64 * 37.0) % 1000.0;
            let cy = (c as f64 * 71.0) % 1000.0;
            (cx + lcg(&mut s) * 2.0, cy + lcg(&mut s) * 2.0)
        })
        .collect()
}

fn golden(pts: &[(f64, f64)]) -> (usize, u64) {
    let d = Delaunay2D::new(pts).expect("delaunay");
    // re-evaluate via find_simplex on a deterministic query grid to fold in simplex order
    let mut acc = 0u64;
    let mut count = 0usize;
    for q in 0..64 {
        let x = (q as f64 * 13.0) % 1000.0;
        let y = (q as f64 * 29.0) % 1000.0;
        if let Some((idx, _, _, _)) = d.find_simplex((x, y)) {
            acc ^= (idx as u64).rotate_left((q % 64) as u32);
            count += 1;
        }
    }
    (count, acc)
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, seed) in &[(500usize, 1u64), (1500, 2)] {
        let r = random_pts(n, seed);
        let (rc, ra) = golden(&r);
        let c = clustered_pts(n, seed);
        let (cc, ca) = golden(&c);
        println!("n={n} random(hits={rc},xor={ra:016x}) clustered(hits={cc},xor={ca:016x})");
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[2000usize, 4000] {
        let r = random_pts(n, 7);
        let reps = 3;
        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..reps {
            acc += Delaunay2D::new(black_box(&r))
                .expect("d")
                .find_simplex((1.0, 1.0))
                .map_or(0, |t| t.0);
        }
        println!(
            "RANDOM    n={n}  {:>10.3?}/build (acc={acc})",
            t0.elapsed() / reps
        );
    }
    for &n in &[2000usize, 4000] {
        let c = clustered_pts(n, 7);
        let reps = 3;
        let t0 = Instant::now();
        let mut acc = 0usize;
        for _ in 0..reps {
            acc += Delaunay2D::new(black_box(&c))
                .expect("d")
                .find_simplex((1.0, 1.0))
                .map_or(0, |t| t.0);
        }
        println!(
            "CLUSTERED n={n}  {:>10.3?}/build (acc={acc})",
            t0.elapsed() / reps
        );
    }
}
