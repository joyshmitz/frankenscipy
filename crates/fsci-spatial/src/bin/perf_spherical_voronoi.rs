//! Timing + bit-identity harness for `SphericalVoronoi::new` parallel face
//! detection.
//!
//! The construction enumerates every (i,j,k) triplet (O(n³)) and validates each
//! against all n points (O(n), early-break) — an O(n⁴) gift-wrapping hull. The
//! per-triplet test is independent, so faces are now detected in parallel over
//! pair-balanced i-ranges and collected in the original (i,j,k) order; the dedup
//! check + push stay serial. Same triplets, same order, same float values =>
//! the `vertices` and `regions` are byte-identical to the sequential build.
//!
//! Dumps a golden checksum of vertices+regions for fixed small point sets
//! (compare across the stashed pre-change build) and times large-n builds.
//! Run: `cargo run --profile release-perf -p fsci-spatial --bin perf_spherical_voronoi`.

use std::hint::black_box;
use std::time::Instant;

use fsci_spatial::SphericalVoronoi;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// n distinct points exactly on the unit sphere (normalize random vectors).
fn sphere_points(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut g = Lcg(seed);
    let mut pts = Vec::with_capacity(n);
    while pts.len() < n {
        let v = [
            g.unit() * 2.0 - 1.0,
            g.unit() * 2.0 - 1.0,
            g.unit() * 2.0 - 1.0,
        ];
        let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if nrm < 1e-6 {
            continue;
        }
        pts.push([v[0] / nrm, v[1] / nrm, v[2] / nrm]);
    }
    pts
}

fn checksum(sv: &SphericalVoronoi) -> u64 {
    let mut h = 1469598103934665603u64;
    for v in &sv.vertices {
        for &c in v {
            h = (h ^ c.to_bits()).wrapping_mul(1099511628211);
        }
    }
    for region in &sv.regions {
        for &idx in region {
            h = (h ^ idx as u64).wrapping_mul(1099511628211);
        }
        h = h.wrapping_mul(31);
    }
    h
}

fn main() {
    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &(n, seed) in &[(8usize, 1u64), (16, 2), (30, 3)] {
        let pts = sphere_points(n, seed);
        match SphericalVoronoi::new(&pts, [0.0, 0.0, 0.0], 1.0) {
            Ok(sv) => println!(
                "n={n} seed={seed} nverts={} chk={:016x}",
                sv.vertices.len(),
                checksum(&sv)
            ),
            Err(e) => println!("n={n} seed={seed} err={e:?}"),
        }
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &[64usize, 200, 320] {
        let pts = sphere_points(n, 7);
        let reps = if n <= 120 { 10 } else { 3 };
        let _ = SphericalVoronoi::new(&pts, [0.0, 0.0, 0.0], 1.0);
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..reps {
            let sv = SphericalVoronoi::new(black_box(&pts), [0.0, 0.0, 0.0], 1.0).unwrap();
            acc ^= sv.vertices.len() as u64;
        }
        println!(
            "n={n:>4}  {:>10.3?}/build  (nverts_acc={acc})",
            t0.elapsed() / reps
        );
    }
}
