//! Same-process A/B for SphericalVoronoi::new. The near-duplicate generator check
//! was an O(n²) all-pairs scan; the new path uses a radius_tol-cell spatial hash
//! (27-cell probe), identical accept/reject. The diagram digest (vertices +
//! regions) must match the all-pairs path exactly. Run via stash A/B.
//! Run: `cargo run --release -p fsci-spatial --bin perf_sphvor`.
use fsci_spatial::SphericalVoronoi;
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

fn sphere_points(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            // Uniform-ish direction, normalized to the unit sphere.
            let x = r.unit() * 2.0 - 1.0;
            let y = r.unit() * 2.0 - 1.0;
            let z = r.unit() * 2.0 - 1.0;
            let nrm = (x * x + y * y + z * z).sqrt().max(1e-12);
            [x / nrm, y / nrm, z / nrm]
        })
        .collect()
}

fn digest(sv: &SphericalVoronoi) -> u64 {
    let mut h = 1469598103934665603u64;
    for v in &sv.vertices {
        for c in v {
            h = (h ^ c.to_bits()).wrapping_mul(1099511628211);
        }
    }
    for region in &sv.regions {
        for &idx in region {
            h = (h ^ idx as u64).wrapping_mul(1099511628211);
        }
        h = (h ^ 0xffff).wrapping_mul(1099511628211);
    }
    h
}

fn main() {
    for &n in &[500usize, 1000, 2000, 4000, 8000] {
        let points = sphere_points(n, 0x5b5b_0000_0001 ^ n as u64);
        let sv = match SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0) {
            Ok(sv) => sv,
            Err(e) => {
                println!("n={n}: err {e:?}");
                continue;
            }
        };
        let dig = digest(&sv);
        let nverts = sv.vertices.len();

        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let out = SphericalVoronoi::new(&points, [0.0, 0.0, 0.0], 1.0).expect("sv");
            std::hint::black_box(&out);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "n={n:>5}  verts={nverts:>6}  {:>10.1} us  digest={dig:016x}",
            best * 1e6
        );
    }
}
