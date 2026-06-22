use fsci_spatial::SphericalVoronoi;
use std::io::Write;
use std::time::Instant;
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn main() {
    let mut r = Lcg(5);
    for &n in &[100usize, 200, 500, 1000, 2000] {
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                let z = 2.0 * r.u() - 1.0;
                let t = 2.0 * std::f64::consts::PI * r.u();
                let s = (1.0 - z * z).sqrt();
                [s * t.cos(), s * t.sin(), z]
            })
            .collect();
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(format!("/tmp/sv_{n}.f64")).unwrap());
        for p in &pts {
            for &v in p {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        drop(f);
        let sv = match SphericalVoronoi::new(&pts, [0.0, 0.0, 0.0], 1.0) {
            Ok(s) => s,
            Err(e) => {
                println!("n={n} ERR: {e:?}");
                continue;
            }
        };
        // dump fsci vertices for structural parity comparison vs scipy
        let mut vf = std::io::BufWriter::new(
            std::fs::File::create(format!("/tmp/sv_{n}_verts.f64")).unwrap(),
        );
        for p in &sv.vertices {
            for &v in p {
                vf.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        drop(vf);
        let v_ok = sv.vertices.len() == 2 * n - 4;
        let r_ok = sv.regions.len() == n && sv.regions.iter().all(|r| r.len() >= 3);
        let on_sphere = sv
            .vertices
            .iter()
            .all(|v| ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - 1.0).abs() < 1e-9);
        let reps = if n >= 1000 { 3 } else { 8 };
        let mut b = std::time::Duration::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let sv = SphericalVoronoi::new(&pts, [0.0, 0.0, 0.0], 1.0).unwrap();
            let e = t.elapsed();
            std::hint::black_box(sv.vertices.len());
            if e < b {
                b = e;
            }
        }
        println!(
            "n={n:>5}  fsci_sphvor={b:>10.3?}  V={} (2n-4={}, ok={v_ok}) regions_ok={r_ok} on_sphere={on_sphere}",
            sv.vertices.len(),
            2 * n - 4
        );
    }
}
