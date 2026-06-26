//! Same-box A/B: fsci griddata (Linear / Cubic) build+eval vs scipy.interpolate.griddata.
//! Dumps the IDENTICAL points/values/queries as little-endian f64 to a scratch file so the
//! Python side feeds Qhull the same input. Best-of-N wall time. Run:
//!   cargo run --profile release-perf -p fsci-interpolate --bin perf_griddata_scipy
//! then: python3 <this dir>/perf_griddata_scipy.py

use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use fsci_interpolate::{
    CloughTocher2DInterpolator, GriddataMethod, LinearNDInterpolator, griddata,
};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    let mut a = std::env::args().skip(1);
    let np: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let nq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(5000);
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let pts: Vec<Vec<f64>> = (0..np).map(|_| vec![lcg(&mut s), lcg(&mut s)]).collect();
    let vals: Vec<f64> = (0..np)
        .map(|i| (pts[i][0] * 6.2).sin() + (pts[i][1] * 4.7).cos())
        .collect();
    let xi: Vec<Vec<f64>> = (0..nq).map(|_| vec![lcg(&mut s), lcg(&mut s)]).collect();

    let best = |m: GriddataMethod| {
        let mut bt = f64::INFINITY;
        let mut acc = 0.0f64;
        for _ in 0..5 {
            let t0 = Instant::now();
            let out = griddata(black_box(&pts), black_box(&vals), black_box(&xi), m).expect("g");
            let dt = t0.elapsed().as_secs_f64();
            acc += out.iter().filter(|v| v.is_finite()).sum::<f64>();
            bt = bt.min(dt);
        }
        (bt, acc)
    };
    if std::env::var("CT_PROFILE").is_ok() {
        let lin_build = {
            let t = Instant::now();
            let it = LinearNDInterpolator::new(&pts, &vals).unwrap();
            let b = t.elapsed();
            let t = Instant::now();
            let _ = black_box(it.eval_many(&xi).unwrap());
            (b, t.elapsed())
        };
        let cub_build = {
            let t = Instant::now();
            let it = CloughTocher2DInterpolator::new(&pts, &vals).unwrap();
            let b = t.elapsed();
            let t = Instant::now();
            let _ = black_box(it.eval_many(&xi).unwrap());
            (b, t.elapsed())
        };
        eprintln!(
            "PHASE linear: build {:?} eval {:?} | cubic: build {:?} eval {:?}",
            lin_build.0, lin_build.1, cub_build.0, cub_build.1
        );
    }
    let (lin, la) = best(GriddataMethod::Linear);
    let (cub, ca) = best(GriddataMethod::Cubic);
    let (near, na) = best(GriddataMethod::Nearest);
    println!("fsci  np={np} nq={nq}");
    println!("  linear  {:>9.3} ms  (acc={la:.4})", lin * 1e3);
    println!("  cubic   {:>9.3} ms  (acc={ca:.4})", cub * 1e3);
    println!("  nearest {:>9.3} ms  (acc={na:.4})", near * 1e3);

    // Dump identical input for scipy.
    let path = "/data/tmp/claude-1000/-data-projects-frankenscipy/652c4f0d-f876-4915-aed6-0c9f74ca1f85/scratchpad/griddata_in.bin";
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(np as u64).to_le_bytes());
    buf.extend_from_slice(&(nq as u64).to_le_bytes());
    for p in &pts {
        buf.extend_from_slice(&p[0].to_le_bytes());
        buf.extend_from_slice(&p[1].to_le_bytes());
    }
    for &v in &vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for q in &xi {
        buf.extend_from_slice(&q[0].to_le_bytes());
        buf.extend_from_slice(&q[1].to_le_bytes());
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("dump");
    println!("wrote {path}");
}
