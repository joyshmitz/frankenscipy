//! Same-box A/B: fsci interpn (RegularGridInterpolator) vs scipy.interpolate.interpn.
//! Dumps identical grid axes/values/queries to a scratch file for the Python side.
//! Run: cargo run --profile release-perf -p fsci-interpolate --bin perf_interpn_scipy -- <gridlen> <nq>

use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use fsci_interpolate::{RegularGridMethod, interpn};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    let mut a = std::env::args().skip(1);
    let glen: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(200);
    let nq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(50000);
    let ndim = 2usize;

    let axes: Vec<Vec<f64>> = (0..ndim)
        .map(|_| (0..glen).map(|i| i as f64).collect())
        .collect();
    let total = glen.pow(ndim as u32);
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let values: Vec<f64> = (0..total).map(|_| lcg(&mut s) * 10.0).collect();
    let xi: Vec<Vec<f64>> = (0..nq)
        .map(|_| (0..ndim).map(|_| lcg(&mut s) * (glen - 1) as f64).collect())
        .collect();

    let best = |m: RegularGridMethod| {
        let mut bt = f64::INFINITY;
        let mut acc = 0.0f64;
        for _ in 0..5 {
            let t0 = Instant::now();
            let out = interpn(
                black_box(axes.clone()),
                black_box(values.clone()),
                black_box(&xi),
                m,
                false,
                None,
            )
            .expect("interpn");
            bt = bt.min(t0.elapsed().as_secs_f64());
            acc += out.iter().filter(|v| v.is_finite()).sum::<f64>();
        }
        (bt, acc)
    };
    let (lin, la) = best(RegularGridMethod::Linear);
    let (cub, ca) = best(RegularGridMethod::Cubic);
    println!("fsci interpn ndim={ndim} glen={glen} nq={nq}");
    println!("  linear {:>9.3} ms (acc={la:.3})", lin * 1e3);
    println!("  cubic  {:>9.3} ms (acc={ca:.3})", cub * 1e3);

    let path = "/data/tmp/claude-1000/-data-projects-frankenscipy/652c4f0d-f876-4915-aed6-0c9f74ca1f85/scratchpad/interpn_in.bin";
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(glen as u64).to_le_bytes());
    buf.extend_from_slice(&(nq as u64).to_le_bytes());
    for &v in &values {
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
