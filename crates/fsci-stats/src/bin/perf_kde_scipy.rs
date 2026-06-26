//! Same-box A/B: fsci GaussianKdeNd::evaluate_many vs scipy.stats.gaussian_kde.
//! Dumps identical dataset + query points so the Python side builds the same KDE.
//! Run: cargo run --profile release-perf -p fsci-stats --bin perf_kde_scipy -- <d> <ntrain> <nquery>

use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

use fsci_stats::GaussianKdeNd;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

// Box-Muller-ish: just use uniform shifted; KDE timing is metric-independent of
// distribution shape, only n/m/d matter.
fn main() {
    let mut a = std::env::args().skip(1);
    let d: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let ntrain: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let nquery: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2000);

    let mut s = 0x1234_5678u64;
    let dataset: Vec<Vec<f64>> = (0..ntrain)
        .map(|_| (0..d).map(|_| lcg(&mut s) * 4.0 - 2.0).collect())
        .collect();
    let queries: Vec<Vec<f64>> = (0..nquery)
        .map(|_| (0..d).map(|_| lcg(&mut s) * 4.0 - 2.0).collect())
        .collect();

    let kde = GaussianKdeNd::new(&dataset).expect("kde build");
    let mut bt = f64::INFINITY;
    let mut acc = 0.0f64;
    for _ in 0..5 {
        let t0 = Instant::now();
        let out = black_box(kde.evaluate_many(black_box(&queries)));
        bt = bt.min(t0.elapsed().as_secs_f64());
        acc = out.iter().sum();
    }
    println!("fsci kde d={d} ntrain={ntrain} nquery={nquery}");
    println!("  evaluate_many {:>9.3} ms (sum={acc:.6})", bt * 1e3);

    let path = "/data/tmp/claude-1000/-data-projects-frankenscipy/652c4f0d-f876-4915-aed6-0c9f74ca1f85/scratchpad/kde_in.bin";
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&(d as u64).to_le_bytes());
    buf.extend_from_slice(&(ntrain as u64).to_le_bytes());
    buf.extend_from_slice(&(nquery as u64).to_le_bytes());
    for p in dataset.iter().chain(queries.iter()) {
        for &v in p {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&buf))
        .expect("dump");
    println!("wrote {path}");
}
