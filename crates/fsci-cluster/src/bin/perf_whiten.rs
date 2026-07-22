//! Median-null-gated A/B for `cluster::whiten`: ORIG serial per-row divide-by-std output map vs
//! row-chunk parallel. BYTE-IDENTICAL (identical per-element expression, row order preserved; the
//! mean/var reductions stay serial). Toggled by `WHITEN_FORCE_SERIAL`. Args: n [dims] [iters].
use fsci_cluster::{WHITEN_FORCE_SERIAL, whiten};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn cv(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_500_000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    let mut s = 0x8fe3_11a7u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0 - 5.0
    };
    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..dims).map(|_| rnd()).collect::<Vec<f64>>())
        .collect();

    WHITEN_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = whiten(&data).expect("whiten");
    WHITEN_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = whiten(&data).expect("whiten");
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .map(|(ra, rb)| {
            ra.iter()
                .zip(rb)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        })
        .sum();
    println!(
        "# cluster::whiten n={n} dims={dims} a[0][0]={} bitmism={bitmism}",
        a[0][0]
    );

    let bench = |serial: bool| -> f64 {
        WHITEN_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(whiten(black_box(&data)).unwrap());
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(whiten(black_box(&data)).unwrap());
        }
        t.elapsed().as_secs_f64() / 5.0 * 1e3
    };

    let (mut ov, mut fv, mut nr, mut cr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        nr.push(o / o2);
        cr.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    WHITEN_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} whiten serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
