//! Median-null-gated A/B for `cluster::inconsistent`: ORIG serial per-merge subtree-stat loop vs
//! step-chunk parallel (each result[step] is an independent read-only tree walk -> BYTE-IDENTICAL).
//! Toggled by `INCONSISTENT_FORCE_SERIAL`. Args: m [depth] [iters].
use fsci_cluster::{INCONSISTENT_FORCE_SERIAL, inconsistent};
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
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(600_000);
    let depth: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    // Valid caterpillar linkage: n = m+1 leaves; step s merges internal node (n+s-1) with fresh
    // singleton (s+1), so collect_depths walks the left chain `depth` levels down per step.
    let n = m + 1;
    let mut z: Vec<[f64; 4]> = Vec::with_capacity(m);
    z.push([0.0, 1.0, 1.0, 2.0]);
    for s in 1..m {
        z.push([
            (n + s - 1) as f64,
            (s + 1) as f64,
            (s + 1) as f64,
            (s + 2) as f64,
        ]);
    }

    INCONSISTENT_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = inconsistent(&z, depth);
    INCONSISTENT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = inconsistent(&z, depth);
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
        "# cluster::inconsistent m={m} depth={depth} a[m/2]={:?} bitmism={bitmism}",
        a[m / 2]
    );

    let bench = |serial: bool| -> f64 {
        INCONSISTENT_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(inconsistent(black_box(&z), depth));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(inconsistent(black_box(&z), depth));
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
    INCONSISTENT_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} inconsistent serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
