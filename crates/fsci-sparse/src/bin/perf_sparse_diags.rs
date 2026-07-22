//! Median-null-gated A/B for `sparse::diags`: ORIG serial per-row diagonal emit vs gather-then-
//! concat across contiguous row-blocks. BYTE-IDENTICAL (same emit order, rows concatenated in row
//! order). Toggled by `SPARSE_DIAGS_FORCE_SERIAL`. Args: n [nbands] [iters].
use fsci_sparse::{SPARSE_DIAGS_FORCE_SERIAL, diags};
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
        .unwrap_or(3_000_000);
    let nbands: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    // Symmetric band: offsets 0, ±1, ±2, ... up to nbands total (odd -> symmetric).
    let half = nbands / 2;
    let mut offsets: Vec<isize> = Vec::new();
    offsets.push(0);
    for k in 1..=half {
        offsets.push(k as isize);
        offsets.push(-(k as isize));
    }
    let diagonals: Vec<Vec<f64>> = offsets
        .iter()
        .map(|&off| {
            let len = n - (off.unsigned_abs());
            (0..len)
                .map(|i| 1.0 + (i as f64) * 1e-6 + off as f64)
                .collect::<Vec<f64>>()
        })
        .collect();

    SPARSE_DIAGS_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let sa = diags(&diagonals, &offsets, None).expect("diags");
    SPARSE_DIAGS_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let pa = diags(&diagonals, &offsets, None).expect("diags");
    let dbit = sa
        .data()
        .iter()
        .zip(pa.data())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let ibit = sa
        .indices()
        .iter()
        .zip(pa.indices())
        .filter(|(x, y)| x != y)
        .count();
    let pbit = sa
        .indptr()
        .iter()
        .zip(pa.indptr())
        .filter(|(x, y)| x != y)
        .count();
    let lenmis = (sa.data().len() != pa.data().len()) as usize;
    let bitmism = dbit + ibit + pbit + lenmis;
    println!(
        "# sparse::diags n={n} nbands={} nnz={} dbit={dbit} ibit={ibit} pbit={pbit} lenmis={lenmis} bitmism={bitmism}",
        offsets.len(),
        sa.data().len()
    );

    let bench = |serial: bool| -> f64 {
        SPARSE_DIAGS_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(diags(black_box(&diagonals), black_box(&offsets), None).unwrap());
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(diags(black_box(&diagonals), black_box(&offsets), None).unwrap());
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
    SPARSE_DIAGS_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} diags serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
