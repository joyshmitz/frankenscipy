//! Median-null-gated A/B for `sparse::sparse_nnz`: ORIG serial filter(v!=0).count() over the stored
//! values vs chunked parallel count. Toggled by `SPARSE_NNZ_FORCE_SERIAL`, alternated per iteration.
//! BYTE-IDENTICAL. Args: n [iters].
use fsci_sparse::{SPARSE_NNZ_FORCE_SERIAL, identity, sparse_nnz};
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    // n x n identity => n stored nonzeros (data.len() == n), the count loop scans all n.
    let mat = identity(n).expect("identity");

    SPARSE_NNZ_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = sparse_nnz(&mat);
    SPARSE_NNZ_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = sparse_nnz(&mat);
    let bitmism = usize::from(a != b);
    println!("# sparse::sparse_nnz n={n} serial={a} parallel={b} bitmism={bitmism}");

    let bench = |serial: bool| -> f64 {
        SPARSE_NNZ_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(sparse_nnz(black_box(&mat)));
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(sparse_nnz(black_box(&mat)));
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
    SPARSE_NNZ_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} sparse_nnz serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
