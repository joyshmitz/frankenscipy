//! Median-null-gated A/B for the FHT coefficient table fill inside `fft::fht`: ORIG serial per-
//! element log-gamma/exp loop vs index-chunk parallel. BYTE-IDENTICAL (each u[m] identical; fht
//! output verified bit-for-bit). Toggled by `FHTCOEFF_FORCE_SERIAL`. Args: n [iters].
use fsci_fft::{FHTCOEFF_FORCE_SERIAL, FftOptions, fht};
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
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    let dln = 0.001_f64;
    let mu = 0.5_f64;
    let (offset, bias) = (0.0_f64, 0.3_f64);
    let opts = FftOptions::default();

    let mut s = 0x1357_bd42u64;
    let input: Vec<f64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect();

    FHTCOEFF_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = fht(&input, dln, mu, offset, bias, &opts).expect("fht");
    FHTCOEFF_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = fht(&input, dln, mu, offset, bias, &opts).expect("fht");
    let bitmism = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    println!("# fft::fht (fhtcoeff) n={n} a[1]={} bitmism={bitmism}", a[1]);

    let bench = |serial: bool| -> f64 {
        FHTCOEFF_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(fht(black_box(&input), dln, mu, offset, bias, &opts).unwrap());
        let t = Instant::now();
        for _ in 0..5 {
            let _ = black_box(fht(black_box(&input), dln, mu, offset, bias, &opts).unwrap());
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
    FHTCOEFF_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} fht[fhtcoeff] serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
