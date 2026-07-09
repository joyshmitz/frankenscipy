use fsci_stats::{RANKDATA_RADIX_DISABLE, cramervonmises_2samp};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut s = 0x71b9_3c2e_5a48_0df1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for &n in &[50_000usize, 200_000, 1_000_000] {
        let x: Vec<f64> = (0..n).map(|_| r() * 1e3 - 500.0).collect();
        let y: Vec<f64> = (0..n).map(|_| r() * 1e3 - 490.0).collect();
        // byte-identity check radix on vs off
        RANKDATA_RADIX_DISABLE.store(true, Ordering::Relaxed);
        let a = cramervonmises_2samp(&x, &y);
        RANKDATA_RADIX_DISABLE.store(false, Ordering::Relaxed);
        let b = cramervonmises_2samp(&x, &y);
        let idok = a.statistic.to_bits() == b.statistic.to_bits()
            && a.pvalue.to_bits() == b.pvalue.to_bits();
        let reps = if n <= 200_000 { 8 } else { 3 };
        let bench = |d: bool| {
            RANKDATA_RADIX_DISABLE.store(d, Ordering::Relaxed);
            let _ = black_box(cramervonmises_2samp(&x, &y));
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(cramervonmises_2samp(black_box(&x), black_box(&y)));
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let base = bench(true).min(bench(true));
        let rdx = bench(false).min(bench(false));
        println!(
            "cvm2 n={n:8}: sort {base:8.2}ms -> radix {rdx:8.2}ms = {:.2}x  bitid={idok}",
            base / rdx
        );
    }
    RANKDATA_RADIX_DISABLE.store(false, Ordering::Relaxed);
}
