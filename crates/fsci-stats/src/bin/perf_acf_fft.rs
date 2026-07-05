// Same-binary A/B for the FFT (Wiener-Khinchin) acf path vs direct O(n*lags).
use fsci_stats::{acf, ACF_FFT_DISABLE};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn main() {
    let mut s = 0x51ed_270b_2e8c_1f37u64;
    let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 };
    for &(n, lag) in &[(4096usize, 64usize), (4096, 512), (8192, 2048), (16384, 8191), (65536, 32767)] {
        let x: Vec<f64> = (0..n).map(|_| r() * 2.0 - 1.0).collect();
        let reps = if n <= 8192 { 10 } else { 4 };
        let bench = |disable: bool| {
            ACF_FFT_DISABLE.store(disable, Ordering::Relaxed);
            let _ = black_box(acf(&x, lag));
            let t = Instant::now();
            for _ in 0..reps { let _ = black_box(acf(black_box(&x), lag)); }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let direct = bench(true).min(bench(true));
        let fftp = bench(false).min(bench(false));
        println!("n={n:6} lag={lag:6}: direct {direct:9.3}ms -> fft {fftp:9.3}ms = {:.2}x", direct / fftp);
    }
    ACF_FFT_DISABLE.store(false, Ordering::Relaxed);
}
