use fsci_signal::{AUTOCORR_FFT_DISABLE, AUTOCORR_PAR_DISABLE, autocorrelation};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    AUTOCORR_FFT_DISABLE.store(true, Ordering::Relaxed);
    let mut s = 0x6a09_e667_f3bc_c908u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for &(n, lag) in &[
        (100_000usize, 63usize),
        (300_000, 63),
        (500_000, 40),
        (1_000_000, 32),
        (1_000_000, 63),
    ] {
        let x: Vec<f64> = (0..n).map(|_| r() * 2.0 - 1.0).collect();
        AUTOCORR_PAR_DISABLE.store(true, Ordering::Relaxed);
        let ser = autocorrelation(&x, lag);
        AUTOCORR_PAR_DISABLE.store(false, Ordering::Relaxed);
        let par = autocorrelation(&x, lag);
        let mism = ser
            .iter()
            .zip(&par)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let reps = 6;
        let bench = |d: bool| {
            AUTOCORR_PAR_DISABLE.store(d, Ordering::Relaxed);
            let _ = black_box(autocorrelation(&x, lag));
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(autocorrelation(black_box(&x), lag));
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        let serial = bench(true).min(bench(true));
        let parallel = bench(false).min(bench(false));
        println!(
            "n={n:8} lag={lag:4}: serial {serial:8.2}ms -> parallel {parallel:8.2}ms = {:.2}x  bitmism={mism}",
            serial / parallel
        );
    }
    AUTOCORR_PAR_DISABLE.store(false, Ordering::Relaxed);
    AUTOCORR_FFT_DISABLE.store(false, Ordering::Relaxed);
}
