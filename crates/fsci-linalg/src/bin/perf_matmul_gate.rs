use fsci_linalg::{DecompOptions, MATMUL_MACS_GATE_OVERRIDE, cholesky, matmul};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;
fn main() {
    let mut seed = 9u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for &n in &[256usize, 320, 384, 512] {
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        let b: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r()).collect()).collect();
        // byte-id serial vs default-new
        MATMUL_MACS_GATE_OVERRIDE.store(u64::MAX, Ordering::Relaxed);
        let cs = matmul(&a, &b).unwrap();
        MATMUL_MACS_GATE_OVERRIDE.store(0, Ordering::Relaxed);
        let cp = matmul(&a, &b).unwrap();
        let mism = cs
            .iter()
            .zip(&cp)
            .flat_map(|(p, q)| p.iter().zip(q))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mm = |g: u64| {
            MATMUL_MACS_GATE_OVERRIDE.store(g, Ordering::Relaxed);
            let _ = black_box(matmul(&a, &b).unwrap());
            let t = Instant::now();
            for _ in 0..6 {
                let _ = black_box(matmul(black_box(&a), black_box(&b)).unwrap());
            }
            t.elapsed().as_secs_f64() / 6.0 * 1000.0
        };
        let ser = mm(u64::MAX).min(mm(u64::MAX));
        let new = mm(0).min(mm(0));
        let mut spd = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i][k] * a[j][k];
                }
                spd[i][j] = s / n as f64 + if i == j { n as f64 } else { 0.0 };
            }
        }
        let ch = |g: u64| {
            MATMUL_MACS_GATE_OVERRIDE.store(g, Ordering::Relaxed);
            let _ = black_box(cholesky(&spd, true, DecompOptions::default()).unwrap());
            let t = Instant::now();
            for _ in 0..6 {
                let _ =
                    black_box(cholesky(black_box(&spd), true, DecompOptions::default()).unwrap());
            }
            t.elapsed().as_secs_f64() / 6.0 * 1000.0
        };
        let chs = ch(u64::MAX).min(ch(u64::MAX));
        let chn = ch(0).min(ch(0));
        println!(
            "n={n:4}: matmul serial {ser:6.2}->new {new:6.2}ms ({:.2}x) mism={mism} | cholesky {chs:6.2}->{chn:6.2}ms ({:.2}x, must be ~1.0)",
            ser / new,
            chs / chn
        );
    }
    MATMUL_MACS_GATE_OVERRIDE.store(0, Ordering::Relaxed);
}
