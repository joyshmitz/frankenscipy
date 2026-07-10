#![feature(portable_simd)]
// Mirrors fsci-linalg's dense inner loop: the SYRK/TRSM row update  dst[t] -= l * src[t]
// over Simd<f64,8> (lib.rs ~17755), and the packed dot (~18337). Pure compute, f64.
use std::simd::Simd;
use std::hint::black_box;

#[inline(never)]
fn syrk_row_update(dst: &mut [f64], l: f64, src: &[f64]) {
    let lv = Simd::<f64, 8>::splat(l);
    let mut t = 0;
    while t + 8 <= dst.len() {
        let d = Simd::<f64, 8>::from_slice(&dst[t..t + 8])
            - lv * Simd::<f64, 8>::from_slice(&src[t..t + 8]);
        d.copy_to_slice(&mut dst[t..t + 8]);
        t += 8;
    }
}

fn main() {
    let n = 2048usize;
    let src: Vec<f64> = (0..n).map(|i| (i as f64 * 0.123).sin() + 1.3).collect();
    let mut dst: Vec<f64> = (0..n).map(|i| (i as f64 * 0.071).cos() - 0.4).collect();
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200000);
    // checksum mode: print the result hash so two ISA builds can be proven bit-identical
    if std::env::args().nth(2).as_deref() == Some("checksum") {
        for k in 0..64 { syrk_row_update(&mut dst, 1.0 + k as f64 * 1e-9, &src); }
        let bits: u64 = dst.iter().fold(0u64, |acc, x| acc.wrapping_mul(1099511628211).wrapping_add(x.to_bits()));
        println!("{bits:016x}");
        return;
    }
    let t = std::time::Instant::now();
    for k in 0..iters {
        syrk_row_update(black_box(&mut dst), black_box(1.0 + (k & 7) as f64 * 1e-12), black_box(&src));
    }
    let ms = t.elapsed().as_secs_f64() / iters as f64 * 1e6; // us/call
    println!("{ms:.6}");
    black_box(dst);
}
