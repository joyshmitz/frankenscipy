#![feature(portable_simd)]
use std::simd::Simd;
#[unsafe(no_mangle)]
pub fn dot8(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = Simd::<f64, 8>::splat(0.0);
    let mut p = 0;
    while p + 8 <= x.len() {
        acc += Simd::<f64, 8>::from_slice(&x[p..p + 8]) * Simd::<f64, 8>::from_slice(&y[p..p + 8]);
        p += 8;
    }
    acc.to_array().iter().sum()
}
