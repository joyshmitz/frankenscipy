#![feature(portable_simd)]
// Faithful copies of fsci-linalg's REAL dense inner loops (lib.rs), to isolate the effect of the
// +avx2,+fma build flag on each kernel family. Same source, compiled SSE2 vs AVX2.
//   simd_dot            -> the dot in panel TRSM / matmul (dtrsm/dgemm inner)
//   simd_dot2_shared_rhs-> cod's shipped MR2 panel TRSM (770c4d490 / a6d7ba897)
//   simd_dot4           -> the SYRK 4-dot row update (cholesky_syrk_flat_rows)
//   syrk_axpy           -> the trailing-update axpy  dst -= l*src
use std::simd::Simd;
use std::hint::black_box;

#[inline(never)]
fn simd_dot(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = Simd::<f64, 8>::splat(0.0);
    let mut p = 0;
    while p + 8 <= x.len() { acc += Simd::<f64,8>::from_slice(&x[p..p+8]) * Simd::<f64,8>::from_slice(&y[p..p+8]); p += 8; }
    let mut s: f64 = acc.to_array().iter().sum();
    while p < x.len() { s += x[p]*y[p]; p += 1; }
    s
}
#[inline(never)]
fn simd_dot2(l0: &[f64], l1: &[f64], r: &[f64]) -> [f64;2] {
    let (mut a0, mut a1) = (Simd::<f64,8>::splat(0.0), Simd::<f64,8>::splat(0.0));
    let mut p = 0;
    while p + 8 <= r.len() { let rr = Simd::<f64,8>::from_slice(&r[p..p+8]);
        a0 += Simd::<f64,8>::from_slice(&l0[p..p+8])*rr; a1 += Simd::<f64,8>::from_slice(&l1[p..p+8])*rr; p += 8; }
    [a0.to_array().iter().sum(), a1.to_array().iter().sum()]
}
#[inline(never)]
fn simd_dot4(l: &[f64], r0:&[f64],r1:&[f64],r2:&[f64],r3:&[f64]) -> [f64;4] {
    let (mut a0,mut a1,mut a2,mut a3)=(Simd::<f64,8>::splat(0.0),Simd::<f64,8>::splat(0.0),Simd::<f64,8>::splat(0.0),Simd::<f64,8>::splat(0.0));
    let mut p=0;
    while p+8<=l.len(){ let lv=Simd::<f64,8>::from_slice(&l[p..p+8]);
        a0+=lv*Simd::<f64,8>::from_slice(&r0[p..p+8]); a1+=lv*Simd::<f64,8>::from_slice(&r1[p..p+8]);
        a2+=lv*Simd::<f64,8>::from_slice(&r2[p..p+8]); a3+=lv*Simd::<f64,8>::from_slice(&r3[p..p+8]); p+=8; }
    [a0.to_array().iter().sum(),a1.to_array().iter().sum(),a2.to_array().iter().sum(),a3.to_array().iter().sum()]
}
#[inline(never)]
fn syrk_axpy(dst:&mut [f64], l:f64, src:&[f64]){
    let lv=Simd::<f64,8>::splat(l); let mut t=0;
    while t+8<=dst.len(){ (Simd::<f64,8>::from_slice(&dst[t..t+8])-lv*Simd::<f64,8>::from_slice(&src[t..t+8])).copy_to_slice(&mut dst[t..t+8]); t+=8; }
}

fn main(){
    let a:Vec<String>=std::env::args().collect();
    let kern = a.get(1).map(|s|s.as_str()).unwrap_or("dot");
    let iters:usize = a.get(2).and_then(|s|s.parse().ok()).unwrap_or(200000);
    let checksum = a.get(3).map(|s|s.as_str())==Some("checksum");
    let n=2048usize;
    let mk=|o:f64|->Vec<f64>{(0..n).map(|i|((i as f64*0.017+o).sin())+1.1).collect()};
    let (x,y,z,w,v)=(mk(0.0),mk(0.3),mk(0.6),mk(0.9),mk(1.2));
    let mut dst=mk(2.0);
    let mut run=|k:usize|->f64{ match kern {
        "dot"  => simd_dot(black_box(&x),black_box(&y)),
        "dot2" => { let r=simd_dot2(black_box(&x),black_box(&y),black_box(&z)); r[0]+r[1] }
        "dot4" => { let r=simd_dot4(black_box(&x),black_box(&y),black_box(&z),black_box(&w),black_box(&v)); r.iter().sum::<f64>() }
        "axpy" => { syrk_axpy(&mut dst,black_box(1.0+(k&7) as f64*1e-12),black_box(&x)); dst[k%n] }
        _=>0.0 } };
    if checksum { let mut h=0u64; for k in 0..128 { h=h.wrapping_mul(1099511628211).wrapping_add(run(k).to_bits()); } println!("{h:016x}"); return; }
    let t=std::time::Instant::now();
    let mut s=0.0; for k in 0..iters { s+=black_box(run(k)); }
    println!("{:.6}", t.elapsed().as_secs_f64()/iters as f64*1e6);
    black_box(s);
}
