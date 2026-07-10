#![feature(portable_simd)]
use std::simd::{Simd, StdFloat}; use std::hint::black_box;
type V=Simd<f64,8>;
#[inline(never)]  // SHIPPED: separate mul+add, single accumulator (bit-identical to naive)
fn base(x:&[f64],y:&[f64])->f64{let mut a=V::splat(0.0);let mut p=0;while p+8<=x.len(){a+=V::from_slice(&x[p..p+8])*V::from_slice(&y[p..p+8]);p+=8;}let mut s:f64=a.to_array().iter().sum();while p<x.len(){s+=x[p]*y[p];p+=1;}s}
#[inline(never)]  // BLAS-style microkernel: 4 accumulators + FMA (latency-hidden fused ops)
fn blas(x:&[f64],y:&[f64])->f64{let(mut a0,mut a1,mut a2,mut a3)=(V::splat(0.0),V::splat(0.0),V::splat(0.0),V::splat(0.0));let mut p=0;
while p+32<=x.len(){a0=V::from_slice(&x[p..p+8]).mul_add(V::from_slice(&y[p..p+8]),a0);a1=V::from_slice(&x[p+8..p+16]).mul_add(V::from_slice(&y[p+8..p+16]),a1);a2=V::from_slice(&x[p+16..p+24]).mul_add(V::from_slice(&y[p+16..p+24]),a2);a3=V::from_slice(&x[p+24..p+32]).mul_add(V::from_slice(&y[p+24..p+32]),a3);p+=32;}
let mut a=(a0+a1)+(a2+a3);while p+8<=x.len(){a=V::from_slice(&x[p..p+8]).mul_add(V::from_slice(&y[p..p+8]),a);p+=8;}let mut s:f64=a.to_array().iter().sum();while p<x.len(){s=x[p].mul_add(y[p],s);p+=1;}s}
fn main(){let a:Vec<String>=std::env::args().collect();let arm=a.get(1).map(|s|s.as_str()).unwrap_or("base");let n:usize=a.get(2).and_then(|s|s.parse().ok()).unwrap_or(256);let iters:usize=a.get(3).and_then(|s|s.parse().ok()).unwrap_or(2000000);
let x:Vec<f64>=(0..n).map(|i|(i as f64*0.017).sin()+1.1).collect();let y:Vec<f64>=(0..n).map(|i|(i as f64*0.017+0.3).sin()+1.1).collect();
if a.get(3).map(|s|s.as_str())==Some("bitcheck"){let mut d=0;for m in 0..500{let xx:Vec<f64>=(0..n).map(|i|((i*7+m)as f64*0.013).sin()).collect();let yy:Vec<f64>=(0..n).map(|i|((i*3+m)as f64*0.019).cos()).collect();if base(&xx,&yy).to_bits()!=blas(&xx,&yy).to_bits(){d+=1;}}println!("base vs blas differ in {}/500 random inputs",d);return;}
let f=|_k:usize|->f64{match arm{"base"=>base(black_box(&x),black_box(&y)),_=>blas(black_box(&x),black_box(&y))}};
let t=std::time::Instant::now();let mut s=0.0;for k in 0..iters{s+=black_box(f(k));}println!("{:.6}",t.elapsed().as_secs_f64()/iters as f64*1e6);black_box(s);}
