#![feature(portable_simd)]
use std::simd::Simd; use std::hint::black_box;
#[inline(never)]
fn dot_1acc(x:&[f64],y:&[f64])->f64{let mut a=Simd::<f64,8>::splat(0.0);let mut p=0;while p+8<=x.len(){a+=Simd::<f64,8>::from_slice(&x[p..p+8])*Simd::<f64,8>::from_slice(&y[p..p+8]);p+=8;}let mut s:f64=a.to_array().iter().sum();while p<x.len(){s+=x[p]*y[p];p+=1;}s}
#[inline(never)]
fn dot_4acc(x:&[f64],y:&[f64])->f64{let(mut a0,mut a1,mut a2,mut a3)=(Simd::<f64,8>::splat(0.0),Simd::<f64,8>::splat(0.0),Simd::<f64,8>::splat(0.0),Simd::<f64,8>::splat(0.0));let mut p=0;while p+32<=x.len(){a0+=Simd::<f64,8>::from_slice(&x[p..p+8])*Simd::<f64,8>::from_slice(&y[p..p+8]);a1+=Simd::<f64,8>::from_slice(&x[p+8..p+16])*Simd::<f64,8>::from_slice(&y[p+8..p+16]);a2+=Simd::<f64,8>::from_slice(&x[p+16..p+24])*Simd::<f64,8>::from_slice(&y[p+16..p+24]);a3+=Simd::<f64,8>::from_slice(&x[p+24..p+32])*Simd::<f64,8>::from_slice(&y[p+24..p+32]);p+=32;}let mut a=(a0+a1)+(a2+a3);while p+8<=x.len(){a+=Simd::<f64,8>::from_slice(&x[p..p+8])*Simd::<f64,8>::from_slice(&y[p..p+8]);p+=8;}let mut s:f64=a.to_array().iter().sum();while p<x.len(){s+=x[p]*y[p];p+=1;}s}
fn main(){let a:Vec<String>=std::env::args().collect();let arm=a.get(1).map(|s|s.as_str()).unwrap_or("1acc");let n:usize=a.get(2).and_then(|s|s.parse().ok()).unwrap_or(256);let iters:usize=a.get(3).and_then(|s|s.parse().ok()).unwrap_or(2000000);
let x:Vec<f64>=(0..n).map(|i|(i as f64*0.017).sin()+1.1).collect();let y:Vec<f64>=(0..n).map(|i|(i as f64*0.017+0.3).sin()+1.1).collect();
let f=|_k:usize|->f64{match arm{"1acc"=>dot_1acc(black_box(&x),black_box(&y)),_=>dot_4acc(black_box(&x),black_box(&y))}};
let t=std::time::Instant::now();let mut s=0.0;for k in 0..iters{s+=black_box(f(k));}println!("{:.6}",t.elapsed().as_secs_f64()/iters as f64*1e6);black_box(s);}
