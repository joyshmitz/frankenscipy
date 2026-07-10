#![feature(portable_simd)]
use std::simd::{Simd, StdFloat}; use std::hint::black_box;
#[inline(never)]  // shipped: separate mul + add (fp-contract=off, bit-identical to naive)
fn dot_noFMA(x:&[f64],y:&[f64])->f64{let mut a=Simd::<f64,8>::splat(0.0);let mut p=0;while p+8<=x.len(){a+=Simd::<f64,8>::from_slice(&x[p..p+8])*Simd::<f64,8>::from_slice(&y[p..p+8]);p+=8;}let mut s:f64=a.to_array().iter().sum();while p<x.len(){s+=x[p]*y[p];p+=1;}s}
#[inline(never)]  // explicit FMA via mul_add -> emits vfmadd (1 instr, single rounding)
fn dot_FMA(x:&[f64],y:&[f64])->f64{let mut a=Simd::<f64,8>::splat(0.0);let mut p=0;while p+8<=x.len(){a=Simd::<f64,8>::from_slice(&x[p..p+8]).mul_add(Simd::<f64,8>::from_slice(&y[p..p+8]),a);p+=8;}let mut s:f64=a.to_array().iter().sum();while p<x.len(){s=x[p].mul_add(y[p],s);p+=1;}s}
fn main(){let a:Vec<String>=std::env::args().collect();let arm=a.get(1).map(|s|s.as_str()).unwrap_or("noFMA");let n:usize=a.get(2).and_then(|s|s.parse().ok()).unwrap_or(256);let iters:usize=a.get(3).and_then(|s|s.parse().ok()).unwrap_or(2000000);
let x:Vec<f64>=(0..n).map(|i|(i as f64*0.017).sin()+1.1).collect();let y:Vec<f64>=(0..n).map(|i|(i as f64*0.017+0.3).sin()+1.1).collect();
if a.get(3).map(|s|s.as_str())==Some("bitcheck"){println!("noFMA={:016x} FMA={:016x} {}",dot_noFMA(&x,&y).to_bits(),dot_FMA(&x,&y).to_bits(),if dot_noFMA(&x,&y).to_bits()==dot_FMA(&x,&y).to_bits(){"BIT-IDENTICAL"}else{"DIFFER"});return;}
let f=|_k:usize|->f64{match arm{"noFMA"=>dot_noFMA(black_box(&x),black_box(&y)),_=>dot_FMA(black_box(&x),black_box(&y))}};
let t=std::time::Instant::now();let mut s=0.0;for k in 0..iters{s+=black_box(f(k));}println!("{:.6}",t.elapsed().as_secs_f64()/iters as f64*1e6);black_box(s);}
