//! Median-null-gated A/B for `mfcc`'s per-frame feature loop: ORIG serial frame walk vs frame-chunk
//! parallel (each frame's FFT+melbank+DCT is independent). BYTE-IDENTICAL (pure per-frame kernel,
//! ordered slots). Toggled by `MFCC_FORCE_SERIAL`. Args: signal_len [frame_len] [iters].
use fsci_signal::{MFCC_FORCE_SERIAL, mfcc};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn cv(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let siglen: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4_000_000);
    let frame_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);
    let (sr, n_mfcc, n_mels, hop_len) = (16000.0_f64, 13usize, 40usize, 512usize);

    let mut s = 0x7a1c_bd93u64;
    let signal: Vec<f64> = (0..siglen)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect();

    MFCC_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let a = mfcc(&signal, sr, n_mfcc, n_mels, frame_len, hop_len);
    MFCC_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let b = mfcc(&signal, sr, n_mfcc, n_mels, frame_len, hop_len);
    let bitmism: usize = a
        .iter()
        .zip(&b)
        .map(|(ra, rb)| ra.iter().zip(rb).filter(|(x, y)| x.to_bits() != y.to_bits()).count())
        .sum::<usize>()
        + usize::from(a.len() != b.len());
    println!("# signal::mfcc siglen={siglen} frame_len={frame_len} nframes={} bitmism={bitmism}", a.len());

    let bench = |serial: bool| -> f64 {
        MFCC_FORCE_SERIAL.store(serial, Ordering::Relaxed);
        let _ = black_box(mfcc(black_box(&signal), sr, n_mfcc, n_mels, frame_len, hop_len));
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(mfcc(black_box(&signal), sr, n_mfcc, n_mels, frame_len, hop_len));
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut fv, mut nr, mut cr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let f = bench(false);
        let o2 = bench(true);
        nr.push(o / o2);
        cr.push(o / f);
        ov.push(o);
        fv.push(f);
    }
    MFCC_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} mfcc serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | \
         CAND(serial/par) median {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
