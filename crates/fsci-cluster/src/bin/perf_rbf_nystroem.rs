//! Median-null-gated A/B for `cluster::rbf_nystroem`: the ORIG serial `E`-block fill
//! (`data.iter().map(|xi| landmarks.iter().map(rbf).collect()).collect()`) vs the parallel
//! row-fill in `nystroem_e_block` (each n×m row is an independent O(m·d) run of `exp`-heavy
//! RBF evals → filled across threads, chunk order preserved). Toggled by the shared
//! `NYSTROEM_FORCE_SERIAL` and ALTERNATED per iteration. BYTE-IDENTICAL (the whole
//! `feature_map`): each row is computed by the same closure in the same intra-row order, and
//! the downstream `W^{-1/2}` / `matmul` are identical in both arms. sklearn has the peer
//! (`kernel_approximation.Nystroem`); scipy has none — byte-identical self-speedup.
//!
//! Run with `d >> m` so the O(n·m·d) `exp` fill dominates the O(n·m²) matmul.
use fsci_cluster::{NYSTROEM_FORCE_SERIAL, rbf_nystroem};
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
// Count differing f64 bit patterns across the flattened feature_map.
fn bitmism(a: &[Vec<f64>], b: &[Vec<f64>]) -> usize {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
        + usize::from(a.len() != b.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40_000);
    let d: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    let m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(15);
    let gamma = 1.0 / d as f64;
    let seed = 0xC0FF_EE12_3456_789Au64;

    // Deterministic n×d data in a few blobs so distances span a useful range.
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let center = (i % 8) as f64;
            (0..d).map(|_| center + r()).collect()
        })
        .collect();

    // Parity: the whole feature_map must be bit-identical across arms.
    NYSTROEM_FORCE_SERIAL.store(true, Ordering::Relaxed);
    let ra = rbf_nystroem(&data, m, gamma, seed).unwrap();
    NYSTROEM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let rb = rbf_nystroem(&data, m, gamma, seed).unwrap();
    let mism = bitmism(&ra.feature_map, &rb.feature_map)
        + usize::from(ra.landmark_indices != rb.landmark_indices);

    let bench = |force_serial: bool| -> f64 {
        NYSTROEM_FORCE_SERIAL.store(force_serial, Ordering::Relaxed);
        let run = || rbf_nystroem(black_box(&data), black_box(m), black_box(gamma), seed).unwrap();
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
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
    NYSTROEM_FORCE_SERIAL.store(false, Ordering::Relaxed);
    let cand_med = med(&mut cr.clone());
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let fb = fv.iter().copied().fold(f64::MAX, f64::min);
    println!("# cluster::rbf_nystroem n={n} d={d} m={m}");
    println!(
        "{} serial {ob:.2}ms (cv {:.1}%) parallel {fb:.2}ms (cv {:.1}%) | CAND(serial/parallel) median \
         {cand_med:.3}x | NULL(A/A) range [{null_lo:.3}, {null_hi:.3}] | bitmism={mism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&fv),
    );
}
