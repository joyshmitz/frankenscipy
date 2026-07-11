//! Median-null-gated A/B for `signal::filtfilt_axis_2d`: the ORIG per-line path (recomputes the
//! query-independent `lfilter_zi(b, a)` O(order³) dense solve on EVERY line) vs the hoisted path
//! (solves `lfilter_zi` ONCE, reuses it across all lines). Both arms live in ONE binary, toggled by
//! `FILTFILT_AXIS_HOIST_DISABLE` and ALTERNATED per iteration. The hoist win grows with filter order
//! (bigger `lfilter_zi` solve) and shrinks with line length (more per-line filter work to amortize
//! against), so this is decisive for HIGH-ORDER filters on many modest-length lines.
use fsci_signal::{FILTFILT_AXIS_HOIST_DISABLE, filtfilt_axis_2d};
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
    let order: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(14);
    let rows: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6000);
    let cols: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(350);
    let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(21);

    let mut s = 0x243f6a8885a308d3u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    // High-order filter with small coefficients → poles near the origin (stable, well-conditioned
    // lfilter_zi solve). a[0] = 1 by convention.
    let mut a = vec![1.0f64];
    a.extend((0..order).map(|_| 0.08 * r()));
    let b: Vec<f64> = (0..=order).map(|_| 0.1 * r()).collect();
    // 2-D input: `rows` independent lines of length `cols`, filtered along axis 1 (rows).
    let x: Vec<Vec<f64>> = (0..rows).map(|_| (0..cols).map(|_| r()).collect()).collect();

    // Parity: hoisted must be byte-identical to the per-line path over every element.
    FILTFILT_AXIS_HOIST_DISABLE.store(true, Ordering::Relaxed);
    let ra = filtfilt_axis_2d(&b, &a, &x, 1).unwrap();
    FILTFILT_AXIS_HOIST_DISABLE.store(false, Ordering::Relaxed);
    let rb = filtfilt_axis_2d(&b, &a, &x, 1).unwrap();
    let bitmism = ra
        .iter()
        .zip(&rb)
        .map(|(row_a, row_b)| {
            row_a.iter().zip(row_b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
                + usize::from(row_a.len() != row_b.len())
        })
        .sum::<usize>()
        + usize::from(ra.len() != rb.len());

    let bench = |disable_hoist: bool| -> f64 {
        FILTFILT_AXIS_HOIST_DISABLE.store(disable_hoist, Ordering::Relaxed);
        let run = || filtfilt_axis_2d(black_box(&b), black_box(&a), black_box(&x), 1).unwrap();
        let _ = black_box(run());
        let t = Instant::now();
        for _ in 0..3 {
            let _ = black_box(run());
        }
        t.elapsed().as_secs_f64() / 3.0 * 1e3
    };

    let (mut ov, mut hv) = (Vec::new(), Vec::new());
    let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let o = bench(true);
        let h = bench(false);
        let o2 = bench(true);
        null_r.push(o / o2);
        cand_r.push(o / h);
        ov.push(o);
        hv.push(h);
    }
    FILTFILT_AXIS_HOIST_DISABLE.store(false, Ordering::Relaxed);
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = med(&mut nr);
    let cand_med = med(&mut cr);
    let null_lo = nr.iter().copied().fold(f64::MAX, f64::min);
    let null_hi = nr.iter().copied().fold(f64::MIN, f64::max);
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let ob = ov.iter().copied().fold(f64::MAX, f64::min);
    let hb = hv.iter().copied().fold(f64::MAX, f64::min);
    println!("# signal::filtfilt_axis_2d order={order} rows={rows} cols={cols}");
    println!(
        "{} per-line-zi {ob:.2}ms (cv {:.1}%) hoisted {hb:.2}ms (cv {:.1}%) | CAND(orig/hoisted) median \
         {cand_med:.3}x | NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&ov),
        cv(&hv),
    );
}
