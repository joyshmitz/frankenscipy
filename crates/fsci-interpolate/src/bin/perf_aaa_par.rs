//! Median-null-gated A/B for `Aaa::eval_many` parallelisation: serial `iter().map(eval)` vs the
//! new `par_query_map`-backed `eval_many`, on a workload above the parallel gate. Both arms live in
//! ONE binary and are ALTERNATED per iteration inside one measured routine, so a single `rch exec`
//! invocation measures both on the same worker.
use fsci_interpolate::Aaa;
use std::hint::black_box;
use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
fn cv(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64;
    if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(15);

    // Build an AAA approximant, then evaluate a large query batch (above the m·support ≥ 2^23 gate).
    let z: Vec<f64> = (0..200).map(|i| -1.0 + i as f64 * (2.0 / 199.0)).collect();
    let f: Vec<f64> = z
        .iter()
        .map(|&x| 1.0 / (x + 1.5) + 0.5 / (x - 2.0) + (3.0 * x).sin())
        .collect();
    let a = Aaa::new(&z, &f, None, 100).unwrap();
    let support = a.support_points().len().max(1);
    let m = ((1usize << 23) / support) * 4 + 8192; // comfortably above the gate
    let xs: Vec<f64> = (0..m)
        .map(|i| -1.3 + (i as f64) * 2.6 / (m as f64))
        .collect();
    println!(
        "# Aaa::eval_many serial vs parallel | support={support} queries={m} (m·support={:.1}M, gate=8.4M)",
        (m * support) as f64 / 1e6
    );

    // Parity: parallel must be byte-identical to serial.
    let par = a.eval_many(&xs);
    let ser: Vec<f64> = xs.iter().map(|&x| a.eval(x)).collect();
    let bitmism = par
        .iter()
        .zip(&ser)
        .filter(|(p, s)| p.to_bits() != s.to_bits())
        .count();

    let bench = |parallel: bool| -> f64 {
        let run = || {
            if parallel {
                a.eval_many(black_box(&xs))
            } else {
                xs.iter()
                    .map(|&x| a.eval(black_box(x)))
                    .collect::<Vec<f64>>()
            }
        };
        drop(black_box(run()));
        let t = Instant::now();
        for _ in 0..reps {
            drop(black_box(run()));
        }
        t.elapsed().as_secs_f64() / reps as f64 * 1e3
    };

    // Interleave serial / parallel / serial-again (A/A null) inside one routine.
    let (mut sv, mut pv, mut nv) = (Vec::new(), Vec::new(), Vec::new());
    let (mut null_r, mut cand_r) = (Vec::new(), Vec::new());
    for _ in 0..iters {
        let s = bench(false);
        let p = bench(true);
        let s2 = bench(false);
        null_r.push(s / s2);
        cand_r.push(s / p);
        sv.push(s);
        pv.push(p);
        nv.push(s2);
    }
    let (mut nr, mut cr) = (null_r.clone(), cand_r.clone());
    let null_med = median(&mut nr);
    let cand_med = median(&mut cr);
    let (null_lo, null_hi) = (
        nr.iter().copied().fold(f64::MAX, f64::min),
        nr.iter().copied().fold(f64::MIN, f64::max),
    );
    let decidable = cand_med > null_hi || cand_med < null_lo;
    let sb = sv.iter().copied().fold(f64::MAX, f64::min);
    let pb = pv.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "{} serial {sb:.2}ms (cv {:.1}%) parallel {pb:.2}ms (cv {:.1}%) | CAND median {cand_med:.3}x | \
         NULL(A/A) median {null_med:.3}x range [{null_lo:.3}, {null_hi:.3}] | bitmism={bitmism}",
        if decidable { "DECIDED " } else { "IN-FLOOR" },
        cv(&sv),
        cv(&pv),
    );
}
