// Same-process A/B for affinity_propagation storage layout: the shipped library
// path (flat row-major `Vec<f64>` for the responsibility/availability matrices)
// vs `ap_vecvec` below (the legacy `Vec<Vec<f64>>` layout, faithful copy of the
// pre-change algorithm). Same input, same process, same worker => no cross-worker
// noise. Both must return BYTE-IDENTICAL labels (the layout change is pure
// storage; arithmetic and order are unchanged). The win is the availability
// update's column walk over the n*n matrices: with `Vec<Vec>` each `r[i][k]` for
// varying i chases a different scattered heap row; flat gives one base pointer +
// a predictable stride-n walk the prefetcher follows.
use fsci_cluster::affinity_propagation;
use std::time::Instant;

// Faithful copy of the legacy Vec<Vec<f64>> affinity_propagation (labels only).
fn ap_vecvec(
    similarity: &[Vec<f64>],
    preference: f64,
    damping: f64,
    max_iter: usize,
    convergence_iter: usize,
) -> Vec<usize> {
    let n = similarity.len();
    let mut s: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|k| if i == k { preference } else { similarity[i][k] })
                .collect()
        })
        .collect();
    let smax = s
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let mut st: u64 = 0x9e37_79b9_7f4a_7c15;
    for row in s.iter_mut() {
        for v in row.iter_mut() {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
            *v += 1e-10 * smax * u;
        }
    }

    let mut r = vec![vec![0.0f64; n]; n];
    let mut a = vec![vec![0.0f64; n]; n];
    let mut last_exemplars: Vec<usize> = Vec::new();
    let mut stable = 0usize;

    for _ in 0..max_iter {
        let update_resp_row = |i: usize, r_row: &mut [f64]| {
            let (mut max1, mut max1_idx, mut max2) = (f64::NEG_INFINITY, 0usize, f64::NEG_INFINITY);
            for k in 0..n {
                let v = a[i][k] + s[i][k];
                if v > max1 {
                    max2 = max1;
                    max1 = v;
                    max1_idx = k;
                } else if v > max2 {
                    max2 = v;
                }
            }
            for (k, rik) in r_row.iter_mut().enumerate() {
                let competitor = if k == max1_idx { max2 } else { max1 };
                let upd = s[i][k] - competitor;
                *rik = damping * *rik + (1.0 - damping) * upd;
            }
        };
        let nthreads_r = if n.saturating_mul(n) < (1 << 18) || n < 2 {
            1
        } else {
            std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(1)
                .min(n)
        };
        if nthreads_r <= 1 {
            for (i, r_row) in r.iter_mut().enumerate() {
                update_resp_row(i, r_row);
            }
        } else {
            let chunk = n.div_ceil(nthreads_r);
            let update_resp_row = &update_resp_row;
            std::thread::scope(|scope| {
                for (t, r_chunk) in r.chunks_mut(chunk).enumerate() {
                    let base = t * chunk;
                    scope.spawn(move || {
                        for (li, r_row) in r_chunk.iter_mut().enumerate() {
                            update_resp_row(base + li, r_row);
                        }
                    });
                }
            });
        }
        for k in 0..n {
            let col_pos: f64 = (0..n).map(|i| r[i][k].max(0.0)).sum();
            let rkk = r[k][k];
            let pos_kk = rkk.max(0.0);
            for i in 0..n {
                let upd = if i == k {
                    col_pos - pos_kk
                } else {
                    (rkk + col_pos - r[i][k].max(0.0) - pos_kk).min(0.0)
                };
                a[i][k] = damping * a[i][k] + (1.0 - damping) * upd;
            }
        }
        let exemplars: Vec<usize> = (0..n).filter(|&k| r[k][k] + a[k][k] > 0.0).collect();
        if !exemplars.is_empty() && exemplars == last_exemplars {
            stable += 1;
            if stable >= convergence_iter {
                break;
            }
        } else {
            stable = 0;
            last_exemplars = exemplars;
        }
    }

    let mut exemplars: Vec<usize> = (0..n).filter(|&k| r[k][k] + a[k][k] > 0.0).collect();
    if exemplars.is_empty() {
        let best = (0..n)
            .max_by(|&p, &q| (r[p][p] + a[p][p]).total_cmp(&(r[q][q] + a[q][q])))
            .unwrap_or(0);
        exemplars.push(best);
    }
    (0..n)
        .map(|i| {
            exemplars
                .iter()
                .enumerate()
                .max_by(|&(_, &p), &(_, &q)| s[i][p].total_cmp(&s[i][q]))
                .map_or(0, |(c, _)| c)
        })
        .collect()
}

fn build_sim(k: usize, per: usize) -> (Vec<Vec<f64>>, f64) {
    let n = k * per;
    let mut s: u64 = 0x51a4_b3c2_d1e0_f9a8;
    let mut rng = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let mut pts = Vec::new();
    for c in 0..k {
        for _ in 0..per {
            pts.push([20.0 * c as f64 + rng(), rng()]);
        }
    }
    let mut sim = vec![vec![0.0; n]; n];
    let mut offdiag = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let d2 = (pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2);
                sim[i][j] = -d2;
                offdiag.push(-d2);
            }
        }
    }
    offdiag.sort_unstable_by(|a, b| a.total_cmp(b));
    let preference = offdiag[offdiag.len() / 2];
    (sim, preference)
}

fn best_of<F: FnMut() -> Vec<usize>>(reps: usize, mut f: F) -> (std::time::Duration, Vec<usize>) {
    let mut best = std::time::Duration::MAX;
    let mut out = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let labels = f();
        let e = t.elapsed();
        if e < best {
            best = e;
        }
        out = labels;
    }
    (best, out)
}

fn main() {
    let damping = 0.9;
    let max_iter = 200usize;
    let conv = 15usize;
    println!(
        "{:>6} {:>14} {:>14} {:>9}  labels",
        "n", "vecvec_ms", "flat_ms", "speedup"
    );
    for &(k, per) in &[(8usize, 50usize), (10, 80), (12, 100)] {
        let n = k * per;
        let (sim, pref) = build_sim(k, per);
        let (t_vv, lab_vv) = best_of(3, || ap_vecvec(&sim, pref, damping, max_iter, conv));
        let (t_flat, lab_flat) = best_of(3, || {
            affinity_propagation(&sim, pref, damping, max_iter, conv)
                .unwrap()
                .labels
        });
        let identical = lab_vv == lab_flat;
        let vv_ms = t_vv.as_secs_f64() * 1e3;
        let flat_ms = t_flat.as_secs_f64() * 1e3;
        println!(
            "{n:>6} {vv_ms:>14.3} {flat_ms:>14.3} {:>8.2}x  {}",
            vv_ms / flat_ms,
            if identical {
                "IDENTICAL"
            } else {
                "*** DIFFER ***"
            }
        );
    }
}
