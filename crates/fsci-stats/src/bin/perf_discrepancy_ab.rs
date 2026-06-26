// Same-process A/B for the discrepancy upper-triangle symmetry lever, all four
// methods. Each `old_full_*` is the pre-change full-n^2 double loop; the library
// fns now sum the diagonal once + each off-diagonal pair twice (the pair product
// is symmetric). Measured d>=3 so BOTH paths use the general kernel and the only
// difference is the symmetry. Same process / same worker => no cross-worker
// noise; values must agree to ~1e-9 (sum reassociation only).
use fsci_stats::{
    centered_discrepancy, centered_discrepancy_iterative, l2_star_discrepancy, mixture_discrepancy,
    wraparound_discrepancy,
};
use std::time::Instant;

fn old_full_iter_centered(s: &[f64], d: usize) -> f64 {
    // Same kernel as old_full_centered but with the iterative (n+1) normalization.
    let n = s.len() / d;
    let leading = (13.0_f64 / 12.0).powi(d as i32);
    let mut single = 0.0;
    for i in 0..n {
        let mut p = 1.0;
        for k in 0..d {
            let c = s[i * d + k] - 0.5;
            p *= 1.0 + 0.5 * c.abs() - 0.5 * c * c;
        }
        single += p;
    }
    let mut double = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut p = 1.0;
            for k in 0..d {
                let (xi, xj) = (s[i * d + k], s[j * d + k]);
                p *= 1.0 + 0.5 * (xi - 0.5).abs() + 0.5 * (xj - 0.5).abs() - 0.5 * (xi - xj).abs();
            }
            double += p;
        }
    }
    let m = (n + 1) as f64;
    leading - 2.0 / m * single + double / (m * m)
}

fn old_full_centered(s: &[f64], d: usize) -> f64 {
    let n = s.len() / d;
    let leading = (13.0_f64 / 12.0).powi(d as i32);
    let mut single = 0.0;
    for i in 0..n {
        let mut p = 1.0;
        for k in 0..d {
            let c = s[i * d + k] - 0.5;
            p *= 1.0 + 0.5 * c.abs() - 0.5 * c * c;
        }
        single += p;
    }
    let mut double = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut p = 1.0;
            for k in 0..d {
                let (xi, xj) = (s[i * d + k], s[j * d + k]);
                p *= 1.0 + 0.5 * (xi - 0.5).abs() + 0.5 * (xj - 0.5).abs() - 0.5 * (xi - xj).abs();
            }
            double += p;
        }
    }
    let nf = n as f64;
    leading - 2.0 / nf * single + double / (nf * nf)
}

fn old_full_mixture(s: &[f64], d: usize) -> f64 {
    let n = s.len() / d;
    let leading = (19.0_f64 / 12.0).powi(d as i32);
    let mut single = 0.0;
    for i in 0..n {
        let mut p = 1.0;
        for k in 0..d {
            let c = s[i * d + k] - 0.5;
            p *= 5.0 / 3.0 - 0.25 * c.abs() - 0.25 * c * c;
        }
        single += p;
    }
    let mut double = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut p = 1.0;
            for k in 0..d {
                let (xi, xj) = (s[i * d + k], s[j * d + k]);
                let dd = (xi - xj).abs();
                p *= 15.0 / 8.0 - 0.25 * (xi - 0.5).abs() - 0.25 * (xj - 0.5).abs() - 0.75 * dd
                    + 0.5 * (xi - xj).powi(2);
            }
            double += p;
        }
    }
    let nf = n as f64;
    leading - 2.0 / nf * single + double / (nf * nf)
}

fn old_full_l2_star(s: &[f64], d: usize) -> f64 {
    let n = s.len() / d;
    let leading = (1.0_f64 / 3.0).powi(d as i32);
    let two_pow = 2.0_f64.powi(1 - d as i32);
    let mut single = 0.0;
    for i in 0..n {
        let mut p = 1.0;
        for k in 0..d {
            let x = s[i * d + k];
            p *= 1.0 - x * x;
        }
        single += p;
    }
    let mut double = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut p = 1.0;
            for k in 0..d {
                p *= 1.0 - s[i * d + k].max(s[j * d + k]);
            }
            double += p;
        }
    }
    let nf = n as f64;
    (leading - two_pow / nf * single + double / (nf * nf)).sqrt()
}

fn old_full_wraparound(s: &[f64], d: usize) -> f64 {
    let n = s.len() / d;
    let leading = -(4.0_f64 / 3.0).powi(d as i32);
    let mut double = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut p = 1.0;
            for k in 0..d {
                let dd = (s[i * d + k] - s[j * d + k]).abs();
                p *= 1.5 - dd * (1.0 - dd);
            }
            double += p;
        }
    }
    let nf = n as f64;
    leading + double / (nf * nf)
}

fn build_sample(n: usize, d: usize) -> Vec<f64> {
    let mut s: u64 = 0x243f_6a88_85a3_08d3;
    (0..n * d)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

fn best_of(reps: usize, mut f: impl FnMut() -> f64) -> (std::time::Duration, f64) {
    let mut best = std::time::Duration::MAX;
    let mut val = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        val = std::hint::black_box(f());
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, val)
}

fn main() {
    type OldFn = fn(&[f64], usize) -> f64;
    type NewFn = fn(&[f64], usize) -> Result<f64, fsci_stats::StatsError>;
    let methods: [(&str, OldFn, NewFn); 5] = [
        ("centered", old_full_centered, centered_discrepancy),
        ("mixture", old_full_mixture, mixture_discrepancy),
        ("l2_star", old_full_l2_star, l2_star_discrepancy),
        ("wrap", old_full_wraparound, wraparound_discrepancy),
        (
            "cd_iter",
            old_full_iter_centered,
            centered_discrepancy_iterative,
        ),
    ];
    println!(
        "{:>8} {:>6} {:>3} {:>11} {:>11} {:>8}  {:>9}",
        "method", "n", "d", "old_us", "new_us", "speedup", "valdiff"
    );
    for &(n, d) in &[(512usize, 4usize), (1024, 4)] {
        let s = build_sample(n, d);
        for (name, oldf, newf) in methods {
            let (t_old, v_old) = best_of(5, || oldf(&s, d));
            let (t_new, v_new) = best_of(5, || newf(&s, d).unwrap());
            let old_us = t_old.as_secs_f64() * 1e6;
            let new_us = t_new.as_secs_f64() * 1e6;
            println!(
                "{name:>8} {n:>6} {d:>3} {old_us:>11.2} {new_us:>11.2} {:>7.2}x  {:>9.2e}",
                old_us / new_us,
                (v_old - v_new).abs()
            );
        }
    }
}
