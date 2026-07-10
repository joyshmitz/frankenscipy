//! A/B: cholesky blocked (parallel SYRK) vs unblocked left-looking SIMD at medium n.
//!
//! Substrate: BOTH arms live in this ONE binary and are selected in-process by the public
//! `CHOL_FACTOR_FLAT_MIN_OVERRIDE` atomic, then ALTERNATED, so a single `rch exec` invocation
//! measures both on the same worker (rch picks workers non-deterministically and the ORIG/CAND
//! ratio is not worker-invariant — an A/B split across two invocations would be invalid).
//!
//! LEDGER-INTEGRITY note: this probe doubles as an EXECUTION PROOF for the blocked path at mid-n.
//! Until 176bccc67 (2026-07-08) the blocked factor was gated at `n >= FLAT_LU_SOLVE_MIN_DIM` =
//! 1000, so every earlier cholesky A/B in the ledger left `cholesky_lower_blocked` — and hence its
//! trailing SYRK — entirely UNEXECUTED for `256 <= n < 1000`. `differing_bits > 0` below proves the
//! two arms really are distinct code paths (blocked and unblocked differ in the last bits); a
//! `differing_bits == 0` at some n would mean the override silently failed and the "A/B" is timing
//! one arm twice — the exact dead-code trap this rule exists to catch.
use fsci_linalg::{CHOL_FACTOR_FLAT_MIN_OVERRIDE, DecompOptions, cholesky};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Gate-override sentinels: above any tested `n` ⇒ unblocked; below ⇒ blocked.
const FORCE_SIMD: usize = 4096;
const FORCE_BLOCKED: usize = 1;

fn mean_cv(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, if m > 0.0 { var.sqrt() / m * 100.0 } else { 0.0 })
}

fn main() {
    let mut seed = 7u64;
    let mut r = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let env_usize = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let reps = env_usize("FSCI_AB_REPS", 6);
    let iters = env_usize("FSCI_AB_ITERS", 7);

    println!("# same-binary A/B: unblocked left-looking SIMD vs blocked (parallel SYRK)");
    println!("# CHOL_FACTOR_FLAT_MIN_DIM=256 today; it was 1000 before 176bccc67, so the blocked");
    println!("# path was DEAD CODE for 256<=n<1000 in every cholesky A/B measured before then.");
    println!("# differing_bits>0 is the execution proof that both arms are distinct code paths.");
    for &n in &[128usize, 256, 384, 512, 768] {
        let m: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..n).map(|_| r() - 0.5).collect())
            .collect();
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += m[i][k] * m[j][k];
                }
                a[i][j] = s / n as f64 + if i == j { n as f64 } else { 0.0 };
            }
        }
        let factor = |ov: usize| {
            CHOL_FACTOR_FLAT_MIN_OVERRIDE.store(ov, Ordering::Relaxed);
            cholesky(&a, true, DecompOptions::default()).unwrap().factor
        };
        let cs = factor(FORCE_SIMD);
        let cb = factor(FORCE_BLOCKED);
        let maxdiff = cs
            .iter()
            .zip(&cb)
            .flat_map(|(pr, br)| pr.iter().zip(br))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        let bits = cs
            .iter()
            .zip(&cb)
            .flat_map(|(pr, br)| pr.iter().zip(br))
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();

        let bench = |ov: usize| {
            CHOL_FACTOR_FLAT_MIN_OVERRIDE.store(ov, Ordering::Relaxed);
            let _ = black_box(cholesky(&a, true, DecompOptions::default()).unwrap());
            let t = Instant::now();
            for _ in 0..reps {
                let _ = black_box(cholesky(black_box(&a), true, DecompOptions::default()).unwrap());
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1000.0
        };
        // Interleave so drift on a shared worker hits both arms equally.
        let (mut sv, mut bv) = (Vec::new(), Vec::new());
        for _ in 0..iters {
            sv.push(bench(FORCE_SIMD));
            bv.push(bench(FORCE_BLOCKED));
        }
        let (sm, scv) = mean_cv(&sv);
        let (bm, bcv) = mean_cv(&bv);
        let sb = sv.iter().copied().fold(f64::MAX, f64::min);
        let bb = bv.iter().copied().fold(f64::MAX, f64::min);
        println!(
            "chol n={n:4}: simd {sb:8.2}ms (mean {sm:.2} cv {scv:.1}%) blocked {bb:8.2}ms \
             (mean {bm:.2} cv {bcv:.1}%) best {:.2}x mean {:.2}x maxdiff={maxdiff:.1e} \
             differing_bits={bits}",
            sb / bb,
            sm / bm
        );
    }
    CHOL_FACTOR_FLAT_MIN_OVERRIDE.store(0, Ordering::Relaxed);
}
