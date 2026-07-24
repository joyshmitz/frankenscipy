//! Cholesky-wall CYCLES self-time gate (frankenscipy-64wo0; prior art d97283534).
//!
//! WHY. A wall-clock A/B of the n=1000 Cholesky factor has a ~±5% floor
//! (subprocess launch + OS scheduling jitter on a shared 64-thread box) that
//! DROWNS the remaining sub-floor kernel / data-movement levers (pack fusion,
//! scratch reuse). `perf stat -e cycles` counts RETIRED work across all threads,
//! immune to scheduling, giving a ~4x tighter null — so a 2-5% real lever that is
//! undecidable on wall-clock becomes DECIDED on cycles.
//!
//! DESIGN. This binary runs ONE arm (argv[1]) `reps` times over a fixed SPD matrix
//! and prints a checksum to stdout. An external harness wraps each invocation in
//! `perf stat -e cycles -x, -- <bin> <arm> <n> <reps>`, alternates the arms so
//! drift on the shared box hits both equally, and takes the median cycle-count
//! ratio plus an A/A (base-vs-base) null. Setup is O(n^2) (<< the O(n^3) factor)
//! so it does not dilute the ratio; `reps` amortises process-launch cost.
//!
//! ARMS (default: the FMA-SYRK lever, used to VALIDATE the gate against its known
//! wall-clock verdict ~1.143x). Both arms share TRSM_ROWS2 + chol_nb_for(n); only
//! the trailing-SYRK kernel differs, so the ratio isolates that kernel:
//!   base = cholesky_wall_mr4_nr8_orig          (SYRK MR4xNR8, plain mul+add)
//!   cand = cholesky_wall_mr4_nr8_fma_candidate  (SYRK MR4xNR8 + fused mul_add)
//! The printed checksum lets the harness assert base != cand (execution proof:
//! identical checksums would mean the arm switch silently failed).

#[cfg(feature = "chol-wall-bench")]
mod gate {
    use fsci_linalg::{cholesky_wall_mr4_nr8_fma_candidate, cholesky_wall_mr4_nr8_orig};
    use std::hint::black_box;

    /// Cheap O(n^2) diagonally-dominant SPD matrix (deterministic).
    fn build_spd(n: usize) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let v = if i == j {
                    (n as f64) * 3.0 + (i as f64) * 0.01
                } else {
                    1.0 / ((i - j + 1) as f64)
                };
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        a
    }

    fn fnv1a_mix(digest: &mut u64, value: f64) {
        for byte in value.to_bits().to_le_bytes() {
            *digest ^= u64::from(byte);
            *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let arm = args.get(1).map(String::as_str).unwrap_or("base");
        let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
        let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
        // digest mode: "full" folds every element (execution proof — FMA-SYRK
        // differs from mul+add in only ~46/1e6 elements, so a full digest is
        // needed to detect the arm switch); "light" (default, for the MEASURED
        // runs) folds a strided sample only, enough to defeat DCE without adding
        // the O(n^2) common work that would dilute the cycle ratio.
        let full_digest = args.get(4).map(String::as_str) == Some("full");
        let a = build_spd(n);

        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..reps {
            let factor = match arm {
                "cand" => cholesky_wall_mr4_nr8_fma_candidate(black_box(&a)),
                _ => cholesky_wall_mr4_nr8_orig(black_box(&a)),
            }
            .expect("cholesky factor");
            if full_digest {
                for &value in &factor {
                    fnv1a_mix(&mut digest, value);
                }
            } else {
                // Strided sample: cheap, only prevents DCE. The DECISION comes
                // from the cycle ratio (a dead arm switch reads as the null), not
                // from this digest.
                let mut idx = 0;
                while idx < factor.len() {
                    fnv1a_mix(&mut digest, factor[idx]);
                    idx += 64;
                }
            }
            black_box(&factor);
        }
        // stdout: checksum only (perf writes its CSV to stderr, so they don't mix).
        println!("arm={arm} n={n} reps={reps} digest={digest:#018x}");
    }
}

#[cfg(feature = "chol-wall-bench")]
fn main() {
    gate::run();
}

#[cfg(not(feature = "chol-wall-bench"))]
fn main() {
    eprintln!("perf_chol_cycles_gate requires --features chol-wall-bench");
    std::process::exit(2);
}
