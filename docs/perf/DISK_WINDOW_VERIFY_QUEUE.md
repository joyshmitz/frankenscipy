# Disk-window verify queue (cc / MistyBirch) — RUN WHEN DISK RECOVERS

During the 2026-06-20 DISK-CRITICAL window (cargo bench/build/check ALL paused), the
commits below were made CODE-ONLY and are **UNVERIFIED-COMPILE**. They are all
byte-identical-by-construction (no new bench needed for correctness — see reasoning
per group), but MUST be compile-checked + run against the existing suites before being
trusted. Do this FIRST on recovery.

## One-shot verify
```
cargo check -p fsci-interpolate -p fsci-stats -p fsci-signal -p fsci-special -p fsci-cluster
cargo test  -p fsci-interpolate   # expect 172/0 (make_smoothing_spline + make_interp_spline scipy-parity included)
cargo test  -p fsci-stats         # full suite (5 pre-existing zscore/mad/sklearn fails are UNRELATED, verified prior)
cargo test  -p fsci-signal -p fsci-special -p fsci-cluster
```

## Commits (newest→oldest), with byte-identity basis
- e3e1c396  stats/cluster: Vec<usize> sort+dedup → sort_unstable — Vec<usize> equal⟹identical
- 79773b3f  signal/special/cluster: 9 Vec<f64> sorts → sort_unstable — total_cmp Equal⟺identical bits
- 2893660c  stats: 60 Vec<f64> sorts → sort_unstable — total_cmp Equal⟺identical bits
- d437f83f  interpolate: band_to_full fill O(n²)→O(n) — out-of-band entries 0 either way
- eab4eea3  interpolate: gcv numer band-restrict O(n²)→O(n) — skipped terms are +0.0 no-ops
- 43eb09b2  interpolate: gcv m/lhs builds band-restrict O(n²)/O(n³)→O(n)/O(n²) — out-of-band 0; solve_banded makes LU fill
- 033f7bd9  interpolate: gcv Gram build band-restrict O(n³)→O(n) — +0.0 no-op skip, ascending-k order
- 08f79b0e  interpolate: make_smoothing_spline + GCV dense→banded solves (bw 2/2/4) — solve_banded == solve_dense_system within band

(Already verified/landed before/within window, NOT pending: 67798376 compact
make_interp_spline [another agent, has its own byte-diff test], 318898bb, 17e29927,
389353dd, 0e4f77e0 — these were built+tested when shipped.)

## After verify
- If all green: delete this file; the smoothing-spline GCV is O(n)+O(n²·iters) and the
  72-site sort_unstable sweep is live. Re-run benches to quantify (make_smoothing_spline
  large-n; the sort-heavy stats fns).
- Then resume measured loss-hunting; the one OPEN loss is hilbert (FFT C-SIMD wall).
- Remaining (cargo-needed) follow-ups: factor-once + n-RHS for the GCV trace loop
  (O(n²·bw) tr); hilbert FFT (FFT-crate, hard).
