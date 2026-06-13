# Vectorize the U-block triangular solve in both LU factorizations

## Lever
The blocked LU factorization has three phases: (1) panel factorization, (2) U12 solve
`U12 = L11⁻¹·A12`, (3) trailing GEMM. Phase (2) was 100% SCALAR in BOTH `lu_factor_blocked`
(f64) and `lu_factor_blocked_f32`. Phase profiling (n=1000) put panel+usolve at ~13.6 ms of
the ~46 ms mixed solve. This vectorizes the U12 solve 8-wide over the trailing columns:

    let mut acc = Simd::<_, 8>::from_slice(&data[i_base+jj .. +8]);   // s = data[i][jj]
    for p in k..i { acc -= Simd::splat(data[i_base+p]) * Simd::from_slice(&data[p*n+jj..+8]); }
    acc.copy_to_slice(&mut data[i_base+jj .. +8]);

Element type is inferred from `data` (f64 / f32), so the single edit covers both kernels.

## Isomorphism (bit-identical)
Each output element keeps the IDENTICAL incremental `s -= L[i][p]·U[p][jj]` over monotonic p,
so the result is bit-identical to the scalar form. Proof: `cargo test -p fsci-linalg --release
--lib -- --include-ignored` = **430 passed, 0 failed**; `flat_lu_golden_digest` still
**0x2fc8ed294ef0427c** (f64 factors unchanged). clippy + fmt clean.

## Benchmark (same-worker A/B, vmi1227854, one binary via DISABLE_MIXED_LU)
| arm                              | time (median)            |
|----------------------------------|--------------------------|
| `baseline_solve/1000x1000`     (mixed) | **31.609 ms** [30.990 32.225] |
| `baseline_solve/1000x1000_f64` (f64)   | **54.072 ms** [53.244 54.939] |
=> **1.71x** mixed vs f64 — best numbers of the campaign.

This phase being shared, the win helps the f64 path too (solve / inv / det at n >= 1000),
not only the mixed-precision route.

## Confirmation run (vmi1227854, fresh)
| arm   | time (median)            |
|-------|--------------------------|
| mixed | 36.304 ms [35.786 36.817] |
| f64   | 55.997 ms [55.122 56.877] |
=> 1.54x. Two runs: 1.54x–1.71x (vmi1227854 load varies); mixed ~32–36 ms, f64 ~54–56 ms.
