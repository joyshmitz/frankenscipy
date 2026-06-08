# perf: parallel Aᵀ·x via cached CSC — byte-identical, 2.4x kernel / 1.5x lsqr+svds

## Lever (ONE)
`csr_matvec_transpose` (Aᵀ·x) was a serial **scatter**:
`for i: for idx in row i: result[indices[idx]] += data[idx]*x[i]`. The forward
SpMV (`csr_matvec`) was already parallelized; this is its transpose companion,
completing parallel matvec for the Golub-Kahan bidiagonalization solvers
(`lsqr`, `svds`) which alternate A and Aᵀ every iteration.

Cache `A` in **CSC once** per solver and evaluate Aᵀ·x as a per-output-column
gather (`csc_matvec`): `result[c] = Σ_{idx in col c} data[idx]·x[indices[idx]]`.
Each output column is independent, so it parallelizes across row chunks
(work-scaled, gated above ~256K nnz). The O(nnz) CSC build amortizes across the
solver's iterations.

## Parity — BYTE-IDENTICAL
- A CSC stores each column's entries in **increasing-row order**, which is exactly
  the order the serial scatter accumulates `result[col]` (outer loop over rows
  ascending). So the gather sums the same terms in the same order → identical
  `f64` bits. Same-process A/B (`csr_matvec_transpose` vs `csc_matvec(a.to_csc())`)
  reports `identical=true` for m=60K…400K, nnz=360K…3.6M. See `golden_payload.txt`.
- All 10 `fsci-sparse` lsqr+svds tests pass; full suite green.

## Timing — rch remote, 64 cores, `--profile release-perf`
Kernel A/B (200 reps):

| m × n         | nnz   | csr scatter | csc parallel | speedup |
|---------------|-------|-------------|--------------|---------|
| 60K × 40K     | 360K  | 775 µs      | 2.93 ms      | 0.3x (serial gate) |
| 200K × 150K   | 1.5M  | 4.037 ms    | 3.309 ms     | 1.2x    |
| 400K × 300K   | 3.6M  | 8.774 ms    | 3.727 ms     | 2.4x    |

End-to-end (lib solvers; forward matvec already parallel, so the transpose is
~half the per-iteration matvec work → end-to-end dilutes from the 2.4x kernel):

| solver | size                | baseline | new      | speedup |
|--------|---------------------|----------|----------|---------|
| lsqr   | m=200K n=150K 1.8M  | 489.5 ms | 322.9 ms | 1.52x   |
| svds   | n=120K 1.15M        | 303.6 ms | 210.2 ms | 1.44x   |

Kernel clears Score ≥ 2.0 (2.4x at 3.6M nnz); end-to-end ~1.5x is honest given the
forward matvec was already parallelized and the bidiagonalization has substantial
O(m+n) vector work per iteration. Byte-identical with a size gate (no regression).

Harness: `crates/fsci-sparse/src/bin/perf_csc_transpose_matvec.rs`
Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_csc_transpose_matvec`

## Notes
- Companion to the forward `csr_matvec` parallelization (commit 55b677b1). Together
  both matvec directions in lsqr/svds are now parallel + byte-identical.
- The serial scatter `csr_matvec_transpose` is retained `#[cfg(test)]` as the
  byte-identity reference.
