# Sparse SpMM Epoch SPA Trial

- Bead: `frankenscipy-vyzkc`
- Target: `sparse_spmm/2000x2000_d1/2000`
- Profile source: fresh `fsci-sparse` RCH reprofile ranked this row first at
  13.087 ms median on `ts2`.
- Lever: replace per-row `seen` cleanup in the Gustavson sparse accumulator
  with an epoch-stamped mark array.
- Baseline: RCH `cargo bench -p fsci-sparse --bench sparse_bench --locked --
  sparse_spmm/2000x2000_d1/2000 --warm-up-time 1 --measurement-time 5
  --sample-size 30 --noplot` on `vmi1149989`: 6.8188 ms median
  `[6.7042, 6.9547]`.
- After: same command and worker: 9.5096 ms median `[8.7255, 10.705]`.
- Verdict: rejected. The same-worker result is a clear regression, so Score is
  0.0 and the code was backed out.

## Behavior Proof

- Strict golden SHA before:
  `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- Strict golden SHA after:
  `0728e7d2e4072bf721c19f6b8d0a85a1e064bad0a69d4f10382efbc8ab4c5af2`
- Strict payload diff: empty.
- Isomorphism checked: row traversal, A/B encounter order, first-seen column
  order, reverse emission order, floating-point accumulation order, explicit
  zero elision, sorted-index flag updates, metadata, and RNG absence were
  preserved by the trial.

## Next Primitive

Do not repeat the epoch-stamp or capacity-family SpMM levers. The next sparse
attack should be a structurally different GraphBLAS-style SpGEMM primitive,
such as a symbolic row-nnz prepass with exact output allocation or a sorted-row
merge/heap accumulator for rows where both input rows are canonical sorted.
