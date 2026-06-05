# fsci-linalg full-rank rectangular pinv materialization trials

Bead: `frankenscipy-vgs9h`

Target row: `pinv/512x256`.

Baseline:
- `tests/artifacts/perf/2026-06-05-linalg-full-rank-svd-vgs9h/baseline_pinv_512x256_criterion_rch.txt`
- Worker: `ts1`
- Time: `[85.492 ms 86.017 ms 86.559 ms]`

Trial 1: scalar direct singular-vector contraction.
- Behavior proof: `cargo test -p fsci-linalg --lib pseudo_inverse_direct_contraction_matches_dense_diagonal_reference --locked -- --nocapture` passed on `ts1`.
- Public proof: `cargo test -p fsci-linalg --lib pinv --locked -- --nocapture` passed `13/13` on `ts1`.
- After benchmark: `tests/artifacts/perf/2026-06-05-linalg-full-rank-svd-vgs9h/baseline_current_pinv_512x256_criterion_rch.txt`
- Worker: `ts2`
- Time: `[453.51 ms 455.60 ms 457.56 ms]`
- Verdict: reject. The scalar loop is much slower than the available baseline and does not clear the keep gate.

Trial 2: scale `V` columns and keep one optimized GEMM.
- Behavior proof: `cargo test -p fsci-linalg --lib pseudo_inverse_direct_contraction_matches_dense_diagonal_reference --locked -- --nocapture` passed on `ts2`.
- Public proof: `cargo test -p fsci-linalg --lib pinv --locked -- --nocapture` passed `14/14` on `ts2`.
- Golden payload SHA-256: `55334858afb28fff30f170aff4b994f3edbe07123c9953a9a1e10677e1e8ffba`
- After benchmark: `tests/artifacts/perf/2026-06-05-linalg-full-rank-svd-vgs9h/after_scaled_gemm_pinv_512x256_criterion_rch.txt`
- Worker: `ts2`
- Time: `[133.88 ms 134.54 ms 135.31 ms]`
- Verdict: reject. Behavior is preserved, but there is no same-worker keep evidence and the after row is slower than the available `ts1` baseline.

No production materialization change is kept. The next valid target is the deeper
blocked Householder bidiagonalization / bidiagonal SVD primitive for the public
full-rank rectangular SVD-family contract.
