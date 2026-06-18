# Performance Negative-Evidence Ledger

This ledger records every code-first performance attempt, including attempts that
are still awaiting the batch benchmark wave. Entries must name the retry
condition so dead ends are not repeated casually.

## 2026-06-18 - frankenscipy-fo9cj - sparse Arnoldi row-major basis arena

- Agent: cod-b / MistyBirch
- Lever: replace the `krylov_arnoldi_eigs` `Vec<Vec<f64>>` basis and allocating
  operator return with a row-major basis arena plus a reusable operator scratch
  buffer; switch `eigsh`, `eigs`, and `svds` callers to `csr_matvec_into` /
  `csc_matvec_into`.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `cargo check -p fsci-sparse` is expected before commit.
- Correctness guard: `csc_matvec_into_matches_allocating_reference` plus existing
  `eigsh`, `eigs`, and `svds` conformance/unit coverage in the sparse crate.
- Benchmark guard: run `cargo run --profile release-perf -p fsci-sparse --bin
  perf_eigsh` and `cargo run --profile release-perf -p fsci-sparse --bin
  perf_svds` against the pre-change commit on the same worker/target dir.
- Retry condition: keep only if same-worker focused sparse eigensolver timings
  show a stable win outside noise without eigs/eigsh/svds residual drift; if the
  arena copy cost erases the allocation savings or regresses any row, reject this
  exact arena/scratch formulation and do not retry without allocator/profile
  evidence showing per-step basis allocation is again a top-5 sparse hotspot.

## 2026-06-18 - frankenscipy-bpzha - RK step scratch double-buffer

- Agent: cod-b / MistyBirch
- Lever: move `rk_step` rejected-attempt storage into solver-owned reusable
  buffers for `dy`, `y_stage`, `y_new`, and `f_new`; accepted steps swap the
  buffers into live state, while rejected attempts overwrite the same scratch on
  retry.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  cargo check -p fsci-integrate` passed before commit.
- Correctness guard: existing RK45/RK23 step unit coverage now exercises the
  scratch-buffer API, including wrong-size RHS rejection; solver accept/reject
  semantics still preserve `y_old`, `f_old`, boundary clamping, and FSAL storage.
- Benchmark guard: compare focused `solve_ivp` RK45/RK23 workloads against the
  pre-change commit on the same worker/target dir, especially high-dimensional
  adaptive problems with rejected steps where per-attempt vector allocation was
  visible in profiles.
- Retry condition: keep only if focused same-worker integrate timings improve
  without changing step counts, `nfev`, or final tolerances; if the swap/copy
  path costs more than the allocation removal, reject this scratch formulation
  and do not retry without allocator-profile evidence showing RK temporary Vec
  churn is again a top-5 integrate hotspot.

## 2026-06-18 - frankenscipy-va60h - linkage row-major distance arena

- Agent: cod-a / MistyBirch
- Lever: replace the nearest-neighbour linkage core's `(2n-1) x (2n-1)`
  `Vec<Vec<f64>>` inter-cluster distance matrix with one row-major `Vec<f64>`
  arena and stride indexing; fill that arena directly from observations or
  condensed precomputed distances for the non-Centroid/Median methods.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-cluster` is expected before commit.
- Correctness guard: `linkage_flat_core_matches_precomputed_condensed_contract`
  plus existing SciPy reference and inactive-cluster regression coverage for
  `linkage` / `linkage_from_distances`.
- Benchmark guard: compare `cargo run --profile release-perf -p fsci-cluster
  --bin perf_linkage` against the pre-change commit on the same worker/target
  dir, with emphasis on Ward/Average linkage at n>=800 where the row-major arena
  should reduce row-allocation and pointer-chasing overhead.
- Retry condition: keep only if same-worker focused linkage timings show a
  stable win outside noise with byte-identical linkage rows; if flat indexing
  loses to the old nested rows due multiply/index overhead or cache effects,
  reject this exact full-square arena formulation and do not retry without a
  profile showing `agglomerate_nnarray` row traversal or allocation is again a
  top-5 cluster hotspot.
