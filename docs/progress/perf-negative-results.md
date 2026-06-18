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

## 2026-06-18 - frankenscipy-6m75u - Wolfe trial-point scratch reuse

- Agent: cod-b / MistyBirch
- Lever: replace per-probe `x + alpha*d` trial `Vec` construction in public
  Wolfe line search with one reusable trial buffer, and thread that buffer
  through the bisection zoom phase.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  cargo check -p fsci-opt` passed before commit.
- Correctness guard: existing Wolfe1/Wolfe2 tests cover alpha selection,
  Armijo/curvature conditions, public-vs-probed equivalence, mismatched
  dimensions, and gradient length rejection; the change only alters trial-point
  storage.
- Benchmark guard: compare high-dimensional BFGS/CG optimizer workloads with
  repeated line-search probes against the pre-change commit on the same worker,
  and include a focused line-search microbench if the batch wave has one.
- Retry condition: keep only if same-worker optimizer or line-search timings
  improve without alpha, evaluation-count, or accepted-gradient drift; if the
  helper call/fill path is neutral or slower, reject this trial-buffer lever and
  do not retry public Wolfe scratch reuse without allocation-profile evidence
  showing trial-point construction is again a top-5 opt hotspot.

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

## 2026-06-18 - frankenscipy-8l8r1.118 - coherence chunk-local spectra accumulator

- Agent: cod-b / MistyBirch
- Lever: keep `coherence` on the fused Welch segment pass, but replace the
  per-segment `CoherenceSegmentSpectra` materialization with chunk-local
  Pxy/Pxx/Pyy accumulators and reusable windowed `wx`/`wy` scratch buffers. This
  removes one spectra allocation triplet per segment and avoids retaining all
  segment spectra before the final fold on the target
  `spectral/coherence/65536_w1024_o512` workload.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  cargo check -p fsci-signal` is the only required pre-commit gate.
- Correctness guard: existing `coherence_matches_compositional_csd_formula`
  compares the fused path against `csd(x,y)`, `csd(x,x)`, and `csd(y,y)`;
  existing SciPy-reference coherence coverage anchors the public tolerance
  contract.
- Benchmark guard: compare focused Criterion
  `spectral/coherence/65536_w1024_o512` and any batch Welch/coherence rows
  against the pre-change commit on the same worker/target dir.
- Retry condition: keep only if same-worker coherence timings improve without
  frequency-grid, range-clamp, or compositional-formula drift; if chunk-local
  reduction grouping or scratch initialization costs erase the allocation win,
  reject this accumulator formulation and do not retry it without allocation
  profiles showing retained segment spectra are again a top-5 signal hotspot.

## 2026-06-18 - frankenscipy-8l8r1.119 - BDF Newton streamed scaled RMS norm

- Agent: cod-a / MistyBirch
- Lever: replace `newton_bdf`'s per-Newton-iteration temporary
  `collect::<Vec<_>>()` for `dy[j] / scale[j]` with an allocation-free streamed
  scaled RMS helper over the LU solve vector and scale slice.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-integrate` is expected before commit.
- Correctness guard: `rms_norm_scaled_matches_collected_reference` proves the
  streamed helper is bit-identical to the old collect-then-`rms_norm` path for
  the same scaled values, while existing BDF tests cover solver convergence and
  validation semantics.
- Benchmark guard: compare focused stiff `solve_ivp(method=BDF)` workloads
  against the pre-change commit on the same worker/target dir, especially
  medium/high-dimensional states where Newton iterations dominate allocation
  churn.
- Retry condition: keep only if same-worker focused BDF timings improve without
  step-count, final-state, tolerance, or failure-mode drift; if the streamed
  helper is neutral/slower, reject this exact norm helper and do not retry
  unless allocation profiles show BDF Newton scaled-vector churn is again a
  top-5 integrate hotspot.

## 2026-06-18 - frankenscipy-8l8r1.120 - Radau streamed scaled RMS norms

- Agent: cod-a / MistyBirch
- Lever: replace Radau's per-step scaled-vector materialization for Newton
  correction norms and embedded error norms with streamed scaled RMS
  accumulation over the LU solve output and scale slices.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-integrate` is expected before commit.
- Correctness guard: `streamed_scaled_norms_match_collected_reference` proves
  the streamed error norm and stage-major Newton correction norm match the old
  collect-then-`rms_norm` path bit-for-bit for the same scaled values; existing
  Radau stiff-solver tests cover convergence and validation semantics.
- Benchmark guard: compare focused stiff `solve_ivp(method=Radau)` workloads
  against the pre-change commit on the same worker/target dir, especially
  medium/high-dimensional states and rejected-step stabilised error estimates.
- Retry condition: keep only if same-worker focused Radau timings improve
  without `nfev`, `njev`, `nlu`, accepted-step, final-state, tolerance, or
  rejection-pattern drift; if streamed norms are neutral/slower, reject this
  exact formulation and do not retry unless allocation profiles show Radau
  scaled-norm Vec churn is again a top-5 integrate hotspot.
