# Performance Negative-Evidence Ledger

This ledger records every code-first performance attempt, including attempts that
are still awaiting the batch benchmark wave. Entries must name the retry
condition so dead ends are not repeated casually.

## 2026-06-19 - Gauntlet verification - fsci-opt least_squares scratch cluster

- Agent: cod-b / MistyBirch
- Beads verified: `frankenscipy-szky7`, `frankenscipy-y1mzk`
- Decision: KEEP. No revert. All measured head-to-head rows are wins vs the
  original SciPy LM path on warmed single-process realistic Python callback
  workloads.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-least-squares-gauntlet/least_squares_vs_scipy.json`
- Rust bench command: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo bench -p fsci-opt --bench optimize_bench -- least_squares --noplot`
- SciPy oracle: Python 3, SciPy 1.17.1, NumPy 2.4.3, `scipy.optimize.least_squares(method="lm")`.

| Workload | Rust Criterion p50 (us) | SciPy p50 (us) | SciPy p95 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `least_squares/rosenbrock_residual` | 2.558 | 1404.547 | 2297.906 | 549.08x | win |
| `least_squares/exp_curve_64` | 16.932 | 753.120 | 1020.621 | 44.48x | win |
| `least_squares/exp_linear_curve_128` | 49.724 | 893.946 | 1015.962 | 17.98x | win |

- Correctness/conformance guards:
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo check -p fsci-opt`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo test -p fsci-opt --test metamorphic_tests mr_least_squares -- --nocapture` (2 passed)
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo run -p fsci-opt --release --bin diff_lsq`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo run -p fsci-opt --release --bin diff_leastsq`
- Blocker recorded separately: broad `cargo test -p fsci-opt least_squares`
  still fails before reaching least-squares assertions because unrelated
  `crates/fsci-opt/src/lib.rs` test modules miss imports for numerical helper
  functions and `OptError`; filed follow-up `frankenscipy-uxs8k`.
- Negative evidence: none for this cluster. Do not retry or revert the
  fixed-shape Jacobian scratch reuse or LM normal-equation scratch reuse on the
  basis of this gauntlet; route future opt perf work to a lower-level hotspot or
  a workload that shows a measured neutral/loss row.

## 2026-06-19 - frankenscipy-u0ucw - GAUNTLET measured wide pinv vs SciPy

- Agent: cod-a / MistyBirch
- Workload: full-row-rank wide pseudo-inverse, 500x1000 dense
  Cauchy-like matrix with a strong diagonal, matching the committed
  `make_underdetermined` Criterion generator and SciPy oracle construction.
- Subject commits: `6c139073` wide `pinv` Cholesky TRSM plus `c39e9394`
  diagonal rcond gate, measured at `41bf34a4`.
- Criterion command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo bench -p fsci-linalg --bench linalg_bench -- u0ucw_gauntlet_scipy_pinv --noplot`
- Environment: local same-host oracle because rch worker `vmi1152480` lacked
  `scipy` for the Python oracle row; local `python3` 3.13.7, NumPy 2.4.3,
  SciPy 1.17.1, rustc 1.98.0-nightly.
- Results use Criterion mean point estimates from
  `target/criterion/u0ucw_gauntlet_scipy_pinv/*/new/estimates.json`:

  | Route | Mean | Ratio vs SciPy | Internal delta | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | SciPy `scipy.linalg.pinv(check_finite=False)` | 7.257573 s | 1.00x | oracle | reference |
  | Rust current Cholesky + diagonal rcond gate | 183.699926 ms | 39.51x faster | 1.40x vs eigengate, 2.82x vs SVD fallback | keep |
  | Rust Cholesky + old eigenspectrum rcond gate | 257.672693 ms | 28.17x faster | current is 28.71% less time | superseded |
  | Rust SVD fallback route | 517.524097 ms | 14.02x faster | current is 64.50% less time | superseded |

- rch evidence: `rch exec -- cargo bench -p fsci-linalg --bench linalg_bench
  -- u0ucw_gauntlet_scipy_pinv --noplot` built and timed the Rust rows on
  worker `vmi1152480` (`current` 171.97 ms, `eigengate` 268.75 ms,
  `svd_fallback` 416.57 ms), then the SciPy oracle row failed with
  `ModuleNotFoundError: No module named 'scipy'`. That worker result is
  retained only as infrastructure evidence and is not used for the
  head-to-head ratio.
- Correctness guard: no revert. Keep the Cholesky route and diagonal rcond gate
  because the measured route is materially faster than the old in-crate routes
  and 39.51x faster than SciPy on this realistic wide full-row-rank workload.
- Retry condition: do not retry the old eigenspectrum rcond gate or SVD
  fallback for this 500x1000 full-row-rank wide workload unless a future
  conformance failure, condition-audit regression, or same-host SciPy/Criterion
  rerun reverses the current >1.4x internal win.

## 2026-06-19 - frankenscipy-y1mzk - least_squares LM normal-equation scratch

- Agent: cod-b / MistyBirch
- Lever: keep the `least_squares` Levenberg-Marquardt algorithm on the same
  residual/Jacobian path while reusing fixed-shape scratch for `J^T J`,
  `J^T r`, the damped normal-equation matrix, Cholesky factor, forward/backward
  solve vectors, and `J*step` predicted-reduction work.
- Status: measured KEEP. 2026-06-19 gauntlet found wins vs SciPy LM on all
  selected least-squares rows: 549.08x, 44.48x, and 17.98x p50 SciPy/Rust.
- Correctness guard: existing `least_squares`, `curve_fit`, and `leastsq`
  coverage exercises convergence and covariance paths; the helper preserves
  residual evaluation order, finite-difference values, LM damping updates,
  fallback diagonal solve semantics, and `nfev`/`njev`/`nit` accounting.
- Benchmark guard: Criterion `least_squares/rosenbrock_residual`,
  `least_squares/exp_curve_64`, and `least_squares/exp_linear_curve_128`
  rows quantify the scratch-reuse path against SciPy LM in the gauntlet
  artifact above.
- Retry condition: keep only if same-worker focused `fsci-opt` least-squares /
  curve-fit timings improve without convergence, cost, parameter, Jacobian, or
  counter drift; if timings are neutral/slower, reject this exact LM
  normal-equation scratch formulation and do not retry unless allocation
  profiles put damped normal-equation matrix/Cholesky/`J*step` scratch back in
  the top-5 `fsci-opt` hotspots.

## 2026-06-19 - frankenscipy-szky7 - least_squares fixed-shape Jacobian scratch

- Agent: cod-b / MistyBirch
- Lever: keep `fsci_opt::curvefit::least_squares` on the same
  Levenberg-Marquardt control path while reusing the finite-difference
  Jacobian rows and perturbation vector across accepted steps instead of
  allocating a fresh `m x n` `Vec<Vec<f64>>` plus `x_perturbed` for every
  Jacobian rebuild.
- Status: measured KEEP. 2026-06-19 gauntlet found wins vs SciPy LM on all
  selected least-squares rows: 549.08x, 44.48x, and 17.98x p50 SciPy/Rust.
- Correctness guard: existing `least_squares`, `curve_fit`, and `leastsq`
  coverage plus the `diff_lsq` probe exercise the residual evaluation contract.
  The helper preserves forward-difference order, residual call count,
  `nfev`/`njev` accounting, accepted/rejected damping updates, and final
  Jacobian row/column shape.
- Benchmark guard: Criterion `least_squares/rosenbrock_residual`,
  `least_squares/exp_curve_64`, and `least_squares/exp_linear_curve_128`
  rows quantify the allocation-removal path against SciPy LM in the gauntlet
  artifact above.
- Retry condition: keep only if same-worker focused `fsci-opt` least-squares /
  curve-fit timings improve without convergence, `nfev`/`njev`, cost, or
  parameter drift; if timings are neutral or slower, reject this exact
  fixed-shape Jacobian scratch route and do not retry unless allocation
  profiles put curvefit Jacobian matrix allocation back in the top-5
  `fsci-opt` hotspots.

## 2026-06-19 - frankenscipy-8l8r1.122 - L-BFGS-B Wolfe probe scratch reuse

- Agent: cod-b / MistyBirch
- Lever: route unconstrained `L-BFGS-B` Strong-Wolfe finite-difference
  gradient probes through `line_search_wolfe2_with_gradient_probe`, reusing the
  line-search trial buffer and gradient `Vec` instead of allocating a fresh
  `g` and `xp` inside every curvature probe.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  cargo check -p fsci-opt` is the pre-commit gate.
- Correctness guard: existing unconstrained/bounded `L-BFGS-B` tests and the
  hard Rosenbrock Strong-Wolfe regression cover accepted-step behavior; the
  patch intentionally keeps the accepted-step `finite_diff_gradient` recompute
  so `OptimizeResult` counters and final gradient shape stay on the previous
  public path.
- Benchmark guard: compare focused `fsci-opt` `L-BFGS-B` Rosenbrock/quadratic
  rows and the optimizer batch wave against the pre-change commit on the same
  worker/target dir, with attention to high-dimensional finite-difference
  line-search probes.
- Retry condition: keep only if same-worker `L-BFGS-B` optimizer timings
  improve without alpha, `nfev`/`njev`, accepted-gradient, or convergence drift;
  if the mutable-probe path is neutral or slower, reject this exact
  finite-difference probe scratch route and do not retry unless allocation
  profiles put `L-BFGS-B` Wolfe gradient-probe Vec churn back in the top-5
  `fsci-opt` hotspots.

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

## 2026-06-18 - frankenscipy-8l8r1.121 - BDF streamed step/order error norms

- Agent: cod-a / MistyBirch
- Lever: reuse the BDF streamed scaled RMS helper for accepted-step error norms
  and order-minus/order-plus selection norms, removing three temporary
  coefficient-scaled error vectors that existed only to call `rms_norm`.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-integrate` is expected before commit.
- Correctness guard: `rms_norm_scaled_matches_collected_reference` now covers
  both direct scaled vectors and coefficient-scaled error vectors with
  bit-identical results to the old collect-then-`rms_norm` path; existing BDF
  tests cover convergence, validation, and order-adaptation behavior.
- Benchmark guard: compare focused stiff `solve_ivp(method=BDF)` workloads
  against the pre-change commit on the same worker/target dir, emphasizing
  variable-order adaptation at medium/high dimension.
- Retry condition: keep only if same-worker focused BDF timings improve without
  order-choice, step-count, final-state, tolerance, or rejection drift; if this
  streaming order-error formulation is neutral/slower, reject it and do not
  retry unless allocation profiles show BDF error-norm Vec churn remains a top
  integrate hotspot.

## 2026-06-18 - frankenscipy-8l8r1.118 - CSD chunk-local cross-spectrum accumulator

- Agent: cod-b / MistyBirch
- Lever: apply the coherence accumulator pattern to `csd_with_scaling` itself:
  each worker chunk now reuses `wx`/`wy` segment scratch and folds
  cross-periodograms directly into one Pxy accumulator instead of allocating
  and retaining one `Vec<(re, im)>` per segment before the final average.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b
  cargo check -p fsci-signal` is the only required pre-commit gate.
- Correctness guard: existing `csd_auto_spectrum_matches_welch`,
  `csd_scaling_matches_scipy_density_and_spectrum`, and
  `coherence_matches_compositional_csd_formula` cover the scaled CSD contract
  and downstream coherence use.
- Benchmark guard: compare focused Welch/CSD rows, especially
  `spectral/csd/65536_w1024_o512` if present, plus the coherence target that
  consumes `csd` compositionally in the guard path.
- Retry condition: keep only if same-worker CSD/Welch timings improve without
  SciPy density/spectrum drift; if chunk-local reduction grouping or scratch
  initialization costs erase the allocation savings, reject this CSD accumulator
  formulation and do not retry unless retained segment spectra reappear as a
  top-5 signal allocation hotspot.

## 2026-06-18 - frankenscipy-u0ucw - Tall normal-equation Gram/vector streaming

- Agent: cod-a / MistyBirch
- Lever: compute the full-rank tall `pinv`/`lstsq` normal-equation products
  directly from DMatrix column slices: `A^T A` is formed symmetrically without
  materializing `A^T`, and `lstsq` computes `A^T b` / `A^T r` with the same
  contiguous column-dot helper.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-linalg` is expected before commit.
- Correctness guard: `normal_equation_column_helpers_match_nalgebra_products`
  compares the new Gram/vector helpers against the previous nalgebra transpose
  products, while existing full-rank tall `pinv`/`lstsq` route tests preserve
  the SVD-reference tolerance contract.
- Benchmark guard: compare focused tall `pinv`/`lstsq` normal-equation
  workloads, especially 1000x500 and 3000x1500 shapes from `u0ucw`, against the
  previous same-worker commit.
- Retry condition: keep only if same-worker tall `pinv`/`lstsq` timings improve
  without fast-route acceptance, rank, singular-value, residual, or
  pseudo-inverse tolerance drift; if the hand-rolled symmetric Gram loses to
  nalgebra's generic transpose product, reject this exact column-dot formulation
  and do not retry unless profiling puts `A^T` materialization or Gram formation
  back in the top linalg hotspot list.

## 2026-06-19 - frankenscipy-u0ucw - Wide lstsq row-streamed normal equations

- Agent: cod-a / MistyBirch
- Lever: route full-row-rank wide `lstsq` normal-equation products through the
  caller's row-major input: form `A A^T` from contiguous row dot products and
  compute `A^T y` / `A^T dy` by streaming rows once, avoiding the old
  materialized `A^T` matrix. `FSCI_DISABLE_WIDE_LSTSQ_ROW_STREAMING` and
  `DISABLE_WIDE_LSTSQ_ROW_STREAMING` keep the old path available for same-binary
  A/B benchmarks.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-linalg` is expected before commit.
- Correctness guard: `wide_normal_equation_row_helpers_match_nalgebra_products`
  compares row-streamed `A A^T` and `A^T rhs` against the previous materialized
  transpose products; the existing wide `lstsq` Cholesky/refinement acceptance
  gate still falls back to SVD on conditioning or refinement uncertainty.
- Benchmark guard: same-binary Criterion group `u0ucw_wide_lstsq` compares
  `_row_streaming` against `_materialized_transpose` on 500x1000 and 1000x2000
  full-row-rank wide workloads.
- Retry condition: keep only if same-worker `u0ucw_wide_lstsq` timings improve
  without rank, singular-value, min-norm, residual, or public `lstsq` tolerance
  drift; if row streaming is neutral/slower due to nalgebra's transpose/multiply
  kernels winning on wide shapes, reject this exact row-streamed wide formulation
  and do not retry unless allocation or cache profiles put wide-route `A^T`
  materialization back in the top linalg hotspot list.

## 2026-06-19 - frankenscipy-u0ucw - Wide pinv normal-equation Cholesky TRSM

- Agent: cod-a / MistyBirch
- Lever: add a full-row-rank wide `pinv` route using
  `A^+ = A^T (A A^T)^-1`: form the small row Gram from row-major input, solve
  the identity RHS through a new 8-wide batched Cholesky TRSM helper, stream the
  final `A^T G^-1` multiply without materializing `A^T`, and certify
  `A A^+ ~= I_rows` before accepting.
- Status: MEASURED KEEP in the 2026-06-19 gauntlet section above. The
  same-host SciPy oracle row measured the current wide route at 39.51x faster
  than SciPy and 2.82x faster than the in-crate SVD fallback on 500x1000
  full-row-rank input.
- Correctness guard: `wide_pinv_cholesky_matches_svd_reference` locks the new
  route against the SVD pseudo-inverse tolerance contract, while
  `wide_pinv_helpers_match_materialized_products` checks identity-RHS Cholesky
  solve, streamed `A^T G^-1`, and streaming left-inverse certificate against
  materialized nalgebra products.
- Benchmark guard: same-binary Criterion group `u0ucw_wide_pinv` compares
  `_normal_equation_cholesky` against `_svd_fallback` on 500x1000 and
  1000x2000 full-row-rank wide workloads.
- Retry condition: keep only if same-worker `u0ucw_wide_pinv` timings improve
  without rank, pseudo-inverse tolerance, Moore-Penrose left-inverse, or public
  `pinv` certificate drift; if the normal-equation Cholesky route is
  neutral/slower because the left-inverse certificate or `A^T G^-1` multiply
  dominates, reject this exact wide-`pinv` formulation and do not retry unless
  profiles put wide full-row-rank SVD work back in the top linalg hotspot list.

## 2026-06-19 - frankenscipy-u0ucw - Wide pinv diagonal rcond gate

- Agent: cod-a / MistyBirch
- Lever: remove the pre-Cholesky symmetric eigensolve from the full-row-rank
  wide `pinv` route. `pinv` does not expose singular values, so the fast path
  now uses the same diagonal rcond sanity estimate as the tall Cholesky route,
  then relies on Cholesky plus the existing streaming `A A^+ ~= I_rows`
  certificate as the fail-closed acceptance test. The old eigenspectrum gate
  stays behind `FSCI_DISABLE_WIDE_PINV_DIAG_RCOND_GATE` and
  `DISABLE_WIDE_PINV_DIAG_RCOND_GATE` for same-binary A/B.
- Status: MEASURED KEEP in the 2026-06-19 gauntlet section above. The
  same-host Criterion run measured the diagonal rcond gate at 1.40x faster than
  the old eigenspectrum rcond gate and 39.51x faster than SciPy on 500x1000
  full-row-rank input.
- Correctness guard:
  `wide_pinv_diag_rcond_gate_matches_eigen_gate_and_rejects_rank_loss` compares
  the diagonal-gated and eigengated pseudo-inverse outputs on a full-row-rank
  wide matrix and verifies an exact rank-deficient wide matrix still rejects.
- Benchmark guard: same-binary Criterion group `u0ucw_wide_pinv` now compares
  `_normal_equation_cholesky`, `_eigen_rcond_gate`, and `_svd_fallback` on
  500x1000 and 1000x2000 full-row-rank wide workloads.
- Retry condition: keep only if same-worker `u0ucw_wide_pinv` timings improve
  versus `_eigen_rcond_gate` without rank, pseudo-inverse tolerance,
  left-inverse certificate, or rcond-estimate audit regressions; if the
  eigenspectrum was not a measurable cost, reject this exact diagonal-rcond
  shortcut and do not retry unless profiles put `AA^T` symmetric eigensolve
  back in the top wide-`pinv` hotspot list.

## 2026-06-18 - frankenscipy-va60h - MDS streamed double-centering

- Agent: cod-a / MistyBirch
- Lever: remove internal full squared-distance materializations from
  `classical_mds`, `landmark_mds`, and `landmark_isomap` by streaming the
  double-centering formula from squared-distance callbacks; stream landmark
  triangulation dot products instead of allocating a per-point `dshift` vector.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-cluster` is expected before commit.
- Correctness guard: `double_centered_gram_matches_materialized_delta_reference`
  locks the streamed centering helper against the materialized formula, while
  existing MDS and Isomap recovery tests preserve embedding-distance behavior.
- Benchmark guard: compare focused `classical_mds`, `landmark_mds`, and
  `landmark_isomap` workloads against the previous same-worker commit,
  emphasizing large n with small k/m where allocation traffic competes with the
  randomized eigensolve.
- Retry condition: keep only if same-worker MDS/Isomap timings improve without
  embedding distance, eigenvalue ordering, or validation drift; if recomputing
  squared distances beats no allocation only in noise or regresses, reject this
  streaming double-centering formulation and do not retry unless allocation
  profiles again show `D^2`/`Delta` or `dshift` churn as a top cluster hotspot.

## 2026-06-18 - frankenscipy-acoco - jnjnp_zeros bracket reuse

- Agent: cod-a / MistyBirch
- Lever: reuse the per-order `J_n` zero sequence already computed by
  `jnjnp_zeros` when bracketing `J_n'` roots, avoiding a duplicate
  `jn_zeros(n, per)` call for every positive order.
- Status: pending batch-test. This is a code-first commit per campaign
  instruction; only local `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo check -p fsci-special` is expected before commit.
- Correctness guard: `derivative_bessel_zeros_match_scipy_reference_points`
  now checks that the helper-fed derivative-zero route is bit-identical to the
  public `jnp_zeros` output for representative order/count, while the existing
  Bessel-zero SciPy anchors preserve ordering and value tolerances.
- Benchmark guard: compare focused `jnjnp_zeros(nt=64..256)` workloads against
  the previous same-worker commit and SciPy oracle ordering/type outputs.
- Retry condition: keep only if same-worker special-zero timings improve
  without zero magnitude, order, serial, type, or tie-order drift; if duplicate
  `jn_zeros` work is hidden in noise by derivative Bessel evaluations, reject
  this bracket-reuse formulation and do not retry unless profiles put
  `jnp_zeros`' internal `jn_zeros` call back on the `jnjnp_zeros` hot path.
