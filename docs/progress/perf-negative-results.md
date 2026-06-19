# Performance Negative-Evidence Ledger

This ledger records every code-first performance attempt, including attempts that
are still awaiting the batch benchmark wave. Entries must name the retry
condition so dead ends are not repeated casually.

## 2026-06-19 - frankenscipy-acdq2 - ndimage gaussian_filter line-walk route

- Agent: cod-a / MistyBirch
- Lever: force `gaussian_filter1d_axis` onto the 1-D slab line walker for every
  axis, then add outermost-axis row-splitting and direct interior tap indexing
  to avoid `boundary_index_1d` on non-border rows.
- Graveyard/artifact route tested: cache-aware line walking, branch removal in
  the interior stencil, and parallel row chunks for the low-outer-slab case.
- Decision: REJECT AND REVERT. No source change shipped.
- Artifact:
  `tests/artifacts/perf/2026-06-19-ndimage-gaussian-linewalk-reject/EVIDENCE.md`
- Candidate command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- correlate_gaussian/gaussian_sigma2/256 --noplot`
- Candidate result, rch worker `vmi1152480`:

  | Workload | Candidate mean | Candidate interval | Verdict |
  | --- | ---: | ---: | --- |
  | `correlate_gaussian/gaussian_sigma2/256` | 4.2236 ms | [4.0337 ms, 4.4246 ms] | loss |

- Clean current result on `origin/main` commit `96a37a83`, rch worker `ovh-a`:

  | Workload | Current mean | Current interval | Candidate/current |
  | --- | ---: | ---: | ---: |
  | `correlate_gaussian/gaussian_sigma2/256` | 2.4792 ms | [2.4545 ms, 2.5044 ms] | 1.704x slower |

- Existing ledger current result before this attempt was also faster than the
  candidate: 3.238 ms. That makes the route a rejection even with cross-worker
  noise.
- Local SciPy oracle (`python3 docs/perf_oracle_ndimage.py`, SciPy 1.17.1):
  `gaussian_filter sigma=2 256x256` p50 was 1.47107 ms. The rejected candidate
  is 2.87x slower than that oracle; the clean current rch row is still slower
  and remains a gap.
- Correctness/conformance guard: no optimized source was kept, so the
  conformance surface is unchanged from `origin/main`.
- Retry condition: do not retry fallback removal, outermost-axis row-splitting,
  or scalar direct-index boundary peeling for this workload without a fresh
  profile. The next plausible route is a lower-level contiguous-span/SIMD dot
  kernel or a cache-tiled separable scratch/transpose path that preserves the
  existing gaussian tolerance contract.

## 2026-06-19 - frankenscipy-nm8ex - spatial pdist dim-4 fast path

- Agent: cod-a / MistyBirch
- Lever: specialize `pdist` for the measured 4-D Euclidean and Cosine gap by
  bypassing per-pair metric dispatch, generic slice reductions, and SIMD-tail
  setup, then force the now-cheap dim-4 path through the serial gate to avoid
  thread-spawn overhead at n=256/512. The kept path uses direct dim-4
  dot/squared-norm arithmetic while leaving the generic pair-balanced row split
  available for other metrics and shapes.
- Graveyard/artifact route tested: cache/constant-factor collapse for the tight
  O(n^2) kernel, branch removal inside the metric loop, and shape-specific
  codegen without unsafe code.
- Decision: KEEP as an internal win, route residual SciPy loss deeper. No
  revert.
- Artifact:
  `tests/artifacts/perf/2026-06-19-nm8ex-pdist-dim4/EVIDENCE.md`
- Baseline/candidate command:
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo bench -p fsci-spatial --bench spatial_bench -- pdist --noplot`
- SciPy oracle command:
  `python3 docs/perf_oracle_pdist.py`
- Same-worker internal A/B on rch worker `hz2`:

  | Workload | Baseline mean | Candidate mean | Candidate/baseline | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | `pdist/euclidean/256` | 2.9264 ms | 107.45 us | 0.0367x time, 27.24x faster | keep |
  | `pdist/cosine/256` | 3.3658 ms | 114.04 us | 0.0339x time, 29.51x faster | keep |
  | `pdist/euclidean/512` | 3.5554 ms | 425.75 us | 0.1197x time, 8.35x faster | keep |
  | `pdist/cosine/512` | 2.5548 ms | 461.16 us | 0.1805x time, 5.54x faster | keep |

- Local SciPy oracle (`scipy.spatial.distance.pdist`, SciPy 1.17.1):

  | Workload | Candidate mean | SciPy p50 | Candidate/SciPy | Verdict |
  | --- | ---: | ---: | ---: | --- |
  | Euclidean n=256 d=4 | 107.45 us | 94.30 us | 1.14x slower | residual near-parity |
  | Cosine n=256 d=4 | 114.04 us | 87.20 us | 1.31x slower | residual loss |
  | Euclidean n=512 d=4 | 425.75 us | 310.83 us | 1.37x slower | residual loss |
  | Cosine n=512 d=4 | 461.16 us | 283.75 us | 1.63x slower | residual loss |

- Correctness/conformance guards:
  - PASS: `rch exec -- cargo test -p fsci-spatial pdist --lib -- --nocapture`
    (10 passed).
  - PASS: `rch exec -- cargo test -p fsci-spatial --lib -- --nocapture`
    (206 passed, 2 ignored).
  - PASS: `rch exec -- cargo check -p fsci-spatial --all-targets`; existing
    warning remains `point_in_circumcircle` dead code.
  - BLOCKED: `rch exec -- cargo test -p fsci-conformance -- --nocapture`
    fails before this spatial lane in `crates/fsci-conformance/tests/e2e_sparse.rs`
    with `SolveResult` passed where `&[f64]` is expected.
  - BLOCKED: `rch exec -- cargo clippy -p fsci-spatial --all-targets -- -D warnings`
    stops in dependency crate `fsci-linalg` on existing lint debt.
  - BLOCKED: `cargo fmt -p fsci-spatial --check` reports pre-existing
    `fsci-spatial` source/bench rustfmt drift outside this patch; `git diff --check`
    is clean.
- Retry condition: do not retry generic metric-dispatch removal, dim-4 scalar
  helper extraction, or dim-4 serial gating for this workload. The next
  credible route is a deeper layout/kernel change: packed SoA/flat
  contiguous point storage, batch several pair outputs per inner loop, or a
  generated AVX-style dim-specialized kernel that removes `Vec<Vec<f64>>`
  pointer chasing and matches SciPy's C distance loop constants.

## 2026-06-19 - frankenscipy-8l8r1.115 - randomized_eigh projected sketch

- Agent: cod-b / MistyBirch
- Lever: keep `randomized_eigh` on a projected symmetric sketch: deterministic
  random block, thin modified Gram-Schmidt basis, two power iterations, and a
  full eigensolve only on the small `q^T A q` matrix.
- Decision: KEEP. No revert. The route is a measured head-to-head win against
  SciPy subset `eigh` on the scoped low-rank symmetric top-k workloads.
- Artifact: `tests/artifacts/perf/frankenscipy-8l8r1.115/EVIDENCE.md`
- Remote guard commands:
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-linalg --all-targets`
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo test -p fsci-linalg randomized_eigh_matches_full_eigh_on_low_rank --lib -- --nocapture`
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo bench -p fsci-linalg --bench linalg_bench randomized_eigh_gauntlet_scipy -- --noplot`
- SciPy oracle command: local same-host run
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo bench -p fsci-linalg --bench linalg_bench randomized_eigh_gauntlet_scipy -- --noplot`
  because rch worker `ovh-a` skipped the SciPy rows with missing
  `scipy.linalg`.

| Workload | Rust randomized mean | Rust full `eigh` mean | SciPy subset `eigh` mean | SciPy/Rust randomized | Full/Rand internal | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 256x256, k=16 | 3.492738 ms | 11.509742 ms | 4.879053 ms | 1.40x faster | 3.30x faster | win |
| 512x512, k=24 | 16.774593 ms | 136.046217 ms | 198.116969 ms | 11.81x faster | 8.11x faster | win |

- Negative evidence: Rust full `eigh` remains 3.30x to 8.11x slower than the
  projected route on these top-k low-rank shapes. Do not route this workload
  back through the full dense eigensolver unless a future conformance failure
  invalidates the randomized projection contract.
- Retry condition: continue only with deeper sketch-quality or blocked-operator
  improvements that preserve the existing low-rank invariant tests and improve
  the same `randomized_eigh_gauntlet_scipy` group; reject scalar housekeeping or
  allocation-only tweaks that do not move this head-to-head SciPy ratio.

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
- Status: measured REJECT and reverted. The attempted mutable-probe path was
  neutral on 2D Rosenbrock and slower on both larger rows, so the source path
  was restored to the parent `line_search_wolfe2` implementation while keeping
  the new Criterion/SciPy measurement harness.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-lbfgsb-gauntlet/lbfgsb_wolfe_probe_reject.json`
- Optimization commit under test: `b5dbf1244e52632edc9bd0edc2102cb3ff78dfad`
  vs parent `69ae5d214f8e90356789b112cff30a5c69b43d2a`.

Same-worker internal A/B (`ovh-a`, Criterion p50):

| Workload | Parent p50 (us) | Candidate p50 (us) | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 17.491 | 17.405 | 0.995x | neutral |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 87.087 | 106.440 | 1.222x | loss |
| `lbfgsb/quadratic_unconstrained_fd/32` | 5.246 | 6.055 | 1.154x | loss |

Post-revert current route vs original SciPy (`hz2` Rust Criterion p50, local
SciPy 1.17.1 / NumPy 2.4.3 oracle p50):

| Workload | Current Rust p50 (us) | SciPy p50 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 22.236 | 4585.899 | 206.24x | current route remains fast |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 105.090 | 18262.642 | 173.78x | current route remains fast |
| `lbfgsb/quadratic_unconstrained_fd/32` | 6.313 | 1447.172 | 229.23x | current route remains fast |

- Correctness/conformance guards:
  - PASS: `cargo fmt -p fsci-opt --check`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-opt --all-targets`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-opt lbfgsb -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_opt_lbfgsb_minimize -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-opt --all-targets -- -D warnings`
- Retry condition: do not retry this mutable finite-difference probe scratch
  formulation. Reopen only if a fresh allocation profile again puts
  `L-BFGS-B` Wolfe gradient-probe Vec churn in the top-5 `fsci-opt` hotspots
  and a new same-worker parent/candidate A/B shows a material win across the
  10D Rosenbrock and 32D quadratic rows.

## 2026-06-19 - frankenscipy-8l8r1.116 - FFT CSD rfft real-spectrum route

- Agent: cod-b / MistyBirch
- Lever: route `fsci_fft::cross_spectral_density` real input pairs through
  two `rfft` calls instead of promoting both inputs to complex buffers and
  running two full complex FFTs before taking the one-sided half.
- Status: measured REJECT and reverted. The rfft route won the large internal
  row but regressed the 4096 row and remained slower than the fastest
  equivalent SciPy `scipy.fft.rfft` cross-spectrum formula on both measured
  sizes. The release route is restored to the full-complex implementation while
  keeping the Criterion CSD rows, SciPy oracle script, and full-complex
  equivalence guard.
- Artifact: `tests/artifacts/perf/2026-06-19-fft-csd-gauntlet/csd_rfft_reject.json`
- Optimization commit under test: `d027c140bc8a937877e8c018cf7265d1f4bc5049`
  vs parent `55f32c99f69991f4ab252621dd86948dc6e95b20`.

Same-worker internal A/B (`hz1`, Criterion mean; parent scratch had only the
same CSD bench harness added, with parent library code unchanged):

| Workload | Parent full-complex mean | Candidate rfft mean | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `fft_helpers/cross_spectral_density/4096` | 112.08 us | 125.88 us | 1.123x | loss |
| `fft_helpers/cross_spectral_density/65536` | 4.9543 ms | 2.3509 ms | 0.475x | win |

Local original-SciPy oracle (`python3 docs/perf_oracle_fft_csd.py --reps 120
--warmups 5`, Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1):

| Workload | SciPy rfft-formula p50 | Candidate Rust mean | SciPy/Rust | Verdict |
| --- | ---: | ---: | ---: | --- |
| `cross_spectral_density/4096` | 72.091 us | 125.88 us | 0.573x | Rust slower |
| `cross_spectral_density/65536` | 1.653584 ms | 2.3509 ms | 0.703x | Rust slower |

- Correctness/conformance guards:
  - PASS: `python3 docs/perf_oracle_fft_csd.py --reps 120 --warmups 5`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo bench -p fsci-fft --bench fft_bench -- fft_helpers/cross_spectral_density --noplot` for the parent/candidate measurement rows above
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-fft --all-targets`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft cross_spectral_density -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-fft --all-targets -- -D warnings`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft rfft_matches_exact_numpy_dft_golden_values -- --nocapture`
  - PASS: `git diff --check` and targeted `rustfmt --edition 2024 --check crates/fsci-fft/benches/fft_bench.rs crates/fsci-fft/src/transforms.rs`
  - PASS: `ubs` on the changed file set exited 0.
  - BLOCKED: broad `fsci-fft` rustfmt still reports pre-existing file-wide drift in `src/lib.rs` and `src/helpers.rs`; this commit avoids unrelated formatting churn.
- Retry condition: do not retry this standalone rfft CSD route. Reopen only if
  a fresh same-worker profile attributes a clearly-above-noise share to full
  complex promotion inside `cross_spectral_density` on a workload of at least
  65536 samples, and the replacement beats the fastest equivalent SciPy rfft
  formula while not regressing the 4096 row.

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
- Status: measured gauntlet complete on 2026-06-19. Decision: KEEP the flat
  arena as an internal win, but record the full routine as a SciPy LOSS on this
  workload. A direct production revert probe showed the nested route would be
  slower on both measured rows, so no revert remains in the release candidate.
- Artifact:
  `tests/artifacts/perf/2026-06-19-va60h-linkage-gauntlet/`
- Correctness guard: benchmark setup asserted current flat rows are
  byte-identical to the benchmark-local legacy nested NN-array route; filtered
  `fsci-cluster linkage` tests passed via rch (28 unit tests, 9 metamorphic
  tests), including `linkage_flat_core_matches_precomputed_condensed_contract`;
  `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test
  diff_cluster_linkage_from_distances -- --nocapture` passed locally against
  SciPy 1.17.1.
- Benchmark command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a
  cargo bench -p fsci-cluster --bench cluster_bench --
  va60h_gauntlet_linkage --noplot`.

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current flat `linkage(Average)`, n=800 d=4 | 6.1713 ms | 1.385x slower than SciPy | SciPy loss, internal keep |
| Rust legacy nested helper `linkage(Average)`, n=800 d=4 | 6.9616 ms | current flat is 1.128x faster | internal win |
| SciPy `scipy.cluster.hierarchy.linkage(method="average")`, n=800 d=4 | 4.4550 ms | 1.00x oracle | reference |
| Rust current flat `linkage(Ward)`, n=800 d=4 | 7.5250 ms | 1.497x slower than SciPy | SciPy loss, internal neutral/win |
| Rust legacy nested helper `linkage(Ward)`, n=800 d=4 | 7.6707 ms | current flat is 1.019x faster | internal neutral/win |
| SciPy `scipy.cluster.hierarchy.linkage(method="ward")`, n=800 d=4 | 5.0256 ms | 1.00x oracle | reference |

- Revert probe: manually reverting production to nested rows and rerunning the
  same Criterion group showed the flat route faster than the reverted production
  route by 1.290x on Average and 1.251x on Ward. The revert was therefore
  undone before commit.
- Gate notes: `cargo check -p fsci-cluster --benches` passed with an existing
  `perf_kmeans.rs` warning. `cargo fmt -p fsci-cluster --check` is blocked by
  existing `perf_isomap.rs` formatting drift; `cargo clippy -p fsci-cluster
  --benches -- -D warnings` is blocked by existing `fsci-linalg` dependency
  lints before this benchmark file is linted.
- Next route: if release parity against SciPy is required for hierarchical
  clustering, route deeper into the algorithmic gap with SciPy's compiled
  linkage implementation rather than retrying full-square arena layout changes.

## 2026-06-18 - frankenscipy-8l8r1.118 - coherence chunk-local spectra accumulator

- Agent: cod-b / MistyBirch
- Lever: keep `coherence` on the fused Welch segment pass, but replace the
  per-segment `CoherenceSegmentSpectra` materialization with chunk-local
  Pxy/Pxx/Pyy accumulators and reusable windowed `wx`/`wy` scratch buffers. This
  removes one spectra allocation triplet per segment and avoids retaining all
  segment spectra before the final fold on the target
  `spectral/coherence/65536_w1024_o512` workload.
- Status: measured gauntlet complete on 2026-06-19. Decision: KEEP. The fused
  coherence route beats original `scipy.signal.coherence` by 8.65x on the
  scoped 65536-sample Hann-window workload, and beats the internal triple-CSD
  composition by 2.98x locally and 2.80x on rch worker `hz1`.
- Artifact: `tests/artifacts/perf/frankenscipy-8l8r1.118/EVIDENCE.md`.
- Correctness guard: existing `coherence_matches_compositional_csd_formula`
  compares the fused path against `csd(x,y)`, `csd(x,x)`, and `csd(y,y)`;
  existing SciPy-reference coherence coverage anchors the public tolerance
  contract. `cargo check -p fsci-signal --all-targets` and the focused
  `coherence_matches` tests passed via rch.

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust fused `coherence`, local SciPy-oracle host | 2.191980 ms | 8.65x faster than SciPy | SciPy win |
| Rust compositional triple-CSD route, local host | 6.536569 ms | fused is 2.98x faster | internal win |
| SciPy `scipy.signal.coherence`, local host | 18.961613 ms | 1.00x oracle | reference |
| Rust fused `coherence`, rch worker `hz1` | 4.3780 ms | 2.80x faster than compositional | internal win |
| Rust compositional triple-CSD route, rch worker `hz1` | 12.269 ms | 1.00x internal baseline | slower |

- Retry condition: do not retry the triple-Welch/triple-CSD coherence route
  unless a same-host gauntlet reverses both the >=2.8x internal fused win and
  the >8x SciPy win. Future signal work should route below this API into FFT
  staging, windowing, or shared Welch segment infrastructure rather than
  decomposing coherence back into independent `csd` calls.

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
- Lever tested: route full-row-rank wide `lstsq` normal-equation products
  through the caller's row-major input: form `A A^T` from contiguous row dot
  products and compute `A^T y` / `A^T dy` by streaming rows once, avoiding the
  materialized `A^T` matrix.
- Decision: REVERT. The row-streamed candidate is slower than the prior
  materialized transpose route on the same `rch` worker. The retained current
  materialized route remains much faster than original SciPy on the realistic
  500x1000 workload, so the regression is in the code-first micro-lever rather
  than in the public algorithm choice.
- Artifact: `tests/artifacts/perf/2026-06-19-u0ucw-wide-lstsq-gauntlet/`
- Commands:
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo bench -p fsci-linalg --bench linalg_bench -- u0ucw_gauntlet_scipy_lstsq --noplot`
  - `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo bench -p fsci-linalg --bench linalg_bench -- u0ucw_gauntlet_scipy_lstsq --noplot`

| Route | Environment | Criterion mean | Ratio | Verdict |
| --- | --- | ---: | ---: | --- |
| Rust row-streamed `A A^T` + `A^T y` | `rch` worker `vmi1227854` | 139.965 ms | 0.966x vs materialized | loss, reverted |
| Rust materialized `A^T` pre-revert | `rch` worker `vmi1227854` | 135.206 ms | 1.00x internal reference | keep old route |
| Rust current materialized `A^T` after revert | local SciPy host | 109.370 ms | 11.46x faster than SciPy | keep |
| SciPy `scipy.linalg.lstsq(check_finite=False)` | local Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1 | 1.253347 s | 1.00x oracle | reference |

- Conformance/correctness guard:
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-linalg --benches`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-linalg wide_pinv -- --nocapture`
  - PASS: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo test -p fsci-linalg public_wide_min_norm_lstsq_route_perf_probe --release -- --ignored --nocapture`
  - The release probe reported `shape=256x512`, `lstsq_speedup=15.571283`,
    and `lstsq_max_abs_diff=3.38840067115597776e-13` against the reference
    route.
- Infrastructure note: `vmi1227854` did not have SciPy importable, so the
  row-streaming-vs-materialized loss uses remote same-worker Rust A/B evidence,
  while the ratio-vs-SciPy row uses the local host with SciPy installed.
- Retry condition: do not retry this exact row-streamed wide `lstsq`
  formulation unless allocation or cache profiles put wide-route `A^T`
  materialization back in the top linalg hotspot list and a same-worker A/B run
  beats the materialized route by at least 10% without rank, singular-value,
  min-norm, residual, or public `lstsq` tolerance drift.

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

## 2026-06-19 - frankenscipy-8l8r1.116 - FFT CSD rfft real-spectrum route

- Agent: cod-a / MistyBirch
- Lever: route `fsci_fft::cross_spectral_density` real input pairs through
  two `rfft` calls instead of promoting both inputs to complex buffers and
  running two full complex FFTs before taking the one-sided half.
- Status: measured REJECT and reverted. The rfft route won the large internal
  row but regressed the 4096 row and remained slower than the fastest
  equivalent SciPy `scipy.fft.rfft` cross-spectrum formula on both measured
  sizes. The release route is restored to the full-complex implementation while
  keeping the Criterion CSD rows, SciPy oracle script, and full-complex
  equivalence guard.
- Artifact:
  `tests/artifacts/perf/2026-06-19-fft-csd-gauntlet/csd_rfft_reject.json`
- Optimization commit under test: `d027c140bc8a937877e8c018cf7265d1f4bc5049`
  vs parent `55f32c99f69991f4ab252621dd86948dc6e95b20`.

Same-worker internal A/B (`hz1`, Criterion mean; parent scratch had only the
same CSD bench harness added, with parent library code unchanged):

| Workload | Parent full-complex mean | Candidate rfft mean | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `fft_helpers/cross_spectral_density/4096` | 112.08 us | 125.88 us | 1.123x | loss |
| `fft_helpers/cross_spectral_density/65536` | 4.9543 ms | 2.3509 ms | 0.475x | win |

Local original-SciPy oracle (`python3 docs/perf_oracle_fft_csd.py --reps 120
--warmups 5`, Python 3.13.7 / NumPy 2.4.3 / SciPy 1.17.1):

| Workload | SciPy rfft-formula p50 | Candidate Rust mean | SciPy/Rust | Verdict |
| --- | ---: | ---: | ---: | --- |
| `cross_spectral_density/4096` | 72.091 us | 125.88 us | 0.573x | Rust slower |
| `cross_spectral_density/65536` | 1.653584 ms | 2.3509 ms | 0.703x | Rust slower |

- Correctness/conformance guards to re-run in this closeout:
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-fft --all-targets`
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft cross_spectral_density -- --nocapture`
  - `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-fft --all-targets -- -D warnings`
  - `python3 docs/perf_oracle_fft_csd.py --reps 120 --warmups 5`
  - `git diff --check` and `ubs` on the changed file set.
- Retry condition: do not retry this standalone rfft CSD route. Reopen only if
  a fresh same-worker profile attributes a clearly-above-noise share to full
  complex promotion inside `cross_spectral_density` on a workload of at least
  65536 samples, and the replacement beats the fastest equivalent SciPy rfft
  formula while not regressing the 4096 row.

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
- Status: MEASURED INTERNAL KEEP / SCIPY LOSS. The bracket-reuse route is
  faster than the benchmark-only recreation of the previous duplicate
  bracketing route, so it stays in tree. The full routine is still much slower
  than original SciPy and must route to a deeper algorithmic optimization.
- Artifact:
  `tests/artifacts/perf/2026-06-19-acoco-jnjnp-zeros-gauntlet/`.
- Head-to-head result, Criterion point-estimate means:

  | Workload | Rust current | Rust legacy duplicate route | SciPy original | Current vs SciPy | Current vs legacy |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `jnjnp_zeros(nt=64)` | 80.728603 ms | 101.762454 ms | 0.493655 ms | 163.53x slower | 1.26x faster |
  | `jnjnp_zeros(nt=128)` | 410.059973 ms | 544.006333 ms | 0.924456 ms | 443.57x slower | 1.33x faster |
- Correctness guard: `derivative_bessel_zeros_match_scipy_reference_points`
  passed via rch, and the existing
  `jnyn_and_jnjnp_zeros_match_scipy` SciPy anchor also passed.
- Build/format guard: `cargo check -p fsci-special --benches` passed via rch;
  `rustfmt --edition 2024 --check crates/fsci-special/benches/special_bench.rs`
  passed.
- Blocked lint guard: `cargo clippy -p fsci-special --benches -- -D warnings`
  failed before this benchmark file on existing dependency lints in
  `fsci-integrate` and `fsci-linalg`.
- Retry condition: do not retry this exact duplicate-`jn_zeros` lever without a
  fresh profile. Future work should target SciPy's much faster zero
  enumeration/root-finding strategy or another measured special-function
  hotspot.
