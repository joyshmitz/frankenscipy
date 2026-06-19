# Performance Release-Readiness Scorecard

## 2026-06-19 - fsci-signal coherence fused-Welch gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.118`
- Decision: KEEP the fused coherence route. Score for this sub-cluster: 5/5.
- Artifact: `tests/artifacts/perf/frankenscipy-8l8r1.118/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b rch exec -- cargo check -p fsci-signal --all-targets` |
| Focused signal tests | PASS | `cargo test -p fsci-signal coherence_matches --lib -- --nocapture` passed via rch worker `hz2` |
| Criterion focused bench | PASS | `coherence_gauntlet_scipy` group only; local SciPy oracle plus rch Rust-only confirmation |
| SciPy head-to-head oracle | PASS | local Python oracle, `scipy.signal.coherence`; rch worker `hz1` lacked `scipy.signal`, so the remote run skipped the SciPy row |
| Changed benchmark formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-signal/benches/signal_bench.rs` |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust fused `coherence`, 65536 samples, Hann 1024/512 | 2.191980 ms | 8.65x faster than SciPy | SciPy win |
| Rust compositional triple-CSD route | 6.536569 ms | fused is 2.98x faster | internal win |
| SciPy `scipy.signal.coherence` | 18.961613 ms | 1.00x oracle | reference |
| Rust fused `coherence`, rch worker `hz1` | 4.3780 ms | 2.80x faster than compositional | internal win |
| Rust compositional triple-CSD route, rch worker `hz1` | 12.269 ms | 1.00x internal baseline | slower |

Readiness notes:

- The fused coherence API is now release evidence: it wins against original
  SciPy and the internal compositional route on the scoped workload.
- Future signal work should route deeper into FFT/window staging or shared
  Welch segment infrastructure instead of decomposing coherence back into
  independent `csd` calls.

## 2026-06-19 - fsci-cluster linkage flat-arena gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-va60h`
- Decision: KEEP the flat row-major linkage arena as an internal win, but mark
  the full `linkage` routine as a SciPy LOSS on the measured rows. Score for
  this sub-cluster: 3/5.
- Artifact:
  `tests/artifacts/perf/2026-06-19-va60h-linkage-gauntlet/`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-cluster --benches` passed; existing `perf_kmeans.rs` warning remains |
| Criterion focused bench | PASS | `va60h_gauntlet_linkage` group only; local SciPy oracle in the same benchmark process |
| SciPy head-to-head oracle | PASS | Python 3.13.7, NumPy 2.4.3, SciPy 1.17.1, `scipy.cluster.hierarchy.linkage` |
| Targeted cluster tests | PASS | `cargo test -p fsci-cluster linkage -- --nocapture` passed via rch: 28 unit tests and 9 metamorphic tests |
| Linkage conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_cluster_linkage_from_distances -- --nocapture` passed locally |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-cluster/benches/cluster_bench.rs crates/fsci-cluster/src/lib.rs` |
| Crate formatting | BLOCKED | `cargo fmt -p fsci-cluster --check` stopped on existing `crates/fsci-cluster/src/bin/perf_isomap.rs` formatting drift |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-cluster --benches -- -D warnings` stopped on existing `fsci-linalg` dependency lints before this benchmark file was linted |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current flat `linkage(Average)`, n=800 d=4 | 6.1713 ms | 1.385x slower than SciPy | SciPy loss, internal keep |
| Rust legacy nested helper `linkage(Average)`, n=800 d=4 | 6.9616 ms | current flat is 1.128x faster | internal win |
| SciPy `linkage(method="average")`, n=800 d=4 | 4.4550 ms | 1.00x oracle | reference |
| Rust current flat `linkage(Ward)`, n=800 d=4 | 7.5250 ms | 1.497x slower than SciPy | SciPy loss, internal neutral/win |
| Rust legacy nested helper `linkage(Ward)`, n=800 d=4 | 7.6707 ms | current flat is 1.019x faster | internal neutral/win |
| SciPy `linkage(method="ward")`, n=800 d=4 | 5.0256 ms | 1.00x oracle | reference |

Readiness notes:

- A manual production revert probe was run and then undone: the nested-route
  probe measured 9.0669 ms on Average and 9.1589 ms on Ward, while the flat
  route measured 7.0279 ms and 7.3221 ms in the same post-revert run.
- The flat arena should stay because reverting it makes the production route
  slower, but this is not release-speed parity against original SciPy.
- Future linkage work should target the algorithmic gap with SciPy's compiled
  linkage implementation rather than another full-square matrix layout lever.

## 2026-06-19 - fsci-special jnjnp_zeros bracket-reuse gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-acoco`
- Decision: KEEP the bracket-reuse optimization as an internal win, but mark
  the full routine as a SciPy LOSS. Score for this sub-cluster: 3/5.
- Artifact:
  `tests/artifacts/perf/2026-06-19-acoco-jnjnp-zeros-gauntlet/`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-special --benches` |
| Criterion focused bench | PASS | `acoco_gauntlet_jnjnp_zeros` group only; local SciPy oracle in the same benchmark process |
| SciPy head-to-head oracle | PASS | Python 3.13.7, NumPy 2.4.3, SciPy 1.17.1, `scipy.special.jnjnp_zeros(nt)` |
| Targeted special tests | PASS | `jnyn_and_jnjnp_zeros_match_scipy` and `derivative_bessel_zeros_match_scipy_reference_points` passed via rch |
| Benchmark formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-special/benches/special_bench.rs` |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-special --benches -- -D warnings` stopped on existing `fsci-integrate` and `fsci-linalg` dependency lints before this benchmark file was linted |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current `jnjnp_zeros(nt=64)` | 80.728603 ms | 163.53x slower than SciPy | SciPy loss |
| Rust legacy duplicate route, `nt=64` | 101.762454 ms | current is 1.26x faster | internal keep |
| SciPy `scipy.special.jnjnp_zeros(nt=64)` | 0.493655 ms | 1.00x oracle | reference |
| Rust current `jnjnp_zeros(nt=128)` | 410.059973 ms | 443.57x slower than SciPy | SciPy loss |
| Rust legacy duplicate route, `nt=128` | 544.006333 ms | current is 1.33x faster | internal keep |
| SciPy `scipy.special.jnjnp_zeros(nt=128)` | 0.924456 ms | 1.00x oracle | reference |

Readiness notes:

- The bracket-reuse optimization should stay because reverting it would make
  the current Rust path slower on both measured rows.
- This is not release-speed evidence against original SciPy. The next
  performance bead should target the zero-enumeration/root-finding algorithm
  rather than another duplicate-bracketing micro-lever.
- The clippy blocker is outside this benchmark/evidence change and should be
  handled as a separate lint-hygiene bead.

## 2026-06-19 - fsci-fft CSD rfft-route gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.116`
- Decision: REJECT and revert the `cross_spectral_density` rfft real-spectrum
  route. Score for this sub-cluster: 3/5.
- Artifact: `tests/artifacts/perf/2026-06-19-fft-csd-gauntlet/csd_rfft_reject.json`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate Criterion bench | PASS | same-worker parent/candidate A/B on `hz1`; post-revert confirmation landed on `ovh-a` and is not used for the A/B |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 / NumPy 2.4.3, equivalent `scipy.fft.rfft` cross-spectrum formula |
| Source revert | PASS | `crates/fsci-fft/src/helpers.rs` restored to the full-complex route |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo check -p fsci-fft --all-targets` |
| CSD conformance guard | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft cross_spectral_density -- --nocapture` |
| Clippy `-D warnings` | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo clippy -p fsci-fft --all-targets -- -D warnings`; required a test-only `dst` import and one golden-value precision cleanup |
| Golden-value guard | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b RUSTFLAGS='-C force-frame-pointers=yes' cargo test -p fsci-fft rfft_matches_exact_numpy_dft_golden_values -- --nocapture` |
| Diff hygiene | PASS | `git diff --check` |
| UBS changed-file scan | PASS | `ubs` on the changed file set exited 0; warnings were broad existing FFT inventory, not commit-blocking criticals |
| Changed-file formatting | PASS/BLOCKED | `rustfmt --edition 2024 --check` passes for the changed bench/import surface; broad crate formatting still reports pre-existing file-wide drift in `src/lib.rs` and `src/helpers.rs`, so this commit does not normalize unrelated formatting |

Same-worker internal A/B:

| Workload | Parent full-complex mean | Candidate rfft mean | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `fft_helpers/cross_spectral_density/4096` | 112.08 us | 125.88 us | 1.123x | loss |
| `fft_helpers/cross_spectral_density/65536` | 4.9543 ms | 2.3509 ms | 0.475x | win |

Candidate vs original SciPy rfft formula:

| Workload | Candidate Rust mean | SciPy p50 | SciPy/Rust | Verdict |
| --- | ---: | ---: | ---: | --- |
| `cross_spectral_density/4096` | 125.88 us | 72.091 us | 0.573x | Rust slower |
| `cross_spectral_density/65536` | 2.3509 ms | 1.653584 ms | 0.703x | Rust slower |

Readiness notes:

- The rfft route is not release evidence: it has one internal win, one
  internal loss, and loses both rows against the fastest equivalent SciPy
  formula.
- The CSD Criterion rows and SciPy oracle script remain as durable gauntlet
  infrastructure for future FFT work.
- Future attempts need a same-worker win on both 4096 and 65536 and must beat
  the SciPy rfft formula before replacing the full-complex route again.

## 2026-06-19 - fsci-opt L-BFGS-B Wolfe-probe gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-8l8r1.122`
- Decision: REJECT and revert the mutable Wolfe finite-difference
  gradient-probe scratch path. Score for this sub-cluster: 4/5.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-lbfgsb-gauntlet/lbfgsb_wolfe_probe_reject.json`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `cargo check -p fsci-opt --all-targets` via rch with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b` |
| Criterion focused bench | PASS | `lbfgsb` group only; same-worker parent/candidate A/B on `ovh-a`, post-revert current-tree sample on `hz2` |
| SciPy head-to-head oracle | PASS | Local Python oracle, SciPy 1.17.1 / NumPy 2.4.3; rch declined the Python oracle as non-compilation |
| Targeted opt tests | PASS | `cargo test -p fsci-opt lbfgsb -- --nocapture` passed |
| L-BFGS-B conformance | PASS | `FSCI_REQUIRE_SCIPY_ORACLE=1 cargo test -p fsci-conformance --test diff_opt_lbfgsb_minimize -- --nocapture` passed |
| Formatting | PASS | `cargo fmt -p fsci-opt --check` passed |
| Clippy `-D warnings` | PASS | `cargo clippy -p fsci-opt --all-targets -- -D warnings` passed after minimal fsci-opt lint cleanup |

Same-worker internal A/B:

| Workload | Parent p50 (us) | Candidate p50 (us) | Candidate time vs parent | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 17.491 | 17.405 | 0.995x | neutral |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 87.087 | 106.440 | 1.222x | loss |
| `lbfgsb/quadratic_unconstrained_fd/32` | 5.246 | 6.055 | 1.154x | loss |

Post-revert current route vs SciPy:

| Workload | Rust p50 (us) | SciPy p50 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | --- |
| `lbfgsb/rosenbrock_unconstrained_fd/2` | 22.236 | 4585.899 | 206.24x | current route remains fast |
| `lbfgsb/rosenbrock_unconstrained_fd/10` | 105.090 | 18262.642 | 173.78x | current route remains fast |
| `lbfgsb/quadratic_unconstrained_fd/32` | 6.313 | 1447.172 | 229.23x | current route remains fast |

Readiness notes:

- The release candidate is the reverted parent-style Strong-Wolfe path, not the
  mutable finite-difference probe scratch path from `b5dbf124`.
- The original SciPy ratios remain strong on realistic Python-callback
  workloads, but the rejected lever made the in-crate route slower on the
  larger same-worker rows and is not release evidence.
- Future opt work should route to a fresh measured hotspot instead of retrying
  Wolfe finite-difference probe Vec reuse without new allocation-profile proof.

## 2026-06-19 - fsci-linalg wide lstsq row-streaming gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-u0ucw`
- Decision: REVERT row-streamed wide `lstsq`; KEEP current materialized
  normal-equation route. Score for this sub-cluster: 4/5.
- Artifact: `tests/artifacts/perf/2026-06-19-u0ucw-wide-lstsq-gauntlet/`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo check -p fsci-linalg --benches` |
| Criterion focused bench | PASS | same-worker row-streaming vs materialized A/B on `vmi1227854`; local SciPy row for original-SciPy ratio |
| SciPy head-to-head oracle | PASS | SciPy 1.17.1 / NumPy 2.4.3, `scipy.linalg.lstsq(check_finite=False)` |
| Targeted linalg tests | PASS | `wide_pinv` filtered tests passed, preserving the surviving row-major wide `pinv` helpers |
| Release route probe | PASS | ignored release probe reported `lstsq_max_abs_diff=3.38840067115597776e-13` |
| Changed bench formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-linalg/benches/linalg_bench.rs` |
| Direct `src/lib.rs` formatting | BLOCKED | file-wide pre-existing rustfmt drift outside this revert; broad-formatting would create unrelated churn in the shared checkout |
| Clippy `-D warnings` | BLOCKED | pre-existing `src/lib.rs` lints plus concurrent `src/cossin.rs` excessive-precision lints; no row-streaming revert-specific issue identified |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust row-streamed wide `lstsq`, 500x1000 | 139.965 ms | 0.966x vs materialized | loss, reverted |
| Rust materialized wide `lstsq`, 500x1000 | 135.206 ms | 1.035x vs row-streamed | keep old route |
| Rust current materialized wide `lstsq`, 500x1000 | 109.370 ms | 11.46x vs SciPy | keep |
| SciPy `scipy.linalg.lstsq`, 500x1000 | 1.253347 s | 1.00x oracle | reference |

Readiness notes:

- The negative result is specific to replacing the wide `lstsq` materialized
  transpose products with row streaming. It does not invalidate the public wide
  normal-equation route, which remains faster than SciPy on the measured
  workload.
- `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a cargo clippy -p fsci-linalg --benches -- -D warnings`
  currently fails on existing lints unrelated to the measured revert, including
  `needless_range_loop` / `needless_borrow` rows in `src/lib.rs` and
  excessive-precision rows in concurrently modified `src/cossin.rs`.
- Future retries need a fresh allocation/cache profile and a same-worker >10%
  win over the materialized route before this formulation should be reconsidered.

## 2026-06-19 - fsci-opt least_squares scratch cluster

- Agent: cod-b / MistyBirch
- Commit under verification: `41bf34a4`
- Beads: `frankenscipy-szky7`, `frankenscipy-y1mzk`
- Decision: KEEP, no revert. Score for this cluster: 4/5.
- Artifact: `tests/artifacts/perf/2026-06-19-opt-least-squares-gauntlet/least_squares_vs_scipy.json`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b cargo check -p fsci-opt` |
| Criterion focused bench | PASS | `least_squares` group only, remote worker `vmi1227854` |
| SciPy head-to-head oracle | PASS | SciPy 1.17.1 / NumPy 2.4.3, warmed single-process `method="lm"` |
| Targeted metamorphic tests | PASS | 2 least-squares rows passed via `--test metamorphic_tests mr_least_squares` |
| Release diff probes | PASS | `diff_lsq` and `diff_leastsq` release binaries converged |
| Broad fsci-opt unit gate | BLOCKED | Pre-existing `src/lib.rs` test imports fail before the target tests run; follow-up `frankenscipy-uxs8k` |

| Workload | Rust p50 (us) | SciPy p50 (us) | SciPy p99 (us) | SciPy/Rust p50 | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `least_squares/rosenbrock_residual` | 2.558 | 1404.547 | 2645.985 | 549.08x | win |
| `least_squares/exp_curve_64` | 16.932 | 753.120 | 1452.558 | 44.48x | win |
| `least_squares/exp_linear_curve_128` | 49.724 | 893.946 | 1414.085 | 17.98x | win |

Readiness notes:

- The measured workloads include Python callback overhead on the SciPy side.
  That is intentional for the original-SciPy realistic usage path targeted by
  this gauntlet; it is not evidence about lower-level C-only kernels.
- No neutral or loss rows were observed for this cluster, so the scratch-reuse
  optimization remains in tree.
- The remaining release risk is test-harness hygiene, not this optimization:
  broad `fsci-opt` unit-test compilation needs the missing helper imports fixed
  before this lane can claim a full crate test pass.
