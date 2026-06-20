# Performance Release-Readiness Scorecard

## 2026-06-20 - fsci-integrate stiff BDF/Radau stream-norm gauntlet

- Agent: cod-a / MistyBirch
- Beads: `frankenscipy-8l8r1.119`, `frankenscipy-8l8r1.120`,
  `frankenscipy-8l8r1.121`
- Decision: KEEP the BDF streamed scaled RMS work, REJECT and REVERT the Radau
  streamed scaled RMS work. Final source scores `3/1/0` against SciPy on the
  stiff suite; remaining loss is `radau-stiff64` and is tracked by
  `frankenscipy-zpunl`.
- Artifact:
  `tests/artifacts/perf/frankenscipy-8l8r1-119-121-integrate-stiff-stream-norms/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust release benchmark | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo run --release -p fsci-integrate --bin perf_integrate -- stiff-suite 10` |
| Same-worker A/B | PASS | Baseline detached worktree at `d502c748` plus harness patch vs candidate on rch `hz2`; BDF64 1.040x faster, BDF128 neutral, Radau32/Radau64 regressed and were reverted |
| SciPy head-to-head oracle | PASS | local SciPy oracle via `docs/perf_oracle_integrate_stiff.py`; final source wins BDF64/BDF128/Radau32 and loses Radau64 |
| Focused BDF tests | PASS | `cargo test -p fsci-integrate bdf --lib -- --nocapture` via rch: 16 passed / 0 failed |
| Focused Radau tests | PASS | `cargo test -p fsci-integrate radau --lib -- --nocapture` via rch: 2 passed / 0 failed |
| IVP conformance | PASS | `cargo test -p fsci-conformance --test e2e_ivp -- --nocapture` via rch: 11 passed / 0 failed |
| Crate compile | PASS | `cargo check -p fsci-integrate --all-targets` via rch |
| Formatting | PASS | touched-file `rustfmt --edition 2024 --check` |
| Diff hygiene | PASS | `git diff --check` |
| Clippy `-D warnings` | BLOCKED | existing `fsci-integrate` lint debt outside touched files; filed `frankenscipy-3qjah` |

| Workload / route | Final Rust | SciPy oracle | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `bdf-stiff64` | 1959.286800 us | 26351.239008 us | 13.45x faster | SciPy win, internal keep |
| `bdf-stiff128` | 11052.293000 us | 29334.902694 us | 2.65x faster | SciPy win, internal neutral/keep |
| `radau-stiff32` | 10191.487800 us | 33444.223704 us | 3.28x faster | SciPy win after revert |
| `radau-stiff64` | 70176.946400 us | 35156.708304 us | 2.00x slower | SciPy loss; route deeper |

Readiness notes:

- The Radau streamed-norm candidate was reverted because same-worker `hz2`
  timings regressed Radau32 by 1.19x and Radau64 by 1.04x with unchanged
  `nfev/njev/nlu` and checksum.
- The remaining integrate performance work is not another scaled-RMS norm
  micro-lever. It should target Radau stage linear algebra, LU reuse, allocation
  in matrix/vector assembly, or structured-Jacobian exploitation.

## 2026-06-20 - fsci-sparse public CSR SpMV row-loop gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-2hclc`
- Decision: KEEP the cached-slice + 4-lane unrolled public CSR row sweep.
  Score for this sub-cluster: 4/5. It closes the public `spmv_csr` scale loss
  on the scoped CSR `.dot(x)` benchmark; future work should profile before
  attempting explicit SIMD or sparse-BLAS-style row blocking.
- Artifact: inline in `docs/progress/perf-negative-results.md` and
  `docs/perf_ledger_cc.md` (no new artifact directory).

| Gate | Result | Notes |
| --- | --- | --- |
| SciPy head-to-head oracle | PASS | `python3 docs/perf_oracle_spmv.py`, local SciPy 1.17.1, n=100/1000/10000 |
| Criterion focused bench | PASS | `cargo bench -p fsci-sparse --bench sparse_bench -- sparse_spmv --sample-size 10 --measurement-time 1 --warm-up-time 1` via rch |
| Same-process old/current A/B | PASS | `env FSCI_PUBLIC_SPMV_AB=1 cargo run --profile release-perf -p fsci-sparse --bin perf_csr_matvec` via rch `ovh-a`; all rows `identical=true` |
| Focused sparse tests | PASS | `cargo test -p fsci-sparse spmv -- --nocapture` via rch: 5 unit/property + 4 metamorphic pass |
| Sparse compile | PASS | `cargo check -p fsci-sparse --all-targets` via rch |
| Sparse clippy | PASS | `cargo clippy -p fsci-sparse --all-targets --no-deps -- -D warnings` via rch |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-sparse/src/ops.rs crates/fsci-sparse/src/bin/perf_csr_matvec.rs` |
| Crate-wide formatting | BLOCKED | `cargo fmt -p fsci-sparse --check` reports existing drift in untouched `sparse_bench.rs`, `src/lib.rs`, and `src/linalg.rs`; not reformatted in this patch |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-sparse/src/ops.rs crates/fsci-sparse/src/bin/perf_csr_matvec.rs .beads/issues.jsonl` |
| Changed-file UBS scan | PASS | `ubs crates/fsci-sparse/src/ops.rs crates/fsci-sparse/src/bin/perf_csr_matvec.rs`: 0 critical; existing warnings only |
| Sparse conformance | PASS | `cargo test -p fsci-conformance --test diff_sparse spmv -- --nocapture` via rch: 11 passed / 0 failed |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current `spmv_csr`, n=100 nnz=500 | 387.54 ns | 11.95x faster than SciPy | SciPy win |
| SciPy CSR `.dot(x)`, n=100 nnz=500 | 4.63 us | 1.00x oracle | reference |
| Rust current `spmv_csr`, n=1000 nnz=10000 | 7.077 us | 1.13x faster than SciPy | SciPy win |
| SciPy CSR `.dot(x)`, n=1000 nnz=10000 | 8.00 us | 1.00x oracle | reference |
| Rust current `spmv_csr`, n=10000 nnz=100000 | 68.820 us | 1.41x faster than SciPy | SciPy win |
| SciPy CSR `.dot(x)`, n=10000 nnz=100000 | 96.95 us | 1.00x oracle | reference |
| Legacy public row sweep A/B, n=100/1000/10000 | 550 ns / 12.074 us / 135.043 us | current is 1.54x / 2.10x / 2.14x faster | internal baseline |

Readiness notes:

- Score vs SciPy is 3 wins / 0 losses / 0 neutral. The prior public SpMV
  ledger row was 1 win / 2 losses, so this closes the measured scale gap.
- The private Krylov/eigensolver `csr_matvec_into` route was already present;
  this patch targets only the public `ops.rs` SpMV API and the perf proof mode.

## 2026-06-20 - fsci-ndimage label mean dense-lookup gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-klb7o`
- Decision: KEEP the compact integer-label dense lookup as an internal win,
  but mark the routine as a SciPy LOSS on the measured rows. Score for this
  sub-cluster: 3.5/5.
- Artifact:
  `tests/artifacts/perf/2026-06-20-label-stats-dense-mean/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo check -p fsci-ndimage --all-targets`; existing `fsci-interpolate` and `diff_geom` warnings remain |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch: 241 passed / 0 failed |
| Same-binary A/B | PASS | `perf_label_stats` compares old O(N*K), bucketed O(N+K), previous flat HashMap O(N+K), and dense O(N+K); mismatches 0/0/0 |
| rch release A/B | PASS | `cargo run --release -p fsci-ndimage --bin perf_label_stats` via rch worker `hz2`: dense is 2.00-2.48x faster than prior flat HashMap |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 via `docs/perf_oracle_label_stats.py`; final source remains 1.9-2.6x slower |
| ndimage conformance | PASS | rch `cargo test -p fsci-conformance ndimage -- --nocapture`: lib-side ndimage filter 5 passed / 0 failed; local live SciPy `diff_ndimage` with `FSCI_REQUIRE_SCIPY_ORACLE=1`: 5 passed / 0 failed |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Changed-file UBS scan | PASS | `ubs` exited 0; no critical issues; broad warning inventory left untouched |
| Clippy `-D warnings` | BLOCKED | dependency clippy stopped on existing `fsci-linalg` lints; no-deps clippy stopped on existing unrelated `fsci-ndimage` lib-file lints outside this patch |
| Full conformance suite | BLOCKED | unrelated existing failures: missing `P2C-007/contracts/contract_table.json`, Array API tolerance fallback mismatch, and missing `legacy_scipy_code/scipy` on rch worker image |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust dense mean, N=65536 K=512 | 301.407 us | 1.91x slower than SciPy; 2.08x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=65536 K=512 | 625.594 us | dense is 2.08x faster | internal baseline |
| SciPy `ndimage.mean`, N=65536 K=512 | 158 us | 1.00x oracle | reference |
| Rust dense mean, N=262144 K=1024 | 1.263 ms | 2.35x slower than SciPy; 2.10x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=262144 K=1024 | 2.652 ms | dense is 2.10x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=1024 | 538 us | 1.00x oracle | reference |
| Rust dense mean, N=262144 K=2048 | 1.383 ms | 2.51x slower than SciPy; 2.02x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=262144 K=2048 | 2.798 ms | dense is 2.02x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=2048 | 550 us | 1.00x oracle | reference |
| Rust dense mean, N=589824 K=4096 | 3.351 ms | 2.64x slower than SciPy; 2.02x faster than prior flat HashMap Rust | SciPy loss, internal keep |
| Rust prior flat HashMap mean, N=589824 K=4096 | 6.772 ms | dense is 2.02x faster | internal baseline |
| SciPy `ndimage.mean`, N=589824 K=4096 | 1.271 ms | 1.00x oracle | reference |

Readiness notes:

- This is a real measured constant-factor win over the prior flat HashMap
  implementation. The final source scores `0/4/0` against SciPy and `4/0/0`
  against the previous Rust flat HashMap route.
- Future label-stat work should route below per-element lookup overhead:
  sorted-label remapping, fused integer-label generation from `label()`, SIMD
  accumulation over contiguous label spans, or cache-tiled sum/count reductions.

## 2026-06-20 - fsci-spatial Delaunay circumcircle-grid gauntlet

- Agent: cod-b / MistyBirch
- Bead: `frankenscipy-9l5oo`
- Decision: KEEP the n>=4096 circumcircle-grid candidate index. Score for this
  sub-cluster: 4/5. It closes the measured 4000/8000 SciPy losses to parity, but
  still needs a beyond-8000 re-test before claiming asymptotic dominance.
- Artifact: inline in `docs/progress/perf-negative-results.md` and
  `docs/perf_ledger_cc.md` (no new artifact directory).

| Gate | Result | Notes |
| --- | --- | --- |
| SciPy head-to-head oracle | PASS | `python3 docs/perf_oracle_delaunay.py`, local SciPy 1.17.1, n=1000/2000/4000/8000 |
| Criterion focused bench | PASS | `cargo bench -p fsci-spatial --bench spatial_bench -- delaunay --sample-size 10 --measurement-time 1 --warm-up-time 1` via rch |
| Focused Delaunay tests | PASS | `cargo test -p fsci-spatial delaunay -- --nocapture`: 8 unit + 1 metamorphic pass |
| Full spatial lib tests | PASS | `cargo test -p fsci-spatial --lib -- --nocapture`: 208 passed / 0 failed / 2 ignored |
| Spatial conformance | PASS | `cargo test -p fsci-conformance --test e2e_spatial -- --nocapture`: 16/0 |
| Check / clippy / fmt | PASS | `cargo check -p fsci-spatial --all-targets`; `cargo clippy -p fsci-spatial --all-targets --no-deps -- -D warnings`; `cargo fmt --check -p fsci-spatial` |
| Changed-file UBS | PASS | `ubs crates/fsci-spatial/src/lib.rs crates/fsci-spatial/benches/spatial_bench.rs docs/perf_oracle_delaunay.py`; no criticals |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust current `Delaunay`, n=1000 | 754.03 us | 2.56x faster than SciPy | SciPy win |
| SciPy `spatial.Delaunay`, n=1000 | 1.93258 ms | 1.00x oracle | reference |
| Rust current `Delaunay`, n=2000 | 2.6129 ms | 1.74x faster than SciPy | SciPy win |
| SciPy `spatial.Delaunay`, n=2000 | 4.54974 ms | 1.00x oracle | reference |
| Rust current `Delaunay`, n=4000 | 9.4632 ms | parity with SciPy | neutral |
| SciPy `spatial.Delaunay`, n=4000 | 9.50086 ms | 1.00x oracle | reference |
| Rust current `Delaunay`, n=8000 | 20.622 ms | parity with SciPy | neutral |
| SciPy `spatial.Delaunay`, n=8000 | 20.62714 ms | 1.00x oracle | reference |

Readiness notes:

- Score vs SciPy is 2 wins / 0 losses / 2 neutral on the scoped deterministic
  2-D workload. The pre-grid expanded probe showed losses at 4000/8000; the
  grid closes those rows to parity.
- This is a candidate-pruning accelerator for Bowyer-Watson, not a full
  randomized incremental/history-DAG point-location rewrite. Re-test at larger
  n before routing remaining work as purely constant-factor.

## 2026-06-20 - fsci-ndimage label mean flat-accumulator gauntlet

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-8l8r1.125`
- Decision: KEEP the flat `mean(labels,index)` accumulator as an internal win,
  but mark the routine as a SciPy LOSS on the measured rows. Score for this
  sub-cluster: 3/5.
- Artifact:
  `tests/artifacts/perf/2026-06-20-label-stats-flat-mean/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate compile | PASS | `RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a rch exec -- cargo check -p fsci-ndimage --all-targets`; existing `fsci-interpolate` and `diff_geom` warnings remain |
| Focused reduction tests | PASS | `cargo test -p fsci-ndimage measurement_reduction_wrappers -- --nocapture` via rch: 2 passed / 0 failed |
| Full ndimage lib tests | PASS | `cargo test -p fsci-ndimage --lib -- --nocapture` via rch: 240 passed / 0 failed |
| Same-binary A/B | PASS | `perf_label_stats` compares old O(N*K), previous bucketed O(N+K), and flat O(N+K) in one optimized binary; mismatches 0/0 |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1 via `docs/perf_oracle_label_stats.py`; final source remains 3.7-4.7x slower |
| Touched-file formatting | PASS | `rustfmt --edition 2024 --check crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Diff hygiene | PASS | `git diff --check -- crates/fsci-ndimage/src/lib.rs crates/fsci-ndimage/src/bin/perf_label_stats.rs` |
| Changed-file UBS scan | PASS | `ubs` exited 0; no critical issues; broad warning inventory left untouched |
| Clippy `-D warnings` | BLOCKED | `cargo clippy -p fsci-ndimage --lib -- -D warnings` stopped before this crate on existing `fsci-linalg` dependency lints outside this patch |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Rust flat mean, N=65536 K=512 | 590.978 us | 3.72x slower than SciPy; 1.77x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=65536 K=512 | 1.047 ms | flat is 1.77x faster | internal baseline |
| SciPy `ndimage.mean`, N=65536 K=512 | 159 us | 1.00x oracle | reference |
| Rust flat mean, N=262144 K=1024 | 2.568 ms | 4.13x slower than SciPy; 1.44x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=262144 K=1024 | 3.695 ms | flat is 1.44x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=1024 | 622 us | 1.00x oracle | reference |
| Rust flat mean, N=262144 K=2048 | 2.713 ms | 4.67x slower than SciPy; 1.53x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=262144 K=2048 | 4.140 ms | flat is 1.53x faster | internal baseline |
| SciPy `ndimage.mean`, N=262144 K=2048 | 581 us | 1.00x oracle | reference |
| Rust flat mean, N=589824 K=4096 | 6.951 ms | 4.12x slower than SciPy; 1.69x faster than prior bucketed Rust | SciPy loss, internal keep |
| Rust prior bucketed mean, N=589824 K=4096 | 11.760 ms | flat is 1.69x faster | internal baseline |
| SciPy `ndimage.mean`, N=589824 K=4096 | 1.688 ms | 1.00x oracle | reference |

Readiness notes:

- This is a real constant-factor win over the prior O(N+K) implementation, not
  a SciPy parity claim. The final source scores `0/4/0` against SciPy and
  `4/0/0` against the previous Rust bucketed route.
- Future label-stat work should target the remaining per-element lookup and
  accumulation constants: dense label lookup for compact integer labels,
  sorted-label remapping, or SIMD/cache-tiled accumulation. Reintroducing
  grouped `Vec<Vec<f64>>` materialization for `mean` is negative evidence.

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

## 2026-06-19 - fsci-special jnjnp_zeros top-k select reject

- Agent: cod-a / MistyBirch
- Bead: `frankenscipy-8l8r1.124`
- Decision: REJECT and revert the top-k candidate partition. Score for this
  sub-cluster: 2/5.
- Artifact:
  `tests/artifacts/perf/2026-06-19-8l8r1-jnjnp-topk-select/EVIDENCE.md`

| Gate | Result | Notes |
| --- | --- | --- |
| Rust per-crate Criterion bench | PASS | baseline current route on RCH worker `vmi1153651`; candidate Criterion run on `hz2` treated as directional only because RCH worker affinity was not stable |
| Same-binary A/B probe | PASS | ignored release probe asserted bit-identical output, then timed full-sort vs top-k in one optimized binary |
| SciPy head-to-head oracle | PASS | local SciPy 1.17.1, `scipy.special.jnjnp_zeros`; RCH workers lacked `scipy.special` |
| Focused special tests | PASS | `cargo test -p fsci-special jnjnp -- --nocapture` passed via RCH during the probe |
| Source revert | PASS | `crates/fsci-special/src/bessel.rs` restored to the cutoff-driven full-sort route |

| Workload / route | Mean | Ratio | Verdict |
| --- | ---: | ---: | --- |
| Restored Rust full-sort `jnjnp_zeros(nt=64)` | 1.5407 ms | 3.65x slower than SciPy | SciPy loss, final source |
| Restored Rust full-sort `jnjnp_zeros(nt=128)` | 3.5199 ms | 4.54x slower than SciPy | SciPy loss, final source |
| SciPy `jnjnp_zeros(nt=64)` | 421.59 us | 1.00x oracle | reference |
| SciPy `jnjnp_zeros(nt=128)` | 774.75 us | 1.00x oracle | reference |
| Top-k candidate, same-binary `nt=64` | 0.911730 ms | 1.019x vs full-sort | reject as near-noise |
| Top-k candidate, same-binary `nt=128` | 1.715855 ms | 1.013x vs full-sort | reject as near-noise |

Readiness notes:

- The final source is unchanged by this rejected probe.
- Future `jnjnp_zeros` work should target root-generation and Bessel
  evaluation constants, not candidate prefix sorting or partial sort variants.

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
