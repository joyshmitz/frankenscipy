# frankenscipy-8l8r1.133 - linkage compact active frontier

- Agent: cod-a / BlackThrush
- Date: 2026-06-20
- Crate: `fsci-cluster`
- Lever: replace the boolean-active range scans in the NN-array linkage core
  with a sorted compact `active_ids` frontier. The frontier preserves ascending
  cluster-id tie order but removes inactive-cluster branch checks from nearest
  selection, Lance-Williams distance updates, and nearest-neighbour refreshes.
- Decision: KEEP. The local SciPy head-to-head gauntlet shows a 1.87x internal
  median speedup on Average and a 2.00x internal median speedup on Ward, moving
  both tracked rows from SciPy losses to near-parity/slight median wins.

## Commands

Baseline:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local \
cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot \
  2>&1 | tee tests/artifacts/perf/2026-06-20-cod-a-linkage-133/baseline_linkage_local_scipy.txt
```

Candidate:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local \
cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot \
  2>&1 | tee tests/artifacts/perf/2026-06-20-cod-a-linkage-133/candidate_linkage_local_scipy.txt
```

Correctness and build gates:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo test -p fsci-cluster linkage_flat_core_matches_precomputed_condensed_contract -- --nocapture
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo test -p fsci-cluster linkage -- --nocapture
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local \
  cargo test -p fsci-conformance --test diff_cluster_linkage_from_distances -- --nocapture
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local \
  cargo test -p fsci-conformance --test diff_cluster_linkage_helpers -- --nocapture
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo build --release -p fsci-cluster
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo check -p fsci-cluster --all-targets
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
  rch exec -- cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings
```

## Measured Results

Criterion medians are used for the ratios. The SciPy oracle rows run in the
same local Criterion process as the Rust rows.

| Workload | Baseline Rust | Candidate Rust | SciPy oracle | Candidate vs baseline | Candidate vs SciPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linkage(Average)`, n=800 d=4 | 8.5503 ms | 4.5727 ms | 4.8204 ms | 1.87x faster | 1.05x faster |
| `linkage(Ward)`, n=800 d=4 | 10.831 ms | 5.4267 ms | 5.6168 ms | 2.00x faster | 1.04x faster |

Prior same-run baseline ratios:

| Workload | Baseline Rust | SciPy oracle | Baseline verdict |
| --- | ---: | ---: | --- |
| `linkage(Average)`, n=800 d=4 | 8.5503 ms | 5.1922 ms | Rust 1.65x slower |
| `linkage(Ward)`, n=800 d=4 | 10.831 ms | 5.2467 ms | Rust 2.06x slower |

Win/loss/neutral:

- Same-machine candidate versus current: `2/0/0`.
- Candidate median versus same-machine SciPy oracle: `2/0/0`, with overlapping
  intervals, so record as near-parity/slight wins rather than a large SciPy
  domination claim.

## Gates

- PASS: rch focused bit-contract test:
  `linkage_flat_core_matches_precomputed_condensed_contract` = 1 passed.
- PASS: rch broader linkage filter: 28 linkage unit tests and 9 metamorphic
  linkage tests passed.
- PASS: live SciPy differential conformance:
  `diff_cluster_linkage_from_distances` = 1 passed.
- PASS: live SciPy differential conformance:
  `diff_cluster_linkage_helpers` = 1 passed.
- PASS: rch per-crate release build:
  `cargo build --release -p fsci-cluster`; existing warning remained in
  `src/bin/perf_kmeans.rs`.
- PASS: rch per-crate all-targets check:
  `cargo check -p fsci-cluster --all-targets`; existing `perf_kmeans.rs`
  warning remained.
- PASS: rch no-deps clippy:
  `cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings`.
- PASS: final local lib check:
  `cargo check -p fsci-cluster --lib`.
- PASS: `git diff --check` on touched files.
- PASS: changed-file UBS scan had 0 critical findings; existing broad warning
  inventory remained in `crates/fsci-cluster/src/lib.rs`.
- BLOCKED: `cargo fmt -p fsci-cluster --check` / `rustfmt --edition 2024
  --check crates/fsci-cluster/src/lib.rs` remain blocked by pre-existing
  crate/file-wide rustfmt drift outside this scoped patch.

## Negative Evidence

- Do not retry full-square arena initialization changes for this path; the
  earlier lazy-arena lever regressed Average and was reverted.
- The active frontier is the profitable near-neighbour maintenance layer: it
  keeps the exact NN-array merge sequence while reducing scanned inactive
  state. Further linkage work should move to a true method-specific NN-chain
  primitive or a smaller-distance frontier, not another storage-fill tweak.
