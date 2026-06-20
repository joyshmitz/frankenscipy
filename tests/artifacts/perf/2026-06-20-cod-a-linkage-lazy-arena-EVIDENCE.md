# 2026-06-20 cod-a linkage row-pack keep + lazy-arena reject

Agent: cod-a / BlackThrush

Bead: `frankenscipy-8l8r1.128`

Parent directive: `frankenscipy-8l8r1`

## Scope

Targeted `fsci-cluster::linkage` after prior linkage work showed a persistent
SciPy loss on `scipy.cluster.hierarchy.linkage(method="average"|"ward")` for
n=800, d=4. The source change kept here only packs nested observation rows once
before pairwise distance construction. The Lance-Williams updates, tie order,
full row-major arena, validation, and public behavior are unchanged.

## Commands

rch baseline attempt:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
rch exec -- cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot
```

Result: Rust rows ran on rch `hz1`, but the SciPy oracle failed on the worker
with `ModuleNotFoundError: No module named 'scipy'`.

Local SciPy head-to-head:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local \
cargo bench -p fsci-cluster --bench cluster_bench -- va60h_gauntlet_linkage --noplot
```

The exact requested rch target dir was used for rch commands. Local SciPy
Criterion rows used `frankenscipy-cod-a-local` because the rch-built artifacts
in the exact target were from an incompatible rustc; no target files were
deleted or cleaned.

Isomorphism harness:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a-local \
cargo run --release -p fsci-cluster --bin perf_linkage
```

Gates:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
rch exec -- cargo build --release -p fsci-cluster

AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
rch exec -- cargo test -p fsci-cluster linkage_flat_core_matches_precomputed_condensed_contract -- --nocapture

AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
rch exec -- cargo test -p fsci-cluster linkage_ -- --nocapture

AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-a \
rch exec -- cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings

ubs crates/fsci-cluster/src/lib.rs docs/NEGATIVE_EVIDENCE.md \
  docs/GAUNTLET_RELEASE_SCORECARD.md docs/progress/perf-negative-results.md \
  docs/progress/perf-release-readiness-scorecard.md \
  tests/artifacts/perf/2026-06-20-cod-a-linkage-lazy-arena-EVIDENCE.md \
  .beads/issues.jsonl

rustfmt --edition 2024 --check crates/fsci-cluster/src/lib.rs
```

## Baseline

Local SciPy version: `1.17.1`.

| Workload | Current Rust mean | SciPy mean | Ratio |
| --- | ---: | ---: | ---: |
| `linkage(Average)`, n=800 d=4 | 7.1834 ms | 5.0346 ms | Rust 1.427x slower |
| `linkage(Ward)`, n=800 d=4 | 8.2387 ms | 5.3080 ms | Rust 1.552x slower |

Baseline SciPy score: `0/2/0`.

## Rejected Candidate: Lazy Full-Arena Zeroing

Candidate: initialize the full inter-cluster arena with zeroes instead of
`f64::INFINITY`, relying on later active-pair writes. This tested whether
removing a large memory fill paid more than preserving the explicit sentinel.

| Workload | Current Rust mean | Candidate mean | SciPy mean | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linkage(Average)`, n=800 d=4 | 7.1834 ms | 7.6203 ms | 4.5097 ms | reject: 1.061x slower than current; 1.690x slower than SciPy |
| `linkage(Ward)`, n=800 d=4 | 8.2387 ms | 8.2002 ms | 5.2550 ms | neutral: 1.005x faster than current; still 1.560x slower than SciPy |

Decision: rejected and reverted. Score versus current: `0/1/1`.

Negative evidence: do not retry zero/lazy initialization of the full arena on
this NN-array linkage route without a new profile showing the full fill is the
dominant cost.

## Kept Candidate: Packed Observation Rows

Candidate: flatten `Vec<Vec<f64>>` observations once after dimension validation
and use contiguous row slices for all pairwise source distances.

| Workload | Baseline Rust mean | Final Rust mean | Internal delta | Same-run SciPy mean | Final vs SciPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linkage(Average)`, n=800 d=4 | 7.1834 ms | 7.1304 ms | 1.007x faster | 4.3843 ms | Rust 1.626x slower |
| `linkage(Ward)`, n=800 d=4 | 8.2387 ms | 6.9591 ms | 1.184x faster | 4.8687 ms | Rust 1.429x slower |

Decision: keep. Internal score versus current baseline: `1/0/1`. Strict SciPy
score remains `0/2/0`.

## Correctness

`perf_linkage`:

```text
isomorphism: 0 mismatches / 7200 linkage matrices (0 == byte-identical)
```

Focused rch test:

```text
linkage_flat_core_matches_precomputed_condensed_contract ... ok
```

Broader rch linkage filter completed with exit code 0.

## Gates

- PASS: `cargo build --release -p fsci-cluster` via rch. One unrelated warning
  remained in `crates/fsci-cluster/src/bin/perf_kmeans.rs`.
- PASS: `cargo test -p fsci-cluster linkage_flat_core_matches_precomputed_condensed_contract -- --nocapture` via rch.
- PASS: `cargo test -p fsci-cluster linkage_ -- --nocapture` via rch.
- PASS: `cargo clippy -p fsci-cluster --lib --no-deps -- -D warnings` via rch.
- PASS: changed-file UBS exited 0 with 0 critical findings; existing warning
  inventory remains in `fsci-cluster/src/lib.rs`.
- BLOCKED: rch SciPy benchmark rows, because the selected worker could not
  import `scipy`.
- BLOCKED: dependency-inclusive clippy stops on existing `fsci-linalg` lints.
- BLOCKED: `rustfmt --edition 2024 --check crates/fsci-cluster/src/lib.rs`
  reports pre-existing file-wide drift outside this scoped patch.

## Next Route

The row-pack lever only reduces distance-frontier constants. Closing the SciPy
gap likely needs a different clustering primitive: NN-chain or
method-specialized nearest-neighbour maintenance, MST/single-linkage style
specialization where applicable, or a lower-constant compiled-distance frontier.
Do not spend another pass on full-arena initialization tricks.
