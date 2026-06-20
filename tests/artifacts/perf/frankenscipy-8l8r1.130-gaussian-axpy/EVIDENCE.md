# frankenscipy-8l8r1.130 - folded AXPY gaussian 2D reflect pass

Date: 2026-06-20

Agent: cod-b / MistyBirch

Decision: KEEP as a measured internal win; residual SciPy loss remains.

## Lever

For the 2-D `BoundaryMode::Reflect`, order-0 `gaussian_filter` fast path, the
axis-0 pass previously computed each scratch pixel by gathering one element from
each reflected source row at stride `cols`. This lever uses the symmetry of the
Gaussian kernel to initialize the full row from the center tap, then folds each
tap pair as contiguous row AXPY:

`scratch_row += w * (input[reflect(row + k)] + input[reflect(row - k)])`

The horizontal pass folds symmetric column taps too. The generic path remains
reachable through the runtime A/B toggle and all non-2D/non-Reflect/order
variants keep the existing implementation.

## Graveyard / Artifact Route

- Data-layout and vectorized-execution route: convert the hot strided gather
  into stride-1 contiguous row loops so the compiler and hardware prefetcher can
  see the stream.
- Polyhedral/stencil route: exploit separability plus symmetric tap pairs while
  preserving the SciPy reflect source plan.
- Rejected alternatives not retried: scalar border/interior tap peeling and
  always-line-walk outer-axis splits from earlier negative evidence.

## Commands

Clean baseline worktree:

```text
git worktree add --detach /data/projects/.scratch/frankenscipy-cod-b-gaussian-axpy-baseline-20260620T1044 origin/main
```

Same-worker clean-current benchmark:

```text
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- \
  correlate_gaussian/gaussian_sigma2/256 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

RCH selected worker `vmi1167313`.

Same-worker candidate benchmark:

```text
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1167313 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- \
  correlate_gaussian/gaussian_sigma2/256 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Same-process A/B:

```text
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1167313 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo test -p fsci-ndimage gaussian_2d_axpy_ab_timing --release -- --ignored --nocapture
```

Final-source routing sanity row:

```text
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1167313 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo bench -p fsci-ndimage --bench ndimage_bench -- \
  correlate_gaussian/gaussian_sigma2/256 --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

RCH routed this final-source sanity row to `vmi1149989`, so it is not used as
the keep/reject comparison.

SciPy oracle:

```text
python3 docs/perf_oracle_ndimage.py
```

## Benchmark Evidence

| Route | Worker | Mean | Interval / value | Ratio |
| --- | --- | ---: | ---: | --- |
| Clean current Rust `0cf3cc42` | `vmi1167313` | 6.9394 ms | [5.1048, 9.1535] ms | baseline; noisy severe outlier |
| Candidate AXPY Rust | `vmi1167313` | 3.3918 ms | [2.9580, 3.7948] ms | 2.05x faster than current |
| Same-process gather toggle | `vmi1167313` | 3585.0 us | single binary interleaved A/B | baseline arm |
| Same-process AXPY toggle | `vmi1167313` | 2943.3 us | single binary interleaved A/B | 1.22x faster than gather |
| Final-source routing sanity | `vmi1149989` | 3.0285 ms | [2.8418, 3.3639] ms | routing-only, not paired |
| SciPy `ndimage.gaussian_filter` | local SciPy oracle | 1.16724 ms | median | oracle |

The strict SciPy result remains a loss:

- Candidate same-worker row vs fresh SciPy oracle: 2.91x slower.
- Final-source routing row vs fresh SciPy oracle: 2.59x slower.

## Win / Loss / Neutral

- Same-worker candidate versus clean current: `1/0/0`.
- Same-process A/B candidate versus gather path: `1/0/0`.
- Strict SciPy score for final source: `0/1/0`.

## Correctness / Conformance

PASS: focused equivalence guard on the exact final local source:

```text
AGENT_NAME=MistyBirch CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  cargo test -p fsci-ndimage gaussian_2d_axpy_matches_gather_dot --lib -- --nocapture
```

Result: `1 passed; 0 failed`.

PASS: broader focused Gaussian tests via RCH:

```text
AGENT_NAME=MistyBirch RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1167313 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  rch exec -- cargo test -p fsci-ndimage gaussian --lib -- --nocapture
```

Result: `31 passed; 0 failed; 1 ignored`.

PASS: live SciPy conformance on the exact final local source:

```text
AGENT_NAME=MistyBirch FSCI_REQUIRE_SCIPY_ORACLE=1 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b \
  cargo test -p fsci-conformance --test diff_ndimage_gaussian_filter -- --nocapture
```

Result: `1 passed; 0 failed`.

Known unrelated warnings appeared in `fsci-interpolate`, `fsci-special`, and
`crates/fsci-ndimage/src/bin/diff_geom.rs`.

PASS: crate-scoped build check on the exact final local source:

```text
AGENT_NAME=MistyBirch CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenscipy-cod-b-check \
  cargo check -p fsci-ndimage --all-targets
```

Result: build passed; inherited warnings remained in `fsci-interpolate` and
`crates/fsci-ndimage/src/bin/diff_geom.rs`.

PASS: `git diff --check`.

PASS: changed-file UBS scan:

```text
ubs crates/fsci-ndimage/src/lib.rs docs/NEGATIVE_EVIDENCE.md \
  docs/progress/perf-negative-results.md docs/GAUNTLET_RELEASE_SCORECARD.md \
  tests/artifacts/perf/frankenscipy-8l8r1.130-gaussian-axpy/EVIDENCE.md \
  .beads/issues.jsonl
```

Result: exit 0, 0 criticals; the scan reported the existing broad warning
inventory in `crates/fsci-ndimage/src/lib.rs`.

BLOCKED: `cargo fmt --check -p fsci-ndimage` remains blocked by pre-existing
format drift in `crates/fsci-ndimage/benches/ndimage_bench.rs`,
`crates/fsci-ndimage/src/bin/diff_fourier.rs`, and existing unrelated
`src/lib.rs` sections. This patch did not apply a broad formatting churn.

BLOCKED: `cargo clippy -p fsci-ndimage --all-targets -- -D warnings` stops
before this patch on existing `fsci-linalg` dependency lints at
`crates/fsci-linalg/src/lib.rs:4291`, `:4292`, `:16582`, `:16605`, `:16630`,
and `:16635`.

## Negative Evidence / Next Route

This AXPY fold is worth keeping because it turns the first-axis gather into a
stride-1 stream and wins in both a paired Criterion row and an in-process A/B.
It still does not close the SciPy gap. Do not retry scalar reflect tap peeling
or outer-axis always-line-walk variants without a new profile.

Next credible route: remove the remaining horizontal gather by transposing the
scratch or using cache-blocked tiles so both separable passes run as row AXPY,
then add a shape-specialized no-atomic production variant once the test toggle
is no longer needed.
